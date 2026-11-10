"""Stack independently rotated, color-selected hole stencils on one solid plate.

Only standard library and NumPy are used. Generation works with array masks;
validation reconstructs registered objects and rotates occupied coordinates.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, num_train, num_test):
    # None is also supported for the supplied inspection helper.
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (not _integer(value) or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train)-1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, int(num_test)+1), 11))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger grid"
    assert np.all((a >= 0) & (a <= 9)), "invalid color"
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
        result.append(np.asarray(cells, dtype=int))
    return result


def _rim(n):
    result = np.ones((n, n), dtype=bool)
    result[1:-1, 1:-1] = False
    return result


def _decode(input_grid):
    """Recognize all roles solely from visible geometry and color equality."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])), "background border"
    candidates = []
    component_cache = {}
    for material in (int(x) for x in np.unique(a) if x != bg):
        comps = _components(a == material)
        component_cache[material] = comps
        for cells in comps:
            r, c = cells.min(axis=0)
            e, f = cells.max(axis=0)+1
            n = int(e-r)
            if n != f-c or n < 5:
                continue
            block = a[r:e, c:f]
            rim = _rim(n)
            tabs = block[rim & (block != material)]
            if (np.all(block[1:-1, 1:-1] == material)
                    and np.all(block[rim] != bg)
                    and len(tabs) >= 1 and len(set(map(int, tabs))) == len(tabs)
                    and all(block[y, x] == material for y, x in ((0, 0), (0, n-1), (n-1, 0), (n-1, n-1)))):
                candidates.append((material, int(r), int(c), n, tuple(sorted(map(int, tabs)))))
    assert len(candidates) == 1, "unique tabbed solid plate required"
    material, tr, tc, n, selected = candidates[0]
    k = n-2
    boxes, patterns = [(tr, tc, n)], {}
    accounted = np.zeros(a.shape, dtype=bool)
    accounted[tr:tr+n, tc:tc+n] = True
    for cells in component_cache[material]:
        r, c = map(int, cells.min(axis=0))
        e, f = map(int, cells.max(axis=0)+1)
        if (r, c) == (tr, tc):
            continue
        assert (e-r, f-c) == (n, n), "all registers share the plate size"
        block = a[r:e, c:f]
        assert np.all(block[_rim(n)] == material), "incomplete register frame"
        inside = block[1:-1, 1:-1]
        colors = [int(x) for x in np.unique(inside) if x != bg]
        assert len(colors) == 1 and colors[0] != material, "monochrome stencil required"
        color = colors[0]
        assert color not in patterns, "duplicate same-colored register"
        patterns[color] = (inside == color).copy()
        boxes.append((r, c, n))
        accounted[r:e, c:f] = True
    assert patterns and set(selected) <= set(patterns), "tab lacks matching stencil"
    assert bg != material and material not in selected
    # Complete rectangles, including empty register cells, have clear halos.
    for i, (r, c, size) in enumerate(boxes):
        assert r >= 1 and c >= 1 and r+size < a.shape[0] and c+size < a.shape[1]
        halo = a[r-1:r+size+1, c-1:c+size+1].copy()
        halo[1:-1, 1:-1] = bg
        assert np.all(halo == bg), "objects need a background gap"
        for rr, cc, ss in boxes[i+1:]:
            assert r+size < rr or rr+ss < r or c+size < cc or cc+ss < c, "register overlap"
    external = (a != bg) & ~accounted
    counts, orientations = {}, {}
    for color in (int(x) for x in np.unique(a[external])):
        assert color in patterns, "unmatched counter color"
        coords = np.argwhere(external & (a == color))
        rr, cc = coords.T
        horizontal = np.ptp(rr) == 0 and np.ptp(cc)+1 == len(rr)
        vertical = np.ptp(cc) == 0 and np.ptp(rr)+1 == len(rr)
        assert horizontal or vertical, "one straight counter per color"
        counts[color] = len(rr)
        orientations[color] = "single" if len(rr) == 1 else ("h" if horizontal else "v")
        for y, x in coords:
            window = a[max(0, y-1):y+2, max(0, x-1):x+2]
            assert np.all((window == bg) | (window == color)), "counters touch"
    return {"background": bg, "material": material, "size": n, "k": k,
            "plate": a[tr:tr+n, tc:tc+n].copy(), "origin": (tr, tc),
            "selected": selected, "patterns": patterns, "counts": counts,
            "orientations": orientations, "boxes": boxes}


def _rotated_coordinates(mask, turns):
    result = np.zeros(mask.shape, dtype=bool)
    k = mask.shape[0]
    for row, col in np.argwhere(mask):
        r, c = int(row), int(col)
        for _ in range(turns % 4):
            r, c = c, k-1-r
        result[r, c] = True
    return result


def _holes(info, hypothesis="correct"):
    colors = tuple(sorted(info["patterns"])) if hypothesis == "all_colors" else info["selected"]
    if hypothesis == "ignore_unmarked":
        colors = tuple(c for c in colors if c in info["counts"])
    if not colors:
        return np.ones(next(iter(info["patterns"].values())).shape, dtype=bool)
    masks = []
    for color in colors:
        length = info["counts"].get(color, 0)
        turns = length
        if hypothesis == "no_rotation":
            turns = 0
        elif hypothesis == "counterclockwise":
            turns = -length
        elif hypothesis == "absent_one" and color not in info["counts"]:
            turns = 1
        elif hypothesis == "global_rotation":
            turns = info["counts"].get(info["selected"][0], 0)
        elif hypothesis == "orientation_direction" and info["orientations"].get(color) == "v":
            turns = -length
        elif hypothesis == "cap_three":
            turns = min(length, 3)
        elif hypothesis == "four_is_one" and length == 4:
            turns = 1
        masks.append(_rotated_coordinates(info["patterns"][color], turns))
    stack = np.stack(masks)
    if hypothesis == "union":
        return np.any(stack, axis=0)
    if hypothesis == "xor":
        return np.sum(stack, axis=0) % 2 == 1
    if hypothesis == "majority":
        return np.sum(stack, axis=0) > len(masks)/2
    if hypothesis.startswith("only_"):
        return masks[int(hypothesis[5:]) % len(masks)]
    holes = np.all(stack, axis=0)
    return ~holes if hypothesis == "positive_copy" else holes


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    result = info["plate"].copy()
    result[1:-1, 1:-1][_holes(info)] = info["background"]
    return result


def _motif(rng, k):
    """Small geometric grammars, including symmetric and disconnected ink."""
    y, x = np.indices((k, k))
    mode = int(rng.integers(9))
    if mode == 0:
        m = x == y
    elif mode == 1:
        m = (x == y) | (x+y == k-1)
    elif mode == 2:
        m = (y == int(rng.integers(k))) | (x == int(rng.integers(k)))
    elif mode == 3:
        m = (x+int(rng.integers(2))) % 2 == 0
    elif mode == 4:
        m = (y == 0) | (x == 0) | (x == k-1)
    elif mode == 5:
        m = np.zeros((k, k), dtype=bool)
        r, c = int(rng.integers(k)), int(rng.integers(k))
        m[r:int(rng.integers(r+1, k+1)), c:int(rng.integers(c+1, k+1))] = True
    elif mode == 6:
        m = np.ones((k, k), dtype=bool)
        m.flat[int(rng.integers(k*k))] = False
    else:
        m = rng.random((k, k)) < float(rng.uniform(.12, .85))
        if mode == 8:
            m |= np.fliplr(m)
    if not m.any():
        m.flat[int(rng.integers(k*k))] = True
    return np.rot90(m, int(rng.integers(4))).copy()


def _sample_patterns(rng, k, colors, selected, counts):
    # A common potential aperture makes coincidences frequent; surrounding
    # independently sampled ink can enlarge it. No fixed output motif is used.
    common = _motif(rng, k)
    if common.all() or common.sum() > .6*k*k:
        common = rng.random((k, k)) < float(rng.uniform(.1, .35))
        if not common.any():
            common.flat[int(rng.integers(k*k))] = True
    patterns = {}
    for color in colors:
        if color in selected:
            aligned = common | _motif(rng, k)
            # Sometimes a participating shutter is symmetric or identical to
            # another; informative checks apply to the combination, not each.
            if rng.random() < .12:
                aligned = common.copy()
            patterns[color] = np.rot90(aligned, counts.get(color, 0) % 4).copy()
        else:
            patterns[color] = _motif(rng, k)
    return patterns


def _block(occupied, r, c, h, w):
    occupied[max(0, r-1):min(len(occupied), r+h+1),
             max(0, c-1):min(occupied.shape[1], c+w+1)] = True


def _layout(rng, frames, colors, counts, forced, bg, material):
    n, total = len(frames[0]), len(frames)
    for attempt in range(24):
        if attempt < 12:
            h, w = int(rng.integers(24, 31)), int(rng.integers(24, 31))
            slots, occupied = [], np.zeros((h, w), dtype=bool)
            candidates = [(r, c) for r in range(1, h-n) for c in range(1, w-n)]
            for j in rng.permutation(len(candidates)):
                r, c = candidates[int(j)]
                if not occupied[r:r+n, c:c+n].any():
                    slots.append((r, c))
                    _block(occupied, r, c, n, n)
                    if len(slots) == total:
                        break
            if len(slots) < total:
                continue
        else:
            # Guaranteed room for six size-eight frames, horizontal counters
            # underneath and a forced vertical counter at the right.
            h = w = 30
            occupied = np.zeros((h, w), dtype=bool)
            nr = 2 if total > 3 else 1
            nc = (total+nr-1)//nr
            slots = [(1+i*(n+1), 1+j*(n+1)) for i in range(nr) for j in range(nc)]
            slots = [slots[int(j)] for j in rng.permutation(len(slots))[:total]]
            for r, c in slots:
                _block(occupied, r, c, n, n)
        a = np.full((h, w), bg, dtype=np.int8)
        for frame, (r, c) in zip(frames, slots):
            a[r:r+n, c:c+n] = frame
        success = True
        for color in sorted(counts, key=lambda c: -counts[c]):
            length = counts[color]
            options = []
            orientations = (forced[color],) if color in forced else ("h", "v")
            for orientation in orientations:
                bh, bw = (1, length) if orientation == "h" else (length, 1)
                starts = [(r, c) for r in range(1, h-bh) for c in range(1, w-bw)
                          if not occupied[r:r+bh, c:c+bw].any()]
                if starts:
                    options.append((bh, bw, starts))
            if not options:
                success = False
                break
            bh, bw, starts = options[int(rng.integers(len(options)))]
            r, c = starts[int(rng.integers(len(starts)))]
            a[r:r+bh, c:c+bw] = color
            _block(occupied, r, c, bh, bw)
        if success:
            yy, xx = np.nonzero(a != bg)
            a = a[yy.min()-1:yy.max()+2, xx.min()-1:xx.max()+2]
            pad = []
            for size in a.shape:
                extra = int(rng.integers(min(4, 30-size)+1))
                before = int(rng.integers(extra+1))
                pad.append((before, extra-before))
            return np.pad(a, pad, constant_values=bg)
    raise RuntimeError("bounded layout attempts exhausted")


_ALTERNATIVES = ("no_rotation", "counterclockwise", "all_colors", "union",
                 "xor", "majority", "global_rotation", "absent_one",
                 "positive_copy", "orientation_direction", "only_0", "only_1", "only_2",
                 "cap_three", "four_is_one", "ignore_unmarked")


def _make_pair(rng, k, selected_count, total, witness, test, previous_bg):
    bg = int(rng.choice([c for c in range(10) if c != previous_bg]))
    palette = [int(c) for c in rng.choice([c for c in range(10) if c != bg], total+1, replace=False)]
    material, colors = palette[0], palette[1:]
    selected = colors[:selected_count]
    counts = {c: int(rng.integers(1, 10 if test else 5)) for c in colors if rng.random() < .78}
    forced = {}
    if witness == 0:
        counts.update({colors[0]: 1, colors[1]: 2, colors[2]: 2})
        forced[colors[1]], forced[colors[2]] = "h", "v"
    elif witness == 1:
        counts.update({colors[0]: 3, colors[1]: 4})
        counts.pop(colors[2], None)
        forced[colors[0]] = "v"
    if test:
        counts[selected[int(rng.integers(selected_count))]] = int(rng.integers(5, 10))
        if not any(counts.get(c, 0) % 4 for c in selected):
            counts[selected[0]] = 5
    for attempt in range(200):
        patterns = _sample_patterns(rng, k, colors, selected, counts)
        info = {"patterns": patterns, "selected": tuple(sorted(selected)), "counts": counts,
                "orientations": {c: forced.get(c, "h") for c in counts}}
        holes = _holes(info)
        if not (0 < holes.sum() < k*k):
            continue
        core = ("union",) + (("no_rotation",) if witness >= 0 or test else ()) + (("all_colors",) if selected_count < total else ())
        if any(np.array_equal(holes, _holes(info, h)) for h in core):
            continue
        if witness == 0 and any(np.array_equal(holes, _holes(info, h)) for h in
                                ("counterclockwise", "global_rotation", "only_0", "only_1")):
            continue
        if witness == 1 and any(np.array_equal(holes, _holes(info, h)) for h in
                                ("absent_one", "majority", "xor", "only_2", "orientation_direction",
                                 "cap_three", "four_is_one", "ignore_unmarked")):
            continue
        n = k+2
        plate = np.full((n, n), material, dtype=np.int8)
        # Protect corners and their neighbors, keeping the material of the
        # target connected even when several tabs occupy one side.
        rim_positions = [(r, c) for r in range(n) for c in range(n)
                         if (r in (0, n-1) and 2 <= c <= n-3)
                         or (c in (0, n-1) and 2 <= r <= n-3)]
        for color, j in zip(selected, rng.choice(len(rim_positions), selected_count, replace=False)):
            plate[rim_positions[int(j)]] = color
        frames = [plate]
        for color in colors:
            frame = np.full((n, n), material, dtype=np.int8)
            frame[1:-1, 1:-1] = np.where(patterns[color], color, bg)
            frames.append(frame)
        inp = _layout(rng, frames, colors, counts, forced, bg, material)
        out = plate.copy()
        # Generation uses np.rot90; the solver uses coordinate arithmetic.
        aperture = np.logical_and.reduce([np.rot90(patterns[c], -counts.get(c, 0) % 4) for c in selected])
        out[1:-1, 1:-1][aperture] = bg
        return {"input": inp, "output": out}
    raise RuntimeError("bounded stencil sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    nt, ne = _counts(rng, num_train, num_test)
    scale_shift = bool(rng.integers(2))
    puzzle, seen, previous_bg = {"train": [], "test": []}, set(), None
    for i in range(nt+ne):
        test = i >= nt
        witness = i if i < 2 else -1
        k = int(rng.integers(5, 7)) if test and scale_shift else int(rng.integers(3, 5 if scale_shift or witness >= 0 else 7))
        if witness == 0:
            selected_count, total = 2, 3
        elif witness == 1:
            selected_count, total = 3, int(rng.integers(4, 6))
        else:
            selected_count = int(rng.integers(2, 5)) if test and scale_shift else (4 if test else int(rng.integers(2, 4)))
            total = int(rng.integers(selected_count, 6))
        for _ in range(16):
            pair = _make_pair(rng, k, selected_count, total, witness, test, previous_bg)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["test" if test else "train"].append(pair)
                previous_bg = int(pair["input"][0, 0])
                break
        else:
            raise RuntimeError("distinct input attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, infos = set(), {"train": [], "test": []}
    defeated = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            assert 3 <= info["k"] <= 6 and 2 <= len(info["selected"]) <= 4
            assert len(info["selected"]) <= len(info["patterns"]) <= 5
            assert all(1 <= length <= 9 for length in info["counts"].values())
            infos[split].append(info)
            holes = _holes(info)
            expected = info["plate"].copy()
            expected[1:-1, 1:-1][holes] = info["background"]
            assert np.array_equal(out, expected), "incorrect intersection, rotation, or crop"
            assert np.array_equal(out[_rim(info["size"])], info["plate"][_rim(info["size"])]), "rim changed"
            assert set(np.unique(out[1:-1, 1:-1])) <= {info["background"], info["material"]}
            assert 0 < holes.sum() < holes.size, "uninformative aperture"
            assert np.count_nonzero(out == info["background"]) == holes.sum(), "carved cell count"
            if split == "train":
                defeated.update(h for h in _ALTERNATIVES if not np.array_equal(holes, _holes(info, h)))
    assert set(_ALTERNATIVES) <= defeated, f"training ambiguity: {set(_ALTERNATIVES)-defeated}"
    lengths = {info["counts"][c] for info in infos["train"] for c in info["selected"] if c in info["counts"]}
    assert {1, 2, 3, 4} <= lengths, "missing unary/cycle teaching evidence"
    assert any(set(i["selected"])-set(i["counts"]) for i in infos["train"]), "missing absent counter evidence"
    # One of two visible structural shifts is present, even at 2/1 counts.
    assert (min(i["k"] for i in infos["test"]) > max(i["k"] for i in infos["train"])
            or min(len(i["selected"]) for i in infos["test"]) > max(len(i["selected"]) for i in infos["train"])), "missing shift"
    return None


def self_test():
    """Bounded 42-seed structural audit and family-specific metamorphic checks."""
    fingerprints, shapes, sizes, selected_counts, distractors, backgrounds = set(), set(), set(), set(), set(), set()
    signatures, lengths, orientations, areas, component_counts, symmetries = set(), set(), set(), set(), set(), set()
    regime_counts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(pair["input"]) for s in puzzle for pair in puzzle[s]))
        local = {s: [_decode(p["input"]) for p in puzzle[s]] for s in puzzle}
        regime_counts["scale" if min(i["k"] for i in local["test"]) > max(i["k"] for i in local["train"]) else "participants"] += 1
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], local[split]):
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                shapes.add(pair["input"].shape); sizes.add(info["k"])
                selected_counts.add(len(info["selected"]))
                distractors.add(len(info["patterns"])-len(info["selected"]))
                backgrounds.add(info["background"])
                lengths.update(info["counts"].values()); orientations.update(info["orientations"].values())
                areas.add(int(_holes(info).sum())); component_counts.add(len(_components(_holes(info))))
                roles = {}
                signatures.add(tuple(tuple(roles.setdefault(int(c), len(roles)) for c in row) for row in pair["input"]))
                for mask in info["patterns"].values():
                    symmetries.add("quarter" if np.array_equal(mask, np.rot90(mask)) else "half" if np.array_equal(mask, np.rot90(mask, 2)) else "asymmetric")
        if seed < 3:
            other = generate_puzzle(seed)
            assert all(np.array_equal(a[role], b[role]) for s in puzzle for a, b in zip(puzzle[s], other[s]) for role in ("input", "output"))
    assert len(fingerprints) == 42 and len(signatures) == 210
    assert sizes == {3, 4, 5, 6} and selected_counts == {2, 3, 4}
    assert {0, 1, 2} <= distractors and backgrounds == set(range(10))
    assert lengths == set(range(1, 10)) and orientations == {"h", "v", "single"}
    assert len(shapes) >= 15 and len(areas) >= 8 and max(component_counts) >= 2
    assert "asymmetric" in symmetries and "quarter" in symmetries
    assert min(regime_counts.values()) >= 5 and len(regime_counts) == 2
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), ("4", 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(99, 2, 1)
    bad = copy.deepcopy(p)
    info = _decode(bad["test"][0]["input"])
    r, c = map(int, np.argwhere(_holes(info))[0])
    bad["test"][0]["output"][r+1, c+1] = info["material"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect carving accepted")
    # Relabel every literal color: geometry and roles alone must still solve.
    pair = p["train"][0]
    permutation = np.array([8, 4, 6, 1, 9, 2, 0, 7, 5, 3], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    # Removing a selected color's command must alter an informative witness.
    pair = p["train"][0]
    info = _decode(pair["input"])
    changed = False
    for color in info["selected"]:
        variant = copy.deepcopy(info)
        variant["counts"].pop(color, None)
        changed |= not np.array_equal(_holes(info), _holes(variant))
    assert changed
    return {"seeds": 42, "pairs": 210, "distinct_structures": len(signatures),
            "input_shapes": len(shapes), "register_sizes": sorted(sizes),
            "participants": sorted(selected_counts), "distractor_counts": sorted(distractors),
            "hole_areas": sorted(areas), "aperture_components": sorted(component_counts),
            "source_symmetries": sorted(symmetries), "shifts": dict(regime_counts)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
