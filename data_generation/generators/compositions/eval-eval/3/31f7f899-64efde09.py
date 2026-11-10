"""Reconstruct complementary bar halves, rank their spans, rebuild keyed copies.

No files, parent modules, stored examples, or generator state are used by solve.
The three-cell L at the head of each two-row rail supplies its local coordinates.
"""

import numpy as np


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _components(a, bg):
    seen = np.zeros(a.shape, dtype=bool)
    result = []
    for r, c in np.argwhere(a != bg):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for u, v in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if (0 <= u < a.shape[0] and 0 <= v < a.shape[1]
                        and not seen[u, v] and a[u, v] != bg):
                    seen[u, v] = True
                    stack.append((u, v))
        result.append(cells)
    return result


def _read_canonical(a, bg, row):
    """Recognize the keyed rail and exact rectangular measurements, if present."""
    h, length = a.shape
    if length < 7 or not 0 <= row < h-1:
        return None
    rail, key = int(a[row, 0]), int(a[row, 1])
    if (len({bg, rail, key}) != 3 or a[row+1, 0] != key
            or a[row+1, 1] != key or np.count_nonzero(a == key) != 3
            or a[row, 2] != rail or a[row, -1] != rail
            or np.any(a[row, 2:] != a[row+1, 2:])):
        return None
    skeleton = a[row:row+2].copy()
    rebuilt = np.full_like(a, bg)
    rebuilt[row:row+2] = skeleton
    slots, col = [], 2
    previous_slot = False
    while col < length:
        color = int(a[row, col])
        end = col+1
        while end < length and a[row, end] == color:
            end += 1
        if color == rail:
            previous_slot = False
        else:
            if color in (bg, key) or previous_slot:
                return None
            positions = np.flatnonzero(np.any(a[:, col:end] != bg, axis=1))
            top, bottom = int(positions[0]), int(positions[-1])
            if not np.all(a[top:bottom+1, col:end] == color):
                return None
            rebuilt[top:bottom+1, col:end] = color
            slots.append((col, end-col, color, row-top, bottom-row-1))
            previous_slot = True
        col = end
    if not 2 <= len(slots) <= 5 or len({s[2] for s in slots}) < 2:
        return None
    if not np.array_equal(a, rebuilt):
        return None
    return skeleton, slots, rail, key


def _read_objects(grid):
    a = _grid(grid)
    frequencies = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    if np.count_nonzero(frequencies == frequencies[bg]) != 1:
        raise ValueError("background must be the unique most frequent color")
    components = _components(a, bg)
    if not 2 <= len(components) <= 4:
        raise ValueError("expected two through four separate keyed rails")
    objects = []
    for cells in components:
        rr, cc = zip(*cells)
        r, c = min(rr), min(cc)
        crop = a[r:max(rr)+1, c:max(cc)+1]
        labels = np.arange(crop.size).reshape(crop.shape)
        candidates = []
        for t in range(8):
            b = _orient(crop, t)
            # A key must occupy the first two columns and be an isolated L of
            # its own color. Test those rows before attempting reconstruction.
            for s in range(b.shape[0]-1):
                if (b.shape[1] < 7 or b[s, 1] == bg
                        or b[s, 1] != b[s+1, 0] or b[s, 1] != b[s+1, 1]
                        or b[s, 0] == b[s, 1]):
                    continue
                parsed = _read_canonical(b, bg, s)
                if parsed is None:
                    continue
                skel, slots, rail, key = parsed
                lab = _orient(labels, t)

                def position(x, y):
                    index = int(lab[x, y])
                    return np.array((r+index//crop.shape[1], c+index % crop.shape[1]))

                origin = position(s, 0)
                down, along = position(s+1, 0)-origin, position(s, 1)-origin
                candidates.append({"origin": tuple(map(int, origin)),
                                   "down": tuple(map(int, down)),
                                   "along": tuple(map(int, along)),
                                   "skeleton": skel, "slots": slots,
                                   "rail": rail, "key_color": key, "cells": cells})
        if len(candidates) != 1:
            raise ValueError("rail key or measurements are ambiguous")
        objects.append(candidates[0])
    first = objects[0]
    for obj in objects[1:]:
        if not np.array_equal(obj["skeleton"], first["skeleton"]):
            raise ValueError("rail skeletons are not exact transformed copies")
    return a, bg, objects


def _pool(objects):
    spans = []
    for j in range(len(objects[0]["slots"])):
        upper = {obj["slots"][j][3] for obj in objects if obj["slots"][j][3] > 0}
        lower = {obj["slots"][j][4] for obj in objects if obj["slots"][j][4] > 0}
        if len(upper) != 1 or len(lower) != 1:
            raise ValueError("every local half needs consistent positive evidence")
        spans.append((upper.pop(), lower.pop()))
    ties = {}
    for span in spans:
        if ties.setdefault(sum(span), span) != span:
            raise ValueError("equal total spans must have identical geometry")
    return spans


def _paint(shape, bg, objects, assignments):
    out = np.full(shape, bg, dtype=np.int8)
    occupied = set()
    for obj, spans in zip(objects, assignments):
        origin, down, along = (np.array(obj[k]) for k in ("origin", "down", "along"))
        values = {(r, c): int(obj["skeleton"][r, c])
                  for r in range(2) for c in range(obj["skeleton"].shape[1])}
        for (col, width, color, _, _), (upper, lower) in zip(obj["slots"], spans):
            for r in range(-upper, 2+lower):
                for c in range(col, col+width):
                    values[r, c] = color
        cells = set()
        for (r, c), color in values.items():
            x, y = map(int, origin+r*down+c*along)
            if not (0 <= x < shape[0] and 0 <= y < shape[1]):
                raise ValueError("reconstructed bar would leave the canvas")
            if (x, y) in occupied:
                raise ValueError("reconstructed objects overlap")
            cells.add((x, y))
            out[x, y] = color
        if any((x+dr, y+dc) in occupied for x, y in cells
               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
            raise ValueError("reconstructed objects would join")
        occupied.update(cells)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer missing half spans and rank complete intervals in keyed coordinates."""
    a, bg, objects = _read_objects(input_grid)
    spans = sorted(_pool(objects), key=sum)
    return _paint(a.shape, bg, objects, [spans]*len(objects))


def _features(a, objects, spans):
    widths = [s[1] for s in objects[0]["slots"]]
    colors = [s[2] for s in objects[0]["slots"]]
    ordered = sorted(spans, key=sum)
    patterns = [tuple((s[3] > 0, s[4] > 0) for s in obj["slots"]) for obj in objects]
    return {"shape": tuple(a.shape), "slots": len(spans), "copies": len(objects),
            "widths": tuple(widths), "spans": tuple(spans),
            "lengths": tuple(2+sum(s) for s in spans),
            "poses": tuple((obj["down"], obj["along"]) for obj in objects),
            "pooled": all(not all(u and d for u, d in p) for p in patterns),
            "no_complete_bar": all(not (u and d) for p in patterns for u, d in p),
            "mixed_sides": any(any(u for u, _ in p) and any(d for _, d in p) for p in patterns),
            "repeated_color": len(set(colors)) < len(colors),
            "changed_order": spans != ordered,
            "non_reverse": spans[::-1] != ordered,
            "area_disagrees": [spans[i] for i in sorted(range(len(spans)),
                                  key=lambda i: (2+sum(spans[i]))*widths[i])] != ordered,
            "upper_disagrees": sorted(spans, key=lambda s: s[0]) != ordered,
            "max_disagrees": sorted(spans, key=max) != ordered,
            "asymmetric": any(u != d for u, d in spans),
            "unhinted": any(not any(u or d for u, d in p) for p in patterns)}


def _check_pair(pair):
    a, bg, objects = _read_objects(pair["input"])
    b, bg2, finished = _read_objects(pair["output"])
    if a.shape != b.shape or bg != bg2 or len(objects) != len(finished):
        raise AssertionError("canvas or object count changed")
    spans = _pool(objects)
    final = {obj["origin"]: obj for obj in finished}
    for obj in objects:
        if obj["origin"] not in final:
            raise AssertionError("a rail moved")
        target = final[obj["origin"]]
        assert obj["down"] == target["down"] and obj["along"] == target["along"]
        assert np.array_equal(obj["skeleton"], target["skeleton"])
        assert [s[:3] for s in obj["slots"]] == [s[:3] for s in target["slots"]]
        result = [(s[3], s[4]) for s in target["slots"]]
        assert all(u > 0 and d > 0 for u, d in result), "completion is mandatory"
        assert all(sum(s) <= sum(t) for s, t in zip(result, result[1:])), "unsorted spans"
        remaining = list(spans)
        for span in result:
            assert span in remaining, "full oriented interval must be conserved"
            remaining.remove(span)
        assert not remaining
    # Component parsing explains every cell in both grids, so these relational
    # constraints determine every output pixel without calling solve or sorting.
    return _features(a, objects, spans)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            features[split].append(_check_pair(pair))
    train = features["train"]
    assert all(f["pooled"] for f in train), "training must require pooling"
    assert sum(f["changed_order"] for f in train) >= 2
    assert any(f["no_complete_bar"] for f in train), "show complementary half reconstruction"
    assert any(f["area_disagrees"] and f["non_reverse"] for f in train)
    assert any(f["upper_disagrees"] and f["asymmetric"] for f in train)
    assert any(f["repeated_color"] and f["mixed_sides"] for f in train)
    poses = [p for f in train for p in f["poses"]]
    assert {abs(p[1][0]) for p in poses} == {0, 1}, "show both rail axes"
    assert any(len({p[1] for p in f["poses"]}) > 1 for f in train), "show local reading direction"


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _skeleton(rng, n, palette, fallback=False, repeated=False):
    bg, rail, key = palette[:3]
    widths = list(map(int, rng.integers(1, 4, size=n)))
    gaps = list(map(int, rng.integers(1, 3, size=n+1)))
    if fallback:
        widths, gaps = [1, 2]+[1]*(n-2), [1]*(n+1)
    # Mixed widths are deliberate: perpendicular length and area must disagree
    # in the first demonstration. Limit rail length to keep packing bounded.
    while 2+sum(widths)+sum(gaps) > 2*n+7:
        candidates = [(widths, i) for i, w in enumerate(widths) if w > 1]
        candidates += [(gaps, i) for i, w in enumerate(gaps) if w > 1]
        values, index = candidates[int(rng.integers(len(candidates)))]
        values[index] -= 1
    color_count = n if n == 2 or (not repeated and rng.random() < .55) else int(rng.integers(2, n))
    colors = list(palette[3:3+color_count])
    colors += list(map(int, rng.choice(colors, n-color_count)))
    rng.shuffle(colors)
    length = 2+sum(widths)+sum(gaps)
    skel = np.full((2, length), rail, dtype=np.int8)
    skel[0, 1] = key
    skel[1, :2] = key
    slots, col = [], 2+gaps[0]
    for j, (width, color) in enumerate(zip(widths, colors)):
        skel[:, col:col+width] = color
        slots.append((col, width, color, 0, 0))
        col += width+gaps[j+1]
    return skel, slots


def _sample_spans(rng, n, widths, witness, fallback=False):
    if fallback:
        # Distinct totals, non-reversal input, asymmetric spans, and area
        # disagreement for widths [1,2,1,...]. No alternate fallback rule.
        base = [(3, 1), (1, 2), (2, 3), (3, 3), (1, 1)][:n]
        if n == 2:
            base = [(3, 2), (1, 2)]
        return base
    for _ in range(160):
        span_pool = [(u, d) for u in range(1, 6) for d in range(1, 6)]
        rng.shuffle(span_pool)
        selected, used = [], set()
        for s in span_pool:
            if sum(s) not in used:
                selected.append(s)
                used.add(sum(s))
            if len(selected) == n:
                break
        if not witness and n >= 3 and rng.random() < .25:
            selected[-1] = selected[0]
        rng.shuffle(selected)
        ordered = sorted(selected, key=sum)
        area = [selected[i] for i in sorted(range(n), key=lambda i: (2+sum(selected[i]))*widths[i])]
        if witness:
            if selected == ordered or selected[::-1] == ordered:
                continue
            if witness == 1 and (area == ordered or sorted(selected, key=lambda s: s[0]) == ordered):
                continue
        return selected
    return None


def _observations(rng, n, copies, mode):
    mask = np.zeros((copies, n, 2), dtype=bool)
    if mode == "halves":
        mask[0, :, 0], mask[1, :, 1] = True, True
    else:
        # Opposite half owners differ. Thus a full interval must be assembled
        # across objects even if a copy happens to show measurements on both sides.
        for j in range(n):
            owners = rng.choice(copies, 2, replace=False)
            mask[int(owners[0]), j, 0] = True
            mask[int(owners[1]), j, 1] = True
        # A crossed pair makes the mixed allocation genuinely different from
        # placing one complete side on each of two objects.
        owners = rng.choice(copies, 2, replace=False)
        mask[:, :2] = False
        mask[int(owners[0]), 0, 0] = mask[int(owners[1]), 0, 1] = True
        mask[int(owners[1]), 1, 0] = mask[int(owners[0]), 1, 1] = True
        # Keep each copy incomplete even after optional redundant observations.
        if rng.random() < .35:
            i, j, side = map(int, (rng.integers(copies), rng.integers(n), rng.integers(2)))
            mask[i, j, side] = True
            if np.all(mask[i]):
                mask[i, j, side] = False
    return mask


def _local_pair(skel, slots, spans, observations, bg, transform):
    upper = max(s[0] for s in spans)
    lower = max(s[1] for s in spans)
    inp = np.full((upper+2+lower, skel.shape[1]), bg, dtype=np.int8)
    out = inp.copy()
    inp[upper:upper+2] = skel
    out[upper:upper+2] = skel
    ordered = sorted(spans, key=sum)
    for j, (col, width, color, _, _) in enumerate(slots):
        u = spans[j][0] if observations[j, 0] else 0
        d = spans[j][1] if observations[j, 1] else 0
        inp[upper-u:upper+2+d, col:col+width] = color
        u, d = ordered[j]
        out[upper-u:upper+2+d, col:col+width] = color
    return _orient(inp, transform), _orient(out, transform)


def _place(rng, locals_, bg, shape, tiled=False):
    h, w = shape
    tile_positions = ((0, 0), (0, 16), (16, 0), (16, 16))
    for _ in range(1 if tiled else 7):
        a = np.full(shape, bg, dtype=np.int8)
        b = a.copy()
        blocked = np.zeros(shape, dtype=bool)
        for i, (inp, out) in enumerate(locals_):
            hh, ww = inp.shape
            if hh > h or ww > w:
                break
            footprint = (inp != bg) | (out != bg)
            rr, cc = np.nonzero(footprint)
            for _ in range(1 if tiled else 70):
                r, c = tile_positions[i] if tiled else (int(rng.integers(h-hh+1)), int(rng.integers(w-ww+1)))
                if r+hh > h or c+ww > w or blocked[rr+r, cc+c].any():
                    continue
                view_a, view_b = a[r:r+hh, c:c+ww], b[r:r+hh, c:c+ww]
                view_a[inp != bg], view_b[out != bg] = inp[inp != bg], out[out != bg]
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        x, y = rr+r+dr, cc+c+dc
                        good = (x >= 0) & (x < h) & (y >= 0) & (y < w)
                        blocked[x[good], y[good]] = True
                break
            else:
                break
        else:
            if all(np.argmax(np.bincount(g.ravel(), minlength=10)) == bg for g in (a, b)):
                return {"input": a, "output": b}
    return None


def _make_pair(rng, witness, is_test, shift):
    for attempt in range(25):
        fallback = attempt == 24
        n = (3 if witness == 1 else 4) if witness else int(rng.integers(2, 6))
        copies = (2 if witness == 1 else 3) if witness else int(rng.integers(2, 5))
        if is_test:
            copies = int(rng.integers(3, 5))
        palette = list(map(int, rng.permutation(10)))
        bg = palette[0]
        skel, slots = _skeleton(rng, n, palette, fallback, repeated=witness == 2)
        spans = _sample_spans(rng, n, [s[1] for s in slots], witness, fallback)
        if spans is None:
            continue
        base = int(rng.integers(8))
        if witness == 1:
            transforms = [base, base ^ 4]
        elif witness == 2:
            transforms = [base, (base//4)*4+(base+1) % 4, base ^ 4]
        else:
            transforms = list(map(int, rng.integers(8, size=copies)))
        mode = "halves" if witness == 1 or (witness == 0 and not is_test and rng.random() < .35) else "mixed"
        obs = _observations(rng, n, copies, mode)
        locals_ = [_local_pair(skel, slots, spans, obs[i], bg, t) for i, t in enumerate(transforms)]
        minimum = max(max(x.shape) for p in locals_ for x in p)
        # Test shift: scattered owners and 3-4 copies, with either more or less
        # background margin. All orientation and pooling primitives are trained.
        low = max(minimum, 24 if (is_test == (shift == "expand")) else 14)
        shape = (30, 30) if fallback else tuple(map(int, rng.integers(low, 31, size=2)))
        if not fallback and rng.random() < .22:
            shape = (max(shape), max(shape))
        pair = _place(rng, locals_, bg, shape, tiled=fallback)
        if pair is None:
            continue
        try:
            f = _check_pair(pair)
        except (ValueError, AssertionError):
            continue
        if witness == 1 and not (f["area_disagrees"] and f["upper_disagrees"]
                                 and f["non_reverse"] and f["changed_order"]):
            continue
        if witness == 2 and not (f["changed_order"] and f["non_reverse"]):
            continue
        return pair
    raise RuntimeError("bounded construction failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["expand", "contract"]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = i+1 if split == "train" and i < 2 else 0
            for _ in range(12):
                pair = _make_pair(rng, witness, split == "test", shift)
                k = _key(pair["input"])
                if k not in seen:
                    break
            else:
                raise RuntimeError("distinct-input retries exhausted")
            puzzle[split].append(pair)
            seen.add(k)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded streaming checks: 42 seeds, no saved samples or execution traces."""
    shapes, structures, poses, widths, counts, spans_seen, backgrounds = (set() for _ in range(7))
    flags = {k: 0 for k in ("repeated_color", "unhinted", "mixed_sides", "max_disagrees")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed == 0:
            repeat = generate_puzzle(seed)
            assert all(_key(p[k]) == _key(q[k]) for split in puzzle
                       for p, q in zip(puzzle[split], repeat[split]) for k in p)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, bg, objects = _read_objects(pair["input"])
                spans = _pool(objects)
                f = _features(a, objects, spans)
                assert np.array_equal(solve(a.copy()), pair["output"])
                shapes.add(a.shape)
                structures.add((a.shape, (a != bg).tobytes()))
                poses.update(f["poses"])
                widths.update(f["widths"])
                counts.add(f["slots"])
                spans_seen.update(spans)
                backgrounds.add(bg)
                for flag in flags:
                    flags[flag] += int(f[flag])
                # Omitting A: complete the observed intervals without ranking.
                if f["changed_order"]:
                    try:
                        wrong = _paint(a.shape, bg, objects, [spans]*len(objects))
                    except ValueError:
                        pass  # The alternative can also produce colliding objects.
                    else:
                        assert not np.array_equal(wrong, pair["output"])
                # Omitting B: sort only each object's incomplete local intervals.
                local = [sorted([(s[3], s[4]) for s in obj["slots"]], key=sum) for obj in objects]
                try:
                    wrong = _paint(a.shape, bg, objects, local)
                except ValueError:
                    pass
                else:
                    assert not np.array_equal(wrong, pair["output"])
                # Recoloring and any whole-scene dihedral transform preserve the rule.
                t = (seed+len(split)) % 8
                mapping = np.roll(np.arange(10, dtype=np.int8), seed % 10)
                assert np.array_equal(solve(_orient(mapping[a], t)), _orient(mapping[pair["output"]], t))
    assert len(shapes) >= 35 and len(structures) >= 200
    assert len(poses) == 8 and widths >= {1, 2, 3} and counts == {2, 3, 4, 5}
    assert len(spans_seen) >= 20 and backgrounds == set(range(10))
    assert all(value > 0 for value in flags.values())
    p = generate_puzzle(1001, 2, 1)
    bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("output corruption escaped relational validation")
    for nt, nq in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(2000+nt, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                   (True, 1), (3.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
