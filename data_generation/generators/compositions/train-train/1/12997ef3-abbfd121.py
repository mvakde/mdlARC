"""Phase-selective stencil printing. Only NumPy and the standard library are used."""
from __future__ import annotations

import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _primitive(tile):
    return all(not np.array_equal(tile, np.roll(tile, k, axis=axis))
               for axis in (0, 1) for k in range(1, tile.shape[axis]))


def _field(tile, shape):
    return tile[np.arange(shape[0])[:, None] % tile.shape[0],
                np.arange(shape[1])[None, :] % tile.shape[1]]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    group.append(p)
                    stack.append(p)
        result.append(group)
    return result


def _stencil(inp, group):
    points = np.asarray(group, dtype=int)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    values, counts = np.unique(inp[points[:, 0], points[:, 1]], return_counts=True)
    if len(values) != 2 or int(counts.min()) != 1 or int(counts.max()) < 3:
        return None
    anchor_color = int(values[np.argmin(counts)])
    body_color = int(values[np.argmax(counts)])
    anchor = points[inp[points[:, 0], points[:, 1]] == anchor_color][0] - lo
    mask = np.zeros(tuple(hi - lo), dtype=bool)
    mask[points[:, 0] - lo[0], points[:, 1] - lo[1]] = True
    return {"mask": mask, "anchor": tuple(map(int, anchor)),
            "bbox": (*map(int, lo), *map(int, hi - lo)),
            "body": body_color, "anchor_color": anchor_color, "area": len(group)}


def _interpret(inp, tile):
    wallpaper = _field(tile, inp.shape)
    covered = inp != wallpaper
    palette = set(map(int, np.unique(tile)))
    foreign = set(map(int, np.unique(inp[covered])))
    if not 2 <= len(palette) <= 3 or 0 in palette | foreign or palette & foreign:
        return None
    groups = _components(covered)
    stencils, markers = [], []
    # Objects have a one-cell moat, even diagonally. Wallpaper is not an object.
    label = np.full(inp.shape, -1, dtype=int)
    for i, g in enumerate(groups):
        for r, c in g:
            label[r, c] = i
    for i, g in enumerate(groups):
        for r, c in g:
            neighborhood = label[max(0, r-1):r+2, max(0, c-1):c+2]
            if np.any((neighborhood >= 0) & (neighborhood != i)):
                return None
        if len(g) == 1:
            r, c = g[0]
            markers.append((r, c, int(inp[r, c])))
        else:
            s = _stencil(inp, g)
            if s is None:
                return None
            stencils.append(s)
    if not 1 <= len(stencils) <= 2 or not 2 <= len(markers) <= 6:
        return None
    anchor_colors = {s["anchor_color"] for s in stencils}
    bodies = {s["body"] for s in stencils}
    inks = {m[2] for m in markers}
    if len(anchor_colors) != 1 or len(bodies) != len(stencils):
        return None
    if bodies & anchor_colors or inks & (bodies | anchor_colors) or len(inks) < 2:
        return None
    areas = sorted((s["area"] for s in stencils), reverse=True)
    if len(areas) == 2 and areas[0] == areas[1]:
        return None
    selected = max(stencils, key=lambda s: s["area"])
    if len({m[0] for m in markers}) == 1:
        axis = 1
    elif len({m[1] for m in markers}) == 1:
        axis = 0
    else:
        return None
    markers.sort(key=lambda m: m[axis])
    if any(b[axis] - a[axis] < 2 for a, b in zip(markers, markers[1:])):
        return None
    h, w = selected["mask"].shape
    ar, ac = selected["anchor"]
    if (h * (len(markers) if axis == 0 else 1) > 30
            or w * (len(markers) if axis == 1 else 1) > 30):
        return None
    if any(not (0 <= r-ar <= inp.shape[0]-h and 0 <= c-ac <= inp.shape[1]-w)
           for r, c, _ in markers):
        return None
    return {"tile": tile, "period": tile.shape, "wallpaper": wallpaper,
            "covered": covered, "stencils": stencils, "selected": selected,
            "markers": markers, "axis": axis, "palette": tuple(sorted(palette))}


def _decode(input_grid):
    inp = _grid(input_grid)
    candidates = []
    for ph in range(1, 5):
        for pw in range(1, 5):
            if ph * pw < 2:
                continue
            tile = np.empty((ph, pw), dtype=np.int8)
            valid = True
            for r in range(ph):
                for c in range(pw):
                    counts = np.bincount(inp[r::ph, c::pw].ravel(), minlength=10)
                    order = np.sort(counts)
                    if order[-1] < 3 or order[-1] <= 2 * order[-2]:
                        valid = False
                        break
                    tile[r, c] = int(np.argmax(counts))
                if not valid:
                    break
            if not valid or not _primitive(tile):
                continue
            info = _interpret(inp, tile)
            if info is not None:
                candidates.append(info)
    if not candidates:
        raise ValueError("no supported periodic field and stencil/dot interpretation")
    candidates.sort(key=lambda a: np.prod(a["period"]))
    best = candidates[0]
    if any(not np.array_equal(best["wallpaper"], a["wallpaper"]) for a in candidates[1:]):
        raise ValueError("ambiguous wallpaper")
    return best


def _stamp(mask, anchor, markers, wallpaper, axis):
    ar, ac = anchor
    h, w = mask.shape
    tiles, gates = [], []
    for r, c, ink in markers:
        patch = wallpaper[r-ar:r-ar+h, c-ac:c-ac+w]
        gate = mask & (patch == wallpaper[r, c])
        tiles.append(np.where(gate, ink, 0).astype(np.int8))
        gates.append(gate)
    return np.concatenate(tiles, axis=axis), gates


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only the supplied input; no generation state or example lookup."""
    a = _decode(input_grid)
    s = a["selected"]
    return _stamp(s["mask"], s["anchor"], a["markers"], a["wallpaper"], a["axis"])[0]


def _sample_tile(rng, period):
    ph, pw = period
    k = int(rng.integers(2, min(3, ph*pw) + 1))
    for _ in range(120):
        mode = int(rng.integers(4))
        if mode == 0 and ph == pw == k:
            t = (np.arange(ph)[:, None] + np.arange(pw)[None, :]) % k
        else:
            t = rng.integers(k, size=period)
            if mode == 1 and ph > 1 and pw > 1:
                t[0, :] = 0
                t[:, 0] = 0
            elif mode == 2 and ph > 1:
                t[0, :] = 0
        if len(np.unique(t)) == k and _primitive(t):
            return t, k
    raise RuntimeError("tile sampling exhausted")


def _sample_mask(rng, axis, count, larger=False):
    cap = 6 if larger else 5
    for _ in range(300):
        h = int(rng.integers(3, cap + 1))
        w = int(rng.integers(3, cap + 1))
        if (h if axis == 0 else w) * count > 30:
            continue
        mode = int(rng.integers(5))
        m = np.zeros((h, w), dtype=bool)
        if mode == 0:  # Off-center crosses and branching shapes.
            m[int(rng.integers(h)), :] = True
            m[:, int(rng.integers(w))] = True
            m |= rng.random((h, w)) < .24
        elif mode == 1:  # A frame can retain genuine internal holes.
            m[0, :] = m[-1, :] = True
            m[:, 0] = m[:, -1] = True
        elif mode == 2:  # A comb.
            m[:, 0] = True
            m[::2, :] = True
            m[-1, -1] = True
        else:
            m = rng.random((h, w)) < float(rng.uniform(.55, .88))
        if (int(m.sum()) >= 8 and not m.all() and m[0].any() and m[-1].any()
                and m[:, 0].any() and m[:, -1].any() and len(_components(m)) == 1):
            return m
    raise RuntimeError("mask sampling exhausted")


def _decoy_mask(rng, mask, side):
    h, w = mask.shape
    area = int(mask.sum())
    choices = []
    for dh in range(2, 9):
        for dw in range(2, 9):
            if (dh*dw > h*w and dh+dw-1 < area
                    and (dh > h if side == 0 else dw > w)):
                choices.append((dh, dw))
    if not choices:
        return None
    dh, dw = choices[int(rng.integers(len(choices)))]
    decoy = np.zeros((dh, dw), dtype=bool)
    decoy[:, 0] = True
    decoy[-1, :] = True
    # Add connected branches without reaching the selected stencil's area.
    desired = int(rng.integers(dh+dw-1, area))
    while int(decoy.sum()) < desired:
        choices = [(r, c) for r in range(dh) for c in range(dw) if not decoy[r, c]
                   and any(0 <= rr < dh and 0 <= cc < dw and decoy[rr, cc]
                           for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)))]
        r, c = choices[int(rng.integers(len(choices)))]
        decoy[r, c] = True
    if rng.random() < .5:
        decoy = decoy[::-1]
    if rng.random() < .5:
        decoy = decoy[:, ::-1]
    return decoy.copy()


def _registration_sites(rng, shape, mask, anchor, axis, count):
    h, w = mask.shape
    ar, ac = anchor
    lo = (ar, ac)
    hi = (shape[0]-h+ar, shape[1]-w+ac)
    steps = rng.integers(2, 5, size=count-1)
    mains = np.r_[0, np.cumsum(steps)]
    if mains[-1] > hi[axis]-lo[axis]:
        return None
    mains += int(rng.integers(lo[axis], hi[axis]-int(mains[-1])+1))
    other = int(rng.integers(lo[1-axis], hi[1-axis]+1))
    return [(int(v), other) if axis == 0 else (other, int(v)) for v in mains]


def _place_stencils(rng, shape, masks, markers):
    blocked = np.zeros(shape, dtype=bool)

    def reserve(points):
        for r, c in points:
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True

    reserve(markers)
    result = []
    for mask in masks:
        h, w = mask.shape
        rr, cc = np.nonzero(mask)
        locations = rng.permutation((shape[0]-h+1) * (shape[1]-w+1))
        for flat in locations:
            r, c = divmod(int(flat), shape[1]-w+1)
            if not blocked[r+rr, c+cc].any():
                result.append((r, c))
                reserve(zip(r+rr, c+cc))
                break
        else:
            return None
    return result


def _make_pair(rng, axis, count, two_stencils, side_goal, larger=False, teach=False):
    periods = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2),
               (3, 4), (4, 3), (4, 4), (1, 3), (3, 1), (1, 4), (4, 1))
    for _ in range(350):
        mask = _sample_mask(rng, axis, count, larger)
        decoy = _decoy_mask(rng, mask, side_goal) if two_stencils else None
        if two_stencils and decoy is None:
            continue
        masks = [mask] + ([decoy] if decoy is not None else [])
        anchor_choices = np.argwhere(mask)
        if teach:
            anchor_choices = anchor_choices[np.any(anchor_choices != (0, 0), axis=1)]
        anchor = tuple(map(int, anchor_choices[int(rng.integers(len(anchor_choices)))]))
        shape = (int(rng.integers(18, 31)), int(rng.integers(18, 31)))
        sites = _registration_sites(rng, shape, mask, anchor, axis, count)
        if sites is None:
            continue
        positions = _place_stencils(rng, shape, masks, sites)
        if positions is None:
            continue
        period = periods[int(rng.integers(len(periods)))]
        raw, k = _sample_tile(rng, period)
        colors = list(map(int, rng.permutation(np.arange(1, 10))))
        field_colors = colors[:k]
        body_colors = colors[k:k+len(masks)]
        anchor_color = colors[k+len(masks)]
        ink_pool = colors[k+len(masks)+1:]
        ink_count = min(count, len(ink_pool), int(rng.integers(2, 4)))
        inks = list(map(int, rng.choice(ink_pool[:ink_count], count)))
        # Different endpoint inks make reversal distinguishable even with repetition.
        inks[0], inks[-1] = ink_pool[0], ink_pool[1]
        markers = [(r, c, ink) for (r, c), ink in zip(sites, inks)]
        wallpaper = _field(np.asarray(field_colors, dtype=np.int8)[raw], shape)
        out, gates = _stamp(mask, anchor, markers, wallpaper, axis)
        # Every pair teaches selective printing and phase-dependent geometry.
        if (any(g.all() or int(g.sum()) < 2 or np.array_equal(g, mask) for g in gates)
                or len({g.tobytes() for g in gates}) < 2):
            continue
        solid, _ = _stamp(np.ones_like(mask), anchor, markers, wallpaper, axis)
        if np.array_equal(solid, out):
            continue
        inp = wallpaper.copy()
        for i, (m, (r, c)) in enumerate(zip(masks, positions)):
            rr, cc = np.nonzero(m)
            inp[r+rr, c+cc] = body_colors[i]
            ap = anchor if i == 0 else tuple(map(int, np.argwhere(m)[int(rng.integers(int(m.sum())))]))
            inp[r+ap[0], c+ap[1]] = anchor_color
        for r, c, ink in markers:
            inp[r, c] = ink
        try:
            a = _decode(inp)
        except ValueError:
            continue
        if (not np.array_equal(a["selected"]["mask"], mask)
                or a["selected"]["anchor"] != anchor or a["period"] != period):
            continue
        if teach and np.array_equal(out, _cellwise_output(a, anchor=(0, 0))):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct a recoverable phase-selective print")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    first_axis = int(rng.integers(2))
    shared_count = int(rng.integers(2, 4))
    schedule = [(first_axis, shared_count, True, 0, False, True),
                (1-first_axis, shared_count, True, 1, False, True)]
    for _ in range(num_train-2):
        schedule.append((int(rng.integers(2)), int(rng.integers(2, 5)),
                         bool(rng.random() < .7), int(rng.integers(2)), False, False))
    rng.shuffle(schedule)
    max_train_count = max(item[1] for item in schedule)
    tests = [(int(rng.integers(2)), int(rng.integers(max_train_count+1, 7)),
              bool(rng.random() < .8), int(rng.integers(2)), True, False)]
    for _ in range(num_test-1):
        tests.append((int(rng.integers(2)), int(rng.integers(2, 7)),
                      bool(rng.random() < .7), int(rng.integers(2)), True, False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, plans in (("train", schedule), ("test", tests)):
        for plan in plans:
            for _ in range(12):
                pair = _make_pair(rng, *plan)
                key = _signature(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("duplicate input construction")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _cellwise_output(a, anchor=None):
    """A slow, direct definition used to check the vectorized renderer."""
    s = a["selected"]
    mask = s["mask"]
    ar, ac = s["anchor"] if anchor is None else anchor
    h, w = mask.shape
    n = len(a["markers"])
    result = np.zeros((h*n if a["axis"] == 0 else h,
                       w*n if a["axis"] == 1 else w), dtype=np.int8)
    tile = a["tile"]
    ph, pw = tile.shape
    for i, (r, c, ink) in enumerate(a["markers"]):
        channel = tile[r % ph, c % pw]
        for u, v in np.argwhere(mask):
            if tile[(r+int(u)-ar) % ph, (c+int(v)-ac) % pw] == channel:
                result[u+(i*h if a["axis"] == 0 else 0),
                       v+(i*w if a["axis"] == 1 else 0)] = ink
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            a = _decode(inp)
            s = a["selected"]
            assert np.array_equal(out, _cellwise_output(a)), "wrong registered print"
            assert np.array_equal(out, solve(inp)), "solver disagreement"
            expected, gates = _stamp(s["mask"], s["anchor"], a["markers"], a["wallpaper"], a["axis"])
            assert np.array_equal(out, expected)
            assert all(2 <= int(g.sum()) < s["area"] for g in gates)
            assert len({g.tobytes() for g in gates}) >= 2, "no phase-dependent shape change"
            solid = _stamp(np.ones_like(s["mask"]), s["anchor"], a["markers"], a["wallpaper"], a["axis"])[0]
            assert not np.array_equal(solid, out), "stencil holes do not affect answer"
            full_mask = np.concatenate([np.where(s["mask"], ink, 0) for _, _, ink in a["markers"]], axis=a["axis"])
            assert not np.array_equal(full_mask, out), "wallpaper does not affect answer"
            ar, ac = s["anchor"]
            assert all(g[ar, ac] for g in gates), "registration pixel must print"
            assert 0 in out and set(map(int, np.unique(out))) == {0} | {m[2] for m in a["markers"]}
            for r in range(a["period"][0]):
                for c in range(a["period"][1]):
                    assert np.sum(inp[r::a["period"][0], c::a["period"][1]] == a["tile"][r, c]) >= 3
            analyses[split].append(a)
    train = analyses["train"]
    assert {a["axis"] for a in train} == {0, 1}
    assert any(not np.array_equal(_cellwise_output(a), _cellwise_output(a, anchor=(0, 0)))
               for a in train), "registration alignment has no demonstrated effect"
    assert any({0, 1} <= {a["axis"] for a in train if len(a["markers"]) == n}
               for n in {len(a["markers"]) for a in train}), "count predicts direction"
    for side in (0, 1):
        assert any(any(d["area"] < a["selected"]["area"]
                           and d["mask"].shape[side] > a["selected"]["mask"].shape[side]
                           and d["mask"].size > a["selected"]["mask"].size
                           for d in a["stencils"]) for a in train), "area-selection evidence missing"
    assert len(analyses["test"][0]["markers"]) > max(len(a["markers"]) for a in train)


def _puzzle_key(p):
    return tuple(_signature(pair[k]) for s in ("train", "test") for pair in p[s] for k in ("input", "output"))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, outputs, masks, periods, roles, layouts, anchors, counts = (set() for _ in range(8))
    hidden_samples = repeated = decoys = aligned = nonaligned = 0
    for p in puzzles:
        for split in ("train", "test"):
            for pair in p[split]:
                a = _decode(pair["input"])
                s = a["selected"]
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                masks.add(_signature(s["mask"]))
                periods.add(a["period"])
                roles.add((a["palette"], s["body"], s["anchor_color"]))
                anchors.add(s["anchor"])
                counts.add(len(a["markers"]))
                layouts.add((a["period"], a["axis"], len(a["markers"]), s["mask"].shape, s["anchor"]))
                repeated += len({m[2] for m in a["markers"]}) < len(a["markers"])
                decoys += len(a["stencils"]) == 2
                ph, pw = a["period"]
                aligned += any(r % ph == c % pw == 0 for r, c, _ in a["markers"])
                nonaligned += any(r % ph != 0 or c % pw != 0 for r, c, _ in a["markers"])
                ar, ac = s["anchor"]
                for r, c, _ in a["markers"]:
                    hidden_samples += sum(bool(a["covered"][r+int(u)-ar, c+int(v)-ac])
                                          for u, v in np.argwhere(s["mask"])
                                          if (int(u), int(v)) != (ar, ac))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(shapes) >= 40 and len(outputs) >= 25 and len(masks) >= 60
    assert len(periods) >= 9 and len(roles) >= 100 and len(layouts) >= 150
    assert len(anchors) >= 10 and {2, 3, 4, 5, 6} <= counts
    assert hidden_samples >= 20 and repeated >= 40 and decoys >= 100
    assert aligned >= 10 and nonaligned >= 150
    assert _puzzle_key(generate_puzzle(12345, 6, 3)) == _puzzle_key(generate_puzzle(12345, 6, 3))
    # Color renaming preserves black backing; transposition changes both axes together.
    rng = np.random.default_rng(892)
    perm = np.r_[0, rng.permutation(np.arange(1, 10))]
    for p in puzzles[:3]:
        for pair in p["train"] + p["test"]:
            assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
            assert np.array_equal(solve(pair["input"].T), pair["output"].T)
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4., 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {nt!r}/{ns!r}")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes), "output_shapes": len(outputs),
            "stencil_masks": len(masks), "periods": sorted(periods), "role_assignments": len(roles),
            "layouts": len(layouts), "anchors": len(anchors), "marker_counts": sorted(counts),
            "repeated_ink_pairs": repeated, "two_stencil_pairs": decoys,
            "hidden_nonanchor_samples": hidden_samples, "has_phase_zero": aligned,
            "has_nonzero_phase": nonaligned}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
