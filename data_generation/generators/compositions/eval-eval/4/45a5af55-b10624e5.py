"""Marked beacons unfold directional color codes into competing square shells.

Only standard-library modules and NumPy are used. No files, global random state,
examples, or generation metadata enter solve(). Generation retains no traces.
"""
from __future__ import annotations

import hashlib
import numpy as np


# Core tuples are (top, left, unit, key_row, key_col). The square side is 2*unit.
# Direction order is north, east, south, west throughout the family.
_INF = 99


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.uint8).tobytes()


def _grid(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        stack = [remaining.pop()]
        part = []
        while stack:
            r, c = stack.pop()
            part.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(part)
    return result


def _core_roles(grid, bg):
    candidates = []
    for body in map(int, np.unique(grid)):
        if body == bg:
            continue
        cores, keys = [], []
        for part in _components(grid == body):
            ys, xs = zip(*part)
            y, x, h, w = min(ys), min(xs), max(ys)-min(ys)+1, max(xs)-min(xs)+1
            if h != w or h % 2 or len(part) != 3*(h//2)**2:
                break
            a = h//2
            blocks = [grid[y+r*a:y+(r+1)*a, x+c*a:x+(c+1)*a]
                      for r in range(2) for c in range(2)]
            if any(not np.all(b == b[0, 0]) for b in blocks):
                break
            colors = [int(b[0, 0]) for b in blocks]
            missing = [i for i, color in enumerate(colors) if color != body]
            if len(missing) != 1 or colors[missing[0]] == bg:
                break
            k = missing[0]
            cores.append((y, x, a, k//2, k % 2))
            keys.append(colors[k])
        else:
            if len(cores) >= 3 and len(set(keys)) == 1:
                key = keys[0]
                if np.count_nonzero(grid == key) == sum(c[2]**2 for c in cores):
                    candidates.append((body, key, sorted(cores)))
    assert len(candidates) == 1, "expected one repeated marked-square identity"
    return candidates[0]


def _strip(core, direction, k):
    y, x, a, _, _ = core
    if direction == 0:
        return y-k*a, y-(k-1)*a, x, x+2*a
    if direction == 1:
        return y, y+2*a, x+(k+1)*a, x+(k+2)*a
    if direction == 2:
        return y+(k+1)*a, y+(k+2)*a, x, x+2*a
    return y, y+2*a, x-k*a, x-(k-1)*a


def _core_paint(grid, core, body, key):
    y, x, a, kr, kc = core
    grid[y:y+2*a, x:x+2*a] = body
    grid[y+kr*a:y+(kr+1)*a, x+kc*a:x+(kc+1)*a] = key


def _bbox(core, length):
    y, x, a, _, _ = core
    return y-length*a, y+(length+2)*a, x-length*a, x+(length+2)*a


def _disjoint(first, second, gap=0):
    a, b, c, d = first
    e, f, g, h = second
    return b+gap <= e or f+gap <= a or d+gap <= g or h+gap <= c


def _decode(value):
    grid = _grid(value)
    bg = int(grid[0, 0])
    assert np.all(grid[0] == bg) and np.all(grid[-1] == bg)
    assert np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)
    body, key, cores = _core_roles(grid, bg)
    for i, core in enumerate(cores):
        assert all(_disjoint(_bbox(core, 0), _bbox(other, 0), 1)
                   for other in cores[i+1:]), "cores need a blank-cell separation"
    base = np.full(grid.shape, bg, dtype=np.int8)
    for core in cores:
        _core_paint(base, core, body, key)
    references = []
    for index, core in enumerate(cores):
        trial = base.copy()
        codes = []
        valid = True
        for direction in range(4):
            code = []
            for k in range(1, 15):
                r0, r1, c0, c1 = _strip(core, direction, k)
                if r0 < 0 or c0 < 0 or r1 > grid.shape[0] or c1 > grid.shape[1]:
                    break
                strip = grid[r0:r1, c0:c1]
                if np.all(strip == bg):
                    break
                color = int(strip[0, 0])
                if color in (bg, body, key) or not np.all(strip == color):
                    valid = False
                    break
                code.append(color)
                trial[r0:r1, c0:c1] = color
            codes.append(code)
        if valid and len({len(c) for c in codes}) == 1 and codes[0] and np.array_equal(trial, grid):
            references.append((index, np.array(codes, dtype=np.int8)))
    assert len(references) == 1, "expected one complete four-spoked code"
    reference, codes = references[0]
    length = codes.shape[1]
    box = _bbox(cores[reference], length)
    assert 1 <= box[0] < box[1] <= grid.shape[0]-1
    assert 1 <= box[2] < box[3] <= grid.shape[1]-1
    assert all(_disjoint(box, _bbox(core, length), 1)
               for i, core in enumerate(cores) if i != reference), "keep the reference wave isolated"
    return {"grid": grid, "background": bg, "body": body, "key": key,
            "cores": cores, "reference": reference, "codes": codes, "length": length}


def _flipped_codes(codes, reference, core, reflect=True):
    order = [0, 1, 2, 3]
    if reflect and reference[3] != core[3]:
        order[0], order[2] = order[2], order[0]
    if reflect and reference[4] != core[4]:
        order[1], order[3] = order[3], order[1]
    return codes[order]


def _combine(arrivals, colors, bg, order=False):
    best = arrivals.min(axis=0)
    winners = (arrivals == best) & (arrivals < _INF)
    low = np.where(winners, colors, 10).min(axis=0)
    high = np.where(winners, colors, -1).max(axis=0)
    conflict = (best < _INF) & (low != high)
    out = np.where((best < _INF) & ~conflict, low, bg).astype(np.int8)
    if order:
        out.fill(bg)
        for t, c in zip(arrivals, colors):
            out[t < _INF] = c[t < _INF]
    count = winners.sum(axis=0)
    visible = arrivals < _INF
    stats = {"conflict_cells": int(conflict.sum()),
             "agreeing_tie_cells": int(((count > 1) & ~conflict & (best > 0)).sum()),
             "overlap_cells": int((visible.sum(axis=0) > 1).sum()),
             "unequal_overlap": int(((visible & (arrivals > best)).any(axis=0) & (best > 0)).sum()),
             "unique_fronts": [int(((t == best) & (best > 0) & (best < _INF) & (count == 1)).sum())
                               for t in arrivals]}
    return out, stats


def _render(data):
    """Forward construction: paint scaled frame tiles, then merge arrival events."""
    shape, bg = data["grid"].shape, data["background"]
    cores, codes, length = data["cores"], data["codes"], data["length"]
    ref = cores[data["reference"]]
    arrivals = np.full((len(cores), *shape), _INF, dtype=np.int16)
    colors = np.full((len(cores), *shape), bg, dtype=np.int8)
    clipped = 0
    for i, core in enumerate(cores):
        a, kr, kc = core[2:]
        code = _flipped_codes(codes, ref, core)
        side = 2+2*length
        tile = np.full((side, side), data["body"], dtype=np.int8)
        times = np.zeros((side, side), dtype=np.int16)
        tile[length+kr, length+kc] = data["key"]
        for k in range(1, length+1):
            lo, hi = length-k, length+2+k
            tile[lo, lo:hi], tile[hi-1, lo:hi] = code[0, k-1], code[2, k-1]
            tile[lo+1:hi-1, lo], tile[lo+1:hi-1, hi-1] = code[3, k-1], code[1, k-1]
            times[lo, lo:hi] = times[hi-1, lo:hi] = k
            times[lo+1:hi-1, lo] = times[lo+1:hi-1, hi-1] = k
        tile = np.repeat(np.repeat(tile, a, axis=0), a, axis=1)
        times = np.repeat(np.repeat(times, a, axis=0), a, axis=1)
        r0, r1, c0, c1 = _bbox(core, length)
        y0, y1, x0, x1 = max(0, r0), min(shape[0], r1), max(0, c0), min(shape[1], c1)
        clipped += (r0 < 0 or c0 < 0 or r1 > shape[0] or c1 > shape[1])
        colors[i, y0:y1, x0:x1] = tile[y0-r0:y1-r0, x0-c0:x1-c0]
        arrivals[i, y0:y1, x0:x1] = times[y0-r0:y1-r0, x0-c0:x1-c0]
    out, stats = _combine(arrivals, colors, bg)
    stats["clipped_copies"] = int(clipped)
    return out, stats


def _pull(data, *, reflect=True, scale=True, complete=True, reverse=False,
          vertical_corners=True, order=False, pixel_priority=False):
    """Input-only pixel query; also supplies explicit competing-rule checks."""
    grid, cores = data["grid"], data["cores"]
    bg, length = data["background"], data["length"]
    ref = cores[data["reference"]]
    codes = data["codes"][:, ::-1] if reverse else data["codes"]
    rr, cc = np.indices(grid.shape)
    arrivals, colors = [], []
    for core in cores:
        y, x, a, kr, kc = core
        unit = a if scale else ref[2]
        # Block distance from the physical core, rather than distance from its center.
        du = np.maximum(y-rr, rr-(y+2*a)+1).clip(min=0)
        dv = np.maximum(x-cc, cc-(x+2*a)+1).clip(min=0)
        u, v = (du+unit-1)//unit, (dv+unit-1)//unit
        k = np.maximum(u, v)
        vertical = u >= v if vertical_corners else u > v
        direction = np.where(vertical, np.where(rr < y, 0, 2), np.where(cc < x, 3, 1))
        code = _flipped_codes(codes, ref, core, reflect)
        color = code[direction, np.clip(k-1, 0, length-1)].copy()
        inside = k == 0
        color[inside] = data["body"]
        key_mask = ((rr >= y+kr*a) & (rr < y+(kr+1)*a) &
                    (cc >= x+kc*a) & (cc < x+(kc+1)*a))
        color[key_mask] = data["key"]
        valid = k <= length
        if not complete:
            valid &= (u == 0) | (v == 0)
        time = np.maximum(du, dv) if pixel_priority else k
        arrivals.append(np.where(valid, time, _INF))
        colors.append(color)
    return _combine(np.array(arrivals), np.array(colors), bg, order=order)[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the sole spoke code and complete all reflected competing beacons."""
    return _pull(_decode(input_grid))


def _rotate_core(core, height, width, turns):
    y, x, a, kr, kc = core
    for _ in range(turns):
        y, x, kr, kc = width-x-2*a, y, 1-kc, kr
        height, width = width, height
    return y, x, a, kr, kc


def _scene(rng, length, source_unit, units, flags, mode, informative=False):
    """Bounded layouts: isolated calibration wave plus freely interacting targets."""
    permutation = [int(c) for c in rng.permutation(10)]
    bg, body, key = permutation[:3]
    palette = permutation[3:3+int(rng.integers(4 if informative else 2, 8))]
    codes = rng.choice(palette, size=(4, length)).astype(np.int8)
    if informative:
        codes[:, 0] = rng.choice(palette, size=4, replace=False)
        repeat = int(rng.integers(4))
        codes[repeat, 1] = codes[repeat, 0]
        # Opposite outer faces differ at the required collision seam.
        if codes[1, 1] == codes[3, 1]:
            change, other = (1, 3) if repeat == 3 else (3, 1)
            codes[change, 1] = rng.choice([c for c in palette if c != codes[other, 1]])
        if np.array_equal(codes[:, 0], codes[:, 1]):
            change = 2 if repeat == 0 else 0
            codes[change, 1] = rng.choice([c for c in palette if c != codes[change, 0]])
    source_side = (2+2*length)*source_unit
    target_sides = [(2+2*length)*a for a in units]
    minimum = source_side+max(target_sides)+5
    assert minimum <= 30
    height = int(rng.integers(minimum, 31))
    width = int(rng.integers(max(21, source_side+2, max(target_sides)+2), 31))
    source_left = int(rng.integers(1, width-source_side))
    corner = tuple(map(int, rng.integers(0, 2, size=2)))
    source = (1+length*source_unit, source_left+length*source_unit, source_unit, *corner)
    lane = source_side+3
    targets = []
    if mode == "separate":
        slack = width-sum(target_sides)-3
        assert slack >= 0
        gaps = rng.multinomial(slack, [1/3]*3)
        left = 1+int(gaps[0])
        for i, a in enumerate(units):
            top = int(rng.integers(lane, height-target_sides[i]))
            targets.append((top+length*a, left+length*a, a,
                            corner[0] ^ flags[i][0], corner[1] ^ flags[i][1]))
            left += target_sides[i]+1+int(gaps[1])
    elif mode == "collision":
        # The first two equal-speed fronts meet in a color-conflict seam.
        left = int(rng.integers(2, width-11))
        targets = [(height-3, left+offset, 1, corner[0] ^ v, corner[1] ^ h)
                   for offset, (v, h) in zip((0, 5), flags[:2])]
    for index in range(len(targets), len(units)):
        a = units[index]
        choices = []
        for y in range(lane+length*a, height-2*a):
            for x in range(1, width-2*a):
                core = (y, x, a, corner[0] ^ flags[index][0], corner[1] ^ flags[index][1])
                if all(_disjoint(_bbox(core, 0), _bbox(t, 0), 1) for t in targets):
                    choices.append(core)
        if not choices:
            return None
        targets.append(choices[int(rng.integers(len(choices)))])
    cores = [source]+targets
    grid = np.full((height, width), bg, dtype=np.int8)
    for core in cores:
        _core_paint(grid, core, body, key)
    for direction in range(4):
        for k, color in enumerate(codes[direction], 1):
            r0, r1, c0, c1 = _strip(source, direction, k)
            grid[r0:r1, c0:c1] = color
    turns = int(rng.integers(4))
    cores = [_rotate_core(c, height, width, turns) for c in cores]
    grid = np.rot90(grid, turns).copy()
    # Under a counterclockwise quarter turn the old E ray becomes the new N ray.
    codes = np.roll(codes, -turns, axis=0)
    data = {"grid": grid, "background": bg, "body": body, "key": key,
            "cores": cores, "reference": 0, "codes": codes, "length": length}
    out, stats = _render(data)
    if any(n == 0 for n in stats["unique_fronts"]):
        return None
    if mode == "collision":
        if not (stats["conflict_cells"] and stats["agreeing_tie_cells"]
                and stats["unequal_overlap"] and stats["clipped_copies"]):
            return None
        if np.array_equal(_pull(data, pixel_priority=True), out):
            return None
    return {"input": grid, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for attempt in range(96):
                if split == "train" and index == 0:
                    args = (2, 2, [1, 2], [(0, 1), (1, 0)], "separate", True)
                elif split == "train" and index == 1:
                    args = (2, 1, [1, 1, 2], [(1, 1), (1, 1), (0, 0)], "collision", True)
                else:
                    length = 3 if split == "test" else int(rng.integers(1, 3))
                    source_unit = int(rng.integers(1, 3))
                    n = int(rng.integers(3, 5)) if split == "test" else int(rng.integers(2, 5))
                    if length == 3 and source_unit == 2:
                        units = [1]*n  # The 30-cell canvas imposes this fit constraint.
                    else:
                        units = [int(v) for v in rng.integers(1, 3, size=n)]
                        if split == "test":
                            units[:2] = [1, 2]
                    flags = [tuple(map(int, rng.integers(0, 2, size=2))) for _ in range(n)]
                    args = (length, source_unit, units, flags, "free", False)
                pair = _scene(rng, *args)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            data = _decode(inp)
            expected, stats = _render(data)
            assert np.array_equal(out, expected), "incorrect shells, orientation, scale, or arrival merge"
            assert np.array_equal(out, _pull(data)), "forward and pixel-query constructions disagree"
            assert np.array_equal(inp[inp != data["background"]], out[inp != data["background"]])
            assert all(n > 0 for n in stats["unique_fronts"]), "every beacon must have visible influence"
            assert not np.array_equal(inp, out)
            (training if split == "train" else testing).append((data, stats))
    assert max(d["length"] for d, _ in training) < min(d["length"] for d, _ in testing)
    flags, ratios = set(), set()
    for d, _ in training:
        ref = d["cores"][d["reference"]]
        for c in d["cores"]:
            flags.add((ref[3] ^ c[3], ref[4] ^ c[4]))
            ratios.add((c[2], ref[2]))
    assert flags == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert (1, 2) in ratios and (2, 1) in ratios
    assert any(s["overlap_cells"] == 0 and s["clipped_copies"] == 0 for _, s in training)
    assert any(s["conflict_cells"] and s["agreeing_tie_cells"] and s["unequal_overlap"]
               and s["clipped_copies"] for _, s in training)
    # Each named shortcut is contradicted by actual returned demonstrations.
    alternatives = ({"complete": False}, {"reflect": False}, {"scale": False},
                    {"reverse": True}, {"vertical_corners": False}, {"order": True},
                    {"pixel_priority": True})
    for alternative in alternatives:
        assert any(not np.array_equal(_pull(d, **alternative), _pull(d)) for d, _ in training), alternative


def family_features(pair):
    """Small input-decoded measurements; contains no duplicated grid payloads."""
    d = _decode(pair["input"])
    _, s = _render(d)
    ref = d["cores"][d["reference"]]
    normalized = {}
    code = tuple(normalized.setdefault(int(v), len(normalized)) for v in d["codes"].flat)
    return {"shape": list(d["grid"].shape), "layers": d["length"],
            "beacons": len(d["cores"]), "source_unit": ref[2],
            "target_units": [c[2] for i, c in enumerate(d["cores"]) if i != d["reference"]],
            "source_key": list(ref[3:]), "code_structure": list(code),
            "palette_roles": [d["background"], d["body"], d["key"]], **s}


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, payload = _key(pair[role])
                h.update(bytes(shape))
                h.update(payload)
    return h.hexdigest()


def self_test():
    """42 seeds, count boundaries, ablations, corruption, and descriptive diversity.

    Streaming: one puzzle at a time, <=14 pairs, no files or retained grid suite.
    """
    signatures, shapes, structures, roles, corners, lengths, counts = (set() for _ in range(7))
    conflicts = clipped = overlap = agreeing = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        assert _digest(puzzle) == _digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = family_features(pair)
                shapes.add(tuple(f["shape"]))
                structures.add(tuple(f["code_structure"]))
                roles.add(tuple(f["palette_roles"]))
                corners.add(tuple(f["source_key"]))
                lengths.add(f["layers"])
                counts.add(f["beacons"])
                conflicts += f["conflict_cells"] > 0
                agreeing += f["agreeing_tie_cells"] > 0
                clipped += f["clipped_copies"] > 0
                overlap += f["overlap_cells"] > 0
                pairs += 1
    assert len(signatures) == 42 and len(shapes) >= 25
    assert len(structures) >= 45 and len(roles) >= 100
    assert corners == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert lengths == {1, 2, 3} and counts == {3, 4, 5}
    assert conflicts >= 42 and clipped >= 42 and overlap >= 42 and agreeing >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(1000+a*10+b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # The input-only rule extends beyond the sampler's usual units/layer counts.
    for length, source_unit, units in ((1, 3, [1, 2]), (4, 1, [1, 1])):
        rng = np.random.default_rng(921+length)
        for _ in range(12):
            pair = _scene(rng, length, source_unit, units, [(0, 1), (1, 0)], "free")
            if pair is not None:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                break
        else:
            raise AssertionError("extended-unit/layer fixture exhausted")
    # Agreeing fronts retain their shared color; disagreeing fronts erase to bg.
    t = np.array([[[2]], [[2]], [[3]]])
    c = np.array([[[4]], [[4]], [[9]]])
    assert _combine(t, c, 0)[0][0, 0] == 4
    t[2, 0, 0] = 2
    assert _combine(t, c, 0)[0][0, 0] == 0
    p = generate_puzzle(17, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "normalized_codes": len(structures), "role_assignments": len(roles),
            "conflict_pairs": conflicts, "overlap_pairs": overlap, "clipped_pairs": clipped,
            "agreeing_tie_pairs": agreeing,
            "independent_solver": False, "blind_prediction": False}
