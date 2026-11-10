"""Registered echo stamps: the same offsets act on tile and pixel addresses.

Only NumPy and the standard library are used. Generation and tests do no I/O.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations

import numpy as np


_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and bool(np.all((0 <= a) & (a <= 9))),
             "grids must contain integer ARC colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _shape_key(offsets):
    lo_r = min(r for r, c in offsets)
    lo_c = min(c for r, c in offsets)
    return tuple(sorted((r - lo_r, c - lo_c) for r, c in offsets))


def _asymmetric(offsets):
    key = _shape_key(offsets)
    variants = []
    points = tuple(offsets)
    for _ in range(4):
        variants.append(_shape_key(points))
        variants.append(_shape_key(tuple((r, -c) for r, c in points)))
        points = tuple((c, -r) for r, c in points)
    return sum(k == key for k in variants) == 1


def _runs(indices):
    groups = []
    for i in map(int, indices):
        if groups and groups[-1][1] + 1 == i:
            groups[-1] = (groups[-1][0], i)
        else:
            groups.append((i, i))
    return groups


def _geometry(grid):
    divider = int(grid[0, 0])
    rr = _runs(np.flatnonzero(np.all(grid == divider, axis=1)))
    cc = _runs(np.flatnonzero(np.all(grid == divider, axis=0)))
    _require(len(rr) >= 3 and len(cc) >= 3, "missing tile lattice")
    _require(rr[0][0] == cc[0][0] == 0 and rr[-1][1] == grid.shape[0] - 1
             and cc[-1][1] == grid.shape[1] - 1, "lattice must have a complete outer border")
    thickness = {b - a + 1 for a, b in rr + cc}
    hh = {rr[i + 1][0] - rr[i][1] - 1 for i in range(len(rr) - 1)}
    ww = {cc[i + 1][0] - cc[i][1] - 1 for i in range(len(cc) - 1)}
    _require(len(thickness) == len(hh) == len(ww) == 1, "irregular lattice")
    t, h, w = thickness.pop(), hh.pop(), ww.pop()
    _require(h > 0 and w > 0, "empty tile")
    return (h, w, t, len(rr) - 1, len(cc) - 1), divider


def _tile(grid, geometry, r, c):
    h, w, t, _, _ = geometry
    y, x = t + r * (h + t), t + c * (w + t)
    return grid[y:y + h, x:x + w]


def _decode(input_grid):
    """All roles, offsets and four-coordinate anchors come from this input."""
    grid = _grid(input_grid)
    geometry, divider = _geometry(grid)
    h, w, t, rows, cols = geometry
    tiles = {(r, c): _tile(grid, geometry, r, c)
             for r in range(rows) for c in range(cols)}
    values = np.concatenate([a.ravel() for a in tiles.values()])
    _require(divider not in values, "divider color occurs inside a tile")
    counts = np.bincount(values, minlength=10)
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous tile background")
    background = int(counts.argmax())
    candidates = [(p, a) for p, a in tiles.items()
                  if len(set(map(int, a.flat)) - {background}) == 3]
    _require(len(candidates) == 1, "exactly one three-role instruction tile is required")
    instruction, tile = candidates[0]
    frequencies = Counter(int(v) for v in tile.flat if int(v) != background)
    neutral_candidates = [c for c, n in frequencies.items() if n >= 3]
    _require(len(neutral_candidates) == 1 and sorted(frequencies.values())[:2] == [1, 1],
             "instruction needs one mask and two single-cell labels")
    neutral = neutral_candidates[0]
    labels = set(frequencies) - {neutral}
    external = set()
    for p, a in tiles.items():
        if p != instruction:
            colors = set(map(int, a.flat)) - {background}
            _require(len(colors) <= 1, "ordinary tiles contain only source dots")
            external |= colors
    _require(len(external) == 1 and external <= labels, "source label must match all external dots")
    source = external.pop()
    target = (labels - {source}).pop()
    sp = tuple(map(int, np.argwhere(tile == source)[0]))
    tp = tuple(map(int, np.argwhere(tile == target)[0]))
    direction = (tp[0] - sp[0], tp[1] - sp[1])
    _require(direction in _DIRECTIONS, "labels must be adjacent along a cardinal direction")
    q = (tp[0] + direction[0], tp[1] + direction[1])
    _require(0 <= q[0] < h and 0 <= q[1] < w and int(tile[q]) in (background, neutral),
             "registration is the cell beyond the target label")
    mask = tuple(tuple(map(int, p)) for p in np.argwhere(tile == neutral))
    offsets = tuple((r - q[0], c - q[1]) for r, c in mask)
    anchors = tuple((r, c, int(u), int(v)) for (r, c), a in tiles.items()
                    if (r, c) != instruction for u, v in np.argwhere(a == source))
    _require(bool(anchors), "no external source dots")
    return {"grid": grid, "geometry": geometry, "divider": divider,
            "background": background, "neutral": neutral, "source": source,
            "target": target, "instruction": instruction, "q": q,
            "direction": direction, "mask": mask, "offsets": offsets, "anchors": anchors}


def _addresses(info, *, macro=True, registered=True, fixed_phase=False):
    """Paint visits in (tile row, tile column, pixel row, pixel column) space."""
    h, w, _, rows, cols = info["geometry"]
    offsets = info["offsets"]
    inner = offsets if registered else info["mask"]
    outer = offsets if macro else ((0, 0),)
    visits = Counter()
    macro_clip = fine_clip = False
    for r, c, u, v in info["anchors"]:
        if fixed_phase:
            u, v = info["q"]
        for dr, dc in outer:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < rows and 0 <= cc < cols):
                macro_clip = True
                continue
            for du, dv in inner:
                uu, vv = u + du, v + dv
                if 0 <= uu < h and 0 <= vv < w:
                    visits[rr, cc, uu, vv] += 1
                else:
                    fine_clip = True
    return visits, macro_clip, fine_clip


def _apply(info, **options):
    output = info["grid"].copy()
    geometry = info["geometry"]
    _tile(output, geometry, *info["instruction"])[:] = info["background"]
    for r, c, u, v in info["anchors"]:
        _tile(output, geometry, r, c)[u, v] = info["background"]
    visits, _, _ = _addresses(info, **options)
    for r, c, u, v in visits:
        _tile(output, geometry, r, c)[u, v] = info["target"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase instructions/dots and union simultaneous registered echo stamps."""
    return _apply(_decode(input_grid))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _full_positions(height, width, offsets):
    return [(r, c) for r in range(height) for c in range(width)
            if all(0 <= r + dr < height and 0 <= c + dc < width for dr, dc in offsets)]


def _instruction(rng, geometry, kind, forbidden):
    h, w, _, rows, cols = geometry

    def admissible(q, cells):
        offsets = tuple(sorted((r - q[0], c - q[1]) for r, c in cells))
        if _shape_key(offsets) in forbidden or not _full_positions(rows, cols, offsets):
            return None
        if kind == "occupied" and ((0, 0) not in offsets or not _asymmetric(offsets)
                                   or min(offsets) == (0, 0)
                                   or (sum(r for r, c in offsets) == 0 and sum(c for r, c in offsets) == 0)):
            return None
        if kind == "hole" and (0, 0) in offsets:
            return None
        return offsets

    for _ in range(256):
        direction = _pick(rng, _DIRECTIONS)
        dr, dc = direction
        choices = [(r, c) for r in range(h) for c in range(w)
                   if 0 <= r - 2 * dr < h and 0 <= c - 2 * dc < w]
        q = _pick(rng, choices)
        labels = {(q[0] - dr, q[1] - dc), (q[0] - 2 * dr, q[1] - 2 * dc)}
        extent = 3 if kind == "occupied" else 4
        bh = int(rng.integers(2, min(extent, rows, h) + 1))
        bw = int(rng.integers(2, min(extent, cols, w) + 1))
        top, left = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
        if not (top - 1 <= q[0] <= top + bh and left - 1 <= q[1] <= left + bw):
            continue
        options = [(r, c) for r in range(top, top + bh) for c in range(left, left + bw)
                   if (r, c) not in labels and (kind != "hole" or (r, c) != q)]
        if len(options) < 3 or (kind == "occupied" and q not in options):
            continue
        count = int(rng.integers(3, min(5 if kind == "occupied" else 7, len(options)) + 1))
        cells = {q} if kind == "occupied" else set()
        pool = [p for p in options if p not in cells]
        cells.update(pool[int(i)] for i in rng.choice(len(pool), count - len(cells), replace=False))
        offsets = admissible(q, cells)
        if offsets is not None:
            return q, direction, offsets
    # Finite constructive fallback: a 3x3 window fits every sampled lattice.
    # Its 7/8 available cells offer many more designs than the <=10 exclusions.
    q, direction = (2, 2), (0, 1)
    options = [(r, c) for r in range(1, 4) for c in range(1, 4)
               if (r, c) != (2, 1) and (kind != "hole" or (r, c) != q)]
    for count in range(3, min(5 if kind == "occupied" else 8, len(options)) + 1):
        for cells in combinations(options, count):
            offsets = admissible(q, cells)
            if offsets is not None:
                return q, direction, offsets
    raise RuntimeError("no registered stencil after bounded construction")


def _construct(geometry, palette, instruction, q, direction, offsets, anchors):
    """Ground truth from sampled addresses, separately from input decoding."""
    h, w, t, rows, cols = geometry
    divider, bg, neutral, source, target = palette
    grid = np.full((rows * (h + t) + t, cols * (w + t) + t), divider, dtype=np.int8)
    output = grid.copy()
    for r in range(rows):
        for c in range(cols):
            _tile(grid, geometry, r, c)[:] = bg
            _tile(output, geometry, r, c)[:] = bg
    tile = _tile(grid, geometry, *instruction)
    for dr, dc in offsets:
        tile[q[0] + dr, q[1] + dc] = neutral
    dr, dc = direction
    tile[q[0] - 2 * dr, q[1] - 2 * dc] = source
    tile[q[0] - dr, q[1] - dc] = target
    for r, c, u, v in anchors:
        _tile(grid, geometry, r, c)[u, v] = source
        # Cartesian product of the stencil with itself, with two independent
        # bounds checks. Flattening this into ordinary pixel translation is wrong.
        for a, b in offsets:
            if 0 <= r + a < rows and 0 <= c + b < cols:
                destination = _tile(output, geometry, r + a, c + b)
                for x, y in offsets:
                    if 0 <= u + x < h and 0 <= v + y < w:
                        destination[u + x, v + y] = target
    return {"input": grid, "output": output}


def _make_pair(rng, palette, regime, forbidden):
    h, w = int(rng.integers(4, 7)), int(rng.integers(4, 7))
    t = int(_pick(rng, (1, 1, 2)))
    rows = int(rng.integers(3, (30 - t) // (h + t) + 1))
    cols = int(rng.integers(3, (30 - t) // (w + t) + 1))
    geometry = h, w, t, rows, cols
    kind = "occupied" if regime == "intro" else "hole" if regime == "phase" else _pick(rng, ("any", "any", "hole"))
    q, direction, offsets = _instruction(rng, geometry, kind, forbidden)
    full_coarse = _full_positions(rows, cols, offsets)
    all_coarse = [(r, c) for r in range(rows) for c in range(cols)]
    # Preserve at least one complete echo location outside the instruction tile.
    instruction = _pick(rng, [p for p in all_coarse if len(full_coarse) > 1 or p != full_coarse[0]])
    usable = [p for p in all_coarse if p != instruction]
    full_coarse = [p for p in full_coarse if p != instruction]
    full_fine = _full_positions(h, w, offsets)
    clipped_coarse = [p for p in usable if p not in full_coarse]
    clipped_fine = [(u, v) for u in range(h) for v in range(w) if (u, v) not in full_fine]
    anchors = []
    if regime == "intro":
        # One complete echo makes the shared shape at both scales explicit.
        anchors.append((*_pick(rng, full_coarse), *q))
    elif regime == "phase":
        p = _pick(rng, full_coarse)
        anchors = [(*p, *_pick(rng, full_fine)), (*p, *_pick(rng, clipped_fine)),
                   (*_pick(rng, clipped_coarse), *_pick(rng, full_fine))]
    elif regime == "test":
        # The shift is a new visible design, not compulsory boundary complexity.
        mode = _pick(rng, ("both", "both", "interior", "fine", "coarse", "free"))
        if mode == "free":
            choices = [(*p, u, v) for p in usable for u in range(h) for v in range(w)]
            anchors = [choices[int(i)] for i in rng.choice(len(choices), int(rng.integers(1, 7)), replace=False)]
        else:
            p = _pick(rng, full_coarse)
            anchors = [(*p, *_pick(rng, full_fine))]
            fine_pool = clipped_fine if mode in ("both", "fine") else full_fine
            coarse_pool = clipped_coarse if mode in ("both", "coarse") else full_coarse
            for a in ((*p, *_pick(rng, fine_pool)),
                      (*_pick(rng, coarse_pool), *_pick(rng, full_fine))):
                if a not in anchors:
                    anchors.append(a)
    else:
        count = int(rng.integers(1, 7))
        choices = [(*p, u, v) for p in usable for u in range(h) for v in range(w)]
        anchors = [choices[int(i)] for i in rng.choice(len(choices), count, replace=False)]
    return _construct(geometry, palette, instruction, q, direction, offsets, anchors)


def _features(info):
    visits, macro_clip, fine_clip = _addresses(info)
    anchor_tiles = {a[:2] for a in info["anchors"]}
    return {"macro_clip": macro_clip, "fine_clip": fine_clip,
            "overlap": sum(n > 1 for n in visits.values()),
            "coarse_witness": any(p[:2] not in anchor_tiles for p in visits),
            "phase_witness": not np.array_equal(_apply(info), _apply(info, fixed_phase=True)),
            "registration_witness": not np.array_equal(_apply(info), _apply(info, registered=False)),
            "hole": (0, 0) not in info["offsets"],
            "phases": len({a[2:] for a in info["anchors"]}),
            "paired_tile": len(anchor_tiles) < len(info["anchors"])}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base = tuple(map(int, rng.choice(10, 5, replace=False)))
    puzzle = {"train": [], "test": []}
    seen, train_shapes = set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            regime = ("intro" if index == 0 else "phase" if index == 1 else "free") if split == "train" else "test"
            for _ in range(96):
                if split == "train" and index == 0:
                    palette = base
                elif split == "train" and index == 1:
                    palette = (*base[:4], int(rng.choice([c for c in range(10) if c not in base])))
                else:
                    palette = base if rng.random() < .2 else tuple(map(int, rng.choice(10, 5, replace=False)))
                forbidden = train_shapes if split == "test" or regime == "phase" else set()
                pair = _make_pair(rng, palette, regime, forbidden)
                info = _decode(pair["input"])
                f = _features(info)
                if _key(pair["input"]) in seen or not f["coarse_witness"] or not f["registration_witness"]:
                    continue
                if regime == "phase" and not (f["macro_clip"] and f["fine_clip"] and f["phase_witness"]):
                    continue
                _require(np.array_equal(_apply(info), pair["output"]), "sampled and decoded echoes disagree")
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if split == "train":
                    train_shapes.add(_shape_key(info["offsets"]))
                break
            else:
                raise RuntimeError("bounded distinct-pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert grid.shape == output.shape and _key(grid) not in seen
            seen.add(_key(grid))
            info = _decode(grid)
            expected = _apply(info)
            assert np.array_equal(output, expected), "incorrect registered two-scale stamp"
            assert not np.array_equal(grid, output)
            # Every divider is untouched, even when a micro stamp is clipped.
            divider = grid == info["divider"]
            assert np.array_equal(output[divider], grid[divider])
            assert set(map(int, np.unique(output))) <= {info["background"], info["divider"], info["target"]}
            f = _features(info)
            assert f["coarse_witness"] and f["registration_witness"]
            assert not np.array_equal(output, _apply(info, macro=False)), "B ablation must change the answer"
            records[split].append((info, f))
    first, second = (records["train"][i][0] for i in (0, 1))
    assert (0, 0) in first["offsets"] and _asymmetric(first["offsets"])
    assert (0, 0) not in second["offsets"]
    assert first["source"] == second["source"] and first["target"] != second["target"]
    assert any(f["phase_witness"] and f["paired_tile"] and f["macro_clip"] and f["fine_clip"]
               for _, f in records["train"])
    shapes = {_shape_key(info["offsets"]) for info, _ in records["train"]}
    assert len(shapes) >= 2
    for info, f in records["test"]:
        assert _shape_key(info["offsets"]) not in shapes, "tests must read a new visible stencil"
    return True


def _digest(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _manual_regressions():
    geometry = (4, 4, 1, 3, 3)
    palette = (9, 0, 8, 1, 2)
    offsets = ((-1, 0), (0, 0), (1, 1))
    pair = _construct(geometry, palette, (0, 0), (2, 2), (0, 1), offsets, [(1, 1, 2, 2)])
    expected = np.full((16, 16), 9, dtype=np.int8)
    for r in range(3):
        for c in range(3):
            _tile(expected, geometry, r, c)[:] = 0
    # Independently listed nine addresses: three small shapes in a big shape.
    for r, c in ((0, 1), (1, 1), (2, 2)):
        for u, v in ((1, 2), (2, 2), (3, 3)):
            _tile(expected, geometry, r, c)[u, v] = 2
    assert np.array_equal(solve(pair["input"]), expected)
    assert int(np.sum(expected == 2)) == 9
    # Hole registration, plus independent coarse and fine boundary clipping.
    pair = _construct(geometry, palette, (2, 2), (2, 2), (0, 1),
                      ((-1, -1), (0, 1), (1, 0)), [(0, 0, 0, 0)])
    expected[expected == 2] = 0
    for r, c in ((0, 1), (1, 0)):
        for u, v in ((0, 1), (1, 0)):
            _tile(expected, geometry, r, c)[u, v] = 2
    assert np.array_equal(solve(pair["input"]), expected)
    assert int(np.sum(expected == 2)) == 4
    # Adjacent source dots each trigger an echo, even as one connected object.
    pair = _construct(geometry, palette, (0, 0), (2, 2), (0, 1), offsets,
                      [(1, 1, 2, 1), (1, 1, 2, 2)])
    assert int(np.sum(solve(pair["input"]) == 2)) == 18


def self_test():
    """Stream 42 seeds; retain only small signatures/counters, never traces."""
    _manual_regressions()
    puzzles, geometries, masks, phases, directions, roles = set(), set(), set(), set(), set(), set()
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = pair["input"].copy()
                predicted = solve(before)
                assert np.array_equal(before, pair["input"]) and np.array_equal(predicted, pair["output"])
                info = _decode(before)
                f = _features(info)
                geometries.add(info["geometry"])
                masks.add(info["offsets"])
                directions.add(info["direction"])
                phases.add(tuple(sorted({a[2:] for a in info["anchors"]})))
                roles.add((info["divider"], info["background"], info["neutral"], info["source"], info["target"]))
                for name in ("hole", "macro_clip", "fine_clip", "paired_tile", "phase_witness"):
                    totals[name] += bool(f[name])
                totals["overlap"] += f["overlap"] > 0
                totals["rectangular"] += info["geometry"][0] != info["geometry"][1]
                totals["instruction_painted"] += bool(np.any(_tile(predicted, info["geometry"], *info["instruction"]) == info["target"]))
        # Palette equivariance and transposition are family properties.
        pair = puzzle["test"][0]
        permutation = np.random.default_rng(seed + 1000).permutation(10)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
        assert np.array_equal(solve(pair["input"].T), pair["output"].T)
    assert len(puzzles) == 42 and len(geometries) >= 40 and len(masks) >= 100
    assert len(phases) >= 80 and len(roles) >= 80 and len(directions) == 4
    assert all(totals[k] >= 20 for k in totals), totals
    assert _digest(generate_puzzle(781, 3, 2)) == _digest(generate_puzzle(781, 3, 2))
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(902, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(9, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(1, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged divider accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "geometries": len(geometries),
            "registered_masks": len(masks), "phase_sets": len(phases), "role_assignments": len(roles),
            "features": dict(totals)}
