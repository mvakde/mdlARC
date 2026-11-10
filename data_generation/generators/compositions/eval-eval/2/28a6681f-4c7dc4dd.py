"""Calibrated communicating vessels: experiments determine hydraulic topology.

Only the standard library and NumPy are used. No files, examples, random state,
or generation metadata are consulted by solve(). Self-tests retain only bounded
sets of hashes and scalar measurements; execution traces are not stored.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or a.dtype.kind not in "iu" or
            not all(1 <= n <= 30 for n in a.shape) or
            not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, points = [start], []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        part = np.zeros(mask.shape, dtype=bool)
        rr, cc = zip(*points)
        part[rr, cc] = True
        result.append(part)
    return result


def _box(mask):
    rr, cc = np.nonzero(mask)
    return int(rr.min()), int(cc.min()), int(rr.max()+1), int(cc.max()+1)


def _crop(a, box):
    r, c, bottom, right = box
    return a[r:bottom, c:right].copy()


def _runs(indices):
    result = []
    for c in indices:
        c = int(c)
        if result and result[-1][-1] + 1 == c:
            result[-1].append(c)
        else:
            result.append([c])
    return result


def _calibration(first, second, background, wall):
    """Read an unordered experimental pair, including its unchanged hardware."""
    if first.shape != second.shape:
        raise ValueError("calibration shapes differ")
    changed = first != second
    colors = set(map(int, np.unique(np.r_[first[changed], second[changed]])))
    if background not in colors or len(colors) != 2:
        raise ValueError("calibration must move exactly one liquid color")
    water = next(iter(colors - {background}))
    if water == wall:
        raise ValueError("liquid cannot be the frame")
    points = [np.argwhere(x == water) for x in (first, second)]
    if not len(points[0]) or len(points[0]) != len(points[1]):
        raise ValueError("calibration must conserve liquid")
    if points[0][:, 0].sum() == points[1][:, 0].sum():
        raise ValueError("calibration direction is ambiguous")
    before, after = ((first, second) if points[0][:, 0].sum() < points[1][:, 0].sum()
                     else (second, first))
    fixed = (before != background) & (before != water)
    if not np.array_equal(before[fixed], after[fixed]):
        raise ValueError("calibration hardware changed")
    materials = set(map(int, np.unique(before))) - {background, water, wall}
    if not 2 <= len(materials) <= 3:
        raise ValueError("expected two or three tested membrane materials")
    h, w = before.shape
    tubes = _runs(np.flatnonzero(before[1] != wall))
    if len(tubes) != len(materials):
        raise ValueError("each material needs one separate test tube")
    expected_before = np.full_like(before, wall)
    expected_after = np.full_like(after, wall)
    open_materials, tested = set(), set()
    for columns in tubes:
        cells = before[1:h-1, columns]
        gate_colors = set(map(int, np.unique(cells))) - {background, water}
        if len(gate_colors) != 1:
            raise ValueError("a test tube needs one material")
        material = next(iter(gate_colors))
        locations = np.argwhere(before[:, columns] == material)
        rows = np.unique(locations[:, 0])
        if len(rows) != 1 or not 3 <= rows[0] <= h-3 or material in tested:
            raise ValueError("invalid calibration gate")
        gate = int(rows[0])
        expected_before[1:h-1, columns] = background
        expected_before[gate, columns] = material
        expected_before[1, columns] = water
        expected_after[1:h-1, columns] = background
        expected_after[gate, columns] = material
        water_rows = np.unique(np.argwhere(after[:, columns] == water)[:, 0])
        if len(water_rows) != 1 or water_rows[0] not in (gate-1, h-2):
            raise ValueError("unreadable membrane experiment")
        destination = int(water_rows[0])
        if destination == h-2:
            open_materials.add(material)
        expected_after[destination, columns] = water
        tested.add(material)
    if (not np.array_equal(before, expected_before) or
            not np.array_equal(after, expected_after) or
            not open_materials or open_materials == materials):
        raise ValueError("calibration must clearly demonstrate passing and blocking")
    return dict(background=background, water=water, wall=wall,
                materials=materials, open=open_materials, before=before, after=after)


def _scene(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    if not (np.all(grid[0] == background) and np.all(grid[-1] == background) and
            np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)):
        raise ValueError("the apparatus needs a clear background margin")
    candidates = []
    for wall in map(int, np.unique(grid)):
        if wall == background:
            continue
        parts = _components(grid == wall)
        if len(parts) != 3:
            continue
        boxes = [_box(p) for p in parts]
        panels = [_crop(grid, b) for b in boxes]
        if not all(min(x.shape) >= 5 and np.all(x[0] == wall) and
                   np.all(x[-1] == wall) and np.all(x[:, 0] == wall) and
                   np.all(x[:, -1] == wall) for x in panels):
            continue
        order = sorted(range(3), key=lambda i: panels[i].shape[0])
        a, b, target = [panels[i] for i in order]
        if a.shape != b.shape or target.shape[0] <= a.shape[0]:
            continue
        try:
            result = _calibration(a, b, background, wall)
        except ValueError:
            continue
        if not set(map(int, np.unique(target))) <= {
                background, result["water"], wall, *result["materials"]}:
            continue
        result.update(target=target, target_box=boxes[order[2]],
                      boxes=boxes, grid=grid)
        candidates.append(result)
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous three-apparatus experiment")
    return candidates[0]


def _equilibrate(target, background, water, open_materials):
    """Membranes conduct but occupy no liquid cells; fill complete levels only.

    Family constraints require all open ports in a wet connected vessel to be
    submerged in its final state. Thus component equalization cannot teleport
    water over a dry high sill. Arbitrary terrain outside that domain is rejected
    by the family validator rather than claimed to be a general fluid simulator.
    """
    free = (target == background) | (target == water)
    conductive = free | np.isin(target, list(open_materials))
    out = target.copy()
    out[target == water] = background
    for part in _components(conductive):
        remaining = int(np.count_nonzero(part & (target == water)))
        for row in range(target.shape[0]-1, -1, -1):
            slots = part[row] & free[row]
            capacity = int(slots.sum())
            if not remaining:
                break
            if remaining < capacity:
                raise ValueError("volume would leave an ambiguous partial level")
            out[row, slots] = water
            remaining -= capacity
        if remaining:
            raise ValueError("liquid exceeds vessel capacity")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer materials from this input, equalize, and return the large interior."""
    scene = _scene(input_grid)
    settled = _equilibrate(scene["target"], scene["background"], scene["water"], scene["open"])
    return settled[1:-1, 1:-1].copy()


def _make_calibration(rng, background, water, wall, materials, open_materials):
    height = int(rng.integers(6, 9))
    widths = rng.integers(1, 3, len(materials))
    width = 1 + sum(int(x)+1 for x in widths)
    before = np.full((height, width), wall, dtype=np.int8)
    after = before.copy()
    column = 1
    for material, span in zip(materials, widths):
        columns = slice(column, column+int(span))
        gate = int(rng.integers(3, height-2))
        before[1:-1, columns] = after[1:-1, columns] = background
        before[gate, columns] = after[gate, columns] = material
        before[1, columns] = water
        after[height-2 if material in open_materials else gate-1, columns] = water
        column += int(span)+1
    return before, after


def _other_calibration(before, water, background, wall, open_materials):
    after = before.copy()
    after[after == water] = background
    for columns in _runs(np.flatnonzero(before[1] != wall)):
        gate = np.argwhere((before[:, columns] != background) &
                           (before[:, columns] != water) & (before[:, columns] != wall))[0]
        row = int(gate[0])
        material = int(before[row, columns[0]])
        after[before.shape[0]-2 if material in open_materials else row-1, columns] = water
    return after


def _profile(rng, height, width, flat=False):
    base = int(rng.integers(height-3, height))
    mode = 0 if flat else int(rng.integers(4))
    if mode == 0:
        return np.full(width, base, dtype=int)
    if mode == 1:
        values = base + np.arange(width) * int(rng.choice([-1, 1]))
    elif mode == 2:
        values = base + rng.integers(-2, 2, width)
    else:
        values = base - np.abs(np.arange(width)-(width-1)/2).astype(int)
    return np.clip(values, height-4, height-1)


def _query_hardware(rng, background, wall, materials, bay_count, twin):
    height = int(rng.integers(10, 16))
    widths = ([int(rng.integers(2, 5))]*2 if twin else
              list(map(int, rng.integers(1, 5, bay_count))))
    thicknesses = list(map(int, rng.integers(1, 3, bay_count-1)))
    while 2 + sum(widths) + sum(thicknesses) > 26:
        widths[int(np.argmax(widths))] -= 1
    width = 2 + sum(widths) + sum(thicknesses)
    target = np.full((height, width), wall, dtype=np.int8)
    bays, floors = [], []
    column = 1
    for i, span in enumerate(widths):
        profile = _profile(rng, height, span, twin)
        columns = list(range(column, column+span))
        bay = np.zeros(target.shape, dtype=bool)
        for c, floor in zip(columns, profile):
            target[1:int(floor), c] = background
            bay[1:int(floor), c] = True
        bays.append(bay)
        floors.append(profile)
        column += span + (thicknesses[i] if i < bay_count-1 else 0)
    if twin:
        gate_materials = [materials[0]]
    else:
        permutation = list(rng.permutation(materials))
        gate_materials = [int(permutation[i % len(materials)]) for i in range(bay_count-1)]
        rng.shuffle(gate_materials)
    column = 1 + widths[0]
    for i, (thickness, material) in enumerate(zip(thicknesses, gate_materials)):
        bottom = min(int(floors[i][-1]), int(floors[i+1][0]))-1
        gate_height = int(rng.integers(1, 3))
        target[bottom-gate_height+1:bottom+1, column:column+thickness] = material
        column += thickness+widths[i+1]
    return target, bays, floors


def _choose_source(rng, eligible, final, volume, style):
    points = np.argwhere(eligible)
    anchor = points[int(rng.integers(len(points)))]
    jitter = rng.random(len(points))
    dr, dc = abs(points[:, 0]-anchor[0]), abs(points[:, 1]-anchor[1])
    if style == 0:  # scattered
        score = jitter
    elif style == 1:  # upper band
        score = points[:, 0] + jitter
    elif style == 2:  # compact patch
        score = dr + dc + jitter
    elif style == 3:  # vertical filament
        score = 3*dc + .2*dr + jitter
    else:  # mostly settled, with a visible displacement
        score = (~final[points[:, 0], points[:, 1]])*10 + jitter
    selected = points[np.argsort(score)[:volume]]
    source = np.zeros(eligible.shape, dtype=bool)
    source[selected[:, 0], selected[:, 1]] = True
    if np.array_equal(source, final):
        old = tuple(map(int, np.argwhere(source)[0]))
        new = tuple(map(int, np.argwhere(eligible & ~final)[0]))
        source[old], source[new] = False, True
    return source


def _make_query(rng, background, water, wall, materials, open_materials, bay_count, twin=False):
    target, bays, floors = _query_hardware(rng, background, wall, materials, bay_count, twin)
    free = target == background
    rows = np.arange(target.shape[0])[:, None]
    source, final = np.zeros_like(free), np.zeros_like(free)
    if twin:
        level = int(rng.integers(3, min(int(f.min()) for f in floors)-1))
        final = free & (rows >= level)
        for i, bay in enumerate(bays):
            volume = int((final & bay).sum()) + (1 if i == 0 else -1)*int(bay[1].sum())
            local_final = bay & (rows >= (level-1 if i == 0 else level+1))
            source |= _choose_source(rng, bay, local_final, volume, int(rng.integers(5)))
        closed = target.copy()
        for i, bay in enumerate(bays):
            closed[bay & (rows >= (level-1 if i == 0 else level+1))] = water
    else:
        conductive = free | np.isin(target, list(open_materials))
        for part in _components(conductive):
            eligible = part & free
            member_bays = [b for b in bays if np.any(b & eligible)]
            minimum_floor = min(int(np.flatnonzero(~free[:, c])[1:].min())
                                for c in np.flatnonzero(eligible[1]))
            # All floors are grounded. A common surface above every floor
            # also submerges every communicating port, even on stepped floors.
            level = int(rng.integers(2, minimum_floor-1))
            local_final = eligible & (rows >= level)
            local_source = _choose_source(rng, eligible, local_final,
                                          int(local_final.sum()), int(rng.integers(5)))
            if len(member_bays) > 1 and all(int((local_source & b).sum()) ==
                                          int((local_final & b).sum()) for b in member_bays):
                old = tuple(map(int, np.argwhere(local_source & member_bays[0])[0]))
                new = tuple(map(int, np.argwhere(member_bays[1] & ~local_source)[0]))
                local_source[old], local_source[new] = False, True
            source |= local_source
            final |= local_final
        closed = None
    out = target.copy()
    target[source] = water
    out[final] = water
    if not np.array_equal(_equilibrate(target, background, water, open_materials), out):
        raise AssertionError("constructed equilibrium disagrees with conservation")
    return target, out, closed


def _pack(rng, before, after, target, background):
    """Place apparatuses on any feasible side, with independent alignments."""
    ch, cw = before.shape
    qh, qw = target.shape
    small_gap, big_gap = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    options = []
    if max(2*cw+small_gap, qw)+2 <= 30 and ch+big_gap+qh+2 <= 30:
        options.append(0)
    if cw+big_gap+qw+2 <= 30 and max(2*ch+small_gap, qh)+2 <= 30:
        options.append(1)
    axis = int(rng.choice(options))
    flip = bool(rng.integers(2))
    if axis == 0:
        min_h, min_w = ch+big_gap+qh+2, max(2*cw+small_gap, qw)+2
    else:
        min_h, min_w = max(2*ch+small_gap, qh)+2, cw+big_gap+qw+2
    height = int(rng.integers(min_h, min(30, min_h+3)+1))
    width = int(rng.integers(min_w, min(30, min_w+3)+1))
    if axis == 0:
        top = int(rng.integers(1, height-ch-big_gap-qh))
        cr, qr = ((top, top+ch+big_gap) if not flip else (top+qh+big_gap, top))
        left = int(rng.integers(1, width-2*cw-small_gap))
        positions = [(cr, left), (cr, left+cw+small_gap)]
        query_pos = (qr, int(rng.integers(1, width-qw)))
    else:
        left = int(rng.integers(1, width-cw-big_gap-qw))
        cc, qc = ((left, left+cw+big_gap) if not flip else (left+qw+big_gap, left))
        top = int(rng.integers(1, height-2*ch-small_gap))
        positions = [(top, cc), (top+ch+small_gap, cc)]
        query_pos = (int(rng.integers(1, height-qh)), qc)
    if rng.integers(2):
        positions.reverse()
    grid = np.full((height, width), background, dtype=np.int8)
    for panel, (r, c) in zip((before, after, target), positions+[query_pos]):
        grid[r:r+panel.shape[0], c:c+panel.shape[1]] = panel
    r, c = positions[1]
    return grid, (r, c, r+ch, c+cw)


def _palette(rng, base=None):
    colors = rng.permutation(10) if base is None else base
    background, water, wall = map(int, colors[:3])
    materials = list(map(int, colors[3:3+int(rng.integers(2, 4))]))
    return background, water, wall, materials


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test) or
            not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    base = rng.permutation(10)
    variation = float(rng.choice([0., .5, 1.]))
    background, water, wall, materials = _palette(rng, base)
    open_materials = {materials[0]}
    if len(materials) == 3 and rng.integers(2):
        open_materials.add(materials[2])
    before, after = _make_calibration(rng, background, water, wall, materials, open_materials)
    target, open_out, closed_out = _make_query(
        rng, background, water, wall, materials, open_materials, 2, twin=True)
    first_input, after_box = _pack(rng, before, after, target, background)
    other_open = set(materials)-open_materials
    second_input = first_input.copy()
    r, c, bottom, right = after_box
    second_input[r:bottom, c:right] = _other_calibration(before, water, background, wall, other_open)
    puzzle = {"train": [{"input": first_input, "output": open_out[1:-1, 1:-1].copy()},
                         {"input": second_input, "output": closed_out[1:-1, 1:-1].copy()}],
              "test": []}
    seen = {_key(first_input), _key(second_input)}
    for split, count in (("train", int(num_train)-2), ("test", int(num_test))):
        for _ in range(count):
            for attempt in range(32):
                palette = None if rng.random() < variation else base
                background, water, wall, materials = _palette(rng, palette)
                ordered = list(map(int, rng.permutation(materials)))
                open_materials = set(ordered[:int(rng.integers(1, len(materials)))])
                bay_count = 5 if split == "test" else int(rng.integers(len(materials)+1, 5))
                before, after = _make_calibration(rng, background, water, wall, materials, open_materials)
                target, out, _ = _make_query(rng, background, water, wall, materials, open_materials, bay_count)
                inp, _ = _pack(rng, before, after, target, background)
                if _key(inp) not in seen:
                    seen.add(_key(inp))
                    puzzle[split].append({"input": inp, "output": out[1:-1, 1:-1].copy()})
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _inspect(scene, output):
    target = scene["target"]
    bg, water, wall = scene["background"], scene["water"], scene["wall"]
    assert output.shape == (target.shape[0]-2, target.shape[1]-2)
    out = target.copy()
    out[1:-1, 1:-1] = output
    free = (target == bg) | (target == water)
    assert np.array_equal(out[~free], target[~free]), "hardware must remain unchanged"
    assert np.all(np.isin(out[free], [bg, water]))
    assert np.any(out != target), "source must visibly settle"
    assert int((out == water).sum()) == int((target == water).sum()) > 0
    assert np.argwhere(out == water)[:, 0].sum() > np.argwhere(target == water)[:, 0].sum()
    bays = _runs(np.flatnonzero(free[1]))
    assert 2 <= len(bays) <= 5
    floor_rows = []
    for columns in bays:
        for c in columns:
            floor = int(np.flatnonzero(target[1:, c] == wall)[0])+1
            assert free[1:floor, c].all() and np.all(target[floor:, c] == wall)
            floor_rows.append(floor)
    conductive = free | np.isin(target, list(scene["open"]))
    levels, communicating = [], 0
    for part in _components(conductive):
        mask = part & free
        wet = part & (out == water)
        assert wet.any(), "generated components are all supplied"
        assert int(wet.sum()) == int((part & (target == water)).sum())
        level = int(np.nonzero(wet)[0].min())
        assert np.array_equal(wet, mask & (np.arange(target.shape[0])[:, None] >= level))
        levels.append(level)
        member_bays = [cols for cols in bays if mask[1, cols].any()]
        if len(member_bays) > 1:
            communicating += 1
            assert any(int((target[:, cols] == water).sum()) != int((out[:, cols] == water).sum())
                       for cols in member_bays), "open ports must cause actual inter-bay transfer"
    gate_shapes = []
    used_materials = set()
    for material in scene["materials"]:
        for gate in _components(target == material):
            used_materials.add(material)
            r, c, bottom, right = _box(gate)
            assert gate[r:bottom, c:right].all()
            assert np.all(target[1:r, c:right] == wall)
            assert np.all(target[bottom:, c:right] == wall)
            assert free[r:bottom, c-1].all() and free[r:bottom, right].all()
            if material in scene["open"]:
                assert np.all(out[r:bottom, c-1] == water) and np.all(out[r:bottom, right] == water)
            gate_shapes.append((bottom-r, right-c))
    assert np.array_equal(_equilibrate(target, bg, water, scene["open"]), out)
    return dict(bays=len(bays), levels=tuple(levels), communicating=communicating,
                gates=tuple(gate_shapes), used=used_materials, floors=tuple(floor_rows),
                volume=int((target == water).sum()), shape=scene["grid"].shape,
                output_shape=output.shape, low_source=int(np.nonzero(target == water)[0].min()),
                source_components=len(_components(target == water)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, twin_scenes = set(), []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            scene = _scene(inp)
            stats = _inspect(scene, out)
            assert np.array_equal(solve(inp), out)
            if split == "train" and index < 2:
                twin_scenes.append(scene)
                assert stats["bays"] == 2
            else:
                assert stats["communicating"] >= 1
                assert stats["used"] & scene["open"]
                assert stats["used"] - scene["open"]
                assert stats["bays"] >= 3
                if split == "test":
                    assert stats["bays"] == 5
    first, second = twin_scenes
    assert np.array_equal(first["target"], second["target"])
    assert np.array_equal(first["before"], second["before"])
    assert first["open"] != second["open"]
    assert not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"])
    assert not np.array_equal(_equilibrate(first["target"], first["background"], first["water"], set()),
                              _equilibrate(first["target"], first["background"], first["water"], first["open"]))


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                grid = np.asarray(pair[name], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _diagnostic_test():
    # Hand-calculated eight-cell supply: six cells in the left bay and two
    # in the right. The low port either joins two two-cell-wide bays or leaves
    # depths three and one. The membrane itself never stores a water pixel.
    def digits(rows):
        return np.array([[int(x) for x in row] for row in rows], dtype=np.int8)

    inp = digits(("2222222", "2112112", "2112002", "2112002", "2002002",
                  "2002002", "2002002", "2003002", "2222222"))
    passing = digits(("2222222", "2002002", "2002002", "2002002", "2002002",
                      "2002002", "2112112", "2113112", "2222222"))
    blocking = digits(("2222222", "2002002", "2002002", "2002002", "2002002",
                       "2112002", "2112002", "2113112", "2222222"))
    assert np.array_equal(_equilibrate(inp, 0, 1, {3}), passing)
    assert np.array_equal(_equilibrate(inp, 0, 1, set()), blocking)
    puzzle = generate_puzzle(77, 2, 1)
    for pair in puzzle["train"]:
        scene = _scene(pair["input"])
        grid = pair["input"].copy()
        a, b = [box for box in scene["boxes"] if box != scene["target_box"]]
        ca, cb = _crop(grid, a), _crop(grid, b)
        for box, panel in ((a, cb), (b, ca)):
            r, c, bottom, right = box
            grid[r:bottom, c:right] = panel
        assert np.array_equal(solve(grid), pair["output"]), "reference order cannot select physics"


def self_test():
    """42 serial seeds, bounded adversarial checks, and all 30 legal counts.

    Maximum retained puzzle: 14 pairs, <=25,200 int8 grid bytes. No puzzle corpus,
    JSON payloads, files, traces, logs, workers, or retry artifacts are retained.
    """
    signatures, shapes, outputs, structures, palettes = set(), set(), set(), set(), set()
    bays, volumes, gate_shapes, floors, positions, sources, levels = set(), set(), set(), set(), set(), set(), set()
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed < 3:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _scene(pair["input"])
                stats = _inspect(scene, pair["output"])
                pair_count += 1
                shapes.add(stats["shape"])
                outputs.add(stats["output_shape"])
                bays.add(stats["bays"])
                volumes.add(stats["volume"])
                gate_shapes.update(stats["gates"])
                floors.add(stats["floors"])
                sources.add(stats["source_components"])
                levels.update(stats["levels"])
                palettes.add((scene["background"], scene["water"], scene["wall"]))
                positions.add(scene["target_box"][:2])
                normalized = np.zeros_like(scene["grid"])
                normalized[scene["grid"] == scene["water"]] = 1
                normalized[scene["grid"] == scene["wall"]] = 2
                for material in scene["materials"]:
                    normalized[scene["grid"] == material] = 3 if material in scene["open"] else 4
                structures.add(hashlib.sha256(normalized.tobytes()).hexdigest())
                original = pair["input"].copy()
                assert np.array_equal(solve(original.tolist()), pair["output"])
                assert np.array_equal(original, pair["input"])
                # A palette permutation must preserve the inferred material roles.
                if seed < 3:
                    permutation = np.array([7, 2, 9, 0, 4, 8, 1, 6, 3, 5], dtype=np.int8)
                    assert np.array_equal(solve(permutation[original]), permutation[pair["output"]])
                    assert np.array_equal(solve(original[:, ::-1]), pair["output"][:, ::-1])
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 30 and len(outputs) >= 30 and len(palettes) >= 30
    assert bays == {2, 3, 4, 5} and gate_shapes == {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert len(volumes) >= 30 and len(floors) >= 100 and len(positions) >= 30
    assert len(sources) >= 10 and len(levels) >= 5
    count_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(314159, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
            count_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.5, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(81, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("damaged answer accepted")
    _diagnostic_test()
    return dict(seeds=42, pairs=pair_count, distinct_puzzles=len(signatures),
                input_shapes=len(shapes), output_shapes=len(outputs),
                normalized_structures=len(structures), role_palettes=len(palettes),
                bay_counts=sorted(bays), volumes=len(volumes), floor_profiles=len(floors),
                gate_shapes=sorted(gate_shapes), legal_count_calls=count_calls)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
