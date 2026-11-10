"""Compositional generators for the cell-centred mirror and bridge puzzle."""

import hashlib

import numpy as np


_SOURCE_DIGESTS = {
    ((20, 20), "0dfc79582bfd9e7af37c8d67d26df903e5923c71843d50c897351eed80b87410"),
    ((21, 18), "7a04b342f8ec7c1dd114c7800c2495b935a7b362df0b6e64b9cd353a4fab638f"),
    ((19, 23), "941e4991d58d41765b920b02f478db7b2c417542ac025e5381b9957c8f3576a0"),
}


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _choose_counts(rng, seed, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_train = 2 + seed % 9
        num_test = 1 + (seed // 9) % min(4, num_train - 1)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    found = []
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            points = []
            while stack:
                r, c = stack.pop()
                points.append((r, c))
                for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            found.append(points)
    return found


def _bounds(points):
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _footprints(shape, orientation, axis, rects):
    height, width = shape
    axis_mask = np.zeros(shape, dtype=bool)
    seed_mask = np.zeros(shape, dtype=bool)
    bridge_mask = np.zeros(shape, dtype=bool)
    mirror_mask = np.zeros(shape, dtype=bool)
    if orientation == "vertical":
        axis_mask[:, axis] = True
    else:
        axis_mask[axis, :] = True
    reflected = []
    for r0, r1, c0, c1, color in rects:
        assert 0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width
        assert not np.any(seed_mask[r0:r1, c0:c1])
        seed_mask[r0:r1, c0:c1] = True
        if orientation == "vertical":
            rc0, rc1 = 2 * axis - c1 + 1, 2 * axis - c0 + 1
            assert 0 <= rc0 < rc1 <= width
            mirrored = (r0, r1, rc0, rc1, color)
            lo, hi = sorted(((c0, c1), (rc0, rc1)))
            bridge_mask[r0:r1, lo[1]:hi[0]] = True
            mirror_mask[r0:r1, rc0:rc1] = True
        else:
            rr0, rr1 = 2 * axis - r1 + 1, 2 * axis - r0 + 1
            assert 0 <= rr0 < rr1 <= height
            mirrored = (rr0, rr1, c0, c1, color)
            lo, hi = sorted(((r0, r1), (rr0, rr1)))
            bridge_mask[lo[1]:hi[0], c0:c1] = True
            mirror_mask[rr0:rr1, c0:c1] = True
        reflected.append(mirrored)
    assert not np.any(seed_mask & (axis_mask | bridge_mask | mirror_mask))
    assert not np.any(bridge_mask & mirror_mask)
    return axis_mask, seed_mask, bridge_mask, mirror_mask, reflected


def _paint(grid, line, bridge_mask, reflected):
    result = grid.copy()
    result[bridge_mask] = line
    for r0, r1, c0, c1, color in reflected:
        result[r0:r1, c0:c1] = color
    return result


def _sample_spans(rng, length, count, thin_index):
    sizes = [2] * count
    if thin_index >= 0:
        sizes[thin_index] = 4
    slack = length - sum(sizes) - (count - 1)
    assert slack >= 0
    for index in rng.permutation(count):
        room = min(6 - sizes[int(index)], slack)
        extra = int(rng.integers(room + 1))
        sizes[int(index)] += extra
        slack -= extra
    gaps = np.ones(count + 1, dtype=int)
    gaps[0] = gaps[-1] = 0
    if slack:
        if rng.random() < 0.22:
            keep_zero = 0 if rng.random() < 0.5 else count
            choices = [i for i in range(count + 1) if i != keep_zero]
        else:
            choices = list(range(count + 1))
        for item in rng.choice(choices, size=slack):
            gaps[int(item)] += 1
    spans = []
    cursor = int(gaps[0])
    for index, size in enumerate(sizes):
        spans.append((cursor, cursor + size))
        cursor += size + int(gaps[index + 1])
    assert cursor == length
    return spans


def _role_maps(rng, total, mode):
    base = [int(x) for x in rng.choice(10, size=4, replace=False)]
    maps = [tuple(base)]
    for index in range(1, total):
        if mode == "stable":
            current = base.copy()
        elif mode == "partial" or (mode == "free" and index > 1 and rng.random() < 0.45):
            current = base.copy()
            changed = rng.choice(4, size=int(rng.integers(1, 3)), replace=False)
            fresh = rng.choice([c for c in range(10) if c not in base], size=len(changed), replace=False)
            for role, color in zip(changed, fresh):
                current[int(role)] = int(color)
        else:
            current = [int(x) for x in rng.choice(10, size=4, replace=False)]
            rng.shuffle(current)
        maps.append(tuple(current))
    return maps


def _place_distractors(rng, grid, color, unavailable, bridge, mirror, required):
    chosen = np.zeros(grid.shape, dtype=bool)
    height, width = grid.shape

    def place(points):
        group = set(points)
        for row, col in group:
            if not (0 <= row < height and 0 <= col < width) or unavailable[row, col] or chosen[row, col]:
                return False
            for rr, cc in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in group and chosen[rr, cc]:
                    return False
        for row, col in group:
            chosen[row, col] = True
        return True

    exterior = ~(unavailable | bridge | mirror)
    if required:
        for mask in (bridge & ~unavailable, mirror & ~unavailable, exterior):
            locations = np.argwhere(mask)
            for item in rng.permutation(len(locations)):
                row, col = (int(x) for x in locations[int(item)])
                if place(((row, col),)):
                    break
            else:
                raise AssertionError("required overwrite mark has no space")

    density = int(rng.choice(3, p=[0.29, 0.48, 0.23]))
    area = height * width
    if density == 0:
        desired = int(rng.integers(1, 6))
    elif density == 1:
        desired = int(rng.integers(6, min(24, area // 14 + 13) + 1))
    else:
        desired = int(rng.integers(24, min(55, area // 6 + 14) + 1))
    desired = max(desired, int(chosen.sum()))
    shapes = (((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 0)),
              ((0, 0), (1, 0), (1, 1)), ((0, 0), (0, 1), (1, 0)),
              ((0, 0), (0, 1), (1, 1)), ((0, 1), (1, 0), (1, 1)))
    for _ in range(area * 4):
        if int(chosen.sum()) >= desired:
            break
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        shape = shapes[int(rng.choice(len(shapes), p=(0.43, 0.17, 0.17, 0.06, 0.06, 0.06, 0.05)))]
        place(tuple((row + dr, col + dc) for dr, dc in shape))
    if not np.any(chosen):
        for row in range(height):
            for col in range(width):
                if place(((row, col),)):
                    break
            if np.any(chosen):
                break
    grid[chosen] = color


def _generate_pair(rng, orientation, role_map, seed_count, mixed, multicolor, thin, required_marks):
    height = int(rng.integers(14, 31))
    width = int(rng.integers(14, 31))
    parallel = height if orientation == "vertical" else width
    perpendicular = width if orientation == "vertical" else height
    max_count = min(5, (parallel + 1 - (2 if thin else 0)) // 3)
    seed_count = min(seed_count, max_count)
    if mixed or multicolor:
        seed_count = max(2, seed_count)
    thin_index = int(rng.integers(seed_count)) if thin else -1
    spans = _sample_spans(rng, parallel, seed_count, thin_index)
    candidates = [a for a in range(3, perpendicular - 3)]
    centered = [a for a in candidates if 2 * a == perpendicular - 1]
    axis = int(centered[0] if centered and rng.random() < 0.27 else rng.choice(candidates))
    radius = min(axis, perpendicular - 1 - axis)
    side = int(rng.choice((-1, 1)))
    sides = [side] * seed_count
    if mixed:
        sides[0], sides[1] = -1, 1
        for index in range(2, seed_count):
            sides[index] = int(rng.choice((-1, 1)))
        rng.shuffle(sides)
    colors = [role_map[2]] * seed_count
    if multicolor:
        second = int(rng.choice([c for c in range(10) if c not in role_map]))
        colors[1] = second
        for index in range(2, seed_count):
            if rng.random() < 0.45:
                colors[index] = second
    rects = []
    for index, (start, end) in enumerate(spans):
        thickness = 1 if index == thin_index else int(rng.integers(2, min(5, radius - 1) + 1))
        gap = int(rng.integers(1, radius - thickness + 1))
        side = sides[index]
        if orientation == "vertical":
            c0 = axis - gap - thickness if side < 0 else axis + gap + 1
            rects.append((start, end, c0, c0 + thickness, colors[index]))
        else:
            r0 = axis - gap - thickness if side < 0 else axis + gap + 1
            rects.append((r0, r0 + thickness, start, end, colors[index]))
    background, line, _, distractor = role_map
    grid = np.full((height, width), background, dtype=np.int8)
    axis_mask, seed_mask, bridge, mirror, reflected = _footprints(grid.shape, orientation, axis, rects)
    grid[axis_mask] = line
    for r0, r1, c0, c1, color in rects:
        grid[r0:r1, c0:c1] = color
    _place_distractors(rng, grid, distractor, axis_mask | seed_mask, bridge, mirror, required_marks)
    return {"input": grid, "output": _paint(grid, line, bridge, reflected)}


def _analyze_input(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert 4 <= len(values) <= 5
    assert int(np.sum(counts == counts.max())) == 1
    background = int(values[int(np.argmax(counts))])
    rows = [r for r in range(grid.shape[0]) if grid[r, 0] != background and np.all(grid[r] == grid[r, 0])]
    cols = [c for c in range(grid.shape[1]) if grid[0, c] != background and np.all(grid[:, c] == grid[0, c])]
    assert len(rows) + len(cols) == 1
    orientation = "vertical" if cols else "horizontal"
    axis = cols[0] if cols else rows[0]
    line = int(grid[0, axis] if cols else grid[axis, 0])
    rects = []
    distractor_colors = []
    for color in values:
        color = int(color)
        if color in (background, line):
            continue
        components = _components(grid == color)
        large = []
        for component in components:
            bounds = _bounds(component)
            r0, r1, c0, c1 = bounds
            if len(component) >= 4:
                assert len(component) == (r1-r0)*(c1-c0)
                assert (r1-r0 >= 2 and c1-c0 >= 2) or max(r1-r0, c1-c0) >= 4
                large.append((*bounds, color))
            else:
                assert len(component) <= 3
        if large:
            assert len(large) == len(components)
            rects.extend(large)
        else:
            distractor_colors.append(color)
    assert 1 <= len(rects) <= 5 and len(distractor_colors) == 1
    distractor = distractor_colors[0]
    assert grid.shape[0] >= 14 and grid.shape[1] >= 14
    parallel = grid.shape[0] if orientation == "vertical" else grid.shape[1]
    perpendicular = grid.shape[1] if orientation == "vertical" else grid.shape[0]
    assert 3 <= axis < perpendicular - 3
    spans = sorted((r0, r1) if cols else (c0, c1) for r0, r1, c0, c1, _ in rects)
    assert all(spans[i+1][0] - spans[i][1] >= 1 for i in range(len(spans)-1))
    assert spans[0][0] >= 0 and spans[-1][1] <= parallel
    sides = []
    for r0, r1, c0, c1, _ in rects:
        a0, a1 = (c0, c1) if cols else (r0, r1)
        assert a1 < axis or a0 > axis + 1
        sides.append(-1 if a1 < axis else 1)
    axis_mask, seed_mask, bridge, mirror, reflected = _footprints(grid.shape, orientation, axis, rects)
    assert np.array_equal(grid == line, axis_mask)
    actual_seed = np.zeros(grid.shape, dtype=bool)
    for r0, r1, c0, c1, color in rects:
        actual_seed |= grid == color
        assert np.all(grid[r0:r1, c0:c1] == color)
    assert np.array_equal(actual_seed, seed_mask)
    distractors = grid == distractor
    assert 1 <= int(distractors.sum()) <= 100
    return {"background": background, "line": line, "distractor": distractor,
            "orientation": orientation, "axis": axis, "rects": tuple(rects),
            "sides": tuple(sides), "bridge": bridge, "mirror": mirror,
            "reflected": reflected, "distractors": distractors,
            "centered": 2*axis == perpendicular-1,
            "edge_seed": spans[0][0] == 0 or spans[-1][1] == parallel,
            "thin": any(r1-r0 == 1 or c1-c0 == 1 for r0,r1,c0,c1,_ in rects),
            "multicolor": len({r[4] for r in rects}) > 1,
            "mixed": len(set(sides)) > 1}


def _signature(pair):
    inp, out = pair["input"], pair["output"]
    return (inp.shape, hashlib.sha256(np.asarray(inp, dtype=np.uint8).tobytes() +
                                      np.asarray(out, dtype=np.uint8).tobytes()).hexdigest())


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = _signature(pair)
            input_key = (inp.shape, np.asarray(inp, dtype=np.uint8).tobytes())
            assert key not in _SOURCE_DIGESTS and key not in pairs and input_key not in inputs
            pairs.add(key)
            inputs.add(input_key)
            info = _analyze_input(inp)
            assert np.array_equal(out, _paint(inp, info["line"], info["bridge"], info["reflected"]))
            assert np.any(inp != out)
            infos[split].append(info)
    training = infos["train"]
    tests = infos["test"]
    train_orientations = {item["orientation"] for item in training}
    assert all(item["orientation"] in train_orientations for item in tests)
    for property_name in ("mixed", "multicolor", "thin"):
        assert not any(item[property_name] for item in tests) or any(item[property_name] for item in training)
    for mask_name in ("bridge", "mirror"):
        if any(np.any(item["distractors"] & item[mask_name]) for item in tests):
            assert any(np.any(item["distractors"] & item[mask_name]) for item in training)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, seed, num_train, num_test)
    orientation_mode = int(rng.integers(3))
    if orientation_mode == 0:
        train_orientations = ["vertical"] * num_train
    elif orientation_mode == 1:
        train_orientations = ["horizontal"] * num_train
    else:
        first = "vertical" if rng.random() < 0.5 else "horizontal"
        other = "horizontal" if first == "vertical" else "vertical"
        train_orientations = [first, other] + [first if rng.random() < 0.5 else other for _ in range(num_train-2)]
    test_orientations = [train_orientations[int(rng.integers(num_train))] for _ in range(num_test)]
    mixed_mode = rng.random() < 0.40
    multicolor_mode = rng.random() < 0.36
    thin_mode = rng.random() < 0.50
    palette_mode = ("stable", "partial", "free")[int(rng.integers(3))]
    role_maps = _role_maps(rng, num_train + num_test, palette_mode)
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=num_train + num_test, dtype=np.int64)
    train = []
    for index, orientation in enumerate(train_orientations):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        count = int(pair_rng.choice((1, 2, 3, 4, 5), p=(0.16, 0.32, 0.29, 0.18, 0.05)))
        if index == 0:
            count = max(2, count)
        pair = _generate_pair(pair_rng, orientation, role_maps[index], count,
                              mixed_mode and (index == 0 or pair_rng.random() < 0.25),
                              multicolor_mode and (index == 0 or pair_rng.random() < 0.30),
                              thin_mode and (index == 1 or (index == 0 and mixed_mode and multicolor_mode)
                                             or pair_rng.random() < 0.30), index == 0)
        train.append(pair)
    test = []
    for index, orientation in enumerate(test_orientations):
        pair_rng = np.random.default_rng(int(pair_seeds[num_train + index]))
        count = int(pair_rng.choice((1, 2, 3, 4, 5), p=(0.11, 0.23, 0.31, 0.27, 0.08)))
        pair = _generate_pair(pair_rng, orientation, role_maps[num_train + index], count,
                              mixed_mode and pair_rng.random() < 0.40,
                              multicolor_mode and pair_rng.random() < 0.65,
                              thin_mode and pair_rng.random() < 0.40, False)
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({tuple(_signature(pair) for split in ("train", "test") for pair in puzzle[split])
                for puzzle in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    records = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            records.extend((split, _analyze_input(pair["input"]), pair["input"].shape)
                           for pair in puzzle[split])
    assert len({shape for _, _, shape in records}) >= 60
    assert len({(shape, info["orientation"], info["axis"], info["rects"])
                for _, info, shape in records}) >= 200
    for name in ("centered", "edge_seed", "thin", "multicolor", "mixed"):
        assert any(info[name] for _, info, _ in records), name
    assert any(split == "test" and info["orientation"] == "horizontal" for split, info, _ in records)
    assert any(split == "train" and len(info["rects"]) == 1 for split, info, _ in records)
    assert any(split == "test" and len(info["rects"]) == 1 for split, info, _ in records)
    assert any(max(r[1]-r[0], r[3]-r[2]) >= 5 for _, info, _ in records for r in info["rects"])
    assert any(len(_components(info["distractors"])) < int(info["distractors"].sum())
               for _, info, _ in records)
    for counts in ((2, 1), (10, 1), (10, 4), (7, 3), (None, 1), (None, 4), (2, None), (10, None)):
        a = generate_puzzle(7301 + (counts[0] or 0) * 10 + (counts[1] or 0), *counts)
        b = generate_puzzle(7301 + (counts[0] or 0) * 10 + (counts[1] or 0), *counts)
        assert all(np.array_equal(x[role], y[role]) for split in ("train", "test")
                   for x, y in zip(a[split], b[split]) for role in ("input", "output"))
        if counts[0] is not None:
            assert len(a["train"]) == counts[0]
        if counts[1] is not None:
            assert len(a["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3, False)):
        try:
            generate_puzzle(17, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
