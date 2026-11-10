"""Generator for reflected, tabbed hollow frames in ARC grids."""

import hashlib
import numpy as np


_DIRECTIONS = ("L", "R", "T", "B")
_HORIZONTAL = frozenset(("L", "R"))
_SOURCE_PAIR_HASHES = frozenset((
    "a2c33f95bc40c29a25830783a7d4df8df113709a3f103fec2f4f2d7ddfa51002",
    "203f43a630779f8d8dd266114101b72d0ee44faf0ecfd5a2583ecc30d96dd2c5",
    "3b68b599c448936cc1c9709a1d92b3dd310e28db5e5a9f4e6927f4274c1dc566",
    "bd189bfe66bbc1db80131041cba12fca2a5c0843d013831631e9fcfae306f010",
))


def _require_count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = None if num_train is None else _require_count("num_train", num_train, 2, 10)
    test = None if num_test is None else _require_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _frame(top, left, height, width):
    bottom, right = top + height - 1, left + width - 1
    return ({(top, col) for col in range(left, right + 1)} |
            {(bottom, col) for col in range(left, right + 1)} |
            {(row, left) for row in range(top, bottom + 1)} |
            {(row, right) for row in range(top, bottom + 1)})


def _geometry(top, left, height, width, direction, offset):
    body = _frame(top, left, height, width)
    if direction == "L":
        tab = (top + offset, left - 1)
        axis = left + width
        reflect = lambda cell: (cell[0], 2 * axis - cell[1])
        gap = {(row, axis) for row in range(top, top + height)}
    elif direction == "R":
        tab = (top + offset, left + width)
        axis = left - 1
        reflect = lambda cell: (cell[0], 2 * axis - cell[1])
        gap = {(row, axis) for row in range(top, top + height)}
    elif direction == "T":
        tab = (top - 1, left + offset)
        axis = top + height
        reflect = lambda cell: (2 * axis - cell[0], cell[1])
        gap = {(axis, col) for col in range(left, left + width)}
    elif direction == "B":
        tab = (top + height, left + offset)
        axis = top - 1
        reflect = lambda cell: (2 * axis - cell[0], cell[1])
        gap = {(axis, col) for col in range(left, left + width)}
    else:
        raise AssertionError("unknown direction")
    return body | {tab}, {reflect(cell) for cell in body}, reflect(tab), gap


def _inside(cell, height, width):
    return 0 <= cell[0] < height and 0 <= cell[1] < width


def _candidate(top, left, spec, grid_height, grid_width):
    original, copied_body, copied_tab, gap = _geometry(
        top, left, spec["height"], spec["width"], spec["direction"], spec["offset"]
    )
    if not all(_inside(cell, grid_height, grid_width) for cell in original | copied_body | gap):
        return None
    clipped = not _inside(copied_tab, grid_height, grid_width)
    copied = copied_body | (set() if clipped else {copied_tab})
    if original & copied or gap & (original | copied):
        return None
    occupied = original | copied
    rows = [cell[0] for cell in occupied]
    cols = [cell[1] for cell in occupied]
    halo = original | {(r + dr, c + dc) for r, c in original
                       for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}
    return {
        "original": original, "copied": copied, "gap": gap,
        "occupied": occupied, "halo": halo, "clipped": clipped,
        "bbox": (min(rows), max(rows), min(cols), max(cols)),
        "center": ((min(rows) + max(rows)) / 2, (min(cols) + max(cols)) / 2),
    }


def _candidates(spec, height, width, clip_mode):
    result = []
    for top in range(height - spec["height"] + 1):
        for left in range(width - spec["width"] + 1):
            candidate = _candidate(top, left, spec, height, width)
            if candidate is None:
                continue
            if clip_mode == "required" and not candidate["clipped"]:
                continue
            if clip_mode == "forbidden" and candidate["clipped"]:
                continue
            result.append(candidate)
    return result


def _compatible(candidate, placed):
    for other in placed:
        if candidate["occupied"] & (other["occupied"] | other["gap"]):
            return False
        if candidate["gap"] & other["occupied"]:
            return False
        if candidate["original"] & other["halo"]:
            return False
        if other["original"] & candidate["halo"]:
            return False
    return True


def _layout_score(candidate, placed, dense):
    if not placed:
        return 0.0
    a, b, c, d = candidate["bbox"]
    overlap = 0
    distance = 100.0
    for other in placed:
        e, f, g, h = other["bbox"]
        overlap += max(0, min(b, f) - max(a, e) + 1) * max(0, min(d, h) - max(c, g) + 1)
        distance = min(distance, abs(candidate["center"][0] - other["center"][0]) +
                       abs(candidate["center"][1] - other["center"][1]))
    return (overlap * 2 - distance) if dense else (distance - overlap * 2)


def _place_specs(rng, height, width, specs, clip_modes, density):
    choices = [_candidates(spec, height, width, mode)
               for spec, mode in zip(specs, clip_modes)]
    if any(not option for option in choices):
        return None
    order = sorted(range(len(specs)), key=lambda i: (len(choices[i]), -len(choices[i][0]["occupied"])))
    for _ in range(24):
        result = [None] * len(specs)
        placed = []
        for index in order:
            indices = rng.permutation(len(choices[index]))
            valid = []
            for j in indices:
                candidate = choices[index][int(j)]
                if _compatible(candidate, placed):
                    valid.append(candidate)
                    if len(valid) == 80:
                        break
            if not valid:
                break
            if density == "random" or not placed:
                chosen = valid[int(rng.integers(len(valid)))]
            else:
                ranked = sorted(valid, key=lambda item: _layout_score(item, placed, density == "dense"), reverse=True)
                chosen = ranked[int(rng.integers(min(4, len(ranked))))]
            result[index] = chosen
            placed.append(chosen)
        else:
            return result
    return None


def _training_plan(rng, count):
    sizes = [int(rng.integers(2 if count == 2 else 1, 5)) for _ in range(count)]
    if count >= 3 and rng.random() < 0.65:
        sizes[int(rng.integers(count))] = 1
    while sum(sizes) < 4:
        i = int(rng.integers(count))
        sizes[i] += 1
    slots = [(i, j) for i, size in enumerate(sizes) for j in range(size)]
    rng.shuffle(slots)
    result = [[None] * size for size in sizes]
    directions = list(_DIRECTIONS)
    rng.shuffle(directions)
    for (i, j), direction in zip(slots, directions):
        result[i][j] = direction
    for i, j in slots[4:]:
        result[i][j] = _DIRECTIONS[int(rng.integers(4))]
    return result


def _specs(rng, directions):
    specs = []
    for direction in directions:
        height = int(rng.integers(3, 9) if rng.random() < 0.35 else rng.integers(3, 6))
        width = int(rng.integers(3, 9) if rng.random() < 0.35 else rng.integers(3, 6))
        side = height if direction in _HORIZONTAL else width
        specs.append({"direction": direction, "height": height, "width": width,
                      "offset": int(rng.integers(1, side - 1))})
    if len(specs) >= 2 and rng.random() < 0.23:
        first, second = specs[:2]
        second["height"], second["width"] = first["height"], first["width"]
        side = second["height"] if second["direction"] in _HORIZONTAL else second["width"]
        offsets = [i for i in range(1, side - 1) if i != first["offset"]]
        second["offset"] = int(rng.choice(offsets or list(range(1, side - 1))))
    return specs


def _generate_pair(rng, regime, directions, background, source, horizontal, vertical,
                   require_clip):
    count = len(directions)
    for attempt in range(110):
        specs = _specs(rng, directions)
        lower = min(27, 13 + 2 * count + (1 if regime == "target_test" else 0))
        height = int(rng.integers(lower, 31))
        width = int(rng.integers(lower, 31))
        modes = ["forbidden"] * count
        if require_clip:
            modes[int(rng.integers(count))] = "required"
        density = rng.choice(("sparse", "random", "dense"), p=(0.28, 0.36, 0.36))
        placed = _place_specs(rng, height, width, specs, modes, density)
        if placed is None:
            continue
        input_grid = np.full((height, width), background, dtype=np.int8)
        output_grid = input_grid.copy()
        for spec, placement in zip(specs, placed):
            for row, col in placement["original"]:
                input_grid[row, col] = source
                output_grid[row, col] = source
            color = horizontal if spec["direction"] in _HORIZONTAL else vertical
            for row, col in placement["copied"]:
                output_grid[row, col] = color
        return input_grid, output_grid
    raise ValueError("could not place a valid pair")


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _source_pair_hash(input_grid, output_grid):
    height, width = input_grid.shape
    data = bytes((height, width)) + input_grid.astype(np.uint8).tobytes() + output_grid.astype(np.uint8).tobytes()
    return hashlib.sha256(data).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    horizontal, vertical = [int(value) for value in rng.choice(10, size=2, replace=False)]
    local_colors = [color for color in range(10) if color not in (horizontal, vertical)]
    rng.shuffle(local_colors)
    shift = int(rng.integers(1, len(local_colors)))
    training_plan = _training_plan(rng, train_count)
    clip_train_index = int(rng.integers(train_count)) if rng.random() < 0.55 else None
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for pair_index in range(count):
            if split == "train":
                directions = training_plan[pair_index]
                source = local_colors[pair_index % len(local_colors)]
                background = local_colors[(pair_index + shift) % len(local_colors)]
                require_clip = pair_index == clip_train_index
                regime = "source_train"
            else:
                object_count = int(rng.integers(1, 6))
                directions = [_DIRECTIONS[int(rng.integers(4))] for _ in range(object_count)]
                source, background = [int(value) for value in rng.choice(local_colors, size=2, replace=False)]
                require_clip = clip_train_index is not None and rng.random() < 0.28
                regime = "target_test"
            for _ in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                input_grid, output_grid = _generate_pair(pair_rng, regime, directions,
                    background, source, horizontal, vertical, require_clip)
                signature = _signature(input_grid)
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise ValueError("could not produce a unique input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for start_row, start_col in zip(*np.nonzero(mask)):
        start = (int(start_row), int(start_col))
        if seen[start]:
            continue
        component, stack = set(), [start]
        seen[start] = True
        while stack:
            row, col = stack.pop()
            component.add((row, col))
            for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(component)
    return result


def _decompose(component):
    min_row = min(r for r, _ in component)
    max_row = max(r for r, _ in component)
    min_col = min(c for _, c in component)
    max_col = max(c for _, c in component)
    candidates = []
    for direction in _DIRECTIONS:
        if direction == "L":
            top, left = min_row, min_col + 1
            height, width = max_row - min_row + 1, max_col - min_col
        elif direction == "R":
            top, left = min_row, min_col
            height, width = max_row - min_row + 1, max_col - min_col
        elif direction == "T":
            top, left = min_row + 1, min_col
            height, width = max_row - min_row, max_col - min_col + 1
        else:
            top, left = min_row, min_col
            height, width = max_row - min_row, max_col - min_col + 1
        if not 3 <= height <= 8 or not 3 <= width <= 8:
            continue
        outline = _frame(top, left, height, width)
        extra = component - outline
        if not outline.issubset(component) or len(extra) != 1:
            continue
        row, col = next(iter(extra))
        if direction == "L":
            offset, valid = row - top, col == left - 1 and top < row < top + height - 1
        elif direction == "R":
            offset, valid = row - top, col == left + width and top < row < top + height - 1
        elif direction == "T":
            offset, valid = col - left, row == top - 1 and left < col < left + width - 1
        else:
            offset, valid = col - left, row == top + height and left < col < left + width - 1
        if valid and _geometry(top, left, height, width, direction, offset)[0] == component:
            candidates.append({"top": top, "left": left, "height": height, "width": width,
                               "direction": direction, "offset": offset})
    assert len(candidates) == 1
    return candidates[0]


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 2
    order = np.argsort(counts)
    assert counts[order[-1]] > counts[order[-2]]
    background, source = int(values[order[-1]]), int(values[order[-2]])
    components = _components(input_grid == source)
    assert 1 <= len(components) <= 5
    expected = input_grid.copy()
    copied_cells, gaps, specs, axis_colors = set(), set(), [], {}
    for component in components:
        spec = _decompose(component)
        original, body, tab, gap = _geometry(**spec)
        assert original == component
        assert all(_inside(cell, *input_grid.shape) for cell in body | gap)
        assert all(int(input_grid[cell]) == background for cell in body | gap)
        visible = body | ({tab} if _inside(tab, *input_grid.shape) else set())
        assert all(int(input_grid[cell]) == background for cell in visible)
        assert copied_cells.isdisjoint(visible)
        copied_cells.update(visible)
        gaps.update(gap)
        colors = {int(output_grid[cell]) for cell in visible}
        assert len(colors) == 1
        color = next(iter(colors))
        assert color not in (background, source)
        axis = "H" if spec["direction"] in _HORIZONTAL else "V"
        assert axis not in axis_colors or axis_colors[axis] == color
        axis_colors[axis] = color
        for cell in visible:
            expected[cell] = color
        spec["clipped"] = not _inside(tab, *input_grid.shape)
        specs.append(spec)
    assert gaps.isdisjoint(copied_cells)
    assert np.array_equal(expected, output_grid)
    return {"background": background, "source": source, "axis_colors": axis_colors,
            "specs": specs}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, analyses = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            sig = _signature(inp)
            pair_sig = (sig, _signature(out))
            assert sig not in seen_inputs and pair_sig not in seen_pairs
            seen_inputs.add(sig)
            seen_pairs.add(pair_sig)
            assert _source_pair_hash(inp, out) not in _SOURCE_PAIR_HASHES
            analyses[split].append(_analyze_pair(inp, out))
    learned = {}
    directions = set()
    for info in analyses["train"]:
        assert info["background"] != info["source"]
        directions.update(spec["direction"] for spec in info["specs"])
        for axis, color in info["axis_colors"].items():
            assert axis not in learned or learned[axis] == color
            learned[axis] = color
    assert directions == set(_DIRECTIONS)
    assert set(learned) == {"H", "V"} and learned["H"] != learned["V"]
    local_roles = {(info["background"], info["source"]) for info in analyses["train"]}
    assert len(local_roles) >= 2
    for split in ("train", "test"):
        for info in analyses[split]:
            assert len({info["background"], info["source"], learned["H"], learned["V"]}) == 4
            for axis, color in info["axis_colors"].items():
                assert learned[axis] == color
    if any(spec["clipped"] for info in analyses["test"] for spec in info["specs"]):
        assert any(spec["clipped"] for info in analyses["train"] for spec in info["specs"])


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[key], b[key]) for a, b in zip(first[split], second[split])
                   for key in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(101)
    assert _puzzles_equal(automatic, generate_puzzle(101))
    assert not _puzzles_equal(automatic, generate_puzzle(102))
    assert _puzzles_equal(generate_puzzle(2024, 5, 2), generate_puzzle(2024, 5, 2))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(3000 + 10 * train + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for seed, train, test in ((4001, None, 1), (4002, None, 4),
                              (4003, 2, None), (4004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(5000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    train_counts, test_counts, palettes, shapes, layouts = set(), set(), set(), set(), set()
    split_shapes = {"train": set(), "test": set()}
    first_two_patterns = set()
    flags = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = {split: [_analyze_pair(p["input"], p["output"]) for p in puzzle[split]]
                 for split in ("train", "test")}
        palettes.add(tuple(sorted(infos["train"][0]["axis_colors"].items())))
        first_two_patterns.add(tuple(
            tuple(sorted(spec["direction"] for spec in info["specs"]))
            for info in infos["train"][:2]
        ))
        train_sizes = [pair["input"].shape for pair in puzzle["train"]]
        test_sizes = [pair["input"].shape for pair in puzzle["test"]]
        if any(height < max(shape[0] for shape in train_sizes) for height, _ in test_sizes):
            flags.add("test_smaller_height")
        if any(width < max(shape[1] for shape in train_sizes) for _, width in test_sizes):
            flags.add("test_smaller_width")
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                shapes.add(pair["input"].shape)
                split_shapes[split].add(pair["input"].shape)
                layouts.add((pair["input"].shape, (pair["input"] == info["source"]).tobytes()))
                directions = [spec["direction"] for spec in info["specs"]]
                axis_list = ["H" if d in _HORIZONTAL else "V" for d in directions]
                axes = set(axis_list)
                if len(directions) == 1: flags.add("single")
                if len(axes) == 1 and len(directions) > 1: flags.add("single_axis")
                if len(set(directions)) < len(directions): flags.add("repeated")
                if any(spec["clipped"] for spec in info["specs"]): flags.add("clipped_" + split)
                if any(spec["width"] >= 6 for spec in info["specs"]): flags.add("wide_" + split)
                if any(spec["height"] >= 6 and spec["width"] <= 4 for spec in info["specs"]): flags.add("tall_narrow")
                if any(spec["width"] >= 6 and spec["height"] <= 4 for spec in info["specs"]): flags.add("wide_shallow")
                if len(axes) == 2: flags.add("mixed")
                if len(axes) == 2 and axis_list.count("H") != axis_list.count("V"):
                    flags.add("uneven_axes")
                if len({(spec["height"], spec["width"]) for spec in info["specs"]}) > 1:
                    flags.add("distinct_frame_sizes")
                if split == "test" and set(directions) == set(_DIRECTIONS):
                    flags.add("test_all_four")
                if split == "test" and all(spec["width"] <= 5 for spec in info["specs"]):
                    flags.add("test_ordinary_only")
                if any(
                    first["height"] == second["height"]
                    and first["width"] == second["width"]
                    and first["offset"] != second["offset"]
                    for index, first in enumerate(info["specs"])
                    for second in info["specs"][index + 1:]
                ):
                    flags.add("same_size_different_tab")
                if any(
                    (spec["height"] if spec["direction"] in _HORIZONTAL else spec["width"]) >= 5
                    and 2 * spec["offset"] !=
                    (spec["height"] if spec["direction"] in _HORIZONTAL else spec["width"]) - 1
                    for spec in info["specs"]
                ):
                    flags.add("off_center_tab")
                coverage = int((pair["input"] != info["background"]).sum()) / pair["input"].size
                if coverage < 0.04:
                    flags.add("sparse")
                if coverage > 0.12:
                    flags.add("crowded")
                boxes = []
                for spec in info["specs"]:
                    original, body, tab, _ = _geometry(
                        spec["top"], spec["left"], spec["height"], spec["width"],
                        spec["direction"], spec["offset"]
                    )
                    occupied = original | body | (set() if spec["clipped"] else {tab})
                    rows = [row for row, _ in occupied]
                    cols = [col for _, col in occupied]
                    boxes.append((min(rows), max(rows), min(cols), max(cols)))
                if any(
                    max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
                    * max(0, min(a[3], b[3]) - max(a[2], b[2]) + 1) > 0
                    for index, a in enumerate(boxes) for b in boxes[index + 1:]
                ):
                    flags.add("interleaved")
        if len({(info["background"], info["source"]) for info in infos["train"]}) >= 2:
            flags.add("local_colors")
        if len({info["background"] for info in infos["train"]}) >= 2:
            flags.add("background_variation")
        if len({info["source"] for info in infos["train"]}) >= 2:
            flags.add("source_variation")
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 15 and len(shapes) >= 40 and len(layouts) >= 150
    assert len(first_two_patterns) >= 12
    assert split_shapes["train"] & split_shapes["test"]
    assert {"single", "single_axis", "repeated", "clipped_train", "clipped_test",
            "wide_train", "wide_test", "tall_narrow", "wide_shallow", "mixed",
            "local_colors", "test_smaller_height", "test_smaller_width", "test_all_four",
            "test_ordinary_only", "same_size_different_tab", "off_center_tab",
            "sparse", "crowded", "interleaved", "uneven_axes",
            "distinct_frame_sizes", "background_variation", "source_variation"} <= flags
