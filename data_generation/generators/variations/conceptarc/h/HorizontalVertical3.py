import numpy as np


# Exact source inputs, retained solely to exclude exact source pairs.  Their
# 11x12 and 12x12 canvas sizes remain available to new examples.
_SOURCE_INPUTS = (
    (12, 12, "000000000000000000000000000000000000000000000000022022022022022022022022000000000000000000000000000000000000000000000000000000000000000000000000", "H"),
    (12, 12, "000000000000000000000000000333300000000333300000000000000000000333300000000333300000000000000000000333300000000333300000000000000000000000000000", "V"),
    (11, 12, "000000000000000004000400000044404440000004000400000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "H"),
    (12, 12, "000000000000000000000000007000007000077700077700777770777770077700077700007000007000000000000000000000000000000000000000000000000000000000000000", "H"),
    (12, 12, "000000000000008880000000000800000000000800000000000000000000008880000000000800000000000800000000000000000000008880000000000800000000000800000000", "V"),
    (12, 12, "000000000000000555000000000555500000000055500000000005500000000005500000000000000000000555000000000555500000000055500000000005500000000005500000", "V"),
)


def _source_pair_keys():
    keys = set()
    for height, width, flat, direction in _SOURCE_INPUTS:
        assert len(flat) == height * width
        grid = np.array([int(value) for value in flat], dtype=np.int8).reshape(height, width)
        output = grid.copy()
        if direction == "H":
            output[0, :] = 1
        else:
            output[:, 0] = 1
        keys.add((_grid_key(grid), _grid_key(output)))
    return keys


def _grid_key(grid):
    return grid.shape, bytes(int(value) for value in grid.flat)


def _choose_counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None and (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or not low <= int(value) <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    options = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not options:
        raise ValueError("num_train must exceed num_test")
    return options[int(rng.integers(len(options)))]


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    seen = set()
    stack = [next(iter(cells))]
    while stack:
        row, col = stack.pop()
        if (row, col) in seen or (row, col) not in cells:
            continue
        seen.add((row, col))
        stack.extend(((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)))
    return seen == cells


def _motif(rng, direction, count, split, both_axes=False):
    major_limit = min(7, (29 - (count - 1)) // count)
    major = int(rng.integers(2, major_limit + 1))
    minor_limit = major_limit if both_axes else 7
    minor = int(rng.integers(2, minor_limit + 1))
    height, width = (minor, major) if direction == "H" else (major, minor)

    choices = ("solid", "L", "plus", "diamond", "stair", "irregular")
    weights = ([22, 18, 16, 11, 12, 21] if split == "train"
               else [18, 15, 15, 12, 13, 27])
    kind = str(rng.choice(choices, p=np.array(weights) / sum(weights)))
    if kind in ("plus", "diamond") and min(height, width) < 3:
        kind = "L"

    mask = np.zeros((height, width), dtype=bool)
    if kind == "solid":
        mask[:] = True
    elif kind == "L":
        mask[:, 0] = True
        mask[-1, :] = True
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
    elif kind == "plus":
        mask[height // 2, :] = True
        mask[:, width // 2] = True
        # A second central stripe gives the source's thicker plus variants.
        if height >= 5 and rng.random() < 0.4:
            mask[height // 2 - 1, :] = True
    elif kind == "diamond":
        cy, cx = (height - 1) / 2, (width - 1) / 2
        for row in range(height):
            for col in range(width):
                mask[row, col] = abs(row - cy) / cy + abs(col - cx) / cx <= 1.15
        mask[height // 2, :] = True
        mask[:, width // 2] = True
    elif kind == "stair":
        previous = 0
        for row in range(height):
            center = round(row * (width - 1) / max(1, height - 1))
            mask[row, min(previous, center):max(previous, center) + 1] = True
            previous = center
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
    else:
        mask[:] = True
        removal_goal = int(rng.integers(max(1, height * width // 5),
                                        max(2, height * width // 2) + 1))
        removed = 0
        for index in rng.permutation(height * width):
            row, col = divmod(int(index), width)
            mask[row, col] = False
            if mask.any(axis=0).all() and mask.any(axis=1).all() and _connected(mask):
                removed += 1
                if removed >= removal_goal:
                    break
            else:
                mask[row, col] = True
        if removed == 0:
            mask[-1, -1] = False
    assert mask.any(axis=0).all() and mask.any(axis=1).all() and _connected(mask)
    return mask


def _canvas(rng, direction, occupied, minor_min):
    mode = str(rng.choice(("compact", "roomy", "square", "cross", "source", "free"),
                          p=(0.23, 0.19, 0.16, 0.18, 0.14, 0.10)))
    if mode == "source":
        height, width = ((11, 12) if rng.random() < 0.5 else (12, 12))
        major, minor = (width, height) if direction == "H" else (height, width)
        if occupied <= major and minor_min <= minor:
            return height, width
        mode = "compact"
    if mode == "square":
        side = min(30, max(occupied, minor_min) + int(rng.integers(0, 6)))
        return side, side
    if mode == "cross" and occupied < 30:
        major = min(29, occupied + int(rng.integers(0, 5)))
        minor = int(rng.integers(max(minor_min, major + 1), 31))
    elif mode == "roomy":
        major = min(30, occupied + int(rng.integers(2, 10)))
        minor = min(30, minor_min + int(rng.integers(5, 19)))
    elif mode == "free":
        major = int(rng.integers(occupied, 31))
        minor = int(rng.integers(minor_min, 31))
    else:
        major = min(30, occupied + int(rng.integers(0, 5)))
        minor = min(30, minor_min + int(rng.integers(0, 6)))
    return (minor, major) if direction == "H" else (major, minor)


def _pick_start(rng, slack, first):
    if slack == 0:
        return first
    choice = rng.random()
    if choice < 0.29:
        return first
    if choice < 0.58:
        return first + slack
    return first + int(rng.integers(0, slack + 1))


def _make_pair(rng, direction, count, mask, background, foreground, indicator):
    height, width = mask.shape
    motif_major = width if direction == "H" else height
    motif_minor = height if direction == "H" else width
    gaps = [int(rng.integers(1, 5)) for _ in range(count - 1)]
    while motif_major * count + sum(gaps) > 30:
        candidates = [i for i, gap in enumerate(gaps) if gap > 1]
        gaps[candidates[int(rng.integers(len(candidates)))]] -= 1
    occupied = motif_major * count + sum(gaps)
    grid_height, grid_width = _canvas(rng, direction, occupied, motif_minor + 1)
    major_size = grid_width if direction == "H" else grid_height
    minor_size = grid_height if direction == "H" else grid_width
    # Keep the canvas color visibly dominant, including on dense compact grids.
    required_area = 2 * count * int(mask.sum()) + 1
    if major_size * minor_size < required_area:
        minor_size = min(30, max(minor_size, (required_area + major_size - 1) // major_size))
        if major_size * minor_size < required_area:
            major_size = min(30, max(major_size, (required_area + minor_size - 1) // minor_size))
        grid_height, grid_width = ((minor_size, major_size) if direction == "H"
                                   else (major_size, minor_size))
    major_start = _pick_start(rng, major_size - occupied, 0)
    minor_start = _pick_start(rng, minor_size - motif_minor - 1, 1)
    grid = np.full((grid_height, grid_width), background, dtype=np.int8)
    position = major_start
    for index in range(count):
        top, left = (minor_start, position) if direction == "H" else (position, minor_start)
        tile = grid[top:top + height, left:left + width]
        tile[mask] = foreground
        if index < count - 1:
            position += motif_major + gaps[index]
    output = grid.copy()
    if direction == "H":
        output[0, :] = indicator
    else:
        output[:, 0] = indicator
    return {"input": grid, "output": output}


def _colors(rng, indicator, base_background, base_foreground, used_train, split, first=False):
    available = [color for color in range(10) if color != indicator]
    if first or rng.random() < 0.32:
        return base_background, base_foreground
    if rng.random() < 0.24:
        return base_foreground, base_background
    background = base_background if rng.random() < 0.55 else int(rng.choice(available))
    foreground_options = [color for color in available if color != background]
    unseen = [color for color in foreground_options if color not in used_train]
    if split == "test" and unseen and rng.random() < 0.33:
        foreground = int(rng.choice(unseen))
    elif base_foreground in foreground_options and rng.random() < 0.48:
        foreground = base_foreground
    else:
        foreground = int(rng.choice(foreground_options))
    return background, foreground


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    indicator = int(rng.integers(0, 10))
    base_background, base_foreground = map(int, rng.choice(
        [color for color in range(10) if color != indicator], size=2, replace=False))

    train_directions = ["H", "V"] + [str(rng.choice(("H", "V")))
                                     for _ in range(train_count - 2)]
    test_directions = (["H", "V"] if test_count >= 2 else []) + [
        str(rng.choice(("H", "V"))) for _ in range(test_count - min(2, test_count))]
    if test_count == 1:
        test_directions = [str(rng.choice(("H", "V")))]

    mirror = rng.random() < 0.58
    mirror_count = int(rng.integers(2, 5)) if mirror else None
    mirror_mask = (_motif(rng, "H", mirror_count, "train", both_axes=True)
                   if mirror else None)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    used_train = set()
    for split, directions in (("train", train_directions), ("test", test_directions)):
        for index, direction in enumerate(directions):
            mirrored = split == "train" and mirror and index < 2
            count = mirror_count if mirrored else int(rng.choice(
                np.arange(2, 8), p=(0.28, 0.24, 0.20, 0.14, 0.09, 0.05)))
            mask = mirror_mask if mirrored else _motif(rng, direction, count, split)
            if mirrored or (split == "train" and index == 0):
                background, foreground = base_background, base_foreground
            else:
                background, foreground = _colors(rng, indicator, base_background,
                                                 base_foreground, used_train, split)
            for _ in range(30):
                pair = _make_pair(rng, direction, count, mask, background, foreground, indicator)
                key = _grid_key(pair["input"])
                if key not in seen_inputs and (_grid_key(pair["input"]), _grid_key(pair["output"])) not in _SOURCE_PAIRS:
                    break
            else:
                raise RuntimeError("Could not construct a unique pair")
            seen_inputs.add(key)
            puzzle[split].append(pair)
            if split == "train":
                used_train.add(foreground)
        if split == "train":
            puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_input(grid):
    background = int(grid[0, 0])
    colors = set(map(int, np.unique(grid)))
    assert len(colors) == 2 and background in colors
    foreground = next(iter(colors - {background}))
    assert np.count_nonzero(grid == background) > np.count_nonzero(grid == foreground)
    remaining = set(map(tuple, np.argwhere(grid == foreground)))
    components = []
    while remaining:
        stack = [next(iter(remaining))]
        component = set()
        while stack:
            cell = stack.pop()
            if cell not in remaining:
                continue
            remaining.remove(cell)
            component.add(cell)
            row, col = cell
            stack.extend(((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)))
        components.append(component)
    assert len(components) >= 2
    origins, shapes = [], []
    for component in components:
        top = min(row for row, _ in component)
        left = min(col for _, col in component)
        origins.append((top, left))
        shapes.append(frozenset((row - top, col - left) for row, col in component))
    shape = shapes[0]
    assert all(other == shape for other in shapes)
    motif_height = max(row for row, _ in shape) + 1
    motif_width = max(col for _, col in shape) + 1
    if len({row for row, _ in origins}) == 1:
        direction = "H"
        starts = sorted(col for _, col in origins)
        major = motif_width
        assert np.all(grid[0, :] == background)
    else:
        assert len({col for _, col in origins}) == 1
        direction = "V"
        starts = sorted(row for row, _ in origins)
        major = motif_height
        assert np.all(grid[:, 0] == background)
    gaps = tuple(starts[i + 1] - starts[i] - major for i in range(len(starts) - 1))
    assert all(gap >= 1 for gap in gaps)
    return {"background": background, "foreground": foreground, "direction": direction,
            "count": len(components), "shape": shape, "motif_height": motif_height,
            "motif_width": motif_width, "gaps": gaps,
            "solid": len(shape) == motif_height * motif_width}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    train_directions, indicators = set(), set()
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for item in (grid, output):
                assert isinstance(item, np.ndarray) and item.ndim == 2
                assert 1 <= item.shape[0] <= 30 and 1 <= item.shape[1] <= 30
                assert np.issubdtype(item.dtype, np.integer)
                assert np.all((0 <= item) & (item <= 9))
            assert grid.shape == output.shape
            input_key, pair_key = _grid_key(grid), (_grid_key(grid), _grid_key(output))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIRS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            info = _analyze_input(grid)
            assert 2 <= info["count"] <= 7
            assert info["background"] != info["foreground"]
            changed = grid != output
            if info["direction"] == "H":
                expected_changed = np.zeros_like(changed)
                expected_changed[0, :] = True
            else:
                expected_changed = np.zeros_like(changed)
                expected_changed[:, 0] = True
            assert np.array_equal(changed, expected_changed)
            marker_values = np.unique(output[changed])
            assert len(marker_values) == 1
            marker = int(marker_values[0])
            assert marker not in (info["background"], info["foreground"])
            assert np.array_equal(output[~changed], grid[~changed])
            indicators.add(marker)
            if split == "train":
                train_directions.add(info["direction"])
    assert len(indicators) == 1
    assert train_directions == {"H", "V"}


def _same_puzzle(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[key], b[key])
                   for a, b in zip(first[split], second[split])
                   for key in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    dimensions, layouts, colors, combinations = set(), set(), set(), set()
    mixed_gaps = border_contact = cross_aspect = source_size = paired_mirror = 0
    train_solid = train_irregular = test_solid = test_irregular = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        training_signatures = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                info = _analyze_input(grid)
                marker = int(output[0, 0])
                prediction = grid.copy()
                if info["direction"] == "H":
                    prediction[0, :] = marker
                else:
                    prediction[:, 0] = marker
                assert np.array_equal(output, prediction)
                dimensions.add(grid.shape)
                colors.add((info["background"], info["foreground"], marker))
                layouts.add((info["direction"], info["count"], info["shape"], grid.shape,
                             info["gaps"]))
                combinations.add((split, info["direction"], info["count"], info["solid"],
                                  grid.shape[0] > grid.shape[1]))
                mixed_gaps += len(set(info["gaps"])) > 1
                border_contact += bool(np.any(grid[-1, :] == info["foreground"]) or
                                       np.any(grid[:, -1] == info["foreground"]))
                cross_aspect += ((info["direction"] == "H" and grid.shape[0] > grid.shape[1]) or
                                 (info["direction"] == "V" and grid.shape[1] > grid.shape[0]))
                source_size += grid.shape in ((11, 12), (12, 12))
                if split == "train":
                    train_solid += info["solid"]
                    train_irregular += not info["solid"] and len(info["shape"]) >= 7
                    training_signatures.add((info["shape"], info["count"], info["direction"]))
                else:
                    test_solid += info["solid"]
                    test_irregular += not info["solid"] and len(info["shape"]) >= 7
        paired_mirror += any((shape, count, "V") in training_signatures
                             for shape, count, direction in training_signatures if direction == "H")
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 35 and len(layouts) >= 150 and len(colors) >= 25
    assert len(combinations) >= 25
    assert mixed_gaps >= 10 and border_contact >= 20 and cross_aspect >= 12
    assert source_size >= 2 and paired_mirror >= 12
    assert min(train_solid, train_irregular, test_solid, test_irregular) >= 5
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (5, 2)):
        seed = 1000 + train_count * 10 + test_count
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        assert _same_puzzle(puzzle, generate_puzzle(seed, train_count, test_count))
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2026, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts were accepted")


_SOURCE_PAIRS = _source_pair_keys()
