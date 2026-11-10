import numpy as np


_DIRECTIONS = ("up", "right", "down", "left")
_FORMS = ("rectangle", "edge_block", "taper", "segment", "bulge")


def _integer_count(value, lo, hi, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


def _counts(rng, num_train, num_test):
    num_train = _integer_count(num_train, 2, 10, "num_train")
    num_test = _integer_count(num_test, 1, 4, "num_test")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
    elif num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _profile_mask(widths):
    width = max(widths)
    mask = np.zeros((len(widths), width), dtype=bool)
    for row, span in enumerate(widths):
        start = (width - span) // 2
        mask[row, start:start + span] = True
    return mask


def _shape(rng, form, direction):
    if form in ("rectangle", "edge_block"):
        if direction in ("up", "down"):
            height = int(rng.integers(2, 10))
            width = int(rng.choice((3, 5, 7, 9)))
        else:
            height = int(rng.choice((3, 5, 7, 9)))
            width = int(rng.integers(2, 10))
        return np.ones((height, width), dtype=bool)
    if form == "segment":
        length = int(rng.choice((3, 5, 7, 9, 11)))
        return np.ones((1, length) if direction in ("up", "down") else (length, 1), dtype=bool)
    if form == "taper":
        base = int(rng.choice((5, 7, 9, 11)))
        depth = int(rng.integers(3, 9))
        widths = [base]
        for _ in range(depth - 2):
            widths.append(int(rng.choice(np.arange(1, widths[-1] + 1, 2))))
        widths.append(1)
        mask = _profile_mask(widths)
    elif form == "bulge":
        peak = int(rng.choice((5, 7, 9, 11)))
        half_length = int(rng.integers(2, 5))
        outer = int(rng.choice((1, 3)))
        interior = sorted(int(rng.choice(np.arange(outer, peak + 1, 2)))
                          for _ in range(half_length - 2))
        half = [outer] + interior + [peak]
        widths = half + ([peak] if int(rng.integers(0, 2)) else []) + half[-2::-1]
        mask = _profile_mask(widths)
    else:
        raise ValueError("unknown form")
    turns = {"down": 0, "right": 1, "up": 2, "left": 3}[direction]
    return np.rot90(mask, turns).copy()


def _ray_offsets(mask):
    """Every visible central exit, including both centers of an even span."""
    height, width = mask.shape
    rays = []
    for col in sorted({(width - 1) // 2, width // 2}):
        rows = np.flatnonzero(mask[:, col])
        rays.append((0, int(rows[0]), col))
        rays.append((2, height - 1 - int(rows[-1]), col))
    for row in sorted({(height - 1) // 2, height // 2}):
        cols = np.flatnonzero(mask[row])
        rays.append((3, int(cols[0]), row))
        rays.append((1, width - 1 - int(cols[-1]), row))
    return rays


def _layout(rng, mask, direction, form, geometry, spacing):
    height, width = mask.shape
    target = _DIRECTIONS.index(direction)
    other = [i for i in range(4) if i != target]
    offsets = _ray_offsets(mask)
    touched = int(rng.choice(other)) if form == "edge_block" else None
    if spacing == "short":
        lengths = list(rng.permutation(np.arange(1, 4)))
    elif spacing == "close":
        lengths = list(rng.permutation(np.arange(2, 12)))
    else:
        lengths = list(rng.permutation(np.arange(5, 14)))
    for gap in lengths:
        candidates = []
        for a in range(gap):
            for b in range(gap):
                for c in range(gap):
                    margins = [0, 0, 0, 0]
                    margins[target] = int(gap)
                    for side, value in zip(other, (a, b, c)):
                        margins[side] = value
                    if touched is not None and margins[touched] != 0:
                        continue
                    rays = sorted((margins[side] + offset, side)
                                  for side, offset, _ in offsets)
                    if rays[-1][1] != target or rays[-1][0] <= rays[-2][0]:
                        continue
                    difference = rays[-1][0] - rays[-2][0]
                    if spacing == "close" and difference not in (1, 2):
                        continue
                    if spacing == "wide" and difference < 4:
                        continue
                    rows = height + margins[0] + margins[2]
                    cols = width + margins[1] + margins[3]
                    if rows > 30 or cols > 30 or 2 * int(mask.sum()) >= rows * cols:
                        continue
                    if geometry == "square" and rows != cols:
                        continue
                    if geometry == "off_axis" and not (
                        (direction in ("left", "right") and cols < rows)
                        or (direction in ("up", "down") and rows < cols)
                    ):
                        continue
                    candidates.append(tuple(margins))
        if candidates:
            return candidates[int(rng.integers(len(candidates)))]
    return None


def _colors(rng, used, index):
    background = 0 if rng.random() < 0.42 else int(rng.integers(1, 10))
    if index == 1 and background == used[0][0]:
        background = (background + int(rng.integers(1, 10))) % 10
    if background != 0 and rng.random() < 0.28:
        foreground = 0
    elif index >= 3 and rng.random() < 0.38:
        foreground = used[int(rng.integers(len(used)))][1]
        if foreground == background:
            foreground = int(rng.choice([c for c in range(10) if c != background]))
    else:
        foreground = int(rng.choice([c for c in range(10) if c != background]))
    return background, foreground


def _make_pair(mask, margins, background, foreground, direction):
    up, right, down, left = margins
    grid = np.full((up + mask.shape[0] + down, left + mask.shape[1] + right),
                   background, dtype=np.int8)
    area = grid[up:up + mask.shape[0], left:left + mask.shape[1]]
    area[mask] = foreground
    output = grid.copy()
    if direction == "up":
        output[:up, left + mask.shape[1] // 2] = foreground
    elif direction == "down":
        output[up + mask.shape[0]:, left + mask.shape[1] // 2] = foreground
    elif direction == "left":
        output[up + mask.shape[0] // 2, :left] = foreground
    else:
        output[up + mask.shape[0] // 2, left + mask.shape[1]:] = foreground
    return {"input": grid, "output": output}


def _grid_key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.uint8).tobytes()


def _source_keys():
    specifications = (
        (5, 14, 3, (1, 4, 1, 4)),
        (14, 7, 2, (1, 7, 1, 6)),
        (12, 8, 6, (1, 7, 1, 8)),
        (6, 9, 4, (1, 4, 1, 2)),
    )
    keys = set()
    for rows, cols, color, (r0, r1, c0, c1) in specifications:
        grid = np.zeros((rows, cols), dtype=np.int8)
        grid[r0:r1, c0:c1] = color
        keys.add(_grid_key(grid))
    taper = np.zeros((11, 7), dtype=np.int8)
    for row, span in enumerate((7, 5, 3, 1), start=2):
        taper[row, (7 - span) // 2:(7 + span) // 2] = 1
    keys.add(_grid_key(taper))
    return keys


_SOURCE_INPUT_KEYS = _source_keys()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _counts(rng, num_train, num_test)
    # The visible longest-ray rule is the only puzzle-level convention.
    source_like = bool(rng.random() < 0.28)
    if source_like:
        directions = ["right", "down"]
    else:
        directions = [str(x) for x in
                      rng.permutation(_DIRECTIONS)[:min(n_train, int(rng.integers(2, 5)))]]
    train_directions = directions + [str(rng.choice(directions))
                                     for _ in range(n_train - len(directions))]
    rng.shuffle(train_directions)
    test_directions = [str(rng.choice(directions)) for _ in range(n_test)]
    if source_like:
        train_forms = ["rectangle"] * n_train
    else:
        train_forms = ["rectangle"]
        train_forms += [str(rng.choice(_FORMS)) for _ in range(n_train - 1)]
    rng.shuffle(train_forms)
    counterexample_index = train_forms.index("rectangle")
    challenging = ["edge_block", "taper", "segment", "bulge"]
    rng.shuffle(challenging)
    test_forms = challenging[:n_test]
    if n_test >= 2 and rng.random() < 0.35:
        test_forms[-1] = "rectangle"
    puzzle = {"train": [], "test": []}
    used_colors = []
    seen = set()
    schedule = ([("train", form, direction) for form, direction
                 in zip(train_forms, train_directions)]
                + [("test", form, direction) for form, direction
                   in zip(test_forms, test_directions)])
    for index, (split, form, direction) in enumerate(schedule):
        background, foreground = _colors(rng, used_colors, index)
        used_colors.append((background, foreground))
        for attempt in range(36):
            shape = _shape(rng, form, direction)
            geometry = str(rng.choice(("ordinary", "square", "off_axis"),
                                      p=(0.38, 0.30, 0.32)))
            spacing = str(rng.choice(("short", "close", "wide"),
                                     p=(0.18, 0.43, 0.39)))
            if index == counterexample_index:
                geometry, spacing = "off_axis", "close"
            if split == "test" and attempt == 0 and rng.random() < 0.45:
                spacing = "close"
            margins = _layout(rng, shape, direction, form, geometry, spacing)
            if margins is None:
                continue
            pair = _make_pair(shape, margins, background, foreground, direction)
            key = _grid_key(pair["input"])
            if key not in seen and key not in _SOURCE_INPUT_KEYS:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("unable to construct a distinct valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= length <= 30 for length in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    grid, output = pair["input"], pair["output"]
    _check_grid(grid)
    _check_grid(output)
    assert grid.shape == output.shape
    assert _grid_key(grid) not in _SOURCE_INPUT_KEYS
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2 and counts[0] != counts[1]
    background = int(colors[int(np.argmax(counts))])
    foreground = int(colors[int(np.argmin(counts))])
    assert set(map(int, np.unique(output))) == {background, foreground}
    mask = grid == foreground
    coordinates = np.argwhere(mask)
    r0, c0 = map(int, coordinates.min(axis=0))
    r1, c1 = map(int, coordinates.max(axis=0))
    occupied = set(map(tuple, coordinates.tolist()))
    reached = {next(iter(occupied))}
    stack = list(reached)
    while stack:
        r, c = stack.pop()
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if neighbor in occupied and neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    assert reached == occupied
    h, w = r1 - r0 + 1, c1 - c0 + 1
    gaps = {"up": r0, "right": grid.shape[1] - c1 - 1,
            "down": grid.shape[0] - r1 - 1, "left": c0}
    cropped = mask[r0:r1 + 1, c0:c1 + 1]
    rays = sorted((gaps[_DIRECTIONS[side]] + offset, side, center)
                  for side, offset, center in _ray_offsets(cropped))
    maximum, winning_side, center = rays[-1]
    assert maximum >= 1 and maximum > rays[-2][0]
    direction = _DIRECTIONS[winning_side]
    assert (w if direction in ("up", "down") else h) % 2 == 1
    profiles = cropped if direction in ("up", "down") else cropped.T
    for row in profiles:
        locations = np.flatnonzero(row)
        assert len(locations) > 0
        assert locations[-1] - locations[0] + 1 == len(locations)
        assert locations[0] + locations[-1] == len(row) - 1
        assert row[len(row) // 2]
    expected = grid.copy()
    if direction == "up":
        expected[:maximum, c0 + center] = foreground
    elif direction == "down":
        expected[grid.shape[0] - maximum:, c0 + center] = foreground
    elif direction == "left":
        expected[r0 + center, :maximum] = foreground
    else:
        expected[r0 + center, grid.shape[1] - maximum:] = foreground
    assert np.array_equal(output, expected)
    assert int(np.count_nonzero(output != grid)) == maximum
    rectangle = bool(cropped.all())
    profile = tuple(int(row.sum()) for row in profiles)
    return {"direction": direction, "gap": maximum,
            "margin": maximum - rays[-2][0],
            "background": background, "foreground": foreground,
            "rectangle": rectangle, "segment": rectangle and (h == 1 or w == 1),
            "edge": tuple(side for side, gap in gaps.items() if gap == 0),
            "profile": profile, "shape": grid.shape,
            "off_axis": (grid.shape[1] < grid.shape[0] if direction in ("left", "right")
                         else grid.shape[0] < grid.shape[1])}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen_inputs, seen_pairs = set(), set()
    train_directions, test_directions = set(), set()
    train_backgrounds = set()
    simple_train = False
    counterexample_train = False
    challenging_test = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if split == "train":
                train_directions.add(info["direction"])
                train_backgrounds.add(info["background"])
                simple_train |= info["rectangle"] and not info["segment"]
                counterexample_train |= (info["rectangle"] and not info["segment"]
                                         and info["off_axis"] and info["margin"] <= 2)
            else:
                test_directions.add(info["direction"])
                challenging_test |= not (info["rectangle"] and not info["edge"]
                                         and not info["segment"])
    assert len(train_directions) >= 2
    assert test_directions <= train_directions
    assert len(train_backgrounds) >= 2
    assert simple_train
    assert counterexample_train
    assert challenging_test


def _same_puzzle(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[role], b[role]) for a, b in zip(first[split], second[split])
        for role in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, dimensions, structures = set(), set(), set(), set()
    directions = {"train": set(), "test": set()}
    close, short, square, off_axis, nonzero_background = 0, 0, 0, 0, 0
    zero_foreground, repeated_foreground, mixed_train_forms = 0, 0, 0
    taper_directions, segment_directions, edge_sides = set(), set(), set()
    plain_test_rectangles = 0
    puzzle_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        foregrounds = []
        train_profiles = set()
        full_key = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                directions[split].add(info["direction"])
                dimensions.add(info["shape"])
                structures.add((info["shape"], info["profile"], info["direction"], info["gap"]))
                close += info["margin"] <= 2
                short += info["gap"] <= 3
                square += info["shape"][0] == info["shape"][1]
                off_axis += info["off_axis"]
                nonzero_background += info["background"] != 0
                zero_foreground += info["foreground"] == 0
                foregrounds.append(info["foreground"])
                edge_sides.update(info["edge"])
                if split == "train":
                    train_profiles.add((info["rectangle"], info["segment"], info["profile"]))
                elif info["rectangle"] and not info["edge"] and not info["segment"]:
                    plain_test_rectangles += 1
                if info["segment"]:
                    segment_directions.add(info["direction"])
                elif not info["rectangle"] and info["profile"][0] != info["profile"][-1]:
                    taper_directions.add(info["direction"])
                full_key.append((_grid_key(pair["input"]), _grid_key(pair["output"])))
        repeated_foreground += len(set(foregrounds)) < len(foregrounds)
        mixed_train_forms += len(train_profiles) > 1
        puzzle_keys.add(tuple(full_key))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 30 and len(structures) >= 100
    assert len(puzzle_keys) == 42
    assert directions["train"] == directions["test"] == set(_DIRECTIONS)
    assert taper_directions == segment_directions == edge_sides == set(_DIRECTIONS)
    assert min(close, short, square, off_axis, nonzero_background, zero_foreground) >= 8
    assert repeated_foreground >= 5 and mixed_train_forms >= 15
    assert plain_test_rectangles >= 2
    count_cases = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None), (4, 2))
    for index, counts in enumerate(count_cases):
        first = generate_puzzle(1000 + index, *counts)
        second = generate_puzzle(1000 + index, *counts)
        assert _same_puzzle(first, second)
        validate_generated_puzzle(first)
        if counts[0] is not None:
            assert len(first["train"]) == counts[0]
        if counts[1] is not None:
            assert len(first["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (1, 1), (10, 10)):
        try:
            generate_puzzle(99, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {counts}")
