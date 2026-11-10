"""Compositional examples of a marker growing through its background column."""

import hashlib

import numpy as np


# SHA-256 of (height, width, row-major uint8 cells) for the six source pairs.
SOURCE_PAIRS = {
    ("6b734deb032c534bc46a84566e75a79dfd88f95f473f18a8597f4e79e3830e5e", "856fbfda927183a81b77a524aeff6dda2cd68e0f5602f15486b38ef8feea2165"),
    ("8608534ee98cc17c696528801ca56bc4eea7693c315cb10a66fbc1a182d7c672", "f4e6dabecccbacdd4ac4fd9639ce7fc8d23dc63051204c82c4d41d073e391043"),
    ("ab953b6873cc6ff0f3eba539871a84a6dbf491029ceccc94aa52b51728970f81", "1c53bf9124ed80f58803fcaee306a6d2be40f22ea3e91bee8d5bfe7a0a70c23d"),
    ("7e05e0dd6ef9d1d38a1923b70be8049bb855845b66e87617fa941c147493a72e", "a354783b92757fdaea031077612624d66c9a08071b4fdc9a585e396959047621"),
    ("3323c6366f56763f3914f7f8fa1dd50ae21972b51dbe7fa25b5bcbeddf0be30b", "df56d24ebc47da3b9dafc0ff9df6e684fb622b5e6caca2904f0fb4813bf34df2"),
    ("9648bb332ee9d459cc00d6b49f9a49a6b7390105de20fe88e9a5f64acd2e6219", "4e76cef64276c039f8b1b73a7afb751c8908be5a0ad30d4f738293959e1d21c0"),
}


def _int(rng, low, high):
    return int(rng.integers(low, high + 1))


def _digest(grid):
    height, width = grid.shape
    return hashlib.sha256(
        bytes((height, width)) + grid.astype(np.uint8).tobytes()
    ).hexdigest()


def _scan(grid, background, marker):
    output = grid.copy()
    row, col = map(int, np.argwhere(grid == marker)[0])
    for direction in (-1, 1):
        r = row + direction
        while 0 <= r < grid.shape[0] and grid[r, col] == background:
            output[r, col] = marker
            r += direction
    return output


def _ring(grid, box, thickness, color, background):
    top, left, bottom, right = box
    grid[top:bottom + 1, left:right + 1] = color
    grid[
        top + thickness:bottom - thickness + 1,
        left + thickness:right - thickness + 1,
    ] = background


def _background(rng, policy, is_train, index, palette):
    if policy == "source":
        return 0
    if policy == "stable":
        return int(palette[0])
    if is_train and index == 0:
        return 0
    if is_train and index == 1:
        return int(palette[0])
    return 0 if _int(rng, 0, 1) else int(palette[_int(rng, 0, 3)])


def _role_colors(rng, background, palette, preferred_marker, preferred_wall,
                 depth, bar_count):
    available = [int(c) for c in palette if int(c) != background]
    if background != 0:
        available.append(0)
    if background != 0 and _int(rng, 0, 3) == 0:
        marker = 0
    elif preferred_marker != background and _int(rng, 0, 2) == 0:
        marker = preferred_marker
    else:
        marker = int(rng.choice(available))
    wall_pool = [c for c in available if c != marker]
    walls = []
    for index in range(max(1, depth + bar_count)):
        if index and _int(rng, 0, 2) == 0:
            color = walls[_int(rng, 0, index - 1)]
        elif preferred_wall in wall_pool and _int(rng, 0, 2) == 0:
            color = preferred_wall
        elif 0 in wall_pool and _int(rng, 0, 3) == 0:
            color = 0
        else:
            color = int(rng.choice(wall_pool))
        walls.append(color)
    return marker, walls


def _background_runs(grid, background, col):
    runs = []
    start = None
    for row in range(grid.shape[0] + 1):
        empty = row < grid.shape[0] and grid[row, col] == background
        if empty and start is None:
            start = row
        if not empty and start is not None:
            if row - start >= 3:
                runs.append((start, row - 1))
            start = None
    return runs


def _layout(rng, background, marker, walls, kind, depth, thicknesses,
            gaps, bar_count, placement, chamber):
    margins = [_int(rng, 0, 3) for _ in range(4)]
    if kind == "anchor":
        margins = [_int(rng, 0, 2) for _ in range(4)]
    top_margin, left_margin, bottom_margin, right_margin = margins
    min_height = top_margin + bottom_margin + 2 * sum(thicknesses)
    min_width = left_margin + right_margin + 2 * sum(thicknesses)
    if gaps:
        min_height += 2 * sum(gaps)
        min_width += 2 * sum(gaps)
    min_height = max(10, min_height + 5)
    min_width = max(10, min_width + 5)
    if min_height > 28 or min_width > 28:
        return None
    height = _int(rng, min_height, 28)
    if _int(rng, 0, 3) == 0 and height >= min_width:
        width = height
    else:
        width = _int(rng, min_width, 28)
    grid = np.full((height, width), background, dtype=np.int8)

    boxes = []
    if depth:
        box = (
            top_margin, left_margin,
            height - 1 - bottom_margin, width - 1 - right_margin,
        )
        boxes.append(box)
        for index in range(1, depth):
            inset = thicknesses[index - 1] + gaps[index - 1]
            previous = boxes[-1]
            boxes.append((
                previous[0] + inset, previous[1] + inset,
                previous[2] - inset, previous[3] - inset,
            ))
        inner = boxes[-1]
        inner_t = thicknesses[-1]
        if inner[2] - inner[0] < 2 * inner_t + 4 or inner[3] - inner[1] < 2 * inner_t + 4:
            return None
        col = _int(rng, inner[1] + inner_t + 1, inner[3] - inner_t - 1)
    else:
        col = _int(rng, 3, width - 4)

    openings = []
    for index, (box, thick) in enumerate(zip(boxes, thicknesses)):
        _ring(grid, box, thick, walls[index], background)
        if kind == "anchor":
            open_top = open_bottom = False
        else:
            open_top = _int(rng, 0, 3) == 0
            open_bottom = _int(rng, 0, 3) == 0
            if kind == "demo" and index == depth - 1:
                choice = _int(rng, 0, 2)
                open_top = choice != 1
                open_bottom = choice != 0
        top, left, bottom, right = box
        if open_top:
            grid[top:top + thick, col] = background
            openings.extend(range(top, top + thick))
        if open_bottom:
            grid[bottom - thick + 1:bottom + 1, col] = background
            openings.extend(range(bottom - thick + 1, bottom + 1))
        # A second slit is visually relevant but cannot change the marker's scan.
        if kind == "random" and _int(rng, 0, 3) == 0:
            off_columns = [c for c in range(left + thick, right - thick + 1)
                           if abs(c - col) >= 2]
            if off_columns:
                other = int(rng.choice(off_columns))
                if _int(rng, 0, 1):
                    grid[top:top + thick, other] = background
                else:
                    grid[bottom - thick + 1:bottom + 1, other] = background

    if depth:
        box = boxes[-1]
        low_row = box[0] + thicknesses[-1] + 1
        high_row = box[2] - thicknesses[-1] - 1
        low_col = box[1] + thicknesses[-1]
        high_col = box[3] - thicknesses[-1]
    else:
        low_row, high_row = 2, height - 3
        low_col, high_col = 2, width - 3

    placed_bars = 0
    for bar_index in range(bar_count):
        if high_row < low_row or high_col - low_col < 3:
            break
        row = _int(rng, low_row, high_row)
        include_marker_column = _int(rng, 0, 3) != 0
        if include_marker_column:
            start = max(low_col, col - _int(rng, 1, 4))
            end = min(high_col, col + _int(rng, 1, 4))
        else:
            if col - low_col >= 4 and (_int(rng, 0, 1) or high_col - col < 4):
                end = col - 2
                start = max(low_col, end - _int(rng, 2, 5))
            else:
                start = col + 2
                end = min(high_col, start + _int(rng, 2, 5))
        if end - start + 1 >= 3 and np.all(grid[row, start:end + 1] == background):
            grid[row, start:end + 1] = walls[depth + bar_index]
            placed_bars += 1
    if depth == 0 and placed_bars == 0:
        return None

    candidates = []
    for start, end in _background_runs(grid, background, col):
        if placement == "above":
            rows = [start] if start > 0 else []
        elif placement == "below":
            rows = [end] if end < height - 1 else []
        else:
            rows = list(range(start + 1, end))
        for row in rows:
            if kind == "demo":
                inner = boxes[-1]
                t = thicknesses[-1]
                if not (inner[0] + t <= row <= inner[2] - t):
                    continue
                if not any(start <= r <= end for r in openings):
                    continue
            if chamber == "between":
                inner = boxes[-1]
                outer = boxes[0]
                if not (outer[0] < row < outer[2] and
                        (row < inner[0] or row > inner[2])):
                    continue
            if chamber == "inside" and depth:
                inner = boxes[-1]
                t = thicknesses[-1]
                if not (inner[0] + t <= row <= inner[2] - t):
                    continue
            candidates.append(row)
    if not candidates:
        return None
    row = int(rng.choice(candidates))
    grid[row, col] = marker
    counts = np.bincount(grid.ravel(), minlength=10)
    if counts[background] <= max(counts[c] for c in range(10) if c != background):
        return None
    return grid, _scan(grid, background, marker)


def _make_pair(rng, background, palette, preferred_marker, preferred_wall,
               kind):
    for _ in range(120):
        if kind == "anchor":
            depth, thicknesses, gaps, bar_count = 1, [1], [], 0
            placement, chamber = "both", "inside"
        else:
            if kind == "demo":
                depth = _int(rng, 1, 2)
            else:
                depth = int(rng.choice([0, 1, 2, 3], p=[0.11, 0.34, 0.36, 0.19]))
            thicknesses = [_int(rng, 1, 3 if depth <= 2 else 2)
                           for _ in range(depth)]
            gaps = [_int(rng, 1, 3) for _ in range(max(0, depth - 1))]
            if kind == "demo":
                bar_count = 0
            else:
                bar_count = _int(rng, 1, 2) if depth == 0 else int(
                    rng.choice([0, 1, 2], p=[0.56, 0.33, 0.11])
                )
            placement = str(rng.choice(
                ["both", "above", "below"],
                p=[0.55, 0.225, 0.225],
            ))
            chamber = "inside"
            if kind == "random" and depth >= 2 and _int(rng, 0, 3) == 0:
                chamber = "between"
            if kind == "random" and depth == 0:
                chamber = "any"
        marker, walls = _role_colors(
            rng, background, palette, preferred_marker, preferred_wall,
            depth, bar_count,
        )
        result = _layout(
            rng, background, marker, walls, kind, depth, thicknesses,
            gaps, bar_count, placement, chamber,
        )
        if result is not None:
            return {"input": result[0], "output": result[1]}
    raise AssertionError("could not construct a valid pair")


def _valid_count(value, low, high, name):
    if (not isinstance(value, (int, np.integer)) or
            isinstance(value, (bool, np.bool_)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer from {low} through {high}")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is not None:
        _valid_count(num_train, 2, 10, "num_train")
    if num_test is not None:
        _valid_count(num_test, 1, 4, "num_test")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_test is None:
        num_test = _int(rng, 1, min(4, num_train - 1) if num_train is not None else 4)
    if num_train is None:
        num_train = _int(rng, max(2, num_test + 1), 10)

    palette = rng.permutation(np.arange(1, 10, dtype=np.int8))
    policy = str(rng.choice(["source", "stable", "mixed"], p=[0.25, 0.2, 0.55]))
    preferred_marker = int(palette[1])
    preferred_wall = int(palette[2])
    puzzle = {"train": [], "test": []}
    seen = set()

    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                kind = "anchor" if index == 0 else "demo" if index == 1 else "random"
            else:
                kind = "anchor" if _int(rng, 0, 5) == 0 else "random"
            background = _background(rng, policy, split == "train", index, palette)
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(
                    pair_rng, background, palette, preferred_marker, preferred_wall,
                    kind,
                )
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen and (_digest(pair["input"]), _digest(pair["output"])) not in SOURCE_PAIRS:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("duplicate pair construction")
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > max(counts[c] for c in range(10) if c != background)
    singletons = [c for c in range(10) if c != background and counts[c] == 1]
    assert len(singletons) == 1
    marker = singletons[0]
    assert any(counts[c] >= 3 for c in range(10) if c not in (background, marker))
    return background, marker


def _simple_closed(grid, background, marker):
    structure = (grid != background) & (grid != marker)
    points = np.argwhere(structure)
    if not len(points):
        return False
    top, left = map(int, points.min(axis=0))
    bottom, right = map(int, points.max(axis=0))
    color = int(grid[top, left])
    if color in (background, marker):
        return False
    expected = np.full(grid.shape, background, dtype=grid.dtype)
    expected[top, left:right + 1] = color
    expected[bottom, left:right + 1] = color
    expected[top:bottom + 1, left] = color
    expected[top:bottom + 1, right] = color
    row, col = map(int, np.argwhere(grid == marker)[0])
    expected[row, col] = marker
    return bool(top < row < bottom and left < col < right and
                np.array_equal(expected, grid))


def _crosses_opening(inp, out, background, marker):
    row, col = map(int, np.argwhere(inp == marker)[0])
    for r in range(inp.shape[0]):
        if r == row or inp[r, col] != background or out[r, col] != marker:
            continue
        left = np.count_nonzero((inp[r, :col] != background) &
                                (inp[r, :col] != marker))
        right = np.count_nonzero((inp[r, col + 1:] != background) &
                                 (inp[r, col + 1:] != marker))
        if left >= 2 and right >= 2:
            return True
    return False


def _observed_structure(grid, background, marker):
    """Recover visible rectangular rings, leaving any short bars behind."""
    remaining = (grid != background) & (grid != marker)
    rings = []
    while np.any(remaining):
        points = np.argwhere(remaining)
        top, left = map(int, points.min(axis=0))
        bottom, right = map(int, points.max(axis=0))
        if bottom - top < 4 or right - left < 4:
            break
        color = int(grid[top, left])
        if not all(np.all(grid[top:bottom + 1, side] == color)
                   for side in (left, right)):
            break
        if (np.count_nonzero(grid[top, left:right + 1] == color) <
                0.7 * (right - left + 1)):
            break
        if (np.count_nonzero(grid[bottom, left:right + 1] == color) <
                0.7 * (right - left + 1)):
            break
        thickness = 0
        while (top + thickness <= bottom - thickness and
               left + thickness <= right - thickness and
               grid[top + thickness, left + thickness] == color and
               grid[top + thickness, right - thickness] == color and
               grid[bottom - thickness, left + thickness] == color and
               grid[bottom - thickness, right - thickness] == color):
            thickness += 1
        if thickness == 0:
            break
        border = np.zeros_like(remaining)
        border[top:top + thickness, left:right + 1] = True
        border[bottom - thickness + 1:bottom + 1, left:right + 1] = True
        border[top:bottom + 1, left:left + thickness] = True
        border[top:bottom + 1, right - thickness + 1:right + 1] = True
        remaining[border] = False
        rings.append((top, left, bottom, right, thickness))
    return rings, remaining


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    inputs, pairs = set(), set()
    train_backgrounds = set()
    train_simple = train_opening = train_two_sided = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape
            input_key = (inp.shape, inp.astype(np.uint8).tobytes())
            pair_key = (input_key, out.astype(np.uint8).tobytes())
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            assert (_digest(inp), _digest(out)) not in SOURCE_PAIRS
            background, marker = _roles(inp)
            assert np.array_equal(out, _scan(inp, background, marker))
            assert not np.array_equal(inp, out)
            assert set(np.unique(inp)) == set(np.unique(out))
            structural = (inp != background) & (inp != marker)
            assert any(np.count_nonzero(structural[r]) >= 3 for r in range(inp.shape[0]))
            row, col = map(int, np.argwhere(inp == marker)[0])
            above = np.count_nonzero(out[:row, col] != inp[:row, col])
            below = np.count_nonzero(out[row + 1:, col] != inp[row + 1:, col])
            if split == "train":
                train_backgrounds.add(background)
                train_simple |= _simple_closed(inp, background, marker)
                train_opening |= _crosses_opening(inp, out, background, marker)
                train_two_sided |= above > 0 and below > 0
            else:
                # A nonzero background is demonstrated before it appears in a test.
                assert background == 0 or any(c != 0 for c in train_backgrounds)
    assert train_simple and train_opening and train_two_sided


def _same_puzzle(first, second):
    return all(
        np.array_equal(a[role], b[role])
        for split in ("train", "test")
        for a, b in zip(first[split], second[split])
        for role in ("input", "output")
    ) and all(len(first[s]) == len(second[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    puzzle_signatures, shapes, structures = set(), set(), set()
    one_sided_train = one_sided_test = edge_train = edge_test = 0
    nonzero_train = nonzero_test = marker_zero = wall_zero = 0
    mixed_background_puzzles = reused_marker_puzzles = reused_wall_puzzles = 0
    simple_test_puzzles = mixed_layout_puzzles = 0
    square_train = rectangle_train = square_test = rectangle_test = 0
    observed = {"train": {}, "test": {}}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signature = tuple(
            (pair["input"].shape, pair["input"].tobytes(), pair["output"].tobytes())
            for split in ("train", "test") for pair in puzzle[split]
        )
        puzzle_signatures.add(signature)
        backgrounds, markers, wall_sets, layout_types = [], [], [], []
        simple_test = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                background, marker = _roles(inp)
                backgrounds.append(background)
                markers.append(marker)
                wall_sets.append(set(map(int, np.unique(inp))) - {background, marker})
                nonzero = int(background != 0)
                if split == "train":
                    nonzero_train += nonzero
                else:
                    nonzero_test += nonzero
                marker_zero += int(marker == 0)
                wall_zero += int(background != 0 and marker != 0 and np.any(inp == 0))
                row, col = map(int, np.argwhere(inp == marker)[0])
                above = np.count_nonzero(out[:row, col] != inp[:row, col])
                below = np.count_nonzero(out[row + 1:, col] != inp[row + 1:, col])
                if (above == 0) != (below == 0):
                    if split == "train":
                        one_sided_train += 1
                    else:
                        one_sided_test += 1
                    if above == 0 and row > 0 and inp[row - 1, col] != background:
                        observed[split]["upper_stop"] = observed[split].get("upper_stop", 0) + 1
                    if below == 0 and row < inp.shape[0] - 1 and inp[row + 1, col] != background:
                        observed[split]["lower_stop"] = observed[split].get("lower_stop", 0) + 1
                changed = np.argwhere((inp != out))
                if np.any(changed[:, 0] == 0) or np.any(changed[:, 0] == inp.shape[0] - 1):
                    if split == "train":
                        edge_train += 1
                    else:
                        edge_test += 1
                square = inp.shape[0] == inp.shape[1]
                if split == "train":
                    square_train += int(square)
                    rectangle_train += int(not square)
                else:
                    square_test += int(square)
                    rectangle_test += int(not square)
                shapes.add(inp.shape)
                rings, remaining = _observed_structure(inp, background, marker)
                layout_types.append((len(rings), bool(np.any(remaining)),
                                     max((ring[4] for ring in rings), default=0)))
                if split == "test" and _simple_closed(inp, background, marker):
                    simple_test = True
                flags = {
                    "thick": any(ring[4] >= 2 for ring in rings),
                    "nested": len(rings) >= 2,
                    "three_frames": len(rings) >= 3,
                    "between": len(rings) >= 2 and not
                    (rings[-1][0] < row < rings[-1][2]),
                    "short_bar": bool(np.any(remaining)),
                    "bar_in_column": bool(np.any(remaining[:, col])),
                    "top_opening": False,
                    "bottom_opening": False,
                    "both_edges_open": False,
                    "off_column_opening": False,
                    "crossed_opening": _crosses_opening(inp, out, background, marker),
                }
                for top, left, bottom, right, thickness in rings:
                    top_open = bool(np.all(inp[top:top + thickness, col] == background))
                    bottom_open = bool(np.all(
                        inp[bottom - thickness + 1:bottom + 1, col] == background
                    ))
                    flags["top_opening"] |= top_open
                    flags["bottom_opening"] |= bottom_open
                    flags["both_edges_open"] |= top_open and bottom_open
                    flags["off_column_opening"] |= any(
                        np.all(inp[top:top + thickness, c] == background) or
                        np.all(inp[bottom - thickness + 1:bottom + 1, c] == background)
                        for c in range(left + thickness, right - thickness + 1)
                        if abs(c - col) >= 2
                    )
                for name, present in flags.items():
                    if present:
                        observed[split][name] = observed[split].get(name, 0) + 1
                # Geometry without role colors or empty exterior margins.
                structural = (inp != background) & (inp != marker)
                points = np.argwhere(structural)
                top, left = points.min(axis=0)
                bottom, right = points.max(axis=0)
                structures.add((inp[top:bottom + 1, left:right + 1] != background).tobytes())
        mixed_background_puzzles += int(len(set(backgrounds)) > 1)
        reused_marker_puzzles += int(len(set(markers)) < len(markers))
        reused_wall_puzzles += int(any(
            wall_sets[i] & wall_sets[j]
            for i in range(len(wall_sets)) for j in range(i)
        ))
        simple_test_puzzles += int(simple_test)
        mixed_layout_puzzles += int(len(set(layout_types)) > 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_signatures) >= 40 and len(shapes) >= 35
    assert len(structures) >= 150
    assert min(one_sided_train, one_sided_test, edge_train, edge_test) >= 1
    assert min(nonzero_train, nonzero_test, marker_zero, wall_zero) >= 1
    assert min(square_train, rectangle_train, square_test, rectangle_test) >= 1
    assert min(mixed_background_puzzles, reused_marker_puzzles,
               reused_wall_puzzles, simple_test_puzzles,
               mixed_layout_puzzles) >= 1
    requested_features = (
        "upper_stop", "lower_stop", "thick", "nested", "three_frames",
        "between", "short_bar", "bar_in_column", "top_opening",
        "bottom_opening", "both_edges_open", "off_column_opening",
        "crossed_opening",
    )
    for split in ("train", "test"):
        assert all(observed[split].get(name, 0) > 0 for name in requested_features)

    cases = ((101, 2, 1), (102, 10, 1), (103, 10, 4),
             (104, None, 1), (105, None, 4), (106, 2, None),
             (107, 10, None))
    for seed, train_count, test_count in cases:
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed, train_count, test_count))
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(999, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError((train_count, test_count))
