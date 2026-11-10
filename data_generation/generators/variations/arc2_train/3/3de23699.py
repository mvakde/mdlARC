import numpy as np


_SOURCE_EXACT_KEYS = frozenset(
    {
        (
            "00000000000000/"
            "00000000000000/"
            "00000000000000/"
            "00000000000000/"
            "00000000000000/"
            "00060000600000/"
            "00000800000000/"
            "00000800000000/"
            "00008880000000/"
            "00000880000000/"
            "00060000600000/"
            "00000000000000",
            "0600/0600/6660/0660",
        ),
        (
            "000000000/"
            "030000030/"
            "000220000/"
            "000220200/"
            "002002000/"
            "030000030/"
            "000000000/"
            "000000000/"
            "000000000/"
            "000000000/"
            "000000000/"
            "000000000",
            "03300/03303/30030",
        ),
        (
            "000000000000000000/"
            "008000000008000000/"
            "000000440000000000/"
            "000004400000000000/"
            "000000000400000000/"
            "000000040400000000/"
            "008000000008000000/"
            "000000000000000000/"
            "000000000000000000/"
            "000000000000000000/"
            "000000000000000000/"
            "000000000000000000",
            "00088000/00880000/00000080/00008080",
        ),
        (
            "0000000/"
            "0400040/"
            "0002000/"
            "0022200/"
            "0002200/"
            "0400040/"
            "0000000",
            "040/444/044",
        ),
        (
            "0000000000000000000/"
            "0000000000000000000/"
            "0020000000000200000/"
            "0000000000000000000/"
            "0000300003300000000/"
            "0000300000000000000/"
            "0000003000000000000/"
            "0000003330000000000/"
            "0000000030030000000/"
            "0000000000030000000/"
            "0000000330000000000/"
            "0020000000000200000/"
            "0000000000000000000/"
            "0000000000000000000/"
            "0000000000000000000/"
            "0000000000000000000/"
            "0000000000000000000/"
            "0000000000000000000",
            "0000000000/"
            "0200002200/"
            "0200000000/"
            "0002000000/"
            "0002220000/"
            "0000020020/"
            "0000000020/"
            "0000220000",
        ),
    }
)


def _validate_optional_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _choose_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(
        num_train, "num_train", 2, 10
    )
    test_count = _validate_optional_count(
        num_test, "num_test", 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        # These formulas guarantee broad count diversity for consecutive seeds.
        test_count = 1 + seed % 4
        number_of_train_choices = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed // 4) % number_of_train_choices
        )
    elif train_count is None:
        minimum_train = test_count + 1
        if minimum_train > 10:
            raise ValueError("no legal num_train exists")
        train_count = int(rng.integers(minimum_train, 11))
    else:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists")
        test_count = int(rng.integers(1, maximum_test + 1))

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError("illegal train/test count combination")

    return train_count, test_count


def _choose_other(rng: np.random.Generator, excluded: tuple[int, ...]) -> int:
    return int(rng.choice([color for color in range(10) if color not in excluded]))


def _sample_shape(rng: np.random.Generator, split: str, source_shift: bool) -> tuple[int, int]:
    if source_shift:
        if split == "train":
            return int(rng.integers(3, 5)), int(rng.integers(3, 9))
        return int(rng.integers(5, 15)), int(rng.integers(9, 19))

    choice = float(rng.random())
    if choice < 0.07:
        return 28, 28
    if choice < 0.22:
        long = int(rng.integers(12, 29))
        thin = int(rng.integers(1, 3))
        return (thin, long) if rng.random() < 0.5 else (long, thin)

    def axis() -> int:
        x = float(rng.random())
        if x < (0.29 if split == "train" else 0.17):
            return int(rng.integers(1, 5))
        if x < (0.66 if split == "train" else 0.48):
            return int(rng.integers(3, 12))
        if x < 0.88:
            return int(rng.integers(10, 23))
        return int(rng.integers(20, 29))

    return axis(), axis()


def _connected_mask(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    row, col = int(rng.integers(height)), int(rng.integers(width))
    mask[row, col] = True
    target = int(rng.integers(1, max(2, int(height * width * 0.7) + 1)))
    frontier = [(row, col)]
    while int(mask.sum()) < target and frontier:
        row, col = frontier[int(rng.integers(len(frontier)))]
        neighbors = [(row + dr, col + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if 0 <= row + dr < height and 0 <= col + dc < width and not mask[row + dr, col + dc]]
        if neighbors:
            row, col = neighbors[int(rng.integers(len(neighbors)))]
            mask[row, col] = True
            frontier.append((row, col))
        else:
            frontier.remove((row, col))
    return mask


def _sample_mask(rng: np.random.Generator, height: int, width: int, profile: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mode = int(rng.choice(9, p=[0.24, 0.16, 0.13, 0.11, 0.07, 0.07, 0.10, 0.07, 0.05]))
    if mode == 0:
        mask = rng.random((height, width)) < float(rng.uniform(0.08, 0.78))
    elif mode == 1:
        mask = _connected_mask(rng, height, width)
    elif mode == 2:
        for _ in range(int(rng.integers(1, 5))):
            if rng.random() < 0.5:
                row = int(rng.integers(height))
                start = int(rng.integers(width))
                end = int(rng.integers(start + 1, width + 1))
                mask[row, start:end] = True
            else:
                col = int(rng.integers(width))
                start = int(rng.integers(height))
                end = int(rng.integers(start + 1, height + 1))
                mask[start:end, col] = True
    elif mode == 3:
        for _ in range(int(rng.integers(2, 7))):
            row, col = int(rng.integers(height)), int(rng.integers(width))
            for _ in range(int(rng.integers(1, 9))):
                mask[row, col] = True
                dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                row, col = min(max(row + dr, 0), height - 1), min(max(col + dc, 0), width - 1)
    elif mode == 4:
        mask[int(rng.integers(height)), int(rng.integers(width))] = True
    elif mode == 5:
        mask[:] = True
    elif mode == 6:
        if rng.random() < 0.5:
            mask[int(rng.integers(height)), :] = True
        else:
            mask[:, int(rng.integers(width))] = True
    elif mode == 7 and height >= 2 and width >= 2:
        rows = sorted(int(v) for v in rng.choice(height, 2, replace=False))
        cols = sorted(int(v) for v in rng.choice(width, 2, replace=False))
        mask[np.ix_(rows, cols)] = True
    else:
        r0, r1 = sorted(int(v) for v in rng.choice(height + 1, 2, replace=False))
        c0, c1 = sorted(int(v) for v in rng.choice(width + 1, 2, replace=False))
        mask[r0:r1, c0:c1] = True

    if profile == "padded":
        row_low, row_high, col_low, col_high = 0, height, 0, width
        sides = [0, 1, 2, 3]
        rng.shuffle(sides)
        for side in sides[:int(rng.integers(1, 4))]:
            if side == 0 and row_high - row_low > 1:
                row_low += int(rng.integers(1, min(3, row_high - row_low - 1) + 1))
            elif side == 1 and row_high - row_low > 1:
                row_high -= int(rng.integers(1, min(3, row_high - row_low - 1) + 1))
            elif side == 2 and col_high - col_low > 1:
                col_low += int(rng.integers(1, min(3, col_high - col_low - 1) + 1))
            elif side == 3 and col_high - col_low > 1:
                col_high -= int(rng.integers(1, min(3, col_high - col_low - 1) + 1))
        allowed = np.zeros_like(mask)
        allowed[row_low:row_high, col_low:col_high] = True
        mask &= allowed
        if not mask.any():
            positions = np.argwhere(allowed)
            row, col = positions[int(rng.integers(len(positions)))]
            mask[row, col] = True
    elif profile == "edge":
        mask[0, int(rng.integers(width))] = True
        mask[-1, int(rng.integers(width))] = True
        mask[int(rng.integers(height)), 0] = True
        mask[int(rng.integers(height)), -1] = True
    elif profile == "partial":
        if width >= 3:
            mask[0, :] = False
            mask[0, int(rng.integers(width))] = True
    if not mask.any():
        mask[int(rng.integers(height)), int(rng.integers(width))] = True
    return mask


def _sample_margins(rng: np.random.Generator, interior: int) -> tuple[int, int]:
    available = 28 - interior
    if available == 0 or rng.random() < 0.22:
        return 0, 0
    choice = float(rng.random())
    if choice < 0.5:
        total = int(rng.integers(1, min(available, 5) + 1))
    elif choice < 0.85:
        total = int(rng.integers(1, min(available, 16) + 1))
    else:
        total = int(rng.integers(1, available + 1))
    first = int(rng.integers(total + 1))
    if rng.random() < 0.18:
        first = 0 if rng.random() < 0.5 else total
    return first, total - first


def _make_pair(rng: np.random.Generator, shape: tuple[int, int], colors: tuple[int, int, int], profile: str,
               allow_axis_swap: bool) -> dict:
    height, width = shape
    background, marker, payload = colors
    mask = _sample_mask(rng, height, width, profile)
    if rng.random() < 0.17:
        margins = (0, 0, 0, 0)
    else:
        margins = (*_sample_margins(rng, height), *_sample_margins(rng, width))
    top, bottom, left, right = margins
    input_grid = np.full((top + height + 2 + bottom, left + width + 2 + right), background, dtype=np.int8)
    marker_bottom, marker_right = top + height + 1, left + width + 1
    for row, col in ((top, left), (top, marker_right), (marker_bottom, left), (marker_bottom, marker_right)):
        input_grid[row, col] = marker
    crop = input_grid[top + 1:marker_bottom, left + 1:marker_right]
    crop[mask] = payload
    output_grid = np.where(mask, marker, background).astype(np.int8)
    quarter_turns = int(rng.integers(4)) if allow_axis_swap else 2 * int(rng.integers(2))
    input_grid = np.rot90(input_grid, quarter_turns)
    output_grid = np.rot90(output_grid, quarter_turns)
    if rng.random() < 0.5:
        input_grid = np.fliplr(input_grid)
        output_grid = np.fliplr(output_grid)
    input_grid = np.ascontiguousarray(input_grid)
    output_grid = np.ascontiguousarray(output_grid)
    return {"input": input_grid, "output": output_grid}


def _array_key(grid: np.ndarray) -> tuple:
    normalized = np.ascontiguousarray(
        grid,
        dtype=np.int16,
    )
    return tuple(grid.shape), normalized.tobytes()


def _grid_text(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _is_source_duplicate(pair: dict) -> bool:
    key = (
        _grid_text(pair["input"]),
        _grid_text(pair["output"]),
    )
    return key in _SOURCE_EXACT_KEYS


def _decode_input(input_grid: np.ndarray) -> dict:
    colors = [int(value) for value in np.unique(input_grid)]
    assert len(colors) == 3
    candidates = []
    for marker in colors:
        coordinates = np.argwhere(input_grid == marker)
        if len(coordinates) != 4:
            continue
        rows = sorted(int(value) for value in np.unique(coordinates[:, 0]))
        cols = sorted(int(value) for value in np.unique(coordinates[:, 1]))
        if len(rows) != 2 or len(cols) != 2:
            continue
        top, bottom = rows
        left, right = cols
        if bottom - top < 2 or right - left < 2:
            continue
        corners = {(top, left), (top, right), (bottom, left), (bottom, right)}
        if {(int(row), int(col)) for row, col in coordinates} != corners:
            continue
        for payload in colors:
            if payload == marker:
                continue
            positions = np.argwhere(input_grid == payload)
            if not len(positions):
                continue
            if not (np.all((positions[:, 0] > top) & (positions[:, 0] < bottom))
                    and np.all((positions[:, 1] > left) & (positions[:, 1] < right))):
                continue
            background = next(color for color in colors if color not in (marker, payload))
            crop = input_grid[top + 1:bottom, left + 1:right]
            if not set(int(value) for value in np.unique(crop)) <= {background, payload}:
                continue
            candidates.append((background, marker, payload, top, bottom, left, right))
    assert len(candidates) == 1, "marker rectangle must uniquely enclose the payload"
    background, marker, payload, top, bottom, left, right = candidates[0]
    crop = input_grid[top + 1:bottom, left + 1:right]
    return {
        "background": background, "marker": marker, "payload": payload,
        "top": top, "bottom": bottom, "left": left, "right": right,
        "margins": (top, input_grid.shape[0] - bottom - 1,
                    left, input_grid.shape[1] - right - 1),
        "mask": crop == payload,
        "expected": np.where(crop == payload, marker, background).astype(np.int8),
    }


def _analyze_pair(pair: dict) -> dict:
    information = _decode_input(pair["input"])
    assert np.array_equal(pair["output"], information["expected"])
    assert pair["output"].shape == information["mask"].shape
    assert bool(information["mask"].any())
    return information


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            input_key = _array_key(pair["input"])
            pair_key = (input_key, _array_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_duplicate(pair)
            info = _analyze_pair(pair)
            assert pair["input"].shape[0] >= pair["output"].shape[0] + 2
            assert pair["input"].shape[1] >= pair["output"].shape[1] + 2
            analyses[split].append(info)

    train = analyses["train"]
    assert any(first["payload"] == second["payload"] and first["marker"] != second["marker"]
               for index, first in enumerate(train) for second in train[index + 1:])
    assert any(any(not info["mask"][side].any() for side in
                   (0, -1)) or any(not info["mask"][:, side].any() for side in (0, -1))
               for info in train)
    assert any(all((info["mask"][0].any(), info["mask"][-1].any(),
                    info["mask"][:, 0].any(), info["mask"][:, -1].any()))
               for info in train)
    # The rule has no hidden palette or scale convention. Each test input uniquely
    # identifies its three roles; training establishes what to do with those roles.
    for info in analyses["test"]:
        assert info["background"] != info["marker"] != info["payload"]


def _palette_plan(rng: np.random.Generator, train_count: int, test_count: int,
                  profiles: list[str]) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    witness_payload = int(rng.integers(10))
    markers = [int(value) for value in rng.choice(
        [color for color in range(10) if color != witness_payload], 2, replace=False)]
    padded_index = profiles.index("padded")
    edge_index = profiles.index("edge")
    train_colors = []
    for _ in range(train_count):
        colors = tuple(int(value) for value in rng.choice(10, 3, replace=False))
        train_colors.append(colors)
    first_background = _choose_other(rng, (markers[0], witness_payload))
    second_background = _choose_other(rng, (markers[1], witness_payload, first_background))
    train_colors[padded_index] = (first_background, markers[0], witness_payload)
    train_colors[edge_index] = (second_background, markers[1], witness_payload)
    extra = [index for index in range(train_count) if index not in (padded_index, edge_index)]
    if extra and rng.random() < 0.72:
        index = extra[int(rng.integers(len(extra)))]
        train_colors[index] = (_choose_other(rng, (witness_payload, markers[0])),
                               witness_payload, markers[0])
    if extra and len({triple[2] for triple in train_colors}) == 1:
        index = extra[0]
        background, marker, _ = train_colors[index]
        train_colors[index] = (background, marker,
                               _choose_other(rng, (background, marker, witness_payload)))

    seen_colors = {color for triple in train_colors for color in triple}
    test_colors = []
    for _ in range(test_count):
        choice = float(rng.random())
        if choice < 0.28:
            marker, payload = witness_payload, markers[int(rng.integers(2))]
            background = _choose_other(rng, (marker, payload))
            colors = (background, marker, payload)
        elif choice < 0.55 and len(seen_colors) < 10:
            novel = int(rng.choice([color for color in range(10) if color not in seen_colors]))
            others = [int(value) for value in rng.choice(
                [color for color in range(10) if color != novel], 2, replace=False)]
            colors = tuple((others[0], novel, others[1]) if rng.random() < 0.5
                           else (others[0], others[1], novel))
        else:
            colors = tuple(int(value) for value in rng.choice(10, 3, replace=False))
        test_colors.append(colors)
    if len({triple[2] for triple in train_colors + test_colors}) == 1:
        background, marker, _ = test_colors[0]
        test_colors[0] = (background, marker,
                          _choose_other(rng, (background, marker, witness_payload)))
    return train_colors, test_colors


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, seed, num_train, num_test)
    # This selects a source-like size shift for some puzzles, not a learned rule.
    source_shift = bool(rng.random() < 0.22)
    profiles = ["free"] * train_count
    witness_slots = rng.choice(train_count, 2, replace=False)
    profiles[int(witness_slots[0])] = "padded"
    profiles[int(witness_slots[1])] = "edge"
    for index in range(train_count):
        if profiles[index] == "free" and rng.random() < 0.16:
            profiles[index] = "partial"
    train_colors, test_colors = _palette_plan(rng, train_count, test_count, profiles)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for split, colors_list in (("train", train_colors), ("test", test_colors)):
        for index, colors in enumerate(colors_list):
            profile = profiles[index] if split == "train" else (
                "partial" if rng.random() < 0.15 else "free")
            accepted = None
            for _ in range(32):
                shape = _sample_shape(rng, split, source_shift)
                if profile == "padded":
                    shape = max(3, shape[0]), max(3, shape[1])
                elif profile == "edge":
                    shape = max(2, shape[0]), max(2, shape[1])
                candidate = _make_pair(rng, shape, colors, profile, not source_shift)
                key = _array_key(candidate["input"])
                if key in seen_inputs or _is_source_duplicate(candidate):
                    continue
                try:
                    _analyze_pair(candidate)
                except AssertionError:
                    continue
                accepted = candidate
                seen_inputs.add(key)
                break
            if accepted is None:
                raise ValueError("unable to construct a unique marker rectangle")
            puzzle[split].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[name], b[name]) for a, b in zip(first[split], second[split])
                   for name in ("input", "output")) for split in ("train", "test"))


def _expect_value_error(**kwargs) -> None:
    try:
        generate_puzzle(98765, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for arguments {kwargs}")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert all(_puzzles_equal(puzzle, generate_puzzle(seed)) for seed, puzzle in enumerate(puzzles))
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(314159, train_count, test_count)
        assert _puzzles_equal(puzzle, generate_puzzle(314159, train_count, test_count))
        assert len(puzzle["train"]) == train_count and len(puzzle["test"]) == test_count
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(271828, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": True},
                      {"num_test": 2.5}):
        _expect_value_error(**arguments)

    shapes, layouts, masks, palettes = set(), set(), set(), set()
    role_colors = {"background": set(), "marker": set(), "payload": set()}
    zero_margin = full_canvas = singleton = solid = inner_corners = 0
    train_role_exchange = new_test_color = small_test_after_large = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_info = [_decode_input(pair["input"]) for pair in puzzle["train"]]
        test_info = [_decode_input(pair["input"]) for pair in puzzle["test"]]
        train_markers = {info["marker"] for info in train_info}
        train_payloads = {info["payload"] for info in train_info}
        train_colors = {color for info in train_info for color in
                        (info["background"], info["marker"], info["payload"])}
        train_role_exchange += int(bool(train_markers & train_payloads))
        new_test_color += int(any(any(color not in train_colors for color in
                                      (info["background"], info["marker"], info["payload"]))
                                  for info in test_info))
        small_test_after_large += int(any(t["mask"].size < s["mask"].size
                                          for t in test_info for s in train_info))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode_input(pair["input"])
                assert np.array_equal(pair["output"], info["expected"])
                for role in role_colors:
                    role_colors[role].add(info[role])
                mask = info["mask"]
                shapes.add((pair["input"].shape, pair["output"].shape))
                layouts.add((pair["input"].shape, pair["output"].shape, info["margins"]))
                masks.add((mask.shape, np.packbits(mask).tobytes()))
                palettes.add((info["background"], info["marker"], info["payload"]))
                zero_margin += int(0 in info["margins"])
                full_canvas += int(info["margins"] == (0, 0, 0, 0))
                singleton += int(mask.sum() == 1)
                solid += int(mask.all())
                if mask.sum() == 4:
                    rows = np.unique(np.argwhere(mask)[:, 0])
                    cols = np.unique(np.argwhere(mask)[:, 1])
                    inner_corners += int(len(rows) == len(cols) == 2 and mask[np.ix_(rows, cols)].all())
    assert len(shapes) >= 85 and len(layouts) >= 140 and len(masks) >= 160
    assert len(palettes) >= 55
    assert all(0 in values and len(values) >= 6 for values in role_colors.values())
    assert zero_margin >= 30 and full_canvas >= 8
    assert singleton >= 5 and solid >= 5 and inner_corners >= 2
    assert train_role_exchange >= 5 and new_test_color >= 5 and small_test_after_large >= 5
