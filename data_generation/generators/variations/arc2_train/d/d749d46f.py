import numpy as np


def _rows_key(rows):
    return (
        len(rows),
        len(rows[0]),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_INPUT_KEYS = frozenset(
    _rows_key(rows)
    for rows in (
        (
            "8818181881818",
            "1118111881118",
            "1111111881118",
            "1111111111118",
        ),
        (
            "7773737737377737",
            "3333737737377737",
            "3333733337333337",
            "3333333337333337",
            "3333333333333337",
        ),
        (
            "76776777767",
            "76776666666",
            "76666666666",
            "76666666666",
            "76666666666",
            "76666666666",
        ),
        (
            "29292922292292",
            "99999922299992",
            "99999922299999",
        ),
    )
)


def _grid_key(grid):
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(v) for v in grid.ravel()),
    )


def _matches_source_input(grid):
    return _grid_key(grid) in _SOURCE_INPUT_KEYS


def _normalize_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train_count = _normalize_count(num_train, "num_train", 2, 10)
    test_count = _normalize_count(num_test, "num_test", 1, 4)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        legal_train_counts = tuple(range(max(2, test_count + 1), 11))
        index = (seed // 4) % len(legal_train_counts)
        train_count = legal_train_counts[index]
        return train_count, test_count

    if train_count is None:
        legal_train_counts = tuple(range(max(2, test_count + 1), 11))
        index = int(rng.integers(0, len(legal_train_counts)))
        train_count = legal_train_counts[index]
        return train_count, test_count

    legal_test_counts = tuple(range(1, min(4, train_count - 1) + 1))
    if not legal_test_counts:
        raise ValueError("no legal test count exists for num_train")
    index = int(rng.integers(0, len(legal_test_counts)))
    test_count = legal_test_counts[index]
    return train_count, test_count


def _sample_palettes(rng, count):
    candidates = [(a, b) for a in range(10) for b in range(a + 1, 10)]
    if count > len(candidates):
        raise ValueError("too many pairs for distinct two-color palettes")

    order = rng.permutation(len(candidates))
    palettes = []
    for raw_index in order[:count]:
        a, b = candidates[int(raw_index)]
        if int(rng.integers(0, 2)) == 0:
            background, foreground = a, b
        else:
            background, foreground = b, a
        palettes.append((background, foreground))
    return palettes


def _output_width(items):
    return sum(max(h, w) for h, w in items) + len(items) - 1


def _render_input(items, gaps, background, foreground):
    assert len(gaps) == len(items) - 1
    assert all(gap >= 0 for gap in gaps)

    height = max(h for h, _ in items)
    width = sum(w for _, w in items) + sum(gaps)
    grid = np.full((height, width), background, dtype=np.int8)

    x = 0
    for index, (item_height, item_width) in enumerate(items):
        grid[:item_height, x : x + item_width] = foreground
        x += item_width
        if index < len(gaps):
            x += gaps[index]

    assert x == width
    return grid


def _render_output(items, background, foreground):
    width = _output_width(items)
    grid = np.full((10, width), background, dtype=np.int8)

    x = 0
    for item_height, item_width in items:
        short_side = min(item_height, item_width)
        long_side = max(item_height, item_width)
        grid[:short_side, x : x + long_side] = foreground
        x += long_side + 1

    x = 0
    for item_height, item_width in items:
        short_side = min(item_height, item_width)
        long_side = max(item_height, item_width)
        grid[10 - long_side :, x : x + short_side] = foreground
        x += short_side + 1

    return grid


def _decode_input(grid):
    colors = np.unique(grid)
    assert colors.size == 2, "an input must contain exactly two colors"

    foreground = int(grid[0, 0])
    assert foreground in colors
    background = int(colors[0] if int(colors[1]) == foreground else colors[1])
    assert background != foreground

    height, width = grid.shape
    skyline = []

    for column_index in range(width):
        column = grid[:, column_index]
        if int(column[0]) == background:
            assert np.all(column == background), "spacers must be background"
            skyline.append(0)
            continue

        assert int(column[0]) == foreground
        prefix_height = 0
        while (
            prefix_height < height
            and int(column[prefix_height]) == foreground
        ):
            prefix_height += 1

        assert prefix_height > 0
        if prefix_height < height:
            assert np.all(
                column[prefix_height:] == background
            ), "foreground columns must be solid top prefixes"
        skyline.append(prefix_height)

    assert skyline[0] > 0 and skyline[-1] > 0
    assert max(skyline) == height, "the input must be vertically tight"

    items = []
    gaps = []
    x = 0

    while x < width:
        assert skyline[x] > 0
        item_height = skyline[x]
        start = x
        while x < width and skyline[x] == item_height:
            x += 1
        items.append((item_height, x - start))

        if x == width:
            break

        if skyline[x] == 0:
            gap_start = x
            while x < width and skyline[x] == 0:
                x += 1
            assert x < width, "trailing background columns are not allowed"
            gaps.append(x - gap_start)
        else:
            gaps.append(0)

    assert items
    assert len(gaps) == len(items) - 1
    return background, foreground, items, gaps


def _random_item(rng, input_height):
    """Draw an oriented rectangle from the supported one-color vocabulary."""
    for _ in range(50):
        roll = rng.random()
        if roll < 0.11:
            item = (1, 1)
        elif roll < 0.32:
            side = int(rng.integers(2, min(4, input_height) + 1))
            item = (side, side)
        else:
            short = 1 if roll < 0.62 else int(rng.integers(2, 4))
            long = int(rng.integers(short + 1, 8))
            item = (short, long) if rng.random() < 0.5 else (long, short)
        if item[0] <= input_height:
            return item
    return (1, 2)


def _sample_gaps(rng, items, require_touch=False, require_wide=False):
    gaps = []
    eligible = []
    for i in range(len(items) - 1):
        if items[i][0] != items[i + 1][0]:
            eligible.append(i)
        roll = rng.random()
        if i in eligible and roll < 0.24:
            gaps.append(0)
        elif roll < 0.73:
            gaps.append(1)
        elif roll < 0.92:
            gaps.append(2)
        else:
            gaps.append(3)
    if require_touch and eligible and 0 not in gaps:
        gaps[eligible[int(rng.integers(len(eligible)))]] = 0
    if require_wide and all(gap < 2 for gap in gaps):
        choices = [i for i, gap in enumerate(gaps) if gap != 0]
        if choices:
            gaps[choices[int(rng.integers(len(choices)))]] = 2
    return gaps


def _sample_items(rng, regime, index):
    if regime == "source_train" and index == 0:
        height = int(rng.integers(4, 7))
        side = int(rng.integers(2, min(3, height) + 1))
        thick_long = int(rng.integers(3, 6))
        items = [
            (height, 1),
            (1, int(rng.integers(3, 7))),
            (side, side),
            (2, thick_long) if rng.random() < 0.5 else (thick_long, 2),
        ]
        extra_count = int(rng.integers(0, 3))
        for _ in range(extra_count):
            items.append(_random_item(rng, height))
        rng.shuffle(items)
        return items

    roll = rng.random()
    if roll < 0.05:
        # A source-like inventory appears occasionally in either section.
        items = [(1, 1)] * 3 + [(3, 3), (1, 2), (2, 1)]
        group = [items[:3], [(3, 3)], [(1, 2)], [(2, 1)]]
        rng.shuffle(group)
        return [item for part in group for item in part]

    if roll < (0.25 if regime == "source_train" else 0.31):
        height = int(rng.integers(5, 8))
        items = [(2, 5), (5, 2)]
        count = int(rng.integers(4, 7))
        while len(items) < count:
            items.append(_random_item(rng, height))
        # Guarantee vertical tightness without correlating it with order.
        if height > 5:
            items.append((height, int(rng.integers(1, 3))))
        rng.shuffle(items)
        return items

    if roll < (0.37 if regime == "source_train" else 0.43):
        height = int(rng.integers(3, 6))
        items = [(1, 1)] * 3 + [(3, 3)]
        count = int(rng.integers(5, 8))
        while len(items) < count:
            items.append(_random_item(rng, height))
        if height > 3:
            items.append((height, 1))
        # The singleton run may occur at any position.
        groups = [items[:3]] + [[item] for item in items[3:]]
        rng.shuffle(groups)
        return [item for part in groups for item in part]

    height = int(rng.integers(3, 8))
    count = int(rng.integers(3, 8))
    items = [(height, int(rng.integers(1, min(3, height) + 1)))]
    while len(items) < count:
        if len(items) >= 2 and rng.random() < 0.18:
            alternatives = [(w, h) for h, w in items if h != w and w <= height]
            if alternatives:
                items.append(alternatives[int(rng.integers(len(alternatives)))])
                continue
        items.append(_random_item(rng, height))
    rng.shuffle(items)
    return items


def _legal_items(items):
    if not 3 <= len(items) <= 7:
        return False
    if not all(1 <= h <= 7 and 1 <= w <= 7 for h, w in items):
        return False
    # Preserve a full background row between the two shelves.
    if max(min(h, w) for h, w in items) + max(max(h, w) for h, w in items) > 9:
        return False
    if not 8 <= _output_width(items) <= 30:
        return False
    return True


def _generate_pair(rng, background, foreground, regime, index,
                   height_bounds=(3, 7)):
    for _ in range(1200):
        items = _sample_items(rng, regime, index)
        if not _legal_items(items):
            continue
        anchor = regime == "source_train" and index == 0
        gaps = _sample_gaps(rng, items, require_touch=anchor,
                            require_wide=anchor)
        if anchor and (0 not in gaps or not any(g >= 2 for g in gaps)):
            continue
        if sum(w for _, w in items) + sum(gaps) > 30:
            continue
        input_grid = _render_input(items, gaps, background, foreground)
        if not height_bounds[0] <= input_grid.shape[0] <= height_bounds[1]:
            continue
        if _matches_source_input(input_grid):
            continue
        return {"input": input_grid,
                "output": _render_output(items, background, foreground)}
    raise ValueError("unable to construct a legal rectangle inventory")


def _assert_pair_transform(input_grid, output_grid):
    background, foreground, items, gaps = _decode_input(input_grid)
    assert 3 <= input_grid.shape[0] <= 7
    assert _legal_items(items)
    assert all(0 <= gap <= 3 for gap in gaps)
    assert all(gap > 0 or items[i][0] != items[i + 1][0]
               for i, gap in enumerate(gaps))
    expected = _render_output(items, background, foreground)
    assert output_grid.shape == expected.shape
    assert np.array_equal(output_grid, expected), "incorrect two-shelf transform"
    return background, foreground, items, gaps


def _assert_training_evidence(train_pairs):
    decoded = [_assert_pair_transform(p["input"], p["output"])
               for p in train_pairs]
    items = [item for _, _, pair_items, _ in decoded for item in pair_items]
    gaps = [gap for _, _, _, pair_gaps in decoded for gap in pair_gaps]
    assert any(h < w for h, w in items)
    assert any(h > w for h, w in items)
    assert any(h == w and h > 1 for h, w in items)
    assert any(h != w and min(h, w) > 1 for h, w in items)
    assert 0 in gaps and any(gap >= 2 for gap in gaps)
    assert len({(tuple(pair_items), tuple(pair_gaps))
                for _, _, pair_items, pair_gaps in decoded}) == len(decoded)
    assert len({p["output"].shape[1] for p in train_pairs}) >= 2


def _output_mask_key(pair):
    foreground = int(pair["input"][0, 0])
    grid = pair["output"]
    return (grid.shape, (grid == foreground).tobytes())


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4
    assert train_count > test_count
    seen_inputs, seen_pairs, seen_layouts = set(), set(), set()
    train_masks, test_masks = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            input_key = _grid_key(pair["input"])
            output_key = _grid_key(pair["output"])
            assert input_key not in seen_inputs
            assert (input_key, output_key) not in seen_pairs
            assert input_key not in _SOURCE_INPUT_KEYS
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            _, _, items, gaps = _assert_pair_transform(pair["input"], pair["output"])
            layout = (tuple(items), tuple(gaps))
            assert layout not in seen_layouts
            seen_layouts.add(layout)
            mask = _output_mask_key(pair)
            if section == "train":
                train_masks.add(mask)
            else:
                assert mask not in train_masks and mask not in test_masks
                test_masks.add(mask)
    _assert_training_evidence(puzzle["train"])


def _sample_palette(rng, previous, force_new=False):
    if previous and not force_new and rng.random() < 0.25:
        return previous[int(rng.integers(len(previous)))]
    for _ in range(100):
        colors = rng.choice(10, size=2, replace=False)
        palette = (int(colors[0]), int(colors[1]))
        if not force_new or palette != previous[0]:
            return palette
    raise ValueError("unable to sample palette")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    shift_mode = int(rng.integers(0, 4))
    # Fixed family conventions: ten rows, two semantic colors per pair,
    # landscape copies above portrait copies, and unit output spacers.
    puzzle = {"train": [], "test": []}
    palettes, layouts, masks = [], set(), set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if shift_mode == 0:
                height_bounds = (4, 7) if section == "train" else ((3, 3) if index == 0 else (3, 7))
            elif shift_mode == 1:
                height_bounds = (3, 4) if section == "train" else ((5, 7) if index == 0 else (3, 7))
            else:
                height_bounds = (3, 7)
            palette = _sample_palette(rng, palettes,
                                      force_new=(section == "train" and index == 1))
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(200):
                pair = _generate_pair(pair_rng, *palette,
                                      "source_train" if section == "train" else "target_test",
                                      index, height_bounds)
                _, _, items, gaps = _decode_input(pair["input"])
                layout = (tuple(items), tuple(gaps))
                mask = _output_mask_key(pair)
                if layout not in layouts and (section == "train" or mask not in masks):
                    layouts.add(layout)
                    masks.add(mask)
                    puzzle[section].append(pair)
                    palettes.append(palette)
                    break
            else:
                raise ValueError("unable to construct distinct layouts")
    # Pair zero supplies the orientation, square, thick-block and gap evidence.
    # The remaining pairs vary composition while applying the same conventions.
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((section, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                                 for p in puzzle[section])) for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                background, foreground, items, _ = _decode_input(pair["input"])
                independent = np.full(pair["output"].shape, background, dtype=np.int8)
                upper_x = lower_x = 0
                for height, width in items:
                    short, long = min(height, width), max(height, width)
                    for y in range(short):
                        for x in range(long):
                            independent[y, upper_x + x] = foreground
                    for y in range(long):
                        for x in range(short):
                            independent[10 - long + y, lower_x + x] = foreground
                    upper_x += long + 1
                    lower_x += short + 1
                assert np.array_equal(independent, pair["output"])
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzle_key(generate_puzzle(314159, 8, 3)) == _puzzle_key(generate_puzzle(314159, 8, 3))
    assert _puzzle_key(generate_puzzle(271828, 8, 3)) != _puzzle_key(generate_puzzle(271829, 8, 3))
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        puzzle = generate_puzzle(1000 + i, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + i, train, test)
        assert (train is None or len(puzzle["train"]) == train)
        assert (test is None or len(puzzle["test"]) == test)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4), (2.5, 1), (3, True)):
        try:
            generate_puzzle(3000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    train_layouts, test_masks = set(), set()
    heights, counts, widths, squares, tails = set(), set(), set(), set(), set()
    train_squares, all_train_counts, train_widths = set(), set(), set()
    test_touch = test_wide = test_repeated_orientation = 0
    train_singleton_runs = test_singleton_runs = test_thick = 0
    test_both_rotations = repeated_palettes = 0
    test_touch_and_rotation = test_touch_thick_square = 0
    reverse_shift = source_shift = similar_regime = 0
    for puzzle in puzzles:
        train_heights = {p["input"].shape[0] for p in puzzle["train"]}
        train_counts = {len(_decode_input(p["input"])[2]) for p in puzzle["train"]}
        train_widths_in_puzzle = {p["output"].shape[1] for p in puzzle["train"]}
        assert len(train_widths_in_puzzle) >= 2
        palettes = [tuple(int(v) for v in np.unique(p["input"]))
                    for section in ("train", "test") for p in puzzle[section]]
        repeated_palettes += len(set(palettes)) < len(palettes)
        for pair in puzzle["train"]:
            _, _, items, gaps = _decode_input(pair["input"])
            train_layouts.add((tuple(items), tuple(gaps)))
            all_train_counts.add(len(items))
            train_widths.add(pair["output"].shape[1])
            train_squares.update(h for h, w in items if h == w and h > 1)
            train_singleton_runs += any(items[i:i + 3] == [(1, 1)] * 3
                                        for i in range(len(items) - 2))
        for pair in puzzle["test"]:
            background, foreground, items, gaps = _decode_input(pair["input"])
            mask = tuple(int(v == foreground) for v in pair["output"].ravel())
            test_masks.add((pair["output"].shape, mask))
            heights.add(pair["input"].shape[0])
            counts.add(len(items))
            widths.add(pair["output"].shape[1])
            squares.update(h for h, w in items if h == w and h > 1)
            tails.add(pair["output"].shape[1] -
                      (sum(min(h, w) for h, w in items) + len(items) - 1))
            test_touch += 0 in gaps
            test_wide += any(g >= 2 for g in gaps)
            test_repeated_orientation += any((w, h) in items for h, w in items if h != w)
            test_both_rotations += (2, 5) in items and (5, 2) in items
            test_thick += any(h != w and min(h, w) > 1 for h, w in items)
            test_touch_and_rotation += (0 in gaps and (2, 5) in items
                                        and (5, 2) in items)
            test_touch_thick_square += (
                0 in gaps
                and any(h != w and min(h, w) > 1 for h, w in items)
                and any(h == w and h >= 3 for h, w in items))
            test_singleton_runs += any(items[i:i + 3] == [(1, 1)] * 3
                                       for i in range(len(items) - 2))
            reverse_shift += pair["input"].shape[0] > min(train_heights)
            source_shift += pair["input"].shape[0] < min(train_heights)
            similar_regime += (pair["input"].shape[0] in train_heights and len(items) in train_counts)
    assert len(train_layouts) >= 100
    assert len(test_masks) >= 50
    assert len(heights) >= 4 and len(counts) >= 4 and len(widths) >= 12
    assert len(all_train_counts) >= 4 and len(train_widths) >= 12
    assert {2, 3}.issubset(train_squares)
    assert {2, 3}.issubset(squares) and len(tails) >= 8
    assert test_touch >= 12 and test_wide >= 25
    assert test_repeated_orientation >= 8 and test_both_rotations >= 4
    assert test_touch_and_rotation >= 2 and test_touch_thick_square >= 2
    assert train_singleton_runs >= 8 and test_singleton_runs >= 4
    assert test_thick >= 20 and repeated_palettes >= 5
    assert reverse_shift >= 12 and source_shift >= 4 and similar_regime >= 5
