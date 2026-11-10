"""Whole-tile horizontal tiling with alternating left-right reflections."""

import numpy as np

_AUTO_COUNTS = ((2, 1), (3, 1), (4, 1), (4, 2), (5, 2), (6, 2),
                (6, 3), (7, 3), (8, 3), (9, 4), (10, 4))
_SOURCE_INPUTS = (
    ((5, 5, 2, 5), (2, 3, 3, 2), (5, 2, 5, 3), (3, 5, 3, 2)),
    ((9, 5, 1, 5), (1, 5, 9, 1), (9, 1, 5, 5), (5, 5, 5, 1)),
    ((4, 1, 1, 4), (7, 7, 4, 7), (1, 4, 1, 1), (4, 1, 1, 1)),
    ((2, 2, 2, 2), (8, 2, 2, 2), (2, 2, 8, 2), (8, 2, 8, 8)),
    ((5, 5, 4, 4), (5, 5, 5, 2), (2, 5, 5, 5), (5, 5, 2, 4)),
)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _transform(grid):
    mirror = grid[:, ::-1]
    return np.concatenate((grid, mirror, grid, mirror, grid), axis=1)


def _height(rng):
    draw = rng.random()
    if draw < 0.07:
        return 1
    if draw < 0.80:
        return int(rng.integers(2, 9))
    if draw < 0.96:
        return int(rng.integers(9, 16))
    return int(rng.integers(16, 31))


def _sizes(rng, train_count, test_count):
    # A source-like same-size puzzle is an occasional case, not a constraint.
    if rng.random() < 0.17:
        return [(4, 4)] * train_count, [(4, 4)] * test_count
    variety = int(rng.integers(2, min(4, train_count) + 1))
    selected = [int(x) for x in rng.choice(np.arange(2, 7), size=variety, replace=False)]
    train_widths = selected[:]
    while len(train_widths) < train_count:
        train_widths.append(int(rng.choice(selected)))
    train_widths = [train_widths[int(i)] for i in rng.permutation(train_count)]
    train_heights = [_height(rng) for _ in range(train_count)]
    if len(set(train_heights)) == 1:
        train_heights[1] = 1 if train_heights[0] != 1 else 4
    unseen = [w for w in range(1, 7) if w not in selected]
    test_sizes = []
    for _ in range(test_count):
        width = int(rng.choice(unseen if unseen and rng.random() < 0.57 else selected))
        test_sizes.append((_height(rng), width))
    return list(zip(train_heights, train_widths)), test_sizes


def _palette(rng, capacity, earlier):
    # No palette is a puzzle-level code. Pairs can reuse, overlap, or be disjoint.
    if earlier and rng.random() < 0.16:
        options = [p for p in earlier if len(p) <= capacity]
        if options:
            return options[int(rng.integers(len(options)))]
    sizes = np.arange(1, min(10, capacity) + 1)
    weights = np.array([0.08, 0.29, 0.28, 0.15, 0.09, 0.05, 0.03, 0.02, 0.006, 0.004])[:len(sizes)]
    weights /= weights.sum()
    size = int(rng.choice(sizes, p=weights))
    if earlier and rng.random() < 0.48:
        prior = earlier[int(rng.integers(len(earlier)))]
        shared = list(prior) if rng.random() < 0.5 else []
        rng.shuffle(shared)
        shared = shared[:min(size, len(shared))]
        available = [x for x in range(10) if x not in shared]
        extra = [int(x) for x in rng.choice(available, size=size - len(shared), replace=False)]
        return tuple(sorted(shared + extra))
    return tuple(sorted(int(x) for x in rng.choice(10, size=size, replace=False)))


def _pattern(rng, height, width, count):
    rows, cols = np.indices((height, width))
    style = int(rng.choice(7, p=[0.52, 0.14, 0.16, 0.04, 0.04, 0.04, 0.06]))
    if style == 0:  # independent cells
        return rng.integers(count, size=(height, width))
    if style == 1:  # sparse accents
        result = np.full((height, width), int(rng.integers(count)), dtype=int)
        mask = rng.random((height, width)) < rng.uniform(0.12, 0.40)
        result[mask] = rng.integers(count, size=int(mask.sum()))
        return result
    if style == 2:  # rectangular patches on irregular cells
        result = rng.integers(count, size=(height, width))
        for _ in range(int(rng.integers(1, 5))):
            r0, c0 = int(rng.integers(height)), int(rng.integers(width))
            r1, c1 = int(rng.integers(r0 + 1, height + 1)), int(rng.integers(c0 + 1, width + 1))
            result[r0:r1, c0:c1] = int(rng.integers(count))
        return result
    if style == 3:  # simple stripes
        by_row = rng.random() < 0.5
        codes = rng.integers(count, size=height if by_row else width)
        return np.broadcast_to(codes[:, None] if by_row else codes[None, :], (height, width)).copy()
    if style == 4:  # repeated row motif
        period = int(rng.integers(1, min(height, 3) + 1))
        motif = rng.integers(count, size=(period, width))
        return np.tile(motif, (height // period + 1, 1))[:height].copy()
    if style == 5:  # diagonals
        return ((rows + cols) // int(rng.integers(1, 4)) + int(rng.integers(count))) % count
    result = (rows * int(rng.integers(1, 4)) + cols) % count
    mask = rng.random((height, width)) < rng.uniform(0.15, 0.5)
    result[mask] = rng.integers(count, size=int(mask.sum()))
    return result


def _make_tile(rng, height, width, palette, require_asym=False):
    count = len(palette)
    symmetric = not require_asym and (count == 1 or width == 1 or rng.random() < 0.11)
    if count > height * ((width + 1) // 2):
        symmetric = False
    indices = _pattern(rng, height, width, count)
    if symmetric:
        left = indices[:, :(width + 1) // 2].copy()
        positions = rng.permutation(left.size)
        for color, position in enumerate(positions[:count]):
            left.flat[int(position)] = color
        indices = np.concatenate((left, left[:, :width // 2][:, ::-1]), axis=1)
    else:
        positions = rng.permutation(indices.size)
        for color, position in enumerate(positions[:count]):
            indices.flat[int(position)] = color
        if height > 1 and width > 1 and rng.random() < 0.30:
            row = indices[0, :(width + 1) // 2].copy()
            indices[0] = np.concatenate((row, row[:width // 2][::-1]))
            missing = [color for color in range(count) if color not in indices]
            lower = rng.permutation(np.arange(width, indices.size))
            for color, position in zip(missing, lower):
                indices.flat[int(position)] = color
        if count > 1 and (require_asym or rng.random() < 0.77):
            if np.array_equal(indices, indices[:, ::-1]):
                row = int(rng.integers(height))
                indices[row, -1] = (int(indices[row, 0]) + 1) % count
        # Repair any color erased while making a row palindromic or asymmetric.
        missing = [color for color in range(count) if color not in indices]
        safe_positions = list(range(indices.size))
        rng.shuffle(safe_positions)
        for color in missing:
            for position in safe_positions:
                old = int(indices.flat[position])
                if np.count_nonzero(indices == old) > 1:
                    indices.flat[position] = color
                    safe_positions.remove(position)
                    break
        if require_asym and np.array_equal(indices, indices[:, ::-1]):
            row = int(rng.integers(height))
            indices[row, -1] = (int(indices[row, 0]) + 1) % count
    return np.asarray(np.asarray(palette, dtype=np.int8)[indices], dtype=np.int8)


def _fallback(seed, ordinal, height, width, used, require_asym):
    for bump in range(1000):
        grid = np.full((height, width), seed % 10, dtype=np.int8)
        code = ordinal + bump
        for position in range(min(4, grid.size)):
            grid.flat[position] = code % 10
            code //= 10
        if require_asym and width > 1:
            grid[-1, -1] = (int(grid[-1, 0]) + 1) % 10
        if _key(grid) not in used and not any(np.array_equal(grid, np.asarray(src)) for src in _SOURCE_INPUTS):
            return grid
    raise RuntimeError("could not construct a unique tile")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    train_sizes, test_sizes = _sizes(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    palettes, used = [], set()
    ordinal = 0
    for split, sizes in (("train", train_sizes), ("test", test_sizes)):
        for index, (height, width) in enumerate(sizes):
            must_reveal = split == "train" and index == 0
            unique = False
            for _ in range(5):
                palette = _palette(rng, height * width, palettes)
                if must_reveal and len(palette) == 1:
                    palette = tuple(sorted((palette[0], (palette[0] + 1) % 10)))
                for _ in range(30):
                    tile = _make_tile(rng, height, width, palette, must_reveal)
                    if _key(tile) not in used and not any(np.array_equal(tile, np.asarray(src)) for src in _SOURCE_INPUTS):
                        unique = True
                        break
                if unique:
                    break
            if not unique:
                tile = _fallback(seed, ordinal, height, width, used, must_reveal)
            used.add(_key(tile))
            palettes.append(tuple(int(x) for x in np.unique(tile)))
            puzzle[split].append({"input": tile, "output": _transform(tile)})
            ordinal += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, train_widths = set(), set(), set()
    asymmetric_train = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            tile, output = pair["input"], pair["output"]
            for grid in (tile, output):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            assert tile.shape[1] <= 6
            assert output.shape == (tile.shape[0], 5 * tile.shape[1])
            panels = np.split(output, 5, axis=1)
            for index, panel in enumerate(panels):
                assert np.array_equal(panel, tile if index % 2 == 0 else tile[:, ::-1])
            input_key = _key(tile)
            pair_key = (input_key, _key(output))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not any(np.array_equal(tile, np.asarray(src)) for src in _SOURCE_INPUTS)
            if split == "train":
                train_widths.add(tile.shape[1])
                asymmetric_train += not np.array_equal(tile, tile[:, ::-1])
    assert asymmetric_train >= 1
    if any(pair["input"].shape[1] not in train_widths for pair in puzzle["test"]):
        assert len(train_widths) >= 2


def _same_puzzle(a, b):
    return all(np.array_equal(x[role], y[role]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for role in ("input", "output"))


def _predict_from_training(train, test_input):
    # This check sees only demonstration grids and the new input. It derives
    # the number and sequence of panels from every returned training output.
    patterns = []
    witnessed_reflection = False
    for pair in train:
        tile, output = pair["input"], pair["output"]
        width = tile.shape[1]
        assert output.shape[0] == tile.shape[0] and output.shape[1] % width == 0
        panels = np.split(output, output.shape[1] // width, axis=1)
        original, mirror = tile, tile[:, ::-1]
        pattern = []
        for index, panel in enumerate(panels):
            expected = original if index % 2 == 0 else mirror
            assert np.array_equal(panel, expected)
            pattern.append(index % 2)
        witnessed_reflection |= not np.array_equal(original, mirror)
        patterns.append(tuple(pattern))
    assert witnessed_reflection and len(set(patterns)) == 1
    return np.concatenate([test_input[:, ::-1] if mirrored else test_input
                           for mirrored in patterns[0]], axis=1)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    input_keys, shapes, palettes, color_counts = [], set(), set(), set()
    mixed = same_4x4 = new_test_width = symmetric = first_row_symmetric = 0
    single_row = tall = single_column = test_only_color = reused_palette = 0
    new_test_aspect = disjoint_test = overlapping_test = 0
    lower_asym = mixed_symmetry_rows = first_row_only_asym = 0
    monochrome = four_plus = repeated_train = repeated_test = 0
    distinct_train = distinct_test = style_match = style_depart = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        widths = {pair["input"].shape[1] for pair in puzzle["train"]}
        train_colors = set().union(*(set(np.unique(pair["input"])) for pair in puzzle["train"]))
        train_palettes = {tuple(int(c) for c in np.unique(pair["input"])) for pair in puzzle["train"]}
        all_pairs = puzzle["train"] + puzzle["test"]
        train_pair_ids = {id(pair) for pair in puzzle["train"]}
        mixed += len({pair["input"].shape for pair in all_pairs}) > 1
        same_4x4 += all(pair["input"].shape == (4, 4) for pair in all_pairs)
        new_test_width += any(pair["input"].shape[1] not in widths for pair in puzzle["test"])
        train_aspects = {p["input"].shape[1] / p["input"].shape[0] for p in puzzle["train"]}
        new_test_aspect += any(p["input"].shape[1] / p["input"].shape[0] not in train_aspects for p in puzzle["test"])
        test_only_color += any(set(np.unique(pair["input"])) - train_colors for pair in puzzle["test"])
        reused_palette += any(tuple(int(c) for c in np.unique(pair["input"])) in train_palettes for pair in puzzle["test"])
        disjoint_test += any(all(not (set(np.unique(p["input"])) & set(np.unique(t["input"])))
                                 for t in puzzle["train"]) for p in puzzle["test"])
        overlapping_test += any(any(0 < len(set(np.unique(p["input"])) & set(np.unique(t["input"])))
                                    < len(set(np.unique(p["input"])) | set(np.unique(t["input"])))
                                    for t in puzzle["train"]) for p in puzzle["test"])
        train_repeat = {len({tuple(row) for row in p["input"]}) < p["input"].shape[0] for p in puzzle["train"]}
        test_repeat = [len({tuple(row) for row in p["input"]}) < p["input"].shape[0] for p in puzzle["test"]]
        style_match += any(value in train_repeat for value in test_repeat)
        style_depart += any(value not in train_repeat for value in test_repeat)
        for pair in all_pairs:
            tile, output = pair["input"], pair["output"]
            assert np.array_equal(output, np.concatenate((tile, tile[:, ::-1], tile, tile[:, ::-1], tile), axis=1))
            input_keys.append(_key(tile))
            shapes.add(tile.shape)
            palettes.add(tuple(int(c) for c in np.unique(tile)))
            color_counts.add(len(np.unique(tile)))
            symmetric += np.array_equal(tile, tile[:, ::-1])
            first_row_symmetric += np.array_equal(tile[0], tile[0, ::-1])
            asym_rows = [not np.array_equal(row, row[::-1]) for row in tile]
            lower_asym += any(asym_rows[1:])
            mixed_symmetry_rows += any(asym_rows) and not all(asym_rows)
            first_row_only_asym += asym_rows[0] and not any(asym_rows[1:])
            single_row += tile.shape[0] == 1
            tall += tile.shape[0] >= 10
            single_column += tile.shape[1] == 1
            monochrome += len(np.unique(tile)) == 1
            four_plus += len(np.unique(tile)) >= 4
            repeated = len({tuple(row) for row in tile}) < tile.shape[0]
            if id(pair) in train_pair_ids:
                repeated_train += repeated
                distinct_train += not repeated
            else:
                repeated_test += repeated
                distinct_test += not repeated
        demonstrations = [{"input": p["input"].copy(), "output": p["output"].copy()}
                          for p in puzzle["train"]]
        for pair in puzzle["test"]:
            assert np.array_equal(_predict_from_training(demonstrations, pair["input"].copy()), pair["output"])
    assert len(shapes) >= 35 and len(palettes) >= 80
    assert {1, 2, 3, 4}.issubset(color_counts)
    assert len(set(input_keys)) >= int(0.94 * len(input_keys))
    assert mixed >= 20 and same_4x4 >= 2 and new_test_width >= 12
    assert symmetric >= 10 and first_row_symmetric >= 30
    assert single_row >= 5 and tall >= 30 and single_column >= 3
    assert test_only_color >= 5 and reused_palette >= 3
    assert new_test_aspect >= 15 and disjoint_test >= 1 and overlapping_test >= 15
    assert lower_asym >= 80 and mixed_symmetry_rows >= 30 and first_row_only_asym >= 5
    assert monochrome >= 5 and four_plus >= 30
    assert min(repeated_train, repeated_test, distinct_train, distinct_test) >= 10
    assert style_match >= 15 and style_depart >= 1
    a = generate_puzzle(123456, 8, 3)
    assert _same_puzzle(a, generate_puzzle(123456, 8, 3))
    assert not _same_puzzle(a, generate_puzzle(123457, 8, 3))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(9000 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(p)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(99, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
