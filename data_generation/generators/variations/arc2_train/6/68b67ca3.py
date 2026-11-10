"""Generate stride-two lattice decoding puzzles."""

import numpy as np


_AUTO_COUNTS = (
    (3, 1), (2, 1), (4, 2), (5, 1), (6, 3),
    (3, 1), (7, 2), (8, 4), (9, 3), (10, 4),
    (3, 1),
)

_SOURCE_OUTPUTS = (
    ((3, 0, 2), (8, 0, 8), (0, 1, 0)),
    ((2, 2, 1), (2, 1, 0), (3, 0, 0)),
    ((1, 0, 0), (0, 2, 0), (6, 0, 6)),
    ((3, 3, 4), (7, 0, 1), (7, 0, 1)),
)


def _signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _puzzle_signature(puzzle):
    return tuple(
        (split, tuple(
            (_signature(pair["input"]), _signature(pair["output"]))
            for pair in puzzle[split]
        ))
        for split in ("train", "test")
    )


def _embed(latent, background):
    canvas = np.full(
        (2 * latent.shape[0], 2 * latent.shape[1]),
        background,
        dtype=np.int8,
    )
    canvas[::2, ::2] = latent
    return canvas


def _is_source_copy(input_grid, output_grid):
    if output_grid.shape != (3, 3):
        return False
    for source in _SOURCE_OUTPUTS:
        if np.array_equal(output_grid, np.asarray(source)):
            return np.array_equal(input_grid, _embed(output_grid, 0))
    return False


def _optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    count = int(value)
    if not low <= count <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return count


def _resolve_counts(seed, num_train, num_test):
    train = _optional_count(num_train, "num_train", 2, 10)
    test = _optional_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        return max(2, test + 1) + seed % (10 - max(2, test + 1) + 1), test
    if test is None:
        return train, 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sample_shape(rng):
    kind = int(rng.choice(4, p=(0.30, 0.27, 0.23, 0.20)))
    if kind == 0:
        return (3, 3)
    if kind == 1:
        return (int(rng.integers(3, 6)), int(rng.integers(2, 6)))
    if kind == 2:
        return (int(rng.integers(4, 10)), int(rng.integers(3, 11)))
    return (int(rng.integers(8, 16)), int(rng.integers(6, 16)))


def _choose_shapes(rng, train_count, test_count):
    total = train_count + test_count
    mode = int(rng.choice(3, p=(0.32, 0.08, 0.60)))
    if mode == 0:
        return [(3, 3)] * total
    if mode == 1:
        return [_sample_shape(rng)] * total

    direction = int(rng.choice(3, p=(0.18, 0.18, 0.64)))
    if direction == 0:
        # The small test remains below every training area.
        train_shapes = [
            (int(rng.integers(6, 16)), int(rng.integers(5, 16)))
            for _ in range(train_count)
        ]
        return train_shapes + [
            (int(rng.integers(3, 6)), int(rng.integers(2, 6)))
            for _ in range(test_count)
        ]
    if direction == 1:
        # The large test exceeds every training area.
        train_shapes = [
            (int(rng.integers(3, 6)), int(rng.integers(2, 6)))
            for _ in range(train_count)
        ]
        return train_shapes + [
            (int(rng.integers(8, 16)), int(rng.integers(6, 16)))
            for _ in range(test_count)
        ]

    train_shapes = [_sample_shape(rng) for _ in range(train_count)]
    if len(set(train_shapes)) == 1:
        candidate = _sample_shape(rng)
        train_shapes[-1] = candidate if candidate != train_shapes[0] else (4, 3)
    test_shapes = []
    for _ in range(test_count):
        if rng.random() < 0.42:
            test_shapes.append(train_shapes[int(rng.integers(train_count))])
        else:
            test_shapes.append(_sample_shape(rng))
    return train_shapes + test_shapes


def _sample_palette(rng, choices, desired=None):
    if desired is None:
        desired = int(rng.choice((1, 1, 2, 2, 3, 4)))
    size = min(desired, len(choices))
    return tuple(
        int(x) for x in rng.choice(np.asarray(choices), size=size, replace=False)
    )


def _draw_line(grid, rng, palette):
    height, width = grid.shape
    horizontal = bool(rng.integers(0, 2))
    extent = width if horizontal else height
    length = int(rng.integers(1, min(extent, 7) + 1))
    if horizontal:
        row = int(rng.integers(height))
        start = int(rng.integers(width - length + 1))
        grid[row, start:start + length] = int(rng.choice(palette))
    else:
        col = int(rng.integers(width))
        start = int(rng.integers(height - length + 1))
        grid[start:start + length, col] = int(rng.choice(palette))


def _draw_block(grid, rng, palette):
    height, width = grid.shape
    block_height = int(rng.integers(1, min(height, 5) + 1))
    block_width = int(rng.integers(1, min(width, 5) + 1))
    top = int(rng.integers(height - block_height + 1))
    left = int(rng.integers(width - block_width + 1))
    grid[top:top + block_height, left:left + block_width] = int(
        rng.choice(palette)
    )


def _repeat_patch(grid, rng, background, palette):
    height, width = grid.shape
    if height < 4 and width < 4:
        return False
    top = int(rng.integers(height - 1))
    left = int(rng.integers(width - 1))
    positions = [
        (r, c) for r in range(height - 1) for c in range(width - 1)
        if abs(r - top) >= 2 or abs(c - left) >= 2
    ]
    if not positions:
        return False
    other_top, other_left = positions[int(rng.integers(len(positions)))]
    pattern = rng.choice((background,) + tuple(palette), size=(2, 2))
    if np.all(pattern == pattern[0, 0]):
        pattern[0, 0] = (
            int(rng.choice(palette))
            if int(pattern[0, 0]) == background else background
        )
    grid[top:top + 2, left:left + 2] = pattern
    grid[other_top:other_top + 2, other_left:other_left + 2] = pattern
    return True


def _make_latent(rng, shape, background, palette, regime, blank_witness=False):
    height, width = shape
    grid = np.full(shape, background, dtype=np.int8)
    mode = str(rng.choice(
        ("scatter", "marks", "lines", "blocks", "bands", "motifs", "dense"),
        p=(0.19, 0.13, 0.16, 0.16, 0.10, 0.13, 0.13)
        if regime == "source_train"
        else (0.12, 0.10, 0.17, 0.17, 0.10, 0.20, 0.14),
    ))
    cells = height * width
    if mode == "scatter":
        density = float(rng.choice((0.12, 0.25, 0.45, 0.68, 0.88)))
        mask = rng.random(shape) < density
        grid[mask] = rng.choice(palette, size=int(mask.sum()))
    elif mode == "marks":
        count = int(rng.integers(1, min(cells, 8) + 1))
        locations = rng.choice(cells, size=count, replace=False)
        grid.flat[locations] = rng.choice(palette, size=count)
    elif mode == "lines":
        for _ in range(int(rng.integers(1, 4))):
            _draw_line(grid, rng, palette)
    elif mode == "blocks":
        for _ in range(int(rng.integers(1, 4))):
            _draw_block(grid, rng, palette)
    elif mode == "bands":
        if rng.random() < 0.5:
            row = int(rng.integers(height))
            grid[row, :] = rng.choice(palette, size=width)
        else:
            col = int(rng.integers(width))
            grid[:, col] = rng.choice(palette, size=height)
        for _ in range(int(rng.integers(1, 4))):
            _draw_line(grid, rng, palette)
    elif mode == "motifs":
        motif_kind = int(rng.integers(3))
        if motif_kind == 2 and _repeat_patch(grid, rng, background, palette):
            pass
        elif motif_kind == 0:
            row = int(rng.integers(height))
            mask = rng.random(width) < float(rng.uniform(0.35, 0.85))
            grid[row, mask] = rng.choice(palette, size=int(mask.sum()))
            copies = rng.choice(
                [r for r in range(height) if r != row],
                size=int(rng.integers(1, min(height, 4))),
                replace=False,
            )
            for target in copies:
                grid[int(target), :] = grid[row, :]
        else:
            col = int(rng.integers(width))
            mask = rng.random(height) < float(rng.uniform(0.35, 0.85))
            grid[mask, col] = rng.choice(palette, size=int(mask.sum()))
            copies = rng.choice(
                [c for c in range(width) if c != col],
                size=int(rng.integers(1, min(width, 4))),
                replace=False,
            )
            for target in copies:
                grid[:, int(target)] = grid[:, col]
    else:
        grid[:, :] = rng.choice(palette, size=shape)
        if rng.random() < 0.73:
            grid[rng.random(shape) < float(rng.uniform(0.04, 0.36))] = background

    # Independent modifiers let geometry and density combine.
    if mode not in ("dense", "scatter") and rng.random() < 0.33:
        _draw_line(grid, rng, palette)
    if mode not in ("dense", "scatter") and rng.random() < 0.19:
        _draw_block(grid, rng, palette)
    if mode != "motifs" and rng.random() < 0.09:
        _repeat_patch(grid, rng, background, palette)
    if rng.random() < 0.15:
        if rng.random() < 0.5:
            grid[int(rng.integers(height)), :] = background
        else:
            grid[:, int(rng.integers(width))] = background
    if rng.random() < (0.15 if regime == "source_train" else 0.30):
        if rng.random() < 0.5:
            source = int(rng.integers(height))
            target = int(rng.integers(height - 1))
            target += target >= source
            grid[target, :] = grid[source, :]
        else:
            source = int(rng.integers(width))
            target = int(rng.integers(width - 1))
            target += target >= source
            grid[:, target] = grid[:, source]

    if blank_witness:
        # A retained blank band proves that decoding keeps sampled background.
        if rng.random() < 0.5:
            band = int(rng.integers(height))
            grid[band, :] = background
            available = [(r, c) for r in range(height) if r != band
                         for c in range(width)]
        else:
            band = int(rng.integers(width))
            grid[:, band] = background
            available = [(r, c) for r in range(height)
                         for c in range(width) if c != band]
    else:
        available = [(r, c) for r in range(height) for c in range(width)]

    if not np.any(grid != background):
        row, col = available[int(rng.integers(len(available)))]
        grid[row, col] = int(rng.choice(palette))
    return grid


def _pair_colors(rng, background, seen, regime):
    universe = [color for color in range(10) if color != background]
    if regime == "source_train":
        return _sample_palette(rng, universe)
    familiar = sorted(seen)
    unseen = sorted(set(universe) - seen)
    if unseen and rng.random() < 0.53:
        novel = int(rng.choice(unseen))
        remainder = [c for c in universe if c != novel]
        count = int(rng.choice((1, 2, 3, 4)))
        return (novel,) + _sample_palette(rng, remainder, count - 1)
    return _sample_palette(rng, familiar)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)

    # The background is the only sampled puzzle-level convention.
    background = 0 if rng.random() < 0.72 else int(rng.integers(1, 10))
    shapes = _choose_shapes(rng, train_count, test_count)
    child_seeds = rng.integers(
        0, np.iinfo(np.int64).max, size=train_count + test_count,
        dtype=np.int64,
    )
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    seen_colors = set()
    for index, (regime, shape) in enumerate(
        [("source_train", s) for s in shapes[:train_count]]
        + [("target_test", s) for s in shapes[train_count:]]
    ):
        pair_rng = np.random.default_rng(int(child_seeds[index]))
        for _ in range(512):
            colors = _pair_colors(pair_rng, background, seen_colors, regime)
            latent = _make_latent(
                pair_rng, shape, background, colors, regime,
                blank_witness=(index == 0),
            )
            input_grid = _embed(latent, background)
            key = _signature(input_grid)
            if key in used_inputs or _is_source_copy(input_grid, latent):
                continue
            used_inputs.add(key)
            split = "train" if regime == "source_train" else "test"
            puzzle[split].append({"input": input_grid, "output": latent.copy()})
            if regime == "source_train":
                seen_colors.update(int(x) for x in np.unique(latent)
                                   if int(x) != background)
            break
        else:
            raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    inputs = set()
    pairs = set()
    backgrounds = set()
    train_has_sampled_background = False
    train_has_blank_band = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            assert inp.shape == (2 * out.shape[0], 2 * out.shape[1])
            background = int(inp[1, 0])
            assert np.all(inp[1::2, :] == background)
            assert np.all(inp[:, 1::2] == background)
            assert np.array_equal(inp[::2, ::2], out)
            assert np.any(out != background)
            assert not _is_source_copy(inp, out)
            backgrounds.add(background)
            inp_key = _signature(inp)
            pair_key = (inp_key, _signature(out))
            assert inp_key not in inputs
            assert pair_key not in pairs
            inputs.add(inp_key)
            pairs.add(pair_key)
            if split == "train":
                train_has_sampled_background |= bool(np.any(out == background))
                train_has_blank_band |= bool(
                    np.any(np.all(out == background, axis=0))
                    or np.any(np.all(out == background, axis=1))
                )
    assert len(backgrounds) == 1
    assert train_has_sampled_background
    assert train_has_blank_band


def _independent_checks(puzzle):
    background = int(puzzle["train"][0]["input"][1, 0])
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            recovered = np.asarray([
                [int(inp[r, c]) for c in range(0, inp.shape[1], 2)]
                for r in range(0, inp.shape[0], 2)
            ], dtype=np.int8)
            assert np.array_equal(recovered, out)
            assert all(int(inp[r, c]) == background
                       for r in range(inp.shape[0])
                       for c in range(inp.shape[1])
                       if r % 2 or c % 2)
            assert not _is_source_copy(inp, out)


def _has_repeated_patch(grid, background):
    height, width = grid.shape
    positions = {}
    for row in range(height - 1):
        for col in range(width - 1):
            patch = grid[row:row + 2, col:col + 2]
            if np.all(patch == background) or np.all(patch == patch[0, 0]):
                continue
            key = tuple(int(x) for x in patch.flat)
            for old_row, old_col in positions.get(key, ()):
                if abs(row - old_row) >= 2 or abs(col - old_col) >= 2:
                    return True
            positions.setdefault(key, []).append((row, col))
    return False


def _normalized_structure(grid, background):
    rows, cols = np.where(grid != background)
    cropped = grid[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    mapping = {background: 0}
    flattened = []
    for value in cropped.flat:
        color = int(value)
        if color not in mapping:
            mapping[color] = len(mapping)
        flattened.append(mapping[color])
    return (cropped.shape, tuple(flattened))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        _independent_checks(puzzle)
    assert _puzzle_signature(generate_puzzle(17)) == _puzzle_signature(
        generate_puzzle(17)
    )
    assert _puzzle_signature(generate_puzzle(1234, 7, 3)) == _puzzle_signature(
        generate_puzzle(1234, 7, 3)
    )
    assert len({_puzzle_signature(p) for p in puzzles}) == 42

    for index, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + index, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
        _independent_checks(puzzle)
    for seed, train, test in (
        (3000, None, 1), (3001, None, 4),
        (3002, 2, None), (3003, 10, None),
    ):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        _independent_checks(puzzle)
    for train, test in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (3, 4), (4, 4),
    ):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_pairs = [pair for p in puzzles for split in ("train", "test")
                 for pair in p[split]]
    shapes = {pair["output"].shape for pair in all_pairs}
    assert len(shapes) >= 25
    assert (3, 3) in shapes
    assert sum(len({pair["output"].shape for split in ("train", "test")
                    for pair in p[split]}) > 1 for p in puzzles) >= 15
    assert sum(int(p["train"][0]["input"][1, 0]) != 0 for p in puzzles) >= 5
    assert sum(len(set(int(x) for x in pair["output"].flat)) == 1
               for pair in all_pairs) >= 3
    assert len({round(float(np.mean(pair["output"] !=
                                    int(pair["input"][1, 0]))), 1)
                for pair in all_pairs}) >= 7

    train_pairs = [pair for p in puzzles for pair in p["train"]]
    test_pairs = [pair for p in puzzles for pair in p["test"]]
    def background(pair):
        return int(pair["input"][1, 0])

    def colors(pair):
        return set(int(x) for x in np.unique(pair["output"])) - {
            background(pair)
        }

    def row_repeats(pair):
        grid = pair["output"]
        return [i for i in range(grid.shape[0] - 1)
                if np.array_equal(grid[i], grid[i + 1])]

    assert sum({pair["output"].shape for split in ("train", "test")
                for pair in p[split]} == {(3, 3)}
               and len(p["train"]) == 3 and len(p["test"]) == 1
               for p in puzzles) >= 1
    assert sum(len({pair["output"].shape for split in ("train", "test")
                    for pair in p[split]}) > 1 for p in puzzles) >= 15
    assert all(len({tuple(sorted(colors(pair))) for pair in p["train"]}) > 1
               for p in puzzles)
    assert sum(len(colors(pair)) == 1 for pair in all_pairs) >= 20
    assert sum(not np.any(pair["output"] == background(pair))
               for pair in all_pairs) >= 5
    assert sum(bool(row_repeats(pair)) for pair in train_pairs) >= 20
    assert sum(not row_repeats(pair) for pair in test_pairs) >= 10
    assert sum(any(i != pair["output"].shape[0] - 2
                   for i in row_repeats(pair)) for pair in test_pairs) >= 10
    assert sum(np.any(np.all(pair["output"] == background(pair), axis=0))
               for pair in all_pairs) >= 20
    assert sum(np.any(np.all(pair["output"] == background(pair), axis=1))
               for pair in all_pairs) >= 20
    assert sum(_has_repeated_patch(pair["output"], background(pair))
               for pair in all_pairs) >= 15
    assert sum(pair["output"].shape == (3, 3) for pair in all_pairs) >= 30
    assert sum(pair["output"].size >= 60
               and pair["output"].shape[0] != pair["output"].shape[1]
               for pair in all_pairs) >= 10

    test_novel = 0
    test_no_novel = 0
    test_novel_outside_last_row = 0
    test_same_size = 0
    test_new_size = 0
    test_larger = 0
    test_smaller = 0
    combined_shift = 0
    for puzzle in puzzles:
        seen = set().union(*(colors(pair) for pair in puzzle["train"]))
        train_shapes = {pair["output"].shape for pair in puzzle["train"]}
        train_areas = [pair["output"].size for pair in puzzle["train"]]
        for pair in puzzle["test"]:
            grid = pair["output"]
            novel = colors(pair) - seen
            test_novel += bool(novel)
            test_no_novel += not novel
            test_novel_outside_last_row += bool(
                novel - set(int(x) for x in grid[-1])
            )
            test_same_size += grid.shape in train_shapes
            test_new_size += grid.shape not in train_shapes
            test_larger += grid.size > max(train_areas)
            test_smaller += grid.size < min(train_areas)
            combined_shift += bool(novel) and grid.shape not in train_shapes
    assert test_novel >= 10 and test_no_novel >= 10
    assert test_novel_outside_last_row >= 5
    assert test_same_size >= 10 and test_new_size >= 10
    assert test_larger >= 3 and test_smaller >= 3
    assert combined_shift >= 5
    assert len({_normalized_structure(pair["output"], background(pair))
                for pair in all_pairs}) >= 100
