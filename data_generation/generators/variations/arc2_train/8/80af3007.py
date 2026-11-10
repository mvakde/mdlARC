"""Solved ARC puzzles built from a 3-by-3 template's self-substitution."""

import numpy as np


# These describe the four original pairs, including their padding and colors.
# Reusing a source motif on a new canvas is allowed; copying a pair is not.
_SOURCE_CASES = frozenset({
    ((6, 1, 6), (16, 18), (1, 7), 0, 5),
    ((5, 2, 5), (16, 18), (1, 1), 0, 5),
    ((7, 3, 5), (16, 18), (4, 3), 0, 5),
    ((7, 2, 5), (17, 19), (3, 7), 0, 5),
})
_SOURCE_MOTIFS = frozenset(case[0] for case in _SOURCE_CASES)


def _template(rows):
    return np.array(
        [[(rows[r] >> (2 - c)) & 1 for c in range(3)] for r in range(3)],
        dtype=np.int8,
    )


def _rows(template):
    return tuple(
        sum(int(template[r, c]) << (2 - c) for c in range(3))
        for r in range(3)
    )


def _fill(rows):
    return sum(int(mask).bit_count() for mask in rows)


def _candidate_rows():
    # Nonempty outer rows and occupied outer columns make the foreground's
    # tight bounding box exactly 9 by 9, even with a blank middle row.
    return tuple(
        (a, b, c)
        for a in range(1, 8)
        for b in range(8)
        for c in range(1, 8)
        if (a | b | c) & 5 == 5
        and 2 <= _fill((a, b, c)) <= 8
    )


_CANDIDATES = _candidate_rows()


def _pick(rng, predicate, excluded=()):
    choices = [
        rows for rows in _CANDIDATES
        if rows not in excluded and predicate(rows)
    ]
    if not choices:
        choices = [rows for rows in _CANDIDATES if rows not in excluded]
    return choices[int(rng.integers(len(choices)))]


def _sample_by_fill(rng, excluded=()):
    fill = int(rng.integers(2, 9))
    return _pick(rng, lambda rows: _fill(rows) == fill, excluded)


def _transform(rows, operation):
    matrix = _template(rows)
    if operation == 0:
        matrix = np.rot90(matrix)
    elif operation == 1:
        matrix = np.fliplr(matrix)
    else:
        matrix = np.flipud(matrix)
    return _rows(matrix)


def _select_motifs(rng, seed, num_train, num_test):
    first_kind = seed % 8
    if first_kind == 0:
        first = (4, 0, 1) if int(rng.integers(2)) == 0 else (1, 0, 4)
    elif first_kind == 1:
        first = _pick(rng, lambda x: x[1] == 0 and _fill(x) >= 3)
    elif first_kind == 2:
        first = (5, 2, 5)
    elif first_kind == 3:
        first = _pick(rng, lambda x: x in _SOURCE_MOTIFS)
    else:
        first = _sample_by_fill(rng)

    if _fill(first) <= 3:
        second = _pick(rng, lambda x: _fill(x) >= 6, (first,))
    elif _fill(first) >= 7:
        second = _pick(rng, lambda x: _fill(x) <= 4, (first,))
    else:
        second = _pick(
            rng, lambda x: abs(_fill(x) - _fill(first)) >= 2, (first,)
        )
    train = [first, second]

    if num_train >= 3:
        if seed % 3 != 0:
            train.append(first)
        else:
            variants = [
                _transform(first, op) for op in range(3)
                if _transform(first, op) not in train
            ]
            train.append(
                variants[int(rng.integers(len(variants)))]
                if variants else _sample_by_fill(rng, train)
            )
    while len(train) < num_train:
        # A broad occupancy range is sampled independently of canvas and color.
        train.append(_sample_by_fill(rng, train))

    test = []
    train_vocabulary = {mask for motif in train for mask in motif}
    for index in range(num_test):
        kind = (seed + 3 * index) % 7
        excluded = set(train) | set(test)
        if kind == 0:
            motif = _pick(rng, lambda x: _fill(x) <= 3, excluded)
        elif kind == 1:
            motif = _pick(rng, lambda x: x[1] == 0, excluded)
        elif kind == 2:
            motif = _pick(
                rng,
                lambda x: sorted(m.bit_count() for m in x) == [1, 2, 3],
                excluded,
            )
        elif kind == 3:
            familiar = [
                x for x in _CANDIDATES
                if x not in excluded
                and all(mask in train_vocabulary for mask in x)
            ]
            motif = (
                familiar[int(rng.integers(len(familiar)))]
                if familiar else _sample_by_fill(rng, excluded)
            )
        elif kind == 4:
            motif = _pick(rng, lambda x: _fill(x) >= 7, excluded)
        elif kind == 5:
            motif = _pick(
                rng, lambda x: _fill(x) == _fill(first), excluded
            )
        else:
            motif = _sample_by_fill(rng, excluded)
        test.append(motif)
    return train, test


def _output(template, background, foreground):
    return np.where(
        np.kron(template, template) == 1, foreground, background
    ).astype(np.int8)


def _input(template, shape, offset, background, foreground):
    patch = np.where(
        np.kron(template, np.ones((3, 3), dtype=np.int8)) == 1,
        foreground,
        background,
    ).astype(np.int8)
    grid = np.full(shape, background, dtype=np.int8)
    row, col = offset
    grid[row:row + 9, col:col + 9] = patch
    return grid


def _checked_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _counts(rng, seed, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = num_test + 1
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _different_colors(rng):
    background = int(rng.integers(10))
    foreground = int(rng.integers(9))
    if foreground >= background:
        foreground += 1
    return background, foreground


def _palettes(rng, seed, pair_count):
    base = _different_colors(rng)
    reverse = (base[1], base[0])
    mode = seed % 3
    if mode == 0:
        return [base] * pair_count
    palettes = [base, reverse]
    for _ in range(pair_count - 2):
        if mode == 1:
            palettes.append((base, reverse)[int(rng.integers(2))])
        else:
            palettes.append(_different_colors(rng))
    return palettes


def _shapes(rng, seed, num_train, num_test):
    mode = seed % 4
    if mode == 1:  # Test canvases are smaller than every training canvas.
        train_bounds, test_bounds = (19, 31), (13, 17)
    elif mode == 2:  # Test canvases substantially exceed all training ones.
        train_bounds, test_bounds = (13, 21), (25, 31)
    else:
        train_bounds, test_bounds = (13, 31), (13, 31)
    train = [
        tuple(int(rng.integers(*train_bounds)) for _ in range(2))
        for _ in range(num_train)
    ]
    if train[1] == train[0]:
        h, w = train[1]
        train[1] = (h, w + 1 if w < train_bounds[1] - 1 else w - 1)
    test = [
        tuple(int(rng.integers(*test_bounds)) for _ in range(2))
        for _ in range(num_test)
    ]
    if mode == 0:
        test[0] = train[int(rng.integers(num_train))]
    return train + test


def _decode(input_grid):
    colors, counts = np.unique(input_grid, return_counts=True)
    assert len(colors) == 2
    background = int(colors[int(np.argmax(counts))])
    foreground = int(colors[int(np.argmin(counts))])
    points = np.argwhere(input_grid == foreground)
    row, col = (int(x) for x in points.min(axis=0))
    last_row, last_col = (int(x) for x in points.max(axis=0))
    height, width = input_grid.shape
    assert (last_row - row + 1, last_col - col + 1) == (9, 9)
    assert 1 <= row and row + 9 < height
    assert 1 <= col and col + 9 < width
    crop = input_grid[row:row + 9, col:col + 9]
    template = np.zeros((3, 3), dtype=np.int8)
    for r in range(3):
        for c in range(3):
            tile = crop[3*r:3*r+3, 3*c:3*c+3]
            assert np.all(tile == tile[0, 0])
            template[r, c] = int(tile[0, 0] == foreground)
    assert 2 <= int(template.sum()) <= 8
    expected = _input(template, input_grid.shape, (row, col),
                      background, foreground)
    assert np.array_equal(input_grid, expected)
    return _rows(template), (row, col), background, foreground


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, seed, num_train, num_test)
    # The substitution rule is the puzzle-level convention. Palette modes
    # vary presentation only; each input reveals its own foreground role.
    train_motifs, test_motifs = _select_motifs(
        rng, seed, num_train, num_test
    )
    motifs = train_motifs + test_motifs
    shapes = _shapes(rng, seed, num_train, num_test)
    palettes = _palettes(rng, seed, len(motifs))
    pair_seeds = rng.integers(0, 2**63 - 1, size=len(motifs),
                              dtype=np.int64)
    pairs = []
    used_inputs = set()
    for motif, shape, palette, pair_seed in zip(
        motifs, shapes, palettes, pair_seeds
    ):
        pair_rng = np.random.default_rng(int(pair_seed))
        offsets = [
            (r, c)
            for r in range(1, shape[0] - 9)
            for c in range(1, shape[1] - 9)
        ]
        template = _template(motif)
        background, foreground = palette
        for ix in pair_rng.permutation(len(offsets)):
            offset = offsets[int(ix)]
            if (motif, shape, offset, background,
                    foreground) in _SOURCE_CASES:
                continue
            inp = _input(template, shape, offset,
                         background, foreground)
            key = _grid_key(inp)
            if key not in used_inputs:
                used_inputs.add(key)
                pairs.append({
                    "input": inp,
                    "output": _output(template, background, foreground),
                })
                break
        else:
            raise AssertionError("no distinct legal placement")
    puzzle = {
        "train": pairs[:num_train],
        "test": pairs[num_train:],
    }
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[part], list) for part in ("train", "test"))
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    train_motifs, test_motifs = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape[0] >= 13 and inp.shape[1] >= 13
            assert out.shape == (9, 9)
            inp_key, out_key = _grid_key(inp), _grid_key(out)
            assert inp_key not in seen_inputs
            assert (inp_key, out_key) not in seen_pairs
            seen_inputs.add(inp_key)
            seen_pairs.add((inp_key, out_key))
            rows, offset, background, foreground = _decode(inp)
            assert (rows, inp.shape, offset, background,
                    foreground) not in _SOURCE_CASES
            template = _template(rows)
            # Reconstruct cell by cell, separately from the generation path.
            expected = np.full((9, 9), background, dtype=np.int8)
            for r in range(9):
                for c in range(9):
                    if template[r//3, c//3] and template[r%3, c%3]:
                        expected[r, c] = foreground
            assert np.array_equal(out, expected)
            (train_motifs if split == "train" else test_motifs).append(rows)
    assert len(set(train_motifs)) >= 2
    assert all(motif in _CANDIDATES for motif in train_motifs + test_motifs)
    assert len(set(test_motifs)) == test_count
    assert set(train_motifs).isdisjoint(test_motifs)
    assert len({pair["input"].shape for pair in puzzle["train"]}) >= 2
    # Any local palette is revealed by its input. The training pairs teach
    # the shared operation; no unseen color code or hidden row vocabulary
    # is needed to predict a test output.


def _puzzle_key(puzzle):
    return tuple(
        tuple(
            (_grid_key(pair["input"]), _grid_key(pair["output"]))
            for pair in puzzle[split]
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    automatic = generate_puzzle(12345)
    explicit = generate_puzzle(54321, 7, 3)
    assert (len(explicit["train"]), len(explicit["test"])) == (7, 3)
    assert _puzzle_key(automatic) == _puzzle_key(generate_puzzle(12345))
    assert _puzzle_key(explicit) == _puzzle_key(generate_puzzle(54321, 7, 3))
    for seed, train_count, test_count in (
        (7001, 2, 1), (7002, 10, 1), (7003, 10, 4),
        (7101, None, 1), (7102, None, 4),
        (7103, 2, None), (7104, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for kwargs in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": True}, {"num_test": 1.0},
    ):
        try:
            generate_puzzle(7200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    train_fills, test_fills = set(), set()
    motifs, placements, shapes, palettes = set(), set(), set(), set()
    empty_train = empty_test = opposite = repeated = 0
    same_shape = smaller = larger = varied_palette = 0
    source_motif = x_motif = 0
    census_train = census_test = familiar_new = 0
    comparable = new_fill = rotations = swaps = combinations = 0
    repeat_recolored_resized = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_rows = []
        train_shapes = []
        local_palettes = set()
        train_details = []
        test_details = []
        local_sparse = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                rows, offset, bg, fg = _decode(inp)
                # An input-only prediction reproduces the expected answer.
                assert np.array_equal(out, _output(_template(rows), bg, fg))
                fill = _fill(rows)
                (train_fills if split == "train" else test_fills).add(fill)
                motifs.add(rows)
                placements.add(offset)
                shapes.add(inp.shape)
                palettes.add((bg, fg))
                local_palettes.add((bg, fg))
                local_sparse |= fill <= 3
                empty_train += int(split == "train" and rows[1] == 0)
                empty_test += int(split == "test" and rows[1] == 0)
                census = sorted(mask.bit_count() for mask in rows) == [1, 2, 3]
                census_train += int(split == "train" and census)
                census_test += int(split == "test" and census)
                opposite += int(rows in ((4, 0, 1), (1, 0, 4)))
                source_motif += int(rows in _SOURCE_MOTIFS)
                x_motif += int(rows == (5, 2, 5))
                if split == "train":
                    train_rows.append(rows)
                    train_shapes.append(inp.shape)
                    train_details.append((rows, inp.shape, offset, (bg, fg)))
                elif inp.shape in train_shapes:
                    same_shape += 1
                    test_details.append((rows, fill))
                elif all(
                    inp.shape[0] < shape[0] and inp.shape[1] < shape[1]
                    for shape in train_shapes
                ):
                    smaller += 1
                    test_details.append((rows, fill))
                elif all(
                    inp.shape[0] >= shape[0] + 4
                    and inp.shape[1] >= shape[1] + 4
                    for shape in train_shapes
                ):
                    larger += 1
                    test_details.append((rows, fill))
                else:
                    test_details.append((rows, fill))
        repeated += int(len(set(train_rows)) < len(train_rows))
        varied_palette += int(len(local_palettes) > 1)
        swaps += int(any((fg, bg) in local_palettes
                         for bg, fg in local_palettes))
        combinations += int(
            len(local_palettes) > 1
            and len({item[1] for item in train_details}) > 1
            and local_sparse
        )
        repeat_recolored_resized += int(any(
            a[0] == b[0] and a[1] != b[1] and a[3] != b[3]
            for i, a in enumerate(train_details)
            for b in train_details[i + 1:]
        ))
        rotations += int(any(
            a != b and _transform(a, op) == b
            for i, a in enumerate(train_rows)
            for b in train_rows[i + 1:]
            for op in range(3)
        ))
        vocabulary = {mask for rows in train_rows for mask in rows}
        familiar_new += int(any(
            all(mask in vocabulary for mask in rows)
            and rows not in train_rows
            for rows, _ in test_details
        ))
        comparable += int(any(
            fill in {_fill(rows) for rows in train_rows}
            for _, fill in test_details
        ))
        new_fill += int(any(
            fill not in {_fill(rows) for rows in train_rows}
            for _, fill in test_details
        ))
        assert len(set(train_shapes)) > 1
    assert train_fills == test_fills == set(range(2, 9))
    assert len(motifs) >= 90 and len(placements) >= 45
    assert len(shapes) >= 90 and len(palettes) >= 35
    assert min(empty_train, empty_test, opposite) >= 5
    assert min(repeated, varied_palette) >= 8
    assert min(same_shape, smaller, larger) >= 5
    assert source_motif >= 5 and x_motif >= 5
    assert min(census_train, census_test) >= 5
    assert min(familiar_new, comparable, new_fill) >= 10
    assert min(rotations, swaps, combinations) >= 8
    assert repeat_recolored_resized >= 5
