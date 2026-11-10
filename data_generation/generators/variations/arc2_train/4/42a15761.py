import numpy as np


_SOURCE_INPUT_ROWS = (
    ("222022202220222", "222020202020222", "222022202220222",
     "222020202020202", "222022202220222", "202020202020222",
     "222022202220222", "202022202020222", "222022202220222"),
    ("22202220222", "20202220202", "22202220222", "20202220202",
     "22202220222", "22202020202", "22202220222"),
    ("222022202220222", "222020202220202", "222022202220222",
     "222020202020202", "222022202220222", "202022202020202",
     "222022202220222", "222020202220202", "222022202220222"),
    ("2220222022202220222", "2220222020202020202",
     "2220222022202220222", "2220222022202020222",
     "2220222022202220222", "2020222020202020202",
     "2220222022202220222", "2220202020202020202",
     "2220222022202220222", "2020222022202020202",
     "2220222022202220222"),
)
_SOURCE_INPUTS = tuple(
    np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)
    for rows in _SOURCE_INPUT_ROWS
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _checked_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    nt = _checked_count(num_train, "num_train", 2, 10)
    ns = _checked_count(num_test, "num_test", 1, 4)
    if nt is None and ns is None:
        ns = 1 + seed % 4
        low = max(2, ns + 1)
        nt = low + (seed // 4) % (11 - low)
    elif nt is None:
        low = max(2, ns + 1)
        nt = low + seed % (11 - low)
    elif ns is None:
        ns = 1 + seed % min(4, nt - 1)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _shape(rows, columns, interior_height, interior_width, gap, pad_y, pad_x):
    return (2 * pad_y + rows * (interior_height + 1) + 1,
            2 * pad_x + columns * (interior_width + 2) + (columns - 1) * gap)


def _encode(logical, structural, empty, geometry, direction):
    rows, columns = logical.shape
    ih, iw, gap, py, px = geometry
    height, width = _shape(rows, columns, ih, iw, gap, py, px)
    grid = np.full((height, width), empty, dtype=np.int8)
    stride = iw + 2 + gap
    body_height = rows * (ih + 1) + 1
    for column in range(columns):
        x = px + column * stride
        grid[py:py + body_height, x:x + iw + 2] = structural
        for row in range(rows):
            if not logical[row, column]:
                y = py + 1 + row * (ih + 1)
                grid[y:y + ih, x + 1:x + 1 + iw] = empty
    return np.rot90(grid, direction).copy()


def _decode_candidates(grid):
    """Recover every supported orientation and lattice directly from pixels."""
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return []
    colors = np.unique(grid)
    if len(colors) != 2:
        return []
    candidates = []
    for direction in range(4):
        canonical = np.rot90(grid, -direction)
        height, width = canonical.shape
        for py in range(3):
            for ih in range(1, 4):
                numerator = height - 2 * py - 1
                if numerator < 2 * (ih + 1) or numerator % (ih + 1):
                    continue
                rows = numerator // (ih + 1)
                if rows > 9:
                    continue
                for px in range(3):
                    for iw in range(1, 4):
                        for gap in range(1, 4):
                            stride = iw + 2 + gap
                            numerator_w = width - 2 * px + gap
                            if numerator_w % stride:
                                continue
                            columns = numerator_w // stride
                            if not 2 <= columns <= min(8, rows + 1):
                                continue
                            for structural, empty in ((int(colors[0]), int(colors[1])),
                                                       (int(colors[1]), int(colors[0]))):
                                logical = np.empty((rows, columns), dtype=np.int8)
                                valid = True
                                for column in range(columns):
                                    x = px + column * stride + 1
                                    for row in range(rows):
                                        y = py + 1 + row * (ih + 1)
                                        tile = canonical[y:y + ih, x:x + iw]
                                        value = int(tile[0, 0])
                                        if not np.all(tile == value):
                                            valid = False
                                            break
                                        logical[row, column] = int(value == structural)
                                    if not valid:
                                        break
                                if not valid:
                                    continue
                                geometry = (ih, iw, gap, py, px)
                                if np.array_equal(grid, _encode(logical, structural,
                                                                empty, geometry, direction)):
                                    candidates.append((logical, structural, empty,
                                                       geometry, direction))
    return candidates


def _sort(logical):
    weights = logical.sum(axis=0, dtype=np.int64)
    return logical[:, np.argsort(-weights, kind="stable")].copy()


def _key(grid):
    return (tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes())


def _sample_geometry(rng, rows, columns):
    classic = (1, 1, 1, 0, 0)
    if _shape(rows, columns, *classic)[0] <= 30 and _shape(rows, columns, *classic)[1] <= 30:
        if float(rng.random()) < 0.20:
            return classic
    possible = []
    for ih in range(1, 4):
        for iw in range(1, 4):
            for gap in range(1, 4):
                for py in range(3):
                    for px in range(3):
                        h, w = _shape(rows, columns, ih, iw, gap, py, px)
                        if h <= 30 and w <= 30:
                            possible.append((ih, iw, gap, py, px))
    if not possible:
        return None
    return possible[int(rng.integers(len(possible)))]


def _sample_counts(rng, rows, columns):
    modes = ["general"]
    if columns <= rows:
        modes += ["source", "nonempty", "full"]
    mode = modes[int(rng.integers(len(modes)))]
    if mode == "source":
        counts = np.arange(columns, dtype=np.int64)
    elif mode == "nonempty":
        counts = rng.choice(np.arange(1, rows + 1), size=columns, replace=False)
    elif mode == "full":
        others = rng.choice(np.arange(rows), size=columns - 1, replace=False)
        counts = np.concatenate((others, [rows]))
    else:
        counts = rng.choice(np.arange(rows + 1), size=columns, replace=False)
    return np.asarray(counts, dtype=np.int64)


def _sample_logical(rng, rows, columns, witness=False, allow_identity=True,
                    classic_counts=False):
    if witness:
        _require(rows >= 3 and columns >= 2, "witness needs at least three rows")
        if classic_counts:
            low, high = 1, 2
            available = [x for x in range(columns) if x not in (low, high)]
        else:
            low = int(rng.integers(1, rows - 1))
            high = int(rng.integers(low + 1, rows))
            available = [x for x in range(rows + 1) if x not in (low, high)]
        extras = rng.choice(available, size=columns - 2, replace=False)
        counts = np.concatenate(([low, high], extras))
    else:
        counts = (np.arange(columns, dtype=np.int64) if classic_counts
                  else _sample_counts(rng, rows, columns))
        order_choice = float(rng.random())
        if allow_identity and order_choice < 0.12:
            counts.sort()
            counts = counts[::-1]
        elif order_choice < 0.27:
            counts.sort()
        else:
            rng.shuffle(counts)
            if not allow_identity and np.all(counts[:-1] >= counts[1:]):
                counts = counts[::-1]
    logical = np.zeros((rows, columns), dtype=np.int8)
    for column, count in enumerate(counts):
        count = int(count)
        if witness and column == 0:
            logical[:count, column] = 1
        elif witness and column == 1:
            logical[rows - count:, column] = 1
        elif count:
            occupied = rng.choice(rows, size=count, replace=False)
            logical[occupied, column] = 1
    if witness:
        # Column 0 has the larger top-to-bottom bit string, but fewer cells.
        _require(int(counts[0]) < int(counts[1]), "invalid witness")
    return logical


def _sample_dimensions(rng, regime, index):
    if regime == "source_train":
        rows = int(rng.integers(3, 8))
        columns = int(rng.integers(2, min(7, rows + 2)))
    else:
        rows = int(rng.integers(2, 10))
        columns = int(rng.integers(2, min(9, rows + 2)))
    return rows, columns


def _pair(rng, regime, index, direction, palette, seen, classic=False):
    for _ in range(100):
        if classic:
            size = ((3 if index == 0 else 4 if index == 1 else int(rng.integers(3, 5)))
                    if regime == "source_train" else int(rng.integers(5, 7)))
            rows, columns = size, size
            geometry = (1, 1, 1, 0, 0)
        else:
            rows, columns = _sample_dimensions(rng, regime, index)
            geometry = _sample_geometry(rng, rows, columns)
        if geometry is None:
            continue
        logical = _sample_logical(rng, rows, columns, witness=(regime == "source_train" and index == 0),
                                  allow_identity=(regime == "target_test" or index > 1),
                                  classic_counts=classic)
        source = _encode(logical, *palette, geometry, direction)
        source_key = _key(source)
        if source_key in seen or any(np.array_equal(source, old) for old in _SOURCE_INPUTS):
            continue
        result = _encode(_sort(logical), *palette, geometry, direction)
        seen.add(source_key)
        return {"input": source, "output": result}
    raise RuntimeError("unable to construct a unique pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _checked_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    classic = float(rng.random()) < 0.10
    direction = 0 if classic else int(rng.integers(4))
    variable_palette = False if classic else bool(rng.integers(3))
    colors = rng.choice(10, size=2, replace=False)
    base_palette = (int(colors[0]), int(colors[1]))
    seen = set()
    puzzle = {"train": [], "test": []}
    previous_palette = None
    for split, total in (("train", train_count), ("test", test_count)):
        for index in range(total):
            pair_seed = int(rng.integers(0, 2**63, dtype=np.int64))
            pair_rng = np.random.default_rng(pair_seed)
            if variable_palette:
                palette = base_palette
                for _ in range(20):
                    choice = pair_rng.choice(10, size=2, replace=False)
                    palette = (int(choice[0]), int(choice[1]))
                    if palette != previous_palette:
                        break
            else:
                palette = base_palette
            previous_palette = palette
            puzzle[split].append(_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                       index, direction, palette, seen, classic))
    validate_generated_puzzle(puzzle)
    return puzzle


def _positional_witness(logical):
    rows, columns = logical.shape
    weights = logical.sum(axis=0)
    positional = np.array(
        [sum(int(logical[row, col]) << (rows - 1 - row) for row in range(rows))
         for col in range(columns)]
    )
    return any(weights[a] > weights[b] and positional[a] < positional[b]
               for a in range(columns) for b in range(columns))


def _inspect_pair(pair):
    source, result = pair["input"], pair["output"]
    source_candidates = _decode_candidates(source)
    result_candidates = _decode_candidates(result)
    solutions = []
    for logical, structural, empty, geometry, direction in source_candidates:
        if not np.array_equal(source, _encode(logical, structural, empty, geometry, direction)):
            continue
        weights = logical.sum(axis=0)
        if len(set(int(x) for x in weights)) != logical.shape[1]:
            continue
        expected = _sort(logical)
        for other, os, oe, og, od in result_candidates:
            if (structural, empty, geometry, direction) != (os, oe, og, od):
                continue
            if np.array_equal(other, expected):
                solutions.append((logical, structural, empty, geometry, direction))
    _require(solutions, "pair violates lattice or decreasing-count whole-stack sort")
    _require(len({item[4] for item in solutions}) == 1,
             "sorting destination is ambiguous")
    return solutions[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"} and len(puzzle) == 2,
             "puzzle must contain exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    directions = set()
    train_info = []
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"} and len(pair) == 2,
                     "pair must contain exactly input and output")
            source, result = pair["input"], pair["output"]
            for grid in (source, result):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be two-dimensional NumPy arrays")
                _require(np.issubdtype(grid.dtype, np.integer), "grids must be integer arrays")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape),
                         "grid dimensions must lie in 1..30")
                _require(np.min(grid) >= 0 and np.max(grid) <= 9, "illegal ARC color")
            _require(source.shape == result.shape, "output dimensions changed")
            source_key, result_key = _key(source), _key(result)
            _require(source_key not in seen_inputs and (source_key, result_key) not in seen_pairs,
                     "duplicate puzzle pair or input")
            seen_inputs.add(source_key)
            seen_pairs.add((source_key, result_key))
            _require(not any(np.array_equal(source, old) for old in _SOURCE_INPUTS),
                     "source task input copied")
            logical, structural, empty, geometry, direction = _inspect_pair(pair)
            directions.add(direction)
            rows, columns = logical.shape
            _require(2 <= columns <= rows + 1, "not enough possible distinct counts")
            if split == "train":
                _require(rows >= 3, "training board is too small to establish the rule")
                train_info.append(logical)
    _require(len(directions) == 1, "sorting direction changes across pairs")
    _require(any(_positional_witness(logical) for logical in train_info),
             "training does not distinguish count sorting from positional sorting")
    _require(any(not np.array_equal(logical, _sort(logical)) for logical in train_info),
             "training does not show a moving stack")


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    train_counts, test_counts, shapes, palettes, directions = set(), set(), set(), set(), set()
    rows_seen, columns_seen, gaps, interiors, pads = set(), set(), set(), set(), set()
    layouts, signatures = set(), set()
    reverse_inputs = identity_inputs = no_empty = full = gap_counts = 0
    source_count_sets = two_stacks = classic_pairs = combined_variants = 0
    variable_palette_puzzles = rectangular_puzzles = 0
    within_dimensions = within_geometry = within_count_sets = classic_puzzles = 0
    larger_tests = smaller_tests = shared_sizes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_signature(puzzle) == _signature(generate_puzzle(seed)), "nondeterministic puzzle")
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_palettes, local_shapes, local_geometry, local_counts = set(), set(), set(), set()
        train_dimensions, test_dimensions = [], []
        all_classic = True
        for split in ("train", "test"):
            for pair in puzzle[split]:
                logical, structural, empty, geometry, direction = _inspect_pair(pair)
                rows, columns = logical.shape
                weights = logical.sum(axis=0)
                _require(np.array_equal(pair["output"],
                                        _encode(_sort(logical), structural, empty, geometry, direction)),
                         "incorrect whole-stack permutation")
                _require(not any(np.array_equal(pair["input"], old) for old in _SOURCE_INPUTS),
                         "source task pair copied")
                rows_seen.add(rows)
                columns_seen.add(columns)
                shapes.add(tuple(pair["input"].shape))
                palettes.add((structural, empty))
                directions.add(direction)
                gaps.add(geometry[2])
                interiors.add(geometry[:2])
                pads.add(geometry[3:])
                local_palettes.add((structural, empty))
                local_shapes.add((rows, columns))
                local_geometry.add(geometry)
                local_counts.add(tuple(sorted(int(x) for x in weights)))
                (train_dimensions if split == "train" else test_dimensions).append((rows, columns))
                layouts.add((logical.shape, logical.tobytes()))
                reverse_inputs += int(np.all(weights[:-1] <= weights[1:]))
                identity_inputs += int(np.array_equal(logical, _sort(logical)))
                no_empty += int(np.all(weights > 0))
                full += int(np.any(weights == rows))
                gap_counts += int(not np.array_equal(np.sort(weights),
                                                     np.arange(columns)))
                source_count_sets += int(np.array_equal(np.sort(weights),
                                                        np.arange(columns)))
                two_stacks += int(columns == 2)
                classic = (direction == 0 and rows == columns
                           and geometry == (1, 1, 1, 0, 0)
                           and np.array_equal(np.sort(weights), np.arange(columns)))
                classic_pairs += int(classic)
                all_classic &= classic
                combined_variants += int(rows != columns and np.all(weights > 0)
                                         and (geometry[0] > 1 or geometry[1] > 1))
        variable_palette_puzzles += int(len(local_palettes) > 1)
        rectangular_puzzles += int(any(r != c for r, c in local_shapes))
        within_dimensions += int(len(local_shapes) > 1)
        within_geometry += int(len(local_geometry) > 1)
        within_count_sets += int(len(local_counts) > 1)
        classic_puzzles += int(all_classic)
        larger_tests += int(any(r * c > max(a * b for a, b in train_dimensions)
                                for r, c in test_dimensions))
        smaller_tests += int(any(r * c < min(a * b for a, b in train_dimensions)
                                 for r, c in test_dimensions))
        shared_sizes += int(any(item in train_dimensions for item in test_dimensions))
    _require(len(signatures) == 42 and len(layouts) >= 150,
             "cross-seed logical diversity is too low")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair count diversity is too low")
    _require(len(shapes) >= 25 and len(rows_seen) >= 6 and len(columns_seen) >= 5,
             "size diversity is too low")
    _require(len(palettes) >= 25 and variable_palette_puzzles >= 8,
             "palette diversity is too low")
    _require(len(directions) == 4 and rectangular_puzzles >= 25,
             "orientation or rectangle diversity is too low")
    _require(len(gaps) == 3 and len(interiors) >= 6 and len(pads) >= 5,
             "lattice geometry diversity is too low")
    _require(min(reverse_inputs, identity_inputs, no_empty, full, gap_counts) >= 5,
             "count or permutation diversity is too low")
    _require(source_count_sets >= 10 and two_stacks >= 5 and classic_pairs >= 5
             and combined_variants >= 5 and classic_puzzles >= 1,
             "source-like, small, or combined examples are missing")
    _require(within_dimensions >= 30 and within_geometry >= 20
             and within_count_sets >= 30,
             "pair-level properties do not vary within puzzles")
    _require(min(larger_tests, smaller_tests, shared_sizes) >= 3,
             "train/test size schedules lack variety")
    a = generate_puzzle(12345, 7, 3)
    b = generate_puzzle(12345, 7, 3)
    _require(_signature(a) == _signature(b), "explicit counts are nondeterministic")
    _require(_signature(a) != _signature(generate_puzzle(12346, 7, 3)),
             "different seeds produced identical puzzles")
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(100000 + nt * 10 + ns, nt, ns)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns),
                 "explicit counts were not honored")
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(200000 + (nt or 0) * 10 + (ns or 0), nt, ns)
        _require((nt is None or len(puzzle["train"]) == nt)
                 and (ns is None or len(puzzle["test"]) == ns),
                 "partial counts were not honored")
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(300000, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal count request did not raise ValueError")
