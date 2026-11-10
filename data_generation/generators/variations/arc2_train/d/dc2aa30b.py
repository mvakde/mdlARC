"""Whole-tile rank sorting puzzles in regularly divided grids."""

import numpy as np


# The four original solved pairs are kept only for exact-copy exclusion.
_SOURCE_ROWS = (
    (("11202120211", "21202120111", "12201220112", "00000000000",
      "12101110222", "21101110212", "12201210222", "00000000000",
      "12201210111", "22102110111", "22201210111"),
     ("21101110111", "11101110111", "11201210111", "00000000000",
      "11201210121", "21202110211", "12201220121", "00000000000",
      "22201220212", "21202210212", "22202220122")),
    (("22102220121", "12202220112", "22201220212", "00000000000",
      "22102120222", "12201210222", "21202210222", "00000000000",
      "21101110111", "12101110211", "12101210112"),
     ("21101110111", "12102110111", "12101120121", "00000000000",
      "22102120121", "12201210112", "21202210212", "00000000000",
      "22202220221", "22202220122", "22201220222")),
    (("22202220221", "22102220122", "22202220212", "00000000000",
      "21101110111", "12101110211", "21201110121", "00000000000",
      "21101210211", "12101210111", "21102220111"),
     ("11102110111", "21101110111", "12101110111", "00000000000",
      "12102110211", "12101210121", "22202120211", "00000000000",
      "22202220221", "22202210122", "22202220212")),
    (("22202120221", "12201220121", "21202120212", "00000000000",
      "11201110111", "12101110112", "12101110111", "00000000000",
      "21102110222", "11101210222", "11201220222"),
     ("21101110111", "11101120111", "11201110111", "00000000000",
      "22102110112", "12101210121", "21201220121", "00000000000",
      "22202220212", "22201220122", "22202120212")),
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _source_key(rows):
    return ((len(rows), len(rows[0])), tuple(int(c) for row in rows for c in row))


_SOURCE_KEYS = frozenset((_source_key(a), _source_key(b)) for a, b in _SOURCE_ROWS)
_BASE = (3, 3, 3, 3)  # tile rows, tile columns, tile height, tile width


def _starts(number, tile_size):
    return tuple(i * (tile_size + 1) for i in range(number))


def _assemble(tiles, geometry, separator):
    rows, cols, height, width = geometry
    grid = np.full((rows * height + rows - 1,
                    cols * width + cols - 1), separator, dtype=np.int8)
    for index, tile in enumerate(tiles):
        top = (index // cols) * (height + 1)
        left = (index % cols) * (width + 1)
        grid[top:top + height, left:left + width] = tile
    return grid


def _runs(markers, length):
    """Lengths and starts of non-divider runs."""
    starts = []
    lengths = []
    beginning = 0
    for position in list(markers) + [length]:
        if position > beginning:
            starts.append(beginning)
            lengths.append(position - beginning)
        beginning = position + 1
    return starts, lengths


def _parse(grid):
    """Infer the local divider color and regular tile geometry from cells."""
    palette = set(int(x) for x in np.unique(grid))
    _require(len(palette) == 3, "A pair must have three distinct colors")
    candidates = []
    for color in palette:
        divider_rows = tuple(int(i) for i in np.flatnonzero(np.all(grid == color, axis=1)))
        divider_cols = tuple(int(i) for i in np.flatnonzero(np.all(grid == color, axis=0)))
        if divider_rows and divider_cols:
            candidates.append((color, divider_rows, divider_cols))
    _require(len(candidates) == 1, "Full divider rows and columns must reveal one color")
    separator, divider_rows, divider_cols = candidates[0]
    row_starts, heights = _runs(divider_rows, grid.shape[0])
    col_starts, widths = _runs(divider_cols, grid.shape[1])
    _require(len(heights) >= 2 and len(widths) >= 2,
             "There must be at least two tile rows and columns")
    _require(len(set(heights)) == 1 and len(set(widths)) == 1,
             "Every tile in a pair must have equal dimensions")
    height, width = heights[0], widths[0]
    _require(height >= 2 and width >= 2, "Tiles are too small")
    geometry = (len(heights), len(widths), height, width)
    _require(divider_rows == tuple(i * (height + 1) + height
                                   for i in range(geometry[0] - 1)) and
             divider_cols == tuple(i * (width + 1) + width
                                   for i in range(geometry[1] - 1)),
             "Dividers must be single internal lines with no outer border")
    _require(height * width + 1 >= geometry[0] * geometry[1],
             "Tile area cannot encode distinct counts")
    tiles = [grid[top:top + height, left:left + width].copy()
             for top in row_starts for left in col_starts]
    _require(all(separator not in tile for tile in tiles),
             "Divider color may not enter a tile")
    return separator, geometry, tiles, palette - {separator}


def _destination_slots(geometry, direction):
    rows, cols = geometry[:2]
    row_order = range(rows) if direction < 2 else range(rows - 1, -1, -1)
    col_order = range(cols - 1, -1, -1) if direction % 2 == 0 else range(cols)
    return tuple(r * cols + c for r in row_order for c in col_order)


def _sort_grid(input_grid, token, direction):
    separator, geometry, tiles, roles = _parse(input_grid)
    _require(token in roles, "Token color must occur inside the tiles")
    counts = [int(np.count_nonzero(tile == token)) for tile in tiles]
    _require(len(counts) == len(set(counts)), "Tile counts must be distinct")
    output = [None] * len(tiles)
    for source_index, destination in zip(np.argsort(counts),
                                         _destination_slots(geometry, direction)):
        output[destination] = tiles[int(source_index)]
    return _assemble(output, geometry, separator)


def _coerce_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_pair_counts(seed, rng, num_train, num_test):
    train = _coerce_count("num_train", num_train, 2, 10)
    test = _coerce_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _geometry_options():
    # A compositional range, rather than a short bank of pictures.
    return tuple((rows, cols, height, width)
                 for rows in (2, 3, 4) for cols in (2, 3, 4)
                 for height in (2, 3, 4, 5) for width in (2, 3, 4, 5)
                 if rows * cols >= 6 and height * width >= rows * cols
                 and rows * height + rows - 1 <= 30
                 and cols * width + cols - 1 <= 30
                 and (rows, cols, height, width) != _BASE)


_GEOMETRIES = _geometry_options()


def sample_puzzle_parameters(rng):
    """Conventions shared by every pair, plus training-covered geometries."""
    token = int(rng.integers(0, 10))
    direction = 0 if rng.random() < 0.5 else int(rng.integers(1, 4))
    return {"token": token, "direction": direction}


def _make_tile(rng, count, geometry, filler, token):
    area = geometry[2] * geometry[3]
    flat = np.full(area, filler, dtype=np.int8)
    if count:
        flat[rng.choice(area, size=count, replace=False)] = token
    return flat.reshape(geometry[2], geometry[3])


def _scramble(rng, sorted_layout, geometry, mode):
    result = list(sorted_layout)
    if mode == 0:  # an exact two-tile swap
        a, b = rng.choice(len(result), size=2, replace=False)
        result[int(a)], result[int(b)] = result[int(b)], result[int(a)]
    elif mode == 1:  # leave every other row ordered
        cols = geometry[1]
        row = int(rng.integers(geometry[0]))
        first = row * cols
        last = first + cols
        original = result[first:last]
        result[first:last] = original[1:] + original[:1]
    elif mode == 2:
        result = [int(x) for x in rng.permutation(result)]
        if result == sorted_layout:
            result[0], result[1] = result[1], result[0]
    else:
        result.reverse()
    return result


def generate_pair(rng, regime, puzzle_parameters, geometry, separator,
                  filler, counts, scramble_mode):
    if regime not in ("source_train", "target_test"):
        raise ValueError("Unknown pair regime")
    token = puzzle_parameters["token"]
    direction = puzzle_parameters["direction"]
    tiles = {count: _make_tile(rng, count, geometry, filler, token) for count in counts}
    sorted_layout = [None] * len(counts)
    for count, destination in zip(sorted(counts), _destination_slots(geometry, direction)):
        sorted_layout[destination] = count
    input_layout = _scramble(rng, sorted_layout, geometry, scramble_mode)
    input_grid = _assemble([tiles[count] for count in input_layout], geometry, separator)
    output_grid = _sort_grid(input_grid, token, direction)
    return input_grid, output_grid


def _count_set(rng, geometry, reserved_gap=None, train_baseline=False,
               reserved_set=None):
    rows, cols, height, width = geometry
    number = rows * cols
    area = height * width
    if geometry == _BASE:
        possible = [gap for gap in range(10) if gap != reserved_gap]
        if train_baseline:
            possible = [gap for gap in possible if 1 <= gap <= 8]
        omission = int(rng.choice(possible))
        return tuple(k for k in range(10) if k != omission)
    selected = tuple(sorted(int(x) for x in rng.choice(area + 1, size=number,
                                                       replace=False)))
    if selected == reserved_set:
        missing = next(value for value in range(area + 1) if value not in selected)
        selected = tuple(sorted((missing,) + selected[1:]))
    return selected


def _choose_colors(rng, token, first_colors=None):
    choices = [color for color in range(10) if color != token]
    filler_choices = [c for c in choices
                      if first_colors is None or c != first_colors[1]]
    filler = int(rng.choice(filler_choices))
    separator_choices = [c for c in choices if c != filler and
                         (first_colors is None or c != first_colors[0])]
    separator = int(rng.choice(separator_choices))
    return separator, filler


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_pair_counts(seed, rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    alternative_count = min(train_count - 1, 3)
    alternative_indices = rng.choice(len(_GEOMETRIES), size=alternative_count,
                                     replace=False)
    geometries = [_BASE] + [_GEOMETRIES[int(i)] for i in alternative_indices]
    train_geometries = list(geometries)
    while len(train_geometries) < train_count:
        train_geometries.append(geometries[int(rng.integers(len(geometries)))])
    rng.shuffle(train_geometries)
    test_geometries = [geometries[int(rng.integers(len(geometries)))]
                       for _ in range(test_count)]
    # The withheld baseline omission is interior. Baseline training uses
    # another interior omission; both ends still occur in later training.
    test_baseline_gap = int(rng.integers(1, 9))
    reserved_sets = {
        geometry: (tuple(k for k in range(10) if k != test_baseline_gap)
                   if geometry == _BASE else _count_set(rng, geometry))
        for geometry in geometries
    }
    train_modes = [int(x) for x in rng.permutation(4)]
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_count_sets = {}
    first_colors = None
    baseline_train_seen = False

    for section, geometry_list in (("train", train_geometries),
                                   ("test", test_geometries)):
        for index, geometry in enumerate(geometry_list):
            if section == "train":
                mode = train_modes[index % 4]
            else:
                mode = int(rng.integers(4))
            color_constraint = first_colors if section == "train" and index == 1 else None
            separator, filler = _choose_colors(rng, parameters["token"], color_constraint)
            if section == "train" and index == 0:
                first_colors = (separator, filler)
            for attempt in range(100):
                if section == "test":
                    counts = reserved_sets[geometry]
                else:
                    counts = _count_set(rng, geometry,
                                        test_baseline_gap if geometry == _BASE else None,
                                        geometry == _BASE and section == "train"
                                        and not baseline_train_seen,
                                        reserved_sets[geometry])
                child_seed = int(rng.integers(0, 2**63, dtype=np.uint64))
                pair_rng = np.random.default_rng(child_seed)
                input_grid, output_grid = generate_pair(
                    pair_rng, "source_train" if section == "train" else "target_test",
                    parameters, geometry, separator, filler, counts, mode)
                pair = {"input": input_grid, "output": output_grid}
                if _grid_key(input_grid) in seen_inputs or _pair_key(pair) in _SOURCE_KEYS:
                    continue
                puzzle[section].append(pair)
                seen_inputs.add(_grid_key(input_grid))
                if section == "train":
                    train_count_sets.setdefault(geometry, set()).add(counts)
                    if geometry == _BASE:
                        baseline_train_seen = True
                break
            else:
                raise ValueError("Unable to construct a fresh pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "A puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "Train and test must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "Illegal train/test counts")
    records = []
    input_keys, pair_keys = set(), set()
    for section, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "A pair must have input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                         np.issubdtype(grid.dtype, np.integer),
                         "Grids must be two-dimensional integer NumPy arrays")
                _require(all(1 <= size <= 30 for size in grid.shape) and
                         np.all((grid >= 0) & (grid <= 9)),
                         "Grid size or colors are out of ARC bounds")
            input_grid, output_grid = pair["input"], pair["output"]
            _require(input_grid.shape == output_grid.shape and
                     not np.array_equal(input_grid, output_grid),
                     "Tiles must move without changing the canvas")
            key, pkey = _grid_key(input_grid), _pair_key(pair)
            _require(key not in input_keys and pkey not in pair_keys,
                     "Duplicate input or pair")
            _require(pkey not in _SOURCE_KEYS, "Exact source pair copied")
            input_keys.add(key)
            pair_keys.add(pkey)
            sep, geometry, tiles, roles = _parse(input_grid)
            out_sep, out_geometry, out_tiles, out_roles = _parse(output_grid)
            _require(sep == out_sep and geometry == out_geometry and roles == out_roles,
                     "Pair geometry and color roles must be preserved")
            _require(sorted(tuple(tile.flat) for tile in tiles) ==
                     sorted(tuple(tile.flat) for tile in out_tiles),
                     "Output must contain exactly the unchanged input tiles")
            records.append((section, pair, geometry, roles))

    # Each local divider is visible. Distinct training fillers make the
    # intersection of tile colors the unique puzzle-wide token role.
    train_roles = [roles for section, _, _, roles in records if section == "train"]
    shared = set.intersection(*train_roles)
    _require(len(shared) == 1, "Training must identify one stable token color")
    token = next(iter(shared))
    _require(all(token in roles for _, _, _, roles in records),
             "Token identity must persist into every test")
    possible_directions = set(range(4))
    for section, pair, _, _ in records:
        if section == "train":
            possible_directions = {direction for direction in possible_directions
                                   if np.array_equal(pair["output"],
                                                     _sort_grid(pair["input"], token,
                                                                direction))}
    _require(len(possible_directions) == 1,
             "Training must uniquely reveal the rank traversal")
    direction = next(iter(possible_directions))
    train_geometries = set()
    train_count_sets = {}
    baseline_interior = False
    for section, pair, geometry, _ in records:
        expected = _sort_grid(pair["input"], token, direction)
        _require(np.array_equal(pair["output"], expected),
                 "Output is not the whole-tile rank sort")
        _, _, tiles, _ = _parse(pair["input"])
        counts = tuple(sorted(int(np.count_nonzero(tile == token)) for tile in tiles))
        _require(len(set(counts)) == len(tiles), "Tile counts must be unique")
        if section == "train":
            train_geometries.add(geometry)
            train_count_sets.setdefault(geometry, set()).add(counts)
            if geometry == _BASE:
                baseline_interior |= (0 in counts and 9 in counts)
        else:
            _require(geometry in train_geometries,
                     "Every test geometry must already occur in training")
            _require(counts not in train_count_sets.get(geometry, set()),
                     "Test must shift the training count set for its geometry")
            if geometry == _BASE:
                _require(0 in counts and 9 in counts,
                         "Baseline tests must have an interior count gap")
    _require(_BASE in train_geometries and len(train_geometries) >= 2 and
             baseline_interior, "Training must show source and alternate geometry")


def _equal_puzzles(a, b):
    return all(len(a[section]) == len(b[section]) and
               all(np.array_equal(x[name], y[name]) and x[name].dtype == y[name].dtype
                   for x, y in zip(a[section], b[section]) for name in ("input", "output"))
               for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, geometries, directions, tokens = set(), set(), set(), set()
    layouts, motifs, omissions, moved_counts = set(), set(), set(), set()
    changed_color_and_geometry = 0
    train_baseline_gaps, test_baseline_gaps = set(), set()
    reverse_count = partial_row_count = two_moved_count = broad_count = 0
    changed_fillers = changed_separators = changed_geometry = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_roles = [_parse(pair["input"])[3] for pair in puzzle["train"]]
        token = next(iter(set.intersection(*train_roles)))
        tokens.add(token)
        directions.add(next(direction for direction in range(4)
                            if all(np.array_equal(p["output"],
                                                  _sort_grid(p["input"], token, direction))
                                   for p in puzzle["train"])))
        puzzle_fillers, puzzle_separators, puzzle_geometries = set(), set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                separator, geometry, tiles, roles = _parse(pair["input"])
                puzzle_separators.add(separator)
                puzzle_fillers.add(next(iter(roles - {token})))
                puzzle_geometries.add(geometry)
                shapes.add(pair["input"].shape)
                geometries.add(geometry)
                counts = tuple(int(np.count_nonzero(t == token)) for t in tiles)
                layouts.add((geometry, counts))
                motifs.add((geometry, tuple(tuple(int(x == token) for x in t.flat)
                                             for t in tiles)))
                omissions.add((geometry, tuple(sorted(counts))))
                if geometry == _BASE:
                    missing = next(count for count in range(10) if count not in counts)
                    (train_baseline_gaps if section == "train" else
                     test_baseline_gaps).add(missing)
                _, _, output_tiles, _ = _parse(pair["output"])
                moved = sum(not np.array_equal(a, b)
                            for a, b in zip(tiles, output_tiles))
                moved_counts.add(moved)
                if moved == 2:
                    two_moved_count += 1
                elif all(np.array_equal(a, b)
                         for a, b in zip(tiles[::-1], output_tiles)):
                    reverse_count += 1
                elif any(all(np.array_equal(tiles[j], output_tiles[j])
                             for j in range(len(tiles)) if j // geometry[1] != r)
                         for r in range(geometry[0])):
                    partial_row_count += 1
                else:
                    broad_count += 1
                if geometry != _BASE and (separator !=
                        _parse(puzzle["train"][0]["input"])[0] or
                        next(iter(roles - {token})) !=
                        next(iter(train_roles[0] - {token}))):
                    changed_color_and_geometry += 1
        changed_fillers += len(puzzle_fillers) >= 2
        changed_separators += len(puzzle_separators) >= 2
        changed_geometry += len(puzzle_geometries) >= 2
    _require(len({len(p["train"]) for p in puzzles}) >= 4 and
             len({len(p["test"]) for p in puzzles}) >= 3,
             "Automatic pair counts lack variety")
    _require(len(shapes) >= 15 and len(geometries) >= 20 and
             len(directions) == 4 and len(tokens) >= 7,
             "Geometry or puzzle conventions lack variety")
    _require(len(layouts) >= 220 and len(motifs) >= 220 and
             len(omissions) >= 100 and 2 in moved_counts and
             max(moved_counts) >= 10 and changed_color_and_geometry >= 50,
             "Count sets, patterns, or scrambling lack variety")
    _require(train_baseline_gaps == set(range(10)) and
             test_baseline_gaps == set(range(1, 9)),
             "Baseline omissions must include both extremes and all interior gaps")
    _require(min(two_moved_count, reverse_count, partial_row_count, broad_count) >= 20,
             "Scramble regimes are not all represented")
    _require((changed_fillers, changed_separators, changed_geometry) == (42, 42, 42),
             "Pair-level color and geometry variation must appear within each puzzle")
    for seed, tr, te in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                         (103, 5, 2), (104, 7, 3)):
        puzzle = generate_puzzle(seed, tr, te)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (tr, te),
                 "Explicit counts were not honored")
    for seed, tr, te in ((200, None, 1), (201, None, 4),
                         (202, 2, None), (203, 10, None)):
        puzzle = generate_puzzle(seed, tr, te)
        _require((tr is None or len(puzzle["train"]) == tr) and
                 (te is None or len(puzzle["test"]) == te),
                 "Partial counts were not honored")
    _require(_equal_puzzles(generate_puzzle(314159), generate_puzzle(314159)) and
             _equal_puzzles(generate_puzzle(271828, 8, 3),
                            generate_puzzle(271828, 8, 3)),
             "Generation must be deterministic")
    _require(len({_grid_key(p["train"][0]["input"]) for p in puzzles}) == 42,
             "Different seeds duplicated their first input")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True}, {"num_test": False}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Illegal counts were accepted: {kwargs}")
