import numpy as np


# NW, N, NE, W, E, SW, S, SE.  The center is never moved.
_PERIMETER = ((0, 0), (0, 1), (0, 2), (1, 0),
              (1, 2), (2, 0), (2, 1), (2, 2))
_MAPPING = (2, 3, 7, 6, 1, 0, 4, 5)
_CORNERS = (0, 2, 5, 7)
_SIDES = (1, 3, 4, 6)


def _digits(*rows):
    return tuple(tuple(int(char) for char in row) for row in rows)


_SOURCE_PAIRS = frozenset((
    (_digits("9657777", "8717777", "0897777", "7777777", "7771847", "7774767", "7776247"),
     _digits("5897777", "8767777", "9107777", "7777777", "7774447", "7772787", "7771667")),
    (_digits("7777777", "7777777", "7752877", "7717977", "7743077", "7777777", "7777777"),
     _digits("7777777", "7777777", "7781077", "7737277", "7759477", "7777777", "7777777")),
    (_digits("7777777", "7776557", "7775767", "7771517", "8887777", "9797777", "0007777"),
     _digits("7777777", "7775517", "7775757", "7776617", "8907777", "0787777", "8907777")),
    (_digits("7777777", "2657777", "1797019", "4097671", "7777592", "7777777", "7777777"),
     _digits("7777777", "5197777", "0767962", "2947971", "7777015", "7777777", "7777777")),
))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _check_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(rng, num_train, num_test):
    _check_count("num_train", num_train, 2, 10)
    _check_count("num_test", num_test, 1, 4)
    train = None if num_train is None else int(num_train)
    test = None if num_test is None else int(num_test)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_tuple(grid):
    return tuple(tuple(int(value) for value in row) for row in grid)


def _grid_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.ravel()))


def _source_copy(inp, out):
    return (_grid_tuple(inp), _grid_tuple(out)) in _SOURCE_PAIRS


def _find_tiles(grid, background):
    height, width = grid.shape
    return [(row, col) for row in range(height - 2) for col in range(width - 2)
            if int(grid[row + 1, col + 1]) == background
            and all(int(grid[row + dr, col + dc]) != background
                    for dr, dc in _PERIMETER)]


def _decompose(grid):
    """Find the unique complete-tile cover, without using color frequency."""
    height, width = grid.shape
    solutions = []
    for background in np.unique(grid):
        background = int(background)
        tiles = _find_tiles(grid, background)
        if not tiles:
            continue
        covered = set()
        occupied = set()
        valid = True
        for row, col in tiles:
            box = {(row + dr, col + dc) for dr in range(3) for dc in range(3)}
            if occupied.intersection(box):
                valid = False
                break
            occupied.update(box)
            covered.update((row + dr, col + dc) for dr, dc in _PERIMETER)
        if not valid:
            continue
        foreground = {(row, col) for row in range(height) for col in range(width)
                      if int(grid[row, col]) != background}
        if covered == foreground:
            solutions.append((background, tiles))
    _require(len(solutions) == 1, "background or tile decomposition is ambiguous")
    return solutions[0]


def _payload(grid, row, col):
    return tuple(int(grid[row + dr, col + dc]) for dr, dc in _PERIMETER)


def _apply_mapping(grid, background, tiles, mapping):
    output = grid.copy()
    for row, col in tiles:
        original = _payload(grid, row, col)
        for (dr, dc), source in zip(_PERIMETER, mapping):
            output[row + dr, col + dc] = original[source]
    return output


def _learn_mapping(train):
    inputs = []
    outputs = []
    for pair in train:
        _, tiles = _decompose(pair["input"])
        for row, col in tiles:
            inputs.append(_payload(pair["input"], row, col))
            outputs.append(_payload(pair["output"], row, col))
    _require(bool(inputs), "training has no examples of the tile operation")
    mapping = []
    for destination in range(8):
        possibilities = [source for source in range(8)
                         if all(after[destination] == before[source]
                                for before, after in zip(inputs, outputs))]
        _require(len(possibilities) == 1, "training does not determine every perimeter movement")
        mapping.append(possibilities[0])
    _require(len(set(mapping)) == 8, "learned perimeter movement is not a permutation")
    return tuple(mapping)


def _canvas_size(rng, min_height, min_width, density):
    def extra(room):
        if density == "tight":
            return int(rng.integers(0, min(room, 3) + 1))
        if density == "compact":
            return int(rng.integers(0, min(room, 7) + 1))
        if density == "large":
            low = min(room, 10)
            return int(rng.integers(low, room + 1))
        return int(rng.integers(0, min(room, 16) + 1))

    vertical = extra(30 - min_height)
    horizontal = extra(30 - min_width)
    if density == "long":
        if int(rng.integers(0, 2)):
            vertical = int(rng.integers(0, min(30 - min_height, 3) + 1))
            horizontal = int(rng.integers(min(30 - min_width, 9), 31 - min_width))
        else:
            horizontal = int(rng.integers(0, min(30 - min_width, 3) + 1))
            vertical = int(rng.integers(min(30 - min_height, 9), 31 - min_height))
    return min_height + vertical, min_width + horizontal


def _two_positions(rng, category, density):
    short = int(rng.integers(1, 3))
    far_r = int(rng.integers(3, 13))
    far_c = int(rng.integers(3, 13))
    if category == "horizontal":
        dr, dc = 0, far_c
    elif category == "vertical":
        dr, dc = far_r, 0
    elif category == "horizontal_stagger":
        dr, dc = short, far_c
    elif category == "vertical_stagger":
        dr, dc = far_r, short
    else:
        dr, dc = far_r, far_c
    if int(rng.integers(0, 2)):
        dr = -dr
    if int(rng.integers(0, 2)):
        dc = -dc
    points = [(0, 0), (dr, dc)]
    low_r = min(point[0] for point in points)
    low_c = min(point[1] for point in points)
    points = [(row - low_r, col - low_c) for row, col in points]
    min_height = max(row for row, _ in points) + 3
    min_width = max(col for _, col in points) + 3
    height, width = _canvas_size(rng, min_height, min_width, density)
    row_margin = int(rng.integers(0, height - min_height + 1))
    col_margin = int(rng.integers(0, width - min_width + 1))
    return height, width, [(r + row_margin, c + col_margin) for r, c in points]


def _many_positions(rng, count, density):
    shapes = [(rows, cols) for rows in range(1, count + 1)
              for cols in range(1, count + 1)
              if count <= rows * cols <= count + 2 and rows * 3 <= 30 and cols * 3 <= 30]
    rows, cols = shapes[int(rng.integers(0, len(shapes)))]
    height, width = _canvas_size(rng, 3 * rows, 3 * cols, density)
    row_slack = rng.multinomial(height - 3 * rows, np.ones(rows + 2) / (rows + 2))
    col_slack = rng.multinomial(width - 3 * cols, np.ones(cols + 2) / (cols + 2))
    row_start = []
    col_start = []
    at = int(row_slack[0])
    for index in range(rows):
        row_start.append(at)
        at += 3 + int(row_slack[index + 1])
    at = int(col_slack[0])
    for index in range(cols):
        col_start.append(at)
        at += 3 + int(col_slack[index + 1])
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    chosen = rng.choice(len(slots), size=count, replace=False)
    positions = []
    for index in chosen:
        r, c = slots[int(index)]
        positions.append((row_start[r] + int(rng.integers(0, int(row_slack[r + 1]) + 1)),
                          col_start[c] + int(rng.integers(0, int(col_slack[c + 1]) + 1))))
    return height, width, positions


_LAYOUTS = ("diagonal", "horizontal_stagger", "vertical_stagger", "horizontal", "vertical")


def _layout_choice(rng, split, bias):
    if bias == "source_shift":
        preferred = "diagonal" if split == "train" else "horizontal_stagger"
    elif bias == "reverse_shift":
        preferred = "horizontal_stagger" if split == "train" else "diagonal"
    else:
        preferred = None
    if preferred is not None and rng.random() < 0.65:
        return preferred
    return _LAYOUTS[int(rng.integers(0, len(_LAYOUTS)))]


def _sample_positions(rng, count, split, bias):
    density = ("tight", "compact", "medium", "large", "long")[int(rng.choice(5, p=[.20, .24, .28, .16, .12]))]
    if count == 4 and density == "tight" and rng.random() < .55:
        positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
        rng.shuffle(positions)
        return 6, 6, positions
    if count == 1:
        if density == "tight" and rng.random() < .35:
            return 3, 3, [(0, 0)]
        height, width = _canvas_size(rng, 3, 3, density)
        return height, width, [(int(rng.integers(0, height - 2)),
                                int(rng.integers(0, width - 2)))]
    if count == 2:
        return _two_positions(rng, _layout_choice(rng, split, bias), density)
    return _many_positions(rng, count, density)


def _two_color_evidence(rng, background, bits):
    colors = rng.choice([color for color in range(10) if color != background], size=2, replace=False)
    return tuple(int(colors[int(bit)]) for bit in bits)


def _sample_payload(rng, background, mode):
    colors = np.array([color for color in range(10) if color != background], dtype=np.int8)
    if mode == "distinct":
        return tuple(int(v) for v in rng.choice(colors, size=8, replace=False))
    if mode == "stationary":
        chosen = rng.choice(colors, size=2, replace=False)
        if rng.random() < .45:
            chosen[1] = chosen[0]
        values = [0] * 8
        for index in _CORNERS:
            values[index] = int(chosen[0])
        for index in _SIDES:
            values[index] = int(chosen[1])
        return tuple(values)
    if mode == "bands":
        chosen = rng.choice(colors, size=3, replace=False)
        return tuple(int(chosen[0] if index in (0, 1, 2)
                         else chosen[1] if index in (3, 4)
                         else chosen[2]) for index in range(8))
    if mode == "alternating":
        chosen = rng.choice(colors, size=4, replace=False)
        return tuple(int(chosen[index]) for index in (0, 2, 1, 2, 3, 1, 3, 0))
    size = int(rng.integers(2, 8))
    palette = rng.choice(colors, size=size, replace=False)
    return tuple(int(v) for v in rng.choice(palette, size=8, replace=True))


def _make_pair(rng, count, split, bias, background, payload_overrides, stationary_pair, low_evidence):
    height, width, positions = _sample_positions(rng, count, split, bias)
    grid = np.full((height, width), background, dtype=np.int8)
    payloads = []
    for index, (row, col) in enumerate(positions):
        if index in payload_overrides:
            values = payload_overrides[index]
        elif stationary_pair:
            values = _sample_payload(rng, background, "stationary")
        elif payloads and rng.random() < .11:
            values = payloads[int(rng.integers(0, len(payloads)))]
        else:
            modes = ("random", "bands", "alternating", "stationary")
            mode = modes[int(rng.choice(4, p=[.46, .22, .19, .13]))]
            values = _sample_payload(rng, background, mode)
            if low_evidence and len(set(values)) == 8:
                values = _sample_payload(rng, background, "bands")
        payloads.append(values)
        for (dr, dc), value in zip(_PERIMETER, values):
            grid[row + dr, col + dc] = value
    detected_background, detected_tiles = _decompose(grid)
    _require(detected_background == background and set(detected_tiles) == set(positions),
             "constructed scene has an ambiguous tile cover")
    output = _apply_mapping(grid, background, detected_tiles, _MAPPING)
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(rng, num_train, num_test)
    bias = ("mixed", "source_shift", "reverse_shift")[int(rng.choice(3, p=[.55, .30, .15]))]
    evidence_style = "distinct" if rng.random() < .66 else "joint"
    shared_background = bool(rng.random() < .42)
    if shared_background:
        backgrounds = [int(rng.integers(0, 10))] * (n_train + n_test)
    else:
        backgrounds = []
        while len(backgrounds) < n_train + n_test:
            backgrounds.extend(int(v) for v in rng.permutation(10))
        backgrounds = backgrounds[:n_train + n_test]

    weights = np.array([.18, .24, .18, .16, .12, .08, .04])
    counts = [int(rng.choice(np.arange(1, 8), p=weights)) for _ in range(n_train + n_test)]
    if max(counts[:n_train]) == 1:
        counts[int(rng.integers(0, n_train))] = int(rng.integers(2, 6))
    if evidence_style == "joint" and sum(counts[:n_train]) < 3:
        counts[int(rng.integers(0, n_train))] = 2

    overrides = [dict() for _ in counts]
    if evidence_style == "distinct":
        pair_index = int(rng.integers(0, n_train))
        tile_index = int(rng.integers(0, counts[pair_index]))
        overrides[pair_index][tile_index] = _sample_payload(rng, backgrounds[pair_index], "distinct")
    else:
        slots = [(i, j) for i in range(n_train) for j in range(counts[i])]
        selected = rng.choice(len(slots), size=3, replace=False)
        codes = rng.permutation(8)
        for bit_index, slot_index in enumerate(selected):
            pair_index, tile_index = slots[int(slot_index)]
            bits = [(int(code) >> bit_index) & 1 for code in codes]
            overrides[pair_index][tile_index] = _two_color_evidence(
                rng, backgrounds[pair_index], bits)

    puzzle = {"train": [], "test": []}
    seen = set()
    for index, count in enumerate(counts):
        split = "train" if index < n_train else "test"
        stationary_pair = (split == "test" and rng.random() < .11)
        for attempt in range(80):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            try:
                pair = _make_pair(pair_rng, count, split, bias, backgrounds[index],
                                  overrides[index], stationary_pair,
                                  evidence_style == "joint")
            except AssertionError:
                continue
            key = _grid_key(pair["input"])
            if key in seen or _source_copy(pair["input"], pair["output"]):
                continue
            seen.add(key)
            puzzle[split].append(pair)
            break
        else:
            raise ValueError(f"could not construct unique pair {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair must have input and output")
    inp, out = pair["input"], pair["output"]
    for name, grid in (("input", inp), ("output", out)):
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                 f"{name} must be a two-dimensional NumPy array")
        _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                 f"{name} dimensions are invalid")
        _require(np.issubdtype(grid.dtype, np.integer), f"{name} dtype must be integer")
        _require(0 <= int(grid.min()) and int(grid.max()) <= 9,
                 f"{name} colors are invalid")
    _require(inp.shape == out.shape, "input and output dimensions differ")
    background, tiles = _decompose(inp)
    _require(np.array_equal(out, _apply_mapping(inp, background, tiles, _MAPPING)),
             "output violates the opposite quarter-turn rule")
    _require(_source_copy(inp, out) is False, "generated pair copies a source pair")
    _require(all(int(out[row + 1, col + 1]) == background for row, col in tiles),
             "tile center changed")
    return background, tiles


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test,
             "invalid pair counts")
    input_keys = set()
    train_tile_counts = []
    decompositions = []
    for index, pair in enumerate(puzzle["train"] + puzzle["test"]):
        background, tiles = _inspect_pair(pair)
        key = _grid_key(pair["input"])
        _require(key not in input_keys, "duplicate input grid")
        input_keys.add(key)
        decompositions.append((background, tiles))
        if index < n_train:
            train_tile_counts.append(len(tiles))
    _require(max(train_tile_counts) >= 2, "training never demonstrates independent tiles")
    learned = _learn_mapping(puzzle["train"])
    _require(learned == _MAPPING, "training implies a contradictory movement")
    for pair, (background, tiles) in zip(puzzle["test"], decompositions[n_train:]):
        predicted = _apply_mapping(pair["input"], background, tiles, learned)
        _require(np.array_equal(predicted, pair["output"]),
                 "test output cannot be inferred from training")


def _puzzle_signature(puzzle):
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(987654321, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {kwargs}")


def _two_layout(tiles):
    (r1, c1), (r2, c2) = tiles
    dr, dc = abs(r2 - r1), abs(c2 - c1)
    if dr == 0:
        return "horizontal"
    if dc == 0:
        return "vertical"
    if dr < 3:
        return "horizontal_stagger"
    if dc < 3:
        return "vertical_stagger"
    return "diagonal"


def self_test() -> None:
    train_counts, test_counts, shapes, backgrounds, layouts = set(), set(), set(), set(), set()
    tile_counts, signatures = set(), set()
    background_changes = stationary_tiles = repeated_tiles = unchanged_pairs = 0
    tight = large = long = joint_evidence = 0
    packed_four = tiny = nonmodal_background = banded = alternating = 0
    late_diagnostic = multitile_diagnostic = heldout_test_layout = 0
    varied_pair_counts = large_multi = tight_multi = stationary_multi = 0
    layout_by_split = {"train": set(), "test": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_backgrounds = set()
        local_counts = set()
        local_layouts = {"train": set(), "test": set()}
        train_payloads = []
        for split in ("train", "test"):
            for pair_index, pair in enumerate(puzzle[split]):
                inp, out = pair["input"], pair["output"]
                background, tiles = _decompose(inp)
                backgrounds.add(background)
                local_backgrounds.add(background)
                local_counts.add(len(tiles))
                shapes.add(inp.shape)
                tile_counts.add(len(tiles))
                counts = np.bincount(inp.ravel().astype(np.int64), minlength=10)
                if counts[background] < int(counts.max()):
                    nonmodal_background += 1
                if inp.shape == (6, 6) and len(tiles) == 4:
                    packed_four += 1
                if inp.shape == (3, 3):
                    tiny += 1
                if inp.shape[0] * inp.shape[1] <= 9 * len(tiles) + 4:
                    tight += 1
                    if len(tiles) >= 3:
                        tight_multi += 1
                if max(inp.shape) >= 24:
                    large += 1
                    if len(tiles) >= 3:
                        large_multi += 1
                if max(inp.shape) >= 3 * min(inp.shape):
                    long += 1
                if len(tiles) == 2:
                    layout = _two_layout(tiles)
                    layouts.add(layout)
                    layout_by_split[split].add(layout)
                    local_layouts[split].add(layout)
                payloads = [_payload(inp, row, col) for row, col in tiles]
                if len(set(payloads)) < len(payloads):
                    repeated_tiles += 1
                for values in payloads:
                    if tuple(values[i] for i in _MAPPING) == values:
                        stationary_tiles += 1
                        if len(tiles) >= 3:
                            stationary_multi += 1
                    if len(set(values[:3])) == 1 and values[3] == values[4] and len(set(values[5:])) == 1:
                        banded += 1
                    if values[0] == values[7] and values[2] == values[5] and values[1] == values[3] and values[4] == values[6] and len(set(values)) >= 3:
                        alternating += 1
                    if split == "train" and len(set(values)) == 8:
                        if pair_index > 0:
                            late_diagnostic += 1
                        if len(tiles) > 1:
                            multitile_diagnostic += 1
                if np.array_equal(inp, out):
                    unchanged_pairs += 1
                if split == "train":
                    train_payloads.extend(payloads)
                # Reconstruct from pixels for every pair, including unchanged ones.
                expected = _apply_mapping(inp, background, tiles, _MAPPING)
                _require(np.array_equal(out, expected), "self-test rule mismatch")
                _require(not _source_copy(inp, out), "self-test source copy")
        if len(local_backgrounds) > 1:
            background_changes += 1
        if len(local_counts) > 1:
            varied_pair_counts += 1
        if local_layouts["test"] - local_layouts["train"]:
            heldout_test_layout += 1
        if all(len(set(values)) <= 7 for values in train_payloads):
            joint_evidence += 1
        _require(_learn_mapping(puzzle["train"]) == _MAPPING,
                 "self-test training evidence failure")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic count diversity is insufficient")
    _require(len(signatures) == 42 and len(shapes) >= 35 and len(backgrounds) == 10,
             "cross-seed grid diversity is insufficient")
    _require(tile_counts.issuperset({1, 2, 3, 4, 5}), "tile counts lack range")
    _require(len(layouts) == 5 and all(len(v) >= 4 for v in layout_by_split.values()),
             "two-tile layouts lack variety in one split")
    _require(background_changes >= 10 and joint_evidence >= 5,
             "background or demonstration modes lack variety")
    _require(min(tight, large, long, stationary_tiles, repeated_tiles, unchanged_pairs) > 0,
             "size or payload regimes lack variety")
    _require(min(packed_four, tiny, nonmodal_background, banded, alternating) > 0,
             "compact geometry or repeated-color patterns lack variety")
    _require(min(late_diagnostic, multitile_diagnostic, heldout_test_layout) > 0,
             "demonstration order or split layouts lack variety")
    _require(min(varied_pair_counts, large_multi, tight_multi, stationary_multi) > 0,
             "pair-level variation axes fail to combine")

    _require(_puzzle_signature(generate_puzzle(123456, 7, 3)) ==
             _puzzle_signature(generate_puzzle(123456, 7, 3)),
             "explicit generation is not deterministic")
    _require(_puzzle_signature(generate_puzzle(654321)) ==
             _puzzle_signature(generate_puzzle(654321)),
             "automatic generation is not deterministic")
    for index, (n_train, n_test) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        puzzle = generate_puzzle(2000 + index, n_train, n_test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (n_train, n_test),
                 "explicit counts not honored")
    for index, arguments in enumerate(({"num_train": None, "num_test": 1},
                                       {"num_train": None, "num_test": 4},
                                       {"num_train": 2, "num_test": None},
                                       {"num_train": 10, "num_test": None})):
        puzzle = generate_puzzle(3000 + index, **arguments)
        for name, split in (("num_train", "train"), ("num_test", "test")):
            if arguments[name] is not None:
                _require(len(puzzle[split]) == arguments[name], "partial count not honored")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        _expect_value_error(**arguments)
