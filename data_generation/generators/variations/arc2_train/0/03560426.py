import numpy as np


_SOURCE_ROWS = (
    (("0000000000",)*6 + ("8880000000",)*2 + ("8880770222",)*2,
     ("8880000000",)*3 + ("8877000000", "0072220000", "0002220000") + ("0000000000",)*4),
    (("0000000000",)*6 + ("0000000004", "1100000004") + ("1102203304",)*2,
     ("1100000000",)*2 + ("1220000000", "0233000000", "0034000000") + ("0004000000",)*3 + ("0000000000",)*2),
    (("0000000000",)*5 + ("0000020000",)*2 + ("0000020333",) + ("4444020333",)*2,
     ("4444000000", "4442000000") + ("0002000000",)*3 + ("0003330000",)*3 + ("0000000000",)*2),
    (("0000000000",)*6 + ("7000000000", "7088060000") + ("7088060330",)*2,
     ("7000000000",)*3 + ("8800000000",)*2 + ("8600000000", "0600000000") + ("0330000000",)*2 + ("0000000000",)),
)


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _source_key(rows):
    return ((len(rows), len(rows[0])), tuple(int(ch) for row in rows for ch in row))


_SOURCE_PAIRS = frozenset((_source_key(a), _source_key(b)) for a, b in _SOURCE_ROWS)


def _required(condition, message):
    if not condition:
        raise AssertionError(message)


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _orient(grid, orientation):
    rotated = np.rot90(grid, orientation % 4)
    return np.fliplr(rotated) if orientation >= 4 else rotated.copy()


def _unorient(grid, orientation):
    if orientation >= 4:
        grid = np.fliplr(grid)
    return np.rot90(grid, -(orientation % 4)).copy()


def _paint_output(shape, specs, background):
    height, width = shape
    result = np.full(shape, background, dtype=np.int8)
    row = col = 0
    for color, piece_h, piece_w in specs:
        _required(row + piece_h <= height and col + piece_w <= width,
                  "chain exceeds canvas")
        result[row:row + piece_h, col:col + piece_w] = color
        row += piece_h - 1
        col += piece_w - 1
    return result


def _parse_canonical(grid):
    height, width = grid.shape
    background = int(grid[0, 0])
    _required(np.all(grid[0] == background), "input needs a blank opposite edge")
    counts = np.bincount(grid.ravel(), minlength=10)
    _required(counts[background] > max(int(counts[c]) for c in range(10) if c != background),
              "background is not the dominant field")
    occupied = grid != background
    _required(np.any(occupied), "empty input")
    baseline = int(np.flatnonzero(np.any(occupied, axis=1))[-1])
    _required(baseline > 0, "no blank row above pieces")
    baseline_columns = np.flatnonzero(occupied[baseline])
    _required(len(baseline_columns) > 0, "empty baseline")
    runs = []
    start = last = int(baseline_columns[0])
    for column in baseline_columns[1:]:
        column = int(column)
        if column != last + 1:
            runs.append((start, last))
            start = column
        last = column
    runs.append((start, last))
    _required(2 <= len(runs) <= 7, "piece count outside 2..7")
    reconstruction = np.full(grid.shape, background, dtype=grid.dtype)
    specs = []
    for left, right in runs:
        color = int(grid[baseline, left])
        _required(np.all(grid[baseline, left:right + 1] == color), "mixed baseline run")
        rows = np.flatnonzero(np.any(occupied[:, left:right + 1], axis=1))
        top = int(rows[0])
        _required(top > 0, "piece touches opposite edge")
        _required(np.all(grid[top:baseline + 1, left:right + 1] == color),
                  "piece is not a solid baseline rectangle")
        piece_h, piece_w = baseline - top + 1, right - left + 1
        _required((piece_h, piece_w) != (1, 1) or len(specs) == len(runs) - 1,
                  "nonterminal 1x1 piece is erased")
        reconstruction[top:baseline + 1, left:right + 1] = color
        specs.append((color, piece_h, piece_w))
    _required(np.array_equal(reconstruction, grid), "unexplained input cells")
    return background, specs, {
        "left": runs[0][0], "trailing": width - 1 - runs[-1][1],
        "gaps": tuple(runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)),
        "bottom": height - 1 - baseline,
    }


def _matching_orientations(pair):
    matches = {}
    for orientation in range(8):
        canonical_input = _unorient(pair["input"], orientation)
        canonical_output = _unorient(pair["output"], orientation)
        try:
            background, specs, layout = _parse_canonical(canonical_input)
            expected = _paint_output(canonical_input.shape, specs, background)
        except AssertionError:
            continue
        if np.array_equal(canonical_output, expected):
            matches[orientation] = (background, specs, layout, canonical_input.shape)
    return matches


def _predict_from_training(train_pairs, test_input):
    possible = set(range(8))
    for pair in train_pairs:
        possible &= set(_matching_orientations(pair))
    _required(len(possible) == 1, "training does not determine orientation")
    orientation = next(iter(possible))
    canonical = _unorient(test_input, orientation)
    background, specs, _ = _parse_canonical(canonical)
    return _orient(_paint_output(canonical.shape, specs, background), orientation)


def _pick_background(rng, shared, base, first, previous):
    if shared:
        return base
    choices = [c for c in range(10) if not (first and c == previous)]
    return int(rng.choice(choices))


def _pick_colors(rng, count, background, demonstration):
    available = [c for c in range(10) if c != background]
    if demonstration or rng.random() < 0.62:
        return [int(c) for c in rng.choice(available, size=count, replace=False)]
    colors = [int(rng.choice(available))]
    for _ in range(1, count):
        if rng.random() < 0.32:
            colors.append(int(rng.choice(colors)))
        else:
            colors.append(int(rng.choice(available)))
    return colors


def _sample_count(rng, index, split, style):
    if split == "train" and index == 0:
        return int(rng.integers(3, 5))
    if style == "source_shift":
        return int(rng.integers(3, 5)) if split == "train" else 4
    if split == "train" and index == 1:
        return int(rng.choice([2, 5, 6]))
    choices = [2, 3, 4, 5, 6, 7]
    weights = [0.17, 0.22, 0.25, 0.19, 0.12, 0.05]
    return int(rng.choice(choices, p=weights))


def _sample_specs(rng, count, colors, demonstration, source_shift_test, source_shift_train):
    widths = []
    heights = []
    special_bar = int(rng.integers(1, count)) if source_shift_test else -1
    training_bar = int(rng.integers(1, count)) if source_shift_train and rng.random() < 0.5 else -1
    for index in range(count):
        if demonstration:
            piece_w = int(rng.integers(2, 6))
            piece_h = int(rng.integers(2, 6))
        elif source_shift_test:
            piece_w = 1 if index in (0, special_bar) else int(rng.integers(2, 4))
            piece_h = int(rng.integers(2, 6)) if piece_w == 1 else int(rng.integers(2, 4))
        elif source_shift_train:
            piece_w = 1 if index == training_bar else int(rng.integers(2, 6))
            piece_h = int(rng.integers(2, 6))
        else:
            piece_w = 1 if rng.random() < 0.26 else int(rng.integers(2, 7))
            piece_h = 1 if rng.random() < 0.15 else int(rng.integers(2, 7))
            if piece_w == piece_h == 1 and index != count - 1:
                piece_h = int(rng.integers(2, 6))
        widths.append(piece_w)
        heights.append(piece_h)
    return [(colors[i], heights[i], widths[i]) for i in range(count)]


def _canvas_dim(rng, minimum):
    chance = rng.random()
    if chance < 0.25:
        return minimum
    if chance < 0.55:
        return min(30, minimum + int(rng.integers(1, 4)))
    if chance < 0.83:
        return min(30, minimum + int(rng.integers(4, 11)))
    if chance < 0.94:
        return int(rng.integers(minimum, 31))
    return 30


def _make_pair(rng, count, background, orientation, demonstration,
               source_shift_test, source_shift_train):
    colors = _pick_colors(rng, count, background, demonstration)
    for _ in range(512):
        specs = _sample_specs(rng, count, colors, demonstration,
                              source_shift_test, source_shift_train)
        chain_h = 1 + sum(h - 1 for _, h, _ in specs)
        chain_w = 1 + sum(w - 1 for _, _, w in specs)
        if source_shift_test and chain_h <= chain_w:
            continue
        gaps = [int(rng.integers(1, 5)) for _ in range(count - 1)]
        left = int(rng.integers(0, 5))
        strip_w = left + sum(w for _, _, w in specs) + sum(gaps)
        minimum_h = max(chain_h, max(h for _, h, _ in specs) + 1)
        minimum_w = max(chain_w, strip_w)
        if minimum_h > 30 or minimum_w > 30:
            continue
        canvas_h = _canvas_dim(rng, minimum_h)
        canvas_w = _canvas_dim(rng, minimum_w)
        if rng.random() < 0.16:
            side = _canvas_dim(rng, max(minimum_h, minimum_w))
            canvas_h = canvas_w = side
        max_bottom = canvas_h - max(h for _, h, _ in specs) - 1
        bottom = 0 if rng.random() < 0.38 else int(rng.integers(0, max_bottom + 1))
        baseline = canvas_h - 1 - bottom
        input_grid = np.full((canvas_h, canvas_w), background, dtype=np.int8)
        column = left
        for index, (color, piece_h, piece_w) in enumerate(specs):
            input_grid[baseline - piece_h + 1:baseline + 1,
                       column:column + piece_w] = color
            column += piece_w + (gaps[index] if index < count - 1 else 0)
        counts = np.bincount(input_grid.ravel(), minlength=10)
        if counts[background] <= max(int(counts[c]) for c in range(10) if c != background):
            continue
        output_grid = _paint_output(input_grid.shape, specs, background)
        pair = {"input": _orient(input_grid, orientation),
                "output": _orient(output_grid, orientation)}
        if _grid_key(pair["input"]) == _grid_key(pair["output"]):
            continue
        if (_grid_key(pair["input"]), _grid_key(pair["output"])) in _SOURCE_PAIRS:
            continue
        return pair
    raise ValueError("could not construct a fitting pair")


def validate_generated_puzzle(puzzle: dict) -> None:
    _required(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
              "puzzle must have train and test")
    _required(all(isinstance(puzzle[s], list) for s in ("train", "test")),
              "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _required(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
              "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    train_matches = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _required(isinstance(pair, dict) and set(pair) == {"input", "output"},
                      "pair must have input and output")
            for role in ("input", "output"):
                grid = pair[role]
                _required(isinstance(grid, np.ndarray) and grid.ndim == 2,
                          "grid must be a two-dimensional NumPy array")
                _required(all(1 <= x <= 30 for x in grid.shape), "illegal canvas size")
                _required(np.issubdtype(grid.dtype, np.integer), "grid must have integer dtype")
                _required(np.all((grid >= 0) & (grid <= 9)), "illegal color")
            _required(pair["input"].shape == pair["output"].shape,
                      "input/output dimensions differ")
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            _required(input_key not in seen_inputs and pair_key not in seen_pairs,
                      "duplicate input or pair")
            _required(pair_key not in _SOURCE_PAIRS, "source pair copied")
            _required(input_key != pair_key[1], "unchanged pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            matches = _matching_orientations(pair)
            _required(bool(matches), "pair violates corner-chain rule")
            if split == "train":
                train_matches.append(matches)
    orientations = set(range(8))
    for matches in train_matches:
        orientations &= set(matches)
    _required(len(orientations) == 1, "training must establish one orientation")
    orientation = next(iter(orientations))
    demonstrated_corner = False
    for pair, matches in zip(train, train_matches):
        _, specs, _, _ = matches[orientation]
        demonstrated_corner |= any(
            specs[i][0] != specs[i + 1][0]
            and specs[i][1] >= 2 and specs[i][2] >= 2
            and specs[i + 1][1] >= 2 and specs[i + 1][2] >= 2
            for i in range(len(specs) - 1)
        )
    _required(demonstrated_corner, "training must expose colored corner overwrite")
    for pair in test:
        _required(orientation in _matching_orientations(pair),
                  "test uses a different orientation")
        predicted = _predict_from_training(train, pair["input"].copy())
        _required(np.array_equal(predicted, pair["output"]),
                  "test output not determined by training and input")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    orientation = int(rng.integers(0, 8))
    source_shift = rng.random() < 0.22
    shared_background = rng.random() < 0.50
    base_background = int(rng.integers(0, 10))
    for _ in range(32):
        puzzle = {"train": [], "test": []}
        used_inputs = set()
        previous_background = None
        for split, total in (("train", train_count), ("test", test_count)):
            for index in range(total):
                background = _pick_background(
                    rng, shared_background, base_background,
                    split == "train" and index == 1, previous_background)
                if split == "train" and index == 0:
                    previous_background = background
                count = _sample_count(rng, index, split,
                                      "source_shift" if source_shift else "broad")
                for _ in range(64):
                    pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                    pair = _make_pair(pair_rng, count, background, orientation,
                                      split == "train" and index == 0,
                                      source_shift and split == "test",
                                      source_shift and split == "train" and index > 0)
                    key = _grid_key(pair["input"])
                    if key not in used_inputs:
                        used_inputs.add(key)
                        puzzle[split].append(pair)
                        break
                else:
                    raise ValueError("could not construct a unique pair")
        try:
            validate_generated_puzzle(puzzle)
            return puzzle
        except AssertionError:
            continue
    raise ValueError("could not construct an unambiguous puzzle")


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = {"count2": 0, "count6": 0, "train_leading_bar": 0,
                "train_multiple_bars": 0, "test_no_bar": 0, "test_wide": 0,
                "short_bar": 0, "horizontal": 0, "repeated_size": 0,
                "repeated_color": 0, "background_nonzero": 0,
                "background_varies": 0, "new_test_color": 0,
                "left_margin": 0, "gap_gt1": 0, "bottom_margin": 0,
                "trailing_zero": 0, "trailing_positive": 0,
                "exact_bottom": 0, "size30": 0, "square": 0}
    orientations = set()
    bar_positions = {"train": set(), "test": set()}
    shapes = set()
    layouts = set()
    structures = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        orientation = next(iter(set.intersection(*(
            set(_matching_orientations(p)) for p in puzzle["train"]))))
        orientations.add(orientation)
        train_colors = set()
        backgrounds = set()
        train_counts = set()
        train_shapes = set()
        original_train_profile = True
        original_test_profile = True
        for split in ("train", "test"):
            for pair in puzzle[split]:
                background, specs, layout, shape = _matching_orientations(pair)[orientation]
                backgrounds.add(background)
                if split == "train":
                    train_colors.update(c for c, _, _ in specs)
                    train_counts.add(len(specs))
                    train_shapes.add(tuple((h, w) for _, h, w in specs))
                    original_train_profile &= (len(specs) in (3, 4) and specs[0][2] > 1
                                               and sum(w == 1 for _, _, w in specs) <= 1)
                else:
                    features["new_test_color"] += any(c not in train_colors for c, _, _ in specs)
                    original_test_profile &= (len(specs) == 4 and specs[0][2] == 1
                                              and sum(w == 1 for _, _, w in specs) == 2)
                features["count2"] += len(specs) == 2
                features["count6"] += len(specs) >= 6
                features["short_bar"] += any(w == 1 and h in (2, 3) for _, h, w in specs)
                features["horizontal"] += any(h == 1 and w > 1 for _, h, w in specs)
                features["repeated_size"] += len({(h, w) for _, h, w in specs}) < len(specs)
                features["repeated_color"] += len({c for c, _, _ in specs}) < len(specs)
                features["background_nonzero"] += background != 0
                features["left_margin"] += layout["left"] > 0
                features["gap_gt1"] += any(g > 1 for g in layout["gaps"])
                features["bottom_margin"] += layout["bottom"] > 0
                features["trailing_zero"] += layout["trailing"] == 0
                features["trailing_positive"] += layout["trailing"] > 0
                features["exact_bottom"] += 1 + sum(h - 1 for _, h, _ in specs) == shape[0]
                features["size30"] += 30 in pair["input"].shape
                features["square"] += pair["input"].shape[0] == pair["input"].shape[1]
                for position, (_, _, width) in enumerate(specs):
                    if width == 1:
                        bar_positions[split].add(position)
                if split == "train":
                    features["train_leading_bar"] += specs[0][2] == 1
                    features["train_multiple_bars"] += sum(w == 1 for _, _, w in specs) >= 2
                else:
                    features["test_no_bar"] += all(w > 1 for _, _, w in specs)
                    features["test_wide"] += 1 + sum(w - 1 for _, _, w in specs) > 1 + sum(h - 1 for _, h, _ in specs)
                    predicted = _predict_from_training(puzzle["train"], pair["input"].copy())
                    _required(np.array_equal(predicted, pair["output"]), "prediction mismatch")
                shapes.add(tuple(pair["input"].shape))
                signature = (orientation, tuple((h, w) for _, h, w in specs), layout["gaps"],
                             layout["left"], layout["trailing"], layout["bottom"])
                layouts.add(signature)
                structures.add((orientation, tuple((h, w) for _, h, w in specs)))
        features["background_varies"] += len(backgrounds) > 1
        features.setdefault("within_train_count_varies", 0)
        features.setdefault("within_train_shape_varies", 0)
        features.setdefault("source_shift_profile", 0)
        features["within_train_count_varies"] += len(train_counts) > 1
        features["within_train_shape_varies"] += len(train_shapes) > 1
        features["source_shift_profile"] += original_train_profile and original_test_profile
    _required(all(value > 0 for value in features.values()),
              f"diversity property absent: {[k for k,v in features.items() if v == 0]}")
    _required(len(orientations) >= 6 and len(shapes) >= 30 and len(layouts) >= 100
              and len(structures) >= 80, "structural diversity too low")
    _required(all(len(bar_positions[split]) >= 5 for split in ("train", "test")),
              "bar positions lack diversity")
    _required(len({len(p["train"]) for p in puzzles}) >= 4 and
              len({len(p["test"]) for p in puzzles}) >= 3, "automatic counts lack diversity")
    _required(len({_puzzle_key(p) for p in puzzles}) == 42, "duplicate puzzles")
    for offset, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + offset, train, test)
        _required((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                  "explicit counts ignored")
        validate_generated_puzzle(puzzle)
    for offset, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + offset, train, test)
        _required((train is None or len(puzzle["train"]) == train) and
                  (test is None or len(puzzle["test"]) == test), "partial count ignored")
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    a = generate_puzzle(912345, 7, 3)
    _required(_puzzle_key(a) == _puzzle_key(generate_puzzle(912345, 7, 3)),
              "generation is nondeterministic")
    _required(_puzzle_key(a) != _puzzle_key(generate_puzzle(912346, 7, 3)),
              "neighbor seeds duplicate")
