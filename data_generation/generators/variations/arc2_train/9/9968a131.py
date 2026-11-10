import numpy as np


_SOURCE_PAIRS = (
    (
        [[7, 3, 8, 7], [7, 8, 3, 7], [7, 3, 8, 7], [7, 8, 3, 7]],
        [[7, 3, 8, 7], [7, 7, 8, 3], [7, 3, 8, 7], [7, 7, 8, 3]],
    ),
    (
        [[0, 5, 7, 7, 7, 7], [5, 0, 7, 7, 7, 7]] * 3,
        [[0, 5, 7, 7, 7, 7], [7, 5, 0, 7, 7, 7]] * 3,
    ),
    (
        [[7, 7, 7, 7, 7], [7, 2, 9, 7, 7], [7, 9, 2, 7, 7],
         [7, 2, 9, 7, 7], [7, 7, 7, 7, 7]],
        [[7, 7, 7, 7, 7], [7, 2, 9, 7, 7], [7, 7, 9, 2, 7],
         [7, 2, 9, 7, 7], [7, 7, 7, 7, 7]],
    ),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        choices = [(tr, te) for te in range(1, 5)
                   for tr in range(max(2, te + 1), 11)]
        return choices[int(rng.integers(len(choices)))]
    if num_train is None:
        return int(rng.integers(max(2, num_test + 1), 11)), num_test
    return num_train, int(rng.integers(1, min(4, num_train - 1) + 1))


def _padding(rng, regime, compact=False):
    # All four margin types occur in both regimes. Padded and longer runs
    # are somewhat more frequent in tests, without encoding a label rule.
    if compact:
        weights = ((0.48, 0.18, 0.25, 0.09) if regime == "source_train"
                   else (0.27, 0.22, 0.35, 0.16))
        maximum = 3
    else:
        weights = ((0.30, 0.27, 0.19, 0.24) if regime == "source_train"
                   else (0.16, 0.27, 0.25, 0.32))
        maximum = 7
    mode = int(rng.choice(4, p=weights))
    if mode == 0:
        return 0, 0
    if mode == 1:
        side = int(rng.integers(1, maximum))
        return (0, side) if int(rng.integers(2)) else (side, 0)
    if mode == 2:
        side = int(rng.integers(1, maximum))
        return side, side
    a = int(rng.integers(1, maximum))
    b = int(rng.integers(1, maximum - 1))
    if b >= a:
        b += 1
    return a, b


def _make_pair(rng, convention, orientation, length, margins, compact):
    progress_sign, lateral_sign = convention
    low, high = margins
    width = int(rng.integers(4, 9 if compact else 17))
    if lateral_sign == 1:
        first = int(rng.integers(0, width - 2))
    else:
        first = int(rng.integers(2, width))
    second = first + lateral_sign
    third = second + lateral_sign
    background, companion, spine = map(int, rng.choice(10, 3, replace=False))
    source = np.full((length + low + high, width), background, dtype=np.int8)
    rows = np.arange(low, low + length)
    if progress_sign == -1:
        rows = rows[::-1]
    for i, row in enumerate(rows):
        source[row, first] = companion if i % 2 == 0 else spine
        source[row, second] = spine if i % 2 == 0 else companion
    target = source.copy()
    for row in rows[1::2]:
        target[row, first] = background
        target[row, second] = spine
        target[row, third] = companion
    if orientation == 1:
        return source.T.copy(), target.T.copy()
    return source, target


def _models(grid, orientation, convention):
    progress_sign, lateral_sign = convention
    view = grid if orientation == 0 else grid.T
    height, width = view.shape
    if width < 4 or len(np.unique(view)) != 3:
        return []
    models = []
    for bg_value in np.unique(view):
        bg = int(bg_value)
        active = np.flatnonzero(np.any(view != bg, axis=1))
        if len(active) < 3 or len(active) > 18:
            continue
        if active[-1] - active[0] + 1 != len(active):
            continue
        positions = np.flatnonzero(view[active[0]] != bg)
        if len(positions) != 2 or positions[1] != positions[0] + 1:
            continue
        if any(not np.array_equal(np.flatnonzero(view[r] != bg), positions)
               for r in active):
            continue
        first = int(positions[0] if lateral_sign == 1 else positions[1])
        second = first + lateral_sign
        third = second + lateral_sign
        if not 0 <= third < width or np.any(view[active, third] != bg):
            continue
        ordered = active if progress_sign == 1 else active[::-1]
        companion = int(view[ordered[0], first])
        spine = int(view[ordered[0], second])
        if len({bg, companion, spine}) != 3:
            continue
        if any(int(view[r, first]) != (companion if i % 2 == 0 else spine)
               or int(view[r, second]) != (spine if i % 2 == 0 else companion)
               for i, r in enumerate(ordered)):
            continue
        result = view.copy()
        for row in ordered[1::2]:
            result[row, first] = bg
            result[row, second] = spine
            result[row, third] = companion
        if orientation == 1:
            result = result.T.copy()
        metadata = {
            "orientation": orientation, "length": len(active),
            "low": int(active[0]), "high": int(height - 1 - active[-1]),
            "width": width, "band": (int(positions[0]), int(positions[1])),
            "background": bg, "companion": companion, "spine": spine,
        }
        models.append((result, metadata))
    return models


def _matching_models(input_grid, output_grid, convention):
    return [(expected, meta) for orientation in (0, 1)
            for expected, meta in _models(input_grid, orientation, convention)
            if np.array_equal(expected, output_grid)]


def _candidates(input_grid, output_grid):
    return {(p, l) for p in (-1, 1) for l in (-1, 1)
            if _matching_models(input_grid, output_grid, (p, l))}


def _key(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def _source_copy(a, b):
    return any(np.array_equal(a, x) and np.array_equal(b, y)
               for x, y in _SOURCE_PAIRS)


def _schedule(rng, train_count, test_count):
    compact = bool(rng.random() < 0.18)
    mixed = bool(rng.random() < (0.25 if compact else 0.68))
    if mixed:
        train_orientations = [0, 1] + [int(rng.integers(2))
                                         for _ in range(train_count - 2)]
        rng.shuffle(train_orientations)
        test_orientations = [int(rng.integers(2)) for _ in range(test_count)]
    else:
        orientation = int(rng.integers(2))
        train_orientations = [orientation] * train_count
        test_orientations = [orientation] * test_count
    train_lengths = [int(rng.integers(3, 9 if compact else 17))
                     for _ in range(train_count)]
    test_lengths = [int(rng.integers(3, 10 if compact else 19))
                    for _ in range(test_count)]
    # Even demonstrations identify the progression end. In a rotated puzzle,
    # every orientation that occurs in tests has its own such demonstration.
    for orientation in set(train_orientations):
        indices = [i for i, value in enumerate(train_orientations)
                   if value == orientation]
        if not any(train_lengths[i] % 2 == 0 for i in indices):
            chosen = int(rng.choice(indices))
            train_lengths[chosen] += 1
    train_margins = [_padding(rng, "source_train", compact)
                     for _ in range(train_count)]
    test_margins = [_padding(rng, "target_test", compact)
                    for _ in range(test_count)]
    all_lengths = train_lengths + test_lengths
    if len(set(all_lengths)) == 1:
        test_lengths[0] = 3 if test_lengths[0] != 3 else 4
    if len(set(train_margins + test_margins)) == 1:
        test_margins[0] = (0, 0) if test_margins[0] != (0, 0) else (0, 1)
    return (train_orientations, test_orientations,
            train_lengths, test_lengths, train_margins, test_margins, compact)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    convention = (1 if int(rng.integers(2)) else -1,
                  1 if int(rng.integers(2)) else -1)
    orientations_tr, orientations_te, lengths_tr, lengths_te, margins_tr, margins_te, compact = (
        _schedule(rng, train_count, test_count)
    )
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count, orientations, lengths, margins in (
        ("train", train_count, orientations_tr, lengths_tr, margins_tr),
        ("test", test_count, orientations_te, lengths_te, margins_te),
    ):
        for i in range(count):
            for attempt in range(100):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(pair_seed)
                a, b = _make_pair(pair_rng, convention, orientations[i],
                                  lengths[i], margins[i], compact)
                if _key(a) in seen or _source_copy(a, b):
                    continue
                matches = _matching_models(a, b, convention)
                if len(matches) != 1:
                    continue
                seen.add(_key(a))
                puzzle[section].append({"input": a, "output": b})
                break
            else:
                raise ValueError(f"could not construct distinct {section} pair {i}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train and test must be lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    candidate_sets = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain input and output")
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= d <= 30 for d in grid.shape),
                         "grid dimensions out of bounds")
                _require(np.issubdtype(grid.dtype, np.integer)
                         and int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid colors or dtype invalid")
            _require(a.shape == b.shape and not np.array_equal(a, b),
                     "pair must change cells without resizing")
            _require(len(np.unique(a)) == 3 and len(np.unique(b)) == 3,
                     "pair must use exactly three colors")
            key = _key(a)
            pair_key = (key, _key(b))
            _require(key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            _require(not _source_copy(a, b), "exact source pair copied")
            candidates = _candidates(a, b)
            _require(bool(candidates), "pair violates the domino rule")
            if section == "train":
                candidate_sets.append(candidates)
    shared = set.intersection(*candidate_sets)
    _require(len(shared) == 1, "training does not identify one shared convention")
    convention = next(iter(shared))
    trained_even_orientations = set()
    trained_orientations = set()
    features = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            a, b = pair["input"], pair["output"]
            models = _matching_models(a, b, convention)
            _require(len(models) == 1, "ambiguous pair interpretation")
            expected, meta = models[0]
            _require(np.array_equal(expected, b), "incorrect output")
            _require(3 <= meta["length"] <= 18 and
                     0 <= meta["low"] <= 6 and 0 <= meta["high"] <= 6 and
                     4 <= meta["width"] <= 16, "pair outside the supported regime")
            for color in (meta["companion"], meta["spine"]):
                _require(int(np.count_nonzero(a == color)) ==
                         int(np.count_nonzero(b == color)),
                         "motif color count changed")
            features.append((meta["length"], meta["low"], meta["high"]))
            if section == "train":
                trained_orientations.add(meta["orientation"])
                if meta["length"] % 2 == 0:
                    trained_even_orientations.add(meta["orientation"])
            else:
                _require(meta["orientation"] in trained_orientations,
                         "test orientation absent from training")
                if meta["length"] % 2 == 0:
                    _require(meta["orientation"] in trained_even_orientations,
                             "even test lacks a directional demonstration")
    _require(trained_orientations == trained_even_orientations,
             "each trained orientation needs an even example")
    _require(len(set(f[0] for f in features)) >= 2 and
             len(set((f[1], f[2]) for f in features)) >= 2,
             "within-puzzle length and margin variation required")


def _puzzle_key(puzzle):
    return tuple((section, tuple((_key(p["input"]), _key(p["output"]))
                                 for p in puzzle[section]))
                 for section in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(12345)
    _require(_puzzle_key(automatic) == _puzzle_key(generate_puzzle(12345)),
             "automatic generation is nondeterministic")
    explicit = generate_puzzle(54321, 7, 3)
    _require((len(explicit["train"]), len(explicit["test"])) == (7, 3) and
             _puzzle_key(explicit) == _puzzle_key(generate_puzzle(54321, 7, 3)),
             "explicit counts or determinism failed")
    for i, (tr, te) in enumerate(((2, 1), (10, 1), (10, 4),
                                  (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(60000 + i, tr, te)
        if tr is not None:
            _require(len(puzzle["train"]) == tr, "partial train count ignored")
        if te is not None:
            _require(len(puzzle["test"]) == te, "partial test count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 2.0, "num_test": 1}):
        try:
            generate_puzzle(62000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    train_counts, test_counts, layouts, shapes, palettes = (
        set(), set(), set(), set(), set()
    )
    mixed = full_test = padded_train = one_sided = unequal = long_run = 0
    odd_train = even_test = balanced = long_asymmetric = compact_puzzles = 0
    simple_pairs = both_test_orientations = varied_parity = 0
    first_train_lengths, first_test_lengths = set(), set()
    puzzle_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        convention = set.intersection(*[
            _candidates(p["input"], p["output"]) for p in puzzle["train"]
        ]).pop()
        orientations = set()
        test_orientations = set()
        compact_features = []
        parities = set()
        for section in ("train", "test"):
            for index, pair in enumerate(puzzle[section]):
                a, b = pair["input"], pair["output"]
                matches = _matching_models(a, b, convention)
                _require(len(matches) == 1, "self-test rule interpretation")
                meta = matches[0][1]
                orientations.add(meta["orientation"])
                shapes.add(a.shape)
                palettes.add((meta["background"], meta["companion"],
                              meta["spine"]))
                layouts.add((section, meta["orientation"], a.shape,
                             meta["length"], meta["low"], meta["high"],
                             meta["band"]))
                low, high = meta["low"], meta["high"]
                length = meta["length"]
                parities.add(length % 2)
                compact_features.append((length, meta["width"], low, high))
                if index == 0:
                    (first_train_lengths if section == "train"
                     else first_test_lengths).add(length)
                one_sided += (low == 0) != (high == 0)
                unequal += low != high and low > 0 and high > 0
                balanced += low == high and low > 0
                long_run += length > 7
                long_asymmetric += length > 7 and low != high
                simple_pairs += length in (4, 6) and low == high == 0
                if section == "train":
                    padded_train += low + high > 0
                    odd_train += length % 2 == 1
                else:
                    test_orientations.add(meta["orientation"])
                    full_test += low == high == 0
                    even_test += length % 2 == 0
        mixed += len(orientations) == 2
        both_test_orientations += len(test_orientations) == 2
        varied_parity += len(parities) == 2
        compact_puzzles += all(length <= 9 and width <= 8 and
                               low <= 2 and high <= 2
                               for length, width, low, high in compact_features)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "count diversity too narrow")
    _require(len(puzzle_keys) >= 40 and len(layouts) >= 150 and
             len(shapes) >= 35 and len(palettes) >= 100,
             "structural or palette diversity too narrow")
    _require(mixed >= 10 and full_test >= 5 and padded_train >= 30 and
             one_sided >= 30 and unequal >= 20 and long_run >= 35 and
             odd_train >= 20 and even_test >= 10,
             "reviewed variation regimes underrepresented")
    _require(balanced >= 20 and long_asymmetric >= 25 and
             compact_puzzles >= 2 and simple_pairs >= 5 and
             both_test_orientations >= 5 and varied_parity >= 25 and
             len(first_train_lengths) >= 8 and len(first_test_lengths) >= 8,
             "reviewed combinations or pair-index diversity too narrow")
