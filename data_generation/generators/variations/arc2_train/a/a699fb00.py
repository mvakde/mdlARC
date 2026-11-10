"""Synthetic puzzles for horizontal one-cell gap completion."""

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _signature(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


_SOURCE_INPUTS = (
    ("0000000000", "0101010100", "0000000000", "0000000000", "0010100000", "0000000000", "0000001010", "0000000000", "0001010000", "0000000000"),
    ("10100", "00000", "00000", "01010", "00000"),
    ("0000000000", "0000001010", "0101000000", "0000000000", "0000000000", "0001010100", "0000000000", "0000101000", "0000000000", "0101000000"),
    ("0101000000", "0000000000", "0010101010", "0000000000", "0101000000", "0000010100", "0000000000", "0001010000", "0000000000", "0000000000"),
)
_SOURCE_SIGNATURES = frozenset(
    ((len(rows), len(rows[0])), tuple(int(c) for row in rows for c in row))
    for rows in _SOURCE_INPUTS
)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + " is outside its legal range")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        # More compact puzzles, while retaining the full legal range.
        num_train = int(rng.choice(np.arange(2, 11), p=[.10, .29, .19, .13, .10, .07, .05, .04, .03]))
    if num_train is None:
        legal = np.arange(max(2, num_test + 1), 11)
        weights = 1 / (1 + np.abs(legal - 3))
        num_train = int(rng.choice(legal, p=weights / weights.sum()))
    if num_test is None:
        legal = np.arange(1, min(4, num_train - 1) + 1)
        weights = 1 / legal
        num_test = int(rng.choice(legal, p=weights / weights.sum()))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _palettes(rng, train_count):
    colors = [int(v) for v in rng.permutation(10)]
    connector = colors[0]
    max_palettes = min(3, train_count)
    choices = np.arange(1, max_palettes + 1)
    weights = np.array([.27, .57, .16][:max_palettes], dtype=float)
    count = int(rng.choice(choices, p=weights / weights.sum()))
    palettes = [(colors[1 + 2*i], colors[2 + 2*i]) for i in range(count)]
    return connector, palettes


def _apply_rule(grid, background, marker, connector):
    output = grid.copy()
    mask = ((grid[:, :-2] == marker) & (grid[:, 1:-1] == background)
            & (grid[:, 2:] == marker))
    output[:, 1:-1][mask] = connector
    return output


def _dimension(rng, regime, axis):
    if regime == "source_train":
        weights = [.07, .46, .30, .17]
    else:
        weights = [.06, .33, .35, .26]
    category = int(rng.choice(4, p=weights))
    lo, hi = ((3, 4), (5, 10), (11, 20), (21, 30))[category]
    return int(rng.integers(lo, hi + 1))


def _row_pattern(rng, width, group_count):
    # Each chain has at least two markers; adjacent chains have >=2 empty cells.
    minimum = 5 * group_count - 2
    remaining = width - minimum
    extra_markers = int(rng.integers(0, remaining // 2 + 1))
    lengths = [2] * group_count
    for _ in range(extra_markers):
        lengths[int(rng.integers(group_count))] += 1
    slack = width - (minimum + 2 * extra_markers)
    spaces = [0] * (group_count + 1)
    for _ in range(slack):
        spaces[int(rng.integers(group_count + 1))] += 1
    pattern = np.zeros(width, dtype=bool)
    col = spaces[0]
    for i, length in enumerate(lengths):
        pattern[col:col + 2 * length - 1:2] = True
        col += 2 * length - 1
        if i + 1 < group_count:
            col += 2 + spaces[i + 1]
    return pattern


def _make_pair(rng, regime, background, marker, connector, force_multi=False):
    if not force_multi and rng.random() < .10:
        # Keep a modest share of the source's square canvas sizes.
        height = width = int(rng.choice((5, 10)))
    else:
        height = _dimension(rng, regime, "height")
        width = _dimension(rng, regime, "width")
    if force_multi:
        width = max(width, 8)
    density = float(rng.uniform(.12, .69 if regime == "target_test" else .58))
    row_count = max(1, min(height, int(rng.binomial(height, density))))
    rows = [int(v) for v in rng.choice(height, size=row_count, replace=False)]
    grid = np.full((height, width), background, dtype=np.int8)
    multi_row = int(rng.choice(rows)) if force_multi else -1
    max_groups = min(3, (width + 2) // 5)
    for row in rows:
        groups = 1
        if row == multi_row:
            groups = 2
        elif max_groups > 1 and rng.random() < (.27 if regime == "source_train" else .38):
            groups = int(rng.integers(2, max_groups + 1))
        grid[row, _row_pattern(rng, width, groups)] = marker
    # Isolated markers exercise the difference between a pair and a single dot.
    if row_count < height and rng.random() < .35:
        free_rows = [row for row in range(height) if row not in rows]
        row = int(rng.choice(free_rows))
        grid[row, int(rng.integers(width))] = marker
    output = _apply_rule(grid, background, marker, connector)
    return {"input": grid, "output": output}


def _roles(pair, connector=None):
    inp, out = pair["input"], pair["output"]
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 2 and counts[0] != counts[1], "ambiguous input roles")
    background = int(colors[int(np.argmax(counts))])
    marker = int(colors[int(np.argmin(counts))])
    changed = inp != out
    _require(bool(np.any(changed)), "pair lacks a completed gap")
    _require(bool(np.all(inp[changed] == background)), "non-background changed")
    new_colors = set(int(v) for v in np.unique(out[changed]))
    _require(len(new_colors) == 1, "inconsistent connector")
    new_connector = next(iter(new_colors))
    _require(new_connector not in (background, marker), "connector appears in input")
    if connector is not None:
        _require(new_connector == connector, "connector changed within puzzle")
    return background, marker, new_connector


def _features(pair):
    background, marker, _ = _roles(pair)
    inp = pair["input"]
    lengths = []
    multi_rows = 0
    lone = 0
    for row in inp:
        positions = np.flatnonzero(row == marker)
        groups = []
        current = 1 if len(positions) else 0
        for distance in np.diff(positions):
            if distance == 2:
                current += 1
            else:
                groups.append(current)
                current = 1
        if current:
            groups.append(current)
        lengths.extend(v for v in groups if v >= 2)
        lone += sum(v == 1 for v in groups)
        multi_rows += sum(v >= 2 for v in groups) >= 2
    return lengths, multi_rows, lone


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen = set()
    connector = None
    train_palettes = set()
    all_roles = {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            for key in ("input", "output"):
                grid = pair[key]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid is not a matrix")
                _require(all(1 <= n <= 30 for n in grid.shape), "illegal grid shape")
                _require(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "illegal ARC color")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape, "shape changed")
            signature = _signature(inp)
            _require(signature not in seen, "duplicate input")
            _require(signature not in _SOURCE_SIGNATURES, "source input copied")
            seen.add(signature)
            background, marker, connector = _roles(pair, connector)
            palette = (background, marker)
            if split == "train":
                train_palettes.add(palette)
            else:
                _require(palette in train_palettes, "test palette not demonstrated")
            _require(background not in all_roles or all_roles[background] == "background", "color roles crossed")
            _require(marker not in all_roles or all_roles[marker] == "marker", "color roles crossed")
            all_roles[background] = "background"
            all_roles[marker] = "marker"
            expected = _apply_rule(inp, background, marker, connector)
            _require(bool(np.array_equal(out, expected)), "horizontal completion incorrect")
            lengths, _, _ = _features(pair)
            _require(bool(lengths), "no dotted chain")
            _require(int(np.count_nonzero(inp == background)) > int(np.count_nonzero(inp == marker)), "background not dominant")
    _require(connector not in all_roles, "connector has an input role")
    _require(any(_features(pair)[1] for pair in puzzle["train"]), "training lacks separated same-row chains")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    connector, palettes = _palettes(rng, num_train)
    entropy = rng.integers(0, 2**63, size=num_train + num_test, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_index = len(puzzle["train"]) + len(puzzle["test"])
            pair_rng = np.random.default_rng(np.random.SeedSequence([int(entropy[pair_index]), pair_index]))
            palette_index = index if split == "train" and index < len(palettes) else int(pair_rng.integers(len(palettes)))
            background, marker = palettes[palette_index]
            for attempt in range(100):
                pair = _make_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                  background, marker, connector,
                                  force_multi=(split == "train" and index == 1))
                signature = _signature(pair["input"])
                if (signature not in seen and signature not in _SOURCE_SIGNATURES
                        and np.count_nonzero(pair["input"] == background)
                        > np.count_nonzero(pair["input"] == marker)):
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("unable to construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    sample = np.array([[1, 0, 1, 0, 0, 1, 0, 1],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0]], dtype=np.int8)
    completed = _apply_rule(sample, 0, 1, 2)
    _require(tuple(int(v) for v in completed[0]) == (1, 2, 1, 0, 0, 1, 2, 1),
             "separate same-row chains were joined")
    _require(int(completed[2, 2]) == 0, "vertical-only gap was filled")
    requests = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))
    for i, counts in enumerate(requests):
        puzzle = generate_puzzle(5000 + i, *counts)
        validate_generated_puzzle(puzzle)
        for actual, requested in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            _require(requested is None or actual == requested, "count request ignored")
        _require(_puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(5000 + i, *counts)), "nondeterministic")
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(6000, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    train_counts, test_counts, signatures, shapes, palettes = set(), set(), set(), set(), set()
    multi, lone, long, single, more_than_five, wide, tall, edge = 0, 0, 0, 0, 0, 0, 0, 0
    within_shapes = 0
    within_palettes = 0
    compact = 0
    test_multi = test_long = test_over_five = combined = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        compact += len(puzzle["train"]) == 3 and len(puzzle["test"]) == 1
        pair_shapes = set()
        pair_palettes = set()
        training = puzzle["train"]
        inferred_connector = _roles(training[0])[2]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg, mk, cn = _roles(pair)
                _require(cn == inferred_connector, "unlearnable connector")
                pair_shapes.add(inp.shape)
                pair_palettes.add((bg, mk))
                shapes.add(inp.shape)
                palettes.add((bg, mk, cn))
                lengths, n_multi, n_lone = _features(pair)
                multi += n_multi > 0
                lone += n_lone > 0
                long += max(lengths) >= 5
                single += len(lengths) == 1
                more_than_five += len(lengths) > 5
                h, w = inp.shape
                wide += w >= 2*h
                tall += h >= 2*w
                edge += bool(np.any(inp[0] == mk) or np.any(inp[-1] == mk)
                             or np.any(inp[:, 0] == mk) or np.any(inp[:, -1] == mk))
                if split == "test":
                    test_multi += n_multi > 0
                    test_long += max(lengths) >= 5
                    test_over_five += len(lengths) > 5
                    combined += n_multi > 0 and max(lengths) >= 5 and max(h, w) >= 21
                # Prediction uses only training transformations and the current input.
                predicted = inp.copy()
                for r in range(h):
                    for c in range(1, w - 1):
                        if inp[r, c] == bg and inp[r, c-1] == mk and inp[r, c+1] == mk:
                            predicted[r, c] = inferred_connector
                _require(bool(np.array_equal(predicted, out)), "input-only prediction failed")
        within_shapes += len(pair_shapes) > 1
        within_palettes += len(pair_palettes) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity weak")
    _require(len(signatures) == 42 and len(shapes) >= 35 and len(palettes) >= 30, "puzzle diversity weak")
    _require(multi >= 30 and lone >= 15 and long >= 25 and single >= 8, "chain diversity weak")
    _require(more_than_five >= 15 and wide >= 5 and tall >= 5 and edge >= 20, "geometry diversity weak")
    _require(within_shapes >= 35 and within_palettes >= 12, "within-puzzle diversity weak")
    _require(compact >= 3 and test_multi >= 8 and test_long >= 8
             and test_over_five >= 8 and combined >= 3, "test regimes too narrow")
