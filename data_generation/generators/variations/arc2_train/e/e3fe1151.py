"""Cross-divided panels with a uniquely recoverable common color multiset."""

import numpy as np


_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4), (5, 1),
                (7, 3), (9, 4))
_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
_PROFILES = ((1, 1, 1, 1), (0, 1, 1, 1), (0, 1, 1, 2),
             (1, 1, 1, 2), (0, 0, 1, 2))
_SOURCE = (
    (("71780", "08771", "77777", "87771", "01785"),
     ("51780", "08751", "77777", "85701", "01785")),
    (("89793", "37778", "77777", "87778", "29792"),
     ("89793", "32728", "77777", "83738", "29792")),
    (("74745", "43773", "77777", "57743", "34757"),
     ("54745", "43743", "77777", "54743", "34754")),
    (("87778", "24749", "77777", "97779", "42728"),
     ("89728", "24749", "77777", "98749", "42728")),
)


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.ravel()))


def _puzzle_key(puzzle):
    return tuple((split, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def _source_copy(inp, out):
    rows = tuple("".join(str(int(x)) for x in row) for row in inp)
    completed = tuple("".join(str(int(x)) for x in row) for row in out)
    return (rows, completed) in _SOURCE


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        choices = tuple(range(test + 1, 11))
        train = choices[seed % len(choices)]
    if test is None:
        choices = tuple(range(1, min(4, train - 1) + 1))
        test = choices[seed % len(choices)]
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _panels(shape):
    rows, cols = shape
    mid_r, mid_c = rows // 2, cols // 2
    return ((slice(0, mid_r), slice(0, mid_c)),
            (slice(0, mid_r), slice(mid_c + 1, cols)),
            (slice(mid_r + 1, rows), slice(0, mid_c)),
            (slice(mid_r + 1, rows), slice(mid_c + 1, cols)))


def _hist(panel):
    return np.bincount(panel.ravel().astype(np.int64), minlength=10)


def _infer_histogram(inp, structural):
    """The pointwise maximum is the sole possible histogram iff it is full."""
    area = ((inp.shape[0] - 1) // 2) * ((inp.shape[1] - 1) // 2)
    observed = []
    for rs, cs in _panels(inp.shape):
        counts = _hist(inp[rs, cs])
        holes = int(counts[structural])
        counts[structural] = 0
        _check(int(counts.sum()) + holes == area, "bad panel area")
        observed.append(counts)
    target = np.maximum.reduce(observed)
    _check(int(target.sum()) == area, "input has no unique histogram")
    for counts in observed:
        deficit = target - counts
        _check(np.all(deficit >= 0), "inconsistent observed panel")
        _check(int(deficit.sum()) <= 2, "too many placeholders")
        _check(np.count_nonzero(deficit) <= 1, "interchangeable holes are ambiguous")
    return target


def _multiplicities(rng, area):
    if area == 4:
        kinds = ((1, 1, 1, 1), (2, 1, 1), (2, 2), (3, 1))
        return list(kinds[int(rng.integers(len(kinds)))])
    colors = int(rng.integers(2, min(area, 6) + 1))
    counts = np.ones(colors, dtype=np.int8)
    for _ in range(area - colors):
        counts[int(rng.integers(colors))] += 1
    return [int(x) for x in counts]


def _make_pair(rng, structural, panel_shape, profile, previous):
    height, width = panel_shape
    shape = (2 * height + 1, 2 * width + 1)
    area = height * width
    profile = list(profile)
    rng.shuffle(profile)
    # Bounded retries handle the simple input-only uniqueness condition.
    for _ in range(400):
        reusable = [hist for hist in previous
                    if sum(hist) == area and hist[structural] == 0]
        if reusable and rng.random() < 0.25:
            common = np.array(reusable[int(rng.integers(len(reusable)))], dtype=np.int8)
        else:
            multiplicities = _multiplicities(rng, area)
            colors = rng.choice([c for c in range(10) if c != structural],
                                size=len(multiplicities), replace=False)
            common = np.zeros(10, dtype=np.int8)
            for color, count in zip(colors, multiplicities):
                common[int(color)] = count
        support = np.flatnonzero(common)
        if 2 in profile and not np.any(common >= 2):
            continue
        missing = []
        for holes in profile:
            choices = np.flatnonzero(common >= holes) if holes == 2 else support
            missing.append(int(rng.choice(choices)) if holes else None)
        # Give the source-like one-per-panel case a useful share of all-four-
        # different examples in both splits.
        if profile == [1, 1, 1, 1] and len(support) == 4 and rng.random() < 0.5:
            missing = [int(x) for x in rng.permutation(support)]
        inp = np.full(shape, structural, dtype=np.int8)
        out = inp.copy()
        patterns = []
        positions = []
        for index, (rs, cs) in enumerate(_panels(shape)):
            values = np.repeat(np.arange(10, dtype=np.int8), common)
            rng.shuffle(values)
            panel = values.reshape(height, width)
            damaged = panel.copy()
            holes = profile[index]
            if holes:
                candidates = np.flatnonzero(panel.ravel() == missing[index])
                chosen = rng.choice(candidates, size=holes, replace=False)
                damaged.ravel()[chosen] = structural
                positions.append(tuple(int(x) for x in sorted(chosen)))
            else:
                positions.append(())
            inp[rs, cs] = damaged
            out[rs, cs] = panel
            patterns.append(tuple(int(x) for x in panel.ravel()))
        if len(set(patterns)) < 2 or len(set(positions)) < 2:
            continue
        try:
            if np.array_equal(_infer_histogram(inp, structural), common):
                return inp, out
        except AssertionError:
            pass
    raise RuntimeError("unable to construct a uniquely repairable pair")


def _validate_grid(grid, label):
    _check(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _check(grid.ndim == 2, f"{label} must be 2D")
    _check(all(1 <= n <= 30 for n in grid.shape), f"{label} bad dimensions")
    _check(np.issubdtype(grid.dtype, np.integer), f"{label} must be integer")
    _check(int(grid.min()) >= 0 and int(grid.max()) <= 9, f"{label} bad color")


def _details(inp, out):
    _check(inp.shape == out.shape, "shape changed")
    rows, cols = inp.shape
    _check((rows, cols) in tuple((2*h+1, 2*w+1) for h, w in _SHAPES),
           "unsupported cross geometry")
    center_r, center_c = rows // 2, cols // 2
    structural = int(inp[center_r, center_c])
    for grid in (inp, out):
        _check(np.all(grid[center_r, :] == structural), "middle row changed")
        _check(np.all(grid[:, center_c] == structural), "middle column changed")
    _check(not np.array_equal(inp, out), "no repair")
    histogram = _infer_histogram(inp, structural)
    profile = []
    replacements = []
    positions = []
    patterns = []
    for rs, cs in _panels(inp.shape):
        ip, op = inp[rs, cs], out[rs, cs]
        mask = ip == structural
        holes = int(np.count_nonzero(mask))
        _check(holes <= 2, "too many placeholders in panel")
        _check(not np.any(op == structural), "structural color in completed panel")
        _check(np.array_equal(ip[~mask], op[~mask]), "visible payload changed")
        _check(np.all(ip[mask] != op[mask]), "placeholder was not repaired")
        _check(np.array_equal(_hist(op), histogram), "unequal completed histograms")
        deficit = histogram - _hist(ip)
        deficit[structural] = 0
        _check(int(deficit.sum()) == holes and np.count_nonzero(deficit) <= 1,
               "placeholder values are ambiguous")
        profile.append(holes)
        replacements.extend(int(x) for x in op[mask])
        positions.append(tuple(int(x) for x in np.flatnonzero(mask)))
        patterns.append(tuple(int(x) for x in op.ravel()))
    _check(tuple(sorted(profile)) in _PROFILES, "unsupported repair profile")
    _check(len(set(patterns)) >= 2, "panels have identical arrangements")
    _check(len(set(positions)) >= 2, "placeholder positions do not vary")
    return (structural, tuple(int(x) for x in histogram),
            tuple(sorted(profile)), tuple(replacements), tuple(positions),
            inp.shape)


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle needs train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "splits must be lists")
    train, test = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= train <= 10 and 1 <= test <= 4 and train > test,
           "illegal pair counts")
    inputs, pairs = set(), set()
    details = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair needs input and output")
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp, f"{split} {index} input")
            _validate_grid(out, f"{split} {index} output")
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            _check(key not in inputs and pair_key not in pairs, "duplicate pair")
            _check(not _source_copy(inp, out), "source pair copied")
            inputs.add(key)
            pairs.add(pair_key)
            details[split].append(_details(inp, out))
    training = details["train"]
    _check(any(d[5] == (5, 5) and d[2] == (1, 1, 1, 1) for d in training),
           "training must demonstrate source geometry and repair count")
    _check(any(d[5] != (5, 5) and 0 in d[2] and 2 in d[2] for d in training),
           "training must demonstrate enlarged panels and variable repair counts")
    # Structural color is locally visible. Completed histograms, dimensions,
    # and placeholder positions are pair-level features, not hidden conventions.


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    fixed_color = bool(rng.integers(2))
    first_color = int(rng.integers(10))
    colors = [first_color]
    if not fixed_color:
        colors.append(int(rng.choice([c for c in range(10) if c != first_color])))
    puzzle = {"train": [], "test": []}
    used = set()
    histograms = []
    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        if index == 0:
            panel_shape, profile = (2, 2), (1, 1, 1, 1)
        elif index == 1:
            panel_shape = _SHAPES[1 + int(rng.integers(3))]
            profile = (0, 1, 1, 2)
        else:
            panel_shape = _SHAPES[int(rng.integers(4))]
            profile = _PROFILES[int(rng.integers(len(_PROFILES)))]
        if fixed_color:
            structural = first_color
        elif index < len(colors):
            structural = colors[index]
        else:
            structural = int(rng.integers(10))
        child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        for _ in range(100):
            inp, out = _make_pair(child, structural, panel_shape, profile, histograms)
            key = _grid_key(inp)
            if key not in used and not _source_copy(inp, out):
                break
        else:
            raise RuntimeError("unable to make a distinct pair")
        used.add(key)
        histograms.append(_details(inp, out)[1])
        puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    shapes, profiles, kinds = set(), set(), set()
    structural, layouts, train_diversity, test_diversity = set(), set(), set(), set()
    train_counts, test_counts, puzzle_keys = set(), set(), set()
    fixed_seeds, changing_seeds, repeated_histograms = 0, 0, 0
    same_regime_seeds, enlarged_two_color_repair = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _check(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)),
               "generation is nondeterministic")
        puzzle_keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_colors, local_histograms = set(), []
        local_regimes = {"train": set(), "test": set()}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                details = _details(inp, out)
                local_colors.add(details[0])
                local_histograms.append((details[5], details[1]))
                local_regimes[split].add((
                    details[5], details[2],
                    tuple(sorted(x for x in details[1] if x)),
                    len(set(details[3])),
                ))
                shapes.add(details[5])
                profiles.add(details[2])
                structural.add(details[0])
                kinds.add(tuple(sorted(x for x in details[1] if x)))
                layouts.add((details[5], details[4],
                             tuple(int(x) for x in out.ravel())))
                diversity = len(set(details[3]))
                (train_diversity if split == "train" else test_diversity).add(diversity)
                enlarged_two_color_repair += (
                    details[5] != (5, 5) and 2 in details[2]
                    and len([x for x in details[1] if x]) == 2
                )
                target = _infer_histogram(inp, details[0])
                for rs, cs in _panels(inp.shape):
                    ip, op = inp[rs, cs], out[rs, cs]
                    deficit = target - _hist(ip)
                    deficit[details[0]] = 0
                    expected = int(np.flatnonzero(deficit)[0]) if np.any(deficit) else None
                    if expected is not None:
                        _check(np.all(op[ip == details[0]] == expected),
                               "input-only repair disagrees with output")
        fixed_seeds += len(local_colors) == 1
        changing_seeds += len(local_colors) > 1
        repeated_histograms += len(local_histograms) != len(set(local_histograms))
        same_regime_seeds += bool(local_regimes["train"] & local_regimes["test"])
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _check(len(puzzle_keys) == 42, "cross-seed duplicate puzzle")
    _check(shapes == {(5, 5), (5, 7), (7, 5), (7, 7)}, "size coverage")
    _check(profiles == set(_PROFILES), "repair-count coverage")
    _check({(1, 1, 1, 1), (1, 1, 2), (2, 2), (1, 3)} <= kinds,
           "four-cell multiset coverage")
    _check(4 in train_diversity and 2 in test_diversity, "split regime coverage")
    _check(len(structural) == 10 and len(layouts) > 150, "structural diversity")
    _check(fixed_seeds > 0 and changing_seeds > 0 and repeated_histograms > 0,
           "color or histogram reuse diversity")
    _check(same_regime_seeds > 0 and enlarged_two_color_repair > 0,
           "compatible train/test regimes or combined variation missing")
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1),
                              (1003, 10, 4), (1004, 6, 3)):
        puzzle = generate_puzzle(seed, train, test)
        _check((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
               "explicit counts ignored")
    for seed, train, test in ((3001, None, 1), (3002, None, 4),
                              (3003, 2, None), (3004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        _check(train is None or len(puzzle["train"]) == train, "partial train count")
        _check(test is None or len(puzzle["test"]) == test, "partial test count")
    for request in ({"num_train": 1}, {"num_train": 11},
                    {"num_test": 0}, {"num_test": 5},
                    {"num_train": 2, "num_test": 2},
                    {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(4001, **request)
        except ValueError:
            continue
        raise AssertionError(f"illegal request accepted: {request}")
