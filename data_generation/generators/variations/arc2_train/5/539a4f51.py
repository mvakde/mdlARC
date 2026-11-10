import numpy as np


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_from_layers(layers):
    a = np.asarray(layers, dtype=np.int64)
    indices = np.maximum(np.arange(len(a))[:, None], np.arange(len(a))[None, :])
    return a[indices]


def _layer_sequence(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert grid.shape[0] == grid.shape[1]
    layers = tuple(int(grid[k, k]) for k in range(grid.shape[0]))
    assert np.array_equal(grid, _grid_from_layers(layers))
    return layers


def _signature(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _pair_signature(pair):
    return (_signature(pair["input"]), _signature(pair["output"]))


def _puzzle_signature(puzzle):
    return tuple(tuple(_pair_signature(p) for p in puzzle[part]) for part in ("train", "test"))


def _has_adjacent_repeat(cycle):
    return any(a == b for a, b in zip(cycle, cycle[1:]))


def _longest_run(cycle):
    longest = current = 1
    for a, b in zip(cycle, cycle[1:]):
        current = current + 1 if a == b else 1
        longest = max(longest, current)
    return longest


def _primitive(cycle):
    return all(any(cycle[i] != cycle[i % q] for i in range(len(cycle))) for q in range(1, len(cycle)))


def _source_specs():
    return (((1, 1, 4, 6, 0), 4), ((2, 2, 2, 3, 0), 4),
            ((2, 3, 4, 1, 6), 5), ((7, 7, 3, 2, 2), 5))


def _matches_source_pair(pair):
    sig = _pair_signature(pair)
    for layers, period in _source_specs():
        reference = {"input": _grid_from_layers(layers),
                     "output": _grid_from_layers(tuple(layers[i % period] for i in range(10)))}
        if sig == _pair_signature(reference):
            return True
    return False


def _cycle_for_background(layers, background):
    if background not in layers:
        return layers
    k = layers.index(background)
    if k == 0 or any(c != background for c in layers[k:]):
        return None
    return layers[:k]


def _background_candidates(train):
    candidates = []
    for background in range(10):
        details = []
        for pair in train:
            layers = _layer_sequence(pair["input"])
            cycle = _cycle_for_background(layers, background)
            if cycle is None:
                break
            expected = tuple(cycle[i % len(cycle)] for i in range(2 * len(layers)))
            if _layer_sequence(pair["output"]) != expected:
                break
            details.append((len(layers), len(cycle), cycle))
        else:
            candidates.append((background, details))
    return candidates


def _choose_size(rng, mode, palette_size, suffix=None):
    if suffix is None:
        suffix = int(rng.choice((0, 0, 0, 0, 1, 1, 2, 2, 3, 4)))
    low = 4 if mode in ("separated", "long") else 3
    if mode == "unique":
        high = min(9, palette_size + suffix)
    elif mode == "separated":
        high = min(15, palette_size + 1 + suffix)
    else:
        high = 15
    if high < low + suffix:
        return None
    if low + suffix <= 5 <= high and rng.random() < 0.2:
        n = 5
    else:
        n = int(rng.integers(low + suffix, high + 1))
    return n, suffix


def _make_cycle(rng, palette, length, mode, forced=None):
    palette = tuple(dict.fromkeys(int(c) for c in palette))
    if forced is not None and forced not in palette:
        return None
    if mode == "unique":
        if length > len(palette):
            return None
        selected = list(rng.choice(palette, size=length, replace=False))
        if forced is not None and forced not in selected:
            selected[-1] = forced
        rng.shuffle(selected)
        return tuple(int(c) for c in selected)
    if mode == "separated":
        if length < 4 or length - 1 > len(palette):
            return None
        chosen = list(rng.choice(palette, size=length - 1, replace=False))
        if forced is not None and forced not in chosen:
            chosen[-1] = forced
        for _ in range(80):
            repeat = int(rng.choice(chosen))
            cycle = [int(c) for c in chosen] + [repeat]
            rng.shuffle(cycle)
            if not _has_adjacent_repeat(cycle) and _primitive(cycle):
                return tuple(cycle)
        return None
    if len(palette) < 2 or length < (4 if mode == "long" else 3):
        return None
    for _ in range(100):
        max_distinct = min(len(palette), length - (3 if mode == "long" else 1))
        if max_distinct < 2:
            return None
        count = int(rng.integers(2, max_distinct + 1))
        chosen = list(rng.choice(palette, size=count, replace=False))
        if forced is not None and forced not in chosen:
            chosen[-1] = forced
        run_color = int(rng.choice(chosen))
        run = 3 if mode == "long" else 2
        other = [int(c) for c in chosen if c != run_color]
        remaining = length - run - len(other)
        if remaining < 0:
            continue
        other += [int(rng.choice(chosen)) for _ in range(remaining)]
        rng.shuffle(other)
        start = int(rng.integers(0, len(other) + 1))
        cycle = tuple(other[:start] + [run_color] * run + other[start:])
        if (len(set(cycle)) == count and _primitive(cycle)
                and (mode != "double" or _longest_run(cycle) == 2)):
            return cycle
    return None


def _build_pair(cycle, suffix, background):
    layers = cycle + (background,) * suffix
    n = len(layers)
    return {"input": _grid_from_layers(layers),
            "output": _grid_from_layers(tuple(cycle[i % len(cycle)] for i in range(2 * n)))}


def _make_pair(rng, background, palette, mode, suffix=None, forced=None):
    for _ in range(80):
        size = _choose_size(rng, mode, len(palette), suffix)
        if size is None:
            break
        n, depth = size
        cycle = _make_cycle(rng, palette, n - depth, mode, forced)
        if cycle is not None:
            return _build_pair(cycle, depth, background)
    raise ValueError(f"unable to construct pair: {mode}, palette={palette}, suffix={suffix}")


def _add_unique(rng, seen, background, palette, mode, suffix=None, forced=None):
    for _ in range(100):
        pair = _make_pair(rng, background, palette, mode, suffix, forced)
        sig = _signature(pair["input"])
        if sig not in seen and not _matches_source_pair(pair):
            seen.add(sig)
            return pair
    raise ValueError("unable to construct unique pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = tuple(int(c) for c in rng.permutation(10))
    background = colors[0]
    train_palette = colors[1:1 + int(rng.integers(4, 9))]
    seen = set()
    train, test = [], []

    # The two core demonstrations establish the neutral outer suffix and
    # separate equal-colored positions from the cycle length.
    if rng.random() < 0.5:
        full_mode, terminated_mode = "unique", str(rng.choice(("double", "long")))
    else:
        full_mode, terminated_mode = str(rng.choice(("double", "separated", "long"))), "unique"
    test_specs = []
    for _ in range(test_count):
        mode = str(rng.choice(("unique", "double", "double", "separated", "long")))
        suffix = int(rng.choice((0, 0, 0, 1, 1, 2, 2, 3, 4)))
        test_specs.append((mode, suffix))
    longest_test_suffix = max(depth for _, depth in test_specs)
    train_suffix = max(int(rng.choice((1, 1, 1, 2, 2, 3, 4))),
                       min(4, longest_test_suffix))
    train.append(_add_unique(rng, seen, background, train_palette, full_mode, suffix=0))
    train.append(_add_unique(rng, seen, background, train_palette, terminated_mode,
                             suffix=train_suffix))
    for _ in range(train_count - 2):
        mode = str(rng.choice(("unique", "double", "separated", "long")))
        train.append(_add_unique(rng, seen, background, train_palette, mode))
    rng.shuffle(train)

    familiar = tuple(sorted({c for pair in train for c in _layer_sequence(pair["output"])}))
    novel = tuple(c for c in range(10) if c != background and c not in familiar)
    for mode, suffix in test_specs:
        new_count = int(rng.choice((0, 0, 0, 1, 1, 2)))
        if mode == "unique":
            new_count = min(new_count, 2)
        new_count = min(new_count, len(novel))
        new_colors = tuple(int(c) for c in rng.choice(novel, size=new_count, replace=False))
        minimum_familiar = 3 if mode in ("separated", "unique") and not new_colors else 2
        familiar_count = int(rng.integers(minimum_familiar, min(len(familiar), 6) + 1))
        known = tuple(int(c) for c in rng.choice(familiar, size=familiar_count, replace=False))
        palette = known + new_colors
        forced = new_colors[0] if new_colors else None
        test.append(_add_unique(rng, seen, background, palette, mode, suffix, forced))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            _layer_sequence(grid)
        n = pair["input"].shape[0]
        assert pair["output"].shape == (2 * n, 2 * n)
        sig = _signature(pair["input"])
        assert sig not in seen
        seen.add(sig)
        assert not _matches_source_pair(pair)
    candidates = _background_candidates(train)
    assert len(candidates) == 1
    background, train_details = candidates[0]
    assert any(p < n for n, p, _ in train_details)
    assert any(p == n for n, p, _ in train_details)
    assert any(len(set(cycle)) < p for _, p, cycle in train_details)
    max_train_suffix = max(n - p for n, p, _ in train_details)
    for n, p, cycle in train_details:
        assert p >= 3 and _primitive(cycle) and background not in cycle
    for pair in test:
        layers = _layer_sequence(pair["input"])
        cycle = _cycle_for_background(layers, background)
        assert cycle is not None and len(cycle) >= 3 and _primitive(cycle)
        depth = len(layers) - len(cycle)
        assert depth <= 1 or max_train_suffix >= depth
        expected = tuple(cycle[i % len(cycle)] for i in range(2 * len(layers)))
        assert _layer_sequence(pair["output"]) == expected


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (6, 2), (8, 3))):
        p = generate_puzzle(1000 + i, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for i, arguments in enumerate(({"num_train": None, "num_test": 1},
                                   {"num_train": None, "num_test": 4},
                                   {"num_train": 2, "num_test": None},
                                   {"num_train": 10, "num_test": None})):
        generate_puzzle(2000 + i, **arguments)
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(3000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    assert _puzzle_signature(generate_puzzle(987654321, 7, 3)) == _puzzle_signature(generate_puzzle(987654321, 7, 3))
    backgrounds = set()
    sizes = set()
    modes = set()
    suffixes = set()
    palettes = set()
    combinations = set()
    novelty = set()
    novel_occurrences = set()
    different_novel_puzzles = 0
    familiar_repeated_tests = 0
    varying_sizes = varying_depths = varying_modes = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        background = _background_candidates(puzzle["train"])[0][0]
        backgrounds.add(background)
        known = {c for pair in puzzle["train"] for c in _layer_sequence(pair["output"])}
        palettes.add((background, tuple(sorted(known))))
        local_sizes, local_depths, local_modes = set(), set(), set()
        test_new_sets = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                layers = _layer_sequence(pair["input"])
                cycle = _cycle_for_background(layers, background)
                n = len(layers)
                depth = n - len(cycle)
                assert _layer_sequence(pair["output"]) == tuple(cycle[i % len(cycle)] for i in range(2 * n))
                mode = "unique" if len(set(cycle)) == len(cycle) else ("separated" if not _has_adjacent_repeat(cycle) else ("long" if _longest_run(cycle) >= 3 else "double"))
                sizes.add(n)
                modes.add((split, mode))
                suffixes.add((split, depth))
                combinations.add((split, mode, depth))
                local_sizes.add(n)
                local_depths.add(depth)
                local_modes.add(mode)
                if split == "test":
                    new = set(cycle) - known
                    assert len(new) <= 2
                    novelty.add(len(new))
                    novel_occurrences.add(sum(color in new for color in cycle))
                    test_new_sets.append(new)
                    familiar_repeated_tests += any(
                        cycle[i] == cycle[i + 1] and cycle[i] in known
                        for i in range(len(cycle) - 1)
                    )
        different_novel_puzzles += len({
            next(iter(colors)) for colors in test_new_sets if len(colors) == 1
        }) >= 2
        varying_sizes += len(local_sizes) > 1
        varying_depths += len(local_depths) > 1
        varying_modes += len(local_modes) > 1
    assert len(backgrounds) >= 5 and len(palettes) >= 28
    assert len(sizes) >= 10 and {13, 14, 15} & sizes
    assert len(modes) == 8 and ("train", 2) in suffixes and ("test", 2) in suffixes
    assert len(combinations) >= 20
    assert ("train", "unique", 2) in combinations
    assert ("train", "double", 0) in combinations
    assert ("train", "separated", 0) in combinations
    assert ("test", "separated", 0) in combinations
    assert novelty == {0, 1, 2} and 1 in novel_occurrences
    assert any(n >= 2 for n in novel_occurrences)
    assert different_novel_puzzles >= 3 and familiar_repeated_tests >= 10
    assert varying_sizes == varying_depths == varying_modes == 42
