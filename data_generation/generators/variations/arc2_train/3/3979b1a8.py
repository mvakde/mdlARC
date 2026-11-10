"""Role-based southeast L growth, reflected to one puzzle-wide quadrant."""

import itertools

import numpy as np


_SIZES = (5, 7, 9, 11, 13, 15)
_SOURCE_ROLES = {(2, 5, 3), (9, 2, 8), (5, 9, 1)}
_ASSIGNMENTS = tuple(itertools.permutations(range(10), 3))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(rng, seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _masks(n, profile):
    patch, reach, half = profile
    corner = np.zeros((n, n), dtype=bool)
    for r in (slice(0, patch), slice(n - patch, n)):
        for c in (slice(0, patch), slice(n - patch, n)):
            corner[r, c] = True
    cross = np.zeros((n, n), dtype=bool)
    mid = n // 2
    cross[mid - half:mid + half + 1, mid - reach:mid + reach + 1] = True
    cross[mid - reach:mid + reach + 1, mid - half:mid + half + 1] = True
    return corner, cross


def _profiles(n):
    result = []
    mid = n // 2
    for patch in range(1, mid + 1):
        for reach in range(1, mid + 1):
            for half in range(min(2, reach - 1) + 1):
                corner, cross = _masks(n, (patch, reach, half))
                if not np.any(corner & cross) and np.any(~(corner | cross)):
                    result.append((patch, reach, half))
    return tuple(result)


_PROFILES = {n: _profiles(n) for n in _SIZES}
_BASE = (1, 1, 0)


def _seed_grid(n, profile, roles):
    corner, cross, filler = roles
    corner_mask, cross_mask = _masks(n, profile)
    grid = np.full((n, n), filler, dtype=np.int8)
    grid[corner_mask] = corner
    grid[cross_mask] = cross
    return grid


def _analyze_seed(grid):
    n = grid.shape[0]
    _require(grid.shape == (n, n) and n in _SIZES, "invalid seed dimensions")
    colors = {int(v) for v in np.unique(grid)}
    _require(len(colors) == 3, "seed must use three distinct roles")
    corner = int(grid[0, 0])
    cross = int(grid[n // 2, n // 2])
    _require(corner != cross, "corner and cross must differ")
    filler = next(iter(colors - {corner, cross}), None)
    _require(filler is not None, "filler role is missing")
    roles = (corner, cross, filler)
    matches = [profile for profile in _PROFILES[n]
               if np.array_equal(grid, _seed_grid(n, profile, roles))]
    _require(len(matches) == 1, "seed has no unique corner/cross motif")
    return roles, matches[0]


def _orient(grid, quadrant):
    # The same involution maps the chosen growth corner to southeast.
    if quadrant in (0, 1):
        grid = np.flipud(grid)
    if quadrant in (0, 2):
        grid = np.fliplr(grid)
    return grid


def _grow(input_grid, roles, quadrant):
    n = input_grid.shape[0]
    seed = _orient(input_grid, quadrant)
    output = np.empty((2 * n, 2 * n), dtype=np.int8)
    output[:n, :n] = seed
    successor = {roles[0]: roles[1], roles[1]: roles[2], roles[2]: roles[0]}
    state = int(seed[n - 1, n - 1])
    for index in range(n, 2 * n):
        output[:index, index] = state
        output[index, :index] = state
        state = successor[state]
        output[index, index] = state
    return _orient(output, quadrant).copy()


def _formula_output(input_grid, roles, quadrant):
    """Cell formula independent of the iterative ground-truth builder."""
    n = input_grid.shape[0]
    seed = _orient(input_grid, quadrant)
    result = np.empty((2 * n, 2 * n), dtype=np.int8)
    for row in range(2 * n):
        for column in range(2 * n):
            if row < n and column < n:
                result[row, column] = seed[row, column]
            elif row == column:
                result[row, column] = roles[(row - n + 1) % 3]
            else:
                result[row, column] = roles[(max(row, column) - n) % 3]
    return _orient(result, quadrant).copy()


def _source_copy(input_grid, output_grid):
    if input_grid.shape != (5, 5) or output_grid.shape != (10, 10):
        return False
    for roles in _SOURCE_ROLES:
        original = _seed_grid(5, _BASE, roles)
        if np.array_equal(input_grid, original) and np.array_equal(
            output_grid, _grow(original, roles, 3)
        ):
            return True
    return False


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _select_profile(rng, n, kind):
    profiles = _PROFILES[n]
    if kind == "base":
        return _BASE
    if kind == "rich":
        eligible = [p for p in profiles if p[0] >= 2 and p[1] >= 2 and p[2] >= 1]
    else:
        eligible = [p for p in profiles if p != _BASE]
    if not eligible:
        eligible = list(profiles)
    return eligible[int(rng.integers(len(eligible)))]


def _choose_roles(rng, candidates, n, profile, quadrant, used):
    _require(bool(candidates), "no role assignments meet palette requirements")
    start = int(rng.integers(len(candidates)))
    for offset in range(len(candidates)):
        roles = candidates[(start + offset) % len(candidates)]
        input_grid = _seed_grid(n, profile, roles)
        if _signature(input_grid) in used:
            continue
        output_grid = _grow(input_grid, roles, quadrant)
        if _source_copy(input_grid, output_grid):
            continue
        used.add(_signature(input_grid))
        return roles, input_grid, output_grid
    raise ValueError("the requested configuration has no unused input")


def _grid_ok(grid, description):
    _require(isinstance(grid, np.ndarray), f"{description} must be an array")
    _require(grid.ndim == 2, f"{description} must have two dimensions")
    _require(all(1 <= v <= 30 for v in grid.shape), f"{description} has invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{description} must be integer")
    _require(np.all((grid >= 0) & (grid <= 9)), f"{description} has invalid colors")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4
             and len(train) > len(test), "invalid pair counts")
    seen_inputs = set()
    seen_pairs = set()
    analyzed = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} must contain input and output")
            input_grid, output_grid = pair["input"], pair["output"]
            _grid_ok(input_grid, label + " input")
            _grid_ok(output_grid, label + " output")
            roles, profile = _analyze_seed(input_grid)
            n = input_grid.shape[0]
            _require(output_grid.shape == (2 * n, 2 * n),
                     f"{label} must double the seed side")
            directions = [q for q in range(4) if np.array_equal(
                output_grid, _formula_output(input_grid, roles, q)
            )]
            _require(len(directions) == 1, f"{label} violates the L-layer cycle")
            _require(not _source_copy(input_grid, output_grid),
                     f"{label} copies a source pair")
            input_key = _signature(input_grid)
            pair_key = (input_key, _signature(output_grid))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     f"{label} duplicates a pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyzed[split].append((roles, profile, n, directions[0]))
    all_directions = {item[3] for split in analyzed.values() for item in split}
    _require(len(all_directions) == 1, "growth direction must be puzzle-wide")
    train_roles = [item[0] for item in analyzed["train"]]
    _require(any(color in right and right.index(color) != position
                 for left_index, left in enumerate(train_roles)
                 for right in train_roles[left_index + 1:]
                 for position, color in enumerate(left)),
             "training must move a literal color to another role")
    train_sizes = {item[2] for item in analyzed["train"]}
    if any(item[2] not in train_sizes for item in analyzed["test"]):
        _require(len(train_sizes) >= 2, "new test sizes require size-varied training")
    if any(item[1] != _BASE for item in analyzed["test"]):
        _require(any(p[0] >= 2 and p[1] >= 2 and p[2] >= 1
                     for _, p, _, _ in analyzed["train"]),
                 "rich test motifs require a rich training demonstration")
    # Every seed is unchanged at the quadrant opposite its growth corner.
    for split in ("train", "test"):
        for pair, (_, _, n, quadrant) in zip(puzzle[split], analyzed[split]):
            oriented = _orient(pair["output"], quadrant)
            _require(np.array_equal(oriented[:n, :n],
                                    _orient(pair["input"], quadrant)),
                     "output must preserve its entire input seed")


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
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    source_like = seed % 11 == 0
    quadrant = 3 if source_like else int(rng.integers(4))
    uniform_size = source_like or bool(rng.integers(4) == 0)
    rich_motifs = not source_like and bool(rng.integers(3) != 0)
    pair_seeds = rng.integers(
        0, np.iinfo(np.int64).max, size=train_count + test_count, dtype=np.int64
    )
    pair_rngs = [np.random.default_rng(int(value)) for value in pair_seeds]
    if uniform_size:
        allowed = (9, 11, 13, 15) if rich_motifs else _SIZES
        size = 5 if source_like else allowed[int(rng.integers(len(allowed)))]
        train_sizes = [size] * train_count
    else:
        first_pool = (7, 9, 11)
        first = first_pool[int(rng.integers(len(first_pool)))]
        second_pool = [n for n in (9, 11, 13) if n != first]
        second = second_pool[int(rng.integers(len(second_pool)))]
        train_sizes = [first, second]
        train_sizes += [int(pair_rngs[i].choice((7, 9, 11, 13)))
                        for i in range(2, train_count)]
    color_bank = tuple(int(v) for v in rng.choice(10, size=6, replace=False))
    train_pool = [a for a in _ASSIGNMENTS if set(a) <= set(color_bank)]
    used = set()
    puzzle = {"train": [], "test": []}
    train_roles = []
    for index, n in enumerate(train_sizes):
        pair_rng = pair_rngs[index]
        if rich_motifs:
            kind = "base" if index == 0 else "rich" if index == 1 else "any"
        else:
            kind = "base"
        profile = _select_profile(pair_rng, n, kind)
        candidates = train_pool
        if index == 1:
            first_roles = train_roles[0]
            candidates = [a for a in train_pool if any(
                color in a and a.index(color) != position
                for position, color in enumerate(first_roles)
            )]
            if bool(pair_rng.integers(2)):
                same_palette = [a for a in candidates if set(a) == set(first_roles)]
                if same_palette:
                    candidates = same_palette
        roles, input_grid, output_grid = _choose_roles(
            pair_rng, candidates, n, profile, quadrant, used
        )
        train_roles.append(roles)
        puzzle["train"].append({"input": input_grid, "output": output_grid})
    seen_train_colors = set().union(*(set(a) for a in train_roles))
    unseen_colors = set(range(10)) - seen_train_colors
    for index in range(test_count):
        pair_rng = pair_rngs[train_count + index]
        if uniform_size:
            n = train_sizes[0]
        else:
            regime = int(pair_rng.integers(3))
            pool = (tuple(sorted(set(train_sizes))) if regime == 0 else
                    tuple(v for v in _SIZES if v < min(train_sizes)) if regime == 1 else
                    tuple(v for v in _SIZES if v > max(train_sizes)))
            if not pool:
                pool = tuple(sorted(set(train_sizes)))
            n = pool[int(pair_rng.integers(len(pool)))]
        profile = _select_profile(pair_rng, n, "any" if rich_motifs else "base")
        novelty = (seed + index) % 4
        novelty = min(novelty, len(unseen_colors))
        candidates = [a for a in _ASSIGNMENTS
                      if len(set(a) & unseen_colors) == novelty]
        if novelty == 0 and bool(pair_rng.integers(2)):
            palettes = {frozenset(a) for a in train_roles}
            reused = [a for a in candidates if frozenset(a) in palettes]
            if reused:
                candidates = reused
        roles, input_grid, output_grid = _choose_roles(
            pair_rng, candidates, n, profile, quadrant, used
        )
        puzzle["test"].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(20260827, 7, 3)
    _require(_puzzle_key(a) == _puzzle_key(generate_puzzle(20260827, 7, 3)),
             "same seed and counts must reproduce the puzzle")
    _require(_puzzle_key(a) != _puzzle_key(generate_puzzle(20260828, 7, 3)),
             "different seeds should change the puzzle")
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(30000 + index, *counts)
        validate_generated_puzzle(puzzle)
        for actual, requested in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            _require(requested is None or actual == requested, "requested count ignored")
    for index, counts in enumerate(((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (True, 1), (4, 1.5))):
        try:
            generate_puzzle(31000 + index, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    sizes, quadrants, profiles, novelty = set(), set(), set(), set()
    train_counts, test_counts, puzzle_keys = set(), set(), set()
    test_size_regimes = set()
    same_palette_training = 0
    simple_pairs = 0
    varied_train_sizes = 0
    varied_train_motifs = 0
    combined_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_analyze_seed(p["input"]) for p in puzzle["train"]]
        train_sizes = {p["input"].shape[0] for p in puzzle["train"]}
        train_colors = set().union(*(set(roles) for roles, _ in train_info))
        if len(train_sizes) > 1:
            varied_train_sizes += 1
        if len({profile for _, profile in train_info}) > 1:
            varied_train_motifs += 1
        if len({frozenset(roles) for roles, _ in train_info}) < len(train_info):
            same_palette_training += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                roles, profile = _analyze_seed(pair["input"])
                n = pair["input"].shape[0]
                sizes.add(n)
                profiles.add((n, profile))
                q = next(q for q in range(4) if np.array_equal(
                    pair["output"], _formula_output(pair["input"], roles, q)
                ))
                quadrants.add(q)
                _require(pair["output"].shape == (2 * n, 2 * n), "wrong output size")
                _require(not _source_copy(pair["input"], pair["output"]),
                         "source pair copied")
                if n == 5 and profile == _BASE and q == 3:
                    simple_pairs += 1
                if profile[0] >= 2 and profile[1] >= 2 and profile[2] >= 1:
                    combined_pairs += 1
                if split == "test":
                    novelty.add(len(set(roles) - train_colors))
                    test_size_regimes.add(
                        "smaller" if n < min(train_sizes) else
                        "larger" if n > max(train_sizes) else "familiar"
                    )
    _require(len(puzzle_keys) >= 40, "too few distinct puzzles")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "counts lack diversity")
    _require(sizes == set(_SIZES) and quadrants == set(range(4)),
             "sizes or growth quadrants lack coverage")
    _require(len(profiles) >= 30 and combined_pairs >= 10,
             "motif shapes or combinations lack diversity")
    _require(novelty == {0, 1, 2, 3}, "test palette novelty lacks coverage")
    _require(test_size_regimes == {"smaller", "larger", "familiar"},
             "test size regimes lack coverage")
    _require(varied_train_sizes >= 20 and varied_train_motifs >= 15,
             "training examples do not vary enough within puzzles")
    _require(same_palette_training >= 5 and simple_pairs >= 2,
             "palette reuse or source-like examples are missing")
