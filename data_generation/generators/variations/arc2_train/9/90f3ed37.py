"""ARC column-profile completion generator."""

import numpy as np


_ORIGINAL_INPUTS = {
    (15, 10, tuple(int(v) for row in block.split() for v in row))
    for block in (
        """
        0000000000 0000000000 8888888888 0000000000 0000000000
        0000000000 8880000000 0000000000 0000000000 0000000000
        0000000000 8888000000 0000000000 8800000000 0000000000
        """,
        """
        0000000000 8800000000 0080000000 0008888888 0000000000
        0000000000 8880000000 0008000000 0000000000 0000000000
        8800000000 0080000000 0000000000 0000000000 0000000000
        """,
        """
        0000000000 8080808080 0808080808 0000000000 0000000000
        0000000000 8080000000 0800000000 0000000000 0000000000
        8080800000 0808000000 0000000000 0000000000 0000000000
        """,
        """
        0000000000 0088888888 8800000000 0088888888 0000000000
        0000000000 0000000000 0000000000 0008800000 8880000000
        0008800000 0000000000 0000000000 0000000000 0000000000
        """,
    )
}


def _key(grid):
    return (grid.shape[0], grid.shape[1], tuple(int(v) for v in grid.flat))


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} out of range")
    return int(value)


def _counts(seed, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _orient(grid, direction):
    if direction == "right":
        return grid
    if direction == "left":
        return np.fliplr(grid)
    if direction == "down":
        return grid.T
    return np.flipud(grid.T)


def _unorient(grid, direction):
    if direction == "right":
        return grid
    if direction == "left":
        return np.fliplr(grid)
    if direction == "down":
        return grid.T
    return np.flipud(grid).T


def _expanded(states, runs):
    return tuple(state for state, run in zip(states, runs) for _ in range(run))


def _profile(rng, kind, width, force_early=False):
    """Return ordered slices, finite states/runs, and the first turn column."""
    thick = int(rng.integers(1, 3)) if kind != "branch" else 1
    if kind == "straight":
        states = [tuple(range(thick))]
        runs = [width]
    elif kind in ("zigzag", "wave3"):
        cycle = [0, 1] if kind == "zigzag" else [0, 1, 2, 1]
        if rng.integers(2):
            cycle = [max(cycle) - v for v in cycle]
        phase = int(rng.integers(0, len(cycle)))
        slices = tuple(
            tuple(range(cycle[(col + phase) % len(cycle)],
                        cycle[(col + phase) % len(cycle)] + thick))
            for col in range(width)
        )
        return slices, None, None, None
    elif kind == "step":
        turns = int(rng.integers(2, 4))
        descending = bool(rng.integers(2))
        levels = list(range(turns + 1))
        if not descending:
            levels.reverse()
        states = [tuple(range(level, level + thick)) for level in levels]
        middle = [int(rng.integers(1, 4)) for _ in range(turns - 1)]
        largest_first = min(8, width - sum(middle) - 3)
        if largest_first < (3 if force_early else 2):
            return _profile(rng, "straight", width)
        first = int(rng.integers(3 if force_early else 2, largest_first + 1))
        runs = [first, *middle, width - first - sum(middle)]
    elif kind == "branch":
        moving_arms = bool(rng.integers(2)) and width >= 10
        states = [(2,), (1, 3), (0, 4)] if moving_arms else [(1,), (0, 2)]
        middle = [int(rng.integers(2, 5))] if moving_arms else []
        largest_first = min(8, width - sum(middle) - 3)
        first = int(rng.integers(3 if force_early else 2, largest_first + 1))
        runs = [first, *middle, width - first - sum(middle)]
    else:
        raise ValueError(kind)
    return _expanded(states, runs), states, runs, runs[0] if len(runs) > 1 else None


def _layout(rng, n, band, minimum_height):
    required = n * band + 2 * (n - 1)
    if required > 30:
        raise ValueError("too many components for profile height")
    height = int(rng.integers(max(required, minimum_height), 31))
    spaces = rng.multinomial(height - required, [1 / (n + 1)] * (n + 1))
    origins = []
    row = int(spaces[0])
    for i in range(n):
        origins.append(row)
        row += band + (2 + int(spaces[i + 1]) if i < n - 1 else 0)
    return height, origins


def _make_pair(rng, convention, kind, split, preturn=False):
    direction, timing, completion, fixed_palette = convention
    width = int(rng.integers(12 if kind == "step" else 9 if kind == "branch" else 8, 31))
    profile, states, runs, turn = _profile(rng, kind, width, preturn)
    band = 1 + max(row for sl in profile for row in sl)
    max_partials = min(4, (32 // (band + 2)) - 1)
    if split == "original_train":
        count = int(rng.integers(2, min(3, max_partials) + 1))
    elif split == "original_test":
        count = 1
    else:
        count = int(rng.integers(1, max_partials + 1))
    maximum_slice = max(len(sl) for sl in profile)
    height, origins = _layout(rng, count + 1, band,
                              2 * (count + 1) * maximum_slice + 1)
    if fixed_palette is None:
        colors = [v for v in range(10) if v != completion]
        background, source = (int(v) for v in rng.choice(colors, size=2, replace=False))
    else:
        background, source = fixed_palette
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    exemplar_index = int(rng.integers(count + 1))
    for i, origin in enumerate(origins):
        if i == exemplar_index:
            local = profile
            end = width - 1
        else:
            local = profile
            if timing == "free" and states is not None and len(states) > 1:
                middle = runs[1:-1]
                largest_first = min(9, width - sum(middle) - 3)
                first = int(rng.integers(2, largest_first + 1))
                local = _expanded(states, [first, *middle, width - first - sum(middle)])
                first_terminal = first + sum(middle)
                end = int(rng.integers(first_terminal, width - 1))
            elif preturn and i != exemplar_index and turn is not None and timing == "aligned":
                end = int(rng.integers(1, turn))
            else:
                minimum = 3 if kind == "wave3" else 2
                end = int(rng.integers(minimum, width - 1))
        for col, rows in enumerate(local):
            for row in rows:
                if col <= end:
                    inp[origin + row, col] = source
                    out[origin + row, col] = source
                else:
                    out[origin + row, col] = completion
    return {"input": _orient(inp, direction).copy(),
            "output": _orient(out, direction).copy()}


def _components(mask):
    remaining = {tuple(int(v) for v in point) for point in np.argwhere(mask)}
    result = []
    while remaining:
        stack = [remaining.pop()]
        component = set(stack)
        while stack:
            row, col = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = (row + dr, col + dc)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
        result.append(component)
    return result


def _slices(component):
    columns = sorted({c for _, c in component})
    assert columns == list(range(columns[-1] + 1))
    return tuple(tuple(sorted(r for r, c in component if c == col)) for col in columns)


def _relative(slices):
    first = min(slices[0])
    return tuple(tuple(r - first for r in sl) for sl in slices)


def _states(slices):
    result = []
    for state in slices:
        if not result or result[-1] != state:
            result.append(state)
    return tuple(result)


def _inspect_pair(pair, direction, timing, completion):
    inp = _unorient(pair["input"], direction)
    out = _unorient(pair["output"], direction)
    assert inp.shape == out.shape and 1 <= min(inp.shape) and max(inp.shape) <= 30
    assert inp.dtype == out.dtype == np.int8
    assert set(int(v) for v in np.unique(inp)) <= set(range(10))
    assert completion not in inp
    assert _key(pair["input"]) not in _ORIGINAL_INPUTS
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 2
    background = int(colors[int(np.argmax(counts))])
    source = int(colors[int(np.argmin(counts))])
    assert background != source and completion not in (background, source)
    changed = inp != out
    assert np.any(changed)
    assert np.all(inp[changed] == background)
    assert np.all(out[changed] == completion)
    assert np.all(out[~changed] == inp[~changed])
    source_components = _components(inp == source)
    finished_components = _components(out != background)
    assert len(source_components) == len(finished_components) >= 2
    exemplar = [c for c in source_components if max(col for _, col in c) == inp.shape[1] - 1]
    assert len(exemplar) == 1
    exemplar = exemplar[0]
    exemplars = _relative(_slices(exemplar))
    assert len(exemplars) == inp.shape[1]
    partials = 0
    first_turn = next((col for col in range(1, len(exemplars))
                       if exemplars[col] != exemplars[col - 1]), None)
    early = False
    matched = set()
    for component in source_components:
        matches = [j for j, finished in enumerate(finished_components)
                   if component <= finished]
        assert len(matches) == 1 and matches[0] not in matched
        j = matches[0]
        matched.add(j)
        finished = finished_components[j]
        assert len(_slices(finished)) == inp.shape[1]
        if component is exemplar:
            assert component == finished
            continue
        partials += 1
        seen = _relative(_slices(component))
        whole = _relative(_slices(finished))
        assert len(seen) < inp.shape[1] and whole[:len(seen)] == seen
        if (first_turn is not None and len(seen) <= first_turn
                and seen == exemplars[:len(seen)]):
            early = True
        if timing == "aligned" or whole == exemplars:
            assert whole == exemplars
        else:
            assert _states(whole) == _states(exemplars)
            assert seen[-1] == whole[-1]
            assert whole[len(seen):] == (whole[-1],) * (inp.shape[1] - len(seen))
    assert len(matched) == len(finished_components)
    return {"shape": inp.shape, "partials": partials,
            "roles": (background, source, completion),
            "states": _states(exemplars), "exemplar": exemplars,
            "early": early}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    original_like = seed % 6 == 0
    direction = "right" if original_like else ("right", "left", "down", "up")[int(rng.integers(4))]
    timing = "free" if original_like else ("aligned", "free")[int(rng.integers(2))]
    completion = int(rng.integers(10))
    if original_like or rng.random() < 0.45:
        colors = [v for v in range(10) if v != completion]
        fixed_palette = tuple(int(v) for v in rng.choice(colors, size=2, replace=False))
    else:
        fixed_palette = None
    convention = (direction, timing, completion, fixed_palette)
    kinds = ("straight", "step", "zigzag", "wave3", "branch")
    if original_like:
        train_kinds = ["step", "zigzag"] + ["straight" if i == 2 else kinds[int(rng.integers(3))]
                                                 for i in range(2, train_count)]
        test_kinds = ["branch"] * test_count
        anchor = None
    else:
        anchor = ("step", "branch")[int(rng.integers(2))] if timing == "aligned" else None
        train_kinds = [anchor if anchor else kinds[int(rng.integers(5))]]
        train_kinds += [kinds[int(rng.integers(5))] for _ in range(train_count - 1)]
        test_kinds = [anchor if anchor else kinds[int(rng.integers(5))]]
        test_kinds += [kinds[int(rng.integers(5))] for _ in range(test_count - 1)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, chosen in (("train", train_kinds), ("test", test_kinds)):
        regime = ("original_" + split) if original_like else split
        for index, kind in enumerate(chosen):
            for _ in range(100):
                pair = _make_pair(rng, convention, kind, regime,
                                  preturn=(anchor is not None and index == 0))
                key = _key(pair["input"])
                if key not in seen and key not in _ORIGINAL_INPUTS:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make a unique pair")
    _validate(puzzle, convention)
    return puzzle


def _validate(puzzle, convention=None):
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    if convention is None:
        possible = []
        for direction in ("right", "left", "down", "up"):
            for timing in ("aligned", "free"):
                try:
                    first = puzzle["train"][0]
                    output_only = set(int(v) for v in np.unique(first["output"])) - set(int(v) for v in np.unique(first["input"]))
                    if len(output_only) != 1:
                        continue
                    completion = output_only.pop()
                    [_inspect_pair(pair, direction, timing, completion)
                     for split in ("train", "test") for pair in puzzle[split]]
                    possible.append((direction, timing, completion, None))
                except AssertionError:
                    pass
        assert possible
        convention = possible[0]
    direction, timing, completion, _ = convention
    keys = set()
    infos = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _key(pair["input"])
            assert key not in keys
            keys.add(key)
            infos.append(_inspect_pair(pair, direction, timing, completion))
    n_train = len(puzzle["train"])
    if any(info["early"] for info in infos[n_train:]):
        assert any(info["early"] for info in infos[:n_train])
    return infos


def validate_generated_puzzle(puzzle: dict) -> None:
    _validate(puzzle)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({(len(p["train"]), len(p["test"])) for p in puzzles}) >= 10
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    summaries = [_validate(p) for p in puzzles]
    all_infos = [info for group in summaries for info in group]
    assert len({info["shape"] for info in all_infos}) >= 100
    assert len({info["roles"] for info in all_infos}) >= 20
    assert {info["partials"] for info in all_infos} == {1, 2, 3, 4}
    assert any(info["early"] for info in all_infos)
    assert any(any(len(state) > 1 for state in info["exemplar"]) and
               any(len(state) == 1 for state in info["exemplar"])
               for info in all_infos)
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1),
                              (1003, 8, 3), (1004, 10, 4)):
        p = generate_puzzle(seed, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
        validate_generated_puzzle(p)
    for seed, train, test in ((2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        p = generate_puzzle(seed, train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
        validate_generated_puzzle(p)
    for args in ({"num_train": 1}, {"num_train": 11},
                 {"num_test": 0}, {"num_test": 5},
                 {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(777), generate_puzzle(777)
    assert all(_key(x["input"]) == _key(y["input"]) and
               _key(x["output"]) == _key(y["output"])
               for split in ("train", "test") for x, y in zip(a[split], b[split]))
    assert len({_key(pair["input"]) for p in puzzles for split in ("train", "test")
                for pair in p[split]}) > 200
    assert len({tuple(_key(pair["input"]) for split in ("train", "test")
                      for pair in p[split]) for p in puzzles}) == 42
