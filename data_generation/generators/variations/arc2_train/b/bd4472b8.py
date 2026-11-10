"""Ordered color keys expanded into cyclic, full-width rows."""

import numpy as np


def _sig(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _expand(grid):
    output = grid.copy()
    width = grid.shape[1]
    for row in range(2, grid.shape[0]):
        output[row, :] = grid[0, (row - 2) % width]
    return output


def _source_pair(key, height):
    width = len(key)
    grid = np.zeros((height, width), dtype=np.int8)
    grid[0] = [int(c) for c in key]
    grid[1] = 5
    return _sig(grid), _sig(_expand(grid))


_SOURCE = frozenset(_source_pair(k, h) for k, h in (
    ("3214", 10), ("214", 8), ("83", 6), ("12348", 12)
))


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
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


def _roles(rng, count, mode):
    first = tuple(int(x) for x in rng.choice(10, size=2, replace=False))
    if mode == "shared":
        return [first] * count
    roles = [first]
    for _ in range(1, count):
        if mode == "background":
            choices = [c for c in range(10) if c not in first]
            roles.append((int(rng.choice(choices)), first[1]))
        elif mode == "separator":
            choices = [c for c in range(10) if c not in first]
            roles.append((first[0], int(rng.choice(choices))))
        else:
            choices = [c for c in range(10) if c not in first]
            changed = rng.choice(choices, size=2, replace=False)
            roles.append((int(changed[0]), int(changed[1])))
    return roles


def _widths(rng, ntrain, ntest, mode, repeats):
    train_max = 8 if repeats else 7
    train = [int(rng.integers(3, train_max + 1))]
    train.append(train[0] if rng.random() < .45 else int(rng.integers(2, train_max + 1)))
    train.extend(int(rng.integers(2, train_max + 1)) for _ in range(ntrain - 2))
    upper = 12 if repeats else 8
    if mode == "longer":
        choices = list(range(max(train) + 1, upper + 1))
    elif mode == "familiar":
        choices = sorted(set(train))
    elif mode == "narrower":
        choices = list(range(2, max(train)))
    else:
        choices = list(range(2, upper + 1))
    test = [int(rng.choice(choices)) for _ in range(ntest)]
    if ntest > 1 and rng.random() < .55:
        test[1] = test[0]
    return train, test


def _rows(rng, width, regime, index):
    if regime == "source_train" and index == 0:
        return 2 * width
    if regime == "source_train" and index == 1:
        return int(rng.integers(1, width)) + width * int(rng.integers(0, 2))
    forms = [
        int(rng.integers(1, width)),
        width,
        width + int(rng.integers(1, width)),
        2 * width,
        min(28, 2 * width + int(rng.integers(1, width))),
        int(rng.integers(max(1, 29 - width), 29)),
    ]
    return int(rng.choice(forms[1:] if regime == "target_test" else forms))


def _key(rng, width, background, separator, repeat_kind):
    palette = np.asarray([c for c in range(10) if c not in (background, separator)], dtype=np.int8)
    if repeat_kind is None or width == 2:
        return rng.choice(palette, size=width, replace=False).astype(np.int8)
    size = int(rng.integers(2, min(6, width, len(palette)) + 1))
    colors = rng.choice(palette, size=size, replace=False)
    key = rng.choice(colors, size=width, replace=True).astype(np.int8)
    if repeat_kind == "adjacent":
        i = int(rng.integers(0, width - 1))
        key[i + 1] = key[i]
        other = 0 if i > 0 else 2
        key[other] = colors[0] if key[i] != colors[0] else colors[1]
    else:
        key[0] = colors[0]
        key[1] = colors[1]
        key[2] = key[0]
        for i in range(3, width):
            available = [c for c in colors if c != key[i - 1]]
            key[i] = rng.choice(available)
    return key


def _kind(grid):
    key = [int(v) for v in grid[0]]
    if len(set(key)) == len(key):
        return None
    if any(key[i] == key[i + 1] for i in range(len(key) - 1)):
        return "adjacent"
    return "separated"


def _pair(rng, width, rows, roles, repeat_kind):
    background, separator = roles
    grid = np.full((rows + 2, width), background, dtype=np.int8)
    grid[0] = _key(rng, width, background, separator, repeat_kind)
    grid[1] = separator
    return {"input": grid, "output": _expand(grid)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(seed, num_train, num_test)
    role_mode = str(rng.choice(("shared", "background", "separator", "both")))
    width_mode = str(rng.choice(("longer", "familiar", "narrower", "mixed")))
    repeats = bool(rng.integers(0, 2))
    preferred = str(rng.choice(("adjacent", "separated"))) if repeats else None
    train_roles = _roles(rng, ntrain, role_mode)
    train_widths, test_widths = _widths(rng, ntrain, ntest, width_mode, repeats)
    test_roles = [train_roles[int(rng.integers(ntrain))] for _ in range(ntest)]
    if role_mode != "shared":
        test_roles[0] = train_roles[-1]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, widths, roles in (
        ("train", train_widths, train_roles),
        ("test", test_widths, test_roles),
    ):
        for index, (width, role) in enumerate(zip(widths, roles)):
            regime = "source_train" if split == "train" else "target_test"
            rows = _rows(rng, width, regime, index)
            if split == "train":
                repeat_kind = preferred if repeats and (index == 0 or (index >= 2 and rng.random() < .55)) else None
            else:
                repeat_kind = preferred if repeats and (width > 8 or rng.random() < .65) else None
            for attempt in range(100):
                pair = _pair(rng, width, rows, role, repeat_kind)
                signature = (_sig(pair["input"]), _sig(pair["output"]))
                if signature not in seen and signature not in _SOURCE:
                    break
                if attempt > 30:
                    rows = 1 + rows % 28
            else:
                raise RuntimeError("failed to make a unique pair")
            puzzle[split].append(pair)
            seen.add(signature)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    roles = {"train": set(), "test": set()}
    train_repeat_kinds = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            height, width = inp.shape
            assert height >= 3 and 2 <= width <= 12
            background, separator = int(inp[2, 0]), int(inp[1, 0])
            key_colors = set(int(c) for c in inp[0])
            assert len(key_colors) >= 2
            assert background != separator
            assert background not in key_colors and separator not in key_colors
            assert np.all(inp[1] == separator) and np.all(inp[2:] == background)
            assert np.array_equal(inp[:2], out[:2])
            assert not np.array_equal(inp, out)
            for row in range(2, height):
                assert np.all(out[row] == inp[0, (row - 2) % width])
            signature = (_sig(inp), _sig(out))
            assert signature not in seen and signature not in _SOURCE
            seen.add(signature)
            roles[split].add((background, separator))
            if split == "train" and _kind(inp) is not None:
                train_repeat_kinds.add(_kind(inp))
            if split == "test" and _kind(inp) is not None:
                assert _kind(inp) in train_repeat_kinds
            if split == "test":
                assert height - 2 >= width
    assert roles["test"] <= roles["train"]
    train = puzzle["train"]
    assert train[0]["input"].shape[0] - 2 >= train[0]["input"].shape[1] + 1
    assert (train[1]["input"].shape[0] - 2) % train[1]["input"].shape[1] != 0
    if train_repeat_kinds:
        assert _kind(train[0]["input"]) is not None
        assert train[0]["input"].shape[0] - 2 >= train[0]["input"].shape[1]


def _puzzle_signature(puzzle):
    return tuple(tuple((_sig(p["input"]), _sig(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for seed, train, test in (
        (100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 5, 2),
        (104, None, 1), (105, None, 4), (106, 2, None), (107, 10, None)
    ):
        p = generate_puzzle(seed, train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
        validate_generated_puzzle(p)
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(200, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts {args}")
    assert _puzzle_signature(generate_puzzle(77)) == _puzzle_signature(generate_puzzle(77))
    assert _puzzle_signature(generate_puzzle(78, 7, 3)) == _puzzle_signature(generate_puzzle(78, 7, 3))
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    widths = {"train": set(), "test": set()}
    forms = {"train": set(), "test": set()}
    role_variation = set()
    kinds = set()
    relations = set()
    same_width_heights = 0
    full_tests = partial_tests = 0
    varied_background = varied_separator = 0
    repeated_puzzles = repeated_role_short_shift = 0
    repeated_test_widths = 0
    full_train = partial_train = first_cycle_train = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        tw = [p["input"].shape[1] for p in puzzle["train"]]
        backgrounds = {int(p["input"][2, 0]) for p in puzzle["train"]}
        separators = {int(p["input"][1, 0]) for p in puzzle["train"]}
        role_variation.add(len(backgrounds) > 1 or len(separators) > 1)
        varied_background += len(backgrounds) > 1
        varied_separator += len(separators) > 1
        has_repeat = any(_kind(p["input"]) is not None for p in puzzle["train"])
        repeated_puzzles += has_repeat
        short = any(p["input"].shape[0] - 2 < p["input"].shape[1] for p in puzzle["train"])
        nonlonger = any(p["input"].shape[1] <= max(tw) for p in puzzle["test"])
        repeated_role_short_shift += has_repeat and (len(backgrounds) > 1 or len(separators) > 1) and short and nonlonger
        repeated_test_widths += len({p["input"].shape[1] for p in puzzle["test"]}) < len(puzzle["test"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                width, rows = inp.shape[1], inp.shape[0] - 2
                widths[split].add(width)
                forms[split].add((rows // width, rows % width))
                kinds.add(_kind(inp))
                assert all(np.all(out[2 + i] == inp[0, i % width]) for i in range(rows))
                if split == "test":
                    relations.add("longer" if width > max(tw) else "familiar" if width in tw else "narrower")
                    full_tests += rows % width == 0
                    partial_tests += rows % width != 0
                else:
                    full_train += rows % width == 0
                    partial_train += rows % width != 0
                    first_cycle_train += rows < width
        same_width_heights += any(
            a["input"].shape[1] == b["input"].shape[1]
            and a["input"].shape[0] != b["input"].shape[0]
            for i, a in enumerate(puzzle["train"]) for b in puzzle["train"][i + 1:]
        )
    assert widths["train"] >= {2, 3, 4, 5, 6, 7}
    assert widths["test"] >= {2, 3, 4, 5, 6, 7, 8}
    assert len(forms["train"]) >= 12 and len(forms["test"]) >= 12
    assert role_variation == {False, True}
    assert kinds == {None, "adjacent", "separated"}
    assert relations == {"longer", "familiar", "narrower"}
    assert same_width_heights >= 4
    assert full_tests >= 5 and partial_tests >= 15
    assert full_train >= 42 and partial_train >= 42 and first_cycle_train >= 8
    assert varied_background >= 8 and varied_separator >= 8
    assert repeated_puzzles >= 8 and repeated_test_widths >= 4
    assert repeated_role_short_shift >= 3
