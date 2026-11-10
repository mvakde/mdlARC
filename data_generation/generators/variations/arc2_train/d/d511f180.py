"""Generate ARC puzzles with a puzzle-wide, learned two-color exchange."""

import numpy as np


_SOURCE_INPUTS = (
    ((3, 5, 1), (4, 5, 8), (2, 4, 9)),
    ((2, 7, 8, 8, 8), (5, 5, 6, 5, 4), (8, 5, 5, 5, 2),
     (8, 8, 4, 3, 6), (6, 5, 1, 9, 3)),
    ((6, 5, 3), (5, 7, 5), (8, 8, 2)),
    ((8, 8, 4, 5), (3, 8, 7, 5), (3, 7, 1, 9), (6, 4, 8, 8)),
)


def _signature(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


_SOURCE_SIGNATURES = frozenset(
    ((len(rows), len(rows[0])), tuple(x for row in rows for x in row))
    for rows in _SOURCE_INPUTS
)


def _counts(rng: np.random.Generator, train: int | None,
            test: int | None) -> tuple[int, int]:
    for name, value, low, high in (("num_train", train, 2, 10),
                                   ("num_test", test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} is outside its legal integer range")
    if test is None and train is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return int(train), int(test)


def _shape(rng: np.random.Generator) -> tuple[int, int]:
    mode = int(rng.integers(10))
    if mode < 3:
        side = int(rng.choice((3, 4, 5)))
        return side, side
    if mode < 5:
        side = int(rng.integers(2, 10))
        return side, side
    if mode < 8:
        height = int(rng.integers(2, 11))
        width = int(rng.integers(2, 11))
        if height == width:
            width = width % 10 + 2
        return height, width
    if mode == 8:
        length = int(rng.integers(3, 15))
        return (1, length) if rng.random() < 0.5 else (length, 1)
    return int(rng.integers(8, 17)), int(rng.integers(8, 17))


def _palette(rng: np.random.Generator, swap: tuple[int, int],
             area: int, first: bool) -> tuple[int, ...]:
    inert = [c for c in range(10) if c not in swap]
    limit = min(8, area - (2 if first else 0))
    if first:
        count = int(rng.integers(2, min(4, limit) + 1))
    elif rng.random() < 0.20:
        count = 0
    else:
        count = int(rng.integers(0, limit + 1))
    chosen = rng.choice(inert, size=count, replace=False)
    return tuple(int(x) for x in chosen)


def _color_counts(rng: np.random.Generator, area: int,
                  swap: tuple[int, int], inert: tuple[int, ...],
                  first: bool) -> dict[int, int]:
    a, b = swap
    if first:
        active = (a, b)
    else:
        choice = int(rng.choice(4, p=(0.70, 0.11, 0.11, 0.08)))
        active = ((a, b), (a,), (b,), ())[choice]
    if not active and len(inert) < 2:
        available = [c for c in range(10) if c not in swap and c not in inert]
        inert = inert + tuple(int(c) for c in rng.choice(
            available, size=2 - len(inert), replace=False))
    elif len(active) == 1 and not inert and rng.random() < 0.8:
        available = [c for c in range(10) if c not in swap]
        inert = (int(rng.choice(available)),)
    if not active and not inert:
        active = (a, b)
    if len(active) + len(inert) > area:
        inert = inert[:area - len(active)]
    if not active:
        swap_total = 0
    else:
        remaining = area - len(inert)
        floor = len(active)
        if remaining == floor:
            swap_total = floor
        else:
            fraction = float(rng.choice((0.08, 0.22, 0.45, 0.70, 0.92)))
            swap_total = max(floor, min(remaining, int(round(area * fraction))))
    counts = {color: 0 for color in swap}
    if len(active) == 1:
        counts[active[0]] = swap_total
    elif len(active) == 2:
        if swap_total == 2:
            count_a = 1
        elif rng.random() < 0.23:
            count_a = swap_total // 2
        else:
            ratio = float(rng.beta(0.85, 0.85))
            count_a = max(1, min(swap_total - 1, int(round(swap_total * ratio))))
        counts[a], counts[b] = count_a, swap_total - count_a
    inert_total = area - swap_total
    if inert:
        if inert_total < len(inert):
            raise AssertionError("palette does not fit")
        weights = rng.dirichlet(np.full(len(inert), float(rng.choice((0.4, 1.0, 2.5)))))
        extras = rng.multinomial(inert_total - len(inert), weights)
        counts.update({color: int(extras[i]) + 1 for i, color in enumerate(inert)})
    elif inert_total:
        if active == (a, b):
            counts[a] += inert_total // 2
            counts[b] += inert_total - inert_total // 2
        else:
            counts[active[0]] += inert_total
    return {color: count for color, count in counts.items() if count}


def _arrange(rng: np.random.Generator, shape: tuple[int, int],
             counts: dict[int, int], swap: tuple[int, int]) -> np.ndarray:
    height, width = shape
    area = height * width
    yy, xx = np.indices(shape)
    y, x = yy.ravel(), xx.ravel()
    noise = rng.random(area)
    mode = int(rng.integers(9))
    if mode == 0:
        score = noise
    elif mode == 1:
        score = y + noise * float(rng.uniform(0.1, 1.4))
    elif mode == 2:
        score = x + noise * float(rng.uniform(0.1, 1.4))
    elif mode == 3:
        score = y / max(1, height - 1) + x / max(1, width - 1) + noise * 0.25
    elif mode == 4:
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        score = (y - cy) ** 2 + (x - cx) ** 2 + noise * 2
    elif mode == 5:
        field = rng.random(shape)
        for _ in range(2):
            field = (field + np.roll(field, 1, 0) + np.roll(field, -1, 0)
                     + np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5
        score = field.ravel() + noise * 0.035
    elif mode == 6:
        score = np.minimum(np.abs(y - int(rng.integers(height))),
                           np.abs(x - int(rng.integers(width)))) + noise * 0.4
    elif mode == 7:
        score = (y + x) % 2 + noise * 0.8
    else:
        score = (y / max(1, height - 1) - x / max(1, width - 1)) + noise * 0.3

    order = np.argsort(score, kind="stable")
    colors = list(counts)
    rng.shuffle(colors)
    flat = np.empty(area, dtype=np.int8)
    start = 0
    for color in colors:
        finish = start + counts[color]
        flat[order[start:finish]] = color
        start = finish
    if mode == 7 and all(c in counts for c in swap):
        selected = np.flatnonzero(np.isin(flat, swap))
        parity = (y[selected] + x[selected]) % 2
        ranks = np.argsort(parity + rng.random(len(selected)) * 0.3)
        flat[selected[ranks[:counts[swap[0]]]]] = swap[0]
        flat[selected[ranks[counts[swap[0]]:]]] = swap[1]
    return flat.reshape(shape)


def _swap(grid: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    output = grid.copy()
    output[grid == a] = b
    output[grid == b] = a
    return output


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    if rng.random() < 0.18:
        swap = (5, 8)
    else:
        chosen = rng.choice(10, size=2, replace=False)
        swap = tuple(sorted(int(x) for x in chosen))
    puzzle = {"train": [], "test": []}
    used = set()
    train_shapes = set()
    for regime, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            first = regime == "train" and index == 0
            for attempt in range(96):
                if regime == "test" and attempt == 0 and rng.random() < 0.35:
                    shape = tuple(train_shapes)[int(rng.integers(len(train_shapes)))]
                else:
                    shape = _shape(rng)
                if first and shape[0] * shape[1] < 4:
                    continue
                if regime == "train" and index == 1 and shape in train_shapes and attempt < 20:
                    continue
                area = shape[0] * shape[1]
                inert = _palette(rng, swap, area, first)
                counts = _color_counts(rng, area, swap, inert, first)
                grid = _arrange(rng, shape, counts, swap)
                signature = _signature(grid)
                if signature in used or signature in _SOURCE_SIGNATURES:
                    continue
                used.add(signature)
                puzzle[regime].append({"input": grid, "output": _swap(grid, swap)})
                if regime == "train":
                    train_shapes.add(shape)
                break
            else:
                raise AssertionError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_swap(train: list[dict]) -> tuple[int, int]:
    transitions: dict[int, set[int]] = {}
    unchanged = set()
    for pair in train:
        for src, dst in zip(pair["input"].flat, pair["output"].flat):
            src, dst = int(src), int(dst)
            if src == dst:
                unchanged.add(src)
            else:
                transitions.setdefault(src, set()).add(dst)
    assert len(transitions) == 2
    a, b = sorted(transitions)
    assert transitions == {a: {b}, b: {a}}
    assert a not in unchanged and b not in unchanged
    assert len(unchanged) >= 2
    return a, b


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert bool(np.all((grid >= 0) & (grid <= 9)))
        inp, out = pair["input"], pair["output"]
        assert inp.shape == out.shape
        sig = _signature(inp)
        pair_sig = sig, _signature(out)
        assert sig not in seen_inputs and pair_sig not in seen_pairs
        assert sig not in _SOURCE_SIGNATURES
        seen_inputs.add(sig)
        seen_pairs.add(pair_sig)

    first = puzzle["train"][0]
    assert np.any(first["input"] != first["output"])
    swap = _infer_swap(puzzle["train"])
    assert all(int(np.count_nonzero(first["input"] == c)) > 0 for c in swap)
    for pair in puzzle["train"] + puzzle["test"]:
        inp, out = pair["input"], pair["output"]
        expected = inp.copy()
        expected[inp == swap[0]] = swap[1]
        expected[inp == swap[1]] = swap[0]
        assert np.array_equal(expected, out)
    for pair in puzzle["test"]:
        inp = pair["input"]
        predicted = np.empty_like(inp)
        for color in np.unique(inp):
            value = int(color)
            predicted[inp == value] = (swap[1] if value == swap[0]
                                        else swap[0] if value == swap[1]
                                        else value)
        assert np.array_equal(predicted, pair["output"])


def _layout_signature(grid: np.ndarray) -> tuple:
    labels = {}
    normalized = []
    for cell in grid.flat:
        value = int(cell)
        if value not in labels:
            labels[value] = len(labels)
        normalized.append(labels[value])
    return tuple(grid.shape), tuple(normalized)


def self_test() -> None:
    train_counts, test_counts, swaps = set(), set(), set()
    shapes, palettes, layouts, inputs = set(), set(), set(), set()
    equal_frequency = sparse_both = common_both = 0
    only_one = neither = two_color = unseen_inert = 0
    train_rectangles = test_rectangles = strips = large = 0
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        swap = _infer_swap(puzzle["train"])
        swaps.add(swap)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_inert = set()
        for pair in puzzle["train"]:
            train_inert.update(int(x) for x in np.unique(pair["input"]) if x not in swap)
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, out = pair["input"], pair["output"]
                for index in np.ndindex(inp.shape):
                    old, new = int(inp[index]), int(out[index])
                    assert new == (swap[1] if old == swap[0]
                                   else swap[0] if old == swap[1] else old)
                counts = [int(np.count_nonzero(inp == c)) for c in swap]
                equal_frequency += counts[0] == counts[1] and counts[0] > 0
                sparse_both += counts[0] in (1, 2) and counts[1] in (1, 2)
                common_both += min(counts) >= 3
                only_one += (counts[0] == 0) != (counts[1] == 0)
                neither += counts == [0, 0]
                two_color += len(np.unique(inp)) == 2 and min(counts) > 0
                train_rectangles += regime == "train" and inp.shape[0] != inp.shape[1]
                test_rectangles += regime == "test" and inp.shape[0] != inp.shape[1]
                strips += 1 in inp.shape
                large += max(inp.shape) >= 10
                if regime == "test":
                    unseen_inert += bool(set(int(x) for x in np.unique(inp))
                                         - train_inert - set(swap))
                shapes.add(inp.shape)
                palettes.add(tuple(sorted(int(x) for x in np.unique(inp))))
                layouts.add(_layout_signature(inp))
                inputs.add(_signature(inp))
                total += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(swaps) >= 15 and (5, 8) in swaps
    assert len(shapes) >= 30 and len(palettes) >= 35
    assert len(layouts) >= int(total * 0.85)
    assert len(inputs) >= int(total * 0.95)
    assert min(equal_frequency, sparse_both, common_both, only_one,
               neither, two_color, unseen_inert, train_rectangles,
               test_rectangles, strips, large) > 0
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 3)):
        first = generate_puzzle(7000 + train * 10 + test, train, test)
        second = generate_puzzle(7000 + train * 10 + test, train, test)
        assert all(np.array_equal(p["input"], q["input"]) and
                   np.array_equal(p["output"], q["output"])
                   for split in ("train", "test")
                   for p, q in zip(first[split], second[split]))
        assert len(first["train"]) == train and len(first["test"]) == test
    for kwargs in ({"num_train": 2}, {"num_train": 10},
                   {"num_test": 1}, {"num_test": 4}):
        validate_generated_puzzle(generate_puzzle(8011, **kwargs))
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    assert _signature(generate_puzzle(9001, 6, 2)["train"][0]["input"]) != \
        _signature(generate_puzzle(9002, 6, 2)["train"][0]["input"])
