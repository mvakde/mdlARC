import numpy as np

UP = ((-1, 0), (-1, 0), (0, 1), (0, 1))
DOWN = ((1, 0), (1, 0), (0, -1), (0, -1))
SOURCE_ANCHORS = {(13, 13, 7, 6), (13, 13, 3, 4), (13, 13, 5, 5)}


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside the allowed range")
    return value


def _counts(seed, train, test):
    if train is not None:
        train = _integer("num_train", train, 2, 10)
    if test is not None:
        test = _integer("num_test", test, 1, 4)
    if train is None and test is None:
        train = 2 + seed % 9
        test = 1 + (seed // 9) % min(4, train - 1)
    elif train is None:
        train = test + 1 + (seed // 4) % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _ray(h, w, anchor, moves):
    row, col = anchor
    cells = []
    step = 0
    while True:
        dr, dc = moves[step % 4]
        nr, nc = row + dr, col + dc
        if not (0 <= nr < h and 0 <= nc < w):
            side = "top" if nr < 0 else "bottom" if nr >= h else "left" if nc < 0 else "right"
            return tuple(cells), (side, step % 4)
        row, col = nr, nc
        cells.append((row, col))
        step += 1


def _geometry(h, w, anchor):
    upper, upper_stop = _ray(h, w, anchor, UP)
    lower, lower_stop = _ray(h, w, anchor, DOWN)
    return {"shape": (h, w), "anchor": anchor, "upper": upper, "lower": lower,
            "stops": (upper_stop, lower_stop), "mask": frozenset(upper + lower)}


def _shape(rng, split):
    low = 9 if split == "train" else 8
    category = int(rng.integers(4))
    if category == 0:
        return int(rng.integers(low, 16)), int(rng.integers(low, 16))
    if category == 1:
        return int(rng.integers(16, 31)), int(rng.integers(low, 16))
    if category == 2:
        return int(rng.integers(low, 16)), int(rng.integers(16, 31))
    return int(rng.integers(12, 31)), int(rng.integers(12, 31))


def _candidates(h, w, cache):
    if (h, w) not in cache:
        cache[(h, w)] = [_geometry(h, w, (r, c)) for r in range(h) for c in range(w)]
    return cache[(h, w)]


def _eligible(g, split, index):
    u, l = len(g["upper"]), len(g["lower"])
    if split == "train":
        return min(u, l) >= (8 if index == 0 else 5)
    return min(u, l) >= 1 and max(u, l) >= 7 and u + l >= 9


def _pick_geometry(rng, split, index, fixed_shape, previous, train_stops, cache):
    for attempt in range(18):
        h, w = fixed_shape if fixed_shape is not None else ((13, 13) if attempt == 17 else _shape(rng, split))
        choices = [g for g in _candidates(h, w, cache)
                   if _eligible(g, split, index)
                   and not any(g["shape"] == p["shape"] and g["anchor"] == p["anchor"] for p in previous)]
        if choices:
            break
    else:
        raise AssertionError("no legal geometry found")
    short_target = split == "test" and rng.random() < 0.32
    novel_target = split == "test" and rng.random() < 0.65
    if len(choices) > 180:
        choices = [choices[int(i)] for i in rng.choice(len(choices), size=180, replace=False)]
    prior_stops = {p["stops"] for p in previous}
    prior_upper = {p["stops"][0] for p in previous}
    prior_lower = {p["stops"][1] for p in previous}
    prior_lengths = {(len(p["upper"]), len(p["lower"])) for p in previous}
    best, best_score = None, -1e9
    for g in choices:
        stops = g["stops"]
        lengths = (len(g["upper"]), len(g["lower"]))
        score = float(rng.random()) * 2.2
        score += 2.0 * (stops not in prior_stops)
        score += 0.7 * (stops[0] not in prior_upper) + 0.7 * (stops[1] not in prior_lower)
        score += 0.4 * (lengths not in prior_lengths)
        if split == "test":
            score += (2.2 if novel_target else -6.0) * (stops not in train_stops)
            score += (3.5 if short_target else -2.5) * (min(lengths) < 5)
        for p in previous:
            if p["shape"] == g["shape"]:
                overlap = len(g["mask"] & p["mask"]) / len(g["mask"] | p["mask"])
                score -= 6.0 * overlap * overlap
                if overlap > 0.72:
                    score -= 3.0
        if score > best_score:
            best, best_score = g, score
    return best


def _make_pair(g, background, seed_color, trail):
    inp = np.full(g["shape"], background, dtype=np.int8)
    inp[g["anchor"]] = seed_color
    out = inp.copy()
    for cell in g["mask"]:
        out[cell] = trail
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _integer("seed", seed, 0, None)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    trail = int(rng.integers(10))
    stable_palette = bool(rng.random() < 0.28)
    stable_shape = bool(rng.random() < 0.24)
    colors = [color for color in range(10) if color != trail]
    fixed_colors = tuple(int(x) for x in rng.choice(colors, size=2, replace=False))
    fixed_shape = (int(rng.integers(13, 27)), int(rng.integers(13, 27))) if stable_shape else None
    cache, previous, color_history, train_stops = {}, [], set(), set()
    result = {"train": [], "test": []}
    for split, count in (("train", train_count), ("test", test_count)):
        generated = []
        for index in range(count):
            g = _pick_geometry(rng, split, index, fixed_shape, previous, train_stops, cache)
            if stable_palette:
                background, seed_color = fixed_colors
            else:
                choices = [(b, s) for b in colors for s in colors if b != s]
                fresh = [choice for choice in choices if choice not in color_history]
                pool = fresh if fresh else choices
                background, seed_color = pool[int(rng.integers(len(pool)))]
            color_history.add((background, seed_color))
            generated.append(_make_pair(g, background, seed_color, trail))
            previous.append(g)
            if split == "train":
                train_stops.add(g["stops"])
        rng.shuffle(generated)
        result[split] = generated
    validate_generated_puzzle(result)
    return result


def _grid_key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _facts(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert all(1 <= side <= 30 for side in grid.shape)
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape and inp.dtype == out.dtype
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 2 and sorted(int(c) for c in counts) == [1, inp.size - 1]
    seed_color = int(values[int(np.argmin(counts))])
    background = int(values[int(np.argmax(counts))])
    anchor = tuple(int(v) for v in np.argwhere(inp == seed_color)[0])
    assert np.count_nonzero(inp != background) == 1
    introduced = set(int(v) for v in np.unique(out)) - {background, seed_color}
    assert len(introduced) == 1
    trail = introduced.pop()
    assert set(int(v) for v in np.unique(out)) == {background, seed_color, trail}
    g = _geometry(*inp.shape, anchor)
    expected = inp.copy()
    for cell in g["mask"]:
        expected[cell] = trail
    assert np.array_equal(out, expected)
    assert out[anchor] == seed_color and not np.array_equal(inp, out)
    assert not (set(g["upper"]) & set(g["lower"]))
    assert not (inp.shape == (13, 13) and (13, 13, *anchor) in SOURCE_ANCHORS
                and (background, seed_color, trail) == (0, 8, 5))
    return g, (background, seed_color, trail)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, trails = set(), set(), set()
    train_facts, test_facts = [], []
    for split, facts_list in (("train", train_facts), ("test", test_facts)):
        for pair in puzzle[split]:
            g, roles = _facts(pair)
            key = _grid_key(pair["input"])
            pair_key = (key, _grid_key(pair["output"]))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            trails.add(roles[2])
            facts_list.append((g, roles))
    assert len(trails) == 1
    assert any(min(len(g["upper"]), len(g["lower"])) >= 8 for g, _ in train_facts)
    assert all(min(len(g["upper"]), len(g["lower"])) >= 5 for g, _ in train_facts)
    assert all(min(len(g["upper"]), len(g["lower"])) >= 1 and
               max(len(g["upper"]), len(g["lower"])) >= 7 and
               len(g["upper"]) + len(g["lower"]) >= 9 for g, _ in test_facts)


def _fingerprint(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_fingerprint(p) for p in puzzles}) == 42
    shapes, layouts, palettes, stops, upper, lower = set(), set(), set(), set(), set(), set()
    mixed_shapes = mixed_visible = stable_visible = short_tests = novel_tests = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        pairs = puzzle["train"] + puzzle["test"]
        facts = [_facts(pair) for pair in pairs]
        train_stops = {g["stops"] for g, _ in facts[:len(puzzle["train"])]}
        mixed_shapes += len({g["shape"] for g, _ in facts}) > 1
        mixed_visible += len({roles[:2] for _, roles in facts}) > 1
        stable_visible += len({roles[:2] for _, roles in facts}) == 1
        for g, roles in facts:
            shapes.add(g["shape"])
            layouts.add((g["shape"], g["anchor"], g["stops"]))
            palettes.add(roles)
            stops.add(g["stops"])
            upper.add(g["stops"][0])
            lower.add(g["stops"][1])
        for g, _ in facts[len(puzzle["train"]):]:
            short_tests += min(len(g["upper"]), len(g["lower"])) < 5
            novel_tests += g["stops"] not in train_stops
    assert len(shapes) >= 35 and len(layouts) >= 150 and len(palettes) >= 30
    assert len(stops) >= 8 and len(upper) >= 3 and len(lower) >= 3
    assert mixed_shapes >= 22 and mixed_visible >= 20 and stable_visible >= 3
    assert short_tests >= 3 and novel_tests >= 10
    assert any(h % 2 == 0 or w % 2 == 0 for h, w in shapes)
    assert any(h < 13 or w < 13 for h, w in shapes)
    assert any(h >= w + 6 for h, w in shapes) and any(w >= h + 6 for h, w in shapes)
    for seed, ntrain, ntest in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, 7, 3),
                                (2001, None, 1), (2002, None, 4), (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    assert _fingerprint(generate_puzzle(3001, 8, 3)) == _fingerprint(generate_puzzle(3001, 8, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                   {"num_train": 2.5}, {"num_test": True}):
        try:
            generate_puzzle(4001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
