import numpy as np


_SOURCE_INPUTS = (
    ("125503", "302313", "153521", "235115", "330120", "315131"),
    ("0959", "9252", "0251", "1292"),
    ("912112", "112636", "616696", "629693", "126261", "163323"),
    ("121000", "061010", "041026", "412111"),
    ("6858", "4964", "5894", "6994", "8895", "9569"),
)
_LAYOUTS = ("shuffle", "patch", "band", "island", "checker", "runs")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(seed, rng, num_train, num_test):
    train = None if num_train is None else _integer(num_train, "num_train")
    test = None if num_test is None else _integer(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be 2 through 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be 1 through 4")
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _shape(rng, aspect=None):
    if aspect is None:
        aspect = ("square", "horizontal", "vertical")[int(rng.integers(3))]
    if aspect == "square":
        side = int(rng.integers(4, 13))
        return side, side
    short = int(rng.integers(4, 11))
    long = int(rng.integers(short + 1, min(17, short + 7)))
    return (short, long) if aspect == "horizontal" else (long, short)


def _profile(rng, n, kind):
    # Rank two is strictly above rank three. Sampling c near n/5 makes a
    # small cutoff gap possible while leaving all five colors substantial.
    for _ in range(80):
        if kind == "sparse":
            c = int(rng.integers(1, max(3, min(5, n // 12 + 1))))
        else:
            upper = max(1, (n - 2) // 5)
            lower = max(1, min(upper, n // 10))
            c = int(rng.integers(lower, upper + 1))
        if kind == "minority_tie" or kind == "sparse":
            a = b = c
        else:
            a = int(rng.integers(max(1, (2 * c) // 3), c + 1))
            b = int(rng.integers(a, c + 1))
            if kind == "varied" and a == b == c:
                if c > 1:
                    a -= 1
                else:
                    continue
        remaining = n - a - b - c
        if kind == "top_tie":
            if remaining % 2:
                continue
            d = remaining // 2
            if d <= c:
                continue
            return (d, d, c, b, a)
        maximum_d = remaining // 2
        if maximum_d <= c:
            continue
        if kind == "sparse":
            d = int(rng.integers(c + 1, maximum_d + 1))
        else:
            d = min(maximum_d, c + int(rng.choice((1, 1, 1, 2, 2, 3, 4))))
        e = remaining - d
        if kind == "minority_tie" and e == d:
            if d == c + 1:
                continue
            d -= 1
            e += 1
        return (e, d, c, b, a)
    # Deterministic exhaustive fallback for the smallest admissible grids.
    for c in range(1, n // 5 + 1):
        for b in range(1, c + 1):
            for a in range(1, b + 1):
                if kind == "minority_tie" and (a, b) != (c, c):
                    continue
                if kind == "varied" and a == b == c:
                    continue
                rem = n - a - b - c
                for d in range(c + 1, rem // 2 + 1):
                    e = rem - d
                    if kind == "top_tie" and e != d:
                        continue
                    if kind == "minority_tie" and e == d:
                        continue
                    return (e, d, c, b, a)
    raise AssertionError("no strict five-color profile")


def _palette(rng, colors, include=None):
    if include is None:
        return tuple(int(x) for x in rng.choice(colors, size=5, replace=False))
    others = [c for c in colors if c != include]
    return (include,) + tuple(int(x) for x in rng.choice(others, size=4, replace=False))


def _assign_counts(rng, palette, profile, forced=None):
    order = [int(x) for x in rng.permutation(palette)]
    if forced is not None:
        color, role = forced
        destination = int(rng.integers(0, 2) if role == "keep" else rng.integers(2, 5))
        position = order.index(color)
        order[position], order[destination] = order[destination], order[position]
    return {color: int(count) for color, count in zip(order, profile)}


def _arrange(rng, shape, counts, mode):
    h, w = shape
    colors = list(counts)
    remaining = np.ones((h, w), dtype=bool)
    grid = np.empty((h, w), dtype=np.int8)
    yy, xx = np.indices((h, w))
    if mode == "shuffle":
        flat = np.repeat(np.array(colors, dtype=np.int8), list(counts.values()))
        rng.shuffle(flat)
        return flat.reshape(shape)
    if mode == "runs":
        flat = np.empty(h * w, dtype=np.int8)
        left = dict(counts)
        previous = None
        for i in range(flat.size):
            available = [c for c in colors if left[c] > 0]
            if previous in available and rng.random() < .68:
                chosen = previous
            else:
                chosen = available[int(rng.choice(len(available), p=np.array([left[c] for c in available]) / sum(left[c] for c in available)))]
            flat[i] = chosen
            left[chosen] -= 1
            previous = chosen
        return flat.reshape(shape) if rng.random() < .5 else flat.reshape(w, h).T.copy()
    if mode == "island":
        # Put a minority color in one connected, locally dense region.
        minority = min(colors, key=lambda c: counts[c])
        ordered = [minority] + [c for c in rng.permutation(colors) if c != minority]
    else:
        ordered = [int(c) for c in rng.permutation(colors)]
    direction = int(rng.integers(4))
    cy = int(rng.integers(h))
    cx = int(rng.integers(w))
    for index, color in enumerate(ordered[:-1]):
        if mode == "band":
            coordinate = (yy, xx, yy + xx, yy - xx)[direction]
            score = coordinate.astype(float) * (1 if index % 2 else -1) + rng.random(shape) * .65
        elif mode == "checker":
            score = ((yy + xx + index) % 2).astype(float) * 3 + rng.random(shape)
        elif mode == "island" and index == 0:
            score = np.abs(yy - cy) + np.abs(xx - cx) + rng.random(shape) * .1
        else:
            if index:
                cy = int(rng.integers(h))
                cx = int(rng.integers(w))
            score = np.abs(yy - cy) + np.abs(xx - cx) + rng.random(shape) * (1.0 if mode == "patch" else 2.5)
        positions = np.flatnonzero(remaining)
        selected = positions[np.argpartition(score.flat[positions], counts[color] - 1)[:counts[color]]]
        grid.flat[selected] = color
        remaining.flat[selected] = False
    grid[remaining] = ordered[-1]
    return grid


def _source_match(grid):
    return any(grid.shape == (len(rows), len(rows[0])) and
               all(''.join(str(int(v)) for v in grid[i]) == row for i, row in enumerate(rows))
               for rows in _SOURCE_INPUTS)


def _solve_input(grid, highlight):
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    assert len(colors) == 5 and counts[order[1]] > counts[order[2]]
    keep = colors[order[:2]]
    result = grid.copy()
    result[~np.isin(grid, keep)] = highlight
    return result


def _pair(rng, shape, palette, profile_kind, mode, highlight, forced=None):
    profile = _profile(rng, shape[0] * shape[1], profile_kind)
    counts = _assign_counts(rng, palette, profile, forced)
    grid = _arrange(rng, shape, counts, mode)
    if _source_match(grid):
        positions = np.flatnonzero(grid != grid.flat[0])
        grid = grid.copy()
        first = grid.flat[0].copy()
        grid.flat[0] = grid.flat[positions[0]]
        grid.flat[positions[0]] = first
    return {"input": grid, "output": _solve_input(grid, highlight)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(seed, rng, num_train, num_test)
    highlight = seed % 10
    colors = tuple(c for c in range(10) if c != highlight)
    source_shift = rng.random() < .28
    held_out = int(rng.choice(colors)) if source_shift else None
    training_colors = tuple(c for c in colors if c != held_out)
    first_palette = _palette(rng, training_colors)
    role_color = int(first_palette[int(rng.integers(5))])
    train_palettes = [first_palette]
    if rng.random() < .6:
        train_palettes.append(first_palette)
    else:
        train_palettes.append(_palette(rng, training_colors, role_color))
    for _ in range(2, ntrain):
        train_palettes.append(train_palettes[int(rng.integers(len(train_palettes)))] if rng.random() < .24 else _palette(rng, training_colors))
    train_shapes = []
    if source_shift:
        train_shapes = [_shape(rng, "square"), _shape(rng, "horizontal")]
    else:
        aspects = rng.permutation(("square", "horizontal", "vertical"))
        train_shapes = [_shape(rng, str(aspects[0])), _shape(rng, str(aspects[1]))]
    for _ in range(2, ntrain):
        train_shapes.append(_shape(rng))
    test_shapes = []
    for i in range(ntest):
        if source_shift and i == 0:
            h, w = train_shapes[1]
            shape = (w, h)
        elif rng.random() < .32:
            shape = train_shapes[int(rng.integers(ntrain))]
        else:
            shape = _shape(rng)
        test_shapes.append(shape)
    train_union = set().union(*map(set, train_palettes))
    test_palettes = []
    for i in range(ntest):
        if source_shift and i == 0:
            other = [c for c in colors if c not in (held_out, role_color)]
            palette = (held_out, role_color) + tuple(int(c) for c in rng.choice(other, size=3, replace=False))
            test_palettes.append(palette)
            continue
        choice = rng.random()
        if choice < .29:
            palette = train_palettes[int(rng.integers(ntrain))]
        elif choice < .63 and len(train_union) >= 5:
            palette = _palette(rng, sorted(train_union))
        elif choice < .83 and len(set(colors) - train_union) > 0:
            unseen = int(rng.choice(sorted(set(colors) - train_union)))
            palette = _palette(rng, colors, unseen)
        else:
            palette = _palette(rng, colors)
        test_palettes.append(palette)
    child_rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(int(rng.integers(2**63))).spawn(ntrain + ntest)]
    layout_order = [str(x) for x in rng.permutation(_LAYOUTS)]
    train = []
    for i in range(ntrain):
        child = child_rngs[i]
        kind = ("top_tie", "minority_tie")[i] if i < 2 else str(child.choice(("top_tie", "minority_tie", "varied", "varied", "sparse")))
        forced = (role_color, "keep") if i == 0 else (role_color, "change") if i == 1 else None
        mode = layout_order[i % len(layout_order)] if i < 2 else str(child.choice(_LAYOUTS))
        train.append(_pair(child, train_shapes[i], train_palettes[i], kind, mode, highlight, forced))
    test = []
    for i in range(ntest):
        child = child_rngs[ntrain + i]
        kind = str(child.choice(("top_tie", "minority_tie", "minority_tie", "varied", "varied", "varied", "sparse")))
        mode = str(child.choice(_LAYOUTS))
        forced = (role_color, "change") if source_shift and i == 0 else None
        test.append(_pair(child, test_shapes[i], test_palettes[i], kind, mode, highlight, forced))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _analyze(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    x, y = pair["input"], pair["output"]
    for grid in (x, y):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert all(1 <= side <= 30 for side in grid.shape)
        assert np.all((0 <= grid) & (grid <= 9))
    assert x.shape == y.shape and not np.array_equal(x, y)
    colors, counts = np.unique(x, return_counts=True)
    assert len(colors) == 5
    ordered = np.argsort(-counts, kind="stable")
    assert counts[ordered[1]] > counts[ordered[2]]
    changed = x != y
    replacement = np.unique(y[changed])
    assert len(replacement) == 1
    highlight = int(replacement[0])
    assert highlight not in colors
    assert np.array_equal(y, _solve_input(x, highlight))
    assert not _source_match(x)
    return {"highlight": highlight, "keep": set(map(int, colors[ordered[:2]])),
            "change": set(map(int, colors[ordered[2:]])),
            "palette": frozenset(map(int, colors)),
            "counts": tuple(sorted((int(v) for v in counts), reverse=True))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            x = pair["input"]
            key = _key(x)
            assert key not in seen
            seen.add(key)
            analyses[split].append(_analyze(pair))
    highlights = {a["highlight"] for split in analyses.values() for a in split}
    assert len(highlights) == 1
    train = analyses["train"]
    assert train[0]["counts"][0] == train[0]["counts"][1]
    assert train[1]["counts"][0] > train[1]["counts"][1]
    assert train[1]["counts"][2] == train[1]["counts"][3] == train[1]["counts"][4]
    assert train[0]["keep"] & train[1]["change"]
    assert puzzle["train"][0]["input"].shape != puzzle["train"][1]["input"].shape
    # Infer the convention from training arrays alone, then predict every test.
    learned = train[0]["highlight"]
    for pair in puzzle["test"]:
        assert np.array_equal(_solve_input(pair["input"], learned), pair["output"])


def _equal(first, second):
    return all(len(first[k]) == len(second[k]) and all(
        np.array_equal(a[t], b[t]) for a, b in zip(first[k], second[k])
        for t in ("input", "output")) for k in ("train", "test"))


def _largest_component_fraction(grid, color):
    cells = set(map(tuple, np.argwhere(grid == color)))
    total = len(cells)
    largest = 0
    while cells:
        frontier = [cells.pop()]
        size = 1
        while frontier:
            y, x = frontier.pop()
            for neighbor in ((y - 1, x), (y + 1, x),
                             (y, x - 1), (y, x + 1)):
                if neighbor in cells:
                    cells.remove(neighbor)
                    frontier.append(neighbor)
                    size += 1
        largest = max(largest, size)
    return largest / total


def self_test() -> None:
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3), (None, 1),
                        (None, 4), (2, None), (10, None), (None, None)):
        p = generate_puzzle(6103, train, test)
        validate_generated_puzzle(p)
        assert _equal(p, generate_puzzle(6103, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4), (3.0, 1), (3, 1.0)):
        try:
            generate_puzzle(7, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    puzzles = [generate_puzzle(s) for s in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert not _equal(puzzles[0], puzzles[1])
    highlights, shapes, palettes, profiles, layouts = set(), set(), set(), set(), set()
    measures = {name: 0 for name in (
        "top_tie", "minority_tie", "minority_unequal", "close", "sparse",
        "substantial", "new_color", "no_new_color", "matched_shape",
        "transposed_shape", "same_palette", "dense_minority_dispersed_keep",
        "combined", "reused_train_palette")}
    aspect = {"train": set(), "test": set()}
    input_colors = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_colors = set().union(*(set(np.unique(p["input"])) for p in puzzle["train"]))
        train_shapes = {p["input"].shape for p in puzzle["train"]}
        train_palettes = {frozenset(map(int, np.unique(p["input"]))) for p in puzzle["train"]}
        measures["reused_train_palette"] += len(train_palettes) < len(puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = pair["input"]
                a = _analyze(pair)
                highlights.add(a["highlight"])
                input_colors.update(a["palette"])
                shapes.add(x.shape)
                palettes.add(a["palette"])
                profiles.add(a["counts"])
                aspect[split].add("square" if x.shape[0] == x.shape[1] else "vertical" if x.shape[0] > x.shape[1] else "horizontal")
                layouts.add((x.shape, int(np.sum(x[1:] == x[:-1])), int(np.sum(x[:, 1:] == x[:, :-1]))))
                if split == "test":
                    c = a["counts"]
                    close = c[1] - c[2] <= 2
                    same_palette = a["palette"] in train_palettes
                    measures["top_tie"] += c[0] == c[1]
                    measures["minority_tie"] += c[2] == c[3] == c[4]
                    measures["minority_unequal"] += len(set(c[2:])) > 1
                    measures["close"] += close
                    measures["sparse"] += sum(c[2:]) == 3
                    measures["substantial"] += c[4] >= 3
                    measures["new_color"] += bool(a["palette"] - train_colors)
                    measures["no_new_color"] += not bool(a["palette"] - train_colors)
                    measures["matched_shape"] += x.shape in train_shapes
                    measures["transposed_shape"] += x.shape[0] != x.shape[1] and (x.shape[1], x.shape[0]) in train_shapes
                    measures["same_palette"] += same_palette
                    measures["combined"] += close and len(set(c[2:])) > 1 and same_palette and x.shape[0] <= x.shape[1]
                    dense_minority = any(_largest_component_fraction(x, color) >= .75 for color in a["change"])
                    dispersed_keep = any(_largest_component_fraction(x, color) <= .5 for color in a["keep"])
                    measures["dense_minority_dispersed_keep"] += dense_minority and dispersed_keep
    assert len(highlights) >= 8 and len(shapes) >= 20
    assert len(palettes) >= 30 and len(profiles) >= 30 and len(layouts) >= 40
    assert highlights <= input_colors
    assert aspect["train"] == aspect["test"] == {"square", "vertical", "horizontal"}
    assert min(measures.values()) >= 3
