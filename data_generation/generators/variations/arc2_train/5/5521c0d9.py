import numpy as np


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} is outside {low}..{high}")
    return value


def _choose_counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + int(seed) % 4
        low = max(2, test + 1)
        train = low + (int(seed) // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _array_key(grid):
    return grid.shape, tuple(int(value) for value in grid.flat)


def _pair_key(pair):
    return _array_key(pair["input"]), _array_key(pair["output"])


def _render(normal, parallel, background, bars, rotation):
    """Build at the bottom edge, then rotate the entire example."""
    inp = np.full((normal, parallel), background, dtype=np.int8)
    out = inp.copy()
    for start, width, depth, color in bars:
        inp[normal - depth:normal, start:start + width] = color
        out[normal - 2 * depth:normal - depth, start:start + width] = color
    if rotation:
        inp = np.rot90(inp, rotation).copy()
        out = np.rot90(out, rotation).copy()
    return {"input": inp, "output": out}


def _source_pair_keys():
    # Transcribed from task.json; the test's middle 4 has width THREE.
    specs = (
        ((1, 4, 6, 4), (11, 2, 5, 2), (7, 2, 2, 1)),
        ((1, 2, 4, 1), (4, 4, 2, 2), (9, 4, 4, 4)),
        ((1, 4, 1, 2), (7, 1, 4, 1), (11, 2, 3, 4)),
        ((0, 4, 7, 2), (5, 3, 6, 4), (10, 5, 3, 1)),
    )
    return {_pair_key(_render(15, 15, 0, bars, 0)) for bars in specs}


def _components(grid, background):
    """Find connected objects without assuming one color per object."""
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == background:
                continue
            color = int(grid[row, col])
            pending = [(row, col)]
            seen[row, col] = True
            cells = []
            while pending:
                r, c = pending.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= nr < height and 0 <= nc < width
                            and not seen[nr, nc] and int(grid[nr, nc]) == color):
                        seen[nr, nc] = True
                        pending.append((nr, nc))
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r0, r1 = min(rows), max(rows) + 1
            c0, c1 = min(cols), max(cols) + 1
            assert len(cells) == (r1 - r0) * (c1 - c0)
            assert np.all(grid[r0:r1, c0:c1] == color)
            result.append((r0, r1, c0, c1, color))
    return result


def _analyze_input(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > max(int(counts[c]) for c in range(10) if c != background)
    objects = _components(grid, background)
    assert 1 <= len(objects) <= 5
    height, width = grid.shape
    common = set(("top", "bottom", "left", "right"))
    for r0, r1, c0, c1, _ in objects:
        edges = set()
        if r0 == 0:
            edges.add("top")
        if r1 == height:
            edges.add("bottom")
        if c0 == 0:
            edges.add("left")
        if c1 == width:
            edges.add("right")
        common &= edges
    assert len(common) == 1
    edge = next(iter(common))
    if edge in ("top", "bottom"):
        objects.sort(key=lambda box: box[2])
        positions = [(box[2], box[3]) for box in objects]
        normal = height
        depths = [box[1] - box[0] for box in objects]
        widths = [box[3] - box[2] for box in objects]
    else:
        objects.sort(key=lambda box: box[0])
        positions = [(box[0], box[1]) for box in objects]
        normal = width
        depths = [box[3] - box[2] for box in objects]
        widths = [box[1] - box[0] for box in objects]
    assert all(b[0] - a[1] >= 1 for a, b in zip(positions, positions[1:]))
    assert all(2 * depth <= normal for depth in depths)
    return {
        "background": background, "objects": objects, "edge": edge,
        "depths": depths, "widths": widths, "positions": positions,
    }


def _predict(grid):
    analysis = _analyze_input(grid)
    result = np.full_like(grid, analysis["background"])
    edge = analysis["edge"]
    for r0, r1, c0, c1, color in analysis["objects"]:
        if edge == "bottom":
            depth = r1 - r0
            r0, r1 = r0 - depth, r1 - depth
        elif edge == "top":
            depth = r1 - r0
            r0, r1 = r0 + depth, r1 + depth
        elif edge == "left":
            depth = c1 - c0
            c0, c1 = c0 + depth, c1 + depth
        else:
            depth = c1 - c0
            c0, c1 = c0 - depth, c1 - depth
        assert np.all(result[r0:r1, c0:c1] == analysis["background"])
        result[r0:r1, c0:c1] = color
    return result


def _sample_bars(rng, normal, parallel, count, background, palette,
                 anchor=None, anchor_depth=None, test=False, mixed=False):
    width_limit = min(9 if test else 8, parallel - 2) if count == 1 else (10 if test else 8)
    for _ in range(40):
        widths = rng.integers(1, width_limit + 1, size=count).astype(int)
        minimum = 2 if count == 1 else count - 1
        if int(widths.sum()) + minimum <= parallel:
            break
    else:
        widths = np.ones(count, dtype=int)
    extra = parallel - int(widths.sum()) - (2 if count == 1 else count - 1)
    weights = rng.dirichlet(np.ones(count + 1))
    spaces = rng.multinomial(extra, weights).astype(int)
    if count == 1:
        spaces[0] += 1
        spaces[-1] += 1
    else:
        spaces[1:-1] += 1
    starts = []
    cursor = int(spaces[0])
    for i, width in enumerate(widths):
        starts.append(cursor)
        cursor += int(width) + int(spaces[i + 1])
    depth_limit = min(normal // 2, 14 if test else 10)
    depths = rng.integers(1, depth_limit + 1, size=count).astype(int)
    if anchor_depth is not None:
        depths[0] = anchor_depth
    elif count > 1 and rng.random() < 0.45:
        depths[int(rng.integers(count))] = 1
    candidates = [c for c in (range(10) if mixed else palette) if c != background]
    colors = []
    for i in range(count):
        if i == 0 and anchor is not None:
            colors.append(anchor)
        elif colors and rng.random() < 0.28:
            colors.append(colors[int(rng.integers(len(colors)))])
        else:
            colors.append(int(rng.choice(candidates)))
    return [(int(starts[i]), int(widths[i]), int(depths[i]), colors[i])
            for i in range(count)]


def _make_pair(rng, edge_rotation, count, base_background, palette,
               mixed, anchor=None, anchor_depth=None, test=False):
    if test and rng.random() < 0.45:
        normal = int(rng.integers(15, 31))
        parallel = int(rng.integers(max(12, 2 * count - 1), 31))
    else:
        normal = int(rng.integers(8, 27))
        parallel = int(rng.integers(max(8, 2 * count - 1), 28))
    background = base_background
    if mixed and rng.random() < 0.55:
        background = int(rng.choice([c for c in range(10) if c != anchor]))
    bars = _sample_bars(rng, normal, parallel, count, background, palette,
                        anchor, anchor_depth, test, mixed)
    return _render(normal, parallel, background, bars, edge_rotation)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    analysis = {"train": [], "test": []}
    source = _source_pair_keys()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            ikey, pkey = _array_key(pair["input"]), _pair_key(pair)
            assert ikey not in inputs and pkey not in pairs and pkey not in source
            inputs.add(ikey)
            pairs.add(pkey)
            item = _analyze_input(pair["input"])
            assert np.array_equal(_predict(pair["input"]), pair["output"])
            assert not np.array_equal(pair["input"], pair["output"])
            analysis[split].append(item)
    train = analysis["train"]
    assert len(train[0]["objects"]) >= 2 and len(train[1]["objects"]) >= 2
    assert len(train[0]["objects"]) != len(train[1]["objects"])
    first = {color: set() for *_, color in train[0]["objects"]}
    second = {color: set() for *_, color in train[1]["objects"]}
    for item, depths, dest in ((train[0], train[0]["depths"], first),
                               (train[1], train[1]["depths"], second)):
        for box, depth in zip(item["objects"], depths):
            dest[box[4]].add(depth)
    assert any(len(first[color] | second[color]) >= 2 for color in first.keys() & second.keys())
    assert {item["edge"] for item in analysis["test"]} <= {item["edge"] for item in train}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(seed, rng, num_train, num_test)
    base_background = int(rng.integers(0, 10))
    palette = tuple(int(c) for c in rng.choice(
        [c for c in range(10) if c != base_background],
        size=int(rng.integers(3, 7)), replace=False))
    anchor = palette[0]
    mixed = bool(rng.random() < 0.55)
    rotations = list(int(k) for k in rng.permutation(4))
    edge_count = 1 if rng.random() < 0.48 else min(train_count, int(rng.integers(2, 5)))
    edges = rotations[:edge_count]
    first_count = int(rng.integers(2, 5))
    second_count = int(rng.choice([n for n in range(2, 6) if n != first_count]))
    first_depth = int(rng.integers(1, 5))
    second_depth = int(rng.choice([d for d in range(1, 5) if d != first_depth]))
    puzzle = {"train": [], "test": []}
    seen = set()
    source_keys = _source_pair_keys()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(80):
                pair_rng = np.random.default_rng(rng.integers(0, 2**63, dtype=np.int64))
                if split == "train":
                    bar_count = (first_count if index == 0 else second_count if index == 1
                                 else int(pair_rng.integers(1, 6)))
                    rotation = edges[index % len(edges)]
                    required_depth = first_depth if index == 0 else second_depth if index == 1 else None
                    required_color = anchor if index < 2 else None
                else:
                    bar_count = int(pair_rng.integers(1, 6))
                    rotation = int(pair_rng.choice(edges))
                    required_depth = None
                    required_color = None
                pair = _make_pair(pair_rng, rotation, bar_count, base_background,
                                  palette, mixed, required_color, required_depth,
                                  test=split == "test")
                key = _array_key(pair["input"])
                if key not in seen and _pair_key(pair) not in source_keys:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _equal(generate_puzzle(123, 8, 3), generate_puzzle(123, 8, 3))
    assert _equal(generate_puzzle(123), generate_puzzle(123))
    assert not _equal(generate_puzzle(123), generate_puzzle(124))
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                       (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1000 + i, train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
        validate_generated_puzzle(p)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(500, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
    items = [item for p in puzzles for split in ("train", "test")
             for item in (_analyze_input(pair["input"]) for pair in p[split])]
    assert len({i["edge"] for i in items}) == 4
    assert len({len(i["objects"]) for i in items}) == 5
    assert len({i["background"] for i in items}) >= 7
    assert len({pair["input"].shape for p in puzzles for s in ("train", "test")
                for pair in p[s]}) >= 50
    assert any(1 in i["depths"] for i in items)
    assert any(len({box[4] for box in i["objects"]}) < len(i["objects"]) for i in items)
    assert any(i["background"] != 0 and any(box[4] == 0 for box in i["objects"])
               for i in items)
    assert any(len({i["background"] for i in
                    [_analyze_input(pair["input"]) for s in ("train", "test")
                     for pair in p[s]]}) > 1 for p in puzzles)
    records = [(split, pair["input"].shape, _analyze_input(pair["input"]))
               for p in puzzles for split in ("train", "test") for pair in p[split]]
    assert all(any(split == s and 1 in a["depths"] for split, _, a in records)
               for s in ("train", "test"))
    assert all({a["edge"] for split, _, a in records if split == s}
               == {"top", "bottom", "left", "right"} for s in ("train", "test"))
    assert all({len(a["objects"]) for split, _, a in records if split == s}
               == {1, 2, 3, 4, 5} for s in ("train", "test"))
    assert max(w for _, _, a in records for w in a["widths"]) >= 8
    assert max(d for _, _, a in records for d in a["depths"]) >= 9
    assert any(w == 1 and d > 1 for _, _, a in records
               for w, d in zip(a["widths"], a["depths"]))
    assert any(d == 1 and w > 1 for _, _, a in records
               for w, d in zip(a["widths"], a["depths"]))
    assert any(w == d for _, _, a in records
               for w, d in zip(a["widths"], a["depths"]))
    assert any(w1 == w2 and d1 != d2 for _, _, a in records
               for i, (w1, d1) in enumerate(zip(a["widths"], a["depths"]))
               for w2, d2 in list(zip(a["widths"], a["depths"]))[i + 1:])
    assert any(d1 == d2 and w1 != w2 for _, _, a in records
               for i, (w1, d1) in enumerate(zip(a["widths"], a["depths"]))
               for w2, d2 in list(zip(a["widths"], a["depths"]))[i + 1:])
    margin_modes = set()
    gaps = set()
    layouts = set()
    for _, shape, a in records:
        parallel = shape[1] if a["edge"] in ("top", "bottom") else shape[0]
        margin_modes.add(int(a["positions"][0][0] > 0)
                         + int(a["positions"][-1][1] < parallel))
        gaps.update(right[0] - left[1] for left, right in
                    zip(a["positions"], a["positions"][1:]))
        layouts.add((a["edge"], shape, tuple(a["positions"]),
                     tuple(a["widths"]), tuple(a["depths"])))
    assert margin_modes == {0, 1, 2}
    assert 1 in gaps and max(gaps) >= 5
    assert len(layouts) >= int(0.9 * len(records))
    assert any(len({_analyze_input(pair["input"])["edge"] for split in ("train", "test")
                    for pair in p[split]}) > 1 for p in puzzles)
    assert all(len({len(_analyze_input(pair["input"])["objects"])
                    for split in ("train", "test") for pair in p[split]}) > 1
               for p in puzzles)
    assert any(any(max(_analyze_input(pair["input"])["depths"]) <= 4
                       and len(_analyze_input(pair["input"])["objects"]) <= 2
                       for pair in p["test"]) for p in puzzles)
    assert any(max(_analyze_input(pair["input"])["depths"])
               > max(max(_analyze_input(train["input"])["depths"]) for train in p["train"])
               for p in puzzles for pair in p["test"])
