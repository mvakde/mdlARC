"""Generate divider-defined compartment compression puzzles."""

import hashlib
import numpy as np

_SOURCE_HASHES = {
    "4ab32f745210fa7aa9d4af52ad7b5fa2bb168e77da4bdd07c30bc57b675fefa3",
    "84cbe920d189d3f615c5a721b2205129cdc3a9d4326e68f209968bd6374b327f",
    "bd23e0d5cd3fada46e160ecdc4eb01c72161df96f0deec646fa92a8cc4b6602a",
    "ae29d5226ea9810128a8fdc53475d572096b3fc1519a0be3516611657ae87e81",
}


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} out of range")
    return value


def _counts(seed, train, test):
    train, test = _count("num_train", train, 2, 10), _count("num_test", test, 1, 4)
    if train is None and test is None:
        pairs = [(a, b) for a in range(2, 11) for b in range(1, min(a - 1, 4) + 1)]
        return pairs[seed % len(pairs)]
    if train is None:
        choices = list(range(max(2, test + 1), 11))
        train = choices[seed % len(choices)]
    if test is None:
        choices = list(range(1, min(train - 1, 4) + 1))
        test = choices[seed % len(choices)]
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _shapes(rng, n_train, n_test):
    regime = int(rng.integers(0, 4))
    if regime == 0:
        train = [(2, 3), (3, 2)]
        train += [(2, 2), (2, 3), (3, 2)] * n_train
        train = train[:n_train]
        test = [(3, 3)] * n_test
    elif regime == 1:
        train = [(3, 3), (2, 3)]
        options = [(2, 2), (3, 2), (3, 3), (4, 2), (2, 4)]
        train += [options[int(rng.integers(0, len(options)))] for _ in range(n_train - 2)]
        options = [(4, 2), (2, 4), (4, 3), (3, 4), (4, 4)]
        test = [options[int(rng.integers(0, len(options)))] for _ in range(n_test)]
    elif regime == 2:
        train = [(3, 3), (2, 3)]
        options = [(2, 2), (2, 3), (3, 2), (3, 3), (4, 2)]
        train += [options[int(rng.integers(0, len(options)))] for _ in range(n_train - 2)]
        options = [(2, 2), (2, 3), (3, 2), (3, 3)]
        test = [options[int(rng.integers(0, len(options)))] for _ in range(n_test)]
    else:
        train = [(2, 3), (3, 2)]
        train += [(int(rng.integers(2, 5)), int(rng.integers(2, 5))) for _ in range(n_train - 2)]
        test = [(int(rng.integers(2, 5)), int(rng.integers(2, 5))) for _ in range(n_test)]
    rng.shuffle(train)
    return train, test


def _backgrounds(rng, n_train, total):
    base = 0 if rng.random() < 0.52 else int(rng.integers(1, 10))
    if rng.random() < 0.55:
        return [base] * total
    result = [base, int(rng.choice([x for x in range(10) if x != base]))]
    result.extend(int(rng.integers(0, 10)) for _ in range(total - 2))
    assert n_train >= 2
    return result


def _colors(rng, core, background, extra=None):
    available = [int(c) for c in core if int(c) != background]
    if len(available) < 2:
        available += [c for c in range(10) if c != background and c not in available]
    size = int(rng.integers(2, len(available) + 1))
    colors = [int(x) for x in rng.choice(available, size=size, replace=False)]
    if extra is not None and extra not in colors:
        colors.append(extra)
    return colors


def _adjacent(a):
    return bool(np.any(a[:, 1:] == a[:, :-1]) or np.any(a[1:, :] == a[:-1, :]))


def _labels(rng, shape, colors, mode, required=None):
    rows, cols = shape
    if required is not None and required not in colors:
        colors = colors + [required]
    if mode == "solid":
        return np.full(shape, required if required is not None else int(rng.choice(colors)), dtype=np.int8)
    a = np.empty(shape, dtype=np.int8)
    if mode == "none":
        if len(colors) == 2:
            offset = int(rng.integers(2))
            for r in range(rows):
                for c in range(cols):
                    a[r, c] = colors[(r + c + offset) % 2]
            return a
        for r in range(rows):
            for c in range(cols):
                forbidden = set()
                if r:
                    forbidden.add(int(a[r - 1, c]))
                if c:
                    forbidden.add(int(a[r, c - 1]))
                choices = [x for x in colors if x not in forbidden]
                assert choices
                a[r, c] = int(rng.choice(choices))
        if required is not None and required not in a:
            a[0, 0] = required
        return a
    a[:] = rng.choice(colors, size=shape, replace=True)
    if mode == "pair":
        r, c = int(rng.integers(rows)), int(rng.integers(cols))
        if rng.random() < 0.5:
            a[r, (c + 1) % cols] = a[r, c]
        else:
            a[(r + 1) % rows, c] = a[r, c]
    elif mode in ("run", "rowcol"):
        if rng.random() < 0.5:
            a[int(rng.integers(rows)), :] = int(rng.choice(colors))
        else:
            a[:, int(rng.integers(cols))] = int(rng.choice(colors))
    if required is not None and required not in a:
        a[int(rng.integers(rows)), int(rng.integers(cols))] = required
    if mode != "none" and not _adjacent(a):
        a[0, 1] = a[0, 0]
    return a


def _sizes(rng, count, target):
    profile = int(rng.choice(4, p=[.22, .24, .36, .18] if target else [.32, .31, .22, .15]))
    capacity = 30 - (count - 1)
    if profile == 3:
        size = int(rng.integers(2, min(9, capacity // count) + 1))
        return (size,) * count
    sizes = []
    if profile == 2:
        sizes.append(int(rng.integers(9, min(16, capacity - 2 * (count - 1)) + 1)))
    if profile == 0:
        base = int(rng.integers(3, min(9, capacity // count) + 1))
    while len(sizes) < count:
        room = capacity - sum(sizes) - 2 * (count - len(sizes) - 1)
        if profile == 0:
            value = max(2, min(room, base + int(rng.integers(-2, 3))))
        else:
            value = int(rng.integers(2, min(7 if profile == 2 else 14, room) + 1))
        sizes.append(value)
    rng.shuffle(sizes)
    return tuple(sizes)


def _mask(rng, h, w, dense_bias):
    probs = [.67, .12, .07, .06, .04, .04] if dense_bias else [.35, .20, .18, .12, .10, .05]
    style = int(rng.choice(6, p=probs))
    if style < 3:
        bounds = [(0.72, 0.97), (0.38, 0.72), (0.12, 0.35)][style]
        mask = rng.random((h, w)) < float(rng.uniform(*bounds))
    elif style == 3:
        rr, cc = np.indices((h, w))
        modulus = int(rng.integers(2, 5))
        mask = ((rr * int(rng.integers(1, 4)) + cc * int(rng.integers(1, 4)) + int(rng.integers(modulus))) % modulus) != 0
        mask &= rng.random((h, w)) >= float(rng.uniform(0, 0.22))
    elif style == 4:
        mask = np.zeros((h, w), dtype=bool)
        row, col = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(h * w // 2, h * w * 2 + 1))):
            mask[row, col] = True
            direction = int(rng.integers(4))
            row = min(h - 1, max(0, row + (direction == 0) - (direction == 1)))
            col = min(w - 1, max(0, col + (direction == 2) - (direction == 3)))
    else:
        mask = np.ones((h, w), dtype=bool)
    for row in range(h):
        if not np.any(mask[row]):
            mask[row, int(rng.integers(w))] = True
    for col in range(w):
        if not np.any(mask[:, col]):
            mask[int(rng.integers(h)), col] = True
    return mask


def _render(rng, labels, background, target):
    rows, cols = labels.shape
    heights, widths = _sizes(rng, rows, target), _sizes(rng, cols, target)
    grid = np.full((sum(heights) + rows - 1, sum(widths) + cols - 1), background, dtype=np.int8)
    dense_bias = bool(rng.random() < (0.52 if target else 0.45))
    top = 0
    for r, h in enumerate(heights):
        left = 0
        for c, w in enumerate(widths):
            block = grid[top:top + h, left:left + w]
            block[_mask(rng, h, w, dense_bias)] = labels[r, c]
            left += w + 1
        top += h + 1
    return grid


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _hash(a):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.int8, copy=False).tobytes()).hexdigest()


def _decode(grid):
    candidates = []
    for color in np.unique(grid):
        row_lines = np.flatnonzero(np.all(grid == color, axis=1))
        col_lines = np.flatnonzero(np.all(grid == color, axis=0))
        if len(row_lines) and len(col_lines):
            candidates.append((int(color), row_lines, col_lines))
    assert len(candidates) == 1
    background, row_lines, col_lines = candidates[0]
    assert 0 < row_lines[0] and row_lines[-1] < grid.shape[0] - 1
    assert 0 < col_lines[0] and col_lines[-1] < grid.shape[1] - 1
    assert np.all(np.diff(row_lines) >= 3) and np.all(np.diff(col_lines) >= 3)
    row_edges = [-1, *map(int, row_lines), grid.shape[0]]
    col_edges = [-1, *map(int, col_lines), grid.shape[1]]
    heights = tuple(b - a - 1 for a, b in zip(row_edges, row_edges[1:]))
    widths = tuple(b - a - 1 for a, b in zip(col_edges, col_edges[1:]))
    assert all(x >= 2 for x in heights + widths)
    labels = np.empty((len(heights), len(widths)), dtype=np.int8)
    for r, (top, bottom) in enumerate(zip(row_edges, row_edges[1:])):
        for c, (left, right) in enumerate(zip(col_edges, col_edges[1:])):
            block = grid[top + 1:bottom, left + 1:right]
            colors = np.unique(block[block != background])
            assert len(colors) == 1
            labels[r, c] = colors[0]
    return labels, background, heights, widths


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _counts(int(seed), num_train, num_test)
    core = rng.choice(np.arange(10), size=int(rng.integers(4, 8)), replace=False)
    repetition = int(rng.integers(4))
    train_shapes, test_shapes = _shapes(rng, n_train, n_test)
    backgrounds = _backgrounds(rng, n_train, n_train + n_test)
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=n_train + n_test, dtype=np.int64)
    pair_rngs = [np.random.default_rng(int(x)) for x in pair_seeds]
    puzzle, seen = {"train": [], "test": []}, set()
    for index, shape in enumerate(train_shapes):
        local, background = pair_rngs[index], backgrounds[index]
        colors = _colors(local, core, background)
        if repetition == 0:
            mode = "none"
        elif repetition == 1 and index == 0:
            mode = ["pair", "run", "rowcol"][int(local.integers(3))]
        else:
            mode = ["none", "pair", "run", "rowcol", "solid"][int(local.choice(5, p=[.28, .29, .17, .20, .06]))]
        if index == 0 and mode == "solid":
            mode = "none"
        labels = _labels(local, shape, colors, mode)
        for _ in range(8):
            grid = _render(local, labels, background, False)
            if _key(grid) not in seen and _hash(grid) not in _SOURCE_HASHES:
                break
        else:
            raise RuntimeError("failed to construct distinct training input")
        seen.add(_key(grid))
        puzzle["train"].append({"input": grid, "output": labels})
    trained = {int(v) for pair in puzzle["train"] for v in pair["output"].flat}
    for offset, shape in enumerate(test_shapes):
        index = n_train + offset
        local, background = pair_rngs[index], backgrounds[index]
        extra = None
        if local.random() < .38:
            choices = [x for x in range(10) if x != background and x not in trained]
            if choices:
                extra = int(local.choice(choices))
        colors = _colors(local, core, background, extra)
        if repetition == 0 or (repetition == 1 and offset == 0):
            mode = ["pair", "run", "rowcol", "solid"][int(local.choice(4, p=[.35, .24, .31, .10]))]
        elif repetition == 2:
            mode = "none"
        else:
            mode = ["none", "pair", "run", "rowcol", "solid"][int(local.choice(5, p=[.28, .27, .17, .20, .08]))]
        labels = _labels(local, shape, colors, mode, extra)
        for _ in range(8):
            grid = _render(local, labels, background, True)
            if _key(grid) not in seen and _hash(grid) not in _SOURCE_HASHES:
                break
        else:
            raise RuntimeError("failed to construct distinct test input")
        seen.add(_key(grid))
        puzzle["test"].append({"input": grid, "output": labels})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, train_shapes, train_backgrounds = set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = pair["input"], pair["output"]
            for grid in (source, target):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            key, pair_key = _key(source), (_key(source), _key(target))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _hash(source) not in _SOURCE_HASHES
            decoded, background, heights, widths = _decode(source)
            assert 2 <= len(heights) <= 4 and 2 <= len(widths) <= 4
            assert np.array_equal(decoded, target) and background not in target
            assert source.shape != target.shape
            if split == "train":
                train_shapes.add(target.shape)
                train_backgrounds.add(background)
    assert len(train_shapes) >= 2
    assert any(len(np.unique(pair["output"])) >= 2 for pair in puzzle["train"])
    all_backgrounds = {int(_decode(pair["input"])[1]) for split in ("train", "test") for pair in puzzle[split]}
    if len(all_backgrounds) > 1:
        assert len(train_backgrounds) > 1


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[x], q[x]) for p, q in zip(a[s], b[s]) for x in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None), (6, 3)):
        puzzle = generate_puzzle(103 + (train or 0) * 5 + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(400, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
    assert _same(generate_puzzle(71, 6, 3), generate_puzzle(71, 6, 3))
    train_counts, test_counts, shapes, layouts, backgrounds, fingerprints = set(), set(), set(), set(), set(), set()
    stats = dict(four=0, train33=0, familiar=0, tall=0, small=0, extra=0, variable_bg=0,
                 noadj_train_repeat_test=0, repeat_both=0, noadj_test=0, dense=0, sparse=0,
                 solid=0, mixed_masks=0, equal_bands=0, short_long=0, large_block=0,
                 zero_label=0, all_one=0, long_run=0, repeated_row=0, repeated_col=0,
                 palette_subset=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        train_shapes = {p["output"].shape for p in puzzle["train"]}
        trained = {int(v) for p in puzzle["train"] for v in p["output"].flat}
        stats["train33"] += (3, 3) in train_shapes
        train_repeat = any(_adjacent(p["output"]) for p in puzzle["train"])
        test_repeat = any(_adjacent(p["output"]) for p in puzzle["test"])
        stats["noadj_train_repeat_test"] += not train_repeat and test_repeat
        stats["repeat_both"] += train_repeat and test_repeat
        stats["noadj_test"] += not test_repeat
        local_backgrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                labels, background, heights, widths = _decode(source)
                assert np.array_equal(labels, target) and _hash(source) not in _SOURCE_HASHES
                local_backgrounds.add(background)
                backgrounds.add(background)
                shapes.add((source.shape, target.shape))
                layouts.add((target.shape, heights, widths))
                stats["four"] += 4 in target.shape
                stats["equal_bands"] += len(set(heights)) == 1 or len(set(widths)) == 1
                stats["short_long"] += min(heights + widths) <= 3 and max(heights + widths) >= 10
                stats["large_block"] += max(heights) >= 11 and max(widths) >= 11
                stats["zero_label"] += 0 in target
                stats["all_one"] += len(np.unique(target)) == 1
                stats["repeated_row"] += any(len(set(map(int, row))) == 1 for row in target)
                stats["repeated_col"] += any(len(set(map(int, col))) == 1 for col in target.T)
                stats["long_run"] += any(
                    len(set(map(int, line[i:i + 3]))) == 1
                    for line in list(target) + list(target.T)
                    for i in range(len(line) - 2)
                )
                if split == "test":
                    stats["familiar"] += target.shape in train_shapes
                    stats["tall"] += source.shape[0] > source.shape[1]
                    stats["small"] += source.shape[1] < 24
                    stats["extra"] += bool(set(map(int, target.flat)) - trained)
                    stats["palette_subset"] += bool(trained - set(map(int, target.flat)))
                row_edges = [-1, *map(int, np.flatnonzero(np.all(source == background, axis=1))), source.shape[0]]
                col_edges = [-1, *map(int, np.flatnonzero(np.all(source == background, axis=0))), source.shape[1]]
                densities = []
                for top, bottom in zip(row_edges, row_edges[1:]):
                    for left, right in zip(col_edges, col_edges[1:]):
                        block = source[top + 1:bottom, left + 1:right]
                        density = np.count_nonzero(block != background) / block.size
                        densities.append(density)
                        stats["dense"] += density >= .7
                        stats["sparse"] += density <= .4
                        stats["solid"] += density == 1
                stats["mixed_masks"] += any(d >= .7 for d in densities) and any(d <= .4 for d in densities)
        stats["variable_bg"] += len(local_backgrounds) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(shapes) >= 50 and len(layouts) >= 100
    assert len(backgrounds) >= 8 and stats["variable_bg"] >= 8
    assert stats["four"] >= 20 and stats["train33"] >= 8 and stats["familiar"] >= 8
    assert stats["tall"] >= 8 and stats["small"] >= 8 and stats["extra"] >= 8
    assert stats["noadj_train_repeat_test"] >= 3 and stats["repeat_both"] >= 5 and stats["noadj_test"] >= 3
    assert stats["dense"] >= 100 and stats["sparse"] >= 30 and stats["solid"] >= 5
    assert stats["mixed_masks"] >= 10 and stats["equal_bands"] >= 10
    assert stats["short_long"] >= 10 and stats["large_block"] >= 5
    assert stats["zero_label"] >= 5 and stats["all_one"] >= 5
    assert stats["long_run"] >= 10 and stats["repeated_row"] >= 10 and stats["repeated_col"] >= 10
    assert stats["palette_subset"] >= 10

    # A source-shaped canvas is valid when its actual contents are new.
    probe = generate_puzzle(0)
    background = _decode(probe["test"][0]["input"])[1]
    colors = [color for color in range(10) if color != background]
    labels = np.asarray(colors, dtype=np.int8).reshape(3, 3)
    grid = np.full((23, 28), background, dtype=np.int8)
    top = 0
    for row, height in enumerate((4, 14, 3)):
        left = 0
        for col, width in enumerate((7, 7, 12)):
            grid[top:top + height, left:left + width] = labels[row, col]
            grid[top, left] = background
            left += width + 1
        top += height + 1
    probe["test"][0] = {"input": grid, "output": labels}
    validate_generated_puzzle(probe)
