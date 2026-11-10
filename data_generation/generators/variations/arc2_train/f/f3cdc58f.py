import numpy as np


_SOURCE_INPUTS = {
    ("0000040330", "0130003000", "0000100104", "3000200020", "0000000000", "0010000000", "0020000200", "0300043200", "0001000030", "0040040101"),
    ("0000000000", "0000000040", "2000030141", "0000001000", "1400000001", "0000200020", "0040000000", "1004040030", "0000000000", "0121000000"),
    ("0000040000", "0000000000", "0030000030", "0100200400", "0000000000", "0004030020", "0200000000", "0000100040", "0040000000", "0000000000"),
    ("0300000000", "0230024000", "0302000003", "4001000100", "0001000200", "3010300000", "0000000300", "0000000000", "1000001003", "0000240242"),
}
_MODES = ("scatter", "same_clumps", "mixed_clump", "row", "edge", "spread")


def _count_arg(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(value)


def _pair_numbers(rng, num_train, num_test):
    t = _count_arg("num_train", num_train, 2, 10)
    s = _count_arg("num_test", num_test, 1, 4)
    if t is None and s is None:
        s = int(rng.integers(1, 5))
        t = int(rng.integers(max(2, s + 1), 11))
    elif t is None:
        t = int(rng.integers(max(2, s + 1), 11))
    elif s is None:
        s = int(rng.integers(1, min(4, t - 1) + 1))
    if t <= s:
        raise ValueError("num_train must exceed num_test")
    return t, s


def _signature(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _source_input(grid):
    return grid.shape == (10, 10) and tuple("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUTS


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)
    assert len(order) >= 2 and counts[order[-1]] > counts[order[-2]]
    return int(colors[order[-1]])


def _counts(grid, colors):
    return tuple(int(np.count_nonzero(grid == color)) for color in colors)


def _output(shape, background, colors, counts):
    h, w = shape
    result = np.full((h, w), background, dtype=np.int8)
    for column, (color, count) in enumerate(zip(colors, counts)):
        if count:
            result[h - count:, column] = color
    return result


def _dimensions(rng, source_like):
    if source_like:
        return (10, 10)
    return int(rng.integers(5, 21)), int(rng.integers(6, 22))


def _sample_counts(rng, height, profile, positive):
    low = 1 if positive else 0
    if profile == "sparse":
        counts = rng.integers(low, min(height, 4) + 1, size=4)
    elif profile == "dense":
        counts = rng.integers(max(low, height // 2), height + 1, size=4)
    elif profile == "full":
        counts = rng.integers(low, height + 1, size=4)
        counts[int(rng.integers(0, 4))] = height
    elif profile == "top_tie":
        counts = rng.integers(low, height + 1, size=4)
        top = int(rng.integers(max(2, height // 2), height + 1))
        slots = rng.choice(4, 2, replace=False)
        counts[slots] = top
        for j in range(4):
            if j not in slots:
                counts[j] = min(int(counts[j]), top - 1)
    elif profile == "multiple_ties":
        values = rng.integers(low, height + 1, size=2)
        counts = np.array([values[0], values[0], values[1], values[1]])
        rng.shuffle(counts)
    else:
        counts = rng.integers(low, height + 1, size=4)
    if positive:
        counts = np.maximum(counts, 1)
    elif not np.any(counts):
        counts[int(rng.integers(0, 4))] = 1
    return tuple(int(v) for v in counts)


def _placement(rng, shape, counts, colors, background, mode):
    h, w = shape
    rr, cc = np.indices(shape)
    coords = np.stack((rr.ravel(), cc.ravel()), axis=1)
    n = sum(counts)
    result = np.full(shape, background, dtype=np.int8)
    labels = np.concatenate([np.full(k, color, dtype=np.int8) for color, k in zip(colors, counts)])
    rng.shuffle(labels)
    jitter = rng.random(h * w)
    if mode == "scatter":
        order = rng.permutation(h * w)
    elif mode == "mixed_clump":
        center = (int(rng.integers(h)), int(rng.integers(w)))
        distance = np.abs(coords[:, 0] - center[0]) + np.abs(coords[:, 1] - center[1])
        order = np.argsort(distance + jitter * 0.8)
    elif mode == "row":
        row = int(rng.integers(h))
        order = np.argsort(np.abs(coords[:, 0] - row) + jitter * 0.8)
    elif mode == "edge":
        side = int(rng.integers(4))
        distance = (coords[:, 0], h - 1 - coords[:, 0], coords[:, 1], w - 1 - coords[:, 1])[side]
        order = np.argsort(distance + jitter * 0.8)
    elif mode == "spread":
        phase = int(rng.integers(2))
        chess = (coords[:, 0] + coords[:, 1] + phase) % 2
        order = np.argsort(chess + jitter * 0.8)
    elif mode == "same_clumps":
        available = np.ones(h * w, dtype=bool)
        for color, k in zip(colors, counts):
            center = (int(rng.integers(h)), int(rng.integers(w)))
            distance = np.abs(coords[:, 0] - center[0]) + np.abs(coords[:, 1] - center[1])
            ranks = np.argsort(distance + jitter * 0.8)
            chosen = ranks[available[ranks]][:k]
            result.flat[chosen] = color
            available[chosen] = False
        return result
    else:
        raise AssertionError("unknown placement mode")
    result.flat[order[:n]] = labels
    return result


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_n, test_n = _pair_numbers(rng, num_train, num_test)
    source_like = bool(rng.integers(0, 6) == 0)
    if source_like:
        colors = (1, 2, 3, 4)
    else:
        colors = tuple(int(x) for x in rng.choice(np.arange(1, 10), 4, replace=False))
        if rng.random() < 0.45:
            colors = tuple(sorted(colors))
    backgrounds = tuple(int(c) for c in range(10) if c not in colors)
    puzzle = {"train": [], "test": []}
    used = set()
    records = []
    # These are sampling arrangements, not output conventions. A repeat gives
    # different spatial evidence with an exactly shared histogram.
    repeat = bool(rng.random() < 0.42)
    similar = bool(rng.random() < 0.5)
    similar_mode = _MODES[int(rng.integers(len(_MODES)))]
    for index in range(train_n + test_n):
        training = index < train_n
        local = index if training else index - train_n
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        repeat_from = 0 if repeat and ((training and index == 2) or (not training and train_n == 2 and local == 0)) else None
        if repeat_from is not None:
            shape, background, counts, _, _ = records[repeat_from]
        else:
            shape = records[0][0] if similar and index == 1 else _dimensions(pair_rng, source_like)
            background = 0 if source_like else (records[0][1] if similar and index == 1 else int(pair_rng.choice(backgrounds)))
            profiles = ("varied", "sparse", "dense", "full", "top_tie", "multiple_ties")
            if training:
                profile = profiles[int(pair_rng.integers(len(profiles)))]
            else:
                profile = profiles[int(pair_rng.choice(len(profiles), p=(0.12, 0.10, 0.20, 0.20, 0.20, 0.18)))]
            counts = _sample_counts(pair_rng, shape[0], profile, positive=index < 2)
            if index == 1:
                # Two different training count profiles expose that output
                # heights change while the color slots stay put.
                first = records[0][2]
                for _ in range(12):
                    if counts != first and any(a != b for a, b in zip(counts, first)):
                        break
                    counts = _sample_counts(pair_rng, shape[0], profile, True)
                if counts == first:
                    temp = list(counts)
                    temp[0] = 1 if temp[0] != 1 else shape[0]
                    counts = tuple(temp)
        mode = similar_mode if similar and index in (0, 1) else _MODES[int(pair_rng.integers(len(_MODES)))]
        if repeat_from is not None:
            first_mode = records[repeat_from][3]
            if mode == first_mode:
                mode = _MODES[(_MODES.index(mode) + 2) % len(_MODES)]
        output = _output(shape, background, colors, counts)
        for attempt in range(24):
            inp = _placement(pair_rng, shape, counts, colors, background, mode)
            sig = _signature(inp)
            if sig not in used and not np.array_equal(inp, output) and not _source_input(inp):
                break
            mode = _MODES[(attempt + _MODES.index(mode) + 1) % len(_MODES)]
        else:
            raise AssertionError("could not construct unique input")
        used.add(sig)
        records.append((shape, background, counts, mode, sig))
        puzzle["train" if training else "test"].append({"input": inp, "output": output})
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    first = puzzle["train"][0]
    assert isinstance(first, dict) and set(first) == {"input", "output"}
    _validate_grid(first["input"])
    _validate_grid(first["output"])
    first_bg = _background(first["input"])
    colors = tuple(int(v) for v in first["output"][-1, :4])
    assert len(set(colors)) == 4 and first_bg not in colors
    seen_inputs, seen_pairs = set(), set()
    train_counts = []
    for split in ("train", "test"):
        for i, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            assert inp.shape == out.shape and inp.shape[0] >= 5 and inp.shape[1] >= 6
            bg = _background(inp)
            assert bg not in colors
            assert set(int(v) for v in np.unique(inp)) <= {bg, *colors}
            if split == "train" and i < 2:
                assert all(np.any(inp == c) for c in colors)
            counts = _counts(inp, colors)
            assert all(0 <= c <= inp.shape[0] for c in counts)
            assert np.array_equal(out, _output(inp.shape, bg, colors, counts))
            assert not np.array_equal(inp, out) and not _source_input(inp)
            signature = _signature(inp)
            pair_signature = (signature, _signature(out))
            assert signature not in seen_inputs and pair_signature not in seen_pairs
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            if split == "train":
                train_counts.append(counts)
    assert train_counts[0] != train_counts[1]
    # The first training output establishes all four slots, even if a
    # later example has a zero-height bar.
    assert all(first["output"][-1, j] == colors[j] for j in range(4))


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    p = generate_puzzle(314159)
    assert _equal_puzzles(p, generate_puzzle(314159))
    assert not _equal_puzzles(p, generate_puzzle(314160))
    for seed, t, s in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, None, 1), (105, None, 4), (106, 2, None), (107, 10, None)):
        puzzle = generate_puzzle(seed, t, s)
        assert t is None or len(puzzle["train"]) == t
        assert s is None or len(puzzle["test"]) == s
        validate_generated_puzzle(puzzle)
    for t, s in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 4), (True, 1), (3.0, 1)):
        try:
            generate_puzzle(0, t, s)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    train_ns, test_ns, sizes, palettes, vectors, modes = set(), set(), set(), set(), set(), set()
    tops = set()
    tied_top = singleton = full = zero = repeated = 0
    same_output = same_canvas = varied_dimensions = varied_backgrounds = source_like_puzzles = 0
    same_touch = mixed_touch = row_focus = edge_focus = isolated_sparse = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_ns.add(len(puzzle["train"]))
        test_ns.add(len(puzzle["test"]))
        local = {}
        puzzle_dimensions, puzzle_backgrounds = set(), set()
        first_two = puzzle["train"][:2]
        if first_two[0]["input"].shape == first_two[1]["input"].shape and _background(first_two[0]["input"]) == _background(first_two[1]["input"]):
            same_canvas += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg = _background(inp)
                # A missing bottom cell identifies a zero bar; the first
                # training output supplies the complete ordered palette.
                colors = tuple(int(x) for x in puzzle["train"][0]["output"][-1, :4])
                counts = _counts(inp, colors)
                independent = np.full(inp.shape, bg, dtype=np.int8)
                for col, color in enumerate(colors):
                    independent[inp.shape[0] - counts[col]:, col] = color
                assert np.array_equal(independent, out)
                sizes.add(inp.shape)
                puzzle_dimensions.add(inp.shape)
                puzzle_backgrounds.add(bg)
                palettes.add((bg, colors))
                vectors.add(counts)
                modes.add(tuple(int(np.count_nonzero(row != bg)) for row in inp))
                if counts in local:
                    repeated += 1
                    same_output += np.array_equal(local[counts], out)
                local[counts] = out
                foreground = inp != bg
                n = int(np.count_nonzero(foreground))
                same_adjacent = mixed_adjacent = 0
                for left, right in ((inp[:, :-1], inp[:, 1:]), (inp[:-1, :], inp[1:, :])):
                    touching = (left != bg) & (right != bg)
                    same_adjacent += int(np.count_nonzero(touching & (left == right)))
                    mixed_adjacent += int(np.count_nonzero(touching & (left != right)))
                same_touch += same_adjacent > 0
                mixed_touch += mixed_adjacent > 0
                row_focus += n > 0 and int(np.max(np.sum(foreground, axis=1))) >= max(3, n * 0.55)
                edge_focus += n > 0 and max(int(np.sum(foreground[0])), int(np.sum(foreground[-1])), int(np.sum(foreground[:, 0])), int(np.sum(foreground[:, -1]))) >= max(3, n * 0.45)
                isolated_sparse += n <= 10 and same_adjacent == 0 and mixed_adjacent == 0
                if split == "train":
                    tops.update(i for i, c in enumerate(counts) if c == max(counts))
                    tied_top += counts.count(max(counts)) >= 2
                    singleton += 1 in counts
                    full += inp.shape[0] in counts
                    zero += 0 in counts
        varied_dimensions += len(puzzle_dimensions) > 1
        varied_backgrounds += len(puzzle_backgrounds) > 1
        source_like_puzzles += puzzle_dimensions == {(10, 10)} and puzzle_backgrounds == {0} and colors == (1, 2, 3, 4)
    assert len(train_ns) >= 4 and len(test_ns) >= 3
    assert len(sizes) >= 35 and len(palettes) >= 15 and len(vectors) >= 100 and len(modes) >= 80
    assert tops == {0, 1, 2, 3} and tied_top >= 10 and singleton >= 10 and full >= 10 and zero >= 5 and repeated >= 5
    assert same_output >= 5 and same_canvas >= 8
    assert varied_dimensions >= 25 and varied_backgrounds >= 25 and source_like_puzzles >= 2
    assert same_touch >= 100 and mixed_touch >= 100 and row_focus >= 20 and edge_focus >= 20 and isolated_sparse >= 10
