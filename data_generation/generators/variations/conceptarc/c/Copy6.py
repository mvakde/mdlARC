"""Whole-grid duplication along the shorter input dimension."""

import numpy as np


_SOURCE_INPUTS = (
    np.array([[2, 2, 2, 0, 3, 3, 3, 0, 4, 4, 4],
              [2, 0, 2, 0, 3, 0, 3, 0, 4, 0, 4],
              [2, 2, 2, 0, 3, 3, 3, 0, 4, 4, 4]], dtype=np.int8),
    np.array([[6, 6, 6, 6], [6, 6, 6, 6], [0, 0, 0, 0],
              [7, 7, 7, 7], [7, 7, 7, 7], [0, 0, 0, 0],
              [0, 0, 0, 0], [9, 9, 9, 9], [9, 9, 9, 9]], dtype=np.int8),
    np.array([[6, 6, 6, 6, 0, 1, 1, 1, 1],
              [6, 4, 4, 6, 0, 1, 3, 3, 1],
              [6, 4, 4, 6, 0, 1, 3, 3, 1],
              [6, 6, 6, 6, 0, 1, 1, 1, 1]], dtype=np.int8),
    np.array([[0, 7, 0], [7, 1, 7], [0, 7, 0], [0, 0, 0],
              [0, 5, 0], [5, 6, 5], [0, 5, 0], [0, 0, 0],
              [0, 1, 0], [1, 3, 1], [0, 1, 0]], dtype=np.int8),
    np.array([[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [0, 0, 0, 0, 0],
              [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [0, 0, 0, 0, 0],
              [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]], dtype=np.int8),
)


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError("Illegal pair count")
    legal = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
    choices = [(a, b) for a, b in legal
               if (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not choices:
        raise ValueError("Impossible pair counts")
    return choices[int(rng.integers(len(choices)))]


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)
    if len(order) < 2 or counts[order[-1]] == counts[order[-2]]:
        return None
    return int(colors[order[-1]])


def _dimensions(rng, orientation, regime):
    short = int(rng.integers(3, 11))
    # Every output aspect outcome occurs in both splits and both directions.
    weights = (0.43, 0.23, 0.34) if regime == "source_train" else (0.43, 0.27, 0.30)
    outcome = str(rng.choice(("cross", "square", "remain"), p=weights))
    if outcome == "square":
        long = 2 * short + 1
    elif outcome == "remain":
        long = int(rng.integers(2 * short + 2, min(30, 3 * short + 7) + 1))
    elif rng.random() < 0.48:
        long = short + 1
    else:
        long = int(rng.integers(short + 1, 2 * short + 1))
    return (short, long) if orientation == "wide" else (long, short)


def _motif(rng, height, width, primary, accent, background, style):
    block = np.full((height, width), primary, dtype=np.int8)
    if style == "frame" and min(height, width) >= 3:
        block[1:-1, 1:-1] = background
        if rng.random() < 0.55:
            block[int(rng.integers(1, height - 1)),
                  int(rng.integers(1, width - 1))] = accent
    elif style == "inset":
        if min(height, width) >= 3:
            block[1:-1, 1:-1] = accent
        else:
            block[height // 2, width // 2] = accent
    elif style == "stripes":
        if height >= width:
            block[1::2, :] = accent
        else:
            block[:, 1::2] = accent
    elif style == "checker":
        block[(np.indices(block.shape).sum(axis=0) % 2) == 0] = accent
    elif style == "cross":
        block[:] = background
        block[height // 2, :] = primary
        block[:, width // 2] = primary
        if min(height, width) >= 3:
            block[height // 2, width // 2] = accent
    elif style == "elbow":
        block[:] = background
        block[-1, :] = primary
        block[:, 0] = primary
        if min(height, width) >= 3:
            block[-1, -1] = accent
    elif style == "irregular":
        block[rng.random(block.shape) < 0.36] = background
        block[int(rng.integers(height)), int(rng.integers(width))] = accent
    if style in ("cross", "elbow", "irregular"):
        if rng.random() < 0.5:
            block = np.flip(block, axis=0)
        if rng.random() < 0.5:
            block = np.flip(block, axis=1)
    return block


_STYLES = ("solid", "frame", "inset", "stripes", "checker", "cross", "elbow", "irregular")


def _style(rng, regime):
    if regime == "source_train":
        weights = (0.25, 0.20, 0.15, 0.10, 0.07, 0.08, 0.08, 0.07)
    else:
        weights = (0.14, 0.15, 0.12, 0.14, 0.10, 0.11, 0.11, 0.13)
    return str(rng.choice(_STYLES, p=weights))


def _paint(canvas, rng, top, left, height, width, palette, background, regime):
    primary = int(rng.choice(palette))
    accent = int(rng.choice((*palette, background)))
    patch = _motif(rng, height, width, primary, accent, background,
                   _style(rng, regime))
    canvas[top:top + height, left:left + width] = patch


def _draw_bands(canvas, rng, palette, background, regime):
    height, width = canvas.shape
    wide = height < width
    short, long = (height, width) if wide else (width, height)
    count = int(rng.integers(2, min(5, (long + 1) // 3) + 1))
    # Positive internal gaps; freely distributed leading/trailing padding.
    extra = long - (2 * count - 1)
    allocation = rng.multinomial(extra, np.ones(2 * count + 1) / (2 * count + 1))
    leading = int(allocation[0])
    lengths = [1 + int(allocation[2 * i + 1]) for i in range(count)]
    gaps = [1 + int(allocation[2 * i + 2]) for i in range(count - 1)]
    pos = leading
    for i, length in enumerate(lengths):
        if wide:
            _paint(canvas, rng, 0, pos, short, length, palette, background, regime)
        else:
            _paint(canvas, rng, pos, 0, length, short, palette, background, regime)
        pos += length
        if i < count - 1:
            pos += gaps[i]


def _draw_scattered(canvas, rng, palette, background, regime, single):
    height, width = canvas.shape
    count = 1 if single else int(rng.integers(2, min(10, height * width // 5) + 1))
    for i in range(count):
        if single:
            obj_h = int(rng.integers(2, max(3, (3 * height) // 4 + 1)))
            obj_w = int(rng.integers(2, max(3, (3 * width) // 4 + 1)))
        else:
            obj_h = int(rng.integers(1, min(5, height) + 1))
            obj_w = int(rng.integers(1, min(5, width) + 1))
        obj_h = min(obj_h, height)
        obj_w = min(obj_w, width)
        top = int(rng.integers(height - obj_h + 1))
        left = int(rng.integers(width - obj_w + 1))
        if i == 0 and not single and rng.random() < 0.50:
            if rng.random() < 0.5:
                top = 0 if rng.random() < 0.5 else height - obj_h
            else:
                left = 0 if rng.random() < 0.5 else width - obj_w
        _paint(canvas, rng, top, left, obj_h, obj_w, palette, background, regime)


def _make_input(rng, orientation, regime, background, base_background, color_pool):
    height, width = _dimensions(rng, orientation, regime)
    available = [color for color in color_pool if color != background]
    if base_background != background:
        available.append(base_background)
    palette_size = int(rng.choice((1, 2, 3, 4, 5, 6),
                                  p=(0.27, 0.29, 0.17, 0.12, 0.09, 0.06)))
    palette_size = min(palette_size, len(available))
    palette = tuple(int(c) for c in rng.choice(available, size=palette_size, replace=False))
    canvas = np.full((height, width), background, dtype=np.int8)
    mode = str(rng.choice(("bands", "single", "scattered"),
                          p=(0.27, 0.20, 0.53) if regime == "source_train"
                          else (0.19, 0.17, 0.64)))
    if mode == "bands" and max(height, width) >= 8:
        _draw_bands(canvas, rng, palette, background, regime)
    else:
        _draw_scattered(canvas, rng, palette, background, regime, mode == "single")
    return canvas, background


def _transform(grid, background):
    height, width = grid.shape
    if height < width:
        line = np.full((1, width), background, dtype=np.int8)
        return np.concatenate((grid, line, grid), axis=0)
    line = np.full((height, 1), background, dtype=np.int8)
    return np.concatenate((grid, line, grid), axis=1)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    base_background = int(rng.integers(10))
    color_pool = tuple(int(c) for c in rng.permutation([c for c in range(10)
                                                        if c != base_background]))
    background_options = (base_background,)
    if rng.random() < 0.36:
        background_options = (base_background, color_pool[0])
    if rng.random() < 0.38:
        orientation = str(rng.choice(("wide", "tall")))
        train_directions = [orientation] * train_count
        test_directions = [orientation] * test_count
    else:
        train_directions = ["wide", "tall"]
        train_directions.extend(str(rng.choice(("wide", "tall")))
                                for _ in range(train_count - 2))
        test_directions = [str(rng.choice(("wide", "tall")))
                           for _ in range(test_count)]
        if test_count >= 2 and rng.random() < 0.55:
            test_directions[:2] = ["wide", "tall"]
        rng.shuffle(train_directions)
        rng.shuffle(test_directions)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_backgrounds = list(background_options)
    train_backgrounds.extend(int(rng.choice(background_options))
                             for _ in range(train_count - len(train_backgrounds)))
    rng.shuffle(train_backgrounds)
    test_backgrounds = [int(rng.choice(background_options))
                        for _ in range(test_count)]
    for split, directions in (("train", train_directions),
                              ("test", test_directions)):
        regime = "source_train" if split == "train" else "target_test"
        backgrounds = train_backgrounds if split == "train" else test_backgrounds
        for direction, background in zip(directions, backgrounds):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for _ in range(120):
                grid, _ = _make_input(pair_rng, direction, regime, background,
                                     base_background, color_pool)
                candidate = _key(grid)
                if (candidate not in seen and _background(grid) == background
                    and np.count_nonzero(grid != background) >= 3
                    and all(candidate != _key(source) for source in _SOURCE_INPUTS)):
                    break
            else:
                raise RuntimeError("Could not construct a distinct, readable grid")
            seen.add(candidate)
            puzzle[split].append({"input": grid,
                                  "output": _transform(grid, background)})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4
    assert train_count > test_count
    seen_inputs = set()
    seen_pairs = set()
    directions = {"train": set(), "test": set()}
    backgrounds = {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            height, width = inp.shape
            assert height != width
            background = _background(inp)
            assert background is not None
            assert np.count_nonzero(inp != background) >= 3
            backgrounds[split].add(background)
            directions[split].add("wide" if height < width else "tall")
            if height < width:
                assert out.shape == (2 * height + 1, width)
                assert np.array_equal(out[:height, :], inp)
                assert np.all(out[height, :] == background)
                assert np.array_equal(out[height + 1:, :], inp)
            else:
                assert out.shape == (height, 2 * width + 1)
                assert np.array_equal(out[:, :width], inp)
                assert np.all(out[:, width] == background)
                assert np.array_equal(out[:, width + 1:], inp)
            key = _key(inp)
            pair_key = (key, _key(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            assert all(key != _key(source) for source in _SOURCE_INPUTS)
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
    assert directions["test"] <= directions["train"]
    assert backgrounds["test"] <= backgrounds["train"]


def _signature(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    signatures = set()
    profiles = set()
    train_counts = set()
    test_counts = set()
    backgrounds = set()
    train_foreground_counts = set()
    split_outcomes = {"train": set(), "test": set()}
    split_directions = {"train": set(), "test": set()}
    same_direction_puzzles = 0
    changing_background_puzzles = 0
    near_square = {"train": 0, "test": 0}
    tall_square = {"train": 0, "test": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_backgrounds = set()
        pair_directions = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                h, w = inp.shape
                background = _background(inp)
                pair_backgrounds.add(background)
                backgrounds.add(background)
                pair_directions.add("wide" if h < w else "tall")
                split_directions[split].add("wide" if h < w else "tall")
                if split == "train":
                    train_foreground_counts.add(len(np.unique(inp)) - 1)
                outcome = "square" if out.shape[0] == out.shape[1] else (
                    "cross" if (h < w) != (out.shape[0] < out.shape[1]) else "remain")
                split_outcomes[split].add(outcome)
                near_square[split] += abs(h - w) == 1
                tall_square[split] += h > w and out.shape[0] == out.shape[1]
                profiles.add((inp.shape, out.shape, len(np.unique(inp)),
                              int(np.any(inp[0] != background)),
                              int(np.any(inp[-1] != background))))
                if h < w:
                    assert np.array_equal(out, np.vstack((inp,
                        np.full((1, w), background, dtype=inp.dtype), inp)))
                else:
                    assert np.array_equal(out, np.hstack((inp,
                        np.full((h, 1), background, dtype=inp.dtype), inp)))
        same_direction_puzzles += len(pair_directions) == 1
        changing_background_puzzles += len(pair_backgrounds) > 1
    assert len(signatures) >= 40 and len(profiles) >= 80
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) >= 7
    assert {1, 2, 3} <= train_foreground_counts
    assert all(values == {"wide", "tall"} for values in split_directions.values())
    assert all(values == {"cross", "square", "remain"}
               for values in split_outcomes.values())
    assert same_direction_puzzles >= 4 and changing_background_puzzles >= 4
    assert all(value >= 3 for value in near_square.values())
    assert all(value >= 2 for value in tall_square.values())
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (104, None, 1), (105, None, 4),
                              (106, 2, None), (107, 10, None), (108, 7, 3)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert _signature(puzzle) == _signature(generate_puzzle(seed, train, test))
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 2, "num_test": 3},
                      {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts were accepted")
