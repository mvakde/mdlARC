"""Whole-input color-mask tile puzzles."""

import numpy as np


_SOURCE_INPUTS = {
    ((6, 6, 6), (6, 8, 8), (8, 8, 6)),
    ((2, 4, 4), (3, 2, 4), (3, 3, 2)),
    ((1, 1, 1, 1, 1), (1, 8, 8, 8, 1), (1, 1, 1, 8, 1),
     (1, 8, 1, 8, 1), (1, 8, 1, 1, 1)),
    ((1, 1, 1, 1, 4), (1, 4, 4, 4, 4), (1, 4, 1, 1, 1),
     (2, 2, 2, 2, 1), (4, 1, 1, 2, 1)),
    ((3, 3, 3, 3), (4, 3, 3, 6), (4, 4, 6, 6), (4, 3, 3, 6)),
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    tr = _optional_count(num_train, "num_train", 2, 10)
    te = _optional_count(num_test, "num_test", 1, 4)
    if tr is None and te is None:
        te = 1 + seed % 4
        tr = te + 1 + (seed // 4) % (10 - te)
    elif tr is None:
        tr = te + 1 + seed % (10 - te)
    elif te is None:
        te = 1 + seed % min(4, tr - 1)
    if tr <= te:
        raise ValueError("num_train must exceed num_test")
    return tr, te


def _construct_output(grid, blank):
    h, w = grid.shape
    selector = np.repeat(np.repeat(grid, h, axis=0), w, axis=1)
    reference = np.tile(grid, (h, w))
    return np.where(selector == reference, selector, blank).astype(np.int8)


def _shape(rng, rectangular=False, larger=False):
    if rectangular:
        shapes = ((2, 3), (3, 2), (3, 4), (4, 3), (3, 5), (5, 3))
        return shapes[int(rng.integers(len(shapes)))]
    shapes = ((2, 2), (3, 3), (4, 4), (5, 5))
    weights = (0.10, 0.29, 0.33, 0.28) if not larger else (0, 0.32, 0.37, 0.31)
    return shapes[int(rng.choice(len(shapes), p=weights))]


def _category_count(rng, shape, limit, prefer_many=False, repeated=True):
    capacity = shape[0] * shape[1]
    maximum = min(limit, 6, capacity // 2 if repeated else capacity - 1)
    maximum = max(2, maximum)
    choices = np.arange(2, maximum + 1)
    if prefer_many and maximum >= 4:
        choices = choices[choices >= 4]
    elif maximum >= 4:
        weights = np.array([0.38 if k == 2 else 0.28 if k == 3 else 0.34 / (maximum - 3) for k in choices])
        weights /= weights.sum()
        return int(rng.choice(choices, p=weights))
    return int(rng.choice(choices))


def _raw_labels(rng, shape, count, style):
    h, w = shape
    rows, cols = np.indices(shape)
    if style == "scatter":
        concentration = float(rng.uniform(0.6, 2.8))
        frequencies = rng.dirichlet(np.full(count, concentration))
        labels = rng.choice(count, size=shape, p=frequencies)
    elif style == "cluster":
        centers = rng.random((count, 2)) * np.array([h - 1, w - 1])
        distance = (rows[None] - centers[:, 0, None, None]) ** 2 + (cols[None] - centers[:, 1, None, None]) ** 2
        labels = np.argmin(distance + rng.random(distance.shape) * 0.35, axis=0)
        noise = rng.random(shape) < rng.uniform(0.04, 0.20)
        labels[noise] = rng.integers(count, size=int(noise.sum()))
    elif style == "modular":
        a, b = int(rng.integers(count)), int(rng.integers(count))
        if a == 0 and b == 0:
            a = 1
        labels = (a * rows + b * cols + int(rng.integers(count))) % count
    elif style == "walk":
        labels = np.full(shape, int(rng.integers(count)), dtype=int)
        for color in range(count):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(int(rng.integers(2, h + w + 3))):
                labels[r, c] = color
                dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                r, c = int(np.clip(r + dr, 0, h - 1)), int(np.clip(c + dc, 0, w - 1))
    elif style == "bands":
        axis = int(rng.integers(2))
        position = rows if axis == 0 else cols
        band_width = int(rng.integers(1, 3))
        labels = (position // band_width + int(rng.integers(count))) % count
    elif style == "blocks":
        row_parts = int(rng.integers(2, min(h, 4) + 1))
        col_parts = int(rng.integers(2, min(w, 4) + 1))
        region = (rows * row_parts // h) * col_parts + cols * col_parts // w
        color_map = rng.integers(count, size=row_parts * col_parts)
        labels = color_map[region]
    elif style == "mirror":
        half = (w + 1) // 2
        if h * half < count:
            return None
        orbit_colors = np.concatenate((np.arange(count), rng.integers(count, size=h * half - count)))
        rng.shuffle(orbit_colors)
        half_grid = orbit_colors.reshape(h, half)
        labels = np.concatenate((half_grid, half_grid[:, :w // 2][:, ::-1]), axis=1)
    else:
        raise ValueError("unknown layout style")
    return np.asarray(labels, dtype=np.int64)


def _sample_labels(rng, shape, count, style, singleton=False, asymmetric=False, repeated=False):
    total = shape[0] * shape[1]
    for attempt in range(150):
        use_style = style if attempt < 75 else ("scatter", "cluster", "modular", "walk")[attempt % 4]
        labels = _raw_labels(rng, shape, count, use_style)
        if labels is None:
            continue
        if use_style != "mirror":
            flat = labels.reshape(-1)
            missing = [c for c in range(count) if not np.any(flat == c)]
            for color in missing:
                donor_counts = np.bincount(flat, minlength=count)
                donor = int(np.argmax(donor_counts))
                locations = np.flatnonzero(flat == donor)
                flat[int(rng.choice(locations))] = color
            labels = flat.reshape(shape)
        if singleton:
            counts = np.bincount(labels.reshape(-1), minlength=count)
            chosen = int(rng.choice(np.arange(count)))
            locations = np.argwhere(labels == chosen)
            if len(locations) > 1:
                keep = int(rng.integers(len(locations)))
                for i, (r, c) in enumerate(locations):
                    if i != keep:
                        labels[int(r), int(c)] = int(rng.choice([x for x in range(count) if x != chosen]))
        counts = np.bincount(labels.reshape(-1), minlength=count)
        if np.any(counts == 0) or (repeated and np.any(counts < 2)):
            continue
        if singleton and not np.any(counts == 1):
            continue
        if asymmetric and (np.array_equal(labels, np.fliplr(labels)) or np.array_equal(labels, np.flipud(labels)) or (shape[0] == shape[1] and (np.array_equal(labels, labels.T) or np.array_equal(labels, np.rot90(labels))))):
            continue
        return labels
    raise RuntimeError(f"cannot sample {shape} with {count} categories and {style} style")


def _pair(rng, shape, count, palette, blank, style, singleton=False, asymmetric=False, repeated=False):
    labels = _sample_labels(rng, shape, count, style, singleton, asymmetric, repeated)
    colors = np.asarray(rng.choice(palette, size=count, replace=False), dtype=np.int8)
    grid = colors[labels]
    return {"input": grid, "output": _construct_output(grid, blank)}


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.reshape(-1)))


def _is_source_input(grid):
    return tuple(tuple(int(x) for x in row) for row in grid) in _SOURCE_INPUTS


def _style(rng):
    styles = ("scatter", "cluster", "modular", "walk", "bands", "blocks", "mirror")
    return styles[int(rng.choice(len(styles), p=(0.19, 0.16, 0.15, 0.13, 0.14, 0.10, 0.13)))]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    blank = 0 if rng.random() < 0.30 else int(rng.integers(1, 10))
    available = np.array([c for c in range(10) if c != blank], dtype=np.int8)
    rectangular = rng.random() < 0.24
    novel = rng.random() < 0.24
    high = rng.random() < 0.44
    singleton = rng.random() < 0.34
    novel_colors = np.asarray(rng.choice(available, size=int(rng.integers(2, 4)), replace=False), dtype=np.int8) if novel else np.array([], dtype=np.int8)
    train_palette = np.array([c for c in available if c not in novel_colors], dtype=np.int8)
    puzzle = {"train": [], "test": []}
    used = set()

    def add(split, shape, count, palette, style, single=False, asym=False, repeat=False):
        for _ in range(300):
            pair = _pair(rng, shape, count, palette, blank, style, single, asym, repeat)
            key = _grid_key(pair["input"])
            if key not in used and not _is_source_input(pair["input"]):
                used.add(key)
                puzzle[split].append(pair)
                return
        raise RuntimeError("unable to sample a distinct input")

    anchor_shape = _shape(rng, larger=True)
    anchor_count = _category_count(rng, anchor_shape, min(3, len(train_palette)), repeated=True)
    add("train", anchor_shape, anchor_count, train_palette, _style(rng), asym=True, repeat=True)

    second_shape = _shape(rng, rectangular=rectangular)
    second_repeat = not singleton
    second_count = _category_count(rng, second_shape, len(train_palette), prefer_many=high, repeated=second_repeat)
    add("train", second_shape, second_count, train_palette, _style(rng), single=singleton, repeat=second_repeat)

    for _ in range(2, train_count):
        shape = _shape(rng, rectangular=rng.random() < 0.14)
        repeat = rng.random() < 0.70
        count = _category_count(rng, shape, len(train_palette), repeated=repeat)
        add("train", shape, count, train_palette, _style(rng), single=not repeat and rng.random() < 0.45, repeat=repeat)

    train_pairs = puzzle["train"]
    max_train_categories = max(len(np.unique(p["input"])) for p in train_pairs)
    demonstrated_singleton = any(np.any(np.unique(p["input"], return_counts=True)[1] == 1) for p in train_pairs)
    demonstrated_rectangle = any(p["input"].shape[0] != p["input"].shape[1] for p in train_pairs)

    for index in range(test_count):
        test_rect = (index == 0 and rectangular) or (demonstrated_rectangle and rng.random() < 0.17)
        shape = _shape(rng, rectangular=test_rect)
        repeat = not demonstrated_singleton or rng.random() < 0.58
        palette = novel_colors if index == 0 and novel else available
        if index == 0 and novel:
            count = min(len(novel_colors), shape[0] * shape[1] // 2) if repeat else len(novel_colors)
        else:
            count = _category_count(rng, shape, min(max_train_categories, len(palette)), prefer_many=max_train_categories >= 4 and rng.random() < 0.43, repeated=repeat)
        add("test", shape, count, palette, _style(rng), single=demonstrated_singleton and not repeat and rng.random() < 0.70, repeat=repeat)

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle container")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "invalid pair counts")
    seen = set()
    blank = None
    training_singleton = False
    training_rectangle = False
    training_high = False
    orientation_anchor = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                _require(isinstance(array, np.ndarray) and array.ndim == 2, "grids must be 2D arrays")
                _require(np.issubdtype(array.dtype, np.integer) and np.all((0 <= array) & (array <= 9)), "invalid ARC color")
                _require(all(1 <= side <= 30 for side in array.shape), "grid out of bounds")
            h, w = grid.shape
            _require(output.shape == (h * h, w * w), "incorrect tile expansion dimensions")
            colors, counts = np.unique(grid, return_counts=True)
            _require(len(colors) >= 2 and len(colors) <= 9, "at least two input categories required")
            additions = set(int(x) for x in np.unique(output)) - set(int(x) for x in colors)
            _require(len(additions) == 1, "output must introduce one mismatch color")
            pair_blank = additions.pop()
            _require(pair_blank not in colors, "blank occurs in input")
            if blank is None:
                blank = pair_blank
            _require(pair_blank == blank, "blank must be stable across puzzle")
            _require(set(int(x) for x in np.unique(output)) == set(int(x) for x in colors) | {blank}, "output colors are incorrect")
            # An independent scalar reconstruction checks every tile position.
            for r in range(h):
                for c in range(w):
                    for u in range(h):
                        for v in range(w):
                            expected = int(grid[r, c]) if grid[r, c] == grid[u, v] else blank
                            _require(int(output[r * h + u, c * w + v]) == expected, "incorrect equality-mask tile")
            key = _grid_key(grid)
            _require(key not in seen and not _is_source_input(grid), "duplicate or source input")
            seen.add(key)
            if split == "train":
                training_singleton |= bool(np.any(counts == 1))
                training_rectangle |= h != w
                training_high |= len(colors) >= 4
                orientation_anchor |= bool(np.all(counts >= 2) and h == w and h >= 3 and not np.array_equal(grid, np.fliplr(grid)) and not np.array_equal(grid, np.flipud(grid)) and not np.array_equal(grid, grid.T))
            else:
                _require(not np.any(counts == 1) or training_singleton, "test singleton lacks training demonstration")
                _require(h == w or training_rectangle, "rectangular test lacks training demonstration")
                _require(len(colors) < 4 or training_high, "high-category test lacks training demonstration")
    _require(orientation_anchor, "training needs an asymmetric repeated-color anchor")


def self_test() -> None:
    seen_blanks, sizes, rectangles, high, singletons, mirrors, novel = set(), set(), 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert len(puzzle["train"]) > len(puzzle["test"])
        assert all(np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"]) for split in ("train", "test") for a, b in zip(puzzle[split], generate_puzzle(seed)[split]))
        train_colors = set(int(x) for p in puzzle["train"] for x in np.unique(p["input"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                sizes.add((split, grid.shape))
                rectangles += grid.shape[0] != grid.shape[1]
                high += len(np.unique(grid)) >= 4
                singletons += np.any(np.unique(grid, return_counts=True)[1] == 1)
                mirrors += np.array_equal(grid, np.fliplr(grid))
                seen_blanks.add((set(int(x) for x in np.unique(pair["output"])) - set(int(x) for x in np.unique(grid))).pop())
        novel += any(set(int(x) for x in np.unique(p["input"])).isdisjoint(train_colors) for p in puzzle["test"])
    assert len(seen_blanks) >= 5 and 0 in seen_blanks
    assert rectangles and high and singletons and mirrors and novel
    assert all((split, (n, n)) in sizes for split in ("train", "test") for n in (3, 4, 5))
    for tr, te in ((2, 1), (10, 4), (7, 3)):
        p = generate_puzzle(1000 + tr + te, tr, te)
        assert len(p["train"]) == tr and len(p["test"]) == te
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
