"""Generate ARC tasks that count distinct nonzero input colors."""

import numpy as np


# Exact source grids, including dimensions; source dimensions themselves are valid.
SOURCE_ROWS = (
    ("000000000000", "000000000088", "011111000088", "011111000088",
     "000000000088", "000000022000", "002220022000", "002220022000",
     "000000022000", "011110022000", "011110022000", "000000022000"),
    ("00000000000000", "00044444400000", "00044444400000",
     "44400000000000", "44400000000000", "44400000000000",
     "44400444444440", "44400444444440", "44400444444440",
     "44400000000000"),
    ("0000", "7474", "7474", "7474", "7474", "0000"),
    ("000000006666", "000660006666", "000660006666", "440660000000",
     "440660000000", "000660066666", "000000066666", "000000000000",
     "088888800000", "088888800000", "000000000000", "000000000000",
     "033330333330", "033330333330"),
    ("000000", "033330", "033330", "033330", "033330", "000000"),
    ("666666", "111111", "444444", "666666"),
)


def _key(grid):
    return grid.shape, grid.tobytes()


SOURCE_KEYS = frozenset(
    ((len(rows), len(rows[0])), bytes(int(c) for row in rows for c in row))
    for rows in SOURCE_ROWS
)


def _checked_count(value, low, high, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _pair_counts(seed, rng, num_train, num_test):
    nt = _checked_count(num_train, 2, 10, "num_train")
    ns = _checked_count(num_test, 1, 4, "num_test")
    if nt is None and ns is None:
        nt = 2 + seed % 9
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    elif nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _dimensions(rng, minimum=3):
    if minimum <= 4 and rng.random() < 0.10:
        return tuple(int(x) for x in ((12, 12), (10, 14), (6, 4),
                                      (14, 12), (6, 6), (4, 6))[int(rng.integers(6))])
    scale = int(rng.choice(3, p=(0.30, 0.45, 0.25)))
    lo, hi = ((3, 9), (8, 19), (18, 30))[scale]
    h = int(rng.integers(max(minimum, lo), hi + 1))
    w = int(rng.integers(max(minimum, lo), hi + 1))
    aspect = int(rng.integers(6))
    if aspect == 0:
        w = h
    elif aspect == 1:
        w = max(w, min(30, h + 4))
    elif aspect == 2:
        h = max(h, min(30, w + 4))
    return h, w


def _mask(rng, shape, count, density, mode):
    h, w = shape
    target = min(h * w - 1, max(count, int(round(density * h * w))))
    yy, xx = np.indices(shape)
    if mode == 0:  # compact region
        cy, cx = rng.uniform(0, h - 1), rng.uniform(0, w - 1)
        ry, rx = rng.uniform(0.6, 2.2), rng.uniform(0.6, 2.2)
        score = -((yy - cy) / ry) ** 2 - ((xx - cx) / rx) ** 2
    elif mode == 1:  # overlapping rectangles
        score = rng.random(shape) * 0.15
        for _ in range(int(rng.integers(2, 7))):
            rh = int(rng.integers(1, min(h, max(2, h // 2)) + 1))
            rw = int(rng.integers(1, min(w, max(2, w // 2)) + 1))
            y = int(rng.integers(0, h - rh + 1))
            x = int(rng.integers(0, w - rw + 1))
            score[y:y + rh, x:x + rw] += rng.uniform(0.8, 1.8)
    elif mode == 2:  # uneven bars
        if rng.integers(2):
            score = np.repeat(rng.random((h, 1)), w, axis=1)
        else:
            score = np.repeat(rng.random((1, w)), h, axis=0)
        score += rng.random(shape) * 0.12
    elif mode == 3:  # short fragments
        score = rng.random(shape) * 0.3
        for _ in range(int(rng.integers(4, 14))):
            rh = int(rng.integers(1, min(3, h) + 1))
            rw = int(rng.integers(1, min(3, w) + 1))
            y = int(rng.integers(0, h - rh + 1))
            x = int(rng.integers(0, w - rw + 1))
            score[y:y + rh, x:x + rw] += rng.uniform(0.3, 1.5)
    elif mode == 4:  # irregular correlated mask
        score = rng.random(shape)
        for _ in range(3):
            padded = np.pad(score, 1, mode="edge")
            score = (2 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]
                     + padded[2:, 1:-1] + padded[1:-1, :-2]
                     + padded[1:-1, 2:]) / 6
    else:  # scattered cells
        score = rng.random(shape)
    score += rng.random(shape) * 0.001
    chosen = np.argpartition(score.ravel(), -target)[-target:]
    mask = np.zeros(h * w, dtype=bool)
    mask[chosen] = True
    return mask.reshape(shape)


def _colors_on_mask(rng, mask, colors, style):
    h, w = mask.shape
    positions = np.argwhere(mask)
    n = len(positions)
    order = [int(c) for c in rng.permutation(colors)]
    grid = np.zeros(mask.shape, dtype=np.int8)
    if len(order) == 1:
        grid[mask] = order[0]
        return grid
    if style == 1:  # dominant color with tiny accents
        grid[mask] = order[0]
        available = np.ones(n, dtype=bool)
        for color_index, color in enumerate(order[1:]):
            anchor = int(rng.choice(np.flatnonzero(available)))
            size = int(rng.integers(1, min(5, max(2, n // (4 * len(order)))) + 1))
            distance = np.sum((positions - positions[anchor]) ** 2, axis=1)
            distance[~available] = h * h + w * w + 1
            leave = len(order) - color_index - 2
            selected = np.argsort(distance, kind="stable")[:min(size, int(available.sum()) - leave)]
            for index in selected:
                y, x = positions[index]
                grid[y, x] = color
            available[selected] = False
    elif style == 2:  # broad horizontal or vertical bands
        axes = [axis for axis, length in ((0, h), (1, w)) if length >= len(order)]
        if axes:
            axis = int(rng.choice(axes))
            length = h if axis == 0 else w
            cuts = np.sort(rng.choice(np.arange(1, length),
                                      size=len(order) - 1, replace=False))
            bands = np.searchsorted(cuts, positions[:, axis], side="right")
            for (y, x), band in zip(positions, bands):
                grid[y, x] = order[int(band)]
        else:
            values = rng.choice(order, size=n)
            for index, (y, x) in enumerate(positions):
                grid[y, x] = values[index]
    elif style == 3:  # irregular fine mosaic
        weights = rng.dirichlet(np.ones(len(order)) * rng.uniform(0.8, 3.0))
        values = rng.choice(order, size=n, p=weights)
        for index, (y, x) in enumerate(positions):
            grid[y, x] = values[index]
    else:  # coherent regions; number of patches differs by color
        anchors, labels = [], []
        available = rng.permutation(n)
        cursor = 0
        for color in order:
            for _ in range(int(rng.integers(1, 4))):
                anchors.append(positions[available[cursor % n]])
                labels.append(color)
                cursor += 1
        anchors = np.asarray(anchors)
        distances = ((positions[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(distances, axis=1)
        for index, (y, x) in enumerate(positions):
            grid[y, x] = labels[int(nearest[index])]
    # Distinct anchor cells guarantee visibility even in extremely small masks.
    if style != 2 or any(not np.any(grid == color) for color in order):
        for color, index in zip(order, rng.choice(n, size=len(order), replace=False)):
            y, x = positions[index]
            grid[y, x] = color
    return grid


def _repeated_one_color(rng, shape, color):
    h, w = shape
    grid = np.zeros(shape, dtype=np.int8)
    dense = bool(rng.random() < 0.4)
    if w >= h:
        limit = (w - 1) // 2
        low_width = max(1, limit // 2) if dense else 1
        width = int(rng.integers(low_width, limit + 1))
        height = int(rng.integers(max(1, int(0.7 * h)) if dense else 1, h + 1))
        y1 = int(rng.integers(0, h - height + 1))
        y2 = int(rng.integers(0, h - height + 1))
        grid[y1:y1 + height, :width] = color
        grid[y2:y2 + height, w - width:] = color
    else:
        limit = (h - 1) // 2
        low_height = max(1, limit // 2) if dense else 1
        height = int(rng.integers(low_height, limit + 1))
        width = int(rng.integers(max(1, int(0.7 * w)) if dense else 1, w + 1))
        x1 = int(rng.integers(0, w - width + 1))
        x2 = int(rng.integers(0, w - width + 1))
        grid[:height, x1:x1 + width] = color
        grid[h - height:, x2:x2 + width] = color
    return grid


def _contiguous_one_color(rng, shape, color):
    h, w = shape
    rh = int(rng.integers(1, h + 1))
    rw = int(rng.integers(1, w + 1))
    if rh == h and rw == w:
        rh -= 1
    y = int(rng.integers(0, h - rh + 1))
    x = int(rng.integers(0, w - rw + 1))
    grid = np.zeros(shape, dtype=np.int8)
    grid[y:y + rh, x:x + rw] = color
    return grid


def _make_input(rng, shape, colors, has_zero, mode, style, density):
    mask = (_mask(rng, shape, len(colors), density, mode) if has_zero
            else np.ones(shape, dtype=bool))
    return _colors_on_mask(rng, mask, colors, style)


def _fallback_input(colors, has_zero, repeated, seen):
    # Exhaustive shape variation makes even identical monochrome fills distinct.
    for nonce in range(28 * 28):
        h, w = 3 + nonce // 28, 3 + nonce % 28
        grid = np.zeros((h, w), dtype=np.int8) if has_zero else np.full(
            (h, w), colors[0], dtype=np.int8)
        for column, color in enumerate(colors):
            grid[0, column] = color
        if repeated:
            grid[-1, -1] = colors[0]
        key = _key(grid)
        if key not in seen and key not in SOURCE_KEYS:
            return grid
    raise RuntimeError("could not construct a distinct input")


def _output_for(grid):
    return np.zeros((1, int(np.count_nonzero(np.unique(grid)))), dtype=np.int8)


def _components_of_color(grid, color):
    mask = grid == color
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    components = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            components += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                r, c = stack.pop()
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
    return components


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    n_train, n_test = _pair_counts(seed, rng, num_train, num_test)
    # Palette is visual coherence only. No sampled value affects the rule.
    palette = [int(x) for x in rng.permutation(np.arange(1, 10))]
    train_palette = palette[:int(rng.integers(4, 8))]
    specs = [(1, True, "repeated"), (2, True, "paired")]
    if n_train >= 3:
        specs.append((3, True, "general"))
    if n_train >= 4:
        specs.append((1, True, "contiguous"))
    while len(specs) < n_train:
        specs.append((int(rng.choice((1, 2, 3, 4), p=(0.20, 0.30, 0.35, 0.15))),
                      True, "general"))
    if n_train >= 4 and rng.random() < 0.65:
        index = int(rng.integers(2, n_train))
        count, _, kind = specs[index]
        specs[index] = (count, False, "general" if kind == "contiguous" else kind)
    accent_index = int(rng.integers(1, n_train))
    if specs[accent_index][0] == 1:
        accent_index = 1

    puzzle = {"train": [], "test": []}
    seen, observed = set(), set()
    shared_shape = _dimensions(rng, minimum=6)
    shared_density = 0.3
    for i, (count, has_zero, kind) in enumerate(specs):
        for _ in range(40):
            shape = shared_shape if i < 2 else _dimensions(rng)
            if i == 0:
                colors = [int(rng.choice(train_palette))]
                grid = _repeated_one_color(rng, shape, colors[0])
                shared_density = np.count_nonzero(grid) / grid.size
            else:
                unseen = [c for c in train_palette if c not in observed]
                chosen = [int(rng.choice(unseen))] if unseen and rng.random() < 0.65 else []
                pool = [c for c in train_palette if c not in chosen]
                colors = chosen + [int(x) for x in rng.choice(
                    pool, size=count - len(chosen), replace=False)]
                if kind == "contiguous" and has_zero:
                    grid = _contiguous_one_color(rng, shape, colors[0])
                else:
                    density = (shared_density if i == 1 or (i == 2 and rng.random() < 0.5)
                               else float(rng.uniform(0.04, 0.94)))
                    style = 1 if i == accent_index and count > 1 else int(rng.integers(4))
                    grid = _make_input(rng, shape, colors, has_zero,
                                       int(rng.integers(6)), style, density)
            key = _key(grid)
            if (key not in seen and key not in SOURCE_KEYS
                    and set(np.unique(grid)) - {0} == set(colors)
                    and bool(np.any(grid == 0)) == has_zero):
                break
        else:
            grid = _fallback_input(colors, has_zero, i == 0, seen)
            key = _key(grid)
        seen.add(key)
        observed.update(colors)
        puzzle["train"].append({"input": grid, "output": _output_for(grid)})

    train_counts = {pair["output"].shape[1] for pair in puzzle["train"]}
    for j in range(n_test):
        for _ in range(40):
            count = int(rng.choice((1, 2, 3, 4, 5),
                                   p=(0.20, 0.22, 0.23, 0.22, 0.13)))
            has_zero = bool(rng.integers(2))
            novelty = bool(rng.integers(2))
            if not novelty and count > len(observed):
                novelty = True
            if j == 0 and count in train_counts and has_zero and not novelty:
                has_zero = False
            novel = [c for c in range(1, 10) if c not in observed]
            if novelty and novel:
                first = int(rng.choice(novel))
                remaining = [c for c in range(1, 10) if c != first]
                colors = [first] + [int(x) for x in rng.choice(
                    remaining, size=count - 1, replace=False)]
            else:
                pool = sorted(observed) if count <= len(observed) else list(range(1, 10))
                colors = [int(x) for x in rng.choice(pool, size=count, replace=False)]
            shape = _dimensions(rng)
            density = float(rng.uniform(0.04, 0.94))
            grid = _make_input(rng, shape, colors, has_zero,
                               int(rng.integers(6)), int(rng.integers(4)), density)
            key = _key(grid)
            if (key not in seen and key not in SOURCE_KEYS
                    and set(np.unique(grid)) - {0} == set(colors)
                    and bool(np.any(grid == 0)) == has_zero):
                break
        else:
            grid = _fallback_input(colors, has_zero, False, seen)
            key = _key(grid)
        seen.add(key)
        puzzle["test"].append({"input": grid, "output": _output_for(grid)})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, observed, train_counts = set(), set(), set()
    zero_train = 0
    repeated_train = False
    test_shift = False
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert np.issubdtype(array.dtype, np.integer)
                assert all(1 <= d <= 30 for d in array.shape)
                assert np.all((0 <= array) & (array <= 9))
            key = _key(grid)
            assert key not in seen and key not in SOURCE_KEYS
            seen.add(key)
            colors = set(int(x) for x in np.unique(grid)) - {0}
            assert colors and np.array_equal(output, _output_for(grid))
            assert not np.array_equal(grid, output)
            if regime == "train":
                train_counts.add(len(colors))
                observed.update(colors)
                zero_train += int(np.any(grid == 0))
                if len(colors) == 1 and np.any(grid == 0):
                    repeated_train |= _components_of_color(grid, next(iter(colors))) >= 2
            else:
                test_shift |= (not np.any(grid == 0) or bool(colors - observed)
                               or len(colors) not in train_counts)
    assert zero_train >= 2 and repeated_train
    assert len(train_counts) >= 2 and 1 in train_counts
    if nt >= 3:
        assert 2 in train_counts and 3 in train_counts
    assert test_shift


def _puzzles_equal(a, b):
    return all(
        len(a[regime]) == len(b[regime]) and all(
            np.array_equal(pa[role], pb[role])
            for pa, pb in zip(a[regime], b[regime]) for role in ("input", "output")
        ) for regime in ("train", "test")
    )


def self_test() -> None:
    train_sizes, test_sizes, shapes, palettes, layouts, regimes = (
        set(), set(), set(), set(), set(), set())
    zero_free_train = single_cell_train = contiguous_train = matched_area = 0
    tiny = large = square = wide = tall = 0
    dense_one = sparse_multi = dominant_tiny = balanced = unequal_components = 0
    full_fill = full_bands = full_irregular = within_count = above_count = 0
    source_sized = set()
    previous = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        if previous is not None:
            assert not _puzzles_equal(puzzle, previous)
        previous = puzzle
        train_sizes.add(len(puzzle["train"]))
        test_sizes.add(len(puzzle["test"]))
        observed = set(int(c) for p in puzzle["train"] for c in np.unique(p["input"])) - {0}
        demonstrated = {pair["output"].shape[1] for pair in puzzle["train"]}
        first, second = puzzle["train"][:2]
        a = np.count_nonzero(first["input"]) / first["input"].size
        b = np.count_nonzero(second["input"]) / second["input"].size
        matched_area += int(first["input"].shape == second["input"].shape
                            and abs(a - b) <= 0.12)
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                grid, output = pair["input"], pair["output"]
                colors = set(int(c) for c in np.unique(grid)) - {0}
                assert output.shape == (1, len(colors)) and np.all(output == 0)
                h, w = grid.shape
                if (h, w) in {(12, 12), (10, 14), (6, 4),
                              (14, 12), (6, 6), (4, 6)}:
                    source_sized.add((h, w))
                shapes.add((h, w))
                palettes.add(tuple(sorted(colors)))
                layouts.add((h, w, tuple((grid != 0).ravel())))
                tiny += int(min(h, w) <= 6)
                large += int(max(h, w) >= 20)
                square += int(h == w)
                wide += int(w >= h + 4)
                tall += int(h >= w + 4)
                fraction = np.count_nonzero(grid) / grid.size
                frequencies = [int(np.count_nonzero(grid == c)) for c in colors]
                components = [_components_of_color(grid, c) for c in colors]
                dense_one += int(len(colors) == 1 and fraction >= 0.6)
                sparse_multi += int(len(colors) >= 3 and fraction <= 0.2)
                dominant_tiny += int(len(colors) >= 2 and
                                     max(frequencies) / sum(frequencies) >= 0.8 and
                                     min(frequencies) <= 3)
                balanced += int(len(colors) >= 2 and
                                max(frequencies) / sum(frequencies) <= 0.65)
                unequal_components += int(len(set(components)) > 1)
                if regime == "train":
                    zero_free_train += int(not np.any(grid == 0))
                    single_cell_train += int(any(np.count_nonzero(grid == c) == 1 for c in colors))
                    contiguous_train += int(len(colors) == 1
                                            and _components_of_color(grid, next(iter(colors))) == 1)
                else:
                    has_zero = bool(np.any(grid == 0))
                    regimes.add((len(colors), has_zero, bool(colors - observed)))
                    within_count += int(len(colors) in demonstrated)
                    above_count += int(len(colors) > max(demonstrated))
                    if not has_zero:
                        full_fill += int(len(colors) == 1)
                        if len(colors) >= 2:
                            band = (all(len(set(row)) == 1 for row in grid) or
                                    all(len(set(grid[:, c])) == 1 for c in range(w)))
                            full_bands += int(band)
                            full_irregular += int(not band)
    assert len(train_sizes) >= 4 and len(test_sizes) >= 3
    assert len(shapes) >= 90 and len(palettes) >= 25 and len(layouts) >= 180
    assert min(tiny, large, square, wide, tall) >= 8
    assert len(source_sized) == 6
    assert zero_free_train >= 3 and single_cell_train >= 3 and contiguous_train >= 3
    assert matched_area >= 20
    assert min(dense_one, sparse_multi, dominant_tiny, balanced,
               unequal_components) >= 5
    assert min(full_fill, full_bands, full_irregular, within_count,
               above_count) >= 5
    assert any(k == 5 for k, _, _ in regimes)
    assert any(k == 1 and not z for k, z, _ in regimes)
    assert any(k == 1 and n for k, _, n in regimes)
    assert any(k == 3 and not z for k, z, _ in regimes)
    assert any(k == 4 and z for k, z, _ in regimes)
    assert len(regimes) >= 14
    for seed, nt, ns in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                         (104, None, 1), (105, None, 4),
                         (106, 2, None), (107, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        assert _puzzles_equal(puzzle, generate_puzzle(seed, nt, ns))
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 3), (3, 4), (1, 1)):
        try:
            generate_puzzle(123, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {nt}, {ns}")
