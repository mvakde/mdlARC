"""Synthetic instances of the solid-square center-axis ARC rule."""

import hashlib
import numpy as np


_SOURCE_PAIR_DIGESTS = {
    "acf7930bc0972c629627042658f204c6f83ee4d2c7097b91b18cb32458cd5822",
    "fa560147eac6d2342075932feb19ceb18708d7503f7aab9be6e93221677b492e",
    "a1ff3b761c163b5cea5a40ea7928fdf6988ca0f30aa92675fd22551900d595a5",
    "d2b78747f473392e6927da1b87e375a8f8128d2f8a3c457761a02d8d9ee9e95b",
    "deff780ac3f72081f1829e80c2da6a4198795e9dcb54397448f032af5b6d17a5",
}


def _optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _optional_count(num_train, "num_train", 2, 10)
    test = _optional_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        choices = [(a, b) for a in range(2, 11) for b in range(1, min(4, a - 1) + 1)]
        return choices[int(rng.integers(len(choices)))]
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _background(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _find_sources(grid, background):
    center = grid[1:-1, 1:-1]
    complete = center != background
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            complete &= center == grid[1 + dr:grid.shape[0] - 1 + dr,
                                       1 + dc:grid.shape[1] - 1 + dc]
    return [(int(r + 1), int(c + 1), int(grid[r + 1, c + 1]))
            for r, c in np.argwhere(complete)]


def _source_mask(shape, sources):
    mask = np.zeros(shape, dtype=bool)
    for r, c, _ in sources:
        mask[r - 1:r + 2, c - 1:c + 2] = True
    return mask


def _claims(shape, sources):
    result = np.zeros(shape, dtype=np.int8)
    for r, c, _ in sources:
        result[r, :] += 1
        result[:, c] += 1
        result[r, c] -= 1
    return result


def _output(grid, background, sources):
    out = grid.copy()
    claims = _claims(grid.shape, sources)
    for r, c, color in sources:
        out[r, claims[r] == 1] = color
        out[claims[:, c] == 1, c] = color
    out[claims > 1] = background
    return out


def _dimension(rng, regime, count):
    # Height and width are sampled separately; count only sets a geometric floor.
    lo = max(12, 3 * count)
    compact = rng.random() < (0.55 if regime == "source_train" else 0.43)
    return int(rng.integers(lo, 19 if compact else 31)) if compact else int(rng.integers(19, 31))


def _centers(rng, length, count):
    # Stars-and-bars sampling gives every legal placement with three-cell spacing.
    compressed = rng.choice(np.arange(1, length - 1 - 2 * (count - 1)),
                            size=count, replace=False)
    return np.sort(compressed) + 2 * np.arange(count)


def _generate_pair(rng, regime, background, count, same_color=False,
                   require_overwrite=False, require_offaxis=False,
                   require_crossing=False, shape_override=None):
    for _ in range(100):
        if shape_override is None:
            h = _dimension(rng, regime, count)
            w = _dimension(rng, regime, count)
        else:
            h, w = shape_override
        rows = _centers(rng, h, count)
        cols = _centers(rng, w, count)
        rng.shuffle(cols)
        available = [x for x in range(10) if x != background]
        colors = [int(x) for x in rng.choice(available, size=count, replace=False)]
        if same_color:
            colors[1] = colors[0]
        sources = [(int(r), int(c), color) for r, c, color in zip(rows, cols, colors)]
        grid = np.full((h, w), background, dtype=np.int8)
        for r, c, color in sources:
            grid[r - 1:r + 2, c - 1:c + 2] = color
        source_mask = _source_mask(grid.shape, sources)
        claims = _claims(grid.shape, sources)
        crossing = (claims > 1) & ~source_mask
        used = source_mask.copy()

        # Each arm may be blank, partially precolored, or open to noise.
        policies = rng.choice(3, size=(count, 2), p=[0.29, 0.23, 0.48])
        if require_overwrite and not np.any(policies == 2):
            policies[int(rng.integers(count)), int(rng.integers(2))] = 2
        protected = np.zeros(grid.shape, dtype=bool)
        for i, (r, c, color) in enumerate(sources):
            for axis in range(2):
                arm = np.zeros(grid.shape, dtype=bool)
                if axis == 0:
                    arm[r, :] = True
                else:
                    arm[:, c] = True
                arm &= (claims == 1) & ~source_mask
                if policies[i, axis] != 2:
                    protected |= arm
                if policies[i, axis] == 1:
                    indices = np.flatnonzero(arm & ~used)
                    if len(indices):
                        selected = rng.choice(indices, size=min(len(indices), int(rng.integers(1, 4))), replace=False)
                        grid.flat[selected] = color
                        used.flat[selected] = True

        # Crossing input states vary independently of the arms and split.
        cross_indices = np.flatnonzero(crossing)
        if len(cross_indices):
            mode = int(rng.integers(3))  # blank, partial, all colored
            n = (0 if mode == 0 else len(cross_indices) if mode == 2
                 else int(rng.integers(1, len(cross_indices))) if len(cross_indices) > 1 else 1)
            if require_crossing:
                n = max(1, n)
            selected = rng.choice(cross_indices, size=n, replace=False)
            # In same-color training, explicitly clear one colored crossing
            # between the two same-colored objects.
            if same_color and require_crossing:
                selected = np.unique(np.append(selected, sources[0][0] * w + sources[1][1]))
            for index in selected:
                grid.flat[index] = int(rng.choice(available))
                used.flat[index] = True

        if require_overwrite:
            possible = np.flatnonzero((claims == 1) & ~source_mask & ~protected & ~used)
            if not len(possible):
                continue
            index = int(rng.choice(possible))
            owners = [color for r, c, color in sources if index // w == r or index % w == c]
            grid.flat[index] = int(rng.choice([x for x in available if x != owners[0]]))
            used.flat[index] = True
        if require_offaxis:
            possible = np.flatnonzero((claims == 0) & ~used)
            if not len(possible):
                continue
            index = int(rng.choice(possible))
            grid.flat[index] = int(rng.choice(available))
            used.flat[index] = True

        clutter_weights = ([0.24, 0.44, 0.32] if regime == "source_train"
                           else [0.19, 0.39, 0.42])
        mode = int(rng.choice(3, p=clutter_weights))
        area = h * w
        if mode == 0:
            target = int(rng.integers(0, 9))
        elif mode == 1:
            target = int(rng.integers(9, min(33, max(10, area // 8)) + 1))
        else:
            target = int(rng.integers(30, min(91, max(31, area // 4)) + 1))
        target = max(target, int(np.count_nonzero(used & ~source_mask)))
        palette_size = int(rng.integers(2, 10))
        extras = [x for x in available if x not in colors]
        rng.shuffle(extras)
        palette = list(dict.fromkeys(colors)) + extras[:max(0, palette_size - len(set(colors)))]

        # Deliberate nonsource shapes: bars, 2x2 patches, and eight-cell
        # incomplete 3x3 lookalikes. Each is independently positioned.
        cluster_budget = int(rng.integers(0, 4)) if target >= 9 else int(rng.integers(0, 2))
        for _cluster in range(cluster_budget):
            kind = int(rng.integers(3))
            for _placement in range(18):
                rr = int(rng.integers(0, h - 3 + 1))
                cc = int(rng.integers(0, w - 3 + 1))
                if kind == 0:
                    cells = [(rr + dr, cc + dc) for dr in range(2) for dc in range(2)]
                elif kind == 1:
                    length = int(rng.integers(2, 5))
                    if rng.random() < 0.5:
                        cells = [(rr, cc + dc) for dc in range(length) if cc + dc < w]
                    else:
                        cells = [(rr + dr, cc) for dr in range(length) if rr + dr < h]
                else:
                    omitted = int(rng.integers(9))
                    cells = [(rr + dr, cc + dc) for dr in range(3) for dc in range(3)
                             if 3 * dr + dc != omitted]
                if len(cells) > target - int(np.count_nonzero(used & ~source_mask)):
                    break
                if any(used[r, c] or protected[r, c] for r, c in cells):
                    continue
                color = int(rng.choice(colors)) if rng.random() < 0.55 else int(rng.choice(palette))
                for r, c in cells:
                    grid[r, c] = color
                    used[r, c] = True
                if _find_sources(grid, background) == sources:
                    break
                for r, c in cells:
                    grid[r, c] = background
                    used[r, c] = False

        need = target - int(np.count_nonzero(used & ~source_mask))
        candidates = np.flatnonzero(~used & ~protected)
        if need > len(candidates):
            continue
        if need:
            selected = rng.choice(candidates, size=need, replace=False)
            grid.flat[selected] = rng.choice(palette, size=need)
        if _find_sources(grid, background) != sources:
            continue
        counts = np.bincount(grid.ravel(), minlength=10)
        if counts[background] <= max(counts[x] for x in range(10) if x != background):
            continue
        out = _output(grid, background, sources)
        return grid, out
    raise RuntimeError("could not construct an unambiguous pair")


def _pair_digest(input_grid, output_grid):
    return hashlib.sha256(bytes(input_grid.shape) +
                          input_grid.astype(np.uint8, copy=False).tobytes() +
                          output_grid.astype(np.uint8, copy=False).tobytes()).hexdigest()


def _features(pair):
    grid = pair["input"]
    bg = _background(grid)
    sources = _find_sources(grid, bg)
    mask = _source_mask(grid.shape, sources)
    claims = _claims(grid.shape, sources)
    return bg, sources, mask, claims


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    train_counts = set()
    train_same = False
    train_overwrite = False
    train_offaxis = False
    train_colored_crossing = False
    test_four = False
    test_same = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert a.shape == b.shape
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            assert _pair_digest(a, b) not in _SOURCE_PAIR_DIGESTS
            bg, sources, mask, claims = _features(pair)
            assert 1 <= len(sources) <= 4
            assert a.shape[0] >= 3 * len(sources) and a.shape[1] >= 3 * len(sources)
            counts = np.bincount(a.ravel(), minlength=10)
            assert counts[bg] > max(counts[c] for c in range(10) if c != bg)
            assert len({r for r, _, _ in sources}) == len(sources)
            assert len({c for _, c, _ in sources}) == len(sources)
            for i, (r, c, color) in enumerate(sources):
                assert 1 <= r < a.shape[0] - 1 and 1 <= c < a.shape[1] - 1
                assert np.all(a[r - 1:r + 2, c - 1:c + 2] == color)
                assert np.all(b[r - 1:r + 2, c - 1:c + 2] == color)
                for r2, c2, _ in sources[i + 1:]:
                    assert abs(r - r2) >= 3 and abs(c - c2) >= 3
            assert np.array_equal(b, _output(a, bg, sources))
            assert not np.array_equal(a, b)
            assert np.array_equal(b[claims == 0], a[claims == 0])
            assert np.all(b[claims > 1] == bg)
            same = len({color for _, _, color in sources}) < len(sources)
            overwrite = np.any((claims == 1) & ~mask & (a != bg) & (a != b))
            offaxis = np.any((claims == 0) & (a != bg))
            colored_cross = np.any((claims > 1) & (a != bg))
            if split == "train":
                train_counts.add(len(sources))
                train_same |= bool(same)
                train_overwrite |= bool(overwrite)
                train_offaxis |= bool(offaxis)
                train_colored_crossing |= bool(colored_cross)
            else:
                test_four |= len(sources) == 4
                test_same |= same
    assert len(train_counts) >= 2 and max(train_counts) >= 2
    assert train_overwrite and train_offaxis and train_colored_crossing
    assert not test_four or 4 in train_counts
    assert not test_same or train_same


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    value = int(seed)
    rng = np.random.default_rng(value if value >= 0 else value % (1 << 64))
    ntrain, ntest = _resolve_counts(rng, num_train, num_test)
    shared_background = rng.random() < 0.55
    base_background = int(rng.integers(10))
    same_color_family = rng.random() < 0.28
    growth_puzzle = rng.random() < 0.18
    test_counts = [int(rng.choice([1, 2, 3, 4], p=[0.18, 0.27, 0.40, 0.15]))
                   for _ in range(ntest)]
    train_counts = [max(2, max(test_counts))]
    alternatives = [x for x in (1, 2, 3, 4) if x != train_counts[0]]
    train_counts.append(int(rng.choice(alternatives)))
    train_counts.extend(int(rng.choice([1, 2, 3, 4], p=[0.25, 0.32, 0.31, 0.12]))
                        for _ in range(ntrain - 2))
    rng.shuffle(train_counts)
    anchor = train_counts.index(max(train_counts))
    backgrounds = [base_background] * (ntrain + ntest)
    if not shared_background:
        backgrounds = [int(rng.integers(10)) for _ in backgrounds]
        if backgrounds[0] == backgrounds[1]:
            backgrounds[1] = (backgrounds[0] + int(rng.integers(1, 10))) % 10
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, counts, offset in (("train", train_counts, 0),
                                  ("test", test_counts, ntrain)):
        for i, count in enumerate(counts):
            force = split == "train" and i == anchor
            same = (count >= 2 and same_color_family and
                    (force or rng.random() < 0.38))
            shape_override = None
            if split == "train" and growth_puzzle:
                shape_override = (int(rng.integers(12, 23)),
                                  int(rng.integers(12, 23)))
            if split == "test" and growth_puzzle and i == 0:
                train_shapes = [pair["input"].shape for pair in puzzle["train"]]
                shape_override = (int(rng.integers(max(h for h, _ in train_shapes) + 1, 31)),
                                  int(rng.integers(max(w for _, w in train_shapes) + 1, 31)))
            elif split == "test":
                mode = int(rng.choice(4, p=[0.20, 0.25, 0.25, 0.30]))
                train_shapes = [pair["input"].shape for pair in puzzle["train"]]
                if mode == 0:  # exact canvas size, new objects and marks
                    shape_override = train_shapes[int(rng.integers(len(train_shapes)))]
                elif mode == 1:  # a smaller canvas
                    h0, w0 = train_shapes[int(rng.integers(len(train_shapes)))]
                    shape_override = (int(rng.integers(12, h0 + 1)),
                                      int(rng.integers(12, w0 + 1)))
                elif mode == 2:  # a larger canvas
                    h0, w0 = train_shapes[int(rng.integers(len(train_shapes)))]
                    shape_override = (int(rng.integers(h0, 31)),
                                      int(rng.integers(w0, 31)))
            for _attempt in range(12):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                a, b = _generate_pair(np.random.default_rng(pair_seed),
                                      "source_train" if split == "train" else "target_test",
                                      backgrounds[offset + i], count, same,
                                      require_overwrite=force,
                                      require_offaxis=force,
                                      require_crossing=force,
                                      shape_override=shape_override)
                key = (a.shape, a.tobytes())
                if key not in seen and _pair_digest(a, b) not in _SOURCE_PAIR_DIGESTS:
                    seen.add(key)
                    puzzle[split].append({"input": a, "output": b})
                    break
            else:
                raise RuntimeError("could not make distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _independent_output(a):
    bg = _background(a)
    sources = _find_sources(a, bg)
    out = a.copy()
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            owners = [(sr, sc, color) for sr, sc, color in sources if r == sr or c == sc]
            if len(owners) == 1:
                out[r, c] = owners[0][2]
            elif len(owners) > 1:
                out[r, c] = bg
    return out


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and
               all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    for kwargs in ({}, {"num_train": 7, "num_test": 3}, {"num_train": None, "num_test": 1},
                   {"num_train": None, "num_test": 4}, {"num_train": 2, "num_test": None},
                   {"num_train": 10, "num_test": None}, {"num_train": 2, "num_test": 1},
                   {"num_train": 10, "num_test": 1}, {"num_train": 10, "num_test": 4}):
        a = generate_puzzle(731, **kwargs)
        assert _puzzles_equal(a, generate_puzzle(731, **kwargs))
        validate_generated_puzzle(a)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(123, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count was accepted")
    train_sizes, test_sizes, shapes, layouts, backgrounds = set(), set(), set(), set(), set()
    source_counts = {"train": set(), "test": set()}
    clutter = {"train": set(), "test": set()}
    clear_axis = own_axis = colored_cross = blank_cross = partial_cross = local_bg = repeated = four = 0
    smaller_test = same_size_test = larger_test = exceeds_all_train = large = compact = rectangular = 0
    blocks = bars = incomplete = source_colored = within_clutter = 0
    train_orders = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_sizes.add(len(puzzle["train"]))
        test_sizes.add(len(puzzle["test"]))
        bgs = set()
        train_dims = [p["input"].shape for p in puzzle["train"]]
        train_orders.add(tuple(len(_find_sources(p["input"], _background(p["input"])))
                               for p in puzzle["train"]))
        puzzle_clutter = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                assert np.array_equal(b, _independent_output(a))
                bg, sources, mask, claims = _features(pair)
                bgs.add(bg)
                backgrounds.add(bg)
                source_counts[split].add(len(sources))
                shapes.add(a.shape)
                layouts.add((a.shape, tuple((r, c) for r, c, _ in sources)))
                incidental = int(np.count_nonzero((a != bg) & ~mask))
                clutter_class = 0 if incidental <= 8 else 1 if incidental < 30 else 2
                clutter[split].add(clutter_class)
                puzzle_clutter.add(clutter_class)
                repeated += len({color for _, _, color in sources}) < len(sources)
                source_colored += any(np.any((a == color) & ~mask) for _, _, color in sources)
                four += len(sources) == 4
                large += max(a.shape) >= 27
                compact += min(a.shape) <= 15
                rectangular += abs(a.shape[0] - a.shape[1]) >= 7
                cross = (claims > 1)
                if np.any(cross):
                    colored_cross += bool(np.any(a[cross] != bg))
                    blank_cross += bool(np.all(a[cross] == bg))
                    partial_cross += bool(np.any(a[cross] != bg) and np.any(a[cross] == bg))
                for r, c, color in sources:
                    for arm in (((r, j) for j in range(a.shape[1])),
                                ((j, c) for j in range(a.shape[0]))):
                        cells = [(rr, cc) for rr, cc in arm if claims[rr, cc] == 1 and not mask[rr, cc]]
                        clear_axis += bool(cells and all(a[rr, cc] in (bg, color) for rr, cc in cells))
                        own_axis += any(a[rr, cc] == color for rr, cc in cells)
                for r in range(a.shape[0] - 1):
                    for c in range(a.shape[1] - 1):
                        tile = a[r:r+2, c:c+2]
                        blocks += bool(a[r, c] != bg and np.all(tile == a[r, c])
                                       and np.all(~mask[r:r+2, c:c+2]))
                for r in range(a.shape[0] - 2):
                    for c in range(a.shape[1] - 2):
                        tile = a[r:r+3, c:c+3]
                        incomplete += bool(np.all(~mask[r:r+3, c:c+3]) and
                                           any(np.count_nonzero(tile == color) == 8
                                               for color in range(10) if color != bg))
                for r in range(a.shape[0]):
                    for c in range(a.shape[1] - 2):
                        bars += bool(a[r, c] != bg and np.all(a[r, c:c+3] == a[r, c])
                                     and np.all(~mask[r, c:c+3]))
                if split == "test" and any(a.shape[0] <= th and a.shape[1] <= tw for th, tw in train_dims):
                    smaller_test += 1
                if split == "test":
                    same_size_test += a.shape in train_dims
                    larger_test += any(a.shape[0] >= th and a.shape[1] >= tw
                                       for th, tw in train_dims)
                    exceeds_all_train += all(a.shape[0] > th and a.shape[1] > tw
                                             for th, tw in train_dims)
        local_bg += len(bgs) > 1
        within_clutter += len(puzzle_clutter) > 1
    assert len(train_sizes) >= 4 and len(test_sizes) >= 3
    assert len(shapes) >= 70 and len(layouts) >= 150 and len(backgrounds) >= 8
    assert source_counts["test"] >= {1, 2, 3, 4}
    assert clutter["train"] == clutter["test"] == {0, 1, 2}
    assert len(train_orders) >= 20 and within_clutter >= 20
    assert min(clear_axis, own_axis, colored_cross, blank_cross, partial_cross,
               local_bg, repeated, four, smaller_test, same_size_test, larger_test,
               exceeds_all_train, large, compact, rectangular, blocks, bars, incomplete,
               source_colored) > 0
    assert not _puzzles_equal(generate_puzzle(1), generate_puzzle(2))
