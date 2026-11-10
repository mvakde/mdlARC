"""Frequency-ranked ARC grids with bottom-up, left-to-right output packing."""

import numpy as np


_SOURCE_PAIR_ROWS = frozenset({
    (("4489494494", "4944544594", "5994594594", "5944594944", "5949444454",
      "5949449454", "5945459444", "5949459449", "5949449548", "4944949544"),
     ("4444499558", "4444499958", "4444499955", "4444499955", "4444499955",
      "4444499955", "4444499955", "4444499955", "4444499955", "4444499955")),
    (("2666656666", "2626652626", "6526252522", "6666256562", "6266266662",
      "8265662868", "5225662668", "5225266626", "6266286526", "6266286566"),
     ("6666622258", "6666622258", "6666622258", "6666622258", "6666622258",
      "6666622258", "6666622255", "6666622255", "6666622255", "6666662255")),
    (("3333933838", "8293383838", "8298983338", "8398382823", "8398392829",
      "8398393329", "8393338329", "8333933389", "3383938389", "3383938389"),
     ("3333888992", "3333888992", "3333888992", "3333888992", "3333888992",
      "3333888992", "3333388992", "3333388892", "3333388899", "3333388899")),
})


def _checked_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_pair_counts(rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        # Give compact puzzles substantial probability without losing long examples.
        num_train = int(rng.choice(np.arange(2, 11), p=[.24, .18, .14, .11, .09, .07, .06, .06, .05]))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _sample_shape(rng, first=False):
    if first and rng.random() < .23:
        return (10, 10)
    kind = int(rng.choice(5, p=[.24, .19, .19, .20, .18]))
    if kind == 0:
        h = int(rng.integers(7, 19))
        w = int(np.clip(h + rng.integers(-2, 3), 6, 22))
    elif kind == 1:
        h = int(rng.integers(5, 14))
        w = int(rng.integers(max(h + 3, 10), 27))
    elif kind == 2:
        w = int(rng.integers(5, 14))
        h = int(rng.integers(max(w + 3, 10), 27))
    elif kind == 3:
        if rng.random() < .5:
            h, w = int(rng.integers(5, 8)), int(rng.integers(9, 23))
        else:
            h, w = int(rng.integers(9, 23)), int(rng.integers(5, 8))
    else:
        h, w = int(rng.integers(17, 29)), int(rng.integers(17, 29))
    return h, w


def _sample_counts(rng, height, width, colors, boundary="free", regime="source_train"):
    total = height * width
    profile_weights = ([.24, .36, .28, .12] if regime == "source_train"
                       else [.40, .35, .18, .07])
    for _ in range(1800):
        profile = int(rng.choice(4, p=profile_weights))
        if profile == 0:  # closely spaced ranks
            weights = rng.dirichlet(np.full(colors, rng.uniform(6., 20.)))
        elif profile == 1:
            weights = rng.dirichlet(np.full(colors, rng.uniform(.9, 3.5)))
        elif profile == 2:  # sometimes one overwhelmingly frequent color
            weights = rng.dirichlet(np.full(colors, rng.uniform(.45, 1.6)))
            weights[int(rng.integers(colors))] *= rng.uniform(2., 8.)
            weights /= weights.sum()
        else:  # a one-cell tail
            weights = rng.dirichlet(np.full(colors - 1, rng.uniform(.8, 4.)))
        if profile == 3:
            counts = np.concatenate((rng.multinomial(total - colors, weights) + 1, [1]))
        else:
            counts = rng.multinomial(total - colors, weights) + 1
        counts = np.sort(counts)[::-1]
        if len(set(int(x) for x in counts)) != colors:
            continue
        boundaries = np.cumsum(counts)[:-1] % height
        if boundary == "partial" and not 2 <= counts[0] % height <= height - 1:
            continue
        if boundary == "aligned" and counts[0] % height != 0:
            continue
        if boundary == "none" and np.any(boundaries == 0):
            continue
        if boundary == "multiple" and np.count_nonzero(boundaries == 0) < 2:
            continue
        return counts.astype(np.int64)
    # Feasible requested profiles are common, but keep construction bounded.
    raise AssertionError("could not sample distinct color frequencies")


def _ranked(grid):
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return colors[order], counts[order]


def _pack(colors, counts, shape):
    height, width = shape
    stream = np.repeat(colors, counts)
    return stream.reshape(width, height)[:, ::-1].T.copy()


def _solve_grid(input_grid):
    return _pack(*_ranked(input_grid), input_grid.shape)


def _scatter(rng, colors, counts, shape, mode):
    height, width = shape
    total = height * width
    stream = np.repeat(colors, counts).astype(np.int8)
    if mode == "near":
        result = _pack(colors, counts, shape).ravel().copy()
        for _ in range(max(2, int(total * rng.uniform(.05, .26)))):
            a, b = rng.choice(total, size=2, replace=False)
            result[a], result[b] = result[b], result[a]
        return result.reshape(shape)
    rows, cols = np.indices(shape)
    if mode == "random":
        key = rng.random(shape)
    elif mode == "stripe":
        angle = rng.uniform(0, 2 * np.pi)
        axis = np.cos(angle) * rows / height + np.sin(angle) * cols / width
        key = axis + rng.uniform(.10, .38) * np.sin(
            rng.integers(2, 6) * np.pi * (cols / width - rows / height)
        ) + rng.normal(0, rng.uniform(.02, .12), shape)
    elif mode == "patch":
        rh, cw = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        field = rng.random((rh, cw))
        key = field[np.minimum(rows * rh // height, rh - 1),
                    np.minimum(cols * cw // width, cw - 1)]
        key = key + rng.normal(0, rng.uniform(.04, .23), shape)
    elif mode == "cluster":
        center_r, center_c = rng.uniform(0, height), rng.uniform(0, width)
        key = ((rows - center_r) / height) ** 2 + ((cols - center_c) / width) ** 2
        key = key + rng.normal(0, rng.uniform(.005, .07), shape)
    else:
        raise AssertionError("unknown scatter mode")
    order = np.argsort(key.ravel(), kind="stable")
    result = np.empty(total, dtype=np.int8)
    result[order] = stream
    return result.reshape(shape)


def _signature(grid):
    return grid.shape, grid.tobytes()


def _is_source_pair(input_grid, output_grid):
    if input_grid.shape != (10, 10) or output_grid.shape != (10, 10):
        return False
    rows = lambda grid: tuple("".join(str(int(x)) for x in row) for row in grid)
    return (rows(input_grid), rows(output_grid)) in _SOURCE_PAIR_ROWS


def _new_pair(rng, shape, colors, counts, mode):
    for _ in range(8):
        input_grid = _scatter(rng, colors, counts, shape, mode)
        output_grid = _solve_grid(input_grid)
        if not np.array_equal(input_grid, output_grid) and not _is_source_pair(input_grid, output_grid):
            return {"input": input_grid, "output": output_grid}
    # An exact packed input is only plausible for the near mode.
    input_grid = _scatter(rng, colors, counts, shape, "random")
    return {"input": input_grid, "output": _solve_grid(input_grid)}


def _pair_parameters(rng, regime, first=False):
    shape = _sample_shape(rng, first)
    palette_size = int(rng.integers(4, 7) if first else rng.integers(3, 8))
    if shape[0] * shape[1] < 9 * palette_size:
        palette_size = min(palette_size, 5)
    colors = rng.choice(10, size=palette_size, replace=False).astype(np.int8)
    boundary = "partial" if first else "free"
    if not first:
        choice = rng.random()
        if choice < .13:
            boundary = "aligned"
        elif choice < .27:
            boundary = "none"
        elif choice < .32 and palette_size >= 4 and shape[0] <= 13:
            boundary = "multiple"
    counts = _sample_counts(rng, *shape, palette_size, boundary, regime)
    return shape, colors, counts


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_pair_counts(rng, num_train, num_test)
    # The sole puzzle convention is the fixed descending-frequency, bottom-up
    # column traversal. Colors themselves have no persistent semantic roles.
    shape0, colors0, counts0 = _pair_parameters(rng, "source_train", first=True)
    modes = ("random", "stripe", "patch", "cluster", "near")
    mode0 = modes[int(rng.integers(len(modes)))]
    puzzle = {"train": [], "test": []}
    puzzle["train"].append(_new_pair(rng, shape0, colors0, counts0, mode0))
    seen = {_signature(puzzle["train"][0]["input"])}
    same_histogram = num_train >= 3 and rng.random() < .58

    for index in range(1, num_train + num_test):
        split = "source_train" if index < num_train else "target_test"
        for _ in range(30):
            if index == 1 and same_histogram:
                shape, colors, counts = shape0, colors0.copy(), counts0.copy()
            elif index == 1 or (index == 2 and same_histogram):
                # Same canvas and palette, but changed per-color counts and bands.
                shape, colors = shape0, colors0.copy()
                positions = rng.choice(len(colors), size=2, replace=False)
                colors[positions[0]], colors[positions[1]] = colors[positions[1]], colors[positions[0]]
                counts = _sample_counts(rng, *shape, len(colors), regime=split)
            elif split == "target_test" and index == num_train and seed % 3 == 0:
                # A test can keep both familiar endpoint ranks while middle ranks exchange.
                shape = _sample_shape(rng)
                colors = colors0.copy()
                colors[1], colors[2] = colors[2], colors[1]
                counts = _sample_counts(rng, *shape, len(colors), regime=split)
            elif split == "target_test" and index == num_train and seed % 3 == 1:
                # Another test reuses the palette but raises or lowers a familiar color.
                shape = _sample_shape(rng)
                colors = np.roll(colors0, 1)
                counts = _sample_counts(rng, *shape, len(colors), regime=split)
            else:
                shape, colors, counts = _pair_parameters(rng, split)
                if (split == "target_test" and index == num_train and seed % 3 == 2
                        and set(int(x) for x in colors).issubset(set(int(x) for x in colors0))):
                    available = np.array([x for x in range(10) if x not in colors0], dtype=np.int8)
                    colors[-1] = int(rng.choice(available))
            mode_choices = [m for m in modes if m != mode0] if index == 1 and same_histogram else modes
            mode = mode_choices[int(rng.integers(len(mode_choices)))]
            pair = _new_pair(rng, shape, colors, counts, mode)
            signature = _signature(pair["input"])
            if signature not in seen and not _is_source_pair(pair["input"], pair["output"]):
                break
        else:
            raise AssertionError("could not create a unique pair")
        seen.add(signature)
        puzzle["train" if split == "source_train" else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            before, after = pair["input"], pair["output"]
            for grid in (before, after):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert before.shape == after.shape and before.dtype == after.dtype
            assert not np.array_equal(before, after)
            colors, counts = np.unique(before, return_counts=True)
            other_colors, other_counts = np.unique(after, return_counts=True)
            assert 3 <= len(colors) <= 7
            assert len(set(int(x) for x in counts)) == len(counts)
            assert np.array_equal(colors, other_colors) and np.array_equal(counts, other_counts)
            assert np.array_equal(after, _solve_grid(before))
            sig = _signature(before)
            assert sig not in seen
            seen.add(sig)
            assert not _is_source_pair(before, after)
    # A partial first boundary reveals the vertical packing direction.
    first = puzzle["train"][0]["input"]
    _, counts = _ranked(first)
    assert 2 <= int(counts[0]) % first.shape[0] <= first.shape[0] - 1
    # More than one training histogram shows that output bands track counts.
    histograms = [tuple(sorted((int(c), int(n)) for c, n in zip(*np.unique(p["input"], return_counts=True))))
                  for p in puzzle["train"]]
    assert len(set(histograms)) >= 2
    demonstrations = [{"input": p["input"].copy(), "output": p["output"].copy()}
                      for p in puzzle["train"]]
    for pair in puzzle["test"]:
        inferred = _blind_predictions(demonstrations, pair["input"].copy())
        assert inferred and all(np.array_equal(candidate, pair["output"]) for candidate in inferred), (
            "training demonstrations do not determine the test output"
        )


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[g], y[g]) for x, y in zip(a[s], b[s]) for g in ("input", "output")
    ) for s in ("train", "test"))


def _blind_predictions(train, test_input):
    # Infer a traversal from demonstration grids, without using any test answer.
    candidates = []
    for major in ("column", "row"):
        for reverse_major in (False, True):
            for reverse_minor in (False, True):
                def predict(grid):
                    height, width = grid.shape
                    colors, counts = _ranked(grid)
                    stream = np.repeat(colors, counts)
                    result = np.empty_like(grid)
                    positions = ((r, c) for c in (range(width - 1, -1, -1) if reverse_major else range(width))
                                 for r in (range(height - 1, -1, -1) if reverse_minor else range(height))) if major == "column" else (
                                 (r, c) for r in (range(height - 1, -1, -1) if reverse_major else range(height))
                                 for c in (range(width - 1, -1, -1) if reverse_minor else range(width)))
                    for (r, c), color in zip(positions, stream):
                        result[r, c] = color
                    return result
                if all(np.array_equal(predict(pair["input"]), pair["output"]) for pair in train):
                    candidates.append(predict(test_input))
    return candidates


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple(_signature(x["input"]) for s in ("train", "test") for x in p[s])
                for p in puzzles}) == 42
    shapes = set()
    palettes = set()
    remainders = set()
    aligned_counts = set()
    low_share = high_share = singleton = near_rank = clear_gap = 0
    same_hist = changed_hist = mixed_shapes = mixed_palettes = same_endpoints = 0
    equal_hist_output = changed_same_canvas = new_test_color = familiar_shift = 0
    train_no_aligned = train_multi_aligned = test_no_aligned = test_multi_aligned = 0
    near_packed = strongly_scattered = compact = small = large = source_like = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        compact += len(puzzle["train"]) == 2 and len(puzzle["test"]) == 1
        all_pairs = puzzle["train"] + puzzle["test"]
        local_shapes, local_palettes = set(), set()
        train_hists = []
        for position, pair in enumerate(all_pairs):
            grid, output = pair["input"], pair["output"]
            colors, counts = np.unique(grid, return_counts=True)
            # Independent direct construction of the count-and-fill rule.
            ranked = sorted(zip(counts.tolist(), colors.tolist()), reverse=True)
            expected = np.empty_like(grid)
            positions = [(r, c) for c in range(grid.shape[1]) for r in range(grid.shape[0] - 1, -1, -1)]
            stream = [color for count, color in ranked for _ in range(count)]
            for (r, c), color in zip(positions, stream):
                expected[r, c] = color
            assert np.array_equal(output, expected)
            shapes.add(grid.shape)
            palettes.add(len(colors))
            local_shapes.add(grid.shape)
            local_palettes.add(len(colors))
            ordered_counts = [count for count, _ in ranked]
            remainders.add(ordered_counts[0] % grid.shape[0])
            aligned = sum(x % grid.shape[0] == 0 for x in np.cumsum(ordered_counts)[:-1])
            aligned_counts.add(aligned)
            if position < len(puzzle["train"]):
                train_no_aligned += aligned == 0
                train_multi_aligned += aligned >= 2
            else:
                test_no_aligned += aligned == 0
                test_multi_aligned += aligned >= 2
            low_share += ordered_counts[0] / grid.size < .35
            high_share += ordered_counts[0] / grid.size > .65
            singleton += ordered_counts[-1] == 1
            near_rank += any(a - b == 1 for a, b in zip(ordered_counts[:-1], ordered_counts[1:]))
            clear_gap += any(a - b >= 10 for a, b in zip(ordered_counts[:-1], ordered_counts[1:]))
            agreement = np.mean(grid == output)
            near_packed += agreement > .8
            strongly_scattered += agreement < .35
            small += min(grid.shape) <= 7
            large += min(grid.shape) >= 17
            source_like += grid.shape == (10, 10) and len(colors) == 4
        mixed_shapes += len(local_shapes) > 1
        mixed_palettes += len(local_palettes) > 1
        for pair in puzzle["train"]:
            colors, counts = np.unique(pair["input"], return_counts=True)
            train_hists.append((pair["input"].shape, tuple(sorted(zip(colors.tolist(), counts.tolist())))))
        same_hist += len(set(train_hists)) < len(train_hists)
        changed_hist += len(set(train_hists)) > 1
        for i, pair_a in enumerate(puzzle["train"]):
            for j in range(i + 1, len(puzzle["train"])):
                pair_b = puzzle["train"][j]
                if train_hists[i] == train_hists[j]:
                    equal_hist_output += np.array_equal(pair_a["output"], pair_b["output"])
                elif pair_a["input"].shape == pair_b["input"].shape and set(np.unique(pair_a["input"])) == set(np.unique(pair_b["input"])):
                    changed_same_canvas += not np.array_equal(pair_a["output"], pair_b["output"])
        for pair in puzzle["test"]:
            predictions = _blind_predictions(puzzle["train"], pair["input"].copy())
            assert predictions and all(np.array_equal(x, pair["output"]) for x in predictions)
        first_rank = _ranked(puzzle["train"][0]["input"])[0]
        for pair in puzzle["test"]:
            ranks = _ranked(pair["input"])[0]
            same_endpoints += len(ranks) == len(first_rank) and int(ranks[0]) == int(first_rank[0]) and int(ranks[-1]) == int(first_rank[-1]) and not np.array_equal(ranks, first_rank)
            new_test_color += bool(set(ranks) - set(first_rank))
            familiar_shift += int(ranks[0]) in set(first_rank) and int(ranks[0]) != int(first_rank[0])
    assert len(shapes) >= 30 and palettes == {3, 4, 5, 6, 7}
    assert len(remainders) >= 10 and {0, 1, 2}.issubset(aligned_counts)
    assert low_share >= 5 and high_share >= 5 and singleton >= 5
    assert near_rank >= 10 and clear_gap >= 10
    assert train_no_aligned >= 10 and train_multi_aligned >= 1
    assert test_no_aligned >= 5 and test_multi_aligned >= 1
    assert near_packed >= 5 and strongly_scattered >= 5
    assert small >= 5 and large >= 5 and source_like >= 1
    assert same_hist >= 3 and equal_hist_output >= 3 and changed_hist == 42
    assert changed_same_canvas >= 10
    assert mixed_shapes >= 25 and mixed_palettes >= 15 and same_endpoints >= 5
    assert new_test_color >= 5 and familiar_shift >= 5
    assert compact >= 2
    assert _puzzles_equal(generate_puzzle(123456, 7, 3), generate_puzzle(123456, 7, 3))
    assert _puzzles_equal(generate_puzzle(654321), generate_puzzle(654321))
    assert not _puzzles_equal(generate_puzzle(111, 6, 2), generate_puzzle(112, 6, 2))
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        puzzle = generate_puzzle(1000 + i, nt, ne)
        assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for i, (nt, ne) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1100 + i, nt, ne)
        assert (nt is None or len(puzzle["train"]) == nt)
        assert (ne is None or len(puzzle["test"]) == ne)
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(99, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
