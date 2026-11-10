"""Aligned tile decoding and four-by-four rendering for ARC puzzles."""

import numpy as np


_SOURCE_LATENT_KEYS = {
    (5, 5, (1, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 2)),
    (5, 5, (2, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 3)),
    (5, 5, (3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 4)),
    (5, 5, (0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 4, 0, 2, 0, 0, 0, 0)),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _validate_optional_count(value, name, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(rng, num_train, num_test):
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(max(2, test + 1), 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_key(grid):
    return (tuple(int(x) for x in grid.shape), tuple(int(x) for x in grid.flat))


def _latent_key(latent):
    return (int(latent.shape[0]), int(latent.shape[1]), tuple(int(x) for x in latent.flat))


def _render_latent(latent):
    return np.repeat(np.repeat(latent, 4, axis=0), 4, axis=1)


def _decode(grid):
    background = int(grid[0, 0])
    return background, grid[1::2, 1::2]


def _sample_parameters(rng):
    # Orientation and scale are the source convention for every puzzle. The
    # palette is only a training sampling pool; colors have no learned roles.
    pool = tuple(int(x) for x in rng.choice(10, size=int(rng.integers(3, 8)), replace=False))
    source_like = bool(rng.random() < 0.17)
    common_background = 0 if source_like else int(rng.choice([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    fixed_size = source_like or bool(rng.random() < 0.14)
    fixed_shape = (5, 5) if source_like else (int(rng.integers(3, 8)), int(rng.integers(3, 8)))
    size_shift = "same" if fixed_size else str(rng.choice(
        ("free", "free", "free", "small_test", "large_test")))
    return {
        "source_like": source_like,
        "pool": pool,
        "common_background": common_background,
        "vary_background": not source_like and bool(rng.random() < 0.70),
        "fixed_size": fixed_size,
        "fixed_shape": fixed_shape,
        "size_shift": size_shift,
    }


def _shape(rng, parameters, split, index, first_shape):
    if parameters["fixed_size"]:
        return parameters["fixed_shape"]
    shift = parameters["size_shift"]
    if split == "test" and index == 0 and shift == "small_test":
        return (2, 2)
    if split == "test" and index == 0 and shift == "large_test":
        return (7, 7)
    if split == "train" and index == 0:
        return (int(rng.integers(4, 7)), int(rng.integers(4, 7)))
    # These broad overlapping choices make both long thin grids and squares.
    for _ in range(16):
        low, high = (3, 8) if split == "train" and shift == "small_test" else (
            (1, 6) if split == "train" and shift == "large_test" else (1, 8))
        h = int(rng.integers(low, high))
        w = int(rng.integers(low, high))
        if split != "train" or index != 1 or (h, w) != first_shape:
            return h, w
    if shift == "small_test":
        return (3 if first_shape[0] != 3 else 4, first_shape[1])
    if shift == "large_test":
        return (1 if first_shape[0] != 1 else 2, first_shape[1])
    return (first_shape[0] % 7 + 1, first_shape[1])


def _background(rng, parameters, split, index, first_background):
    if not parameters["vary_background"]:
        return parameters["common_background"]
    if split == "train" and index == 1:
        options = [x for x in range(10) if x != first_background]
        return int(rng.choice(options))
    if rng.random() < 0.25:
        return parameters["common_background"]
    return int(rng.integers(0, 10))


def _bounds(rng, h, w, requested):
    if requested == "full":
        return 0, h, 0, w
    if requested == "top":
        top = int(rng.integers(1, min(3, h))) if h > 1 else 0
        return top, h, 0, w
    if requested == "all_sides" and h >= 3 and w >= 3:
        return 1, h - 1, 1, w - 1
    top = int(rng.integers(0, min(3, h)))
    bottom = int(rng.integers(0, min(3, h - top)))
    left = int(rng.integers(0, min(3, w)))
    right = int(rng.integers(0, min(3, w - left)))
    return top, h - bottom, left, w - right


def _random_subset(rng, cells, count):
    count = min(count, len(cells))
    if count == 0:
        return set()
    indexes = rng.choice(len(cells), size=count, replace=False)
    return {cells[int(i)] for i in np.atleast_1d(indexes)}


def _layout(rng, h, w, kind, margin_mode):
    top, bottom, left, right = _bounds(rng, h, w, margin_mode)
    cells = [(r, c) for r in range(top, bottom) for c in range(left, right)]
    ah, aw = bottom - top, right - left
    capacity = len(cells)
    if kind == "full":
        return set(cells)
    if kind == "isolated":
        parity = int(rng.integers(0, 2))
        choices = [(r, c) for r, c in cells if (r + c) % 2 == parity]
        if not choices:
            choices = cells
        maximum = min(len(choices), 7)
        count = int(rng.integers(1, maximum + 1))
        return _random_subset(rng, choices, count)
    if kind == "sparse":
        return _random_subset(rng, cells, int(rng.integers(1, min(4, capacity) + 1)))
    if kind == "dense":
        count = max(1, int(np.ceil(capacity * rng.uniform(0.66, 0.95))))
        return _random_subset(rng, cells, count)
    if kind == "scatter":
        count = int(rng.integers(1, capacity + 1))
        return _random_subset(rng, cells, count)
    if kind == "horizontal" and aw >= 2:
        length = int(rng.integers(2, min(aw, 6) + 1))
        row = int(rng.integers(top, bottom))
        start = int(rng.integers(left, right - length + 1))
        chosen = {(row, c) for c in range(start, start + length)}
    elif kind == "vertical" and ah >= 2:
        length = int(rng.integers(2, min(ah, 6) + 1))
        column = int(rng.integers(left, right))
        start = int(rng.integers(top, bottom - length + 1))
        chosen = {(r, column) for r in range(start, start + length)}
    elif kind == "patch" and ah >= 2 and aw >= 2:
        ph = int(rng.integers(2, min(ah, 4) + 1))
        pw = int(rng.integers(2, min(aw, 4) + 1))
        r0 = int(rng.integers(top, bottom - ph + 1))
        c0 = int(rng.integers(left, right - pw + 1))
        chosen = {(r, c) for r in range(r0, r0 + ph) for c in range(c0, c0 + pw)}
    elif kind == "bent" and ah >= 2 and aw >= 2:
        r0 = int(rng.integers(top, bottom))
        c0 = int(rng.integers(left, right))
        r1 = int(rng.integers(top, bottom))
        c1 = int(rng.integers(left, right))
        chosen = {(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)}
        chosen |= {(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)}
    elif kind == "hollow" and ah >= 3 and aw >= 3:
        ph = int(rng.integers(3, ah + 1))
        pw = int(rng.integers(3, aw + 1))
        r0 = int(rng.integers(top, bottom - ph + 1))
        c0 = int(rng.integers(left, right - pw + 1))
        chosen = {(r, c) for r in range(r0, r0 + ph) for c in range(c0, c0 + pw)
                  if r in (r0, r0 + ph - 1) or c in (c0, c0 + pw - 1)}
    elif kind == "groups" and capacity >= 5:
        first = cells[int(rng.integers(0, capacity))]
        far = [p for p in cells if abs(p[0] - first[0]) + abs(p[1] - first[1]) >= 3]
        second = far[int(rng.integers(0, len(far)))] if far else cells[-1]
        chosen = {first, second}
        for center in (first, second):
            neighbors = [p for p in cells if abs(p[0] - center[0]) + abs(p[1] - center[1]) == 1]
            chosen |= _random_subset(rng, neighbors, min(len(neighbors), int(rng.integers(1, 3))))
    else:
        return _random_subset(rng, cells, int(rng.integers(1, capacity + 1)))
    # Additions vary density without replacing the deliberately constructed core.
    if kind not in ("hollow", "bent") and rng.random() < 0.33:
        remaining = [p for p in cells if p not in chosen]
        if remaining:
            chosen |= _random_subset(rng, remaining, int(rng.integers(1, min(4, len(remaining)) + 1)))
    return chosen


def _anchor_layout(rng, h, w):
    # A multicolor demonstration shows adjacency, vacant sampled tiles, and
    # the first and last lattice rows. Other demonstrations may be very sparse.
    c = int(rng.integers(0, w - 1))
    points = {(0, c), (0, c + 1), (h - 1, int(rng.integers(0, w)))}
    possible = [(r, col) for r in range(h) for col in range(w) if (r, col) not in points]
    points |= _random_subset(rng, possible, int(rng.integers(1, min(5, len(possible)) + 1)))
    return points


def _choose_colors(rng, split, background, positions, pool, trained_colors, source_style):
    count = len(positions)
    if source_style:
        familiar = [x for x in sorted(trained_colors) if x != background]
        novel = [x for x in range(10) if x != background and x not in trained_colors]
        selected = []
        if familiar:
            selected.append(int(rng.choice(familiar)))
        if novel:
            selected.append(int(rng.choice(novel)))
        remaining = [x for x in range(10) if x != background and x not in selected]
        selected += [int(x) for x in rng.choice(remaining, size=5 - len(selected), replace=False)]
        return selected
    if split == "train":
        candidates = [x for x in pool if x != background]
        mode = "familiar"
    else:
        familiar = [x for x in sorted(trained_colors) if x != background]
        novel = [x for x in range(10) if x != background and x not in trained_colors]
        mode = str(rng.choice(("familiar", "subset", "one_new", "several_new")))
        if not familiar:
            mode = "several_new"
        if mode in ("familiar", "subset") or not novel:
            candidates = familiar or novel
        elif mode == "one_new":
            candidates = familiar + [int(rng.choice(novel))]
        else:
            candidates = familiar + novel
    if split == "train" and count >= 3 and rng.random() < 0.22:
        requested = min(count, len(candidates), int(rng.integers(5, 8)))
    elif count <= len(candidates) and rng.random() < 0.27:
        requested = count
    else:
        requested = int(rng.integers(1, min(count, len(candidates), 7) + 1))
    if split == "test" and mode == "subset" and len(candidates) > 1:
        requested = min(requested, len(candidates) - 1)
    if split == "test" and mode == "several_new" and len(novel) >= 2 and count >= 2:
        new_count = min(int(rng.integers(2, min(3, len(novel), count) + 1)), requested)
        requested = max(requested, new_count)
        new_colors = [int(x) for x in rng.choice(novel, size=new_count, replace=False)]
        rest = [x for x in candidates if x not in new_colors]
        selected = new_colors + [int(x) for x in rng.choice(rest, size=requested - new_count, replace=False)]
    elif split == "test" and mode == "one_new" and novel:
        new_color = candidates[-1]
        rest = [x for x in candidates if x != new_color]
        selected = [new_color] + [int(x) for x in rng.choice(rest, size=requested - 1, replace=False)]
    else:
        selected = [int(x) for x in rng.choice(candidates, size=requested, replace=False)]
    if split == "train" and count >= 3 and len(selected) < 2:
        selected = [int(x) for x in rng.choice(candidates, size=2, replace=False)]
    return selected


def _make_pair(rng, shape, background, positions, colors):
    h, w = shape
    latent = np.full((h, w), background, dtype=np.int8)
    ordered = sorted(positions)
    rng.shuffle(ordered)
    palette = list(colors)
    rng.shuffle(palette)
    for i, (r, c) in enumerate(ordered):
        latent[r, c] = palette[i] if i < len(palette) else int(rng.choice(palette))
    inp = np.full((2 * h, 2 * w), background, dtype=np.int8)
    inp[1::2, 1::2] = latent
    return {"input": inp, "output": _render_latent(latent).astype(np.int8)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = _sample_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_shape = None
    first_background = None
    trained_colors = set()
    maximum_seed = np.iinfo(np.int64).max
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            child = np.random.default_rng(int(rng.integers(0, maximum_seed, dtype=np.int64)))
            for attempt in range(40):
                shape = _shape(child, parameters, split, index, first_shape)
                h, w = shape
                background = _background(child, parameters, split, index, first_background)
                source_style = split == "test" and parameters["size_shift"] in ("free", "same") and child.random() < (0.28 if index == 0 else 0.14)
                if source_style and not parameters["fixed_size"]:
                    h, w = int(child.integers(5, 8)), int(child.integers(5, 8))
                    shape = (h, w)
                source_style = source_style and h >= 5 and w >= 5
                if source_style:
                    parity = int(child.integers(0, 2))
                    positions = _random_subset(child, [(r, c) for r in range(1, h) for c in range(w)
                                                       if (r + c) % 2 == parity], 5)
                elif split == "train" and index == 0:
                    positions = _anchor_layout(child, h, w)
                elif parameters["source_like"]:
                    top = int(child.integers(0, 2)) if split == "test" else 0
                    cells = [(r, c) for r in range(top, h) for c in range(w)]
                    positions = _random_subset(child, cells, int(child.integers(4, 10)))
                else:
                    kind = str(child.choice(("sparse", "isolated", "scatter", "horizontal", "vertical",
                                             "patch", "bent", "hollow", "groups", "dense", "full")))
                    margin = str(child.choice(("free", "free", "free", "top", "all_sides", "full")))
                    if kind == "full":
                        margin = "full"
                    positions = _layout(child, h, w, kind, margin)
                colors = _choose_colors(child, split, background, positions, parameters["pool"],
                                        trained_colors, source_style)
                if split == "train" and index == 0:
                    candidates = [x for x in parameters["pool"] if x != background]
                    if len(colors) < 2:
                        colors = [int(x) for x in child.choice(candidates, size=2, replace=False)]
                pair = _make_pair(child, shape, background, positions, colors)
                latent = pair["input"][1::2, 1::2]
                key = _grid_key(pair["input"])
                if key not in seen and (background != 0 or _latent_key(latent) not in _SOURCE_LATENT_KEYS):
                    puzzle[split].append(pair)
                    seen.add(key)
                    if split == "train":
                        trained_colors.update(int(x) for x in np.unique(latent) if int(x) != background)
                    if first_shape is None:
                        first_shape = shape
                        first_background = background
                    break
            else:
                raise RuntimeError("could not make a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    train_latents = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair schema")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be 2D ndarray")
                _require(all(1 <= d <= 30 for d in grid.shape), "grid dimensions outside 1..30")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "illegal ARC color")
            inp, out = pair["input"], pair["output"]
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(inp.shape[0] % 2 == 0 and inp.shape[1] % 2 == 0, "input must have even dimensions")
            background, latent = _decode(inp)
            _require(bool(np.all(inp[0::2, 0::2] == background)
                          and np.all(inp[0::2, 1::2] == background)
                          and np.all(inp[1::2, 0::2] == background)), "unclean spacer lattice")
            _require(bool(np.any(latent != background)), "a pair needs a signal")
            _require(out.shape == (4 * latent.shape[0], 4 * latent.shape[1]), "output size mismatch")
            _require(np.array_equal(out, _render_latent(latent)), "incorrect 4x4 block rendering")
            _require(out.shape == (2 * inp.shape[0], 2 * inp.shape[1]), "wrong dimension relationship")
            _require(background != 0 or _latent_key(latent) not in _SOURCE_LATENT_KEYS, "source pair copied")
            if split == "train":
                train_latents.append((background, latent))
    _require(any(len({int(x) for x in np.unique(latent) if int(x) != bg}) >= 2
                 and np.any(latent == bg) and
                 (np.any((latent[:, :-1] != bg) & (latent[:, 1:] != bg)) or
                  np.any((latent[:-1, :] != bg) & (latent[1:, :] != bg)))
                 for bg, latent in train_latents), "training lacks a clear multicolor lattice example")


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    def component_sizes(mask):
        pending = {tuple(int(x) for x in cell) for cell in np.argwhere(mask)}
        sizes = []
        while pending:
            group = [pending.pop()]
            for row, column in group:
                for neighbor in ((row - 1, column), (row + 1, column),
                                 (row, column - 1), (row, column + 1)):
                    if neighbor in pending:
                        pending.remove(neighbor)
                        group.append(neighbor)
            sizes.append(len(group))
        return sizes

    automatic = generate_puzzle(12345)
    _require(_puzzles_equal(automatic, generate_puzzle(12345)), "automatic generation is not deterministic")
    explicit = generate_puzzle(54321, 6, 3)
    _require((len(explicit["train"]), len(explicit["test"])) == (6, 3), "explicit counts ignored")
    _require(_puzzles_equal(explicit, generate_puzzle(54321, 6, 3)), "explicit generation is not deterministic")
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(60000 + i, *counts)
        _require((len(p["train"]), len(p["test"])) == counts, "boundary counts ignored")
    for i, (args, expected, split) in enumerate((
        ((None, 1), 1, "test"), ((None, 4), 4, "test"),
        ((2, None), 2, "train"), ((10, None), 10, "train"))):
        _require(len(generate_puzzle(61000 + i, *args)[split]) == expected, "partial count ignored")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(62000, *args)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {args}")
    train_counts, test_counts, shapes, layouts, occupied, backgrounds = set(), set(), set(), set(), set(), set()
    palette_sizes, overlap_modes = set(), set()
    feature_counts = {name: 0 for name in (
        "small", "moderate", "dense", "full", "isolated", "all_distinct", "source_style_first_test",
        "top_train", "active_test", "top_test", "all_margins", "bottom_margin",
        "left_margin", "right_margin", "internal_row", "internal_column",
        "horizontal_run", "vertical_run", "patch", "hollow", "groups",
        "single_axis", "five_colors", "zero_signal", "zero_background_puzzle",
        "fixed_size", "varied_size", "varied_train_size", "varied_test_size",
        "fixed_background", "varied_background", "source_like_puzzle", "combined_axes",
        "novel_one", "novel_many", "familiar_test", "test_subset",
        "novel_familiar", "combined_test_shift", "test_larger", "test_smaller")}
    keys = set()
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(tuple(tuple(_grid_key(p["input"]) for p in puzzle[s]) for s in ("train", "test")))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_shapes, test_shapes, pair_backgrounds, train_colors = [], [], [], set()
        test_colors = set()
        has_zero_signal = False
        combined_shift_candidates = []
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                inp, out = pair["input"], pair["output"]
                bg, latent = _decode(inp)
                mask = latent != bg
                count = int(np.count_nonzero(mask))
                colors = {int(x) for x in np.unique(latent[mask])}
                _require(np.array_equal(out, np.kron(latent, np.ones((4, 4), dtype=np.int8))), "independent rendering check failed")
                spacer = np.full_like(inp, bg)
                spacer[1::2, 1::2] = latent
                _require(np.array_equal(inp, spacer), "independent spacer check failed")
                _require(bg != 0 or _latent_key(latent) not in _SOURCE_LATENT_KEYS, "source copy in self-test")
                shapes.add(latent.shape)
                layouts.add((latent.shape, tuple(bool(x) for x in mask.flat)))
                occupied.add(count)
                backgrounds.add(bg)
                palette_sizes.add(len(colors))
                pair_backgrounds.append(bg)
                (train_shapes if split == "train" else test_shapes).append(latent.shape)
                (train_colors if split == "train" else test_colors).update(colors)
                feature_counts["small"] += count <= 4
                feature_counts["moderate"] += 0.20 <= count / latent.size <= 0.60
                feature_counts["dense"] += count / latent.size >= 0.70
                feature_counts["full"] += count == latent.size
                feature_counts["all_distinct"] += count == len(colors)
                feature_counts["single_axis"] += min(latent.shape) == 1
                feature_counts["five_colors"] += len(colors) >= 5
                feature_counts["zero_signal"] += bg != 0 and 0 in colors
                has_zero_signal |= bg != 0 and 0 in colors
                feature_counts["all_margins"] += latent.shape[0] >= 3 and latent.shape[1] >= 3 and not (
                    np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]))
                feature_counts["bottom_margin"] += not np.any(mask[-1])
                feature_counts["left_margin"] += not np.any(mask[:, 0])
                feature_counts["right_margin"] += not np.any(mask[:, -1])
                feature_counts["internal_row"] += any(not np.any(mask[r]) for r in range(1, mask.shape[0] - 1))
                feature_counts["internal_column"] += any(not np.any(mask[:, c]) for c in range(1, mask.shape[1] - 1))
                feature_counts["horizontal_run"] += bool(mask.shape[1] >= 3 and np.any(
                    mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]))
                feature_counts["vertical_run"] += bool(mask.shape[0] >= 3 and np.any(
                    mask[:-2, :] & mask[1:-1, :] & mask[2:, :]))
                feature_counts["patch"] += bool(mask.shape[0] >= 2 and mask.shape[1] >= 2 and np.any(
                    mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]))
                feature_counts["hollow"] += any(
                    mask[r, c] and mask[r, c + 1] and mask[r, c + 2]
                    and mask[r + 1, c] and not mask[r + 1, c + 1] and mask[r + 1, c + 2]
                    and mask[r + 2, c] and mask[r + 2, c + 1] and mask[r + 2, c + 2]
                    for r in range(mask.shape[0] - 2) for c in range(mask.shape[1] - 2))
                feature_counts["groups"] += sum(size >= 2 for size in component_sizes(mask)) >= 2
                isolated = not (np.any(mask[:, :-1] & mask[:, 1:]) or np.any(mask[:-1, :] & mask[1:, :]))
                feature_counts["isolated"] += isolated and count >= 2
                if split == "train":
                    feature_counts["top_train"] += not np.any(mask[0])
                else:
                    feature_counts["active_test"] += bool(np.any(mask[0]))
                    feature_counts["top_test"] += not np.any(mask[0])
                    feature_counts["source_style_first_test"] += index == 0 and count == 5 and len(colors) == 5 and isolated and not np.any(mask[0])
                    feature_counts["test_subset"] += bool(colors < train_colors)
                    feature_counts["novel_familiar"] += bool((colors - train_colors) and (colors & train_colors))
                    combined_shift_candidates.append(bool(colors - train_colors) and not np.any(mask[0]) and bg != 0)
        feature_counts["fixed_size"] += len(set(train_shapes + test_shapes)) == 1
        feature_counts["varied_size"] += len(set(train_shapes + test_shapes)) > 1
        feature_counts["varied_train_size"] += len(set(train_shapes)) > 1
        feature_counts["varied_test_size"] += len(set(test_shapes)) > 1
        feature_counts["fixed_background"] += len(set(pair_backgrounds)) == 1
        feature_counts["varied_background"] += len(set(pair_backgrounds)) > 1
        feature_counts["zero_background_puzzle"] += set(pair_backgrounds) == {0}
        feature_counts["source_like_puzzle"] += set(pair_backgrounds) == {0} and set(train_shapes + test_shapes) == {(5, 5)}
        feature_counts["combined_axes"] += len(set(train_shapes + test_shapes)) > 1 and len(set(pair_backgrounds)) > 1 and has_zero_signal
        feature_counts["combined_test_shift"] += len(set(train_shapes + test_shapes)) > 1 and any(combined_shift_candidates)
        feature_counts["test_larger"] += any(h * w > max(a * b for a, b in train_shapes) for h, w in test_shapes)
        feature_counts["test_smaller"] += any(h * w < min(a * b for a, b in train_shapes) for h, w in test_shapes)
        novel = test_colors - train_colors
        overlap_modes.add((len(novel), bool(test_colors & train_colors)))
        feature_counts["novel_one"] += len(novel) == 1
        feature_counts["novel_many"] += len(novel) >= 2
        feature_counts["familiar_test"] += len(novel) == 0
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity too low")
    _require(len(keys) >= 40 and len(shapes) >= 18 and len(layouts) >= 100, "structural diversity too low")
    _require(len(occupied) >= 12 and len(backgrounds) >= 7 and len(palette_sizes) >= 5, "property diversity too low")
    _require(all(value > 0 for value in feature_counts.values()), f"missing review feature: {feature_counts}")
    _require(len(overlap_modes) >= 3, "palette overlap lacks variation")
