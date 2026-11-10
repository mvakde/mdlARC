"""Whole-grid, four-quadrant mirror puzzles.

The reflection rule is the only puzzle-wide convention.  Colors and all seed
geometry are local to a pair, as they are in the source task.
"""

import numpy as np


_SOURCE_INPUTS = (
    np.array([[0, 6, 0, 0], [4, 6, 0, 3], [4, 6, 3, 0],
              [4, 3, 3, 0]], dtype=np.int8),
    np.array([[3, 0, 0], [3, 4, 0], [3, 4, 2]], dtype=np.int8),
    np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 2, 1, 0, 9, 0, 0, 0],
        [0, 2, 1, 0, 9, 0, 0, 0],
        [0, 2, 1, 0, 9, 1, 1, 1],
        [9, 2, 0, 0, 9, 0, 0, 0],
        [9, 2, 0, 0, 9, 0, 0, 9],
        [1, 2, 0, 0, 9, 0, 0, 9],
        [9, 9, 0, 0, 9, 0, 0, 9],
    ], dtype=np.int8),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _mirror_seed(grid):
    lower = np.concatenate((grid[:, ::-1], grid), axis=1)
    return np.concatenate((lower[::-1, :], lower), axis=0)


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _is_source(grid):
    return any(grid.shape == source.shape and np.array_equal(grid, source)
               for source in _SOURCE_INPUTS)


def _orientation_rich(grid):
    return (not np.array_equal(grid, grid[:, ::-1])
            and not np.array_equal(grid, grid[::-1, :])
            and not np.array_equal(grid, grid[::-1, ::-1]))


def _infer_quadrants(train):
    """Identify each quadrant from demonstrations without using test labels."""
    possible = [set(range(4)) for _ in range(4)]
    for pair in train:
        inp, out = pair["input"], pair["output"]
        h, w = inp.shape
        variants = (inp, inp[:, ::-1], inp[::-1, :], inp[::-1, ::-1])
        quadrants = (out[:h, :w], out[:h, w:], out[h:, :w], out[h:, w:])
        for index, quadrant in enumerate(quadrants):
            possible[index] &= {variant for variant in range(4)
                                if np.array_equal(quadrant, variants[variant])}
    _require(all(len(options) == 1 for options in possible), "ambiguous training quadrant mapping")
    return tuple(next(iter(options)) for options in possible)


def _counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _shape(rng, split, index, first_shape, source_like):
    if source_like:
        if split == "train":
            return (3, 3) if index % 2 == 0 else (4, 4)
        side = int(rng.integers(6, 16))
        return side, side
    if split == "train" and index == 0:
        # This demonstration always identifies both reflection axes.
        choices = ((3, 3), (4, 4), (5, 5), (3, 4), (4, 3),
                   (4, 5), (5, 4), (6, 6))
        return choices[int(rng.integers(len(choices)))]
    if split == "train" and index == 1:
        choices = [(3, 3), (4, 4), (5, 5), (3, 5), (5, 3),
                   (2, 6), (6, 2), (5, 7), (7, 5), (6, 6),
                   (4, 8), (8, 4)]
        choices = [shape for shape in choices if shape != first_shape]
        return choices[int(rng.integers(len(choices)))]
    if split == "train":
        category = int(rng.choice(4, p=[0.20, 0.38, 0.31, 0.11]))
        if category == 0:
            side = int(rng.integers(3, 10))
            return side, side
        if category == 1:
            return int(rng.integers(2, 11)), int(rng.integers(2, 11))
        if category == 2:
            return int(rng.integers(3, 13)), int(rng.integers(3, 13))
        return (1, int(rng.integers(3, 13))) if rng.random() < 0.5 else (int(rng.integers(3, 13)), 1)
    category = int(rng.choice(5, p=[0.42, 0.20, 0.18, 0.11, 0.09]))
    if category == 0:
        return int(rng.integers(6, 16)), int(rng.integers(6, 16))
    if category == 1:
        return int(rng.integers(2, 7)), int(rng.integers(2, 7))
    if category == 2:
        return int(rng.integers(2, 16)), int(rng.integers(2, 16))
    if category == 3:
        side = int(rng.integers(2, 16))
        return side, side
    return (1, int(rng.integers(2, 16))) if rng.random() < 0.5 else (int(rng.integers(2, 16)), 1)


def _palette(rng, first_train=False, prior_train=()):
    sizes = np.array([1, 2, 3, 4, 5, 6])
    weights = [0.035, 0.15, 0.29, 0.29, 0.17, 0.065]
    size = int(rng.choice(sizes[2:] if first_train else sizes,
                          p=np.array(weights[2:]) / sum(weights[2:]) if first_train else weights))
    if prior_train and rng.random() < 0.23:
        return np.array(prior_train[int(rng.integers(len(prior_train)))], dtype=np.int8)
    include_zero = rng.random() < 0.76
    if include_zero:
        colors = [0] + [int(v) for v in rng.choice(np.arange(1, 10), size=size - 1, replace=False)]
    else:
        colors = [int(v) for v in rng.choice(np.arange(1, 10), size=size, replace=False)]
    return np.array(colors, dtype=np.int8)


def _other_color(rng, palette, current):
    options = palette[palette != current]
    return int(rng.choice(options)) if len(options) else int(current)


def _paint_shape(rng, grid, color, outline=False):
    height, width = grid.shape
    r0 = int(rng.integers(height))
    c0 = int(rng.integers(width))
    r1 = int(rng.integers(r0 + 1, height + 1))
    c1 = int(rng.integers(c0 + 1, width + 1))
    if outline and r1 - r0 >= 2 and c1 - c0 >= 2:
        grid[r0, c0:c1] = color
        grid[r1 - 1, c0:c1] = color
        grid[r0:r1, c0] = color
        grid[r0:r1, c1 - 1] = color
    else:
        grid[r0:r1, c0:c1] = color


def _layout(rng, shape, palette, mode):
    height, width = shape
    dominant = int(rng.choice(palette))
    if 0 in palette and rng.random() < 0.55:
        dominant = 0
    grid = np.full(shape, dominant, dtype=np.int8)
    area = height * width

    if mode == "texture":
        strength = float(rng.uniform(0.20, 0.82))
        grid = rng.choice(palette, size=shape).astype(np.int8)
        grid[rng.random(shape) < strength] = dominant
    elif mode == "sparse":
        for _ in range(int(rng.integers(1, max(3, area // 7 + 2)))):
            r, c = int(rng.integers(height)), int(rng.integers(width))
            grid[r, c] = _other_color(rng, palette, dominant)
            if rng.random() < 0.25:
                grid[r, c:min(width, c + int(rng.integers(2, 5)))] = grid[r, c]
    elif mode == "bars":
        for _ in range(int(rng.integers(1, 6))):
            color = _other_color(rng, palette, dominant)
            if rng.random() < 0.5:
                c = int(rng.integers(width))
                r0 = int(rng.integers(height)) if rng.random() < 0.45 else 0
                r1 = int(rng.integers(r0 + 1, height + 1)) if r0 else height
                grid[r0:r1, c] = color
            else:
                r = int(rng.integers(height))
                c0 = int(rng.integers(width)) if rng.random() < 0.45 else 0
                c1 = int(rng.integers(c0 + 1, width + 1)) if c0 else width
                grid[r, c0:c1] = color
    elif mode == "blocks":
        for _ in range(int(rng.integers(1, 5))):
            _paint_shape(rng, grid, _other_color(rng, palette, dominant),
                         outline=rng.random() < 0.46)
    elif mode == "strokes":
        for _ in range(int(rng.integers(1, 4))):
            r, c = int(rng.integers(height)), int(rng.integers(width))
            color = _other_color(rng, palette, dominant)
            for _ in range(int(rng.integers(2, 6))):
                if rng.random() < 0.5:
                    end = int(rng.integers(height))
                    grid[min(r, end):max(r, end) + 1, c] = color
                    r = end
                else:
                    end = int(rng.integers(width))
                    grid[r, min(c, end):max(c, end) + 1] = color
                    c = end
    elif mode == "repeat":
        if rng.random() < 0.5:
            period = int(rng.integers(1, min(5, height) + 1))
            tile = rng.choice(palette, size=(period, width)).astype(np.int8)
            grid = tile[np.arange(height) % period, :].copy()
        else:
            period = int(rng.integers(1, min(5, width) + 1))
            tile = rng.choice(palette, size=(height, period)).astype(np.int8)
            grid = tile[:, np.arange(width) % period].copy()
    elif mode == "objects":
        object_color = _other_color(rng, palette, dominant)
        for _ in range(int(rng.integers(2, 7))):
            r, c = int(rng.integers(height)), int(rng.integers(width))
            rh = int(rng.integers(1, min(4, height - r) + 1))
            cw = int(rng.integers(1, min(4, width - c) + 1))
            grid[r:r + rh, c:c + cw] = object_color if rng.random() < 0.7 else rng.choice(palette)
    elif mode == "patches":
        for _ in range(int(rng.integers(2, 7))):
            _paint_shape(rng, grid, int(rng.choice(palette)), outline=False)
    else:  # uniform or a single deliberate irregularity
        if rng.random() < 0.62 and len(palette) > 1:
            r, c = int(rng.integers(height)), int(rng.integers(width))
            grid[r, c] = _other_color(rng, palette, dominant)
    return grid


def _set_edge(rng, grid, palette, axis, style):
    edge = grid[0, :] if axis == 0 else grid[:, 0]
    nonzero = palette[palette != 0]
    if style == "blank":
        edge[:] = 0
    elif style == "solid" and len(nonzero):
        edge[:] = rng.choice(nonzero)
    elif style == "colorful" and len(nonzero) >= 2 and len(edge) >= 2:
        edge[:] = rng.choice(nonzero, size=len(edge))
        edge[0], edge[-1] = nonzero[:2]
    elif style == "mixed" and 0 in palette and len(nonzero) and len(edge) >= 2:
        edge[:] = rng.choice(palette, size=len(edge))
        edge[0], edge[-1] = 0, nonzero[0]


def _make_grid(rng, shape, palette, mode, edge_styles, symmetry):
    grid = _layout(rng, shape, palette, mode)
    if symmetry == "none":
        _set_edge(rng, grid, palette, 0, edge_styles[0])
        _set_edge(rng, grid, palette, 1, edge_styles[1])
        # The corner is an independent choice whenever it does not contradict
        # an intentionally continuous edge.
        if edge_styles[0] not in ("blank", "solid", "colorful") and edge_styles[1] not in ("blank", "solid", "colorful"):
            if rng.random() < 0.5:
                grid[0, 0] = rng.choice(palette)
    else:
        if symmetry == "lr":
            grid[:, :grid.shape[1] // 2] = grid[:, grid.shape[1] - grid.shape[1] // 2:][:, ::-1]
        elif symmetry == "tb":
            grid[:grid.shape[0] // 2, :] = grid[grid.shape[0] - grid.shape[0] // 2:, :][::-1, :]
        else:
            h, w = grid.shape
            for r in range(h):
                for c in range(w):
                    if (r, c) < (h - 1 - r, w - 1 - c):
                        grid[r, c] = grid[h - 1 - r, w - 1 - c]
    return grid


def _sample_grid(rng, split, index, shape, palette):
    modes = ("bars", "blocks", "strokes", "repeat", "objects",
             "sparse", "patches", "texture", "uniform")
    if split == "train" and index == 0:
        modes = modes[:8]
    mode = modes[int(rng.integers(len(modes)))]
    edge_styles = tuple(rng.choice(("natural", "blank", "solid", "colorful", "mixed"),
                                   p=(0.31, 0.19, 0.21, 0.16, 0.13)) for _ in range(2))
    symmetry = "none" if split == "train" and index == 0 else str(rng.choice(
        ("none", "lr", "tb", "rot"), p=(0.73, 0.10, 0.10, 0.07)))
    grid = _make_grid(rng, shape, palette, mode, edge_styles, symmetry)
    if split == "train" and index == 0:
        # Bound the only constructive repair.  An informative first example
        # remains possible even when a sampled motif is accidentally symmetric.
        if not _orientation_rich(grid) or np.array_equal(_mirror_seed(grid), np.repeat(np.repeat(grid, 2, 0), 2, 1)):
            for r, c in ((0, 0), (0, 1), (1, 0)):
                grid[r, c] = _other_color(rng, palette, grid[r, c])
    return grid


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    # The source supports no variable semantic convention: every puzzle uses
    # the same mirror rule.  Pair RNG streams vary local properties freely.
    source_like = bool(rng.random() < 0.15)
    streams = np.random.SeedSequence(int(rng.integers(0, 2**63))).spawn(num_train + num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_palettes = []
    first_shape = None
    for global_index in range(num_train + num_test):
        split = "train" if global_index < num_train else "test"
        index = global_index if split == "train" else global_index - num_train
        pair_rng = np.random.default_rng(streams[global_index])
        for attempt in range(32):
            shape = _shape(pair_rng, split, index, first_shape, source_like)
            palette = _palette(pair_rng, split == "train" and index == 0,
                               train_palettes if split == "test" else ())
            if split == "train" and index == 1 and set(palette) == set(train_palettes[0]):
                continue
            grid = _sample_grid(pair_rng, split, index, shape, palette)
            if split == "train" and index == 0 and not _orientation_rich(grid):
                continue
            key = _key(grid)
            if key not in seen and not _is_source(grid):
                break
        else:
            # A deterministic fallback avoids a rejection loop on tiny seeds.
            shape = (4 + global_index % 3, 5 + global_index % 4)
            grid = np.zeros(shape, dtype=np.int8)
            grid[0, 0] = 1 + global_index % 9
            grid[1, 2] = 1 + (global_index + 3) % 9
            key = _key(grid)
            if key in seen or _is_source(grid):
                raise AssertionError("fallback grid collision")
        seen.add(key)
        if split == "train":
            train_palettes.append(tuple(int(v) for v in np.unique(grid)))
            if index == 0:
                first_shape = grid.shape
        puzzle[split].append({"input": grid, "output": _mirror_seed(grid)})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split schema")
    nt, nx = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nx <= 4 and nt > nx, "pair counts")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            for role in ("input", "output"):
                grid = pair[role]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
                _require(all(1 <= d <= 30 for d in grid.shape), "grid shape")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "grid colors")
            inp, out = pair["input"], pair["output"]
            h, w = inp.shape
            _require(h <= 15 and w <= 15 and out.shape == (2 * h, 2 * w), "dimension doubling")
            _require(np.array_equal(out[h:, w:], inp), "unchanged lower-right seed")
            _require(np.array_equal(out[h:, :w], inp[:, ::-1]), "lower-left reflection")
            _require(np.array_equal(out[:h, w:], inp[::-1, :]), "upper-right reflection")
            _require(np.array_equal(out[:h, :w], inp[::-1, ::-1]), "upper-left rotation")
            _require(np.array_equal(np.unique(inp), np.unique(out)), "palette preservation")
            ik, pk = _key(inp), (_key(inp), _key(out))
            _require(ik not in seen_inputs and pk not in seen_pairs, "duplicate pair")
            _require(not _is_source(inp), "source pair copied")
            seen_inputs.add(ik)
            seen_pairs.add(pk)
    first, second = (p["input"] for p in puzzle["train"][:2])
    _require(min(first.shape) >= 2 and _orientation_rich(first), "first training example is uninformative")
    _require(first.shape != second.shape, "training shapes do not vary")
    _require(set(np.unique(first)) != set(np.unique(second)), "training palettes do not vary")
    _require(not np.array_equal(puzzle["train"][0]["output"],
                                np.repeat(np.repeat(first, 2, axis=0), 2, axis=1)),
             "first training example resembles cell enlargement")
    mapping = _infer_quadrants(puzzle["train"])
    for pair in puzzle["test"]:
        inp = pair["input"]
        options = (inp, inp[:, ::-1], inp[::-1, :], inp[::-1, ::-1])
        predicted = np.block([[options[mapping[0]], options[mapping[1]]],
                              [options[mapping[2]], options[mapping[3]]]])
        _require(np.array_equal(predicted, pair["output"]), "test not predicted from training")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40, "cross-seed puzzle diversity")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    features = {name: 0 for name in ("rectangle", "square", "line", "small_test",
                                    "test_not_larger", "one_color", "two_color",
                                    "zero_absent", "nonzero_dominant", "symmetric",
                                    "sparse", "dense", "full_top", "full_left",
                                    "blank_top", "blank_left", "mixed_top",
                                    "mixed_left", "corner_zero", "corner_nonzero",
                                    "repeated_rows", "repeated_columns", "palette_reuse",
                                    "palette_change", "palette_disjoint",
                                    "palette_partial", "single_mark", "long_run",
                                    "colorful_top", "colorful_left", "source_like",
                                    "rect_nonzero_full_top", "small_symmetric_test")}
    shapes, structures, palettes = set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_shapes = [pair["input"].shape for pair in puzzle["train"]]
        train_palettes = [set(np.unique(pair["input"])) for pair in puzzle["train"]]
        features["source_like"] += int(
            all(shape in ((3, 3), (4, 4)) for shape in train_shapes)
            and all(pair["input"].shape[0] == pair["input"].shape[1] >= 6
                    for pair in puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, out = pair["input"], pair["output"]
                h, w = grid.shape
                # Independently inspect every output cell by reflected index.
                for r in range(2 * h):
                    for c in range(2 * w):
                        source_r = h - 1 - r if r < h else r - h
                        source_c = w - 1 - c if c < w else c - w
                        _require(int(out[r, c]) == int(grid[source_r, source_c]),
                                 "independent coordinate prediction")
                colors, counts = np.unique(grid, return_counts=True)
                dominant = int(colors[np.argmax(counts)])
                palette = tuple(map(int, colors))
                palettes.add(palette)
                shapes.add(grid.shape)
                labels = {}
                normalized = []
                for cell in grid.flat:
                    v = int(cell)
                    if v not in labels:
                        labels[v] = len(labels)
                    normalized.append(labels[v])
                structures.add((grid.shape, tuple(normalized)))
                features["rectangle" if h != w and min(h, w) > 1 else "square" if h == w else "line"] += 1
                features["one_color" if len(colors) == 1 else "two_color"] += int(len(colors) <= 2)
                features["zero_absent"] += int(0 not in colors)
                features["nonzero_dominant"] += int(dominant != 0)
                features["symmetric"] += int(not _orientation_rich(grid))
                density = float(np.count_nonzero(grid != dominant)) / grid.size
                features["sparse"] += int(density <= 0.12)
                features["dense"] += int(density >= 0.48)
                features["full_top"] += int(bool(np.all(grid[0] != 0)))
                features["full_left"] += int(bool(np.all(grid[:, 0] != 0)))
                features["blank_top"] += int(bool(np.all(grid[0] == 0)))
                features["blank_left"] += int(bool(np.all(grid[:, 0] == 0)))
                features["mixed_top"] += int(len(np.unique(grid[0])) >= 2)
                features["mixed_left"] += int(len(np.unique(grid[:, 0])) >= 2)
                features["corner_zero"] += int(grid[0, 0] == 0)
                features["corner_nonzero"] += int(grid[0, 0] != 0)
                features["repeated_rows"] += int(len(np.unique(grid, axis=0)) < h)
                features["repeated_columns"] += int(len(np.unique(grid, axis=1)) < w)
                features["single_mark"] += int(np.count_nonzero(grid != dominant) == 1)
                features["colorful_top"] += int(len(np.unique(grid[0])) >= 2 and np.all(grid[0] != 0))
                features["colorful_left"] += int(len(np.unique(grid[:, 0])) >= 2 and np.all(grid[:, 0] != 0))
                run = 0
                for line in list(grid) + list(grid.T):
                    old, length = -1, 0
                    for cell in line:
                        value = int(cell)
                        length = length + 1 if value == old else 1
                        old = value
                        if value != 0:
                            run = max(run, length)
                features["long_run"] += int(run >= 4)
                if split == "test":
                    features["small_test"] += int(max(h, w) <= 5)
                    features["test_not_larger"] += int(h * w <= max(a * b for a, b in train_shapes))
                    features["palette_reuse"] += int(any(set(colors) == p for p in train_palettes))
                    features["palette_change"] += int(all(set(colors) != p for p in train_palettes))
                    features["palette_disjoint"] += int(set(colors).isdisjoint(train_palettes[0]))
                    features["palette_partial"] += int(bool(set(colors) & train_palettes[0])
                                                       and set(colors) != train_palettes[0])
                    features["rect_nonzero_full_top"] += int(h != w and min(h, w) > 1
                                                            and dominant != 0 and np.all(grid[0] != 0))
                    features["small_symmetric_test"] += int(max(h, w) <= 5 and min(h, w) > 1
                                                             and not _orientation_rich(grid))
    for name in ("rectangle", "square", "line", "small_test", "test_not_larger",
                 "one_color", "two_color", "zero_absent", "nonzero_dominant",
                 "symmetric", "sparse", "dense", "full_top", "full_left",
                 "blank_top", "blank_left", "mixed_top", "mixed_left",
                 "corner_zero", "corner_nonzero", "repeated_rows",
                 "repeated_columns", "palette_reuse", "palette_change",
                 "palette_disjoint", "palette_partial", "single_mark",
                 "long_run", "colorful_top", "colorful_left", "source_like",
                 "rect_nonzero_full_top", "small_symmetric_test"):
        _require(features[name] >= 2, f"missing review coverage: {name}")
    _require(len(shapes) >= 30 and len(structures) >= 120 and len(palettes) >= 35,
             "insufficient structural, shape, or palette diversity")
    for i, (nt, nx) in enumerate(((2, 1), (5, 2), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + i, nt, nx)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, nx), "explicit counts")
        validate_generated_puzzle(puzzle)
    for i, (nt, nx) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + i, nt, nx)
        _require((nt is None or len(puzzle["train"]) == nt)
                 and (nx is None or len(puzzle["test"]) == nx), "partial counts")
    _require(_puzzle_key(generate_puzzle(314159, 8, 3)) == _puzzle_key(generate_puzzle(314159, 8, 3)),
             "explicit determinism")
    _require(_puzzle_key(generate_puzzle(271828)) == _puzzle_key(generate_puzzle(271828)),
             "automatic determinism")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.0}):
        try:
            generate_puzzle(99, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {kwargs}")
