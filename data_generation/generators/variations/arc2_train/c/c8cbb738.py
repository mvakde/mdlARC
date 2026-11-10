"""Sparse, mirror-symmetric color layers translated to one common center."""

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rows(*rows: str) -> np.ndarray:
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


# Exact source inputs are excluded, without excluding their sizes or structures.
_SOURCE_INPUTS = (
    _rows(
        "11111111", "18181111", "11111111", "18181111", "11111111",
        "11113111", "11131311", "11113111", "11111111", "11111111",
    ),
    _rows(
        "33333334343", "33333333333", "33331333333", "33333333333",
        "33133314343", "33333333333", "33331323332", "33333333333",
        "83338333333", "33333333333", "83338323332", "33333333333",
    ),
    _rows(
        "44444444444444", "44144414444444", "44444444444444",
        "44444444444444", "44444444444444", "44144414474444",
        "44444444444444", "44444447444744", "44444444444444",
        "44444444474444", "44444444444444", "44444444444444",
    ),
    _rows(
        "8888888888888888888", "8888888888888888888",
        "8888881888888888888", "8888888888888888888",
        "8888888888888888888", "8881888881888888888",
        "8888888888828888828", "8888888888888888888",
        "8888881888828888828", "8883888883888888888",
        "8888888888888868688", "8888888888888888888",
        "8888888888888888888", "8888888888888888888",
        "8888888888888888888", "8883888883888888888",
        "8888888888888868688", "8888888888888888888",
    ),
)


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes())


_SOURCE_INPUT_KEYS = frozenset(_grid_key(grid) for grid in _SOURCE_INPUTS)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _count_value(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _choose_counts(seed, rng, num_train, num_test) -> tuple[int, int]:
    num_train = _count_value("num_train", num_train, 2, 10)
    num_test = _count_value("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _color_context(rng: np.random.Generator) -> dict:
    repeated = bool(rng.random() < 0.25)
    size = 5 if repeated else int(rng.integers(5, 11))
    pool = [int(c) for c in rng.choice(10, size=size, replace=False)]
    held_out = None
    if not repeated and size >= 6 and rng.random() < 0.55:
        held_out = pool[-1]
    return {
        "train_pool": pool[:-1] if held_out is not None else pool,
        "all_pool": pool,
        "held_out": held_out,
    }


def _pair_colors(rng, context, regime, index, count) -> tuple[int, list[int]]:
    train_pool = context["train_pool"]
    if regime == "source_train" and index == 0:
        return train_pool[0], train_pool[1:3]
    if regime == "source_train" and index == 1:
        background = train_pool[1]
        colors = [train_pool[0]]
        available = [c for c in train_pool if c not in (background, train_pool[0])]
        colors += [int(c) for c in rng.choice(available, size=count - 1, replace=False)]
        return background, [colors[int(i)] for i in rng.permutation(len(colors))]
    if regime == "target_test" and index == 0 and context["held_out"] is not None:
        background = int(rng.choice(train_pool))
        colors = [context["held_out"]]
        available = [c for c in context["all_pool"] if c not in (background, colors[0])]
        colors += [int(c) for c in rng.choice(available, size=count - 1, replace=False)]
    else:
        available_pool = train_pool if regime == "source_train" else context["all_pool"]
        background = int(rng.choice(available_pool))
        available = [c for c in available_pool if c != background]
        colors = [int(c) for c in rng.choice(available, size=count, replace=False)]
    return background, [colors[int(i)] for i in rng.permutation(len(colors))]


def _base_motifs(height: int, width: int) -> list[tuple[str, int, int, frozenset]]:
    result = []
    for radius_row in range(1, height // 2 + 1):
        for radius_col in range(1, width // 2 + 1):
            corners = frozenset(
                (r, c)
                for r in (-radius_row, radius_row)
                for c in (-radius_col, radius_col)
            )
            cross = frozenset(
                ((-radius_row, 0), (radius_row, 0),
                 (0, -radius_col), (0, radius_col))
            )
            result.append(("corners", radius_row, radius_col, corners))
            result.append(("cross", radius_row, radius_col, cross))
    return result


def _reflection_orbits(height: int, width: int) -> list[frozenset]:
    result = []
    for row in range(height // 2 + 1):
        for col in range(width // 2 + 1):
            result.append(frozenset(
                (r, c)
                for r in ((-row, row) if row else (0,))
                for c in ((-col, col) if col else (0,))
            ))
    return result


def _choose_layers(rng, height, width, count, split) -> list[frozenset]:
    candidates = _base_motifs(height, width)
    radius_row, radius_col = height // 2, width // 2
    for _ in range(60):
        shuffled = [candidates[int(i)] for i in rng.permutation(len(candidates))]
        chosen = []
        occupied = set()
        if split:
            first_options = [
                item for item in shuffled
                if item[1] == radius_row and item[2] < radius_col
            ]
        else:
            first_options = [
                item for item in shuffled
                if item[1] == radius_row and item[2] == radius_col
            ]
        if not first_options:
            continue
        first = first_options[0]
        chosen.append(first)
        occupied.update(first[3])
        if split:
            second_options = [
                item for item in shuffled
                if item[2] == radius_col and item[1] < radius_row
                and not occupied.intersection(item[3])
            ]
            if not second_options:
                continue
            second = second_options[0]
            chosen.append(second)
            occupied.update(second[3])
        for item in shuffled:
            if len(chosen) == count:
                break
            if split and item[1] == radius_row and item[2] == radius_col:
                continue
            if item in chosen or occupied.intersection(item[3]):
                continue
            chosen.append(item)
            occupied.update(item[3])
        if len(chosen) != count:
            continue
        layers = [set(item[3]) for item in chosen]
        # Richer motifs use whole horizontal/vertical reflection orbits only.
        for layer_index in rng.permutation(count):
            layer_index = int(layer_index)
            if min(height, width) == 3 or rng.random() >= 0.26:
                continue
            _, r, c, _ = chosen[layer_index]
            orbits = _reflection_orbits(2 * r + 1, 2 * c + 1)
            for orbit_index in rng.permutation(len(orbits)):
                orbit = orbits[int(orbit_index)]
                if orbit.intersection(occupied):
                    continue
                if len(occupied) + len(orbit) > height * width - 2:
                    continue
                layers[layer_index].update(orbit)
                occupied.update(orbit)
                break
        return [frozenset(layer) for layer in layers]
    raise AssertionError("could not choose compatible symmetric motifs")


def _boxes_overlap(a, b) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar < br + bh and br < ar + ah and ac < bc + bw and bc < ac + aw


def _boxes_adjacent(a, b) -> bool:
    if _boxes_overlap(a, b):
        return False
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    row_touch = (ar + ah == br or br + bh == ar) and ac < bc + bw and bc < ac + aw
    col_touch = (ac + aw == bc or bc + bw == ac) and ar < br + bh and br < ar + ah
    return row_touch or col_touch


def _boxes_clear(a, b) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (
        ar + ah < br or br + bh < ar
        or ac + aw < bc or bc + bw < ac
    )


def _placements(rng, layers, shapes, grid_height, grid_width, layout, edge):
    choices = [
        [(r, c) for r in range(grid_height - h + 1)
         for c in range(grid_width - w + 1)]
        for h, w in shapes
    ]

    def cells(index, position):
        h, w = shapes[index]
        center_r = position[0] + h // 2
        center_c = position[1] + w // 2
        return {(center_r + r, center_c + c) for r, c in layers[index]}

    def box(index, position):
        return position + shapes[index]

    def at_edge(index, position):
        r, c, h, w = box(index, position)
        return r == 0 or c == 0 or r + h == grid_height or c + w == grid_width

    first_choices = [p for p in choices[0] if not edge or at_edge(0, p)]
    for _ in range(35):
        first = first_choices[int(rng.integers(len(first_choices)))]
        used = cells(0, first)
        first_box = box(0, first)
        second_options = []
        for p in choices[1]:
            other_box = box(1, p)
            if layout == "overlap" and not _boxes_overlap(first_box, other_box):
                continue
            if layout == "adjacent" and not _boxes_adjacent(first_box, other_box):
                continue
            if layout == "clear" and not _boxes_clear(first_box, other_box):
                continue
            if used.isdisjoint(cells(1, p)):
                second_options.append(p)
        if not second_options:
            continue
        second = second_options[int(rng.integers(len(second_options)))]
        positions = [first, second]
        used.update(cells(1, second))
        for index in range(2, len(layers)):
            valid = []
            for p in choices[index]:
                if layout == "clear" and any(
                    not _boxes_clear(box(index, p), box(j, old))
                    for j, old in enumerate(positions)
                ):
                    continue
                if used.isdisjoint(cells(index, p)):
                    valid.append(p)
            if not valid:
                break
            chosen = valid[int(rng.integers(len(valid)))]
            positions.append(chosen)
            used.update(cells(index, chosen))
        if len(positions) == len(layers):
            return positions
    # A centered placement is always collision-free because the layers are
    # disjoint in output coordinates. It is a bounded fallback for tight grids.
    center_r, center_c = grid_height // 2, grid_width // 2
    return [
        (center_r - h // 2, center_c - w // 2)
        for h, w in shapes
    ]


def _render_pair(rng, background, colors, layers, height, width) -> dict:
    shapes = []
    for layer in layers:
        max_r = max(abs(r) for r, _ in layer)
        max_c = max(abs(c) for _, c in layer)
        shapes.append((2 * max_r + 1, 2 * max_c + 1))
    layout = ("clear", "adjacent", "overlap")[int(rng.choice(3, p=[0.25, 0.40, 0.35]))]
    edge = bool(rng.random() < 0.38)
    min_height = max(height + 4, 8, 21 if layout == "clear" and len(layers) >= 3 else 0)
    min_width = max(width + 4, 8, 21 if layout == "clear" and len(layers) >= 3 else 0)
    grid_height = int(rng.integers(min_height, 31))
    grid_width = int(rng.integers(min_width, 31))
    positions = _placements(
        rng, layers, shapes, grid_height, grid_width, layout, edge
    )
    input_grid = np.full((grid_height, grid_width), background, dtype=np.int8)
    output_grid = np.full((height, width), background, dtype=np.int8)
    for color, layer, shape, (top, left) in zip(colors, layers, shapes, positions):
        center_r, center_c = top + shape[0] // 2, left + shape[1] // 2
        for r, c in layer:
            _require(input_grid[center_r + r, center_c + c] == background,
                     "input collision")
            _require(output_grid[height // 2 + r, width // 2 + c] == background,
                     "output collision")
            input_grid[center_r + r, center_c + c] = color
            output_grid[height // 2 + r, width // 2 + c] = color
    return {"input": input_grid, "output": output_grid}


_SQUARE_SIZES = ((3, 3), (5, 5), (7, 7), (9, 9))
_RECT_SIZES = ((3, 5), (5, 3), (3, 7), (7, 3),
               (5, 7), (7, 5), (5, 9), (9, 5),
               (7, 9), (9, 7))
_SPLIT_SIZES = ((5, 7), (7, 5), (5, 9), (9, 5), (7, 9), (9, 7))


def _sample_layer_count(rng, height, width):
    # Corners have one orbit per radius pair; disjoint crosses can use
    # each horizontal and vertical radius only once.
    r, c = height // 2, width // 2
    maximum = min(4, r * c + min(r, c))
    return int(rng.integers(2, maximum + 1))


def _train_spec(rng, index):
    if index == 0:
        sizes = ((3, 3), (3, 3), (5, 5), (5, 5), (3, 5), (5, 3))
        height, width = sizes[int(rng.integers(len(sizes)))]
        return height, width, 2, False
    if index == 1:
        sizes = ((5, 5), (5, 5), (7, 7), (3, 3), (5, 7), (7, 5))
        height, width = sizes[int(rng.integers(len(sizes)))]
        count = _sample_layer_count(rng, height, width)
        split = height >= 5 and width >= 5 and height != width and rng.random() < 0.45
        return height, width, count, split
    if index == 2:
        height, width = _SPLIT_SIZES[int(rng.integers(len(_SPLIT_SIZES)))]
        return height, width, _sample_layer_count(rng, height, width), True
    choices = _SQUARE_SIZES + _RECT_SIZES
    height, width = choices[int(rng.integers(len(choices)))]
    count = _sample_layer_count(rng, height, width)
    split = height >= 5 and width >= 5 and rng.random() < 0.4
    return height, width, count, split


def _test_spec(rng, train_specs, index):
    rectangular_trained = any(h != w for h, w, _, _ in train_specs)
    split_trained = any(split for _, _, _, split in train_specs)
    if index == 0 and rng.random() < 0.30:
        scale_choices = ((11, 11),)
        if rectangular_trained:
            scale_choices += ((7, 11), (11, 7), (9, 11), (11, 9))
        height, width = scale_choices[int(rng.integers(len(scale_choices)))]
    elif rng.random() < 0.38:
        height, width, _, _ = train_specs[int(rng.integers(len(train_specs)))]
    else:
        sizes = _SQUARE_SIZES + (_RECT_SIZES if rectangular_trained else ())
        height, width = sizes[int(rng.integers(len(sizes)))]
    count = _sample_layer_count(rng, height, width)
    split = (
        split_trained and height >= 5 and width >= 5
        and rng.random() < 0.55
    )
    return height, width, count, split


def _make_pair(rng, context, regime, index, spec):
    height, width, count, split = spec
    background, colors = _pair_colors(rng, context, regime, index, count)
    layers = _choose_layers(rng, height, width, count, split)
    return _render_pair(rng, background, colors, layers, height, width)


def _analyze_input(grid: np.ndarray) -> dict:
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    maximum = int(np.max(counts))
    _require(np.count_nonzero(counts == maximum) == 1,
             "background is not uniquely dominant")
    background = int(np.argmax(counts))
    _require(maximum > grid.size // 2, "background must occupy most input cells")
    colors = [int(c) for c in np.flatnonzero(counts) if int(c) != background]
    _require(2 <= len(colors) <= 4, "expected two to four color layers")
    layers = []
    for color in colors:
        coords = np.argwhere(grid == color)
        min_r, min_c = np.min(coords, axis=0)
        max_r, max_c = np.max(coords, axis=0)
        height, width = int(max_r - min_r + 1), int(max_c - min_c + 1)
        _require(height >= 3 and width >= 3 and height % 2 and width % 2,
                 "layer bounding box must have odd two-dimensional spans")
        center_r, center_c = (int(min_r) + int(max_r)) // 2, (int(min_c) + int(max_c)) // 2
        layer = frozenset(
            (int(r) - center_r, int(c) - center_c) for r, c in coords
        )
        _require(len(layer) >= 4, "layer is too small")
        _require(
            all((-r, c) in layer and (r, -c) in layer for r, c in layer),
            "layer lacks both mirror symmetries",
        )
        layers.append((color, height, width, layer))
    output_height = max(item[1] for item in layers)
    output_width = max(item[2] for item in layers)
    output = np.full((output_height, output_width), background, dtype=np.int8)
    for color, _, _, layer in layers:
        for r, c in layer:
            rr, cc = output_height // 2 + r, output_width // 2 + c
            _require(output[rr, cc] == background, "layers collide after centering")
            output[rr, cc] = color
    return {"background": background, "layers": layers, "output": output}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle schema")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "pair lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4
             and len(train) > len(test), "pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair schema")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2
                         and np.issubdtype(grid.dtype, np.integer), "grid type")
                _require(all(1 <= s <= 30 for s in grid.shape), "grid size")
                _require(np.all((grid >= 0) & (grid <= 9)), "ARC colors")
            inp, out = pair["input"], pair["output"]
            input_key, output_key = _grid_key(inp), _grid_key(out)
            _require(input_key not in seen_inputs, "duplicate input")
            _require((input_key, output_key) not in seen_pairs, "duplicate pair")
            _require(input_key not in _SOURCE_INPUT_KEYS, "source pair copied")
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            analysis = _analyze_input(inp)
            _require(np.array_equal(out, analysis["output"]),
                     "output disagrees with centered transparent overlay")
            _require(inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1],
                     "input must exceed the compact output")
            analyses[section].append(analysis)
    # Color roles are local to each pair. Training demonstrates role changes.
    first_bg = analyses["train"][0]["background"]
    first_foreground = {item[0] for item in analyses["train"][0]["layers"]}
    second_bg = analyses["train"][1]["background"]
    _require(first_bg != second_bg and second_bg in first_foreground,
             "training does not demonstrate a background role change")
    _require(first_bg in {item[0] for item in analyses["train"][1]["layers"]},
             "training does not demonstrate a payload role change")
    rectangular_train = any(
        a["output"].shape[0] != a["output"].shape[1]
        for a in analyses["train"]
    )
    split_train = any(
        not any(
            h == a["output"].shape[0] and w == a["output"].shape[1]
            for _, h, w, _ in a["layers"]
        )
        for a in analyses["train"]
    )
    for a in analyses["test"]:
        h, w = a["output"].shape
        if h != w:
            _require(rectangular_train, "rectangular test crop lacks a train example")
        split = not any(xh == h and xw == w for _, xh, xw, _ in a["layers"])
        if split:
            _require(split_train, "split-extrema test crop lacks a train example")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(seed, rng, num_train, num_test)
    context = _color_context(rng)
    train_specs = [_train_spec(rng, index) for index in range(train_count)]
    test_specs = [_test_spec(rng, train_specs, index) for index in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, specs, target in (
        ("source_train", train_specs, puzzle["train"]),
        ("target_test", test_specs, puzzle["test"]),
    ):
        for index, spec in enumerate(specs):
            for _ in range(12):
                pair = _make_pair(rng, context, regime, index, spec)
                key = _grid_key(pair["input"])
                if key not in seen and key not in _SOURCE_INPUT_KEYS:
                    seen.add(key)
                    target.append(pair)
                    break
            else:
                raise AssertionError("could not generate a distinct source-free pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b) -> bool:
    return all(
        np.array_equal(pa[name], pb[name])
        for section in ("train", "test")
        for pa, pb in zip(a[section], b[section])
        for name in ("input", "output")
    ) and all(len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    from collections import Counter

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                predicted = _analyze_input(pair["input"])["output"]
                _require(np.array_equal(predicted, pair["output"]), "wrong transform")
    _require(_puzzles_equal(generate_puzzle(17), generate_puzzle(17)),
             "same seed is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(17), generate_puzzle(18)),
             "different seeds gave same puzzle")
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic count diversity")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (3, 2), (5, 3)):
        puzzle = generate_puzzle(100 + train_count * 10 + test_count,
                                 train_count, test_count)
        _require(len(puzzle["train"]) == train_count
                 and len(puzzle["test"]) == test_count, "explicit counts")
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(23, train_count, test_count))
    for train_count, test_count in ((1, 1), (11, 1), (2, 0), (2, 5),
                                    (2, 2), (3, 4)):
        try:
            generate_puzzle(0, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    metrics = Counter()
    output_shapes = set()
    motif_counts = set()
    puzzle_palettes = set()
    for puzzle in puzzles:
        palette = set()
        train_colors = set()
        train_backgrounds = []
        train_shapes = {p["output"].shape for p in puzzle["train"]}
        max_train_extent = max(max(shape) for shape in train_shapes)
        max_train_motifs = max(len(np.unique(p["input"])) - 1 for p in puzzle["train"])
        for pair in puzzle["train"]:
            train_colors.update(np.unique(pair["input"]).tolist())
            train_backgrounds.append(_analyze_input(pair["input"])["background"])
        metrics["puzzles_with_multiple_train_backgrounds"] += (
            len(set(train_backgrounds)) > 1
        )
        metrics["puzzles_with_multiple_train_shapes"] += len(train_shapes) > 1
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                analysis = _analyze_input(inp)
                palette.update(np.unique(inp).tolist())
                output_shapes.add(out.shape)
                motif_counts.add(len(analysis["layers"]))
                boxes = []
                four_in_pair = True
                for color, h, w, layer in analysis["layers"]:
                    metrics["motifs"] += 1
                    metrics["four_point"] += len(layer) == 4
                    metrics["rich"] += len(layer) > 4
                    four_in_pair &= len(layer) == 4
                    metrics["cross"] += len(layer) == 4 and all(
                        (r == 0) != (c == 0) for r, c in layer
                    )
                    metrics["motif_5x3"] += (h, w) == (5, 3)
                    metrics["motif_3x7"] += (h, w) == (3, 7)
                    coords = np.argwhere(inp == color)
                    top, left = np.min(coords, axis=0)
                    boxes.append((int(top), int(left), h, w))
                    metrics["edge"] += (
                        top == 0 or left == 0
                        or top + h == inp.shape[0] or left + w == inp.shape[1]
                    )
                metrics["pairs"] += 1
                metrics["all_four_pairs"] += four_in_pair
                metrics["rectangular"] += out.shape[0] != out.shape[1]
                split = not any(
                    h == out.shape[0] and w == out.shape[1]
                    for _, h, w, _ in analysis["layers"]
                )
                metrics["split"] += split
                metrics["rectangular_split"] += split and out.shape[0] != out.shape[1]
                metrics["adjacent"] += any(
                    _boxes_adjacent(boxes[i], boxes[j])
                    for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                )
                metrics["overlap"] += any(
                    _boxes_overlap(boxes[i], boxes[j])
                    for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                )
                metrics["tight_test"] += section == "test" and max(inp.shape) <= 21
                metrics["large_test"] += section == "test" and max(inp.shape) >= 25
                metrics["reuse_test_size"] += section == "test" and out.shape in {
                    p["output"].shape for p in puzzle["train"]
                }
                metrics["test_scale_shift"] += (
                    section == "test" and max(out.shape) > max_train_extent
                )
                metrics["test_fewer_motifs"] += (
                    section == "test" and len(analysis["layers"]) < max_train_motifs
                )
                metrics["test_split"] += section == "test" and split
        puzzle_palettes.add(len(palette))
        metrics["novel_test_color_puzzles"] += any(
            not set(np.unique(p["input"]).tolist()).issubset(train_colors)
            for p in puzzle["test"]
        )
    _require(metrics["four_point"] >= metrics["motifs"] // 2,
             "source-like four-point motifs underrepresented")
    for name in ("cross", "edge", "adjacent", "overlap", "rectangular",
                 "tight_test", "large_test", "reuse_test_size",
                 "novel_test_color_puzzles", "rich", "all_four_pairs",
                 "motif_5x3", "motif_3x7", "split", "rectangular_split",
                 "test_split", "test_scale_shift", "test_fewer_motifs"):
        _require(metrics[name] >= 3, f"missing diversity: {name}")
    _require(metrics["puzzles_with_multiple_train_backgrounds"] == len(puzzles),
             "background roles do not vary within each training set")
    _require(metrics["puzzles_with_multiple_train_shapes"] >= 30,
             "training shape diversity is too narrow")
    _require({2, 3, 4}.issubset(motif_counts), "motif count diversity")
    _require(len(output_shapes) >= 8 and len(puzzle_palettes) >= 3,
             "shape or palette diversity")
