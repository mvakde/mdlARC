import numpy as np


def _rows_signature(rows):
    return f"{len(rows)}x{len(rows[0])}:" + "/".join(rows)


_SOURCE_PAIR_SIGNATURES = frozenset(
    {
        (
            _rows_signature(
                (
                    "0000000000000000",
                    "0000002000400000",
                    "0000002044040000",
                    "0000002444400000",
                    "0000002440000000",
                    "0000502444000000",
                    "0000002400000000",
                    "0000000000000000",
                    "0000000000000000",
                    "0000000000000000",
                    "0000000000000000",
                    "0000420000000000",
                    "0044420000000000",
                    "4440420007000000",
                    "0444020000000000",
                    "0000000000000000",
                )
            ),
            _rows_signature(
                (
                    "0000000000000000",
                    "0050002000400000",
                    "0505502044040000",
                    "0055552444400000",
                    "0000552440000000",
                    "0005552444000000",
                    "0000052400000000",
                    "0000000000000000",
                    "0000000000000000",
                    "0000000000000000",
                    "0000000000000000",
                    "0000427000000000",
                    "0044427770000000",
                    "4440427077700000",
                    "0444020777000000",
                    "0000000000000000",
                )
            ),
        ),
        (
            _rows_signature(
                (
                    "0000000000000000",
                    "0000042000000000",
                    "0004442000000000",
                    "0004442008000000",
                    "0004442000000000",
                    "0000442000000000",
                    "0000042000000000",
                    "0000000000000000",
                    "0000000000000000",
                    "0000000000420000",
                    "0000000004020300",
                    "0000000440420000",
                    "0000000044020000",
                    "0000000000000000",
                )
            ),
            _rows_signature(
                (
                    "0000000000000000",
                    "0000042800000000",
                    "0004442888000000",
                    "0004442888000000",
                    "0004442888000000",
                    "0000442880000000",
                    "0000042800000000",
                    "0000000000000000",
                    "0000000000000000",
                    "0000000000423000",
                    "0000000004020300",
                    "0000000440423033",
                    "0000000044020330",
                    "0000000000000000",
                )
            ),
        ),
        (
            _rows_signature(
                (
                    "000000000000000000",
                    "044420100000000000",
                    "044020000000000000",
                    "444420000000420000",
                    "000000000044420000",
                    "000000000004020000",
                    "000000000044420080",
                    "004000200000000000",
                    "000400200000000000",
                    "000040200000000000",
                    "000444200700000000",
                    "004044200000000000",
                    "000040200000002400",
                    "000000000000002040",
                    "000000000000302440",
                    "000000000000000000",
                )
            ),
            _rows_signature(
                (
                    "000000000000000000",
                    "044421110000000000",
                    "044020110000000000",
                    "444421111000428000",
                    "000000000044428880",
                    "000000000004020800",
                    "000000000044428880",
                    "004000200070000000",
                    "000400200700000000",
                    "000040207000000000",
                    "000444277700000000",
                    "004044277070000000",
                    "000040207000032400",
                    "000000000000302040",
                    "000000000000332440",
                    "000000000000000000",
                )
            ),
        ),
    }
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _validate_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _validate_optional_count(
        num_train, "num_train", 2, 10
    )
    num_test = _validate_optional_count(
        num_test, "num_test", 1, 4
    )
    code = abs(int(seed))

    if num_train is None and num_test is None:
        num_test = 1 + code % 4
        span = 10 - num_test
        num_train = (
            num_test
            + 1
            + ((code // 4 + 3 * code) % span)
        )
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        upper = min(4, num_train - 1)
        num_test = int(rng.integers(1, upper + 1))

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def _sample_puzzle_parameters(rng):
    colors = [int(c) for c in rng.permutation(10)]
    return {
        "background": colors[0],
        "templates": tuple(colors[1:3]),
        "axes": tuple(colors[3:5]),
        "accents": tuple(colors[5:]),
        "orientation_mode": ("vertical", "horizontal", "mixed")[int(rng.integers(0, 3))],
        "palette_mode": ("shared", "recolored")[int(rng.integers(0, 2))],
    }


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _grid_signature(grid):
    return _rows_signature(tuple("".join(str(int(x)) for x in row) for row in grid))


def _is_source_pair(input_grid, output_grid):
    return (_grid_signature(input_grid), _grid_signature(output_grid)) in _SOURCE_PAIR_SIGNATURES


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        component = {remaining.pop()}
        stack = list(component)
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def _sample_shape(rng, length, reach, style):
    area = length * reach
    if style == "sparse":
        low, high = 3, max(3, int(np.floor(area * 0.48)))
    elif style == "full":
        low, high = max(3, int(np.ceil(area * 0.58))), max(3, int(np.ceil(area * 0.85)))
    else:
        low, high = max(3, int(np.ceil(area * 0.36))), max(3, int(np.ceil(area * 0.66)))
    target = min(area - 1, int(rng.integers(low, min(area - 1, high) + 1)))
    cells = {(int(rng.integers(length)), 1), (int(rng.integers(length)), reach)}
    options = [(r, d) for r in range(length) for d in range(1, reach + 1) if (r, d) not in cells]
    rng.shuffle(options)
    cells.update(options[:max(0, target - len(cells))])
    return cells


def _sample_specs(rng, regime, parameters, pair_index, count):
    if regime == "source_train":
        n = 2 if pair_index == 0 or rng.random() < 0.72 else int(rng.integers(3, 5))
    else:
        n = int(rng.choice((1, 2, 3, 4), p=(0.12, 0.21, 0.25, 0.42)))
    orientations = []
    mode = parameters["orientation_mode"]
    orientation_offset = int(rng.integers(2))
    palette_offset = int(rng.integers(2))
    for i in range(n):
        if mode == "mixed":
            orientation = ("vertical", "horizontal")[(i + orientation_offset) % 2] if regime == "source_train" and pair_index == 0 and i < 2 else ("vertical", "horizontal")[int(rng.integers(2))]
        else:
            orientation = mode
        orientations.append(orientation)

    available_accents = parameters["accents"] if parameters["palette_mode"] == "recolored" else tuple(c for c in range(10) if c not in (parameters["background"], parameters["templates"][0], parameters["axes"][0]))
    accents = [int(x) for x in rng.choice(available_accents, size=n, replace=n > len(available_accents))]
    if n > 1 and rng.random() < 0.2:
        accents[-1] = accents[0]
    specs = []
    for i in range(n):
        length = int(rng.integers(3, 9))
        reach = int(rng.integers(2, 6))
        style = ("sparse", "medium", "full")[int(rng.choice(3, p=(0.36, 0.36, 0.28)))]
        if parameters["palette_mode"] == "shared":
            palette_index = 0
        elif regime == "source_train" and pair_index == 0 and i < 2:
            palette_index = (i + palette_offset) % 2
        else:
            palette_index = int(rng.integers(2))
        specs.append({
            "orientation": orientations[i], "length": length, "reach": reach,
            "side": (-1, 1)[int(rng.integers(2))],
            "shape": _sample_shape(rng, length, reach, style),
            "template": parameters["templates"][palette_index],
            "axis": parameters["axes"][palette_index],
            "accent": accents[i],
        })
    return specs


def _rectangle(spec):
    if spec["orientation"] == "vertical":
        return spec["length"], 2 * spec["reach"] + 1
    return 2 * spec["reach"] + 1, spec["length"]


def _place_specs(rng, specs, height, width, background):
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    occupied_rectangles = []
    order = sorted(range(len(specs)), key=lambda i: np.prod(_rectangle(specs[i])), reverse=True)
    placements = {}
    for index in order:
        spec = specs[index]
        box_h, box_w = _rectangle(spec)
        if box_h > height - 2 or box_w > width - 2:
            return None
        candidates = [(r, c) for r in range(1, height - box_h) for c in range(1, width - box_w)]
        rng.shuffle(candidates)
        found = None
        for row, col in candidates:
            if all(row + box_h + 1 <= r0 or r0 + h0 + 1 <= row or col + box_w + 1 <= c0 or c0 + w0 + 1 <= col for r0, c0, h0, w0 in occupied_rectangles):
                found = row, col
                break
        if found is None:
            return None
        row, col = found
        occupied_rectangles.append((row, col, box_h, box_w))
        placements[index] = (row, col)

    for index, spec in enumerate(specs):
        top, left = placements[index]
        length, reach = spec["length"], spec["reach"]
        vertical = spec["orientation"] == "vertical"
        for t in range(length):
            row, col = (top + t, left + reach) if vertical else (top + reach, left + t)
            inp[row, col] = out[row, col] = spec["axis"]
        seed = sorted(spec["shape"])[int(rng.integers(len(spec["shape"])))]
        for t, d in spec["shape"]:
            if vertical:
                source = (top + t, left + reach + spec["side"] * d)
                destination = (top + t, left + reach - spec["side"] * d)
            else:
                source = (top + reach + spec["side"] * d, left + t)
                destination = (top + reach - spec["side"] * d, left + t)
            inp[source] = out[source] = spec["template"]
            out[destination] = spec["accent"]
            if (t, d) == seed:
                inp[destination] = spec["accent"]
    return inp, out


def _analyze_pair(inp, out, background, template_colors, axis_colors):
    height, width = inp.shape
    axes = []
    for color in axis_colors:
        for component in _components(set(zip(*np.where(inp == color)))):
            rows = {r for r, _ in component}
            cols = {c for _, c in component}
            if not 3 <= len(component) <= 8:
                return None
            if len(cols) == 1 and len(rows) == len(component):
                orientation, start, coordinate = "vertical", min(rows), next(iter(cols))
                if rows != set(range(start, start + len(component))):
                    return None
            elif len(rows) == 1 and len(cols) == len(component):
                orientation, start, coordinate = "horizontal", min(cols), next(iter(rows))
                if cols != set(range(start, start + len(component))):
                    return None
            else:
                return None
            axes.append((orientation, start, coordinate, len(component), color, component))
    if not axes:
        return None
    axes.sort(key=lambda a: (a[1], a[2], a[0], a[4]))
    groups = [set() for _ in axes]
    template_cells = set(zip(*np.where(np.isin(inp, list(template_colors)))))
    for row, col in template_cells:
        matches = []
        for i, (orientation, start, coordinate, length, _, _) in enumerate(axes):
            t, d = (row - start, abs(col - coordinate)) if orientation == "vertical" else (col - start, abs(row - coordinate))
            if 0 <= t < length and 1 <= d <= 5:
                matches.append(i)
        if len(matches) != 1:
            return None
        groups[matches[0]].add((row, col))
    all_axis = set().union(*(a[5] for a in axes))
    destinations_seen = set()
    seeds = set()
    predicted = inp.copy()
    metrics = []
    for axis, group in zip(axes, groups):
        orientation, start, coordinate, length, axis_color, _ = axis
        if len(group) < 3 or len({int(inp[cell]) for cell in group}) != 1:
            return None
        template_color = int(inp[next(iter(group))])
        sides = {(-1 if (c if orientation == "vertical" else r) < coordinate else 1) for r, c in group}
        if len(sides) != 1:
            return None
        side = next(iter(sides))
        reach = max(abs((c if orientation == "vertical" else r) - coordinate) for r, c in group)
        if not 2 <= reach <= 5:
            return None
        destinations = {(r, 2 * coordinate - c) if orientation == "vertical" else (2 * coordinate - r, c) for r, c in group}
        if any(not (0 <= r < height and 0 <= c < width) for r, c in destinations):
            return None
        if destinations & (destinations_seen | all_axis | template_cells):
            return None
        destinations_seen.update(destinations)
        occupied = [cell for cell in destinations if int(inp[cell]) != background]
        if len(occupied) != 1:
            return None
        seed = occupied[0]
        accent = int(inp[seed])
        if accent in template_colors or accent in axis_colors or accent == background:
            return None
        if any(int(inp[cell]) != background for cell in destinations if cell != seed):
            return None
        for cell in destinations:
            predicted[cell] = accent
        seeds.add(seed)
        local = tuple(sorted(((r - start, abs(c - coordinate)) if orientation == "vertical" else (c - start, abs(r - coordinate))) for r, c in group))
        metrics.append({"orientation": orientation, "side": side, "length": length, "reach": reach,
                        "occupancy": len(group) / (length * reach), "connected": len(_components(group)) == 1,
                        "template": template_color, "axis": axis_color, "accent": accent,
                        "shape": local, "anchor": (start, coordinate)})
    actual_non_background = set(zip(*np.where(inp != background)))
    if actual_non_background != all_axis | template_cells | seeds:
        return None
    if out is not None and not np.array_equal(predicted, out):
        return None
    if np.array_equal(predicted, inp):
        return None
    return {"objects": metrics, "predicted": predicted}


def _infer_roles(train_pairs):
    counts = np.zeros(10, dtype=np.int64)
    for pair in train_pairs:
        values, frequency = np.unique(pair["input"], return_counts=True)
        counts[values] += frequency
    background = int(np.argmax(counts))
    if np.count_nonzero(counts == counts[background]) != 1:
        return None
    accents = set()
    colors = set()
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        accents.update(int(x) for x in np.unique(out[inp != out]))
        colors.update(int(x) for x in np.unique(inp))
    structural = sorted(colors - accents - {background})
    candidates = []
    for mask in range(1, (1 << len(structural)) - 1):
        axes = {structural[i] for i in range(len(structural)) if mask & (1 << i)}
        templates = set(structural) - axes
        analyses = [_analyze_pair(pair["input"], pair["output"], background, templates, axes) for pair in train_pairs]
        if all(a is not None for a in analyses):
            candidates.append((background, templates, axes, analyses))
    return candidates[0] if len(candidates) == 1 else None


def _generate_pair(rng, regime, parameters, pair_index, count):
    for attempt in range(120):
        specs = _sample_specs(rng, regime, parameters, pair_index, count)
        n = len(specs)
        low = (10, 14, 18, 19)[n - 1]
        high = (22, 25, 29, 30)[n - 1]
        if attempt < 75:
            height, width = (int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1)))
        else:
            height = width = 30
        placed = _place_specs(rng, specs, height, width, parameters["background"])
        if placed is None:
            continue
        inp, out = placed
        template_colors = {parameters["templates"][0]} if parameters["palette_mode"] == "shared" else set(parameters["templates"])
        axis_colors = {parameters["axes"][0]} if parameters["palette_mode"] == "shared" else set(parameters["axes"])
        if _analyze_pair(inp, out, parameters["background"], template_colors, axis_colors) is not None and not _is_source_pair(inp, out):
            return inp, out
    raise RuntimeError("could not place unambiguous reflection objects")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "pair lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts")
    seen = set()
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
        for grid in (pair["input"], pair["output"]):
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
            _require(np.issubdtype(grid.dtype, np.integer), "grid integer dtype")
            _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid size")
            _require(np.all((0 <= grid) & (grid <= 9)), "ARC colors")
        inp, out = pair["input"], pair["output"]
        _require(inp.shape == out.shape, "dimensions changed")
        key = _grid_key(inp)
        _require(key not in seen, "duplicate input")
        seen.add(key)
        _require(not _is_source_pair(inp, out), "source pair copied")
    inferred = _infer_roles(train)
    _require(inferred is not None, "training does not identify color roles uniquely")
    background, templates, axes, train_analyses = inferred
    train_orientations = {o["orientation"] for a in train_analyses for o in a["objects"]}
    train_sides = {o["side"] for a in train_analyses for o in a["objects"]}
    train_max_reach = max(o["reach"] for a in train_analyses for o in a["objects"])
    _require(train_sides == {-1, 1}, "training must show both reflection sides")
    object_counts = set()
    for section, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            analysis = _analyze_pair(pair["input"], pair["output"], background, templates, axes)
            _require(analysis is not None, "invalid reflection pair")
            objects = analysis["objects"]
            object_counts.add(len(objects))
            _require(1 <= len(objects) <= 4, "object count")
            if section == "train":
                _require(2 <= len(objects) <= 4, "training object count")
            else:
                _require({o["orientation"] for o in objects} <= train_orientations, "unseen test orientation")
                _require(max(o["reach"] for o in objects) <= train_max_reach, "unseen test reach")
                _require({o["template"] for o in objects} <= {o["template"] for a in train_analyses for o in a["objects"]}, "unseen template color")
                _require({o["axis"] for o in objects} <= {o["axis"] for a in train_analyses for o in a["objects"]}, "unseen axis color")
    _require(len(object_counts) >= 2, "object count must vary within a puzzle")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, int(seed), num_train, num_test)
    for puzzle_attempt in range(80):
        parameters = _sample_puzzle_parameters(rng)
        puzzle = {"train": [], "test": []}
        seen = set()
        try:
            for section, count, regime in (("train", num_train, "source_train"), ("test", num_test, "target_test")):
                for index in range(count):
                    for duplicate_attempt in range(20):
                        pair_seed = int(rng.integers(0, 2**63, dtype=np.int64))
                        inp, out = _generate_pair(np.random.default_rng(pair_seed), regime, parameters, index, count)
                        key = _grid_key(inp)
                        if key not in seen:
                            seen.add(key)
                            puzzle[section].append({"input": inp, "output": out})
                            break
                    else:
                        raise RuntimeError("duplicate puzzle pairs")
            validate_generated_puzzle(puzzle)
            return puzzle
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("could not construct a valid puzzle")


def _puzzle_key(puzzle):
    return tuple((section, tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[section])) for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        roles = _infer_roles(puzzle["train"])
        assert roles is not None
        background, templates, axes, _ = roles
        for pair in puzzle["test"]:
            prediction = _analyze_pair(pair["input"].copy(), None, background, templates, axes)
            assert prediction is not None and np.array_equal(prediction["predicted"], pair["output"])
    assert _puzzle_key(generate_puzzle(314159, 7, 3)) == _puzzle_key(generate_puzzle(314159, 7, 3))
    assert _puzzle_key(generate_puzzle(17)) == _puzzle_key(generate_puzzle(17))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    for offset, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + offset, nt, ns)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
        validate_generated_puzzle(puzzle)
    for offset, (nt, ns) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + offset, nt, ns)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                   {"num_train": True}, {"num_test": 2.0}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    objects = []
    pair_counts, dimensions, layouts, role_assignments = set(), set(), set(), set()
    side_compositions, orientation_compositions = set(), set()
    mixed_orientations = mixed_palettes = repeated_accents = 0
    deep_recolored = mixed_crowded_disconnected = 0
    one_sided_pairs = overlapping_axes = staggered_axes = 0
    for puzzle in puzzles:
        bg, templates, axes, _ = _infer_roles(puzzle["train"])
        role_assignments.add((bg, tuple(sorted(templates)), tuple(sorted(axes))))
        puzzle_counts, puzzle_dimensions = set(), set()
        for pair in puzzle["train"] + puzzle["test"]:
            analysis = _analyze_pair(pair["input"], pair["output"], bg, templates, axes)
            assert analysis is not None
            group = analysis["objects"]
            objects.extend(group)
            pair_counts.add(len(group))
            puzzle_counts.add(len(group))
            dimensions.add(pair["input"].shape)
            puzzle_dimensions.add(pair["input"].shape)
            layouts.add((pair["input"].shape, tuple((o["orientation"], o["side"], o["anchor"]) for o in group)))
            sides = tuple(sorted(o["side"] for o in group))
            orientations = {o["orientation"] for o in group}
            palette = {o["template"] for o in group}
            side_compositions.add(sides)
            orientation_compositions.add(tuple(sorted(orientations)))
            one_sided_pairs += len(group) >= 2 and len(set(sides)) == 1
            mixed_orientations += len(orientations) == 2
            mixed_palettes += len(palette) == 2
            repeated_accents += len({o["accent"] for o in group}) < len(group)
            deep_recolored += len(palette) == 2 and any(o["reach"] >= 4 for o in group)
            mixed_crowded_disconnected += len(group) >= 3 and len(orientations) == 2 and any(not o["connected"] for o in group)
            for first in range(len(group)):
                for second in range(first + 1, len(group)):
                    left, right = group[first], group[second]
                    if left["orientation"] == right["orientation"]:
                        shared = max(left["anchor"][0], right["anchor"][0]) < min(left["anchor"][0] + left["length"], right["anchor"][0] + right["length"])
                        overlapping_axes += shared
                        staggered_axes += shared and left["anchor"][0] != right["anchor"][0]
        assert len(puzzle_counts) >= 2 and len(puzzle_dimensions) >= 2
    assert pair_counts == {1, 2, 3, 4}
    assert len(dimensions) >= 40 and len(layouts) >= 100 and len(role_assignments) >= 25
    assert {o["orientation"] for o in objects} == {"vertical", "horizontal"}
    assert {o["side"] for o in objects} == {-1, 1}
    assert {o["reach"] for o in objects} == {2, 3, 4, 5}
    assert {o["length"] for o in objects} == {3, 4, 5, 6, 7, 8}
    assert any(not o["connected"] for o in objects)
    assert any(o["occupancy"] < 0.5 for o in objects)
    assert any(o["occupancy"] > 0.7 for o in objects)
    assert len(side_compositions) >= 8 and len(orientation_compositions) == 3
    assert one_sided_pairs >= 30 and mixed_orientations >= 20
    assert mixed_palettes >= 30 and repeated_accents >= 10
    assert deep_recolored >= 20 and mixed_crowded_disconnected >= 5
    assert overlapping_axes >= 30 and staggered_axes >= 20
