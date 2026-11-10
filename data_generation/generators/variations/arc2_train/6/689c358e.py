import numpy as np


_DIRECTIONS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_SOURCE_PAIRS = {
    (
        ("66666666666", "67777777776", "67777777876", "67777778886",
         "67797777876", "67999977876", "67797772776", "67757222276",
         "67555572776", "67757777776", "66666666666"),
        ("66666666866", "67777777776", "67777777876", "67777778886",
         "67797777876", "97999977870", "67797772776", "07757222272",
         "57555572770", "67757777776", "66666666066"),
    ),
    (
        ("66666666666", "67777777776", "67778777576", "67788875556",
         "67778777576", "67778777576", "67777772776", "67797222276",
         "67999972776", "67797777776", "66666666666"),
        ("66668666566", "67777777776", "67778777576", "67788875556",
         "67778777576", "67778777576", "67777772776", "07797222272",
         "97999972770", "67797777776", "66660666066"),
    ),
    (
        ("66666666666", "67777777776", "67577777776", "65557727776",
         "67577222776", "67577727976", "67777727976", "67778779996",
         "67788887976", "67778777776", "66666666666"),
        ("66566626066", "67777777776", "67577777776", "65557727776",
         "67577222776", "67577727976", "67777727976", "67778779996",
         "87788887970", "67778777776", "66066606966"),
    ),
}
_SOURCE_INPUTS = {pair[0] for pair in _SOURCE_PAIRS}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _count_argument(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count_argument("num_train", num_train, 2, 10)
    num_test = _count_argument("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_train = int(rng.choice(np.arange(2, 11), p=[.20, .17, .15, .13, .11, .08, .06, .05, .05]))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _footprint(head, direction, shaft, wing):
    row, col = head
    dr, dc = _DIRECTIONS[direction]
    pr, pc = -dc, dr
    cells = {(row + k * pr, col + k * pc) for k in range(-wing, wing + 1)}
    cells.add((row + dr, col + dc))
    cells.update((row - k * dr, col - k * dc) for k in range(1, shaft + 1))
    return cells


def _endpoints(shape, head, direction):
    height, width = shape
    row, col = head
    if direction == "U":
        return (0, col), (height - 1, col)
    if direction == "D":
        return (height - 1, col), (0, col)
    if direction == "L":
        return (row, 0), (row, width - 1)
    return (row, width - 1), (row, 0)


def _dimension(rng):
    band = int(rng.choice(4, p=[.27, .37, .28, .08]))
    if band == 0:
        return int(rng.integers(9, 13))
    if band == 1:
        return int(rng.integers(13, 22))
    if band == 2:
        return int(rng.integers(22, 30))
    return 30


def _dimensions(rng, scale_profile):
    if scale_profile == "compact":
        height = int(rng.integers(9, 15))
        width = height if rng.random() < .28 else int(rng.integers(9, 15))
    elif scale_profile == "spacious":
        height = int(rng.integers(18, 31))
        width = height if rng.random() < .24 else int(rng.integers(18, 31))
    else:
        height = _dimension(rng)
        width = height if rng.random() < .24 else _dimension(rng)
    return height, width


def _arrow_count(rng, regime, scale_profile):
    if scale_profile == "compact":
        weights = [.32, .40, .28] if regime == "source_train" else [.27, .36, .37]
        return int(rng.choice(np.arange(2, 5), p=weights))
    weights = ([.22, .31, .27, .15, .05] if regime == "source_train"
               else [.16, .25, .26, .21, .12])
    return int(rng.choice(np.arange(2, 7), p=weights))


def _palette(rng, void_color, stable_roles):
    available = [color for color in range(10) if color != void_color]
    if stable_roles is None:
        frame, field = (int(x) for x in rng.choice(available, 2, replace=False))
    else:
        frame, field = stable_roles
    arrow_pool = [color for color in available if color not in (frame, field)]
    return frame, field, arrow_pool


def _specs(rng, count, palette, regime, pattern=None, allowed_directions=None,
           required_direction=None, required_shaft=None, colors_override=None):
    frame, field, arrow_pool = palette
    if colors_override is None:
        repeat = count >= 3 and rng.random() < .32
        distinct_count = count - int(repeat)
        colors = [int(x) for x in rng.choice(arrow_pool, distinct_count, replace=False)]
        if repeat:
            colors.append(int(rng.choice(colors)))
        rng.shuffle(colors)
    else:
        colors = list(colors_override)
    specs = []
    for index in range(count):
        if pattern is None:
            choices = allowed_directions if allowed_directions else tuple(_DIRECTIONS)
            direction = required_direction if index == 0 and required_direction else str(rng.choice(choices))
            compact_rate = .23 if regime == "source_train" else .19
            shaft = (required_shaft if index == 0 and required_shaft is not None else
                     (0 if rng.random() < compact_rate else int(rng.integers(2, 8))))
            wing = 2 if shaft >= 2 and rng.random() < .16 else 1
        else:
            direction, shaft, wing = pattern[index]
        specs.append((colors[index], direction, shaft, wing, rng.random() < .30))
    return specs


def _place_arrows(rng, shape, specs):
    height, width = shape
    placed = []
    occupied = set()
    endpoints = set()
    order = sorted(range(len(specs)), key=lambda i: -(4 + specs[i][2] + 2 * (specs[i][3] - 1)))
    for index in order:
        color, direction, shaft, wing, prefer_near = specs[index]
        candidates = []
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                head = (row, col)
                cells = _footprint(head, direction, shaft, wing)
                if all(1 <= r < height - 1 and 1 <= c < width - 1 for r, c in cells):
                    tip, tail = _endpoints(shape, head, direction)
                    near = any(r in (1, height - 2) or c in (1, width - 2) for r, c in cells)
                    candidates.append((head, cells, tip, tail, near))
        rng.shuffle(candidates)
        if prefer_near:
            candidates.sort(key=lambda item: not item[4])
        if placed and rng.random() < .30:
            heads = [item[1] for item in placed]
            candidates.sort(key=lambda item: min(max(abs(item[0][0] - h[0]), abs(item[0][1] - h[1])) for h in heads))
        chosen = None
        for head, cells, tip, tail, _ in candidates:
            if tip in endpoints or tail in endpoints or cells & occupied:
                continue
            if any(color == old_color and any(
                max(abs(r - rr), abs(c - cc)) <= 1
                for r, c in cells for rr, cc in old_cells
            ) for old_color, _, old_cells in placed):
                continue
            chosen = (color, head, cells, tip, tail)
            break
        if chosen is None:
            return None
        placed.append((color, head, cells))
        occupied.update(cells)
        endpoints.update((chosen[3], chosen[4]))
    return placed


def _make_pair(rng, regime, void_color, stable_roles, scale_profile,
               forced_count=None, pattern=None, allowed_directions=None,
               required_direction=None, required_shaft=None, fixed_shape=None,
               fixed_palette=None, colors_override=None):
    for _ in range(60):
        shape = fixed_shape if fixed_shape is not None else _dimensions(rng, scale_profile)
        count = len(pattern) if pattern is not None else (
            forced_count if forced_count is not None else _arrow_count(rng, regime, scale_profile))
        palette = fixed_palette if fixed_palette is not None else _palette(rng, void_color, stable_roles)
        specs = _specs(rng, count, palette, regime, pattern, allowed_directions,
                       required_direction, required_shaft, colors_override)
        placed = _place_arrows(rng, shape, specs)
        if placed is None:
            continue
        frame, field, _ = palette
        input_grid = np.full(shape, field, dtype=np.int8)
        input_grid[0, :] = frame
        input_grid[-1, :] = frame
        input_grid[:, 0] = frame
        input_grid[:, -1] = frame
        for color, _, cells in placed:
            for row, col in cells:
                input_grid[row, col] = color
        output_grid = input_grid.copy()
        # Placement order differs from spec order, so recognize the direction from each shape.
        for color, head, cells in placed:
            _, direction, _, _ = _recognize_arrow(cells)
            tip, tail = _endpoints(shape, head, direction)
            output_grid[tip] = color
            output_grid[tail] = void_color
        try:
            _analyze_pair(input_grid, output_grid)
        except AssertionError:
            continue
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not place a valid set of arrows")


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = remaining.pop()
        component = {start}
        frontier = [start]
        while frontier:
            row, col = frontier.pop()
            for next_cell in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if next_cell in remaining:
                    remaining.remove(next_cell)
                    component.add(next_cell)
                    frontier.append(next_cell)
        result.append(component)
    return result


def _recognize_arrow(cells):
    candidates = []
    for wing in (1, 2):
        for shaft in (0, 2, 3, 4, 5, 6, 7):
            if wing == 2 and shaft == 0:
                continue
            if len(cells) != 2 * wing + 2 + shaft:
                continue
            for head in cells:
                for direction in _DIRECTIONS:
                    if _footprint(head, direction, shaft, wing) == cells:
                        candidates.append((head, direction, shaft, wing))
    _require(len(candidates) == 1, "arrow has no unique geometric direction")
    return candidates[0]


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(all(1 <= x <= 30 for x in grid.shape), f"{label} has illegal dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must have integer cells")
    _require(np.all((grid >= 0) & (grid <= 9)), f"{label} has illegal colors")


def _analyze_input(grid):
    _validate_grid(grid, "input")
    height, width = grid.shape
    _require(height >= 9 and width >= 9, "arrow grid is too small")
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    frame = int(border[0])
    _require(np.all(border == frame), "frame is incomplete")
    interior = grid[1:-1, 1:-1]
    _require(not np.any(interior == frame), "frame color appears inside")
    values, counts = np.unique(interior, return_counts=True)
    _require(np.count_nonzero(counts == max(counts)) == 1, "field is not identifiable")
    field = int(values[int(np.argmax(counts))])
    arrows = []
    for value in values:
        color = int(value)
        if color == field:
            continue
        all_cells = {tuple(int(x) for x in coord) for coord in np.argwhere(grid == color)}
        for cells in _components(all_cells):
            head, direction, shaft, wing = _recognize_arrow(cells)
            tip, tail = _endpoints(grid.shape, head, direction)
            arrows.append({"color": color, "head": head, "direction": direction,
                           "shaft": shaft, "wing": wing, "cells": cells,
                           "tip": tip, "tail": tail})
    _require(2 <= len(arrows) <= 6, "arrow count is outside the supported range")
    endpoints = [endpoint for arrow in arrows for endpoint in (arrow["tip"], arrow["tail"])]
    _require(len(set(endpoints)) == len(endpoints), "arrows claim conflicting frame cells")
    for index, first in enumerate(arrows):
        for second in arrows[index + 1:]:
            if first["color"] == second["color"]:
                _require(all(max(abs(r - rr), abs(c - cc)) > 1
                             for r, c in first["cells"] for rr, cc in second["cells"]),
                         "same-colored arrows are not separate")
    return {"frame": frame, "field": field, "arrows": arrows}


def _analyze_pair(input_grid, output_grid):
    analysis = _analyze_input(input_grid)
    _validate_grid(output_grid, "output")
    _require(input_grid.shape == output_grid.shape, "shape changed")
    tails = [int(output_grid[arrow["tail"]]) for arrow in analysis["arrows"]]
    _require(len(set(tails)) == 1, "tail colors disagree")
    void = tails[0]
    _require(void not in set(int(x) for x in np.unique(input_grid)), "void appears in input")
    expected = input_grid.copy()
    for arrow in analysis["arrows"]:
        expected[arrow["tip"]] = arrow["color"]
        expected[arrow["tail"]] = void
    _require(np.array_equal(output_grid, expected), "frame endpoint rule failed")
    analysis["void"] = void
    return analysis


def _signature(grid):
    return (grid.shape, np.asarray(grid, dtype=np.int8).tobytes())


def _rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _predict_from_training(train, test_input):
    voids = {_analyze_pair(pair["input"], pair["output"])["void"] for pair in train}
    _require(len(voids) == 1, "training pairs disagree about void color")
    void = next(iter(voids))
    analysis = _analyze_input(test_input)
    _require(void not in set(int(x) for x in np.unique(test_input)), "test uses void as an input color")
    prediction = test_input.copy()
    for arrow in analysis["arrows"]:
        prediction[arrow["tip"]] = arrow["color"]
        prediction[arrow["tail"]] = void
    return prediction


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    inputs = set()
    pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            input_grid, output_grid = pair["input"], pair["output"]
            analysis = _analyze_pair(input_grid, output_grid)
            analyses[split].append(analysis)
            input_key = _signature(input_grid)
            pair_key = (input_key, _signature(output_grid))
            _require(input_key not in inputs and pair_key not in pairs, "duplicate puzzle pair")
            inputs.add(input_key)
            pairs.add(pair_key)
            input_rows, output_rows = _rows(input_grid), _rows(output_grid)
            _require(input_rows not in _SOURCE_INPUTS and (input_rows, output_rows) not in _SOURCE_PAIRS,
                     "source pair copied")
    voids = {analysis["void"] for split in analyses for analysis in analyses[split]}
    _require(len(voids) == 1, "void must be one puzzle-level convention")
    _require(len({arrow["direction"] for analysis in analyses["train"]
                  for arrow in analysis["arrows"]}) >= 2, "training lacks direction evidence")
    _require(len({len(analysis["arrows"]) for split in analyses for analysis in analyses[split]}) >= 2,
             "puzzle lacks pair-level arrow-count variation")
    for pair in puzzle["test"]:
        _require(np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"]),
                 "test cannot be inferred from training")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    void = int(rng.integers(0, 10))
    scale_profile = str(rng.choice(("compact", "mixed", "spacious"), p=[.28, .54, .18]))
    stable_roles = None
    if rng.random() < (.64 if scale_profile == "compact" else .22):
        choices = [color for color in range(10) if color != void]
        stable_roles = tuple(int(x) for x in rng.choice(choices, 2, replace=False))
    training_pool = tuple(_DIRECTIONS)
    if rng.random() < .45:
        training_pool = tuple(str(x) for x in rng.choice(tuple(_DIRECTIONS), 3, replace=False))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    train_templates = []
    train_directions = set()
    for split, count in (("train", num_train), ("test", num_test)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pattern = None
            allowed = None
            required = None
            required_shaft = None
            target_count = None
            fixed_shape = None
            fixed_palette = None
            colors_override = None
            if split == "test":
                mode = pair_rng.random()
                if mode < .22:
                    template = train_templates[int(pair_rng.integers(0, len(train_templates)))]
                    pattern = tuple((direction, shaft, wing)
                                    for _, direction, shaft, wing in template[3])
                    colors_override = tuple(color for color, _, _, _ in template[3])
                    fixed_shape = template[0]
                    frame, field = template[1:3]
                    fixed_palette = (frame, field,
                                     [color for color in range(10)
                                      if color not in (void, frame, field)])
                elif mode < .52:
                    unseen = tuple(direction for direction in _DIRECTIONS
                                   if direction not in train_directions)
                    if unseen:
                        required = str(pair_rng.choice(unseen))
                        upper = 5 if scale_profile == "compact" else 7
                        prior_counts = {len(template[3]) for template in train_templates}
                        count_options = [n for n in range(2, upper) if n not in prior_counts]
                        if count_options:
                            target_count = int(pair_rng.choice(count_options))
                        prior_shafts = {shaft for template in train_templates
                                        for _, _, shaft, _ in template[3]}
                        shaft_options = [s for s in (0, 2, 3, 4, 5, 6, 7)
                                         if s not in prior_shafts]
                        if shaft_options:
                            required_shaft = int(pair_rng.choice(shaft_options))
                elif mode < .76:
                    allowed = tuple(direction for direction in _DIRECTIONS
                                    if direction in train_directions)
            else:
                allowed = training_pool
            for attempt in range(80):
                forced_count = target_count
                if split == "test" and index == count - 1:
                    prior_counts = {len(_analyze_input(pair["input"])["arrows"])
                                    for pair in puzzle["train"] + puzzle["test"]}
                    if len(prior_counts) == 1 and pattern is None:
                        upper = 5 if scale_profile == "compact" else 7
                        forced_count = int(pair_rng.choice([n for n in range(2, upper) if n not in prior_counts]))
                if split == "train" and index == count - 1 and len(puzzle["train"]) > 0:
                    prior_counts = {len(_analyze_input(pair["input"])["arrows"])
                                    for pair in puzzle["train"]}
                    if len(prior_counts) == 1:
                        upper = 5 if scale_profile == "compact" else 7
                        forced_count = int(pair_rng.choice([n for n in range(2, upper) if n not in prior_counts]))
                try:
                    pair = _make_pair(pair_rng, regime, void, stable_roles, scale_profile,
                                      forced_count, pattern, allowed, required, required_shaft,
                                      fixed_shape, fixed_palette, colors_override)
                except RuntimeError:
                    if attempt >= 3:
                        pattern = fixed_shape = fixed_palette = colors_override = None
                        required = required_shaft = target_count = None
                    continue
                analysis = _analyze_pair(pair["input"], pair["output"])
                if split == "train" and index == 0 and len({a["direction"] for a in analysis["arrows"]}) < 2:
                    continue
                if split == "test" and index == count - 1:
                    all_counts = {len(_analyze_input(p["input"])["arrows"])
                                  for p in puzzle["train"] + puzzle["test"]}
                    if len(all_counts) == 1 and len(analysis["arrows"]) in all_counts:
                        pattern = fixed_shape = fixed_palette = colors_override = None
                        continue
                key = _signature(pair["input"])
                if key in used_inputs:
                    continue
                used_inputs.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_directions.update(a["direction"] for a in analysis["arrows"])
                    train_templates.append((pair["input"].shape, analysis["frame"],
                                            analysis["field"], tuple(
                                                (a["color"], a["direction"], a["shaft"], a["wing"])
                                                for a in analysis["arrows"])))
                break
            else:
                raise RuntimeError(f"could not generate {split} pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(np.array_equal(a[k], b[k]) for split in ("train", "test")
               for a, b in zip(left[split], right[split]) for k in ("input", "output"))


def self_test() -> None:
    train_counts, test_counts, shapes, arrow_counts, voids = set(), set(), set(), set(), set()
    puzzle_signatures = set()
    layout_signatures = set()
    compact_train = compact_test = wide = same_color = near = close = down_train = 0
    varying_roles = stable_roles = seen_only_test = new_direction_test = 0
    placement_only = compound_shift = retained = changed_direction = 0
    all_extended_test = square = rectangle = small = side_30 = axis_crossing = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_signatures.add(tuple((_signature(pair["input"]), _signature(pair["output"]))
                                    for split in ("train", "test") for pair in puzzle[split]))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first = _analyze_pair(puzzle["train"][0]["input"], puzzle["train"][0]["output"])
        first_color_directions = {a["color"]: a["direction"] for a in first["arrows"]}
        voids.add(first["void"])
        roles = set()
        train_dirs = set()
        train_arrow_counts = set()
        train_shafts = set()
        train_full_signatures = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_pair(pair["input"], pair["output"])
                changed = {tuple(int(x) for x in coordinate)
                           for coordinate in np.argwhere(pair["input"] != pair["output"])}
                projected = {endpoint for arrow in analysis["arrows"]
                             for endpoint in (arrow["tip"], arrow["tail"])}
                _require(changed == projected and
                         np.array_equal(pair["input"][1:-1, 1:-1],
                                        pair["output"][1:-1, 1:-1]),
                         "interior preservation or endpoint coverage failed")
                roles.add((analysis["frame"], analysis["field"]))
                shapes.add(pair["input"].shape)
                arrow_counts.add(len(analysis["arrows"]))
                directions = [a["direction"] for a in analysis["arrows"]]
                full_signature = (pair["input"].shape, analysis["frame"], analysis["field"],
                                  tuple(sorted((a["color"], a["direction"], a["shaft"], a["wing"])
                                               for a in analysis["arrows"])))
                layout_signatures.add((pair["input"].shape, tuple(sorted(
                    (a["head"], a["direction"], a["shaft"], a["wing"])
                    for a in analysis["arrows"]))))
                height, width = pair["input"].shape
                square += height == width
                rectangle += height != width
                small += height <= 12 and width <= 12
                side_30 += height == 30 or width == 30
                axis_crossing += any(
                    any((r, a["head"][1]) in b["cells"] for r in range(1, height - 1))
                    if a["direction"] in ("U", "D") else
                    any((a["head"][0], c) in b["cells"] for c in range(1, width - 1))
                    for a in analysis["arrows"] for b in analysis["arrows"] if a is not b)
                for arrow in analysis["arrows"]:
                    if arrow["color"] in first_color_directions:
                        retained += arrow["direction"] == first_color_directions[arrow["color"]]
                        changed_direction += arrow["direction"] != first_color_directions[arrow["color"]]
                compact_train += split == "train" and any(a["shaft"] == 0 for a in analysis["arrows"])
                compact_test += split == "test" and any(a["shaft"] == 0 for a in analysis["arrows"])
                wide += any(a["wing"] == 2 for a in analysis["arrows"])
                same_color += len({a["color"] for a in analysis["arrows"]}) < len(analysis["arrows"])
                near += any(any(r in (1, pair["input"].shape[0] - 2) or
                                    c in (1, pair["input"].shape[1] - 2)
                                    for r, c in a["cells"]) for a in analysis["arrows"])
                close += any(min(max(abs(r - rr), abs(c - cc))
                                 for r, c in a["cells"] for rr, cc in b["cells"]) == 1
                             for i, a in enumerate(analysis["arrows"])
                             for b in analysis["arrows"][i + 1:])
                if split == "train":
                    train_dirs.update(directions)
                    train_arrow_counts.add(len(analysis["arrows"]))
                    train_shafts.update(a["shaft"] for a in analysis["arrows"])
                    train_full_signatures.add(full_signature)
                    down_train += "D" in directions
                else:
                    seen_only_test += set(directions) <= train_dirs
                    new_direction_test += bool(set(directions) - train_dirs)
                    placement_only += full_signature in train_full_signatures
                    compound_shift += (bool(set(directions) - train_dirs) and
                                       len(analysis["arrows"]) not in train_arrow_counts and
                                       any(a["shaft"] not in train_shafts for a in analysis["arrows"]))
                    all_extended_test += all(a["shaft"] >= 2 for a in analysis["arrows"])
                    _require(np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                                            pair["output"]), "prediction check failed")
        varying_roles += len(roles) > 1
        stable_roles += len(roles) == 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "pair-count diversity failed")
    _require(len(puzzle_signatures) >= 40, "cross-seed puzzles repeat")
    _require(len(shapes) >= 55 and min(min(s) for s in shapes) <= 11 and
             max(max(s) for s in shapes) == 30, "size diversity failed")
    _require(len(arrow_counts) >= 5 and len(layout_signatures) >= 140, "layout diversity failed")
    _require(len(voids) >= 7 and varying_roles >= 20 and stable_roles >= 2,
             "color-role diversity failed")
    _require(all(x > 0 for x in (compact_train, compact_test, wide, same_color,
                                 near, close, down_train, seen_only_test, new_direction_test,
                                 placement_only, compound_shift, retained, changed_direction,
                                 all_extended_test, square, rectangle, small, side_30,
                                 axis_crossing)),
             "review feature coverage failed")
    _require(_puzzles_equal(generate_puzzle(918273, 7, 3), generate_puzzle(918273, 7, 3)),
             "determinism failed")
    for seed, nt, nq in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (2001, None, 1), (2002, None, 4), (2003, 2, None),
                         (2004, 10, None)):
        puzzle = generate_puzzle(seed, nt, nq)
        _require((nt is None or len(puzzle["train"]) == nt) and
                 (nq is None or len(puzzle["test"]) == nq), "requested count failed")
        validate_generated_puzzle(puzzle)
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(3001, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts were accepted")
