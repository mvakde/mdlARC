"""Connector-guided assembly puzzles derived from the staged ARC example."""

import numpy as np


_SOURCE_PAIR_ROWS = (
    (("00000050000", "22205880000", "00500000566"),
     ("000000000", "222008666", "002888000")),
    (("00051500000", "22000000333", "05000005300"),
     ("000000000", "220000333", "021113300")),
    (("01500000220", "11005205200", "00000500000"),
     ("01122022", "11002220", "00000000")),
    (("050000000", "220510522", "000050000"),
     ("0211000", "2201222", "0000000")),
    (("05051005058", "22001053008", "00005000000"),
     ("02110000", "22010388", "00013308")),
)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _key(grid):
    return (grid.shape, grid.tobytes())


def _source_copy(pair):
    for input_rows, output_rows in _SOURCE_PAIR_ROWS:
        if (pair["input"].shape == (len(input_rows), len(input_rows[0]))
                and pair["output"].shape == (len(output_rows), len(output_rows[0]))
                and all(''.join(map(str, row)) == expected
                        for row, expected in zip(pair["input"], input_rows))
                and all(''.join(map(str, row)) == expected
                        for row, expected in zip(pair["output"], output_rows))):
            return True
    return False


def _components(cells):
    remaining = set(cells)
    found = []
    while remaining:
        component = {remaining.pop()}
        stack = list(component)
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        found.append(component)
    return found


def _encloses_background(component):
    top = min(r for r, _ in component)
    bottom = max(r for r, _ in component)
    left = min(c for _, c in component)
    right = max(c for _, c in component)
    empty = {(r, c) for r in range(top, bottom + 1)
             for c in range(left, right + 1) if (r, c) not in component}
    edge = {cell for cell in empty
            if cell[0] in (top, bottom) or cell[1] in (left, right)}
    reachable = set(edge)
    stack = list(edge)
    while stack:
        r, c = stack.pop()
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if neighbor in empty and neighbor not in reachable:
                reachable.add(neighbor)
                stack.append(neighbor)
    return bool(empty - reachable)


def _shape(rng, height, width, lead_row, tail_row):
    """Grow one connected monochrome payload around its end tabs."""
    markers = set()
    anchors = []
    if lead_row is None:
        anchors.append((int(rng.integers(height)), 0))
    else:
        markers.add((lead_row, 0))
        anchors.append((lead_row, 1))
    if tail_row is None:
        anchors.append((int(rng.integers(height)), width - 1))
    else:
        markers.add((tail_row, width - 1))
        anchors.append((tail_row, width - 2))
    if height > 1:
        low = 1 if lead_row is not None else 0
        high = width - (2 if tail_row is not None else 1)
        anchors += [(0, int(rng.integers(low, high + 1))),
                    (height - 1, int(rng.integers(low, high + 1)))]

    inner_left = 1 if lead_row is not None else 0
    inner_right = width - (2 if tail_row is not None else 1)
    ring = (height >= 3 and inner_right - inner_left >= 2
            and rng.random() < 0.18)
    if ring:
        payload = {(r, c) for r in range(height)
                   for c in range(inner_left, inner_right + 1)
                   if r in (0, height - 1) or c in (inner_left, inner_right)}
        # Endpoint anchors keep the same tight bounding box as the tabs.
        payload.update(anchors[:2])
    else:
        payload = {anchors[0]}
        for target in anchors[1:]:
            row, col = anchors[0]
            while (row, col) != target:
                if row != target[0] and col != target[1]:
                    move_row = bool(rng.integers(2))
                else:
                    move_row = row != target[0]
                row += (1 if target[0] > row else -1) if move_row else 0
                col += (1 if target[1] > col else -1) if not move_row else 0
                payload.add((row, col))
        available = {(r, c) for r in range(height) for c in range(width)} - markers
        density = float(rng.uniform(0.28, 0.94))
        target_size = max(len(payload), int(round(density * len(available))))
        while len(payload) < target_size:
            frontier = {near for r, c in payload
                        for near in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                        if near in available and near not in payload}
            if not frontier:
                break
            ordered = sorted(frontier)
            payload.add(ordered[int(rng.integers(len(ordered)))])
    _require(len(_components(payload)) == 1, "shape growth disconnected a payload")
    return payload, markers, ring


def _piece_widths(rng, count, canvas_height):
    widths = [2 if i in (0, count - 1) else 3 for i in range(count)]
    minimum = max(sum(widths), canvas_height + 1)
    maximum = min(26, 30 - 2 - (count - 1), sum(widths) + 11)
    target = int(rng.integers(minimum, maximum + 1))
    while sum(widths) < target:
        eligible = [i for i, value in enumerate(widths) if value < 8]
        widths[eligible[int(rng.integers(len(eligible)))]] += 1
    return widths


def _docking_rows(rng, count, height):
    style = int(rng.integers(4))
    if style == 0:
        return [int(rng.integers(height))] * (count - 1)
    if style == 1:
        first = int(rng.integers(height))
        direction = int(rng.choice([-1, 1]))
        return [min(height - 1, max(0, first + direction * i))
                for i in range(count - 1)]
    if style == 2:
        rows = [int(rng.integers(height))]
        for _ in range(count - 2):
            rows.append(min(height - 1, max(0, rows[-1] + int(rng.integers(-2, 3)))))
        return rows
    return [int(rng.integers(height)) for _ in range(count - 1)]


def _choose_colors(rng, count, palette, mode, unseen=None):
    palette = list(palette)
    if mode == "repeat":
        colors = [int(rng.choice(palette))]
        for _ in range(count - 1):
            colors.append(colors[-1] if rng.random() < 0.73
                          else int(rng.choice(palette)))
        if all(a != b for a, b in zip(colors, colors[1:])):
            colors[-1] = colors[-2]
    elif mode == "distinct" and len(palette) >= count:
        colors = [int(x) for x in rng.choice(palette, size=count, replace=False)]
    else:
        colors = [int(x) for x in rng.choice(palette, size=count, replace=True)]
    if unseen is not None:
        colors[int(rng.integers(count))] = int(unseen)
    return colors


def _make_pair(rng, connector, palette, axis, regime, count,
               color_mode="random", unseen=None, aligned=None, anchor=False,
               forbidden_background=None):
    height = int(rng.integers(2, 9))
    if anchor:
        height = max(3, height)
    widths = _piece_widths(rng, count, height)
    docking = _docking_rows(rng, count, height)
    if anchor:
        docking[0] = max(1, docking[0])
    piece_heights = []
    output_tops = []
    for i in range(count):
        linked = docking[max(0, i - 1):min(count - 1, i + 1)]
        minimum = max(linked) - min(linked) + 1
        if regime == "source_train" and rng.random() < 0.35:
            upper = min(height, max(minimum, 4))
        else:
            upper = height
        if anchor and i in (0, count - 1):
            upper = min(upper, height - 1)
        piece_height = int(rng.integers(minimum, upper + 1))
        lower_top = max(0, max(linked) - piece_height + 1)
        upper_top = min(height - piece_height, min(linked))
        if anchor and i == 0 and upper_top >= max(1, lower_top):
            lower_top = max(1, lower_top)
        top = int(rng.integers(lower_top, upper_top + 1))
        piece_heights.append(piece_height)
        output_tops.append(top)
    if aligned is None:
        aligned = bool(rng.random() < 0.2)
    input_tops = [output_tops[0]]
    for i in range(1, count):
        input_tops.append(output_tops[i] if aligned else
                          int(rng.integers(height - piece_heights[i] + 1)))
    if anchor and input_tops[-1] == output_tops[-1]:
        alternatives = [top for top in range(height - piece_heights[-1] + 1)
                        if top != output_tops[-1]]
        input_tops[-1] = alternatives[int(rng.integers(len(alternatives)))]

    colors = _choose_colors(rng, count, palette, color_mode, unseen)
    background_choices = [c for c in range(10) if c != connector and c not in colors
                          and c != forbidden_background]
    background = int(rng.choice(background_choices))
    shapes = []
    for i in range(count):
        lead = None if i == 0 else docking[i - 1] - output_tops[i]
        tail = None if i == count - 1 else docking[i] - output_tops[i]
        shapes.append(_shape(rng, piece_heights[i], widths[i], lead, tail))

    compact_width = sum(widths)
    gaps = [1] * (count - 1)
    left_margin = 1
    right_margin = 1
    slack = 30 - compact_width - sum(gaps) - left_margin - right_margin
    additions = int(rng.integers(min(9, slack) + 1))
    for _ in range(additions):
        slot = int(rng.integers(count + 1))
        if slot == 0:
            left_margin += 1
        elif slot == count:
            right_margin += 1
        else:
            gaps[slot - 1] += 1
    input_width = compact_width + sum(gaps) + left_margin + right_margin
    inp = np.full((height, input_width), background, dtype=np.int8)
    out = np.full((height, compact_width), background, dtype=np.int8)
    input_col = left_margin
    output_col = 0
    for i, ((payload, markers, _), width, color) in enumerate(zip(shapes, widths, colors)):
        for r, c in payload:
            inp[input_tops[i] + r, input_col + c] = color
            out[output_tops[i] + r, output_col + c] = color
        for r, c in markers:
            inp[input_tops[i] + r, input_col + c] = connector
            out[output_tops[i] + r, output_col + c] = color
        input_col += width + (gaps[i] if i < count - 1 else 0)
        output_col += width
    if axis == "vertical":
        inp = inp.T.copy()
        out = out.T.copy()
    return {"input": inp, "output": out}


def _analyze_pair(pair, connector):
    inp, out = pair["input"], pair["output"]
    _require(inp.shape[0] != inp.shape[1], "input must have a visible assembly axis")
    axis = "horizontal" if inp.shape[1] > inp.shape[0] else "vertical"
    if axis == "vertical":
        inp, out = inp.T, out.T
    height, width = inp.shape
    background = int(inp[0, 0])
    _require(all(int(inp[r, c]) == background
                 for r, c in ((0, 0), (0, width - 1), (height - 1, 0),
                              (height - 1, width - 1))),
             "input corners do not establish a unique background")
    _require(background != connector, "connector is the background")
    _require(out.shape[0] == height and out.shape[1] < width,
             "assembly must preserve cross-axis extent and crop the chain axis")
    _require(out.shape[1] > height, "compact chain does not show its axis")
    coords = {(r, c) for r in range(height) for c in range(width)
              if int(inp[r, c]) != background}
    components = _components(coords)
    components.sort(key=lambda part: min(c for _, c in part))
    _require(2 <= len(components) <= 6, "piece count is outside the construction range")
    pieces = []
    holes = False
    densities = []
    last_right = -2
    for i, component in enumerate(components):
        top = min(r for r, _ in component)
        left = min(c for _, c in component)
        bottom = max(r for r, _ in component)
        right = max(c for _, c in component)
        _require(left >= last_right + 2, "pieces overlap along input axis")
        last_right = right
        markers = {(r, c) for r, c in component if int(inp[r, c]) == connector}
        payload = component - markers
        _require(payload and len(_components(payload)) == 1,
                 "payload is empty or disconnected")
        holes |= _encloses_background(component)
        densities.append(len(payload) / ((bottom - top + 1) * (right - left + 1)))
        colors = {int(inp[r, c]) for r, c in payload}
        _require(len(colors) == 1 and background not in colors and connector not in colors,
                 "payload is not one legal local color")
        color = next(iter(colors))
        _require(len(markers) == (1 if i in (0, len(components) - 1) else 2),
                 "wrong number of tabs on a piece")
        for r, c in markers:
            _require(any((r + dr, c + dc) in payload
                         for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))),
                     "tab has no unique payload attachment")
        lead = None if i == 0 else (next((r - top for r, c in markers if c == left), None))
        tail = None if i == len(components) - 1 else (
            next((r - top for r, c in markers if c == right), None))
        _require((i == 0 or lead is not None) and
                 (i == len(components) - 1 or tail is not None),
                 "tabs must occupy the correct extremes")
        if i == 0:
            _require(all(c == right for _, c in markers), "first tab is not trailing")
        elif i == len(components) - 1:
            _require(all(c == left for _, c in markers), "last tab is not leading")
        else:
            _require({c for _, c in markers} == {left, right} and left < right,
                     "intermediate tabs are not on opposite extremes")
        pieces.append({"top": top, "left": left, "height": bottom - top + 1,
                       "width": right - left + 1, "lead": lead, "tail": tail,
                       "color": color, "cells": component})
    expected = np.full((height, sum(piece["width"] for piece in pieces)),
                       background, dtype=np.int8)
    top = pieces[0]["top"]
    col = 0
    moves = []
    levels = []
    for i, piece in enumerate(pieces):
        _require(0 <= top and top + piece["height"] <= height,
                 "docked piece leaves the canvas")
        for r, c in piece["cells"]:
            expected[top + r - piece["top"], col + c - piece["left"]] = piece["color"]
        if i + 1 < len(pieces):
            level = top + piece["tail"]
            levels.append(level)
            next_top = level - pieces[i + 1]["lead"]
            moves.append(next_top - pieces[i + 1]["top"])
            top = next_top
        col += piece["width"]
    _require(np.array_equal(expected, out), "output is not the docked and recolored chain")
    _require(connector not in set(int(x) for x in np.unique(out)),
             "connector survived recoloring")
    return {"axis": axis, "background": background, "count": len(pieces),
            "colors": tuple(piece["color"] for piece in pieces),
            "heights": tuple(piece["height"] for piece in pieces),
            "widths": tuple(piece["width"] for piece in pieces),
            "levels": tuple(levels), "moves": tuple(moves),
            "first_top": pieces[0]["top"], "cross": height,
            "hole": holes, "max_density": max(densities),
            "margins": (pieces[0]["left"], width - 1 - last_right),
            "gaps": tuple(pieces[i + 1]["left"] -
                          (pieces[i]["left"] + pieces[i]["width"])
                          for i in range(len(pieces) - 1))}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "wrong puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "pair sections must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen = set()
    connectors = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "wrong pair schema")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                         np.issubdtype(grid.dtype, np.integer) and
                         all(1 <= size <= 30 for size in grid.shape) and
                         0 <= int(grid.min()) <= int(grid.max()) <= 9,
                         "invalid ARC grid")
            _require(_key(pair["input"]) not in seen, "duplicate input")
            seen.add(_key(pair["input"]))
            _require(not np.array_equal(pair["input"], pair["output"]),
                     "unchanged pair")
            _require(not _source_copy(pair), "exact source copy")
            if section == "train":
                vanished = set(int(x) for x in np.unique(pair["input"])) - set(
                    int(x) for x in np.unique(pair["output"]))
                _require(len(vanished) == 1, "training does not reveal the connector")
                connectors.update(vanished)
    _require(len(connectors) == 1, "connector color changes within puzzle")
    connector = next(iter(connectors))
    analyzed = {section: [_analyze_pair(pair, connector) for pair in puzzle[section]]
                for section in ("train", "test")}
    _require(any(any(move != 0 for move in item["moves"])
                 for item in analyzed["train"]),
             "training does not demonstrate cross-axis docking movement")
    _require(any(item["first_top"] > 0 for item in analyzed["train"]),
             "training does not demonstrate the first-piece anchor")
    train_axes = {item["axis"] for item in analyzed["train"]}
    test_axes = {item["axis"] for item in analyzed["test"]}
    _require(test_axes <= train_axes, "test orientation has no training example")
    _require(len({item["background"] for item in analyzed["train"]}) >= 2,
             "training does not show that background can vary by pair")
    _require(any(any(a == b for a, b in zip(item["colors"], item["colors"][1:]))
                 for item in analyzed["train"]),
             "training does not demonstrate neighboring same-color pieces")
    _require(any(len(set(item["colors"])) == item["count"]
                 for item in analyzed["train"]),
             "training does not demonstrate distinct piece colors")
    training_colors = set().union(*(set(item["colors"]) for item in analyzed["train"]))
    _require(any(set(item["colors"]) - training_colors for item in analyzed["test"]),
             "test does not exercise locally inferable unseen payload colors")


def _checked_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its legal range")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + (seed // 4) % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    connector = int(rng.integers(10))
    available = [c for c in range(10) if c != connector]
    rng.shuffle(available)
    train_palette = available[:int(rng.integers(3, 7))]
    unseen_palette = [c for c in available if c not in train_palette]
    axis_mode = int(rng.integers(3))
    if axis_mode == 0:
        train_axes = ["horizontal"] * train_count
    elif axis_mode == 1:
        train_axes = ["vertical"] * train_count
    else:
        train_axes = ["horizontal", "vertical"] + [
            "horizontal" if rng.random() < 0.5 else "vertical"
            for _ in range(train_count - 2)]
        rng.shuffle(train_axes)
    test_axes = [str(rng.choice(train_axes)) for _ in range(test_count)]
    train_plans = ["repeat", "distinct"] + [
        str(rng.choice(["repeat", "distinct", "random"]))
        for _ in range(train_count - 2)]
    rng.shuffle(train_plans)
    pair_specs = [("train", i) for i in range(train_count)] + [
        ("test", i) for i in range(test_count)]
    puzzle = {"train": [], "test": []}
    used = set()
    for section, index in pair_specs:
        axis = train_axes[index] if section == "train" else test_axes[index]
        regime = "source_train" if section == "train" else "target_test"
        count_weights = [2, 3, 4, 5] if section == "train" else [2, 3, 4, 5, 6]
        probabilities = [0.19, 0.33, 0.32, 0.16] if section == "train" else [
            0.10, 0.17, 0.31, 0.27, 0.15]
        unseen = (int(rng.choice(unseen_palette))
                  if section == "test" and index == 0 and unseen_palette else None)
        mode = train_plans[index] if section == "train" else str(
            rng.choice(["repeat", "distinct", "random"]))
        accepted = None
        for attempt in range(80):
            count = int(rng.choice(count_weights, p=probabilities))
            if section == "train" and mode == "distinct" and count > len(train_palette):
                count = len(train_palette)
            if section == "train" and index == 0 and train_count >= 3 and seed % 4 == 0:
                count = 3
            if section == "test" and index == 0 and seed % 4 == 0:
                count = 4
            previous_background = (int(puzzle["train"][0]["input"][0, 0])
                                   if section == "train" and index == 1 else None)
            candidate = _make_pair(rng, connector, train_palette if section == "train"
                                   else available, axis, regime, count, mode, unseen,
                                   aligned=(False if section == "train" and index == 0
                                            else None),
                                   anchor=(section == "train" and index == 0),
                                   forbidden_background=previous_background)
            if _key(candidate["input"]) not in used and not _source_copy(candidate):
                accepted = candidate
                break
        _require(accepted is not None, "could not create a unique pair")
        used.add(_key(accepted["input"]))
        puzzle[section].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4,
             "automatic training counts lack variation")
    _require(len({len(p["test"]) for p in puzzles}) >= 3,
             "automatic test counts lack variation")
    _require(len({_key(p["train"][0]["input"]) for p in puzzles}) >= 40,
             "seeds lack structural variation")
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    observed = {"train": [], "test": []}
    puzzle_axes = []
    backgrounds = set()
    connectors = set()
    varied_lengths = 0
    for puzzle in puzzles:
        vanished = set(np.unique(puzzle["train"][0]["input"])).difference(
            np.unique(puzzle["train"][0]["output"]))
        connector = int(next(iter(vanished)))
        connectors.add(connector)
        local = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                feature = _analyze_pair(pair, connector)
                observed[section].append(feature)
                local.append(feature)
                backgrounds.add(feature["background"])
        puzzle_axes.append({feature["axis"] for feature in local})
        varied_lengths += len({feature["count"] for feature in local}) > 1
    all_features = observed["train"] + observed["test"]
    _require(len(backgrounds) == 10 and len(connectors) == 10,
             "semantic color assignments do not cover all ARC colors")
    _require(varied_lengths >= 30, "piece counts seldom vary within a puzzle")
    _require(sum(len(axes) == 2 for axes in puzzle_axes) >= 5 and
             sum(len(axes) == 1 for axes in puzzle_axes) >= 5,
             "single and mixed axis puzzles are not both represented")
    for section, allowed in (("train", {2, 3, 4, 5}),
                             ("test", {2, 3, 4, 5, 6})):
        features = observed[section]
        _require({feature["count"] for feature in features} == allowed,
                 f"{section} does not cover the supported chain lengths")
        _require({feature["axis"] for feature in features} ==
                 {"horizontal", "vertical"},
                 f"{section} does not cover both assembly axes")
        _require({feature["cross"] for feature in features} == set(range(2, 9)),
                 f"{section} canvas sizes are narrow")
        _require(sum(feature["hole"] for feature in features) >= 8,
                 f"{section} lacks enclosed-hole silhouettes")
        _require(sum(max(feature["widths"]) >= 5 for feature in features) >= 20,
                 f"{section} lacks wide pieces")
        _require(sum(feature["heights"][0] >= 3 or feature["heights"][-1] >= 3
                     for feature in features) >= 20,
                 f"{section} lacks tall endpoints")
        _require(sum(any(a == b for a, b in zip(feature["colors"],
                                                feature["colors"][1:]))
                     for feature in features) >= 15,
                 f"{section} lacks adjacent repeated payload colors")
        _require(sum(any(feature["colors"][i] == feature["colors"][i + 1] ==
                         feature["colors"][i + 2]
                         for i in range(feature["count"] - 2))
                     for feature in features) >= 3,
                 f"{section} lacks longer same-color runs")
        _require(sum(all(move == 0 for move in feature["moves"])
                     for feature in features) >= 8,
                 f"{section} lacks already-aligned pieces")
        _require(sum(len(feature["levels"]) > 2 and any(
            (b - a) * (c - b) < 0 for a, b, c in zip(
                feature["levels"], feature["levels"][1:],
                feature["levels"][2:])) for feature in features) >= 5,
            f"{section} lacks docking reversals")
    _require(max(max(feature["gaps"]) for feature in all_features) >= 4 and
             max(max(feature["margins"]) for feature in all_features) >= 4,
             "spacing and outer margins do not scale")
    _require(sum(feature["max_density"] >= 0.75 for feature in all_features) >= 30,
             "dense payload silhouettes are rare")
    _require(sum(feature["hole"] and any(a == b for a, b in zip(
        feature["colors"], feature["colors"][1:]))
        for feature in all_features) >= 8,
        "hole geometry does not combine with neighboring repeated colors")
    _require(sum(any(a == b for a, b in zip(feature["colors"],
                                                feature["colors"][1:])) and
                 len(set(feature["levels"])) > 1
                 for feature in observed["train"]) >= 10,
             "repeated colors remain coupled to flat docking levels")
    _require(sum(len(set(feature["colors"])) == feature["count"] and
                 len(feature["levels"]) > 1 and
                 len(set(feature["levels"])) == 1
                 for feature in observed["train"]) >= 10,
             "distinct colors remain coupled to changing docking levels")
    _require(sum(len(feature["levels"]) > 1 and
                 len(set(feature["levels"])) == 1 and
                 any(move != 0 for move in feature["moves"])
                 for feature in all_features) >= 10,
             "equal docking levels never require moving a piece")
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(7000 + train * 10 + test, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts ignored")
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(7100 + (train or 0) + (test or 0), train, test)
        _require((train is None or len(puzzle["train"]) == train) and
                 (test is None or len(puzzle["test"]) == test),
                 "partial count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2)):
        try:
            generate_puzzle(7200, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    first = generate_puzzle(7777, 5, 2)
    second = generate_puzzle(7777, 5, 2)
    _require(all(np.array_equal(a[k], b[k]) for section in ("train", "test")
                 for a, b in zip(first[section], second[section])
                 for k in ("input", "output")), "generation is not deterministic")
    connector = next(iter(set(int(x) for x in np.unique(first["train"][0]["input"])) -
                          set(int(x) for x in np.unique(first["train"][0]["output"]))))
    _require(_analyze_pair(first["test"][0], connector)["count"] >= 2,
             "independent input reconstruction failed")
