"""Compositional instances of original-anchor lattice interval completion."""

import hashlib

import numpy as np


_SOURCE_INPUT_DIGESTS = frozenset({
    "e393d4d1759bb5870173439b95bdd8f13053f04c1463d4ecf92ffb2e114c8e69",
    "6a9d4038d6a45cc882104d606f0182a0df46ba18a7e28bc1175e711f1328fd62",
    "8c8d62ea374ca9fbf5f356a2b5f72eb36b3fd0047affbdfc64dac4aead39dfc9",
    "5c2a99aec7d05a30d56c6fe545c4ae9ba6163d8b3ef047017e571a4f10b7c141",
})
_RICH_KINDS = ("disjoint", "tee", "cross", "rectangle")
_CELL_SIZES = ((2, 2), (2, 2), (2, 2), (1, 1), (2, 3), (3, 2),
               (1, 2), (2, 1), (3, 3))


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _positions(coarse, color):
    return {(int(r), int(c)) for r, c in np.argwhere(coarse == color)}


def _claims_for_positions(positions):
    """Use only original anchors, never cells painted by another interval."""
    claims = set(positions)
    for row in {r for r, _ in positions}:
        columns = [c for r, c in positions if r == row]
        if len(columns) > 1:
            claims.update((row, c) for c in range(min(columns), max(columns) + 1))
    for column in {c for _, c in positions}:
        rows = [r for r, c in positions if c == column]
        if len(rows) > 1:
            claims.update((r, column) for r in range(min(rows), max(rows) + 1))
    return claims


def _complete_coarse_grid(coarse, background):
    result = coarse.copy()
    owner = {}
    for value in np.unique(coarse):
        color = int(value)
        if color == background:
            continue
        for position in _claims_for_positions(_positions(coarse, color)):
            assert position not in owner or owner[position] == color
            assert int(coarse[position]) in (background, color)
            owner[position] = color
    for position, color in owner.items():
        result[position] = color
    return result


def _choose_pair(rng, length, gap=2):
    options = [(a, b) for a in range(length) for b in range(a + gap, length)]
    if not options:
        raise ValueError("lattice too small")
    return options[int(rng.integers(len(options)))]


def _make_motif(rng, kind, rows, columns):
    if kind in ("line", "line_h", "line_v"):
        horizontal = (bool(rng.integers(2)) if kind == "line" else kind == "line_h")
        if horizontal:
            row = int(rng.integers(rows))
            a, b = _choose_pair(rng, columns)
            return {(row, a), (row, b)}
        column = int(rng.integers(columns))
        a, b = _choose_pair(rng, rows)
        return {(a, column), (b, column)}
    if kind in ("collinear", "collinear_h", "collinear_v"):
        horizontal = (bool(rng.integers(2)) if kind == "collinear" else kind == "collinear_h")
        length = columns if horizontal else rows
        a, b, c = sorted(int(x) for x in rng.choice(length, size=3, replace=False))
        if c - a < 3:
            raise ValueError("no missing interval")
        fixed = int(rng.integers(rows if horizontal else columns))
        return ({(fixed, x) for x in (a, b, c)} if horizontal
                else {(x, fixed) for x in (a, b, c)})
    if kind == "ell":
        ra, rb = _choose_pair(rng, rows)
        ca, cb = _choose_pair(rng, columns)
        corner_row = (ra, rb)[int(rng.integers(2))]
        corner_col = (ca, cb)[int(rng.integers(2))]
        other_row = rb if corner_row == ra else ra
        other_col = cb if corner_col == ca else ca
        return {(corner_row, corner_col), (other_row, corner_col),
                (corner_row, other_col)}
    if kind in ("off_axis", "off_axis_h", "off_axis_v"):
        horizontal = (bool(rng.integers(2)) if kind == "off_axis" else kind == "off_axis_h")
        if horizontal:
            high, low = _choose_pair(rng, rows)
            left, right = _choose_pair(rng, columns)
            middle = int(rng.integers(left + 1, right))
            return {(high, middle), (low, left), (low, right)}
        left, right = _choose_pair(rng, columns)
        high, low = _choose_pair(rng, rows)
        middle = int(rng.integers(high + 1, low))
        return {(middle, left), (high, right), (low, right)}
    if kind == "disjoint":
        r1, r2 = _choose_pair(rng, rows, 1)
        a, b = _choose_pair(rng, columns)
        c, d = _choose_pair(rng, columns)
        if len({a, b, c, d}) != 4 or not (b < c or d < a):
            raise ValueError("runs are not disjoint")
        return {(r1, a), (r1, b), (r2, c), (r2, d)}
    if kind == "tee":
        bar, end = int(rng.integers(rows)), int(rng.integers(rows))
        if abs(bar - end) < 2:
            raise ValueError("stem too short")
        left, right = _choose_pair(rng, columns, 3)
        middle = int(rng.integers(left + 1, right))
        return {(bar, left), (bar, middle), (bar, right), (end, middle)}
    if kind == "cross":
        high, low = _choose_pair(rng, rows, 3)
        left, right = _choose_pair(rng, columns, 3)
        center_row = int(rng.integers(high + 1, low))
        center_col = int(rng.integers(left + 1, right))
        return {(high, center_col), (center_row, center_col),
                (low, center_col), (center_row, left), (center_row, right)}
    if kind == "rectangle":
        high, low = _choose_pair(rng, rows)
        left, right = _choose_pair(rng, columns)
        return {(high, left), (high, right), (low, left), (low, right)}
    if kind == "inert":
        count = int(rng.integers(1, min(3, rows, columns) + 1))
        rr = rng.choice(rows, size=count, replace=False)
        cc = rng.choice(columns, size=count, replace=False)
        return {(int(r), int(c)) for r, c in zip(rr, cc)}
    if kind == "complete":
        if rng.integers(2):
            row = int(rng.integers(rows))
            start = int(rng.integers(columns - 1))
            return {(row, start), (row, start + 1)}
        column = int(rng.integers(columns))
        start = int(rng.integers(rows - 1))
        return {(start, column), (start + 1, column)}
    raise ValueError("unknown motif")


def _layout(rng, rows, columns, required, group_count, interior_only=False, orientation_hint=None):
    groups, claimed = [], set()
    sample_rows = rows - 2 if interior_only else rows
    sample_columns = columns - 2 if interior_only else columns
    for index in range(group_count):
        kind = required if index == 0 else str(rng.choice(
            ("line", "collinear", "ell", "inert", "inert", "complete")))
        if kind in ("line", "collinear") and orientation_hint is not None:
            kind += "_h" if orientation_hint == "horizontal" else "_v"
        for attempt in range(180):
            if attempt == 100 and index > 0:
                kind = "inert"
            try:
                points = _make_motif(rng, kind, sample_rows, sample_columns)
            except ValueError:
                continue
            if interior_only:
                points = {(r + 1, c + 1) for r, c in points}
            claims = _claims_for_positions(points)
            if claims.isdisjoint(claimed):
                groups.append((kind, points))
                claimed.update(claims)
                break
        else:
            raise ValueError("could not place nonconflicting groups")
    return groups


def _geometry(rng, regime, index, size_mode, interior_only=False):
    cell_h, cell_w = _CELL_SIZES[int(rng.integers(len(_CELL_SIZES)))]
    max_rows = min(14, 31 // (cell_h + 1))
    max_columns = min(14, 31 // (cell_w + 1))
    if regime == "source_train" and index < 2:
        rlo, clo, rhi, chi = 7, 7, min(9, max_rows), min(9, max_columns)
    elif regime == "target_test" and size_mode == "larger":
        rlo, clo, rhi, chi = min(8, max_rows), min(8, max_columns), max_rows, max_columns
    elif regime == "target_test" and size_mode == "smaller":
        rlo, clo, rhi, chi = 5, 5, min(7, max_rows), min(7, max_columns)
    elif regime == "target_test" and size_mode == "equal":
        rlo, clo, rhi, chi = min(7, max_rows), min(7, max_columns), min(9, max_rows), min(9, max_columns)
    else:
        rlo, clo, rhi, chi = 5, 5, max_rows, max_columns
    if interior_only:
        rlo, clo = max(7, rlo), max(7, clo)
        if rlo > rhi or clo > chi:
            raise ValueError("geometry cannot hold interior motif")
    rows = int(rng.integers(rlo, rhi + 1))
    columns = int(rng.integers(clo, chi + 1))
    return rows, columns, cell_h, cell_w


def _palette(rng, count, parameters, previous):
    if parameters["palette_mode"] == "stable":
        return parameters["background"], parameters["divider"], parameters["accents"]
    background = int(rng.integers(10))
    if previous is not None and rng.random() < 0.7:
        while background == previous[0]:
            background = int(rng.integers(10))
    available = [x for x in range(10) if x != background]
    divider = int(rng.choice(available))
    available.remove(divider)
    accents = tuple(int(x) for x in rng.choice(available, size=count, replace=False))
    return background, divider, accents


def _render_lattice(coarse, background, divider, cell_h, cell_w):
    rows, columns = coarse.shape
    height, width = rows * (cell_h + 1) - 1, columns * (cell_w + 1) - 1
    result = np.full((height, width), background, dtype=np.int8)
    result[cell_h::cell_h + 1, :] = divider
    result[:, cell_w::cell_w + 1] = divider
    for row in range(rows):
        for column in range(columns):
            color = int(coarse[row, column])
            if color != background:
                y, x = row * (cell_h + 1), column * (cell_w + 1)
                result[y:y + cell_h, x:x + cell_w] = color
    return result


def _generate_pair(rng, regime, index, required, parameters, previous):
    interior_only = rng.random() < 0.30
    if required == "disjoint" and parameters["size_mode"] == "smaller" and regime == "target_test":
        interior_only = False
    for _ in range(90):
        try:
            rows, columns, cell_h, cell_w = _geometry(
                rng, regime, index, parameters["size_mode"], interior_only
            )
        except ValueError:
            continue
        count = 2 if parameters["palette_mode"] == "stable" else int(rng.integers(1, 5))
        if required in _RICH_KINDS:
            count = min(count, 2)
        if interior_only:
            count = min(count, 2)
        try:
            hint = parameters["train_orientation"] if regime == "source_train" else parameters["test_orientation"]
            groups = _layout(rng, rows, columns, required, count, interior_only, hint)
        except ValueError:
            continue
        background, divider, accents = _palette(rng, count, parameters, previous)
        coarse = np.full((rows, columns), background, dtype=np.int8)
        for color, (_, points) in zip(accents, groups):
            for point in points:
                coarse[point] = color
        if np.count_nonzero(coarse == background) <= rows * columns // 2:
            continue
        completed = _complete_coarse_grid(coarse, background)
        inp = _render_lattice(coarse, background, divider, cell_h, cell_w)
        out = _render_lattice(completed, background, divider, cell_h, cell_w)
        return {"input": inp, "output": out}, (background, divider, accents)
    raise ValueError("could not construct a valid pair")


def _grid_key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _source_digest(grid):
    array = np.asarray(grid, dtype=np.int8)
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    palette_mode = "stable" if rng.random() < 0.22 else "mixed"
    parameters = {
        "palette_mode": palette_mode,
        "size_mode": str(rng.choice(("larger", "equal", "smaller", "mixed"))),
    }
    orientation_mode = str(rng.choice(("free", "free", "consistent_h", "consistent_v",
                                       "shift_h_to_v", "shift_v_to_h")))
    train_orientation = (
        "horizontal" if orientation_mode in ("consistent_h", "shift_h_to_v") else "vertical"
    )
    parameters["train_orientation"] = None if orientation_mode == "free" else train_orientation
    parameters["test_orientation"] = (
        None if orientation_mode == "free" else
        ("vertical" if orientation_mode == "shift_h_to_v" else
         "horizontal" if orientation_mode == "shift_v_to_h" else train_orientation)
    )
    if palette_mode == "stable":
        colors = rng.choice(10, size=4, replace=False)
        parameters.update(background=int(colors[0]), divider=int(colors[1]),
                          accents=tuple(int(x) for x in colors[2:]))
    rich_kind = str(rng.choice(_RICH_KINDS))
    puzzle = {"train": [], "test": []}
    seen, palettes = set(), []
    rich_in_training = False
    for section, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if section == "train" else "target_test"
        for index in range(count):
            if section == "train" and index == 0:
                hint = parameters["train_orientation"]
                required = "off_axis" if hint is None else (
                    "off_axis_h" if hint == "horizontal" else "off_axis_v"
                )
            elif section == "train" and index == 1:
                required = "ell"
            elif section == "train" and index == 2:
                required = rich_kind
                rich_in_training = True
            elif section == "test" and index == 0 and orientation_mode.startswith("shift"):
                required = "line_h" if parameters["test_orientation"] == "horizontal" else "line_v"
            elif section == "test" and rich_in_training and rng.random() < 0.45:
                required = rich_kind
            else:
                required = str(rng.choice(
                    ("line", "line", "collinear", "ell", "off_axis", "inert", "complete")))
                hint = parameters["train_orientation"] if section == "train" else parameters["test_orientation"]
                if required in ("line", "collinear", "off_axis") and hint is not None:
                    required += "_h" if hint == "horizontal" else "_v"
            for _ in range(60):
                child_seed = int(rng.integers(0, 2**63, dtype=np.int64))
                pair, palette = _generate_pair(np.random.default_rng(child_seed), regime, index,
                                               required, parameters, palettes[-1] if palettes else None)
                key = _grid_key(pair["input"])
                if key not in seen and _source_digest(pair["input"]) not in _SOURCE_INPUT_DIGESTS:
                    break
            else:
                raise ValueError("could not generate unique non-source input")
            seen.add(key)
            palettes.append(palette)
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _decode_lattice(grid, expected_background=None):
    height, width = grid.shape
    uniform_rows = [r for r in range(height) if np.all(grid[r, :] == grid[r, 0])]
    uniform_columns = [c for c in range(width) if np.all(grid[:, c] == grid[0, c])]
    assert uniform_rows and uniform_columns
    divider = int(grid[uniform_rows[0], 0])
    assert all(int(grid[r, 0]) == divider for r in uniform_rows)
    assert all(int(grid[0, c]) == divider for c in uniform_columns)
    cell_h, cell_w = uniform_rows[0], uniform_columns[0]
    assert cell_h >= 1 and cell_w >= 1
    assert uniform_rows == list(range(cell_h, height, cell_h + 1))
    assert uniform_columns == list(range(cell_w, width, cell_w + 1))
    assert (height + 1) % (cell_h + 1) == 0
    assert (width + 1) % (cell_w + 1) == 0
    rows, columns = (height + 1) // (cell_h + 1), (width + 1) // (cell_w + 1)
    assert rows >= 2 and columns >= 2
    coarse = np.empty((rows, columns), dtype=np.int8)
    for row in range(rows):
        for column in range(columns):
            y, x = row * (cell_h + 1), column * (cell_w + 1)
            block = grid[y:y + cell_h, x:x + cell_w]
            assert block.shape == (cell_h, cell_w)
            assert np.all(block == block[0, 0])
            assert int(block[0, 0]) != divider
            coarse[row, column] = block[0, 0]
    values, counts = np.unique(coarse, return_counts=True)
    assert len(values) >= 2
    if expected_background is None:
        assert np.count_nonzero(counts == max(counts)) == 1
        background = int(values[int(np.argmax(counts))])
    else:
        assert expected_background in values
        background = expected_background
    return coarse, background, divider, cell_h, cell_w


def _interval_directions(coarse, background):
    directions = set()
    for value in np.unique(coarse):
        color = int(value)
        if color == background:
            continue
        points = _positions(coarse, color)
        for row in {r for r, _ in points}:
            columns = [c for r, c in points if r == row]
            if len(columns) > 1 and max(columns) - min(columns) > len(columns) - 1:
                directions.add("horizontal")
        for column in {c for _, c in points}:
            rows = [r for r, c in points if c == column]
            if len(rows) > 1 and max(rows) - min(rows) > len(rows) - 1:
                directions.add("vertical")
    return directions


def _iterative_differs(coarse, background):
    once = _complete_coarse_grid(coarse, background)
    try:
        twice = _complete_coarse_grid(once, background)
    except AssertionError:
        return True
    return not np.array_equal(once, twice)


def _rich_tags(coarse, background):
    tags = set()
    for value in np.unique(coarse):
        color = int(value)
        if color == background:
            continue
        points = _positions(coarse, color)
        row_counts = {r: sum(p[0] == r for p in points) for r, _ in points}
        col_counts = {c: sum(p[1] == c for p in points) for _, c in points}
        if len(points) == 4 and sorted(row_counts.values()) == [2, 2]:
            by_row = [set(c for r, c in points if r == row) for row in row_counts]
            if by_row[0] == by_row[1]:
                tags.add("rectangle")
            elif by_row[0].isdisjoint(by_row[1]):
                tags.add("disjoint")
        if len(points) == 4 and 3 in row_counts.values() and 2 in col_counts.values():
            tags.add("tee")
        if len(points) == 5 and 3 in row_counts.values() and 3 in col_counts.values():
            tags.add("cross")
    return tags


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    directions, train_rich, test_rich = set(), set(), set()
    contrast = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            key = _grid_key(inp)
            pair_key = key, _grid_key(out)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _source_digest(inp) not in _SOURCE_INPUT_DIGESTS
            coarse, background, divider, cell_h, cell_w = _decode_lattice(inp)
            out_coarse, _, out_divider, out_h, out_w = _decode_lattice(out, background)
            assert (divider, cell_h, cell_w) == (out_divider, out_h, out_w)
            assert np.array_equal(out_coarse, _complete_coarse_grid(coarse, background))
            assert np.array_equal(out, _render_lattice(out_coarse, background, divider, cell_h, cell_w))
            assert np.count_nonzero(coarse == background) > coarse.size // 2
            tags = _rich_tags(coarse, background)
            if section == "train":
                train_rich.update(tags)
                directions.update(_interval_directions(coarse, background))
                contrast |= _iterative_differs(coarse, background)
            else:
                test_rich.update(tags)
    assert directions == {"horizontal", "vertical"}
    assert contrast
    assert test_rich <= train_rich


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[role], y[role])
        for x, y in zip(a[split], b[split]) for role in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    signatures, shapes, cells, palettes, topology = set(), set(), set(), set(), set()
    accent_counts, anchor_counts, changed_counts, directions = set(), set(), set(), set()
    measurements = {name: 0 for name in (
        "zero_background", "unchanged", "one_cell", "interior", "edge",
        "elongated", "multi_inert", "collinear_three", "ell", "rich_non_22",
        "stable_palette_puzzles", "varied_palette_puzzles", "varied_cells_puzzles",
        "new_test_accent_puzzles", "larger_test", "smaller_test", "equal_test",
        "direction_shift_puzzles", "consistent_direction_puzzles",
        "mixed_direction_puzzles", "combined_variations",
        "two_ell_pairs", "only_straight_pairs",
    )}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_grid_key(p["input"]) for split in ("train", "test")
                             for p in puzzle[split]))
        local_palettes, local_cells, local_directions = set(), set(), set()
        training_directions, training_accents = set(), set()
        test_accents = set()
        train_shapes, test_shapes = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                coarse, bg, div, ch, cw = _decode_lattice(pair["input"])
                completed, _, _, _, _ = _decode_lattice(pair["output"], bg)
                shapes.add(coarse.shape)
                cells.add((ch, cw))
                accent_colors = {int(x) for x in np.unique(coarse) if x != bg}
                palette = (bg, div, tuple(sorted(accent_colors)))
                palettes.add(palette)
                local_palettes.add(palette)
                local_cells.add((ch, cw))
                tags = _rich_tags(coarse, bg)
                topology.update(tags)
                new_cells = int(np.count_nonzero(coarse != completed))
                changed_counts.add(new_cells)
                accent_counts.add(len(accent_colors))
                anchor_counts.add(int(np.count_nonzero(coarse != bg)))
                observed = _interval_directions(coarse, bg)
                directions.add(tuple(sorted(observed)))
                if len(observed) == 1:
                    local_directions.update(observed)
                    if split == "train":
                        training_directions.update(observed)
                if split == "train":
                    training_accents.update(accent_colors)
                    train_shapes.append(coarse.shape)
                else:
                    test_accents.update(accent_colors)
                    test_shapes.append(coarse.shape)
                measurements["zero_background"] += bg == 0
                measurements["unchanged"] += new_cells == 0
                measurements["one_cell"] += new_cells == 1
                measurements["elongated"] += max(coarse.shape) / min(coarse.shape) >= 1.5
                measurements["rich_non_22"] += bool(tags) and (ch, cw) != (2, 2)
                measurements["combined_variations"] += bool(tags) and (ch, cw) != (2, 2) and coarse.shape[0] != coarse.shape[1]
                on_edge = any(r in (0, coarse.shape[0] - 1) or c in (0, coarse.shape[1] - 1)
                              for r, c in np.argwhere(coarse != bg))
                measurements["edge" if on_edge else "interior"] += 1
                ell_count = 0
                for color in accent_colors:
                    points = _positions(coarse, color)
                    inert = sum(not any(
                        (r == other_r and c != other_c) or
                        (c == other_c and r != other_r)
                        for other_r, other_c in points
                    ) for r, c in points)
                    measurements["multi_inert"] += inert >= 2
                    measurements["collinear_three"] += any(
                        sum(other_r == r for other_r, _ in points) >= 3 or
                        sum(other_c == c for _, other_c in points) >= 3
                        for r, c in points
                    )
                    is_ell = len(points) == 3 and sorted(
                        sum(other_r == r for other_r, _ in points) for r, _ in points
                    ) == [1, 2, 2] and sorted(
                        sum(other_c == c for _, other_c in points) for _, c in points
                    ) == [1, 2, 2]
                    measurements["ell"] += is_ell
                    ell_count += is_ell
                measurements["two_ell_pairs"] += ell_count >= 2
                measurements["only_straight_pairs"] += (
                    bool(observed) and ell_count == 0 and not tags and
                    all(len(_positions(coarse, color)) <= 3 for color in accent_colors)
                )
        measurements["stable_palette_puzzles" if len(local_palettes) == 1
                     else "varied_palette_puzzles"] += 1
        measurements["varied_cells_puzzles"] += len(local_cells) > 1
        measurements["new_test_accent_puzzles"] += bool(test_accents - training_accents)
        measurements["larger_test"] += any(
            all(a > b for a, b in zip(test, train))
            for test in test_shapes for train in train_shapes
        )
        measurements["smaller_test"] += any(
            all(a < b for a, b in zip(test, train))
            for test in test_shapes for train in train_shapes
        )
        measurements["equal_test"] += bool(set(train_shapes) & set(test_shapes))
        measurements["direction_shift_puzzles"] += bool(local_directions - training_directions)
        measurements["consistent_direction_puzzles"] += len(local_directions) == 1
        measurements["mixed_direction_puzzles"] += len(local_directions) == 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42
    assert len(shapes) >= 20 and len(cells) >= 5 and len(palettes) >= 40
    assert accent_counts == {1, 2, 3, 4}
    assert len(anchor_counts) >= 8 and {0, 1, 2} <= changed_counts
    assert directions == {(), ("horizontal",), ("vertical",), ("horizontal", "vertical")}
    assert topology == set(_RICH_KINDS)
    for name, minimum in (
        ("zero_background", 10), ("unchanged", 10), ("one_cell", 10),
        ("interior", 40), ("edge", 100), ("elongated", 20),
        ("multi_inert", 20), ("collinear_three", 20), ("ell", 20),
        ("rich_non_22", 15), ("stable_palette_puzzles", 2),
        ("varied_palette_puzzles", 20), ("varied_cells_puzzles", 30),
        ("new_test_accent_puzzles", 10), ("larger_test", 10),
        ("smaller_test", 10), ("equal_test", 5),
        ("direction_shift_puzzles", 5), ("consistent_direction_puzzles", 5),
        ("mixed_direction_puzzles", 5), ("combined_variations", 10),
        ("two_ell_pairs", 5), ("only_straight_pairs", 30),
    ):
        assert measurements[name] >= minimum, (name, measurements[name])
    source_train_three = np.array([
        [int(cell) for cell in line] for line in (
            "00000000", "03000200", "00000000", "00020020",
            "00000000", "00000000", "03000300", "00000000",
        )
    ], dtype=np.int8)
    one_pass = _complete_coarse_grid(source_train_three, 0)
    assert one_pass[2, 5] == 0 and one_pass[3, 5] == 2
    rectangle = np.zeros((5, 5), dtype=np.int8)
    rectangle[1, 1] = rectangle[1, 3] = rectangle[3, 1] = rectangle[3, 3] = 2
    assert _complete_coarse_grid(rectangle, 0)[2, 2] == 0
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(100000 + 10 * train + test, train, test)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(200000 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    same = generate_puzzle(987654, 6, 2)
    assert _puzzles_equal(same, generate_puzzle(987654, 6, 2))
    assert not _puzzles_equal(same, generate_puzzle(987655, 6, 2))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(300000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
