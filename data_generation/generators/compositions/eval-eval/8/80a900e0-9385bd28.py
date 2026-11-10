"""Stereo-registered diagonal exposure. Only NumPy and the standard library."""

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _inside(cell, shape):
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _outline(center, radius):
    return [(center[0] + delta_row, center[1] + delta_col)
            for delta_row in range(-radius, radius + 1)
            for delta_col in range(-radius, radius + 1)
            if abs(delta_row) + abs(delta_col) == radius]


def _cap(center, radius, direction):
    delta_row, delta_col = direction
    return [(center[0] + delta_row * (offset + 1),
             center[1] + delta_col * (radius - offset + 1))
            for offset in range(radius + 1)]


def _corners(bounds, diagonal):
    top, left, bottom, right = bounds
    if diagonal == 0:
        return {(top, left), (top, left + 1), (top + 1, left),
                (bottom, right), (bottom, right - 1), (bottom - 1, right)}
    return {(top, right), (top, right - 1), (top + 1, right),
            (bottom, left), (bottom, left + 1), (bottom - 1, left)}


def _contains(bounds, cell):
    return bounds[0] <= cell[0] <= bounds[2] and bounds[1] <= cell[1] <= bounds[3]


def _base(shape, background):
    parity = np.indices(shape).sum(axis=0) % 2
    return np.asarray(background, dtype=np.int8)[parity]


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= extent <= 30 for extent in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("expected an integer 1..30 grid with colors 0..9")
    return array


def _parse(input_grid):
    grid = _grid(input_grid)
    parity = np.indices(grid.shape).sum(axis=0) % 2
    background = []
    for phase in (0, 1):
        values = grid[parity == phase]
        counts = np.bincount(values.astype(int), minlength=10)
        color = int(counts.argmax())
        assert counts[color] * 2 > len(values), "background must dominate each parity"
        background.append(color)
    assert background[0] != background[1]
    base = _base(grid.shape, background)
    foreground = grid != base
    height, width = grid.shape
    emitters = []
    for radius in (2, 3, 4):
        if min(grid.shape) < 2 * radius + 1:
            continue
        scan_height, scan_width = height - 2 * radius, width - 2 * radius
        offsets = _outline((0, 0), radius)
        first_row, first_col = offsets[0]
        colors = grid[radius + first_row:radius + first_row + scan_height,
                      radius + first_col:radius + first_col + scan_width]
        matches = ~np.isin(colors, background)
        for delta_row, delta_col in offsets[1:]:
            matches &= grid[radius + delta_row:radius + delta_row + scan_height,
                            radius + delta_col:radius + delta_col + scan_width] == colors
        for row_index, col_index in np.argwhere(matches):
            emitters.append(dict(center=(int(row_index) + radius, int(col_index) + radius),
                                 radius=radius, color=int(colors[row_index, col_index]), caps={}))
    assert 1 <= len(emitters) <= 2, "ambiguous or missing diamonds"
    assert len({emitter["color"] for emitter in emitters}) == 1
    glyphs = set()
    for emitter in emitters:
        cells = set(_outline(emitter["center"], emitter["radius"]))
        assert not glyphs & cells
        glyphs.update(cells)
    for emitter in emitters:
        for direction in DIRECTIONS:
            visible = [cell for cell in _cap(emitter["center"], emitter["radius"], direction)
                       if _inside(cell, grid.shape)]
            values = [int(grid[cell]) for cell in visible]
            if (len(values) >= 2 and len(set(values)) == 1
                    and values[0] not in background and values[0] != emitter["color"]):
                assert not glyphs.intersection(visible)
                glyphs.update(visible)
                emitter["caps"][direction] = values[0]
        assert emitter["caps"]
    role_colors = {emitter["color"] for emitter in emitters}
    role_colors.update(color for emitter in emitters for color in emitter["caps"].values())
    assert len({color for emitter in emitters for color in emitter["caps"].values()}) <= 2
    remaining = foreground.copy()
    for cell in glyphs:
        remaining[cell] = False
    candidates = []
    for column in range(width - 1):
        rows = np.flatnonzero(remaining[:, column])
        for run in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1):
            if not 2 <= len(run) <= 4:
                continue
            top, bottom = int(run[0]), int(run[-1])
            table = (top, column, bottom, column + 1)
            cues = list(map(int, grid[top:bottom + 1, column]))
            paints = list(map(int, grid[top:bottom + 1, column + 1]))
            if (len(set(cues)) != len(cues) or set(cues) & (set(background) | role_colors)
                    or set(paints) & set(background)):
                continue
            if any(foreground[row, col]
                   for row in range(max(0, top - 1), min(height, bottom + 2))
                   for col in range(max(0, column - 1), min(width, column + 3))
                   if not _contains(table, (row, col))):
                continue
            scene = remaining.copy()
            scene[top:bottom + 1, column:column + 2] = False
            if not set(map(int, grid[scene])) <= set(cues):
                continue
            sheets = []
            for cue, paint in zip(cues, paints):
                cells = set(map(tuple, np.argwhere(scene & (grid == cue))))
                if len(cells) != 6:
                    break
                bounds = (min(cell[0] for cell in cells), min(cell[1] for cell in cells),
                          max(cell[0] for cell in cells), max(cell[1] for cell in cells))
                if min(bounds[2] - bounds[0], bounds[3] - bounds[1]) < 2:
                    break
                diagonals = [diagonal for diagonal in (0, 1) if cells == _corners(bounds, diagonal)]
                if len(diagonals) != 1 or any(_contains(bounds, (row, col))
                        for row in range(top, bottom + 1) for col in (column, column + 1)):
                    break
                sheets.append(dict(bounds=tuple(map(int, bounds)), cue=cue, paint=paint,
                                   diagonal=diagonals[0], markers=cells))
            if len(sheets) == len(cues):
                candidates.append(dict(background=tuple(background), base=base,
                                       emitters=emitters, sheets=sheets, table=table,
                                       foreground=foreground))
    assert len(candidates) == 1, "table or rectangle cues are ambiguous"
    return candidates[0]


def _slices(parsed, shape):
    for emitter in parsed["emitters"]:
        for direction, color in emitter["caps"].items():
            cap = _cap(emitter["center"], emitter["radius"], direction)
            for step in range(1, 31):
                cells = [(row + step * direction[0], col + step * direction[1])
                         for row, col in cap]
                endpoints = (cells[0], cells[-1])
                if not any(_inside(cell, shape) for cell in cells):
                    break
                yield emitter, direction, step, color, cells, endpoints


def _draw(grid, parsed, mode="normal"):
    output = grid.copy()
    free = ~parsed["foreground"]
    masks = [np.zeros(grid.shape, dtype=bool) for sheet in parsed["sheets"]]
    rails = {}
    for emitter, direction, step, color, cells, endpoints in _slices(parsed, grid.shape):
        for cell in endpoints:
            if _inside(cell, grid.shape):
                assert cell not in rails or rails[cell] == color, "contrasting rail collision"
                rails[cell] = color
        if mode == "whole_cap":
            for cell in cells:
                if _inside(cell, grid.shape) and free[cell]:
                    output[cell] = color
        for sheet_index, sheet in enumerate(parsed["sheets"]):
            hits = [_contains(sheet["bounds"], cell) for cell in endpoints]
            eligible = any(hits) if mode == "any_endpoint" else all(hits)
            if eligible:
                for cell in cells:
                    if _inside(cell, grid.shape) and _contains(sheet["bounds"], cell):
                        masks[sheet_index][cell] = True
    for cell, color in rails.items():
        if free[cell]:
            output[cell] = color
    if mode == "rails_only":
        return output
    if mode == "whole_boxes":
        for sheet_index, sheet in enumerate(parsed["sheets"]):
            top, left, bottom, right = sheet["bounds"]
            masks[sheet_index][top:bottom + 1, left:right + 1] = True
    ranks = range(len(masks)) if mode == "reverse_priority" else reversed(range(len(masks)))
    for sheet_index in ranks:
        output[masks[sheet_index] & free] = parsed["sheets"][sheet_index]["paint"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    return _draw(grid, _parse(grid))


def _algebraic_output(grid, parsed):
    """A separate closed-form geometric computation, sharing only input parsing."""
    rows, columns = np.indices(grid.shape)
    free = ~parsed["foreground"]
    output = grid.copy()
    masks = [np.zeros(grid.shape, dtype=bool) for sheet in parsed["sheets"]]
    for emitter in parsed["emitters"]:
        center_row, center_col = emitter["center"]
        radius = emitter["radius"]
        for (delta_row, delta_col), color in emitter["caps"].items():
            outward_row = (rows - center_row) * delta_row
            outward_col = (columns - center_col) * delta_col
            rails = (np.abs(outward_row - outward_col) == radius) & (np.minimum(outward_row, outward_col) >= 2)
            output[rails & free] = color
            twice_step = outward_row + outward_col - radius - 2
            step = twice_step // 2
            offset = outward_row - step - 1
            rung = (twice_step % 2 == 0) & (step >= 1) & (offset >= 0) & (offset <= radius)
            first_row = center_row + delta_row * (step + 1)
            first_col = center_col + delta_col * (step + radius + 1)
            last_row = center_row + delta_row * (step + radius + 1)
            last_col = center_col + delta_col * (step + 1)
            for sheet_index, sheet in enumerate(parsed["sheets"]):
                top, left, bottom, right = sheet["bounds"]
                eligible = (first_row >= top) & (first_row <= bottom) & (last_row >= top) & (last_row <= bottom)
                eligible &= (first_col >= left) & (first_col <= right) & (last_col >= left) & (last_col <= right)
                masks[sheet_index] |= rung & eligible
    for sheet_index in reversed(range(len(masks))):
        output[masks[sheet_index] & free] = parsed["sheets"][sheet_index]["paint"]
    return output


def _witnesses(grid, output, parsed):
    found = set()
    either_output = _draw(grid, parsed, "any_endpoint")
    box_output = _draw(grid, parsed, "whole_boxes")
    for alternative in ("rails_only", "whole_boxes", "whole_cap", "any_endpoint", "reverse_priority"):
        if not np.array_equal(output, _draw(grid, parsed, alternative)):
            found.add(alternative)
    free = ~parsed["foreground"]
    if any(free[row, col] and output[row, col] == grid[row, col]
           and box_output[row, col] != output[row, col]
           for row, col in np.argwhere(free)):
        found.add("unfilled_box")
    for emitter, direction, step, color, cells, endpoints in _slices(parsed, grid.shape):
        if (step <= 2 and all(_inside(cell, grid.shape) and free[cell] and output[cell] == color
                             for cell in endpoints)):
            found.add((direction, emitter["radius"], step))
        for cell in cells[1:-1]:
            if (_inside(cell, grid.shape) and free[cell] and either_output[cell] != output[cell]
                    and any(_contains(sheet["bounds"], cell)
                            and sum(_contains(sheet["bounds"], endpoint) for endpoint in endpoints) == 1
                            for sheet in parsed["sheets"])):
                found.add("single_interior")
            if _inside(cell, grid.shape) and free[cell] and output[cell] != grid[cell]:
                if any(all(_contains(sheet["bounds"], endpoint) for endpoint in endpoints)
                       for sheet in parsed["sheets"]):
                    found.add(("bridge", emitter["radius"]))
        if any(_inside(cell, grid.shape) and free[cell] and output[cell] != color for cell in endpoints):
            if any(all(_contains(sheet["bounds"], endpoint) for endpoint in endpoints)
                   for sheet in parsed["sheets"]):
                found.add("mapped_rail")
    return found


def _emitters(rng, shape, maximum_radius, count, all_sides):
    emitters, occupied, rails = [], set(), {}
    for emitter_index in range(count):
        for attempt in range(100):
            radius = maximum_radius if all_sides else int(rng.integers(2, maximum_radius + 1))
            margin = radius + 4 if all_sides else radius
            center = tuple(int(rng.integers(margin, extent - margin)) for extent in shape)
            mask = 15 if all_sides else int(rng.integers(1, 16))
            caps = {direction: 0 for index, direction in enumerate(DIRECTIONS) if mask & (1 << index)}
            emitter = dict(center=center, radius=radius, caps=caps)
            cells = set(_outline(center, radius))
            for direction in caps:
                visible = [cell for cell in _cap(center, radius, direction) if _inside(cell, shape)]
                if len(visible) < 2:
                    break
                cells.update(visible)
            else:
                if cells & occupied:
                    continue
                temporary = dict(emitters=[emitter])
                paths = {cell for _, _, _, _, _, endpoints in _slices(temporary, shape)
                         for cell in endpoints if _inside(cell, shape)}
                if paths & occupied or cells & set(rails):
                    continue
                occupied.update(cells)
                rails.update(dict.fromkeys(paths, 0))
                emitters.append(emitter)
                break
        else:
            return None
    return emitters


def _sample_sheet(rng, shape, slices, anchor=None):
    if anchor is None:
        anchor = slices[int(rng.integers(len(slices)))]
    cells, endpoints = anchor[-2:]
    top = min(cell[0] for cell in endpoints)
    left = min(cell[1] for cell in endpoints)
    bottom = max(cell[0] for cell in endpoints)
    right = max(cell[1] for cell in endpoints)
    bounds = (max(0, top - int(rng.integers(0, 5))), max(0, left - int(rng.integers(0, 5))),
              min(shape[0] - 1, bottom + int(rng.integers(0, 5))),
              min(shape[1] - 1, right + int(rng.integers(0, 5))))
    if rng.random() < .32:
        shift_row, shift_col = map(int, rng.integers(-3, 4, size=2))
        bounds = (max(0, bounds[0] + shift_row), max(0, bounds[1] + shift_col),
                  min(shape[0] - 1, bounds[2] + shift_row), min(shape[1] - 1, bounds[3] + shift_col))
    if min(bounds[2] - bounds[0], bounds[3] - bounds[1]) < 2:
        return None
    return dict(bounds=bounds, diagonal=int(rng.integers(2)))


def _candidate(rng, maximum_radius, emitter_count, sheet_count, all_sides, required):
    lower = max(17, 2 * maximum_radius + 9, 23 if emitter_count == 2 else 17)
    shape = tuple(int(rng.integers(lower, 31)) for axis in range(2))
    emitters = _emitters(rng, shape, maximum_radius, emitter_count, all_sides)
    if emitters is None:
        return None
    palette = list(map(int, rng.permutation(10)))
    background, structural = tuple(palette[:2]), palette[2]
    cap_bank = palette[3:5] if rng.random() < .3 else palette[3:4]
    cues = palette[5:5 + sheet_count]
    occupied = set()
    for emitter in emitters:
        emitter["color"] = structural
        emitter["caps"] = {direction: int(rng.choice(cap_bank)) for direction in emitter["caps"]}
        occupied.update(_outline(emitter["center"], emitter["radius"]))
        occupied.update(cell for direction in emitter["caps"]
                        for cell in _cap(emitter["center"], emitter["radius"], direction) if _inside(cell, shape))
    slices = [item for item in _slices(dict(emitters=emitters), shape)
              if all(_inside(cell, shape) for cell in item[-1]) and item[2] >= 3]
    if not slices:
        return None
    shared_anchor = slices[int(rng.integers(len(slices)))]
    sheets = []
    for sheet_index in range(sheet_count):
        for attempt in range(45):
            sheet = _sample_sheet(rng, shape, slices, shared_anchor if sheet_index < 2 else None)
            if sheet is None:
                continue
            markers = _corners(sheet["bounds"], sheet["diagonal"])
            if markers & occupied or any(sheet["bounds"] == previous["bounds"] for previous in sheets):
                continue
            sheet.update(cue=cues[sheet_index], markers=markers)
            occupied.update(markers)
            sheets.append(sheet)
            break
        else:
            return None
    rng.shuffle(sheets)
    paint_bank = [color for color in palette if color not in background]
    for sheet in sheets:
        sheet["paint"] = int(rng.choice(paint_bank))
    if sheets[0]["paint"] == sheets[1]["paint"]:
        sheets[1]["paint"] = int(rng.choice([color for color in paint_bank if color != sheets[0]["paint"]]))
    grid = _base(shape, background)
    for emitter in emitters:
        for cell in _outline(emitter["center"], emitter["radius"]):
            grid[cell] = structural
        for direction, color in emitter["caps"].items():
            for cell in _cap(emitter["center"], emitter["radius"], direction):
                if _inside(cell, shape):
                    grid[cell] = color
    for sheet in sheets:
        for cell in sheet["markers"]:
            grid[cell] = sheet["cue"]
    blocked = np.zeros(shape, dtype=bool)
    for sheet in sheets:
        top, left, bottom, right = sheet["bounds"]
        blocked[top:bottom + 1, left:right + 1] = True
    for cell in occupied:
        blocked[cell] = True
    for _, _, _, _, _, endpoints in _slices(dict(emitters=emitters), shape):
        for cell in endpoints:
            if _inside(cell, shape):
                blocked[cell] = True
    locations = [(row, col) for row in range(shape[0] - sheet_count + 1)
                 for col in range(shape[1] - 1)
                 if not blocked[max(0, row - 1):min(shape[0], row + sheet_count + 1),
                                max(0, col - 1):min(shape[1], col + 3)].any()]
    if not locations:
        return None
    table_row, table_col = locations[int(rng.integers(len(locations)))]
    for index, sheet in enumerate(sheets):
        grid[table_row + index, table_col:table_col + 2] = (sheet["cue"], sheet["paint"])
    try:
        parsed = _parse(grid)
        if len(parsed["emitters"]) != emitter_count:
            return None
        output = _draw(grid, parsed)
        if not required <= _witnesses(grid, output, parsed):
            return None
        if all_sides:
            needed = {(direction, maximum_radius, step) for direction in DIRECTIONS for step in (1, 2)}
            if not needed <= _witnesses(grid, output, parsed):
                return None
            if {sheet["diagonal"] for sheet in parsed["sheets"]} != {0, 1}:
                return None
        if not np.array_equal(output, _algebraic_output(grid, parsed)):
            return None
    except AssertionError:
        return None
    return dict(input=grid, output=output)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    maximum_radius = int(rng.integers(2, 5))
    shift = str(rng.choice(["sheets", "emitters"]))
    slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    core = {"rails_only", "whole_boxes", "whole_cap", "any_endpoint", "reverse_priority", "mapped_rail",
            "single_interior", "unfilled_box"}
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            all_sides = split == "train" and index == slots[0]
            required = core | {("bridge", maximum_radius)} if all_sides else {"rails_only", "whole_boxes"}
            if split == "train" and index == slots[1]:
                required |= {"any_endpoint", "reverse_priority"}
            emitter_count = 2 if split == "test" and shift == "emitters" else 1
            if split == "train" and shift == "sheets" and not all_sides and rng.random() < .3:
                emitter_count = 2
            sheet_count = 4 if split == "test" and shift == "sheets" else int(rng.integers(2, 4))
            for attempt in range(700):
                pair = _candidate(rng, maximum_radius, emitter_count, sheet_count, all_sides, required)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded stereo-exposure construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence, diagonals, test_radii = set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            assert grid.shape == output.shape and not np.array_equal(grid, output)
            assert _key(grid) not in seen
            seen.add(_key(grid))
            parsed = _parse(grid)
            assert np.array_equal(output, _draw(grid, parsed)), "incorrect walking result"
            assert np.array_equal(output, _algebraic_output(grid, parsed)), "incorrect algebraic result"
            assert not np.array_equal(output, _draw(grid, parsed, "rails_only")), "B has no visible contribution"
            assert not np.array_equal(output, _draw(grid, parsed, "whole_boxes")), "A has no visible contribution"
            foreground = parsed["foreground"]
            assert np.array_equal(grid[foreground], output[foreground]), "changed annotation"
            if split == "train":
                evidence.update(_witnesses(grid, output, parsed))
                diagonals.update(sheet["diagonal"] for sheet in parsed["sheets"])
            else:
                test_radii.update(emitter["radius"] for emitter in parsed["emitters"])
    assert {"rails_only", "whole_boxes", "whole_cap", "any_endpoint", "reverse_priority", "mapped_rail",
            "single_interior", "unfilled_box"} <= evidence
    assert diagonals == {0, 1}, "both corner diagonals must be taught"
    for direction in DIRECTIONS:
        for step in (1, 2):
            radii = [item[1] for item in evidence if isinstance(item, tuple) and len(item) == 3
                     and item[0] == direction and item[2] == step]
            assert radii and max(radii) >= max(test_radii), "untaught direction or radius"
    bridge_radii = [item[1] for item in evidence if isinstance(item, tuple) and item[0] == "bridge"]
    assert bridge_radii and max(bridge_radii) >= max(test_radii)


def _regression_tests():
    grid = _base((18, 18), (0, 1))
    for cell in _outline((5, 5), 2):
        grid[cell] = 2
    for cell in _cap((5, 5), 2, (1, 1)):
        grid[cell] = 3
    for cue, bounds, diagonal in ((6, (6, 9, 9, 13), 0), (7, (7, 7, 12, 12), 1)):
        for cell in _corners(bounds, diagonal):
            assert grid[cell] == _base(grid.shape, (0, 1))[cell]
            grid[cell] = cue
    grid[1, 1:3] = (6, 8)
    grid[2, 1:3] = (7, 9)
    parsed = _parse(grid)
    output = solve(grid)
    assert output[9, 9] == 9 and _draw(grid, parsed, "any_endpoint")[9, 9] == 8
    assert output[12, 12] == grid[12, 12] == 0
    assert output[11, 13] == 3 and output[8, 9] == grid[8, 9] == 1
    assert output[7, 9] == 6
    assert np.array_equal(output, _algebraic_output(grid, parsed))
    clipped = _base((18, 18), (0, 1))
    for cell in _outline((2, 5), 2):
        clipped[cell] = 2
    for cell in _cap((2, 5), 2, (-1, -1)):
        if _inside(cell, clipped.shape):
            clipped[cell] = 3
    for cue, bounds, diagonal in ((6, (8, 8, 11, 11), 0), (7, (10, 12, 14, 16), 1)):
        for cell in _corners(bounds, diagonal):
            clipped[cell] = cue
    clipped[16, 1:3] = (6, 8)
    clipped[17, 1:3] = (7, 9)
    expected = clipped.copy()
    expected[0, 1] = 3
    assert np.array_equal(solve(clipped), expected)
    assert np.array_equal(expected, _algebraic_output(clipped, _parse(clipped)))
    boundary = _base((18, 18), (0, 1))
    for cell in _outline((3, 3), 2):
        boundary[cell] = 2
    for direction, color in (((-1, -1), 3), ((1, 1), 4)):
        for cell in _cap((3, 3), 2, direction):
            boundary[cell] = color
    for cue, bounds, diagonal in ((6, (8, 8, 11, 11), 0), (7, (10, 12, 14, 16), 1)):
        for cell in _corners(bounds, diagonal):
            boundary[cell] = cue
    boundary[16, 1:3] = (6, 8)
    boundary[17, 1:3] = (7, 9)
    boundary_output = solve(boundary)
    assert boundary_output[0, 0] == boundary[0, 0] == 0
    assert _draw(boundary, _parse(boundary), "whole_cap")[0, 0] == 3
    assert np.array_equal(boundary_output, _algebraic_output(boundary, _parse(boundary)))


def self_test():
    _regression_tests()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, layouts, radii, counts, subsets, table_positions, paint_orders = set(), set(), set(), set(), set(), set(), set()
    clipped, geometry_overlaps = 0, set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                parsed = _parse(pair["input"])
                shapes.add(pair["input"].shape)
                table_positions.add(parsed["table"][:2])
                counts.add((len(parsed["emitters"]), len(parsed["sheets"])))
                layouts.add((pair["input"].shape,
                             tuple((emitter["center"], emitter["radius"], tuple(emitter["caps"])) for emitter in parsed["emitters"]),
                             tuple((sheet["bounds"], sheet["diagonal"]) for sheet in parsed["sheets"])))
                paint_orders.add(tuple(sheet["paint"] for sheet in parsed["sheets"]))
                for emitter in parsed["emitters"]:
                    radii.add(emitter["radius"])
                    subsets.add(tuple(emitter["caps"]))
                    clipped += sum(any(not _inside(cell, pair["input"].shape)
                                       for cell in _cap(emitter["center"], emitter["radius"], direction))
                                   for direction in emitter["caps"])
                sheets = parsed["sheets"]
                edges = sum(max(first["bounds"][0], second["bounds"][0]) <= min(first["bounds"][2], second["bounds"][2])
                            and max(first["bounds"][1], second["bounds"][1]) <= min(first["bounds"][3], second["bounds"][3])
                            for index, first in enumerate(sheets) for second in sheets[index + 1:])
                geometry_overlaps.add(edges)
    assert len(shapes) >= 45 and len(layouts) >= 150 and radii == {2, 3, 4}
    assert {count[0] for count in counts} == {1, 2} and {count[1] for count in counts} == {2, 3, 4}
    assert len(subsets) >= 10 and len(table_positions) >= 50 and len(paint_orders) >= 80
    assert len(geometry_overlaps) >= 4 and clipped > 0
    repeat = generate_puzzle(0)
    assert all(_key(first[role]) == _key(second[role])
               for split in ("train", "test") for first, second in zip(puzzles[0][split], repeat[split])
               for role in ("input", "output"))
    permutation = np.random.default_rng(123).permutation(10)
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    witness_pair = next(pair for pair in puzzles[0]["train"]
                        if "reverse_priority" in _witnesses(pair["input"], pair["output"], _parse(pair["input"])))
    reordered = witness_pair["input"].copy()
    parsed = _parse(reordered)
    top, left, bottom, right = parsed["table"]
    reordered[top:bottom + 1, left:right + 1] = reordered[top:bottom + 1, left:right + 1][::-1].copy()
    reordered_output = solve(reordered)
    assert np.any(reordered_output[~parsed["foreground"]] != witness_pair["output"][~parsed["foreground"]])
    for num_train in range(2, 11):
        for num_test in range(1, min(4, num_train - 1) + 1):
            puzzle = generate_puzzle(123 + num_train * 5 + num_test, num_train, num_test)
            assert len(puzzle["train"]) == num_train and len(puzzle["test"]) == num_test
    for num_train, num_test in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                                (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, num_train, num_test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {split: [{role: grid.copy() for role, grid in pair.items()} for pair in puzzles[0][split]]
               for split in ("train", "test")}
    changed = tuple(np.argwhere(damaged["test"][0]["input"] != damaged["test"][0]["output"])[0])
    damaged["test"][0]["output"][changed] = damaged["test"][0]["input"][changed]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a damaged exposure")
    return dict(seeds=42, shapes=len(shapes), geometry_layouts=len(layouts), radii=sorted(radii),
                emitter_sheet_counts=sorted(counts), cap_subsets=len(subsets), table_positions=len(table_positions),
                overlap_counts=sorted(geometry_overlaps), clipped_caps=clipped)
