import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "111111111111",
            "111111111111",
            "114111444111",
            "114111444111",
            "114444444111",
            "114444444111",
            "112222222111",
            "112222112111",
            "111111111111",
            "111111111111",
            "111111111111",
            "111111111111",
        ),
        (
            "111111111111",
            "111111111111",
            "141114441111",
            "141114441111",
            "144444441111",
            "144444441111",
            "111222222211",
            "111222211211",
            "111111111111",
            "111111111111",
            "111111111111",
            "111111111111",
        ),
    ),
    (
        (
            "888888888888",
            "886666888888",
            "886666888888",
            "886666666888",
            "886666666888",
            "883333333888",
            "883333333888",
            "883333333888",
            "883333333888",
            "883338833888",
            "888888888888",
            "888888888888",
            "888888888888",
            "888888888888",
            "888888888888",
            "888888888888",
        ),
        (
            "888888888888",
            "866668888888",
            "866668888888",
            "866666668888",
            "866666668888",
            "888333333388",
            "888333333388",
            "888333333388",
            "888333333388",
            "888333883388",
            "888888888888",
            "888888888888",
            "888888888888",
            "888888888888",
            "888888888888",
            "888888888888",
        ),
    ),
    (
        (
            "3333333333333",
            "3333333333333",
            "3333333222333",
            "3333333777333",
            "3333333333333",
            "3883333333333",
            "3883333333333",
            "3888833333333",
            "3888833333333",
            "3888833333333",
            "3666633333333",
            "3666633333333",
        ),
        (
            "3333333333333",
            "3333333333333",
            "3333332223333",
            "3333333377733",
            "3333333333333",
            "8833333333333",
            "8833333333333",
            "8888333333333",
            "8888333333333",
            "8888333333333",
            "3366663333333",
            "3366663333333",
        ),
    ),
    (
        (
            "8888888888888",
            "8818818888888",
            "8811118888888",
            "8811118888888",
            "8822228888888",
            "8888888888888",
            "8888888888888",
            "8888888888888",
            "8888888888888",
            "8888882228888",
            "8888882228888",
            "8888884448888",
            "8888884848888",
            "8888888888888",
        ),
        (
            "8888888888888",
            "8188188888888",
            "8111188888888",
            "8111188888888",
            "8882222888888",
            "8888888888888",
            "8888888888888",
            "8888888888888",
            "8888888888888",
            "8888822288888",
            "8888822288888",
            "8888888444888",
            "8888888484888",
            "8888888888888",
        ),
    ),
)



def _validate_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _validate_optional_count(num_train, "num_train", 2, 10)
    num_test = _validate_optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _connected_components(mask):
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    height, width = mask.shape
    for start_row, start_col in np.argwhere(mask):
        start = (int(start_row), int(start_col))
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        cells = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for nr, nc in ((row - 1, col), (row + 1, col),
                           (row, col - 1), (row, col + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        components.append(cells)
    return components


def _cells_are_connected(cells):
    cells = set(cells)
    stack = [next(iter(cells))]
    visited = {stack[0]}
    while stack:
        row, col = stack.pop()
        for neighbor in ((row - 1, col), (row + 1, col),
                         (row, col - 1), (row, col + 1)):
            if neighbor in cells and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == len(cells)


def _section_mask(rng, height, width, interface_bottom, richer=False):
    """Full interface plus optional tapered edges, open notches and holes."""
    if height == 1:
        return np.ones((1, width), dtype=bool)
    outer = 0 if interface_bottom else height - 1
    direction = 1 if interface_bottom else -1
    modes = ("rectangle", "taper", "notch", "hole", "combined")
    weights = (0.21, 0.24, 0.27, 0.10, 0.18) if richer else (0.27, 0.28, 0.25, 0.08, 0.12)
    for _ in range(24):
        mask = np.ones((height, width), dtype=bool)
        mode = str(rng.choice(modes, p=weights))
        if mode in ("taper", "combined"):
            depth = int(rng.integers(1, height))
            for offset in range(depth):
                row = outer + direction * offset
                left_cut = int(rng.integers(0, min(3, width - 1)))
                right_cut = int(rng.integers(0, min(3, width - left_cut)))
                if left_cut:
                    mask[row, :left_cut] = False
                if right_cut:
                    mask[row, width - right_cut:] = False
        if mode in ("notch", "combined") and width >= 3:
            for _ in range(int(rng.integers(1, 3))):
                gap_width = int(rng.integers(1, min(width - 2, 4) + 1))
                gap_col = int(rng.integers(1, width - gap_width))
                depth = int(rng.integers(1, min(height - 1, 3) + 1))
                for offset in range(depth):
                    mask[outer + direction * offset, gap_col:gap_col + gap_width] = False
        if mode in ("hole", "combined") and height >= 3 and width >= 3:
            gap_height = int(rng.integers(1, min(height - 2, 2) + 1))
            row = int(rng.integers(1, height - gap_height))
            gap_width = int(rng.integers(1, min(width - 2, 3) + 1))
            col = int(rng.integers(1, width - gap_width))
            mask[row:row + gap_height, col:col + gap_width] = False
        interface_row = height - 1 if interface_bottom else 0
        mask[interface_row, :] = True
        if all(bool(np.any(mask[row])) for row in range(height)) and len(_connected_components(mask)) == 1:
            return mask
    return np.ones((height, width), dtype=bool)


def _make_spec(rng, colors, richer=False):
    width = int(rng.integers(3, 10))
    upper_height = int(rng.integers(1, 7))
    lower_height = int(rng.integers(1, 7))
    return {
        "upper": _section_mask(rng, upper_height, width, True, richer),
        "lower": _section_mask(rng, lower_height, width, False, richer),
        "upper_color": int(colors[0]),
        "lower_color": int(colors[1]),
    }


def _spec_cells(spec, top, left, shifted):
    cells = []
    for row, col in np.argwhere(spec["upper"]):
        cells.append((top + int(row), left + int(col) - int(shifted), spec["upper_color"]))
    lower_top = top + spec["upper"].shape[0]
    for row, col in np.argwhere(spec["lower"]):
        cells.append((lower_top + int(row), left + int(col) + int(shifted), spec["lower_color"]))
    return cells


def _separated(first, second):
    ft, fb, fl, fr = first
    st, sb, sl, sr = second
    return fb + 1 < st or sb + 1 < ft or fr + 1 < sl or sr + 1 < fl


def _place_specs(rng, specs, height, width):
    placed = []
    occupied_output = set()
    order = sorted(range(len(specs)), key=lambda i: specs[i]["upper"].size + specs[i]["lower"].size, reverse=True)
    for index in order:
        spec = specs[index]
        oh = spec["upper"].shape[0] + spec["lower"].shape[0]
        ow = spec["upper"].shape[1]
        if oh > height or ow + 2 > width:
            return None
        positions = [(top, left) for top in range(height - oh + 1)
                     for left in range(1, width - ow)]
        rng.shuffle(positions)
        found = False
        for top, left in positions:
            rect = (top, top + oh - 1, left, left + ow - 1)
            if not all(_separated(rect, previous[3]) for previous in placed):
                continue
            output_cells = {(row, col) for row, col, _ in _spec_cells(spec, top, left, True)}
            if any((row + dr, col + dc) in occupied_output
                   for row, col in output_cells
                   for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))):
                continue
            placed.append((index, top, left, rect))
            occupied_output.update(output_cells)
            found = True
            break
        if not found:
            return None
    result = [None] * len(specs)
    for index, top, left, _ in placed:
        result[index] = (top, left)
    return result


def _sample_dimensions(rng, specs, attempt):
    count = len(specs)
    max_height = max(s["upper"].shape[0] + s["lower"].shape[0] for s in specs)
    max_width = max(s["upper"].shape[1] for s in specs) + 2
    if attempt >= 14:
        return 30, 30
    mode = int(rng.integers(0, 5))
    if count == 1 and (mode == 0 or (attempt < 5 and bool(rng.integers(0, 3) == 0))):
        return (min(30, max_height + int(rng.integers(1, 5))),
                min(30, max_width + int(rng.integers(1, 5))))
    if mode == 1:
        return (int(rng.integers(max(max_height, 17), 31)),
                int(rng.integers(max(max_width, 9), 17)))
    if mode == 2:
        return (int(rng.integers(max(max_height, 9), 17)),
                int(rng.integers(max(max_width, 17), 31)))
    if mode == 3:
        return (int(rng.integers(max(max_height, 27), 31)),
                int(rng.integers(max(max_width, 27), 31)))
    minimum = max(7, 5 + count * 3)
    return (int(rng.integers(max(max_height, minimum), 31)),
            int(rng.integers(max(max_width, minimum), 31)))


def _choose_colors(rng, background, count, bridge=None, train_colors=None, force_reuse=False):
    allowed = [c for c in range(10) if c != background]
    if bridge is not None:
        pairs = list(bridge)
    else:
        if train_colors and bool(rng.integers(0, 2)):
            unseen = [c for c in allowed if c not in train_colors]
            first = int(rng.choice(unseen if unseen else allowed))
        else:
            first = int(rng.choice(allowed))
        second = int(rng.choice([c for c in allowed if c != first]))
        pairs = [(first, second)]
    while len(pairs) < count:
        if force_reuse and len(pairs) == 1:
            upper = pairs[0][1]
        elif bool(rng.integers(0, 3)):
            upper = int(rng.choice([c for pair in pairs for c in pair if c != background]))
        else:
            upper = int(rng.choice(allowed))
        lower = int(rng.choice([c for c in allowed if c != upper]))
        pairs.append((upper, lower))
    return pairs


def _infer_background(grid):
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    winners = np.flatnonzero(counts == counts.max())
    assert len(winners) == 1 and int(counts.max()) > grid.size // 2
    return int(winners[0])


def _analyze_vertical(grid):
    background = _infer_background(grid)
    components = _connected_components(grid != background)
    assert components
    objects = []
    for component in components:
        colors = {int(grid[r, c]) for r, c in component}
        assert len(colors) == 2
        sections = []
        for color in colors:
            cells = {(r, c) for r, c in component if int(grid[r, c]) == color}
            assert _cells_are_connected(cells)
            sections.append((min(r for r, _ in cells), max(r for r, _ in cells), color, cells))
        sections.sort()
        ut, ub, uc, upper = sections[0]
        lt, lb, lc, lower = sections[1]
        assert ub + 1 == lt
        left = min(c for _, c in component)
        right = max(c for _, c in component)
        assert right - left + 1 >= 3
        assert left >= 1 and right <= grid.shape[1] - 2
        assert min(c for _, c in upper) == min(c for _, c in lower) == left
        assert max(c for _, c in upper) == max(c for _, c in lower) == right
        assert {r for r, _ in upper} == set(range(ut, ub + 1))
        assert {r for r, _ in lower} == set(range(lt, lb + 1))
        assert all((ub, c) in upper and (lt, c) in lower for c in range(left, right + 1))
        objects.append({"upper_color": uc, "lower_color": lc,
                        "upper_cells": tuple(sorted(upper)), "lower_cells": tuple(sorted(lower)),
                        "upper_row_min": ut, "upper_row_max": ub,
                        "lower_row_min": lt, "lower_row_max": lb,
                        "left": left, "right": right, "top": ut, "bottom": lb})
    objects.sort(key=lambda obj: (obj["top"], obj["left"]))
    return background, objects


def _analyze_input(grid):
    try:
        background, objects = _analyze_vertical(grid)
        return background, objects, "vertical", grid
    except AssertionError:
        normalized = np.rot90(grid, 1)
        background, objects = _analyze_vertical(normalized)
        return background, objects, "rotated", normalized


def _solve_input(input_grid):
    background, objects, axis, normalized = _analyze_input(input_grid)
    output = np.full_like(normalized, background)
    occupied = np.zeros(normalized.shape, dtype=bool)
    for obj in objects:
        for row, col in obj["upper_cells"]:
            assert not occupied[row, col - 1]
            occupied[row, col - 1] = True
            output[row, col - 1] = obj["upper_color"]
        for row, col in obj["lower_cells"]:
            assert not occupied[row, col + 1]
            occupied[row, col + 1] = True
            output[row, col + 1] = obj["lower_color"]
    return np.rot90(output, -1).copy() if axis == "rotated" else output


def _make_pair(rng, count, background, color_pairs, axis, richer=False):
    for attempt in range(28):
        specs = [_make_spec(rng, pair, richer) for pair in color_pairs]
        for dimension_try in range(18):
            height, width = _sample_dimensions(rng, specs, dimension_try)
            placements = _place_specs(rng, specs, height, width)
            if placements is None:
                continue
            input_grid = np.full((height, width), background, dtype=np.int8)
            output_grid = np.full((height, width), background, dtype=np.int8)
            for spec, (top, left) in zip(specs, placements):
                for row, col, color in _spec_cells(spec, top, left, False):
                    input_grid[row, col] = color
                for row, col, color in _spec_cells(spec, top, left, True):
                    assert output_grid[row, col] == background
                    output_grid[row, col] = color
            if np.count_nonzero(input_grid == background) <= input_grid.size // 2:
                continue
            if axis == "rotated":
                input_grid = np.rot90(input_grid, -1).copy()
                output_grid = np.rot90(output_grid, -1).copy()
            try:
                if np.array_equal(output_grid, _solve_input(input_grid)):
                    return input_grid, output_grid
            except AssertionError:
                pass
    raise RuntimeError("could not construct a separated pair")


def _rows_key(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _grid_key(grid):
    return grid.shape, tuple(int(v) for v in grid.ravel())


def _is_source_pair(input_grid, output_grid):
    return (_rows_key(input_grid), _rows_key(output_grid)) in _SOURCE_PAIR_ROWS


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert bool(np.all((grid >= 0) & (grid <= 9)))


def _section_stats(obj, prefix):
    cells = obj[prefix + "_cells"]
    row_min = obj[prefix + "_row_min"]
    row_max = obj[prefix + "_row_max"]
    width = obj["right"] - obj["left"] + 1
    height = row_max - row_min + 1
    mask = np.zeros((height, width), dtype=bool)
    for row, col in cells:
        mask[row - row_min, col - obj["left"]] = True
    rectangle = bool(np.all(mask))
    runs = []
    max_gap = 0
    for row in mask:
        cols = np.flatnonzero(row)
        runs.append(1 + sum(b > a + 1 for a, b in zip(cols, cols[1:])))
        for a, b in zip(cols, cols[1:]):
            max_gap = max(max_gap, int(b - a - 1))
    outer = mask[0] if prefix == "upper" else mask[-1]
    outer_cols = np.flatnonzero(outer)
    outer_split = any(b > a + 1 for a, b in zip(outer_cols, outer_cols[1:]))
    holes = [component for component in _connected_components(~mask)
             if all(0 < row < height - 1 and 0 < col < width - 1
                    for row, col in component)]
    return {"height": height, "width": width, "rectangle": rectangle,
            "max_runs": max(runs), "max_gap": max_gap,
            "outer_split": outer_split, "hole": bool(holes),
            "max_hole_area": max((len(hole) for hole in holes), default=0),
            "cells": len(cells)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_axes = set()
    test_axes = set()
    train_roles = {}
    backgrounds = []
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            assert not _is_source_pair(inp, out)
            background, objects, axis, _ = _analyze_input(inp)
            assert 1 <= len(objects) <= 5
            assert np.array_equal(out, _solve_input(inp))
            assert _infer_background(out) == background
            backgrounds.append(background)
            if regime == "train":
                train_axes.add(axis)
            else:
                test_axes.add(axis)
            for obj in objects:
                for prefix in ("upper", "lower"):
                    stats = _section_stats(obj, prefix)
                    assert 3 <= stats["width"] <= 9
                    assert 1 <= stats["height"] <= 6
                if regime == "train":
                    train_roles.setdefault(obj["upper_color"], set()).add("upper")
                    train_roles.setdefault(obj["lower_color"], set()).add("lower")
    assert test_axes <= train_axes
    assert len(set(backgrounds)) >= 2
    assert any(len(roles) == 2 for roles in train_roles.values())
    assert len({len(_analyze_input(p["input"])[1]) for p in puzzle["train"]}) >= 2


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    bridge = tuple(int(c) for c in rng.choice(10, size=5, replace=False))
    mode = str(rng.choice(("vertical", "rotated", "mixed")))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_colors = set()
    for overall in range(train_count + test_count):
        train = overall < train_count
        index = overall if train else overall - train_count
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        if train and index == 0:
            count = 1
            background = bridge[2]
            color_pairs = [(bridge[0], bridge[1])]
        elif train and index == 1:
            count = 2
            background = bridge[0]
            second_object = ((bridge[2], bridge[1]),
                             (bridge[3], bridge[1]),
                             (bridge[3], bridge[4]))[int(pair_rng.integers(0, 3))]
            color_pairs = [(bridge[1], bridge[2]), second_object]
        else:
            count = int(pair_rng.integers(1, 6))
            background = int(pair_rng.integers(0, 10))
            color_pairs = _choose_colors(pair_rng, background, count,
                                         train_colors=train_colors if not train else None,
                                         force_reuse=(count > 1 and bool(pair_rng.integers(0, 2))))
        if mode == "mixed":
            axis = "vertical" if train and index == 0 else (
                "rotated" if train and index == 1 else str(pair_rng.choice(("vertical", "rotated"))))
        else:
            axis = mode
        for _ in range(12):
            inp, out = _make_pair(pair_rng, count, background, color_pairs, axis,
                                  richer=(not train))
            key = _grid_key(inp)
            if key in seen or _is_source_pair(inp, out):
                continue
            seen.add(key)
            puzzle["train" if train else "test"].append({"input": inp, "output": out})
            if train:
                for upper, lower in color_pairs:
                    train_colors.update((upper, lower))
            break
        else:
            raise RuntimeError("could not make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[k]) == len(second[k]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[k], second[k])) for k in ("train", "test"))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    dims, layouts, counts, axes, shapes, backgrounds = set(), set(), set(), set(), set(), set()
    widths_by_count = {count: set() for count in range(1, 6)}
    measured = {name: 0 for name in (
        "rectangles", "bars", "splits", "three_runs", "wide_gaps",
        "holes", "large_holes", "combined", "unequal", "compact",
        "full", "elongated", "top", "bottom", "test_single",
        "test_simple", "test_new_color", "mixed_axis",
    )}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        pair_backgrounds = set()
        pair_palettes = set()
        train_axes = set()
        train_colors = set()
        for regime, pair in ((regime, pair) for regime in ("train", "test")
                             for pair in puzzle[regime]):
            inp, out = pair["input"], pair["output"]
            background, objects, axis, normalized = _analyze_input(inp)
            dims.add(inp.shape)
            counts.add(len(objects))
            axes.add(axis)
            backgrounds.add(background)
            pair_backgrounds.add(background)
            pair_palettes.add(tuple(sorted(int(color) for color in np.unique(inp))))
            if regime == "train":
                train_axes.add(axis)
                train_colors.update(color for obj in objects
                                    for color in (obj["upper_color"], obj["lower_color"]))
            else:
                measured["test_single"] += len(objects) == 1
                measured["test_new_color"] += bool({color for obj in objects
                    for color in (obj["upper_color"], obj["lower_color"])} - train_colors)
                measured["test_simple"] += all(
                    _section_stats(obj, part)["rectangle"]
                    for obj in objects for part in ("upper", "lower"))
            measured["compact"] += max(inp.shape) <= 12
            measured["full"] += min(inp.shape) >= 27
            measured["elongated"] += max(inp.shape) / min(inp.shape) >= 1.8
            measured["top"] += any(obj["top"] == 0 for obj in objects)
            measured["bottom"] += any(obj["bottom"] == normalized.shape[0] - 1
                                      for obj in objects)
            layouts.add((inp.shape, axis, tuple((o["top"], o["left"], o["right"], o["bottom"]) for o in objects)))
            for obj in objects:
                upper = _section_stats(obj, "upper")
                lower = _section_stats(obj, "lower")
                widths_by_count[len(objects)].add(upper["width"])
                measured["unequal"] += abs(upper["height"] - lower["height"]) >= 3
                for section in (upper, lower):
                    measured["rectangles"] += section["rectangle"]
                    measured["bars"] += section["height"] == 1
                    measured["splits"] += section["max_runs"] > 1
                    measured["three_runs"] += section["max_runs"] >= 3
                    measured["wide_gaps"] += section["max_gap"] >= 2
                    measured["holes"] += section["hole"]
                    measured["large_holes"] += section["max_hole_area"] >= 2
                    measured["combined"] += section["outer_split"] and section["hole"]
                shapes.add((upper["width"], upper["height"], lower["height"], upper["cells"], lower["cells"]))
            assert np.array_equal(out, _solve_input(inp))
        assert len(pair_backgrounds) >= 2 and len(pair_palettes) >= 2
        measured["mixed_axis"] += len(train_axes) == 2
    assert len(dims) >= 60 and len(layouts) >= 100
    assert counts == {1, 2, 3, 4, 5} and axes == {"vertical", "rotated"}
    assert len(backgrounds) == 10 and len(shapes) >= 100
    assert all(widths == set(range(3, 10)) for widths in widths_by_count.values())
    assert all(value > 0 for value in measured.values())
    for seed, nt, ns in ((500, 2, 1), (501, 10, 1), (502, 10, 4),
                         (600, None, 1), (601, None, 4),
                         (602, 2, None), (603, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
    assert _puzzles_equal(generate_puzzle(987654, 7, 3), generate_puzzle(987654, 7, 3))
    assert _puzzles_equal(generate_puzzle(7654321), generate_puzzle(7654321))
    assert len({_grid_key(generate_puzzle(seed, 5, 2)["train"][0]["input"])
                for seed in range(700, 710)}) >= 9
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4},
                 {"num_train": True}, {"num_test": False}):
        try:
            generate_puzzle(123, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {args}")
