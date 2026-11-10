"""Solved examples of marker relocation to the midpoint of odd bars."""

import numpy as np


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    choices = [
        (n, m)
        for n in range(2, 11)
        for m in range(1, 5)
        if n > m
        and (num_train is None or n == num_train)
        and (num_test is None or m == num_test)
    ]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _cells(orientation, fixed, start, length):
    if orientation == "H":
        return [(fixed, x) for x in range(start, start + length)]
    return [(y, fixed) for y in range(start, start + length)]


def _crossing(orientation, fixed, start, length):
    middle = start + length // 2
    if orientation == "H":
        return [(fixed - 1, middle), (fixed, middle), (fixed + 1, middle)]
    return [(middle, fixed - 1), (middle, fixed), (middle, fixed + 1)]


def _distance(point, cells):
    return min(abs(point[0] - y) + abs(point[1] - x) for y, x in cells)


def _clearance(first, second):
    return min(
        max(abs(y - v), abs(x - u))
        for y, x in first
        for v, u in second
    )


def _marker_options(rng, orientation, fixed, start, length, mode):
    end = start + length - 1
    middle = start + length // 2
    if mode == "endpoint":
        projections = [start, end]
        offsets = [1]
    elif mode == "inner":
        projections = [x for x in range(start + 1, end) if x != middle]
        offsets = [1]
    elif mode == "midpoint":
        projections = [middle]
        offsets = [1]
    elif mode == "beyond":
        projections = [start - 1, end + 1]
        offsets = [1]
    elif mode == "detached":
        projections = [start, middle, end]
        offsets = [2]
    else:
        raise ValueError(mode)
    if not projections:
        projections = [start, end]
    options = []
    for projection in projections:
        for offset in offsets:
            for side in (-1, 1):
                if orientation == "H":
                    point = (fixed + side * offset, projection)
                else:
                    point = (projection, fixed + side * offset)
                if 0 <= point[0] < 30 and 0 <= point[1] < 30:
                    options.append(point)
    rng.shuffle(options)
    return options


def _geometry_ok(objects):
    for index, obj in enumerate(objects):
        bar = obj["cells"]
        cross = obj["cross"]
        marker = obj["marker"]
        if marker in bar:
            return False
        if any(not (0 <= y < 30 and 0 <= x < 30) for y, x in cross):
            return False
        own_distance = _distance(marker, bar)
        if own_distance not in (1, 2):
            return False
        for other in objects[:index]:
            if _clearance(bar, other["cells"]) <= 1:
                return False
            if _clearance(cross, other["cells"]) <= 1:
                return False
            if _clearance(other["cross"], bar) <= 1:
                return False
            if _clearance(cross, other["cross"]) <= 1:
                return False
            if _clearance([marker], [other["marker"]]) <= 1:
                return False
            if _clearance([marker], other["cross"]) <= 1:
                return False
            if _clearance([other["marker"]], cross) <= 1:
                return False
            if _distance(marker, other["cells"]) < own_distance + 2:
                return False
            other_distance = _distance(other["marker"], other["cells"])
            if _distance(other["marker"], bar) < other_distance + 2:
                return False
    return True


def _candidate_object(rng, orientation, length, mode, fixed=None, start=None):
    if fixed is None:
        fixed = int(rng.integers(4, 26))
    if start is None:
        start = int(rng.integers(3, 28 - length))
    cells = _cells(orientation, fixed, start, length)
    crossing = _crossing(orientation, fixed, start, length)
    for marker in _marker_options(rng, orientation, fixed, start, length, mode):
        yield {
            "orientation": orientation,
            "length": length,
            "cells": cells,
            "cross": crossing,
            "marker": marker,
        }


def _place_objects(rng, orientations, lengths, modes, aligned):
    count = len(lengths)
    for _ in range(35):
        objects = []
        aligned_positions = None
        if aligned:
            gaps = [int(rng.integers(2, 5)) for _ in range(count - 1)]
            span = sum(lengths) + sum(gaps)
            if span <= 26:
                first = int(rng.integers(2, 30 - span))
                fixed = int(rng.integers(5, 25))
                aligned_positions = []
                for index, length in enumerate(lengths):
                    aligned_positions.append((fixed, first))
                    first += length + (gaps[index] if index < count - 1 else 0)
        for index, (orientation, length, mode) in enumerate(
            zip(orientations, lengths, modes)
        ):
            found = None
            tries = 1 if aligned_positions is not None else 100
            for _ in range(tries):
                placement = (
                    aligned_positions[index] if aligned_positions is not None else (None, None)
                )
                for candidate in _candidate_object(
                    rng, orientation, length, mode, *placement
                ):
                    if _geometry_ok(objects + [candidate]):
                        found = candidate
                        break
                if found is not None:
                    break
            if found is None:
                break
            objects.append(found)
        if len(objects) == count:
            return objects
        if aligned_positions is None and count >= 3:
            # Wider placement coordinates are a deterministic fallback for dense cases.
            for _ in range(80):
                objects = []
                for orientation, length, mode in zip(orientations, lengths, modes):
                    found = None
                    for _ in range(140):
                        fixed = int(rng.integers(2, 28))
                        start = int(rng.integers(2, 29 - length))
                        for candidate in _candidate_object(
                            rng, orientation, length, mode, fixed, start
                        ):
                            if _geometry_ok(objects + [candidate]):
                                found = candidate
                                break
                        if found is not None:
                            break
                    if found is None:
                        break
                    objects.append(found)
                if len(objects) == count:
                    return objects
    raise RuntimeError("Could not place separate bars")


def _make_pair(rng, background, orientations, lengths, modes, split_bars, split_markers):
    count = len(lengths)
    aligned = (
        count in (2, 3)
        and len(set(orientations)) == 1
        and sum(lengths) + 2 * (count - 1) <= 26
        and rng.random() < 0.24
    )
    try:
        objects = _place_objects(rng, orientations, lengths, modes, aligned)
    except RuntimeError:
        if not aligned:
            raise
        objects = _place_objects(rng, orientations, lengths, modes, False)
    palette = [color for color in range(10) if color != background]
    rng.shuffle(palette)
    bar_count = count if split_bars and count > 1 else 1
    marker_count = count if split_markers and count > 1 else 1
    bar_colors = palette[:bar_count]
    marker_colors = palette[bar_count:bar_count + marker_count]

    occupied = [
        point
        for obj in objects
        for point in obj["cells"] + obj["cross"] + [obj["marker"]]
    ]
    top = min(y for y, _ in occupied)
    bottom = max(y for y, _ in occupied)
    left = min(x for _, x in occupied)
    right = max(x for _, x in occupied)
    margins = [0, 0, 1, 1, 2, 3, 4]
    top = max(0, top - int(rng.choice(margins)))
    bottom = min(29, bottom + int(rng.choice(margins)))
    left = max(0, left - int(rng.choice(margins)))
    right = min(29, right + int(rng.choice(margins)))
    shape = (bottom - top + 1, right - left + 1)
    input_grid = np.full(shape, background, dtype=np.int8)
    output_grid = input_grid.copy()
    for index, obj in enumerate(objects):
        bar_color = bar_colors[index] if bar_count > 1 else bar_colors[0]
        for y, x in obj["cells"]:
            input_grid[y - top, x - left] = bar_color
            output_grid[y - top, x - left] = bar_color
    for index, obj in enumerate(objects):
        marker_color = marker_colors[index] if marker_count > 1 else marker_colors[0]
        y, x = obj["marker"]
        input_grid[y - top, x - left] = marker_color
        for cy, cx in obj["cross"]:
            output_grid[cy - top, cx - left] = marker_color
    return {"input": input_grid, "output": output_grid}


def _components(mask):
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        visited[row, col] = True
        pending = [(row, col)]
        cells = []
        while pending:
            y, x = pending.pop()
            cells.append((y, x))
            for v, u in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= v < mask.shape[0]
                    and 0 <= u < mask.shape[1]
                    and mask[v, u]
                    and not visited[v, u]
                ):
                    visited[v, u] = True
                    pending.append((v, u))
        components.append(cells)
    return components


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(counts)])
    assert int(max(counts)) > inp.size // 2
    bars, markers = [], []
    for color in colors:
        color = int(color)
        if color == background:
            continue
        for cells in _components(inp == color):
            if len(cells) == 1:
                markers.append({"point": cells[0], "color": color})
                continue
            assert len(cells) >= 3 and len(cells) % 2 == 1
            rows = {y for y, _ in cells}
            cols = {x for _, x in cells}
            assert len(rows) == 1 or len(cols) == 1
            if len(rows) == 1:
                orientation = "H"
                fixed = next(iter(rows))
                start = min(cols)
                assert max(cols) - start + 1 == len(cells)
            else:
                orientation = "V"
                fixed = next(iter(cols))
                start = min(rows)
                assert max(rows) - start + 1 == len(cells)
            assert len(cells) <= 21
            bars.append({
                "orientation": orientation,
                "fixed": fixed,
                "start": start,
                "length": len(cells),
                "cells": cells,
                "cross": _crossing(orientation, fixed, start, len(cells)),
                "color": color,
            })
    assert 1 <= len(bars) <= 4 and len(markers) == len(bars)
    assert not ({bar["color"] for bar in bars} & {m["color"] for m in markers})
    for i, bar in enumerate(bars):
        for other in bars[:i]:
            assert _clearance(bar["cells"], other["cells"]) > 1
            assert _clearance(bar["cross"], other["cells"]) > 1
            assert _clearance(other["cross"], bar["cells"]) > 1
            assert _clearance(bar["cross"], other["cross"]) > 1
    assignments = {}
    for marker in markers:
        distances = [_distance(marker["point"], bar["cells"]) for bar in bars]
        index = int(np.argmin(distances))
        own = distances[index]
        assert own in (1, 2)
        assert all(d >= own + 2 for j, d in enumerate(distances) if j != index)
        assert index not in assignments
        assignments[index] = marker
    assert len(assignments) == len(bars)
    expected = inp.copy()
    for marker in markers:
        expected[marker["point"]] = background
    modes = []
    for index, bar in enumerate(bars):
        marker = assignments[index]
        point = marker["point"]
        projection = point[1] if bar["orientation"] == "H" else point[0]
        offset = abs(point[0] - bar["fixed"]) if bar["orientation"] == "H" else abs(point[1] - bar["fixed"])
        middle = bar["start"] + bar["length"] // 2
        end = bar["start"] + bar["length"] - 1
        assert offset in (1, 2) and bar["start"] - 1 <= projection <= end + 1
        if offset == 2:
            assert projection in (bar["start"], middle, end)
            mode = "detached"
        elif projection in (bar["start"] - 1, end + 1):
            mode = "beyond"
        elif projection in (bar["start"], end):
            mode = "endpoint"
        elif projection == middle:
            mode = "midpoint"
        else:
            mode = "inner"
        modes.append(mode)
        assert point not in bar["cells"]
        for target in bar["cross"]:
            y, x = target
            assert 0 <= y < inp.shape[0] and 0 <= x < inp.shape[1]
            assert inp[target] == (
                bar["color"] if target == bar["cross"][1]
                else marker["color"] if target == point
                else background
            )
            expected[target] = marker["color"]
        for other_index, other in enumerate(bars):
            if other_index != index:
                assert _clearance([point], other["cross"]) > 1
    for i, first in enumerate(markers):
        for second in markers[:i]:
            assert _clearance([first["point"]], [second["point"]]) > 1
    assert np.array_equal(out, expected)
    return {
        "background": background,
        "bars": bars,
        "markers": markers,
        "assignments": assignments,
        "modes": modes,
        "orientations": frozenset(bar["orientation"] for bar in bars),
        "split_bars": len({bar["color"] for bar in bars}) > 1,
        "split_markers": len({marker["color"] for marker in markers}) > 1,
        "edge": any(
            y in (0, inp.shape[0] - 1) or x in (0, inp.shape[1] - 1)
            for bar in bars for y, x in bar["cross"]
        ),
    }


def _original_inputs():
    specs = [
        ((13, 13), 3, 2, [("H", 2, 2, 9)], [(1, 2)]),
        ((15, 15), 4, 8, [("H", 2, 1, 3), ("H", 8, 8, 5)], [(1, 3), (7, 12)]),
        ((11, 11), 6, 7, [("H", 4, 3, 7), ("H", 9, 1, 9)], [(3, 3), (8, 1)]),
        ((12, 12), 5, 3, [("H", 3, 2, 9)], [(2, 10)]),
        ((11, 11), 1, 7, [("V", 4, 2, 7)], [(2, 5)]),
    ]
    originals = []
    for shape, bar_color, marker_color, bars, markers in specs:
        grid = np.zeros(shape, dtype=np.int8)
        for orientation, fixed, start, length in bars:
            for point in _cells(orientation, fixed, start, length):
                grid[point] = bar_color
        for point in markers:
            grid[point] = marker_color
        originals.append(grid)
    return originals


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    originals = _original_inputs()
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            info = _inspect_pair(pair)
            inp_key = _grid_key(pair["input"])
            pair_key = (inp_key, _grid_key(pair["output"]))
            assert inp_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(inp_key)
            seen_pairs.add(pair_key)
            assert not any(np.array_equal(pair["input"], old) for old in originals)
            infos[section].append(info)
    training = infos["train"]
    testing = infos["test"]
    assert any(len(info["bars"]) == 1 for info in training)
    assert any(len(info["bars"]) >= 2 for info in training)
    assert any("endpoint" in info["modes"] for info in training)
    for property_name in ("split_bars", "split_markers"):
        if any(info[property_name] for info in testing):
            assert any(info[property_name] for info in training)
    if any(
        mode in ("detached", "beyond")
        for info in testing for mode in info["modes"]
    ):
        assert any(
            mode in ("detached", "beyond")
            for info in training for mode in info["modes"]
        )
    if any(info["background"] != training[0]["background"] for info in testing):
        assert len({info["background"] for info in training}) > 1


def _same_puzzle(left, right):
    return all(
        len(left[section]) == len(right[section])
        and all(
            np.array_equal(first["input"], second["input"])
            and np.array_equal(first["output"], second["output"])
            for first, second in zip(left[section], right[section])
        )
        for section in ("train", "test")
    )


def _normalized_structure(grid):
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    points = np.argwhere(grid != background)
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0)
    cropped = grid[top:bottom + 1, left:right + 1]
    mapping = {background: 0}
    values = []
    for value in cropped.flat:
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        values.append(mapping[value])
    return cropped.shape, tuple(values)


def _orientation_list(rng, regime, plan, count):
    if plan == "source_shift":
        choices = ["H"] if regime == "train" else ["H", "V"]
    elif plan == "reverse_shift":
        choices = ["V"] if regime == "train" else ["H", "V"]
    elif plan == "same_h":
        choices = ["H"]
    elif plan == "same_v":
        choices = ["V"]
    else:
        choices = ["H", "V", "M"]
    choice = str(rng.choice(choices))
    if count == 1 and choice == "M":
        choice = str(rng.choice(["H", "V"]))
    if choice == "M":
        orientations = ["H", "V"] + [
            str(rng.choice(["H", "V"])) for _ in range(count - 2)
        ]
        rng.shuffle(orientations)
        return orientations
    return [choice] * count


def _lengths(rng, count, regime, forced_long=None):
    if forced_long is not None:
        return [forced_long]
    ceiling = 15 if regime == "train" and count == 1 else 19 if count == 1 else 13 if count == 2 else 9
    options = list(range(3, ceiling + 1, 2))
    return [int(rng.choice(options)) for _ in range(count)]


def _make_spec(rng, count, regime, plan, extension, forced_mode=None, forced_long=None):
    orientations = _orientation_list(rng, regime, plan, count)
    lengths = _lengths(rng, count, regime, forced_long)
    modes = []
    for length in lengths:
        options = ["endpoint", "endpoint", "endpoint", "inner", "midpoint"]
        if extension:
            options += ["beyond", "detached"]
        mode = str(rng.choice(options))
        if length == 3 and mode == "inner":
            mode = "endpoint"
        modes.append(mode)
    if forced_mode is not None:
        modes[0] = forced_mode
    return orientations, lengths, modes


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    plan = str(rng.choice([
        "source_shift", "reverse_shift", "same_h", "same_v",
        "seen_both", "mixed_grid",
    ]))
    first_background = int(rng.integers(10))
    variable_background = bool(rng.integers(2))
    color_profile = int(rng.integers(4))
    extension = bool(rng.integers(2))
    long_test = bool(rng.random() < 0.27)
    double_index = int(rng.integers(2))
    shifted_index = int(rng.integers(test_count))
    puzzle = {"train": [], "test": []}
    used = set()
    max_train_length = 3

    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if section == "train" and index < 2:
                bar_count = 2 if index == double_index else 1
            elif section == "test" and index == 0 and long_test:
                bar_count = 1
            else:
                bar_count = int(rng.choice([1, 2, 2, 3, 3, 4]))
            forced_long = None
            if section == "test" and index == 0 and long_test:
                forced_long = min(19, max_train_length + 2)
                if forced_long % 2 == 0:
                    forced_long -= 1
            forced_mode = None
            if section == "train" and index < 2:
                forced_mode = (
                    "detached" if extension and bar_count == 2 else "endpoint"
                )
            orientations, lengths, modes = _make_spec(
                rng, bar_count, section, plan, extension, forced_mode, forced_long
            )
            if section == "train" and index < 2 and bar_count == 2:
                modes[1] = "endpoint"
                if plan == "mixed_grid":
                    orientations = ["H", "V"]
                    rng.shuffle(orientations)
            if section == "train" and index < 2 and plan == "seen_both":
                orientations = ["H" if index == 0 else "V"] * bar_count
            if section == "test" and plan in ("source_shift", "reverse_shift"):
                shifted = "V" if plan == "source_shift" else "H"
                if index == shifted_index:
                    orientations = [shifted] * bar_count
                else:
                    orientations = [("H" if shifted == "V" else "V")] * bar_count
            if section == "train" and index == 0:
                background = first_background
            elif variable_background and section == "train" and index == 1:
                choices = [color for color in range(10) if color != first_background]
                background = int(rng.choice(choices))
            elif variable_background:
                background = int(rng.integers(10))
            else:
                background = first_background
            if bar_count == 1:
                split_bars = split_markers = False
            elif section == "train" and index == double_index:
                split_bars = bool(color_profile & 1)
                split_markers = bool(color_profile & 2)
            else:
                split_bars = bool(color_profile & 1) and bool(rng.integers(2))
                split_markers = bool(color_profile & 2) and bool(rng.integers(2))
            for _ in range(30):
                pair = _make_pair(
                    rng, background, orientations, lengths, modes,
                    split_bars, split_markers,
                )
                if _grid_key(pair["input"]) not in used:
                    try:
                        _inspect_pair(pair)
                    except AssertionError:
                        continue
                    if not any(
                        np.array_equal(pair["input"], old) for old in _original_inputs()
                    ):
                        break
            else:
                raise RuntimeError("Could not generate a valid unique pair")
            puzzle[section].append(pair)
            used.add(_grid_key(pair["input"]))
            if section == "train":
                max_train_length = max(max_train_length, *lengths)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert all(_same_puzzle(p, generate_puzzle(seed)) for seed, p in enumerate(puzzles))
    signatures = {
        tuple(
            (_grid_key(pair["input"]), _grid_key(pair["output"]))
            for section in ("train", "test")
            for pair in puzzle[section]
        )
        for puzzle in puzzles
    }
    assert len(signatures) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_infos = []
    shapes = set()
    structures = set()
    orientation_patterns = set()
    measured = {
        name: 0 for name in (
            "source_shift", "reverse_shift", "same_h", "same_v",
            "mixed_train", "mixed_test", "variable_background",
            "constant_background", "longer_test", "same_line",
            "staggered", "compact", "combined",
        )
    }
    sides = set()
    edge_sides = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_infos = [_inspect_pair(pair) for pair in puzzle["train"]]
        test_infos = [_inspect_pair(pair) for pair in puzzle["test"]]
        all_infos.extend(train_infos + test_infos)
        measured["source_shift"] += (
            all(info["orientations"] == frozenset({"H"}) for info in train_infos)
            and any("V" in info["orientations"] for info in test_infos)
        )
        measured["reverse_shift"] += (
            all(info["orientations"] == frozenset({"V"}) for info in train_infos)
            and any("H" in info["orientations"] for info in test_infos)
        )
        measured["same_h"] += all(
            info["orientations"] == frozenset({"H"})
            for info in train_infos + test_infos
        )
        measured["same_v"] += all(
            info["orientations"] == frozenset({"V"})
            for info in train_infos + test_infos
        )
        measured["mixed_train"] += any(
            len(info["orientations"]) == 2 for info in train_infos
        )
        measured["mixed_test"] += any(
            len(info["orientations"]) == 2 for info in test_infos
        )
        backgrounds = {info["background"] for info in train_infos + test_infos}
        measured["variable_background"] += len(backgrounds) > 1
        measured["constant_background"] += len(backgrounds) == 1
        measured["longer_test"] += (
            max(bar["length"] for info in test_infos for bar in info["bars"])
            > max(bar["length"] for info in train_infos for bar in info["bars"])
        )
        orientation_patterns.add((
            tuple(sorted(set.union(*(set(i["orientations"]) for i in train_infos)))),
            tuple(sorted(set.union(*(set(i["orientations"]) for i in test_infos)))),
        ))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                shapes.add(pair["input"].shape)
                structures.add(_normalized_structure(pair["input"]))
                info = _inspect_pair(pair)
                bars = info["bars"]
                measured["same_line"] += any(
                    first["orientation"] == second["orientation"]
                    and first["fixed"] == second["fixed"]
                    for index, first in enumerate(bars)
                    for second in bars[:index]
                )
                measured["staggered"] += any(
                    first["orientation"] == second["orientation"]
                    and first["fixed"] != second["fixed"]
                    and first["start"] != second["start"]
                    for index, first in enumerate(bars)
                    for second in bars[:index]
                )
                measured["compact"] += len(bars) == 1 and pair["input"].size <= 100
                measured["combined"] += (
                    len(bars) >= 3
                    and len(info["orientations"]) == 2
                    and (info["split_bars"] or info["split_markers"])
                )
                height, width = pair["input"].shape
                for index, bar in enumerate(bars):
                    row, col = info["assignments"][index]["point"]
                    if bar["orientation"] == "H":
                        sides.add("above" if row < bar["fixed"] else "below")
                    else:
                        sides.add("left" if col < bar["fixed"] else "right")
                    for y, x in bar["cross"]:
                        if y == 0:
                            edge_sides.add("top")
                        if y == height - 1:
                            edge_sides.add("bottom")
                        if x == 0:
                            edge_sides.add("left")
                        if x == width - 1:
                            edge_sides.add("right")
    assert len(shapes) >= 30
    assert len(structures) >= 100
    assert len(orientation_patterns) >= 4
    assert all(measured.values())
    assert sides == {"above", "below", "left", "right"}
    assert edge_sides == {"top", "bottom", "left", "right"}
    assert {len(info["bars"]) for info in all_infos} >= {1, 2, 3, 4}
    assert {bar["orientation"] for info in all_infos for bar in info["bars"]} == {"H", "V"}
    assert any(len(info["orientations"]) == 2 for info in all_infos)
    assert any(info["split_bars"] for info in all_infos)
    assert any(info["split_markers"] for info in all_infos)
    assert any(info["edge"] for info in all_infos)
    assert any(bar["length"] >= 11 for info in all_infos for bar in info["bars"])
    assert {"endpoint", "inner", "midpoint", "beyond", "detached"} <= {
        mode for info in all_infos for mode in info["modes"]
    }
    assert any(
        len({_inspect_pair(pair)["background"] for section in ("train", "test") for pair in puzzle[section]}) > 1
        for puzzle in puzzles
    )
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(500 + train_count + test_count, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(600 + (train_count or 0) + (test_count or 0), train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for train_count, test_count in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (4, 4), (3, 4),
    ):
        try:
            generate_puzzle(0, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal pair counts accepted: {(train_count, test_count)}")
