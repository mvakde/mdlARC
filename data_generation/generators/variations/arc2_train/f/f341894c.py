"""Marker-directed bicolor domino puzzles."""

import numpy as np


SIDES = ("T", "B", "L", "R")
SOURCE_INPUTS = (
    ("8888888788", "8888888888", "1688888887", "8888888888",
     "8888618887", "8868886888", "8818881888", "8888888688",
     "8888888188", "8878887888"),
    ("87888887", "88888888", "86888881", "81818886",
     "88868888", "88888888", "78888168", "88878888"),
    ("78888888888878", "88168888888887", "88888888888888",
     "88888888888888", "88888888618887", "88888888168887",
     "88881888888868", "88886888888818", "88888888888888",
     "88888881688887", "88888888888888", "18888888888888",
     "68887888888888"),
    ("888888778878878888", "786188888888888888",
     "888888888888886187", "788816888888888888",
     "888888888888888888", "888888881688888887",
     "888888886188888887", "788888881688888888",
     "888888886188888887", "888888888818888888",
     "888888888861888888", "888888888886888888",
     "788168888888888888", "888888118888868888",
     "888888668888818888", "788888888888881688",
     "888888888888888888", "888888888887888888"),
)


def _signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


SOURCE_SIGNATURES = {
    ((len(rows), len(rows[0])), tuple(int(x) for row in rows for x in row))
    for rows in SOURCE_INPUTS
}


def _counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} is outside {low}..{high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _marker_position(side, line, height, width):
    return {"T": (0, line), "B": (height - 1, line),
            "L": (line, 0), "R": (line, width - 1)}[side]


def _ray(row, column, side, height, width):
    if side == "T":
        return [(r, column) for r in range(1, height)]
    if side == "B":
        return [(r, column) for r in range(height - 2, -1, -1)]
    if side == "L":
        return [(row, c) for c in range(1, width)]
    return [(row, c) for c in range(width - 2, -1, -1)]


def _run_features(records):
    longest = 1
    mixed_longest = 0
    staggered = False
    for axis in ("horizontal", "vertical"):
        aligned = {}
        by_line = {}
        for record in records:
            if record["axis"] != axis:
                continue
            aligned.setdefault(record["axis_cells"], []).append(record)
            by_line[record["line"]] = record
        for group in aligned.values():
            group.sort(key=lambda item: item["line"])
            run = []
            for record in group:
                if run and record["line"] != run[-1]["line"] + 1:
                    longest = max(longest, len(run))
                    if len({item["side"] for item in run}) > 1:
                        mixed_longest = max(mixed_longest, len(run))
                    run = []
                run.append(record)
            longest = max(longest, len(run))
            if len({item["side"] for item in run}) > 1:
                mixed_longest = max(mixed_longest, len(run))
        for line, record in by_line.items():
            following = by_line.get(line + 1)
            if following and record["axis_cells"] != following["axis_cells"]:
                staggered = True
    return longest, mixed_longest, staggered


def _inspect_pair(inp, out):
    assert inp.shape == out.shape
    height, width = inp.shape
    values, frequencies = np.unique(inp, return_counts=True)
    assert len(values) == 4
    assert np.count_nonzero(frequencies == max(frequencies)) == 1
    background = int(values[np.argmax(frequencies)])
    assert np.array_equal(inp == background, out == background)
    boundary = np.zeros(inp.shape, dtype=bool)
    boundary[0, :] = boundary[-1, :] = True
    boundary[:, 0] = boundary[:, -1] = True
    others = {int(x) for x in values if int(x) != background}
    marker_candidates = [color for color in others
                         if np.any(inp == color) and not np.any((inp == color) & ~boundary)]
    assert len(marker_candidates) == 1, "marker must be the unique boundary-only role"
    marker = marker_candidates[0]
    payload = others - {marker}
    assert len(payload) == 2
    assert all(np.any((inp == color) & ~boundary) for color in payload)
    assert np.array_equal(inp == marker, out == marker)
    payload_mask = np.isin(inp, list(payload))
    assert np.array_equal(payload_mask, np.isin(out, list(payload)))
    assert {int(x) for x in np.unique(out)} == set(int(x) for x in values)

    marker_positions = [tuple(int(x) for x in position)
                        for position in np.argwhere(inp == marker)]
    assert marker_positions
    records = []
    covered = set()
    convention = None
    changed = unchanged = 0
    changed_sides = set()
    for row, column in marker_positions:
        sides = []
        if row == 0:
            sides.append("T")
        if row == height - 1:
            sides.append("B")
        if column == 0:
            sides.append("L")
        if column == width - 1:
            sides.append("R")
        candidates = []
        for side in sides:
            ray = _ray(row, column, side, height, width)
            for near, far in zip(ray, ray[1:]):
                if {int(inp[near]), int(inp[far])} == payload:
                    candidates.append((side, near, far))
        assert len(candidates) == 1, "every marker needs one unambiguous inward domino"
        side, near, far = candidates[0]
        assert near not in covered and far not in covered
        covered.update((near, far))
        output_order = (int(out[near]), int(out[far]))
        assert set(output_order) == payload
        if convention is None:
            convention = output_order
        assert output_order == convention, "one inward color order per puzzle"
        input_order = (int(inp[near]), int(inp[far]))
        assert set(input_order) == payload
        if input_order == convention:
            unchanged += 1
        else:
            changed += 1
            changed_sides.add(side)
        axis = "vertical" if side in ("T", "B") else "horizontal"
        axis_cells = tuple(sorted((near[0], far[0]) if axis == "vertical"
                                  else (near[1], far[1])))
        records.append({"side": side, "line": column if axis == "vertical" else row,
                        "axis": axis, "axis_cells": axis_cells,
                        "gap": _ray(row, column, side, height, width).index(near)})
    assert covered == {tuple(int(x) for x in cell) for cell in np.argwhere(payload_mask)}
    assert len(covered) == 2 * len(marker_positions)
    longest, mixed_longest, staggered = _run_features(records)
    corner_markers = sum(r in (0, height - 1) and c in (0, width - 1)
                         for r, c in marker_positions)
    corner_payload = sum(r in (0, height - 1) and c in (0, width - 1)
                         for r, c in covered)
    return {"background": background, "marker": marker,
            "convention": convention, "sides": {record["side"] for record in records},
            "changed_sides": changed_sides, "objects": len(records),
            "changed": changed, "unchanged": unchanged,
            "boundary_payload": int(np.count_nonzero(payload_mask & boundary)),
            "corner_markers": corner_markers, "corner_payload": corner_payload,
            "longest_run": longest, "mixed_run": mixed_longest,
            "staggered": staggered, "records": records}


def _choose_sides(rng, first=None):
    if first is None:
        count = int(rng.choice([1, 2, 2, 3, 3, 4]))
        return tuple(str(x) for x in rng.choice(SIDES, size=count, replace=False))
    missing = [side for side in SIDES if side not in first]
    extra = [side for side in SIDES if side in first]
    rng.shuffle(extra)
    take = int(rng.integers(0, len(extra) + 1))
    return tuple(missing + extra[:take]) if missing else _choose_sides(rng)


def _dimensions(rng, regime):
    if regime == "source_train":
        return int(rng.integers(8, 23)), int(rng.integers(8, 23))
    if rng.random() < 0.6:
        return int(rng.integers(15, 28)), int(rng.integers(15, 28))
    return int(rng.integers(8, 23)), int(rng.integers(8, 23))


def _layout(rng, count):
    mode = str(rng.choice(("scattered", "staggered", "block2", "block3", "block4"),
                          p=(.31, .24, .17, .17, .11)))
    if mode.startswith("block") and count < int(mode[-1]):
        return "scattered"
    return mode


def _objects(rng, sides, count, layout):
    group_size = 0 if layout == "scattered" else (2 if layout == "staggered" else int(layout[-1]))
    groups = []
    if group_size:
        axes = [("T", "B"), ("L", "R")]
        usable = [axis for axis in axes if any(side in sides for side in axis)]
        mixed = [axis for axis in usable if all(side in sides for side in axis)]
        axis_pool = mixed if mixed and rng.random() < .8 else usable
        axis = axis_pool[int(rng.integers(0, len(axis_pool)))]
        available = [side for side in axis if side in sides]
        if len(available) == 2:
            first = available[int(rng.integers(0, 2))]
            second = available[1] if first == available[0] else available[0]
            groups = [first if index % 2 == 0 else second for index in range(group_size)]
        else:
            groups = available * group_size
    remaining = [side for side in sides if side not in groups]
    if len(groups) + len(remaining) > count:
        raise ValueError("not enough objects for side coverage and cluster")
    chosen = list(groups) + remaining
    dominant = sides[int(rng.integers(0, len(sides)))]
    while len(chosen) < count:
        chosen.append(dominant if rng.random() < .58 else sides[int(rng.integers(0, len(sides)))])
    return [{"side": side, "group": index < group_size} for index, side in enumerate(chosen)]


def _assign_lines(rng, objects, height, width):
    for axis_sides, length in ((("L", "R"), height), (("T", "B"), width)):
        indices = [i for i, obj in enumerate(objects) if obj["side"] in axis_sides]
        if len(indices) > length - 3:
            raise ValueError("too many marker rays on an axis")
        grouped = [i for i in indices if objects[i]["group"]]
        used = set()
        if grouped:
            starts = list(range(1, length - len(grouped)))
            if not starts:
                raise ValueError("no room for a block")
            start = starts[int(rng.integers(0, len(starts)))]
            for offset, index in enumerate(grouped):
                objects[index]["line"] = start + offset
                used.add(start + offset)
        for index in indices:
            if index in grouped:
                continue
            free = [line for line in range(length) if line not in used]
            corners = [line for line in (0, length - 1) if line in free]
            pool = corners if corners and rng.random() < .25 else free
            line = pool[int(rng.integers(0, len(pool)))]
            objects[index]["line"] = line
            used.add(line)
    markers = [_marker_position(obj["side"], obj["line"], height, width)
               for obj in objects]
    if len(set(markers)) != len(markers):
        raise ValueError("corner markers overlap")


def _starts(obj, objects, height, width):
    side = obj["side"]
    if side in ("T", "B"):
        length = height
        forbidden = {other["line"] for other in objects if other["side"] in ("L", "R")}
    else:
        length = width
        forbidden = {other["line"] for other in objects if other["side"] in ("T", "B")}
    positions = [start for start in range(length - 1)
                 if start not in forbidden and start + 1 not in forbidden]
    if side in ("T", "L"):
        positions = [start for start in positions if start >= 1]
    else:
        positions = [start for start in positions if start <= length - 3]
    return positions


def _assign_starts(rng, objects, height, width, layout):
    choices = [_starts(obj, objects, height, width) for obj in objects]
    if any(not candidates for candidates in choices):
        raise ValueError("no unobstructed domino position")
    group = [i for i, obj in enumerate(objects) if obj["group"]]
    if group:
        common = set(choices[group[0]])
        for index in group[1:]:
            common &= set(choices[index])
        if not common:
            raise ValueError("no shared block position")
        common = sorted(common)
        anchor = common[int(rng.integers(0, len(common)))]
        for index in group:
            if layout == "staggered":
                nearby = [x for x in choices[index] if x != anchor and abs(x - anchor) <= 3]
                if index % 2 and nearby:
                    objects[index]["start"] = nearby[int(rng.integers(0, len(nearby)))]
                else:
                    objects[index]["start"] = anchor
            else:
                objects[index]["start"] = anchor
    for index, obj in enumerate(objects):
        if index not in group:
            candidates = choices[index]
            obj["start"] = candidates[int(rng.integers(0, len(candidates)))]
    if rng.random() < .38:
        eligible = [i for i in range(len(objects)) if i not in group]
        if eligible:
            corner = [i for i in eligible if objects[i]["line"] in
                      (0, (width - 1 if objects[i]["side"] in ("T", "B") else height - 1))]
            pool = corner if corner and rng.random() < .5 else eligible
            index = pool[int(rng.integers(0, len(pool)))]
            side = objects[index]["side"]
            far_edge = (height - 2 if side == "T" else 0 if side == "B" else
                        width - 2 if side == "L" else 0)
            if far_edge in choices[index]:
                objects[index]["start"] = far_edge


def _cells(obj, height, width):
    side, line, start = obj["side"], obj["line"], obj["start"]
    if side == "T":
        return (start, line), (start + 1, line)
    if side == "B":
        return (start + 1, line), (start, line)
    if side == "L":
        return (line, start), (line, start + 1)
    return (line, start + 1), (line, start)


def _make_pair(rng, regime, sides, convention, required_changes, require_correct):
    for _ in range(240):
        height, width = _dimensions(rng, regime)
        maximum = min(16, max(3, (height + width) // 2 - 5))
        minimum = max(3, len(sides) + (1 if require_correct else 0))
        if maximum < minimum:
            continue
        count = int(rng.integers(minimum, maximum + 1))
        layout = _layout(rng, count)
        try:
            objects = _objects(rng, sides, count, layout)
            _assign_lines(rng, objects, height, width)
            _assign_starts(rng, objects, height, width, layout)
        except ValueError:
            continue
        bg, marker = [int(x) for x in rng.choice(
            [x for x in range(10) if x not in convention], size=2, replace=False)]
        proximal, distal = convention
        inp = np.full((height, width), bg, dtype=np.int8)
        out = np.full((height, width), bg, dtype=np.int8)
        for obj in objects:
            position = _marker_position(obj["side"], obj["line"], height, width)
            inp[position] = out[position] = marker
        rate = float(rng.choice((0, .15, .3, .5, .7, .85, 1)))
        reversed_flags = [bool(rng.random() < rate) for _ in objects]
        forced = set()
        for side in required_changes:
            index = next(i for i, obj in enumerate(objects) if obj["side"] == side)
            reversed_flags[index] = True
            forced.add(index)
        if require_correct:
            correct_choices = [i for i in range(count) if i not in forced]
            if not correct_choices:
                continue
            reversed_flags[correct_choices[int(rng.integers(0, len(correct_choices)))]] = False
        for obj, reverse in zip(objects, reversed_flags):
            near, far = _cells(obj, height, width)
            inp[near], inp[far] = (distal, proximal) if reverse else (proximal, distal)
            out[near], out[far] = proximal, distal
        try:
            info = _inspect_pair(inp, out)
            if layout == "scattered" and info["longest_run"] > 1:
                continue
            if layout == "staggered" and not info["staggered"]:
                continue
            if layout.startswith("block") and info["longest_run"] < int(layout[-1]):
                continue
            if info["sides"] != set(sides):
                continue
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise RuntimeError("bounded placement failed")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    convention = tuple(int(x) for x in rng.choice(10, size=2, replace=False))
    first_sides = _choose_sides(rng)
    side_plans = [first_sides, _choose_sides(rng, first_sides)]
    side_plans.extend(_choose_sides(rng) for _ in range(train_count - 2))
    required = [set() for _ in side_plans]
    for side in SIDES:
        candidates = [i for i, plan in enumerate(side_plans) if side in plan]
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        required[chosen].add(side)
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=train_count + test_count, dtype=np.int64)
    train = [_make_pair(np.random.default_rng(pair_seeds[i]), "source_train",
                        side_plans[i], convention, required[i], i == 0)
             for i in range(train_count)]
    test = [_make_pair(np.random.default_rng(pair_seeds[train_count + i]),
                       "target_test", _choose_sides(rng), convention, set(), False)
            for i in range(test_count)]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    infos = {"train": [], "test": []}
    convention = None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _signature(pair["input"])
            pair_key = (input_key, _signature(pair["output"]))
            assert input_key not in inputs and pair_key not in pairs
            assert input_key not in SOURCE_SIGNATURES
            inputs.add(input_key)
            pairs.add(pair_key)
            info = _inspect_pair(pair["input"], pair["output"])
            infos[split].append(info)
            if convention is None:
                convention = info["convention"]
            assert convention == info["convention"]
    assert set.union(*(info["sides"] for info in infos["train"])) == set(SIDES)
    assert set.union(*(info["changed_sides"] for info in infos["train"])) == set(SIDES)
    assert sum(info["unchanged"] for info in infos["train"]) > 0
    for info in infos["test"]:
        assert info["sides"] <= set.union(*(item["sides"] for item in infos["train"]))


def _puzzles_equal(first, second):
    return all(np.array_equal(a[key], b[key])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for key in ("input", "output")) and all(
        len(first[split]) == len(second[split]) for split in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(30000 + i) for i in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert _puzzles_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(40000 + (train_count or 0) * 10 + (test_count or 0),
                                 train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (4, 4), (3, 4)):
        try:
            generate_puzzle(41000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair counts accepted")
    all_infos = []
    palette_pairs = set()
    shapes = set()
    layouts = set()
    varied_colors = 0
    smaller_tests = 0
    four_side_training = 0
    subset_tests = 0
    dominant_pairs = rectangular_pairs = 0
    boundary_corner = four_side_mixed = boundary_mixed = 0
    test_unrepaired_side = large_four_side_test = 0
    observed_gaps = set()
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        infos = [_inspect_pair(pair["input"], pair["output"])
                 for pair in puzzle["train"] + puzzle["test"]]
        all_infos.extend(infos)
        palette_pairs.add(infos[0]["convention"])
        colors = {(info["background"], info["marker"]) for info in infos}
        varied_colors += len(colors) > 1
        train_shapes = [pair["input"].shape for pair in puzzle["train"]]
        train_counts = [info["objects"] for info in infos[:len(puzzle["train"])]]
        four_side_training += any(len(info["sides"]) == 4
                                  for info in infos[:len(puzzle["train"])])
        subset_tests += any(len(info["sides"]) < 4
                            for info in infos[len(puzzle["train"]):])
        for pair, info in zip(puzzle["train"] + puzzle["test"], infos):
            shapes.add(pair["input"].shape)
            layouts.add((pair["input"].shape,
                         tuple(sorted((r["side"], r["line"], r["axis_cells"])
                                      for r in info["records"]))))
            dominant_pairs += max(sum(r["side"] == side for r in info["records"])
                                  for side in info["sides"]) / info["objects"] >= .7
            height, width = pair["input"].shape
            rectangular_pairs += max(height, width) / min(height, width) >= 1.5
            boundary_corner += info["boundary_payload"] > 0 and info["corner_markers"] > 0
            four_side_mixed += len(info["sides"]) == 4 and info["mixed_run"] >= 3
            boundary_mixed += info["boundary_payload"] > 0 and info["mixed_run"] >= 3
            observed_gaps.update(record["gap"] for record in info["records"])
        for info in infos[len(puzzle["train"]):]:
            test_unrepaired_side += bool(info["sides"] - info["changed_sides"])
            large_four_side_test += len(info["sides"]) == 4 and info["objects"] >= 9
        smaller_tests += any(pair["input"].size <= max(h * w for h, w in train_shapes)
                             and info["objects"] <= max(train_counts)
                             for pair, info in zip(puzzle["test"], infos[len(puzzle["train"]):]))
    assert len(palette_pairs) >= 20 and len(shapes) >= 40 and len(layouts) >= 200
    assert varied_colors >= 35 and smaller_tests >= 2
    assert four_side_training >= 5 and subset_tests >= 5
    assert dominant_pairs >= 20 and rectangular_pairs >= 20
    assert boundary_corner >= 20 and four_side_mixed >= 3 and boundary_mixed >= 5
    assert test_unrepaired_side >= 5 and large_four_side_test >= 1
    assert min(observed_gaps) == 0 and max(observed_gaps) >= 15
    assert sum(info["boundary_payload"] > 0 for info in all_infos) >= 30
    assert sum(info["corner_markers"] > 0 for info in all_infos) >= 30
    assert sum(info["corner_payload"] > 0 for info in all_infos) >= 5
    assert sum(info["mixed_run"] >= 4 for info in all_infos) >= 3
    assert sum(info["longest_run"] == 1 for info in all_infos) >= 20
    assert sum(info["staggered"] for info in all_infos) >= 20
    assert sum(len(info["sides"]) == 1 for info in all_infos) >= 10
    assert sum(info["sides"] != info["changed_sides"] for info in all_infos) >= 20
    assert sum(info["changed"] == 0 for info in all_infos) >= 5
    assert sum(info["unchanged"] == 0 for info in all_infos) >= 5
