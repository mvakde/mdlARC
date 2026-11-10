import hashlib

import numpy as np


_SOURCE_HASHES = {
    "390681b501bccbeba90812a286c50d7793b67e6fdff878aa297d92c2a9e5133e",
    "685aa2c87684b72de6a8ecf7dadafdaf781b7d223484090dd596cb44a0f00d21",
    "0a230d67732a400b30ee428e57cf4b29ca8a27f0284e0127e3a62f179dda7028",
    "e38e4bae78cab3f154e8d3bb8161d8dcd796c6df35a462695b9538cee107082e",
}


def _counts(seed, rng, num_train, num_test):
    for name, value, lower, upper in (("num_train", num_train, 2, 10),
                                       ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lower <= value <= upper):
            raise ValueError(f"{name} is outside its legal integer range")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _geometry(rng, kind, rows, cols):
    if kind == "source":
        height, width = ((3, 4) if rng.integers(2) else (4, 3))
        return height, width, (height - 2, width - 2, 1, 1)
    if kind == "tiny":
        return 3, 3, (1, 1, 1, 1)
    height = int(rng.integers(4, min(9, rows - 3) + 1))
    width = int(rng.integers(4, min(9, cols - 3) + 1))
    if kind == "thick":
        hole_height = int(rng.integers(1, min(2, height - 2) + 1))
        hole_width = int(rng.integers(1, min(2, width - 2) + 1))
        if hole_height * hole_width == 1:
            hole_width = 2
    elif kind == "thin":
        hole_height, hole_width = height - 2, width - 2
    else:
        hole_height = int(rng.integers(1, height - 1))
        hole_width = int(rng.integers(1, width - 1))
        if hole_height * hole_width == 1:
            hole_width = 2
    offset_row = int(rng.integers(1, height - hole_height))
    offset_col = int(rng.integers(1, width - hole_width))
    return height, width, (hole_height, hole_width, offset_row, offset_col)


def _touches(first, second):
    row_a, col_a, height_a, width_a = first
    row_b, col_b, height_b, width_b = second
    row_overlap = max(row_a, row_b) < min(row_a + height_a, row_b + height_b)
    col_overlap = max(col_a, col_b) < min(col_a + width_a, col_b + width_b)
    return ((row_overlap and (col_a + width_a == col_b or col_b + width_b == col_a))
            or (col_overlap and (row_a + height_a == row_b or row_b + height_b == row_a)))


def _positions(rows, cols, specs, rng, contact, contact_target):
    boxes = [None] * len(specs)
    edge_index = int(rng.integers(len(specs))) if rng.random() < .42 else None
    first = 0 if contact_target else 1
    second = 1 if contact_target else 2
    if contact:
        order = [first, second] + sorted(
            (index for index in range(len(specs)) if index not in (first, second)),
            key=lambda index: -specs[index][0] * specs[index][1],
        )
    else:
        order = sorted(range(len(specs)), key=lambda index: -specs[index][0] * specs[index][1])
    for index in order:
        height, width, color = specs[index]
        candidates = []
        preferred = []
        edge_object = index == edge_index
        row_range = range(rows - height + 1) if edge_object else range(1, rows - height)
        col_range = range(cols - width + 1) if edge_object else range(1, cols - width)
        for row in row_range:
            for col in col_range:
                if edge_object and sum((row == 0, col == 0,
                                        row + height == rows, col + width == cols)) != 1:
                    continue
                box = (row, col, height, width)
                if any(not (row + height <= old[0] or old[0] + old[2] <= row
                            or col + width <= old[1] or old[1] + old[3] <= col)
                       for old in boxes if old is not None):
                    continue
                touching = [old_index for old_index, old in enumerate(boxes)
                            if old is not None and _touches(box, old)]
                if any(specs[old_index][2] == color for old_index in touching):
                    continue
                if not contact and touching:
                    continue
                if contact and index == second and first not in touching:
                    continue
                candidates.append(box)
                if contact and index == second:
                    anchor = boxes[first]
                    if row != anchor[0] and col != anchor[1]:
                        preferred.append(box)
                elif touching:
                    preferred.append(box)
        if not candidates:
            return None
        choices = preferred if preferred and (index == second or rng.random() < .45) else candidates
        boxes[index] = choices[int(rng.integers(len(choices)))]
    return boxes


def _make_pair(rng, params, split, index):
    is_test = split == "test"
    tiny_demo = params["tiny_demo"]
    prefer_compact = rng.random() < (.65 if is_test else .55)
    for attempt in range(100):
        compact = prefer_compact and attempt < 80
        rows = int(rng.integers(13, 20) if compact else rng.integers(20, 31))
        cols = int(rng.integers(13, 20) if compact else rng.integers(20, 31))
        count = int(rng.integers(4, 7) if compact else rng.integers(4, 9))
        if index == 0 and not is_test:
            kind = "tiny" if tiny_demo else "source"
        elif index == 1 and not is_test:
            kind = "source" if tiny_demo else "thin"
        else:
            kinds = ("source", "general", "general", "thin", "thick")
            if tiny_demo:
                kinds += ("tiny",)
            kind = str(rng.choice(kinds))
        if is_test and index == 0 and tiny_demo and params["tiny_test"]:
            kind = "tiny"
        target_height, target_width, hole = _geometry(rng, kind, rows, cols)
        target_color = (params["reversal_color"] if is_test and params["role_mode"] == "reversal"
                        else params["train_colors"][index % len(params["train_colors"])])
        background = params["backgrounds"][(index + (1 if is_test else 0)) % len(params["backgrounds"])]
        same_foil = rng.random() < .38 and count >= 4
        largest = rng.random() < (.30 if is_test else .20) and not same_foil
        small = rng.random() < (.28 if is_test else .15) and not same_foil and not largest
        sizes = []
        for solid_index in range(count - 1):
            if solid_index == 0 and same_foil:
                size = (target_height, target_width)
            else:
                maximum = 7 if compact else 8
                size = (int(rng.integers(2, maximum + 1)),
                        int(rng.integers(2, maximum + 1)))
                if largest:
                    for _ in range(20):
                        if size[0] * size[1] < target_height * target_width:
                            break
                        size = (int(rng.integers(2, maximum + 1)),
                                int(rng.integers(2, maximum + 1)))
                    if size[0] * size[1] >= target_height * target_width:
                        size = (2, 2)
                if small and solid_index == 0:
                    size = (min(8, max(4, target_height + 2)),
                            min(8, max(4, target_width + 2)))
            sizes.append(size)
        palette = [color for color in params["palette"] if color not in (background, target_color)]
        rng.shuffle(palette)
        if not is_test and index == 0 and params["role_mode"] == "reversal":
            palette.remove(params["reversal_color"])
            palette.insert(0, params["reversal_color"])
        colors = palette[:count - 1]
        if count >= 5 and rng.random() < .30:
            colors[-1] = colors[1]
        specs = [(target_height, target_width, target_color)] + [
            (height, width, color) for (height, width), color in zip(sizes, colors)
        ]
        contact = rng.random() < (.66 if is_test else .38)
        contact_target = contact and rng.random() < .26
        boxes = _positions(rows, cols, specs, rng, contact, contact_target)
        if boxes is None:
            continue
        area = sum(height * width for height, width, _ in specs) - hole[0] * hole[1]
        if area >= rows * cols * .60 or (compact and area < rows * cols * .31):
            continue
        grid = np.full((rows, cols), background, dtype=np.int8)
        for (height, width, color), (row, col, _, _) in zip(specs, boxes):
            grid[row:row + height, col:col + width] = color
        target_row, target_col = boxes[0][:2]
        hole_height, hole_width, offset_row, offset_col = hole
        grid[target_row + offset_row:target_row + offset_row + hole_height,
             target_col + offset_col:target_col + offset_col + hole_width] = background
        return {"input": grid, "output": np.array([[target_color]], dtype=np.int8)}
    raise ValueError("could not place a valid layout")


def _components(grid, color):
    remaining = set(zip(*np.where(grid == color)))
    result = []
    while remaining:
        start = remaining.pop()
        component = {start}
        frontier = [start]
        while frontier:
            row, col = frontier.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        result.append(component)
    return result


def _analyze(grid):
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    border_colors, border_counts = np.unique(border, return_counts=True)
    background = int(border_colors[np.argmax(border_counts)])
    objects = []
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        for component in _components(grid, color):
            top = min(row for row, _ in component)
            bottom = max(row for row, _ in component)
            left = min(col for _, col in component)
            right = max(col for _, col in component)
            box = grid[top:bottom + 1, left:right + 1]
            assert np.all((box == color) | (box == background))
            missing = np.argwhere(box == background)
            hole = None
            if len(missing):
                assert np.all(box[0] == color) and np.all(box[-1] == color)
                assert np.all(box[:, 0] == color) and np.all(box[:, -1] == color)
                hole_top, hole_left = missing.min(axis=0)
                hole_bottom, hole_right = missing.max(axis=0)
                assert hole_top > 0 and hole_left > 0
                assert hole_bottom < box.shape[0] - 1 and hole_right < box.shape[1] - 1
                assert len(missing) == (hole_bottom - hole_top + 1) * (hole_right - hole_left + 1)
                hole = (int(hole_bottom - hole_top + 1), int(hole_right - hole_left + 1),
                        int(hole_top), int(hole_left))
            assert len(component) == box.size - len(missing)
            objects.append({"color": color, "box": (top, left, bottom + 1, right + 1),
                            "shape": box.shape, "hole": hole, "area": box.size})
    assert 3 <= len(objects) <= 8
    targets = [obj for obj in objects if obj["hole"] is not None]
    assert len(targets) == 1
    for index, first in enumerate(objects):
        for second in objects[index + 1:]:
            top_a, left_a, bottom_a, right_a = first["box"]
            top_b, left_b, bottom_b, right_b = second["box"]
            assert bottom_a <= top_b or bottom_b <= top_a or right_a <= left_b or right_b <= left_a
    return {"background": background, "objects": objects, "target": targets[0]}


def _source_hash(grid):
    rows = ["".join(str(int(value)) for value in row) for row in grid]
    return hashlib.sha256("|".join(rows).encode()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_analyses = []
    test_analyses = []
    for split, analyses in (("train", train_analyses), ("test", test_analyses)):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            grid, output = pair["input"], pair["output"]
            assert output.shape == (1, 1) and grid.shape != output.shape
            signature = (grid.shape, grid.tobytes())
            assert signature not in seen
            seen.add(signature)
            assert _source_hash(grid) not in _SOURCE_HASHES
            analysis = _analyze(grid)
            assert all(int(grid[row, col]) == analysis["background"]
                       for row in (0, grid.shape[0] - 1)
                       for col in (0, grid.shape[1] - 1))
            assert int(output[0, 0]) == analysis["target"]["color"]
            assert analysis["background"] != analysis["target"]["color"]
            analyses.append(analysis)
    assert len({a["target"]["color"] for a in train_analyses}) >= 2
    train_target_colors = {a["target"]["color"] for a in train_analyses}
    train_solid_colors = {obj["color"] for a in train_analyses
                          for obj in a["objects"] if obj["hole"] is None}
    assert all(a["target"]["color"] in train_target_colors | train_solid_colors
               for a in test_analyses)
    assert any(a["target"]["hole"][:2] != b["target"]["hole"][:2]
               for a in train_analyses for b in train_analyses)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    palette = [int(color) for color in rng.permutation(10)]
    train_colors = palette[:min(4, train_count)]
    params = {"palette": palette, "train_colors": train_colors,
              "reversal_color": palette[len(train_colors)],
              "backgrounds": palette[len(train_colors) + 1:],
              "role_mode": "reversal" if rng.random() < .52 else "recurrence",
              "tiny_demo": bool(rng.random() < .42),
              "tiny_test": bool(rng.random() < .45)}
    train = []
    test = []
    for split, count, pairs in (("train", train_count, train), ("test", test_count, test)):
        for index in range(count):
            pair_rng = np.random.default_rng(rng.integers(0, 2**63, dtype=np.int64))
            pairs.append(_make_pair(pair_rng, params, split, index))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((pair["input"].shape, pair["input"].tobytes(),
                  pair["output"].tobytes())
                 for split in ("train", "test") for pair in puzzle[split])


def _predict_from_examples(train_pairs, test_input):
    examples = [_analyze(pair["input"].copy()) for pair in train_pairs]
    assert len(examples) >= 2
    assert len({example["target"]["color"] for example in examples}) >= 2
    for pair, example in zip(train_pairs, examples):
        assert int(pair["output"][0, 0]) == example["target"]["color"]
    return np.array([[_analyze(test_input.copy())["target"]["color"]]], dtype=np.int8)


def _normalized_structure(grid, background):
    rows, cols = np.where(grid != background)
    crop = grid[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    color_map = {background: 0}
    values = []
    for value in crop.flat:
        value = int(value)
        if value not in color_map:
            color_map[value] = len(color_map)
        values.append(color_map[value])
    return crop.shape, tuple(values)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert len({_signature(puzzle) for puzzle in puzzles}) == 42
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3),
                                    (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(999 + (train_count or 0) + (test_count or 0),
                                 train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(123, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _signature(generate_puzzle(321, 8, 3)) == _signature(generate_puzzle(321, 8, 3))
    assert _signature(generate_puzzle(321, 8, 3)) != _signature(generate_puzzle(322, 8, 3))
    analyses = [(split, _analyze(pair["input"])) for puzzle in puzzles
                for split in ("train", "test") for pair in puzzle[split]]
    assert len({a["target"]["shape"] for _, a in analyses}) >= 12
    assert len({a["target"]["hole"] for _, a in analyses}) >= 20
    assert len({a["background"] for _, a in analyses}) >= 5
    assert len({len(a["objects"]) for _, a in analyses}) >= 5
    assert any(a["target"]["shape"] == (3, 4) for _, a in analyses)
    assert any(a["target"]["shape"] == (4, 3) for _, a in analyses)
    assert any(a["target"]["hole"][:2] == (1, 1) for split, a in analyses if split == "train")
    assert any(any(obj["shape"] == a["target"]["shape"] and obj["hole"] is None
                   for obj in a["objects"]) for _, a in analyses)
    measures = {}
    structures = set()
    total_pairs = 0
    contact_geometries = set()
    for puzzle in puzzles:
        train = [{"input": pair["input"].copy(), "output": pair["output"].copy()}
                 for pair in puzzle["train"]]
        for pair in puzzle["test"]:
            assert np.array_equal(_predict_from_examples(train, pair["input"].copy()),
                                  pair["output"])
        pair_analyses = [_analyze(pair["input"]) for split in ("train", "test")
                         for pair in puzzle[split]]
        for property_name, values in (
            ("canvas", [pair["input"].shape for split in ("train", "test")
                        for pair in puzzle[split]]),
            ("background", [analysis["background"] for analysis in pair_analyses]),
            ("target_shape", [analysis["target"]["shape"] for analysis in pair_analyses]),
            ("object_count", [len(analysis["objects"]) for analysis in pair_analyses]),
        ):
            measures["within_" + property_name] = measures.get("within_" + property_name, 0) + (len(set(values)) > 1)
        train_targets = {_analyze(pair["input"])["target"]["color"] for pair in puzzle["train"]}
        train_solids = {obj["color"] for pair in puzzle["train"]
                        for obj in _analyze(pair["input"])["objects"] if obj["hole"] is None}
        test_targets = {_analyze(pair["input"])["target"]["color"] for pair in puzzle["test"]}
        for name, condition in (("reversal", test_targets <= train_solids - train_targets),
                                ("recurrence", test_targets <= train_targets)):
            measures[name] = measures.get(name, 0) + condition
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total_pairs += 1
                grid = pair["input"]
                analysis = _analyze(grid)
                objects = analysis["objects"]
                target = analysis["target"]
                solids = [obj for obj in objects if obj is not target]
                hole_height, hole_width, _, _ = target["hole"]
                occupancy = np.count_nonzero(grid != analysis["background"]) / grid.size
                structures.add(_normalized_structure(grid, analysis["background"]))
                contacts = []
                for first_index, first in enumerate(objects):
                    for second in objects[first_index + 1:]:
                        top_a, left_a, bottom_a, right_a = first["box"]
                        top_b, left_b, bottom_b, right_b = second["box"]
                        if _touches((top_a, left_a, bottom_a - top_a, right_a - left_a),
                                    (top_b, left_b, bottom_b - top_b, right_b - left_b)):
                            contacts.append((first, second))
                            if right_a == left_b or right_b == left_a:
                                contact_geometries.add(("horizontal", abs(top_a - top_b),
                                                        min(bottom_a, bottom_b) - max(top_a, top_b)))
                            else:
                                contact_geometries.add(("vertical", abs(left_a - left_b),
                                                        min(right_a, right_b) - max(left_a, left_b)))
                features = {
                    "compact": max(grid.shape) <= 19,
                    "crowded": .33 <= occupancy <= .57,
                    "edge_touch": any(top == 0 or left == 0 or bottom == grid.shape[0]
                                      or right == grid.shape[1]
                                      for top, left, bottom, right in
                                      (obj["box"] for obj in objects)),
                    "source_shape": target["shape"] in ((3, 4), (4, 3))
                                    and (hole_height, hole_width) in ((1, 2), (2, 1)),
                    "tiny_hole": (hole_height, hole_width) == (1, 1),
                    "same_size_foil": any(obj["shape"] == target["shape"] for obj in solids),
                    "strict_largest": all(target["area"] > obj["area"] for obj in solids),
                    "smaller_than_solid": any(target["area"] < obj["area"] for obj in solids),
                    "thick_small_hole": min(target["shape"]) >= 6 and hole_height * hole_width <= 4,
                    "thin_broad_hole": hole_height == target["shape"][0] - 2
                                       and hole_width == target["shape"][1] - 2
                                       and hole_height * hole_width >= 9,
                    "contact": bool(contacts),
                    "separated": not contacts,
                    "target_contact": any(target in contact for contact in contacts),
                    "reused_color": len({obj["color"] for obj in objects}) < len(objects),
                    "nonzero_background": analysis["background"] != 0,
                }
                for name, condition in features.items():
                    key = split + "_" + name
                    measures[key] = measures.get(key, 0) + condition
                combination = (features["compact"], features["contact"],
                               features["same_size_foil"])
                measures["combination_" + str(combination)] = measures.get(
                    "combination_" + str(combination), 0) + 1
    assert len(structures) >= total_pairs * .95
    assert len(contact_geometries) >= 25
    assert all(measures["within_" + name] >= 35 for name in
               ("canvas", "background", "target_shape", "object_count"))
    assert measures["reversal"] >= 8 and measures["recurrence"] >= 8
    assert all(measures[split + "_" + name] >= 3
               for split in ("train", "test")
               for name in ("compact", "crowded", "source_shape", "tiny_hole",
                            "same_size_foil", "strict_largest", "smaller_than_solid",
                            "thick_small_hole", "thin_broad_hole", "contact",
                            "separated", "target_contact", "reused_color",
                            "nonzero_background", "edge_touch"))
    assert len([key for key in measures if key.startswith("combination_")]) == 8
