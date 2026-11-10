"""Generate centered unary-bar ARC puzzles."""

import numpy as np

_SOURCE_INPUT_ROWS = (
    ("88888888888", "88888888888", "88899888888", "88889988888",
     "88888999888", "88889898888", "88888888888"),
    ("8888888888", "8888883888", "8888333888", "8888833888",
     "8888888888"),
    ("888888888888888", "888888888822288", "888882882288288",
     "888888282228288", "888888828882288", "888888888888888",
     "888888888888888", "888888888888888", "888888888888888"),
    ("8888888888888888888888888888", "8888888858888888888888888888",
     "8888888585558888888888888888", "8888888858558888888888888888",
     "8888888588885888888888888888", "8888888888885888888888888888",
     "8888888888858888888888888888", "8888888888885888888888888888",
     "8888888888885888888888888888", "8888888888888888888888888888",
     "8888888888888888888888888888"),
)


def _grid_key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _is_original_input(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid) in _SOURCE_INPUT_ROWS


def _check_optional_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _check_optional_count("num_train", num_train, 2, 10)
    test = _check_optional_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def sample_puzzle_parameters(rng):
    colors = tuple(int(x) for x in rng.permutation(10))
    return {
        "colors": colors,
        "stable_background": bool(rng.random() < 0.35),
        "size_theme": str(rng.choice(
            ("free", "scale_up", "scale_down", "matched"),
            p=(0.45, 0.22, 0.13, 0.20),
        )),
        "train_full": bool(rng.random() < 0.55),
    }


def _dimensions(rng, regime, theme):
    if theme == "scale_up":
        band = "small" if regime == "source_train" else "large"
    elif theme == "scale_down":
        band = "large" if regime == "source_train" else "small"
    else:
        probabilities = ((0.50, 0.32, 0.18) if regime == "source_train"
                         else (0.29, 0.34, 0.37))
        band = str(rng.choice(("small", "middle", "large"), p=probabilities))
    height_ranges = {"small": (5, 11), "middle": (11, 19), "large": (19, 29)}
    width_ranges = {"small": (8, 15), "middle": (14, 23), "large": (22, 30)}
    low_h, high_h = height_ranges[band]
    low_w, high_w = width_ranges[band]
    if theme in ("free", "matched") and rng.random() < 0.32:
        other = str(rng.choice(("small", "middle", "large")))
        low_w, high_w = width_ranges[other]
    return (int(rng.choice(np.arange(low_h, high_h + 1, 2))),
            int(rng.integers(low_w, high_w + 1)))


def _choose_margin(rng, width, kind):
    maximum = (width - 5) // 2
    if kind == "full":
        return 0
    if kind == "substantial":
        return int(rng.integers(min(3, maximum), maximum + 1))
    if kind == "partial":
        return int(rng.integers(1, maximum + 1))
    if rng.random() < 0.17:
        return 0
    return int(rng.integers(1, maximum + 1))


def _shape_box(rng, height, width, count):
    options = [(r, c) for r in range(2, min(height, count) + 1)
               for c in range(2, min(width, count) + 1)
               if count < r * c <= 4 * count]
    assert options
    style = str(rng.choice(("tall", "wide", "compact", "sparse", "dense")))
    if style == "tall":
        preferred = [(r, c) for r, c in options if r >= c + 2]
    elif style == "wide":
        preferred = [(r, c) for r, c in options if c >= r + 2]
    elif style == "compact":
        preferred = [(r, c) for r, c in options if abs(r - c) <= 2]
    elif style == "sparse":
        preferred = [(r, c) for r, c in options if count / (r * c) <= 0.45]
    else:
        preferred = [(r, c) for r, c in options if count / (r * c) >= 0.68]
    choices = preferred if preferred else options
    return choices[int(rng.integers(len(choices)))]


def _shape_is_irregular(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    rows = int(np.ptp(points[:, 0])) + 1
    cols = int(np.ptp(points[:, 1])) + 1
    return rows >= 2 and cols >= 2 and int(mask.sum()) < rows * cols


def _make_connected_shape(rng, height, width, count, box=None):
    rows, cols = box if box is not None else _shape_box(rng, height, width, count)
    occupied = {(0, 0)}
    row = col = 0
    while row != rows - 1 or col != cols - 1:
        moves = []
        for dr, dc in ((1, 1), (1, 0), (0, 1)):
            nr, nc = row + dr, col + dc
            if nr >= rows or nc >= cols:
                continue
            if len(occupied) + 1 + max(rows - 1 - nr, cols - 1 - nc) <= count:
                moves.append((dr, dc))
        weights = np.array([2.0 if move == (1, 1) else 1.0 for move in moves])
        weights /= weights.sum()
        dr, dc = moves[int(rng.choice(len(moves), p=weights))]
        row += dr
        col += dc
        occupied.add((row, col))
    frontier = set()
    def add_neighbors(point):
        r, c = point
        for rr in range(max(0, r - 1), min(rows, r + 2)):
            for cc in range(max(0, c - 1), min(cols, c + 2)):
                if (rr, cc) not in occupied:
                    frontier.add((rr, cc))
    for point in occupied:
        add_neighbors(point)
    growth = str(rng.choice(("random", "branch", "dense", "mixed")))
    while len(occupied) < count:
        candidates = sorted(frontier)
        degrees = [sum((r + dr, c + dc) in occupied
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                       if dr or dc) for r, c in candidates]
        if growth in ("branch", "dense", "mixed"):
            best = (min(degrees) if growth == "branch" or
                    (growth == "mixed" and rng.random() < 0.5) else max(degrees))
            candidates = [p for p, degree in zip(candidates, degrees) if degree == best]
        point = candidates[int(rng.integers(len(candidates)))]
        occupied.add(point)
        frontier.remove(point)
        add_neighbors(point)
    local = np.zeros((rows, cols), dtype=bool)
    for r, c in occupied:
        local[r, c] = True
    if rng.random() < 0.5:
        local = local[::-1]
    if rng.random() < 0.5:
        local = local[:, ::-1]
    if rng.random() < 0.34:
        top = 0 if rng.random() < 0.5 else height - rows
    else:
        top = int(rng.integers(height - rows + 1))
    if rng.random() < 0.34:
        left = 0 if rng.random() < 0.5 else width - cols
    else:
        left = int(rng.integers(width - cols + 1))
    mask = np.zeros((height, width), dtype=bool)
    mask[top:top + rows, left:left + cols] = local
    return mask


def _choose_colors(rng, parameters, previous, regime, index):
    palette = parameters["colors"]
    if parameters["stable_background"]:
        background = palette[0]
        choices = [color for color in palette[1:]
                   if not (regime == "source_train" and index == 1
                           and color == previous[0][1])]
        if regime == "target_test" and previous and rng.random() < 0.38:
            foreground = previous[int(rng.integers(len(previous)))][1]
        else:
            foreground = choices[int(rng.integers(len(choices)))]
        return background, foreground
    if previous and rng.random() < 0.26:
        background, foreground = previous[int(rng.integers(len(previous)))][::-1]
    else:
        background, foreground = tuple(int(x) for x in rng.choice(palette, 2, replace=False))
    if regime == "source_train" and index == 1 and foreground == previous[0][1]:
        foreground = next(color for color in palette if color not in (background, previous[0][1]))
    if regime == "target_test" and previous and rng.random() < 0.28:
        foreground = previous[int(rng.integers(len(previous)))][1]
        if foreground == background:
            background = next(color for color in palette if color != foreground)
    return background, foreground


def _make_pair(rng, dimensions, margin_kind, colors, forced_margin=None, box=None):
    height, width = dimensions
    margin = _choose_margin(rng, width, margin_kind) if forced_margin is None else forced_margin
    count = width - 2 * margin
    mask = _make_connected_shape(rng, height, width, count, box)
    background, foreground = colors
    source = np.full((height, width), background, dtype=np.int8)
    source[mask] = foreground
    target = np.full((height, width), background, dtype=np.int8)
    target[height // 2, margin:margin + count] = foreground
    return {"input": source, "output": target}


def _is_connected_8(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    first = tuple(int(x) for x in points[0])
    seen, stack = {first}, [first]
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = (r + dr, c + dc)
                if ((dr or dc) and 0 <= p[0] < mask.shape[0] and
                    0 <= p[1] < mask.shape[1] and mask[p] and p not in seen):
                    seen.add(p)
                    stack.append(p)
    return len(seen) == int(mask.sum())


def _component_count_4(mask):
    remaining = {tuple(int(v) for v in point) for point in np.argwhere(mask)}
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
    return components


def _analyze_pair(pair):
    source, target = pair["input"], pair["output"]
    assert source.shape == target.shape
    height, width = source.shape
    assert height % 2 == 1
    colors, counts = np.unique(source, return_counts=True)
    assert len(colors) == 2 and counts[0] != counts[1]
    background = int(colors[int(np.argmax(counts))])
    foreground = int(colors[int(np.argmin(counts))])
    mask = source == foreground
    count = int(mask.sum())
    assert 5 <= count <= width and (width - count) % 2 == 0
    assert _is_connected_8(mask) and _shape_is_irregular(mask)
    margin = (width - count) // 2
    expected = np.full(source.shape, background, dtype=source.dtype)
    expected[height // 2, margin:margin + count] = foreground
    assert np.array_equal(target, expected)
    assert not np.array_equal(source, target)
    points = np.argwhere(mask)
    row_span = int(np.ptp(points[:, 0])) + 1
    col_span = int(np.ptp(points[:, 1])) + 1
    return {"height": height, "width": width, "background": background,
            "foreground": foreground, "count": count, "margin": margin,
            "row_span": row_span, "col_span": col_span,
            "box_area": row_span * col_span,
            "edge": bool(np.any(mask[0]) or np.any(mask[-1]) or
                         np.any(mask[:, 0]) or np.any(mask[:, -1]))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    analyzed = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert input_key not in inputs and pair_key not in pairs
            assert not _is_original_input(pair["input"])
            inputs.add(input_key)
            pairs.add(pair_key)
            analyzed[split].append(_analyze_pair(pair))
    train = analyzed["train"]
    assert len({d["count"] for d in train}) >= 2
    assert len({d["foreground"] for d in train}) >= 2
    assert any(d["margin"] > 0 for d in train)
    assert any(d["count"] != d["col_span"] for d in train)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    theme = parameters["size_theme"]
    dimensions = {split: [_dimensions(rng, split, theme) for _ in range(count)]
                  for split, count in (("source_train", train_count),
                                       ("target_test", test_count))}
    if theme == "matched":
        dimensions["target_test"][int(rng.integers(test_count))] = (
            dimensions["source_train"][int(rng.integers(train_count))])
    full_slot = int(rng.integers(train_count)) if parameters["train_full"] else -1
    partial_slot = int(rng.integers(train_count))
    if partial_slot == full_slot:
        partial_slot = (partial_slot + 1) % train_count
    later_slots = [i for i in range(2, train_count)]
    rng.shuffle(later_slots)
    repeat_slot = later_slots[0] if later_slots else -1
    similar_slot = later_slots[1] if len(later_slots) > 1 else -1
    if partial_slot == repeat_slot:
        partial_slot = next(i for i in range(train_count)
                            if i not in (full_slot, repeat_slot))
    puzzle = {"train": [], "test": []}
    colors_seen, input_keys, train_counts = [], set(), set()
    base_data = None
    for split, count in (("source_train", train_count), ("target_test", test_count)):
        for index in range(count):
            kind = ("full" if index == full_slot else
                    "substantial" if index == partial_slot else "any"
                    ) if split == "source_train" else "any"
            dims = dimensions[split][index]
            forced_margin = None
            box = None
            if split == "source_train" and index == 1:
                if kind == "full" and dims[1] == base_data["count"]:
                    width = dims[1] + 2 if dims[1] <= 28 else dims[1] - 2
                    dims = (dims[0], width)
                if kind != "full":
                    maximum = (dims[1] - 5) // 2
                    choices = [m for m in range(1, maximum + 1)
                               if dims[1] - 2 * m != base_data["count"]]
                    if not choices:
                        choices = [0]
                    forced_margin = choices[int(rng.integers(len(choices)))]
            elif split == "source_train" and index == repeat_slot:
                dims = dimensions["source_train"][0]
                forced_margin = base_data["margin"]
            elif split == "source_train" and index == similar_slot:
                dims = dimensions["source_train"][0]
                height, width = dims
                margin_choices = [m for m in range((width - 5) // 2 + 1)
                                  if m != base_data["margin"]]
                bbox_choices = []
                for margin in margin_choices:
                    count_for_box = width - 2 * margin
                    for rows in range(2, min(height, count_for_box) + 1):
                        for cols in range(2, min(width, count_for_box) + 1):
                            if count_for_box < rows * cols <= 4 * count_for_box:
                                bbox_choices.append((
                                    abs(rows - base_data["row_span"]) +
                                    abs(cols - base_data["col_span"]),
                                    abs(rows * cols - base_data["box_area"]),
                                    abs(margin - base_data["margin"]),
                                    margin, rows, cols,
                                ))
                _, _, _, forced_margin, rows, cols = min(bbox_choices)
                box = (rows, cols)
            colors = _choose_colors(rng, parameters, colors_seen, split, index)
            for _ in range(80):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair = _make_pair(np.random.default_rng(pair_seed), dims, kind,
                                  colors, forced_margin, box)
                data = _analyze_pair(pair)
                key = _grid_key(pair["input"])
                if (key in input_keys or _is_original_input(pair["input"]) or
                    (split == "source_train" and index == 1 and
                     data["count"] in train_counts) or
                    (split == "source_train" and index == repeat_slot and
                     _normalized_shape_key(pair) ==
                     _normalized_shape_key(puzzle["train"][0]))):
                    continue
                break
            else:
                raise RuntimeError("unable to construct a distinct pair")
            puzzle["train" if split == "source_train" else "test"].append(pair)
            input_keys.add(key)
            colors_seen.append(colors)
            if split == "source_train":
                train_counts.add(data["count"])
                if index == 0:
                    base_data = data
    validate_generated_puzzle(puzzle)
    return puzzle


def _normalized_shape_key(pair):
    data = _analyze_pair(pair)
    points = np.argwhere(pair["input"] == data["foreground"])
    points -= points.min(axis=0)
    return tuple(tuple(int(v) for v in point) for point in points)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    dimensions, counts, shapes, colors, margins = set(), set(), set(), set(), set()
    orientations = {"train": set(), "test": set()}
    diagonal_links = {"train": 0, "test": 0}
    edge_by_split = {"train": 0, "test": 0}
    size_relations = set()
    stable = variable = full_train = partial_only = full_test = edge = 0
    reused_test_colors = new_test_colors = swapped_roles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_analyze_pair(p) for p in puzzle["train"]]
        test = [_analyze_pair(p) for p in puzzle["test"]]
        all_data = train + test
        backgrounds = {d["background"] for d in all_data}
        stable += len(backgrounds) == 1
        variable += len(backgrounds) > 1
        full_train += any(d["margin"] == 0 for d in train)
        partial_only += all(d["margin"] > 0 for d in train)
        full_test += any(d["margin"] == 0 for d in test)
        reused_test_colors += any(d["foreground"] in {t["foreground"] for t in train}
                                  for d in test)
        new_test_colors += any(d["foreground"] not in {t["foreground"] for t in train}
                               for d in test)
        swapped_roles += any(a["background"] == b["foreground"] and
                             a["foreground"] == b["background"]
                             for a in all_data for b in all_data if a is not b)
        if len(train) >= 3:
            assert any(a["count"] == b["count"] and
                       _normalized_shape_key(puzzle["train"][i]) !=
                       _normalized_shape_key(puzzle["train"][j])
                       for i, a in enumerate(train) for j, b in enumerate(train)
                       if i < j)
        if len(train) >= 4:
            assert any(a["count"] != b["count"] and
                       abs(a["row_span"] - b["row_span"]) +
                       abs(a["col_span"] - b["col_span"]) <= 2
                       for i, a in enumerate(train) for j, b in enumerate(train)
                       if i < j)
        for d in all_data:
            dimensions.add((d["height"], d["width"]))
            counts.add(d["count"])
            colors.add((d["background"], d["foreground"]))
            margins.add(d["margin"])
            edge += d["edge"]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _analyze_pair(pair)
                shapes.add(_normalized_shape_key(pair))
                mask = pair["input"] == d["foreground"]
                diagonal_links[split] += _component_count_4(mask) > 1
                edge_by_split[split] += d["edge"]
                orientations[split].add("wide" if d["col_span"] > d["row_span"] else
                                        "tall" if d["row_span"] > d["col_span"] else "square")
        for d in test:
            train_areas = [t["height"] * t["width"] for t in train]
            area = d["height"] * d["width"]
            size_relations.add("larger" if area > max(train_areas) else
                               "smaller" if area < min(train_areas) else "overlap")
            if (d["height"], d["width"]) in {(t["height"], t["width"]) for t in train}:
                size_relations.add("matched")
    assert stable >= 5 and variable >= 5
    assert full_train >= 5 and partial_only >= 5 and full_test >= 3
    assert reused_test_colors >= 5 and new_test_colors >= 5 and swapped_roles >= 3
    assert edge >= 30 and len(dimensions) >= 40 and len(counts) >= 15
    assert all(diagonal_links[split] >= 5 and edge_by_split[split] >= 5
               for split in ("train", "test"))
    assert len(colors) >= 35 and len(margins) >= 8 and len(shapes) >= 150
    assert orientations["train"] == orientations["test"] == {"wide", "tall", "square"}
    assert size_relations == {"larger", "smaller", "overlap", "matched"}
    explicit = generate_puzzle(20260827, 7, 3)
    validate_generated_puzzle(explicit)
    assert len(explicit["train"]) == 7 and len(explicit["test"]) == 3
    again = generate_puzzle(20260827, 7, 3)
    assert all(_pair_key(a) == _pair_key(b)
               for split in ("train", "test")
               for a, b in zip(explicit[split], again[split]))
    assert len({_pair_key(generate_puzzle(seed, 2, 1)["test"][0])
                for seed in range(100, 105)}) >= 4
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1000 + 10 * train_count + test_count,
                                 train_count, test_count)
        assert len(puzzle["train"]) == train_count and len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for offset, (train_count, test_count) in enumerate(((None, 1), (None, 4),
                                                        (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + offset, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(3000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
