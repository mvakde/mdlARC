"""Synthetic puzzles for linking diagonally aligned orthogonal crosses."""

import numpy as np


SOURCE_INPUTS = {
    (14, 16, ((1, 3), (2, 10), (5, 7), (10, 10))),
    (14, 10, ((2, 2), (7, 7), (11, 3))),
    (12, 13, ((1, 10), (3, 6), (6, 3), (8, 10))),
    (17, 13, ((1, 2), (7, 8), (11, 4), (14, 10))),
}


def _check_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside [{low}, {high}]")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _check_count("num_train", num_train, 2, 10)
    test = _check_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _cross_cells(center):
    r, c = center
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _edges(centers):
    result = []
    for i, first in enumerate(centers):
        for second in centers[i + 1:]:
            dr = second[0] - first[0]
            dc = second[1] - first[1]
            if dr and abs(dr) == abs(dc):
                result.append((first, second, abs(dr), 1 if dr * dc > 0 else -1))
    return result


def _between(first, second):
    dr = 1 if second[0] > first[0] else -1
    dc = 1 if second[1] > first[1] else -1
    return [(first[0] + k * dr, first[1] + k * dc)
            for k in range(1, abs(second[0] - first[0]))]


def _valid_centers(centers, height, width):
    if len(set(centers)) != len(centers):
        return False
    used = set()
    for center in centers:
        r, c = center
        if not (1 <= r < height - 1 and 1 <= c < width - 1):
            return False
        cells = _cross_cells(center)
        if cells & used:
            return False
        used |= cells
    center_set = set(centers)
    for first, second, _, _ in _edges(centers):
        for cell in _between(first, second):
            if cell in used and cell not in center_set:
                return False
    return True


def _infer_crosses(grid):
    """Find the unique background and monochrome five-cell cross partition."""
    height, width = grid.shape
    solutions = []
    colors = [int(v) for v in np.unique(grid)]
    for background in colors:
        foreground = {(int(r), int(c)) for r, c in np.argwhere(grid != background)}
        if not foreground or len(foreground) % 5:
            continue
        candidates = []
        for r, c in sorted(foreground):
            if not (1 <= r < height - 1 and 1 <= c < width - 1):
                continue
            cells = _cross_cells((r, c))
            color = int(grid[r, c])
            if all(int(grid[rr, cc]) == color for rr, cc in cells):
                candidates.append((r, c))
        used = set()
        good = True
        for center in candidates:
            cells = _cross_cells(center)
            if cells & used:
                good = False
                break
            used |= cells
        if good and used == foreground:
            centers = tuple(candidates)
            if _valid_centers(centers, height, width):
                solutions.append((background, centers))
    assert len(solutions) == 1, "ambiguous or malformed crosses"
    return solutions[0]


def _paint(input_grid, connector):
    background, centers = _infer_crosses(input_grid)
    assert connector not in set(int(v) for v in np.unique(input_grid))
    output = input_grid.copy()
    for first, second, _, _ in _edges(centers):
        for r, c in _between(first, second):
            if input_grid[r, c] == background:
                output[r, c] = connector
    return output


def _sample_dimensions(rng, regime):
    if regime == "source_train":
        bound = (9, 18) if rng.random() < 0.67 else (17, 27)
    else:
        bound = (9, 18) if rng.random() < 0.42 else (17, 30)
    return (int(rng.integers(bound[0], bound[1] + 1)),
            int(rng.integers(bound[0], bound[1] + 1)))


def _sample_distance(rng, height, width, cap=18):
    maximum = min(height - 3, width - 3, cap)
    return int(rng.integers(2, maximum + 1))


def _translate(relative, height, width, rng):
    low_r = 1 - min(r for r, _ in relative)
    high_r = height - 2 - max(r for r, _ in relative)
    low_c = 1 - min(c for _, c in relative)
    high_c = width - 2 - max(c for _, c in relative)
    if low_r > high_r or low_c > high_c:
        return None
    shift_r = int(rng.integers(low_r, high_r + 1))
    shift_c = int(rng.integers(low_c, high_c + 1))
    centers = [(r + shift_r, c + shift_c) for r, c in relative]
    return centers if _valid_centers(centers, height, width) else None


def _base_layout(rng, height, width, kind):
    slope = 1 if rng.random() < 0.5 else -1
    if kind == "single":
        d = _sample_distance(rng, height, width)
        relative = [(0, 0), (d, slope * d)]
    elif kind == "zigzag":
        d1 = _sample_distance(rng, height, width, 9)
        d2 = _sample_distance(rng, height, width, 9)
        relative = [(0, 0), (d1, slope * d1),
                    (d1 + d2, slope * (d1 - d2))]
    elif kind == "straight":
        d1 = _sample_distance(rng, height, width, 8)
        d2 = _sample_distance(rng, height, width, 8)
        relative = [(0, 0), (d1, slope * d1),
                    (d1 + d2, slope * (d1 + d2))]
    elif kind == "star":
        d1 = _sample_distance(rng, height, width, 8)
        d2 = _sample_distance(rng, height, width, 8)
        d3 = _sample_distance(rng, height, width, 8)
        relative = [(0, 0), (-d1, -d1), (-d2, d2), (d3, -d3)]
    elif kind == "intersection":
        maximum = min(height - 3, width - 3, 12)
        choices = list(range(4, maximum + 1, 2))
        if not choices:
            return None
        d = int(rng.choice(choices))
        relative = [(0, 0), (d, d), (0, d), (d, 0)]
    elif kind == "independent":
        d = _sample_distance(rng, height, width, 8)
        first = _translate([(0, 0), (d, slope * d)], height, width, rng)
        if first is None:
            return None
        for _ in range(80):
            anchor = (int(rng.integers(1, height - 1)),
                      int(rng.integers(1, width - 1)))
            d2 = _sample_distance(rng, height, width, 8)
            second = (anchor[0] + d2, anchor[1] - slope * d2)
            proposal = first + [anchor, second]
            if (_valid_centers(proposal, height, width)
                    and len(_edges(proposal)) == 2):
                return proposal
        return None
    else:
        raise ValueError("unknown motif")
    return _translate(relative, height, width, rng)


def _extend_layout(rng, centers, height, width, kind):
    centers = list(centers)
    available_area = (height - 2) * (width - 2)
    limit = min(10, max(len(centers), available_area // 16 + 1))
    extra = (limit - len(centers) if rng.random() < 0.22
             else int(rng.integers(0, max(1, limit - len(centers) + 1))))
    if kind == "single" and rng.random() < 0.40:
        extra = 0
    for _ in range(extra):
        strategy = str(rng.choice(["linked", "near", "isolated"],
                                  p=[0.43, 0.27, 0.30]))
        choices = []
        if strategy == "linked":
            for anchor in centers:
                for distance in range(2, min(11, max(height, width))):
                    for dr, dc in ((distance, distance), (distance, -distance),
                                   (-distance, distance), (-distance, -distance)):
                        choices.append((anchor[0] + dr, anchor[1] + dc))
        else:
            choices = [(r, c) for r in range(1, height - 1)
                       for c in range(1, width - 1)]
        rng.shuffle(choices)
        for candidate in choices:
            if not _valid_centers(centers + [candidate], height, width):
                continue
            diagonal = any(abs(candidate[0] - r) == abs(candidate[1] - c)
                           for r, c in centers)
            near = any(abs(abs(candidate[0] - r) - abs(candidate[1] - c)) == 1
                       and max(abs(candidate[0] - r), abs(candidate[1] - c)) >= 2
                       for r, c in centers)
            if ((strategy == "linked" and diagonal)
                    or (strategy == "near" and not diagonal and near)
                    or (strategy == "isolated" and not diagonal)):
                centers.append(candidate)
                break
    return tuple(sorted(centers))


def _sample_colors(rng, params, index):
    connector = params["connector"]
    if params["vary_palette"]:
        options = [c for c in range(10) if c != connector]
        background = (0 if connector != 0 and rng.random() < 0.34
                      else int(rng.choice(options)))
        objects = [c for c in options if c != background]
        primary = int(rng.choice(objects))
        if index == 1 and (background, primary) == params["first_palette"]:
            primary = objects[(objects.index(primary) + 1) % len(objects)]
        secondary = int(rng.choice([c for c in objects if c != primary]))
    else:
        background = params["background"]
        primary = params["primary"]
        secondary = params["secondary"]
    return background, primary, secondary


def _materialize_pair(rng, regime, params, index, height, width, centers):
    edges = _edges(centers)
    if not edges:
        return None
    background, primary, secondary = _sample_colors(rng, params, index)
    colors = {center: primary for center in centers}
    mixed = params["mixed"] and (index == 0 and regime == "source_train"
                                 or rng.random() < 0.46)
    if mixed:
        for center in centers:
            colors[center] = primary if rng.random() < 0.5 else secondary
        first, second, _, _ = edges[0]
        colors[first] = primary
        colors[second] = secondary
    input_grid = np.full((height, width), background, dtype=np.int8)
    for center in centers:
        for r, c in _cross_cells(center):
            input_grid[r, c] = colors[center]
    try:
        inferred_background, inferred_centers = _infer_crosses(input_grid)
    except AssertionError:
        return None
    if inferred_background != background or set(inferred_centers) != set(centers):
        return None
    if regime == "source_train" and index == 0:
        params["first_palette"] = (background, primary)
    return {"input": input_grid, "output": _paint(input_grid, params["connector"])}


def _make_pair(rng, regime, params, index, force_two_slopes=False):
    all_kinds = ["single", "zigzag", "straight", "star",
                 "intersection", "independent"]
    for _ in range(500):
        height, width = _sample_dimensions(rng, regime)
        if force_two_slopes:
            kind = str(rng.choice(["zigzag", "star", "intersection", "independent"]))
        else:
            weights = ([0.23, 0.18, 0.16, 0.15, 0.11, 0.17]
                       if regime == "source_train"
                       else [0.17, 0.17, 0.19, 0.18, 0.12, 0.17])
            kind = str(rng.choice(all_kinds, p=weights))
        base = _base_layout(rng, height, width, kind)
        if base is None:
            continue
        centers = _extend_layout(rng, base, height, width, kind)
        edges = _edges(centers)
        if not edges:
            continue
        if force_two_slopes and {edge[3] for edge in edges} != {-1, 1}:
            continue
        pair = _materialize_pair(rng, regime, params, index,
                                 height, width, centers)
        if pair is not None:
            return pair
    # A fixed, valid core provides a deterministic exit if repeated random
    # layouts happen to be rejected. Translation and palette still vary.
    for _ in range(40):
        if force_two_slopes:
            height, width = 12, 12
            relative = [(0, 0), (3, 3), (6, 0)]
        else:
            height, width = 9, 9
            relative = [(0, 0), (2, 2)]
        centers = _translate(relative, height, width, rng)
        if centers is None:
            continue
        pair = _materialize_pair(rng, regime, params, index,
                                 height, width, tuple(sorted(centers)))
        if pair is not None:
            return pair
    raise ValueError(f"failed to construct {regime} pair")


def _array_key(array):
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    connector = int(rng.integers(0, 10))
    palette = [c for c in range(10) if c != connector]
    background = (0 if connector != 0 and rng.random() < 0.36
                  else int(rng.choice(palette)))
    object_options = [c for c in palette if c != background]
    primary, secondary = [int(v) for v in rng.choice(object_options, 2, replace=False)]
    params = {"connector": connector, "background": background,
              "primary": primary, "secondary": secondary,
              "vary_palette": bool(rng.random() < 0.62),
              "mixed": bool(rng.random() < 0.38),
              "first_palette": None}
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count, regime in (("train", train_count, "source_train"),
                                   ("test", test_count, "target_test")):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(40):
                pair = _make_pair(pair_rng, regime, params, index,
                                  force_two_slopes=section == "train" and index == 0)
                key = _array_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise ValueError("failed to construct unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    assert input_grid.shape == output_grid.shape
    background, centers = _infer_crosses(input_grid)
    edges = _edges(centers)
    assert edges and _valid_centers(centers, *input_grid.shape)
    input_colors = {int(v) for v in np.unique(input_grid)}
    output_colors = {int(v) for v in np.unique(output_grid)}
    added = output_colors - input_colors
    assert len(added) == 1
    connector = next(iter(added))
    assert output_colors == input_colors | {connector}
    assert np.array_equal(output_grid, _paint(input_grid, connector))
    assert not np.array_equal(input_grid, output_grid)
    if background == 0 and input_colors == {0, 3}:
        source_key = (*input_grid.shape, tuple(sorted(centers)))
        assert source_key not in SOURCE_INPUTS
    degree = {c: 0 for c in centers}
    for first, second, _, _ in edges:
        degree[first] += 1
        degree[second] += 1
    return {"background": background, "centers": centers, "edges": edges,
            "connector": connector, "isolated": sum(v == 0 for v in degree.values()),
            "cross_colors": {int(input_grid[c]) for c in centers},
            "unlike_link": any(input_grid[a] != input_grid[b]
                               for a, b, _, _ in edges)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _array_key(pair["input"])
            pair_key = (input_key, _array_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            info = _inspect_pair(pair)
            infos[section].append(info)
    connector = infos["train"][0]["connector"]
    assert all(info["connector"] == connector
               for info in infos["train"] + infos["test"])
    assert {edge[3] for info in infos["train"] for edge in info["edges"]} == {-1, 1}
    assert any(info["edges"] for info in infos["train"])
    if any(len(info["cross_colors"]) > 1 for info in infos["test"]):
        assert any(info["unlike_link"] for info in infos["train"])


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s])
               and all(np.array_equal(a[k], b[k])
                       for a, b in zip(first[s], second[s])
                       for k in ("input", "output"))
               for s in ("train", "test"))


def _self_test_prediction(grid, connector):
    """Trace center pairs afresh; connector is learned from demonstrations."""
    background, centers = _infer_crosses(grid)
    result = grid.copy()
    for i, (row, col) in enumerate(centers):
        for other_row, other_col in centers[i + 1:]:
            row_delta = other_row - row
            col_delta = other_col - col
            if row_delta != 0 and abs(row_delta) == abs(col_delta):
                for step in range(1, abs(row_delta)):
                    r = row + step * (1 if row_delta > 0 else -1)
                    c = col + step * (1 if col_delta > 0 else -1)
                    if grid[r, c] == background:
                        result[r, c] = connector
    return result


def self_test() -> None:
    train_counts, test_counts = set(), set()
    dimensions, palettes, layouts = set(), set(), set()
    categories, distances, backgrounds = set(), set(), set()
    coverage = {name: 0 for name in (
        "test_single", "test_independent", "test_crossing", "test_straight",
        "test_degree_three", "test_all_linked", "test_isolated",
        "test_compact", "test_close", "test_long", "test_near",
        "test_two_cross", "test_dense", "train_crossing", "train_close",
        "train_mixed", "combined", "varying_background",
        "varying_cross_colors")}
    for seed in range(20_000, 20_042):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_palettes = set()
        pair_backgrounds = set()
        pair_cross_colors = set()
        added = {int(v) for pair in puzzle["train"]
                 for v in np.unique(pair["output"][pair["input"] != pair["output"]])}
        assert len(added) == 1
        learned_connector = next(iter(added))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _inspect_pair(pair)
                assert np.array_equal(
                    pair["output"], _self_test_prediction(pair["input"], learned_connector))
                height, width = pair["input"].shape
                dimensions.add((height, width))
                backgrounds.add(info["background"])
                pair_backgrounds.add(info["background"])
                pair_cross_colors.add(tuple(sorted(info["cross_colors"])))
                pair_palettes.add((info["background"], tuple(sorted(info["cross_colors"]))))
                palettes.add((info["background"], tuple(sorted(info["cross_colors"])),
                              info["connector"]))
                layouts.add((height, width, info["centers"]))
                distances.update(e[2] for e in info["edges"])
                degree = {center: 0 for center in info["centers"]}
                paths = {}
                for first, second, _, slope in info["edges"]:
                    degree[first] += 1
                    degree[second] += 1
                    for cell in _between(first, second):
                        if pair["input"][cell] == info["background"]:
                            paths.setdefault(cell, set()).add(slope)
                near = any(
                    degree[a] == 0 and abs(abs(a[0] - b[0]) - abs(a[1] - b[1])) == 1
                    for a in info["centers"] for b in info["centers"] if a != b)
                crossing = any(len(slopes) == 2 for slopes in paths.values())
                if section == "train":
                    coverage["train_crossing"] += crossing
                    coverage["train_close"] += any(e[2] == 2 for e in info["edges"])
                    coverage["train_mixed"] += info["unlike_link"]
                if section == "test":
                    coverage["test_single"] += len(info["edges"]) == 1
                    coverage["test_independent"] += (
                        len(info["edges"]) == 2
                        and len({c for e in info["edges"] for c in e[:2]}) == 4)
                    coverage["test_crossing"] += crossing
                    coverage["test_straight"] += any(
                        cell in info["centers"]
                        for a, b, _, _ in info["edges"] for cell in _between(a, b))
                    coverage["test_degree_three"] += max(degree.values()) >= 3
                    coverage["test_all_linked"] += info["isolated"] == 0
                    coverage["test_isolated"] += info["isolated"] > 0
                    coverage["test_compact"] += height <= 16 and width <= 16
                    coverage["test_close"] += any(e[2] == 2 for e in info["edges"])
                    coverage["test_long"] += any(e[2] >= 10 for e in info["edges"])
                    coverage["test_near"] += near
                    coverage["test_two_cross"] += len(info["centers"]) == 2
                    coverage["test_dense"] += len(info["centers"]) * 5 / (height * width) >= .18
                    coverage["combined"] += near and (
                        len(info["cross_colors"]) > 1 or len(info["edges"]) >= 3)
                    categories.add((len(info["centers"]), len(info["edges"]),
                                    info["isolated"], tuple(sorted({e[3] for e in info["edges"]}))))
        coverage["varying_background"] += len(pair_backgrounds) > 1
        coverage["varying_cross_colors"] += len(pair_cross_colors) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 40 and len(layouts) >= 150
    assert len(categories) >= 14 and len(palettes) >= 30
    assert 0 in backgrounds
    assert 2 in distances and max(distances) >= 10
    assert all(coverage.values()), coverage
    a = generate_puzzle(777_777, 7, 3)
    assert _puzzles_equal(a, generate_puzzle(777_777, 7, 3))
    assert not _puzzles_equal(a, generate_puzzle(777_778, 7, 3))
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                       (None, 1), (None, 4),
                                       (2, None), (10, None))):
        puzzle = generate_puzzle(30_000 + i, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for counts in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(40_000, **counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {counts}")
