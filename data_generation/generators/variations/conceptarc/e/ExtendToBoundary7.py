"""Generate grids where each marker extends to its aligned separator cell."""

import hashlib

import numpy as np


_SOURCE_PAIRS = {
    "81c8926e9c6aeb21dadf8840541c78af8c5135c976aba930f7ca8a04bcb69b92",
    "f4d3930b67b6fb28191770f9214b83893bdb5a4dac2890079e4b2940dd2a2dff",
    "59d8b0a9974c2e6affdde899735dbb314b2946ed015d3e75e43e58120f5caffd",
    "319c2ae45d25ed6d8ca9034ffed6fe3edb2329dbf3020c882b702bedc9536d45",
    "a71c7f5b2bc46b6d5d09b637aba6c87c10e9840a77c9c740da30573c7ecf26aa",
}


def _counts(rng, num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"),
                                (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train-1)+1))
        return num_train, num_test
    choices = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _dimensions(rng, orientation):
    if orientation.startswith("diag"):
        transverse = int(rng.integers(6, 19))
        axis = int(rng.integers(transverse + 4, 31))
        return ((transverse, axis) if orientation == "diag_rows"
                else (axis, transverse))
    aspect = int(rng.integers(3))
    if aspect == 0:
        height = int(rng.integers(6, 31))
        width = int(np.clip(height + rng.integers(-2, 3), 6, 30))
    else:
        long_side = int(rng.integers(10, 31))
        short_side = int(rng.integers(6, long_side - 2))
        height, width = ((long_side, short_side) if aspect == 1
                         else (short_side, long_side))
    return height, width


def _geometry(rng, orientation, height, width):
    horizontal_fill = orientation in ("vertical", "diag_rows")
    transverse = height if horizontal_fill else width
    axis = width if horizontal_fill else height
    if orientation.startswith("diag"):
        start = int(rng.integers(2, axis - transverse - 1))
        slope = int(rng.choice((-1, 1)))
        return [start + (i if slope == 1 else transverse - 1 - i)
                for i in range(transverse)]
    boundary_class = int(rng.choice(3, p=(0.22, 0.28, 0.50)))
    if boundary_class == 0:
        options = [0, axis - 1]
    elif boundary_class == 1:
        options = sorted({1, 2, axis - 3, axis - 2})
    else:
        options = list(range(1, axis - 1))
    return [int(rng.choice(options))] * transverse


def _cell(orientation, line, position):
    return ((line, position) if orientation in ("vertical", "diag_rows")
            else (position, line))


def _compatible(candidate, chosen, orientation):
    r, c = _cell(orientation, candidate[0], candidate[1])
    for other in chosen:
        if candidate[0] == other[0] and candidate[2] == other[2]:
            return False
        rr, cc = _cell(orientation, other[0], other[1])
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _choose_markers(rng, positions, axis, orientation, count,
                    want_adjacent, want_same_line):
    candidates = [(line, pos, -1 if pos < boundary else 1,
                   abs(pos - boundary))
                  for line, boundary in enumerate(positions)
                  for pos in range(axis) if pos != boundary]
    distant = [x for x in candidates if x[3] >= 2]
    if want_same_line:
        distant = [x for x in distant if any(y[0] == x[0]
                   and y[2] != x[2] for y in candidates)]
    for _ in range(48):
        chosen = [distant[int(rng.integers(len(distant)))]]
        if want_same_line:
            pool = [x for x in candidates if x[0] == chosen[0][0]
                    and x[2] != chosen[0][2]
                    and (not want_adjacent or x[3] == 1)]
            if not pool:
                continue
            chosen.append(pool[int(rng.integers(len(pool)))])
        elif want_adjacent:
            pool = [x for x in candidates if x[3] == 1
                    and _compatible(x, chosen, orientation)]
            if not pool:
                continue
            chosen.append(pool[int(rng.integers(len(pool)))])
        while len(chosen) < count:
            pool = [x for x in candidates if _compatible(x, chosen, orientation)]
            if not pool:
                break
            chosen.append(pool[int(rng.integers(len(pool)))])
        if len(chosen) == count:
            return chosen
    raise AssertionError("could not place isolated markers")


def _make_pair(rng, orientation, background, separator, marker_colors,
               count, adjacent=False, same_line=False):
    for _ in range(40):
        height, width = _dimensions(rng, orientation)
        positions = _geometry(rng, orientation, height, width)
        axis = width if orientation in ("vertical", "diag_rows") else height
        try:
            markers = _choose_markers(rng, positions, axis, orientation,
                                      count, adjacent, same_line)
        except AssertionError:
            continue
        inp = np.full((height, width), background, dtype=np.int8)
        for line, boundary in enumerate(positions):
            inp[_cell(orientation, line, boundary)] = separator
        colors = [int(marker_colors[0])] * count
        if len(marker_colors) > 1:
            colors[1] = int(marker_colors[1])
            for index in range(2, count):
                colors[index] = int(rng.choice(marker_colors))
        for item, color in zip(markers, colors):
            inp[_cell(orientation, item[0], item[1])] = color
        out = inp.copy()
        for (line, pos, _, _), color in zip(markers, colors):
            boundary = positions[line]
            for coordinate in range(min(pos, boundary) + 1,
                                    max(pos, boundary)):
                out[_cell(orientation, line, coordinate)] = color
        return inp, out
    raise AssertionError("could not construct a pair")


def _separator_candidates(inp, background):
    height, width = inp.shape
    results = []
    for color in (int(x) for x in np.unique(inp) if x != background):
        points = np.argwhere(inp == color)
        if len(points) == height and np.array_equal(points[:, 0], np.arange(height)):
            columns = points[:, 1]
            if np.all(columns == columns[0]):
                results.append((color, "vertical", columns.tolist()))
            elif np.all(np.diff(columns) == 1) or np.all(np.diff(columns) == -1):
                results.append((color, "diag_rows", columns.tolist()))
        by_column = points[np.argsort(points[:, 1])]
        if len(points) == width and np.array_equal(by_column[:, 1], np.arange(width)):
            rows = by_column[:, 0]
            if np.all(rows == rows[0]):
                results.append((color, "horizontal", rows.tolist()))
            elif np.all(np.diff(rows) == 1) or np.all(np.diff(rows) == -1):
                results.append((color, "diag_cols", rows.tolist()))
    return results


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.all((grid >= 0) & (grid <= 9))
    assert inp.shape == out.shape
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(counts)])
    assert len(colors) >= 3 and int(max(counts)) > int(sorted(counts)[-2])
    separators = _separator_candidates(inp, background)
    assert len(separators) == 1
    separator, orientation, positions = separators[0]
    height, width = inp.shape
    if orientation.startswith("diag"):
        assert (width >= height + 4 if orientation == "diag_rows"
                else height >= width + 4)
    marker_mask = (inp != background) & (inp != separator)
    points = np.argwhere(marker_mask)
    assert 1 <= len(points) <= 5
    for i, (r, c) in enumerate(points):
        for rr, cc in points[i + 1:]:
            assert max(abs(int(r - rr)), abs(int(c - cc))) > 1
    horizontal_fill = orientation in ("vertical", "diag_rows")
    needed = np.zeros(inp.shape, dtype=bool)
    same_line_side = set()
    sides, distances, lines = [], [], []
    for r0, c0 in points:
        r, c = int(r0), int(c0)
        line, pos = (r, c) if horizontal_fill else (c, r)
        boundary = positions[line]
        assert pos != boundary
        side = -1 if pos < boundary else 1
        assert (line, side) not in same_line_side
        same_line_side.add((line, side))
        sides.append(side)
        distances.append(abs(pos - boundary))
        lines.append(line)
        for coordinate in range(min(pos, boundary) + 1, max(pos, boundary)):
            rr, cc = _cell(orientation, line, coordinate)
            assert inp[rr, cc] == background and not needed[rr, cc]
            needed[rr, cc] = True
            assert out[rr, cc] == inp[r, c]
    assert np.any(needed)
    assert np.array_equal(out != inp, needed)
    assert np.array_equal(out[~needed], inp[~needed])
    digest = hashlib.sha256(bytes(inp.shape) + inp.astype(np.int8).tobytes()
                            + out.astype(np.int8).tobytes()).hexdigest()
    assert digest not in _SOURCE_PAIRS
    return {"background": background, "separator": separator,
            "orientation": orientation, "positions": positions,
            "markers": len(points), "marker_colors": frozenset(int(inp[r, c])
            for r, c in points), "sides": tuple(sides),
            "distances": tuple(distances), "lines": tuple(lines),
            "marker_positions": tuple((int(r), int(c)) for r, c in points),
            "shape": inp.shape}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for group in ("train", "test"):
        for pair in puzzle[group]:
            info = _inspect_pair(pair)
            infos[group].append(info)
            inp, out = pair["input"], pair["output"]
            input_key = (inp.shape, inp.astype(np.int8).tobytes())
            pair_key = (input_key, out.astype(np.int8).tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    axes = {info["orientation"] in ("vertical", "diag_rows")
            for info in infos["train"]}
    assert axes == {False, True}
    train_palettes = {(x["background"], x["separator"])
                      for x in infos["train"]}
    if any((x["background"], x["separator"]) not in train_palettes
           for x in infos["test"]):
        assert len(train_palettes) >= 2
    if any(len(x["marker_colors"]) > 1 for x in infos["test"]):
        assert any(len(x["marker_colors"]) > 1 for x in infos["train"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    palette = [int(x) for x in rng.permutation(10)]
    marker_palette, role_palette = palette[:4], palette[4:]
    varying_roles = bool(rng.integers(0, 3) == 0)
    multicolor = bool(rng.integers(0, 3) == 0)
    stable_roles = tuple(role_palette[:2])
    orientations = ["vertical", "horizontal"]
    if rng.integers(4) == 0:
        orientations = (["vertical", "diag_cols"] if rng.integers(2) == 0
                        else ["horizontal", "diag_rows"])
    rng.shuffle(orientations)
    puzzle = {"train": [], "test": []}
    seen, role_history = set(), []
    for group, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if group == "train" and index < 2:
                orientation = orientations[index]
            else:
                orientation = str(rng.choice(
                    ["vertical", "horizontal", "diag_rows", "diag_cols"],
                    p=[0.29, 0.29, 0.21, 0.21]))
            global_index = len(puzzle["train"]) + len(puzzle["test"])
            if varying_roles:
                options = [(a, b) for a in role_palette for b in role_palette
                           if a != b and (not (group == "train" and index == 1)
                           or (a, b) != role_history[0])]
                background, separator = options[int(rng.integers(len(options)))]
            else:
                background, separator = stable_roles
            role_history.append((background, separator))
            primary = marker_palette[0 if global_index < 2 else
                                     (2 if multicolor else 1) if global_index == 2 else
                                     int(rng.integers(4))]
            marker_count = int(rng.integers(2, 5) if group == "train"
                               else rng.integers(1, 6))
            use_multiple = multicolor and ((group == "train" and index == 0)
                                           or bool(rng.integers(3) == 0))
            if use_multiple:
                marker_count = max(marker_count, 2)
                secondary = next(c for c in marker_palette if c != primary)
                marker_colors = [primary, secondary]
            else:
                marker_colors = [primary]
            adjacent = marker_count >= 2 and bool(rng.integers(3) == 0)
            same_line = (orientation.startswith("diag") and marker_count >= 2
                         and bool(rng.integers(4) == 0))
            for _ in range(32):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                inp, out = _make_pair(pair_rng, orientation, background,
                                      separator, marker_colors, marker_count,
                                      adjacent, same_line)
                key = (inp.shape, inp.tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[group].append({"input": inp, "output": out})
                    break
            else:
                raise AssertionError("could not construct unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((group, tuple((p["input"].shape, p["input"].tobytes(),
                                p["output"].tobytes()) for p in puzzle[group]))
                 for group in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    features = {"orientations": set(), "shapes": set(), "layouts": set(),
                "marker_counts": set(), "aspects": set(), "colors": set()}
    counts = {name: 0 for name in (
        "edge", "near_edge", "near_center", "adjacent", "multi",
        "same_line", "same_side_interior", "consecutive_lines",
        "diagonal_within_span", "positive_diagonal", "negative_diagonal",
        "portrait_vertical", "landscape_horizontal", "square_straight",
        "source_shape", "diagonal_in_train", "edge_in_train",
        "adjacent_in_train", "straight_in_test", "ordinary_in_test",
        "varying_roles", "varying_counts", "reused_and_new_colors")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        all_pairs = puzzle["train"] + puzzle["test"]
        info = [_inspect_pair(pair) for pair in all_pairs]
        if len({(x["background"], x["separator"]) for x in info}) > 1:
            counts["varying_roles"] += 1
        counts["varying_counts"] += len({x["markers"] for x in info}) > 1
        seen_marker_colors = set()
        reused = novel = False
        for index, x in enumerate(info):
            h, w = x["shape"]
            if seen_marker_colors:
                reused |= bool(x["marker_colors"] & seen_marker_colors)
                novel |= bool(x["marker_colors"] - seen_marker_colors)
            seen_marker_colors.update(x["marker_colors"])
            features["orientations"].add(x["orientation"])
            features["shapes"].add(x["shape"])
            features["layouts"].add((x["shape"], x["orientation"],
                                     tuple(x["positions"]), x["lines"],
                                     x["sides"], x["distances"]))
            features["marker_counts"].add(x["markers"])
            features["aspects"].add("square" if abs(h-w) <= 2 else
                                    "portrait" if h > w else "landscape")
            features["colors"].add((x["background"], x["separator"]))
            boundary_axis = w if x["orientation"] == "vertical" else h
            counts["edge"] += (x["orientation"] in ("vertical", "horizontal")
                               and x["positions"][0] in (0, boundary_axis-1))
            counts["adjacent"] += 1 in x["distances"]
            counts["multi"] += len(x["marker_colors"]) > 1
            counts["same_line"] += len(set(x["lines"])) < len(x["lines"])
            counts["source_shape"] += x["shape"] in {(6,23),(17,6),(6,14),(12,8),(6,17)}
            counts["consecutive_lines"] += any(
                abs(a-b) == 1 for i, a in enumerate(x["lines"])
                for b in x["lines"][i+1:])
            if x["orientation"].startswith("diag"):
                counts["positive_diagonal"] += x["positions"][1] > x["positions"][0]
                counts["negative_diagonal"] += x["positions"][1] < x["positions"][0]
                marker_axis = (1 if x["orientation"] == "diag_rows" else 0)
                counts["diagonal_within_span"] += any(
                    min(x["positions"]) <= cell[marker_axis] <= max(x["positions"])
                    for cell in x["marker_positions"])
            else:
                axis = w if x["orientation"] == "vertical" else h
                boundary = x["positions"][0]
                counts["near_edge"] += boundary in (1, axis-2)
                counts["near_center"] += abs(2*boundary-(axis-1)) <= 2
                counts["same_side_interior"] += (boundary not in (0,axis-1)
                                                and x["markers"] > 1
                                                and len(set(x["sides"])) == 1)
                counts["portrait_vertical"] += x["orientation"] == "vertical" and h>w
                counts["landscape_horizontal"] += x["orientation"] == "horizontal" and w>h
                counts["square_straight"] += abs(h-w) <= 2
            if index < len(puzzle["train"]):
                counts["diagonal_in_train"] += x["orientation"].startswith("diag")
                counts["edge_in_train"] += (x["orientation"] in ("vertical", "horizontal")
                                            and x["positions"][0] in (0,boundary_axis-1))
                counts["adjacent_in_train"] += 1 in x["distances"]
            else:
                counts["straight_in_test"] += not x["orientation"].startswith("diag")
                counts["ordinary_in_test"] += (x["orientation"] in ("vertical", "horizontal")
                                               and x["positions"][0] not in (0,boundary_axis-1)
                                               and min(x["distances"]) >= 2)
        counts["reused_and_new_colors"] += reused and novel
    assert features["orientations"] == {"vertical", "horizontal", "diag_rows", "diag_cols"}
    assert features["marker_counts"] == {1, 2, 3, 4, 5}
    assert features["aspects"] == {"square", "portrait", "landscape"}
    assert len(features["shapes"]) >= 90 and len(features["layouts"]) >= 150
    assert len(features["colors"]) >= 15
    assert all(value > 0 for value in counts.values())
    source_shapes = {(6, 23), (17, 6), (6, 14), (12, 8), (6, 17)}
    reachable_shapes = set(features["shapes"]) & source_shapes
    for seed in range(42, 120):
        if reachable_shapes == source_shapes:
            break
        generated = generate_puzzle(seed)
        reachable_shapes.update(pair["input"].shape
                                for group in ("train", "test")
                                for pair in generated[group]
                                if pair["input"].shape in source_shapes)
    assert reachable_shapes == source_shapes
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 2)):
        p = generate_puzzle(900 + 10*a + b, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
        assert _signature(p) == _signature(generate_puzzle(900 + 10*a + b, a, b))
    for kwargs in ({"num_train": None, "num_test": 1},
                   {"num_train": None, "num_test": 4},
                   {"num_train": 2, "num_test": None},
                   {"num_train": 10, "num_test": None}):
        p = generate_puzzle(1200, **kwargs)
        validate_generated_puzzle(p)
        if kwargs["num_train"] is not None:
            assert len(p["train"]) == kwargs["num_train"]
        if kwargs["num_test"] is not None:
            assert len(p["test"]) == kwargs["num_test"]
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
