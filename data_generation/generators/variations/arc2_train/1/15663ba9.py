"""Synthetic ARC puzzles about convex and reentrant corners of closed contours."""

import numpy as np


# Original inputs are retained only to reject exact source copies.
_SOURCE_INPUTS = (
    ("000000000000000", "088888880000000", "080000080000000",
     "088000080000000", "008000080088888", "008088880080008",
     "008880000088008", "000000000008008", "000000000888088",
     "008880000800080", "008080000888880", "008880000000000"),
    ("0000000000000", "0000333300000", "0033300300000",
     "0030000300000", "0030033300000", "0033330003330",
     "0000000033030", "0000000030030", "0000333330030",
     "0000300000030", "0000300033330", "0000333330000",
     "0000000000000"),
    ("0000000000000000", "0111111000000000", "0100001110000000",
     "0100000010000000", "0111000010000000", "0001000010000000",
     "0001001110000000", "0001111000000000", "0000000000111100",
     "0000000011100110", "0001111110000010", "0001000000111010",
     "0001111111101110", "0000000000000000"),
    ("0000000000000000", "0033300333000000", "0330333303000000",
     "0300000003000000", "0300000003033330", "0333003333030030",
     "0003003000033030", "0333003300003030", "0300000300003030",
     "0300333300333030", "0333300000300030", "0000000000300030",
     "0003330000333330", "0003030000000000", "0003330000000000"),
)


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _resolve_counts(seed, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(num_test) + 1 + seed % (10 - int(num_test))
    elif num_test is None:
        num_test = 1 + seed % min(4, int(num_train) - 1)
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _components(mask):
    remaining = {tuple(map(int, cell)) for cell in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return sorted(result, key=min)


def _neighbors(cell, component):
    r, c = cell
    return sorted(x for x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                  if x in component)


def _trace(component):
    assert len(component) >= 8
    assert all(len(_neighbors(cell, component)) == 2 for cell in component)
    start = min(component)
    path = [start]
    previous, current = start, _neighbors(start, component)[0]
    seen = {start}
    while current != start:
        assert current not in seen
        seen.add(current)
        path.append(current)
        following = [x for x in _neighbors(current, component)
                     if x != previous]
        assert len(following) == 1
        previous, current = current, following[0]
        assert len(path) <= len(component)
    assert len(path) == len(component)
    return path


def _interior_mask(component):
    rows, cols = zip(*component)
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    blocked = np.zeros((max_r-min_r+3, max_c-min_c+3), dtype=bool)
    for r, c in component:
        blocked[r-min_r+1, c-min_c+1] = True
    outside = np.zeros_like(blocked)
    outside[0, 0] = True
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]
                    and not blocked[nr, nc] and not outside[nr, nc]):
                outside[nr, nc] = True
                stack.append((nr, nc))
    return (~blocked) & (~outside)


def _classify(component):
    path = _trace(component)
    area = sum(c * path[(i+1) % len(path)][0]
               - path[(i+1) % len(path)][1] * r
               for i, (r, c) in enumerate(path))
    assert area != 0
    assert _interior_mask(component).any()
    classes = {}
    for i, (r, c) in enumerate(path):
        pr, pc = path[i-1]
        nr, nc = path[(i+1) % len(path)]
        cross = (c-pc)*(nr-r) - (r-pr)*(nc-c)
        classes[(r, c)] = 0 if cross == 0 else (1 if cross*area > 0 else 2)
    convex = sum(kind == 1 for kind in classes.values())
    concave = sum(kind == 2 for kind in classes.values())
    assert convex >= 4 and convex-concave == 4
    rows, cols = zip(*component)
    return {
        "cells": component, "path": path, "classes": classes,
        "convex": convex, "concave": concave,
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
    }


def _analyze(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) >= 2
    background = int(values[int(np.argmax(counts))])
    assert int(max(counts)) > grid.size // 2
    components = []
    for cells in _components(grid != background):
        colors = {int(grid[r, c]) for r, c in cells}
        assert len(colors) == 1
        info = _classify(cells)
        info["color"] = colors.pop()
        components.append(info)
    assert components
    return {"background": background, "components": components}


def _paint(grid, analysis, convex_color, concave_color):
    result = grid.copy()
    for info in analysis["components"]:
        for (r, c), kind in info["classes"].items():
            if kind == 1:
                result[r, c] = convex_color
            elif kind == 2:
                result[r, c] = concave_color
    return result


def _polygon_mask(vertices):
    if len(vertices) < 4 or len(set(vertices)) != len(vertices):
        return None
    cells = []
    for i, start in enumerate(vertices):
        end = vertices[(i+1) % len(vertices)]
        dr, dc = end[0]-start[0], end[1]-start[1]
        if (dr == 0) == (dc == 0):
            return None
        step_r, step_c = int(np.sign(dr)), int(np.sign(dc))
        for t in range(abs(dr)+abs(dc)):
            cells.append((start[0]+t*step_r, start[1]+t*step_c))
    if len(cells) != len(set(cells)):
        return None
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    height = max(r for r, _ in cells)-min_r+1
    width = max(c for _, c in cells)-min_c+1
    if height < 3 or width < 3:
        return None
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r-min_r, c-min_c] = True
    try:
        components = _components(mask)
        if len(components) != 1:
            return None
        _classify(components[0])
    except AssertionError:
        return None
    return mask


def _mutate_corner(vertices, rng):
    index = int(rng.integers(len(vertices)))
    previous = vertices[index-1]
    current = vertices[index]
    following = vertices[(index+1) % len(vertices)]
    first_length = abs(previous[0]-current[0])+abs(previous[1]-current[1])
    second_length = abs(following[0]-current[0])+abs(following[1]-current[1])
    if min(first_length, second_length) < 2:
        return None
    a = int(rng.integers(1, min(4, first_length)))
    b = int(rng.integers(1, min(4, second_length)))
    toward_previous = ((previous[0]-current[0]) // first_length,
                       (previous[1]-current[1]) // first_length)
    toward_following = ((following[0]-current[0]) // second_length,
                        (following[1]-current[1]) // second_length)
    p = (current[0]+a*toward_previous[0],
         current[1]+a*toward_previous[1])
    q = (current[0]+b*toward_following[0],
         current[1]+b*toward_following[1])
    middle = (p[0]+b*toward_following[0],
              p[1]+b*toward_following[1])
    return vertices[:index]+[p, middle, q]+vertices[index+1:]


def _mutate_edge(vertices, rng):
    index = int(rng.integers(len(vertices)))
    start = vertices[index]
    end = vertices[(index+1) % len(vertices)]
    length = abs(start[0]-end[0])+abs(start[1]-end[1])
    if length < 3:
        return None
    u = int(rng.integers(1, length-1))
    v = int(rng.integers(u+1, length))
    depth = int(rng.integers(1, 5))
    dr, dc = (end[0]-start[0]) // length, (end[1]-start[1]) // length
    side = int(rng.choice([-1, 1]))
    pr, pc = side*dc, -side*dr
    additions = [
        (start[0]+u*dr, start[1]+u*dc),
        (start[0]+u*dr+depth*pr, start[1]+u*dc+depth*pc),
        (start[0]+v*dr+depth*pr, start[1]+v*dc+depth*pc),
        (start[0]+v*dr, start[1]+v*dc),
    ]
    return vertices[:index+1]+additions+vertices[index+1:]


def _shape(rng, max_h, max_w, complexity):
    # Successive local edits compose staircases, offset shoulders, arms,
    # and notches on independently chosen sides.
    for _ in range(18):
        if complexity == 0 and rng.random() < .15:
            h = w = 3
        elif complexity == 0 and rng.random() < .45:
            h = int(rng.integers(3, min(max_h, 14)+1))
            w = int(rng.integers(3, min(max_w, 14)+1))
        else:
            h = int(rng.integers(4, min(max_h, 13)+1))
            w = int(rng.integers(4, min(max_w, 13)+1))
        if rng.random() < .22 and max(max_h, max_w) >= 10:
            if rng.random() < .5:
                h = min(h, int(rng.integers(3, 6)))
                w = int(rng.integers(8, min(max_w, 18)+1))
            else:
                w = min(w, int(rng.integers(3, 6)))
                h = int(rng.integers(8, min(max_h, 18)+1))
        vertices = [(0, 0), (0, w-1), (h-1, w-1), (h-1, 0)]
        best = _polygon_mask(vertices)
        target = (0 if complexity == 0 else
                  int(rng.integers(1, 4)) if complexity == 1 else
                  int(rng.integers(4, 10)) if complexity == 2 else
                  int(rng.integers(8, 11)))
        for _ in range(125 if complexity == 3 else
                 95 if complexity == 2 else 45):
            if best is None:
                break
            current = _classify(_components(best)[0])["concave"]
            if current >= target:
                break
            proposal = (_mutate_corner(vertices, rng) if rng.random() < .52
                        else _mutate_edge(vertices, rng))
            if proposal is None:
                continue
            candidate = _polygon_mask(proposal)
            if (candidate is not None and candidate.shape[0] <= max_h
                    and candidate.shape[1] <= max_w):
                candidate_count = _classify(_components(candidate)[0])["concave"]
                if candidate_count > current:
                    vertices, best = proposal, candidate
        if best is not None and (
            complexity == 0 or _classify(_components(best)[0])["concave"]
            >= (1 if complexity == 1 else 8 if complexity == 3 else 4)
        ):
            return np.ascontiguousarray(
                np.fliplr(np.rot90(best, int(rng.integers(4))))
                if rng.random() < .5 else np.rot90(best, int(rng.integers(4))),
                dtype=bool,
            )
    return None


def _overlap_area(first, second):
    return max(0, min(first[2], second[2])-max(first[0], second[0])+1) * (
        max(0, min(first[3], second[3])-max(first[1], second[1])+1))


def _pack(rng, shapes, height, width):
    # Test collision on actual occupied cells, including a one-cell halo.
    # Bounding boxes may interlock at indentations.
    order = sorted(range(len(shapes)), key=lambda i: -int(shapes[i].sum()))
    for _ in range(7):
        forbidden = np.zeros((height, width), dtype=bool)
        positions = [None] * len(shapes)
        boxes = []
        for i in order:
            shape = shapes[i]
            sh, sw = shape.shape
            feasible = []
            for row in range(height-sh+1):
                for col in range(width-sw+1):
                    if np.any(forbidden[row:row+sh, col:col+sw] & shape):
                        continue
                    box = (row, col, row+sh-1, col+sw-1)
                    if any(
                        (box[0] >= old[0] and box[1] >= old[1]
                         and box[2] <= old[2] and box[3] <= old[3])
                        or (old[0] >= box[0] and old[1] >= box[1]
                            and old[2] <= box[2] and old[3] <= box[3])
                        for old in boxes
                    ):
                        continue
                    overlap = sum(_overlap_area(box, old) for old in boxes)
                    edge = int(row == 0 or col == 0 or row+sh == height
                               or col+sw == width)
                    feasible.append((row, col, box, overlap, edge))
            if not feasible:
                break
            if boxes and rng.random() < .30:
                largest = max(x[3] for x in feasible)
                if largest:
                    feasible = [x for x in feasible if x[3] == largest]
            elif rng.random() < .24:
                edges = [x for x in feasible if x[4]]
                if edges:
                    feasible = edges
            row, col, box, _, _ = feasible[int(rng.integers(len(feasible)))]
            positions[i] = (row, col)
            boxes.append(box)
            for sr, sc in np.argwhere(shape):
                rr, cc = row+int(sr), col+int(sc)
                forbidden[max(0, rr-1):min(height, rr+2),
                          max(0, cc-1):min(width, cc+2)] = True
        if all(p is not None for p in positions):
            return positions
        order = [int(x) for x in rng.permutation(len(shapes))]
    return None


def _make_pair(rng, regime, params, index, dimensions=None):
    modest_complex_test = (regime == "target_test" and index == 0
                           and dimensions is None and rng.random() < .30)
    for attempt in range(100):
        height, width = (dimensions if dimensions is not None else
                         (int(rng.integers(13, 17)), int(rng.integers(13, 17)))
                         if modest_complex_test else
                         (int(rng.integers(9, 31)), int(rng.integers(9, 31))))
        area = height*width
        maximum = min(5, max(1, area//105))
        count = int(rng.integers(1, maximum+1))
        if regime == "target_test" and rng.random() < .32:
            count = min(maximum, max(2, count))
        max_h = min(height, 17 if count <= 2 else 13)
        max_w = min(width, 17 if count <= 2 else 13)
        small_collection = count >= 3 and rng.random() < .10
        if small_collection:
            max_h = max_w = 5
        shapes = []
        for i in range(count):
            if small_collection and not (index == 0 and i == 0):
                complexity = 0
            elif regime == "source_train" and index == 0 and i == 0:
                complexity = 1
            elif modest_complex_test and i == 0:
                complexity = 3
            elif regime == "target_test" and index == 0 and i == 0:
                complexity = 2 if min(height, width) >= 15 else 1
            else:
                complexity = int(rng.choice([0, 1, 2], p=[.25, .43, .32]))
            shape = _shape(rng, max_h, max_w, complexity)
            if shape is None:
                break
            shapes.append(shape)
        if len(shapes) != count:
            continue
        positions = _pack(rng, shapes, height, width)
        if positions is None:
            continue
        background = params["backgrounds"][
            (index + (0 if regime == "source_train" else 2)) % len(params["backgrounds"])
        ]
        grid = np.full((height, width), background, dtype=np.int8)
        available = [c for c in (params["train_colors"] if regime == "source_train"
                                 else params["all_foregrounds"])
                     if c != background]
        colors = [int(available[int(rng.integers(len(available)))])
                  for _ in shapes]
        if regime == "target_test" and index == 0:
            heldout = [c for c in params["heldout"] if c != background]
            colors[0] = int(heldout[int(rng.integers(len(heldout)))])
        if len(shapes) > 1 and rng.random() < .52:
            choices = [c for c in available if c != colors[0]]
            if choices:
                colors[1] = int(choices[int(rng.integers(len(choices)))])
        for shape, (row, col), color in zip(shapes, positions, colors):
            grid[row:row+shape.shape[0], col:col+shape.shape[1]][shape] = color
        if int(np.count_nonzero(grid == background)) <= grid.size//2:
            continue
        try:
            analysis = _analyze(grid)
        except AssertionError:
            continue
        if len(analysis["components"]) != count:
            continue
        output = _paint(grid, analysis, params["convex"], params["concave"])
        return {"input": grid, "output": output}
    raise RuntimeError("Could not place valid contours")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.permutation(10)]
    params = {
        "convex": palette[0], "concave": palette[1],
        "backgrounds": palette[2:5],
        "train_colors": palette[5:8],
        "heldout": palette[8:10],
        "all_foregrounds": palette[5:10],
    }
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count, split in (("source_train", train_count, "train"),
                                 ("target_test", test_count, "test")):
        for index in range(count):
            for _ in range(8):
                pair_rng = np.random.default_rng(
                    int(rng.integers(0, 2**63, dtype=np.uint64)))
                dimensions = None
                if (regime == "target_test" and rng.random() < .32):
                    source = puzzle["train"][int(rng.integers(len(puzzle["train"])))]
                    dimensions = source["input"].shape
                pair = _make_pair(pair_rng, regime, params, index, dimensions)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not generate a unique input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    analyses = []
    seen_inputs = set()
    seen_pairs = set()
    labels = {1: set(), 2: set()}
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        grid, output = pair["input"], pair["output"]
        _validate_grid(grid)
        _validate_grid(output)
        assert grid.shape == output.shape
        assert tuple("".join(map(str, row)) for row in grid) not in _SOURCE_INPUTS
        input_key = _grid_key(grid)
        pair_key = (input_key, _grid_key(output))
        assert input_key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)
        analysis = _analyze(grid)
        analyses.append(analysis)
        assert not np.array_equal(grid, output)
        for i, first in enumerate(analysis["components"]):
            a = first["bbox"]
            for second in analysis["components"][i+1:]:
                b = second["bbox"]
                assert not ((a[0] >= b[0] and a[1] >= b[1]
                             and a[2] <= b[2] and a[3] <= b[3])
                            or (b[0] >= a[0] and b[1] >= a[1]
                                and b[2] <= a[2] and b[3] <= a[3]))
                assert all((abs(r-s) > 1 or abs(c-t) > 1)
                           for r, c in first["cells"]
                           for s, t in second["cells"])
        for info in analysis["components"]:
            for (r, c), kind in info["classes"].items():
                if kind:
                    labels[kind].add(int(output[r, c]))
                else:
                    assert output[r, c] == grid[r, c]
        assert np.array_equal(output[grid == analysis["background"]],
                              grid[grid == analysis["background"]])
    assert len(labels[1]) == len(labels[2]) == 1
    convex, concave = next(iter(labels[1])), next(iter(labels[2]))
    assert convex != concave
    input_colors = {int(x) for pair in puzzle["train"]+puzzle["test"]
                    for x in np.unique(pair["input"])}
    assert convex not in input_colors and concave not in input_colors
    for pair, analysis in zip(puzzle["train"]+puzzle["test"], analyses):
        assert np.array_equal(
            pair["output"], _paint(pair["input"], analysis, convex, concave))
    train_analyses = analyses[:len(puzzle["train"])]
    assert any(info["concave"] for a in train_analyses
               for info in a["components"])
    assert len({a["background"] for a in train_analyses}) >= 2
    train_foregrounds = {info["color"] for a in train_analyses
                         for info in a["components"]}
    test_foregrounds = {info["color"] for a in analyses[len(train_analyses):]
                        for info in a["components"]}
    assert test_foregrounds - train_foregrounds
    assert any(info["concave"] for a in analyses[len(train_analyses):]
               for info in a["components"])


def _equal(first, second):
    return all(
        len(first[split]) == len(second[split]) and
        all(np.array_equal(a[key], b[key])
            for a, b in zip(first[split], second[split])
            for key in ("input", "output"))
        for split in ("train", "test"))


def _point_inside(path, row, col):
    crossings = 0
    for (r1, c1), (r2, c2) in zip(path, path[1:]+path[:1]):
        if c1 == c2 and min(r1, r2) < row < max(r1, r2) and c1 > col:
            crossings += 1
    return crossings % 2 == 1


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, layouts, conventions, object_counts, concaves = (
        set(), set(), set(), set(), set())
    turn_patterns = set()
    adjacent = small = edge = overlapping = multicolor = 0
    all_irregular = convex_only = same_complexity = several_small = 0
    source_sized = elongated = 0
    dense_modest_tests = 0
    tall = wide = 0
    mixed_styles = heldout_axes = familiar_axes = 0
    smaller_tests = larger_tests = matched_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_shapes = []
        train_axes, test_axes, styles = set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                analysis = _analyze(grid)
                pair_shapes.append(grid.shape)
                shapes.add(grid.shape)
                tall += grid.shape[0] > grid.shape[1]
                wide += grid.shape[1] > grid.shape[0]
                source_sized += (12 <= grid.shape[0] <= 16
                                 and 13 <= grid.shape[1] <= 16)
                object_counts.add(len(analysis["components"]))
                conventions.add(tuple(sorted(
                    set(np.unique(pair["output"])) - set(np.unique(grid)))))
                layouts.add((grid.shape, tuple(sorted(
                    (i["bbox"], i["concave"]) for i in analysis["components"]))))
                multicolor += len({i["color"] for i in analysis["components"]}) > 1
                complexities = [i["concave"] for i in analysis["components"]]
                all_irregular += all(complexities)
                convex_only += not any(complexities)
                same_complexity += (len(complexities) > 1
                                    and len(set(complexities)) == 1)
                compact_count = 0
                for info in analysis["components"]:
                    concaves.add(info["concave"])
                    dense_modest_tests += (split == "test"
                                           and grid.shape[0] <= 16
                                           and grid.shape[1] <= 16
                                           and info["concave"] >= 8)
                    br, bc, er, ec = info["bbox"]
                    bh, bw = er-br+1, ec-bc+1
                    compact_count += bh <= 5 and bw <= 5
                    elongated += max(bh, bw) >= 2*min(bh, bw)
                    axis = ("vertical" if bh >= 1.5*bw else
                            "horizontal" if bw >= 1.5*bh else "compact")
                    (train_axes if split == "train" else test_axes).add(axis)
                    small += (er-br+1, ec-bc+1) == (3, 3)
                    edge += br == 0 or bc == 0 or er == grid.shape[0]-1 or ec == grid.shape[1]-1
                    path = info["path"]
                    kinds = [info["classes"][x] for x in path]
                    turns = [kind for kind in kinds if kind]
                    canonical = min(tuple(turns[i:]+turns[:i])
                                    for i in range(len(turns)))
                    reverse = turns[::-1]
                    canonical = min(canonical, *(tuple(reverse[i:]+reverse[:i])
                                                 for i in range(len(reverse))))
                    turn_patterns.add(canonical)
                    adjacent += any(kinds[i] and kinds[(i+1) % len(kinds)]
                                    for i in range(len(kinds)))
                    if info["concave"] >= 2:
                        sides = set()
                        for (r, c), kind in info["classes"].items():
                            if kind == 2:
                                distances = (r-br, er-r, c-bc, ec-c)
                                sides.add(distances.index(min(distances)))
                        styles.add("single" if len(sides) == 1 else "multiple")
                    # Point-in-polygon gives an independent local-angle check.
                    for (r, c), kind in info["classes"].items():
                        if kind:
                            quadrants = sum(_point_inside(
                                path, r+dr, c+dc)
                                for dr in (-.25, .25) for dc in (-.25, .25))
                            assert quadrants == (1 if kind == 1 else 3)
                for i, a in enumerate(analysis["components"]):
                    overlapping += any(_overlap_area(a["bbox"], b["bbox"])
                                       for b in analysis["components"][i+1:])
                several_small += compact_count >= 3
        mixed_styles += len(styles) == 2
        heldout_axes += bool(test_axes-train_axes)
        familiar_axes += bool(test_axes & train_axes)
        train_areas = [h*w for h, w in pair_shapes[:len(puzzle["train"])]]
        test_areas = [h*w for h, w in pair_shapes[len(puzzle["train"]):]]
        smaller_tests += min(test_areas) < max(train_areas)
        larger_tests += max(test_areas) > max(train_areas)
        matched_tests += bool(set(pair_shapes[:len(puzzle["train"])]) &
                              set(pair_shapes[len(puzzle["train"]):]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 70 and len(layouts) >= 150
    assert len(conventions) >= 20 and len(turn_patterns) >= 30
    assert {1, 2, 3, 4, 5}.issubset(object_counts)
    assert {0, 1, 2, 3, 4, 5, 6, 7, 8}.issubset(concaves)
    assert adjacent and small and edge and overlapping and multicolor
    assert all_irregular and convex_only and same_complexity and several_small
    assert source_sized and elongated and tall and wide
    assert dense_modest_tests
    assert mixed_styles and heldout_axes and familiar_axes
    assert smaller_tests and larger_tests and matched_tests
    assert _equal(generate_puzzle(777, 5, 2), generate_puzzle(777, 5, 2))
    assert not _equal(generate_puzzle(778, 5, 2), generate_puzzle(779, 5, 2))
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None),
                              (3001, 3, 1), (3002, 6, 2), (3003, 9, 3)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(4001, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid count accepted")
