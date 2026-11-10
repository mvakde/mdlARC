import numpy as np


def _grid(rows):
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = (
    (_grid(("0000000000000000", "0004444400000000", "2004000400111000",
            "0004000400101000", "0004000400111000", "0004444400000000",
            "0000000000000000", "0888888800000000", "0800000800000000",
            "0800033333333300", "0800030800000300", "0800030800000300",
            "0800030800000300", "0888838800000300", "0000030000000300",
            "0000030000000300")),
     _grid(("333333333", "388888883", "384444483", "384111483",
            "384121483", "384111483", "384444483", "388888883",
            "333333333"))),
    (_grid(("0000000000000", "8888880033330", "8000080030030",
            "8000080030030", "8002222233330", "8002080000200",
            "8882880000200", "0002000000200", "0002000000200",
            "4402000000200", "4402000000200", "0002222222200",
            "0000000000000")),
     _grid(("22222222", "28888882", "28333382", "28344382",
            "28344382", "28333382", "28888882", "22222222"))),
    (_grid(("0000000000000000000", "0066666666660000000",
            "0060000000060002200", "0060888888060002200",
            "0060800008060000000", "0060833338333000000",
            "0060830008063000000", "0060830008063000000",
            "0060888888063000000", "0060030000063000000",
            "0066636666663000000", "0000030000003000000",
            "0000033333333000000", "0000000000000000000",
            "0000000000044440000", "0000000000040040000",
            "0000000000040040000", "0000000000044440000",
            "0000000000000000000")),
     _grid(("6666666666", "6333333336", "6388888836", "6384444836",
            "6384224836", "6384224836", "6384444836", "6388888836",
            "6333333336", "6666666666"))),
)


def _resolve_counts(seed, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (not isinstance(value, (int, np.integer))
                                  or isinstance(value, (bool, np.bool_))
                                  or not low <= value <= high):
            raise ValueError(name)
    key = abs(int(seed))
    if num_train is None and num_test is None:
        num_test = 1 + key % 4
        options = list(range(max(2, num_test + 1), 11))
        num_train = options[(key // 4) % len(options)]
    elif num_train is None:
        options = list(range(max(2, num_test + 1), 11))
        num_train = options[key % len(options)]
    elif num_test is None:
        options = list(range(1, min(4, num_train - 1) + 1))
        num_test = options[key % len(options)]
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def sample_puzzle_parameters(rng):
    # The source has no hidden color map. This only biases the palette and
    # background; the geometric rule is the same in every pair.
    return {"home_background": int(rng.integers(0, 10)),
            "palette": tuple(int(x) for x in rng.permutation(10)),
            "training_palette_size": int(rng.integers(7, 9)),
            "first_parity": int(rng.integers(0, 2))}


def _perimeter(top, left, side, height, width):
    bottom, right = top + side - 1, left + side - 1
    cells = set()
    for col in range(max(left, 0), min(right + 1, width)):
        if 0 <= top < height:
            cells.add((top, col))
        if 0 <= bottom < height:
            cells.add((bottom, col))
    for row in range(max(top + 1, 0), min(bottom, height)):
        if 0 <= left < width:
            cells.add((row, left))
        if 0 <= right < width:
            cells.add((row, right))
    return cells


def _output(colors, side):
    grid = np.empty((side, side), dtype=np.int8)
    for row in range(side):
        for col in range(side):
            grid[row, col] = colors[min(row, col, side - row - 1,
                                         side - col - 1)]
    return grid


def _candidates(grid, background, color, side):
    height, width = grid.shape
    visible_array = np.argwhere(grid == color)
    if not len(visible_array):
        return []
    visible = {(int(r), int(c)) for r, c in visible_array}
    rows = [r for r, _ in visible]
    cols = [c for _, c in visible]
    low_top = max(1 - side, max(rows) - side + 1)
    high_top = min(height - 1, min(rows))
    low_left = max(1 - side, max(cols) - side + 1)
    high_left = min(width - 1, min(cols))
    nonbackground = set(zip(*np.where(grid != background)))
    result = []
    for top in range(low_top, high_top + 1):
        for left in range(low_left, high_left + 1):
            bottom, right = top + side - 1, left + side - 1
            if any(not ((r in (top, bottom) and left <= c <= right)
                        or (c in (left, right) and top <= r <= bottom))
                   for r, c in visible):
                continue
            cells = _perimeter(top, left, side, height, width)
            if cells and cells <= nonbackground:
                result.append((top, left))
    return result


def _infer(grid):
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    colors = [int(x) for x in values if int(x) != background]
    n = len(colors)
    assert 2 <= n <= 9
    matches = []
    for parity in (1, 0):
        side = 2 * n - 1 if parity else 2 * n
        sizes = [side - 2 * depth for depth in range(n)]
        candidate_lists = [[_candidates(grid, background, color, size)
                            for size in sizes] for color in colors]
        if any(not any(row) for row in candidate_lists):
            continue
        # Enumerate at most two color-to-size assignments. A unique assignment
        # means the output is fixed from the input alone.
        assigned = [None] * n
        def search(depth, remaining):
            if len(matches) > 1:
                return
            if depth == n:
                matches.append((side, tuple(assigned), candidate_lists,
                                tuple(colors), background))
                return
            choices = [i for i in remaining if candidate_lists[i][depth]]
            for i in choices:
                assigned[depth] = i
                search(depth + 1, remaining - {i})
        search(0, set(range(n)))
        if len(matches) > 1:
            break
    assert len(matches) == 1
    side, assigned, candidate_lists, colors, background = matches[0]
    ordered_colors = [colors[i] for i in assigned]
    boxes = []
    for depth, i in enumerate(assigned):
        possible = candidate_lists[i][depth]
        assert len(possible) == 1
        boxes.append(possible[0])
    return background, side, ordered_colors, boxes


def _analyze_pair(input_grid, output_grid):
    background, side, colors, boxes = _infer(input_grid)
    assert output_grid.shape == (side, side)
    assert np.array_equal(output_grid, _output(colors, side))
    height, width = input_grid.shape
    perimeters = [_perimeter(top, left, side - 2 * i, height, width)
                  for i, (top, left) in enumerate(boxes)]
    nonbackground = set(zip(*np.where(input_grid != background)))
    assert set().union(*perimeters) == nonbackground
    edges = [set() for _ in colors]
    overlap_cells = 0
    for r, c in nonbackground:
        covering = [i for i, cells in enumerate(perimeters) if (r, c) in cells]
        if len(covering) > 1:
            overlap_cells += 1
        owner = colors.index(int(input_grid[r, c]))
        assert owner in covering
        for i in covering:
            if i != owner:
                edges[i].add(owner)
    indegree = [0] * len(colors)
    for successors in edges:
        for i in successors:
            indegree[i] += 1
    ready = [i for i, d in enumerate(indegree) if d == 0]
    visited = 0
    while ready:
        i = ready.pop()
        visited += 1
        for j in edges[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(j)
    assert visited == len(colors)
    clipped_frames = [i for i, (top, left) in enumerate(boxes)
                      if top < 0 or left < 0 or top + side - 2 * i > height
                      or left + side - 2 * i > width]
    centers = {(2 * top + side - 2 * i - 1,
                2 * left + side - 2 * i - 1)
               for i, (top, left) in enumerate(boxes)}
    return {"background": background, "side": side, "colors": tuple(colors),
            "boxes": tuple(boxes), "overlap_cells": overlap_cells,
            "clipped_frames": tuple(clipped_frames),
            "all_concentric": len(centers) == 1,
            "full_witnesses": tuple(_has_full_side(input_grid, color, side - 2 * i)
                                    for i, color in enumerate(colors))}


def _has_full_side(grid, color, side):
    if side == 1:
        return True
    for row in range(grid.shape[0]):
        for left in range(grid.shape[1] - side + 1):
            if np.all(grid[row, left:left + side] == color):
                return True
    for col in range(grid.shape[1]):
        for top in range(grid.shape[0] - side + 1):
            if np.all(grid[top:top + side, col] == color):
                return True
    return False


def _inside(rng, side, height, width):
    return (int(rng.integers(0, height - side + 1)),
            int(rng.integers(0, width - side + 1)))


def _clipped(rng, side, height, width):
    edge = int(rng.integers(0, 4))
    amount = int(rng.integers(1, min(3, side - 1) + 1))
    top, left = _inside(rng, side, height, width)
    if edge == 0:
        top = -amount
    elif edge == 1:
        top = height - side + amount
    elif edge == 2:
        left = -amount
    else:
        left = width - side + amount
    return top, left


def _boxes(rng, sizes, height, width, style):
    n = len(sizes)
    boxes = [_inside(rng, side, height, width) for side in sizes]
    if style == "disjoint":
        for i in range(1, n):
            occupied = set().union(*(_perimeter(*boxes[j], sizes[j], height, width)
                                     for j in range(i)))
            for _ in range(80):
                trial = _inside(rng, sizes[i], height, width)
                if not (_perimeter(*trial, sizes[i], height, width) & occupied):
                    boxes[i] = trial
                    break
            else:
                return None
    elif style == "nested":
        anchor_top, anchor_left = boxes[0]
        for i in range(1, max(2, n - 1)):
            center_top = anchor_top + (sizes[0] - sizes[i]) // 2
            center_left = anchor_left + (sizes[0] - sizes[i]) // 2
            top = min(max(0, center_top + int(rng.integers(-2, 3))),
                      height - sizes[i])
            left = min(max(0, center_left + int(rng.integers(-2, 3))),
                       width - sizes[i])
            boxes[i] = top, left
    elif style in ("crossing", "crowded", "fragmented"):
        anchor_top, anchor_left = boxes[0]
        limit = n if style == "crowded" else min(n, 3)
        for i in range(1, limit):
            top = anchor_top + int(rng.integers(-sizes[i], sizes[0] + 1))
            left = anchor_left + int(rng.integers(-sizes[i], sizes[0] + 1))
            boxes[i] = (min(max(top, 0), height - sizes[i]),
                        min(max(left, 0), width - sizes[i]))
    if style == "fragmented" and sizes[0] >= 7:
        top, left = boxes[0]
        other = sizes[1]
        edge = int(rng.integers(0, 4))
        if edge == 0:
            boxes[0] = (-1, left)
            crossing = sizes[0] - 2
            boxes[1] = (max(0, min(height - other,
                                   crossing - other // 2)),
                        max(0, min(width - other, left + 1)))
        elif edge == 1:
            top = height - sizes[0] + 1
            boxes[0] = (top, left)
            boxes[1] = (max(0, min(height - other,
                                   top - other // 2)),
                        max(0, min(width - other, left + 1)))
        elif edge == 2:
            boxes[0] = (top, -1)
            crossing = sizes[0] - 2
            boxes[1] = (max(0, min(height - other, top + 1)),
                        max(0, min(width - other,
                                   crossing - other // 2)))
        else:
            left = width - sizes[0] + 1
            boxes[0] = (top, left)
            boxes[1] = (max(0, min(height - other, top + 1)),
                        max(0, min(width - other,
                                   left - other // 2)))
    elif style != "disjoint" and rng.random() < 0.42:
        eligible = [i for i, size in enumerate(sizes) if size >= 4]
        for i in rng.choice(eligible, size=min(len(eligible),
                                               1 + int(rng.random() < 0.2)),
                            replace=False):
            boxes[int(i)] = _clipped(rng, sizes[int(i)], height, width)
    return boxes


def _draw(rng, sizes, colors, boxes, height, width, background, style):
    grid = np.full((height, width), background, dtype=np.int8)
    order = [int(i) for i in rng.permutation(len(sizes))]
    if style == "fragmented":
        order.remove(0)
        order.remove(1)
        order = [0, 1] + order
    for i in order:
        for r, c in _perimeter(*boxes[i], sizes[i], height, width):
            grid[r, c] = colors[i]
    return grid


def _generate_pair(rng, regime, parameters, pair_index, parity=None):
    for attempt in range(250):
        n = int(rng.integers(3, 7 if regime == "source_train" else 9))
        chosen_parity = int(rng.integers(0, 2)) if parity is None else parity
        side = 2 * n - (1 if chosen_parity else 0)
        sizes = [side - 2 * i for i in range(n)]
        style = str(rng.choice(("disjoint", "crossing", "nested",
                                "crowded", "fragmented")))
        if style == "disjoint" and n > 5:
            style = "crossing"
        low = max(side + (7 if style == "disjoint" else 2), 11)
        height = int(rng.integers(low, 31))
        width = int(rng.integers(low, 31))
        if pair_index == 0 and regime == "source_train":
            background = parameters["home_background"]
        elif pair_index == 1 and regime == "source_train":
            background = (parameters["home_background"] +
                          int(rng.integers(1, 10))) % 10
        else:
            background = int(rng.integers(0, 10))
        palette = (parameters["palette"][:parameters["training_palette_size"]]
                   if regime == "source_train" else parameters["palette"])
        available = [c for c in palette if c != background]
        colors = [int(c) for c in rng.choice(available, size=n, replace=False)]
        boxes = _boxes(rng, sizes, height, width, style)
        if boxes is None:
            continue
        grid = _draw(rng, sizes, colors, boxes, height, width, background, style)
        output = _output(colors, side)
        try:
            analysis = _analyze_pair(grid, output)
        except AssertionError:
            continue
        if analysis["all_concentric"]:
            continue
        if style == "disjoint" and analysis["overlap_cells"]:
            continue
        if style == "fragmented" and all(analysis["full_witnesses"][:-1]):
            continue
        return grid, output
    raise RuntimeError("could not construct an unambiguous frame puzzle")


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    used = set()
    for regime, name, count in (("source_train", "train", train_count),
                                ("target_test", "test", test_count)):
        for i in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            parity = (parameters["first_parity"] + i) % 2 if name == "train" and i < 2 else None
            for _ in range(10):
                input_grid, output_grid = _generate_pair(pair_rng, regime,
                                                         parameters, i, parity)
                if _key(input_grid) not in used:
                    used.add(_key(input_grid))
                    puzzle[name].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise RuntimeError("duplicate inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_parities = set()
    backgrounds = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert _key(input_grid) not in seen
            seen.add(_key(input_grid))
            assert input_grid.shape != output_grid.shape or not np.array_equal(input_grid, output_grid)
            assert not any(np.array_equal(input_grid, src_in) and
                           np.array_equal(output_grid, src_out)
                           for src_in, src_out in _SOURCE_PAIRS)
            analysis = _analyze_pair(input_grid, output_grid)
            assert not analysis["all_concentric"]
            assert 3 <= len(analysis["colors"]) <= 8
            backgrounds.add(analysis["background"])
            if split == "train":
                assert 3 <= len(analysis["colors"]) <= 6
                train_parities.add(analysis["side"] % 2)
    assert train_parities == {0, 1}
    assert len(backgrounds) >= 2


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_key(p["train"][0]["input"]) for p in puzzles}) == 42
    assert _equal(generate_puzzle(991, 7, 3), generate_puzzle(991, 7, 3))
    assert not _equal(puzzles[0], puzzles[1])
    for seed, train, test in ((1201, 2, 1), (1202, 10, 1), (1203, 10, 4),
                              (1301, None, 1), (1302, None, 4),
                              (1303, 2, None), (1304, 10, None)):
        p = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(p["train"]) == train
        if test is not None:
            assert len(p["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(1400, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)
    analyses = [_analyze_pair(pair["input"], pair["output"])
                for p in puzzles for split in ("train", "test") for pair in p[split]]
    assert len({a["side"] for a in analyses}) >= 9
    assert len({a["background"] for a in analyses}) == 10
    assert any(a["overlap_cells"] == 0 for a in analyses)
    assert any(a["overlap_cells"] >= 5 for a in analyses)
    assert any(a["clipped_frames"] for a in analyses)
    assert any(not a["clipped_frames"] for a in analyses)
    assert any(not all(a["full_witnesses"][:-1]) for a in analyses)
    assert any(len(a["clipped_frames"]) >= 2 for a in analyses)
    assert any(a["background"] != 0 for a in analyses)
    assert any(a["background"] == 0 for a in analyses)
    assert any(0 in a["colors"] for a in analyses)
    novel_patterns = set()
    novel_depths = set()
    test_parities = set()
    relative_scales = set()
    interrupted_cores = 0
    clip_edges = set()
    clipped_roles = set()
    for puzzle in puzzles:
        train_colors = set()
        train_sizes = []
        for pair in puzzle["train"]:
            a = _analyze_pair(pair["input"], pair["output"])
            train_colors.update(a["colors"])
            train_sizes.append(a["side"])
        assert len(set(train_sizes)) >= 2
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair["input"], pair["output"])
                if a["side"] % 2 == 0:
                    interrupted_cores += int(np.count_nonzero(
                        pair["input"] == a["colors"][-1]) < 4)
                height, width = pair["input"].shape
                for i in a["clipped_frames"]:
                    clipped_roles.add(i)
                    top, left = a["boxes"][i]
                    size = a["side"] - 2 * i
                    if top < 0:
                        clip_edges.add("top")
                    if left < 0:
                        clip_edges.add("left")
                    if top + size > height:
                        clip_edges.add("bottom")
                    if left + size > width:
                        clip_edges.add("right")
                if split == "test":
                    novel = [i for i, color in enumerate(a["colors"])
                             if color not in train_colors]
                    novel_depths.update(novel)
                    novel_patterns.add("none" if not novel else
                                       "multiple" if len(novel) > 1 else
                                       "outer" if novel[0] == 0 else "inner")
                    test_parities.add(a["side"] % 2)
                    relative_scales.add(a["side"] > max(train_sizes))
    assert novel_patterns == {"none", "multiple", "outer", "inner"}
    assert novel_depths == set(range(8))
    assert test_parities == {0, 1}
    assert relative_scales == {False, True}
    assert interrupted_cores > 0
    assert clip_edges == {"top", "bottom", "left", "right"}
    assert len(clipped_roles) >= 4
