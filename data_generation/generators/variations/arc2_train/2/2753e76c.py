from __future__ import annotations

import numpy as np


_SOURCE_INPUT_ROWS = (
    ("0000000044000220", "0002220044000220", "0002220000000000", "0002220000222000", "0000000000222000", "0000003300222000", "8888003300000000", "8888000000002222", "8888000333002222", "8888000333002222", "0000000333002222", "0008800000000000", "0008800000000000", "0000000000003300", "0000000000003300", "0000000000000000"),
    ("8888800000000000", "8888800000000000", "8888800088880000", "8888800088880000", "8888800088880000", "0000000088880000", "0000000000000000", "0000088880011000", "0000088880011000", "0000088880000000", "0000088880000000", "0220000000000000", "0220000222000000", "0000000222000000", "0000000222000000", "0000000000000000"),
    ("0000000088088800", "0110000088088800", "0110111000088800", "0000111000000000", "0000111000000880", "0000000000000880", "0000000333000000", "2220000333011111", "2220000333011111", "2220000000011111", "0000000000011111", "0000111100011111", "1100111100000000", "1100111102220000", "0000111102220000", "0000000002220000"),
    ("0000008800000000", "0333008800000000", "0333000000330088", "0333000000330088", "0000000000000000", "0000444000000000", "0000444004400000", "0000444004400033", "0000000000000033", "0000044400000000", "0000044400000000", "0330044403333000", "0330000003333000", "0000440003333000", "0000440003333000", "0000000000000000"),
)


def _signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _background(grid):
    values, amounts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(amounts))])


def _components(grid, background):
    height, width = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    result = {}
    for row in range(height):
        for col in range(width):
            color = int(grid[row, col])
            if color == background or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                    if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            box = (min(rows), max(rows), min(cols), max(cols), len(cells))
            result.setdefault(color, []).append(box)
    return result


def _chart(grid):
    background = _background(grid)
    components = _components(grid, background)
    ranked = sorted(((color, len(parts)) for color, parts in components.items()), key=lambda x: -x[1])
    output = np.full((len(ranked), ranked[0][1]), background, dtype=np.int8)
    for row, (color, count) in enumerate(ranked):
        output[row, -count:] = color
    return output


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(value)


def _sample_counts(rng, categories, regime, require_gap=False):
    upper = 9 if categories <= 4 else 8
    if regime == "source_train":
        upper = min(upper, 7)
    if categories <= 3 and not require_gap and rng.random() < .32:
        upper = int(rng.integers(categories, min(upper, categories + 2) + 1))
    for _ in range(80):
        # Choosing a lower bound separately gives singleton and no-singleton charts.
        minimum = 1 if rng.random() < 0.55 else 2
        choices = np.arange(minimum, upper + 1)
        if len(choices) < categories:
            continue
        counts = tuple(sorted((int(x) for x in rng.choice(choices, size=categories, replace=False)), reverse=True))
        if sum(counts) > 30 or (require_gap and not any(a-b > 1 for a, b in zip(counts, counts[1:]))):
            continue
        return counts
    counts = tuple(range(categories + 1, 1, -1))
    if require_gap:
        counts = (counts[0] + 1,) + counts[1:]
    return counts


def _sample_shapes(rng, counts, colors, area_reversal, cap):
    shapes = []
    all_square = area_reversal or rng.random() < 0.29
    for rank, (color, count) in enumerate(zip(colors, counts)):
        for index in range(count):
            if area_reversal and rank == 0:
                h = w = 1 if counts[1] == 1 and index > 0 else 2
            elif area_reversal and rank == 1:
                h = w = 5 if index == 0 else 4
            else:
                size_limit = min(cap, 3 if sum(counts) > 21 else cap)
                if all_square:
                    h = w = int(rng.integers(2 if size_limit >= 3 and rng.random() < .7 else 1, size_limit + 1))
                else:
                    h = int(rng.integers(1, size_limit + 1))
                    w = int(rng.integers(1, size_limit + 1))
            shapes.append((int(color), h, w))
    return shapes


def _dimensions(rng, shapes, area_reversal, expansion):
    area = sum(h*w for _, h, w in shapes)
    density = float(rng.uniform(.19, .39)) if area_reversal else float(rng.uniform(.12, .39))
    required = area / density
    ratio = float(rng.uniform(.55, 1.8))
    height = int(np.ceil(np.sqrt(required / ratio))) + expansion
    width = int(np.ceil(np.sqrt(required * ratio))) + expansion
    height = max(9, min(30, height))
    width = max(9, min(30, width))
    return height, width


def _place(rng, shape, background, shapes, style):
    height, width = shape
    grid = np.full(shape, background, dtype=np.int8)
    blocked = np.zeros(shape, dtype=bool)
    order = rng.permutation(len(shapes))
    order = sorted((int(i) for i in order), key=lambda i: -shapes[i][1] * shapes[i][2])
    anchor = (int(rng.integers(height)), int(rng.integers(width)))
    lattice = (int(rng.integers(3, 7)), int(rng.integers(3, 7)))
    for index in order:
        color, rh, rw = shapes[index]
        if rh > height or rw > width:
            return None
        candidates = [(r, c) for r in range(height-rh+1) for c in range(width-rw+1)]
        rng.shuffle(candidates)
        if style == "cluster":
            candidates.sort(key=lambda p: abs(p[0]-anchor[0]) + abs(p[1]-anchor[1]) + float(rng.random())*5)
        elif style == "orderly":
            candidates.sort(key=lambda p: min(p[0] % lattice[0], lattice[0]-p[0] % lattice[0]) + min(p[1] % lattice[1], lattice[1]-p[1] % lattice[1]) + float(rng.random())*2)
        placed = False
        for row, col in candidates:
            if blocked[row:row+rh, col:col+rw].any():
                continue
            grid[row:row+rh, col:col+rw] = color
            blocked[max(0,row-1):min(height,row+rh+1), max(0,col-1):min(width,col+rw+1)] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _generate_pair(rng, regime, categories, force_gap=False, area_reversal=False, background=None, rank_colors=None):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    counts = _sample_counts(rng, categories, regime, force_gap)
    if background is None:
        background = int(rng.integers(10))
    available = [color for color in range(10) if color != background]
    if rank_colors is not None:
        colors = list(rank_colors)
    elif background != 0:
        available.remove(0)
        colors = [0] + [int(x) for x in rng.choice(available, size=categories-1, replace=False)]
        rng.shuffle(colors)
    else:
        colors = [int(x) for x in rng.choice(available, size=categories, replace=False)]
        rng.shuffle(colors)
    assert len(colors) == categories and len(set(colors)) == categories and background not in colors
    style = str(rng.choice(("scatter", "cluster", "orderly"), p=(.55, .30, .15)))
    for attempt in range(18):
        cap = 5 if attempt < 6 else 4 if attempt < 12 else 2
        shapes = _sample_shapes(rng, counts, colors, area_reversal, cap)
        shape = _dimensions(rng, shapes, area_reversal, attempt // 4)
        for expansion in (0, 2, 4, 6, 8, 30):
            enlarged = (min(30, shape[0] + expansion), min(30, shape[1] + expansion))
            grid = _place(rng, enlarged, background, shapes, style)
            if grid is not None and int(np.count_nonzero(grid != background)) < grid.size * .48:
                return {"input": grid, "output": _chart(grid)}
    raise RuntimeError("could not place separated objects")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count = _checked_count("num_train", num_train, 2, 10)
    test_count = _checked_count("num_test", num_test, 1, 4)
    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(max(2, test_count+1), 11))
    elif train_count is None:
        train_count = int(rng.integers(max(2, test_count+1), 11))
    elif test_count is None:
        test_count = int(rng.integers(1, min(4, train_count-1)+1))
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    # The learned convention is fixed: count color components, sort by count,
    # and right-align the rows. Colors and backgrounds have no fixed meanings.
    puzzle = {"train": [], "test": []}
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=train_count+test_count, dtype=np.uint64)
    first_categories = int(rng.integers(2, 5))
    second_categories = int(rng.choice([k for k in range(3, 6) if k != first_categories]))
    for index in range(train_count+test_count):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        if index == 0:
            pair = _generate_pair(pair_rng, "source_train", first_categories, force_gap=True, area_reversal=True, background=0)
        elif index == 1:
            first_parts = _components(puzzle["train"][0]["input"], 0)
            first_rank = sorted(first_parts, key=lambda color: -len(first_parts[color]))
            higher, lower = first_rank[:2]
            candidates = [color for color in range(1, 10) if color not in (higher, lower)]
            second_background = int(rng.choice(candidates))
            remaining = [color for color in range(1, 10) if color not in (second_background, higher, lower)]
            second_colors = [lower, higher, 0]
            second_colors.extend(int(x) for x in rng.choice(remaining, size=second_categories-3, replace=False))
            pair = _generate_pair(pair_rng, "source_train", second_categories, background=second_background, rank_colors=second_colors)
        else:
            split = "train" if index < train_count else "test"
            categories = int(rng.choice(np.arange(2, 7), p=(.19,.29,.26,.18,.08)))
            pair = _generate_pair(pair_rng, "source_train" if split == "train" else "target_test", categories)
        puzzle["train" if index < train_count else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _source_signatures():
    return {_signature(np.array([[int(c) for c in row] for row in rows], dtype=np.int8)) for rows in _SOURCE_INPUT_ROWS}


def _inspect_pair(pair):
    grid, output = pair["input"], pair["output"]
    background = _background(grid)
    assert int(np.count_nonzero(grid == background)) > grid.size // 2
    components = _components(grid, background)
    assert 2 <= len(components) <= 6
    counts = [len(parts) for parts in components.values()]
    assert len(counts) == len(set(counts))
    assert max(counts) <= 9
    boxes = []
    for color, parts in components.items():
        for r0, r1, c0, c1, area in parts:
            assert (r1-r0+1)*(c1-c0+1) == area
            assert np.all(grid[r0:r1+1, c0:c1+1] == color)
            boxes.append((r0,r1,c0,c1))
    for i, (r0,r1,c0,c1) in enumerate(boxes):
        for q0,q1,d0,d1 in boxes[i+1:]:
            assert r1+1 < q0 or q1+1 < r0 or c1+1 < d0 or d1+1 < c0
    assert output.shape == (len(components), max(counts))
    assert np.array_equal(output, _chart(grid))
    return background, tuple(sorted(counts, reverse=True)), components


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    source_inputs = _source_signatures()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _signature(pair["input"])
            pair_key = (input_key, _signature(pair["output"]))
            assert input_key not in seen_inputs and input_key not in source_inputs
            assert pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyses[split].append(_inspect_pair(pair))
    # Training exposes both a nonzero background and zero-colored objects,
    # variable category counts, a count gap, and an area/count rank reversal.
    assert analyses["train"][0][0] == 0
    assert analyses["train"][1][0] != 0 and 0 in analyses["train"][1][2]
    assert len(analyses["train"][0][2]) != len(analyses["train"][1][2])
    assert any(a-b > 1 for a,b in zip(analyses["train"][0][1], analyses["train"][0][1][1:]))
    first = analyses["train"][0][2]
    ordered = sorted(first, key=lambda color: -len(first[color]))
    areas = [sum(box[4] for box in first[color]) for color in ordered]
    assert any(areas[j] < areas[j+1] for j in range(len(areas)-1))
    second = analyses["train"][1][2]
    shared = set(first) & set(second)
    assert len(shared) >= 2
    assert any(
        len(first[a]) > len(first[b]) and len(second[a]) < len(second[b])
        for a in shared for b in shared
    )


def _puzzles_equal(a, b):
    return all(np.array_equal(pa[role], pb[role]) for split in ("train", "test") for pa, pb in zip(a[split], b[split]) for role in ("input", "output")) and all(len(a[s]) == len(b[s]) for s in ("train", "test"))


def _independent_prediction(train_pairs, test_input):
    # Infer the chart convention from visible demonstrations, then count
    # components in the unlabelled test input using a separate flood fill.
    for pair in train_pairs:
        assert np.array_equal(pair["output"], _chart(pair["input"]))
    grid = test_input
    values, amounts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(amounts))])
    totals = {}
    for color in values:
        color = int(color)
        if color == background:
            continue
        mask = grid == color
        count = 0
        while np.any(mask):
            row, col = np.argwhere(mask)[0]
            mask[row, col] = False
            frontier = [(int(row), int(col))]
            count += 1
            while frontier:
                r, c = frontier.pop()
                for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and mask[rr, cc]:
                        mask[rr, cc] = False
                        frontier.append((rr, cc))
        totals[color] = count
    ranks = sorted(totals, key=lambda c: -totals[c])
    result = np.full((len(ranks), totals[ranks[0]]), background, dtype=np.int8)
    for row, color in enumerate(ranks):
        result[row, -totals[color]:] = color
    return result


def self_test() -> None:
    puzzles = [generate_puzzle(10000 + 7919*i) for i in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    count_profiles, canvas_shapes, palettes, backgrounds = set(), set(), set(), set()
    all_square = mixed = gap = singleton = no_singleton = reversal = five_plus = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                bg, counts, components = _inspect_pair(pair)
                backgrounds.add(bg)
                palettes.add(tuple(sorted(components)))
                count_profiles.add(counts)
                canvas_shapes.add(grid.shape)
                boxes = [box for parts in components.values() for box in parts]
                squares = all(box[1]-box[0] == box[3]-box[2] for box in boxes)
                all_square += squares
                mixed += not squares
                gap += any(a-b > 1 for a,b in zip(counts, counts[1:]))
                singleton += counts[-1] == 1
                no_singleton += counts[-1] > 1
                five_plus += len(counts) >= 5
                ordered = sorted(components, key=lambda c: -len(components[c]))
                areas = [sum(box[4] for box in components[c]) for c in ordered]
                reversal += any(areas[i] < areas[i+1] for i in range(len(areas)-1))
        training = [{"input": p["input"].copy(), "output": p["output"].copy()} for p in puzzle["train"]]
        for pair in puzzle["test"]:
            assert np.array_equal(_independent_prediction(training, pair["input"].copy()), pair["output"])
    assert len(count_profiles) >= 35 and len(canvas_shapes) >= 45
    assert len(palettes) >= 35 and len(backgrounds) >= 8
    assert all_square >= 40 and mixed >= 40 and gap >= 80
    assert singleton >= 30 and no_singleton >= 30 and reversal >= 42 and five_plus >= 15
    for train, test in ((2,1),(10,1),(10,4),(7,3)):
        puzzle = generate_puzzle(5000+10*train+test, train, test)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None,1),(None,4),(2,None),(10,None)):
        puzzle = generate_puzzle(6000+(train or 0)+(test or 0), train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    a = generate_puzzle(314159, 7, 3)
    assert _puzzles_equal(a, generate_puzzle(314159, 7, 3))
    assert not _puzzles_equal(a, generate_puzzle(314160, 7, 3))
    for kwargs in ({"num_train":1},{"num_train":11},{"num_test":0},{"num_test":5},{"num_train":2,"num_test":2},{"num_train":3,"num_test":4},{"num_train":True,"num_test":1}):
        try:
            generate_puzzle(7000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
