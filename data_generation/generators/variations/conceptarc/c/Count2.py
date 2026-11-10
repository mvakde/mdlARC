"""Closed polygon outlines whose straight-side count becomes a unary strip."""

import hashlib

import numpy as np


# Original input hashes let the validator permit original sizes and palettes.
SOURCE_INPUTS = {
    "7f84ec5715acb76e1826fe7bb0885729be2eca743e02edf9f7ce2ab9d72d0cd9",
    "367eacf70a622cba7abc557c88a95971b0207c7c381b241ff26170d94c846bd0",
    "8a87594eec4475a6746682703a4d3935f3cf00d7f6c7e8e32b8c0f7208e11efe",
    "6736c6afa3116b2b991d225bbeb22b0cacc21bdf661a65c80e831d313929d34b",
    "5b9aa12ed883613d13bf1c5a20ae678352400bbab93dd3ae2b157f62bcf10ba5",
    "a2e55d3a88c9d81ae0640a23e30051833ef2cb43dc32769691f3fe1563f503f6",
}

FOUR_SIDED = ("rectangle", "diamond", "trapezoid", "parallelogram")
DIAGONAL = ("triangle_iso", "triangle_right", "diamond", "trapezoid",
            "parallelogram", "bevel5", "bevel6", "bevel7", "bevel8")
ALL_KINDS = FOUR_SIDED + ("triangle_iso", "triangle_right", "l_shape",
                           "bevel5", "bevel6", "bevel7", "bevel8")


def _resolve_counts(rng, num_train, num_test):
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer))):
            raise ValueError(f"{name} must be an integer or None")
    if num_train is not None:
        num_train = int(num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")
    if num_test is not None:
        num_test = int(num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
    elif num_train is None and num_test is None:
        options = ((2, 1), (3, 1), (4, 2), (4, 3), (5, 2),
                   (6, 3), (7, 2), (8, 4), (9, 3), (10, 4))
        num_train, num_test = options[int(rng.integers(len(options)))]
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _side_count(kind):
    if kind.startswith("bevel"):
        return int(kind[-1])
    if kind.startswith("triangle"):
        return 3
    return 6 if kind == "l_shape" else 4


def _plan(rng, train_count, test_count):
    source_like = train_count >= 3 and rng.random() < 0.2
    companion = not source_like and train_count >= 4 and rng.random() < 0.42
    shared_box = None
    if source_like:
        train = ["rectangle", "triangle_iso", "bevel6"]
        test = ["diamond", "bevel5", "bevel8"][:test_count]
    elif companion:
        diagonal = _choice(rng, DIAGONAL)
        train = ["rectangle", "l_shape", "bevel5" if diagonal == "diamond" else diagonal]
        shared_box = (int(rng.integers(10, 21)), int(rng.integers(10, 21)),
                      int(rng.integers(2)), int(rng.integers(4)))
        test = [_choice(rng, ("diamond", "trapezoid", "parallelogram"))]
    else:
        train = ["rectangle", _choice(rng, ("triangle_iso", "triangle_right",
                                          "bevel5", "bevel6", "bevel7", "bevel8"))]
        test = [_choice(rng, ("diamond", "trapezoid", "parallelogram"))]
    while len(train) < train_count:
        train.append(_choice(rng, ALL_KINDS))
    if len(test) < test_count:
        missing = [kind for kind in ("bevel5", "bevel6", "bevel7", "bevel8")
                   if int(kind[-1]) not in {_side_count(k) for k in train}]
        test.append(_choice(rng, missing or ("bevel7", "bevel8", "l_shape")))
    while len(test) < test_count:
        test.append(_choice(rng, ALL_KINDS))
    return train, test, shared_box


def _vertices(kind, rng, box=None):
    if kind in ("rectangle", "l_shape"):
        width, height = box[:2] if box is not None else (
            int(rng.integers(7, 23)), int(rng.integers(7, 23)))
        if kind == "rectangle":
            points = [(0, 0), (width - 1, 0),
                      (width - 1, height - 1), (0, height - 1)]
        else:
            x = int(rng.integers(3, width - 3))
            y = int(rng.integers(3, height - 3))
            points = [(0, 0), (width - 1, 0), (width - 1, y),
                      (x, y), (x, height - 1), (0, height - 1)]
    elif kind in ("triangle_iso", "triangle_right", "diamond"):
        arm = int(rng.integers(3, 12))
        if kind == "triangle_iso":
            points = [(arm, 0), (2 * arm, arm), (0, arm)]
        elif kind == "triangle_right":
            points = [(0, 0), (arm, 0), (0, arm)]
        else:
            points = [(arm, 0), (2 * arm, arm), (arm, 2 * arm), (0, arm)]
    elif kind in ("trapezoid", "parallelogram"):
        height = int(rng.integers(3, 9))
        width = int(rng.integers(height + 4, 22 - height))
        if kind == "trapezoid":
            points = [(height, 0), (width, 0), (width, height), (0, height)]
        else:
            points = [(height, 0), (height + width, 0),
                      (width, height), (0, height)]
    elif kind.startswith("bevel"):
        count = int(kind[-1]) - 4
        width = int(rng.integers(9, 23))
        height = int(rng.integers(9, 23))
        corners = rng.choice(4, size=count, replace=False)
        for _ in range(60):
            cuts = [0, 0, 0, 0]
            for corner in corners:
                cuts[int(corner)] = int(rng.integers(3, 8))
            tl, tr, br, bl = cuts
            if (width - tl - tr >= 3 and height - tr - br >= 3 and
                    width - br - bl >= 3 and height - bl - tl >= 3):
                break
        else:
            cuts = [3 if i in corners else 0 for i in range(4)]
            tl, tr, br, bl = cuts
        points = [(tl, 0), (width - tr, 0), (width, tr),
                  (width, height - br), (width - br, height),
                  (bl, height), (0, height - bl), (0, tl)]
        points = [p for i, p in enumerate(points) if p != points[i - 1]]
    else:
        raise ValueError(f"unknown polygon kind: {kind}")
    flip = box[2] if box is not None else int(rng.integers(2))
    turns = box[3] if box is not None else int(rng.integers(4))
    if flip:
        points = [(-x, y) for x, y in points]
    for _ in range(turns):
        points = [(-y, x) for x, y in points]
    min_x, min_y = min(x for x, _ in points), min(y for _, y in points)
    return [(x - min_x, y - min_y) for x, y in points]


def _segment(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    assert dx == 0 or dy == 0 or abs(dx) == abs(dy)
    steps = max(abs(dx), abs(dy))
    assert steps >= 3
    sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    return [(start[0] + i * sx, start[1] + i * sy)
            for i in range(steps + 1)]


def _margins(rng, span):
    spare = 30 - span
    assert spare >= 0
    if spare == 0:
        return 0, 0
    if spare >= 6 and rng.random() < 0.28:
        extra = int(rng.integers(6, min(14, spare) + 1))
        before = int(rng.integers(3, extra - 2))
        return before, extra - before
    if rng.random() < 0.46:
        after = int(rng.integers(min(10, spare) + 1))
        return (0, after) if rng.random() < 0.5 else (after, 0)
    if spare >= 2:
        extra = int(rng.integers(2, min(12, spare) + 1))
        before = int(rng.integers(1, extra))
        return before, extra - before
    return 0, 1


def _palette(rng, result_color, fixed=None):
    if fixed is not None:
        return fixed
    if result_color not in (0, 1) and rng.random() < 0.19:
        return 0, 1
    available = [c for c in range(10) if c != result_color]
    choices = rng.choice(available, size=2, replace=False)
    return int(choices[0]), int(choices[1])


def _make_pair(rng, kind, palette, result_color, box=None):
    vertices = _vertices(kind, rng, box)
    pixels = set()
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        pixels.update(_segment(a, b))
    span_x = max(x for x, _ in pixels) + 1
    span_y = max(y for _, y in pixels) + 1
    left, right = _margins(rng, span_x)
    top, bottom = _margins(rng, span_y)
    background, outline = palette
    grid = np.full((top + span_y + bottom, left + span_x + right),
                   background, dtype=np.int8)
    for x, y in pixels:
        grid[top + y, left + x] = outline
    if np.count_nonzero(grid == background) <= len(pixels):
        if grid.shape[0] < 30:
            grid = np.pad(grid, ((0, 1), (0, 0)), constant_values=background)
        if np.count_nonzero(grid == background) <= len(pixels) and grid.shape[1] < 30:
            grid = np.pad(grid, ((0, 0), (0, 1)), constant_values=background)
    output = np.full((1, len(vertices)), result_color, dtype=np.int8)
    return {"input": grid, "output": output}


def _input_key(grid):
    return tuple(grid.shape), grid.tobytes()


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def _inspect_input(grid):
    _check_grid(grid)
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2 and counts[0] != counts[1]
    bg = int(colors[int(np.argmax(counts))])
    fg = int(colors[int(np.argmin(counts))])
    points = {(int(x), int(y)) for y, x in np.argwhere(grid == fg)}
    runs = []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        for x, y in points:
            if (x - dx, y - dy) in points:
                continue
            run = []
            while (x, y) in points:
                run.append((x, y))
                x += dx
                y += dy
            if len(run) >= 4:
                runs.append(run)
    assert 3 <= len(runs) <= 8
    use = {p: 0 for p in points}
    links = {}
    for run in runs:
        for point in run:
            use[point] += 1
        a, b = run[0], run[-1]
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    assert all(n in (1, 2) for n in use.values())
    assert all(len(links.get(p, ())) == n for p, n in use.items() if n == 2)
    assert all(n == 1 for p, n in use.items() if p not in links)
    assert len(links) == len(runs)
    assert all(len(neighbors) == 2 for neighbors in links.values())
    start = next(iter(links))
    ordered = [start]
    previous = None
    current = start
    while True:
        neighbors = links[current]
        following = neighbors[0] if neighbors[0] != previous else neighbors[1]
        if following == start:
            break
        assert following not in ordered
        ordered.append(following)
        previous, current = current, following
    assert len(ordered) == len(runs)
    crosses = []
    for i in range(len(ordered)):
        a, b, c = ordered[i - 1], ordered[i], ordered[(i + 1) % len(ordered)]
        cross = ((b[0] - a[0]) * (c[1] - b[1]) -
                 (b[1] - a[1]) * (c[0] - b[0]))
        assert cross != 0
        crosses.append(cross)
    concave = min(crosses) < 0 < max(crosses)
    for i, first in enumerate(runs):
        for second in runs[i + 1:]:
            shared = set(first) & set(second)
            assert not shared or (len(shared) == 1 and
                                  next(iter(shared)) in links)
            if not shared:
                assert min(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                           for a in first for b in second) >= 2
    min_x, max_x = min(x for x, _ in points), max(x for x, _ in points)
    min_y, max_y = min(y for _, y in points), max(y for _, y in points)
    edge_sides = sum((min_x == 0, min_y == 0,
                      max_x == grid.shape[1] - 1,
                      max_y == grid.shape[0] - 1))
    directions = tuple(sorted((abs(run[-1][0] - run[0][0]) > 0,
                               abs(run[-1][1] - run[0][1]) > 0)
                              for run in runs))
    return {"background": bg, "outline": fg, "sides": len(runs),
            "concave": concave, "pixels": len(points),
            "bbox": (max_x - min_x + 1, max_y - min_y + 1),
            "edge_sides": edge_sides,
            "has_diagonal": any(x and y for x, y in directions),
            "directions": directions}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    result_color = int(rng.integers(10))
    multi_palette = rng.random() < 0.7
    shared_palette = None if multi_palette else _palette(rng, result_color)
    train, test, box = _plan(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    used = set()
    first_palette = None
    for regime, kinds in (("train", train), ("test", test)):
        for index, kind in enumerate(kinds):
            for _ in range(120):
                palette = _palette(rng, result_color, shared_palette)
                if multi_palette and regime == "train" and index == 1 and palette == first_palette:
                    continue
                pair_box = box if regime == "train" and index < 2 else None
                pair = _make_pair(rng, kind, palette, result_color, pair_box)
                key = _input_key(pair["input"])
                if key in used:
                    continue
                try:
                    features = _inspect_input(pair["input"])
                except AssertionError:
                    continue
                if features["sides"] != _side_count(kind):
                    continue
                if hashlib.sha256(pair["input"].tobytes()).hexdigest() in SOURCE_INPUTS:
                    continue
                if first_palette is None:
                    first_palette = palette
                puzzle[regime].append(pair)
                used.add(key)
                break
            else:
                raise RuntimeError(f"cannot place a valid {kind} outline")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    result_color = None
    seen = set()
    train_features = []
    test_features = []
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            _check_grid(output)
            features = _inspect_input(grid)
            assert output.shape == (1, features["sides"])
            assert bool(np.all(output == output[0, 0]))
            color = int(output[0, 0])
            assert color not in (features["background"], features["outline"])
            if result_color is None:
                result_color = color
            else:
                assert color == result_color
            key = _input_key(grid)
            assert key not in seen
            seen.add(key)
            assert hashlib.sha256(grid.tobytes()).hexdigest() not in SOURCE_INPUTS
            if regime == "train":
                train_features.append(features)
            else:
                test_features.append(features)
    assert len({f["sides"] for f in train_features}) >= 2
    assert any(f["has_diagonal"] for f in train_features)
    assert any(not f["has_diagonal"] for f in train_features)
    assert any(f["sides"] == 4 and f["has_diagonal"] for f in test_features)


def _signature(puzzle):
    return tuple((regime, tuple((_input_key(p["input"]),
                                 _input_key(p["output"]))
                                for p in puzzle[regime]))
                 for regime in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, results = set(), set(), set()
    shapes, cropped, palettes, signatures = set(), set(), set(), set()
    sides, concave, edges, mixed = set(), 0, 0, 0
    same_count_different_shape = 0
    padded = zero_one = companions = new_seven = compact = 0
    four_geometries = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_signature(puzzle))
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        train = [_inspect_input(p["input"]) for p in puzzle["train"]]
        test = [_inspect_input(p["input"]) for p in puzzle["test"]]
        if len({(f["background"], f["outline"]) for f in train}) > 1:
            mixed += 1
        if any(a["sides"] == b["sides"] and
               (a["directions"], a["concave"]) !=
               (b["directions"], b["concave"])
               for a in train for b in test):
            same_count_different_shape += 1
        if any(a["sides"] != b["sides"] and a["bbox"] == b["bbox"] and
               a["pixels"] == b["pixels"] for a in train for b in train if a is not b):
            companions += 1
        if any(f["sides"] == 7 for f in test) and not any(f["sides"] == 7 for f in train):
            new_seven += 1
        compact += (len(puzzle["train"]), len(puzzle["test"])) == (4, 3)
        for pair, f in zip(puzzle["train"] + puzzle["test"], train + test):
            grid = pair["input"]
            results.add(int(pair["output"][0, 0]))
            shapes.add(tuple(grid.shape))
            palettes.add((f["background"], f["outline"]))
            sides.add(f["sides"])
            concave += f["concave"]
            edges += f["edge_sides"] > 0
            padded += f["edge_sides"] == 0
            zero_one += (f["background"], f["outline"]) == (0, 1)
            if f["sides"] == 4:
                four_geometries.add(f["directions"])
            ys, xs = np.where(grid == f["outline"])
            crop = grid[ys.min():ys.max()+1, xs.min():xs.max()+1] == f["outline"]
            cropped.add((f["sides"], tuple(crop.shape), crop.tobytes()))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(results) >= 7
    assert len(shapes) >= 45 and len(cropped) >= 90
    assert len(palettes) >= 20 and sides == {3, 4, 5, 6, 7, 8}
    assert concave >= 5 and edges >= 30 and mixed >= 15
    assert same_count_different_shape >= 20
    assert padded >= 50 and zero_one >= 20 and len(four_geometries) >= 4
    assert companions >= 5 and new_seven >= 3 and compact >= 1
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (4, 3)):
        p = generate_puzzle(1000 + 10 * train_count + test_count,
                            train_count, test_count)
        assert (len(p["train"]), len(p["test"])) == (train_count, test_count)
    for args in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(2000 + (args[0] or 0) + (args[1] or 0), *args)
        assert args[0] is None or len(p["train"]) == args[0]
        assert args[1] is None or len(p["test"]) == args[1]
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (2, 4), (True, 1)):
        try:
            generate_puzzle(3000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
