import hashlib
import numpy as np


_DIRECTIONS = ("top", "bottom", "left", "right")
_OUTWARD = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}
_CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_SOURCE_HASHES = {
    "8128b126151047320ab364dec74feeab19142eccc58da4d7ea819bb625cc866b",
    "9aeb9c7bccc579416aeab7368bb6eca17022b63cd84a9ba39cadd5ee602ef200",
    "4d829a975b801b0f0e9c0e2eeedc7bd809ad55b67d934da43773d8a67a4bf7ac",
}


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(seed, num_train, num_test):
    if not _is_int(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if num_train is not None and (not _is_int(num_train) or not 2 <= num_train <= 10):
        raise ValueError("num_train must be 2..10")
    if num_test is not None and (not _is_int(num_test) or not 1 <= num_test <= 4):
        raise ValueError("num_test must be 1..4")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _palette(rng):
    colors = [int(v) for v in rng.choice(10, 7, replace=False)]
    return dict(zip(("neutral", "background", "fill", "top", "bottom", "left", "right"), colors))


def _blank(roles, height, width):
    grid = np.full((height, width), roles["background"], dtype=np.int8)
    grid[0, 1:-1] = roles["top"]
    grid[-1, 1:-1] = roles["bottom"]
    grid[1:-1, 0] = roles["left"]
    grid[1:-1, -1] = roles["right"]
    grid[0, 0] = grid[0, -1] = grid[-1, 0] = grid[-1, -1] = roles["neutral"]
    return grid


def _roles(grid):
    h, w = grid.shape
    assert h >= 14 and w >= 14
    neutral = int(grid[0, 0])
    assert grid[0, -1] == grid[-1, 0] == grid[-1, -1] == neutral
    sides = {"top": int(grid[0, 1]), "bottom": int(grid[-1, 1]),
             "left": int(grid[1, 0]), "right": int(grid[1, -1])}
    assert np.all(grid[0, 1:-1] == sides["top"])
    assert np.all(grid[-1, 1:-1] == sides["bottom"])
    assert np.all(grid[1:-1, 0] == sides["left"])
    assert np.all(grid[1:-1, -1] == sides["right"])
    interior = grid[1:-1, 1:-1]
    values, counts = np.unique(interior, return_counts=True)
    assert len(values) == 2
    # The empty inner rim identifies the background even when a large rectangle
    # occupies more cells than it. No single protected corner cell is assumed.
    rim = np.concatenate((interior[0], interior[-1], interior[:, 0], interior[:, -1]))
    background = int(np.bincount(rim.astype(int), minlength=10).argmax())
    assert background in values
    fill = int(values[0] if values[1] == background else values[1])
    roles = {"neutral": neutral, "background": background, "fill": fill, **sides}
    assert len(set(roles.values())) == 7
    return roles


def _cross(y, x):
    return {(y + dy, x + dx) for dy, dx in _CROSS}


def _notches(grid, background, fill):
    h, w = grid.shape
    found = []
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            if any(grid[yy, xx] != background for yy, xx in _cross(y, x)):
                continue
            for direction in _DIRECTIONS:
                dy, dx = _OUTWARD[direction]
                ty, tx = dx, -dy
                outside = ((y + dy + ty, x + dx + tx),
                           (y + dy - ty, x + dx - tx))
                support = ((y + 2 * ty, x + 2 * tx),
                           (y - 2 * ty, x - 2 * tx),
                           (y - dy + ty, x - dx + tx),
                           (y - dy - ty, x - dx - tx),
                           (y - 2 * dy, x - 2 * dx),
                           (y - 2 * dy + ty, x - 2 * dx + tx),
                           (y - 2 * dy - ty, x - 2 * dx - tx))
                if not all(grid[a, b] == background for a, b in outside):
                    continue
                if not all(grid[a, b] == fill for a, b in support):
                    continue
                far_y, far_x = y + 2 * dy, x + 2 * dx
                # The far cell may be the frame, as in two source examples.
                if 0 < far_y < h - 1 and 0 < far_x < w - 1:
                    if grid[far_y, far_x] != background:
                        continue
                found.append((direction, y, x))
    assert len({(y, x) for _, y, x in found}) == len(found)
    occupied = set()
    for _, y, x in found:
        cells = _cross(y, x)
        assert occupied.isdisjoint(cells)
        occupied.update(cells)
    return found


def _components(grid, fill):
    mask = grid == fill
    seen = set()
    boxes = []
    for y, x in zip(*np.nonzero(mask)):
        start = (int(y), int(x))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        cells = []
        while stack:
            a, b = stack.pop()
            cells.append((a, b))
            for point in ((a-1, b), (a+1, b), (a, b-1), (a, b+1)):
                if point not in seen and 0 <= point[0] < grid.shape[0] and 0 <= point[1] < grid.shape[1] and mask[point]:
                    seen.add(point)
                    stack.append(point)
        ys, xs = zip(*cells)
        boxes.append((min(ys), max(ys), min(xs), max(xs)))
    return boxes


def _inspect(grid):
    roles = _roles(grid)
    notches = _notches(grid, roles["background"], roles["fill"])
    assert notches
    return {"roles": roles, "notches": notches, "boxes": _components(grid, roles["fill"])}


def _solve(grid):
    info = _inspect(grid)
    result = grid.copy()
    for direction, y, x in info["notches"]:
        for a, b in _cross(y, x):
            result[a, b] = info["roles"][direction]
    return result


def _assemble(roles, height, width, rectangles, specified):
    grid = _blank(roles, height, width)
    for y0, y1, x0, x1 in rectangles:
        assert 2 <= y0 <= y1 <= height - 3
        assert 2 <= x0 <= x1 <= width - 3
        grid[y0:y1+1, x0:x1+1] = roles["fill"]
    occupied = set()
    for direction, y, x in specified:
        cells = _cross(y, x)
        assert occupied.isdisjoint(cells)
        assert all(1 <= a < height-1 and 1 <= b < width-1 for a, b in cells)
        dy, dx = _OUTWARD[direction]
        assert grid[y+dy, x+dx] == roles["background"]
        occupied.update(cells)
        for a, b in cells:
            grid[a, b] = roles["background"]
    info = _inspect(grid)
    assert set(info["notches"]) == set(specified)
    return grid, _solve(grid)


def _zones(rng, height, width, n):
    full = (2, height-3, 2, width-3)
    if n == 1:
        return [full]
    if n == 2:
        if rng.random() < 0.5:
            cut = int(rng.integers(11, height-10))
            return [(2, cut-1, 2, width-3), (cut+1, height-3, 2, width-3)]
        cut = int(rng.integers(11, width-10))
        return [(2, height-3, 2, cut-1), (2, height-3, cut+1, width-3)]
    cy = int(rng.integers(11, height-10))
    cx = int(rng.integers(11, width-10))
    zones = [(2, cy-1, 2, cx-1), (2, cy-1, cx+1, width-3),
             (cy+1, height-3, 2, cx-1), (cy+1, height-3, cx+1, width-3)]
    if n == 3:
        return [zones[int(i)] for i in rng.choice(4, 3, replace=False)]
    return zones


def _box(rng, zone):
    ya, yb, xa, xb = zone
    max_h, max_w = yb-ya+1, xb-xa+1
    min_h, min_w = min(7, max_h), min(7, max_w)
    assert min_h >= 6 and min_w >= 6
    h = int(rng.integers(min_h, min(max_h, 24)+1))
    w = int(rng.integers(min_w, min(max_w, 24)+1))
    y0 = ya if rng.random() < 0.27 else int(rng.integers(ya, yb-h+2))
    x0 = xa if rng.random() < 0.27 else int(rng.integers(xa, xb-w+2))
    return y0, y0+h-1, x0, x0+w-1


def _cluster(rng, box, directions, rectangles, notches):
    y0, y1, x0, x1 = box
    cy = int(rng.integers(y0+2, y1-1))
    cx = int(rng.integers(x0+2, x1-1))
    for direction in directions:
        if direction in ("top", "bottom"):
            left = int(rng.integers(2, min(12, cx-x0)+1))
            right = int(rng.integers(2, min(12, x1-cx)+1))
            rx0, rx1 = cx-left, cx+right
            nx = int(rng.integers(rx0+2, rx1-1))
            if direction == "top":
                ry0, ry1, ny = y0, min(y1, cy+int(rng.integers(0, 3))), y0
            else:
                ry0, ry1, ny = max(y0, cy-int(rng.integers(0, 3))), y1, y1
        else:
            above = int(rng.integers(2, min(12, cy-y0)+1))
            below = int(rng.integers(2, min(12, y1-cy)+1))
            ry0, ry1 = cy-above, cy+below
            ny = int(rng.integers(ry0+2, ry1-1))
            if direction == "left":
                rx0, rx1, nx = x0, min(x1, cx+int(rng.integers(0, 3))), x0
            else:
                rx0, rx1, nx = max(x0, cx-int(rng.integers(0, 3))), x1, x1
        rectangles.append((ry0, ry1, rx0, rx1))
        notches.append((direction, ny, nx))


def _make_pair(rng, split, index):
    roles = _palette(rng)
    if split == "train" and index == 0:
        directions = list(_DIRECTIONS)
    else:
        high = 8 if split == "train" else 9
        count = int(rng.integers(1 if split == "test" else 2, high))
        directions = [_DIRECTIONS[int(v)] for v in rng.choice(np.repeat(np.arange(4), 4), count, replace=False)]
    rng.shuffle(directions)
    count = len(directions)
    min_dimension = 25 if count >= 5 else (22 if count >= 3 else 18)
    height = int(rng.integers(min_dimension, 31))
    width = int(rng.integers(min_dimension, 31))
    # More than four bites need multiple clusters. A cluster may have several
    # directions, while separate clusters may repeat or omit directions.
    minimum_clusters = max(1, (count+3)//4, max(directions.count(d) for d in _DIRECTIONS))
    maximum_clusters = min(count, 4)
    for attempt in range(320):
        nclusters = int(rng.integers(minimum_clusters, maximum_clusters+1))
        groups = [[] for _ in range(nclusters)]
        ordered = sorted(directions, key=lambda d: -directions.count(d))
        for d in ordered:
            choices = [i for i, group in enumerate(groups) if len(group) < 4 and d not in group]
            if not choices:
                break
            smallest = min(len(groups[i]) for i in choices)
            choices = [i for i in choices if len(groups[i]) == smallest]
            groups[int(rng.choice(choices))].append(d)
        else:
            try:
                zones = _zones(rng, height, width, nclusters)
                rectangles, notches = [], []
                for zone, group in zip(zones, groups):
                    _cluster(rng, _box(rng, zone), group, rectangles, notches)
                return _assemble(roles, height, width, rectangles, notches)
            except (AssertionError, ValueError):
                pass
    # A fresh layout is tried at a larger scale before giving up; the direction
    # list and pair palette remain fixed.
    height = width = 30
    for attempt in range(320):
        try:
            nclusters = minimum_clusters
            groups = [[] for _ in range(nclusters)]
            ordered = sorted(directions, key=lambda d: -directions.count(d))
            for d in ordered:
                candidates = [i for i, group in enumerate(groups) if len(group) < 4 and d not in group]
                if not candidates:
                    raise ValueError("no cluster assignment")
                smallest = min(len(groups[i]) for i in candidates)
                candidates = [i for i in candidates if len(groups[i]) == smallest]
                groups[int(rng.choice(candidates))].append(d)
            zones = _zones(rng, height, width, nclusters)
            rectangles, notches = [], []
            for zone, group in zip(zones, groups):
                _cluster(rng, _box(rng, zone), group, rectangles, notches)
            return _assemble(roles, height, width, rectangles, notches)
        except (AssertionError, ValueError):
            pass
    raise RuntimeError("could not construct an unambiguous pair")


def _pair_hash(pair):
    inp, out = pair["input"], pair["output"]
    if inp.shape != (30, 30) or out.shape != (30, 30):
        return ""
    return hashlib.sha256(inp.astype(np.uint8).tobytes() + out.astype(np.uint8).tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    ntrain, ntest = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            for attempt in range(20):
                inp, out = _make_pair(rng, split, index)
                key = (inp.shape, inp.tobytes())
                pair = {"input": inp, "output": out}
                if key not in seen and _pair_hash(pair) not in _SOURCE_HASHES:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_directions = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            input_key = (inp.shape, inp.astype(np.uint8).tobytes())
            pair_key = (input_key, out.astype(np.uint8).tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _pair_hash(pair) not in _SOURCE_HASHES
            info = _inspect(inp)
            assert np.array_equal(out, _solve(inp)) and not np.array_equal(inp, out)
            assert len(info["notches"]) <= (7 if split == "train" else 8)
            if split == "train":
                train_directions.update(d for d, _, _ in info["notches"])
    assert train_directions == set(_DIRECTIONS)


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    palettes, shapes, structures = set(), set(), set()
    train_counts, test_counts, notch_counts, component_counts = set(), set(), set(), set()
    near_frame = repeated = omitted = mixed_palette = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                info = _inspect(inp)
                roles = tuple(info["roles"].values())
                pair_palettes.add(roles)
                palettes.add(roles)
                shapes.add(inp.shape)
                structures.add((inp.shape, (inp == info["roles"]["fill"]).tobytes()))
                notch_counts.add(len(info["notches"]))
                component_counts.add(len(info["boxes"]))
                directions = [d for d, _, _ in info["notches"]]
                repeated += len(directions) != len(set(directions))
                omitted += len(set(directions)) < 4
                near_frame += any((y == 2 or y == inp.shape[0]-3) if d in ("top", "bottom") else
                                  (x == 2 or x == inp.shape[1]-3) for d, y, x in info["notches"])
        mixed_palette += len(pair_palettes) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 100 and len(shapes) >= 20 and len(structures) >= 100
    assert len(notch_counts) >= 5 and len(component_counts) >= 3
    assert near_frame > 0 and repeated > 0 and omitted > 0 and mixed_palette == 42
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(47, *counts)
        validate_generated_puzzle(puzzle)
        for actual, expected in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            assert expected is None or actual == expected
    for counts in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2)):
        try:
            generate_puzzle(1, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    assert not _equal_puzzles(generate_puzzle(0), generate_puzzle(1))
