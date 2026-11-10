"""Solve a framed maze once, then transport its completed drawing by square anchors.

Only Python's standard library and NumPy are used. No files, parent generators,
seed-dependent solver state, or saved examples participate in solving.
"""
from collections import deque
import hashlib
import numpy as np


HOST_SEEDS = (
    0, 438941388, 999268072, 743744794, 119070106, 513259643,
    195354677, 746332259, 618449225, 395554442, 853751066, 400163972,
    957617984, 451586253, 421288558, 419673785, 850047199, 474003418,
    75776913, 656893437, 569148802, 244025891, 153232599, 705343351,
    238736639, 357730680, 661010904, 627445407, 279534005, 257566236,
    650367052, 821484505, 935932062, 732655433, 439693546, 260318903,
    790509052, 192318633, 517675465, 326765539, 780986066, 160482545,
)
_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _array(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), "invalid dimensions"
    assert np.issubdtype(g.dtype, np.integer), "noninteger grid"
    assert np.all((g >= 0) & (g <= 9)), "invalid color"
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.uint8).tobytes()


def _shortest(passable, start, goal):
    """BFS with counts capped at two: ties are rejected, never broken by order."""
    h, w = passable.shape
    distance = {start: 0}
    ways, parent = {start: 1}, {}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _STEPS:
            q = r + dr, c + dc
            if not (0 <= q[0] < h and 0 <= q[1] < w and passable[q]):
                continue
            nd = distance[r, c] + 1
            if q not in distance:
                distance[q], ways[q], parent[q] = nd, ways[r, c], (r, c)
                queue.append(q)
            elif distance[q] == nd:
                ways[q] = min(2, ways[q] + ways[r, c])
    if goal not in distance:
        return [], 0
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1], ways[goal]


def _turns(path):
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _has_longer_route(passable, path):
    # A surviving route after removing a shortest-route cell is necessarily longer
    # when the original minimum is unique. This tests actual minimization.
    for cell in path[1:-1]:
        modified = passable.copy()
        modified[cell] = False
        other, _ = _shortest(modified, path[0], path[-1])
        if other:
            return len(other) > len(path)
    return False


def _bounds(h, w, r, c):
    return ((0, r, 0, c), (0, r, c + 1, w),
            (r + 1, h, 0, c), (r + 1, h, c + 1, w))


def _square(panel, color):
    cells = np.argwhere(panel == color)
    assert len(cells), "missing square"
    top, left = map(int, cells.min(0))
    bottom, right = map(int, cells.max(0) + 1)
    side = bottom - top
    assert side == right - left and len(cells) == side * side, "not one solid square"
    return top, left, side


def _decode(grid):
    rows = [r for r in range(1, grid.shape[0] - 1) if np.all(grid[r] == grid[r, 0])]
    cols = [c for c in range(1, grid.shape[1] - 1) if np.all(grid[:, c] == grid[0, c])]
    crosses = [(r, c) for r in rows for c in cols if grid[r, 0] == grid[0, c]]
    assert len(crosses) == 1, "expected one full monochrome cross"
    row, col = crosses[0]
    ink = int(grid[row, col])
    bounds = _bounds(*grid.shape, row, col)
    panels = [grid[r0:r1, c0:c1] for r0, r1, c0, c1 in bounds]
    colors = [set(map(int, np.unique(p))) for p in panels]
    references = [i for i, cs in enumerate(colors) if len(cs) > 2]
    assert len(references) == 1, "expected one multicolor reference"
    ref = references[0]
    common = set.intersection(*colors)
    assert len(common) == 2 and ink not in common
    candidates = []
    for color in sorted(common):
        try:
            anchors = [_square(p, color) for p in panels]
            candidates.append((color, anchors))
        except AssertionError:
            pass
    assert len(candidates) == 1, "anchor must be geometrically unambiguous"
    anchor, anchors = candidates[0]
    bg = next(iter(common - {anchor}))
    assert all(colors[i] == {bg, anchor} for i in range(4) if i != ref)
    source = panels[ref]
    assert ink not in colors[ref], "divider ink occurs only on the input cross"
    extras = colors[ref] - {bg, anchor}
    terminals = [c for c in sorted(extras) if np.count_nonzero(source == c) == 2]
    assert len(terminals) == 1, "exactly two singleton terminals required"
    terminal = terminals[0]
    walls = extras - {terminal}
    assert walls and all(np.count_nonzero(source == c) >= 3 for c in walls)
    board_cells = np.argwhere((source != bg) & (source != anchor))
    bt, bl = map(int, board_cells.min(0))
    bb, br = map(int, board_cells.max(0) + 1)
    board = source[bt:bb, bl:br]
    assert board.shape[0] >= 5 and board.shape[1] >= 5
    assert all(int(x) in walls for edge in (board[0], board[-1], board[:, 0], board[:, -1]) for x in edge), "board must be enclosed by walls"
    ar, ac, side = anchors[ref]
    attached = ((ar + side == bt or ar == bb) and bl <= ac and ac + side <= br or
                (ac + side == bl or ac == br) and bt <= ar and ar + side <= bb)
    assert attached, "source anchor must attach outside one frame edge"
    endpoints = [tuple(map(int, p)) for p in np.argwhere(source == terminal)]
    assert all(bt < r < bb - 1 and bl < c < br - 1 for r, c in endpoints)
    passable = (source == bg) | (source == terminal)
    path, count = _shortest(passable, *endpoints)
    assert count == 1 and len(path) > 2, "reference route must be uniquely shortest"
    assert all(source[p] == bg for p in path[1:-1])
    for _, _, t in anchors:
        assert t % side == 0, "nonintegral source-pixel scale"
    return {"bounds": bounds, "panels": panels, "reference": ref,
            "background": bg, "ink": ink, "anchor": anchor, "terminal": terminal,
            "walls": walls, "anchors": anchors, "path": path, "passable": passable,
            "board": (bt, bb, bl, br)}


def _raster(data, destination, *, route=True, reflection=True, scaling=True):
    source = data["panels"][data["reference"]].copy()
    if route:
        for p in data["path"][1:-1]:
            source[p] = data["ink"]
    occupied = np.argwhere(source != data["background"])
    top, left = map(int, occupied.min(0))
    bottom, right = map(int, occupied.max(0) + 1)
    tile = source[top:bottom, left:right].copy()
    ar, ac, side = data["anchors"][data["reference"]]
    offset = [ar - top, ac - left]
    if reflection and destination // 2 != data["reference"] // 2:
        tile = tile[::-1]
        offset[0] = tile.shape[0] - offset[0] - side
    if reflection and destination % 2 != data["reference"] % 2:
        tile = tile[:, ::-1]
        offset[1] = tile.shape[1] - offset[1] - side
    k = data["anchors"][destination][2] // side if scaling else 1
    return np.repeat(np.repeat(tile, k, axis=0), k, axis=1), (offset[0] * k, offset[1] * k)


def _stamp(panel, tile, top, left, bg):
    r0, c0 = max(0, top), max(0, left)
    r1, c1 = min(panel.shape[0], top + tile.shape[0]), min(panel.shape[1], left + tile.shape[1])
    if r0 < r1 and c0 < c1:
        patch = tile[r0 - top:r1 - top, c0 - left:c1 - left]
        target = panel[r0:r1, c0:c1]
        mask = patch != bg
        target[mask] = patch[mask]


def _render(grid, data, *, route=True, reflection=True, scaling=True):
    out = grid.copy()
    ref = data["reference"]
    for i, (r0, r1, c0, c1) in enumerate(data["bounds"]):
        if i == ref:
            if route:
                for r, c in data["path"][1:-1]:
                    out[r0 + r, c0 + c] = data["ink"]
            continue
        tile, offset = _raster(data, i, route=route, reflection=reflection, scaling=scaling)
        ar, ac, _ = data["anchors"][i]
        _stamp(out[r0:r1, c0:c1], tile, ar - offset[0], ac - offset[1], data["background"])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, the route, and every copy solely from the given input."""
    grid = _array(input_grid)
    return _render(grid, _decode(grid))


def _maze(rng, low, high):
    """Unscheduled random fields and segment unions, accepted by graph properties."""
    for _ in range(1800):
        h, w = map(int, rng.integers(low, high + 1, size=2))
        mode = int(rng.integers(3))
        wall = rng.random((h, w)) < rng.uniform(.23, .43)
        if mode:
            wall[:] = False
            for _ in range(int(rng.integers(2, h + w))):
                r, c = int(rng.integers(h)), int(rng.integers(w))
                length = int(rng.integers(1, max(h, w)))
                if rng.integers(2):
                    wall[r:min(h, r + length), c] = True
                else:
                    wall[r, c:min(w, c + length)] = True
            if mode == 2:
                # Holes and wall branches break a pure bar/slalom vocabulary.
                wall ^= rng.random((h, w)) < .12
        free = np.argwhere(~wall)
        if len(free) < 8:
            continue
        for _ in range(10):
            indices = rng.choice(len(free), size=2, replace=False)
            start, goal = (tuple(map(int, free[i])) for i in indices)
            manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            if manhattan < 2:
                continue
            path, count = _shortest(~wall, start, goal)
            if (count != 1 or len(path) - 1 <= manhattan or _turns(path) < 2
                    or len(path) - 1 < (10 if low >= 6 else 6)):
                continue
            if len(free) - len(path) < 3 or not _has_longer_route(~wall, path):
                continue
            return wall, start, goal, path
    raise RuntimeError("bounded maze search exhausted")


def _prototype(rng, low, high, source_side=None):
    wall, start, goal, path = _maze(rng, low, high)
    # All roles are pair-local. Divider color doubles as the visible ink sample.
    palette = list(map(int, rng.permutation(10)))
    bg, ink, anchor, terminal = palette[:4]
    wall_colors = palette[4:4 + int(rng.integers(1, 4))]
    h, w = wall.shape
    sampled_side = int(rng.choice((1, 1, 2)))
    side = sampled_side if source_side is None else source_side
    board = np.full((h + 2, w + 2), wall_colors[0], dtype=np.int8)
    board[1:-1, 1:-1] = np.where(wall, wall_colors[0], bg)
    # Color connected or repeated wall runs without changing traversability.
    for color in wall_colors[1:]:
        available = np.argwhere(board == wall_colors[0])
        if len(available) < 8:
            break
        amount = int(rng.integers(3, min(len(available) - 3, 10) + 1))
        for p in available[rng.choice(len(available), size=amount, replace=False)]:
            board[tuple(p)] = color
    board[start[0] + 1, start[1] + 1] = terminal
    board[goal[0] + 1, goal[1] + 1] = terminal
    augmented = board.copy()
    for r, c in path[1:-1]:
        augmented[r + 1, c + 1] = ink
    # Attach the anchor to any edge at a varying position, independently of maze.
    edge = int(rng.integers(4))
    vertical = edge < 2
    bh, bw = board.shape
    shape = (bh + side, bw) if vertical else (bh, bw + side)
    raw, finished = (np.full(shape, bg, dtype=np.int8) for _ in range(2))
    if edge == 0:
        bt, bl, ar, ac = side, 0, 0, int(rng.integers(bw - side + 1))
    elif edge == 1:
        bt, bl, ar, ac = 0, 0, bh, int(rng.integers(bw - side + 1))
    elif edge == 2:
        bt, bl, ar, ac = 0, side, int(rng.integers(bh - side + 1)), 0
    else:
        bt, bl, ar, ac = 0, 0, int(rng.integers(bh - side + 1)), bw
    for target, drawing in ((raw, board), (finished, augmented)):
        target[bt:bt + bh, bl:bl + bw] = drawing
        target[ar:ar + side, ac:ac + side] = anchor
    return raw, finished, (ar, ac, side), (bg, ink, anchor, terminal)


def _prefix(mask):
    return np.pad(mask.astype(int).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _visible(prefix, shape, top, left):
    h, w = prefix.shape[0] - 1, prefix.shape[1] - 1
    r0, c0 = max(0, -top), max(0, -left)
    r1, c1 = min(h, shape[0] - top), min(w, shape[1] - left)
    if r0 >= r1 or c0 >= c1:
        return 0
    return int(prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0])


def _placement(rng, shape, tile, offset, side, bg, ink, clipped):
    colored, route = _prefix(tile != bg), _prefix(tile == ink)
    total = int(colored[-1, -1])
    candidates = []
    for ar in range(shape[0] - side + 1):
        for ac in range(shape[1] - side + 1):
            top, left = ar - offset[0], ac - offset[1]
            n = _visible(colored, shape, top, left)
            if (n < total) == clipped and _visible(route, shape, top, left) >= 2:
                candidates.append((ar, ac))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _axis(rng, minimum):
    if sum(minimum) + 1 > 30:
        return None
    extra = int(rng.integers(min(5, 30 - sum(minimum))))
    first = int(rng.integers(extra + 1))
    return minimum[0] + first, minimum[1] + extra - first


def _build_pair(rng, reference, clipped_count, test=False, witness=False, source_side=None):
    raw, finished, (ar, ac, side), (bg, ink, anchor, terminal) = _prototype(
        rng, 6 if test else 3, 8 if test else 5, source_side)
    if source_side is not None and any(np.array_equal(finished, variant) for variant in
                                      (finished[::-1], finished[:, ::-1], finished[::-1, ::-1])):
        return None
    destinations = [p for p in range(4) if p != reference]
    factors = list(map(int, rng.permutation([1, 2, 1 if witness else int(rng.choice((1, 2, 3)))])))
    units = {reference: 1, **dict(zip(destinations, factors))}
    clip_panels = set(map(int, rng.choice(destinations, size=clipped_count, replace=False)))
    rasters, offsets = {}, {}
    for p in range(4):
        tile, offset = finished.copy(), [ar, ac]
        if reference // 2 != p // 2:
            tile = tile[::-1]
            offset[0] = tile.shape[0] - offset[0] - side
        if reference % 2 != p % 2:
            tile = tile[:, ::-1]
            offset[1] = tile.shape[1] - offset[1] - side
        k = units[p]
        rasters[p] = np.repeat(np.repeat(tile, k, 0), k, 1)
        offsets[p] = offset[0] * k, offset[1] * k
    minrows, mincols = [5, 5], [5, 5]
    for p in range(4):
        shape = (max(5, side * units[p] + 2),) * 2 if p in clip_panels else rasters[p].shape
        minrows[p // 2] = max(minrows[p // 2], shape[0])
        mincols[p % 2] = max(mincols[p % 2], shape[1])
    rows, cols = _axis(rng, minrows), _axis(rng, mincols)
    if rows is None or cols is None:
        return None
    bounds = _bounds(sum(rows) + 1, sum(cols) + 1, rows[0], cols[0])
    inp = np.full((sum(rows) + 1, sum(cols) + 1), bg, dtype=np.int8)
    inp[rows[0], :] = ink
    inp[:, cols[0]] = ink
    out = inp.copy()
    for p, (r0, r1, c0, c1) in enumerate(bounds):
        tile, offset = rasters[p], offsets[p]
        anchor_side = side * units[p]
        if p == reference:
            top = int(rng.integers(r1 - r0 - raw.shape[0] + 1))
            left = int(rng.integers(c1 - c0 - raw.shape[1] + 1))
            _stamp(inp[r0:r1, c0:c1], raw, top, left, bg)
            _stamp(out[r0:r1, c0:c1], finished, top, left, bg)
        else:
            where = _placement(rng, (r1 - r0, c1 - c0), tile, offset,
                               anchor_side, bg, ink, p in clip_panels)
            if where is None:
                return None
            ay, ax = where
            inp[r0 + ay:r0 + ay + anchor_side, c0 + ax:c0 + ax + anchor_side] = anchor
            out[r0 + ay:r0 + ay + anchor_side, c0 + ax:c0 + ax + anchor_side] = anchor
            _stamp(out[r0:r1, c0:c1], tile, ay - offset[0], ax - offset[1], bg)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    references = list(map(int, rng.permutation(4)))
    source_sides = list(map(int, rng.permutation([1, 2])))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            test = split == "test"
            witness = not test and index == 0
            reference = references[index] if not test and index < 2 else int(rng.integers(4))
            clips = 0 if witness else 1 if not test and index == 1 else int(rng.integers(1 if test else 0, 4))
            for _ in range(160):
                source_side = source_sides[index] if not test and index < 2 else None
                pair = _build_pair(rng, reference, clips, test, witness, source_side)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded panel-placement search exhausted")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _array(pair["input"]), _array(pair["output"])
    assert inp.shape == out.shape
    data = _decode(inp)
    assert np.array_equal(out, _render(inp, data)), "wrong route or transported drawing"
    assert np.array_equal(out[inp != data["background"]], inp[inp != data["background"]]), "input objects changed"
    path = data["path"]
    detour = len(path) - 1 - sum(abs(a - b) for a, b in zip(path[0], path[-1]))
    assert detour > 0 and _turns(path) >= 2
    assert _has_longer_route(data["passable"], path), "shortest-path choice not exercised"
    bt, bb, bl, br = data["board"]
    source = data["panels"][data["reference"]]
    assert np.count_nonzero(source[bt:bb, bl:br] == data["background"]) >= len(path) + 1
    clips, factors, route_counts = [], [], []
    for p, (r0, r1, c0, c1) in enumerate(data["bounds"]):
        if p == data["reference"]:
            continue
        tile, offset = _raster(data, p)
        ar, ac, side = data["anchors"][p]
        top, left = ar - offset[0], ac - offset[1]
        shape = (r1 - r0, c1 - c0)
        visible = _visible(_prefix(tile != data["background"]), shape, top, left)
        ink = _visible(_prefix(tile == data["ink"]), shape, top, left)
        assert ink >= 2, "destination lost all meaningful route evidence"
        assert np.count_nonzero(out[r0:r1, c0:c1] == data["ink"]) == ink
        clips.append(visible < np.count_nonzero(tile != data["background"]))
        factors.append(side // data["anchors"][data["reference"]][2])
        route_counts.append(ink)
    data.update(clipped_count=int(sum(clips)), factors=factors,
                interior_shape=(bb - bt - 2, br - bl - 2),
                route_length=len(path) - 1, detour=detour, route_counts=route_counts)
    assert not np.array_equal(out == data["ink"], _render(inp, data, route=False) == data["ink"]), "B is unnecessary"
    assert not np.array_equal(out == data["ink"], _render(inp, data, reflection=False) == data["ink"]), "reflection is unnecessary"
    assert not np.array_equal(out == data["ink"], _render(inp, data, scaling=False) == data["ink"]), "anchor scale is unnecessary"
    return data


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _analyze(pair)
            key = _key(_array(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            analyses[split].append(analysis)
    train = analyses["train"]
    assert len({d["reference"] for d in train}) >= 2, "relative-reference demonstrations missing"
    assert {d["anchors"][d["reference"]][2] for d in train} == {1, 2}, "anchor-ratio demonstrations missing"
    assert any(d["clipped_count"] == 0 and max(d["factors"]) > 1 for d in train)
    assert any(d["clipped_count"] > 0 for d in train), "clipping demonstration missing"
    max_train_area = max(np.prod(d["interior_shape"]) for d in train)
    assert all(np.prod(d["interior_shape"]) > max_train_area for d in analyses["test"]), "missing larger-board test shift"


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = _array(pair[role])
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded checks: 42 puzzles retained one at a time; no disk writes."""
    signatures, shapes, references, palettes, maze_shapes, lengths = (set() for _ in range(6))
    clipping, factors, sides = set(), set(), set()
    maze_structures, wall_counts, wall_palettes, anchor_edges = set(), set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = _array(pair["input"])
                before = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, before), "solve mutated input"
                d = _analyze(pair)
                shapes.add(inp.shape)
                references.add(d["reference"])
                palettes.add((d["background"], d["ink"], d["anchor"], d["terminal"]))
                maze_shapes.add(d["interior_shape"])
                lengths.add(d["route_length"])
                clipping.add(d["clipped_count"])
                factors.update(d["factors"])
                sides.add(d["anchors"][d["reference"]][2])
                bt, bb, bl, br = d["board"]
                source = d["panels"][d["reference"]]
                interior = source[bt + 1:bb - 1, bl + 1:br - 1]
                normalized = np.where(interior == d["background"], 0,
                                      np.where(interior == d["terminal"], 2, 1)).astype(np.int8)
                maze_structures.add(_key(normalized))
                wall_counts.add(int(np.count_nonzero(normalized == 1)))
                wall_palettes.add(len(d["walls"]))
                ar, ac, a = d["anchors"][d["reference"]]
                anchor_edges.add("top" if ar + a == bt else "bottom" if ar == bb else
                                 "left" if ac + a == bl else "right")
        if seed == 0:
            assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
    assert len(signatures) == 42 and len(shapes) >= 25 and references == {0, 1, 2, 3}
    assert len(palettes) >= 30 and len(maze_shapes) >= 12 and len(lengths) >= 8
    assert clipping == {0, 1, 2, 3} and factors == {1, 2, 3} and sides == {1, 2}
    assert len(maze_structures) >= 150 and len(wall_counts) >= 8
    assert wall_palettes == {1, 2, 3} and anchor_edges == {"top", "bottom", "left", "right"}
    # An explicit small fixture checks path selection without relying on generation.
    passable = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool)
    path, count = _shortest(passable, (0, 0), (0, 3))
    assert count == 1 and path == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert _has_longer_route(passable, path)
    assert _shortest(np.ones((2, 2), bool), (0, 0), (1, 1))[1] == 2
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(123, 2, 1)
    p["test"][0]["output"] = p["test"][0]["output"].copy()
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
