"""Glyph-constructed dividers and lower-left crossing-parity flood fill.

Only NumPy and the standard library are used. Generation metadata is never
consulted by solve. No samples, traces or caches are written by this module.
"""
from collections import deque
from functools import lru_cache
import hashlib
import json

import numpy as np


# Four source-A masks retain their exact meanings; the ring adds a longer
# vertical stroke, within the final A generator's straight-stroke vocabulary.
MASKS = (
    ((1, 1, 0), (1, 0, 1), (0, 1, 0)),  # R3
    ((1, 0, 1), (1, 0, 1), (1, 1, 1)),  # L2
    ((1, 0, 1), (0, 1, 0), (0, 1, 0)),  # D2
    ((1, 1, 1), (0, 1, 0), (1, 0, 1)),  # L4
    ((1, 1, 1), (1, 0, 1), (1, 1, 1)),  # D4
)
STROKES = ((0, 1, 3), (0, -1, 2), (1, 0, 2), (0, -1, 4), (1, 0, 4))
R3, L2, D2, L4, D4 = range(5)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "right", "bottom", "left")
HOST_SEEDS = (0, 368581431, 504657885, 458186105, 401985296, 694281363,
              759243135, 541912331, 304648902, 819340627, 823158573,
              177856320, 963653826, 261031611, 167415072, 587758319,
              480799168, 8188882, 569206433, 448182584, 922048968,
              575308463, 474629125, 981113076, 513960471, 688708709,
              306704791, 888107093, 642577085, 154385534, 589014172,
              116468650, 281036676, 995437865, 267217958, 436378047,
              118633851, 108520576, 374105336, 601363175, 943618091, 39287957)


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "nonintegral grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _global(depth, transverse, side, height, width):
    if side == "top":
        return depth, transverse
    if side == "right":
        return transverse, width - 1 - depth
    if side == "bottom":
        return height - 1 - depth, width - 1 - transverse
    return height - 1 - transverse, depth


def _local_start(start, height, width):
    r, c = start
    _require((r in (0, height - 1)) != (c in (0, width - 1)),
             "seed must lie on exactly one boundary side")
    if r == 0:
        return "top", c
    if c == width - 1:
        return "right", r
    if r == height - 1:
        return "bottom", width - 1 - c
    return "left", height - 1 - r


def _components(mask):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        index = len(groups)
        labels[r, c] = index
        group, pending = [], [(r, c)]
        while pending:
            y, x = pending.pop()
            group.append((y, x))
            for dy, dx in N4:
                qy, qx = y + dy, x + dx
                if (0 <= qy < h and 0 <= qx < w and mask[qy, qx]
                        and labels[qy, qx] < 0):
                    labels[qy, qx] = index
                    pending.append((qy, qx))
        groups.append(group)
    return labels, groups


def _geometry(canvas, field):
    """Check actual cell topology, rather than assuming a recipe separates."""
    h, w = canvas.shape
    wall = canvas != field
    _require(not wall[-1, 0], "lower-left anchor must stay open")
    owners, paths = _components(wall)
    _require(1 <= len(paths) <= 4, "invalid number of dividers")
    turns, boundary_counts, endpoint_sides = [], [], []
    for i, path in enumerate(paths):
        ink = {int(canvas[p]) for p in path}
        _require(len(ink) == 1, "divider colors touch")
        cells = set(path)
        ends = []
        for r, c in path:
            degree = sum((r + dy, c + dx) in cells for dy, dx in N4)
            _require(degree in (1, 2), "divider branches or self-touches")
            if degree == 1:
                ends.append((r, c))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y, x = r + dy, c + dx
                    if 0 <= y < h and 0 <= x < w:
                        _require(owners[y, x] in (-1, i), "dividers touch diagonally")
        _require(len(ends) == 2, "divider is not a simple open path")
        _require(all(r in (0, h - 1) or c in (0, w - 1) for r, c in ends),
                 "divider does not end on boundary")
        previous, current = None, ends[0]
        directions = []
        while current != ends[1]:
            neighbors = [(current[0] + dy, current[1] + dx) for dy, dx in N4
                         if (current[0] + dy, current[1] + dx) in cells
                         and (current[0] + dy, current[1] + dx) != previous]
            _require(len(neighbors) == 1, "ambiguous divider traversal")
            nxt = neighbors[0]
            directions.append((nxt[0] - current[0], nxt[1] - current[1]))
            previous, current = current, nxt
        turns.append(sum(a != b for a, b in zip(directions, directions[1:])))
        boundary_counts.append(sum(r in (0, h - 1) or c in (0, w - 1) for r, c in path))
        endpoint_sides.append(tuple(sorted(_local_start(p, h, w)[0] for p in ends)))
    labels, regions = _components(~wall)
    _require(len(regions) == len(paths) + 1, "ineffective divider or closed pocket")
    adjacency = [set() for _ in regions]
    for path in paths:
        neighbors = set()
        for r, c in path:
            for dy, dx in N4:
                y, x = r + dy, c + dx
                if 0 <= y < h and 0 <= x < w and labels[y, x] >= 0:
                    neighbors.add(int(labels[y, x]))
        _require(len(neighbors) == 2, "each divider must separate exactly two regions")
        a, b = sorted(neighbors)
        adjacency[a].add(b)
        adjacency[b].add(a)
    root = int(labels[-1, 0])
    depths = [-1] * len(regions)
    depths[root] = 0
    queue = deque([root])
    while queue:
        a = queue.popleft()
        for b in adjacency[a]:
            if depths[b] < 0:
                depths[b] = depths[a] + 1
                queue.append(b)
            else:
                _require((depths[a] - depths[b]) % 2 == 1, "inconsistent crossing parity")
    _require(min(depths) == 0, "disconnected region graph")
    _require(sum(map(len, adjacency)) == 2 * len(paths), "region graph must be a tree")
    through_count = sum((SIDES.index(a) - SIDES.index(b)) % 4 == 2
                        for a, b in endpoint_sides)
    return {"labels": labels, "depths": tuple(depths), "paths": paths,
            "root": root, "turns": tuple(turns), "boundary_counts": tuple(boundary_counts),
            "endpoint_sides": tuple(endpoint_sides),
            "degree": max(map(len, adjacency)), "max_depth": max(depths),
            "through_count": through_count, "corner_count": len(paths) - through_count}


def _draw(height, width, field, streams):
    canvas = np.full((height, width), field, dtype=np.int8)
    ordered_paths = []
    for color, start, commands in streams:
        side, col = _local_start(start, height, width)
        _require(canvas[start] == field, "duplicate seed")
        canvas[start] = color
        path = [start]
        depth = 0
        for command in commands:
            dr, dc, length = STROKES[command]
            for step in range(length):
                point = _global(depth + 1 + step * dr, col + step * dc,
                                side, height, width)
                r, c = point
                _require(0 <= r < height and 0 <= c < width, "stroke leaves canvas")
                _require(canvas[point] == field, "stroke overwrites an earlier cell")
                canvas[point] = color
                path.append(point)
            depth += 1 + dr * (length - 1)
            col += dc * (length - 1)
        ordered_paths.append(path)
    info = _geometry(canvas, field)
    _require(len(info["paths"]) == len(streams), "dividers merged")
    return canvas, info, ordered_paths


def _fill(canvas, field, fills, info):
    out = canvas.copy()
    for region, depth in enumerate(info["depths"]):
        out[info["labels"] == region] = fills[depth % 2]
    return out


def _read_input(input_grid, row_major=False):
    inp = _grid(input_grid)
    h, total_w = inp.shape
    candidates = []
    for lanes in (2, 3, 4):
        sep = 4 * lanes - 1
        if not (sep + 4 < total_w and h >= 5):
            continue
        if not np.all(inp[:, sep] == inp[0, sep]):
            continue
        pane = inp[:, sep + 1:]
        field = int(pane[-1, 0])
        separator = int(inp[0, sep])
        fills = (int(inp[0, 0]), int(inp[0, 1]))
        if len({field, separator, *fills}) != 4:
            continue
        marks = np.argwhere(pane != field)
        if not 1 <= len(marks) <= 4:
            continue
        try:
            starts = {}
            for r, c in marks:
                p = int(r), int(c)
                _local_start(p, h, pane.shape[1])
                color = int(pane[p])
                _require(color not in starts and color not in {field, separator, *fills},
                         "invalid seed colors")
                starts[color] = p
            expected = np.full(inp.shape, field, dtype=np.int8)
            expected[:, sep] = separator
            expected[0, 0:2] = fills
            expected[:, sep + 1:] = pane
            slots = {}
            max_band = -1
            for lane in range(lanes):
                for band in range((h - 1) // 4):
                    top, left = 2 + 4 * band, 4 * lane
                    block = inp[top:top + 3, left:left + 3]
                    ink = block[block != field]
                    if not len(ink):
                        continue
                    _require(len(np.unique(ink)) == 1 and int(ink[0]) in starts,
                             "glyph ink must match a boundary seed")
                    mask = tuple(tuple(int(x) for x in row) for row in block != field)
                    _require(mask in MASKS, "unknown glyph mask")
                    command = MASKS.index(mask)
                    color = int(ink[0])
                    slots[lane, band] = command, color
                    expected[top:top + 3, left:left + 3] = block
                    max_band = max(max_band, band)
            _require(max_band >= 0 and np.array_equal(inp, expected), "panel contains noise")
            bands = max_band + 1
            normal = [(lane, band) for lane in range(lanes) for band in range(bands)]
            occupied = [p for p in normal if p in slots]
            _require(occupied == normal[:len(occupied)], "only trailing slots may be blank")
            _require(all(any(p[0] == lane for p in occupied) for lane in range(lanes)),
                     "every lane must contain a glyph")
            order = ([(lane, band) for band in range(bands) for lane in range(lanes)]
                     if row_major else normal)
            tokens = [slots[p] for p in order if p in slots]
            commands = {c: [] for c in starts}
            for command, color in tokens:
                commands[color].append(command)
            _require(all(commands.values()), "seed has no program")
            streams = [(c, p, tuple(commands[c])) for c, p in starts.items()]
            candidates.append({"field": field, "separator": separator, "fills": fills,
                               "pane": pane.copy(), "streams": streams, "tokens": tokens,
                               "lanes": lanes, "bands": bands})
        except AssertionError:
            continue
    _require(len(candidates) == 1, "input has no unique valid panel separator")
    return candidates[0]


def _execute(input_grid, row_major=False):
    info = _read_input(input_grid, row_major)
    h, w = info["pane"].shape
    canvas, geometry, paths = _draw(h, w, info["field"], info["streams"])
    out = _fill(canvas, info["field"], info["fills"], geometry)
    return out, info, canvas, geometry, paths


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only the supplied input; no seed, examples or learned state."""
    return _execute(input_grid)[0]


def _through_program(rng, normal_size, low, high):
    """Finite exact-depth search for a meandering, orthogonally simple stroke."""
    depth_budget = normal_size - 1
    preference = tuple(int(x) for x in rng.permutation(5))

    @lru_cache(None)
    def possible(room, col, previous_horizontal):
        if room == 0:
            return True
        for k in preference:
            dr, dc, length = STROKES[k]
            cost, shift = 1 + dr * (length - 1), dc * (length - 1)
            if cost > room or not low <= col + shift <= high:
                continue
            # Reversing adjacent horizontal strokes makes a thick self-touch.
            if dc and previous_horizontal and (dc > 0) != (previous_horizontal > 0):
                continue
            if possible(room - cost, col + shift, dc):
                return True
        return False

    starts = [int(c) for c in rng.permutation(np.arange(low, high + 1))
              if possible(depth_budget, int(c), 0)]
    _require(starts, "no through program in corridor")
    start = starts[0]
    room, col, previous = depth_budget, start, 0
    commands = []
    while room:
        choices = []
        for k, (dr, dc, length) in enumerate(STROKES):
            cost, shift = 1 + dr * (length - 1), dc * (length - 1)
            if cost > room or not low <= col + shift <= high:
                continue
            if dc and previous and (dc > 0) != (previous > 0):
                continue
            if possible(room - cost, col + shift, dc):
                choices.append(k)
        # Long vertical commands make the program compact, while retaining
        # short vertical and horizontal commands as genuinely sampled choices.
        weights = np.array([4 if k == D4 else 2 if k == D2 else 1 for k in choices], float)
        k = int(rng.choice(choices, p=weights / weights.sum()))
        dr, dc, length = STROKES[k]
        commands.append(k)
        room -= 1 + dr * (length - 1)
        col += dc * (length - 1)
        previous = dc
    return start, tuple(commands)


def _corner_stream(rng, corner, height, width, forced=None):
    """Cut a small corner using an A program, not an independently drawn wall."""
    # Side and local transverse target: 0 is local-left; 1 is local-right.
    possibilities = {"TL": (("top", 0), ("left", 1)),
                     "TR": (("top", 1), ("right", 0)),
                     "BR": (("bottom", 0), ("right", 1)),
                     "BL": (("bottom", 1), ("left", 0))}
    side, target = possibilities[corner][int(rng.integers(2))]
    if forced is not None:
        side, target, commands = forced
    else:
        if target == 0:
            horizontal = [L4, L2] if int(rng.integers(2)) else [L2, L4]
        else:
            horizontal = [R3, R3]
        if int(rng.integers(3)) == 0:
            horizontal = [L4] if target == 0 else [R3]
        commands = [horizontal[0]]
        commands.append(D4 if int(rng.integers(3)) == 0 else D2)
        commands.extend(horizontal[1:])
        if int(rng.integers(4)) == 0:
            commands.insert(0, D2)
        commands = tuple(commands)
    span = width if side in ("top", "bottom") else height
    shift = sum(STROKES[k][1] * (STROKES[k][2] - 1) for k in commands)
    transverse = -shift if target == 0 else span - 1 - shift
    start = _global(0, transverse, side, height, width)
    return start, tuple(commands)


def _parallel_streams(rng, height, width, count, vertical, all_sides=False):
    span = width if vertical else height
    normal = height if vertical else width
    _require(span >= 5 * count + 1, "not enough corridor spacing")
    extra = span - (5 * count + 1)
    offset = int(rng.integers(extra + 1))
    result = []
    for i in range(count):
        low, high = 1 + offset + 5 * i, 4 + offset + 5 * i
        reverse = bool(i % 2) if all_sides else bool(rng.integers(2))
        side = (("top", "bottom") if vertical else ("right", "left"))[reverse]
        if side in ("bottom", "left"):
            low, high = span - 1 - high, span - 1 - low
        transverse, program = _through_program(rng, normal, low, high)
        result.append((_global(0, transverse, side, height, width), program))
    return result


def _recipes(rng, kind):
    """Sample program parameters. Geometry is subsequently checked from cells."""
    if kind == "phase":
        height = int(rng.integers(19, 29))
        width = int(rng.integers(12, 19))
        recipes = [
            _corner_stream(rng, "TL", height, width, ("top", 0, (L2, D4, L4))),
            _corner_stream(rng, "TR", height, width, ("top", 1, (R3, D2, R3))),
            _corner_stream(rng, "BR", height, width, ("bottom", 0, (L4, D2, L2))),
        ]
    elif kind == "depth":
        height = int(rng.integers(17, 30))
        width = int(rng.integers(10, 19))
        recipes = _parallel_streams(rng, height, width, 2, False, all_sides=True)
    elif kind == "mixed":
        height = int(rng.integers(24, 31))
        width = int(rng.integers(16, 19))
        # A central top-bottom program plus independent corner programs.
        side = ("top", "bottom")[int(rng.integers(2))]
        low, high = 6, width - 7
        if side == "bottom":
            low, high = width - 1 - high, width - 1 - low
        transverse, program = _through_program(rng, height, low, high)
        recipes = [(_global(0, transverse, side, height, width), program)]
        count = int(rng.integers(1, 4))
        corners = ["TL", "TR", "BR", "BL"]
        for index in rng.choice(4, count, replace=False):
            recipes.append(_corner_stream(rng, corners[int(index)], height, width))
    else:
        height = int(rng.integers(13, 31))
        width = int(rng.integers(10, 23))
        if kind == "corners":
            count = int(rng.integers(1, 5))
            corners = ["TL", "TR", "BR", "BL"]
            recipes = [_corner_stream(rng, corners[int(i)], height, width)
                       for i in rng.choice(4, count, replace=False)]
        else:
            vertical = bool(rng.integers(2))
            span = width if vertical else height
            count = int(rng.integers(1, min(3, (span - 1) // 5) + 1))
            recipes = _parallel_streams(rng, height, width, count, vertical)
    return height, width, recipes


def _interleave(rng, streams):
    positions = [0] * len(streams)
    result = []
    while True:
        active = [i for i, s in enumerate(streams) if positions[i] < len(s[2])]
        if not active:
            return result
        i = int(rng.choice(active))
        color, start, program = streams[i]
        result.append((program[positions[i]], color))
        positions[i] += 1


def _make_pair(rng, fills, kind, seen, require_order=False):
    for _ in range(192):
        try:
            height, width, recipes = _recipes(rng, kind)
            count = sum(len(program) for start, program in recipes)
            choices = []
            for lanes in (2, 3, 4):
                bands = (count + lanes - 1) // lanes
                if (width + 4 * lanes <= 30 and 4 * bands + 1 <= height
                        and count > (lanes - 1) * bands):
                    choices.append((lanes, bands))
            _require(choices, "program panel does not fit")
            lanes, bands = choices[int(rng.integers(len(choices)))]
            available = [c for c in range(10) if c not in fills]
            roles = [int(c) for c in rng.choice(available, 2 + len(recipes), replace=False)]
            field, separator = roles[:2]
            streams = [(color, start, program) for color, (start, program)
                       in zip(roles[2:], recipes)]
            canvas, geometry, paths = _draw(height, width, field, streams)
            if kind == "phase":
                _require(all(geometry["depths"][geometry["labels"][p]] % 2 == 1
                             for p in ((0, 0), (0, width - 1), (height - 1, width - 1))),
                         "corner phase witness failed")
            if kind == "depth":
                _require(geometry["max_depth"] >= 2, "missing even-depth witness")
            sep = 4 * lanes - 1
            inp = np.full((height, sep + 1 + width), field, dtype=np.int8)
            inp[:, sep] = separator
            inp[0, :2] = fills
            for color, start, program in streams:
                inp[start[0], sep + 1 + start[1]] = color
            tokens = _interleave(rng, streams)
            for i, (command, color) in enumerate(tokens):
                lane, band = divmod(i, bands)
                top, left = 2 + 4 * band, 4 * lane
                block = inp[top:top + 3, left:left + 3]
                block[np.asarray(MASKS[command], dtype=bool)] = color
            out = _fill(canvas, field, fills, geometry)
            _require(_key(inp) not in seen, "duplicate input")
            if require_order:
                try:
                    alternative = _execute(inp, row_major=True)[0]
                except AssertionError:
                    alternative = None
                _require(alternative is None or not np.array_equal(alternative, out),
                         "row-major traversal is not distinguished")
            seen.add(_key(inp))
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise RuntimeError(f"bounded construction failed for {kind}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    fills = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    seen = set()
    result = {"train": [], "test": []}
    for i in range(num_train):
        kind = ("phase" if i == 0 else "depth" if i == 1
                else ("corners", "parallel")[int(rng.integers(2))])
        result["train"].append(_make_pair(rng, fills, kind, seen, require_order=i == 0))
    for _ in range(num_test):
        result["test"].append(_make_pair(rng, fills, "mixed", seen))
    validate_generated_puzzle(result)
    return result


def _wrong_order_changes(inp, out):
    try:
        return not np.array_equal(_execute(inp, row_major=True)[0], out)
    except AssertionError:
        return True


def validate_generated_puzzle(puzzle):
    """Check schema, instruction semantics, wall topology and teaching witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal counts")
    seen, fills, analyses = set(), None, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate inputs")
            seen.add(_key(inp))
            expected, info, canvas, geometry, paths = _execute(inp)
            _require(np.array_equal(out, expected), "wrong instruction/parity output")
            fills = info["fills"] if fills is None else fills
            _require(info["fills"] == fills, "palette conventions change within puzzle")
            _require(out.shape == info["pane"].shape, "wrong output crop")
            wall = canvas != info["field"]
            _require(np.array_equal(out[wall], canvas[wall]), "divider not preserved")
            _require(set(int(x) for x in out[~wall]) == set(fills), "both parities must occur")
            _require(np.all(out[geometry["labels"] == geometry["root"]] == fills[0]),
                     "incorrect anchor phase")
            # Both ablations change the answer for every generated pair.
            without_a = np.full(out.shape, fills[0], dtype=np.int8)
            for color, start, commands in info["streams"]:
                without_a[start] = color
            _require(not np.array_equal(without_a, out), "A has no effect")
            _require(not np.array_equal(canvas, out), "B has no effect")
            analyses[split].append((info, geometry))
    first_info, first_geometry = analyses["train"][0]
    first_out = np.asarray(puzzle["train"][0]["output"])
    _require({k for k, c in first_info["tokens"]} == set(range(5)),
             "first demonstration must expose the full codebook")
    _require(all(int(first_out[p]) == fills[1] for p in ((0, 0), (0, -1), (-1, -1))),
             "first demonstration fails to distinguish corner anchors")
    _require(_wrong_order_changes(puzzle["train"][0]["input"], first_out),
             "column-major order is not demonstrated")
    _require(analyses["train"][1][1]["max_depth"] >= 2, "even crossing depth missing")
    depth_info, depth_geometry = analyses["train"][1]
    depth_out = np.asarray(puzzle["train"][1]["output"])
    even_nonroot = [i for i, d in enumerate(depth_geometry["depths"])
                    if d > 0 and d % 2 == 0]
    _require(even_nonroot, "no non-anchor region repeats the anchor fill")
    _require(all(np.all(depth_out[depth_geometry["labels"] == i] == fills[0])
                 for i in even_nonroot), "all-other-regions alternate shortcut survives")
    # A star demonstration and parallel demonstration also distinguish raw
    # row-major component numbering from topological parity.
    _require(any(any((i % 2) != (d % 2) for i, d in enumerate(g["depths"]))
                 for _, g in analyses["train"]), "component-index shortcut survives")
    # The fixed first two examples expose starts on all four boundary sides.
    sides = {_local_start(start, *i["pane"].shape)[0]
             for i, g in analyses["train"][:2] for c, start, cmd in i["streams"]}
    _require(sides == set(SIDES), "boundary orientations not demonstrated")
    for info, geometry in analyses["test"]:
        _require(len(info["streams"]) >= 2, "test must combine divider programs")
        _require(geometry["through_count"] == 1 and geometry["corner_count"] >= 1,
                 "test lacks a through/corner interaction")
    _require(all(not (g["through_count"] and g["corner_count"])
                 for i, g in analyses["train"]), "training violates topology shift")
    return None


def _signature(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"]))
                              for p in puzzle[split])) for split in ("train", "test"))


def self_test():
    """42-seed streaming audit; at most one maximum-count puzzle retained."""
    shapes, walls, palettes, vocab, starts, turns, degrees, counts = set(), set(), set(), set(), set(), set(), set(), set()
    signatures = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(hashlib.sha256(repr(_signature(puzzle)).encode()).hexdigest())
        for pair in puzzle["train"] + puzzle["test"]:
            out, info, canvas, geometry, paths = _execute(pair["input"])
            shapes.add((pair["input"].shape, out.shape))
            wall = (canvas != info["field"]).astype(np.uint8)
            walls.add((wall.shape, wall.tobytes()))
            palettes.add(info["fills"])
            vocab.update(k for k, c in info["tokens"])
            starts.update(_local_start(p, *canvas.shape)[0] for c, p, cmd in info["streams"])
            turns.update(geometry["turns"])
            degrees.add(geometry["degree"])
            counts.add(len(info["streams"]))
            _require(np.array_equal(out, pair["output"]), "solver disagreement")
            _require(np.array_equal(solve(pair["input"].tolist()), out), "list input unsupported")
    _require(len(signatures) == 42 and len(walls) >= 170, "structural diversity too small")
    _require(len(shapes) >= 50 and len(palettes) >= 20, "size/palette diversity too small")
    _require(vocab == set(range(5)) and starts == set(SIDES), "missing commands/orientations")
    _require(max(turns) >= 6 and max(degrees) >= 3 and counts == {1, 2, 3, 4},
             "missing turns, branching or divider counts")
    # Move a turn while preserving the seed, endpoint, stroke multiset and
    # canvas. A changes the *field parity*, not merely some divider pixels.
    original = (D4, R3, D4, D4, D2, R3)
    rerouted = (R3, D4, D4, D4, D2, R3)
    ca, ga, pa = _draw(17, 14, 0, [(3, (0, 4), original)])
    cb, gb, pb = _draw(17, 14, 0, [(3, (0, 4), rerouted)])
    _require(pa[0][-1] == pb[0][-1], "counterfactual endpoints differ")
    oa, ob = _fill(ca, 0, (7, 6), ga), _fill(cb, 0, (7, 6), gb)
    common_field = (ca == 0) & (cb == 0)
    _require(np.any(oa[common_field] != ob[common_field]), "glyph routing has no parity effect")
    # Execute that same controlled case through the public input-only solver.
    for commands, expected in ((original, oa), (rerouted, ob)):
        inp = np.zeros((17, 22), dtype=np.int8)
        inp[:, 7] = 8
        inp[0, :2] = (7, 6)
        inp[0, 8 + 4] = 3
        for j, command in enumerate(commands):
            lane, band = divmod(j, 3)
            block = inp[2 + 4 * band:5 + 4 * band, 4 * lane:4 * lane + 3]
            block[np.asarray(MASKS[command], bool)] = 3
        _require(np.array_equal(solve(inp), expected), "controlled routing regression failed")
    # Minimal one-band layout is a valid family member even though the
    # teaching-oriented sampler normally chooses taller canvases.
    small = np.zeros((5, 14), dtype=np.int8)
    small[:, 7] = 8
    small[0, :2] = (7, 6)
    small[0, 9] = 3
    for left in (0, 4):
        block = small[2:5, left:left + 3]
        block[np.asarray(MASKS[R3], bool)] = 3
    cs, gs, ps = _draw(5, 6, 0, [(3, (0, 1), (R3, R3))])
    _require(np.array_equal(solve(small), _fill(cs, 0, (7, 6), gs)), "one-band layout failed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(104729 + nt * 10 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(123, 10, 4)
    _require(_signature(p) == _signature(generate_puzzle(123, 10, 4)), "nondeterminism")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    out = damaged["test"][0]["output"]
    out[-1, 0] = (int(out[-1, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted fill was accepted")
    return {"seeds": 42, "pairs": 210, "distinct_shapes": len(shapes),
            "distinct_wall_layouts": len(walls), "fill_palettes": len(palettes),
            "turn_counts": sorted(turns), "region_degrees": sorted(degrees),
            "divider_counts": sorted(counts)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
