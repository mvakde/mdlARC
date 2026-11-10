"""Rail mosaics: a marker-encoded path and finite rectangle inventory agree.

The solver reconstructs roles, routes and exact covers from the input alone.
Generation paints labels from sampled placements, then checks joint uniqueness.
"""
from __future__ import annotations

import json
import numpy as np

VOID, BAR, MARKER, DIVIDER = 0, 5, 8, 9
LOCAL_COLORS = (1, 2, 3, 4, 6, 7)


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    seen = set()
    h, w = mask.shape
    for y, x in map(tuple, np.argwhere(mask)):
        y, x = int(y), int(x)
        if (y, x) in seen:
            continue
        cells, stack = [], [(y, x)]
        seen.add((y, x))
        while stack:
            yy, xx = stack.pop()
            cells.append((yy, xx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = yy + dy, xx + dx
                if (0 <= p[0] < h and 0 <= p[1] < w
                        and mask[p] and p not in seen):
                    seen.add(p)
                    stack.append(p)
        yield cells


def _split(grid):
    lines = [(axis, int(i)) for axis in (0, 1)
             for i in np.flatnonzero(np.all(grid == DIVIDER, axis=1-axis))]
    if len(lines) != 1:
        raise ValueError("Exactly one full-span divider is required")
    axis, cut = lines[0]
    if cut == 0 or cut == grid.shape[axis]-1:
        raise ValueError("Divider must separate two nonempty panels")
    if axis == 0:
        parts = (grid[:cut], grid[cut+1:])
    else:
        parts = (grid[:, :cut], grid[:, cut+1:])
    if any(np.any(p == DIVIDER) for p in parts):
        raise ValueError("Divider color cannot occur elsewhere")
    board_flags = [bool(np.any(p == BAR) and np.any(p == MARKER)) for p in parts]
    if sum(board_flags) != 1:
        raise ValueError("The bars must identify the destination panel")
    index = board_flags.index(True)
    board, tray = parts[index], parts[1-index]
    if np.any((tray == BAR) | (tray == MARKER)):
        raise ValueError("Bar colors are reserved for the destination")
    return board.copy(), tray.copy(), (axis, index)


def _normalize(board, tray):
    colors = set(map(int, np.unique(board))) - {VOID, BAR, MARKER}
    if len(colors) != 2:
        raise ValueError("Destination needs one background and one seed color")
    singles = [c for c in colors if np.count_nonzero(board == c) == 1]
    if len(singles) != 1:
        raise ValueError("The seed color must occur exactly once on the board")
    route = singles[0]
    background = (colors - {route}).pop()
    if background not in LOCAL_COLORS or route not in LOCAL_COLORS:
        raise ValueError("Invalid local color role")
    y, x = map(int, np.argwhere(board == route)[0])
    h, w = board.shape
    edges = [y == 0, x == w-1, y == h-1, x == 0]
    if sum(edges) != 1:
        raise ValueError("Seed must occupy a noncorner boundary cell")
    k = edges.index(True)
    return (np.rot90(board, k).copy(), np.rot90(tray, k).copy(),
            background, route, k)


def _tiles(tray, background):
    if np.any(tray == background) or not np.any(tray == VOID):
        raise ValueError("Tray background must be the vacancy color")
    tiles = []
    for cells in _components(tray != VOID):
        yy, xx = zip(*cells)
        t = tray[min(yy):max(yy)+1, min(xx):max(xx)+1]
        if len(cells) != t.size or np.any(t == VOID):
            raise ValueError("Source components must be filled rectangles")
        tiles.append(t.copy())
    if not 3 <= len(tiles) <= 10:
        raise ValueError("Family supports 3..10 supplied rectangles")
    return tiles


def _routes(board, route):
    """Enumerate the geometrically possible B-style routes, including symmetry."""
    h, w = board.shape
    structure = (board == BAR) | (board == MARKER)
    rows = list(map(int, np.flatnonzero(structure.any(axis=1))))
    if not 1 <= len(rows) <= 4:
        raise ValueError("Need 1..4 parallel bars")
    bars = []
    previous = -1
    for f in rows:
        occupied = np.flatnonzero(structure[f])
        lo, hi = int(occupied[0]), int(occupied[-1])
        if len(occupied) != hi-lo+1 or not np.any(board[f] == BAR):
            raise ValueError("A bar must be a contiguous span with permanent cells")
        markers = tuple(map(int, np.flatnonzero(board[f] == MARKER)))
        n = len(markers)
        if not 1 <= n <= 4:
            raise ValueError("Marker count must be 1..4")
        low = markers == tuple(lo+2*j for j in range(n))
        high = markers == tuple(hi-2*j for j in reversed(range(n)))
        d, turn = n+1, f-n-1
        if not (low or high) or not (1 <= turn and turn > previous and f < h-1):
            raise ValueError("Malformed endpoint command or forward spacing")
        candidates = []
        if low and hi+d < w:
            candidates.append((1, hi+d))
        if high and lo-d >= 0:
            candidates.append((-1, lo-d))
        if not candidates:
            raise ValueError("No in-bounds bypass")
        bars.append((f, lo, hi, n, turn, low and high, candidates))
        previous = f
    initial = int(np.argwhere(board == route)[0, 1])
    answers = []

    def trace(i, current, start, path, signs):
        if i == len(bars):
            path = path.copy()
            path[start:, current] = True
            if not np.any(path & structure):
                answers.append({"path": path, "signs": tuple(signs), "bars": bars})
            return
        f, lo, hi, n, turn, symmetric, candidates = bars[i]
        if not lo <= current <= hi:
            return
        for side, target in candidates:
            p = path.copy()
            p[start:turn+1, current] = True
            p[turn, min(current, target):max(current, target)+1] = True
            if not np.any(p & structure):
                trace(i+1, target, turn, p, signs+[side])

    trace(0, initial, 0, np.zeros(board.shape, dtype=bool), [])
    if not answers:
        raise ValueError("No collision-free route intercepts every bar")
    return answers


def _pack(mask, tiles, rail, route, max_nodes=None):
    """Finite-inventory exact cover, filtering placements by embedded rail pixels.

    Two distinct rendered solutions suffice to prove ambiguity. Interchangeable
    identical copies are grouped. Dead-state caching records only impossibility,
    so it never suppresses distinct payload renderings through the same geometry.
    """
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not coords:
        raise ValueError("No vacancies")
    h, w = mask.shape
    cell_index = np.full(mask.shape, -1, dtype=int)
    for i, p in enumerate(coords):
        cell_index[p] = i
    groups, lookup = [], {}
    for t in tiles:
        key = _key(t)
        if key in lookup:
            groups[lookup[key]]["count"] += 1
        else:
            lookup[key] = len(groups)
            groups.append({"tile": t, "count": 1})
    placements, by_cell = [], [[] for _ in coords]
    for g, item in enumerate(groups):
        t = item["tile"]
        th, tw = t.shape
        for y in range(h-th+1):
            for x in range(w-tw+1):
                if not np.all(mask[y:y+th, x:x+tw]):
                    continue
                if rail is not None and not np.array_equal(t == route, rail[y:y+th, x:x+tw]):
                    continue
                indices = tuple(map(int, cell_index[y:y+th, x:x+tw].flat))
                bits = sum(1 << i for i in indices)
                j = len(placements)
                placements.append((g, bits, indices, tuple(map(int, t.flat)), y, x))
                for i in indices:
                    by_cell[i].append(j)
    full = (1 << len(coords))-1
    initial = [g["count"] for g in groups]
    remaining, rendered, path = initial.copy(), [-1]*len(coords), []
    dead, outcomes = set(), {}
    nodes = 0

    def search(covered):
        nonlocal nodes
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            raise ValueError("Exact-cover construction budget exhausted")
        if covered == full:
            key = tuple(rendered)
            used = tuple(a-b for a, b in zip(initial, remaining))
            if key not in outcomes:
                outcomes[key] = {"placements": tuple(path), "used": used}
            return True
        state = covered, tuple(remaining)
        if state in dead:
            return False
        best = None
        for i in range(len(coords)):
            if covered & (1 << i):
                continue
            choices = [j for j in by_cell[i]
                       if remaining[placements[j][0]] and not (placements[j][1] & covered)]
            if not choices:
                dead.add(state)
                return False
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
        success = False
        for j in best:
            g, bits, indices, values, y, x = placements[j]
            remaining[g] -= 1
            for i, v in zip(indices, values):
                rendered[i] = v
            path.append((g, y, x))
            success |= search(covered | bits)
            path.pop()
            for i in indices:
                rendered[i] = -1
            remaining[g] += 1
            if len(outcomes) >= 2:
                return True
        if not success:
            dead.add(state)
        return success

    search(0)
    for record in outcomes.values():
        record.update(nodes=nodes, dead_states=len(dead))
    return outcomes, groups


def _complete(board, tiles, background, route, max_nodes=None):
    mask = board == VOID
    routes = _routes(board, route)
    results = {}
    for candidate in routes:
        outcomes, groups = _pack(mask, tiles, candidate["path"], route, max_nodes)
        for rendering, record in outcomes.items():
            out = board.copy()
            out[out == MARKER] = BAR
            out[candidate["path"] & ~mask] = route
            out[mask] = rendering
            key = _key(out)
            results[key] = dict(output=out, mask=mask, route=route, background=background,
                                tiles=tiles, groups=groups, record=record,
                                selected_route=candidate, geometric_routes=len(routes))
            if len(results) > 1:
                raise ValueError("Multiple joint route/mosaic renderings")
    if len(results) != 1:
        raise ValueError("No inventory can reconstruct a commanded route")
    return next(iter(results.values()))


def _analyze(input_grid, max_nodes=None):
    board, tray, split = _split(_grid(input_grid))
    board, tray, background, route, k = _normalize(board, tray)
    info = _complete(board, _tiles(tray, background), background, route, max_nodes)
    info.update(board=board, tray=tray, rotation=k, split=split)
    info["output"] = np.rot90(info["output"], -k).copy()
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique completed destination, without any generation state."""
    return _analyze(input_grid)["output"]


def _span_options(width, current, n, symmetric):
    result = []
    d = n+1
    for lo in range(current+1):
        for hi in range(max(current, lo+max(2, 2*n-1)-1), width):
            sym = hi-lo+1 == 2*n-1
            if sym != symmetric:
                continue
            for sign in (-1, 1):
                target = lo-d if sign < 0 else hi+d
                if 0 <= target < width:
                    if symmetric and not (lo-d >= 0 and hi+d < width):
                        continue
                    result.append((lo, hi, target, sign))
    return result


def _sample_plan(rng, width, counts, signs, symmetric):
    failed = set()

    def visit(i, current):
        if i == len(counts):
            return []
        if (i, current) in failed:
            return None
        opts = _span_options(width, current, counts[i], symmetric and i == len(counts)-1)
        if signs is not None:
            opts = [o for o in opts if o[3] == signs[i]]
        for j in rng.permutation(len(opts)):
            o = opts[int(j)]
            tail = visit(i+1, o[2])
            if tail is not None:
                return [o]+tail
        failed.add((i, current))
        return None

    for initial in rng.permutation(np.arange(1, width-1)):
        tail = visit(0, int(initial))
        if tail is not None:
            return int(initial), tail
    return None


def _make_board(rng, counts, signs, symmetric):
    minimum = sum(n+1 for n in counts)+len(counts)+2
    height = int(rng.integers(max(10, minimum), 27))
    width = int(rng.integers(max(12, 3*max(counts)+2), 23))
    plan = _sample_plan(rng, width, counts, signs, symmetric)
    if plan is None:
        return None
    initial, spans = plan
    background, route = map(int, rng.choice(LOCAL_COLORS, 2, replace=False))
    board = np.full((height, width), background, dtype=np.int8)
    board[0, initial] = route
    path = np.zeros(board.shape, dtype=bool)
    gaps = rng.multinomial(height-minimum, np.full(len(counts)+1, 1/(len(counts)+1)))
    previous, current, start = -1, initial, 0
    for i, (n, (lo, hi, target, side)) in enumerate(zip(counts, spans)):
        turn = (1 if i == 0 else previous+1)+int(gaps[i])
        f = turn+n+1
        board[f, lo:hi+1] = BAR
        board[f, [lo+2*j if side > 0 else hi-2*j for j in range(n)]] = MARKER
        path[start:turn+1, current] = True
        path[turn, min(current, target):max(current, target)+1] = True
        previous, current, start = f, target, turn
    path[start:, current] = True
    return board, path, background, route


def _grow(rng, board, path, count, same_size):
    """Irregular unions of rectangles; contact offsets and islands are sampled."""
    height, width = board.shape
    forbidden = (board == BAR) | (board == MARKER)
    forbidden[0, int(np.flatnonzero(path[0])[0])] = True
    occupied = np.zeros(board.shape, dtype=bool)
    rects = []

    def valid(y, x, h, w):
        return (0 <= y and 0 <= x and y+h <= height and x+w <= width
                and not np.any(forbidden[y:y+h, x:x+w] | occupied[y:y+h, x:x+w]))

    def add(rect):
        y, x, h, w = rect
        rects.append(rect)
        occupied[y:y+h, x:x+w] = True

    if same_size:
        # A teaching witness: adjacent equal footprints carrying different rail
        # fragments. Dimensions alone cannot decide which colored tile goes where.
        dims = [(2, 2), (2, 3), (3, 2), (3, 3)]
        rng.shuffle(dims)
        found = False
        for h, w in dims:
            candidates = [(y, x) for y in range(height-h+1) for x in range(width-w+1)
                          if valid(y, x, h, w) and np.any(path[y:y+h, x:x+w])]
            rng.shuffle(candidates)
            for y, x in candidates:
                signature = path[y:y+h, x:x+w]
                starts = [(y+s*h, x+o) for s in (-1, 1) for o in range(1-w, w)]
                starts += [(y+o, x+s*w) for s in (-1, 1) for o in range(1-h, h)]
                rng.shuffle(starts)
                for yy, xx in starts:
                    if (valid(yy, xx, h, w) and np.any(path[yy:yy+h, xx:xx+w])
                            and not np.array_equal(signature, path[yy:yy+h, xx:xx+w])):
                        add((y, x, h, w))
                        add((yy, xx, h, w))
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return None
    for i in range(len(rects), count):
        candidates = []
        for _ in range(180):
            h, w = map(int, rng.integers(1, 5, size=2))
            attach = bool(rects) and (i == 2 or rng.random() < .85)
            if attach:
                y0, x0, h0, w0 = rects[int(rng.integers(len(rects)))]
                side = int(rng.integers(4))
                if side < 2:
                    y = y0-h if side == 0 else y0+h0
                    x = x0+int(rng.integers(1-w, w0))
                else:
                    x = x0-w if side == 2 else x0+w0
                    y = y0+int(rng.integers(1-h, h0))
            else:
                yy, xx = map(int, np.argwhere(path)[int(rng.integers(path.sum()))])
                y, x = yy-int(rng.integers(h)), xx-int(rng.integers(w))
            if valid(y, x, h, w):
                candidates.append((y, x, h, w))
        if not candidates:
            return None
        add(candidates[int(rng.integers(len(candidates)))])
    return occupied, rects


def _pattern(rng, shape, palette):
    a = int(rng.choice(palette))
    tile = np.full(shape, a, dtype=np.int8)
    if len(palette) > 1:
        b = int(rng.choice([c for c in palette if c != a]))
        yy, xx = np.indices(shape)
        mode = int(rng.integers(6))
        if mode == 1:
            tile[yy % 2 == 1] = b
        elif mode == 2:
            tile[xx % 2 == 1] = b
        elif mode == 3:
            tile[(yy+xx) % 2 == 1] = b
        elif mode == 4:
            tile = rng.choice(palette, size=shape).astype(np.int8)
        elif mode == 5:
            tile[0, 0] = b
    return tile


def _make_tiles(rng, board, path, background, route, rects):
    palette = [c for c in LOCAL_COLORS if c not in (background, route)]
    rng.shuffle(palette)
    palette = palette[:int(rng.integers(1, 5))]
    tiles, cache = [], {}
    output = board.copy()
    output[output == MARKER] = BAR
    output[path] = route
    for y, x, h, w in rects:
        fragment = path[y:y+h, x:x+w]
        signature = h, w, fragment.tobytes()
        if signature not in cache:
            tile = _pattern(rng, (h, w), palette)
            tile[fragment] = route
            cache[signature] = tile
        tile = cache[signature]
        tiles.append(tile.copy())
        output[y:y+h, x:x+w] = tile
    return tiles, output, palette


def _scatter(rng, tiles, height, max_width):
    min_width = max(t.shape[1] for t in tiles)
    if min_width > max_width or max(t.shape[0] for t in tiles) > height:
        return None
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
    for attempt in range(14):
        width = max_width if attempt >= 9 else int(rng.integers(min_width, max_width+1))
        occupied = np.zeros((height, width), dtype=bool)
        positions = {}
        for i in order:
            h, w = tiles[i].shape
            legal = [(y, x) for y in range(height-h+1) for x in range(width-w+1)
                     if not np.any(occupied[max(0, y-1):min(height, y+h+1),
                                            max(0, x-1):min(width, x+w+1)])]
            if not legal:
                break
            y, x = legal[int(rng.integers(len(legal)))]
            positions[i] = y, x
            occupied[y:y+h, x:x+w] = True
        if len(positions) == len(tiles) and not np.all(occupied):
            tray = np.zeros(occupied.shape, dtype=np.int8)
            for i, (y, x) in positions.items():
                h, w = tiles[i].shape
                tray[y:y+h, x:x+w] = tiles[i]
            return tray
    return None


def _join(rng, board, output, tiles, rotation):
    for axis in rng.permutation(2):
        axis = int(axis)
        # Scatter has a horizontal packing API. Transpose tiles for a top/bottom
        # tray, then transpose the tray back; visible tile orientation is preserved.
        if axis == 1:
            tray = _scatter(rng, tiles, board.shape[0], 29-board.shape[1])
        else:
            t = _scatter(rng, [t.T for t in tiles], board.shape[1], 29-board.shape[0])
            tray = None if t is None else t.T
        if tray is None:
            continue
        divider_shape = (board.shape[0], 1) if axis == 1 else (1, board.shape[1])
        divider = np.full(divider_shape, DIVIDER, dtype=np.int8)
        panels = [board, divider, tray]
        if rng.random() < .5:
            panels.reverse()
        inp = np.concatenate(panels, axis=axis)
        return {"input": np.rot90(inp, rotation).copy(),
                "output": np.rot90(output, rotation).copy()}
    return None


def _build_pair(rng, counts, signs, symmetric, tile_count, same_size, surplus, rotation):
    for _ in range(160):
        data = _make_board(rng, counts, signs, symmetric)
        if data is None:
            continue
        board, path, background, route = data
        grown = _grow(rng, board, path, tile_count, same_size)
        if grown is None:
            continue
        mask, rects = grown
        tiles, expected, palette = _make_tiles(rng, board, path, background, route, rects)
        board[mask] = VOID
        try:
            check = _complete(board, tiles, background, route, max_nodes=20000)
        except ValueError:
            continue
        if not np.array_equal(check["output"], expected):
            continue
        if symmetric and check["geometric_routes"] < 2:
            continue
        # Reuse a needed size for surplus. It fits the silhouette geometrically;
        # different embedded fragments make it irrelevant to the joint solution.
        added = 0
        for _ in range(24):
            if added == surplus:
                break
            shape = tiles[int(rng.integers(tile_count))].shape
            decoy = _pattern(rng, shape, palette)
            decoy[rng.random(shape) < rng.uniform(.15, .6)] = route
            if any(np.array_equal(decoy, t) for t in tiles):
                continue
            try:
                candidate = _complete(board, tiles+[decoy], background, route, max_nodes=20000)
            except ValueError:
                continue
            if np.array_equal(candidate["output"], expected):
                tiles.append(decoy)
                added += 1
        if added != surplus:
            continue
        pair = _join(rng, board, expected, tiles, rotation)
        if pair is not None:
            return pair
    raise RuntimeError("Could not sample a unique rail mosaic within the bounded attempts")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    maximum = int(rng.choice((3, 4)))
    orientation_shift = bool(rng.random() < .65)
    rotations = list(map(int, rng.permutation(4)))
    # The obligations are assigned to random training slots, then presentation is
    # shuffled. They guarantee useful evidence even for a two-demonstration puzzle.
    witness_slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(num_train+num_test):
        training = i < num_train
        packing_witness = i == witness_slots[0]
        symmetry_witness = i == witness_slots[1]
        if packing_witness:
            counts = list(map(int, rng.permutation(np.arange(1, maximum+1))))
            sign = int(rng.choice((-1, 1)))
            signs = [sign]*len(counts)
            opposite_slot = int(rng.choice((0, len(counts)-1)))
            signs[opposite_slot] = -sign
            symmetric = False
        elif symmetry_witness:
            counts = [int(rng.integers(2, maximum+1))]
            signs, symmetric = None, True
        else:
            count = int(rng.integers(1, 5))
            counts = list(map(int, rng.integers(1, maximum+1, size=count)))
            while sum(n+1 for n in counts)+len(counts)+2 > 26:
                index = max(range(len(counts)), key=lambda j: counts[j])
                counts[index] -= 1
            symmetric = bool(rng.random() < .35 and counts[-1] >= 2)
            signs = None
        if orientation_shift:
            rotation = int(rng.choice(rotations[:3] if training else rotations[3:]))
            tile_count = int(rng.integers(3, 8))
        else:
            rotation = int(rng.choice(rotations))
            tile_count = int(rng.integers(3, 6) if training else rng.integers(6, 9))
        same_size = packing_witness or bool(rng.random() < .55)
        surplus = 1 if packing_witness else int(rng.choice((0, 0, 1, 2)))
        for _ in range(8):
            pair = _build_pair(rng, counts, signs, symmetric, tile_count, same_size, surplus, rotation)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("Failed to obtain distinct inputs")
        seen.add(_key(pair["input"]))
        puzzle["train" if training else "test"].append(pair)
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(j)] for j in rng.permutation(len(puzzle[split]))]
    return puzzle


def _features(info):
    selected = info["selected_route"]
    used = info["record"]["used"]
    shapes = [g["tile"].shape for g, n in zip(info["groups"], used) if n]
    dimension_patterns = {}
    for g, n in zip(info["groups"], used):
        if n:
            dimension_patterns.setdefault(g["tile"].shape, set()).add(_key(g["tile"]))
    owner = np.full(info["mask"].shape, -1, dtype=int)
    for i, (group, y, x) in enumerate(info["record"]["placements"]):
        h, w = info["groups"][group]["tile"].shape
        owner[y:y+h, x:x+w] = i
    component_tiles = [len({int(owner[p]) for p in comp}) for comp in _components(info["mask"])]
    return dict(rotation=info["rotation"], split=info["split"],
                board_shape=info["board"].shape, bar_count=len(selected["bars"]),
                markers=tuple(b[3] for b in selected["bars"]), signs=selected["signs"],
                geometric_routes=info["geometric_routes"], used=sum(used),
                supplied=len(info["tiles"]), surplus=len(info["tiles"])-sum(used),
                repeated=any(n > 1 for n in used),
                unequal_same_size=any(len(v) > 1 for v in dimension_patterns.values()),
                max_component_tiles=max(component_tiles), components=len(component_tiles),
                max_area=max(h*w for h, w in shapes),
                payload_colors=len(set(c for t in info["tiles"] for c in map(int, t.flat)) - {info["route"]}),
                coverage=float(info["mask"].mean()), nodes=info["record"]["nodes"])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(info["output"], out)
            mask, path, route = info["mask"], info["selected_route"]["path"], info["route"]
            canonical = np.rot90(out, info["rotation"])
            assert canonical.shape == info["board"].shape
            assert np.array_equal(canonical == route, path)
            assert not np.any((canonical == VOID) | (canonical == MARKER) | (canonical == DIVIDER))
            bar_cells = (info["board"] == BAR) | (info["board"] == MARKER)
            assert np.all(canonical[bar_cells] == BAR)
            stable = ~mask & ~path & ~bar_cells
            assert np.array_equal(canonical[stable], info["board"][stable])
            occupied = np.zeros(mask.shape, dtype=int)
            for group, y, x in info["record"]["placements"]:
                t = info["groups"][group]["tile"]
                h, w = t.shape
                assert np.array_equal(canonical[y:y+h, x:x+w], t)
                occupied[y:y+h, x:x+w] += 1
            assert np.array_equal(occupied, mask.astype(int))
            assert all(n <= g["count"] for n, g in zip(info["record"]["used"], info["groups"]))
            features[split].append(_features(info))
    tr, te = features["train"], features["test"]
    assert any(f["unequal_same_size"] and f["max_component_tiles"] >= 3 for f in tr)
    assert any(f["geometric_routes"] >= 2 for f in tr)
    assert any(f["surplus"] for f in tr)
    assert {n for f in te for n in f["markers"]} <= {n for f in tr for n in f["markers"]}
    assert {s for f in tr for s in f["signs"]} == {-1, 1}
    assert any(any(a == b for a, b in zip(f["signs"], f["signs"][1:])) for f in tr)
    # Actual output-derived shifts, rather than hidden generation flags.
    assert ({f["rotation"] for f in te} - {f["rotation"] for f in tr}
            or min(f["used"] for f in te) > max(f["used"] for f in tr))


def self_test():
    measurements, signatures, shapes, normalized = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple((_key(p["input"]), _key(p["output"]))
                          for s in ("train", "test") for p in puzzle[s])
        signatures.add(signature)
        for s in ("train", "test"):
            for pair in puzzle[s]:
                info = _analyze(pair["input"])
                measurements.append((s, _features(info)))
                shapes.add(pair["input"].shape)
                roles = {}
                normalized.add(tuple(roles.setdefault(int(c), len(roles)) for c in pair["input"].flat))
        if seed in (0, 17, 41):
            other = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], other[s]) for k in ("input", "output"))
    assert len(signatures) == 42 and len(shapes) >= 45 and len(normalized) >= 190
    fs = [f for s, f in measurements]
    assert {f["rotation"] for f in fs} == {0, 1, 2, 3}
    assert {f["bar_count"] for f in fs} == {1, 2, 3, 4}
    assert {n for f in fs for n in f["markers"]} == {1, 2, 3, 4}
    assert {f["surplus"] for f in fs} >= {0, 1, 2}
    assert max(f["used"] for f in fs) >= 8 and min(f["used"] for f in fs) == 3
    assert any(f["repeated"] for f in fs)
    assert any(f["components"] > 1 for f in fs)
    assert max(f["max_component_tiles"] for f in fs) >= 6
    assert len({f["board_shape"] for f in fs}) >= 45
    assert len({f["coverage"] for f in fs}) >= 100
    assert {f["payload_colors"] for f in fs} >= {1, 2, 3, 4}
    # Mechanism ablations: geometry alone has multiple colored completions,
    # while routing alone has multiple feasible detours in a training example.
    zero = generate_puzzle(0)
    infos = [_analyze(p["input"]) for p in zero["train"]]
    witness = next(i for i in infos if _features(i)["unequal_same_size"])
    outcomes, _ = _pack(witness["mask"], witness["tiles"], None, witness["route"])
    assert len(outcomes) == 2
    intended = tuple(map(int, np.rot90(witness["output"], witness["rotation"])[witness["mask"]]))
    assert any(rendering != intended for rendering in outcomes)
    assert any(i["geometric_routes"] > 1 for i in infos)
    for info in infos:
        rail_only = info["board"].copy()
        rail_only[rail_only == MARKER] = BAR
        rail_only[info["selected_route"]["path"]] = info["route"]
        assert not np.array_equal(rail_only, np.rot90(info["output"], info["rotation"]))
    # Whole-scene rotation and cue-preserving recoloring remain input-only.
    sample = zero["train"][0]
    permutation = np.arange(10, dtype=np.int8)
    permutation[list(LOCAL_COLORS)] = np.array(LOCAL_COLORS)[::-1]
    for k in range(4):
        inp = permutation[np.rot90(sample["input"], k)]
        out = permutation[np.rot90(sample["output"], k)]
        assert np.array_equal(solve(inp), out)
    # Exhausted finite inventory, distinguishable patterns, and symmetric ties.
    mask = np.ones((1, 4), dtype=bool)
    tile = np.array([[1, 2]], dtype=np.int8)
    assert not _pack(mask, [tile], None, 1)[0]
    assert len(_pack(mask, [tile, tile], None, 1)[0]) == 1
    assert len(_pack(mask, [tile, np.array([[3, 4]], dtype=np.int8)], None, 1)[0]) == 2
    # A symmetric route without a discriminating inventory must be rejected.
    ambiguous = np.full((9, 15), 2, dtype=np.int8)
    ambiguous[0, 7] = 1
    ambiguous[5, 6:9] = (MARKER, BAR, MARKER)
    ambiguous[1, 12:15] = VOID
    try:
        _complete(ambiguous, [np.array([[3]], dtype=np.int8)]*3, 2, 1)
    except ValueError as e:
        assert "Multiple" in str(e)
    else:
        raise AssertionError("Accepted unresolved symmetric routing")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1042+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted illegal counts")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in zero[s]] for s in zero}
    damaged["test"][0]["output"][0, 0] = DIVIDER
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Accepted corrupted output")
    return {"seeds": 42, "pairs": len(measurements), "distinct_input_shapes": len(shapes),
            "distinct_normalized_inputs": len(normalized),
            "directions": 4, "bars": sorted({f["bar_count"] for f in fs}),
            "used_tile_counts": sorted({f["used"] for f in fs}),
            "ambiguous_route_pairs": sum(f["geometric_routes"] > 1 for f in fs),
            "equal_size_distinct_pattern_pairs": sum(f["unequal_same_size"] for f in fs)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
