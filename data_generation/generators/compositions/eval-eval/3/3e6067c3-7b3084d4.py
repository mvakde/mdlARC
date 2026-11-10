"""Route-constrained jigsaw maps. Only NumPy and the standard library are used.

The decoder searches rigid square packings and station routes together. In
particular, it does not commit to the first geometrically valid packing.
"""
from math import isqrt

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "Invalid grid color")
    return a.astype(np.int8, copy=True)


def _components(mask, neighbors=NEIGHBORS):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
        result.append(sorted(cells))
    return result


def _tight(array, mask):
    points = np.argwhere(mask)
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0) + 1
    return np.where(mask[r0:r1, c0:c1], array[r0:r1, c0:c1], -1).astype(np.int8)


def _orientations(piece):
    seen, result = set(), []
    for k in range(4):
        a = np.rot90(piece, k).copy()
        sig = (a.shape, a.tobytes())
        if sig not in seen:
            seen.add(sig)
            result.append(a)
    return result


def _patches(array, label_colors):
    result = []
    for color in sorted(label_colors):
        for cells in _components(array == color):
            rr, cc = zip(*cells)
            r, c = min(rr), min(cc)
            width = max(rr) - r + 1
            _require(max(cc) - c + 1 == width and len(cells) == width ** 2,
                     "A station must be a solid square")
            result.append((r, c, width, color))
    return sorted(result)


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "The surrounding border must identify the background")
    components = _components(grid != bg)
    tokens = sorted((cells[0] for cells in components if len(cells) == 1), key=lambda p: p[1])
    objects = [cells for cells in components if len(cells) > 1]
    _require(4 <= len(tokens) <= 7 and 3 <= len(objects) <= 5, "Invalid key or piece count")
    key_y = tokens[0][0]
    _require(all(r == key_y for r, c in tokens)
             and all(b[1] - a[1] == 2 for a, b in zip(tokens, tokens[1:])), "Invalid bottom key")
    _require(key_y >= max(r for cells in objects for r, c in cells) + 2,
             "Key must be detached below all pieces")
    key = tuple(int(grid[p]) for p in tokens)
    _require(len(set(key)) >= 2, "At least two station labels are required")
    pieces = []
    for cells in objects:
        mask = np.zeros(grid.shape, dtype=bool)
        for p in cells:
            mask[p] = True
        pieces.append(_tight(grid, mask))
    absent_counts = {}
    for piece in pieces:
        for v in piece[piece >= 0]:
            v = int(v)
            if v not in key:
                absent_counts[v] = absent_counts.get(v, 0) + 1
    markers = [v for v, n in absent_counts.items() if n == 1]
    _require(len(markers) == 1, "One singleton color absent from the key must mark the anchor")
    marker = markers[0]
    carriers = []
    for piece in pieces:
        colors = set(map(int, piece[piece >= 0])) - set(key) - {marker}
        _require(len(colors) == 1, "Each piece must have one carrier color")
        carrier = colors.pop()
        _require(np.sum(piece == carrier) >= 2, "Each carrier must occur at least twice")
        carriers.append(carrier)
    _require(len(set(carriers)) == len(pieces), "Piece carriers must be distinct")
    patches = [p for piece in pieces for p in _patches(piece, set(key))]
    _require(patches and len({p[2] for p in patches}) == 1, "Common station width required")
    width = patches[0][2]
    _require(width in (1, 2), "Station widths are one or two cells")
    _require(set(p[3] for p in patches) == set(key), "All key labels must occur on pieces")
    area = sum(int(np.sum(piece >= 0)) for piece in pieces)
    side = isqrt(area)
    _require(side * side == area and 6 <= side <= 13, "Piece area must define a square of side 6..13")
    return dict(grid=grid, background=bg, marker=marker, carriers=tuple(carriers),
                pieces=pieces, side=side, width=width, key=key, key_y=key_y)


def _band(a, b, side):
    """Flat indices strictly between the facing edges of two square patches."""
    r, c, width = a[:3]
    y, x, other_width = b[:3]
    if width != other_width:
        return None
    if r == y and abs(c - x) > width:
        return tuple(rr * side + cc for rr in range(r, r + width)
                     for cc in range(min(c, x) + width, max(c, x)))
    if c == x and abs(r - y) > width:
        return tuple(rr * side + cc for rr in range(min(r, y) + width, max(r, y))
                     for cc in range(c, c + width))
    return None


def _routes(board, stations, key, carriers, limit=2):
    side = board.shape[0]
    edges = {}
    for i, a in enumerate(stations):
        for j, b in enumerate(stations):
            if i != j:
                cells = _band(a, b, side)
                if cells and all(int(board.flat[p]) in carriers for p in cells):
                    edges[i, j] = (cells, sum(1 << p for p in cells))
    candidates = [[i for i, p in enumerate(stations) if p[3] == color] for color in key]
    answers = []

    def visit(path, used, painted):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for i in candidates[len(path)]:
            if used & (1 << i):
                continue
            mask = 0
            if path:
                edge = edges.get((path[-1], i))
                if edge is None or edge[1] & painted:
                    continue
                mask = edge[1]
            visit(path + [i], used | (1 << i), painted | mask)

    visit([], 0, 0)
    return answers, edges


def _paint(board, stations, route):
    result = board.copy()
    for i, j in zip(route, route[1:]):
        cells = _band(stations[i], stations[j], board.shape[0])
        result.flat[list(cells)] = stations[i][3]
    return result


def _placements(facts):
    side, marker, labels = facts['side'], facts['marker'], set(facts['key'])
    candidates = []
    for piece in facts['pieces']:
        marked = bool(np.any(piece == marker))
        by_cells = {}
        for oriented in _orientations(piece):
            h, w = oriented.shape
            if h > side or w > side:
                continue
            if marked and oriented[0, 0] != marker:
                continue
            local = [(int(r), int(c), int(oriented[r, c])) for r, c in np.argwhere(oriented >= 0)]
            patches = _patches(oriented, labels)
            offsets = [(0, 0)] if marked else [(r, c) for r in range(side - h + 1)
                                             for c in range(side - w + 1)]
            for r0, c0 in offsets:
                cells = tuple(sorted(((r + r0) * side + c + c0, color) for r, c, color in local))
                mask = sum(1 << pos for pos, color in cells)
                stations = tuple((r + r0, c + c0, p, color) for r, c, p, color in patches)
                # Equal silhouettes with different station orientations are distinct.
                by_cells.setdefault(cells, (mask, cells, stations))
        candidates.append(list(by_cells.values()))
    return candidates


def _search(facts, limit=2, geometry_only=False):
    """Exact cover over colored rigid placements, then route feasibility.

    No cap on geometric packings: all must be considered to certify joint
    uniqueness. The generated small cavity maps keep this search bounded.
    """
    candidates = _placements(facts)
    side = facts['side']
    full = (1 << (side * side)) - 1
    marked = [i for i, p in enumerate(facts['pieces']) if np.any(p == facts['marker'])]
    _require(len(marked) == 1, "Exactly one marked piece required")
    anchor = marked[0]
    chosen = [None] * len(candidates)
    answers = []

    def visit(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            board = np.empty((side, side), dtype=np.int8)
            stations = []
            for p in chosen:
                for pos, color in p[1]:
                    board.flat[pos] = color
                stations.extend(p[2])
            stations.sort()
            if not all(_separated(a, b) for i, a in enumerate(stations) for b in stations[i + 1:]):
                return
            if geometry_only:
                answers.append((board, stations, None))
                return
            routes, _ = _routes(board, stations, facts['key'], facts['carriers'], limit - len(answers))
            for route in routes:
                answers.append((board, stations, route))
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in candidates[i]:
                if candidate[0] & first and not candidate[0] & occupied:
                    chosen[i] = candidate
                    visit(occupied | candidate[0], rest)
                    if len(answers) >= limit:
                        return
            chosen[i] = None

    for p in candidates[anchor]:
        chosen[anchor] = p
        visit(p[0], tuple(i for i in range(len(candidates)) if i != anchor))
        if len(answers) >= limit:
            break
    return answers


def _decode(input_grid):
    facts = _parse(input_grid)
    answers = _search(facts)
    _require(len(answers) == 1, "The input must have exactly one joint assembly and route")
    board, stations, route = answers[0]
    facts.update(board=board, stations=stations, route=route)
    return _paint(board, stations, route), facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible input only, independent of seed and demonstrations."""
    return _decode(input_grid)[0]


def _random_shape(rng, width, max_extent):
    for _ in range(80):
        h = w = max_extent
        mask = np.zeros((h, w), dtype=bool)
        if width == 2:
            mask[:2, :2] = True
        else:
            mask[0, 0] = True
        area = int(rng.integers(max(5, width ** 2 + 2), min(h * w - 1, 14) + 1))
        while int(mask.sum()) < area:
            frontier = sorted({(r + dr, c + dc) for r, c in np.argwhere(mask)
                               for dr, dc in EDGE if 0 <= r + dr < h and 0 <= c + dc < w
                               and not mask[r + dr, c + dc]})
            mask[frontier[int(rng.integers(len(frontier)))]] = True
        p = _tight(mask.astype(np.int8), mask) >= 0
        if len(_orientations(p.astype(np.int8))) == 4 and min(p.shape) >= 2:
            return p
    # An asymmetric staircase, with a 2x2 station position in either width mode.
    return np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool)


def _partition(rng, side, count, width):
    extent = min(4, max(3, (side - 2) // 2))
    shared = _random_shape(rng, width, extent)
    shapes = [shared, shared.copy()]
    shapes += [_random_shape(rng, width, extent) for _ in range(count - 3)]
    owners = np.zeros((side, side), dtype=np.int8)
    for i, shape in enumerate(shapes, 1):
        for _ in range(100):
            p = np.rot90(shape, int(rng.integers(4)))
            h, w = p.shape
            r, c = int(rng.integers(side - h + 1)), int(rng.integers(side - w + 1))
            points = [(r + int(y), c + int(x)) for y, x in np.argwhere(p)]
            if (0, 0) in points:
                continue
            if any(np.any(owners[max(0, y-1):min(side, y+2), max(0, x-1):min(side, x+2)])
                   for y, x in points):
                continue
            for pos in points:
                owners[pos] = i
            break
        else:
            return None
    if len(_components(owners == 0, EDGE)) != 1:
        return None
    if _tight(owners, owners == 0).shape != (side, side):
        return None
    return owners


def _patch_cells(p):
    r, c, width = p[:3]
    return {(y, x) for y in range(r, r + width) for x in range(c, c + width)}


def _separated(a, b):
    return (a[0] + a[2] < b[0] or b[0] + b[2] < a[0]
            or a[1] + a[2] < b[1] or b[1] + b[2] < a[1])


def _clear_route(route, side):
    if not all(_separated(a, b) for i, a in enumerate(route) for b in route[i + 1:]):
        return False
    dots = {y * side + x for p in route for y, x in _patch_cells(p)}
    occupied = set()
    for a, b in zip(route, route[1:]):
        band = _band(a, b, side)
        if not band or 0 in band or dots.intersection(band) or occupied.intersection(band):
            return False
        occupied.update(band)
    return True


def _choose_route(rng, owners, width, length, corresponding=False):
    side = len(owners)
    positions, by_owner = [], {}
    for r in range(side - width + 1):
        for c in range(side - width + 1):
            region = owners[r:r+width, c:c+width]
            if (r, c) != (0, 0) and np.all(region == region[0, 0]):
                p = (r, c, width)
                positions.append(p)
                by_owner.setdefault(int(region[0, 0]), []).append(p)
    if not all(by_owner.get(i) for i in (0, 1, 2)):
        return None
    position_set = set(positions)
    corresponding_pairs = []
    if corresponding:
        bounds = [np.argwhere(owners == i).min(axis=0) for i in (1, 2)]
        masks = [_tight(owners, owners == i) >= 0 for i in (1, 2)]
        turns = next(k for k in range(4) if np.array_equal(np.rot90(masks[0], k), masks[1]))
        for a in by_owner[1]:
            local = np.zeros(masks[0].shape, dtype=bool)
            r, c = a[0] - int(bounds[0][0]), a[1] - int(bounds[0][1])
            local[r:r+width, c:c+width] = True
            target = np.argwhere(np.rot90(local, turns)).min(axis=0) + bounds[1]
            b = (int(target[0]), int(target[1]), width)
            corresponding_pairs.append((a, b))
    for _ in range(140):
        if corresponding:
            a, b = corresponding_pairs[int(rng.integers(len(corresponding_pairs)))]
        else:
            a = by_owner[1][int(rng.integers(len(by_owner[1])))]
            b = by_owner[2][int(rng.integers(len(by_owner[2])))]
        if abs(a[0] - b[0]) <= width or abs(a[1] - b[1]) <= width:
            continue
        elbows = [(a[0], b[1], width), (b[0], a[1], width)]
        rng.shuffle(elbows)
        for e in elbows:
            if e not in position_set or owners[e[0], e[1]] != 0:
                continue
            starts = [g for g in by_owner[0] if _band(g, a, side)]
            rng.shuffle(starts)
            for g in starts[:20]:
                route = [g, a, e, b]
                if not _clear_route(route, side):
                    continue
                if rng.random() < .5:
                    route.reverse()
                for _ in range(length - 4):
                    options = [p for p in positions if _band(route[-1], p, side)
                               and _clear_route(route + [p], side)]
                    if not options:
                        break
                    route.append(options[int(rng.integers(len(options)))])
                if len(route) == length:
                    return route, positions
    return None


def _scatter(rng, pieces, background, anchor_rotation, marker):
    rotations = list(map(int, rng.integers(4, size=len(pieces))))
    for i, p in enumerate(pieces):
        if np.any(p == marker):
            rotations[i] = anchor_rotation
        else:
            # All inserted silhouettes are asymmetric; demonstrate rotation.
            rotations[i] = int(rng.integers(1, 4))
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
    for _ in range(12):
        canvas = np.full((26, 28), -1, dtype=np.int8)
        blocked = np.zeros(canvas.shape, dtype=bool)
        order = sorted(range(len(pieces)), key=lambda i: -int(np.sum(oriented[i] >= 0)))
        okay = True
        for i in order:
            p = oriented[i]
            h, w = p.shape
            coords = np.argwhere(p >= 0)
            options = []
            for r in range(27 - h):
                for c in range(29 - w):
                    if not np.any(blocked[r:r+h, c:c+w][p >= 0]):
                        options.append((r, c))
            if not options:
                okay = False
                break
            r0, c0 = options[int(rng.integers(len(options)))]
            view = canvas[r0:r0+h, c0:c0+w]
            view[p >= 0] = p[p >= 0]
            for r, c in coords:
                y, x = r0 + int(r), c0 + int(c)
                blocked[max(0, y-1):min(26, y+2), max(0, x-1):min(28, x+2)] = True
        if okay:
            return _tight(canvas, canvas >= 0)
    return None


def _make_pair(rng, side, count, width, length, distractors, anchor_rotation, palette, corresponding=False):
    owners = _partition(rng, side, count, width)
    if owners is None:
        return None
    picked = _choose_route(rng, owners, width, length, corresponding)
    if picked is None:
        return None
    route, positions = picked
    background, marker = map(int, palette[:2])
    carriers = list(map(int, palette[2:2+count]))
    pool = list(map(int, palette[2+count:]))
    k = min(4 if length == 4 else 3, len(pool), length)
    key = pool[:k] + [int(rng.choice(pool[:k])) for _ in range(length - k)]
    rng.shuffle(key)
    if key == key[::-1]:
        return None
    board = np.asarray(carriers, dtype=np.int8)[owners]
    board[0, 0] = marker
    stations = [p + (color,) for p, color in zip(route, key)]
    occupied_bands = {p for a, b in zip(route, route[1:]) for p in _band(a, b, side)}
    for _ in range(distractors):
        options = [p for p in positions if all(_separated(p, q) for q in stations)
                   and not any(y*side+x in occupied_bands for y, x in _patch_cells(p))]
        # An unused station must tempt an extra aligned connection. Thus the
        # second demonstration explicitly rejects connecting all visible dots.
        dots = {y*side+x for q in stations for y, x in _patch_cells(q)}
        options = [p for p in options if any(
            (band := _band(p, q, side)) and 0 not in band and not dots.intersection(band)
            for q in stations)]
        if not options:
            return None
        p = options[int(rng.integers(len(options)))]
        stations.append(p + (int(rng.choice(key)),))
    for r, c, p, color in stations:
        board[r:r+p, c:c+p] = color
    if any(np.sum(board[owners == i] == carriers[i]) < 2 for i in range(count)):
        return None
    pieces = [_tight(board, owners == i) for i in range(count)]
    # Check the joint semantics before spending work scattering the pieces.
    facts = dict(pieces=pieces, side=side, marker=marker, key=tuple(key), carriers=tuple(carriers))
    if len(_search(facts, geometry_only=True)) < 2:
        return None
    answers = _search(facts)
    if len(answers) != 1 or not np.array_equal(answers[0][0], board):
        return None
    expected = _paint(board, stations, tuple(range(length)))
    if not np.array_equal(_paint(*answers[0]), expected):
        return None
    if corresponding and not any(len({int(owners.flat[p]) for p in _band(a, b, side)}) >= 2
                                 for a, b in zip(route, route[1:])):
        return None
    scattered = _scatter(rng, pieces, background, anchor_rotation, marker)
    if scattered is None:
        return None
    h, w = scattered.shape
    key_width = 2 * length - 1
    H = h + 4 + int(rng.integers(min(2, 26 - h) + 1))
    W = max(w + 2, key_width + 2)
    W += int(rng.integers(min(3, 30 - W) + 1))
    top = int(rng.integers(1, H - h - 2))
    left = int(rng.integers(1, W - w))
    key_row = int(rng.integers(top + h + 1, H - 1))
    key_col = int(rng.integers(1, W - key_width))
    grid = np.full((H, W), background, dtype=np.int8)
    grid[top:top+h, left:left+w] = np.where(scattered >= 0, scattered, background)
    grid[key_row, key_col:key_col+key_width:2] = key
    return dict(input=grid, output=expected)


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(name + " outside allowed range")
    return int(value)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    width = int(rng.choice([1, 1, 1, 2]))
    held = int(rng.integers(4))
    training_rotations = list(map(int, rng.permutation([k for k in range(4) if k != held])))
    stable_palette = bool(rng.random() < .25)
    palette = rng.permutation(10)
    puzzle, seen = dict(train=[], test=[]), set()
    for split, total in (('train', num_train), ('test', num_test)):
        for index in range(total):
            train = split == 'train'
            length = (4 if index == 0 else 5 if index == 1 else int(rng.integers(4, 7))) if train else int(rng.integers(6, 8))
            count = 3 if train and index == 0 else int(rng.integers(3, 6))
            distractors = 0 if train and index == 0 else 1 if train and index == 1 else int(rng.integers(3))
            rotation = training_rotations[index % 3] if train else held
            for attempt in range(240):
                side = int(rng.integers(8, 10)) if train else int(rng.integers(10, 13))
                if width == 2:
                    side = int(rng.integers(10, 12)) if train else int(rng.integers(12, 14))
                if not stable_palette:
                    palette = rng.permutation(10)
                pair = _make_pair(rng, side, count, width, length, distractors, rotation, palette,
                                  corresponding=train and index == 0)
                if pair is not None and _signature(pair['input']) not in seen:
                    seen.add(_signature(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError(f'Could not construct a joint-unique map: {split}/{index}, width={width}, count={count}, length={length}')
    return puzzle


def _features(pair):
    expected, f = _decode(pair['input'])
    _require(np.array_equal(expected, _grid(pair['output'])), 'Incorrect route-constrained assembly')
    geometric = _search(f, limit=2, geometry_only=True)
    f['ambiguous_geometry'] = len(geometric) >= 2
    sites = lambda stations: tuple(sorted(p[:3] for p in stations))
    f['same_sites_alternative'] = any(not np.array_equal(board, f['board'])
                                     and sites(stations) == sites(f['stations'])
                                     for board, stations, route in geometric)
    f['unused'] = len(f['stations']) - len(f['key'])
    f['areas'] = tuple(sorted(int(np.sum(p >= 0)) for p in f['pieces']))
    f['cross_seam'] = False
    for a, b in zip(f['route'], f['route'][1:]):
        band = _band(f['stations'][a], f['stations'][b], f['side'])
        if len({int(f['board'].flat[p]) for p in band}) >= 2:
            f['cross_seam'] = True
    routes, edges = _routes(f['board'], f['stations'], f['key'], f['carriers'])
    route_edges = {frozenset((a, b)) for a, b in zip(f['route'], f['route'][1:])}
    f['extra_visible_edge'] = any(frozenset((a, b)) not in route_edges for a, b in edges)
    painted = {p for a, b in zip(f['route'], f['route'][1:])
               for p in _band(f['stations'][a], f['stations'][b], f['side'])}
    unchanged = np.ones(f['board'].shape, dtype=bool)
    unchanged.flat[list(painted)] = False
    out = _grid(pair['output'])
    _require(np.array_equal(out[unchanged], f['board'][unchanged]), 'Changed a non-connector cell')
    _require(all(int(f['board'].flat[p]) in f['carriers'] for p in painted),
             'A connector overwrote a station or the anchor')
    # Recover each whole colored pattern by its carrier's placement mask.
    # Labels can repeat, so color-thresholding the finished board cannot do this.
    candidates = _placements(f)
    changes = []
    for piece, placements in zip(f['pieces'], candidates):
        matching = [p for p in placements if all(f['board'].flat[pos] == color for pos, color in p[1])]
        _require(matching, 'Lost rigid piece')
        # Carrier uniqueness identifies its actual placement even with repeated labels.
        p = matching[0]
        mask = np.zeros_like(f['board'], dtype=bool)
        for pos, color in p[1]:
            mask.flat[pos] = True
        changes.append(not np.array_equal(piece, _tight(f['board'], mask)))
    f['rotation_required'] = any(changes)
    f['structure'] = (f['side'], f['width'], f['areas'], tuple((r, c, p) for r, c, p, color in f['stations']),
                      tuple(f['route']))
    return f


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'Invalid example counts')
    seen, info = set(), {}
    for split in ('train', 'test'):
        info[split] = []
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys')
            output = _grid(pair['output'])
            sig = _signature(pair['input'])
            _require(sig not in seen, 'Duplicate input')
            seen.add(sig)
            f = _features(pair)
            _require(f['ambiguous_geometry'], 'The route must resolve actual assembly ambiguity')
            _require(f['rotation_required'], 'Rigid rotations must be demonstrated')
            _require(np.any(output != f['board']), 'Route drawing must change carrier cells')
            _require(int(output[0, 0]) == f['marker'], 'Anchor was overwritten')
            info[split].append(f)
    train, test = info['train'], info['test']
    _require(any(f['unused'] for f in train), 'Demonstrate unused stations')
    _require(any(len(set(f['key'])) < len(f['key']) for f in train), 'Demonstrate repeated labels')
    _require(any(f['cross_seam'] for f in train), 'Demonstrate a connection across a piece seam')
    _require(train[0]['same_sites_alternative'], 'The first key must disambiguate identical station sites')
    _require(train[1]['extra_visible_edge'], 'Demonstrate excluding a tempting extra connection')
    _require(len({f['width'] for f in train + test}) == 1, 'Keep station width consistent within a puzzle')


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    records = []
    puzzles = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        records.extend((split, _features(pair)) for split in ('train', 'test') for pair in p[split])
    features = [f for s, f in records]
    _require(len({f['structure'] for f in features}) >= 180, 'Insufficient geometric variety')
    _require({f['width'] for f in features} == {1, 2}, 'Exercise both band widths')
    _require({len(f['pieces']) for f in features} == {3, 4, 5}, 'Exercise piece counts')
    _require(len({f['grid'].shape for f in features}) >= 30, 'Vary scattering and margins')
    _require(len({f['areas'] for f in features}) >= 50, 'Vary piece shapes and areas')
    _require(all(f['ambiguous_geometry'] for f in features), 'A must disambiguate every example')
    _require(_equal(puzzles[0], generate_puzzle(0)), 'Non-deterministic generation')
    _require(not _equal(puzzles[0], puzzles[1]), 'Different seeds must differ')
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(700 + a + b, a, b)
        validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Corrupted anchor accepted')


if __name__ == '__main__':
    self_test()
