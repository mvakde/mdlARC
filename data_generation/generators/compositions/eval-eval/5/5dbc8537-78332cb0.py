"""Anchored circuit mosaics: joint rectangle packing and geometric seam matching.

Only standard-library modules and NumPy are used. The public solver reads the
input alone. Generation paints a construction, then checks its unique rendering
by a separate search over the visible inventory and stencil.
"""

from collections import Counter
from itertools import combinations
import json

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _key(grid):
    a = np.asarray(grid)
    return a.shape, tuple(map(int, a.flat))


def _neighbors(point):
    y, x = point
    return [(y + dy, x + dx) for dy, dx in _DIRS]


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            for q in _neighbors(stack.pop()):
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _cells(mask):
    return set(map(tuple, np.argwhere(mask).tolist()))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _corners(a):
    return {int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1])}


def _parse(input_grid):
    """Infer the divider, semantic panels, anchors, and rectangular inventory."""
    grid = _grid(input_grid)
    candidates = []
    for divider in np.unique(grid):
        for axis in (0, 1):
            lines = np.flatnonzero(np.all(grid == divider, axis=1 - axis))
            if len(lines) != 1:
                continue
            cut = int(lines[0])
            if cut == 0 or cut == grid.shape[axis] - 1:
                continue
            parts = ((grid[:cut], grid[cut + 1:]) if axis == 0 else
                     (grid[:, :cut], grid[:, cut + 1:]))
            if any(np.any(p == divider) for p in parts):
                continue
            for board, tray in (parts, parts[::-1]):
                if len(_corners(board)) != 1 or len(_corners(tray)) != 1:
                    continue
                background = next(iter(_corners(board)))
                vacancy = next(iter(_corners(tray)))
                if background == vacancy or np.any(tray == background):
                    continue
                other = set(map(int, np.unique(board))) - {background, vacancy}
                if len(other) != 1:
                    continue
                ink = next(iter(other))
                mask, anchors = board == vacancy, _cells(board == ink)
                occupied = _cells(mask)
                if (len(anchors) != 2 or len(_components(occupied)) != 1
                        or mask[0].any() or mask[-1].any()
                        or mask[:, 0].any() or mask[:, -1].any()
                        or any(sum(q in occupied for q in _neighbors(a)) != 1
                               for a in anchors)
                        or len(_components(anchors)) != 2):
                    continue
                tiles = []
                valid = True
                for part in _components(_cells(tray != vacancy)):
                    ys, xs = zip(*part)
                    tile = tray[min(ys):max(ys) + 1, min(xs):max(xs) + 1].copy()
                    ink_cells = _cells(tile == ink)
                    if (tile.size != len(part) or np.any(tile == vacancy)
                            or len(_components(ink_cells)) != 1
                            or any(tile[r, c] == ink for r, c in
                                   ((0, 0), (0, -1), (-1, 0), (-1, -1)))
                            or not (np.any(tile[0] == ink) or np.any(tile[-1] == ink)
                                    or np.any(tile[:, 0] == ink) or np.any(tile[:, -1] == ink))):
                        valid = False
                        break
                    tiles.append(tile)
                if not valid or not 2 <= len(tiles) <= 10:
                    continue
                candidates.append(dict(board=board.copy(), tray=tray.copy(), tiles=tiles,
                                       background=background, vacancy=vacancy, ink=ink,
                                       divider=int(divider), mask=mask.copy(), anchors=anchors))
    if len(candidates) != 1:
        raise ValueError("Input must have one visible stencil/tray interpretation")
    return candidates[0]


class _SearchLimit(Exception):
    pass


def _cover(parsed, *, ports=True, connectivity=True, limit=None):
    """Enumerate at most two distinct rendered answers, grouping identical copies.

    A placement carries requirements for the ink status of neighboring vacancy
    cells across its boundary. Static boundaries must agree with visible anchors
    or be closed. This compares global positions, including partial edge joins.
    """
    board, mask, ink = parsed['board'], parsed['mask'], parsed['ink']
    coords = sorted(_cells(mask))
    index = {p: i for i, p in enumerate(coords)}
    groups, lookup = [], {}
    for tile in parsed['tiles']:
        k = _key(tile)
        if k in lookup:
            groups[lookup[k]]['count'] += 1
        else:
            lookup[k] = len(groups)
            groups.append(dict(pattern=tile, count=1))
    placements, by_cell = [], [[] for _ in coords]
    bh, bw = board.shape
    for gi, group in enumerate(groups):
        tile = group['pattern']
        h, w = tile.shape
        for y in range(bh - h + 1):
            for x in range(bw - w + 1):
                if not mask[y:y + h, x:x + w].all():
                    continue
                assigned = tuple((index[(y + r, x + c)], int(tile[r, c]))
                                 for r in range(h) for c in range(w))
                bits = sum(1 << i for i, _ in assigned)
                requirements, valid = [], True
                if ports:
                    for r in range(h):
                        for c in range(w):
                            mark = bool(tile[r, c] == ink)
                            for dr, dc in _DIRS:
                                if 0 <= r + dr < h and 0 <= c + dc < w:
                                    continue
                                q = y + r + dr, x + c + dc
                                if q in index:
                                    requirements.append((index[q], mark))
                                else:
                                    fixed = (0 <= q[0] < bh and 0 <= q[1] < bw
                                             and board[q] == ink)
                                    if mark != fixed:
                                        valid = False
                    if not valid:
                        continue
                pi = len(placements)
                placements.append((gi, y, x, bits, assigned, tuple(requirements)))
                for i, _ in assigned:
                    by_cell[i].append(pi)
    remaining = [g['count'] for g in groups]
    rendered = [-1] * len(coords)
    full = (1 << len(coords)) - 1
    outcomes, path = {}, []
    nodes = 0

    def compatible(pi, covered):
        gi, _, _, bits, _, req = placements[pi]
        return (remaining[gi] > 0 and not bits & covered
                and all(not (covered >> i & 1) or (rendered[i] == ink) == mark
                        for i, mark in req))

    def visit(covered):
        nonlocal nodes
        nodes += 1
        if limit is not None and nodes > limit:
            raise _SearchLimit("Construction search budget exhausted")
        if covered == full:
            out = board.copy()
            for p, color in zip(coords, rendered):
                out[p] = color
            if ports and connectivity and len(_components(_cells(out == ink))) != 1:
                return
            outcomes.setdefault(_key(out), (out, tuple(path)))
            return
        best = None
        for i in range(len(coords)):
            if covered >> i & 1:
                continue
            legal = [pi for pi in by_cell[i] if compatible(pi, covered)]
            if not legal:
                return
            if best is None or len(legal) < len(best):
                best = legal
                if len(best) == 1:
                    break
        for pi in best:
            gi, y, x, bits, assigned, _ = placements[pi]
            remaining[gi] -= 1
            for i, color in assigned:
                rendered[i] = color
            path.append((gi, y, x))
            visit(covered | bits)
            path.pop()
            for i, _ in assigned:
                rendered[i] = -1
            remaining[gi] += 1
            if len(outcomes) >= 2:
                return

    visit(0)
    return outcomes, groups, nodes


def _decode(input_grid, limit=None):
    parsed = _parse(input_grid)
    outcomes, groups, nodes = _cover(parsed, limit=limit)
    if len(outcomes) != 1:
        raise ValueError("Expected exactly one rendered circuit completion")
    output, path = next(iter(outcomes.values()))
    parsed.update(output=output, path=path, groups=groups, nodes=nodes)
    return parsed


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the completed stencil, deriving every convention from this input."""
    return _decode(input_grid)['output'].copy()


def _contact(a, b):
    """Pairs of neighboring non-corner edge cells in two rectangles."""
    y, x, h, w = a
    v, u, k, l = b
    result = []
    if x + w == u or u + l == x:
        for r in range(max(y + 1, v + 1), min(y + h - 1, v + k - 1)):
            result.append(((r, x + w - 1), (r, u)) if x + w == u else
                          ((r, x), (r, u + l - 1)))
    if y + h == v or v + k == y:
        for c in range(max(x + 1, u + 1), min(x + w - 1, u + l - 1)):
            result.append(((y + h - 1, c), (v, c)) if y + h == v else
                          ((y, c), (v + k - 1, c)))
    return result


def _grow(rng, count):
    height, width = int(rng.integers(17, 23)), int(rng.integers(9, 13))
    h0, w0 = int(rng.integers(3, 7)), int(rng.integers(3, 6))
    rects = [(int(rng.integers(4, height - h0 - 3)),
              int(rng.integers(1, width - w0)), h0, w0)]
    mask = np.zeros((height, width), dtype=bool)
    y, x, h, w = rects[0]
    mask[y:y + h, x:x + w] = True
    for i in range(1, count):
        h, w = (h0, w0) if i == 1 else (int(rng.integers(3, 7)), int(rng.integers(3, 6)))
        options = set()
        for a in rects:
            y, x, k, l = a
            for xx in range(x - w + 3, x + l - 2):
                options.add((y - h, xx, h, w))
                options.add((y + k, xx, h, w))
            for yy in range(y - h + 3, y + k - 2):
                options.add((yy, x - w, h, w))
                options.add((yy, x + l, h, w))
        options = sorted(p for p in options
                         if 0 <= p[0] <= height - h and 0 <= p[1] <= width - w
                         and not mask[p[0]:p[0] + h, p[1]:p[1] + w].any()
                         and any(_contact(a, p) for a in rects))
        if not options:
            return None
        p = options[int(rng.integers(len(options)))]
        rects.append(p)
        y, x = p[:2]
        mask[y:y + h, x:x + w] = True
    ys, xs = np.where(mask)
    top, left = int(ys.min()), int(xs.min())
    pads = list(map(int, rng.integers(1, 4, size=4)))
    board_shape = (int(ys.max()) - top + 1 + pads[0] + pads[1],
                   int(xs.max()) - left + 1 + pads[2] + pads[3])
    if board_shape[0] > 29 or board_shape[1] > 19:
        return None
    rects = [(y - top + pads[0], x - left + pads[2], h, w) for y, x, h, w in rects]
    return rects, board_shape


def _circuit(rng, rects, board_shape, required, capability, branch):
    count = len(rects)
    possible = [(i, j, _contact(a, b)) for i, a in enumerate(rects)
                for j, b in enumerate(rects) if i < j and _contact(a, b)]
    rng.shuffle(possible)
    parents = list(range(count))

    def root(i):
        while i != parents[i]:
            i = parents[i]
        return i

    edges, leftovers = [], []
    for i, j, contacts in possible:
        a, b = root(i), root(j)
        if a != b:
            parents[a] = b
            edges.append((i, j, contacts))
        else:
            leftovers.append((i, j, contacts))
    if len(edges) != count - 1:
        return None
    for edge in leftovers:
        if rng.random() < .30:
            edges.append(edge)
    degree = Counter(i for edge in edges for i in edge[:2])
    if branch and max(degree.values()) < 3:
        return None
    kinds = ['single'] * len(edges)
    available = set(range(len(edges)))
    for kind in sorted(required, key=lambda k: {'single': 1, 'thick': 2, 'separated': 3}[k], reverse=True):
        needed = {'single': 1, 'thick': 2, 'separated': 3}[kind]
        options = sorted(i for i in available if len(edges[i][2]) >= needed)
        if not options:
            return None
        i = options[int(rng.integers(len(options)))]
        kinds[i] = kind
        available.remove(i)
    for i in sorted(available):
        choices = ['single']
        if capability >= 1 and len(edges[i][2]) >= 2:
            choices.append('thick')
        if capability >= 2 and len(edges[i][2]) >= 3:
            choices.append('separated')
        kinds[i] = choices[int(rng.integers(len(choices)))]
    ports = [set() for _ in rects]
    for (i, j, contacts), kind in zip(edges, kinds):
        if kind == 'single':
            selected = [contacts[int(rng.integers(len(contacts)))]]
        elif kind == 'thick':
            length = int(rng.integers(2, min(3, len(contacts)) + 1))
            start = int(rng.integers(len(contacts) - length + 1))
            selected = contacts[start:start + length]
        else:
            choices = [(a, b) for a in range(len(contacts)) for b in range(a + 2, len(contacts))]
            a, b = choices[int(rng.integers(len(choices)))]
            selected = [contacts[a], contacts[b]]
        for p, q in selected:
            ports[i].add(p)
            ports[j].add(q)
    mask = np.zeros(board_shape, dtype=bool)
    for y, x, h, w in rects:
        mask[y:y + h, x:x + w] = True
    occupied = _cells(mask)
    anchor_options = []
    for i, (y, x, h, w) in enumerate(rects):
        sides = [((y, c), (y - 1, c)) for c in range(x + 1, x + w - 1)]
        sides += [((y + h - 1, c), (y + h, c)) for c in range(x + 1, x + w - 1)]
        sides += [((r, x), (r, x - 1)) for r in range(y + 1, y + h - 1)]
        sides += [((r, x + w - 1), (r, x + w)) for r in range(y + 1, y + h - 1)]
        for p, q in sides:
            if (q not in occupied and 0 <= q[0] < board_shape[0]
                    and 0 <= q[1] < board_shape[1]
                    and sum(t in occupied for t in _neighbors(q)) == 1):
                anchor_options.append((i, p, q))
    rng.shuffle(anchor_options)
    if not anchor_options:
        return None
    first = anchor_options[0]
    choices = [a for a in anchor_options if a[0] != first[0]
               and sum(abs(a[2][k] - first[2][k]) for k in (0, 1)) >= 4]
    if not choices:
        return None
    second = choices[int(rng.integers(len(choices)))]
    for i, p, q in (first, second):
        ports[i].add(p)
    local_ports = [{(r - y, c - x) for r, c in pp}
                   for pp, (y, x, h, w) in zip(ports, rects)]
    return mask, {first[2], second[2]}, local_ports


def _tile(rng, shape, ports, ink, palette):
    h, w = shape
    mode = int(rng.integers(5))
    base = int(rng.choice(palette))
    tile = np.full(shape, base, dtype=np.int8)
    if len(palette) > 1:
        yy, xx = np.indices(shape)
        other = int(rng.choice([c for c in palette if c != base]))
        if mode == 1:
            tile[(yy + xx) % 2 == 1] = other
        elif mode == 2:
            tile[xx % 2 == 1] = other
        elif mode == 3:
            tile = rng.choice(palette, size=shape).astype(np.int8)
        elif mode == 4:
            tile[-1, -1] = other
    body = {(int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))}

    def connect(start):
        target = sorted(body)[int(rng.integers(len(body)))]
        r, c = start
        body.add((r, c))
        while (r, c) != target:
            choices = []
            if r != target[0]:
                choices.append((r + (1 if r < target[0] else -1), c))
            if c != target[1]:
                choices.append((r, c + (1 if c < target[1] else -1)))
            r, c = choices[int(rng.integers(len(choices)))]
            body.add((r, c))

    for r, c in sorted(ports):
        connect((min(h - 2, max(1, r)), min(w - 2, max(1, c))))
    for _ in range(int(rng.integers(4))):
        connect((int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))))
    if rng.random() < .30:
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if rng.random() < .25 and any(q in body for q in _neighbors((r, c))):
                    body.add((r, c))
    for p in body | ports:
        tile[p] = ink
    return tile


def _extensions(board, rects, tiles, background, ink):
    """Trace-equivalent rectangles that would spill onto board background."""
    options = []
    bh, bw = board.shape
    for i, ((y, x, h, w), tile) in enumerate(zip(rects, tiles)):
        base = int(tile[0, 0])
        if y > 0 and np.all(board[y - 1, x:x + w] == background) and not np.any(tile[0] == ink):
            options.append((i, y - 1, x, np.vstack((np.full((1, w), base), tile)).astype(np.int8)))
        if y + h < bh and np.all(board[y + h, x:x + w] == background) and not np.any(tile[-1] == ink):
            options.append((i, y, x, np.vstack((tile, np.full((1, w), base))).astype(np.int8)))
        if x > 0 and np.all(board[y:y + h, x - 1] == background) and not np.any(tile[:, 0] == ink):
            options.append((i, y, x - 1, np.hstack((np.full((h, 1), base), tile)).astype(np.int8)))
        if x + w < bw and np.all(board[y:y + h, x + w] == background) and not np.any(tile[:, -1] == ink):
            options.append((i, y, x, np.hstack((tile, np.full((h, 1), base))).astype(np.int8)))
    return options


def _scatter(rng, tiles, height, max_width, vacancy):
    minimum = max(t.shape[1] for t in tiles) + 1
    if minimum > max_width or max(t.shape[0] for t in tiles) > height:
        return None
    cache = {}
    for _ in range(25):
        width = int(rng.integers(minimum, max_width + 1))
        forbidden = sum(1 << k for k in (0, width - 1, (height - 1) * width, height * width - 1))
        order = sorted(range(len(tiles)), key=lambda i: (-tiles[i].size, i))
        positions = {}
        for i in order:
            h, w = tiles[i].shape
            k = width, h, w
            if k not in cache:
                choices = []
                for y in range(height - h + 1):
                    for x in range(width - w + 1):
                        bits = sum(((1 << w) - 1) << ((y + r) * width + x) for r in range(h))
                        x0, x1 = max(0, x - 1), min(width, x + w + 1)
                        moat = sum(((1 << (x1 - x0)) - 1) << (r * width + x0)
                                   for r in range(max(0, y - 1), min(height, y + h + 1)))
                        choices.append((y, x, bits, moat))
                cache[k] = choices
            choices = [p for p in cache[k] if not p[2] & forbidden]
            if not choices:
                break
            y, x, _, moat = choices[int(rng.integers(len(choices)))]
            positions[i] = (y, x)
            forbidden |= moat
        if len(positions) == len(tiles):
            tray = np.full((height, width), vacancy, dtype=np.int8)
            for i, (y, x) in positions.items():
                h, w = tiles[i].shape
                tray[y:y + h, x:x + w] = tiles[i]
            return tray
    return None


def _network_valid(parsed, out, path=None):
    """Check seams and global connectivity from the recovered cell ownership."""
    owner = np.full(out.shape, -1, dtype=int)
    path = parsed['path'] if path is None else path
    for i, (gi, y, x) in enumerate(path):
        h, w = parsed['groups'][gi]['pattern'].shape
        if np.any(owner[y:y + h, x:x + w] >= 0):
            return False
        owner[y:y + h, x:x + w] = i
    if not np.array_equal(owner >= 0, parsed['mask']):
        return False
    ink = parsed['ink']
    for y, x in _cells(parsed['mask']):
        for v, u in _neighbors((y, x)):
            if not (0 <= v < out.shape[0] and 0 <= u < out.shape[1]):
                if out[y, x] == ink:
                    return False
            elif owner[v, u] != owner[y, x] and (out[y, x] == ink) != (out[v, u] == ink):
                return False
    return len(_components(_cells(out == ink))) == 1


def _witnesses(parsed):
    out, path, groups = parsed['output'], parsed['path'], parsed['groups']
    b_witness = False
    rects, tiles = [], []
    used = Counter(gi for gi, _, _ in path)
    for gi, y, x in path:
        tile = groups[gi]['pattern']
        rects.append((y, x, *tile.shape))
        tiles.append(tile)
    for i, j in combinations(range(len(path)), 2):
        if tiles[i].shape != tiles[j].shape or np.array_equal(tiles[i], tiles[j]):
            continue
        wrong = out.copy()
        for k, replacement in ((i, tiles[j]), (j, tiles[i])):
            y, x, h, w = rects[k]
            wrong[y:y + h, x:x + w] = replacement
        if not _network_valid(parsed, wrong):
            b_witness = True
            break
    a_witness = False
    for _, y, x, bigger in _extensions(parsed['board'], rects, tiles,
                                      parsed['background'], parsed['ink']):
        if any(g['count'] > used[i] and np.array_equal(g['pattern'], bigger)
               for i, g in enumerate(groups)):
            # The extension adds only non-ink substrate on background. Replacing
            # this tile therefore preserves the complete global ink image.
            a_witness = True
            break
    return a_witness, b_witness


def _make_pair(rng, count, required, capability, decoy, branch):
    for _ in range(240):
        geometry = _grow(rng, count)
        if geometry is None:
            continue
        rects, shape = geometry
        circuit = _circuit(rng, rects, shape, required, capability, branch)
        if circuit is None:
            continue
        mask, anchors, ports = circuit
        colors = list(map(int, rng.permutation(10)))
        background, vacancy, ink, divider = colors[:4]
        palette = colors[4:4 + int(rng.integers(1, 7))]
        tiles = [_tile(rng, (h, w), pp, ink, palette)
                 for (_, _, h, w), pp in zip(rects, ports)]
        if np.array_equal(tiles[0], tiles[1]):
            continue
        board = np.full(shape, background, dtype=np.int8)
        board[mask] = vacancy
        for p in anchors:
            board[p] = ink
        out = board.copy()
        for (y, x, h, w), tile in zip(rects, tiles):
            out[y:y + h, x:x + w] = tile
        supplied = list(tiles)
        if decoy:
            choices = _extensions(board, rects, tiles, background, ink)
            if not choices:
                continue
            supplied.append(choices[int(rng.integers(len(choices)))][3])
        tray = _scatter(rng, supplied, shape[0], 29 - shape[1], vacancy)
        if tray is None:
            continue
        delimiter = np.full((shape[0], 1), divider, dtype=np.int8)
        parts = (board, delimiter, tray) if rng.random() < .5 else (tray, delimiter, board)
        inp = np.concatenate(parts, axis=1)
        turns = int(rng.integers(4))
        inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
        if rng.random() < .5:
            inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        try:
            parsed = _decode(inp, limit=16000)
        except (ValueError, _SearchLimit):
            continue
        if not np.array_equal(parsed['output'], out):
            continue
        a, b = _witnesses(parsed)
        if not b or (decoy and not a):
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError("Could not construct a unique circuit mosaic within bounded attempts")


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    capability = int(rng.integers(3))
    required = {'single'} | ({'thick'} if capability >= 1 else set())
    required |= {'separated'} if capability == 2 else set()
    puzzle, seen = {'train': [], 'test': []}, set()
    plans = [(int(rng.integers(3, 6)), set(), True, False),
             (5, required, False, True)]
    plans += [(int(rng.integers(3, 6)), set(), bool(rng.random() < .40), False)
              for _ in range(num_train - 2)]
    rng.shuffle(plans)
    plans += [(int(rng.integers(6, 9)), set(), bool(rng.random() < .45), False)
              for _ in range(num_test)]
    for index, (n, kinds, decoy, branch) in enumerate(plans):
        for _ in range(12):
            pair = _make_pair(rng, n, kinds, capability, decoy, branch)
            k = _key(pair['input'])
            if k not in seen:
                break
        else:
            raise RuntimeError('Distinct-input construction exhausted')
        seen.add(k)
        puzzle['train' if index < num_train else 'test'].append(pair)
    return puzzle


def _features(parsed):
    path, groups, ink = parsed['path'], parsed['groups'], parsed['ink']
    rects = [(y, x, *groups[gi]['pattern'].shape) for gi, y, x in path]
    kinds, degree, joins, partial = set(), Counter(), 0, False
    for i, j in combinations(range(len(rects)), 2):
        contacts = _contact(rects[i], rects[j])
        active = [(p, q) for p, q in contacts if parsed['output'][p] == ink]
        if not active:
            continue
        degree[i] += 1
        degree[j] += 1
        joins += 1
        # Vertical seam pairs differ in column and are indexed by row.
        positions = [p[0] if p[1] != q[1] else p[1] for p, q in active]
        kinds.add('single' if len(positions) == 1 else
                  'thick' if max(positions) - min(positions) + 1 == len(positions) else 'separated')
        a, b = rects[i], rects[j]
        if a[1] + a[3] == b[1] or b[1] + b[3] == a[1]:
            overlap = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
            partial |= overlap < max(a[2], b[2])
        else:
            overlap = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            partial |= overlap < max(a[3], b[3])
    return dict(count=len(path), supplied=len(parsed['tiles']), kinds=kinds,
                branch=max(degree.values(), default=0) >= 3,
                cycle=joins >= len(path), partial=bool(partial),
                dimensions=tuple(sorted((h, w) for y, x, h, w in rects)),
                roles=(parsed['background'], parsed['vacancy'], ink, parsed['divider']),
                shape=parsed['board'].shape, nodes=parsed['nodes'])


def validate_generated_puzzle(puzzle):
    """Check family semantics, finite copies, uniqueness, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, records, witnesses = set(), [], []
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        inp, out = _grid(pair['input']), _grid(pair['output'])
        k = _key(inp)
        assert k not in seen, 'inputs must be distinct'
        seen.add(k)
        parsed = _decode(inp)
        assert np.array_equal(out, parsed['output']), 'unique visible solution must equal label'
        assert out.shape == parsed['board'].shape
        assert np.array_equal(out[~parsed['mask']], parsed['board'][~parsed['mask']])
        assert not np.any(out[parsed['mask']] == parsed['vacancy'])
        assert _network_valid(parsed, out), 'all seams closed or matched; one connected circuit'
        used = Counter(gi for gi, _, _ in parsed['path'])
        assert all(n <= parsed['groups'][gi]['count'] for gi, n in used.items())
        for gi, y, x in parsed['path']:
            tile = parsed['groups'][gi]['pattern']
            h, w = tile.shape
            assert np.array_equal(out[y:y + h, x:x + w], tile), 'translate each full tile intact'
        records.append(_features(parsed))
        witnesses.append(_witnesses(parsed))
    train, test = records[:nt], records[nt:]
    assert any(a for a, b in witnesses[:nt]), 'training must show stencil-essential surplus'
    assert all(b for a, b in witnesses), 'same-size swaps must make the circuit wrong'
    assert any(r['branch'] for r in train), 'teach spatial branching before longer tests'
    demonstrated = set().union(*(r['kinds'] for r in train))
    tested = set().union(*(r['kinds'] for r in test))
    assert tested <= demonstrated, 'test port categories must be demonstrated'
    assert max(r['count'] for r in train) < min(r['count'] for r in test)


def _puzzle_key(puzzle):
    return tuple((_key(pair['input']), _key(pair['output']))
                 for split in ('train', 'test') for pair in puzzle[split])


def _connectivity_regression():
    """One anchored trace and one isolated loop; all local seams still match."""
    board = np.full((10, 8), 9, dtype=np.int8)
    board[1:9, 1:7] = 0
    board[2, 0] = board[2, 7] = 2
    line = np.full((3, 3), 3, dtype=np.int8)
    line[1, :] = 2
    left = np.full((5, 3), 3, dtype=np.int8)
    left[1:4, 1] = 2
    left[1, 2] = left[3, 2] = 2
    right = left[:, ::-1].copy()
    tray = np.zeros((10, 11), dtype=np.int8)
    tray[1:4, 1:4] = tray[1:4, 5:8] = line
    tray[5:10, 1:4], tray[5:10, 5:8] = left, right
    inp = np.concatenate((board, np.full((10, 1), 8, dtype=np.int8), tray), axis=1)
    parsed = _parse(inp)
    local, _, _ = _cover(parsed, connectivity=False)
    assert len(local) == 1
    out, _ = next(iter(local.values()))
    assert len(_components(_cells(out == 2))) == 2
    complete, _, _ = _cover(parsed)
    assert not complete, 'a separate closed loop is not an anchored completion'


def _inventory_regression():
    """A compatible two-tile member needs both visible identical physical copies."""
    board = np.full((5, 8), 9, dtype=np.int8)
    board[1:4, 1:7] = 0
    board[2, 0] = board[2, 7] = 2
    tile = np.full((3, 3), 3, dtype=np.int8)
    tile[1, :] = 2
    tray = np.zeros((5, 9), dtype=np.int8)
    tray[1:4, 1:4] = tray[1:4, 5:8] = tile
    inp = np.concatenate((board, np.full((5, 1), 8, dtype=np.int8), tray), axis=1)
    parsed = _decode(inp)
    assert len(parsed['groups']) == 1 and parsed['groups'][0]['count'] == 2
    assert len(parsed['path']) == 2 and _network_valid(parsed, parsed['output'])
    assert np.array_equal(solve(inp), parsed['output'])
    insufficient = dict(parsed, tiles=[tile])
    outcomes, _, _ = _cover(insufficient)
    assert not outcomes


def self_test():
    """Bounded 42-seed semantic/diversity checks and targeted negative checks."""
    signatures, features, inputs = set(), [], set()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if first is None:
            first = puzzle
        signatures.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                parsed = _decode(pair['input'])
                features.append(_features(parsed))
                inputs.add(_key(pair['input']))
                assert np.array_equal(solve(pair['input']), pair['output'])
    assert len(signatures) == 42 and len(inputs) == 210
    assert {r['count'] for r in features} == set(range(3, 9))
    assert {'single', 'thick', 'separated'} == set().union(*(r['kinds'] for r in features))
    assert any(r['cycle'] for r in features) and any(r['partial'] for r in features)
    assert any(r['supplied'] == r['count'] for r in features)
    assert any(r['supplied'] > r['count'] for r in features)
    assert len({r['shape'] for r in features}) >= 35
    assert len({r['dimensions'] for r in features}) >= 100
    for role in range(4):
        assert {r['roles'][role] for r in features} == set(range(10))
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(first)
    # Footprint-only packing has multiple rendered answers: B really matters.
    parsed = _decode(first['train'][0]['input'])
    alternatives, _, _ = _cover(parsed, ports=False, limit=30000)
    assert len(alternatives) == 2
    # Ink-only substitution keeps the network but overwrites fixed background:
    # A's footprint constraint changes the answer, rather than just its label.
    for pair in first['train']:
        a_parsed = _decode(pair['input'])
        if _witnesses(a_parsed)[0]:
            break
    used = Counter(gi for gi, _, _ in a_parsed['path'])
    rects = [(y, x, *a_parsed['groups'][gi]['pattern'].shape)
             for gi, y, x in a_parsed['path']]
    tiles = [a_parsed['groups'][gi]['pattern'] for gi, _, _ in a_parsed['path']]
    for _, y, x, bigger in _extensions(a_parsed['board'], rects, tiles,
                                      a_parsed['background'], a_parsed['ink']):
        if any(g['count'] > used[i] and np.array_equal(g['pattern'], bigger)
               for i, g in enumerate(a_parsed['groups'])):
            wrong = a_parsed['output'].copy()
            h, w = bigger.shape
            wrong[y:y + h, x:x + w] = bigger
            assert np.array_equal(wrong == a_parsed['ink'], a_parsed['output'] == a_parsed['ink'])
            assert np.any(wrong[~a_parsed['mask']] != a_parsed['board'][~a_parsed['mask']])
            break
    else:
        raise AssertionError('No trace-equivalent background-spilling alternative')
    _connectivity_regression()
    _inventory_regression()
    # Validators must reject changed anchor/board cells, not just trust labels.
    damaged = {s: [dict(input=p['input'].copy(), output=p['output'].copy())
                   for p in first[s]] for s in ('train', 'test')}
    y, x = next(iter(parsed['anchors']))
    damaged['train'][0]['output'][y, x] = parsed['background']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('Changed anchor accepted')
    # Missing copies cannot be reused to fill a board.
    insufficient = dict(parsed, tiles=[parsed['tiles'][0]])
    results, _, _ = _cover(insufficient)
    assert not results
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            if (nt, ne) in ((2, 1), (10, 4)):
                validate_generated_puzzle(generate_puzzle(100 + nt, nt, ne))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Compact, inspectable variation measurements; no files or state are needed.
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(signatures),
                board_shapes=len({r['shape'] for r in features}),
                dimension_multisets=len({r['dimensions'] for r in features}),
                branches=sum(r['branch'] for r in features),
                cycles=sum(r['cycle'] for r in features),
                partial_joins=sum(r['partial'] for r in features),
                maximum_search_nodes=max(r['nodes'] for r in features))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
