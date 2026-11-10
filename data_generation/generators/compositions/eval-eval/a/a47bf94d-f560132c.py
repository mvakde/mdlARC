"""Torn circuit boards: jointly assemble footprints and reconnect their cables.

Only NumPy and the standard library are used. Generation constructs a marked
board before cutting it; solve reconstructs only from the scattered input.
No files, examples, random state, or construction metadata are read by solve.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations
from math import isqrt
import hashlib
import json
import numpy as np

BG, WIRE, REGISTER, RECEIVER = 0, 5, 7, 9
INKS = (1, 2, 3, 4, 6, 8)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(np.asarray(sorted(cells), dtype=int))
    return result


def _parse(input_grid):
    g = _grid(input_grid)
    components = _components(g != BG)
    _require(len(components) == 4, "expected four disconnected board pieces")
    for a, b in combinations(components, 2):
        _require(int(np.max(np.abs(a[:, None] - b[None, :]), axis=2).min()) >= 2,
                 "pieces need a full background-cell gap")
    counts = Counter(map(int, g[g != BG]))
    carrier = [v for v in INKS if counts[v] > 1]
    labels = [v for v in INKS if counts[v] == 1]
    _require(len(carrier) == 1 and len(labels) == 4, "one common substrate and four singleton source colors required")
    _require(counts[RECEIVER] == 4 and counts[REGISTER] == 4, "four receivers and one four-cell registration patch required")
    carrier = carrier[0]
    pieces, anchor = [], None
    for i, cells in enumerate(components):
        r, c = cells.min(axis=0)
        rr, cc = cells.max(axis=0) + 1
        tile = g[r:rr, c:cc].copy()
        # Other pieces can lie in this bounding box's concavities.
        keep = np.zeros(tile.shape, dtype=bool)
        keep[cells[:, 0] - r, cells[:, 1] - c] = True
        tile[~keep] = BG
        _require(np.count_nonzero(tile == RECEIVER) == 1, "each piece needs one receiver")
        _require(sum(np.count_nonzero(tile == v) for v in labels) == 1, "each piece needs one source")
        _require(bool(np.any(tile == carrier)), "each piece must show the common substrate")
        if np.any(tile == REGISTER):
            _require(anchor is None, "registration patch split across pieces")
            loc = np.argwhere(tile == REGISTER)
            p, q = loc.min(axis=0)
            _require(set(map(tuple, loc)) == {(p, q), (p+1, q), (p, q+1), (p+1, q+1)}, "registration patch must be 2x2")
            anchor = i
        pieces.append(tile)
    _require(anchor is not None, "missing registered anchor")
    pieces = [pieces[anchor]] + [p for i, p in enumerate(pieces) if i != anchor]
    area = sum(int(np.count_nonzero(p)) for p in pieces)
    side = isqrt(area)
    _require(side * side == area and side <= 30, "combined footprint area must be square")
    return dict(grid=g, pieces=pieces, side=side, labels=labels, carrier=carrier)


def _placements(tile, side, corner, anchor=False):
    result, seen = [], set()
    corners = ((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))
    for turns in ((0,) if anchor else range(4)):
        a = np.rot90(tile, turns).copy()
        h, w = a.shape
        if h > side or w > side:
            continue
        r = 0 if corner < 2 else side - h
        c = 0 if corner % 2 == 0 else side - w
        cells = [(int(y+r), int(x+c)) for y, x in np.argwhere(a != BG)]
        hits = [p in cells for p in corners]
        if sum(hits) != 1 or not hits[corner]:
            continue
        signature = (r, c, a.shape, a.tobytes())
        if signature in seen:
            continue
        seen.add(signature)
        mask = sum(1 << (y*side+x) for y, x in cells)
        result.append((mask, a, r, c, turns))
    return result


def _geometric_assemblies(info):
    """Every legal corner placement; no circuit information is used here."""
    side, pieces = info['side'], info['pieces']
    table = {(i, c): _placements(p, side, c, i == 0)
             for i, p in enumerate(pieces) for c in range(4) if (i == 0) == (c == 0)}
    full = (1 << (side * side)) - 1
    for base in table.get((0, 0), []):
        chosen = [(0, base)]

        def visit(corner, remaining, used):
            if corner == 4:
                if used == full:
                    yield tuple(chosen)
                return
            for i in remaining:
                for candidate in table[i, corner]:
                    if candidate[0] & used:
                        continue
                    chosen.append((i, candidate))
                    yield from visit(corner+1, tuple(j for j in remaining if j != i), used | candidate[0])
                    chosen.pop()

        yield from visit(1, (1, 2, 3), base[0])


def _paint_assembly(info, placement):
    side = info['side']
    board = np.zeros((side, side), dtype=np.int8)
    owners = np.full((side, side), -1, dtype=np.int8)
    for i, (_, a, r, c, _) in placement:
        rr, cc = np.nonzero(a)
        board[rr+r, cc+c] = a[rr, cc]
        owners[rr+r, cc+c] = i
    return board, owners


def _edge(a, b):
    return (a, b) if a < b else (b, a)


def _trace(board, owners, labels):
    """Directional graph decoder: degree-four contacts are straight crossings."""
    conductive = set(map(tuple, np.argwhere(np.isin(board, [WIRE, RECEIVER] + list(labels)))))
    graph = {p: [q for dr, dc in DIRS if (q := (p[0]+dr, p[1]+dc)) in conductive] for p in conductive}
    for p, neighbors in graph.items():
        allowed = (2, 4) if board[p] == WIRE else (1,)
        _require(len(neighbors) in allowed, "broken cable, branching bend, or nonterminal end")
    sources = {v: tuple(map(int, np.argwhere(board == v)[0])) for v in labels}
    used_edges, reached, routes, colors = set(), set(), [], {}
    for color, start in sources.items():
        previous, p = None, start
        path = [start]
        while True:
            if p != start and board[p] != WIRE:
                _require(board[p] == RECEIVER and p not in reached, "route must connect one source to one unused receiver")
                reached.add(p)
                owner = int(owners[p])
                _require(owner not in colors, "a piece has multiple receivers")
                colors[owner] = color
                break
            neighbors = graph[p]
            if previous is None:
                q = neighbors[0]
            elif len(neighbors) == 4:
                q = (2*p[0]-previous[0], 2*p[1]-previous[1])
                _require(q in neighbors, "crossing does not continue straight")
            else:
                q = next(q for q in neighbors if q != previous)
            edge = _edge(p, q)
            _require(edge not in used_edges, "cycle or reused cable strand")
            used_edges.add(edge)
            previous, p = p, q
            path.append(p)
            _require(len(path) <= 2 * board.size, "cable loop")
        routes.append((color, tuple(path)))
    all_edges = {_edge(p, q) for p in graph for q in graph[p]}
    _require(used_edges == all_edges and len(reached) == 4 and set(colors) == set(range(4)), "unattached circuit or missing receiver")
    out = np.asarray([colors[i] for i in range(4)], dtype=np.int8)[owners]
    crossings = [p for p in graph if len(graph[p]) == 4]
    seams = {_edge(int(owners[a]), int(owners[b])) for _, path in routes for a, b in zip(path, path[1:]) if owners[a] != owners[b]}
    remote = sum(owners[path[0]] != owners[path[-1]] for _, path in routes)
    bends = sum(sum((b[0]-a[0], b[1]-a[1]) != (c[0]-b[0], c[1]-b[1])
                    for a, b, c in zip(path, path[1:], path[2:])) for _, path in routes)
    return dict(output=out, routes=routes, crossings=crossings, seams=seams, remote=int(remote), bends=bends)


def _analyze(input_grid):
    info = _parse(input_grid)
    solutions, geometric = [], 0
    for placement in _geometric_assemblies(info):
        geometric += 1
        board, owners = _paint_assembly(info, placement)
        try:
            result = _trace(board, owners, info['labels'])
        except (ValueError, StopIteration):
            continue
        solutions.append(dict(result, board=board, owners=owners, placement=placement))
        _require(len(solutions) <= 1, "more than one valid circuit assembly")
    _require(len(solutions) == 1, "no valid circuit assembly")
    info.update(solutions[0])
    info['geometric_assemblies'] = geometric
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the sole corner tiling with valid cables, then paint by receivers."""
    return _analyze(input_grid)['output'].copy()


def _connected_owner(owner):
    return all(len(_components(owner == i)) == 1 for i in range(4))


def _partition(rng, side, kind):
    if kind == 0:
        y = x = side // 2
        owner = np.empty((side, side), dtype=np.int8)
        owner[:y, :x], owner[:y, x:] = 0, 1
        owner[y:, :x], owner[y:, x:] = 2, 3
        return owner
    y = int(rng.integers(3, side-2))
    x = [int(rng.integers(2, side-2)), int(rng.integers(2, side-2))]
    owner = np.empty((side, side), dtype=np.int8)
    for r in range(side):
        half = int(r >= y)
        if rng.random() < .65:
            x[half] = int(np.clip(x[half] + rng.integers(-1, 2), 2, side-2))
        owner[r, :x[half]], owner[r, x[half]:] = 2*half, 2*half+1
    if kind >= 2:
        corners = {(0, 0), (0, side-1), (side-1, 0), (side-1, side-1)}
        for _ in range(3*side):
            r, c = int(rng.integers(side)), int(rng.integers(side))
            if (r, c) in corners:
                continue
            dr, dc = DIRS[int(rng.integers(4))]
            if not (0 <= r+dr < side and 0 <= c+dc < side):
                continue
            old, new = int(owner[r, c]), int(owner[r+dr, c+dc])
            owner[r, c] = new
            if not _connected_owner(owner):
                owner[r, c] = old
    return owner


def _fine_path(path):
    cells = [path[0]]
    for a, b in zip(path, path[1:]):
        cells.append(((a[0]+b[0])//2, (a[1]+b[1])//2))
        cells.append(b)
    return cells


def _find_path(rng, start, finish, vertices, reserved, occupied, used_edges, blocked):
    """Bounded randomized DFS on a coarse orthogonal mesh, allowing crossings."""
    distance = lambda p: (abs(p[0]-finish[0])+abs(p[1]-finish[1]))//2
    maximum = min(14, distance(start) + int(rng.choice([0, 2, 2, 4])))
    visited, path, budget = {start}, [start], [1200]

    def visit(p, previous):
        budget[0] -= 1
        if budget[0] < 0 or len(path)-1+distance(p) > maximum:
            return None
        if p == finish:
            return path.copy()
        if p in occupied:
            axes = occupied[p]
            if len(axes) != 2 or previous is None:
                return None
            incoming = (p[0]-previous[0], p[1]-previous[1])
            if incoming in axes or (-incoming[0], -incoming[1]) in axes:
                return None
            choices = [(p[0]+incoming[0], p[1]+incoming[1])]
        else:
            choices = [(p[0]+2*dr, p[1]+2*dc) for dr, dc in DIRS]
            noise = {q: float(rng.random()) for q in choices}
            choices.sort(key=lambda q: distance(q) + 3*noise[q])
        for q in choices:
            mid = ((p[0]+q[0])//2, (p[1]+q[1])//2)
            if q not in vertices or q in visited or q in blocked or mid in blocked:
                continue
            if q in reserved and q != finish:
                continue
            if _edge(p, q) in used_edges:
                continue
            if q in occupied:
                axes = occupied[q]
                direction = (q[0]-p[0], q[1]-p[1])
                if len(axes) != 2 or direction in axes or (-direction[0], -direction[1]) in axes:
                    continue
                # Only an already straight strand can be crossed.
                a, b = tuple(axes)
                if a != (-b[0], -b[1]):
                    continue
            visited.add(q)
            path.append(q)
            answer = visit(q, p)
            if answer is not None:
                return answer
            path.pop()
            visited.remove(q)
        return None

    return visit(start, None)


def _make_circuit(rng, owner, require_crossing, avoid_pairing=None):
    side = len(owner)
    register_options = [(r, c) for r in range(side-1) for c in range(side-1)
                        if np.all(owner[r:r+2, c:c+2] == 0)]
    if not register_options:
        return None
    reg = register_options[int(rng.integers(len(register_options)))]
    blocked = {(reg[0]+dr, reg[1]+dc) for dr in range(2) for dc in range(2)}
    offset = int(rng.integers(1, 3))
    vertices = {(r, c) for r in range(offset, side-1, 2) for c in range(offset, side-1, 2)} - blocked
    pools = [sorted(p for p in vertices if owner[p] == i) for i in range(4)]
    if any(len(p) < 2 for p in pools):
        return None
    for _ in range(48):
        sources, sinks = [], []
        for pool in pools:
            a, b = map(int, rng.choice(len(pool), 2, replace=False))
            sources.append(pool[a]); sinks.append(pool[b])
        permutation = list(map(int, rng.permutation(4)))
        if permutation == list(range(4)) or permutation == avoid_pairing:
            continue
        reserved = set(sources + sinks)
        occupied, used_edges, paths = {}, set(), [None]*4
        for i in map(int, rng.permutation(4)):
            path = _find_path(rng, sources[i], sinks[permutation[i]], vertices, reserved, occupied, used_edges, blocked)
            if path is None:
                break
            paths[i] = path
            for a, b in zip(path, path[1:]):
                used_edges.add(_edge(a, b))
            for j, p in enumerate(path):
                directions = set()
                for k in (j-1, j+1):
                    if 0 <= k < len(path):
                        q = path[k]
                        directions.add((q[0]-p[0], q[1]-p[1]))
                occupied.setdefault(p, set()).update(directions)
        else:
            carrier, *labels = map(int, rng.choice(INKS, 5, replace=False))
            board = np.full(owner.shape, carrier, dtype=np.int8)
            for path in paths:
                for p in _fine_path(path):
                    board[p] = WIRE
            board[reg[0]:reg[0]+2, reg[1]:reg[1]+2] = REGISTER
            for i, p in enumerate(sources):
                board[p] = labels[i]
            for p in sinks:
                board[p] = RECEIVER
            try:
                measured = _trace(board, owner, labels)
            except (ValueError, StopIteration):
                continue
            if require_crossing and not measured['crossings']:
                continue
            if len(measured['seams']) < 3:
                continue
            # The four pieces participate in a connected cross-seam circuit.
            reachable = {0}
            for _ in range(3):
                reachable |= {v for a, b in measured['seams'] if a in reachable or b in reachable for v in (a, b)}
            if len(reachable) < 4:
                continue
            piece_colors = [None]*4
            for source, receiver_piece in enumerate(permutation):
                piece_colors[receiver_piece] = labels[source]
            expected = np.asarray(piece_colors, dtype=np.int8)[owner]
            _require(np.array_equal(measured['output'], expected), "constructed paths and pixel tracer disagree")
            return board, expected
    return None


def _scatter(rng, tiles):
    """Free placement when it fits; enumerate shelf layouts as a bounded fallback."""
    oriented = [p.copy() for p in tiles]
    attempts = []
    # General scattered layouts use footprint halos, not fixed quadrant slots.
    if rng.random() < .65:
        for _ in range(10):
            h, w = int(rng.integers(23, 31)), int(rng.integers(23, 31))
            taken = np.zeros((h, w), dtype=bool)
            offsets = [None]*4
            for i in map(int, rng.permutation(4)):
                a = oriented[i]
                hh, ww = a.shape
                rr, cc = np.nonzero(a)
                for _ in range(35):
                    r, c = int(rng.integers(h-hh+1)), int(rng.integers(w-ww+1))
                    if taken[rr+r, cc+c].any():
                        continue
                    offsets[i] = (r, c)
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            taken[np.clip(rr+r+dr, 0, h-1), np.clip(cc+c+dc, 0, w-1)] = True
                    break
                else:
                    break
            else:
                attempts.append(offsets)
                break
    if not attempts:
        for transpose in (False, True):
            dims = [p.shape[::-1] if transpose else p.shape for p in oriented]
            for order in permutations(range(4)):
                for cuts in range(8):
                    groups, current = [], []
                    for j, i in enumerate(order):
                        current.append(i)
                        if j == 3 or (cuts >> j) & 1:
                            groups.append(current); current = []
                    gap = int(rng.integers(1, 4))
                    hs = [max(dims[i][0] for i in group) for group in groups]
                    ws = [sum(dims[i][1] for i in group)+gap*(len(group)-1) for group in groups]
                    if sum(hs)+gap*(len(groups)-1) > 28 or max(ws) > 28:
                        continue
                    offsets, r = [None]*4, 0
                    for group, height in zip(groups, hs):
                        c = 0
                        for i in group:
                            y = r + int(rng.integers(height-dims[i][0]+1))
                            offsets[i] = (c, y) if transpose else (y, c)
                            c += dims[i][1]+gap
                        r += height+gap
                    attempts.append(offsets)
        if not attempts:
            dims = [p.shape for p in oriented]
            second = max(dims[0][0], dims[1][0])+1
            attempts.append([(0, 0), (0, dims[0][1]+1),
                             (second, 0), (second, dims[2][1]+1)])
    offsets = attempts[int(rng.integers(len(attempts)))]
    minima = np.min(offsets, axis=0)
    offsets = [(r-int(minima[0]), c-int(minima[1])) for r, c in offsets]
    extent = [max(offsets[i][axis]+p.shape[axis] for i, p in enumerate(oriented)) for axis in (0, 1)]
    pads = []
    for length in extent:
        spare = 30-length
        before = int(rng.integers(min(3, spare)+1))
        after = int(rng.integers(min(3, spare-before)+1))
        pads.append((before, after))
    grid = np.zeros(tuple(extent[i]+sum(pads[i]) for i in (0, 1)), dtype=np.int8)
    for i, a in enumerate(oriented):
        r, c = offsets[i]
        rr, cc = np.nonzero(a)
        grid[rr+r+pads[0][0], cc+c+pads[1][0]] = a[rr, cc]
    return grid


def _build_pair(rng, side, kind, turns, require_crossing=False, avoid_pairing=None):
    for _ in range(220):
        owner = _partition(rng, side, kind)
        circuit = _make_circuit(rng, owner, require_crossing, avoid_pairing)
        if circuit is None:
            continue
        board, expected = circuit
        tiles = []
        for i in range(4):
            coords = np.argwhere(owner == i)
            lo, hi = coords.min(axis=0), coords.max(axis=0)+1
            a = np.where(owner == i, board, BG)[lo[0]:hi[0], lo[1]:hi[1]]
            tiles.append(np.rot90(a, turns[i]).copy())
        # Check uniqueness before scattering. This is a joint check, not a
        # stipulation that the original construction must be the answer.
        inp = _scatter(rng, tiles)
        try:
            analysis = _analyze(inp)
        except ValueError:
            continue
        if not np.array_equal(analysis['output'], expected):
            continue
        return {'input': inp, 'output': expected.copy()}
    raise RuntimeError("bounded circuit construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    _require(_integer(num_train) and 2 <= num_train <= 10, "num_train must be 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4 and num_train > num_test, "num_test must be 1..4 and smaller than num_train")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if split == 'train' and index == 0:
                side = 12 if regime == 1 else int(rng.choice([10, 12]))
                kind, turns = 0, [0, 0, 0, 0]
            elif split == 'train' and index == 1:
                side = int(rng.integers(11, 14)) if regime == 1 else int(rng.integers(10, 13))
                kind, turns = 1, [0, 1, 2, 3]
            else:
                if split == 'test' and regime == 0:
                    side = 13
                elif split == 'test' and regime == 1:
                    side = 10
                elif split == 'test':
                    side = int(rng.choice([p['output'].shape[0] for p in puzzle['train']]))
                elif regime == 0:
                    side = int(rng.integers(10, 13))
                elif regime == 1:
                    side = int(rng.integers(11, 14))
                else:
                    side = int(rng.integers(10, 14))
                kind = int(rng.integers(3))
                turns = [0] + list(map(int, rng.integers(0, 4, 3)))
                if split == 'test' and regime == 2:
                    kind = 2
                    turns = [0] + [int(rng.integers(1, 4))]*3
            for _ in range(12):
                avoid = None
                if split == 'train' and index == 1:
                    avoid = _metrics(_analyze(puzzle['train'][0]['input']))['pairing']
                pair = _build_pair(rng, side, kind, turns,
                                   require_crossing=(split == 'train' and index == 0),
                                   avoid_pairing=avoid)
                if split == 'train' and index == 1:
                    before = _parse(puzzle['train'][0]['input'])
                    current = _parse(pair['input'])
                    if current['carrier'] == before['carrier']:
                        unused = next(v for v in INKS if v not in current['labels'] and v != current['carrier'])
                        pair['input'][pair['input'] == current['carrier']] = unused
                signature = (pair['input'].shape, pair['input'].tobytes())
                if signature not in seen:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            seen.add(signature)
            puzzle[split].append(pair)
    return puzzle


def _metrics(a):
    areas = tuple(int(np.count_nonzero(p)) for p in a['pieces'])
    local_colors = [next(v for v in a['labels'] if np.any(p == v)) for p in a['pieces']]
    local = np.asarray(local_colors, dtype=np.int8)[a['owners']]
    corner = {piece: c for c, (piece, _) in enumerate(a['placement'])}
    pairing = [None]*4
    for _, path in a['routes']:
        pairing[corner[int(a['owners'][path[0]])]] = corner[int(a['owners'][path[-1]])]
    normalized = np.asarray([corner[i] for i in range(4)], dtype=np.int8)[a['owners']]
    irregular = int(sum(np.count_nonzero(p) < p.size for p in a['pieces']))
    return dict(side=a['side'], canvas=list(a['grid'].shape), areas=list(areas),
                area_ratio=max(areas)/min(areas), geometric=a['geometric_assemblies'],
                crossings=len(a['crossings']), bends=a['bends'], remote=a['remote'],
                seams=len(a['seams']), local_shortcut_fails=not np.array_equal(local, a['output']),
                rotations=[int(p[1][4]) for p in a['placement']], carrier=a['carrier'],
                labels=a['labels'], pairing=pairing, irregular=irregular,
                cut_hash=hashlib.sha256(bytes(normalized.shape)+normalized.tobytes()).hexdigest())


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "exact train/test schema required")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test']), "illegal pair counts")
    seen, evidence = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "exact pair schema required")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = (inp.shape, inp.tobytes())
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            a = _analyze(inp)
            _require(np.array_equal(a['output'], out), "output disagrees with input-derived circuit assembly")
            _require(out.shape == (a['side'], a['side']), "output area mismatch")
            _require(set(map(int, np.unique(out))) == set(a['labels']), "output must use exactly the four visible source colors")
            _require(a['remote'] >= 2 and len(a['seams']) >= 3, "both transport and cross-seam reconstruction must matter")
            reached = {0}
            for _ in range(3):
                reached |= {v for x, y in a['seams'] if x in reached or y in reached for v in (x, y)}
            _require(len(reached) == 4, "the cross-seam circuit must involve all four pieces")
            metric = _metrics(a)
            _require(metric['local_shortcut_fails'], "local source coloring must differ")
            if split == 'train':
                evidence.append(metric)
    _require(any(m['geometric'] > 1 and m['crossings'] > 0 for m in evidence), "training must demonstrate crossing-based circuit disambiguation")
    _require(any(any(k != 0 for k in m['rotations']) for m in evidence), "training must demonstrate rotation")
    _require(len({tuple(m['pairing']) for m in evidence}) >= 2, "training must refute a fixed corner-to-corner color permutation")
    _require(len({m['carrier'] for m in evidence}) >= 2, "training must expose temporary substrate color")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape)); h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, bounded in-memory feature summaries; never write a corpus."""
    records, hashes = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        hashes.add(_digest(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _analyze(pair['input'])
                records.append(_metrics(a))
                _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), "list input solver failed")
    _require(len(hashes) == 42, "seeds must produce different puzzles")
    _require(len({tuple(m['canvas']) for m in records}) >= 30, "scattering shapes collapsed")
    _require({m['side'] for m in records} == {10, 11, 12, 13}, "assembly scale variation missing")
    _require({m['carrier'] for m in records} == set(INKS), "substrate palette variation missing")
    _require(len({tuple(sorted(m['areas'])) for m in records}) >= 30, "piece-area variation missing")
    _require(len({m['cut_hash'] for m in records}) >= 100, "normalized cuts collapsed")
    _require(len({tuple(m['pairing']) for m in records}) >= 12, "routed permutations collapsed")
    _require(any(m['irregular'] == 0 for m in records) and any(m['irregular'] == 4 for m in records), "silhouette spectrum missing")
    _require(any(m['crossings'] == 0 for m in records) and any(m['crossings'] >= 2 for m in records), "crossing spectrum missing")
    _require(any(m['bends'] >= 4 for m in records), "bent routes missing")
    _require({m['remote'] for m in records} >= {2, 3, 4}, "route permutations collapsed")
    _require(any(m['area_ratio'] >= 3 for m in records), "unequal pieces missing")
    _require({k for m in records for k in m['rotations']} == {0, 1, 2, 3}, "quarter-turn coverage missing")
    first = generate_puzzle(123, 2, 1)
    _require(_digest(first) == _digest(generate_puzzle(123, 2, 1)), "determinism failed")
    # Metamorphic checks change the presentation, not the intended answer.
    rng = np.random.default_rng(70123)
    a = _analyze(first['train'][1]['input'])
    rearranged = _scatter(rng, [p if i == 0 else np.rot90(p, int(rng.integers(4)))
                               for i, p in enumerate(a['pieces'])])
    _require(np.array_equal(solve(rearranged), a['output']), "scattering/rotation changed the answer")
    lut = np.arange(10, dtype=np.int8)
    lut[list(INKS)] = np.roll(np.asarray(INKS, dtype=np.int8), 1)
    _require(np.array_equal(solve(lut[a['grid']]), lut[a['output']]), "ink-role permutation changed the rule")
    unused = next(v for v in INKS if v not in a['labels'] and v != a['carrier'])
    changed_substrate = a['grid'].copy()
    changed_substrate[changed_substrate == a['carrier']] = unused
    _require(np.array_equal(solve(changed_substrate), a['output']), "temporary substrate affected output colors")
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(42000+10*train+test, train, test)
            validate_generated_puzzle(p)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    damaged['test'][0]['output'][0, 0] = WIRE
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(hashes),
                input_shapes=len({tuple(m['canvas']) for m in records}),
                piece_area_profiles=len({tuple(sorted(m['areas'])) for m in records}),
                distinct_cuts=len({m['cut_hash'] for m in records}),
                distinct_pairings=len({tuple(m['pairing']) for m in records}),
                geometric_ambiguity=sum(m['geometric'] > 1 for m in records),
                crossing_range=[min(m['crossings'] for m in records), max(m['crossings'] for m in records)],
                bend_range=[min(m['bends'] for m in records), max(m['bends'] for m in records)],
                remote_counts=sorted({m['remote'] for m in records}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
