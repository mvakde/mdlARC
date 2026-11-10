"""Fuel relay jigsaw: rigid assembly reveals which local counter pays each step.

Standard library + NumPy only. No files, caches, latent solving state, or traces.
"""
from __future__ import annotations

from hashlib import sha256
from math import isqrt
import numpy as np

BG, START, FUEL, END, SPENT, ROAD = 0, 1, 2, 7, 8, 9
CARRIERS = (3, 4, 5, 6)
EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEAR = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), 'invalid dimensions')
    _need(np.issubdtype(g.dtype, np.integer) and bool(np.all((g >= 0) & (g <= 9))), 'invalid cells')
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.tobytes()


def _components(mask, neighbors=NEAR):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(cells)
    return result


def _tight(board, mask):
    cells = np.argwhere(mask)
    lo, hi = cells.min(0), cells.max(0) + 1
    b = board[lo[0]:hi[0], lo[1]:hi[1]]
    m = mask[lo[0]:hi[0], lo[1]:hi[1]]
    return np.where(m, b, -1).astype(np.int8)


def _rotations(piece):
    result, seen = [], set()
    for k in range(4):
        p = np.rot90(piece, k).copy()
        if _key(p) not in seen:
            seen.add(_key(p))
            result.append(p)
    return result


def _read_pieces(grid):
    _need(int(np.sum(grid == START)) == 1, 'one start marker required')
    components = _components(grid != BG)
    _need(3 <= len(components) <= 4, 'three or four separated pieces required')
    pieces, carriers = [], []
    for cells in components:
        mask = np.zeros(grid.shape, dtype=bool)
        for p in cells:
            mask[p] = True
        piece = _tight(grid, mask)
        colors = set(map(int, np.unique(piece))) & set(CARRIERS)
        _need(len(colors) == 1, 'each rigid piece needs one carrier')
        carriers.append(next(iter(colors)))
        _need(np.sum((piece == FUEL) | (piece == SPENT)) >= 2, 'piece lacks a counter')
        _need(np.sum(piece >= 0) >= 10, 'piece too small')
        pieces.append(piece)
    _need(len(set(carriers)) == len(carriers), 'distinct carriers required')
    area = int(np.sum(grid != BG))
    side = isqrt(area)
    _need(side * side == area and 7 <= side <= 14, 'area must define a square of side 7..14')
    return pieces, side


def _scan(t, b, l, r, direction):
    if direction == 'U':
        return [(y, x) for y in range(t, b) for x in range(l, r)]
    if direction == 'D':
        return [(y, x) for y in range(b-1, t-1, -1) for x in range(r-1, l-1, -1)]
    if direction == 'L':
        return [(y, x) for x in range(l, r) for y in range(b-1, t-1, -1)]
    return [(y, x) for x in range(r-1, l-1, -1) for y in range(t, b)]


def _bar(t, b, l, r, direction):
    if direction == 'U':
        return [(t-1, x) for x in range(l, r)]
    if direction == 'D':
        return [(b, x) for x in range(l, r)]
    if direction == 'L':
        return [(y, l-1) for y in range(t, b)]
    return [(y, r) for y in range(t, b)]


def _counters(board, owners, count):
    records = []
    for i in range(count):
        points = np.argwhere((owners == i) & ((board == FUEL) | (board == SPENT)))
        _need(len(points) >= 2, 'missing counter')
        t, l = map(int, points.min(0))
        b, r = map(int, points.max(0) + 1)
        _need(len(points) == (b-t)*(r-l), 'counter must be rectangular')
        _need(bool(np.all(owners[t:b, l:r] == i)), 'counter crosses seam')
        choices = []
        for direction in ('U', 'D', 'L', 'R'):
            bar = _bar(t, b, l, r, direction)
            if len(bar) >= 2 and all(0 <= y < len(board) and 0 <= x < len(board)
                                     and owners[y, x] == i and board[y, x] == END for y, x in bar):
                choices.append((direction, bar))
        _need(len(choices) == 1, 'counter needs one unambiguous leading bar')
        direction, bar = choices[0]
        coords = _scan(t, b, l, r, direction)
        values = [int(board[p]) for p in coords]
        advanced = values.count(SPENT)
        _need(values == [SPENT]*advanced + [FUEL]*(len(values)-advanced), 'invalid unary prefix')
        records.append(dict(coords=coords, advanced=advanced, capacity=len(coords),
                            direction=direction, bar=bar, bounds=(t, b, l, r)))
    return records


def _route(board):
    """Only the 7 touching the road is a destination; counter bars are isolated."""
    _need(board[0, 0] == START and np.sum(board == START) == 1, 'anchor must be upper-left')
    candidates = (board == ROAD) | (board == START) | (board == END)
    component = next(c for c in _components(candidates, EDGE) if (0, 0) in c)
    cells = set(component)
    destinations = [p for p in cells if board[p] == END]
    _need(len(destinations) == 1, 'one road destination required')
    _need(sum(board[p] == ROAD for p in cells) == int(np.sum(board == ROAD)), 'disconnected road')
    end = destinations[0]
    for p in cells:
        degree = sum((p[0]+dr, p[1]+dc) in cells for dr, dc in EDGE)
        _need(degree == (1 if p in ((0, 0), end) else 2), 'road must be a simple path')
    path, previous, current = [], None, (0, 0)
    while True:
        path.append(current)
        if current == end:
            break
        nxt = [(current[0]+dr, current[1]+dc) for dr, dc in EDGE
               if (current[0]+dr, current[1]+dc) in cells and (current[0]+dr, current[1]+dc) != previous]
        _need(len(nxt) == 1, 'ambiguous road')
        previous, current = current, nxt[0]
        _need(len(path) <= board.size, 'road cycle')
    _need(len(path) == len(cells), 'unused road cells')
    return path


def _facts(board, owners, count):
    counters, path = _counters(board, owners, count), _route(board)
    road_set = set(path)
    for counter in counters:
        _need(not any((y+dy, x+dx) in road_set for y, x in counter['bar']
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)), 'counter bar touches road')
    explained_bars = {p for counter in counters for p in counter['bar']}
    _need({tuple(map(int, p)) for p in np.argwhere(board == END)} == explained_bars | {path[-1]},
          'unexplained destination/bar cells')
    _need({int(owners[p]) for p in path} == set(range(count)), 'road must visit every piece')
    return counters, path


def _relay(board, owners, count):
    counters, path = _facts(board, owners, count)
    out = board.copy()
    used = [0]*count
    position = 0
    for j, p in enumerate(path[:-1]):
        owner = int(owners[p])
        counter = counters[owner]
        index = counter['advanced'] + used[owner]
        if index == counter['capacity']:
            break
        out[counter['coords'][index]] = SPENT
        used[owner] += 1
        position = j + 1
    out[0, 0] = ROAD
    out[path[position]] = START
    return out, dict(position=position, length=len(path)-1, used=used,
                     remaining=[c['capacity']-c['advanced'] for c in counters],
                     capacities=[c['capacity'] for c in counters],
                     directions=[c['direction'] for c in counters], path=path,
                     visits=[int(owners[p]) for p in path], counters=counters)


def _assemble(pieces, side):
    """Exact cover retains colored rotations, then checks the road and counters.

    Stop at two distinct valid colored/ownership assemblies. No random/search
    budget is used in solving; the small family bounds make exhaustive search finite.
    """
    n = len(pieces)
    anchor = next(i for i, p in enumerate(pieces) if np.any(p == START))
    candidates = []
    for i, piece in enumerate(pieces):
        options = []
        for oriented in _rotations(piece):
            h, w = oriented.shape
            if h > side or w > side:
                continue
            cells = [(int(y)*side+int(x), int(oriented[y, x])) for y, x in np.argwhere(oriented >= 0)]
            base = sum(1 << pos for pos, _ in cells)
            offsets = [0] if i == anchor and oriented[0, 0] == START else [] if i == anchor else [
                y*side+x for y in range(side-h+1) for x in range(side-w+1)]
            for offset in offsets:
                options.append((base << offset, offset, cells))
        _need(bool(options), 'piece cannot fit anchored square')
        candidates.append(options)
    full = (1 << (side*side))-1
    placements = [None]*n
    solutions, seen = [], set()

    def search(occupied, remaining):
        if len(solutions) >= 2:
            return
        if not remaining:
            if occupied != full:
                return
            board = np.empty((side, side), dtype=np.int8)
            owners = np.empty_like(board)
            for i, (_, offset, cells) in enumerate(placements):
                for pos, color in cells:
                    board.flat[pos+offset] = color
                    owners.flat[pos+offset] = i
            try:
                _facts(board, owners, n)
            except AssertionError:
                return
            key = board.tobytes(), owners.tobytes()
            if key not in seen:
                seen.add(key)
                solutions.append((board, owners))
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in candidates[i]:
                if candidate[0] & first and not candidate[0] & occupied:
                    placements[i] = candidate
                    search(occupied | candidate[0], rest)
                    if len(solutions) >= 2:
                        return
            placements[i] = None

    for candidate in candidates[anchor]:
        placements[anchor] = candidate
        search(candidate[0], tuple(i for i in range(n) if i != anchor))
        if len(solutions) >= 2:
            break
    _need(len(solutions) == 1, 'input must have one valid anchored assembly')
    return solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the assembled relay and final marker from this input alone."""
    grid = _grid(input_grid)
    pieces, side = _read_pieces(grid)
    board, owners = _assemble(pieces, side)
    return _relay(board, owners, len(pieces))[0]


def _partition(rng, side, count, style):
    if style == 'bands':
        # Procedural jagged interfaces, with independently moving seams.
        cuts = np.linspace(0, side, count+1).astype(int)
        labels = np.empty((side, side), dtype=np.int8)
        for y in range(side):
            if y:
                for k in range(1, count):
                    cuts[k] = int(np.clip(cuts[k]+rng.integers(-1, 2), cuts[k-1]+1, side-(count-k)))
            for i in range(count):
                labels[y, cuts[i]:cuts[i+1]] = i
        if rng.random() < .5:
            labels = labels.T.copy()
    else:
        # Unequal, arbitrary-seed growth gives wrapping and interior regions.
        labels = np.full((side, side), -1, dtype=np.int8)
        seeds = [0] + list(map(int, rng.choice(np.arange(1, side*side), count-1, replace=False)))
        weights = np.exp(rng.uniform(-.65, .65, count))
        frontier = [set() for _ in range(count)]
        sizes = [1]*count
        adjacent = []
        for y in range(side):
            for x in range(side):
                adjacent.append([(y+dy)*side+x+dx for dy, dx in EDGE
                                 if 0 <= y+dy < side and 0 <= x+dx < side])
        for i, p in enumerate(seeds):
            labels.flat[p] = i
        for i, p in enumerate(seeds):
            frontier[i].update(q for q in adjacent[p] if labels.flat[q] < 0)
        for _ in range(side*side-count):
            eligible = [i for i in range(count) if frontier[i]]
            i = min(eligible, key=lambda j: sizes[j]/weights[j]+float(rng.random()))
            options = sorted(frontier[i])
            p = options[int(rng.integers(len(options)))]
            labels.flat[p] = i
            sizes[i] += 1
            for f in frontier:
                f.discard(p)
            frontier[i].update(q for q in adjacent[p] if labels.flat[q] < 0)
    if any(len(_components(labels == i, EDGE)) != 1 or np.sum(labels == i) < 12 for i in range(count)):
        return None
    return labels


def _make_path(rng, side, labels, count, long):
    # Scale a simple coarse-grid walk by two. This keeps nonconsecutive road
    # cells from touching and still permits turns, seam reentry, and detours.
    size = (side+1)//2
    target = int(rng.integers(9, 15) if long else rng.integers(6, 11))
    coarse, occupied = [(0, 0)], {(0, 0)}
    for _ in range(target-1):
        y, x = coarse[-1]
        options = [(y+dy, x+dx) for dy, dx in EDGE if 0 <= y+dy < size and 0 <= x+dx < size
                   and (y+dy, x+dx) not in occupied]
        if not options:
            break
        q = options[int(rng.integers(len(options)))]
        coarse.append(q)
        occupied.add(q)
    path = [(0, 0)]
    for a, b in zip(coarse, coarse[1:]):
        path.append((a[0]+b[0], a[1]+b[1]))
        path.append((2*b[0], 2*b[1]))
    if len(path) < (15 if long else 11) or {int(labels[p]) for p in path} != set(range(count)):
        return None
    if not any((b[0]-a[0], b[1]-a[1]) != (c[0]-b[0], c[1]-b[1])
               for a, b, c in zip(path, path[1:], path[2:])):
        return None
    return path


def _stamp_options(labels, path, owner, needed):
    side = len(labels)
    road = np.zeros(labels.shape, dtype=bool)
    barred = road.copy()
    for y, x in path:
        road[y, x] = True
        barred[max(0, y-1):min(side, y+2), max(0, x-1):min(side, x+2)] = True
    integral = np.pad(((labels != owner) | road).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    choices = []
    for h in (1, 2, 3):
        for w in (2, 3, 4, 5):
            if h*w < max(2, needed):
                continue
            canonical = np.full((h+1, w), FUEL, dtype=np.int8)
            canonical[0] = END
            for k in range(4):
                stamp = np.rot90(canonical, k)
                H, W = stamp.shape
                sums = integral[H:, W:]-integral[:-H, W:]-integral[H:, :-W]+integral[:-H, :-W]
                bar_points = np.argwhere(stamp == END)
                for yy, xx in np.argwhere(sums == 0):
                    y, x = int(yy), int(xx)
                    if np.any(barred[y+bar_points[:, 0], x+bar_points[:, 1]]):
                        continue
                    choices.append((y, x, stamp.copy(), h*w))
    return choices


def _scatter(rng, pieces):
    oriented = [np.rot90(p, int(rng.integers(1, 4) if np.any(p == START) else rng.integers(4))).copy() for p in pieces]
    # Mask-based compact placement. Every attempt is finite; a rectangular
    # bounding-box arrangement is a guaranteed-fit final placement at side <=13.
    canvas = None
    for attempt in range(10):
        grid = np.zeros((30, 30), dtype=np.int8)
        blocked = np.zeros_like(grid, dtype=bool)
        order = sorted(range(len(pieces)), key=lambda i: -oriented[i].size) if not attempt else list(map(int, rng.permutation(len(pieces))))
        maxy = maxx = 0
        okay = True
        aspect = float(rng.uniform(.65, 1.5))
        for i in order:
            p = oriented[i]
            H, W = p.shape
            points = np.argwhere(p >= 0)
            choices = []
            for y in range(31-H):
                for x in range(31-W):
                    if np.any(blocked[y+points[:, 0], x+points[:, 1]]):
                        continue
                    hh, ww = max(maxy, y+H), max(maxx, x+W)
                    score = hh*ww+abs(hh-aspect*ww)*3+float(rng.random())*18
                    choices.append((score, y, x))
            if not choices:
                okay = False
                break
            _, y, x = min(choices)
            for r, c in points:
                rr, cc = y+int(r), x+int(c)
                grid[rr, cc] = p[r, c]
                blocked[max(0, rr-1):min(30, rr+2), max(0, cc-1):min(30, cc+2)] = True
            maxy, maxx = max(maxy, y+H), max(maxx, x+W)
        if okay:
            canvas = grid[:maxy, :maxx]
            break
    if canvas is None:
        canvas = np.zeros((29, 29), dtype=np.int8)
        for i, p in enumerate(oriented):
            y, x = (i//2)*15, (i%2)*15
            view = canvas[y:y+p.shape[0], x:x+p.shape[1]]
            view[p >= 0] = p[p >= 0]
    rr, cc = np.nonzero(canvas)
    canvas = canvas[rr.min():rr.max()+1, cc.min():cc.max()+1]
    H, W = canvas.shape
    extras = [int(rng.integers(min(4, 30-H)+1)), int(rng.integers(min(4, 30-W)+1))]
    top, left = int(rng.integers(extras[0]+1)), int(rng.integers(extras[1]+1))
    out = np.zeros((H+extras[0], W+extras[1]), dtype=np.int8)
    out[top:top+H, left:left+W] = canvas
    return out


def _make_pair(rng, side, count, regime, long, style):
    for attempt in range(320):
        labels = _partition(rng, side, count, style if attempt < 160 else 'growth')
        if labels is None:
            continue
        path = _make_path(rng, side, labels, count, long)
        if path is None:
            continue
        visits = [int(labels[p]) for p in path]
        runs = [x for j, x in enumerate(visits) if j == 0 or x != visits[j-1]]
        if regime == 'arrival' and not long and (len(runs) == len(set(runs)) or visits[0] == visits[-1]):
            continue
        needed = [sum(labels[p] == i for p in path[:-1]) for i in range(count)]
        stamps = []
        for i in range(count):
            options = _stamp_options(labels, path, i, needed[i])
            if not options:
                break
            # Mix short and broad counters; do not invariably choose maximum fuel.
            stamps.append(options[int(rng.integers(len(options)))])
        if len(stamps) != count:
            continue
        capacities = [s[3] for s in stamps]
        if regime == 'arrival' and sum(capacities) <= len(path)-1:
            continue
        remaining = [int(rng.integers(needed[i], capacities[i]+1)) for i in range(count)]
        if regime == 'arrival' and sum(remaining) == len(path)-1:
            i = next(i for i in range(count) if capacities[i] > needed[i])
            remaining[i] += 1
        if regime == 'fuel':
            positions = []
            for j in range(3, len(path)-2):
                owner = int(labels[path[j]])
                paid = sum(labels[p] == owner for p in path[:j])
                changes = sum(labels[a] != labels[b] for a, b in zip(path[:j], path[1:j+1]))
                prefix_runs = [x for k, x in enumerate(visits[:j+1]) if k == 0 or x != visits[k-1]]
                if paid >= 1 and changes >= 2 and prefix_runs.count(owner) >= 2:
                    positions.append((j, owner, paid))
            if not positions:
                continue
            j, owner, paid = positions[int(rng.integers(len(positions)))]
            remaining[owner] = paid
            if sum(remaining) <= j:
                continue
        elif regime == 'exact':
            remaining = needed[:]
        elif regime == 'zero':
            remaining[int(labels[0, 0])] = 0
        colors = list(map(int, rng.permutation(CARRIERS)))[:count]
        board = np.empty((side, side), dtype=np.int8)
        for i, color in enumerate(colors):
            board[labels == i] = color
        for p in path:
            board[p] = ROAD
        board[path[0]], board[path[-1]] = START, END
        for i, (y, x, stamp, capacity) in enumerate(stamps):
            board[y:y+stamp.shape[0], x:x+stamp.shape[1]] = stamp
        try:
            counters = _counters(board, labels, count)
            for i, counter in enumerate(counters):
                for j, p in enumerate(counter['coords']):
                    board[p] = SPENT if j < capacities[i]-remaining[i] else FUEL
            out, facts = _relay(board, labels, count)
            if regime == 'arrival' and not long and not any(
                    c['bounds'][1]-c['bounds'][0] >= 2 and c['bounds'][3]-c['bounds'][2] >= 2 and
                    c['direction'] != 'U' and c['advanced'] > 0 and facts['used'][i] > 0 and
                    c['advanced']+facts['used'][i] < c['capacity'] for i, c in enumerate(facts['counters'])):
                continue
            pieces = [_tight(board, labels == i) for i in range(count)]
            if any(not np.any(np.isin(p, CARRIERS)) for p in pieces):
                continue
            assembled, own = _assemble(pieces, side)
            if not np.array_equal(assembled, board):
                continue
            if regime == 'fuel' and (facts['position'] == facts['length'] or sum(facts['used']) < 3):
                continue
        except AssertionError:
            continue
        return dict(input=_scatter(rng, pieces), output=out)
    raise RuntimeError(f'bounded construction exhausted for {side}/{count}/{regime}/{style}')


def _counts(num_train, num_test):
    for value, name, lower, upper in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lower <= value <= upper:
            raise ValueError(f'{name} must be an integer in {lower}..{upper}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    # Lessons are shuffled, with both stop causes even for a two-example request.
    regimes = ['arrival', 'fuel'] + [str(rng.choice(('arrival', 'fuel', 'exact', 'zero'), p=(.35, .35, .2, .1))) for _ in range(ntrain-2)]
    rng.shuffle(regimes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for i in range(count):
            pieces = 3 if i % 2 == 0 else 4
            side = int(rng.integers(9 if pieces == 3 else 10, 12) if split == 'train' else rng.integers(12, 14))
            regime = regimes[i] if split == 'train' else str(rng.choice(('arrival', 'fuel', 'exact'), p=(.4, .4, .2)))
            style = str(rng.choice(('bands', 'growth')))
            for retry in range(8):
                try:
                    pair = _make_pair(rng, side, pieces, regime, split == 'test', style)
                except RuntimeError:
                    # Retry the same rule on a roomier board, never weaken its
                    # road, ownership, or uniqueness requirements.
                    side = 11 if split == 'train' else 13
                    style = 'growth' if style == 'bands' else 'bands'
                    continue
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not make distinct inputs')
    return puzzle


def _pair_facts(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    pieces, side = _read_pieces(inp)
    board, owners = _assemble(pieces, side)
    predicted, facts = _relay(board, owners, len(pieces))
    _need(np.array_equal(predicted, out), 'wrong relay output')
    _need(out.shape == (side, side), 'wrong output size')
    _need(np.sum(inp != BG) == out.size and not np.any(out == BG), 'lost board area')
    _need(np.sum(out == START) == 1, 'marker count changed')
    changes = int(np.sum(board != out))
    used = sum(facts['used'])
    _need(int(np.sum(board == FUEL)-np.sum(out == FUEL)) == used, 'wrong fuel accounting')
    _need(int(np.sum(out == SPENT)-np.sum(board == SPENT)) == used, 'wrong spent accounting')
    _need(changes == used+(2 if used else 0), 'changed unrelated cells')
    for c in CARRIERS:
        _need(np.sum(inp == c) == np.sum(out == c), 'carrier recolored')
    # Reconstruct each intact input piece after rotation from the assembly.
    for i, piece in enumerate(pieces):
        target = _tight(board, owners == i)
        _need(any(np.array_equal(target, p) for p in _rotations(piece)), 'piece deformed/reflected')
    normalized = np.empty_like(owners)
    identities = {}
    for j, value in enumerate(owners.flat):
        value = int(value)
        normalized.flat[j] = identities.setdefault(value, len(identities))
    facts.update(side=side, count=len(pieces), shape=inp.shape,
                 seams=sum(a != b for a, b in zip(facts['visits'], facts['visits'][1:])),
                 turns=sum((b[0]-a[0], b[1]-a[1]) != (c[0]-b[0], c[1]-b[1])
                           for a, b, c in zip(facts['path'], facts['path'][1:], facts['path'][2:])),
                 arrived=facts['position'] == facts['length'],
                 rotated=any(not np.array_equal(p, _tight(board, owners == i)) for i, p in enumerate(pieces)),
                 structure=sha256(bytes((side, len(pieces)))+normalized.tobytes()).hexdigest())
    return facts


def _reset_position(facts):
    """Counterfactual rule used only to check that demonstrations distinguish it."""
    fuel = facts['remaining'][:]
    visits = facts['visits']
    for j, owner in enumerate(visits[:-1]):
        if j and owner != visits[j-1]:
            fuel[owner] = facts['remaining'][owner]
        if fuel[owner] == 0:
            return j
        fuel[owner] -= 1
    return facts['length']


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    try:
        _counts(len(puzzle['train']), len(puzzle['test']))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair schema')
            signature = _key(_grid(pair['input']))
            _need(signature not in seen, 'duplicate input')
            seen.add(signature)
            records[split].append(_pair_facts(pair))
    train = records['train']
    _need(any(f['arrived'] and sum(f['remaining']) > f['length'] for f in train), 'teach destination-limited stopping')
    _need(any(f['arrived'] and f['visits'][0] != f['visits'][-1] for f in train), 'distinguish departure from receiving payment')
    _need(any(0 < f['position'] < f['length'] and sum(f['remaining']) > f['position'] and
              sum(x > 0 for x in f['used']) >= 2 and _reset_position(f) > f['position']
              for f in train), 'teach local fuel and persistent seam handoff')
    _need(any(f['rotated'] for f in train), 'teach rigid rotations')
    _need(any(c['direction'] != 'U' and c['advanced'] > 0 and f['used'][i] > 0 and c['advanced']+f['used'][i] < c['capacity'] and
              c['bounds'][1]-c['bounds'][0] >= 2 and c['bounds'][3]-c['bounds'][2] >= 2
              for f in train for i, c in enumerate(f['counters'])), 'teach two-dimensional counter update')
    _need(any(len([x for i, x in enumerate(f['visits'][:f['position']+1])
                   if i == 0 or x != f['visits'][i-1]]) > len(set(f['visits'][:f['position']+1]))
              for f in train), 'teach persistent reservoir on reentry')
    _need(all(f['seams'] >= 2 and f['turns'] >= 1 for f in train), 'teach nonstraight multi-piece road')
    _need(min(f['side'] for f in records['test']) > max(f['side'] for f in train), 'missing scale shift')
    return None


def self_test():
    """Stream 42 seeds; retain small metrics/hashes, never a sample corpus."""
    shapes, geometries, lengths, directions, capacities, puzzle_hashes = set(), set(), set(), set(), set(), set()
    counts, stops, reentries = set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _pair_facts(pair)
                digest.update(pair['input'].tobytes())
                shapes.add(f['shape'])
                geometries.add(f['structure'])
                lengths.add(f['length'])
                directions.update(f['directions'])
                capacities.update(f['capacities'])
                counts.add(f['count'])
                stops.add('arrival' if f['arrived'] else 'fuel' if f['position'] else 'zero')
                runs = [x for i, x in enumerate(f['visits']) if i == 0 or x != f['visits'][i-1]]
                reentries += len(runs) > len(set(runs))
        puzzle_hashes.add(digest.hexdigest())
    _need(len(puzzle_hashes) == 42 and len(shapes) >= 25 and len(geometries) >= 100, 'diversity collapsed')
    _need(counts == {3, 4} and directions == {'U', 'D', 'L', 'R'}, 'missing counts/directions')
    _need(len(lengths) >= 6 and len(capacities) >= 5 and reentries >= 10, 'missing structural variation')
    _need({'arrival', 'fuel', 'zero'} <= stops, 'missing stopping regimes')
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    _need(all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p), 'nondeterministic')
    probe = a['test'][0]['input'].copy()
    before = probe.copy()
    _need(np.array_equal(solve(probe), a['test'][0]['output']) and np.array_equal(probe, before),
          'input-only solve disagrees or mutates its input')
    for nt, ne in ((2, 1), (10, 4), (10, 1)):
        validate_generated_puzzle(generate_puzzle(71, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad['test'][0]['output'][0, 0] = BG
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'puzzles': len(puzzle_hashes), 'input_shapes': len(shapes),
            'ownership_structures': len(geometries), 'route_lengths': sorted(lengths),
            'capacities': sorted(capacities), 'reentry_pairs': reentries}
