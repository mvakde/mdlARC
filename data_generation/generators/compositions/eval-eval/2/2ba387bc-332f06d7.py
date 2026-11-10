"""Rank-matched docking on one shared corridor; standard library + NumPy.

No parent generator is imported. All interpretation, matching and movement
use visible pixels. Self-tests stream 42 puzzles and retain only small metrics.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _neighbors(p, shape):
    r, c = p
    for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
        if 0 <= x < shape[0] and 0 <= y < shape[1]:
            yield x, y


def _distance(mask, sources):
    d = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for p in sources:
        p = tuple(map(int, p))
        if mask[p] and d[p] < 0:
            queue.append(p)
            d[p] = 0
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, mask.shape):
            if mask[q] and d[q] < 0:
                d[q] = d[p] + 1
                queue.append(q)
    return d


def _slice(obj, k):
    r, c = obj['position'] if isinstance(obj, dict) else obj
    return slice(r, r+k), slice(c, c+k)


def _valid(mask, k):
    return np.lib.stride_tricks.sliding_window_view(mask, (k, k)).all(axis=(-1, -2))


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    objects = []
    while todo:
        p = min(todo)
        todo.remove(p)
        cells, stack = [p], [p]
        while stack:
            for q in _neighbors(stack.pop(), mask.shape):
                if q in todo:
                    todo.remove(q)
                    cells.append(q)
                    stack.append(q)
        objects.append(cells)
    return objects


def _parse_floor(inp, wall, floor):
    objects = []
    for cells in _components((inp != wall) & (inp != floor)):
        r, c = map(int, np.min(cells, axis=0))
        b, e = map(int, np.max(cells, axis=0) + 1)
        k = b-r
        _require(k >= 3 and k == e-c, "square markers required")
        tile = inp[r:b, c:e]
        color = int(tile[0, 0])
        _require(color not in (wall, floor), "marker color")
        template = np.full((k, k), color, dtype=np.int8)
        if np.array_equal(tile, template):
            kind = 'solid'
        else:
            template[1:-1, 1:-1] = floor
            _require(np.array_equal(tile, template), "malformed frame")
            kind = 'hollow'
        objects.append(dict(position=(r, c), color=color, kind=kind, size=k))
    _require(objects and len({o['size'] for o in objects}) == 1, "common square size")
    solids = sorted((o for o in objects if o['kind'] == 'solid'), key=lambda o: o['position'])
    hollows = sorted((o for o in objects if o['kind'] == 'hollow'), key=lambda o: o['position'])
    _require(len(solids) == len(hollows) and 2 <= len(solids) <= 4, "balanced marker queues")
    for queue in (solids, hollows):
        _require(len({o['position'][0] for o in queue}) == len(queue), "same-class top-row tie")
    k = objects[0]['size']
    base = inp != wall
    d = _distance(base, [tuple(np.argwhere(base)[0])])
    _require(np.all(d[base] >= 0), "disconnected corridor")
    boundary = np.zeros(inp.shape, dtype=bool)
    boundary[0] = boundary[-1] = True
    boundary[:, 0] = boundary[:, -1] = True
    wd = _distance(~base, np.argwhere(boundary))
    _require(np.all(wd[~base] >= 0), "closed corridor loop")
    goal_cells = np.zeros(inp.shape, dtype=bool)
    for o in hollows:
        goal_cells[_slice(o, k)] = True
    to_goals = _distance(base, np.argwhere(goal_cells))
    # All starts are upstream: their common nearest hollow is the first dock.
    goal_distances = [_distance(base, np.argwhere(_footprint(inp.shape, o['position'], k)))
                      for o in hollows]
    nearest = [int(np.argmin([int(d[_slice(o, k)].min()) for d in goal_distances]))
               for o in solids]
    _require(len(set(nearest)) == 1, "cargos must precede the dock region")
    fronts = [int(to_goals[_slice(o, k)].min()) for o in solids]
    _require(len(set(fronts)) == len(fronts), "ambiguous traffic order")
    order = sorted(range(len(solids)), key=lambda i: fronts[i])
    # A visibly separated marker footprint must be wholly walkable initially.
    placements = _valid(base, k)
    _require(all(placements[o['position']] for o in objects), "invalid marker footprint")
    return dict(inp=inp, wall=wall, floor=floor, k=k, solids=solids, hollows=hollows,
                base=base, order=order, goal_distances=goal_distances)


def _footprint(shape, position, k):
    a = np.zeros(shape, dtype=bool)
    a[_slice(position, k)] = True
    return a


def _analyze(input_grid):
    inp = _grid(input_grid)
    wall = int(inp[0, 0])
    _require(all(np.all(edge == wall) for edge in (inp[0], inp[-1], inp[:, 0], inp[:, -1])),
             "uniform outer wall border required")
    candidates = []
    for floor in map(int, np.unique(inp)):
        if floor == wall:
            continue
        try:
            candidates.append(_parse_floor(inp, wall, floor))
        except AssertionError:
            continue
    _require(len(candidates) == 1, "floor/marker interpretation must be unique")
    return candidates[0]


def _destination(reach, target, cell_distance, k):
    if reach[target] >= 0:
        return target
    reachable = reach >= 0
    windows = np.lib.stride_tricks.sliding_window_view(cell_distance, (k, k))
    destinations = []
    for score in (windows.min(axis=(-1, -2)), windows.max(axis=(-1, -2)),
                  windows.sum(axis=(-1, -2))):
        best = np.argwhere(reachable & (score == score[reachable].min()))
        _require(len(best) == 1, "ambiguous stopping alignment")
        destinations.append(tuple(map(int, best[0])))
    _require(len(set(destinations)) == 1, "ambiguous forward front")
    return destinations[0]


def _simulate(a, pairing='rank', traffic=True):
    """Move front to back, equivalent to sliding a no-passing cargo train.

    Base geodesic distances ignore cargos, so a stationary cargo cannot make
    its own blocked destination appear closer by cutting the distance field.
    Reject ambiguous lateral stops instead of inventing a wall-hugging rule.
    """
    inp, k, base = a['inp'], a['k'], a['base']
    solids, hollows = a['solids'], a['hollows']
    positions = [o['position'] for o in solids]
    results = []
    for i in a['order']:
        if pairing == 'rank':
            j = i
        elif pairing == 'reverse':
            j = len(hollows)-1-i
        elif pairing == 'nearest':
            j = int(np.argmin([int(d[_slice(solids[i], k)].min()) for d in a['goal_distances']]))
        else:
            raise ValueError('unknown pairing')
        target = hollows[j]['position']
        allowed = base.copy()
        if traffic:
            for other, p in enumerate(positions):
                if other != i:
                    allowed[_slice(p, k)] = False
        valid = _valid(allowed, k)
        reach = _distance(valid, [positions[i]])
        success = bool(reach[target] >= 0)
        destination = _destination(reach, target, a['goal_distances'][j], k)
        # A dock remains recognizable unless a final cargo covers it exactly.
        footprint = _footprint(inp.shape, destination, k)
        for h in hollows:
            overlap = footprint[_slice(h, k)].sum()
            _require(overlap in (0, k*k), "partial dock coverage")
        unoccupied = _valid(base, k)
        alone = _distance(unoccupied, [solids[i]['position']])
        alone_destination = _destination(alone, target, a['goal_distances'][j], k)
        # An unreachable bare goal does not prove that THIS stop is at its
        # throat: another cargo may block the approach even earlier.
        traffic_stop = bool(not success and destination != alone_destination)
        results.append(dict(rank=i, target=j, start=solids[i]['position'], destination=destination,
                            success=success, travel=int(reach[destination]),
                            traffic_stop=traffic_stop,
                            width_stop=bool(not success and not traffic_stop)))
        positions[i] = destination
    out = inp.copy()
    for o in solids:
        out[_slice(o, k)] = a['floor']
    for o, p in zip(solids, positions):
        out[_slice(p, k)] = o['color']
    return out, results


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _simulate(_analyze(input_grid))[0]


def _walk(rng, rows, cols, minimum):
    best = []
    for _ in range(24):
        start = (int(rng.integers(rows)), int(rng.integers(cols)))
        path, used = [start], {start}
        target = int(rng.integers(minimum, rows*cols+1))
        while len(path) < target:
            options = [p for p in _neighbors(path[-1], (rows, cols)) if p not in used]
            if not options:
                break
            nxt = options[int(rng.integers(len(options)))]
            path.append(nxt)
            used.add(nxt)
        if len(path) > len(best):
            best = path
        if len(path) >= minimum:
            return path
    _require(len(best) >= minimum, "short walk")
    return best


def _layout(rng, n, k, blocked, wide):
    q = k + int(wide)
    maximum = min(5, 29//(q+1))
    choices = [(r, c) for r in range(2, maximum+1) for c in range(2, maximum+1)
               if r*c >= 2*n+int(blocked) and r*(q-k+1) >= n]
    nr, nc = choices[int(rng.integers(len(choices)))]
    def coordinates(count):
        gaps = np.full(count-1, q+1, dtype=int)
        budget = 28-q-int(gaps.sum())
        for i in rng.permutation(count-1):
            extra = int(rng.integers(0, min(3, budget)+1))
            gaps[i] += extra
            budget -= extra
        return np.concatenate(([1], 1+np.cumsum(gaps)))
    rows, cols = coordinates(nr), coordinates(nc)
    nodes = _walk(rng, nr, nc, 2*n+int(blocked))
    path = [(int(rows[r]), int(cols[c])) for r, c in nodes]
    shape = (int(rows[-1])+q+1, int(cols[-1])+q+1)
    mask = np.zeros(shape, dtype=bool)
    for p in path:
        mask[_slice(p, q)] = True
    def capacity(points):
        return sum(min(sum(p[0] == r for p in points), q-k+1) for r in {p[0] for p in points})
    separation = int(blocked)
    splits = [i for i in range(n, len(path)-n+1-separation)
              if capacity(path[:i]) >= n and capacity(path[i+separation:]) >= n]
    _require(splits, 'insufficient distinct marker top rows')
    split = int(rng.choice(splits))
    markers, site_lists = [], []
    for kind, candidates in (('solid', range(split)), ('hollow', range(split+separation, len(path)))):
        used_tops, selected = set(), []
        for index in rng.permutation(list(candidates)):
            r, c = path[int(index)]
            tops = [t for t in range(r, r+q-k+1) if t not in used_tops]
            if tops:
                top = int(rng.choice(tops))
                used_tops.add(top)
                markers.append((kind, (top, c+int(rng.integers(q-k+1)))))
                selected.append(int(index))
            if len(selected) == n:
                break
        _require(len(selected) == n, 'marker placement capacity')
        site_lists.append(sorted(selected))
    solid_sites, hollow_sites = site_lists
    # Any blocked edge is after the last initial cargo. Every cargo is
    # therefore on the same approaching side of the impassable width.
    cut = int(rng.integers(solid_sites[-1]+1, hollow_sites[0])) if blocked else -1
    for i, (p, t) in enumerate(zip(path, path[1:])):
        width = int(rng.integers(1, k)) if i == cut else (
            k if wide and i == solid_sites[-1] else int(rng.integers(k, q+1)))
        offset = int(rng.integers(q-width+1))
        if p[0] == t[0]:
            lo, hi = sorted((p[1], t[1]))
            mask[p[0]+offset:p[0]+offset+width, lo+q:hi] = True
        else:
            lo, hi = sorted((p[0], t[0]))
            mask[lo+q:hi, p[1]+offset:p[1]+offset+width] = True
    return mask, markers


def _fallback_layout(rng, n, k):
    """A finite blocked U-route, used after 64 rejected free-walk trials.

    Unequal marker gaps, lane spacing, throat width/offset and reflection
    still vary. The throat is beyond the bend and before the first dock.
    """
    rows = [1]
    for _ in range(n-1):
        rows.append(rows[-1]+k+int(rng.integers(1, 3)))
    bottom = rows[-1]+k+2
    right = k+1+int(rng.integers(2, 6))
    mask = np.zeros((bottom+k+1, right+k+1), dtype=bool)
    mask[1:bottom+k, 1:1+k] = True
    mask[1:bottom+k, right:right+k] = True
    mask[bottom:bottom+k, 1:right+k] = True
    width = int(rng.integers(1, k))
    offset = int(rng.integers(k-width+1))
    mask[bottom-1, 1:1+k] = False
    mask[bottom-1, 1+offset:1+offset+width] = True
    markers = [('solid', (r, right)) for r in rows] + [('hollow', (r, 1)) for r in rows]
    return mask, markers


def _counterexample(a, mode):
    try:
        return not np.array_equal(_simulate(a)[0], _simulate(a, pairing=mode)[0])
    except AssertionError:
        # An ambiguous alternative cannot count as a measured counterexample.
        return False


def _make_pair(rng, n, k, blocked, witness, seen):
    failures = {}
    for attempt in range(384):
        try:
            # Exact-width rooms keep blockage fronts unique; wider successful
            # rooms expose passable narrowings and sideways placement freedom.
            wide = bool(witness or (not blocked and rng.integers(2)))
            mask, markers = (_fallback_layout(rng, n, k) if blocked and attempt >= 64
                             else _layout(rng, n, k, blocked, wide))
            palette = list(map(int, rng.permutation(10)))
            wall, floor = palette[:2]
            inp = np.full(mask.shape, wall, dtype=np.int8)
            inp[mask] = floor
            colors = list(map(int, rng.choice(palette[2:], 2*n, replace=True)))
            # Repeated color across classes removes color-as-type shortcuts.
            colors[n] = colors[0]
            for (kind, p), color in zip(markers, colors):
                inp[_slice(p, k)] = color
                if kind == 'hollow':
                    r, c = p
                    inp[r+1:r+k-1, c+1:c+k-1] = floor
            inp = np.rot90(inp, 2*int(rng.integers(2)))
            if rng.integers(2):
                inp = np.fliplr(inp)
            inp = inp.copy()
            a = _analyze(inp)
            out, info = _simulate(a)
            _require(any(v['travel'] > 0 for v in info), "must move")
            if blocked:
                _require(any(v['width_stop'] and v['travel'] > 0 for v in info), "width witness")
            if witness:
                _require(any(v['success'] for v in info), "need docking success")
                _require(any(v['traffic_stop'] and v['travel'] > 0 for v in info), "need moving traffic stop")
                _require(_counterexample(a, 'nearest') and _counterexample(a, 'reverse'), "rank witness")
                _require(not np.array_equal(out, _simulate(a, traffic=False)[0]), "shared traffic witness")
                _require(len({o['color'] for o in a['solids']}) >= 2, "observable cargo order")
                _require(any(v['success'] and a['solids'][v['rank']]['color'] != a['hollows'][v['target']]['color']
                             for v in info), "demonstrate routing across unequal colors")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            return dict(input=inp, output=out)
        except AssertionError as error:
            reason = str(error)
            failures[reason] = failures.get(reason, 0)+1
            continue
    raise RuntimeError(f'bounded construction exhausted: n={n},k={k},blocked={blocked},witness={witness}; {failures}')


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    seen = set()
    puzzle = dict(train=[], test=[])
    specs = [(3, 3, False, True), (int(rng.integers(2, 4)), 3, True, False)]
    specs += [(int(rng.integers(2, 4)), int(rng.integers(3, 5)), bool(rng.integers(2)), False)
              for _ in range(num_train-2)]
    for n, k, blocked, witness in specs:
        puzzle['train'].append(_make_pair(rng, n, k, blocked, witness, seen))
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    for _ in range(num_test):
        puzzle['test'].append(_make_pair(rng, 4, int(rng.integers(3, 5)), bool(rng.integers(2)), False, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "split lists")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen = set()
    training = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyze(inp)
            expected, info = _simulate(a)
            _require(np.array_equal(out, expected), "incorrect movement or scenery")
            _require(not np.array_equal(inp, out), "identity pair")
            # Preserve cargo identity, square area, separation and wall cells.
            occupied = np.zeros(inp.shape, dtype=bool)
            for v in info:
                sl = _slice(v['destination'], a['k'])
                _require(not occupied[sl].any(), "cargo overlap")
                occupied[sl] = True
                _require(np.all(out[sl] == a['solids'][v['rank']]['color']), "cargo footprint")
            _require(np.array_equal(out[inp == a['wall']], inp[inp == a['wall']]), "wall changed")
            if split == 'train':
                training.append((a, info))
    _require(any(any(v['success'] and v['travel'] > 0 for v in info) for _, info in training), "teach docking")
    _require(any(any(v['width_stop'] and v['travel'] > 0 for v in info) for _, info in training), "teach footprint bottleneck")
    _require(any(any(v['traffic_stop'] and v['travel'] > 0 for v in info) and
                 _counterexample(a, 'nearest') and _counterexample(a, 'reverse') and
                 any(v['success'] and a['solids'][v['rank']]['color'] != a['hollows'][v['target']]['color'] for v in info) and
                 not np.array_equal(_simulate(a)[0], _simulate(a, traffic=False)[0])
                 for a, info in training), "teach rank matching and shared occupancy")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    # Coordinate fixtures have explicit destinations, rather than labels
    # obtained by calling the generation/placement implementation.
    for blocked in (False, True):
        height, bottom = (19, 15) if blocked else (17, 13)
        inp = np.zeros((height, 13), dtype=np.int8)
        inp[1:height-1, 1:4] = inp[1:height-1, 9:12] = 1
        inp[bottom:bottom+3, 1:12] = 1
        if blocked:
            inp[14, 9] = 0  # A two-cell throat blocks a three-cell cargo.
        for i, r in enumerate((1, 5, 9)):
            inp[r:r+3, 9:12] = i+2
            inp[r:r+3, 1:4] = i+5
            inp[r+1, 2] = 1
        expected = inp.copy()
        for r in (1, 5, 9):
            expected[r:r+3, 9:12] = 1
        destinations = ((5, 9), (8, 9), (11, 9)) if blocked else ((13, 4), (12, 1), (9, 1))
        for i, p in enumerate(destinations):
            expected[_slice(p, 3)] = i+2
        before = inp.copy()
        _require(np.array_equal(solve(inp), expected), "explicit U-corridor fixture")
        _require(np.array_equal(inp, before), "solve mutated its input")
        _require(np.array_equal(solve(9-inp), 9-expected), "geometric color roles")
        if not blocked:
            # A throat beyond the parked front cargo is NOT the cause of
            # either following cargo's earlier traffic stop.
            hidden_throat = inp.copy()
            hidden_throat[4, 1] = 0
            prediction, info = _simulate(_analyze(hidden_throat))
            wanted = expected.copy()
            wanted[4, 1] = 0
            _require(np.array_equal(prediction, wanted), "traffic before a later throat")
            _require(sum(v['traffic_stop'] for v in info) == 2 and not any(v['width_stop'] for v in info),
                     "do not misclassify an earlier traffic stop as width")
    signatures, layouts, shapes, sizes, palettes, patterns = set(), set(), set(), set(), set(), set()
    outcomes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze(pair['input'])
                expected, info = _simulate(a)
                _require(np.array_equal(expected, pair['output']), "self-test solve")
                sizes.add(a['k'])
                shapes.add(a['inp'].shape)
                palettes.add((a['wall'], a['floor']))
                layouts.add((a['inp'].shape, a['k'], tuple((o['kind'], o['position']) for o in a['solids']+a['hollows']), a['base'].tobytes()))
                patterns.add(tuple((v['success'], v['traffic_stop'], v['width_stop']) for v in info))
                if split == 'test':
                    outcomes.update((v['success'], v['traffic_stop'], v['width_stop']) for v in info)
        if seed == 0:
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "determinism")
            for kind in ('wall', 'cargo', 'dock'):
                damaged = {s: [{r: v.copy() for r, v in p.items()} for p in puzzle[s]] for s in puzzle}
                pair = damaged['test'][0]
                a = _analyze(pair['input'])
                point = ((0, 0) if kind == 'wall' else a['solids'][0]['position'] if kind == 'cargo'
                         else a['hollows'][0]['position'])
                pair['output'][point] = (int(pair['output'][point])+1) % 10
                try:
                    validate_generated_puzzle(damaged)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('accepted corrupted '+kind)
    _require(len(signatures) == 42 and len(layouts) >= 180, "42-seed structural variation")
    _require(len(shapes) >= 30 and sizes == {3, 4}, "shape and scale variation")
    _require(len(palettes) >= 35 and len(patterns) >= 12, "palette and interaction variation")
    _require(any(v[0] for v in outcomes) and any(v[1] for v in outcomes) and any(v[2] for v in outcomes), "test outcomes")
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(901+n*10+m, n, m)
            _require(len(p['train']) == n and len(p['test']) == m, "explicit counts")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
