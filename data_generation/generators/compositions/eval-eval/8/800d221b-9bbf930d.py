"""Consensus rails: majority identities select and transport ballots to boxes."""
from __future__ import annotations

import heapq
import itertools
from collections import deque
import numpy as np

_DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), 'grid dimensions')
    _need(np.issubdtype(g.dtype, np.integer), 'integer cells required')
    _need(bool(np.all((0 <= g) & (g <= 9))), 'ARC colors required')
    return g


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _components(mask):
    labels = np.full(mask.shape, -1, dtype=np.int16)
    parts = []
    for rr, cc in zip(*np.nonzero(mask)):
        p = int(rr), int(cc)
        if labels[p] >= 0:
            continue
        index = len(parts)
        labels[p] = index
        stack, cells = [p], []
        while stack:
            q = stack.pop()
            cells.append(q)
            for d in _DIRS:
                v = _add(q, d)
                if _inside(mask, v) and mask[v] and labels[v] < 0:
                    labels[v] = index
                    stack.append(v)
        parts.append(cells)
    return parts, labels


def _normalise(inp):
    choices = []
    for k in range(4):
        g = np.rot90(inp, k)
        token = int(g[0, 0])
        if np.all(g[:, 0] == token) and np.count_nonzero(g == token) == g.shape[0]:
            if g.shape[1] > 1 and np.all(g[:, 1] == g[0, 1]) and g[0, 1] != token:
                choices.append((g, k, token, int(g[0, 1])))
    _need(len(choices) == 1, 'unique token strip and clear approach required')
    return choices[0]


def _roles(inp):
    g, k, token, bg = _normalise(inp)
    values = set(map(int, np.unique(g)))
    _need(len(values) == 5, 'five visible roles required')
    choices = []
    for marker in sorted(values - {token, bg}):
        boxes, box_ids = _components(g == marker)
        rectangles = []
        for cells in boxes:
            rs, cs = zip(*cells)
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            bh, bw = r1-r0+1, c1-c0+1
            if (bh < 3 or bw < 3 or bh % 2 != 1 or bw % 2 != 1
                    or len(cells) != bh*bw):
                break
            rectangles.append((r0, r1, c0, c1))
        else:
            if 1 <= len(rectangles) <= 2:
                signals = tuple(sorted(values - {token, bg, marker}))
                rails, rail_ids = _components(np.isin(g, signals))
                majorities = []
                for cells in rails:
                    n0 = sum(int(g[p]) == signals[0] for p in cells)
                    if n0 == 0 or n0 == len(cells) or 2*n0 == len(cells):
                        break
                    majorities.append(signals[int(2*n0 < len(cells))])
                else:
                    if rails:
                        choices.append(dict(grid=g, rotation=k, token=token, background=bg,
                                            marker=marker, signals=signals, boxes=rectangles,
                                            box_ids=box_ids, rails=rails, rail_ids=rail_ids,
                                            majorities=majorities))
    _need(len(choices) == 1, 'one geometric role interpretation required')
    return choices[0]


def _flanks(g, ids, p, direction):
    d = (-direction[1], direction[0])
    a, b = _add(p, d), _add(p, (-d[0], -d[1]))
    if _inside(g, a) and _inside(g, b) and ids[a] >= 0 and ids[b] >= 0:
        return int(ids[a]), int(ids[b])
    return None


def _entrances(a):
    g, ids, bg = a['grid'], a['rail_ids'], a['background']
    entrances = []
    for row in range(1, len(g)-1):
        for col in range(1, g.shape[1]):
            if int(g[row, col]) != bg:
                break
            f = _flanks(g, ids, (row, col), (0, 1))
            if f is not None:
                entrances.append((row, col, f))
                break
    return entrances


def _walk(a, row, color=None):
    """Direction and local rail evidence, rather than free-space flood fill."""
    g, ids, bg = a['grid'], a['rail_ids'], a['background']
    p, d = (row, 1), (0, 1)
    path, turns, seen, supports = [p], [], set(), set()
    for _ in range(4*g.size):
        _need((p, d) not in seen, 'cyclic route')
        seen.add((p, d))
        f = _flanks(g, ids, p, d)
        if f is not None:
            supports.update(f)
        q = _add(p, d)
        if not _inside(g, q):
            return dict(start=(row, 0), end=p, box=-1, path=path, turns=turns, supports=supports)
        if int(g[q]) == bg:
            p = q
            path.append(p)
            continue
        if a['box_ids'][q] >= 0:
            return dict(start=(row, 0), end=p, box=int(a['box_ids'][q]), path=path,
                        turns=turns, supports=supports)
        choices = []
        for dd in ((-d[1], d[0]), (d[1], -d[0])):
            qq = _add(p, dd)
            if not _inside(g, qq) or int(g[qq]) != bg:
                continue
            ff = _flanks(g, ids, qq, dd)
            if ff is not None and (color is None or all(a['majorities'][i] == color for i in ff)):
                choices.append(dd)
        _need(len(choices) <= 1, 'ambiguous rail turn')
        if not choices:
            return dict(start=(row, 0), end=p, box=-1, path=path, turns=turns, supports=supports)
        turns.append(p)
        d = choices[0]
    raise ValueError('route exceeded bounded grid')


def _analyse(inp, *, pixel_match=False):
    a = _roles(inp)
    routes, inactive = [], []
    g = a['grid']
    for row, col, f in _entrances(a):
        labels = [a['majorities'][i] for i in f]
        matched = labels[0] == labels[1]
        if pixel_match:
            matched = int(g[row-1, col]) == int(g[row+1, col])
        if not matched:
            inactive.append((row, col, f))
            continue
        route = _walk(a, row, labels[0])
        _need(route['box'] >= 0, 'active token must reach a ballot box')
        _need(len(route['path']) == len(set(route['path'])), 'route repeats cells')
        _need(all(a['majorities'][i] == labels[0] for i in route['supports']),
              'active route must retain one consensus identity')
        route.update(color=labels[0], entrance=(row, col), flanks=f)
        routes.append(route)
    _need(routes and inactive, 'both agreed and disagreed openings required')
    used = [p for r in routes for p in r['path']]
    _need(len(used) == len(set(used)), 'moving routes must be disjoint')
    votes, winners = [], []
    for i in range(len(a['boxes'])):
        colors = [r['color'] for r in routes if r['box'] == i]
        n0 = colors.count(a['signals'][0])
        _need(len(colors) >= 3 and 0 < n0 < len(colors) and 2*n0 != len(colors),
              'each box needs a mixed strict arrival majority')
        votes.append(colors)
        winners.append(a['signals'][int(2*n0 < len(colors))])
    a.update(routes=routes, inactive=inactive, votes=votes, winners=winners)
    return a


def _answer(a):
    out = a['grid'].copy()
    for route in a['routes']:
        out[route['start']] = a['background']
        out[route['end']] = route['color']
    for (r0, r1, c0, c1), color in zip(a['boxes'], a['winners']):
        out[(r0+r1)//2, (c0+c1)//2] = color
    return np.rot90(out, -a['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _answer(_analyse(_grid(input_grid)))


def _near(cells, diagonal=False):
    steps = tuple(itertools.product((-1, 0, 1), repeat=2)) if diagonal else _DIRS
    return {_add(p, d) for p in cells for d in steps}


def _rect(r, c, h, w):
    return {(rr, cc) for rr in range(r, r+h) for cc in range(c, c+w)}


def _find_path(rng, shape, port, outward, forbidden, roughness):
    h, w = shape
    apron = [port, _add(port, outward), _add(_add(port, outward), outward)]
    if any(not (1 <= r < h-1 and 2 <= c < w-1) or (r, c) in forbidden for r, c in apron):
        return None
    start = apron[-1]
    direction = _DIRS.index(outward)
    state = (*start, direction, 2)
    costs = 1 + rng.random(shape)*roughness
    heap, distances, previous = [(start[1]-2, 0., state)], {state: 0.}, {}
    goal = None
    for _ in range(20000):
        if not heap:
            break
        _, cost, state = heapq.heappop(heap)
        if cost != distances.get(state):
            continue
        r, c, d, run = state
        if c == 2 and d == 2:
            goal = state
            break
        for nd in (d, (d+1)%4, (d-1)%4):
            if nd != d and run < 2:
                continue
            dr, dc = _DIRS[nd]
            q = r+dr, c+dc
            if not (1 <= q[0] < h-1 and 2 <= q[1] < w-1) or q in forbidden:
                continue
            if q in apron[:-1]:
                continue
            ns = (*q, nd, min(2, run+1) if nd == d else 1)
            new = cost + costs[q] + (2.4 if nd != d else 0.)
            if new < distances.get(ns, float('inf')):
                distances[ns], previous[ns] = new, state
                heapq.heappush(heap, (new + q[1]-2, new, ns))
    if goal is None:
        return None
    chain = []
    while goal != (*start, direction, 2):
        chain.append(goal[:2])
        goal = previous[goal]
    path = apron + list(reversed(chain))
    path.append((path[-1][0], 1))
    if len(path) != len(set(path)):
        return None
    cells = set(path)
    # A simple channel cannot touch itself except along consecutive cells.
    if any(sum(_add(p, d) in cells for d in _DIRS) != (1 if i in (0, len(path)-1) else 2)
           for i, p in enumerate(path)):
        return None
    return list(reversed(path))


def _link_rails(g, path_cells, box_specs, rng):
    """Join districts across two boxes without crossing any channel."""
    parts, ids = _components(g == 1)
    adjacent = []
    for cells in parts:
        touching = set()
        for i, box in enumerate(box_specs):
            if set(cells) & _near(_rect(*box), True):
                touching.add(i)
        adjacent.append(touching)
    if any(len(v) == 2 for v in adjacent):
        return True
    pairs = [(i, j) for i in range(len(parts)) for j in range(i+1, len(parts))
             if adjacent[i] and adjacent[j] and adjacent[i] != adjacent[j]]
    rng.shuffle(pairs)
    pairs.sort(key=lambda ij: min(abs(p[0]-q[0])+abs(p[1]-q[1])
                                 for p in parts[ij[0]] for q in parts[ij[1]]))
    for i, j in pairs[:8]:
        # Other rails cannot become accidental bridges; a connector uses blank
        # space between the two selected ribbons and never crosses a centerline.
        blocked = path_cells | (_near(path_cells, True) - set(parts[i]) - set(parts[j]))
        todo = deque(sorted(parts[i]))
        previous = {p: None for p in parts[i]}
        goal = None
        while todo:
            p = todo.popleft()
            if ids[p] == j:
                goal = p
                break
            for d in _DIRS:
                q = _add(p, d)
                if (not _inside(g, q) or q[1] < 3 or q in blocked or q in previous
                        or g[q] == 8 or (g[q] == 1 and ids[q] not in (i, j))):
                    continue
                previous[q] = p
                todo.append(q)
        if goal is not None:
            connector = []
            while previous[goal] is not None:
                connector.append(goal)
                goal = previous[goal]
            if len(connector) <= 12:
                for p in connector:
                    g[p] = 1
                return True
    return False


def _geometry(rng, rich=False, double=False, large=False, compact=False, extra=False):
    upper = 27 if compact else 31
    h = int(rng.integers(27 if large else (23 if double else 17), upper))
    w = int(rng.integers(27 if large else (21 if rich or double else 18), upper))
    box_specs = []
    if double:
        for band in (0, 1):
            bh, bw = int(rng.choice([5, 7], p=[.15, .85])), int(rng.choice([3, 5]))
            low, high = (2, h//2-bh) if band == 0 else (h//2+1, h-bh-1)
            if low > high:
                return None
            box_specs.append((int(rng.integers(low, high+1)), int(rng.integers(w//2, w-bw-1)), bh, bw))
    else:
        bh, bw = int(rng.choice([3, 5, 7])), int(rng.choice([3, 5, 7]))
        box_specs.append((int(rng.integers(3, h-bh-2)), int(rng.integers(w//2, w-bw-1)), bh, bw))
    boxes = set().union(*(_rect(*b) for b in box_specs))
    port_groups = []
    for r, c, bh, bw in box_specs:
        candidates = [((r+j, c-1), (0, -1)) for j in range(0, bh, 2)]
        candidates += [((r-1, c+j), (-1, 0)) for j in range(0, bw, 2)]
        candidates += [((r+bh, c+j), (1, 0)) for j in range(0, bw, 2)]
        if rich:
            candidates += [((r+j, c+bw), (0, 1)) for j in range(0, bh, 2)]
        rng.shuffle(candidates)
        if double and not extra and rng.random() < .8:
            candidates.sort(key=lambda pd: pd[1] != (0, -1))
        target = int(rng.integers(5 if extra else 4, min(6 if double else (8 if rich else 6), len(candidates))+1))
        picked = []
        for p, d in candidates:
            apron = {p, _add(p, d), _add(_add(p, d), d)}
            other = {q for pp, dd in picked for q in (pp, _add(pp, dd), _add(_add(pp, dd), dd))}
            if apron & _near(other, True) or apron & boxes:
                continue
            picked.append((p, d))
            if len(picked) == target:
                break
        if len(picked) < 4:
            return None
        port_groups.append(picked)
    ports = [p for group in port_groups for p in group]
    aprons = [{p, _add(p, d), _add(_add(p, d), d)} for p, d in ports]
    # Right-facing and long-around routes are placed before direct left entries.
    order = sorted(range(len(ports)), key=lambda i: (-ports[i][0][1], float(rng.random())))
    paths = []
    for i in order:
        port, d = ports[i]
        forbidden = boxes | _near(boxes, True)
        forbidden |= _near({q for path in paths for q in path}, True)
        forbidden |= _near(set().union(*(v for j, v in enumerate(aprons) if j != i)), True)
        forbidden -= aprons[i]
        path = _find_path(rng, (h, w), port, d, forbidden, 1.5 if rich else .5)
        if path is None:
            return None
        paths.append(path)
    path_cells = {p for path in paths for p in path}
    walls = _near(path_cells, True) - path_cells - boxes
    margin = int(rng.integers(2, 6))
    walls = {p for p in walls if 0 <= p[0] < h and margin <= p[1] < w}
    g = np.zeros((h, w), dtype=np.int8)
    g[:, 0] = 9
    for p in boxes:
        g[p] = 8
    for p in walls:
        g[p] = 1
    if double and not _link_rails(g, path_cells, box_specs, rng):
        return None
    components, ids = _components(g == 1)
    if not 4 <= len(components) <= 24 or any(len(x) < 3 for x in components):
        return None
    # Geometry is read without knowing any labels. Invalid incidental openings
    # may exist, but must be mismatched after identities are assigned.
    _, box_ids = _components(g == 8)
    a = dict(grid=g, rail_ids=ids, box_ids=box_ids, background=0)
    channels = []
    for row, col, f in _entrances(a):
        try:
            route = _walk(a, row)
        except ValueError:
            route = None
        valid = route is not None and route['box'] >= 0 and len(route['path']) == len(set(route['path']))
        channels.append((row, col, f, route, valid))
    intended = {path[0][0]: path for path in paths}
    actual = {r: route['path'] for r, c, f, route, valid in channels if valid}
    if any(actual.get(row) != path for row, path in intended.items()):
        return None
    return g, components, channels, box_specs


def _coupled(a):
    owners = {}
    for route in a['routes']:
        for rid in route['flanks']:
            owners.setdefault(rid, set()).add(route['box'])
    return any(len(boxes) > 1 for boxes in owners.values())


def _paint(rng, geometry, witness=False, opposite=False, min_active=3, max_active=20,
           require_bend=False):
    g, components, channels, box_specs = geometry
    for _ in range(256):
        labels = list(map(int, rng.integers(1, 3, size=len(components))))
        active, inactive = [], []
        for row, col, f, route, valid in channels:
            if labels[f[0]] == labels[f[1]]:
                if not valid or any(labels[i] != labels[f[0]] for i in route['supports']):
                    break
                active.append((row, col, f, route, labels[f[0]]))
            else:
                inactive.append((row, col, f))
        else:
            if not inactive:
                continue
            valid_inactive = {row for row, col, f, route, valid in channels
                              if valid and labels[f[0]] != labels[f[1]]}
            if not valid_inactive:
                continue
            if not min_active <= len(active) <= max_active:
                continue
            votes = [[color for r, c, f, route, color in active if route['box'] == i]
                     for i in range(len(box_specs))]
            if not all(len(v) >= 3 and 0 < v.count(1) < len(v) and 2*v.count(1) != len(v) for v in votes):
                continue
            if opposite and len({int(v.count(1) > len(v)/2) for v in votes}) != 2:
                continue
            if require_bend and not any(route['turns'] for r, c, f, route, color in active):
                continue
            cells = [p for r, c, f, route, color in active for p in route['path']]
            if len(cells) != len(set(cells)):
                continue
            inp = g.copy()
            for cells, majority in zip(components, labels):
                # Close, moderate, and strong local votes; no ban on solid patches.
                n = len(cells)
                minority = int(rng.integers(1, (n-1)//2+1))
                style = int(rng.integers(3))
                if style == 0:
                    order = list(map(int, rng.permutation(n)))
                elif style == 1:
                    origin = cells[int(rng.integers(n))]
                    order = sorted(range(n), key=lambda i: (abs(cells[i][0]-origin[0])+abs(cells[i][1]-origin[1]), cells[i]))
                else:
                    order = sorted(range(n), key=lambda i: ((cells[i][0]+cells[i][1])%2, cells[i]))
                for i, p in enumerate(cells):
                    inp[p] = majority
                for i in order[:minority]:
                    inp[cells[i]] = 3-majority
            # Discriminating entrance pixels: the majority operation cannot be
            # replaced by equality of the two neighboring literal pixels.
            # Accept patterns that visibly distinguish these two notions of
            # equality, while leaving the sampled local counts untouched.
            for row, col, f, route, color in active:
                a, b = (row-1, col), (row+1, col)
                if inp[a] != inp[b]:
                    break
            else:
                continue
            if not any(row in valid_inactive and inp[row-1, col] == inp[row+1, col]
                       for row, col, f in inactive):
                continue
            try:
                info = _analyse(inp)
            except ValueError:
                continue
            if witness and not _area_disagrees(info):
                continue
            if len(box_specs) == 2 and not _coupled(info):
                continue
            out = inp.copy()
            for row, col, f, route, color in active:
                out[row, 0] = 0
                out[route['end']] = color
            # Labels are constructed from records; solve reads the final input.
            for (r, c, bh, bw), v in zip(box_specs, votes):
                out[r+bh//2, c+bw//2] = 1 if v.count(1) > len(v)/2 else 2
            if not np.array_equal(out, _answer(info)):
                continue
            return inp, out
    return None


def _area_disagrees(a):
    g = a['grid']
    n0 = int(np.count_nonzero(g == a['signals'][0]))
    n1 = int(np.count_nonzero(g == a['signals'][1]))
    raw = a['signals'][int(n1 > n0)] if n0 != n1 else None
    return raw is not None and any(w != raw for w in a['winners'])


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError('illegal example count')
    _need(num_train > num_test, 'num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _need(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
          'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    palette = list(map(int, rng.permutation(10)))
    rotation = int(rng.integers(4))
    shift = int(rng.integers(2))  # larger canvases or more transported ballots
    mixing = bool(rng.integers(2))
    turning = bool(rng.integers(2))
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            is_test = split == 'test'
            double = (i == 1 or (i > 1 and rng.random() < .35)) if not is_test else (shift == 1 or rng.random() < .65)
            for attempt in range(1600):
                geometry = _geometry(rng, rich=is_test or i % 2 == 1, double=double,
                                     large=is_test and shift == 0, compact=not is_test and shift == 0,
                                     extra=is_test and shift == 1)
                if geometry is None:
                    continue
                pair = _paint(rng, geometry, witness=not is_test and i == 0,
                              opposite=not is_test and i == 1,
                              min_active=7 if is_test and shift == 1 else 3,
                              max_active=6 if not is_test else 20,
                              require_bend=not is_test and i == 0)
                if pair is None:
                    continue
                inp, out = pair
                k = int(rng.integers(4)) if turning else rotation
                roles = list(map(int, rng.permutation(10))) if mixing else palette
                inp, out = np.rot90(np.array(roles, dtype=np.int8)[inp], k).copy(), np.rot90(np.array(roles, dtype=np.int8)[out], k).copy()
                key = inp.shape, inp.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(dict(input=inp, output=out))
                break
            else:
                raise RuntimeError('bounded railway construction exhausted')
    rng.shuffle(puzzle['train'])
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, measurements = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'shape preservation')
            key = inp.shape, inp.astype(np.int8).tobytes()
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            a = _analyse(inp)
            _need(np.array_equal(out, _answer(a)), 'incorrect output')
            normal = np.rot90(out, a['rotation'])
            fixed = np.isin(a['grid'], (*a['signals'], a['marker']))
            for r0, r1, c0, c1 in a['boxes']:
                fixed[(r0+r1)//2, (c0+c1)//2] = False
            _need(np.array_equal(normal[fixed], a['grid'][fixed]), 'rails and box borders preserved')
            _need(np.count_nonzero(inp != out) == 2*len(a['routes']) + len(a['boxes']), 'terminal-only changes')
            _need(np.count_nonzero(inp == a['token']) - np.count_nonzero(out == a['token']) == len(a['routes']), 'token consumption')
            m = _measure(a)
            _need(m['false_pixel_mismatch'] and m['false_pixel_match'], 'majority equality witnesses')
            _need(m['valid_inactive'] > 0, 'a mismatched opening must geometrically reach a box')
            _need(len(a['boxes']) == 1 or _coupled(a), 'two boxes must share an active entrance rail')
            measurements[split].append(m)
    train, test = measurements['train'], measurements['test']
    _need(any(m['area_disagrees'] and m['bends'] > 0 for m in train), 'bent raw-area counterexample')
    _need(any(m['boxes'] == 2 and m['opposite_winners'] and m['coupled'] for m in train),
          'coupled opposite collector example required')
    _need({m['boxes'] for m in train} == {1, 2}, 'training must show both collector counts')
    scale_shift = all(min(m['shape']) >= 27 for m in test) and all(max(m['shape']) <= 26 for m in train)
    count_shift = min(m['active'] for m in test) > max(m['active'] for m in train)
    _need(scale_shift or count_shift, 'a visible scale or multiplicity shift is required')


def _measure(a):
    g = a['grid']
    false_mismatch = any(g[r['entrance'][0]-1, r['entrance'][1]] != g[r['entrance'][0]+1, r['entrance'][1]]
                         for r in a['routes'])
    valid_inactive, false_match = 0, False
    for row, col, f in a['inactive']:
        try:
            route = _walk(a, row)
        except ValueError:
            continue
        if route['box'] >= 0 and len(route['path']) == len(set(route['path'])):
            valid_inactive += 1
            false_match |= bool(g[row-1, col] == g[row+1, col])
    usage = {}
    for route in a['routes']:
        for rid in set(route['flanks']):
            usage[rid] = usage.get(rid, 0)+1
    margins = [abs(2*v.count(a['signals'][0])-len(v)) for v in a['votes']]
    local_margins = [abs(sum(1 if g[p] == a['signals'][0] else -1 for p in cells)) for cells in a['rails']]
    return dict(shape=g.shape, boxes=len(a['boxes']), active=len(a['routes']), inactive=len(a['inactive']),
                valid_inactive=valid_inactive, false_pixel_mismatch=bool(false_mismatch),
                false_pixel_match=bool(false_match), coupled=_coupled(a),
                opposite_winners=len(set(a['winners'])) > 1, area_disagrees=_area_disagrees(a),
                bends=sum(len(r['turns']) for r in a['routes']),
                turn_counts=[len(r['turns']) for r in a['routes']],
                lengths=[len(r['path']) for r in a['routes']],
                arrivals=list(map(len, a['votes'])), margins=margins,
                rail_sizes=list(map(len, a['rails'])), local_margins=local_margins,
                multiplicity=max(usage.values()), approach=min(c-1 for r, c, f in _entrances(a)),
                box_dimensions=[(r1-r0+1, c1-c0+1) for r0, r1, c0, c1 in a['boxes']],
                rotation=a['rotation'], roles=(a['background'], a['token'], a['marker'], *a['signals']))


def self_test():
    """42 puzzles streamed through bounded, input-derived checks; no file I/O."""
    import hashlib
    records, shapes, structures, palettes, fingerprints = [], set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprint = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                original = inp.copy()
                _need(np.array_equal(solve(inp), out) and np.array_equal(inp, original), 'pure input-only solve')
                a = _analyse(inp)
                m = _measure(a)
                m['split'] = split
                records.append(m)
                shapes.add(inp.shape)
                palettes.add(m['roles'])
                geometry = np.zeros(a['grid'].shape, dtype=np.int8)
                geometry[a['grid'] == a['token']] = 1
                geometry[a['grid'] == a['marker']] = 2
                geometry[np.isin(a['grid'], a['signals'])] = 3
                structures.add((geometry.shape, geometry.tobytes()))
                fingerprint.update(inp.tobytes())
                # Omitting B's matching rule incorrectly erases every inactive
                # entrance token. Omitting A's majority decoding uses literal
                # entrance equality and has a different source-erasure set.
                correct = {r['start'][0] for r in a['routes']}
                literal = {row for row, col, f in _entrances(a)
                           if a['grid'][row-1, col] == a['grid'][row+1, col]}
                all_openings = {row for row, col, f in _entrances(a)}
                _need(correct != literal and correct != all_openings, 'both parent ablations must change the answer')
        fingerprints.add(fingerprint.digest())
    _need(len(fingerprints) == 42 and len(shapes) >= 50 and len(structures) >= 180 and len(palettes) >= 30,
          'structural and palette variation across 42 seeds')
    _need({m['boxes'] for m in records} == {1, 2}, 'collector count variation')
    _need(len({m['active'] for m in records}) >= 5 and len({m['inactive'] for m in records}) >= 3,
          'selection count variation')
    _need({0, 1, 2} <= {n for m in records for n in m['turn_counts']}, 'straight and bent trajectories')
    _need({0, 1, 2, 3} == {m['rotation'] for m in records}, 'all orientations')
    _need(max(n for m in records for n in m['margins']) >= 3, 'vote margins must vary')
    _need(max(m['multiplicity'] for m in records) >= 3, 'shared rail must control multiple votes')
    _need(len({v for m in records for v in m['rail_sizes']}) >= 25, 'local electorate size variation')
    for n, m in ((2, 1), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(470242213, n, m)
        validate_generated_puzzle(p)
    for n, m in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(976301808, 2, 1), generate_puzzle(976301808, 2, 1)
    _need(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
              for k in ('input', 'output')), 'whole-puzzle determinism')
    # The validator must reject wrong port ballots and wrong center votes.
    for target in ('port', 'center'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad['test'][0]
        a = _analyse(pair['input'])
        canonical = np.rot90(pair['output'], a['rotation'])
        if target == 'port':
            canonical[a['routes'][0]['end']] = a['background']
        else:
            r0, r1, c0, c1 = a['boxes'][0]
            canonical[(r0+r1)//2, (c0+c1)//2] = a['marker']
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('wrong ballot accepted')
    # Rearranging minority pixels in a region preserves identities and paths;
    # only the preserved rail pixels themselves may differ in the output.
    pair = p['train'][0]
    inp = pair['input'].copy()
    a = _analyse(inp)
    canonical = np.rot90(inp, a['rotation'])
    rail = a['rails'][0]
    values = [int(canonical[z]) for z in rail]
    for z, color in zip(rail, values[1:]+values[:1]):
        canonical[z] = color
    expected = pair['output'].copy()
    changed = np.rot90(expected, a['rotation'])
    for z in rail:
        changed[z] = canonical[z]
    _need(np.array_equal(solve(inp), expected), 'local vote permutation invariance')
    return dict(seeds=42, pairs=len(records), shapes=len(shapes), rail_geometries=len(structures),
                palettes=len(palettes), active_counts=sorted({m['active'] for m in records}),
                inactive_counts=sorted({m['inactive'] for m in records}),
                turn_counts=sorted({v for m in records for v in m['turn_counts']}),
                arrival_counts=sorted({v for m in records for v in m['arrivals']}),
                global_margins=sorted({v for m in records for v in m['margins']}),
                rail_size_range=[min(v for m in records for v in m['rail_sizes']), max(v for m in records for v in m['rail_sizes'])],
                route_length_range=[min(v for m in records for v in m['lengths']), max(v for m in records for v in m['lengths'])],
                shared_pairs=sum(m['coupled'] for m in records),
                area_counterexamples=sum(m['area_disagrees'] for m in records),
                all_pairs_defeat_pixel_matching=True, all_pairs_defeat_unconditional_movement=True,
                independent_solver=False)
