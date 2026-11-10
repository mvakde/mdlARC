"""Replumbed territories: reroutable fences determine simultaneous color floods.

Only Python/NumPy are used. No parent imports, external files, saved examples,
training labels, random state, or generator metadata participate in solve().
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

WALL, ARROW, TURN, REPAIR = 5, 2, 3, 8
FILLS = (0, 1, 4, 6, 7, 9)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _add(p, d, n=1):
    return p[0] + n*d[0], p[1] + n*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _edge(p, shape):
    return p[0] in (0, shape[0]-1) or p[1] in (0, shape[1]-1)


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _line(a, b):
    _need(a[0] == b[0] or a[1] == b[1], 'nonorthogonal segment')
    d = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
    return [_add(a, d, i) for i in range(max(abs(b[0]-a[0]), abs(b[1]-a[1]))+1)]


def _polyline(vertices):
    result = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _ray(p, d, shape):
    result = set()
    p = _add(p, d)
    while _inside(p, shape):
        result.add(p)
        p = _add(p, d)
    return result


def _regions(shape, walls):
    """Four-connected chambers; input fill colors do not define their borders."""
    h, w = shape
    # Bit frontiers avoid repeated Python/NumPy scalar indexing inside the
    # bounded layout search. Edge masks prevent horizontal wraparound.
    remaining = (1 << (h*w))-1
    for r, c in walls:
        remaining &= ~(1 << (int(r)*w+int(c)))
    left_edge = sum(1 << (r*w) for r in range(h))
    right_edge = left_edge << (w-1)
    labels, cells = [-1]*(h*w), []
    while remaining:
        frontier = remaining & -remaining
        region = frontier
        while frontier:
            adjacent = ((frontier << w) | (frontier >> w)
                        | ((frontier & ~right_edge) << 1)
                        | ((frontier & ~left_edge) >> 1))
            frontier = adjacent & remaining & ~region
            region |= frontier
        remaining ^= region
        points, label = set(), len(cells)
        while region:
            bit = region & -region
            index = bit.bit_length()-1
            labels[index] = label
            points.add(divmod(index, w))
            region ^= bit
        cells.append(points)
    return np.array(labels, dtype=np.int16).reshape(shape), cells


def _thin(shape, walls):
    mask = np.zeros(shape, dtype=bool)
    for p in walls:
        mask[p] = True
    return not np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:])


def _geometry(center, d, shape):
    side = (-d[1], d[0])
    tip = _add(center, d)
    arms = {_add(center, side), _add(center, side, -1)}
    write = {center, tip} | arms
    support = {_add(center, d, -1), _add(tip, side), _add(tip, side, -1)}
    fence, target = _add(center, d, 2), _add(center, d, 3)
    if not all(_inside(p, shape) for p in write | support | {fence, target}):
        return None
    return dict(center=center, d=d, tip=tip, arms=arms, write=write,
                support=support, fence=fence, targetcell=target)


def _arrows(inp):
    """Read T geometry and literal/implicit payloads before changing any wall."""
    arrows, covered = [], set()
    ink = set(map(tuple, np.argwhere(inp == ARROW)))
    centers = {q for p in ink for d in DIRS if _inside(q := _add(p, d), inp.shape)}
    for center in sorted(centers):
        for d in DIRS:
            g = _geometry(center, d, inp.shape)
            if g is None or not all(inp[p] == ARROW for p in g['arms'] | {g['tip']}):
                continue
            if int(inp[center]) not in (*FILLS, ARROW):
                continue
            support = {int(inp[p]) for p in g['support']}
            if len(support) != 1 or not support <= set(FILLS):
                continue
            host = support.pop()
            _need(not covered & g['write'], 'overlapping or ambiguous T arrows')
            covered.update(g['write'])
            arrows.append(dict(g, host=host, explicit=int(inp[center]) != ARROW,
                               payload=host if inp[center] == ARROW else int(inp[center])))
    _need(bool(arrows) and ink == covered & ink, 'unexplained T ink')
    clean = inp.copy()
    for a in arrows:
        for p in a['write']:
            clean[p] = a['host']
    return arrows, clean


def _repair_walls(inp, protected=()):
    walls = set(map(tuple, np.argwhere((inp == WALL) | (inp == REPAIR))))
    original, additions = set(walls), set()
    protected = set(protected)
    # Only single-cell, opposite-neighbor omissions are part of this family.
    # Read all witnesses from the same snapshot: no iterative gap propagation.
    for p in np.ndindex(inp.shape):
        if p in original or p in protected or inp[p] == TURN:
            continue
        if any(_add(p, d) in original and _add(p, d, -1) in original
               for d in ((1, 0), (0, 1))):
            additions.add(p)
    return walls | additions, additions


def _edits(inp, walls):
    edits = []
    for q0 in np.argwhere(inp == TURN):
        q = tuple(map(int, q0))
        candidates = []
        for d in DIRS:
            pivot, start = _add(q, d, -1), _add(q, d, -2)
            if pivot not in walls or start not in walls:
                continue
            side = (-d[1], d[0])
            if not any(_add(pivot, side, sign) in walls for sign in (-1, 1)):
                continue
            previous, current, old = pivot, start, []
            for _ in range(inp.size):
                if current in old:
                    break
                old.append(current)
                following = [p for p in _neighbors(current, walls) if p != previous]
                if not following:
                    if _edge(current, inp.shape):
                        candidates.append(dict(pivot=pivot, d=d, old=set(old),
                                               walk=[pivot]+old,
                                               new=_ray(pivot, d, inp.shape)))
                    break
                if len(following) != 1:
                    break
                previous, current = current, following[0]
        _need(len(candidates) == 1, 'redirect must select one terminal boundary branch')
        edits.append(candidates[0])
    _need(bool(edits), 'at least one redirect is required')
    for i, a in enumerate(edits):
        for b in edits[i+1:]:
            _need(not a['old'] & b['old'] and a['pivot'] not in b['old']
                  and b['pivot'] not in a['old'], 'overlapping old branches')
    return edits


def _replumb(walls, edits):
    removed = set().union(*(e['old'] for e in edits))
    added = set().union(*(e['new'] for e in edits))
    return (walls - removed) | added


def _render(shape, walls, ids, assignments):
    out = np.full(shape, WALL, dtype=np.int8)
    for label, color in assignments.items():
        out[ids == label] = color
    return out


def _decode(input_grid):
    inp = np.asarray(input_grid)
    _need(inp.ndim == 2 and all(1 <= n <= 30 for n in inp.shape), 'bad grid shape')
    _need(np.issubdtype(inp.dtype, np.integer) and bool(np.all((inp >= 0) & (inp <= 9))),
          'bad grid colors or dtype')
    arrows, clean = _arrows(inp)
    protected = set().union(*(a['write'] for a in arrows))
    walls, gaps = _repair_walls(clean, protected)
    _need(_thin(inp.shape, walls), 'old fences contain a solid 2x2 block')
    _, original_regions = _regions(inp.shape, walls)
    for region in original_regions:
        visible = {int(clean[p]) for p in region if clean[p] != TURN}
        _need(len(visible) == 1 and visible <= set(FILLS), 'old chambers must have uniform recoverable fills')
    edits = _edits(clean, walls)
    final = _replumb(walls, edits)
    _need(_thin(inp.shape, final), 'new fences contain a solid 2x2 block')
    ids, cells = _regions(inp.shape, final)
    assignments = {}
    for a in arrows:
        source, target = int(ids[a['center']]), int(ids[a['targetcell']])
        _need(a['fence'] in final and source >= 0 and target >= 0 and source != target,
              'T must point across a final fence into another chamber')
        _need(all(ids[p] == source for p in a['write'] | a['support']),
              'T source straddles a final fence')
        _need(not (a['write'] | a['support']) & walls, 'T obscures an old fence')
        _need(target not in assignments or assignments[target] == a['payload'],
              'conflicting incoming colors')
        assignments[target] = a['payload']
        a.update(source=source, target=target)
    _need(set(assignments) == set(range(len(cells))), 'every final chamber needs a color instruction')
    out = _render(inp.shape, final, ids, assignments)
    info = dict(arrows=arrows, clean=clean, walls=walls, gaps=gaps, edits=edits,
                final=final, ids=ids, cells=cells, assignments=assignments)
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict solely from the supplied input using the fixed family conventions."""
    return _decode(input_grid)[0]


def _candidates(base, old, final, ids, direction_sites):
    result = []
    for fence in sorted(final):
        for d in DIRS:
            g = _geometry(_add(fence, d, -2), d, base.shape)
            if g is None:
                continue
            context = g['write'] | g['support'] | {g['targetcell']}
            if context & (old | direction_sites):
                continue
            source, target = int(ids[g['center']]), int(ids[g['targetcell']])
            if source < 0 or target < 0 or source == target:
                continue
            if any(ids[p] != source for p in g['write'] | g['support']):
                continue
            colors = {int(base[p]) for p in g['write'] | g['support']}
            if len(colors) != 1 or not colors <= set(FILLS):
                continue
            result.append(dict(g, source=source, target=target, host=colors.pop()))
    return result


def _layout(rng, redirects, cuts, shape):
    h, w = shape
    n = int(rng.integers(2, min(5, (h-5)//3+1)))
    # Independent pixel coordinates, with spacing for readable thin paths.
    levels = sorted(map(int, rng.choice(np.arange(3, h-3), n, replace=False)))
    if min(np.diff(levels)) < 4 or (redirects == 2 and levels[-1]-levels[0] < 7):
        return None
    columns = [int(rng.integers(3, w-3))]
    for _ in levels:
        options = [c for c in range(3, w-3) if abs(c-columns[-1]) >= 4]
        if not options:
            return None
        columns.append(int(rng.choice(options)))
    vertices = [(0, columns[0])]
    for i, row in enumerate(levels):
        vertices.extend(((row, columns[i]), (row, columns[i+1])))
    vertices.append((h-1, columns[-1]))
    path = _polyline(vertices)
    old = set(path)
    if len(old) != len(path) or any(len(_neighbors(p, old)) > 2 for p in old):
        return None
    choices = [(kind, i) for kind in ('upper_start', 'upper_end', 'lower_start', 'lower_end')
               for i in range(n)]
    selected = ([('upper_'+str(rng.choice(('start', 'end'))), 0),
                 ('lower_'+str(rng.choice(('start', 'end'))), n-1)]
                if redirects == 2 else [choices[int(rng.integers(len(choices)))]] )
    edits = []
    for kind, i in selected:
        if kind.startswith('upper'):
            pivot = (levels[i], columns[i] if kind.endswith('start') else columns[i+1])
            d = (1, 0) if kind.endswith('start') else (0, int(np.sign(columns[i+1]-columns[i])))
            at = path.index(pivot)
            branch, walk = set(path[:at]), list(reversed(path[:at+1]))
        else:
            pivot = (levels[i], columns[i] if kind.endswith('start') else columns[i+1])
            d = (0, int(np.sign(columns[i]-columns[i+1]))) if kind.endswith('start') else (-1, 0)
            at = path.index(pivot)
            branch, walk = set(path[at+1:]), path[at:]
        edits.append(dict(pivot=pivot, d=d, old=branch, walk=walk, new=_ray(pivot, d, shape)))
    final = _replumb(old, edits)
    if not _thin(shape, final):
        return None
    removed = set().union(*(e['old'] for e in edits))
    forbidden = removed | {_add(p, d) for p in removed for d in DIRS}
    forbidden |= {e['pivot'] for e in edits} | {_add(e['pivot'], d) for e in edits for d in DIRS}
    # Additional cuts are selected from actual current chamber runs. They may
    # terminate on a future ray, producing an initially dangling fence.
    for _ in range(cuts):
        ids, regions = _regions(shape, final)
        options = []
        for axis in (0, 1):
            for across in range(3, shape[1-axis]-3):
                run = []
                for along in range(shape[axis]+1):
                    p = (along, across) if axis == 0 else (across, along)
                    if along < shape[axis] and p not in final:
                        run.append(p)
                    else:
                        if len(run) >= 7 and not set(run) & forbidden:
                            options.append(set(run))
                        run = []
        rng.shuffle(options)
        for cut in options[:80]:
            trial = final | cut
            _, regions2 = _regions(shape, trial)
            if len(regions2) != len(regions)+1 or min(map(len, regions2)) < 30:
                continue
            if not _thin(shape, old | cut) or not _thin(shape, trial):
                continue
            # No phantom one-cell omissions even in the undamaged geometry.
            wall_grid = np.zeros(shape, dtype=np.int8)
            for p in old | cut:
                wall_grid[p] = WALL
            if _repair_walls(wall_grid)[0] != old | cut:
                continue
            old |= cut
            final = trial
            break
        else:
            return None
    raw = np.zeros(shape, dtype=np.int8)
    for p in old:
        raw[p] = WALL
    for e in edits:
        q = _add(e['pivot'], e['d'])
        if q in old:
            return None
        raw[q] = TURN
    try:
        recovered = _edits(raw, old)
    except AssertionError:
        return None
    if _replumb(old, recovered) != final or _repair_walls(raw)[0] != old:
        return None
    return old, final, edits


def _plan(rng, candidates, count):
    # Cover constrained chambers first; randomized greedy attempts are bounded.
    by_target = [[a for a in candidates if a['target'] == i] for i in range(count)]
    if any(not group for group in by_target):
        return None
    for _ in range(20):
        chosen, blocked, context = [], set(), set()
        order = sorted(range(count), key=lambda i: len(by_target[i]) + float(rng.random()))
        for label in order:
            options = [a for a in by_target[label] if not a['write'] & (blocked | context)
                       and not (a['support'] | {a['targetcell']}) & blocked]
            if not options:
                break
            a = dict(options[int(rng.integers(len(options)))])
            chosen.append(a)
            blocked |= a['write'] | {_add(p, d) for p in a['write'] for d in DIRS}
            context |= a['support'] | {a['targetcell']}
        if len(chosen) == count:
            return chosen
    return None


def _features(inp, out, info):
    arrows, ids = info['arrows'], info['ids']
    simultaneous = any(not a['explicit'] and info['assignments'][a['source']] != a['host'] for a in arrows)
    original_mask = np.isin(inp, FILLS)
    for a in arrows:
        if a['explicit']:
            original_mask[a['center']] = False
    original_colors = set(map(int, inp[original_mask]))
    locality = any(len(set(map(int, out[(inp == color) & (out != WALL)]))) > 1 for color in FILLS)
    branches = [e['walk'] for e in info['edits']]
    bends = []
    for walk in branches:
        ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(walk, walk[1:])]
        bends.append(sum(a != b for a, b in zip(ds, ds[1:])))
    # Geometry-only ablation: retain the old fences and apply the same T seeds
    # to the old chambers. A new target footprint must differ from its old one.
    old_ids, _ = _regions(inp.shape, info['walls'])
    footprint_changed = any(not np.array_equal(old_ids == old_ids[a['targetcell']], ids == a['target'])
                            for a in arrows)
    return dict(shape=inp.shape, chambers=len(info['cells']), arrows=len(arrows),
                explicit=sum(a['explicit'] for a in arrows),
                singleton=any(a['explicit'] and a['payload'] not in original_colors for a in arrows),
                simultaneous=simultaneous, locality=locality,
                directions=tuple(DIRS.index(e['d']) for e in info['edits']),
                redirects=len(info['edits']), bends=tuple(bends), gaps=len(info['gaps']),
                repairs=int(np.count_nonzero(inp == REPAIR)), footprint_changed=footprint_changed,
                new_fence_arrows=sum(a['fence'] in info['final']-info['walls'] for a in arrows),
                erased_repairs=sum(p in e['old'] for p in map(tuple, np.argwhere(inp == REPAIR)) for e in info['edits']),
                fill_changes=int(np.count_nonzero(np.isin(inp, FILLS) & np.isin(out, FILLS) & (inp != out))))


def _build_pair(rng, redirects, cuts, mode, rotation):
    shape = (int(rng.integers(17, 31)), int(rng.integers(17, 31)))
    layout = _layout(rng, redirects, cuts, shape)
    if layout is None:
        return None
    old, final, edits = layout
    original_ids, original_regions = _regions(shape, old)
    ids, regions = _regions(shape, final)
    if not 2 <= len(regions) <= 6 or min(map(len, regions)) < 30:
        return None
    palette = list(map(int, rng.permutation(FILLS)))
    # Keep one potential explicit color absent from the underlying map.
    pool = palette[:-1]
    colors = [pool[i % len(pool)] for i in range(len(original_regions))]
    if len(colors) > 2 and rng.random() < .65:
        colors[-1] = colors[0]
    rng.shuffle(colors)
    base = np.full(shape, WALL, dtype=np.int8)
    for i, c in enumerate(colors):
        base[original_ids == i] = c
    direction_sites = {_add(e['pivot'], e['d']) for e in edits}
    candidates = _candidates(base, old, final, ids, direction_sites)
    arrows = _plan(rng, candidates, len(regions))
    if arrows is None:
        return None
    assignments = {a['target']: a['host'] for a in arrows}
    explicit_target = int(rng.integers(len(regions))) if mode == 'mixed' else None
    if explicit_target is not None:
        assignments[explicit_target] = palette[-1]
    for a in arrows:
        a.update(payload=assignments[a['target']], explicit=a['target'] == explicit_target)
    # Optional consistent redundant instructions teach that there is no order
    # priority. They also permit fan-out without changing the flood rule.
    used = set().union(*(a['write'] | a['support'] | {a['targetcell']} for a in arrows))
    halo = used | {_add(p, d) for a in arrows for p in a['write'] for d in DIRS}
    extras = int(rng.integers(0, 3))
    rng.shuffle(candidates)
    for a0 in candidates:
        if not extras:
            break
        if (a0['write'] | a0['support'] | {a0['targetcell']}) & halo:
            continue
        payload = assignments[a0['target']]
        if mode == 'implicit' and payload != a0['host']:
            continue
        a = dict(a0, payload=payload, explicit=payload != a0['host'])
        arrows.append(a)
        halo |= a['write'] | a['support'] | {a['targetcell']} | {_add(p, d) for p in a['write'] for d in DIRS}
        extras -= 1
    inp = base.copy()
    for a in arrows:
        for p in a['write']:
            inp[p] = ARROW
        if a['explicit']:
            inp[a['center']] = a['payload']
    for q in direction_sites:
        inp[q] = TURN
    # Explicit waypoints may occupy bends, pivots, retained or deleted cells.
    wall_sites = sorted(old)
    rng.shuffle(wall_sites)
    for p in wall_sites[:int(rng.integers(1, 5))]:
        inp[p] = REPAIR
    protected = set().union(*(a['write'] | a['support'] | {a['targetcell']} for a in arrows))
    gap_sites = [p for p in wall_sites if not _edge(p, shape) and p not in protected
                 and inp[p] == WALL and len(_neighbors(p, old)) == 2
                 and any(_add(p, d) in old and _add(p, d, -1) in old for d in ((1, 0), (0, 1)))]
    rng.shuffle(gap_sites)
    for p in gap_sites[:int(rng.integers(1, 4))]:
        fills = [int(base[q]) for d in DIRS if _inside(q := _add(p, d), shape) and q not in old]
        if not fills:
            continue
        prior = int(inp[p])
        inp[p] = fills[0]
        if _repair_walls(inp, protected)[0] != old:
            inp[p] = prior
    expected = _render(shape, final, ids, assignments)
    try:
        predicted, info = _decode(inp)
    except AssertionError:
        return None
    if not np.array_equal(predicted, expected) or info['walls'] != old:
        return None
    f = _features(inp, expected, info)
    if not f['fill_changes'] or not f['footprint_changed']:
        return None
    if mode == 'implicit' and not f['simultaneous']:
        return None
    if mode == 'mixed' and not (f['singleton'] and f['locality'] and f['new_fence_arrows']):
        return None
    reflect = bool(rng.integers(2))
    pair = {key: np.rot90(np.fliplr(a) if reflect else a, rotation).copy()
            for key, a in (('input', inp), ('output', expected))}
    return pair


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _integer(num_train) or not _integer(num_test) or not 2 <= num_train <= 10
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('require integer counts 2..10 / 1..4 with num_train > num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    orientation = int(rng.integers(4))
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            mode = 'implicit' if split == 'train' and i == 0 else 'mixed'
            redirects = 1 if split == 'train' else 2
            cuts = int(rng.integers(0 if i == 0 else 1, 3 if split == 'train' else 4))
            rotation = (orientation+i) % 4 if split == 'train' and i < 2 else int(rng.integers(4))
            for attempt in range(350):
                # Simplify only the number of optional cuts after 175 attempts.
                # Both inherited mechanisms and the curriculum remain intact.
                pair = _build_pair(rng, redirects, cuts if attempt < 175 else min(cuts, 1), mode, rotation)
                if pair is not None and _key(pair['input']) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    break
            else:
                raise RuntimeError('bounded family construction exhausted')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair schema')
            inp, out = (np.asarray(pair[k]) for k in ('input', 'output'))
            for grid in (inp, out):
                _need(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), 'bad dimensions')
                _need(np.issubdtype(grid.dtype, np.integer) and bool(np.all((0 <= grid) & (grid <= 9))), 'bad colors')
            _need(inp.shape == out.shape and _key(inp) not in seen, 'shape change or duplicate input')
            seen.add(_key(inp))
            predicted, info = _decode(inp)
            _need(np.array_equal(predicted, out), 'incorrect fence or chamber pixel')
            _need(not np.any(np.isin(out, (ARROW, TURN, REPAIR))), 'annotation survived')
            f = _features(inp, out, info)
            _need(f['fill_changes'] > 0 and f['footprint_changed'], 'one parent is inert')
            # Reversing the edit order must preserve every final fence pixel.
            _need(_replumb(info['walls'], list(reversed(info['edits']))) == info['final'], 'sequential edit leak')
            features[split].append(f)
    for name in ('singleton', 'simultaneous', 'locality', 'new_fence_arrows'):
        _need(any(f[name] for f in features['train']), 'missing demonstration: '+name)
    _need(any(f['repairs'] for f in features['train']), 'wall repair is not taught')


def _manual_regressions():
    """Small coordinate-derived answers; not generated or derived via solve()."""
    base = np.full((13, 15), 4, dtype=np.int8)
    base[:5, 6:] = 1
    base[:6, 5] = WALL
    base[5, 5:] = WALL
    inp = base.copy()
    inp[1, 5], inp[3, 5], inp[6, 5] = REPAIR, 4, TURN
    for p in ((3, 8), (3, 9), (3, 10), (4, 9),
              (7, 10), (7, 11), (7, 12), (6, 11)):
        inp[p] = ARROW
    expected = np.full(base.shape, 4, dtype=np.int8)
    expected[6:, 6:] = 1
    expected[5, 5:] = WALL
    expected[5:, 5] = WALL
    _need(np.array_equal(solve(inp), expected), 'manual implicit flood regression')
    # A removed: retain the original patch colors on the correct new geometry.
    without_a = base.copy()
    without_a[:5, 5] = 4
    without_a[5:, 5] = WALL
    _need(without_a[8, 10] == 4 and expected[8, 10] == 1, 'A ablation failed')
    # B removed: the same two transfers paint the original chambers instead.
    without_b = np.full(base.shape, 1, dtype=np.int8)
    without_b[:5, 6:] = 4
    without_b[:6, 5] = WALL
    without_b[5, 5:] = WALL
    _need(without_b[10, 2] == 1 and expected[10, 2] == 4, 'B flood ablation failed')
    _need(inp[3, 9] == ARROW and expected[3, 9] == 4 and expected[8, 10] == 1,
          'original-color snapshot witness failed')
    literal, literal_expected = inp.copy(), expected.copy()
    literal[3, 9] = 6
    literal_expected[6:, 6:] = 6
    _need(np.array_equal(solve(literal), literal_expected), 'manual singleton regression')
    conflicting = inp.copy()
    for p in ((8, 3), (10, 3), (9, 4)):
        conflicting[p] = ARROW
    conflicting[9, 3] = 9
    try:
        solve(conflicting)
    except AssertionError:
        pass
    else:
        raise AssertionError('conflicting incoming payloads accepted')
    return [dict(input=inp, output=expected), dict(input=literal, output=literal_expected)]


def self_test():
    """42 seeds, streamed measurements, metamorphic and semantic negative tests."""
    _manual_regressions()
    signatures, shapes, structures, directions = set(), set(), set(), set()
    chambers, arrow_counts, bends, repairs, gaps = set(), set(), set(), set(), set()
    measured = dict(pairs=0, singleton=0, locality=0, simultaneous=0,
                    new_fence_arrows=0, erased_repairs=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                fingerprint.update(inp.tobytes())
                fingerprint.update(out.tobytes())
                _, info = _decode(inp)
                f = _features(inp, out, info)
                measured['pairs'] += 1
                for name in measured.keys()-{'pairs'}:
                    measured[name] += int(f[name])
                shapes.add(inp.shape)
                # Binary fence geometry and chamber identity ignore palettes.
                structures.add((inp.shape, hashlib.sha256((info['ids']+1).astype(np.int8).tobytes()).hexdigest()))
                directions.update(f['directions'])
                chambers.add(f['chambers'])
                arrow_counts.add(f['arrows'])
                bends.update(f['bends'])
                repairs.add(f['repairs'])
                gaps.add(f['gaps'])
                if seed < 3:
                    for k in range(4):
                        transformed = np.rot90(np.fliplr(inp), k).copy()
                        _need(np.array_equal(solve(transformed), np.rot90(np.fliplr(out), k)), 'symmetry failure')
        signatures.add(fingerprint.hexdigest())
        if seed < 2:
            duplicate = generate_puzzle(seed)
            _need(all(_key(p[k]) == _key(q[k]) for s in puzzle for p, q in zip(puzzle[s], duplicate[s])
                      for k in ('input', 'output')), 'nondeterministic seed')
        # Reject changed labels, including a fence cell and a remote flood cell.
        if seed < 3:
            query = puzzle['test'][0]
            sites = [tuple(np.argwhere(query['output'] == WALL)[0]),
                     tuple(np.argwhere(query['output'] != WALL)[-1])]
            for p in sites:
                bad = {'train': puzzle['train'], 'test': [dict(pair) for pair in puzzle['test']]}
                bad['test'][0]['output'] = query['output'].copy()
                bad['test'][0]['output'][p] = (int(query['output'][p])+1) % 10
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('corrupt output accepted')
    _need(len(signatures) == 42 and len(shapes) >= 50 and len(structures) >= 150, 'collapsed structural variation')
    _need(directions == set(range(4)), 'missing redirect orientations')
    _need({2, 3, 4} <= chambers and len(arrow_counts) >= 4, 'collapsed chamber/instruction counts')
    _need({0, 1, 2} <= bends and max(bends) >= 3, 'missing bent old branches')
    _need(len(repairs) >= 3 and max(gaps) >= 2, 'collapsed damage')
    _need(all(measured[k] > 0 for k in measured), 'missing interaction coverage')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+10*a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, **measured, distinct_puzzles=len(signatures), shapes=len(shapes),
                fence_partitions=len(structures), chamber_counts=sorted(chambers),
                arrow_counts=sorted(arrow_counts), old_bends=sorted(bends),
                repair_counts=sorted(repairs), gap_counts=sorted(gaps))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
