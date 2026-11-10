"""Relational grafting: simultaneous color-addressed transport of geometry.

Only the input is used by solve. Construction paints normalized cells directly;
the solver rediscovers the cores, reads their pixel motifs, and maps rectangles.
No files, execution traces, caches, or random state are used during solving.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 468836276, 2582458, 533927283, 571926843, 157752095,
              530026506, 326084494, 482195096, 559220889, 66638699, 614006091,
              763399116, 265626068, 552765967, 427655327, 573740119, 959004900,
              974155395, 196610683, 936978403, 601085482, 511986924, 941366829,
              738878290, 495096517, 418397922, 774846684, 541627603, 13124500,
              792020659, 62545381, 162770403, 418314759, 499712445, 407470560,
              490751730, 86637749, 438074091, 66546448, 689405533, 27507149)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " is out of range")
    return value


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= s <= 30 for s in g.shape)
    assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.uint8).tobytes()


def _bounds(h, w, r, c):
    return ((0, r, 0, c), (0, r, c+1, w), (r+1, h, 0, c), (r+1, h, c+1, w))


def _connected(cells):
    todo = set(cells)
    if not todo:
        return False
    stack = [todo.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in D4:
            q = r+dr, c+dc
            if q in todo:
                todo.remove(q)
                stack.append(q)
    return not todo


def _flip(tail, source_panel, target_panel):
    return {(2-r if source_panel//2 != target_panel//2 else r,
             2-c if source_panel % 2 != target_panel % 2 else c): color
            for (r, c), color in tail.items()}


def _core_cells():
    return {(r, c) for r in range(3) for c in range(3)}


def _pixels(tail, u, top, left):
    """Physical occupancy, including the 3u square core."""
    return {(top+r*u+dr, left+c*u+dc)
            for r, c in (_core_cells() | set(tail))
            for dr in range(u) for dc in range(u)}


def _paint(panel, top, left, u, body, value, tail):
    """Construction renderer, with exact cell scaling and panel clipping."""
    h, w = panel.shape
    for (r, c), color in tail.items():
        y, x = top+r*u, left+c*u
        y0, y1, x0, x1 = max(0, y), min(h, y+u), max(0, x), min(w, x+u)
        if y0 < y1 and x0 < x1:
            panel[y0:y1, x0:x1] = color
    panel[top:top+3*u, left:left+3*u] = body
    panel[top+u:top+2*u, left+u:left+2*u] = value


def _attachment(p, top, left, u):
    r, c = p
    return (((r == top-1 or r == top+3*u) and left+u <= c < left+2*u)
            or ((c == left-1 or c == left+3*u) and top+u <= r < top+2*u))


def _trail(rng, shape, occupied, top, left, u, interior=False):
    """One nonbranching path: attached end, or a contrasting interior contact."""
    h, w = shape
    free = lambda p: 1 <= p[0] < h-1 and 1 <= p[1] < w-1 and p not in occupied
    if interior:
        choices = [([(top-1, c) for c in range(left-1, left+3*u+1)]),
                   ([(top+3*u, c) for c in range(left-1, left+3*u+1)]),
                   ([(r, left-1) for r in range(top-1, top+3*u+1)]),
                   ([(r, left+3*u) for r in range(top-1, top+3*u+1)])]
        for i in rng.permutation(4):
            path = choices[int(i)]
            if all(free(p) for p in path):
                return set(path)
        return None
    sockets = []
    for offset in range(u, 2*u):
        sockets.extend([((top-1, left+offset), (-1, 0)),
                        ((top+3*u, left+offset), (1, 0)),
                        ((top+offset, left-1), (0, -1)),
                        ((top+offset, left+3*u), (0, 1))])
    for index in rng.permutation(len(sockets)):
        start, outward = sockets[int(index)]
        if not free(start):
            continue
        for _ in range(5):
            path = [start]
            desired = int(rng.integers(3, 9))
            while len(path) < desired:
                p = path[-1]
                candidates = [(p[0]+dr, p[1]+dc) for dr, dc in D4]
                if len(path) == 1:
                    candidates = [(p[0]+outward[0], p[1]+outward[1])]
                candidates = [q for q in candidates if free(q) and q not in path
                              and not _attachment(q, top, left, u)
                              and sum(abs(q[0]-a)+abs(q[1]-b) == 1
                                      for a, b in path) == 1]
                if not candidates:
                    break
                path.append(candidates[int(rng.integers(len(candidates)))])
            if len(path) >= 3:
                return set(path)
    return None


def _grow_tail(rng, colors, count):
    """Connected polyomino branches, with independently varying attachment sites."""
    core, points = _core_cells(), set()
    preferred = D4[int(rng.integers(4))]
    for _ in range(count):
        frontier = sorted({(r+dr, c+dc) for r, c in core | points for dr, dc in D4
                           if -3 <= r+dr <= 5 and -3 <= c+dc <= 5
                           and (r+dr, c+dc) not in core | points})
        # Bias some growth toward a long arm, without freezing a shape table.
        if points and rng.random() < .55:
            score = max(r*preferred[0]+c*preferred[1] for r, c in frontier)
            frontier = [p for p in frontier if p[0]*preferred[0]+p[1]*preferred[1] == score]
        points.add(frontier[int(rng.integers(len(frontier)))])
    return {p: int(rng.choice(colors)) for p in sorted(points)}


def _asymmetric(tail):
    return all({(2-r if v else r, 2-c if h else c): x for (r, c), x in tail.items()} != tail
               for v, h in ((0, 1), (1, 0), (1, 1)))


def _make_pair(rng, mode):
    """Return None when a sampled geometry cannot meet its visible constraints."""
    is_test = mode == "test"
    n = 3 if is_test else 4 if mode in ("fork", "chain") else int(rng.integers(3, 5))
    panels = list(map(int, rng.choice(4, n, replace=False)))
    palette = list(map(int, rng.permutation(10)))
    bg, line = palette[:2]
    bodies, missing = palette[2:2+n], palette[2+n]
    details = palette[3+n:]
    details = details[:int(rng.integers(1, len(details)+1))]
    units = list(map(int, rng.choice([1, 2], n, p=[.7, .3])))
    if mode == "fork":
        # Node zero is a remote donor; the other three show all three flips.
        values = [bodies[1]] + [bodies[0]]*(n-1)
        active = [False, True, True, True]
        units = [int(rng.integers(1, 3)), 1, 2, int(rng.integers(1, 3))]
        tail_counts = [int(rng.integers(2, 6))] + list(map(int, rng.integers(0, 4, n-1)))
    elif mode == "chain":
        # A->B->C->missing, all active: B changes while supplying A; C is
        # deleted while supplying B. A fourth inactive token has a valid lookup.
        values = [bodies[1], bodies[2], missing, bodies[0]]
        active = [True, True, True, False]
        units = [1, 2, 1, int(rng.integers(1, 3))]
        tail_counts = list(map(int, rng.integers(1, 5, n)))
    else:
        values = [int(rng.choice([b for b in bodies if b != body]+[missing])) for body in bodies]
        active = list(map(bool, rng.integers(0, 2, n)))
        tail_counts = list(map(int, rng.integers(7, 10, n))) if is_test else list(map(int, rng.integers(0, 7, n)))
    tails = [_grow_tail(rng, details, k) for k in tail_counts]
    if mode == "fork" and not _asymmetric(tails[0]):
        return None
    donors = [bodies.index(v) if v in bodies else None for v in values]
    successes = [i for i in range(n) if active[i] and donors[i] is not None and tails[donors[i]]]
    if not successes:
        return None
    clip_flags = [False]*n
    if mode == "chain":
        clip_flags[int(rng.choice(successes))] = True
    elif mode not in ("fork",):
        for i in successes:
            clip_flags[i] = bool(rng.random() < (.6 if is_test else .3))
    # Axis spans are sampled independently subject to the 30-cell bound.
    low = 11 if mode in ("fork", "chain") else 9
    top_h = int(rng.integers(low, 16))
    bot_h = int(rng.integers(low, min(15, 29-top_h)+1))
    left_w = int(rng.integers(low, 16))
    right_w = int(rng.integers(low, min(15, 29-left_w)+1))
    bounds = _bounds(top_h+bot_h+1, left_w+right_w+1, top_h, left_w)
    inp = np.full((top_h+bot_h+1, left_w+right_w+1), bg, dtype=np.int8)
    inp[top_h, :] = line
    inp[:, left_w] = line
    out = inp.copy()
    for i, p in enumerate(panels):
        r0, r1, c0, c1 = bounds[p]
        shape, u = (r1-r0, c1-c0), units[i]
        d = donors[i]
        retained = active[i] and d is not None
        new_tail = _flip(tails[d], panels[d], p) if retained else {}
        old_points = _pixels(tails[i], u, 0, 0)
        new_points = _pixels(new_tail, u, 0, 0) if retained else set()
        positions = []
        for y in range(shape[0]-3*u+1):
            for x in range(shape[1]-3*u+1):
                if not all(0 <= y+r < shape[0] and 0 <= x+c < shape[1] for r, c in old_points):
                    continue
                if retained:
                    visible = sum(0 <= y+r < shape[0] and 0 <= x+c < shape[1] for r, c in new_points)
                    if (visible < len(new_points)) != clip_flags[i] or visible <= 9*u*u:
                        continue
                positions.append((y, x))
        if not positions:
            return None
        selected = None
        for j in rng.permutation(len(positions)):
            y, x = positions[int(j)]
            occupied = {(y+r, x+c) for r, c in old_points | new_points}
            path = None
            if active[i]:
                path = _trail(rng, shape, occupied, y, x, u)
                if path is None:
                    continue
            elif mode == "fork" and i == 0:
                path = _trail(rng, shape, occupied, y, x, u, interior=True)
                if path is None:
                    continue
            selected = y, x, path or set()
            break
        if selected is None:
            return None
        y, x, path = selected
        ip, op = inp[r0:r1, c0:c1], out[r0:r1, c0:c1]
        _paint(ip, y, x, u, bodies[i], values[i], tails[i])
        if retained:
            _paint(op, y, x, u, bodies[i], values[d], new_tail)
        for q in path:
            ip[q] = op[q] = line
    return {"input": inp, "output": out}


def _candidate_core(panel, bg, line):
    """Locate a unique square ring with a different square central third."""
    h, w = panel.shape
    candidates = []
    for u in range(1, min(h, w)//3+1):
        for r in range(h-3*u+1):
            for c in range(w-3*u+1):
                body, value = int(panel[r, c]), int(panel[r+u, c+u])
                if body in (bg, line) or value in (bg, line, body):
                    continue
                if not (panel[r+3*u-1, c] == body and panel[r, c+3*u-1] == body
                        and panel[r+3*u-1, c+3*u-1] == body):
                    continue
                core = panel[r:r+3*u, c:c+3*u]
                if not (np.all(core[:u] == body) and np.all(core[2*u:] == body)
                        and np.all(core[u:2*u, :u] == body)
                        and np.all(core[u:2*u, 2*u:] == body)
                        and np.all(core[u:2*u, u:2*u] == value)):
                    continue
                tail, valid = {}, True
                foreground = np.argwhere((panel != bg) & (panel != line))
                offsets = {((int(y)-r)//u, (int(x)-c)//u) for y, x in foreground}
                for a, b in offsets-_core_cells():
                    y, x = r+a*u, c+b*u
                    if y < 0 or x < 0 or y+u > h or x+u > w:
                        valid = False
                        break
                    tile = panel[y:y+u, x:x+u]
                    color = int(tile[0, 0])
                    if color in (bg, line, body, value) or not np.all(tile == color):
                        valid = False
                        break
                    tail[a, b] = color
                if valid and _connected(_core_cells() | set(tail)):
                    candidates.append(dict(top=r, left=c, unit=u, body=body,
                                           value=value, tail=tail, pixels=foreground))
    assert len(candidates) == 1, "core and its motif must be uniquely identifiable"
    return candidates[0]


def _decode(grid):
    g = _grid(grid)
    rows = [r for r in range(1, g.shape[0]-1) if np.all(g[r] == g[r, 0])]
    cols = [c for c in range(1, g.shape[1]-1) if np.all(g[:, c] == g[0, c])]
    crosses = [(r, c) for r in rows for c in cols if g[r, 0] == g[0, c]]
    assert len(crosses) == 1, "one full cross is required"
    r, c = crosses[0]
    line = int(g[r, c])
    bounds = _bounds(*g.shape, r, c)
    values = np.concatenate([g[a:b, x:y].ravel() for a, b, x, y in bounds])
    counts = np.bincount(values, minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(counts.argmax())
    assert bg != line
    nodes = []
    for p, (a, b, x, y) in enumerate(bounds):
        panel = g[a:b, x:y]
        if not np.any((panel != bg) & (panel != line)):
            assert not np.any(panel == line), "empty panels carry no trails"
            continue
        node = _candidate_core(panel, bg, line)
        path = set(map(tuple, np.argwhere(panel == line).tolist()))
        ends = set()
        if path:
            assert len(path) >= 3 and _connected(path)
            graph = {q: {(q[0]+dr, q[1]+dc) for dr, dc in D4} & path for q in path}
            assert all(len(v) in (1, 2) for v in graph.values()), "trails must be open paths"
            ends = {q for q, v in graph.items() if len(v) == 1}
            assert len(ends) == 2
            assert all(0 < q[0] < panel.shape[0]-1 and 0 < q[1] < panel.shape[1]-1 for q in path)
        attached = {q for q in ends if _attachment(q, node['top'], node['left'], node['unit'])}
        assert len(attached) <= 1
        node.update(panel=p, bounds=(a, b, x, y), path=path, active=bool(attached),
                    interior_contact=any(_attachment(q, node['top'], node['left'], node['unit'])
                                         for q in path-ends))
        nodes.append(node)
    assert 2 <= len(nodes) <= 4
    bodies = {n['body'] for n in nodes}
    assert len(bodies) == len(nodes), "donor bodies must be unique"
    identities = bodies | {n['value'] for n in nodes}
    assert all(not (set(n['tail'].values()) & identities) for n in nodes)
    return g, bg, line, bounds, nodes


def _predict(grid, ablation=None):
    """Input-only rectangle transport; ablations are used only by validation."""
    g, bg, line, bounds, nodes = _decode(grid)
    output = g.copy()
    by_body = {n['body']: n for n in nodes}
    records = []
    for n in nodes:
        a, b, x, y = n['bounds']
        panel = output[a:b, x:y]
        # Erase all original object pixels before any graft. Trails survive.
        if not (ablation == 'add_old' and n['active'] and n['value'] in by_body):
            for r, c in n['pixels']:
                panel[r, c] = bg
        d = by_body.get(n['value'])
        if ablation == 'prune_failed_donors' and d is not None and d['value'] not in by_body:
            d = None
        active = n['active'] or ablation == "all_tokens"
        if not active or d is None:
            continue
        if ablation == "own_motif":
            d = n
        elif ablation == "extra_hop":
            d = by_body.get(d['value'], d)
        u, t, l = n['unit'], n['top'], n['left']
        total = visible = 0
        transported = {} if ablation == "centers_only" else d['tail']
        for (r, c), color in transported.items():
            # A rectangle at normalized r,c has reflected upper edge 2-r,
            # since the entire anchor occupies [0,3) on each normalized axis.
            rr = 2-r if d['panel']//2 != n['panel']//2 and ablation != "no_reflection" else r
            cc = 2-c if d['panel'] % 2 != n['panel'] % 2 and ablation != "no_reflection" else c
            if isinstance(ablation, tuple) and ablation[0] == 'absolute' and ablation[1] == n['panel']:
                rr = 2-r if ablation[2]//2 else r
                cc = 2-c if ablation[2] % 2 else c
            scale = d['unit'] if ablation == "no_scale" else u
            y0, x0 = t+rr*scale, l+cc*scale
            total += scale*scale
            for iy in range(max(0, y0), min(panel.shape[0], y0+scale)):
                for ix in range(max(0, x0), min(panel.shape[1], x0+scale)):
                    if ablation is None:
                        assert panel[iy, ix] != line, "graft/trail overlaps are excluded"
                    if panel[iy, ix] != line:
                        panel[iy, ix] = color
                    visible += 1
        panel[t:t+3*u, l:l+3*u] = n['body']
        panel[t+u:t+2*u, l+u:l+2*u] = d['value']
        records.append(dict(panel=n['panel'], donor=d['panel'], ratio=(u, d['unit']),
                            lost=total-visible, visible=visible,
                            flip=(d['panel']//2 != n['panel']//2, d['panel'] % 2 != n['panel'] % 2)))
    return output, (bg, line, nodes, records)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _predict(input_grid)[0]


def _features(pair):
    expected, (bg, line, nodes, records) = _predict(pair['input'])
    assert np.array_equal(expected, _grid(pair['output'])), "incorrect relational graft"
    assert expected.shape == np.asarray(pair['input']).shape
    assert records and any(r['visible'] for r in records), "a visible geometric transfer is required"
    assert np.array_equal(expected[np.asarray(pair['input']) == line], np.asarray(pair['input'])[np.asarray(pair['input']) == line])
    bodies = {n['body']: n for n in nodes}
    return dict(shape=expected.shape, nodes=len(nodes), survivors=len(records),
                active=sum(n['active'] for n in nodes),
                missing=sum(n['active'] and n['value'] not in bodies for n in nodes),
                interior=sum(not n['active'] and n['interior_contact'] and n['value'] in bodies for n in nodes),
                units=tuple(sorted({n['unit'] for n in nodes})),
                tails=tuple(len(n['tail']) for n in nodes),
                clipped=sum(r['lost'] > 0 for r in records),
                flips={r['flip'] for r in records}, ratios={r['ratio'] for r in records},
                roles=(bg, line),
                simultaneous=any(n['active'] and n['value'] in bodies and bodies[n['value']]['active']
                                 and bodies[n['value']]['value'] in bodies for n in nodes),
                failed_donor_reuse=any(n['active'] and n['value'] in bodies
                                      and bodies[n['value']]['active']
                                      and bodies[n['value']]['value'] not in bodies for n in nodes),
                motifs=tuple(tuple(sorted((r, c) for r, c in n['tail'])) for n in nodes))


def _absolute_shortcut_excluded(train):
    """Eliminate all 4**4 fixed per-quadrant flip tables by factorizing them.

    A table fits iff every panel has at least one flip fitting all its examples.
    Compare actual pixels, including clipping, rather than assuming asymmetry.
    """
    allowed = [set(range(4)) for _ in range(4)]
    for pair in train:
        _, _, _, _, nodes = _decode(pair['input'])
        bodies = {n['body'] for n in nodes}
        for n in nodes:
            if not n['active'] or n['value'] not in bodies:
                continue
            p = n['panel']
            for flip in list(allowed[p]):
                alternative = _predict(pair['input'], ('absolute', p, flip))[0]
                if not np.array_equal(alternative, pair['output']):
                    allowed[p].remove(flip)
            if not allowed[p]:
                return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            features[split].append(_features(pair))
    train = features['train']
    assert any(f['interior'] for f in train), "teach endpoint versus interior contact"
    assert any(f['missing'] for f in train), "teach failure of a missing donor"
    assert any(f['simultaneous'] for f in train), "teach original-state donor reuse"
    assert any(f['failed_donor_reuse'] for f in train), "teach reuse of a failed endpoint as a donor"
    assert any(f['clipped'] for f in train), "teach clipping before the larger test motifs"
    assert len({f['survivors'] for f in train}) > 1, "teach variable survivor counts"
    assert {(True, False), (False, True), (True, True)} <= set.union(*(f['flips'] for f in train))
    assert any(a < b for f in train for a, b in f['ratios'])
    assert any(a > b for f in train for a, b in f['ratios'])
    # Concrete counterexamples to omitting either parent and to nearby shortcuts.
    for alternative in ('centers_only', 'own_motif', 'all_tokens', 'no_reflection', 'no_scale',
                        'extra_hop', 'add_old', 'prune_failed_donors'):
        assert any(not np.array_equal(_predict(p['input'], alternative)[0], p['output'])
                   for p in puzzle['train']), "training does not exclude " + alternative
    assert _absolute_shortcut_excluded(puzzle['train']), "fixed quadrant orientations fit training"


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, modes in (('train', ['fork', 'chain']+['random']*(nt-2)), ('test', ['test']*ne)):
        for mode in modes:
            for _ in range(256):
                pair = _make_pair(rng, mode)
                if pair is None or _key(pair['input']) in seen:
                    continue
                try:
                    f = _features(pair)
                    if mode == 'fork':
                        assert f['interior'] and len(f['flips']) == 3
                        assert any(a != b for a, b in f['ratios'])
                    if mode == 'chain':
                        assert f['clipped'] and f['simultaneous'] and f['missing'] and f['failed_donor_reuse']
                        assert any(a < b for a, b in f['ratios'])
                        assert any(not np.array_equal(_predict(q['input'], 'add_old')[0], q['output'])
                                   for q in puzzle['train']+[pair])
                        assert _absolute_shortcut_excluded(puzzle['train']+[pair])
                except AssertionError:
                    continue
                puzzle[split].append(pair)
                seen.add(_key(pair['input']))
                break
            else:
                raise RuntimeError('bounded sampler exhausted for ' + mode)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for p in puzzle[split]:
            for key in ('input', 'output'):
                g = np.asarray(p[key])
                h.update(bytes(g.shape))
                h.update(g.astype(np.uint8).tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, bounded memory, semantic negatives and structural measurements."""
    digests, shapes, motifs, roles, ratios, flips = set(), set(), set(), set(), set(), set()
    survivors, clipping, node_counts, units = set(), set(), set(), set()
    pairs = within_shapes = within_motifs = 0
    max_train_tail = max_test_tail = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        if seed == 0:
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
        local = [_features(p) for s in ('train', 'test') for p in puzzle[s]]
        within_shapes += len({f['shape'] for f in local}) > 1
        within_motifs += len({f['motifs'] for f in local}) > 1
        for s in ('train', 'test'):
            for p in puzzle[s]:
                f = _features(p)
                pairs += 1
                shapes.add(f['shape'])
                motifs.update(f['motifs'])
                roles.add(f['roles'])
                ratios.update(f['ratios'])
                flips.update(f['flips'])
                survivors.add(f['survivors'])
                clipping.add(f['clipped'])
                node_counts.add(f['nodes'])
                units.update(f['units'])
                if s == 'train':
                    max_train_tail = max(max_train_tail, max(f['tails']))
                else:
                    max_test_tail = max(max_test_tail, max(f['tails']))
    assert len(digests) == 42 and pairs == 210
    assert len(shapes) >= 20 and len(motifs) >= 100 and len(roles) >= 40
    assert within_shapes == within_motifs == 42
    assert node_counts == {3, 4} and units == {1, 2}
    assert {(1, 1), (1, 2), (2, 1), (2, 2)} <= ratios
    assert {(True, False), (False, True), (True, True)} <= flips
    assert {1, 2, 3} <= survivors and {0, 1} <= clipping
    assert max_train_tail <= 6 and max_test_tail == 9
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(31+nt*7+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # An intended family member outside the sampler's 3/4-object envelope:
    # two active opposite-panel objects swap branches at opposite scales and
    # both become monochrome cores. The solver must not assume four objects.
    fixture = np.zeros((29, 29), dtype=np.int8)
    fixture[14, :] = fixture[:, 14] = 5
    expected = fixture.copy()
    first = {(-1, 0): 7, (0, 3): 8}
    second = {(3, 1): 8, (4, 1): 7}
    _paint(fixture[:14, :14], 4, 4, 1, 1, 2, first)
    _paint(fixture[15:, 15:], 2, 3, 2, 2, 1, second)
    _paint(expected[:14, :14], 4, 4, 1, 1, 1, _flip(second, 3, 0))
    _paint(expected[15:, 15:], 2, 3, 2, 2, 2, _flip(first, 0, 3))
    for q in ((7, 5), (8, 5), (9, 5), (19, 24), (19, 25), (19, 26)):
        fixture[q] = expected[q] = 5
    assert np.array_equal(solve(fixture), expected)
    # Reject corrupted output and conflicting geometry, rather than trusting labels.
    p = generate_puzzle(0, 2, 1)
    p['test'][0]['output'] = p['test'][0]['output'].copy()
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corruption accepted')
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(digests), grid_shapes=len(shapes),
                motif_footprints=len(motifs), role_pairs=len(roles),
                survivor_counts=sorted(survivors), clipped_counts=sorted(clipping),
                within_puzzle_shape_variation=within_shapes, within_puzzle_motif_variation=within_motifs)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
