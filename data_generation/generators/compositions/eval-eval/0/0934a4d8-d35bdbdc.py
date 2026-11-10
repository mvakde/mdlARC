"""Color-carrying mirrors: endpoint-selected relation keys govern a hidden crop.

No parent module, stored example, or construction state is used by solve().
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 170641049, 943885701, 17340561, 860554904, 837032493,
              415701356, 5120073, 854226966, 675151507, 680323643, 864325134,
              881500661, 879803543, 200538113, 630213971, 428015024,
              720628667, 654578856, 633980741, 237055616, 559050321,
              142434291, 651018675, 202737629, 174552840, 352362418,
              384585063, 176134287, 426765125, 287399703, 680137718,
              260054465, 948316003, 85430263, 517378628, 161638706,
              420563406, 325435001, 201338722, 469812625, 393124103)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(cells):
    todo, result = set(cells), []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, part = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in D4:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _support(center, kind, shape):
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (kind == 'square' or dr == 0 or dc == 0)
            and 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]}


def _frame(a):
    candidates = []
    for color in np.unique(a):
        cells = set(zip(*np.where(a == color)))
        r0, c0 = np.min(list(cells), axis=0)
        r1, c1 = np.max(list(cells), axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if min(r1 - r0, c1 - c0) < 6:
            continue
        ring = ({(r0, c) for c in range(c0, c1 + 1)} |
                {(r1, c) for c in range(c0, c1 + 1)} |
                {(r, c0) for r in range(r0, r1 + 1)} |
                {(r, c1) for r in range(r0, r1 + 1)})
        missing = ring - cells
        if len(missing) == 1 and cells == ring - missing:
            corner = next(iter(missing))
            if corner[0] in (r0, r1) and corner[1] in (c0, c1):
                candidates.append((int(color), (r0, r1, c0, c1), corner))
    assert len(candidates) == 1, 'Unique rectangular frame with one marked corner required'
    return candidates[0]


def _interpret(a):
    frame, box, anchor = _frame(a)
    r0, r1, c0, c1 = box
    line = int(a[anchor])
    colors, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(colors[np.argmax(counts)])
    assert len({bg, line, frame}) == 3
    outside = np.ones(a.shape, dtype=bool)
    outside[r0:r1 + 1, c0:c1 + 1] = False
    line_cells = set(zip(*np.where(outside & (a == line))))
    assert len(line_cells) >= 3
    graph = {p: {tuple(np.add(p, d)) for d in D4
                 if tuple(np.add(p, d)) in line_cells} for p in line_cells}
    assert all(len(adj) in (1, 2) for adj in graph.values()), 'Open nonbranching paths required'
    ends = {p for p, adj in graph.items() if len(adj) == 1}
    parts = _components(line_cells)
    assert all(len(part & ends) == 2 for part in parts)
    tokens = []
    for cells in _components(zip(*np.where(outside & (a != bg) & (a != line)))):
        possible = []
        for center in sorted(cells):
            for kind in ('square', 'cross'):
                if _support(center, kind, a.shape) != cells:
                    continue
                bodies = {int(a[p]) for p in cells if p != center}
                if len(bodies) == 1:
                    body = bodies.pop()
                    if body != int(a[center]):
                        possible.append(dict(center=center, kind=kind, cells=cells,
                                             body=body, value=int(a[center])))
        assert len(possible) == 1, 'Token body and center must be unambiguous'
        t = possible[0]
        r, c = t['center']
        t['ends'] = {(r + 2*dr, c + 2*dc) for dr, dc in D4} & ends
        assert len(t['ends']) <= 1
        t['eligible'] = bool(t['ends'])
        t['clipped'] = len(cells) < (9 if t['kind'] == 'square' else 5)
        tokens.append(t)
    assert len(tokens) >= 4 and any(t['eligible'] for t in tokens)
    assert all(sum(p in t['ends'] for t in tokens) <= 1 for p in ends)
    original = {}
    for t in tokens:
        assert t['body'] not in original or original[t['body']] == t['value'], 'Conflicting donors'
        original[t['body']] = t['value']
    transport = np.arange(10, dtype=np.int8)
    for i, t in enumerate(tokens):
        donors = [u for j, u in enumerate(tokens) if j != i and u['body'] == t['value']]
        t['donors'] = donors
        if t['eligible']:
            assert donors, 'Every selected key must have a donor'
            transport[t['body']] = donors[0]['value']
    assert len(set(map(int, transport))) == 10, 'Composed active map must be a permutation'
    identities = {x for t in tokens for x in (t['body'], t['value'])}
    assert not identities & {bg, line, frame}
    field = a[r0 + 1:r1, c0 + 1:c1].copy()
    absent = set(map(int, np.unique(field))) - identities
    assert len(absent) == 1, 'One private mask color required inside frame'
    mask = absent.pop()
    assert mask not in (bg, line, frame) and not np.any(a[outside] == mask)
    rr, cc = np.where(field == mask)
    bbox = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
    r, s, c, d = bbox
    assert len(rr) == (s - r)*(d - c), 'Mask must be a solid rectangle'
    assert field.size > len(rr) and set(map(int, np.unique(field[field != mask]))) <= identities
    h, w = field.shape
    u, v = np.indices(field.shape)
    if anchor[0] == r1:
        u = h - 1 - u
    if anchor[1] == c1:
        v = w - 1 - v
    power = (u >= (h + 1)//2).astype(int) + (v >= (w + 1)//2).astype(int)
    ids = np.minimum(u, h - 1 - u)*((w + 1)//2) + np.minimum(v, w - 1 - v)
    return dict(field=field, frame=frame, frame_box=box, anchor=anchor, background=bg,
                line=line, mask=mask, bbox=bbox, tokens=tokens, original=original,
                transport=transport, ids=ids, power=power, graph=graph, parts=parts)


def _recover(rec, transport=None, strict=True):
    """Undo color transport before pooling spatial reflection orbits."""
    table = rec['transport'] if transport is None else np.asarray(transport, dtype=np.int8)
    assert len(set(map(int, table))) == 10
    inverse = np.argsort(table)
    field, ids, power = rec['field'], rec['ids'], rec['power']
    visible = field != rec['mask']
    normalized = field.copy()
    for k in range(2):
        selected = visible & (power > k)
        normalized[selected] = inverse[normalized[selected]]
    values = np.full(int(ids.max()) + 1, -1, dtype=np.int8)
    for r, c in zip(*np.where(visible)):
        orbit, value = int(ids[r, c]), int(normalized[r, c])
        if values[orbit] >= 0:
            if strict:
                assert values[orbit] == value, 'Visible mirror witnesses disagree'
        else:
            values[orbit] = value
    assert np.all(values >= 0), 'A reflection orbit is entirely erased'
    full = values[ids].copy()
    # A through-cell axis fixes its central row/column. These colors must stay fixed.
    if strict:
        h, w = field.shape
        if h % 2:
            assert np.all(table[full[h//2]] == full[h//2]), 'Central row cannot change color'
        if w % 2:
            assert np.all(table[full[:, w//2]] == full[:, w//2]), 'Central column cannot change color'
    for k in range(2):
        selected = power > k
        full[selected] = table[full[selected]]
    if strict:
        assert np.array_equal(full[visible], field[visible])
    r, s, c, d = rec['bbox']
    return full[r:s, c:d].copy(), full


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the target rectangle solely from the visible input."""
    return _recover(_interpret(_grid(input_grid)))[0]


def _features(a):
    rec = _interpret(a)
    out, full = _recover(rec)
    field, mask = rec['field'], rec['mask']
    r, s, c, d = rec['bbox']
    active = {t['body'] for t in rec['tokens'] if t['eligible']}
    changed = {x for x in active if rec['transport'][x] != x}
    inactive = set(rec['original']) - active
    # Raw reflection copying omits B; it uses the nearest intact reflected cell.
    literal = np.empty(out.shape, dtype=np.int8)
    for y in range(r, s):
        for x in range(c, d):
            mates = list(zip(*np.where((rec['ids'] == rec['ids'][y, x]) & (field != mask))))
            q = min(mates, key=lambda p: (abs(y-p[0]) + abs(x-p[1]), p))
            literal[y-r, x-c] = field[q]
    all_keys = np.arange(10, dtype=np.int8)
    for body, value in rec['original'].items():
        if value in rec['original']:
            all_keys[body] = rec['original'][value]
    all_out, _ = _recover(rec, all_keys, strict=False)
    interior_contact = any(not t['eligible'] and t['donors'] and
                           any((p[0]+dr, p[1]+dc) in rec['graph']
                               for p in t['cells'] for dr, dc in D4)
                           for t in rec['tokens'])
    h, w = field.shape
    return dict(shape=a.shape, field_shape=field.shape, output_shape=out.shape,
                active=len(active), tokens=len(rec['tokens']), lines=len(rec['parts']),
                kinds=tuple(sorted({t['kind'] for t in rec['tokens']})),
                clipped=sum(t['clipped'] for t in rec['tokens']),
                mask=mask, anchor=(rec['anchor'][0] == rec['frame_box'][1],
                                  rec['anchor'][1] == rec['frame_box'][3]),
                transport_order=next(k for k in range(1, 7)
                                     if np.array_equal(np.linalg.matrix_power(np.eye(10, dtype=int)[rec['transport']], k), np.eye(10, dtype=int))),
                seam=(r < h/2 < s or c < w/2 < d),
                edge=(r == 0 or s == h or c == 0 or d == w),
                changed_hidden=bool(set(map(int, out.ravel())) & changed),
                inactive_hidden=bool(set(map(int, out.ravel())) & inactive),
                literal_diff=not np.array_equal(literal, out),
                all_keys_diff=not np.array_equal(all_out, out),
                orientation_diff=not np.array_equal(out, out[::-1, ::-1]),
                interior_contact=interior_contact,
                bends=sum(len(adj) == 2 and
                          tuple(np.add(*sorted(adj))) != tuple(2*x for x in p)
                          for p, adj in rec['graph'].items()))


def _route(rng, shape, blocked, start, finish):
    if start == finish or start in blocked or finish in blocked:
        return None
    costs = rng.uniform(1., 1.5, shape)
    distances, previous = {start: 0.}, {}
    queue = [(0., start)]
    while queue:
        distance, p = heapq.heappop(queue)
        if distance != distances[p]:
            continue
        if p == finish:
            path = [p]
            while p != start:
                p = previous[p]
                path.append(p)
            return path[::-1] if len(path) >= 3 else None
        for dr, dc in D4:
            q = p[0]+dr, p[1]+dc
            if not (0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]) or q in blocked:
                continue
            new = distance + float(costs[q])
            if new < distances.get(q, float('inf')):
                distances[q], previous[q] = new, p
                heapq.heappush(queue, (new, q))
    return None


def _layout(rng, shape, frame_box, kinds, active, force_interior, clip_probability):
    r0, r1, c0, c1 = frame_box
    blocked_frame = {(r, c) for r in range(r0-1, r1+2) for c in range(c0-1, c1+2)}
    centers, supports = [], []
    # The last duplicate donor is placed after routing for the contrast witness.
    count = len(kinds) - int(force_interior)
    for i in range(count):
        clipped = rng.random() < clip_probability
        candidates = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                      if ((r in (0, shape[0]-1)) + (c in (0, shape[1]-1)) == int(clipped))
                      and not (_support((r, c), kinds[i], shape) & blocked_frame)
                      and all(max(abs(r-y), abs(c-x)) >= 4 for y, x in centers)]
        if not candidates:
            return None
        p = candidates[int(rng.integers(len(candidates)))]
        centers.append(p)
        supports.append(_support(p, kinds[i], shape))
    order = list(map(int, rng.permutation(active)))
    groups = [order[:2]] + ([order[2:]] if active == 3 else [])
    line_cells = set()
    for group in groups:
        routed = False
        for _ in range(16):
            stubs = []
            for i in group:
                r, c = centers[i]
                choices = [(r+2*dr, c+2*dc) for dr, dc in D4
                           if 0 <= r+2*dr < shape[0] and 0 <= c+2*dc < shape[1]
                           and (r+2*dr, c+2*dc) not in blocked_frame]
                if not choices:
                    break
                stubs.append(choices[int(rng.integers(len(choices)))])
            if len(stubs) != len(group):
                continue
            blocked = set(blocked_frame)
            for cells in supports:
                blocked.update((r+dr, c+dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            for r, c in line_cells:
                blocked.add((r, c))
                blocked.update((r+dr, c+dc) for dr, dc in D4)
            if len(stubs) == 1:
                choices = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                           if (r, c) not in blocked and
                           all(max(abs(r-y), abs(c-x)) >= 4 for y, x in centers)]
                if not choices:
                    continue
                stubs.append(choices[int(rng.integers(len(choices)))])
            blocked.difference_update(stubs)
            path = _route(rng, shape, blocked, *stubs)
            if path is not None:
                line_cells.update(path)
                routed = True
                break
        if not routed:
            return None
    if force_interior:
        graph = {p: {tuple(np.add(p, d)) for d in D4 if tuple(np.add(p, d)) in line_cells}
                 for p in line_cells}
        ends = {p for p, adj in graph.items() if len(adj) == 1}
        candidates = sorted({(r+2*dr, c+2*dc) for (r, c), adj in graph.items() if len(adj) == 2
                             for dr, dc in D4})
        candidates = [p for p in candidates if 1 <= p[0] < shape[0]-1 and 1 <= p[1] < shape[1]-1
                      and not (_support(p, kinds[-1], shape) & (blocked_frame | line_cells))
                      and all(max(abs(p[0]-r), abs(p[1]-c)) >= 4 for r, c in centers)
                      and not any((p[0]+2*dr, p[1]+2*dc) in ends for dr, dc in D4)]
        if not candidates:
            return None
        p = candidates[int(rng.integers(len(candidates)))]
        centers.append(p)
        supports.append(_support(p, kinds[-1], shape))
    return centers, supports, line_cells


def _pattern(rng, shape, palette, style):
    if style == 'patches':
        small = rng.choice(palette, ((shape[0]+1)//2, (shape[1]+1)//2))
        return np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)[:shape[0], :shape[1]].copy()
    if style == 'biased':
        weights = rng.dirichlet(np.ones(len(palette))) + .15
        return rng.choice(palette, shape, p=weights/weights.sum())
    return rng.choice(palette, shape)


def _make_pair(rng, params, mode):
    active = 3 if mode == 'witness' else (2 if mode == 'seam' else int(rng.integers(2, 4)))
    roles = np.asarray(rng.permutation(10) if params['vary_roles'] else params['roles'], dtype=np.int8)
    bg, frame, line, mask = map(int, roles[:4])
    sources, middle = roles[4:4+active], roles[4+active:4+2*active]
    if active == 2:
        perm = np.array([1, 0])
    elif mode == 'witness' or rng.random() < .6:
        perm = np.roll(np.arange(3), int(rng.choice([-1, 1])))
    else:
        perm = np.arange(3)
        i, j = rng.choice(3, 2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
    relations = list(zip(sources, middle)) + list(zip(middle, sources[perm]))
    extra = 1 if mode == 'witness' else int(rng.integers(0, 3))
    relations += [relations[active + int(rng.integers(active))] for _ in range(extra)]
    table = np.arange(10, dtype=np.int8)
    table[sources] = sources[perm]
    palette = np.concatenate((sources, middle))
    for _ in range(180):
        if mode == 'witness':
            fh, fw = int(rng.choice([9, 11, 13])), int(rng.integers(9, 14))
        else:
            fh, fw = int(rng.integers(7, 15)), int(rng.integers(7, 15))
        shape = (int(rng.integers(max(22, fh+10), 31)), int(rng.integers(max(22, fw+10), 31)))
        r0 = int(rng.integers(4, shape[0]-fh-5))
        c0 = int(rng.integers(4, shape[1]-fw-5))
        frame_box = r0, r0+fh+1, c0, c0+fw+1
        if params['morphology'] == 'mixed':
            kinds = [str(v) for v in rng.choice(['square', 'cross'], len(relations))]
            if mode == 'witness':
                kinds[:2] = ['square', 'cross']
        else:
            kinds = [params['morphology']]*len(relations)
        layout = _layout(rng, shape, frame_box, kinds, active, mode == 'witness', params['clip_probability'])
        if layout is None:
            continue
        centers, supports, line_cells = layout
        h0, w0 = (fh+1)//2, (fw+1)//2
        style = str(rng.choice(['random', 'patches', 'biased'])) if params['vary_style'] else params['style']
        base = _pattern(rng, (h0, w0), palette, style).astype(np.int8)
        if fh % 2:
            base[-1] = rng.choice(middle, w0)
        if fw % 2:
            base[:, -1] = rng.choice(middle, h0)
        u, v = np.indices((fh, fw))
        power = (u >= h0).astype(int) + (v >= w0).astype(int)
        full = base[np.minimum(u, fh-1-u), np.minimum(v, fw-1-v)].copy()
        for k in range(2):
            selected = power > k
            full[selected] = table[full[selected]]
        flip_r, flip_c = bool(rng.integers(2)), bool(rng.integers(2))
        if flip_r:
            full = full[::-1].copy()
        if flip_c:
            full = full[:, ::-1].copy()
        canvas = np.full(shape, bg, dtype=np.int8)
        r1, c1 = r0+fh+1, c0+fw+1
        canvas[r0, c0:c1+1] = canvas[r1, c0:c1+1] = frame
        canvas[r0:r1+1, c0] = canvas[r0:r1+1, c1] = frame
        canvas[r1 if flip_r else r0, c1 if flip_c else c0] = line
        canvas[r0+1:r1, c0+1:c1] = full
        for p in line_cells:
            canvas[p] = line
        for (body, value), center, cells in zip(relations, centers, supports):
            for p in cells:
                canvas[p] = body
            canvas[center] = value
        for _ in range(45):
            if mode == 'witness':
                hh, ww = int(rng.integers(2, 4)), int(rng.integers(2, 4))
                r = int(rng.integers(1, fh//2-hh+1))
                c = int(rng.integers(1, fw//2-ww+1))
                if rng.random() < .5:
                    r = fh-r-hh
                if rng.random() < .5:
                    c = fw-c-ww
            else:
                across_rows = bool(rng.integers(2))
                cross = int(rng.integers(3, min(9, (fh if across_rows else fw)-2)+1))
                narrow = int(rng.integers(1, min(4, (fw if across_rows else fh)//2)))
                hh, ww = (cross, narrow) if across_rows else (narrow, cross)
                r = int(np.clip(fh//2-hh//2+int(rng.integers(-1, 2)), 0, fh-hh))
                c = int(np.clip(fw//2-ww//2+int(rng.integers(-1, 2)), 0, fw-ww))
                if across_rows:
                    c = int(rng.choice([0, fw-ww])) if mode == 'test' else int(rng.choice([1, fw-ww-1]))
                else:
                    r = int(rng.choice([0, fh-hh])) if mode == 'test' else int(rng.choice([1, fh-hh-1]))
            out = full[r:r+hh, c:c+ww].copy()
            inp = canvas.copy()
            inp[r0+1+r:r0+1+r+hh, c0+1+c:c0+1+c+ww] = mask
            try:
                predicted = solve(inp)
                assert np.array_equal(predicted, out)
                f = _features(inp)
                assert f['changed_hidden'] and f['inactive_hidden'] and f['literal_diff'] and f['all_keys_diff']
                if mode == 'witness':
                    assert f['interior_contact'] and f['orientation_diff'] and f['transport_order'] == 3
                if mode in ('seam', 'test'):
                    assert f['seam']
                if mode == 'test':
                    assert f['edge']
                else:
                    assert not f['edge']
            except AssertionError:
                continue
            return dict(input=inp, output=out)
    raise RuntimeError('Bounded sampler could not construct an unambiguous example')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) if seed >= 0 else int(seed) % (1 << 64))
    params = dict(roles=rng.permutation(10), vary_roles=bool(rng.random() < .65),
                  morphology=str(rng.choice(['square', 'cross', 'mixed'], p=[.3, .2, .5])),
                  clip_probability=float(rng.choice([0., .15, .4])),
                  vary_style=bool(rng.random() < .6), style=str(rng.choice(['random', 'patches', 'biased'])))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        modes = ['witness', 'seam'] + ['random']*(count-2) if split == 'train' else ['test']*count
        for mode in modes:
            for _ in range(8):
                pair = _make_pair(rng, params, mode)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not produce distinct inputs')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    features, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen, 'Duplicate input'
            seen.add(_key(a))
            rec = _interpret(a)
            predicted, full = _recover(rec)
            assert np.array_equal(predicted, b), 'Incorrect target-oriented crop'
            assert rec['mask'] not in b and b.shape != a.shape
            assert set(map(int, np.unique(full))) <= set(rec['original'])
            f = _features(a)
            assert f['changed_hidden'] and f['inactive_hidden']
            assert f['literal_diff'] and f['all_keys_diff'], 'Both inherited concepts must affect the crop'
            features[split].append(f)
    tr, te = features['train'], features['test']
    assert any(f['interior_contact'] and f['transport_order'] == 3 and f['orientation_diff'] for f in tr)
    assert any(f['seam'] and not f['edge'] for f in tr)
    assert any(not f['seam'] and not f['edge'] for f in tr)
    assert all(f['seam'] and f['edge'] for f in te), 'Generated tests combine seam overlap and a new edge'


def self_test():
    """Bounded semantic fixtures, 42-seed diversity, and every explicit count."""
    signatures, rows = set(), []
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        signatures.add(tuple((_key(q['input']), _key(q['output'])) for s in p for q in p[s]))
        rows.extend(_features(q['input']) for s in p for q in p[s])
    assert len(signatures) == 42
    diversity = {name: len({f[name] for f in rows}) for name in
                 ('shape', 'field_shape', 'output_shape', 'tokens', 'kinds', 'mask', 'anchor', 'clipped')}
    assert diversity['shape'] >= 35 and diversity['field_shape'] >= 35 and diversity['output_shape'] >= 15
    assert diversity['tokens'] >= 4 and diversity['kinds'] == 3 and diversity['mask'] == 10
    assert diversity['anchor'] == 4 and diversity['clipped'] >= 2
    assert {f['transport_order'] for f in rows} == {2, 3}
    assert {f['active'] for f in rows} == {2, 3}
    assert any(f['lines'] == 2 for f in rows) and any(f['bends'] >= 3 for f in rows)
    assert any(h % 2 and w % 2 for h, w in (f['field_shape'] for f in rows))
    assert any(not (h % 2 or w % 2) for h, w in (f['field_shape'] for f in rows))
    assert any(min(f['output_shape']) == 1 for f in rows)
    p = generate_puzzle(0)
    again = generate_puzzle(0)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], again[s]) for k in x)
    # Anchors make every D4 orientation and every color bijection equivalent.
    for turn in range(4):
        transformed = {s: [{k: np.rot90(v, turn).copy() for k, v in q.items()} for q in p[s]] for s in p}
        validate_generated_puzzle(transformed)
    for s in p:
        for q in p[s]:
            for flip in (np.fliplr, np.flipud):
                assert np.array_equal(solve(flip(q['input'])), flip(q['output']))
            perm = np.array([8, 4, 0, 9, 2, 7, 1, 6, 5, 3], dtype=np.int8)
            assert np.array_equal(solve(perm[q['input']]), perm[q['output']])
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            generate_puzzle(2026+train*10+test, train, test)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    direct_pair = next(q for q in p['train'] if _features(q['input'])['interior_contact'])
    rec = _interpret(direct_pair['input'])
    # A rectangular overwrite crossing both axes erases a central orbit.
    damaged = direct_pair['input'].copy()
    r0, r1, c0, c1 = rec['frame_box']
    interior = damaged[r0+1:r1, c0+1:c1]
    _, complete = _recover(rec)
    interior[:] = complete
    h, w = interior.shape
    interior[(h-1)//2:h//2+1, (w-1)//2:w//2+1] = rec['mask']
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('Unrecoverable evidence accepted')
    # Corrupt one visible member of a multiply witnessed spatial orbit.
    damaged = direct_pair['input'].copy()
    interior = damaged[r0+1:r1, c0+1:c1]
    visible_corners = list(zip(*np.where((rec['ids'] == 0) & (interior != rec['mask']))))
    q = visible_corners[0]
    interior[q] = next(x for x in rec['original'] if x != int(interior[q]))
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('Conflicting mirror evidence accepted')
    # Repeated donor bodies have no nearest-donor tie breaker.
    damaged = direct_pair['input'].copy()
    donor = next(t for t in rec['tokens']
                 if sum(u['body'] == t['body'] for u in rec['tokens']) > 1)
    damaged[donor['center']] = next(x for x in rec['original'] if x not in (donor['body'], donor['value']))
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('Conflicting donors accepted')
    bad = {s: [dict(input=q['input'].copy(), output=q['output'].copy()) for q in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Incorrect label accepted')
    return dict(seeds=42, pairs=len(rows), diversity=diversity,
                literal_copy_failures=sum(f['literal_diff'] for f in rows),
                ignore_endpoint_failures=sum(f['all_keys_diff'] for f in rows))


if __name__ == '__main__':
    report = self_test()
    Path('scratch').mkdir(exist_ok=True)
    Path('scratch/self_test.json').write_text(json.dumps(report, separators=(',', ':')) + '\n')
    print(json.dumps(report, separators=(',', ':')))
