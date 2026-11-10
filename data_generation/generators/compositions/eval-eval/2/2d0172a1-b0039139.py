"""ARC family: scoped motif factories.

Each marked contour expands its direct payload, using only its own tokens,
axis and palette. The outer contour is a scope, not a copy counter itself.
No parent module, stored puzzle or generation state is used by solve().
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)
             if (y or x) and (diagonal or abs(y) + abs(x) == 1)]
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        cells, stack = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(cells)
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, dtype=bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        r, c = stack.pop()
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < h and 0 <= x < w and not padded[y, x] and not outside[y, x]:
                outside[y, x] = True
                stack.append((y, x))
    return (~padded & ~outside)[1:-1, 1:-1]


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _crop(a, mask):
    points = np.argwhere(mask)
    lo, hi = points.min(0), points.max(0) + 1
    return a[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _parse(input_grid):
    """Recover actual holes and immediate ownership, never bounding-box containment."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(np.all(g[0] == bg) and np.all(g[-1] == bg)
             and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg),
             "input needs a visible background border")
    components = _components(g != bg)
    _require(components and len(components) == len(_components(g != bg, True)),
             "distinct objects may not touch diagonally")
    records, contour_inks = [], set()
    for cells in components:
        coords = np.array(cells, dtype=int)
        mask = np.zeros(g.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        colors, amounts = np.unique(g[mask], return_counts=True)
        rec = dict(mask=mask, cells=coords, lo=coords.min(0), hi=coords.max(0),
                   hole=_holes(mask), area=len(cells))
        if len(colors) == 1:
            rec.update(kind="pure", color=int(colors[0]))
        else:
            _require(len(colors) == 3 and sorted(amounts.tolist())[:2] == [1, 1]
                     and int(amounts.max()) >= 8, "bad contour header")
            ink = int(colors[np.argmax(amounts)])
            header = coords[g[coords[:, 0], coords[:, 1]] != ink]
            _require(int(np.abs(header[0] - header[1]).sum()) == 1,
                     "swatches must be two adjacent cells")
            axis = 1 if header[0, 0] == header[1, 0] else 0
            header = sorted(map(tuple, header.tolist()), key=lambda p: p[axis])
            low, high = header
            before, after = list(low), list(high)
            before[axis] -= 1
            after[axis] += 1
            _require(all(0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]
                         and g[tuple(p)] == ink for p in (before, after)),
                     "header needs contour ink at both ends")
            _require(len(_components(rec['hole'])) == 1, "contour must enclose one region")
            exterior = ~mask & ~rec['hole']
            for r, c in header:
                _require(any(0 <= y < g.shape[0] and 0 <= x < g.shape[1]
                             and exterior[y, x] for y, x in
                             ((r-1, c), (r+1, c), (r, c-1), (r, c+1))),
                         "header must sit on an exterior edge")
            rec.update(kind="contour", ink=ink, axis=axis,
                       primary=int(g[low]), secondary=int(g[high]))
            contour_inks.add(ink)
        records.append(rec)
    _require(len(contour_inks) == 1, "all contours share one structural ink")
    ink = next(iter(contour_inks))
    stencil_inks = set()
    for rec in records:
        if rec['kind'] == 'pure':
            rec['kind'] = 'token' if rec['color'] == ink else 'stencil'
            if rec['kind'] == 'token':
                _require(not rec['hole'].any(), "tokens must be filled, hole-free objects")
            else:
                stencil_inks.add(rec['color'])
    _require(len(stencil_inks) == 1, "all leaf stencils share one separate ink")
    stencil_ink = next(iter(stencil_inks))
    for rec in records:
        if rec['kind'] == 'contour':
            _require(len({bg, ink, stencil_ink, rec['primary'], rec['secondary']}) == 5,
                     "structural and swatch roles must be distinct in each scope")
    parents, children = [], {i: [] for i in range(len(records))}
    for i, child in enumerate(records):
        owners = []
        for j, candidate in enumerate(records):
            if j == i or candidate['kind'] != 'contour':
                continue
            inside = candidate['hole'][child['cells'][:, 0], child['cells'][:, 1]]
            _require(inside.all() or not inside.any(), "objects cross a contour")
            if inside.all():
                owners.append(j)
        parent = min(owners, key=lambda j: int(records[j]['hole'].sum())) if owners else None
        parents.append(parent)
        if parent is not None:
            children[parent].append(i)
    roots = [i for i, p in enumerate(parents) if p is None]
    _require(len(roots) == 1 and records[roots[0]]['kind'] == 'contour',
             "one outer contour must own the entire scene")

    def build(i):
        rec = records[i]
        _require(rec['kind'] == 'contour', "payload error")
        direct = children[i]
        tokens = [j for j in direct if records[j]['kind'] == 'token']
        payload = [j for j in direct if records[j]['kind'] != 'token']
        _require(tokens and payload, "every scope needs tokens and a payload")
        stencils = [j for j in payload if records[j]['kind'] == 'stencil']
        _require(not stencils or len(stencils) == len(payload) == 1,
                 "a scope contains one stencil or a sequence of child scopes")
        axis, cross = rec['axis'], 1 - rec['axis']
        ordered = sorted(payload, key=lambda j: int(records[j]['lo'][axis]))
        for a, b in zip(ordered, ordered[1:]):
            _require(records[b]['lo'][axis] - records[a]['hi'][axis] >= 2,
                     "payload order needs a blank projected gap")
        centers = [int(records[j]['lo'][cross] + records[j]['hi'][cross]) for j in payload]
        _require(max(centers) - min(centers) <= 1, "payloads must share a clear centerline")
        node = dict(axis=axis, count=len(tokens), primary=rec['primary'],
                    secondary=rec['secondary'], token_areas=[records[j]['area'] for j in tokens])
        if stencils:
            node['stencil'] = _crop(records[stencils[0]]['mask'], records[stencils[0]]['mask'])
            node['children'] = []
        else:
            node['stencil'] = None
            node['children'] = [build(j) for j in ordered]
        return node

    return dict(root=build(roots[0]), records=records, parents=parents,
                background=bg, ink=ink, stencil_ink=stencil_ink)


def _join(parts, axis, fill, gaps=1):
    cross = 1 - axis
    shape = [0, 0]
    shape[axis] = sum(p.shape[axis] for p in parts) + gaps * (len(parts)-1)
    shape[cross] = max(p.shape[cross] for p in parts)
    out = np.full(shape, fill, dtype=np.int8)
    position = 0
    for part in parts:
        start = [0, 0]
        start[axis] = position
        start[cross] = (shape[cross] - part.shape[cross]) // 2
        r, c = start
        out[r:r+part.shape[0], c:c+part.shape[1]] = part
        position += part.shape[axis] + gaps
    return out


def _evaluate(node, ablation=None):
    p, s, axis = node['primary'], node['secondary'], node['axis']
    if node['stencil'] is not None:
        word = np.where(node['stencil'], p, s).astype(np.int8)
    else:
        word = _join([_evaluate(c, ablation) for c in node['children']], axis, s)
    n = node['count']
    if ablation == 'one':
        n = 1
    elif ablation == 'descendants':
        n = sum(c['count'] for c in _walk(node))
    repeated = _join([word] * n, axis, s)
    out = np.pad(repeated, 2, constant_values=s)
    out[0] = out[-1] = p
    out[:, 0] = out[:, -1] = p
    return out.astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible input pixels alone; no learned or seed-specific state."""
    out = _evaluate(_parse(input_grid)['root'])
    _require(all(d <= 30 for d in out.shape), "expanded output exceeds the ARC budget")
    return out


def _walk(node):
    yield node
    for child in node['children']:
        yield from _walk(child)


def _depth(node):
    return 1 + max((_depth(c) for c in node['children']), default=0)


def _shape(node):
    if node['stencil'] is not None:
        size = list(node['stencil'].shape)
    else:
        dims, axis = [_shape(c) for c in node['children']], node['axis']
        size = [max(d[0] for d in dims), max(d[1] for d in dims)]
        size[axis] = sum(d[axis] for d in dims) + len(dims) - 1
    size[node['axis']] = node['count'] * (size[node['axis']] + 1) - 1
    return tuple(d + 4 for d in size)


def _audit_output(node, out):
    """Check local frame, moat, centering, unchanged copies and separators by slices."""
    assert out.shape == _shape(node)
    p, s, axis = node['primary'], node['secondary'], node['axis']
    assert np.all(out[0] == p) and np.all(out[-1] == p)
    assert np.all(out[:, 0] == p) and np.all(out[:, -1] == p)
    assert np.all(out[1, 1:-1] == s) and np.all(out[-2, 1:-1] == s)
    assert np.all(out[1:-1, 1] == s) and np.all(out[1:-1, -2] == s)
    content = out[2:-2, 2:-2]
    parts = ([node['stencil'].shape] if node['stencil'] is not None
             else [_shape(c) for c in node['children']])
    word_length = sum(d[axis] for d in parts) + len(parts) - 1
    cross = 1 - axis
    for k in range(node['count']):
        position = k * (word_length + 1)
        for i, dims in enumerate(parts):
            start = [0, 0]
            start[axis] = position
            start[cross] = (content.shape[cross] - dims[cross]) // 2
            r, c = start
            tile = content[r:r+dims[0], c:c+dims[1]]
            if node['stencil'] is not None:
                assert np.array_equal(tile, np.where(node['stencil'], p, s))
            else:
                _audit_output(node['children'][i], tile)
            # The unoccupied cross-axis margin is this scope's secondary color.
            block = np.take(content, range(position, position+dims[axis]), axis=axis)
            before = np.take(block, range(start[cross]), axis=cross)
            after = np.take(block, range(start[cross]+dims[cross], block.shape[cross]), axis=cross)
            assert np.all(before == s) and np.all(after == s)
            position += dims[axis]
            if i < len(parts)-1 or k < node['count']-1:
                assert np.all(np.take(content, position, axis=axis) == s)
            position += 1


def _mask(rng, h, w, witness=False):
    if witness:
        a = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
        return np.rot90(a, int(rng.integers(4))).copy()
    a = np.ones((h, w), dtype=bool)
    target = int(rng.integers(max(h, w), h*w+1))
    for flat in rng.permutation(h*w):
        if a.sum() <= target:
            break
        r, c = divmod(int(flat), w)
        a[r, c] = False
        if not a.any() or len(_components(a)) != 1 or _crop(a, a).shape != (h, w):
            a[r, c] = True
    return a


def _new_node(rng, palette, count, axis, children=None, stencil=None, witness=False):
    choices = [c for c in range(10) if c not in palette]
    p, s = map(int, rng.choice(choices, 2, replace=False))
    return dict(count=int(count), axis=int(axis), primary=p, secondary=s,
                children=children or [], stencil=stencil, equal_area=witness)


def _scene(rng, palette, mode):
    axis = int(rng.integers(2))

    def leaf(n=1, a=None, tiny=False, witness=False):
        a = int(rng.integers(2)) if a is None else a
        if witness:
            st = _mask(rng, 2, 3, True)
        else:
            cap = 2 if tiny else 5
            h, w = map(int, rng.integers(1, cap+1, 2))
            st = _mask(rng, h, w)
        return _new_node(rng, palette, n, a, stencil=st, witness=witness)

    if mode == 'simple_witness':
        node = leaf(int(rng.choice([2, 3])), axis, witness=True)
        # Prefer a narrow axial stencil; the input loop will be wider across it.
        if node['stencil'].shape[axis] < node['stencil'].shape[1-axis]:
            node['stencil'] = node['stencil'].T.copy()
        return node
    if mode == 'nested_witness':
        children = [leaf(int(rng.choice([2, 3])), 1-axis, witness=True) for _ in range(2)]
        # Cross-axis size 2 keeps this witness feasible even at minimal counts.
        for child in children:
            if child['stencil'].shape[axis] != 2:
                child['stencil'] = child['stencil'].T.copy()
        node = _new_node(rng, palette, 1, axis, children, witness=True)
        used = {(node['primary'], node['secondary'])}
        choices = [(p, s) for p in range(10) for s in range(10)
                   if p != s and p not in palette and s not in palette]
        for child in children:
            if (child['primary'], child['secondary']) in used:
                available = [pair for pair in choices if pair not in used]
                child['primary'], child['secondary'] = available[int(rng.integers(len(available)))]
            used.add((child['primary'], child['secondary']))
        return node
    if mode == 'deep':
        node = leaf(1, tiny=True)
        node = _new_node(rng, palette, 1, int(rng.integers(2)), [node])
        node = _new_node(rng, palette, 1, axis, [node])
        list(_walk(node))[int(rng.integers(3))]['count'] = 2
        return node
    if mode == 'branch':
        children = [leaf(1, tiny=True) for _ in range(3)]
        if rng.random() < .5:
            children[int(rng.integers(3))]['count'] = 2
        return _new_node(rng, palette, 1, axis, children)
    if mode == 'word':
        children = [leaf(1, tiny=True), leaf(1, tiny=True)]
        if np.array_equal(_evaluate(children[0]), _evaluate(children[1])):
            children[1]['primary'], children[1]['secondary'] = \
                children[1]['secondary'], children[1]['primary']
        return _new_node(rng, palette, 2, axis, children)
    if mode == 'count':
        node = leaf(4, tiny=True)
        return _new_node(rng, palette, 1, axis, [node]) if rng.random() < .6 else node
    if rng.random() < .4:
        return leaf(int(rng.integers(1, 4)))
    k = int(rng.integers(1, 3))
    children = [leaf(int(rng.integers(1, 4)), tiny=k == 2) for _ in range(k)]
    # Multi-child repetition is reserved for the word-composition test shift.
    return _new_node(rng, palette, int(rng.integers(1, 4)) if k == 1 else 1,
                     axis, children)


def _token(rng, area=None, compact=False):
    if area is None:
        h, w = map(int, rng.integers(1, 3 if compact else 5, 2))
        a = _mask(rng, h, w)
        # Hollow stencils are valid; tokens specifically may not have holes.
        a |= _holes(a)
        return a
    # Exact area 6 per scope in the witnesses, regardless of count 1, 2 or 3.
    patterns = {2: [[1, 1]], 3: [[1, 1], [1, 0]], 6: [[1, 1, 1], [1, 1, 1]]}
    a = np.array(patterns[area], dtype=bool)
    return np.rot90(a, int(rng.integers(4))).copy()


def _outline(rng, h, w, padding, axis, compact):
    """Closed orthogonal path in four safe corridors, with a straight header edge."""
    side = int(rng.choice([0, 1] if axis == 1 else [2, 3]))
    lengths = [w, w, h, h]
    profiles = []
    rectangular = rng.random() < .16
    for edge, (length, pad) in enumerate(zip(lengths, padding)):
        profile = np.zeros(length, dtype=int)
        cap = max(0, pad-2)
        if edge != side and cap and not rectangular and length >= 6:
            for _ in range(int(rng.integers(1, 4))):
                lo = int(rng.integers(2, length-3))
                hi = int(rng.integers(lo+1, length-1))
                profile[lo:hi] = int(rng.integers(1, cap+1))
        profiles.append(profile)
    top, bottom, left, right = profiles
    path = [(int(top[x]), x) for x in range(w)]
    path += [(y, w-1-int(right[y])) for y in range(h)]
    path += [(h-1-int(bottom[x]), x) for x in range(w-1, -1, -1)]
    path += [(y, int(left[y])) for y in range(h-1, -1, -1)]
    path.append(path[0])
    mask = np.zeros((h, w), dtype=bool)
    for (r, c), (y, x) in zip(path, path[1:]):
        mask[r, min(c, x):max(c, x)+1] = True
        mask[min(r, y):max(r, y)+1, x] = True
    # Tight staircase turns can surround a tiny incidental pocket. Fill those
    # pockets with contour ink; the large safe interior is the only scope.
    pockets = _components(_holes(mask))
    if len(pockets) > 1:
        main = max(range(len(pockets)), key=lambda i: len(pockets[i]))
        for i, pocket in enumerate(pockets):
            if i != main:
                for point in pocket:
                    mask[point] = True
    length = w if axis == 1 else h
    offset = int(rng.integers(1, length-2))
    if side in (0, 1):
        r = 0 if side == 0 else h-1
        header = [(r, offset), (r, offset+1)]
    else:
        c = 0 if side == 2 else w-1
        header = [(offset, c), (offset+1, c)]
    return mask, header


def _draw(node, rng, palette, compact=False, force_cross_long=False):
    bg, ink, stencil_ink = palette
    axis, cross = node['axis'], 1-node['axis']
    if node['stencil'] is not None:
        payload = np.where(node['stencil'], stencil_ink, bg).astype(np.int8)
    else:
        payload = _join([_draw(c, rng, palette, compact) for c in node['children']],
                        axis, bg, 1 if compact else int(rng.integers(1, 4)))
    n = node['count']
    tokens = [_token(rng, 6//n if node.get('equal_area') else None, compact) for _ in range(n)]
    tokens = [np.where(a, ink, bg).astype(np.int8) for a in tokens]
    rack_axis = axis if compact or rng.random() < .7 else cross
    rack = _join(tokens, rack_axis, bg, 1 if compact else int(rng.integers(1, 4)))
    # Tokens may be on any side. Compact retries use a space-efficient rail.
    rail_axis = cross if compact or rng.random() < .7 else axis
    pieces = [payload, rack] if rng.random() < .5 else [rack, payload]
    content = _join(pieces, rail_axis, bg, 1 if compact else int(rng.integers(1, 4)))
    padding = [2]*4 if compact else list(map(int, rng.integers(2, 5, 4)))
    h = content.shape[0] + padding[0] + padding[1]
    w = content.shape[1] + padding[2] + padding[3]
    if force_cross_long:
        dims = [h, w]
        extra = max(0, dims[axis]+1-dims[cross])
        padding[0 if cross == 0 else 2] += extra
        h = content.shape[0] + padding[0] + padding[1]
        w = content.shape[1] + padding[2] + padding[3]
    outline, header = _outline(rng, h, w, padding, axis, compact)
    result = np.full((h, w), bg, dtype=np.int8)
    result[outline] = ink
    y, x = padding[0], padding[2]
    result[y:y+content.shape[0], x:x+content.shape[1]] = content
    result[header[0]], result[header[1]] = node['primary'], node['secondary']
    return result


def _make_pair(rng, palette, mode):
    for scene_attempt in range(80):
        node = _scene(rng, palette, mode)
        if max(_shape(node)) > 30:
            continue
        expected = _evaluate(node)
        for draw_attempt in range(24):
            compact = draw_attempt >= 6
            core = _draw(node, rng, palette, compact, mode == 'simple_witness')
            if max(core.shape) > 28:
                continue
            room = [30-d for d in core.shape]
            extra = [int(rng.integers(2, min(6, d)+1)) for d in room]
            before = [int(rng.integers(1, d)) for d in extra]
            inp = np.pad(core, tuple((b, e-b) for b, e in zip(before, extra)),
                         constant_values=palette[0])
            try:
                inferred = _parse(inp)
                if np.array_equal(_evaluate(inferred['root']), expected):
                    return dict(input=inp, output=expected)
            except ValueError:
                pass
    raise RuntimeError("bounded construction search exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable = bool(rng.integers(2))
    base = tuple(map(int, rng.choice(10, 3, replace=False)))
    result, seen = {'train': [], 'test': []}, set()
    modes = ['simple_witness', 'nested_witness'] + ['ordinary']*(num_train-2)
    test_modes = list(rng.permutation(['deep', 'branch', 'count', 'word']))[:num_test]
    for split, specs in (('train', modes), ('test', test_modes)):
        for mode in specs:
            for _ in range(20):
                palette = base if stable else tuple(map(int, rng.choice(10, 3, replace=False)))
                pair = _make_pair(rng, palette, mode)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain a distinct input")
        rng.shuffle(result[split])
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, tests = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(solve(inp), out)
            _audit_output(info['root'], out)
            (training if split == 'train' else tests).append(info)
    # These are learning-evidence checks, not fixed literal-color or canvas rules.
    simple = [i for i in training if not i['root']['children'] and i['root']['count'] > 1
              and not i['root']['stencil'].all()]
    nested = [i for i in training if len(i['root']['children']) >= 2
              and any(c['count'] > 1 and c['axis'] != i['root']['axis']
                      and (c['primary'], c['secondary']) != (i['root']['primary'], i['root']['secondary'])
                      for c in i['root']['children'])]
    assert simple and nested, "missing the two concept-interaction witnesses"
    assert any(i['root']['token_areas'] and sum(i['root']['token_areas']) > i['root']['count'] for i in simple)
    scopes = [n for i in training for n in _walk(i['root'])]
    assert any(a['count'] != b['count'] and sum(a['token_areas']) == sum(b['token_areas'])
               for a in scopes for b in scopes), "ink area is confounded with count"
    assert any(not np.array_equal(_evaluate(i['root'], 'one'), _evaluate(i['root'])) for i in training)
    assert any(not np.array_equal(_evaluate(i['root'], 'descendants'), _evaluate(i['root'])) for i in nested)
    # Every test transfers a taught operation to a new structural or quantity setting.
    assert all(_depth(i['root']) == 3 or any(len(n['children']) == 3 or n['count'] == 4
               or (len(n['children']) == 2 and n['count'] > 1) for n in _walk(i['root'])) for i in tests)


def _scope_transfer_fixture():
    """Same boxes, stencils and total token count; move one token across a loop."""
    g = np.zeros((26, 26), dtype=np.int8)
    for lo, hi in ((1, 24), (8, 17)):
        g[lo, lo:hi+1] = g[hi, lo:hi+1] = 7
        g[lo:hi+1, lo] = g[lo:hi+1, hi] = 7
    g[1, 6:8] = [2, 3]
    g[10:12, 8] = [4, 5]
    g[11, 12:14] = 6
    g[12, 12] = 6
    g[4, 4] = g[14, 10] = g[4, 18] = 7
    moved = g.copy()
    moved[4, 18], moved[15, 14] = 0, 7
    return g, moved


def self_test():
    """Bounded 42-seed coverage, causal fixtures and corruption checks."""
    shapes, answers, trees, token_areas, palettes, axes = set(), set(), set(), set(), set(), set()
    signatures, shifts = set(), set()
    distorted = 0
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info, out = _parse(pair['input']), pair['output']
                root = info['root']
                shapes.add(pair['input'].shape)
                # Canonical role-normalized outputs count geometry, not recoloring.
                roles = {}
                answers.add((out.shape, tuple(roles.setdefault(int(c), len(roles)) for c in out.flat)))
                trees.add(tuple((n['axis'], n['count'], len(n['children']),
                                  None if n['stencil'] is None else _key(n['stencil'])) for n in _walk(root)))
                palettes.add((info['background'], info['ink'], info['stencil_ink']))
                for n in _walk(root):
                    axes.add(n['axis'])
                    token_areas.update(n['token_areas'])
                for rec in info['records']:
                    if rec['kind'] == 'contour':
                        local = _crop(rec['mask'], rec['mask'])
                        distorted += not (local[0].all() and local[-1].all()
                                          and local[:, 0].all() and local[:, -1].all())
                if split == 'test':
                    if _depth(root) == 3:
                        shifts.add('deep')
                    if len(root['children']) == 3:
                        shifts.add('branch')
                    if any(n['count'] == 4 for n in _walk(root)):
                        shifts.add('count')
                    if len(root['children']) == 2 and root['count'] == 2:
                        shifts.add('word')
        # A bijection on colors changes no structural decision.
        permutation = np.random.default_rng(seed+712).permutation(10)
        pair = puzzle['test'][0]
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    assert len(signatures) == 42 and len(shapes) >= 35 and len(answers) >= 65
    assert len(trees) >= 80 and len(palettes) >= 35 and len(token_areas) >= 8
    assert axes == {0, 1} and shifts == {'deep', 'branch', 'count', 'word'}
    assert distorted >= 35
    assert tuple(_key(p['input']) for s in ('train', 'test') for p in generate_puzzle(0)[s]) == \
           tuple(_key(p['input']) for s in ('train', 'test') for p in puzzles[0][s])
    # All legal counts, including the minimal 2/1 case, retain the witnesses.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(9071, a, b)
            assert len(puzzle['train']) == a and len(puzzle['test']) == b
            validate_generated_puzzle(puzzle)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    first, moved = _scope_transfer_fixture()
    assert solve(first).shape == (10, 17) and solve(moved).shape == (13, 10)
    assert np.count_nonzero(first == 7) == np.count_nonzero(moved == 7)
    assert not np.array_equal(solve(first), solve(moved))
    # Redrawing contour geometry and token shapes preserves the same answer.
    rng = np.random.default_rng(509)
    node = _scene(rng, (0, 7, 6), 'simple_witness')
    expected, realizations = _evaluate(node), set()
    for _ in range(12):
        core = _draw(node, rng, (0, 7, 6), compact=False)
        if max(core.shape) <= 28:
            inp = np.pad(core, 1)
            assert np.array_equal(solve(inp), expected)
            realizations.add(_key(inp))
    assert len(realizations) >= 4
    assert np.array_equal(solve(np.pad(first[1:-1, 1:-1], ((2, 1), (1, 3)))), solve(first))
    # A visible two-child word is repeated as ABAB, not as AABB.
    def dot(p, s):
        return dict(axis=1, count=1, primary=p, secondary=s,
                    children=[], stencil=np.ones((1, 1), dtype=bool))
    word = dict(axis=1, count=2, primary=8, secondary=9,
                children=[dot(2, 3), dot(4, 5)], stencil=None)
    word_input = np.pad(_draw(word, rng, (0, 7, 6), compact=True), 1)
    word_output = solve(word_input)
    assert word_output.shape == (9, 27)
    assert [int(word_output[2, 2+x]) for x in (0, 6, 12, 18)] == [2, 4, 2, 4]
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    corrupt['test'][0]['output'].flat[0] = (int(corrupt['test'][0]['output'].flat[0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    opened = first.copy()
    opened[1, 15] = 0
    try:
        solve(opened)
    except ValueError:
        pass
    else:
        raise AssertionError("open contour accepted")
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), normalized_outputs=len(answers),
                structural_signatures=len(trees), structural_palettes=len(palettes),
                token_areas=sorted(token_areas), distorted_contours=distorted,
                test_shifts=sorted(shifts), independent_solver=False)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
