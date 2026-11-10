"""Keyed mold packing: one-owner cavities and two-owner geometric tabs.

Only NumPy and the standard library are used. Generation paints a construction;
an exhaustive input-only search checks the rendered answer and its uniqueness.
No examples, random seeds, or construction state are consulted by solve().
"""
from itertools import combinations
import hashlib
import json
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SEARCH_LIMIT = 40000


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _rect(r, c, h, w):
    return {(y, x) for y in range(r, r + h) for x in range(c, c + w)}


def _shift(cells, d):
    return {(r + d[0], c + d[1]) for r, c in cells}


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in N4} - cells


def _components(cells):
    left, result = set(cells), []
    while left:
        q = min(left)
        left.remove(q)
        comp, todo = {q}, [q]
        while todo:
            r, c = todo.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    comp.add(q)
                    todo.append(q)
        result.append(comp)
    return result


def _key(grid):
    a = np.ascontiguousarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _normalized(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError('Expected an integer ARC grid of size 1..30 and colors 0..9')
    return a.astype(np.int8, copy=True)


def _parse_piece(grid, cells, marker):
    body = {q for q in cells if grid[q] != marker}
    pins = cells - body
    if not body or not pins:
        return None
    t, l, b, r = _box(body)
    if body != _rect(t, l, b - t + 1, r - l + 1) or not pins <= _halo(body):
        return None
    pattern = grid[t:b + 1, l:r + 1].copy()
    return {'pattern': pattern, 'pins': _shift(pins, (-t, -l)),
            'origin': (t, l)}


def _interpretations(grid):
    """Find a two-color framed mold, then infer the common external tab hue."""
    for wall in (int(c) for c in np.unique(grid)):
        cells = set(map(tuple, np.argwhere(grid == wall).tolist()))
        t, l, b, r = _box(cells)
        if b - t < 3 or r - l < 3:
            continue
        mold = grid[t:b + 1, l:r + 1]
        if not (np.all(mold[0] == wall) and np.all(mold[-1] == wall)
                and np.all(mold[:, 0] == wall) and np.all(mold[:, -1] == wall)):
            continue
        colors = set(int(c) for c in np.unique(mold))
        if len(colors) != 2:
            continue
        bg = (colors - {wall}).pop()
        outside = set(map(tuple, np.argwhere(grid != bg).tolist())) - _rect(t, l, *mold.shape)
        cc = _components(outside)
        if len(cc) < 2:
            continue
        # The frame and inventory must be separated spatially in the input.
        if any(q in _halo(_rect(t, l, *mold.shape)) for q in outside):
            continue
        common = set.intersection(*({int(grid[q]) for q in c} for c in cc)) - {bg, wall}
        for marker in sorted(common):
            pieces = [_parse_piece(grid, comp, marker) for comp in cc]
            if any(p is None for p in pieces):
                continue
            yield {'mold': mold.copy(), 'bg': bg, 'wall': wall,
                   'marker': marker, 'pieces': pieces, 'origin': (t, l)}


def _placements(a):
    mold, pieces, bg = a['mold'], a['pieces'], a['bg']
    h, w = mold.shape
    holes = mold == bg
    target = sum(1 << (r * w + c) for r, c in np.argwhere(holes).tolist())
    by_cell = {i: [] for i in range(h * w) if target >> i & 1}
    records = []
    for i, p in enumerate(pieces):
        ph, pw = p['pattern'].shape
        for r in range(1, h - ph):
            for c in range(1, w - pw):
                if not np.all(holes[r:r + ph, c:c + pw]):
                    continue
                pins = _shift(p['pins'], (r, c))
                if any(y <= 0 or x <= 0 or y >= h - 1 or x >= w - 1
                       or mold[y, x] != a['wall'] for y, x in pins):
                    continue
                ids = tuple(y * w + x for y in range(r, r + ph) for x in range(c, c + pw))
                body = sum(1 << j for j in ids)
                pinbits = sum(1 << (y * w + x) for y, x in pins)
                index = len(records)
                records.append((i, r, c, body, pinbits))
                for j in ids:
                    by_cell[j].append(index)
    return target, records, by_cell


def _connected(path, records):
    if not path:
        return False
    reached, todo = {0}, [0]
    while todo:
        i = todo.pop()
        pins = records[path[i]][4]
        for j in range(len(path)):
            if j not in reached and pins & records[path[j]][4]:
                reached.add(j)
                todo.append(j)
    return len(reached) == len(path)


def _search(a, docking=True, limit=SEARCH_LIMIT):
    """Enumerate rendered answers; stop after two. Budget exhaustion is an error.

    Physical pieces are indexed separately: finite inventories are explicit.
    Two interchangeable identical copies can yield the same rendered answer.
    docking=False is a diagnostic ablation, never used to label a pair.
    """
    target, records, by_cell = _placements(a)
    outcomes, path = {}, []
    nodes = 0
    dead = set()

    def visit(covered, used, singles, doubles):
        nonlocal nodes
        nodes += 1
        if nodes > limit:
            raise ValueError('Packing uniqueness search exceeded its bounded node budget')
        if covered == target:
            if docking and (singles or not _connected(path, records)):
                return False
            out = a['mold'].copy()
            w = out.shape[1]
            allpins = 0
            for k in path:
                i, r, c, body, pins = records[k]
                tile = a['pieces'][i]['pattern']
                out[r:r + tile.shape[0], c:c + tile.shape[1]] = tile
                allpins |= pins
            for q in range(out.size):
                if allpins >> q & 1:
                    out[q // w, q % w] = a['marker']
            outcomes.setdefault(_key(out), (out, tuple(records[k] for k in path)))
            return True
        # Connectivity depends on which complete placements reached this state.
        # Include the path so a failed graph cannot suppress a different graph
        # with the same occupancy and marker multiplicities.
        state = covered, used, singles, doubles, tuple(sorted(path))
        if state in dead:
            return False
        best = None
        possible_pins = 0
        for cell, indices in by_cell.items():
            if covered >> cell & 1:
                continue
            choices = []
            for k in indices:
                i, r, c, body, pins = records[k]
                if used >> i & 1 or body & covered or (docking and pins & doubles):
                    continue
                choices.append(k)
                possible_pins |= pins
            if not choices:
                dead.add(state)
                return False
            if best is None or len(choices) < len(best):
                best = choices
        if docking and singles & ~possible_pins:
            dead.add(state)
            return False
        found = False
        for k in best:
            i, r, c, body, pins = records[k]
            path.append(k)
            found |= visit(covered | body, used | (1 << i),
                           singles ^ pins if docking else 0,
                           doubles | (singles & pins) if docking else 0)
            path.pop()
            if len(outcomes) >= 2:
                return True
        # Cache only dead states. A successful state's rendering may depend on
        # earlier assignments, so it must not suppress a second pixel answer.
        if not found:
            dead.add(state)
        return found

    visit(0, 0, 0, 0)
    return list(outcomes.values()), nodes


def _infer(grid):
    answers = []
    for a in _interpretations(_grid(grid)):
        outcomes, nodes = _search(a)
        if len(outcomes) > 1:
            raise ValueError('More than one rendered keyed packing')
        if outcomes:
            a.update(output=outcomes[0][0], placements=outcomes[0][1], nodes=nodes)
            answers.append(a)
    if len(answers) != 1:
        raise ValueError('Input must identify one framed mold and one rendered packing')
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the visible frame, cavity mask, and source inventory."""
    return _infer(input_grid)['output'].copy()


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _layout(rng, count, branch):
    """Grow rectangles whose adjacent halos share sparse two-cell keys.

    A zero-width rib allows payload rectangles to touch along a partial edge;
    shared tab locations can then wrap around corners of the merged cavity.
    """
    for restart in range(70):
        h, w = int(rng.integers(3, 7)), int(rng.integers(2, 5))
        bodies, pins, edges = [_rect(0, 0, h, w)], [set()], []
        for i in range(1, count):
            for attempt in range(180):
                parent = int(rng.integers(i)) if branch else i - 1
                pt, pl, pb, pr = _box(bodies[parent])
                h, w = int(rng.integers(2, 7)), int(rng.integers(2, 6))
                side = int(rng.integers(4))
                gap = int(rng.choice([0, 1], p=[.35, .65]))
                if side < 2:
                    r = pt - h - gap if side == 0 else pb + 1 + gap
                    c = int(rng.integers(pl - w + 1, pr + 1))
                else:
                    c = pl - w - gap if side == 2 else pr + 1 + gap
                    r = int(rng.integers(pt - h + 1, pb + 1))
                body = _rect(r, c, h, w)
                oldbody = set().union(*bodies)
                oldpins = set().union(*pins)
                if body & (oldbody | oldpins):
                    continue
                common = sorted((_halo(body) & _halo(bodies[parent])) - oldbody - oldpins)
                if len(common) < 2:
                    continue
                ix = rng.choice(len(common), 2, replace=False)
                key = {common[int(j)] for j in ix}
                allcells = oldbody | oldpins | body | key
                t, l, b, rr = _box(allcells)
                if b - t + 1 > 22 or rr - l + 1 > 12:
                    continue
                bodies.append(body)
                pins.append(set(key))
                pins[parent] |= key
                edges.append((parent, i, key))
                break
            else:
                break
        if len(bodies) != count:
            continue
        degrees = [sum(i in e[:2] for e in edges) for i in range(count)]
        if branch and max(degrees) < 3:
            continue
        if len({_normalized(k) for _, _, k in edges}) < 2:
            continue
        # Reserve an endpoint whose straight key admits both required ablations.
        eligible = []
        occupied = set().union(*bodies, *pins)
        for i in range(count):
            if degrees[i] != 1:
                continue
            t, l, b, r = _box(bodies[i])
            ps = pins[i]
            sides = [(all(y == t - 1 for y, x in ps), (1, 0)),
                     (all(y == b + 1 for y, x in ps), (-1, 0)),
                     (all(x == l - 1 for y, x in ps), (0, 1)),
                     (all(x == r + 1 for y, x in ps), (0, -1))]
            for side, (ok, away) in enumerate(sides):
                if not ok:
                    continue
                length = r - l + 1 if side < 2 else b - t + 1
                if length < 3:
                    continue
                dr, dc = away
                extended = bodies[i] | _shift(bodies[i], (dr, dc))
                if (extended - bodies[i]) & occupied:
                    continue
                eligible.append((i, side, extended))
        if eligible:
            return bodies, pins, edges, _pick(rng, eligible)
    raise ValueError('Bounded layout construction could not fit the assembly')


def _pattern(rng, shape, palette, asymmetric=False):
    h, w = shape
    a, b = [int(c) for c in rng.choice(palette, 2, replace=False)]
    yy, xx = np.indices(shape)
    mode = int(rng.integers(6))
    tile = np.full(shape, a, dtype=np.int8)
    if mode == 1:
        tile[yy % 2 == 1] = b
    elif mode == 2:
        tile[xx % 2 == 1] = b
    elif mode == 3:
        tile[(xx + yy) % 2 == 1] = b
    elif mode == 4:
        tile = rng.choice(palette, size=shape).astype(np.int8)
    elif mode == 5 or asymmetric:
        tile[0, 0] = b
    if asymmetric:
        tile.fill(a)
        tile[0, 0] = b
    return tile


def _source(body, pins, pattern):
    t, l, _, _ = _box(body)
    return {'pattern': pattern.copy(), 'pins': _shift(pins, (-t, -l))}


def _scatter(rng, pieces, bg, marker, height, width):
    normalized = []
    for p in pieces:
        h, w = p['pattern'].shape
        cells = _rect(0, 0, h, w) | p['pins']
        t, l, b, r = _box(cells)
        tile = np.full((b - t + 1, r - l + 1), bg, dtype=np.int8)
        tile[-t:h - t, -l:w - l] = p['pattern']
        for y, x in p['pins']:
            tile[y - t, x - l] = marker
        normalized.append(tile)
    for restart in range(32):
        tray = np.full((height, width), bg, dtype=np.int8)
        taken = np.zeros_like(tray, dtype=bool)
        order = list(rng.permutation(len(pieces)))
        order.sort(key=lambda i: -normalized[i].size)
        for i in order:
            tile = normalized[i]
            h, w = tile.shape
            legal = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)
                     if not np.any(taken[max(0, r - 1):min(height, r + h + 1),
                                          max(0, c - 1):min(width, c + w + 1)])]
            if not legal:
                break
            r, c = _pick(rng, legal)
            tray[r:r + h, c:c + w] = tile
            taken[r:r + h, c:c + w] = True
        else:
            return tray
    return None


def _make_pair(rng, count, branch, rotation, asymmetric):
    for attempt in range(90):
        try:
            bodies, pins, edges, (endpoint, side, extended) = _layout(rng, count, branch)
        except ValueError:
            continue
        colors = [int(c) for c in rng.permutation(10)]
        bg, wall, marker = colors[:3]
        palette = colors[3:3 + int(rng.integers(2, 7))]
        pats = []
        for i, body in enumerate(bodies):
            t, l, b, r = _box(body)
            pats.append(_pattern(rng, (b - t + 1, r - l + 1), palette,
                                 asymmetric=asymmetric and i == endpoint))
        pieces = [_source(body, ps, pat) for body, ps, pat in zip(bodies, pins, pats)]
        # Spare 1: same payload dimensions, different equal-area geometric key.
        t, l, b, r = _box(bodies[endpoint])
        oldkey = pins[endpoint]
        tangent = list(range(l, r + 1)) if side < 2 else list(range(t, b + 1))
        alternatives = []
        for v1, v2 in combinations(tangent, 2):
            key = ({(t - 1 if side == 0 else b + 1, v1),
                    (t - 1 if side == 0 else b + 1, v2)} if side < 2 else
                   {(v1, l - 1 if side == 2 else r + 1),
                    (v2, l - 1 if side == 2 else r + 1)})
            if _normalized(key) != _normalized(oldkey):
                alternatives.append(key)
        if not alternatives:
            continue
        wrongkey = _pick(rng, alternatives)
        badpat = pats[endpoint].copy()
        badpat[0, 0] = _pick(rng, [c for c in palette if c != badpat[0, 0]])
        pieces.append(_source(bodies[endpoint], wrongkey, badpat))
        # Spare 2: identical absolute pins, but extend the payload away from them.
        et, el, eb, er = _box(extended)
        extpat = _pattern(rng, (eb - et + 1, er - el + 1), palette)
        extpat[t - et:b - et + 1, l - el:r - el + 1] = pats[endpoint]
        pieces.append(_source(extended, oldkey, extpat))
        # Some examples have another wrong-key spare, with no change of rule.
        if rng.random() < .25 and len(alternatives) > 1:
            key = alternatives[(alternatives.index(wrongkey) + 1) % len(alternatives)]
            pieces.append(_source(bodies[endpoint], key, _pattern(rng, badpat.shape, palette)))
        # The frame also contains the connector-only alternate, so the A ablation
        # cannot be explained by a grid-boundary violation.
        allcells = set().union(*bodies, *pins, extended)
        at, al, ab, ar = _box(allcells)
        top, left = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        bottom, right = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        height, width = ab - at + 1 + top + bottom, ar - al + 1 + left + right
        if height > 29 or width > 15:
            continue
        d = top - at, left - al
        mold = np.full((height, width), wall, dtype=np.int8)
        expected = mold.copy()
        for body, pat in zip(bodies, pats):
            bt, bl, bb, br = _box(_shift(body, d))
            mold[bt:bb + 1, bl:br + 1] = bg
            expected[bt:bb + 1, bl:br + 1] = pat
        for q in _shift(set().union(*pins), d):
            expected[q] = marker
        canvas_h = int(rng.integers(max(height, 23), 31))
        tray_w = 29 - width
        tray = _scatter(rng, pieces, bg, marker, canvas_h, tray_w)
        if tray is None:
            continue
        boardpanel = np.full((canvas_h, width), bg, dtype=np.int8)
        offset = int(rng.integers(canvas_h - height + 1))
        boardpanel[offset:offset + height] = mold
        inp = np.concatenate((boardpanel, np.full((canvas_h, 1), bg, dtype=np.int8), tray), axis=1)
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(expected, rotation).copy()
        if rng.random() < .5:
            inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        try:
            a = _infer(inp)
            if not np.array_equal(a['output'], out):
                continue
            alternatives_a, _ = _search(a, docking=False)
            if len(alternatives_a) < 2 or not _a_witness(a):
                continue
        except ValueError:
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError('Bounded construction could not produce a unique keyed mold')


def _a_witness(a):
    """Find a connector-complete one-piece replacement that violates the mold.

    Keep all other selected pieces, the outer frame, exact tabs, and collision
    rules. Only the cavity constraint is relaxed. No generation metadata used.
    """
    selected = {rec[0] for rec in a['placements']}
    h, w = a['mold'].shape
    positions = []
    for i, r, c, bits, pinbits in a['placements']:
        p = a['pieces'][i]
        positions.append((i, _rect(r, c, *p['pattern'].shape), _shift(p['pins'], (r, c))))
    allpins = set().union(*(ps for _, _, ps in positions))
    for i, body, ps in positions:
        others = set().union(*(bb for j, bb, _ in positions if j != i))
        for j, p in enumerate(a['pieces']):
            if j in selected or len(p['pins']) != len(ps):
                continue
            for anchor in sorted(ps):
                for pin in sorted(p['pins']):
                    d = anchor[0] - pin[0], anchor[1] - pin[1]
                    if _shift(p['pins'], d) != ps:
                        continue
                    replacement = _rect(*d, *p['pattern'].shape)
                    if replacement == body or replacement & (others | allpins):
                        continue
                    if any(r <= 0 or c <= 0 or r >= h - 1 or c >= w - 1
                           for r, c in replacement):
                        continue
                    if any(a['mold'][q] == a['wall'] for q in replacement):
                        return True
    return False


def _check_counts(num_train, num_test):
    for name, value, lo, hi in [('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)]:
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    phase = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in [('train', num_train), ('test', num_test)]:
        for index in range(count):
            if split == 'train':
                n = 3 if index == 0 else 4 if index == 1 else int(rng.integers(3, 5 if regime == 0 else 6))
                branch = index == 1 or (n >= 4 and rng.random() < .45)
                rotation = (phase + 2 * (index % 2)) % 4 if regime == 1 else int(rng.integers(4))
            else:
                n = int(rng.integers(5, 7)) if regime != 1 else int(rng.integers(4, 6))
                branch = regime == 2 or (n >= 4 and rng.random() < .6)
                rotation = (phase + 1 + 2 * (index % 2)) % 4 if regime == 1 else int(rng.integers(4))
            for retry in range(8):
                pair = _make_pair(rng, n, branch, rotation, asymmetric=split == 'train' and index == 0)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Bounded distinct-input construction failed')
    return puzzle


def _features(a):
    rec = a['placements']
    degrees = [0] * len(rec)
    keys = []
    for i in range(len(rec)):
        for j in range(i):
            overlap = rec[i][4] & rec[j][4]
            if overlap:
                degrees[i] += 1
                degrees[j] += 1
                w = a['mold'].shape[1]
                keys.append(_normalized({(q // w, q % w) for q in range(a['mold'].size) if overlap >> q & 1}))
    holes = set(map(tuple, np.argwhere(a['mold'] == a['bg']).tolist()))
    bodies = [_rect(r, c, *a['pieces'][i]['pattern'].shape) for i, r, c, _, _ in rec]
    cc = _components(holes)
    return {'used': len(rec), 'supplied': len(a['pieces']), 'frame': a['mold'].shape,
            'roles': (a['bg'], a['wall'], a['marker']), 'degrees': tuple(degrees),
            'body_shapes': tuple(sorted(a['pieces'][i]['pattern'].shape for i, *_ in rec)),
            'keys': tuple(keys), 'merged': max(sum(bool(body & comp) for body in bodies) for comp in cc),
            'components': len(cc), 'noncollinear': any(len({r for r, c in k}) > 1 and len({c for r, c in k}) > 1 for k in keys),
            'nodes': a['nodes']}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _infer(inp)
            assert np.array_equal(a['output'], out)
            expected = a['mold'].copy()
            h, w = expected.shape
            count_body = np.zeros(expected.shape, dtype=np.int8)
            count_pin = np.zeros(expected.shape, dtype=np.int8)
            used = set()
            for i, r, c, _, _ in a['placements']:
                assert i not in used
                used.add(i)
                p = a['pieces'][i]
                ph, pw = p['pattern'].shape
                count_body[r:r + ph, c:c + pw] += 1
                expected[r:r + ph, c:c + pw] = p['pattern']
                for q in _shift(p['pins'], (r, c)):
                    count_pin[q] += 1
            assert np.array_equal(count_body, (a['mold'] == a['bg']).astype(np.int8))
            assert np.all((count_pin == 0) | (count_pin == 2))
            assert np.all(a['mold'][count_pin > 0] == a['wall'])
            assert np.all(count_pin[[0, -1], :] == 0) and np.all(count_pin[:, [0, -1]] == 0)
            expected[count_pin == 2] = a['marker']
            assert np.array_equal(expected, out) and len(used) < len(a['pieces'])
            assert np.array_equal(out[[0, -1]], a['mold'][[0, -1]])
            assert np.array_equal(out[:, [0, -1]], a['mold'][:, [0, -1]])
            # Counterfactuals establish both inherited mechanisms in every pair.
            alternatives, _ = _search(a, docking=False)
            assert len(alternatives) == 2 and _a_witness(a)
            f = _features(a)
            assert max(f['degrees']) >= 1 and len(set(f['keys'])) >= 2
            features.append(f)
    nt = len(puzzle['train'])
    assert features[0]['used'] == 3 and features[1]['used'] == 4
    assert max(features[1]['degrees']) >= 3
    assert len({f['body_shapes'] for f in features[:nt]}) >= 2


def self_test():
    """42 seeds, streamed: no stored grids, traces, or unbounded retry logs."""
    signatures, frames, shapes, keys, roles = set(), set(), set(), set(), set()
    structures, used_counts, supplied_counts, merged = set(), set(), set(), set()
    noncollinear = branches = pairs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _infer(pair['input'])
                f = _features(a)
                digest.update(pair['input'].tobytes())
                frames.add(f['frame'])
                shapes.update(f['body_shapes'])
                keys.update(f['keys'])
                roles.add(f['roles'])
                used_counts.add(f['used'])
                supplied_counts.add(f['supplied'])
                merged.add(f['merged'])
                noncollinear += f['noncollinear']
                branches += max(f['degrees']) >= 3
                pairs += 1
                # Color-independent destination geometry, beyond recoloring.
                structures.add((a['mold'].shape, (a['mold'] == a['bg']).tobytes()))
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(frames) >= 30 and len(structures) >= 150
    assert len(shapes) >= 15 and len(keys) >= 8 and len(roles) >= 30
    assert used_counts >= {3, 4, 5, 6} and len(supplied_counts) >= 4
    assert max(merged) >= 3 and noncollinear >= 10 and branches >= 42
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            _check_counts(nt, ne)
    for nt, ne in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)]:
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    p, q = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    # Reject a label that loses a shared connector or changes a copied payload.
    a = _infer(p['test'][0]['input'])
    bad = {s: [{'input': x['input'].copy(), 'output': x['output'].copy()} for x in p[s]] for s in p}
    q = tuple(np.argwhere(bad['test'][0]['output'] == a['marker'])[0])
    bad['test'][0]['output'][q] = a['wall']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Erased shared tab accepted')
    return {'seeds': 42, 'pairs': pairs, 'distinct_puzzles': len(signatures),
            'frames': len(frames), 'body_shapes': len(shapes), 'key_shapes': len(keys),
            'role_assignments': len(roles), 'cavity_structures': len(structures),
            'selected_counts': sorted(used_counts), 'supplied_counts': sorted(supplied_counts),
            'max_rectangles_in_cavity': max(merged), 'noncollinear_pairs': noncollinear,
            'branched_pairs': branches}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
