"""Growing relay: glyph-directed, color-linked, rigid module assembly.

Only NumPy and the standard library are used. Prediction accepts one input and
never consults generation, a seed, training examples, files, or cached state.
Self-tests stream bounded data; no grids, traces, or logs are written here.
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIR = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N,E,S,W
_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1)
                   for c in (-1, 0, 1) if r or c)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer colors required"
    assert np.all((0 <= a) & (a <= 9)), "color range"
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _port(h, w, neck, side):
    r, c = h // 2 - 1 + neck, w // 2 - 1
    return np.array((((0, c), (0, c + 1)),
                     ((r, w - 1), (r + 1, w - 1)),
                     ((neck + h - 1, c), (neck + h - 1, c + 1)),
                     ((r, 0), (r + 1, 0)))[side], dtype=int)


def _glyph(h, w, side):
    """A thin, visibly routed miniature path from north to one outlet."""
    assert side in (1, 2, 3)
    mask = np.zeros((h, w), dtype=bool)
    r, c = h // 2 - 1, w // 2 - 1
    mask[1:r + 1, c] = True
    if side == 2:
        mask[r:h - 1, c] = True
    elif side == 1:
        mask[r, c:w - 1] = True
    else:
        mask[r, 1:c + 1] = True
    return mask


def _module(h, w, neck, side, labels, bg, body, ink):
    a = np.full((h + neck, w), bg, dtype=np.int8)
    a[neck, :] = a[-1, :] = body
    a[neck:, 0] = a[neck:, -1] = body
    a[:neck, w // 2 - 1:w // 2 + 1] = body
    for d, color in enumerate(labels):
        p = _port(h, w, neck, d)
        a[p[:, 0], p[:, 1]] = color
    interior = a[neck:]
    interior[_glyph(h, w, side)] = ink
    return dict(array=a, height=h, width=w, neck=neck, side=side,
                labels=tuple(labels), inlet=int(labels[0]), undo=0)


def _root(label, body):
    return np.array(((body, body, label), (body, body, label)), dtype=np.int8)


def _components(a, bg):
    """Eight-connected multicolor objects; complete bounding rectangles returned."""
    unseen = a != bg
    objects = []
    for r0, c0 in np.argwhere(unseen):
        r0, c0 = int(r0), int(c0)
        if not unseen[r0, c0]:
            continue
        unseen[r0, c0] = False
        stack, cells = [(r0, c0)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1] and unseen[rr, cc]:
                    unseen[rr, cc] = False
                    stack.append((rr, cc))
        pts = np.array(cells, dtype=int)
        lo, hi = pts.min(axis=0), pts.max(axis=0) + 1
        block = a[lo[0]:hi[0], lo[1]:hi[1]].copy()
        assert np.count_nonzero(block != bg) == len(cells), "nested/overlapping object boxes"
        objects.append(dict(array=block, origin=lo, coords=pts))
    # Also disallow interleaved, non-touching components inside another box.
    for i, x in enumerate(objects):
        for y in objects[i + 1:]:
            x0, y0 = x['origin'], y['origin']
            x1, y1 = x0 + x['array'].shape, y0 + y['array'].shape
            assert np.any((x1 < y0) | (y1 < x0)), "object boxes need background clearance"
    return objects


def _parse_module(obj, bg, body):
    candidates = []
    for undo in range(4):
        a = np.rot90(obj['array'], undo)
        for neck in (1, 2):
            h, w = a.shape[0] - neck, a.shape[1]
            if h not in (6, 8) or w not in (6, 8) or (h == 8 and neck == 2):
                continue
            labels = []
            for side in range(4):
                p = _port(h, w, neck, side)
                values = a[p[:, 0], p[:, 1]]
                if values[0] != values[1] or values[0] in (bg, body):
                    break
                labels.append(int(values[0]))
            if len(labels) != 4 or len(set(labels)) != 4:
                continue
            inside = a[neck + 1:neck + h - 1, 1:w - 1]
            colors = np.unique(inside[inside != bg])
            if len(colors) != 1 or colors[0] in (body, *labels):
                continue
            ink = int(colors[0])
            for side in (1, 2, 3):
                m = _module(h, w, neck, side, labels, bg, body, ink)
                if np.array_equal(m['array'], a):
                    m.update(undo=undo, ink=ink, source=obj)
                    candidates.append(m)
    assert len(candidates) == 1, "module inlet/glyph must have one interpretation"
    return candidates[0]


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > a.size // 2, "background must be the unique majority"
    counts[bg] = 0
    body = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[body]) == 1, "frame color must be unique mode"
    objects = _components(a, bg)
    roots, modules = [], []
    for obj in objects:
        if obj['array'].size == 6:
            for turns in range(4):
                b = np.rot90(obj['array'], turns)
                if b.shape != (2, 3):
                    continue
                label = int(b[0, 2])
                if label not in (bg, body) and np.array_equal(b, _root(label, body)):
                    port = np.argwhere(obj['array'] == label) + obj['origin']
                    roots.append(dict(source=obj, label=label, port=port,
                                      heading=(1 + turns) % 4))
        else:
            modules.append(_parse_module(obj, bg, body))
    assert len(roots) == 1 and modules, "one start and some modules required"
    inks = {m['ink'] for m in modules}
    assert len(inks) == 1
    ink = next(iter(inks))
    assert all(ink not in m['labels'] for m in modules)
    assert roots[0]['label'] != ink
    inlets = {m['inlet']: i for i, m in enumerate(modules)}
    assert len(inlets) == len(modules), "inlet colors must be unique"
    return dict(background=bg, body=body, ink=ink, root=roots[0],
                modules=modules, inlets=inlets)


def _position(m, active_port, heading, bg):
    """One rigid transform: canonical north inlet faces the incoming endpoint."""
    turns = (2 - heading) % 4
    a = np.rot90(m['array'], turns)
    inlet = np.argwhere(a == m['inlet'])
    targets = active_port + np.array(_DIR[heading])
    origin = targets.min(axis=0) - inlet.min(axis=0)
    assert set(map(tuple, inlet + origin)) == set(map(tuple, targets))
    pts = np.argwhere(a != bg)
    cells = pts + origin
    colors = a[pts[:, 0], pts[:, 1]]
    exit_color = m['labels'][m['side']]
    next_port = np.argwhere(a == exit_color) + origin
    assert next_port.shape == (2, 2)
    return dict(cells=cells, colors=colors, port=next_port,
                heading=(m['side'] - turns) % 4, label=int(exit_color),
                turns=turns, origin=origin)


def _execute(a, info):
    bg = info['background']
    out = np.full_like(a, bg)
    root = info['root']
    obj = root['source']
    pts = obj['coords']
    out[pts[:, 0], pts[:, 1]] = a[pts[:, 0], pts[:, 1]]
    port, heading, label = root['port'].copy(), root['heading'], root['label']
    used, steps = set(), []
    while label in info['inlets']:
        index = info['inlets'][label]
        assert index not in used, "active chain must not cycle"
        used.add(index)
        m = info['modules'][index]
        p = _position(m, port, heading, bg)
        rr, cc = p['cells'].T
        assert np.all((0 <= rr) & (rr < a.shape[0]))
        assert np.all((0 <= cc) & (cc < a.shape[1])), "assembly exceeds canvas"
        assert np.all(out[rr, cc] == bg), "assembled modules overlap"
        out[rr, cc] = p['colors']
        steps.append(dict(index=index, placement=p, module=m))
        port, heading, label = p['port'], p['heading'], p['label']
    assert steps, "root must start a real chain"
    return out, steps


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input only, without mutating it."""
    a = _grid(input_grid)
    return _execute(a, _parse(a))[0]


def _palette(rng, previous_bg):
    p = list(map(int, rng.permutation(10)))
    if p[0] == previous_bg:
        p[0], p[3] = p[3], p[0]
    return p[0], p[1], p[2], p[3:]


def _plan(rng, count, distractors, prefix, colors, small=False, witness=None):
    bg, body, ink, pool = colors
    labels = list(map(int, rng.permutation(pool)))
    chain_in = labels[:count]
    decoy_in = labels[count:count + distractors]
    terminal = labels[count + distractors]
    commands = list(prefix) + list(map(int, rng.choice((1, 2, 3), count - len(prefix))))
    modules = []
    for i in range(count + distractors):
        h, w = (6, 6) if small else tuple(map(int, rng.choice((6, 8), size=2)))
        neck = 1 if h == 8 or small else int(rng.integers(1, 3))
        if i == 0 and witness is not None:
            h, w, neck = witness
        if i < count:
            entry, side = chain_in[i], commands[i]
            successor = chain_in[i + 1] if i + 1 < count else terminal
            others = [x for x in pool if x not in (entry, successor, decoy_in[0])]
            rest = [decoy_in[0], int(rng.choice(others))]
        else:
            entry, side = decoy_in[i - count], int(rng.choice((1, 2, 3)))
            rest = list(map(int, rng.choice([x for x in pool if x != entry], 3, replace=False)))
            successor = rest.pop()
        rng.shuffle(rest)
        ports = [entry, None, None, None]
        ports[side] = successor
        for d, x in zip([d for d in (1, 2, 3) if d != side], rest):
            ports[d] = x
        modules.append(_module(h, w, neck, side, ports, bg, body, ink))
    root_turns = int(rng.integers(4))
    root = np.rot90(_root(chain_in[0], body), root_turns)
    root_pts = np.argwhere(root != bg)
    root_values = root[root_pts[:, 0], root_pts[:, 1]]
    active = np.argwhere(root == chain_in[0])
    heading = (1 - root_turns) % 4
    placements, cells, values = [], [root_pts], [root_values]
    occupied = set(map(tuple, root_pts))
    for m in modules[:count]:
        p = _position(m, active, heading, bg)
        points = set(map(tuple, p['cells']))
        if points & occupied:
            return None
        occupied.update(points)
        cells.append(p['cells'])
        values.append(p['colors'])
        placements.append(p)
        active, heading = p['port'], p['heading']
    all_cells = np.concatenate(cells)
    low, high = all_cells.min(axis=0), all_cells.max(axis=0)
    if np.any(high - low + 1 > 30):
        return None
    return dict(modules=modules, root=root, placements=placements,
                cells=all_cells, values=np.concatenate(values), low=low, high=high)


def _block(mask, top, left, h, w):
    mask[max(0, top - 1):min(mask.shape[0], top + h + 1),
         max(0, left - 1):min(mask.shape[1], left + w + 1)] = True


def _pack(rng, shape, root, origin, arrays, bg, fallback=False):
    a = np.full(shape, bg, dtype=np.int8)
    top, left = map(int, origin)
    h, w = root.shape
    a[top:top + h, left:left + w] = root
    blocked = np.zeros(shape, dtype=bool)
    _block(blocked, top, left, h, w)
    if fallback:
        assert shape == (30, 30)
        slots = [(r, c) for r in (0, 10, 20) for c in (0, 10, 20)
                 if not blocked[r:r + 9, c:c + 9].any()]
        assert len(slots) >= len(arrays), "nine-slot packing bound"
        rng.shuffle(slots)
        for b, (r, c) in zip(arrays, slots):
            dh, dw = b.shape
            r += int(rng.integers(10 - dh))
            c += int(rng.integers(10 - dw))
            a[r:r + dh, c:c + dw] = b
        return a
    order = list(map(int, rng.permutation(len(arrays))))
    order.sort(key=lambda i: -arrays[i].size)
    for i in order:
        b = arrays[i]
        h, w = b.shape
        if h > shape[0] or w > shape[1]:
            return None
        integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
        options = np.argwhere(sums == 0)
        if not len(options):
            return None
        # Prefer edges half the time to reduce fragmentation; retain free layouts.
        if rng.random() < .5:
            score = np.minimum(options[:, 0], shape[0] - h - options[:, 0])
            score += np.minimum(options[:, 1], shape[1] - w - options[:, 1])
            options = options[score == score.min()]
        r, c = map(int, options[int(rng.integers(len(options)))])
        a[r:r + h, c:c + w] = b
        _block(blocked, r, c, h, w)
    return a


def _make_pair(rng, count, distractors, prefix, previous_bg, deltas=(), witness=None):
    colors = _palette(rng, previous_bg)
    bg, body, ink, pool = colors
    plan = None
    for _ in range(96):
        plan = _plan(rng, count, distractors, prefix, colors, witness=witness)
        if plan is not None:
            break
    if plan is None:
        # Small training chains fit every two-command prefix. Longer tests use
        # the same actions in a known fitting pattern, never a different rule.
        p = prefix if count <= 3 else (2, 1, 2)
        plan = _plan(rng, count, distractors, p, colors, small=True, witness=witness)
    assert plan is not None, "bounded plan construction failed"
    arrays = []
    for i, m in enumerate(plan['modules']):
        if i < len(deltas):
            # Required CCW rotation from scattered input to the final placement.
            turns = (plan['placements'][i]['turns'] - deltas[i]) % 4
        else:
            turns = int(rng.integers(4))
        arrays.append(np.rot90(m['array'], turns))
    span = plan['high'] - plan['low'] + 1
    minimum = np.maximum(span, (18, 18))
    first_shape = np.array([int(rng.integers(v, 31)) for v in minimum])
    for attempt in range(13):
        shape = tuple(map(int, np.minimum(30, first_shape + 2 * (attempt // 3))))
        low = -plan['low']
        high = np.array(shape) - 1 - plan['high']
        if rng.random() < .35:
            origin = np.array([int(rng.choice((x, y))) for x, y in zip(low, high)])
        else:
            origin = np.array([int(rng.integers(x, y + 1)) for x, y in zip(low, high)])
        a = _pack(rng, shape, plan['root'], origin, arrays, bg)
        if a is not None and np.count_nonzero(a == bg) > a.size // 2:
            break
    else:
        shape = (30, 30)
        low, high = -plan['low'], 29 - plan['high']
        origin = np.array([int(rng.integers(x, y + 1)) for x, y in zip(low, high)])
        a = _pack(rng, shape, plan['root'], origin, arrays, bg, fallback=True)
    out = np.full_like(a, bg)
    pts = plan['cells'] + origin
    out[pts[:, 0], pts[:, 1]] = plan['values']
    assert np.array_equal(out, solve(a)), "input reconstruction differs from construction"
    return dict(input=a, output=out), bg


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    actions = list(map(int, rng.permutation((1, 2, 3))))
    rotation_order = list(map(int, rng.permutation(4)))
    witnesses = [(6, 6, 1), (6, 8, 2)]
    rng.shuffle(witnesses)
    result, seen, previous_bg = dict(train=[], test=[]), set(), None
    for i in range(nt + nq):
        split = 'train' if i < nt else 'test'
        count = 3 if i < 2 else int(rng.integers(2, 4)) if i < nt else 4
        distractors = int(rng.integers(1, min(2, 5 - count) + 1))
        prefix = (actions[:2] if i == 0 else
                  [actions[2], int(rng.choice((1, 2, 3)))] if i == 1 else [])
        deltas = rotation_order[:3] if i == 0 else rotation_order[3:] if i == 1 else ()
        witness = witnesses[i] if i < 2 else None
        for _ in range(8):
            pair, bg = _make_pair(rng, count, distractors, prefix, previous_bg, deltas, witness)
            key = _key(pair['input'])
            # Real input-derived order witnesses; no raster-order control flow.
            info = _parse(pair['input'])
            _, steps = _execute(pair['input'], info)
            order = [s['index'] for s in steps]
            if key not in seen and (i >= 2 or order != sorted(order)):
                break
        else:
            raise AssertionError('distinct non-raster input construction failed')
        result[split].append(pair)
        seen.add(key)
        previous_bg = bg
    rng.shuffle(result['train'])
    validate_generated_puzzle(result)
    return result


def _analyze(pair):
    a, out = _grid(pair['input']), _grid(pair['output'])
    assert a.shape == out.shape and not np.array_equal(a, out)
    info = _parse(a)
    expected, steps = _execute(a, info)
    assert np.array_equal(expected, out), "wrong relay assembly"
    bg, root = info['background'], info['root']['source']
    chosen = {s['index'] for s in steps}
    unused = [m for i, m in enumerate(info['modules']) if i not in chosen]
    unused_inlets = {m['inlet'] for m in unused}
    active_sides, turns, signatures, dims, headings = [], set(), set(), set(), set()
    # Conservation measured from input objects, not hidden construction choices.
    expected_counts = np.bincount(a[tuple(root['coords'].T)], minlength=10)
    branch_witness = False
    for step in steps:
        m, p = step['module'], step['placement']
        block = m['source']['array']
        expected_counts += np.bincount(block[block != bg], minlength=10)
        assert np.array_equal(out[tuple(p['cells'].T)], p['colors'])
        turns.add((m['undo'] + p['turns']) % 4)
        dims.add((m['height'], m['width'], m['neck']))
        headings.add(p['heading'])
        signatures.add((m['height'], m['width'], m['neck'], m['side']))
        inactive = {m['labels'][d] for d in (1, 2, 3) if d != m['side']}
        assert inactive & unused_inlets, "each selected module needs a real unused branch"
        branch_witness |= bool(inactive & unused_inlets)
    actual_counts = np.bincount(out[out != bg], minlength=10)
    expected_counts[bg] = 0
    assert np.array_equal(actual_counts, expected_counts), "selected material not conserved"
    for step in steps[:-1]:
        active_sides.append(step['module']['side'])
    return dict(info=info, steps=steps, count=len(steps), distractors=len(unused),
                routes=set(active_sides), turns=turns, dims=dims,
                signatures=signatures, headings=headings, branch=branch_witness,
                nonraster=[s['index'] for s in steps] != sorted(s['index'] for s in steps))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen, "duplicate input"
            seen.add(key)
            feature = _analyze(pair)
            assert feature['branch'], "missing genuine inactive-branch decoy"
            assert 1 <= feature['distractors'] <= 2
            assert feature['count'] in ((2, 3) if split == 'train' else (4,))
            analyses[split].append(feature)
    train = analyses['train']
    assert set().union(*(x['routes'] for x in train)) == {1, 2, 3}
    assert set().union(*(x['turns'] for x in train)) == {0, 1, 2, 3}
    assert len(set().union(*(x['dims'] for x in train))) >= 2
    assert any(x['nonraster'] for x in train)
    assert len({x['info']['background'] for x in train}) >= 2


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
    """Forty-two streamed seeds, meaningful invariants, and bounded regressions."""
    puzzles, shapes, geometries, paths = set(), set(), set(), set()
    body_colors, bg_colors, all_turns, entry_headings, edge_touches = set(), set(), set(), set(), 0
    pairs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.add(_digest(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                f = _analyze(pair)
                pairs += 1
                shapes.add(pair['input'].shape)
                geometries.update(f['signatures'])
                paths.add(tuple(s['module']['side'] for s in f['steps']))
                body_colors.add(f['info']['body'])
                bg_colors.add(f['info']['background'])
                all_turns.update(f['turns'])
                entry_headings.update((s['placement']['heading'] for s in f['steps']))
                a = pair['input'].copy()
                assert np.array_equal(solve(a), pair['output'])
                assert np.array_equal(a, pair['input']), "solve mutated input"
                if split == 'test' and seed < 3:
                    # The frame is local, and labels are equality roles.
                    for turns in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(a, turns)),
                                              np.rot90(pair['output'], turns))
                    palette = np.random.default_rng(9000 + seed).permutation(10)
                    assert np.array_equal(solve(palette[a]), palette[pair['output']])
                mask = pair['output'] != f['info']['background']
                edge_touches += bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    assert len(puzzles) == 42 and pairs == 210
    assert len(shapes) >= 20 and len(geometries) >= 12 and len(paths) >= 30
    assert bg_colors == body_colors == set(range(10))
    assert all_turns == entry_headings == set(range(4)) and edge_touches >= 10
    assert _digest(generate_puzzle(19)) == _digest(generate_puzzle(19))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(2026 + nt * 5 + nq, nt, nq)
            assert (len(p['train']), len(p['test'])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Semantic mutation: losing one output pixel must be caught.
    p = generate_puzzle(18, 2, 1)
    f = _analyze(p['test'][0])
    out = p['test'][0]['output']
    r, c = f['steps'][0]['placement']['cells'][0]
    out[r, c] = f['info']['background']
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged material accepted')
    # Wrong routing is observable independently of the sampler's output labels.
    p = generate_puzzle(20, 2, 1)
    f = _analyze(p['train'][0])
    first = f['steps'][0]['module']
    info = f['info']
    used = {s['index'] for s in f['steps']}
    inactive_inlets = {first['labels'][s] for s in (1, 2, 3) if s != first['side']}
    assert any(m['inlet'] in inactive_inlets for i, m in enumerate(info['modules']) if i not in used)
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(puzzles),
                input_shapes=len(shapes), module_geometries=len(geometries),
                route_sequences=len(paths), boundary_touch_pairs=edge_touches)
