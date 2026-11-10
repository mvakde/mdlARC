"""ARC family: remembered-color rays dissolve compatible rectangular gates.

Only visible geometry and literal colors enter solve().  Generation uses planned
flights to paint an answer; validation reconstructs events from the actual grid.
"""
from __future__ import annotations

import numbers
import numpy as np


DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
GATE_SHAPES = ((3, 4), (4, 3), (3, 5), (5, 3), (3, 6), (6, 3),
               (3, 7), (7, 3), (4, 4), (4, 5), (5, 4), (5, 5),
               (4, 6), (6, 4), (5, 6), (6, 5), (4, 7), (7, 4))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(a, bg):
    todo = set(map(tuple, np.argwhere(a != bg)))
    result = []
    while todo:
        p = min(todo)
        todo.remove(p)
        cells = [p]
        for r, c in cells:
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in todo:
                    todo.remove(q)
                    cells.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > a.size / 2, "Background must be a strict majority"
    sources, mirrors, gates = [], [], []
    for cells in _components(a, bg):
        rs, cs = zip(*cells)
        r0, r1, c0, c1 = min(rs), max(rs)+1, min(cs), max(cs)+1
        h, w = r1-r0, c1-c0
        colors = {int(a[p]) for p in cells}
        if len(cells) == 3 and (h, w) == (2, 2) and len(colors) == 1:
            missing = next(iter({(r, c) for r in range(r0, r1)
                                 for c in range(c0, c1)} - set(cells)))
            elbow = (r0+r1-1-missing[0], c0+c1-1-missing[1])
            d = (elbow[0]-missing[0], elbow[1]-missing[1])
            sources.append(dict(cells=cells, elbow=elbow, direction=d,
                                color=int(a[elbow])))
        elif len(colors) == 1 and (h == 1 or w == 1) and len(cells) % 2 == 1:
            assert len(cells) == h*w
            mirrors.append(dict(cells=cells, midpoint=((r0+r1-1)//2, (c0+c1-1)//2),
                                axis=None if len(cells) == 1 else 0 if h == 1 else 1,
                                color=int(a[cells[0]])))
        elif h >= 3 and w >= 3 and len(cells) == h*w:
            border = np.concatenate((a[r0, c0:c1], a[r1-1, c0:c1],
                                     a[r0:r1, c0], a[r0:r1, c1-1]))
            assert len(set(map(int, border))) == 1, "Uniform gate border required"
            filler = int(border[0])
            values, n = np.unique(a[r0+1:r1-1, c0+1:c1-1], return_counts=True)
            marks = {int(v): int(k) for v, k in zip(values, n) if v != filler}
            gates.append(dict(cells=cells, box=(r0, r1, c0, c1), filler=filler,
                              labels=frozenset(marks), counts=marks))
        else:
            raise AssertionError("Unrecognizable or touching objects")
    assert sources, "At least one L source is required"
    if gates:
        assert len({g['filler'] for g in gates}) == 1
        filler = gates[0]['filler']
        assert filler not in {o['color'] for o in sources+mirrors}
    return dict(background=bg, sources=sources, mirrors=mirrors, gates=gates)


def _contact(m, p, d):
    r, c = m['midpoint']
    if m['axis'] in (None, 0) and p == (r-d[0], c):
        return 0
    if m['axis'] in (None, 1) and p == (r, c-d[1]):
        return 1
    return None


def _trace(a, *, memory="all", open_all=False, reflect=True):
    """Trace original objects; gate ownership makes ray ordering immaterial."""
    info = _parse(a)
    bg, sources, mirrors, gates = (info[k] for k in
                                   ('background', 'sources', 'mirrors', 'gates'))
    solids = {p for obj in sources+mirrors for p in obj['cells']}
    gate_at = {p: i for i, g in enumerate(gates) for p in g['cells']}
    paint, erase, owners, routes = {}, set(), {}, []
    for si, source in enumerate(sources):
        d, color = source['direction'], source['color']
        history, passed, visited = {color}, set(), set()
        p = (source['elbow'][0]+d[0], source['elbow'][1]+d[1])
        path, contacts, events = [], [], []
        while 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]:
            state = (p, d, frozenset(history), frozenset(passed))
            assert state not in visited, "Nonterminating ray"
            visited.add(state)
            if not reflect and p in solids:
                break  # ablation: unrefracted beams stop at opaque object bodies
            assert p not in solids, "A ray must meet a mirror face, not its body"
            hits = [(mi, _contact(m, p, d)) for mi, m in enumerate(mirrors)
                    if reflect and _contact(m, p, d) is not None]
            assert len(hits) <= 1, "Ambiguous midpoint contact"
            gi = gate_at.get(p)
            assert not (gi is not None and hits), "Gate and mirror compete"
            if gi is not None:
                assert gi not in owners or owners[gi] == si, "A gate belongs to at most one ray"
                owners[gi] = si
                if gi not in passed:
                    required = history if memory == 'all' else {source['color'], color}
                    positive = open_all or set(required) <= gates[gi]['labels']
                    events.append(dict(gate=gi, query=frozenset(history), positive=positive,
                                       position=p, color=color, epoch=len(contacts)))
                    if not positive:
                        break
                    passed.add(gi)
                    erase.add(gi)
            if hits:
                mi, axis = hits[0]
                color = mirrors[mi]['color']
                history.add(color)
                contacts.append((mi, axis, p, color))
                d = (-d[0], d[1]) if axis == 0 else (d[0], -d[1])
            assert p not in paint or paint[p] == color, "Conflicting ray colors"
            paint[p] = color
            path.append((p, color, len(contacts), frozenset(history)))
            p = (p[0]+d[0], p[1]+d[1])
        routes.append(dict(source=si, path=path, contacts=contacts, events=events,
                           stopped=bool(events and not events[-1]['positive'])))
    answer = a.copy()
    for gi in erase:
        r0, r1, c0, c1 = gates[gi]['box']
        answer[r0:r1, c0:c1] = bg
    for p, color in paint.items():
        answer[p] = color
    info.update(routes=routes, erased=erase, owners=owners, paint=paint)
    return answer, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the input. Malformed or conflicting scenes are rejected."""
    return _trace(_grid(input_grid))[0]


def _source_cells(elbow, d):
    r, c = elbow
    return {(r, c), (r-d[0], c), (r, c-d[1])}


def _mirror_cells(mid, axis, length):
    r, c = mid
    radius = length // 2
    return {(r, c+k) if axis == 0 else (r+k, c)
            for k in range(-radius, radius+1)}


def _place_object(a, cells, color, bg):
    h, w = a.shape
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    for r, c in cells:
        if a[r, c] != bg:
            return False
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (rr, cc) not in cells and 0 <= rr < h and 0 <= cc < w and a[rr, cc] != bg:
                return False
    for p in cells:
        a[p] = color
    return True


def _skeleton(rng, a, colors, nmirrors, needed_epochs, terminal_min):
    """Add a random flight plan and check it against all visible old mirrors."""
    bg = int(np.argmax(np.bincount(a.ravel(), minlength=10)))
    _, old_info = _trace(a) if np.any(a != bg) else (None, None)
    old_routes = {} if old_info is None else {
        s['elbow']: r['path'] for s, r in zip(old_info['sources'], old_info['routes'])}
    for _ in range(1600):
        trial = a.copy()
        d = DIRS[int(rng.integers(4))]
        elbow = (int(rng.integers(1, a.shape[0]-1)), int(rng.integers(1, a.shape[1]-1)))
        if not _place_object(trial, _source_cells(elbow, d), colors[0], bg):
            continue
        anchor, direction, planned = elbow, d, []
        base_axis = int(rng.integers(2))
        ok = True
        for k in range(nmirrors):
            distance = int(rng.integers(7, 10)) if k in needed_epochs else int(rng.integers(2, 6))
            axis = base_axis if rng.random() < .78 else 1-base_axis
            contact = (anchor[0]+distance*direction[0], anchor[1]+distance*direction[1])
            mid = (contact[0]+(direction[0] if axis == 0 else 0),
                   contact[1]+(direction[1] if axis == 1 else 0))
            length = int(rng.choice((1, 3, 3, 3, 5)))
            if not _place_object(trial, _mirror_cells(mid, axis, length), colors[k+1], bg):
                ok = False
                break
            planned.append(mid)
            direction = (-direction[0], direction[1]) if axis == 0 else (direction[0], -direction[1])
            anchor = contact
        if not ok:
            continue
        try:
            _, info = _trace(trial)
        except AssertionError:
            continue
        by_elbow = {s['elbow']: r for s, r in zip(info['sources'], info['routes'])}
        route = by_elbow[elbow]
        actual = [info['mirrors'][i]['midpoint'] for i, _, _, _ in route['contacts']]
        if actual != planned or any(by_elbow[e]['path'] != path for e, path in old_routes.items()):
            continue
        if sum(t[2] == nmirrors for t in route['path']) < terminal_min:
            continue
        if any(sum(t[2] == e for t in route['path']) < 6 for e in needed_epochs):
            continue
        return trial, elbow
    raise RuntimeError("Flight sampling exhausted")


def _in_box(p, box):
    r0, r1, c0, c1 = box
    return r0 <= p[0] < r1 and c0 <= p[1] < c1


def _gate_box(rng, a, route, all_routes, epoch, marks, bg, blocked=False):
    """Find an irregular footprint intersecting one constant-memory flight."""
    h, w = a.shape
    choices = [i for i, t in enumerate(route['path']) if t[2] == epoch]
    rng.shuffle(choices)
    shapes = [s for s in GATE_SHAPES if (s[0]-2)*(s[1]-2) >= len(marks)]
    rng.shuffle(shapes)
    seen = set()
    for i in choices:
        pr, pc = route['path'][i][0]
        for gh, gw in shapes:
            offsets = [(x, y) for x in range(gh) for y in range(gw)]
            rng.shuffle(offsets)
            for x, y in offsets:
                box = (pr-x, pr-x+gh, pc-y, pc-y+gw)
                if box in seen:
                    continue
                seen.add(box)
                r0, r1, c0, c1 = box
                if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
                    continue
                if np.any(a[max(0, r0-1):min(h, r1+1), max(0, c0-1):min(w, c1+1)] != bg):
                    continue
                hits = [j for j, t in enumerate(route['path']) if _in_box(t[0], box)]
                if (len(hits) < 2 or hits != list(range(hits[0], hits[-1]+1))
                        or any(route['path'][j][2] != epoch for j in hits)):
                    continue
                if blocked and hits[-1] > len(route['path'])-4:
                    continue
                if any(_in_box(t[0], box) for r in all_routes if r is not route for t in r['path']):
                    continue
                return box, hits[0]
    raise RuntimeError("No clear gate footprint on the chosen flight")


def _paint_gate(rng, a, box, marks, filler):
    r0, r1, c0, c1 = box
    a[r0:r1, c0:c1] = filler
    spots = [(r, c) for r in range(r0+1, r1-1) for c in range(c0+1, c1-1)]
    rng.shuffle(spots)
    for p, color in zip(spots, rng.permutation(marks)):
        a[p] = int(color)


def _decoy(rng, a, routes, marks, filler, bg):
    shapes = [s for s in GATE_SHAPES if (s[0]-2)*(s[1]-2) >= len(marks)]
    for _ in range(700):
        gh, gw = shapes[int(rng.integers(len(shapes)))]
        r0 = int(rng.integers(a.shape[0]-gh+1))
        c0 = int(rng.integers(a.shape[1]-gw+1))
        box = (r0, r0+gh, c0, c0+gw)
        if np.any(a[max(0, r0-1):min(a.shape[0], r0+gh+1),
                    max(0, c0-1):min(a.shape[1], c0+gw+1)] != bg):
            continue
        if any(_in_box(t[0], box) for r in routes for t in r['path']):
            continue
        _paint_gate(rng, a, box, marks, filler)
        return box
    raise RuntimeError("Decoy packing exhausted")


def _transform(a, code):
    a = np.rot90(a, code % 4)
    return (a[:, ::-1] if code >= 4 else a).copy()


def _make_pair(rng, package, rotation=None):
    # Plans are only a construction device; they are not stored with a puzzle.
    for attempt in range(100):
        palette = [int(c) for c in rng.permutation(10)]
        bg, filler, labels = palette[0], palette[1], palette[2:]
        if package == 'identity':
            beam_specs = [(2, (), 7), (3, (1,), 7)]
        elif package == 'memory':
            beam_specs = [(3, (1, 2), 7)]
        elif package == 'test':
            has_block = rng.random() < .55
            n = int(rng.choice((3, 3, 4)))
            beam_specs = [(n, (), 12 if has_block else 7)]
        else:
            n = int(rng.integers(4))
            gate_mode = str(rng.choice(('none', 'all', 'mixed', 'two_pass'), p=(.2, .25, .4, .15)))
            has_block = gate_mode == 'mixed'
            beam_specs = [(n, (), 12 if gate_mode in ('mixed', 'two_pass') else 7)]
        if package in ('identity', 'memory', 'test'):
            shape = (int(rng.integers(24, 31)), int(rng.integers(24, 31)))
        else:
            low = 24 if beam_specs[0][0] >= 2 else 14
            shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        a = np.full(shape, bg, dtype=int)
        elbows, beam_colors = [], []
        try:
            for bi, (n, epochs, tail) in enumerate(beam_specs):
                ordered = list(rng.permutation(labels))[:n+1]
                if package == 'identity' and bi == 1:
                    ordered[-1] = ordered[-2]  # visibly unchanged-color reflection
                if package == 'free' and n and (n == 3 or rng.random() < .5):
                    ordered[int(rng.integers(1, n+1))] = ordered[0]
                if package == 'test' and n == 4:
                    ordered[int(rng.integers(1, n+1))] = ordered[0]
                a, elbow = _skeleton(rng, a, ordered, n, epochs, tail)
                elbows.append(elbow)
                beam_colors.append(ordered)
            _, skeleton = _trace(a)
            routes = skeleton['routes']
            by_elbow = {s['elbow']: r for s, r in zip(skeleton['sources'], routes)}
            plans = []
            for bi, elbow in enumerate(elbows):
                route, sequence = by_elbow[elbow], beam_colors[bi]
                n = len(sequence)-1
                extra = [c for c in labels if c not in sequence]
                if package == 'identity':
                    q = sorted(set(sequence[:2]))
                    specs = [(1, q+[extra[0]], False)] if bi else []
                    specs.append((n, [c for c in dict.fromkeys(sequence)
                                      if c != sequence[bi]]+[extra[0]], True))
                elif package == 'memory':
                    q = sorted(set(sequence[:3]))
                    specs = [(1, sorted(set(sequence[:2])), False),
                             (2, q+[extra[0], q[0]], False),
                             (3, sequence[:-1]+[extra[0], extra[1], sequence[0]], True)]
                else:
                    q = sorted(set(sequence))
                    specs = [(n, q, False)]
                    if package == 'free' and gate_mode == 'none':
                        missing = q[int(rng.integers(len(q)))]
                        specs = [(n, [c for c in q if c != missing]+[extra[0]], True)]
                    elif package == 'free' and gate_mode == 'two_pass':
                        specs.append((n, q+[extra[0]], False))
                    if has_block:
                        missing = q[int(rng.integers(len(q)))]
                        specs.append((n, [c for c in q if c != missing]+[extra[0]], True))
                for epoch, marks, blocked in specs:
                    box, first = _gate_box(rng, a, route, routes, epoch, marks, bg, blocked)
                    _paint_gate(rng, a, box, marks, filler)
                    plans.append(dict(box=box, first=first, elbow=elbow, blocked=blocked,
                                      marks=marks, query=route['path'][first][3]))
            # At least one untouched fully compatible gate: a global filter fails.
            if package in ('identity', 'memory') or rng.random() < .7:
                q = sorted(set(beam_colors[0]))
                _decoy(rng, a, routes, q, filler, bg)
            if package == 'memory' or rng.random() < .35:
                _decoy(rng, a, routes, [], filler, bg)
            if rng.random() < .4:
                marks = list(rng.choice(labels, size=int(rng.integers(1, 6)), replace=True))
                _decoy(rng, a, routes, marks, filler, bg)
            # Paint from the known flights, with one causal cutoff per blocked ray.
            answer = a.copy()
            for plan in plans:
                assert (set(plan['query']) <= set(plan['marks'])) == (not plan['blocked'])
                if not plan['blocked']:
                    r0, r1, c0, c1 = plan['box']
                    answer[r0:r1, c0:c1] = bg
            for elbow in elbows:
                cutoff = min([p['first'] for p in plans if p['elbow'] == elbow and p['blocked']]
                             or [len(by_elbow[elbow]['path'])])
                for p, color, _, _ in by_elbow[elbow]['path'][:cutoff]:
                    answer[p] = color
            predicted, info = _trace(a)
            assert np.array_equal(answer, predicted)
            assert len(info['sources']) == len(beam_specs)
            assert len(info['erased']) == sum(not p['blocked'] for p in plans)
            if package == 'test':
                assert any(len(e['query']) == 4 and e['positive']
                           for r in info['routes'] for e in r['events'])
            code = int(rng.integers(8)) if rotation is None else rotation
            return dict(input=_transform(a, code), output=_transform(answer, code))
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError(f"Could not pack a {package} scene after bounded retries")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    first = _make_pair(rng, 'identity')
    first_info = _trace(first['input'])[1]
    for _ in range(20):
        second = _make_pair(rng, 'memory', 0)
        possible = []
        for code in range(8):
            candidate = {k: _transform(v, code) for k, v in second.items()}
            try:
                _training_coverage([first_info, _trace(candidate['input'])[1]])
            except AssertionError:
                continue
            possible.append(candidate)
        if possible:
            second = possible[int(rng.integers(len(possible)))]
            break
    else:
        raise RuntimeError("Could not select sufficient mirror demonstrations")
    puzzle['train'] = [first, second]
    seen.update(_key(p['input']) for p in puzzle['train'])
    # Rotations are a sampling preference; all directions use the same local rule.
    for split, total in (('train', num_train), ('test', num_test)):
        for i in range(len(puzzle[split]), total):
            package = 'free' if split == 'train' else 'test'
            for _ in range(12):
                pair = _make_pair(rng, package)
                k = _key(pair['input'])
                if k not in seen:
                    break
            else:
                raise RuntimeError("Distinct-input sampling exhausted")
            seen.add(k)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_coverage(infos):
    events = [(i, r, e) for i in infos for r in i['routes'] for e in r['events']]
    positives = [(i, r, e) for i, r, e in events if e['positive']]
    negatives = [(i, r, e) for i, r, e in events if not e['positive']]
    assert positives and negatives
    assert any(r['events'] and not r['events'][0]['positive'] for i in infos for r in i['routes'])
    assert any(any(a['positive'] and b['positive'] for a, b in zip(r['events'], r['events'][1:]))
               for i in infos for r in i['routes'])
    assert any(len(e['query']) == 2 and sum(i['gates'][e['gate']]['counts'].values()) == 2
               for i, _, e in positives)
    assert any(len(e['query']) == 3 and i['gates'][e['gate']]['labels'] > e['query']
               and any(n > 1 for n in i['gates'][e['gate']]['counts'].values())
               for i, _, e in positives)
    # Memory is needed even when the current color is visibly present.
    assert any(e['color'] in i['gates'][e['gate']]['labels']
               and i['sources'][r['source']]['color'] not in i['gates'][e['gate']]['labels']
               for i, r, e in negatives)
    assert any({i['sources'][r['source']]['color'], e['color']} <= i['gates'][e['gate']]['labels']
               for i, r, e in negatives)
    assert any(e['color'] not in i['gates'][e['gate']]['labels']
               and e['query']-{e['color']} <= i['gates'][e['gate']]['labels']
               for i, _, e in negatives)
    assert any(sum(i['gates'][e['gate']]['counts'].values()) ==
               sum(j['gates'][f['gate']]['counts'].values())
               and len(i['gates'][e['gate']]['labels']) == len(j['gates'][f['gate']]['labels'])
               for i, _, e in positives for j, _, f in negatives)
    assert max(len(e['query']) for _, _, e in positives) <= 3
    assert any(not g['labels'] for i in infos for g in i['gates'])
    assert any(gi not in i['owners'] and any(e['query'] <= g['labels']
                                           for r in i['routes'] for e in r['events'])
               for i in infos for gi, g in enumerate(i['gates']))
    axes = {axis for i in infos for r in i['routes'] for mi, axis, _, _ in r['contacts']
            if len(i['mirrors'][mi]['cells']) > 1}
    assert axes == {0, 1}, "Both mirror normals must be demonstrated"
    assert any(len(m['cells']) == 1 for i in infos for m in i['mirrors'])
    assert any(previous == color for i in infos for r in i['routes']
               for previous, color in zip([i['sources'][r['source']]['color']]+
                                         [t[3] for t in r['contacts']],
                                         [t[3] for t in r['contacts']]))
    assert len({s['direction'] for i in infos for s in i['sources']}) >= 2


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    seen, train_infos = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            expected, info = _trace(a)
            assert info['gates'], "A scene must have rectangular gates"
            assert np.array_equal(expected, out), "Incorrect gate erasure, cutoff, or ray color"
            assert not np.array_equal(a, out)
            assert set(np.unique(out)) <= set(np.unique(a))
            for o in info['sources']+info['mirrors']:
                assert all(out[p] == a[p] for p in o['cells'])
            for gi, gate in enumerate(info['gates']):
                if gi not in info['erased']:
                    assert all(out[p] == a[p] for p in gate['cells'])
                else:
                    assert all(out[p] == info['paint'].get(p, info['background']) for p in gate['cells'])
            if split == 'train':
                train_infos.append(info)
            else:
                assert any(e['positive'] and len(e['query']) == 4
                           for r in info['routes'] for e in r['events'])
    _training_coverage(train_infos)


def self_test():
    """Bounded 42-seed diversity audit and semantic counterfactuals."""
    puzzles = [generate_puzzle(s) for s in range(42)]
    signatures, structures, shapes, backgrounds, boxes, histories, lengths = (set() for _ in range(7))
    for puzzle in puzzles:
        signatures.add(tuple(_key(p['input']) for split in ('train', 'test') for p in puzzle[split]))
        endpoint_witness = transparent_witness = unlock_witness = reflection_witness = False
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                out, info = _trace(a)
                roles, encoded = {}, []
                for v in a.flat:
                    encoded.append(roles.setdefault(int(v), len(roles)))
                structures.add((a.shape, bytes(encoded)))
                shapes.add(a.shape)
                backgrounds.add(info['background'])
                boxes.update((g['box'][1]-g['box'][0], g['box'][3]-g['box'][2]) for g in info['gates'])
                lengths.update(len(m['cells']) for m in info['mirrors'])
                histories.update(len(e['query']) for r in info['routes'] for e in r['events'])
                if split == 'train':
                    endpoint_witness |= not np.array_equal(_trace(a, memory='endpoints')[0], out)
                    transparent_witness |= not np.array_equal(_trace(a, open_all=True)[0], out)
                    if len(info['sources']) == 1:
                        reflection_witness |= not np.array_equal(_trace(a, reflect=False)[0], out)
                    # Filling the sole missing query identity unlocks the stopped ray.
                    for route in info['routes']:
                        for e in route['events']:
                            g = info['gates'][e['gate']]
                            missing = e['query']-g['labels']
                            if e['positive'] or len(missing) != 1:
                                continue
                            r0, r1, c0, c1 = g['box']
                            spots = [(r, c) for r in range(r0+1, r1-1) for c in range(c0+1, c1-1)
                                     if a[r, c] == g['filler']]
                            if not spots:
                                spots = [(r, c) for r in range(r0+1, r1-1) for c in range(c0+1, c1-1)
                                         if int(a[r, c]) not in e['query']]
                            if not spots:
                                continue
                            changed = a.copy()
                            changed[spots[0]] = next(iter(missing))
                            new, ni = _trace(changed)
                            outside = np.ones(a.shape, dtype=bool)
                            outside[r0:r1, c0:c1] = False
                            assert e['gate'] in ni['erased']
                            unlock_witness |= bool(np.any(new[outside] != out[outside]))
        assert endpoint_witness and transparent_witness and unlock_witness and reflection_witness, (
            "Missing ablation/counterfactual witness", endpoint_witness,
            transparent_witness, unlock_witness, reflection_witness)
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 30 and backgrounds == set(range(10))
    assert len(boxes) >= 12 and histories >= {1, 2, 3, 4} and lengths >= {1, 3, 5}
    assert signatures
    p = generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], puzzles[0][s])
               for k in ('input', 'output'))
    for a in range(2, 11):
        for b in range(1, min(a-1, 4)+1):
            q = generate_puzzle(9100+a*10+b, a, b)
            assert len(q['train']) == a and len(q['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    # The validator must notice even one omitted trajectory/deletion cell.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    pair = bad['train'][0]
    position = tuple(np.argwhere(pair['input'] != pair['output'])[0])
    pair['output'][position] = pair['input'][position]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    _direct_regression()
    # Literal colors and whole-scene symmetry cannot be hidden conventions.
    rng = np.random.default_rng(127)
    for puzzle in puzzles[:3]:
        for pair in puzzle['train'][:2]:
            palette = rng.permutation(10)
            assert np.array_equal(solve(palette[pair['input']]), palette[pair['output']])
            for code in range(8):
                assert np.array_equal(solve(_transform(pair['input'], code)),
                                      _transform(pair['output'], code))


def _direct_regression():
    """Hand-specified route, erasure and a history-only blocking decision."""
    a = np.zeros((15, 23), dtype=int)
    for p in ((2, 2), (1, 2), (2, 1)):
        a[p] = 1
    a[7, 5:8] = 2
    a[0, 10:13] = 3
    a[2:5, 7:11] = 5
    a[3, 8:10] = [1, 2]
    a[4:7, 13:18] = 5
    a[5, 14:17] = [1, 3, 8]
    a[10:13, 1:6] = 5
    a[11, 2:5] = [1, 2, 3]
    expected = a.copy()
    expected[2:5, 7:11] = 0
    for p in ((3, 3), (4, 4), (5, 5)):
        expected[p] = 1
    for p in ((6, 6), (5, 7), (4, 8), (3, 9), (2, 10)):
        expected[p] = 2
    for p in ((1, 11), (2, 12), (3, 13)):
        expected[p] = 3
    assert np.array_equal(solve(a), expected)
    # Adding the remembered 2 opens the second gate; a tail appears beyond it.
    a[5, 16] = 2
    expected[4:7, 13:18] = 0
    for k in range(3, 12):
        expected[1+k, 11+k] = 3
    assert np.array_equal(solve(a), expected)


if __name__ == '__main__':
    self_test()
    print("42-seed self-test passed")
