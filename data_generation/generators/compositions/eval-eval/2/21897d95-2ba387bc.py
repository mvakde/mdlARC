"""Color transfers through the ordinal adjacency of two scattered tile queues.

The solver reads only its argument. Construction metadata is never retained.
All semantic conventions are fixed; sampled profiles change only distribution.
"""
from __future__ import annotations

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _need(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _tile(size, color, background, kind):
    a = np.full((size, size), color, dtype=np.int8)
    if kind == 0:
        a[1:-1, 1:-1] = background
    return a


def _stencil(center, direction):
    r, c = center
    dr, dc = direction
    return ((r, c), (r-dc, c+dr), (r+dc, c-dr), (r+dr, c+dc))


def _sites(size, direction):
    return [(r, c) for r in range(1, size-1) for c in range(1, size-1)
            if all(1 <= y < size-1 and 1 <= x < size-1
                   for y, x in _stencil((r, c), direction))]


def _read_tile(tile, background):
    s = len(tile)
    color = int(tile[0, 0])
    rim = np.concatenate((tile[0], tile[-1], tile[:, 0], tile[:, -1]))
    _need(color != background and bool(np.all(rim == color)), "broken square rim")
    answers = []
    for kind, base in ((0, background), (1, color)):
        if np.all(tile[1:-1, 1:-1] == base):
            answers.append(dict(kind=kind, color=color, arrow=None))
            continue
        for direction in DIRECTIONS:
            for center in _sites(s, direction):
                stencil = _stencil(center, direction)
                ink = int(tile[stencil[1]])
                if ink in (background, color) or any(tile[p] != ink for p in stencil[1:]):
                    continue
                junction = int(tile[center])
                if junction == background:
                    continue
                restored = tile.copy()
                for p in stencil:
                    restored[p] = base
                if not np.all(restored[1:-1, 1:-1] == base):
                    continue
                answers.append(dict(kind=kind, color=color, arrow=dict(
                    direction=direction, center=center, ink=ink,
                    payload=None if junction == ink else junction)))
    _need(len(answers) == 1, "ambiguous or malformed tile interior")
    return answers[0]


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _need(all(np.all(edge == bg) for edge in
              (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "background border missing")
    todo = grid != bg
    objects = []
    while np.any(todo):
        r, c = map(int, np.argwhere(todo)[0])
        color, s = int(grid[r, c]), 0
        while c+s < grid.shape[1] and grid[r, c+s] == color:
            s += 1
        _need(5 <= s <= 8 and r+s < grid.shape[0] and c+s < grid.shape[1],
              "tile size or border invalid")
        tile = grid[r:r+s, c:c+s]
        obj = _read_tile(tile, bg)
        obj.update(top=r, left=c, size=s)
        objects.append(obj)
        todo[r:r+s, c:c+s] = False
    _need(objects and len({o['size'] for o in objects}) == 1, "one common size required")
    s = objects[0]['size']
    for i, a in enumerate(objects):
        for b in objects[:i]:
            _need(abs(a['top']-b['top']) > s or abs(a['left']-b['left']) > s,
                  "tiles need a complete background gap")
    _need({o['kind'] for o in objects} == {0, 1}, "both tile classes required")
    inks = {o['arrow']['ink'] for o in objects if o['arrow']}
    _need(len(inks) == 1, "one annotation ink and at least one arrow required")
    ink = next(iter(inks))
    _need(all(o['color'] != ink for o in objects), "ink also used as a tile color")
    _need(all(not o['arrow'] or o['arrow']['payload'] != ink for o in objects),
          "explicit ink payload is indistinguishable")
    return dict(grid=grid, background=bg, ink=ink, size=s, objects=objects)


def _queues(info, order='vertical'):
    objects = info['objects']
    if order == 'greedy':
        rows = [[None, None]]
        for i in sorted(range(len(objects)), key=lambda j: (objects[j]['top'], objects[j]['left'])):
            kind = objects[i]['kind']
            if rows[-1][kind] is not None:
                rows.append([None, None])
            rows[-1][kind] = i
        return [[row[k] for row in rows] for k in (0, 1)]
    key = ((lambda i: (objects[i]['left'], objects[i]['top'])) if order == 'horizontal'
           else (lambda i: (objects[i]['top'], objects[i]['left'])))
    return [sorted([i for i, o in enumerate(objects) if o['kind'] == kind],
                   key=key, reverse=order == 'reverse') for kind in (0, 1)]


def _target(info, queues, source, direction):
    kind = info['objects'][source]['kind']
    rank = queues[kind].index(source)
    dr, dc = direction
    if dr:
        rank += dr
    elif (kind == 0 and dc == 1) or (kind == 1 and dc == -1):
        kind = 1-kind
    else:
        return None  # off the left or right side of the imagined ladder
    return queues[kind][rank] if 0 <= rank < len(queues[kind]) else None


def _physical_target(info, source, direction):
    """Counterfactual: nearest physical tile strictly in the pointed half-plane."""
    a = info['objects'][source]
    dr, dc = direction
    options = [i for i, b in enumerate(info['objects']) if i != source and
               (b['top']-a['top'])*dr + (b['left']-a['left'])*dc > 0]
    return min(options, key=lambda i: (
        abs(info['objects'][i]['top']-a['top']) + abs(info['objects'][i]['left']-a['left']),
        info['objects'][i]['top'], info['objects'][i]['left'])) if options else None


def _assignments(info, order='vertical', physical=False, cascade=False, ignore_payload=False):
    queues = _queues(info, order)
    assignments = {}
    current = [o['color'] for o in info['objects']]
    for source, obj in enumerate(info['objects']):
        arrow = obj['arrow']
        if arrow is None:
            continue
        target = (_physical_target(info, source, arrow['direction']) if physical else
                  _target(info, queues, source, arrow['direction']))
        if target is None:
            continue
        payload = arrow['payload']
        if payload is None or ignore_payload:
            payload = current[source] if cascade else obj['color']
        if not cascade:
            _need(target not in assignments or assignments[target] == payload,
                  "conflicting incoming colors")
        assignments[target] = payload
        current[target] = payload
    return assignments


def _paint(info, assignments):
    out = np.full(info['grid'].shape, info['background'], dtype=np.int8)
    for i, obj in enumerate(info['objects']):
        r, c, s = obj['top'], obj['left'], obj['size']
        out[r:r+s, c:c+s] = _tile(s, assignments.get(i, obj['color']),
                                 info['background'], obj['kind'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the input's tiles, build rank adjacency, then paint simultaneously."""
    info = _parse(input_grid)
    return _paint(info, _assignments(info))


def _different(info, **alternative):
    try:
        alt = _paint(info, _assignments(info, **alternative))
    except AssertionError:
        return True  # an alternative also fails if it contradicts a demonstration
    return not np.array_equal(_paint(info, _assignments(info)), alt)


def _features(info):
    objects, queues = info['objects'], _queues(info)
    assignments = _assignments(info)
    originals = [o['color'] for o in objects]
    final = [assignments.get(i, c) for i, c in enumerate(originals)]
    transfers, ghosts, explicit = [], [], []
    for i, obj in enumerate(objects):
        if not obj['arrow']:
            continue
        arrow = obj['arrow']
        t = _target(info, queues, i, arrow['direction'])
        (ghosts if t is None else transfers).append((i, t, arrow))
        if t is not None and arrow['payload'] is not None:
            explicit.append(arrow)
    ties = any(len({objects[i]['top'] for i in q}) < len(q) for q in queues)
    return dict(
        counts=tuple(map(len, queues)), total=len(objects), size=info['size'],
        shape=tuple(info['grid'].shape), background=info['background'], ink=info['ink'],
        arrows=sum(o['arrow'] is not None for o in objects), ghosts=len(ghosts),
        boundary_payload=any(a['payload'] is not None and a['payload'] != objects[i]['color']
                             for i, t, a in ghosts),
        new_payload=any(a['payload'] not in originals for a in explicit),
        existing_payload=any(a['payload'] in originals and a['payload'] != objects[i]['color']
                             for i, t, a in transfers if a['payload'] is not None),
        directions={a['direction'] for i, t, a in transfers},
        changed=any(c != f for c, f in zip(originals, final)), ties=ties,
        locality=any(originals[i] == originals[j] and final[i] != final[j]
                     for i in range(len(objects)) for j in range(i)),
        dependency=any(a['payload'] is None and final[i] != originals[i]
                       for i, t, a in transfers),
        advanced_rank=any((queues[objects[i]['kind']].index(i) >= 4 or
                           queues[objects[t]['kind']].index(t) >= 4) and
                          final[t] != originals[t] for i, t, a in transfers),
        repeated=len(set(originals)) < len(originals),
        consistent_incoming=len(transfers) > len(assignments),
        physical_witness=_different(info, physical=True),
        greedy_witness=_different(info, order='greedy'),
        horizontal_witness=_different(info, order='horizontal'),
        reverse_witness=_different(info, order='reverse'),
        cascade_witness=_different(info, cascade=True),
        payload_witness=_different(info, ignore_payload=True),
        layout=tuple((o['top'], o['left'], o['kind']) for o in objects))


def _positions(rng, s, n, tie=False):
    """Feasible initial rectangles, then freely relocate them without a lattice."""
    capacity = 29 // (s+1)
    cols = 2 if tie else int(rng.integers(max(1, (n+capacity-1)//capacity),
                                       min(capacity, n)+1))
    rows = (n+cols-1)//cols
    _need(rows <= capacity, "unpackable tile count")
    min_h, min_w = rows*(s+1)+1, cols*(s+1)+1
    h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
    vgaps = rng.multinomial(h-min_h, np.full(rows+1, 1/(rows+1)))
    positions, top = [], 1+int(vgaps[0])
    for row in range(rows):
        count = min(cols, n-len(positions))
        hgaps = rng.multinomial(w-(count*(s+1)+1), np.full(count+1, 1/(count+1)))
        left = 1+int(hgaps[0])
        for _ in range(count):
            positions.append((top, left))
            left += s+1+int(hgaps[_+1])
        top += s+1+int(vgaps[row+1])
    rr, cc = np.indices((h-s-1, w-s-1))
    rr, cc = rr+1, cc+1
    for _ in range(2):
        for i in rng.permutation(n):
            i = int(i)
            if tie and i < 2:
                continue
            legal = np.ones(rr.shape, dtype=bool)
            if tie:
                legal &= rr > positions[0][0]
            for j, (r, c) in enumerate(positions):
                if i != j:
                    legal &= (np.abs(rr-r) > s) | (np.abs(cc-c) > s)
            sites = np.argwhere(legal)
            _need(len(sites) > 0, "lost a feasible placement")
            y, x = sites[int(rng.integers(len(sites)))]
            positions[i] = (int(rr[y, x]), int(cc[y, x]))
    return (h, w), positions


def _random_plan(rng, info, advanced=False):
    objects, queues = info['objects'], _queues(info)
    assignments, used = {}, set()
    candidates = [(i, d) for i in range(len(objects)) for d in DIRECTIONS
                  if _target(info, queues, i, d) is not None]
    rng.shuffle(candidates)
    first_candidates = candidates
    if advanced:
        first_candidates = [(i, d) for i, d in candidates if
            queues[objects[i]['kind']].index(i) >= 4 or
            queues[objects[_target(info, queues, i, d)]['kind']].index(
                _target(info, queues, i, d)) >= 4]
        _need(first_candidates, "no instruction can exercise the new ranks")
    # The first instruction is guaranteed to have a visible effect.
    first = next(((i, d) for i, d in first_candidates
                  if objects[i]['color'] != objects[_target(info, queues, i, d)]['color']),
                 first_candidates[0])
    want = int(rng.integers(1, len(objects)+1))
    order = [first[0]] + [int(i) for i in rng.permutation(len(objects)) if i != first[0]]
    for source in order:
        if len(used) >= want:
            break
        directions = [first[1]] if not used else [DIRECTIONS[int(i)] for i in rng.permutation(4)]
        for direction in directions:
            target = _target(info, queues, source, direction)
            payload = None
            if target is not None:
                if target in assignments:
                    payload = assignments[target]
                    if payload == objects[source]['color'] and rng.random() < .5:
                        payload = None
                elif rng.random() < .35 or (not used and objects[source]['color'] == objects[target]['color']):
                    choices = [c for c in range(10) if c not in
                               (info['background'], info['ink'], objects[target]['color'])]
                    payload = int(rng.choice(choices))
                delivered = objects[source]['color'] if payload is None else payload
                if target in assignments and assignments[target] != delivered:
                    continue
                if not used and delivered == objects[target]['color']:
                    continue
                assignments[target] = delivered
            objects[source]['arrow'] = dict(direction=direction, payload=payload)
            used.add(source)
            break
    return assignments


def _build(rng, size, counts, bg, ink, demo=None, demonstration_color=None, advanced=False):
    hcount, scount = counts
    n = hcount+scount
    shape, positions = _positions(rng, size, n, tie=demo == 1)
    kinds = [0]*hcount + [1]*scount
    if demo == 1:
        kinds = [1, 1] + list(rng.permutation([0]*hcount + [1]*(scount-2)))
    else:
        rng.shuffle(kinds)
    objects = [dict(top=r, left=c, size=size, kind=int(k), color=0, arrow=None)
               for (r, c), k in zip(positions, kinds)]
    # Use the same row-major traversal in construction and visible parsing;
    # the queue ranks themselves are computed separately for each class.
    objects.sort(key=lambda o: (o['top'], o['left']))
    info = dict(grid=np.full(shape, bg, dtype=np.int8), background=bg,
                ink=ink, size=size, objects=objects)
    palette = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, ink)])))
    queues = _queues(info)
    if demo is not None:
        position = 0 if demo == 0 else 2
        j = palette.index(demonstration_color)
        palette[position], palette[j] = palette[j], palette[position]
        a, b, c, d, e, payload = palette[:6]
        colors = ([a, b, b, e], [c, d]) if demo == 0 else ([a, b], [c, d, c, e])
        for q, values in zip(queues, colors):
            for i, color in zip(q, values):
                objects[i]['color'] = color
        h, s = queues
        plan = ([(h[0], (1, 0), payload), (h[1], (0, 1), None),
                 (s[1], (-1, 0), None), (s[0], (0, -1), None),
                 (h[2], (0, 1), a), (h[3], (1, 0), a)] if demo == 0 else
                [(h[0], (0, 1), None), (s[0], (1, 0), None),
                 (s[1], (0, -1), c), (h[1], (-1, 0), None),
                 (s[2], (0, -1), b), (s[3], (0, 1), a)])
        for i, direction, p in plan:
            objects[i]['arrow'] = dict(direction=direction, payload=p)
        assignments = _assignments(info)
    else:
        pool = palette[:int(rng.integers(2, 9))]
        for obj in objects:
            obj['color'] = int(rng.choice(pool))
        assignments = _random_plan(rng, info, advanced=advanced)
    inp, out = _paint(info, {}), _paint(info, assignments)
    for obj in objects:
        arrow = obj['arrow']
        if arrow is None:
            continue
        choices = _sites(size, arrow['direction'])
        center = choices[int(rng.integers(len(choices)))]
        for r, c in _stencil(center, arrow['direction']):
            inp[obj['top']+r, obj['left']+c] = ink
        if arrow['payload'] is not None:
            r, c = center
            inp[obj['top']+r, obj['left']+c] = arrow['payload']
    return dict(input=inp, output=out)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if not _integer(value) or not lo <= value <= hi:
            raise ValueError(name + " outside its legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    demonstration_color = int(rng.integers(10))
    bg = int(rng.choice([c for c in range(10) if c != demonstration_color]))
    ink = int(rng.choice([c for c in range(10) if c not in (bg, demonstration_color)]))
    vary_palette = bool(rng.integers(2))
    constant_size = int(rng.integers(5, 7)) if rng.random() < .25 else None
    first_size = constant_size or int(rng.integers(5, 7))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            demo = index if split == 'train' and index < 2 else None
            size = constant_size or (first_size if demo == 0 else
                                    (11-first_size if demo == 1 else int(rng.integers(5, 9))))
            capacity = (29//(size+1))**2
            if demo is not None:
                counts = (4, 2) if demo == 0 else (2, 4)
            elif split == 'train':
                counts = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
            elif size == 5 and rng.random() < .25:
                same = int(rng.integers(5, 9))
                counts = (same, same)
            else:
                long = int(rng.integers(5, min(8, capacity-1)+1))
                short = int(rng.integers(1, min(4, capacity-long)+1))
                counts = (long, short) if rng.integers(2) else (short, long)
            for attempt in range(96):
                local_bg, local_ink = bg, ink
                if vary_palette:
                    local_bg = int(rng.choice([c for c in range(10) if c != demonstration_color]))
                    local_ink = int(rng.choice([c for c in range(10) if c not in
                                               (local_bg, demonstration_color)]))
                pair = _build(rng, size, counts, local_bg, local_ink, demo, demonstration_color,
                              advanced=split == 'test')
                info = _parse(pair['input'])
                _need(np.array_equal(_paint(info, _assignments(info)), pair['output']),
                      "construction and input reconstruction disagree")
                if _key(pair['input']) in seen:
                    continue
                if demo is not None:
                    f = _features(info)
                    if not all(f[k] for k in ('changed', 'physical_witness', 'horizontal_witness',
                                              'reverse_witness', 'cascade_witness', 'payload_witness')):
                        continue
                    if demo == 1 and not f['greedy_witness']:
                        continue
                break
            else:
                raise RuntimeError("bounded demonstration construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "wrong puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, training, testing, roles = set(), [], [], {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "wrong pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            assignments = _assignments(info)
            _need(out.shape == inp.shape, "physical canvas must be preserved")
            _need(np.array_equal(out, _paint(info, assignments)), "wrong transfer, erasure or geometry")
            _need(info['ink'] not in out, "annotation ink survived")
            f = _features(info)
            _need(f['changed'], "pair has no effective transfer")
            (training if split == 'train' else testing).append(f)
            if split == 'train':
                for obj in info['objects']:
                    roles.setdefault(obj['color'], set()).add(obj['kind'])
    for name in ('new_payload', 'existing_payload', 'dependency', 'locality', 'ties', 'ghosts',
                 'boundary_payload',
                 'physical_witness', 'horizontal_witness', 'reverse_witness',
                 'greedy_witness', 'cascade_witness', 'payload_witness'):
        _need(any(f[name] for f in training), "missing training evidence: " + name)
    _need(set.union(*(f['directions'] for f in training)) == set(DIRECTIONS),
          "effective transfers must demonstrate all four directions")
    differences = [f['counts'][0]-f['counts'][1] for f in training]
    _need(min(differences) < 0 < max(differences), "both missing-partner sides must be taught")
    _need(any(v == {0, 1} for v in roles.values()), "shape roles must not be literal-color roles")
    _need(all(max(f['counts']) <= 4 for f in training), "training rank range exceeded")
    _need(all(max(f['counts']) >= 5 for f in testing), "missing rank extrapolation in test")
    _need(all(f['advanced_rank'] for f in testing), "test does not use a new rank effectively")


def self_test():
    """Bounded authored checks: 42 seeds, ablations, bounds, determinism, corruption."""
    signatures, shapes, layouts, patterns, sizes, bgs, inks, arrows = (set() for _ in range(8))
    features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                validate = solve(np.asarray(pair['input']).copy())
                _need(np.array_equal(validate, pair['output']), "solver mismatch")
                info = _parse(pair['input'])
                f = _features(info)
                features.append(f)
                shapes.add(f['shape']); layouts.add((f['size'], f['layout']))
                patterns.add(f['counts']); sizes.add(f['size'])
                bgs.add(f['background']); inks.add(f['ink']); arrows.add(f['arrows'])
                clean = _paint(info, {})
                _need(not np.array_equal(clean, pair['output']), "omitting A made no difference")
        if seed < 2:
            again = generate_puzzle(seed)
            _need(all(_key(p[k]) == _key(q[k]) for s in puzzle
                      for p, q in zip(puzzle[s], again[s]) for k in ('input', 'output')),
                  "seed is nondeterministic")
    _need(len(signatures) == 42 and len(layouts) >= 180, "collapsed geometry")
    _need(len(shapes) >= 40 and len(patterns) >= 20, "collapsed dimensions/counts")
    _need(sizes == {5, 6, 7, 8} and bgs == set(range(10)) and inks == set(range(10)),
          "missing size or palette variation")
    _need(len(arrows) >= 8 and min(arrows) == 1, "collapsed instruction counts")
    _need(any(f['consistent_incoming'] for f in features), "no agreeing multiple incoming transfers")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1234, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # A input-only negative control for contradictory incoming instructions.
    p = generate_puzzle(0, 2, 1)['train'][0]
    info = _parse(p['input'])
    h, s = _queues(info)
    objects = info['objects']
    # H0 -> H1 already has a singleton payload; make H2 -> H1 disagree.
    conflict = p['input'].copy()
    obj = objects[h[2]]
    r, c, z = obj['top'], obj['left'], obj['size']
    conflict[r:r+z, c:c+z] = _tile(z, obj['color'], info['background'], obj['kind'])
    center = _sites(z, (-1, 0))[0]
    for y, x in _stencil(center, (-1, 0)):
        conflict[r+y, c+x] = info['ink']
    try:
        solve(conflict)
    except AssertionError as error:
        _need('conflicting' in str(error), "conflict failed for an unrelated reason")
    else:
        raise AssertionError("contradictory incoming transfers accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds and all legal explicit counts')
