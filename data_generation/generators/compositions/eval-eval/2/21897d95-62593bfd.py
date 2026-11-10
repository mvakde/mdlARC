"""Echoed color arrows: input-only frame transport and simultaneous recoloring.

Only the standard library and NumPy are used. No file I/O during generation,
solving, validation, or self_test; diagnostics retain aggregate hashes only.
"""
from collections import Counter
from hashlib import sha256

import numpy as np


DIR4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "noninteger grid")
    _need(bool(np.all((0 <= g) & (g <= 9))), "invalid color")
    return g.astype(np.int8, copy=False)


def _crop(mask):
    r, c = np.nonzero(mask)
    return mask[r.min():r.max()+1, c.min():c.max()+1].copy()


def _mask_key(mask):
    return mask.shape, mask.tobytes()


def _canonical(mask):
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _turn_point(point, shape, k):
    r, c = point
    h, w = shape
    for _ in range(k % 4):
        r, c, h, w = w-1-c, r, w, h
    return r, c


def _turn_direction(direction, k):
    r, c = direction
    for _ in range(k % 4):
        r, c = -c, r
    return r, c


def _rotation(a, b):
    ks = [k for k in range(4) if np.array_equal(np.rot90(a, k), b)]
    _need(len(ks) == 1, "active shape does not have a unique rotational frame")
    return ks[0]


def _stencil(center, direction):
    r, c = center
    dr, dc = direction
    return {(r, c), (r-dc, c+dr), (r+dc, c-dr), (r+dr, c+dc)}


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR8:
                p = r+dr, c+dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _scene(objects, shape, background):
    ids = np.full(shape, -1, dtype=np.int16)
    for i, obj in enumerate(objects):
        cells = {(obj['top']+int(r), obj['left']+int(c)) for r, c in np.argwhere(obj['mask'])}
        obj['cells'] = cells
        obj['key'] = _canonical(obj['mask'])
        for p in cells:
            _need(0 <= p[0] < shape[0] and 0 <= p[1] < shape[1], "object clipped")
            _need(ids[p] == -1, "overlapping objects")
            ids[p] = i
    return dict(objects=objects, ids=ids, bg=background, arrows=[])


def _read(value):
    g = _grid(value)
    values, counts = np.unique(g, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    _need(int(counts.max())*2 > g.size, "background is not a strict majority")
    objects, arrows = [], []
    for cells in _components(g != background):
        top, left = min(r for r, c in cells), min(c for r, c in cells)
        bottom, right = max(r for r, c in cells), max(c for r, c in cells)
        mask = np.zeros((bottom-top+1, right-left+1), dtype=bool)
        for r, c in cells:
            mask[r-top, c-left] = True
        counter = Counter(int(g[p]) for p in cells)
        body, count = counter.most_common(1)[0]
        _need(count*2 > len(cells), "body color is not a strict majority")
        different = {p for p in cells if int(g[p]) != body}
        index = len(objects)
        objects.append(dict(mask=mask, top=top, left=left, color=body))
        if not different:
            continue
        _need(len(cells) >= 10 and len(different) in (3, 4), "invalid annotation size")
        candidates = []
        for r, c in sorted(cells):
            for d in DIR4:
                footprint = _stencil((r, c), d)
                if not footprint <= cells:
                    continue
                ends = footprint - {(r, c)}
                inks = {int(g[p]) for p in ends}
                if len(inks) != 1:
                    continue
                ink = next(iter(inks))
                if ink in (body, background) or not different <= footprint:
                    continue
                center_color = int(g[r, c])
                candidates.append(dict(master=index, center=(r-top, c-left), direction=d,
                                       explicit=center_color != ink,
                                       payload=center_color if center_color != ink else None, ink=ink))
        _need(len(candidates) == 1, "T interpretation is ambiguous")
        arrows.append(candidates[0])
    _need(bool(arrows), "no arrow program")
    scene = _scene(objects, g.shape, background)
    scene['arrows'] = arrows
    inks = {a['ink'] for a in arrows}
    _need(len(inks) == 1, "different annotation inks within one example")
    ink = next(iter(inks))
    _need(all(o['color'] != ink for o in objects), "ink is also a body fill")
    keys = [objects[a['master']]['key'] for a in arrows]
    _need(len(keys) == len(set(keys)), "more than one program in an active class")
    for a in arrows:
        m = objects[a['master']]['mask']
        _need(len({_mask_key(np.rot90(m, k)) for k in range(4)}) == 4, "symmetric active mask")
    return scene


def _ray(scene, source, center, direction):
    """Distinct other objects in encounter order, continuing for diagnostics."""
    r, c = center
    dr, dc = direction
    r, c = r+2*dr, c+2*dc
    h, w = scene['ids'].shape
    hits, seen, blanks = [], set(), 0
    first_gap = None
    while 0 <= r < h and 0 <= c < w:
        target = int(scene['ids'][r, c])
        if target < 0:
            blanks += 1
        elif target != source and target not in seen:
            seen.add(target)
            hits.append(target)
            if first_gap is None:
                first_gap = blanks
        r, c = r+dr, c+dc
    return tuple(hits), blanks if first_gap is None else first_gap


def _echoes(scene, arrow):
    objects = scene['objects']
    master = objects[arrow['master']]
    result = []
    for i, obj in enumerate(objects):
        if obj['key'] != master['key']:
            continue
        k = _rotation(master['mask'], obj['mask'])
        local = _turn_point(arrow['center'], master['mask'].shape, k)
        center = obj['top']+local[0], obj['left']+local[1]
        direction = _turn_direction(arrow['direction'], k)
        hits, gap = _ray(scene, i, center, direction)
        result.append(dict(source=i, target=hits[0] if hits else None, hits=hits, gap=gap,
                           center=center, direction=direction, rotation=k,
                           physical=i == arrow['master'], explicit=arrow['explicit'],
                           color=arrow['payload'] if arrow['explicit'] else obj['color']))
    return result


def _emissions(scene):
    return [e for a in scene['arrows'] for e in _echoes(scene, a)]


def _assign(emissions):
    assignments = {}
    for e in emissions:
        t = e['target']
        if t is None:
            continue
        _need(t not in assignments or assignments[t] == e['color'], "conflicting incoming colors")
        assignments[t] = e['color']
    return assignments


def _paint(scene, assignments):
    out = np.full(scene['ids'].shape, scene['bg'], dtype=np.int8)
    for i, obj in enumerate(scene['objects']):
        out[scene['ids'] == i] = assignments.get(i, obj['color'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the full answer from this input alone; no training or seed needed."""
    scene = _read(input_grid)
    return _paint(scene, _assign(_emissions(scene)))


def _features(scene):
    objects = scene['objects']
    es = _emissions(scene)
    assignments = _assign(es)
    final = {i: assignments.get(i, o['color']) for i, o in enumerate(objects)}
    useful = [e for e in es if e['target'] is not None and e['color'] != objects[e['target']]['color']]
    physical = _assign(e for e in es if e['physical'])
    echo = any(final[i] != physical.get(i, o['color']) for i, o in enumerate(objects))
    virtual_only = _assign(e for e in es if not e['physical'])
    physical_effect = any(final[i] != virtual_only.get(i, o['color']) for i, o in enumerate(objects))
    marked_color_es = []
    for arrow in scene['arrows']:
        for e in _echoes(scene, arrow):
            marked_color_es.append(dict(e, color=objects[arrow['master']]['color'])
                                   if not e['explicit'] else e)
    try:
        broadcast = _assign(marked_color_es)
        own_color = any(final[i] != broadcast.get(i, o['color']) for i, o in enumerate(objects))
    except AssertionError:
        own_color = False
    # A precise wrong rule: let implicit emissions read their donors' final fills.
    final_read = [dict(e, color=final[e['source']]) if not e['explicit'] else e for e in es]
    try:
        bad = _assign(final_read)
        dependency = any(final[i] != bad.get(i, o['color']) for i, o in enumerate(objects))
    except AssertionError:
        dependency = False
    global_es = []
    for a in scene['arrows']:
        for e in _echoes(scene, a):
            hits, _ = _ray(scene, e['source'], e['center'], a['direction'])
            global_es.append(dict(e, target=hits[0] if hits else None))
    try:
        global_assign = _assign(global_es)
        rotated = any(final[i] != global_assign.get(i, o['color']) for i, o in enumerate(objects))
    except AssertionError:
        rotated = False
    mirror = False
    active_keys = {objects[a['master']]['key'] for a in scene['arrows']}
    for a in scene['arrows']:
        m = objects[a['master']]
        reflected = np.fliplr(m['mask'])
        rk = _canonical(reflected)
        if rk == m['key'] or rk in active_keys:
            continue
        for i, obj in enumerate(objects):
            if obj['key'] != rk or obj['color'] != m['color']:
                continue
            k = _rotation(reflected, obj['mask'])
            reflected_center = a['center'][0], m['mask'].shape[1]-1-a['center'][1]
            local = _turn_point(reflected_center, reflected.shape, k)
            direction = _turn_direction((a['direction'][0], -a['direction'][1]), k)
            hits, _ = _ray(scene, i, (obj['top']+local[0], obj['left']+local[1]), direction)
            payload = a['payload'] if a['explicit'] else obj['color']
            mirror |= bool(hits and hits[0] not in assignments and final[hits[0]] != payload)
    occlusion = any(len(e['hits']) >= 2 and e['color'] != final[e['hits'][1]]
                    for e in useful)
    multiplicities = Counter(o['key'] for o in objects)
    return dict(objects=len(objects), active=len(scene['arrows']),
                multiplicity=max(multiplicities[objects[a['master']]['key']] for a in scene['arrows']),
                active_sizes=tuple(sorted(multiplicities[objects[a['master']]['key']] for a in scene['arrows'])),
                echo=echo, physical_effect=physical_effect, own_color=own_color,
                dependency=dependency, rotated=rotated, mirror=mirror, occlusion=occlusion,
                implicit=any(not e['explicit'] for e in useful),
                explicit=any(e['explicit'] for e in useful),
                novel=any(a['explicit'] and a['payload'] not in {o['color'] for o in objects}
                          for a in scene['arrows']),
                useful=len(useful), open_rays=sum(e['target'] is None for e in es),
                fanin=len([e for e in es if e['target'] is not None])-len(assignments),
                directions={e['direction'] for e in es}, gaps={e['gap'] for e in es if e['target'] is not None},
                colors={o['color'] for o in objects}, background=scene['bg'],
                palette=len({o['color'] for o in objects}),
                changed=sum(final[i] != o['color'] for i, o in enumerate(objects)))


def _mask(rng, carrier=False, forbidden=()):
    for _ in range(160):
        if not carrier and rng.random() < .25:
            kind = int(rng.integers(3))
            if kind == 0:
                mask = np.ones((1, int(rng.integers(1, 6))), dtype=bool)
            else:
                h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
                mask = np.ones((h, w), dtype=bool)
                if kind == 2:
                    mask[1:-1, 1:-1] = False
            if _canonical(mask) not in forbidden:
                return mask
            continue
        if carrier:
            side = int(rng.choice([3, 4], p=[.85, .15]))
            limit = side+int(rng.integers(1, 3))
            cells = {(r, c) for r in range(side) for c in range(side)}
            target = side*side+int(rng.integers(2, 9))
        else:
            limit = int(rng.integers(2, 6))
            cells = {(0, 0)}
            target = int(rng.integers(1, min(15, limit*limit)+1))
        directions = DIR8 if rng.random() < .4 else DIR4
        for _ in range(target-len(cells)):
            candidates = set()
            for r, c in sorted(cells):
                for dr, dc in directions:
                    p = r+dr, c+dc
                    if p in cells:
                        continue
                    trial = cells | {p}
                    if (max(x for x, y in trial)-min(x for x, y in trial) < limit
                            and max(y for x, y in trial)-min(y for x, y in trial) < limit):
                        candidates.add(p)
            if not candidates:
                break
            ordered = sorted(candidates)
            cells.add(ordered[int(rng.integers(len(ordered)))])
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        mask = np.zeros((max(r for r, c in cells)-r0+1, max(c for r, c in cells)-c0+1), dtype=bool)
        for r, c in cells:
            mask[r-r0, c-c0] = True
        key = _canonical(mask)
        if key in forbidden:
            continue
        if carrier and (len({_mask_key(np.rot90(mask, k)) for k in range(4)}) != 4
                        or key == _canonical(np.fliplr(mask))):
            continue
        return mask
    # Guaranteed small asymmetric chiral alternatives for a finite fallback.
    for length in range(1, 6):
        mask = np.zeros((3+length, 3), dtype=bool)
        mask[:3, :] = True
        mask[3:, 0] = True
        if _canonical(mask) not in forbidden:
            return mask
    raise AssertionError("mask vocabulary exhausted")


def _pack(rng, objects, shape):
    blocked = set()
    for obj in sorted(objects, key=lambda o: -int(o['mask'].sum())):
        h, w = obj['mask'].shape
        possible = [(r, c) for r in range(shape[0]-h+1) for c in range(shape[1]-w+1)]
        success = False
        for j in rng.permutation(len(possible))[:200]:
            r, c = possible[int(j)]
            cells = {(r+int(y), c+int(x)) for y, x in np.argwhere(obj['mask'])}
            if cells & blocked:
                continue
            obj['top'], obj['left'] = r, c
            blocked.update((y+dy, x+dx) for y, x in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1))
            success = True
            break
        if not success:
            return False
    return True


def _program_options(rng, scene, indices, explicit, payload, ink, require_echo=True):
    master = int(rng.choice(indices))
    mask = scene['objects'][master]['mask']
    cells = set(map(tuple, np.argwhere(mask)))
    options = []
    for center in sorted(cells):
        for direction in DIR4:
            if not _stencil(center, direction) <= cells:
                continue
            a = dict(master=master, center=center, direction=direction,
                     explicit=explicit, payload=payload if explicit else None, ink=ink)
            es = _echoes(scene, a)
            try:
                _assign(es)
            except AssertionError:
                continue
            useful = [e for e in es if e['target'] is not None
                      and e['color'] != scene['objects'][e['target']]['color']]
            if useful and (not require_echo or any(not e['physical'] for e in useful)):
                options.append(a)
    rng.shuffle(options)
    return options


def _draw(scene):
    inp = _paint(scene, {})
    for a in scene['arrows']:
        obj = scene['objects'][a['master']]
        for r, c in _stencil(a['center'], a['direction']):
            inp[obj['top']+r, obj['left']+c] = a['ink']
        if a['explicit']:
            r, c = a['center']
            inp[obj['top']+r, obj['left']+c] = a['payload']
    out = _paint(scene, _assign(_emissions(scene)))
    # Reparse only the public grid: construction metadata cannot rescue it.
    recovered = _read(inp)
    _need(np.array_equal(solve(inp), out), "construction and input reader disagree")
    return dict(input=inp, output=out), recovered


def _random_pair(rng, kind):
    """Kinds curate evidence, never select different solving conventions."""
    mixed = kind == 'mixed'
    background, ink, payload, *palette = [int(v) for v in rng.permutation(10)]
    palette = palette[:int(rng.integers(3, 8))]
    objects, classes, forbidden = [], [], set()
    groups = [3 if kind == 'chain' or mixed else int(rng.integers(2, 4))]
    if mixed:
        groups.append(int(rng.integers(1, 3)))
    for group, count in enumerate(groups):
        base = _mask(rng, True, forbidden)
        forbidden.update((_canonical(base), _canonical(np.fliplr(base))))
        indices = []
        turns = [int(rng.integers(4)) for _ in range(count)]
        if count > 1:
            turns[1] = (turns[0]+int(rng.choice([1, 3]))) % 4
        for j, k in enumerate(turns):
            indices.append(len(objects))
            objects.append(dict(mask=np.rot90(base, k).copy(), color=palette[j % len(palette)]))
        classes.append(indices)
    mirror_index = None
    if kind == 'explicit' or rng.random() < .30:
        mirror_index = len(objects)
        base = objects[classes[0][0]]['mask']
        objects.append(dict(mask=np.rot90(np.fliplr(base), int(rng.integers(4))).copy(), color=palette[0]))
    extra = int(rng.integers(2, 5)) if not mixed else int(rng.integers(2, 4))
    while len(objects) < min(9, sum(groups)+(mirror_index is not None)+extra):
        mask = _mask(rng, False, forbidden)
        forbidden.add(_canonical(mask))
        objects.append(dict(mask=np.rot90(mask, int(rng.integers(4))).copy(), color=int(rng.choice(palette))))
    h, w = int(rng.integers(19, 31)), int(rng.integers(19, 31))
    if rng.random() < .2:
        w = h
    if not _pack(rng, objects, (h, w)):
        return None
    if sum(int(o['mask'].sum()) for o in objects)*2 >= h*w:
        return None
    scene = _scene(objects, (h, w), background)
    styles = [kind == 'explicit'] + ([True] if mixed else [])
    options = [_program_options(rng, scene, inds, style, payload, ink, require_echo=i == 0)
               for i, (inds, style) in enumerate(zip(classes, styles))]
    if any(not opts for opts in options):
        return None
    combinations = [(a,) for a in options[0]] if not mixed else [(a, b) for a in options[0] for b in options[1]]
    rng.shuffle(combinations)
    for programs in combinations[:160]:
        scene['arrows'] = list(programs)
        if mirror_index is not None:
            objects[mirror_index]['color'] = objects[programs[0]['master']]['color']
        try:
            f = _features(scene)
        except AssertionError:
            continue
        if not (f['echo'] and f['rotated'] and f['useful'] >= 2):
            continue
        if kind == 'chain' and not (f['dependency'] and f['own_color']):
            continue
        if kind == 'explicit' and not (f['mirror'] and f['occlusion'] and f['physical_effect']):
            continue
        if mixed and not (f['implicit'] and f['explicit']):
            continue
        return _draw(scene)
    return None


def _fallback(rng, kind):
    """Constructive bounded safety net; transformed as a whole for variation."""
    bg, ink, payload, a, b, c, d, e, f, unused = [int(v) for v in rng.permutation(10)]
    base = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=bool)
    objects = [dict(mask=base.copy(), top=2, left=2, color=a),
               dict(mask=np.rot90(base).copy(), top=2, left=12, color=b),
               dict(mask=np.rot90(base, 3).copy(), top=14, left=12, color=c),
               dict(mask=np.fliplr(base).copy(), top=8, left=2, color=a)]
    # Distinct inactive masks; two lie on the third carrier's ray.
    for mask, top, left, color in ((np.ones((1, 1), bool), 0, 13, d),
                                    (np.ones((1, 2), bool), 23, 14, e),
                                    (np.ones((2, 1), bool), 27, 14, f),
                                    (np.ones((2, 1), bool), 8, 0, d)):
        objects.append(dict(mask=mask, top=top, left=left, color=color))
    programs = [dict(master=0, center=(1, 1), direction=(0, 1), explicit=kind == 'explicit',
                     payload=payload if kind == 'explicit' else None, ink=ink)]
    if kind == 'mixed':
        mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool)
        objects.append(dict(mask=mask, top=8, left=21, color=e))
        programs.append(dict(master=8, center=(1, 1), direction=(0, -1), explicit=True,
                             payload=payload, ink=ink))
    scene = _scene(objects, (30, 30), bg)
    scene['arrows'] = programs
    pair, _ = _draw(scene)
    k = int(rng.integers(4))
    pair = {role: np.rot90(g, k).copy() for role, g in pair.items()}
    return pair, _read(pair['input'])


def _generate_pair(rng, kind):
    for _ in range(90):
        result = _random_pair(rng, kind)
        if result is not None:
            return result
    return _fallback(rng, kind)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _puzzle_digest(puzzle):
    h = sha256()
    for split in ('train', 'test'):
        h.update(bytes([len(puzzle[split])]))
        for p in puzzle[split]:
            for role in ('input', 'output'):
                g = np.asarray(p[role], dtype=np.uint8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test) or not 2 <= num_train <= 10
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("need integer counts 2..10 / 1..4 with num_train > num_test")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {'train': [], 'test': []}, set()
    kinds = ['chain', 'explicit'] + [str(rng.choice(['chain', 'explicit', 'mixed'], p=[.4, .35, .25]))
                                      for _ in range(num_train-2)]
    for split, schedule in (('train', kinds), ('test', ['mixed']*num_test)):
        for kind in schedule:
            for _ in range(8):
                pair, _ = _generate_pair(rng, kind)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct input budget exhausted")
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid counts")
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            _need(key not in seen, "duplicate input")
            seen.add(key)
            scene = _read(inp)
            es = _emissions(scene)
            assignments = _assign(es)
            _need(inp.shape == out.shape, "canvas changed")
            _need(np.array_equal(_paint(scene, assignments), out), "incorrect color transfer")
            _need(np.array_equal(inp != scene['bg'], out != scene['bg']), "occupied cells changed")
            # Every original component is painted wholly and retains all its cells.
            for i, obj in enumerate(scene['objects']):
                _need(bool(np.all(out[scene['ids'] == i] == assignments.get(i, obj['color']))),
                      "partial component recoloring")
            f = _features(scene)
            _need(f['echo'] and f['changed'] > 0, "parent ablation does not change the answer")
            features[split].append(f)
    train, test = features['train'], features['test']
    for name in ('physical_effect', 'dependency', 'own_color', 'rotated', 'mirror', 'occlusion',
                 'implicit', 'explicit', 'novel'):
        _need(any(f[name] for f in train), "missing teaching witness: "+name)
    _need(max(f['multiplicity'] for f in train) >= max(f['multiplicity'] for f in test),
          "tested multiplicity was never demonstrated")


def self_test():
    """Bounded 42-seed audit; retain digests/sets, never full puzzle histories."""
    puzzles, structures, shapes, bodies, backgrounds, counts = set(), set(), set(), set(), set(), set()
    directions = {'train': set(), 'test': set()}
    stats = Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = _puzzle_digest(p)
        puzzles.add(digest)
        if seed < 2:
            _need(digest == _puzzle_digest(generate_puzzle(seed)), "nondeterminism")
        for split in ('train', 'test'):
            for pair in p[split]:
                scene = _read(pair['input'])
                f = _features(scene)
                shapes.add(pair['input'].shape)
                backgrounds.add(f['background'])
                counts.add(f['objects'])
                directions[split].update(f['directions'])
                bodies.update(o['key'] for o in scene['objects'])
                # Color-normalized structural digest, without retaining arrays.
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair['input'].ravel())
                structures.add(sha256(bytes(pair['input'].shape)+normalized).digest())
                for name in ('mirror', 'dependency', 'occlusion', 'explicit', 'implicit', 'open_rays', 'fanin'):
                    stats[name] += bool(f[name])
                _need(np.array_equal(solve(pair['input'].copy()), pair['output']), "input-only solve failed")
    _need(len(puzzles) == 42 and len(structures) >= 150, "collapsed puzzle variation")
    _need(len(shapes) >= 20 and len(bodies) >= 80 and len(counts) >= 4, "collapsed structural vocabulary")
    _need(len(backgrounds) == 10, "missing palette roles")
    _need(all(ds == set(DIR4) for ds in directions.values()), "missing cardinal directions")
    _need(all(stats[n] >= 20 for n in ('mirror', 'dependency', 'occlusion', 'explicit', 'implicit')),
          "missing concept coverage")
    for kind in ('chain', 'explicit', 'mixed'):
        pair, scene = _fallback(np.random.default_rng(17), kind)
        f = _features(scene)
        _need(f['echo'] and f['rotated'], "fallback lacks echo/frame evidence")
        _need(kind != 'chain' or f['dependency'], "fallback lacks dependency")
        _need(kind != 'explicit' or (f['mirror'] and f['occlusion']), "fallback lacks negative evidence")
        _need(np.array_equal(solve(pair['input']), pair['output']), "fallback failed")
    # The host verifier exercises all 30 legal combinations; keep this audit
    # bounded to boundary requests rather than repeating its full matrix.
    for tr, te in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(4000+tr*10+te, tr, te)
        _need(len(p['train']) == tr and len(p['test']) == te, "explicit counts ignored")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Output corruption and transport-coordinate regression.
    p = generate_puzzle(91, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    marker = np.arange(20).reshape(4, 5)
    for k in range(4):
        for point in np.ndindex(marker.shape):
            _need(np.rot90(marker, k)[_turn_point(point, marker.shape, k)] == marker[point],
                  "coordinate rotation error")


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, boundary counts, bounded in-memory audit)')
