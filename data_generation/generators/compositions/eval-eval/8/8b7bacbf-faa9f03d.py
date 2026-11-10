"""Reclosed paint circuits: repair, replace terminal arms, dock, then supply faces.

Only standard-library modules and NumPy are used. No file access or saved state is
needed for generation or inference. Diagnostics are streamed by self_test; no
execution traces or seed corpus are retained.
"""
import hashlib
import json
import numpy as np


D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
REPAIR, TURN = 8, 9
HOST_SEEDS = (0,335624999,990597743,923758021,166622749,617363962,510352745,510217667,790961138,789685928,813350694,587270568,14827744,186497562,528516960,254092175,326434366,370020815,779002072,999886223,19147874,319577551,541372771,110205973,533995602,909215872,936153663,126403572,443182946,788205882,725312695,593816260,880180548,131099562,952826461,717793848,558537832,302610248,296848848,442239063,611531291,849661912)


def _add(p, d, n=1):
    return p[0] + n*d[0], p[1] + n*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _border(p, shape):
    return p[0] in (0, shape[0]-1) or p[1] in (0, shape[1]-1)


def _near(cells):
    return {_add(p, d) for p in cells for d in D4} - set(cells)


def _groups(cells):
    left, groups = set(cells), []
    while left:
        p = min(left)
        left.remove(p)
        group, todo = {p}, [p]
        for p in todo:
            for d in D4:
                q = _add(p, d)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
    return [_add(a, d, i) for i in range(max(abs(b[0]-a[0]), abs(b[1]-a[1]))+1)]


def _polyline(vertices):
    path = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        path.extend(_line(a, b)[1:])
    return path


def _ray(p, d, shape):
    result = []
    p = _add(p, d)
    while _inside(p, shape):
        result.append(p)
        p = _add(p, d)
    return result


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _faces(shape, wire):
    canvas = {(r, c) for r in range(shape[0]) for c in range(shape[1])}
    return [g for g in _groups(canvas-wire) if not any(_border(p, shape) for p in g)]


def _paint(shape, bg, color, wire, sources, fill=True):
    """Input-derived planar component/face ownership, independent of labels."""
    out = np.full(shape, bg, dtype=np.int8)
    labels = {p: i for i, g in enumerate(_groups(wire)) for p in g}
    supplies = {}
    for p, paint in sources.items():
        neighbors = {labels[q] for q in _near({p}) if q in labels}
        assert len(neighbors) == 1, 'source must have one circuit owner'
        owner = next(iter(neighbors))
        assert owner not in supplies or supplies[owner] == paint, 'conflicting sources'
        supplies[owner] = paint
    for p in wire:
        out[p] = color
    painted, empty, faces = [], [], []
    # Sources are endpoints outside faces; count them as occupied during flood.
    for face in _faces(shape, wire | set(sources)):
        owners = {labels[q] for q in _near(face) if q in labels}
        assert len(owners) == 1, 'nested/multiply bounded faces are outside this family'
        owner = next(iter(owners))
        paint = supplies.get(owner)
        faces.append(face)
        if paint is None:
            empty.append(face)
        else:
            painted.append(face)
            if fill:
                for p in face:
                    out[p] = paint
    for p, paint in sources.items():
        out[p] = paint
    return out, {'painted': painted, 'empty': empty, 'faces': faces, 'labels': labels}


def _read_input(value):
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert bg < 8 and counts[bg]*2 > a.size, 'background must be a strict majority'
    roles = [c for c in range(8) if c != bg and counts[c]]
    assert len(roles) >= 2
    color = max(roles, key=lambda c: counts[c])
    assert counts[color] > max(counts[c] for c in roles if c != color)
    wire = set(map(tuple, np.argwhere((a == color) | (a == REPAIR))))
    sources = {tuple(p): int(a[tuple(p)]) for p in np.argwhere(~np.isin(a, [bg,color,REPAIR,TURN]))}
    assert sources
    for p in sources:
        assert len(_near({p}) & wire) == 1, 'paint markers are terminal cells'
    return a, bg, color, wire, sources


def _edits(a, wire):
    edits = []
    for q in map(tuple, np.argwhere(a == TURN)):
        candidates = []
        for d in D4:
            pivot = _add(q, d, -1)
            start = _add(pivot, d, -1)
            if pivot not in wire or start not in wire:
                continue
            if not any(_add(pivot, t) in wire for t in D4 if t not in (d,(-d[0],-d[1]))):
                continue
            previous, current, walk = pivot, start, []
            for _ in range(a.size):
                if current in walk:
                    break
                walk.append(current)
                following = sorted((_near({current}) & wire) - {previous})
                if not following:
                    if _border(current, a.shape):
                        candidates.append({'pivot': pivot, 'd': d, 'old': set(walk), 'walk': [pivot]+walk})
                    break
                if len(following) != 1:
                    break
                previous, current = current, following[0]
        assert len(candidates) == 1, 'direction marker must select a unique terminal arm'
        edits.append(candidates[0])
    assert edits
    for i, e in enumerate(edits):
        for other in edits[:i]:
            assert not e['old'] & other['old']
            assert e['pivot'] not in other['old'] and other['pivot'] not in e['old']
    return edits


def _analyze(value, *, redirect=True, fill=True):
    a, bg, color, wire, sources = _read_input(value)
    edits = _edits(a, wire)
    removed = set().union(*(e['old'] for e in edits))
    retained = wire-removed if redirect else set(wire)
    new = set()
    if redirect:
        for edit in edits:
            ray = []
            for p in _ray(edit['pivot'], edit['d'], a.shape):
                assert p not in sources, 'rays may not erase paint endpoints'
                ray.append(p)
                if p in retained:
                    break
            assert ray
            edit['new'] = set(ray)
            edit['dock'] = ray[-1] in retained
            new.update(ray)
    final = retained | new
    out, face_info = _paint(a.shape, bg, color, final, sources, fill)
    info = dict(bg=bg, color=color, wire=wire, sources=sources, edits=edits,
                removed=removed, retained=retained, new=new, final=final, **face_info)
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only visible input roles, arms, docking contacts and final topology."""
    return _analyze(input_grid)[0]


def _ri(rng, low, high):
    return int(rng.integers(low, high+1))


def _pick(rng, values):
    return values[_ri(rng, 0, len(values)-1)]


def _polygon(rng, shape, bands):
    """An orthogonal row-monotone cycle with independent stepped side profiles."""
    h, w = shape
    heights = [_ri(rng, 3, 5) for _ in range(bands)]
    hh = sum(heights)
    if hh > h-10:
        return None
    ww = _ri(rng, 7, min(13, w-12))
    r0 = _ri(rng, 5, h-hh-5)
    c0 = _ri(rng, 5, w-ww-5)
    left = [c0+_pick(rng, (0,0,3)) for _ in heights]
    right = [c0+ww-_pick(rng, (0,0,3)) for _ in heights]
    if any(b-a < 5 for a,b in zip(left,right)):
        return None
    ys = [r0]
    for n in heights:
        ys.append(ys[-1]+n)
    vertices = [(ys[0],left[0]),(ys[0],right[0])]
    for i in range(bands):
        vertices.append((ys[i+1],right[i]))
        if i+1 < bands:
            vertices.append((ys[i+1],right[i+1]))
    vertices.append((ys[-1],left[-1]))
    for i in reversed(range(bands)):
        vertices.append((ys[i],left[i]))
        if i:
            vertices.append((ys[i],left[i-1]))
    # Remove duplicate and collinear intermediate vertices, including wraparound.
    vertices = list(dict.fromkeys(vertices[:-1]))
    changed = True
    while changed:
        changed = False
        for i in range(len(vertices)):
            a,b,c = vertices[i-1],vertices[i],vertices[(i+1)%len(vertices)]
            if a[0] == b[0] == c[0] or a[1] == b[1] == c[1]:
                vertices.pop(i)
                changed = True
                break
    path = _polyline(vertices+[vertices[0]])[:-1]
    cells = set(path)
    if len(cells) != len(path) or any(len(_near({p}) & cells) != 2 for p in cells):
        return None
    faces = _faces(shape, cells)
    if len(faces) != 1:
        return None
    return vertices, cells, faces[0]


def _old_arm(rng, pivot, d, forbidden, shape):
    back = (-d[0],-d[1])
    straight = _ray(pivot, back, shape)
    options = []
    for _ in range(18):
        distance = _ri(rng, 2, 5)
        elbow = _add(pivot, back, distance)
        side = _pick(rng, ((-back[1],back[0]),(back[1],-back[0])))
        if not _inside(elbow, shape):
            continue
        if rng.random() < .45:
            tail = _ray(elbow, side, shape)
            if tail:
                options.append(_polyline([pivot,elbow,tail[-1]])[1:])
        else:
            elbow2 = _add(elbow, side, _ri(rng,2,5))
            tail = _ray(elbow2, back, shape) if _inside(elbow2,shape) else []
            if tail:
                options.append(_polyline([pivot,elbow,elbow2,tail[-1]])[1:])
    if rng.random() < .3:
        options.insert(0, straight)
    else:
        options.append(straight)
    for walk in options:
        cells = set(walk)
        if not cells or len(cells) != len(walk) or cells & forbidden:
            continue
        if not all(_inside(p,shape) for p in cells):
            continue
        if (_near(cells) & forbidden) != {pivot}:
            continue
        full = cells | {pivot}
        if any(len(_near({p}) & full) > 2 for p in cells):
            continue
        return walk
    return None


def _attach_source(rng, retained, occupied, interior, shape):
    candidates = [(p,d) for p in sorted(retained) for d in D4]
    rng.shuffle(candidates)
    for p,d in candidates:
        length = _ri(rng,2,4)
        walk = [_add(p,d,i) for i in range(1,length+1)]
        cells = set(walk)
        if not all(_inside(q,shape) for q in cells) or cells & (occupied|interior):
            continue
        if (_near(cells) & occupied) != {p}:
            continue
        return set(walk[:-1]), walk[-1], p
    return None


def _satellite(rng, anchors, occupied, interior, shape):
    candidates = [(p,d) for p in sorted(anchors) for d in D4]
    rng.shuffle(candidates)
    for p,d in candidates:
        length = _ri(rng,2,3)
        height, width = _ri(rng,3,5), _ri(rng,3,5)
        q = _add(p,d,length+1)
        if d == (0,1):
            r0,c0 = q[0]-height//2,q[1]
        elif d == (0,-1):
            r0,c0 = q[0]-height//2,q[1]-width+1
        elif d == (1,0):
            r0,c0 = q[0],q[1]-width//2
        else:
            r0,c0 = q[0]-height+1,q[1]-width//2
        box = {(r,c) for r in range(r0,r0+height) for c in range(c0,c0+width)}
        if not all(_inside(v,shape) for v in box) or box & interior:
            continue
        inner = {(r,c) for r in range(r0+1,r0+height-1) for c in range(c0+1,c0+width-1)}
        edge = box-inner
        stem = {_add(p,d,i) for i in range(1,length+1)}
        added = edge | stem
        if (box|stem) & occupied or (_near(added)&occupied) != {p}:
            continue
        return added, inner
    return None


def _detached(rng, occupied, interior, shape):
    for _ in range(100):
        h,w = _ri(rng,3,5),_ri(rng,3,5)
        r0,c0 = _ri(rng,0,shape[0]-h),_ri(rng,0,shape[1]-w)
        box = {(r,c) for r in range(r0,r0+h) for c in range(c0,c0+w)}
        if box & (occupied|interior) or _near(box) & occupied:
            continue
        inner = {(r,c) for r in range(r0+1,r0+h-1) for c in range(c0+1,c0+w-1)}
        return box-inner, inner
    return None


def _make_pair(rng, gates, lesson, palette):
    # Finite rejection: every attempt contains <=900 cells and <=3 old arms.
    for attempt in range(160):
        shape = (_ri(rng,19,30),_ri(rng,19,30))
        bands = _ri(rng,1,3)
        polygon = _polygon(rng,shape,bands)
        if polygon is None:
            continue
        vertices, cycle, interior = polygon
        choices = [(a,b) for a,b in zip(vertices,vertices[1:]+vertices[:1]) if len(_line(a,b)) >= 5]
        choices += [(b,a) for a,b in list(choices)]
        rng.shuffle(choices)
        edits, used, missing = [], set(), set()
        for pivot,end in choices:
            if pivot in used or end in used:
                continue
            d = (int(np.sign(end[0]-pivot[0])),int(np.sign(end[1]-pivot[1])))
            arm = _old_arm(rng,pivot,d,cycle|set().union(*(e['old'] for e in edits)),shape)
            if arm is None:
                continue
            edits.append(dict(pivot=pivot,d=d,old=set(arm),walk=[pivot]+arm,end=end))
            used.update((pivot,end))
            missing.update(_line(pivot,end)[1:-1])
            if len(edits) == gates:
                break
        if len(edits) != gates:
            continue
        retained = cycle-missing
        old = set().union(*(e['old'] for e in edits))
        source = _attach_source(rng,retained,cycle|old,interior,shape)
        if source is None:
            continue
        tail, source_pos, anchor = source
        wire = retained|old|tail
        final = cycle|tail
        occupied = cycle|old|tail|{source_pos}
        # A second lesson feeds an initially disconnected satellite circuit.
        parts = _groups(retained)
        supplied = next(g for g in parts if anchor in g)
        anchors = retained-supplied if gates > 1 else retained
        attached_count = 1 if lesson < 2 else _ri(rng,0,2)
        extra_inners = []
        for _ in range(attached_count):
            attached = _satellite(rng,anchors,occupied,interior,shape)
            if attached is None:
                break
            extra, extra_inner = attached
            extra_inners.append(extra_inner)
            wire.update(extra)
            final.update(extra)
            occupied.update(extra|extra_inner)
            # A later loop may be supplied through an earlier loop's stem/shell.
            anchors = anchors | extra
        if len(extra_inners) != attached_count:
            continue
        detached = _detached(rng,occupied,interior,shape)
        if detached is None:
            continue
        edge, empty_inner = detached
        wire.update(edge)
        final.update(edge)
        if rng.random() < .45:
            more = _detached(rng,occupied|edge|empty_inner,interior,shape)
            if more is not None:
                wire.update(more[0]); final.update(more[0])
        bg, color, paint = palette
        a = np.full(shape,bg,dtype=np.int8)
        b = np.full(shape,bg,dtype=np.int8)
        for p in wire:
            a[p] = color
        for p in final:
            b[p] = color
        for p in interior | set().union(*extra_inners):
            b[p] = paint
        a[source_pos] = b[source_pos] = paint
        for e in edits:
            a[_add(e['pivot'],e['d'])] = TURN
        # Explicit repairs include erased arms, retained circuit cells, pivots
        # and docking targets; no invisible arbitrary missing-cell convention.
        repair_sites = {_pick(rng,sorted(old)),_pick(rng,sorted(retained))}
        if lesson == 1:
            repair_sites.update((edits[0]['pivot'],edits[0]['end']))
        else:
            repair_sites.add(_pick(rng,[e['pivot'] for e in edits]))
        candidates = sorted(wire-{source_pos})
        rng.shuffle(candidates)
        repair_sites.update(candidates[:_ri(rng,0,3)])
        for p in repair_sites:
            a[p] = REPAIR
        k = _ri(rng,0,3)
        reflect = bool(_ri(rng,0,1))
        pair = {role:np.rot90(np.fliplr(x) if reflect else x,k).copy() for role,x in (('input',a),('output',b))}
        try:
            prediction, info = _analyze(pair['input'])
            assert len(info['edits']) == gates
            assert all(e['dock'] for e in info['edits'])
            assert len(info['painted']) == 1+attached_count and info['empty']
            assert np.array_equal(prediction,pair['output'])
            # Both inherited mechanisms must affect region labels, not only ink.
            original, initial = _analyze(pair['input'],redirect=False)
            assert np.any((original == bg)&(pair['output'] == paint))
            if gates > 1:
                assert len(initial['painted']) == 0
        except AssertionError:
            continue
        return pair
    raise RuntimeError('bounded circuit construction exhausted; no ambiguous fallback emitted')


def _counts(num_train, num_test):
    def integer(x):
        return isinstance(x,(int,np.integer)) and not isinstance(x,(bool,np.bool_))
    if not integer(num_train) or not integer(num_test):
        raise ValueError('counts must be integers')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    return int(num_train),int(num_test)


def _key(a):
    return (a.shape,a.astype(np.int8).tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    # Literal roles are visible in every input; palettes vary across examples.
    paints = list(map(int,rng.permutation(8)))
    puzzle, seen = {'train':[],'test':[]}, set()
    for split,count in (('train',nt),('test',ne)):
        for i in range(count):
            gates = (1 if i == 0 else 2) if split == 'train' else 3
            if split == 'train' and i > 1:
                gates = _ri(rng,1,2)
            paint = paints[(i if split == 'train' else nt+i)%8]
            other = [c for c in range(8) if c != paint]
            rng.shuffle(other)
            for _ in range(8):
                pair = _make_pair(rng,gates,i if split == 'train' else 2,(other[0],other[1],paint))
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct-input retries exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    return puzzle


def _bends(walk):
    directions = [(b[0]-a[0],b[1]-a[1]) for a,b in zip(walk,walk[1:])]
    return sum(a != b for a,b in zip(directions,directions[1:]))


def measure_pair(pair):
    a, b = _grid(pair['input']),_grid(pair['output'])
    _, f = _analyze(a)
    _, initial = _analyze(a,redirect=False)
    repairs = set(map(tuple,np.argwhere(a == REPAIR)))
    return {'shape':list(a.shape),'background':f['bg'],'wire_color':f['color'],
            'paint':sorted(set(f['sources'].values())),'gates':len(f['edits']),
            'directions':[list(e['d']) for e in f['edits']],
            'old_bends':[_bends(e['walk']) for e in f['edits']],
            'old_lengths':[len(e['old']) for e in f['edits']],
            'new_lengths':[len(e['new']) for e in f['edits']],
            'repairs':len(repairs),'erased_repairs':len(repairs & f['removed']),
            'retained_repairs':len(repairs-f['removed']),
            'marked_pivots':sum(e['pivot'] in repairs for e in f['edits']),
            'marked_docks':sum(bool(e['new'] & repairs & f['retained']) for e in f['edits']),
            'initial_components':len(_groups(f['wire'])),
            'final_components':len(_groups(f['final'])),
            'painted_faces':len(f['painted']),'empty_faces':len(f['empty']),
            'initial_painted_faces':len(initial['painted']),
            'areas':sorted(map(len,f['painted'])),
            'fills':int(np.count_nonzero((a == f['bg']) & (b == next(iter(f['sources'].values())))))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen, features = set(), {'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            a,b = _grid(pair['input']),_grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            predicted,f = _analyze(a)
            assert np.array_equal(predicted,b), 'full-grid repair/docking/supply mismatch'
            assert not np.any(np.isin(b,[REPAIR,TURN]))
            assert all(b[p] == col for p,col in f['sources'].items())
            assert all(e['dock'] for e in f['edits'])
            assert f['painted'] and f['empty']
            assert not f['removed'] & f['final']
            assert all(b[p] == f['bg'] for p in f['removed'])
            no_fill,_ = _analyze(a,fill=False)
            no_turn,_ = _analyze(a,redirect=False)
            assert not np.array_equal(b,no_fill), 'A contribution is vacuous'
            assert np.any((no_turn == f['bg']) & (b != no_turn) & (b != f['color'])), 'B must change paint reachability/enclosure'
            extended = f['retained'] | set().union(*(
                set(_ray(e['pivot'],e['d'],a.shape)) for e in f['edits']))
            assert extended-f['final'], 'first-contact convention is untested'
            # Face fillings are uniform; retained geometry and outside are checked
            # by full prediction, not by an output-derived role assignment.
            features[split].append(measure_pair(pair))
    train = features['train']
    assert {1,2} <= {f['gates'] for f in train}
    assert len({c for f in train for c in f['paint']}) >= 2
    assert any(f['initial_painted_faces'] == 1 for f in train)
    assert any(f['initial_painted_faces'] == 0 and f['gates'] == 2 for f in train)
    assert any(f['marked_docks'] for f in train)
    assert all(f['erased_repairs'] and f['retained_repairs'] for f in train)
    assert all(f['gates'] == 3 for f in features['test'])


def self_test():
    """Bounded 42-seed streaming audit, symmetries, ablations and corruption."""
    measurements, signatures, shapes, structures = [],set(),set(),set()
    first = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        token = hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                a,b = pair['input'],pair['output']
                before = a.copy()
                token.update(a.tobytes()); token.update(b.tobytes())
                token.update(bytes(a.shape))
                measurements.append(measure_pair(pair))
                shapes.add(a.shape)
                roles = {}
                norm = bytes(roles.setdefault(int(c),len(roles)) for c in a.flat)
                structures.add((a.shape,hashlib.sha256(norm).digest()))
                transformed = np.rot90(np.fliplr(a))
                assert np.array_equal(solve(transformed),np.rot90(np.fliplr(b)))
                assert np.array_equal(a,before), 'solver mutated input'
        signatures.add(token.digest())
        if first is None:
            first = puzzle
    assert len(signatures) == 42 and len(structures) >= 200 and len(shapes) >= 40
    assert {tuple(d) for f in measurements for d in f['directions']} == set(D4)
    assert {0,1,2} <= {v for f in measurements for v in f['old_bends']}
    assert {1,2,3} == {f['gates'] for f in measurements}
    assert {f['background'] for f in measurements} == set(range(8))
    assert len({tuple(f['areas']) for f in measurements}) >= 40
    assert max(f['empty_faces'] for f in measurements) >= 2
    assert any(f['marked_docks'] and max(f['old_bends']) > 0 for f in measurements)
    repeat = generate_puzzle(HOST_SEEDS[0])
    assert all(_key(p[k]) == _key(q[k]) for s in first for p,q in zip(first[s],repeat[s]) for k in p)
    # Causal source copying and palette covariance keep literal 0..7 roles free.
    q = first['test'][0]
    _, f = _analyze(q['input'])
    paint = next(iter(f['sources'].values()))
    replacement = next(c for c in range(8) if c not in (f['bg'],f['color'],paint))
    recolored = np.where(q['input'] == paint,replacement,q['input'])
    assert np.array_equal(solve(recolored),np.where(q['output'] == paint,replacement,q['output']))
    palette = np.array([3,5,0,7,6,2,1,4,8,9],dtype=np.int8)
    assert np.array_equal(solve(palette[q['input']]),palette[q['output']])
    readonly = q['input'].copy()
    readonly.flags.writeable = False
    assert np.array_equal(solve(readonly),q['output'])
    # Every legal count combination is cheap enough to exercise once.
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            validate_generated_puzzle(generate_puzzle(117+nt*7+ne,nt,ne))
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.2),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count request accepted')
    # Whole-grid negative tests: face, erased arm, docking stroke, exterior.
    q = first['test'][0]
    _, f = _analyze(q['input'])
    sites = [min(f['painted'][0]),min(f['removed']),min(f['new'])]
    sites.append(next(p for p in np.ndindex(q['input'].shape) if q['input'][p] == q['output'][p] == f['bg']))
    for p in sites:
        bad = {'train':first['train'],'test':[dict(q)]}
        bad['test'][0]['output'] = q['output'].copy()
        bad['test'][0]['output'][p] = (int(q['output'][p])+1)%8
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupt answer accepted')
    return {'seeds':42,'pairs':len(measurements),'shapes':len(shapes),
            'normalized_structures':len(structures),'gate_counts':sorted({f['gates'] for f in measurements}),
            'old_bends':sorted({v for f in measurements for v in f['old_bends']}),
            'face_area_patterns':len({tuple(f['areas']) for f in measurements}),
            'note':'Generator-owned semantic checks; no independent solver or blind review.'}


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
