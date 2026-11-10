"""Cargo discrepancy vents: align, compare, restore, and count all mismatches.

Standard library + NumPy only. The solver reads visible geometry and bitmaps;
construction paints outputs directly from sampled coordinates, not via solve().
"""
from collections import Counter
import json
import numbers
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # U, R, D, L


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected an integer ARC grid of size 1..30')
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(cells):
    rest, result = set(cells), []
    while rest:
        p = min(rest)
        rest.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in rest:
                        rest.remove(q)
                        group.add(q)
                        stack.append(q)
        result.append(group)
    return result


def _canonical(p, direction):
    r, c = p
    return ((r, c), (-c, r), (-r, -c), (c, -r))[direction]


def _world(p, direction):
    y, x = p
    return ((y, x), (x, -y), (-y, -x), (-x, y))[direction]


def _ray(shape, tip, direction, quantity):
    dr, dc = _DIRS[direction]
    r, c = tip
    result = []
    for _ in range(quantity):
        r, c = r + dr, c + dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            break
        result.append((r, c))
    return result


def _read_shell(frame, direction):
    """Recognize a thin closed pointed envelope by its cross-sections.

    The first widest row is an empty shoulder. Its following rows, up to
    the solid rear wall, form the rectangular cargo bay. Cap walks can
    differ between objects, and need not be mirror symmetric.
    """
    points = {_canonical(p, direction) for p in frame}
    y0, y1 = min(y for y, x in points), max(y for y, x in points)
    x0, x1 = min(x for y, x in points), max(x for y, x in points)
    w = x1 - x0 + 1
    if w < 4 or y1 - y0 < 4:
        return None
    rows = [sorted(x for y, x in points if y == r) for r in range(y0, y1 + 1)]
    if len(rows[0]) != 1 or rows[-1] != list(range(x0, x1 + 1)):
        return None
    tipx = rows[0][0]
    if not x0 < tipx < x1:
        return None
    lo = hi = tipx
    shoulder = None
    for i, row in enumerate(rows[:-1]):
        if not row or len(row) > 2:
            return None
        if row[0] > lo or row[-1] < hi or lo - row[0] > 1 or row[-1] - hi > 1:
            return None
        if len(row) == 1 and row[0] != tipx:
            return None
        lo, hi = row[0], row[-1]
        if (lo, hi) == (x0, x1) and shoulder is None:
            shoulder = y0 + i
    if shoulder is None or y1 - shoulder - 1 < 2:
        return None
    bay = [[_world((r, c), direction) for c in range(x0 + 1, x1)]
           for r in range(shoulder + 1, y1)]
    bbox = {_world((r, c), direction) for r in range(y0, y1 + 1)
            for c in range(x0, x1 + 1)}
    return dict(frame=frame, direction=direction, tip=_world((y0, tipx), direction),
                bay=bay, bbox=bbox, profile=tuple(tuple(x-x0 for x in row) for row in rows))


def _analyze(input_grid):
    """Input-only role, envelope, canonical stamp, and discrepancy inference."""
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    if len(values) != 3 or counts.max() * 2 <= a.size:
        raise ValueError('three roles and a majority background are required')
    bg = int(values[np.argmax(counts)])
    candidates = []
    for fc in (int(c) for c in values if c != bg):
        pc = next(int(c) for c in values if c not in (bg, fc))
        objects = []
        valid = True
        for frame in _components(map(tuple, np.argwhere(a == fc))):
            readings = [o for d in range(4) if (o := _read_shell(frame, d)) is not None]
            if len(readings) != 1:
                valid = False
                break
            obj = readings[0]
            bay_cells = {p for row in obj['bay'] for p in row}
            if any(a[p] != bg for p in obj['bbox'] - frame - bay_cells):
                valid = False
                break
            stamp = np.array([[a[p] == pc for p in row] for row in obj['bay']], dtype=bool)
            if any(a[p] not in (bg, pc) for p in bay_cells):
                valid = False
                break
            objects.append(dict(obj, stamp=stamp, bay_cells=bay_cells))
        if not valid or len(objects) < 3:
            continue
        occupied = set()
        for obj in objects:
            if occupied & obj['bbox']:
                valid = False
            occupied.update(obj['bbox'])
        if not valid or any(tuple(p) not in occupied for p in np.argwhere(a != bg)):
            continue
        if len({o['stamp'].shape for o in objects}) != 1:
            continue
        tally = Counter(_key(o['stamp']) for o in objects)
        mode, frequency = tally.most_common(1)[0]
        if frequency * 2 <= len(objects) or frequency == len(objects):
            continue
        master = next(o['stamp'].copy() for o in objects if _key(o['stamp']) == mode)
        if not 0 < master.sum() < master.size:
            continue
        rays = set()
        for obj in objects:
            diff = obj['stamp'] != master
            obj['diff'] = diff
            obj['quantity'] = int(diff.sum())
            obj['additions'] = int(np.count_nonzero(obj['stamp'] & ~master))
            obj['removals'] = int(np.count_nonzero(~obj['stamp'] & master))
            obj['ray'] = _ray(a.shape, obj['tip'], obj['direction'], obj['quantity'])
            obj['available'] = len(_ray(a.shape, obj['tip'], obj['direction'], 30))
            if any(a[p] != bg for p in obj['ray']) or rays & set(obj['ray']):
                valid = False
            # Entire bounding boxes stay outside all actual exports.
            if occupied & set(obj['ray']):
                valid = False
            rays.update(obj['ray'])
        if valid:
            candidates.append(dict(grid=a, background=bg, frame_color=fc, cargo_color=pc,
                                   objects=objects, master=master, rays=rays))
    if len(candidates) != 1:
        raise ValueError('input must have one unambiguous container interpretation')
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    output = info['grid'].copy()
    for obj in info['objects']:
        for r, row in enumerate(obj['bay']):
            for c, p in enumerate(row):
                output[p] = info['cargo_color'] if info['master'][r, c] else info['background']
        for p in obj['ray']:
            output[p] = info['cargo_color']
    return output


def _shell(rng, bh, bw, simple=False):
    """Build monotone boundary walks, followed by bay sides and rear wall."""
    width = bw + 2
    tip = int(rng.integers(1, width - 1))
    stem = int(rng.integers(0, 2))
    rows, lo, hi = [(tip, tip)] * (stem + 1), tip, tip
    while lo > 0 or hi < width - 1:
        dl = lo > 0 and (simple or hi == width - 1 or rng.random() < .75)
        dh = hi < width - 1 and (simple or lo == 0 or rng.random() < .75)
        if not dl and not dh:
            dl = lo > 0
            dh = not dl
        lo, hi = lo - int(dl), hi + int(dh)
        rows.append((lo, hi))
    shoulder = len(rows) - 1
    rows.extend([(0, width - 1)] * bh)
    frame = {(r, x) for r, (lo, hi) in enumerate(rows) for x in (lo, hi)}
    base = len(rows)
    frame.update((base, x) for x in range(width))
    bay = [[(r, c) for c in range(1, width - 1)] for r in range(shoulder + 1, base)]
    return dict(frame=frame, bay=bay, tip=(0, tip))


def _orient(spec, direction):
    all_points = spec['frame'] | {p for row in spec['bay'] for p in row}
    turned = [_world(p, direction) for p in all_points]
    r0, c0 = min(p[0] for p in turned), min(p[1] for p in turned)
    def f(p):
        r, c = _world(p, direction)
        return r-r0, c-c0
    frame = {f(p) for p in spec['frame']}
    return dict(frame=frame, bay=[[f(p) for p in row] for row in spec['bay']],
                tip=f(spec['tip']), direction=direction,
                height=max(r for r, c in frame)+1, width=max(c for r, c in frame)+1)


def _pattern(rng, bh, bw, witness=False):
    n = bh * bw
    # Include structured and unstructured stamps without requiring asymmetry.
    style = int(rng.integers(5))
    rr, cc = np.indices((bh, bw))
    if style == 0:
        p = (rr + cc) % 2 == 0
    elif style == 1:
        p = rr % 2 == 0
    elif style == 2:
        p = cc % 2 == 0
    elif style == 3:
        p = (rr == int(rng.integers(bh))) | (cc == int(rng.integers(bw)))
    else:
        p = rng.random((bh, bw)) < rng.uniform(.25, .75)
    ink = int(p.sum())
    if witness:
        # Witnesses need >=3 cells of both roles, and adjacent blank cells.
        if ink < 3 or n-ink < 3 or not any(
                not p[r,c] and not p[s,t] for r in range(bh) for c in range(bw)
                for s,t in ((r+1,c),(r,c+1)) if s < bh and t < bw):
            p = np.zeros((bh, bw), dtype=bool)
            p.flat[rng.choice(n, n//2, replace=False)] = True
            # Guarantee two adjacent blank sites while keeping >=3 ink sites.
            p[0, :2] = False
            if p.sum() < 3:
                available = np.flatnonzero(~p.ravel())
                available = [int(x) for x in available if x not in (0, 1)]
                p.flat[available[:3-int(p.sum())]] = True
    elif ink < 2 or ink > n-2:
        p = np.zeros((bh,bw), dtype=bool)
        p.flat[rng.choice(n, max(2, n//2), replace=False)] = True
    if witness == 2:
        # Explicitly teach tip-relative orientation, including chirality.
        # The fixed fallback is valid for all supported witness dimensions.
        for _ in range(32):
            transforms = [p[::-1,:],p[:,::-1],p[::-1,::-1]]
            if bh == bw:
                transforms += [p.T,p.T[::-1,::-1],np.rot90(p),np.rot90(p,3)]
            if all(not np.array_equal(p,t) for t in transforms):
                break
            p = np.zeros((bh,bw),dtype=bool)
            p.flat[rng.choice(n,n//2,replace=False)] = True
        else:
            p = np.zeros((bh,bw),dtype=bool)
            p[0,0] = p[0,1] = p[1,-1] = True
    return p


def _edits(rng, master, add, remove, adjacent=False):
    blank = [tuple(p) for p in np.argwhere(~master)]
    ink = [tuple(p) for p in np.argwhere(master)]
    selected = []
    if adjacent and add >= 2:
        pairs = [(p,q) for i,p in enumerate(blank) for q in blank[i+1:]
                 if abs(p[0]-q[0])+abs(p[1]-q[1]) == 1]
        if pairs:
            selected = list(pairs[int(rng.integers(len(pairs)))])
    others = [p for p in blank if p not in selected]
    selected += [others[int(i)] for i in rng.choice(len(others), add-len(selected), replace=False)]
    selected += [ink[int(i)] for i in rng.choice(len(ink), remove, replace=False)]
    return selected


def _pack(rng, specs, height, width):
    """Greedy random packing with silhouette guards and actual ray reservation."""
    # Larger boxes first; random tie-breaks avoid a repeated scene template.
    order = sorted(range(len(specs)), key=lambda i: -specs[i]['height']*specs[i]['width'])
    placed, boxes, guards, ray_guards = {}, set(), set(), set()
    for i in order:
        s = specs[i]
        possible = []
        for r in range(height-s['height']+1):
            for c in range(width-s['width']+1):
                tip = (r+s['tip'][0], c+s['tip'][1])
                room = len(_ray((height,width), tip, s['direction'], 30))
                q, mode = s['quantity'], s['clearance']
                if q and ((mode == 'free' and room <= q) or
                          (mode == 'exact' and room != q) or
                          (mode == 'clip' and not 0 < room < q) or
                          (mode == 'zero' and room != 0)):
                    continue
                possible.append((r,c,tip))
        for j in rng.permutation(len(possible)):
            r,c,tip = possible[int(j)]
            box = {(r+y,c+x) for y in range(s['height']) for x in range(s['width'])}
            ray = set(_ray((height,width),tip,s['direction'],s['quantity']+1)) if s['quantity'] else set()
            if box & (guards | ray_guards) or ray & (boxes | ray_guards):
                continue
            placed[i] = (r,c,tip)
            boxes.update(box)
            guards.update((y+dy,x+dx) for y,x in box for dy in (-1,0,1) for dx in (-1,0,1))
            ray_guards.update(ray)
            break
        else:
            return None
    return placed


def _make_pair(rng, dims, count, witness, palette):
    bh,bw = dims
    master = _pattern(rng,bh,bw,witness=witness)
    if witness == 1:
        plans = [(2,0,True), (0,3,False)]
    elif witness == 2:
        plans = [(2,2,False), (3,0,False)]
    else:
        nbad = int(rng.integers(1,(count-1)//2+1))
        plans = []
        for _ in range(nbad):
            q = int(rng.integers(1,min(8,master.size)+1))
            nadd = int(rng.integers(max(0,q-int(master.sum())),min(q,int((~master).sum()))+1))
            plans.append((nadd,q-nadd,False))
    # Which containers differ is randomized, independent of position/direction.
    rng.shuffle(plans)
    jobs = plans + [(0,0,False)]*(count-len(plans))
    rng.shuffle(jobs)
    specs = []
    directions = list(map(int,rng.integers(0,4,size=count)))
    if witness:
        # Tip-relative correspondence is demonstrated in both mandatory inputs.
        directions[0],directions[1] = 0,1
        directions[2],directions[3] = 2,3
        rng.shuffle(directions)
        odd_indices = [i for i,(a,r,_) in enumerate(jobs) if a+r]
        if directions[odd_indices[0]] == directions[odd_indices[1]]:
            # Swap with a normal container; retain all four demonstrated rotations.
            other = next(i for i in range(count) if directions[i] != directions[odd_indices[0]]
                         and i not in odd_indices)
            i = odd_indices[1]
            directions[i],directions[other] = directions[other],directions[i]
    for index,(add,remove,adjacent) in enumerate(jobs):
        changed = _edits(rng,master,add,remove,adjacent)
        s = _orient(_shell(rng,bh,bw,simple=bool(witness)),directions[index])
        s.update(changed=changed, quantity=len(changed), clearance='free')
        if witness == 2 and (add,remove) == (3,0):
            s['clearance'] = 'clip'
        if not witness and changed:
            s['clearance'] = str(rng.choice(['free','free','free','exact','clip','zero']))
        specs.append(s)
    if witness and len({_read_shell(s['frame'],s['direction'])['profile']
                        for s in specs if not s['quantity']}) < 2:
        # Make cargo identity independent of envelope silhouette in training.
        return None
    # At least one test/ordinary exception has a full, visibly ending ray.
    if not witness:
        next(s for s in specs if s['quantity'])['clearance'] = 'free'
    placed = None
    for attempt in range(36):
        if attempt < 16:
            height,width = int(rng.integers(18,31)),int(rng.integers(18,31))
        else:
            height = width = 30
        placed = _pack(rng,specs,height,width)
        if placed is not None:
            break
    if placed is None:
        return None
    bg,fc,pc = palette
    inp = np.full((height,width),bg,dtype=np.int8)
    out = inp.copy()
    for i,s in enumerate(specs):
        r0,c0,tip = placed[i]
        for r,c in s['frame']:
            inp[r0+r,c0+c] = out[r0+r,c0+c] = fc
        altered = master.copy()
        for p in s['changed']:
            altered[p] = ~altered[p]
        for r,row in enumerate(s['bay']):
            for c,(y,x) in enumerate(row):
                inp[r0+y,c0+x] = pc if altered[r,c] else bg
                out[r0+y,c0+x] = pc if master[r,c] else bg
        for p in _ray((height,width),tip,s['direction'],s['quantity']):
            out[p] = pc
    if np.count_nonzero(inp==bg)*2 <= inp.size:
        return None
    return {'input':inp,'output':out}


def _count(value, name, low, high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral):
        raise ValueError(name+' must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name+' is out of range')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ne = _count(num_train,'num_train',2,10),_count(num_test,'num_test',1,4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,numbers.Integral):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    shift_axis = int(rng.integers(2))
    shared_palette = bool(rng.random() < .35)
    palette = tuple(map(int,rng.choice(10,3,replace=False)))
    # The first two training inputs isolate addition/removal versus balanced edits.
    witness_dims = [3,int(rng.integers(3,5))]
    witness_dims[shift_axis] = 3
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',nt),('test',ne)):
        for index in range(count):
            witness = index+1 if split=='train' and index<2 else 0
            for attempt in range(24):
                dims = list(witness_dims) if witness else [int(rng.integers(2,5)),int(rng.integers(2,5))]
                if not witness:
                    dims[shift_axis] = int(rng.integers(2,4)) if split=='train' else int(rng.integers(4,6))
                number = 5 if witness else int(rng.integers(3,7))
                if not shared_palette:
                    palette = tuple(map(int,rng.choice(10,3,replace=False)))
                pair = _make_pair(rng,tuple(dims),number,witness,palette)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_invariants(pair):
    assert isinstance(pair,dict) and set(pair)=={'input','output'}
    a,b = _grid(pair['input']),_grid(pair['output'])
    assert a.shape==b.shape and not np.array_equal(a,b)
    info = _analyze(a)
    bg,pc,master = info['background'],info['cargo_color'],info['master']
    allowed = set(info['rays'])
    for obj in info['objects']:
        allowed.update(obj['bay_cells'])
        for p in obj['frame']:
            assert b[p]==a[p], 'frame changed'
        for r,row in enumerate(obj['bay']):
            for c,p in enumerate(row):
                assert b[p] == (pc if master[r,c] else bg), 'bay not restored to exact majority'
        for p in obj['ray']:
            assert b[p]==pc, 'ray color or length is wrong'
        if obj['quantity'] and obj['available']>obj['quantity']:
            endpoint = _ray(a.shape,obj['tip'],obj['direction'],obj['quantity']+1)[-1]
            # Constructor reserves a blank cell after a complete ray.
            assert b[endpoint]==bg, 'ray failed to stop at mismatch count'
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            if (r,c) not in allowed:
                assert b[r,c]==a[r,c], 'unrelated cell changed'
    assert np.array_equal(solve(a),b)
    return info


def _training_evidence(infos):
    odds = [o for info in infos for o in info['objects'] if o['quantity']]
    assert any(o['additions']>0 and o['removals']==0 for o in odds)
    assert any(o['removals']>0 and o['additions']==0 for o in odds)
    assert any(o['additions']==o['removals']>0 for o in odds), 'no equal-ink bitmap witness'
    full = [o for o in odds if o['available']>o['quantity']]
    assert {2,3,4} <= {o['quantity'] for o in full}, 'full counts do not identify cell counting'
    assert any(0<o['available']<o['quantity'] for o in odds), 'no demonstrated boundary clipping'
    assert any(len(_components(map(tuple,np.argwhere(o['diff']))))<o['quantity'] for o in full)
    assert any(int(o['stamp'].sum())!=o['quantity'] for o in full), 'counting entire cargo is not distinguished'
    assert len({o['direction'] for info in infos for o in info['objects']})==4
    assert any(len({o['profile'] for o in info['objects'] if not o['quantity']})>1
               for info in infos), 'no normal copies in different shell profiles'
    def asymmetric(p):
        ts = [p[::-1,:],p[:,::-1],p[::-1,::-1]]
        if p.shape[0]==p.shape[1]:
            ts += [p.T,p.T[::-1,::-1],np.rot90(p),np.rot90(p,3)]
        return all(not np.array_equal(p,t) for t in ts)
    assert any(asymmetric(info['master']) for info in infos), 'no asymmetric alignment witness'
    # Each literal fixed direction and fixed-position policy fails on witnesses.
    assert any(len({o['direction'] for o in info['objects'] if o['quantity']})>1 for info in infos) or len(
        {o['direction'] for o in odds})>1


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    assert 2<=nt<=10 and 1<=ne<=4 and nt>ne
    seen,infos = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            info = _pair_invariants(pair)
            key = _key(pair['input'])
            assert key not in seen, 'duplicate input'
            seen.add(key)
            infos[split].append(info)
    _training_evidence(infos['train'])
    train_dims = [i['master'].shape for i in infos['train']]
    assert any(all(i['master'].shape[axis]>max(d[axis] for d in train_dims)
                   for i in infos['test']) for axis in range(2)), 'missing bay-size shift'
    assert all(any(o['quantity'] and o['available']>o['quantity'] for o in i['objects'])
               for i in infos['test']), 'counting is hidden by clipping'


def _fingerprint(puzzle):
    return tuple((_key(p['input']),_key(p['output'])) for s in ('train','test') for p in puzzle[s])


def self_test():
    fingerprints,shapes,palettes,profiles,stamps = set(),set(),set(),set(),set()
    directions,counts,quantities,modes = set(),set(),set(),set()
    balanced,empty,both_signs,multi = 0,0,0,0
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(_fingerprint(p))
        for split in ('train','test'):
            for pair in p[split]:
                info = _pair_invariants(pair)
                shapes.add(info['grid'].shape)
                palettes.add((info['background'],info['frame_color'],info['cargo_color']))
                stamps.add(_key(info['master']))
                counts.add(len(info['objects']))
                multi += sum(o['quantity']>0 for o in info['objects'])>1
                for o in info['objects']:
                    directions.add(o['direction'])
                    profiles.add(o['profile'])
                    if not o['quantity']:
                        continue
                    q,room = o['quantity'],o['available']
                    quantities.add(q)
                    modes.add('zero' if room==0 else 'clip' if room<q else 'exact' if room==q else 'free')
                    balanced += o['additions']==o['removals']>0
                    empty += not o['stamp'].any()
                    both_signs += o['additions']>0 and o['removals']>0
                # Whole-scene rotations and arbitrary palette permutations must commute.
                if seed<4:
                    for rotation in range(4):
                        lookup = np.array([(c+seed+1)%10 for c in range(10)],dtype=np.int8)
                        assert np.array_equal(solve(lookup[np.rot90(pair['input'],rotation)]),
                                              lookup[np.rot90(pair['output'],rotation)])
    assert len(fingerprints)==42 and len(shapes)>=30 and len(palettes)>=40
    assert len(profiles)>=40 and len(stamps)>=65
    assert directions==set(range(4)) and {3,4,5,6}<=counts
    assert {1,2,3,4,5,6}<=quantities and modes=={'zero','clip','exact','free'}
    assert balanced>=42 and both_signs>=60 and multi>=84
    assert _fingerprint(generate_puzzle(947,7,3))==_fingerprint(generate_puzzle(947,7,3))
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(1000+nt*10+ne,nt,ne)
            assert (len(p['train']),len(p['test']))==(nt,ne)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Corrupt outputs, then remove an exceptional stamp without updating its label.
    base = generate_puzzle(0,2,1)
    for defect in ('ray','bay','shell','input'):
        bad = {s:[{k:np.array(v,copy=True) for k,v in pair.items()} for pair in base[s]] for s in base}
        pair = bad['train'][0]
        info = _analyze(pair['input'])
        obj = next(o for o in info['objects'] if o['quantity'])
        if defect=='ray':
            cell = obj['ray'][0]
        elif defect=='bay':
            cell = obj['bay'][0][0]
        elif defect=='shell':
            cell = min(obj['frame'])
        else:
            for r,row in enumerate(obj['bay']):
                for c,cell in enumerate(row):
                    pair['input'][cell] = info['cargo_color'] if info['master'][r,c] else info['background']
        if defect!='input':
            pair['output'][cell] = (int(pair['output'][cell])+1)%10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError,ValueError):
            pass
        else:
            raise AssertionError('validator accepted '+defect)
    return {'seeds':42,'distinct_puzzles':len(fingerprints),'input_shapes':len(shapes),
            'role_palettes':len(palettes),'shell_profiles':len(profiles),'canonical_stamps':len(stamps),
            'object_counts':sorted(counts),'mismatch_counts':sorted(quantities),'clearance_modes':sorted(modes)}


if __name__=='__main__':
    print(json.dumps(self_test(),separators=(',',':')))
