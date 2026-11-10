"""Branch casting: repaired terminal tails become half-turned Cartesian stencils.

Only standard-library modules and NumPy are used. The solver reads one input;
no puzzle conventions, random choices, or labels are needed at inference time.
"""
from __future__ import annotations

import copy
import json
import numpy as np

BG, FRAME, REPAIR, DIRECTION = 0, 8, 2, 4
PAYLOAD = (1, 3, 5, 6, 7, 9)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(a, b):
    a, b = _integer(a, "num_train"), _integer(b, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "color bounds"
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    return [(a[0] + i * dr, a[1] + i * dc)
            for i in range(max(abs(b[0] - a[0]), abs(b[1] - a[1])) + 1)]


def _walk(points):
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _connected(cells):
    if not cells:
        return False
    seen, stack = {min(cells)}, [min(cells)]
    while stack:
        for q in _neighbors(stack.pop(), cells):
            if q not in seen:
                seen.add(q)
                stack.append(q)
    return seen == cells


def _bridge(cells):
    """Simultaneous one-cell straight completion; never extrapolate endpoints."""
    result = set(cells)
    for r, c in cells:
        for dr, dc in ((1, 0), (0, 1)):
            if (r + 2 * dr, c + 2 * dc) in cells:
                result.add((r + dr, c + dc))
    return result


def _box(cells):
    a = np.array(sorted(cells), dtype=int)
    lo, hi = a.min(axis=0), a.max(axis=0)
    return tuple(map(int, lo)), tuple(map(int, hi))


def _stencil(pivot, old):
    cells = set(old) | {pivot}
    lo, hi = _box(cells)
    dims = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1
    assert dims[0] % 2 and dims[1] % 2, "odd stencil box required"
    center = (lo[0] + hi[0]) // 2, (lo[1] + hi[1]) // 2
    # A half-turn about the bounding-box center negates both offsets.
    offsets = {(center[0] - r, center[1] - c) for r, c in cells}
    return dims, center, offsets


def _parse(inp):
    h, w = inp.shape
    assert h >= 7 and w >= 7
    assert all(inp[p] == FRAME for p in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)))
    border = np.concatenate((inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]))
    assert set(map(int, border)) <= {FRAME, *PAYLOAD}, "reserved colors on frame"
    assert np.array_equal(inp[0, 1:-1], inp[-1, 1:-1]), "paired column labels"
    assert np.array_equal(inp[1:-1, 0], inp[1:-1, -1]), "paired row labels"
    work = inp.copy()
    assert not np.any(work[1:-1, 1:-1] == FRAME), "frame color is reserved"
    colors = sorted(set(map(int, np.unique(work[1:-1, 1:-1]))) - {BG, REPAIR, DIRECTION})
    assert colors and set(colors) <= set(PAYLOAD)
    interior = lambda p: 1 <= p[0] < h-1 and 1 <= p[1] < w-1
    pending = {tuple(map(int, p + 1)) for p in np.argwhere(work[1:-1, 1:-1] == REPAIR)}
    repairs = set(pending)
    for _ in range(len(pending) + 1):
        changes = []
        for p in sorted(pending):
            candidates = {int(work[q]) for d in DIRS
                          if interior(q := _add(p, d)) and int(work[q]) in colors}
            if len(candidates) == 1:
                changes.append((p, candidates.pop()))
        if not changes:
            break
        for p, color in changes:
            work[p] = color
            pending.remove(p)
    assert not pending, "repair marker lacks a unique visible color witness"
    layers = {c: _bridge({tuple(map(int, p + 1)) for p in
                           np.argwhere(work[1:-1, 1:-1] == c)}) for c in colors}
    for c, cells in layers.items():
        assert all(interior(p) for p in cells)
        assert _connected(cells), "one connected source network per color"
        assert all(len(_neighbors(p, cells)) <= 3 for p in cells)
        assert sum(len(_neighbors(p, cells)) for p in cells) == 2 * (len(cells)-1), "acyclic source"
        for p in cells:
            assert int(work[p]) in (BG, c), "no crossing or direction cell may be bridged"
        for other, s in layers.items():
            if c != other:
                assert not cells & s and not any(_neighbors(p, s) for p in cells), "separate source colors"
    rows = {c: [r for r in range(1, h-1) if inp[r, 0] == c] for c in PAYLOAD}
    cols = {c: [x for x in range(1, w-1) if inp[0, x] == c] for c in PAYLOAD}
    edits = []
    for raw in np.argwhere(inp[1:-1, 1:-1] == DIRECTION):
        q = tuple(map(int, raw + 1))
        candidates = []
        for d in DIRS:
            p = q[0]-d[0], q[1]-d[1]
            start = p[0]-d[0], p[1]-d[1]
            for c, cells in layers.items():
                if p not in cells or start not in cells or q in cells:
                    continue
                retained = [x for x in _neighbors(p, cells) if x != start]
                if not retained or not all(x[0] == p[0] if d[0] else x[1] == p[1] for x in retained):
                    continue
                old, prev, at = [], p, start
                for _ in range(inp.size):
                    if at in old:
                        break
                    old.append(at)
                    nxt = [x for x in _neighbors(at, cells) if x != prev]
                    if not nxt:
                        dims, center, offsets = _stencil(p, old)
                        candidates.append(dict(color=c, pivot=p, q=q, direction=d, old=old,
                                               dims=dims, center=center, offsets=offsets,
                                               rows=rows[c], cols=cols[c]))
                        break
                    if len(nxt) != 1:
                        break
                    prev, at = at, nxt[0]
        assert len(candidates) == 1, "direction marker must select one terminal tail"
        edit = candidates[0]
        edit['active'] = bool(edit['rows'] and edit['cols'])
        edit['stamps'] = []
        if edit['active']:
            for r in edit['rows']:
                for c in edit['cols']:
                    stamp = {(r+dy, c+dx) for dy, dx in edit['offsets']}
                    assert all(interior(x) for x in stamp), "casts may not clip or change the frame"
                    edit['stamps'].append(stamp)
        edits.append(edit)
    assert edits, "at least one directional branch is present"
    assert len({e['color'] for e in edits}) == len(edits), "at most one direction marker per source color"
    edits.sort(key=lambda e: e['pivot'])
    return dict(layers=layers, edits=edits, rows=rows, cols=cols, repairs=repairs,
                gaps={p for s in layers.values() for p in s if inp[p] == BG})


def _predict(inp, parsed=None, alternative=None):
    p = _parse(inp) if parsed is None else parsed
    out = inp.copy()
    out[1:-1, 1:-1] = BG
    active = [e for e in p['edits'] if e['active']]
    deleted = {e['color']: set(e['old']) for e in active}
    for c, cells in p['layers'].items():
        for pos in cells - deleted.get(c, set()):
            out[pos] = c
    if alternative == 'no_a':
        # Parent-B-like repair and straight rerouting, with no addressed casts.
        for e in p['edits']:
            for pos in e['old']:
                out[pos] = BG
            pos = _add(e['pivot'], e['direction'])
            while 1 <= pos[0] < inp.shape[0]-1 and 1 <= pos[1] < inp.shape[1]-1:
                out[pos] = e['color']
                pos = _add(pos, e['direction'])
        return out
    if alternative == 'destination':
        active.sort(key=lambda e: (min(e['rows']), min(e['cols'])))
    elif alternative == 'network':
        active.sort(key=lambda e: min(p['layers'][e['color']]))
    elif alternative == 'size':
        active.sort(key=lambda e: e['dims'][0]*e['dims'][1])
    elif alternative == 'reverse':
        active.reverse()
    occupied = set()
    for e in active:
        stamps = e['stamps']
        if alternative == 'single':
            stamps = stamps[:1]
        elif alternative in ('unturned', 'unrepaired', 'whole_network'):
            if alternative == 'unturned':
                offsets = {(-dy, -dx) for dy, dx in e['offsets']}
            elif alternative == 'unrepaired':
                offsets = {(e['center'][0]-r, e['center'][1]-c)
                           for r, c in set(e['old']) | {e['pivot']} if inp[r, c] == e['color']}
            else:
                whole = p['layers'][e['color']]
                # Remove only the topological mask selection in this ablation;
                # keep the same registration center and turn convention.
                cy, cx = e['center']
                offsets = {(cy-r, cx-c) for r, c in whole}
            stamps = [{(r+dy, c+dx) for dy, dx in offsets
                       if 1 <= r+dy < inp.shape[0]-1 and 1 <= c+dx < inp.shape[1]-1}
                      for r in e['rows'] for c in e['cols']]
        for stamp in stamps:
            for pos in stamp - occupied:
                out[pos] = e['color']
            occupied.update(stamp)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct, select, half-turn, and address branches using input only."""
    return _predict(_grid(input_grid))


def _rotate(p, k):
    r, c = p
    for _ in range(k % 4):
        r, c = c, -r
    return r, c


def _model(rng, max_extent=4, kind=None, allowed=None, junction=None):
    if kind == 'large':
        points = [(0,0), (0,4), (4,4), (4,0)]
    elif kind == 'small':
        points = [(0,0), (0,2), (2,2)]
    elif kind == 'straight':
        points = [(0,0), (0,2)]
    elif kind == 'long':
        points = [(0,0), (0,6), (6,6), (6,0), (3,0), (3,3)]
    else:
        points = None
        for _ in range(80):
            x = int(rng.choice(list(range(2, max_extent+1, 2))))
            trial = [(0,0), (0,x)]
            axis = 0
            for _ in range(int(rng.integers(0,6))):
                a = trial[-1]
                choices = [v for v in range(max_extent+1) if abs(v-a[axis]) >= 2]
                rng.shuffle(choices)
                for v in choices:
                    b = list(a); b[axis] = int(v)
                    walk = _walk(trial+[tuple(b)])
                    s = set(walk)
                    if len(walk) == len(s) and all(len(_neighbors(p,s)) <= 2 for p in s):
                        trial.append(tuple(b)); break
                else:
                    break
                axis = 1-axis
            s = set(_walk(trial))
            lo, hi = _box(s)
            if hi[0] % 2 == hi[1] % 2 == 0 and _bridge(s) == s:
                points = trial; break
        if points is None:
            points = [(0,0),(0,max_extent),(max_extent,max_extent)]
    tail = _walk(points)
    retained = _walk([(0,0), (-int(rng.integers(2,5)),0)])
    if kind not in ('large','small','straight') and rng.random() < .35:
        last = retained[-1]
        # Retained arms have independent bends, outside the stencil box.
        retained += _line(last,(last[0],-int(rng.integers(2,5))))[1:]
    s = set(tail) | set(retained)
    add_junction = bool(rng.random() < .3) if junction is None else junction
    if add_junction:
        arm = set(_line((0,0),(int(rng.integers(1,4)),0)))
        proposal = s | arm
        if not arm.intersection(set(tail[1:])) and _bridge(proposal) == proposal and all(
                len(_neighbors(p,proposal)) <= (3 if p == (0,0) else 2) for p in proposal):
            s = proposal
    assert _bridge(s) == s and _connected(s)
    choices = [k for k in range(4) if allowed is None or _rotate((0,-1),k) in allowed]
    k = int(rng.choice(choices))
    return dict(cells={_rotate(p,k) for p in s}, pivot=(0,0),
                old=[_rotate(p,k) for p in tail[1:]], q=_rotate((0,-1),k),
                direction=_rotate((0,-1),k))


def _place(rng, model, shape, occupied, band=None, left_limit=None):
    h, w = shape
    points = set(model['cells']) | {model['q']}
    lo, hi = _box(points)
    top, bottom = (1,h-2) if band is None else band
    right = w-2 if left_limit is None else left_limit
    positions = [(r,c) for r in range(top-lo[0], bottom-hi[0]+1)
                 for c in range(1-lo[1], right-hi[1]+1)]
    rng.shuffle(positions)
    for pivot in positions:
        cells = {_add(p,pivot) for p in model['cells']}
        q = _add(model['q'],pivot)
        if (cells | {q}) & occupied or any(_neighbors(p,occupied) for p in cells | {q}):
            continue
        return dict(cells=cells, pivot=pivot, old=[_add(p,pivot) for p in model['old']],
                    q=q, direction=model['direction'])
    return None


def _stamp_sets(record):
    _, _, offsets = _stencil(record['pivot'],record['old'])
    return [{(r+dy,c+dx) for dy,dx in offsets} for r in record['rows'] for c in record['cols']]


def _coords(rng, records, shape, max_mult=2, priority=False, cartesian=False, force_three=False):
    h,w = shape
    used_r,used_c = set(),set()
    active = [t for t in records if t.get('state') == 'A']
    if priority:
        a,b = active[:2]
        da,_,oa = _stencil(a['pivot'],a['old'])
        _,_,ob = _stencil(b['pivot'],b['old'])
        options = []
        for dr in range(1,5):
            for dc in (-3,-2,-1,1,2,3):
                aa = {(r+dr,c+dc) for r,c in oa}
                box = {(r+dr,c+dc) for r in range(-(da[0]//2), da[0]//2+1)
                       for c in range(-(da[1]//2), da[1]//2+1)}
                if aa & ob and (box-aa) & ob and aa-ob and ob-aa:
                    options.append((dr,dc))
        assert options, 'priority masks lack a distinguishing overlap'
        dr,dc = options[int(rng.integers(len(options)))]
        br = h//2 - 2
        bc = w-7
        a['rows'],a['cols']=[br+dr],[bc+dc]
        b['rows'],b['cols']=[br],[bc]
        for t in (a,b):
            used_r.update(t['rows']);used_c.update(t['cols'])
    for index,t in enumerate(records):
        t.setdefault('rows',[]);t.setdefault('cols',[])
        if t.get('q') is None:
            # An unmarked network is not a stencil, even with both axes.
            # The priority witness establishes this negative case as well.
            if priority or rng.random() < .5:
                r=int(rng.choice([v for v in range(1,h-1) if v not in used_r]))
                c=int(rng.choice([v for v in range(1,w-1) if v not in used_c]))
                t['rows'],t['cols']=[r],[c]
                used_r.add(r);used_c.add(c)
            continue
        dims,_,_ = _stencil(t['pivot'],t['old'])
        state = t['state']
        if state == 'A':
            counts = (2,2) if cartesian and t is active[0] else (int(rng.integers(1,max_mult+1)),int(rng.integers(1,max_mult+1)))
            if force_three and t is active[0]:
                counts = (3,int(rng.integers(1,4))) if rng.random()<.5 else (int(rng.integers(1,4)),3)
        else:
            counts = (1,0) if state == 'R' else (0,1) if state == 'C' else (0,0)
        for axis,(limit,name,used) in enumerate(((h,'rows',used_r),(w,'cols',used_c))):
            if t[name]:
                continue
            rad = dims[axis]//2 if state == 'A' else 0
            choices = [v for v in range(1+rad,limit-1-rad) if v not in used]
            # Replication witness places casts away from its packed sources.
            if cartesian and state == 'A' and axis == 1:
                choices = [v for v in choices if v >= w//2+2]
            rng.shuffle(choices)
            assert len(choices) >= counts[axis]
            t[name]=list(map(int,choices[:counts[axis]]));used.update(t[name])
    return used_r,used_c


def _construct(shape, records, rng, absent=0):
    h,w = shape
    inp = np.full(shape,BG,dtype=np.int8)
    inp[[0,-1],:]=FRAME;inp[:,[0,-1]]=FRAME
    for t in records:
        for p in t['cells']:
            assert inp[p] == BG
            inp[p]=t['color']
        if t.get('q') is not None:
            assert inp[t['q']] == BG
            inp[t['q']]=DIRECTION
        for r in t.get('rows',[]):
            assert inp[r,0] in (FRAME,t['color'])
            inp[r,0]=inp[r,-1]=t['color']
        for c in t.get('cols',[]):
            assert inp[0,c] in (FRAME,t['color'])
            inp[0,c]=inp[-1,c]=t['color']
    available = [c for c in PAYLOAD if c not in {t['color'] for t in records}]
    rng.shuffle(available)
    for c in available[:absent]:
        rs=[r for r in range(1,h-1) if inp[r,0]==FRAME]
        cs=[x for x in range(1,w-1) if inp[0,x]==FRAME]
        if rs and cs:
            r=int(rng.choice(rs));x=int(rng.choice(cs))
            inp[r,0]=inp[r,-1]=c;inp[0,x]=inp[-1,x]=c
    # Ground truth uses undamaged construction layers, reverse mask painting.
    out=inp.copy();out[1:-1,1:-1]=BG
    active=sorted([t for t in records if t.get('state')=='A'],key=lambda t:t['pivot'])
    for t in records:
        removed=set(t['old']) if t in active else set()
        for pos in t['cells']-removed:
            out[pos]=t['color']
    for t in reversed(active):
        for stamp in _stamp_sets(t):
            for pos in stamp:
                assert 1<=pos[0]<h-1 and 1<=pos[1]<w-1
                out[pos]=t['color']
    return inp,out


def _damage(rng, inp, records, witness=False):
    latent={t['color']:set(t['cells']) for t in records}
    forced=[]
    if witness:
        for t in records:
            if t.get('q') is not None:
                forced.append(t['pivot'])
                if t.get('state') == 'A':
                    forced.append(t['old'][min(3,len(t['old'])-1)])
                    kept=sorted(t['cells']-set(t['old'])-{t['pivot']})
                    if kept:
                        forced.append(kept[len(kept)//2])
    pool=[p for t in records for p in t['cells']]
    rng.shuffle(pool)
    candidates=list(dict.fromkeys(forced+pool))
    desired=max(len(forced),int(rng.integers(1,9)))
    successes=0
    for pos in candidates:
        old=int(inp[pos]);inp[pos]=REPAIR
        try:
            assert _parse(inp)['layers']==latent
        except (AssertionError,IndexError):
            inp[pos]=old
        else:
            successes+=1
            if successes>=desired:
                break
    rng.shuffle(pool)
    desired=int(rng.integers(1,5)) if witness else int(rng.integers(0,5))
    successes=0
    for pos in pool:
        if successes>=desired:
            break
        if inp[pos]==REPAIR:
            continue
        old=int(inp[pos]);inp[pos]=BG
        try:
            assert _parse(inp)['layers']==latent
        except (AssertionError,IndexError):
            inp[pos]=old
        else:
            successes+=1
    assert _parse(inp)['layers']==latent
    return inp


def _make_witness(rng, kind, allowed):
    for _ in range(80):
        shape=(int(rng.integers(25,31)),int(rng.integers(25,31)))
        h,w=shape
        records=[];occupied=set();palette=list(map(int,rng.permutation(PAYLOAD)))
        if kind=='replicate':
            requests=[('large','A',(1,10)),('small','R',(12,18)),('straight','C',(20,h-2))]
        else:
            requests=[('large','A',(4,13)),('long_retained','A',None),('straight','N',(12,h-11))]
        for motif,state,band in requests:
            if motif=='long_retained':
                # The later pivot owns the earliest visible network pixel.
                # This separates pivot priority from whole-object reading order.
                d=tuple(map(int,allowed[int(rng.choice([i for i,x in enumerate(allowed) if x[0]==0]))]))
                pivot=(h-6+int(rng.integers(2)),11)
                points=[pivot,(pivot[0],11-2*d[1]),(pivot[0]+2,11-2*d[1])]
                walk=_walk(points)
                cells=set(walk)|{(r,11) for r in range(1,pivot[0]+1)}
                t=dict(cells=cells,pivot=pivot,old=walk[1:],q=_add(pivot,d),direction=d)
            else:
                model=_model(rng,kind=motif,allowed=allowed,junction=False)
                t=_place(rng,model,shape,occupied,band,left_limit=7 if kind=='priority' else 10)
            if t is None:
                break
            t.update(color=palette[len(records)],state=state)
            records.append(t);occupied.update(t['cells']|({t['q']} if t['q'] is not None else set()))
        if len(records)!=3:
            continue
        try:
            _coords(rng,records,shape,priority=kind=='priority',cartesian=kind=='replicate')
            if kind=='priority':
                a,b=records[:2]
                stamps=_stamp_sets(a)+_stamp_sets(b)
                union=set().union(*stamps)
                minc=min(c for r,c in union);maxc=max(c for r,c in union)
                choices=[r for r in sorted({r for r,c in union}) if
                         any((r,c) in union for c in range(minc,maxc+1)) and
                         any((r,c) not in union for c in range(minc,maxc+1))]
                rng.shuffle(choices)
                for r in choices:
                    line=set(_line((r,max(1,minc-1)),(r,min(w-2,maxc+1))))
                    if not line & occupied and not any(_neighbors(p,occupied) for p in line):
                        records.append(dict(color=palette[3],cells=line,q=None,rows=[],cols=[],state='S'))
                        # Both axes exist, but no arrow selects a castable tail.
                        free_r=[v for v in range(1,h-1) if all(v not in t['rows'] for t in records)]
                        free_c=[v for v in range(1,w-1) if all(v not in t['cols'] for t in records)]
                        records[-1]['rows']=[int(rng.choice(free_r))]
                        records[-1]['cols']=[int(rng.choice(free_c))]
                        break
                else:
                    continue
            inp,out=_construct(shape,records,rng,absent=int(rng.integers(0,3)))
            inp=_damage(rng,inp,records,witness=True)
            p=_parse(inp)
            assert np.array_equal(_predict(inp,p),out)
            f=_features(inp,p)
            if kind=='replicate':
                assert f['cartesian'] and f['unrepaired'] and f['unturned']
                assert f['deleted_repairs'] and f['retained_repairs'] and f['marked_pivots'] and f['gaps']
            else:
                assert all(f[k] for k in ('destination','network','size','reverse','hole_exposure','over_retained'))
            pair = {'input':inp,'output':out}
            # A one-active-stencil scene is equivariant under every symmetry.
            # The priority witness can reflect horizontally: distinct pivot
            # rows keep source ownership unchanged. Filter directions to
            # preserve a promised held-out-orientation shift.
            symmetries = [(k,flip) for k in range(4) for flip in (False,True)] if kind=='replicate' else [(0,False),(0,True)]
            rng.shuffle(symmetries)
            for k,flip in symmetries:
                candidate = {role:np.rot90(np.fliplr(a) if flip else a,k).copy()
                             for role,a in pair.items()}
                if {e['direction'] for e in _parse(candidate['input'])['edits']} <= set(allowed):
                    assert np.array_equal(solve(candidate['input']),candidate['output'])
                    return candidate
            raise AssertionError('identity symmetry must remain allowed')
        except (AssertionError,IndexError):
            continue
    return _fallback_witness(rng,kind,allowed)


def _fallback_witness(rng,kind,allowed):
    """Guaranteed scaffolds retain every witness after bounded random search."""
    palette=list(map(int,rng.permutation(PAYLOAD)))

    def record(index, points, retained, state, rows=(), cols=()):
        path=_walk(points);pivot=path[0]
        return dict(color=palette[index],pivot=pivot,old=path[1:],
                    cells=set(path)|set(_walk(retained)),q=(pivot[0],pivot[1]-1),
                    direction=(0,-1),state=state,rows=list(rows),cols=list(cols))

    if kind=='replicate':
        records=[
            record(0,[(6,4),(6,8),(10,8),(10,4)],[(6,4),(3,4)],'A',(18,25),(18,25)),
            record(1,[(16,4),(16,6),(18,6)],[(16,4),(14,4)],'R',(12,)),
            record(2,[(24,6),(24,8)],[(24,6),(22,6)],'C',cols=(13,))]
        markers=[(6,4),(6,8),(4,4),(16,4),(24,6)]
        gap=(8,8)
    else:
        assert kind=='priority'
        records=[
            record(0,[(8,4),(8,8),(12,8),(12,4)],[(8,4),(5,4)],'A',(14,),(23,)),
            record(1,[(24,11),(24,13),(26,13)],[(24,11),(1,11)],'A',(13,),(22,)),
            record(2,[(18,4),(18,6)],[(18,4),(16,4)],'N')]
        records.append(dict(color=palette[3],cells=set(_line((14,19),(14,27))),
                            q=None,state='S',rows=[10],cols=[18]))
        markers=[(8,4),(8,8),(5,4),(24,11),(26,13),(12,11),(18,4),(14,19)]
        gap=(10,8)
    inp,out=_construct((30,30),records,rng)
    for pos in markers:
        inp[pos]=REPAIR
    inp[gap]=BG
    # Main generation allows all directions or excludes just one, so one of
    # these two horizontal directions is always available.
    if (0,-1) not in allowed:
        assert (0,1) in allowed
        inp,out=np.fliplr(inp).copy(),np.fliplr(out).copy()
    assert np.array_equal(solve(inp),out)
    f=_features(inp)
    required=('cartesian','unturned','unrepaired','deleted_repairs','retained_repairs','gaps') if kind=='replicate' else (
        'destination','network','size','reverse','hole_exposure','over_retained','addressed_unmarked')
    assert all(f[k] for k in required)
    return {'input':inp,'output':out}


def _make_random(rng, params, test=False):
    for attempt in range(100):
        h,w=int(rng.integers(15,31)),int(rng.integers(15,31))
        count=int(rng.integers(1,5))
        allowed=params['test_dirs'] if test else params['train_dirs']
        extent=6 if test and params['shift']=='size' else 4 if not test else int(rng.choice([4,6]))
        records=[];occupied=set();palette=list(map(int,rng.permutation(PAYLOAD)))
        states=list(map(str,rng.choice(['A','R','C','N','S'],count,p=[.55,.12,.12,.11,.10])))
        states[int(rng.integers(count))]='A'
        forced=next(i for i,s in enumerate(states) if s=='A')
        for index,state in enumerate(states):
            kind='long' if test and params['shift']=='size' and index==forced else None
            model=_model(rng,max_extent=extent,kind=kind,allowed=allowed)
            t=_place(rng,model,(h,w),occupied)
            if t is None:
                break
            t.update(color=palette[index],state=state)
            if state=='S':
                t['q']=None
            records.append(t);occupied.update(t['cells']|({t['q']} if t['q'] is not None else set()))
        if len(records)!=count:
            continue
        try:
            _coords(rng,records,(h,w),max_mult=2 if not test else 3,
                    force_three=test and params['shift']=='multiplicity')
            inp,out=_construct((h,w),records,rng,absent=int(rng.integers(0,3)))
            inp=_damage(rng,inp,records)
            assert np.array_equal(solve(inp),out)
            return {'input':inp,'output':out}
        except (AssertionError,IndexError):
            continue
    # Large, uncluttered fallback keeps the declared shift and visible rule.
    h,w=29,30
    palette=list(map(int,rng.permutation(PAYLOAD)))
    model=_model(rng,kind='long' if test and params['shift']=='size' else 'large',
                 allowed=params['test_dirs'] if test else params['train_dirs'],junction=False)
    t=_place(rng,model,(h,w),set(),left_limit=12)
    assert t is not None
    t.update(color=palette[0],state='A')
    records=[t]
    _coords(rng,records,(h,w),max_mult=2 if not test else 3,
            force_three=test and params['shift']=='multiplicity')
    inp,out=_construct((h,w),records,rng)
    inp=_damage(rng,inp,records)
    assert np.array_equal(solve(inp),out)
    return {'input':inp,'output':out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed=_integer(seed,'seed')
    nt,ne=_counts(num_train,num_test)
    rng=np.random.default_rng(seed % (1<<64))
    shift=str(rng.choice(['multiplicity','size','direction']))
    held=DIRS[int(rng.integers(4))]
    params=dict(shift=shift,train_dirs=[d for d in DIRS if shift!='direction' or d!=held],
                test_dirs=[held] if shift=='direction' else list(DIRS))
    train=[_make_witness(rng,'replicate',params['train_dirs']),
           _make_witness(rng,'priority',params['train_dirs'])]
    train.extend(_make_random(rng,params) for _ in range(nt-2))
    test=[_make_random(rng,params,test=True) for _ in range(ne)]
    seen=set()
    for pair in train+test:
        if _key(pair['input']) in seen:
            # Payload bijections keep reserved meanings and all equality tests.
            original={k:v.copy() for k,v in pair.items()}
            for perm in (np.roll(PAYLOAD,i) for i in range(1,6)):
                mapping=np.arange(10,dtype=np.int8)
                mapping[list(PAYLOAD)]=perm
                candidate={k:mapping[v] for k,v in original.items()}
                if _key(candidate['input']) not in seen:
                    pair.update(candidate);break
            else:
                # Geometric regeneration is bounded and retains the shift.
                for _ in range(30):
                    candidate=_make_random(rng,params,test=any(pair is x for x in test))
                    if _key(candidate['input']) not in seen:
                        pair.update(candidate);break
                else:
                    raise RuntimeError('uniqueness retries exhausted')
        seen.add(_key(pair['input']))
    rng.shuffle(train)
    puzzle={'train':train,'test':test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, parsed=None):
    p=_parse(inp) if parsed is None else parsed
    correct=_predict(inp,p)
    active=[e for e in p['edits'] if e['active']]
    retained={c:set(s) for c,s in p['layers'].items()}
    for e in active:
        retained[e['color']].difference_update(e['old'])
    source_repairs=set(p['repairs'])
    deleted=set().union(*(set(e['old']) for e in active)) if active else set()
    unions=[set().union(*e['stamps']) for e in active]
    holes=0
    for e in active:
        bh,bw=e['dims']
        for r in e['rows']:
            for c in e['cols']:
                stamp={(r+dy,c+dx) for dy,dx in e['offsets']}
                holes+=sum(correct[y,x] not in (BG,e['color'])
                           for y in range(r-bh//2,r+bh//2+1)
                           for x in range(c-bw//2,c+bw//2+1) if (y,x) not in stamp)
    features=dict(shape=inp.shape,count=len(p['layers']),active=len(active),
                  directions=tuple(e['direction'] for e in p['edits']),
                  products=tuple((len(e['rows']),len(e['cols'])) for e in active),
                  dims=tuple(e['dims'] for e in active),
                  masks=tuple((e['dims'],tuple(sorted(e['offsets']))) for e in active),
                  states=tuple('A' if e['active'] else 'R' if e['rows'] else 'C' if e['cols'] else 'N'
                               for e in p['edits']),
                  cartesian=any(len(e['rows'])>1 and len(e['cols'])>1 for e in active),
                  deleted_repairs=len(source_repairs & deleted),
                  retained_repairs=len(source_repairs-deleted),
                  marked_pivots=sum(e['pivot'] in source_repairs for e in p['edits']),
                  gaps=len(p['gaps']),repairs=len(source_repairs),hole_exposure=holes,
                  conflicts=sum(len(a & b) for i,a in enumerate(unions) for b in unions[i+1:]),
                  merges=sum(sum(map(len,e['stamps']))-len(u) for e,u in zip(active,unions)),
                  over_retained=sum(len(u & s) for e,u in zip(active,unions)
                                    for c,s in retained.items() if c!=e['color']),
                  bends=tuple(sum(a!=b for a,b in zip(ds,ds[1:])) for e in active
                              for ds in [[(b[0]-a[0],b[1]-a[1]) for a,b in
                                          zip([e['pivot']]+e['old'],e['old'])]]),
                  centerless=sum(e['center'] not in set(e['old'])|{e['pivot']} for e in active),
                  junctions=sum(len(_neighbors(e['pivot'],p['layers'][e['color']]))==3 for e in p['edits']))
    unmarked=set(p['layers'])-{e['color'] for e in p['edits']}
    features['addressed_unmarked']=sum(bool(p['rows'][c] and p['cols'][c]) for c in unmarked)
    for name in ('no_a','whole_network','single','unturned','unrepaired','destination','network','size','reverse'):
        features[name]=int(np.count_nonzero(correct!=_predict(inp,p,alternative=name)))
    return features


def validate_generated_puzzle(puzzle):
    """Full-grid equality plus visible-input evidence and topology invariants."""
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen=set();fs={'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            inp,out=_grid(pair['input']),_grid(pair['output'])
            assert inp.shape==out.shape and not np.array_equal(inp,out)
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            p=_parse(inp)
            assert all(sum(len(_neighbors(e['q'],s)) for s in p['layers'].values())==1
                       for e in p['edits']), 'direction annotations must touch only their pivot'
            assert np.array_equal(_predict(inp,p),out), 'wrong branch, mask, rotation, priority, or preserved cell'
            assert np.array_equal(inp[0],out[0]) and np.array_equal(inp[-1],out[-1])
            assert np.array_equal(inp[:,0],out[:,0]) and np.array_equal(inp[:,-1],out[:,-1])
            assert not np.any(np.isin(out,[REPAIR,DIRECTION]))
            f=_features(inp,p);assert f['active']>0
            fs[split].append(f)
    train=fs['train']
    for key in ('cartesian','deleted_repairs','retained_repairs','marked_pivots',
                'unturned','unrepaired','no_a','whole_network','single','addressed_unmarked','gaps'):
        assert any(f[key] for f in train), f'missing training evidence: {key}'
    assert {'R','C','N'} <= {s for f in train for s in f['states']}, 'incomplete addresses must be taught'
    for key in ('conflicts','over_retained'):
        if any(f[key] for f in fs['test']):
            required=('destination','network','size','reverse','hole_exposure') if key=='conflicts' else ('over_retained',)
            assert all(any(f[k] for f in train) for k in required), 'underdemonstrated interaction'
    # A visible, measurable shift: greater multiplicity, larger stencil, or a
    # direction absent from all demonstrations. No sampled metadata is consulted.
    td={d for f in train for d in f['directions']}
    max_mult=max(v for f in train for product in f['products'] for v in product)
    max_dim=max(v for f in train for dims in f['dims'] for v in dims)
    for f in fs['test']:
        assert (any(v>max_mult for p in f['products'] for v in p) or
                any(v>max_dim for d in f['dims'] for v in d) or
                any(d not in td for d in f['directions'])), 'missing intended train/test shift'


def self_test():
    """Bounded 42-seed audit, ablations, transformations, and mutation checks."""
    fixture = _regression_fixture()
    assert np.array_equal(solve(fixture['input']),fixture['output'])
    for i,held in enumerate(DIRS):
        rng=np.random.default_rng(8000+i)
        allowed=[d for d in DIRS if d!=held]
        train=[_fallback_witness(rng,k,allowed) for k in ('replicate','priority')]
        query=_make_random(rng,dict(shift='direction',train_dirs=allowed,test_dirs=[held]),test=True)
        validate_generated_puzzle({'train':train,'test':[query]})
    puzzles=[];features=[];signatures=set();shapes=set();masks=set();products=set();dirs=set()
    for seed in range(42):
        p=generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature=tuple(_key(q['input']) for q in p['train']+p['test'])
        signatures.add(signature);puzzles.append(p)
        for q in p['train']+p['test']:
            inp=np.asarray(q['input']);f=_features(inp);features.append(f)
            shapes.add(inp.shape);masks.update(f['masks']);products.update(f['products']);dirs.update(f['directions'])
            assert np.array_equal(solve(inp.copy()),q['output'])
    assert len(signatures)==42 and len(shapes)>=40 and len(masks)>=25
    assert dirs==set(DIRS) and {(1,1),(2,2),(3,1),(1,3)} <= products
    assert {1,2,3} <= {f['active'] for f in features}
    assert any(f['junctions'] for f in features) and any(f['centerless'] for f in features)
    assert any(f['merges'] for f in features) and any(max(f['bends'])>=4 for f in features)
    assert len({f['count'] for f in features})>=4
    for name in ('no_a','whole_network','single','unturned','unrepaired','destination','network','size','reverse'):
        assert all(any(_features(q['input'])[name] for q in p['train']) for p in puzzles)
        p=puzzles[0]
        index=next(i for i,q in enumerate(p['train']) if _features(q['input'])[name])
        bad=copy.deepcopy(p)
        bad['train'][index]['output']=_predict(bad['train'][index]['input'],alternative=name)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'validator accepted the {name} shortcut')
    assert tuple(_key(q['input']) for q in puzzles[0]['train']+puzzles[0]['test']) == tuple(
        _key(q['input']) for q in generate_puzzle(0)['train']+generate_puzzle(0)['test'])
    # Test rotation/reflection for isolated casts: reading-order ownership must
    # be recomputed, rather than assumed rotation-invariant in colliding scenes.
    for p in puzzles[:3]:
        for q in p['train']+p['test']:
            if _features(q['input'])['conflicts']==0:
                for k in (1,2,3):
                    assert np.array_equal(solve(np.rot90(q['input'],k)),np.rot90(q['output'],k))
                assert np.array_equal(solve(np.fliplr(q['input'])),np.fliplr(q['output']))
    for p in puzzles[:2]:
        inp=p['test'][0]['input'];out=p['test'][0]['output']
        for pos in ((0,0),tuple(map(int,np.argwhere(inp==out)[-1])),
                    tuple(map(int,np.argwhere(inp!=out)[0]))):
            bad=copy.deepcopy(p);bad['test'][0]['output'][pos]=(int(out[pos])+1)%10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('validator accepted a corrupt output')
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p=generate_puzzle(1000+nt*10+ne,nt,ne)
            assert len(p['train'])==nt and len(p['test'])==ne
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds':42,'pairs':len(features),'shapes':len(shapes),'branch_masks':len(masks),
            'products':sorted(products),'directions':sorted(dirs),
            'source_counts':sorted({f['count'] for f in features}),
            'active_counts':sorted({f['active'] for f in features}),
            'observed_features':{k:sum(bool(f[k]) for f in features) for k in
                                 ('junctions','centerless','merges','conflicts','over_retained',
                                  'marked_pivots','deleted_repairs','retained_repairs','gaps')},
            'verification':'generator-owned; not independent semantic approval'}


def _regression_fixture():
    """Hand-specified 13x15 example; its expected cells use no generator parser.

    A marked/gapped L tail of color 1 is half-turned into a 2x2 product.
    An incomplete-address color 3 tail remains. Absent color 5 labels are inert.
    """
    inp=np.zeros((13,15),dtype=np.int8)
    inp[[0,-1],:]=FRAME;inp[:,[0,-1]]=FRAME
    for r in (8,10):
        inp[r,0]=inp[r,-1]=1
    for c in (5,11):
        inp[0,c]=inp[-1,c]=1
    inp[2,0]=inp[2,-1]=3
    inp[5,0]=inp[5,-1]=5;inp[0,9]=inp[-1,9]=5
    source_one={(2,3),(3,3),(4,3),(4,4),(4,5),(4,6),(4,7),(5,7),(6,7)}
    source_three={(1,11),(2,11),(2,12),(2,13),(3,13),(4,13)}
    for pos in source_one:
        inp[pos]=1
    for pos in source_three:
        inp[pos]=3
    for pos in ((2,3),(4,3),(4,7),(4,13)):
        inp[pos]=REPAIR
    inp[4,5]=BG
    inp[4,2]=inp[2,10]=DIRECTION
    expected=inp.copy();expected[1:-1,1:-1]=BG
    for pos in ((2,3),(3,3),(4,3)):
        expected[pos]=1
    for pos in source_three:
        expected[pos]=3
    # These are explicit target strokes, not results of _stencil or _predict.
    for row in (9,11):
        expected[row,3:8]=1;expected[row,9:14]=1
    expected[7:12,3]=1;expected[7:12,9]=1
    return {'input':inp,'output':expected}


if __name__=='__main__':
    print(json.dumps(self_test(),separators=(',',':')))
