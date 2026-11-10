"""Territory shadows: simultaneous anchor-registered silhouette transfers.

Only NumPy and the standard library are used. Generation holds one candidate
at a time; diagnostics and execution traces are never accumulated on disk.
"""
from __future__ import annotations

import hashlib
import numbers
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(ok, message):
    if not ok:
        raise AssertionError(message)


def _point(p, d, n=1):
    return p[0] + n*d[0], p[1] + n*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(grid):
    ids = np.full(grid.shape, -1, dtype=np.int16)
    cells, colors = [], []
    for p in np.ndindex(grid.shape):
        if ids[p] >= 0:
            continue
        index, color = len(cells), int(grid[p])
        stack, found = [p], []
        ids[p] = index
        while stack:
            u = stack.pop()
            found.append(u)
            for d in DIRS:
                v = _point(u, d)
                if _inside(v, grid.shape) and ids[v] < 0 and grid[v] == color:
                    ids[v] = index
                    stack.append(v)
        cells.append(found)
        colors.append(color)
    return ids, cells, colors


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    stack = [points.pop()]
    while stack:
        p = stack.pop()
        for d in DIRS:
            q = _point(p, d)
            if q in points:
                points.remove(q)
                stack.append(q)
    return not points


def _anchor_candidates(grid):
    _, cells, colors = _components(grid)
    answers = []
    for color in sorted(set(colors)):
        anchors = []
        for points, c in zip(cells, colors):
            if c != color:
                continue
            r0 = min(p[0] for p in points)
            c0 = min(p[1] for p in points)
            r1 = max(p[0] for p in points)+1
            c1 = max(p[1] for p in points)+1
            side = r1-r0
            if not (1 <= side <= 3 and side == c1-c0 and len(points) == side**2
                    and r0 > 0 and c0 > 0 and r1 < grid.shape[0] and c1 < grid.shape[1]):
                break
            ring = {(r,c) for r in range(r0-1,r1+1) for c in range(c0-1,c1+1)
                    if not (r0 <= r < r1 and c0 <= c < c1)}
            values = {int(grid[p]) for p in ring}
            if len(values) != 1 or color in values:
                break
            anchors.append(dict(top=r0,left=c0,side=side,host=values.pop(),
                                cells=set(points),support=set(points)|ring))
        else:
            if 2 <= len(anchors) <= 7:
                answers.append((color, anchors))
    return answers


def _geometry(center, direction, shape):
    perpendicular = (-direction[1], direction[0])
    arms = (_point(center,perpendicular), _point(center,perpendicular,-1))
    tip = _point(center,direction)
    target = _point(center,direction,2)
    flanks = (_point(tip,perpendicular), _point(tip,perpendicular,-1))
    write = {center,tip,*arms}
    context = {target,*flanks}
    if not all(_inside(p,shape) for p in write|context):
        return None
    return dict(center=center,d=direction,arms=arms,tip=tip,targetcell=target,
                flanks=flanks,write=write,context=context)


def _read(input_grid):
    grid = np.asarray(input_grid)
    _need(grid.ndim == 2 and all(1 <= s <= 30 for s in grid.shape), 'bad grid shape')
    _need(np.issubdtype(grid.dtype,np.integer) and np.all((0 <= grid)&(grid <= 9)), 'bad grid colors')
    options = _anchor_candidates(grid)
    _need(len(options) == 1, 'square-anchor ink is not uniquely identifiable')
    anchor_ink, anchors = options[0]
    bare = grid.copy()
    for a in anchors:
        for p in a['cells']:
            bare[p] = a['host']
    raw_ids, raw_cells, raw_colors = _components(bare)
    answers = []
    for ink in sorted(set(raw_colors)-{anchor_ink}):
        if any(len(s) > 4 for s,c in zip(raw_cells,raw_colors) if c == ink):
            continue
        ink_cells = set(map(tuple,np.argwhere(bare == ink)))
        if len(ink_cells) < 3:
            continue
        arrows = []
        for center in sorted({_point(p,d) for p in ink_cells for d in DIRS}):
            for d in DIRS:
                g = _geometry(center,d,grid.shape)
                if g is None or not all(bare[p] == ink for p in (*g['arms'],g['tip'])):
                    continue
                host = int(bare[g['flanks'][0]])
                if host == ink or any(bare[p] != host for p in g['flanks']):
                    continue
                if bare[g['targetcell']] in (ink,host):
                    continue
                if any(not set(raw_cells[raw_ids[p]]) <= g['write'] for p in (*g['arms'],g['tip'])):
                    continue
                explicit = int(bare[center]) != ink
                payload = int(bare[center]) if explicit else host
                if payload == anchor_ink:
                    continue
                arrows.append(dict(g,host=host,payload=payload,explicit=explicit))
        used, valid = set(), bool(arrows)
        for arrow in arrows:
            if used & arrow['write']:
                valid = False
            used.update(arrow['write'])
        if not valid or used & ink_cells != ink_cells:
            continue
        clean = bare.copy()
        for arrow in arrows:
            for p in arrow['write']:
                clean[p] = arrow['host']
        ids,cells,colors = _components(clean)
        if len(cells) != len(anchors) or ink in colors or anchor_ink in colors:
            continue
        by_id = {}
        for a in anchors:
            label = int(ids[a['top'],a['left']])
            if label in by_id or any(ids[p] != label for p in a['support']):
                valid = False
            by_id[label] = a
        targets = set()
        for arrow in arrows:
            source = int(ids[arrow['center']])
            target = int(ids[arrow['targetcell']])
            if (source == target or target in targets
                    or any(ids[p] != source for p in arrow['write']|set(arrow['flanks']))):
                valid = False
            arrow.update(source=source,target=target)
            targets.add(target)
        if valid:
            answers.append(dict(grid=grid,clean=clean,ids=ids,cells=cells,colors=colors,
                                anchors=by_id,arrows=arrows,anchor_ink=anchor_ink,ink=ink))
    _need(len(answers) == 1, 'T-arrow interpretation is not unique')
    return answers[0]


def _shadow(data, arrow, reflect=True, scale=True, bounding_box=False, clip=True):
    """Input-only binary raster, exact even for noninteger scale ratios."""
    s,t = arrow['source'],arrow['target']
    a,b = data['anchors'][s],data['anchors'][t]
    side = a['side']
    dest = b['side'] if scale else side
    points = np.asarray(data['cells'][s])-np.array([a['top'],a['left']])
    top,left = (points.min(0)//side*side).tolist()
    bottom,right = ((points.max(0)+side)//side*side).tolist()
    raw = np.zeros((bottom-top,right-left),dtype=np.uint8)
    if bounding_box:
        lo,hi = points.min(0),points.max(0)+1
        raw[lo[0]-top:hi[0]-top,lo[1]-left:hi[1]-left] = 1
    else:
        raw[points[:,0]-top,points[:,1]-left] = 1
    expanded = np.repeat(np.repeat(raw,dest,axis=0),dest,axis=1)
    h,w = expanded.shape[0]//side,expanded.shape[1]//side
    blocks = expanded.reshape(h,side,w,side)
    low,high = blocks.min(axis=(1,3)),blocks.max(axis=(1,3))
    _need(np.array_equal(low,high), 'a scaled silhouette boundary straddles a pixel')
    offset = [-top*dest//side,-left*dest//side]
    if reflect and arrow['d'][0]:
        low = low[::-1,:]
        offset[0] = h-offset[0]-dest
    if reflect and arrow['d'][1]:
        low = low[:,::-1]
        offset[1] = w-offset[1]-dest
    r,c = b['top']-offset[0],b['left']-offset[1]
    rr,cc = np.nonzero(low)
    rr,cc = rr+r,cc+c
    inside = (rr >= 0)&(rr < data['ids'].shape[0])&(cc >= 0)&(cc < data['ids'].shape[1])
    mask = np.zeros(data['ids'].shape,dtype=bool)
    mask[rr[inside],cc[inside]] = True
    return mask & (data['ids'] == t) if clip else mask


def _predict(data, mode='normal'):
    out = data['clean'].copy()
    for arrow in data['arrows']:
        mask = ((data['ids'] == arrow['target']) if mode == 'whole' else
                _shadow(data,arrow,mode != 'no_reflection',mode != 'no_scale',
                        mode == 'bounding_box',mode != 'no_clip'))
        if mode != 'no_transfer':
            out[mask] = arrow['payload']
    for a in data['anchors'].values():
        for p in a['cells']:
            out[p] = data['anchor_ink']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, training examples, files, or generation state are consulted."""
    return _predict(_read(input_grid))


def _partition(rng, shape, count, q):
    ids = np.zeros(shape,dtype=np.int16)
    yy,xx = np.indices(shape)
    minimum = max(2, (18+q*q-1)//(q*q))
    for new in range(1,count):
        for _ in range(60):
            areas = np.bincount(ids.ravel()).astype(float)
            label = int(rng.choice(new,p=areas/areas.sum()))
            mask = ids == label
            rows,cols = np.nonzero(mask)
            r0,r1,c0,c1 = rows.min(),rows.max()+1,cols.min(),cols.max()+1
            axis = int(rng.integers(2))
            lo,hi = (r0,r1) if axis == 0 else (c0,c1)
            if hi-lo < 2:
                continue
            cut = int(rng.integers(lo+1,hi))
            coord,along = (yy,xx) if axis == 0 else (xx,yy)
            path = np.full(shape[1-axis],cut)
            if rng.random() < .68:
                value = cut
                step = int(rng.integers(1,5))
                for start in range(0,len(path),step):
                    value = int(np.clip(value+rng.integers(-1,2),lo+1,hi-1))
                    path[start:start+step] = value
            section = coord < path[along]
            if min(r1-r0,c1-c0) >= 5 and rng.random() < .16:
                top = int(rng.integers(r0+1,r1-2))
                left = int(rng.integers(c0+1,c1-2))
                bottom = int(rng.integers(top+2,r1))
                right = int(rng.integers(left+2,c1))
                section = (yy >= top)&(yy < bottom)&(xx >= left)&(xx < right)
            a,b = mask&section,mask&~section
            if min(int(a.sum()),int(b.sum())) < minimum or not _connected(a) or not _connected(b):
                continue
            ids[a] = new
            break
        else:
            return None
    return ids


def _anchor_places(ids, label, side):
    h,w = ids.shape
    return [(r,c) for r in range(side,h-side,side) for c in range(side,w-side,side)
            if np.all(ids[r-1:r+side+1,c-1:c+side+1] == label)]


def _construction_shadow(base, q, source, target, anchors, d):
    """Constructor paints transformed q-cell rectangles, separately from solver."""
    a,b = anchors[source],anchors[target]
    side,dest = a['side'],b['side']
    _need(q*dest % side == 0, 'construction lattice is incompatible')
    mask = np.zeros((base.shape[0]*q,base.shape[1]*q),dtype=bool)
    h,w = mask.shape
    for r,c in np.argwhere(base == source):
        y0,y1 = int(r)*q-a['top'],(int(r)+1)*q-a['top']
        x0,x1 = int(c)*q-a['left'],(int(c)+1)*q-a['left']
        if d[0]:
            y0,y1 = side-y1,side-y0
        if d[1]:
            x0,x1 = side-x1,side-x0
        _need(all(v*dest % side == 0 for v in (y0,y1,x0,x1)), 'nonintegral rectangle')
        y0,y1 = y0*dest//side+b['top'],y1*dest//side+b['top']
        x0,x1 = x0*dest//side+b['left'],x1*dest//side+b['left']
        if max(0,y0) < min(h,y1) and max(0,x0) < min(w,x1):
            mask[max(0,y0):min(h,y1),max(0,x0):min(w,x1)] = True
    ids = np.repeat(np.repeat(base,q,axis=0),q,axis=1)
    return mask & (ids == target)


def _build(rng, count, q, witness_axis=None, explicit=False):
    low = max(4,12//q)
    high = 30//q
    shape = tuple(int(rng.integers(low,high+1)) for _ in range(2))
    base = _partition(rng,shape,count,q)
    if base is None:
        return None
    ids = np.repeat(np.repeat(base,q,axis=0),q,axis=1)
    palette = [int(c) for c in rng.permutation(10)]
    ink,anchor_ink = palette[:2]
    colors = palette[2:2+count]
    # Repeat fills only across disconnected, nonadjacent territories.
    if count >= 4 and rng.random() < .55:
        neighbors = set()
        for x,y in ((ids[:-1],ids[1:]),(ids[:,:-1],ids[:,1:])):
            neighbors.update(tuple(sorted((int(a),int(b)))) for a,b in zip(x[x != y],y[x != y]))
        choices = [(a,b) for a in range(count) for b in range(a) if (b,a) not in neighbors]
        if choices:
            a,b = choices[int(rng.integers(len(choices)))]
            colors[a] = colors[b]
    anchors, blocked = {}, set()
    sizes = [a for a in (1,2,3) if q % a == 0]
    for label in range(count):
        eligible = {s:_anchor_places(ids,label,s) for s in sizes}
        eligible = {s:p for s,p in eligible.items() if p}
        if not eligible:
            return None
        side = int(rng.choice(list(eligible)))
        points = eligible[side]
        r,c = points[int(rng.integers(len(points)))]
        cells = {(y,x) for y in range(r,r+side) for x in range(c,c+side)}
        support = {(y,x) for y in range(r-1,r+side+1) for x in range(c-1,c+side+1)}
        anchors[label] = dict(top=r,left=c,side=side,cells=cells,support=support)
        blocked.update(support)
    if witness_axis is not None and len({a['side'] for a in anchors.values()}) < 2:
        return None
    clean = np.asarray(colors,dtype=np.int8)[ids]
    masks, candidates = {}, []
    for center in np.ndindex(ids.shape):
        for d in DIRS:
            g = _geometry(center,d,ids.shape)
            if g is None or g['write'] & blocked or g['context'] & blocked:
                continue
            s,t = int(ids[center]),int(ids[g['targetcell']])
            if s == t or colors[s] == colors[t] or any(ids[p] != s for p in g['write']|set(g['flanks'])):
                continue
            key = s,t,bool(d[0])
            if key not in masks:
                mask = _construction_shadow(base,q,s,t,anchors,d)
                usable = mask.copy()
                for p in anchors[t]['cells']:
                    usable[p] = False
                masks[key] = (mask,int(usable.sum()))
            mask,area = masks[key]
            if area < 2:
                continue
            candidates.append(dict(g,source=s,target=t,mask=mask))
    if not candidates:
        return None
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    chosen,targets,used,context = [],set(),set(),set()

    def legal(a):
        halo = a['write']|{_point(p,d) for p in a['write'] for d in DIRS}
        return (a['target'] not in targets and not (halo & used)
                and not (a['write'] & context) and not (a['context'] & used))

    def add(a):
        chosen.append(a)
        targets.add(a['target'])
        used.update(a['write'])
        context.update(a['context'])

    if witness_axis is not None:
        found = False
        for a in candidates:
            if bool(a['d'][0]) != bool(witness_axis):
                continue
            if anchors[a['source']]['side'] == anchors[a['target']]['side']:
                continue
            for b in candidates:
                if b['source'] != a['target'] or b['target'] != a['source'] or b['d'] != (-a['d'][0],-a['d'][1]):
                    continue
                add(a)
                if legal(b):
                    add(b)
                    found = True
                    break
                chosen.clear(); targets.clear(); used.clear(); context.clear()
            if found:
                break
        if not found:
            return None
    minimum_arrows = 3 if witness_axis is not None and explicit else 2
    desired = int(rng.integers(minimum_arrows,min(count,6)+1))
    for a in candidates:
        if len(chosen) >= desired:
            break
        if legal(a):
            add(a)
    if len(chosen) < minimum_arrows:
        return None
    inp,out = clean.copy(),clean.copy()
    for index,a in enumerate(chosen):
        payload = colors[a['source']]
        is_explicit = explicit and index == len(chosen)-1
        if is_explicit:
            choices = [c for c in range(10) if c not in (ink,anchor_ink,colors[a['source']],colors[a['target']])]
            payload = int(rng.choice(choices))
        for p in a['write']:
            inp[p] = ink
        if is_explicit:
            inp[a['center']] = payload
        out[a['mask']] = payload
    for a in anchors.values():
        for p in a['cells']:
            inp[p] = out[p] = anchor_ink
    try:
        data = _read(inp)
        predicted = _predict(data)
        if not np.array_equal(predicted,out):
            return None
        f = _features(data)
        if not (f['partial'] and f['reflection_witness'] and f['a_witness'] and f['b_witness']):
            return None
        if witness_axis is not None and not (f['scale_witness'] and f['dependency']
                                             and bool(witness_axis) in f['reflection_axes']):
            return None
        if witness_axis is not None and explicit and not (f['shape_witness'] and f['clip_witness']):
            return None
    except AssertionError:
        return None
    return dict(input=inp,output=out),f


def _features(data):
    out = _predict(data)
    partial,axes,scaled,explicit,dependency = False,set(),False,False,False
    areas = []
    for arrow in data['arrows']:
        mask = _shadow(data,arrow)
        target = arrow['target']
        available = data['ids'] == target
        for p in data['anchors'][target]['cells']:
            available[p] = False
        area = int(np.count_nonzero(mask & available))
        areas.append(area)
        partial |= 0 < area < int(available.sum())
        scaled |= data['anchors'][arrow['source']]['side'] != data['anchors'][target]['side']
        explicit |= arrow['explicit']
        if not np.array_equal(mask & available,_shadow(data,arrow,False) & available):
            axes.add(bool(arrow['d'][0]))
        dependency |= (not arrow['explicit'] and any(b['target'] == arrow['source']
                         and b['payload'] != data['colors'][arrow['source']]
                         and _shadow(data,b)[arrow['center']] for b in data['arrows']))
    return dict(shape=data['grid'].shape,regions=len(data['cells']),arrows=len(data['arrows']),
                partial=partial,reflection_witness=bool(axes),reflection_axes=axes,scaled=scaled,
                explicit=explicit,dependency=dependency,areas=areas,
                scale_witness=not np.array_equal(out,_predict(data,'no_scale')),
                shape_witness=not np.array_equal(out,_predict(data,'bounding_box')),
                clip_witness=not np.array_equal(out,_predict(data,'no_clip')),
                a_witness=not np.array_equal(out,_predict(data,'no_transfer')),
                b_witness=not np.array_equal(out,_predict(data,'whole')),
                repeated=len(set(data['colors'])) < len(data['colors']),
                sides={a['side'] for a in data['anchors'].values()},
                ratios={(data['anchors'][a['target']]['side'],data['anchors'][a['source']]['side'])
                        for a in data['arrows']},
                directions={a['d'] for a in data['arrows']},
                nonrectangular=sum(len(points) != (max(p[0] for p in points)-min(p[0] for p in points)+1)
                                  *(max(p[1] for p in points)-min(p[1] for p in points)+1) for points in data['cells']))


def _integer(value):
    return isinstance(value,numbers.Integral) and not isinstance(value,(bool,np.bool_))


def _key(grid):
    a = np.asarray(grid,dtype=np.uint8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _integer(num_train) or not _integer(num_test) or not 2 <= num_train <= 10
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed)
    puzzle,seen = {'train':[],'test':[]},set()
    axes = [int(x) for x in rng.permutation(2)]
    teaching_q = [2,3] if rng.random() < .5 else [3,2]
    for split,n in (('train',num_train),('test',num_test)):
        for index in range(n):
            witness = axes[index] if split == 'train' and index < 2 else None
            explicit = (index == 1) if witness is not None else bool(rng.random() < .5)
            for attempt in range(240):
                if witness is not None:
                    q = teaching_q[index]
                    count = 3 if index == 0 else 4
                else:
                    q = int(rng.choice([1,2,3,6],p=[.25,.32,.33,.10]))
                    count = int(rng.integers(5,7)) if split == 'test' else int(rng.integers(2,5))
                    if q == 6 and split == 'test':
                        count = 5
                result = _build(rng,count,q,witness,explicit)
                if result is None:
                    continue
                pair,f = result
                key = _key(pair['input'])
                if key in seen:
                    continue
                # Explicit payloads are held out of tests unless demonstrated.
                if split == 'test' and f['explicit'] and not any(_features(_read(p['input']))['explicit'] for p in puzzle['train']):
                    explicit = False
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded territory-shadow construction exhausted')
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, 'bad puzzle schema')
    _need(all(isinstance(puzzle[s],list) for s in puzzle), 'splits must be lists')
    a,b = len(puzzle['train']),len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'bad counts')
    seen,analyses = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            _need(isinstance(pair,dict) and set(pair) == {'input','output'}, 'bad pair schema')
            inp,out = np.asarray(pair['input']),np.asarray(pair['output'])
            for grid in (inp,out):
                _need(grid.ndim == 2 and all(1 <= s <= 30 for s in grid.shape), 'bad dimensions')
                _need(np.issubdtype(grid.dtype,np.integer) and np.all((0 <= grid)&(grid <= 9)), 'bad colors')
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            data = _read(inp)
            _need(np.array_equal(_predict(data),out), 'incorrect shadow output')
            _need(np.array_equal(inp == data['anchor_ink'],out == data['anchor_ink']), 'anchors changed')
            _need(data['ink'] not in out, 'arrow ink survived')
            f = _features(data)
            _need(all(x >= 2 for x in f['areas']), 'invisible instruction')
            _need(f['partial'] and f['a_witness'] and f['b_witness'] and f['reflection_witness'], 'missing parent interaction')
            analyses[split].append(f)
    taught_axes = set().union(*(f['reflection_axes'] for f in analyses['train']))
    _need(taught_axes == {False,True}, 'both reflection axes need visible training evidence')
    _need(any(f['scale_witness'] and f['dependency'] for f in analyses['train']), 'missing scaled simultaneous demonstration')
    _need(any(f['shape_witness'] and f['clip_witness'] for f in analyses['train']), 'missing silhouette/clipping demonstration')
    _need(not any(f['explicit'] for f in analyses['test']) or any(f['explicit'] for f in analyses['train']), 'undemonstrated explicit payload')


def self_test():
    """42 seeds, bounded aggregates, corruption checks, and exact scale tests."""
    # Hand-computed three-band exchange, independent of constructor geometry:
    # left 2 receives 3 over its whole footprint; its original 2 is still sent
    # into rows 4..9, columns 9..12 of the middle band. The right 2 is untouched.
    inp = np.full((12,24),2,dtype=np.int8)
    inp[:,8:16] = 3
    inp[4:6,2:4] = 9
    inp[6,11] = inp[6,20] = 9
    for p in ((1,6),(2,6),(3,6),(2,7),(8,9),(9,9),(10,9),(9,8)):
        inp[p] = 0
    expected = np.full((12,24),3,dtype=np.int8)
    expected[:,16:] = 2
    expected[4:10,9:13] = 2
    expected[4:6,2:4] = 9
    expected[6,11] = expected[6,20] = 9
    for permutation in (np.arange(10),np.arange(9,-1,-1),np.array([4,8,1,9,0,3,7,5,2,6])):
        for transpose in (False,True):
            x,y = permutation[inp],permutation[expected]
            if transpose:
                x,y = x.T,y.T
            _need(np.array_equal(solve(x),y), 'hand-computed locality/scale/clip regression')
    data = _read(inp)
    arrow = next(a for a in data['arrows'] if a['d'] == (0,1))
    # Add one fine source pixel: it cannot survive a half-size transfer exactly.
    fine = dict(data,cells=[list(c) for c in data['cells']])
    fine['cells'][arrow['source']].append((1,8))
    try:
        _shadow(fine,arrow)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous fractional silhouette was accepted')
    shapes,structures,ratios,regions,directions,sides = set(),set(),set(),set(),set(),set()
    repeated = irregular = explicit = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        if seed < 2:
            again = generate_puzzle(seed)
            _need(all(_key(x[k]) == _key(y[k]) for s in p for x,y in zip(p[s],again[s])
                      for k in ('input','output')), 'nondeterministic seed')
        for split in ('train','test'):
            for pair in p[split]:
                data = _read(pair['input'])
                f = _features(data)
                shapes.add(f['shape']); ratios.update(f['ratios']); regions.add(f['regions'])
                directions.update(f['directions']); sides.update(f['sides'])
                repeated += f['repeated']; irregular += bool(f['nonrectangular']); explicit += f['explicit']
                structures.add(hashlib.sha256(data['ids'].astype(np.uint8).tobytes()).digest())
                _need(np.array_equal(solve(pair['input']),pair['output']), 'solver mismatch')
    _need(len(shapes) >= 40 and len(structures) >= 150, 'collapsed geometry')
    _need({1,2,3} <= sides and len(ratios) >= 7 and len(regions) >= 4, 'collapsed scales/counts')
    _need(directions == set(DIRS) and repeated >= 5 and irregular >= 100 and explicit >= 5, 'missing structural variation')
    p = generate_puzzle(812,2,1)
    corrupted = {s:[{k:np.asarray(v).copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    corrupted['test'][0]['output'][0,0] = (int(corrupted['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a corrupted output')
    for counts in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(None,1)):
        try:
            generate_puzzle(0,*counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
