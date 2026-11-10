"""Registered jigsaw printing: coordinate candidates filtered by exact cover.

Only the standard library and NumPy are used. No files are read or written.
The constructor paints from placement records; solve searches visible inputs.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


HOST_SEEDS = (0,138076036,875101419,766753488,771355718,704672878,499605230,343555512,589798976,892364976,334798861,671807857,775861622,816165321,926281422,517037222,277946905,6873120,768547986,914112299,726108486,291634284,645752898,526816349,931364953,967794084,695696223,452566958,134220300,368485501,605388962,306550107,374333129,372248062,3147037,85419332,796665563,375190187,761045226,833077574,498984440,49853696)


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        cells, stack = [first], [first]
        while stack:
            r, c = stack.pop()
            for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if p in unseen:
                    unseen.remove(p)
                    cells.append(p)
                    stack.append(p)
        result.append(np.asarray(sorted(cells), dtype=int))
    return result


def _parse(value):
    a = _grid(value)
    h, w = a.shape
    if min(h,w) < 7:
        raise ValueError("registered plate requires a frame and interior")
    pin, wall, bg = int(a[0,0]), int(a[0,1]), int(a[1,1])
    assert len({pin,wall,bg}) == 3
    assert all(a[r,c] == pin for r,c in ((0,0),(0,w-1),(h-1,0),(h-1,w-1)))
    assert all(a[r,c] == wall for r,c in ((0,1),(1,0),(0,w-2),(1,w-1),
                                         (h-1,1),(h-2,0),(h-1,w-2),(h-2,w-1)))
    structural = np.argwhere(a[1:-1,1:-1] == wall) + 1
    assert len(structural)
    lo, hi = structural.min(axis=0), structural.max(axis=0)
    r0,c0 = map(int,lo)
    r1,c1 = map(int,hi+1)
    assert 2 <= r0 < r1 <= h-2 and 2 <= c0 < c1 <= w-2
    plate = a[r0:r1,c0:c1]
    assert np.all((plate == wall) | (plate == bg))
    assert np.all(plate[[0,-1],:] == wall) and np.all(plate[:,[0,-1]] == wall)
    assert len(_components(plate == wall)) == 1
    target_cells = np.argwhere(a == bg)
    target_cells = target_cells[(target_cells[:,0] >= r0) & (target_cells[:,0] < r1)
                                & (target_cells[:,1] >= c0) & (target_cells[:,1] < c1)]
    assert len(target_cells)
    target = set(map(tuple,target_cells))
    outside = np.ones(a.shape, dtype=bool)
    outside[[0,-1],:] = False
    outside[:,[0,-1]] = False
    outside[r0:r1,c0:c1] = False
    templates = []
    for cells in _components(outside & (a != bg)):
        values = a[cells[:,0],cells[:,1]]
        colors = set(map(int,values)) - {pin}
        assert len(colors) == 1 and wall not in colors
        assert np.count_nonzero(values == pin) == 1
        color = colors.pop()
        pivot = cells[np.flatnonzero(values == pin)[0]]
        templates.append(dict(color=color, cells=cells, offsets=cells-pivot,
                              values=values.copy(), pivot=tuple(map(int,pivot))))
    assert 1 <= len(templates) <= 7
    assert len({t['color'] for t in templates}) == len(templates)
    templates.sort(key=lambda t: tuple(t['cells'][0]))
    body_colors = {t['color'] for t in templates}
    edge = set(map(int,np.concatenate((a[0,1:-1],a[-1,1:-1],a[1:-1,0],a[1:-1,-1]))))
    assert edge <= body_colors | {wall}
    assert np.array_equal(a[0,1:-1],a[-1,1:-1])
    assert np.array_equal(a[1:-1,0],a[1:-1,-1])
    for t in templates:
        t['rows'] = [r for r in range(1,h-1) if a[r,0] == a[r,-1] == t['color']]
        t['cols'] = [c for c in range(1,w-1) if a[0,c] == a[-1,c] == t['color']]
    return dict(grid=a, bg=bg, pin=pin, wall=wall, box=(r0,c0,r1,c1),
                templates=templates, target=target, holes=len(_components(plate == bg)))


def _candidates(p, unrestricted=False):
    cells = sorted(p['target'])
    indices = {cell:i for i,cell in enumerate(cells)}
    result = []
    for i,t in enumerate(p['templates']):
        if unrestricted:
            # Ablation only. No metadata or intended placements are involved.
            rows = range(p['box'][0]+1,p['box'][2]-1)
            cols = range(p['box'][1]+1,p['box'][3]-1)
        else:
            rows,cols = t['rows'],t['cols']
        for r in rows:
            for c in cols:
                shifted = t['offsets'] + (r,c)
                coordinates = list(map(tuple,shifted))
                if all(cell in indices for cell in coordinates):
                    bits = sum(1 << indices[cell] for cell in coordinates)
                    result.append(dict(template=i, anchor=(r,c), cells=shifted, bits=bits))
    return cells,result


def _render(p, chosen):
    out = p['grid'].copy()
    used = {x['template'] for x in chosen}
    for i in used:
        cells = p['templates'][i]['cells']
        out[cells[:,0],cells[:,1]] = p['bg']
    for x in chosen:
        t = p['templates'][x['template']]
        out[x['cells'][:,0],x['cells'][:,1]] = t['values']
    return out


def _search(p, unrestricted=False):
    cells,candidates = _candidates(p,unrestricted)
    by_cell = [[] for _ in cells]
    for i,x in enumerate(candidates):
        bits = x['bits']
        while bits:
            bit = bits & -bits
            by_cell[bit.bit_length()-1].append(i)
            bits -= bit
    full = (1 << len(cells))-1
    answers = {}

    def visit(covered,chosen):
        if len(answers) >= 2:
            return
        if covered == full:
            placements = [candidates[i] for i in chosen]
            out = _render(p,placements)
            answers.setdefault(out.tobytes(),(out,placements))
            return
        remaining = full ^ covered
        best = None
        while remaining:
            bit = remaining & -remaining
            options = [i for i in by_cell[bit.bit_length()-1]
                       if not candidates[i]['bits'] & covered]
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
            remaining -= bit
        for i in best:
            visit(covered | candidates[i]['bits'],chosen+[i])

    visit(0,[])
    return list(answers.values()),candidates


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only visible tiles, pin legend, frame coordinates and empty plate cells."""
    p = _parse(input_grid)
    answers,_ = _search(p)
    if len(answers) != 1:
        raise ValueError("plate must have exactly one distinct colored completion")
    return answers[0][0].copy()


def _profiles(rng,width,n):
    root = rng.integers(0,3,size=width)
    root -= root.min()
    if not root.any():
        root[int(rng.integers(width))] = 1
    cuts = [root]
    for _ in range(n-1):
        q = rng.integers(0,3,size=width)
        q -= q.min()
        if not q.any():
            q[int(rng.integers(width))] = 1
        old = cuts[-1]
        offset = max(int(np.max(old+1-q)),
                     int(np.max(np.maximum(old[:-1],old[1:])+1-np.minimum(q[:-1],q[1:]))))
        cuts.append(q+offset+int(rng.integers(2)))
    cuts.append(np.full(width,int(cuts[-1].max())+int(rng.integers(2,4))))
    masks = [np.asarray([(r,c) for c in range(width) for r in range(int(a[c]),int(b[c]))],dtype=int)
             for a,b in zip(cuts,cuts[1:])]
    return cuts,masks


def _pin_choices(rng,masks,positions,assignments,type_count):
    """Visible pins make arbitrary box parities and shared coordinate axes safe."""
    pivots = [None]*len(masks)
    def visit(layer,used_rows,used_cols):
        if layer == len(masks):
            return True
        for idx in rng.permutation(len(masks[layer])):
            point = masks[layer][int(idx)]
            groups = []
            for typ in range(type_count):
                occurrences = [pos+point for pos,types in zip(positions,assignments) if types[layer] == typ]
                if occurrences:
                    groups.append(({int(v[0]) for v in occurrences},{int(v[1]) for v in occurrences}))
            rs,cs = set(used_rows),set(used_cols)
            okay = True
            for rows,cols in groups:
                if rows & rs or cols & cs:
                    okay = False
                    break
                rs.update(rows); cs.update(cols)
            if okay:
                pivots[layer] = point.copy()
                if visit(layer+1,rs,cs):
                    return True
        return False
    if not visit(0,set(),set()):
        raise ValueError("no nonconflicting registration pins")
    return pivots


def _scatter(rng,base,box,templates,bg):
    h,w = base.shape
    r0,c0,r1,c1 = box
    for _ in range(24):
        out = base.copy()
        sources = [None]*len(templates)
        forbidden = np.zeros(base.shape,dtype=bool)
        forbidden[:2,:] = forbidden[-2:,:] = True
        forbidden[:,:2] = forbidden[:,-2:] = True
        forbidden[r0-1:r1+1,c0-1:c1+1] = True
        for i in sorted(range(len(templates)),key=lambda j:-len(templates[j]['local'])):
            t = templates[i]
            local = t['local']
            bh,bw = local.max(axis=0)+1
            valid = np.ones((h-bh+1,w-bw+1),dtype=bool)
            for r,c in local:
                valid &= ~forbidden[r:r+h-bh+1,c:c+w-bw+1]
            origins = np.argwhere(valid)
            if not len(origins):
                break
            coords = local + origins[int(rng.integers(len(origins)))]
            sources[i] = coords
            out[coords[:,0],coords[:,1]] = t['values']
            for r,c in coords:
                forbidden[max(0,r-1):r+2,max(0,c-1):c+2] = True
        else:
            return out,sources
    raise ValueError("source tile clearance")


def _construct(rng,kind,layers=None):
    width = int(rng.integers(3,7))
    n = layers if layers is not None else 2 if kind == 'cartesian' else int(rng.integers(2,4))
    cuts,masks = _profiles(rng,width,n)
    depth = int(cuts[-1][0])
    if kind == 'cartesian':
        if depth > 8:
            raise ValueError("four-window height")
        dx,dy = width+int(rng.integers(1,4)),depth+int(rng.integers(1,4))
        positions = [(0,0),(0,dx),(dy,0),(dy,dx)]
    elif kind == 'twins':
        dx,dy = width+int(rng.integers(1,4)),int(rng.integers(2,6))
        positions = [(0,0),(dy,dx)]
    elif kind == 'elbow':
        if depth > 8:
            raise ValueError("three-window height")
        dx,dy = width+int(rng.integers(1,4)),depth+int(rng.integers(1,4))
        positions = [(0,0),(0,dx),(dy,0),(dy,dx)]
        positions.pop(int(rng.integers(4)))
    elif kind == 'diagonal':
        dx,dy = width+int(rng.integers(1,4)),int(rng.integers(1,7))
        positions = [(0,0),(dy,dx)]
    else:
        positions = [(0,0)]
    positions = [np.asarray(x,dtype=int) for x in positions]
    assignments = [list(range(n)) for _ in positions]
    if kind == 'twins':
        assignments = [[0]+list(range(2,n+1)),[1]+list(range(2,n+1))]
    type_count = n+(kind == 'twins')
    pivots = _pin_choices(rng,masks,positions,assignments,type_count)
    ph = max(int(pos[0]) for pos in positions)+depth+2
    pw = max(int(pos[1]) for pos in positions)+width+2
    min_h,min_w = max(20,ph+6),max(22,pw+width+7)
    if min_h > 30 or min_w > 30:
        raise ValueError("plate clearance")
    h,w = int(rng.integers(min_h,31)),int(rng.integers(min_w,31))
    r0 = int(rng.integers(3,h-ph-2))
    c0 = 3 if rng.random() < .5 else w-pw-3
    r1,c1 = r0+ph,c0+pw
    positions = [pos+(r0+1,c0+1) for pos in positions]
    bg,wall,pin,*colors = map(int,rng.permutation(10))
    base = np.full((h,w),bg,dtype=np.int8)
    base[[0,-1],:] = wall
    base[:,[0,-1]] = wall
    base[0,0] = base[0,-1] = base[-1,0] = base[-1,-1] = pin
    base[r0:r1,c0:c1] = wall
    intended = []
    target = set()
    templates = [None]*type_count
    for pos,types in zip(positions,assignments):
        for j,typ in enumerate(types):
            cells = masks[j]+pos
            target.update(map(tuple,cells))
            base[cells[:,0],cells[:,1]] = bg
            origin = masks[j].min(axis=0)
            local = masks[j]-origin
            values = np.full(len(local),colors[typ],dtype=np.int8)
            pin_index = np.flatnonzero(np.all(masks[j] == pivots[j],axis=1))[0]
            values[pin_index] = pin
            if templates[typ] is None:
                templates[typ] = dict(local=local,values=values, color=colors[typ],
                                      rows=set(),cols=set(),offsets=masks[j]-pivots[j])
            anchor = pos+pivots[j]
            templates[typ]['rows'].add(int(anchor[0]))
            templates[typ]['cols'].add(int(anchor[1]))
            intended.append((typ,cells,values))
    # Mark one geometrically legal but globally incompatible translated copy.
    occupied_rows = set().union(*(t['rows'] for t in templates))
    options = []
    for typ,t in enumerate(templates):
        for rr in range(r0+1,r1-1):
            if rr in occupied_rows:
                continue
            for cc in t['cols']:
                shifted = t['offsets']+(rr,cc)
                if all(tuple(cell) in target for cell in shifted):
                    options.append((typ,rr))
    if not options:
        raise ValueError("need a valid rejected coordinate candidate")
    typ,rr = options[int(rng.integers(len(options)))]
    templates[typ]['rows'].add(rr)
    # A visibly unrequested library tile, retained in the answer.
    if kind == 'cartesian' or rng.random() < .35:
        local = np.asarray([(0,0),(1,0),(1,1),(2,0)],dtype=int)
        if rng.random() < .5:
            local[:,1] = 1-local[:,1]
        values = np.full(len(local),colors[type_count],dtype=np.int8)
        values[int(rng.integers(len(values)))] = pin
        templates.append(dict(local=local,values=values,color=colors[type_count],rows=set(),cols=set()))
    inp,sources = _scatter(rng,base,(r0,c0,r1,c1),templates,bg)
    for t in templates:
        for r in t['rows']:
            inp[r,0] = inp[r,-1] = t['color']
        for c in t['cols']:
            inp[0,c] = inp[-1,c] = t['color']
    out = inp.copy()
    # Constructor knows placements, but never calls solve to make its labels.
    for cells in sources[:type_count]:
        out[cells[:,0],cells[:,1]] = bg
    for _,cells,values in intended:
        out[cells[:,0],cells[:,1]] = values
    turns = int(rng.integers(4))
    inp,out = np.rot90(inp,turns),np.rot90(out,turns)
    if rng.random() < .5:
        inp,out = inp[:,::-1],out[:,::-1]
    pair = dict(input=inp.copy(),output=out.copy())
    info = _validate_pair(pair)
    assert info['rejected_candidates'] > 0 and info['stamp_all_difference'] > 0 and info['off_center'] > 0
    if kind == 'twins':
        assert info['coordinate_necessary']
    return pair


def _stamp_all(p,candidates):
    # Deliberately wrong B-only rule, even granting it wall avoidance.
    return _render(p,candidates)


def _features(p,answer,chosen,candidates):
    selected = {(x['template'],x['anchor']) for x in chosen}
    rejected = sum((x['template'],x['anchor']) not in selected for x in candidates)
    # A coordinate-free witness: identical registered masks in two colors can
    # exchange all their destinations. This is an actual alternative output.
    used = {x['template'] for x in chosen}
    coordinate_necessary = False
    for i in used:
        for j in used:
            if j <= i:
                continue
            a,b = p['templates'][i],p['templates'][j]
            aset = set(map(tuple,a['offsets']))
            bset = set(map(tuple,b['offsets']))
            if aset == bset and a['color'] != b['color']:
                exchanged = [dict(x,template=j if x['template']==i else i if x['template']==j else x['template']) for x in chosen]
                # Values arrays may have different source traversal order.
                for x in exchanged:
                    t = p['templates'][x['template']]
                    x['cells'] = t['offsets']+x['anchor']
                if not np.array_equal(_render(p,exchanged),answer):
                    coordinate_necessary = True
    return dict(shape=p['grid'].shape,roles=(p['bg'],p['wall'],p['pin']),
                holes=p['holes'],templates=len(p['templates']),copies=len(chosen),
                layers=len(chosen)//p['holes'],
                inactive=len(p['templates'])-len(used),rejected_candidates=rejected,
                candidate_count=len(candidates),coordinate_necessary=coordinate_necessary,
                off_center=sum(not np.array_equal(2*np.asarray(t['pivot']),
                               t['cells'].min(axis=0)+t['cells'].max(axis=0))
                               for i,t in enumerate(p['templates']) if i in used),
                stamp_all_difference=int(np.count_nonzero(_stamp_all(p,candidates) != answer)),
                products=tuple((len(t['rows']),len(t['cols'])) for t in p['templates'] if t['rows'] and t['cols']),
                masks=tuple((tuple(map(tuple,t['offsets']))) for t in p['templates']),
                panel_shape=(p['box'][2]-p['box'][0],p['box'][3]-p['box'][1]))


def _validate_pair(pair):
    assert isinstance(pair,dict) and set(pair) == {'input','output'}
    p = _parse(pair['input'])
    actual = _grid(pair['output'])
    assert actual.shape == p['grid'].shape
    answers,candidates = _search(p)
    assert len(answers) == 1, 'nonunique or incomplete registered tiling'
    expected,chosen = answers[0]
    assert np.array_equal(actual,expected), 'wrong tiling or changed preserved cells'
    assert not np.array_equal(actual,p['grid'])
    # Independent conservation and geometry checks on the recovered placements.
    occupied = set()
    body_counts = Counter()
    for x in chosen:
        t = p['templates'][x['template']]
        cells = set(map(tuple,x['cells']))
        assert not occupied & cells
        occupied.update(cells)
        assert cells == {tuple(v) for v in t['offsets']+x['anchor']}
        assert np.array_equal(actual[x['cells'][:,0],x['cells'][:,1]],t['values'])
        body_counts[t['color']] += int(np.count_nonzero(t['values'] == t['color']))
    assert occupied == p['target']
    assert np.array_equal(actual == p['wall'],p['grid'] == p['wall'])
    assert np.array_equal(actual[[0,-1],:],p['grid'][[0,-1],:])
    assert np.array_equal(actual[:,[0,-1]],p['grid'][:,[0,-1]])
    for t in p['templates']:
        source = actual[t['cells'][:,0],t['cells'][:,1]]
        if t['color'] in body_counts:
            assert np.all(source == p['bg'])
        else:
            assert np.array_equal(source,t['values'])
    return _features(p,actual,chosen,candidates)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    fs = {'train':[],'test':[]}
    for split in fs:
        for pair in puzzle[split]:
            key = _key(np.asarray(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            fs[split].append(_validate_pair(pair))
    assert any(f['coordinate_necessary'] for f in fs['train']), 'missing coordinate witness'
    assert all(f['rejected_candidates'] and f['stamp_all_difference'] for f in fs['train'])
    assert any(any(r >= 2 and c >= 2 for r,c in f['products']) for f in fs['train'])
    assert any(f['inactive'] for f in fs['train'])
    assert any(f['off_center'] for f in fs['train']), 'pin alignment underdemonstrated'
    missing = all(f['holes'] != 3 for f in fs['train']) and all(f['holes'] == 3 for f in fs['test'])
    deeper = all(f['layers'] == 2 for f in fs['train']) and all(f['layers'] == 3 for f in fs['test'])
    assert missing or deeper, 'missing intended structural generalization'
    return None


def _count(value,name,lo,hi):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(name+' must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(name+' outside legal range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ne = _count(num_train,'num_train',2,10),_count(num_test,'num_test',1,4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    shift = int(rng.integers(2))  # Sampling policy; never a semantic parameter.
    puzzle = {'train':[],'test':[]}
    seen = set()
    for split,count in (('train',nt),('test',ne)):
        for i in range(count):
            if split == 'train' and i < 2:
                kind = ('twins','cartesian')[i]
            elif split == 'test' and shift == 0:
                kind = 'elbow'
            else:
                choices = ('single','diagonal','twins') if split == 'test' else ('single','diagonal','cartesian','twins')
                kind = choices[int(rng.integers(len(choices)))]
            layers = (2 if split == 'train' else 3) if shift else None
            for _ in range(160):
                try:
                    pair = _construct(rng,kind,layers)
                except (ValueError,AssertionError):
                    continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded construction exhausted for '+kind)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    # Random order prevents the two pedagogic witness types becoming positions.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _digest(puzzle):
    payload = {s:[{k:np.asarray(v).tolist() for k,v in p.items()} for p in puzzle[s]] for s in ('train','test')}
    return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()


def self_test():
    """Bounded streaming sweep: 42 puzzles, no retained grids or disk writes."""
    shapes,roles,masks,panels,products,structures = set(),set(),set(),set(),set(),set()
    holes,copies,counts = set(),set(),set()
    signatures = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        witness = next(pair for pair in puzzle['train'] if _validate_pair(pair)['coordinate_necessary'])
        assert len(_search(_parse(witness['input']),unrestricted=True)[0]) == 2
        for pair in puzzle['train']+puzzle['test']:
            f = _validate_pair(pair)
            shapes.add(f['shape']); roles.add(f['roles']); panels.add(f['panel_shape'])
            masks.update(f['masks']); products.update(f['products'])
            holes.add(f['holes']); copies.add(f['copies']); counts.add(f['templates'])
            a = np.asarray(pair['input'])
            mapping = {}
            normalized = tuple(mapping.setdefault(int(c),len(mapping)) for c in a.flat)
            structures.add((a.shape,normalized))
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(roles) >= 50 and len(masks) >= 70
    assert len(panels) >= 25 and len(structures) >= 200
    assert {1,2,3,4} <= holes and len(copies) >= 5 and len(counts) >= 3
    assert any(r >= 2 and c >= 2 for r,c in products)
    assert any(r != c for r,c in products)
    for nt in range(2,11):
        for ne in range(1,min(nt-1,4)+1):
            validate_generated_puzzle(generate_puzzle(90210+nt*5+ne,nt,ne))
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    assert _digest(generate_puzzle(13,2,1)) == _digest(generate_puzzle(13,2,1))
    # A wrong output must be rejected.
    p = generate_puzzle(77,2,1)
    bad = {s:[{k:np.asarray(v).copy() for k,v in x.items()} for x in p[s]] for s in p}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('identity mutation accepted')
    # Uniform palette bijections do not change registration or packing.
    for pair in p['train']+p['test']:
        perm = np.array([3,8,1,6,9,0,4,2,7,5],dtype=np.int8)
        assert np.array_equal(solve(perm[pair['input']]),perm[pair['output']])
    return {'seeds':42,'puzzles':len(signatures),'shapes':len(shapes),'roles':len(roles),
            'tile_masks':len(masks),'panel_shapes':len(panels),'structures':len(structures),
            'hole_counts':sorted(holes),'copy_counts':sorted(copies),'template_counts':sorted(counts)}


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
