"""Perforated paint sheets: infer contours, apply parity, composite, crop.

Only NumPy and the standard library are used. All answer conventions are fixed
here or visible in the input. No generation descriptors reach solve().
"""
from collections import Counter
import json
import numpy as np


HOST_SEEDS = (0,333839185,672672350,74483778,194125245,936445935,631553991,
71742398,723300839,122176020,668032954,698760129,647336817,741688631,
461209229,417237129,729226479,77347521,692805445,817190519,906129618,
634503776,403614316,896750202,75406005,358432311,832747615,303437496,
622433893,761224453,626726107,522263445,302124519,663559905,573214866,
257268945,499405078,597773133,606204288,50911911,624722597,222781611)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.tobytes()


def _inside(p, b, strict=False):
    t, l, z, r = b
    y, x = p
    return t < y < z and l < x < r if strict else t <= y <= z and l <= x <= r


def _area(b):
    t, l, z, r = b
    return {(y, x) for y in range(t, z+1) for x in range(l, r+1)}


def _edge(b):
    t, l, z, r = b
    return ({(t, x) for x in range(l, r+1)} | {(z, x) for x in range(l, r+1)}
            | {(y, l) for y in range(t+1, z)} | {(y, r) for y in range(t+1, z)})


def _bbox(points):
    return (int(min(y for y, x in points)), int(min(x for y, x in points)),
            int(max(y for y, x in points)), int(max(x for y, x in points)))


def _halo(points):
    return {q for y, x in points for q in
            ((y,x),(y-1,x),(y+1,x),(y,x-1),(y,x+1))}


def _components(points):
    remaining = set(points)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            y, x = stack.pop()
            for q in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _contains(a, b):
    # A one-cell gap between contours, including in the unseen part.
    return a[0]+2 <= b[0] and a[1]+2 <= b[1] and b[2]+2 <= a[2] and b[3]+2 <= a[3]


def _separate(a, b):
    return a[2]+2 <= b[0] or b[2]+2 <= a[0] or a[3]+2 <= b[1] or b[3]+2 <= a[1]


def _compatible(a, b):
    return _contains(a, b) or _contains(b, a) or _separate(a, b)


def _find_cover(a, bg):
    candidates = []
    # Find a nonbackground, uniformly filled rectangular component enclosed by
    # a solid contrasting frame. A contour's empty interior has background fill.
    for color in np.unique(a):
        color = int(color)
        if color == bg:
            continue
        for group in _components(map(tuple, np.argwhere(a == color))):
            t, l, z, r = _bbox(group)
            if min(z-t+1, r-l+1) < 4 or len(group) != (z-t+1)*(r-l+1):
                continue
            if t == 0 or l == 0 or z == a.shape[0]-1 or r == a.shape[1]-1:
                continue
            box = (t-1,l-1,z+1,r+1)
            edge_colors = {int(a[p]) for p in _edge(box)}
            if len(edge_colors) == 1 and not edge_colors & {bg,color}:
                candidates.append((box,color,next(iter(edge_colors))))
    if len(candidates) != 1:
        raise ValueError("Cover is not uniquely identifiable")
    return candidates[0]


def _find_table(a, bg, cover, cover_fill, stitch):
    h, w = a.shape
    candidates = []
    mask = _area(cover)
    for n in range(2,5):
        for t in range(h-n-1):
            for l in range(w-3):
                box = (t,l,t+n+1,l+3)
                if _area(box) & mask:
                    continue
                if not all(a[p] == stitch for p in _edge(box)):
                    continue
                cues = tuple(map(int,a[t+1:t+n+1,l+1]))
                if len(set(cues)) != n or set(cues) & {bg,cover_fill,stitch}:
                    continue
                # The table has a local blank moat and a fixed vertical reading.
                if any(0 <= y < h and 0 <= x < w and a[y,x] != bg
                       for y,x in _halo(_area(box)) - _area(box)):
                    continue
                fills = tuple(map(int,a[t+1:t+n+1,l+2]))
                candidates.append((box,cues,fills))
    if len(candidates) != 1:
        raise ValueError("Paint table is not uniquely identifiable")
    return candidates[0]


def _complete(a, box, visible, cue, stitch):
    """Infer the fixed r+c repetition model, periods 1..3, with no guessed phase.

    Every residue has at least two exposed witnesses. If several explanations
    fit, they must agree on the entire missing edge, including under the frame.
    """
    hidden = sorted(_edge(box) - visible)
    fits = []
    for period in range(1,4):
        word, count = {}, Counter()
        for y, x in sorted(visible):
            q = (y+x) % period
            value = int(a[y,x])
            if q in word and word[q] != value:
                break
            word[q] = value
            count[q] += 1
        else:
            if len(word) == period and min(count.values()) >= 2:
                if set(word.values()) <= {cue,stitch} and cue in word.values():
                    fits.append((period,word,tuple(word[(y+x)%period] for y,x in hidden)))
    if not fits or len({f[2] for f in fits}) != 1:
        return None
    values = {p:int(a[p]) for p in visible}
    values.update(zip(hidden,fits[0][2]))
    return values, min(f[0] for f in fits)


def _read(input_grid):
    a = _grid(input_grid)
    bg = int(a[0,0])
    if any(int(a[p]) != bg for p in ((0,-1),(-1,0),(-1,-1))):
        raise ValueError("The four corners must show the background")
    cover, base, stitch = _find_cover(a,bg)
    table, cues, fills = _find_table(a,bg,cover,base,stitch)
    excluded = _area(cover) | _area(table)
    visible = {tuple(map(int,p)) for p in np.argwhere(a != bg)} - excluded
    if any(int(a[p]) not in set(cues) | {stitch} for p in visible):
        raise ValueError("Unidentified contour color")
    groups = _components(visible)
    candidates = []
    mask = _area(cover)
    for i, first in enumerate(groups):
        for j in range(i,len(groups)):
            points = first if i == j else first | groups[j]
            labels = {int(a[p]) for p in points} & set(cues)
            if len(labels) != 1:
                continue
            cue = next(iter(labels))
            box = _bbox(points)
            if min(box[2]-box[0],box[3]-box[1]) < 2:
                continue
            if _edge(box) - mask != points or _area(table) & _area(box):
                continue
            completion = _complete(a,box,points,cue,stitch)
            if completion is None:
                continue
            colors,period = completion
            candidates.append(dict(parts=frozenset((i,j)),box=box,cue=cue,
                                   values=colors,visible=points,period=period))
    solutions = []

    def partition(remaining, chosen):
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(chosen)
            return
        index = min(remaining)
        for obj in candidates:
            if index not in obj['parts'] or not obj['parts'] <= remaining:
                continue
            if any(obj['cue'] == old['cue'] and not _compatible(obj['box'],old['box'])
                   for old in chosen):
                continue
            if any(obj['box'] == old['box'] for old in chosen):
                continue
            partition(remaining-obj['parts'],chosen+[obj])

    partition(set(range(len(groups))),[])
    if len(solutions) != 1:
        raise ValueError("Contour geometry has no unique interpretation")
    contours = solutions[0]
    if set(o['cue'] for o in contours) != set(cues):
        raise ValueError("Every table row must name at least one contour")
    return dict(grid=a,background=bg,cover=cover,base=base,stitch=stitch,
                table=table,cues=cues,fills=fills,contours=contours)


def _pixel(o, p, mode='parity'):
    edges = [c for c in o if p in c['values']]
    if edges:
        return True,edges[0]['values'][p]
    depths = sum(_inside(p,c['box'],True) for c in o)
    if mode == 'union':
        return bool(depths),None
    if mode == 'outer_minus_all_inner':
        return depths == 1,None
    return bool(depths % 2),None


def _answer(scene, *, mode='parity', reverse=False, cue_edges=False, transparent_base=False, transparent_background=False):
    t,l,z,r = scene['cover']
    out = np.full((z-t-1,r-l-1),scene['base'],dtype=np.int8)
    rows = list(zip(scene['cues'],scene['fills']))
    if reverse:
        rows.reverse()
    by_cue = {cue:[o for o in scene['contours'] if o['cue'] == cue] for cue in scene['cues']}
    for y in range(t+1,z):
        for x in range(l+1,r):
            p = (y,x)
            for cue,paint in rows:
                covered,edge_color = _pixel(by_cue[cue],p,mode)
                if not covered:
                    continue
                if edge_color is None and ((transparent_base and paint == scene['base'])
                                          or (transparent_background and paint == scene['background'])):
                    continue
                if mode == 'lines_only' and edge_color is None:
                    continue
                out[y-t-1,x-l-1] = (cue if cue_edges else edge_color) if edge_color is not None else paint
                break
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only input pixels; return the composited strict interior of the cover."""
    return _answer(_read(input_grid))


def _axis(rng,depth,se_count,detach):
    # Random insets, margins and aspect ratios, independently on the two axes.
    for _ in range(80):
        t = int(rng.integers(2*depth+1,9))
        starts = sorted(map(int,rng.choice(np.arange(1,t-1),depth,replace=False)))
        if any(b-a < 2 for a,b in zip(starts,starts[1:])):
            continue
        last = t + int(rng.integers(4 if depth == 3 else 3,6))
        bottoms = [last]
        for _ in range(depth-1):
            bottoms.insert(0,bottoms[0]+int(rng.integers(3,5)))
        end = bottoms[0]+int(rng.integers(4 if detach else 3,6))
        exterior = 2*se_count-1
        extent = end+exterior+2+int(rng.integers(0,2))
        if extent > 30:
            continue
        se_starts = [t+1+2*i for i in range(se_count)]
        if detach:
            se_starts[-1] = max(se_starts[-1],bottoms[0]+2)
        if se_starts[-1] > end-2:
            continue
        se_ends = [end+exterior-2*i for i in range(se_count)]
        return extent,t,end,list(zip(starts,bottoms)),list(zip(se_starts,se_ends))
    return None


def _orient(h,w,cover,objects,turns,flip):
    def point(y,x):
        hh,ww = h,w
        for _ in range(turns):
            y,x = x,hh-1-y
            hh,ww = ww,hh
        if flip:
            x = ww-1-x
        return y,x
    def box(b):
        return _bbox([point(b[0],b[1]),point(b[2],b[3])])
    hh,ww = (w,h) if turns % 2 else (h,w)
    return hh,ww,box(cover),[(role,box(b)) for role,b in objects]


def _structured_geometry(rng,n,depth,detach):
    se_count = n-1+int(detach)
    ay,ax = _axis(rng,depth,se_count,detach),_axis(rng,depth,se_count,detach)
    if ay is None or ax is None:
        return None
    h,t,z,north_y,south_y = ay
    w,l,r,north_x,south_x = ax
    cover = (t,l,z,r)
    objects = [(0,(a,c,b,d)) for (a,b),(c,d) in zip(north_y,north_x)]
    for i,((a,b),(c,d)) in enumerate(zip(south_y,south_x)):
        objects.append((0 if detach and i == se_count-1 else i+1,(a,c,b,d)))
    return _orient(h,w,cover,objects,int(rng.integers(4)),bool(rng.integers(2)))


def _free_geometry(rng,n):
    h,w = map(int,rng.integers(19,31,size=2))
    oh,ow = map(int,rng.integers(5,13,size=2))
    t,l = int(rng.integers(3,h-oh-3)),int(rng.integers(3,w-ow-3))
    cover = (t,l,t+oh+1,l+ow+1)
    mask = _area(cover)
    objects,occupied = [],set()
    def accept(role,b):
        vis = _edge(b)-mask
        if not vis or _bbox(vis) != b or vis & _halo(occupied):
            return False
        if any(role == rr and not _compatible(b,bb) for rr,bb in objects):
            return False
        if any(b == bb for rr,bb in objects):
            return False
        objects.append((role,b))
        occupied.update(vis)
        return True
    for role in range(n):
        for _ in range(170):
            yy = sorted(map(int,rng.choice(np.arange(1,h-1),2,replace=False)))
            xx = sorted(map(int,rng.choice(np.arange(1,w-1),2,replace=False)))
            b = (yy[0],xx[0],yy[1],xx[1])
            if min(b[2]-b[0],b[3]-b[1]) < 3:
                continue
            if not (_area(b) & _area((t+1,l+1,t+oh,l+ow))):
                continue
            if accept(role,b):
                break
        else:
            return None
    # Add an offset hole or a separate piece if it has sufficient exposed evidence.
    for role in rng.permutation(n):
        if rng.random() > .65:
            continue
        parent = next(bb for rr,bb in objects if rr == role)
        for _ in range(60):
            if rng.random() < .65 and min(parent[2]-parent[0],parent[3]-parent[1]) >= 7:
                a,c,z,r = parent
                yy = sorted(map(int,rng.choice(np.arange(a+2,z-1),2,replace=False)))
                xx = sorted(map(int,rng.choice(np.arange(c+2,r-1),2,replace=False)))
            else:
                yy = sorted(map(int,rng.choice(np.arange(1,h-1),2,replace=False)))
                xx = sorted(map(int,rng.choice(np.arange(1,w-1),2,replace=False)))
            b = (yy[0],xx[0],yy[1],xx[1])
            if min(b[2]-b[0],b[3]-b[1]) >= 3 and accept(int(role),b):
                break
    if not any(_edge(b) & mask for role,b in objects):
        return None
    return h,w,cover,objects


def _table_locations(h,w,cover,objects,n):
    blocked = _area(cover)
    for role,b in objects:
        blocked |= _area(b)
    blocked = _halo(blocked)
    return [(t,l) for t in range(1,h-n-2) for l in range(1,w-4)
            if not (_area((t,l,t+n+1,l+3)) & blocked)]


def _render(rng,geometry,n,kind):
    h,w,cover,objects = geometry
    locations = _table_locations(h,w,cover,objects,n)
    if not locations:
        return None
    palette = list(map(int,rng.permutation(10)))
    bg,stitch,base = palette[:3]
    cues = palette[3:3+n]
    paints = list(map(int,rng.integers(0,10,size=n)))
    # Required demonstrations break the blank-paint/transparency ambiguity.
    if kind == 'first':
        paints[0],paints[1] = palette[3+n],palette[4+n]
        order = [0,1]
    elif kind == 'second':
        paints[0],paints[1],paints[2] = base,palette[3+n],bg
        order = [2,0,1]
    else:
        order = list(map(int,rng.permutation(n)))
        if rng.random() < .4:
            paints[int(rng.integers(n))] = base
        if rng.random() < .25:
            paints[int(rng.integers(n))] = bg
        if rng.random() < .25:
            paints[int(rng.integers(n))] = paints[int(rng.integers(n))]
    contours = []
    under = np.full((h,w),bg,dtype=np.int8)
    mask = _area(cover)
    for index,(role,b) in enumerate(objects):
        period = (1 if kind == 'first' else 3 if kind == 'second' and role == 0
                  else int(rng.integers(1,4)))
        word = ([cues[role]] if period == 1 else [cues[role],stitch] if period == 2
                else [cues[role],stitch,cues[role]] if rng.random() < .5
                else [stitch,cues[role],stitch])
        phase = int(rng.integers(period))
        values = {p:word[(p[0]+p[1]+phase)%period] for p in _edge(b)}
        for p,value in values.items():
            if p not in mask:
                under[p] = value
        contours.append(dict(cue=cues[role],box=b,values=values))
    # Ground truth uses binary masks per sheet and reverse-order composition.
    # The public solver instead selects a covering row separately at each pixel.
    t,l,z,r = cover
    output = np.full((z-t-1,r-l-1),base,dtype=np.int8)
    for role in reversed(order):
        material = np.zeros(output.shape,dtype=bool)
        edges = {}
        for obj in contours:
            if obj['cue'] != cues[role]:
                continue
            a,c,b,d = obj['box']
            y0,y1 = max(t+1,a+1),min(z,b)
            x0,x1 = max(l+1,c+1),min(r,d)
            if y0 < y1 and x0 < x1:
                material[y0-t-1:y1-t-1,x0-l-1:x1-l-1] ^= True
            edges.update({p:v for p,v in obj['values'].items() if _inside(p,cover,True)})
        output[material] = paints[role]
        for (y,x),value in edges.items():
            output[y-t-1,x-l-1] = value
    a = under.copy()
    a[t+1:z,l+1:r] = base
    for p in _edge(cover):
        a[p] = stitch
    tt,ll = locations[int(rng.integers(len(locations)))]
    for p in _edge((tt,ll,tt+n+1,ll+3)):
        a[p] = stitch
    for i,role in enumerate(order):
        a[tt+i+1,ll+1:ll+3] = cues[role],paints[role]
    pair = dict(input=a,output=output)
    try:
        scene = _read(a)
    except ValueError:
        return None
    if not np.array_equal(_answer(scene),output):
        return None
    if kind in ('first','second'):
        witness = _witnesses(scene,output)
        needed = {'holes','priority','paint','detached'} if kind == 'first' else {'island','opaque_base','opaque_background','phase','priority'}
        if not needed <= witness:
            return None
    return pair


def _witnesses(scene,out):
    found = set()
    alternatives = [('holes',dict(mode='union')),('island',dict(mode='outer_minus_all_inner')),
                    ('priority',dict(reverse=True)),('paint',dict(mode='lines_only')),
                    ('phase',dict(cue_edges=True)),('opaque_base',dict(transparent_base=True)),
                    ('opaque_background',dict(transparent_background=True))]
    for name,args in alternatives:
        if not np.array_equal(out,_answer(scene,**args)):
            found.add(name)
    for cue in scene['cues']:
        loops = [o for o in scene['contours'] if o['cue'] == cue]
        roots = [o for o in loops if not any(_contains(p['box'],o['box']) for p in loops if p is not o)]
        if len(roots) >= 2:
            # Every detached root has material or boundary within the answer.
            if all(any(_inside(p,scene['cover'],True) for p in _area(o['box'])) for o in roots):
                found.add('detached')
    if any(o['period'] == 3 for o in scene['contours']):
        found.add('period3')
    return found


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError("counts must be integers")
    num_train,num_test = int(num_train),int(num_test)
    if not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("Require 2..10 training, 1..4 test and num_train > num_test")
    rng = np.random.default_rng(seed)
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',num_train),('test',num_test)):
        for i in range(count):
            kind = 'first' if split == 'train' and i == 0 else 'second' if split == 'train' and i == 1 else 'other'
            n = 2 if kind == 'first' else 3 if kind == 'second' else 4 if split == 'test' else int(rng.integers(2,4))
            for attempt in range(200):
                if kind == 'other' and attempt < 70 and rng.random() < .65:
                    geometry = _free_geometry(rng,n)
                else:
                    depth = 2 if kind == 'first' else 3 if kind == 'second' else int(rng.integers(1,4))
                    detach = kind == 'first' or (kind == 'other' and depth < 3 and rng.random() < .4)
                    geometry = _structured_geometry(rng,n,depth,detach)
                if geometry is None:
                    continue
                pair = _render(rng,geometry,n,kind)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not sample a uniquely supported sheet scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen,training = set(),set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            a,out = _grid(pair['input']),_grid(pair['output'])
            assert _key(a) not in seen,"Duplicate input"
            seen.add(_key(a))
            scene = _read(a)
            assert np.array_equal(_answer(scene),out),"Incorrect sheet composite"
            t,l,z,r = scene['cover']
            assert out.shape == (z-t-1,r-l-1)
            assert len(scene['contours']) >= len(scene['cues'])
            assert any(_edge(o['box']) & _area(scene['cover']) for o in scene['contours'])
            assert np.any(out != scene['base']),"Entirely blank answer"
            assert all(o['period'] <= 3 for o in scene['contours'])
            # Completing edges must leave every exposed stitch unchanged.
            for o in scene['contours']:
                assert _bbox(o['visible']) == o['box']
                assert all(o['values'][p] == a[p] for p in o['visible'])
            witness = _witnesses(scene,out)
            if split == 'train':
                assert len(scene['cues']) <= 3
                training |= witness
            else:
                assert len(scene['cues']) == 4
    assert {'holes','island','priority','paint','phase','opaque_base','opaque_background','detached','period3'} <= training,"Training lacks semantic distinctions"


def self_test():
    inputs,outputs,layouts,patterns,roles,counts,topologies = set(),set(),set(),set(),set(),set(),set()
    signatures = set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(q['input']) for s in ('train','test') for q in p[s]))
        for split in ('train','test'):
            for pair in p[split]:
                scene = _read(pair['input'])
                inputs.add(pair['input'].shape)
                outputs.add(pair['output'].shape)
                layouts.add((scene['cover'],tuple(sorted((o['box'],o['cue']) for o in scene['contours']))))
                patterns.update(o['period'] for o in scene['contours'])
                roles.add(scene['background'])
                counts.add(len(scene['cues']))
                boxes = [o['box'] for o in scene['contours']]
                topologies.add((len(boxes),sum(_contains(a,b) for a in boxes for b in boxes if a != b),
                                sum(_separate(a,b) for i,a in enumerate(boxes) for b in boxes[i+1:])))
                assert np.array_equal(solve(pair['input']),pair['output'])
        # A repeated call reproduces every array, including its palette and split.
        if seed == 0:
            again = generate_puzzle(seed)
            assert all(np.array_equal(q[k],z[k]) for s in p for q,z in zip(p[s],again[s]) for k in q)
    assert len(signatures) == 42
    assert len(inputs) >= 40 and len(outputs) >= 40 and len(layouts) >= 150
    assert patterns == {1,2,3} and roles == set(range(10)) and counts == {2,3,4}
    assert len(topologies) >= 12
    for nt,ne in ((2,1),(10,4),(5,3),(7,1)):
        p = generate_puzzle(91,nt,ne)
        assert (len(p['train']),len(p['test'])) == (nt,ne)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(3.0,1),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Palette order is a real intervention: move complete cue/paint rows.
    pair = generate_puzzle(0,2,1)['train'][0]
    a = pair['input'].copy()
    scene = _read(a)
    t,l,z,r = scene['table']
    a[t+1:z,l+1:l+3] = a[t+1:z,l+1:l+3][::-1].copy()
    assert not np.array_equal(solve(a),pair['output'])
    # Literal color permutations retain the rule, including black as a cue.
    perm = np.array([7,6,5,4,3,2,1,0,9,8],dtype=np.int8)
    assert np.array_equal(solve(perm[pair['input']]),perm[pair['output']])
    bad = generate_puzzle(0,2,1)
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted answer accepted")
    return {'seeds':42,'input_shapes':len(inputs),'output_shapes':len(outputs),
            'layouts':len(layouts),'topologies':len(topologies),'periods':sorted(patterns)}


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
