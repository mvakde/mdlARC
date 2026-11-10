"""Defect odometers: unary arrow payloads address true erasures by rank.

Only NumPy and the standard library are used. The reader reconstructs the
surface and arrows from the input; generation records never enter solve().
No files, caches, execution traces, or logs are written by this module.
"""
from itertools import permutations
import hashlib
import numpy as np

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0,63329274,43755447,85459012,384428429,972300774,787622532,262388807,475306798,873533175,415822957,31771572,974575868,200422470,686621717,904827286,906949903,899168828,876037864,813461114,977169279,476690141,321008451,404856625,547582260,885532237,406393905,105526594,606224877,861453731,862788013,183092294,256016307,782320311,312927929,621861590,995073029,224147321,111919074,787661514,8092597,43754616)


def _components(points, diagonal=True):
    remaining = set(points)
    steps = [(r,c) for r in (-1,0,1) for c in (-1,0,1)
             if (r or c) and (diagonal or abs(r)+abs(c)==1)]
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            r,c = stack.pop()
            for dr,dc in steps:
                p = (r+dr,c+dc)
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'invalid dimensions'
    assert np.issubdtype(a.dtype,np.integer) and np.all((0 <= a)&(a <= 9)), 'invalid cells'
    return a.astype(np.int8,copy=True)


def _key(a):
    return a.shape,a.tobytes()


def _border_background(a):
    border = np.concatenate((a[0],a[-1],a[1:-1,0],a[1:-1,-1]))
    colors,counts = np.unique(border,return_counts=True)
    assert counts.max() > len(border)/2, 'no clear exterior'
    return int(colors[np.argmax(counts)])


def _core(a,color):
    """Pixels belonging to a monochromatic 2x2 square, not smoothing votes."""
    mask = a == color
    squares = mask[:-1,:-1]&mask[1:,:-1]&mask[:-1,1:]&mask[1:,1:]
    result = np.zeros(a.shape,dtype=bool)
    for dr,dc in ((0,0),(0,1),(1,0),(1,1)):
        result[dr:dr+squares.shape[0],dc:dc+squares.shape[1]] |= squares
    return result


def _paint(shape,bg,boxes):
    a = np.full(shape,bg,dtype=np.int8)
    for r0,c0,r1,c1,color in boxes:
        a[r0:r1,c0:c1] = color
    return a


def _recover_surface(a):
    bg = _border_background(a)
    cores = {int(c):_core(a,int(c)) for c in np.unique(a) if c != bg}
    fills = tuple(sorted(c for c,m in cores.items() if int(m.sum()) >= 20))
    assert len(fills)==2, 'exactly two dense surface fills are required'
    boxes = []
    for color in fills:
        for group in _components(map(tuple,np.argwhere(a==color))):
            evidence = [p for p in group if cores[color][p]]
            if len(evidence) < 16:
                continue
            rs,cs = zip(*evidence)
            box = (min(rs),min(cs),max(rs)+1,max(cs)+1,color)
            assert min(box[2]-box[0],box[3]-box[1]) >= 5, 'surface strip too thin'
            boxes.append(box)
    assert 2 <= len(boxes) <= 3, 'unsupported rectangle count'
    # Intact two-cell runs expose every bounding side. Cover order is read
    # by comparing all possible visible compositions, not by color ordering.
    observed = np.where(np.isin(a,(bg,*fills)),a,bg)
    candidates = {}
    for order in permutations(boxes):
        scene = _paint(a.shape,bg,order)
        k = scene.tobytes()
        loss = int(np.count_nonzero(scene!=observed))
        if k not in candidates:
            candidates[k] = (loss,scene,order)
    ranked = sorted(candidates.values(),key=lambda v:v[0])
    loss,clean,order = ranked[0]
    assert len(ranked)==1 or ranked[1][0] >= loss+2, 'ambiguous covering order'
    assert loss <= max(12,int(.22*np.count_nonzero(clean!=bg))), 'surface too damaged'
    return clean,bg,fills,tuple(order)


def _ray(shape,tip,direction):
    r,c = tip
    dr,dc = direction
    result = []
    while True:
        r,c = r+dr,c+dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            return result
        result.append((r,c))


def _read_frame(frame,direction):
    dr,dc = direction
    er,ec = -dc,dr
    rows = {}
    for r,c in frame:
        t,x = -(r*dr+c*dc),r*er+c*ec
        rows.setdefault(t,[]).append(x)
    t0,t1 = min(rows),max(rows)
    if t1-t0 < 2 or len(rows[t0]) != 1 or len(rows) != t1-t0+1:
        return None
    tip_x = rows[t0][0]
    left = right = tip_x
    split = False
    cavity = set()
    for t in range(t0+1,t1+1):
        xs = sorted(rows[t])
        if len(xs)==1 and not split and xs[0]==tip_x:
            continue
        if len(xs)!=2 or xs[0]>left or xs[-1]<right:
            return None
        left,right = xs
        split = True
        for x in range(left+1,right):
            cavity.add((-t*dr+x*er,-t*dc+x*ec))
    if not cavity or not left < tip_x < right:
        return None
    tip = (-t0*dr+tip_x*er,-t0*dc+tip_x*ec)
    return tip,cavity


def _read_arrows(a,bg,fills):
    arrows = []
    for color in sorted(set(map(int,np.unique(a)))-{bg,*fills}):
        for frame in _components(map(tuple,np.argwhere(a==color))):
            if len(frame)<5:
                continue
            alternatives = []
            for direction in DIRECTIONS:
                geom = _read_frame(frame,direction)
                if geom is None:
                    continue
                tip,cavity = geom
                if any(not(0<=r<a.shape[0] and 0<=c<a.shape[1]) for r,c in cavity):
                    continue
                values = {int(a[p]) for p in cavity}-{bg}
                if len(values)!=1 or values & {color,*fills}:
                    continue
                payload_color = next(iter(values))
                payload = {p for p in cavity if a[p]==payload_color}
                if not 1<=len(payload)<len(cavity):
                    continue
                alternatives.append(dict(frame=frame,cavity=cavity,tip=tip,direction=direction,
                    frame_color=color,payload_color=payload_color,payload=payload,q=len(payload)))
            assert len(alternatives)<=1, 'ambiguous arrow direction'
            arrows.extend(alternatives)
    assert arrows, 'no directional query'
    used = set()
    for arrow in arrows:
        assert not used & (arrow['frame']|arrow['cavity']), 'overlapping query envelopes'
        used |= arrow['frame']|arrow['cavity']
    non_surface = set(map(tuple,np.argwhere(~np.isin(a,(bg,*fills)))))
    assert non_surface == set().union(*(o['frame']|o['payload'] for o in arrows)), 'unowned query marks'
    return sorted(arrows,key=lambda o:o['tip'])


def _ring(shape,p):
    r,c = p
    return [(y,x) for y in range(max(0,r-1),min(shape[0],r+2))
            for x in range(max(0,c-1),min(shape[1],c+2)) if (y,x)!=p]


def _overlap_level(analysis):
    """0=disjoint, 1=shared rings, 2=a center lies in another ring."""
    points = list(analysis['targets'])
    distances = [max(abs(p[0]-q[0]),abs(p[1]-q[1]))
                 for i,p in enumerate(points) for q in points[:i]]
    return 2 if 1 in distances else 1 if 2 in distances else 0


def _render_answer(clean,bg,fills,arrows,holes):
    """Simultaneous demands; centers take priority over every ring."""
    targets,demands = {},{}
    frame_cells = set().union(*(o['frame'] for o in arrows))
    for arrow in arrows:
        stream = [p for p in _ray(clean.shape,arrow['tip'],arrow['direction']) if p in holes]
        assert len(stream)>=arrow['q'], 'query rank exceeds true erasures'
        p = stream[arrow['q']-1]
        color = arrow['payload_color']
        assert p not in targets or targets[p]==color, 'incompatible shared center'
        targets[p] = color
    for p in targets:
        complement = next(c for c in fills if c!=int(clean[p]))
        for q in _ring(clean.shape,p):
            if q in targets:
                continue
            assert q not in frame_cells, 'symbol touches an arrow'
            assert q not in demands or demands[q]==complement, 'incompatible ring overlap'
            demands[q] = complement
    out = clean.copy()
    for arrow in arrows:
        for p in arrow['frame']:
            assert clean[p]==bg, 'query overlaps surface'
            out[p] = arrow['frame_color']
    for p,color in demands.items():
        out[p] = color
    for p,color in targets.items():
        out[p] = color
    return out,targets


def _analyze(a):
    clean,bg,fills,boxes = _recover_surface(a)
    arrows = _read_arrows(a,bg,fills)
    holes = set(map(tuple,np.argwhere((clean!=bg)&(a==bg))))
    out,targets = _render_answer(clean,bg,fills,arrows,holes)
    return dict(clean=clean,bg=bg,fills=fills,boxes=boxes,arrows=arrows,holes=holes,targets=targets,output=out)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read an unlabeled input only. Does not mutate it or consult generation."""
    return _analyze(_grid(input_grid))['output']


def _profile(rng,anchor=False):
    if anchor:
        return (2,2,0,3)
    return (int(rng.integers(1,3)),int(rng.integers(1,3)),
            int(rng.integers(0,2)),int(rng.integers(2,4)))


def _arrow_geometry(tip,direction,profile):
    a,b,stem,tail = profile
    dr,dc = direction
    er,ec = -dc,dr
    frame,cavity = set(),set()
    depth = stem+max(a,b)+tail
    for t in range(depth):
        step = max(0,t-stem)
        left,right = -min(a,step),min(b,step)
        for x in {left,right}:
            frame.add((tip[0]-t*dr+x*er,tip[1]-t*dc+x*ec))
        for x in range(left+1,right):
            cavity.add((tip[0]-t*dr+x*er,tip[1]-t*dc+x*ec))
    return frame,cavity


def _sample_surface(rng,shape,window,count):
    r0,c0,r1,c1 = window
    hh,ww = r1-r0,c1-c0
    boxes = []
    for _ in range(100):
        # Three-piece chains need room for two separated pieces of one fill.
        max_h = max(5,hh*3//4) if count==3 else hh
        max_w = max(5,ww*3//4) if count==3 else ww
        h = int(rng.integers(5,max_h+1));w = int(rng.integers(5,max_w+1))
        if h*w<40:
            continue
        r = int(rng.integers(r0,r1-h+1));c = int(rng.integers(c0,c1-w+1))
        b = (r,c,r+h,c+w,1+len(boxes)%2)
        def area(x,y):
            return max(0,min(x[2],y[2])-max(x[0],y[0]))*max(0,min(x[3],y[3])-max(x[1],y[1]))
        if boxes:
            overlap = area(b,boxes[-1])
            if not 4<=overlap<=.45*min(h*w,(boxes[-1][2]-boxes[-1][0])*(boxes[-1][3]-boxes[-1][1])):
                continue
            if any(not (b[2]+1<=o[0] or o[2]+1<=b[0] or b[3]+1<=o[1] or o[3]+1<=b[1]) for o in boxes[:-1]):
                continue
        boxes.append(b)
        if len(boxes)==count:
            ordered = [boxes[int(i)] for i in rng.permutation(count)]
            clean = _paint(shape,0,ordered)
            try:
                recovered,bg,_,read_boxes = _recover_surface(clean)
                if bg==0 and np.array_equal(recovered,clean) and sorted(read_boxes)==sorted(boxes):
                    return clean,ordered
            except AssertionError:
                pass
            return None
    return None


def _place_arrow(rng,clean,bg,direction,q,profile,others,desired=None):
    points = np.argwhere(clean!=bg)
    r0,c0 = points.min(axis=0);r1,c1 = points.max(axis=0)
    dr,dc = direction
    axes = list(range(int(r0),int(r1)+1)) if dc else list(range(int(c0),int(c1)+1))
    rng.shuffle(axes)
    for axis in axes:
        gap = int(rng.integers(1,3))
        tip = ((axis,int(c0)-gap) if dc==1 else (axis,int(c1)+gap) if dc==-1
               else (int(r0)-gap,axis) if dr==1 else (int(r1)+gap,axis))
        frame,cavity = _arrow_geometry(tip,direction,profile)
        occupied = frame|cavity
        if len(cavity)<=q or any(not(0<=r<clean.shape[0] and 0<=c<clean.shape[1]) for r,c in occupied):
            continue
        if any(clean[p]!=bg for p in occupied):
            continue
        if any(max(abs(p[0]-z[0]),abs(p[1]-z[1]))<=1 for other in others for p in occupied for z in other['frame']|other['cavity']):
            continue
        fg = [p for p in _ray(clean.shape,tip,direction) if clean[p]!=bg]
        if len(fg)<2*q+2:
            continue
        possible = [i for i in range(q,len(fg)-1) if desired is None or clean[fg[i]]==desired]
        rng.shuffle(possible)
        for j in possible:
            # A sparse preferred arrangement keeps intact surface evidence.
            prior = list(range(j));rng.shuffle(prior)
            selected = []
            for i in prior:
                if all(abs(i-k)>=2 for k in selected+[j]):
                    selected.append(i)
                if len(selected)==q-1:
                    break
            if len(selected)<q-1:
                continue
            later = list(range(j+1,len(fg)));rng.shuffle(later)
            last = next((i for i in later if i-j>=2),later[0])
            holes = {fg[i] for i in selected+[j,last]}
            wrong = next((p for p in fg[:j] if p not in holes),None)
            if wrong is None:
                continue
            return dict(frame=frame,cavity=cavity,tip=tip,direction=direction,q=q,
                        holes=holes,wrong=wrong,profile=profile)
    return None


def _construct_pair(rng,primary_direction,q,desired,palette,anchor=False,test=False,attempt=0):
    # Only distribution preferences depend on split; decoding never does.
    number = int(rng.choice([1,1,2,2,3] if test else [1,1,1,2,2]))
    directions = [primary_direction]+[DIRECTIONS[int(rng.integers(4))] for _ in range(number-1)]
    if attempt>=60:
        number,directions = 1,[primary_direction]
    margins = [int(rng.integers(1,4)) for _ in range(4)] # top,right,bottom,left
    for d in directions:
        side = {(-1,0):2,(0,1):3,(1,0):0,(0,-1):1}[d]
        margins[side] = 7
    h = int(rng.integers(24 if test else 22,31));w = int(rng.integers(24 if test else 22,31))
    if attempt>=60:
        h=w=30
    top,right,bottom,left = margins
    window = (top,left,h-bottom,w-right)
    if window[2]-top<8 or window[3]-left<8:
        return None
    count = int(rng.choice([2,2,2,3]))
    scene = _sample_surface(rng,(h,w),window,count)
    if scene is None:
        # Parameterized two-rectangle reserve, still fitted from input later.
        r0,c0,r1,c1 = window
        boxes = [(r0,c0,r1-2,c1-2,1),(r0+2,c0+3,r1,c1,2)]
        if rng.integers(2):
            boxes.reverse()
        clean = _paint((h,w),0,boxes)
    else:
        clean,boxes = scene
    arrows = []
    for i,direction in enumerate(directions):
        rank = q if i==0 else int(rng.integers(1,4 if not test else 5))
        profile = _profile(rng,anchor=anchor and i==0)
        arrow = _place_arrow(rng,clean,0,direction,rank,profile,arrows,desired if i==0 else None)
        if arrow is None:
            return None
        arrows.append(arrow)
    holes = set().union(*(o['holes'] for o in arrows))
    axes = set().union(*(set(_ray(clean.shape,o['tip'],o['direction'])) for o in arrows))
    off_axis = [tuple(p) for p in np.argwhere(clean!=0) if tuple(p) not in axes]
    rng.shuffle(off_axis)
    holes.update(off_axis[:int(rng.integers(1,4))])
    corrupted = clean.copy()
    for p in holes:
        corrupted[p] = 0
    wrongs = {o['wrong'] for o in arrows}-holes
    extra = [tuple(p) for p in np.argwhere(clean!=0) if tuple(p) not in holes|wrongs]
    rng.shuffle(extra)
    wrongs.update(extra[:int(rng.integers(0,3))])
    for p in wrongs:
        corrupted[p] = 3-int(clean[p])
    forbidden = set().union(*(o['frame']|o['cavity'] for o in arrows))
    exterior = [tuple(p) for p in np.argwhere(clean==0) if tuple(p) not in forbidden
                and 1<=p[0]<h-1 and 1<=p[1]<w-1
                and all(clean[z]==0 for z in _ring(clean.shape,tuple(p)))
                and all(max(abs(p[0]-z[0]),abs(p[1]-z[1]))>1 for z in forbidden)]
    rng.shuffle(exterior)
    for p in exterior[:int(rng.integers(1,4))]:
        corrupted[p] = int(rng.integers(1,3))
    # Local palette roles are visible, including each center color in its query.
    bg,fill1,fill2,frame_color = palette[:4]
    mapping = np.array((bg,fill1,fill2),dtype=np.int8)
    actual_clean,inp = mapping[clean],mapping[corrupted]
    for i,o in enumerate(arrows):
        o['frame_color'] = int(frame_color)
        o['payload_color'] = int(palette[4+i])
        cells = sorted(o['cavity'])
        if anchor and i==0 and q==2:
            # Adjacent payload cells distinguish cells from components.
            choices = [(p,z) for p in cells for z in cells if p<z and abs(p[0]-z[0])+abs(p[1]-z[1])==1]
            if not choices:
                return None
            payload = set(choices[int(rng.integers(len(choices)))])
        else:
            payload = {cells[int(j)] for j in rng.choice(len(cells),o['q'],replace=False)}
        o['payload'] = payload
        for p in o['frame']:
            inp[p] = frame_color
        for p in payload:
            inp[p] = o['payload_color']
    try:
        expected,targets = _render_answer(actual_clean,int(bg),(int(fill1),int(fill2)),arrows,holes)
        if desired is not None:
            stream = [p for p in _ray(clean.shape,arrows[0]['tip'],arrows[0]['direction']) if p in holes]
            if clean[stream[q-1]]!=desired:
                return None
        parsed = _analyze(inp)
        if not np.array_equal(parsed['clean'],actual_clean) or not np.array_equal(expected,parsed['output']):
            return None
        # Each query has a genuine extra erasure after its selected one and
        # a repaired wrong-fill error before it. Count all defects is wrong.
        for o in parsed['arrows']:
            ray = _ray(inp.shape,o['tip'],o['direction'])
            stream = [p for p in ray if p in parsed['holes']]
            if len(stream)<=o['q']:
                return None
            before = ray[:ray.index(stream[o['q']-1])]
            if not any(actual_clean[p]!=bg and inp[p]!=bg and inp[p] in (fill1,fill2) and inp[p]!=actual_clean[p] for p in before):
                return None
        assert len(targets)<len(holes)
        touched = set(targets)|set().union(*(_ring(inp.shape,p) for p in targets))
        if not any(p not in touched for p in holes):
            return None
        return {'input':inp,'output':expected}
    except AssertionError:
        return None


def _counts(num_train,num_test):
    for name,value,low,high in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low<=int(value)<=high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train<=num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train),int(num_test)


def generate_puzzle(seed: int,num_train: int=4,num_test: int=1) -> dict:
    num_train,num_test = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    anchor_direction = DIRECTIONS[int(rng.integers(4))]
    base_palette = tuple(map(int,rng.choice(10,7,replace=False)))
    puzzle,seen = {'train':[],'test':[]},set()
    taught_overlap = 0
    for split,n in (('train',num_train),('test',num_test)):
        for i in range(n):
            anchor = split=='train' and i<2
            q = 2+i if anchor else 4 if split=='test' else int(rng.integers(1,4))
            desired = 1+i if anchor else None
            direction = anchor_direction if anchor else DIRECTIONS[int(rng.integers(4))]
            palette = base_palette if anchor else tuple(map(int,rng.choice(10,7,replace=False)))
            for attempt in range(180):
                pair = _construct_pair(rng,direction,q,desired,palette,anchor,split=='test',attempt)
                if pair is not None and _key(pair['input']) not in seen:
                    level = _overlap_level(_analyze(pair['input']))
                    if split=='test' and level>taught_overlap:
                        continue
                    if split=='train':
                        taught_overlap = max(taught_overlap,level)
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    # Randomize demonstration order so there is no position-to-rank convention.
    order = rng.permutation(num_train)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _shape_signature(o):
    dr,dc = o['direction'];er,ec = -dc,dr
    r,c = o['tip']
    return tuple(sorted((-(p[0]-r)*dr-(p[1]-c)*dc,(p[0]-r)*er+(p[1]-c)*ec) for p in o['frame']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen,train_roles,matched = set(),set(),{}
    analyses = {'train':[],'test':[]}
    component_witness = False
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape==out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp);analyses[split].append(a)
            assert np.array_equal(a['output'],out), 'incorrect ordinal addressing or repair'
            assert not np.array_equal(inp,out)
            assert len(a['targets'])<len(a['holes']), 'mark-all-holes shortcut not contradicted'
            for o in a['arrows']:
                assert all(out[p]==a['bg'] for p in o['payload']), 'payload not consumed'
                assert all(out[p]==o['frame_color'] for p in o['frame']), 'frame changed'
                ray = _ray(inp.shape,o['tip'],o['direction'])
                stream = [p for p in ray if p in a['holes']]
                assert len(stream)>o['q']
                target = stream[o['q']-1]
                assert out[target]==o['payload_color']
                before = ray[:ray.index(target)]
                wrong = [p for p in before if a['clean'][p]!=a['bg'] and inp[p] in a['fills'] and inp[p]!=a['clean'][p]]
                assert wrong, 'missing wrong-fill-versus-erasure witness'
                # Every-deviation rank must select a different location.
                deviations = [p for p in ray if a['clean'][p]!=a['bg'] and inp[p]!=a['clean'][p]]
                assert deviations[o['q']-1]!=target
                if split=='train':
                    train_roles.add(int(a['clean'][target]))
                    signature = (_shape_signature(o),o['direction'],o['frame_color'],o['payload_color'])
                    matched.setdefault(signature,set()).add(o['q'])
                    component_witness |= len(_components(o['payload']))<o['q']
            # Repair an off-axis hole without a mark: selected centers and rings
            # are the only places where output may deviate from the clean scene.
            touched = set(a['targets'])|set().union(*(_ring(inp.shape,p) for p in a['targets']))
            assert any(p not in touched for p in a['holes']), 'no unselected erasure visibly restored'
    assert any({2,3}<=ranks for ranks in matched.values()), 'unmatched unary count demonstrations'
    assert component_witness, 'cell count indistinguishable from component count'
    assert len(train_roles)>=2, 'training does not expose complementary fills'
    assert max(o['q'] for a in analyses['train'] for o in a['arrows'])<=3
    assert all(any(o['q']==4 for o in a['arrows']) for a in analyses['test'])
    assert max(_overlap_level(a) for a in analyses['test']) <= max(_overlap_level(a) for a in analyses['train']), 'untaught overlap difficulty'
    return None


def self_test():
    """42 supplied seeds, streamed; no payloads or diagnostics saved to disk."""
    signatures,shapes,profiles,dirs,palettes,layouts,counts = set(),set(),set(),set(),set(),set(),set()
    overlap_witness = False
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                a = _analyze(pair['input'])
                for grid in pair.values():
                    digest.update(bytes(grid.shape));digest.update(grid.tobytes())
                shapes.add(pair['input'].shape)
                palettes.add((a['bg'],a['fills']))
                counts.add(len(a['arrows']))
                boxes = a['boxes']
                layouts.add(tuple(sorted((b[0],b[1],b[2],b[3]) for b in boxes)))
                for o in a['arrows']:
                    profiles.add(_shape_signature(o));dirs.add(o['direction'])
                targets = list(a['targets'])
                overlap_witness |= any(max(abs(p[0]-q[0]),abs(p[1]-q[1]))<=2 for i,p in enumerate(targets) for q in targets[:i])
                # Geometric equivariance checks the input-only role conventions.
                for k in (1,2,3):
                    assert np.array_equal(solve(np.rot90(pair['input'],k)),np.rot90(pair['output'],k))
                assert np.array_equal(solve((pair['input']+3)%10),(pair['output']+3)%10)
        signatures.add(digest.digest())
    assert len(signatures)==42 and len(shapes)>=25 and len(layouts)>=150
    assert dirs==set(DIRECTIONS) and len(profiles)>=12 and len(palettes)>=35
    assert {1,2}<=counts
    # Independent hand fixture for the simultaneous symbol compositor.
    clean = np.ones((7,9),dtype=np.int8);clean[:,6:]=2
    arrows = [dict(frame={(6,0)},tip=(3,0),direction=(0,1),q=2,frame_color=5,payload_color=7),
              dict(frame={(6,1)},tip=(4,0),direction=(0,1),q=1,frame_color=5,payload_color=7)]
    clean[6,:2] = 0
    holes = {(3,2),(3,4),(4,4),(3,7)}
    out,targets = _render_answer(clean,0,(1,2),arrows,holes)
    assert targets=={(3,4):7,(4,4):7}
    assert out[3,4]==out[4,4]==7 and out[2,3]==2 and out[3,2]==1
    assert np.array_equal(out,_render_answer(clean,0,(1,2),arrows[::-1],holes)[0])
    # A changed legal label must be rejected without consulting that label.
    p = generate_puzzle(0,2,1)
    again = generate_puzzle(0,2,1)
    assert all(np.array_equal(x[k],y[k]) for s in p for x,y in zip(p[s],again[s]) for k in x)
    p['test'][0]['output'][0,0] = (int(p['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('bad label accepted')
    for n,m in ((1,1),(11,1),(2,0),(6,5),(2,2),(3,4),(True,1),(4,False),(2.5,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,n,m)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    return {'seeds':42,'pairs':210,'distinct_puzzles':len(signatures),'shapes':len(shapes),
            'surface_layouts':len(layouts),'frame_profiles':len(profiles),
            'directions':len(dirs),'palette_roles':len(palettes),'query_counts':sorted(counts),
            'overlapping_symbols_observed':bool(overlap_witness)}
