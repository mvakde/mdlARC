"""Damage census: reconstructed-material object counts gate repair stamps.

Only standard-library modules and NumPy are used. No file I/O, stored examples,
seed lookup, or generation metadata occurs in solve().
"""
from itertools import permutations
import hashlib
import numpy as np


HOST_SEEDS = (0,175626960,801189766,453189326,394613923,168860369,599046182,
832471773,120520715,670331699,460036715,214490714,269174833,433150430,
902879348,777335842,923459918,570543136,321855700,623068936,617004218,
619111834,599422334,922126703,584901315,865676751,175117761,563891708,
21237356,448318847,640733447,374068658,675869407,842101120,181664164,
776679334,646710862,753457617,401861834,590227153,172784922,254177405)
STEPS8 = tuple((r,c) for r in (-1,0,1) for c in (-1,0,1) if r or c)
STEPS4 = ((-1,0),(1,0),(0,-1),(0,1))
X = ((0,0),(0,2),(1,1),(2,0),(2,2))
DIAMOND = ((0,1),(1,0),(1,2),(2,1))
DIAGONAL = ((0,0),(1,1),(2,2))
SQUARE = ((0,0),(0,1),(1,0),(1,1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'Bad dimensions'
    assert np.issubdtype(a.dtype, np.integer), 'Noninteger grid'
    assert np.all((a >= 0) & (a <= 9)), 'Bad colors'
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=True):
    """Deterministic flood fill; connectivity is a semantic parameter."""
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = STEPS8 if diagonal else STEPS4
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, cells = [p], []
        while stack:
            r,c = stack.pop()
            cells.append((int(r),int(c)))
            for dr,dc in steps:
                q = (r+dr,c+dc)
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _bbox(cells, color):
    rr,cc = zip(*cells)
    return (min(rr),min(cc),max(rr)+1,max(cc)+1,int(color))


def _paint(shape, bg, boxes):
    a = np.full(shape, bg, np.int8)
    for r0,c0,r1,c1,color in boxes:
        a[r0:r1,c0:c1] = color
    return a


def _overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))


def _geometry(boxes, shape):
    if len(boxes) not in (2,3):
        return False
    degree = [0]*len(boxes)
    for i,a in enumerate(boxes):
        r0,c0,r1,c1,_ = a
        if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
            return False
        if min(r1-r0,c1-c0) < 7:
            return False
        for j,b in enumerate(boxes[:i]):
            area = _overlap(a,b)
            if a[4] == b[4]:
                # Same-material rectangles are separate, even diagonally.
                if not (a[2]+1 <= b[0] or b[2]+1 <= a[0] or
                        a[3]+1 <= b[1] or b[3]+1 <= a[1]):
                    return False
            elif area:
                if area < 4:
                    return False
                degree[i] += 1
                degree[j] += 1
    return sorted(degree) == ([1,1] if len(boxes)==2 else [1,1,2])


def _supported(clean, observed, bg, boxes):
    if not _geometry(boxes,clean.shape):
        return False
    for r0,c0,r1,c1,color in boxes:
        visible = clean[r0:r1,c0:c1] == color
        correct = visible & (observed[r0:r1,c0:c1] == color)
        if visible.sum() < 32 or correct.sum() < .76*visible.sum():
            return False
        if min(correct[0].sum(),correct[-1].sum(),correct[:,0].sum(),correct[:,-1].sum()) < 3:
            return False
    return np.count_nonzero(clean==bg) >= .15*clean.size


def _fit(input_grid):
    """Infer visible rectangles from dense components, then compare orders.

    Family members preserve each visible rectangle's bounding extrema in its
    large observed four-connected core. Small components are sparse noise.
    This is an explicit evidence constraint, not a guess supplied by labels.
    """
    a = _grid(input_grid)
    colors,counts = np.unique(a,return_counts=True)
    assert len(colors)==4, 'Expect background, two materials, and an ink swatch'
    inks = [int(c) for c,n in zip(colors,counts) if n==1]
    assert len(inks)==1, 'Exactly one ink-color swatch'
    ink = inks[0]
    swatch = tuple(map(int,np.argwhere(a==ink)[0]))
    border = np.concatenate((a[0],a[-1],a[1:-1,0],a[1:-1,-1]))
    freq = np.bincount(border,minlength=10)
    bg = int(np.argmax(freq))
    assert freq[bg] > .55*len(border) and bg != ink, 'Ambiguous exterior'
    fills = tuple(int(c) for c in colors if c not in (bg,ink))
    boxes = tuple(_bbox(comp,color) for color in fills
                  for comp in _components(a==color,False) if len(comp)>=18)
    assert _geometry(boxes,a.shape), 'Unsupported material cores'
    candidates = {}
    for order in permutations(boxes):
        clean = _paint(a.shape,bg,order)
        if clean[swatch] != bg or not _supported(clean,a,bg,order):
            continue
        cost = int(np.count_nonzero((a!=clean)&(a!=ink)))
        key = clean.tobytes()
        if key not in candidates or tuple(order) < candidates[key][2]:
            candidates[key] = (cost,clean,tuple(order))
    assert candidates, 'No supported rectangle reconstruction'
    ranked = sorted(candidates.values(),key=lambda item:(item[0],item[2]))
    cost,clean,order = ranked[0]
    assert cost <= max(10,int(.13*a.size)), 'Excessive damage'
    assert all(item[0] >= cost+3 for item in ranked[1:]), 'Ambiguous covering order'
    # Also reject a nearly tied visible scene obtained by moving one side.
    for i,box in enumerate(order):
        for side in range(4):
            for delta in (-1,1):
                changed = list(box)
                changed[side] += delta
                trial = list(order)
                trial[i] = tuple(changed)
                if not _geometry(trial,a.shape):
                    continue
                alternative = _paint(a.shape,bg,trial)
                if np.array_equal(alternative,clean) or not _supported(alternative,a,bg,trial):
                    continue
                loss = int(np.count_nonzero((a!=alternative)&(a!=ink)))
                assert loss >= cost+3, 'Ambiguous boundary'
    # The swatch must be visibly separate from the material scene.
    r,c = swatch
    patch = clean[max(0,r-1):r+2,max(0,c-1):c+2]
    assert np.all(patch==bg), 'Swatch touches a material'
    return clean,bg,fills,ink,swatch,order


def _census(clean,holes,fills,diagonal=True):
    return {c:len(_components(holes & (clean==c),diagonal)) for c in fills}


def _winners(counts):
    return {c for c,n in counts.items() if n==max(counts.values()) and n>0}


def _render(clean,holes,counts,fills,ink,swatch,mark_all=False):
    chosen = set(fills) if mark_all else _winners(counts)
    selected = holes & np.isin(clean,list(chosen))
    output = clean.copy()
    demands = {}
    for r,c in np.argwhere(selected):
        other = fills[1] if clean[r,c]==fills[0] else fills[0]
        for dr,dc in STEPS8:
            q = (int(r+dr),int(c+dc))
            if not (0<=q[0]<clean.shape[0] and 0<=q[1]<clean.shape[1]) or selected[q]:
                continue
            assert q not in demands or demands[q]==other, 'Incompatible tied rings'
            demands[q] = other
    for q,color in demands.items():
        output[q] = color
    output[selected] = ink
    output[swatch] = ink
    return output


def _analyze(input_grid):
    a = _grid(input_grid)
    clean,bg,fills,ink,swatch,boxes = _fit(a)
    holes = (a==bg)&(clean!=bg)
    assert holes.any(), 'There must be damage to census'
    counts = _census(clean,holes,fills)
    expected = _render(clean,holes,counts,fills,ink,swatch)
    area = {c:int(np.count_nonzero(holes&(clean==c))) for c in fills}
    four = _census(clean,holes,fills,False)
    wrong = (a!=clean)&(a!=bg)&(a!=ink)&(clean!=bg)
    speckles = (a!=bg)&(a!=ink)&(clean==bg)
    return dict(clean=clean,background=bg,fills=fills,ink=ink,swatch=swatch,
                boxes=boxes,holes=holes,counts=counts,area=area,four=four,
                wrong=wrong,speckles=speckles,expected=expected)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input only, including its single ink-color key."""
    return _analyze(input_grid)['expected'].copy()


def _shape(rng,large=False):
    choices = (((0,0),),((0,0),(0,1)),((0,0),(1,1)),SQUARE,DIAMOND,X,
               ((0,0),(1,0),(1,1)))
    if not large:
        return choices[int(rng.integers(len(choices)))]
    cells = {(1,1)}
    target = int(rng.integers(6,9))
    while len(cells)<target:
        frontier = sorted({(r+dr,c+dc) for r,c in cells for dr,dc in STEPS8
                           if 0<=r+dr<4 and 0<=c+dc<4}-cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    r0,c0 = min(r for r,c in cells),min(c for r,c in cells)
    return tuple(sorted((r-r0,c-c0) for r,c in cells))


def _scene(rng,n):
    """Random unequal boxes; the fallback varies dimensions and offsets too."""
    h = int(rng.integers(23 if n==3 else 18,31))
    w = int(rng.integers(25 if n==3 else 18,31))
    for _ in range(160):
        boxes = []
        for i in range(n):
            rh = int(rng.integers(7,min(h-1,19)+1))
            rw = int(rng.integers(7,min(w-1,17)+1))
            r0 = int(rng.integers(h-rh+1))
            c0 = int(rng.integers(w-rw+1))
            boxes.append((r0,c0,r0+rh,c0+rw,1+i%2))
        if not _geometry(boxes,(h,w)):
            continue
        rng.shuffle(boxes)
        clean = _paint((h,w),0,boxes)
        border = np.concatenate((clean[0],clean[-1],clean[1:-1,0],clean[1:-1,-1]))
        if np.mean(border==0) <= .58 or not _supported(clean,clean,0,boxes):
            continue
        cores = sorted(_bbox(comp,color) for color in (1,2)
                       for comp in _components(clean==color,False) if len(comp)>=18)
        if cores==sorted(boxes):
            return clean,tuple(boxes)
    # General offset path: unequal boxes, free covering order, optional edge contact.
    a = (int(rng.integers(0,3)),0,h//2+int(rng.integers(1,4)),w//3+2,1)
    b = (h//3,w//3-2,h-1,2*w//3+1,2)
    boxes = [a,b]
    if n==3:
        boxes.append((int(rng.integers(0,3)),2*w//3-1,h//2+3,w,1))
    rng.shuffle(boxes)
    clean = _paint((h,w),0,boxes)
    assert _supported(clean,clean,0,boxes)
    return clean,tuple(boxes)


def _neighbors(points):
    return {(r+dr,c+dc) for r,c in points for dr,dc in STEPS8}


def _placements(clean,color,shape,used,tie,box=None,near=None):
    sh,sw = 1+max(r for r,c in shape),1+max(c for r,c in shape)
    same = {p for p in used if clean[p]==color}
    blocked = same | _neighbors(same)
    if tie:
        other = [p for p in used if clean[p]!=color]
        blocked |= {(r+dr,c+dc) for r,c in other
                    for dr in range(-2,3) for dc in range(-2,3)}
    result = []
    for r in range(clean.shape[0]-sh+1):
        for c in range(clean.shape[1]-sw+1):
            cells = tuple((r+dr,c+dc) for dr,dc in shape)
            if any(clean[p]!=color or p in blocked for p in cells):
                continue
            if box is not None and any(not (box[0]<=rr<box[2] and box[1]<=cc<box[3]) for rr,cc in cells):
                continue
            if near is not None and not any(p in near for p in cells):
                continue
            result.append(cells)
    return result


def _damage(rng,clean,boxes,kind,mode):
    # Local role 1 has two separated rectangles in the census witness.
    if kind=='census':
        planned = {1:2,2:1}
    elif kind=='tie':
        n = int(rng.integers(1,3))
        planned = {1:n,2:n}
    elif kind=='test' and mode==0:
        planned = {1:4,2:int(rng.integers(1,4))}
        if rng.integers(2):
            planned = {1:planned[2],2:planned[1]}
    elif kind=='test' and mode==1:
        planned = {1:2,2:1} if rng.integers(2) else {1:1,2:2}
    else:
        planned = {1:int(rng.integers(0,4)),2:int(rng.integers(1,4))}
    tie = planned[1]==planned[2]
    used = set()
    islands = []
    def add(color,shape,box=None,near=None):
        candidates = _placements(clean,color,shape,used,tie,box,near)
        if not candidates:
            return False
        cells = candidates[int(rng.integers(len(candidates)))]
        used.update(cells)
        islands.append((color,cells))
        return True
    if kind=='census':
        # A single large diagonal wound loses to two tiny wounds. One of the
        # tiny wounds touches it across the seam; the other is in the other box.
        glyph = (X,DIAMOND,DIAGONAL)[int(rng.integers(3))]
        candidates = _placements(clean,2,glyph,used,False)
        rng.shuffle(candidates)
        endpoints = [b for b in boxes if b[4]==1]
        for cells in candidates:
            near = _neighbors(cells)
            options = [(b,_placements(clean,1,((0,0),),set(cells),False,b,near)) for b in endpoints]
            options = [(b,opts) for b,opts in options if opts]
            if not options:
                continue
            used.update(cells)
            islands.append((2,cells))
            b,opts = options[int(rng.integers(len(options)))]
            p = opts[int(rng.integers(len(opts)))]
            used.update(p)
            islands.append((1,p))
            other = next(q for q in endpoints if q!=b)
            if not add(1,((0,0),),other):
                return None
            break
        else:
            return None
    else:
        for color in (1,2):
            for i in range(planned[color]):
                shape = _shape(rng,large=(kind=='test' and mode==1 and i==0))
                if kind=='tie' and i==0:
                    shape = ((0,0),(0,1)) if color==1 else SQUARE
                # Extrapolated counts include easy singleton objects too.
                if kind=='test' and mode==0 and i%2==0:
                    shape = ((0,0),)
                if not add(color,shape):
                    return None
    holes = np.zeros(clean.shape,bool)
    for p in used:
        holes[p] = True
    assert _census(clean,holes,(1,2))==planned
    a = clean.copy()
    a[holes] = 0
    # Wrong-fill damage is unrelated to the census, even at overlap seams.
    pool = [tuple(map(int,p)) for p in np.argwhere((clean!=0)&~holes)]
    rng.shuffle(pool)
    wrong = []
    wrong_target = int(rng.integers(1,4))
    for p in pool:
        if any(max(abs(p[0]-q[0]),abs(p[1]-q[1]))<=1 for q in wrong):
            continue
        a[p] = 3-int(clean[p])
        wrong.append(p)
        if len(wrong)>=wrong_target:
            break
    # Isolated exterior speckles never alter a dense material component's box.
    foreground = set(map(tuple,np.argwhere(clean!=0)))
    blocked = foreground | _neighbors(foreground)
    exterior = [tuple(map(int,p)) for p in np.argwhere(clean==0) if tuple(p) not in blocked]
    rng.shuffle(exterior)
    if not exterior:
        return None
    swatch = exterior.pop()
    a[swatch] = 3
    for p in exterior[:int(rng.integers(1,5))]:
        a[p] = int(rng.integers(1,3))
    output = _render(clean,holes,planned,(1,2),3,swatch)
    return a,output,planned


def _make_pair(rng,kind,mode,palette):
    for _ in range(90):
        n = 3 if kind=='census' else 2 if kind=='tie' else int(rng.integers(2,4))
        clean,boxes = _scene(rng,n)
        try:
            result = _damage(rng,clean,boxes,kind,mode)
            if result is None:
                continue
            a,out,planned = result
            mapping = np.array(palette,np.int8)
            a,out = mapping[a],mapping[out]
            if rng.integers(2):
                a,out = np.fliplr(a),np.fliplr(out)
            rotation = int(rng.integers(4))
            a,out = np.rot90(a,rotation).copy(),np.rot90(out,rotation).copy()
            meta = _analyze(a)
            actual = {palette[c]:n for c,n in planned.items()}
            if meta['counts']!=actual or not np.array_equal(meta['expected'],out):
                continue
            if kind=='census' and not (_winners(meta['counts'])!=_winners(meta['area']) and
                                      _winners(meta['counts'])!=_winners(meta['four'])):
                continue
            return {'input':a,'output':out}
        except AssertionError:
            continue
    raise RuntimeError('Bounded construction could not retain enough reconstruction evidence')


def _legal_counts(num_train,num_test):
    for name,value,lo,hi in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo<=int(value)<=hi:
            raise ValueError(name+' must be a legal integer count')
    if num_train<=num_test:
        raise ValueError('num_train must exceed num_test')


def _key(a):
    a = _grid(a)
    return (a.shape,a.tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _legal_counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(2))
    fixed = bool(rng.integers(2))
    base_palette = list(map(int,rng.choice(10,4,replace=False)))
    puzzle,seen = {'train':[],'test':[]},set()
    for split,n in (('train',num_train),('test',num_test)):
        for index in range(n):
            kind = ('census' if index==0 else 'tie' if index==1 else 'mix') if split=='train' else 'test'
            palette = base_palette if fixed else list(map(int,rng.choice(10,4,replace=False)))
            for _ in range(5):
                pair = _make_pair(rng,kind,mode,palette)
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Distinct-input construction exhausted')
    rng.shuffle(puzzle['train'])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    _legal_counts(len(puzzle['train']),len(puzzle['test']))
    seen,training = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            a,out = _grid(pair['input']),_grid(pair['output'])
            signature = _key(a)
            assert signature not in seen, 'Duplicate input'
            seen.add(signature)
            m = _analyze(a)
            assert out.shape==a.shape and np.array_equal(out,m['expected']), 'Incorrect label'
            assert not np.array_equal(a,out)
            selected = m['holes'] & np.isin(m['clean'],list(_winners(m['counts'])))
            ink_mask = out==m['ink']
            ink_mask[m['swatch']] = False
            assert np.array_equal(ink_mask,selected), 'Only winning holes are centers'
            assert out[m['swatch']]==m['ink']
            assert not np.any(m['wrong']&ink_mask), 'Wrong-fill noise became a center'
            if split=='train':
                training.append(m)
    assert any(_winners(m['counts'])!=_winners(m['area']) and
               _winners(m['counts'])!=_winners(m['four']) for m in training), 'Missing counting witness'
    assert any(len(_winners(m['counts']))==2 for m in training), 'Missing tie demonstration'
    assert any(len(m['boxes'])==3 and
               all(np.any(m['holes'][r0:r1,c0:c1] & (m['clean'][r0:r1,c0:c1]==color))
                   for r0,c0,r1,c1,color in m['boxes']) for m in training), 'Missing material pooling witness'
    assert any(m['wrong'].any() and m['speckles'].any() for m in training)
    assert any(any(0<=r+dr<m['clean'].shape[0] and 0<=c+dc<m['clean'].shape[1] and
                       m['holes'][r+dr,c+dc] and m['clean'][r+dr,c+dc]!=m['clean'][r,c]
                       for r,c in np.argwhere(m['holes']) for dr,dc in STEPS8)
               for m in training), 'Missing seam-touching category witness'
    assert any(any(0<=r+dr<m['clean'].shape[0] and 0<=c+dc<m['clean'].shape[1] and
                       m['holes'][r+dr,c+dc] and m['clean'][r+dr,c+dc]==m['clean'][r,c]
                       for r,c in np.argwhere(m['holes']) for dr,dc in STEPS8)
               and len(_winners(m['counts']))==2 for m in training), 'Missing adjacent tied centers'


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train','test'):
        for pair in puzzle[split]:
            for role in ('input','output'):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, streaming measurements, bounded memory, no saved grids/logs."""
    shapes,layouts,palettes,signatures,counts,areas,structures = (set() for _ in range(7))
    clipped = repeated = zero = tie = seam = wrong_at_seam = adjacent = 0
    pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_digest(p))
        assert _digest(generate_puzzle(seed))==_digest(p), 'Nondeterminism'
        for split in ('train','test'):
            for pair in p[split]:
                a,out = pair['input'],pair['output']
                m = _analyze(a)
                pairs += 1
                shapes.add(a.shape)
                layouts.add(tuple(b[:4] for b in m['boxes']))
                palettes.add((m['background'],*m['fills'],m['ink']))
                counts.add(tuple(sorted(m['counts'].values())))
                areas.update(len(c) for color in m['fills'] for c in _components(m['holes']&(m['clean']==color)))
                repeated += len(m['boxes'])==3
                zero += 0 in m['counts'].values()
                tie += len(_winners(m['counts']))==2
                chosen = m['holes'] & np.isin(m['clean'],list(_winners(m['counts'])))
                clipped += any(r in (0,a.shape[0]-1) or c in (0,a.shape[1]-1) for r,c in np.argwhere(chosen))
                for r,c in np.argwhere(m['holes']):
                    seam += any(0<=r+dr<a.shape[0] and 0<=c+dc<a.shape[1] and
                                m['holes'][r+dr,c+dc] and m['clean'][r+dr,c+dc]!=m['clean'][r,c]
                                for dr,dc in STEPS8)
                    adjacent += any(0<=r+dr<a.shape[0] and 0<=c+dc<a.shape[1] and
                                    chosen[r,c] and chosen[r+dr,c+dc] for dr,dc in STEPS8)
                for r,c in np.argwhere(m['wrong']):
                    wrong_at_seam += any(0<=r+dr<a.shape[0] and 0<=c+dc<a.shape[1] and
                                         m['clean'][r+dr,c+dc] in m['fills'] and
                                         m['clean'][r+dr,c+dc]!=m['clean'][r,c] for dr,dc in STEPS8)
                # Color-normalized scene (damage removed), not just input noise.
                normalized = np.zeros(a.shape,np.int8)
                for i,color in enumerate(m['fills'],1):
                    normalized[m['clean']==color] = i
                structures.add((a.shape,normalized.tobytes()))
                # A separate vectorized simultaneous stamp check.
                expected = m['clean'].copy()
                for color in m['fills']:
                    mask = chosen & (m['clean']==color)
                    ring = np.zeros(a.shape,bool)
                    for dr,dc in STEPS8:
                        rs = slice(max(0,dr),min(a.shape[0],a.shape[0]+dr))
                        cs = slice(max(0,dc),min(a.shape[1],a.shape[1]+dc))
                        sr = slice(max(0,-dr),min(a.shape[0],a.shape[0]-dr))
                        sc = slice(max(0,-dc),min(a.shape[1],a.shape[1]-dc))
                        ring[rs,cs] |= mask[sr,sc]
                    expected[ring] = next(c for c in m['fills'] if c!=color)
                expected[chosen] = m['ink']
                expected[m['swatch']] = m['ink']
                assert np.array_equal(expected,out)
                assert np.array_equal(solve(a.copy()),out)
                if seed==0:
                    # Geometry and literal color labels carry no extra convention.
                    assert np.array_equal(solve(np.rot90(a)),np.rot90(out))
                    permutation = np.array([9,0,8,1,7,2,6,3,5,4],np.int8)
                    assert np.array_equal(solve(permutation[a]),permutation[out])
    assert len(signatures)==42 and len(shapes)>=40 and len(layouts)>=120
    assert len(structures)>=150 and len(palettes)>=35 and len(counts)>=8
    assert {1,2,3,4,5} <= areas and max(areas)>=7
    assert min(clipped,repeated,zero,tie,seam,wrong_at_seam,adjacent)>0
    # Every legal request; illegal types and ranges, including None, are rejected.
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            q = generate_puzzle(42+nt*10+nq,nt,nq)
            assert len(q['train'])==nt and len(q['test'])==nq
            validate_generated_puzzle(q)
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.0,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = generate_puzzle(0,2,1)
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Incorrect label accepted')
    return dict(seeds=42,pairs=pairs,distinct_puzzles=len(signatures),grid_shapes=len(shapes),
                clean_layouts=len(layouts),clean_structures=len(structures),palettes=len(palettes),
                count_combinations=len(counts),island_areas=sorted(areas),clipped_pairs=int(clipped),
                repeated_material_pairs=int(repeated),zero_count_pairs=int(zero),tie_pairs=int(tie),
                seam_touching_hole_cells=int(seam),seam_noise_cells=int(wrong_at_seam),
                adjacent_selected_hole_cells=int(adjacent))
