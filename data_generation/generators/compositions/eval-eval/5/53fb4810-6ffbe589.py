"""Radial looms: color-keyed rotation routes phase-preserving periodic ribbons.

Only NumPy and the standard library are needed. No files, examples, seed state,
or generation records are read by solve(). All saved diagnostics are opt-in.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import numpy as np

HOST_SEEDS = (0,691811509,847047162,308720356,854316628,12394974,653499400,26671440,159694808,172389776,404378636,28423142,919002132,980405187,300443668,115057699,81627632,635690921,546503172,182632706,194077905,631008607,534642722,930100920,653303092,242633951,89088497,268883248,10167843,858227372,921746342,607114169,593278985,159883930,762317008,346293178,777581619,297175856,856373083,816654356,701477833,982188216)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid shape"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "color range"
    return a


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _halo(cells):
    return {(r+dr,c+dc) for r,c in cells for dr in (-1,0,1) for dc in (-1,0,1)}


def _hub_mask(n, kind, arm):
    m = np.ones((n,n), dtype=bool)
    if kind == "clipped":
        m[0,0] = m[0,-1] = m[-1,0] = m[-1,-1] = False
    elif kind == "cross":
        m[:] = False
        lo = (n-arm)//2
        m[lo:lo+arm,:] = True
        m[:,lo:lo+arm] = True
    return m


def _hub_kind(mask):
    n, w = mask.shape
    if n != w or not 3 <= n <= 8:
        return None
    if np.all(mask):
        return "solid"
    if np.array_equal(mask, _hub_mask(n,"clipped",1)):
        return "clipped"
    for arm in range(1,n-1):
        if (n-arm)%2 == 0 and np.array_equal(mask, _hub_mask(n,"cross",arm)):
            return "cross"
    return None


def _coord(box, direction, width, d, j):
    """Clockwise-rotate an east-facing port coordinate about its local hub."""
    t, l, n = box
    r, c = (n-width)//2+j, n-1+d
    for _ in range(direction%4):
        r, c = c, n-1-r
    return t+r, l+c


def _distance(shape, box, direction):
    t,l,n = box
    return (shape[1]-l-n, shape[0]-t-n, l, t)[direction%4]


def _period(word, whole=True):
    return next(p for p in range(1,len(word)+1)
                if (not whole or len(word)%p == 0)
                and np.array_equal(word, word[np.arange(len(word))%p]))


def _parse_candidate(a, bg, hc):
    hubs, claimed, assemblies = [], set(), []
    for cells in _components(a == hc):
        rr, cc = zip(*cells)
        t,l = min(rr),min(cc)
        h,w = max(rr)-t+1,max(cc)-l+1
        mask = a[t:t+h,l:l+w] == hc
        kind = _hub_kind(mask)
        assert kind is not None, "hub geometry"
        box = (t,l,h)
        choices = []
        for direction in range(4):
            if _distance(a.shape,box,direction) < 1:
                continue
            face = [_coord(box,direction,h,1,j) for j in range(h)]
            payload = [j for j,p in enumerate(face) if int(a[p]) not in (bg,hc)]
            if not payload:
                continue
            width = len(payload)
            lo = (h-width)//2
            if not (1 <= width <= 4 and width <= h-2 and (h-width)%2 == 0
                    and payload == list(range(lo,lo+width))):
                continue
            # Every cell of the ribbon touches a real hub cell, including cross hubs.
            if not all(_coord(box,direction,width,0,j) in cells for j in range(width)):
                continue
            choices.append((direction,width))
        assert len(choices) == 1, "one centered ribbon per hub"
        direction,width = choices[0]
        word, tape = [], set()
        for d in range(1,_distance(a.shape,box,direction)+1):
            points = [_coord(box,direction,width,d,j) for j in range(width)]
            values = [int(a[p]) for p in points]
            if all(v == bg for v in values):
                break
            assert all(v not in (bg,hc) for v in values), "partial tape slice"
            word.append(values)
            tape.update(points)
        assert word and len(word) <= 10, "whole visible tape"
        assembly = cells | tape
        assert not claimed.intersection(assembly)
        assert all(not (_halo(other) & assembly) for other in assemblies), "separate assemblies"
        assemblies.append(assembly)
        claimed.update(assembly)
        hubs.append(dict(box=box,mask=mask.copy(),cells=cells,direction=direction,
                         width=width,word=np.array(word,dtype=np.int8),kind=kind))
    assert 1 <= len(hubs) <= 4
    palette = set(int(v) for s in hubs for v in s['word'].ravel())
    residual = set(map(tuple,np.argwhere(a != bg).tolist())) - claimed
    counts, markers = {}, {}
    for color in sorted(palette):
        cells = {p for p in residual if int(a[p]) == color}
        if not cells:
            continue
        rr,cc = zip(*cells)
        horizontal = len(set(rr)) == 1 and max(cc)-min(cc)+1 == len(cells)
        vertical = len(set(cc)) == 1 and max(rr)-min(rr)+1 == len(cells)
        assert 1 <= len(cells) <= 9 and (horizontal or vertical), "one unary bar per color"
        assert not (_halo(cells) & claimed), "detached counter"
        assert all(not (_halo(cells) & m['cells']) for m in markers.values()), "separate counters"
        counts[color] = len(cells)
        markers[color] = dict(cells=cells,orientation='single' if len(cells)==1 else 'horizontal' if horizontal else 'vertical')
    assert markers and residual == set().union(*(m['cells'] for m in markers.values())), "unmatched material"
    return dict(background=bg,hub_color=hc,hubs=hubs,counts=counts,markers=markers,palette=palette)


def _parse(input_grid):
    a = _grid(input_grid)
    colors, masses = np.unique(a,return_counts=True)
    assert np.sum(masses == masses.max()) == 1, "unique dominant background"
    bg = int(colors[masses.argmax()])
    candidates = []
    for hc in colors:
        if hc == bg:
            continue
        try:
            candidates.append(_parse_candidate(a,bg,int(hc)))
        except AssertionError:
            pass
    assert len(candidates) == 1, "unique input-only loom interpretation"
    return candidates[0]


def _writes(shape, hubs, counts, *, direction_sign=1, extend=True, prefix=False, source_clip=False):
    hub_cells = set().union(*(s['cells'] for s in hubs))
    writes = {}
    for s in hubs:
        word = s['word']
        if prefix:
            word = word[:_period(word,whole=False)]
        for color in sorted(set(map(int,word.ravel()))):
            direction = (s['direction']+direction_sign*counts.get(color,0))%4
            limit = _distance(shape,s['box'],direction)
            if source_clip:
                limit = min(limit,_distance(shape,s['box'],s['direction']))
            if not extend:
                limit = min(limit,len(s['word']))
            for d in range(1,limit+1):
                row = word[(d-1)%len(word)]
                for j,value in enumerate(row):
                    if int(value) != color:
                        continue
                    p = _coord(s['box'],direction,s['width'],d,j)
                    assert p not in hub_cells, "emission strikes another hub"
                    assert p not in writes or writes[p] == color, "different colors collide"
                    writes[p] = color
    return writes


def _predict(shape, info, **kwargs):
    out = np.full(shape,info['background'],dtype=np.int8)
    for s in info['hubs']:
        for p in s['cells']:
            out[p] = info['hub_color']
    for p,color in _writes(shape,info['hubs'],info['counts'],**kwargs).items():
        out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Route each color's periodic slots from the visible local hub and counter."""
    a = _grid(input_grid)
    return _predict(a.shape,_parse(a))


def _build_output(shape, specs, bg, hc, counts):
    """Construction uses array rotations on padded local masks, then canvas clipping.

    The solver instead computes each coordinate and its actual directional limit.
    """
    out = np.full(shape,bg,dtype=np.int8)
    occupied = np.zeros(shape,dtype=bool)
    for s in specs:
        for p in s['cells']:
            out[p] = hc
            occupied[p] = True
    radius = max(shape)
    for s in specs:
        t,l,n = s['box']
        side = n+2*radius
        local = np.full((side,side),bg,dtype=np.int8)
        local_box = (radius,radius,n)
        for d in range(1,radius+1):
            for j,value in enumerate(s['word'][(d-1)%len(s['word'])]):
                local[_coord(local_box,s['direction'],s['width'],d,j)] = value
        for color in np.unique(s['word']):
            color = int(color)
            rotated = np.rot90(local == color,-counts.get(color,0)%4)
            mask = rotated[radius-t:radius-t+shape[0],radius-l:radius-l+shape[1]]
            assert mask.shape == shape
            assert not np.any(mask & occupied & (out != color))
            out[mask] = color
            occupied |= mask
    return out


def _word(rng, width, length, colors, aba=False):
    assert width*length >= len(colors)
    word = rng.choice(colors,size=(length,width)).astype(np.int8)
    if aba:
        assert width == 2 and length == 3 and len(colors) == 4
        values = np.asarray(rng.permutation(colors),dtype=np.int8)
        word[:2] = values.reshape(2,2)
        word[2] = word[0]
    else:
        slots = rng.choice(width*length,len(colors),replace=False)
        word.ravel()[slots] = rng.permutation(colors)
    return word


def _spec(rng, shape, width, word, *, box=None, direction=None, minimal=False):
    n = width+2 if minimal else int(rng.choice([width+2,width+4]))
    if box is not None:
        n = box[2]
    if direction is None:
        direction = int(rng.integers(4))
    if box is None:
        t = int(rng.integers(0,shape[0]-n+1))
        l = int(rng.integers(0,shape[1]-n+1))
        box = (t,l,n)
    if _distance(shape,box,direction) < len(word):
        return None
    kind = str(rng.choice(['solid','clipped','cross']))
    arms = list(range(width,n-1,2))
    mask = _hub_mask(n,kind,int(rng.choice(arms)))
    t,l,_ = box
    cells = {(t+int(r),l+int(c)) for r,c in np.argwhere(mask)}
    tape = {_coord(box,direction,width,d,j) for d in range(1,len(word)+1) for j in range(width)}
    return dict(box=box,mask=mask,cells=cells,direction=direction,width=width,
                word=word.copy(),kind=kind,assembly=cells|tape)


def _pack(rng, shape, recipes, counts, teaching=False):
    specs, assembly = [], set()
    for width,word in recipes:
        for _ in range(100):
            s = _spec(rng,shape,width,word)
            if s is None or _halo(s['assembly']) & assembly:
                continue
            if teaching and min(_distance(shape,s['box'],d) for d in range(4)) < len(word)+2:
                continue
            try:
                _writes(shape,specs+[s],counts)
            except AssertionError:
                continue
            specs.append(s)
            assembly.update(s['assembly'])
            break
        else:
            return None
    return specs


def _put_counters(rng, inp, specs, bg, counts, forced):
    blocked = _halo(set().union(*(s['assembly'] for s in specs)))
    h,w = inp.shape
    for color,length in sorted(counts.items(),key=lambda item:-item[1]):
        if not length:
            continue
        choices = []
        for orient in ('horizontal','vertical'):
            if color in forced and orient != forced[color]:
                continue
            dh,dw = (1,length) if orient == 'horizontal' else (length,1)
            starts = []
            for r in range(h-dh+1):
                for c in range(w-dw+1):
                    cells = {(r+i if dh>1 else r,c+i if dw>1 else c) for i in range(length)}
                    if not (cells & blocked):
                        starts.append((r,c))
            if starts:
                choices.append((orient,starts))
        if not choices:
            return False
        orient,starts = choices[int(rng.integers(len(choices)))]
        r,c = starts[int(rng.integers(len(starts)))]
        cells = {(r+i if orient=='vertical' else r,c+i if orient=='horizontal' else c) for i in range(length)}
        for p in cells:
            inp[p] = color
        blocked.update(_halo(cells))
    return True


def _materialize(rng, shape, specs, bg, hc, counts, forced, teach_clip=False):
    inp = np.full(shape,bg,dtype=np.int8)
    for s in specs:
        for p in s['cells']:
            inp[p] = hc
        for d,row in enumerate(s['word'],1):
            for j,color in enumerate(row):
                inp[_coord(s['box'],s['direction'],s['width'],d,j)] = color
    if not _put_counters(rng,inp,specs,bg,counts,forced):
        return None
    masses = np.bincount(inp.ravel(),minlength=10)
    if masses[bg] <= max(masses[c] for c in range(10) if c != bg):
        return None
    out = _build_output(shape,specs,bg,hc,counts)
    try:
        info = _parse(inp)
        assert np.array_equal(_predict(shape,info),out)
        # Both inherited mechanisms must affect actual cells, even in small cases.
        assert not np.array_equal(_predict(shape,{**info,'counts':{}}),out)
        assert not np.array_equal(_predict(shape,info,extend=False),out)
        if teach_clip:
            assert not np.array_equal(_predict(shape,info,source_clip=True),out)
    except AssertionError:
        return None
    return {'input':inp,'output':out}


def _fallback(rng, recipes, colors, bg, hc, counts, forced, witness, long_counts=True):
    """Bounded escape geometry; still the same visible family and transformation."""
    shape = (30,30)
    if witness == 0:
        # The missing relative north channel faces inward; all emission is outward.
        specs = [_spec(rng,shape,width,word,box=(12,3 if i==0 else 23,4),
                       direction=1 if i==0 else 3,minimal=True)
                 for i,(width,word) in enumerate(recipes)]
    elif witness == 1:
        width,word = recipes[0]
        specs = [_spec(rng,shape,width,word,box=(12,19,4),direction=0,minimal=True)]
    else:
        # Parallel vertical emissions have disjoint columns. Even counters route
        # a north-facing sample north/south, independently of their bar direction.
        counts = {c:int(rng.choice([0,2,4,6,8] if long_counts else [0,2,4])) for c in colors}
        counts[colors[0]] = 6 if long_counts else 2
        specs = []
        for i,(width,word) in enumerate(recipes):
            n = width+2
            specs.append(_spec(rng,shape,width,word,box=(12,1+7*i,n),direction=3,minimal=True))
        forced = {}
    _writes(shape,specs,counts)
    pair = _materialize(rng,shape,specs,bg,hc,counts,forced,teach_clip=witness==1)
    assert pair is not None, 'deterministic escape geometry'
    return pair


def _generate_pair(rng, mode, split, index):
    witness = index if split == 'train' and index < 2 else -1
    bg,hc,*colors = map(int,rng.permutation(10))
    p = 4 if witness >= 0 else int(rng.integers(2,5))
    colors = colors[:p]
    forced = {}
    if witness == 0:
        counts = dict(zip(colors,[1,2,2,0]))
        forced = {colors[1]:'horizontal',colors[2]:'vertical'}
        count = 2
        recipes = [(2,_word(rng,2,int(rng.integers(2,4)),colors)) for _ in range(count)]
    elif witness == 1:
        counts = dict(zip(colors,[3,4,5,0]))
        recipes = [(2,_word(rng,2,3,colors,aba=True))]
    else:
        count = int(rng.integers(1,3 if split=='train' else 4))
        if split == 'test' and mode == 2:
            count = int(rng.integers(3,5))
        counts = {c:int(rng.integers(0,6 if split=='train' else 10)) for c in colors}
        if split == 'test':
            counts[colors[0]] = int(rng.integers(6,10))
        elif not any(v%4 for v in counts.values()):
            counts[colors[0]] = 1
        recipes = []
        for _ in range(count):
            width = int(rng.integers(1,3 if mode==0 and split=='train' else 4))
            if split == 'test' and mode == 0:
                width = int(rng.integers(3,5))
            if split == 'train' and mode == 1 and p > width*3:
                width = 2
            low = (p+width-1)//width
            length = int(rng.integers(low,max(low,3)+1))
            if split == 'test' and mode == 1:
                length = int(rng.integers(max(low,4),6))
            word = _word(rng,width,length,colors)
            if split == 'test' and mode == 1 and _period(word) < length:
                word[-1,-1] = next(c for c in colors if c != int(word[-1,-1]))
                assert _period(word) == length
            recipes.append((width,word))
    counts = {c:k for c,k in counts.items() if k}
    for _ in range(12):
        low = 22 if len(recipes)>2 else 19 if len(recipes)==2 else 13
        shape = (int(rng.integers(low,31)),int(rng.integers(low,31)))
        specs = _pack(rng,shape,recipes,counts,teaching=witness>=0)
        if specs is None:
            continue
        pair = _materialize(rng,shape,specs,bg,hc,counts,forced,teach_clip=witness==1)
        if pair is not None:
            return pair
    return _fallback(rng,recipes,colors,bg,hc,counts,forced,witness,long_counts=split=='test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer from 2 through 10')
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError('num_test must be an integer from 1 through 4, below num_train')
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(4))
    puzzle,used = {'train':[],'test':[]},set()
    for split,n in (('train',num_train),('test',num_test)):
        for i in range(n):
            pair = _generate_pair(rng,mode,split,i)
            if _key(pair['input']) in used:
                # At most 13 excluded inputs versus 10! injective recolorings.
                present = sorted(map(int,np.unique(pair['input'])))
                for values in itertools.permutations(range(10),len(present)):
                    mapping = np.arange(10,dtype=np.int8)
                    mapping[present] = values
                    candidate = {k:mapping[v] for k,v in pair.items()}
                    if _key(candidate['input']) not in used:
                        pair = candidate
                        break
                else:
                    raise AssertionError('recoloring space exhausted')
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _different_or_invalid(shape, info, out, **kwargs):
    try:
        return not np.array_equal(_predict(shape,info,**kwargs),out)
    except AssertionError:
        return True


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    a,b = len(puzzle['train']),len(puzzle['test'])
    assert 2<=a<=10 and 1<=b<=4 and a>b
    seen,train = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape==out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(out,_predict(inp.shape,info)), 'wrong routing, phase or clipping'
            for s in info['hubs']:
                assert all(out[p]==info['hub_color'] for p in s['cells'])
            assert _different_or_invalid(inp.shape,{**info,'counts':{}},out), 'counter mechanism inactive'
            assert _different_or_invalid(inp.shape,info,out,extend=False), 'periodic growth inactive'
            if split=='train':
                train.append((inp,out,info))
    lengths = {k for _,_,i in train for k in i['counts'].values()}
    assert {1,2,3,4,5} <= lengths, 'show counting and wraparound'
    assert any(i['palette']-i['counts'].keys() for _,_,i in train), 'show absent-counter default'
    assert any(_different_or_invalid(x.shape,i,y,direction_sign=-1) for x,y,i in train), 'clockwise witness'
    assert any(_different_or_invalid(x.shape,i,y,prefix=True) for x,y,i in train), 'whole-word witness'
    assert any(_different_or_invalid(x.shape,i,y,source_clip=True) for x,y,i in train), 'rotate before clipping'
    assert any(any(len({i['markers'][c]['orientation'] for c,k in i['counts'].items() if k==raw})>=2
                       for raw in set(i['counts'].values())) for _,_,i in train), 'count not orientation'
    assert any(any(sum(c in set(map(int,s['word'].ravel())) for s in i['hubs'])>=2 for c in i['palette'])
               for _,_,i in train), 'shared counters across hubs'


def measure_pair(pair):
    inp = _grid(pair['input'])
    info = _parse(inp)
    return dict(shape=list(inp.shape),background=info['background'],hub_color=info['hub_color'],
                hubs=len(info['hubs']),widths=sorted({s['width'] for s in info['hubs']}),
                periods=sorted({_period(s['word']) for s in info['hubs']}),
                sizes=sorted({s['box'][2] for s in info['hubs']}),
                hub_shapes=sorted({s['kind'] for s in info['hubs']}),
                source_directions=sorted({s['direction'] for s in info['hubs']}),
                counts=sorted(info['counts'].values()),unmarked=len(info['palette']-info['counts'].keys()),
                orientations=sorted({m['orientation'] for m in info['markers'].values()}),
                changed=int(np.count_nonzero(np.asarray(pair['output'])!=inp)))


def self_test():
    """Streaming 42-seed audit; no grid corpus, logs or traces written to disk."""
    # Hand-specified ABA regression: a's holes survive a turn, b goes west,
    # and the original sample and both detached counters disappear.
    inp = np.zeros((12,19),dtype=np.int8)
    expected = np.zeros_like(inp)
    cross = [(4,4),(5,3),(5,4),(5,5),(6,4)]
    for p in cross:
        inp[p] = expected[p] = 1
    inp[0,0] = 2
    inp[0,12:14] = 3
    inp[5,6:9] = [2,3,2]
    expected[[7,9,10],4] = 2
    expected[5,1] = 3
    assert np.array_equal(solve(inp),expected)
    signatures,shapes,widths,periods,hubs,kinds,backgrounds,counts,directions = (set() for _ in range(9))
    records = 0
    regimes = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        local = {'train':[],'test':[]}
        for split in ('train','test'):
            for pair in puzzle[split]:
                f = measure_pair(pair)
                local[split].append(f)
                digest.update(repr(_key(pair['input'])).encode())
                shapes.add(tuple(f['shape']))
                widths.update(f['widths'])
                periods.update(f['periods'])
                hubs.add(f['hubs'])
                kinds.update(f['hub_shapes'])
                backgrounds.add(f['background'])
                counts.update(f['counts'])
                directions.update(f['source_directions'])
                records += 1
        assert max(k for f in local['train'] for k in f['counts']) <= 5
        assert all(max(f['counts']) >= 6 for f in local['test'])
        for name,field in (('wider','widths'),('longer','periods')):
            if max(k for f in local['test'] for k in f[field]) > max(k for f in local['train'] for k in f[field]):
                regimes.add(name)
        if max(f['hubs'] for f in local['test']) > max(f['hubs'] for f in local['train']):
            regimes.add('more_hubs')
        signatures.add(digest.digest())
    assert len(signatures)==42 and records==210
    assert len(shapes)>=40
    assert widths=={1,2,3,4} and periods>={1,2,3,4,5}
    assert hubs=={1,2,3,4} and kinds=={'solid','clipped','cross'}
    assert backgrounds==set(range(10)) and counts==set(range(1,10)) and directions==set(range(4))
    assert regimes == {'wider','longer','more_hubs'}
    a = generate_puzzle(90210,2,1)
    b = generate_puzzle(90210,2,1)
    assert all(np.array_equal(p[k],q[k]) for s in a for p,q in zip(a[s],b[s]) for k in p)
    for x,y in ((1,1),(11,1),(4,0),(6,5),(2,2),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,x,y)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Exercise the escape layouts explicitly, including broad four-hub ribbons.
    rng = np.random.default_rng(314)
    colors = [2,3,4,5]
    for witness in (0,1,-1):
        recipes = ([(2,_word(rng,2,3,colors)) for _ in range(2)] if witness==0 else
                   [(2,_word(rng,2,3,colors,aba=True))] if witness==1 else
                   [(4,_word(rng,4,5,colors)) for _ in range(4)])
        cs = dict(zip(colors,[1,2,2,0] if witness==0 else [3,4,5,0]))
        pair = _fallback(rng,recipes,colors,0,1,cs,{3:'horizontal',4:'vertical'} if witness==0 else {},witness)
        assert np.array_equal(solve(pair['input']),pair['output'])
    bad = {s:[{k:np.asarray(v).copy() for k,v in p.items()} for p in a[s]] for s in a}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    return dict(seeds=42,pairs=records,input_shapes=len(shapes),widths=sorted(widths),
                periods=sorted(periods),hub_counts=sorted(hubs),hub_shapes=sorted(kinds),
                backgrounds=sorted(backgrounds),counter_lengths=sorted(counts),directions=sorted(directions))


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
