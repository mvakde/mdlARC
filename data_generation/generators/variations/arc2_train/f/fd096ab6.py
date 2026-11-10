"""ARC translated-stencil completion generator."""
import numpy as np


def _counts(seed, train, test):
    for value, low, high in ((train, 2, 10), (test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError('invalid pair count')
    if train is None and test is None:
        test = 1 + seed % 4
        choices = list(range(max(2, test + 1), 11))
        train = choices[(seed + seed // 4) % len(choices)]
    elif train is None:
        choices = list(range(max(2, test + 1), 11))
        train = choices[(seed + 3) % len(choices)]
    elif test is None:
        choices = list(range(1, min(4, train - 1) + 1))
        test = choices[(seed + 5) % len(choices)]
    if train <= test:
        raise ValueError('train count must exceed test count')
    return int(train), int(test)


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _connected(cells):
    remaining = set(cells)
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        adjacent = {q for q in remaining if max(abs(q[0]-r), abs(q[1]-c)) <= 1}
        stack.extend(adjacent)
        remaining -= adjacent
    return not remaining


def _has_hole(shape):
    h,w = _box(shape)[2]+1,_box(shape)[3]+1
    empty = {(r,c) for r in range(h) for c in range(w)}-set(shape)
    outside = {(r,c) for r,c in empty if r in (0,h-1) or c in (0,w-1)}
    stack = list(outside)
    while stack:
        r,c = stack.pop()
        added = ({(r+1,c),(r-1,c),(r,c+1),(r,c-1)} & empty)-outside
        outside |= added
        stack.extend(added)
    return bool(empty-outside)


def _stencil(rng, used, connected):
    for _ in range(500):
        h, w = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        if h*w < 8:
            continue
        size = int(rng.integers(6, min(13, h*w-1)+1))
        if connected:
            cells = {(int(rng.integers(h)), int(rng.integers(w)))}
            while len(cells) < size:
                frontier = {(r+dr,c+dc) for r,c in cells for dr in (-1,0,1) for dc in (-1,0,1)
                            if 0 <= r+dr < h and 0 <= c+dc < w} - cells
                cells.add(sorted(frontier)[int(rng.integers(len(frontier)))])
        else:
            indexes = rng.choice(h*w, size=size, replace=False)
            cells = {(int(i)//w,int(i)%w) for i in indexes}
            if _connected(cells):
                continue
        shape = tuple(sorted(cells))
        if _box(cells) == (0,0,h-1,w-1) and shape not in used:
            return shape
    raise ValueError('no stencil')


def _anchors(points, shape, h, w):
    candidates = None
    for r,c in points:
        local = {(r-dr,c-dc) for dr,dc in shape}
        candidates = local if candidates is None else candidates & local
    if candidates is None:
        return set()
    sh, sw = _box(shape)[2]+1, _box(shape)[3]+1
    return {(r,c) for r,c in candidates if 0 <= r <= h-sh and 0 <= c <= w-sw}


def _partial(rng, shape, h, w, mode):
    sizes = list(range(2,len(shape)))
    if mode == 'near':
        sizes.reverse()
    elif mode == 'mixed':
        rng.shuffle(sizes)
    for size in sizes:
        for _ in range(100):
            pick = {shape[int(i)] for i in rng.choice(len(shape),size=size,replace=False)}
            if mode == 'sparse' and _box(pick) == _box(shape):
                continue
            if _anchors(pick,shape,h,w) == {(0,0)}:
                return pick
    raise ValueError('no unique partial')


def _distance(a,b):
    return min(max(abs(r-s),abs(c-d)) for r,c in a for s,d in b)


def _place(rng,h,w,shape,colors):
    sh,sw = _box(shape)[2]+1,_box(shape)[3]+1
    footprints, anchors = [],[]
    for i,color in enumerate(colors):
        choices = [(r,c) for r in range(h-sh+1) for c in range(w-sw+1)]
        rng.shuffle(choices)
        if i == 0 and rng.random() < .3:
            edge = [(r,c) for r,c in choices if r in (0,h-sh) or c in (0,w-sw)]
            edge_set = set(edge)
            choices = edge + [p for p in choices if p not in edge_set]
        for ar,ac in choices:
            if i == 0 and (h,w,ar,ac) in ((24,24,4,5),(22,22,3,6),(30,30,3,5)):
                continue
            cells = {(ar+r,ac+c) for r,c in shape}
            if any(_distance(cells,old) <= (6 if color == colors[j] else 1)
                   for j,old in enumerate(footprints)):
                continue
            footprints.append(cells)
            anchors.append((ar,ac))
            break
        else:
            raise ValueError('no placement')
    return footprints,anchors


def _clusters(points):
    remaining = set(points)
    result = []
    while remaining:
        component = {remaining.pop()}
        stack = list(component)
        while stack:
            r,c = stack.pop()
            close = {q for q in remaining if max(abs(q[0]-r),abs(q[1]-c)) <= 6}
            remaining -= close
            component |= close
            stack.extend(close)
        result.append(component)
    return result


def _infer(inp):
    h,w = inp.shape
    counts = np.bincount(inp.ravel().astype(int),minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > inp.size//2
    groups = [(color,group) for color in range(10) if color != bg
              for group in _clusters(set(map(tuple,np.argwhere(inp == color))))]
    assert len(groups) >= 3
    largest = max(len(group) for _,group in groups)
    refs = [(color,group) for color,group in groups if len(group) == largest]
    assert len(refs) == 1
    ref_color,ref_cells = refs[0]
    r0,c0,_,_ = _box(ref_cells)
    shape = tuple(sorted((r-r0,c-c0) for r,c in ref_cells))
    sh,sw = _box(shape)[2]+1,_box(shape)[3]+1
    assert 6 <= len(shape) <= 13 and 2 <= sh <= 6 and 2 <= sw <= 6
    assert (h,w,r0,c0) not in ((24,24,4,5),(22,22,3,6),(30,30,3,5))
    assert _anchors(ref_cells,shape,h,w) == {(r0,c0)}
    predicted = inp.copy()
    footprints = [(ref_color,ref_cells)]
    colors,sizes,growth = [],[],[]
    for color,seed in groups:
        if seed is ref_cells:
            continue
        assert 2 <= len(seed) < len(shape)
        anchors = _anchors(seed,shape,h,w)
        assert len(anchors) == 1
        ar,ac = next(iter(anchors))
        complete = {(ar+r,ac+c) for r,c in shape}
        assert all(_distance(complete,old) > (6 if color == old_color else 1)
                   for old_color,old in footprints)
        assert all(int(inp[r,c]) in (bg,color) for r,c in complete)
        for r,c in complete:
            predicted[r,c] = color
        footprints.append((color,complete))
        colors.append(color)
        sizes.append(len(seed))
        growth.append(_box(seed) != _box(complete))
    return predicted,{'bg':bg,'ref':ref_color,'anchor':(r0,c0),'shape':shape,
                      'dims':(sh,sw),'connected':_connected(shape),'copies':len(colors),
                      'colors':tuple(colors),'sizes':tuple(sizes),'growth':tuple(growth)}


def _pair(rng,regime,index,base,used,first_count):
    choices = list(range(3,7)) if regime == 'source_train' else list(range(2,9))
    if index == 1 and first_count in choices:
        choices.remove(first_count)
    copies = int(rng.choice(choices))
    low = max(17,12+2*copies)
    h,w = int(rng.integers(low,31)),int(rng.integers(low,31))
    if regime == 'source_train' and index == 0:
        bg,ref = base
    elif regime == 'source_train' and index == 1:
        bg = int(rng.choice([x for x in range(10) if x != base[0]]))
        ref = int(rng.choice([x for x in range(10) if x not in (bg,base[1])]))
    else:
        bg = base[0] if rng.random() < .5 else int(rng.integers(10))
        ref = base[1] if base[1] != bg and rng.random() < .5 else int(rng.choice([x for x in range(10) if x != bg]))
    available = [x for x in range(10) if x not in (bg,ref)]
    minimum_colors = max(1, (copies + 1)//2)
    maximum_colors = copies-1 if regime == 'source_train' and index == 1 else copies
    distinct = int(rng.integers(minimum_colors,maximum_colors+1))
    selected = [int(x) for x in rng.choice(available,size=distinct,replace=False)]
    copy_colors = [selected[i % distinct] for i in range(copies)]
    rng.shuffle(copy_colors)
    colors = [ref]+copy_colors
    for _ in range(100):
        shape = _stencil(rng,used,index == 0 or rng.random() < .78)
        try:
            fragments = [_partial(rng,shape,h,w,'sparse' if i == 0 else 'near' if i == 1 else 'mixed')
                         for i in range(copies)]
            rng.shuffle(fragments)
            partials = [None]+fragments
            footprints,anchors = _place(rng,h,w,shape,colors)
        except ValueError:
            continue
        inp,out = np.full((h,w),bg,dtype=np.int8),np.full((h,w),bg,dtype=np.int8)
        for i,cells in enumerate(footprints):
            for r,c in cells:
                out[r,c] = colors[i]
                if i == 0:
                    inp[r,c] = colors[i]
            if i:
                ar,ac = anchors[i]
                for dr,dc in partials[i]:
                    inp[ar+dr,ac+dc] = colors[i]
        try:
            predicted,info = _infer(inp)
            if not np.array_equal(predicted,out):
                continue
            if regime == 'source_train' and index == 0 and len(set(info['sizes'])) < 2:
                continue
        except AssertionError:
            continue
        used.add(shape)
        return {'input':inp,'output':out}
    raise ValueError('could not construct pair')


def _key(grid):
    return grid.shape,grid.astype(np.uint8,copy=False).tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert isinstance(puzzle['train'],list) and isinstance(puzzle['test'],list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen,infos = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            for grid in pair.values():
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype,np.integer) and np.all((0 <= grid)&(grid <= 9))
            inp,out = pair['input'],pair['output']
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            predicted,info = _infer(inp)
            assert np.array_equal(predicted,out)
            assert set(np.unique(inp)) == set(np.unique(out))
            assert any(info['growth'])
            assert info['copies'] in (range(3,7) if split == 'train' else range(2,9))
            infos[split].append(info)
    a,b = infos['train'][:2]
    assert a['connected'] and len(set(a['sizes'])) >= 2
    assert len(set(b['colors'])) < b['copies']
    assert a['bg'] != b['bg'] and a['ref'] != b['ref'] and a['copies'] != b['copies']


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    code = int(seed) % (1 << 64)
    num_train,num_test = _counts(code,num_train,num_test)
    rng = np.random.default_rng(code)
    base = tuple(int(x) for x in rng.choice(10,size=2,replace=False))
    pair_seeds = rng.integers(0,np.iinfo(np.int64).max,size=num_train+num_test)
    puzzle,used,cursor = {'train':[],'test':[]},set(),0
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[cursor]))
            cursor += 1
            first_count = None if index == 0 else _infer(puzzle[split][0]['input'])[1]['copies']
            puzzle[split].append(_pair(pair_rng,'source_train' if split == 'train' else 'target_test',
                                       index,base,used,first_count))
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    for i,args in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        p = generate_puzzle(3100+i,*args)
        validate_generated_puzzle(p)
        if args[0] is not None:
            assert len(p['train']) == args[0]
        if args[1] is not None:
            assert len(p['test']) == args[1]
    for args in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4)):
        try:
            generate_puzzle(0,*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    signatures,dimensions,copy_counts,backgrounds,references,connectivity = set(),set(),set(),set(),set(),set()
    train_counts,test_counts,fingerprints = set(),set(),set()
    growth,repeated_train = 0,0
    edge,interior,non_nw,non_se,holes,nonsquare = 0,0,0,0,0,0
    sparse,nearly,repeated_test,busier,comparable = 0,0,0,0,0
    for seed in range(42):
        p,q = generate_puzzle(seed),generate_puzzle(seed)
        def fingerprint(puzzle):
            return tuple((_key(pair['input']),_key(pair['output'])) for split in ('train','test') for pair in puzzle[split])
        assert fingerprint(p) == fingerprint(q)
        fingerprints.add(fingerprint(p))
        train_counts.add(len(p['train']))
        test_counts.add(len(p['test']))
        train_max = max(_infer(pair['input'])[1]['copies'] for pair in p['train'])
        test_max = max(_infer(pair['input'])[1]['copies'] for pair in p['test'])
        busier += test_max > train_max
        comparable += test_max <= train_max
        for split in ('train','test'):
            for pair in p[split]:
                predicted,info = _infer(pair['input'])
                assert np.array_equal(predicted,pair['output'])
                signatures.add(info['shape'])
                dimensions.add(pair['input'].shape)
                copy_counts.add(info['copies'])
                backgrounds.add(info['bg'])
                references.add(info['ref'])
                connectivity.add(info['connected'])
                growth += sum(info['growth'])
                ar,ac = info['anchor']
                sh,sw = info['dims']
                touches = ar == 0 or ac == 0 or ar+sh == pair['input'].shape[0] or ac+sw == pair['input'].shape[1]
                edge += touches
                interior += not touches
                non_nw += (0,0) not in info['shape']
                non_se += (sh-1,sw-1) not in info['shape']
                holes += _has_hole(info['shape'])
                nonsquare += pair['input'].shape[0] != pair['input'].shape[1]
                sparse += sum(size <= 4 for size in info['sizes'])
                nearly += sum(size >= len(info['shape'])-2 for size in info['sizes'])
                if split == 'train' and len(set(info['colors'])) < info['copies']:
                    repeated_train += 1
                if split == 'test' and len(set(info['colors'])) < info['copies']:
                    repeated_test += 1
    assert len(fingerprints) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 80 and len(dimensions) >= 35
    assert len(copy_counts) >= 5 and len(backgrounds) >= 8 and len(references) >= 8
    assert connectivity == {False,True} and repeated_train >= 42 and growth >= 100
    assert edge >= 20 and interior >= 20 and non_nw >= 20 and non_se >= 20
    assert holes >= 5 and nonsquare >= 100
    assert sparse >= 100 and nearly >= 100 and repeated_test >= 10
    assert busier >= 5 and comparable >= 5
