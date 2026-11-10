import numpy as np

_SOURCE_INPUT_ROWS = (
    (
        "9999999999999999",
        "7777777777777777",
        "7777777777777777",
        "7997797797777777",
        "7797797999777777",
        "7777797797777777",
        "7777777777777777",
        "7877777777777777",
        "8787787777777777",
        "7877888778779777",
        "7877878777899978",
        "7777777777779778",
        "7777777777779778",
        "7777777777778777",
        "7777777777777777",
        "8888888888888888",
    ),
    (
        "2222222222222222",
        "7777777777777777",
        "7777777777777777",
        "7777277777727277",
        "7777727777772777",
        "7777727777775777",
        "7777777777755572",
        "7777777777777772",
        "7757777777777777",
        "7555777777777777",
        "7777772275777777",
        "7777777777577777",
        "7777777777777777",
        "7777777777777777",
        "7777777777777777",
        "5555555555555555",
    ),
    (
        "1111111111111111",
        "7777777777774777",
        "7777777777774777",
        "7777777777777777",
        "7777771777177777",
        "1777771771717777",
        "7177771777777777",
        "7717717117777777",
        "7777777777777774",
        "7777744477777744",
        "7777477747777777",
        "7777777777777777",
        "7777777777777777",
        "7777777777777777",
        "7777777777777777",
        "4444444444444444",
    ),
)


def _check_count(name, value, low, high):
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}, inclusive"
        )
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _check_count("num_train", num_train, 2, 10)
    num_test = _check_count("num_test", num_test, 1, 4)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = (
            num_test
            + 1
            + (seed // 4) % (10 - num_test)
        )
    elif num_train is None:
        num_train = (
            num_test
            + 1
            + seed % (10 - num_test)
        )
    else:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + seed % maximum_test

    return int(num_train), int(num_test)


def _normal(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r-r0, c-c0) for r,c in cells))


def _shape(rng, limit_h, limit_w, kind=None):
    """Build a connected mask, then choose its rotation and reflection."""
    kinds = ('dot', 'line', 'diagonal', 'bend', 'branch', 'arch',
             'ring', 'solid', 'walk')
    for _ in range(80):
        mode = kind or kinds[int(rng.integers(len(kinds)))]
        if mode == 'dot':
            cells = {(0,0)}
        elif mode in ('line', 'diagonal'):
            n = int(rng.integers(2,7))
            cells = {(i, i if mode == 'diagonal' else 0) for i in range(n)}
        elif mode == 'bend':
            a,b = int(rng.integers(2,6)), int(rng.integers(2,6))
            cells = {(r,0) for r in range(a)} | {(a-1,c) for c in range(b)}
        elif mode == 'branch':
            a,b = int(rng.integers(2,6)), int(rng.integers(3,7))
            joint = int(rng.integers(a))
            cells = {(r,b//2) for r in range(a)} | {(joint,c) for c in range(b)}
        elif mode == 'arch':
            a,b = int(rng.integers(2,6)), int(rng.integers(3,8))
            cells = {(0,c) for c in range(b)} | {(r,0) for r in range(a)} | {(r,b-1) for r in range(a)}
        elif mode == 'ring':
            a,b = int(rng.integers(3,7)), int(rng.integers(3,8))
            cells = {(r,c) for r in range(a) for c in range(b)
                     if r in (0,a-1) or c in (0,b-1)}
        elif mode == 'solid':
            a,b = int(rng.integers(2,5)), int(rng.integers(2,6))
            cells = {(r,c) for r in range(a) for c in range(b)}
        else:
            cells = {(0,0)}
            for _ in range(int(rng.integers(3,15))):
                points = sorted(cells)
                r,c = points[int(rng.integers(len(points)))]
                cells.add((r+int(rng.integers(-1,2)), c+int(rng.integers(-1,2))))
        if rng.random() < .5:
            cells = {(c,r) for r,c in cells}
        if rng.random() < .5:
            cells = {(-r,c) for r,c in cells}
        if rng.random() < .5:
            cells = {(r,-c) for r,c in cells}
        result = _normal(cells)
        if 1+max(r for r,_ in result) <= limit_h and 1+max(c for _,c in result) <= limit_w:
            return result
    return ((0,0),(1,1)) if kind == 'diagonal' else ((0,0),)


def _near(a,b):
    return any(abs(r-s)<=1 and abs(c-d)<=1 for r,c in a for s,d in b)


def _distance(rng, maximum, hint=None):
    if maximum <= 0:
        return 0
    if hint == 'near':
        return min(maximum, int(rng.integers(0,2)))
    if hint == 'far':
        return int(rng.integers(max(0,3*maximum//4),maximum+1))
    mode = int(rng.choice(4,p=(.13,.22,.38,.27)))
    if mode == 0:
        return 0
    if mode == 1:
        return int(rng.integers(1,min(3,maximum)+1))
    if mode == 2:
        return int(rng.integers(maximum+1))
    return int(rng.integers(max(0,3*maximum//4),maximum+1))


def _dimension(rng):
    bucket = int(rng.choice(3,p=(.23,.40,.37)))
    return int(rng.integers(*((9,14),(14,22),(22,31))[bucket]))


def _pair(rng, orientation, palette, background, first_train, regime):
    assert regime in ('source_train','target_test')
    for _ in range(36):
        h,w = _dimension(rng),_dimension(rng)
        if first_train:
            h = max(h,12)
        max_count = min(10,max(3,h*w//33))
        low_count = 3 if first_train else 2
        count = int(rng.integers(low_count,max_count+1))
        if rng.random() < (.45 if regime == 'target_test' else .33):
            count = max(low_count,max_count-int(rng.integers(2)))
        top_count = 2 if first_train else int(rng.integers(1,count))
        roles = [0]*top_count + [1]*(count-top_count)
        if not first_train:
            rng.shuffle(roles)
        size_limit = 5 if h*w//count < 32 else 7
        shapes = []
        for i in range(count):
            kind = 'diagonal' if first_train and i == 0 else None
            if kind is None and regime == 'target_test' and rng.random() < .35:
                kind = ('arch','ring','branch','walk')[int(rng.integers(4))]
            shapes.append(_shape(rng,min(h-2,size_limit),min(w,size_limit),kind))
        if count >= 3 and rng.random() < .38:
            shapes[2] = shapes[0]
        if count >= 4 and rng.random() < .38:
            shapes[1] = shapes[0]
        placed = []
        for i,(role,shape) in enumerate(zip(roles,shapes)):
            sh = 1+max(r for r,_ in shape)
            sw = 1+max(c for _,c in shape)
            maximum = h-2-sh
            found = False
            for _ in range(160):
                hint = 'near' if first_train and i == 0 else ('far' if first_train and i == 1 else None)
                if hint is None and regime == 'target_test' and rng.random() < .28:
                    hint = 'far'
                distance = _distance(rng,maximum,hint)
                base_r = 1+distance if role == 0 else h-1-sh-distance
                if placed and rng.random() < .28:
                    old = placed[int(rng.integers(len(placed)))][1]
                    base_c = min(max(0,min(c for _,c in old)+int(rng.integers(-sw,3))),w-sw)
                else:
                    base_c = int(rng.integers(w-sw+1))
                cells = {(base_r+r,base_c+c) for r,c in shape}
                if any(cells & old for _,old,_ in placed):
                    continue
                if any(role == old_role and _near(cells,old) for old_role,old,_ in placed):
                    continue
                shift = -distance if role == 0 else distance
                moved = {(r+shift,c) for r,c in cells}
                if any(moved & old for _,_,old in placed):
                    continue
                placed.append((role,cells,moved))
                found = True
                break
            if not found:
                break
        if len(placed) != count:
            continue
        shifts = {0:[],1:[]}
        for role,cells,moved in placed:
            shifts[role].append(abs(min(r for r,_ in cells)-min(r for r,_ in moved)))
        if not any(s for part in shifts.values() for s in part):
            continue
        if first_train and not any(len(part)>=2 and len(set(part))>=2 for part in shifts.values()):
            continue
        inp = np.full((h,w),background,dtype=np.int8)
        out = np.full((h,w),background,dtype=np.int8)
        inp[0,:] = out[0,:] = palette[0]
        inp[-1,:] = out[-1,:] = palette[1]
        for role,cells,moved in placed:
            for r,c in cells:
                inp[r,c] = palette[role]
            for r,c in moved:
                out[r,c] = palette[role]
        return (inp,out) if orientation == 'horizontal' else (inp.T.copy(),out.T.copy())
    raise RuntimeError('could not place a nonconflicting scene')


def _key(grid):
    return (grid.shape,np.ascontiguousarray(grid,dtype=np.int16).tobytes())


def _components(grid,color):
    remaining = {(int(r)+1,int(c)) for r,c in np.argwhere(grid[1:-1,:]==color)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = {start}
        while stack:
            r,c = stack.pop()
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if dr == dc == 0:
                        continue
                    point = (r+dr,c+dc)
                    if point in remaining:
                        remaining.remove(point)
                        component.add(point)
                        stack.append(point)
        result.append(component)
    return result


def _four_connected(cells):
    remaining = set(cells)
    stack = [remaining.pop()]
    while stack:
        r,c = stack.pop()
        for point in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if point in remaining:
                remaining.remove(point)
                stack.append(point)
    return not remaining


def _analyze_pair(inp,out):
    assert inp.shape == out.shape
    horizontal = (np.all(inp[0]==inp[0,0]) and np.all(inp[-1]==inp[-1,0])
                  and inp[0,0] != inp[-1,0])
    vertical = (np.all(inp[:,0]==inp[0,0]) and np.all(inp[:,-1]==inp[0,-1])
                and inp[0,0] != inp[0,-1])
    assert horizontal != vertical
    direction = 'horizontal' if horizontal else 'vertical'
    a = inp if horizontal else inp.T
    b = out if horizontal else out.T
    n,m = a.shape
    upper,lower = int(a[0,0]),int(a[-1,0])
    assert np.all(a[0]==upper) and np.all(a[-1]==lower)
    assert np.all(b[0]==upper) and np.all(b[-1]==lower)
    vals = np.unique(a[1:-1])
    assert len(vals)==3
    background = next(int(x) for x in vals if x not in (upper,lower))
    assert set(int(x) for x in vals)=={background,upper,lower}
    assert set(int(x) for x in np.unique(b))=={background,upper,lower}
    expected = np.full_like(a,background)
    expected[0] = upper
    expected[-1] = lower
    occupied = set()
    objects = []
    for role,color in ((0,upper),(1,lower)):
        comps = _components(a,color)
        assert comps
        for cells in comps:
            shift = 1-min(r for r,_ in cells) if role==0 else n-2-max(r for r,_ in cells)
            assert shift<=0 if role==0 else shift>=0
            moved = {(r+shift,c) for r,c in cells}
            assert all(1<=r<n-1 and 0<=c<m for r,c in moved)
            assert not moved & occupied
            occupied.update(moved)
            for r,c in moved:
                expected[r,c] = color
            objects.append({'role':role,'color':color,'cells':cells,
                            'shape':_normal(cells),'shift':shift,
                            'height':1+max(r for r,_ in cells)-min(r for r,_ in cells),
                            'width':1+max(c for _,c in cells)-min(c for _,c in cells)})
    assert any(obj['shift'] != 0 for obj in objects)
    assert np.array_equal(expected,b)
    return {'orientation':direction,'height':inp.shape[0],'width':inp.shape[1],
            'background':background,'colors':(upper,lower),'objects':objects}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    train_count,test_count = _resolve_counts(seed,num_train,num_test)
    rng = np.random.default_rng(seed)
    # Palette anchors supply optional literal repetition, not a hidden color map.
    anchor = [int(x) for x in rng.choice(10,size=2,replace=False)]
    mode = int(rng.integers(3))
    orientations = []
    total = train_count+test_count
    for i in range(total):
        if mode == 2:
            orientations.append(('horizontal','vertical')[i] if i<2 else
                                ('horizontal','vertical')[int(rng.integers(2))])
        else:
            orientations.append(('horizontal','vertical')[mode])
    puzzle = {'train':[],'test':[]}
    used = set()
    for i in range(total):
        section = 'train' if i<train_count else 'test'
        for _ in range(60):
            background = int(rng.integers(10))
            if rng.random()<.53 and background not in anchor:
                colors = anchor.copy()
                if rng.random()<.5:
                    colors.reverse()
            else:
                colors = [int(x) for x in rng.choice(
                    [color for color in range(10) if color!=background],size=2,replace=False)]
            if background in colors:
                continue
            inp,out = _pair(rng,orientations[i],colors,background,i==0,
                            'source_train' if section=='train' else 'target_test')
            key = _key(inp)
            if key in used or tuple(''.join(str(int(x)) for x in row) for row in inp) in _SOURCE_INPUT_ROWS:
                continue
            used.add(key)
            puzzle[section].append({'input':inp,'output':out})
            break
        else:
            raise RuntimeError('could not construct a unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert isinstance(puzzle['train'],list) and isinstance(puzzle['test'],list)
    assert 2<=len(puzzle['train'])<=10
    assert 1<=len(puzzle['test'])<=4
    assert len(puzzle['train'])>len(puzzle['test'])
    seen = set()
    analyses = {'train':[],'test':[]}
    for section in ('train','test'):
        for pair in puzzle[section]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            inp,out = pair['input'],pair['output']
            for grid in (inp,out):
                assert isinstance(grid,np.ndarray) and grid.ndim==2
                assert 1<=grid.shape[0]<=30 and 1<=grid.shape[1]<=30
                assert np.issubdtype(grid.dtype,np.integer)
                assert np.all((grid>=0)&(grid<=9))
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            assert tuple(''.join(str(int(x)) for x in row) for row in inp) not in _SOURCE_INPUT_ROWS
            analyses[section].append(_analyze_pair(inp,out))
    assert all({obj['role'] for obj in a['objects']}=={0,1}
               for section in analyses.values() for a in section)
    assert any(any(len(shifts)>=2 and len(set(shifts))>=2 for shifts in
                   ([obj['shift'] for obj in a['objects'] if obj['role']==role]
                    for role in (0,1))) for a in analyses['train'])
    assert any(not _four_connected(obj['cells']) for a in analyses['train']
               for obj in a['objects'])
    assert {a['orientation'] for a in analyses['test']} <= {
        a['orientation'] for a in analyses['train']}


def _puzzles_equal(a,b):
    return all(len(a[s])==len(b[s]) and all(
        np.array_equal(x[k],y[k]) for x,y in zip(a[s],b[s]) for k in ('input','output'))
        for s in ('train','test'))


def _is_rectangular_ring(shape):
    cells = set(shape)
    h = 1+max(r for r,c in cells)
    w = 1+max(c for r,c in cells)
    return (h>=3 and w>=3 and
            all((r,c) in cells for r in range(h) for c in range(w)
                if r in (0,h-1) or c in (0,w-1)) and
            any((r,c) not in cells for r in range(1,h-1)
                for c in range(1,w-1)))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(12345,7,3),generate_puzzle(12345,7,3))
    assert _puzzles_equal(generate_puzzle(31415),generate_puzzle(31415))
    for seed,train,test in ((900,2,1),(901,10,1),(902,10,4),
                            (910,None,1),(911,None,4),(912,2,None),(913,10,None)):
        puzzle = generate_puzzle(seed,train,test)
        if train is not None:
            assert len(puzzle['train'])==train
        if test is not None:
            assert len(puzzle['test'])==test
    for args in ((1,1,None),(1,11,None),(1,None,0),(1,None,5),
                 (1,2,2),(1,3,4),(1,True,1)):
        try:
            generate_puzzle(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal pair counts accepted')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles})>=4
    assert len({len(p['test']) for p in puzzles})>=3
    signatures = {tuple(_key(pair['input']) for section in ('train','test')
                        for pair in p[section]) for p in puzzles}
    assert len(signatures)==42
    analyses = [_analyze_pair(pair['input'],pair['output'])
                for p in puzzles for section in ('train','test') for pair in p[section]]
    assert {a['orientation'] for a in analyses}=={'horizontal','vertical'}
    assert len({(a['height'],a['width']) for a in analyses})>=70
    assert len({a['background'] for a in analyses})==10
    assert max(len(a['objects']) for a in analyses)>=8
    assert any(len(a['objects'])==2 for a in analyses)
    assert any(obj['shift']==0 for a in analyses for obj in a['objects'])
    assert len({obj['shape'] for a in analyses for obj in a['objects']})>=75
    assert {len(a['objects']) for a in analyses}==set(range(2,11))
    assert any(a['background']==7 for a in analyses)
    assert any(abs(a['height']-a['width'])>=8 for a in analyses)
    assert any(obj['height']>=4 and obj['width']>=4 for a in analyses
               for obj in a['objects'])
    assert any(any(x['shape']==y['shape'] and x['role']!=y['role']
                   for i,x in enumerate(a['objects']) for y in a['objects'][i+1:])
               for a in analyses)
    assert any(any(x['shape']==y['shape'] and x['role']==y['role']
                   for i,x in enumerate(a['objects']) for y in a['objects'][i+1:])
               for a in analyses)
    assert any(any(x['role']!=y['role'] and _near(x['cells'],y['cells'])
                   for i,x in enumerate(a['objects']) for y in a['objects'][i+1:])
               for a in analyses)
    assert any(any(_near({(r+x['shift'],c) for r,c in x['cells']},
                         {(r+y['shift'],c) for r,c in y['cells']})
                   for i,x in enumerate(a['objects']) for y in a['objects'][i+1:])
               for a in analyses)
    assert any(any(obj['shift']==0 for obj in a['objects']) and
               any(obj['shift']!=0 for obj in a['objects']) for a in analyses)
    for role in (0,1):
        assert any(obj['role']==role and abs(obj['shift'])>=8 for a in analyses
                   for obj in a['objects'])
    assert any(max(sum(obj['role']==role for obj in a['objects']) for role in (0,1))>=6
               for a in analyses)
    assert any(len({a['orientation'] for a in
                    [_analyze_pair(pair['input'],pair['output'])
                     for section in ('train','test') for pair in p[section]]})==2
               for p in puzzles)
    assert all(len({pair['input'].shape for section in ('train','test')
                    for pair in p[section]})>=2 for p in puzzles)
    assert all(len({_analyze_pair(pair['input'],pair['output'])['background']
                    for section in ('train','test') for pair in p[section]})>=2
               for p in puzzles)
    assert any(any(a['orientation']=='vertical' and len(a['objects'])>=6 and
                   max(a['height'],a['width'])>=22 for a in
                   [_analyze_pair(pair['input'],pair['output'])
                    for section in ('train','test') for pair in p[section]])
               for p in puzzles)
    regimes = []
    for p in puzzles:
        train = [_analyze_pair(x['input'],x['output']) for x in p['train']]
        test = [_analyze_pair(x['input'],x['output']) for x in p['test']]
        both = train+test
        assert len({len(a['objects']) for a in both})>=2
        train_colors = {color for a in train for color in a['colors']}
        test_colors = {color for a in test for color in a['colors']}
        train_shapes = {x['shape'] for a in train for x in a['objects']}
        test_shapes = {x['shape'] for a in test for x in a['objects']}
        palettes = [a['colors'] for a in both]
        regimes.append((bool(train_colors & test_colors),
                        bool(test_colors-train_colors),
                        bool(train_shapes & test_shapes),
                        bool(test_shapes-train_shapes),
                        any((b,a) in palettes for a,b in palettes)))
    assert all(any(row[i] for row in regimes) for i in range(5))
    assert any(_is_rectangular_ring(obj['shape'])
               for a in analyses for obj in a['objects'])
