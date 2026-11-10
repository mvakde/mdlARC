import numpy as np


_SOURCE_EXAMPLES = {
    (
        (
            "2222222222",
            "2777777882",
            "2770777772",
            "2677777772",
            "2677777772",
            "2677557772",
            "2777557772",
            "2777777772",
            "2777777772",
            "2222222222",
        ),
        (
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7776777777",
            "7776557777",
            "0886557777",
        ),
    ),
    (
        (
            "1111777777",
            "1111777777",
            "1111777777",
            "7777777557",
            "7777877557",
            "7777777777",
            "7999777777",
            "7999777777",
            "7777777777",
            "7777777777",
        ),
        (
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777771111",
            "7559991111",
            "8559991111",
        ),
    ),
    (
        (
            "7777777777",
            "7777777717",
            "7777777711",
            "7777777797",
            "7777777797",
            "3333777797",
            "7777777797",
            "7777777797",
            "7777777777",
            "7777777777",
        ),
        (
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777779777",
            "7777779777",
            "7777779777",
            "1777779777",
            "1133339777",
        ),
    ),
    (
        (
            "7717774477",
            "7711774477",
            "7711774477",
            "7777774477",
            "7777771777",
            "7777771777",
            "7777671777",
            "7776671777",
            "7777771777",
            "7777771177",
        ),
        (
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777177777",
            "7777177777",
            "7777174477",
            "7717174477",
            "7611174477",
            "6611114477",
        ),
    ),
}


def _grid_lines(grid):
    return tuple(''.join(str(int(v)) for v in row) for row in grid)


def _grid_key(grid):
    a = np.ascontiguousarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    reached = {next(iter(cells))}
    stack = list(reached)
    while stack:
        r, c = stack.pop()
        for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if p in cells and p not in reached:
                reached.add(p)
                stack.append(p)
    return reached == cells


def _trim(mask):
    rr, cc = np.where(mask)
    return mask[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()


def _variants(mask):
    found = {}
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for turns in range(4):
            v = _trim(np.rot90(base, turns))
            found[(v.shape, v.tobytes())] = v
    return list(found.values())


def _shape(rng, kind):
    if kind == 'rectangle':
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        mask = np.ones((h, w), dtype=bool)
    elif kind == 'bar':
        length = int(rng.integers(2, 11))
        mask = np.ones((1, length), dtype=bool)
    elif kind == 'l':
        h, w = int(rng.integers(2, 9)), int(rng.integers(2, 8))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
    else:
        target = int(rng.integers(4, 19))
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if p in cells:
                        continue
                    trial = cells | {p}
                    rr = [q[0] for q in trial]
                    cc = [q[1] for q in trial]
                    if max(rr)-min(rr) < 8 and max(cc)-min(cc) < 8:
                        frontier.add(p)
            cells.add(sorted(frontier)[int(rng.integers(len(frontier)))])
        rr = [p[0] for p in cells]
        cc = [p[1] for p in cells]
        mask = np.zeros((max(rr)-min(rr)+1, max(cc)-min(cc)+1), dtype=bool)
        for r, c in cells:
            mask[r-min(rr), c-min(cc)] = True
    variants = _variants(mask)
    return variants[int(rng.integers(len(variants)))].copy()


def _evidence_pair_shapes(rng):
    # The smaller sparse shape has a larger box than the next solid shape.
    if rng.random() < .55:
        for _ in range(100):
            sparse = _shape(rng, 'irregular')
            sparse_area = int(sparse.sum())
            if 4 <= sparse_area <= 10 and sparse.size > sparse_area + 1:
                break
        else:
            raise RuntimeError('evidence shape')
    else:
        sparse_area = int(rng.choice([4, 5, 7, 8]))
        sparse = None
    if sparse is None and sparse_area == 4:
        sparse = np.array([[1,1,1],[1,0,0]], dtype=bool)
    elif sparse is None and sparse_area == 5:
        sparse = np.array([[1,1,1,1],[1,0,0,0]], dtype=bool)
    elif sparse is None and sparse_area == 7:
        sparse = np.array([[1,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0]], dtype=bool)
    elif sparse is None:
        sparse = np.array([[1,1,1,1,1],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]], dtype=bool)
    dense_area = sparse_area + 1
    if dense_area % 3 == 0:
        dense = np.ones((3, dense_area//3), dtype=bool)
    elif dense_area % 2 == 0:
        dense = np.ones((2, dense_area//2), dtype=bool)
    else:
        dense = np.ones((1, dense_area), dtype=bool)
    return [_variants(sparse)[int(rng.integers(len(_variants(sparse))))],
            _variants(dense)[int(rng.integers(len(_variants(dense))))]]


def _sample_shapes(rng, count, evidence):
    masks = _evidence_pair_shapes(rng) if evidence else []
    used = {int(m.sum()) for m in masks}
    kinds = ('rectangle', 'bar', 'l', 'irregular')
    for _ in range(count-len(masks)):
        for attempt in range(100):
            kind = kinds[int(rng.choice(4, p=[.30,.24,.20,.26]))]
            mask = _shape(rng, kind)
            area = int(mask.sum())
            if area not in used:
                masks.append(mask)
                used.add(area)
                break
        else:
            raise RuntimeError('distinct shapes unavailable')
    return masks


def _colors(rng, count, background, frame_color, repeat):
    palette = [c for c in range(10) if c != background and c != frame_color]
    rng.shuffle(palette)
    colors = palette[:count]
    if repeat and count >= 2:
        group_size = 3 if count >= 4 and rng.random() < .25 else 2
        indices = list(map(int, rng.choice(count, size=group_size, replace=False)))
        for index in indices[1:]:
            colors[index] = colors[indices[0]]
        if count >= 5 and rng.random() < .32:
            rest = [i for i in range(count) if i not in indices]
            a, b = map(int, rng.choice(rest, size=2, replace=False))
            colors[b] = colors[a]
    return colors


def _dimensions(rng, masks, frame):
    packed = sum(m.shape[1] for m in masks)
    tallest = max(m.shape[0] for m in masks)
    area = sum(int(m.sum()) for m in masks)
    if packed > 30:
        raise RuntimeError('packed width')
    min_h = max(8, tallest + (3 if frame else 1))
    min_w = max(8, packed, max(m.shape[1] for m in masks) + (3 if frame else 1))
    if min_h > 30 or min_w > 30:
        raise RuntimeError('grid extent')
    if packed <= 10 and tallest <= (7 if frame else 9) and area <= 32 and rng.random() < .28:
        return 10, 10
    mode = int(rng.integers(4))
    h_extra = (0, 3, 9, 24)[mode]
    w_extra = (2, 9, 3, 13)[mode]
    height = min(30, min_h + int(rng.integers(h_extra+1)))
    width = min(30, min_w + int(rng.integers(w_extra+1)))
    while height*width < area*2.8:
        if height <= width and height < 30:
            height += 1
        elif width < 30:
            width += 1
        else:
            raise RuntimeError('grid too full')
    return height, width


def _neighbors(cells):
    return {(r+dr,c+dc) for r,c in cells for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))}


def _place(rng, masks, colors, height, width, bg, frame_color, frame, contact, edge):
    n = len(masks)
    coords = [list(map(tuple, np.argwhere(m))) for m in masks]
    possible = []
    for mask, local in zip(masks, coords):
        mh,mw = mask.shape
        low = 1 if frame else 0
        origins = []
        for r in range(low, height-mh-low+1):
            for c in range(low, width-mw-low+1):
                cells = frozenset((r+int(y),c+int(x)) for y,x in local)
                on_edge = any(y == low or y == height-1-low or x == low or x == width-1-low for y,x in cells)
                origins.append((r,c,cells,on_edge))
        possible.append(origins)
    for _ in range(90):
        order = list(map(int, rng.permutation(n)))
        if contact:
            pairs = [(a,b) for a in range(n) for b in range(n) if a != b and colors[a] != colors[b]]
            if pairs:
                a,b = pairs[int(rng.integers(len(pairs)))]
                order = [a,b] + [i for i in order if i not in (a,b)]
            else:
                contact = False
        placed = {}
        for j,i in enumerate(order):
            choices = []
            for r,c,cells,on_edge in possible[i]:
                if j == 0 and edge and not on_edge:
                    continue
                if any(cells & other[2] for other in placed.values()):
                    continue
                if any(colors[i] == colors[k] and cells & _neighbors(other[2]) for k,other in placed.items()):
                    continue
                if j == 1 and contact and not (cells & _neighbors(placed[order[0]][2])):
                    continue
                choices.append((r,c,cells,on_edge))
            if not choices:
                break
            placed[i] = choices[int(rng.integers(len(choices)))]
        if len(placed) == n:
            grid = np.full((height,width), bg, dtype=np.int8)
            if frame:
                grid[0,:] = frame_color
                grid[-1,:] = frame_color
                grid[:,0] = frame_color
                grid[:,-1] = frame_color
            for i,(_,_,cells,_) in placed.items():
                for r,c in cells:
                    grid[r,c] = colors[i]
            return grid
    raise RuntimeError('placement')


def _pack(masks, colors, height, width, bg):
    output = np.full((height,width), bg, dtype=np.int8)
    cursor = 0
    for i in sorted(range(len(masks)), key=lambda i: int(masks[i].sum())):
        mask = masks[i]
        h,w = mask.shape
        assert cursor+w <= width
        output[height-h:height, cursor:cursor+w][mask] = colors[i]
        cursor += w
    return output


def _components(grid, bg):
    height,width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for r in range(height):
        for c in range(width):
            if seen[r,c] or int(grid[r,c]) == bg:
                continue
            color = int(grid[r,c])
            seen[r,c] = True
            stack = [(r,c)]
            cells = []
            while stack:
                y,x = stack.pop()
                cells.append((y,x))
                for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0 <= yy < height and 0 <= xx < width and not seen[yy,xx] and int(grid[yy,xx]) == color:
                        seen[yy,xx] = True
                        stack.append((yy,xx))
            cells.sort()
            rr = [p[0] for p in cells]
            cc = [p[1] for p in cells]
            result.append({'cells':frozenset(cells),'color':color,'count':len(cells),
                           'top':min(rr),'left':min(cc),'height':max(rr)-min(rr)+1,
                           'width':max(cc)-min(cc)+1})
    return result


def _perimeter(h,w):
    return frozenset([(0,c) for c in range(w)] + [(h-1,c) for c in range(w)] +
                     [(r,0) for r in range(1,h-1)] + [(r,w-1) for r in range(1,h-1)])


def _analyse_pair(pair):
    source,target = pair['input'],pair['output']
    freq = np.bincount(target.ravel().astype(int), minlength=10)
    assert np.count_nonzero(freq == freq.max()) == 1
    bg = int(freq.argmax())
    h,w = source.shape
    components = _components(source,bg)
    perimeter = _perimeter(h,w)
    frames = [p for p in components if p['cells'] == perimeter]
    assert len(frames) <= 1
    pieces = [p for p in components if p not in frames]
    sizes = [p['count'] for p in pieces]
    assert len(sizes) == len(set(sizes))
    assert 2 <= len(pieces) <= 6
    assert sum(p['width'] for p in pieces) <= w
    expected = np.full(source.shape,bg,dtype=np.int8)
    cursor = 0
    ordered = sorted(pieces,key=lambda p:p['count'])
    placed = []
    for p in ordered:
        cells = frozenset((h-p['height']+r-p['top'],cursor+c-p['left']) for r,c in p['cells'])
        for r,c in cells:
            expected[r,c] = p['color']
        placed.append(cells)
        cursor += p['width']
    assert np.array_equal(expected,target)
    box_order = sorted(pieces,key=lambda p:p['height']*p['width'])
    inversion = [p['count'] for p in box_order] != [p['count'] for p in ordered]
    contacts = any(a['color'] != b['color'] and a['cells'] & _neighbors(b['cells'])
                   for i,a in enumerate(pieces) for b in pieces[i+1:])
    same_merge = any(a['color'] == b['color'] and placed[i] & _neighbors(placed[j])
                     for i,a in enumerate(ordered) for j,b in enumerate(ordered) if j>i)
    return {'bg':bg,'frame':bool(frames),'frame_color':frames[0]['color'] if frames else None,
            'pieces':pieces,'ordered':ordered,'inversion':inversion,'contacts':contacts,
            'same_merge':same_merge}


def _valid_count(name,value,minimum,maximum):
    if value is None:
        return None
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not minimum <= int(value) <= maximum:
        raise ValueError(name)
    return int(value)


def _resolve_counts(seed,num_train,num_test):
    nt = _valid_count('num_train',num_train,2,10)
    ns = _valid_count('num_test',num_test,1,4)
    if nt is None and ns is None:
        ns = 1 + seed%4
        nt = max(2,ns+1) + (seed//4)%(11-max(2,ns+1))
    elif nt is None:
        nt = max(2,ns+1) + seed%(11-max(2,ns+1))
    elif ns is None:
        ns = 1 + seed%min(4,nt-1)
    if nt <= ns:
        raise ValueError('train must exceed test')
    return nt,ns


def _make_pair(rng, count, evidence, repeat, frame, contact, edge):
    for _ in range(160):
        try:
            masks = _sample_shapes(rng,count,evidence)
            bg = int(rng.integers(10))
            frame_color = int(rng.choice([c for c in range(10) if c != bg]))
            colors = _colors(rng,count,bg,frame_color,repeat)
            if evidence and colors[0] == colors[1]:
                colors[1] = next(c for c in range(10) if c not in (bg,frame_color,colors[0]))
            effective_contact = contact and len(set(colors)) > 1
            h,w = _dimensions(rng,masks,frame)
            source = _place(rng,masks,colors,h,w,bg,frame_color,frame,effective_contact,edge)
            target = _pack(masks,colors,h,w,bg)
            pair = {'input':source,'output':target}
            if np.array_equal(source,target):
                continue
            info = _analyse_pair(pair)
            if evidence and not info['inversion']:
                continue
            if effective_contact and not info['contacts']:
                continue
            return pair
        except (RuntimeError, AssertionError):
            continue
    raise RuntimeError('could not construct pair')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or int(seed) < 0:
        raise ValueError('seed')
    seed = int(seed)
    train_count,test_count = _resolve_counts(seed,num_train,num_test)
    rng = np.random.default_rng(seed)
    # Rules stay fixed. These modes only allocate visible examples, never change the rule.
    repeat_mode = int(rng.integers(3))   # unique, mixed, first repeated in test
    frame_mode = int(rng.integers(3))    # absent, occasional, frequent
    count_shift = rng.random() < .22
    frame_demo = int(rng.integers(train_count)) if frame_mode else -1
    puzzle = {'train':[],'test':[]}
    seen = set()
    for section,count in (('train',train_count),('test',test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0,np.iinfo(np.int64).max)))
            for _ in range(100):
                if section == 'train':
                    pieces = (int(pair_rng.integers(3,5)) if index == 0 else int(pair_rng.integers(2,5))) if count_shift else (int(pair_rng.integers(3,6)) if index == 0 else int(pair_rng.integers(2,7)))
                    evidence = index == 0
                    repeat = repeat_mode == 1 and (index == 1 or pair_rng.random() < .55)
                    frame = frame_mode != 0 and (index == frame_demo or (index != 0 and pair_rng.random() < (.20 if frame_mode == 1 else .55)))
                    contact = index == 0 or pair_rng.random() < .58
                else:
                    pieces = int(pair_rng.integers(4,7)) if count_shift else int(pair_rng.integers(2,7))
                    evidence = False
                    repeat = repeat_mode != 0 and pair_rng.random() < (.85 if repeat_mode == 2 else .78)
                    frame = frame_mode != 0 and pair_rng.random() < (.35 if frame_mode == 1 else .60)
                    contact = pair_rng.random() < .72
                edge = frame and pair_rng.random() < .53 or (not frame and pair_rng.random() < .28)
                pair = _make_pair(pair_rng,pieces,evidence,repeat,frame,contact,edge)
                key = _grid_key(pair['input'])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[section].append(pair)
                break
            else:
                raise RuntimeError('duplicate inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert isinstance(puzzle['train'],list) and isinstance(puzzle['test'],list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    infos = {'train':[],'test':[]}
    for section in ('train','test'):
        for pair in puzzle[section]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            source,target = pair['input'],pair['output']
            for grid in (source,target):
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype,np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            assert source.shape == target.shape and not np.array_equal(source,target)
            key = _grid_key(source)
            assert key not in seen
            seen.add(key)
            assert (_grid_lines(source),_grid_lines(target)) not in _SOURCE_EXAMPLES
            info = _analyse_pair(pair)
            assert len({p['count'] for p in info['pieces']}) == len(info['pieces'])
            if info['frame']:
                assert info['frame_color'] != info['bg']
                assert info['frame_color'] not in {p['color'] for p in info['pieces']}
            infos[section].append(info)
    assert any(i['inversion'] for i in infos['train'])
    assert any(i['contacts'] for i in infos['train'])
    if any(i['frame'] for i in infos['test']):
        assert any(i['frame'] for i in infos['train'])
    if any(len({p['color'] for p in i['pieces']}) < len(i['pieces']) for i in infos['test']):
        # Distinct-color contact in training demonstrates component boundaries.
        assert any(i['contacts'] for i in infos['train'])


def _same_puzzle(a,b):
    return all(len(a[k]) == len(b[k]) and all(np.array_equal(x[g],y[g]) for x,y in zip(a[k],b[k]) for g in ('input','output')) for k in ('train','test'))


def self_test() -> None:
    for args in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        p = generate_puzzle(200+int(args[0] or 0)+int(args[1] or 0),*args)
        validate_generated_puzzle(p)
    for args in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4)):
        try:
            generate_puzzle(3,*args)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count')
    a = generate_puzzle(902,6,3)
    assert _same_puzzle(a,generate_puzzle(902,6,3))
    assert not _same_puzzle(a,generate_puzzle(903,6,3))
    puzzles = [generate_puzzle(i) for i in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    all_pairs = [pair for p in puzzles for section in ('train','test') for pair in p[section]]
    assert len({_grid_key(pair['input']) for pair in all_pairs}) >= .98*len(all_pairs)
    infos = [_analyse_pair(pair) for pair in all_pairs]
    assert len({pair['input'].shape for pair in all_pairs}) >= 25
    assert len({i['bg'] for i in infos}) == 10
    assert {len(i['pieces']) for i in infos} >= {2,3,4,5,6}
    assert sum(i['frame'] for i in infos) > 0
    assert sum(i['contacts'] for i in infos) > 0
    assert sum(i['same_merge'] for i in infos) > 0
    repetition_regimes = set()
    background_varying = 0
    count_varying = 0
    for puzzle in puzzles:
        train_info = [_analyse_pair(p) for p in puzzle['train']]
        test_info = [_analyse_pair(p) for p in puzzle['test']]
        combined = train_info + test_info
        repeated = lambda i: len({p['color'] for p in i['pieces']}) < len(i['pieces'])
        repetition_regimes.add((any(map(repeated,train_info)),any(map(repeated,test_info))))
        background_varying += len({i['bg'] for i in combined}) > 1
        count_varying += len({len(i['pieces']) for i in combined}) > 1
    assert repetition_regimes >= {(False,False),(False,True),(True,True)}
    assert background_varying >= 30 and count_varying >= 30
    assert sum(i['inversion'] for i in infos) >= 42
    assert sum(i['contacts'] for i in infos) >= 100
    assert sum(i['frame'] for i in infos) >= 20
    assert sum(i['same_merge'] for i in infos) >= 10
    assert sum(p['input'].shape == (10,10) for p in all_pairs) >= 2
    assert sum(p['input'].shape[0] >= 28 for p in all_pairs) >= 2
    assert sum(p['input'].shape[1] >= 28 for p in all_pairs) >= 2
    assert sum(sum(r['width'] for r in i['pieces']) == p['input'].shape[1]
               for p,i in zip(all_pairs,infos)) >= 5
    assert sum(r['height'] > 1 and r['width'] > 1 and r['count'] == r['height']*r['width']
               for i in infos for r in i['pieces']) >= 40
    assert sum(r['height'] >= 6 for i in infos for r in i['pieces']) >= 10
    assert sum(r['count'] >= 12 for i in infos for r in i['pieces']) >= 20
    assert sum(any(i['ordered'][a]['color'] == i['ordered'][b]['color']
                   for a in range(len(i['ordered'])) for b in range(a+2,len(i['ordered'])))
               for i in infos) >= 5
    assert sum(any(sum(r['color'] == color for r in i['pieces']) >= 3
                   for color in {r['color'] for r in i['pieces']})
               for i in infos) >= 1
    assert sum(i['frame'] and any(r == 1 or r == p['input'].shape[0]-2 or
                                  c == 1 or c == p['input'].shape[1]-2
                                  for q in i['pieces'] for r,c in q['cells'])
               for p,i in zip(all_pairs,infos)) >= 5
