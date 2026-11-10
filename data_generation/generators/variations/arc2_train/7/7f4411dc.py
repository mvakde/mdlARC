import numpy as np


def _literal_grid_key(rows):
    return f"{len(rows)}x{len(rows[0])}:" + "/".join(rows)


_SOURCE_PAIR_KEYS = {
    (
        _literal_grid_key((
            "00000000000000000",
            "00000000000060000",
            "06000060000666000",
            "00000000000666006",
            "00006000000000000",
            "00066660000000000",
            "00066660000060000",
            "00066660000000000",
            "00000000000000000",
            "00000000000006000",
            "00060000066666000",
            "00000000066666000",
            "00660006066666000",
            "00660000000000000",
            "00660000000000000",
            "00000000000006000",
            "00000000006000000",
        )),
        _literal_grid_key((
            "00000000000000000",
            "00000000000000000",
            "00000000000666000",
            "00000000000666000",
            "00000000000000000",
            "00066660000000000",
            "00066660000000000",
            "00066660000000000",
            "00000000000000000",
            "00000000000000000",
            "00000000066666000",
            "00000000066666000",
            "00660000066666000",
            "00660000000000000",
            "00660000000000000",
            "00000000000000000",
            "00000000000000000",
        )),
    ),
    (
        _literal_grid_key((
            "0700000077777",
            "0000000077777",
            "0000700077777",
            "0077770000000",
            "0077770000000",
            "0077770000700",
            "0000000000000",
            "0000000000000",
            "0700000077700",
            "0000000077700",
            "0000000000070",
            "0000000000000",
            "0000700000000",
        )),
        _literal_grid_key((
            "0000000077777",
            "0000000077777",
            "0000000077777",
            "0077770000000",
            "0077770000000",
            "0077770000000",
            "0000000000000",
            "0000000000000",
            "0000000077700",
            "0000000077700",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        )),
    ),
    (
        _literal_grid_key((
            "0500000",
            "0000005",
            "0555500",
            "0555500",
            "0555550",
            "5000000",
            "0000050",
        )),
        _literal_grid_key((
            "0000000",
            "0000000",
            "0555500",
            "0555500",
            "0555500",
            "0000000",
            "0000000",
        )),
    ),
    (
        _literal_grid_key((
            "0000000008",
            "0800000800",
            "0888000000",
            "0888000800",
            "8000000000",
            "0000888888",
            "0000888880",
            "8000888880",
            "0080000000",
            "0000800008",
        )),
        _literal_grid_key((
            "0000000000",
            "0000000000",
            "0888000000",
            "0888000000",
            "0000000000",
            "0000888880",
            "0000888880",
            "0000888880",
            "0000000000",
            "0000000000",
        )),
    ),
}


def _support_mask(mask):
    supported = np.zeros_like(mask, dtype=bool)
    blocks = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    supported[:-1, :-1] |= blocks
    supported[1:, :-1] |= blocks
    supported[:-1, 1:] |= blocks
    supported[1:, 1:] |= blocks
    return supported


def _array_key(grid):
    return (grid.shape, grid.tobytes())


def _compact_grid(grid):
    rows = ("".join(str(int(x)) for x in row) for row in grid)
    return f"{grid.shape[0]}x{grid.shape[1]}:" + "/".join(rows)


def _components(mask):
    remaining = mask.copy()
    height, width = mask.shape
    result = []
    while remaining.any():
        r, c = map(int, np.argwhere(remaining)[0])
        remaining[r, c] = False
        stack = [(r, c)]
        cells = []
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if 0 <= nx < height and 0 <= ny < width and remaining[nx, ny]:
                    remaining[nx, ny] = False
                    stack.append((nx, ny))
        result.append(cells)
    return result


def _rectangles(mask):
    rectangles = []
    for cells in _components(mask):
        rr = [r for r, _ in cells]
        cc = [c for _, c in cells]
        r, c = min(rr), min(cc)
        h, w = max(rr)-r+1, max(cc)-c+1
        assert h >= 2 and w >= 2
        assert len(cells) == h*w and mask[r:r+h,c:c+w].all()
        rectangles.append((r,c,h,w))
    return rectangles


def _noise_stats(noise, output):
    attached = False
    isolated = False
    cluster = False
    h, w = noise.shape
    for cells in _components(noise):
        if len(cells) >= 2:
            cluster = True
        for r, c in cells:
            if any(0 <= nr < h and 0 <= nc < w and output[nr,nc]
                   for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))):
                attached = True
        if len(cells) == 1:
            r,c = cells[0]
            if not output[max(0,r-1):min(h,r+2),max(0,c-1):min(w,c+2)].any():
                isolated = True
    return attached, isolated, cluster


def _analyze_pair(pair):
    inp, out = pair['input'], pair['output']
    assert inp.shape == out.shape
    values, counts = np.unique(out, return_counts=True)
    assert len(values) == 2
    bg = int(values[int(np.argmax(counts))])
    fg = int(values[int(np.argmin(counts))])
    assert int((out == fg).sum()) * 3 < out.size
    assert set(map(int, np.unique(inp))) == {bg, fg}
    input_mask = inp == fg
    output_mask = out == fg
    assert np.array_equal(output_mask, _support_mask(input_mask))
    assert np.all(input_mask[output_mask])
    rects = _rectangles(output_mask)
    assert rects
    noise = input_mask & ~output_mask
    attached, isolated, cluster = _noise_stats(noise, output_mask)
    return {'background':bg, 'foreground':fg, 'rectangles':rects,
            'noise_count':int(noise.sum()), 'attached':attached,
            'isolated':isolated, 'cluster':cluster}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert set(puzzle) == {'train','test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    analyses = {'train': [], 'test': []}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert set(pair) == {'input','output'}
            for grid in (pair['input'], pair['output']):
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype,np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            key = _array_key(pair['input'])
            assert key not in seen
            seen.add(key)
            assert (_compact_grid(pair['input']),_compact_grid(pair['output'])) not in _SOURCE_PAIR_KEYS
            analyses[split].append(_analyze_pair(pair))
    assert any(a['noise_count'] > 0 for a in analyses['train'])
    assert any(a['attached'] for a in analyses['train'])
    assert any(a['isolated'] for a in analyses['train'])
    assert any(a['cluster'] for a in analyses['train'])


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _checked_count('num_train',num_train,2,10)
    test = _checked_count('num_test',num_test,1,4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError('num_train must exceed num_test')
    elif train is None and test is None:
        test = 1 + seed % 4
        train = int(rng.integers(max(2,test+1),11))
    elif train is None:
        train = int(rng.integers(max(2,test+1),11))
    else:
        test = int(rng.integers(1,min(4,train-1)+1))
    return train,test


def _sample_shape(rng, count, regime):
    # The small and near-limit strata are independent of palette and count.
    bands = ((6,10),(10,18),(18,25),(25,30))
    weights = (0.18,0.44,0.28,0.10) if regime == 'source_train' else (0.12,0.36,0.34,0.18)
    band = int(rng.choice(4,p=weights))
    low, high = bands[band]
    height = int(rng.integers(low,high+1))
    width = int(rng.integers(low,high+1))
    if count >= 3:
        height = max(height, 10 + 2*(count-3))
        width = max(width, 10 + 2*(count-3))
    return height,width


def _sample_size(rng, height, width, count):
    # Long sides are uncommon but sampled independently for every rectangle.
    max_h = min(12,height-2)
    max_w = min(12,width-2)
    if count >= 4:
        max_h = min(max_h, 7)
        max_w = min(max_w, 7)
    def side(limit):
        if rng.random() < 0.68:
            return int(rng.integers(2,min(5,limit)+1))
        return int(rng.integers(2,limit+1))
    return side(max_h),side(max_w)


def _can_place(occupied, r, c, h, w):
    H,W = occupied.shape
    if r < 0 or c < 0 or r+h > H or c+w > W:
        return False
    if occupied[max(0,r-1):min(H,r+h+1),c:c+w].any():
        return False
    if occupied[r:r+h,max(0,c-1):min(W,c+w+1)].any():
        return False
    return True


def _place_rectangles(rng, height, width, sizes, relation):
    occupied = np.zeros((height,width),dtype=bool)
    rects = []
    for index,(h,w) in enumerate(sizes):
        candidates = []
        if index == 1 and relation != 'free':
            r0,c0,h0,w0 = rects[0]
            if relation == 'diagonal':
                for r in (r0-h,r0+h0):
                    for c in (c0-w,c0+w0):
                        candidates.append((r,c))
            elif relation == 'corridor':
                candidates.extend(((r0+h0+1,c0),(r0-h-1,c0),
                                   (r0,c0+w0+1),(r0,c0-w-1)))
            elif relation == 'aligned':
                candidates.extend(((r0,c0+w0+2),(r0+h0+2,c0),
                                   (r0,c0-w-2),(r0-h-2,c0)))
            rng.shuffle(candidates)
        positions = [(r,c) for r in range(height-h+1) for c in range(width-w+1)]
        rng.shuffle(positions)
        candidates.extend(positions)
        placed = False
        for r,c in candidates:
            if _can_place(occupied,r,c,h,w):
                occupied[r:r+h,c:c+w] = True
                rects.append((r,c,h,w))
                placed = True
                break
        if not placed:
            return None,None
    if not np.array_equal(_support_mask(occupied),occupied):
        return None,None
    return occupied,rects


def _try_add(current, intended, cells):
    h,w = current.shape
    if any(r < 0 or c < 0 or r >= h or c >= w or current[r,c] for r,c in cells):
        return False
    proposal = current.copy()
    for r,c in cells:
        proposal[r,c] = True
    if not np.array_equal(_support_mask(proposal),intended):
        return False
    current[:] = proposal
    return True


def _add_noise(rng, intended, mode):
    current = intended.copy()
    h,w = current.shape
    if mode == 'clean':
        return current
    # A separated run, bend, or three-corner 2x2 teaches that size and
    # connectedness alone do not determine whether a mark survives.
    if mode == 'cluster':
        patterns = [((0,0),(0,1)), ((0,0),(1,0),(2,0)),
                    ((0,0),(0,1),(1,0)), ((0,0),(1,0),(1,1),(2,1))]
        anchors = [(r,c) for r in range(h) for c in range(w)]
        rng.shuffle(anchors)
        rng.shuffle(patterns)
        made = False
        for pattern in patterns:
            for ar,ac in anchors:
                cells = [(ar+dr,ac+dc) for dr,dc in pattern]
                if any(not (0 <= r < h and 0 <= c < w) for r,c in cells):
                    continue
                if any(intended[max(0,r-1):min(h,r+2),max(0,c-1):min(w,c+2)].any()
                       for r,c in cells):
                    continue
                if _try_add(current,intended,cells):
                    made = True
                    break
            if made:
                break
        if not made:
            return None
    if mode in ('attached','mixed','cluster'):
        candidates = [(r,c) for r in range(h) for c in range(w)
                      if not current[r,c] and
                      any(0 <= nr < h and 0 <= nc < w and intended[nr,nc]
                          for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)))]
        rng.shuffle(candidates)
        if not any(_try_add(current,intended,[(r,c)]) for r,c in candidates):
            return None
    if mode in ('isolated','mixed','cluster'):
        candidates = [(r,c) for r in range(h) for c in range(w)
                      if not current[max(0,r-1):min(h,r+2),max(0,c-1):min(w,c+2)].any()]
        rng.shuffle(candidates)
        if not any(_try_add(current,intended,[(r,c)]) for r,c in candidates):
            return None
    target = int(rng.integers(1, min(22,max(3,h*w//15))+1))
    if mode == 'isolated':
        target = max(1,target//2)
    # Keep tiny supported rectangles visually dominant even on large grids.
    noise_budget = max(6,min(24,int(intended.sum()*0.7)))
    already_added = int((current & ~intended).sum())
    target = min(target,max(0,noise_budget-already_added))
    for _ in range(target):
        candidates = [(r,c) for r in range(h) for c in range(w) if not current[r,c]]
        rng.shuffle(candidates)
        added = False
        for r,c in candidates:
            if mode == 'isolated' and current[max(0,r-1):min(h,r+2),max(0,c-1):min(w,c+2)].any():
                continue
            if mode == 'attached' and not any(0 <= nr < h and 0 <= nc < w and intended[nr,nc]
                                              for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))):
                continue
            if _try_add(current,intended,[(r,c)]):
                added = True
                break
        if not added:
            break
    if not np.array_equal(_support_mask(current),intended):
        return None
    return current


def _pair_colors(rng, palette, style, split, index):
    if style == 'stable':
        return palette[0],palette[1]
    if style == 'reused':
        if split == 'train' and index < 2:
            return palette[0],palette[1+index]
        return palette[0],palette[1+int(rng.integers(2))]
    if style == 'novel':
        return palette[0],palette[4+int(rng.integers(2))] if split == 'test' else palette[1+int(rng.integers(3))]
    # Both roles can change; the majority color identifies background locally.
    bg = palette[int(rng.integers(4))]
    fg = palette[4+int(rng.integers(4))]
    return bg,fg


def _make_pair(rng, count, regime, colors, mode, relation):
    bg,fg = colors
    for _ in range(100):
        h,w = _sample_shape(rng,count,regime)
        sizes = [_sample_size(rng,h,w,count) for _ in range(count)]
        intended,rects = _place_rectangles(rng,h,w,sizes,relation)
        if intended is None or intended.sum()*3 >= h*w:
            continue
        noisy = _add_noise(rng,intended,mode)
        if noisy is None or noisy.sum()*2 >= h*w:
            continue
        inp = np.full((h,w),bg,dtype=np.int8)
        out = inp.copy()
        inp[noisy] = fg
        out[intended] = fg
        return {'input':inp,'output':out}
    raise RuntimeError('could not construct a valid pair')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count,test_count = _resolve_counts(seed,rng,num_train,num_test)
    palette = tuple(map(int,rng.permutation(10)))
    style = ('stable','reused','novel','both_vary')[int(rng.integers(4))]
    puzzle = {'train':[],'test':[]}
    seen = set()
    for split,total in (('train',train_count),('test',test_count)):
        for index in range(total):
            pair_rng = np.random.default_rng(int(rng.integers(0,np.iinfo(np.int64).max)))
            if split == 'train' and index == 0:
                count,mode = 1,'mixed'
            elif split == 'train' and index == 1:
                count,mode = 2,'cluster'
            else:
                count = int(pair_rng.choice((1,2,3,4,5,6),p=(.17,.23,.23,.18,.13,.06)))
                mode = ('clean','isolated','attached','mixed','cluster')[int(pair_rng.choice(5,p=(.08,.19,.19,.30,.24)))]
            relation = ('free','corridor','aligned','diagonal')[int(pair_rng.choice(4,p=(.46,.23,.17,.14)))]
            colors = _pair_colors(pair_rng,palette,style,split,index)
            for _ in range(20):
                pair = _make_pair(pair_rng,count, 'source_train' if split == 'train' else 'target_test',colors,mode,relation)
                key = _array_key(pair['input'])
                if key not in seen and (_compact_grid(pair['input']),_compact_grid(pair['output'])) not in _SOURCE_PAIR_KEYS:
                    break
            else:
                raise RuntimeError('could not make unique pair')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    counts = set()
    sizes = set()
    shapes = set()
    palettes = set()
    modes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        counts.add((len(puzzle['train']),len(puzzle['test'])))
        for split in ('train','test'):
            for pair in puzzle[split]:
                a = _analyze_pair(pair)
                shapes.add(pair['input'].shape)
                palettes.add((a['background'],a['foreground']))
                modes.add((a['attached'],a['isolated'],a['cluster'],a['noise_count']==0))
                sizes.update((h,w) for _,_,h,w in a['rectangles'])
    assert len(counts) >= 8 and len(shapes) >= 100
    assert len(sizes) >= 30 and len(palettes) >= 20
    assert len(modes) >= 4
    for train,test in ((2,1),(10,4),(10,1)):
        a = generate_puzzle(902,train,test)
        b = generate_puzzle(902,train,test)
        assert all(np.array_equal(x['input'],y['input']) and np.array_equal(x['output'],y['output'])
                   for split in ('train','test') for x,y in zip(a[split],b[split]))
    for train,test in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4)):
        try:
            generate_puzzle(4,train,test)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted invalid counts')
