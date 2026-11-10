import numpy as np


_SOURCE_INPUTS = (
    ('0000100100', '0000100100', '0000100111', '0000100000', '0000100000', '0000111111', '0000000000', '0000000000', '0000000000', '0000000000'),
    ('0000000000', '0000000000', '0000000000', '0000000000', '8888880000', '0000080000', '8888080000', '0008080000', '8808080000', '0808080000'),
    ('0202020200', '0202220200', '0200000200', '0222222200', '0000000000', '0000000000', '0000000000', '0000000000', '0000000000', '0000000000'),
    ('4444440000', '0000040000', '0000040000', '4440040000', '0040040000', '0040040000', '4440040000', '0000040000', '0000040000', '4444440000'),
)


def _count(value, low, high, name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(name)
    return int(value)


def _outline(h, w, top, left, side):
    bottom, right = top + side - 1, left + side - 1
    mask = np.zeros((h, w), dtype=bool)
    x0, x1 = max(0, left), min(w, right + 1)
    y0, y1 = max(0, top), min(h, bottom + 1)
    if x0 < x1:
        if 0 <= top < h:
            mask[top, x0:x1] = True
        if 0 <= bottom < h:
            mask[bottom, x0:x1] = True
    if y0 < y1:
        if 0 <= left < w:
            mask[y0:y1, left] = True
        if 0 <= right < w:
            mask[y0:y1, right] = True
    return mask


def _family(h, w, top, left, side, step):
    masks = []
    for k in range(40):
        mask = _outline(h, w, top - step*k, left - step*k, side + 2*step*k)
        if not mask.any():
            break
        masks.append(mask)
    return masks


def _corners(h, w, top, left, side):
    bottom, right = top + side - 1, left + side - 1
    return sum(0 <= y < h and 0 <= x < w for y, x in ((top,left),(top,right),(bottom,left),(bottom,right)))


def _signature(mask):
    return (mask.shape, mask.tobytes())


def _interpret(mask):
    """Enumerate square families with an observed in-grid inner corner."""
    h, w = mask.shape
    ys, xs = np.where(mask)
    anchors = []
    for y, x in zip(ys, xs):
        horizontal = (x > 0 and mask[y,x-1]) or (x+1 < w and mask[y,x+1])
        vertical = (y > 0 and mask[y-1,x]) or (y+1 < h and mask[y+1,x])
        if horizontal and vertical:
            anchors.append((int(y), int(x)))
    seen = set()
    matches = []
    for y, x in anchors:
        for side in range(2, max(31, 2*max(h,w)+1)):
            for top in (y, y-side+1):
                for left in (x, x-side+1):
                    key = (top, left, side)
                    if key in seen:
                        continue
                    seen.add(key)
                    first = _outline(h,w,top,left,side)
                    if not first.any() or np.any(first & ~mask):
                        continue
                    for step in range(2, 9):
                        observed = first.copy()
                        for shown in range(2, 6):
                            kth = _outline(h,w,top-step*(shown-1),left-step*(shown-1),side+2*step*(shown-1))
                            if not kth.any():
                                break
                            observed |= kth
                            if np.any(observed & ~mask):
                                break
                            if np.array_equal(observed,mask):
                                full = observed.copy()
                                missing = 0
                                for later in range(shown, 40):
                                    outer = _outline(h,w,top-step*later,left-step*later,side+2*step*later)
                                    if not outer.any():
                                        break
                                    full |= outer
                                    missing += 1
                                if missing:
                                    matches.append(((top,left,side,step,shown,missing),full))
    return matches


def _sample_geometry(rng, category):
    h = int(rng.integers(9, 22))
    w = int(rng.integers(9, 22))
    step = int(rng.integers(2, 6))
    side = int(rng.integers(2, 12))
    if category == 'complete':
        side = min(side,h-2,w-2)
        top = int(rng.integers(1,h-side))
        left = int(rng.integers(1,w-side))
    else:
        y = int(rng.integers(0,h))
        x = int(rng.integers(0,w))
        top = y if rng.integers(2) else y-side+1
        left = x if rng.integers(2) else x-side+1
    masks = _family(h,w,top,left,side,step)
    if len(masks) < 3 or _corners(h,w,top,left,side) == 0 or _corners(h,w,top-step,left-step,side+2*step) == 0:
        return None
    if category == 'clipped' and _corners(h,w,top,left,side) == 4:
        return None
    shown = int(rng.integers(2,min(5,len(masks))))
    inp = np.logical_or.reduce(masks[:shown])
    out = np.logical_or.reduce(masks)
    if np.array_equal(inp,out) or out.mean() > 0.65 or inp.mean() >= 0.50:
        return None
    if sum(np.any(m & ~inp) for m in masks[shown:]) < 1:
        return None
    return inp,out


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _count(num_train,2,10,'num_train')
    num_test = _count(num_test,1,4,'num_test')
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if num_test is None:
        num_test = int(rng.integers(1,min(4,(num_train or 10)-1)+1))
    if num_train is None:
        num_train = int(rng.integers(max(2,num_test+1),11))
    filler = int(rng.integers(0,10))
    colors = [c for c in range(10) if c != filler]
    rng.shuffle(colors)
    foregrounds = [int(colors[int(rng.integers(0,len(colors)))]) for _ in range(num_train+num_test)]
    if len(set(foregrounds[:num_train])) == 1:
        foregrounds[1] = next(c for c in colors if c != foregrounds[0])
    puzzle = {'train':[],'test':[]}
    used = set()
    for i in range(num_train+num_test):
        section = 'train' if i < num_train else 'test'
        category = ('complete' if i == 0 else 'clipped' if i == 1 else 'any')
        fg = foregrounds[i]
        for attempt in range(1000):
            sample = _sample_geometry(rng,category)
            if sample is None:
                continue
            foreground_mask, complete_mask = sample
            if _signature(foreground_mask) in used:
                continue
            interpretations = _interpret(foreground_mask)
            if not interpretations or any(not np.array_equal(complete_mask, answer) for _,answer in interpretations):
                continue
            background = int(colors[int(rng.integers(0,len(colors)))])
            if background == fg:
                continue
            inp = np.where(foreground_mask,fg,background).astype(np.int8)
            out = np.where(complete_mask,fg,filler).astype(np.int8)
            if inp.shape == (10,10) and tuple(''.join(map(str,row)) for row in inp) in _SOURCE_INPUTS:
                continue
            used.add(_signature(foreground_mask))
            puzzle[section].append({'input':inp,'output':out})
            break
        else:
            raise RuntimeError(f'could not sample unique {section} pair {i}')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert isinstance(puzzle['train'],list) and isinstance(puzzle['test'],list)
    nt,nv = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv
    filler = None
    raw_inputs,raw_pairs,shapes = set(),set(),set()
    train_foregrounds = set()
    train_clipping = set()
    for section in ('train','test'):
        for pair in puzzle[section]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            inp,out = pair['input'],pair['output']
            for grid in (inp,out):
                assert isinstance(grid,np.ndarray) and grid.ndim==2 and np.issubdtype(grid.dtype,np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0)&(grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            ins,outs = set(map(int,np.unique(inp))),set(map(int,np.unique(out)))
            assert len(ins)==2 and len(outs)==2 and len(ins & outs)==1
            fg = next(iter(ins & outs))
            bg = next(iter(ins-{fg}))
            fill = next(iter(outs-{fg}))
            assert len({fg,bg,fill})==3
            if filler is None:
                filler=fill
            assert fill==filler
            if section=='train':
                train_foregrounds.add(fg)
            imask = inp==fg
            omask = out==fg
            assert int(imask.sum()) < int((~imask).sum())
            assert np.all(omask[imask]) and np.any(omask & ~imask) and np.any(~omask)
            assert np.all(inp[~imask]==bg) and np.all(out[~omask]==filler)
            in_sig, out_sig = _signature(inp),_signature(out)
            assert in_sig not in raw_inputs and (in_sig,out_sig) not in raw_pairs
            raw_inputs.add(in_sig)
            raw_pairs.add((in_sig,out_sig))
            mask_sig = _signature(imask)
            assert mask_sig not in shapes
            shapes.add(mask_sig)
            if inp.shape == (10,10):
                assert tuple(''.join(map(str,row)) for row in inp) not in _SOURCE_INPUTS
            matches = _interpret(imask)
            assert matches and all(np.array_equal(omask,answer) for _,answer in matches)
            assert all(params[4]>=2 and params[5]>=1 for params,_ in matches)
            if section == 'train':
                top,left,side,_,_,_ = matches[0][0]
                train_clipping.add(_corners(*inp.shape,top,left,side)==4)
    assert len(train_foregrounds)>=2
    assert train_clipping=={False,True}


def _predict_from_training(train_pairs, test_input):
    """Use only demonstrations and the unsolved input to infer its output."""
    fillers = set()
    for pair in train_pairs:
        inp, out = pair['input'], pair['output']
        colors, counts = np.unique(inp, return_counts=True)
        assert len(colors) == 2 and counts[0] != counts[1]
        fg = int(colors[np.argmin(counts)])
        answer_colors = set(map(int, np.unique(out)))
        fillers.add(next(iter(answer_colors - {fg})))
    assert len(fillers) == 1
    filler = next(iter(fillers))
    colors, counts = np.unique(test_input, return_counts=True)
    assert len(colors) == 2 and counts[0] != counts[1]
    fg = int(colors[np.argmin(counts)])
    matches = _interpret(test_input == fg)
    assert matches
    answer = matches[0][1]
    assert all(np.array_equal(answer, candidate) for _, candidate in matches)
    return np.where(answer, fg, filler).astype(np.int8)


def _same(a,b):
    if len(a['train'])!=len(b['train']) or len(a['test'])!=len(b['test']):
        return False
    return all(np.array_equal(x[k],y[k]) for s in ('train','test') for x,y in zip(a[s],b[s]) for k in ('input','output'))


def self_test() -> None:
    puzzles=[generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles})>=4
    assert len({len(p['test']) for p in puzzles})>=3
    assert _same(puzzles[0],generate_puzzle(0))
    assert not _same(puzzles[0],puzzles[1])
    features=set()
    fillers=set()
    backgrounds=set()
    shapes=set()
    completions=set()
    shown_counts=set()
    phases=set()
    first_corner_counts=set()
    final_band_types=set()
    clipped_directions=set()
    repeated_test_foreground_count=0
    new_test_foreground_count=0
    changing_background_count=0
    for p in puzzles:
        validate_generated_puzzle(p)
        local_backgrounds=set()
        train_foregrounds=set()
        test_foregrounds=set()
        for pair in p['test']:
            assert np.array_equal(
                _predict_from_training(p['train'], pair['input']), pair['output']
            )
        for s in ('train','test'):
            for pair in p[s]:
                inp,out=pair['input'],pair['output']
                fg=next(iter(set(map(int,np.unique(inp)))&set(map(int,np.unique(out)))))
                bg=next(iter(set(map(int,np.unique(inp)))-{fg}))
                fill=next(iter(set(map(int,np.unique(out)))-{fg}))
                local_backgrounds.add(bg)
                (train_foregrounds if s=='train' else test_foregrounds).add(fg)
                backgrounds.add(bg)
                fillers.add(fill)
                shapes.add(inp.shape)
                params=_interpret(inp==fg)[0][0]
                top,left,side,step,shown,missing=params
                features.add((step,side%2,shown))
                completions.add(missing)
                shown_counts.add(shown)
                phases.add(side%2)
                corner_count=_corners(*inp.shape,top,left,side)
                first_corner_counts.add(corner_count)
                if corner_count==2:
                    corners=[(y,x) for y,x in ((top,left),(top,left+side-1),(top+side-1,left),(top+side-1,left+side-1)) if 0<=y<inp.shape[0] and 0<=x<inp.shape[1]]
                    if corners[0][0]==corners[1][0]:
                        clipped_directions.add('top' if corners[0][0]==top else 'bottom')
                    else:
                        clipped_directions.add('left' if corners[0][1]==left else 'right')
                outer=_family(*inp.shape,top,left,side,step)[-1]
                ys,xs=np.where(outer)
                final_band_types.add('horizontal' if len(set(ys))==1 else 'vertical' if len(set(xs))==1 else 'other')
                assert np.array_equal(out==fg,_interpret(inp==fg)[0][1])
        changing_background_count += len(local_backgrounds)>1
        repeated_test_foreground_count += bool(train_foregrounds & test_foregrounds)
        new_test_foreground_count += bool(test_foregrounds-train_foregrounds)
        if len(p['train'])+len(p['test'])>=5:
            assert len(local_backgrounds)>=2
    assert len(fillers)>=7 and len(backgrounds)>=7
    assert len(shapes)>=30 and len(features)>=20
    assert len(completions)>=3
    assert shown_counts=={2,3,4} and phases=={0,1}
    assert first_corner_counts=={1,2,4}
    assert final_band_types=={'horizontal','vertical','other'}
    assert clipped_directions=={'top','bottom','left','right'}
    assert changing_background_count>=30
    assert repeated_test_foreground_count>=15 and new_test_foreground_count>=15
    for seed,nt,nv in ((81,2,1),(82,10,1),(83,10,4),(84,None,1),(85,None,4),(86,2,None),(87,10,None)):
        p=generate_puzzle(seed,nt,nv)
        validate_generated_puzzle(p)
        assert (nt is None or len(p['train'])==nt) and (nv is None or len(p['test'])==nv)
        assert _same(p,generate_puzzle(seed,nt,nv))
    for kwargs in ({'num_train':1},{'num_train':11},{'num_test':0},{'num_test':5},{'num_train':2,'num_test':2},{'num_train':3,'num_test':4}):
        try:
            generate_puzzle(99,**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)
    damaged=generate_puzzle(101,2,1)
    pair=damaged['test'][0]
    inp,out=pair['input'],pair['output']
    fg=next(iter(set(map(int,np.unique(inp)))&set(map(int,np.unique(out)))))
    y,x=np.argwhere(out!=fg)[0]
    out[y,x]=fg
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('accepted damaged contour')
