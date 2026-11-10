"""Anchored glyph weaving through a global, marker-selected seam.

Only NumPy and the standard library are used. No files are read during solving
or generation; saved experiments are the responsibility of the caller.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0,933183003,239123320,467664154,859036875,300141972,
              147249482,142360322,211397686,417077267,573626113,243764017,
              102714597,309793949,267851720,585404975,224486288,528369636,
              371926992,453427870,338243056,745167162,887063497,379638917,
              723378137,712876953,199158541,335276934,84684445,302704297,
              488880052,1841792,454564222,247639453,686967900,270918100,
              655053103,681642275,753953209,691904274,77819998,38639017)


def _check(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid size")
    _check(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _check(np.all((0 <= a) & (a <= 9)), "ARC colors required")
    return a


def _key(a):
    return tuple(a.shape), np.asarray(a, dtype=np.int8).tobytes()


def _components(mask, diagonal=False):
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    directions = ((-1,0),(1,0),(0,-1),(0,1))
    if diagonal:
        directions += ((-1,-1),(-1,1),(1,-1),(1,1))
    result = []
    while cells:
        first = min(cells)
        cells.remove(first)
        part, stack = [first], [first]
        while stack:
            r,c = stack.pop()
            for dr,dc in directions:
                q = r+dr,c+dc
                if q in cells:
                    cells.remove(q)
                    part.append(q)
                    stack.append(q)
        result.append(part)
    return result


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for p in cells:
        a[p] = True
    return a


def _geometry(cut):
    """Reject branches/loops and require exactly two four-connected regions."""
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {}
    for r,c in cells:
        neighbors = []
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                q = r+dr,c+dc
                if not (dr or dc) or q not in cells:
                    continue
                # Do not count a diagonal shortcut across an orthogonal bend.
                if dr and dc and ((r+dr,c) in cells or (r,c+dc) in cells):
                    continue
                neighbors.append(q)
        graph[r,c] = neighbors
    ends = [p for p in graph if len(graph[p]) == 1]
    _check(len(cells) >= 3 and len(ends) == 2, "cut needs two endpoints")
    _check(all(1 <= len(v) <= 2 for v in graph.values()), "branched cut")
    H,W = cut.shape
    border = lambda p: p[0] in (0,H-1) or p[1] in (0,W-1)
    _check(all(border(p) == (p in ends) for p in cells), "cut boundary contacts")
    _check(len(_components(cut, True)) == 1, "cut disconnected")
    regions = _components(~cut)
    _check(len(regions) == 2, "cut must separate two regions")
    near = cut.copy()
    for r,c in cells:
        near[max(0,r-1):r+2,max(0,c-1):c+2] = True
    masks = [_mask(cut.shape, region) for region in regions]
    sites = [[tuple(map(int,p)) for p in np.argwhere(region & ~near)]
             for region in masks]
    _check(all(sites), "both regions need a separated marker site")
    return masks, sites


def _decode(input_grid):
    a = _grid(input_grid)
    F = int(a[0,0])
    _check(np.all(a[[0,-1],:] == F) and np.all(a[:,[0,-1]] == F), "outer frame")
    horizontal = [r for r in range(1,a.shape[0]-1) if np.all(a[r] == F)]
    _check(len(horizontal) == 1, "one header separator required")
    sep = horizontal[0]
    h = sep-1
    _check(h >= 2, "swatch height")
    dividers = [c for c in range(1,a.shape[1]-1) if np.all(a[1:sep,c] == F)]
    _check(bool(dividers), "swatch divider")
    w = dividers[0]-1
    _check(w >= 2 and a.shape[1] >= 2*w+3, "swatch width")
    left = a[1:sep,1:w+1]
    right = a[1:sep,-w-1:-1]
    _check(np.all(a[1:sep,w+1:-w-1] == F), "header spacer")
    lv,rv = set(map(int,np.unique(left))),set(map(int,np.unique(right)))
    common = lv & rv
    _check(len(common) == 1, "shared swatch background")
    bg = next(iter(common))
    _check(len(lv) == len(rv) == 2, "monochrome swatches with blanks required")
    p,q = next(iter(lv-{bg})),next(iter(rv-{bg}))
    U,V = left == p,right == q
    for glyph in (U,V):
        _check(all((glyph[0].any(),glyph[-1].any(),glyph[:,0].any(),glyph[:,-1].any())),
               "swatches must have tight bounds")
    _check(not np.array_equal(U,V), "different swatch masks required")
    field = a[sep+1:-1,1:-1]
    _check(min(field.shape) >= 5, "workfield too small")
    controls = set(map(int,np.unique(field)))-{bg,p,q}
    _check(len(controls) == 1, "one control role required")
    C = next(iter(controls))
    _check(len({F,bg,p,q,C}) == 5, "five distinct semantic roles")
    control_parts = sorted(_components(field == C,True),key=len)
    _check(len(control_parts) == 2 and len(control_parts[0]) == 1,
           "cut plus isolated marker required")
    marker = control_parts[0][0]
    cut = _mask(field.shape,control_parts[1])
    regions,sites = _geometry(cut)
    side = next((i for i in range(2) if marker in sites[i]),None)
    _check(side is not None, "marker touches cut")
    points = [tuple(map(int,z)) for z in np.argwhere((field == p) | (field == q))]
    anchors = [z for z in points if field[z] == p]
    _check(len(anchors) == 1 and len(points) >= 2, "one exceptional anchor required")
    ar,ac = anchors[0]
    _check(all((r-ar)%2 == 0 and (c-ac)%2 == 0 for r,c in points), "pitch-two map")
    pr,pc = min(r for r,c in points),min(c for r,c in points)
    mh,mw = (max(r for r,c in points)-pr)//2+1,(max(c for r,c in points)-pc)//2+1
    anchor = (ar-pr)//2,(ac-pc)//2
    layout = np.zeros((mh,mw),dtype=bool)
    for r,c in points:
        layout[(r-pr)//2,(c-pc)//2] = True
    top,left_edge = ar-anchor[0]*h,ac-anchor[1]*w
    _check(0 <= top and top+mh*h <= field.shape[0] and
           0 <= left_edge and left_edge+mw*w <= field.shape[1], "expanded bounds")
    _check(not np.any(cut & ((field == p)|(field == q))), "map on cut")
    return dict(a=a,F=F,bg=bg,p=p,q=q,C=C,h=h,w=w,U=U,V=V,sep=sep,
                field=field,cut=cut,marker=marker,side=side,regions=regions,sites=sites,
                points=points,anchor=anchor,layout=layout,top=top,left=left_edge)


def _layers(d, *, pitch=None, anchor=None, solid=False, compress=False):
    """Forward construction and counterfactual layers; not used by solve."""
    h,w = d['h'],d['w']
    dh,dw = (h,w) if pitch is None else pitch
    ah,aw = d['anchor'] if anchor is None else anchor
    original_anchor = next(z for z in d['points'] if d['field'][z] == d['p'])
    top,left = original_anchor[0]-ah*dh,original_anchor[1]-aw*dw
    coords = [tuple(map(int,z)) for z in np.argwhere(d['layout'])]
    if compress:
        rows,cols = sorted({r for r,c in coords}),sorted({c for r,c in coords})
        coords = [(rows.index(r),cols.index(c)) for r,c in coords]
        ah,aw = rows.index(d['anchor'][0]),cols.index(d['anchor'][1])
        top,left = original_anchor[0]-ah*dh,original_anchor[1]-aw*dw
    A = np.full(d['field'].shape,d['bg'],dtype=np.int8)
    B = A.copy()
    for r,c in coords:
        exceptional = (r,c) == (ah,aw)
        for glyph,color,layer in ((d['U'],d['q'] if exceptional else d['p'],A),
                                  (d['V'],d['p'] if exceptional else d['q'],B)):
            for u,v in np.argwhere(np.ones_like(glyph) if solid else glyph):
                y,x = top+r*dh+int(u),left+c*dw+int(v)
                if 0 <= y < layer.shape[0] and 0 <= x < layer.shape[1]:
                    layer[y,x] = color
    return A,B


def _compose_layers(d,A,B,side=None,priority='left',tilewise=False):
    marked = d['regions'][d['side'] if side is None else side]
    if tilewise:
        marked = marked.copy()
        for r,c in np.argwhere(d['layout']):
            y,x = d['top']+int(r)*d['h'],d['left']+int(c)*d['w']
            marked[y:y+d['h'],x:x+d['w']] = marked[y,x]
        return np.where(marked,A,B).astype(np.int8)
    result = np.where(marked,A,B).astype(np.int8)
    if priority == 'left':
        result[d['cut']] = np.where(A != d['bg'],A,B)[d['cut']]
    else:
        result[d['cut']] = np.where(B != d['bg'],B,A)[d['cut']]
    return result


def _forward(d):
    return _compose_layers(d,*_layers(d))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only pixels; inverse-map each output cell into its coarse tile."""
    d = _decode(input_grid)
    out = d['a'].copy()
    target = out[d['sep']+1:-1,1:-1]
    target[:] = d['bg']
    marked = d['regions'][d['side']]
    mh,mw = d['layout'].shape
    for y in range(target.shape[0]):
        r,u = divmod(y-d['top'],d['h'])
        for x in range(target.shape[1]):
            c,v = divmod(x-d['left'],d['w'])
            if not (0 <= r < mh and 0 <= c < mw and d['layout'][r,c]):
                continue
            first,second = bool(d['U'][u,v]),bool(d['V'][u,v])
            chosen = None
            if d['cut'][y,x]:
                chosen = 0 if first else (1 if second else None)
            elif marked[y,x]:
                chosen = 0 if first else None
            else:
                chosen = 1 if second else None
            if chosen is not None:
                colors = (d['q'],d['p']) if (r,c) == d['anchor'] else (d['p'],d['q'])
                target[y,x] = colors[chosen]
    return out


def _flags(d):
    A,B = _layers(d)
    am,bm = A != d['bg'],B != d['bg']
    cut = d['cut']
    marked,other = d['regions'][d['side']],d['regions'][1-d['side']]
    truth = _compose_layers(d,A,B)
    flags = set()
    for label,condition in (
        ('selected_left',marked & am),('selected_right',other & bm),
        ('suppress_right',marked & ~am & bm),('suppress_left',other & am & ~bm),
        ('seam_collision',cut & am & bm),('seam_right_only',cut & ~am & bm),
        ('seam_left_only',cut & am & ~bm),('seam_blank',cut & ~am & ~bm)):
        if np.any(condition):
            flags.add(label)
    if not np.array_equal(truth,_compose_layers(d,A,B,side=1-d['side'])):
        flags.add('marker_effect')
    if not np.array_equal(truth,_compose_layers(d,A,B,tilewise=True)):
        flags.add('pixelwise')
    ar,ac = d['anchor']
    y,x = d['top']+ar*d['h'],d['left']+ac*d['w']
    if np.any(truth[y:y+d['h'],x:x+d['w']] != d['bg']):
        flags.add('inversion')
    hybrid = False
    anchor_collision = False
    for r,c in np.argwhere(d['layout']):
        y,x = d['top']+int(r)*d['h'],d['left']+int(c)*d['w']
        sl = np.s_[y:y+d['h'],x:x+d['w']]
        if np.any(marked[sl] & am[sl]) and np.any(other[sl] & bm[sl]):
            hybrid = True
        if (int(r),int(c)) == d['anchor']:
            anchor_collision = bool(np.any(cut[sl] & am[sl] & bm[sl]))
    if hybrid:
        flags.add('hybrid')
    if anchor_collision:
        flags.add('anchor_collision')
    return flags


_REQUIRED = {'selected_left','selected_right','suppress_right','suppress_left',
             'seam_collision','seam_right_only','seam_left_only','seam_blank',
             'marker_effect','pixelwise','inversion','hybrid','anchor_collision'}


def _line(p,q):
    r,c = p
    r1,c1 = q
    dr,dc = abs(r1-r),abs(c1-c)
    sr,sc = (1 if r < r1 else -1),(1 if c < c1 else -1)
    error = dc-dr
    points = []
    while True:
        points.append((r,c))
        if (r,c) == (r1,c1):
            return points
        twice = error*2
        if twice > -dr:
            error -= dr
            c += sc
        if twice < dc:
            error += dc
            r += sr


def _cut(rng,H,W,family):
    a = np.zeros((H,W),dtype=bool)
    if family == 'horizontal':
        a[int(rng.integers(2,H-2)),:] = True
    elif family == 'vertical':
        a[:,int(rng.integers(2,W-2))] = True
    elif family == 'diagonal':
        c0,c1 = sorted(map(int,rng.choice(np.arange(1,W-1),2,replace=False)))
        if rng.random() < .5:
            c0,c1 = c1,c0
        for p in _line((0,c0),(H-1,c1)):
            a[p] = True
        if rng.random() < .5:
            a = a.T.copy()
            # Transpose is allowed only when it keeps the intended field size.
            if H != W:
                a = a.T.copy()
    elif family == 'elbow':
        r,c = int(rng.integers(2,H-2)),int(rng.integers(2,W-2))
        a[:r+1,c] = True
        a[r,c:] = True
    else:
        rows = sorted(map(int,rng.choice(np.arange(2,H-2,2),
                                         min(int(rng.integers(1,3)),len(range(2,H-2,2))),
                                         replace=False)))
        col,last = int(rng.integers(2,W-2)),0
        for r in rows:
            nxt = _pick(rng,[c for c in range(2,W-2) if c != col])
            a[last:r+1,col] = True
            a[r,min(col,nxt):max(col,nxt)+1] = True
            col,last = nxt,r
        a[last:,col] = True
    if rng.random() < .5:
        a = a[::-1].copy()
    if rng.random() < .5:
        a = a[:,::-1].copy()
    return a


def _asymmetric(glyph):
    return all(not np.array_equal(glyph,t) for t in
               (glyph[::-1],glyph[:,::-1],glyph[::-1,::-1]))


def _glyph(rng,h,w):
    mode = int(rng.integers(7))
    a = np.zeros((h,w),dtype=bool)
    if mode == 0:
        a[:] = True
        a[_pick(rng,[(0,0),(0,w-1),(h-1,0),(h-1,w-1)])] = False
    elif mode == 1:
        a[[0,-1],:] = True
        a[:,[0,-1]] = True
        if np.all(a):
            a[0,0] = False
    elif mode == 2:
        a[int(rng.integers(h)),:] = True
        a[:,int(rng.integers(w))] = True
    elif mode == 3:
        for p in _line((0,0),(h-1,w-1)) + _line((0,w-1),(h-1,0)):
            a[p] = True
        if a.all():
            a[0,0] = False
    elif mode == 4:
        r,c = 0,0
        a[r,c] = True
        steps = [0]*(h-1)+[1]*(w-1)
        rng.shuffle(steps)
        for axis in steps:
            r += axis == 0
            c += axis == 1
            a[r,c] = True
        a |= rng.random((h,w)) < float(rng.uniform(0,.4))
    else:
        for _ in range(32):
            a = rng.random((h,w)) < float(rng.uniform(.25,.8))
            if all((a[0].any(),a[-1].any(),a[:,0].any(),a[:,-1].any())) and not a.all():
                break
        else:
            a[:] = True
            a[0,0] = False
    if a.all():
        a[0,0] = False
    return a


def _layout(rng,m,n,gaps):
    if gaps:
        rows = [r for r in range(m) if r != 2]
        cols = [c for c in range(n) if c != 2]
        a = np.zeros((m,n),dtype=bool)
        for r in rows:
            for c in cols:
                a[r,c] = rng.random() < float(rng.uniform(.3,.9))
        for r in rows:
            a[r,_pick(rng,cols)] = True
        for c in cols:
            a[_pick(rng,rows),c] = True
        a[0,0] = a[0,1] = a[1,0] = True
        return a
    a = np.zeros((m,n),dtype=bool)
    r,c = 0,0
    a[r,c] = True
    steps = [0]*(m-1)+[1]*(n-1)
    rng.shuffle(steps)
    for axis in steps:
        r += axis == 0
        c += axis == 1
        a[r,c] = True
    target = int(rng.integers(int(a.sum()),m*n+1))
    while a.sum() < target:
        frontier = sorted({(r+dr,c+dc) for r,c in np.argwhere(a)
                           for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
                           if 0 <= r+dr < m and 0 <= c+dc < n and not a[r+dr,c+dc]})
        a[_pick(rng,frontier)] = True
    if rng.random() < .5:
        a = a[::-1].copy()
    if rng.random() < .5:
        a = a[:,::-1].copy()
    return a


def _new_pair(rng,params,split,index,previous=None,missing=frozenset()):
    teaching = split == 'train' and index < 2
    gaps = params['gaps'] and ((split == 'train' and index == 1) or
                              (not teaching and rng.random() < .45))
    for attempt in range(900):
        sizes = [(h,w) for h in range(2,6) for w in range(2,6)
                 if not teaching or (index == 0 and h != w) or
                 (index == 1 and h != previous['h'] and w != previous['w'])]
        if split == 'test' and params['shift'] == 'extent':
            sizes = [(h,w) for h,w in sizes if max(h,w) <= 3]
        h,w = _pick(rng,sizes)
        maxm = min(6,(27-h)//h)
        maxn = min(6,28//w)
        lower = 4 if gaps else 2 if teaching else 1
        dims = [(m,n) for m in range(lower,maxm+1) for n in range(lower,maxn+1)
                if m*n >= 2]
        if params['shift'] == 'extent':
            dims = [(m,n) for m,n in dims if
                    (max(m,n) >= 5 if split == 'test' else max(m,n) <= 4)]
        elif params['shift'] == 'aspect':
            sign = params['aspect']*(1 if split == 'train' else -1)
            dims = [(m,n) for m,n in dims if np.sign(n-m) == sign]
        if not dims:
            continue
        m,n = _pick(rng,dims)
        layout = _layout(rng,m,n,gaps)
        anchors = [tuple(map(int,z)) for z in np.argwhere(layout)
                   if not teaching or
                   (index == 0 and tuple(z) != ((m-1)//2,(n-1)//2)) or
                   (index == 1 and z[0] != previous['anchor'][0] and
                    z[1] != previous['anchor'][1])]
        if not anchors:
            continue
        anchor = _pick(rng,anchors)
        U,V = _glyph(rng,h,w),_glyph(rng,h,w)
        if np.array_equal(U,V) or (teaching and index == 0 and not _asymmetric(U)):
            continue
        H = int(rng.integers(max(9,m*h),28-h))
        W = int(rng.integers(max(9,n*w,2*w+1),29))
        top,left = int(rng.integers(H-m*h+1)),int(rng.integers(W-n*w+1))
        ar,ac = top+anchor[0]*h,left+anchor[1]*w
        point_colors = {(ar+2*(int(r)-anchor[0]),ac+2*(int(c)-anchor[1])):
                        ((int(r),int(c)) == anchor) for r,c in np.argwhere(layout)}
        if params['shift'] == 'bend':
            families = ('horizontal','vertical') if split == 'train' else ('elbow','stepped')
        else:
            families = ('horizontal','vertical','diagonal','elbow','stepped')
        family = _pick(rng,families)
        # The first two bend demonstrations show both axes, and independent sides.
        if teaching and params['shift'] == 'bend':
            family = ('horizontal','vertical')[(index+params['axis'])%2]
        cut = _cut(rng,H,W,family)
        if any(cut[z] for z in point_colors):
            continue
        try:
            regions,sites = _geometry(cut)
        except AssertionError:
            continue
        side = (params['side']+index)%2 if teaching else int(rng.integers(2))
        available = [z for z in sites[side] if z not in point_colors]
        if not available:
            continue
        marker = _pick(rng,available)
        F,bg,C,p,q = (params['palette'] if params['fixed_palette'] else
                      tuple(map(int,rng.choice(10,5,replace=False))))
        inp = np.full((h+H+3,W+2),F,dtype=np.int8)
        inp[1:h+1,1:w+1] = np.where(U,p,bg)
        inp[1:h+1,-w-1:-1] = np.where(V,q,bg)
        field = inp[h+2:-1,1:-1]
        field[:] = bg
        field[cut] = C
        field[marker] = C
        for z,exceptional in point_colors.items():
            field[z] = p if exceptional else q
        d = _decode(inp)
        flags = _flags(d)
        essential = {'selected_left','selected_right','marker_effect','inversion'}
        if not essential <= flags or not missing <= flags:
            continue
        if teaching and not {'pixelwise','hybrid'} <= flags:
            continue
        out = inp.copy()
        out[h+2:-1,1:-1] = _forward(d)
        return {'input':inp,'output':out}
    raise RuntimeError('bounded construction exhausted; no exceptional rule fallback')


def _counts(num_train,num_test):
    for v,lo,hi in ((num_train,2,10),(num_test,1,4)):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or not lo <= int(v) <= hi:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train),int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,nq = _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    params = dict(shift=_pick(rng,('bend','extent','aspect')),aspect=_pick(rng,(-1,1)),
                  gaps=bool(rng.random() < .45),fixed_palette=bool(rng.random() < .25),
                  palette=tuple(map(int,rng.choice(10,5,replace=False))),
                  side=int(rng.integers(2)),axis=int(rng.integers(2)))
    puzzle,seen,covered = {'train':[],'test':[]},set(),set()
    previous = None
    for split,count in (('train',nt),('test',nq)):
        for i in range(count):
            missing = _REQUIRED-covered if split == 'train' and i == nt-1 else set()
            for _ in range(20):
                pair = _new_pair(rng,params,split,i,previous,missing)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded uniqueness construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            d = _decode(pair['input'])
            if split == 'train':
                covered |= _flags(d)
                if i == 0:
                    previous = d
    validate_generated_puzzle(puzzle)
    return puzzle


def _counterfactuals(d):
    A,B = _layers(d)
    bg = d['bg']
    center = ((d['layout'].shape[0]-1)//2,(d['layout'].shape[1]-1)//2)
    truth = _compose_layers(d,A,B)
    no_invert = truth.copy()
    ar,ac = d['anchor']
    y,x = d['top']+ar*d['h'],d['left']+ac*d['w']
    tile = no_invert[y:y+d['h'],x:x+d['w']]
    old = tile.copy()
    tile[old == d['p']] = d['q']
    tile[old == d['q']] = d['p']
    return {
        'left_everywhere':A,'right_everywhere':B,
        'union_left_priority':np.where(A != bg,A,B),
        'ignore_marker_first_region':_compose_layers(d,A,B,side=0),
        'reverse_marker':_compose_layers(d,A,B,side=1-d['side']),
        'reverse_seam_priority':_compose_layers(d,A,B,priority='right'),
        'select_whole_tiles':_compose_layers(d,A,B,tilewise=True),
        'no_anchor_inversion':no_invert,
        'raw_pitch_two':_compose_layers(d,*_layers(d,pitch=(2,2))),
        'fixed_pitch_three':_compose_layers(d,*_layers(d,pitch=(3,3))),
        'swapped_pitches':_compose_layers(d,*_layers(d,pitch=(d['w'],d['h']))),
        'geometric_center_anchor':_compose_layers(d,*_layers(d,anchor=center)),
        'solidify_masks':_compose_layers(d,*_layers(d,solid=True)),
        'compress_empty_bands':_compose_layers(d,*_layers(d,compress=True))}


def validate_generated_puzzle(puzzle) -> None:
    _check(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, "puzzle keys")
    _check(all(isinstance(puzzle[s],list) for s in puzzle), "split lists")
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen,coverage,training = set(),set(),[]
    survivors = None
    for split in ('train','test'):
        for pair in puzzle[split]:
            _check(isinstance(pair,dict) and set(pair) == {'input','output'}, "pair keys")
            inp,out = _grid(pair['input']),_grid(pair['output'])
            _check(inp.shape == out.shape, "canvas preserved")
            key = _key(inp)
            _check(key not in seen, "duplicate input")
            seen.add(key)
            d = _decode(inp)
            _check(np.array_equal(solve(inp),out), "input-only solve disagreement")
            target = out[d['sep']+1:-1,1:-1]
            _check(np.array_equal(_forward(d),target), "forward/inverse disagreement")
            _check(np.all(target != d['C']) and np.all(target != d['F']), "controls erased")
            _check(set(map(int,np.unique(target))) <= {d['bg'],d['p'],d['q']}, "output roles")
            for r,c in np.argwhere(~d['layout']):
                y,x = d['top']+int(r)*d['h'],d['left']+int(c)*d['w']
                _check(np.all(target[y:y+d['h'],x:x+d['w']] == d['bg']), "empty macrocell")
            flags = _flags(d)
            _check({'selected_left','selected_right','marker_effect','inversion'} <= flags,
                   "either parent became ineffective")
            alternatives = _counterfactuals(d)
            matching = {k for k,v in alternatives.items() if np.array_equal(v,target)}
            if split == 'train':
                training.append(d)
                coverage |= flags
                survivors = matching if survivors is None else survivors & matching
            else:
                _check(all(np.array_equal(alternatives[k],target) for k in survivors),
                       "test distinguishes a shortcut left ambiguous by training")
    _check(_REQUIRED <= coverage, "training omits a rule witness")
    _check({d['side'] for d in training} == {0,1}, "both marker sides must be demonstrated")
    _check(len({d['anchor'][0] for d in training}) >= 2 and
           len({d['anchor'][1] for d in training}) >= 2, "training anchor coordinates")
    _check(len({d['h'] for d in training}) >= 2 and len({d['w'] for d in training}) >= 2,
           "training tile dimensions")
    _check(any(d['h'] != d['w'] and _asymmetric(d['U']) for d in training),
           "asymmetric rectangular demonstration")
    _check(not (survivors-{'compress_empty_bands'}), "unexcluded geometric shortcut")


def diversity_metrics(seeds=HOST_SEEDS):
    """Streaming, decoded measurements; stores signatures, never all puzzles."""
    shapes,masks,maps,anchors,palettes,signatures = set(),set(),set(),set(),set(),set()
    stats = dict(seeds=0,pairs=0,hybrid_pairs=0,anchor_collision_pairs=0,
                 gapped_maps=0,line_maps=0,empty_macrocell_pairs=0,
                 marker_sides=[0,0],min_sites=999,max_sites=0)
    for seed in seeds:
        puzzle = generate_puzzle(int(seed))
        digest = hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                for role in ('input','output'):
                    digest.update(str(pair[role].shape).encode())
                    digest.update(pair[role].tobytes())
                flags = _flags(d)
                shapes.add(pair['input'].shape)
                masks.add((d['h'],d['w'],d['U'].tobytes(),d['V'].tobytes()))
                maps.add((d['layout'].shape,d['layout'].tobytes()))
                anchors.add(d['anchor'])
                palettes.add((d['F'],d['bg'],d['C'],d['p'],d['q']))
                gaps = (~d['layout'].any(axis=0)).any() or (~d['layout'].any(axis=1)).any()
                count = int(d['layout'].sum())
                stats['pairs'] += 1
                stats['hybrid_pairs'] += int('hybrid' in flags)
                stats['anchor_collision_pairs'] += int('anchor_collision' in flags)
                stats['gapped_maps'] += int(gaps)
                stats['line_maps'] += int(min(d['layout'].shape) == 1)
                stats['empty_macrocell_pairs'] += int(not d['layout'].all())
                stats['marker_sides'][d['side']] += 1
                stats['min_sites'] = min(stats['min_sites'],count)
                stats['max_sites'] = max(stats['max_sites'],count)
        signatures.add(digest.hexdigest())
        stats['seeds'] += 1
    stats.update(distinct_puzzles=len(signatures),canvas_shapes=len(shapes),
                 glyph_pairs=len(masks),map_masks=len(maps),anchors=len(anchors),
                 role_palettes=len(palettes))
    return stats


def self_test() -> None:
    # Hand-calculated fixture: neither random construction nor _forward defines
    # its expected pixels. The anchor tile crosses the vertical seam at column 6.
    fixture = np.full((14,13),8,dtype=np.int8)
    fixture[1:3,1:4] = [[2,2,0],[0,2,2]]
    fixture[1:3,-4:-1] = [[0,4,4],[4,0,4]]
    work = fixture[4:-1,1:-1]
    work[:] = 0
    work[:,6] = 6
    work[0,0] = 6
    for site in ((3,3),(5,5),(5,7)):
        work[site] = 4
    work[3,5] = 2
    expected_rows = ('00000000000','00000000000','00000000000',
                     '00220442000','00022042000','00000224044',
                     '00000024404','00000000000','00000000000')
    expected = fixture.copy()
    expected[4:-1,1:-1] = [[int(c) for c in row] for row in expected_rows]
    _check(np.array_equal(solve(fixture),expected), "hand-calculated splice fixture")
    swapped_fixture = fixture.copy()
    swapped_fixture[4+3,1+5],swapped_fixture[4+5,1+7] = 4,2
    shifted = _decode(swapped_fixture)
    _check(shifted['left'] == 1 and _decode(fixture)['left'] == 2,
           "exceptional point must change geometric registration")
    _check(not np.array_equal(solve(swapped_fixture),expected), "anchor swap ineffective")
    _check(np.array_equal(solve(swapped_fixture)[4:-1,1:-1],_forward(shifted)),
           "valid anchor swap forward/inverse agreement")
    metrics = diversity_metrics()
    _check(metrics['seeds'] == metrics['distinct_puzzles'] == 42, "seed diversity")
    _check(metrics['canvas_shapes'] >= 35 and metrics['glyph_pairs'] >= 100 and
           metrics['map_masks'] >= 70 and metrics['anchors'] >= 12, "structural diversity")
    _check(metrics['gapped_maps'] >= 10 and metrics['line_maps'] >= 1, "map topology diversity")
    _check(metrics['hybrid_pairs'] >= 84 and metrics['anchor_collision_pairs'] >= 42,
           "cross-scale interaction coverage")
    _check(metrics['min_sites'] <= 3 and metrics['max_sites'] >= 15, "count diversity")
    first,again = generate_puzzle(0),generate_puzzle(0)
    _check(all(np.array_equal(a[k],b[k]) for s in first for a,b in
               zip(first[s],again[s]) for k in a), "determinism")
    # Counterfactual edits use valid visible input conventions, no latent metadata.
    for pair in first['train'][:2]:
        d = _decode(pair['input'])
        changed = pair['input'].copy()
        field = changed[d['sep']+1:-1,1:-1]
        field[d['marker']] = d['bg']
        new_marker = next(z for z in d['sites'][1-d['side']] if z not in d['points'])
        field[new_marker] = d['C']
        answer = solve(changed)[d['sep']+1:-1,1:-1]
        A,B = _layers(d)
        _check(np.array_equal(answer,_compose_layers(d,A,B,side=1-d['side'])), "marker swap")
        _check(np.array_equal(answer[d['cut']],_forward(d)[d['cut']]), "marker altered seam")
        _check(not np.array_equal(answer,_forward(d)), "marker has no effect")
        # An anchor swap may move the assembly; choose one that still fits.
        swapped = False
        for site in d['points']:
            if d['field'][site] != d['q']:
                continue
            candidate = pair['input'].copy()
            f = candidate[d['sep']+1:-1,1:-1]
            old_anchor = next(z for z in d['points'] if d['field'][z] == d['p'])
            f[old_anchor],f[site] = d['q'],d['p']
            try:
                decoded = _decode(candidate)
            except AssertionError:
                continue
            predicted = solve(candidate)[d['sep']+1:-1,1:-1]
            _check(np.array_equal(predicted,_forward(decoded)), "anchor edit reconstruction")
            if not np.array_equal(predicted,_forward(d)):
                swapped = True
                break
        # Fit constraints can prevent a swap in this particular canvas; the
        # first two demonstrations still vary both anchor coordinates.
        if swapped:
            _check(not np.array_equal(predicted,_forward(d)), "anchor ignored")
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            generate_puzzle(0,nt,nq)
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s:[{k:np.array(v,copy=True) for k,v in pair.items()} for pair in first[s]] for s in first}
    bad['train'][0]['output'][-2,1] = (int(bad['train'][0]['output'][-2,1])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')


if __name__ == '__main__':
    self_test()
