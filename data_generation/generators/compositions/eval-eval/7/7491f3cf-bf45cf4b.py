"""Contour-woven mosaics: regional compositing inside expanded stamp tiles.

Only standard-library modules and NumPy are used. Generation writes no files.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _require(np.all((a >= 0) & (a <= 9)), 'invalid color')
    return a.astype(np.int8, copy=False)


def _components(mask):
    """Four-connected components, in row-major discovery order."""
    h, w = mask.shape
    seen = np.zeros(mask.shape, bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _region_masks(cut):
    """Label horizontal open spans, joining overlaps in neighboring rows.

    This is the same four-connectivity as pixel flood fill. Region order remains
    the order of the first open cell, so marker sampling is unchanged.
    """
    open_cells = ~cut
    edges = np.diff(np.pad(open_cells,((0,0),(1,1))).astype(np.int8),axis=1)
    sr,sc = np.nonzero(edges == 1)
    _,ec = np.nonzero(edges == -1)
    rows = [[] for _ in range(cut.shape[0])]
    parents = list(range(len(sr)))
    for i,(r,left,right) in enumerate(zip(sr,sc,ec)):
        rows[int(r)].append((int(left),int(right),i))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for previous,current in zip(rows,rows[1:]):
        i = j = 0
        while i < len(previous) and j < len(current):
            left,right,p = previous[i]
            low,high,q = current[j]
            if right <= low:
                i += 1
            elif high <= left:
                j += 1
            else:
                a,b = root(p),root(q)
                parents[max(a,b)] = min(a,b)
                if right <= high:
                    i += 1
                else:
                    j += 1
    representatives = [root(i) for i in range(len(parents))]
    ordered = list(dict.fromkeys(representatives))
    _require(len(ordered) == 2, 'cut must leave exactly two regions')
    labels = np.full(cut.shape,-1,np.int8)
    indices = {r:i for i,r in enumerate(ordered)}
    for r,spans in enumerate(rows):
        for left,right,i in spans:
            labels[r,left:right] = indices[representatives[i]]
    return [labels == 0,labels == 1]


def _cut_info(cut):
    h, w = cut.shape
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {p: [q for q in ((p[0]-1,p[1]), (p[0]+1,p[1]),
                            (p[0],p[1]-1), (p[0],p[1]+1)) if q in cells] for p in cells}
    ends = sorted(p for p, q in graph.items() if len(q) == 1)
    _require(len(ends) == 2 and all(1 <= len(q) <= 2 for q in graph.values()), 'branched or cyclic cut')
    def boundary(p):
        return p[0] in (0,h-1) or p[1] in (0,w-1)
    def corner(p):
        return p[0] in (0,h-1) and p[1] in (0,w-1)
    _require(all(boundary(p) and not corner(p) for p in ends), 'invalid cut endpoints')
    _require(all(not boundary(p) for p in cells - set(ends)), 'extra boundary contact')
    path, previous, p = [], None, ends[0]
    while p is not None:
        _require(p not in path, 'cut loop')
        path.append(p)
        following = [q for q in graph[p] if q != previous]
        previous, p = p, following[0] if following else None
    _require(len(path) == len(cells), 'disconnected cut')
    regions = _region_masks(cut)
    near = cut.copy()
    for r, c in cells:
        near[max(0,r-1):r+2,max(0,c-1):c+2] = True
    sites = [[tuple(map(int, p)) for p in np.argwhere(region & ~near)] for region in regions]
    _require(all(sites), 'each side must admit a separated marker')
    return path, regions, sites


def _refine(cut, stamp_shape, marker, *, path=None):
    """Trace a one-pixel contour through tile centers; include edge extensions."""
    if path is None:
        path, _, _ = _cut_info(cut)
    sh, sw = stamp_shape
    centers = [(r*sh+sh//2, c*sw+sw//2) for r, c in path]
    fine = np.zeros((cut.shape[0]*sh,cut.shape[1]*sw), bool)
    for (r,c), (rr,cc) in zip(centers,centers[1:]):
        fine[min(r,rr):max(r,rr)+1,min(c,cc):max(c,cc)+1] = True
    for coarse, center in ((path[0],centers[0]),(path[-1],centers[-1])):
        r, c = coarse
        y, x = center
        if r == 0:
            fine[:y+1,x] = True
        elif r == cut.shape[0]-1:
            fine[y:,x] = True
        elif c == 0:
            fine[y,:x+1] = True
        else:
            fine[y,x:] = True
    regions = _region_masks(fine)
    point = (marker[0]*sh+sh//2,marker[1]*sw+sw//2)
    selected = 0 if regions[0][point] else 1
    _require(not fine[point], 'marker center is on contour')
    return fine, regions[selected], regions[1-selected]


def _frames(grid, bg):
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for group in _components(grid == color):
            yy, xx = zip(*group)
            r0, r1, c0, c1 = min(yy),max(yy),min(xx),max(xx)
            if r1-r0 < 4 or c1-c0 < 4:
                continue
            box = grid[r0:r1+1,c0:c1+1]
            if (np.all(box[[0,-1],:] == color) and np.all(box[:,[0,-1]] == color)):
                candidates.append((r0,r1,c0,c1,color))
    # A payload ring entirely inside a real frame is content, not another object.
    outer = [a for a in candidates if not any(
        b[0] < a[0] and a[1] < b[1] and b[2] < a[2] and a[3] < b[3]
        for b in candidates)]
    _require(len(outer) == 3 and len({b[4] for b in outer}) == 3, 'need three distinct frames')
    for i,a in enumerate(outer):
        for b in outer[i+1:]:
            _require(a[1]+1 < b[0] or b[1]+1 < a[0] or a[3]+1 < b[2] or b[3]+1 < a[2],
                     'frames need blank separation')
    covered = np.zeros(grid.shape, bool)
    for r0,r1,c0,c1,_ in outer:
        covered[r0:r1+1,c0:c1+1] = True
    _require(np.all(grid[~covered] == bg), 'unrelated content outside frames')
    return sorted(outer)


def _decode(input_grid):
    inp = _grid(input_grid)
    bg = int(inp[0,0])
    _require(np.all(inp[[0,-1],:] == bg) and np.all(inp[:,[0,-1]] == bg), 'background rim missing')
    frames = _frames(inp,bg)
    interiors = [inp[r0+1:r1,c0+1:c1].copy() for r0,r1,c0,c1,_ in frames]
    keys = [f[4] for f in frames]
    board_candidates = [i for i,p in enumerate(interiors)
                        if np.any(p == keys[i]) and sum(int(np.count_nonzero(p == k))
                            for k in keys if k != keys[i]) == 1]
    _require(len(board_candidates) == 1, 'ambiguous board role')
    bi = board_candidates[0]
    board, board_color = interiors[bi], keys[bi]
    stamp_indices = [i for i in range(3) if i != bi]
    matches = [(i,tuple(map(int,np.argwhere(board == keys[i])[0])))
               for i in stamp_indices if np.count_nonzero(board == keys[i]) == 1]
    _require(len(matches) == 1, 'board needs one keyed marker')
    ai, marker = matches[0]
    oi = next(i for i in stamp_indices if i != ai)
    a, b = interiors[ai],interiors[oi]
    _require(a.shape == b.shape and all(n >= 3 and n % 2 == 1 for n in a.shape), 'invalid stamp dimensions')
    _require(all(n >= 3 for n in board.shape), 'board too small')
    _require(all(n*s <= 30 for n,s in zip(board.shape,a.shape)), 'expanded board too large')
    signal_colors = set(map(int,np.unique(board))) - {bg,board_color,keys[ai]}
    _require(len(signal_colors) == 1, 'board needs one placement signal color')
    signal = next(iter(signal_colors))
    _require(signal not in keys, 'signal collides with frame key')
    for stamp in (a,b):
        palette = set(map(int,np.unique(stamp))) - {bg}
        _require(len(palette) >= 2 and not palette & (set(keys) | {signal}), 'invalid texture palette')
    cut = board == board_color
    path, regions, sites = _cut_info(cut)
    side = 0 if regions[0][marker] else 1
    _require(marker in sites[side], 'marker touches the cut')
    active = board != bg
    fine, marked, other = _refine(cut,a.shape,marker,path=path)
    return dict(inp=inp,bg=bg,frames=frames,board_index=bi,board=board,board_color=board_color,
                marker=marker,marker_key=keys[ai],other_key=keys[oi],signal=signal,a=a,b=b,
                cut=cut,path=path,regions=regions,sites=sites,side=side,active=active,
                fine=fine,marked=marked,other=other)


def _fields(d, a=None, b=None):
    a = d['a'] if a is None else a
    b = d['b'] if b is None else b
    sh, sw = a.shape
    rr, cc = np.indices(d['fine'].shape)
    return a[rr%sh,cc%sw], b[rr%sh,cc%sw], d['active'][rr//sh,cc//sw]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the displayed frames, cut, marker, signal, and textures."""
    return _solve_decoded(_decode(input_grid))


def _solve_decoded(d):
    """Fine-cell prediction after one input-only parse, with no persistent cache."""
    a,b,active = _fields(d)
    seam = np.where(a != d['bg'],a,b)
    regional = np.where(d['marked'],a,b)
    return np.where(active,np.where(d['fine'],seam,regional),d['bg']).astype(np.int8)


def _block_output(cut, marker, active, a, b, bg, *, geometry=None):
    """Generation path: write complete tiles, then splice each tile locally."""
    fine, marked, _ = _refine(cut,a.shape,marker) if geometry is None else geometry
    sh, sw = a.shape
    out = np.full(fine.shape,bg,np.int8)
    for r,c in np.argwhere(active):
        r,c = int(r),int(c)
        sl = np.s_[r*sh:(r+1)*sh,c*sw:(c+1)*sw]
        tile = b.copy()
        tile[marked[sl]] = a[marked[sl]]
        seam = fine[sl]
        tile[seam] = b[seam]
        use_a = seam & (a != bg)
        tile[use_a] = a[use_a]
        out[sl] = tile
    return out


def _flags(d):
    a,b,active = _fields(d)
    bg, seam, m, o = d['bg'],d['fine'],d['marked'],d['other']
    sh,sw = d['a'].shape
    expected = np.where(seam,np.where(a != bg,a,b),np.where(m,a,b))
    mixed = False
    for r,c in np.argwhere(d['cut']):
        sl = np.s_[r*sh:(r+1)*sh,c*sw:(c+1)*sw]
        tile = expected[sl]
        local_a,local_b = d['a'],d['b']
        if (np.any(m[sl] & (local_a != bg)) and np.any(o[sl] & (local_b != bg))
                and all(not np.array_equal(tile,x) for x in
                        (local_a,local_b,np.where(local_a != bg,local_a,local_b)))):
            mixed = True
            break
    return {name for name,present in dict(
        marked_visible=np.any(active & m & (a != bg)),
        other_visible=np.any(active & o & (b != bg)),
        marked_suppresses=np.any(active & m & (a == bg) & (b != bg)),
        other_suppresses=np.any(active & o & (b == bg) & (a != bg)),
        collision=np.any(seam & (a != bg) & (b != bg) & (a != b)),
        fallback=np.any(seam & (a == bg) & (b != bg)),
        a_only=np.any(seam & (a != bg) & (b == bg)),
        blank_seam=np.any(seam & (a == bg) & (b == bg)),
        inactive=np.any(~active & ((a != bg) | (b != bg))),
        mixed_tile=mixed).items() if bool(present)}


_REQUIRED = {'marked_visible','other_visible','marked_suppresses','other_suppresses',
             'collision','fallback','a_only','blank_seam','inactive','mixed_tile'}


def _transforms(a):
    return [f(np.rot90(a,k)) for k in range(4) for f in (lambda x:x,np.fliplr)]


def _asymmetric(a):
    return all(not np.array_equal(a,x) for x in _transforms(a)[1:])


def _alternative(d, mode):
    """Small explicit set of plausible shortcuts, not an independent solver."""
    aa,bb = d['a'],d['b']
    bg,cut,m,o = d['bg'],d['fine'],d['marked'],d['other']
    if mode.startswith('texture:'):
        _,source,operation = mode.split(':')
        def change(x):
            if operation == 'flat':
                colors,counts = np.unique(x[x != bg],return_counts=True)
                return np.where(x != bg,colors[np.argmax(counts)],bg)
            if operation == 'row':
                return np.roll(x,1,axis=0)
            if operation == 'col':
                return np.roll(x,1,axis=1)
            return _transforms(x)[int(operation)]
        if source in ('a','both'):
            aa = change(aa)
        if source in ('b','both'):
            bb = change(bb)
        if aa.shape != d['a'].shape or bb.shape != d['b'].shape:
            # Rotated rectangular stamps imply changed output dimensions.
            return np.empty((0,0),np.int8)
    a,b,active = _fields(d,aa,bb)
    selected = m
    seam = np.where(a != bg,a,b)
    result = np.where(selected,a,b)
    if mode == 'fixed0' or mode == 'fixed1':
        selected = m if d['side'] == int(mode[-1]) else o
        result = np.where(selected,a,b)
    elif mode == 'reverse':
        result = np.where(m,b,a)
    elif mode == 'only_a':
        result = a
    elif mode == 'only_b':
        result = b
    elif mode == 'union':
        result = seam
    elif mode.startswith('whole:'):
        coarse_m = d['regions'][d['side']]
        coarse_o = d['regions'][1-d['side']]
        sh,sw = aa.shape
        repeat = lambda x:np.repeat(np.repeat(x,sh,axis=0),sw,axis=1)
        result = np.where(repeat(coarse_m),a,b)
        coarse_cut = repeat(d['cut'])
        kind = mode.split(':')[1]
        value = {'a':a,'b':b,'union':seam,'blank':np.full(a.shape,bg,np.int8)}[kind]
        result = np.where(coarse_cut,value,result)
        return np.where(active,result,bg).astype(np.int8)
    if mode == 'seam_other':
        seam = np.where(b != bg,b,a)
    elif mode == 'seam_opaque':
        seam = a
    elif mode == 'seam_blank':
        seam = np.full(a.shape,bg,np.int8)
    if mode == 'fill_inactive':
        active = np.ones(active.shape,bool)
    return np.where(active,np.where(cut,seam,result),bg).astype(np.int8)


_ALTERNATIVES = (['fixed0','fixed1','reverse','only_a','only_b','union',
                  'whole:a','whole:b','whole:union','whole:blank',
                  'seam_other','seam_opaque','seam_blank','fill_inactive']
                 + [f'texture:{s}:{op}' for s in ('a','b','both')
                    for op in ('flat','row','col','1','2','3','4','5','6','7')])


def _audit_alternatives(train,test):
    for mode in _ALTERNATIVES:
        if all(np.array_equal(_alternative(d,mode),out) for d,out in train):
            _require(all(np.array_equal(_alternative(d,mode),out) for d,out in test),
                     'training leaves a consequential shortcut unresolved: '+mode)


def _sample_cut(rng, shape, kind):
    h,w = shape
    cut = np.zeros(shape,bool)
    if kind == 'horizontal':
        cut[int(rng.integers(2,h-2)),:] = True
    elif kind == 'vertical':
        cut[:,int(rng.integers(2,w-2))] = True
    elif kind == 'elbow':
        r,c = int(rng.integers(2,h-2)),int(rng.integers(2,w-2))
        cut[:r+1,c] = True
        cut[r,c:] = True
    elif kind == 'stepped':
        transpose = not (h >= 7 and w >= 6) or (w >= 7 and h >= 6 and rng.random() < .5)
        hh,ww = (w,h) if transpose else (h,w)
        trial = np.zeros((hh,ww),bool)
        possible = list(range(2,hh-2,2))
        n = int(rng.integers(1,len(possible)+1))
        rows = sorted(map(int,rng.choice(possible,n,replace=False)))
        col,last = int(rng.integers(2,ww-2)),0
        for row in rows:
            trial[last:row+1,col] = True
            choices = [c for c in range(2,ww-2) if c != col]
            newcol = choices[int(rng.integers(len(choices)))]
            trial[row,min(col,newcol):max(col,newcol)+1] = True
            col,last = newcol,row
        trial[last:,col] = True
        cut = trial.T.copy() if transpose else trial
    else:
        raise ValueError('unknown cut family')
    if rng.random() < .5:
        cut = cut[::-1,:].copy()
    if rng.random() < .5:
        cut = cut[:,::-1].copy()
    _cut_info(cut)
    return cut


def _shape_choices(regime,split,kind,wide,previous=None):
    options = []
    for h in range(3,11):
        for w in range(3,11):
            if kind == 'horizontal' and h < 5 or kind == 'vertical' and w < 5:
                continue
            if kind == 'elbow' and min(h,w) < 5:
                continue
            if kind == 'stepped' and not (h >= 7 and w >= 6 or h >= 6 and w >= 7):
                continue
            if regime == 'scale' and (split == 'train' and max(h,w) > 6 or split == 'test' and max(h,w) <= 6):
                continue
            for sh in (3,5,7,9):
                for sw in (3,5,7,9):
                    if h*sh > 30 or w*sw > 30:
                        continue
                    if regime == 'aspect':
                        want_wide = wide if split == 'train' else not wide
                        if not (sw > sh if want_wide else sh > sw):
                            continue
                    if previous is not None and ((h,w) == previous[0] or (sh,sw) == previous[1]):
                        continue
                    options.append(((h,w),(sh,sw)))
    _require(options,'no compatible shapes')
    return options


def _placement(rng,shape):
    h,w = shape
    rr,cc = np.indices(shape)
    style = int(rng.integers(7))
    if style == 0:
        mask = (rr == 0) | (rr == h-1) | (cc == 0) | (cc == w-1)
    elif style == 1:
        mask = (rr+cc+int(rng.integers(2))) % 2 == 0
    elif style == 2:
        mask = (rr % 2 == 0) & (cc % 2 == 0)
    elif style == 3:
        mask = (rr if rng.random() < .5 else cc) % int(rng.integers(2,4)) == 0
    elif style == 4:
        mask = np.ones(shape,bool)
    else:
        mask = rng.random(shape) < float(rng.uniform(.12,.85))
    return mask


def _texture(rng,shape,bg,payload,witness=False,preferred=0):
    h,w = shape
    rr,cc = np.indices(shape)
    for _ in range(96):
        style = int(rng.integers(7)) if not witness else int(rng.choice([2,4,6]))
        if not witness and rng.random() < .35:
            style = preferred
        k = int(rng.integers(2,len(payload)+1))
        colors = rng.choice(payload,k,replace=False)
        if style == 0:
            labels = rr if rng.random() < .5 else cc
        elif style == 1:
            labels = np.minimum.reduce([rr,cc,h-1-rr,w-1-cc])
        elif style == 2:
            labels = rr+cc*int(rng.integers(1,k+1))
        elif style == 3:
            labels = (rr+cc) % 2
        elif style == 4:
            labels = abs(rr-h//2)+abs(cc-w//2)
        elif style == 5:
            labels = (rr == h//2) | (cc == w//2)
        else:
            labels = rng.integers(k,size=shape)
        a = colors[(labels+int(rng.integers(k))) % k].astype(np.int8)
        holes = np.zeros(shape,bool)
        hole_style = int(rng.integers(5))
        if witness or hole_style == 1:
            holes = rng.random(shape) < float(rng.uniform(.12,.35))
        elif hole_style == 2:
            holes[h//2,w//2] = True
        elif hole_style == 3:
            holes[1:-1,1:-1] = True
        elif hole_style == 4:
            holes = (rr+2*cc+int(rng.integers(3))) % 4 == 0
        a[holes] = bg
        if len(set(map(int,np.unique(a)))-{bg}) < 2:
            continue
        if witness and not _asymmetric(a):
            continue
        return a
    # This explicit asymmetric corner motif has two colors and a hole.
    a = np.full(shape,payload[0],np.int8)
    a[0,0:2] = payload[1]
    a[1,1] = bg
    a[-1,0] = payload[1]
    return a


def _framed(content,color):
    out = np.full((content.shape[0]+2,content.shape[1]+2),color,np.int8)
    out[1:-1,1:-1] = content
    return out


def _pack(rng,board,stamps,bg):
    """Three separated rectangles; the matching key, never layout, orders sources."""
    stamps = list(stamps)
    if rng.random() < .5:
        stamps.reverse()
    bh,bw = board.shape
    sh,sw = stamps[0].shape
    feasible = []
    for mode in range(4):
        for gap in range(1,4):
            if mode < 2:
                base = (max(bh,2*sh+gap),bw+gap+sw)
            else:
                base = (bh+gap+sh,max(bw,2*sw+gap))
            if max(base) <= 28:
                feasible.append((mode,gap,base))
    mode,gap,(hh,ww) = feasible[int(rng.integers(len(feasible)))]
    h = hh+2+int(rng.integers(min(2,28-hh)+1))
    w = ww+2+int(rng.integers(min(2,28-ww)+1))
    out = np.full((h,w),bg,np.int8)
    if mode < 2:
        left = 1+int(rng.integers(w-(bw+gap+sw)-1))
        bc,sc = (left,left+bw+gap) if mode == 0 else (left+sw+gap,left)
        br = 1+int(rng.integers(h-bh-1))
        sr = 1+int(rng.integers(h-(2*sh+gap)-1))
        positions = [(board,br,bc),(stamps[0],sr,sc),(stamps[1],sr+sh+gap,sc)]
    else:
        top = 1+int(rng.integers(h-(bh+gap+sh)-1))
        br,sr = (top,top+bh+gap) if mode == 2 else (top+sh+gap,top)
        bc = 1+int(rng.integers(w-bw-1))
        sc = 1+int(rng.integers(w-(2*sw+gap)-1))
        positions = [(board,br,bc),(stamps[0],sr,sc),(stamps[1],sr,sc+sw+gap)]
    for content,r,c in positions:
        out[r:r+content.shape[0],c:c+content.shape[1]] = content
    return out


def _new_pair(rng,params,split,kind,side,previous=None,witness=False):
    choices = _shape_choices(params['regime'],split,kind,params['wide'],previous)
    # Select coarse shape uniformly, then an independently compatible stamp shape.
    coarse = sorted({x[0] for x in choices})
    shape = coarse[int(rng.integers(len(coarse)))]
    sizes = [s for m,s in choices if m == shape]
    sshape = sizes[int(rng.integers(len(sizes)))]
    cut = _sample_cut(rng,shape,kind)
    _,regions,sites = _cut_info(cut)
    marker = sites[side][int(rng.integers(len(sites[side])))]
    colors = params['palette'] if params['stable'] else list(map(int,rng.permutation(10)))
    bg,bkey,key0,key1,signal = colors[:5]
    matching = int(rng.integers(2))
    akey,bkey_other = (key0,key1) if matching == 0 else (key1,key0)
    a = _texture(rng,sshape,bg,colors[5:],witness,params['preferred'])
    b = _texture(rng,sshape,bg,colors[5:],witness,params['preferred'])
    active = _placement(rng,shape) | cut
    active[marker] = True
    board = np.where(active,signal,bg).astype(np.int8)
    board[cut] = bkey
    board[marker] = akey
    if not np.any(board == signal):
        # Require a visible signal, without imposing connectivity on placements.
        point = next(p for p in sites[1-side] if p != marker)
        board[point] = signal
        active[point] = True
    inp = _pack(rng,_framed(board,bkey),(_framed(a,akey),_framed(b,bkey_other)),bg)
    out = _block_output(cut,marker,active,a,b,bg)
    return {'input':inp,'output':out}


def _counts(num_train,num_test):
    for name,v,lo,hi in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or not lo <= int(v) <= hi:
            raise ValueError(name+' must be a legal integer count')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train),int(num_test)


def _key(a):
    return a.shape,a.astype(np.int8,copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt,nq = _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    params = dict(regime=str(rng.choice(['scale','bends','aspect'])),
                  palette=list(map(int,rng.permutation(10))),stable=bool(rng.random() < .3),
                  preferred=int(rng.integers(6)),wide=bool(rng.integers(2)))
    first_side = int(rng.integers(2))
    puzzle = {'train':[],'test':[]}
    seen,coverage = set(),set()
    previous = None
    for i in range(nt):
        kind = 'elbow' if i == 0 else str(rng.choice(['horizontal','vertical','elbow']))
        if i == 1:
            kind = ('horizontal' if params['wide'] else 'vertical') if params['regime'] == 'aspect' else str(rng.choice(['horizontal','vertical']))
        side = first_side if i == 0 else (1-first_side if i == 1 else int(rng.integers(2)))
        missing = _REQUIRED-coverage if i == nt-1 else set()
        for attempt in range(384):
            pair = _new_pair(rng,params,'train',kind,side,previous if i == 1 else None,i == 0)
            d = _decode(pair['input'])
            flags = _flags(d)
            # Each of the first two cases makes its marker-side choice effective.
            field_a,field_b,field_active = _fields(d)
            effective = np.any((field_a != field_b) & ~d['fine'] & field_active)
            if _key(pair['input']) in seen or not effective or not missing <= flags:
                continue
            if i == 0 and not {'marked_visible','other_visible','mixed_tile'} <= flags:
                continue
            break
        else:
            raise RuntimeError('bounded training sampler could not expose the rule')
        seen.add(_key(pair['input']))
        coverage |= flags
        puzzle['train'].append(pair)
        previous = d['board'].shape,d['a'].shape
    for i in range(nq):
        kind = 'stepped' if params['regime'] == 'bends' else str(rng.choice(['horizontal','vertical','elbow']))
        for attempt in range(96):
            pair = _new_pair(rng,params,'test',kind,int(rng.integers(2)))
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError('bounded sampler could not find distinct test')
        seen.add(_key(pair['input']))
        puzzle['test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s],list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']),len(puzzle['test']))
    records = {'train':[],'test':[]}
    seen,coverage,sides = set(),set(),set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'}, 'invalid pair keys')
            inp,out = _grid(pair['input']),_grid(pair['output'])
            key = _key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            d = _decode(inp)
            # Fine-cell solver and separate tile-write construction must agree.
            _require(np.array_equal(_solve_decoded(d),out), 'incorrect mosaic')
            _require(np.array_equal(_block_output(d['cut'],d['marker'],d['active'],d['a'],d['b'],d['bg'],
                                                geometry=(d['fine'],d['marked'],d['other'])),out),
                     'incorrect block mosaic')
            _require(np.any(out != d['bg']), 'empty result')
            _require(not set(map(int,np.unique(out))) & {d['board_color'],d['marker_key'],d['other_key'],d['signal']},
                     'control color leaked to output')
            records[split].append((d,out))
            if split == 'train':
                coverage |= _flags(d)
                a,b,active = _fields(d)
                if np.any((a != b) & ~d['fine'] & active):
                    sides.add(d['side'])
    _require(_REQUIRED <= coverage, 'training lacks a rule witness: '+','.join(sorted(_REQUIRED-coverage)))
    _require(sides == {0,1}, 'marker side not effectively varied')
    _require(len({d['board'].shape for d,_ in records['train']}) >= 2, 'board shapes do not vary')
    _require(len({d['a'].shape for d,_ in records['train']}) >= 2, 'stamp shapes do not vary')
    train_families = {_family(d)[0] for d,_ in records['train']}
    _require({'straight','elbow'} <= train_families, 'training must demonstrate straight and turning contours')
    scale_shift = all(max(d['board'].shape) > max(max(t['board'].shape) for t,_ in records['train'])
                      for d,_ in records['test'])
    bend_shift = (all(_family(d)[1] <= 1 for d,_ in records['train'])
                  and all(_family(d)[1] >= 2 for d,_ in records['test']))
    aspect_shift = any(all((d['a'].shape[1] > d['a'].shape[0]) == wide
                          and d['a'].shape[0] != d['a'].shape[1] for d,_ in records['train'])
                       and all((d['a'].shape[1] > d['a'].shape[0]) != wide
                               and d['a'].shape[0] != d['a'].shape[1] for d,_ in records['test'])
                       for wide in (False,True))
    _require(scale_shift or bend_shift or aspect_shift, 'no intended geometric train/test shift')
    _audit_alternatives(records['train'],records['test'])


def _family(d):
    steps = [(q[0]-p[0],q[1]-p[1]) for p,q in zip(d['path'],d['path'][1:])]
    bends = sum(p != q for p,q in zip(steps,steps[1:]))
    return 'straight' if bends == 0 else ('elbow' if bends == 1 else 'stepped'),bends


def _canonical(a):
    mapping = {}
    return a.shape,bytes(mapping.setdefault(int(x),len(mapping)) for x in a.flat)


def _digest(puzzle):
    h = hashlib.sha256()
    for s in ('train','test'):
        for pair in puzzle[s]:
            for k in ('input','output'):
                a = np.asarray(pair[k],np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _move_marker(inp):
    d = _decode(inp)
    out = inp.copy()
    r0,r1,c0,c1,_ = d['frames'][d['board_index']]
    old,new = d['marker'],d['sites'][1-d['side']][0]
    out[r0+1+old[0],c0+1+old[1]] = d['signal']
    out[r0+1+new[0],c0+1+new[1]] = d['marker_key']
    # Preserve activity at the new site for a controlled region-only comparison.
    original = inp.copy()
    original[r0+1+new[0],c0+1+new[1]] = d['signal']
    return original,out


def self_test():
    """Bounded: 42 default seeds, 30 count cases, and small metamorphic checks."""
    # Hand-worked cut tile: left column from A, center foreground-priority seam,
    # right column from B. This checks refinement against literal expectations.
    cut = np.zeros((5,5),bool)
    cut[:,2] = True
    active = np.ones((5,5),bool)
    active[4,4] = False
    a = np.array([[2,0,3],[3,2,0],[0,3,2]],np.int8)
    b = np.array([[3,2,0],[0,3,2],[2,0,3]],np.int8)
    expected_tile = np.array([[2,2,0],[3,2,2],[0,3,3]],np.int8)
    manual = _block_output(cut,(2,0),active,a,b,0)
    _require(np.array_equal(manual[:3,6:9],expected_tile), 'hand-worked splice failed')
    _require(np.all(manual[12:15,12:15] == 0), 'inactive tile not blank')
    fine,_,_ = _refine(cut,a.shape,(2,0))
    _require(np.array_equal(np.argwhere(fine),np.array([(r,7) for r in range(15)])),
             'straight fine contour is not at tile centers')
    transposed = _block_output(cut.T,(0,2),active.T,a.T,b.T,0)
    _require(np.array_equal(transposed,manual.T), 'transposed contour and textures failed')
    for kind in ('branch','loop','two_cuts'):
        invalid = cut.copy()
        if kind == 'branch':
            invalid[2,2:4] = True
        elif kind == 'loop':
            invalid = np.zeros((7,7),bool)
            invalid[2:5,2] = invalid[2:5,4] = True
            invalid[2,2:5] = invalid[4,2:5] = True
        else:
            invalid[:,1] = True
        try:
            _cut_info(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError('malformed cut accepted: '+kind)
    fingerprints,shapes,boards,stamps,patterns,palettes = set(),set(),set(),set(),set(),set()
    families,sides,counts,bends = set(),set(),set(),set()
    pair_count = mixed = blank_blocks = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprint = _digest(p)
        fingerprints.add(fingerprint)
        _require(fingerprint == _digest(generate_puzzle(seed)), 'nondeterministic puzzle')
        for split in ('train','test'):
            for pair in p[split]:
                d = _decode(pair['input'])
                pair_count += 1
                family,n = _family(d)
                families.add((split,family))
                sides.add((family,d['side']))
                bends.add(n)
                shapes.add(pair['input'].shape)
                shapes.add(pair['output'].shape)
                boards.add(d['board'].shape)
                stamps.add(d['a'].shape)
                patterns.add(hashlib.sha256(repr(_canonical(pair['input'])).encode()).hexdigest())
                palettes.add(tuple(map(int,np.unique(pair['input']))))
                counts.add(int(d['active'].sum()))
                flags = _flags(d)
                mixed += 'mixed_tile' in flags
                blank_blocks += 'inactive' in flags
        original,moved = _move_marker(p['train'][0]['input'])
        d = _decode(original)
        x,y = solve(original),solve(moved)
        _require(np.array_equal(x[d['fine']],y[d['fine']]), 'marker relocation changed seam priority')
        _require(np.any(x[~d['fine']] != y[~d['fine']]), 'marker relocation has no effect')
        mapping = np.random.default_rng(seed+4200).permutation(10)
        pair = p['test'][0]
        _require(np.array_equal(solve(mapping[pair['input']]),mapping[pair['output']]), 'color equivariance failed')
    _require(len(fingerprints) == 42 and len(patterns) >= 180, 'insufficient structural variation')
    _require(len(shapes) >= 35 and len(boards) >= 20 and len(stamps) >= 5, 'size variation collapsed')
    _require(len(palettes) >= 10 and len(counts) >= 20, 'palette or count variation collapsed')
    _require(('test','stepped') in families and max(bends) >= 4, 'contour variation collapsed')
    _require(all((kind,side) in sides for kind in ('straight','elbow','stepped') for side in (0,1)), 'fixed marker-side shortcut')
    _require(mixed >= 100 and blank_blocks >= 100, 'interaction evidence missing')
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(42000+10*nt+nq,nt,nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'wrong explicit counts')
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    for seed in (-1,True,1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal seed accepted')
    damaged = generate_puzzle(0)
    out = damaged['test'][0]['output']
    out[0,0] = (int(out[0,0])+1)%10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    return dict(seeds=42,pairs=pair_count,distinct_puzzles=len(fingerprints),
                normalized_inputs=len(patterns),grid_shapes=len(shapes),board_shapes=len(boards),
                stamp_shapes=len(stamps),palettes=len(palettes),active_counts=len(counts),
                mixed_pairs=mixed,inactive_pairs=blank_blocks,max_bends=max(bends))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(),separators=(',',':')))
