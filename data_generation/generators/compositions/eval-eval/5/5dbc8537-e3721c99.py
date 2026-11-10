"""Cavity-addressed mosaics. Standard library and NumPy only.

No generation state is read by solve(). Diagnostics remain in memory; importing
this module writes nothing. All search, construction and self-test loops are
bounded. No execution traces or puzzle collections are saved.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _bitmap(mask):
    """Row-major bits and edge masks, calculated solely from this input mask."""
    h,w = mask.shape
    full = (1 << (h*w))-1
    row = (1 << w)-1
    left = full // row
    right = left << (w-1)
    bits = int.from_bytes(np.packbits(mask,bitorder='little').tobytes(),'little')
    return bits,full,left,right


def _flood_bits(seed, allowed, width, left, right):
    """Four-neighbor flood fill; edge masks prevent horizontal row wrapping."""
    reached = seed
    while True:
        grown = (reached | ((reached << 1) & ~left)
                 | ((reached >> 1) & ~right)
                 | (reached << width) | (reached >> width)) & allowed
        if grown == reached:
            return reached
        reached = grown


def _components(mask):
    if not mask.size:
        return []
    h,w = mask.shape
    remaining,full,left,right = _bitmap(mask)
    result = []
    while remaining:
        component = _flood_bits(remaining & -remaining,remaining,w,left,right)
        remaining ^= component
        cells = []
        while component:
            bit = component & -component
            cells.append(divmod(bit.bit_length()-1,w))
            component ^= bit
        result.append(cells)
    return result


def _mask(cells):
    p = np.asarray(cells, dtype=int)
    lo, hi = p.min(axis=0), p.max(axis=0)+1
    a = np.zeros(tuple(hi-lo), dtype=bool)
    a[tuple((p-lo).T)] = True
    return a, tuple(int(v) for v in lo)


def _holes(mask):
    if not mask.size:
        return 0
    h,w = mask.shape
    foreground,full,left,right = _bitmap(mask)
    background = full ^ foreground
    row = (1 << w)-1
    boundary = left | right | row | (row << ((h-1)*w))
    # Flood all boundary background together, exactly as an exterior padded
    # border would connect it. Diagonal-only escapes remain enclosed.
    exterior = _flood_bits(background & boundary,background,w,left,right)
    remaining = background ^ exterior
    count = 0
    while remaining:
        component = _flood_bits(remaining & -remaining,remaining,w,left,right)
        remaining ^= component
        count += 1
    return count


def _positions(allowed, h, w):
    if h > allowed.shape[0] or w > allowed.shape[1]:
        return []
    s = np.zeros((allowed.shape[0]+1,allowed.shape[1]+1),dtype=np.int32)
    np.cumsum(allowed,axis=0,dtype=np.int32,out=s[1:,1:])
    np.cumsum(s,axis=1,dtype=np.int32,out=s)
    t = s[h:,w:]-s[:-h,w:]-s[h:,:-w]+s[:-h,:-w]
    rows,cols = np.nonzero(t == h*w)
    return list(zip(rows.tolist(),cols.tolist()))


def _parse(input_grid):
    g = _grid(input_grid)
    candidates = []
    for axis in (0, 1):
        for cut in range(1, g.shape[axis]-1):
            line = g[cut,:] if axis == 0 else g[:,cut]
            sep = int(line[0])
            if not np.all(line == sep):
                continue
            a, b = (g[:cut],g[cut+1:]) if axis == 0 else (g[:,:cut],g[:,cut+1:])
            for board, tray, side in ((a,b,0),(b,a,1)):
                bc, tc = set(map(int,np.unique(board))), set(map(int,np.unique(tray)))
                shared = bc & tc
                if len(bc) != 2 or len(shared) != 1 or sep in bc | tc:
                    continue
                blank = next(iter(shared))
                ground = next(iter(bc-{blank}))
                cards, common = [], None
                good = True
                for cells in _components(tray != blank):
                    m, (y,x) = _mask(cells)
                    if not m.all():
                        good = False
                        break
                    tile = tray[y:y+m.shape[0],x:x+m.shape[1]].copy()
                    colors = set(map(int,np.unique(tile)))
                    if len(colors) != 2:
                        good = False
                        break
                    common = colors if common is None else common & colors
                    cards.append(tile)
                if not good or not cards or common is None or len(common) != 1:
                    continue
                ink = next(iter(common))
                class_paper, paper_class, typed = {}, {}, []
                for tile in cards:
                    printed = tile == ink
                    if len(_components(printed)) != 1:
                        good = False
                        break
                    k = _holes(printed)
                    paper = next(c for c in map(int,np.unique(tile)) if c != ink)
                    if ((k in class_paper and class_paper[k] != paper)
                            or (paper in paper_class and paper_class[paper] != k)):
                        good = False
                        break
                    class_paper[k], paper_class[paper] = paper, k
                    typed.append((k,tile))
                if not good:
                    continue
                labels = np.full(board.shape, -1, dtype=int)
                regions = []
                for cells in _components(board == blank):
                    m, pos = _mask(cells)
                    k = _holes(m)
                    labels[tuple(np.asarray(cells).T)] = k
                    regions.append((k,m,pos,len(cells)))
                if not regions:
                    continue
                candidates.append(dict(board=board.copy(),tray=tray.copy(),blank=blank,
                                       ground=ground,ink=ink,separator=sep,tiles=typed,
                                       labels=labels,regions=regions,class_paper=class_paper,
                                       side=(axis,side)))
    if len(candidates) != 1:
        raise ValueError("Input must identify exactly one stencil, tray and common ink")
    return candidates[0]


def _pack(labels, tiles, *, ignore_topology=False, max_nodes=30000):
    """Enumerate up to two distinct pixel renderings, with finite inventories.

    Returning one answer is accepted only if the search was exhaustive. Identical
    matrices share inventory, but tile identities are never output tie-breakers.
    """
    coords = [tuple(map(int,p)) for p in np.argwhere(labels >= 0)]
    index = np.full(labels.shape, -1, dtype=int)
    for i, p in enumerate(coords):
        index[p] = i
    groups, lookup = [], {}
    for k, tile in tiles:
        key = (k,_key(tile))
        if key in lookup:
            groups[lookup[key]][2] += 1
        else:
            lookup[key] = len(groups)
            groups.append([k,tile,1])
    placements, by_cell = [], [[] for _ in coords]
    for gi, (k,tile,_) in enumerate(groups):
        h,w = tile.shape
        allowed = labels >= 0 if ignore_topology else labels == k
        for y,x in _positions(allowed,h,w):
            ids = tuple(int(i) for i in index[y:y+h,x:x+w].flat)
            bits = sum(1 << i for i in ids)
            pi = len(placements)
            placements.append((gi,bits,ids,y,x))
            for i in ids:
                by_cell[i].append(pi)
    full = (1 << len(coords))-1
    remaining = [g[2] for g in groups]
    rendered = np.full(len(coords), -1, dtype=np.int8)
    answers, keys, path, dead = [], set(), [], set()
    nodes, exhaustive = 0, True

    def search(covered):
        nonlocal nodes, exhaustive
        nodes += 1
        if nodes > max_nodes:
            exhaustive = False
            return False
        if covered == full:
            key = rendered.tobytes()
            if key not in keys:
                keys.add(key)
                answers.append((rendered.copy(),tuple(path)))
            return True
        state = (covered,tuple(remaining))
        if state in dead:
            return False
        best = None
        for i in range(len(coords)):
            if covered & (1 << i):
                continue
            opts = [p for p in by_cell[i] if remaining[placements[p][0]]
                    and not (covered & placements[p][1])]
            if not opts:
                dead.add(state)
                return False
            if best is None or len(opts) < len(best):
                best = opts
                if len(opts) == 1:
                    break
        found = False
        for pi in best:
            gi,bits,ids,y,x = placements[pi]
            remaining[gi] -= 1
            rendered[list(ids)] = groups[gi][1].ravel()
            path.append((gi,y,x))
            found = search(covered | bits) or found
            path.pop()
            rendered[list(ids)] = -1
            remaining[gi] += 1
            if len(answers) >= 2 or not exhaustive:
                return found
        if not found:
            dead.add(state)
        return found

    search(0)
    return answers, dict(nodes=nodes,exhaustive=exhaustive,groups=groups,
                         placements=len(placements),coords=coords)


def _analyze(input_grid):
    p = _parse(input_grid)
    answers, info = _pack(p['labels'],p['tiles'])
    if not info['exhaustive'] or len(answers) != 1:
        raise ValueError("The topology-constrained packing must have one rendered answer")
    out = p['board'].copy()
    out[p['labels'] >= 0] = answers[0][0]
    p.update(output=out,path=answers[0][1],info=info)
    return p


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all conventions, topology, patterns and inventories from the input."""
    return _analyze(input_grid)['output']


def _glyph(k, shape, rng):
    h,w = shape
    if h < 3 or w < 3 or (k >= 2 and max(h,w) < 5):
        raise ValueError("Card too small for requested glyph")
    if k == 0:
        # Crooked strokes with one orthogonal neighbor per added cell. Digital
        # topology still needs checking: diagonal contacts can enclose a
        # four-connected background cell even without a foreground graph cycle.
        m = np.zeros(shape,dtype=bool)
        m[int(rng.integers(h)),int(rng.integers(w))] = True
        target = int(rng.integers(max(2,h*w//3), max(3,3*h*w//4)))
        for _ in range(h*w):
            neighbors = np.zeros(shape,dtype=int)
            neighbors[1:] += m[:-1]; neighbors[:-1] += m[1:]
            neighbors[:,1:] += m[:,:-1]; neighbors[:,:-1] += m[:,1:]
            options = np.argwhere(~m & (neighbors == 1))
            if not len(options):
                break
            pos = tuple(options[int(rng.integers(len(options)))])
            candidate = m.copy()
            candidate[pos] = True
            if _holes(candidate) == 0:
                m = candidate
            if int(m.sum()) >= target:
                break
        return m
    m = np.ones(shape,dtype=bool)
    if k == 1:
        seeds = [(int(rng.integers(1,h-1)),int(rng.integers(1,w-1)))]
    elif h >= 5:
        seeds = [(1,int(rng.integers(1,w-1))),(h-2,int(rng.integers(1,w-1)))]
    else:
        seeds = [(int(rng.integers(1,h-1)),1),(int(rng.integers(1,h-1)),w-2)]
    for pos in seeds:
        m[pos] = False
    # Grow unequal, potentially bent cavities, without merging or severing ink.
    for _ in range(min(18,h*w//2)):
        pos = (int(rng.integers(1,h-1)),int(rng.integers(1,w-1)))
        if not m[pos]:
            continue
        candidate = m.copy()
        candidate[pos] = False
        if len(_components(candidate)) == 1 and _holes(candidate) == k:
            m = candidate
    # Exterior notches and diagonal-only escapes are allowed if topology survives.
    corners = [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]
    for pos in corners:
        if rng.random() < .35:
            candidate = m.copy(); candidate[pos] = False
            if len(_components(candidate)) == 1 and _holes(candidate) == k:
                m = candidate
    assert _holes(m) == k and len(_components(m)) == 1 and not m.all()
    return m


def _trim_rects(rects):
    y0,x0 = min(r[0] for r in rects),min(r[1] for r in rects)
    return [(y-y0,x-x0,h,w) for y,x,h,w in rects]


def _rect_mask(rects):
    h = max(y+a for y,x,a,b in rects)
    w = max(x+b for y,x,a,b in rects)
    m = np.zeros((h,w),dtype=bool)
    for y,x,a,b in rects:
        if m[y:y+a,x:x+b].any():
            raise ValueError("Overlapping construction rectangles")
        m[y:y+a,x:x+b] = True
    return m


def _frame(k, rng):
    top,bottom = int(rng.integers(3,5)),int(rng.integers(3,5))
    middle = int(rng.integers(5,7))
    walls = [3 for _ in range(k+1)]
    holes = [int(rng.integers(1, 5 if k == 1 else 3)) for _ in range(k)]
    width = sum(walls)+sum(holes)
    height = top+middle+bottom
    rects = [(0,0,top,width),(top+middle,0,bottom,width)]
    x = 0
    for i,wall in enumerate(walls):
        rects.append((top,x,middle,wall))
        if i < k:
            x += wall+holes[i]
    if k == 1 and rng.random() < .4:
        left,right = walls[0],walls[-1]
        rects = [(0,0,top,width-right),(0,width-right,height-bottom,right),
                 (height-bottom,left,bottom,width-left),(top,0,height-top,left)]
    elif rng.random() < .55:
        # Hidden seams need not follow the same frame decomposition.
        which = int(rng.integers(2))
        y,x,h,w = rects[which]
        low = 3 if k == 1 else 5
        if w >= 2*low:
            cut = int(rng.integers(low,w-low+1))
            rects[which:which+1] = [(y,x,h,cut),(y,x+cut,h,w-cut)]
    if rng.random() < .5:
        rects = [(x,y,w,h) for y,x,h,w in rects]
    rects = _trim_rects(rects)
    # Pick a small shared footprint. Even an absent class has a local geometric fit.
    shared = min(((h,w) for y,x,h,w in rects if max(h,w) >= 5),
                 key=lambda s:s[0]*s[1])
    return rects,shared


def _grow_open(shared, max_h, max_w, count, rng):
    h,w = shared
    if h > max_h or w > max_w:
        return None
    rects = [(0,0,h,w)]
    for _ in range(count-1):
        options = []
        for _ in range(60):
            ry,rx,rh,rw = rects[int(rng.integers(len(rects)))]
            a,b = int(rng.integers(3,6)),int(rng.integers(3,6))
            side = int(rng.integers(4))
            if side < 2:
                y = ry-a if side == 0 else ry+rh
                x = rx+int(rng.integers(1-b,rw))
            else:
                x = rx-b if side == 2 else rx+rw
                y = ry+int(rng.integers(1-a,rh))
            candidate = _trim_rects(rects+[(y,x,a,b)])
            hh = max(r+c for r,s,c,d in candidate)
            ww = max(s+d for r,s,c,d in candidate)
            if hh > max_h or ww > max_w:
                continue
            try:
                m = _rect_mask(candidate)
            except ValueError:
                continue
            if _holes(m) == 0:
                options.append(candidate)
        if not options:
            return None
        rects = options[int(rng.integers(len(options)))]
    return rects


def _scatter(tiles, height, width, blank, rng):
    # Largest first is a private placement aid, not an input ordering convention.
    order = sorted(range(len(tiles)),key=lambda i:-tiles[i][1].size)
    for _ in range(18):
        tray = np.full((height,width),blank,dtype=np.int8)
        free = np.ones((height,width),dtype=bool)
        for i in order:
            tile = tiles[i][1]
            h,w = tile.shape
            positions = _positions(free,h,w)
            if not positions:
                break
            y,x = positions[int(rng.integers(len(positions)))]
            tray[y:y+h,x:x+w] = tile
            free[max(0,y-1):y+h+1,max(0,x-1):x+w+1] = False
        else:
            return tray
    return None


def _candidate(rng, k, orientation, extra):
    frame,shared = _frame(k,rng)
    fm = _rect_mask(frame)
    if fm.shape[1] > 13:
        return None
    # Reserve a small independently grown open component beside the enclosure.
    width = int(rng.integers(fm.shape[1],min(15,29-max(w for y,x,h,w in frame))+1))
    open_rects = _grow_open(shared, min(11,28-fm.shape[0]),width,
                            int(rng.integers(2,5)),rng)
    if open_rects is None:
        return None
    om = _rect_mask(open_rects)
    gap = int(rng.integers(1,3))
    margin = int(rng.integers(2))
    height = fm.shape[0]+om.shape[0]+gap+2*margin
    if height > 30:
        return None
    # Component order, offsets, source positions and final orientation are unrelated.
    components = [(k,frame,fm),(0,open_rects,om)]
    if rng.random() < .5:
        components.reverse()
    labels = np.full((height,width),-1,dtype=int)
    rects = []
    row = margin
    for cls,rr,mm in components:
        col = int(rng.integers(width-mm.shape[1]+1))
        for y,x,h,w in rr:
            labels[row+y:row+y+h,col+x:col+x+w] = cls
            rects.append((cls,row+y,col+x,h,w))
        row += mm.shape[0]+gap
    roles = [int(c) for c in rng.permutation(10)]
    ground,blank,ink,sep = roles[:4]
    paper = dict(enumerate(roles[4:7]))
    patterns = {}
    def pattern(cls,shape):
        key = (cls,shape)
        if key not in patterns:
            glyph = _glyph(cls,shape,rng)
            patterns[key] = np.where(glyph,ink,paper[cls]).astype(np.int8)
        return patterns[key]
    board = np.full(labels.shape,ground,dtype=np.int8)
    board[labels >= 0] = blank
    expected = board.copy()
    tiles = []
    for cls,y,x,h,w in rects:
        tile = pattern(cls,(h,w))
        tiles.append((cls,tile.copy()))
        expected[y:y+h,x:x+w] = tile
    absent = 2 if k == 1 else 1
    tiles.append((absent,pattern(absent,shared).copy()))
    if extra:
        # Optional same-class, locally fitting surplus; retain a unique image
        # with a cover using the original number of cards. Pixel-identical
        # alternative covers are permitted, including a different unused copy.
        for cls in (0,k):
            d = (3,3) if cls != 2 else (3,5)
            if (cls,d) in patterns or not _positions(labels == cls,*d):
                continue
            trial = tiles+[(cls,pattern(cls,d).copy())]
            aa,ii = _pack(labels,trial,max_nodes=6000)
            if ii['exhaustive'] and len(aa) == 1 and np.array_equal(aa[0][0],expected[labels>=0]):
                if len(aa[0][1]) == len(rects):
                    tiles = trial
                    break
    answers,info = _pack(labels,tiles,max_nodes=6000)
    if (not info['exhaustive'] or len(answers) != 1
            or not np.array_equal(answers[0][0],expected[labels >= 0])):
        return None
    minimum_tray = max(tile.shape[1] for cls,tile in tiles)
    tray_width = int(rng.integers(minimum_tray,30-width))
    tray = _scatter(tiles,height,tray_width,blank,rng)
    if tray is None:
        return None
    line = np.full((height,1),sep,dtype=np.int8)
    inp = np.concatenate((board,line,tray),axis=1)
    out = expected
    # Whole-pair isometry never rotates a card relative to its destination.
    inp,out = np.rot90(inp,orientation).copy(),np.rot90(out,orientation).copy()
    if rng.random() < .5:
        inp,out = inp[::-1].copy(),out[::-1].copy()
    return {'input':inp,'output':out}


def _counts(num_train, num_test):
    for name,value,low,high in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train),int(num_test)


def _fallback(rng, k, orientation, slot=0):
    """Guaranteed-fitting construction of the very same rule, with free identity
    in 26 tray positions. Used only after bounded free-growth rejection.
    """
    ground,blank,ink,sep,p0,p1,p2 = map(int,rng.permutation(10)[:7])
    papers = [p0,p1,p2]
    patterns = {}
    def tile(cls,shape):
        if (cls,shape) not in patterns:
            m = _glyph(cls,shape,rng)
            patterns[(cls,shape)] = np.where(m,ink,papers[cls]).astype(np.int8)
        return patterns[(cls,shape)]
    width = 4*k+3
    selected = [(k,1,1,(3,width)),(k,9,1,(3,width))]
    selected += [(k,4,1+4*j,(5,3)) for j in range(k+1)]
    selected += [(0,14,2,(5,3)),(0,19,3,(3,4))]
    board = np.full((24,13),ground,dtype=np.int8)
    out = board.copy()
    for cls,y,x,shape in selected:
        h,w = shape
        board[y:y+h,x:x+w] = blank
        out[y:y+h,x:x+w] = tile(cls,shape)
    absent = 2 if k == 1 else 1
    tray = np.full((24,16),blank,dtype=np.int8)
    tray[0:3,0:width] = tile(k,(3,width))
    tray[4:7,0:width] = tile(k,(3,width))
    small = [(k,(5,3))]*(k+1)+[(0,(5,3)),(absent,(5,3))]
    for i,(cls,shape) in enumerate(small):
        y,x = 8+(i//4)*6,(i%4)*4
        tray[y:y+5,x:x+3] = tile(cls,shape)
    y,x = 20+slot//13,slot%13
    tray[y:y+3,x:x+4] = tile(0,(3,4))
    inp = np.concatenate((board,np.full((24,1),sep,dtype=np.int8),tray),axis=1)
    return {'input':np.rot90(inp,orientation).copy(),'output':np.rot90(out,orientation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ne = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    result = {'train':[],'test':[]}
    seen = set()
    base = int(rng.integers(4))
    turn = int(rng.integers(2))
    for split,count in (('train',nt),('test',ne)):
        for i in range(count):
            k = 1 if split == 'train' else 2
            orientation = (base + (i % 2)*2) % 4 if split == 'train' else (base+turn) % 4
            for _ in range(100):
                pair = _candidate(rng,k,orientation,extra=(i % 2 == 1))
                if pair is None or _key(pair['input']) in seen:
                    continue
                # Input-only validation checks parsing; no expected shape is supplied.
                try:
                    p = _analyze(pair['input'])
                except ValueError:
                    continue
                if not np.array_equal(p['output'],pair['output']):
                    continue
                seen.add(_key(pair['input']))
                result[split].append(pair)
                break
            else:
                # 26 different positions for the last tray card, versus at most
                # 13 previous inputs, guarantee a distinct candidate. Reuse one
                # palette/glyph draw across the position trials.
                fallback_seed = int(rng.integers(2**63))
                for slot in range(26):
                    pair = _fallback(np.random.default_rng(fallback_seed),k,orientation,slot)
                    if _key(pair['input']) not in seen:
                        assert np.array_equal(_analyze(pair['input'])['output'],pair['output'])
                        seen.add(_key(pair['input']))
                        result[split].append(pair)
                        break
                else:
                    raise AssertionError('Distinct fallback positions exhausted')
    return result


def _features(p):
    groups = p['info']['groups']
    used = [groups[gi] for gi,y,x in p['path']]
    total = len(p['tiles'])
    shapes = {}
    for cls,tile in p['tiles']:
        shapes.setdefault(tile.shape,set()).add(cls)
    no_b,bi = _pack(p['labels'],p['tiles'],ignore_topology=True,max_nodes=6000)
    no_a = p['board'].copy()
    for k in np.unique(p['labels'][p['labels'] >= 0]):
        no_a[p['labels'] == k] = p['class_paper'][int(k)]
    return dict(side=p['side'],shape=p['board'].shape,classes=tuple(sorted(k for k,m,pos,n in p['regions'])),
                tiles=total,used=len(used),surplus=total-len(used),
                dimensions=tuple(sorted({tile.shape for cls,tile in p['tiles']})),
                repeats=any(g[2] > 1 for g in groups),
                shared_footprints=sum(len(v)>1 for v in shapes.values()),
                no_b_ambiguous=len(no_b)>=2,
                no_a_differs=not np.array_equal(no_a,p['output']),
                nodes=p['info']['nodes'],
                ink=p['ink'],roles=(p['ground'],p['blank'],p['ink'],p['separator']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen,features = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            p = _analyze(inp)
            assert np.array_equal(p['output'],out), 'Wrong copied mosaic'
            assert out.shape == p['board'].shape
            assert np.array_equal(out[p['labels'] < 0],p['board'][p['labels'] < 0])
            assert not np.any(out[p['labels'] >= 0] == p['blank'])
            # Verify the selected path uses complete same-class cards once each.
            inventory = [g[2] for g in p['info']['groups']]
            owner = np.zeros(out.shape,dtype=int)
            for gi,y,x in p['path']:
                cls,tile,n = p['info']['groups'][gi]
                h,w = tile.shape
                inventory[gi] -= 1
                assert inventory[gi] >= 0 and np.all(p['labels'][y:y+h,x:x+w] == cls)
                assert np.array_equal(out[y:y+h,x:x+w],tile)
                owner[y:y+h,x:x+w] += 1
            assert np.array_equal(owner,(p['labels'] >= 0).astype(int))
            f = _features(p)
            assert f['shared_footprints'] and f['no_b_ambiguous'] and f['no_a_differs']
            assert all(n > max(t.size for cls,t in p['tiles'] if cls == k)
                       for k,m,pos,n in p['regions']), 'Each region must require multiple cards'
            features.append((split,f))
    training = [f for s,f in features if s == 'train']
    testing = [f for s,f in features if s == 'test']
    assert all(f['classes'] == (0,1) for f in training)
    assert all(f['classes'] == (0,2) for f in testing)
    assert all(f['surplus'] >= 1 for s,f in features)


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train','test'):
        for pair in puzzle[split]:
            for role in ('input','output'):
                a = np.asarray(pair[role],dtype=np.int8)
                h.update(bytes(a.shape)); h.update(a.tobytes())
    return h.hexdigest()


def _three_region_fixture():
    """Hand-laid input outside the sampler: three destinations, two classes."""
    ink,blank,ground,sep = 7,0,1,9
    open_tile = np.array([[7,2,7],[7,2,7],[7,7,7]],dtype=np.int8)
    ring_tile = np.full((3,3),ink,dtype=np.int8); ring_tile[1,1] = 4
    bar_tile = np.full((3,11),ink,dtype=np.int8)
    bar_tile[1,2:8] = 4
    board = np.full((18,11),ground,dtype=np.int8)
    output = board.copy()
    for y,x,tile in ((0,0,bar_tile),(6,0,bar_tile),(3,0,ring_tile),
                     (3,8,ring_tile),(11,0,open_tile),(11,8,open_tile)):
        h,w = tile.shape
        board[y:y+h,x:x+w] = blank
        output[y:y+h,x:x+w] = tile
    tray = np.full((18,13),blank,dtype=np.int8)
    for y,x,tile in ((0,0,bar_tile),(4,0,bar_tile),(8,0,ring_tile),
                     (8,4,ring_tile),(12,0,open_tile),(12,4,open_tile)):
        h,w = tile.shape
        tray[y:y+h,x:x+w] = tile
    inp = np.concatenate((board,np.full((18,1),sep,dtype=np.int8),tray),axis=1)
    return inp,output


def self_test():
    signatures,features,normalized = set(),[],set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_digest(p))
        for pair in p['train']+p['test']:
            parsed = _analyze(pair['input'])
            features.append(_features(parsed))
            roles = {}
            a = pair['input']
            normalized.add((a.shape,bytes(roles.setdefault(int(c),len(roles)) for c in a.flat)))
    assert len(signatures) == 42 and len(normalized) > 150
    assert len({f['shape'] for f in features}) >= 20
    assert len({f['dimensions'] for f in features}) >= 30
    assert len({f['used'] for f in features}) >= 4
    assert len({f['roles'] for f in features}) >= 100
    assert len({f['side'] for f in features}) == 4
    assert any(f['repeats'] for f in features)
    assert any(f['surplus'] > 1 for f in features)
    assert all(f['no_a_differs'] and f['no_b_ambiguous'] for f in features)
    assert _digest(generate_puzzle(99,2,1)) == _digest(generate_puzzle(99,2,1))
    # Hand-written topology regressions, independent of shape construction.
    diagonal = np.array([[0,1,1],[1,0,1],[1,1,1]],dtype=bool)
    assert _holes(diagonal) == 1
    diagonal[0,1] = False
    assert _holes(diagonal) == 0
    assert _holes(np.array([[1,0,1],[1,0,1],[1,1,1]],dtype=bool)) == 0
    inp,out = _three_region_fixture()
    assert len(_parse(inp)['regions']) == 3
    assert np.array_equal(solve(inp.tolist()),out)
    # A permutation of all color roles cannot change the geometry of the answer.
    palette = np.array([3,5,0,8,6,1,9,2,4,7],dtype=np.int8)
    assert np.array_equal(solve(palette[inp]),palette[out])
    # Inventory, matching and rendered uniqueness regressions.
    labels = np.zeros((1,4),dtype=int)
    tile = np.array([[3,4]],dtype=np.int8)
    assert not _pack(labels,[(0,tile)])[0]
    assert len(_pack(labels,[(0,tile),(0,tile)])[0]) == 1
    assert not _pack(labels,[(1,tile),(1,tile)])[0]
    assert len(_pack(labels,[(0,tile),(0,np.array([[5,6]],dtype=np.int8))])[0]) == 2
    same = [(0,np.full((1,4),3,dtype=np.int8)),(0,np.full((1,2),3,dtype=np.int8)),
            (0,np.full((1,2),3,dtype=np.int8))]
    assert len(_pack(labels,same)[0]) == 1
    for nt,ne in ((2,1),(10,4)):
        validate_generated_puzzle(generate_puzzle(700+nt,nt,ne))
    fallback = {'train':[_fallback(np.random.default_rng(600+i),1,i%4,i) for i in range(10)],
                'test':[_fallback(np.random.default_rng(700+i),2,i%4,i+10) for i in range(4)]}
    validate_generated_puzzle(fallback)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Validator must reject changed payloads and changed board margins alike.
    p = generate_puzzle(123,2,1)
    for occupied in (True,False):
        bad = {s:[{k:np.array(v,copy=True) for k,v in pair.items()} for pair in p[s]] for s in p}
        q = bad['test'][0]
        parsed = _parse(q['input'])
        pos = tuple(np.argwhere((parsed['labels'] >= 0) == occupied)[0])
        q['output'][pos] = (int(q['output'][pos])+1)%10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('Corrupted output accepted')
    return {'seeds':42,'pairs':210,'distinct_puzzles':len(signatures),
            'normalized_inputs':len(normalized),'output_shapes':len({f['shape'] for f in features}),
            'tile_counts':sorted({f['used'] for f in features}),
            'role_assignments':len({f['roles'] for f in features}),
            'both_parent_ablation_witnesses':len(features)}


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(",", ":")))
