"""Cavity abacus: deal color-matched object counts among enclosed chambers.

Only NumPy and the standard library are used. Nothing is read from disk.
"""
from __future__ import annotations

import json

import numpy as np


_FOUR = ((-1, 0), (0, -1), (0, 1), (1, 0))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_CROSS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
_X = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    steps = _EIGHT if diagonal else _FOUR
    answer = []
    h, w = mask.shape
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        answer.append(cells)
    return answer


def _bounds(cells):
    a = np.asarray(cells, dtype=int)
    lo, hi = a.min(axis=0), a.max(axis=0) + 1
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def _holes(mask, diagonal=False):
    """Local coordinates, with a real exterior supplied beyond every edge."""
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    h, w = padded.shape
    return [tuple((r - 1, c - 1) for r, c in cells)
            for cells in _components(~padded, diagonal)
            if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells)]


def _allocation(n, k):
    q, remainder = divmod(n, k)
    return tuple(q + (i < remainder) for i in range(k))


def _check_grid(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _parse(input_grid):
    """Recover roles, chambers, and loose components solely from pixels."""
    g = _check_grid(input_grid)
    bg = int(g[0, 0])
    assert all(int(g[r, c]) == bg for r, c in
               ((0, -1), (-1, 0), (-1, -1))), 'Background corners must agree'
    candidates = []
    for wall in map(int, np.unique(g)):
        if wall == bg:
            continue
        devices, wall_total = [], 0
        good = True
        for cells in _components(g == wall):
            wall_total += len(cells)
            r0, r1, c0, c1 = _bounds(cells)
            mask = np.zeros((r1-r0, c1-c0), dtype=bool)
            for r, c in cells:
                mask[r-r0, c-c0] = True
            holes = _holes(mask)
            if not holes:
                good = False
                break
            chambers, markers = [], []
            for hole in holes:
                points = tuple((r+r0, c+c0) for r, c in hole)
                marks = [(r, c) for r, c in points if int(g[r, c]) != bg]
                if len(marks) > 1:
                    good = False
                    break
                # An unmarked chamber still supplies a destination. The
                # canonical origin is its upper-left cell in reading order.
                anchor = min(points)
                if marks and marks[0] != anchor:
                    good = False
                    break
                markers.extend(marks)
                chambers.append(dict(cells=frozenset(points), anchor=anchor))
            if not good or len(markers) != 1:
                good = False
                break
            chambers.sort(key=lambda x: x['anchor'])
            devices.append(dict(cells=tuple(cells), mask=mask, bounds=(r0,r1,c0,c1),
                                color=int(g[markers[0]]), marker=markers[0], chambers=chambers))
        if not good or not devices or len({d['color'] for d in devices}) != len(devices):
            continue
        excluded = np.zeros(g.shape, dtype=bool)
        for d in devices:
            for r, c in d['cells']:
                excluded[r, c] = True
            for chamber in d['chambers']:
                for r, c in chamber['cells']:
                    excluded[r, c] = True
        sources, counts = {}, {}
        for color in map(int, np.unique(g)):
            if color in (bg, wall):
                continue
            comp = _components((g == color) & ~excluded, True)
            sources[color] = comp
            counts[color] = len(comp)
        if any(counts.get(d['color'], 0) < 1 for d in devices):
            continue
        # Destinations must be inside the same chamber, with a blank terminal cell.
        for d in devices:
            n, k = counts[d['color']], len(d['chambers'])
            d['n'], d['allocation'] = n, _allocation(n, k)
            for chamber, amount in zip(d['chambers'], d['allocation']):
                r, c = chamber['anchor']
                if amount and any((r, c + j) not in chamber['cells']
                                  for j in range(2*amount)):
                    good = False
                    break
        if good:
            devices.sort(key=lambda d: d['bounds'])
            candidates.append(dict(background=bg, wall=wall, devices=devices,
                                   sources=sources, counts=counts, excluded=excluded))
    assert len(candidates) == 1, 'Need one keyed hollow-device interpretation'
    return candidates[0]


def _answer_from_meta(meta):
    cells = [p for d in meta['devices'] for p in d['cells']]
    r0, r1, c0, c1 = _bounds(cells)
    out = np.full((r1-r0, c1-c0), meta['background'], dtype=np.int8)
    for d in meta['devices']:
        for r, c in d['cells']:
            out[r-r0, c-c0] = meta['wall']
        for ch, n in zip(d['chambers'], d['allocation']):
            r, c = ch['anchor']
            for j in range(n):
                out[r-r0, c-c0+2*j] = d['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _answer_from_meta(_parse(input_grid))


def _normalize(cells):
    r0, _, c0, _ = _bounds(cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _source_shape(rng, rich=True):
    mode = int(rng.integers(8 if rich else 5))
    if mode < 4:
        cells = (_DIAMOND, _SQUARE, _CROSS, _X)[mode]
    elif mode == 4:
        cells = tuple((0, c) for c in range(int(rng.integers(1, 5))))
    else:
        side = int(rng.integers(3, 5))
        cells = {(1, 1)}
        target = int(rng.integers(3, 10))
        for _ in range(target-1):
            frontier = sorted({(r+dr,c+dc) for r,c in cells for dr,dc in _EIGHT
                               if 0 <= r+dr < side and 0 <= c+dc < side} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    cells = list(cells)
    if rng.random() < .5:
        cells = [(r, -c) for r,c in cells]
    for _ in range(int(rng.integers(4))):
        cells = [(-c, r) for r,c in cells]
    return _normalize(cells)


def _chamber(rng, capacity, rich, force_rect=False):
    lead = 0
    w = lead + 2*capacity + int(rng.integers(0, 3) if rich else 0)
    h = int(rng.integers(1, 5) if rich else rng.integers(1, 3))
    lane = 0  # Upper-left chamber cell is an input-visible tally origin.
    hole = np.ones((h, w), dtype=bool)
    if rich and not force_rect and h > 1:
        center = w//2
        for r in range(h):
            if r == lane:
                continue
            left = int(rng.integers(center+1))
            right = int(rng.integers(center+1, w+1))
            hole[r, :left] = False
            hole[r, right:] = False
    return hole, (lane, lead)


def _make_device(rng, k, n, rich, *, bay=False, diagonal=False, compact=False):
    cap = (n+k-1)//k
    if compact:
        prototypes = [(np.ones((1, 2*cap), bool), (0,0)) for _ in range(k)]
        groups, margin, gap = [list(range(k))], 1, 1
    else:
        prototypes = [_chamber(rng, cap, rich, force_rect=diagonal and i == 0)
                      for i in range(k)]
        rows = int(rng.integers(1, min(k, 3)+1))
        # Random compositions of k; no permanent grid of slots.
        cuts = sorted(map(int, rng.choice(np.arange(1,k), rows-1, replace=False)))
        groups = [list(range(a,b)) for a,b in zip([0]+cuts, cuts+[k])]
        margin, gap = 1, int(rng.integers(1, 3))
        # Split any over-wide row. A single chamber always fits these bounds.
        changed = []
        for group in groups:
            current, width = [], 0
            for i in group:
                sw = prototypes[i][0].shape[1]
                if current and width+gap+sw+2*margin > 25:
                    changed.append(current)
                    current, width = [], 0
                width += (gap if current else 0)+sw
                current.append(i)
            changed.append(current)
        groups = changed
    if diagonal:
        # The first, smaller chamber receives the remainder. This explicitly
        # distinguishes reading-order dealing from allocation by cavity area.
        assert k == 2
        prototypes = [(np.ones((1,2*cap),bool),(0,0)),
                      (np.ones((2,2*cap),bool),(0,0))]
    widths = [sum(prototypes[i][0].shape[1] for i in g)+gap*(len(g)-1) for g in groups]
    heights = [max(prototypes[i][0].shape[0] for i in g) for g in groups]
    h, w = sum(heights)+gap*(len(groups)-1)+2*margin, max(widths)+2*margin
    wall = np.ones((h, w), dtype=bool)
    anchors, cavity_sets = [], []
    r = margin
    for group_index, (group, height, width) in enumerate(zip(groups, heights, widths)):
        c = margin if group_index == 0 and diagonal else margin+int(rng.integers(w-2*margin-width+1))
        for i in group:
            hole, (kr,kc) = prototypes[i]
            hh, ww = hole.shape
            rr = r if i == 0 and diagonal else r+int(rng.integers(height-hh+1))
            wall[rr:rr+hh,c:c+ww][hole] = False
            anchors.append((rr+kr,c+kc))
            cavity_sets.append(frozenset((rr+int(y),c+int(x)) for y,x in np.argwhere(hole)))
            c += ww+gap
        r += height+gap
    if diagonal:
        # First hole includes (1,1), but only a diagonal path leads outside.
        wall[0,0] = False
    if bay:
        # Open downward prongs: a deep exterior bay, never an extra chamber.
        depth = 2 if compact else int(rng.integers(2, 5))
        left = max(0, (w-5)//2)
        right = min(w-1, left+4)
        extra = np.zeros((depth,w), dtype=bool)
        extra[:,left] = extra[:,right] = True
        wall = np.vstack((wall,extra))
    if rich and not compact:
        # Sculpt only topology-preserving exterior pixels; keyed holes stay exact.
        for _ in range(10):
            candidate = wall.copy()
            side = int(rng.integers(4))
            reach = int(rng.integers(1, 4))
            if side < 2:
                c = int(rng.integers(w))
                if side == 0:
                    candidate[:reach,c] = False
                else:
                    candidate[-reach:,c] = False
            else:
                rr = int(rng.integers(wall.shape[0]))
                if side == 2:
                    candidate[rr,:reach] = False
                else:
                    candidate[rr,-reach:] = False
            if candidate.any() and len(_components(candidate)) == 1:
                actual = {frozenset(x) for x in _holes(candidate)}
                if actual == set(cavity_sets):
                    wall = candidate
    assert len(_components(wall)) == 1
    assert {frozenset(x) for x in _holes(wall)} == set(cavity_sets)
    if diagonal:
        assert len(_holes(wall, True)) < k
    # Retain blank outer mask padding only if occupied extents actually need it.
    y0,y1,x0,x1 = _bounds(np.argwhere(wall))
    wall = wall[y0:y1,x0:x1].copy()
    anchors = sorted((r-y0,c-x0) for r,c in anchors)
    marker_index = k-1 if n < k or diagonal else int(rng.integers(k))
    return dict(mask=wall, anchors=anchors, marker_index=marker_index, n=n, k=k)


def _positions(free, h, w):
    if h > free.shape[0] or w > free.shape[1]:
        return []
    # Integral image of forbidden cells.
    s = np.pad((~free).astype(int), ((1,0),(1,0))).cumsum(0).cumsum(1)
    return np.argwhere(s[h:,w:]-s[:-h,w:]-s[h:,:-w]+s[:-h,:-w] == 0)


def _reserve(free, r, c, h, w):
    free[max(0,r-1):r+h+1,max(0,c-1):c+w+1] = False


def _put_device(grid, free, device, color, wall, rng, position=None):
    mask = device['mask']
    h,w = mask.shape
    if position is None:
        choices = _positions(free,h,w)
        if not len(choices):
            return None
        r,c = map(int, choices[int(rng.integers(len(choices)))])
    else:
        r,c = position
    grid[r:r+h,c:c+w][mask] = wall
    y,x = device['anchors'][device['marker_index']]
    grid[r+y,c+x] = color
    _reserve(free,r,c,h,w)
    return r,c


def _put_source(grid, free, shape, color, rng, position=None):
    h,w = 1+max(r for r,c in shape), 1+max(c for r,c in shape)
    if position is None:
        choices = _positions(free,h,w)
        if not len(choices):
            return False
        r,c = map(int, choices[int(rng.integers(len(choices)))])
    else:
        r,c = position
    for y,x in shape:
        grid[r+y,c+x] = color
    _reserve(free,r,c,h,w)
    return True


def _construct_output(bg, wall_color, devices, locations, colors):
    """Labels use sampled supply counts, not solve or the visible-grid parser."""
    cells = [(r+int(y),c+int(x)) for d,(r,c) in zip(devices,locations)
             for y,x in np.argwhere(d['mask'])]
    r0,r1,c0,c1 = _bounds(cells)
    out = np.full((r1-r0,c1-c0), bg, dtype=np.int8)
    for d,(r,c),color in zip(devices,locations,colors):
        for y,x in np.argwhere(d['mask']):
            out[r+int(y)-r0,c+int(x)-c0] = wall_color
        for (y,x),amount in zip(d['anchors'],_allocation(d['n'],d['k'])):
            for i in range(amount):
                out[r+y-r0,c+x-c0+2*i] = color
    return out


def _make_pair(rng, specs, colors, bg, wall, distractors, rich, witness=False,
               force_compact=False):
    sources = []
    for ci,(color,(_,n)) in enumerate(zip(colors,specs)):
        for i in range(n):
            shape = (_DIAMOND,_SQUARE,((0,0),))[i] if witness and ci == 0 and i < 3 else _source_shape(rng,rich)
            sources.append((color,shape))
    for color,n in distractors:
        for _ in range(n):
            sources.append((color,_source_shape(rng,rich)))
    rng.shuffle(sources)
    for attempt in ((24,) if force_compact else range(25)):
        compact = attempt == 24
        devices = [_make_device(rng,k,n,rich and not compact,
                                bay=witness and i == 0, diagonal=witness and i == 0,
                                compact=compact) for i,(k,n) in enumerate(specs)]
        if compact:
            h=w=30
        else:
            h = int(rng.integers(20,31)) if attempt < 12 else 30
            w = int(rng.integers(20,31)) if attempt < 12 else 30
        grid = np.full((h,w),bg,dtype=np.int8)
        free = np.ones((h,w),dtype=bool)
        free[0,0] = free[0,-1] = free[-1,0] = free[-1,-1] = False
        locations = [None]*len(devices)
        if compact:
            row = 1
            for i,d in enumerate(devices):
                dh,dw = d['mask'].shape
                col = int(rng.integers(30-dw+1))
                locations[i] = _put_device(grid,free,d,colors[i],wall,rng,(row,col))
                row += dh+1
            slots = [(r,c) for r in range(row,28,4) for c in range(0,28,4)]
            assert len(slots) >= len(sources), 'Finite fallback supply capacity'
            rng.shuffle(slots)
            for (color,shape),position in zip(sources,slots):
                if max(max(p) for p in shape) > 2:
                    shape = _CROSS
                _put_source(grid,free,shape,color,rng,position)
        else:
            order = sorted(range(len(devices)),key=lambda i: -devices[i]['mask'].size)
            for i in order:
                locations[i] = _put_device(grid,free,devices[i],colors[i],wall,rng)
                if locations[i] is None:
                    break
            if any(x is None for x in locations):
                continue
            objects = sorted(sources,key=lambda item: -(1+max(r for r,c in item[1]))*(1+max(c for r,c in item[1])))
            if not all(_put_source(grid,free,shape,color,rng) for color,shape in objects):
                continue
        out = _construct_output(bg,wall,devices,locations,colors)
        assert np.array_equal(solve(grid),out), 'Placement altered the intended answer'
        return {'input':grid,'output':out}
    raise AssertionError('Unreachable finite fallback')


def _integer(value, low, high, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= value <= high:
        raise ValueError(name+' must be an integer in '+str((low,high)))
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train,2,10,'num_train')
    nq = _integer(num_test,1,4,'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = list(map(int,rng.permutation(10)))
    stable = bool(rng.random() < .25)
    regime = int(rng.integers(3))  # larger counts, more chambers, richer geometry
    result,seen = {'train':[],'test':[]},set()
    for split,count in (('train',nt),('test',nq)):
        for index in range(count):
            witness = split == 'train' and index == 0
            second = split == 'train' and index == 1
            if witness:
                specs,colors = [(2,5),(3,5)],palette[2:4]
                distractors = [(palette[4],int(rng.integers(1,4)))]
            elif second:
                specs,colors = [(3,2),(1,3)],palette[3:5]
                distractors = [(palette[2],int(rng.integers(1,4)))]
            else:
                nd = int(rng.integers(1,4))
                colors = list(map(int,rng.choice(palette[2:] if stable else palette,nd,replace=False)))
                specs = [(int(rng.integers(1,4)),int(rng.integers(1,7))) for _ in colors]
                if split == 'test':
                    if regime == 0:
                        specs[0] = (int(rng.integers(2,4)),int(rng.integers(7,11)))
                    elif regime == 1:
                        specs[0] = (int(rng.integers(4,6)),int(rng.integers(2,7)))
                    else:
                        specs[0] = (int(rng.integers(2,4)),int(rng.integers(2,7)))
                while sum(n for k,n in specs) > 18:
                    j = next(j for j in range(len(specs)-1,-1,-1) if specs[j][1] > 1)
                    k,n = specs[j]
                    specs[j] = k,n-1
                available = [c for c in palette[2:] if c not in colors] if stable else [c for c in palette if c not in colors]
                rng.shuffle(available)
                distractors = [(c,int(rng.integers(1,3))) for c in available[:int(rng.integers(0,3))]]
            occupied = set(colors) | {c for c,n in distractors}
            structural = [c for c in palette if c not in occupied]
            if stable:
                bg,wall = palette[:2]
            else:
                bg,wall = map(int,rng.choice(structural,2,replace=False))
            rich = split == 'test' or regime != 2 or witness or second
            for _ in range(8):
                pair = _make_pair(rng,specs,colors,bg,wall,distractors,rich,witness)
                signature = _grid_key(pair['input'])
                if signature not in seen:
                    break
            else:
                raise AssertionError('Distinct-input retries exhausted')
            seen.add(signature)
            result[split].append(pair)
    rng.shuffle(result['train'])
    validate_generated_puzzle(result)
    return result


def _grid_key(grid):
    a = np.asarray(grid,dtype=np.int8)
    return a.shape,a.tobytes()


def _features(g):
    m = _parse(g)
    devices = m['devices']
    sources = [obj for objects in m['sources'].values() for obj in objects]
    keyed = frozenset(d['color'] for d in devices)
    diagonal = any(len(_components(_mask(obj))) > 1
                   for c in keyed for obj in m['sources'][c])
    heterogeneous = any({_normalize(obj) for obj in m['sources'][c]} >=
                        {_DIAMOND,_SQUARE,((0,0),)} for c in keyed)
    area_witness = any(any(a > b and len(ca['cells']) < len(cb['cells'])
                          for ca,a in zip(d['chambers'],d['allocation'])
                          for cb,b in zip(d['chambers'],d['allocation'])) for d in devices)
    m.update(diagonal_source=diagonal,
             heterogeneous_source=heterogeneous, area_witness=area_witness,
             diagonal_chamber=any(len(_holes(d['mask'],True)) < len(d['chambers']) for d in devices),
             geometry=tuple(_normalize(obj) for obj in sources),
             holes=tuple(len(d['chambers']) for d in devices),
             totals=tuple(d['n'] for d in devices),
             allocations=tuple(d['allocation'] for d in devices),
             keyed=keyed)
    m['ignored'] = frozenset(c for c,objects in m['sources'].items() if objects and c not in m['keyed'])
    return m


def _mask(cells):
    r0,r1,c0,c1 = _bounds(cells)
    a = np.zeros((r1-r0,c1-c0),bool)
    for r,c in cells:
        a[r-r0,c-c0] = True
    return a


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    nt,nq = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen,training = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _check_grid(pair['input']),_check_grid(pair['output'])
            signature = _grid_key(inp)
            assert signature not in seen
            seen.add(signature)
            m = _features(inp)
            assert np.array_equal(_answer_from_meta(m),out)
            # Conserved item count, balanced loads, fixed order for the remainder.
            for d in m['devices']:
                a = d['allocation']
                assert sum(a) == d['n'] and max(a)-min(a) <= 1
                assert tuple(sorted(a,reverse=True)) == a
                assert len(_components(d['mask'])) == 1
                # Source bounding boxes cannot conceal objects inside a cavity.
                assert all(len(ch['cells']) >= 1 for ch in d['chambers'])
            # Different physical objects cannot touch, even at corners.
            owners = np.full(inp.shape,-1,int)
            physical = [d['cells'] for d in m['devices']]
            physical += [obj for objects in m['sources'].values() for obj in objects]
            for i,obj in enumerate(physical):
                for r,c in obj:
                    owners[r,c] = i
            for r,c in np.argwhere(owners >= 0):
                block = owners[max(0,r-1):r+2,max(0,c-1):c+2]
                assert np.all((block < 0) | (block == owners[r,c]))
            if split == 'train':
                training.append(m)
    assert any(m['diagonal_source'] and m['diagonal_chamber'] for m in training)
    assert any(m['heterogeneous_source'] and m['area_witness'] for m in training)
    assert any(any(0 in d['allocation'] and
                   d['allocation'][next(i for i,ch in enumerate(d['chambers'])
                                        if ch['anchor'] == d['marker'])] == 0
                   for d in m['devices']) for m in training)
    assert any(any(max(a) >= 3 for a in m['allocations']) for m in training)
    assert any(len({k for k,n in zip(m['holes'],m['totals']) if n == 5}) >= 2 for m in training)
    assert any(a['keyed'] & b['ignored'] for a in training for b in training)
    # Explicit omission checks: one mark per chamber, one destination per
    # visible key (only one!), and ceiling-rounded loads all disagree.
    assert any(any(any(x != 1 for x in a) for a in m['allocations']) for m in training)
    assert any(any(any(x != d['n'] for x in d['allocation']) for d in m['devices']) for m in training)
    assert any(any(sum(x > 0 for x in d['allocation']) > 1 for d in m['devices']) for m in training)
    assert any(any(len(set(a)) > 1 for a in m['allocations']) for m in training)


def self_test():
    # Hand-specified connectivity/topology witnesses, including a true opening.
    assert len(_components(_mask(_DIAMOND),True)) == 1
    assert len(_components(_mask(_DIAMOND))) == 4
    diagonal = np.array([[0,1,1],[1,0,1],[1,1,1]],bool)
    assert len(_holes(diagonal)) == 1 and len(_holes(diagonal,True)) == 0
    diagonal[0,1] = False
    assert not _holes(diagonal)
    assert _allocation(5,2) == (3,2) and _allocation(5,3) == (2,2,1)
    assert _allocation(2,3) == (1,1,0) and _allocation(6,3) == (2,2,2)
    signatures,shapes,roles,geometries,ks,ns,layouts = set(),set(),set(),set(),set(),set(),set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_grid_key(p['input']) for s in ('train','test') for p in puzzle[s]))
        local_shapes = set()
        for split in ('train','test'):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair['input'].copy()),pair['output'])
                m = _features(pair['input'])
                shapes.add(pair['input'].shape)
                local_shapes.add(pair['input'].shape)
                roles.add((m['background'],m['wall']))
                geometries.update(m['geometry'])
                ks.update(m['holes']); ns.update(m['totals'])
                layouts.update((d['mask'].shape,tuple(ch['anchor'][0]-d['bounds'][0] for ch in d['chambers'])) for d in m['devices'])
        assert len(local_shapes) >= 2
    assert len(signatures) == 42 and len(shapes) >= 25 and len(roles) >= 30
    assert len(geometries) >= 60 and len(layouts) >= 60
    assert ks >= {1,2,3,4,5} and ns >= set(range(1,10))
    # Exercise the finite fallback directly, including a population of ten.
    for witness,specs in ((True,[(2,5),(3,5)]),
                          (False,[(2,10),(5,6),(1,2)])):
        pair = _make_pair(np.random.default_rng(909),specs,[2,3,4][:len(specs)],
                          0,5,[(6,2)],True,witness,force_compact=True)
        assert np.array_equal(solve(pair['input']),pair['output'])
        meta = _features(pair['input'])
        assert max(meta['totals']) == max(n for k,n in specs)
        if witness:
            assert meta['diagonal_source'] and meta['diagonal_chamber'] and meta['area_witness']
    a,b = generate_puzzle(123,2,1),generate_puzzle(123,2,1)
    assert all(np.array_equal(p[k],q[k]) for s in a for p,q in zip(a[s],b[s]) for k in p)
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(190+nt*4+nq,nt,nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt,nq in ((1,1),(11,1),(4,0),(5,5),(2,2),(3,4),(True,1),(3.5,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Wrong labels must fail validation.
    a['train'][0]['output'][0,0] = (int(a['train'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(a)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test':'passed','seeds':42},separators=(',',':')))
