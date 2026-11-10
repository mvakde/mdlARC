"""Topology-gated printing: chamber beads address copies of a topology key.

Only Python's standard library and NumPy are used. No parent code or data is
loaded. Generation keeps construction records locally; solve reads pixels only.
"""

import numpy as np


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
                    stack.append(p)
        result.append(cells)
    return result


def _holes(mask):
    """Enclosed regions in local coordinates; one-cell exterior padding."""
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    return [[(r-1, c-1) for r, c in cells]
            for cells in _components(~padded)
            if all(0 < r < padded.shape[0]-1 and
                   0 < c < padded.shape[1]-1 for r, c in cells)]


def _local_mask(cells):
    coords = np.asarray(cells, dtype=int)
    lo, hi = coords.min(0), coords.max(0) + 1
    mask = np.zeros(tuple(hi-lo), dtype=bool)
    mask[tuple((coords-lo).T)] = True
    return mask, tuple(map(int, lo))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _decode(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 4, "four visible color roles required"
    assert counts.max() > sorted(counts)[-2], "ambiguous background"
    bg = int(values[np.argmax(counts)])
    comps = {int(v): _components(grid == v) for v in values if v != bg}
    inks = [v for v, groups in comps.items()
            if len(groups) == 2 and sorted(map(len, groups))[0] == 1
            and sorted(map(len, groups))[1] >= 3]
    assert len(inks) == 1, "one glyph and one exceptional bead required"
    ink = inks[0]
    bead_roles = [v for v, groups in comps.items() if v != ink
                  and len(groups) >= 2 and all(len(x) == 1 for x in groups)]
    assert len(bead_roles) == 1, "ordinary beads must be isolated"
    bead = bead_roles[0]
    wall = next(v for v in comps if v not in (ink, bead))
    assert all(len(c) >= 2 for c in comps[wall])
    glyph_cells = next(x for x in comps[ink] if len(x) > 1)
    anchor = next(x[0] for x in comps[ink] if len(x) == 1)
    mask, top = _local_mask(glyph_cells)
    h, w = mask.shape
    k = len(_holes(mask))
    assert k >= 1
    points = {x[0] for x in comps[bead]} | {anchor}
    assert all((r-anchor[0]) % 2 == 0 and (c-anchor[1]) % 2 == 0
               for r, c in points), "beads must share pitch-two parity"
    selected, closed, groups, ownership = set(), set(), [], {}
    for cells in comps[wall]:
        obj, (r0, c0) = _local_mask(cells)
        regions = _holes(obj)
        sites, areas = [], []
        for region in regions:
            absolute = {(r+r0, c+c0) for r, c in region}
            assert all(int(grid[p]) in (bg, bead, ink) for p in absolute)
            assert all(grid[p] != ink or p == anchor for p in absolute), \
                "the glyph cannot be nested inside a wall"
            marked = absolute & points
            assert len(marked) == 1, "each closed chamber has one bead"
            point = next(iter(marked))
            assert point not in ownership, "nested wall chambers excluded"
            ownership[point] = len(groups)
            sites.append(point)
            areas.append(len(region))
        closed.update(sites)
        if len(regions) == k:
            selected.update(sites)
        groups.append(dict(cells=cells, mask=obj, top=(r0, c0),
                           count=len(regions), sites=tuple(sites),
                           areas=tuple(areas)))
    assert anchor in selected, "the anchor must be in an admitted chamber"
    assert len(selected) >= 2
    # Separate the entire address field from the glyph; this is a presentation
    # convention, unlike the enclosed-region test above.
    field_cells = [p for g in groups for p in g['cells']] + sorted(points)
    coords = np.asarray(field_cells)
    lo, hi = coords.min(0), coords.max(0)
    tr, tc = top
    assert (hi[0]+1 < tr or tr+h < lo[0] or
            hi[1]+1 < tc or tc+w < lo[1]), "glyph/blueprint separation required"
    origins = {}
    for r, c in selected:
        rr = tr + ((r-anchor[0])//2)*h
        cc = tc + ((c-anchor[1])//2)*w
        assert 0 <= rr and rr+h <= grid.shape[0]
        assert 0 <= cc and cc+w <= grid.shape[1], "expansion must fit"
        origins[(r, c)] = (rr, cc)
    ar = min(r for r, c in selected)
    ac = min(c for r, c in selected)
    coarse = tuple(sorted(((r-ar)//2, (c-ac)//2) for r, c in selected))
    return dict(grid=grid, bg=bg, ink=ink, bead=bead, wall=wall,
                mask=mask, top=top, k=k, anchor=anchor, points=points,
                groups=groups, closed=closed, selected=selected,
                open=points-closed, origins=origins, coarse=coarse,
                normalized_anchor=((anchor[0]-ar)//2, (anchor[1]-ac)//2))


def _render(meta):
    out = np.full(meta['grid'].shape, meta['bg'], dtype=np.int8)
    mr, mc = np.nonzero(meta['mask'])
    for p, (r, c) in meta['origins'].items():
        color = meta['bead'] if p == meta['anchor'] else meta['ink']
        out[r+mr, c+mc] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive classification, containment, lattice and alignment from input."""
    return _render(_decode(input_grid))


def _valid_shape(mask, k, closed_points=(), open_points=()):
    if mask.sum() < 3 or len(_components(mask)) != 1:
        return False
    holes = _holes(mask)
    if len(holes) != k:
        return False
    regions = [set(cells) for cells in holes]
    if closed_points and any(len(set(closed_points) & hole) != 1
                             for hole in regions):
        return False
    enclosed = set().union(*regions) if regions else set()
    coords = np.argwhere(mask)
    lo,hi = coords.min(0),coords.max(0)
    return (set(closed_points) <= enclosed and
            not (set(open_points) & enclosed) and
            all(lo[0] <= r <= hi[0] and lo[1] <= c <= hi[1]
                for r,c in open_points) and
            all(not mask[p] for p in closed_points + open_points))


def _sculpt(rng, mask, k, closed_points=(), open_points=(), rich=False,
            preserve_corners=False, preserve_box=False, hole_growth=True):
    """Grow existing holes and carve exterior cells, preserving topology."""
    a = mask.copy()
    # Only interior foreground cells adjacent to a hole are hole-growth edits.
    for _ in range((18 if rich else 5) if hole_growth else 0):
        holes = _holes(a)
        neighbors = set()
        for region in holes:
            for r, c in region:
                for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if (0 < p[0] < a.shape[0]-1 and
                            0 < p[1] < a.shape[1]-1 and a[p]):
                        neighbors.add(p)
        if not neighbors:
            break
        p = _pick(rng, sorted(neighbors))
        candidate = a.copy()
        candidate[p] = False
        if _valid_shape(candidate, k, closed_points, open_points):
            a = candidate
    if not preserve_corners:
        for _ in range(10 if rich else 3):
            # Candidate cells include edges of open bays as well as the box.
            exterior = ~np.pad(a, 1)
            holes = _holes(a)
            for region in holes:
                for r, c in region:
                    exterior[r+1, c+1] = False
            options = []
            for r, c in np.argwhere(a):
                if any(exterior[r+1+dr, c+1+dc]
                       for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))):
                    options.append((int(r), int(c)))
            if not options:
                break
            candidate = a.copy()
            candidate[_pick(rng, options)] = False
            if (not preserve_box or (candidate[0].any() and candidate[-1].any()
                                     and candidate[:,0].any() and candidate[:,-1].any())):
                if _valid_shape(candidate, k, closed_points, open_points):
                    a = candidate
    return a


def _glyph(rng, k, dims, informative=False, rich=False):
    h, w = dims
    mask = np.ones((h, w), dtype=bool)
    slots = [(r, c) for r in range(1, h-1, 2) for c in range(1, w-1, 2)]
    chosen = rng.choice(len(slots), size=k, replace=False)
    for i in chosen:
        mask[slots[int(i)]] = False
    if informative:
        # Exactly one corner notch leaves three full corners. For a rectangular
        # mask this distinguishes flips, half turns and transposition. Choose
        # one diagonally adjacent to a cavity, teaching four-connectivity.
        corners = ((0,0),(0,w-1),(h-1,0),(h-1,w-1))
        corners = [p for p in corners if any(
            abs(p[0]-slots[int(i)][0]) == 1 and abs(p[1]-slots[int(i)][1]) == 1
            for i in chosen)]
        assert corners
        mask[_pick(rng,corners)] = False
    mask = _sculpt(rng, mask, k, rich=rich, preserve_corners=informative,
                   preserve_box=True)
    assert _valid_shape(mask, k)
    assert mask[0].any() and mask[-1].any() and mask[:,0].any() and mask[:,-1].any()
    return mask


def _wall_object(rng, k, open_bay=False, rich=False, simple=False, hole_growth=True):
    needed = k + int(open_bay)
    choices = [(r, c) for r in range(1, 4) for c in range(1, 4)
               if needed <= r*c and r*c <= max(needed+2, 2)]
    if simple:
        choices = [(1, max(1, needed))] if needed <= 3 else [(2, 2)]
    sr, sc = _pick(rng, choices)
    mask = np.ones((2*sr+1, 2*sc+1), dtype=bool)
    slots = [(1+2*r, 1+2*c) for r in range(sr) for c in range(sc)]
    opened = []
    if open_bay:
        boundary = [p for p in slots if p[0] in (1, 2*sr-1)
                    or p[1] in (1, 2*sc-1)]
        p = _pick(rng, boundary)
        opened = [p]
        directions = []
        if p[0] == 1:
            directions.append((-1, 0))
        if p[0] == 2*sr-1:
            directions.append((1, 0))
        if p[1] == 1:
            directions.append((0, -1))
        if p[1] == 2*sc-1:
            directions.append((0, 1))
        dr, dc = _pick(rng, directions)
        mask[p] = mask[p[0]+dr, p[1]+dc] = False
        slots.remove(p)
    chosen = rng.choice(len(slots), size=k, replace=False)
    closed = [slots[int(i)] for i in chosen]
    for p in closed:
        mask[p] = False
    assert _valid_shape(mask, k, closed, opened)
    if not simple:
        mask = _sculpt(rng, mask, k, closed, opened, rich=rich, hole_growth=hole_growth)
    # Rotate a whole component with its chamber addresses. Parity stays odd
    # because both dimensions are odd. The printed glyph does not rotate.
    turns = int(rng.integers(4))
    for _ in range(turns):
        old_w = mask.shape[1]
        mask = np.rot90(mask).copy()
        closed = [(old_w-1-c, r) for r, c in closed]
        opened = [(old_w-1-c, r) for r, c in opened]
    return dict(mask=mask, count=k, closed=closed, open=opened)


def _pack(rng, obj, free, selected, dims, board, records, forced=None):
    mask = obj['mask']
    h, w = mask.shape
    positions = []
    for r in range(0, board.shape[0]-h+1, 2):
        for c in range(0, board.shape[1]-w+1, 2):
            if forced is not None and (r, c) != forced:
                continue
            if not free[r:r+h, c:c+w].all():
                continue
            sites = [(r+a, c+b) for a, b in obj['closed']]
            new_selected = selected + (sites if obj['count'] == dims[2] else [])
            if new_selected:
                coords = np.asarray(new_selected)
                spans = (coords.max(0)-coords.min(0))//2 + 1
                if spans[0]*dims[0] > 28 or spans[1]*dims[1] > 28:
                    continue
            positions.append((r, c))
    if not positions:
        return False
    r, c = _pick(rng, positions)
    board[r:r+h, c:c+w][mask] = 1
    rec = dict(count=obj['count'], mask=mask,
               closed=[(r+a,c+b) for a,b in obj['closed']],
               open=[(r+a,c+b) for a,b in obj['open']])
    records.append(rec)
    if obj['count'] == dims[2]:
        selected.extend(rec['closed'])
    free[max(0,r-1):r+h+1, max(0,c-1):c+w+1] = False
    return True


def _blueprint(rng, k, dims, rich=False, extra_matches=False, simple=False,
               match_limit=None, hole_growth=True):
    board = np.zeros((13, 13), dtype=np.int8)
    free = np.ones_like(board, dtype=bool)
    records, selected = [], []
    # The selected wall has an open bay: membership is not interchangeable
    # with its enclosing object's cavity count.
    mandatory = [(k, True)] + ([(k, False)] if k == 1 else [])
    wrong = _pick(rng, [v for v in (1,2,3,4) if v != k])
    mandatory.append((wrong, False))
    if extra_matches:
        mandatory.insert(1, (k, False))
    for cls, opened in mandatory:
        obj = _wall_object(rng, cls, opened, rich, simple, hole_growth)
        if not _pack(rng, obj, free, selected, (*dims, k), board, records):
            return None
    if not simple:
        for _ in range(int(rng.integers(0, 4))):
            pool = [0,1,2,3,4,k]
            if match_limit is not None and sum(rec['count'] == k for rec in records) >= match_limit:
                pool = [v for v in pool if v != k]
            cls = _pick(rng, pool)
            obj = _wall_object(rng, cls, cls == 0 or rng.random() < .25, rich,
                               hole_growth=hole_growth)
            _pack(rng, obj, free, selected, (*dims,k), board, records)
    points = [p for rec in records for p in rec['closed'] + rec['open']]
    coords = np.asarray([tuple(map(int,p)) for p in np.argwhere(board)] + points)
    lo, hi = coords.min(0), coords.max(0)+1
    board = board[lo[0]:hi[0], lo[1]:hi[1]].copy()
    for rec in records:
        for name in ('closed','open'):
            rec[name] = [(r-int(lo[0]),c-int(lo[1])) for r,c in rec[name]]
    selected = [p for rec in records if rec['count'] == k for p in rec['closed']]
    return board, records, selected


def _fallback_blueprint(rng, k, dims, diagonal=(6,6), extra_matches=False,
                        irregular=False):
    """Guaranteed compact geometry, still produced from masks rather than data."""
    board = np.zeros((13,13), dtype=np.int8)
    free = np.ones_like(board, dtype=bool)
    selected, records = [], []
    # k=2 first teaching example: 3x7 with two holes plus an open bay.
    # k=1 second: diagonal 3x5 and 3x3 walls give a true gap on both axes.
    placement = [(k,True,(0,0))]
    if k == 1:
        placement.append((k,False,(6,6) if irregular else (4,4) if extra_matches else diagonal))
        if extra_matches:
            placement.append((k,False,(6,8)))
    placement.append((1 if k != 1 else 2,False,(10,0)))
    for cls, opened, pos in placement:
        needed = cls + int(opened)
        obj = _wall_object(rng, cls, opened, simple=True)
        # Force horizontal proportions so the bottom wrong-class object fits.
        if obj['mask'].shape[0] > 3:
            obj['mask'] = obj['mask'].T.copy()
            obj['closed'] = [(c,r) for r,c in obj['closed']]
            obj['open'] = [(c,r) for r,c in obj['open']]
        # k=3 needs a 2x2 chamber group; use the separate general fallback below.
        if needed >= 4:
            obj = dict(mask=np.ones((5,5),dtype=bool), count=cls,
                       closed=[(1,1),(1,3),(3,1)][:cls], open=[])
            for p in obj['closed']:
                obj['mask'][p] = False
            if opened:
                obj['open'] = [(3,3)]
                obj['mask'][3,3] = obj['mask'][4,3] = False
        if cls == 2 and opened:
            obj = dict(mask=np.ones((3,7),dtype=bool),count=2,
                       closed=[(1,1),(1,3)],open=[(1,5)])
            for p in ((1,1),(1,3),(1,5),(0,5)):
                obj['mask'][p] = False
        if cls == 1 and opened:
            obj = dict(mask=np.ones((3,5),dtype=bool),count=1,
                       closed=[(1,1)],open=[(1,3)])
            for p in ((1,1),(1,3),(0,3)):
                obj['mask'][p] = False
        if irregular and opened:
            assert cls == 1
            obj = dict(mask=np.ones((5,5),dtype=bool),count=1,
                       closed=[(1,1)],open=[(3,3)])
            for p in ((1,1),(1,2),(2,1),(3,3),(4,3)):
                obj['mask'][p] = False
        assert _pack(rng, obj, free, selected, (*dims,k), board, records, pos)
    points = [p for rec in records for p in rec['closed']+rec['open']]
    coords = np.asarray([tuple(map(int,p)) for p in np.argwhere(board)] + points)
    hi = coords.max(0)+1
    return board[:hi[0],:hi[1]].copy(), records, selected


def _place_pair(rng, mask, blueprint, anchor, colors):
    board, records, selected = blueprint
    h, w = mask.shape
    coords = np.asarray(selected)
    low, high = coords.min(0), coords.max(0)
    mh, mw = tuple((high-low)//2+1)
    ah, aw = int(mh*h), int(mw*w)
    bh, bw = board.shape
    tr = ((anchor[0]-int(low[0]))//2)*h
    tc = ((anchor[1]-int(low[1]))//2)*w
    positions = []
    for r in range(-bh-1, ah+2):
        for c in range(-bw-1, aw+2):
            gh = max(ah,r+bh)-min(0,r)
            gw = max(aw,c+bw)-min(0,c)
            if gh > 30 or gw > 30:
                continue
            if (r+bh < tr or tr+h < r or c+bw < tc or tc+w < c):
                positions.append((r,c,gh,gw))
    assert positions, "compact source field must fit beside the glyph"
    if rng.random() < .55:
        area = min(p[2]*p[3] for p in positions)
        positions = [p for p in positions if p[2]*p[3] <= area*1.25]
    r,c,gh,gw = _pick(rng,positions)
    bg, ink, wall, bead = colors
    npoints = sum(len(rec['closed'])+len(rec['open']) for rec in records)
    fg = int(mask.sum()+board.sum()+npoints)
    minimum_area = fg + max(int(board.sum()),int(mask.sum())+1,npoints-1)+1
    choices = [(a,b) for a in range(gh,31) for b in range(gw,31)
               if a*b >= minimum_area]
    assert choices
    if rng.random() < .2:
        gh,gw = (30,30)
    elif rng.random() < .55:
        area = min(a*b for a,b in choices)
        gh,gw = _pick(rng,[p for p in choices if p[0]*p[1] == area])
    else:
        gh,gw = _pick(rng,choices)
    base_h,base_w = max(ah,r+bh)-min(0,r),max(aw,c+bw)-min(0,c)
    top = int(rng.integers(gh-base_h+1))-min(0,r)
    left = int(rng.integers(gw-base_w+1))-min(0,c)
    inp = np.full((gh,gw),bg,dtype=np.int8)
    rr,cc = np.nonzero(mask)
    inp[top+tr+rr,left+tc+cc] = ink
    wr,wc = np.nonzero(board)
    inp[top+r+wr,left+c+wc] = wall
    for rec in records:
        for pr,pc in rec['closed']+rec['open']:
            inp[top+r+pr,left+c+pc] = ink if (pr,pc) == anchor else bead
    # Generation uses constructed chamber records, not the input-only parser.
    out = np.full_like(inp,bg)
    for pr,pc in selected:
        dr,dc = (pr-anchor[0])//2,(pc-anchor[1])//2
        out[top+tr+dr*h+rr,left+tc+dc*w+cc] = bead if (pr,pc) == anchor else ink
    return dict(input=inp,output=out)


def _dimensions(rng, k, previous=None, informative=False, larger=False, limit=6):
    choices = [(h,w) for h in range(3,limit+1) for w in range(3,limit+1)
               if ((h-1)//2)*((w-1)//2) >= k
               and (not informative or h != w)
               and (previous is None or (h != previous[0] and w != previous[1]))
               and (not larger or max(h,w) == 6)]
    return _pick(rng,choices)


def _anchor_candidates(selected, previous=None, first=False, gap=False):
    coords = np.asarray(selected)
    lo,hi = coords.min(0),coords.max(0)
    coarse = [tuple(map(int,(np.asarray(p)-lo)//2)) for p in selected]
    if first and not any((r+1,c) in coarse or (r,c+1) in coarse for r,c in coarse):
        return []
    if gap:
        # Both axis gaps are witnessed by the second example even at counts 2/1.
        for axis in (0,1):
            values = {p[axis] for p in coarse}
            if max(values)-min(values)+1 <= len(values):
                return []
    result = []
    for p,q in zip(selected,coarse):
        if previous is not None and (q[0] == previous[0] or q[1] == previous[1]):
            continue
        if first and q == tuple(map(int,(hi-lo)//4)):
            continue
        result.append(p)
    return result


def _counts(num_train,num_test):
    for name,value,lo,hi in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if (isinstance(value,(bool,np.bool_)) or
                not isinstance(value,(int,np.integer)) or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train),int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,nq = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    stable = rng.random() < .25
    base = tuple(map(int,rng.choice(10,4,replace=False)))
    rich = bool(rng.integers(2))
    puzzle = dict(train=[],test=[])
    used = set()
    previous_dims = previous_anchor = None
    for split,count in (('train',nt),('test',nq)):
        for index in range(count):
            first = split == 'train' and index == 0
            second = split == 'train' and index == 1
            k = (2 if first else 1 if second else
                 3 if split == 'test' and regime == 0 else
                 1 if split == 'test' and regime in (2,3) else
                 int(rng.integers(1,3 if split == 'train' and regime in (0,2) else 4)))
            dims = _dimensions(rng,k,previous_dims if second else None,first,
                               split == 'test' and regime == 1,
                               5 if split == 'train' or regime == 2 else 6)
            glyph = _glyph(rng,k,dims,first,rich or (split == 'test' and regime == 3))
            palette = base if stable else tuple(map(int,rng.choice(10,4,replace=False)))
            extra = split == 'test' and regime == 2
            irregular_shift = split == 'test' and regime == 3
            for attempt in range(64):
                blueprint = _blueprint(rng,k,dims,rich or irregular_shift,extra,
                                       match_limit=2 if split == 'train' and regime == 2 else None,
                                       hole_growth=not (split == 'train' and regime == 3))
                if blueprint is None:
                    continue
                if irregular_shift and not any(
                    len(hole) >= 3 and not _local_mask(hole)[0].all()
                    for rec in blueprint[1] if rec['count'] == k
                    for hole in _holes(rec['mask'])):
                    continue
                candidates = _anchor_candidates(blueprint[2],previous_anchor if second else None,
                                                 first,second)
                if not candidates:
                    continue
                anchor = _pick(rng,candidates)
                pair = _place_pair(rng,glyph,blueprint,anchor,palette)
                if split == 'test' and regime == 3:
                    # A joint transpose preserves every rule, including h/w
                    # scaling. Individual output stamps are never transposed.
                    pair = {name:grid.T.copy() for name,grid in pair.items()}
                if _key(pair['input']) in used:
                    continue
                meta = _decode(pair['input'])
                assert np.array_equal(_render(meta),pair['output'])
                break
            else:
                for diagonal in ((6,6),(4,4),(4,6),(6,4)):
                    blueprint = _fallback_blueprint(rng,k,dims,diagonal,extra,irregular_shift)
                    candidates = _anchor_candidates(blueprint[2],previous_anchor if second else None,
                                                     first,second)
                    if candidates:
                        break
                assert candidates
                pair = _place_pair(rng,glyph,blueprint,_pick(rng,candidates),palette)
                if split == 'test' and regime == 3:
                    pair = {name:grid.T.copy() for name,grid in pair.items()}
                # A finite recoloring fallback guarantees input uniqueness.
                if _key(pair['input']) in used:
                    done = False
                    # 10*9*8*7 distinct role assignments exceed 14 inputs.
                    from itertools import permutations
                    for roles in permutations(range(10),4):
                        remap = np.arange(10,dtype=np.int8)
                        remap[list(palette)] = roles
                        candidate = {name:remap[grid] for name,grid in pair.items()}
                        if _key(candidate['input']) not in used:
                            pair,done = candidate,True
                            break
                    assert done
                meta = _decode(pair['input'])
            puzzle[split].append(pair)
            used.add(_key(pair['input']))
            if first:
                previous_dims = dims
                previous_anchor = meta['normalized_anchor']
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(meta):
    groups = meta['groups']
    mr,mc = meta['mask'].shape
    coarse = meta['coarse']
    all_areas = [a for g in groups for a in g['areas']]
    return dict(shape=meta['grid'].shape, glyph_shape=meta['mask'].shape,
                glyph_mask=_key(meta['mask'].astype(np.int8)), k=meta['k'],
                roles=(meta['bg'],meta['ink'],meta['wall'],meta['bead']),
                layout=coarse, anchor=meta['normalized_anchor'],
                objects=len(groups), copies=len(meta['selected']),
                matches=sum(g['count'] == meta['k'] for g in groups),
                discarded=len(meta['points']-meta['selected']),
                wrong_closed=len(meta['closed']-meta['selected']),
                open_beads=len(meta['open']),
                areas=tuple(all_areas),
                irregular=sum(not _local_mask(hole)[0].all()
                              for g in groups for hole in _holes(g['mask'])),
                map_shape=(max(r for r,c in coarse)+1,max(c for r,c in coarse)+1),
                asym=not np.array_equal(meta['mask'],meta['mask'][::-1]) and
                     not np.array_equal(meta['mask'],meta['mask'][:,::-1]))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen,training = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            m = _decode(inp)
            assert np.array_equal(_render(m),out), 'incorrect topology-gated printing'
            assert not np.array_equal(inp,out)
            pop = int(m['mask'].sum())
            assert np.count_nonzero(out != m['bg']) == pop*len(m['selected'])
            assert np.count_nonzero(out == m['bead']) == pop
            assert np.count_nonzero(out == m['ink']) == pop*(len(m['selected'])-1)
            assert not np.any(out == m['wall'])
            if split == 'train':
                training.append(m)
    a,b = training[:2]
    assert a['k'] != b['k']
    assert all(x != y for x,y in zip(a['mask'].shape,b['mask'].shape))
    assert all(x != y for x,y in zip(a['normalized_anchor'],b['normalized_anchor']))
    assert a['mask'].shape[0] != a['mask'].shape[1] and _features(a)['asym']
    h,w = a['mask'].shape
    enclosed = set().union(*(set(hole) for hole in _holes(a['mask'])))
    assert any(not a['mask'][r,c] and (rr,cc) in enclosed
               for r,c,rr,cc in ((0,0,1,1),(0,w-1,1,w-2),
                                  (h-1,0,h-2,1),(h-1,w-1,h-2,w-2)))
    assert a['normalized_anchor'] != tuple((n-1)//2 for n in _features(a)['map_shape'])
    assert any((r+1,c) in a['coarse'] or (r,c+1) in a['coarse'] for r,c in a['coarse'])
    for m in (a,b):
        assert m['closed']-m['selected'], 'wrong-class closed-bead witness missing'
        # Determine which wall's open bay contains each discarded exterior bead
        # by its bounding box; generated bays are inside that box.
        assert any(g['count'] == m['k'] and any(
            g['top'][0] <= r < g['top'][0]+g['mask'].shape[0] and
            g['top'][1] <= c < g['top'][1]+g['mask'].shape[1]
            for r,c in m['open']) for g in m['groups']), 'admitted open-bay witness missing'
    assert any(_features(m)['matches'] >= 2 for m in (a,b))
    for axis in (0,1):
        values = {p[axis] for p in b['coarse']}
        assert max(values)+1 > len(values), 'physical gap witness missing'


def _equal(a,b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[n],y[n])
               for x,y in zip(a[s],b[s]) for n in ('input','output')) for s in a)


def self_test():
    """Bounded semantic, metamorphic, counterfactual and 42-seed checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = [_features(_decode(pair['input'])) for puzzle in puzzles
                for split in puzzle for pair in puzzle[split]]
    assert _equal(puzzles[0],generate_puzzle(0))
    assert len({tuple(_key(pair['input']) for s in p for pair in p[s])
                for p in puzzles}) == 42
    assert len({f['shape'] for f in features}) >= 30
    assert len({f['glyph_mask'] for f in features}) >= 35
    assert len({f['layout'] for f in features}) >= 35
    assert len({f['roles'] for f in features}) >= 40
    assert {f['k'] for f in features} == {1,2,3}
    assert len({f['anchor'] for f in features}) >= 10
    assert len({f['objects'] for f in features}) >= 4
    assert len({f['copies'] for f in features}) >= 4
    assert sum(f['irregular'] > 0 for f in features) >= 30
    assert max(a for f in features for a in f['areas']) >= 3
    regimes = {i:0 for i in range(4)}
    for seed,puzzle in enumerate(puzzles):
        regime = int(np.random.default_rng(seed).integers(4))
        regimes[regime] += 1
        train = [_features(_decode(pair['input'])) for pair in puzzle['train']]
        test = [_features(_decode(pair['input'])) for pair in puzzle['test']]
        if regime == 0:
            assert all(f['k'] <= 2 for f in train) and all(f['k'] == 3 for f in test)
        elif regime == 1:
            assert all(max(f['glyph_shape']) <= 5 for f in train)
            assert all(max(f['glyph_shape']) == 6 for f in test)
        elif regime == 2:
            assert all(f['matches'] <= 2 for f in train)
            assert all(f['matches'] >= 3 for f in test)
        else:
            assert all(all(a == 1 for a in f['areas']) for f in train)
            for pair in puzzle['test']:
                m = _decode(pair['input'])
                assert any(len(hole) >= 3 and not _local_mask(hole)[0].all()
                           for g in m['groups'] if g['count'] == m['k']
                           for hole in _holes(g['mask']))
    assert all(regimes.values())
    # Every legal count combination and several illegal types/ranges.
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(1000+10*nt+nq,nt,nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),
                  (2.0,1),(4,1.0),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    diagonal = np.asarray([[0,1,1],[1,0,1],[1,1,1]],dtype=bool)
    assert len(_holes(diagonal)) == 1
    diagonal[0,1] = False
    assert not _holes(diagonal)
    assert not _holes(np.asarray([[1,0,1],[1,0,1],[1,1,1]],dtype=bool))
    # A palette permutation must commute with solve, including color zero.
    rng = np.random.default_rng(912)
    for pair in puzzles[0]['train']:
        palette = rng.permutation(10)
        assert np.array_equal(solve(palette[pair['input']]),palette[pair['output']])
    local_rng = np.random.default_rng(5)
    # Hand-specified spacious fixture: glyph class 2, one matching wall with two
    # closed beads plus an open bead, and a one-hole rejected wall. All ablated
    # copies and changed-anchor copies fit, so differences are actual grids.
    inp = np.zeros((30,30),dtype=np.int8)
    die = np.ones((3,5),dtype=bool)
    die[1,1] = die[1,3] = die[0,0] = False
    inp[8:11,13:18][die] = 2
    wall = np.ones((3,7),dtype=bool)
    wall[1,1] = wall[1,3] = wall[1,5] = wall[0,5] = False
    inp[:3,:7][wall] = 6
    inp[1,1],inp[1,3],inp[1,5] = 2,8,8
    inp[6:9,:3] = 6
    inp[7,1] = 8
    p = _decode(inp)
    answer = solve(inp)
    assert len(p['selected']) == 2 and len(p['closed']) == 3
    # Omitting B entirely or just its count gate creates extra exact stamps.
    mr,mc = np.nonzero(die)
    for sites in (p['points'],p['closed']):
        ablated = np.zeros_like(inp)
        for r,c in sites:
            rr = 8+((r-1)//2)*3
            cc = 13+((c-1)//2)*5
            ablated[rr+mr,cc+mc] = 8 if (r,c) == p['anchor'] else 2
        assert not np.array_equal(ablated,answer)
    # Omitting A while retaining B leaves only admitted bead coordinates.
    ablated = np.zeros_like(inp)
    for point in p['selected']:
        ablated[point] = inp[point]
    assert not np.array_equal(ablated,answer)
    changed = inp.copy()
    changed[1,5] = 0  # Open-bay bead removal is semantically irrelevant.
    assert np.array_equal(solve(changed),answer)
    changed = inp.copy()
    changed[1,1],changed[1,3] = 8,2
    assert not np.array_equal(solve(changed),answer)
    # Add a second ring lobe to the rejected wall, changing its class while
    # leaving the existing bead coordinate and the key unchanged.
    changed = inp.copy()
    changed[6:9,3:5] = 6
    changed[7,3] = 8
    assert _decode(changed)['selected'] > p['selected']
    assert not np.array_equal(solve(changed),answer)
    # Opening its second chamber changes the whole wall back to class 1. Both
    # its open bead and its still-enclosed first bead now lose admission.
    changed[6,3] = 0
    assert np.array_equal(solve(changed),answer)
    # Finite fallback variants are also decoded and solved, including class 3.
    for k,dims in ((1,(3,3)),(2,(3,5)),(3,(5,5))):
        bp = _fallback_blueprint(local_rng,k,dims)
        die = _glyph(local_rng,k,dims)
        q = _place_pair(local_rng,die,bp,bp[2][0],(0,2,6,8))
        assert np.array_equal(solve(q['input']),q['output'])
    # Every possible first-anchor coordinate has a second teaching fallback
    # changing both coordinates and retaining gaps on both axes.
    for r in range(9):
        for c in range(9):
            assert any(_anchor_candidates(
                _fallback_blueprint(local_rng,1,(5,5),diagonal)[2],(r,c),gap=True)
                for diagonal in ((6,6),(4,4),(4,6),(6,4)))
    bad = {s:[{n:np.asarray(g).copy() for n,g in pair.items()} for pair in puzzles[0][s]]
           for s in puzzles[0]}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted answer accepted')
    return dict(seeds=42,pairs=len(features),shapes=len({f['shape'] for f in features}),
                glyphs=len({f['glyph_mask'] for f in features}),
                layouts=len({f['layout'] for f in features}),
                roles=len({f['roles'] for f in features}),
                cavity_classes=sorted({f['k'] for f in features}),
                object_counts=sorted({f['objects'] for f in features}),
                copy_counts=sorted({f['copies'] for f in features}),
                irregular_pairs=sum(f['irregular'] > 0 for f in features),
                regimes=regimes)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(),separators=(',',':')))
