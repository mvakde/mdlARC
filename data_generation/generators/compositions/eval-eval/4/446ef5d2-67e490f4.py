"""Anchored negative-space jigsaws whose completed cavities are indexed by votes."""
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CORNERS = ('TL', 'TR', 'BL', 'BR')


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        cells = [first]
        for r, c in cells:
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    cells.append(p)
        result.append(tuple(cells))
    return result


def _normalize(cells):
    cells = tuple(cells)
    r = min(y for y, _ in cells)
    c = min(x for _, x in cells)
    return tuple(sorted((int(y-r), int(x-c)) for y, x in cells))


@lru_cache(maxsize=4096)
def _variants(shape, reflections=True):
    shapes = set()
    for flip in range(2 if reflections else 1):
        current = _normalize((r, -c if flip else c) for r, c in shape)
        for _ in range(4):
            shapes.add(current)
            current = _normalize((c, -r) for r, c in current)
    return tuple(sorted(shapes))


def _canonical(cells):
    return min(_variants(_normalize(cells)))


def _extent(shape):
    return max(r for r, _ in shape)+1, max(c for _, c in shape)+1


def _marker_cells(point, corner):
    r, c = point
    dr = -1 if corner[0] == 'T' else 1
    dc = -1 if corner[1] == 'L' else 1
    return ((r+dr, c), (r, c+dc), (r+dr, c+dc))


def _checked_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected a 1..30 integer grid with colors 0..9')
    return a.astype(np.int8, copy=True)


def _decode(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError('background must be the unique most frequent color')
    marks = []
    for color in np.unique(grid):
        if color == bg or counts[color] != 3:
            continue
        q = np.argwhere(grid == color)
        r, c = q.min(0)
        y, x = q.max(0)
        if (y-r, x-c) != (1, 1):
            continue
        absent = {(int(r), int(c)), (int(r), int(x)),
                  (int(y), int(c)), (int(y), int(x))} - set(map(tuple, q.tolist()))
        point = absent.pop()
        carrier = int(grid[point])
        if carrier in (bg, int(color)):
            continue
        # Ballots are separated from panels; only the marker touches carrier.
        corner = {(1, 1): 'TL', (1, 0): 'TR',
                  (0, 1): 'BL', (0, 0): 'BR'}[(point[0]-r, point[1]-c)]
        marks.append((int(color), carrier, point, corner))
    if len(marks) != 1:
        raise ValueError('expected one external L marker touching a panel corner')
    marker, carrier, point, corner = marks[0]
    tiles, anchor = [], None
    panel_mask = np.zeros_like(grid, dtype=bool)
    for cells in _components(grid == carrier):
        q = np.array(cells)
        r, c = map(int, q.min(0))
        y, x = map(int, q.max(0)+1)
        tile = grid[r:y, c:x].copy()
        if min(tile.shape) < 3 or not set(np.unique(tile)) <= {bg, carrier}:
            raise ValueError('panels have carrier/background contents and sides >= 3')
        if any(tile[p] != carrier for p in ((0,0), (0,x-c-1), (y-r-1,0), (y-r-1,x-c-1))):
            raise ValueError('all four panel corners must be occupied')
        if panel_mask[r:y, c:x].any():
            raise ValueError('panel bounding boxes overlap')
        panel_mask[r:y, c:x] = True
        if point in cells:
            anchor = len(tiles)
        tiles.append((tile, (r,c)))
    if not 3 <= len(tiles) <= 6 or anchor is None:
        raise ValueError('expected 3..6 carrier panels, one marked')
    atile, (ar, ac) = tiles[anchor]
    ah, aw = atile.shape
    if point != (ar+(ah-1 if corner[0]=='B' else 0), ac+(aw-1 if corner[1]=='R' else 0)):
        raise ValueError('marker does not mark the specified panel corner')
    exterior_mask = (grid != bg) & (grid != marker) & ~panel_mask
    ballots, objects = defaultdict(Counter), []
    for cells in _components(exterior_mask):
        colors = {int(grid[p]) for p in cells}
        if len(colors) != 1:
            raise ValueError('exterior objects must be isolated monochrome components')
        color = colors.pop()
        if color in (carrier, marker):
            raise ValueError('ballot colors must differ from structural colors')
        shape = _canonical(cells)
        ballots[shape][color] += 1
        objects.append((color, cells))
    if not ballots:
        raise ValueError('missing shape ballots')
    winners = {}
    for shape, bycolor in ballots.items():
        best = max(bycolor.values())
        choices = [c for c, n in bycolor.items() if n == best]
        if len(choices) != 1:
            raise ValueError('tied shape ballot')
        winners[shape] = choices[0]
    return dict(background=bg, carrier=carrier, marker=marker, point=point,
                corner=corner, tiles=tiles, anchor=anchor, ballots=dict(ballots),
                winners=winners, objects=objects, panel_mask=panel_mask)


def _assemblies(data, canvas):
    """Complete enumeration of unrotated rectangular-footprint tilings.

    An input panel's four occupied corners visibly fix its rectangular footprint;
    background cells inside the footprint are cavities, never packing gaps.
    """
    tiles = data['tiles']
    n = len(tiles)
    area = sum(tile.size for tile, _ in tiles)
    point, corner, anchor = data['point'], data['corner'], data['anchor']
    nodes = 0
    for h in range(3, canvas[0]+1):
        if area % h:
            continue
        w = area // h
        oy = point[0]-(h-1 if corner[0]=='B' else 0)
        ox = point[1]-(w-1 if corner[1]=='R' else 0)
        if w > canvas[1] or min(oy, ox) < 0 or oy+h > canvas[0] or ox+w > canvas[1]:
            continue
        if any(tile.shape[0] > h or tile.shape[1] > w for tile, _ in tiles):
            continue
        ar, ac = tiles[anchor][1][0]-oy, tiles[anchor][1][1]-ox
        ah, aw = tiles[anchor][0].shape
        if ar < 0 or ac < 0 or ar+ah > h or ac+aw > w:
            continue
        bits = []
        for tile, _ in tiles:
            ph, pw = tile.shape
            bit = sum(((1 << pw)-1) << (r*w) for r in range(ph))
            bits.append(bit)
        full = (1 << area)-1
        positions = [None]*n
        positions[anchor] = (ar, ac)
        occupied = bits[anchor] << (ar*w+ac)

        def visit(occ, remaining):
            nonlocal nodes
            nodes += 1
            # Refuse incomplete searches rather than claiming uniqueness.
            if nodes > 100000:
                raise ValueError('family tiling search exceeded its node bound')
            if not remaining:
                picture = np.full((h,w), data['background'], dtype=np.int8)
                labels = np.full((h,w), -1, dtype=np.int8)
                for i, (r,c) in enumerate(positions):
                    tile = tiles[i][0]
                    ph, pw = tile.shape
                    picture[r:r+ph,c:c+pw] = tile
                    labels[r:r+ph,c:c+pw] = i
                yield picture, labels, (oy,ox), tuple(positions)
                return
            free = full ^ occ
            loc = (free & -free).bit_length()-1
            r,c = divmod(loc,w)
            # The first uncovered cell must be the top-left of its rectangle.
            for i in remaining:
                ph,pw = tiles[i][0].shape
                if r+ph > h or c+pw > w:
                    continue
                bit = bits[i] << (r*w+c)
                if not bit & occ:
                    positions[i] = (r,c)
                    yield from visit(occ | bit, tuple(j for j in remaining if j != i))
        yield from visit(occupied, tuple(i for i in range(n) if i != anchor))


def _reconstruct(grid):
    data = _decode(grid)
    bg, carrier = data['background'], data['carrier']
    distinct, candidate_count, border_count, geometry_count = {}, 0, 0, 0
    border_pictures = set()
    wanted = set(data['ballots'])
    chosen_info = None
    for picture, labels, origin, positions in _assemblies(data, grid.shape):
        geometry_count += 1
        if not (np.all(picture[0] == carrier) and np.all(picture[-1] == carrier)
                and np.all(picture[:,0] == carrier) and np.all(picture[:,-1] == carrier)):
            continue
        if len(_components(picture == carrier)) != 1:
            continue
        border_count += 1
        border_pictures.add((picture.shape, origin, picture.tobytes()))
        holes = _components(picture == bg)
        if {_canonical(c) for c in holes} != wanted:
            continue
        candidate_count += 1
        filled = picture.copy()
        for cells in holes:
            for p in cells:
                filled[p] = data['winners'][_canonical(cells)]
        out = np.full_like(grid,bg)
        r,c = origin
        out[r:r+picture.shape[0],c:c+picture.shape[1]] = filled
        key = out.tobytes()
        distinct[key] = out
        chosen_info = (picture,labels,origin,positions,holes)
        if len(distinct) > 1:
            raise ValueError('shape ballots permit multiple distinct assembled outputs')
    if len(distinct) != 1:
        raise ValueError('no assembly with a closed border and the supported cavity classes')
    picture,labels,origin,positions,holes = chosen_info
    data.update(picture=picture, labels=labels, origin=origin, positions=positions,
                holes=holes, geometry_count=geometry_count, border_count=border_count,
                candidate_count=candidate_count, border_picture_count=len(border_pictures),
                crossing=sum(len({int(labels[p]) for p in hole}) > 1 for hole in holes))
    return next(iter(distinct.values())).copy(), data


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use input geometry, marker and ballots only; return the whole canvas."""
    return _reconstruct(_checked_grid(input_grid))[0]


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _shape_pool(rng):
    shapes = set()
    for length in range(1,6):
        shapes.add(_canonical((0,c) for c in range(length)))
    for h in range(2,5):
        for w in range(2,5):
            shapes.add(_canonical({(r,0) for r in range(h)} | {(h-1,c) for c in range(w)}))
            if w >= 3 and h <= 3:
                shapes.add(_canonical({(r,0) for r in range(h)} | {(r,w-1) for r in range(h)} | {(h-1,c) for c in range(w)}))
    shapes.update(_canonical(s) for s in (
        ((0,0),(0,1),(0,2),(1,1)),
        ((0,0),(0,1),(1,0),(1,1)),
        ((0,1),(1,0),(1,1),(1,2),(2,1)),
        ((0,0),(1,0),(2,0),(2,1)),
        ((0,1),(0,2),(1,0),(1,1))))
    for _ in range(120):
        cells = {(0,0)}
        for _ in range(int(rng.integers(3,8))):
            frontier = sorted({(r+dr,c+dc) for r,c in cells for dr,dc in _DIRS}-cells)
            cells.add(_pick(rng,frontier))
        shape = _canonical(cells)
        if max(_extent(shape)) <= 4:
            h,w = _extent(shape)
            mask = np.ones((h+2,w+2),bool)
            for r,c in shape:
                mask[r+1,c+1] = False
            if len(_components(mask)) == 1:
                shapes.add(shape)
    return tuple(sorted(shapes))


def _partition(rng, h, w, n, mode):
    rectangles = [(0,0,h,w)]
    if mode == 'bands':
        bands = 2 if n <= 4 else 3
        widths = [3]*bands
        for _ in range(w-3*bands):
            widths[int(rng.integers(bands))] += 1
        counts = [1]*bands
        for _ in range(n-bands):
            eligible = [i for i in range(bands) if counts[i] < h//3]
            if not eligible:
                return None
            counts[_pick(rng,eligible)] += 1
        rectangles = []
        x = 0
        for pw,k in zip(widths,counts):
            heights = [3]*k
            for _ in range(h-3*k):
                heights[int(rng.integers(k))] += 1
            y = 0
            for ph in heights:
                rectangles.append((y,x,ph,pw)); y += ph
            x += pw
    else:
        for _ in range(n-1):
            options = [(i,axis) for i,(_,_,ph,pw) in enumerate(rectangles)
                       for axis,length in enumerate((ph,pw)) if length >= 6]
            if not options:
                return None
            i,axis = _pick(rng,options)
            r,c,ph,pw = rectangles.pop(i)
            cut = int(rng.integers(3,(ph if axis == 0 else pw)-2))
            if axis == 0:
                rectangles.extend(((r,c,cut,pw),(r+cut,c,ph-cut,pw)))
            else:
                rectangles.extend(((r,c,ph,cut),(r,c+cut,ph,pw-cut)))
    if rng.integers(2):
        rectangles = [(c,r,pw,ph) for r,c,ph,pw in rectangles]
        h,w = w,h
    labels = np.empty((h,w),np.int8)
    for i,(r,c,ph,pw) in enumerate(rectangles):
        labels[r:r+ph,c:c+pw] = i
    return rectangles, labels


def _make_stencil(rng, shapes, rects, labels, min_crossing):
    h,w = labels.shape
    blank = np.zeros((h,w),bool)
    allowed = np.ones((h,w),bool)
    allowed[[0,-1],:] = False; allowed[:,[0,-1]] = False
    for r,c,ph,pw in rects:
        for p in ((r,c),(r+ph-1,c),(r,c+pw-1),(r+ph-1,c+pw-1)):
            allowed[p] = False
    sequence = list(shapes)
    sequence += [_pick(rng,shapes) for _ in range(int(rng.integers(1,3)))]
    rng.shuffle(sequence)
    sequence.sort(key=lambda s: -len(s))
    crossing = 0
    for shape in sequence:
        options = []
        for raw in _variants(shape):
            sh,sw = _extent(raw)
            nr,nc = h-sh+1,w-sw+1
            legal = np.ones((nr,nc),bool)
            cross = np.zeros((nr,nc),bool)
            dr0,dc0 = raw[0]
            first = labels[dr0:dr0+nr,dc0:dc0+nc]
            for dr,dc in raw:
                legal &= allowed[dr:dr+nr,dc:dc+nc]
                cross |= labels[dr:dr+nr,dc:dc+nc] != first
            if crossing < min_crossing and len(shape)>1:
                legal &= cross
            for r,c in np.argwhere(legal):
                options.append((raw,int(r),int(c),bool(cross[r,c])))
        for j in rng.permutation(len(options))[:100]:
            raw,r,c,is_cross = options[int(j)]
            cells = tuple((r+dy,c+dx) for dy,dx in raw)
            for q in cells: blank[q] = True
            if all(len(_components(~blank[y:y+ph,x:x+pw])) == 1 for y,x,ph,pw in rects):
                for y,x in cells:
                    allowed[max(0,y-1):y+2,max(0,x-1):x+2] = False
                crossing += int(is_cross)
                break
            for q in cells: blank[q] = False
        else:
            return None
    if crossing < min_crossing:
        return None
    return blank, _components(blank)


def _place_rectangles(rng, tile_shapes, anchor, anchor_position, markers, canvas):
    rects = [(*anchor_position,*tile_shapes[anchor])]
    positions = [None]*len(tile_shapes)
    positions[anchor] = anchor_position
    def gap(a,b):
        r,c,h,w = a; y,x,ph,pw = b
        return r+h < y or y+ph < r or c+w < x or x+pw < c
    for i in sorted((i for i in range(len(tile_shapes)) if i != anchor),key=lambda i:-np.prod(tile_shapes[i])):
        ph,pw = tile_shapes[i]
        options = [(r,c) for r in range(canvas[0]-ph+1) for c in range(canvas[1]-pw+1)]
        rng.shuffle(options)
        for r,c in options:
            if any(r-1 <= y <= r+ph and c-1 <= x <= c+pw for y,x in markers):
                continue
            if all(gap((r,c,ph,pw),other) for other in rects):
                positions[i] = (r,c); rects.append((r,c,ph,pw)); break
        if positions[i] is None:
            return None
    return positions


def _pack_votes(rng, grid, objects, data):
    bg = data['background']
    available = grid == bg
    # Reserve even the blank cells of each panel's footprint and a one-cell moat.
    for tile,(r,c) in data['tiles']:
        ph,pw = tile.shape
        available[max(0,r-1):min(grid.shape[0],r+ph+1),max(0,c-1):min(grid.shape[1],c+pw+1)] = False
    for p in np.argwhere(grid != bg):
        r,c = map(int,p)
        available[max(0,r-1):r+2,max(0,c-1):c+2] = False
    rng.shuffle(objects)
    objects.sort(key=lambda a: -max(_extent(a[2])))
    for shape,color,raw in objects:
        ph,pw = _extent(raw)
        nr,nc = grid.shape[0]-ph+1,grid.shape[1]-pw+1
        legal = np.ones((nr,nc),bool)
        for dr,dc in raw:
            legal &= available[dr:dr+nr,dc:dc+nc]
        options = np.argwhere(legal)
        if not len(options):
            return False
        r,c = map(int,options[int(rng.integers(len(options)))])
        for dr,dc in raw:
            y,x = r+dr,c+dc
            grid[y,x] = color
            available[max(0,y-1):y+2,max(0,x-1):x+2] = False
    return True


def _make_pair(rng, shapes, corner, teaching=False, harder=False, forbidden_winner=None):
    palette = list(map(int,rng.permutation(10)))
    bg,carrier,marker = palette[:3]
    colors = palette[3:]
    winners = {s:colors[i] for i,s in enumerate(shapes)}
    if not teaching and rng.random() < .35:
        winners[shapes[-1]] = winners[shapes[0]]
    if forbidden_winner is not None and winners[shapes[0]] == forbidden_winner:
        replacement = _pick(rng,[c for c in colors if c != forbidden_winner])
        winners[shapes[0]], winners[shapes[1]] = replacement, _pick(rng,[c for c in colors if c != replacement])
    ballots = {}
    for i,s in enumerate(shapes):
        counts = (3,2,2) if teaching and i==0 else _pick(rng,((1,),(2,1),(3,1),(3,2),(2,1,1),(3,2,2),(4,3,1)))
        losers = [c for c in colors if c != winners[s]]
        rng.shuffle(losers)
        ballots[s] = list(zip([winners[s]]+losers,counts))
    for attempt in range(180):
        if harder:
            if rng.random() < .35:
                h,w = int(rng.integers(8,12)),int(rng.integers(15,21))
            else:
                h,w = int(rng.integers(10,15)),int(rng.integers(11,17))
            n = int(rng.integers(4,7))
        else:
            h,w,n = int(rng.integers(9,13)),int(rng.integers(10,17)),int(rng.integers(3,6))
        mode = _pick(rng,('bands','recursive'))
        partition = _partition(rng,h,w,n,mode)
        if partition is None:
            continue
        rects,labels = partition
        result = _make_stencil(rng,shapes,rects,labels,2 if harder else 1)
        if result is None:
            continue
        blank,holes = result
        h,w = blank.shape
        picture = np.where(blank,bg,carrier).astype(np.int8)
        tiles = [(picture[r:r+ph,c:c+pw].copy(),(r,c)) for r,c,ph,pw in rects]
        local = (h-1 if corner[0]=='B' else 0,w-1 if corner[1]=='R' else 0)
        anchor = int(labels[local])
        canvas = (30,30) if attempt > 20 else (int(rng.integers(max(23,h+3),31)),int(rng.integers(max(23,w+3),31)))
        oy,ox = int(rng.integers(1,canvas[0]-h)),int(rng.integers(1,canvas[1]-w))
        point = oy+local[0],ox+local[1]
        markers = _marker_cells(point,corner)
        ar,ac = tiles[anchor][1]
        positions = _place_rectangles(rng,[t.shape for t,_ in tiles],anchor,(oy+ar,ox+ac),markers,canvas)
        if positions is None:
            continue
        scattered = [(t,p) for (t,_),p in zip(tiles,positions)]
        data = dict(background=bg,carrier=carrier,marker=marker,point=point,
                    corner=corner,tiles=scattered,anchor=anchor,ballots=dict(ballots),winners=winners)
        inp = np.full(canvas,bg,np.int8)
        for tile,(r,c) in scattered:
            ph,pw = tile.shape; inp[r:r+ph,c:c+pw] = tile
        for p in markers: inp[p] = marker
        # Teaching winners use a reflected handedness; all losers use the cavity
        # handedness. Rotation-only voting then returns a loser instead.
        witnesses = {s:_normalize(next(c for c in holes if _canonical(c)==s)) for s in shapes}
        objects = []
        for s in shapes:
            raw = witnesses[s]
            for color,count in ballots[s]:
                for _ in range(count):
                    if teaching and s==shapes[0]:
                        variants = _variants(_normalize((r,-c) for r,c in raw),False) if color==winners[s] else _variants(raw,False)
                    else:
                        variants = _variants(s)
                    objects.append((s,color,_pick(rng,variants)))
        if not _pack_votes(rng,inp,objects,data):
            continue
        if int(np.bincount(inp.ravel(),minlength=10).argmax()) != bg:
            continue
        expected = np.full(canvas,bg,np.int8)
        filled = picture.copy()
        for hole in holes:
            for p in hole: filled[p] = winners[_canonical(hole)]
        expected[oy:oy+h,ox:ox+w] = filled
        try:
            recovered,info = _reconstruct(inp)
        except ValueError:
            continue
        if (not np.array_equal(recovered,expected)
                or info['crossing'] < (2 if harder else 1)
                or (teaching and info['border_picture_count'] <= 1)):
            continue
        return {'input':inp,'output':expected}, info, attempt+1
    raise RuntimeError('failed to construct an unambiguous negative-space jigsaw')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name,value,lo,hi in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    pool = _shape_pool(rng)
    eligible = [s for s in pool if 4 <= len(s) <= 6 and max(_extent(s)) <= 3
                and len(_variants(s)) > len(_variants(s,False))
                and any(t!=s and len(t)==len(s) and sorted(_extent(t))==sorted(_extent(s)) for t in pool)]
    core = _pick(rng,eligible)
    companion = _pick(rng,[t for t in pool if t!=core and len(t)==len(core) and sorted(_extent(t))==sorted(_extent(core))])
    train_corners = _pick(rng,(('TL','BR'),('TR','BL')))
    test_corners = tuple(c for c in _CORNERS if c not in train_corners)
    puzzle = {'train':[],'test':[]}
    seen,known,previous = set(),set(),None
    for split,count in (('train',num_train),('test',num_test)):
        for i in range(count):
            teaching = split=='train' and i<2
            if teaching:
                shapes = (core,companion)
            else:
                k = int(rng.integers(2,5)) if split=='test' else int(rng.integers(2,4))
                if split=='test':
                    novel = [s for s in pool if s not in known]
                    shapes = [_pick(rng,novel)]
                else:
                    shapes = [_pick(rng,(core,companion))]
                choices = [s for s in pool if s not in shapes and len(s) <= 7]
                shapes += [choices[int(j)] for j in rng.choice(len(choices),k-1,replace=False)]
                shapes = tuple(shapes)
            corner = train_corners[i%2] if split=='train' else _pick(rng,test_corners)
            for _ in range(8):
                pair,info,tries = _make_pair(rng,shapes,corner,teaching,split=='test',previous if teaching and i==1 else None)
                key = pair['input'].shape,pair['input'].tobytes()
                if key not in seen: break
            else:
                raise RuntimeError('duplicate input after bounded retries')
            puzzle[split].append(pair); seen.add(key)
            if split=='train':
                known.update(shapes)
                if i==0: previous = info['winners'][core]
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_fill(data, mode):
    """Ablations retain the correct assembly to isolate shape/voting evidence.

    A failed (unsupported or tied) alternative is represented by None; failure
    concerns filling an actual cavity, never bookkeeping about unused objects.
    """
    def signature(cells):
        raw = _normalize(cells)
        if mode == 'area':
            return len(raw)
        if mode == 'bbox':
            return tuple(sorted(_extent(raw)))
        if mode == 'rotation':
            return min(_variants(raw, False))
        if mode == 'translation':
            return raw
        if mode == 'global':
            return 0
        return _canonical(raw)
    counts = defaultdict(Counter)
    for color, cells in data['objects']:
        counts[signature(cells)][color] += 1
    picture = data['picture'].copy()
    for cells in data['holes']:
        ballot = counts.get(signature(cells))
        if not ballot:
            return None
        high = max(ballot.values())
        winners = [c for c,n in ballot.items() if n == high]
        if len(winners) != 1:
            return None
        for p in cells:
            picture[p] = winners[0]
    return picture


def _alternative_fails(data, output, mode):
    predicted = _alternative_fill(data, mode)
    r,c = data['origin']
    h,w = data['picture'].shape
    return predicted is None or not np.array_equal(predicted,output[r:r+h,c:c+w])


def _fragment_classes(data):
    return {_canonical(cells) for tile,_ in data['tiles']
            for cells in _components(tile == data['background'])}


def _assert_separation(grid, data):
    bg, marker = data['background'], data['marker']
    footprints = np.zeros_like(grid,dtype=bool)
    for tile,(r,c) in data['tiles']:
        h,w = tile.shape
        assert not footprints[max(0,r-1):min(grid.shape[0],r+h+1),
                              max(0,c-1):min(grid.shape[1],c+w+1)].any(), 'panels need a blank moat'
        footprints[r:r+h,c:c+w] = True
    # Vote objects cannot occupy holes or touch the panels, even diagonally.
    for color,cells in data['objects']:
        assert color not in (bg,marker,data['carrier'])
        for r,c in cells:
            assert not footprints[max(0,r-1):r+2,max(0,c-1):c+2].any()
    all_object_cells = [set(cells) for _,cells in data['objects']]
    for i,cells in enumerate(all_object_cells):
        neighbors = {(r+dr,c+dc) for r,c in cells
                     for dr in (-1,0,1) for dc in (-1,0,1)}
        assert all(not neighbors.intersection(other) for other in all_object_cells[i+1:])
        assert all(grid[p] != marker for p in neighbors
                   if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1])


def validate_generated_puzzle(puzzle):
    """Validate visible geometry, preservation, support, and teaching witnesses."""
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _checked_grid(pair['input']),_checked_grid(pair['output'])
            assert inp.shape == out.shape
            key = inp.shape,inp.tobytes()
            assert key not in seen, 'duplicate input'; seen.add(key)
            predicted,data = _reconstruct(inp)
            assert np.array_equal(predicted,out), 'output differs from the input-derived rule'
            bg,carrier,marker = data['background'],data['carrier'],data['marker']
            _assert_separation(inp,data)
            assert not np.any(out == marker)
            assert all(out[p] == bg for p in _marker_cells(data['point'],data['corner']))
            assert out[data['point']] == carrier
            assert np.count_nonzero(inp == carrier) == np.count_nonzero(out == carrier)
            assert np.count_nonzero(out != bg) == sum(tile.size for tile,_ in data['tiles'])
            r,c = data['origin']; h,w = data['picture'].shape
            crop = out[r:r+h,c:c+w]
            assert np.all(crop != bg)
            outer = out.copy(); outer[r:r+h,c:c+w] = bg
            assert np.all(outer == bg), 'output must clear the exterior canvas'
            assert data['positions'][data['anchor']] == (
                data['tiles'][data['anchor']][1][0]-r,
                data['tiles'][data['anchor']][1][1]-c)
            for (tile,_),(y,x) in zip(data['tiles'],data['positions']):
                ph,pw = tile.shape
                recovered = data['picture'][y:y+ph,x:x+pw]
                assert np.array_equal(recovered,tile), 'panel rotated, modified, or incorrectly translated'
            assert data['crossing'] >= (2 if split=='test' else 1)
            assert 2 <= len(data['ballots']) <= 4
            assert {_canonical(cells) for cells in data['holes']} == set(data['ballots'])
            assert len(_components(data['picture'] == carrier)) == 1
            analyses[split].append(data)
    train = analyses['train']
    # The first two demonstrations always survive even at the legal 2/1 limit.
    assert all(d['border_picture_count'] >= 2 for d in train[:2]), 'shape support must resolve geometric alternatives'
    assert all(_fragment_classes(d)-set(d['ballots']) for d in train[:2]), 'unfinished cavities must require assembly'
    assert any(_alternative_fails(d,np.asarray(p['output']),'area') for d,p in zip(train,puzzle['train']))
    assert any(_alternative_fails(d,np.asarray(p['output']),'bbox') for d,p in zip(train,puzzle['train']))
    assert any(_alternative_fails(d,np.asarray(p['output']),'rotation') for d,p in zip(train,puzzle['train']))
    assert any(_alternative_fails(d,np.asarray(p['output']),'global') for d,p in zip(train,puzzle['train']))
    assert any(len(ballot) >= 3 and 2*max(ballot.values()) <= sum(ballot.values())
               for d in train[:2] for ballot in d['ballots'].values())
    history = defaultdict(set)
    for d in train:
        for shape,color in d['winners'].items():
            history[shape].add(color)
    assert any(len(colors)>1 for colors in history.values()), 'training must reject a fixed shape-color dictionary'
    train_corners = {d['corner'] for d in train}
    assert train_corners in ({'TL','BR'},{'TR','BL'})
    known = set().union(*(set(d['ballots']) for d in train))
    for d in analyses['test']:
        assert d['corner'] not in train_corners
        assert set(d['ballots'])-known, 'test must include a locally supported unseen class'


_SELF_SEEDS = (0,134886316,678285023,800310289,878286230,422709435,540385974,
    796710321,442489493,434551364,888984528,206559742,677157016,933054888,
    593400459,838489753,79039534,680271747,624293025,223720873,7160943,
    483400056,543827256,572165305,240675091,809484671,657701055,783824105,
    460664975,17051926,971067518,926920889,468907594,746547907,821198193,
    497076353,775362464,625906779,264853465,357659315,949544084,398406411)


def _puzzle_key(puzzle):
    return tuple((np.asarray(pair[k]).shape,np.asarray(pair[k]).astype(np.int8).tobytes())
                 for split in ('train','test') for pair in puzzle[split] for k in ('input','output'))


def self_test():
    """Bounded 42-seed coverage, ablations, counterfactuals, and corruption checks.

    These are generator-owned checks, not an independent semantic certification.
    """
    puzzles = [generate_puzzle(seed) for seed in _SELF_SEEDS]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    records,partitions,normalized = [],set(),set()
    for puzzle in puzzles:
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp,out = np.asarray(pair['input']),np.asarray(pair['output'])
                predicted,data = _reconstruct(inp)
                assert np.array_equal(predicted,out)
                records.append(data)
                labels = data['labels']; rename = {}
                partition = tuple(rename.setdefault(int(v),len(rename)) for v in labels.ravel())
                partitions.add((labels.shape,partition))
                rename = {}
                normalized.add((inp.shape,tuple(rename.setdefault(int(v),len(rename)) for v in inp.ravel())))
    assert len(normalized) >= 180, 'variation must go beyond recoloring'
    assert len(partitions) >= 130
    assert len({d['picture'].shape for d in records}) >= 35
    assert len({pair['input'].shape for p in puzzles for s in ('train','test') for pair in p[s]}) >= 35
    assert {len(d['tiles']) for d in records} == {3,4,5,6}
    assert {len(d['ballots']) for d in records} >= {2,3,4}
    assert {d['corner'] for d in records} == set(_CORNERS)
    vocab = set().union(*(set(d['ballots']) for d in records))
    assert len(vocab) >= 55
    assert max(map(len,vocab)) >= 8 and any(len(s)==1 for s in vocab)
    ballots = {tuple(sorted(b.values(),reverse=True)) for d in records for b in d['ballots'].values()}
    assert {(1,),(2,1),(3,2),(3,2,2),(4,3,1)} <= ballots
    assert len({len(d['holes']) for d in records}) >= 4
    assert max(d['crossing'] for d in records) >= 4
    assert any(len(set(d['winners'].values())) < len(d['winners']) for d in records)
    assert any(max(d['picture'].shape)/min(d['picture'].shape) >= 2 for d in records)
    # Change only a ballot's plurality, preserving its shape support and panels.
    # The assembly must stay put while its corresponding cavity color changes.
    pair = puzzles[0]['train'][0]
    inp = pair['input'].copy(); _,data = _reconstruct(inp)
    shape = next(s for s,b in data['ballots'].items() if len(b)==3)
    counts = data['ballots'][shape]
    winner = data['winners'][shape]
    loser = next(c for c in counts if c != winner)
    for color,cells in data['objects']:
        if _canonical(cells) == shape and color in (winner,loser):
            for p in cells:
                inp[p] = loser if color==winner else winner
    changed,changed_data = _reconstruct(inp)
    assert changed_data['origin'] == data['origin']
    assert np.array_equal(changed_data['picture'],data['picture'])
    assert changed_data['winners'][shape] == loser
    assert not np.array_equal(changed,pair['output'])
    # Segmentation order cannot pick a hidden assembly.
    reverse = dict(data); n = len(data['tiles'])
    reverse['tiles'] = list(reversed(data['tiles']))
    reverse['anchor'] = n-1-data['anchor']
    pictures = set()
    for pic,labels,origin,positions in _assemblies(reverse,pair['input'].shape):
        if (np.all(pic[0]==data['carrier']) and np.all(pic[-1]==data['carrier'])
                and np.all(pic[:,0]==data['carrier']) and np.all(pic[:,-1]==data['carrier'])
                and {_canonical(c) for c in _components(pic==data['background'])} == set(data['ballots'])):
            pictures.add((origin,pic.shape,pic.tobytes()))
    assert pictures == {(data['origin'],data['picture'].shape,data['picture'].tobytes())}
    for nt,ne in ((2,1),(10,4)):
        validate_generated_puzzle(generate_puzzle(8100+nt+ne,nt,ne))
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Wrong filling, retained marker, moved anchor and duplicates must be caught.
    for corruption in ('fill','marker','anchor','duplicate'):
        broken = {s:[{k:v.copy() for k,v in q.items()} for q in puzzles[0][s]] for s in ('train','test')}
        q = broken['test'][0]; _,d = _reconstruct(q['input'])
        if corruption == 'fill':
            y,x = d['holes'][0][0]; r,c = d['origin']
            q['output'][r+y,c+x] = d['background']
        elif corruption == 'marker':
            q['output'][_marker_cells(d['point'],d['corner'])[0]] = d['marker']
        elif corruption == 'anchor':
            q['output'] = np.roll(q['output'],1,axis=0)
        else:
            broken['train'][1] = {k:v.copy() for k,v in broken['train'][0].items()}
        try:
            validate_generated_puzzle(broken)
        except (AssertionError,ValueError):
            pass
        else:
            raise AssertionError('accepted corruption: '+corruption)
