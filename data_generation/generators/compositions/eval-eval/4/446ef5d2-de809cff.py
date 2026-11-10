"""Scar stitching: infer a rectangular assembly from incomplete erasure rings.

Four local color roles: majority background, external L/highlight, and two fills.
All inference is input-only. No parent code, saved fixtures, or generation state
is used. See task_spec.md for the semantic distinction between substrate, holes,
and partially printed opposite-fill witnesses.
"""
from functools import lru_cache
import numpy as np


_CORNERS = ("TL", "TR", "BL", "BR")


def _integer(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return g.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        cells = [p]
        for r, c in cells:
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
        result.append(cells)
    return result


def _marker_cells(point, corner):
    r, c = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return ((r+dy, c), (r, c+dx), (r+dy, c+dx))


def _extract(input_grid):
    g = _grid(input_grid)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= g.size:
        raise ValueError("Background must occupy more than half the input")
    candidates = []
    for color in range(10):
        if color == bg or counts[color] != 3:
            continue
        q = np.argwhere(g == color)
        r, c = map(int, q.min(axis=0))
        y, x = map(int, q.max(axis=0))
        if (y-r, x-c) != (1, 1):
            continue
        missing = ({(r, c), (r, x), (y, c), (y, x)}
                   - set(map(tuple, q.tolist()))).pop()
        if int(g[missing]) in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}[
            (missing[0]-r, missing[1]-c)]
        candidates.append((color, missing, corner))
    if len(candidates) != 1:
        raise ValueError("Expected one external three-cell L")
    marker, point, corner = candidates[0]
    fills = tuple(int(c) for c in np.unique(g) if c not in (bg, marker))
    if len(fills) != 2:
        raise ValueError("Expected two fill colors")
    pieces = []
    anchor = None
    for cells in _components((g != bg) & (g != marker)):
        q = np.array(cells)
        r, c = map(int, q.min(axis=0))
        y, x = map(int, q.max(axis=0)+1)
        matrix = g[r:y, c:x].copy()
        if np.any(matrix == marker) or any(v not in (bg, *fills) for v in np.unique(matrix)):
            raise ValueError("Tiles contain only fills and erased cells")
        # Disjoint bounding rectangles, not incomplete fragments of one tile.
        if np.count_nonzero(matrix != bg) != len(cells):
            raise ValueError("Tile bounding boxes overlap another component")
        ns = [int(np.count_nonzero(matrix == f)) for f in fills]
        if ns[0] == ns[1]:
            raise ValueError("Tile substrate must have a strict fill majority")
        base = fills[int(ns[1] > ns[0])]
        holes = tuple(map(tuple, np.argwhere(matrix == bg).tolist()))
        witnesses = tuple((int(a), int(b), int(matrix[a, b])) for a, b in
                          np.argwhere((matrix != bg) & (matrix != base)))
        if point in cells:
            anchor = len(pieces)
        pieces.append(dict(shape=matrix.shape, old=(r, c), base=base,
                           holes=holes, witnesses=witnesses, matrix=matrix))
    if not 2 <= len(pieces) <= 7 or anchor is None:
        raise ValueError("Expected 2..7 separated tiles and an anchored corner")
    a = pieces[anchor]
    expected = (a['old'][0] + (a['shape'][0]-1 if corner[0] == 'B' else 0),
                a['old'][1] + (a['shape'][1]-1 if corner[1] == 'R' else 0))
    if point != expected:
        raise ValueError("L must mark the corresponding tile corner")
    for i, p in enumerate(pieces):
        r, c = p['old']; h, w = p['shape']
        if h < 3 or w < 3:
            raise ValueError("Tile sides must be at least three")
        for other in pieces[:i]:
            y, x = other['old']; ph, pw = other['shape']
            if not (r+h < y or y+ph < r or c+w < x or x+pw < c):
                raise ValueError("Input tiles require a blank separating row or column")
    if not any(p['holes'] for p in pieces):
        raise ValueError("At least one erased cell is required")
    return g, bg, marker, fills, point, corner, pieces, anchor


@lru_cache(maxsize=128)
def _packings(shapes, anchor, anchor_old, point, corner, canvas):
    """Complete rectangle-tiling enumeration; first vacant cell fixes a new TL.

    The cache key consists solely of input-derived geometry. At most seven
    rectangles and 30 candidate heights bound this finite exhaustive search.
    """
    area = sum(h*w for h, w in shapes)
    results = []
    for h in range(1, canvas[0]+1):
        if area % h:
            continue
        w = area // h
        oy = point[0] - (h-1 if corner[0] == 'B' else 0)
        ox = point[1] - (w-1 if corner[1] == 'R' else 0)
        if w > canvas[1] or oy < 0 or ox < 0 or oy+h > canvas[0] or ox+w > canvas[1]:
            continue
        if any(ph > h or pw > w for ph, pw in shapes):
            continue
        ar, ac = anchor_old[0]-oy, anchor_old[1]-ox
        ah, aw = shapes[anchor]
        if ar < 0 or ac < 0 or ar+ah > h or ac+aw > w:
            continue
        bits = [sum(((1 << pw)-1) << (r*w) for r in range(ph)) for ph, pw in shapes]
        positions = [None]*len(shapes)
        positions[anchor] = (ar, ac)
        full = (1 << area)-1

        def visit(occupied, remaining):
            if not remaining:
                results.append(((h, w), (oy, ox),
                                tuple((r+oy, c+ox) for r, c in positions)))
                return
            free = full ^ occupied
            k = (free & -free).bit_length()-1
            r, c = divmod(k, w)
            for i in remaining:
                ph, pw = shapes[i]
                if r+ph > h or c+pw > w:
                    continue
                mask = bits[i] << k
                if occupied & mask:
                    continue
                positions[i] = (r, c)
                visit(occupied | mask, tuple(j for j in remaining if j != i))

        visit(bits[anchor] << (ar*w+ac), tuple(i for i in range(len(shapes)) if i != anchor))
    return tuple(results)


def _ring_demands(pieces, positions, canvas):
    holes = {}
    for p, (r, c) in zip(pieces, positions):
        for y, x in p['holes']:
            holes[(r+y, c+x)] = p['base']
    fills = {p['base'] for p in pieces}
    if len(fills) != 2:
        raise ValueError("Both substrate fill roles must occur")
    opposite = {a: b for a in fills for b in fills if a != b}
    demands = {}
    for (r, c), base in holes.items():
        for y in range(max(0, r-1), min(canvas[0], r+2)):
            for x in range(max(0, c-1), min(canvas[1], c+2)):
                if (y, x) in holes:
                    continue
                value = opposite[base]
                if (y, x) in demands and demands[(y, x)] != value:
                    return None
                demands[(y, x)] = value
    return holes, demands


def _supported(pieces, positions, canvas):
    annotation = _ring_demands(pieces, positions, canvas)
    if annotation is None:
        return None
    holes, demands = annotation
    for p, (r, c) in zip(pieces, positions):
        for y, x, value in p['witnesses']:
            if demands.get((r+y, c+x)) != value:
                return None
    return annotation


def _paint(pieces, positions, canvas, bg, marker, annotation):
    holes, demands = annotation
    output = np.full(canvas, bg, dtype=np.int8)
    for p, (r, c) in zip(pieces, positions):
        h, w = p['shape']
        output[r:r+h, c:c+w] = p['base']
    for q, value in demands.items():
        output[q] = value
    for q in holes:
        output[q] = marker
    return output


def _infer(input_grid):
    g, bg, marker, fills, point, corner, pieces, anchor = _extract(input_grid)
    candidates = _packings(tuple(p['shape'] for p in pieces), anchor,
                           pieces[anchor]['old'], point, corner, g.shape)
    solutions = {}
    for shape, origin, positions in candidates:
        annotation = _supported(pieces, positions, g.shape)
        if annotation is None:
            continue
        output = _paint(pieces, positions, g.shape, bg, marker, annotation)
        # Interchanging indistinguishable tiles is allowed when the entire
        # annotated output is identical. No iteration-order tie convention.
        key = output.tobytes()
        if key not in solutions:
            solutions[key] = (output, shape, origin, positions, annotation)
        if len(solutions) > 1:
            raise ValueError("Distinct supported scar assemblies give different outputs")
    if not solutions:
        raise ValueError("No assembly explains every partial ring")
    output, shape, origin, positions, annotation = next(iter(solutions.values()))
    return dict(output=output, shape=shape, origin=origin, positions=positions,
                annotation=annotation, pieces=pieces, anchor=anchor, background=bg,
                marker=marker, fills=fills, point=point, corner=corner,
                geometric_candidates=len(candidates))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique scar-stitched picture from the input alone."""
    return _infer(input_grid)['output'].copy()


def _partition(rng, h, w, count):
    """Unequal recursive partitions; mixed strips, T junctions, and grids arise."""
    boxes = [(0, 0, h, w)]
    while len(boxes) < count:
        options = [(i, axis) for i, (_, _, ph, pw) in enumerate(boxes)
                   for axis, size in enumerate((ph, pw)) if size >= 6]
        if not options:
            return None
        i, axis = options[int(rng.integers(len(options)))]
        r, c, ph, pw = boxes.pop(i)
        cut = int(rng.integers(3, (ph if axis == 0 else pw)-2))
        if axis == 0:
            boxes.extend(((r, c, cut, pw), (r+cut, c, ph-cut, pw)))
        else:
            boxes.extend(((r, c, ph, cut), (r, c+cut, ph, pw-cut)))
    if any(ph*pw < 15 for _, _, ph, pw in boxes):
        return None
    return boxes


def _scatter(rng, pieces, anchor, final, marker_cells, canvas):
    rest = sorted((i for i in range(len(pieces)) if i != anchor),
                  key=lambda i: -np.prod(pieces[i]['shape']))
    for _ in range(16):
        placed = {anchor: final[anchor]}
        occupied = np.zeros(canvas, dtype=bool)
        r, c = final[anchor]; h, w = pieces[anchor]['shape']
        occupied[max(0, r-1):min(canvas[0], r+h+1), max(0, c-1):min(canvas[1], c+w+1)] = True
        for y, x in marker_cells:
            occupied[max(0, y-1):min(canvas[0], y+2), max(0, x-1):min(canvas[1], x+2)] = True
        for i in rest:
            h, w = pieces[i]['shape']
            choices = [(r, c) for r in range(canvas[0]-h+1) for c in range(canvas[1]-w+1)
                       if not occupied[r:r+h, c:c+w].any()]
            if not choices:
                break
            # Source layout is deliberately unrelated to assembly layout.
            r, c = choices[int(rng.integers(len(choices)))]
            placed[i] = (r, c)
            occupied[max(0, r-1):min(canvas[0], r+h+1), max(0, c-1):min(canvas[1], c+w+1)] = True
        if len(placed) == len(pieces):
            return tuple(placed[i] for i in range(len(pieces)))
    return None


def _try_pair(rng, palette, corner, kind, larger):
    bg, marker, f0, f1 = palette
    for _ in range(100):
        count = int(rng.integers(3, 7 if larger else 6))
        h = int(rng.integers(7, 14 if larger else 12))
        w = int(rng.integers(8, 17 if larger else 14))
        if h*w > (180 if larger else 140):
            continue
        boxes = _partition(rng, h, w, count)
        if boxes is None:
            continue
        # Swap whole-layout axes freely; individual tiles are never rotated.
        if rng.integers(2):
            boxes = [(c, r, pw, ph) for r, c, ph, pw in boxes]
            h, w = w, h
        labels = np.full((h, w), -1, dtype=int)
        bases = [int(rng.choice((f0, f1))) for _ in boxes]
        if len(set(bases)) != 2:
            continue
        clean = np.empty((h, w), dtype=np.int8)
        for i, (r, c, ph, pw) in enumerate(boxes):
            labels[r:r+ph, c:c+pw] = i
            clean[r:r+ph, c:c+pw] = bases[i]
        local_point = (h-1 if corner[0] == 'B' else 0, w-1 if corner[1] == 'R' else 0)
        anchor = int(labels[local_point])
        centers = []

        def add(q):
            if q in centers or q == local_point:
                return False
            if any(clean[q] != clean[p] and max(abs(q[0]-p[0]), abs(q[1]-p[1])) <= 2
                   for p in centers):
                return False
            centers.append(q)
            return True

        # The join lesson/test places an adjacent same-fill pair across a cut.
        joins = [(r, c, y, x) for r in range(h) for c in range(w)
                 for y, x in ((r+1, c), (r, c+1)) if y < h and x < w
                 and labels[r, c] != labels[y, x] and clean[r, c] == clean[y, x]
                 and (r, c) != local_point and (y, x) != local_point]
        if kind in ('join', 'test'):
            if not joins:
                continue
            r, c, y, x = joins[int(rng.integers(len(joins)))]
            add((r, c)); add((y, x))
        coords = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(coords)
        # Demonstrate both complementary directions, independent of literal fill.
        if kind == 'dual':
            for f in (f0, f1):
                for q in coords:
                    if clean[q] == f and add(q):
                        break
                else:
                    break
            if {int(clean[q]) for q in centers} != {f0, f1}:
                continue
        target = int(rng.integers(4, 10 if larger else 8))
        # Prefer defects near cuts so their neighborhood has another owner.
        coords.sort(key=lambda q: 0 if any(0 <= y < h and 0 <= x < w
                           and labels[y, x] != labels[q] for y, x in
                           ((q[0]-1,q[1]),(q[0]+1,q[1]),(q[0],q[1]-1),(q[0],q[1]+1))) else 1)
        for q in coords:
            if len(centers) >= target:
                break
            add(q)
        pieces = []
        center_set = set(centers)
        demands = {}
        for r, c in centers:
            for y in range(max(0,r-1),min(h,r+2)):
                for x in range(max(0,c-1),min(w,c+2)):
                    if (y,x) not in center_set:
                        demands[(y,x)] = f1 if clean[r,c] == f0 else f0
        possible = [(q, val) for q, val in demands.items() if clean[q] != val]
        rng.shuffle(possible)
        printed = {q: val for q, val in possible if rng.random() < float(rng.uniform(.45,.85))}
        if not printed:
            continue
        damaged = clean.copy()
        for q, val in printed.items():
            damaged[q] = val
        for q in centers:
            damaged[q] = bg
        okay = True
        for i, (r,c,ph,pw) in enumerate(boxes):
            m = damaged[r:r+ph,c:c+pw].copy()
            visible = m != bg
            if (len(_components(visible)) != 1 or not visible[0].any() or not visible[-1].any()
                    or not visible[:,0].any() or not visible[:,-1].any()
                    or np.count_nonzero(m == bases[i])*2 <= np.count_nonzero(visible)):
                okay = False
                break
            pieces.append(dict(shape=m.shape, base=bases[i], matrix=m,
                               holes=tuple(map(tuple,np.argwhere(m == bg).tolist())),
                               witnesses=tuple((int(y),int(x),int(m[y,x])) for y,x in
                                               np.argwhere((m != bg)&(m != bases[i])))))
        if not okay:
            continue
        canvas = (int(rng.integers(max(18,h+4),31)), int(rng.integers(max(18,w+4),31)))
        if h*w+3 >= np.prod(canvas)/2:
            continue
        lo_y = 1 if corner[0] == 'T' else 0
        hi_y = canvas[0]-h-(1 if corner[0] == 'B' else 0)
        lo_x = 1 if corner[1] == 'L' else 0
        hi_x = canvas[1]-w-(1 if corner[1] == 'R' else 0)
        origin = (int(rng.integers(lo_y,hi_y+1)),int(rng.integers(lo_x,hi_x+1)))
        point = (origin[0]+local_point[0],origin[1]+local_point[1])
        marks = _marker_cells(point,corner)
        final = tuple((origin[0]+r,origin[1]+c) for r,c,_,_ in boxes)
        scattered = _scatter(rng,pieces,anchor,final,marks,canvas)
        if scattered is None:
            continue
        g = np.full(canvas,bg,dtype=np.int8)
        for p,(r,c) in zip(pieces,scattered):
            ph,pw = p['shape'];g[r:r+ph,c:c+pw] = p['matrix']
        for q in marks:
            g[q] = marker
        annotation = _ring_demands(pieces,final,canvas)
        if annotation is None:
            continue
        expected = _paint(pieces,final,canvas,bg,marker,annotation)
        try:
            info = _infer(g)
        except ValueError:
            continue
        if not np.array_equal(info['output'],expected):
            continue
        features = _features(info)
        if kind in ('join','test') and not (features['cross_witnesses'] and features['adjacent_cross_holes']
                                           and features['geometric_candidates'] > 1):
            continue
        if kind in ('join','test') and _without_witnesses(info) < 2:
            continue
        if not features['unfinished_cells']:
            continue
        return {'input':g,'output':expected}
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train,num_test = _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    # Palette recurrence is a distribution choice. No hidden color dictionary.
    base_palette = tuple(map(int,rng.choice(10,4,replace=False)))
    varied = bool(rng.integers(2))
    corners = list(_CORNERS)
    rng.shuffle(corners)
    train_plan = [('dual',corners[0],False),('join',corners[1],False)]
    train_plan += [(str(rng.choice(('dual','join','free'))),str(rng.choice(corners[:2])),
                    bool(rng.integers(2))) for _ in range(num_train-2)]
    rng.shuffle(train_plan)
    puzzle = {'train':[],'test':[]}
    seen = set()
    for split,plan in (('train',train_plan),('test',[('test',corners[2+i%2],True)
                                                  for i in range(num_test)])):
        for kind,corner,larger in plan:
            for attempt in range(12):
                palette = tuple(map(int,rng.choice(10,4,replace=False))) if varied else base_palette
                pair = _try_pair(rng,palette,corner,kind,larger)
                if pair is None:
                    continue
                key = (pair['input'].shape,pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not construct a distinct, uniquely inferable scar assembly')
    return puzzle


def _features(info):
    pieces = info['pieces']; positions = info['positions']
    holes,demands = info['annotation']
    owners = {}
    substrate = {}
    witnesses = {}
    for i,(p,(r,c)) in enumerate(zip(pieces,positions)):
        h,w = p['shape']
        for y in range(h):
            for x in range(w):
                owners[(r+y,c+x)] = i
                substrate[(r+y,c+x)] = p['base']
        for y,x,val in p['witnesses']:
            witnesses[(r+y,c+x)] = val
    cross = 0
    for q in witnesses:
        relevant = [p for p in holes if max(abs(p[0]-q[0]),abs(p[1]-q[1])) <= 1]
        # Strong evidence: no center in its own tile can explain this witness.
        cross += bool(relevant and all(owners[p] != owners[q] for p in relevant))
    hp = list(holes)
    adjacent = any(owners[p] != owners[q] and holes[p] == holes[q]
                   and max(abs(p[0]-q[0]),abs(p[1]-q[1])) == 1
                   for i,p in enumerate(hp) for q in hp[:i])
    unfinished = sum(substrate.get(q,info['background']) != val and q not in witnesses
                     for q,val in demands.items())
    ordered = sorted(zip(pieces,positions),key=lambda item:item[1])
    normalized_bases = {}
    layout = []
    for p,(r,c) in ordered:
        role = normalized_bases.setdefault(p['base'],len(normalized_bases))
        layout.append((r-info['origin'][0],c-info['origin'][1],*p['shape'],role))
    layout = tuple(layout)
    return dict(shape=info['output'].shape,target_shape=info['shape'],corner=info['corner'],
                piece_count=len(pieces),hole_count=len(holes),hole_fills=tuple(sorted(set(holes.values()))),
                cross_witnesses=cross,adjacent_cross_holes=adjacent,unfinished_cells=unfinished,
                geometric_candidates=info['geometric_candidates'],layout=layout,
                clipped=any(r in (0,info['output'].shape[0]-1) or c in (0,info['output'].shape[1]-1)
                            for r,c in holes),
                external_ring=any(q not in owners for q in demands),
                witness_count=len(witnesses),background=info['background'],marker=info['marker'],
                piece_dimensions=tuple(sorted(p['shape'] for p in pieces)))


def _without_witnesses(info):
    """Outputs possible with geometric packing and ring compatibility alone."""
    pieces = info['pieces']
    candidates = _packings(tuple(p['shape'] for p in pieces),info['anchor'],
                           pieces[info['anchor']]['old'],info['point'],info['corner'],
                           info['output'].shape)
    answers = set()
    for _,_,positions in candidates:
        annotation = _ring_demands(pieces,positions,info['output'].shape)
        if annotation is not None:
            out = _paint(pieces,positions,info['output'].shape,info['background'],
                         info['marker'],annotation)
            answers.add(out.tobytes())
    return len(answers)


def _local_only(info):
    """A deliberately incomplete alternative: complete each tile independently."""
    out = np.full(info['output'].shape,info['background'],dtype=np.int8)
    for p,(r,c) in zip(info['pieces'],info['positions']):
        h,w = p['shape']
        patch = np.full((h,w),p['base'],dtype=np.int8)
        opposite = next(f for f in info['fills'] if f != p['base'])
        for y,x in p['holes']:
            patch[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)] = opposite
        for q in p['holes']:
            patch[q] = info['marker']
        out[r:r+h,c:c+w] = patch
    return out


def _without_assembly(info):
    """B-style repair at the scattered coordinates, omitting A's translations."""
    positions = tuple(p['old'] for p in info['pieces'])
    annotation = _ring_demands(info['pieces'],positions,info['output'].shape)
    if annotation is None:
        return None
    return _paint(info['pieces'],positions,info['output'].shape,info['background'],
                  info['marker'],annotation)


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle,dict) or set(puzzle) != {'train','test'}:
        raise ValueError('Puzzle must have exactly train and test')
    if not all(isinstance(puzzle[s],list) for s in puzzle):
        raise ValueError('Splits must be lists')
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen = set(); features = {'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            if not isinstance(pair,dict) or set(pair) != {'input','output'}:
                raise ValueError('Pairs must have exactly input and output')
            g,out = _grid(pair['input']),_grid(pair['output'])
            if g.shape != out.shape:
                raise ValueError('Canvas shape must be preserved')
            key = (g.shape,g.tobytes())
            if key in seen:
                raise ValueError('Duplicate input')
            seen.add(key)
            info = _infer(g)
            if not np.array_equal(info['output'],out):
                raise ValueError('Output differs from input-derived scar stitching')
            assert info['positions'][info['anchor']] == info['pieces'][info['anchor']]['old']
            h,w = info['shape'];r,c = info['origin']
            assert h*w == sum(np.prod(p['shape']) for p in info['pieces'])
            occupied = np.zeros(g.shape,dtype=bool)
            for p,(y,x) in zip(info['pieces'],info['positions']):
                ph,pw = p['shape']
                assert not occupied[y:y+ph,x:x+pw].any()
                occupied[y:y+ph,x:x+pw] = True
            assert occupied[r:r+h,c:c+w].all() and occupied.sum() == h*w
            holes,demands = info['annotation']
            assert all(out[q] == info['marker'] for q in holes)
            assert np.count_nonzero(out == info['marker']) == len(holes)
            assert all(out[q] == value for q,value in demands.items())
            for p,(y,x) in zip(info['pieces'],info['positions']):
                assert all(out[y+dy,x+dx] == value for dy,dx,value in p['witnesses'])
            allowed = occupied.copy()
            for q in demands:
                allowed[q] = True
            assert np.all(out[~allowed] == info['background'])
            assert not np.array_equal(g,out)
            features[split].append(_features(info))
    # Sampling guarantees keep the minimum two demonstrations informative.
    fills_shown = set()
    for pair,f in zip(puzzle['train'],features['train']):
        info = _infer(pair['input'])
        if set(f['hole_fills']) == set(info['fills']):
            fills_shown.add('both')
    assert 'both' in fills_shown
    assert any(f['cross_witnesses'] and f['adjacent_cross_holes']
               and _without_witnesses(_infer(pair['input'])) > 1
               for pair,f in zip(puzzle['train'],features['train']))
    assert all(f['unfinished_cells'] > 0 for fs in features.values() for f in fs)


def self_test():
    """Bounded input-only semantic checks and geometry diversity over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = set(); features = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((np.asarray(p[k]).shape,np.asarray(p[k]).tobytes())
                             for s in ('train','test') for p in puzzle[s] for k in ('input','output')))
        for s in ('train','test'):
            for pair in puzzle[s]:
                info = _infer(pair['input'])
                features.append(_features(info))
                # Reordering extracted components must not select a different answer.
                annotation = _ring_demands(list(reversed(info['pieces'])),
                                           tuple(reversed(info['positions'])),pair['input'].shape)
                assert annotation == info['annotation']
                assert np.array_equal(solve(pair['input'].tolist()),pair['output'])
                # A loses cross-tile neighborhood ownership under this shortcut.
                if _features(info)['cross_witnesses']:
                    assert not np.array_equal(_local_only(info),pair['output'])
    assert len(signatures) == 42
    assert {f['corner'] for f in features} == set(_CORNERS)
    assert len({f['shape'] for f in features}) >= 25
    assert len({f['target_shape'] for f in features}) >= 25
    assert len({f['layout'] for f in features}) >= 100
    assert len({f['piece_dimensions'] for f in features}) >= 80
    assert len({f['piece_count'] for f in features}) >= 3
    assert len({f['hole_count'] for f in features}) >= 4
    assert any(f['external_ring'] for f in features)
    assert any(f['clipped'] for f in features)
    assert any(not f['clipped'] for f in features)
    assert any(f['target_shape'][0] > 1.5*f['target_shape'][1] for f in features)
    assert any(f['target_shape'][1] > 1.5*f['target_shape'][0] for f in features)
    for puzzle in puzzles:
        # Omitting B's printed-neighborhood evidence leaves multiple answers.
        assert any(_without_witnesses(_infer(p['input'])) > 1 for p in puzzle['train'])
        # Omitting A's translations changes the repair or creates ring conflicts.
        needs_assembly = False
        for pair in puzzle['train']:
            repaired = _without_assembly(_infer(pair['input']))
            needs_assembly |= repaired is None or not np.array_equal(repaired,pair['output'])
        assert needs_assembly
    # Role permutations preserve the whole answer, including the reused L color.
    permutation = np.array([6,3,8,0,9,1,7,4,2,5],dtype=np.int8)
    for puzzle in puzzles[:6]:
        pair = puzzle['test'][0]
        assert np.array_equal(solve(permutation[pair['input']]),permutation[pair['output']])
    again = generate_puzzle(7)
    assert all(np.array_equal(a[k],b[k]) for s in ('train','test')
               for a,b in zip(puzzles[7][s],again[s]) for k in ('input','output'))
    for a,b in ((2,1),(10,4)):
        validate_generated_puzzle(generate_puzzle(700+a*10+b,a,b))
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Reject labels that move the anchor, overwrite a center, or retain the L.
    for kind in ('shift','center','duplicate'):
        bad = {s:[{k:v.copy() for k,v in p.items()} for p in puzzles[0][s]] for s in ('train','test')}
        p = bad['test'][0]
        if kind == 'shift':
            p['output'] = np.roll(p['output'],1,axis=0)
        elif kind == 'center':
            info = _infer(p['input']);q = next(iter(info['annotation'][0]))
            p['output'][q] = info['fills'][0]
        else:
            bad['train'][1] = {k:v.copy() for k,v in bad['train'][0].items()}
        try:
            validate_generated_puzzle(bad)
        except (ValueError,AssertionError):
            pass
        else:
            raise AssertionError('Incorrect puzzle accepted')
    # Mixed-fill ring collisions are rejected, not assigned an ordering priority.
    probe = [dict(shape=(4,4),base=1,holes=((1,3),)),
             dict(shape=(4,4),base=2,holes=((1,0),))]
    assert _ring_demands(probe,((0,0),(0,4)),(8,8)) is None
    # Explicit hand-computed simultaneous and canvas-clipping regression.
    probe = [dict(shape=(3,4),base=1,holes=((0,0),(0,1))),
             dict(shape=(3,3),base=2,holes=())]
    positions = ((0,0),(0,4))
    annotation = _ring_demands(probe,positions,(3,7))
    expected = np.array([[8,8,2,1,2,2,2],
                         [2,2,2,1,2,2,2],
                         [1,1,1,1,2,2,2]],dtype=np.int8)
    assert np.array_equal(_paint(probe,positions,(3,7),0,8,annotation),expected)
    return dict(seeds=42,pairs=len(features),layouts=len({f['layout'] for f in features}),
                canvas_shapes=len({f['shape'] for f in features}),
                target_shapes=len({f['target_shape'] for f in features}))
