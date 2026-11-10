"""Register keyed fragments, then exchange glyph and segment arrangements.

Only the standard library and NumPy are used. The solver reads pixels alone.
The retained two-color scaffold makes fragment membership and registration visible.
"""
from collections import deque
from itertools import combinations
import hashlib
import json

import numpy as np


HOST_SEEDS = (0,676492781,284224165,109362842,557509814,719858710,481155084,499488646,301831239,597812514,395732214,979384923,612194713,641418642,332867985,127804767,170849107,216440846,117670442,544687096,940960818,573490022,173184232,449995417,134389958,403250490,59986970,430856635,576154987,890372351,807729331,950392221,504159685,149284618,887610627,519820138,345884295,330679015,768139340,580763821,411158306,924719600)
_NEIGHBORS = tuple((r,c) for r in (-1,0,1) for c in (-1,0,1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "grid must contain integer colors 0..9")
    return a


def _components(points):
    unseen = set(points)
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, group = [p], {p}
        while stack:
            r,c = stack.pop()
            for dr,dc in _NEIGHBORS:
                q = r+dr,c+dc
                if q in unseen:
                    unseen.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _bounds(points):
    return min(r for r,c in points),min(c for r,c in points),max(r for r,c in points)+1,max(c for r,c in points)+1


def _normalize(cells):
    r0,c0,_,_ = _bounds(cells)
    return {(r-r0,c-c0):int(v) for (r,c),v in cells.items()}


def _shape(cells):
    r0,c0,r1,c1 = _bounds(cells)
    return r1-r0,c1-c0


def _render(cells, background):
    cells = _normalize(cells)
    result = np.full(_shape(cells), background, dtype=np.int8)
    for p,v in cells.items():
        result[p] = v
    return result


def _line(a,b):
    dr,dc = b[0]-a[0],b[1]-a[1]
    _require(dr == 0 or dc == 0 or abs(dr) == abs(dc), "unsupported line slope")
    n = max(abs(dr),abs(dc))
    sr,sc = int(np.sign(dr)),int(np.sign(dc))
    return {(a[0]+k*sr,a[1]+k*sc) for k in range(n+1)}


def _figure(centers):
    return set().union(*(_line(a,b) for a,b in combinations(centers,2)))


def _stamp(centers,motif):
    return {(r+dr,c+dc) for r,c in centers for dr,dc in motif}


def _direction(endpoints):
    a,b = endpoints
    return "horizontal" if a[0] == b[0] else "vertical" if a[1] == b[1] else "diagonal"


def _windows(cells):
    """Dense multicolor signatures; whole overlap context is checked separately."""
    found = {}
    for h,w in ((2,3),(3,2)):
        for r,c in sorted(cells):
            points = [(r+i,c+j) for i in range(h) for j in range(w)]
            if all(p in cells for p in points):
                pattern = tuple(cells[p] for p in points)
                if len(set(pattern)) >= 2:
                    found.setdefault((h,w,pattern),[]).append((r,c))
    return found


def _matches(a,b,wa,wb):
    ah,aw = _shape(a)
    bh,bw = _shape(b)
    shifts = {(r-u,c-v) for key in wa.keys() & wb.keys()
              for r,c in wa[key] for u,v in wb[key]}
    candidates = []
    for dr,dc in sorted(shifts):
        r0,c0,r1,c1 = max(0,dr),max(0,dc),min(ah,dr+bh),min(aw,dc+bw)
        # A fragment is an observed rectangle, including its background holes.
        if any(a.get((r,c)) != b.get((r-dr,c-dc))
               for r in range(r0,r1) for c in range(c0,c1)):
            continue
        hit_a = frozenset(p for p in a if (p[0]-dr,p[1]-dc) in b)
        hit_b = frozenset((r-dr,c-dc) for r,c in hit_a)
        candidates.append(((dr,dc),hit_a,hit_b))
    # Remove only nested partial sightings of the SAME supported feature.
    # Do not maximize overlap or minimize the assembled rectangle.
    return [x for x in candidates if not any(x[1] <= y[1] and x[2] <= y[2]
             and (x[1] != y[1] or x[2] != y[2]) for y in candidates)]


def _register(parts):
    n = len(parts)
    _require(2 <= n <= 6, "need two to six separated keyed fragments")
    windows = [_windows(p) for p in parts]
    adjacency = [[] for _ in parts]
    edges = []
    for i,j in combinations(range(n),2):
        matches = _matches(parts[i],parts[j],windows[i],windows[j])
        if matches:
            ds = {m[0] for m in matches}
            adjacency[i].append((j,ds))
            adjacency[j].append((i,{(-r,-c) for r,c in ds}))
            edges.append((i,j,matches))
    offsets,solutions = {0:(0,0)},[]
    visits = 0

    def search():
        nonlocal visits
        visits += 1
        _require(visits <= 1000, "registration search bound exceeded")
        if len(solutions) >= 2:
            return
        if len(offsets) == n:
            union = {}
            for i,part in enumerate(parts):
                dr,dc = offsets[i]
                for (r,c),v in part.items():
                    q = r+dr,c+dc
                    if q in union and union[q] != v:
                        return
                    union[q] = v
            # Background inside every observed view is evidence too. Two
            # fragments can overlap without sharing a dense signature directly;
            # a third fragment must not fill a hole in either of those views.
            for i,part in enumerate(parts):
                dr,dc = offsets[i]
                ph,pw = _shape(part)
                if any(0 <= r-dr < ph and 0 <= c-dc < pw
                       and part.get((r-dr,c-dc)) != v
                       for (r,c),v in union.items()):
                    return
            if max(_shape(union)) <= 30:
                solutions.append((dict(offsets),union))
            return
        domains = []
        for j in range(n):
            if j in offsets:
                continue
            domain = None
            for i,deltas in adjacency[j]:
                if i in offsets:
                    options = {(offsets[i][0]-r,offsets[i][1]-c) for r,c in deltas}
                    domain = options if domain is None else domain & options
            if domain is not None:
                if not domain:
                    return
                domains.append((len(domain),j,domain))
        if not domains:
            return
        _,j,domain = min(domains,key=lambda x:(x[0],x[1]))
        for position in sorted(domain):
            offsets[j] = position
            search()
            del offsets[j]

    search()
    _require(len(solutions) == 1, "registration is disconnected, inconsistent or ambiguous")
    offsets,union = solutions[0]
    hits = []
    for i,j,matches in edges:
        delta = offsets[j][0]-offsets[i][0],offsets[j][1]-offsets[i][1]
        hits.extend((i,j,len(a),_shape(a)) for d,a,b in matches if d == delta)
    return union,offsets,hits


def _roles(cells):
    colors = set(cells.values())
    _require(len(colors) == 4, "scene needs two scaffold colors, one glyph color and one segment color")
    layers = {v:{p for p,x in cells.items() if x == v} for v in colors}
    candidates = []
    for edge in sorted(colors):
        points = sorted(layers[edge])
        if len(points) < 4 or _line_if_valid(points[0],points[-1]) != set(points):
            continue
        for glyph in sorted(colors-{edge}):
            groups = _components(layers[glyph])
            if len(groups) not in (2,3):
                continue
            centers,motifs = [],[]
            for group in groups:
                r0,c0,r1,c1 = _bounds(group)
                if (r1-r0)%2 != 1 or (c1-c0)%2 != 1:
                    break
                center = (r0+r1-1)//2,(c0+c1-1)//2
                centers.append(center)
                motifs.append(frozenset((r-center[0],c-center[1]) for r,c in group))
            if len(motifs) != len(groups) or len(set(motifs)) != 1:
                continue
            if len(motifs[0]) < 4 or min(_shape(motifs[0])) < 3:
                continue
            centers = tuple(sorted(centers))
            try:
                _figure(centers)
            except ValueError:
                continue
            if len(centers) == 3:
                a,b,c = centers
                if (b[0]-a[0])*(c[1]-a[1]) == (b[1]-a[1])*(c[0]-a[0]):
                    continue
            candidates.append({"edge":edge,"glyph":glyph,"endpoints":(points[0],points[-1]),
                               "centers":centers,"motif":motifs[0]})
    _require(len(candidates) == 1, "glyph/segment roles are ambiguous or missing")
    return candidates[0]


def _line_if_valid(a,b):
    try:
        return _line(a,b)
    except ValueError:
        return None


def _exchange(cells,role):
    scaffold = {p:v for p,v in cells.items() if v not in (role["edge"],role["glyph"])}
    figure = _figure(role["centers"])
    stamps = _stamp(role["endpoints"],role["motif"])
    _require(not figure & stamps, "exchanged roles collide")
    _require(not (figure | stamps) & scaffold.keys(), "exchange collides with scaffold")
    _require(len(_components(stamps)) == 2, "endpoint glyph copies merge")
    result = dict(scaffold)
    result.update({p:role["edge"] for p in figure})
    result.update({p:role["glyph"] for p in stamps})
    _require(max(_shape(result)) <= 30, "exchanged union exceeds 30")
    return result


def _analyze(input_grid):
    a = _grid(input_grid)
    values,counts = np.unique(a,return_counts=True)
    _require(counts.max()*2 > a.size, "background must occupy more than half the input")
    bg = int(values[counts.argmax()])
    _require(len(values) == 5, "expected five colors including background")
    points = {tuple(map(int,p)):int(a[tuple(p)]) for p in np.argwhere(a != bg)}
    groups = _components(points)
    parts = [_normalize({p:points[p] for p in group}) for group in groups]
    union,offsets,hits = _register(parts)
    role = _roles(union)
    out = _render(_exchange(union,role),bg)
    return {"background":bg,"parts":parts,"union":union,"offsets":offsets,
            "hits":hits,"role":role,"output":out}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate exact signatures, identify roles in the union, exchange and crop."""
    return _analyze(input_grid)["output"]


def _motif(rng,kind):
    if kind == "asymmetric":
        # An occupied bbox center, with an orientation-sensitive off-center arm.
        points = {(-1,-1),(-1,0),(-1,1),(0,0),(1,0),(1,1)}
    elif kind == "hollow":
        h,w = (3,3) if rng.random() < .5 else ((3,5) if rng.random() < .5 else (5,3))
        points = {(r,c) for r in range(-(h//2),h//2+1) for c in range(-(w//2),w//2+1)
                  if abs(r) == h//2 or abs(c) == w//2}
    elif kind == "cross":
        h,w = int(rng.choice([3,5])),int(rng.choice([3,5]))
        points = {(r,0) for r in range(-(h//2),h//2+1)} | {(0,c) for c in range(-(w//2),w//2+1)}
    else:
        # Odd-box irregular glyphs, independently grown from a full cross.
        h,w = ((3,5) if rng.random() < .5 else (5,3))
        points = {(r,0) for r in range(-(h//2),h//2+1)} | {(0,c) for c in range(-(w//2),w//2+1)}
        for r in range(-(h//2),h//2+1):
            for c in range(-(w//2),w//2+1):
                if rng.random() < .25:
                    points.add((r,c))
    # This transforms the sampled prototype; solving only translates its copies.
    k,reflect = int(rng.integers(4)),bool(rng.integers(2))
    transformed = set()
    for r,c in sorted(points):
        if reflect:
            c = -c
        for _ in range(k):
            r,c = c,-r
        transformed.add((r,c))
    return frozenset(transformed)


def _translated(points,rng,h,w,motif):
    all_points = _stamp(points,motif)
    r0,c0,r1,c1 = _bounds(all_points)
    if r1-r0 > h or c1-c0 > w:
        return None
    dr,dc = int(rng.integers(-r0,h-r1+1)),int(rng.integers(-c0,w-c1+1))
    return tuple((r+dr,c+dc) for r,c in points)


def _scene(rng,count,motif,direction):
    for _ in range(160):
        h,w = int(rng.integers(14,21)),int(rng.integers(14,21))
        d = int(rng.integers(5,10))
        if count == 2:
            dr,dc = ((0,1),(1,0),(1,1),(1,-1))[int(rng.integers(4))]
            relative = ((0,0),(dr*d,dc*d))
        elif rng.random() < .3:
            d = int(rng.integers(4,7))
            relative = ((0,0),(d,d),(d,-d))
        else:
            relative = ((0,0),(0,d),(d,0))
        # Independently orient the center arrangement, leaving glyph orientation intact.
        for _ in range(int(rng.integers(4))):
            relative = tuple((c,-r) for r,c in relative)
        centers = _translated(relative,rng,h,w,motif)
        distance = int(rng.integers(4,11))
        dr,dc = {"horizontal":(0,1),"vertical":(1,0),"diagonal":(1,int(rng.choice([-1,1])))}[direction]
        endpoints = _translated(((0,0),(distance*dr,distance*dc)),rng,h,w,motif)
        if centers is None or endpoints is None:
            continue
        glyphs,segment = _stamp(centers,motif),_line(*endpoints)
        if len(_components(glyphs)) != count:
            continue
        halo = {(r+dr,c+dc) for r,c in glyphs for dr in (-1,0,1) for dc in (-1,0,1)}
        if segment & halo:
            continue
        figure,stamps = _figure(centers),_stamp(endpoints,motif)
        if figure & stamps or len(_components(stamps)) != 2:
            continue
        return h,w,centers,endpoints
    raise ValueError("could not sample a feasible role scene")


def _views(h,w,n,rng):
    """Overlapping rectangles with a chain/branch/cycle of seam keys."""
    r,c = int(rng.integers(h//2-1,h//2+1)),int(rng.integers(w//2-1,w//2+1))
    if n == 2:
        if rng.random() < .5:
            return [(0,0,h,c+3),(0,c,h,w)],[(0,c,h,c+3)]
        return [(0,0,r+3,w),(r,0,h,w)],[(r,0,r+3,w)]
    if n == 3:
        if rng.random() < .5:
            return [(0,0,h,c+3),(0,c,r+3,w),(r,c,h,w)],[(0,c,r,c+3),(r,c+3,r+3,w)]
        return [(0,0,r+3,w),(r,0,h,c+3),(r,c,h,w)],[(r,0,r+3,c),(r+3,c,h,c+3)]
    return [(0,0,r+3,c+3),(0,c,r+3,w),(r,0,h,c+3),(r,c,h,w)],[(0,c,r,c+3),(r,0,r+3,c),(r,c+3,r+3,w)] + ([(r+3,c,h,c+3)] if rng.random() < .5 else [])


def _in(p,box):
    r,c = p
    r0,c0,r1,c1 = box
    return r0 <= r < r1 and c0 <= c < c1


def _glyph_boxes(centers,motif):
    r0,c0,r1,c1 = _bounds(motif)
    return {(r+i,c+j) for r,c in centers for i in range(r0,r1) for j in range(c0,c1)}


def _connect_view(cells,box,banned,base,rng):
    # Connect observed components using short paths in visible free space.
    # Multi-source BFS is bounded by the at-most-400-cell scene.
    for _ in range(35):
        groups = _components(p for p in cells if _in(p,box))
        if len(groups) <= 1:
            return bool(groups)
        group = groups[0]
        targets = set().union(*groups[1:])
        start = sorted(group)
        rng.shuffle(start)
        queue,parent = deque(start),{p:None for p in start}
        moves = list(_NEIGHBORS)
        rng.shuffle(moves)
        found = None
        while queue and found is None:
            r,c = queue.popleft()
            for dr,dc in moves:
                p = r+dr,c+dc
                if not _in(p,box) or p in parent:
                    continue
                if p in targets:
                    parent[p] = (r,c)
                    found = p
                    break
                if p in banned:
                    continue
                parent[p] = (r,c)
                queue.append(p)
        if found is None:
            return False
        p = parent[found]
        while p is not None:
            if p not in cells:
                cells[p] = base
            p = parent[p]
    return False


def _scaffold(rng,h,w,centers,endpoints,motif,colors,n):
    base,pin,glyph,edge = colors
    glyphs,segment = _stamp(centers,motif),_line(*endpoints)
    figure,stamps = _figure(centers),_stamp(endpoints,motif)
    cells = {p:glyph for p in glyphs}
    cells.update({p:edge for p in segment})
    # Reserve both old and new glyph rectangles, including every empty motif cell.
    banned = glyphs | segment | figure | stamps | _glyph_boxes(centers+endpoints,motif)
    views,seams = _views(h,w,n,rng)
    codes = set()
    for box in seams:
        r0,c0,r1,c1 = box
        shapes = [(a,b) for a,b in ((2,3),(3,2),(3,3)) if a <= r1-r0 and b <= c1-c0]
        candidates = []
        for kh,kw in shapes:
            for r in range(r0,r1-kh+1):
                for c in range(c0,c1-kw+1):
                    footprint = {(r+i,c+j) for i in range(kh) for j in range(kw)}
                    if not footprint & (banned | cells.keys()):
                        candidates.append((r,c,kh,kw))
        if not candidates:
            raise ValueError("no free seam signature position")
        r,c,kh,kw = candidates[int(rng.integers(len(candidates)))]
        for _ in range(30):
            pattern = tuple(int(x) for x in rng.integers(0,2,size=kh*kw))
            if 1 < sum(pattern) < len(pattern)-1 and (kh,kw,pattern) not in codes:
                codes.add((kh,kw,pattern))
                break
        else:
            raise ValueError("could not sample distinct signatures")
        for i in range(kh):
            for j in range(kw):
                cells[r+i,c+j] = (base,pin)[pattern[i*kw+j]]
    # New routes in one view can add a clipped branch to another. Iterate until
    # every returned view is one foreground fragment, with an explicit bound.
    for _ in range(6):
        for box in views:
            if not _connect_view(cells,box,banned,base,rng):
                raise ValueError("scaffold cannot connect this view without painting a role")
        if all(len(_components(p for p in cells if _in(p,box))) == 1 for box in views):
            break
    else:
        raise ValueError("view connectivity did not converge")
    parts = [_normalize({p:v for p,v in cells.items() if _in(p,box)}) for box in views]
    # Every view must add material, and some role must actually be cut.
    _require(all(len(p) < len(cells) for p in parts), "uncut or redundant full view")
    role_cells = glyphs | segment
    _require(all(not all(_in(p,box) for p in role_cells) for box in views), "one fragment contains the complete exchange scene")
    _require(any(0 < sum(_in(p,box) for p in segment) < len(segment) for box in views)
             or any(0 < sum(_in(p,box) for p in _stamp((center,),motif)) < len(motif)
                    for center in centers for box in views), "roles were not fractured")
    union,_,_ = _register(parts)
    _require(_normalize(union) == _normalize(cells), "signatures do not uniquely reconstruct intended scene")
    return cells,parts


def _pack(parts,bg,rng):
    shapes = [_shape(p) for p in parts]
    for attempt in range(28):
        if attempt < 16:
            h = int(rng.integers(max(s[0] for s in shapes),31))
            w = int(rng.integers(max(s[1] for s in shapes),31))
        else:
            h,w = 30,30
        if h*w <= 2*sum(map(len,parts)):
            continue
        occupied = np.zeros((h,w),dtype=bool)
        grid = np.full((h,w),bg,dtype=np.int8)
        order = sorted(range(len(parts)),key=lambda i:-(shapes[i][0]*shapes[i][1]+float(rng.random())*20))
        success = True
        for i in order:
            ph,pw = shapes[i]
            positions = [(r,c) for r in range(h-ph+1) for c in range(w-pw+1)]
            rng.shuffle(positions)
            chosen = None
            rows = np.array([p[0] for p in parts[i]])
            cols = np.array([p[1] for p in parts[i]])
            for r,c in positions:
                if not occupied[rows+r,cols+c].any():
                    chosen = r,c
                    break
            if chosen is None:
                success = False
                break
            r,c = chosen
            for (dr,dc),v in parts[i].items():
                rr,cc = r+dr,c+dc
                grid[rr,cc] = v
                occupied[max(0,rr-1):min(h,rr+2),max(0,cc-1):min(w,cc+2)] = True
        if success and len(_components(map(tuple,np.argwhere(grid != bg)))) == len(parts):
            return grid
    raise ValueError("could not pack separated fragments")


def _make_pair(rng,n,count,kind,direction):
    last_error = ""
    for _ in range(100):
        colors = [int(x) for x in rng.choice(10,size=5,replace=False)]
        bg,base,pin,glyph,edge = colors
        motif = _motif(rng,kind)
        try:
            h,w,centers,endpoints = _scene(rng,count,motif,direction)
            cells,parts = _scaffold(rng,h,w,centers,endpoints,motif,(base,pin,glyph,edge),n)
            role = _roles(cells)
            intended = _render(_exchange(cells,role),bg)
            inp = _pack(parts,bg,rng)
            analysis = _analyze(inp)
            _require(np.array_equal(analysis["output"],intended), "packed reconstruction changed result")
            # Registration must not be replaceable by using the original packed positions.
            raw = {tuple(map(int,p)):int(inp[tuple(p)]) for p in np.argwhere(inp != bg)}
            try:
                raw_role = _roles(raw)
                wrong = _render(_exchange(raw,raw_role),bg)
                _require(not np.array_equal(wrong,intended), "registration-free exchange accidentally works")
            except ValueError as exc:
                if str(exc) == "registration-free exchange accidentally works":
                    raise
            return {"input":inp,"output":intended}
        except ValueError as exc:
            last_error = str(exc)
    raise RuntimeError("bounded pair generation exhausted: "+last_error)


def _counts(a,b):
    for name,x,lo,hi in (("num_train",a,2,10),("num_test",b,1,4)):
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or not lo <= x <= hi:
            raise ValueError(name+" has an illegal count")
    _require(a > b, "num_train must exceed num_test")
    return int(a),int(b)


def _key(grid):
    a = np.asarray(grid,dtype=np.int8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,nq = _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    # The minimal demonstrations cross count, fragmentation and motif structure.
    first_counts = [2,3]
    first_fragments = [2,3]
    first_kinds = ["asymmetric","hollow"]
    rng.shuffle(first_counts)
    rng.shuffle(first_fragments)
    rng.shuffle(first_kinds)
    directions = list(rng.choice(["horizontal","vertical","diagonal"],size=2,replace=False))
    puzzle,seen = {"train":[],"test":[]},set()
    for split,n in (("train",nt),("test",nq)):
        for i in range(n):
            n_parts = first_fragments[i] if split == "train" and i < 2 else int(rng.integers(2,4)) if split == "train" else 4
            count = first_counts[i] if split == "train" and i < 2 else int(rng.integers(2,4))
            kind = first_kinds[i] if split == "train" and i < 2 else str(rng.choice(["asymmetric","hollow","cross","irregular"]))
            direction = directions[i] if split == "train" and i < 2 else str(rng.choice(["horizontal","vertical","diagonal"]))
            for _ in range(8):
                pair = _make_pair(rng,n_parts,count,kind,direction)
                k = _key(pair["input"])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input retry bound exceeded")
    return puzzle


def _pair_info(pair):
    _require(isinstance(pair,dict) and set(pair) == {"input","output"}, "pair schema")
    inp,out = _grid(pair["input"]),_grid(pair["output"])
    info = _analyze(inp)
    _require(np.array_equal(out,info["output"]), "incorrect exchange or crop")
    bg,union,role = info["background"],info["union"],info["role"]
    _require(set(map(int,np.unique(out))) == set(map(int,np.unique(inp))), "palette changed")
    _require(not np.array_equal(_render(union,bg),out), "plain registered union shortcut works")
    _require(all(np.any(edge != bg) for edge in (out[0],out[-1],out[:,0],out[:,-1])), "output is not tightly cropped")
    scaffold = {p:v for p,v in union.items() if v not in (role["edge"],role["glyph"])}
    exchanged = _exchange(union,role)
    _require(all(exchanged[p] == v for p,v in scaffold.items()), "registration scaffold changed")
    _require(set(exchanged.values()) == set(union.values()), "lost a semantic color role")
    # Compare coordinate maps, not only crops, for source erasure and exact stamping.
    actual_glyph = {p for p,v in exchanged.items() if v == role["glyph"]}
    actual_edge = {p for p,v in exchanged.items() if v == role["edge"]}
    _require(actual_glyph == _stamp(role["endpoints"],role["motif"]), "glyph shape or endpoint placement changed")
    _require(actual_edge == _figure(role["centers"]), "center connections incorrect")
    source = {p for p,v in union.items() if v in (role["edge"],role["glyph"])}
    _require(source - (actual_glyph | actual_edge), "no observable source erasure")
    _require((actual_glyph | actual_edge) - source, "exchange adds no new placement")
    raw = {tuple(map(int,p)):int(inp[tuple(p)]) for p in np.argwhere(inp != bg)}
    try:
        wrong = _render(_exchange(raw,_roles(raw)),bg)
    except ValueError:
        pass
    else:
        _require(not np.array_equal(wrong,out), "registration is unnecessary")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s],list) for s in ("train","test")), "splits must be lists")
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen,infos = set(),{"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            info = _pair_info(pair)
            k = _key(pair["input"])
            _require(k not in seen, "duplicate input")
            seen.add(k)
            infos[split].append(info)
    train = infos["train"]
    _require({len(i["role"]["centers"]) for i in train} == {2,3}, "demos must teach segment and triangle")
    _require({len(i["parts"]) for i in train} >= {2,3}, "demos must teach multi-fragment registration")
    _require(len({_direction(i["role"]["endpoints"]) for i in train}) >= 2, "demos need multiple source directions")
    _require(any((0,0) not in i["role"]["motif"] for i in train), "hollow center anchor not demonstrated")
    _require(any(_asymmetric(i["role"]["motif"]) for i in train), "orientation-sensitive copying not demonstrated")
    _require(all(len(i["parts"]) == 4 for i in infos["test"]), "test fragmentation shift missing")
    return None


def _asymmetric(motif):
    return frozenset((-r,-c) for r,c in motif) != frozenset(motif)


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                a = np.asarray(pair[role],dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 host seeds; streaming aggregates only, no files or saved traces."""
    stats = {"puzzles":0,"pairs":0,"input_shapes":set(),"output_shapes":set(),"motifs":set(),
             "backgrounds":set(),"signatures":set(),"directions":set(),"fragment_counts":set(),
             "glyph_counts":set(),"overlap_shapes":set(),"hollow_pairs":0,"asymmetric_pairs":0,
             "split_glyph_pairs":0,"split_segment_pairs":0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        stats["puzzles"] += 1
        stats["signatures"].add(_digest(puzzle))
        for split in ("train","test"):
            for pair in puzzle[split]:
                info = _pair_info(pair)
                role = info["role"]
                stats["pairs"] += 1
                stats["input_shapes"].add(pair["input"].shape)
                stats["output_shapes"].add(pair["output"].shape)
                stats["motifs"].add(tuple(sorted(role["motif"])))
                stats["backgrounds"].add(info["background"])
                stats["directions"].add(_direction(role["endpoints"]))
                stats["fragment_counts"].add(len(info["parts"]))
                stats["glyph_counts"].add(len(role["centers"]))
                stats["overlap_shapes"].update(x[3] for x in info["hits"])
                stats["hollow_pairs"] += (0,0) not in role["motif"]
                stats["asymmetric_pairs"] += _asymmetric(role["motif"])
                stats["split_segment_pairs"] += all(sum(v == role["edge"] for v in p.values()) < sum(v == role["edge"] for v in info["union"].values()) for p in info["parts"])
                # A cut glyph appears as a smaller per-color component somewhere.
                stats["split_glyph_pairs"] += any(0 < len(g) < len(role["motif"]) for p in info["parts"] for g in _components(q for q,v in p.items() if v == role["glyph"]))
        if seed == 0:
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic seed")
            _metamorphic_checks(puzzle["train"][0])
    _require(len(stats["signatures"]) == 42, "seed diversity missing")
    _require(len(stats["input_shapes"]) >= 25 and len(stats["output_shapes"]) >= 25, "shape diversity missing")
    _require(len(stats["motifs"]) >= 15 and len(stats["backgrounds"]) == 10, "glyph/palette diversity missing")
    _require(stats["fragment_counts"] == {2,3,4} and stats["glyph_counts"] == {2,3}, "structural counts missing")
    _require(len(stats["directions"]) == 3 and len(stats["overlap_shapes"]) >= 10, "registration geometry diversity missing")
    _require(stats["split_glyph_pairs"] >= 20 and stats["split_segment_pairs"] >= 10, "causal role reconstruction missing")
    for a,b in ((2,1),(10,4)):
        validate_generated_puzzle(generate_puzzle(17,a,b))
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A bare signature that could be absorbed at either of two equal sightings
    # has no unique translation. Neither overlap size nor crop area breaks ties.
    key = {(0,0):1,(0,1):2,(0,2):1,(1,0):2,(1,1):1,(1,2):2}
    repeated = dict(key)
    repeated.update({(r,c+6):v for (r,c),v in key.items()})
    repeated.update({(2,c):1 for c in range(9)})
    try:
        _register([repeated,key])
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous signature accepted")
    # Three views have unique A-B and B-C keys, but C paints a background
    # cell observed by A. A and C share no dense key, so checking only the
    # directly matched pairs would miss this negative-space contradiction.
    other = {(-4,6):2,(-4,7):2,(-4,8):1,
             (-3,6):1,(-3,7):1,(-3,8):2}
    a,b,c = dict(key),dict(key),dict(other)
    a.update({(r,0):1 for r in range(2,8)})
    a.update({(7,j):1 for j in range(10)})
    b.update(other)
    b.update({(r,5):1 for r in range(-3,0)})
    b.update({(-1,j):1 for j in range(6)})
    c.update({(-4,j):1 for j in range(9,13)})
    c.update({(r,12):1 for r in range(-3,6)})
    c.update({(5,j):1 for j in range(8,13)})
    parts = list(map(_normalize,(a,b,c)))
    matches = [len(_matches(parts[i],parts[j],_windows(parts[i]),_windows(parts[j])))
               for i,j in ((0,1),(1,2),(0,2))]
    _require(matches == [1,1,0], "negative-space regression lost its key chain")
    try:
        _register(parts)
    except ValueError:
        pass
    else:
        raise AssertionError("unmatched view's empty cells overwritten")
    # Report only aggregates, never duplicate reconstructible grid payloads.
    return {k:(len(v) if isinstance(v,set) else v) for k,v in stats.items()}


def _metamorphic_checks(pair):
    inp,out = pair["input"],pair["output"]
    info = _analyze(inp)
    # Input placement and margins carry no assembled geometry.
    moved = _pack(info["parts"],info["background"],np.random.default_rng(9001))
    _require(np.array_equal(solve(moved),out), "repacking changes the answer")
    permutation = np.array([9,4,2,7,0,8,1,6,3,5],dtype=np.int8)
    _require(np.array_equal(solve(permutation[inp]),permutation[out]), "literal colors leak semantic roles")
    _require(np.array_equal(solve(inp.T),out.T), "transposition changes semantics")
    # Mutation must fail validation, rather than being trusted as a generated label.
    bad = out.copy()
    bad.flat[0] = (int(bad.flat[0])+1)%10
    try:
        _pair_info({"input":inp,"output":bad})
    except ValueError:
        pass
    else:
        raise AssertionError("mutated label accepted")


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",",":")))
