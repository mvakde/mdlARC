"""Signal-addressed jigsaw receivers: input-only rays plus colored exact cover.

The sampler uses layered polyominoes, but the solver searches arbitrary
translation-only placements. No parent generator or stored puzzle is imported.
"""
from collections import Counter, defaultdict
from math import gcd
import numpy as np


_PORTS = {(r, c): ((-1 if r == -1 else 1 if r == 2 else 0),
                   (-1 if c == -1 else 1 if c == 2 else 0))
          for r in range(-1, 3) for c in range(-1, 3)
          if r in (-1, 2) or c in (-1, 2)}
_KINDS = ("diagonals", "perpendicular", "staircase")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((a >= 0) & (a <= 9)), "grid colors")
    return a.astype(np.int64, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(a, eligible):
    remaining = set(map(tuple, np.argwhere(eligible)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        color = int(a[first])
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining and a[p] == color:
                    remaining.remove(p)
                    stack.append(p)
        result.append((np.array(sorted(cells), dtype=int), color))
    return result


def _rim(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
            if r in (r0, r1) or c in (c0, c1)}


def _in_cavity(point, box):
    r, c = point
    r0, c0, r1, c1 = box
    return r0 < r < r1 and c0 < c < c1


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(sum(counts == counts[bg]) == 1, "background must be the unique modal color")
    receivers = []
    for cells, color in _components(a, a != bg):
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        if np.any(hi-lo+1 < 5):
            continue
        box = (*map(int, lo), *map(int, hi))
        r0, c0, r1, c1 = box
        rim = _rim(box)
        if not np.all(a[r0+1:r1, c0+1:c1] == bg):
            continue
        if not all(a[p] in (bg, color) for p in rim):
            continue
        gates = {p for p in rim if a[p] == bg}
        if not 2 <= len(gates) <= 3:
            continue
        if not all(a[p] == color for p in ((r0,c0),(r0,c1),(r1,c0),(r1,c1))):
            continue
        if not any(all(p[axis] == edge for p in gates)
                   for axis, edges in ((0,(r0,r1)),(1,(c0,c1))) for edge in edges):
            continue
        # Several gates can detach short portions of the same rectangular rim.
        # The long U-shaped component identifies its box; all rim cells are
        # checked by geometry, rather than demanding one connected component.
        if not set(map(tuple, cells)) <= rim-gates:
            continue
        receivers.append((box, color, gates))
    _require(len(receivers) == 1, "one empty rectangular receiver frame required")
    box, structure, gates = receivers[0]
    fixed = _rim(box)-gates
    launchers = []
    for r in range(1, a.shape[0]-2):
        for c in range(1, a.shape[1]-2):
            color = int(a[r,c])
            if color in (bg, structure) or not np.all(a[r:r+2,c:c+2] == color):
                continue
            ports = [((r+dr,c+dc),v) for (dr,dc),v in _PORTS.items()
                     if a[r+dr,c+dc] == bg]
            if len(ports) != 1:
                continue
            if sum(a[r+dr,c+dc] == structure for dr,dc in _PORTS) != 11:
                continue
            footprint = {(rr,cc) for rr in range(r-1,r+3) for cc in range(c-1,c+3)}
            _require(not footprint & fixed, "launcher overlaps receiver or another launcher")
            fixed |= footprint-{ports[0][0]}
            launchers.append({"core":(r,c), "color":color, "port":ports[0][0],
                              "v":ports[0][1], "footprint":footprint})
    _require(2 <= len(launchers) <= 3, "two or three complete one-port launchers required")
    _require(len({l['color'] for l in launchers}) == len(launchers), "distinct signal colors required")
    accounted = np.zeros(a.shape, dtype=bool)
    for p in fixed:
        accounted[p] = True
    _require(set(map(tuple,np.argwhere(a == structure))) ==
             {p for p in fixed if a[p] == structure}, "unaccounted structural cells")
    pieces = _components(a, (a != bg) & ~accounted)
    _require(4 <= len(pieces) <= 9, "four to nine movable components required")
    _require(all(color != structure for _,color in pieces), "pieces cannot use structure color")
    _require(sum(len(cells) for cells,_ in pieces) == (box[2]-box[0]-1)*(box[3]-box[1]-1),
             "piece area must equal cavity area")
    base = a.copy()
    for cells,_ in pieces:
        base[cells[:,0],cells[:,1]] = bg
    # Pieces are isolated from all other foreground, including same-color cores.
    owner = np.full(a.shape,-1,dtype=int)
    for i,(cells,_) in enumerate(pieces):
        owner[cells[:,0],cells[:,1]] = i
    for i,(cells,_) in enumerate(pieces):
        for r,c in cells:
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    rr,cc = r+dr,c+dc
                    if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]:
                        _require(a[rr,cc] == bg or owner[rr,cc] == i, "movable pieces must be separated")
    return {"bg":bg, "structure":structure, "box":box, "gates":gates,
            "launchers":launchers, "pieces":pieces, "base":base}


def _walk(start, velocity, shape, box=None, phase="floor"):
    vr,vc = velocity
    scale = max(abs(vr),abs(vc))
    points = []
    for t in range(3*max(shape)+1):
        bias = scale-1 if phase == 'ceil' else 0
        p = (start[0]+(1 if vr >= 0 else -1)*((t*abs(vr)+bias)//scale),
             start[1]+(1 if vc >= 0 else -1)*((t*abs(vc)+bias)//scale))
        if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]):
            break
        points.append(p)
        if box is not None and _in_cavity(p,box):
            break
    return points


def _hit(paths_a, paths_b, box):
    hits = []
    for ia,a in enumerate(paths_a):
        lookup = {p:t for t,p in enumerate(a) if not _in_cavity(p,box)}
        mids = {(p[0]+q[0],p[1]+q[1]):t for t,(p,q) in enumerate(zip(a,a[1:]))
                if p[0] != q[0] and p[1] != q[1] and not _in_cavity(q,box)}
        for ib,b in enumerate(paths_b):
            for s,p in enumerate(b):
                if p in lookup:
                    hits.append((lookup[p]-1,s-1,ia,ib,"cell"))
            for s,(p,q) in enumerate(zip(b,b[1:])):
                if p[0] == q[0] or p[1] == q[1] or _in_cavity(q,box):
                    continue
                mid = (p[0]+q[0],p[1]+q[1])
                if mid in mids:
                    t = mids[mid]
                    if {a[t],a[t+1]} != {p,q}:
                        hits.append((t,s,ia,ib,"between"))
    return min(hits,key=lambda h:(sum(h[:2]),max(h[:2]),h[2:])) if hits else None


def _trace(info, shape, variant="correct"):
    """Spatial encounters; the receiver absorbs each ray at its first interior cell."""
    box,base,bg = info['box'],info['base'],info['bg']
    groups = [{'v':l['v'],'members':[(i,l['port'])]} for i,l in enumerate(info['launchers'])]
    occupied, last, anchors, events = {}, {}, {}, []
    used_gates = set()

    def paint(sid,path,terminal=False):
        color = info['launchers'][sid]['color']
        for p in path:
            _require(base[p] == bg, "ray hits a fixed object")
            _require(p not in occupied or (occupied[p] == sid and last.get(sid) == p), "ray overlap")
            if _in_cavity(p,box):
                _require(terminal and p == path[-1], "receiver must absorb at first interior cell")
                _require(p not in anchors, "two signals have the same anchor")
                anchors[p] = color
            else:
                occupied[p] = sid
                last[sid] = p
                if p in info['gates']:
                    used_gates.add(p)

    for _ in range(4):
        forecasts = [[_walk(p,g['v'],shape,box,'ceil' if variant == 'ceil' else 'floor')
                      for _,p in g['members']] for g in groups]
        candidates = []
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                hit = _hit(forecasts[i],forecasts[j],box)
                if hit is not None and variant != 'straight':
                    candidates.append((i,j,hit))
        if not candidates:
            for group,paths in zip(groups,forecasts):
                for (sid,_),path in zip(group['members'],paths):
                    _require(path and _in_cavity(path[-1],box), "every ray must reach the receiver")
                    _require(len(path) >= 2 and path[-2] in info['gates'], "ray must enter through a gate")
                    paint(sid,path,True)
            break
        first = {}
        for i,j,hit in candidates:
            first[i] = min(first.get(i,1000),hit[0])
            first[j] = min(first.get(j,1000),hit[1])
        ready = [(i,j,h) for i,j,h in candidates if h[0] == first[i] and h[1] == first[j]]
        _require(bool(ready), "cyclic encounter order")
        i,j,hit = min(ready,key=lambda e:(max(e[2][:2]),sum(e[2][:2]),e[:2]))
        _require(sum(k in (i,j) and h[0] == first[k] or l in (i,j) and h[1] == first[l]
                     for k,l,h in candidates) == 1, "multiway encounter")
        va,vb = groups[i]['v'],groups[j]['v']
        _require(max(map(abs,va)) == max(map(abs,vb)) == 1, "no staircase re-encounters")
        sa,sb = sum(map(abs,va)),sum(map(abs,vb))
        if sa == sb == 1:
            _require(va[0]*vb[0]+va[1]*vb[1] == 0, "cardinal encounter must be perpendicular")
            kind = 'perpendicular'
        elif sa == sb == 2:
            kind = 'diagonals'
        else:
            _require(va[0]*vb[0]+va[1]*vb[1] > 0, "cardinal/diagonal encounter must be acute")
            kind = 'staircase'
        velocity = (va[0]+vb[0],va[1]+vb[1])
        divisor = gcd(abs(velocity[0]),abs(velocity[1]))
        _require(divisor > 0 and min(hit[:2]) >= 0, "cancellation or origin encounter")
        velocity = tuple(x//divisor for x in velocity)
        if variant == 'sign':
            velocity = tuple(int(np.sign(x)) for x in velocity)
        members = []
        for index,cut in ((i,hit[0]),(j,hit[1])):
            for (sid,_),path in zip(groups[index]['members'],forecasts[index]):
                _require(cut < len(path)-1, "clipped bundle")
                paint(sid,path[:cut+1])
                members.append((sid,path[cut]))
        events.append({'kind':kind,'intersection':hit[4],'cuts':hit[:2],
                       'incoming':(va,vb),'outgoing':velocity})
        _require(len(events) <= 1, "family permits at most one collision")
        groups = [g for k,g in enumerate(groups) if k not in (i,j)]
        groups.append({'v':velocity,'members':members})
    else:
        raise ValueError("ray tracing did not terminate")
    _require(used_gates == info['gates'] and len(anchors) == len(info['launchers']), "unserved gate")
    return occupied,anchors,events


def _assemblies(info, anchors, max_answers=2):
    """Colored exact covers of the entire cavity, deduplicating identical tiles."""
    r0,c0,r1,c1 = info['box']
    height,width = r1-r0-1,c1-c0-1
    pins = {(r-r0-1)*width+c-c0-1:color for (r,c),color in anchors.items()}
    grouped = defaultdict(list)
    for i,(cells,color) in enumerate(info['pieces']):
        local = cells-cells.min(axis=0)
        grouped[(color,tuple(map(tuple,local)))].append(i)
    types = list(grouped)
    remaining = [len(grouped[t]) for t in types]
    by_cell = [[] for _ in range(height*width)]
    candidates = []
    for t,(color,mask) in enumerate(types):
        ph = max(p[0] for p in mask)+1
        pw = max(p[1] for p in mask)+1
        for r in range(height-ph+1):
            for c in range(width-pw+1):
                cells = tuple((r+dr)*width+c+dc for dr,dc in mask)
                if any(p in pins and pins[p] != color for p in cells):
                    continue
                bits = sum(1 << int(p) for p in cells)
                cid = len(candidates)
                candidates.append((t,bits,cells))
                for p in cells:
                    by_cell[p].append(cid)
    full = (1 << (height*width))-1
    answers = {}

    def search(occupied,chosen):
        if len(answers) >= max_answers:
            return
        if occupied == full:
            if any(remaining):
                return
            answer = np.full((height,width),info['bg'],dtype=int)
            placements = []
            used = Counter()
            for cid in chosen:
                t,_,cells = candidates[cid]
                color,mask = types[t]
                sid = grouped[types[t]][used[t]]
                used[t] += 1
                coords = np.array([(p//width+r0+1,p%width+c0+1) for p in cells],dtype=int)
                answer.flat[list(cells)] = color
                placements.append((sid,coords))
            answers.setdefault(answer.tobytes(),(answer,placements))
            return
        best = None
        for p in range(height*width):
            if occupied & (1 << p):
                continue
            options = [cid for cid in by_cell[p] if remaining[candidates[cid][0]]
                       and not occupied & candidates[cid][1]]
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        for cid in best:
            t,bits,_ = candidates[cid]
            remaining[t] -= 1
            search(occupied | bits,chosen+[cid])
            remaining[t] += 1
            if len(answers) >= max_answers:
                break

    search(0,[])
    return list(answers.values())


def _analyze_and_solve(a):
    info = _parse(a)
    rays,anchors,events = _trace(info,a.shape)
    _require(all(a[p] == info['bg'] for p in rays), "original tiles must not obstruct signal routes")
    answers = _assemblies(info,anchors)
    _require(len(answers) == 1, "signals and shapes must determine one colored exact cover")
    mosaic,placements = answers[0]
    result = info['base'].copy()
    for p,sid in rays.items():
        result[p] = info['launchers'][sid]['color']
    r0,c0,r1,c1 = info['box']
    result[r0+1:r1,c0+1:c1] = mosaic
    info.update(rays=rays,anchors=anchors,events=events,placements=placements)
    return result,info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the full answer from the visible input alone."""
    return _analyze_and_solve(_grid(input_grid))[0]


def _dilate(mask):
    padded = np.pad(mask,1)
    result = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            result |= padded[dr:dr+mask.shape[0],dc:dc+mask.shape[1]]
    return result


def _profile(rng,width,amplitude):
    mode = int(rng.integers(3))
    if mode == 0:
        p = rng.integers(amplitude+1,size=width)
    elif mode == 1:
        p = np.floor(np.linspace(0,amplitude,width)).astype(int)
        if rng.random() < .5:
            p = p[::-1]
    else:
        p = np.zeros(width,dtype=int)
        p[int(rng.integers(width)):] = amplitude
    p = p-p.min()
    if not np.any(p):
        p[int(rng.integers(width))] = 1
    return p


def _cuts(rng,width,layers):
    cuts = [np.zeros(width,dtype=int)]
    for _ in range(layers-1):
        p = _profile(rng,width,int(rng.integers(1,3)))
        cuts.append(p+int(cuts[-1].max())+int(rng.integers(1,3)))
    cuts.append(np.full(width,int(cuts[-1].max())+int(rng.integers(1,3))))
    return cuts


def _to_row(start,v,row):
    """Unbounded canonical construction route, ending one row inside the receiver."""
    scale = max(map(abs,v))
    path = []
    for t in range(80):
        p = (start[0]-(t*abs(v[0])//scale),start[1]+(t*abs(v[1])//scale))
        path.append(p)
        if p[0] == row:
            return path
    raise ValueError("construction route")


def _make_pair(rng,kind,groups,layers,palette,rotation,reflection,force_unequal=False):
    bg,structure,*colors = palette
    width = int(rng.integers(3,5))
    cuts = _cuts(rng,width,layers)
    depth = int(cuts[-1][0])
    distance = int(rng.integers(4,7))
    a,b = map(int,rng.integers(3,7,size=2))
    if a == b and force_unequal:
        b = 3+(b-2)%4
    if kind == 'diagonals':
        ports = [(a,-a), (b,b+1)]
        offsets = [(-1,2),(-1,-1)]
        incoming = [(-1,1),(-1,-1)]
        lasts,velocity = [(0,0),(0,1)],(-1,0)
        bottom = -distance
    elif kind == 'perpendicular':
        ports = [(0,-a),(b,0)]
        offsets = [(int(rng.integers(2)),2),(-1,int(rng.integers(2)))]
        incoming = [(0,1),(-1,0)]
        lasts,velocity = [(0,-1),(1,0)],(-1,1)
        bottom = -distance
    elif kind == 'staircase':
        ports = [(a,-a),(b,0)]
        offsets = [(-1,2),(-1,int(rng.integers(2)))]
        incoming = [(-1,1),(-1,0)]
        lasts,velocity = [(1,-1),(1,0)],(-2,1)
        bottom = 1-distance
    else:
        ports = [(a,0),(b,2*width-1)]
        offsets = [(-1,0),(-1,0)]
        incoming = [(-1,0),(-1,0)]
        lasts,velocity = ports,(-1,0)
        bottom = -distance
    routes = []
    for port,v,last in zip(ports,incoming,lasts):
        delta = max(abs(last[0]-port[0]),abs(last[1]-port[1]))
        before = [(port[0]+t*v[0],port[1]+t*v[1]) for t in range(delta)]
        routes.append(before+_to_row(last,velocity,bottom-1))
    pins = [path[-1] for path in routes]
    seam = pins[1][1] if kind != 'perpendicular' else pins[0][1]+1
    if kind == 'parallel':
        seam = width
    left = seam-width
    if groups == 3:
        third_col = left+3*width-1
        third_port = (bottom+2,third_col)
        ports.append(third_port)
        offsets.append((-1,0))
        incoming.append((-1,0))
        routes.append(_to_row(third_port,(-1,0),bottom-1))
        pins.append(routes[-1][-1])
    box = (bottom-depth-1,left-1,bottom,left+groups*width)
    gates = {path[-2] for path in routes}
    _require(gates <= _rim(box), "construction gates outside frame")
    fixed = {p:structure for p in _rim(box)-gates}
    specs = []
    for i,(port,offset,v) in enumerate(zip(ports,offsets,incoming)):
        core = (port[0]-offset[0],port[1]-offset[1])
        footprint = {(r,c) for r in range(core[0]-1,core[0]+3)
                     for c in range(core[1]-1,core[1]+3)}
        _require(not footprint & set(fixed), "construction fixed objects overlap")
        for p in footprint-{port}:
            fixed[p] = colors[i] if core[0] <= p[0] < core[0]+2 and core[1] <= p[1] < core[1]+2 else structure
        specs.append((core,v))
    occupied = set(fixed)
    for sid,path in enumerate(routes):
        for p in path[:-1]:
            _require(p not in occupied, "construction beam overlaps")
            occupied.add(p)
    all_points = set(fixed) | {p for path in routes for p in path}
    rmin = min(p[0] for p in all_points)
    rmax = max(p[0] for p in all_points)
    cmin = min(p[1] for p in all_points)
    cmax = max(p[1] for p in all_points)
    bh,bw = rmax-rmin+1,cmax-cmin+1
    _require(bh <= 28 and bw <= 28, "construction extent")
    h = int(rng.integers(max(20,bh+1),31))
    w = int(rng.integers(max(20,bw+1),31))
    offset = np.array([int(rng.integers(h-bh+1))-rmin,int(rng.integers(w-bw+1))-cmin])
    base = np.full((h,w),bg,dtype=int)
    for p,color in fixed.items():
        base[tuple(np.array(p)+offset)] = color
    output = base.copy()
    ray_mask = np.zeros((h,w),dtype=bool)
    for sid,path in enumerate(routes):
        for p in path[:-1]:
            q = tuple(np.array(p)+offset)
            output[q] = colors[sid]
            ray_mask[q] = True
    target_pieces = []
    for strip in range(groups):
        strip_cuts = cuts
        # An optional different third contour broadens shape matching while the
        # first two identical chains retain signal-dependent ambiguity.
        if strip == 2 and rng.random() < .5:
            proposal = _cuts(rng,width,layers)
            if int(proposal[-1][0]) == depth:
                strip_cuts = proposal
        for layer in range(layers):
            color = colors[strip] if layer == 0 else colors[groups+layer-1]
            cells = np.array([(bottom-1-u,left+strip*width+c)
                              for c in range(width)
                              for u in range(int(strip_cuts[layer][c]),int(strip_cuts[layer+1][c]))],dtype=int)+offset
            _require(np.all(output[cells[:,0],cells[:,1]] == bg), "target tile overlap")
            output[cells[:,0],cells[:,1]] = color
            target_pieces.append((cells,color))
    for sid,p in enumerate(pins):
        _require(output[tuple(np.array(p)+offset)] == colors[sid], "signal addresses wrong tile")
    cavity_reserved = np.zeros((h,w),dtype=bool)
    r0,c0,r1,c1 = (box[0]+offset[0],box[1]+offset[1],box[2]+offset[0],box[3]+offset[1])
    cavity_reserved[r0:r1+1,c0:c1+1] = True
    source = None
    for _ in range(30):
        inp = base.copy()
        forbidden = _dilate(base != bg) | ray_mask | cavity_reserved
        order = sorted(range(len(target_pieces)),key=lambda i:-len(target_pieces[i][0])+rng.random())
        for i in order:
            target,color = target_pieces[i]
            local = target-target.min(axis=0)
            ph,pw = local.max(axis=0)+1
            valid = np.ones((h-ph+1,w-pw+1),dtype=bool)
            for r,c in local:
                valid &= ~forbidden[r:r+h-ph+1,c:c+w-pw+1]
            choices = np.argwhere(valid)
            if not len(choices):
                break
            cells = local+choices[int(rng.integers(len(choices)))]
            inp[cells[:,0],cells[:,1]] = color
            piece_mask = np.zeros((h,w),dtype=bool)
            piece_mask[cells[:,0],cells[:,1]] = True
            forbidden |= _dilate(piece_mask)
        else:
            source = inp
            break
    _require(source is not None, "scatter clearance exhausted")
    source,output = np.rot90(source,rotation),np.rot90(output,rotation)
    if reflection:
        source,output = source[:,::-1],output[:,::-1]
    pair = {'input':source.copy(),'output':output.copy()}
    expected,info = _analyze_and_solve(pair['input'])
    _require(np.array_equal(expected,pair['output']), "construction and input-derived solution differ")
    _require(len(_assemblies(info,{})) == 2, "beam-free packing must have multiple colored answers")
    return pair


def _count(value,name,low,high):
    _require(isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_)), name+" must be an integer")
    value = int(value)
    _require(low <= value <= high, name+" outside legal range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ne = _count(num_train,'num_train',2,10),_count(num_test,'num_test',1,4)
    _require(nt > ne, "num_train must exceed num_test")
    _require(isinstance(seed,(int,np.integer)) and not isinstance(seed,(bool,np.bool_)), "integer seed required")
    rng = np.random.default_rng(int(seed))
    kinds = list(_KINDS)
    rng.shuffle(kinds)
    taught_kinds = kinds[:min(3,nt-1)]
    palette = list(map(int,rng.permutation(10)))
    stable_palette = rng.random() < .35
    shared_rotation = int(rng.integers(4))
    mixed_orientation = rng.random() < .6
    test_rotation = (shared_rotation+int(rng.choice((1,3)))) % 4
    # With only two demonstrations, teach shape depth and source count
    # separately. Tests recombine them or transfer the receiving axis.
    curriculum = [('parallel',3,2)] + [(k,2,3 if i == 0 else int(rng.integers(2,4)))
                                     for i,k in enumerate(taught_kinds)]
    while len(curriculum) < nt:
        groups,layers = int(rng.integers(2,4)),int(rng.integers(2,4))
        if mixed_orientation and groups == layers == 3:
            layers = 2  # Reserve this structural recombination for the tests.
        curriculum.append((str(rng.choice(['parallel']+taught_kinds)),groups,layers))
    rng.shuffle(curriculum)
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',nt),('test',ne)):
        for index in range(count):
            if split == 'train':
                kind,groups,layers = curriculum[index]
            else:
                groups,layers = (3,3) if mixed_orientation else (int(rng.integers(2,4)),int(rng.integers(2,4)))
                kind = str(rng.choice(taught_kinds))
            roles = palette if stable_palette else list(map(int,rng.permutation(10)))
            rotation = int(rng.integers(4)) if mixed_orientation else (
                       shared_rotation if split == 'train' else test_rotation)
            reflection = bool(rng.integers(2))
            for _ in range(180):
                try:
                    pair = _make_pair(rng,kind,groups,layers,roles,rotation,reflection,
                                      split == 'train' and kind == taught_kinds[0])
                except ValueError:
                    continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("could not construct a distinct uniquely addressed receiver")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "split lists required")
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "puzzle counts")
    seen,taught,needed = set(),set(),set()
    train_axes,train_sizes,test_features = set(),[],[]
    teaching = {'parallel':False,'three_sources':False,'deep':False,'A':False,'B':False,'unequal':False}
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'}, "pair keys")
            inp,out = _grid(pair['input']),_grid(pair['output'])
            _require(inp.shape == out.shape and _key(inp) not in seen, "shape or duplicate input")
            seen.add(_key(inp))
            expected,info = _analyze_and_solve(inp)
            _require(np.array_equal(expected,out), "incorrect label")
            _require(not np.array_equal(inp,out), "unchanged example")
            # Receiver and launcher foreground is fixed; movable objects are
            # matched individually, even when equal-color pieces meet in output.
            fixed = info['base'] != info['bg']
            _require(np.array_equal(inp[fixed],out[fixed]), "fixed structure changed")
            _require(len(info['placements']) == len(info['pieces']), "missing piece")
            for sid,target in info['placements']:
                source,color = info['pieces'][sid]
                shift = target.min(axis=0)-source.min(axis=0)
                _require(np.any(shift) and set(map(tuple,source+shift)) == set(map(tuple,target)), "tile mask altered")
                _require(np.all(out[target[:,0],target[:,1]] == color), "tile recolored")
            r0,c0,r1,c1 = info['box']
            _require(np.all(out[r0+1:r1,c0+1:c1] != info['bg']), "cavity hole")
            _require(all(out[p] == color for p,color in info['anchors'].items()), "signal anchor mismatch")
            for color in range(10):
                if color not in (info['bg'],info['structure']):
                    additions = sum(info['launchers'][sid]['color'] == color for sid in info['rays'].values())
                    _require(np.count_nonzero(out == color) == np.count_nonzero(inp == color)+additions, "payload mass accounting")
            # Counterfactual ablation of B: identical shapes with different
            # colors can exchange destinations when signal anchors are removed.
            _require(len(_assemblies(info,{})) == 2, "B contribution redundant")
            tags = {e['kind'] for e in info['events']}
            receiving_axis = 0 if len({p[0] for p in info['gates']}) == 1 else 1
            if split == 'train':
                train_axes.add(receiving_axis)
                train_sizes.append(len(info['pieces']))
                taught |= tags
                teaching['parallel'] |= not tags
                teaching['three_sources'] |= len(info['launchers']) == 3
                teaching['deep'] |= len(info['pieces']) >= 6 and len(info['launchers']) == 2
                teaching['B'] = True
                teaching['unequal'] |= any(e['cuts'][0] != e['cuts'][1] for e in info['events'])
                teaching['A'] |= any(len(cells) < np.prod(cells.max(axis=0)-cells.min(axis=0)+1)
                                     for cells,_ in info['pieces'])
            else:
                needed |= tags
                test_features.append((receiving_axis,len(info['pieces'])))
    _require(needed <= taught and bool(taught), "test collision kind must be demonstrated")
    _require(all(teaching.values()), "training lacks required mechanism evidence")
    _require(all(axis not in train_axes or size > max(train_sizes)
                 for axis,size in test_features), "test lacks receiving-axis or piece-count shift")


def _equal(p,q):
    return all(len(p[s]) == len(q[s]) and all(np.array_equal(a[k],b[k])
               for a,b in zip(p[s],q[s]) for k in ('input','output')) for s in ('train','test'))


def self_test():
    """Bounded 42-seed structural and semantic checks; returns measured diversity."""
    shapes,structures,puzzles,roles,modes,piece_counts,depths = set(),set(),set(),set(),set(),set(),set()
    # Source-supported contact and phase fixtures, expressed as paths rather
    # than copied source puzzle pairs. They are explicit expected values.
    dummy_box = (0,0,1,1)
    hit = _hit([_walk((15,5),(-1,1),(30,30))],
               [_walk((13,14),(-1,-1),(30,30))],dummy_box)
    _require(hit[:2] == (5,3) and hit[4] == 'between', "unequal between-cell fixture")
    hit = _hit([_walk((10,3),(0,1),(30,30))],
               [_walk((17,12),(-1,0),(30,30))],dummy_box)
    _require(hit[:2] == (8,6) and hit[4] == 'cell', "perpendicular contact fixture")
    _require(_walk((6,17),(-2,1),(24,24)) ==
             [(6,17),(5,17),(4,18),(3,18),(2,19),(1,19),(0,20)], "major-axis-first staircase fixture")
    # A non-layer fixture: the small L caps do not span the strip width. It
    # exercises the general placement search, outside the sampler's profiles.
    hook = np.array([(0,0),(1,0),(2,0),(2,1),(2,2),(1,2)])
    cap = np.array([(0,1),(0,2),(1,1)])
    packing = {'box':(0,0,4,7),'bg':0,
               'pieces':[(hook,2),(hook+[0,8],3),(cap+[10,0],5),(cap+[10,8],5)]}
    _require(len(_assemblies(packing,{})) == 2, "unaddressed fixture must be ambiguous")
    addressed = _assemblies(packing,{(3,3):2,(3,4):3})
    _require(len(addressed) == 1 and np.array_equal(addressed[0][0],
              np.array([[2,5,5,3,5,5],[2,5,2,3,5,3],[2,2,2,3,3,3]])), "non-layer exact-cover fixture")
    flat = np.array([[5,5,5,5,5,5],[2,2,2,3,3,3],[2,2,2,3,3,3]])
    _require(Counter(flat.flat) == Counter(addressed[0][0].flat) and
             flat[2,2] == 2 and flat[2,3] == 3 and not np.array_equal(flat,addressed[0][0]),
             "A ablation must preserve signal colors and mass while altering the mask-defined answer")
    unequal,between,sign_failures,straight_failures,phase_failures = 0,0,0,0,0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(tuple(_key(p['input']) for p in puzzle['train']+puzzle['test']))
        for pair in puzzle['train']+puzzle['test']:
            a = pair['input']
            _,info = _analyze_and_solve(a)
            shapes.add(a.shape)
            roles.add((info['bg'],info['structure']))
            piece_counts.add(len(info['pieces']))
            box = info['box']
            depths.add(tuple(sorted((box[2]-box[0]-1,box[3]-box[1]-1))))
            palette = {}
            structures.add((a.shape,tuple(palette.setdefault(int(x),len(palette)) for x in a.flat)))
            for event in info['events']:
                modes.add(event['kind'])
                unequal += event['cuts'][0] != event['cuts'][1]
                between += event['intersection'] == 'between'
                for variant in ('straight','sign','ceil'):
                    try:
                        rays,pins,_ = _trace(info,a.shape,variant)
                        differs = rays != info['rays'] or pins != info['anchors']
                    except ValueError:
                        differs = True
                    if variant == 'straight':
                        straight_failures += differs
                    elif variant == 'sign' and event['kind'] == 'staircase':
                        sign_failures += differs
                    elif variant == 'ceil' and event['kind'] == 'staircase':
                        phase_failures += differs
    _require(len(puzzles) == 42 and len(structures) >= 200, "structural diversity collapsed")
    _require(len(shapes) >= 30 and len(roles) >= 20 and len(depths) >= 10, "geometry or role diversity collapsed")
    _require(modes == set(_KINDS) and piece_counts >= {4,6,9}, "mechanism or object-count coverage")
    _require(unequal >= 30 and between >= 20 and straight_failures >= 30 and sign_failures >= 10
             and phase_failures >= 10, "collision counterexamples missing")
    first = generate_puzzle(0)
    _require(_equal(first,generate_puzzle(0)), "seed nondeterminism")
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(100+nt*7+ne,nt,ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, "explicit counts ignored")
    for counts in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(4,None)):
        try:
            generate_puzzle(0,*counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = {s:[{k:v.copy() for k,v in p.items()} for p in first[s]] for s in first}
    corrupt['test'][0]['output'] = corrupt['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {'seeds':42,'pairs':210,'shapes':len(shapes),'normalized_structures':len(structures),
            'roles':len(roles),'cavity_shapes':len(depths),'piece_counts':sorted(piece_counts),
            'collision_kinds':sorted(modes),'unequal_encounters':unequal,'between_cell_encounters':between,
            'straight_ray_counterexamples':straight_failures,'sign_sum_counterexamples':sign_failures,
            'reversed_phase_counterexamples':phase_failures}
