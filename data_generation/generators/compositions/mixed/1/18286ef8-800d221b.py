"""Addressed cargo changes a terminal vote before port-weighted collection.

No parent generator, files, seed, or construction metadata is used by solve.
All iteration over geometric sets is sorted where it influences random sampling.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools

import numpy as np

DIRECTIONS = tuple(p for p in itertools.product(range(3), repeat=2) if p != (1, 1))
ORTHOGONAL = ((-1, 0), (0, -1), (0, 1), (1, 0))
HOST_SEEDS = (0,360452526,272385342,870230779,96720759,748581989,125006081,765836841,933264598,478446094,906835804,460958008,925629181,983656566,509565787,271489653,900145869,79365871,942033439,165088900,942411286,72175879,149223929,548916152,846864215,697936116,153577070,185869247,90360941,332575419,292130590,657672886,238355150,154399678,972702788,868929938,498998470,978328764,359859599,895563275,72094356,795979757)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid shape')
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), 'invalid cells')
    return a.astype(np.int8, copy=True)


def _rect(r, c, h, w):
    return {(x, y) for x in range(r, r+h) for y in range(c, c+w)}


def _neighbors(cells, shape):
    h, w = shape
    return {(r+dr, c+dc) for r, c in cells for dr, dc in ORTHOGONAL
            if 0 <= r+dr < h and 0 <= c+dc < w}


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ORTHOGONAL:
                q = r+dr, c+dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _points(mask):
    return set(map(tuple, np.argwhere(mask).tolist()))


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs)-min(rs)+1, max(cs)-min(cs)+1


def _macro(p, rows, cols):
    _require(p[0] not in rows and p[1] not in cols, 'point on a divider')
    return sum(p[0] > r for r in rows), sum(p[1] > c for c in cols)


def _ports(hub):
    r, c = hub
    return [(r-1,c+j) for j in range(3)] + [(r+3,c+j) for j in range(3)] + \
           [(r+j,c-1) for j in range(3)] + [(r+j,c+3) for j in range(3)]


def _majority(a, signals):
    counts = [int(np.count_nonzero(a == s)) for s in signals]
    _require(counts[0] != counts[1], 'tied local or port vote')
    return signals[int(counts[1] > counts[0])]


def _parse_frames(a, color, bg, signals, rows, cols):
    """Discover frame rings and their one optional complete interior crossbar."""
    frames = []
    for cells in _components(_points(a == color)):
        r, c, h, w = _box(cells)
        _require(h >= 3 and w >= 3, 'small frame')
        whole = _rect(r,c,h,w)
        interior = _rect(r+1,c+1,h-2,w-2)
        ring = whole-interior
        _require(ring <= cells, 'broken frame')
        macro = _macro((r,c),rows,cols)
        _require(macro != (1,1) and all(_macro(p,rows,cols)==macro for p in whole), 'frame crosses compartment')
        bars = cells-ring
        if not bars:
            zones = [interior]
        else:
            br, bc, bh, bw = _box(bars)
            horizontal = bh == 1 and bc == c+1 and bw == w-2 and r+1 < br < r+h-2
            vertical = bw == 1 and br == r+1 and bh == h-2 and c+1 < bc < c+w-2
            _require(horizontal or vertical, 'invalid crossbar')
            _require(bars == _rect(br,bc,bh,bw), 'nonrectangular crossbar')
            zones = ([_rect(r+1,c+1,br-r-1,w-2), _rect(br+1,c+1,r+h-br-2,w-2)]
                     if horizontal else
                     [_rect(r+1,c+1,h-2,bc-c-1), _rect(r+1,bc+1,h-2,c+w-bc-2)])
        patterns = [a[tuple(zip(*sorted(z)))] for z in zones]
        if len(zones) == 1:
            _require(set(map(int,patterns[0])) == set(signals), 'source needs both signals')
            frames.append(dict(cells=cells, whole=whole, macro=macro, source=True,
                               cargo=zones[0], box=(r,c,h,w)))
        else:
            empty = [i for i,p in enumerate(patterns) if np.all(p == bg)]
            _require(len(empty)==1, 'receiver needs one empty socket')
            socket, resident = zones[empty[0]], zones[1-empty[0]]
            _require(set(map(int,patterns[1-empty[0]])) == set(signals), 'resident needs two signals')
            _majority(patterns[1-empty[0]],signals)
            frames.append(dict(cells=cells, whole=whole, macro=macro, source=False,
                               socket=socket, resident=resident, box=(r,c,h,w)))
    _require(4 <= len(frames) <= 8 and len({f['macro'] for f in frames})==len(frames), 'frame count or ownership')
    _require(sum(f['source'] for f in frames)==1, 'unique source required')
    return frames


def _analyze(input_grid):
    a = _grid(input_grid)
    h,w = a.shape
    bg = int(a[0,0])
    _require(all(int(a[p]) == bg for p in ((0,0),(0,w-1),(h-1,0),(h-1,w-1))), 'background corners')
    values, counts = np.unique(a,return_counts=True)
    _require(len(values)==7, 'expected seven visible color roles')
    singleton = [int(v) for v,n in zip(values,counts) if n==1]
    _require(len(singleton)==1, 'unique singleton instruction required')
    marker_color = singleton[0]
    marker = tuple(map(int,np.argwhere(a == marker_color)[0]))
    _require(0<marker[0]<h-1 and 0<marker[1]<w-1, 'instruction must be inside canvas border')
    dividers = []
    for color in map(int,values):
        if color in (bg,marker_color):
            continue
        mask = a == color
        rr = tuple(map(int,np.flatnonzero(mask.sum(axis=1) > w/2)))
        cc = tuple(map(int,np.flatnonzero(mask.sum(axis=0) > h/2)))
        if len(rr) != 2 or len(cc) != 2:
            continue
        support = np.zeros_like(mask)
        support[list(rr),:] = True
        support[:,list(cc)] = True
        if not np.all(~mask | support):
            continue
        if min(rr+cc) <= 0 or rr[-1]>=h-1 or cc[-1]>=w-1:
            continue
        if all(int(a[r,c])==color for r in rr for c in cc):
            dividers.append((color,rr,cc))
    _require(len(dividers)==1, 'unique macro partition required')
    divider, rows, cols = dividers[0]
    hubs = []
    for r in range(rows[0]+1,rows[1]-2):
        for c in range(cols[0]+1,cols[1]-2):
            color = int(a[r,c])
            if color not in (bg,divider,marker_color) and np.all(a[r:r+3,c:c+3]==color):
                hubs.append((color,(r,c)))
    _require(len(hubs)==1, 'unique central scaffold collector required')
    scaffold, hub = hubs[0]
    role_candidates = [int(v) for v in values if int(v) not in (bg,divider,marker_color,scaffold)]
    interpretations = []
    for frame_color in role_candidates:
        signals = tuple(v for v in role_candidates if v != frame_color)
        try:
            frames = _parse_frames(a,frame_color,bg,signals,rows,cols)
            interpretations.append((frame_color,signals,frames))
        except ValueError:
            continue
    _require(len(interpretations)==1, 'unique terminal interpretation required')
    frame_color, signals, frames = interpretations[0]
    _require(all(marker not in f['whole'] for f in frames), 'instruction inside terminal')
    source = next(f for f in frames if f['source'])
    receivers = [f for f in frames if not f['source']]
    address = _macro(marker,rows,cols)
    destinations = [i for i,f in enumerate(receivers) if f['macro']==address]
    _require(len(destinations)==1, 'instruction must address a receiving terminal')
    target = destinations[0]
    cargo_box = _box(source['cargo'])
    for f in receivers:
        _require(_box(f['socket'])[2:] == cargo_box[2:], 'socket must fit whole payload')
    covered_signals = source['cargo'] | set().union(*(f['resident'] for f in receivers))
    _require(_points(np.isin(a,signals)) == covered_signals, 'signal outside terminal')
    hubcells = _rect(*hub,3,3)
    routes = _components(_points(a == scaffold)-hubcells)
    _require(bool(routes), 'no routes')
    owners, weights = [], [0]*len(receivers)
    portset = set(_ports(hub))
    for route in routes:
        nearby = _neighbors(route,a.shape)
        _require(not (nearby & source['cells']), 'source tray must not own routes')
        touched = [i for i,f in enumerate(receivers) if nearby & f['cells']]
        _require(len(touched)==1, 'route has ambiguous terminal ownership')
        _require(bool(route & portset), 'route has no collector port')
        owners.append(touched[0])
        weights[touched[0]] += len(route & portset)
    _require(all(weights) and 3 <= sum(weights) <= 8, 'invalid terminal port weights')
    _require(all(int(a[p]) in (bg,scaffold) for p in portset), 'obstructed port')
    # A route crosses a divider at a straight gate. No other divider damage.
    boundary = {(r,c) for r in range(h) for c in range(w) if r in rows or c in cols}
    for p in boundary:
        _require(int(a[p]) in (divider,scaffold), 'corrupt divider')
        if int(a[p])==scaffold:
            r,c = p
            _require((r in rows) != (c in cols), 'gate at divider intersection')
            ends = ((r-1,c),(r+1,c)) if r in rows else ((r,c-1),(r,c+1))
            _require(all(0<=q[0]<h and 0<=q[1]<w and int(a[q])==scaffold for q in ends), 'bent or incomplete gate')
    cargo_r,cargo_c,ch,cw = cargo_box
    moved = a.copy()
    payload = a[cargo_r:cargo_r+ch,cargo_c:cargo_c+cw].copy()
    for p in source['cargo']:
        moved[p] = bg
    tr,tc,_,_ = _box(receivers[target]['socket'])
    moved[tr:tr+ch,tc:tc+cw] = payload
    before = [_majority(a[tuple(zip(*sorted(f['resident'])))],signals) for f in receivers]
    labels = []
    for f in receivers:
        votes = f['resident'] | f['socket']
        labels.append(_majority(moved[tuple(zip(*sorted(votes)))],signals))
    port_labels = [s for s,n in zip(labels,weights) for _ in range(n)]
    winner = _majority(np.asarray(port_labels),signals)
    # Pre-move global ties are allowed by the semantic rule. None occur in the
    # first counterfactual witness; retain None elsewhere rather than inventing a tie-break.
    pre_ports = [s for s,n in zip(before,weights) for _ in range(n)]
    pre_winner = None if pre_ports.count(signals[0])==pre_ports.count(signals[1]) else _majority(np.asarray(pre_ports),signals)
    return dict(a=a, bg=bg, divider=divider, rows=rows, cols=cols, scaffold=scaffold,
                hub=hub, frame_color=frame_color, signals=signals, marker=marker,
                address=address, frames=frames, source=source, receivers=receivers,
                target=target, routes=routes, owners=owners, weights=weights,
                moved=moved, before=before, labels=labels, winner=winner,
                pre_winner=pre_winner, payload_majority=_majority(payload,signals))


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    out = info['moved'].copy()
    for route,owner in zip(info['routes'],info['owners']):
        for p in route:
            out[p] = info['labels'][owner]
    r,c = info['hub']
    out[r+1,c+1] = info['winner']
    return out


def _path(start, targets, allowed, rows, cols, rng, roughness):
    costs = 1 + rng.random(allowed.shape)*roughness
    distances, previous = {start:0.0}, {}
    heap = [(0.0,start)]
    h,w = allowed.shape
    while heap:
        distance,p = heapq.heappop(heap)
        if distance != distances[p]:
            continue
        if p in targets:
            path = {p}
            while p != start:
                p = previous[p]
                path.add(p)
            return path
        r,c = p
        for dr,dc in ORTHOGONAL:
            q = r+dr,c+dc
            if not (0<=q[0]<h and 0<=q[1]<w and allowed[q]):
                continue
            if (r in rows or q[0] in rows) and dr==0:
                continue
            if (c in cols or q[1] in cols) and dc==0:
                continue
            nd = distance+costs[q]
            if nd < distances.get(q,float('inf')):
                distances[q], previous[q] = nd,p
                heapq.heappush(heap,(nd,q))
    return None


def _pattern(rng, h, w, majority, minority, nminor):
    style = int(rng.integers(4))
    rr,cc = np.indices((h,w))
    if style==0:
        order = rng.permutation(h*w)
    elif style==1:
        r,c = int(rng.integers(h)),int(rng.integers(w))
        order = np.argsort((abs(rr-r)+abs(cc-c)+rng.random((h,w))*.3).ravel())
    elif style==2:
        order = np.argsort(((rr+cc)%2*h*w+rr*w+cc).ravel())
    else:
        order = np.argsort((cc*h+rr if rng.random()<.5 else rr*w+cc).ravel())
    a = np.full(h*w,majority,dtype=np.int8)
    a[order[:nminor]] = minority
    return a.reshape(h,w)


def _axis(rng):
    outer0,outer1 = map(int,rng.integers(7,11,size=2))
    middle = int(rng.integers(5,min(8,28-outer0-outer1)+1))
    return (outer0,middle,outer1)


def _region(direction, sizes_r, sizes_c):
    mr,mc = direction
    return sum(sizes_r[:mr])+mr, sum(sizes_c[:mc])+mc, sizes_r[mr], sizes_c[mc]


def _receiver(rng, region, ch, cw):
    r,c,h,w = region
    choices = []
    for side in range(4):
        for depth in (1,2):
            bh,bw = (ch+depth+3,cw+2) if side<2 else (ch+2,cw+depth+3)
            area = depth*(cw if side<2 else ch)
            if bh <= h-2 and bw <= w-2 and area>=3:
                choices.append((side,depth,bh,bw))
    if not choices:
        return None
    side,depth,bh,bw = choices[int(rng.integers(len(choices)))]
    pr = r+int(rng.integers(1,h-bh))
    pc = c+int(rng.integers(1,w-bw))
    whole = _rect(pr,pc,bh,bw)
    inner = _rect(pr+1,pc+1,bh-2,bw-2)
    frame = whole-inner
    if side<2:
        bar = pr+1+(depth if side==0 else ch)
        frame |= _rect(bar,pc+1,1,cw)
        resident = _rect(pr+1 if side==0 else bar+1,pc+1,depth,cw)
        socket = _rect(bar+1 if side==0 else pr+1,pc+1,ch,cw)
    else:
        bar = pc+1+(depth if side==2 else cw)
        frame |= _rect(pr+1,bar,ch,1)
        resident = _rect(pr+1,pc+1 if side==2 else bar+1,ch,depth)
        socket = _rect(pr+1,bar+1 if side==2 else pc+1,ch,cw)
    return dict(whole=whole,cells=frame,resident=resident,socket=socket)


def _construct(rng, roles, address, witness, payload_index, richer):
    bg,divider,scaffold,frame_color,marker_color,s0,s1 = roles
    signals = (s0,s1)
    sizes_r,sizes_c = _axis(rng),_axis(rng)
    h,w = sum(sizes_r)+2,sum(sizes_c)+2
    rows = (sizes_r[0],sizes_r[0]+sizes_r[1]+1)
    cols = (sizes_c[0],sizes_c[0]+sizes_c[1]+1)
    a = np.full((h,w),bg,dtype=np.int8)
    a[list(rows),:] = divider
    a[:,list(cols)] = divider
    center_region = _region((1,1),sizes_r,sizes_c)
    cr,cc,rh,rw = center_region
    hub = cr+int(rng.integers(1,rh-3)),cc+int(rng.integers(1,rw-3))
    a[hub[0]:hub[0]+3,hub[1]:hub[1]+3] = scaffold
    ch,cw = (3,3) if witness==2 else ((2,3),(3,2),(3,3))[int(rng.integers(3))]
    source_options = [d for d in DIRECTIONS if d!=address
                      and sizes_r[d[0]]>=ch+4 and sizes_c[d[1]]>=cw+4]
    source_direction = source_options[int(rng.integers(len(source_options)))]
    sr,sc,sh,sw = _region(source_direction,sizes_r,sizes_c)
    sr += int(rng.integers(1,sh-ch-2))
    sc += int(rng.integers(1,sw-cw-2))
    source_whole = _rect(sr,sc,ch+2,cw+2)
    cargo = _rect(sr+1,sc+1,ch,cw)
    for p in source_whole-cargo:
        a[p] = frame_color
    payload_label,other = signals[payload_index],signals[1-payload_index]
    nminor = (3 if witness==2 else int(rng.integers(1,(ch*cw-(3 if witness else 1))//2+1)))
    payload = _pattern(rng,ch,cw,payload_label,other,nminor)
    a[sr+1:sr+ch+1,sc+1:sc+cw+1] = payload
    cargo_margin = ch*cw-2*nminor
    panel_count = 3 if witness else int(rng.integers(3,7 if richer else 5))
    pool = [d for d in DIRECTIONS if d not in (source_direction,address)]
    rng.shuffle(pool)
    directions = [address]+pool[:panel_count-1]
    panels = []
    for direction in directions:
        panel = _receiver(rng,_region(direction,sizes_r,sizes_c),ch,cw)
        if panel is None:
            return None
        panel['macro'] = direction
        panel['angle'] = float(np.arctan2(np.mean([p[0] for p in panel['whole']])-hub[0]-1,
                                          np.mean([p[1] for p in panel['whole']])-hub[1]-1))
        panels.append(panel)
    panels.sort(key=lambda p:p['angle'])
    target = next(i for i,p in enumerate(panels) if p['macro']==address)
    if witness==1:
        weights = [1]*panel_count
        weights[target] = 3
        labels = [other]*panel_count
        labels[target] = payload_label
    elif witness==2:
        weights = [2]*panel_count
        weights[target] = 1
        labels = [other]*panel_count
        labels[next(i for i in range(panel_count) if i!=target)] = payload_label
    else:
        total = int(rng.integers(max(3,panel_count),9))
        weights = [1]*panel_count
        for _ in range(total-panel_count):
            options = [i for i,n in enumerate(weights) if n<3]
            weights[int(rng.choice(options))] += 1
        labels = [signals[int(rng.integers(2))] for _ in panels]
    for i,panel in enumerate(panels):
        resident = panel['resident']
        r,c,ph,pw = _box(resident)
        area = ph*pw
        majority = (other if witness else signals[int(rng.integers(2))]) if i==target else labels[i]
        minority = next(s for s in signals if s!=majority)
        choices = [n for n in range(1,(area-1)//2+1)
                   if i!=target or not witness or
                   (area-2*n < cargo_margin if witness==1 else area-2*n > cargo_margin)]
        if not choices:
            return None
        nm = choices[0] if witness==1 and i!=target else int(rng.choice(choices))
        a[r:r+ph,c:c+pw] = _pattern(rng,ph,pw,majority,minority,nm)
        if i==target:
            combined = np.concatenate((a[r:r+ph,c:c+pw].ravel(),payload.ravel()))
            if np.count_nonzero(combined==s0)==np.count_nonzero(combined==s1):
                return None
            labels[i] = _majority(combined,signals)
        for p in panel['cells']:
            a[p] = frame_color
    counts = [sum(n for n,s in zip(weights,labels) if s==color) for color in signals]
    if counts[0]==counts[1] or min(counts)==0:
        return None
    # Circular panel order is matched to circular port groups; branches may share
    # their own network but unrelated networks have an orthogonal buffer.
    ring = [(hub[0]-1,hub[1]),(hub[0]-1,hub[1]+2),(hub[0],hub[1]+3),(hub[0]+2,hub[1]+3),
            (hub[0]+3,hub[1]+2),(hub[0]+3,hub[1]),(hub[0]+2,hub[1]-1),(hub[0],hub[1]-1)]
    selected = [ring[int(j)] for j in rng.choice(8,size=sum(weights),replace=False)]
    selected.sort(key=lambda p:float(np.arctan2(p[0]-hub[0]-1,p[1]-hub[1]-1)))
    candidates = []
    for shift in range(len(selected)):
        sequence = selected[shift:]+selected[:shift]
        offset,cost,groups = 0,0.0,[]
        for panel,n in zip(panels,weights):
            group = set(sequence[offset:offset+n])
            offset += n
            for p in group:
                delta = abs(panel['angle']-float(np.arctan2(p[0]-hub[0]-1,p[1]-hub[1]-1)))
                cost += min(delta,2*np.pi-delta)**2
            groups.append(group)
        candidates.append((cost+float(rng.random()),groups))
    groups = min(candidates,key=lambda x:x[0])[1]
    hubcells = _rect(*hub,3,3)
    collar = _rect(hub[0]-1,hub[1]-1,5,5)-hubcells
    allpanels = source_whole | set().union(*(p['whole'] for p in panels))
    networks = [set(g) for g in groups]
    intersections = set(itertools.product(rows,cols))
    for owner in sorted(range(panel_count),key=lambda i:-weights[i]):
        foreign_panels = allpanels-panels[owner]['whole']
        foreign_networks = set().union(*(n for i,n in enumerate(networks) if i!=owner))
        blocked = allpanels|hubcells|collar|foreign_networks|_neighbors(foreign_panels|foreign_networks,a.shape)|intersections
        allowed = np.ones(a.shape,dtype=bool)
        allowed[0,:] = allowed[-1,:] = False
        allowed[:,0] = allowed[:,-1] = False
        for p in blocked:
            allowed[p] = False
        for p in groups[owner]:
            allowed[p] = True
        border = _neighbors(panels[owner]['whole'],a.shape)-panels[owner]['whole']
        border = {p for p in border if allowed[p] and p[0] not in rows and p[1] not in cols}
        if not border:
            return None
        net = set()
        for port in sorted(groups[owner]):
            path = _path(port,border|net,allowed,rows,cols,rng,8 if richer else float(rng.choice([.2,2,5])))
            if path is None:
                return None
            net |= path
        if net & collar != groups[owner]:
            return None
        networks[owner] = net
    for net in networks:
        for p in net:
            a[p] = scaffold
    r,c,dh,dw = _region(address,sizes_r,sizes_c)
    marker_options = sorted(p for p in _rect(r,c,dh,dw) if int(a[p])==bg and p not in allpanels
                            and 0<p[0]<h-1 and 0<p[1]<w-1)
    if not marker_options:
        return None
    a[marker_options[int(rng.integers(len(marker_options)))]] = marker_color
    # Construct expected output from records before invoking the input parser.
    out = a.copy()
    out[sr+1:sr+ch+1,sc+1:sc+cw+1] = bg
    tr,tc,_,_ = _box(panels[target]['socket'])
    out[tr:tr+ch,tc:tc+cw] = payload
    for net,label in zip(networks,labels):
        for p in net:
            out[p] = label
    port_values = np.asarray([s for s,n in zip(labels,weights) for _ in range(n)])
    out[hub[0]+1,hub[1]+1] = _majority(port_values,signals)
    try:
        info = _analyze(a)
        if not np.array_equal(solve(a),out):
            return None
        if witness==1 and info['pre_winner']==info['winner']:
            return None
        if witness==1:
            loser = next(s for s in info['signals'] if s!=info['winner'])
            if np.count_nonzero(a==info['winner']) >= np.count_nonzero(a==loser):
                return None
    except ValueError:
        return None
    return dict(input=a,output=out)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value,lo,hi,name in ((num_train,2,10,'num_train'),(num_test,1,4,'num_test')):
        _require(_integer(value) and lo<=int(value)<=hi, name+' out of range')
    _require(num_train>num_test,'num_train must exceed num_test')
    _require(_integer(seed) and int(seed)>=0,'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base_roles = list(map(int,rng.choice(10,size=7,replace=False)))
    varying_palette = bool(rng.integers(2))
    held = DIRECTIONS[int(rng.integers(8))]
    corner = [d for d in DIRECTIONS if d[0]!=1 and d[1]!=1 and d!=held]
    cardinal = [d for d in DIRECTIONS if (d[0]==1 or d[1]==1) and d!=held]
    schedule = [corner[int(rng.integers(len(corner)))],cardinal[int(rng.integers(len(cardinal)))]]
    while len(schedule)<num_train:
        pool = [d for d in DIRECTIONS if d!=held]
        schedule.append(pool[int(rng.integers(len(pool)))])
    # The two training witnesses are shuffled; neither is tied to an ordinal.
    witnesses = [1,2]+[0]*(num_train-2)
    order = list(map(int,rng.permutation(num_train)))
    train_plan = [(schedule[i],witnesses[i]) for i in order]
    test_plan = [(held,0)] + [(DIRECTIONS[int(rng.integers(8))],0) for _ in range(num_test-1)]
    payload_index = int(rng.integers(2))
    result,seen = {'train':[],'test':[]},set()
    for split,plan in (('train',train_plan),('test',test_plan)):
        for address,witness in plan:
            roles = list(map(int,rng.choice(10,size=7,replace=False))) if varying_palette else base_roles
            for attempt in range(400):
                pair = _construct(rng,roles,address,witness,payload_index if witness else int(rng.integers(2)),split=='test')
                if pair is not None:
                    key = pair['input'].shape,pair['input'].tobytes()
                    if key not in seen:
                        seen.add(key)
                        result[split].append(pair)
                        break
            else:
                raise RuntimeError('bounded layout construction exhausted')
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle)=={'train','test'},'puzzle keys')
    _require(all(isinstance(puzzle[s],list) for s in puzzle),'split list')
    n,m = len(puzzle['train']),len(puzzle['test'])
    _require(2<=n<=10 and 1<=m<=4 and n>m,'example counts')
    seen,infos = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair)=={'input','output'},'pair keys')
            a,b = _grid(pair['input']),_grid(pair['output'])
            key = a.shape,a.tobytes()
            _require(key not in seen,'duplicate input')
            seen.add(key)
            info = _analyze(a)
            _require(a.shape==b.shape and np.array_equal(solve(a),b),'incorrect output')
            target = info['target']
            _require(all(x==y for i,(x,y) in enumerate(zip(info['before'],info['labels'])) if i!=target),'changed unrelated majority')
            cargo = info['source']['cargo']
            _require(all(int(b[p])==info['bg'] for p in cargo),'source not cleared')
            source_r,source_c,ph,pw = _box(cargo)
            socket_r,socket_c,_,_ = _box(info['receivers'][target]['socket'])
            _require(np.array_equal(a[source_r:source_r+ph,source_c:source_c+pw],
                                    b[socket_r:socket_r+ph,socket_c:socket_c+pw]), 'payload changed during translation')
            _require(all(int(b[p])==int(a[p]) for f in info['frames'] for p in f['cells']),'frame changed')
            _require(int(b[info['marker']])==int(a[info['marker']]),'instruction changed')
            for f in info['receivers']:
                _require(all(int(b[p])==int(a[p]) for p in f['resident']),'resident changed')
            infos[split].append(info)
    train = infos['train']
    _require(any(i['before'][i['target']] != i['labels'][i['target']] for i in train),'missing local flip witness')
    _require(any(i['labels'][i['target']] != i['payload_majority'] for i in train),'missing resident-survives witness')
    _require(any(i['winner'] != i['pre_winner'] and i['pre_winner'] is not None for i in train),'missing causal collector witness')
    _require(any(2*i['labels'].count(i['winner'])<len(i['labels']) for i in train),'missing panel-count counterexample')
    _require(any(np.count_nonzero(i['a']==i['winner']) <
                 np.count_nonzero(i['a']==next(s for s in i['signals'] if s!=i['winner']))
                 for i in train), 'missing raw-cell counterexample')
    _require(any(i['winner'] != i['payload_majority'] for i in train),'missing non-payload collector witness')
    _require(any(i['address'][0]!=1 and i['address'][1]!=1 for i in train),'missing diagonal address')
    _require(any(i['address'][0]==1 or i['address'][1]==1 for i in train),'missing cardinal address')
    _require(infos['test'][0]['address'] not in {i['address'] for i in train},'missing held-address shift')
    return None


def self_test():
    shapes,addresses,source_addresses,payload_shapes,profiles,palettes,signatures = (set() for _ in range(7))
    flip_count = disagree_count = forks = gates = pairs = raw_disagrees = panel_disagrees = route_disagrees = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                a = pair['input']
                info = _analyze(a)
                digest.update(bytes(a.shape)+a.tobytes()+pair['output'].tobytes())
                shapes.add(a.shape)
                addresses.add(info['address'])
                source_addresses.add(info['source']['macro'])
                payload_shapes.add(_box(info['source']['cargo'])[2:])
                profiles.add(tuple(sorted(info['weights'])))
                palettes.add((info['bg'],info['divider'],info['scaffold'],info['frame_color'],int(a[info['marker']]),*info['signals']))
                flip_count += info['pre_winner'] is not None and info['winner']!=info['pre_winner']
                disagree_count += info['winner']!=info['payload_majority']
                loser = next(s for s in info['signals'] if s!=info['winner'])
                raw_disagrees += np.count_nonzero(a==info['winner']) < np.count_nonzero(a==loser)
                panel_disagrees += info['labels'].count(info['winner']) < info['labels'].count(loser)
                route_disagrees += sum(len(route)*(1 if info['labels'][owner]==info['winner'] else -1)
                                       for route,owner in zip(info['routes'],info['owners'])) < 0
                routecells = set().union(*info['routes'])
                forks += any(sum((r+dr,c+dc) in routecells for dr,dc in ORTHOGONAL)>=3 for r,c in routecells)
                gates += sum(p[0] in info['rows'] or p[1] in info['cols'] for p in routecells)
                pairs += 1
                # Shifting the marker within its own compartment leaves the vote
                # unchanged and simply preserves the newly placed instruction.
                r,c = info['marker']
                candidates = [tuple(map(int,p)) for p in np.argwhere(a==info['bg'])
                              if 0<p[0]<a.shape[0]-1 and 0<p[1]<a.shape[1]-1
                              and p[0] not in info['rows'] and p[1] not in info['cols']
                              and _macro(p,info['rows'],info['cols'])==info['address']
                              and all(tuple(p) not in f['whole'] for f in info['frames'])]
                if candidates:
                    moved = a.copy()
                    moved[r,c] = info['bg']
                    moved[candidates[0]] = int(a[r,c])
                    expected = pair['output'].copy()
                    expected[r,c] = info['bg']
                    expected[candidates[0]] = int(a[r,c])
                    _require(np.array_equal(solve(moved),expected),'marker metric-position shortcut')
        signatures.add(digest.hexdigest())
        # Reassign all ten literal colors, then predict again from input alone.
        # This includes background and structural roles, not just signal swaps.
        permutation = np.random.default_rng(seed+173).permutation(10).astype(np.int8)
        for pair in puzzle['train'][:1]:
            _require(np.array_equal(solve(permutation[pair['input']]),permutation[pair['output']]),'palette equivariance')
    _require(len(signatures)==42,'seed diversity')
    _require(addresses==set(DIRECTIONS) and source_addresses==set(DIRECTIONS),'address coverage')
    _require(len(shapes)>=20 and len(profiles)>=8 and len(palettes)>=30,'structural diversity')
    _require(len(payload_shapes)==3 and forks>=10 and gates>200,'payload/routing diversity')
    _require(flip_count>=42 and disagree_count>=42,'causal witnesses')
    _require(raw_disagrees>=42 and panel_disagrees>=42 and route_disagrees>0,'vote shortcut counterexamples')
    first = generate_puzzle(0)
    second = generate_puzzle(0)
    _require(all(np.array_equal(p[k],q[k]) for s in first for p,q in zip(first[s],second[s]) for k in p),'determinism')
    # Output mutation must be rejected. No generated examples are saved by tests.
    broken = {s:[{k:v.copy() for k,v in p.items()} for p in first[s]] for s in first}
    info = _analyze(broken['test'][0]['input'])
    r,c = info['hub']
    broken['test'][0]['output'][r+1,c+1] = info['scaffold']
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong collector output accepted')
    return dict(seeds=42,pairs=pairs,shapes=len(shapes),weight_profiles=len(profiles),
                palettes=len(palettes),collector_move_flips=flip_count,
                collector_differs_from_payload=disagree_count,branched_pairs=forks,gate_cells=gates,
                raw_cell_vote_disagrees=int(raw_disagrees),panel_vote_disagrees=int(panel_disagrees),
                route_area_vote_disagrees=int(route_disagrees))
