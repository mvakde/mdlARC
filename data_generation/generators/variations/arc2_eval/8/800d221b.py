import numpy as np
import heapq

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _resolve_counts(seed, num_train, num_test):
    def checked(x, name, lo, hi):
        if x is None:
            return None
        if isinstance(x, (bool, np.bool_)) or not isinstance(
            x, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer or None")
        x = int(x)
        if not lo <= x <= hi:
            raise ValueError(f"{name} is out of range")
        return x

    tr = checked(num_train, "num_train", 2, 10)
    te = checked(num_test, "num_test", 1, 4)
    s = int(seed)

    if tr is None and te is None:
        te = 1 + s % 4
        tr = te + 1 + (s // 4) % (10 - te)
    elif tr is None:
        lo = max(2, te + 1)
        tr = lo + s % (11 - lo)
    elif te is None:
        te = 1 + s % min(4, tr - 1)

    if tr <= te:
        raise ValueError("num_train must be greater than num_test")
    return tr, te

def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    labels = np.full((h, w), -1, dtype=np.int16)

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells = []

            while stack:
                x, y = stack.pop()
                cells.append((x, y))

                for dx, dy in _DIRS:
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < h
                        and 0 <= ny < w
                        and mask[nx, ny]
                        and not seen[nx, ny]
                    ):
                        seen[nx, ny] = True
                        stack.append((nx, ny))

            idx = len(comps)
            for x, y in cells:
                labels[x, y] = idx
            comps.append(cells)

    return comps, labels

def _ports(cr, cc):
    return [
        [(cr - 1, cc + j) for j in range(3)],
        [(cr + 3, cc + j) for j in range(3)],
        [(cr + j, cc - 1) for j in range(3)],
        [(cr + j, cc + 3) for j in range(3)],
    ]

def _puzzles_equal(a, b):
    if set(a) != set(b):
        return False

    for split in ("train", "test"):
        if len(a[split]) != len(b[split]):
            return False

        for x, y in zip(a[split], b[split]):
            if not np.array_equal(
                x["input"],
                y["input"],
            ):
                return False
            if not np.array_equal(
                x["output"],
                y["output"],
            ):
                return False

    return True

# Exact source fixtures: regression checks and content-based copy prevention.
_SOURCE = (

    (

        '3434899999984444 4343899999984434 3334888999983344 3433898999988888 8888898999998999 9899998999998999 9899988888888999 9888888899999999 9999988888888889 9999998999999989 9999998999999989 9999998999999989 9999998999999989 9999988889999888 9999984489999834 9999984389999844',

        '3434399999944444 4343399999944434 3334333999943344 3433393999944444 3333393999994999 9399993999994999 9399988844444999 9333384899999999 9999988844444449 9999994999999949 9999994999999949 9999994999999949 9999994999999949 9999944449999444 9999944449999434 9999944349999444',

    ),

    (

        '0000000522 0000555521 0000500555 0005550000 0555555550 0505550050 0500000050 5555000555 2115000522 1215000512',

        '0000000222 0000222221 0000200222 0005550000 0115252220 0105550020 0100000020 1111000222 2111000222 1211000212',

    ),

    (

        '66666788887565788888 66566788887655788888 56666777887777788888 66666787888788888888 66666787788788888888 77777788787788888777 88878888787888888756 88878888787887777755 88877777777777888756 88888888777888888765 88877777777888888777 88878888787888888888 88878888787888888888 77777778787888888888 65666677787888888888 66666678887888888888 66666678887888888888 66666578877777778888 65665678876556578888 66666678876556578888',

        '66666688885565588888 66566688885655588888 56666666885555588888 66666686888588888888 66666686688588888888 66666688685588888555 88868888685888888556 88868888685885555555 88866666777555888556 88888888767888888565 88866666777888888555 88868888685888888888 88868888685888888888 66666668685888888888 65666666685888888888 66666668885888888888 66666668885888888888 66666568855555558888 65665668856556558888 66666668856556558888',

    ),

    (

        '333333333466666666666666666666 333333333466666666666666666666 333323333466666666666666666666 332233333466666666666666664444 223233333444444666666666664232 233333333466664666666644444232 333333233466664666666646664444 223333323444464666666646666666 333333333466464666666446666666 444444444466464666666466666666 666664666666464666664466666666 666664644444444444444666666666 666664446666444666666666666666 666666666444444666666666666666 666666666466464666666666666666 666666666466464446666666666666 666666444466466646666666666666 666666466666466646666666666666 666666466666466646666666666666 666666466666466646666666666666 666666466666464444444444444444 444666466666464222222222222222 334444466666444222222222222222 324666666666664222222322222232 324666666666664222232222222332 444666666666664222222222222232 666666666666664222222222322232 666666666666664222222222222222 666666666666664222222222222222 666666666666664222222222222222',

        '333333333366666666666666666666 333333333366666666666666666666 333323333366666666666666666666 332233333366666666666666662222 223233333333333666666666662232 233333333366663666666622222232 333333233366663666666626662222 223333323333363666666626666666 333333333366363666666226666666 333333333366363666666266666666 666663666666363666662266666666 666663633333444222222666666666 666663336666434666666666666666 666666666333444666666666666666 666666666366262666666666666666 666666666366262226666666666666 666666333366266626666666666666 666666366666266626666666666666 666666366666266626666666666666 666666366666266626666666666666 666666366666262222222222222222 333666366666262222222222222222 333333366666222222222222222222 323666666666662222222322222232 323666666666662222232222222332 333666666666662222222222222232 666666666666662222222222322232 666666666666662222222222222222 666666666666662222222222222222 666666666666662222222222222222',

    ),

)


def _grid_key(grid):
    return grid.shape, tuple(map(int, grid.flat))


def _source_arrays():
    return [tuple(np.array([[int(c) for c in row] for row in text.split()],
                           dtype=np.int8) for text in pair) for pair in _SOURCE]


def _solid_windows(mask):
    h, w = mask.shape
    solid = np.ones((h - 2, w - 2), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            solid &= mask[dr:dr + h - 2, dc:dc + w - 2]
    return list(zip(*np.nonzero(solid)))


def _interpret(grid, bg, scaffold, signals, hub, panels, panel_ids):
    h, w = grid.shape
    cr, cc = hub
    assert 0 < cr < h - 3 and 0 < cc < w - 3
    boxes, areas, majorities, minority_counts = [], [], [], []
    for cells in panels:
        rs, cs = zip(*cells)
        box = min(rs), max(rs), min(cs), max(cs)
        area = len(cells)
        assert area == (box[1] - box[0] + 1) * (box[3] - box[2] + 1)
        n0 = sum(grid[p] == signals[0] for p in cells)
        n1 = area - n0
        assert 0 < n0 < area and n0 != n1
        boxes.append(box)
        areas.append(area)
        majorities.append(signals[int(n1 > n0)])
        minority_counts.append(int(min(n0, n1)))
    route_mask = grid == scaffold
    route_mask[cr:cr + 3, cc:cc + 3] = False
    routes, route_ids = _components(route_mask)
    assert routes
    owners, labels = [], []
    for cells in routes:
        adjacent = set()
        for r, c in cells:
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and panel_ids[nr, nc] >= 0:
                    adjacent.add(int(panel_ids[nr, nc]))
        assert len(adjacent) == 1
        owner = adjacent.pop()
        owners.append(owner)
        labels.append(majorities[owner])
    panel_ports = [[] for _ in panels]
    port_labels, side_counts = [], []
    for side, ports in enumerate(_ports(cr, cc)):
        count = 0
        for pos, p in enumerate(ports):
            if grid[p] == scaffold:
                rid = int(route_ids[p])
                assert rid >= 0
                panel_ports[owners[rid]].append((side, pos))
                port_labels.append(labels[rid])
                count += 1
            else:
                assert grid[p] == bg
        side_counts.append(count)
    assert all(panel_ports)
    weights = list(map(len, panel_ports))
    counts = [port_labels.count(s) for s in signals]
    assert min(counts) > 0 and counts[0] != counts[1]
    winner = signals[int(counts[1] > counts[0])]
    return dict(shape=(h, w), background=bg, scaffold=scaffold, signals=signals,
                hub=hub, panels=panels, panel_boxes=boxes, panel_areas=areas,
                panel_majorities=majorities, minority_counts=minority_counts,
                routes=routes, route_owners=owners, route_majorities=labels,
                panel_ports=panel_ports, panel_weights=weights,
                route_cells=sum(map(len, routes)), port_labels=port_labels,
                port_count=len(port_labels), side_counts=side_counts,
                label_counts=dict(zip(signals, counts)), winner=winner)


def _inspect_input(grid):
    """Joint role/collector inference; a monochrome signal patch is not a hub."""
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    values = tuple(map(int, np.unique(grid)))
    assert len(values) == 4 and min(grid.shape) >= 5
    interpretations = []
    for scaffold in values:
        hubs = _solid_windows(grid == scaffold)
        if not hubs:
            continue
        # A scaffold interpretation must have one solid collector. Signal colors
        # may have any number of solid patches; they are checked as panels below.
        if len(hubs) != 1:
            continue
        for bg in values:
            if bg == scaffold:
                continue
            signals = tuple(s for s in values if s not in (bg, scaffold))
            mask = (grid == signals[0]) | (grid == signals[1])
            panels, ids = _components(mask)
            if not 2 <= len(panels) <= 8:
                continue
            try:
                interpretations.append(_interpret(grid, bg, scaffold, signals,
                                                  hubs[0], panels, ids))
            except AssertionError:
                continue
    assert len(interpretations) == 1, 'input needs one geometric interpretation'
    return interpretations[0]


def _solve_grid(grid):
    info = _inspect_input(grid)
    out = grid.copy()
    for cells, color in zip(info['routes'], info['route_majorities']):
        for p in cells:
            out[p] = color
    r, c = info['hub']
    out[r + 1, c + 1] = info['winner']
    return out, info


def _measure(grid, info=None):
    """All measurements are reconstructed from visible cells, never metadata."""
    if info is None:
        info = _inspect_input(grid)
    winner = info['winner']
    loser = next(s for s in info['signals'] if s != winner)
    areas = info['panel_areas']
    majorities = info['panel_majorities']
    def sign(x):
        return int(x > 0) - int(x < 0)
    def ballot(values, labels):
        return sign(sum(v if s == winner else -v for v, s in zip(values, labels)))
    h, w = grid.shape
    cr, cc = info['hub']
    ports = info['panel_ports']
    route_set = {p for cells in info['routes'] for p in cells}
    forks = sum(sum((r + dr, c + dc) in route_set for dr, dc in _DIRS) >= 3
                for r, c in route_set)
    thin_forks = sum(
        sum((r+dr,c+dc) in route_set for dr,dc in _DIRS)>=3 and not any(
            {(a,b),(a+1,b),(a,b+1),(a+1,b+1)} <= route_set
            for a in (r-1,r) for b in (c-1,c)) for r,c in route_set)
    bends = sum(any((r + dr, c) in route_set for dr in (-1, 1)) and
                any((r, c + dc) in route_set for dc in (-1, 1)) for r, c in route_set)
    interior = sum(r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1
                   for r0, r1, c0, c1 in info['panel_boxes'])
    solid = sum(len(_solid_windows(grid == s)) for s in info['signals'])
    return dict(panel_count=len(areas), port_count=info['port_count'],
                panel_weights=sorted(info['panel_weights']),
                side_counts=info['side_counts'], panel_areas=areas,
                panel_dimensions=[(r1-r0+1,c1-c0+1) for r0,r1,c0,c1 in info['panel_boxes']],
                local_minority_fractions=[m/a for m,a in zip(info['minority_counts'],areas)],
                local_vote_margins=[a-2*m for m,a in zip(info['minority_counts'],areas)],
                global_margin=abs(info['label_counts'][winner]-info['label_counts'][loser]),
                raw_vote=sign(int(np.sum(grid == winner))-int(np.sum(grid == loser))),
                panel_vote=ballot([1]*len(areas), majorities),
                panel_area_vote=ballot(areas, majorities),
                route_area_vote=ballot(list(map(len,info['routes'])),info['route_majorities']),
                multi_side=any(len({s for s,p in ps}) > 1 for ps in ports),
                separated_ports=any(any((s,0) in ps and (s,2) in ps and (s,1) not in ps
                                        for s in range(4)) for ps in ports),
                broad_ports=any(any((s,p+1) in ps for s,p in ps if p<2) for ps in ports),
                middle_singleton=any(len(ps)==1 and ps[0][1]==1 for ps in ports),
                forks=forks, thin_forks=thin_forks, bends=bends, route_cells=info['route_cells'],
                interior_panels=interior, edge_panels=len(areas)-interior,
                offset=max(abs(cr-(h-cr-3)),abs(cc-(w-cc-3))),
                area_ratio=max(areas)/min(areas), solid_signal_patches=solid,
                winner_index=info['signals'].index(winner),
                roles=[info['background'],info['scaffold'],*info['signals']])


def _neighbors(cells, h, w):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS
            if 0 <= r + dr < h and 0 <= c + dc < w}


def _rectangle(r, c, h, w):
    return {(x, y) for x in range(r, r + h) for y in range(c, c + w)}


def _pattern(rng, h, w, majority, minority):
    area = h*w
    kind = int(rng.integers(3))
    if kind == 0:
        nminor = (area-1)//2
    elif kind == 1:
        nminor = max(1, int(area*rng.uniform(.16,.36)))
    else:
        nminor = int(rng.integers(1, max(2, int(area*.10)+1)))
    nminor = min(nminor, (area-1)//2)
    arrangement = int(rng.integers(4))
    rr, cc = np.indices((h,w))
    if arrangement == 0:
        order = rng.permutation(area)
    elif arrangement == 1:
        r,c = int(rng.integers(h)),int(rng.integers(w))
        order = np.argsort((abs(rr-r)+abs(cc-c)+rng.random((h,w))*.4).ravel())
    elif arrangement == 2:
        order = np.argsort(((rr+cc)%2*area+rr*w+cc).ravel())
    else:
        order = np.argsort((rr*w+cc if rng.random()<.5 else cc*h+rr).ravel())
    flat = np.full(area, majority, dtype=np.int8)
    flat[order[:nminor]] = minority
    return flat.reshape(h,w)


def _path(start, targets, allowed, rng, roughness):
    """Random-cost shortest path, with a coarse field that varies route turns."""
    h,w = allowed.shape
    if start in targets:
        return {start}
    if not allowed[start]:
        return None
    noise = rng.random((h,w))*roughness
    # Coarse penalties create multi-turn detours instead of only L connectors.
    if roughness > 1:
        field = rng.random(((h+3)//4,(w+3)//4))*roughness*3
        noise += np.repeat(np.repeat(field,4,axis=0),4,axis=1)[:h,:w]
    costs = 1 + noise
    distances = {start:0.0}
    parent = {}
    heap = [(0.0,*start)]
    while heap:
        distance,r,c = heapq.heappop(heap)
        p = r,c
        if distance != distances[p]:
            continue
        if p in targets:
            cells = {p}
            while p != start:
                p = parent[p]
                cells.add(p)
            return cells
        for dr,dc in _DIRS:
            q = r+dr,c+dc
            if 0<=q[0]<h and 0<=q[1]<w and allowed[q]:
                nd = distance + costs[q]
                if nd < distances.get(q, float('inf')):
                    distances[q] = nd
                    parent[q] = p
                    heapq.heappush(heap,(nd,*q))
    return None


def _sample_panels(rng, h, w, hub, count, imbalance):
    cr,cc = hub
    reserved = _rectangle(cr-2,cc-2,7,7)
    panels = []
    for i in range(count):
        for attempt in range(90):
            maxh = max(2, min(h-6, 16)) if imbalance else max(2,min(h//2,10))
            maxw = max(2, min(w-6, 18)) if imbalance else max(2,min(w//2,10))
            # The two dimensions and strength are independent. Many-panel scenes
            # necessarily allocate smaller footprints, but not a fixed depth.
            if count >= 6:
                maxh,maxw = min(maxh,5),min(maxw,5)
            if rng.random()<.35:
                ph,pw = int(rng.integers(1,4)),int(rng.integers(2,5))
            else:
                ph,pw = int(rng.integers(2,maxh+1)),int(rng.integers(2,maxw+1))
            if ph*pw<3 or ph>h or pw>w:
                continue
            side = int(rng.integers(4))
            pad = 0 if rng.random()<.78 else int(rng.integers(1,4))
            if side<2:
                r = pad if side==0 else h-ph-pad
                c = int(rng.integers(w-pw+1))
            else:
                c = pad if side==2 else w-pw-pad
                r = int(rng.integers(h-ph+1))
            if r<0 or c<0 or r+ph>h or c+pw>w:
                continue
            cells = _rectangle(r,c,ph,pw)
            if cells & reserved:
                continue
            panels.append((r,c,ph,pw,cells))
            reserved |= cells | _neighbors(cells,h,w)
            break
        else:
            return None
    # Clockwise order makes the panel-to-port assignment planar.
    panels.sort(key=lambda p:float(np.arctan2(p[0]+(p[2]-1)/2-cr-1,
                                             p[1]+(p[3]-1)/2-cc-1)))
    return panels


def _sample_topology(rng, panel_count, evidence):
    if evidence:
        # Different constructions defeat unweighted panel counting. Rotation and
        # partition locations are sampled later; this isn't a fixed first pair.
        # These are all feasible weight multisets for 3--5 panels, at most
        # eight ports and weight <=4, with a minority of panels winning.
        options = {3:([3,1,1],[4,1,1],[4,2,1]),4:([4,1,1,1],),
                   5:([2,2,1,1,1],[3,2,1,1,1],
                      [3,1,1,1,1],[4,1,1,1,1])}
        weights = list(options[panel_count][int(rng.integers(len(options[panel_count])))])
        rng.shuffle(weights)
    else:
        total = int(rng.integers(max(3,panel_count),9))
        # Uniform bounded compositions retain lopsided port weights. Allocating
        # each extra port independently would favor nearly equal panel weights.
        def compositions(remaining, count):
            if count==0:
                return [()] if remaining==0 else []
            return [(weight,)+tail for weight in range(1,5)
                    if count-1<=remaining-weight<=4*(count-1)
                    for tail in compositions(remaining-weight,count-1)]
        options=compositions(total,panel_count)
        weights=list(options[int(rng.integers(len(options)))])
    assignments=[]
    for bits in range(1, (1<<panel_count)-1):
        nwin=sum(weights[i] for i in range(panel_count) if bits>>i&1)
        if nwin>sum(weights)/2:
            if not evidence or bits.bit_count()<panel_count/2:
                assignments.append(bits)
    if not assignments:
        return None
    # Sample agreement, tie and disagreement with unweighted panel voting when
    # the visible topology permits them; the port vote itself is always strict.
    categories={}
    for bits in assignments:
        margin=2*bits.bit_count()-panel_count
        categories.setdefault(int(margin>0)-int(margin<0),[]).append(bits)
    category=int(rng.choice(sorted(categories)))
    return weights,int(rng.choice(categories[category]))


def _route_layout(rng,h,w,hub,panels,weights,complexity):
    cr,cc=hub
    # Clockwise around the collector, starting at the upper left port.
    ring=[(cr-1,cc),(cr-1,cc+2),(cr,cc+3),(cr+2,cc+3),
          (cr+3,cc+2),(cr+3,cc),(cr+2,cc-1),(cr,cc-1)]
    chosen=sorted(map(int,rng.choice(8,size=sum(weights),replace=False)))
    # Match consecutive groups with panels in circular order. Minimize angular
    # travel, while retaining stochastic assignments among near alternatives.
    pangle=[float(np.arctan2(r+(ph-1)/2-cr-1,c+(pw-1)/2-cc-1))
            for r,c,ph,pw,cells in panels]
    candidates=[]
    for shift in range(len(chosen)):
        sequence=chosen[shift:]+chosen[:shift]
        groups=[]
        index=0
        cost=0.0
        for angle,weight in zip(pangle,weights):
            group=[ring[i] for i in sequence[index:index+weight]]
            for r,c in group:
                delta=abs(angle-float(np.arctan2(r-cr-1,c-cc-1)))
                cost+=min(delta,2*np.pi-delta)**2
            groups.append(set(group))
            index+=weight
        candidates.append((cost+float(rng.random())*1.5,groups))
    groups=min(candidates,key=lambda x:x[0])[1]
    # Sometimes turn a separated pair into a two-cell cable interval. Different
    # owners never get adjacent same-side port cells.
    for ports in _ports(cr,cc):
        occupied={p for group in groups for p in group if p in ports}
        if len(occupied)==1 and rng.random()<.35:
            old=next(iter(occupied))
            group=next(group for group in groups if old in group)
            group.remove(old)
            group.add(ports[1])
        for group in groups:
            if ports[0] in group and ports[2] in group and rng.random()<.22:
                old=ports[int(rng.choice([0,2]))]
                group.remove(old)
                group.add(ports[1])
    hubcells=_rectangle(cr,cc,3,3)
    collar=_rectangle(cr-1,cc-1,5,5)-hubcells
    allpanels=set().union(*(p[4] for p in panels))
    networks=[set(group) for group in groups]
    order=sorted(range(len(panels)),key=lambda i:-weights[i]+float(rng.random())*.8)
    for owner in order:
        ownpanel=panels[owner][4]
        otherpanels=allpanels-ownpanel
        othernets=set().union(*(n for i,n in enumerate(networks) if i!=owner))
        blocked=hubcells|collar|allpanels|othernets|_neighbors(otherpanels|othernets,h,w)
        blocked-=groups[owner]
        allowed=np.ones((h,w),dtype=bool)
        for p in blocked:
            allowed[p]=False
        # Border routing is optional; clipping at the canvas edge is natural.
        border=_neighbors(ownpanel,h,w)-ownpanel
        border={p for p in border if allowed[p]}
        if not border:
            return None
        net=set()
        for port in sorted(groups[owner]):
            # Trace each branch independently to the panel or established branch.
            targets=border|net
            path=None
            if complexity>2 and rng.random()<.70:
                candidates=list(zip(*np.nonzero(allowed)))
                if candidates:
                    # Explore a sizeable detour. Reaching the waypoint and then
                    # retracing a dead-end spur would add unevidenced noise, so
                    # the second leg must avoid the first leg entirely.
                    waypoint=candidates[int(rng.integers(len(candidates)))]
                    first=_path(port,{waypoint},allowed,rng,complexity)
                    second_allowed=allowed.copy()
                    if first is not None:
                        for p in first-{waypoint}:
                            second_allowed[p]=False
                    second=_path(waypoint,targets,second_allowed,rng,complexity)
                    if first is not None and second is not None:
                        path=first|second
            if path is None:
                path=_path(port,targets,allowed,rng,complexity)
            if path is None:
                return None
            net |= path
        if rng.random()<(.8 if complexity>1 else .35):
            # Follow one or more full panel edges; this produces varied attachment
            # positions, branches and genuine panel-border scaffold sections.
            border_components,_=_components(np.array([[((r,c) in border) for c in range(w)]
                                                      for r in range(h)],dtype=bool))
            attached=[set(cells) for cells in border_components if set(cells)&_neighbors(net,h,w) or set(cells)&net]
            for cells in attached:
                if rng.random()<.8:
                    net |= cells
        # All ports of this owner must be included, with no extra occupied ports.
        if not groups[owner]<=net or (net&collar)!=groups[owner]:
            return None
        networks[owner]=net
    scaffold=hubcells|set().union(*networks)
    mask=np.zeros((h,w),dtype=bool)
    for p in scaffold:
        mask[p]=True
    if _solid_windows(mask)!=[hub]:
        return None
    return networks


def _construct_pair(rng, roles, winner_index, bounds, evidence=False,
                    complexity=2, imbalance=False):
    for attempt in range(160):
        h,w=map(int,rng.integers(bounds[0],bounds[1]+1,size=2))
        if bounds[1]==30 and rng.random()<.16:
            h=w=30
        elif rng.random()<.16:
            w=h
        if min(h,w)<14:
            count=int(rng.choice([2,3])) if not evidence else 3
        else:
            count=int(rng.choice([2,3,4,5,6,7,8],p=[.13,.20,.20,.18,.12,.09,.08])) if not evidence else int(rng.choice([3,4,5]))
        topology=_sample_topology(rng,count,evidence)
        if topology is None:
            continue
        weights,bits=topology
        cr=int(rng.integers(3,h-5))
        cc=int(rng.integers(3,w-5))
        hub=(cr,cc)
        panels=_sample_panels(rng,h,w,hub,count,imbalance)
        if panels is None:
            continue
        networks=_route_layout(rng,h,w,hub,panels,weights,complexity)
        if networks is None:
            continue
        bg,scaffold,s0,s1=roles
        signals=sorted([s0,s1])
        winner=signals[winner_index]
        loser=signals[1-winner_index]
        inp=np.full((h,w),bg,dtype=np.int8)
        inp[cr:cr+3,cc:cc+3]=scaffold
        out=inp.copy()
        for i,((r,c,ph,pw,cells),network) in enumerate(zip(panels,networks)):
            majority=winner if bits>>i&1 else loser
            minority=loser if majority==winner else winner
            pattern=_pattern(rng,ph,pw,majority,minority)
            inp[r:r+ph,c:c+pw]=pattern
            out[r:r+ph,c:c+pw]=pattern
            for p in network:
                inp[p]=scaffold
                out[p]=majority
        out[cr+1,cc+1]=winner
        # Labels above come from the construction records. The independent
        # geometric reconstruction below never sees those records.
        try:
            predicted,info=_solve_grid(inp)
            assert np.array_equal(predicted,out)
            if evidence:
                assert _measure(inp,info)['panel_vote']==-1
            return {'input':inp,'output':out}
        except AssertionError:
            continue
    return None

def _fallback_pair(roles, winner_index, bounds, evidence, complexity, imbalance, excluded):
    # This separate deterministic stream is a last resort after bounded random
    # construction. Dimensions retain the requested regime and colors retain
    # the caller's roles; candidates still pass the same input-derived checks.
    for fallback_seed in range(256):
        pair = _construct_pair(np.random.default_rng(74123+fallback_seed), roles,
                               winner_index,bounds,evidence,complexity,imbalance)
        if pair is not None and _grid_key(pair['input']) not in excluded and (not evidence or all(
                _measure(pair['input'])[key] == -1 for key in
                ('panel_vote','raw_vote','panel_area_vote','route_area_vote'))):
            return pair
    raise AssertionError('bounded geometric fallback exhausted')


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    ntrain,ntest=_resolve_counts(int(seed),num_train,num_test)
    rng=np.random.default_rng(int(seed) % (1<<64))
    # The invariant conventions are geometric and shared by every puzzle. Only
    # palette policy and difficulty regime are chosen once per puzzle.
    mode=int(rng.integers(6))
    palette=list(map(int,rng.choice(10,4,replace=False)))
    varying_palette=mode in (1,2,3,4) and rng.random()<.85
    source_bounds=[(12,30),(10,20),(14,27),(16,28),(14,28),(12,27)][mode]
    target_bounds=[(12,30),(24,30),(18,30),(16,28),(14,28),(12,27)][mode]
    # Coverage is a constraint on a random sequence, not ordinal alternation.
    winner_indices=list(map(int,rng.integers(2,size=ntrain)))
    if len(set(winner_indices))==1:
        winner_indices[int(rng.integers(ntrain))]^=1
    evidence_slot=int(rng.integers(ntrain))
    puzzle={'train':[],'test':[]}
    seen=set()
    source_keys={(_grid_key(a),_grid_key(b)) for a,b in _source_arrays()}
    first_signals=None
    for split,count,bounds in [('train',ntrain,source_bounds),('test',ntest,target_bounds)]:
        for i in range(count):
            if varying_palette or (mode in (1,5) and split=='test'):
                roles=list(map(int,rng.choice(10,4,replace=False)))
                if split=='train' and i==1:
                    pool=[c for c in range(10) if c not in roles[:2] and c not in first_signals]
                    roles[2:]=list(map(int,rng.choice(pool,2,replace=False)))
            else:
                roles=palette.copy()
            if split=='train' and i==0:
                first_signals=set(roles[2:])
            winner_index=winner_indices[i] if split=='train' else int(rng.integers(2))
            evidence=split=='train' and i==evidence_slot
            complexity=float(rng.choice([0.,.6,2.,4.,7.]))
            if mode==3:
                complexity=float(rng.choice([.0,.6,2.]) if split=='train' else rng.choice([4.,7.,10.]))
            imbalance=bool(rng.random()<.4 or (mode==4 and split=='test'))
            # Evidence against total area and cable area is sampled from actual
            # outcomes; it does not encode the majority into a particular panel.
            best=None
            for _ in range(128 if evidence else 8):
                pair=_construct_pair(rng,roles,winner_index,bounds,evidence,complexity,imbalance)
                if pair is None:
                    continue
                ikey=_grid_key(pair['input'])
                if ikey in seen or (ikey,_grid_key(pair['output'])) in source_keys:
                    continue
                m=_measure(pair['input'])
                if not evidence or (m['raw_vote']==-1 and m['panel_area_vote']==-1 and m['route_area_vote']==-1):
                    best=pair
                    break
            if best is None:
                best=_fallback_pair(roles,winner_index,bounds,evidence,complexity,imbalance,seen)
            seen.add(_grid_key(best['input']))
            puzzle[split].append(best)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    tr,te=len(puzzle['train']),len(puzzle['test'])
    assert 2<=tr<=10 and 1<=te<=4 and tr>te
    source_keys={(_grid_key(a),_grid_key(b)) for a,b in _source_arrays()}
    seen_inputs,seen_pairs=set(),set()
    measures={'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            for a in pair.values():
                assert isinstance(a,np.ndarray) and a.ndim==2
                assert np.issubdtype(a.dtype,np.integer)
                assert all(1<=x<=30 for x in a.shape)
                assert np.all((a>=0)&(a<=9))
            inp,out=pair['input'],pair['output']
            assert inp.shape==out.shape and not np.array_equal(inp,out)
            ikey=_grid_key(inp)
            pkey=ikey,_grid_key(out)
            assert ikey not in seen_inputs and pkey not in seen_pairs and pkey not in source_keys
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            predicted,info=_solve_grid(inp.copy())
            assert np.array_equal(predicted,out)
            m=_measure(inp,info)
            assert 2<=m['panel_count']<=8 and 3<=m['port_count']<=8
            assert max(m['panel_weights'])<=4
            assert max(m['side_counts'])<=2
            assert all(0<x<.5 for x in m['local_minority_fractions'])
            measures[split].append(m)
    train=measures['train']
    # A minority of panels winning a majority of ports excludes panel-counting;
    # each pair has both local winners and untouched, meaningful minority cells.
    assert any(m['panel_vote']==-1 for m in train)
    assert any(all(m[key]==-1 for key in
                   ('panel_vote','raw_vote','panel_area_vote','route_area_vote'))
               for m in train)
    assert {m['winner_index'] for m in train}=={0,1}
    # Every test uses the identical role-recovery and propagation rule. Literal
    # palettes and sizes are observable input properties, not hidden conventions.


def self_test() -> None:
    # The original fixtures are regression evidence, not construction templates.
    for inp,out in _source_arrays():
        prediction,_=_solve_grid(inp.copy())
        assert np.array_equal(prediction,out)
    train_counts,test_counts,shapes,layouts,palettes=set(),set(),set(),set(),set()
    metrics=[]
    stable_count,varying_signals,consecutive_count=0,0,0
    for seed in range(42):
        puzzle=generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        train_info=[_inspect_input(p['input']) for p in puzzle['train']]
        varying_signals+=len({x['signals'] for x in train_info})>1
        roles=[tuple(_measure(p['input'])['roles'])
               for split in ('train','test') for p in puzzle[split]]
        stable_count+=len(set(roles))==1
        winners=[x['signals'].index(x['winner']) for x in train_info]
        consecutive_count+=any(a==b for a,b in zip(winners,winners[1:]))
        for split in ('train','test'):
            for pair in puzzle[split]:
                # No output is passed into inference.
                prediction,info=_solve_grid(pair['input'].copy())
                assert np.array_equal(prediction,pair['output'])
                metrics.append(_measure(pair['input'],info))
                shapes.add(pair['input'].shape)
                layouts.add((info['shape'],info['hub'],tuple(info['panel_boxes']),
                             tuple(tuple(p) for p in info['panel_ports'])))
                palettes.add(tuple([info['background'],info['scaffold'],*info['signals']]))
    assert len(train_counts)>=4 and len(test_counts)>=3
    assert stable_count>=3 and varying_signals>=10 and consecutive_count>=15
    assert len(shapes)>90 and len(layouts)>250 and len(palettes)>80
    assert {m['panel_count'] for m in metrics}==set(range(2,9))
    assert {m['port_count'] for m in metrics}==set(range(3,9))
    assert len({tuple(m['panel_weights']) for m in metrics})>=15
    assert len({m['global_margin'] for m in metrics})>=5
    assert max(max(m['panel_areas']) for m in metrics)>=100
    assert max(m['area_ratio'] for m in metrics)>=20
    assert max(m['route_cells'] for m in metrics)>=143
    assert max(m['offset'] for m in metrics)>=15
    assert any(m['solid_signal_patches'] for m in metrics)
    minorities=[f for m in metrics for f in m['local_minority_fractions']]
    assert min(minorities)<.05 and max(minorities)>.45
    assert sum(.15<f<.35 for f in minorities)>=100
    assert sum(m['thin_forks']>0 for m in metrics)>=40
    assert (30,30) in shapes
    for name in ('multi_side','separated_ports','broad_ports','middle_singleton','interior_panels'):
        assert sum(bool(m[name]) for m in metrics)>=10
    for name in ('raw_vote','panel_vote','panel_area_vote','route_area_vote'):
        assert {-1,1}<={m[name] for m in metrics}
    assert any(m['multi_side'] and m['separated_ports'] and m['area_ratio']>=8
               and m['solid_signal_patches'] for m in metrics)
    for seed,tr,te in [(101,2,1),(102,10,1),(103,10,4),(104,6,3),
                       (111,None,1),(112,None,4),(113,2,None),(114,10,None)]:
        p=generate_puzzle(seed,tr,te)
        validate_generated_puzzle(p)
        assert tr is None or len(p['train'])==tr
        assert te is None or len(p['test'])==te
    assert _puzzles_equal(generate_puzzle(107,4,2),generate_puzzle(107,4,2))
    assert not _puzzles_equal(generate_puzzle(107,4,2),generate_puzzle(108,4,2))
    for tr,te in [(1,1),(11,1),(3,0),(6,5),(2,2),(3,4),(True,1),(2.5,1)]:
        try:
            generate_puzzle(0,tr,te)
        except ValueError:
            continue
        raise AssertionError('invalid pair counts accepted')
    # Mutation checks exercise the input-derived validator and the collector fix.
    p=generate_puzzle(913,3,1)
    q={s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    inp=q['test'][0]['input']
    info=_inspect_input(inp)
    r,c=info['hub']
    q['test'][0]['output'][r+1,c+1]=info['scaffold']
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong collector label accepted')
    # Rearranging a panel preserves its vote and cannot invalidate the collector.
    inp,out=_source_arrays()[2]
    info=_inspect_input(inp)
    r0,r1,c0,c1=info['panel_boxes'][0]
    changed=inp.copy()
    block=changed[r0:r1+1,c0:c1+1]
    block[:]=np.sort(block,axis=None).reshape(block.shape)
    expected=out.copy()
    expected[r0:r1+1,c0:c1+1]=block
    assert np.array_equal(_solve_grid(changed)[0],expected)
