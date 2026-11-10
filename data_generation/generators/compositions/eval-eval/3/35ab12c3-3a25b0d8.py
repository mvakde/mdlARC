"""Shadow-selected rooms: sparse translated routes address a deformed color map.

Only NumPy and the standard library are used. No files, examples, or random state
are consulted by solve(). All diagnostic sweeps are streaming and bounded.
"""

from collections import Counter
import hashlib
import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0,880835766,811246978,674504570,164885997,623114392,791528071,105815100,986876149,836026124,4348973,922424461,522357678,916441764,203727127,313750856,972690420,235098685,386275476,994956608,510790577,137036905,547767433,940152987,546342358,884343304,384143022,235330160,797183769,675063613,729581097,497712689,52089267,486437811,860480695,154964164,839953256,582434533,654826513,235542255,380665519,291732071)


def _flood_parts(mask, diagonal=False, enclosed_only=False):
    """Flat-index flood fill; discard exterior components before allocation."""
    height,width = mask.shape
    pending = set(np.flatnonzero(mask).tolist())
    result = []
    bottom = (height-1)*width
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, part = [start], []
        enclosed = True
        while stack:
            point = stack.pop()
            part.append(point)
            column = point % width
            if enclosed_only and (point < width or point >= bottom or
                                  column == 0 or column == width-1):
                enclosed = False
            neighbors = (point-width,point+width)
            if column:
                neighbors += ((point-1,point-width-1,point+width-1)
                              if diagonal else (point-1,))
            if column+1 < width:
                neighbors += ((point+1,point-width+1,point+width+1)
                              if diagonal else (point+1,))
            for neighbor in neighbors:
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        if not enclosed_only or enclosed:
            result.append(np.array([divmod(p,width) for p in sorted(part)],dtype=int))
    return result


def _components(mask, diagonal=False):
    return _flood_parts(mask,diagonal)


def _bbox(points):
    return tuple(map(int, points.min(0))) + tuple(map(int, points.max(0)+1))


def _rooms(wall_mask):
    # A complement component is an interior iff it never reaches a crop edge.
    # This is equivalent to padded exterior flooding without allocating/sorting
    # the exterior component or translating every retained room afterwards.
    return _flood_parts(~wall_mask,enclosed_only=True)


def _relations(rooms):
    boxes = [_bbox(p) for p in rooms]
    return [[tuple(-1 if a[k+2] <= b[k] else
                   1 if b[k+2] <= a[k] else 0 for k in (0,1))
             for b in boxes] for a in boxes]


def _match(donor_rooms, target_rooms):
    """Exact orientation-preserving spatial graph match; never use area/order."""
    a,b = _relations(donor_rooms),_relations(target_rooms)
    n = len(a)
    assert n == len(b)
    # Relation histograms prune without assigning a meaning to discovery order.
    a_hist,b_hist = [Counter(row) for row in a],[Counter(row) for row in b]
    candidates = [[j for j in range(n) if a_hist[i] == b_hist[j]]
                  for i in range(n)]
    answers = []

    def visit(mapping, used):
        if len(answers) > 1:
            return
        if len(mapping) == n:
            answers.append(dict(mapping))
            return
        options = {i:[j for j in candidates[i] if j not in used and all(
            a[i][k] == b[j][v] for k,v in mapping.items())]
                   for i in range(n) if i not in mapping}
        i = min(options, key=lambda k:(len(options[k]),k))
        for j in options[i]:
            mapping[i] = j
            visit(mapping, used | {j})
            del mapping[i]

    visit({},set())
    assert len(answers) == 1, "room correspondence must be unique"
    return answers[0]


def _line(a,b):
    dr,dc = b[0]-a[0],b[1]-a[1]
    assert (dr or dc) and (dr == 0 or dc == 0 or abs(dr) == abs(dc))
    sr,sc = int(dr > 0)-int(dr < 0),int(dc > 0)-int(dc < 0)
    return {(a[0]+i*sr,a[1]+i*sc) for i in range(max(abs(dr),abs(dc))+1)}


def _edges(points):
    ps = set(points)
    result = []
    for i,a in enumerate(points):
        for j,b in enumerate(points[:i]):
            dr,dc = abs(a[0]-b[0]),abs(a[1]-b[1])
            if (dr == 0 or dc == 0 or dr == dc) and not (
                    (_line(a,b)-{a,b}) & ps):
                result.append((j,i))
    return result


def _cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _intersects(a,b,c,d):
    if (max(a[0],b[0]) < min(c[0],d[0]) or
            max(c[0],d[0]) < min(a[0],b[0]) or
            max(a[1],b[1]) < min(c[1],d[1]) or
            max(c[1],d[1]) < min(a[1],b[1])):
        return False
    return _cross(a,b,c)*_cross(a,b,d) <= 0 and _cross(c,d,a)*_cross(c,d,b) <= 0


def _infer_route(points):
    """Prefer one simple full cycle; otherwise require one simple full path."""
    assert 2 <= len(points) <= 8 and len(set(points)) == len(points)
    edges = _edges(points)
    adjacency = [[] for p in points]
    for k,(i,j) in enumerate(edges):
        adjacency[i].append((j,k))
        adjacency[j].append((i,k))
    conflicts = [set() for e in edges]
    for k,(a,b) in enumerate(edges):
        for l,(c,d) in enumerate(edges[:k]):
            if not {a,b} & {c,d} and _intersects(points[a],points[b],points[c],points[d]):
                conflicts[k].add(l)
                conflicts[l].add(k)

    def search(closed):
        answers,visits = set(),[0]

        def visit(order,used,chosen):
            visits[0] += 1
            assert visits[0] <= 20000, "route search budget exceeded"
            if len(answers) > 1:
                return
            if len(order) == len(points):
                if closed:
                    closing = [k for v,k in adjacency[order[-1]] if v == order[0]]
                    if not closing or conflicts[closing[0]] & set(chosen):
                        return
                    chosen = chosen+[closing[0]]
                answers.add(tuple(sorted(chosen)))
                return
            for v,k in adjacency[order[-1]]:
                if v not in used and not conflicts[k] & set(chosen):
                    visit(order+[v],used|{v},chosen+[k])

        for start in ([0] if closed else range(len(points))):
            visit([start],{start},[])
            if len(answers) > 1:
                break
        return answers

    cycles = search(True) if len(points) >= 3 else set()
    assert len(cycles) <= 1, "ambiguous cycles"
    choices = cycles or search(False)
    assert len(choices) == 1, "ambiguous or disconnected route"
    chosen = [edges[k] for k in next(iter(choices))]
    pixels = set().union(*(_line(points[i],points[j]) for i,j in chosen))
    return chosen,pixels,bool(cycles),len(edges)-len(chosen)


def _selected(rooms,pixels):
    return {i for i,room in enumerate(rooms) if any(tuple(p) in pixels for p in room)}


def _extract(inp):
    assert inp.ndim == 2 and np.issubdtype(inp.dtype,np.integer)
    assert all(1 <= n <= 30 for n in inp.shape) and np.all((inp >= 0)&(inp <= 9))
    bg = int(inp[0,0])
    assert all(np.all(edge == bg) for edge in (inp[0],inp[-1],inp[:,0],inp[:,-1]))
    possibilities = []
    for color in np.unique(inp):
        if color == bg or np.count_nonzero(inp == color) < 18:
            continue
        parts = _components(inp == color,diagonal=True)
        if len(parts) != 2:
            continue
        objects = []
        for part in parts:
            r,c,R,C = _bbox(part)
            mask = np.zeros((R-r,C-c),dtype=bool)
            mask[part[:,0]-r,part[:,1]-c] = True
            rooms = _rooms(mask)
            objects.append((mask,rooms,(r,c,R,C)))
        if all(3 <= len(o[1]) <= 9 for o in objects):
            possibilities.append((int(color),objects))
    assert len(possibilities) == 1, "expected two recognizable compartment frameworks"
    wall,objects = possibilities[0]
    donors = []
    for i,(mask,rooms,(r,c,R,C)) in enumerate(objects):
        crop = inp[r:R,c:C]
        values = [np.unique(crop[tuple(room.T)]) for room in rooms]
        if all(len(v) == 1 and int(v[0]) not in (bg,wall) for v in values):
            donors.append((i,[int(v[0]) for v in values]))
    assert len(donors) == 1
    di,colors = donors[0]
    assert len(set(colors)) >= 2
    ti = 1-di
    dmask,dr,db = objects[di]
    tmask,tr,tb = objects[ti]
    mapping = _match(dr,tr)
    room_index = {tuple(p):i for i,room in enumerate(tr) for p in room.tolist()}

    def reached(pixels):
        return {room_index[p] for p in pixels if p in room_index}

    r,c,R,C = tb
    target = np.full(tmask.shape,bg,dtype=np.int8)
    target[tmask] = wall
    extra = (inp != bg)&(inp != wall)
    extra[db[0]:db[2],db[1]:db[3]] = False
    groups = {int(color):[tuple(map(int,p)) for p in np.argwhere(extra&(inp == color))]
              for color in np.unique(inp[extra])}
    assert len(groups) == 2
    route_groups = [(color,ps) for color,ps in groups.items() if 2 <= len(ps) <= 8]
    markers = [(color,ps[0]) for color,ps in groups.items() if len(ps) == 1]
    assert len(route_groups) == len(markers) == 1
    route_color,points = route_groups[0]
    marker_color,marker = markers[0]
    assert route_color not in colors and marker_color not in colors
    anchors = [p for p in points if max(abs(p[0]-marker[0]),abs(p[1]-marker[1])) == 1]
    assert len(anchors) == 1
    offset = marker[0]-anchors[0][0],marker[1]-anchors[0][1]
    assert all(r-1 <= p[0] <= R and c-1 <= p[1] <= C for p in points+[marker])
    assert all(np.any(inp[r:R,c:C][tuple(room.T)] == bg) for room in tr)
    edges,stroke,closed,chords = _infer_route(points)
    shadow = {(a+offset[0],b+offset[1]) for a,b in stroke}
    assert all(0 <= a < inp.shape[0] and 0 <= b < inp.shape[1] for a,b in shadow)
    local_shadow = {(a-r,b-c) for a,b in shadow}
    selected = reached(local_shadow)
    assert 0 < len(selected) < len(tr)
    out = target.copy()
    target_colors = {j:colors[i] for i,j in mapping.items()}
    for j in selected:
        out[tuple(tr[j].T)] = target_colors[j]
    no_shift = reached({(a-r,b-c) for a,b in stroke})
    points_only = reached({(a+offset[0]-r,b+offset[1]-c) for a,b in points})
    all_lines = set().union(*(_line(points[i],points[j]) for i,j in _edges(points)))
    all_selected = reached({(a+offset[0]-r,b+offset[1]-c) for a,b in all_lines})
    scan_wrong = any(target_colors[j] != colors[j] for j in selected)
    # Measure the strongest open-chain alternative to a closed route: delete
    # each boundary link in turn, retaining its endpoints via neighboring links.
    open_alternatives = []
    if closed:
        for omitted in range(len(edges)):
            open_stroke = set().union(*(_line(points[i],points[j])
                                       for k,(i,j) in enumerate(edges) if k != omitted))
            open_alternatives.append(reached({(a+offset[0]-r,b+offset[1]-c) for a,b in open_stroke}))
    sizes = [len(room) for room in tr]
    donor_sizes = [len(dr[i]) for i in range(len(dr))]
    room_boxes = [_bbox(room) for room in tr]
    area_wrong = any(target_colors[j] != colors[i] for i,j in zip(
        sorted(range(len(dr)),key=lambda i:(donor_sizes[i],i)),
        sorted(range(len(tr)),key=lambda j:(sizes[j],j))) if j in selected)
    info = {"background":bg,"wall":wall,"rooms":len(tr),"selected":len(selected),
            "vertices":len(points),"closed":closed,"chords":chords,"offset":offset,
            "completion_matters":selected != points_only,"translation_matters":selected != no_shift,
            "chord_matters":selected != all_selected,"scan_wrong":scan_wrong,"area_wrong":area_wrong,
            "closure_distinguished":bool(open_alternatives) and all(s != selected for s in open_alternatives),
            "closure_some_links_matter":any(s != selected for s in open_alternatives),
            "shape_changed":dmask.shape != tmask.shape or not np.array_equal(dmask,tmask),
            "palette":len(np.unique(inp)),"target_shape":target.shape,
            "room_shapes":tuple(sorted(len(room) for room in tr)),
            "nonrectangular":sum(len(room) != (R-r)*(C-c)
                                 for room,(r,c,R,C) in zip(tr,room_boxes)),
            "geometry":(target.shape,tmask.tobytes()),"target":target,"target_rooms":tr,
            "selected_ids":selected,"target_colors":target_colors,"target_box":tb,
            "source_color":route_color,"marker_color":marker_color}
    info["donor_rooms"],info["donor_box"],info["mapping"] = dr,db,mapping
    return out,info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input alone."""
    return _extract(np.asarray(input_grid))[0]


def _plan(rng,count):
    options = [(2,2),(2,3),(3,2),(3,3)]
    options = [s for s in options if s[0]*s[1] >= count]
    rows,cols = options[int(rng.integers(len(options)))]
    labels = np.arange(rows*cols).reshape(rows,cols)
    while len(np.unique(labels)) > count:
        edges = sorted({tuple(sorted((int(labels[r,c]),int(labels[r+dr,c+dc]))))
                        for r in range(rows) for c in range(cols) for dr,dc in ((1,0),(0,1))
                        if r+dr < rows and c+dc < cols and labels[r,c] != labels[r+dr,c+dc]})
        choices = []
        for a,b in edges:
            trial = labels.copy()
            trial[trial == b] = a
            # A room must be simply connected, with no enclosed other room.
            if not _rooms(trial == a):
                choices.append(trial)
        assert choices, "no compatible connected-room merger"
        labels = choices[int(rng.integers(len(choices)))]
    result = np.empty_like(labels)
    for i,label in enumerate(np.unique(labels)):
        result[labels == label] = i
    return result


def _geometry(rng,plan):
    heights = rng.integers(2,4,size=plan.shape[0])
    widths = rng.integers(2,4,size=plan.shape[1])
    # A pair fits with route padding, a separating gap and a blank outer border.
    while sum(heights)+len(heights)+1 > 12:
        heights[int(rng.choice(np.flatnonzero(heights > 2)))] -= 1
    while sum(widths)+len(widths)+1 > 12:
        widths[int(rng.choice(np.flatnonzero(widths > 2)))] -= 1
    ys,xs = np.r_[0,np.cumsum(heights+1)],np.r_[0,np.cumsum(widths+1)]
    labels = np.full((int(ys[-1])+1,int(xs[-1])+1),-1,dtype=int)
    for r in range(plan.shape[0]):
        for c in range(plan.shape[1]):
            label = plan[r,c]
            labels[ys[r]+1:ys[r+1],xs[c]+1:xs[c+1]] = label
            if r and plan[r-1,c] == label:
                labels[ys[r],xs[c]+1:xs[c+1]] = label
            if c and plan[r,c-1] == label:
                labels[ys[r]+1:ys[r+1],xs[c]] = label
            if r and c and np.all(plan[r-1:r+1,c-1:c+1] == label):
                labels[ys[r],xs[c]] = label
    for label in np.unique(plan):
        room = labels == label
        r,c,R,C = _bbox(np.argwhere(room))
        # Independent local erosion reorders same-band room starts. It also
        # makes steps/pockets; connectivity and visible correspondence decide
        # whether the resulting pair is accepted.
        for kind in rng.permutation(3):
            if rng.random() > .48:
                continue
            trial = room.copy()
            if kind == 0:
                trial[r,:] = False
            elif kind == 1:
                trial[:,c] = False
            else:
                trial[r,c] = False
            if trial.sum() >= 2 and len(_components(trial)) == 1:
                room = trial
        labels[(labels == label)&~room] = -1
    # One-cell walls surround every room. Shared walls connect the figure.
    occupied = labels >= 0
    near = np.zeros(labels.shape,dtype=bool)
    padded = np.pad(occupied,1)
    for dr,dc in ((r,c) for r in range(3) for c in range(3)):
        near |= padded[dr:dr+labels.shape[0],dc:dc+labels.shape[1]]
    labels[(labels < 0)&near] = -2
    r,c,R,C = _bbox(np.argwhere(labels != -1))
    return labels[r:R,c:C]


def _paint(labels,bg,wall,colors=None):
    grid = np.full(labels.shape,bg,dtype=np.int8)
    grid[labels == -2] = wall
    if colors is not None:
        for i,color in enumerate(colors):
            grid[labels == i] = color
    return grid


def _propose_route(rng,shape,kind,allowed=None):
    h,w = shape
    if kind == "path":
        count = int(rng.integers(2,7))
        points = [(int(rng.integers(h)),int(rng.integers(w)))]
        for i in range(count-1):
            dr,dc = OFFSETS[int(rng.integers(8))]
            length = int(rng.integers(2,max(h,w)))
            points.append((points[-1][0]+dr*length,points[-1][1]+dc*length))
        closed = False
    else:
        mode = 0 if kind == "square" else int(rng.integers(4))
        a,b = int(rng.integers(2,h)),int(rng.integers(2,w))
        if mode == 0:
            a = b = min(a,b)
            points = [(0,0),(0,b),(a,b),(a,0)]
        elif mode == 1:
            points = [(0,0),(0,b),(a,b),(a,0)]
        elif mode == 2:
            a = min(a,b)//2+1
            points = [(0,a),(a,2*a),(2*a,a),(a,0)]
        else:
            cut = int(rng.integers(1,min(a,b)))
            points = [(0,0),(0,b-cut),(cut,b),(a,b),(a,0)]
        ph,pw = max(r for r,c in points)+1,max(c for r,c in points)+1
        if ph > h or pw > w:
            return None
        sr,sc = int(rng.integers(h-ph+1)),int(rng.integers(w-pw+1))
        points = [(r+sr,c+sc) for r,c in points]
        closed = True
    if len(set(points)) != len(points) or not all(0 <= r < h and 0 <= c < w for r,c in points):
        return None
    if allowed is not None and not all(allowed[p] for p in points):
        return None
    if len(points) == 2:
        return points,_line(points[0],points[1]),False,0
    intended = {(i,i+1) for i in range(len(points)-1)}
    if closed:
        intended.add((0,len(points)-1))
    try:
        edges,pixels,inferred_closed,chords = _infer_route(points)
    except AssertionError:
        return None
    if set(edges) != intended or inferred_closed != closed:
        return None
    return points,pixels,closed,chords


def _place(rng,donor,target_patch,bg):
    # Both common orientations and both role orders. Offset along the other
    # axis is independent of the geometry and route displacement.
    stacked = bool(rng.integers(2))
    first_donor = bool(rng.integers(2))
    a,b = (donor,target_patch) if first_donor else (target_patch,donor)
    # Transpose temporary arrays only to reuse the placement calculation;
    # transpose the entire canvas back, leaving each original object unchanged.
    if stacked:
        a,b = a.T,b.T
    gap = int(rng.integers(1,3))
    height = max(a.shape[0],b.shape[0])+int(rng.integers(0,4))
    width = a.shape[1]+b.shape[1]+gap+2
    height = min(28,height)
    assert width <= 30
    canvas = np.full((height+2,width),bg,dtype=np.int8)
    ay,by = int(rng.integers(1,height-a.shape[0]+2)),int(rng.integers(1,height-b.shape[0]+2))
    canvas[ay:ay+a.shape[0],1:1+a.shape[1]] = a
    col = 1+a.shape[1]+gap
    canvas[by:by+b.shape[0],col:col+b.shape[1]] = b
    return canvas.T if stacked else canvas


def _square_proposals(rng,patch,bg):
    """Enumerate a small finite geometric domain instead of retrying squares."""
    h,w = patch.shape
    result = []
    for size in range(2,min(h,w)):
        for r in range(h-size):
            for c in range(w-size):
                points = [(r,c),(r,c+size),(r+size,c+size),(r+size,c)]
                if all(patch[p] == bg for p in points):
                    stroke = set().union(*(_line(points[i],points[(i+1)%4]) for i in range(4)))
                    result.append((points,stroke,True,2))
    rng.shuffle(result)
    return result


def _pair(rng,count,offset,kind,need_chord=False,need_scan=False):
    for geometry_attempt in range(160):
        plan = _plan(rng,count)
        d,t = _geometry(rng,plan),_geometry(rng,plan)
        turns = int(rng.integers(4))
        d,t = np.rot90(d,turns).copy(),np.rot90(t,turns).copy()
        if rng.integers(2):
            d,t = d[:,::-1].copy(),t[:,::-1].copy()
        dr,tr = _rooms(d == -2),_rooms(t == -2)
        if len(dr) != count or len(tr) != count:
            continue
        if len(_components(d == -2,True)) != 1 or len(_components(t == -2,True)) != 1:
            continue
        try:
            mapping = _match(dr,tr)
        except AssertionError:
            continue
        if any(int(d[tuple(dr[i][0])]) != int(t[tuple(tr[j][0])]) for i,j in mapping.items()):
            continue
        if d.shape == t.shape and np.array_equal(d == -2,t == -2):
            continue
        if need_scan and all(i == j for i,j in mapping.items()):
            continue
        palette = list(map(int,rng.permutation(10)))
        bg,wall,route_color,marker_color = palette[:4]
        richness = int(rng.integers(2,min(count,6)+1))
        colors = palette[4:4+richness]
        colors += [colors[int(rng.integers(richness))] for i in range(count-richness)]
        rng.shuffle(colors)
        donor = _paint(d,bg,wall,colors)
        target = _paint(t,bg,wall)
        labels = np.pad(t,1,constant_values=-1)
        patch = np.pad(target,1,constant_values=bg)
        room_ids = set(range(count))
        donor_scan = [colors[int(d[tuple(p[0])])] for p in dr]
        target_scan = [int(t[tuple(p[0])]) for p in tr]
        area_wrong_labels = {target_scan[j] for i,j in zip(
            sorted(range(count),key=lambda i:(len(dr[i]),i)),
            sorted(range(count),key=lambda j:(len(tr[j]),j)))
            if colors[target_scan[j]] != donor_scan[i]}
        if need_scan and not any(colors[label] != donor_scan[j] for j,label in enumerate(target_scan)):
            continue

        label_rows = labels.tolist()

        def hit(pixels):
            return {label_rows[r][c] for r,c in pixels if label_rows[r][c] >= 0}

        proposals = _square_proposals(rng,patch,bg) if kind == "square" else None
        allowed = patch == bg
        for route_attempt in range(len(proposals) if proposals is not None else 650):
            proposed = proposals[route_attempt] if proposals is not None else _propose_route(rng,patch.shape,kind,allowed)
            if proposed is None:
                continue
            points,stroke,closed,chords = proposed
            if any(patch[p] != bg for p in points):
                continue
            shadow = {(r+offset[0],c+offset[1]) for r,c in stroke}
            if not all(0 <= r < patch.shape[0] and 0 <= c < patch.shape[1] for r,c in shadow):
                continue
            selected = hit(shadow)
            point_selected = hit({(r+offset[0],c+offset[1]) for r,c in points})
            if not selected or selected == room_ids or selected == hit(stroke) or selected == point_selected:
                continue
            if need_chord:
                all_pixels = set().union(*(_line(points[i],points[j]) for i,j in _edges(points)))
                if not chords or selected == hit({(r+offset[0],c+offset[1]) for r,c in all_pixels}):
                    continue
            if need_scan and not any(label in selected and colors[label] != donor_scan[j]
                                     for j,label in enumerate(target_scan)):
                continue
            anchors = []
            for p in points:
                q = p[0]+offset[0],p[1]+offset[1]
                if q in points or patch[q] != bg:
                    continue
                if sum(max(abs(q[0]-v[0]),abs(q[1]-v[1])) == 1 for v in points) == 1:
                    anchors.append(q)
            if not anchors:
                continue
            marker = anchors[int(rng.integers(len(anchors)))]
            marked = patch.copy()
            for p in points:
                marked[p] = route_color
            marked[marker] = marker_color
            if any(not np.any(marked[labels == label] == bg) for label in room_ids):
                continue
            expected = target.copy()
            for label in selected:
                expected[t == label] = colors[label]
            inp = _place(rng,donor,marked,bg)
            # Reject known witness failures before parsing the whole canvas.
            # Keep this after marker/placement draws to preserve the RNG stream
            # and every already accepted seeded puzzle byte for byte.
            if need_scan and not selected & area_wrong_labels:
                continue
            if need_chord:
                side_hits = [hit({(r+offset[0],c+offset[1]) for r,c in
                                  _line(points[i],points[(i+1)%len(points)])})
                             for i in range(len(points))]
                side_counts = Counter(label for side in side_hits for label in side)
                if not all(any(side_counts[label] == 1 for label in side) for side in side_hits):
                    continue
            try:
                predicted,info = _extract(inp)
            except AssertionError:
                continue
            if not np.array_equal(predicted,expected):
                continue
            if not info["completion_matters"] or not info["translation_matters"]:
                continue
            if need_chord and not (info["chord_matters"] and info["closure_distinguished"]):
                continue
            if need_scan and not (info["scan_wrong"] and info["area_wrong"]):
                continue
            return {"input":inp,"output":expected}
    raise RuntimeError("bounded map/route construction exhausted")


def _counts(rng,num_train,num_test):
    for name,value,low,high in (("num_train",num_train,2,10),("num_test",num_test,1,4)):
        if value is not None and (isinstance(value,(bool,np.bool_)) or
                not isinstance(value,(int,np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None:
        num_train = int(rng.integers((num_test or 1)+1,11))
    if num_test is None:
        num_test = int(rng.integers(1,min(4,num_train-1)+1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train),int(num_test)


def _key(grid):
    return grid.shape,np.asarray(grid,dtype=np.int8).tobytes()


def generate_puzzle(seed: int,num_train: int=4,num_test: int=1) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    nt,nq = _counts(rng,num_train,num_test)
    # Only the presentation curriculum is puzzle-specific. All answer rules,
    # palette roles, graph matching and overlap conventions are input-derived.
    direction_order = rng.permutation(8)
    held = OFFSETS[int(direction_order[0])]
    train_offsets = [OFFSETS[int(k)] for k in direction_order[1:]]
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",nt),("test",nq)):
        for index in range(count):
            witness = split == "train" and index == 1
            room_count = 9 if witness else int(rng.integers(3,6)) if split == "train" and index == 0 else int(rng.integers(3,10))
            kind = "square" if witness else "path" if split == "train" and index == 0 else str(rng.choice(["path","cycle"]))
            offset = train_offsets[index % 7] if split == "train" else held
            for retry in range(12):
                pair = _pair(rng,room_count,offset,kind,need_chord=witness,need_scan=witness)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct input budget exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen,infos = set(),{"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            for grid in pair.values():
                a = np.asarray(grid)
                assert a.ndim == 2 and np.issubdtype(a.dtype,np.integer)
                assert all(1 <= d <= 30 for d in a.shape) and np.all((a >= 0)&(a <= 9))
            inp,out = np.asarray(pair["input"]),np.asarray(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted,info = _extract(inp)
            assert np.array_equal(out,predicted), "wrong selected-room colors or crop"
            assert info["shape_changed"]
            assert info["completion_matters"] and info["translation_matters"]
            # Preserve all walls/exterior, fill whole chosen rooms uniformly,
            # leave every unchosen room blank, and erase both waypoint colors.
            target = info["target"]
            assert np.array_equal(out == info["wall"],target == info["wall"])
            interior = np.zeros(out.shape,dtype=bool)
            for j,room in enumerate(info["target_rooms"]):
                interior[tuple(room.T)] = True
                color = info["target_colors"][j] if j in info["selected_ids"] else info["background"]
                assert np.all(out[tuple(room.T)] == color)
            assert np.array_equal(out[~interior],target[~interior])
            assert info["source_color"] not in out and info["marker_color"] not in out
            # Counterfactual B omission: changing a selected donor label must
            # change the answer, whereas route-color painting is always wrong.
            j = min(info["selected_ids"])
            assert info["target_colors"][j] != info["source_color"]
            infos[split].append(info)
    training = infos["train"]
    assert any(not i["closed"] for i in training)
    assert any(i["chord_matters"] for i in training)
    assert any(i["closure_distinguished"] for i in training)
    assert any(i["scan_wrong"] for i in training)
    assert any(i["area_wrong"] for i in training)
    train_dirs = {i["offset"] for i in training}
    test_dirs = {i["offset"] for i in infos["test"]}
    assert len(train_dirs) >= 2 and not train_dirs & test_dirs
    return None


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for k in ("input","output"):
                g = np.asarray(pair[k],dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds, retaining only small hashes and feature sets."""
    signatures,structures,shapes,offsets,room_counts,vertices,roles = (set() for i in range(7))
    flags = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        parsed = {split:[_extract(pair["input"]) for pair in puzzle[split]]
                  for split in ("train","test")}
        train_dirs = {m["offset"] for _,m in parsed["train"]}
        test_dirs = {m["offset"] for _,m in parsed["test"]}
        assert len(train_dirs) >= 2 and not train_dirs & test_dirs
        for split in ("train","test"):
            for pair,(_,m) in zip(puzzle[split],parsed[split]):
                structures.add(hashlib.sha256(m["geometry"][1]).hexdigest())
                shapes.add(pair["input"].shape)
                offsets.add(m["offset"])
                room_counts.add(m["rooms"])
                vertices.add(m["vertices"])
                roles.add((m["background"],m["wall"]))
                for name in ("closed","chord_matters","scan_wrong","area_wrong","nonrectangular"):
                    flags[name] += bool(m[name])
        # A direct B counterfactual changes one donor payload, holding A's
        # entire spatial selection fixed. Only the corresponding selected
        # target room can change; this is checked on a real query each seed.
        pair = puzzle["test"][0]
        original,m = parsed["test"][0]
        j = min(m["selected_ids"])
        i = next(i for i,v in m["mapping"].items() if v == j)
        banned = {m["background"],m["wall"],m["source_color"],m["marker_color"],m["target_colors"][j]}
        unused = [c for c in range(10) if c not in banned and c not in m["target_colors"].values()]
        replacement = unused[0] if unused else next(color for color in m["target_colors"].values() if color != m["target_colors"][j])
        changed = pair["input"].copy()
        donor_room = m["donor_rooms"][i]+np.array(m["donor_box"][:2])
        changed[tuple(donor_room.T)] = replacement
        expected = original.copy()
        expected[tuple(m["target_rooms"][j].T)] = replacement
        assert np.array_equal(solve(changed),expected)
    assert len(signatures) == 42
    assert len(structures) >= 80 and len(shapes) >= 30 and len(roles) >= 35
    assert offsets == set(OFFSETS) and room_counts == set(range(3,10))
    assert {2,3,4} <= vertices and max(vertices) >= 5
    assert min(flags.values()) >= 10
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            assert _counts(np.random.default_rng(12000+a*10+b),a,b) == (a,b)
    # The host already generates all 30 explicit combinations. Keep the same
    # count-contract coverage here, with full generation at its three bounds,
    # rather than repeating that entire expensive host sweep inside self_test.
    for a,b in ((2,1),(10,1),(10,4)):
        puzzle = generate_puzzle(12000+a*10+b,a,b)
        assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,True),(2.0,1),(4,1.5),(-1,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    assert _puzzle_digest(generate_puzzle(410)) == _puzzle_digest(generate_puzzle(410))
    for counts in ((None,1),(None,4),(2,None),(10,None),(None,None)):
        validate_generated_puzzle(generate_puzzle(55,*counts))
    p = generate_puzzle(0,2,1)
    # Translation/shape semantics must survive changing every literal color
    # and changing the common orientation of the complete input.
    query = p["test"][0]
    palette = np.random.default_rng(917).permutation(10)
    assert np.array_equal(solve(palette[query["input"]]),palette[query["output"]])
    for turns in range(4):
        for reflected in (False,True):
            inp,out = np.rot90(query["input"],turns),np.rot90(query["output"],turns)
            if reflected:
                inp,out = inp[:,::-1],out[:,::-1]
            assert np.array_equal(solve(inp),out)
    _,m = _extract(p["test"][0]["input"])
    room = m["target_rooms"][min(m["selected_ids"])]
    p["test"][0]["output"] = p["test"][0]["output"].copy()
    p["test"][0]["output"][tuple(room[0])] = m["background"]
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        return None
    raise AssertionError("a missing selected-room fill was accepted")
