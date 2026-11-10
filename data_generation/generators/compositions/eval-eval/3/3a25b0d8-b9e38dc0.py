"""Shadow-activated rooms: a spatial atlas coupled to opaque sensor switches.

Only standard-library and NumPy operations are used. Nothing is read from disk;
solve receives just one input. Construction uses explicit masks, whereas solve
segments the input and scans optical lanes. All retries and self-tests are bounded.
"""
from collections import Counter
import hashlib
import numpy as np


HOST_SEEDS = (0,93618718,362142426,791016182,942289857,23979861,795790334,
              573664519,751117583,333879037,522128894,159059141,601130841,
              397579221,889593352,322940522,44725176,415411316,267683210,
              866631159,900260856,19242846,833319882,129829444,260547994,
              729747340,955396306,372124578,225284640,606455878,861343531,
              774019353,155153777,379792746,769931146,695577280,945279474,
              284899513,19601311,712647435,80792300,993057164)
_FOUR = ((-1,0),(1,0),(0,-1),(0,1))


def _components(mask, diagonal=False):
    left = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = tuple((r,c) for r in (-1,0,1) for c in (-1,0,1) if r or c) if diagonal else _FOUR
    while left:
        start = min(left)
        left.remove(start)
        part, pending = [start], [start]
        while pending:
            r,c = pending.pop()
            for dr,dc in steps:
                q = r+dr,c+dc
                if q in left:
                    left.remove(q)
                    pending.append(q)
                    part.append(q)
        result.append(np.array(part, dtype=int))
    return result


def _bbox(points):
    return (*map(int, points.min(0)), *map(int, points.max(0)+1))


def _rooms(barrier):
    a = np.pad(~barrier, 1, constant_values=True)
    return [p-1 for p in _components(a)
            if np.all(p > 0) and np.all(p < np.array(a.shape)-1)]


def _relations(rooms):
    boxes = [_bbox(p) for p in rooms]
    return [[tuple(-1 if a[k+2] <= b[k] else 1 if b[k+2] <= a[k] else 0
                   for k in (0,1)) for b in boxes] for a in boxes]


def _match(donor, target, colors):
    """All exact, orientation-preserving spatial-relationship matches must agree.

    Areas, absolute coordinates, discovery order and sensor states are not keys.
    The supported deformations preserve separated/overlapping room projections.
    """
    a,b = _relations(donor),_relations(target)
    n = len(a)
    options = {i:[j for j in range(n) if sorted(a[i]) == sorted(b[j])] for i in range(n)}
    solutions, visits = [],0

    def search(mapping, used):
        nonlocal visits
        visits += 1
        assert visits <= 4000, 'spatial correspondence exceeds supported grammar'
        if len(mapping) == n:
            values = [None]*n
            for i,j in mapping.items():
                values[j] = colors[i]
            if solutions:
                assert values == solutions[0], 'ambiguous atlas colors'
            else:
                solutions.append(values)
            return
        available = {i:[j for j in options[i] if j not in used and all(
            a[i][k] == b[j][v] for k,v in mapping.items())]
            for i in range(n) if i not in mapping}
        i = min(available, key=lambda x: (len(available[x]),x))
        for j in available[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({},set())
    assert solutions, 'no visible room correspondence'
    return solutions[0]


def _fit_flank(xs, ys, sign):
    end,value = int(xs[-1]),int(ys[-1])
    candidates = []
    for period in (1,2):
        for phase in range(period):
            pred = value + sign*((xs-end+phase)//period)
            length = 0
            for equal in (pred == ys)[::-1]:
                if not equal:
                    break
                length += 1
            if length >= 2*period+1 and len(set(ys[-length:])) >= 3:
                candidates.append((length,-period,phase))
    assert candidates, 'two readable terminal outward steps required'
    length,negative,phase = max(candidates)
    return -negative,phase,end,value,length


def _inspect_emitter(grid, wall_mask, background, wall):
    found = []
    for rotation in range(4):
        a = np.rot90(grid,-rotation)
        mask = np.rot90(wall_mask,-rotation)
        points = np.argwhere(mask)
        if len(points) < 12:
            continue
        rear,mouth = int(points[:,1].min()),int(points[:,1].max())
        cap = np.flatnonzero(mask[:,rear])
        if len(cap) < 3 or np.any(np.diff(cap) != 1) or mouth-rear < 4:
            continue
        upper,lower = [],[]
        for x in range(rear+1,mouth+1):
            rows = np.flatnonzero(mask[:,x])
            if len(rows) != 2 or rows[1]-rows[0] < 2:
                break
            upper.append(int(rows[0]))
            lower.append(int(rows[1]))
        else:
            xs = np.arange(rear+1,mouth+1)
            try:
                fits = (_fit_flank(xs,np.array(upper),-1),
                        _fit_flank(xs,np.array(lower),1))
            except AssertionError:
                continue
            footprint = np.zeros(a.shape, dtype=bool)
            for x in range(rear+1,a.shape[1]):
                if x <= mouth:
                    lo,hi = upper[x-rear-1],lower[x-rear-1]
                else:
                    lo,hi = [v+s*((x-e+p)//k) for (k,p,e,v,_),s in zip(fits,(-1,1))]
                footprint[:,x] = (np.arange(a.shape[0]) > lo) & (np.arange(a.shape[0]) < hi)
            if np.any(mask & footprint):
                continue
            chamber = footprint.copy()
            chamber[:,mouth+1:] = False
            patches = np.unique(a[chamber & (a != background)])
            if len(patches) != 1 or patches[0] == wall:
                continue
            light = int(patches[0])
            if len(_components(chamber & (a == light))) != 1:
                continue
            found.append(dict(rotation=rotation,canonical=a,background=background,
                              wall=wall,light=light,rear=rear,mouth=mouth,
                              fits=fits,footprint=footprint,chamber=chamber))
    assert len(found) == 1, 'one closed-back, open-ended spotlight required'
    return found[0]


def _analyze(input_grid):
    inp = np.asarray(input_grid)
    assert inp.ndim == 2 and np.issubdtype(inp.dtype,np.integer)
    assert all(1 <= n <= 30 for n in inp.shape) and np.all((0 <= inp) & (inp <= 9))
    counts = np.bincount(inp.ravel(),minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > inp.size//2, 'input background must dominate'
    parts = _components(inp != bg, diagonal=True)
    donors = []
    for part in parts:
        if len(part) < 12:
            continue
        r,c,R,C = _bbox(part)
        crop = np.full((R-r,C-c),bg,dtype=np.int8)
        crop[part[:,0]-r,part[:,1]-c] = inp[tuple(part.T)]
        if len(set(crop.flat)-{bg}) < 3:
            continue
        for wall in sorted(set(crop.flat)-{bg}):
            rooms = _rooms(crop == wall)
            if not 3 <= len(rooms) <= 7:
                continue
            colors = [np.unique(crop[tuple(p.T)]) for p in rooms]
            if not all(len(v) == 1 and v[0] not in (bg,wall) for v in colors):
                continue
            if sum(map(len,rooms)) != np.count_nonzero((crop != bg) & (crop != wall)):
                continue
            if len({int(v[0]) for v in colors}) < 2:
                continue
            donors.append((part,int(wall),[p+(r,c) for p in rooms],[int(v[0]) for v in colors]))
    assert len(donors) == 1, 'one atlas of uniformly colored rooms required'
    donor,wall,dr,colors = donors[0]
    receivers = []
    for part in parts:
        palette = set(map(int,inp[tuple(part.T)]))
        if len(palette) != 2 or wall not in palette:
            continue
        sensor = next(iter(palette-{wall}))
        r,c,R,C = _bbox(part)
        crop = np.full((R-r,C-c),bg,dtype=np.int8)
        crop[part[:,0]-r,part[:,1]-c] = inp[tuple(part.T)]
        rooms = _rooms(crop != bg)
        if len(rooms) != len(dr) or not all(np.all(crop[tuple(p.T)] == bg) for p in rooms):
            continue
        ids = np.full(inp.shape,-1,dtype=int)
        tr = [p+(r,c) for p in rooms]
        for i,p in enumerate(tr):
            ids[tuple(p.T)] = i
        sensors = {}
        for y,x in part:
            if inp[y,x] != sensor:
                continue
            adjacent = {int(ids[y+dy,x+dx]) for dy,dx in _FOUR
                        if 0 <= y+dy < inp.shape[0] and 0 <= x+dx < inp.shape[1]
                        and ids[y+dy,x+dx] >= 0}
            if len(adjacent) != 1:
                break
            sensors[int(y),int(x)] = adjacent.pop()
        else:
            if len(sensors) == len(tr) and set(sensors.values()) == set(range(len(tr))):
                receiver_colors = _match(dr,tr,colors)
                receivers.append((part,sensor,tr,sensors,receiver_colors))
    assert len(receivers) == 1, 'one receiver with one unambiguous sensor per room required'
    receiver,sensor,tr,sensors,room_colors = receivers[0]
    wall_mask = inp == wall
    wall_mask[tuple(donor.T)] = False
    wall_mask[tuple(receiver.T)] = False
    info = _inspect_emitter(inp,wall_mask,bg,wall)
    rotation = info['rotation']
    # Rotate coordinate fields, so correspondence and sensor recognition remain
    # independent of the optical simulation's canonical right-facing frame.
    ids = np.full(inp.shape,-1,dtype=int)
    for i,p in enumerate(tr):
        ids[tuple(p.T)] = i
    sensor_ids = np.full(inp.shape,-1,dtype=int)
    for p,i in sensors.items():
        sensor_ids[p] = i
    donor_mask = np.zeros(inp.shape,dtype=bool)
    donor_mask[tuple(donor.T)] = True
    receiver_mask = np.zeros(inp.shape,dtype=bool)
    receiver_mask[tuple(receiver.T)] = True
    donor_mask = np.rot90(donor_mask,-rotation)
    assert not np.any(donor_mask & info['footprint']), 'atlas must be outside the optical footprint'
    info.update(room_ids=np.rot90(ids,-rotation),sensor_ids=np.rot90(sensor_ids,-rotation),
                donor_mask=donor_mask,receiver_mask=np.rot90(receiver_mask,-rotation),
                room_colors=room_colors,sensor=sensor,donor_rooms=dr,donor_colors=colors)
    return info


def _scan(info, ignore_extras=False):
    a = info['canonical']
    bg,light,mouth = info['background'],info['light'],info['mouth']
    out = a.copy()
    out[info['chamber'] & (a == bg)] = light
    active = set()
    for y in range(a.shape[0]):
        blocked = False
        for x in range(mouth+1,a.shape[1]):
            if not info['footprint'][y,x]:
                continue
            occupied = a[y,x] != bg
            if ignore_extras and not info['receiver_mask'][y,x] and not info['donor_mask'][y,x]:
                occupied = False
            if occupied:
                if not blocked and info['sensor_ids'][y,x] >= 0:
                    active.add(int(info['sensor_ids'][y,x]))
                blocked = True
            elif not blocked and a[y,x] == bg:
                out[y,x] = light
    for i in active:
        out[info['room_ids'] == i] = info['room_colors'][i]
    return out,active


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, room correspondence, light and switches from this input only."""
    info = _analyze(input_grid)
    result,_ = _scan(info)
    return np.rot90(result,info['rotation']).copy()


def _plan(rng, count):
    choices = [(2,2),(2,3),(3,2),(2,4),(4,2)]
    rows,cols = choices[int(rng.choice([i for i,s in enumerate(choices) if np.prod(s) >= count]))]
    plan = np.arange(rows*cols).reshape(rows,cols)
    for r,c in ((0,0),(0,cols-1),(rows-1,0),(rows-1,cols-1)):
        if np.count_nonzero(plan >= 0) > count and rng.random() < .28:
            trial = plan.copy()
            trial[r,c] = -1
            if len(_components(trial >= 0)) == 1:
                plan = trial
    while len(set(plan.flat)-{-1}) > count:
        edges = [(int(plan[r,c]),int(plan[r+dr,c+dc])) for r in range(rows)
                 for c in range(cols) for dr,dc in ((1,0),(0,1))
                 if r+dr < rows and c+dc < cols and plan[r,c] >= 0
                 and plan[r+dr,c+dc] >= 0 and plan[r,c] != plan[r+dr,c+dc]]
        u,v = edges[int(rng.integers(len(edges)))]
        plan[plan == v] = u
    remap = {old:new for new,old in enumerate(sorted(set(plan.flat)-{-1}))}
    return np.array([[remap.get(int(v),-1) for v in row] for row in plan])


def _geometry(rng, plan, maximum, detail):
    heights = rng.integers(1,maximum+1,size=plan.shape[0])
    widths = rng.integers(1,maximum+1,size=plan.shape[1])
    if maximum == 3 and max(int(heights.max()),int(widths.max())) < 3:
        heights[int(rng.integers(len(heights)))] = 3
    ys = np.r_[0,np.cumsum(heights+1)]
    xs = np.r_[0,np.cumsum(widths+1)]
    labels = np.full((int(ys[-1])+1,int(xs[-1])+1),-1,dtype=int)
    for r,c in np.argwhere(plan >= 0):
        i = plan[r,c]
        labels[ys[r]+1:ys[r+1],xs[c]+1:xs[c+1]] = i
        if r and plan[r-1,c] == i:
            labels[ys[r],xs[c]+1:xs[c+1]] = i
        if c and plan[r,c-1] == i:
            labels[ys[r]+1:ys[r+1],xs[c]] = i
        if r and c and np.all(plan[r-1:r+1,c-1:c+1] == i):
            labels[ys[r],xs[c]] = i
    for i in sorted(set(plan.flat)-{-1}):
        p = np.argwhere(labels == i)
        if detail and len(p) >= 3 and rng.random() < .5:
            r,c,R,C = _bbox(p)
            y,x = [(r,c),(r,C-1),(R-1,c),(R-1,C-1)][int(rng.integers(4))]
            trial = labels == i
            trial[y,x] = False
            if labels[y,x] == i and len(_components(trial)) == 1:
                labels[y,x] = -1
    occupied = labels >= 0
    near = np.zeros(labels.shape,dtype=bool)
    padded = np.pad(occupied,1)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr+labels.shape[0],dc:dc+labels.shape[1]]
    labels[(labels < 0) & near] = -2
    r,c,R,C = _bbox(np.argwhere(labels != -1))
    return labels[r:R,c:C]


def _optics(rng, shape, center, fallback=False):
    h,w = shape
    rear = 1 if fallback else int(rng.integers(0,3))
    period = 2 if fallback else int(rng.integers(1,3))
    phase = 0 if fallback else int(rng.integers(period))
    back = 1 if fallback else int(rng.integers(1,4))
    initial = (1,1) if fallback else tuple(map(int,rng.integers(1,3,size=2)))
    profiles = []
    for half in initial:
        prefix = [half]*back
        if back == 3 and rng.random() < .7:
            prefix[1] = min(3,half+1)
        tail = [half+(t+phase)//period for t in range(1,2*period+3)]
        profiles.append(prefix+tail)
    mouth = rear+len(profiles[0])
    wall = np.zeros(shape,dtype=bool)
    footprint = np.zeros(shape,dtype=bool)
    wall[center-initial[0]:center+initial[1]+1,rear] = True
    for j,(upper,lower) in enumerate(zip(*profiles),rear+1):
        if center-upper < 0 or center+lower >= h:
            raise AssertionError('spotlight clipped before mouth')
        wall[center-upper,j] = wall[center+lower,j] = True
        footprint[center-upper+1:center+lower,j] = True
    for x in range(mouth+1,w):
        t = x-rear-back
        lo = center-initial[0]-(t+phase)//period
        hi = center+initial[1]+(t+phase)//period
        footprint[max(0,lo+1):min(h,hi),x] = True
    return wall,footprint,rear,mouth


def _empty_neighborhood(grid, cells, bg):
    for y,x in cells:
        if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
            return False
        if np.any(grid[max(0,y-1):y+2,max(0,x-1):x+2] != bg):
            return False
    return True


def _construct(rng, style, rotation, richer=False):
    palette = list(map(int,rng.permutation(10)))
    bg,wall,sensor,light = palette[:4]
    count = int(rng.integers(3,8))
    plan = _plan(rng,count)
    d = _geometry(rng,plan,2,True)
    t = _geometry(rng,plan,3 if richer else 2,True)
    assert max(d.shape) <= 13 and max(t.shape) <= 17
    donor_y,donor_x = tuple(map(int,rng.integers(0,3,size=2)))
    center = donor_y+d.shape[0]+int(rng.integers(6,9))
    center = min(center,23)
    receiver_y = max(1,center-t.shape[0]//2 + int(rng.integers(-2,3)))
    height = max(24,receiver_y+t.shape[0]+1,center+7)
    assert height <= 30
    height = int(rng.integers(height,31))
    # Start with maximum width; trim only after the source and receiver fit.
    emitter,ideal,rear,mouth = _optics(rng,(height,30),center)
    xmax = min(mouth+7,29-t.shape[1])
    assert xmax >= mouth+3
    receiver_x = int(rng.integers(mouth+3,xmax+1))
    width = int(rng.integers(max(24,receiver_x+t.shape[1]+1),31))
    emitter,ideal = emitter[:,:width],ideal[:,:width]
    grid = np.full((height,width),bg,dtype=np.int8)
    grid[emitter] = wall
    donor_mask = np.zeros(grid.shape,dtype=bool)
    donor_mask[donor_y:donor_y+d.shape[0],donor_x:donor_x+d.shape[1]] = d != -1
    assert not np.any(donor_mask & (ideal | emitter))
    receiver_ids = np.full(grid.shape,-1,dtype=int)
    receiver_ids[receiver_y:receiver_y+t.shape[0],receiver_x:receiver_x+t.shape[1]] = t
    target_wall = receiver_ids == -2
    grid[target_wall] = wall
    # Candidate sensors lie in real wall cells adjacent to precisely one room.
    # Visibility is computed from the unmarked receiver wall, before any colors
    # or switches are attached to it.
    visible,hidden = {i:[] for i in range(count)},{i:[] for i in range(count)}
    for y,x in np.argwhere(target_wall):
        adjacent = {int(receiver_ids[y+dy,x+dx]) for dy,dx in _FOUR
                    if 0 <= y+dy < height and 0 <= x+dx < width and receiver_ids[y+dy,x+dx] >= 0}
        if len(adjacent) != 1:
            continue
        i = adjacent.pop()
        lit = ideal[y,x] and not np.any(target_wall[y,mouth+1:x] & ideal[y,mouth+1:x])
        (visible if lit else hidden)[i].append((int(y),int(x)))
    possible = [i for i in range(count) if visible[i]]
    assert len(possible) >= 2
    rng.shuffle(possible)
    desired = set(possible[:2])
    sensors = {}
    for i in range(count):
        options = visible[i] if i in desired else hidden[i]
        if not options:
            options = visible[i] or hidden[i]
        assert options
        sensors[i] = options[int(rng.integers(len(options)))]
        grid[sensors[i]] = sensor
    active_before = {i for i,p in sensors.items() if p in visible[i]}
    assert 2 <= len(active_before) < count
    assert any(ideal[p] for i,p in sensors.items() if i not in active_before)
    ncolors = int(rng.integers(2,min(count,5)+1))
    fills = palette[4:4+ncolors]
    colors = [fills[int(rng.integers(ncolors))] for _ in range(count)]
    # The clean demonstration contains two differently colored live rooms.
    live = sorted(active_before)
    colors[live[0]],colors[live[1]] = fills[:2]
    donor_view = grid[donor_y:donor_y+d.shape[0],donor_x:donor_x+d.shape[1]]
    donor_view[d == -2] = wall
    for i,color in enumerate(colors):
        donor_view[d == i] = color
    # A connected, sometimes off-centre source patch. Source and room colors
    # are separate visible roles, so patch selection never needs a color table.
    patch_x = rear+1
    patch_y = center
    grid[patch_y,patch_x] = light
    if rng.random() < .5:
        grid[patch_y,patch_x+1] = light
        if rng.random() < .5 and ideal[patch_y+1,patch_x+1]:
            grid[patch_y+1,patch_x+1] = light
    obstacle_color = palette[-1]
    if style != 'clean':
        candidates = []
        shapes = (((0,0),),) if style == 'single' else (
            ((0,0),(1,0)),((0,0),(0,1),(1,0),(1,1)),((0,0),(1,0),(1,1)))
        for i in sorted(active_before):
            sy,sx = sensors[i]
            protected = {sensors[j][0] for j in active_before if j != i}
            for x in range(mouth+1,sx-1):
                if not ideal[sy,x]:
                    continue
                for offsets in shapes:
                    cells = [(sy+dy,x+dx) for dy,dx in offsets]
                    if any(y in protected for y,_ in cells):
                        continue
                    if _empty_neighborhood(grid,cells,bg):
                        candidates.append(cells)
        assert candidates, 'no upstream shutter position'
        cells = candidates[int(rng.integers(len(candidates)))]
        for p in cells:
            grid[p] = obstacle_color
        # Repeated blockers in a lane do not widen its shadow.
        if style == 'group' and rng.random() < .5:
            y,x = cells[0]
            repeat = (y,x+3)
            if repeat[1] < receiver_x-1 and _empty_neighborhood(grid,[repeat],bg):
                grid[repeat] = obstacle_color
    # An inert object can have exactly the same color as an effective shutter.
    options = [(int(y),int(x)) for y,x in np.argwhere(~ideal & (grid == bg))
               if _empty_neighborhood(grid,[(int(y),int(x))],bg)
               and x > mouth]
    if options and (style == 'clean' or rng.random() < .7):
        grid[options[int(rng.integers(len(options)))]] = obstacle_color
    # Construction labels form the answer; solve's independent segmentation and
    # lane scan must subsequently agree. All optical effects use the input state.
    shadow = np.zeros(grid.shape,dtype=bool)
    for y,x in np.argwhere(ideal & (grid != bg)):
        if x > mouth:
            shadow[y,x+1:] = True
    out = grid.copy()
    out[ideal & (grid == bg) & ~shadow] = light
    active = {i for i,p in sensors.items() if ideal[p] and not shadow[p]}
    assert active
    if style != 'clean':
        assert active_before-active, 'shutter must change room activation'
    for i in active:
        out[receiver_ids == i] = colors[i]
    pair = {k:np.rot90(v,rotation).copy() for k,v in (('input',grid),('output',out))}
    info = _analyze(pair['input'])
    pred,_ = _scan(info)
    assert np.array_equal(np.rot90(pred,info['rotation']),pair['output'])
    return pair


def _fallback_pair(rng, style, rotation):
    """A guaranteed fitting four-room witness, with no rejection conditions."""
    p = list(map(int,rng.permutation(10)))
    bg,wall,sensor,light = p[:4]
    colors = [p[4],p[int(rng.integers(4,7))],p[5],p[int(rng.integers(4,7))]]
    grid = np.full((28,28),bg,dtype=np.int8)
    emitter,ideal,_,mouth = _optics(rng,grid.shape,17,fallback=True)
    grid[emitter] = wall
    grid[17,2] = light
    donor = np.full((7,7),-2,dtype=int)
    target = np.full((9,8),-2,dtype=int)
    for i,(r,c) in enumerate(((0,0),(0,1),(1,0),(1,1))):
        donor[1+3*r:3+3*r,1+3*c:3+3*c] = i
        target[1+4*r:4+4*r,1+4*c:(4 if c == 0 else 3)+4*c] = i
    dv = grid[1:8,1:8]
    dv[donor == -2] = wall
    for i,color in enumerate(colors):
        dv[donor == i] = color
    tv = grid[13:22,15:23]
    tv[target == -2] = wall
    positions = ((15,15),(15,22),(19,15),(19,22))
    for q in positions:
        grid[q] = sensor
    if style != 'clean':
        grid[15,12] = p[9]
        if style == 'group':
            grid[16,12] = p[9]
    grid[0,25] = p[9]  # Same-color inert distractor, outside the cone.
    shadow = np.zeros(grid.shape,dtype=bool)
    for y,x in np.argwhere(ideal & (grid != bg)):
        if x > mouth:
            shadow[y,x+1:] = True
    out = grid.copy()
    out[ideal & (grid == bg) & ~shadow] = light
    ov = out[13:22,15:23]
    for i,q in enumerate(positions):
        if ideal[q] and not shadow[q]:
            ov[target == i] = colors[i]
    return {k:np.rot90(v,rotation).copy() for k,v in (('input',grid),('output',out))}


def _distinct_fallback(rng,style,rotation,seen):
    pair = _fallback_pair(rng,style,rotation)
    if _key(pair['input']) not in seen:
        return pair
    # At most 13 earlier inputs can exist; 90 ordered background/wall choices
    # guarantee an unused literal role assignment without another random retry.
    info = _analyze(pair['input'])
    old = [info['background'],info['wall']]
    old += [c for c in range(10) if c not in old]
    for bg in range(10):
        for wall in range(10):
            if bg == wall:
                continue
            new = [bg,wall]+[c for c in range(10) if c not in (bg,wall)]
            recolor = np.empty(10,dtype=np.int8)
            recolor[old] = new
            candidate = {k:recolor[v] for k,v in pair.items()}
            if _key(candidate['input']) not in seen:
                return candidate
    raise AssertionError('unreachable: fewer than 90 earlier inputs')


def _integer(value, name, low, high=None):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f'{name} out of range')
    return value


def _key(a):
    a = np.asarray(a,dtype=np.int8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed,'seed',0)
    num_train = _integer(num_train,'num_train',2,10)
    num_test = _integer(num_test,'num_test',1,4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))
    preferred = int(rng.integers(4))
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            if split == 'train':
                style = 'clean' if index == 0 else ('single' if index == 1 else str(rng.choice(['clean','single'])))
                rotation = (preferred+index%2)%4 if shift == 0 else int(rng.integers(4))
                richer = False
            else:
                style = 'group' if shift == 1 else str(rng.choice(['clean','single']))
                rotation = (preferred+2+index%2)%4 if shift == 0 else int(rng.integers(4))
                richer = shift == 2
            for attempt in range(64):
                try:
                    pair = _construct(rng,style,rotation,richer)
                    key = _key(pair['input'])
                    if key in seen:
                        continue
                    info = measure_pair(pair)
                    if split == 'train' and index == 0 and info['active_colors'] < 2:
                        continue
                    if split == 'train' and index == 0 and not info['deformed_frameworks']:
                        continue
                    if split == 'train' and index == 1 and not info['shutter_changes_activation']:
                        continue
                    break
                except AssertionError:
                    continue
            else:
                pair = _distinct_fallback(rng,style,rotation,seen)
                key = _key(pair['input'])
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    return puzzle


def measure_pair(pair):
    info = _analyze(np.asarray(pair['input']))
    a = info['canonical']
    out,active = _scan(info)
    assert np.array_equal(np.rot90(out,info['rotation']),np.asarray(pair['output'])), 'wrong output'
    _,unshuttered = _scan(info,ignore_extras=True)
    n = len(info['room_colors'])
    in_cone = {int(info['sensor_ids'][y,x]) for y,x in np.argwhere(
        (info['sensor_ids'] >= 0) & info['footprint'])}
    foreground = a != info['background']
    assert np.array_equal(out[foreground],a[foreground]), 'existing material changed'
    for i,color in enumerate(info['room_colors']):
        assert np.all(out[info['room_ids'] == i] == (color if i in active else info['background']))
    assert active and len(active) < n
    assert np.any((out == info['light']) & (a == info['background']))
    # This is a necessary-contribution check, not merely solve/label agreement:
    # omitting transfer leaves an active room blank; ungated transfer fills a
    # dark room; cone membership alone can wrongly activate a shadowed sensor.
    room_pixels = int(np.count_nonzero(np.isin(info['room_ids'],list(active))))
    assert room_pixels > 0
    room_mask = info['room_ids'] >= 0
    wall_mask = info['receiver_mask']
    r,c,R,C = _bbox(np.argwhere(wall_mask))
    normalized = np.stack((room_mask[r:R,c:C],wall_mask[r:R,c:C],
                           info['sensor_ids'][r:R,c:C] >= 0),axis=0)
    y,x,Y,X = _bbox(np.argwhere(info['donor_mask']))
    donor_framework = (a == info['wall'])[y:Y,x:X]
    receiver_framework = wall_mask[r:R,c:C]
    extra = foreground & ~info['receiver_mask'] & ~info['donor_mask'] & ~info['chamber'] & (a != info['wall'])
    effective = extra & info['footprint']
    return dict(rooms=n,active=len(active),active_colors=len({info['room_colors'][i] for i in active}),
                hidden_in_cone=len(in_cone-active),shutter_changes_activation=bool(unshuttered-active),
                effective_blocker_cells=int(effective.sum()),inert_blocker_cells=int((extra & ~info['footprint']).sum()),
                filled_room_cells=room_pixels,direction=info['rotation'],shape=a.shape,
                role_palette=(info['background'],info['wall'],info['sensor'],info['light']),
                periods=tuple(f[0] for f in info['fits']),
                structure=hashlib.sha256(normalized.tobytes()+bytes(normalized.shape)).hexdigest(),
                deformed_frameworks=not np.array_equal(donor_framework,receiver_framework),
                donor_room_areas=tuple(map(len,info['donor_rooms'])),
                spatial_profile=tuple(sorted(tuple(sorted(row)) for row in _relations(_rooms(wall_mask)))),
                room_areas=tuple(int(np.count_nonzero(info['room_ids'] == i)) for i in range(n)),
                nonrectangular_rooms=sum(len(p) != (_bbox(p)[2]-_bbox(p)[0])*(_bbox(p)[3]-_bbox(p)[1])
                                         for p in _rooms(info['receiver_mask'])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen,training = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            for value in pair.values():
                a = np.asarray(value)
                assert a.ndim == 2 and np.issubdtype(a.dtype,np.integer)
                assert all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9))
            assert np.asarray(pair['input']).shape == np.asarray(pair['output']).shape
            key = _key(pair['input'])
            assert key not in seen
            seen.add(key)
            m = measure_pair(pair)
            if split == 'train':
                training.append(m)
    assert any(m['active_colors'] >= 2 for m in training), 'teach literal multicolor transfer'
    assert any(m['deformed_frameworks'] for m in training), 'teach correspondence across deformation'
    assert any(m['hidden_in_cone'] > 0 for m in training), 'teach visibility rather than cone membership'
    assert any(m['shutter_changes_activation'] for m in training), 'teach an obstacle suppressing a room'
    assert any(m['effective_blocker_cells'] == 0 for m in training), 'teach unobstructed projection'


def self_test():
    """42 specified seeds, streamed metrics, metamorphic checks and corruption.

    Returns small measurements for an optional report; retains no grids or logs.
    """
    structures,shapes,palettes,rooms,directions,periods = set(),set(),set(),set(),set(),set()
    totals = Counter()
    for levels,next_level in (((6,5,4),3),((6,6,5,5,4,4),3),((6,5,5,4,4,3),3)):
        xs = np.arange(1,len(levels)+1)
        k,p,e,v,_ = _fit_flank(xs,np.array(levels),-1)
        assert v-((e+1-e+p)//k) == next_level
    # Forced fallback checks also serve as coordinate-level optical fixtures:
    # its west sensors are live and its east sensors are hidden behind walls.
    # A shutter at row 15 suppresses only the upper-left room.
    for seed in range(8):
        for style in ('clean','single','group'):
            pair = _fallback_pair(np.random.default_rng(seed),style,seed%4)
            info = _analyze(pair['input'])
            out,active = _scan(info)
            ids = info['sensor_ids']
            expected = {int(ids[19,15])}
            if style == 'clean':
                expected.add(int(ids[15,15]))
            assert active == expected
            assert out[15,22] == info['sensor']
            assert np.array_equal(solve(pair['input']),pair['output'])
            m = measure_pair(pair)
            assert m['shutter_changes_activation'] == (style != 'clean')
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        repeat = generate_puzzle(seed)
        assert all(_key(p[k]) == _key(q[k]) for s in puzzle
                   for p,q in zip(puzzle[s],repeat[s]) for k in ('input','output'))
        for split in ('train','test'):
            for pair in puzzle[split]:
                m = measure_pair(pair)
                structures.add(m['structure'])
                shapes.add(m['shape'])
                palettes.add(m['role_palette'])
                rooms.add(m['rooms'])
                directions.add(m['direction'])
                periods.update(m['periods'])
                totals['pairs'] += 1
                totals['hidden_sensors'] += m['hidden_in_cone']
                totals['shutter_witnesses'] += m['shutter_changes_activation']
                totals['nonrectangular_rooms'] += m['nonrectangular_rooms']
                # Color permutation, including changed background and markers,
                # must commute with solve; a quarter-turn must also commute.
                if split == 'test':
                    recolor = np.array([3,8,1,6,0,9,4,2,7,5],dtype=np.int8)
                    assert np.array_equal(solve(recolor[pair['input']]),recolor[pair['output']])
                    assert np.array_equal(solve(np.rot90(pair['input'])),np.rot90(pair['output']))
        # Changing a live atlas room changes its receiver, not light or shadows.
        pair = next(p for p in puzzle['train'] if measure_pair(p)['active_colors'] >= 2)
        info = _analyze(pair['input'])
        _,active = _scan(info)
        live_color = info['room_colors'][min(active)]
        other_color = next(c for c in info['donor_colors'] if c != live_color)
        mutant = pair['input'].copy()
        for p,c in zip(info['donor_rooms'],info['donor_colors']):
            if c == live_color:
                mutant[tuple(p.T)] = other_color
            elif c == other_color:
                mutant[tuple(p.T)] = live_color
        changed = solve(mutant) != pair['output']
        assert np.any(changed)
        receiver = np.rot90(info['room_ids'] >= 0,info['rotation'])
        donor = np.rot90(info['donor_mask'],info['rotation'])
        assert np.any(changed & receiver)
        assert not np.any(changed & ~(receiver | donor))
    assert len(structures) >= 100 and len(shapes) >= 15 and len(palettes) >= 80
    assert rooms == set(range(3,8)) and directions == set(range(4)) and periods == {1,2}
    assert totals['shutter_witnesses'] >= 42 and totals['nonrectangular_rooms'] >= 10
    for train in range(2,11):
        for test in range(1,min(4,train-1)+1):
            p = generate_puzzle(73,train,test)
            validate_generated_puzzle(p)
            assert (len(p['train']),len(p['test'])) == (train,test)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    for seed in (-1,True,1.5,None):
        try:
            generate_puzzle(seed)
        except ValueError:
            continue
        raise AssertionError('illegal seed accepted')
    p = generate_puzzle(80,2,1)
    q = p['test'][0]
    y,x = np.argwhere(q['input'] != q['output'])[0]
    q['output'][y,x] = q['input'][y,x]
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42,**totals,distinct_receiver_structures=len(structures),
                canonical_shapes=len(shapes),role_palettes=len(palettes),room_counts=sorted(rooms),
                directions=sorted(directions),periods=sorted(periods))
