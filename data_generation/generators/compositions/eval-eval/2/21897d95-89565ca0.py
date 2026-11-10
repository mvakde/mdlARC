"""Count-powered room waves. No parent code or generation state is used by solve.

Fixed roles: 0 exterior, 1 T ink, 2 removable noise/wall occlusion.
Colors 3..9 are visibly inferred wall colors and reservoir/payload colors.
"""
from collections import deque
import json
import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("Require integer counts 2..10 train, 1..4 test, and train > test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    _need(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "invalid grid dimensions")
    _need(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= grid) & (grid <= 9))), "invalid colors")
    return grid.astype(np.int8, copy=True)


def _add(p, d, amount=1):
    return p[0] + amount*d[0], p[1] + amount*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _walls(shape, rectangles):
    mask = np.zeros(shape, dtype=bool)
    for t, l, b, r in rectangles:
        mask[t, l:r+1] = mask[b, l:r+1] = True
        mask[t:b+1, l] = mask[t:b+1, r] = True
    return mask


def _rooms(mask, box):
    """Flood-fill positive-area elementary interiors; outer frame stays closed."""
    t, l, b, r = box
    labels = np.full(mask.shape, -1, dtype=np.int16)
    rooms = []
    for y in range(t+1, b):
        for x in range(l+1, r):
            if mask[y, x] or labels[y, x] >= 0:
                continue
            index = len(rooms)
            stack, points = [(y, x)], []
            labels[y, x] = index
            while stack:
                p = stack.pop()
                points.append(p)
                for d in DIRECTIONS:
                    q = _add(p, d)
                    if t < q[0] < b and l < q[1] < r and not mask[q] and labels[q] < 0:
                        labels[q] = index
                        stack.append(q)
            arr = np.array(points)
            lo, hi = arr.min(axis=0), arr.max(axis=0)
            _need(len(points) == int(np.prod(hi-lo+1)), "nonrectangular or open room")
            rooms.append((int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])))
    _need(bool(rooms), "empty frame")
    expected = _walls(mask.shape, [(t-1, l-1, b+1, r+1) for t, l, b, r in rooms])
    _need(np.array_equal(expected, mask), "walls must exactly delimit elementary rooms")
    return rooms, labels


def _adjacency(labels, mask, number):
    graph = [set() for _ in range(number)]
    for y, x in np.argwhere(mask):
        p = int(y), int(x)
        for d in ((1, 0), (0, 1)):
            a, b = _add(p, d), _add(p, d, -1)
            if _inside(a, labels.shape) and _inside(b, labels.shape):
                i, j = int(labels[a]), int(labels[b])
                if i >= 0 and j >= 0 and i != j:
                    graph[i].add(j)
                    graph[j].add(i)
    return graph


def _ball(graph, entry, strength):
    distances, queue = {entry: 0}, deque([entry])
    while queue:
        node = queue.popleft()
        if distances[node] + 1 >= strength:
            continue
        for other in graph[node]:
            if other not in distances:
                distances[other] = distances[node] + 1
                queue.append(other)
    return set(distances)


def _recover_frame(grid, color):
    visible = grid == color
    positions = np.argwhere(visible)
    if len(positions) < 8:
        return None
    # Wall unions with positive-area rooms have no filled 2x2 block. Each
    # reservoir's original fill must retain one, making roles input-visible.
    if np.any(visible[:-1, :-1] & visible[1:, :-1] & visible[:-1, 1:] & visible[1:, 1:]):
        return None
    t, l = map(int, positions.min(axis=0))
    b, r = map(int, positions.max(axis=0))
    if b-t < 2 or r-l < 2 or any(grid[p] != color for p in ((t,l), (t,r), (b,l), (b,r))):
        return None
    perimeter = _walls(grid.shape, [(t,l,b,r)])
    if not np.all(np.isin(grid[perimeter], (color, 2))):
        return None
    mask = visible.copy()
    for yy, xx in np.argwhere((grid == 2) & (np.indices(grid.shape)[0] >= t)
                             & (np.indices(grid.shape)[0] <= b)
                             & (np.indices(grid.shape)[1] >= l)
                             & (np.indices(grid.shape)[1] <= r)):
        p = int(yy), int(xx)
        for d in ((1,0), (0,1)):
            a, z = _add(p,d), _add(p,d,-1)
            if _inside(a,grid.shape) and _inside(z,grid.shape) and grid[a] == grid[z] == color:
                mask[p] = True
    if not mask[perimeter].all():
        return None
    try:
        rooms, labels = _rooms(mask, (t,l,b,r))
    except AssertionError:
        return None
    return dict(color=color, box=(t,l,b,r), mask=mask, rooms=rooms, labels=labels,
                graph=_adjacency(labels,mask,len(rooms)))


def _glyph(center, direction, shape):
    perpendicular = (-direction[1], direction[0])
    tip = _add(center, direction)
    arms = (_add(center, perpendicular), _add(center, perpendicular, -1))
    write = {center, tip, *arms}
    if not all(_inside(p, shape) for p in write):
        return None
    return dict(center=center, direction=direction, tip=tip, arms=arms, write=write)


def _candidate_arrows(frames, shape):
    """Use only geometry; target is the first frame struck beyond the source."""
    ownership = np.full(shape, -1, dtype=np.int8)
    wall_owner = np.full(shape, -1, dtype=np.int8)
    for i, frame in enumerate(frames):
        ownership[frame['labels'] >= 0] = i
        wall_owner[frame['mask']] = i
    arrows = []
    for source, frame in enumerate(frames):
        t,l,b,r = frame['box']
        for direction in DIRECTIONS:
            centers = ([(t+2,x) for x in range(l+1,r)] if direction == (-1,0) else
                       [(y,r-2) for y in range(t+1,b)] if direction == (0,1) else
                       [(b-2,x) for x in range(l+1,r)] if direction == (1,0) else
                       [(y,l+2) for y in range(t+1,b)])
            for center in centers:
                geo = _glyph(center, direction, shape)
                if geo is None:
                    continue
                room = int(frame['labels'][center])
                if room < 0 or any(frame['labels'][p] != room for p in geo['write']):
                    continue
                p = _add(center,direction,2)
                if not _inside(p,shape) or wall_owner[p] != source:
                    continue
                p = _add(p,direction)
                while _inside(p,shape) and wall_owner[p] < 0 and ownership[p] < 0:
                    p = _add(p,direction)
                if not _inside(p,shape) or wall_owner[p] < 0 or wall_owner[p] == source:
                    continue
                target = int(wall_owner[p])
                entry_cell = _add(p,direction)
                if not _inside(entry_cell,shape) or ownership[entry_cell] != target:
                    continue
                entry = int(frames[target]['labels'][entry_cell])
                arrows.append(dict(geo,source=source,target=target,source_room=room,
                                   entry=entry,entry_cell=entry_cell))
    return arrows


def _analyze(value):
    """Derive frames, original fills, arrows and wave graphs from pixels only."""
    grid = _grid(value)
    frames = []
    for color in range(3,10):
        frame = _recover_frame(grid,color)
        if frame is not None:
            frames.append(frame)
    _need(2 <= len(frames) <= 4, "need two to four identifiable frames")
    frames.sort(key=lambda f: (f['box'][0],f['box'][1]))
    wall_colors = {f['color'] for f in frames}
    for i, f in enumerate(frames):
        t,l,b,r = f['box']
        for g in frames[:i]:
            u,v,w,z = g['box']
            _need(b+1 < u or w+1 < t or r+1 < v or z+1 < l, "frames need an exterior gap")
        values = grid[f['labels'] >= 0]
        candidates = [int(c) for c in values if c >= 3 and c not in wall_colors]
        _need(bool(candidates), "frame lacks fill evidence")
        hist = np.bincount(candidates,minlength=10)
        fill = int(hist.argmax())
        _need(np.count_nonzero(hist == hist[fill]) == 1, "ambiguous original fill")
        f['fill'] = fill
    geometries = _candidate_arrows(frames,grid.shape)
    arrows, used = [], set()
    for a in geometries:
        if not all(grid[p] == 1 for p in (*a['arms'],a['tip'])):
            continue
        c = int(grid[a['center']])
        if c != 1 and (c < 3 or c in wall_colors):
            continue
        _need(not used & a['write'], "overlapping or ambiguous T annotations")
        used.update(a['write'])
        arrows.append(dict(a, explicit=c != 1, payload=c if c != 1 else frames[a['source']]['fill']))
    _need(bool(arrows), "no arrow instruction")
    ink = set(map(tuple,np.argwhere(grid == 1)))
    _need(ink == {p for p in used if grid[p] == 1}, "unexplained T ink")
    _need(len({a['target'] for a in arrows}) == len(arrows), "multiple incoming arrows")
    _need(all(_separate(a,b) for i,a in enumerate(arrows) for b in arrows[:i]),
          'T stencils must remain separated')
    clean = np.zeros_like(grid)
    for f in frames:
        interior = f['labels'] >= 0
        clean[interior] = f['fill']
        clean[f['mask']] = f['color']
        for p in map(tuple,np.argwhere(interior)):
            _need(p in used or grid[p] == f['fill'], "interior is not originally monochrome")
        evidence = (grid == f['fill']) & interior
        _need(bool(np.any(evidence[:-1,:-1] & evidence[1:,:-1]
                          & evidence[:-1,1:] & evidence[1:,1:])), "fill lacks solid evidence")
    _need(all(grid[p] in (0,2) for p in map(tuple,np.argwhere(clean == 0))), "unexplained exterior cells")
    for a in arrows:
        a['strength'] = len(frames[a['source']]['rooms'])
        a['reached'] = _ball(frames[a['target']]['graph'],a['entry'],a['strength'])
    return dict(grid=grid,clean=clean,frames=frames,arrows=arrows)


def _paint(info, mode='wave'):
    output = info['clean'].copy()
    for a in info['arrows']:
        frame = info['frames'][a['target']]
        reached = ({a['entry']} if mode == 'entry' else set(range(len(frame['rooms'])))
                   if mode == 'whole' else a['reached'])
        output[np.isin(frame['labels'],list(reached))] = a['payload']
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Snapshot color waves; has no seed, training examples or latent plan."""
    return _paint(_analyze(input_grid))


def _capacity(box):
    t,l,b,r = box
    return ((b-t)//2)*((r-l)//2)


def _partition(rng, box, count, style):
    rooms = [box]
    first_axis = None
    for step in range(count-1):
        options = []
        for i,(t,l,b,r) in enumerate(rooms):
            for axis,lo,hi in ((0,t,b),(1,l,r)):
                if style == 'strip' and first_axis is not None and axis != first_axis:
                    continue
                if style == 'partial' and step == 1 and axis == first_axis:
                    continue
                for cut in range(lo+2,hi-1):
                    halves = ((t,l,cut,r),(cut,l,b,r)) if axis == 0 else ((t,l,b,cut),(t,cut,b,r))
                    remaining_capacity = sum(_capacity(z) for j,z in enumerate(rooms) if j != i)
                    if remaining_capacity + sum(_capacity(z) for z in halves) >= count:
                        options.append((i,axis,halves))
        if not options:
            return None
        i,axis,halves = options[int(rng.integers(len(options)))]
        first_axis = axis if step == 0 else first_axis
        rooms[i:i+1] = halves
    return rooms


def _slots(rng, shape, number, style):
    h,w = shape
    if style in ('row','column'):
        span = w if style == 'row' else h
        gap = int(rng.integers(1,3))
        if span < 7*number+gap*(number-1):
            return None
        widths = np.full(number,7,dtype=int)
        for _ in range(span-int(widths.sum())-gap*(number-1)):
            widths[int(rng.integers(number))] += 1
        slots, pos = [], 0
        for width in widths:
            slots.append((0,pos,h-1,pos+int(width)-1) if style == 'row'
                         else (pos,0,pos+int(width)-1,w-1))
            pos += int(width)+gap
        return slots
    slots = [(0,0,h-1,w-1)]
    for _ in range(number-1):
        options = []
        for i,(t,l,b,r) in enumerate(slots):
            for axis,lo,hi in ((0,t,b),(1,l,r)):
                for gap in (1,2,3):
                    for cut in range(lo+7,hi-gap-5):
                        halves = ((t,l,cut-1,r),(cut+gap,l,b,r)) if axis == 0 else ((t,l,b,cut-1),(t,cut+gap,b,r))
                        options.append((i,halves))
        if not options:
            return None
        i,halves = options[int(rng.integers(len(options)))]
        slots[i:i+1] = halves
    return slots


def _rect_graph(rectangles):
    """Construction truth: shared interval boundaries, not recovered pixels."""
    graph = [set() for _ in rectangles]
    for i,(t,l,b,r) in enumerate(rectangles):
        for j,(u,v,w,z) in enumerate(rectangles[:i]):
            touch = ((b == u or w == t) and min(r,z)-max(l,v) >= 2
                     or (r == v or z == l) and min(b,w)-max(t,u) >= 2)
            if touch:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _partial(frame):
    t,l,b,r = frame['box']
    mask = frame['mask']
    ys = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:],axis=1))
    xs = np.flatnonzero(np.any(mask[:-1, :] & mask[1:, :],axis=0))
    return any(not mask[y,l:r+1].all() for y in ys) or any(not mask[t:b+1,x].all() for x in xs)


def _lattice_count(frame):
    mask = frame['mask']
    rows = np.count_nonzero(np.any(mask[:, :-1] & mask[:, 1:],axis=1))
    cols = np.count_nonzero(np.any(mask[:-1, :] & mask[1:, :],axis=0))
    return int((rows-1)*(cols-1))


def _separate(a,b):
    halo = a['write'] | {_add(p,d) for p in a['write'] for d in DIRECTIONS}
    return not bool(halo & b['write'])


def _select(rng, pool, frames, kind, desired):
    pool = [pool[int(i)] for i in rng.permutation(len(pool))]
    chosen = []
    if kind == 'entry':
        options = [a for a in pool if a['source'] == 0 and a['target'] == 1]
        if not options:
            return None
        return [options[0]]
    if kind == 'cycle':
        for a in pool:
            if (a['source'],a['target']) != (0,1):
                continue
            reached = _ball(frames[1]['graph'],a['entry'],len(frames[0]['rooms']))
            # The witness distinguishes radius from a quota, and distinguishes
            # actual partial-network counting from multiplying line counts.
            wrong_reached = _ball(frames[1]['graph'],a['entry'],_lattice_count(frames[0]))
            if not len(frames[0]['rooms']) < len(reached) < len(frames[1]['rooms']):
                continue
            if reached == wrong_reached:
                continue
            for b in pool:
                if ((b['source'],b['target']) == (1,0) and b['source_room'] in reached
                        and _separate(a,b)):
                    return [a,b]
        return None
    if kind == 'reach':
        options = [a for a in pool if a['source'] == 0]
        if not options:
            return None
        chosen = [options[0]]
    for a in pool:
        if len(chosen) >= desired:
            break
        if a['target'] in {b['target'] for b in chosen} or any(not _separate(a,b) for b in chosen):
            continue
        chosen.append(a)
    return chosen if len(chosen) == desired else None


def _build(rng, parameters, kind, fallback=False):
    cap = parameters['cap']
    if kind == 'entry':
        number, counts = 2, [1,int(rng.integers(3,cap+1))]
    elif kind == 'cycle':
        number, counts = 3, [3,int(rng.integers(5,cap+1)),int(rng.integers(1,cap+1))]
    elif kind == 'reach':
        number = int(rng.integers(3,5))
        counts = [int(rng.integers(cap+1,cap+3))] + list(map(int,rng.integers(1,cap+3,size=number-1)))
    else:
        number = 4 if kind == 'routing' else int(rng.integers(2,4))
        counts = list(map(int,rng.integers(1,cap+1,size=number)))
    if fallback:
        shape = (30,30)
        slots = ([(0,0,29,13),(0,16,29,29)] if number == 2 else
                 [(0,0,29,8),(0,10,29,18),(0,20,29,29)] if number == 3 else
                 [(0,0,13,13),(0,16,13,29),(16,0,29,13),(16,16,29,29)])
    else:
        low = 13 if number == 2 else 20 if number == 3 else 23
        shape = tuple(map(int,rng.integers(low,31,size=2)))
        style = str(rng.choice(['row','column','slicing','slicing'])) if number < 4 else 'slicing'
        slots = _slots(rng,shape,number,style)
        if slots is None:
            return None
    colors = list(map(int,rng.choice(np.arange(3,10),size=number,replace=False)))
    fills = [c for c in range(3,10) if c not in colors]
    chosen_fills = list(map(int,rng.choice(fills,size=number,replace=True)))
    # Guaranteed visible payload change in the two curriculum witnesses.
    if kind in ('entry','cycle'):
        chosen_fills[1] = int(rng.choice([c for c in fills if c != chosen_fills[0]]))
    if kind == 'cycle':
        # A disconnected, same-filled reservoir stays untouched while the
        # donor changes, so color-wide substitution cannot explain training.
        chosen_fills[2] = chosen_fills[0]
    frames, clean = [], np.zeros(shape,dtype=np.int8)
    for i,(slot,count,color,fill) in enumerate(zip(slots,counts,colors,chosen_fills)):
        t,l,b,r = slot
        if not fallback:
            bh,bw = int(rng.integers(7,b-t+2)),int(rng.integers(7,r-l+2))
            t,l = int(rng.integers(t,b-bh+2)),int(rng.integers(l,r-bw+2))
            b,r = t+bh-1,l+bw-1
        box = t,l,b,r
        if _capacity(box) < count:
            return None
        style = ('partial' if (kind == 'entry' and i == 1 or kind == 'cycle' and i == 0)
                 else str(rng.choice(['strip','partial','mixed','mixed'])))
        if fallback and kind == 'cycle' and i == 0:
            # Finite safety witness: three rooms rather than the four that
            # completing its partial divider would invent.
            middle_y,middle_x = (t+b)//2,(l+r)//2
            rectangles = [(t,l,middle_y,r),(middle_y,l,b,middle_x),(middle_y,middle_x,b,r)]
        elif fallback and kind == 'cycle' and i == 1:
            # A long chain ends in a split room. Starting in one end half
            # reaches four rooms at radius two, but one more at radius three.
            ys = [t+(b-t)*j//(count-1) for j in range(count)]
            rectangles = [(ys[j],l,ys[j+1],r) for j in range(count-2)]
            middle_x = (l+r)//2
            rectangles.extend([(ys[-2],l,b,middle_x),(ys[-2],middle_x,b,r)])
        else:
            rectangles = _partition(rng,box,count,style)
        if rectangles is None:
            return None
        mask = _walls(shape,rectangles)
        labels = np.full(shape,-1,dtype=np.int16)
        for j,(u,v,w,z) in enumerate(rectangles):
            labels[u+1:w,v+1:z] = j
        frames.append(dict(color=color,fill=fill,box=box,mask=mask,labels=labels,
                           rooms=rectangles,graph=_rect_graph(rectangles)))
        clean[mask] = color
        clean[labels >= 0] = fill
    if kind == 'entry' and not _partial(frames[1]):
        return None
    pool = _candidate_arrows(frames,shape)
    desired = (3 if kind == 'routing' else 2 if kind in ('reach','cycle')
               else 1 if kind == 'entry' else int(rng.integers(1,number+1)))
    plan = _select(rng,pool,frames,kind,desired)
    if plan is None:
        return None
    grid, output = clean.copy(),clean.copy()
    for index,a in enumerate(plan):
        explicit = (parameters['payload'] and kind == 'cycle' and index == 1
                    or parameters['payload'] and kind not in ('entry','cycle') and rng.random() < .35)
        if explicit:
            choices = [c for c in fills if c != frames[a['source']]['fill'] and c != frames[a['target']]['fill']]
            if not choices:
                return None
            payload = int(rng.choice(choices))
        else:
            payload = frames[a['source']]['fill']
        for p in a['write']:
            grid[p] = 1
        if explicit:
            grid[a['center']] = payload
        frame = frames[a['target']]
        for room in _ball(frame['graph'],a['entry'],len(frames[a['source']]['rooms'])):
            output[frame['labels'] == room] = payload
    # Only straight, non-junction pixels may be obscured. No adjacent gaps.
    safe = []
    for frame in frames:
        mask = frame['mask']
        for yy,xx in np.argwhere(mask):
            p = int(yy),int(xx)
            neighbors = [d for d in DIRECTIONS if _inside(_add(p,d),shape) and mask[_add(p,d)]]
            if len(neighbors) == 2 and _add(neighbors[0],neighbors[1]) == (0,0):
                safe.append(p)
    rng.shuffle(safe)
    damage = []
    amount = int(rng.integers(1,7)) if kind == 'cycle' else int(rng.integers(0,5))
    for p in safe:
        if len(damage) >= amount:
            break
        if all(abs(p[0]-q[0])+abs(p[1]-q[1]) > 1 for q in damage):
            damage.append(p)
            grid[p] = 2
    sites = np.argwhere(clean == 0)
    if len(sites):
        for y,x in sites[rng.choice(len(sites),size=min(len(sites),int(rng.integers(0,7))),replace=False)]:
            grid[y,x] = 2
    try:
        info = _analyze(grid)
    except AssertionError:
        return None
    if not np.array_equal(_paint(info),output) or not np.array_equal(info['clean'],clean):
        return None
    if kind == 'cycle' and not _features(info)['simultaneous']:
        return None
    if kind in ('reach','routing') and not _features(info)['count_essential']:
        return None
    return dict(input=grid,output=output)


def _features(info):
    frames,arrows = info['frames'],info['arrows']
    wave = _paint(info)
    simultaneous = any(not a['explicit'] and b['target'] == a['source']
                       and a['source_room'] in b['reached'] and b['payload'] != a['payload']
                       for a in arrows for b in arrows if a is not b)
    lattice_output = info['clean'].copy()
    for a in arrows:
        frame = frames[a['target']]
        wrong = _ball(frame['graph'],a['entry'],_lattice_count(frames[a['source']]))
        lattice_output[np.isin(frame['labels'],list(wrong))] = a['payload']
    branching = any(len(a['reached']) > a['strength']
                    and frames[a['target']]['fill'] != a['payload'] for a in arrows)
    final_colors = [set(map(int,wave[f['labels'] >= 0])) for f in frames]
    locality = any(f['fill'] == g['fill'] and final_colors[i] != final_colors[j]
                   for i,f in enumerate(frames) for j,g in enumerate(frames[:i]))
    return dict(frames=len(frames),counts=tuple(len(f['rooms']) for f in frames),arrows=len(arrows),
                explicit=any(a['explicit'] for a in arrows),simultaneous=simultaneous,
                partial=any(_partial(f) for f in frames),
                count_essential=not np.array_equal(wave,_paint(info,'entry')),
                limited=not np.array_equal(wave,_paint(info,'whole')),
                transfer_essential=not np.array_equal(wave,info['clean']),
                damage=any(np.any(f['mask'] & (info['grid'] == 2)) for f in frames),
                lattice_essential=not np.array_equal(wave,lattice_output),branching=branching,
                locality=locality,
                repeated=len({f['fill'] for f in frames}) < len(frames),
                reached=tuple(len(a['reached']) for a in arrows),
                directions={a['direction'] for a in arrows})


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train,num_test = _counts(num_train,num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    parameters = dict(cap=int(rng.integers(5,7)),payload=bool(rng.random() < .6))
    shift = str(rng.choice(['reach','routing']))
    puzzle,seen = dict(train=[],test=[]),set()
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            kind = ('entry' if index == 0 else 'cycle' if index == 1 else 'ordinary') if split == 'train' else shift
            for attempt in range(240):
                pair = _build(rng,parameters,kind,fallback=attempt >= 140)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('bounded scene construction exhausted: '+kind)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, 'wrong puzzle schema')
    _need(all(isinstance(puzzle[s],list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen,train_features,test_features = set(),[],[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            _need(isinstance(pair,dict) and set(pair) == {'input','output'}, 'wrong pair schema')
            inp,out = _grid(pair['input']),_grid(pair['output'])
            _need(inp.shape == out.shape,'output must preserve canvas shape')
            _need(_key(inp) not in seen,'duplicate input')
            seen.add(_key(inp))
            info = _analyze(inp)
            _need(np.array_equal(_paint(info),out),'incorrect wave output')
            for f in info['frames']:
                _need(np.array_equal(out[f['mask']],info['clean'][f['mask']]),'wall changed')
                _need(len(_ball(f['graph'],0,len(f['rooms']))) == len(f['rooms']),
                      'disconnected room graph')
                for p in map(tuple,np.argwhere(f['mask'] & (inp == 2))):
                    neighbors = [d for d in DIRECTIONS
                                 if _inside(_add(p,d),inp.shape) and f['mask'][_add(p,d)]]
                    _need(len(neighbors) == 2 and _add(neighbors[0],neighbors[1]) == (0,0)
                          and all(inp[_add(p,d)] == f['color'] for d in neighbors),
                          'wall damage must have immediate straight support')
            _need(not np.any(np.isin(out,(1,2))),'annotations or noise survived')
            features = _features(info)
            (train_features if split == 'train' else test_features).append(features)
    for feature in ('count_essential','limited','transfer_essential','partial','damage','simultaneous',
                    'branching','lattice_essential','locality'):
        _need(any(f[feature] for f in train_features),'training lacks '+feature)
    if any(f['explicit'] for f in test_features):
        _need(any(f['explicit'] for f in train_features),'unshown junction payload')
    # All tests have a real shift: a new emitting count or a larger interacting graph.
    tr_count = max(max(f['counts']) for f in train_features)
    tr_frames = max(f['frames'] for f in train_features)
    for pair,f in zip(puzzle['test'],test_features):
        info = _analyze(pair['input'])
        _need(f['count_essential'] and f['transfer_essential'],'test omits an inherited mechanism')
        _need(any(a['strength'] > tr_count for a in info['arrows']) or f['frames'] > tr_frames,
              'missing train/test structural shift')


def _canonical(grid):
    roles = {}
    return tuple(roles.setdefault(int(c),len(roles)) for c in np.asarray(grid).ravel())


def self_test():
    """Bounded pixel-derived checks over 42 seeds, plus counts and equivariance."""
    shapes,structures,puzzles,counts,directions,topologies = set(),set(),set(),set(),set(),set()
    records,fixtures = [],[]
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.add(tuple(_key(q['input']) for s in ('train','test') for q in p[s]))
        if seed < 2:
            again = generate_puzzle(seed)
            _need(all(_key(a[k]) == _key(b[k]) for s in p for a,b in zip(p[s],again[s])
                      for k in ('input','output')),'nondeterministic seed')
        for split in ('train','test'):
            for pair in p[split]:
                info = _analyze(pair['input'])
                f = _features(info)
                records.append(f)
                shapes.add(pair['input'].shape)
                structures.add((pair['input'].shape,_canonical(info['clean'])))
                counts.update(f['counts'])
                directions.update(f['directions'])
                for frame in info['frames']:
                    # A graph signature removes scale, colors and positions.
                    topologies.add(tuple(sorted(len(g) for g in frame['graph'])))
                _need(np.array_equal(solve(pair['input'].copy()),pair['output']),'input-only solve failure')
                if seed < 2:
                    fixtures.append(pair)
    _need(len(puzzles) == 42 and len(shapes) >= 35 and len(structures) >= 140,'collapsed geometry')
    _need(set(range(1,7)) <= counts and len(topologies) >= 12,'collapsed count/topology variation')
    _need(directions == set(DIRECTIONS),'missing arrow directions')
    for feature in ('partial','count_essential','limited','simultaneous','repeated','explicit','damage',
                    'branching','lattice_essential','locality'):
        _need(sum(f[feature] for f in records) >= 8,'missing family variation: '+feature)
    # Fixed role colors stay fixed; permute only semantic wall/fill colors.
    palette = np.array([0,1,2,7,9,3,8,4,6,5],dtype=np.int8)
    for pair in fixtures:
        for operation in (lambda a: np.rot90(a),lambda a: np.fliplr(a),lambda a: a.T):
            _need(np.array_equal(solve(operation(pair['input'])),operation(pair['output'])),
                  'direction/topology equivariance failure')
        _need(np.array_equal(solve(palette[pair['input']]),palette[pair['output']]),'palette equivariance failure')
    # A count beyond the sampled ceiling: 12-room source, seven-room target.
    # The expected saturated answer is constructed directly, with no _paint.
    extended = np.zeros((30,26),dtype=np.int8)
    source_rects = [(t,l,b,r) for t,b in zip((0,7,14,21),(7,14,21,29))
                    for l,r in ((0,3),(3,6),(6,9))]
    target_rects = [(t,13,b,24) for t,b in zip((0,4,8,12,16,20,24),(4,8,12,16,20,24,29))]
    for rectangles,wall,fill in ((source_rects,3,5),(target_rects,4,6)):
        for t,l,b,r in rectangles:
            extended[t+1:b,l+1:r] = fill
        extended[_walls(extended.shape,rectangles)] = wall
    expected = extended.copy()
    expected[expected == 6] = 5
    for point in ((1,7),(2,7),(3,7),(2,8)):
        extended[point] = 1
    extended[0,1] = 2
    _need(np.array_equal(solve(extended),expected),'solver imposes sampled count ceiling')
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            validate_generated_puzzle(generate_puzzle(1200+a*10+b,a,b))
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Changing a label must be detected, including a distant reached room.
    bad = generate_puzzle(0,2,1)
    bad['test'][0]['output'] = bad['test'][0]['output'].copy()
    info = _analyze(bad['test'][0]['input'])
    arrow = info['arrows'][0]
    room = next(iter(arrow['reached']))
    point = tuple(np.argwhere(info['frames'][arrow['target']]['labels'] == room)[0])
    bad['test'][0]['output'][point] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42,pairs=len(records),shapes=len(shapes),clean_structures=len(structures),
                graph_degree_signatures=len(topologies),counts=sorted(counts),
                directions=len(directions),semantic_verification='generator-owned checks only')


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
