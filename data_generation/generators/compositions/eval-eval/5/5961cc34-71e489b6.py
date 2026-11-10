"""Hole-controlled relay shutters. Only NumPy and the standard library are used.

The decoder sees just pixels. Generation keeps explicit rectangles and defects;
its labels use a cell-stepping traversal, whereas solve projects ray/box distances.
Both checks are authored here, not an independent semantic oracle.
"""
from collections import deque
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected a 1..30 integer grid using colors 0..9')
    return a.astype(np.int64, copy=True)


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _cells(box):
    y, v, x, u = box
    return {(r, c) for r in range(y, v) for c in range(x, u)}


def _step(p, direction):
    dr, dc = DIRS[direction]
    return p[0] + dr, p[1] + dc


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _port(box, side, start, length):
    y, v, x, u = box
    if side == 0:
        return {(y, c) for c in range(start, start + length)}
    if side == 2:
        return {(v - 1, c) for c in range(start, start + length)}
    if side == 1:
        return {(r, u - 1) for r in range(start, start + length)}
    return {(r, x) for r in range(start, start + length)}


def _object(box, side, start, length):
    return {'box': tuple(box), 'side': int(side),
            'port': _port(box, side, start, length), 'holes': set()}


def _emitting(obj, ignore_gates=False):
    # Project defects onto the port, perpendicular to its extreme side.
    axis = 0 if obj['side'] % 2 else 1
    shadows = set() if ignore_gates else {p[axis] for p in obj['holes']}
    return sorted(p for p in obj['port'] if p[axis] not in shadows)


def _decode(inp):
    h, w = inp.shape
    bg = int(inp[0, 0])
    if any(int(inp[p]) != bg for p in ((0, w-1), (h-1, 0), (h-1, w-1))):
        raise ValueError('the four canvas corners must be field')
    components = _components(map(tuple, np.argwhere(inp != bg)))
    candidates = []
    for comp in components:
        rows, cols = {p[0] for p in comp}, {p[1] for p in comp}
        edge = [p for p in comp if p[0] in (0, h-1) or p[1] in (0, w-1)]
        if 3 <= len(comp) <= 5 and len(edge) == 1 and (len(rows) == 1 or len(cols) == 1):
            p = edge[0]
            direction = 2 if p[0] == 0 else 0 if p[0] == h-1 else 1 if p[1] == 0 else 3
            line = [p]
            for _ in range(len(comp)-1):
                line.append(_step(line[-1], direction))
            if set(line) == comp:
                sig, halo = int(inp[line[0]]), int(inp[line[-1]])
                if sig != halo and all(int(inp[q]) == sig for q in line[:-1]):
                    candidates.append((comp, line, direction, sig, halo))
    if len(candidates) != 1:
        raise ValueError('need one visible boundary tail ending in a distinct cap')
    launcher, line, direction, sig, halo = candidates[0]
    objects, extras = [], set()
    roles = None
    for comp in components:
        if comp == launcher:
            continue
        if len(comp) <= 2:
            extras |= comp
            continue
        y, v = min(p[0] for p in comp), max(p[0] for p in comp)+1
        x, u = min(p[1] for p in comp), max(p[1] for p in comp)+1
        if min(v-y, u-x) < 4:
            raise ValueError('relays have both dimensions at least four')
        colors, counts = np.unique([inp[p] for p in comp], return_counts=True)
        if len(colors) != 2 or counts[0] == counts[1]:
            raise ValueError('relay needs majority carrier and minority port')
        carrier, pc = int(colors[np.argmax(counts)]), int(colors[np.argmin(counts)])
        if len({bg, sig, halo, carrier, pc}) != 5:
            raise ValueError('semantic colors must be disjoint')
        if roles is not None and roles != (carrier, pc):
            raise ValueError('all relays in one input share carrier and port colors')
        roles = carrier, pc
        port = {p for p in comp if int(inp[p]) == pc}
        sides = [side for side, ok in enumerate((all(r == y for r, c in port),
                 all(c == u-1 for r, c in port), all(r == v-1 for r, c in port),
                 all(c == x for r, c in port))) if ok]
        if len(sides) != 1:
            raise ValueError('port must identify one extreme side')
        side = sides[0]
        span = sorted(p[0 if side % 2 else 1] for p in port)
        lo, hi = (y, v) if side % 2 else (x, u)
        if (span != list(range(span[0], span[0]+len(span)))
                or not lo < span[0] <= span[-1] < hi-1):
            raise ValueError('port must be contiguous and exclude corners')
        box = (y, v, x, u)
        cells = _cells(box)
        if any(int(inp[p]) != carrier for p in ((y,x), (y,u-1), (v-1,x), (v-1,u-1))):
            raise ValueError('the four relay corners anchor its filled rectangle')
        if not all(int(inp[p]) in (bg, carrier, pc) for p in cells):
            raise ValueError('foreign marking inside relay')
        holes = cells - comp
        if any(int(inp[p]) != bg for p in holes):
            raise ValueError('missing carrier cells use field color')
        if len(holes) > (len(cells)-len(port))//3:
            raise ValueError('too much corruption for the intact frame convention')
        objects.append({'box': box, 'side': side, 'port': port, 'holes': holes})
    if not objects or roles is None:
        raise ValueError('need at least one relay')
    carrier, pc = roles
    if any(int(inp[p]) != carrier for p in extras):
        raise ValueError('stray specks must use carrier color')
    # Rectangles and launcher have two empty cells of separation. Specks may
    # lie on ray paths, but cannot attach to a relay or masquerade as one.
    full = [_cells(o['box']) for o in objects]
    for i, cells in enumerate(full):
        foreign = launcher | set().union(*full[:i], *full[i+1:])
        if any(max(abs(r-a), abs(c-b)) <= 2 for r,c in cells for a,b in foreign):
            raise ValueError('relay rectangles and launcher need two empty cells')
    if any(max(abs(r-a), abs(c-b)) <= 1 for r,c in extras for a,b in set().union(*full, launcher)):
        raise ValueError('specks must stay separate from real objects')
    return {'shape': inp.shape, 'colors': (bg, sig, halo, carrier, pc),
            'tail': set(line[:-1]), 'cap': line[-1], 'initial': direction,
            'objects': objects, 'extras': extras}


def _trace_project(scene, ignore_gates=False, force_all=False):
    """Input solver: nearest original rectangle on each independent lane."""
    shape, objects = scene['shape'], scene['objects']
    trace = set(scene['tail']) | {scene['cap']}
    active = set(range(len(objects))) if force_all else set()
    pending = deque([(scene['cap'], scene['initial'])])
    if force_all:
        for o in objects:
            pending.extend((p, o['side']) for p in _emitting(o, ignore_gates))
    while pending:
        origin, direction = pending.popleft()
        r, c = origin
        limit = (r+1, shape[1]-c, shape[0]-r, c+1)[direction]
        best, hit = limit, None
        for i, obj in enumerate(objects):
            y, v, x, u = obj['box']
            distance = None
            if direction == 0 and x <= c < u and v-1 < r:
                distance = r-(v-1)
            elif direction == 2 and x <= c < u and y > r:
                distance = y-r
            elif direction == 1 and y <= r < v and x > c:
                distance = x-c
            elif direction == 3 and y <= r < v and u-1 < c:
                distance = c-(u-1)
            if distance is not None and distance < best:
                best, hit = distance, i
        dr, dc = DIRS[direction]
        trace.update((r+k*dr, c+k*dc) for k in range(1, best))
        if hit is not None and hit not in active:
            active.add(hit)
            obj = objects[hit]
            pending.extend((p, obj['side']) for p in _emitting(obj, ignore_gates))
    for i in active:
        trace |= _cells(objects[i]['box'])
    return trace, active


def _trace_step(scene):
    """Construction traversal: walk individual cells, emitting each relay once."""
    owners = {p: i for i,o in enumerate(scene['objects']) for p in _cells(o['box'])}
    trace = set(scene['tail']) | {scene['cap']}
    active, pending = set(), deque([(scene['cap'], scene['initial'])])
    while pending:
        origin, direction = pending.popleft()
        p = _step(origin, direction)
        while _inside(p, scene['shape']):
            if p in owners:
                i = owners[p]
                if i not in active:
                    active.add(i)
                    obj = scene['objects'][i]
                    trace |= _cells(obj['box'])
                    # Construction computes the shutters directly, rather than
                    # using the solver's emitting helper.
                    for port in sorted(obj['port']):
                        blocked = any(port[0 if obj['side'] % 2 else 1] ==
                                      hole[0 if obj['side'] % 2 else 1]
                                      for hole in obj['holes'])
                        if not blocked:
                            pending.append((port, obj['side']))
                break
            trace.add(p)
            p = _step(p, direction)
    return trace, active


def _paint(scene, trace, active):
    bg, sig, halo, _, _ = scene['colors']
    out = np.full(scene['shape'], bg, dtype=np.int64)
    for p in trace:
        out[p] = sig
    holes = set().union(*(scene['objects'][i]['holes'] for i in active))
    for r, c in holes:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = r+dr, c+dc
                if _inside(p, scene['shape']) and p not in holes:
                    out[p] = halo
    for p in holes:
        out[p] = bg
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, rectangles, holes, and reachability from this input."""
    scene = _decode(_grid(input_grid))
    return _paint(scene, *_trace_project(scene))


def _separate(box, boxes, margin=2):
    y,v,x,u = box
    return all(v+margin <= a or b+margin <= y or u+margin <= c or d+margin <= x
               for a,b,c,d in boxes)


def _random_port(rng, box, side=None, width_limit=4):
    if side is None:
        side = int(rng.integers(4))
    lo, hi = (box[0], box[1]) if side % 2 else (box[2], box[3])
    length = int(rng.integers(1, min(width_limit, hi-lo-2)+1))
    start = int(rng.integers(lo+1, hi-length))
    return _object(box, side, start, length)


def _legal_hole(obj, p):
    y,v,x,u = obj['box']
    return (p in _cells(obj['box']) and p not in obj['port']
            and p not in ((y,x), (y,u-1), (v-1,x), (v-1,u-1)))


def _add_holes(rng, obj, budget, protected_rows=()):
    # Small motifs include isolated, orthogonal, diagonal, and L clusters.
    motifs = (((0,0),), ((0,0),(0,1)), ((0,0),(1,1)),
              ((0,0),(1,0),(1,1)), ((0,0),(0,2)))
    y,v,x,u = obj['box']
    for _ in range(35):
        if len(obj['holes']) >= budget:
            break
        motif = motifs[int(rng.integers(len(motifs)))]
        r, c = int(rng.integers(y,v)), int(rng.integers(x,u))
        points = {(r+dr,c+dc) for dr,dc in motif}
        if (len(points | obj['holes']) > budget
                or any(p[0] in protected_rows or not _legal_hole(obj,p) for p in points)):
            continue
        proposed = obj['holes'] | points
        if len(_components(_cells(obj['box']) - proposed)) == 1:
            obj['holes'] = proposed


def _layout(rng, style, extend):
    """Variable fork geometry, with optional deeper and freely packed relays.

    Two targets at different distances intercept different parts of a broad
    port. Other relays are packed freely; the traversal decides their fate.
    """
    fork = style != 'single'
    heights = [int(rng.integers(4,7)) for _ in range(2 if fork else 1)]
    gap_y = int(rng.integers(2,5)) if fork else 0
    root_h = sum(heights) + gap_y + int(rng.integers(2,5))
    root_w = int(rng.integers(4,8))
    y, x = int(rng.integers(1,4)), int(rng.integers(5,8))
    h = int(rng.integers(max(15,y+root_h+2),31))
    w = int(rng.integers(26 if extend else 22,31))
    rootbox = (y,y+root_h,x,x+root_w)
    first_y = y + 1
    target_boxes = []
    for i, th in enumerate(heights):
        ty = first_y if i == 0 else first_y+heights[0]+gap_y
        tx = rootbox[3] + int(rng.integers(2,5))
        tw = int(rng.integers(4,7))
        if extend and i == len(heights)-1:
            tw = 4
            tx = rootbox[3]+2
        target_boxes.append((ty,ty+th,tx,tx+tw))
    if any(b[1] > h or b[3] > w for b in target_boxes):
        return None
    # Partial target coverage gives useful one-lane counterexamples as well
    # as wide split beams. Port corners are never used.
    start = first_y + int(rng.integers(0,heights[0]-1))
    end = target_boxes[-1][1] - int(rng.integers(0,heights[-1]-1))
    if not fork:
        end = max(end,start+2)
    end = min(end,rootbox[1]-1)
    root = _object(rootbox,1,start,end-start)
    objects = [root]
    for i, box in enumerate(target_boxes):
        side = 1 if i == len(target_boxes)-1 else int(rng.choice([0,2]))
        objects.append(_random_port(rng,box,side,width_limit=2))
    extension_index = None
    if extend:
        target = objects[-1]
        # Keep a visible two-lane port, with a hole closing just one lane.
        box = target['box']
        target['port'] = _port(box,1,box[0]+1,2)
        er = box[0] + int(rng.integers(0,2))
        eh, ew = int(rng.integers(4,7)), int(rng.integers(4,6))
        ex = box[3] + 2
        if ex+ew > w or er+eh > h:
            return None
        eb = (er,er+eh,ex,ex+ew)
        if not _separate(eb,[o['box'] for o in objects]):
            return None
        extension_index = len(objects)
        objects.append(_random_port(rng,eb,width_limit=3))
    # Extra nodes need not be decoys. They may merge paths or be occluded;
    # only actual first-contact reachability determines their status.
    wanted = int(rng.integers(0,3))
    for _ in range(wanted):
        for _ in range(55):
            oh, ow = int(rng.integers(4,9)), int(rng.integers(4,9))
            r, c = int(rng.integers(0,h-oh+1)), int(rng.integers(0,w-ow+1))
            box = (r,r+oh,c,c+ow)
            if any(p in _cells(box) for p in ((0,0),(0,w-1),(h-1,0),(h-1,w-1))):
                continue
            # Reserve the launch corridor left of the root.
            if c < x and r <= y+root_h-1 and r+oh > y:
                continue
            if _separate(box,[o['box'] for o in objects]):
                objects.append(_random_port(rng,box))
                break
    lr = int(rng.integers(y,y+root_h))
    tail_len = int(rng.integers(2,min(4,x-2)))
    tail = {(lr,c) for c in range(tail_len)}
    cap = (lr,tail_len)
    # Sometimes enter the root perpendicular to its right-facing port.
    if h-rootbox[1] >= 5 and rng.random() < .4:
        lc = int(rng.integers(x,x+root_w))
        tail = {(h-1,lc),(h-2,lc)}
        cap, initial = (h-3,lc), 0
        if any(max(abs(r-a),abs(c-b)) <= 2 for r,c in tail|{cap}
               for o in objects for a,b in _cells(o['box'])):
            return None
    else:
        initial = 1
    return {'shape': (h,w), 'tail': tail, 'cap': cap, 'initial': initial,
            'objects': objects, 'extras': set()}, extension_index


def _encode(scene):
    bg,sig,halo,carrier,pc = scene['colors']
    inp = np.full(scene['shape'],bg,dtype=np.int64)
    for o in scene['objects']:
        for p in _cells(o['box'])-o['holes']:
            inp[p] = pc if p in o['port'] else carrier
    for p in scene['extras']:
        inp[p] = carrier
    for p in scene['tail']:
        inp[p] = sig
    inp[scene['cap']] = halo
    return inp


def _make_pair(rng, colors, style, extend, rotation, reflected):
    for attempt in range(240):
        layout = _layout(rng,style,extend)
        if layout is None:
            continue
        scene, extension = layout
        scene['colors'] = colors
        root, target = scene['objects'][:2]
        overlap = sorted({p[0] for p in root['port']
                          if target['box'][0] <= p[0] < target['box'][1]})
        if not overlap:
            continue
        # Coverage obligations, not seed-visible conventions.
        closed = overlap if style == 'closed' else overlap[:-1]
        if not closed and style == 'open':
            continue  # demonstrate partial blocking with a surviving lane
        hole_col = int(rng.integers(root['box'][2]+1,root['box'][3]-1))
        root['holes'].update((r,hole_col) for r in closed)
        protected = set(overlap)-set(closed)
        if style != 'single':
            lower = scene['objects'][2]
            lower_rows = sorted({p[0] for p in root['port']
                                 if lower['box'][0] <= p[0] < lower['box'][1]})
            if not lower_rows:
                continue
            protected |= {lower_rows[-1]}
        # Add an off-port defect: hole count alone cannot determine shadows.
        outside = [r for r in range(root['box'][0],root['box'][1])
                   if r not in {p[0] for p in root['port']}]
        if outside:
            r = int(rng.choice(outside))
            root['holes'].add((r,hole_col))
        _add_holes(rng,root,len(root['holes'])+int(rng.integers(0,3)),protected)
        for i,o in enumerate(scene['objects'][1:],1):
            budget = int(rng.integers(1,4))
            # Keep the extension's feeder open on one lane. In closed/open
            # examples this still teaches downstream shutters on the other.
            protect = set()
            if extend and i == 2:
                rows = sorted(p[0] for p in o['port'])
                o['holes'].add((rows[0],o['box'][2]+1))
                protect = {rows[-1]}
                budget = max(1,budget)
            _add_holes(rng,o,budget,protect)
        if style == 'closed' and not target['holes']:
            continue
        all_objects = set().union(*(_cells(o['box']) for o in scene['objects']))
        forbidden = all_objects | scene['tail'] | {scene['cap']}
        # Specks can sit on an incoming ray; they are transparent after
        # cleanup, rather than spurious one-cell relays.
        available = [(r,c) for r in range(1,scene['shape'][0]-1)
                     for c in range(1,scene['shape'][1]-1)
                     if all(max(abs(r-a),abs(c-b)) > 1 for a,b in forbidden)]
        count = int(rng.integers(1,5))
        rng.shuffle(available)
        for p in available:
            if all(max(abs(p[0]-a),abs(p[1]-b)) > 2 for a,b in scene['extras']):
                scene['extras'].add(p)
                if len(scene['extras']) == count:
                    break
        inp = _encode(scene)
        try:
            decoded = _decode(inp)
        except ValueError:
            continue
        trace, active = _trace_step(scene)
        predicted, decoded_active = _trace_project(decoded)
        if trace != predicted or len(active) != len(decoded_active):
            raise AssertionError('construction and input projection disagree')
        _, clean_active = _trace_project(scene,ignore_gates=True)
        if style == 'closed' and not (1 not in active and 1 in clean_active and 2 in active):
            continue
        if style == 'open' and not {0,1,2} <= active:
            continue
        if extend and extension not in active:
            continue
        # Non-extended demonstrations stop after the first fork. Freely
        # packed extra nodes remain decoys in these shallower examples.
        if not extend and not active <= set(range(2 if style == 'single' else 3)):
            continue
        if style != 'single' and 2 not in active:
            continue
        out = _paint(scene,trace,active)
        def transform(a):
            return np.rot90(np.fliplr(a) if reflected else a,rotation).copy()
        return {'input': transform(inp),'output': transform(out)}
    raise RuntimeError('could not construct the requested shutter evidence')


def _counts(num_train,num_test):
    for value,lo,hi in ((num_train,2,10),(num_test,1,4)):
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError('training count must be 2..10 and test count 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int,rng.permutation(10)[:5]))
    vary_colors = bool(rng.integers(2))
    orientation_shift = bool(rng.integers(2))
    base_rotation = int(rng.integers(4))
    puzzle, seen = {'train': [],'test': []}, set()
    for split,count in (('train',num_train),('test',num_test)):
        for i in range(count):
            style = ('open','closed')[i] if split == 'train' and i < 2 else str(rng.choice(['open','closed','single']))
            if split == 'test':
                style = str(rng.choice(['open','closed']))
            extend = split == 'test' or (orientation_shift and split == 'train' and i == 0)
            rotation = ((base_rotation+int(rng.choice([1,3])))%4 if split == 'test'
                        else base_rotation) if orientation_shift else int(rng.integers(4))
            colors = tuple(map(int,rng.permutation(10)[:5])) if vary_colors else palette
            for _ in range(12):
                pair = _make_pair(rng,colors,style,extend,rotation,bool(rng.integers(2)))
                key = _signature(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('could not obtain distinct input')
            seen.add(key)
            puzzle[split].append(pair)
    order = rng.permutation(num_train)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(a):
    a = np.asarray(a)
    return a.shape,a.astype(np.uint8).tobytes()


def _facts(inp):
    scene = _decode(inp)
    trace, active = _trace_project(scene)
    _, ungated = _trace_project(scene,ignore_gates=True)
    holes = set().union(*(scene['objects'][i]['holes'] for i in active))
    ports = [len(o['port']) for o in scene['objects']]
    blocked = [len(o['port'])-len(_emitting(o)) for o in scene['objects']]
    return {'scene': scene,'trace': trace,'active': active,'ungated': ungated,
            'holes': holes,'ports': ports,'blocked': blocked}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen, train_facts = set(), []
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape == out.shape
            key = _signature(inp)
            assert key not in seen
            seen.add(key)
            f = _facts(inp)
            scene = f['scene']
            expected = solve(inp)
            assert np.array_equal(out,expected), 'incorrect output'
            bg,sig,halo,carrier,pc = scene['colors']
            assert set(map(int,np.unique(out))) <= {bg,sig,halo}
            assert scene['extras'], 'generated examples show removable specks'
            assert f['holes'] and f['active'], 'generated scenes expose active defects'
            assert all(int(out[p]) == bg for p in f['holes']), 'holes are never filled'
            for r,c in f['holes']:
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        p = r+dr,c+dc
                        if _inside(p,inp.shape) and p not in f['holes']:
                            assert int(out[p]) == halo, 'exact Moore halo required'
            # A speck may legitimately become a signal/halo cell. Its input
            # carrier color must never survive simply because it was present.
            assert all(int(out[p]) != carrier for p in scene['extras'])
            if split == 'train':
                train_facts.append(f)
    assert any(f['active'] != f['ungated'] for f in train_facts), 'teach that holes change reachability'
    assert any(len(f['active']) >= 3 for f in train_facts), 'teach independent lanes and chaining'
    assert any(len(f['active']) < len(f['scene']['objects']) for f in train_facts), 'teach removal of inactive relays'
    assert any(0 < f['blocked'][i] < len(f['scene']['objects'][i]['port'])
               for f in train_facts for i in f['active']), 'teach partial shutters'
    # The all-active ablation retains the exact same local shutter/halo rule;
    # this check therefore isolates A's reachability contribution.
    assert any(not np.array_equal(_paint(f['scene'],*_trace_project(f['scene'],force_all=True)),
                                 solve(p['input']))
               for f,p in zip(train_facts,puzzle['train'])), 'routing must affect the answer'


def self_test():
    """Bounded 42-seed audit plus focused rule/symmetry/corruption checks."""
    signatures, shapes, structures, records, palettes = set(),set(),set(),[],set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p[k]) for s in ('train','test') for p in puzzle[s] for k in ('input','output')))
        for split in ('train','test'):
            for p in puzzle[split]:
                inp = np.asarray(p['input'])
                f = _facts(inp)
                records.append(f)
                shapes.add(inp.shape)
                palettes.add(f['scene']['colors'])
                mapping = {}
                structures.add((inp.shape,bytes(mapping.setdefault(int(c),len(mapping)) for c in inp.flat)))
                if seed < 3:
                    for k in range(4):
                        for flip in (False,True):
                            a = np.rot90(np.fliplr(inp) if flip else inp,k)
                            b = np.rot90(np.fliplr(p['output']) if flip else p['output'],k)
                            assert np.array_equal(solve(a),b)
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k],q[k]) for s in puzzle for p,q in zip(puzzle[s],again[s]) for k in p)
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 35 and len(palettes) >= 20
    assert {f['scene']['initial'] for f in records} == set(range(4))
    assert len({len(f['scene']['objects']) for f in records}) >= 3
    assert len({len(f['active']) for f in records}) >= 3
    assert len({len(f['holes']) for f in records}) >= 6
    assert len({n for f in records for n in f['ports']}) >= 6
    # Fixed hand-built graph: two holes in the same row block ONE lane, not
    # two. An off-port boundary hole creates a halo but casts no port shadow.
    root = _object((2,10,7,12),1,3,5)
    root['holes'] = {(4,8),(4,10),(2,9)}
    target = _object((3,7,15,19),0,16,1)
    target['holes'] = {(3,17),(4,17)}
    scene = {'shape': (14,24),'colors': (0,2,7,1,3), 'tail': {(5,0),(5,1)},
             'cap': (5,2),'initial': 1,'objects': [root,target],'extras': {(5,4),(1,21)}}
    inp = _encode(scene)
    decoded = _decode(inp)
    assert len(_emitting(decoded['objects'][0])) == 4
    assert np.array_equal(solve(inp),_paint(scene,*_trace_step(scene)))
    assert solve(inp)[5,4] == 2, 'carrier speck on the incoming ray is transparent'
    assert solve(inp)[2,17] == 7, 'boundary halo crosses the rectangle edge'
    # Moving a hole perpendicular to the outgoing direction changes the
    # shadow although count and distance to the port remain unchanged.
    moved = inp.copy()
    moved[4,10],moved[6,10] = 1,0
    assert len(_emitting(_decode(moved)['objects'][0])) == 3
    assert not np.array_equal(solve(inp),solve(moved))
    # Fully closed ports are sinks, even though the relay itself activates.
    closed_root = _object((2,10,7,12),1,3,5)
    closed_root['holes'] = {(r,8) for r in range(3,8)}
    closed_scene = dict(scene,objects=[closed_root,target])
    closed_input = _encode(closed_scene)
    assert _trace_project(_decode(closed_input))[1] == {0}
    assert _trace_project(_decode(closed_input),ignore_gates=True)[1] == {0,1}
    # Clipping at a canvas edge, with two adjacent missing cells excluded
    # from each other's overlapping halos.
    edge_root = _object((0,8,7,12),1,1,6)
    edge_root['holes'] = {(0,8),(0,9)}
    edge_scene = dict(scene,objects=[edge_root,target])
    edge_input = _encode(edge_scene)
    edge_output = solve(edge_input)
    assert edge_output[0,8] == edge_output[0,9] == 0
    assert edge_output[0,7] == edge_output[1,10] == 7
    # Re-entering an already active relay consumes that ray without emitting
    # again. Two opposing ports therefore terminate even in a directed cycle.
    cycle_root = _object((2,10,7,12),1,3,5)
    cycle_target = _object((3,7,15,19),3,4,1)
    cycle_scene = dict(scene,objects=[cycle_root,cycle_target])
    cycle_input = _encode(cycle_scene)
    assert _trace_project(_decode(cycle_input))[1] == {0,1}
    assert np.array_equal(solve(cycle_input),_paint(cycle_scene,*_trace_step(cycle_scene)))
    # Malformed count requests are rejected, including bool/float/None.
    for counts in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,*counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(41,2,1)
    broken['test'][0]['output'][0,0] = (int(broken['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged label accepted')
