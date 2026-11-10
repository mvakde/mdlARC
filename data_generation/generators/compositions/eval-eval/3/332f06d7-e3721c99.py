"""Topology-keyed cargo: a topology key changes the configuration space.

Only NumPy and the standard library are used. Generation and checks keep all
grids in memory; this module never writes samples, logs, or execution traces.
"""
from __future__ import annotations

from collections import deque
import hashlib
import numpy as np

MOVER, GOAL, DIVIDER = 5, 8, 9
FREE_COLORS = (0, 1, 2, 3, 4, 6, 7)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _neighbors(p, shape):
    r, c = p
    for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _components(mask):
    remaining = np.asarray(mask, dtype=bool).copy()
    result = []
    for p in map(tuple, np.argwhere(remaining)):
        if not remaining[p]:
            continue
        remaining[p] = False
        cells = [p]
        for q in cells:
            for u in _neighbors(q, remaining.shape):
                if remaining[u]:
                    remaining[u] = False
                    cells.append(u)
        result.append(cells)
    return result


def _mask(cells):
    points = np.asarray(cells, dtype=int)
    lo, hi = points.min(0), points.max(0) + 1
    mask = np.zeros(tuple(hi-lo), dtype=bool)
    mask[tuple((points-lo).T)] = True
    return mask, tuple(map(int, lo))


def _holes(mask):
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    return [cells for cells in _components(~padded)
            if all(0 < r < padded.shape[0]-1 and 0 < c < padded.shape[1]-1 for r, c in cells)]


def _distance(mask, sources):
    d = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for p in sources:
        p = tuple(map(int, p))
        if mask[p] and d[p] < 0:
            d[p] = 0
            queue.append(p)
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, mask.shape):
            if mask[q] and d[q] < 0:
                d[q] = d[p] + 1
                queue.append(q)
    return d


def _placements(allowed, shape):
    h, w = shape.shape
    if h > allowed.shape[0] or w > allowed.shape[1]:
        return np.zeros((0, 0), dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(allowed, (h, w))
    return np.all(windows[..., shape], axis=-1)


def _canonical(grid):
    """The fixed 9 divider and the 5/8 markers encode the panel's side."""
    grid = _grid(grid)
    candidates = []
    for turns in range(4):
        a = np.rot90(grid, turns)
        for r in np.flatnonzero(np.all(a == DIVIDER, axis=1)):
            if 0 < r < a.shape[0]-1 and np.any(a[r+1:] == MOVER) and np.any(a[r+1:] == GOAL):
                if not np.any(np.isin(a[:r], (MOVER, GOAL))):
                    candidates.append((a, turns, int(r)))
    _require(len(candidates) == 1, "ambiguous legend orientation")
    return candidates[0]


def _parse(grid):
    a, turns, divider = _canonical(grid)
    bg = int(a[0, 0])
    _require(bg in FREE_COLORS and all(int(x) == bg for x in (a[0,-1], a[-1,0], a[-1,-1])),
             "corners must identify background")
    _require(int((a == DIVIDER).sum()) == a.shape[1], "divider color outside divider")
    panel, arena = a[:divider], a[divider+1:]
    _require(np.all(panel[0] == bg) and np.all(arena[-1] == bg), "missing outer background frame")
    _require(np.all(a[:divider, 0] == bg) and np.all(a[:divider,-1] == bg)
             and np.all(arena[:,0] == bg) and np.all(arena[:,-1] == bg), "missing side frame")
    mapping, entries = {}, []
    for color in map(int, np.unique(panel)):
        if color == bg:
            continue
        _require(color in FREE_COLORS, "reserved marker in legend")
        parts = _components(panel == color)
        _require(len(parts) == 1, "legend color has multiple glyphs")
        shape, pos = _mask(parts[0])
        cavities = _holes(shape)
        k = len(cavities)
        _require(k not in mapping, "duplicate legend topology")
        for cavity in cavities:
            _require(all(panel[pos[0]+r-1, pos[1]+c-1] == bg for r, c in cavity),
                     "legend cavity contains another glyph")
        mapping[k] = color
        entries.append((pos, k, color, shape))
    _require(bool(mapping), "empty legend")
    rest = set(map(int, np.unique(arena))) - {bg, MOVER, GOAL, DIVIDER} - set(mapping.values())
    _require(len(rest) == 1, "floor color is not uniquely encoded")
    floor = rest.pop()
    _require(floor in FREE_COLORS, "reserved floor color")
    cargo_parts = _components(arena == MOVER)
    _require(len(cargo_parts) == 1, "expected one connected cargo")
    cargo, start = _mask(cargo_parts[0])
    count = len(_holes(cargo))
    _require(count in mapping, "cargo has no topology key")
    for cavity in _holes(cargo):
        _require(all(arena[start[0]+r-1,start[1]+c-1] == floor for r,c in cavity),
                 "cargo cavities must be floor")
    target_cells = np.argwhere(arena == GOAL)
    _require(len(target_cells) > 0, "absent target")
    target_shape, target = _mask(target_cells)
    _require(target_shape.shape == cargo.shape and target_shape.all(), "target must fill cargo's bounding rectangle")
    _require(not np.any(arena[start[0]:start[0]+cargo.shape[0], start[1]:start[1]+cargo.shape[1]] == GOAL),
             "overlapping terminals")
    route = arena != bg
    for color in mapping.values():
        for cells in _components(arena == color):
            gate, (r,c) = _mask(cells)
            gh,gw = gate.shape
            vertical = (gw == 1 and r > 0 and r+gh < arena.shape[0]
                        and not route[r-1,c] and not route[r+gh,c])
            horizontal = (gh == 1 and c > 0 and c+gw < arena.shape[1]
                          and not route[r,c-1] and not route[r,c+gw])
            _require(gate.all() and (vertical or horizontal), "door must span one corridor cross-section")
    d = _distance(route, [target])
    _require(np.all(d[route] >= 0), "disconnected route or free-floating doors")
    # No wall island: the visible corridor has a single macroscopic route.
    walls = arena == bg
    border = np.zeros_like(walls)
    border[0] = border[-1] = True
    border[:,0] = border[:,-1] = True
    wd = _distance(walls, np.argwhere(walls & border))
    _require(np.all(wd[walls] >= 0), "closed corridor loop")
    return dict(grid=a, turns=turns, divider=divider, arena=arena, bg=bg, floor=floor,
                mapping=mapping, entries=entries, cargo=cargo, start=start, target=target,
                count=count, key=mapping[count], route=route)


def _movement(p, *, open_all=False, point=False):
    arena, shape = p['arena'], p['cargo']
    if point:
        shape = np.ones((1, 1), dtype=bool)
    allowed = p['route'].copy()
    if not open_all:
        allowed &= ~np.isin(arena, [c for c in p['mapping'].values() if c != p['key']])
    valid = _placements(allowed, shape)
    reach = _distance(valid, [p['start']])
    reachable = reach >= 0
    _require(reachable[p['start']], "invalid starting placement")
    if reachable[p['target']]:
        return p['target'], int(reach[p['target']]), True, reachable
    # Distances run along the corridor with ALL doors open. They establish
    # progress toward the target, independently of which obstacle stopped us.
    d = _distance(p['route'], np.argwhere(arena == GOAL))
    h, w = shape.shape
    windows = np.lib.stride_tricks.sliding_window_view(d, (h, w))[..., shape]
    destinations = []
    for score in (windows.min(-1), windows.max(-1), windows.sum(-1)):
        best = np.argwhere(reachable & (score == score[reachable].min()))
        _require(len(best) == 1, "ambiguous lateral stopping position")
        destinations.append(tuple(map(int, best[0])))
    _require(len(set(destinations)) == 1, "inconsistent stopping front")
    dest = destinations[0]
    return dest, int(reach[dest]), False, reachable


def _predict(p):
    dest, travel, success, reachable = _movement(p)
    out = p['grid'].copy()
    arena = out[p['divider']+1:]
    # Unlock all matching doors, including any beyond the first obstruction.
    arena[arena == p['key']] = p['floor']
    arena[arena == MOVER] = p['floor']
    h, w = p['cargo'].shape
    if success:
        r, c = p['target']
        arena[r:r+h, c:c+w] = p['floor']
    r, c = dest
    patch = arena[r:r+h, c:c+w]
    _require(np.all(np.isin(patch[p['cargo']], (p['floor'], GOAL))), "invalid final footprint")
    patch[p['cargo']] = p['key']
    info = dict(destination=dest, travel=travel, success=success,
                placements=int(reachable.sum()))
    return np.rot90(out, -p['turns']).copy(), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the local key, open its doors, and move the complete rigid cargo."""
    return _predict(_parse(input_grid))[0]


def _compact(count):
    if count == 0:
        return np.ones((2, 3), dtype=bool)
    rows = 1 if count <= 3 else 2
    cols = (count+rows-1)//rows
    a = np.ones((2*rows+1, 2*cols+1), dtype=bool)
    for i in range(count):
        a[1+2*(i//cols), 1+2*(i%cols)] = False
    return a


def _cargo(rng, count, larger=False):
    """Grown unequal cavities and topology-preserving exterior carving."""
    minimum = 3 if count <= 1 else 4 if count <= 2 else 5
    h = int(rng.integers(max(minimum, 4 if larger else 3), 7 if larger else 6))
    w = int(rng.integers(max(minimum, 4 if larger else 3), 7 if larger else 6))
    if larger:
        # A guaranteed geometric shift, independent of the cavity-count key.
        if rng.integers(2):
            h = 6
        else:
            w = 6
    shape = np.ones((h,w), dtype=bool)
    # Checkerboard seeds keep distinct holes four-disconnected.
    sites = [(r,c) for r in range(1,h-1,2) for c in range(1,w-1,2)]
    if len(sites) < count:
        h = w = max(5, minimum)
        shape = np.ones((h,w), dtype=bool)
        sites = [(r,c) for r in range(1,h-1,2) for c in range(1,w-1,2)]
    for i in rng.choice(len(sites), count, replace=False):
        shape[sites[int(i)]] = False
    for _ in range(10):
        candidate = shape.copy()
        r, c = int(rng.integers(h)), int(rng.integers(w))
        candidate[r,c] = False
        if (candidate.any() and len(_components(candidate)) == 1 and len(_holes(candidate)) == count
                and candidate[0].any() and candidate[-1].any()
                and candidate[:,0].any() and candidate[:,-1].any()):
            shape = candidate
    return shape


def _walk(rng, q, max_h, max_w, long):
    gap_r, gap_c = int(rng.integers(2,5)), int(rng.integers(2,5))
    nr = min(3, (max_h+gap_r)//(q+gap_r))
    nc = min(4, (max_w+gap_c)//(q+gap_c))
    _require(nr >= 2 and nc >= 2, "rooms do not fit")
    limit = min(nr*nc, 8 if long else 5)
    wanted = int(rng.integers(4, limit+1))
    best = []
    for _ in range(12):
        path = [(int(rng.integers(nr)), int(rng.integers(nc)))]
        while len(path) < wanted:
            choices = [x for x in _neighbors(path[-1], (nr,nc)) if x not in path]
            if not choices:
                break
            path.append(choices[int(rng.integers(len(choices)))])
        if len(path) > len(best):
            best = path
        if len(best) == wanted:
            break
    _require(len(best) >= 4, "short walk")
    return [(r*(q+gap_r), c*(q+gap_c)) for r,c in best]


def _edge(a, b, q, width, offset):
    if a[0] == b[0]:
        lo, hi = sorted((a[1],b[1]))
        return slice(a[0]+offset,a[0]+offset+width), slice(lo+q,hi)
    lo, hi = sorted((a[0],b[0]))
    return slice(lo+q,hi), slice(a[1]+offset,a[1]+offset+width)


def _terminal(rng, a, b, q, h, w):
    if a[0] == b[0]:
        return a[0]+int(rng.integers(q-h+1)), a[1] if a[1] < b[1] else a[1]+q-w
    return a[0] if a[0] < b[0] else a[0]+q-h, a[1]+int(rng.integers(q-w+1))


def _panel(rng, keys, colors, bg):
    glyphs = []
    for k in keys:
        g = _compact(k)
        if rng.random() < .35 and g.shape[1] <= 5:
            g = g.T
        # Size and outline of a legend prototype do not describe cargo size.
        glyphs.append(g.copy())
    width = sum(g.shape[1] for g in glyphs) + len(glyphs)+1
    height = max(g.shape[0] for g in glyphs)+2
    _require(width <= 28 and height <= 7, "legend too large")
    panel = np.full((height,width), bg, dtype=np.int8)
    c = 1
    for g,color in zip(glyphs,colors):
        r = int(rng.integers(1,height-g.shape[0]))
        panel[r:r+g.shape[0],c:c+g.shape[1]][g] = color
        c += g.shape[1]+1
    return panel


def _make_pair(rng, roles, count, outcome, long=False, compound=False, rank_witness=False):
    bg, floor = roles
    pool = [c for c in FREE_COLORS if c not in roles]
    nkeys = int(rng.integers(2,5))
    keys = [count] + [int(x) for x in rng.choice([k for k in range(5) if k != count], nkeys-1, replace=False)]
    rng.shuffle(keys)
    if rank_witness and keys.index(count) == count:
        keys = keys[1:]+keys[:1]
    colors = [int(x) for x in rng.choice(pool,nkeys,replace=False)]
    key = colors[keys.index(count)]
    other = int(rng.choice([c for c in colors if c != key]))
    panel = _panel(rng,keys,colors,bg)
    cargo = _cargo(rng,count,long)
    h,w = cargo.shape
    q = max(h,w)+int(rng.integers(2))
    max_h, max_w = 30-panel.shape[0]-3, 28
    path = _walk(rng,q,max_h,max_w,long)
    canvas = np.full((max_h,max_w),bg,dtype=np.int8)
    for r,c in path:
        canvas[r:r+q,c:c+q] = floor
    blocking = int(rng.integers(1,len(path)-1)) if outcome != 'success' else -1
    # Include a matching door before the requested first blockage. Later doors
    # need not be visited: all matching doors open as soon as the key is read.
    doors = {0:key}
    if outcome == 'gate':
        doors[blocking] = other
    for i in range(1,len(path)-1):
        if i == blocking:
            continue
        if rng.random() < (.85 if long else .55):
            doors[i] = key
            if compound and i > blocking >= 0 and rng.random() < .4:
                doors[i] = int(rng.choice(colors))
    for i,(a,b) in enumerate(zip(path,path[1:])):
        horizontal = a[0] == b[0]
        needed = h if horizontal else w
        low = needed
        if outcome == 'size' and i == blocking:
            width = int(rng.integers(1,needed))
        elif compound and blocking >= 0 and i > blocking and rng.random() < .3:
            width = int(rng.integers(1,q+1))
        else:
            width = int(rng.integers(low,q+1))
        offset = int(rng.integers(q-width+1))
        # At a stop, a snug collar makes the lateral alignment observable.
        if i == blocking:
            collar = int(rng.integers(q-needed+1))
            offset = collar+int(rng.integers(needed-width+1)) if width < needed else collar
            if outcome == 'gate':
                width = needed
            r,c = a
            if horizontal:
                end_c = c+q-w if b[1] > c else c
                canvas[r:r+q,end_c:end_c+w] = bg
                canvas[r+collar:r+collar+h,end_c:end_c+w] = floor
            else:
                end_r = r+q-h if b[0] > r else r
                canvas[end_r:end_r+h,c:c+q] = bg
                canvas[end_r:end_r+h,c+collar:c+collar+w] = floor
        rs,cs = _edge(a,b,q,width,offset)
        canvas[rs,cs] = floor
        if i in doors:
            if horizontal:
                col = int(rng.integers(cs.start,cs.stop))
                canvas[rs,col] = doors[i]
            else:
                row = int(rng.integers(rs.start,rs.stop))
                canvas[row,cs] = doors[i]
    start = _terminal(rng,path[0],path[1],q,h,w)
    target = _terminal(rng,path[-1],path[-2],q,h,w)
    r,c = start
    _require(np.all(canvas[r:r+h,c:c+w] == floor), "start collar collision")
    canvas[r:r+h,c:c+w][cargo] = MOVER
    r,c = target
    _require(np.all(canvas[r:r+h,c:c+w] == floor), "target collar collision")
    canvas[r:r+h,c:c+w] = GOAL
    used = np.argwhere(canvas != bg)
    lo,hi = used.min(0),used.max(0)+1
    canvas = canvas[lo[0]:hi[0],lo[1]:hi[1]]
    # Leave a one-cell background frame. Extra padding is asymmetric.
    room_h = 30-panel.shape[0]-1-canvas.shape[0]
    room_w = 30-max(canvas.shape[1]+2,panel.shape[1])
    _require(room_h >= 2 and room_w >= 0, "canvas overflow")
    extra_h = int(rng.integers(2,min(room_h,5)+1))
    width = max(canvas.shape[1]+2,panel.shape[1])+int(rng.integers(room_w+1))
    top = int(rng.integers(1,extra_h))
    left = int(rng.integers(1,width-canvas.shape[1]))
    arena = np.full((canvas.shape[0]+extra_h,width),bg,dtype=np.int8)
    arena[top:top+canvas.shape[0],left:left+canvas.shape[1]] = canvas
    full_panel = np.full((panel.shape[0],width),bg,dtype=np.int8)
    pc = int(rng.integers(width-panel.shape[1]+1))
    full_panel[:,pc:pc+panel.shape[1]] = panel
    inp = np.concatenate((full_panel,np.full((1,width),DIVIDER,dtype=np.int8),arena))
    p = _parse(inp)
    out,info = _predict(p)
    _require(info['travel'] > 0, "stationary cargo")
    _require(info['success'] == (outcome == 'success'), "wrong requested outcome")
    if outcome == 'gate':
        alternate,_,_,_ = _movement(p,open_all=True)
        _require(alternate != info['destination'], "gate did not control movement")
    elif outcome == 'size':
        # A point can get past this geometric blockage; the actual object cannot.
        _,_,_,point_reach = _movement(p,point=True,open_all=True)
        _require(point_reach[p['target']], "not a footprint-only obstruction")
    turns = int(rng.integers(4))
    reflect = rng.random() < .5
    if rank_witness and len(keys)-1-keys.index(count) == count:
        reflect = False
    if reflect:
        inp,out = np.fliplr(inp),np.fliplr(out)
    return {'input':np.rot90(inp,turns).copy(),'output':np.rot90(out,turns).copy()}


def _check_counts(num_train,num_test):
    for value,name,low,high in ((num_train,'num_train',2,10),(num_test,'num_test',1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= value <= high:
            raise ValueError(name+' is outside its legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _signature(a):
    a = np.asarray(a,dtype=np.int8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    roles = tuple(map(int,rng.choice(FREE_COLORS,2,replace=False)))
    variable_roles = bool(rng.integers(2))
    first_count = int(rng.integers(3))
    second_count = int(rng.choice([x for x in range(3) if x != first_count]))
    train_specs = [(first_count,'gate'),(second_count,'size')]
    if num_train >= 3:
        train_specs.append((int(rng.integers(4)),'success'))
    while len(train_specs) < num_train:
        train_specs.append((int(rng.integers(5)),str(rng.choice(['gate','size','success']))))
    test_outcomes = ['gate','size','success'] if num_train >= 3 else ['gate','size']
    test_specs = [(int(rng.integers(5)),str(rng.choice(test_outcomes))) for _ in range(num_test)]
    # Teaching cases are shuffled: example rank never signals the output.
    rng.shuffle(train_specs)
    puzzle,seen = {'train':[],'test':[]},set()
    for split,specs in (('train',train_specs),('test',test_specs)):
        for count,outcome in specs:
            for _ in range(160):
                local = tuple(map(int,rng.choice(FREE_COLORS,2,replace=False))) if variable_roles else roles
                try:
                    pair = _make_pair(rng,local,count,outcome,long=split == 'test',
                                      compound=split == 'test' and num_train >= 3,
                                      rank_witness=split == 'train')
                    signature = _signature(pair['input'])
                    if signature in seen:
                        continue
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError('bounded construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    p = _parse(pair['input'])
    _,info = _predict(p)
    key_colors = list(p['mapping'].values())
    door_counts = {c:len(_components(p['arena'] == c)) for c in key_colors}
    gate_witness = False
    if not info['success']:
        try:
            gate_witness = _movement(p,open_all=True)[0] != info['destination']
        except AssertionError:
            pass
    # This test isolates footprint collision, keeping the real door policy.
    size_witness = False
    if not info['success']:
        allowed = p['route'] & ~np.isin(p['arena'],[c for c in key_colors if c != p['key']])
        size_witness = bool(_distance(allowed,[p['start']])[p['target']] >= 0)
    return dict(shape=tuple(pair['input'].shape),cargo=p['cargo'].shape,count=p['count'],
                area=int(p['cargo'].sum()),cavity_areas=tuple(len(x) for x in _holes(p['cargo'])),
                key_order=tuple(x[1] for x in sorted(p['entries'],key=lambda e:e[0][::-1])),
                keys=tuple(sorted(p['mapping'])),palette=(p['bg'],p['floor'],p['key']),
                panel_side=p['turns'],doors=sum(door_counts.values()),
                matched_doors=door_counts[p['key']],locked_doors=sum(v for c,v in door_counts.items() if c != p['key']),
                gate_witness=gate_witness,size_witness=size_witness,**info)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "splits must be lists")
    _check_counts(len(puzzle['train']),len(puzzle['test']))
    seen,training = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'}, "invalid pair")
            inp,out = _grid(pair['input']),_grid(pair['output'])
            _require(inp.shape == out.shape, "shape changed")
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            p = _parse(inp)
            expected,info = _predict(p)
            _require(np.array_equal(out,expected), "wrong unlock, translation, or preserved scenery")
            output_arena = np.rot90(out,p['turns'])[p['divider']+1:]
            _require(not np.any(output_arena == MOVER), "neutral cargo must disappear")
            _require(int((output_arena == p['key']).sum()) == int(p['cargo'].sum()),
                     "cargo area was not conserved after unlocking")
            _require(len(_components(output_arena == p['key'])) == 1, "cargo fragmented")
            placed,_ = _mask(np.argwhere(output_arena == p['key']))
            _require(np.array_equal(placed,p['cargo']), "cargo was deformed")
            _require(np.array_equal(output_arena == p['bg'],p['arena'] == p['bg']), "wall changed")
            _require(np.array_equal(output_arena == GOAL,
                                   np.zeros_like(output_arena,dtype=bool) if info['success'] else p['arena'] == GOAL),
                     "target preservation/replacement error")
            _require(info['travel'] > 0, "degenerate motion")
            _require(np.any(p['arena'] == p['key']), "no matching door")
            if split == 'train':
                training.append(_features(pair))
    _require(any(x['gate_witness'] for x in training), "training lacks a topology-controlled stop")
    _require(any(x['size_witness'] for x in training), "training lacks a footprint-controlled stop")
    _require(len({x['count'] for x in training}) >= 2, "training needs two topology classes")
    _require(any(x['key_order'].index(x['count']) != x['count'] for x in training),
             "training lacks a witness against legend-rank decoding")
    if any(_features(pair)['success'] for pair in puzzle['test']):
        _require(any(x['success'] for x in training), "goal replacement not demonstrated")


def self_test():
    """Bounded 42-seed coverage, geometric fixtures, and negative checks."""
    diagonal = np.array([[0,1,1],[1,0,1],[1,1,1]],dtype=bool)
    _require(len(_holes(diagonal)) == 1,"diagonal escape is not orthogonal")
    diagonal[0,1] = False
    _require(len(_holes(diagonal)) == 0,"orthogonal opening must remove hole")
    notch = np.array([[1,0,0,1],[1,0,0,1],[1,1,1,1]],dtype=bool)
    _require(len(_holes(notch)) == 0,"open notch is not a cavity")
    # Actual shape collision, not bounding-box collision: the peg fits the hole.
    ring = np.ones((3,3),dtype=bool)
    ring[1,1] = False
    room = np.ones((5,5),dtype=bool)
    room[2,2] = False
    _require(_placements(room,ring)[1,1],"holes must not collide with a peg")
    _require(not _placements(room,np.ones((3,3),dtype=bool))[1,1],"solid footprint missed peg")
    # Hand-specified end-to-end contrast, independent of the random constructor.
    # Identical rectangle and start, but changing topology changes the open door.
    panel = np.zeros((5,28),dtype=np.int8)
    panel[1:3,1:4] = 2  # class zero
    panel[1:4,6:9][ring] = 3  # class one
    for case,destination,key in (('size',(2,9),3),('gate',(2,17),3),
                                 ('success',(2,24),3),('solid',(2,3),2)):
        arena = np.zeros((7,28),dtype=np.int8)
        arena[2:5,1:27] = 1
        arena[2:5,6] = 3
        arena[2:5,20] = 3 if case == 'success' else 2
        if case == 'size':
            arena[4,12:16] = 0
        shape = np.ones((3,3),dtype=bool) if case == 'solid' else ring
        arena[2:5,1:4][shape] = MOVER
        arena[2:5,24:27] = GOAL
        inp = np.concatenate((panel,np.full((1,28),DIVIDER,dtype=np.int8),arena))
        expected = inp.copy()
        work = expected[6:]
        work[work == key] = 1
        work[work == MOVER] = 1
        if case == 'success':
            work[2:5,24:27] = 1
        r,c = destination
        work[r:r+3,c:c+3][shape] = key
        _require(np.array_equal(solve(inp),expected),'hand-specified '+case+' regression')
        for turns in range(4):
            _require(np.array_equal(solve(np.rot90(inp,turns)),np.rot90(expected,turns)),
                     'orientation changed fixture behavior')
    records,hashes,normalized = [],set(),set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                for key in ('input','output'):
                    a = pair[key]
                    digest.update(bytes(a.shape)+a.tobytes())
                _require(np.array_equal(solve(pair['input']),pair['output']),"solve disagreement")
                records.append(_features(pair))
                roles = {}
                normalized.add((pair['input'].shape,bytes(roles.setdefault(int(c),len(roles)) for c in pair['input'].flat)))
        hashes.add(digest.digest())
    _require(len(hashes) == 42,"seed diversity collapsed")
    _require(len(normalized) >= 180,"structural diversity collapsed")
    _require(len({x['shape'] for x in records}) >= 35,"canvas shape diversity collapsed")
    _require(len({x['cargo'] for x in records}) >= 10,"cargo geometry diversity collapsed")
    _require({x['count'] for x in records} == set(range(5)),"topology classes missing")
    _require({x['panel_side'] for x in records} == set(range(4)),"panel side diversity collapsed")
    _require(len({x['keys'] for x in records}) >= 12,"key subsets collapsed")
    _require(len({x['key_order'] for x in records}) >= 25,"key orders collapsed")
    _require({x['success'] for x in records} == {False,True},"outcomes missing")
    _require(sum(x['gate_witness'] for x in records) >= 42,"gate stops missing")
    _require(sum(x['size_witness'] for x in records) >= 42,"size stops missing")
    _require(max(x['travel'] for x in records) >= 20,"long traversals missing")
    _require(max(x['matched_doors'] for x in records) >= 3,"repeated matching doors missing")
    _require(any(len(set(x['cavity_areas'])) > 1 for x in records),"unequal cavities missing")
    a,b = generate_puzzle(2026,2,1),generate_puzzle(2026,2,1)
    _require(all(np.array_equal(x[k],y[k]) for s in a for x,y in zip(a[s],b[s]) for k in x),"nondeterminism")
    # A genuine mutation of the answer must be rejected, including panel cells.
    for where in ('panel','cargo','door','goal'):
        bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in a[s]] for s in a}
        pair = bad['train'][0]
        p = _parse(pair['input'])
        canonical_out = np.rot90(pair['output'],p['turns']).copy()
        if where == 'panel':
            loc = tuple(np.argwhere(p['grid'][:p['divider']] == p['key'])[0])
        elif where == 'cargo':
            dest = _movement(p)[0]
            delta = tuple(np.argwhere(p['cargo'])[0])
            loc = dest[0]+delta[0]+p['divider']+1,dest[1]+delta[1]
        else:
            color = p['key'] if where == 'door' else GOAL
            loc = tuple(np.argwhere(p['grid'][p['divider']+1:] == color)[0])
            loc = loc[0]+p['divider']+1,loc[1]
        canonical_out[loc] = (int(canonical_out[loc])+1)%10
        pair['output'] = np.rot90(canonical_out,-p['turns']).copy()
        try:
            validate_generated_puzzle(bad)
        except (AssertionError,ValueError):
            pass
        else:
            raise AssertionError('accepted corrupted '+where)
    for counts in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(2.0,1),(None,1)):
        try:
            generate_puzzle(0,*counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds)')
