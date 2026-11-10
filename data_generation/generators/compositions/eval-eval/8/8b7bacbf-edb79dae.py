"""Port-registered pattern irrigation. Standard library and NumPy only.

No files, RNG, training pairs, or generation metadata are used by solve.
Generation and self_test keep all per-seed work in memory and write no artifacts.
"""
from collections import deque
import hashlib
import numpy as np

D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
HOST_SEEDS = (0,384644116,871979208,299654331,6265073,72131573,165392347,468275739,908605736,644516184,206448542,705954240,424812038,567050362,694208649,762849510,312232872,359301790,160364366,447753042,35259229,563041280,481761586,456594445,409762000,168428714,427144492,513581418,550192763,18412981,685009069,180688505,675484417,688509340,196847565,370539662,420451701,624635134,432105815,324384367,626800347,826167554)


def _components(cells, steps=D4):
    left = set(cells)
    result = []
    while left:
        p = min(left)
        left.remove(p)
        part, queue = {p}, [p]
        for r, c in queue:
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    part.add(q)
                    queue.append(q)
        result.append(part)
    return result


def _neighbors(cells, steps=D4):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - set(cells)


def _box(cells):
    return min(r for r, c in cells), max(r for r, c in cells) + 1, min(c for r, c in cells), max(c for r, c in cells) + 1


def _rectangle(box):
    r, e, c, d = box
    return {(y, x) for y in range(r, e) for x in range(c, d)}


def _shell(box):
    r, e, c, d = box
    return {(y, x) for y, x in _rectangle(box) if y in (r, e - 1) or x in (c, d - 1)}


def _move(cells, r, c):
    return {(y + r, x + c) for y, x in cells}


def _turn(cells, k):
    for _ in range(k):
        cells = {(-c, r) for r, c in cells}
    return cells


_CUP_NEIGHBORS = frozenset({(0,-1),(0,1),(1,-1),(1,0),(1,1)})
_CUPS = tuple(_turn(set(_CUP_NEIGHBORS), k) for k in range(4))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), 'invalid grid size'
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), 'invalid grid colors'
    return a


def _analyze(value):
    """Recover roles, cards, conducting components, and chamber coordinate frames."""
    a = _grid(value)
    h, w = a.shape
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), 'background rim required'
    cards, excluded = {}, set()
    for color in sorted(set(map(int, a.flat)) - {bg}):
        for cells in _components(zip(*np.where(a == color))):
            r, e, c, d = _box(cells)
            if not (4 <= e-r <= 6 and 4 <= d-c <= 6) or cells != _shell((r,e,c,d)):
                continue
            tile = a[r+1:e-1, c+1:d-1]
            inks = set(map(int, tile.flat)) - {bg}
            if len(inks) != 1 or color in inks:
                continue
            ink = next(iter(inks))
            assert color not in cards, 'duplicate card identity'
            rect = _rectangle((r,e,c,d))
            assert not (rect | _neighbors(rect, D8)) & excluded, 'touching cards'
            cards[color] = dict(mask=tile == ink, ink=ink, box=(r,e,c,d))
            excluded |= rect
    assert 1 <= len(cards) <= 3, 'need one to three pattern cards'
    scene = {(r,c) for r in range(h) for c in range(w)} - excluded
    markers, wire_colors = [], set()
    for p in sorted(scene):
        source = int(a[p])
        if source not in cards:
            continue
        adjacent = {q for q in _neighbors({p}, D8) if q in scene and a[q] != bg}
        colors = {int(a[q]) for q in adjacent}
        relative = {(r-p[0],c-p[1]) for r,c in adjacent}
        assert len(colors) == 1 and relative in _CUPS, 'identity outside a five-neighbor source cup'
        wire = next(iter(colors))
        assert wire not in cards and wire != bg
        markers.append((p, source))
        wire_colors.add(wire)
    assert {s for p,s in markers} == set(cards) and len(wire_colors) == 1, 'missing source cups'
    wire = next(iter(wire_colors))
    remaining = {int(a[p]) for p in scene} - {bg, wire} - set(cards)
    assert len(remaining) == 1, 'need one exclusive chamber wall color'
    wall = next(iter(remaining))
    assert wall not in {card['ink'] for card in cards.values()}
    assert wire not in {card['ink'] for card in cards.values()}
    wires = _components(p for p in scene if a[p] == wire)
    wire_at = {p: i for i, cells in enumerate(wires) for p in cells}
    supplies = {}
    for p, source in markers:
        ids = {wire_at[q] for q in _neighbors({p}) if q in wire_at}
        assert len(ids) == 1
        index = next(iter(ids))
        assert index not in supplies or supplies[index] == source, 'conflicting sources'
        supplies[index] = source
    cavities = []
    for cells in _components(p for p in scene if a[p] == bg):
        if any(r in (0,h-1) or c in (0,w-1) for r,c in cells):
            continue
        edge = _neighbors(cells)
        assert edge <= scene and all(a[p] in (wall,wire) for p in edge), 'unintended enclosure'
        ports = {p for p in edge if a[p] == wire}
        assert len(ports) <= 1 and any(a[p] == wall for p in edge), 'ambiguous chamber ports'
        port, q, inward, source = None, None, None, None
        if ports:
            port = next(iter(ports))
            inside = cells & _neighbors({port})
            assert len(inside) == 1, 'port must face exactly one interior pixel'
            q = next(iter(inside))
            inward = q[0]-port[0], q[1]-port[1]
            source = supplies.get(wire_at[port])
        cavities.append(dict(cells=cells, port=port, q=q, inward=inward, source=source))
    assert cavities and any(c['source'] is not None for c in cavities), 'no supplied chamber'
    # Card rectangles must be isolated, including their inactive pixels.
    for p in _neighbors(excluded, D8) & scene:
        assert a[p] == bg, 'card touches scene geometry'
    return dict(background=bg, wire=wire, wall=wall, cards=cards, markers=markers,
                cavities=cavities, wires=wires)


def _render(a, info, alternative=None):
    out = a.copy()
    bg = info['background']
    for cavity in info['cavities']:
        source = cavity['source']
        if alternative == 'all_ports' and source is None and cavity['port'] is not None:
            source = min(info['cards'])
        if source is None:
            continue
        if alternative == 'nearest':
            q = cavity['q']
            distances = [(abs(p[0]-q[0])+abs(p[1]-q[1]), s) for p,s in info['markers']]
            minimum = min(distance for distance,s in distances)
            closest = {s for distance,s in distances if distance == minimum}
            if len(closest) == 1:
                source = next(iter(closest))
        card = info['cards'][source]
        if alternative == 'recursive' and card['ink'] in info['cards']:
            card = info['cards'][card['ink']]
        mask, ink = card['mask'], card['ink']
        q, (dr,dc) = cavity['q'], cavity['inward']
        if alternative == 'upright':
            r,e,c,d = _box(cavity['cells'])
            q, dr, dc = (r,c), 1, 0
        elif alternative == 'no_rotation':
            dr,dc = 1,0
        tr,tc = -dc,dr
        if alternative == 'bbox_origin':
            u0 = min(r*dr+c*dc for r,c in cavity['cells'])
            v0 = min(r*tr+c*tc for r,c in cavity['cells'])
        if alternative == 'row_reset':
            row_starts = {}
            for r,c in cavity['cells']:
                u = (r-q[0])*dr+(c-q[1])*dc
                v = (r-q[0])*tr+(c-q[1])*tc
                row_starts[u] = min(v,row_starts.get(u,v))
        for r,c in cavity['cells']:
            u = (r-q[0])*dr + (c-q[1])*dc
            v = (r-q[0])*tr + (c-q[1])*tc
            if alternative == 'bbox_origin':
                u,v = r*dr+c*dc-u0,r*tr+c*tc-v0
            elif alternative == 'row_reset':
                v -= row_starts[u]
            if alternative == 'solid_source':
                out[r,c] = source
            elif alternative == 'solid_ink':
                out[r,c] = ink
            elif alternative == 'one_copy' and not (0 <= u < mask.shape[0] and 0 <= v < mask.shape[1]):
                out[r,c] = bg
            else:
                out[r,c] = ink if mask[u % mask.shape[0], v % mask.shape[1]] else bg
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = np.asarray(input_grid)
    return _render(a, _analyze(a))


def _ri(rng, low, high):
    return int(rng.integers(low, high+1))


def _pick(rng, choices):
    return choices[_ri(rng, 0, len(choices)-1)]


def _asymmetric(mask):
    if mask.shape[0] != mask.shape[1]:
        return False
    return all(not np.array_equal(mask, other) for other in
               [np.rot90(mask,k) for k in (1,2,3)] + [np.rot90(np.fliplr(mask),k) for k in range(4)])


def _tile(rng, informative=False):
    h,w = (3,3) if informative else (_ri(rng,2,4),_ri(rng,2,4))
    y,x = np.indices((h,w))
    for _ in range(100):
        kind = 0 if informative else _ri(rng,0,7)
        if kind == 0:
            mask = rng.random((h,w)) < float(rng.uniform(.25,.8))
        elif kind == 1:
            mask = (x+y) % 2 == 0
        elif kind == 2:
            mask = (x == 0) | (x == w-1) | (y == 0) | (y == h-1)
        elif kind == 3:
            mask = (x == w//2) | (y == h//2)
        elif kind == 4:
            mask = x % 2 == 0
        elif kind == 5:
            mask = (x-y) % max(h,w) == 0
        elif kind == 6:
            mask = rng.random((h,w)) < .9
        else:
            mask = np.ones((h,w),dtype=bool)
        if mask.any() and (not informative or (not mask.all() and _asymmetric(mask)
                    and mask.any(axis=0).all() and mask.any(axis=1).all())):
            return mask
    return np.array([[1,1,0],[0,1,0],[1,0,0]], dtype=bool)


def _shape(rng, h, w, kind):
    cells = {(r,c) for r in range(h) for c in range(w)}
    if kind == 'chamfer':
        cells -= {(0,0),(h-1,w-1)}
    elif kind == 'L':
        cells = {p for p in cells if p[0] < max(1,h//2) or p[1] < max(1,w//2)}
    elif kind == 'steps':
        cells = {(r,c) for r,c in cells if c >= min(w-1, (r*max(1,w-2))//max(1,h-1))}
    elif kind == 'waist':
        cells = {(r,c) for r,c in cells if r in (0,h-1) or c < max(1,w-1)}
    elif kind == 'diamond':
        cells = {(r,c) for r,c in cells if abs(2*r-h+1)/h + abs(2*c-w+1)/w <= 1.05}
    if not cells or len(_components(cells)) != 1:
        cells = {(r,c) for r in range(h) for c in range(w)}
    # Full box shells also reserve corners, making the wall easy to recognize.
    edge = _neighbors(cells)
    if kind == 'rectangle':
        edge = _shell((-1,h+1,-1,w+1))
    r,e,c,d = _box(cells | edge)
    return _move(cells,-r,-c), _move(edge,-r,-c)


def _cup(k):
    wire = _turn(set(_CUP_NEIGHBORS),k)
    source = {(0,0)}
    exit_cell = next(iter(_turn({(2,0)},k)))
    r,e,c,d = _box(wire | source)
    return _move(source,-r,-c), _move(wire,-r,-c), (exit_cell[0]-r,exit_cell[1]-c)


def _place(rng, cells, occupied, shape):
    """Pack full object bounding rectangles with at least one blank-cell gap."""
    r,e,c,d = _box(cells)
    hh,ww = e-r,d-c
    if hh > shape[0]-2 or ww > shape[1]-2:
        return None
    for _ in range(140):
        rr,cc = _ri(rng,1,shape[0]-hh-1),_ri(rng,1,shape[1]-ww-1)
        rect = _rectangle((rr,rr+hh,cc,cc+ww))
        if not (rect | _neighbors(rect,D8)) & occupied:
            return rr-r,cc-c,rect
    return None


def _bfs(rng, start, goals, allowed):
    directions = [D4[int(i)] for i in rng.permutation(4)]
    queue = deque([start])
    previous = {start:None}
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr,dc in directions:
            q = p[0]+dr,p[1]+dc
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        return None
    path = set()
    while end is not None:
        path.add(end)
        end = previous[end]
    return path


def _ports(cells, edge, inward):
    dr,dc = inward
    result = []
    for p in sorted(edge):
        inside = _neighbors({p}) & cells
        q = p[0]+dr,p[1]+dc
        if inside != {q}:
            continue
        outside = p[0]-dr,p[1]-dc
        outside2 = p[0]-2*dr,p[1]-2*dc
        if outside not in cells | edge and outside2 not in cells | edge | _neighbors(edge,D8):
            result.append((p,q,outside))
    return result


def _scene(rng, recipe):
    sources, masks, mapping = recipe['sources'],recipe['masks'],recipe['mapping']
    minimum = 28 if recipe['dense'] else 24
    h,w = _ri(rng,minimum,30),_ri(rng,minimum,30)
    bg,wire,wall = recipe['colors']
    a = np.full((h,w),bg,dtype=np.int8)
    board = {(r,c) for r in range(1,h-1) for c in range(1,w-1)}
    occupied, protected = set(),set()
    cards = []
    objects = []
    for i,owner in enumerate(recipe['owners']):
        if recipe['large'] and i == 0:
            hh,ww = _ri(rng,7,9),_ri(rng,3,6)
            if rng.random() < .5:
                hh,ww = ww,hh
        elif recipe['lesson'] == 0 and i == 0:
            hh,ww = _ri(rng,4,6),_ri(rng,4,6)
        else:
            hi = 5 if recipe['dense'] else 6
            hh,ww = _ri(rng,2,hi),_ri(rng,2,hi)
        kind = 'rectangle' if i == 0 and recipe['lesson'] == 0 else _pick(rng,('rectangle','rectangle','chamfer','L','steps','waist','diamond'))
        if recipe['lesson'] == 0 and i == 1:
            kind = 'chamfer'
            hh,ww = max(3,hh),max(3,ww)
        if recipe['lesson'] == 0 and owner is None:
            kind = 'rectangle'
            hh,ww = _ri(rng,3,4),_ri(rng,3,4)
        if recipe['large'] and i == 0:
            kind = _pick(rng,('rectangle','chamfer','L','waist'))
        cells,edge = _shape(rng,hh,ww,kind)
        objects.append(dict(cells=cells,edge=edge,owner=owner,index=i))
    # Largest objects first improves bounded packing without imposing scene order.
    items = [('chamber', o) for o in objects] + [('card', s) for s in sources]
    rng.shuffle(items)
    def area(item):
        kind,o = item
        if kind == 'card':
            return (masks[o].shape[0]+2)*(masks[o].shape[1]+2)
        r,e,c,d = _box(o['cells']|o['edge'])
        return (e-r)*(d-c)
    items.sort(key=area, reverse=True)
    for kind,obj in items:
        if kind == 'card':
            mh,mw = masks[obj].shape
            shape_cells = _rectangle((0,mh+2,0,mw+2))
        else:
            shape_cells = obj['cells'] | obj['edge']
        placed = _place(rng,shape_cells,occupied,(h,w))
        if placed is None:
            return None
        r,c,rect = placed
        occupied |= rect
        protected |= rect | _neighbors(rect,D8)
        if kind == 'card':
            box = (r,r+mh+2,c,c+mw+2)
            for p in _shell(box):
                a[p] = obj
            tile = a[r+1:r+mh+1,c+1:c+mw+1]
            tile[masks[obj]] = mapping[obj]
            cards.append(box)
        else:
            obj['cells'] = _move(obj['cells'],r,c)
            obj['edge'] = _move(obj['edge'],r,c)
            for p in obj['edge']:
                a[p] = wall
    route_sets, marker_positions, exits = [],[],[]
    for source in sources:
        k = _ri(rng,0,3)
        src,cup,exit_cell = _cup(k)
        placed = _place(rng,src|cup|{exit_cell},occupied,(h,w))
        if placed is None:
            return None
        r,c,rect = placed
        occupied |= rect
        src,cup = _move(src,r,c),_move(cup,r,c)
        cup_rect = _rectangle(_box(src | cup))
        protected |= cup_rect | _neighbors(cup_rect,D8)
        exit_cell = exit_cell[0]+r,exit_cell[1]+c
        for p in src:
            a[p] = source
        marker_positions.append(next(iter(src)))
        route_sets.append(cup | {exit_cell})
        exits.append(exit_cell)
    # Choose actual entry directions, with two minimal-count lessons covering all four.
    inactive_wires = set()
    for obj in objects:
        i,owner = obj['index'],obj['owner']
        if owner is None and recipe['lesson'] != 0 and rng.random() < .45:
            obj.update(port=None,q=None,inward=None)
            continue
        if recipe['lesson'] == 0 and i < 2:
            directions = [((1,0),(0,1))[i]]
        elif recipe['lesson'] == 1 and i < 2:
            directions = [((-1,0),(0,-1))[i]]
        else:
            directions = [D4[int(j)] for j in rng.permutation(4)]
        options = []
        own_box = _rectangle(_box(obj['cells'] | obj['edge']))
        other = protected - (own_box | _neighbors(own_box,D8))
        for direction in directions:
            candidates = _ports(obj['cells'],obj['edge'],direction)
            if recipe['lesson'] == 0 and i == 0:
                tr,tc = -direction[1],direction[0]
                projections = [r*tr+c*tc for r,c in obj['cells']]
                candidates = [(p,q,outside) for p,q,outside in candidates
                              if min(projections) < q[0]*tr+q[1]*tc < max(projections)
                              and (q[0]*tr+q[1]*tc-min(projections)) % 3 != 0]
            rng.shuffle(candidates)
            for port,q,approach in candidates:
                dr,dc = direction
                extension = {(port[0]-j*dr,port[1]-j*dc) for j in range(1,3)}
                if extension <= board and not extension & other:
                    options.append((port,q,approach,direction))
        if not options:
            return None
        obj['options'] = options
        port,q,approach,direction = options[0]
        obj.update(port=port,q=q,inward=direction,approach=approach)
        if owner is None:
            dr,dc = direction
            tail = {port,(port[0]-dr,port[1]-dc),(port[0]-2*dr,port[1]-2*dc)}
            inactive_wires |= tail
    for owner in rng.permutation(len(sources)):
        owner = int(owner)
        own = route_sets[owner]
        for obj in objects:
            if obj['owner'] != owner:
                continue
            others = inactive_wires | set().union(*(s for j,s in enumerate(route_sets) if j != owner))
            forbidden = protected | others | _neighbors(others,D8)
            goals = own - protected | {exits[owner]}
            path = None
            for port,q,start,direction in obj['options']:
                if {port,start} & (others | _neighbors(others,D8)):
                    continue
                allowed = (board - forbidden) | goals | {start}
                path = _bfs(rng,start,goals,allowed)
                if path is not None:
                    obj.update(port=port,q=q,inward=direction,approach=start)
                    break
            if path is None:
                return None
            own |= path | {obj['port']}
    for p in inactive_wires | set().union(*route_sets):
        if a[p] not in (bg,wall):
            return None
        a[p] = wire
    # Paint using intended geometric ownership, without invoking solve.
    b = a.copy()
    for obj in objects:
        owner = obj['owner']
        if owner is None:
            continue
        source = sources[owner]
        mask,ink = masks[source],mapping[source]
        r0,c0 = obj['q']
        dr,dc = obj['inward']
        for r,c in obj['cells']:
            row = (r-r0)*dr+(c-c0)*dc
            col = -(r-r0)*dc+(c-c0)*dr
            if mask[row % mask.shape[0],col % mask.shape[1]]:
                b[r,c] = ink
    # Trim only unused outer space; retain at least a one-cell background rim.
    cells = set(zip(*np.where(a != bg)))
    r,e,c,d = _box(cells)
    r,c = max(0,r-_ri(rng,1,3)),max(0,c-_ri(rng,1,3))
    e,d = min(h,e+_ri(rng,1,3)),min(w,d+_ri(rng,1,3))
    a,b = a[r:e,c:d].copy(),b[r:e,c:d].copy()
    if np.array_equal(a,b):
        return None
    try:
        info = _analyze(a)
        if len(info['cavities']) != len(objects) or not np.array_equal(_render(a,info),b):
            return None
        if recipe['lesson'] == 1 and np.array_equal(_render(a,info,'nearest'),b):
            return None
        if recipe['lesson'] == 0:
            needed = set(_ALTERNATIVES) - {'nearest','recursive'}
            if not needed <= {name for name,yes in _witnesses(a,info,b).items() if yes}:
                return None
    except (AssertionError,KeyError):
        return None
    return dict(input=a,output=b)


def _recipe(rng, split, index, shift):
    lesson = index if split == 'train' and index < 2 else -1
    dense = split == 'test' and shift == 'network'
    if lesson == 0:
        k,active,inactive = 1,2,1
    elif lesson == 1:
        k,active,inactive = 2,3,_ri(rng,0,1)
    elif dense:
        k,active,inactive = _ri(rng,1,3),_ri(rng,4,5),_ri(rng,0,1)
    else:
        k = _ri(rng,1,3)
        active = _ri(rng,k,3)
        inactive = _ri(rng,0,2)
    palette = list(map(int,rng.permutation(10)))
    bg,wire,wall = palette[:3]
    sources = palette[3:3+k]
    targets = palette[3+k:]
    mode = 'chain' if lesson == 1 and rng.random() < .5 else 'cycle' if lesson == 1 else _pick(rng,('disjoint','chain','cycle','many','mixed'))
    order = [int(i) for i in rng.permutation(k)]
    mapping = {}
    for rank,j in enumerate(order):
        s = sources[j]
        if mode in ('chain','cycle') and rank < k-1:
            ink = sources[order[rank+1]]
        elif mode == 'cycle' and k > 1:
            ink = sources[order[0]]
        elif mode == 'many':
            ink = targets[0]
        elif mode == 'mixed':
            ink = _pick(rng,[v for v in palette[3:] if v != s])
        else:
            ink = _pick(rng,targets)
        mapping[s] = ink
    masks = {}
    for j,s in enumerate(sources):
        if j and lesson != 1 and rng.random() < .15:
            masks[s] = masks[sources[0]].copy()
        else:
            masks[s] = _tile(rng,informative=lesson == 0 or lesson == 1)
    owners = list(range(k)) + [_ri(rng,0,k-1) for _ in range(active-k)]
    if lesson == 1:
        owners = [0,1,0]
    elif lesson < 0:
        rng.shuffle(owners)
    owners += [None]*inactive
    return dict(sources=sources,masks=masks,mapping=mapping,colors=(bg,wire,wall),
                owners=owners,lesson=lesson,large=split == 'test' and shift == 'extent',dense=dense)


def _count(value, low, high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'count must be an integer in {low}..{high}')
    return int(value)


def _key(a):
    a = np.asarray(a,dtype=np.int8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ne = _count(num_train,2,10),_count(num_test,1,4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = _pick(rng,('extent','network'))
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',nt),('test',ne)):
        for index in range(count):
            for attempt in range(96):
                # Resample the recipe as well as packing; never drop a lesson or shift.
                recipe = _recipe(rng,split,index,shift)
                pair = _scene(rng,recipe)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError(f'bounded packing exhausted at seed {seed}, {split} pair {index}')
    validate_generated_puzzle(puzzle)
    return puzzle


_ALTERNATIVES = ('all_ports','solid_source','solid_ink','upright','no_rotation',
                 'bbox_origin','one_copy','row_reset','nearest','recursive')


def _witnesses(a, info, b):
    return {name:not np.array_equal(_render(a,info,name),b) for name in
            _ALTERNATIVES}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen,directions,witnesses = set(),set(),set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            a,b = _grid(pair['input']),_grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            assert np.array_equal(_render(a,info),b), 'input-derived output mismatch'
            assert not np.array_equal(a,b), 'uninformative unchanged example'
            assert np.all(a[a != b] == info['background']), 'changed existing foreground'
            assert set(map(int,b.flat)) <= set(map(int,a.flat))
            if split == 'train':
                directions.update(c['inward'] for c in info['cavities'] if c['source'] is not None)
                witnesses.update(name for name,yes in _witnesses(a,info,b).items() if yes)
    assert directions == set(D4), 'training must teach all entry directions'
    assert witnesses == set(_ALTERNATIVES), 'missing shortcut counterexample'


def measure_puzzle(puzzle):
    records = []
    for split in ('train','test'):
        for index,pair in enumerate(puzzle[split]):
            a,b = np.asarray(pair['input']),np.asarray(pair['output'])
            info = _analyze(a)
            active = [c for c in info['cavities'] if c['source'] is not None]
            counts = [sum(c['source'] == s for c in active) for s in info['cards']]
            records.append(dict(split=split,index=index,shape=list(a.shape),background=info['background'],
                sources=len(info['cards']),chambers=len(info['cavities']),supplied=len(active),
                unsupplied=len(info['cavities'])-len(active),branch_max=max(counts),
                tile_sizes=[list(card['mask'].shape) for card in info['cards'].values()],
                disconnected=sum(len(_components(zip(*np.where(card['mask'])))) > 1 for card in info['cards'].values()),
                overlap=sum(card['ink'] in info['cards'] for card in info['cards'].values()),
                directions=[list(c['inward']) for c in active],
                cavity_sizes=[[e-r,d-c] for r,e,c,d in (_box(c['cells']) for c in info['cavities'])],
                nonrectangular=sum(len(c['cells']) != (lambda z:(z[1]-z[0])*(z[3]-z[2]))(_box(c['cells'])) for c in info['cavities']),
                changed=int(np.count_nonzero(a != b)),witnesses=_witnesses(a,info,b)))
    return records


def _manual_fixture():
    a = np.zeros((15,19),dtype=np.int8)
    for p in _shell((1,5,1,6)):
        a[p] = 1
    a[2:4,2:5] = np.array([[2,0,2],[0,2,0]],dtype=np.int8)
    for p in _shell((4,10,9,16)) | _shell((6,10,1,6)):
        a[p] = 8
    a[11,2] = 1
    for p in _move(set(_CUP_NEIGHBORS),11,2):
        a[p] = 5
    a[13,2:8] = 5
    a[6:14,7] = 5
    a[6,8:10] = 5
    b = a.copy()
    b[5:9,10:15] = np.array([[0,2,0,2,0],[2,0,2,0,2],[2,0,2,0,2],[0,2,0,2,0]],dtype=np.int8)
    return a,b


def self_test():
    """Bounded streaming checks on the supplied 42 seeds; no disk writes."""
    a,b = _manual_fixture()
    assert np.array_equal(solve(a),b), 'manual right-entry/negative-phase fixture'
    rows,signatures,shapes,structures,tiles = [],set(),set(),set(),set()
    first = None
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        if seed == 0:
            first = p
        token = hashlib.sha256()
        rows.extend(measure_puzzle(p))
        for split in ('train','test'):
            for pair in p[split]:
                aa,bb = pair['input'],pair['output']
                token.update(bytes(aa.shape)+aa.tobytes()+bb.tobytes())
                shapes.add(aa.shape)
                roles,normalized = {},[]
                for v in aa.flat:
                    normalized.append(roles.setdefault(int(v),len(roles)))
                structures.add((aa.shape,bytes(normalized)))
                for card in _analyze(aa)['cards'].values():
                    tiles.add((card['mask'].shape,card['mask'].tobytes()))
        signatures.add(token.digest())
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 20 and len(tiles) >= 60
    assert {r['sources'] for r in rows} == {1,2,3}
    assert max(r['branch_max'] for r in rows) >= 3
    assert sum(r['disconnected'] for r in rows) >= 20
    assert sum(r['nonrectangular'] for r in rows) >= 40
    assert {r['background'] for r in rows} == set(range(10))
    assert any(max(max(size) for size in r['cavity_sizes']) >= 7 for r in rows if r['split'] == 'test')
    assert any(r['supplied'] >= 4 for r in rows if r['split'] == 'test')
    again = generate_puzzle(0)
    assert all(_key(x[role]) == _key(y[role]) for split in first for x,y in zip(first[split],again[split]) for role in ('input','output'))
    # All legal explicit counts are checked externally; exercise extremes here.
    for nt,ne in ((2,1),(10,1),(10,4),(5,4)):
        p = generate_puzzle(712+nt+ne,nt,ne)
        assert len(p['train']) == nt and len(p['test']) == ne
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,True),(2.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Recoloring every visible role must commute with solving.
    permutation = np.random.default_rng(837).permutation(10)
    pair = first['train'][1]
    assert np.array_equal(solve(permutation[pair['input']]),permutation[pair['output']])
    # Move the card in the manual fixture; route ownership and output paint stay put.
    moved = a.copy()
    moved[1:5,1:6] = 0
    moved[1:5,2:7] = a[1:5,1:6]
    expected = b.copy()
    expected[1:5,1:6] = 0
    expected[1:5,2:7] = a[1:5,1:6]
    assert np.array_equal(solve(moved),expected)
    bad = {s:[{role:value.copy() for role,value in p.items()} for p in first[s]] for s in first}
    pair = bad['test'][0]
    r,c = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][r,c] = pair['input'][r,c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('partial incorrect texture accepted')
    return dict(seeds=42,pairs=len(rows),distinct_puzzles=len(signatures),grid_shapes=len(shapes),
                color_normalized_structures=len(structures),distinct_tiles=len(tiles),
                disconnected_tiles=sum(r['disconnected'] for r in rows),
                nonrectangular_chambers=sum(r['nonrectangular'] for r in rows),
                max_branch=max(r['branch_max'] for r in rows))
