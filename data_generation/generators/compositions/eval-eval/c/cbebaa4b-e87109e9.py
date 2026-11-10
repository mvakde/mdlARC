"""Turn-gated docking: geometry places tiles; relative turns activate their ports.

Only standard library and NumPy are used. The input-only solver first decodes
visible objects/ports, then follows activation from the sole unkeyed source.
It never consults generation metadata, a seed, or demonstration outputs.
"""
from __future__ import annotations

from itertools import combinations
import json

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W
HOST_SEEDS = (0,169205579,39839368,363526083,826848778,766343319,
              178813425,901475980,666797806,935880894,144996037,577730060,
              60251431,989760254,370988197,858038328,320187286,665916485,
              927804295,568162409,684081539,294652647,973522036,213196547,
              674958853,762949741,835778432,378895632,982106308,109560900,
              296999702,71844158,656981893,311406186,639077809,160494295,
              642050185,459161898,442395139,246747622,19342516,361580583)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape), "invalid grid dimensions")
    _require(np.issubdtype(array.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= array) & (array <= 9)), "invalid grid color")
    return array.astype(np.int8, copy=False)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _box(cells):
    return min(r for r,c in cells), min(c for r,c in cells), max(r for r,c in cells), max(c for r,c in cells)


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r,c in cells}


def _normal(cells):
    t,l,_,_ = _box(cells)
    return tuple(sorted((r-t,c-l) for r,c in cells))


def _rectangle(t, l, h, w):
    return {(r,c) for r in range(t,t+h) for c in range(l,l+w)}


def _signature(grid):
    return grid.shape, grid.tobytes()


def _decode_header(grid):
    frame = int(grid[0,0])
    _require(np.all(grid[0] == frame), "missing top frame")
    bottoms = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing bottom frame")
    bottom = int(bottoms[0])
    _require(2 <= bottom < grid.shape[0]-1, "invalid header height")
    band = grid[1:bottom]
    _require(np.all(band == band[0]), "inconsistent key bars")
    row = band[0]
    _require(row[0] == row[-1] == frame, "missing key walls")
    runs, col = [], 0
    while col < len(row):
        if row[col] == frame:
            col += 1
            continue
        left = col
        while col < len(row) and row[col] != frame:
            col += 1
        runs.append((left,col))
    _require(2 <= len(runs) <= 5 and all(b-a >= 3 for a,b in runs), "ambiguous panels")
    background = int(row[runs[0][0]+1])
    turns, widths = {}, []
    for a,b in runs:
        _require(np.all(row[a+1:b-1] == background), "inconsistent panel interiors")
        left, right = int(row[a]), int(row[b-1])
        _require((left == background) != (right == background), "ambiguous turn bar")
        color, turn = (right,1) if left == background else (left,-1)
        _require(color not in turns and color not in (frame,background), "conflicting key")
        turns[color] = turn
        widths.append(b-a+2)
    return bottom+1, frame, background, turns, tuple(widths)


def _decode(input_grid):
    grid = _grid(input_grid)
    crop,frame,bg,turns,widths = _decode_header(grid)
    field = grid[crop:]
    components = _components(map(tuple, np.argwhere(field != bg).tolist()))
    _require(4 <= len(components) <= 10, "need four through ten separated tiles")
    cell_owner = {q:i for i,cells in enumerate(components) for q in cells}
    _require(all(cell_owner.get((r+dr,c+dc),i)==i
                 for (r,c),i in cell_owner.items()
                 for dr in (-1,0,1) for dc in (-1,0,1)),
             "input tiles need an eight-neighbor separation buffer")
    palettes = [{int(field[q]) for q in cells} for cells in components]
    common = set.intersection(*palettes)
    _require(len(common) == 1 and all(len(p) == 2 for p in palettes), "tiles lack a common connector")
    marker = common.pop()
    _require(marker not in (bg,frame) and marker not in turns, "invalid connector role")
    pieces, sources = [], []
    for cells, colors in zip(components, palettes):
        color = (colors - {marker}).pop()
        _require(color != frame, "payload reuses the frame color")
        body = {q for q in cells if field[q] == color}
        pins = cells - body
        t,l,b,r = _box(body)
        _require(b-t >= 2 and r-l >= 2, "tile body too small")
        _require(len(_components(body)) == 1, "disconnected payload")
        boundary = {q for q in _rectangle(t,l,b-t+1,r-l+1)
                    if q[0] in (t,b) or q[1] in (l,r)}
        _require(boundary <= body, "tile needs a complete rectangular rim")
        ports = {}
        for q in pins:
            x,y = q
            if x == t-1 and l <= y <= r:
                d = 0
            elif y == r+1 and t <= x <= b:
                d = 1
            elif x == b+1 and l <= y <= r:
                d = 2
            elif y == l-1 and t <= x <= b:
                d = 3
            else:
                raise ValueError("connector is not immediately outside a side")
            ports.setdefault(d,set()).add(q)
        _require(all(2 <= len(p) <= 3 for p in ports.values()), "each port has two or three teeth")
        _require(1 <= len(ports) <= 4, "invalid port count")
        i = len(pieces)
        if color not in turns:
            sources.append(i)
            _require(b-t == r-l and len(body) == (b-t+1)**2, "source must be a solid square")
        pieces.append(dict(color=color,body=body,pins=pins,cells=cells,ports=ports,box=(t,l,b,r)))
    _require(len(sources) == 1, "source must be the sole unkeyed payload")
    source = sources[0]
    groups = {}
    for i,piece in enumerate(pieces):
        for d,pins in piece['ports'].items():
            groups.setdefault(_normal(pins),[]).append((i,d))
    links = {}
    for key,ends in groups.items():
        _require(len(ends) == 2, "a docking geometry must occur exactly twice")
        (i,d),(j,e) = ends
        _require(i != j and e == (d+2)%4, "matching ports must face opposite ways")
        links[i,d] = (j,e)
        links[j,e] = (i,d)
    _require(len(links)//2 == len(pieces)-1, "potential docks must form a tree")
    # Recover the rooted *potential* graph to validate inlet/outlet conventions.
    # Active reconstruction below deliberately does not place dormant branches.
    parent = {source:None}
    arrivals = {}
    queue = [source]
    for i in queue:
        for d in pieces[i]['ports']:
            j,e = links[i,d]
            if j == parent[i]:
                continue
            _require(j not in parent, "cyclic docking graph")
            parent[j] = i
            arrivals[j] = d
            queue.append(j)
    _require(len(parent) == len(pieces), "disconnected docking graph")
    for i,d in arrivals.items():
        allowed = {(d+2)%4,(d-1)%4,(d+1)%4}
        _require(set(pieces[i]['ports']) <= allowed, "router has a straight-through outlet")
    return dict(grid=grid,field=field,crop=crop,frame=frame,background=bg,
                marker=marker,turns=turns,widths=widths,pieces=pieces,source=source,
                links=links,parent=parent,arrivals=arrivals)


def _assemble(info, turn_override=None, all_branches=False, keep_scatter=False):
    """Joint activation/placement, one port at a time; no complete-assembly stage."""
    pieces, source = info['pieces'], info['source']
    turns = info['turns'] if turn_override is None else turn_override
    positions, depths, events = {source:(0,0)}, {source:0}, []
    frontier = [(source,d) for d in sorted(pieces[source]['ports'])]
    for i,d in frontier:
        j,e = info['links'][i,d]
        _require(j not in positions, "an activated tile was reached twice")
        a = min(pieces[i]['ports'][d])
        b = min(pieces[j]['ports'][e])
        di = positions[i]
        delta = a[0]+di[0]-b[0], a[1]+di[1]-b[1]
        _require(_shift(pieces[j]['ports'][e],delta) == _shift(pieces[i]['ports'][d],di),
                 "incomplete geometric dock")
        positions[j] = delta
        depths[j] = depths[i]+1
        outgoing = (d + turns[pieces[j]['color']]) % 4
        outlets = set(pieces[j]['ports']) - {e}
        if outlets:
            events.append((j,d,turns[pieces[j]['color']],outgoing,outgoing in outlets))
        chosen = sorted(outlets) if all_branches else ([outgoing] if outgoing in outlets else [])
        frontier.extend((j,f) for f in chosen)
    output = np.full(info['field'].shape, info['background'], dtype=np.int8)
    owners = {}
    payload = set()
    for i,delta in positions.items():
        if keep_scatter:
            delta = (0,0)
        p = pieces[i]
        body, pins = _shift(p['body'],delta), _shift(p['pins'],delta)
        for r,c in body | pins:
            _require(0 <= r < output.shape[0] and 0 <= c < output.shape[1], "assembly outside field")
        _require(not body & (payload | set(owners)) and not pins & payload, "payload collision")
        payload |= body
        for q in body:
            output[q] = p['color']
        for q in pins:
            owners.setdefault(q,[]).append(i)
            _require(len(owners[q]) <= 2, "three tiles share a connector")
    for q in owners:
        output[q] = info['marker']
    return output, dict(positions=positions,depths=depths,events=events,owners=owners)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the local table and follow geometry-matched, turn-selected docks."""
    return _assemble(_decode(input_grid))[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _attach(rng, nodes, parent, direction, used_keys, limits, force_two=False):
    """Grow one rigid tile with a shared sparse side port, using uneven offsets."""
    p = nodes[parent]
    if direction in p['ports']:
        return None
    t,l,b,r = _box(p['rect'])
    for _ in range(90):
        h,w = int(rng.integers(3,7)), int(rng.integers(3,7))
        if direction in (1,3):
            top = int(rng.integers(t-h+2,b))
            left = r+2 if direction == 1 else l-w-1
            span = list(range(max(t,top),min(b,top+h-1)+1))
        else:
            left = int(rng.integers(l-w+2,r))
            top = b+2 if direction == 2 else t-h-1
            span = list(range(max(l,left),min(r,left+w-1)+1))
        if len(span) < 2:
            continue
        size = 2 if force_two or rng.random() < .86 or len(span) < 3 else 3
        values = sorted(int(v) for v in rng.choice(span,size,replace=False))
        if direction == 0:
            pins = {(t-1,v) for v in values}
        elif direction == 1:
            pins = {(v,r+1) for v in values}
        elif direction == 2:
            pins = {(b+1,v) for v in values}
        else:
            pins = {(v,l-1) for v in values}
        key = _normal(pins)
        if key in used_keys:
            continue
        rect = _rectangle(top,left,h,w)
        cells = rect | pins
        if cells & (p['rect'] | set().union(*p['ports'].values()) if p['ports'] else p['rect']):
            continue
        if any(cells & (z['rect'] | set().union(*z['ports'].values()))
               for i,z in enumerate(nodes) if i != parent):
            continue
        total = cells | set().union(*(z['rect'] | set().union(*z['ports'].values()) for z in nodes))
        a,c,e,f = _box(total)
        if e-a+1 > limits[0] or f-c+1 > limits[1]:
            continue
        used_keys.add(key)
        p['ports'][direction] = pins
        node = dict(rect=rect,ports={(direction+2)%4:set(pins)},parent=parent,arrival=direction)
        nodes.append(node)
        return len(nodes)-1
    return None


def _active_nodes(nodes, turns):
    active, depths = {0}, {0:0}
    for j in range(1,len(nodes)):
        p = nodes[j]['parent']
        if p in active and (p == 0 or nodes[j]['arrival'] == (nodes[p]['arrival']+turns[nodes[p]['color']])%4):
            active.add(j)
            depths[j] = depths[p]+1
    return active, depths


def _layout(rng, count, depth, direction, first_turn, colors, turns, limits):
    k = int(rng.integers(3,7))
    nodes = [dict(rect=_rectangle(0,0,k,k),ports={},parent=None,arrival=None,color=None)]
    keys = set()
    fork = _attach(rng,nodes,0,direction,keys,limits,True)
    if fork is None:
        return None
    nodes[fork]['color'] = colors[0]
    chosen = (direction+first_turn)%4
    main = _attach(rng,nodes,fork,chosen,keys,limits,True)
    other = _attach(rng,nodes,fork,(direction-first_turn)%4,keys,limits,True)
    if main is None or other is None:
        return None
    nodes[main]['color'] = colors[1]
    nodes[other]['color'] = _pick(rng,colors)
    last = main
    for step in range(2,depth):
        d = (nodes[last]['arrival']+turns[nodes[last]['color']])%4
        child = _attach(rng,nodes,last,d,keys,limits,force_two=step==2)
        if child is None:
            return None
        nodes[child]['color'] = _pick(rng,colors)
        last = child
    while len(nodes) < count:
        active, depths = _active_nodes(nodes,turns)
        slots = []
        for i,p in enumerate(nodes):
            allowed = range(4) if i == 0 else [(p['arrival']-1)%4,(p['arrival']+1)%4]
            for d in allowed:
                if d in p['ports']:
                    continue
                # Keep the main depth as the maximum in training and tests.
                if i in active and i != 0 and d == (p['arrival']+turns[p['color']])%4 and depths[i] >= depth:
                    continue
                slots.append((i,d))
        rng.shuffle(slots)
        for i,d in slots:
            j = _attach(rng,nodes,i,d,keys,limits)
            if j is not None:
                nodes[j]['color'] = _pick(rng,colors)
                break
        else:
            return None
    return nodes


def _body(rng, node, source=False):
    rect = node['rect']
    if source:
        return set(rect)
    t,l,b,r = _box(rect)
    style = int(rng.integers(4))
    thickness = 2 if style == 2 else 1
    rim = {q for q in rect if min(q[0]-t,b-q[0],q[1]-l,r-q[1]) < thickness}
    if style == 0:
        return set(rect)
    if style == 3:
        target = int(rng.integers(0,len(rect-rim)+1))
        for _ in range(target):
            frontier = sorted(q for q in rect-rim if any(
                (q[0]+dr,q[1]+dc) in rim for dr,dc in DIRECTIONS))
            if not frontier:
                break
            rim.add(_pick(rng,frontier))
    return rim


def _scatter(rng, shape, pieces, background, marker):
    h,w = shape
    root = pieces[0]['body'] | pieces[0]['pins']
    for _ in range(14):
        forbidden = {(r+dr,c+dc) for r,c in root for dr in (-1,0,1) for dc in (-1,0,1)}
        positions = {0:(0,0)}
        order = list(range(1,len(pieces)))
        rng.shuffle(order)
        order.sort(key=lambda i:-len(pieces[i]['body'] | pieces[i]['pins']))
        for i in order:
            cells = pieces[i]['body'] | pieces[i]['pins']
            t,l,b,r = _box(cells)
            for _ in range(280):
                delta = int(rng.integers(h-(b-t)))-t, int(rng.integers(w-(r-l)))-l
                if delta == (0,0):
                    continue
                moved = _shift(cells,delta)
                if moved & forbidden:
                    continue
                positions[i] = delta
                forbidden |= {(a+dr,c+dc) for a,c in moved for dr in (-1,0,1) for dc in (-1,0,1)}
                break
            else:
                break
        if len(positions) == len(pieces):
            grid = np.full(shape,background,dtype=np.int8)
            for i,p in enumerate(pieces):
                for q in _shift(p['body'],positions[i]):
                    grid[q] = p['color']
                for q in _shift(p['pins'],positions[i]):
                    grid[q] = marker
            return grid
    return None


def _header(rng, width, height, colors, turns, bg, frame):
    widths = np.full(len(colors),5,dtype=int)
    widths += rng.multinomial(width-5*len(colors),np.full(len(colors),1/len(colors)))
    band = np.full((height,width),bg,dtype=np.int8)
    band[[0,-1]] = frame
    c = 0
    for color,size in zip(rng.permutation(colors),widths):
        band[:,c] = band[:,c+size-1] = frame
        band[1:-1,c+1 if turns[int(color)] == -1 else c+size-2] = color
        c += int(size)
    return band


def _generate_pair(rng, params, count, depth, direction, first_turn):
    bg,marker,source,frame = params['roles']
    nkeys = int(rng.integers(2,6))
    colors = params['colors'][:nkeys]
    turns = {color:int(rng.choice([-1,1])) for color in colors}
    turns[colors[0]], turns[colors[1]] = first_turn, -first_turn
    for _ in range(160):
        hh = int(rng.integers(3,6))
        nodes = _layout(rng,count,depth,direction,first_turn,colors,turns,(30-hh,30))
        if nodes is None:
            continue
        total = set().union(*(p['rect'] | set().union(*p['ports'].values()) for p in nodes))
        t,l,b,r = _box(total)
        bh,bw = b-t+1,r-l+1
        h = int(rng.integers(max(bh,19),31-hh))
        w = int(rng.integers(max(bw,19,5*nkeys),31))
        offset = int(rng.integers(h-bh+1))-t, int(rng.integers(w-bw+1))-l
        pieces = [dict(body=_shift(_body(rng,p,i==0),offset),
                       pins=_shift(set().union(*p['ports'].values()),offset),
                       color=source if i==0 else p['color']) for i,p in enumerate(nodes)]
        inp_field = _scatter(rng,(h,w),pieces,bg,marker)
        if inp_field is None:
            continue
        header = _header(rng,w,hh,colors,turns,bg,frame)
        inp = np.vstack((header,inp_field))
        info = _decode(inp)
        out,state = _assemble(info)
        # Ground truth is drawn from construction's active rigid objects, rather
        # than copied from solve. Agreement is subsequently checked explicitly.
        active, depths = _active_nodes(nodes,turns)
        expected = np.full((h,w),bg,dtype=np.int8)
        for i in active:
            for q in pieces[i]['body']:
                expected[q] = pieces[i]['color']
        for i in active:
            for q in pieces[i]['pins']:
                expected[q] = marker
        _require(np.array_equal(expected,out), "construction and input reconstruction disagree")
        if max(state['depths'].values()) != depth:
            continue
        return dict(input=inp,output=expected)
    raise RuntimeError("bounded docking construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be 1 through 4 and smaller than num_train")
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    params = dict(roles=tuple(palette[:4]),colors=palette[4:])
    heading, turn = int(rng.integers(4)), int(rng.choice([-1,1]))
    puzzle, seen = {'train':[],'test':[]}, set()
    for split,count in [('train',int(num_train)),('test',int(num_test))]:
        for i in range(count):
            depth = 3 if split == 'train' and i < 2 else int(rng.integers(2,4)) if split == 'train' else int(rng.integers(4,6))
            pieces = int(rng.integers(max(5 if split=='train' else 6,depth+2),8 if split=='train' else 10))
            d = (heading if i == 0 else (heading-turn)%4) if split == 'train' and i < 2 else int(rng.integers(4))
            side = turn if split == 'train' and i == 0 else -turn if split == 'train' and i == 1 else int(rng.choice([-1,1]))
            for _ in range(12):
                pair = _generate_pair(rng,params,pieces,depth,d,side)
                key = _signature(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _decode(pair['input'])
    predicted,state = _assemble(info)
    _require(np.array_equal(predicted,_grid(pair['output'])), "incorrect turn-gated assembly")
    pieces, positions = info['pieces'], state['positions']
    all_output,_ = _assemble(info,all_branches=True)
    unmoved,_ = _assemble(info,keep_scatter=True)
    _require(not np.array_equal(all_output,predicted), "turn selection has no effect")
    _require(not np.array_equal(unmoved,predicted), "geometric translation has no effect")
    _require(positions[info['source']] == (0,0), "source moved")
    _require(all(delta != (0,0) for i,delta in positions.items() if i != info['source']), "activated movable tile stayed in place")
    # Each activated edge shares its full port; dormant outlet teeth remain
    # single-owned. Thus there are also valid exposed connectors in the output.
    shared = {q for q,owners in state['owners'].items() if len(owners)==2}
    expected_shared = set()
    for j in positions:
        if j == info['source']:
            continue
        d = info['arrivals'][j]
        expected_shared |= _shift(pieces[j]['ports'][(d+2)%4],positions[j])
    _require(shared == expected_shared, "connector sharing differs from activated docks")
    selected_pin_count = sum(len(pieces[i]['pins']) for i in positions)
    _require(np.count_nonzero(predicted==info['marker']) == selected_pin_count-len(shared), "connector accounting failed")
    for color in set(p['color'] for p in pieces):
        count = sum(len(pieces[i]['body']) for i in positions if pieces[i]['color']==color)
        _require(np.count_nonzero(predicted==color)==count, "payload was recolored or lost")
    keys = list({_normal(pins) for p in pieces for pins in p['ports'].values()})
    different_equal = any(len(a)==len(b) and a!=b and a[-1][0]==b[-1][0]
                          for a,b in combinations(keys,2)) or any(
        len(a)==len(b) and a!=b and a[-1][1]==b[-1][1] for a,b in combinations(keys,2))
    _require(different_equal, "no equal-cardinality geometry distinction")
    boxes = [(p['box'][2]-p['box'][0]+1,p['box'][3]-p['box'][1]+1) for p in pieces]
    turns_observed = [(pieces[j]['color'],d,t) for j,d,t,_,exists in state['events'] if exists]
    return dict(shape=tuple(info['field'].shape),header=info['crop'],keys=tuple(keys),
                boxes=tuple(boxes),bodies=tuple(_normal(p['body']) for p in pieces),
                piece_count=len(pieces),active_count=len(positions),erased=len(pieces)-len(positions),
                depth=max(state['depths'].values()),source_degree=len(pieces[info['source']]['ports']),
                observed=turns_observed,turns=info['turns'],widths=info['widths'],
                roles=(info['background'],info['marker'],pieces[info['source']]['color'],info['frame']),
                reuse=len(pieces)-len({p['color'] for p in pieces}),
                sparse=any(len(_components(key))>1 for key in keys))


def validate_generated_puzzle(puzzle):
    """Check rigid activation, geometric docks, erasure, crop, and teaching evidence."""
    _require(isinstance(puzzle,dict) and set(puzzle)=={'train','test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "splits must be lists")
    a,b = len(puzzle['train']),len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid counts")
    seen, features = set(), {'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair)=={'input','output'}, "invalid pair keys")
            inp,out = _grid(pair['input']),_grid(pair['output'])
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            features[split].append(_features(dict(input=inp,output=out)))
    records = features['train']+features['test']
    _require(len({f['roles'] for f in records})==1, "puzzle role palette changed")
    headings = {-1:set(),1:set()}
    for f in features['train']:
        for _,d,t in f['observed']:
            headings[t].add(d)
    _require(all(len(v)>=2 for v in headings.values()), "both turns need multiple arrival headings")
    _require(all(f['erased']>=1 for f in records), "missing inactive branch")
    _require(max(f['depth'] for f in features['train'])==3, "training must demonstrate three docks")
    _require(all(f['depth']>3 for f in features['test']), "missing deeper-route test shift")
    changing = any(len({t for f in features['train'] for color,_,t in f['observed'] if color==c})>1 for c in range(10))
    _require(changing, "training must change a recurring color's turn")
    # Test all four literal mappings of the two bar positions against training.
    fits = []
    for left,right in ((-1,-1),(-1,1),(1,-1),(1,1)):
        okay = True
        for pair in puzzle['train']:
            info = _decode(pair['input'])
            table = {c:left if t==-1 else right for c,t in info['turns'].items()}
            alternative,_ = _assemble(info,turn_override=table)
            if not np.array_equal(alternative,np.asarray(pair['output'])):
                okay = False
                break
        if okay:
            fits.append((left,right))
    _require(fits == [(-1,1)], "turn convention is not uniquely demonstrated")


def _puzzles_equal(a,b):
    return all(len(a[s])==len(b[s]) and all(np.array_equal(x[k],y[k])
               for x,y in zip(a[s],b[s]) for k in ('input','output')) for s in ('train','test'))


def self_test():
    """Bounded 42-seed variation study plus legal counts and semantic negatives."""
    records, signatures = [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p['input']) for s in puzzle for p in puzzle[s]))
        records.extend(_features(p) for s in puzzle for p in puzzle[s])
    _require(len(signatures)==42, "seed diversity failed")
    _require(_puzzles_equal(generate_puzzle(0),generate_puzzle(0)), "determinism failed")
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p = generate_puzzle(2026,a,b)
            _require(len(p['train'])==a and len(p['test'])==b, "explicit count mismatch")
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # The input solver also handles the structural family endpoints, which
    # are outside the ordinary generated puzzle's 5..9-piece sampling window.
    fixture_rng = np.random.default_rng(8192)
    fixture_params = dict(roles=(0,2,8,5),colors=[1,3,4,6,7,9])
    for count,depth in ((4,2),(10,4)):
        pair = _generate_pair(fixture_rng,fixture_params,count,depth,1,-1)
        _require(np.array_equal(solve(pair['input']),pair['output']), "family endpoint solve failed")
    _require(len({f['shape'] for f in records})>=30, "insufficient canvas variation")
    _require(len({x for f in records for x in f['bodies']})>=70, "insufficient tile-mask variation")
    _require(len({x for f in records for x in f['keys']})>=12, "insufficient port geometry variation")
    _require({f['piece_count'] for f in records} >= set(range(5,10)), "missing piece counts")
    _require({f['depth'] for f in records} >= {2,3,4,5}, "missing route depths")
    _require(len({f['source_degree'] for f in records})>=3, "missing simultaneous-arm variation")
    _require(any(f['reuse']>=2 for f in records), "missing repeated payload colors")
    _require(any(len(set(f['widths']))>1 for f in records), "missing unequal panels")
    _require(any(f['sparse'] for f in records), "missing sparse connectors")
    _require(len({f['roles'] for f in records})>=30, "insufficient palette-role variation")
    # Mutating a dock or a preserved body must invalidate a labeled example.
    puzzle = generate_puzzle(0)
    bad = {s:[{k:np.asarray(v).copy() for k,v in p.items()} for p in puzzle[s]] for s in puzzle}
    info = _decode(bad['test'][0]['input'])
    q = tuple(np.argwhere(bad['test'][0]['output']==info['marker'])[0])
    bad['test'][0]['output'][q] = info['background']
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("erased output connector accepted")
    bad = {s:[{k:np.asarray(v).copy() for k,v in p.items()} for p in puzzle[s]] for s in puzzle}
    q = min(info['pieces'][info['source']]['body'])
    bad['test'][0]['output'][q] = info['background']
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("erased source payload accepted")
    broken = puzzle['test'][0]['input'].copy()
    q = min(info['pieces'][next(i for i in range(len(info['pieces'])) if i != info['source'])]['pins'])
    broken[info['crop']+q[0],q[1]] = info['background']
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete docking key accepted")
    return dict(seeds=42,pairs=len(records),field_shapes=len({f['shape'] for f in records}),
                body_masks=len({x for f in records for x in f['bodies']}),
                port_masks=len({x for f in records for x in f['keys']}),
                piece_counts=sorted({f['piece_count'] for f in records}),
                active_depths=sorted({f['depth'] for f in records}),
                source_degrees=sorted({f['source_degree'] for f in records}))


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
