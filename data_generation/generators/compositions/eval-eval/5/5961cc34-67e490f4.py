"""Path-conditioned shape ballots select relay exits. See task_spec.md.

Only visible input data is used by solve; construction labels use planned
ballots and route geometry, which validation reconstructs from the grids.
"""
from collections import Counter
from functools import lru_cache
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _norm(cells):
    cells = tuple(cells)
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


@lru_cache(maxsize=2048)
def _rotations(shape):
    shape = _norm(shape)
    result = set()
    for _ in range(4):
        result.add(shape)
        shape = _norm((c, -r) for r, c in shape)
    return tuple(sorted(result))


@lru_cache(maxsize=2048)
def _variants(shape):
    return tuple(sorted(set(_rotations(shape)) |
                        set(_rotations(_norm((r, -c) for r, c in shape)))))


def _key(cells, grouping='free'):
    raw = _norm(cells)
    if grouping == 'area':
        return len(raw)
    if grouping == 'box':
        return tuple(sorted(_size(raw)))
    if grouping == 'translation':
        return raw
    if grouping == 'rotation':
        return min(_rotations(raw))
    return min(_variants(raw))


def _size(shape):
    return max(r for r, c in shape)+1, max(c for r, c in shape)+1


def _move(cells, r0, c0):
    return {(r+r0, c+c0) for r, c in cells}


def _step(cell, direction):
    dr, dc = DIRS[direction]
    return cell[0]+dr, cell[1]+dc


@lru_cache(maxsize=1)
def _pool():
    """All small connected, hole-free 3x3 polyominoes plus long motifs."""
    pool = set()
    for bits in range(1, 512):
        cells = {(i//3, i % 3) for i in range(9) if bits & (1 << i)}
        if not 3 <= len(cells) <= 7 or len(_components(cells)) != 1:
            continue
        empty = {(r, c) for r in range(-1, 4) for c in range(-1, 4)}-cells
        if len(_components(empty)) == 1:
            pool.add(_key(cells))
    for length in range(4, 7):
        pool.add(_key({(0, c) for c in range(length)}))
        pool.add(_key({(0, c) for c in range(length)} | {(1, 0)}))
    return tuple(sorted(pool))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            np.any((a < 0) | (a > 9))):
        raise ValueError('Expected an integer ARC grid with dimensions 1..30')
    return a


def _parse(value):
    g = _grid(value)
    h, w = g.shape
    bg = int(g[0, 0])
    if any(int(g[p]) != bg for p in ((0,w-1),(h-1,0),(h-1,w-1))):
        raise ValueError('Corners identify the background')
    groups = _components(map(tuple, np.argwhere(g != bg)))
    launchers = [s for s in groups if any(r in (0,h-1) or c in (0,w-1) for r,c in s)]
    if len(launchers) != 1:
        raise ValueError('One boundary launcher is required')
    launcher = launchers[0]
    boundary = [p for p in launcher if p[0] in (0,h-1) or p[1] in (0,w-1)]
    if len(boundary) != 1:
        raise ValueError('Launcher must touch one boundary cell')
    start = boundary[0]
    direction = 2 if start[0] == 0 else 0 if start[0] == h-1 else 1 if start[1] == 0 else 3
    dr, dc = DIRS[direction]
    line = [(start[0]+k*dr, start[1]+k*dc) for k in range(len(launcher))]
    if set(line) != launcher or not 3 <= len(line) <= 5:
        raise ValueError('Launcher is a straight tail and cap')
    signal, cap_color = int(g[start]), int(g[line[-1]])
    if signal == cap_color or any(int(g[p]) != signal for p in line[:-1]):
        raise ValueError('Launcher colors must distinguish its cap')
    gates, voters, carriers, port_palettes = [], [], set(), []
    for cells in groups:
        if cells == launcher:
            continue
        colors = {int(g[p]) for p in cells}
        if len(colors) == 1:
            voters.append(dict(cells=cells, color=next(iter(colors)), raw=_norm(cells)))
            continue
        r0,r1 = min(r for r,c in cells), max(r for r,c in cells)
        c0,c1 = min(c for r,c in cells), max(c for r,c in cells)
        box = {(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)}
        hole = box-cells
        if (len(colors) != 5 or not hole or len(_components(hole)) != 1 or
                any(r in (r0,r1) or c in (c0,c1) for r,c in hole)):
            raise ValueError('Relay must have four ports and one enclosed cavity')
        counts = Counter(int(g[p]) for p in cells)
        carrier = max(counts, key=counts.get)
        body = {p for p in cells if int(g[p]) == carrier}
        if len(_components(body)) != 1 or counts[carrier] <= len(cells)//2:
            raise ValueError('Connected carrier must dominate each relay')
        ports = {}
        for color in colors-{carrier}:
            port = {p for p in cells if int(g[p]) == color}
            sides = [d for d,ok in enumerate((all(r == r0 for r,c in port),
                     all(c == c1 for r,c in port), all(r == r1 for r,c in port),
                     all(c == c0 for r,c in port))) if ok]
            if len(sides) != 1 or not 1 <= len(port) <= 3:
                raise ValueError('A port occupies one noncorner extreme side')
            side = sides[0]
            span = sorted(c if side % 2 == 0 else r for r,c in port)
            if (span != list(range(span[0],span[0]+len(span))) or
                    any(_step(p,(side+2) % 4) not in body for p in port)):
                raise ValueError('Port must be contiguous with inward carrier neighbors')
            ports[color] = (side, port)
        if {d for d,p in ports.values()} != set(range(4)):
            raise ValueError('Exactly one port per side')
        gates.append(dict(cells=cells, box=box, hole=hole, raw=_norm(hole),
                          ports=ports, carrier=carrier))
        carriers.add(carrier)
        port_palettes.append(set(ports))
    if not gates or len(carriers) != 1 or any(p != port_palettes[0] for p in port_palettes):
        raise ValueError('Relays share a visible carrier and four ballot colors')
    palette = port_palettes[0]
    if len({bg,signal,cap_color,*carriers,*palette}) != 8:
        raise ValueError('Semantic color roles must be disjoint')
    if any(v['color'] not in palette for v in voters):
        raise ValueError('Every voter uses a port color')
    # One-cell separation, including diagonals, makes object counts unambiguous.
    owners = {p:i for i,s in enumerate(groups) for p in s}
    for p,i in owners.items():
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if owners.get((p[0]+dr,p[1]+dc),i) != i:
                    raise ValueError('Components need an eight-neighbor gap')
    return dict(grid=g, bg=bg, signal=signal, cap=line[-1], direction=direction,
                launcher=launcher, gates=gates, voters=voters, palette=palette)


def _ray(shape, origins, direction, gates):
    """First contact is measured on original relay footprints, not painted rays."""
    h,w = shape
    front = [_step(p,direction) for p in sorted(origins)]
    owner = {p:i for i,g in enumerate(gates) for p in g['cells']}
    segment = set()
    for _ in range(max(h,w)+1):
        inside = [0 <= r < h and 0 <= c < w for r,c in front]
        if not any(inside):
            return segment, None
        if not all(inside):
            raise ValueError('Beam lanes must end simultaneously')
        hits = {owner[p] for p in front if p in owner}
        if hits:
            if len(hits) != 1 or any(owner.get(p) not in hits for p in front):
                raise ValueError('Full-width unique first contact is required')
            return segment, next(iter(hits))
        segment.update(front)
        front = [_step(p,direction) for p in front]
    raise ValueError('Ray failed to terminate')


def _derive(value, grouping='free', scope='leg'):
    data = _parse(value)
    g, gates, voters = data['grid'], data['gates'], data['voters']
    out = np.full(g.shape, data['bg'], dtype=int)
    for p in data['launcher']:
        out[p] = data['signal']
    origins, direction = {data['cap']}, data['direction']
    reached, records, trace = [], [], set(data['launcher'])
    previous_segments = set()
    for _ in range(len(gates)+1):
        segment, index = _ray(g.shape,origins,direction,gates)
        trace |= segment
        for p in segment:
            out[p] = data['signal']
        if index is None:
            data.update(reached=reached, records=records, trace=trace)
            return out, data
        if index in reached:
            raise ValueError('Repeated relay activation is outside this family')
        gate = gates[index]
        population = segment | previous_segments if scope == 'prefix' else segment
        selected = [v for v in voters if (scope == 'global' or v['cells'] & population)
                    and _key(v['raw'],grouping) == _key(gate['raw'],grouping)]
        ballot = Counter(v['color'] for v in selected)
        high = max(ballot.values(),default=0)
        winners = [c for c,n in ballot.items() if n == high]
        if len(winners) != 1:
            raise ValueError('Reached cavities need a supported unique plurality')
        winner = winners[0]
        outgoing, port = gate['ports'][winner]
        if outgoing == (direction+2) % 4:
            raise ValueError('Selected port may continue or turn, not reverse')
        for p in gate['cells']:
            out[p] = data['signal']
        for p in gate['hole']:
            out[p] = winner
        records.append(dict(index=index, incoming=direction, outgoing=outgoing,
                            width=len(port), winner=winner, ballot=dict(ballot),
                            key=_key(gate['raw']), segment=segment,
                            touched=[v for v in voters if v['cells'] & segment]))
        reached.append(index)
        trace |= gate['cells']
        previous_segments |= segment
        origins, direction = port, outgoing
    raise ValueError('Unterminated relay chain')


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles, segment-local ballots, chosen ports and route from input."""
    return _derive(input_grid)[0]


def _choice(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _blocked(shape, occupied, forbidden=()):
    available = np.ones(shape,dtype=bool)
    available[[0,-1],:] = False
    available[:,[0,-1]] = False
    h,w = shape
    for r,c in occupied:
        available[max(0,r-1):min(h,r+2),max(0,c-1):min(w,c+2)] = False
    for p in forbidden:
        available[p] = False
    return available


def _positions(available, raw, intersect=None):
    sh,sw = _size(raw)
    h,w = available.shape
    if sh > h or sw > w:
        return []
    ok = np.ones((h-sh+1,w-sw+1),dtype=bool)
    touches = np.zeros_like(ok)
    for r,c in raw:
        ok &= available[r:r+ok.shape[0],c:c+ok.shape[1]]
        if intersect is not None:
            touches |= intersect[r:r+ok.shape[0],c:c+ok.shape[1]]
    if intersect is not None:
        ok &= touches
    return [tuple(map(int,p)) for p in np.argwhere(ok)]


def _pack_votes(rng, shape, specs, segment, occupied, old_rays):
    available = _blocked(shape,occupied,old_rays)
    mask = np.zeros(shape,dtype=bool)
    for p in segment:
        mask[p] = True
    # Every vote lies longitudinally between the emitting object and the hit.
    rows,cols = zip(*segment)
    if len(set(rows)) <= 2 and max(cols)-min(cols) > 2:
        available[:,:min(cols)] = False
        available[:,max(cols)+1:] = False
    else:
        available[:min(rows),:] = False
        available[max(rows)+1:,:] = False
    result = []
    # Place larger bounding boxes first; positions remain irregular.
    ordered = [specs[int(i)] for i in rng.permutation(len(specs))]
    ordered.sort(key=lambda s: min(_size(v)[0]*_size(v)[1] for v in s[1]), reverse=True)
    for color, variants in ordered:
        options = []
        for raw in variants:
            positions = _positions(available,raw,mask)
            if positions:
                options.append((raw,positions))
        if not options:
            return None
        raw, positions = _choice(rng,options)
        cells = _move(raw,*_choice(rng,positions))
        result.append(dict(cells=cells,color=color,raw=raw))
        for r,c in cells:
            available[max(0,r-1):r+2,max(0,c-1):c+2] = False
    return result


def _gate(rng, raw, box, selected_side, selected_width, winner, colors):
    r0,c0,gh,gw = box
    kh,kw = _size(raw)
    hr,hc = r0+2+int(rng.integers(gh-kh-3)), c0+2+int(rng.integers(gw-kw-3))
    hole = _move(raw,hr,hc)
    whole = {(r,c) for r in range(r0,r0+gh) for c in range(c0,c0+gw)}
    other_colors = [c for c in colors if c != winner]
    rng.shuffle(other_colors)
    ports = {}
    for side in range(4):
        color = winner if side == selected_side else other_colors.pop()
        length = selected_width if side == selected_side else int(rng.integers(1,3))
        extent = gw if side % 2 == 0 else gh
        start = int(rng.integers(2,extent-length-1))
        if side == 0:
            port = {(r0,c0+start+k) for k in range(length)}
        elif side == 2:
            port = {(r0+gh-1,c0+start+k) for k in range(length)}
        elif side == 1:
            port = {(r0+start+k,c0+gw-1) for k in range(length)}
        else:
            port = {(r0+start+k,c0) for k in range(length)}
        ports[color] = (side,port)
    return dict(cells=whole-hole,box=whole,hole=hole,raw=raw,ports=ports,winner=winner)


def _vote_specs(rng, key, winner, colors, mode, companion=None):
    raw = key
    variants = _variants(key)
    losers = [c for c in colors if c != winner]
    rng.shuffle(losers)
    pattern = (2,1,1) if mode == 'plurality' else (2,1) if mode == 'geometry' else mode
    specs = []
    for color,count in zip([winner]+losers,pattern):
        allowed = variants
        if mode in ('plurality','geometry'):
            # Winner votes have opposite handedness to the cavity; losing
            # votes retain its orientation. Reflection affects the answer.
            allowed = tuple(v for v in variants if v not in _rotations(raw)) if color == winner else (raw,)
        specs.extend((color,allowed) for _ in range(count))
    if companion is not None:
        specs.extend((losers[0],_variants(companion)) for _ in range(2))
    return specs, losers[0]


def _layout(rng, keys, specs, winners, widths, colors, demo):
    n = len(keys)
    h = int(rng.integers(26 if n > 1 else 15,31))
    w = int(rng.integers(28 if demo is not None else 22,31))
    length = int(rng.integers(2,4))
    row = int(rng.integers(2,h-2))
    launcher = {(row,c) for c in range(length+1)}
    cap = row,length
    budget = [220]

    def grow(i,gates,voters,segments,origins,direction,occupied,old_rays):
        if i == n:
            try:
                segment,hit = _ray((h,w),origins,direction,gates)
            except ValueError:
                return None
            return (gates,voters,segments+[segment]) if hit is None else None
        raw = _choice(rng,_variants(keys[i])) if demo is None or i > 0 else keys[i]
        kh,kw = _size(raw)
        gh,gw = max(6,kh+4)+int(rng.integers(2)), max(6,kw+4)+int(rng.integers(2))
        options = []
        available = _blocked((h,w),occupied,old_rays)
        # Rectangular boxes are kept out of all earlier ray legs.
        rectangle = tuple((r,c) for r in range(gh) for c in range(gw))
        for r0,c0 in _positions(available,rectangle):
            dummy = dict(cells=_move(rectangle,r0,c0))
            try:
                segment,hit = _ray((h,w),origins,direction,gates+[dummy])
            except ValueError:
                continue
            if hit != len(gates) or any(v['cells'] & segment for v in voters):
                continue
            # A necessary capacity bound reduces hopeless packing attempts.
            needed = sum(min(_size(v)[1 if direction % 2 else 0] for v in vs)+1 for _,vs in specs[i])
            if len(segment)//len(origins) < needed:
                continue
            options.append((r0,c0,segment))
        rng.shuffle(options)
        for r0,c0,segment in options[:45]:
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            turns = [0] if demo == 0 else [-1,1] if demo == 1 and i == 0 else [-1,0,1]
            rng.shuffle(turns)
            for turn in turns:
                gate = _gate(rng,raw,(r0,c0,gh,gw),(direction+turn) % 4,widths[i],winners[i],colors)
                local = _pack_votes(rng,(h,w),specs[i],segment,occupied|gate['box'],old_rays)
                if local is None:
                    continue
                side,port = gate['ports'][winners[i]]
                result = grow(i+1,gates+[gate],voters+local,segments+[segment],port,side,
                              occupied|gate['box']|set().union(*(v['cells'] for v in local)),
                              old_rays|segment)
                if result is not None:
                    return result
        return None

    result = grow(0,[],[],[],{cap},1,launcher,launcher)
    if result is None:
        return None
    return (h,w,launcher,cap,*result)


def _fallback(rng, keys, specs, winners, widths, colors, demo):
    """Randomized rectangular circuit with guaranteed space for up to 3 gates.

    Conservative packing path; it retains the planned ballots, key classes,
    activation count and width sequence instead of silently simplifying them.
    """
    h = w = 30
    dims = [_size(k) for k in keys]
    dims = [(max(6,a+4),max(6,b+4)) for a,b in dims]
    boxes = [(29-dims[0][0],29-dims[0][1],*dims[0])]
    if len(keys) >= 2:
        boxes.append((1,29-dims[1][1],*dims[1]))
    if len(keys) == 3:
        boxes.append((1,1,*dims[2]))
    sides = [1 if demo == 0 else 0,3,2]
    for _ in range(120):
        gates = [_gate(rng,k,b,sides[i],widths[i],winners[i],colors) for i,(k,b) in enumerate(zip(keys,boxes))]
        # Align subsequent boxes to the preceding selected port.
        valid = True
        for i in range(1,len(gates)):
            prev_side,prev_port = gates[i-1]['ports'][winners[i-1]]
            lane = min(c if prev_side % 2 == 0 else r for r,c in prev_port)
            r0,c0,gh,gw = boxes[i]
            if prev_side % 2 == 0:
                c0 = min(29-gw,max(1,lane-2))
            else:
                r0 = min(29-gh,max(1,lane-2))
            gates[i] = _gate(rng,keys[i],(r0,c0,gh,gw),sides[i],widths[i],winners[i],colors)
        first = gates[0]
        row = min(r for r,c in first['cells'])+2
        launcher = {(row,c) for c in range(3)}
        cap = row,2
        origins,direction = {cap},1
        segments,voters = [],[]
        occupied = launcher | set().union(*(g['box'] for g in gates))
        old_rays = set(launcher)
        for i in range(len(gates)):
            try:
                segment,hit = _ray((h,w),origins,direction,gates)
            except ValueError:
                valid = False
                break
            if hit != i or any(v['cells'] & segment for v in voters):
                valid = False
                break
            local = _pack_votes(rng,(h,w),specs[i],segment,occupied,old_rays)
            if local is None:
                valid = False
                break
            voters += local
            occupied |= set().union(*(v['cells'] for v in local))
            segments.append(segment)
            old_rays |= segment
            direction,origins = gates[i]['ports'][winners[i]]
        if not valid:
            continue
        segment,hit = _ray((h,w),origins,direction,gates)
        if hit is None:
            return h,w,launcher,cap,gates,voters,segments+[segment]
    raise RuntimeError('Conservative route packing exhausted its bound')


def _add_exterior(rng, layout, keys, colors, off_specs, decoy_count):
    h,w,launcher,cap,gates,voters,segments = layout
    trace = launcher | set().union(*segments) | set().union(*(g['box'] for g in gates))
    occupied = launcher | set().union(*(g['box'] for g in gates)) | set().union(*(v['cells'] for v in voters))
    decoys = []
    for _ in range(decoy_count):
        key = _choice(rng,keys)
        raw = _choice(rng,_variants(key))
        kh,kw = _size(raw)
        gh,gw = max(6,kh+4),max(6,kw+4)
        rectangle = tuple((r,c) for r in range(gh) for c in range(gw))
        positions = _positions(_blocked((h,w),occupied,trace),rectangle)
        if not positions:
            return None
        r0,c0 = _choice(rng,positions)
        gate = _gate(rng,raw,(r0,c0,gh,gw),int(rng.integers(4)),int(rng.integers(1,3)),_choice(rng,colors),colors)
        decoys.append(gate)
        occupied |= gate['box']
    for color, variants in off_specs:
        available = _blocked((h,w),occupied,trace)
        options = [(raw,_positions(available,raw)) for raw in variants]
        options = [o for o in options if o[1]]
        if not options:
            return None
        raw,positions = _choice(rng,options)
        cells = _move(raw,*_choice(rng,positions))
        voters.append(dict(cells=cells,color=color,raw=raw))
        occupied |= cells
    return h,w,launcher,cap,gates,voters,segments,decoys


def _make_pair(rng, keys, palette, demo=None, companion=None, first_winner=None):
    bg,signal,cap_color,carrier = palette[:4]
    colors = palette[4:8]
    winners = [_choice(rng,colors) for _ in keys]
    if first_winner is not None:
        winners[0] = _choice(rng,[c for c in colors if c != first_winner])
    if demo == 1:
        winners[1] = _choice(rng,[c for c in colors if c != winners[0]])
    widths = [int(rng.integers(1,3)) for _ in keys]
    if demo == 0:
        widths[0] = 1
    elif demo == 1:
        widths[:2] = [2,1]
    specs,losers = [],[]
    for i,key in enumerate(keys):
        mode = ('plurality' if demo == 0 else 'geometry') if i == 0 and demo is not None else (
            _choice(rng,[(1,),(2,),(2,1),(2,1,1),(3,1),(3,2)]) if i == 0 else _choice(rng,[(1,),(1,),(2,)]))
        if demo == 1 and i == 1:
            mode = (1,)
        if isinstance(mode,tuple):
            extent = min(min(_size(v)) for v in _variants(key))
            compatible = [m for m in ((1,),(2,),(2,1),(2,1,1),(3,1),(3,2))
                          if sum(m)*(extent+1) <= 16]
            if mode not in compatible:
                mode = _choice(rng,compatible)
        ballot,loser = _vote_specs(rng,key,winners[i],colors,mode,companion if demo == 1 and i == 0 else None)
        specs.append(ballot)
        losers.append(loser)
    off_specs = []
    if demo is not None:
        off_specs = [(losers[0],_variants(keys[0])) for _ in range(3)]
    else:
        off_specs = [(_choice(rng,colors),_variants(_choice(rng,keys))) for _ in range(int(rng.integers(0,5)))]
    decoy_count = 1 if demo is not None else int(rng.integers(0,3))
    result = None
    for attempt in range(16):
        layout = _layout(rng,keys,specs,winners,widths,colors,demo) if attempt < 12 else _fallback(rng,keys,specs,winners,widths,colors,demo)
        if layout is not None:
            result = _add_exterior(rng,layout,keys,colors,off_specs,decoy_count)
            if result is not None:
                break
    if result is None:
        raise RuntimeError('No room for planned route and exterior objects')
    h,w,launcher,cap,gates,voters,segments,decoys = result
    inp = np.full((h,w),bg,dtype=int)
    out = np.full_like(inp,bg)
    for gate in gates+decoys:
        for p in gate['cells']:
            inp[p] = carrier
        for color,(_,port) in gate['ports'].items():
            for p in port:
                inp[p] = color
    for v in voters:
        for p in v['cells']:
            inp[p] = v['color']
    for p in launcher:
        inp[p] = out[p] = signal
    inp[cap] = cap_color
    for p in set().union(*segments):
        out[p] = signal
    for gate in gates:
        for p in gate['cells']:
            out[p] = signal
        for p in gate['hole']:
            out[p] = gate['winner']
    # Rotate and reflect the whole scene, preserving the spatial convention.
    if rng.random() < .5:
        inp,out = np.fliplr(inp),np.fliplr(out)
    turns = int(rng.integers(4))
    return dict(input=np.rot90(inp,turns).copy(),output=np.rot90(out,turns).copy())


def _input_key(g):
    a = np.asarray(g)
    return a.shape,a.astype(np.int8).tobytes()


def _fails(pair, grouping='free', scope='leg'):
    try:
        pred,_ = _derive(pair['input'],grouping,scope)
    except ValueError:
        return True
    return not np.array_equal(pred,pair['output'])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name,value,lo,hi in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer from {lo} to {hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    s = _key(((0,1),(0,2),(1,0),(1,1)))
    ell = _key(((0,0),(1,0),(2,0),(2,1)))
    anchor,companion = (s,ell) if rng.random() < .5 else (ell,s)
    small = [k for k in _pool() if max(_size(k)) <= 3]
    train_pool = [anchor,companion]+[k for k in small if k not in (anchor,companion)]
    rng.shuffle(train_pool)
    # A seed-level repertoire allows local reuse; tests visibly supply new keys.
    train_pool = list(dict.fromkeys([anchor,companion]+train_pool[:int(rng.integers(3,8))]))
    base = list(map(int,rng.permutation(10)))
    vary_palette = rng.random() < .75
    puzzle,seen,known = {'train':[],'test':[]},set(),set()
    previous_winner = None
    for split,count in (('train',num_train),('test',num_test)):
        for i in range(count):
            demo = i if split == 'train' and i < 2 else None
            if demo == 0:
                keys = [anchor]
            elif demo == 1:
                keys = [anchor,anchor]
            elif split == 'train':
                keys = [_choice(rng,train_pool) for _ in range(int(rng.integers(1,3)))]
            else:
                novel = [k for k in small if k not in known]
                n = 3 if i == 0 else int(rng.integers(1,4))
                keys = [_choice(rng,novel)]+[_choice(rng,small) for _ in range(n-1)]
                # Longer bar/L motifs are compatible in a single-relay test.
                if n == 1 and rng.random() < .4:
                    keys[0] = _choice(rng,[k for k in _pool() if max(_size(k)) > 3])
            palette = list(map(int,rng.permutation(10))) if vary_palette and not (split == 'train' and i == 0) else base
            for _ in range(8):
                pair = _make_pair(rng,keys,palette,demo,companion,
                                  previous_winner if demo == 1 else None)
                if _input_key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('Could not produce distinct inputs')
            puzzle[split].append(pair)
            seen.add(_input_key(pair['input']))
            if split == 'train':
                _,facts = _derive(pair['input'])
                known.update(_key(g['raw']) for g in facts['gates'])
                if demo == 0:
                    previous_winner = facts['records'][0]['winner']
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle,dict) or set(puzzle) != {'train','test'}:
        raise ValueError('Puzzle must have exactly train and test')
    if not all(isinstance(puzzle[s],list) for s in puzzle):
        raise ValueError('Splits must be lists')
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    if not (2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne):
        raise ValueError('Illegal split counts')
    seen,facts = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            if not isinstance(pair,dict) or set(pair) != {'input','output'}:
                raise ValueError('Pair must have exactly input and output')
            inp,out = _grid(pair['input']),_grid(pair['output'])
            if inp.shape != out.shape or _input_key(inp) in seen:
                raise ValueError('Output size must match and inputs must differ')
            seen.add(_input_key(inp))
            pred,data = _derive(inp)
            if not np.array_equal(pred,out):
                raise ValueError('Output disagrees with visible route and ballots')
            reached = set(data['reached'])
            for j,gate in enumerate(data['gates']):
                if j not in reached and any(out[p] != data['bg'] for p in gate['box']):
                    raise ValueError('Unreached relays must disappear')
            if any(out[p] != data['signal'] for p in data['trace']):
                raise ValueError('Trace must copy the launcher signal')
            facts[split].append(data)
    train = puzzle['train']
    obligations = [('area','leg'),('box','leg'),('rotation','leg'),('translation','leg'),('free','global'),('free','prefix')]
    if not all(any(_fails(p,k,s) for p in train) for k,s in obligations):
        raise ValueError('Training lacks shape/local-ballot witnesses')
    records = [r for d in facts['train'] for r in d['records']]
    if not any(len(d['reached']) >= 2 for d in facts['train']):
        raise ValueError('Training must demonstrate relay chaining')
    if not any(len(d['reached']) < len(d['gates']) for d in facts['train']):
        raise ValueError('Training must demonstrate erasure')
    if not ({1,2} <= {r['width'] for r in records} and
            any(r['incoming'] == r['outgoing'] for r in records) and
            any(r['incoming'] != r['outgoing'] for r in records)):
        raise ValueError('Training must show widths, straight exits and turns')
    if not any(len(r['ballot']) >= 3 and 2*max(r['ballot'].values()) <= sum(r['ballot'].values()) for r in records):
        raise ValueError('Training must show a non-majority plurality')
    history = {}
    for r in records:
        history.setdefault(r['key'],set()).add(r['winner'])
    if not any(len(v) >= 2 for v in history.values()):
        raise ValueError('Training must reject a fixed shape/color dictionary')
    return None


def self_test():
    """Bounded tests: 42 seeds, meaningful witnesses and geometric covariance."""
    puzzles,records,structures,dimensions,vocabulary = [],[],set(),set(),set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ('train','test'):
            for pair in p[split]:
                inp = pair['input']
                saved = inp.copy()
                pred,data = _derive(inp)
                assert np.array_equal(pred,pair['output']) and np.array_equal(inp,saved)
                dimensions.add(inp.shape)
                mapping = {}
                structures.add(tuple(mapping.setdefault(int(c),len(mapping)) for c in inp.flat))
                vocabulary.update(_key(g['raw']) for g in data['gates'])
                records.append((split,data))
        # Geometry and color roles are covariant, not secretly seed-specific.
        pair = p['test'][0]
        assert np.array_equal(solve(np.rot90(pair['input'])),np.rot90(pair['output']))
        assert np.array_equal(solve(np.fliplr(pair['input'])),np.fliplr(pair['output']))
        palette = np.arange(9,-1,-1)
        assert np.array_equal(solve(palette[pair['input']]),palette[pair['output']])
    assert len({tuple(_input_key(x['input']) for s in p for x in p[s]) for p in puzzles}) == 42
    assert len(dimensions) >= 20 and len(structures) >= 180 and len(vocabulary) >= 20
    assert {d['direction'] for _,d in records} == set(range(4))
    assert {len(d['reached']) for _,d in records} == {1,2,3}
    assert {len(d['gates'])-len(d['reached']) for _,d in records} >= {0,1,2}
    assert len({tuple(sorted(r['ballot'].values(),reverse=True)) for _,d in records for r in d['records']}) >= 6
    for p in puzzles:
        trainkeys = {_key(g['raw']) for pair in p['train'] for g in _parse(pair['input'])['gates']}
        testkeys = {_key(g['raw']) for g in _parse(p['test'][0]['input'])['gates']}
        assert testkeys-trainkeys
        assert len(_derive(p['test'][0]['input'])[1]['reached']) == 3
    # Boundary split counts; verify.py separately enumerates every legal split.
    for nt,ne in ((2,1),(10,1),(10,4),(7,3)):
        validate_generated_puzzle(generate_puzzle(1701,nt,ne))
    # Exercise the conservative constructor directly, even if ordinary seeds
    # rarely need it. Check actual touched-component ballots after packing.
    fallback_rng = np.random.default_rng(40291)
    s = _key(((0,1),(0,2),(1,0),(1,1)))
    dense = _key(((0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,2)))
    for n in (1,2,3):
        for width in (1,2):
            keys = [s]+[dense]*(n-1)
            colors,winners = [4,5,6,7],[4,5,6][:n]
            specs = [_vote_specs(fallback_rng,k,winners[i],colors,(2,1,1) if i == 0 else (1,))[0]
                     for i,k in enumerate(keys)]
            layout = _fallback(fallback_rng,keys,specs,winners,[width]*n,colors,None)
            _,_,_,cap,gates,voters,segments = layout
            origins,direction = {cap},1
            for i in range(n):
                segment,hit = _ray((30,30),origins,direction,gates)
                assert hit == i and segment == segments[i]
                ballot = Counter(v['color'] for v in voters if v['cells'] & segment and _key(v['raw']) == keys[i])
                assert [c for c,count in ballot.items() if count == max(ballot.values())] == [winners[i]]
                direction,origins = gates[i]['ports'][winners[i]]
            assert _ray((30,30),origins,direction,gates)[1] is None
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    assert all(_input_key(a['input']) == _input_key(b['input']) and np.array_equal(a['output'],b['output'])
               for s in ('train','test') for a,b in zip(puzzles[0][s],generate_puzzle(0)[s]))
    broken = {s:[{k:np.asarray(v).copy() for k,v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    broken['test'][0]['output'][0,0] = (int(broken['test'][0]['output'][0,0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('Incorrect output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, split boundaries and fallback cases)')
