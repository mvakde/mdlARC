"""Clockwork track: assemble geometric ports, then propagate run-length clocks.

No parent imports, files, seeds, or training examples are used by solve().
Generation paints from constructed paths; validation recovers paths from inputs.
All diagnostics stay in memory; self_test retains only bounded feature sets.
"""
from __future__ import annotations

import hashlib
import itertools
import numbers

import numpy as np


HOST_SEEDS = (0,187429653,21121656,399684972,657863170,337215475,616692025,212342269,147760883,975229910,183691832,450142901,566498316,427855008,530330942,734495758,901028371,93452021,841244192,701340564,348591459,491509433,446562978,771224372,465261674,963407133,976318334,862870446,318175821,667522501,630163350,837563921,771423040,864066989,778733498,508009793,136728015,978103649,170627207,424834456,536038775,146479726)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer grid required"
    assert np.all((0 <= a) & (a <= 9)), "color range"
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _neighbors(p):
    r, c = p
    return ((r-1,c),(r+1,c),(r,c-1),(r,c+1))


def _layout(panels, order, shape, divider):
    nr, nc = shape
    h, w = panels[0].shape
    a = np.full((nr*(h+1)-1, nc*(w+1)-1), divider, dtype=np.int8)
    for slot, i in enumerate(order):
        r, c = divmod(slot, nc)
        a[r*(h+1):r*(h+1)+h, c*(w+1):c*(w+1)+w] = panels[i]
    return a


def _partition(a):
    """Full divider lines and corner background specify the equal panels."""
    answers = []
    for d in np.unique(a):
        rr = [-1] + np.flatnonzero(np.all(a == d, axis=1)).tolist() + [a.shape[0]]
        cc = [-1] + np.flatnonzero(np.all(a == d, axis=0)).tolist() + [a.shape[1]]
        if len(rr) == len(cc) == 2:
            continue
        hs = [b-a-1 for a,b in zip(rr,rr[1:])]
        ws = [b-a-1 for a,b in zip(cc,cc[1:])]
        if min(hs+ws) < 3 or len(set(hs)) != 1 or len(set(ws)) != 1:
            continue
        panels = [a[r+1:rr[i+1], c+1:cc[j+1]].copy()
                  for i,r in enumerate(rr[:-1]) for j,c in enumerate(cc[:-1])]
        if not 2 <= len(panels) <= 6 or any(np.any(p == d) for p in panels):
            continue
        corners = {int(p[r,c]) for p in panels for r,c in ((0,0),(0,-1),(-1,0),(-1,-1))}
        if len(corners) == 1:
            answers.append((panels, int(d), corners.pop(), (len(hs),len(ws))))
    assert len(answers) == 1, "unique full-line partition required"
    return answers[0]


def _orders(signatures):
    starts = [i for i,(a,b) in enumerate(signatures) if a is None and b is not None]
    ends = [i for i,(a,b) in enumerate(signatures) if a is not None and b is None]
    if len(starts) != 1 or len(ends) != 1:
        return []
    answers = []

    def visit(order, remaining):
        if len(answers) >= 2:
            return
        if not remaining:
            if order[-1] == ends[0]:
                answers.append(tuple(order))
            return
        outgoing = signatures[order[-1]][1]
        if outgoing is None:
            return
        for i in remaining:
            if signatures[i][0] == outgoing:
                visit(order+[i], tuple(j for j in remaining if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _canonical_panel(p, bg, guide):
    h,w = p.shape
    assert h >= 7 and w >= 3
    assert np.all(p[0] == bg) and np.all(p[-1] == bg)
    assert np.all(p[2] == bg), "header/track moat"
    assert np.all(p[1,[0,-1]] == bg)
    header = np.flatnonzero(p[1] != bg)
    source = None
    if len(header):
        assert header.tolist() == list(range(int(header[0]),int(header[-1])+1))
        color = int(p[1,header[0]])
        assert color not in (bg,guide) and np.all(p[1,header] == color)
        source = (len(header), color)
    assert np.all((p[3:-1] == bg) | (p[3:-1] == guide)), "one guide color"
    points = {tuple(map(int,x)) for x in np.argwhere(p == guide)}
    assert len(points) >= 2 and all(3 <= r <= h-2 for r,c in points)
    adjacency = {x: [y for y in _neighbors(x) if y in points] for x in points}
    endpoints = sorted(x for x in points if len(adjacency[x]) == 1)
    assert len(endpoints) == 2 and all(1 <= len(v) <= 2 for v in adjacency.values()), "simple unbranched track"
    left = sorted(r for r,c in points if c == 0)
    right = sorted(r for r,c in points if c == w-1)
    assert len(left) <= 1 and len(right) <= 1 and (left or right)
    contacts = {(r,0) for r in left} | {(r,w-1) for r in right}
    assert contacts <= set(endpoints), "ports must be track endpoints"
    # Flow is left-to-right in these canonical coordinates. A terminal's
    # interior endpoint is used when its corresponding edge port is absent.
    start = (left[0],0) if left else next(x for x in endpoints if x not in contacts)
    path, previous, current = [], None, start
    while True:
        path.append(current)
        following = [x for x in adjacency[current] if x != previous]
        if not following:
            break
        previous,current = current,following[0]
        assert len(path) <= len(points), "track cycle"
    assert len(path) == len(points), "disconnected track"
    assert path[-1] == ((right[0],w-1) if right else next(x for x in endpoints if x not in contacts))
    return dict(source=source, path=path,
                ports=(left[0] if left else None, right[0] if right else None))


def _decode(input_grid):
    a = _grid(input_grid)
    raw,divider,bg,shape = _partition(a)
    candidates = []
    for rotation in range(4):
        panels = [np.rot90(p,rotation).copy() for p in raw]
        boundary = {int(x) for p in panels for x in np.concatenate((p[:,0],p[:,-1])) if x != bg}
        if len(boundary) != 1:
            continue
        guide = next(iter(boundary))
        try:
            infos = [_canonical_panel(p,bg,guide) for p in panels]
            assert any(i['source'] is not None for i in infos)
            orders = _orders([i['ports'] for i in infos])
            assert len(orders) == 1
            assert len(panels)*(panels[0].shape[1]+1)-1 <= 30
        except (AssertionError,StopIteration):
            continue
        candidates.append(dict(panels=panels,infos=infos,order=orders[0],
                               rotation=rotation,divider=divider,bg=bg,guide=guide,
                               input_layout=shape))
    assert len(candidates) == 1, "unique header orientation and directed chain required"
    info = candidates[0]
    coordinates,sources,offsets = [],[],[]
    for i in info['order']:
        offsets.append(len(coordinates))
        if info['infos'][i]['source'] is not None:
            period,color = info['infos'][i]['source']
            sources.append((len(coordinates),period,color))
        coordinates.extend((i,r,c) for r,c in info['infos'][i]['path'])
    info.update(coordinates=coordinates,sources=sources,offsets=offsets)
    return info


def _render(info, mode='correct'):
    """Pointwise modular solver, with explicit rival rules for teaching checks."""
    panels = [p.copy() for p in info['panels']]
    tile_position = {i:j for j,i in enumerate(info['order'])}
    source_position = {s:next(j for j,x in enumerate(info['offsets']) if x == s)
                       for s,p,c in info['sources']}
    for t,(i,r,c) in enumerate(info['coordinates']):
        active = [(s,p,color) for s,p,color in info['sources'] if s <= t]
        if mode == 'reset':
            entry = max(x for x in info['offsets'] if x <= t)
            hits = [x for x in active if (t-entry) % x[1] == 0]
        elif mode == 'local':
            entry = max(x for x in info['offsets'] if x <= t)
            hits = [x for x in active if x[0] == entry and (t-x[0]) % x[1] == 0]
        elif mode == 'count_dividers':
            hits = [x for x in active if (t-x[0]+tile_position[i]-source_position[x[0]]) % x[1] == 0]
        elif mode == 'straight':
            width = panels[0].shape[1]
            x = tile_position[i]*width+c
            hits = [s for s in active if (x-(source_position[s[0]]*width+info['coordinates'][s[0]][2])) % s[1] == 0]
        elif mode == 'header_phase':
            gaps = {j:int(np.flatnonzero(info['panels'][info['order'][j]][1] != info['bg'])[0])-1
                    for j in source_position.values()}
            hits = [s for s in active if (t-s[0]-gaps[source_position[s[0]]]) % s[1] == 0]
        else:
            hits = [x for x in active if (t-x[0]) % x[1] == 0]
        if hits:
            if mode == 'oldest':
                winner = hits[0]
            elif mode == 'shortest':
                winner = min(hits,key=lambda x:(x[1],-x[0]))
            elif mode == 'longest':
                winner = min(hits,key=lambda x:(-x[1],-x[0]))
            elif mode == 'color_min':
                winner = min(hits,key=lambda x:x[2])
            elif mode == 'color_max':
                winner = max(hits,key=lambda x:x[2])
            else:
                winner = hits[-1]
            panels[i][r,c] = winner[2]
    out = _layout(panels,info['order'],(1,len(panels)),info['divider'])
    return np.rot90(out,-info['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive partition, orientation, chain, sources, and all phases from input."""
    return _render(_decode(input_grid))


def _path(rng,h,w,incoming,outgoing):
    start_c = 0 if incoming is not None else int(rng.integers(1,w-1))
    end_c = w-1 if outgoing is not None else int(rng.integers(1,w-1))
    start_r = int(incoming) if incoming is not None else int(rng.integers(3,h-1))
    end_r = int(outgoing) if outgoing is not None else int(rng.integers(3,h-1))
    # Terminals may be short, straight, bent, or have a detour. Separated
    # vertical columns ensure that no nonconsecutive track cells touch.
    hubs = list(range(max(1,start_c),min(w-2,end_c)+1))
    hub = int(rng.choice(hubs))
    waypoints = [(start_r,start_c)]
    possible = [x for x in hubs if x >= hub+2]
    if possible and rng.random() < .55:
        second = int(rng.choice(possible))
        middle = int(rng.integers(3,h-1))
        waypoints += [(start_r,hub),(middle,hub),(middle,second),(end_r,second)]
    else:
        waypoints += [(start_r,hub),(end_r,hub)]
    waypoints.append((end_r,end_c))
    result = [waypoints[0]]
    for target in waypoints[1:]:
        while result[-1] != target:
            r,c = result[-1]
            result.append((r+int(np.sign(target[0]-r)),c+int(np.sign(target[1]-c))))
    return result


def _paint_clocks(panels,paths,specs):
    """Construction uses stride overwrites, independently of input parsing."""
    out = [p.copy() for p in panels]
    coordinates = [(i,r,c) for i,path in enumerate(paths) for r,c in path]
    offset = 0
    for path,source in zip(paths,specs):
        if source is not None:
            period,color = source
            for i,r,c in coordinates[offset::period]:
                out[i][r,c] = color
        offset += len(path)
    return out


def _features(info):
    """Visible recurrence, overlap, order and geometry, from the returned input."""
    sources = info['sources']
    total = len(info['coordinates'])
    short_near = long_near = old_survives = collision = False
    repeat = set()
    wins = set()
    for t in range(total):
        hits = [x for x in sources if x[0] <= t and (t-x[0]) % x[1] == 0]
        if not hits:
            continue
        winner = hits[-1]
        if t >= winner[0]+winner[1]:
            repeat.add(winner[1])
        if len(sources) >= 2 and t >= sources[-1][0] and winner[0] < sources[-1][0]:
            old_survives = True
        for far in hits[:-1]:
            if far[2] != winner[2] and t > winner[0]:
                collision = True
                short_near |= winner[1] < far[1]
                long_near |= winner[1] > far[1]
                wins.add((winner[2],far[2]))
    return dict(short_near=short_near,long_near=long_near,old_survives=old_survives,
                collision=collision,repeat=repeat,wins=wins)


def _make_pair(rng,n,w,h,specs,roles,rotation,force_shuffle=True):
    bg,divider,guide = roles
    # Distinct joins or one forced self-loop: both have a unique complete chain.
    links = list(map(int,rng.choice(range(3,h-1),size=n-1,replace=False)))
    if n >= 3 and rng.random() < .30:
        j = int(rng.integers(n-2))
        links[j+1] = links[j]
    paths,panels = [],[]
    for i,source in enumerate(specs):
        path = _path(rng,h,w,None if i == 0 else links[i-1],None if i == n-1 else links[i])
        panel = np.full((h,w),bg,dtype=np.int8)
        for point in path:
            panel[point] = guide
        if source is not None:
            length,color = source
            col = int(rng.integers(1,w-length))
            panel[1,col:col+length] = color
        paths.append(path)
        panels.append(panel)
    output_panels = _paint_clocks(panels,paths,specs)
    layouts = [(nr,n//nr) for nr in range(1,n+1) if n % nr == 0
               and nr*(h+1)-1 <= 30 and (n//nr)*(w+1)-1 <= 30]
    layout = layouts[int(rng.integers(len(layouts)))]
    order = list(map(int,rng.permutation(n)))
    if force_shuffle and order == list(range(n)):
        order[0],order[-1] = order[-1],order[0]
    inp = _layout(panels,order,layout,divider)
    out = _layout(output_panels,range(n),(1,n),divider)
    return {'input':np.rot90(inp,rotation).copy(),'output':np.rot90(out,rotation).copy()}


def _count(name,value,lo,hi):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count('num_train',num_train,2,10)
    num_test = _count('num_test',num_test,1,4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,numbers.Integral):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int,rng.permutation(10)))
    bg,divider,guide = palette[:3]
    colors = palette[3:]
    a,b = ((2,3),(2,5),(3,4))[int(rng.integers(3))]
    slots = list(map(int,rng.choice(num_train,2,replace=False)))
    witness = {slots[0]:[(a,colors[0]),(b,colors[1])],
               slots[1]:[(b,colors[1]),(a,colors[0])]}
    base_rotation = int(rng.integers(4))
    profile = int(rng.integers(3))
    vary_roles = bool(rng.integers(2))
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            # No files, growing retry logs, or retained failed grids.
            for attempt in range(160):
                is_witness = split == 'train' and index in witness
                n = 3 if is_witness else int(rng.choice([2,3] if split == 'train' else [4,5]))
                wmin = max(5,max(a,b)+2) if is_witness else 4
                w = int(rng.integers(wmin, min(9,31//n-1)+1))
                h = int(rng.integers(max(7,n+3),13))
                if is_witness:
                    specs = witness[index] + [None]
                    # Source positions may be separated by a passive module.
                    if attempt < 100 and rng.random() < .30:
                        specs = [specs[0],None,specs[1]]
                else:
                    specs = [(int(rng.integers(1,min(w-2,max(a,b))+1)),int(rng.choice(colors)))
                             if rng.random() < .65 else None for _ in range(n)]
                    if all(s is None for s in specs):
                        specs[int(rng.integers(n))] = (int(rng.integers(1,min(w-2,max(a,b))+1)),colors[0])
                roles = (bg,divider,guide)
                if vary_roles and not is_witness:
                    # Witness colors remain reserved, so precedence reversals
                    # also rule out a persistent payload-color ranking.
                    candidates = [c for c in range(10) if c not in colors[:2]]
                    roles = tuple(map(int,rng.choice(candidates,3,replace=False)))
                    payloads = [c for c in range(10) if c not in roles]
                    specs = [None if s is None else (s[0],int(rng.choice(payloads))) for s in specs]
                rotation = int(rng.integers(4)) if profile == 1 else (base_rotation+2)%4 if profile == 2 and split == 'test' else base_rotation
                pair = _make_pair(rng,n,w,h,specs,roles,rotation)
                try:
                    info = _decode(pair['input'])
                    assert np.array_equal(_render(info),pair['output'])
                    if is_witness:
                        f = _features(info)
                        assert f['old_survives'] and f['collision']
                        assert {a,b} <= f['repeat']
                        assert f['short_near'] or f['long_near']
                        assert info['order'] != tuple(range(n))
                        assert not np.array_equal(_render(info,'reset'),pair['output'])
                        assert not np.array_equal(_render(info,'straight'),pair['output'])
                        assert not np.array_equal(_render(info,'count_dividers'),pair['output'])
                        assert not np.array_equal(_render(info,'header_phase'),pair['output'])
                    assert _key(pair['input']) not in seen
                except AssertionError:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded construction could not meet teaching constraints')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen,train_infos,test_infos = set(),[],[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(_render(info),out), 'phase, path order, precedence, or preserved-cell error'
            # Check preserved controls and geometry directly against output tiles.
            canonical = np.rot90(out,info['rotation'])
            n = len(info['panels'])
            h,w = info['panels'][0].shape
            assert canonical.shape == (h,n*(w+1)-1)
            assert np.all(canonical[:,w::w+1] == info['divider'])
            for position,i in enumerate(info['order']):
                tile = canonical[:,position*(w+1):position*(w+1)+w]
                panel = info['panels'][i]
                preserved = panel != info['guide']
                assert np.array_equal(tile[preserved],panel[preserved]), 'change outside track'
                allowed = {info['guide']} | {s[2] for s in info['sources']}
                assert set(map(int,tile[~preserved])) <= allowed
            (train_infos if split == 'train' else test_infos).append(info)
    evidence = [_features(i) for i in train_infos]
    assert any(f['short_near'] and f['old_survives'] for f in evidence)
    assert any(f['long_near'] and f['old_survives'] for f in evidence)
    wins = set().union(*(f['wins'] for f in evidence))
    assert any((b,a) in wins for a,b in wins), 'need opposite color-priority witnesses'
    assert len(set().union(*(f['repeat'] for f in evidence))) >= 2, 'visible period recurrence'
    assert any(i['order'] != tuple(range(len(i['panels']))) for i in train_infos)
    for mode in ('reset','local','oldest','shortest','longest','color_min','color_max','straight','count_dividers','header_phase'):
        assert any(not np.array_equal(_render(info,mode),np.asarray(pair['output']))
                   for info,pair in zip(train_infos,puzzle['train'])), 'training admits shortcut: '+mode
    assert max(len(i['panels']) for i in train_infos) < min(len(i['panels']) for i in test_infos), 'chain-length shift'


def self_test():
    shapes,geometry,rotations,roles,counts,periods,lengths,layouts = (set() for _ in range(8))
    hashes = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        hashes.add(hashlib.sha256(repr(_puzzle_key(puzzle)).encode()).hexdigest())
        for split in ('train','test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                signatures = [i['ports'] for i in info['infos']]
                complete = [order for order in itertools.permutations(range(len(signatures)))
                            if signatures[order[0]][0] is None and signatures[order[-1]][1] is None
                            and all(signatures[a][1] is not None and signatures[a][1] == signatures[b][0]
                                    for a,b in zip(order,order[1:]))]
                assert complete == [info['order']], 'permutation enumeration of chain uniqueness'
                assert np.array_equal(solve(pair['input'].tolist()),pair['output'])
                shapes.add(pair['input'].shape)
                rotations.add(info['rotation'])
                roles.add((info['bg'],info['divider'],info['guide']))
                counts.add(len(info['panels']))
                periods.update(s[1] for s in info['sources'])
                lengths.add(len(info['coordinates']))
                layouts.add(info['input_layout'])
                for p in info['panels']:
                    geometry.add(_key((p == info['guide']).astype(np.int8)))
                # Permuting input modules must not change the answer.
                reversed_input = _layout(info['panels'],range(len(info['panels'])-1,-1,-1),
                                         (1,len(info['panels'])),info['divider'])
                expected = np.rot90(pair['output'],info['rotation'])
                assert np.array_equal(solve(reversed_input),expected)
                # Arbitrary bijective recoloring is role-equivariant.
                mapping = np.arange(9,-1,-1,dtype=np.int8)
                assert np.array_equal(solve(mapping[pair['input']]),mapping[pair['output']])
                # Quarter-turns change neither path distance nor the rule.
                assert np.array_equal(solve(np.rot90(pair['input'])),np.rot90(pair['output']))
    assert len(hashes) == 42
    assert counts == {2,3,4,5} and rotations == {0,1,2,3}
    assert {1,2,3,4,5} <= periods
    assert len(shapes) >= 35 and len(geometry) >= 150
    assert len(roles) >= 35 and len(lengths) >= 30
    assert (2,2) in layouts and any(1 in x for x in layouts)
    # Hand-computed clock arithmetic, separate from grid construction/solve.
    # Source 2 at t=0; source 3 at t=5. Newest wins at t=8 and t=14;
    # the older source still contributes at t=6,10,12.
    dummy = dict(panels=[np.full((1,16),5,dtype=np.int8)],order=(0,),divider=8,
                 rotation=0,coordinates=[(0,0,t) for t in range(16)],
                 sources=[(0,2,1),(5,3,2)],offsets=[0,5])
    assert _render(dummy).tolist() == [[1,5,1,5,1,2,1,5,2,5,1,2,1,5,2,5]]
    # Unique self-loop versus interchangeable self-loops.
    assert len(_orders([(None,3),(3,3),(3,None)])) == 1
    assert len(_orders([(None,3),(3,3),(3,3),(3,None)])) == 2
    # Supported semantic members outside the narrower random sampler:
    # six width-3 modules, forced self-loops, and an initial passive prefix.
    links = [3,3,4,4,5]
    panels,paths = [],[]
    specs = [None,(1,7),None,(1,2),None,None]
    for i,source in enumerate(specs):
        r0 = 3 if i == 0 else links[i-1]
        r1 = 5 if i == 5 else links[i]
        c0,c1 = (1 if i == 0 else 0),(1 if i == 5 else 2)
        path = [(r0,c0)]
        if c0 == 0:
            path.append((r0,1))
        for r in range(r0+1,r1+1):
            path.append((r,1))
        if c1 == 2:
            path.append((r1,2))
        p = np.zeros((8,3),dtype=np.int8)
        for point in path:
            p[point] = 5
        if source:
            p[1,1] = source[1]
        panels.append(p)
        paths.append(path)
    inp = _layout(panels,[2,5,0,4,1,3],(2,3),8)
    expected = _layout(_paint_clocks(panels,paths,specs),range(6),(1,6),8)
    for k in range(4):
        assert np.array_equal(solve(np.rot90(inp,k)),np.rot90(expected,k))
    # A valid path may double back: the solver follows adjacency rather than
    # relying on the constructor's monotone coordinate vocabulary.
    paths = [[(3,c) for c in range(1,6)]+[(4,5),(5,5),(5,4),(5,3),(6,3),(7,3)]
             +[(7,c) for c in range(4,9)],[(7,c) for c in range(5)]]
    specs = [(4,1),(3,2)]
    panels = []
    for path,(period,color) in zip(paths,specs):
        p = np.zeros((9,9),dtype=np.int8)
        for point in path:
            p[point] = 5
        p[1,1:period+1] = color
        panels.append(p)
    inp = _layout(panels,[1,0],(2,1),8)
    expected = _layout(_paint_clocks(panels,paths,specs),range(2),(1,2),8)
    assert np.array_equal(solve(inp),expected)
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            validate_generated_puzzle(generate_puzzle(991+a*7+b,a,b))
    for counts_bad in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(None,1)):
        try:
            generate_puzzle(0,*counts_bad)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    original = generate_puzzle(0,2,1)
    damaged = {s:[{k:np.asarray(v).copy() for k,v in p.items()} for p in original[s]] for s in original}
    damaged['test'][0]['output'][0,0] = (int(damaged['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('preserved-cell mutation accepted')
