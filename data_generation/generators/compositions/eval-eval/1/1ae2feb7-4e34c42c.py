"""Broken pulse trail: translation changes the runs that drive a path clock.

Only standard-library modules and NumPy are used. No files are read or written
by generation, solving, validation, or the bounded 42-seed self-test.
"""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np


HOST_SEEDS = (0,231913237,17524755,4646178,809633268,962018864,822081,437677216,158258751,55866999,244684389,699037891,155204035,984675284,647893531,398768349,835610240,312586150,739678576,297704909,62922267,494222436,764321785,889843538,878657144,412417634,442095638,181491512,628462288,328484630,66103786,425667241,298423312,468442517,501596498,966372280,711881064,692977355,88742970,223941534,231065202,74560330)
DIRS = ((-1,0),(1,0),(0,-1),(0,1))


def _neighbors(p):
    return [(p[0]+r,p[1]+c) for r,c in DIRS]


def _groups(points):
    remaining, result = set(points), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            for q in _neighbors(stack.pop()):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _normalize(cells):
    r0 = min(r for r,c in cells)
    c0 = min(c for r,c in cells)
    return {(r-r0,c-c0):int(v) for (r,c),v in cells.items()}


def _shape(cells):
    return max(r for r,c in cells)+1,max(c for r,c in cells)+1


def _render(cells, background):
    cells = _normalize(cells)
    out = np.full(_shape(cells),background,dtype=np.int8)
    for p,v in cells.items():
        out[p] = v
    return out


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid dimensions'
    assert np.issubdtype(a.dtype,np.integer) and np.all((a >= 0)&(a <= 9)), 'invalid colors'
    return a


def _key(a):
    return a.shape,a.astype(np.int8).tobytes()


def _windows(cells):
    """Four distinct colors in a filled 2x2 are the registration alphabet."""
    found = {}
    for r,c in sorted(cells):
        square = ((r,c),(r,c+1),(r+1,c),(r+1,c+1))
        if all(p in cells for p in square):
            pattern = tuple(cells[p] for p in square)
            if len(set(pattern)) == 4:
                assert pattern not in found, 'a key pattern occurs twice in one figure'
                found[pattern] = (r,c)
    return found


def _register(parts):
    """All repeated keys impose equations; there is no overlap-size score."""
    occurrences = {}
    for i,part in enumerate(parts):
        for pattern,p in _windows(part).items():
            occurrences.setdefault(pattern,[]).append((i,p))
    edges = [[] for _ in parts]
    for copies in occurrences.values():
        for i,p in copies:
            for j,q in copies:
                if i != j:
                    edges[i].append((j,(p[0]-q[0],p[1]-q[1])))
    offsets, stack = {0:(0,0)}, [0]
    while stack:
        i = stack.pop()
        for j,(dr,dc) in edges[i]:
            candidate = offsets[i][0]+dr,offsets[i][1]+dc
            if j in offsets:
                assert offsets[j] == candidate, 'inconsistent registration'
            else:
                offsets[j] = candidate
                stack.append(j)
    assert len(offsets) == len(parts), 'disconnected key graph'
    union, placed = {}, []
    for i,part in enumerate(parts):
        dr,dc = offsets[i]
        moved = {(r+dr,c+dc):v for (r,c),v in part.items()}
        for p,v in moved.items():
            assert p not in union or union[p] == v, 'conflicting foreground'
            union[p] = v
        placed.append(moved)
    # Every key has exactly one location in the recovered object.
    _windows(union)
    return union,placed


def _analyze(input_grid):
    a = _grid(input_grid)
    values,counts = np.unique(a,return_counts=True)
    assert counts.max()*2 > a.size, 'background must occupy over half the input'
    bg = int(values[counts.argmax()])
    cells = {tuple(map(int,p)):int(a[tuple(p)]) for p in np.argwhere(a != bg)}
    parts = [_normalize({p:cells[p] for p in group}) for group in _groups(cells)]
    assert 2 <= len(parts) <= 8, 'need separated keyed fragments'
    union,placed = _register(parts)
    assert all(1 <= n <= 30 for n in _shape(_normalize(union))), 'assembly too large'
    windows = _windows(union)
    assert windows, 'no registration keys'
    marker_colors = set().union(*(set(p) for p in windows))
    gates = []
    for r,c in sorted(union):
        color = union[r,c]
        if color in marker_colors:
            continue
        square = {(r+i,c+j) for i in range(3) for j in range(3)}
        if all(union.get(p) == color for p in square):
            gates.append(((r+1,c+1),color,square))
    assert len(gates) == 1, 'need a unique solid 3x3 gate'
    center,gate_color,gate = gates[0]
    assert {p for p,v in union.items() if v == gate_color} == gate, 'gate color is exclusive'
    body = {p:v for p,v in union.items() if v not in marker_colors and v != gate_color}
    groups = _groups(body)
    assert len(groups) == 2, 'gate must separate source and trail'
    sources = [g for g in groups if len({body[p] for p in g}) >= 2]
    trails = [g for g in groups if len({body[p] for p in g}) == 1]
    assert len(sources) == len(trails) == 1, 'multicolor source and plain trail required'
    source,trail = sources[0],trails[0]
    distance = lambda p: abs(p[0]-center[0])+abs(p[1]-center[1])
    source_order = sorted(source,key=lambda p:(distance(p),p))
    first = source_order[0]
    assert distance(first) == 2, 'source must meet a side center'
    delta = ((first[0]-center[0])//2,(first[1]-center[1])//2)
    assert delta in DIRS
    assert source_order == [(center[0]+delta[0]*(i+2),center[1]+delta[1]*(i+2))
                            for i in range(len(source_order))], 'source must be a straight strip'
    start = center[0]-2*delta[0],center[1]-2*delta[1]
    assert start in trail and len(trail) >= 2
    degree = {p:sum(q in trail for q in _neighbors(p)) for p in trail}
    assert degree[start] == 1 and list(degree.values()).count(1) == 2
    assert all(d in (1,2) for d in degree.values()), 'trail must be an unbranched simple path'
    path,previous,current = [],None,start
    while True:
        path.append(current)
        next_cells = [q for q in _neighbors(current) if q in trail and q != previous]
        if not next_cells:
            break
        assert len(next_cells) == 1 and next_cells[0] not in path
        previous,current = current,next_cells[0]
    assert len(path) == len(trail)
    blocks = []
    for p in source_order:
        if not blocks or blocks[-1]['color'] != body[p]:
            blocks.append({'color':body[p],'positions':[]})
        blocks[-1]['positions'].append(p)
    for block in blocks:
        block['length'] = len(block['positions'])
    # Marker stems may touch the body, but never become clock steps. Every
    # marker-color region must contain a dense key, rather than unrelated noise.
    marker_points = {p for p,v in union.items() if v in marker_colors}
    key_points = {q for r,c in windows.values() for q in ((r,c),(r,c+1),(r+1,c),(r+1,c+1))}
    assert all(g & key_points for g in _groups(marker_points)), 'unkeyed annotation'
    return {'background':bg,'parts':parts,'placed':placed,'union':union,'path':path,
            'blocks':blocks,'source':source_order,'gate':gate,'center':center,
            'marker_colors':marker_colors,'windows':windows,
            'trail_color':body[start]}


def _paint(info, alternative=None):
    cells,blocks = dict(info['union']),info['blocks']
    lengths = [b['length'] for b in blocks]
    if alternative in ('raw_sum','raw_max'):
        lengths = []
        for block in blocks:
            points = set(block['positions'])
            counts = [len(points & set(part)) for part in info['placed']]
            lengths.append(sum(counts) if alternative == 'raw_sum' else max(counts))
    local,old_direction = 0,None
    for t,p in enumerate(info['path']):
        clock = t
        if alternative == 'bend_reset':
            if t:
                q = info['path'][t-1]
                direction = p[0]-q[0],p[1]-q[1]
                if old_direction is not None and direction != old_direction:
                    local = 0
                old_direction = direction
            clock = local
            local += 1
        if alternative == 'manhattan':
            q = info['path'][0]
            clock = abs(p[0]-q[0])+abs(p[1]-q[1])
        hits = [i for i,n in enumerate(lengths) if clock % n == 0]
        if hits:
            if alternative == 'short':
                winner = min(hits,key=lambda i:(lengths[i],i))
            elif alternative == 'long':
                winner = min(hits,key=lambda i:(-lengths[i],i))
            elif alternative == 'far':
                winner = hits[-1]
            else:
                winner = hits[0]
            cells[p] = blocks[winner]['color']
    return _render(cells,info['background'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, registrations, periods and path distances from pixels."""
    return _paint(_analyze(input_grid))


def _walk(rng,length):
    """Non-self-touching walks with straight stretches; all four turns allowed."""
    for _ in range(100):
        path = [(0,2),(0,3),(0,4)]
        occupied = set(path) | {(r,c) for r in (-1,0,1) for c in (-1,0,1)}
        direction,stretch = (0,1),3
        while len(path) < length:
            choices,weights = [],[]
            for dr,dc in DIRS:
                if (dr,dc) == (-direction[0],-direction[1]):
                    continue
                if (dr,dc) != direction and stretch < 3:
                    continue
                q = path[-1][0]+dr,path[-1][1]+dc
                if not (-10 <= q[0] <= 10 and 2 <= q[1] <= 15):
                    continue
                if q in occupied or any(p in occupied and p != path[-1] for p in _neighbors(q)):
                    continue
                choices.append((dr,dc))
                weights.append(3.0 if (dr,dc) == direction and stretch < 7 else 1.0)
            if not choices:
                break
            step = choices[int(rng.choice(len(choices),p=np.array(weights)/sum(weights)))]
            stretch = stretch+1 if step == direction else 1
            direction = step
            q = path[-1][0]+step[0],path[-1][1]+step[1]
            path.append(q)
            occupied.add(q)
        if len(path) == length and _bends(path) >= 2:
            return path
    raise ValueError('walk construction exhausted')


def _bends(path):
    ds = [(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:])]
    return sum(a != b for a,b in zip(ds,ds[1:]))


def _motif(rng,colors,h,w):
    for _ in range(80):
        a = rng.choice(colors,size=(h,w))
        if all(len(set(map(int,a[r:r+2,c:c+2].flat))) == 4
               for r in range(h-1) for c in range(w-1)):
            return a
    # Constructive alternating four-color rectangle, not a saved motif bank.
    cs = list(map(int,rng.choice(colors,4,replace=False)))
    return np.array([[cs[2*(r%2)+c%2] for c in range(w)] for r in range(h)])


def _ornament(rng,anchor,tangent,occupied,marker_colors):
    normals = [d for d in DIRS if d[0]*tangent[0]+d[1]*tangent[1] == 0]
    for _ in range(30):
        normal = normals[int(rng.integers(2))]
        stem = int(rng.integers(1,4))
        h,w = ((2,2),(2,3),(3,2))[int(rng.integers(3))]
        offset = int(rng.integers(-w+1,1))
        motif = _motif(rng,marker_colors,h,w)
        color = int(rng.choice(marker_colors))
        additions = {(anchor[0]+normal[0]*d,anchor[1]+normal[1]*d):color for d in range(1,stem+1)}
        for r in range(h):
            for c in range(w):
                p = (anchor[0]+normal[0]*(stem+1+r)+tangent[0]*(offset+c),
                     anchor[1]+normal[1]*(stem+1+r)+tangent[1]*(offset+c))
                additions[p] = int(motif[r,c])
        if set(additions) & set(occupied):
            continue
        if any(q in occupied and q != anchor for p in additions for q in _neighbors(p)):
            continue
        try:
            _windows({**occupied,**additions})
        except AssertionError:
            continue
        return additions
    raise ValueError('key placement exhausted')


def _pack(parts,bg,rng):
    parts = [_normalize(p) for p in parts]
    shapes = [_shape(p) for p in parts]
    mh,mw = max(h for h,w in shapes),max(w for h,w in shapes)
    if mh > 30 or mw > 30:
        raise ValueError('fragment too large')
    for attempt in range(16):
        h = int(rng.integers(max(mh,12),31)) if attempt < 10 else 30
        w = int(rng.integers(max(mw,12),31)) if attempt < 10 else 30
        if h*w <= 2*sum(map(len,parts)):
            continue
        a = np.full((h,w),bg,dtype=np.int8)
        blocked = np.zeros((h,w),dtype=bool)
        order = sorted(range(len(parts)),key=lambda i:-(len(parts[i])+float(rng.random())*20))
        for i in order:
            ph,pw = shapes[i]
            feasible = np.ones((h-ph+1,w-pw+1),dtype=bool)
            for r,c in parts[i]:
                feasible &= ~blocked[r:r+h-ph+1,c:c+w-pw+1]
            positions = np.argwhere(feasible)
            if not len(positions):
                break
            r0,c0 = map(int,positions[int(rng.integers(len(positions)))])
            for (r,c),v in parts[i].items():
                r,c = r+r0,c+c0
                a[r,c] = v
                blocked[max(0,r-1):min(h,r+2),max(0,c-1):min(w,c+2)] = True
        else:
            return a
    raise ValueError('packing exhausted')


def _make_pair(rng,lengths,source_colors,trail_length,route_cuts,redundant,rotation):
    used = set(source_colors)
    available = [c for c in range(10) if c not in used]
    bg,gate,rail = map(int,rng.choice(available,3,replace=False))
    marker_colors = [c for c in available if c not in (bg,gate,rail)]
    if len(marker_colors) > 4 and rng.random() < .5:
        marker_colors = list(map(int,rng.choice(marker_colors,4,replace=False)))
    near,colors_at_source = [],[]
    for length,color in zip(lengths,source_colors):
        for _ in range(length):
            near.append((0,-2-len(near)))
            colors_at_source.append(color)
    path = _walk(rng,trail_length)
    gate_cells = {(r,c):gate for r in (-1,0,1) for c in (-1,0,1)}
    body = {p:c for p,c in zip(near,colors_at_source)}
    body.update(gate_cells)
    body.update({p:rail for p in path})
    body_order = near[::-1]+[(0,0)]+path
    gate_index = len(near)
    # Split inside a run: each view is strictly shorter than its merged period.
    eligible = [i for i,n in enumerate(lengths) if n >= 3]
    block = int(rng.choice(eligible))
    near_index = sum(lengths[:block])+int(rng.integers(1,lengths[block]-1))
    cuts = [len(near)-1-near_index]
    straight = [i for i in range(4,len(path)-4)
                if (path[i][0]-path[i-1][0],path[i][1]-path[i-1][1]) ==
                   (path[i+1][0]-path[i][0],path[i+1][1]-path[i][1])]
    selected = []
    for i in rng.permutation(straight):
        i = int(i)
        if all(abs(i-j) >= 5 for j in selected):
            selected.append(i)
            if len(selected) == route_cuts:
                break
    if route_cuts and len(selected) < route_cuts:
        raise ValueError('not enough separated straight joins')
    if route_cuts:
        cuts += [gate_index+1+i for i in selected]
    cuts.sort()
    ornaments,whole = {},dict(body)
    for cut in cuts:
        anchor = body_order[cut]
        if cut < gate_index:
            tangent = (0,1)
        else:
            i = cut-gate_index-1
            tangent = path[i+1][0]-path[i][0],path[i+1][1]-path[i][1]
        additions = _ornament(rng,anchor,tangent,whole,marker_colors)
        ornaments[cut] = additions
        whole.update(additions)
    assert max(_shape(_normalize(whole))) <= 30
    parts = []
    bounds = [0]+cuts+[len(body_order)-1]
    for lo,hi in zip(bounds,bounds[1:]):
        part = {}
        for index in range(lo,hi+1):
            if index == gate_index:
                part.update(gate_cells)
            else:
                p = body_order[index]
                part[p] = body[p]
            if index in ornaments:
                part.update(ornaments[index])
        parts.append(part)
    if redundant:
        cut = cuts[int(rng.integers(len(cuts)))]
        anchor = body_order[cut]
        parts.append({anchor:body[anchor],**ornaments[cut]})
    assert all(len(_groups(p)) == 1 for p in parts)
    inp = _pack(parts,bg,rng)
    # Labels use known full runs and far-to-near overwrite, without the parser.
    expected = dict(whole)
    for length,color in reversed(list(zip(lengths,source_colors))):
        for p in path[::length]:
            expected[p] = color
    out = _render(expected,bg)
    pair = {'input':np.rot90(inp,rotation).copy(),'output':np.rot90(out,rotation).copy()}
    info = _analyze(pair['input'])
    assert np.array_equal(_paint(info),pair['output']), 'construction and input-only reconstruction disagree'
    assert not np.array_equal(_paint(info,'raw_max'),pair['output']), 'assembly must affect periods'
    assert not np.array_equal(_paint(info,'raw_sum'),pair['output']), 'shared cells must count once'
    assert not np.array_equal(_paint(info,'bend_reset'),pair['output']), 'turns must expose clock continuity'
    assert not np.array_equal(_paint(info,'manhattan'),pair['output']), 'path distance must exceed displacement observably'
    return pair


def _counts(num_train,num_test):
    for name,n,low,high in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or not low <= n <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int,num_train: int=4,num_test: int=1) -> dict:
    _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    a,b = ((2,3),(2,5),(3,4),(3,5),(4,5))[int(rng.integers(5))]
    witness_colors = list(map(int,rng.choice(10,2,replace=False)))
    base_rotation = int(rng.integers(4))
    mixed = bool(rng.integers(2))
    absorb = bool(rng.integers(2))
    puzzle,seen,train_lengths = {'train':[],'test':[]},set(),[]
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            for attempt in range(100):
                if split == 'train' and index < 2:
                    lengths = [a,b] if index == 0 else [b,a]
                    colors = witness_colors if index == 0 else witness_colors[::-1]
                    trail = int(rng.integers(max(18,math.lcm(a,b)+2),25))
                else:
                    n = int(rng.choice([2,2,3]))
                    lengths = list(map(int,rng.integers(2,7,size=n)))
                    if sum(lengths) > 12 or max(lengths) < 3:
                        continue
                    colors = list(map(int,rng.choice(10,n,replace=False)))
                    trail = int(rng.integers(18,25)) if split == 'train' else int(rng.integers(max(train_lengths)+3,max(train_lengths)+10))
                route_cuts = (int(rng.integers(0,2)) if split == 'train' and index != 1
                              else 1 if split == 'train' else 2)
                rotation = int(rng.integers(4)) if mixed else base_rotation
                redundant = absorb and (index == 0 if split == 'train' else rng.random() < .55)
                try:
                    pair = _make_pair(rng,lengths,colors,trail,route_cuts,redundant,rotation)
                except (ValueError,AssertionError):
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train':
                        train_lengths.append(trail)
                    break
            else:
                raise RuntimeError(f'bounded construction exhausted at {split}/{index}')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    nt,nq = len(puzzle['train']),len(puzzle['test'])
    _counts(nt,nq)
    seen,infos = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _analyze(inp)
            union,path,bg = info['union'],info['path'],info['background']
            r0,c0 = min(r for r,c in union),min(c for r,c in union)
            assert out.shape == _shape(_normalize(union)), 'wrong crop'
            path_set = set(path)
            for p,v in union.items():
                if p not in path_set:
                    assert out[p[0]-r0,p[1]-c0] == v, 'preserved material changed'
            for r in range(out.shape[0]):
                for c in range(out.shape[1]):
                    if (r+r0,c+c0) not in union:
                        assert out[r,c] == bg, 'empty space changed'
            for t,p in enumerate(path):
                firing = [b for b in info['blocks'] if t % b['length'] == 0]
                expected = firing[0]['color'] if firing else info['trail_color']
                assert out[p[0]-r0,p[1]-c0] == expected, 'wrong period, phase, or priority'
            assert not np.array_equal(_paint(info,'raw_max'),out), 'split-run interaction absent'
            assert not np.array_equal(_paint(info,'raw_sum'),out), 'duplicate counting would be harmless'
            assert _bends(path) >= 2
            infos[split].append(info)
    assert min(len(i['path']) for i in infos['test']) > max(len(i['path']) for i in infos['train']), 'missing path-length shift'
    edges,orders = set(),set()
    for info in infos['train']:
        blocks = info['blocks']
        for i,near in enumerate(blocks):
            for far in blocks[i+1:]:
                visible_far = any(t % far['length'] == 0 and all(t % b['length'] != 0 for b in blocks[:i+1])
                                  and next(b for b in blocks if t % b['length'] == 0) is far
                                  for t in range(len(info['path'])))
                later_collision = any(t % near['length'] == t % far['length'] == 0
                                      for t in range(1,len(info['path'])))
                if visible_far and later_collision:
                    orders.add(int(np.sign(near['length']-far['length'])))
                    edges.add((near['color'],far['color']))
    assert {-1,1} <= orders and any((b,a) in edges for a,b in edges), 'priority demonstrations insufficient'
    for rival in ('short','long','far','raw_max','raw_sum','bend_reset','manhattan'):
        assert any(not np.array_equal(_paint(i,rival),np.asarray(p['output']))
                   for i,p in zip(infos['train'],puzzle['train'])), f'rival fits demonstrations: {rival}'
    train_absorbs = any(any(set(p) <= set().union(*(set(q) for q in i['placed'] if q is not p))
                               for p in i['placed']) for i in infos['train'])
    test_absorbs = any(any(set(p) <= set().union(*(set(q) for q in i['placed'] if q is not p))
                              for p in i['placed']) for i in infos['test'])
    assert not test_absorbs or train_absorbs, 'redundancy not demonstrated'


def self_test():
    signatures,structures,shapes,periods,backgrounds,bends,fragments,key_shapes = (set() for _ in range(8))
    total = 0
    first = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        if first is None:
            first = puzzle
        digest = hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                total += 1
                info = _analyze(pair['input'])
                assert np.array_equal(solve(pair['input']),pair['output'])
                for role in ('input','output'):
                    grid = pair[role]
                    digest.update(grid.tobytes())
                    labels = {}
                    normalized = bytes(labels.setdefault(int(v),len(labels)) for v in grid.flat)
                    structures.add((grid.shape,normalized))
                    shapes.add((role,*grid.shape))
                periods.add(tuple(b['length'] for b in info['blocks']))
                backgrounds.add(info['background'])
                bends.add(_bends(info['path']))
                fragments.add(len(info['parts']))
                marker = {p for p,v in info['union'].items() if v in info['marker_colors']}
                for group in _groups(marker):
                    key_shapes.add(_shape(_normalize({p:info['union'][p] for p in group})))
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(structures) >= 300
    assert len(shapes) >= 40 and len(periods) >= 25
    assert backgrounds == set(range(10)) and len(bends) >= 4
    assert len(fragments) >= 3 and len(key_shapes) >= 6
    # Exact arithmetic independent of sampled labels.
    fake = {'union':{(0,t):5 for t in range(13)},'path':[(0,t) for t in range(13)],
            'blocks':[{'length':3,'color':1},{'length':2,'color':2}], 'background':0}
    assert ''.join(map(str,_paint(fake)[0])) == '1521251521251'
    fake['blocks'].reverse()
    assert ''.join(map(str,_paint(fake)[0])) == '2521252521252'
    p = first['train'][0]
    mapping = np.array([7,2,8,0,9,3,1,6,4,5],dtype=np.int8)
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(p['input'],turn)),np.rot90(p['output'],turn))
        assert np.array_equal(solve(mapping[np.rot90(p['input'],turn)]),mapping[np.rot90(p['output'],turn)])
    info = _analyze(p['input'])
    repacked = _pack(info['parts'],info['background'],np.random.default_rng(991))
    assert np.array_equal(solve(repacked),p['output']), 'input positions must not encode registration'
    again = generate_puzzle(HOST_SEEDS[0])
    assert all(np.array_equal(a[k],b[k]) for s in first for a,b in zip(first[s],again[s]) for k in a)
    bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in first[s]] for s in first}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    for counts in ((2,1),(10,4)):
        validate_generated_puzzle(generate_puzzle(6543,*counts))
    for counts in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(None,1)):
        try:
            generate_puzzle(0,*counts)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    return {'seeds':42,'pairs':total,'distinct_puzzles':len(signatures),
            'color_normalized_structures':len(structures),'grid_shapes':len(shapes),
            'ordered_period_tuples':len(periods),'backgrounds':sorted(backgrounds),
            'bend_counts':sorted(bends),'fragment_counts':sorted(fragments),
            'annotation_shapes':len(key_shapes),'independent_solver':False}


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
