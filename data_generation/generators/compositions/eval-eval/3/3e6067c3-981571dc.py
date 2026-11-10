"""Mirror-addressed relays: jointly recover labels and decode an ordered route.

Only NumPy and the standard library are used. Generation has no file I/O;
solve receives only the visible input. Search is over at most 25 stations,
one marked start, and at most 11 key entries.
"""
from hashlib import sha256
import numpy as np


HOST_SEEDS = (0,426994139,740647619,622508679,606062220,547121107,699382954,
              440239629,987054927,42784409,458340573,979319017,16588789,
              209199222,360361587,239784831,240958493,952388779,126404185,
              591065957,163472133,804611765,743097444,380874259,544458137,
              358363002,195643289,719357985,965058638,164169943,184677820,
              426383194,449212865,325163758,684556967,301393039,925284506,
              648466571,499411122,998764902,993253510,962474530)

# Guaranteed termination scenes, indexed by (train/test, unwitnessed orbits).
# Values are indexed by sorted geometric orbits. Keys are derived from paths.
# These are small emergency constructions, not the normal layout distribution.
_FALLBACKS = {
    (False,0): ([(0,0),(0,1),(0,2),(1,1),(2,0),(2,1),(2,2)],
                [0,0,1],[-1,0,-1,1,0,0,-1],[2,1,3,5,4,0]),
    (False,1): ([(0,0),(0,1),(0,2),(1,1),(2,0),(2,1),(2,2)],
                [2,1,0],[-1,-1,-1,0,2,-1,-1],[4,0,1,2,6,5]),
    (False,2): ([(0,0),(0,2),(1,0),(1,1),(1,2),(2,0),(2,2)],
                [1,2,0],[-1,-1,-1,-1,-1,-1,1],[0,2,3,4,6,5]),
    (True,0): ([(r,c) for r in range(3) for c in range(3)],
               [1,1,0,2],[-1,1,-1,0,2,0,1,-1,-1],[0,1,2,5,4,7,6,3]),
    (True,1): ([(r,c) for r in range(3) for c in range(3)],
               [0,0,1,1],[-1,-1,-1,-1,1,1,0,-1,-1],[0,1,2,5,4,7,6,3]),
    (True,2): ([(r,c) for r in range(3) for c in range(3)],
               [3,1,0,2],[-1,-1,-1,0,-1,0,3,-1,-1],[0,1,2,5,4,7,6,3]),
}


def _fallback_scene(test, unknown_count):
    nodes, colors, visible, path = _FALLBACKS[test,unknown_count]
    groups = [(min(r,2-r),min(c,2-c)) for r,c in nodes]
    values = dict(zip(sorted(set(groups)),colors))
    return dict(nr=3,nc=3,nodes=nodes,values=values,visible=visible,
                key=[values[groups[i]] for i in path],start=path[0],path=path,groups=groups)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        todo, part = [p], []
        while todo:
            r, c = todo.pop()
            part.append((int(r), int(c)))
            for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(part)
    return result


def _band(a, b, size, patch):
    off = (size-patch)//2
    if a[0] == b[0] and abs(a[1]-b[1]) > size:
        return (slice(a[0]+off,a[0]+off+patch),
                slice(min(a[1],b[1])+size,max(a[1],b[1])))
    if a[1] == b[1] and abs(a[0]-b[0]) > size:
        return (slice(min(a[0],b[0])+size,max(a[0],b[0])),
                slice(a[1]+off,a[1]+off+patch))
    return None


def _parse(grid):
    grid = _grid(grid)
    bg = int(grid[0,0])
    assert all(np.all(edge == bg) for edge in
               (grid[0],grid[-1],grid[:,0],grid[:,-1]))
    squares, keys = [], []
    for part in _components(grid != bg):
        yy, xx = zip(*part)
        y, x, h, w = min(yy), min(xx), max(yy)-min(yy)+1, max(xx)-min(xx)+1
        block = grid[y:y+h,x:x+w]
        if h != w:
            assert h == 3 and w % 2 == 1
            keys.append((y,x,block))
        else:
            assert 3 <= h <= 6
            squares.append((y,x,block))
    assert len(keys) == 1 and 4 <= len(squares) <= 25
    ky, kx, kb = keys[0]
    start_color = int(kb[0,0])
    assert start_color != bg
    assert np.all(kb[[0,-1],:] == start_color) and np.all(kb[:,[0,-1]] == start_color)
    key = [int(c) for c in kb[1,1:-1:2]]
    assert 4 <= len(key) <= 11 and len(set(key)) >= 2
    assert np.all(kb[1,2:-1:2] == bg)
    squares.sort(key=lambda p:(p[0],p[1]))
    stations, labels, frames, signature = [], [], [], None
    for y, x, block in squares:
        size, frame = len(block), int(block[0,0])
        dy, dx = np.nonzero(block != frame)
        assert len(dy)
        off, patch = int(dy.min()), int(dy.max()-dy.min()+1)
        assert off > 0 and 2*off+patch == size and len(dy) == patch*patch
        assert int(dx.min()) == off and int(dx.max()) == off+patch-1
        label = int(block[off,off])
        assert np.all(block[off:off+patch,off:off+patch] == label)
        current = (size,patch)
        assert signature is None or current == signature
        signature = current
        stations.append((y,x))
        labels.append(label)
        frames.append(frame)
    size, patch = signature
    assert frames.count(start_color) == 1
    start = frames.index(start_color)
    carriers = set(frames)-{start_color}
    assert len(carriers) == 1
    carrier = next(iter(carriers))
    assert carrier not in (bg,start_color)
    assert all(c not in (bg,carrier,start_color) for c in key)
    assert all(c == bg or c not in (carrier,start_color) for c in labels)
    assert ky >= max(y+size for y,x in stations)+1
    rs = min(y for y,x in stations)+max(y for y,x in stations)
    cs = min(x for y,x in stations)+max(x for y,x in stations)
    positions = set(stations)
    assert len({y for y,x in positions}) >= 2 and len({x for y,x in positions}) >= 2
    assert all((rs-y,x) in positions and (y,cs-x) in positions for y,x in stations)
    groups = [(min(y,rs-y),min(x,cs-x)) for y,x in stations]
    known = {}
    for g, c in zip(groups,labels):
        if c != bg:
            assert known.get(g,c) == c, 'surviving mirror labels disagree'
            known[g] = c
    unknown = set(groups)-set(known)
    adj = [[] for _ in stations]
    bands, bits = {}, {}
    for i, a in enumerate(stations):
        for j in range(i+1,len(stations)):
            b = stations[j]
            band = _band(a,b,size,patch)
            if band is None or not np.all(grid[band] == bg):
                continue
            adj[i].append(j)
            adj[j].append(i)
            bit = sum(1 << (r*grid.shape[1]+c)
                      for r in range(band[0].start,band[0].stop)
                      for c in range(band[1].start,band[1].stop))
            bands[i,j] = bands[j,i] = band
            bits[i,j] = bits[j,i] = bit
    assert all(len(n) <= 4 for n in adj)
    return dict(bg=bg,carrier=carrier,start_color=start_color,size=size,patch=patch,
                key=key,key_box=(ky,kx,kb.shape[1]),stations=stations,labels=labels,
                groups=groups,known=known,unknown=unknown,start=start,adj=adj,
                bands=bands,bits=bits,axes=(rs+size-1,cs+size-1))


def _routes(info, symmetry=True, limit=2):
    """Joint equality/route constraints; no assignment is taken from generation."""
    key, groups = info['key'], info['groups']
    solutions = []
    def visit(path, used, assigned, painted):
        if len(solutions) >= limit:
            return
        if len(path) == len(key):
            if not symmetry or info['unknown'] <= assigned.keys():
                solutions.append((tuple(path),assigned.copy()))
            return
        color = key[len(path)]
        choices = [info['start']] if not path else info['adj'][path[-1]]
        for j in choices:
            if used & (1 << j):
                continue
            edge = info['bits'][path[-1],j] if path else 0
            if painted & edge:
                continue
            if symmetry:
                group = groups[j]
                if assigned.get(group,color) != color:
                    continue
                visit(path+[j],used | (1 << j),assigned | {group:color},painted | edge)
            elif info['labels'][j] in (info['bg'],color):
                visit(path+[j],used | (1 << j),assigned,painted | edge)
    visit([],0,info['known'].copy() if symmetry else {},0)
    return solutions


def _complete(grid, info, route, assigned):
    out = np.array(grid,copy=True)
    off = (info['size']-info['patch'])//2
    for (y,x), group in zip(info['stations'],info['groups']):
        assert group in assigned
        out[y+off:y+off+info['patch'],x+off:x+off+info['patch']] = assigned[group]
    for k,(i,j) in enumerate(zip(route,route[1:])):
        out[info['bands'][i,j]] = info['key'][k]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the unique route and all labels using this input alone."""
    grid = np.asarray(input_grid)
    info = _parse(grid)
    answers = _routes(info)
    if len(answers) != 1:
        raise ValueError('input must have one joint label-and-route solution')
    return _complete(grid,info,*answers[0])


def _visibility(nodes):
    graph = [[] for _ in nodes]
    for i,a in enumerate(nodes):
        for j,b in enumerate(nodes):
            if i == j:
                continue
            if a[0] == b[0]:
                clear = not any(c[0] == a[0] and min(a[1],b[1]) < c[1] < max(a[1],b[1]) for c in nodes)
            elif a[1] == b[1]:
                clear = not any(c[1] == a[1] and min(a[0],b[0]) < c[0] < max(a[0],b[0]) for c in nodes)
            else:
                continue
            if clear:
                graph[i].append(j)
    return graph


def _sample_scene(rng, unknown_count, length, max_levels):
    nr, nc = (int(rng.integers(3,max_levels+1)) for _ in range(2))
    universe = [(r,c) for r in range(nr) for c in range(nc)]
    orbit = lambda p:(min(p[0],nr-1-p[0]),min(p[1],nc-1-p[1]))
    all_groups = sorted({orbit(p) for p in universe})
    removed = {all_groups[int(rng.integers(len(all_groups)))]} if rng.random() < .4 else set()
    nodes = [p for p in universe if orbit(p) not in removed]
    if length >= len(nodes):
        return None
    # Removing a whole outer level would move the inferred axes. Reject it.
    if {0,nr-1} - {r for r,c in nodes} or {0,nc-1} - {c for r,c in nodes}:
        return None
    groups = [orbit(p) for p in nodes]
    graph = _visibility(nodes)
    start = int(rng.integers(len(nodes)))
    path = [start]
    for _ in range(length-1):
        choices = [j for j in graph[path[-1]] if j not in path]
        if not choices:
            return None
        path.append(int(rng.choice(choices)))
    axes = {nodes[i][0] == nodes[j][0] for i,j in zip(path,path[1:])}
    if len(axes) < 2:
        return None
    group_set = sorted(set(groups))
    count = int(rng.integers(2,min(7,len(group_set))+1))
    values = {g:int(rng.integers(count)) for g in group_set}
    key = [values[groups[i]] for i in path]
    if len(set(key)) < 2:
        return None
    used_groups = sorted({groups[i] for i in path})
    if len(used_groups) <= unknown_count:
        return None
    unknown = {used_groups[int(j)] for j in rng.choice(len(used_groups),unknown_count,replace=False)}
    if unknown and not any(groups.count(g) >= 2 for g in unknown):
        return None
    hard_groups = [g for g in group_set if g not in unknown and groups.count(g) == 4
                   and any(groups[i] == g for i in path[:-1])]
    if not hard_groups:
        return None
    hard = hard_groups[int(rng.integers(len(hard_groups)))]
    target = int(rng.choice([i for i in path[:-1] if groups[i] == hard]))
    r,c = nodes[target]
    donor = nodes.index((nr-1-r,nc-1-c))
    visible = [-1]*len(nodes)
    for g in group_set:
        if g in unknown:
            continue
        members = [i for i,x in enumerate(groups) if x == g]
        survivors = [donor] if g == hard else rng.choice(members,int(rng.integers(1,len(members)+1)),replace=False)
        for i in survivors:
            visible[int(i)] = values[g]
    return dict(nr=nr,nc=nc,nodes=nodes,values=values,visible=visible,key=key,
                start=start,path=path,groups=groups)


def _coords(rng, count, size, limit):
    gaps = [1]*(count-1)
    spare = limit-count*size-(count-1)
    assert spare >= 0
    for j in rng.permutation((count-1+1)//2):
        j = int(j)
        mirror = count-2-j
        cost = 1 if j == mirror else 2
        extra = int(rng.integers(0,min(5,spare//cost)+1))
        gaps[j] += extra
        if j != mirror:
            gaps[mirror] += extra
        spare -= cost*extra
    return [0]+[int(v) for v in np.cumsum(size+np.array(gaps))]


def _draw(rng, scene, size, patch):
    bg, carrier, start_color = (int(v) for v in rng.choice(10,3,replace=False))
    palette = [int(v) for v in rng.permutation([c for c in range(10) if c not in (bg,carrier,start_color)])]
    yy = _coords(rng,scene['nr'],size,24)
    xx = _coords(rng,scene['nc'],size,28)
    bh, bw = yy[-1]+size, xx[-1]+size
    key = [palette[c] for c in scene['key']]
    kh, kw = 3, 2*len(key)+1
    h = int(rng.integers(bh+6,31))
    w = int(rng.integers(max(bw,kw)+2,31))
    top = int(rng.integers(1,h-bh-4))
    left = int(rng.integers(1,w-bw))
    ky = int(rng.integers(top+bh+1,h-3))
    kx = int(rng.integers(1,w-kw))
    inp = np.full((h,w),bg,dtype=np.int8)
    out = inp.copy()
    positions = [(yy[r]+top,xx[c]+left) for r,c in scene['nodes']]
    off = (size-patch)//2
    for i,(y,x) in enumerate(positions):
        frame = start_color if i == scene['start'] else carrier
        inp[y:y+size,x:x+size] = frame
        out[y:y+size,x:x+size] = frame
        label = palette[scene['values'][scene['groups'][i]]]
        visible = bg if scene['visible'][i] < 0 else palette[scene['visible'][i]]
        inp[y+off:y+off+patch,x+off:x+off+patch] = visible
        out[y+off:y+off+patch,x+off:x+off+patch] = label
    for grid in (inp,out):
        grid[ky:ky+kh,kx:kx+kw] = start_color
        grid[ky+1,kx+1:kx+kw-1] = bg
        grid[ky+1,kx+1:kx+kw-1:2] = key
    for k,(i,j) in enumerate(zip(scene['path'],scene['path'][1:])):
        out[_band(positions[i],positions[j],size,patch)] = key[k]
    return dict(input=inp,output=out)


def _signature(array):
    a = np.asarray(array,dtype=np.int8)
    return a.shape,a.tobytes()


def _count(value, name, low, high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train,'num_train',2,10), _count(num_test,'num_test',1,4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    widths = [int(v) for v in rng.permutation([1,3])]
    puzzle, seen = {'train':[],'test':[]}, set()
    for split,count in (('train',nt),('test',nq)):
        for index in range(count):
            anchor = split == 'train' and index < 2
            size = 5 if anchor else int(rng.choice([3,3,4,5,6]))
            widths_allowed = list(range(1 if size%2 else 2,size-1,2))
            patch = widths[index] if anchor else int(rng.choice(widths_allowed))
            unknown_count = index if anchor else int(rng.integers(1 if split == 'test' else 0,3))
            max_levels = min(5,25//(size+1))
            length = int(rng.integers(4,8)) if split == 'train' else int(rng.integers(8,11))
            # Size 6 permits only nine stations, with at least one unused.
            length = min(length,max_levels*max_levels-1)
            for attempt in range(512):
                scene = _sample_scene(rng,unknown_count,length,max_levels)
                if scene is None:
                    continue
                if split == 'train' and index == 0 and not any(
                        i not in scene['path'] and c < 0 for i,c in enumerate(scene['visible'])):
                    continue
                pair = _draw(rng,scene,size,patch)
                info = _parse(pair['input'])
                answers = _routes(info)
                if len(answers) != 1 or answers[0][0] != tuple(scene['path']):
                    continue
                if len(_routes(info,False)) < 2:
                    continue
                if not np.array_equal(_complete(pair['input'],info,*answers[0]),pair['output']):
                    continue
                sig = _signature(pair['input'])
                if sig in seen:
                    continue
                seen.add(sig)
                puzzle[split].append(pair)
                break
            else:
                pair = _draw(rng,_fallback_scene(split == 'test',unknown_count),size,patch)
                # At least five colors are present; the 40 affine permutations
                # act distinctly. At most 13 earlier inputs can be forbidden.
                found = None
                for multiplier in (1,3,7,9):
                    for shift in range(10):
                        remap = (multiplier*np.arange(10)+shift)%10
                        candidate = {k:remap[a].astype(np.int8) for k,a in pair.items()}
                        sig = _signature(candidate['input'])
                        if sig not in seen:
                            found = candidate
                            break
                    if found is not None:
                        break
                assert found is not None
                seen.add(_signature(found['input']))
                puzzle[split].append(found)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, route):
    diagonal = 0
    for i,((y,x),color) in enumerate(zip(info['stations'],info['labels'])):
        g = info['groups'][i]
        if color == info['bg'] and g in info['known']:
            members = [j for j,h in enumerate(info['groups']) if h == g]
            survivors = [j for j in members if info['labels'][j] != info['bg']]
            if len(members) == 4 and len(survivors) == 1:
                yy,xx = info['stations'][survivors[0]]
                diagonal += yy != y and xx != x and i in route[:-1]
    lengths = [int((info['bands'][i,j][0].stop-info['bands'][i,j][0].start)*
                   (info['bands'][i,j][1].stop-info['bands'][i,j][1].start)//info['patch'])
               for i,j in zip(route,route[1:])]
    ys, xs = sorted({p[0] for p in info['stations']}), sorted({p[1] for p in info['stations']})
    return dict(stations=len(info['stations']),length=len(route),unknown=len(info['unknown']),
                erased=sum(c == info['bg'] for c in info['labels']),diagonal=int(diagonal),
                healed_unused=sum(i not in route and c == info['bg'] for i,c in enumerate(info['labels'])),
                size=info['size'],patch=info['patch'],rows=len(ys),cols=len(xs),
                sparse=len(info['stations']) < len(ys)*len(xs),gaps=tuple(lengths),
                orbit_count=len(set(info['groups'])),palette=len(set(info['key'])),
                axes_parity=tuple(a%2 for a in info['axes']),
                layout=tuple((y-ys[0],x-xs[0]) for y,x in info['stations']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape == out.shape
            sig = _signature(inp)
            assert sig not in seen
            seen.add(sig)
            info = _parse(inp)
            answers = _routes(info)
            assert len(answers) == 1, 'ambiguous joint decoding'
            route, assigned = answers[0]
            expected = _complete(inp,info,route,assigned)
            assert np.array_equal(out,expected), 'incorrect labels, links or preservation'
            assert not np.array_equal(inp,out)
            # Symmetry must constrain the physical route, not just decorate it.
            relaxed = _routes(info,False)
            assert len(relaxed) == 2, 'mirrors are unnecessary for this route'
            masks = []
            for relaxed_route,_ in relaxed:
                mask = 0
                for i,j in zip(relaxed_route,relaxed_route[1:]):
                    mask |= info['bits'][i,j]
                masks.append(mask)
            assert masks[0] != masks[1], 'ablation routes must paint different masks'
            assert not info['unknown'] or any(info['groups'].count(g) >= 2 for g in info['unknown'])
            f = _features(info,route)
            assert f['diagonal'] > 0, 'no outgoing station needs both folds'
            assert f['stations'] > f['length'], 'unused station evidence missing'
            features[split].append(f)
    train, test = features['train'],features['test']
    assert any(f['unknown'] == 0 for f in train)
    assert any(f['unknown'] >= 1 for f in train), 'key must recover a whole erased orbit in training'
    assert any(f['healed_unused'] > 0 for f in train), 'training must heal an unused station'
    assert any(a['size'] == b['size'] and a['patch'] != b['patch'] for a in train for b in train)
    assert min(f['length'] for f in test) > max(f['length'] for f in train)


def self_test():
    """Bounded 42-seed study, count contracts, metamorphisms and corruption tests."""
    shapes, layouts, roles, parities, widths, lengths, signatures = set(),set(),set(),set(),set(),set(),set()
    sparse = unknown_two = diagonal = 0
    # Exercise termination guards directly, including larger patch geometries.
    for test in (False,True):
        for unknown in range(3):
            for size,patch in ((3,1),(4,2),(5,1),(5,3),(6,2),(6,4)):
                p = _draw(np.random.default_rng(60+unknown),_fallback_scene(test,unknown),size,patch)
                info = _parse(p['input'])
                answers = _routes(info)
                assert len(answers) == 1 and len(_routes(info,False)) == 2
                assert np.array_equal(solve(p['input']),p['output'])
                assert _features(info,answers[0][0])['diagonal'] > 0
    first = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = sha256()
        if first is None:
            first = puzzle
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp = pair['input']
                before = inp.copy()
                predicted = solve(inp)
                assert np.array_equal(inp,before), 'solve mutated input'
                assert np.array_equal(predicted,pair['output'])
                info = _parse(inp)
                route,_ = _routes(info)[0]
                f = _features(info,route)
                shapes.add(inp.shape)
                layouts.add(f['layout'])
                roles.add((info['bg'],info['carrier'],info['start_color']))
                parities.add(f['axes_parity'])
                widths.add((f['size'],f['patch']))
                lengths.add(f['length'])
                sparse += f['sparse']
                unknown_two += f['unknown'] == 2
                diagonal += f['diagonal']
                digest.update(repr(inp.shape).encode()+inp.tobytes()+pair['output'].tobytes())
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(layouts) >= 50 and len(shapes) >= 20
    assert len(roles) >= 40 and len(parities) == 4
    assert {3,4,5,6} <= {s for s,p in widths} and {1,2,3,4} <= {p for s,p in widths}
    assert {4,5,6,7,8,9,10} <= lengths
    assert sparse >= 10 and unknown_two >= 20 and diagonal >= 210
    repeat = generate_puzzle(HOST_SEEDS[0])
    assert all(np.array_equal(a[k],b[k]) for s in ('train','test')
               for a,b in zip(first[s],repeat[s]) for k in ('input','output'))
    for nt,nq in ((2,1),(3,2),(5,4),(10,1),(10,4)):
        validate_generated_puzzle(generate_puzzle(8000+nt*10+nq,nt,nq))
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Arbitrary palette permutations preserve all semantic roles.
    remap = np.array([9,3,7,0,6,1,8,2,5,4],dtype=np.int8)
    for pair in first['train']:
        assert np.array_equal(solve(remap[pair['input']]),remap[pair['output']])
    # Changed background, visible payload, healed label and a link must be rejected.
    inp = first['test'][0]['input']
    info = _parse(inp)
    route,assigned = _routes(info)[0]
    off = (info['size']-info['patch'])//2
    erased = next(i for i,c in enumerate(info['labels']) if c == info['bg'])
    visible = next(i for i,c in enumerate(info['labels']) if c != info['bg'])
    points = [(0,0)] + [(info['stations'][i][0]+off,info['stations'][i][1]+off) for i in (erased,visible)]
    band = info['bands'][route[0],route[1]]
    points.append((band[0].start,band[1].start))
    for point in points:
        bad = {s:[{k:a.copy() for k,a in p.items()} for p in first[s]] for s in first}
        out = bad['test'][0]['output']
        out[point] = (int(out[point])+1)%10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted output accepted')
    return dict(seeds=42,pairs=210,shapes=len(shapes),layouts=len(layouts),role_assignments=len(roles),
                axis_parities=sorted(parities),geometries=sorted(widths),route_lengths=sorted(lengths),
                sparse_pairs=int(sparse),two_unknown_orbit_pairs=int(unknown_two),diagonal_outgoing_nodes=int(diagonal))
