"""Relay printing: minimum-station routing with one binary exchange per hop.

Only NumPy and the standard library are used. No parent module is imported.
The source tile is itself one station, so there is no separate instruction panel.
"""
from collections import deque
from itertools import combinations
import heapq
import numpy as np


HOST_SEEDS = (0,806597772,126143676,526367887,833710199,856623172,204297860,822544243,380290316,968199378,947652493,120645816,692020927,511093005,661689909,808380748,869536110,455347320,981038037,52796934,599490799,925290640,933838046,522583310,345173894,868486513,402309432,572790369,836659309,571479627,136306791,572709064,873879389,821316158,670467928,584672732,788177997,385441614,708089945,961948048,696293843,87950432)
N4 = ((-1,0),(0,1),(1,0),(0,-1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r,c = stack.pop()
            cells.append((r,c))
            for dr,dc in N4:
                q = r+dr,c+dc
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(np.asarray(cells, dtype=np.int16))
    return result


def _square(cells):
    if not len(cells):
        return 0
    h,w = cells.max(0)-cells.min(0)+1
    return int(h) if h == w and len(cells) == h*w else 0


def _bfs(adj, start, target):
    dist, ways, prev = [-1]*len(adj), [0]*len(adj), [None]*len(adj)
    dist[start], ways[start] = 0,1
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v,e in adj[u]:
            if dist[v] < 0:
                dist[v],prev[v] = dist[u]+1,(u,e)
                queue.append(v)
            if dist[v] == dist[u]+1:
                ways[v] = min(2, ways[v]+ways[u])
    assert ways[target] == 1, "disconnected destination or tied minimum-hop routes"
    nodes,edges = [target],[]
    while nodes[-1] != start:
        u,e = prev[nodes[-1]]
        nodes.append(u)
        edges.append(e)
    return nodes[::-1],edges[::-1]


def _decode(input_grid):
    """Recover every role, complete station footprint, edge and route from pixels."""
    a = _grid(input_grid)
    colors,counts = np.unique(a, return_counts=True)
    assert len(colors) == 6, "six distinct local color roles required"
    bg = int(colors[np.argmax(counts)])
    assert np.count_nonzero(a == bg) > a.size/2, "background must be a strict majority"
    by = {int(c): _components(a == c) for c in colors if c != bg}
    candidates = []
    for c,cc in by.items():
        sizes = [_square(x) for x in cc]
        if len(cc) >= 3 and len(set(sizes)) == 1 and 2 <= sizes[0] <= 4:
            candidates.append((c,sizes[0]))
    assert len(candidates) == 1, "ordinary stations must be unambiguous"
    ordinary,n = candidates[0]
    target_colors = [c for c,cc in by.items() if c != ordinary and len(cc) == 1 and _square(cc[0]) == n]
    assert len(target_colors) == 1
    target = target_colors[0]
    rest = sorted(set(by)-{ordinary,target})
    sources = []
    for x,y in combinations(rest,2):
        cells = np.argwhere((a == x)|(a == y))
        if _square(cells) == n:
            sources.append(((x,y),cells))
    assert len(sources) == 1, "one complete binary source square required"
    tile_colors,source_cells = sources[0]
    connector = next(c for c in rest if c not in tile_colors)
    stations = by[ordinary]+by[target]+[source_cells]
    stations.sort(key=lambda x: tuple(x.min(0)))
    station_map = np.full(a.shape,-1,dtype=np.int16)
    origins = []
    for i,cells in enumerate(stations):
        assert _square(cells) == n
        station_map[tuple(cells.T)] = i
        origins.append(tuple(map(int,cells.min(0))))
    source = next(i for i,cells in enumerate(stations) if a[tuple(cells[0])] in tile_colors)
    destination = next(i for i,cells in enumerate(stations) if a[tuple(cells[0])] == target)
    sr,sc = origins[source]
    tile = a[sr:sr+n,sc:sc+n].copy()
    assert set(map(int,np.unique(tile))) == set(tile_colors)
    edges,edge_cells,adj = [],_components(a == connector),[[] for _ in stations]
    for e,cells in enumerate(edge_cells):
        touching = set()
        for r,c in cells.tolist():
            for dr,dc in N4:
                rr,ccc = r+dr,c+dc
                if 0 <= rr < len(a) and 0 <= ccc < a.shape[1] and station_map[rr,ccc] >= 0:
                    touching.add(int(station_map[rr,ccc]))
        assert len(touching) == 2, "an edge must join exactly two visible stations"
        u,v = sorted(touching)
        edges.append((u,v))
        adj[u].append((v,e))
        adj[v].append((u,e))
    assert len(set(edges)) == len(edges), "parallel connector components are excluded"
    assert all(1 <= len(neighbors) <= 4 for neighbors in adj)
    reached,stack = {source},[source]
    while stack:
        for v,_ in adj[stack.pop()]:
            if v not in reached:
                reached.add(v)
                stack.append(v)
    assert len(reached) == len(stations)
    path = _bfs(adj,source,destination)
    return dict(grid=a,background=bg,ordinary=ordinary,target=target,connector=connector,
                tile_colors=tile_colors,tile=tile,n=n,stations=stations,origins=origins,
                edges=edges,edge_cells=edge_cells,adj=adj,source=source,destination=destination,path=path)


def _render(d, nodes=None, behavior="normal"):
    nodes = d['path'][0] if nodes is None else nodes
    out = d['grid'].copy()
    tile = d['tile']
    swapped = d['tile_colors'][0]+d['tile_colors'][1]-tile
    for hop,node in enumerate(nodes[1:],1):
        r,c = d['origins'][node]
        stamp = swapped if hop%2 else tile
        if behavior == "all_exchanged":
            stamp = swapped
        elif behavior == "no_exchange":
            stamp = tile
        elif behavior == "rotate_tile":
            # A natural competing rule: point the tile's top toward the incoming port.
            prev = nodes[hop-1]
            e = next(e for v,e in d['adj'][node] if v == prev)
            cells = d['edge_cells'][e]
            ports = [(rr,cc) for rr,cc in cells.tolist() if
                     any(abs(rr-x)+abs(cc-y) == 1 for x,y in d['stations'][node].tolist())]
            pr,pc = np.mean(ports,axis=0)
            dr,dc = pr-(r+(d['n']-1)/2),pc-(c+(d['n']-1)/2)
            k = (0 if dr < 0 else 2) if abs(dr) > abs(dc) else (1 if dc > 0 else 3)
            stamp = np.rot90(stamp,-k)
        out[r:r+d['n'],c:c+d['n']] = stamp
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution; one role exchange per station hop, no tile rotation."""
    return _render(_decode(input_grid))


def _pixel_route(d):
    """A competing physical-ink-cost route; used only to check the curriculum."""
    source,target = d['source'],d['destination']
    best,prev = {source:0},{}
    queue = [(0,source)]
    while queue:
        cost,u = heapq.heappop(queue)
        if cost != best[u]:
            continue
        if u == target:
            break
        for v,e in d['adj'][u]:
            candidate = cost+len(d['edge_cells'][e])
            if candidate < best.get(v,float('inf')):
                best[v],prev[v] = candidate,u
                heapq.heappush(queue,(candidate,v))
    nodes = [target]
    while nodes[-1] != source:
        nodes.append(prev[nodes[-1]])
    return nodes[::-1],best[target]


def _distance_witness(d):
    alt,cost = _pixel_route(d)
    chosen = sum(len(d['edge_cells'][e]) for e in d['path'][1])
    return chosen > cost and not np.array_equal(_render(d,alt),_render(d))


def _phase_witness(d):
    alt,_ = _pixel_route(d)
    return _distance_witness(d) and (len(alt)-len(d['path'][0]))%2 == 1


def _axes(rng,count,n):
    pitch = n+2
    slack = min(30-(n+(count-1)*pitch),int(rng.integers(0,7)))
    gaps = np.full(count-1,pitch)+rng.multinomial(slack,[1/(count-1)]*(count-1))
    return [0]+list(map(int,np.cumsum(gaps)))


def _draw(rng,n,nr,nc,active,links,visible,source,target,tile,roles):
    """Construct lattice corridors; suppressed degree-two sites are connector bends."""
    rows,cols = _axes(rng,nr,n),_axes(rng,nc,n)
    h,w = rows[-1]+n,cols[-1]+n
    a = np.zeros((h,w),dtype=np.int8)
    off = n//2
    for u,v in links:
        ar,ac = rows[u[0]]+off,cols[u[1]]+off
        br,bc = rows[v[0]]+off,cols[v[1]]+off
        assert ar == br or ac == bc
        if ar == br:
            a[ar,min(ac,bc):max(ac,bc)+1] = 1
        else:
            a[min(ar,br):max(ar,br)+1,ac] = 1
    for u in visible:
        r,c = rows[u[0]],cols[u[1]]
        a[r:r+n,c:c+n] = 3 if u == target else 2
    r,c = rows[source[0]],cols[source[1]]
    a[r:r+n,c:c+n] = np.where(tile,4,5)
    # The tile stays in screen orientation after this whole-input transformation;
    # the transformed source, rather than a latent pre-transform tile, is copied.
    a = np.rot90(a,int(rng.integers(4)))
    if rng.random() < .5:
        a = a[:,::-1]
    h,w = a.shape
    pad_h,pad_w = int(rng.integers(min(4,31-h))),int(rng.integers(min(4,31-w)))
    top,left = int(rng.integers(pad_h+1)),int(rng.integers(pad_w+1))
    padded = np.zeros((h+pad_h,w+pad_w),dtype=np.int8)
    padded[top:top+h,left:left+w] = a
    return np.asarray(roles,dtype=np.int8)[padded]


def _tile_bits(rng,n):
    for _ in range(32):
        bits = rng.random((n,n)) < rng.uniform(.2,.8)
        if bits.any() and not bits.all() and not np.array_equal(bits,np.rot90(bits,2)):
            return bits
    bits = np.zeros((n,n),bool)
    bits[0,0] = True
    return bits


def _witness_network(rng):
    # Dense direct rail versus sparse, longer detour. A branch leaves a selected relay.
    nc = 5
    nr = int(rng.integers(3,5))
    active = {(0,c) for c in range(nc)}|{(nr-1,c) for c in range(nc)}
    active |= {(r,c) for r in range(nr) for c in (0,nc-1)}
    links = [((0,c),(0,c+1)) for c in range(nc-1)]
    links += [((nr-1,c),(nr-1,c+1)) for c in range(nc-1)]
    links += [((r,c),(r+1,c)) for c in (0,nc-1) for r in range(nr-1)]
    sparse = sorted(map(int,rng.choice([1,2,3],size=2,replace=False)))
    branch_col = int(rng.choice(sparse))
    branch = (nr-2,branch_col)
    active.add(branch)
    links.append(((nr-1,branch_col),branch))
    visible = {(0,c) for c in range(nc)}|{(nr-1,c) for c in sparse}|{branch}
    source,target = (0,0),(0,nc-1)
    if rng.random() < .5:
        source,target = target,source
    return nr,nc,active,links,visible,source,target


def _random_network(rng,n,big):
    nr,nc = map(int,rng.integers(3,6,size=2))
    vertices = [(r,c) for r in range(nr) for c in range(nc)]
    candidates = [(u,v) for u,v in combinations(vertices,2) if abs(u[0]-v[0])+abs(u[1]-v[1]) == 1]
    wanted = int(rng.integers(10 if big else 7,len(vertices)+1)) if big and len(vertices)>=10 else int(rng.integers(6,len(vertices)+1))
    active = {vertices[int(rng.integers(len(vertices)))]}
    links = []
    while len(active) < wanted:
        choices = [(u,v) for u,v in candidates if (u in active) != (v in active)]
        u,v = choices[int(rng.integers(len(choices)))]
        links.append((u,v))
        active.update((u,v))
    extra = [(u,v) for u,v in candidates if u in active and v in active and (u,v) not in links]
    rng.shuffle(extra)
    links += extra[:int(rng.integers(0,min(len(extra),5 if big else 3)+1))]
    degree = {u:sum(u in e for e in links) for u in active}
    visible = {u for u in active if degree[u] != 2 or rng.random() < .55}
    hidden = sorted(active-visible)
    rng.shuffle(hidden)
    while len(visible) < 5:
        visible.add(hidden.pop())
    # Select endpoints after geometric reconstruction, not from this latent list.
    return nr,nc,active,links,visible


def _long_network(rng):
    """Bounded fallback for a five-hop winning route versus a seven-hop rival."""
    ring = ([(0,c) for c in range(5)]+[(r,4) for r in range(1,5)]
            +[(4,c) for c in range(3,-1,-1)]+[(r,0) for r in range(3,0,-1)])
    links = [(u,ring[(i+1)%len(ring)]) for i,u in enumerate(ring)]
    first = list(map(int,rng.choice(range(1,8),4,replace=False)))
    second = list(map(int,rng.choice(range(9,16),6,replace=False)))
    visible = {ring[i] for i in [0,8]+first+second}
    source,target = ring[0],ring[8]
    if rng.random() < .5:
        source,target = target,source
    return 5,5,set(ring),links,visible,source,target


def _make_pair(rng,n,witness=False,big=False,hop_bounds=(2,30)):
    for attempt in range(65):
        guaranteed = witness or attempt == 64
        roles = tuple(map(int,rng.choice(10,6,replace=False)))
        bits = _tile_bits(rng,n)
        if guaranteed:
            network = _long_network(rng) if hop_bounds[0] >= 5 else _witness_network(rng)
            nr,nc,active,links,visible,source,target = network
            inp = _draw(rng,n,nr,nc,active,links,visible,source,target,bits,roles)
            d = _decode(inp)
        else:
            nr,nc,active,links,visible = _random_network(rng,n,big)
            # Temporary endpoints allow the same pixel reader to certify the graph.
            source,target = sorted(visible)[:2]
            inp = _draw(rng,n,nr,nc,active,links,visible,source,target,bits,roles)
            try:
                d = _decode(inp)
            except AssertionError:
                continue
            choices = []
            for s,t in combinations(range(len(d['stations'])),2):
                try:
                    path = _bfs(d['adj'],s,t)
                except AssertionError:
                    continue
                if hop_bounds[0] <= len(path[1]) <= hop_bounds[1]:
                    choices.append((s,t,path))
            if not choices:
                continue
            s,t,_ = choices[int(rng.integers(len(choices)))]
            if rng.random() < .5:
                s,t = t,s
            # Reset every station, then paint the selected source and destination.
            inp = inp.copy()
            for cells in d['stations']:
                inp[tuple(cells.T)] = roles[2]
            rr,cc = d['origins'][s]
            inp[rr:rr+n,cc:cc+n] = np.where(bits,roles[4],roles[5])
            inp[tuple(d['stations'][t].T)] = roles[3]
            d = _decode(inp)
        assert not guaranteed or _distance_witness(d)
        assert hop_bounds[0] <= len(d['path'][1]) <= hop_bounds[1]
        if witness:
            assert len(d['path'][0]) == 4
            assert any(len(d['adj'][v]) >= 3 for v in d['path'][0][1:-1])
            if np.array_equal(_render(d),_render(d,behavior='rotate_tile')):
                continue
        return {'input':inp,'output':_render(d)}
    raise AssertionError('bounded relay construction exhausted')


def _count(value,name,lo,hi):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ne = _count(num_train,'num_train',2,10),_count(num_test,'num_test',1,4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    # Every curriculum includes familiar tile copying and exchange at both phases.
    small,large = int(rng.choice([2,3])),int(rng.choice([3,4]))
    if large <= small:
        large = 4
    witness_index = int(rng.integers(nt))
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',nt),('test',ne)):
        for i in range(count):
            if regime == 0:
                n = small if split == 'train' else large
            elif regime == 1:
                n = int(rng.choice([2,3,4]))
            else:
                n = (2,3)[i%2] if split == 'train' else int(rng.choice([2,3]))
            witness = split == 'train' and i == witness_index
            bounds = (2,30) if regime == 0 else ((2,4) if split == 'train' else (5,30))
            for _ in range(8):
                pair = _make_pair(rng,n,witness,big=split == 'test' and regime != 0,hop_bounds=bounds)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('input uniqueness retries exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen,analyses = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            d = _decode(inp)
            # Validate phase locally, independently of the vectorized tile painter.
            expected = inp.copy()
            for hop,node in enumerate(d['path'][0][1:],1):
                r,c = d['origins'][node]
                for y in range(d['n']):
                    for x in range(d['n']):
                        v = int(d['tile'][y,x])
                        expected[r+y,c+x] = (sum(d['tile_colors'])-v) if hop%2 else v
            assert np.array_equal(out,expected), 'wrong phase, tile, route, or off-route edit'
            assert np.array_equal(solve(inp.copy()),out)
            assert np.all(out[inp == d['connector']] == d['connector'])
            assert np.all(out[inp == d['background']] == d['background'])
            analyses[split].append(d)
    training,testing = analyses['train'],analyses['test']
    assert any(_distance_witness(d) for d in training), 'missing station-versus-ink witness'
    assert any(_phase_witness(d) for d in training), 'route choice must change destination phase'
    assert any(len(d['path'][0]) >= 4 for d in training), 'both phases must be demonstrated'
    for behavior in ('all_exchanged','no_exchange','rotate_tile'):
        assert any(not np.array_equal(_render(d),_render(d,behavior=behavior)) for d in training), behavior
    assert any(any(len(d['adj'][v]) >= 3 for v in d['path'][0][1:-1]) for d in training)
    assert any(len(d['path'][0]) < len(d['stations']) for d in training), 'off-route relays must be preserved'
    enlargement = max(d['n'] for d in training) < min(d['n'] for d in testing)
    extension = max(len(d['path'][1]) for d in training) <= 4 and min(len(d['path'][1]) for d in testing) >= 5
    assert enlargement or extension, 'missing the specified quantitative train/test shift'


def family_features(pair):
    d = _decode(pair['input'])
    degrees = [len(x) for x in d['adj']]
    return dict(shape=d['grid'].shape,tile_size=d['n'],stations=len(degrees),
                hops=len(d['path'][1]),loops=len(d['edges'])-len(degrees)+1,
                leaves=degrees.count(1),junctions=sum(x>=3 for x in degrees),
                bent_edges=sum(not (np.all(c[:,0] == c[0,0]) or np.all(c[:,1] == c[0,1])) for c in d['edge_cells']),
                station_vs_ink=_distance_witness(d),
                route_changes_destination_phase=_phase_witness(d),
                selected_unused_branches=sum(len(d['adj'][v])>2 for v in d['path'][0][1:-1]))


def self_test():
    """Bounded: 42 puzzles, legal count cases, mutation checks; no disk writes."""
    shapes,sizes,structures,counts,hops,loops,bent = set(),set(),set(),set(),set(),set(),0
    puzzle_keys = set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        puzzle_keys.add(tuple(_key(x['input']) for s in ('train','test') for x in p[s]))
        for split in ('train','test'):
            for pair in p[split]:
                d = _decode(pair['input'])
                f = family_features(pair)
                shapes.add(f['shape']); sizes.add(f['tile_size']); counts.add(f['stations'])
                hops.add(f['hops']); loops.add(f['loops']); bent += f['bent_edges']
                roles = (d['background'],d['connector'],d['ordinary'],d['target'],*d['tile_colors'])
                semantic = np.zeros(d['grid'].shape,dtype=np.int8)
                for i,c in enumerate(roles):
                    semantic[d['grid'] == c] = i
                structures.add(_key(semantic))
        if seed == 0:
            q = generate_puzzle(seed)
            assert all(np.array_equal(x[k],y[k]) for s in p for x,y in zip(p[s],q[s]) for k in x)
            # Color relabeling and whole-input geometry cannot change the rule.
            permutation = np.array([4,8,1,9,3,0,6,2,5,7],dtype=np.int8)
            for pair in p['train']:
                assert np.array_equal(solve(permutation[pair['input']]),permutation[pair['output']])
                assert np.array_equal(solve(np.rot90(pair['input'])),np.rot90(pair['output']))
                assert np.array_equal(solve(pair['input'][:,::-1]),pair['output'][:,::-1])
            d = _decode(p['train'][0]['input'])
            direct = d['grid'].copy()
            direct[tuple(d['stations'][d['destination']].T)] = d['ordinary']
            neighbor = d['adj'][d['source']][0][0]
            direct[tuple(d['stations'][neighbor].T)] = d['target']
            expected = direct.copy()
            r,c = d['origins'][neighbor]
            expected[r:r+d['n'],c:c+d['n']] = sum(d['tile_colors'])-d['tile']
            assert np.array_equal(solve(direct),expected)
            erased = d['grid'].copy()
            r,c = d['origins'][d['source']]
            erased[r+d['n']-1,c:c+d['n']] = d['background']
            try:
                solve(erased)
            except AssertionError:
                pass
            else:
                raise AssertionError('an incomplete source tile was silently completed')
    assert len(puzzle_keys) == 42 and sizes == {2,3,4}
    assert len(shapes) >= 40 and len(structures) >= 150 and len(counts) >= 10
    assert len(hops) >= 5 and max(loops) >= 3 and bent >= 42
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(97,nt,ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0,2,1)
    p['test'][0]['output'] = p['test'][0]['output'].copy()
    p['test'][0]['output'][0,0] = (int(p['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    # A four-station square has tied two-hop routes across its diagonal.
    adj = [[(1,0),(3,3)],[(0,0),(2,1)],[(1,1),(3,2)],[(2,2),(0,3)]]
    try:
        _bfs(adj,0,2)
    except AssertionError:
        pass
    else:
        raise AssertionError('tied routes accepted')
    return dict(seeds=42,pairs=210,shapes=len(shapes),tile_sizes=sorted(sizes),
                color_normalized_structures=len(structures),station_counts=sorted(counts),
                hop_counts=sorted(hops),cycle_ranks=sorted(loops),bent_edges=bent)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(),separators=(',',':')))
