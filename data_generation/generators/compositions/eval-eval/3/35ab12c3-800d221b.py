"""Voting shadows: enclosed ballots travel on translated sparse contours.

Only standard library and NumPy are used. Construction writes labels from its
geometric records; solve reconstructs roles and contours from input cells.
No puzzle files, seed tables, or parent modules are read at runtime.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0,44506779,805047651,797546875,868061291,675671781,841034242,85071826,513880950,530091762,208312644,723577677,159344003,980652088,764227040,249755833,876928823,840461345,62086798,67412048,25702684,162488232,379322513,602579292,510621760,496230688,280730178,642846102,28386431,25691467,947462253,931960686,518136611,970102908,803027462,885373013,411259978,534542107,177756231,932525254,973130045,367977593)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sign(x):
    return int(x > 0) - int(x < 0)


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    length = max(abs(dr), abs(dc))
    if not length or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return ()
    return tuple((a[0] + i * _sign(dr), a[1] + i * _sign(dc)) for i in range(length + 1))


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _intersect(a, b, c, d):
    if any(max(a[k], b[k]) < min(c[k], d[k]) or max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
        return False
    return _cross(a,b,c)*_cross(a,b,d) <= 0 and _cross(c,d,a)*_cross(c,d,b) <= 0


def _contour(points):
    """Unique noncrossing spanning cycle; continuous crossings also count."""
    points = tuple(points)
    n = len(points)
    _require(4 <= n <= 10 and len(set(points)) == n, "invalid landmark count")
    point_set = set(points)
    edges, lines = [], []
    for i, j in combinations(range(n), 2):
        line = _line(points[i], points[j])
        if line and not set(line[1:-1]) & point_set:
            edges.append((i, j))
            lines.append(set(line))
    adjacency = [[] for _ in points]
    conflicts = [set() for _ in edges]
    for k, (i,j) in enumerate(edges):
        adjacency[i].append((j,k))
        adjacency[j].append((i,k))
        for l, (a,b) in enumerate(edges[:k]):
            if not {i,j} & {a,b} and _intersect(points[i],points[j],points[a],points[b]):
                conflicts[k].add(l)
                conflicts[l].add(k)
    solutions = {}
    visits = 0

    def search(order, chosen):
        nonlocal visits
        visits += 1
        _require(visits <= 20000, "contour search budget")
        if len(solutions) > 1:
            return
        if len(order) == n:
            for neighbor, edge in adjacency[order[-1]]:
                if neighbor == 0 and not conflicts[edge] & chosen:
                    solutions[frozenset(chosen | {edge})] = order
            return
        for neighbor, edge in adjacency[order[-1]]:
            if neighbor not in order and not conflicts[edge] & chosen:
                search(order + [neighbor], chosen | {edge})

    search([0], set())
    _require(len(solutions) == 1, "contour is not uniquely closed")
    selected, order = next(iter(solutions.items()))
    pixels = set().union(*(lines[e] for e in selected))
    rejected = set().union(*lines) - pixels
    polygon = [points[i] for i in order]
    turns = {_sign(_cross(polygon[i-1], polygon[i], polygon[(i+1) % n])) for i in range(n)}
    return dict(points=points, polygon=polygon, pixels=pixels, rejected=rejected,
                concave=(-1 in turns and 1 in turns), edges=[edges[e] for e in selected])


def _inside(p, polygon):
    r, c = p
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[0] > r) != (b[0] > r):
            crossing = a[1] + (r-a[0])*(b[1]-a[1])/(b[0]-a[0])
            if c < crossing:
                inside = not inside
    return inside


def _interior(polygon, boundary):
    return {(r,c) for r in range(min(p[0] for p in polygon), max(p[0] for p in polygon)+1)
            for c in range(min(p[1] for p in polygon), max(p[1] for p in polygon)+1)
            if (r,c) not in boundary and _inside((r,c), polygon)}


def _shift(points, offset):
    dr, dc = offset
    return {(r+dr,c+dc) for r,c in points}


def _near(a, b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1])) == 1


def _hub_geometry(hub):
    r, c = hub
    cells = {(r+i,c+j) for i in range(3) for j in range(3)}
    sides = [{(r-1,c+j) for j in range(3)}, {(r+3,c+j) for j in range(3)},
             {(r+j,c-1) for j in range(3)}, {(r+j,c+3) for j in range(3)}]
    return cells, sides, set().union(*sides)


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=True)


def _interpret(inp, bg, neutral, hub):
    hubcells, sides, ports = _hub_geometry(hub)
    h, w = inp.shape
    _require(all(0 <= r < h and 0 <= c < w for r,c in ports), "collector needs its collar")
    groups = {int(x): {tuple(map(int,p)) for p in np.argwhere(inp == x)} for x in np.unique(inp) if x != bg}
    markers = groups[neutral] - hubcells
    _require(2 <= len(markers) <= 4, "one marker per contour")
    _require(all(not any(_near(m,p) for p in hubcells) for m in markers), "marker touches collector")
    associations = []
    for marker in sorted(markers):
        nearby = [(color,p) for color,ps in groups.items() if color != neutral for p in ps if _near(marker,p)]
        _require(len(nearby) == 1, "marker must have one adjacent landmark")
        color, vertex = nearby[0]
        associations.append((color,vertex,marker))
    source_colors = {color for color,_,_ in associations}
    _require(len(source_colors) == len(markers), "source colors are unique")
    signals = sorted(set(groups) - source_colors - {neutral})
    _require(len(signals) == 2, "two ballot colors")
    regions, occupied = [], set()
    for color, vertex, marker in associations:
        info = _contour(sorted(groups[color]))
        source = info['pixels']
        offset = (marker[0]-vertex[0], marker[1]-vertex[1])
        shadow = _shift(source, offset)
        interior = _interior(info['polygon'], source)
        footprint = source | shadow | interior
        _require(all(0 <= r < h and 0 <= c < w for r,c in footprint), "clipped contour")
        _require(not source & (hubcells | ports) and not shadow & hubcells, "collector collision")
        _require(not footprint & occupied, "contours or territories overlap")
        occupied |= footprint
        contact = shadow & ports
        _require(contact, "every contour must vote")
        ballots = {p for s in signals for p in groups[s] if p in interior}
        _require(not ballots & shadow, "shadow crosses ballots")
        _require(all(not _near(p,marker) for p in ballots), "ballot resembles a landmark")
        counts = [sum(int(inp[p] == s) for p in ballots) for s in signals]
        _require(min(counts) > 0 and counts[0] != counts[1], "local vote must be mixed and strict")
        info.update(color=color, marker=marker, offset=offset, shadow=shadow, interior=interior,
                    ballots=ballots, counts=counts, label=signals[int(counts[1] > counts[0])],
                    contact=contact, weight=len(contact), sides=sum(bool(contact & s) for s in sides))
        regions.append(info)
    _require(set().union(*(x['ballots'] for x in regions)) == groups[signals[0]] | groups[signals[1]], "unowned ballot")
    votes = [sum(x['weight'] for x in regions if x['label'] == s) for s in signals]
    _require(min(votes) > 0 and votes[0] != votes[1], "collector vote must be mixed and strict")
    return dict(background=bg, neutral=neutral, hub=hub, signals=signals, regions=regions,
                votes=votes, winner=signals[int(votes[1] > votes[0])])


def _inspect(inp):
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(max(counts)) > inp.size // 2, "background must occupy over half the input")
    interpretations = []
    h, w = inp.shape
    for color in values:
        color = int(color)
        if color == bg:
            continue
        mask = inp == color
        if not 11 <= int(mask.sum()) <= 13:
            continue
        for r in range(h-2):
            for c in range(w-2):
                if np.all(mask[r:r+3,c:c+3]):
                    try:
                        interpretations.append(_interpret(inp,bg,color,(r,c)))
                    except ValueError:
                        pass
    _require(len(interpretations) == 1, "input needs one geometric interpretation")
    return interpretations[0]


def _paint(inp, info):
    out = inp.copy()
    bg = info['background']
    for region in info['regions']:
        for p in region['shadow']:
            if inp[p] == bg or p == region['marker']:
                out[p] = region['label']
        for p in region['pixels']:
            out[p] = region['color']
    r,c = info['hub']
    out[r+1,c+1] = info['winner']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _array(input_grid)
    return _paint(inp, _inspect(inp))


def _shape(rng, kind=None):
    kind = kind or str(rng.choice(['rectangle','square','clip','notch','L','diamond'], p=[.25,.13,.20,.16,.16,.10]))
    h,w = map(int,rng.integers(4,10,size=2))
    if kind == 'square':
        w = h
    if kind in ('rectangle','square'):
        points = [(0,0),(0,w),(h,w),(h,0)]
    elif kind == 'diamond':
        d = int(rng.integers(3,6))
        points = [(0,d),(d,2*d),(2*d,d),(d,0)]
    elif kind == 'clip':
        a,b,c,d = map(int,rng.integers(0,3,size=4))
        if not a+b+c+d:
            a = 1
        points = [(0,a),(0,w-b),(b,w),(h-c,w),(h,w-c),(h,d),(h-d,0),(a,0)]
        points = [p for i,p in enumerate(points) if p != points[i-1]]
    elif kind == 'notch':
        h,w = map(int,rng.integers(6,11,size=2))
        k = int(rng.integers(2,w-2))
        depth = int(rng.integers(1,min(3,h-3)))
        points = [(0,0),(0,w),(h,w),(h,k+1),(h-depth,k+1),(h-depth,k),(h,k),(h,0)]
    else:
        h,w = map(int,rng.integers(7,12,size=2))
        a,b = int(rng.integers(3,h-2)),int(rng.integers(3,w-2))
        points = [(0,0),(0,w),(a,w),(a,b),(h,b),(h,0)]
    code = int(rng.integers(8))
    transformed = []
    for r,c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r,c = c,-r
        transformed.append((r,c))
    mr,mc = min(p[0] for p in transformed),min(p[1] for p in transformed)
    points = [(r-mr,c-mc) for r,c in transformed]
    info = _contour(points)
    intended = {frozenset((i,(i+1)%len(points))) for i in range(len(points))}
    _require({frozenset(e) for e in info['edges']} == intended, "unintended boundary")
    info['interior'] = _interior(points, info['pixels'])
    return info


def _place_one(rng, hub, previous, offsets, weight_range, kind=None, minimum_ballots=3):
    hubcells, _, ports = _hub_geometry(hub)
    blocked = set().union(*(x['pixels'] | x['shadow'] | x['interior'] for x in previous)) if previous else set()
    previous_vertices = [p for x in previous for p in x['points']]
    previous_markers = [x['marker'] for x in previous]
    for _ in range(70):
        try:
            local = _shape(rng,kind)
        except ValueError:
            continue
        offset = offsets[int(rng.integers(len(offsets)))]
        local_shadow = _shift(local['pixels'], offset)
        local_footprint = local['pixels'] | local_shadow | local['interior']
        placements = sorted({(q[0]-p[0],q[1]-p[1]) for q in ports for p in local_shadow})
        rng.shuffle(placements)
        for delta in placements:
            source = _shift(local['pixels'],delta)
            shadow = _shift(local_shadow,delta)
            contact = shadow & ports
            if not weight_range[0] <= len(contact) <= weight_range[1] or source & (hubcells | ports) or shadow & hubcells:
                continue
            footprint = _shift(local_footprint,delta)
            if footprint & blocked or not all(0 <= r < 30 and 0 <= c < 30 for r,c in footprint):
                continue
            points = [(r+delta[0],c+delta[1]) for r,c in local['points']]
            if any(_near(m,p) for m in previous_markers for p in points):
                continue
            interior = _shift(local['interior'],delta)
            marker_options = []
            for vertex in points:
                marker = (vertex[0]+offset[0],vertex[1]+offset[1])
                if marker in source or any(_near(marker,p) for p in hubcells):
                    continue
                if sum(_near(marker,p) for p in points + previous_vertices) != 1:
                    continue
                available = sorted(p for p in interior-shadow if not _near(p,marker))
                if len(available) >= minimum_ballots:
                    marker_options.append((marker,available))
            if not marker_options:
                continue
            marker, available = marker_options[int(rng.integers(len(marker_options)))]
            return dict(points=points,pixels=source,shadow=shadow,interior=interior,
                        marker=marker,offset=offset,contact=contact,available=available,
                        concave=local['concave'],rejected=_shift(local['rejected'],delta))
    return None


def _metrics(inp, info):
    regions = info['regions']
    winner = info['winner']
    def margin(weights):
        return sum(weight if x['label'] == winner else -weight for x,weight in zip(regions,weights))
    raw = sum(int(np.count_nonzero(inp == s)) * (1 if s == winner else -1) for s in info['signals'])
    vertex_votes = [sum(len(_shift(x['points'],x['offset']) & x['contact']) for x in regions if x['label'] == s) for s in info['signals']]
    sparse_winner = info['neutral'] if vertex_votes[0] == vertex_votes[1] else info['signals'][int(vertex_votes[1] > vertex_votes[0])]
    return dict(contours=len(regions), weights=sorted(x['weight'] for x in regions),
                ports=sum(x['weight'] for x in regions), margin=abs(info['votes'][0]-info['votes'][1]),
                raw_sign=_sign(raw), panel_sign=_sign(margin([1]*len(regions))),
                perimeter_sign=_sign(margin([len(x['pixels']) for x in regions])),
                area_sign=_sign(margin([len(x['interior']) for x in regions])),
                completion_changes_center=sparse_winner != winner,
                concave=sum(x['concave'] for x in regions), chords=sum(len(x['rejected']) for x in regions),
                multi_side=sum(x['sides'] > 1 for x in regions),
                source_priority=sum(len((x['shadow'] & x['pixels']) - set(x['points'])) for x in regions),
                offsets=sorted(set(x['offset'] for x in regions)),
                local_margins=sorted(abs(x['counts'][0]-x['counts'][1]) for x in regions),
                ballot_counts=sorted(sum(x['counts']) for x in regions),
                winner_index=info['signals'].index(winner), shape=list(inp.shape))


def _build_pair(rng, roles, count, offsets, winner_index, witness=False, demo=False, novel=None):
    bg, neutral, *signals = roles
    signals = sorted(signals)
    winner, loser = signals[winner_index],signals[1-winner_index]
    for _ in range(100):
        hub = tuple(map(int,rng.integers(11,17,size=2)))
        regions = []
        for i in range(count):
            kind = 'square' if demo and i == 0 else 'notch' if demo and i == 1 else None
            if novel is not None and i == 0:
                allowed = (novel,)
            else:
                allowed = offsets
            wrange = (3,6) if witness and i == 0 else (1,1) if witness else (1,6)
            spec = _place_one(rng,hub,regions,allowed,wrange,kind,5 if witness and i else 3)
            if spec is None:
                break
            regions.append(spec)
        if len(regions) != count:
            continue
        assignments = [bits for bits in range(1,(1<<count)-1)
                       if sum((1 if bits>>i&1 else -1)*len(x['contact']) for i,x in enumerate(regions)) > 0]
        if witness:
            assignments = [1] if 1 in assignments else []
        if not assignments:
            continue
        bits = assignments[int(rng.integers(len(assignments)))]
        inp = np.full((30,30),bg,dtype=np.int8)
        r,c = hub
        inp[r:r+3,c:c+3] = neutral
        used_colors = list(map(int,rng.choice([x for x in range(10) if x not in roles],count,replace=False)))
        for i, (x,color) in enumerate(zip(regions,used_colors)):
            label = winner if bits>>i&1 else loser
            other = loser if label == winner else winner
            maximum = min(24,len(x['available']))
            n = 3 if witness and i == 0 else int(rng.integers(5,maximum+1)) if witness else int(rng.integers(3,maximum+1))
            minor = int(rng.integers(1,(n-1)//2+1))
            locations = [x['available'][int(j)] for j in rng.choice(len(x['available']),n,replace=False)]
            locations.sort()
            minority_locations = set(int(j) for j in rng.choice(n,minor,replace=False))
            if witness and i == 0:
                minority_locations = {0}
            x.update(color=color,label=label)
            for j,p in enumerate(locations):
                inp[p] = other if j in minority_locations else label
            for p in x['points']:
                inp[p] = color
            inp[x['marker']] = neutral
        # Construction labels use the sampled geometric records, not solve().
        out = inp.copy()
        for x in regions:
            for p in x['shadow']:
                out[p] = x['label']
            for p in x['pixels']:
                out[p] = x['color']
        out[r+1,c+1] = winner
        footprint = set().union(*(x['pixels'] | x['shadow'] | x['interior'] for x in regions)) | _hub_geometry(hub)[0]
        lo_r,hi_r = min(p[0] for p in footprint),max(p[0] for p in footprint)
        lo_c,hi_c = min(p[1] for p in footprint),max(p[1] for p in footprint)
        lo_r -= int(rng.integers(min(3,lo_r)+1))
        lo_c -= int(rng.integers(min(3,lo_c)+1))
        hi_r += int(rng.integers(min(3,29-hi_r)+1))
        hi_c += int(rng.integers(min(3,29-hi_c)+1))
        # Retain the complete collector collar even at a tight crop.
        lo_r,lo_c = min(lo_r,r-1),min(lo_c,c-1)
        hi_r,hi_c = max(hi_r,r+3),max(hi_c,c+3)
        inp,out = inp[lo_r:hi_r+1,lo_c:hi_c+1].copy(),out[lo_r:hi_r+1,lo_c:hi_c+1].copy()
        try:
            info = _inspect(inp)
            _require(np.array_equal(_paint(inp,info),out), "construction/reconstruction mismatch")
            m = _metrics(inp,info)
            if witness:
                _require(all(m[k] == -1 for k in ('raw_sign','panel_sign','perimeter_sign','area_sign')), "weak voting witness")
                _require(m['completion_changes_center'], "weak completion witness")
            if demo:
                _require(m['chords'] and m['concave'] and m['source_priority'], "weak contour witness")
            return {'input':inp,'output':out}
        except ValueError:
            continue
    raise RuntimeError("bounded geometric construction exhausted")


def _counts(rng, num_train, num_test):
    for value,lo,hi in ((num_train,2,10),(num_test,1,4)):
        if value is not None:
            _require(not isinstance(value,(bool,np.bool_)) and isinstance(value,(int,np.integer)) and lo <= int(value) <= hi, "illegal example count")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1,5))
        num_train = int(rng.integers(num_test+1,11))
    elif num_train is None:
        num_train = int(rng.integers(num_test+1,11))
    elif num_test is None:
        num_test = int(rng.integers(1,min(4,num_train-1)+1))
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train),int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed,(bool,np.bool_)) and isinstance(seed,(int,np.integer)) and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    nt,nq = _counts(rng,num_train,num_test)
    mode = int(rng.integers(2))  # density transfer or a held-out unit vector
    held = OFFSETS[int(rng.integers(8))]
    train_offsets = tuple(x for x in OFFSETS if mode == 0 or x != held)
    stable = bool(rng.integers(2))
    palette = list(map(int,rng.choice(10,4,replace=False)))
    winners = list(map(int,rng.integers(2,size=nt)))
    winners[1] = 1-winners[0]
    puzzle,seen = {'train':[],'test':[]},set()
    for split,total in (('train',nt),('test',nq)):
        for i in range(total):
            witness,demo = split == 'train' and i == 0, split == 'train' and i == 1
            count = 3 if witness else 2 if demo else 4 if mode == 0 and split == 'test' else int(rng.integers(2,4 if mode == 0 and split == 'train' else 5))
            roles = palette if stable else list(map(int,rng.choice(10,4,replace=False)))
            winner_index = winners[i] if split == 'train' else int(rng.integers(2))
            for _ in range(12):
                pair = _build_pair(rng,roles,count,train_offsets if split == 'train' else OFFSETS,
                                   winner_index,witness,demo,held if mode == 1 and split == 'test' else None)
                key = pair['input'].shape,pair['input'].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input budget exhausted")
    # Demonstration order carries no meaning.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "split schema")
    nt,nq = len(puzzle['train']),len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "puzzle counts")
    seen,train_metrics = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'}, "pair schema")
            inp,out = _array(pair['input']),_array(pair['output'])
            _require(inp.shape == out.shape, "output size changed")
            key = inp.shape,inp.tobytes()
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _inspect(inp)
            _require(np.array_equal(out,_paint(inp,info)), "incorrect output")
            exceptions = {x['marker'] for x in info['regions']}
            r,c = info['hub']
            exceptions.add((r+1,c+1))
            _require(all(inp[p] == out[p] for p in zip(*np.nonzero(inp != info['background'])) if p not in exceptions), "original evidence changed")
            for x in info['regions']:
                _require(len(x['pixels']-set(x['points'])) >= 2, "no missing boundary")
                _require(len(x['shadow']-x['pixels']-{x['marker']}) >= 2, "no visible translated shadow")
            if split == 'train':
                train_metrics.append(_metrics(inp,info))
    _require(any(all(m[k] == -1 for k in ('raw_sign','panel_sign','perimeter_sign','area_sign')) and m['completion_changes_center'] for m in train_metrics), "missing majority/port/completion witness")
    _require(any(m['chords'] and m['concave'] and m['source_priority'] for m in train_metrics), "missing contour/priority witness")
    _require({m['winner_index'] for m in train_metrics} == {0,1}, "both signals must win in training")


def self_test():
    """Stream 42 seeds; retain only bounded sets of feature signatures."""
    shapes,offsets,weights,layouts,palettes,counts = set(),set(),set(),set(),set(),set()
    concave = multi_side = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp,out = pair['input'],pair['output']
                info = _inspect(inp)
                m = _metrics(inp,info)
                assert np.array_equal(solve(inp),out)
                shapes.add(inp.shape)
                offsets.update(m['offsets'])
                weights.add(tuple(m['weights']))
                counts.add(m['contours'])
                palettes.add((info['background'],info['neutral'],*info['signals']))
                layouts.add((inp.shape,info['hub'],tuple(sorted(tuple(x['points']) for x in info['regions']))))
                concave += m['concave']
                multi_side += m['multi_side']
    assert len(shapes) >= 30 and len(layouts) >= 150 and len(palettes) >= 30
    assert offsets == set(OFFSETS) and counts == {2,3,4} and len(weights) >= 10
    assert concave >= 42 and multi_side > 0
    a,b = generate_puzzle(123,2,1),generate_puzzle(123,2,1)
    assert all(np.array_equal(x[k],y[k]) for s in a for x,y in zip(a[s],b[s]) for k in x)
    # Changing only enclosed ballot arrangement with counts fixed changes no
    # completed contour, shadow label, port weight, or collector decision.
    inp = a['train'][0]['input'].copy()
    info = _inspect(inp)
    expected = solve(inp)
    x = info['regions'][0]
    p = next(p for p in sorted(x['ballots']) if inp[p] == info['signals'][0])
    q = next(p for p in sorted(x['ballots']) if inp[p] == info['signals'][1])
    inp[p],inp[q] = inp[q],inp[p]
    expected[p],expected[q] = inp[p],inp[q]
    assert np.array_equal(solve(inp),expected)
    # An independently altered center must be rejected by puzzle validation.
    changed = {s:[{k:v.copy() for k,v in pair.items()} for pair in a[s]] for s in a}
    pair = changed['test'][0]
    info = _inspect(pair['input'])
    r,c = info['hub']
    pair['output'][r+1,c+1] = info['neutral']
    try:
        validate_generated_puzzle(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong collector accepted")
    return {'seeds':len(HOST_SEEDS),'pairs':210,'shapes':len(shapes),'layouts':len(layouts),
            'palettes':len(palettes),'weight_profiles':len(weights),'offsets':len(offsets),
            'concave_contours':concave,'multi_side_contours':multi_side}
