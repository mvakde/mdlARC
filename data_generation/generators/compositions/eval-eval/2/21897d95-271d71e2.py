"""Paint couriers: counter-limited motion addresses original terrain components.

Only NumPy and the standard library are used. Nothing is written on import,
generation, validation, or self-test. Structural colors are fixed: spent=0,
remaining=1, machinery=2. Terrain and literal paint use colors 3..9.
"""
from collections import deque
import hashlib
import heapq

import numpy as np


DIRS = ((-1, 0), (0, -1), (1, 0), (0, 1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError('Require integer counts 2..10 / 1..4 with num_train > num_test')
    return int(a), int(b)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), 'invalid grid dimensions')
    _need(np.issubdtype(g.dtype, np.integer), 'grid must contain integers')
    _need(bool(np.all((g >= 0) & (g <= 9))), 'invalid grid color')
    return g.astype(np.int8, copy=False)


def _components(g, mask=None):
    """Orthogonal components, with original IDs independent of color identity."""
    ids = np.full(g.shape, -1, dtype=np.int16)
    cells, colors = [], []
    h, w = g.shape
    for r, c in np.ndindex(g.shape):
        if ids[r, c] >= 0 or (mask is not None and not mask[r, c]):
            continue
        index, color = len(cells), int(g[r, c])
        todo, component = [(r, c)], []
        ids[r, c] = index
        while todo:
            y, x = todo.pop()
            component.append((y, x))
            for dy, dx in DIRS:
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w and ids[yy, xx] < 0
                        and g[yy, xx] == color and (mask is None or mask[yy, xx])):
                    ids[yy, xx] = index
                    todo.append((yy, xx))
        cells.append(component)
        colors.append(color)
    return ids, cells, colors


def _shift(points, direction, distance):
    dy, dx = direction
    return [(r + dy * distance, c + dx * distance) for r, c in points]


def _border(t, b, l, r):
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)
            if y in (t, b) or x in (l, r)}


def _read(input_grid):
    """Decode a complete scene from pixels, without generation metadata."""
    g = _grid(input_grid)
    h, w = g.shape
    _, machinery, _ = _components(g, g == 2)
    frame_components = []
    for component in machinery:
        ys, xs = zip(*component)
        t, b, l, r = min(ys), max(ys), min(xs), max(xs)
        if b-t >= 2 and r-l >= 2 and set(component) == _border(t, b, l, r):
            interior = g[t+1:b, l+1:r]
            if np.all((interior == 0) | (interior == 1)):
                frame_components.append(component)
    _need(bool(frame_components), 'no framed courier')
    index_grid = np.arange(g.size).reshape(g.shape)
    vehicles = []
    for component in frame_components:
        interpretations = []
        for k in range(4):
            view = np.rot90(g, k)
            mapping = np.rot90(index_grid, k)
            inverse = {int(v): (int(y), int(x)) for (y, x), v in np.ndenumerate(mapping)}
            coords = [inverse[y*w+x] for y, x in component]
            ys, xs = zip(*coords)
            t, b, l, r = min(ys), max(ys), min(xs), max(xs)
            if t < 4 or b+1 >= view.shape[0] or l < 1 or r+1 >= view.shape[1]:
                continue

            def original(points):
                return [divmod(int(mapping[p]), w) for p in points]

            for x in range(l+1, r):
                center = (t-2, x)
                arrow = [center, (t-2, x-1), (t-2, x+1), (t-3, x)]
                if any(view[p] != 2 for p in arrow[1:]) or int(view[center]) not in (2,3,4,5,6,7,8,9):
                    continue
                host = int(view[b+1, l-1])
                if host < 3:
                    continue
                # The whole initial device lies in a recoverable homogeneous
                # source patch, including a visible one-cell outer moat.
                body = [(y, xx) for y in range(t, b+1) for xx in range(l, r+1)]
                source_patch = [(y, xx) for y in range(t-4, b+2) for xx in range(l-1, r+2)]
                occupied = set(body + arrow)
                if any(int(view[p]) != host for p in source_patch if p not in occupied):
                    continue
                counter = [(y, xx) for y in range(t+1, b) for xx in range(l+1, r)]
                values = [int(view[p]) for p in counter]
                spent = values.count(0)
                if values != [0]*spent + [1]*(len(values)-spent):
                    continue
                probe = (t-4, x)
                gates = []
                # A gate is an isolated three-cell bar in a 3x5 land patch.
                for distance in range(1, probe[0]+1):
                    y = probe[0]-distance
                    if y < 1 or x < 2 or x+2 >= view.shape[1]:
                        continue
                    gate = [(y, x-1), (y, x), (y, x+1)]
                    if any(view[p] != 2 for p in gate):
                        continue
                    patch = [(yy, xx) for yy in range(y-1, y+2) for xx in range(x-2, x+3)]
                    remaining = {int(view[p]) for p in patch if p not in gate}
                    if len(remaining) == 1 and next(iter(remaining)) >= 3:
                        gates.append((distance, gate, next(iter(remaining))))
                if len(gates) != 1:
                    continue
                distance, gate, gate_host = gates[0]
                p, ahead = original([(t, l), (t-1, l)])
                direction = (ahead[0]-p[0], ahead[1]-p[1])
                interpretations.append(dict(body=original(body), arrow=original(arrow),
                    counter=original(counter), source_patch=original(source_patch), host=host,
                    explicit=int(view[center]) != 2, junction=int(view[center]),
                    payload=host if view[center] == 2 else int(view[center]),
                    probe=original([probe])[0], gate=original(gate), gate_host=gate_host,
                    direction=direction, distance=distance, spent=spent,
                    capacity=len(counter), remaining=len(counter)-spent,
                    depth=b-t-1, width=r-l-1))
        _need(len(interpretations) == 1, 'courier direction, attachment, or gate is ambiguous')
        vehicles.append(interpretations[0])
    clean = g.copy()
    covered = set()
    swept = set()
    footprints = []
    for v in vehicles:
        writes = set(v['body'] + v['arrow'] + v['gate'])
        _need(not (writes & covered), 'shared devices or gates')
        covered.update(writes)
        for p in v['body'] + v['arrow']:
            clean[p] = v['host']
        for p in v['gate']:
            clean[p] = v['gate_host']
        footprint = {p for d in range(v['distance']+1)
                     for p in _shift(v['body']+v['arrow']+[v['probe']], v['direction'], d)}
        _need(all(0 <= y < h and 0 <= x < w for y, x in footprint), 'travel leaves canvas')
        _need(not footprint & swept, 'intersecting courier paths')
        swept.update(footprint)
        footprints.append(footprint)
        v['move'] = min(v['remaining'], v['distance'])
    for i,footprint in enumerate(footprints):
        _need(all(not footprint.intersection(v['gate']) for j,v in enumerate(vehicles) if i != j),
              'a path crosses another courier gate')
    _need(all(tuple(p) in covered for p in np.argwhere(g < 3)), 'unexplained machinery or counter')
    _need(bool(np.all(clean >= 3)), 'terrain restoration failed')
    ids, cells, colors = _components(clean)
    assignments = {}
    for v in vehicles:
        _need(all(clean[p] == v['host'] for p in v['source_patch']), 'source restoration disagrees')
        v['source'] = int(ids[v['body'][0]])
        v['endpoint'] = _shift([v['probe']], v['direction'], v['move'])[0]
        v['target'] = int(ids[v['endpoint']])
        if v['target'] in assignments:
            _need(assignments[v['target']] == v['payload'], 'conflicting paint assignments')
        assignments[v['target']] = v['payload']
    return dict(grid=g, clean=clean, ids=ids, cells=cells, colors=colors,
                vehicles=vehicles, assignments=assignments)


def _draw(clean, ids, vehicles, *, paint=True, policy='normal'):
    """Reconstruct terrain first, then overlay fixed gates and moved devices."""
    out = clean.copy()
    assignments = {}
    for v in vehicles:
        move = v['distance'] if policy == 'destination' else v['move']
        endpoint = _shift([v['probe']], v['direction'], move)[0]
        assignments[int(ids[endpoint])] = v['payload']
    if paint:
        for target, color in assignments.items():
            out[ids == target] = color
    for v in vehicles:
        for p in v['gate']:
            out[p] = 2
    for v in vehicles:
        move = v['distance'] if policy == 'destination' else v['move']
        for p in _shift(v['body']+v['arrow'], v['direction'], move):
            out[p] = 2
        for i, p in enumerate(_shift(v['counter'], v['direction'], move)):
            out[p] = 0 if i < v['spent']+move else 1
        out[_shift(v['arrow'][:1], v['direction'], move)[0]] = v['junction']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The fixed family rule; no seed, examples, cache, or latent convention."""
    scene = _read(input_grid)
    return _draw(scene['clean'], scene['ids'], scene['vehicles'])


def _local_spec(rng, distance, stop):
    candidates = [(a,b) for a in range(1,6) for b in range(1,5)
                  if a*b >= (distance+1 if stop == 'destination' else distance if stop == 'exact' else 2)]
    a, b = candidates[int(rng.integers(len(candidates)))]
    capacity = a*b
    if stop == 'destination':
        fuel = int(rng.integers(distance+1, capacity+1))
    elif stop == 'fuel':
        fuel = int(rng.integers(2, min(capacity, distance-2)+1))
    elif stop == 'exact':
        fuel = distance
    else:
        fuel = 0
    height, width = distance+a+8, b+4
    top, left = distance+5, 1
    bottom, right = top+a+1, left+b+1
    x = 1+int(rng.integers(1,b+1))
    arrow = [(top-2,x), (top-2,x-1), (top-2,x+1), (top-3,x)]
    body = [(y,xx) for y in range(top,bottom+1) for xx in range(left,right+1)]
    counter = [(y,xx) for y in range(top+1,bottom) for xx in range(left+1,right)]
    return dict(shape=(height,width), body=body, counter=counter, arrow=arrow,
        source_patch=[(y,xx) for y in range(distance+1,height) for xx in range(width)],
        gate=[(1,x-1),(1,x),(1,x+1)],
        gate_patch=[(y,xx) for y in range(3) for xx in range(x-2,x+3)],
        probe=(distance+1,x), distance=distance, remaining=fuel, spent=capacity-fuel,
        capacity=capacity, move=min(fuel,distance), depth=a, width=b)


def _place(rng, shape, specs, directions):
    occupied = np.zeros(shape, dtype=bool)
    vehicles = []
    for spec, k in zip(specs, directions):
        h,w = spec['shape']
        table = np.rot90(np.arange(h*w).reshape(h,w), k)
        ph,pw = table.shape
        if ph > shape[0] or pw > shape[1]:
            return None
        blocked = occupied.copy()
        for dy,dx in DIRS:
            for y,x in np.argwhere(occupied):
                yy,xx = int(y)+dy,int(x)+dx
                if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
                    blocked[yy,xx] = True
        positions = [(y,x) for y in range(shape[0]-ph+1) for x in range(shape[1]-pw+1)
                     if not blocked[y:y+ph,x:x+pw].any()]
        if not positions:
            return None
        y0,x0 = positions[int(rng.integers(len(positions)))]
        lookup = {int(value): (y0+int(y),x0+int(x)) for (y,x),value in np.ndenumerate(table)}
        def transform(points):
            return [lookup[y*w+x] for y,x in points]
        v = {name:value for name,value in spec.items() if name not in ('shape','gate_patch')}
        for name in ('body','counter','arrow','source_patch','gate'):
            v[name] = transform(spec[name])
        v['gate_patch'] = transform(spec['gate_patch'])
        v['probe'] = transform([spec['probe']])[0]
        # DIRS follows the counterclockwise coordinate rotations used here.
        v['direction'] = DIRS[k]
        vehicles.append(v)
        occupied[y0:y0+ph,x0:x0+pw] = True
    return vehicles


def _terrain(rng, shape, vehicles, targets):
    """Join required sites, then grow connected regions by random weighted waves.

    These are terrain constraints only, not an output program. Routing paths are
    temporary in-memory lists. No execution traces or per-attempt logs are saved.
    """
    n = len(vehicles)
    labels = np.full(shape,-1,dtype=np.int16)
    required = [[] for _ in range(n+2)]
    for i,v in enumerate(vehicles):
        required[i].extend(v['source_patch'])
        endpoint = _shift([v['probe']],v['direction'],v['move'])[0]
        required[targets[i]].append(endpoint)
        gate_label = targets[i] if v['move'] == v['distance'] else n+1
        required[gate_label].extend(v['gate_patch'])
    for label,points in enumerate(required):
        for p in points:
            if labels[p] not in (-1,label):
                return None
            labels[p] = label
    # Connect separated required patches without crossing another region.
    for label,points in enumerate(required):
        if not points:
            continue
        root = points[0]
        for _ in range(len(points)+1):
            _, components, _ = _components(labels,labels == label)
            if len(components) <= 1:
                break
            main = next(c for c in components if root in c)
            todo = deque(main)
            previous = {p:None for p in main}
            other = {p for c in components if c is not main for p in c}
            found = None
            order = [DIRS[int(k)] for k in rng.permutation(4)]
            while todo and found is None:
                y,x = todo.popleft()
                for dy,dx in order:
                    p = y+dy,x+dx
                    if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]):
                        continue
                    if p in previous or labels[p] not in (-1,label):
                        continue
                    previous[p] = (y,x)
                    if p in other:
                        found = p
                        break
                    todo.append(p)
            if found is None:
                return None
            while found is not None:
                labels[found] = label
                found = previous[found]
    # Add unused labels and additional islands as new connected territories.
    used = set(map(int,np.unique(labels))) - {-1}
    desired = int(rng.integers(max(n+2,4),8))
    for label in range(desired):
        if label in used:
            continue
        free = np.argwhere(labels < 0)
        if not len(free):
            return None
        p = tuple(map(int,free[int(rng.integers(len(free)))]))
        labels[p] = label
    queue = []
    weights = rng.uniform(.4,2.4,size=(shape[0],shape[1],4))
    for (y,x),label in np.ndenumerate(labels):
        if label >= 0:
            heapq.heappush(queue,(0.,y,x,int(label)))
    while queue:
        distance,y,x,label = heapq.heappop(queue)
        for k,(dy,dx) in enumerate(DIRS):
            yy,xx = y+dy,x+dx
            if 0 <= yy < shape[0] and 0 <= xx < shape[1] and labels[yy,xx] < 0:
                labels[yy,xx] = label
                heapq.heappush(queue,(distance+float(weights[y,x,k]),yy,xx,label))
    return labels


def _features(scene):
    vs, colors, ids = scene['vehicles'], scene['colors'], scene['ids']
    final = [scene['assignments'].get(i,c) for i,c in enumerate(colors)]
    return dict(count=len(vs), shape=scene['grid'].shape,
        stops={('zero' if v['remaining'] == 0 else 'fuel' if v['remaining'] < v['distance']
                else 'exact' if v['remaining'] == v['distance'] else 'destination') for v in vs},
        directions={v['direction'] for v in vs},
        explicit=any(v['explicit'] and v['payload'] != v['host'] for v in vs),
        implicit=any(not v['explicit'] for v in vs),
        dependency=any(not a['explicit'] and not b['explicit'] and a['target'] == b['source']
            and a['payload'] != b['payload'] and b['target'] != b['source']
            for a in vs for b in vs if a is not b),
        locality=any(colors[i] == colors[j] and final[i] != final[j]
                     for i in range(len(colors)) for j in range(i)),
        nonrectangular=sum(len(c) != (max(y for y,x in c)-min(y for y,x in c)+1)*
                            (max(x for y,x in c)-min(x for y,x in c)+1) for c in scene['cells']),
        target_not_gate=any(v['target'] != int(ids[v['gate'][1]]) and v['move'] > 0 for v in vs),
        off_ray_paint=any(any((y-v['endpoint'][0])*v['direction'][1] !=
                                 (x-v['endpoint'][1])*v['direction'][0]
                                for y,x in scene['cells'][v['target']])
                         and colors[v['target']] != v['payload'] for v in vs),
        row_update=any(v['width'] > 1 and v['depth'] > 1 and v['move'] > 0
                       and v['spent']//v['width'] != (v['spent']+v['move'])//v['width']
                       and (v['spent']+v['move']) % v['width'] != 0 for v in vs))


def _candidate(rng, n, distances, directions, lesson, repeat_allowed):
    if lesson == 0:
        stops = ['destination','fuel']
        explicit = [False,False]
    elif lesson == 1:
        stops = ['fuel','destination']
        explicit = [True,False]
    else:
        stops = [str(rng.choice(('destination','fuel','exact','zero') if n == 1
                                else ('destination','fuel','exact'))) for _ in range(n)]
        explicit = [bool(rng.random() < .45) for _ in range(n)]
    specs = [_local_spec(rng,int(rng.integers(*distances)),stop) for stop in stops]
    # Random free placement of swept footprints; no spatial slot template.
    shape = int(rng.integers(24,31)),int(rng.integers(24,31))
    vehicles = _place(rng,shape,specs,directions)
    if vehicles is None:
        return None
    targets = [i+1 for i in range(n)]
    for i,v in enumerate(vehicles):
        if v['move'] == 0:
            targets[i] = i
    labels = _terrain(rng,shape,vehicles,targets)
    if labels is None:
        return None
    count = int(labels.max())+1
    palette = list(map(int,rng.permutation(np.arange(3,10))))[:count]
    if repeat_allowed and rng.random() < .65:
        # Repeated colors only for nonadjacent original regions, preserving IDs.
        adjacent = set()
        for a,b in ((labels[:-1],labels[1:]),(labels[:,:-1],labels[:,1:])):
            for x,y in zip(a[a != b],b[a != b]):
                adjacent.add((int(x),int(y)))
        choices = [(a,b) for a in range(count) for b in range(a)
                   if (a,b) not in adjacent and a not in range(n) and b not in range(n)]
        if choices:
            a,b = choices[int(rng.integers(len(choices)))]
            palette[a] = palette[b]
    clean = np.asarray(palette,dtype=np.int8)[labels]
    inp = clean.copy()
    for i,v in enumerate(vehicles):
        v['host'] = palette[i]
        v['explicit'] = explicit[i]
        choices = [c for c in range(3,10) if c not in (v['host'],palette[targets[i]])]
        v['junction'] = int(rng.choice(choices)) if explicit[i] else 2
        v['payload'] = v['junction'] if explicit[i] else v['host']
        for p in v['body']+v['arrow']+v['gate']:
            inp[p] = 2
        for j,p in enumerate(v['counter']):
            inp[p] = 0 if j < v['spent'] else 1
        inp[v['arrow'][0]] = v['junction']
    # Constructor uses its terrain IDs; the solver separately recovers terrain.
    out = _draw(clean,labels,vehicles)
    try:
        scene = _read(inp)
        if not np.array_equal(scene['clean'],clean) or not np.array_equal(solve(inp),out):
            return None
        features = _features(scene)
        if lesson == 0 and not (features['dependency'] and features['row_update']):
            return None
        if not (features['off_ray_paint'] and (features['target_not_gate'] if lesson in (0,1) else True)):
            return None
    except AssertionError:
        return None
    # Crop only unused outer terrain, preserving all recovery moats and gates.
    # Here the random canvas itself is retained; irregular regions can touch it.
    return dict(input=inp,output=out),features


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    a,b = _counts(num_train,num_test)
    rng = np.random.default_rng(int(seed))
    orientation_pool = list(map(int,rng.choice(4,size=2,replace=False)))
    shift = int(rng.integers(2))  # 0: longer journeys; 1: more couriers.
    puzzle,seen = dict(train=[],test=[]),set()
    train_features = []
    for split,count in (('train',a),('test',b)):
        for index in range(count):
            lesson = index if split == 'train' and index < 2 else -1
            n = 2 if lesson >= 0 else (3 if split == 'test' and shift else int(rng.integers(1,3)))
            distances = (8,12) if split == 'test' and shift == 0 else (4,8)
            for attempt in range(240):
                # Try mixed directions, then parallel travel when packing tight.
                ks = ([orientation_pool[(index+i)%2] for i in range(n)] if attempt < 60
                      else [orientation_pool[index%2] if lesson >= 0 else int(rng.choice(orientation_pool))]*n)
                repeat_allowed = split == 'train' or any(f['locality'] for f in train_features)
                candidate = _candidate(rng,n,distances,ks,lesson,repeat_allowed)
                if candidate is None:
                    continue
                pair,features = candidate
                key = (pair['input'].shape,pair['input'].tobytes())
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == 'train':
                    train_features.append(features)
                break
            else:
                raise RuntimeError('bounded scene construction exhausted')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, 'wrong puzzle schema')
    _need(all(isinstance(puzzle[s],list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen,features = set(),dict(train=[],test=[])
    for split in ('train','test'):
        for pair in puzzle[split]:
            _need(isinstance(pair,dict) and set(pair) == {'input','output'}, 'wrong pair schema')
            g,out = _grid(pair['input']),_grid(pair['output'])
            _need(g.shape == out.shape,'canvas dimensions changed')
            key = g.shape,g.tobytes()
            _need(key not in seen,'duplicate input')
            seen.add(key)
            scene = _read(g)
            predicted = _draw(scene['clean'],scene['ids'],scene['vehicles'])
            _need(np.array_equal(out,predicted),'incorrect motion, counter, or component painting')
            _need(not np.array_equal(g,out),'identity pair')
            f = _features(scene)
            features[split].append(f)
            # Off-ray paint separates connected fill from a ray or trail rule.
            _need(f['off_ray_paint'],'pair lacks a visible territory-transfer witness')
            _need(not np.array_equal(out,_draw(scene['clean'],scene['ids'],scene['vehicles'],paint=False)),
                  'omitting parent A does not affect this pair')
    train = features['train']
    _need({'destination','fuel'} <= set().union(*(f['stops'] for f in train)), 'missing stopping contrast')
    for feature in ('dependency','implicit','explicit','row_update','target_not_gate'):
        _need(any(f[feature] for f in train),'missing training witness: '+feature)
    active = set().union(*(f['directions'] for f in train))
    _need(len(active) >= 2,'missing orientation contrast')
    _need(all(f['directions'] <= active for f in features['test']), 'untaught test orientation')
    _need(not any(f['locality'] for f in features['test']) or any(f['locality'] for f in train),
          'test locality lacks a training witness')
    # A genuine held-out count or distance, not an arbitrary palette change.
    train_distance = max(v['distance'] for p in puzzle['train'] for v in _read(p['input'])['vehicles'])
    for p,f in zip(puzzle['test'],features['test']):
        ds = [v['distance'] for v in _read(p['input'])['vehicles']]
        _need(f['count'] > max(x['count'] for x in train) or min(ds) > train_distance,
              'missing structural train/test shift')


def self_test():
    """Bounded 42-seed census plus semantic counterexamples; writes no files."""
    shapes,structures,fingerprints,directions,stops,counts,capacities,distances = (set() for _ in range(8))
    totals = dict(pairs=0,dependency=0,locality=0,nonrectangular=0,row_update=0,
                  target_not_gate=0,explicit=0,implicit=0,fresh=0,zero=0)
    example = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train','test'):
            for pair in puzzle[split]:
                g,out = pair['input'],pair['output']
                digest.update(g.tobytes())
                shapes.add(g.shape)
                scene = _read(g)
                f = _features(scene)
                structures.add(hashlib.sha256(scene['ids'].tobytes()).digest())
                directions.update(f['directions'])
                stops.update(f['stops'])
                counts.add(f['count'])
                totals['pairs'] += 1
                for key in ('dependency','locality','nonrectangular','row_update','target_not_gate','explicit','implicit'):
                    totals[key] += int(bool(f[key]))
                for v in scene['vehicles']:
                    capacities.add(v['capacity'])
                    distances.add(v['distance'])
                    totals['fresh'] += int(v['remaining'] == v['capacity'])
                    totals['zero'] += int(v['remaining'] == 0)
                    # Tick the actual direction-relative sequence one cell at a time.
                    counter = [int(g[p]) for p in v['counter']]
                    y,x = v['probe']
                    for _ in range(v['move']):
                        pending = counter.index(1)
                        counter[pending] = 0
                        y,x = y+v['direction'][0],x+v['direction'][1]
                    _need((y,x) == v['endpoint'],'unit-step endpoint mismatch')
                    moved = _shift(v['counter'],v['direction'],v['move'])
                    _need(counter == [int(out[p]) for p in moved],'unit-step counter mismatch')
                if f['target_not_gate']:
                    wrong = _draw(scene['clean'],scene['ids'],scene['vehicles'],policy='destination')
                    _need(not np.array_equal(wrong,out),'omitting parent B has no effect')
                # A whole-scene quarter-turn must commute with the rule.
                if seed < 3:
                    for k in (1,2,3):
                        _need(np.array_equal(solve(np.rot90(g,k)),np.rot90(out,k)), 'rotation covariance failed')
                if example is None:
                    example = puzzle
        fingerprints.add(digest.digest())
    _need(len(fingerprints) == 42 and len(structures) >= 180,'collapsed structural variation')
    _need(len(shapes) >= 25 and len(capacities) >= 10 and len(distances) == 8,'collapsed dimensions or counters')
    _need(counts == {1,2,3} and directions == set(DIRS),'missing courier counts or directions')
    _need(stops == {'destination','fuel','exact','zero'},'missing stopping relationships')
    _need(all(totals[k] > 0 for k in totals),'missing family feature')
    # NumPy and list representations both work; input arrays are never mutated.
    sample = example['test'][0]
    original = sample['input'].copy()
    _need(np.array_equal(solve(original.tolist()),sample['output']),'list input failed')
    _need(np.array_equal(solve(original),sample['output']),'array input failed')
    _need(np.array_equal(original,sample['input']),'solver mutated input')
    broken = {s:[{k:g.copy() for k,g in p.items()} for p in example[s]] for s in example}
    broken['test'][0]['output'][0,0] = (int(broken['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted corrupted output')
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            continue
        raise AssertionError('accepted illegal counts')
    return dict(seeds=42,distinct_puzzles=len(fingerprints),terrain_structures=len(structures),
                grid_shapes=len(shapes),capacities=sorted(capacities),distances=sorted(distances),
                courier_counts=sorted(counts),features=totals)
