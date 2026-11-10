"""A state sequence of square stamps travels on a visible corridor spine.

Only Python/NumPy are used. Labels are rendered from constructed terrain and
state sequences; solve recovers those sequences and the route from input pixels.
"""
import numpy as np


class _RetryPair(Exception):
    """A sampled immediately blocked identity has no teaching signal."""


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "colors outside 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _corners(bb):
    r0, r1, c0, c1 = bb
    return {(r0, c0), (r0, c0+1), (r0+1, c0),
            (r1, c1), (r1, c1-1), (r1-1, c1)}


def _marker(a):
    found = []
    for color in map(int, np.unique(a)):
        cells = np.argwhere(a == color)
        if len(cells) != 6:
            continue
        r0, c0 = map(int, cells.min(axis=0))
        r1, c1 = map(int, cells.max(axis=0))
        if r1-r0 < 1 or c1-c0 < 1 or r1-r0+c1-c0 < 4:
            continue
        bb = r0, r1, c0, c1
        if set(map(tuple, cells)) == _corners(bb):
            found.append((color, bb))
    _require(len(found) == 1, "need one pair of canonical disconnected L corners")
    return found[0]


def _neighbors(p, shape):
    r, c = p
    for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
        if 0 <= x < shape[0] and 0 <= y < shape[1]:
            yield x, y


def _ordered_path(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if len(points) < 12:
        return None
    ends = []
    for p in points:
        degree = sum(q in points for q in _neighbors(p, mask.shape))
        if degree == 1:
            ends.append(p)
        elif degree != 2:
            return None
    if len(ends) != 2:
        return None
    path, previous, p = [], None, min(ends)
    while True:
        path.append(p)
        options = [q for q in _neighbors(p, mask.shape) if q in points and q != previous]
        if not options:
            break
        previous, p = p, options[0]
        if len(path) > len(points):
            return None
    return path if len(path) == len(points) else None


def _square_slice(p, k, shape):
    r, c = p
    d = k // 2
    if k < 3 or k % 2 == 0 or r-d < 0 or c-d < 0 or r+d >= shape[0] or c+d >= shape[1]:
        return None
    return slice(r-d, r+d+1), slice(c-d, c+d+1)


def _size_sequence(observed, count):
    """All supported explanations must agree, including constant degeneracy."""
    predictions = []
    delta = observed[1] - observed[0]
    if all(v == observed[0] + i*delta for i, v in enumerate(observed)):
        predicted = tuple(observed[0] + i*delta for i in range(count))
        if all(3 <= k <= 29 and k % 2 for k in predicted):
            predictions.append((predicted, "arithmetic", delta))
    for period in range(1, min(3, len(observed)-1)+1):
        if all(v == observed[i % period] for i, v in enumerate(observed)):
            predictions.append((tuple(observed[i % period] for i in range(count)), "periodic", period))
    _require(predictions, "sizes lack a demonstrated arithmetic or short cyclic rule")
    _require(len({v[0] for v in predictions}) == 1, "ambiguous size continuation")
    return predictions[0]


def _color_sequence(observed, count):
    for period in range(1, min(3, len(observed)-1)+1):
        if len(set(observed[:period])) == period and all(v == observed[i % period] for i, v in enumerate(observed)):
            return tuple(observed[i % period] for i in range(count)), period
    raise AssertionError("colors need a repeated cycle, not an unseen phase")


def _parse(a):
    annotation, bb = _marker(a)
    border = np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))
    vals, counts = np.unique(border[border != annotation], return_counts=True)
    _require(len(vals) and np.count_nonzero(counts == counts.max()) == 1, "ambiguous surrounding wall")
    wall = int(vals[counts.argmax()])
    clean = a.copy()
    clean[clean == annotation] = wall
    _require(np.all(clean[0] == wall) and np.all(clean[-1] == wall)
             and np.all(clean[:, 0] == wall) and np.all(clean[:, -1] == wall), "wall frame required")
    colors = sorted(set(map(int, np.unique(clean))) - {wall})
    candidates = []
    for road in colors:
        for pin in colors:
            if pin == road or not 4 <= np.count_nonzero(clean == pin) <= 9:
                continue
            spine = (clean == road) | (clean == pin)
            raw_path = _ordered_path(spine)
            if raw_path is None:
                continue
            for path in (raw_path, raw_path[::-1]):
                station_indices = [i for i, p in enumerate(path) if clean[p] == pin]
                if station_indices[0] != 0 or station_indices[-1] != len(path)-1:
                    continue
                if any(y-x < 4 for x, y in zip(station_indices, station_indices[1:])):
                    continue
                stations = [path[i] for i in station_indices]
                for floor in colors:
                    if floor in (road, pin):
                        continue
                    sizes, phases = [], []
                    blank = False
                    good = True
                    accounted = spine | (clean == wall) | (clean == floor)
                    history_mask = np.zeros(clean.shape,dtype=bool)
                    for p in stations:
                        r, c = p
                        if min(r, c) < 1 or r+1 >= len(a) or c+1 >= a.shape[1]:
                            good = False
                            break
                        diagonal = {int(clean[r+dr, c+dc]) for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1))}
                        if diagonal == {floor}:
                            blank = True
                            continue
                        if blank or len(diagonal) != 1 or next(iter(diagonal)) in (wall, road, pin, floor):
                            good = False
                            break
                        phase = next(iter(diagonal))
                        fits = []
                        for k in range(3, 30, 2):
                            ss = _square_slice(p, k, clean.shape)
                            if ss is None:
                                break
                            patch = clean[ss]
                            if np.all(spine[ss] | (patch == phase)):
                                fits.append(k)
                            else:
                                break
                        if not fits:
                            good = False
                            break
                        k = max(fits)
                        ss = _square_slice(p, k, clean.shape)
                        if np.any(history_mask[ss]):
                            good = False
                            break
                        history_mask[ss] = True
                        accounted[ss] |= clean[ss] == phase
                        sizes.append(k)
                        phases.append(phase)
                    if not good or not blank or len(sizes) < 3 or not np.all(accounted):
                        continue
                    try:
                        whole_sizes, mode, size_parameter = _size_sequence(sizes, len(stations))
                        whole_colors, period = _color_sequence(phases, len(stations))
                    except AssertionError:
                        continue
                    # Historical states must also be physically compatible with the trace.
                    valid_history = True
                    for i in range(1, len(sizes)):
                        for p in path[station_indices[i-1]:station_indices[i]+1]:
                            ss = _square_slice(p, sizes[i], clean.shape)
                            if ss is None or np.any(clean[ss] == wall):
                                valid_history = False
                                break
                        if not valid_history:
                            break
                    if not valid_history:
                        continue
                    candidates.append(dict(clean=clean, wall=wall, floor=floor, road=road,
                                           pin=pin, annotation=annotation, bounds=bb, spine=spine,
                                           path=path, stations=stations, indices=station_indices,
                                           visible=len(sizes), sizes=whole_sizes, phases=whole_colors,
                                           size_mode=mode, size_parameter=size_parameter,
                                           color_period=period))
    _require(candidates, "input has no recoverable station trail")
    signatures = {(v['wall'], v['floor'], v['road'], v['pin'], tuple(v['path']),
                   v['visible'], v['sizes'], v['phases']) for v in candidates}
    _require(len(signatures) == 1, "competing interpretations of route or states")
    return candidates[0]


def _put_stamp(scene, spine, p, k, color):
    ss = _square_slice(p, k, scene.shape)
    _require(ss is not None, "stamp extends outside canvas")
    patch = scene[ss]
    patch[~spine[ss]] = color


def _continue(info, sizes=None, phases=None, ignore_walls=False):
    sizes = info['sizes'] if sizes is None else sizes
    phases = info['phases'] if phases is None else phases
    scene = info['clean'].copy()
    start_i = info['visible']-1
    reached, halted, travel = [], None, 0
    for i in range(start_i+1, len(info['stations'])):
        k, color = sizes[i], phases[i]
        begin, end = info['indices'][i-1:i+1]
        if ignore_walls:
            _put_stamp(scene, info['spine'], info['path'][end], k, color)
            reached.append(i)
            continue
        start_ss = _square_slice(info['path'][begin], k, scene.shape)
        _require(start_ss is not None and not np.any(info['clean'][start_ss] == info['wall']),
                 "next square must fit when its state changes at a station")
        last = begin
        for j in range(begin+1, end+1):
            ss = _square_slice(info['path'][j], k, scene.shape)
            if ss is None or np.any(info['clean'][ss] == info['wall']):
                halted = dict(stage=i, index=last, center=info['path'][last], size=k, color=color)
                break
            last = j
            travel += 1
        _put_stamp(scene, info['spine'], info['path'][last], k, color)
        if halted is not None:
            break
        reached.append(i)
    return scene, dict(reached=tuple(reached), halted=halted, success=halted is None, travel=travel)


def _crop(a, bb):
    r0, r1, c0, c1 = bb
    return a[r0:r1+1, c0:c1+1].copy()


def _analyze(a):
    info = _parse(a)
    scene, movement = _continue(info)
    info.update(movement)
    return _crop(scene, info['bounds']), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every role, direction, state and stopping position from input only."""
    return _analyze(_grid(input_grid))[0]


def _walk(rng, nr, nc, count):
    # Randomized depth first construction, at most 9 stations and 49 lattice nodes.
    starts = list(rng.permutation(nr*nc))
    for raw in starts:
        start = (int(raw)//nc, int(raw)%nc)
        route, used = [start], {start}
        def extend():
            if len(route) == count:
                return True
            options = list(_neighbors(route[-1], (nr, nc)))
            rng.shuffle(options)
            for q in options:
                if q in used:
                    continue
                used.add(q)
                route.append(q)
                if extend():
                    return True
                route.pop()
                used.remove(q)
            return False
        if extend():
            return route
    raise AssertionError("could not grow station walk")


def _expand(centers):
    route = [centers[0]]
    indices = [0]
    for a, b in zip(centers, centers[1:]):
        dr = (b[0] > a[0]) - (b[0] < a[0])
        dc = (b[1] > a[1]) - (b[1] < a[1])
        p = a
        while p != b:
            p = p[0]+dr, p[1]+dc
            route.append(p)
        indices.append(len(route)-1)
    return route, indices


def _fields(rng, sizes, q, nvisible, success, gate_contrast=False):
    count = len(sizes)
    gap = int(rng.integers(1, 4))
    options = [(nr,nc) for nr in range(1,8) for nc in range(1,8)
               if nr*nc >= count and q*nr+gap*(nr-1) <= 28 and q*nc+gap*(nc-1) <= 28]
    _require(options, "station lattice does not fit")
    nr, nc = options[int(rng.integers(len(options)))]
    nodes = _walk(rng, nr, nc, count)
    def coordinates(n):
        steps = [q+gap]*(n-1)
        budget = 28-q-sum(steps)
        for j in rng.permutation(n-1):
            extra = int(rng.integers(min(4,budget)+1))
            steps[int(j)] += extra
            budget -= extra
        return [q//2]+[q//2+int(v) for v in np.cumsum(steps)]
    rows,cols = coordinates(nr),coordinates(nc)
    centers = [(rows[r],cols[c]) for r,c in nodes]
    h,w = rows[-1]+q//2+1,cols[-1]+q//2+1
    terrain = np.zeros((h,w), dtype=bool)
    rooms = []
    for i, p in enumerate(centers):
        needed = max(sizes[i], sizes[min(i+1,count-1)])
        room = int(rng.choice(list(range(needed,q+1,2))))
        rooms.append(room)
        r,c = p
        d = room//2
        terrain[r-d:r+d+1,c-d:c+d+1] = True
    collision = None if success else (nvisible if gate_contrast else
                                     int(rng.integers(nvisible-1,count-1)))
    widths = []
    for i, (a,b) in enumerate(zip(centers,centers[1:])):
        cap = min(rooms[i],rooms[i+1])
        lower = sizes[i+1] if collision is None or i < collision else 1
        width = int(rng.integers(lower,cap+1))
        if i == collision:
            width = int(rng.integers(1,sizes[i+1]))
        if gate_contrast and i in (nvisible-1,nvisible):
            width = 3
        widths.append(width)
        # Even widths may have either alignment; every passage contains the spine.
        low = width//2 if width%2 or bool(rng.integers(2)) else width//2-1
        if a[0] == b[0]:
            terrain[a[0]-low:a[0]-low+width,min(a[1],b[1]):max(a[1],b[1])+1] = True
        else:
            terrain[min(a[0],b[0]):max(a[0],b[0])+1,a[1]-low:a[1]-low+width] = True
    # Trim unused lattice margins, then add a visibly uniform surrounding wall.
    cells = np.argwhere(terrain)
    lo, hi = cells.min(axis=0), cells.max(axis=0)+1
    terrain = terrain[lo[0]:hi[0],lo[1]:hi[1]]
    pads = []
    for d in terrain.shape:
        spare = 30-d-2
        before = 1+int(rng.integers(min(3,spare)+1))
        after = 1+int(rng.integers(min(3,spare-(before-1))+1))
        pads.append((before,after))
    terrain = np.pad(terrain,pads)
    centers = [(r-int(lo[0])+pads[0][0],c-int(lo[1])+pads[1][0]) for r,c in centers]
    route, indices = _expand(centers)
    spine = np.zeros(terrain.shape,dtype=bool)
    for p in route:
        spine[p] = True
    return terrain, spine, centers, route, indices, tuple(widths)


def _latent_render(inp, terrain, spine, route, indices, sizes, phases, nvisible):
    """Construction-side labels: centered brush and first invalid center."""
    out = inp.copy()
    stopped = False
    for stage in range(nvisible,len(indices)):
        radius = sizes[stage]//2
        destination = indices[stage-1]
        for t in range(indices[stage-1]+1,indices[stage]+1):
            r,c = route[t]
            if not terrain[r-radius:r+radius+1,c-radius:c+radius+1].all():
                stopped = True
                break
            destination = t
        r,c = route[destination]
        for rr in range(r-radius,r+radius+1):
            for cc in range(c-radius,c+radius+1):
                if not spine[rr,cc]:
                    out[rr,cc] = phases[stage]
        if stopped:
            break
    return out, stopped


def _transform(a, turns, reflect):
    a = np.rot90(a,turns)
    return (a[:,::-1] if reflect else a).copy()


def _choose_crop(rng, inp, scene, wall, required_differences, partial):
    changed = np.argwhere(inp != scene)
    _require(len(changed), "continuation must paint something")
    h,w = inp.shape
    candidates = []
    for _ in range(100):
        rr,cc = map(int,changed[int(rng.integers(len(changed)))])
        if partial:
            r0 = int(rng.integers(rr+1))
            r1 = int(rng.integers(rr,h))
            c0 = int(rng.integers(cc+1))
            c1 = int(rng.integers(cc,w))
        else:
            lo,hi = changed.min(axis=0),changed.max(axis=0)
            r0 = int(rng.integers(int(lo[0])+1))
            r1 = int(rng.integers(int(hi[0]),h))
            c0 = int(rng.integers(int(lo[1])+1))
            c1 = int(rng.integers(int(hi[1]),w))
        bb = r0,r1,c0,c1
        if r1-r0 < 1 or c1-c0 < 1 or r1-r0+c1-c0 < 4:
            continue
        if any(inp[p] != wall for p in _corners(bb)):
            continue
        if all(np.any(_crop(mask,bb)) for mask in required_differences):
            candidates.append(bb)
    if candidates:
        return candidates[int(rng.integers(len(candidates)))]
    return 0,h-1,0,w-1


def _make_pair(rng, conventions, kind, outcome=None):
    wall,floor,road,pin,annotation = conventions[:5]
    palette = conventions[5:]
    contrast = kind == 'contrast'
    if kind == 'arithmetic':
        q,visible,count = 9,3,4
        sizes = (3,5,7,9) if bool(rng.integers(2)) else (9,7,5,3)
        color_period = 2
        success = True
    elif contrast:
        q,visible,count = 7,4,6
        sizes = (3,5,3,5,3,5)
        color_period = 3
        success = False
    else:
        test = kind == 'test'
        size_period = int(rng.integers(1,4))
        color_period = int(rng.integers(1,4))
        visible = max(3,size_period+1,color_period+1)
        future = int(rng.integers(3,6)) if test else int(rng.integers(1,3))
        count = min(9,visible+future)
        if size_period == 1:
            k = int(rng.choice([3,5,7]))
            pattern = (k,)
            q = int(rng.choice(list(range(k,8,2))))
        else:
            pattern = tuple(map(int,rng.choice([3,5,7],size_period,replace=False)))
            q = max(pattern)
        sizes = tuple(pattern[i%size_period] for i in range(count))
        success = bool(rng.integers(2)) if outcome is None else outcome
    palette = tuple(palette[int(i)] for i in rng.permutation(len(palette)))
    phases = tuple(palette[i%color_period] for i in range(count))
    terrain,spine,stations,path,indices,widths = _fields(rng,sizes,q,visible,success,contrast)
    inp = np.full(terrain.shape,wall,dtype=np.int8)
    inp[terrain] = floor
    inp[spine] = road
    for p in stations:
        inp[p] = pin
    for i in range(visible):
        _put_stamp(inp,spine,stations[i],sizes[i],phases[i])
    scene,stopped = _latent_render(inp,terrain,spine,path,indices,sizes,phases,visible)
    _require(stopped != success, "construction outcome mismatch")
    if np.array_equal(inp,scene):
        raise _RetryPair()
    latent = dict(clean=inp,wall=wall,spine=spine,path=path,indices=indices,
                  stations=stations,visible=visible,sizes=sizes,phases=phases)
    fixed_sizes = tuple([sizes[visible-1]]*count)
    size_only,_ = _continue(latent,sizes=fixed_sizes)
    fixed_phase,_ = _continue(latent,phases=tuple([phases[visible-1]]*count))
    no_walls,_ = _continue(latent,ignore_walls=True)
    turns,reflect = int(rng.integers(4)),bool(rng.integers(2))
    inp,scene = _transform(inp,turns,reflect),_transform(scene,turns,reflect)
    differences = [inp != scene]
    if kind in ('arithmetic','contrast'):
        differences.extend([scene != _transform(size_only,turns,reflect),
                            scene != _transform(fixed_phase,turns,reflect)])
    if contrast:
        differences.append(scene != _transform(no_walls,turns,reflect))
    bb = _choose_crop(rng,inp,scene,wall,differences,kind not in ('arithmetic','contrast'))
    answer = _crop(scene,bb)
    for p in _corners(bb):
        inp[p] = annotation
    predicted,info = _analyze(inp)
    _require(np.array_equal(predicted,answer), "pixel recovery disagrees with constructed trace")
    _require(info['success'] == success, "recovered outcome")
    return {'input':inp,'output':answer}


def _counts(num_train,num_test):
    for value,low,high in ((num_train,2,10),(num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= value <= high:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train),int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train,num_test = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    conventions = tuple(map(int,rng.permutation(10)))[:8]
    variable_palette = bool(rng.integers(2))
    puzzle,seen = {'train':[],'test':[]},set()
    train_kinds = ['arithmetic','contrast']+['ordinary']*(num_train-2)
    # Demonstration order is not an outcome cue.
    train_kinds = [train_kinds[int(i)] for i in rng.permutation(num_train)]
    for split,kinds in (('train',train_kinds),('test',['test']*num_test)):
        for kind in kinds:
            for _ in range(60):
                roles = tuple(map(int,rng.permutation(10)))[:8] if variable_palette else conventions
                try:
                    pair = _make_pair(rng,roles,kind)
                except _RetryPair:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "split lists")
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen,training,testing = set(),[],[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'}, "pair schema")
            inp,out = _grid(pair['input']),_grid(pair['output'])
            _require(_key(inp) not in seen,"duplicate input")
            seen.add(_key(inp))
            prediction,info = _analyze(inp)
            _require(np.array_equal(prediction,out),"wrong crop, phase, footprint or stop")
            _require(info['annotation'] not in out,"annotation leaked into answer")
            _require(set(map(int,np.unique(out))) <= set(map(int,np.unique(inp))),"invented color")
            scene,movement = _continue(info)
            _require(np.array_equal(scene[info['spine']],info['clean'][info['spine']]),"spine/pins changed")
            _require(np.array_equal(scene[info['clean'] == info['wall']],info['clean'][info['clean'] == info['wall']]),"wall painted")
            _require(np.any(out != _crop(info['clean'],info['bounds'])),"window contains no continuation evidence")
            if movement['halted']:
                stop = movement['halted']
                ss = _square_slice(stop['center'],stop['size'],inp.shape)
                _require(not np.any(info['clean'][ss] == info['wall']),"stopped square overlaps wall")
                next_point = info['path'][stop['index']+1]
                nxt = _square_slice(next_point,stop['size'],inp.shape)
                _require(nxt is None or np.any(info['clean'][nxt] == info['wall']),"square stopped before legal motion")
                _require(all(i < stop['stage'] for i in movement['reached']),"stamps after obstruction")
            fixed,_ = _continue(info,sizes=tuple([info['sizes'][info['visible']-1]]*len(info['sizes'])))
            phase,_ = _continue(info,phases=tuple([info['phases'][info['visible']-1]]*len(info['sizes'])))
            unconstrained,_ = _continue(info,ignore_walls=True)
            info['size_ablation'] = not np.array_equal(out,_crop(fixed,info['bounds']))
            info['color_ablation'] = not np.array_equal(out,_crop(phase,info['bounds']))
            info['wall_ablation'] = not np.array_equal(out,_crop(unconstrained,info['bounds']))
            (training if split == 'train' else testing).append(info)
    _require(any(v['success'] and v['travel'] > 0 for v in training),"need a moving arrival demonstration")
    _require(any(not v['success'] and v['travel'] > 0 and v['wall_ablation'] for v in training),"need an informative collision demonstration")
    _require(any(v['size_ablation'] for v in training) and any(v['color_ablation'] for v in training),"A contribution not demonstrated")
    _require(any(v['size_mode'] == 'arithmetic' and v['size_parameter'] != 0 for v in training),"arithmetic object change missing")
    _require(any(v['size_mode'] == 'periodic' and v['size_parameter'] > 1 for v in training),"size recurrence missing")
    _require(max(len(v['stations'])-v['visible'] for v in training) <= 2,"training continuation span")
    _require(all(len(v['stations'])-v['visible'] >= 3 for v in testing),"test must continue over more stations")


def _fingerprint(puzzle):
    return tuple((_key(p['input']),_key(p['output'])) for s in ('train','test') for p in puzzle[s])


def self_test():
    """Bounded semantic fixtures, 42-seed variation, counts and corruption tests."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_fingerprint(p) for p in puzzles}) == 42,"cross-seed repetition")
    features = []
    structures,shapes,roles = set(),set(),set()
    for puzzle in puzzles:
        for split in ('train','test'):
            for pair in puzzle[split]:
                expected,info = _analyze(pair['input'])
                _require(np.array_equal(expected,pair['output']),"42-seed solve disagreement")
                shapes.add((pair['input'].shape,pair['output'].shape))
                roles.add(tuple(info[k] for k in ('wall','floor','road','pin','annotation')))
                normalized = np.zeros(pair['input'].shape,dtype=np.int8)
                for code,key in enumerate(('wall','floor','road','pin','annotation')):
                    normalized[pair['input'] == info[key]] = code
                for i,color in enumerate(dict.fromkeys(info['phases'])):
                    normalized[pair['input'] == color] = i+5
                structures.add(_key(normalized))
                headings = [(b[0]-a[0],b[1]-a[1]) for a,b in zip(info['path'],info['path'][1:])]
                bends = sum(a != b for a,b in zip(headings,headings[1:]))
                features.append((split,info,bends,headings[0]))
    _require(len(structures) >= 160 and len(shapes) >= 80,"structural variation collapsed")
    _require(len(roles) >= 35,"palette-role variation missing")
    _require({v[3] for v in features} == {(-1,0),(1,0),(0,-1),(0,1)},"route orientations")
    _require(max(v[2] for v in features) >= 5,"winding route variation")
    _require({v[1]['color_period'] for v in features} == {1,2,3},"color periods")
    _require({v[1]['size_parameter'] for v in features if v[1]['size_mode'] == 'periodic'} >= {2,3},"size periods")
    _require({v[1]['size_parameter'] for v in features if v[1]['size_mode'] == 'arithmetic'} >= {-2,0,2},"growth and shrinkage")
    _require({v[1]['success'] for v in features if v[0] == 'test'} == {False,True},"fixed test outcome")
    _require(len({len(v[1]['stations']) for v in features}) >= 5,"station counts")
    _require(sum(len(set(b-a for a,b in zip(v[1]['indices'],v[1]['indices'][1:]))) > 1
                 for v in features) >= 100,"unequal route-leg lengths missing")
    _require(sum(v[1]['halted'] is not None and v[1]['halted']['index'] not in v[1]['indices']
                 for v in features) >= 20,"off-station collision positions missing")
    # Explicit coordinate fixture: same width-three gate first passes k=3,
    # then blocks k=5. Last fitting center is one cell left of station 4.
    rng = np.random.default_rng(93)
    sizes,phases = (3,5,3,5,3,5),(4,5,6,4,5,6)
    centers = [(4,4),(4,12),(4,20),(12,20),(12,12),(12,4)]
    route,indices = _expand(centers)
    terrain = np.zeros((17,25),dtype=bool)
    for r,c in centers:
        terrain[r-3:r+4,c-3:c+4] = True
    for i,(a,b) in enumerate(zip(centers,centers[1:])):
        width = 3 if i >= 3 else 7
        d = width//2
        if a[0] == b[0]:
            terrain[a[0]-d:a[0]+d+1,min(a[1],b[1]):max(a[1],b[1])+1] = True
        else:
            terrain[min(a[0],b[0]):max(a[0],b[0])+1,a[1]-d:a[1]+d+1] = True
    spine = np.zeros(terrain.shape,dtype=bool)
    for p in route:
        spine[p] = True
    inp = np.zeros(terrain.shape,dtype=np.int8)
    inp[terrain],inp[spine] = 8,1
    for p in centers:
        inp[p] = 2
    for i in range(4):
        _put_stamp(inp,spine,centers[i],sizes[i],phases[i])
    expected_scene = inp.copy()
    _put_stamp(expected_scene,spine,(12,12),3,5)
    _put_stamp(expected_scene,spine,(12,11),5,6)
    bb = (0,16,0,24)
    for p in _corners(bb):
        inp[p] = 9
    result,info = _analyze(inp)
    _require(np.array_equal(result,expected_scene),"coordinate bottleneck fixture")
    _require(info['reached'] == (4,) and info['halted']['center'] == (12,11),"gate and stop fixture")
    # Orientation and arbitrary role permutations do not alter the physics.
    for turns in range(4):
        mapping = rng.permutation(10)
        base = inp.copy()
        base[base == 9] = 0
        transformed = _transform(mapping[base],turns,True)
        truth = _transform(mapping[expected_scene],turns,True)
        corners = (0,len(transformed)-1,0,transformed.shape[1]-1)
        for p in _corners(corners):
            transformed[p] = mapping[9]
        _require(np.array_equal(solve(transformed),truth),"orientation/role fixture")
    for n in range(2,11):
        for m in range(1,min(4,n-1)+1):
            p = generate_puzzle(211,n,m)
            _require(len(p['train']) == n and len(p['test']) == m,"explicit counts")
    _require(_fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0)),"determinism")
    for n,m in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(2.0,1),(True,1),(4,True),(None,1)):
        try:
            generate_puzzle(0,n,m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    for mutation in ('label','pin','corner'):
        bad = {s:[{k:np.asarray(v).copy() for k,v in p.items()} for p in puzzles[0][s]] for s in ('train','test')}
        pair = bad['test'][0]
        _,info = _analyze(pair['input'])
        if mutation == 'label':
            pair['output'][0,0] = (int(pair['output'][0,0])+1)%10
        elif mutation == 'pin':
            pair['input'][info['stations'][-1]] = info['road']
        else:
            pair['input'][min(_corners(info['bounds']))] = info['wall']
        try:
            validate_generated_puzzle(bad)
        except (AssertionError,ValueError):
            continue
        raise AssertionError("validator accepted "+mutation+" corruption")
    return {'seeds':42,'pairs':len(features),'normalized_inputs':len(structures),
            'shape_pairs':len(shapes),'role_assignments':len(roles),
            'max_route_bends':max(v[2] for v in features)}
