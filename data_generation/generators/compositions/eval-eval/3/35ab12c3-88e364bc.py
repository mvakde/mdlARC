"""Dock a completed, hollow wireframe by one common wall-limited translation.

Standard library and NumPy only. No files, seeds, or examples are read by solve.
All generation retries, geometry searches, and self-test sweeps are bounded.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)
CHECK_SEEDS = (0,785761583,312124423,242384987,373997867,375883322,962776944,163382235,108089863,467923613,454315177,447991022,659657969,112275306,206397888,154871986,437253914,511290381,47271124,515722032,456631652,168347620,873790524,393087834,992751859,908425191,173702,573580777,110022433,925598856,855103100,250920567,164982852,823786374,303939192,928662551,684529926,731234804,170183596,153556205,935052228,71855335)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _sign(x):
    return int(x > 0) - int(x < 0)


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if not (dr or dc) or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        raise ValueError("unaligned or coincident landmarks")
    return {(a[0] + k * _sign(dr), a[1] + k * _sign(dc))
            for k in range(max(abs(dr), abs(dc)) + 1)}


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _outline(points):
    """Unique noncrossing Hamiltonian cycle first, otherwise unique full path.

    Edges cannot pass through another landmark; continuous crossings, including
    half-cell diagonal crossings, are forbidden. Uniqueness concerns edge sets,
    so reversal and a cycle's starting vertex do not create false ambiguity.
    """
    points = tuple(sorted(points))
    n = len(points)
    if not 2 <= n <= 10 or len(set(points)) != n:
        raise ValueError("need 2..10 distinct landmarks")
    edges, lines = [], []
    landmarks = set(points)
    for i, a in enumerate(points):
        for j in range(i + 1, n):
            try:
                cells = _segment(a, points[j])
            except ValueError:
                continue
            if (cells - {a, points[j]}) & landmarks:
                continue
            edges.append((i, j))
            lines.append(cells)
    adjacent = [[] for _ in points]
    for e, (i, j) in enumerate(edges):
        adjacent[i].append((j, e))
        adjacent[j].append((i, e))
    if not all(adjacent):
        raise ValueError("disconnected landmark")
    conflicts = [set() for _ in edges]
    for e, (i, j) in enumerate(edges):
        for f, (k, l) in enumerate(edges[:e]):
            if not {i, j} & {k, l} and _intersects(points[i], points[j], points[k], points[l]):
                conflicts[e].add(f)
                conflicts[f].add(e)

    def search(closed):
        answers, visits = {}, 0

        def walk(order, used, chosen):
            nonlocal visits
            visits += 1
            if visits > 50000:
                raise ValueError("ambiguous geometry exceeds bounded search")
            if len(answers) > 1:
                return
            if len(order) == n:
                final = chosen
                if closed:
                    closing = [e for v, e in adjacent[order[-1]] if v == order[0]]
                    if not closing or conflicts[closing[0]] & set(chosen):
                        return
                    final = chosen + [closing[0]]
                answers[tuple(sorted(final))] = order
                return
            for v, e in adjacent[order[-1]]:
                if v not in used and not conflicts[e] & set(chosen):
                    walk(order + [v], used | {v}, chosen + [e])

        for first in ([0] if closed else range(n)):
            walk([first], {first}, [])
            if len(answers) > 1:
                break
        return answers

    cycles = search(True) if n >= 3 else {}
    if len(cycles) > 1:
        raise ValueError("competing closed outlines")
    choices = cycles or search(False)
    if len(choices) != 1:
        raise ValueError("no unique simple outline")
    selected, order = next(iter(choices.items()))
    cells = set().union(*(lines[e] for e in selected))
    if cells == landmarks:
        raise ValueError("no missing wire cells")
    all_cells = set().union(*lines)
    long_cells = set().union(*(lines[e] for e in selected if len(lines[e]) > 2))
    turns = [_sign(_cross(points[order[i-1]], points[order[i]], points[order[(i+1) % n]]))
             for i in range(n)] if cycles else []
    return dict(vertices=set(points), cells=cells, unit_only=cells-long_cells, closed=bool(cycles),
                order=[points[i] for i in order],
                chords=all_cells - cells, concave=(-1 in turns and 1 in turns))


def _shift(cells, d, k=1):
    return {(r + k * d[0], c + k * d[1]) for r, c in cells}


def _step_cells(cells, d, k, corner=True):
    """Cells tested when stepping from translation k-1 to translation k."""
    result = _shift(cells, d, k)
    if corner and d[0] and d[1]:
        result |= {(r + k*d[0], c + (k-1)*d[1]) for r, c in cells}
        result |= {(r + (k-1)*d[0], c + k*d[1]) for r, c in cells}
    return result


def _distance(cells, d, walls, shape, corner=True):
    """Set-based solver collision check, with the starting source transparent."""
    h, w = shape
    for k in range(1, 32):
        tested = _step_cells(cells, d, k, corner)
        if tested & walls or any(not (0 <= r < h and 0 <= c < w) for r, c in tested):
            return k - 1
    raise ValueError("unbounded motion")


def _mask_distance(cells, d, wall):
    """Construction uses array indexing rather than the solver's set traversal."""
    rc = np.array(sorted(cells), dtype=int)
    offsets = [(d[0], d[1])]
    if d[0] and d[1]:
        offsets += [(d[0], 0), (0, d[1])]
    for k in range(1, 32):
        current = rc + np.array(d) * (k-1)
        for offset in offsets:
            next_cells = current + np.array(offset)
            if (np.any(next_cells < 0) or np.any(next_cells[:, 0] >= wall.shape[0])
                    or np.any(next_cells[:, 1] >= wall.shape[1])
                    or np.any(wall[next_cells[:, 0], next_cells[:, 1]])):
                return k - 1
    raise AssertionError("closed chamber failed to stop motion")


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return a


def _read(input_grid):
    a = _grid(input_grid)
    bg = int(np.bincount(a.astype(np.int64,copy=False).ravel(), minlength=10).argmax())
    if np.count_nonzero(a == bg) <= a.size // 2:
        raise ValueError("background must occupy more than half the input")
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    wall_color = int(border[0])
    if wall_color == bg or np.any(border != wall_color):
        raise ValueError("wall frame is not monochrome")
    colors = set(map(int, np.unique(a))) - {bg, wall_color}
    if len(colors) != 2:
        raise ValueError("expected source and launch colors")
    singleton = [c for c in colors if np.count_nonzero(a == c) == 1]
    if len(singleton) != 1:
        raise ValueError("launch must have its own singleton color")
    launch_color = singleton[0]
    source_color = next(iter(colors - {launch_color}))
    info = _outline([tuple(map(int, p)) for p in np.argwhere(a == source_color)])
    marker = tuple(map(int, np.argwhere(a == launch_color)[0]))
    anchors = [p for p in info['vertices'] if max(abs(p[0]-marker[0]), abs(p[1]-marker[1])) == 1]
    if len(anchors) != 1 or marker in info['cells']:
        raise ValueError("launch must mark exactly one vertex and be off the wire")
    direction = (marker[0]-anchors[0][0], marker[1]-anchors[0][1])
    walls = {tuple(map(int, p)) for p in np.argwhere(a == wall_color)}
    if info['cells'] & walls:
        raise ValueError("source wire crosses a wall")
    distance = _distance(info['cells'], direction, walls, a.shape)
    if distance < 1:
        raise ValueError("launch's one-cell companion must be legal")
    info.update(background=bg, wall_color=wall_color, source_color=source_color,
                launch_color=launch_color, marker=marker, direction=direction,
                walls=walls, distance=distance)
    return a, info


def _render(a, info, distance):
    out = a.copy()
    out[info['marker']] = info['background']
    for p in _shift(info['cells'], info['direction'], distance):
        out[p] = info['launch_color']
    for p in info['cells']:
        out[p] = info['source_color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, info = _read(input_grid)
    return _render(a, info, info['distance'])


def _inside(p, polygon):
    inside = False
    r, c = p
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[0] > r) != (b[0] > r):
            if c < a[1] + (r-a[0]) * (b[1]-a[1]) / (b[0]-a[0]):
                inside = not inside
    return inside


def _measure(a, info, out):
    s, d, walls, t = info['cells'], info['direction'], info['walls'], info['distance']
    r0, r1 = min(r for r,c in s), max(r for r,c in s)
    c0, c1 = min(c for r,c in s), max(c for r,c in s)
    return dict(shape=a.shape, background=info['background'], wall_color=info['wall_color'],
                direction=d, vertices=len(info['vertices']), closed=info['closed'],
                concave=info['concave'], distance=t,
                vertex_distance=_distance(info['vertices'], d, walls, a.shape),
                cornerless_distance=_distance(s, d, walls, a.shape, corner=False),
                chord_witness=any(a[p] != info['wall_color'] and out[p] != info['source_color'] for p in info['chords']),
                hollow_wall=info['closed'] and any(_inside(p, info['order']) for p in walls - s),
                box_wall=any((r,c) in walls for r in range(r0,r1+1) for c in range(c0,c1+1)),
                overlap=len(s & _shift(s, d, t)),
                new_overlap=len((s & _shift(s, d, t)) - info['vertices']),
                unit_copy_witness=len(_shift(info['unit_only'],d,t) - s),
                marker_erased=out[info['marker']] == info['background'],
                visible_copy=len(_shift(s, d, t) - s),
                geometry=(info['closed'], tuple(sorted((r-r0,c-c0) for r,c in info['vertices']))),
                wall_cells=len(walls))


def _transform(points, code):
    result = []
    for r, c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    mr, mc = min(r for r,c in result), min(c for r,c in result)
    return [(r-mr,c-mc) for r,c in result]


def _sample_shape(rng, family, query=False):
    h, w = int(rng.integers(3,8)), int(rng.integers(3,9))
    if family == 'hollow_demo':
        mode = int(rng.integers(3))
        if mode == 0:
            h = int(rng.integers(5,10))
            points = [(0,0),(0,h),(h,h),(h,0)]
        elif mode == 1:
            k = int(rng.integers(3,6))
            points = [(0,k),(k,2*k),(2*k,k),(k,0)]
        else:
            h = int(rng.integers(6,10))
            a,b,c,d = [int(rng.integers(1,3)) for _ in range(4)]
            points = [(0,a),(0,h-b),(b,h),(h-c,h),(h,h-c),(h,d),(h-d,0),(a,0)]
        closed = True
    elif family == 'corner_demo':
        length = int(rng.integers(4,7))
        tail = int(rng.integers(2,length))
        points = [(0,0),(length,length),(length-tail,length+tail),
                  (length-tail-1,length+tail+1)]
        closed = False
    elif family == 'segment':
        length = int(rng.integers(3,10))
        points, closed = [(0,0),(0,length)], False
    elif family == 'rectangle':
        if rng.random() < .3:
            w = h
        points, closed = [(0,0),(0,w),(h,w),(h,0)], True
    elif family == 'diamond':
        k = int(rng.integers(2,5))
        points, closed = [(0,k),(k,2*k),(2*k,k),(k,0)], True
    elif family == 'notch':
        h, w = int(rng.integers(5,9)), int(rng.integers(5,10))
        left = int(rng.integers(1,w-2))
        right = int(rng.integers(left+1,w))
        depth = int(rng.integers(1,h-1))
        points = [(0,0),(0,w),(h,w),(h,right),(h-depth,right),
                  (h-depth,left),(h,left),(h,0)]
        closed = True
    elif family == 'clipped':
        cuts = [int(rng.integers(0,max(2,min(h,w)//2))) for _ in range(4)]
        a,b,c,d = cuts
        points = [(0,a),(0,w-b),(b,w),(h-c,w),(h,w-c),(h,d),(h-d,0),(a,0)]
        points = [p for i,p in enumerate(points) if p != points[i-1]]
        closed = True
    else:
        points = [(0,0)]
        n = int(rng.integers(4 if query else 3, 8 if query else 6))
        for _ in range(n-1):
            d = DIRECTIONS[int(rng.integers(8))]
            length = int(rng.integers(1,5))
            points.append((points[-1][0]+length*d[0],points[-1][1]+length*d[1]))
        closed = False
    code = int(rng.integers(8))
    points = _transform(points, code)
    if len(points) != len(set(points)) or max(max(p) for p in points) > 13:
        return None
    try:
        info = _outline(points)
    except ValueError:
        return None
    expected = set().union(*(_segment(a,b) for a,b in zip(points,points[1:])))
    if closed:
        expected |= _segment(points[-1],points[0])
    if info['closed'] != closed or info['cells'] != expected:
        return None
    return info


def _wall_frame(shape):
    wall = np.zeros(shape, dtype=bool)
    wall[0] = wall[-1] = True
    wall[:,0] = wall[:,-1] = True
    return wall


def _construct_pair(rng, family, direction_pool, query=False):
    """Construct a critical collision, then add walls away from the legal sweep."""
    for attempt in range(180):
        local = _sample_shape(rng, family, query)
        if local is None:
            continue
        s = local['cells']
        sh, sw = max(r for r,c in s)+1, max(c for r,c in s)+1
        h = int(rng.integers(max(15,sh+9), min(30,sh+16)+1))
        w = int(rng.integers(max(15,sw+9), min(30,sw+16)+1))
        d = direction_pool[int(rng.integers(len(direction_pool)))]
        # Keep useful space ahead in each moving coordinate. Still vary the
        # transverse placement and the exact distance to the outer frame.
        def place(length, extent, delta):
            slack = length - extent - 2
            if delta > 0:
                return int(rng.integers(1,max(2,slack-4)))
            if delta < 0:
                return int(rng.integers(6,slack+2))
            return int(rng.integers(1,slack+2))
        r0, c0 = place(h,sh,d[0]), place(w,sw,d[1])
        vertices = {(r+r0,c+c0) for r,c in local['vertices']}
        s = {(r+r0,c+c0) for r,c in s}
        anchors = [p for p in sorted(vertices) if (p[0]+d[0],p[1]+d[1]) not in s
                   and sum(max(abs(p[0]+d[0]-q[0]),abs(p[1]+d[1]-q[1])) == 1 for q in vertices) == 1]
        if not anchors:
            continue
        anchor = anchors[int(rng.integers(len(anchors)))]
        marker = (anchor[0]+d[0],anchor[1]+d[1])
        wall = _wall_frame((h,w))
        free_limit = _mask_distance(s,d,wall)
        if free_limit < 4:
            continue
        demo = family in ('hollow_demo','corner_demo')
        max_target = min(5 if demo else 8,free_limit-2)
        target = int(rng.integers(2 if demo else 1,max_target+1))
        safe = set(s)
        for k in range(1,target+1):
            safe |= _step_cells(s,d,k)
        next_cells = _shift(s,d,target+1)
        candidates = _step_cells(s,d,target+1) - safe
        if family == 'corner_demo':
            candidates -= next_cells
        # A missing edge must be decisive, rather than just a visible vertex.
        vertex_sweep = set(vertices)
        for k in range(1,target+2):
            vertex_sweep |= _step_cells(vertices,d,k)
        candidates -= vertex_sweep
        candidates = sorted(p for p in candidates if 0 < p[0] < h-1 and 0 < p[1] < w-1)
        if family == 'hollow_demo':
            order = [(r+r0,c+c0) for r,c in local['order']]
            candidates = [p for p in candidates if _inside(p,order)]
        if not candidates:
            continue
        blocker = candidates[int(rng.integers(len(candidates)))]
        wall[blocker] = True
        # Bars, islands, frame-connected baffles. No repair or clipping of the
        # wire: obstacles are simply excluded from its entire legal sweep.
        safe_mask = np.zeros((h,w),bool)
        for p in safe:
            safe_mask[p] = True
        extra_count = int(rng.integers(2 if query else 0,8 if query else 5))
        for _ in range(extra_count):
            for _ in range(12):
                rr, cc = int(rng.integers(h)), int(rng.integers(w))
                bh, bw = int(rng.integers(1,5)), int(rng.integers(1,5))
                if rng.random() < .65:
                    if rng.random() < .5:
                        bh = 1
                    else:
                        bw = 1
                rr1, cc1 = min(h,rr+bh), min(w,cc+bw)
                if not np.any(safe_mask[rr:rr1,cc:cc1]):
                    wall[rr:rr1,cc:cc1] = True
                    break
        distance = _mask_distance(s,d,wall)
        if distance != target:
            continue
        palette = list(map(int,rng.choice(10,4,replace=False)))
        bg, wc, sc, lc = palette
        a = np.full((h,w),bg,dtype=np.int8)
        a[wall] = wc
        for p in vertices:
            a[p] = sc
        a[marker] = lc
        if np.count_nonzero(a == bg) <= a.size//2:
            continue
        construction = dict(cells=s,direction=d,marker=marker,background=bg,
                            source_color=sc,launch_color=lc)
        out = _render(a,construction,distance)
        _, info = _read(a)
        measured = _measure(a,info,out)
        if measured['vertex_distance'] <= distance or measured['visible_copy'] < 2:
            continue
        if family == 'hollow_demo' and not (measured['hollow_wall'] and measured['chord_witness'] and measured['new_overlap']):
            continue
        if family == 'corner_demo' and not (measured['cornerless_distance'] > distance
                                           and measured['unit_copy_witness'] and measured['marker_erased']):
            continue
        return dict(input=a,output=out)
    raise RuntimeError(f"bounded construction exhausted for {family}")


def _fingerprint(grid):
    a = np.asarray(grid,dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer(num_train,'num_train',2,10)
    nq = _integer(num_test,'num_test',1,4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # These are presentation choices only; every semantic convention is fixed.
    required = ['hollow_demo','corner_demo']
    general = ['segment','rectangle','diamond','notch','clipped','walk']
    extra = [general[int(rng.integers(len(general)))] for _ in range(nt-2)]
    plans = required + extra
    rng.shuffle(plans)
    puzzle, seen, familiar = {'train':[],'test':[]}, set(), set()
    for split, count in (('train',nt),('test',nq)):
        for index in range(count):
            family = plans[index] if split == 'train' else str(rng.choice(['notch','clipped','walk','diamond'],p=[.30,.30,.25,.15]))
            if family == 'hollow_demo':
                pool = CARDINAL
            elif family == 'corner_demo':
                pool = DIAGONAL
            else:
                pool = DIRECTIONS
            if split == 'test' and rng.random() < .65:
                novel = tuple(d for d in DIRECTIONS if d not in familiar)
                if novel:
                    pool = novel
            for _ in range(8):
                pair = _construct_pair(rng,family,pool,query=(split == 'test'))
                fp = _fingerprint(pair['input'])
                if fp not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
            seen.add(fp)
            puzzle[split].append(pair)
            if split == 'train':
                familiar.add(_read(pair['input'])[1]['direction'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'},'puzzle keys')
    _require(all(isinstance(puzzle[s],list) for s in puzzle),'split type')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq,'pair counts')
    seen, records = set(), {'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'},'pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape,'canvas changed')
            fp = _fingerprint(a)
            _require(fp not in seen,'duplicate input')
            seen.add(fp)
            _, info = _read(a)
            predicted = _render(a,info,info['distance'])
            _require(np.array_equal(predicted,b),'incorrect wireframe docking')
            _require(np.array_equal(a[a == info['wall_color']],b[a == info['wall_color']]),'walls changed')
            _require(all(b[p] == info['source_color'] for p in info['cells']),'source missing or occluded')
            expected_copy = _shift(info['cells'],info['direction'],info['distance']) - info['cells']
            actual_copy = {tuple(map(int,p)) for p in np.argwhere(b == info['launch_color'])}
            _require(actual_copy == expected_copy,'intermediate trail or missing companion cells')
            _require(np.any(a != b),'identity pair')
            measure = _measure(a,info,b)
            _require(measure['distance'] >= 1 and measure['visible_copy'] >= 2,'uninformative move')
            _require(measure['vertex_distance'] > measure['distance'],'missing edge does not affect docking')
            records[split].append(measure)
    demos = records['train']
    _require(any(r['closed'] and r['chord_witness'] and r['hollow_wall'] and r['new_overlap'] and r['distance'] >= 2 for r in demos),
             'missing closure, hollow collision, or occlusion demonstration')
    _require(any(not r['closed'] and r['cornerless_distance'] > r['distance'] and r['distance'] >= 2
                 and r['unit_copy_witness'] and r['marker_erased'] for r in demos),
             'missing open-path diagonal corner, unit-stub, or launch consumption demonstration')
    _require(any(0 in r['direction'] for r in demos) and any(0 not in r['direction'] for r in demos),
             'need cardinal and diagonal motion')
    return records


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train','test'):
        for pair in puzzle[split]:
            for role in ('input','output'):
                a = np.asarray(pair[role],dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _fixed_regressions():
    # Expected cells below are explicit coordinate formulae, with no call to
    # inference, collision helpers, rasterization, or the output renderer.
    a = np.zeros((13,16),dtype=np.int8)
    a[0] = a[-1] = 5
    a[:,0] = a[:,-1] = 5
    a[6,6] = 5
    for p in ((3,3),(3,9),(9,9),(9,3)):
        a[p] = 2
    a[3,10] = 7
    expected = a.copy()
    expected[3,10] = 0
    source = {(3,c) for c in range(3,10)} | {(9,c) for c in range(3,10)}
    source |= {(r,3) for r in range(3,10)} | {(r,9) for r in range(3,10)}
    for r,c in source:
        expected[r,c+2] = 7
    for p in source:
        expected[p] = 2
    before = a.copy()
    _require(np.array_equal(solve(a),expected),'hollow trailing-edge collision regression')
    _require(np.array_equal(a,before),'solver mutated input')
    # The diagonal cell beyond the stop is empty; only its side cell is a wall.
    a = np.zeros((14,16),dtype=np.int8)
    a[0] = a[-1] = 5
    a[:,0] = a[:,-1] = 5
    a[8,6] = 5
    a[3,6] = a[8,1] = 2
    a[4,7] = 7
    expected = a.copy()
    expected[4,7] = 0
    for r in range(3,9):
        expected[r,9-r] = 2
        expected[r+2,11-r] = 7
    _require(np.array_equal(solve(a),expected),'diagonal side-only missing-edge regression')
    # Input-only solving also covers the specified ten-vertex limit and travel
    # beyond the sampler's eight-cell range. All ten vertices lie on one wire.
    a = np.zeros((30,27),dtype=np.int8)
    a[0] = a[-1] = 5
    a[:,0] = a[:,-1] = 5
    a[18,11] = 5
    for c in range(2,21,2):
        a[4,c] = 2
    a[5,2] = 7
    expected = a.copy()
    expected[5,2] = 0
    expected[4,2:21] = 2
    expected[17,2:21] = 7
    _require(np.array_equal(solve(a),expected),'ten vertices and thirteen-cell travel regression')


def self_test():
    """42 sequential seeds; retain aggregate counters, never a sample archive."""
    _fixed_regressions()
    signatures, shapes, geometries, directions = set(), set(), set(), set()
    metrics = Counter()
    backgrounds, walls, distances, vertex_counts = set(), set(), set(), set()
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        records = validate_generated_puzzle(puzzle)
        for split in ('train','test'):
            for r in records[split]:
                metrics['pairs'] += 1
                for flag in ('closed','concave','chord_witness','hollow_wall','overlap','unit_copy_witness','marker_erased'):
                    metrics[flag] += bool(r[flag])
                metrics['corner_witness'] += r['cornerless_distance'] > r['distance']
                metrics['edge_decisive'] += r['vertex_distance'] > r['distance']
                shapes.add(tuple(r['shape']))
                geometries.add(r['geometry'])
                directions.add(tuple(r['direction']))
                backgrounds.add(r['background'])
                walls.add(r['wall_color'])
                distances.add(r['distance'])
                vertex_counts.add(r['vertices'])
        # Equivariance exercises inference with changed enumeration and palette.
        if seed == 0:
            for pair in puzzle['train'] + puzzle['test']:
                for turns in range(4):
                    for reflect in (False,True):
                        a, b = np.rot90(pair['input'],turns), np.rot90(pair['output'],turns)
                        if reflect:
                            a,b = np.fliplr(a),np.fliplr(b)
                        _require(np.array_equal(solve(a),b),'dihedral equivariance')
                permutation = np.array([7,3,8,0,6,9,1,5,2,4],dtype=np.int8)
                _require(np.array_equal(solve(permutation[pair['input']]),permutation[pair['output']]),'color equivariance')
    _require(len(signatures) == 42,'seed diversity')
    _require(len(shapes) >= 35 and len(geometries) >= 50,'structural diversity')
    _require(directions == set(DIRECTIONS),'all eight directions')
    _require(backgrounds == set(range(10)) and walls == set(range(10)),'palette-role variation')
    _require(len(distances) >= 3 and len(vertex_counts) >= 5,'scale and vertex counts')
    _require(metrics['concave'] >= 5 and metrics['corner_witness'] >= 42,'geometric variation')
    _require(metrics['edge_decisive'] == metrics['pairs'],'A omission must change travel')
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(42+nt*10+nq,nt,nq)
            _require(len(p['train']) == nt and len(p['test']) == nq,'explicit counts')
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    _require(_digest(generate_puzzle(0)) == _digest(generate_puzzle(0)),'determinism')
    # Mutation must be rejected, including an otherwise structurally valid pair.
    bad = generate_puzzle(0)
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('identity corruption accepted')
    return dict(seeds=42,explicit_regressions=3,metrics=dict(metrics),distinct_puzzles=len(signatures),
                canvas_shapes=len(shapes),normalized_geometries=len(geometries),
                directions=len(directions),backgrounds=len(backgrounds),wall_colors=len(walls),
                distances=sorted(distances),vertex_counts=sorted(vertex_counts))


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
