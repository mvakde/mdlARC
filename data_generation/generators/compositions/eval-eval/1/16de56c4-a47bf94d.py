"""Cable cadence: arithmetic progressions measured along traceable paths.

Generation paints from explicit unit-step paths. The input-only solver instead
builds a directional graph, including metrically defined opaque bridges.
"""
from collections import Counter
from itertools import permutations
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SELF_TEST_SEEDS = (0, 678112121, 95985072, 643602489, 274974621, 177875042,
    53353448, 685850022, 86548244, 450888194, 67901591, 66454638, 427597895,
    94888976, 427567385, 602082256, 402822477, 451280758, 508041129,
    917379439, 785880220, 138098668, 764473762, 516874950, 351918716,
    304976360, 784669201, 726637410, 456218793, 493763079, 408581314,
    664514525, 835387883, 626952802, 67768864, 292536790, 683275167,
    746416442, 950930398, 153148521, 99937635, 857433999)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "bad dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer colors required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "colors must be 0..9")
    return g.astype(np.int8, copy=True)


def _components(cells):
    unseen = set(cells)
    answer = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, found = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    found.add(q)
                    stack.append(q)
        answer.append(found)
    return answer


def _mask_rectangles(g, bg):
    candidates = []
    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        rects = []
        for cells in _components(map(tuple, np.argwhere(g == color))):
            rr, cc = zip(*cells)
            r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
            if r1 == r0 or c1 == c0 or len(cells) != (r1-r0+1)*(c1-c0+1):
                break
            rects.append((r0, r1, c0, c1))
        else:
            candidates.append((color, rects))
    _require(len(candidates) <= 1, "ambiguous opaque role")
    return candidates[0] if candidates else (None, [])


def _geometry(g, turn_crossings=False):
    border = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    _require(len(set(map(int, border))) == 1, "uniform blank border required")
    bg = int(border[0])
    mask_color, rects = _mask_rectangles(g, bg)
    visible = {tuple(map(int, p)) for p in np.argwhere(g != bg)
               if int(g[tuple(p)]) != mask_color}
    _require(bool(visible), "missing cables")
    graph = {p: {} for p in visible}
    for p in sorted(visible):
        for d, (dr, dc) in enumerate(DIRS):
            q = p[0]+dr, p[1]+dc
            if q in visible:
                graph[p][d] = (q, (d+2) % 4, 1)
    mask_links = []
    for rect in rects:
        r0, r1, c0, c1 = rect
        ports = []
        sides = [((r0-1, c), 2) for c in range(c0, c1+1)]
        sides += [((r1+1, c), 0) for c in range(c0, c1+1)]
        sides += [((r, c0-1), 1) for r in range(r0, r1+1)]
        sides += [((r, c1+1), 3) for r in range(r0, r1+1)]
        # A parallel degree-two strand beside a rectangle is not a port.
        for p, inward in sides:
            if p in graph and len(graph[p]) in (1, 3):
                ports.append((p, inward))
        _require(len(ports) >= 2 and len(ports) % 2 == 0, "unpaired mask ports")
        links, remaining = [], set(ports)
        if len(ports) == 2:
            a, b = ports
            _require(a[1] != b[1], "same-side hidden U-turn is unspecified")
            links.append((a, b))
        else:
            while remaining:
                a = min(remaining)
                remaining.remove(a)
                p, d = a
                matches = [b for b in remaining if b[1] == (d+2) % 4
                    and b[0][1 if d % 2 == 0 else 0] == p[1 if d % 2 == 0 else 0]]
                _require(len(matches) == 1, "mask conceals an unspecified permutation")
                b = matches[0]
                remaining.remove(b)
                links.append((a, b))
        for (p, dp), (q, dq) in links:
            _require(dp not in graph[p] and dq not in graph[q], "conflicting mask link")
            distance = abs(p[0]-q[0]) + abs(p[1]-q[1])
            graph[p][dp] = q, dq, distance
            graph[q][dq] = p, dp, distance
        mask_links.append((rect, links))
    _require(all(len(edges) in (1, 2, 4) for edges in graph.values()), "branch or break")
    crosses = {p for p in graph if len(graph[p]) == 4}
    counts = Counter(int(g[p]) for p in visible - crosses)
    carrier, frequency = counts.most_common(1)[0]
    _require(sum(v == frequency for v in counts.values()) == 1, "carrier role tied")
    cross_hues = {int(g[p]) for p in crosses}
    _require(len(cross_hues) <= 1 and carrier not in cross_hues, "crosses need a separate hue")
    _require(not cross_hues.intersection(counts), "cross hue used outside intersections")
    endpoints = {p for p in graph if len(graph[p]) == 1}
    _require(2 <= len(endpoints) <= 8 and len(endpoints) % 2 == 0, "invalid cable ends")
    for p in endpoints:
        _require(not any(r0-1 <= p[0] <= r1+1 and c0-1 <= p[1] <= c1+1
                         for r0, r1, c0, c1 in rects), "end touches an opaque region")
    remaining, routes, seen_cells, seen_edges = set(endpoints), [], set(), set()
    while remaining:
        start = min(remaining)
        p, incoming, index = start, None, 0
        samples, visited = [], set()
        while True:
            _require(p not in visited, "self-intersecting or cyclic cable")
            visited.add(p)
            samples.append((p, index))
            seen_cells.add(p)
            edges = graph[p]
            if incoming is not None and p in endpoints:
                _require(p in remaining and p != start, "nonbijective endpoint pairing")
                remaining.remove(start)
                remaining.remove(p)
                break
            if incoming is None:
                outgoing = next(iter(edges))
            elif len(edges) == 4:
                outgoing = incoming ^ 1 if turn_crossings else (incoming+2) % 4
            else:
                _require(incoming in edges, "bad arrival direction")
                outgoing = next(d for d in edges if d != incoming)
            q, arrival, distance = edges[outgoing]
            edge = frozenset(((p, outgoing), (q, arrival)))
            _require(edge not in seen_edges, "two cables share an edge")
            seen_edges.add(edge)
            p, incoming, index = q, arrival, index + distance
        routes.append(samples)
    _require(seen_cells == visible, "unattached path fragment")
    _require(len(seen_edges)*2 == sum(len(e) for e in graph.values()), "unused strand")
    return dict(background=bg, carrier=carrier, crosses=crosses,
                cross_color=next(iter(cross_hues)) if cross_hues else None,
                mask_color=mask_color, masks=mask_links, graph=graph, routes=routes)


def _signal(g, samples, info):
    marks = [(index, int(g[p])) for p, index in samples
             if p not in info['crosses'] and int(g[p]) != info['carrier']]
    counts = Counter(color for _, color in marks)
    repeated = [c for c, n in counts.items() if n > 1]
    _require(len(repeated) == 1 and counts[repeated[0]] == 2, "need one reference pair per cable")
    reference = repeated[0]
    a, b = sorted(i for i, c in marks if c == reference)
    d = b-a
    _require(d > 0, "zero cadence")
    extras = [(i, c) for i, c in marks if c != reference]
    aligned = [(i, c) for i, c in extras if (i-a) % d == 0]
    _require(len(aligned) <= 1, "competing on-phase terminals")
    terminal = aligned[0] if aligned else None
    _require(terminal is None or terminal[0] < a or terminal[0] > b, "interior terminal")
    return dict(a=a, b=b, period=d, reference=reference, terminal=terminal,
                distractors=[(i, c) for i, c in extras if terminal != (i, c)])


def _analyze(input_grid):
    g = _grid(input_grid)
    info = _geometry(g)
    info['signals'] = [_signal(g, samples, info) for samples in info['routes']]
    return g, info


def _render(g, info):
    out = g.copy()
    for samples, s in zip(info['routes'], info['signals']):
        if s['terminal'] is None:
            low, high, color = 0, samples[-1][1], s['reference']
        else:
            t, color = s['terminal']
            low, high = min(s['a'], t), max(s['b'], t)
        for p, i in samples:
            if p not in info['crosses'] and low <= i <= high and (i-s['a']) % s['period'] == 0:
                out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, pairings, distances, references and stops from the input."""
    g, info = _analyze(input_grid)
    return _render(g, info)


def _append_line(path, q):
    p = path[-1]
    _require(p[0] == q[0] or p[1] == q[1], "nonorthogonal construction segment")
    dr, dc = int(np.sign(q[0]-p[0])), int(np.sign(q[1]-p[1]))
    while p != q:
        p = p[0]+dr, p[1]+dc
        path.append(p)


def _make_geometry(rng, n, long_tail, target=None):
    # Disjoint source/destination column sets permit every permutation without
    # a fixed adjacent-swap lattice. Excursions in source gaps add true distance.
    source_gaps = [4] * (n-1)
    dest_gaps = [3] * (n-1)
    left, middle, right = 3, 3, 1
    min_width = left + sum(source_gaps) + middle + sum(dest_gaps) + right + 1
    budget = 30-min_width
    slots = ['left', 'middle', 'right'] + [('s', i) for i in range(n-1)] + [('d', i) for i in range(n-1)]
    for _ in range(int(rng.integers(min(8, budget)+1))):
        slot = slots[int(rng.integers(len(slots)))]
        if slot == 'left': left += 1
        elif slot == 'middle': middle += 1
        elif slot == 'right': right += 1
        elif slot[0] == 's': source_gaps[slot[1]] += 1
        else: dest_gaps[slot[1]] += 1
    sources = [left]
    for gap in source_gaps: sources.append(sources[-1]+gap)
    destinations = [sources[-1]+middle]
    for gap in dest_gaps: destinations.append(destinations[-1]+gap)
    width = destinations[-1]+right+1
    starts = list(map(int, rng.integers(1, 3, n)))
    wiggles = [bool(rng.random() < .7) for _ in range(n)]
    wiggles[0] = True
    depths = [int(rng.integers(3, 5)) for _ in range(n)]
    first_turn = max(a+2+b if w else a for a, b, w in zip(starts, depths, wiggles))+3
    # Cap spare vertical variation before allocating individual gaps/tails.
    turn_rows = [first_turn+3*i for i in range(n)]
    min_tail = 6 if long_tail else 4
    if turn_rows[-1]+min_tail+2 > 30:
        first_turn -= turn_rows[-1]+min_tail+2-30
        turn_rows = [first_turn+3*i for i in range(n)]
    _require(first_turn >= max(a+2+b if w else a for a,b,w in zip(starts,depths,wiggles))+2,
             "insufficient wiggle clearance")
    spare = 30-(turn_rows[-1]+min_tail+2)
    for _ in range(int(rng.integers(min(4, spare)+1))):
        k = int(rng.integers(n))
        for j in range(k, n): turn_rows[j] += 1
    assignments = list(map(int, rng.permutation(n)))
    permutation = list(map(int, rng.permutation(n))) if target is None else list(target)
    _require(sorted(permutation) == list(range(n)), "invalid endpoint permutation")
    end_base = turn_rows[-1]+min_tail
    height = min(30, end_base+2+int(rng.integers(min(3, 30-end_base-2)+1)))
    paths = []
    for i in range(n):
        x, y = sources[i], destinations[permutation[i]]
        start = starts[i]
        path = [(start, x)]
        if wiggles[i]:
            reach = min(3, (source_gaps[i-1] if i else left+1)-2)
            reach = max(2, reach)
            reach = int(rng.integers(2, reach+1))
            for q in ((start+2, x), (start+2, x-reach),
                      (start+2+depths[i], x-reach), (start+2+depths[i], x)):
                _append_line(path, q)
        turn = turn_rows[assignments[i]]
        for q in ((turn, x), (turn, y), (int(rng.integers(end_base, height-1)), y)):
            _append_line(path, q)
        paths.append(path)
    field = np.zeros((height, width), dtype=np.int8)
    owners = {}
    for i, path in enumerate(paths):
        for p in path:
            field[p] = 1
            owners.setdefault(p, set()).add(i)
    for p, ids in owners.items():
        if len(ids) > 1:
            _require(len(ids) == 2, "triple crossing")
            field[p] = 2
    _geometry(field)
    return field, paths


def _inside(p, rect):
    r0, r1, c0, c1 = rect
    return r0 <= p[0] <= r1 and c0 <= p[1] <= c1


def _try_mask(field, paths, rect, previous):
    r0, r1, c0, c1 = rect
    h, w = field.shape
    if not (1 <= r0 < r1 < h-1 and 1 <= c0 < c1 < w-1): return None
    if any(not (r1+2 < a or b+2 < r0 or c1+2 < x or y+2 < c0)
           for a, b, x, y in previous): return None
    if any(r0-2 <= p[0] <= r1+2 and c0-2 <= p[1] <= c1+2
           for path in paths for p in (path[0], path[-1])): return None
    actual = set()
    for path in paths:
        entry = None
        for k, (a, b) in enumerate(zip(path, path[1:])):
            if not _inside(a, rect) and _inside(b, rect): entry = (a, k)
            if _inside(a, rect) and not _inside(b, rect):
                if entry is None: return None
                p, start_index = entry
                if k+1-start_index != abs(p[0]-b[0])+abs(p[1]-b[1]): return None
                actual.add(frozenset((p, b)))
                entry = None
    if not actual: return None
    candidate = field.copy()
    candidate[r0:r1+1, c0:c1+1] = 3
    try:
        info = _geometry(candidate)
    except ValueError:
        return None
    links = next(links for r, links in info['masks'] if r == rect)
    if {frozenset((a[0], b[0])) for a,b in links} != actual: return None
    # Direct path identity and metric checks also catch side-adjacent false ports.
    inferred = {frozenset((s[0][0], s[-1][0])): s[-1][1] for s in info['routes']}
    expected = {frozenset((p[0], p[-1])): len(p)-1 for p in paths}
    if inferred != expected: return None
    return candidate


def _add_masks(rng, field, paths, wanted, mandatory):
    rects = []
    if mandatory:
        # Hide the first return elbow of route zero's excursion. It has two
        # perpendicular ports and a shortest length of four steps, explicitly
        # teaching hidden bends before they occur in a query.
        path = paths[0]
        r,c = next(p for a,p,b in zip(path,path[1:],path[2:])
                   if p[0]-a[0] == 1 and b[1]-p[1] == 1)
        rect = (r-1,r+1,c,c+1)
        candidate = _try_mask(field, paths, rect, [])
        if candidate is None: return None
        field, rects = candidate, [rect]
    points = sorted({p for path in paths for p in path})
    for _ in range(90):
        if len(rects) >= wanted: break
        p = points[int(rng.integers(len(points)))]
        hh, ww = int(rng.integers(2, 6)), int(rng.integers(2, 10))
        r0, c0 = p[0]-int(rng.integers(hh)), p[1]-int(rng.integers(ww))
        rect = (r0, r0+hh-1, c0, c0+ww-1)
        candidate = _try_mask(field, paths, rect, rects)
        if candidate is not None:
            field, rects = candidate, rects+[rect]
    return field


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _choose_signal(rng, field, path, mode, lesson=None):
    length = len(path)
    eligible = {i for i,p in enumerate(path) if field[p] == 1}
    masked_runs = [(i, j) for i, j in zip(sorted(eligible), sorted(eligible)[1:])
                   if j-i > 1 and any(field[path[k]] == 3 for k in range(i+1,j))]
    periods = [1] if lesson == 'solid' else list(range(2, min(12, (length-1)//2)+1)) if lesson else list(range(1,min(14,(length-1)//2)+1))
    rng.shuffle(periods)
    for d in periods:
        geometries = []
        for a in sorted(eligible):
            b = a+d
            if b not in eligible: continue
            phase = [i for i in sorted(eligible) if (i-a) % d == 0]
            if mode == 'boundary':
                if not any(i not in (a,b) for i in phase): continue
                if lesson == 'boundary':
                    if not (any(i<a for i in phase) and any(i>b for i in phase)): continue
                    if path[a][0] == path[b][0] or path[a][1] == path[b][1]: continue
                geometries.append((a,b,None))
            else:
                for t in phase:
                    if a <= t <= b: continue
                    lo, hi = min(a,t), max(b,t)
                    if not any(i < lo or i > hi for i in phase): continue
                    if lesson == 'masked' and not any(lo <= i < j <= hi for i,j in masked_runs): continue
                    if lesson == 'masked' and not any(i in eligible and (i-a)%d for i in range(lo,hi+1)): continue
                    geometries.append((a,b,t))
        if geometries:
            a,b,t = _pick(rng, geometries)
            off = sorted(i for i in eligible if (i-a) % d)
            noise = []
            if lesson == 'masked':
                near = [i for i in off if min(b,t) < i < max(a,t)]
                noise = [int(_pick(rng, near or off))]
            elif off and rng.random() < .4:
                noise = list(map(int,rng.choice(off,int(rng.integers(1,min(3,len(off))+1)),replace=False)))
            return dict(a=a,b=b,period=d,terminal=t,noise=noise,mode=mode)
    return None


def _alternative(g, info, metric):
    """Explicit weak-rule ablations, for evidence checks rather than solving."""
    alt = dict(info)
    routes = []
    for samples in info['routes']:
        start = samples[0][0]
        routes.append([(p, j if metric == 'visible' else abs(p[0]-start[0])+abs(p[1]-start[1]))
                       for j,(p,_) in enumerate(samples)])
    alt['routes'] = routes
    try:
        alt['signals'] = [_signal(g,samples,alt) for samples in routes]
    except ValueError:
        return None
    return _render(g,alt)


def _different(a, b):
    return a is None or not np.array_equal(a,b)


def _turn_ablation(g):
    try:
        info = _geometry(g, turn_crossings=True)
        info['signals'] = [_signal(g,r,info) for r in info['routes']]
        return _render(g,info)
    except ValueError:
        return None


def _make_pair(rng, n, lesson, wanted_masks, transform, palette):
    field, paths = _make_geometry(rng,n,long_tail=wanted_masks > 0)
    if lesson == 0 and not np.any(field == 2): return None
    if wanted_masks:
        field = _add_masks(rng,field,paths,wanted_masks,mandatory=lesson == 1)
        if field is None: return None
        if len(_geometry(field)['masks']) < wanted_masks: return None
    inp, out = field.copy(), field.copy()
    markers = list(range(4,10))
    shared = int(_pick(rng,markers))
    for k,path in enumerate(paths):
        special = ('boundary' if k == 0 else 'solid' if k == 1 else None) if lesson == 0 else ('masked' if k == 0 else 'solid' if k == 1 else None) if lesson == 1 else None
        mode = ('boundary' if k == 0 else 'terminal') if lesson == 0 else ('terminal' if k == 0 else 'boundary') if lesson == 1 else str(_pick(rng,('boundary','terminal')))
        s = _choose_signal(rng,field,path,mode,special)
        if s is None: return None
        ref = shared if lesson == 0 and k < 2 else int(_pick(rng,markers))
        available = [c for c in markers if c != ref]
        rng.shuffle(available)
        terminal_color = available.pop() if s['terminal'] is not None else None
        inp[path[s['a']]] = inp[path[s['b']]] = ref
        if s['terminal'] is not None: inp[path[s['terminal']]] = terminal_color
        for i,c in zip(s['noise'],available): inp[path[i]] = c
        # Paint from the original complete path, not the graph used by solve.
        for i,p in enumerate(path):
            out[p] = inp[p] if field[p] == 1 else out[p]
        lo,hi,color = (0,len(path)-1,ref) if s['terminal'] is None else (min(s['a'],s['terminal']),max(s['b'],s['terminal']),terminal_color)
        for i,p in enumerate(path):
            if field[p] == 1 and lo <= i <= hi and (i-s['a']) % s['period'] == 0:
                out[p] = color
    lookup = np.asarray(palette,dtype=np.int8)
    inp,out = lookup[inp],lookup[out]
    if transform >= 4: inp,out = inp[:,::-1],out[:,::-1]
    inp,out = np.rot90(inp,transform % 4).copy(),np.rot90(out,transform % 4).copy()
    try:
        g,info = _analyze(inp)
    except ValueError:
        return None
    _require(np.array_equal(_render(g,info),out), "construction/tracer disagreement")
    if lesson == 0 and not _different(_alternative(g,info,'manhattan'),out): return None
    if lesson == 1:
        weak = _alternative(g,info,'visible')
        if weak is None or np.array_equal(weak,out): return None
    return {'input': inp, 'output': out}


def _count(value, name, low, high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(name+' must be an integer')
    if not low <= int(value) <= high: raise ValueError(name+' out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,nq = _count(num_train,'num_train',2,10),_count(num_test,'num_test',1,4)
    if nt <= nq: raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    shift = str(_pick(rng,('count','occlusion','orientation','reference')))
    base_transform = int(rng.integers(8))
    stable_palette = bool(rng.random() < .25)
    palette = list(map(int,rng.permutation(10)))
    lessons = list(map(int,rng.choice(nt,2,replace=False)))
    puzzle,seen = {'train': [],'test': []},set()
    for split,total in (('train',nt),('test',nq)):
        for index in range(total):
            lesson = (0 if index == lessons[0] else 1 if index == lessons[1] else None) if split == 'train' else None
            n = int(rng.integers(2 if lesson is not None else 1,4)) if split == 'train' else 4 if shift == 'count' else int(rng.integers(1,5))
            masks = 0 if lesson == 0 else 1 if lesson == 1 else int(rng.integers(3))
            if split == 'test' and shift == 'occlusion': masks = 2
            transform = ((base_transform+1) % 4 + (4 if base_transform >= 4 else 0)) if split == 'test' and shift == 'orientation' else base_transform if shift == 'orientation' else int(rng.integers(8))
            local_palette = palette if stable_palette else list(map(int,rng.permutation(10)))
            for _ in range(100):
                pair = _make_pair(rng,n,lesson,masks,transform,local_palette)
                if pair is None: continue
                key = _key(pair['input'])
                if key in seen: continue
                if split == 'test' and shift == 'reference':
                    # Query rhythm is read across a bend or an opaque interval.
                    _,info = _analyze(pair['input'])
                    if not any(_reference_crosses_event(s,route,info) for s,route in zip(info['signals'],info['routes'])): continue
                break
            else:
                raise RuntimeError('bounded pair construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(g):
    a = np.asarray(g,dtype=np.int8)
    return a.shape,a.tobytes()


def _reference_crosses_event(s, route, info):
    segment = [(p,i) for p,i in route if s['a'] <= i <= s['b']]
    directions = [(int(np.sign(b[0][0]-a[0][0])),int(np.sign(b[0][1]-a[0][1])))
                  for a,b in zip(segment,segment[1:])]
    return len(set(directions)) > 1 or any(b[1]-a[1] > 1 for a,b in zip(segment,segment[1:]))


def inspect_pair(pair):
    g,info = _analyze(pair['input'])
    bends, lengths, gaps = [],[],[]
    for route in info['routes']:
        dirs = [(int(np.sign(b[0][0]-a[0][0])),int(np.sign(b[0][1]-a[0][1]))) for a,b in zip(route,route[1:])]
        bends.append(sum(a != b for a,b in zip(dirs,dirs[1:])))
        lengths.append(route[-1][1])
        gaps += [b[1]-a[1] for a,b in zip(route,route[1:]) if b[1]-a[1] > 1]
    return dict(shape=g.shape, routes=len(info['routes']), crossings=len(info['crosses']),
        bends=tuple(bends), lengths=tuple(lengths), periods=tuple(s['period'] for s in info['signals']),
        terminals=sum(s['terminal'] is not None for s in info['signals']),
        noise=sum(len(s['distractors']) for s in info['signals']),
        masks=len(info['masks']), mask_shapes=[(r1-r0+1,c1-c0+1) for (r0,r1,c0,c1),_ in info['masks']],
        mask_ports=[2*len(links) for _,links in info['masks']], hidden_distances=gaps,
        roles=(info['background'],info['carrier'],info['cross_color'],info['mask_color']),
        shared_references=len({s['reference'] for s in info['signals']}) < len(info['signals']),
        spanning_references=sum(_reference_crosses_event(s,r,info) for s,r in zip(info['signals'],info['routes'])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen,evidence = set(),set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            g,out = _grid(pair['input']),_grid(pair['output'])
            assert g.shape == out.shape
            assert _key(g) not in seen
            seen.add(_key(g))
            g,info = _analyze(g)
            assert np.array_equal(_render(g,info),out), 'wrong arclength completion'
            assert not np.array_equal(g,out), 'unchanged example'
            editable = {p for r in info['routes'] for p,_ in r} - info['crosses']
            preserve = np.ones(g.shape,dtype=bool)
            for p in editable: preserve[p] = False
            assert np.array_equal(g[preserve],out[preserve]), 'background/bridge/cross corruption'
            for route,s in zip(info['routes'],info['signals']):
                at = {i:p for p,i in route}
                for i,c in s['distractors']: assert out[at[i]] == c, 'off-phase mark changed'
                if split == 'train':
                    if s['period'] == 1: evidence.add('solid')
                    else: evidence.add('dotted')
                    if s['terminal'] is not None:
                        evidence.add('terminal')
                        t,_ = s['terminal']; lo,hi = min(s['a'],t),max(s['b'],t)
                        if any(p not in info['crosses'] and (i-s['a']) % s['period'] == 0 and (i<lo or i>hi) and g[p] == info['carrier'] for p,i in route): evidence.add('stop')
                    else:
                        evidence.add('boundary')
                        if any(i<s['a'] and out[p] != g[p] for p,i in route): evidence.add('before')
                        if any(i>s['b'] and out[p] != g[p] for p,i in route): evidence.add('after')
                    if s['distractors']: evidence.add('noise')
                    if _reference_crosses_event(s,route,info): evidence.add('spanning')
            if split == 'train':
                if info['crosses']: evidence.add('crossing')
                if info['crosses'] and _different(_turn_ablation(g),out): evidence.add('crossing_required')
                if info['masks']: evidence.add('mask')
                if any(len(links) == 1 and links[0][0][1] % 2 != links[0][1][1] % 2
                       for _,links in info['masks']): evidence.add('elbow')
                if len({s['reference'] for s in info['signals']}) < len(info['signals']): evidence.add('shared')
                weak = _alternative(g,info,'visible')
                if weak is not None and not np.array_equal(weak,out): evidence.add('hidden_metric')
                if _different(_alternative(g,info,'manhattan'),out): evidence.add('route_metric')
                # Omitting A: uniformly propagate each route's final label.
                flood = g.copy()
                for route,s in zip(info['routes'],info['signals']):
                    color = s['reference'] if s['terminal'] is None else s['terminal'][1]
                    for p,_ in route:
                        if p not in info['crosses']: flood[p] = color
                if not np.array_equal(flood,out): evidence.add('cadence_required')
    assert {'solid','dotted','terminal','stop','boundary','before','after','noise',
            'spanning','crossing','crossing_required','mask','elbow','shared','hidden_metric','route_metric','cadence_required'} <= evidence, evidence


def self_test():
    """Bounded 42-seed coverage, API counts, symmetry and corruption checks."""
    route_rng = np.random.default_rng(777)
    for target in permutations(range(4)):
        field,paths = _make_geometry(route_rng,4,True,target)
        routes = _geometry(field)['routes']
        actual = {frozenset((r[0][0],r[-1][0])):r[-1][1] for r in routes}
        expected_routes = {frozenset((p[0],p[-1])):len(p)-1 for p in paths}
        assert actual == expected_routes and max(field.shape) <= 30
    # Hand-specified answers exercise an elbow hidden by a rectangle. The two
    # ports are (2,7) and (4,9), four unit steps apart, even though they are
    # consecutive visible samples. An invisible cadence position remains hidden.
    elbow = np.zeros((10,15),dtype=np.int8)
    path = [(2,3)]
    for q in ((2,9),(7,9),(7,12)): _append_line(path,q)
    for p in path: elbow[p] = 8
    elbow[1:4,8:11] = 9
    elbow[2,3] = elbow[2,5] = 2
    elbow[7,11] = 6
    expected = elbow.copy()
    for p in ((2,7),(4,9),(6,9),(7,10),(7,12)): expected[p] = 2
    assert np.array_equal(solve(elbow),expected)
    assert _different(_alternative(elbow,_analyze(elbow)[1],'visible'),expected)
    # Two crossing cables reuse one reference color, but have different periods
    # and completion modes. The intersection is counted and stays a glyph.
    cross = np.zeros((10,13),dtype=np.int8)
    cross[4,1:12] = 8
    cross[1:9,6] = 8
    cross[4,6] = 5
    cross[4,2] = cross[4,4] = cross[1,6] = cross[3,6] = 2
    cross[7,6] = 3
    expected = cross.copy()
    expected[4,8] = expected[4,10] = 2
    expected[1,6] = expected[3,6] = expected[5,6] = 3
    assert np.array_equal(solve(cross),expected)
    assert _different(_turn_ablation(cross),expected)
    metrics,keys = [],set()
    for seed in SELF_TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train','test'):
            for pair in puzzle[split]:
                metrics.append(inspect_pair(pair))
                keys.add(_key(pair['input']))
        # A new presentation/palette must not carry a new rule.
        sample = puzzle['test'][0]
        for turns in (1,2,3):
            a,b = np.rot90(sample['input'],turns),np.rot90(sample['output'],turns)
            assert np.array_equal(solve(a),b)
        assert np.array_equal(solve(9-sample['input']),9-sample['output'])
    assert len(keys) == 210
    assert len({m['shape'] for m in metrics}) >= 40
    assert {m['routes'] for m in metrics} == {1,2,3,4}
    assert len({m['periods'] for m in metrics}) >= 60
    assert {m['masks'] for m in metrics} >= {0,1,2}
    assert any(max(m['mask_ports'],default=0) >= 4 for m in metrics)
    assert any(m['crossings'] >= 6 for m in metrics)
    assert any(len(set(m['lengths'])) > 1 for m in metrics)
    assert len({m['roles'][:2] for m in metrics}) >= 25
    assert any(0 in m['roles'][1:] for m in metrics)
    assert max(max(m['shape']) for m in metrics) == 30
    # All 30 legal explicit combinations, including the smallest lesson set.
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            validate_generated_puzzle(generate_puzzle(112121,nt,nq))
    for counts in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try: generate_puzzle(0,*counts)
        except ValueError: pass
        else: raise AssertionError('illegal counts accepted')
    a,b = generate_puzzle(0,2,1),generate_puzzle(0,2,1)
    assert all(_key(p[k]) == _key(q[k]) for s in a for p,q in zip(a[s],b[s]) for k in ('input','output'))
    damaged = {s:[{k:np.asarray(v).copy() for k,v in p.items()} for p in a[s]] for s in a}
    damaged['test'][0]['output'][0,0] = (int(damaged['test'][0]['output'][0,0])+1) % 10
    try: validate_generated_puzzle(damaged)
    except AssertionError: pass
    else: raise AssertionError('structural corruption accepted')
    return {'seeds':len(SELF_TEST_SEEDS),'pairs':len(metrics),'shapes':len({m['shape'] for m in metrics}),
            'period_signatures':len({m['periods'] for m in metrics}),
            'mask_shapes':len({s for m in metrics for s in m['mask_shapes']}),
            'maximum_crossings':max(m['crossings'] for m in metrics)}
