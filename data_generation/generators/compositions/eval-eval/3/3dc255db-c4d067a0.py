"""Fuel-limited trail printer: an input-only ARC composition of A and B.

Only NumPy and the standard library are used. No files are read or written.
The miniature is a colored induced path; seed squares register its lattice.
The hopper's unary fuel advances one end of the seed along that path.
"""
import hashlib
import numbers

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _components(points, diagonal=False):
    remaining = set(points)
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _neighbors(p, cells):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in _DIRS if (r + dr, c + dc) in cells]


def _local(p, d):
    r, c = p
    dr, dc = d
    return (-dr * r - dc * c, dc * r - dr * c)


def _offset(p, d):
    y, x = p
    dr, dc = d
    return (-dr * y + dc * x, -dc * y - dr * x)


def _read_hopper(frame, d):
    """Recognize monotone thin sides from cross sections, not a stencil."""
    coords = {_local(p, d) for p in frame}
    y0, y1 = min(y for y, x in coords), max(y for y, x in coords)
    if y1 - y0 < 3:
        return None
    rows = [sorted(x for y, x in coords if y == level) for level in range(y0, y1 + 1)]
    if len(rows[0]) != 1:
        return None
    tx = rows[0][0]
    lo = hi = tx
    split = False
    cavity = set()
    for i, row in enumerate(rows[1:], 1):
        if len(row) == 1 and not split and row[0] == tx:
            continue
        if len(row) != 2:
            return None
        a, b = row
        if not (a <= tx <= b and 0 <= lo - a <= 1 and 0 <= b - hi <= 1):
            return None
        lo, hi, split = a, b, True
        for x in range(a + 1, b):
            cavity.add(_offset((y0 + i, x), d))
    if not cavity or not lo < tx < hi:
        return None
    return _offset((y0, tx), d), cavity


def _path_order(cells):
    adj = {p: _neighbors(p, cells) for p in cells}
    ends = sorted(p for p, ns in adj.items() if len(ns) == 1)
    _require(len(ends) == 2 and all(1 <= len(ns) <= 2 for ns in adj.values()), "trail branches or cycles")
    route, previous, current = [], None, ends[0]
    while True:
        route.append(current)
        nxt = [p for p in adj[current] if p != previous]
        if not nxt:
            break
        previous, current = current, nxt[0]
        _require(len(route) <= len(cells), "cyclic trail")
    _require(len(route) == len(cells), "disconnected trail")
    return route


def _parse_candidate(inp, bg, frame, d, tip, cavity):
    h, w = inp.shape
    _require(all(0 <= r < h and 0 <= c < w for r, c in cavity), "cavity outside canvas")
    values = {int(inp[p]) for p in cavity} - {bg}
    _require(len(values) == 1 and int(inp[next(iter(frame))]) not in values, "invalid fuel")
    payload_color = next(iter(values))
    payload = {p for p in cavity if inp[p] == payload_color}
    _require(0 < len(payload) < len(cavity), "hopper must be visibly hollow")
    remaining = set(map(tuple, np.argwhere(inp != bg))) - frame - cavity
    components = _components(remaining)
    swatches, squares = [], []
    for cells in components:
        if len(cells) == 1:
            swatches.extend(cells)
            continue
        rr, cc = zip(*cells)
        r, c, s = min(rr), min(cc), max(rr) - min(rr) + 1
        _require(s >= 2 and max(cc) - c + 1 == s and len(cells) == s * s, "nonsquare seed")
        color = int(inp[r, c])
        _require(np.all(inp[r:r+s, c:c+s] == color), "seed is not solid")
        squares.append((r, c, s, color))
    _require(len(swatches) >= 5 and 2 <= len(squares) <= 4, "missing miniature or seed")
    mr, mc = min(r for r, c in swatches), min(c for r, c in swatches)
    er, ec = max(r for r, c in swatches) + 1, max(c for r, c in swatches) + 1
    _require(all((r - mr) % 2 == (c - mc) % 2 == 0 for r, c in swatches), "invalid miniature pitch")
    plan = inp[mr:er:2, mc:ec:2].copy()
    _require(min(plan.shape) >= 2 and np.count_nonzero(plan != bg) == len(swatches), "invalid miniature bounds")
    cells = set(map(tuple, np.argwhere(plan != bg)))
    route = _path_order(cells)
    axis = 0 if d[0] else 1
    other = 1 - axis
    _require(len({s[other] for s in squares}) == 1, "seed does not follow tip axis")
    squares.sort(key=lambda s: s[axis])
    side = squares[0][2]
    _require(all(s[2] == side for s in squares), "unequal seed scale")
    diffs = np.diff([s[axis] for s in squares])
    stride = int(diffs[0])
    _require(stride > side and np.all(diffs == stride), "invalid seed stride")
    sequence = tuple(s[3] for s in squares)
    length = len(sequence)
    matches = []
    for r, c in sorted(cells):
        sites = [(r + (axis == 0) * k, c + (axis == 1) * k) for k in range(length)]
        if all(p in cells for p in sites) and tuple(int(plan[p]) for p in sites) == sequence:
            matches.append(sites)
    _require(len(matches) == 1, "seed colors do not uniquely register miniature")
    canonical_match = matches[0][0]
    seed_sites = list(matches[0])
    base = (squares[0][0] - seed_sites[0][0] * stride,
            squares[0][1] - seed_sites[0][1] * stride)
    bh, bw = ((n - 1) * stride + side for n in plan.shape)
    _require(0 <= base[0] and 0 <= base[1] and base[0] + bh <= h and base[1] + bw <= w,
             "complete lattice bounds outside canvas")
    # Miniature and all logical tile rectangles are separate representations.
    _require(base[0] + bh <= mr or er <= base[0] or base[1] + bw <= mc or ec <= base[1],
             "miniature overlaps enlarged lattice rectangle")
    fr, fc = zip(*frame)
    _require(max(fr) < mr or er <= min(fr) or max(fc) < mc or ec <= min(fc),
             "miniature overlaps hopper rectangle")
    if d[axis] < 0:
        seed_sites.reverse()
        directed_squares = list(reversed(squares))
    else:
        directed_squares = squares
    first = directed_squares[0]
    _require(first[other] <= tip[other] < first[other] + side, "tip misses seed band")
    face = first[axis] if d[axis] > 0 else first[axis] + side - 1
    distance = (face - tip[axis]) * d[axis]
    _require(2 <= distance <= 4, "hopper is not immediately behind seed")
    for k in range(1, distance):
        _require(inp[tip[0] + k*d[0], tip[1] + k*d[1]] == bg, "blocked hopper mouth")
    if route.index(seed_sites[0]) > route.index(seed_sites[1]):
        route.reverse()
    start = route.index(seed_sites[0])
    _require(route[start:start+length] == seed_sites, "seed is not a trail segment")
    continuation = route[start+length:]
    _require(continuation, "no forward trail to print")
    seed_set = set(seed_sites)
    for r, c in cells:
        y, x = base[0] + r*stride, base[1] + c*stride
        region = inp[y:y+side, x:x+side]
        _require(np.all(region == (plan[r, c] if (r, c) in seed_set else bg)), "occupied unprinted tile")
    expected = inp.copy()
    for p in payload:
        expected[p] = bg
    printed = continuation[:len(payload)]
    for r, c in printed:
        y, x = base[0] + r*stride, base[1] + c*stride
        expected[y:y+side, x:x+side] = plan[r, c]
    # Every point outside the hopper, its cavity, miniature swatches, and seed
    # was already ruled out by the component classification above.
    local_frame = [_local(p, d) for p in frame]
    fy, fx = min(y for y, x in local_frame), min(x for y, x in local_frame)
    profile = tuple(sorted((y-fy, x-fx) for y, x in local_frame))
    chain = seed_sites[-1:] + printed
    turns = sum((chain[i][0]-chain[i-1][0], chain[i][1]-chain[i-1][1]) !=
                (d if i == 1 else (chain[i-1][0]-chain[i-2][0], chain[i-1][1]-chain[i-2][1]))
                for i in range(1, len(chain)))
    meta = dict(background=bg, direction=d, quantity=len(payload), available=len(continuation),
                printed=len(printed), backward=start, turns=turns, plan_shape=plan.shape,
                path_length=len(cells), seed_length=length, match=canonical_match,
                base=base, miniature=(mr, mc), side=side, gutter=stride-side,
                frame_color=int(inp[next(iter(frame))]), payload_color=payload_color,
                profile=profile, components=len(_components(payload)),
                palette=tuple(sorted(set(int(v) for v in plan.flat) - {bg})),
                payload=payload, frame=frame, cavity=cavity, plan=plan,
                continuation=continuation, seed_sites=seed_sites, route=route)
    return expected, meta


def _recover(value):
    inp = _grid(value)
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > inp.size//2, "background must occupy a strict majority")
    accepted = []
    for color in sorted(set(int(v) for v in inp.flat) - {bg}):
        for frame in _components(map(tuple, np.argwhere(inp == color)), diagonal=True):
            if len(frame) < 5:
                continue
            for d in _DIRS:
                geometry = _read_hopper(frame, d)
                if geometry is None:
                    continue
                try:
                    accepted.append(_parse_candidate(inp, bg, frame, d, *geometry))
                except ValueError:
                    pass
    _require(len(accepted) == 1, "input must have exactly one coherent hopper/trail interpretation")
    return accepted[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer background, hopper, fuel, map, scale, registration, and path."""
    return _recover(input_grid)[0]


def _profile(rng, minimum_capacity):
    for _ in range(40):
        a, b = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        stem, tail = int(rng.integers(2)), int(rng.integers(2, 5))
        rows = [(0, 0)] * (stem+1)
        l = r = 0
        while l < a or r < b:
            dl = l < a and (r == b or rng.random() < .85)
            dr = r < b and (l == a or rng.random() < .85)
            if not dl and not dr:
                dl = l < a
                dr = not dl
            l, r = l + int(dl), r + int(dr)
            rows.append((-l, r))
        rows.extend([(-a, b)] * tail)
        frame, cavity = set(), set()
        for y, (l, r) in enumerate(rows):
            frame.update(((y, l), (y, r)))
            cavity.update((y, x) for x in range(l+1, r))
        if len(cavity) > minimum_capacity:
            return frame, cavity
    # Always enough capacity for every supported fuel budget (at most nine).
    return ({(0, 0)} | {(y, -min(2, y)) for y in range(1, 6)} |
            {(y, min(2, y)) for y in range(1, 6)},
            {(y, x) for y in range(1, 6) for x in range(-min(2,y)+1, min(2,y))})


def _sample_path(rng, minimum):
    for _ in range(35):
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        current = (int(rng.integers(h)), int(rng.integers(w)))
        path, used = [current], {current}
        last = None
        target = int(rng.integers(minimum, min(23, h*w) + 1)) if h*w >= minimum else minimum
        while len(path) < target:
            choices = []
            for d in _DIRS:
                p = current[0] + d[0], current[1] + d[1]
                if (0 <= p[0] < h and 0 <= p[1] < w and p not in used and
                        _neighbors(p, used) == [current]):
                    choices.append((d, p))
            if not choices:
                break
            weights = np.array([3 if d == last else 1 for d, p in choices], dtype=float)
            last, current = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
            used.add(current)
            path.append(current)
        if len(path) >= minimum:
            r0, c0 = min(r for r, c in path), min(c for r, c in path)
            result = [(r-r0, c-c0) for r, c in path]
            if min(max(r for r,c in result), max(c for r,c in result)) >= 1:
                return result
    return None


def _fuel(rng, cavity, q, compact):
    points = sorted(cavity)
    if not compact:
        return {points[int(i)] for i in rng.choice(len(points), q, replace=False)}
    selected = {points[int(rng.integers(len(points)))]}
    while len(selected) < q:
        frontier = sorted(p for p in cavity-selected if _neighbors(p, selected))
        if not frontier:
            frontier = sorted(cavity-selected)
        selected.add(frontier[int(rng.integers(len(frontier)))])
    return selected


def _layout(rng, shape, frame_world, cavity_world, side, stride):
    bh, bw = ((n-1)*stride+side for n in shape)
    occupied = frame_world | cavity_world
    minr, minc = min(0, min(r for r,c in occupied)), min(0, min(c for r,c in occupied))
    maxr, maxc = max(bh, max(r for r,c in occupied)+1), max(bw, max(c for r,c in occupied)+1)
    scene = (maxr-minr, maxc-minc)
    mini = tuple(2*n-1 for n in shape)
    options = [axis for axis in (0,1) if scene[axis]+mini[axis]+2 <= 30 and
               max(scene[1-axis], mini[1-axis]) <= 30]
    if not options:
        return None
    axis = int(rng.choice(options))
    other = 1-axis
    gap = int(rng.integers(2, min(4,30-scene[axis]-mini[axis])+1))
    need = [max(scene[k], mini[k]) for k in (0,1)]
    need[axis] = scene[axis]+mini[axis]+gap
    canvas = [int(rng.integers(n,min(30,n+4)+1)) for n in need]
    if rng.random() < .12:
        canvas = [30,30]
    elif rng.random() < .15:
        canvas = [max(canvas)]*2
    origin, anchor = [0,0], [0,0]
    slack = canvas[axis]-need[axis]
    before = int(rng.integers(slack+1))
    if rng.integers(2):
        origin[axis] = before
        anchor[axis] = before+scene[axis]+gap
    else:
        anchor[axis] = before
        origin[axis] = before+mini[axis]+gap
    origin[other] = int(rng.integers(canvas[other]-scene[other]+1))
    anchor[other] = int(rng.integers(canvas[other]-mini[other]+1))
    base = (origin[0]-minr, origin[1]-minc)
    return tuple(canvas), tuple(anchor), base


def _make_pair(rng, q, *, demo=False, fixed=None, avoid_match=None, test=False):
    """Bounded rejection over small in-memory paths and feasible geometry."""
    for attempt in range(1800):
        path = _sample_path(rng, q+5 if test else max(q+4,8))
        if path is None:
            continue
        shape = (max(r for r,c in path)+1, max(c for r,c in path)+1)
        if fixed:
            frame, cavity, side, direction, fc, pc, bg = fixed
        else:
            frame, cavity = _profile(rng, q)
            side, direction = int(rng.choice([2,2,3,3,4])), None
            colors = [int(c) for c in rng.permutation(10)]
            bg, fc, pc = colors[:3]
        gutter = int(rng.choice([1,1,2,3]))
        stride = side+gutter
        options = []
        for reverse in (False,True):
            route = path[::-1] if reverse else path
            for start in range(len(path)-2):
                d = (route[start+1][0]-route[start][0], route[start+1][1]-route[start][1])
                if direction is not None and d != direction:
                    continue
                if demo and start == 0:
                    continue
                for length in (2,3,4):
                    end = start+length
                    if end >= len(path) or any((route[k][0]-route[k-1][0],route[k][1]-route[k-1][1]) != d
                                               for k in range(start+1,min(end,len(path)))):
                        continue
                    available = len(path)-end
                    if (demo or test) and available <= q:
                        continue
                    forward = route[end:end+q]
                    chain = route[end-1:end]+forward
                    turns = sum((chain[k][0]-chain[k-1][0],chain[k][1]-chain[k-1][1]) !=
                                (d if k==1 else (chain[k-1][0]-chain[k-2][0],chain[k-1][1]-chain[k-2][1]))
                                for k in range(1,len(chain)))
                    if (demo or test) and turns == 0:
                        continue
                    axis = 0 if d[0] else 1
                    match = min(route[start:end], key=lambda p:p[axis])
                    if match == avoid_match:
                        continue
                    options.append((route,start,length,d,match))
        rng.shuffle(options)
        for route,start,length,d,match in options[:28]:
            near = route[start]
            lane = int(rng.choice([(side-1)//2,side//2]))
            distance = int(rng.integers(2,5))
            tr, tc = near[0]*stride, near[1]*stride
            if d[0]:
                tr += (-distance if d[0] > 0 else side-1+distance)
                tc += lane
            else:
                tc += (-distance if d[1] > 0 else side-1+distance)
                tr += lane
            tip = (tr,tc)
            def place(p):
                a,b = _offset(p,d)
                return a+tr,b+tc
            fw, cw = {place(p) for p in frame}, {place(p) for p in cavity}
            # Reserve every path square, even dormant sites behind the seed.
            rmin,rmax = min(r for r,c in fw),max(r for r,c in fw)
            cmin,cmax = min(c for r,c in fw),max(c for r,c in fw)
            if any(not (rmax < r*stride-1 or rmin > r*stride+side or
                        cmax < c*stride-1 or cmin > c*stride+side) for r,c in path):
                continue
            layout = _layout(rng,shape,fw,cw,side,stride)
            if layout is None:
                continue
            canvas,anchor,base = layout
            pool = [v for v in range(10) if v not in (bg,fc)]
            if rng.random() < .25:
                pool.append(fc)
            palette = rng.choice(pool,int(rng.integers(3,6)),replace=False)
            seed_sites = route[start:start+length]
            ordered = sorted(seed_sites, key=lambda p:p[0 if d[0] else 1])
            axis = 0 if d[0] else 1
            plan = np.full(shape,bg,dtype=np.int8)
            for paint_attempt in range(24):
                for p,v in zip(path,rng.choice(palette,len(path))):
                    plan[p] = v
                seq = tuple(int(plan[p]) for p in ordered)
                found = []
                for r,c in path:
                    pts = [(r+(axis==0)*k,c+(axis==1)*k) for k in range(length)]
                    if all(p in set(path) for p in pts) and tuple(int(plan[p]) for p in pts)==seq:
                        found.append((r,c))
                if found == [match]:
                    printed_colors = {int(plan[p]) for p in route[start+length:start+length+q]}
                    if not (demo or test) or len(printed_colors) >= 2:
                        break
            else:
                continue
            inp = np.full(canvas,bg,dtype=np.int8)
            ar,ac = anchor
            inp[ar:ar+2*shape[0]:2,ac:ac+2*shape[1]:2] = plan
            def shift(p):
                return p[0]+base[0],p[1]+base[1]
            for p in fw:
                inp[shift(p)] = fc
            fuel = _fuel(rng,cavity,q,compact=demo or rng.random()<.35)
            for p in fuel:
                inp[shift(place(p))] = pc
            for r,c in seed_sites:
                y,x = base[0]+r*stride,base[1]+c*stride
                inp[y:y+side,x:x+side] = plan[r,c]
            if np.count_nonzero(inp==bg) <= inp.size//2:
                continue
            out = inp.copy()
            for p in fuel:
                out[shift(place(p))] = bg
            for r,c in route[start+length:start+length+q]:
                y,x = base[0]+r*stride,base[1]+c*stride
                out[y:y+side,x:x+side] = plan[r,c]
            return {"input":inp,"output":out}
    raise RuntimeError("bounded construction exhausted")


def _count(value, name, low, high):
    _require(isinstance(value,numbers.Integral) and not isinstance(value,(bool,np.bool_)), name+" must be an integer")
    _require(low <= int(value) <= high, name+" outside legal range")
    return int(value)


def _key(a):
    return a.shape,a.astype(np.int8,copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ns = _count(num_train,"num_train",2,10),_count(num_test,"num_test",1,4)
    _require(nt>ns,"num_train must exceed num_test")
    _require(isinstance(seed,numbers.Integral) and not isinstance(seed,(bool,np.bool_)) and seed>=0,"seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    low,high = int(rng.integers(2,4)),int(rng.integers(4,7))
    frame,cavity = _profile(rng,high)
    side = int(rng.choice([2,2,3]))
    direction = _DIRS[int(rng.integers(4))]
    colors = [int(v) for v in rng.permutation(10)]
    fc,pc,bg0,bg1 = colors[:4]
    fixed0 = (frame,cavity,side,direction,fc,pc,bg0)
    fixed1 = (frame,cavity,side,direction,fc,pc,bg1)
    puzzle,seen = {"train":[],"test":[]},set()
    previous_match = None
    for split,count in (("train",nt),("test",ns)):
        for i in range(count):
            for attempt in range(16):
                if split=="train" and i<2:
                    pair = _make_pair(rng,low if i==0 else high,demo=True,
                                      fixed=fixed0 if i==0 else fixed1,avoid_match=previous_match if i else None)
                else:
                    q = int(rng.integers(high+1,high+3)) if split=="test" else int(rng.integers(1,high+1))
                    pair = _make_pair(rng,q,test=split=="test")
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("duplicate exclusion exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split=="train" and i==0:
                previous_match = _recover(pair["input"])[1]["match"]
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle)=={"train","test"},"invalid puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in puzzle),"splits must be lists")
    nt,ns = len(puzzle["train"]),len(puzzle["test"])
    _count(nt,"training count",2,10)
    _count(ns,"test count",1,4)
    _require(nt>ns,"training count must exceed test count")
    seen,metas = set(),{"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair)=={"input","output"},"invalid pair keys")
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            _require(inp.shape==out.shape,"output shape changed")
            key = _key(inp)
            _require(key not in seen,"duplicate input")
            seen.add(key)
            expected,m = _recover(inp)
            _require(np.array_equal(expected,out),"wrong output or changed unrelated pixel")
            _require(not np.array_equal(inp,out),"identity pair")
            # Conservation in logical sites, rather than pixel distance.
            _require(m["printed"]==min(m["quantity"],m["available"]),"incorrect fuel budget")
            _require(np.count_nonzero(inp!=out)==m["quantity"]+m["printed"]*m["side"]**2,
                     "payload removal / square-area accounting failed")
            metas[split].append(m)
    full = [m for m in metas["train"] if m["quantity"]<m["available"]]
    _require(len({m["quantity"] for m in full})>=2,"training permits constant fuel")
    groups = {}
    for m in full:
        signature = (m["profile"],m["direction"],m["frame_color"],m["payload_color"],m["side"])
        groups.setdefault(signature,set()).add(m["quantity"])
    _require(any(len(qs)>=2 for qs in groups.values()),"fuel is confounded with hopper geometry/color/direction/scale")
    _require(any(m["components"]<m["quantity"] for m in full),"cell versus component count not taught")
    _require(len({m["match"] for m in metas["train"]})>=2,"matching offsets do not vary in training")
    _require(any(m["backward"]>0 and m["quantity"]<m["available"] and m["turns"]>0 for m in full),
             "training lacks directed, turning, finite continuation")
    _require(any(len({int(m["plan"][p]) for p in m["continuation"][:m["printed"]]}) >= 2 for m in full),
             "training does not demonstrate literal map colors")
    _require(all(m["quantity"]>max(t["quantity"] for t in metas["train"]) and
                 m["available"]>m["quantity"] and m["turns"]>0 for m in metas["test"]),
             "test shift must extrapolate visible fuel on a turning route")
    return None


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for s in ("train","test"):
        for pair in puzzle[s]:
            for k in ("input","output"):
                shape,data = _key(np.asarray(pair[k]))
                h.update(bytes(shape))
                h.update(data)
    return h.hexdigest()


def self_test():
    """42 streaming seeds, bounded to one puzzle plus small feature sets."""
    fingerprints = set()
    features = {k:set() for k in ("direction","quantity","plan_shape","side","gutter","match",
                                 "path_length","seed_length","profile","background","palette","canvas")}
    clipped = 0
    topology_rejections = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train","test"):
            for pair in puzzle[split]:
                expected,m = _recover(pair["input"])
                assert np.array_equal(expected,pair["output"])
                for k in features:
                    features[k].add(pair["input"].shape if k=="canvas" else m[k])
                clipped += m["quantity"]>m["available"]
        # Recoloring and rotating the entire input commute with the rule.
        p = puzzle["test"][0]
        for k in range(4):
            assert np.array_equal(solve(np.rot90(p["input"],k)),np.rot90(p["output"],k))
        palette = np.random.default_rng(seed+9000).permutation(10)
        assert np.array_equal(solve(palette[p["input"]]),palette[p["output"]])
        # A counterfactual: delete one fuel cell. Exactly the final newly
        # printed square disappears; the new input still has a valid hopper.
        _,m = _recover(p["input"])
        reduced = p["input"].copy()
        removed = min(m["payload"])
        reduced[removed] = m["background"]
        target = p["output"].copy()
        r,c = m["continuation"][m["quantity"]-1]
        stride = m["side"]+m["gutter"]
        y,x = m["base"][0]+r*stride,m["base"][1]+c*stride
        target[y:y+m["side"],x:x+m["side"]] = m["background"]
        assert np.array_equal(solve(reduced),target), "fuel counterfactual failed"
        # B counterfactual: change one printed miniature swatch to an unused
        # literal color. Registration stays unique; its square must change.
        recolored = p["input"].copy()
        new_color = next(v for v in range(10) if v not in set(int(c) for c in recolored.flat))
        r,c = m["continuation"][0]
        sy,sx = m["miniature"][0]+2*r,m["miniature"][1]+2*c
        recolored[sy,sx] = new_color
        target = p["output"].copy()
        target[sy,sx] = new_color
        y,x = m["base"][0]+r*stride,m["base"][1]+c*stride
        target[y:y+m["side"],x:x+m["side"]] = new_color
        assert np.array_equal(solve(recolored),target), "miniature counterfactual failed"
        # A branch or shortcut contact in the miniature is invalid. Check
        # rejection at a real blank site when this sampled map supplies one.
        occupied = set(map(tuple,np.argwhere(m["plan"] != m["background"])))
        for r,c in np.ndindex(m["plan"].shape):
            ns = _neighbors((r,c),occupied)
            if (r,c) in occupied or not ns:
                continue
            if len(ns)==1 and len(_neighbors(ns[0],occupied))==1:
                continue  # Extending an endpoint is a valid family member.
            broken = p["input"].copy()
            broken[m["miniature"][0]+2*r,m["miniature"][1]+2*c] = new_color
            try:
                solve(broken)
            except ValueError:
                topology_rejections += 1
                break
            raise AssertionError("branch/cycle contact was accepted")
    assert len(fingerprints)==42
    for key,minimum in (("direction",4),("quantity",7),("plan_shape",10),("side",3),("gutter",3),
                        ("match",10),("path_length",8),("seed_length",3),("profile",20),
                        ("background",10),("palette",20),("canvas",30)):
        assert len(features[key])>=minimum,(key,len(features[key]))
    assert clipped>0
    assert topology_rejections>0
    assert _fingerprint(generate_puzzle(123,7,3))==_fingerprint(generate_puzzle(123,7,3))
    for nt in range(2,11):
        for ns in range(1,min(4,nt-1)+1):
            p = generate_puzzle(5000+nt*10+ns,nt,ns)
            assert len(p["train"])==nt and len(p["test"])==ns
    for nt,ns in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(1234,2,1)
    damaged["test"][0]["output"][0,0] = (int(damaged["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        return
    raise AssertionError("validator accepted a changed output cell")
