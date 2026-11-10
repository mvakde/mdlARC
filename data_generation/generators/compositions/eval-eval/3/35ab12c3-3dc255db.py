"""Threaded export: a counted, translated prefix of an inferred waypoint route.

No parent module, fixture, seed, or generation record is used by solve().
Only NumPy and the Python standard library are required. Nothing is saved by
generation, validation, or self_test; all diagnostic sets are bounded.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


HOST_SEEDS = (0,750503923,339618460,111406525,911164858,852861509,614578437,
              26637261,277132139,714722583,163810118,59959722,658854777,
              511709754,130715976,296979812,452376293,567434280,230378093,
              851580257,111859261,680276304,190919606,48424571,759222347,
              744733179,27162917,801442322,414790706,262478829,539977006,
              10786852,666160349,585600191,677711753,824020078,485565692,
              280512279,287552025,748977564,882486044,86975664)
STEPS = tuple((r,c) for r in (-1,0,1) for c in (-1,0,1) if r or c)
CARDINAL = ((-1,0),(0,1),(1,0),(0,-1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _add(a,b):
    return a[0]+b[0], a[1]+b[1]


def _sub(a,b):
    return a[0]-b[0], a[1]-b[1]


def _sign(x):
    return int(x > 0)-int(x < 0)


def _line(a,b):
    dr,dc = _sub(b,a)
    n = max(abs(dr),abs(dc))
    if not n or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return ()
    return tuple((a[0]+i*_sign(dr),a[1]+i*_sign(dc)) for i in range(n+1))


def _cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _intersects(a,b,c,d):
    if (max(a[0],b[0]) < min(c[0],d[0]) or max(c[0],d[0]) < min(a[0],b[0])
            or max(a[1],b[1]) < min(c[1],d[1]) or max(c[1],d[1]) < min(a[1],b[1])):
        return False
    return _cross(a,b,c)*_cross(a,b,d) <= 0 and _cross(c,d,a)*_cross(c,d,b) <= 0


def _infer_route(points,start,direction):
    """Unique simple full path, docked at start and initially pointing outward.

    Candidate edges cannot jump over another waypoint. Search rejects continuous
    crossings, including half-cell crossings of two diagonal segments. No
    shortest-path or nearest-neighbor tie-break is hidden in the rule.
    """
    points = sorted(points)
    if not 2 <= len(points) <= 9 or start not in points:
        raise ValueError("a route needs 2..9 waypoints including the dock")
    point_set = set(points)
    edges, adjacency = [], {p:[] for p in points}
    for i,a in enumerate(points):
        for b in points[i+1:]:
            line = _line(a,b)
            if line and not set(line[1:-1]) & point_set:
                edges.append((a,b))
                adjacency[a].append(b)
                adjacency[b].append(a)
    solutions, visits = [], 0

    def visit(order,selected):
        nonlocal visits
        visits += 1
        if visits > 20000:
            raise ValueError("route search limit exceeded")
        if len(solutions) > 1:
            return
        if len(order) == len(points):
            solutions.append(order)
            return
        a = order[-1]
        for b in adjacency[a]:
            if b in order:
                continue
            vector = (_sign(b[0]-a[0]),_sign(b[1]-a[1]))
            if len(order) == 1 and vector != direction:
                continue
            if any(_intersects(a,b,c,d) for c,d in selected if a not in (c,d)):
                continue
            visit(order+[b],selected+[(a,b)])

    visit([start],[])
    if len(solutions) != 1:
        raise ValueError("the dock must determine one noncrossing full route")
    order = solutions[0]
    route = [start]
    for a,b in zip(order,order[1:]):
        route.extend(_line(a,b)[1:])
    if len(route) != len(set(route)):
        raise ValueError("route revisits a cell")
    return order,route,len(edges)-(len(points)-1)


def _canonical(p,d):
    # y increases into the hopper; x is lateral to its outward direction d.
    r,c = p
    dr,dc = d
    return -dr*r-dc*c, dc*r-dr*c


def _world(p,d):
    y,x = p
    dr,dc = d
    return -dr*y+dc*x, -dc*y-dr*x


def _hopper_geometry(frame,d):
    local = {_canonical(p,d) for p in frame}
    y0,y1 = min(y for y,x in local),max(y for y,x in local)
    if y1-y0 < 3:
        raise ValueError("hopper is too shallow")
    first = sorted(x for y,x in local if y == y0)
    if len(first) != 1:
        raise ValueError("tip is not unique")
    tip_x = first[0]
    previous = (tip_x,tip_x)
    cavity = set()
    for y in range(y0+1,y1+1):
        row = sorted(x for yy,x in local if yy == y)
        if not row or len(row) > 2:
            raise ValueError("hopper sides are incomplete")
        lo,hi = row[0],row[-1]
        if not (0 <= previous[0]-lo <= 1 and 0 <= hi-previous[1] <= 1):
            raise ValueError("hopper sides must widen monotonically")
        cavity.update(_world((y,x),d) for x in range(lo+1,hi))
        previous = lo,hi
    if not previous[0] < tip_x < previous[1] or not cavity:
        raise ValueError("no readable hollow")
    return _world((y0,tip_x),d),cavity


def _inside_grid(points,shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r,c in points)


def _read(input_grid):
    a = _grid(input_grid)
    colors,counts = np.unique(a,return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if len(colors) != 4 or int(max(counts))*2 <= a.size:
        raise ValueError("four local color roles and a majority background are required")
    groups = {int(c):set(map(tuple,np.argwhere(a == c).tolist())) for c in colors if c != bg}
    candidates = []
    for fc,frame in groups.items():
        for d in CARDINAL:
            try:
                tip,cavity = _hopper_geometry(frame,d)
                if not _inside_grid(cavity,a.shape):
                    continue
                values = {int(a[p]) for p in cavity}-{bg}
                if len(values) != 1 or fc in values:
                    continue
                pc = next(iter(values))
                payload = groups[pc] & cavity
                exterior = groups[pc]-cavity
                if not payload or len(exterior) != 1:
                    continue
                seed = next(iter(exterior))
                tc = next(c for c in groups if c not in (fc,pc))
                start = _add(tip,d)
                offset = _sub(seed,start)
                if offset not in STEPS:
                    continue
                waypoints = groups[tc]
                order,route,chords = _infer_route(waypoints,start,d)
                source = set(route)
                copy = [_add(p,offset) for p in route]
                if (seed in source or source & (frame|cavity)
                        or set(copy) & (frame|cavity)
                        or not _inside_grid(copy,a.shape)):
                    continue
                # These four roles account for every foreground cell.
                if any(a[p] not in (bg,tc) for p in route):
                    continue
                candidates.append(dict(grid=a,background=bg,frame_color=fc,
                    track_color=tc,payload_color=pc,frame=frame,cavity=cavity,
                    tip=tip,direction=d,payload=payload,quantity=len(payload),
                    seed=seed,offset=offset,waypoints=waypoints,order=order,
                    route=route,copy=copy,chords=chords))
            except (ValueError,StopIteration,KeyError):
                continue
    if len(candidates) != 1:
        raise ValueError("input does not specify one hopper and one unambiguous docked route")
    return candidates[0]


def _paint(info,mode="correct"):
    a = info["grid"]
    out = a.copy()
    bg,tc,pc = info["background"],info["track_color"],info["payload_color"]
    for p in info["payload"]:
        out[p] = bg
    source = set(info["route"])
    for p in source:
        out[p] = tc
    q = info["quantity"]
    if mode == "entire":
        cells = info["copy"][1:]
    elif mode == "straight":
        d,s = info["direction"],info["seed"]
        cells = [(s[0]+i*d[0],s[1]+i*d[1]) for i in range(1,q+1)]
    elif mode == "components":
        cells = info["copy"][1:1+_component_count(info["payload"])]
    elif mode == "visible_only":
        cells = [p for p in info["copy"][1:] if p not in source][:q]
    elif mode == "skip_waypoints":
        cells = [_add(p,info["offset"]) for p in info["route"][1:]
                 if p not in info["waypoints"]][:q]
    elif mode == "include_seed":
        cells = info["copy"][1:q]
    else:
        cells = info["copy"][1:1+q]
    for p in cells:
        if (_inside_grid([p],a.shape) and p not in source
                and a[p] == bg):
            out[p] = pc
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all roles and geometry from the input and apply the fixed rule."""
    return _paint(_read(input_grid))


def _component_count(points):
    remaining = set(points)
    count = 0
    while remaining:
        stack = [remaining.pop()]
        count += 1
        while stack:
            p = stack.pop()
            for step in STEPS:
                q = _add(p,step)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
    return count


def _profile(rng):
    left,right = int(rng.integers(2,5)),int(rng.integers(2,5))
    stem,tail = int(rng.integers(0,3)),int(rng.integers(4,7))
    frame = {(y,0) for y in range(stem+1)}
    cavity = set()
    lo = hi = 0
    y = stem
    while -lo < left or hi < right:
        y += 1
        # Independent sides produce off-center tips and delayed slope steps.
        dl = -lo < left and (hi == right or rng.random() < .8)
        dr = hi < right and (-lo == left or rng.random() < .8)
        if not dl and not dr:
            dl = -lo < left
            dr = not dl and hi < right
        lo -= int(dl)
        hi += int(dr)
        frame.update(((y,lo),(y,hi)))
        cavity.update((y,x) for x in range(lo+1,hi))
    for _ in range(tail):
        y += 1
        frame.update(((y,lo),(y,hi)))
        cavity.update((y,x) for x in range(lo+1,hi))
    return frame,cavity


def _path_from_order(vertices):
    result = [vertices[0]]
    for a,b in zip(vertices,vertices[1:]):
        result.extend(_line(a,b)[1:])
    return result


def _safe_hopper():
    frame = {(0,0),(1,-1),(1,1)}|{(y,x) for y in range(2,9) for x in (-2,2)}
    cavity = {(1,0)}|{(y,x) for y in range(2,9) for x in (-1,0,1)}
    return frame,cavity


def _staircase(n,diagonal):
    # Cardinal legs have unequal aspect ratio (3 vertical : 1 horizontal),
    # preventing the diagonal chords of an equal-step staircase. A diagonal
    # staircase uses only N and NE: any mixed span is strictly steeper than NE.
    # Thus every candidate edge is an actual consecutive edge in either case.
    vertices = [(-1,0)]
    for i in range(n-1):
        d = (-1,0) if i % 2 == 0 else ((-1,1) if diagonal else (0,1))
        length = 2 if diagonal else (3 if i % 2 == 0 else 1)
        vertices.append((vertices[-1][0]+d[0]*length,vertices[-1][1]+d[1]*length))
    return vertices


def _anchor(rng):
    for _ in range(80):
        a,b,c,e = (int(rng.integers(2,5)),int(rng.integers(2,6)),
                   int(rng.integers(3,6)),int(rng.integers(2,5)))
        side = int(rng.choice([-1,1]))
        vertices = [(-1,0),(-1-a,0),(-1-a,side*b),(-1-a-c,side*b),
                    (-1-a-c,0),(-1-a-c-e,0)]
        try:
            order,route,chords = _infer_route(vertices,(-1,0),(-1,0))
        except ValueError:
            continue
        if order != vertices or not chords:
            continue
        frame,cavity = _profile(rng)
        low,high = a+b+1,a+b+c
        if high >= len(cavity):
            continue
        return dict(vertices=vertices,frame=frame,cavity=cavity,offset=(0,side),
                    low=low,high=high)
    frame,cavity = _safe_hopper()
    return dict(vertices=[(-1,0),(-5,0),(-5,4),(-9,4),(-9,0),(-11,0)],
                frame=frame,cavity=cavity,offset=(0,1),low=9,high=12)


def _random_route(rng,n,diagonal,short=False):
    steps = STEPS if diagonal else CARDINAL
    for _ in range(240):
        vertices = [(-1,0)]
        previous = None
        for i in range(n-1):
            choices = [s for s in steps if s != previous and
                       (previous is None or s != (-previous[0],-previous[1]))]
            d = (-1,0) if i == 0 else choices[int(rng.integers(len(choices)))]
            length = int(rng.integers(1,4 if short else 6))
            vertices.append((vertices[-1][0]+length*d[0],vertices[-1][1]+length*d[1]))
            previous = d
        if (len(set(vertices)) != n or any(not (-18 <= y <= -1 and -11 <= x <= 11)
                                          for y,x in vertices)):
            continue
        try:
            order,route,_ = _infer_route(vertices,(-1,0),(-1,0))
        except ValueError:
            continue
        if order != vertices or (diagonal and not any(
                a[0] != b[0] and a[1] != b[1] for a,b in zip(vertices,vertices[1:]))):
            continue
        if all(len(_line(a,b)) == 2 for a,b in zip(vertices,vertices[1:])):
            continue
        if short and len(route) > 17:
            continue
        return vertices
    vertices = _staircase(n,diagonal)
    assert _infer_route(vertices,(-1,0),(-1,0))[0] == vertices
    return vertices


def _payload(rng,cavity,q,pattern):
    cells = sorted(cavity)
    if pattern == "random":
        return {cells[int(i)] for i in rng.choice(len(cells),q,replace=False)}
    chosen = {cells[int(rng.integers(len(cells)))]}
    while len(chosen) < q:
        available = sorted(cavity-chosen)
        if pattern == "compact":
            candidates = [p for p in available if any(
                abs(p[0]-s[0])+abs(p[1]-s[1]) == 1 for s in chosen)]
            candidates = candidates or available
        else:
            distances = [min(max(abs(p[0]-s[0]),abs(p[1]-s[1])) for s in chosen)
                         for p in available]
            best = max(distances)
            candidates = [p for p,d in zip(available,distances) if d == best]
        chosen.add(candidates[int(rng.integers(len(candidates)))])
    return chosen


def _render(rng,vertices,frame,cavity,offset,q,pattern,colors,direction):
    route = _path_from_order(vertices)
    source = set(route)
    companion = [_add(p,offset) for p in route]
    payload = _payload(rng,cavity,q,pattern)
    # Full translated geometry, not only the fueled part, must fit. There are
    # no seed-dependent edge extrapolation or collision choices.
    footprint = source|set(companion)|frame|cavity
    mapped = {_world(p,direction) for p in footprint}
    r0,c0 = min(r for r,c in mapped),min(c for r,c in mapped)
    h,w = max(r for r,c in mapped)-r0+1,max(c for r,c in mapped)-c0+1
    if max(h,w) > 30:
        raise ValueError("scene cannot fit")
    er,ec = int(rng.integers(0,min(7,31-h))),int(rng.integers(0,min(7,31-w)))
    before_r,before_c = int(rng.integers(er+1)),int(rng.integers(ec+1))
    def move(p):
        r,c = _world(p,direction)
        return r-r0+before_r,c-c0+before_c
    bg,fc,tc,pc = colors
    inp = np.full((h+er,w+ec),bg,dtype=np.int8)
    out = inp.copy()
    for p in frame:
        inp[move(p)] = out[move(p)] = fc
    for p in vertices:
        inp[move(p)] = tc
    for p in source:
        out[move(p)] = tc
    for p in payload:
        inp[move(p)] = pc
    seed = companion[0]
    inp[move(seed)] = out[move(seed)] = pc
    for p in companion[1:1+q]:
        if p not in source:
            out[move(p)] = pc
    return {"input":inp,"output":out}


def _fallback_pair(rng,n,diagonal,exhaust,direction=None,offset_hint=None):
    vertices = _staircase(n,diagonal)
    frame,cavity = _safe_hopper()
    length = len(_path_from_order(vertices))-1
    q = length+2 if exhaust else min(5,length-1)
    offset = offset_hint if offset_hint is not None else (0,int(rng.choice([-1,1])))
    colors = tuple(int(x) for x in rng.choice(10,4,replace=False))
    d = direction if direction is not None else CARDINAL[int(rng.integers(4))]
    return _render(rng,vertices,frame,cavity,offset,q,"compact",colors,d)


def _ordinary_pair(rng,n,diagonal,exhaust=False,direction=None,offset_hint=None,require_turn=False):
    for _ in range(60):
        vertices = _random_route(rng,n,diagonal,short=exhaust)
        route = _path_from_order(vertices)
        frame,cavity = _profile(rng)
        source = set(route)
        choices = []
        for offset in STEPS:
            copied = {_add(p,offset) for p in route}
            if (_add(vertices[0],offset) not in source and
                    not copied & (frame|cavity)):
                choices.append(offset)
        if offset_hint is not None:
            choices = [x for x in choices if x == offset_hint]
        if not choices:
            continue
        offset = choices[int(rng.integers(len(choices)))]
        length = len(route)-1
        if exhaust:
            if length >= min(22,len(cavity)):
                continue
            q = int(rng.integers(length+1,min(23,len(cavity)+1)))
        else:
            upper = min(20,len(cavity)-1,length-1)
            if upper < 2:
                continue
            # Usually cross at least the first bend, while retaining small
            # counts on some ordinary demonstrations.
            first = len(_line(vertices[0],vertices[1]))-1
            low = min(upper,first+1) if rng.random() < .8 else 1
            q = int(rng.integers(low,upper+1))
        colors = tuple(int(x) for x in rng.choice(10,4,replace=False))
        d = direction if direction is not None else CARDINAL[int(rng.integers(4))]
        try:
            pair = _render(rng,vertices,frame,cavity,offset,q,
                           str(rng.choice(["compact","spread","random"])),colors,d)
            info = _read(pair["input"])
        except ValueError:
            continue
        if require_turn and np.array_equal(pair["output"],_paint(info,"straight")):
            continue
        if require_turn and not exhaust and np.array_equal(pair["output"],_paint(info,"entire")):
            continue
        if (any(pair["input"][p] == info["background"] for p in info["route"])
                and np.array_equal(pair["output"],_paint(info)) and any(
                pair["input"][p] == info["background"] and
                pair["output"][p] == info["payload_color"] for p in info["copy"][1:])):
            return pair
    return _fallback_pair(rng,n,diagonal,exhaust,direction,offset_hint)


def _counts(num_train,num_test):
    for name,value,lo,hi in (("num_train",num_train,2,10),("num_test",num_test,1,4)):
        if (isinstance(value,(bool,np.bool_)) or
                not isinstance(value,numbers.Integral) or not lo <= int(value) <= hi):
            raise ValueError(f"{name} is outside its legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train),int(num_test)


def _fingerprint(a):
    return a.shape,a.tobytes()


def generate_puzzle(seed: int,num_train: int = 4,num_test: int = 1) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,numbers.Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    nt,nq = _counts(num_train,num_test)
    anchor = _anchor(rng)
    fc,tc,pc = (int(x) for x in rng.choice(10,3,replace=False))
    backgrounds = [c for c in range(10) if c not in (fc,tc,pc)]
    b0,b1 = (int(x) for x in rng.choice(backgrounds,2,replace=False))
    d = CARDINAL[int(rng.integers(4))]
    puzzle = {"train":[],"test":[]}
    for q,bg,pattern in ((anchor["low"],b0,"compact"),(anchor["high"],b1,"spread")):
        puzzle["train"].append(_render(rng,anchor["vertices"],anchor["frame"],
            anchor["cavity"],anchor["offset"],q,pattern,(bg,fc,tc,pc),d))
    # Shuffle the matched count witnesses so the answer is not an index rule.
    if rng.random() < .5:
        puzzle["train"].reverse()
    seen = {_fingerprint(p["input"]) for p in puzzle["train"]}
    for split,count,start_index in (("train",nt,2),("test",nq,0)):
        for i in range(start_index,count):
            for _ in range(20):
                if split == "train":
                    n = int(rng.integers(3,6 if i == 2 else 7))
                    pair = _ordinary_pair(rng,n,diagonal=True,exhaust=i == 2)
                else:
                    n = int(rng.integers(7,10))
                    # The minimal training set has cardinal routes and one
                    # local offset; hold that offset fixed in tip coordinates.
                    pair = _ordinary_pair(rng,n,diagonal=nt >= 3,
                        exhaust=nt >= 3 and i > 0 and rng.random() < .35,
                        offset_hint=anchor["offset"] if nt == 2 else None,require_turn=True)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _geometry_signature(info):
    d,tip = info["direction"],info["tip"]
    norm = lambda points: tuple(sorted(_canonical(_sub(p,tip),d) for p in points))
    return norm(info["frame"]),norm(info["waypoints"]),_canonical(info["offset"],d)


def _validate_pair(pair):
    assert isinstance(pair,dict) and set(pair) == {"input","output"}
    inp,out = _grid(pair["input"]),_grid(pair["output"])
    assert inp.shape == out.shape
    info = _read(inp)
    assert np.array_equal(out,_paint(info)), "wrong threaded export"
    assert np.any(inp != out)
    bg,pc,tc = info["background"],info["payload_color"],info["track_color"]
    # Conservation before clipping/occlusion: each fuel cell selects one
    # successive translated position, including original waypoint positions.
    q,length = info["quantity"],len(info["route"])-1
    emitted = info["copy"][1:1+q]
    visible = set(emitted)-set(info["route"])
    assert len(emitted) == min(q,length)
    assert all(out[p] == bg for p in info["payload"])
    assert all(out[p] == info["frame_color"] for p in info["frame"])
    assert all(out[p] == tc for p in info["route"])
    assert out[info["seed"]] == pc
    expected_pc = visible|{info["seed"]}
    assert set(map(tuple,np.argwhere(out == pc).tolist())) == expected_pc
    assert visible and any(inp[p] == bg and out[p] == tc for p in info["route"])
    assert np.count_nonzero(out == pc) == 1+min(q,length)-len(set(emitted)&set(info["route"]))
    info["visible"] = len(visible)
    info["length"] = length
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    nt,nq = len(puzzle["train"]),len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen,infos = set(),{"train":[],"test":[]}
    for split in puzzle:
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            key = _fingerprint(info["grid"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    # Quantitative evidence on unchanged geometry, palette and offset.
    matched = {}
    for info in train:
        if info["quantity"] < info["length"]:
            key = (_geometry_signature(info),info["direction"],info["frame_color"],
                   info["track_color"],info["payload_color"])
            matched.setdefault(key,set()).add(info["quantity"])
    assert any(len(qs) >= 2 for qs in matched.values()), "no matched count witnesses"
    assert any(_component_count(i["payload"]) < i["quantity"] for i in train)
    assert any(i["chords"] for i in train), "no rejected-alignment evidence"
    def extra_chord_cells(i):
        points = i["waypoints"]
        candidate_pixels = set()
        for a in points:
            for b in points:
                line = _line(a,b)
                if line and not set(line[1:-1]) & points:
                    candidate_pixels.update(line)
        return candidate_pixels-set(i["route"])
    assert any(extra_chord_cells(i) for i in train), "all-alignment shortcut is untested"
    assert any(set(i["copy"][1:1+i["quantity"]]) &
               (set(i["route"])-i["waypoints"]) for i in train), "priority is unobserved"
    for mode in ("straight","entire","components","visible_only","skip_waypoints","include_seed"):
        assert any(not np.array_equal(_paint(i,mode),_paint(i)) for i in train), mode
    train_exhaust = any(i["quantity"] > i["length"] for i in train)
    if any(i["quantity"] > i["length"] for i in infos["test"]):
        assert train_exhaust, "route exhaustion was not demonstrated"
    diagonal = lambda i: any(a[0] != b[0] and a[1] != b[1] for a,b in zip(i["order"],i["order"][1:]))
    if any(diagonal(i) for i in infos["test"]):
        assert any(diagonal(i) for i in train), "diagonal stepping was not demonstrated"
    assert max(len(i["waypoints"]) for i in train) <= 6
    assert all(7 <= len(i["waypoints"]) <= 9 for i in infos["test"])
    assert infos["test"][0]["quantity"] < infos["test"][0]["length"]
    assert all(not np.array_equal(_paint(i,"straight"),_paint(i)) for i in infos["test"])
    assert not np.array_equal(_paint(infos["test"][0],"entire"),_paint(infos["test"][0]))


def self_test():
    """Bounded 42-seed coverage, ablations, count checks and metamorphisms.

    Coverage retains one puzzle at a time, hashes and small feature sets; later
    mutation checks retain a few bounded copies. No grid archive,
    JSON, log, execution trace, subprocess, network or file I/O is created.
    """
    shapes,profiles,paths,offsets,directions,backgrounds,quantities = (set() for _ in range(7))
    signatures,clipped,diagonals,unit_links = set(),0,0,0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train","test"):
            for pair in puzzle[split]:
                i = _validate_pair(pair)
                digest.update(bytes(i["grid"].shape))
                digest.update(i["grid"].tobytes())
                shapes.add(i["grid"].shape)
                profile,path,offset = _geometry_signature(i)
                profiles.add(profile)
                paths.add(path)
                offsets.add(i["offset"])
                directions.add(i["direction"])
                backgrounds.add(i["background"])
                quantities.add(i["quantity"])
                clipped += i["quantity"] > i["length"]
                for a,b in zip(i["order"],i["order"][1:]):
                    diagonals += a[0] != b[0] and a[1] != b[1]
                    unit_links += max(abs(a[0]-b[0]),abs(a[1]-b[1])) == 1
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 60
    assert len(profiles) >= 80 and len(paths) >= 100
    assert offsets == set(STEPS) and directions == set(CARDINAL)
    assert backgrounds == set(range(10)) and len(quantities) >= 12
    assert clipped >= 42 and diagonals >= 42 and unit_links >= 42
    # Exercise constructive fallbacks directly, including their input-derived
    # uniqueness, counts, bounds and query ablations, without a rare RNG event.
    rng = np.random.default_rng(1234)
    for diagonal in (False,True):
        for n in range(3,10):
            points = _staircase(n,diagonal)
            order,_,chords = _infer_route(points,points[0],(-1,0))
            assert order == points and chords == 0
            for exhaust in (False,True):
                for offset in ((0,-1),(0,1)):
                    i = _validate_pair(_fallback_pair(rng,n,diagonal,exhaust,offset_hint=offset))
                    if n >= 7:
                        assert not np.array_equal(_paint(i),_paint(i,"straight"))
                        if not exhaust:
                            assert not np.array_equal(_paint(i),_paint(i,"entire"))
    # Every legal count combination, with no retained corpus.
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(173+nt*11+nq,nt,nq)
            assert (len(p["train"]),len(p["test"])) == (nt,nq)
    for nt,nq in ((1,1),(11,1),(4,0),(4,5),(2,2),(3,4),
                  (True,1),(4,False),(3.0,1),(3,1.0),(-1,1),(4,-1),
                  (None,1),(4,None),(None,None),("4",1),(4,"1")):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1,True,1.5,"0",None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    original = generate_puzzle(0)
    repeat = generate_puzzle(0)
    assert all(np.array_equal(a[k],b[k]) for s in original
               for a,b in zip(original[s],repeat[s]) for k in a)
    # Rotation/reflection and arbitrary color permutations commute with solve.
    rng = np.random.default_rng(987)
    for pair in original["train"]+original["test"]:
        for turns in range(4):
            for reflected in (False,True):
                a,b = np.rot90(pair["input"],turns),np.rot90(pair["output"],turns)
                if reflected:
                    a,b = np.fliplr(a),np.fliplr(b)
                palette = rng.permutation(10)
                assert np.array_equal(solve(palette[a]),palette[b])
    bad = {s:[{k:v.copy() for k,v in p.items()} for p in original[s]] for s in original}
    info = _read(bad["test"][0]["input"])
    p = info["route"][1]
    bad["test"][0]["output"][p] = info["payload_color"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("invalid source priority accepted")
    return {"seeds":42,"distinct_puzzles":len(signatures),"shapes":len(shapes),
            "hopper_profiles":len(profiles),"route_geometries":len(paths),
            "offsets":len(offsets),"directions":len(directions),
            "quantities":sorted(quantities),"exhausted_routes":clipped,
            "diagonal_links":diagonals,"unit_links":unit_links}
