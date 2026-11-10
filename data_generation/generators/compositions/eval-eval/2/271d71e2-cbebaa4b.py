"""Docking activation wave: geometric dependencies gate fuel-limited movement.

No imports from parents and no saved examples, seed state, or solver cache.
All construction retries are bounded. Generation and tests write no files.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

BG, PIN, NOSE = 0, 2, 5
DIRS = ("U", "R", "D", "L")
VEC = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}
OPP = {"U": "D", "D": "U", "L": "R", "R": "L"}
PALETTE = (1, 3, 4, 6, 7, 8, 9)
HOST_SEEDS = (0,788199361,566978585,3959319,416203668,224738641,347059756,174538462,978840135,851668869,828455989,973113905,29627669,899110103,615790292,792412398,221292963,745238873,574985198,580244536,815635733,16373040,940545392,656029632,986844218,834818861,994245742,757693170,817380978,589826535,648676083,528545462,194748591,252106435,569433573,708928820,423938112,359731915,460241136,444535673,267777847,429610630)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _need(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _components(cells):
    left = set(cells)
    result = []
    while left:
        q = min(left)
        left.remove(q)
        comp, stack = {q}, [q]
        while stack:
            y, x = stack.pop()
            for dy, dx in VEC.values():
                p = y + dy, x + dx
                if p in left:
                    left.remove(p)
                    comp.add(p)
                    stack.append(p)
        result.append(comp)
    return result


def _bbox(cells):
    ys, xs = zip(*cells)
    return min(ys), min(xs), max(ys), max(xs)


def _rect(t, l, h, w):
    return {(y, x) for y in range(t, t+h) for x in range(l, l+w)}


def _shift(cells, d):
    return {(y+d[0], x+d[1]) for y, x in cells}


def _norm(cells):
    t, l, _, _ = _bbox(cells)
    return tuple(sorted((y-t, x-l) for y, x in cells))


def _dilate(cells):
    return {(y+dy, x+dx) for y, x in cells for dy in (-1,0,1) for dx in (-1,0,1)}


def _counter(t, l, h, w, direction):
    if direction == "U":
        return [(y,x) for y in range(t+1,t+h-1) for x in range(l+1,l+w-1)]
    if direction == "D":
        return [(y,x) for y in range(t+h-2,t,-1) for x in range(l+w-2,l,-1)]
    if direction == "L":
        return [(y,x) for x in range(l+1,l+w-1) for y in range(t+h-2,t,-1)]
    return [(y,x) for x in range(l+w-2,l,-1) for y in range(t+1,t+h-1)]


def _nose(node):
    t,l,h,w = (node[k] for k in ("t","l","h","w"))
    return {"U":(t,l+w//2), "D":(t+h-1,l+w//2),
            "L":(t+h//2,l), "R":(t+h//2,l+w-1)}[node["direction"]]


def _port(node, side, start, pattern):
    t,l,h,w = (node[k] for k in ("t","l","h","w"))
    if side == "U":
        return {(t-1,l+start+i) for i in pattern}
    if side == "D":
        return {(t+h,l+start+i) for i in pattern}
    if side == "L":
        return {(t+start+i,l-1) for i in pattern}
    return {(t+start+i,l+w) for i in pattern}


def _cells(node):
    return _rect(node["t"],node["l"],node["h"],node["w"]) | set().union(*node["ports"].values())


def _ancestors(parents, i):
    result = set()
    while parents[i] is not None:
        i = parents[i]
        _need(i not in result, "cycle")
        result.add(i)
    return result


def _path(delta, direction):
    """Offsets from the input: tangent alignment, then forward approach."""
    dy,dx = delta
    vy,vx = VEC[direction]
    normal = dy*vy+dx*vx
    _need(normal > 0, "dock must be ahead of the incoming face")
    steps = ((0,dx),(dy,0)) if vy else ((dy,0),(0,dx))
    offsets = [(0,0)]
    y=x=0
    for a,b in steps:
        sy,sx = int(np.sign(a)),int(np.sign(b))
        for _ in range(abs(a)+abs(b)):
            y,x = y+sy,x+sx
            offsets.append((y,x))
    return offsets


def _parse(input_grid):
    g = _grid(input_grid)
    comps = _components(map(tuple,np.argwhere(g != BG).tolist()))
    nodes = []
    for comp in comps:
        body = {p for p in comp if g[p] != PIN}
        _need(bool(body), "unattached key")
        t,l,b,r = _bbox(body)
        h,w = b-t+1,r-l+1
        _need(h >= 3 and w >= 3 and body == _rect(t,l,h,w), "body must be a filled rectangle")
        frame = int(g[t,l])
        _need(frame not in (BG,PIN,NOSE), "invalid frame")
        ports = {}
        for y,x in comp-body:
            if y == t-1 and l <= x <= r:
                side = "U"
            elif y == b+1 and l <= x <= r:
                side = "D"
            elif x == l-1 and t <= y <= b:
                side = "L"
            elif x == r+1 and t <= y <= b:
                side = "R"
            else:
                raise AssertionError("key is not immediately beside a face")
            ports.setdefault(side,set()).add((y,x))
        _need(ports and all(2 <= len(p) <= 3 for p in ports.values()), "invalid key size")
        noses = [p for p in body if g[p] == NOSE]
        _need(len(noses) <= 1, "multiple noses")
        direction = None
        if noses:
            y,x = noses[0]
            possibilities = [d for d,yes in (("U",y==t and l<x<r),("D",y==b and l<x<r),
                                             ("L",x==l and t<y<b),("R",x==r and t<y<b)) if yes]
            _need(len(possibilities) == 1, "nose must mark one noncorner border cell")
            direction = possibilities[0]
            _need(direction in ports, "nose lacks incoming key")
        border = body-_rect(t+1,l+1,h-2,w-2)
        _need(all(g[p] == (NOSE if p in noses else frame) for p in border), "broken frame")
        nodes.append(dict(t=t,l=l,h=h,w=w,frame=frame,ports=ports,direction=direction,
                          body=body,cells=comp,noses=noses))
    roots = [i for i,n in enumerate(nodes) if n["direction"] is None]
    _need(len(roots) == 1 and len(nodes) >= 2, "one nose-free anchor required")
    root = roots[0]
    a = nodes[root]
    _need(a["h"] >= 4 and a["w"] >= 5, "anchor too small for legend")
    swatches = ((a["t"]+1,a["l"]+1),(a["t"]+1,a["l"]+2))
    spent,fuel = (int(g[p]) for p in swatches)
    _need(spent != fuel and {spent,fuel} <= set(PALETTE)-{a["frame"]}, "invalid ordered legend")
    _need(all(g[p] == a["frame"] for p in a["body"]-set(swatches)), "invalid anchor body")
    incoming,outgoing = {},{}
    for i,n in enumerate(nodes):
        _need(n["frame"] not in (spent,fuel), "frame/state conflict")
        if i != root:
            coords = _counter(n["t"],n["l"],n["h"],n["w"],n["direction"])
            values = [int(g[p]) for p in coords]
            advanced = values.count(spent)
            _need(values == [spent]*advanced+[fuel]*(len(values)-advanced), "counter is not a raster prefix")
            n.update(coords=coords,advanced=advanced,fuel=len(values)-advanced,capacity=len(values))
        for side,pins in n["ports"].items():
            key = _norm(pins)
            table = incoming if side == n["direction"] else outgoing
            _need(key not in table, "a key geometry has more than one owner of a polarity")
            table[key] = i,side,pins
    _need(set(incoming) == set(outgoing), "unmatched keys")
    parents = [None]*len(nodes)
    joints = {}
    for key,(child,side,cpins) in incoming.items():
        parent,pside,ppins = outgoing[key]
        _need(child != parent and side == OPP[pside], "keys must face each other")
        _need(parents[child] is None, "multiple parents")
        parents[child] = parent
        joints[child] = ppins,cpins
    order,depth,targets = [root],{root:0},{root:(0,0)}
    for parent in order:
        for i in range(len(nodes)):
            if parents[i] == parent:
                _need(i not in depth, "cycle")
                p,c = (min(s) for s in joints[i])
                dy,dx = targets[parent]
                targets[i] = dy+p[0]-c[0],dx+p[1]-c[1]
                depth[i] = depth[parent]+1
                order.append(i)
    _need(len(order) == len(nodes), "keys do not form a rooted tree")
    paths = {root:[(0,0)]}
    for i in order[1:]:
        paths[i] = _path(targets[i],nodes[i]["direction"])
    scene = dict(grid=g,nodes=nodes,root=root,spent=spent,fuel=fuel,parents=parents,
                 order=order,depth=depth,targets=targets,paths=paths,joints=joints)
    _check_motion(scene)
    return scene


def _check_motion(s):
    """Prove clearance even with unlimited fuel and any sibling processing order."""
    ns,ps,root = s["nodes"],s["parents"],s["root"]
    sweeps,final = {},{}
    H,W = s["grid"].shape
    for i,n in enumerate(ns):
        sweeps[i] = set().union(*(_shift(n["cells"],d) for d in s["paths"][i]))
        final[i] = _shift(n["cells"],s["targets"][i])
        _need(all(0 <= y < H and 0 <= x < W for y,x in sweeps[i]), "potential path leaves canvas")
    for i in range(len(ns)):
        ancestors = _ancestors(ps,i)
        for j in range(i):
            _need(not ns[i]["cells"] & _dilate(ns[j]["cells"]), "input pieces touch")
            if j in ancestors or i in _ancestors(ps,j):
                child,ancestor = (i,j) if j in ancestors else (j,i)
                allowed = set()
                if ps[child] == ancestor:
                    allowed = _shift(s["joints"][child][0],s["targets"][ancestor])
                _need(not ns[child]["cells"] & sweeps[ancestor], "waiting piece blocks ancestor")
                _need(sweeps[child] & final[ancestor] <= allowed, "approach strikes docked ancestor")
            else:
                _need(not sweeps[i] & sweeps[j], "independent branch paths intersect")
    _need(ps[root] is None, "root has parent")


def _evaluate(s):
    active,docked,moves,positions = {},{s["root"]:True},{s["root"]:0},{s["root"]:(0,0)}
    for i in s["order"][1:]:
        active[i] = docked[s["parents"][i]]
        distance = len(s["paths"][i])-1
        moves[i] = min(s["nodes"][i]["fuel"],distance) if active[i] else 0
        docked[i] = active[i] and moves[i] == distance
        positions[i] = s["paths"][i][moves[i]]
    return active,docked,moves,positions


def _predict(s):
    _,_,moves,positions = _evaluate(s)
    out = np.zeros(s["grid"].shape,dtype=np.int8)
    for i,n in enumerate(s["nodes"]):
        dy,dx = positions[i]
        for y,x in n["body"]:
            _need(out[y+dy,x+dx] == BG, "body collision")
            out[y+dy,x+dx] = s["grid"][y,x]
        if i != s["root"]:
            for j,(y,x) in enumerate(n["coords"]):
                out[y+dy,x+dx] = s["spent"] if j < n["advanced"]+moves[i] else s["fuel"]
    for i,n in enumerate(s["nodes"]):
        dy,dx = positions[i]
        for y,x in n["cells"]-n["body"]:
            _need(out[y+dy,x+dx] in (BG,PIN), "pin/body collision")
            out[y+dy,x+dx] = PIN
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only visible geometry, the anchor legend, and the fixed family rule."""
    return _predict(_parse(input_grid))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _layout(rng, count, lesson, is_test, colors):
    parents = [None,0,1,0] if lesson is not None else [None]
    if is_test:
        parents = [None,0,1,2]
    while len(parents) <= count:
        depths = [0]
        for p in parents[1:]:
            depths.append(depths[p]+1)
        options = [i for i,d in enumerate(depths) if d < (3 if is_test else 2)]
        parents.append(_pick(rng,options))
    if len(parents) > count+1:
        return None
    root = dict(t=0,l=0,h=int(rng.integers(4,8)),w=int(rng.integers(5,9)),
                direction=None,ports={},frame=int(_pick(rng,colors)))
    nodes = [root]
    used = set()
    rotation = int(rng.integers(4))
    teaching_sides = [None,DIRS[(1+rotation)%4],DIRS[rotation],DIRS[rotation]]
    for i in range(1,count+1):
        p = parents[i]
        parent = nodes[p]
        for _ in range(100):
            h,w = int(rng.integers(3,8)),int(rng.integers(3,8))
            if lesson is not None:
                h,w = max(4,h),max(4,w)
            sides = [d for d in DIRS if d != parent["direction"] and d not in parent["ports"]]
            if not sides:
                return None
            side = teaching_sides[i] if lesson is not None else _pick(rng,sides)
            if side not in sides:
                continue
            a = parent["w"] if side in ("U","D") else parent["h"]
            b = w if side in ("U","D") else h
            span = int(rng.integers(1,min(a,b)))
            pattern = (0,span)
            if lesson is None and span >= 2 and rng.random() < .22:
                pattern = (0,int(rng.integers(1,span)),span)
            offa,offb = int(rng.integers(a-span)),int(rng.integers(b-span))
            pins = _port(parent,side,offa,pattern)
            if _norm(pins) in used:
                continue
            if side == "U":
                t,l = parent["t"]-h-1,parent["l"]+offa-offb
            elif side == "D":
                t,l = parent["t"]+parent["h"]+1,parent["l"]+offa-offb
            elif side == "L":
                t,l = parent["t"]+offa-offb,parent["l"]-w-1
            else:
                t,l = parent["t"]+offa-offb,parent["l"]+parent["w"]+1
            node = dict(t=t,l=l,h=h,w=w,direction=OPP[side],ports={OPP[side]:pins},
                        frame=int(_pick(rng,colors)))
            body = _rect(t,l,h,w)
            bad = False
            for j,other in enumerate(nodes):
                oc = _cells(other)
                if body & oc or pins & _rect(other["t"],other["l"],other["h"],other["w"]):
                    bad = True
                    break
                if pins & oc:
                    bad = True
                    break
            if bad:
                continue
            total = set().union(*(_cells(n) for n in nodes),body,pins)
            y0,x0,y1,x1 = _bbox(total)
            if y1-y0 > 26 or x1-x0 > 26:
                continue
            parent["ports"][side] = pins
            nodes.append(node)
            used.add(_norm(pins))
            break
        else:
            return None
    if lesson is not None:
        nodes[2]["frame"] = nodes[3]["frame"]
    return nodes,parents


def _scatter(rng, nodes, parents, lesson):
    cells = [_cells(n) for n in nodes]
    ances = [_ancestors(parents,i) for i in range(len(nodes))]
    for _ in range(18):
        starts,sweeps,paths = {0:(0,0)},{0:cells[0]},{0:[(0,0)]}
        occupied = _dilate(cells[0])
        allcells = set(cells[0])
        for i,n in enumerate(nodes[1:],1):
            dy,dx = VEC[n["direction"]]
            for _ in range(150):
                normal = int(rng.integers(2,8))
                tangent = int(rng.integers(-4,5))
                if lesson is not None and i == 1:
                    tangent = int(rng.choice((-3,-2,2,3)))
                if lesson == 1 and i == 2:
                    tangent = int(rng.choice((-2,-1,1,2)))
                offset = (-normal*dy+(tangent if dx else 0),
                          -normal*dx+(tangent if dy else 0))
                distance = normal+abs(tangent)
                capacity = (n["h"]-2)*(n["w"]-2)
                if lesson is not None and i in (1,2,3) and capacity <= distance:
                    continue
                initial = _shift(cells[i],offset)
                if initial & occupied:
                    continue
                path = _path((-offset[0],-offset[1]),n["direction"])
                sweep = set().union(*(_shift(initial,d) for d in path))
                okay = True
                for j in starts:
                    if j in ances[i]:
                        allowed = n["ports"][n["direction"]] if parents[i] == j else set()
                        if initial & sweeps[j] or not (sweep & cells[j] <= allowed):
                            okay = False
                            break
                    elif sweep & sweeps[j]:
                        okay = False
                        break
                if not okay:
                    continue
                total = allcells | sweep
                t,l,b,r = _bbox(total)
                if b-t >= 30 or r-l >= 30:
                    continue
                starts[i],sweeps[i],paths[i] = offset,sweep,path
                occupied |= _dilate(initial)
                allcells = total
                break
            else:
                break
        if len(starts) == len(nodes):
            return starts,paths,allcells
    return None


def _budget(rng, capacity, distance):
    mode = _pick(rng,("fresh","destination","exact","fuel","zero"))
    if mode == "fresh":
        return capacity
    if mode == "destination" and capacity > distance:
        return int(rng.integers(distance+1,capacity+1))
    if mode == "exact" and capacity >= distance:
        return distance
    if mode == "zero":
        return 0
    return int(rng.integers(1,min(capacity,distance-1)+1))


def _render(nodes, positions, remaining, consumed, shape, origin, spent, fuel):
    out = np.zeros(shape,dtype=np.int8)
    oy,ox = origin
    for i,n in enumerate(nodes):
        dy,dx = positions[i]
        t,l,h,w = n["t"]+dy+oy,n["l"]+dx+ox,n["h"],n["w"]
        _need(np.all(out[t:t+h,l:l+w] == BG), "construction body overlap")
        out[t:t+h,l:l+w] = n["frame"]
        if i == 0:
            out[t+1,l+1],out[t+1,l+2] = spent,fuel
        else:
            threshold = (h-2)*(w-2)-remaining[i]+consumed[i]
            for k,p in enumerate(_counter(t,l,h,w,n["direction"])):
                out[p] = spent if k < threshold else fuel
            ny,nx = _nose(n)
            out[ny+dy+oy,nx+dx+ox] = NOSE
    for i,n in enumerate(nodes):
        dy,dx = positions[i]
        for pins in n["ports"].values():
            for y,x in pins:
                p = y+dy+oy,x+dx+ox
                _need(out[p] in (BG,PIN), "construction pin overlap")
                out[p] = PIN
    return out


def _make_pair(rng, spent, fuel, colors, lesson, is_test):
    count = 3 if lesson is not None else int(rng.integers(3,6) if is_test else rng.integers(2,5))
    for _ in range(160):
        layout = _layout(rng,count,lesson,is_test,colors)
        if layout is None:
            continue
        nodes,parents = layout
        scatter = _scatter(rng,nodes,parents,lesson)
        if scatter is None:
            continue
        starts,paths,allcells = scatter
        remaining,consumed,realized = {},{0:0},{0:(0,0)}
        docked = {0:True}
        for i,n in enumerate(nodes[1:],1):
            cap = (n["h"]-2)*(n["w"]-2)
            distance = len(paths[i])-1
            tangent = abs(starts[i][0] if n["direction"] in ("L","R") else starts[i][1])
            if lesson == 0 and i == 1:
                rem = 1
            elif (lesson is not None and i == 1) or (lesson == 0 and i in (2,3)):
                rem = int(rng.integers(distance+1,cap+1))
            elif lesson == 1 and i == 2:
                rem = tangent+1
            elif lesson == 1 and i == 3:
                rem = distance
            elif is_test and i == 1 and cap >= distance:
                rem = int(rng.integers(distance,cap+1))
            elif is_test and i == 2:
                rem = int(rng.integers(1,min(cap,distance-1)+1))
            elif is_test and i == 3:
                rem = cap
            else:
                rem = _budget(rng,cap,distance)
            remaining[i] = rem
            move = min(rem,distance) if docked[parents[i]] else 0
            consumed[i] = move
            docked[i] = docked[parents[i]] and move == distance
            step = paths[i][move]
            realized[i] = starts[i][0]+step[0],starts[i][1]+step[1]
        t,l,b,r = _bbox(allcells)
        h,w = b-t+1,r-l+1
        pady,padx = int(rng.integers(min(4,30-h)+1)),int(rng.integers(min(4,30-w)+1))
        my,mx = int(rng.integers(pady+1)),int(rng.integers(padx+1))
        shape,origin = (h+pady,w+padx),(my-t,mx-l)
        inp = _render(nodes,starts,remaining,{i:0 for i in starts},shape,origin,spent,fuel)
        out = _render(nodes,realized,remaining,consumed,shape,origin,spent,fuel)
        # Parsing uses grids only and checks the independently constructed paths.
        scene = _parse(inp)
        _need(np.array_equal(_predict(scene),out), "construction/inference disagree")
        return {"input":inp,"output":out}
    raise RuntimeError("bounded construction exhausted")


def _signature(g):
    return g.shape, hashlib.sha256(np.ascontiguousarray(g,dtype=np.int8).tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain,ntest = _counts(num_train,num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    state = list(map(int,rng.choice(PALETTE,size=2,replace=False)))
    spent,fuel = state
    colors = [c for c in PALETTE if c not in state]
    lessons = [0,1]+[None]*(ntrain-2)
    rng.shuffle(lessons)
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",ntrain),("test",ntest)):
        for i in range(count):
            for _ in range(16):
                pair = _make_pair(rng,spent,fuel,colors,lessons[i] if split == "train" else None,split == "test")
                sig = _signature(pair["input"])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(s):
    active,docked,moves,_ = _evaluate(s)
    vehicles = []
    for i in s["order"][1:]:
        n = s["nodes"][i]
        distance = len(s["paths"][i])-1
        dy,dx = s["targets"][i]
        tangent = abs(dx if n["direction"] in ("U","D") else dy)
        status = "blocked" if not active[i] else "zero" if n["fuel"] == 0 else "fuel" if n["fuel"] < distance else "exact" if n["fuel"] == distance else "destination"
        vehicles.append(dict(index=i,parent=s["parents"][i],direction=n["direction"],depth=s["depth"][i],
                             interior=(n["h"]-2,n["w"]-2),fuel=n["fuel"],advanced=n["advanced"],
                             capacity=n["capacity"],distance=distance,tangent=tangent,movement=moves[i],
                             status=status,docked=docked[i],preturn=active[i] and 0<moves[i]<tangent,
                             postturn=active[i] and tangent>0 and tangent<moves[i]<distance))
    # Parallel equal-area keys attached to parents in different activation states.
    geometry_pair = None
    for i in s["order"][1:]:
        for j in s["order"][1:]:
            if i >= j:
                continue
            a,b = s["joints"][i][1],s["joints"][j][1]
            ni,nj = s["nodes"][i],s["nodes"][j]
            if (ni["direction"] == nj["direction"] and len(a) == len(b) and _norm(a) != _norm(b)
                    and active[i] != active[j] and ni["fuel"] > 0 and nj["fuel"] > 0):
                geometry_pair = (i,j)
    return dict(shape=s["grid"].shape,count=len(s["nodes"]),depth=max(s["depth"].values()),
                states=(s["spent"],s["fuel"]),keys=[_norm(p) for p,c in s["joints"].values()],
                vehicles=vehicles,geometry_witness=geometry_pair is not None,geometry_pair=geometry_pair)


def _check_ticks(s, out):
    """Check counter advancement by rotating real interiors and ticking singly."""
    turns = {"U":0,"D":2,"L":3,"R":1}
    _,_,moves,positions = _evaluate(s)
    for i in s["order"][1:]:
        n = s["nodes"][i]
        t,l,h,w = (n[k] for k in ("t","l","h","w"))
        k = turns[n["direction"]]
        counter = np.rot90(s["grid"][t+1:t+h-1,l+1:l+w-1],k).copy()
        for _ in range(moves[i]):
            pending = np.flatnonzero(counter.ravel() == s["fuel"])
            _need(len(pending) > 0, "unit tick has no fuel")
            counter.flat[int(pending[0])] = s["spent"]
        dy,dx = positions[i]
        expected = np.rot90(counter,-k)
        _need(np.array_equal(expected,out[t+dy+1:t+dy+h-1,l+dx+1:l+dx+w-1]),
              "raster update disagrees with rotated unit ticks")


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle,dict) and set(puzzle) == {"train","test"}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s],list) for s in ("train","test")), "splits must be lists")
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen,states,training = set(),set(),[]
    for split in ("train","test"):
        for pair in puzzle[split]:
            _need(isinstance(pair,dict) and set(pair) == {"input","output"}, "invalid pair schema")
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            _need(inp.shape == out.shape, "dimensions changed")
            sig = _signature(inp)
            _need(sig not in seen, "duplicate inputs")
            seen.add(sig)
            s = _parse(inp)
            states.add((s["spent"],s["fuel"]))
            _need(np.array_equal(_predict(s),out), "incorrect output")
            _check_ticks(s,out)
            _,docked,moves,positions = _evaluate(s)
            spent,fuel = s["spent"],s["fuel"]
            total = sum(moves.values())
            _need(np.count_nonzero(inp == fuel)-np.count_nonzero(out == fuel) == total, "fuel not conserved")
            _need(np.count_nonzero(out == spent)-np.count_nonzero(inp == spent) == total, "spent count mismatch")
            shared = sum(len(s["joints"][i][1]) for i in s["order"][1:] if docked[i])
            _need(np.count_nonzero(inp == PIN)-np.count_nonzero(out == PIN) == shared, "incomplete or extra docking")
            for i,n in enumerate(s["nodes"]):
                dy,dx = positions[i]
                rigid = n["body"]-set(n.get("coords",()))
                _need(all(out[y+dy,x+dx] == inp[y,x] for y,x in rigid), "frame/nose/legend deformed")
            f = _features(s)
            if split == "train":
                training.append(f)
    _need(len(states) == 1, "state alphabet changes within puzzle")
    vs = [v for f in training for v in f["vehicles"]]
    _need(any(v["status"] == "destination" for v in vs), "missing arrival with surplus")
    _need(any(v["status"] == "fuel" for v in vs), "missing fuel-limited movement")
    _need(any(v["status"] == "blocked" and v["fuel"] > v["distance"] for v in vs), "missing well-fuelled waiting descendant")
    _need(any(v["preturn"] for v in vs) and any(v["postturn"] for v in vs), "approach order not demonstrated")
    _need(any(f["geometry_witness"] for f in training), "geometry does not distinguish activation")


def self_test():
    """42 streaming seeds; no payload retention, files, traces or subprocesses."""
    shapes,states,keys,interiors,statuses,directions,structures,depths,counts = (set() for _ in range(9))
    pairs=vehicles=0
    fuel_ablation_puzzles=geometry_ablation_puzzles=0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fuel_witness=geometry_witness=False
        for split in ("train","test"):
            for pair in p[split]:
                s = _parse(pair["input"])
                f = _features(s)
                if split == "train":
                    active,_,moves,positions = _evaluate(s)
                    # Without A every edge can complete, so all vehicles reach
                    # their ideal assembly translations. Compare actual poses.
                    fuel_witness |= any(positions[i] != s["targets"][i] and active[i] and moves[i] > 0
                                        for i in s["order"][1:])
                    if f["geometry_pair"] is not None:
                        i,j = f["geometry_pair"]
                        # An area-only swap assigns these positive-fuel pieces
                        # to parents with opposite docking outcomes. It changes
                        # which counter/pose can advance, even before redrawing.
                        geometry_witness |= (active[i] != active[j] and
                                             s["nodes"][i]["fuel"] > 0 and s["nodes"][j]["fuel"] > 0)
                shapes.add(f["shape"])
                states.add(f["states"])
                keys.update(f["keys"])
                counts.add(f["count"])
                depths.add((split,f["depth"]))
                pairs += 1
                # Normalize variable hues but keep their structural equality.
                palette = {}
                normalized = bytes(palette.setdefault(int(c),len(palette)) for c in pair["input"].flat)
                structures.add((pair["input"].shape,hashlib.sha256(normalized).hexdigest()))
                for v in f["vehicles"]:
                    vehicles += 1
                    interiors.add(v["interior"])
                    statuses.add(v["status"])
                    directions.add(v["direction"])
                before = pair["input"].copy()
                _need(np.array_equal(solve(before),pair["output"]), "solver mismatch")
                _need(np.array_equal(before,pair["input"]), "solve mutates input")
        _need(fuel_witness and geometry_witness, "parent-concept ablation has no effect")
        fuel_ablation_puzzles += int(fuel_witness)
        geometry_ablation_puzzles += int(geometry_witness)
        if seed == 0:
            q = generate_puzzle(seed)
            _need(all(np.array_equal(x[k],y[k]) for split in p for x,y in zip(p[split],q[split]) for k in x), "nondeterministic")
            bad = {split:[{k:g.copy() for k,g in pair.items()} for pair in p[split]] for split in p}
            bad["test"][0]["output"][0,0] = (int(bad["test"][0]["output"][0,0])+1)%10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError("wrong output accepted")
            broken_input = p["train"][0]["input"].copy()
            pin = tuple(np.argwhere(broken_input == PIN)[0])
            broken_input[pin] = BG
            try:
                solve(broken_input)
            except AssertionError:
                pass
            else:
                raise AssertionError("incomplete key accepted")
    _need(len(shapes) >= 40 and len(states) >= 20 and len(interiors) >= 15, "structural variation collapsed")
    _need(len(keys) >= 12 and len(structures) >= 200, "key/structure variation collapsed")
    _need(directions == set(DIRS), "missing direction")
    _need(statuses == {"blocked","zero","fuel","exact","destination"}, "missing stopping/activation case")
    _need({("train",2),("test",3)} <= depths and len(counts) >= 3, "missing depth/count variation")
    for tr,te in ((2,1),(10,4)):
        p = generate_puzzle(42,tr,te)
        _need((len(p["train"]),len(p["test"])) == (tr,te), "count mismatch")
    for tr,te in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,np.bool_(True))):
        try:
            generate_puzzle(0,tr,te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return dict(seeds=42,pairs=pairs,vehicles=vehicles,shapes=len(shapes),state_palettes=len(states),
                key_geometries=len(keys),interior_shapes=len(interiors),normalized_inputs=len(structures),
                piece_counts=sorted(counts),statuses=sorted(statuses),depths=sorted(depths),
                fuel_ablation_puzzles=fuel_ablation_puzzles,geometry_ablation_puzzles=geometry_ablation_puzzles)


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",", ":")))
