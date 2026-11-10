"""Cooperative enclosure: topology chooses a transparent-sheet registration.

Only standard library and NumPy. No parent imports, disk I/O, or saved state.
"""
from __future__ import annotations

from itertools import product

import numpy as np


HOST_SEEDS = (0,827917635,917863479,272128701,592162565,436760136,44010853,68787961,253478791,165416898,903439143,499383624,215774673,960929125,429904300,439912488,479139682,956216163,941586388,154953009,608112679,852968079,179140541,103858239,339275646,340456954,305300250,524991702,902203620,342007498,448586619,937442380,894817825,988187090,361051952,393300856,215426252,881005762,529189004,459723178,750528387,194080680)
DIR4 = ((-1,0),(1,0),(0,-1),(0,1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _parts(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                q = (r+dr, c+dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _box(cells):
    return (min(r for r,c in cells), min(c for r,c in cells),
            max(r for r,c in cells), max(c for r,c in cells))


def _shift(cells, dr, dc):
    return {(r+dr,c+dc) for r,c in cells}


def _holes(cells):
    t,l,b,r = _box(cells)
    blanks = set(product(range(t-1,b+2),range(l-1,r+2))) - cells
    return [p for p in _parts(blanks)
            if not any(y in (t-1,b+1) or x in (l-1,r+1) for y,x in p)]


def _mask(cells):
    t,l,b,r = _box(cells)
    a = np.zeros((b-t+1,r-l+1),dtype=bool)
    for y,x in cells:
        a[y-t,x-l] = True
    return a


def _key(a):
    a = np.asarray(a,dtype=np.int8)
    return (a.shape,a.tobytes())


def _parse_scene(cells):
    """Reject unfinished walls; derive parents by flood-filled actual holes."""
    _require(bool(cells), "empty drawing")
    objects = []
    for part in _parts(cells):
        t,l,b,r = _box(part)
        holes = _holes(part)
        if holes:
            _require(len(holes) == 1 and b-t >= 4 and r-l >= 4, "wall must enclose one region")
            _require(all(sum((y+dy,x+dx) in part for dy,dx in DIR4) == 2 for y,x in part),
                     "wall is not a simple thin cycle")
            kind, hole = "C", holes[0]
        else:
            _require(b-t <= 2 and r-l <= 2 and len(part)*2 >= (b-t+1)*(r-l+1), "unclosed wall or noncompact blob")
            kind, hole = "B", set()
        objects.append(dict(cells=part,box=(t,l,b,r),kind=kind,hole=hole))
    _require(any(o["kind"] == "C" for o in objects), "no completed wall")
    ownership = {p:i for i,o in enumerate(objects) for p in o["cells"]}
    for (y,x),i in ownership.items():
        for dy,dx in product((-1,0,1),repeat=2):
            _require(ownership.get((y+dy,x+dx),i) == i, "separate objects touch")
    parents = []
    for i,o in enumerate(objects):
        containers = []
        for j,p in enumerate(objects):
            if i == j or p["kind"] != "C":
                continue
            overlap = o["cells"] & p["hole"]
            _require(not overlap or overlap == o["cells"], "crossing wall")
            if overlap:
                containers.append(j)
        parents.append(min(containers,key=lambda j:len(objects[j]["hole"])) if containers else None)
    children = {i:[] for i in range(len(objects))}
    roots = []
    for i,p in enumerate(parents):
        (roots if p is None else children[p]).append(i)
    for i,o in enumerate(objects):
        _require(bool(children[i]) == (o["kind"] == "C"), "empty wall")
    return dict(objects=objects,parents=parents,children=children,roots=roots)


def _join(tiles, axis, offsets, gaps):
    """Tile = (boolean drawing, (leaf id,row,col) records)."""
    offsets = [x-min(offsets) for x in offsets]
    shape, starts, position = [0,0], [], 0
    for i,(a,leaves) in enumerate(tiles):
        start = [0,0]
        start[axis],start[1-axis] = position,offsets[i]
        starts.append(start)
        position += a.shape[axis] + (gaps[i] if i < len(gaps) else 0)
        shape[1-axis] = max(shape[1-axis],offsets[i]+a.shape[1-axis])
    shape[axis] = position
    drawing, points = np.zeros(shape,dtype=bool), []
    for (a,leaves),(y,x) in zip(tiles,starts):
        h,w = a.shape
        drawing[y:y+h,x:x+w] |= a
        points.extend((k,r+y,c+x) for k,r,c in leaves)
    return drawing,points


def _normalize(info):
    """Use whitespace cuts and terminal projections, not whole-frame centers."""
    objects, memo = info["objects"], {}

    def group(ids):
        ids = tuple(sorted(ids))
        if ids in memo:
            return memo[ids]
        if len(ids) == 1:
            i = ids[0]
            if objects[i]["kind"] == "B":
                result = np.ones((1,1),dtype=bool),[(i,0,0)]
            else:
                a,leaves = group(info["children"][i])
                a = np.pad(a,2)
                a[0,:] = a[-1,:] = a[:,0] = a[:,-1] = True
                result = a,[(k,y+2,x+2) for k,y,x in leaves]
        else:
            candidates = []
            for axis in (0,1):
                ordered = sorted(ids,key=lambda i:objects[i]["box"][axis])
                partitions, end = [], -100000
                for i in ordered:
                    box = objects[i]["box"]
                    if box[axis] > end+1:
                        partitions.append([])
                    partitions[-1].append(i)
                    end = max(end,box[axis+2])
                if len(partitions) == 1:
                    continue
                try:
                    tiles = [group(p) for p in partitions]
                    edges = []
                    for i in range(len(tiles)):
                        for j in range(i+1,len(tiles)):
                            deltas = set()
                            for k,y,x in tiles[i][1]:
                                for m,v,w in tiles[j][1]:
                                    a,b = objects[k]["box"],objects[m]["box"]
                                    cross = 1-axis
                                    if max(a[cross],b[cross]) <= min(a[cross+2],b[cross+2]):
                                        deltas.add((y,x)[cross]-(v,w)[cross])
                            _require(len(deltas) <= 1,"competing alignment anchors")
                            if deltas:
                                edges.append((i,j,deltas.pop()))
                    offsets = {0:0}
                    for _ in tiles:
                        for i,j,d in edges:
                            if i in offsets and j in offsets:
                                _require(offsets[j] == offsets[i]+d,"inconsistent anchors")
                            elif i in offsets:
                                offsets[j] = offsets[i]+d
                            elif j in offsets:
                                offsets[i] = offsets[j]-d
                    _require(len(offsets) == len(tiles),"unaligned diagonal placement")
                    candidates.append(_join(tiles,axis,[offsets[i] for i in range(len(tiles))],[1]*(len(tiles)-1)))
                except ValueError:
                    continue
            _require(bool(candidates),"no cardinal layout")
            result = candidates[0]
            _require(all(np.array_equal(a,result[0]) and sorted(leaves) == sorted(result[1])
                         for a,leaves in candidates[1:]),"ambiguous layout")
        memo[ids] = result
        return result

    a,leaves = group(info["roots"])
    pads = [0,0,0,0]
    for k,y,x in leaves:
        if k in info["roots"] and objects[k]["kind"] == "B":
            pads = [max(pads[0],1-y),max(pads[1],y+2-a.shape[0]),
                    max(pads[2],1-x),max(pads[3],x+2-a.shape[1])]
    return np.pad(a,((pads[0],pads[1]),(pads[2],pads[3])))


def _decode(input_grid):
    """Decode visible colors, carriers and docks; no output or RNG access."""
    a = _grid(input_grid)
    values,counts = np.unique(a,return_counts=True)
    _require(len(values) == 4 and np.count_nonzero(counts == counts.max()) == 1,"four colors and unique dominant background required")
    bg = int(values[np.argmax(counts)])
    colors = [int(x) for x in values if x != bg]
    choices = []
    for marker in colors:
        bars = _parts({tuple(map(int,p)) for p in np.argwhere(a == marker)})
        if len(bars) not in (3,4) or len({len(b) for b in bars}) != 1 or len(bars[0]) < 2:
            continue
        for axis in (0,1):
            def oriented(cells):
                return {(p[axis],p[1-axis]) for p in cells}
            bs = [oriented(b) for b in bars]
            if any(len({y for y,x in b}) != 1 or _box(b)[3]-_box(b)[1]+1 != len(b) for b in bs):
                continue
            aa = a if axis == 0 else a.T
            for carrier in colors:
                if carrier == marker:
                    continue
                pieces = _parts({tuple(map(int,p)) for p in np.argwhere(aa == carrier)})
                boards = []
                for piece in pieces:
                    t,l,b,r = _box(piece)
                    if b-t < 4 or r-l < 4:
                        continue
                    border = ({(y,x) for y in range(t,b+1) for x in (l,r)} |
                              {(y,x) for y in (t,b) for x in range(l,r+1)})
                    if border <= piece:
                        boards.append((t,l,b,r))
                if len(boards) != 2:
                    continue
                drawing = next(x for x in colors if x not in (marker,carrier))
                payloads, docks, explained = [], [], set()
                valid = True
                for t,l,b,r in boards:
                    border = ({(y,x) for y in range(t,b+1) for x in (l,r)} |
                              {(y,x) for y in (t,b) for x in range(l,r+1)})
                    explained |= border
                    payloads.append({tuple(map(int,p)) for p in np.argwhere(aa == drawing)
                                     if t < p[0] < b and l < p[1] < r})
                    own = []
                    for bar in bs:
                        if not all(t < y < b and l < x < r for y,x in bar):
                            continue
                        signs = [s for s in (-1,1) if all(aa[y+s,x] == carrier for y,x in bar)]
                        if len(signs) != 1:
                            valid = False
                            break
                        s = signs[0]
                        if any(aa[y-s,x] != bg for y,x in bar):
                            valid = False
                            break
                        explained |= _shift(bar,s,0)
                        own.append((_box(bar)[:2],s))
                    docks.append(own)
                if not valid or sorted(map(len,docks)) != [1,len(bs)-1] or sum(map(len,payloads)) != int(np.sum(aa == drawing)):
                    continue
                if explained != {tuple(map(int,p)) for p in np.argwhere(aa == carrier)}:
                    continue
                anchor = 0 if len(docks[0]) > 1 else 1
                mover = 1-anchor
                signset = {s for p,s in docks[anchor]}
                if len(signset) != 1 or docks[mover][0][1] != -next(iter(signset)):
                    continue
                choices.append(dict(bg=bg,fg=drawing,carrier=carrier,marker=marker,axis=axis,
                                    payloads=payloads,docks=docks,anchor=anchor,mover=mover))
    _require(len(choices) == 1,"sheet palette or dock geometry is ambiguous")
    return choices[0]


def _registrations(decoded):
    a,b = decoded["anchor"],decoded["mover"]
    (my,mx),_ = decoded["docks"][b][0]
    results = []
    for index,((y,x),s) in enumerate(decoded["docks"][a]):
        dr,dc = y+s-my,x-mx
        moved = _shift(decoded["payloads"][b],dr,dc)
        cells = decoded["payloads"][a] | moved
        try:
            info = _parse_scene(cells)
            mask = _normalize(info)
            _require(max(mask.shape) <= 30,"canonical drawing too large")
        except ValueError:
            continue
        results.append(dict(index=index,shift=(dr,dc),cells=cells,info=info,mask=mask,
                            overlap=len(decoded["payloads"][a] & moved)))
    return results


def _nearest_offers(decoded):
    a,b = decoded["anchor"],decoded["mover"]
    (my,mx),_ = decoded["docks"][b][0]
    distances = [(y-my)**2+(x-mx)**2 for (y,x),s in decoded["docks"][a]]
    return {i for i,d in enumerate(distances) if d == min(distances)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    decoded = _decode(input_grid)
    choices = _registrations(decoded)
    _require(len(choices) == 1,"exactly one dock must complete a valid diagram")
    out = np.where(choices[0]["mask"],decoded["fg"],decoded["bg"]).astype(np.int8)
    return out.T.copy() if decoded["axis"] else out


# Generation uses a tree of objects and non-object spatial groups. Its expected
# answer is drawn alongside metric geometry, before fragmentation and docking.
def _group(rng,children,axis=None):
    if len(children) == 1:
        return children[0]
    return ("G",int(rng.integers(2)) if axis is None else axis,tuple(children),
            tuple(int(rng.integers(_leaves(c))) for c in children))


def _leaves(node):
    return 1 if node[0] == "B" else (_leaves(node[1]) if node[0] == "C" else sum(_leaves(c) for c in node[2]))


def _scene(rng,kind):
    b = ("B",)
    if kind == 0:
        return ("C",_group(rng,[b]*int(rng.integers(2,4))))
    if kind == 1:
        axis = int(rng.integers(2))
        inner = ("C",_group(rng,[b,b],1-axis))
        children = [b,inner]
        rng.shuffle(children)
        return ("C",_group(rng,children,axis))
    nested = bool(rng.integers(2))
    children = [b]*int(rng.integers(1,4))
    if nested:
        children[int(rng.integers(len(children)))] = ("C",_group(rng,[b]*int(rng.integers(1,4))))
    if len(children) >= 3 and rng.random() < .65:
        cut = int(rng.integers(1,len(children)))
        axis = int(rng.integers(2))
        content = _group(rng,[_group(rng,children[:cut],1-axis),_group(rng,children[cut:],1-axis)],axis)
    else:
        content = _group(rng,children)
    root = ("C",content)
    if not nested and rng.random() < .25:
        return _group(rng,[root,("C",b)])
    if rng.random() < .35:
        roots = [root,b]
        rng.shuffle(roots)
        return _group(rng,roots)
    return root


def _wall(rng,h,w,pads):
    """A rectangle or a rectangle with safe orthogonal bays on its four sides."""
    profiles = []
    for n,pad in zip((w,w,h,h),pads):
        profile = np.zeros(n,dtype=int)
        if pad >= 3 and n >= 7 and rng.random() < .8:
            left = int(rng.integers(2,n-4))
            right = int(rng.integers(left+2,n-2))
            profile[left:right+1] = int(rng.integers(1,pad-1))
        profiles.append(profile)
    top,bottom,left,right = profiles
    path = ([(int(top[x]),x) for x in range(w)] +
            [(y,w-1-int(right[y])) for y in range(1,h)] +
            [(h-1-int(bottom[x]),x) for x in range(w-2,-1,-1)] +
            [(y,int(left[y])) for y in range(h-2,-1,-1)])
    a = np.zeros((h,w),dtype=bool)
    for (y,x),(v,z) in zip(path,path[1:]+path[:1]):
        a[min(y,v):max(y,v)+1,x] = True
        a[v,min(x,z):max(x,z)+1] = True
    return a


def _draw(node,rng,compact=False):
    serial = [0]
    exterior = set()

    def draw(n,depth):
        if n[0] == "B":
            k = serial[0]
            serial[0] += 1
            if depth == 0:
                exterior.add(k)
            h,w = (1,1) if compact else (int(rng.integers(1,4)),int(rng.integers(1,4)))
            a = np.ones((h,w),dtype=bool)
            if h*w >= 4 and rng.random() < .5:
                a[0,0] = False
            return ((a,[(k,(h-1)//2,(w-1)//2)]),(np.ones((1,1),dtype=bool),[(k,0,0)]))
        if n[0] == "G":
            pairs = [draw(c,depth) for c in n[2]]
            axis = n[1]
            versions = []
            for version in (0,1):
                tiles = [p[version] for p in pairs]
                offsets = [-p[1][k][2-axis] for p,k in zip(tiles,n[3])]
                gaps = [1 if compact or version else int(rng.integers(1,4)) for _ in tiles[:-1]]
                versions.append(_join(tiles,axis,offsets,gaps))
            return tuple(versions)
        (a,points),(out,opoints) = draw(n[1],depth+1)
        pads = [2]*4 if compact else [int(rng.integers(2,5)) for _ in range(4)]
        top,bottom,left,right = pads
        h,w = a.shape
        wall = _wall(rng,h+top+bottom,w+left+right,pads)
        wall[top:top+h,left:left+w] |= a
        out = np.pad(out,2)
        out[0,:] = out[-1,:] = out[:,0] = out[:,-1] = True
        return ((wall,[(k,y+top,x+left) for k,y,x in points]),
                (out,[(k,y+2,x+2) for k,y,x in opoints]))

    (a,points),(out,leaves) = draw(node,0)
    pads = [0,0,0,0]
    for k,y,x in leaves:
        if k in exterior:
            pads = [max(pads[0],1-y),max(pads[1],y+2-out.shape[0]),
                    max(pads[2],1-x),max(pads[3],x+2-out.shape[1])]
    return a,np.pad(out,((pads[0],pads[1]),(pads[2],pads[3])))


def _split_scene(mask,rng,force_overlap=False):
    layers = [set(),set()]
    first_wall = True
    for cells in _parts({tuple(map(int,p)) for p in np.argwhere(mask)}):
        t,l,b,r = _box(cells)
        if _holes(cells):
            axis = int(rng.integers(2))
            low,high = (t,b) if axis == 0 else (l,r)
            cut = int(rng.integers(low+2,high-1))
            for p in cells:
                layers[int(p[axis] >= cut)].add(p)
            # Shared wall pixels are genuine transparent union, not XOR.
            if (force_overlap and first_wall) or rng.random() < .65:
                overlap = {p for p in cells if p[axis] == cut}
                layers[0] |= overlap
            first_wall = False
        else:
            who = int(rng.integers(3))
            layers[who % 2] |= cells
            if who == 2:
                layers[1] |= cells
    return layers


def _board(payload,bars,sign,palette,rng):
    bg,fg,carrier,marker = palette
    t,l,b,r = _box(payload | set.union(*bars) | set.union(*[_shift(x,sign,0) for x in bars]))
    # Independent optional margins alter local marker offsets and carrier sizes.
    top = int(rng.integers(2,4)) if sign > 0 else int(rng.integers(1,3))
    bottom = int(rng.integers(1,3))
    left,right = 1,1
    a = np.full((b-t+1+top+bottom,r-l+1+left+right),bg,dtype=np.int8)
    a[0,:] = a[-1,:] = a[:,0] = a[:,-1] = carrier
    for cells,color in [(payload,fg)]+[(x,marker) for x in bars]+[(_shift(x,sign,0),carrier) for x in bars]:
        for y,x in cells:
            a[y-t+top,x-l+left] = color
    return a


def _stage(boards,bg,rng):
    order = list(boards)
    if rng.integers(2):
        order.reverse()
    options = [axis for axis in (0,1) if sum(a.shape[axis] for a in order)+1 <= 30 and max(a.shape[1-axis] for a in order) <= 30]
    _require(bool(options),"sheets do not fit")
    axis = int(rng.choice(options))
    gap = int(rng.integers(1,min(4,31-sum(a.shape[axis] for a in order))))
    crossmax = max(a.shape[1-axis] for a in order)
    crosssize = int(rng.integers(crossmax,min(31,crossmax+4)))
    starts,position = [],0
    for a in order:
        start = [0,0]
        start[axis] = position
        start[1-axis] = int(rng.integers(crosssize-a.shape[1-axis]+1))
        starts.append(start)
        position += a.shape[axis]+gap
    shape = [0,0]
    shape[axis],shape[1-axis] = position-gap,crosssize
    grid = np.full(shape,bg,dtype=np.int8)
    for a,(y,x) in zip(order,starts):
        h,w = a.shape
        grid[y:y+h,x:x+w] = a
    return grid


def _make_pair(rng,kind,dock_count,correct_slot,palette,rotation,used):
    for attempt in range(256):
        compact = attempt % 16 >= 8
        node = _scene(rng,kind)
        metric,expected = _draw(node,rng,compact)
        if metric.shape[1] > metric.shape[0] and rng.random() < .7:
            metric,expected = metric.T,expected.T
        if metric.shape[1] > 12 or metric.shape[0] > 23:
            continue
        if kind == 0 and np.array_equal(metric,expected):
            continue
        try:
            scene = _parse_scene({tuple(map(int,p)) for p in np.argwhere(metric)})
            if not np.array_equal(_normalize(scene),expected):
                continue
            layers = _split_scene(metric,rng,force_overlap=kind == 1)
            length = int(rng.integers(2,4))
            span = dock_count*length+dock_count-1
            if span > 12:
                continue
            start = int(rng.integers(max(1,metric.shape[1]-span+1)))
            slots = [{(0,start+i*(length+1)+j) for j in range(length)} for i in range(dock_count)]
            # Docks are ordered along the visible bar axis. Rotations 1 and 2
            # reverse that order; the two witnesses must still teach 0 and 1.
            slot = (correct_slot if rotation in (0,3) else dock_count-1-correct_slot) if correct_slot is not None else int(rng.integers(dock_count))
            single = _shift(slots[slot],-1,0)
            payloads = [_shift(x,4,0) for x in layers]
            boards = [_board(payloads[0],slots,-1,palette,rng),_board(payloads[1],[single],1,palette,rng)]
            inp = np.rot90(_stage(boards,palette[0],rng),rotation).copy()
            if _key(inp) in used:
                continue
            decoded = _decode(inp)
            registrations = _registrations(decoded)
            if len(registrations) != 1:
                continue
            if correct_slot is not None and registrations[0]["index"] != correct_slot:
                continue
            if kind == 0 and registrations[0]["index"] in _nearest_offers(decoded):
                continue
            out = np.rot90(np.where(expected,palette[1],palette[0]).astype(np.int8),rotation).copy()
            if not np.array_equal(solve(inp),out):
                continue
            # Every wall must need both layers, not just one complete diagram.
            if any(_holes(p) for layer in layers for p in _parts(layer)):
                continue
            return dict(input=inp,output=out)
        except ValueError:
            continue
    raise RuntimeError("bounded cooperative-wall construction exhausted")


def _counts(num_train,num_test):
    for name,n,lo,hi in (("num_train",num_train,2,10),("num_test",num_test,1,4)):
        if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or not lo <= n <= hi:
            raise ValueError(name+" must be an integer in range")
    _require(num_train > num_test,"num_train must exceed num_test")
    return int(num_train),int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train,num_test = _counts(num_train,num_test)
    _require(not isinstance(seed,(bool,np.bool_)) and isinstance(seed,(int,np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int,rng.choice(10,4,replace=False)))
    changing_palette = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    mixed = bool(rng.integers(2))
    result,used = dict(train=[],test=[]),set()
    for split,count in (("train",num_train),("test",num_test)):
        for i in range(count):
            colors = tuple(map(int,rng.choice(10,4,replace=False))) if changing_palette else palette
            # The two mandatory witnesses rule out blob counting, ignoring
            # nesting, and selecting a constant ordinal dock even at 2/1.
            kind = i if split == "train" and i < 2 else 2
            slot = i if split == "train" and i < 2 else None
            rot = (rotation+(i % 2 if mixed else 0)) % 4
            pair = _make_pair(rng,kind,2 if split == "train" else 3,slot,colors,rot,used)
            used.add(_key(pair["input"]))
            result[split].append(pair)
        rng.shuffle(result[split])
    return result


def _features(pair):
    decoded = _decode(pair["input"])
    registrations = _registrations(decoded)
    _require(len(registrations) == 1,"ambiguous registration")
    reg = registrations[0]
    info = reg["info"]
    objs = info["objects"]
    depths = []
    for i,o in enumerate(objs):
        if o["kind"] == "C":
            d,p = 1,info["parents"][i]
            while p is not None:
                d,p = d+1,info["parents"][p]
            depths.append(d)
    nonrect = sum(o["kind"] == "C" and len(o["cells"]) != 2*(o["box"][2]-o["box"][0]+o["box"][3]-o["box"][1]) for o in objs)
    return dict(decoded=decoded,registration=reg,contours=len(depths),depth=max(depths),
                blobs=sum(o["kind"] == "B" for o in objs),nonrect=nonrect,
                blob_sizes={len(o["cells"]) for o in objs if o["kind"] == "B"})


def _topology(info,ids=None):
    ids = info["roots"] if ids is None else ids
    return tuple(sorted(("B",) if info["objects"][i]["kind"] == "B"
                        else ("C",_topology(info,info["children"][i])) for i in ids))


def _xor_differs(feature):
    d,reg = feature["decoded"],feature["registration"]
    a,b = d["anchor"],d["mover"]
    wrong = d["payloads"][a] ^ _shift(d["payloads"][b],*reg["shift"])
    try:
        return not np.array_equal(_normalize(_parse_scene(wrong)),reg["mask"])
    except ValueError:
        return True


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"},"puzzle schema")
    _require(all(isinstance(puzzle[s],list) for s in puzzle),"split lists required")
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen,train = set(),[]
    for split in ("train","test"):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {"input","output"},"pair schema")
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            _require(_key(inp) not in seen,"duplicate input")
            seen.add(_key(inp))
            f = _features(pair)
            d,reg = f["decoded"],f["registration"]
            expected = np.where(reg["mask"],d["fg"],d["bg"]).astype(np.int8)
            if d["axis"]:
                expected = expected.T
            _require(np.array_equal(out,expected),"incorrect canonical output")
            _require(set(map(int,np.unique(out))) == {d["bg"],d["fg"]},"temporary marks survived")
            normalized = _parse_scene({tuple(map(int,p)) for p in np.argwhere(out == d["fg"])})
            _require(sum(o["kind"] == "C" for o in normalized["objects"]) == f["contours"],"lost wall")
            _require(sum(o["kind"] == "B" for o in normalized["objects"]) == f["blobs"],"lost blob")
            _require(_topology(normalized) == _topology(reg["info"]),"containment changed")
            _require(np.array_equal(_normalize(normalized),out == d["fg"]),"noncanonical crop or spacing")
            for i,o in enumerate(normalized["objects"]):
                if o["kind"] == "B":
                    _require(len(o["cells"]) == 1,"blob not reduced")
                else:
                    t,l,b,r = o["box"]
                    _require(len(o["cells"]) == 2*(b-t+r-l),"nonrectangular output wall")
                    child = set.union(*(normalized["objects"][j]["cells"] for j in normalized["children"][i]))
                    _require(_box(child) == (t+2,l+2,b-2,r-2),"wrong frame clearance")
            for layer in d["payloads"]:
                _require(not any(_holes(p) for p in _parts(layer)),"a sheet already contains a completed wall")
            _require(len(d["docks"][d["anchor"]]) == (2 if split == "train" else 3),"dock-count shift missing")
            if split == "train":
                train.append(f)
    _require(any(f["depth"] >= 2 for f in train),"nested training witness missing")
    _require(any(f["blobs"] >= 2 for f in train),"multiple-blob witness missing")
    _require({f["registration"]["index"] for f in train} == {0,1},"fixed first/last dock shortcut survives")
    _require(any(not np.array_equal(_mask(f["registration"]["cells"]),f["registration"]["mask"]) for f in train),
             "normalization is not demonstrated")
    _require(any(_xor_differs(f) for f in train),"union versus XOR is not demonstrated")
    _require(any(f["registration"]["index"] not in _nearest_offers(f["decoded"]) for f in train),
             "nearest-dock shortcut survives")


def self_test():
    """42-seed streaming check; bounded memory, no artifact or trace writes."""
    shapes,outputs,structures,palettes,counts,slots,axes = set(),set(),set(),set(),set(),set(),set()
    totals = dict(puzzles=42,pairs=0,nonrect=0,overlap=0,nested=0)
    signatures = set()
    rng = np.random.default_rng(73019)
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple((_key(x["input"]),_key(x["output"])) for s in ("train","test") for x in p[s]))
        for s in ("train","test"):
            for pair in p[s]:
                f = _features(pair)
                d,reg = f["decoded"],f["registration"]
                totals["pairs"] += 1
                totals["nonrect"] += f["nonrect"]
                totals["overlap"] += reg["overlap"] > 0
                totals["nested"] += f["depth"] >= 2
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                structures.add(_key(reg["mask"]))
                palettes.add((d["bg"],d["fg"],d["carrier"],d["marker"]))
                counts.add((f["contours"],f["blobs"]))
                slots.add(reg["index"])
                axes.add(d["axis"])
        # Palette and whole-scene rigid transformations commute with solving.
        pair = p["test"][0]
        perm = rng.permutation(10)
        _require(np.array_equal(solve(perm[pair["input"]]),perm[pair["output"]]),"recoloring failure")
        _require(np.array_equal(solve(np.rot90(pair["input"])),np.rot90(pair["output"])),"rotation failure")
        _require(np.array_equal(solve(pair["input"][:,::-1]),pair["output"][:,::-1]),"reflection failure")
        if seed == 0:
            again = generate_puzzle(seed)
            _require(all(np.array_equal(x[k],y[k]) for s in p for x,y in zip(p[s],again[s]) for k in x),"nondeterminism")
            broken = {s:[{k:v.copy() for k,v in q.items()} for q in p[s]] for s in p}
            out = broken["test"][0]["output"]
            out.flat[0] = (int(out.flat[0])+1)%10
            try:
                validate_generated_puzzle(broken)
            except ValueError:
                pass
            else:
                raise AssertionError("wrong answer accepted")
    _require(len(signatures) == 42,"identical puzzles")
    _require(len(shapes) >= 30 and len(outputs) >= 12 and len(structures) >= 20,"insufficient geometry variation")
    _require(len(palettes) >= 30 and len(counts) >= 6 and axes == {0,1} and slots == {0,1,2},"insufficient structural variation")
    _require(totals["nonrect"] >= 5 and totals["overlap"] >= 30 and totals["nested"] >= 42,"missing interaction coverage")
    # Hold the abstract scene fixed while independently redrawing its geometry.
    redraws = 0
    for i in range(12):
        node = _scene(rng,i % 3)
        drawings,answers = set(),set()
        for _ in range(8):
            metric,expected = _draw(node,rng)
            if max(metric.shape) > 30:
                continue
            try:
                actual = _normalize(_parse_scene({tuple(map(int,p)) for p in np.argwhere(metric)}))
                _require(np.array_equal(actual,expected),"metric redraw changed abstract answer")
            except ValueError:
                continue
            drawings.add(_key(metric))
            answers.add(_key(expected))
        _require(len(answers) <= 1,"one abstract scene has multiple canonical answers")
        redraws += len(drawings) > 1
    _require(redraws >= 6,"insufficient independent metric redraws")
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    totals.update(input_shapes=len(shapes),output_shapes=len(outputs),normalized_outputs=len(structures),
                  palettes=len(palettes),count_combinations=len(counts),scenes_with_multiple_redraws=redraws)
    return totals
