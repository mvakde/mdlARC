"""Color-keyed rigid gates restore a communicating, possibly leaking mould.

Only standard library and NumPy. No files, parent imports, latent state, or RNG
are used by solve. Generation and self_test retain one puzzle at a time.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0,242554020,223514244,329876612,900832476,725785107,52349411,533593519,911160735,648530176,752861426,804552300,393371097,114882196,691220019,784278100,678850485,324097725,153047913,40223383,739397299,677527283,656850571,52989414,642201825,173692722,669497264,313636107,524701304,34612717,964292554,951359444,736044876,506252561,160779311,654980193,17814378,935917348,523634549,317201758,769582688,894445970)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _counts(a, b):
    for name, x, low, high in (("num_train", a, 2, 10), ("num_test", b, 1, 4)):
        _require(isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
                 and low <= int(x) <= high, f"illegal {name}")
    _require(a > b, "num_train must exceed num_test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _components(cells, diagonal=False):
    left, result = set(cells), []
    steps = N4 + ((-1,-1),(-1,1),(1,-1),(1,1)) if diagonal else N4
    while left:
        q = min(left)
        left.remove(q)
        found, todo = {q}, [q]
        while todo:
            r, c = todo.pop()
            for dr, dc in steps:
                p = r+dr, c+dc
                if p in left:
                    left.remove(p)
                    found.add(p)
                    todo.append(p)
        result.append(found)
    return result


def _coords(mask):
    return set(map(tuple, np.argwhere(mask).tolist()))


def _box(cells):
    return min(r for r,c in cells), min(c for r,c in cells), max(r for r,c in cells), max(c for r,c in cells)


def _normal(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r-t,c-l) for r,c in cells))


def _shift(cells, d):
    return {(r+d[0],c+d[1]) for r,c in cells}


def _halo(cells):
    return {(r+dr,c+dc) for r,c in cells for dr in (-1,0,1) for dc in (-1,0,1)}


def _key(g):
    counts = np.bincount(g.ravel(), minlength=10)
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous background")
    bg = int(counts.argmax())
    candidates = []
    h, w = g.shape
    for cells in _components(_coords(g != bg), True):
        t,l,b,r = _box(cells)
        sh, sw = b-t+1, r-l+1
        if len(cells) != sh*sw:
            continue
        lanes = []
        if sh == 2 and sw >= 2:
            if t == 0:
                lanes.append(("top",g[t,l:r+1],g[b,l:r+1]))
            if b == h-1:
                lanes.append(("bottom",g[b,l:r+1],g[t,l:r+1]))
        if sw == 2 and sh >= 2:
            if l == 0:
                lanes.append(("left",g[t:b+1,l],g[t:b+1,r]))
            if r == w-1:
                lanes.append(("right",g[t:b+1,r],g[t:b+1,l]))
        for side, outer, inner in lanes:
            s, f = tuple(map(int,outer)), tuple(map(int,inner))
            if len(set(s)) == len(s) and set(s).isdisjoint(f):
                candidates.append((bg,side,dict(zip(s,f)),cells))
    _require(len(candidates) == 1, "expected one unambiguous local edge key")
    return candidates[0]


def _parse(g):
    bg, side, mapping, keycells = _key(g)
    src, dst = set(mapping), set(mapping.values())
    ink = _coords(g != bg) - keycells
    marker_candidates = {int(g[q]) for q in ink} - src - dst
    _require(len(marker_candidates) == 1, "expected one docking-dot color absent from key")
    marker = marker_candidates.pop()
    base = np.full(g.shape,bg,dtype=np.int8)
    for q in keycells:
        base[q] = g[q]
    gates, mouths, boxes = [], [], {}
    for cells in _components(ink):
        colors = {int(g[q]) for q in cells}
        if colors & dst:
            _require(len(colors & dst) == 1 and colors <= dst | {marker}, "malformed loose gate")
            fill = next(iter(colors & dst))
            pins = {q for q in cells if g[q] == marker}
            _require(len(pins) == 2, "gate must have two sparse docking dots")
            _require(len(_components(cells)) == 1, "gate must be rigid connected ink")
            gates.append({"cells":cells,"pins":pins,"fill":fill})
        else:
            _require(colors <= src | {marker}, "unknown wall ink")
            for q in cells:
                base[q] = g[q]
    allpins = _coords(base == marker)
    assigned = set()
    for color, fill in mapping.items():
        walls = _coords(base == color) - keycells
        _require(bool(walls), "each source names one room")
        boxes[color] = _box(walls)
        pins = {q for q in allpins if any((q[0]+dr,q[1]+dc) in walls for dr,dc in N4)}
        _require(len(pins) == 2 and not pins & assigned, "ambiguous mouth dots")
        assigned |= pins
        mouths.append({"color":color,"fill":fill,"pins":pins})
    _require(assigned == allpins and len(gates) == len(mouths), "unpaired mouth or gate")
    return {"bg":bg,"side":side,"mapping":mapping,"key":keycells,"marker":marker,
            "base":base,"gates":gates,"mouths":mouths,"boxes":boxes}


def _assemble(g, data):
    base, bg, marker = data["base"].copy(), data["bg"], data["marker"]
    choices = {}
    for gate in data["gates"]:
        signature = (gate["fill"], _normal(gate["pins"]))
        _require(signature not in choices, "same-color docking signatures must be distinct")
        choices[signature] = gate
    used, placements = set(), []
    for mouth in data["mouths"]:
        signature = (mouth["fill"], _normal(mouth["pins"]))
        _require(signature in choices and signature not in used, "no unique compatible gate")
        used.add(signature)
        gate = choices[signature]
        a, b = min(mouth["pins"]), min(gate["pins"])
        d = a[0]-b[0], a[1]-b[1]
        placed = _shift(gate["cells"], d)
        _require(_shift(gate["pins"],d) == mouth["pins"], "inexact docking")
        _require(d != (0,0), "gate is already docked")
        for old in gate["cells"]:
            q = old[0]+d[0], old[1]+d[1]
            _require(0 <= q[0] < g.shape[0] and 0 <= q[1] < g.shape[1], "gate outside canvas")
            _require(base[q] == bg or (q in mouth["pins"] and base[q] == marker and g[old] == marker), "payload collision")
            base[q] = g[old]
        placements.append((gate,mouth,d,placed))
    _require(len(used) == len(choices), "unused gate")
    return base, placements


def _flood(assembled, data, fill=True):
    out = assembled.copy()
    h, w = out.shape
    bg, mapping, marker = data["bg"], data["mapping"], data["marker"]
    regions = []
    for cells in _components(_coords(assembled == bg)):
        edge = any(r in (0,h-1) or c in (0,w-1) for r,c in cells)
        boundary = {(r+dr,c+dc) for r,c in cells for dr,dc in N4
                    if 0 <= r+dr < h and 0 <= c+dc < w and assembled[r+dr,c+dc] != bg}
        if edge:
            regions.append((cells,True,None))
            continue
        _require(not boundary & data["key"], "key must not enclose a chamber")
        source_colors = {int(assembled[q]) for q in boundary if int(assembled[q]) in mapping}
        pigments = {mapping[c] for c in source_colors}
        _require(source_colors and len(pigments) == 1, "a sealed region must have one mapped pigment")
        pigment = next(iter(pigments))
        _require(all(int(assembled[q]) in source_colors | {marker,pigment} for q in boundary), "incompatible gate on chamber boundary")
        if fill:
            for q in cells:
                out[q] = pigment
        regions.append((cells,False,pigment))
    return out, regions


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read visible roles, dock rigid gates, and fill only enclosed regions."""
    g = _grid(input_grid)
    data = _parse(g)
    assembled, _ = _assemble(g,data)
    return _flood(assembled,data)[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _local(q, side, h, w):
    r,c = q
    if side == 0:
        return r,c
    if side == 1:
        return c,w-1-r
    if side == 2:
        return h-1-r,c
    return c,r


def _polyomino(rng, count, lesson):
    if lesson:
        return [(0,0),(0,1),(0,2)]
    cells = [(0,0)]
    for _ in range(count-1):
        options = sorted({(r+dr,c+dc) for r,c in cells for dr,dc in N4} - set(cells))
        options = [q for q in options if max(r for r,c in cells+[q])-min(r for r,c in cells+[q]) <= 2
                   and max(c for r,c in cells+[q])-min(c for r,c in cells+[q]) <= 2
                   and all(sum((r+dr,c+dc) in set(cells+[q]) for dr,dc in N4) < 4 for r,c in cells+[q])]
        if not options:
            return None
        cells.append(_pick(rng,options))
    t,l,_,_ = _box(cells)
    return [(r-t,c-l) for r,c in cells]


def _roof(rng, length, depth, style, tooth):
    """A connected, column-monotone barrier; endpoints are its only markers."""
    last = length-3
    levels = [0] * (last+1)
    if style == 1:
        levels[1:-1] = [-1] * max(0,last-1)
    elif style == 2:
        levels[1:-1] = [-int(rng.integers(1,3)) for _ in range(max(0,last-1))]
    cells = {(0,1),(0,length-2)}
    for i in range(1,len(levels)):
        old, new = levels[i-1], levels[i]
        # Horizontal step at the previous height, then vertical within new column.
        for r in range(min(old,new),max(old,new)+1):
            cells.add((r,1+i))
    if tooth and length >= 6:
        c = int(rng.integers(2,length-2))
        start = min(r for r,x in cells if x == c)
        cells |= {(r,c) for r in range(start,depth-1)}
    pins = {(0,1),(0,length-2)}
    return cells, pins


def _construct(rng, count, lesson, bg, marker, colors):
    tiles = _polyomino(rng,count,lesson is not None)
    if tiles is None:
        return None
    nr, nc = max(r for r,c in tiles)+1, max(c for r,c in tiles)+1
    heights = [int(rng.integers(5,9)) for _ in range(nr)]
    widths = [int(rng.integers(5,9)) for _ in range(nc)]
    if lesson is not None:
        widths[0] = int(rng.integers(7,9))
        widths[2] = widths[0]
        if widths[1] == widths[0]:
            widths[1] = 5+(widths[0]-4)%4
    rows, cols = [0], [0]
    for x in heights:
        rows.append(rows[-1]+x)
    for x in widths:
        cols.append(cols[-1]+x)
    src = colors[:count]
    nf = 2 if count <= 3 else int(rng.integers(2,min(3,8-count)+1))
    fills = colors[count:count+nf]
    groups = [0,0,1] if lesson is not None else [int(rng.integers(nf)) for _ in range(count)]
    if lesson is None:
        groups[0], groups[-1] = 0,1
        if count >= 3:
            groups[1] = 0
    mapping = {src[i]:fills[groups[i]] for i in range(count)}
    wall = {}
    specs, gates, mouths, signatures = [], [], [], set()
    for i,(tr,tc) in enumerate(tiles):
        top,left,h,w = rows[tr],cols[tc],heights[tr],widths[tc]
        sides = [s for s,(dr,dc) in enumerate(((-1,0),(0,1),(1,0),(0,-1))) if (tr+dr,tc+dc) not in tiles]
        if not sides:
            return None
        if lesson is not None:
            side = 0
        else:
            possible = [s for s in sides if (mapping[src[i]], ("h" if s%2 == 0 else "v"), (w if s%2 == 0 else h)-3) not in signatures]
            if not possible:
                return None
            side = _pick(rng,possible)
        length, depth = (w,h) if side%2 == 0 else (h,w)
        sig = (mapping[src[i]], "h" if side%2 == 0 else "v",length-3)
        if sig in signatures:
            return None
        signatures.add(sig)
        def trans(q):
            a,b = _local(q,side,h,w)
            return top+a,left+b
        for r in range(h):
            for c in range(w):
                if r in (0,h-1) or c in (0,w-1):
                    wall[top+r,left+c] = src[i]
        pins = {trans((0,1)),trans((0,length-2))}
        for c in range(1,length-1):
            q = trans((0,c))
            if q in pins:
                wall[q] = marker
            else:
                wall.pop(q)
        # Internal peninsulas and full dividers change cavity shape/count.
        divider = (lesson is not None and i == 2) or (lesson is None and rng.random() < .32)
        if divider and length >= 7:
            c = int(rng.integers(3,length-3))
            for r in range(1,depth-1):
                wall[trans((r,c))] = src[i]
        elif lesson is None and rng.random() < .5:
            c = int(rng.integers(1,length-1))
            for r in range(depth-int(rng.integers(2,min(4,depth-1))),depth-1):
                wall[trans((r,c))] = src[i]
        style = (1 if i == 0 else 0) if lesson is not None else int(rng.integers(3))
        cells, lpins = _roof(rng,length,depth,style,lesson is None and not divider and rng.random() < .2)
        gate = {trans(q):(marker if q in lpins else mapping[src[i]]) for q in cells}
        if any(q in wall and q not in pins for q in gate):
            return None
        gates.append(gate)
        mouths.append(pins)
        specs.append((top,left,h,w,side))
    # Double walls isolate unlike pigments; windows join same-pigment rooms.
    for i,(tr,tc) in enumerate(tiles):
        for j in range(i):
            ur,uc = tiles[j]
            if abs(tr-ur)+abs(tc-uc) != 1 or groups[i] != groups[j]:
                continue
            if lesson is None and rng.random() < .22:
                continue
            if tr == ur:
                lefti,righti = (i,j) if tc < uc else (j,i)
                t,l,h,w,_ = specs[lefti]
                r = t+int(rng.integers(1,h-1))
                for q in ((r,l+w-1),(r,l+w)):
                    wall.pop(q,None)
            else:
                upper,lower = (i,j) if tr < ur else (j,i)
                t,l,h,w,_ = specs[upper]
                c = l+int(rng.integers(1,w-1))
                for q in ((t+h-1,c),(t+h,c)):
                    wall.pop(q,None)
    vent_candidates = [0] if lesson == 1 else ([] if lesson == 0 else list(range(count)))
    for i in vent_candidates:
        if lesson is None and (groups[i] == 1 or rng.random() >= .35):
            continue
        tr,tc = tiles[i]
        t,l,h,w,mouthside = specs[i]
        sides = [s for s,(dr,dc) in enumerate(((-1,0),(0,1),(1,0),(0,-1)))
                 if (tr+dr,tc+dc) not in tiles and s != mouthside]
        if sides:
            s = 2 if lesson is not None else _pick(rng,sides)
            length = w if s%2 == 0 else h
            a,b = _local((0,int(rng.integers(1,length-1))),s,h,w)
            wall.pop((t+a,l+b),None)
    # A cap may extend outwards; no non-dot overlap is legal.
    occupied = dict(wall)
    for gate,pins in zip(gates,mouths):
        for q,c in gate.items():
            if q in occupied and not (q in pins and occupied[q] == c == marker):
                return None
            occupied[q] = c
    return wall,gates,mapping,occupied


def _place(rng, built, bg, marker):
    wall,gates,mapping,assembled = built
    t,l,b,r = _box(assembled)
    bh,bw = b-t+1,r-l+1
    if bh > 27 or bw > 27:
        return None
    h = int(rng.integers(max(20,bh+3),31))
    w = int(rng.integers(max(20,bw+3),31))
    n = len(mapping)
    start = int(rng.integers(1,w-n))
    order = list(map(int,rng.permutation(n)))
    sources = list(mapping)
    key = {(lane,start+k):(sources[i] if lane == 0 else mapping[sources[i]]) for k,i in enumerate(order) for lane in (0,1)}
    keyhalo = _halo(key)
    positions = [(a-t,c-l) for a in range(h-bh+1) for c in range(w-bw+1)]
    rng.shuffle(positions)
    offset = next((d for d in positions if not _shift(assembled,d) & keyhalo),None)
    if offset is None:
        return None
    fixed = {(a+offset[0],c+offset[1]):v for (a,c),v in wall.items()}
    final = {(a+offset[0],c+offset[1]):v for (a,c),v in assembled.items()}
    g = np.full((h,w),bg,dtype=np.int8)
    for q,c in {**fixed,**key}.items():
        g[q] = c
    forbidden = _halo(set(fixed)|set(key))
    for i in map(int,rng.permutation(len(gates))):
        gate = gates[i]
        gt,gl,gb,gr = _box(gate)
        candidates = [(a-gt,c-gl) for a in range(h-(gb-gt)) for c in range(w-(gr-gl))]
        rng.shuffle(candidates)
        d = next((d for d in candidates if d != offset and not _shift(gate,d) & forbidden),None)
        if d is None:
            return None
        for (a,c),color in gate.items():
            g[a+d[0],c+d[1]] = color
        forbidden |= _halo(_shift(gate,d))
    if np.count_nonzero(g == bg) <= max(np.count_nonzero(g == c) for c in range(10) if c != bg):
        return None
    expected = np.full((h,w),bg,dtype=np.int8)
    for q,c in {**final,**key}.items():
        expected[q] = c
    data = _parse(g)
    restored, _ = _assemble(g,data)
    if not np.array_equal(expected,restored):
        return None
    output, regions = _flood(expected,data)
    if not any(not edge for _,edge,_ in regions):
        return None
    return {"input":g,"output":output}


def _features(pair):
    g, out = _grid(pair["input"]), _grid(pair["output"])
    data = _parse(g)
    restored, placements = _assemble(g,data)
    expected, regions = _flood(restored,data)
    _require(np.array_equal(expected,out), "incorrect output")
    bg, marker = data["bg"],data["marker"]
    _require(np.array_equal(out[data["base"] != bg],data["base"][data["base"] != bg]), "stationary ink or key changed")
    _require(np.count_nonzero(g == marker) == 2*np.count_nonzero(out == marker), "docking dots must overlap exactly twice")
    _require(len(_components(_coords(restored != bg)-data["key"])) == 1, "restored mould must be connected")
    sealed = [cells for cells,edge,_ in regions if not edge]
    _require(sealed, "scene must show a sealed region")
    # Before docking every room is open: filling input first cannot work.
    _require(all(any(r in (0,g.shape[0]-1) or c in (0,g.shape[1]-1) for r,c in comp)
                 for comp in _components(_coords(g == bg))), "input has a presealed cavity")
    outside = set().union(*(cells for cells,edge,_ in regions if edge))
    nonlocal_leak = False
    vents = 0
    boxes = data["boxes"]
    for color,(t,l,b,r) in boxes.items():
        external = False
        pins_for_room = next(m["pins"] for m in data["mouths"] if m["color"] == color)
        mouth_edge = next((axis,value) for axis,value in ((0,t),(0,b),(1,l),(1,r))
                          if all(q[axis] == value for q in pins_for_room))
        for a,c in {(t,c) for c in range(l+1,r)} | {(b,c) for c in range(l+1,r)} | {(a,l) for a in range(t+1,b)} | {(a,r) for a in range(t+1,b)}:
            if restored[a,c] != bg or (a,c)[mouth_edge[0]] == mouth_edge[1]:
                continue
            for dr,dc in N4:
                q = a+dr,c+dc
                if t <= q[0] <= b and l <= q[1] <= r:
                    continue
                if not any(tt <= q[0] <= bb and ll <= q[1] <= rr for tt,ll,bb,rr in boxes.values()):
                    external = True
        vents += int(external)
        interior = {(a,c) for a in range(t+1,b) for c in range(l+1,r) if restored[a,c] == bg}
        nonlocal_leak |= bool(interior & outside) and not external
    keys = [(m["fill"],_normal(m["pins"])) for m in data["mouths"]]
    color_collision = any(a[1] == b[1] and a[0] != b[0] for i,a in enumerate(keys) for b in keys[:i])
    geometry_witness = any(a[0] == b[0] and a[1] != b[1] for i,a in enumerate(keys) for b in keys[:i])
    windows = 0
    room_boxes = list(boxes.values())
    for i,(t,l,b,r) in enumerate(room_boxes):
        for tt,ll,bb,rr in room_boxes[:i]:
            if r+1 == ll or rr+1 == l:
                c = r if r+1 == ll else rr
                windows += int(any(restored[a,c] == restored[a,c+1] == bg
                                   for a in range(max(t,tt)+1,min(b,bb))))
            if b+1 == tt or bb+1 == t:
                a = b if b+1 == tt else bb
                windows += int(any(restored[a,c] == restored[a+1,c] == bg
                                   for c in range(max(l,ll)+1,min(r,rr))))
    irregular = sum(len(cells) != (_box(cells)[2]-_box(cells)[0]+1)*(_box(cells)[3]-_box(cells)[1]+1) for cells in sealed)
    multi_source = sum(len({int(restored[r+dr,c+dc]) for r,c in cells for dr,dc in N4
                            if 0 <= r+dr < g.shape[0] and 0 <= c+dc < g.shape[1] and int(restored[r+dr,c+dc]) in data["mapping"]}) > 1 for cells in sealed)
    return {"rooms":len(boxes),"shape":g.shape,"side":data["side"],"background":bg,"marker":marker,
            "closed_regions":len(sealed),"irregular":irregular,"multi_source":multi_source,
            "vents":vents,"nonlocal_leak":nonlocal_leak,"color_collision":color_collision,
            "geometry_witness":geometry_witness,"windows":windows,
            "room_boxes":tuple(sorted((b-t+1,r-l+1) for t,l,b,r in boxes.values())),
            "gate_shapes":tuple(_normal(gate["cells"]) for gate,_,_,_ in placements),
            "pin_shapes":tuple(k[1] for k in keys),"filled_cells":sum(map(len,sealed)),
            "destinations":len(set(data["mapping"].values()))}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a,b = _counts(num_train,num_test)
    rng = np.random.default_rng(seed)
    puzzle = {"train":[],"test":[]}
    seen = set()
    first_rotation = int(rng.integers(4))
    palette_order = list(map(int,rng.permutation(10)))
    for split,n in (("train",a),("test",b)):
        for i in range(n):
            lesson = i if split == "train" and i < 2 else None
            for attempt in range(240):
                count = 3 if lesson is not None else int(rng.integers(2,5) if split == "train" else rng.integers(4,6))
                # Role schedules vary by example; no color convention is learned.
                palette = list(np.roll(palette_order,int(rng.integers(10))))
                bg,marker = map(int,palette[:2])
                colors = list(map(int,palette[2:]))
                built = _construct(rng,count,lesson,bg,marker,colors)
                if built is None:
                    continue
                try:
                    pair = _place(rng,built,bg,marker)
                    if pair is None:
                        continue
                    turns = (first_rotation+i) % 4 if split == "train" else (first_rotation+2+i) % 4
                    pair = {k:np.rot90(v,turns).copy() for k,v in pair.items()}
                    if rng.random() < .5:
                        pair = {k:v[:,::-1].copy() for k,v in pair.items()}
                    key = (pair["input"].shape,pair["input"].tobytes())
                    if key in seen:
                        continue
                    features = _features(pair)
                    if lesson == 0 and not (features["multi_source"] and features["color_collision"] and features["irregular"]):
                        continue
                    if lesson == 1 and not (features["nonlocal_leak"] and features["vents"]):
                        continue
                    if split == "test" and not features["windows"]:
                        continue
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError(f"bounded construction failed for {split} example {i}")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in ("train","test")), "splits must be lists")
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen, records = set(), {"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {"input","output"}, "bad pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _require(g.shape == out.shape, "canvas changed")
            key = (g.shape,g.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            records[split].append(_features(pair))
    train = records["train"]
    _require(any(f["color_collision"] for f in train), "missing color-lookup docking witness")
    _require(any(f["geometry_witness"] for f in train), "missing geometric docking witness")
    _require(any(f["multi_source"] for f in train), "missing communicating sealed chambers")
    _require(any(f["nonlocal_leak"] for f in train), "missing remote-vent witness")
    _require(any(f["irregular"] for f in train), "missing nonrectangular enclosure witness")
    _require(len({f["side"] for f in train}) >= 2, "key direction requires two training sides")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for p in puzzle[split]:
            for k in ("input","output"):
                g = np.asarray(p[k],dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def _hand_checks():
    """Small explicit expectations, separate from procedural construction."""
    g = np.zeros((20,22),dtype=np.int8)
    g[0,2:4],g[1,2:4] = [1,2],[3,4]
    for color,left in ((1,1),(2,8)):
        g[5:10,left] = g[5:10,left+6] = color
        g[5,left:left+7] = g[9,left:left+7] = color
        g[5,left+1:left+6] = [9,0,0,0,9]
    g[9,4] = 0  # An exterior vent drains only the first room.
    g[14,2:7],g[16,12:17] = [9,3,3,3,9],[9,4,4,4,9]
    expected = g.copy()
    expected[14,2:7] = expected[16,12:17] = 0
    expected[5,2:7],expected[5,9:14] = [9,3,3,3,9],[9,4,4,4,9]
    expected[6:9,9:14] = 4
    assert np.array_equal(solve(g),expected)
    closed, filled = g.copy(),expected.copy()
    closed[9,4] = filled[9,4] = 1
    filled[6:9,2:7] = 3
    assert np.array_equal(solve(closed),filled)
    swapped, changed = g.copy(),expected.copy()
    swapped[1,2:4] = changed[1,2:4] = [4,3]
    changed[5,2:7],changed[5,9:14] = [9,4,4,4,9],[9,3,3,3,9]
    changed[6:9,9:14] = 3
    assert np.array_equal(solve(swapped),changed)
    assert not np.array_equal(expected,changed)
    # A corner opening offers only diagonal escape. Four-neighbor background
    # is the specified rule even on shapes beyond the sampler's plain corners.
    diagonal, still_filled = g.copy(),expected.copy()
    diagonal[9,8] = still_filled[9,8] = 0
    assert np.array_equal(solve(diagonal),still_filled)
    broken = g.copy()
    broken[14,2] = 0
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("a missing gate dot must be rejected")


def self_test():
    """Bounded, in-memory checks over the exact 42 host seeds plus mutations."""
    _hand_checks()
    digests, shapes, boxes, gates, pins = set(), set(), set(), set(), set()
    totals = Counter()
    backgrounds, markers, sides, counts = set(),set(),set(),set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        digests.add(_digest(p))
        for split in ("train","test"):
            for pair in p[split]:
                f = _features(pair)
                shapes.add(f["shape"])
                boxes.update(f["room_boxes"])
                gates.update(f["gate_shapes"])
                pins.update(f["pin_shapes"])
                backgrounds.add(f["background"])
                markers.add(f["marker"])
                sides.add(f["side"])
                counts.add(f["rooms"])
                totals.update({k:int(bool(f[k])) for k in ("irregular","multi_source","nonlocal_leak","color_collision")})
                # Array-like inputs and caller-owned data are both supported.
                before = pair["input"].copy()
                assert np.array_equal(solve(before.tolist()),pair["output"])
                assert np.array_equal(before,pair["input"])
    assert len(digests) == 42
    assert len(shapes) >= 30 and len(boxes) >= 10 and len(gates) >= 40
    assert len(pins) >= 6 and counts >= {2,3,4,5}
    assert len(backgrounds) >= 8 and len(markers) >= 8 and len(sides) == 4
    assert all(totals[k] >= 42 for k in totals)
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p = generate_puzzle(17,a,b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    bad = {s:[{k:v.copy() for k,v in q.items()} for q in p[s]] for s in p}
    data = _parse(bad["test"][0]["input"])
    q = min(data["key"])
    bad["test"][0]["output"][q] = data["bg"]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("changed key accepted")
    # Suppressing either docking or enclosure changes the expected answer.
    for pair in p["train"][:2]:
        g = pair["input"]
        data = _parse(g)
        restored,_ = _assemble(g,data)
        assert not np.array_equal(restored,pair["output"])
        assert not np.array_equal(g,pair["output"])
    return {"seeds":42,"pairs":210,"distinct_puzzles":len(digests),"grid_shapes":len(shapes),
            "room_box_shapes":len(boxes),"gate_shapes":len(gates),"pin_signatures":len(pins),
            "backgrounds":len(backgrounds),"markers":len(markers),"witness_pairs":dict(totals)}
