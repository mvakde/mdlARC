"""Seam-built docking: assemble a registration field, then dock rigid pieces.

All semantic roles are visible. There is no training-dependent color dictionary.
Construction starts with complete final pieces; the solver instead reads only
the fragmented tapes and scrambled pieces in its input. No files are written.
"""
from __future__ import annotations

import hashlib
import numbers
import numpy as np


HOST_SEEDS = (0,543113206,851194558,528972099,103041822,925011419,262380992,
10961306,379986104,502754582,59226819,952972835,377171288,411526694,
579096878,711118232,458085470,231694375,765893323,230457066,766914458,
153325893,559973282,819788558,10261513,73509289,201690212,853253586,
583316355,930287603,316639693,328334114,982481872,609600782,6439525,
20906642,276098200,518779768,40191188,119479572,889865358,589637382)
NEIGHBORS = tuple((r,c) for r in (-1,0,1) for c in (-1,0,1) if r or c)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, component = [first], {first}
        while stack:
            r,c = stack.pop()
            for dr,dc in NEIGHBORS:
                q = r+dr,c+dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _bbox(points):
    return (min(r for r,c in points), max(r for r,c in points)+1,
            min(c for r,c in points), max(c for r,c in points)+1)


def _normalize(cells):
    r0,_,c0,_ = _bbox(cells)
    return {(r-r0,c-c0): int(v) for (r,c),v in cells.items()}


def _transform(cells, t):
    result = {}
    for (r,c),v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r,c = c,-r
        result[r,c] = v
    return _normalize(result)


def _shift(cells, r0, c0):
    return {(r+r0,c+c0): v for (r,c),v in cells.items()}


def _signature(cells):
    return tuple(sorted(_normalize(cells).items()))


def _halo(points):
    return {(r+dr,c+dc) for r,c in points for dr in (-1,0,1) for dc in (-1,0,1)}


def _frame(canvas, compass, bg):
    h,w = canvas.shape
    out = np.full((h+2,w+2), bg, dtype=np.int8)
    out[1:-1,1:-1] = canvas
    out[0,1:-1],out[1:-1,-1],out[-1,1:-1],out[1:-1,0] = compass
    return out


def _assemble(blocks, reg, bg, align="center"):
    width = max(b.shape[1] for b in blocks)
    out = np.full((2*len(blocks),width), bg, dtype=np.int8)
    for i,b in enumerate(blocks):
        left = (width-b.shape[1])//2 if align == "center" else 0
        out[2*i:2*i+2,left:left+b.shape[1]][b == reg] = reg
    return out


def _read(input_grid):
    grid = _grid(input_grid)
    _require(min(grid.shape) >= 8, "frame too small")
    corners = [int(grid[0,0]),int(grid[0,-1]),int(grid[-1,-1]),int(grid[-1,0])]
    bg = next((c for c in set(corners) if corners.count(c) == 3), None)
    _require(bg is not None, "three background corners and one origin required")
    origin_index = next(i for i,c in enumerate(corners) if c != bg)
    origin = corners[origin_index]
    sides = (grid[0,1:-1],grid[1:-1,-1],grid[-1,1:-1],grid[1:-1,0])
    _require(all(np.all(s == s[0]) for s in sides), "nonuniform frame side")
    bands = tuple(int(s[0]) for s in sides)
    _require(len(set(bands) | {bg,origin}) == 6, "frame roles collide")
    inside = grid[1:-1,1:-1]
    _require(all(np.all(edge == bg) for edge in (inside[0],inside[-1],inside[:,0],inside[:,-1])),
             "input objects must be separated from the frame")
    objects = _components(zip(*np.nonzero(inside != bg)))
    strips,sources = [],[]
    for points in objects:
        cells = {p:int(inside[p]) for p in points}
        if set(cells.values()) & set(bands):
            sources.append(cells)
        else:
            r0,r1,c0,c1 = _bbox(points)
            _require(len(points) == (r1-r0)*(c1-c0), "tape is not a full rectangle")
            _require(min(r1-r0,c1-c0) == 2 and max(r1-r0,c1-c0) >= 4, "tape dimensions")
            strips.append((r0,r1,c0,c1,inside[r0:r1,c0:c1].copy()))
    _require(3 <= len(strips) <= 14 and 1 <= len(sources) <= 4, "tape/piece counts")
    axes = {int(s[1]-s[0] > 2) for s in strips}
    _require(len(axes) == 1, "tapes must have one common long axis")
    axis = axes.pop()
    body_colors = {int(s[4][r,c]) for s in strips for r,c in ((0,0),(0,-1),(-1,0),(-1,-1))}
    _require(len(body_colors) == 1, "tape corners must identify one carrier color")
    body = body_colors.pop()
    mark_colors = set().union(*(set(map(int,s[4].flat)) for s in strips)) - {body}
    _require(len(mark_colors) == 1, "tapes must encode one registration color")
    reg = mark_colors.pop()
    _require(len(set(bands) | {bg,origin,body,reg}) == 8, "semantic roles collide")
    for source in sources:
        _require({body,reg} <= set(source.values()) <= {body,reg,*bands}, "invalid piece colors")
        _require(_label_directions(source, bands) is not None, "unclear label direction")
    bottom = origin_index in (2,3)
    right = origin_index in (1,2)
    reverse_fast = bottom if axis else right
    reverse_slow = right if axis else bottom
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    lanes = {}
    for s in strips:
        lanes.setdefault(s[2] if axis else s[0], []).append(s)
    vs = sorted(lanes)
    _require(all(b-a >= 3 for a,b in zip(vs,vs[1:])), "overlapping tape lanes")
    ordered,groups = [],[]
    for v in sorted(lanes, reverse=reverse_slow):
        lane = sorted(lanes[v],key=lambda s:s[0] if axis else s[2],reverse=reverse_fast)
        groups.append(list(range(len(ordered),len(ordered)+len(lane))))
        ordered.extend(lane)
    blocks = [np.rot90(s[4],rotation).copy() for s in ordered]
    lengths = [b.shape[1] for b in blocks]
    _require(len({n % 2 for n in lengths}) == 1, "mixed tape length parity")
    _require(2*len(blocks)+2 <= 30, "assembled canvas too tall")
    target_grid = _assemble(blocks,reg,bg)
    targets = _components(zip(*np.nonzero(target_grid == reg)))
    _require(len(targets) == len(sources), "assembled stencil/piece count mismatch")
    compass = tuple(bands[(i+rotation) % 4] for i in range(4))
    return {"grid":grid,"bg":bg,"origin":origin,"body":body,"reg":reg,
            "bands":bands,"compass":compass,"axis":axis,"bottom":bottom,"right":right,
            "rotation":rotation,"sources":sources,"blocks":blocks,"groups":groups,
            "boxes":ordered,"targets":targets,"target_grid":target_grid,
            "lengths":lengths,"shape":target_grid.shape}


def _label_directions(cells, bands):
    core = {p for p,v in cells.items() if v not in bands}
    if not core:
        return None
    r0,r1,c0,c1 = _bbox(core)
    labels = []
    for (r,c),v in cells.items():
        if v not in bands:
            continue
        ds = [i for i,yes in enumerate((r < r0,c >= c1,r >= r1,c < c0)) if yes]
        if len(ds) != 1:
            return None
        labels.append((ds[0],v))
    return labels if labels else None


def _choices(info, use_labels=True):
    h,w = info["shape"]
    choices = []
    for source in info["sources"]:
        candidates = {}
        for t in range(8):
            cells = _transform(source,t)
            labels = _label_directions(cells,info["bands"])
            if labels is None or (use_labels and any(info["compass"][d] != v for d,v in labels)):
                continue
            marks = {p for p,v in cells.items() if v == info["reg"]}
            mr,_,mc,_ = _bbox(marks)
            for j,target in enumerate(info["targets"]):
                tr,_,tc,_ = _bbox(target)
                dr,dc = tr-mr,tc-mc
                if {(r+dr,c+dc) for r,c in marks} != target:
                    continue
                placed = _shift(cells,dr,dc)
                if any(not (0 <= r < h and 0 <= c < w) for r,c in placed):
                    continue
                signature = j,tuple(sorted(placed.items()))
                candidates.setdefault(signature,(j,placed,t))
        choices.append(list(candidates.values()))
    return choices


def _maps(info, use_labels=True, limit=2):
    choices = _choices(info,use_labels)
    order = sorted(range(len(choices)),key=lambda i:len(choices[i]))
    results = {}
    def search(depth,used,occupied,transforms):
        if len(results) >= limit:
            return
        if depth == len(order):
            results.setdefault(tuple(sorted(occupied.items())),(occupied,transforms))
            return
        for target,cells,t in choices[order[depth]]:
            if target not in used and not (occupied.keys() & cells.keys()):
                search(depth+1,used | {target},{**occupied,**cells},transforms+(t,))
    search(0,set(),{},())
    return list(results.values())


def _prediction(info):
    answers = _maps(info)
    _require(len(answers) == 1, "input does not determine one docked pixel map")
    canvas = np.full(info["shape"],info["bg"],dtype=np.int8)
    for p,v in answers[0][0].items():
        canvas[p] = v
    return _frame(canvas,info["compass"],info["bg"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the key and fragments; enumerate rigid docking from visible pixels."""
    return _prediction(_read(input_grid))


def _a_differences(info):
    """Counterfactual stencil maps for common shortcuts; no expected labels used."""
    blocks,groups = info["blocks"],info["groups"]
    orders = {
        "reverse_fast":[i for g in groups for i in g[::-1]],
        "reverse_slow":[i for g in groups[::-1] for i in g],
        "fast_major":sorted(range(len(blocks)),key=lambda i:info["boxes"][i][0 if info["axis"] else 2],
                            reverse=info["bottom"] if info["axis"] else info["right"]),
    }
    expected = info["target_grid"]
    result = {name:not np.array_equal(expected,_assemble([blocks[i] for i in order],info["reg"],info["bg"]))
              for name,order in orders.items()}
    result["left_align"] = not np.array_equal(expected,_assemble(blocks,info["reg"],info["bg"],"left"))
    result["reflect_thickness"] = not np.array_equal(expected,_assemble([b[::-1] for b in blocks],info["reg"],info["bg"]))
    return result


def _physical(a, configuration):
    axis,bottom,right = configuration
    if axis:
        a = a.T
    if bottom:
        a = a[::-1]
    if right:
        a = a[:,::-1]
    return a.copy()


def _pattern(rng,h,w,kind=None):
    if kind == "diagonal":
        return {(0,0),(1,1)}
    if kind == "domino":
        return {(0,0),(0,1)} if rng.random() < .5 else {(0,0),(1,0)}
    size = int(rng.choice([1,2,3,4,5,6],p=[.17,.29,.20,.16,.11,.07]))
    points = {(int(rng.integers(h)),int(rng.integers(w)))}
    for _ in range(40):
        if len(points) >= size:
            break
        frontier = sorted({(r+dr,c+dc) for r,c in points for dr,dc in NEIGHBORS
                           if 0 <= r+dr < h and 0 <= c+dc < w}-points)
        if not frontier:
            break
        points.add(frontier[int(rng.integers(len(frontier)))])
    return points


def _piece(rng,body,reg,compass,kind=None):
    h,w = int(rng.integers(2,6)),int(rng.integers(2,6))
    marks = _pattern(rng,h,w,kind)
    # A simple path tree keeps gaps real while establishing the core envelope.
    endpoints = sorted(marks)+[(0,int(rng.integers(w))),(h-1,int(rng.integers(w))),
                              (int(rng.integers(h)),0),(int(rng.integers(h)),w-1)]
    selected = {int(rng.integers(4))}
    selected.add((next(iter(selected))+1) % 4)
    for d in range(4):
        if rng.random() < .25:
            selected.add(d)
    labels = {}
    for d in sorted(selected):
        extent = w if d % 2 == 0 else h
        positions = rng.choice(extent,size=int(rng.integers(1,min(2,extent)+1)),replace=False)
        for pos in positions:
            p,q = (((-1,int(pos)),(0,int(pos))),((int(pos),w),(int(pos),w-1)),
                   ((h,int(pos)),(h-1,int(pos))),((int(pos),-1),(int(pos),0)))[d]
            labels[p] = compass[d]
            endpoints.append(q)
    core = {min(marks)}
    for end in endpoints:
        nearest = min(abs(r-end[0])+abs(c-end[1]) for r,c in core)
        anchors = sorted(p for p in core if abs(p[0]-end[0])+abs(p[1]-end[1]) == nearest)
        r,c = anchors[int(rng.integers(len(anchors)))]
        while (r,c) != end:
            if r != end[0] and (c == end[1] or rng.random() < .5):
                r += 1 if end[0] > r else -1
            else:
                c += 1 if end[1] > c else -1
            core.add((r,c))
    for r in range(h):
        for c in range(w):
            if rng.random() < .16:
                core.add((r,c))
    return _normalize({**{p:body for p in core},**{p:reg for p in marks},**labels})


def _pack(rng,blocks,sources,bg,configuration):
    """Pack variable shelves; source pieces can share or interrupt tape lanes."""
    axis,bottom,right = configuration
    flip = (bottom != right) if not axis else (bottom == right)
    items = [b[::-1].copy() if flip else b.copy() for b in blocks]
    for source in sources:
        _,h,_,w = _bbox(source)
        a = np.full((h,w),bg,dtype=np.int8)
        for p,v in source.items():
            a[p] = v
        items.append(a)
    shelves = []
    capacity = int(rng.integers(25,27))
    # Consecutive tape occurrences share shelves. Their sequence is intentional;
    # source-only shelves may be inserted anywhere and sources may share a lane.
    for i in range(len(blocks)):
        h,w = items[i].shape
        if shelves and shelves[-1]["width"]+1+w <= capacity and rng.random() < .85:
            s = shelves[-1]
            s["items"].append(i)
            s["width"] += w+1
        else:
            shelves.append({"items":[i],"width":w,"height":h})
    source_order = sorted(rng.permutation(np.arange(len(blocks),len(items))),
                          key=lambda i:-items[i].shape[0])
    for i in source_order:
        h,w = items[i].shape
        fits = [s for s in shelves if s["width"]+1+w <= capacity]
        # Prefer sharing an already tall shelf, but retain mixed tape/piece rows.
        fits.sort(key=lambda s:max(0,h-s["height"])+float(rng.random()))
        if fits and (fits[0]["height"] >= h or rng.random() < .55):
            s = fits[0]
            s["items"].append(int(i))
            s["width"] += w+1
            s["height"] = max(s["height"],h)
        else:
            shelves.insert(int(rng.integers(len(shelves)+1)),
                           {"items":[int(i)],"width":w,"height":h})
    min_h = sum(s["height"] for s in shelves)+len(shelves)+1
    min_w = max(s["width"] for s in shelves)+2
    if min_h > 28 or min_w > 28:
        return None
    H = min_h+int(rng.integers(29-min_h))
    W = min_w+int(rng.integers(29-min_w))
    a = np.full((H,W),bg,dtype=np.int8)
    gaps = np.ones(len(shelves)+1,dtype=int)
    for _ in range(H-min_h):
        gaps[int(rng.integers(len(gaps)))] += 1
    r = int(gaps[0])
    for j,s in enumerate(shelves):
        # Move sources among the tape occurrences without permuting the tapes.
        tape_ids = [i for i in s["items"] if i < len(blocks)]
        source_ids = [i for i in s["items"] if i >= len(blocks)]
        for i in source_ids:
            tape_ids.insert(int(rng.integers(len(tape_ids)+1)),i)
        s["items"] = tape_ids
        horizontal = np.ones(len(s["items"])+1,dtype=int)
        for _ in range(W-s["width"]-2):
            horizontal[int(rng.integers(len(horizontal)))] += 1
        c = int(horizontal[0])
        tape_offset = int(rng.integers(s["height"]-1))
        for k,i in enumerate(s["items"]):
            h,w = items[i].shape
            off = tape_offset if i < len(blocks) else int(rng.integers(s["height"]-h+1))
            a[r+off:r+off+h,c:c+w] = items[i]
            c += w+int(horizontal[k+1])
        r += s["height"]+int(gaps[j+1])
    return _physical(a,configuration)


def _construct(rng,palette,configuration,n,count,witness):
    bg,origin,body,reg,*bands = palette
    axis,bottom,right = configuration
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    compass = tuple(bands[(i+rotation) % 4] for i in range(4))
    h = 2*n
    w = int(rng.choice([9,10,11,12,13,14,16]))
    finals,occupied = [],set()
    for i in range(count):
        kind = ("diagonal" if i == 0 else "domino") if witness == 0 else None
        piece = _piece(rng,body,reg,compass,kind)
        _,ph,_,pw = _bbox(piece)
        origins = [(r,c) for r in range(h-ph+1) for c in range(w-pw+1)]
        rng.shuffle(origins)
        found = None
        for r,c in origins:
            placed = _shift(piece,r,c)
            marks = {p for p,v in placed.items() if v == reg}
            if placed.keys() & occupied or any(not (1 <= x < w-1) for _,x in marks):
                continue
            if witness is not None and i == 0 and len({r//2 for r,c in marks}) < 2:
                continue
            found = placed
            break
        if found is None:
            return None
        finals.append(found)
        occupied.update(_halo(found))
    mark_canvas = np.full((h,w),bg,dtype=np.int8)
    expected_canvas = mark_canvas.copy()
    for final in finals:
        for p,v in final.items():
            expected_canvas[p] = v
            if v == reg:
                mark_canvas[p] = reg
    blocks,lengths = [],[]
    for j in range(n):
        chunk = mark_canvas[2*j:2*j+2]
        options = [L for L in range(4+w%2,w+1,2)
                   if not np.any(chunk[:,:((w-L)//2)+1] == reg)
                   and not np.any(chunk[:,w-((w-L)//2)-1:] == reg)]
        if not options:
            return None
        lengths.append(int(rng.choice(options)))
    lengths[int(rng.integers(n))] = w
    if witness is not None and not any(L < w and np.any(mark_canvas[2*i:2*i+2] == reg)
                                       for i,L in enumerate(lengths)):
        return None
    for j,L in enumerate(lengths):
        left = (w-L)//2
        b = np.full((2,L),body,dtype=np.int8)
        b[mark_canvas[2*j:2*j+2,left:left+L] == reg] = reg
        blocks.append(b)
    # Geometry is independently packed while retaining the intended tape order.
    scrambled = [_transform(final,int(rng.integers(8))) for final in finals]
    scene = _pack(rng,blocks,scrambled,bg,configuration)
    if scene is None:
        return None
    inp = _frame(scene,bands,bg)
    origin_pos = (-1 if bottom else 0,-1 if right else 0)
    inp[origin_pos] = origin
    try:
        info = _read(inp)
        # Certify both the packed order and proper (rather than mirrored) rotation.
        if info["lengths"] != lengths:
            return None
        for b,actual in zip(blocks,info["blocks"]):
            if not np.array_equal(b,actual):
                return None
        predicted = _prediction(info)
        out = _frame(expected_canvas,compass,bg)
        if not np.array_equal(predicted,out):
            return None
        transforms = _maps(info)[0][1]
        if witness == 0:
            if not all(_a_differences(info).values()) or not any(0 < t < 4 for t in transforms):
                return None
            if len(_maps(info,use_labels=False)) < 2:
                return None
        if witness == 1 and not any(t >= 4 for t in transforms):
            return None
        return {"input":inp,"output":out}
    except ValueError:
        return None


def _counts(num_train,num_test):
    for v,lo,hi in ((num_train,2,10),(num_test,1,4)):
        _require(isinstance(v,numbers.Integral) and not isinstance(v,(bool,np.bool_)) and lo <= v <= hi,
                 "illegal example count")
    _require(num_train > num_test,"num_train must exceed num_test")
    return int(num_train),int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train,num_test = _counts(num_train,num_test)
    _require(isinstance(seed,numbers.Integral) and not isinstance(seed,(bool,np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    configs = [(a,b,c) for a in (False,True) for b in (False,True) for c in (False,True)]
    first = configs[int(rng.integers(8))]
    second = tuple(not v for v in first)
    remaining = [c for c in configs if c not in (first,second)]
    rng.shuffle(remaining)
    configs = [first,second]+remaining
    # All tests recombine directions; a sampled regime also increases tape count.
    increase = bool(rng.integers(2))
    puzzle,seen = {"train":[],"test":[]},set()
    for split,number in (("train",num_train),("test",num_test)):
        for i in range(number):
            witness = i if split == "train" and i < 2 else None
            configuration = (configs[i] if witness is not None else
                             configs[int(rng.integers(6))] if split == "train" else
                             configs[int(rng.integers(6,8))])
            for attempt in range(256):
                palette = [int(c) for c in rng.permutation(10)[:8]]
                n = int(rng.integers(5,8) if witness is not None else
                        rng.integers(8,11) if split == "test" and increase else rng.integers(3,10))
                count = 2 if witness == 0 else int(rng.integers(1,min(4,n//2)+1))
                pair = _construct(rng,palette,configuration,n,count,witness)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"},"puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in ("train","test")),"split must be a list")
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen,training = set(),[]
    for split in ("train","test"):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {"input","output"},"pair keys")
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            _require(_key(inp) not in seen,"duplicate input")
            seen.add(_key(inp))
            info = _read(inp)
            _require(np.array_equal(_prediction(info),out),"incorrect docked output")
            _require(not np.array_equal(inp,out),"identity pair")
            canvas = out[1:-1,1:-1]
            _require(np.array_equal(canvas == info["reg"],info["target_grid"] == info["reg"]),"lost registration marks")
            for color in (info["body"],*info["bands"]):
                source_count = sum(list(s.values()).count(color) for s in info["sources"])
                _require(np.count_nonzero(canvas == color) == source_count,"piece color count changed")
            if split == "train":
                training.append(info)
    _require(all(any(_a_differences(i)[name] for i in training)
                 for name in ("reverse_fast","reverse_slow","fast_major","left_align","reflect_thickness")),
             "training does not distinguish strip shortcuts")
    _require(any(len(_maps(i,use_labels=False)) > 1 for i in training),"compass not demonstrated")
    ts = [t for i in training for t in _maps(i)[0][1]]
    _require(any(0 < t < 4 for t in ts) and any(t >= 4 for t in ts),"rotation/reflection not demonstrated")
    _require(any(any(len({r//2 for r,c in target}) > 1 for target in i["targets"]) for i in training),
             "no seam-crossing registration demonstration")


def self_test():
    """Bounded 42-seed invariants, counterfactuals, and exact-count checks."""
    signatures,shapes,layouts,configs,patterns,lengths,counts = set(),set(),set(),set(),set(),set(),set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train","test"):
            for pair in puzzle[split]:
                info = _read(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()),pair["output"])
                digest.update(bytes(pair["input"].shape)+pair["input"].tobytes())
                shapes.add((pair["input"].shape,pair["output"].shape))
                configs.add((info["axis"],info["bottom"],info["right"]))
                layouts.add((tuple(map(len,info["groups"])),tuple(info["lengths"])))
                lengths.update(info["lengths"])
                counts.add(len(info["sources"]))
                for target in info["targets"]:
                    patterns.add(_signature({p:1 for p in target}))
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 35 and len(layouts) >= 80
    assert len(configs) == 8 and len(patterns) >= 12 and len(lengths) >= 7
    assert {1,2,3} <= counts
    # Rotation of the whole keyed input leaves the normalized answer unchanged;
    # a bijective recoloring changes only the copied color identities.
    p = generate_puzzle(195,2,1)
    palette = np.array([7,0,9,1,6,2,8,4,3,5],dtype=np.int8)
    for pair in p["train"]+p["test"]:
        for k in range(4):
            assert np.array_equal(solve(np.rot90(pair["input"],k)),pair["output"])
        assert np.array_equal(solve(palette[pair["input"]]),palette[pair["output"]])
    # Reader breadth beyond sampled dimensions and label subsets: 14 tapes,
    # width 26, and a single-label piece with asymmetric registration geometry.
    for n,w in ((14,6),(3,26)):
        piece = {(0,3):4,(1,0):3,(2,0):3,(3,0):3,(3,1):3,
                 (1,3):2,(2,3):2,(3,3):2,**{(4,c):2 for c in range(4)}}
        final = _shift(piece,0,w//2-2)
        canvas = np.zeros((2*n,w),dtype=np.int8)
        marks = np.zeros_like(canvas)
        for point,value in final.items():
            canvas[point] = value
            if value == 3:
                marks[point] = 3
        scene = np.zeros((28,28),dtype=np.int8)
        for i in range(n):
            tape = np.full((2,w),2,dtype=np.int8)
            tape[marks[2*i:2*i+2] == 3] = 3
            row,col = (1+3*(i//3),1+7*(i % 3)) if w == 6 else (1+3*i,1)
            scene[row:row+2,col:col+w] = tape
        for point,value in _shift(_transform(piece,5),20,2).items():
            scene[point] = value
        inp = _frame(scene,(4,5,6,7),0)
        inp[0,0] = 1
        assert np.array_equal(solve(inp),_frame(canvas,(4,5,6,7),0))
    for a,b in ((2,1),(10,1),(10,4),(6,3)):
        p,q = generate_puzzle(195,a,b),generate_puzzle(195,a,b)
        assert all(np.array_equal(x[k],y[k]) for s in ("train","test")
                   for x,y in zip(p[s],q[s]) for k in ("input","output"))
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = generate_puzzle(195,2,1)
    bad["test"][0]["output"][1,1] = (int(bad["test"][0]["output"][1,1])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")

