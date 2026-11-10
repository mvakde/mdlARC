"""Seam-spliced counterprints: one stamp, two registered views, one global cut.

Only NumPy and the standard library are used. Generation never reads files.
No execution traces or sample payloads are retained by this module.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0,393346451,843992463,449113674,916076844,597527363,599561983,
              770878272,372923623,322192546,164022426,479403153,828242672,
              217270496,506091650,725986556,843020829,505451661,950646021,
              256199084,402117726,308372103,185975866,295983156,818410401,
              160250786,769180726,369677646,88090105,316682396,213526630,
              664997712,218131116,846453845,977581080,384888264,619961568,
              156203598,926250731,427210290,56388760,125714870)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    nt, nq = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return nt, nq


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "grid colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    h, w = mask.shape
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    directions = ((-1,0),(1,0),(0,-1),(0,1))
    if diagonal:
        directions += ((-1,-1),(-1,1),(1,-1),(1,1))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                p = r+dr, c+dc
                if p in todo:
                    todo.remove(p)
                    cells.append(p)
                    stack.append(p)
        result.append(cells)
    return result


def _mask(shape, cells):
    result = np.zeros(shape, dtype=bool)
    for p in cells:
        result[p] = True
    return result


def _regions(cut):
    """Check a digital path and recover its two four-connected regions."""
    points = set(map(tuple, np.argwhere(cut).tolist()))
    _require(len(points) >= 3, "cut too short")
    graph = {}
    for r, c in sorted(points):
        neighbors = []
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                q = r+dr, c+dc
                if not (dr or dc) or q not in points:
                    continue
                # Suppress diagonal shortcuts around ordinary orthogonal corners.
                if dr and dc and ((r+dr,c) in points or (r,c+dc) in points):
                    continue
                neighbors.append(q)
        graph[r,c] = neighbors
    ends = [p for p in graph if len(graph[p]) == 1]
    _require(len(ends) == 2 and all(1 <= len(v) <= 2 for v in graph.values()),
             "cut must be an unbranched open path")
    h, w = cut.shape
    boundary = lambda p: p[0] in (0,h-1) or p[1] in (0,w-1)
    _require(all(boundary(p) == (p in ends) for p in graph), "invalid cut boundary contacts")
    _require(len(_components(cut, True)) == 1, "cut is disconnected")
    parts = _components(~cut)
    _require(len(parts) == 2, "cut must make exactly two regions")
    return [_mask(cut.shape, p) for p in parts]


def _sites(cut, region, anchors=None):
    near = cut.copy()
    h, w = cut.shape
    for r, c in np.argwhere(cut):
        near[max(0,r-1):min(h,r+2), max(0,c-1):min(w,c+2)] = True
    legal = region & ~near
    if anchors is not None:
        legal &= ~anchors
    legal[0,0] = legal[0,-1] = legal[-1,0] = legal[-1,-1] = False
    return list(map(tuple, np.argwhere(legal).tolist()))


def _squares(mask):
    result = []
    for cells in _components(mask):
        rr, cc = zip(*cells)
        r, c = min(rr), min(cc)
        s = max(rr)-r+1
        _require(s == max(cc)-c+1 and len(cells) == s*s, "anchor is not a solid square")
        result.append((r,c,s))
    return result


def _decode(grid):
    grid = _grid(grid)
    f, bg = int(grid[0,0]), int(grid[1,1])
    _require(f != bg and np.all(grid[[0,-1],:] == f) and np.all(grid[:,[0,-1]] == f),
             "broken outer frame")
    interior = grid[1:-1,1:-1]
    rows = [r for r in range(1,interior.shape[0]-1) if np.all(interior[r] == f)]
    cols = [c for c in range(1,interior.shape[1]-1) if np.all(interior[:,c] == f)]
    _require(len(rows)+len(cols) == 1, "expected one straight frame-colored separator")
    if rows:
        r = rows[0]+1
        bounds = [(1,r,1,grid.shape[1]-1),(r+1,grid.shape[0]-1,1,grid.shape[1]-1)]
    else:
        c = cols[0]+1
        bounds = [(1,grid.shape[0]-1,1,c),(1,grid.shape[0]-1,c+1,grid.shape[1]-1)]
    panels = [grid[r0:r1,c0:c1] for r0,r1,c0,c1 in bounds]
    for p in panels:
        _require(min(p.shape) >= 3 and np.all(p[np.ix_([0,-1],[0,-1])] == bg),
                 "panel corners must expose background")
        _require(not np.any(p == f), "frame ink inside a panel")
    colors = [set(map(int,np.unique(p)))-{bg} for p in panels]
    common = colors[0] & colors[1]
    _require(len(common) == 1, "expected one common anchor identity")
    anchor = next(iter(common))
    sources = [i for i in range(2) if len(colors[i]) >= 3 and len(colors[1-i]) == 2]
    _require(len(sources) == 1, "source needs at least two detail colors")
    source_index = sources[0]
    scene_index = 1-source_index
    source, scene = panels[source_index], panels[scene_index]
    control = next(iter(colors[scene_index]-{anchor}))
    source_anchors = _squares(source == anchor)
    _require(len(source_anchors) == 1, "source must have one anchor")
    targets = _squares(scene == anchor)
    _require(1 <= len(targets) <= 4, "scene needs 1..4 separate square anchors")
    components = sorted(_components(scene == control, True), key=len)
    _require(len(components) == 2 and len(components[0]) == 1, "cut and singleton marker are ambiguous")
    marker = components[0][0]
    cut = _mask(scene.shape, components[1])
    regions = _regions(cut)
    side = next(i for i,region in enumerate(regions) if region[marker])
    _require(marker in _sites(cut,regions[side],scene == anchor), "marker touches cut or anchor")
    return {"grid":grid,"frame":f,"bg":bg,"anchor":anchor,"control":control,
            "source":source,"scene":scene,"bounds":bounds[scene_index],
            "source_anchor":source_anchors[0],"targets":targets,"cut":cut,
            "regions":regions,"marker":marker,"side":side}


def _scale_source(data, side):
    """Exact rational rasterization, including sub-anchor-size details.

    Refine source pixels by destination-side, then collapse source-side blocks.
    A block containing two colors is ambiguous and outside this family.
    """
    source, bg = data["source"], data["bg"]
    ar, ac, s = data["source_anchor"]
    points = np.argwhere(source != bg)
    lower = points.min(0)-[ar,ac]
    upper = points.max(0)+1-[ar,ac]
    top,left = (lower//s*s).tolist()
    bottom,right = ((upper+s-1)//s*s).tolist()
    raw = np.full((bottom-top,right-left),bg,dtype=np.int8)
    for y,x in points:
        raw[int(y)-ar-top,int(x)-ac-left] = source[y,x]
    high = np.repeat(np.repeat(raw,side,axis=0),side,axis=1)
    h,w = high.shape[0]//s, high.shape[1]//s
    blocks = high.reshape(h,s,w,s)
    low,maximum = blocks.min(axis=(1,3)),blocks.max(axis=(1,3))
    _require(np.array_equal(low,maximum), "scale crosses a color boundary")
    return low,(-top*side//s,-left*side//s)


def _paste(target, raster, top, left, bg):
    r0,c0 = max(0,top),max(0,left)
    r1,c1 = min(target.shape[0],top+raster.shape[0]),min(target.shape[1],left+raster.shape[1])
    if r0 < r1 and c0 < c1:
        patch = raster[r0-top:r1-top,c0-left:c1-left]
        view = target[r0:r1,c0:c1]
        view[patch != bg] = patch[patch != bg]


def _views(data, *, unit_scale=False, no_turn=False):
    bg = data["bg"]
    normal = np.full(data["scene"].shape,bg,dtype=np.int8)
    turned = normal.copy()
    occupied = np.zeros(normal.shape,dtype=bool)
    clipped = 0
    for r,c,side in data["targets"]:
        requested = data["source_anchor"][2] if unit_scale else side
        raster,offset = _scale_source(data,requested)
        a = np.full(normal.shape,bg,dtype=np.int8)
        b = a.copy()
        _paste(a,raster,r-offset[0],c-offset[1],bg)
        reverse_offset = raster.shape[0]-offset[0]-requested,raster.shape[1]-offset[1]-requested
        _paste(b,raster[::-1,::-1],r-reverse_offset[0],c-reverse_offset[1],bg)
        union = (a != bg) | (b != bg)
        if not unit_scale:
            _require(not np.any(occupied & union), "candidate stamp supports overlap")
            clipped += int(np.count_nonzero(a != bg) < np.count_nonzero(raster != bg)
                           or np.count_nonzero(b != bg) < np.count_nonzero(raster != bg))
        occupied |= union
        normal[a != bg] = a[a != bg]
        turned[b != bg] = b[b != bg]
    return normal, normal.copy() if no_turn else turned, clipped


def _combine(normal, turned, cut, marked, bg, seam="normal"):
    result = np.where(marked,normal,turned).astype(np.int8)
    if seam == "normal":
        result[cut] = np.where(normal[cut] != bg,normal[cut],turned[cut])
    elif seam == "turned":
        result[cut] = np.where(turned[cut] != bg,turned[cut],normal[cut])
    elif seam == "blank":
        result[cut] = bg
    else:
        raise ValueError("unknown seam convention")
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the supplied pixels, without seed or generation state."""
    data = _decode(input_grid)
    normal,turned,_ = _views(data)
    result = _combine(normal,turned,data["cut"],data["regions"][data["side"]],data["bg"])
    out = data["grid"].copy()
    r0,r1,c0,c1 = data["bounds"]
    out[r0:r1,c0:c1] = result
    return out


def _cut(rng,h,w,family):
    # Random quarter turns are implemented by transposition and axis reversal.
    for _ in range(64):
        across = bool(rng.integers(2))
        hh,ww = (w,h) if across else (h,w)
        cut = np.zeros((hh,ww),dtype=bool)
        c = int(rng.integers(2,ww-2))
        if family == "straight":
            cut[:,c] = True
        elif family == "diagonal":
            end = int(rng.integers(2,ww-2))
            end = c+1 if end == c and c < ww-3 else c-1 if end == c else end
            columns = np.rint(np.linspace(c,end,hh)).astype(int)
            cut[np.arange(hh),columns] = True
        elif family == "elbow":
            r = int(rng.integers(2,hh-2))
            cut[:r+1,c] = True
            cut[r,c:] = True
        else:
            levels = [hh//2] if family == "step" else [hh//3,2*hh//3]
            last = 0
            for row in levels:
                cut[last:row+1,c] = True
                options = [x for x in range(2,ww-2) if x != c]
                nxt = int(rng.choice(options))
                cut[row,min(c,nxt):max(c,nxt)+1] = True
                last,c = row,nxt
            cut[last:,c] = True
        if across:
            cut = cut.T.copy()
        if rng.integers(2):
            cut = cut[::-1,:].copy()
        if rng.integers(2):
            cut = cut[:,::-1].copy()
        try:
            regions = _regions(cut)
            if all(_sites(cut,region) for region in regions):
                return cut,regions
        except AssertionError:
            pass
    raise RuntimeError("bounded cut construction exhausted")


def _motif(rng,anchor,details,witness=False):
    a = 1 if witness else int(rng.choice([1,1,2]))
    motif = {(r,c):anchor for r in range(a) for c in range(a)}
    def paint(cells,color):
        for p in cells:
            if motif.get(p) != anchor:
                motif[p] = color
    style = int(rng.integers(4))
    if witness or style == 0:
        # Opposed colored arms provide possible seam collisions; unequal lengths
        # and perpendicular branches provide foreground/background disagreements.
        up,down = int(rng.integers(1,4)),int(rng.integers(1,4))
        if witness and up == down:
            up = 1+up%3
        paint(((r,c) for r in range(-up,0) for c in range(a)),details[0])
        paint(((r,c) for r in range(a,a+down) for c in range(a)),details[1])
        length = int(rng.integers(1,4))
        paint(((r,c) for r in range(a) for c in range(a,a+length)),details[-1])
        if rng.random() < .45:
            paint(((0,c) for c in range(-int(rng.integers(1,3)),0)),details[0])
    elif style == 1:
        length = int(rng.integers(2,5))
        paint(((r,0) for r in range(-length,0)),details[0])
        for r in range(-length,0,2):
            reach = int(rng.integers(1,4))
            sign = int(rng.choice([-1,1]))
            paint(((r,sign*c) for c in range(1,reach+1)),details[1])
        paint(((a,c) for c in range(a)),details[-1])
    else:
        for step in range(int(rng.integers(4,11))):
            cells = sorted(motif)
            r,c = cells[int(rng.integers(len(cells)))]
            dr,dc = ((-1,0),(1,0),(0,-1),(0,1))[int(rng.integers(4))]
            length = int(rng.integers(1,4))
            region = [(r+dr*t,c+dc*t) for t in range(1,length+1)]
            if all(-3 <= y < a+3 and -3 <= x < a+3 for y,x in region):
                paint(region,details[step%len(details)])
        if style == 3:
            paint(((r,a) for r in range(-1,a+2)),details[0])
            paint(((-1,c) for c in range(a+1)),details[1])
    for _ in range(int(rng.integers(4))):
        motif = {(c,a-1-r):v for (r,c),v in motif.items()}
    if rng.integers(2):
        motif = {(r,a-1-c):v for (r,c),v in motif.items()}
    return motif,a


def _offsets(motif,a,unit,reverse=False):
    result = []
    for (r,c),color in sorted(motif.items()):
        if reverse:
            r,c = a-1-r,a-1-c
        result.extend((r*unit+dr,c*unit+dc,color) for dr in range(unit) for dc in range(unit))
    return result


def _logical_view(shape, offsets, position, bg):
    result = np.full(shape,bg,dtype=np.int8)
    r,c = position
    for dr,dc,value in offsets:
        if 0 <= r+dr < shape[0] and 0 <= c+dc < shape[1]:
            result[r+dr,c+dc] = value
    return result


def _flags(normal,turned,cut,marked,bg,anchor):
    n,t = (normal != bg) & (normal != anchor),(turned != bg) & (turned != anchor)
    other = ~marked & ~cut
    return {name for name,present in (
        ("upright_visible",np.any(marked & n)),
        ("turned_visible",np.any(other & t)),
        ("suppress_turned",np.any(marked & (normal == bg) & t)),
        ("suppress_upright",np.any(other & (turned == bg) & n)),
        ("seam_priority",np.any(cut & n & t & (normal != turned)))) if present}


_REQUIRED = {"upright_visible","turned_visible","suppress_turned","suppress_upright","seam_priority"}


def _new_base(rng,family,witness=False,allow_clip=False):
    for _ in range(256):
        palette = list(map(int,rng.permutation(10)))
        bg,frame,control,anchor = palette[:4]
        details = palette[4:4+int(rng.integers(2,4))]
        motif,a = _motif(rng,anchor,details,witness)
        if len(set(motif.values())-{anchor}) < 2:
            continue
        rs,cs = zip(*motif)
        mh,mw = max(rs)-min(rs)+1,max(cs)-min(cs)+1
        units = [u for u in range(1,4) if mh*u+2 <= 13 and mw*u+2 <= 28]
        if not units:
            continue
        su = int(rng.choice(units))
        source_h = mh*su+2
        scene_h = int(rng.integers(11,28-source_h))
        width = int(rng.integers(max(17,mw*su+2),29))
        shape = scene_h,width
        source = np.full((source_h,width),bg,dtype=np.int8)
        source_r = 1-min(rs)*su
        source_c = int(rng.integers(1,width-mw*su))-min(cs)*su
        for (r,c),color in motif.items():
            source[source_r+r*su:source_r+(r+1)*su,source_c+c*su:source_c+(c+1)*su] = color
        cut,regions = _cut(rng,*shape,family)
        count = 2 if witness else int(rng.integers(1,4))
        target_units = [su,int(rng.choice([u for u in (1,2,3) if u != su]))] if witness else list(map(int,rng.integers(1,4,count)))
        # Place the larger candidate first; support overlap, rather than a fixed
        # tile layout, determines whether the next stamp fits.
        target_units.sort(reverse=True)
        normal = np.full(shape,bg,dtype=np.int8)
        turned = normal.copy()
        anchors = np.zeros(shape,dtype=bool)
        occupied = anchors.copy()
        target_records = []
        success = True
        for unit in target_units:
            side = a*unit
            no,to = _offsets(motif,a,unit),_offsets(motif,a,unit,True)
            all_offsets = no+to
            ymin,ymax = min(p[0] for p in all_offsets),max(p[0] for p in all_offsets)
            xmin,xmax = min(p[1] for p in all_offsets),max(p[1] for p in all_offsets)
            full = witness or not allow_clip or rng.random() < .50
            rlo,rhi = (-ymin,scene_h-ymax) if full else (0,scene_h-side+1)
            clo,chi = (-xmin,width-xmax) if full else (0,width-side+1)
            if rlo >= rhi or clo >= chi:
                success = False
                break
            for attempt in range(160):
                r,c = int(rng.integers(rlo,rhi)),int(rng.integers(clo,chi))
                if np.any(cut[r:r+side,c:c+side]):
                    continue
                if np.any(anchors[max(0,r-1):r+side+1,max(0,c-1):c+side+1]):
                    continue
                av = _logical_view(shape,no,(r,c),bg)
                bv = _logical_view(shape,to,(r,c),bg)
                support = (av != bg) | (bv != bg)
                if np.any(occupied & support):
                    continue
                if not all(np.any(support & region) for region in regions):
                    # Later targets may lie wholly on one side; the global
                    # control must affect at least the first target.
                    if not target_records or witness:
                        continue
                if any(support[p] for p in ((0,0),(0,width-1),(scene_h-1,0),(scene_h-1,width-1))):
                    continue
                if np.count_nonzero((av != bg) & (av != anchor)) == 0 or np.count_nonzero((bv != bg) & (bv != anchor)) == 0:
                    continue
                normal[av != bg] = av[av != bg]
                turned[bv != bg] = bv[bv != bg]
                anchors[r:r+side,c:c+side] = True
                occupied |= support
                target_records.append((r,c,side))
                break
            else:
                success = False
                break
        if not success:
            continue
        sites = [_sites(cut,region,anchors) for region in regions]
        if not all(sites):
            continue
        side = int(rng.integers(2))
        marked = regions[side]
        result = _combine(normal,turned,cut,marked,bg)
        if np.array_equal(result,normal) or np.array_equal(result,turned):
            continue
        if witness:
            covered = _flags(normal,turned,cut,marked,bg,anchor) | _flags(normal,turned,cut,regions[1-side],bg,anchor)
            if covered != _REQUIRED:
                continue
        return {"source":source,"normal":normal,"turned":turned,"anchors":anchors,
                "cut":cut,"regions":regions,"sites":sites,"side":side,"bg":bg,
                "frame":frame,"control":control,"anchor":anchor,"targets":target_records,
                "rotation":int(rng.integers(4))}
    raise RuntimeError(f"bounded stamp construction exhausted: {family}, witness={witness}")


def _assemble(base,rng,side=None):
    side = base["side"] if side is None else side
    marker = base["sites"][side][int(rng.integers(len(base["sites"][side])))]
    scene = np.full(base["cut"].shape,base["bg"],dtype=np.int8)
    scene[base["anchors"]] = base["anchor"]
    scene[base["cut"]] = base["control"]
    scene[marker] = base["control"]
    result = _combine(base["normal"],base["turned"],base["cut"],base["regions"][side],base["bg"])
    source = base["source"]
    h,w = source.shape
    inp = np.full((h+scene.shape[0]+3,w+2),base["frame"],dtype=np.int8)
    inp[1:h+1,1:-1] = source
    inp[h+2:-1,1:-1] = scene
    out = inp.copy()
    out[h+2:-1,1:-1] = result
    k = base["rotation"]
    return {"input":np.rot90(inp,k).copy(),"output":np.rot90(out,k).copy()}


def _key(grid):
    return grid.shape,grid.astype(np.uint8).tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                a = np.asarray(pair[role],dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,nq = _counts(num_train,num_test)
    seed = _integer(seed,"seed")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(seed)
    puzzle = {"train":[],"test":[]}
    seen = set()
    regime = int(rng.integers(2))
    base = _new_base(rng,"straight",witness=True)
    for side in (base["side"],1-base["side"]):
        pair = _assemble(base,rng,side)
        puzzle["train"].append(pair)
        seen.add(_key(pair["input"]))
    taught_clip = False
    for split,count in (("train",nt-2),("test",nq)):
        for index in range(count):
            family = (str(rng.choice(["straight","diagonal"] if regime == 0 else ["straight","diagonal","elbow"]))
                      if split == "train" else "elbow" if regime == 0 else str(rng.choice(["step","zigzag"])))
            for _ in range(32):
                base = _new_base(rng,family,allow_clip=(split == "train" or taught_clip))
                pair = _assemble(base,rng)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded unique-input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                taught_clip |= _views(_decode(pair["input"]))[2] > 0
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternatives(data,normal,turned):
    bg,cut = data["bg"],data["cut"]
    marked = data["regions"][data["side"]]
    union = np.where(normal != bg,normal,turned)
    unscaled_n,unscaled_t,_ = _views(data,unit_scale=True)
    # A credible no-scale shortcut still preserves the supplied anchor squares.
    unscaled_n[unscaled_n == data["anchor"]] = bg
    unscaled_t[unscaled_t == data["anchor"]] = bg
    whole = np.full(normal.shape,bg,dtype=np.int8)
    for r,c,s in data["targets"]:
        unscaled_n[r:r+s,c:c+s] = data["anchor"]
        unscaled_t[r:r+s,c:c+s] = data["anchor"]
        local = dict(data,targets=[(r,c,s)])
        n,t,_ = _views(local)
        view = n if marked[r,c] else t
        whole[view != bg] = view[view != bg]
    return {
        "upright_everywhere":normal,"turned_everywhere":turned,"foreground_union":union,
        "reversed_marker":_combine(normal,turned,cut,data["regions"][1-data["side"]],bg),
        "turned_seam_priority":_combine(normal,turned,cut,marked,bg,"turned"),
        "blank_seam":_combine(normal,turned,cut,marked,bg,"blank"),
        "ignore_anchor_scale":_combine(unscaled_n,unscaled_t,cut,marked,bg),
        "whole_stamp_by_anchor_side":whole}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen,covered,eliminated = set(),set(),set()
    scaled_taught = clip_taught = clip_test = False
    for split in ("train","test"):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {"input","output"}, "invalid pair keys")
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            _require(inp.shape == out.shape, "input/output dimensions differ")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            data = _decode(inp)
            normal,turned,clipped = _views(data)
            expected = solve(inp)
            _require(np.array_equal(expected,out), "wrong composite, control erasure, or changed source/frame")
            _require(not np.array_equal(inp,out), "identity example")
            r0,r1,c0,c1 = data["bounds"]
            result = out[r0:r1,c0:c1]
            _require(np.all(result[data["scene"] == data["anchor"]] == data["anchor"]), "changed anchor")
            _require(not np.any(result == data["control"]), "control ink must disappear")
            if split == "train":
                covered |= _flags(normal,turned,data["cut"],data["regions"][data["side"]],data["bg"],data["anchor"])
                for name,alternative in _alternatives(data,normal,turned).items():
                    if not np.array_equal(result,alternative):
                        eliminated.add(name)
                scaled_taught |= any(s != data["source_anchor"][2] for r,c,s in data["targets"])
                clip_taught |= clipped > 0
            else:
                clip_test |= clipped > 0
    _require(covered == _REQUIRED, "training does not expose both views, suppression, and seam priority")
    _require(scaled_taught, "no scaled training witness")
    _require(eliminated == {"upright_everywhere","turned_everywhere","foreground_union","reversed_marker",
                            "turned_seam_priority","blank_seam","ignore_anchor_scale","whole_stamp_by_anchor_side"},
             "a documented shortcut survives all demonstrations")
    _require(not clip_test or clip_taught, "clipping test without clipping demonstration")


def self_test():
    """42 seeds; bounded counters, hashes, and at most one puzzle live at a time."""
    signatures,shapes,motifs,ratios,counts,roles = set(),set(),set(),set(),set(),set()
    clipped = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        if seed == 0:
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterminism")
        for split in ("train","test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                n,t,lost = _views(d)
                clipped += lost > 0
                shapes.add(pair["input"].shape)
                counts.add(len(d["targets"]))
                roles.add((d["bg"],d["frame"],d["anchor"],d["control"]))
                sr,sc,ss = d["source_anchor"]
                boundaries = [ss]
                source = d["source"]
                for y,x in np.argwhere(source != d["bg"]):
                    color = source[y,x]
                    for dy,dx,v in ((-1,0,int(y)-sr),(1,0,int(y)+1-sr),
                                    (0,-1,int(x)-sc),(0,1,int(x)+1-sc)):
                        yy,xx = int(y)+dy,int(x)+dx
                        if not (0 <= yy < source.shape[0] and 0 <= xx < source.shape[1]) or source[yy,xx] != color:
                            boundaries.append(v)
                quantum = abs(int(np.gcd.reduce(boundaries)))
                colors = {d["anchor"]:1}
                points = set()
                for y,x in np.argwhere(d["source"] != d["bg"]):
                    color = int(d["source"][y,x])
                    points.add(((int(y)-sr)//quantum,(int(x)-sc)//quantum,colors.setdefault(color,len(colors)+1)))
                motifs.add(hashlib.sha256(repr(sorted(points)).encode()).hexdigest())
                for r,c,s in d["targets"]:
                    g = int(np.gcd(s,ss))
                    ratios.add((s//g,ss//g))
                # Moving the control marker exchanges only the open regions.
                modified = pair["input"].copy()
                r0,r1,c0,c1 = d["bounds"]
                panel = modified[r0:r1,c0:c1]
                panel[d["marker"]] = d["bg"]
                sites = _sites(d["cut"],d["regions"][1-d["side"]],d["scene"] == d["anchor"])
                panel[sites[0]] = d["control"]
                changed = solve(modified)[r0:r1,c0:c1]
                actual = pair["output"][r0:r1,c0:c1]
                _require(np.array_equal(changed[d["cut"]],actual[d["cut"]]), "marker changed seam priority")
                _require(np.array_equal(changed[d["regions"][d["side"]]],t[d["regions"][d["side"]]]), "marker did not exchange views")
        # Rotation and palette equivariance probe input-only role recovery.
        pair = puzzle["test"][0]
        _require(np.array_equal(solve(np.rot90(pair["input"])),np.rot90(pair["output"])), "rotation equivariance")
        palette = np.array([9,7,4,6,0,8,2,5,1,3],dtype=np.int8)
        _require(np.array_equal(solve(palette[pair["input"]]),palette[pair["output"]]), "palette equivariance")
    _require(len(signatures) == 42 and len(shapes) >= 25 and len(motifs) >= 70, "structural variation collapsed")
    _require({1,2,3} <= counts and len(ratios) >= 5 and len(roles) >= 40, "anchor/palette variation collapsed")
    _require(clipped > 0, "no clipped examples")
    # Every legal explicit pair count, not only the boundaries.
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(1234+nt*10+nq,nt,nq)
            _require((len(p["train"]),len(p["test"])) == (nt,nq), "count request ignored")
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(0,2,1)
    p["test"][0]["output"][0,0] = (int(p["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # A hand-specified four-target member, outside the sampler's 1..3 targets.
    # Its simple L has an upright arm above/right, or a turned arm below/left.
    source = np.zeros((5,15),dtype=np.int8)
    source[2,4],source[1,4],source[2,5] = 2,5,6
    scene = np.zeros((12,15),dtype=np.int8)
    scene[:,7] = 8
    scene[1,0] = 8
    expected_scene = np.zeros_like(scene)
    for r,c in ((3,3),(8,3),(3,11),(8,11)):
        scene[r,c] = expected_scene[r,c] = 2
        if c < 7:
            expected_scene[r-1,c],expected_scene[r,c+1] = 5,6
        else:
            expected_scene[r+1,c],expected_scene[r,c-1] = 5,6
    fixture = np.full((20,17),9,dtype=np.int8)
    fixture[1:6,1:-1],fixture[7:-1,1:-1] = source,scene
    expected = fixture.copy()
    expected[7:-1,1:-1] = expected_scene
    _require(np.array_equal(solve(fixture),expected), "four-target hand fixture")
    # Fine details scaled by 3/2 are genuinely ambiguous, unlike lattice-aligned
    # reductions in the generated family. Enlargement by 2 remains exact.
    fine = np.zeros((5,5),dtype=np.int8)
    fine[1:3,1:3],fine[0,1],fine[1,3] = 2,5,6
    data = {"source":fine,"source_anchor":(1,1,2),"bg":0}
    _scale_source(data,4)
    try:
        _scale_source(data,3)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous fractional scale accepted")
    return {"seeds":42,"distinct_puzzles":len(signatures),"canvas_shapes":len(shapes),
            "source_lattice_signatures":len(motifs),"ratios":sorted(ratios),
            "target_counts":sorted(counts),"clipped_pairs":clipped}
