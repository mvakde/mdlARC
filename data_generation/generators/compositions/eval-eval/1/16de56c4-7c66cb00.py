"""Cadenced presses: color-routed component masks printed at ruler phases.

Only visible input geometry determines the result. No file I/O, parent imports,
global RNG, retained puzzle cache, or generation metadata is used by solve.
"""
from collections import Counter
import hashlib

import numpy as np


HOST_SEEDS = (0,343433891,343482728,316748395,18856795,224890413,438338978,
              821843580,47899259,363674821,129620006,423245551,772382582,
              965178906,892329831,772675161,452796849,475715631,919512079,
              670369902,257800863,400029161,451609596,444655058,559613852,
              641806990,150294857,462288420,370576443,487099999,242343605,
              598817445,128649531,622633729,375507025,536595618,564565242,
              783145225,178201032,736198098,932586547,228144054)
FOUR = ((1,0),(-1,0),(0,1),(0,-1))
EIGHT = tuple((r,c) for r in (-1,0,1) for c in (-1,0,1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be 1..30 by 1..30 integer arrays with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(grid, points, diagonal=False):
    unseen, result = set(points), []
    while unseen:
        first = min(unseen)
        color = int(grid[first])
        unseen.remove(first)
        stack, cells = [first], []
        while stack:
            r,c = stack.pop()
            cells.append((r,c))
            for dr,dc in EIGHT if diagonal else FOUR:
                p = r+dr,c+dc
                if p in unseen and int(grid[p]) == color:
                    unseen.remove(p)
                    stack.append(p)
        result.append(dict(color=color,cells=tuple(sorted(cells))))
    return result


def _signature(cells):
    r0,c0 = min(r for r,c in cells),min(c for r,c in cells)
    return tuple(sorted((r-r0,c-c0) for r,c in cells))


def _anchors(band, ignore_terminal=False, dense=False, visible=False):
    a,b,p,t = (band[k] for k in ("a","b","period","terminal"))
    if visible:
        return [a,b] + ([] if t is None else [t])
    lo,hi = band["top"],band["bottom"]
    if t is not None and not ignore_terminal:
        lo,hi = min(a,t),max(b,t)
    return [r for r in range(lo,hi+1) if dense or (r-a) % p == 0]


def _render(grid, background, bands, components, mode="rule"):
    """Input-derived rendering, also with explicit diagnostic counter-rules."""
    out = grid.copy()
    by_color = {b["route"]:b for b in bands}
    for comp in components:
        for p in comp["cells"]:
            out[p] = background
    if mode == "group_color":
        components = [dict(color=color,cells=tuple(p for c in components
                      if c["color"] == color for p in c["cells"]))
                      for color in sorted({c["color"] for c in components})]
    for comp in components:
        band = by_color[comp["color"]]
        if mode == "offphase_stop" and band["distractors"]:
            band = dict(band,terminal=band["distractors"][0])
        cells = comp["cells"]
        if mode == "bounding_box":
            cells = tuple((r,c) for r in range(min(r for r,c in cells),max(r for r,c in cells)+1)
                          for c in range(min(c for r,c in cells),max(c for r,c in cells)+1))
        end = min(r for r,c in cells) if mode == "top_anchor" else max(r for r,c in cells)
        anchors = ([band["bottom"]] if mode == "one_landing" else
                   _anchors(band,ignore_terminal=mode in ("no_terminal","unbounded"),
                            dense=mode == "dense",visible=mode == "visible_ticks"))
        paint = band["reference"]
        if band["terminal"] is not None and mode != "no_terminal":
            paint = int(grid[band["terminal"],band["left"]])
        for anchor in anchors:
            target = [(r-end+anchor,c) for r,c in cells]
            if mode == "clip":
                target = [(r,c) for r,c in target if band["top"] <= r <= band["bottom"]]
            elif any(not band["top"] <= r <= band["bottom"] for r,c in target):
                continue
            if mode in ("span_clip","span_fit") and band["terminal"] is not None:
                lo,hi = min(band["a"],band["terminal"]),max(band["b"],band["terminal"])
                if mode == "span_fit" and any(not lo <= r <= hi for r,c in target):
                    continue
                if mode == "span_clip":
                    target = [(r,c) for r,c in target if lo <= r <= hi]
            for r,c in target:
                if not band["left"] < c < band["right"]:
                    raise ValueError("component lies outside its chamber's lateral span")
                out[r,c] = paint
    return out


def _canonical(grid):
    """Recognize complete chamber suffixes; a solid rail locates its ruler."""
    h,w = grid.shape
    if w < 8:
        return
    bg = int(grid[0,0])
    if int(grid[0,-1]) != bg:
        return
    rows = []
    for row in grid:
        nz = np.flatnonzero(row != bg)
        if len(nz) == 0:
            rows.append(None)
            continue
        right = int(nz[-1])
        if right < 7 or right == 0:
            rows.append(False)
            continue
        reference,route = int(row[right]),int(row[right-1])
        start = right-1
        while start >= 0 and int(row[start]) == route:
            start -= 1
        left = start
        valid = (left >= 0 and right-left >= 7 and route not in (bg,reference)
                 and np.all(row[:left] == bg) and np.all(row[right+1:] == bg)
                 and int(row[left]) != route)
        rows.append((left,right,route,reference) if valid else False)
    for first in range(2,h):
        if not rows[first] or rows[first-1] is not None:
            continue
        if any(row is False for row in rows[first:]):
            continue
        bands, r, valid = [],first,True
        while r < h:
            if rows[r] is None:
                r += 1
                continue
            info,top = rows[r],r
            while r+1 < h and rows[r+1] == info:
                r += 1
            bottom = r
            left,right,route,reference = info
            ticks = [y for y in range(top,bottom+1) if int(grid[y,left]) == reference]
            extras = [y for y in range(top,bottom+1) if int(grid[y,left]) not in (bg,reference)]
            if bottom-top < 2 or len(ticks) != 2:
                valid = False
                break
            a,b = ticks
            p = b-a
            terminals = [y for y in extras if (y-a) % p == 0]
            if len(terminals) > 1:
                valid = False
                break
            terminal = terminals[0] if terminals else None
            if terminal is not None and a < terminal < b:
                valid = False
                break
            bands.append(dict(top=top,bottom=bottom,left=left,right=right,route=route,
                              reference=reference,a=a,b=b,period=p,terminal=terminal,
                              distractors=tuple(y for y in extras if y != terminal)))
            r += 1
        if not valid or not bands or len({b["route"] for b in bands}) != len(bands):
            continue
        if any(y["top"] <= x["bottom"]+1 for x,y in zip(bands,bands[1:])):
            continue
        points = [tuple(map(int,p)) for p in np.argwhere(grid[:first] != bg)]
        if not points or any(c in (0,w-1) for r,c in points):
            continue
        comps = _components(grid,points)
        lookup = {b["route"]:b for b in bands}
        if any(c["color"] not in lookup for c in comps):
            continue
        if any(not lookup[c["color"]]["left"] < x < lookup[c["color"]]["right"]
               for c in comps for y,x in c["cells"]):
            continue
        if any(not any(anchor-(max(y for y,x in c["cells"])-min(y for y,x in c["cells"]))
                       >= lookup[c["color"]]["top"]
                       for anchor in _anchors(lookup[c["color"]])) for c in comps):
            continue
        yield dict(canonical=grid,background=bg,bands=bands,components=comps,first=first)


def _analyze(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for mirror in (False,True):
        reflected = np.fliplr(grid) if mirror else grid
        for turn in range(4):
            for analysis in _canonical(np.rot90(reflected,turn)):
                analysis.update(turn=turn,mirror=mirror)
                candidates.append(analysis)
    if len(candidates) != 1:
        raise ValueError("input must have one staging/chamber/ruler interpretation")
    return candidates[0]


def _restore(grid, analysis):
    grid = np.rot90(grid,-analysis["turn"])
    return (np.fliplr(grid) if analysis["mirror"] else grid).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _analyze(input_grid)
    return _restore(_render(a["canonical"],a["background"],a["bands"],a["components"]),a)


def _counter_output(analysis, mode):
    return _restore(_render(analysis["canonical"],analysis["background"],
                           analysis["bands"],analysis["components"],mode),analysis)


def _mask(rng, maxh, maxw):
    h,w = int(rng.integers(1,maxh+1)),int(rng.integers(1,maxw+1))
    kind = int(rng.integers(6))
    if kind == 0:
        cells = {(r,c) for r in range(h) for c in range(w)}
    elif kind == 1:
        cells = {(r,0) for r in range(h)} | {(h-1,c) for c in range(w)}
    elif kind == 2 and h >= 3 and w >= 3:
        cells = {(r,c) for r in range(h) for c in range(w)
                 if r in (0,h-1) or c in (0,w-1)}
    elif kind == 3:
        spine = int(rng.integers(w))
        cells = {(r,spine) for r in range(h)} | {(h//2,c) for c in range(w)}
    else:
        cells = {(int(rng.integers(h)),int(rng.integers(w)))}
        count = int(rng.integers(1,h*w+1))
        while len(cells) < count:
            options = sorted({(r+dr,c+dc) for r,c in cells for dr,dc in FOUR
                              if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            cells.add(options[int(rng.integers(len(options)))])
    return _signature(cells)


def _place(grid, group, bands, stage, bg, rng, depth_different=None):
    """Place a possibly touching multicolor group, preserving same-color units."""
    gh = 1+max(r for i,cells in group for r,c in cells)
    gw = 1+max(c for i,cells in group for r,c in cells)
    lo = max(bands[i]["left"]+1 for i,cells in group)
    hi = min(bands[i]["right"] for i,cells in group)-gw
    if gh > stage or hi < lo:
        return False
    positions = [(int(r),int(c)) for r in rng.permutation(stage-gh+1)
                 for c in rng.permutation(np.arange(lo,hi+1))]
    for top,left in positions:
        additions = {}
        okay = True
        for i,mask in group:
            color = bands[i]["route"]
            cells = {(r+top,c+left) for r,c in mask}
            if depth_different is not None and max(r for r,c in cells) == depth_different:
                okay = False
                break
            if any(grid[p] != bg or p in additions for p in cells):
                okay = False
                break
            if any(0 <= r+dr < stage and 0 <= c+dc < grid.shape[1] and
                   int(grid[r+dr,c+dc]) == color for r,c in cells for dr,dc in EIGHT):
                okay = False
                break
            additions.update({p:color for p in cells})
        if okay:
            for p,color in additions.items():
                grid[p] = color
            return True
    return False


def _ruler_options(depth, height, terminal, lesson, rng):
    options = []
    for p in range(1,min(6,(depth-1)//2)+1):
        if lesson and not terminal and p == 1:
            continue
        for a in range(depth-p):
            b = a+p
            phase = list(range(a % p,depth,p))
            fit = [x for x in phase if x >= height-1]
            if len(fit) < 2:
                continue
            if terminal:
                for t in phase:
                    if a <= t <= b:
                        continue
                    lo,hi = min(a,t),max(b,t)
                    inside = [x for x in fit if lo <= x <= hi]
                    if len(inside) < 2:
                        continue
                    if lesson and not any(x < lo or x > hi for x in fit):
                        continue
                    options.append((a,b,p,t))
            elif len(phase) >= 3 and any(x not in (a,b) for x in fit):
                options.append((a,b,p,None))
    if not options:
        return None
    # Uniform period first avoids swamping larger periods with many placements.
    periods = sorted({x[2] for x in options})
    p = int(rng.choice(periods))
    options = [x for x in options if x[2] == p]
    if terminal and lesson:
        sign = -1 if lesson == "split" else 1
        preferred = [x for x in options if (x[3] < x[0]) == (sign == -1)]
        options = preferred or options
    return options[int(rng.integers(len(options)))]


def _build_pair(rng, count, lesson, turn, mirror, noisy, unused):
    stage = int(rng.integers(6,9)) if count < 3 else int(rng.integers(5,8))
    width = int(rng.integers(17 if lesson else 13,31))
    gap = int(rng.integers(1,3)) if count == 1 else 1
    if lesson:
        depths = list(map(int,rng.integers(8,11,size=count)))
    elif count == 1:
        depths = [int(rng.integers(8,19))]
    elif count == 2:
        depths = list(map(int,rng.integers(6,11,size=2)))
    else:
        depths = list(map(int,rng.integers(5,7,size=3)))
    margin = int(rng.integers(0,3))
    if stage+gap*count+sum(depths)+margin > 30:
        margin = 0
    while stage+gap*count+sum(depths) > 30:
        j = int(np.argmax(depths))
        depths[j] -= 1
    h = stage+gap*count+sum(depths)+margin
    palette = list(map(int,rng.permutation(10)))
    bg,routes = palette[0],palette[1:count+1]
    grid = np.full((h,width),bg,dtype=np.int8)
    bands,cursor = [],stage
    for route,depth in zip(routes,depths):
        cursor += gap
        left = int(rng.integers(0,3)) if width >= 18 else 0
        right = width-1-int(rng.integers(0,3)) if width >= 18 else width-1
        reference = int(rng.choice([c for c in palette if c not in (bg,route)]))
        bands.append(dict(top=cursor,bottom=cursor+depth-1,left=left,right=right,
                          route=route,reference=reference))
        grid[cursor:cursor+depth,left+1:right] = route
        grid[cursor:cursor+depth,right] = reference
        cursor += depth
    active = list(range(count))
    if unused and count > 1 and lesson != "split":
        active.remove(int(rng.choice(active)))
    if lesson == "split":
        outer,inner = list(map(int,rng.permutation(active)))[:2]
        fh,fw = int(rng.integers(3,5)),int(rng.integers(3,8))
        frame = tuple((r,c) for r in range(fh) for c in range(fw)
                      if r in (0,fh-1) or c in (0,fw-1))
        center = tuple((r,c) for r in range(1,fh-1) for c in range(1,fw-1))
        if not _place(grid,[(outer,frame),(inner,center)],bands,stage,bg,rng):
            return None
    if lesson == "independent":
        i = int(rng.choice(active))
        for j in range(2):
            old = np.argwhere(grid[:stage] == routes[i])
            depth = int(old[:,0].max()) if j else None
            if not _place(grid,[(i,_mask(rng,3,4))],bands,stage,bg,rng,depth):
                return None
    for i in rng.permutation(active):
        i = int(i)
        if not np.any(grid[:stage] == routes[i]):
            if not _place(grid,[(i,_mask(rng,min(stage,depths[i]-2),min(7,width//3)))],
                          bands,stage,bg,rng):
                return None
    for _ in range(int(rng.integers(1,5 if count < 3 else 4))):
        i = int(rng.choice(active))
        _place(grid,[(i,_mask(rng,min(stage,depths[i]-2),min(8,width//3)))],
               bands,stage,bg,rng)
    points = [tuple(map(int,p)) for p in np.argwhere(grid[:stage] != bg)]
    comps = _components(grid,points)
    flags = list(map(bool,rng.integers(0,2,size=count)))
    if lesson:
        flags[0],flags[1] = True,False
        if rng.random() < .5:
            flags.reverse()
    elif count > 1 and rng.random() < .65:
        flags[0],flags[1] = True,False
        rng.shuffle(flags)
    for i,band in enumerate(bands):
        heights = [1+max(r for r,c in comp["cells"])-min(r for r,c in comp["cells"])
                   for comp in comps if comp["color"] == band["route"]]
        height = max(heights,default=1)
        geom = _ruler_options(depths[i],height,flags[i],lesson,rng)
        if geom is None:
            return None
        a,b,p,t = geom
        top,left = band["top"],band["left"]
        grid[top+a,left] = grid[top+b,left] = band["reference"]
        available = [c for c in palette if c not in (bg,band["route"],band["reference"])]
        if t is not None:
            grid[top+t,left] = available.pop()
        options = [r for r in range(depths[i]) if (r-a) % p]
        noise_count = min(len(options),int(rng.integers(1,3))) if noisy and options else 0
        for r,color in zip(rng.choice(options,noise_count,replace=False),available):
            grid[top+int(r),left] = color
        band.update(a=top+a,b=top+b,period=p,terminal=None if t is None else top+t,
                    distractors=tuple(top+int(x) for x in options if grid[top+int(x),left] != bg))
    # Construction paints from sampled rulers and already placed source masks.
    # It does not call solve or rediscover the chamber layout.
    out = grid.copy()
    out[:stage] = bg
    for comp in comps:
        band = next(b for b in bands if b["route"] == comp["color"])
        bottom = max(r for r,c in comp["cells"])
        first = min(r for r,c in comp["cells"])
        color = band["reference"] if band["terminal"] is None else int(grid[band["terminal"],band["left"]])
        lower = band["top"] if band["terminal"] is None else min(band["a"],band["terminal"])
        upper = band["bottom"] if band["terminal"] is None else max(band["b"],band["terminal"])
        for r in range(lower,upper+1):
            if (r-band["a"]) % band["period"] or r-bottom+first < band["top"]:
                continue
            for y,x in comp["cells"]:
                out[r-bottom+y,x] = color
    if mirror:
        grid,out = np.fliplr(grid),np.fliplr(out)
    return {"input":np.rot90(grid,turn).copy(),"output":np.rot90(out,turn).copy()}


def _features(a):
    grid,bands,comps = a["canonical"],a["bands"],a["components"]
    points = {p for c in comps for p in c["cells"]}
    contact = any((r+dr,c+dc) in points and grid[r,c] != grid[r+dr,c+dc]
                  for r,c in points for dr,dc in FOUR)
    unequal = any(x["color"] == y["color"] and
                  max(r for r,c in x["cells"]) != max(r for r,c in y["cells"])
                  for i,x in enumerate(comps) for y in comps[i+1:])
    return dict(contact=contact,unequal=unequal,unused=len(bands)-len({c["color"] for c in comps}),
                boundary=any(b["terminal"] is None and any(c["color"] == b["route"] for c in comps) for b in bands),
                terminal=any(b["terminal"] is not None and any(c["color"] == b["route"] for c in comps) for b in bands),
                noise=sum(len(b["distractors"]) for b in bands))


COUNTER_RULES = ("one_landing","visible_ticks","dense","no_terminal","unbounded",
                 "group_color","bounding_box","top_anchor","clip","span_clip","span_fit")


def _teaches(pairs, required):
    analyses = [_analyze(p["input"]) for p in pairs]
    return all(any(not np.array_equal(p["output"],_counter_output(a,mode))
                   for p,a in zip(pairs,analyses)) for mode in required)


def _count(value, low, high):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
            or not low <= value <= high):
        raise ValueError("illegal training/test counts")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train,num_test = _count(num_train,2,10),_count(num_test,1,4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    transfer = int(rng.integers(2))
    base = int(rng.integers(4))
    mirror = bool(rng.integers(2))
    noisy = bool(rng.integers(2))
    unused = bool(rng.random() < .3)
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",num_train),("test",num_test)):
        for index in range(count):
            lesson = ("split" if index == 0 else "independent" if index == 1 else None) if split == "train" else None
            n = 2 if lesson else int(rng.choice((1,2) if transfer == 0 and split == "train" else (1,2,3)))
            if split == "test" and transfer == 0:
                n = 3
            turn = (base+(index % 2 if split == "train" else 2+index % 2)) % 4 if transfer else base
            for attempt in range(180):
                pair = _build_pair(rng,n,lesson,turn,mirror,noisy,unused and (lesson == "independent" or split == "test"))
                if pair is None:
                    continue
                key = (pair["input"].shape,pair["input"].tobytes())
                if key in seen:
                    continue
                try:
                    a = _analyze(pair["input"])
                except ValueError:
                    continue
                if lesson == "split":
                    if not _features(a)["contact"]:
                        continue
                    if not _teaches([pair],("one_landing","visible_ticks","dense","no_terminal","unbounded","bounding_box","clip","span_clip","span_fit")):
                        continue
                    if noisy and not _features(a)["noise"]:
                        continue
                if lesson == "independent":
                    if not _features(a)["unequal"] or not _teaches([pair],("group_color","top_anchor")):
                        continue
                    if unused and not _features(a)["unused"]:
                        continue
                break
            else:
                raise RuntimeError("bounded placement/ruler construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"},"wrong puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in puzzle),"splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]),"wrong pair counts")
    seen,analyses = set(),{"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {"input","output"},"wrong pair keys")
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            _require(inp.shape == out.shape,"output dimensions changed")
            key = inp.shape,inp.tobytes()
            _require(key not in seen,"duplicate input")
            seen.add(key)
            a = _analyze(inp)
            _require(np.array_equal(solve(inp),out),"incorrect shape printing")
            _require(not np.array_equal(inp,out),"identity pair")
            comps,bands,bg = a["components"],a["bands"],a["background"]
            points = {p for c in comps for p in c["cells"]}
            _require(len(_components(a["canonical"],points,True)) == len(comps),"ambiguous diagonal grouping")
            canout = np.rot90(np.fliplr(out) if a["mirror"] else out,a["turn"])
            _require(np.all(canout[:a["first"]] == bg),"staging not cleared")
            for band in bands:
                ys = slice(band["top"],band["bottom"]+1)
                for x in (band["left"],band["right"]):
                    _require(np.array_equal(canout[ys,x],a["canonical"][ys,x]),"ruler/solid rail changed")
            analyses[split].append(a)
    fs = [_features(a) for a in analyses["train"]]
    for feature in ("contact","unequal","boundary","terminal"):
        _require(any(f[feature] for f in fs),"missing training lesson: "+feature)
    _require(_teaches(puzzle["train"],COUNTER_RULES),"training admits a documented shortcut")
    if any(_features(a)["noise"] for a in analyses["test"]):
        _require(any(f["noise"] for f in fs),"off-phase preservation not taught")
        _require(_teaches(puzzle["train"],("offphase_stop",)),"off-phase stopping shortcut not excluded")
    if any(_features(a)["unused"] for a in analyses["test"]):
        _require(any(f["unused"] for f in fs),"unused chamber not taught")
    turns = {a["turn"] for a in analyses["train"]}
    if any(a["turn"] not in turns for a in analyses["test"]):
        _require(len(turns) >= 2,"orientation transfer requires varied demonstrations")


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                g = np.asarray(pair[role],dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, streaming 42-seed coverage plus analytic metamorphic checks."""
    shapes,masks,periods,palettes,firsts,counts,dirs = set(),set(),set(),set(),set(),set(),set()
    counter = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        firsts.add(_puzzle_digest(puzzle))
        for split in ("train","test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                palettes.add(a["background"])
                counts.add(len(a["bands"]))
                dirs.add((a["turn"],a["mirror"]))
                masks.update(_signature(c["cells"]) for c in a["components"])
                periods.update(b["period"] for b in a["bands"])
                f = _features(a)
                counter.update({k:int(bool(v)) for k,v in f.items()})
                counter["left_terminal"] += any(b["terminal"] is not None and b["terminal"] < b["a"] for b in a["bands"])
                counter["right_terminal"] += any(b["terminal"] is not None and b["terminal"] > b["b"] for b in a["bands"])
        # Equivariance checks use arbitrary bijective hues and all scene poses.
        pair = puzzle["test"][0]
        perm = np.random.default_rng(seed+1).permutation(10)
        for turn in range(4):
            for flip in (False,True):
                inp = np.rot90(pair["input"],turn)
                out = np.rot90(pair["output"],turn)
                if flip:
                    inp,out = np.fliplr(inp),np.fliplr(out)
                _require(np.array_equal(solve(perm[inp]),perm[out]),"pose/palette equivariance failed")
    _require(len(firsts) == 42 and len(shapes) >= 40 and len(masks) >= 40,"insufficient geometric variation")
    _require(counts == {1,2,3} and len(dirs) == 8 and len(palettes) == 10,"missing structural/palette modes")
    _require({1,2,3} <= periods and all(counter[k] for k in
             ("noise","unused","contact","unequal","left_terminal","right_terminal")),"unexercised concepts")
    for train in range(2,11):
        for test in range(1,min(4,train-1)+1):
            generate_puzzle(4000+train*10+test,train,test)
    for args in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,*args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_puzzle_digest(generate_puzzle(0)) == _puzzle_digest(generate_puzzle(0)),"nondeterministic generation")
    # Exact miniature formula: a 2-cell vertical mask, period two, no clipping.
    inp = np.zeros((12,10),dtype=np.int8)
    inp[0:2,3] = 1
    inp[5:12,1:9] = 1
    inp[5:12,9] = 2
    inp[[7,9],0] = 2
    expected = inp.copy()
    expected[:5] = 0
    expected[6:12,3] = 2
    _require(np.array_equal(solve(inp),expected),"hand-computed stencil regression failed")
    inp[11,0] = 3
    expected[11,0] = 3
    expected[6:12,3] = 3
    _require(np.array_equal(solve(inp),expected),
             "hand-computed terminal regression failed")
    # Unequal-depth L + singleton: the ruler bounds anchors, not stencil cells.
    inp = np.zeros((17,12),dtype=np.int8)
    inp[0,3] = inp[1,3] = inp[1,4] = inp[3,7] = 1
    inp[6:17,1:11] = 1
    inp[6:17,11] = 2
    inp[[9,12],0] = 2
    inp[6,0],inp[7,0] = 3,4  # upper terminal and an off-phase speck
    expected = inp.copy()
    expected[:6] = 0
    for y in (9,12):
        expected[y-1,3] = expected[y,3] = expected[y,4] = 3
    for y in (6,9,12):
        expected[y,7] = 3
    _require(np.array_equal(solve(inp),expected),"L/singleton/upper-terminal regression failed")
    # Minimal stage and six-cell interior: supported beyond the sampled sizes.
    inp = np.zeros((5,8),dtype=np.int8)
    inp[0,3] = 1
    inp[2:5,1:7] = 1
    inp[2:5,7] = 2
    inp[[2,3],0] = 2
    expected = inp.copy()
    expected[0,3] = 0
    expected[2:5,3] = 2
    _require(np.array_equal(solve(inp),expected),"minimum-stage family member failed")
    puzzle = generate_puzzle(55,2,1)
    puzzle["test"][0]["output"][0,0] = (int(puzzle["test"][0]["output"][0,0])+1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
