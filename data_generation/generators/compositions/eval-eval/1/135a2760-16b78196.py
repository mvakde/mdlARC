"""Reweave periodic silhouettes with conserved, translation-only colored scraps.

The prediction path (_parse, _recover, _covers) receives only the input. No parent
module, seed, cached example, output label, or generation metadata enters it.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


HOST_SEEDS = (0,220033151,979690549,83394994,947754863,625658095,2061547,274115354,908791594,828256513,321783301,111400832,135901973,338489939,159984847,751826530,752872679,374609728,752338395,933551420,443690245,889392659,440680775,595497623,686277250,332675007,890934869,147209650,752921858,271914349,403090134,651611418,735342535,431213011,525589428,172148471,528354026,914851201,341134939,838447213,124060304,330871448)


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grids must be two dimensional with dimensions 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid cells must be integers 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _neighbors(p):
    r, c = p
    return ((r-1,c),(r+1,c),(r,c-1),(r,c+1))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, component = [first], []
        while stack:
            p = stack.pop()
            component.append(p)
            for q in _neighbors(p):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(component)))
    return result


def _dilate(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    result = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            result |= padded[dr:dr+h, dc:dc+w]
    return result


def _parse(grid):
    """Locate a rectangular frame geometrically, then normalize its long axis."""
    candidates = []
    for color in np.unique(grid):
        cells = np.argwhere(grid == color)
        low, high = cells.min(axis=0), cells.max(axis=0)+1
        r0, c0 = map(int, low)
        r1, c1 = map(int, high)
        box = grid[r0:r1, c0:c1]
        if min(box.shape) < 7 or max(box.shape) < 20:
            continue
        if not (np.all(box[0] == color) and np.all(box[-1] == color)
                and np.all(box[:,0] == color) and np.all(box[:,-1] == color)):
            continue
        transpose = box.shape[0] > box.shape[1]
        canonical = grid.T if transpose else grid
        if transpose:
            r0, c0, r1, c1 = c0, r0, c1, r1
        box = canonical[r0:r1,c0:c1]
        s = int(color)
        if box.shape[1] < box.shape[0] or box.shape[1]-4 < 16:
            continue
        dividers = np.flatnonzero(np.all(box == s, axis=1)).tolist()
        if not (2 <= len(dividers) <= 3 and dividers[0] == 0
                and dividers[-1] == box.shape[0]-1):
            continue
        rectangles = []
        padding_values = set()
        valid = True
        for first, last in zip(dividers[:-1], dividers[1:]):
            thickness = last-first-3
            if not 3 <= thickness <= 6:
                valid = False
                break
            rr = slice(r0+first+2, r0+last-1)
            cc = slice(c0+2, c1-2)
            rectangles.append((rr,cc))
            inner = box[first+1:last,1:-1]
            padding_values.update(map(int, inner[0]))
            padding_values.update(map(int, inner[-1]))
            padding_values.update(map(int, inner[:,0]))
            padding_values.update(map(int, inner[:,-1]))
        if not valid or len(padding_values) != 1 or s in padding_values:
            continue
        bg = padding_values.pop()
        if any(not set(map(int,np.unique(canonical[rect]))) <= {bg,s}
               for rect in rectangles):
            continue
        inside = np.zeros_like(canonical, dtype=bool)
        inside[r0:r1,c0:c1] = True
        # All fixed material belongs to the framed panel; all colored scraps
        # are outside it and are separate components, even when colors repeat.
        if np.any((canonical == s) & ~inside):
            continue
        if np.any(inside & (canonical != bg) & (canonical != s)):
            continue
        pieces = []
        for payload in np.unique(canonical):
            if payload in (bg,s):
                continue
            for component in _components(map(tuple,np.argwhere(canonical == payload))):
                cells2 = np.asarray(component,dtype=int)
                pieces.append((cells2,int(payload)))
        if not 2 <= len(pieces) <= 5:
            continue
        forbidden = _dilate(inside)
        occupied = np.zeros_like(inside)
        for cells2, _ in pieces:
            if np.any(forbidden[cells2[:,0],cells2[:,1]]) or np.any(
                    _dilate(occupied)[cells2[:,0],cells2[:,1]]):
                valid = False
                break
            occupied[cells2[:,0],cells2[:,1]] = True
        if valid:
            candidates.append({"grid":canonical,"transpose":transpose,
                               "background":bg,"structure":s,"rectangles":rectangles,
                               "pieces":pieces,"panel":(r0,c0,r1,c1)})
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous padded ribbon panel")
    return candidates[0]


def _recover(observed):
    """All strict-majority, deletion-only explanations with >=3 copies/phase.

    The erasure budget is one third of the motif rectangle. Redundant periods
    may agree; competing repaired masks are rejected instead of tie-broken.
    """
    h, length = observed.shape
    repairs = {}
    for period in range(1,length//3+1):
        unit = np.zeros((h,period),dtype=bool)
        valid = True
        for phase in range(period):
            samples = observed[:,phase::period]
            count = samples.sum(axis=1)
            if np.any(2*count == samples.shape[1]):
                valid = False
                break
            unit[:,phase] = 2*count > samples.shape[1]
        if not valid:
            continue
        clean = unit[:,np.arange(length)%period]
        errors = int(np.count_nonzero(clean & ~observed))
        if np.any(observed & ~clean) or not 1 <= errors <= observed.size//3:
            continue
        if not np.any(clean) or np.all(clean):
            continue
        key = clean.tobytes()
        if key not in repairs:
            repairs[key] = (clean,period,errors)
    if len(repairs) != 1:
        raise ValueError("periodic silhouette has competing or absent repairs")
    return next(iter(repairs.values()))


def _demand(info):
    canonical = info["grid"]
    target = np.zeros_like(canonical,dtype=bool)
    clean_masks, periods, errors = [], [], []
    for rect in info["rectangles"]:
        observed = canonical[rect] == info["structure"]
        clean, period, count = _recover(observed)
        target[rect] = clean & ~observed
        clean_masks.append(clean)
        periods.append(period)
        errors.append(count)
    return target, clean_masks, periods, errors


def _covers(target, pieces, limit=2):
    """Exact cover over every translation into the recovered demand.

    Deduplicate actual target colors, not permutations of identical components.
    Stop after two genuinely different answers: uniqueness is all we require.
    """
    points = sorted(map(tuple,np.argwhere(target)))
    point_index = {p:i for i,p in enumerate(points)}
    full = (1 << len(points))-1
    if not points or sum(len(cells) for cells,_ in pieces) != len(points):
        return []
    candidates = [[] for _ in points]
    for i,(cells,color) in enumerate(pieces):
        local = cells-cells.min(axis=0)
        anchor = local[0]
        seen = set()
        for p in points:
            moved = local+np.asarray(p)-anchor
            indices = [point_index.get(tuple(q)) for q in moved]
            if any(k is None for k in indices):
                continue
            bits = sum(1 << k for k in indices)
            if bits in seen:
                continue
            seen.add(bits)
            item = (i,bits,tuple(indices),moved,color)
            for k in indices:
                candidates[k].append(item)
    solutions = {}
    all_pieces = (1 << len(pieces))-1

    def visit(used, filled, placements, colors):
        if len(solutions) >= limit:
            return
        if filled == full:
            if used == all_pieces:
                solutions.setdefault(bytes(colors),list(placements))
            return
        best = None
        for k in range(len(points)):
            if filled & (1 << k):
                continue
            options = [x for x in candidates[k]
                       if not used & (1 << x[0]) and not filled & x[1]]
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        local_seen = set()
        for i,bits,indices,moved,color in best:
            symmetry_key = (bits,color)
            if symmetry_key in local_seen:
                continue
            local_seen.add(symmetry_key)
            next_colors = colors.copy()
            for k in indices:
                next_colors[k] = color
            visit(used | (1 << i),filled | bits,placements+[(i,moved)],next_colors)
    visit(0,0,[],[0]*len(points))
    return list(solutions.values())


def _analyze(input_grid):
    original = _grid(input_grid)
    info = _parse(original)
    target, clean, periods, errors = _demand(info)
    solutions = _covers(target,info["pieces"])
    if len(solutions) != 1:
        raise ValueError("scraps must have one distinct colored exact cover")
    answer = info["grid"].copy()
    for cells,_ in info["pieces"]:
        answer[cells[:,0],cells[:,1]] = info["background"]
    for i,cells in solutions[0]:
        answer[cells[:,0],cells[:,1]] = info["pieces"][i][1]
    info.update(target=target,clean=clean,periods=periods,errors=errors,
                placements=solutions[0],answer=answer)
    return (answer.T if info["transpose"] else answer).copy(),info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the demand by repetition and translate all scraps into it."""
    return _analyze(input_grid)[0]


def _template(rng,height,period):
    # A full spine ensures connected silhouettes. Sparse extremal teeth make
    # intentional bays distinguishable from an all-solid completion.
    spine = int(rng.integers(1,height-1))
    lower = rng.integers(1,spine+1,size=period)
    upper = rng.integers(spine+1,height,size=period)
    lower[int(rng.integers(period))] = 0
    for c in rng.choice(period,size=int(rng.integers(1,max(2,period//2+1))),replace=False):
        upper[int(c)] = height
    return np.asarray([[lower[c] <= r < upper[c] for c in range(period)]
                       for r in range(height)],dtype=bool)


def _partition(rng,cells,count):
    """Connected randomized region growth; complementary edges emerge jointly."""
    domain = set(cells)
    if len(domain) < 4*count:
        raise ValueError("not enough material for nontrivial scraps")
    ordered = sorted(domain)
    seeds = [ordered[int(rng.integers(len(ordered)))]]
    while len(seeds) < count:
        distances = [min(abs(p[0]-q[0])+abs(p[1]-q[1]) for q in seeds) for p in ordered]
        maximum = max(distances)
        choices = [p for p,d in zip(ordered,distances) if d == maximum]
        seeds.append(choices[int(rng.integers(len(choices)))])
    labels = {p:i for i,p in enumerate(seeds)}
    frontiers = [set(q for q in _neighbors(p) if q in domain and q not in labels) for p in seeds]
    sizes = [1]*count
    while len(labels) < len(domain):
        available = [i for i,f in enumerate(frontiers) if f]
        if not available:
            raise ValueError("disconnected cut")
        weights = np.array([1/(sizes[i]**.6) for i in available])
        i = int(rng.choice(available,p=weights/weights.sum()))
        options = sorted(frontiers[i])
        p = options[int(rng.integers(len(options)))]
        labels[p] = i
        sizes[i] += 1
        for f in frontiers:
            f.discard(p)
        frontiers[i].update(q for q in _neighbors(p) if q in domain and q not in labels)
    regions = [np.array(sorted(p for p,v in labels.items() if v == i),dtype=int) for i in range(count)]
    if any(len(p) < 3 or min(np.ptp(p,axis=0)) == 0 for p in regions):
        raise ValueError("scraps must have area and two-dimensional geometry")
    return regions


def _interlocks(placements):
    boxes = [(p.min(axis=0),p.max(axis=0)) for p in placements]
    return any(np.all(np.maximum(a,c) <= np.minimum(b,d))
               for i,(a,b) in enumerate(boxes) for c,d in boxes[:i])


def _nonperiodic_packing(grid,rectangles,clean_masks,pieces,max_nodes=10000):
    """Bounded omission witness: use all pieces in voids, but break repetition.

    This is a diagnostic, never a prediction path. It removes only the recovered
    occupancy constraint, retaining all rigid shapes, colors and non-overlap.
    A failed/capped search is not a proof that no witness exists.
    """
    options = []
    for cells,_ in pieces:
        local = cells-cells.min(axis=0)
        choices = []
        for rect,clean in zip(rectangles,clean_masks):
            view = grid[rect]
            # Existing material is non-background; its exact number is irrelevant.
            bg = int(grid[rect[0].start-1,rect[1].start])
            for r in range(view.shape[0]-int(local[:,0].max())):
                for c in range(view.shape[1]-int(local[:,1].max())):
                    moved = local+(r,c)
                    if not np.all(view[moved[:,0],moved[:,1]] == bg):
                        continue
                    cells2 = moved+np.array([rect[0].start,rect[1].start])
                    wrong = bool(np.any(~clean[moved[:,0],moved[:,1]]))
                    choices.append((cells2,frozenset(map(tuple,cells2)),wrong))
        options.append(sorted(choices,key=lambda x:not x[2]))
    order = sorted(range(len(pieces)),key=lambda i:len(options[i]))
    nodes = 0

    def visit(k,occupied,wrong,path):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return None
        if k == len(order):
            return path if wrong else None
        i = order[k]
        for cells,points,bad in options[i]:
            if not occupied & points:
                result = visit(k+1,occupied | points,wrong or bad,path+[(i,cells)])
                if result is not None:
                    return result
            if nodes > max_nodes:
                break
        return None
    return visit(0,frozenset(),False,[])


def _scatter(rng,base,panel,pieces,bg):
    inside = np.zeros_like(base,dtype=bool)
    r0,c0,r1,c1 = panel
    inside[r0:r1,c0:c1] = True
    local = [(cells-cells.min(axis=0),color) for cells,color in pieces]
    for _ in range(12):
        result = base.copy()
        blocked = _dilate(inside)
        order = sorted(range(len(local)),key=lambda i:-len(local[i][0])+rng.random()*3)
        success = True
        for i in order:
            cells,color = local[i]
            h,w = cells.max(axis=0)+1
            vh,vw = base.shape[0]-h+1,base.shape[1]-w+1
            if min(vh,vw) < 1:
                success = False
                break
            valid = np.ones((vh,vw),dtype=bool)
            for r,c in cells:
                valid &= ~blocked[r:r+vh,c:c+vw]
            origins = np.argwhere(valid)
            if not len(origins):
                success = False
                break
            moved = cells+origins[int(rng.integers(len(origins)))]
            result[moved[:,0],moved[:,1]] = color
            placed = np.zeros_like(base,dtype=bool)
            placed[moved[:,0],moved[:,1]] = True
            blocked |= _dilate(placed)
        if success:
            return result
    raise ValueError("scatter clearance")


def _propose(rng,kind,transpose,palette):
    bg,s,payload = palette
    # kind 0: single cut; kind 1: multiple demands; kind 2: test recombination;
    # kind 3: guarantee an independent-period demonstration even at (2,1).
    lanes = 1 if kind == 0 else (2 if kind == 3 else int(rng.integers(1,3)))
    length = int(rng.integers(18 if kind != 2 else 23,27))
    heights = [int(rng.integers(3,6 if kind != 2 else 7)) for _ in range(lanes)]
    periods = [int(rng.integers(2,min(5,length//4)+1)) for _ in range(lanes)]
    if lanes == 2 and periods[0] == periods[1]:
        periods[1] = int(rng.choice([p for p in range(2,min(5,length//4)+1) if p != periods[0]]))
    panel_h,panel_w = 1+sum(h+3 for h in heights),length+4
    if panel_h > 20:
        raise ValueError("panel clearance")
    canvas_w = int(rng.integers(panel_w,31))
    canvas_h = int(rng.integers(max(panel_h+9,19),31))
    r0 = int(rng.integers(0,min(3,canvas_h-panel_h-8)+1))
    c0 = int(rng.integers(canvas_w-panel_w+1))
    panel = (r0,c0,r0+panel_h,c0+panel_w)
    base = np.full((canvas_h,canvas_w),bg,dtype=np.int8)
    base[r0:r0+panel_h,c0] = s
    base[r0:r0+panel_h,c0+panel_w-1] = s
    rects,cleans = [],[]
    row = r0
    base[row,c0:c0+panel_w] = s
    for h,p in zip(heights,periods):
        clean = _template(rng,h,p)[:,np.arange(length)%p]
        rect = (slice(row+2,row+2+h),slice(c0+2,c0+2+length))
        base[rect] = np.where(clean,s,bg)
        rects.append(rect)
        cleans.append(clean)
        row += h+3
        base[row,c0:c0+panel_w] = s
    cuts = 1 if kind == 0 else 2
    lane_ids = [0] if cuts == 1 else ([0,1] if lanes == 2 else [0,0])
    cut_domains = []
    demand = np.zeros_like(base,dtype=bool)
    for cut,lane in enumerate(lane_ids):
        rect = rects[lane]
        h = heights[lane]
        width = int(rng.integers(3,6 if kind != 2 else 7))
        starts = list(range(length-width+1))
        rng.shuffle(starts)
        for start in starts:
            # Separate cuts leave visible material between them. The mask may
            # cover only some cross-sectional rows, including edge-only cuts.
            if any(k == lane and not (start+width+1 <= a or a+w+1 <= start)
                   for k,a,w in cut_domains):
                continue
            cut_domains.append((lane,start,width))
            break
        else:
            raise ValueError("cut clearance")
        top,bottom = (0,h)
        if kind != 0 and h >= 4 and rng.random() < .45:
            if rng.random() < .5:
                bottom -= 1
            else:
                top += 1
        patch = np.zeros_like(cleans[lane])
        patch[top:bottom,start:start+width] = cleans[lane][top:bottom,start:start+width]
        demand[rect] |= patch
    pieces = []
    regions = _components(map(tuple,np.argwhere(demand)))
    if len(regions) != cuts:
        raise ValueError("cuts must be connected and separate")
    if not any(len(region) < int(np.prod(np.ptp(np.asarray(region),axis=0)+1))
               for region in regions):
        raise ValueError("need a nonrectangular periodic demand")
    counts = [2] if cuts == 1 else ([2,1] if kind in (1,3) else [2,int(rng.integers(1,4))])
    rng.shuffle(counts)
    for cells,count in zip(regions,counts):
        pieces.extend(_partition(rng,cells,count))
    if not _interlocks(pieces):
        raise ValueError("need genuine complementary interlocking")
    for rect,clean in zip(rects,cleans):
        observed = clean & ~demand[rect]
        recovered,_,_ = _recover(observed)
        if not np.array_equal(recovered,clean):
            raise ValueError("construction and repetition disagree")
    colors = list(payload[:len(pieces)])
    if len(pieces) >= 3 and rng.random() < .45:
        colors[-1] = colors[0]
    rng.shuffle(colors)
    colored = list(zip(pieces,colors))
    # Check uniqueness before scattering; the cover depends only on the masks
    # and colors, so origins cannot disambiguate an otherwise ambiguous answer.
    covers = _covers(demand,colored)
    if len(covers) != 1:
        raise ValueError("ambiguous colored packing")
    damaged = base.copy()
    damaged[demand] = bg
    if kind == 3 and _nonperiodic_packing(damaged,rects,cleans,colored) is None:
        raise ValueError("teaching example needs a complete nonperiodic packing")
    inp = _scatter(rng,damaged,panel,colored,bg)
    out = damaged.copy()
    for i,cells in covers[0]:
        out[cells[:,0],cells[:,1]] = colored[i][1]
    if rng.random() < .5:
        inp,out = inp[::-1],out[::-1]
    if rng.random() < .5:
        inp,out = inp[:,::-1],out[:,::-1]
    if transpose:
        inp,out = inp.T,out.T
    return {"input":inp.copy(),"output":out.copy()}


def _checked_count(value,name,low,high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _checked_count(num_train,"num_train",2,10)
    b = _checked_count(num_test,"num_test",1,4)
    if a <= b:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    # Stable palettes are a sampling preference, never a hidden decoding rule.
    stable_palette = rng.random() < .35
    shared = tuple(map(int,rng.permutation(10)))
    orientation_mode = int(rng.integers(5))
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",a),("test",b)):
        kinds = ([0,3]+[int(rng.integers(2)) for _ in range(count-2)]
                 if split == "train" else [2]*count)
        if split == "train":
            rng.shuffle(kinds)
        for index,kind in enumerate(kinds):
            if orientation_mode < 2:
                transpose = bool(orientation_mode)
            elif orientation_mode < 4:
                transpose = (split == "test") == (orientation_mode == 2)
            else:
                transpose = bool(index % 2)
            palette_values = shared if stable_palette else tuple(map(int,rng.permutation(10)))
            palette = (palette_values[0],palette_values[1],palette_values[2:])
            for _ in range(400):
                try:
                    pair = _propose(rng,kind,transpose,palette)
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded ribbon construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _pair_invariants(pair):
    if not isinstance(pair,dict) or set(pair) != {"input","output"}:
        raise ValueError("pair must have exactly input and output")
    inp,out = _grid(pair["input"]),_grid(pair["output"])
    assert inp.shape == out.shape
    expected,info = _analyze(inp)
    assert np.array_equal(expected,out), "incorrect assembly label"
    bg,s = info["background"],info["structure"]
    assert np.array_equal(inp == s,out == s), "fixed material changed"
    assert np.array_equal(np.bincount(inp.ravel(),minlength=10),
                          np.bincount(out.ravel(),minlength=10)), "material not conserved"
    canonical = out.T if info["transpose"] else out
    assert not np.array_equal(inp,out)
    assert len({color for _,color in info["pieces"]}) >= 2
    assert _interlocks([cells for _,cells in info["placements"]])
    regions = _components(map(tuple,np.argwhere(info["target"])))
    assert 1 <= len(regions) <= 2
    assert any(len(region) < int(np.prod(np.ptp(np.asarray(region),axis=0)+1))
               for region in regions), "periodic mold must include a nonrectangular demand"
    for i,cells in info["placements"]:
        source,color = info["pieces"][i]
        assert len(source) >= 3 and min(np.ptp(source,axis=0)) >= 1
        translated = source+cells.min(axis=0)-source.min(axis=0)
        assert set(map(tuple,translated)) == set(map(tuple,cells))
        assert np.any(cells.min(axis=0) != source.min(axis=0))
        assert np.all(canonical[cells[:,0],cells[:,1]] == color)
    for rect,clean in zip(info["rectangles"],info["clean"]):
        assert np.array_equal(canonical[rect] != bg,clean)
        assert np.any(~clean), "intentional negative space must remain"
    return info


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle,dict) or set(puzzle) != {"train","test"}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s],list) for s in ("train","test")):
        raise ValueError("splits must be lists")
    a = _checked_count(len(puzzle["train"]),"num_train",2,10)
    b = _checked_count(len(puzzle["test"]),"num_test",1,4)
    if a <= b:
        raise ValueError("num_train must be greater than num_test")
    seen = set()
    train_counts,train_cuts,train_lanes = set(),set(),set()
    omission_witness = False
    for split in ("train","test"):
        for pair in puzzle[split]:
            info = _pair_invariants(pair)
            key = _key(np.asarray(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            if split == "train":
                train_counts.add(len(info["pieces"]))
                train_cuts.add(len(_components(map(tuple,np.argwhere(info["target"])))))
                train_lanes.add(len(info["rectangles"]))
                if not omission_witness:
                    omission_witness = _nonperiodic_packing(
                        info["grid"],info["rectangles"],info["clean"],info["pieces"]) is not None
    assert {2,3} <= train_counts, "demonstrate different fragment counts"
    assert {1,2} <= train_cuts, "demonstrate single and separated reconstruction regions"
    assert train_lanes == {1,2}, "demonstrate independent lanes at every legal count"
    assert omission_witness, "training must disambiguate a full packing by periodicity"


def _signature(puzzle):
    return tuple(_key(np.asarray(p[k])) for split in ("train","test")
                 for p in puzzle[split] for k in ("input","output"))


def self_test():
    """Stream 42 host seeds; retain small fixtures; write no files or traces."""
    shapes,periods,lanes,counts,cuts,roles,orientations,structures = (set() for _ in range(8))
    total,repeat_colors = 0,0
    signatures = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(hashlib.sha256(repr(_signature(puzzle)).encode()).hexdigest())
        for split in ("train","test"):
            for pair in puzzle[split]:
                _,info = _analyze(pair["input"])
                total += 1
                shapes.add(pair["input"].shape)
                periods.update(info["periods"])
                lanes.add(len(info["rectangles"]))
                counts.add(len(info["pieces"]))
                cuts.add(len(_components(map(tuple,np.argwhere(info["target"])))))
                roles.add((info["background"],info["structure"]))
                orientations.add(info["transpose"])
                repeat_colors += len({color for _,color in info["pieces"]}) < len(info["pieces"])
                mapping,normalized = {},[]
                for value in pair["input"].flat:
                    value = int(value)
                    normalized.append(mapping.setdefault(value,len(mapping)))
                structures.add((pair["input"].shape,bytes(normalized)))
                # Concrete omission outputs, not just feature-presence checks.
                no_b = info["grid"].copy()
                for cells,_ in info["pieces"]:
                    no_b[cells[:,0],cells[:,1]] = info["background"]
                no_b[info["target"]] = info["structure"]
                assert not np.array_equal(no_b,info["answer"])
                # Periodic occupancy plus pigment counts alone also admits a
                # wrong answer: exchange two differently colored demanded cells.
                positions = np.argwhere(info["target"])
                first = tuple(positions[0])
                second = next(tuple(q) for q in positions
                              if info["answer"][tuple(q)] != info["answer"][first])
                mask_only = info["answer"].copy()
                mask_only[first],mask_only[second] = mask_only[second],mask_only[first]
                assert not np.array_equal(mask_only,info["answer"])
                assert np.array_equal(mask_only != info["background"],
                                      info["answer"] != info["background"])
                assert np.array_equal(np.bincount(mask_only.ravel(),minlength=10),
                                      np.bincount(info["answer"].ravel(),minlength=10))
                no_a = info["answer"].copy()
                for rect in info["rectangles"]:
                    no_a[rect] = np.where(no_a[rect] == info["background"],
                                          info["structure"],no_a[rect])
                assert not np.array_equal(no_a,info["answer"])
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(structures) >= 200
    assert periods >= {2,3,4,5} and lanes == {1,2}
    assert counts >= {2,3,4,5} and cuts == {1,2}
    assert orientations == {False,True} and len(roles) >= 35
    assert repeat_colors >= 20
    p = generate_puzzle(19,2,1)
    assert _signature(p) == _signature(generate_puzzle(19,2,1))
    for a,b in ((2,1),(10,4),(7,3)):
        q = generate_puzzle(100+a, a,b)
        validate_generated_puzzle(q)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong label accepted")
    # Distinct colored outputs vs component permutations.
    target = np.zeros((4,7),dtype=bool)
    target[:2,:2] = True
    target[:2,4:6] = True
    square = np.argwhere(target[:2,:2])
    assert len(_covers(target,[(square,2),(square,3)])) == 2
    assert len(_covers(target,[(square,2),(square,2)])) == 1
    # The only fitting shape would need rotation: translation cannot fill it.
    target2 = np.zeros((4,4),dtype=bool)
    target2[:3,:2] = True
    bar = np.array([(r,c) for r in range(2) for c in range(3)])
    assert not _covers(target2,[(bar,2)])
    return {"seeds":42,"pairs":total,"shapes":len(shapes),"structures":len(structures),
            "periods":sorted(periods),"lane_counts":sorted(lanes),"piece_counts":sorted(counts),
            "missing_regions":sorted(cuts),"role_assignments":len(roles),
            "reused_color_pairs":repeat_colors,"independent_solver":False}


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",",":")))
