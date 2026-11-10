"""Pinwheel loom: intrinsic-route sorting of local-frame cuff intervals.

Only NumPy and the standard library are used. Generation has no parent imports,
fixtures, stored examples, or hidden conventions consumed by solve().
"""
from __future__ import annotations

from collections import Counter
import numpy as np


# Cardinal directions in clockwise order. A right normal to (dy,dx) is (dx,-dy).
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grids must be 2D integer arrays of size 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _paths(radius, shape, center):
    """Four disjoint infinite-plane paths, continued beyond all possible re-entry.

    A complete four-stage cycle translates the enclosing envelope outward by
    eight. Two extra cycles suffice after the furthest canvas coordinate.
    Each list starts at distance 1 from the hole; its index plus one is distance.
    """
    distance = max(center[0], center[1], shape[0]-1-center[0], shape[1]-1-center[1])
    stages = 4 * (distance // 8 + 2)
    paths = []
    for phase in range(4):
        y, x = center
        points = []
        moves = [(phase, radius)]
        for k in range(stages):
            direction = (phase + 1 + k) % 4
            moves.extend(((direction, 2+4*k), ((direction+1) % 4, radius-2),
                          (direction, radius)))
        for direction, count in moves:
            dy, dx = _DIRECTIONS[direction]
            for _ in range(count):
                y, x = y+dy, x+dx
                points.append((y, x))
        paths.append(points)
    return paths


def _carrier_mask(radius, shape, center):
    mask = np.zeros(shape, dtype=bool)
    for points in _paths(radius, shape, center):
        for y, x in points:
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                mask[y, x] = True
    return mask


def _analytic_carrier(radius, shape, center):
    """Closed-form segment membership; no walk or painter calls.

    Stage-start formulas follow the stepped recurrence in parent B's final
    generator. This is an author cross-check, not an independent ARC oracle.
    """
    yy, xx = np.indices(shape)
    yy, xx = yy-center[0], xx-center[1]
    bound = max(int(np.abs(yy).max()), int(np.abs(xx).max()))
    mask = np.zeros(shape, dtype=bool)
    for _ in range(4):
        mask |= (xx == 0) & (yy >= -radius) & (yy <= -1)
        for q in range(bound // 8 + 3):
            starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                      (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
            for s, ((y, x), (dy, dx)) in enumerate(zip(starts,
                    ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along = (yy-y)*dy + (xx-x)*dx
                across = (yy-y)*dx - (xx-x)*dy
                run = 16*q + 4*s + 2
                mask |= (across == 0) & (along >= 0) & (along <= run)
                mask |= (along == run) & (across >= 0) & (across <= radius-2)
                mask |= (across == radius-2) & (along >= run) & (along <= run+radius)
        yy, xx = xx, -yy
    return mask


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _slot_cells(slot, span):
    y, x = slot["position"]
    dy, dx = slot["direction"]
    lo, hi = span
    return [(y+j*dy+k*dx, x+j*dx-k*dy)
            for j in range(slot["width"]) for k in range(lo, hi+1)]


def _collar_cells(slot):
    y, x = slot["position"]
    dy, dx = slot["direction"]
    return [(y+j*dy, x+j*dx) for j in (-2, -1, slot["width"], slot["width"]+1)]


def _sparse_input(shape, center, radius, slots, background, carrier):
    grid = np.full(shape, background, dtype=np.int8)
    cy, cx = center
    for dy, dx in _DIRECTIONS:
        for d in range(1, radius+1):
            grid[cy+d*dy, cx+d*dx] = carrier
    for slot in slots:
        for y, x in _collar_cells(slot):
            grid[y, x] = carrier
        for y, x in _slot_cells(slot, slot["span"]):
            grid[y, x] = slot["color"]
    return grid


def _paint_output(shape, center, radius, slots, background, carrier):
    out = np.where(_carrier_mask(radius, shape, center), carrier, background).astype(np.int8)
    ranked = sorted(slots, key=lambda s: s["distance"])
    spans = sorted((s["span"] for s in slots), key=lambda p: p[1]-p[0])
    for slot, span in zip(ranked, spans):
        for y, x in _slot_cells(slot, span):
            out[y, x] = slot["color"]
    return out


def _parse(input_grid):
    """Recover all state from visible geometry, including collar orientation."""
    grid = _grid(input_grid)
    h, w = grid.shape
    background = int(grid[0, 0])
    assert all(int(grid[y, x]) == background for y, x in ((0,w-1),(h-1,0),(h-1,w-1)))
    counts = np.bincount(grid.ravel(), minlength=10)
    assert counts[background] > max(counts[c] for c in range(10) if c != background)
    holes = []
    for y, x in np.argwhere(grid[1:-1, 1:-1] == background) + 1:
        neighbors = [int(grid[y+dy, x+dx]) for dy, dx in _DIRECTIONS]
        if neighbors[0] != background and len(set(neighbors)) == 1:
            holes.append((int(y), int(x), neighbors[0]))
    assert len(holes) == 1, "one unambiguous hollow cross required"
    cy, cx, carrier = holes[0]
    lengths = []
    for dy, dx in _DIRECTIONS:
        d = 1
        while (0 <= cy+d*dy < h and 0 <= cx+d*dx < w
               and grid[cy+d*dy, cx+d*dx] == carrier):
            d += 1
        lengths.append(d-1)
    assert len(set(lengths)) == 1 and 2 <= lengths[0] <= 5
    radius = lengths[0]
    center = (cy, cx)
    paths = _paths(radius, grid.shape, center)
    address = {p: (phase, t+1) for phase, pts in enumerate(paths) for t, p in enumerate(pts)}
    assert sum(map(len, paths)) == len(address), "winding arms may not intersect"
    slots = []
    accents = sorted(set(map(int, np.unique(grid))) - {background, carrier})
    assert 2 <= len(accents) <= 7
    for color in accents:
        cells = np.argwhere(grid == color)
        top, left = map(int, cells.min(axis=0))
        bottom, right = map(int, cells.max(axis=0))
        assert len(cells) == (bottom-top+1)*(right-left+1), "cuffs must be solid rectangles"
        candidates = []
        for horizontal in (True, False):
            for axis in range(top if horizontal else left, (bottom if horizontal else right)+1):
                line = ([(axis,x) for x in range(left-2,right+3)] if horizontal else
                        [(y,axis) for y in range(top-2,bottom+3)])
                if any(not (0 <= y < h and 0 <= x < w) for y, x in line):
                    continue
                if any(grid[y,x] != carrier for y,x in line[:2]+line[-2:]):
                    continue
                if any(p not in address for p in line):
                    continue
                phase, first = address[line[0]]
                step = address[line[1]][1]-first
                if abs(step) != 1 or any(address[p] != (phase,first+i*step) for i,p in enumerate(line)):
                    continue
                base = line[2:-2]
                if step < 0:
                    base.reverse()
                y, x = base[0]
                dy, dx = ((0,step) if horizontal else (step,0))
                lo, hi = ((top-y,bottom-y) if dx == 1 else
                          (y-bottom,y-top) if dx == -1 else
                          (x-right,x-left) if dy == 1 else (left-x,right-x))
                if lo <= 0 <= hi and 1 <= len(base) <= 3:
                    candidates.append({"position": (y,x), "direction": (dy,dx),
                        "width": len(base), "span": (lo,hi), "color": color,
                        "phase": phase, "distance": address[base[0]][1]})
        assert len(candidates) == 1, "cuff collars must select one straight gate"
        slots.append(candidates[0])
    slots.sort(key=lambda s: s["distance"])
    assert len({s["distance"] for s in slots}) == len(slots), "gate distance ties excluded"
    templates = {}
    for slot in slots:
        lo, hi = slot["span"]
        assert 1 <= hi-lo+1 <= 6
        assert templates.setdefault(hi-lo+1, (lo,hi)) == (lo,hi), "unequal-offset length ties excluded"
    assert np.array_equal(grid, _sparse_input(grid.shape, center, radius, slots, background, carrier)), "unexplained input cells"
    # All source intervals must fit at every destination, without touching seed
    # or another cuff/collar. This also makes output overlays unambiguous.
    protected = {(cy+dy,cx+dx) for dy in range(-radius-1,radius+2)
                 for dx in range(-radius-1,radius+2)}
    envelopes = []
    lo = min(s["span"][0] for s in slots)
    hi = max(s["span"][1] for s in slots)
    for slot in slots:
        envelope = set(_slot_cells(slot,(lo,hi)) + _collar_cells(slot))
        assert all(1 <= y < h-1 and 1 <= x < w-1 for y,x in envelope)
        assert not envelope & protected
        assert all(not envelope & other for other in envelopes)
        envelopes.append(envelope)
    return {"shape": grid.shape, "center": center, "radius": radius, "slots": slots,
            "background": background, "carrier": carrier}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution under the family's fixed clockwise convention."""
    a = _parse(input_grid)
    return _paint_output(a["shape"], a["center"], a["radius"], a["slots"],
                         a["background"], a["carrier"])


def _gate_options(shape, center, radius, low, high, max_width):
    """Candidate straight gates with room for any source interval at that gate."""
    h, w = shape
    cy, cx = center
    result = []
    for phase, points in enumerate(_paths(radius, shape, center)):
        for start in range(2, len(points)-max_width-2):
            line = points[start-2:start+max_width+2]
            dy, dx = line[1][0]-line[0][0], line[1][1]-line[0][1]
            if any((b[0]-a[0],b[1]-a[1]) != (dy,dx) for a,b in zip(line,line[1:])):
                continue
            corners = [(y+k*dx,x-k*dy) for y,x in (line[0],line[-1]) for k in (low,high)]
            top,bottom = min(y for y,x in corners), max(y for y,x in corners)
            left,right = min(x for y,x in corners), max(x for y,x in corners)
            if top < 1 or left < 1 or bottom >= h-1 or right >= w-1:
                continue
            if not (bottom < cy-radius-1 or top > cy+radius+1 or
                    right < cx-radius-1 or left > cx+radius+1):
                continue
            result.append({"position": points[start], "direction": (dy,dx),
                           "phase": phase, "distance": start+1,
                           "squared_distance": sum((p-c)**2 for p,c in zip(points[start],center)),
                           "box": (top,bottom,left,right)})
    return result


def _select_gates(rng, options, widths, strong=False):
    # Bounding boxes reserve all possible transferred intervals and a separating
    # background cell. Bounded greedy attempts, then the caller grows the canvas.
    for _ in range(24):
        order = rng.permutation(len(options))
        chosen = []
        for index in order:
            candidate = options[int(index)]
            top,bottom,left,right = candidate["box"]
            if any(candidate["distance"] == s["distance"] for s in chosen):
                continue
            if any(not (bottom < s["box"][0]-1 or top > s["box"][1]+1 or
                        right < s["box"][2]-1 or left > s["box"][3]+1) for s in chosen):
                continue
            chosen.append(candidate)
            if len(chosen) == len(widths):
                chosen.sort(key=lambda s: s["distance"])
                if strong:
                    if len({s["direction"] for s in chosen}) < 2:
                        break
                    raster = sorted(range(len(chosen)), key=lambda i: chosen[i]["position"])
                    if raster == list(range(len(chosen))):
                        break
                    if strong == 2:
                        euclidean = sorted(range(len(chosen)),key=lambda i:chosen[i]["squared_distance"])
                        if euclidean == list(range(len(chosen))) or chosen[0]["direction"] == chosen[-1]["direction"]:
                            break
                return [dict(s, width=width) for s,width in zip(chosen,widths)]
    return None


def _span_plan(rng, witness, count):
    if witness == 1:
        return [(-2,2),(0,1),(-1,1)], [1,3,2]
    if witness == 2:
        # Both signed one-sided intervals visibly move; no ambiguous ties.
        return [(-3,0),(0,0),(0,2),(-1,0)], [2,1,2,1]
    cap = int(rng.integers(2,7))
    lengths = [int(x) for x in rng.integers(1,cap+1,size=count)]
    templates = {}
    for length in set(lengths):
        if rng.random() < .7:
            lo = -(length//2)
            if length % 2 == 0 and rng.random() < .5:
                lo += 1
        else:
            lo = -int(rng.integers(length))
        templates[length] = (lo,lo+length-1)
    style = int(rng.integers(4))
    if style == 0:
        lengths.sort()
    elif style == 1:
        lengths.sort(reverse=True)
    widths = ([int(rng.integers(1,4))]*count if rng.random()<.25 else
              [int(x) for x in rng.integers(1,4,size=count)])
    return [templates[n] for n in lengths], widths


def _force_unsorted(spans):
    spans = list(spans)
    lengths = [hi-lo+1 for lo,hi in spans]
    if len(set(lengths))==1:
        spans[-1] = (0,0) if lengths[0]>1 else (-1,1)
        lengths = [hi-lo+1 for lo,hi in spans]
    if lengths==sorted(lengths):
        spans = spans[1:]+spans[:1]
    return spans


def _sample_pair(rng, radius, witness, regime, is_test, common_palette, require_change=False):
    count = 3 if witness == 1 else 4 if witness == 2 else int(rng.integers(2,8))
    spans,widths = _span_plan(rng,witness,count)
    low,high = min(p[0] for p in spans),max(p[1] for p in spans)
    max_width = max(widths)
    # Different length/color plans remain intact when geometry needs more room.
    # Random rectangles and placements precede a finite maximum-canvas fallback.
    if regime == "compact" and not is_test:
        side_low,side_high = 25,30
    elif witness == 1:
        side_low,side_high = (20,26) if regime == "advance" else (20,30)
    elif witness == 2:
        side_low,side_high = (24,27) if regime == "advance" else (24,30)
    elif is_test and regime == "compact":
        side_low,side_high = 15,24
        count = int(rng.integers(2,4))
        lengths = [int(x) for x in rng.integers(1,4,size=count)]
        two = (-1,0) if rng.random()<.5 else (0,1)
        spans = [{1:(0,0),2:two,3:(-1,1)}[n] for n in lengths]
        widths = [int(x) for x in rng.integers(1,3,size=count)]
        low,high = min(p[0] for p in spans),max(p[1] for p in spans)
        max_width = max(widths)
    elif is_test and regime == "advance":
        side_low,side_high = 28,30
    else:
        side_low,side_high = 16,30
    if require_change:
        spans = _force_unsorted(spans)
    low,high = min(p[0] for p in spans),max(p[1] for p in spans)
    selected = None
    for attempt in range(40):
        if attempt < 16:
            shape = tuple(int(x) for x in rng.integers(side_low,side_high+1,size=2))
            if rng.random() < .25:
                shape = (shape[0],shape[0])
        else:
            shape = (24,24) if is_test and regime == "compact" else (30,30)
        margin = radius+1
        if attempt == 39:
            center = ((shape[0]-1)//2,(shape[1]-1)//2)
        elif rng.random() < .3:
            center = ((shape[0]-1)//2,(shape[1]-1)//2)
        else:
            center = tuple(int(rng.integers(margin,d-margin)) for d in shape)
        options = _gate_options(shape,center,radius,low,high,max_width)
        selected = _select_gates(rng,options,widths,strong=witness)
        if selected is not None:
            break
    if selected is None:
        # Dense, broad offset combinations are not required for every count.
        # A centered interval family preserves count and width diversity.
        spans = [(-1,1) if i%2 else (0,0) for i in range(count)]
        if require_change:
            spans = _force_unsorted(spans)
        if witness:
            raise AssertionError("teaching geometry exhausted")
        if is_test and regime == "compact":
            raise AssertionError("compact geometry exhausted")
        for cy in range(radius+1,29-radius):
            for cx in range(radius+1,29-radius):
                shape,center = (30,30),(cy,cx)
                options = _gate_options(shape,center,radius,-1,1,max_width)
                selected = _select_gates(rng,options,widths)
                if selected is not None:
                    break
            if selected is not None:
                break
        assert selected is not None, "finite gate fallback exhausted"
    palette = list(common_palette) if common_palette is not None else [int(x) for x in rng.permutation(10)]
    if common_palette is not None:
        palette[2:] = [int(x) for x in rng.permutation(palette[2:])]
    background,carrier = palette[:2]
    slots = [dict(s,span=span,color=color) for s,span,color in zip(selected,spans,palette[2:])]
    inp = _sparse_input(shape,center,radius,slots,background,carrier)
    out = _paint_output(shape,center,radius,slots,background,carrier)
    return {"input":inp,"output":out}


def _count(value, name, low, high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low<=int(value)<=high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,nq = _count(num_train,"num_train",2,10),_count(num_test,"num_test",1,4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["advance","relocate","compact"]))
    common_palette = rng.permutation(10) if rng.random()<.25 else None
    radii = [2,3] + [int(rng.integers(2,5 if regime=="advance" else 6)) for _ in range(nt-2)]
    max_radius = max(radii)
    test_radii = ([max_radius+1]*nq if regime == "advance" else [2]*nq if regime=="compact" else
                  [int(rng.choice(sorted(set(radii)))) for _ in range(nq)])
    puzzle = {"train":[],"test":[]}
    seen = set()
    for split,plan in (("train",radii),("test",test_radii)):
        for i,radius in enumerate(plan):
            witness = i+1 if split=="train" and i<2 else 0
            pair = _sample_pair(rng,radius,witness,regime,split=="test",common_palette,
                                require_change=split=="test" and i==0)
            # Ninety ordered role assignments exceed the thirteen possible
            # exclusions, so duplicate avoidance cannot exhaust its fallback.
            if _key(pair["input"]) in seen:
                a = _parse(pair["input"])
                old = [a["background"],a["carrier"]]+[s["color"] for s in a["slots"]]
                found = False
                for bg in range(10):
                    for carrier in range(10):
                        if bg==carrier:
                            continue
                        new = [bg,carrier]+[c for c in range(10) if c not in (bg,carrier)]
                        mapping = np.arange(10,dtype=np.int8)
                        for x,y in zip(old,new):
                            mapping[x]=y
                        candidate = {k:mapping[v] for k,v in pair.items()}
                        if _key(candidate["input"]) not in seen:
                            pair,found = candidate,True
                            break
                    if found:
                        break
                assert found, "finite distinct-input fallback exhausted"
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    """Relational cuff checks and analytic carrier check, without solve()."""
    inp,out = _grid(pair["input"]),_grid(pair["output"])
    assert inp.shape == out.shape
    a = _parse(inp)
    assert set(map(int,np.unique(inp))) == set(map(int,np.unique(out)))
    spans = []
    colored = np.zeros(out.shape,dtype=bool)
    for slot in a["slots"]:
        cells = np.argwhere(out==slot["color"])
        assert len(cells)>0
        y,x = slot["position"]
        dy,dx = slot["direction"]
        along = (cells[:,0]-y)*dy+(cells[:,1]-x)*dx
        normal = (cells[:,0]-y)*dx-(cells[:,1]-x)*dy
        assert int(along.min()) == 0 and int(along.max()) == slot["width"]-1
        lo,hi = int(normal.min()),int(normal.max())
        assert lo<=0<=hi
        assert len(cells)==slot["width"]*(hi-lo+1), "nonsolid or moved output cuff"
        spans.append((lo,hi))
        colored[out==slot["color"]] = True
        assert all(out[py,px]==a["carrier"] for py,px in _collar_cells(slot))
    lengths = [hi-lo+1 for lo,hi in spans]
    assert all(x<=y for x,y in zip(lengths,lengths[1:])), "wrong intrinsic rank order"
    assert Counter(spans)==Counter(s["span"] for s in a["slots"]), "span templates not conserved"
    carrier = _analytic_carrier(a["radius"],out.shape,a["center"])
    expected = np.where(carrier,a["carrier"],a["background"])
    assert np.array_equal(out[~colored],expected[~colored]), "wrong winding or uncleared source pixels"
    assert out[a["center"]]==a["background"]
    assert np.any((inp==a["background"]) & (out==a["carrier"]))
    return a,spans


def _features(pair):
    a,assigned = _check_pair(pair)
    slots = a["slots"]
    spans = [s["span"] for s in slots]
    lengths = [p[1]-p[0]+1 for p in spans]
    area_order = sorted(range(len(slots)),key=lambda i:lengths[i]*slots[i]["width"])
    raster_order = sorted(range(len(slots)),key=lambda i:slots[i]["position"])
    euclidean_order = sorted(range(len(slots)),key=lambda i:
        sum((p-c)**2 for p,c in zip(slots[i]["position"],a["center"])))
    def assigned_by(order):
        result = [None]*len(slots)
        for i,p in zip(order,sorted(spans,key=lambda p:p[1]-p[0])):
            result[i]=p
        return result
    return {"radius":a["radius"],"shape":a["shape"],"center":a["center"],
            "count":len(slots),"widths":tuple(s["width"] for s in slots),
            "lengths":tuple(lengths),"directions":tuple(s["direction"] for s in slots),
            "phases":tuple(s["phase"] for s in slots),
            "distances":tuple(s["distance"] for s in slots),
            "background":a["background"],"carrier":a["carrier"],
            "changed":spans!=assigned,"area_disagrees":[spans[i] for i in area_order]!=assigned,
            "reversal_disagrees":spans[::-1]!=assigned,
            "raster_disagrees":assigned_by(raster_order)!=assigned,
            "euclidean_disagrees":assigned_by(euclidean_order)!=assigned,
            "ties":len(set(lengths))<len(lengths),
            "signed_transfer":any(p!=q and abs(sum(p))>1 for p,q in zip(spans,assigned)),
            "rotated_signed_transfer":any(abs(sum(p))>1 and p!=spans[i]
                and any(p==spans[j] and slots[j]["direction"]!=slots[i]["direction"]
                        for j in range(len(slots))) for i,p in enumerate(assigned)),
            "layout":(a["shape"],a["center"],a["radius"],tuple(
                (s["phase"],s["distance"],s["width"],s["span"]) for s in slots))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle)=={"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    assert 2<=len(puzzle["train"])<=10 and 1<=len(puzzle["test"])<=4
    assert len(puzzle["train"])>len(puzzle["test"])
    seen = set()
    features = {"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={"input","output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen
            seen.add(key)
            features[split].append(_features(pair))
    train = features["train"]
    assert sum(f["changed"] for f in train)>=2
    assert any(f["area_disagrees"] for f in train)
    assert any(f["changed"] and f["reversal_disagrees"] for f in train)
    assert any(f["raster_disagrees"] for f in train)
    assert any(f["euclidean_disagrees"] for f in train)
    assert any(f["rotated_signed_transfer"] for f in train)
    assert {2,3} <= {f["radius"] for f in train}
    assert len({f["count"] for f in train})>=2
    for f in features["test"]:
        assert f["radius"] <= min(5,max(t["radius"] for t in train)+1)
    assert any(f["changed"] for f in features["test"])


def self_test():
    """Bounded 42-seed variation sweep, geometry, counts and mutation checks."""
    for radius in range(2,6):
        for shape,center in (((30,30),(radius+1,radius+1)),
                             ((19,30),(10,20)),((30,17),(21,8))):
            mask = _carrier_mask(radius,shape,center)
            assert np.array_equal(mask,_analytic_carrier(radius,shape,center))
            rot_center = (shape[1]-1-center[1],center[0])
            assert np.array_equal(np.rot90(mask),_carrier_mask(radius,shape[::-1],rot_center))
            big = _analytic_carrier(radius,(80,80),(40,40))
            y,x = 40-center[0],40-center[1]
            assert np.array_equal(mask,big[y:y+shape[0],x:x+shape[1]])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    all_features = [_features(pair) for puzzle in puzzles for split in ("train","test") for pair in puzzle[split]]
    signatures = {tuple(_key(p[k]) for s in ("train","test") for p in puzzle[s]
                        for k in ("input","output")) for puzzle in puzzles}
    assert len(signatures)==42
    assert len({f["shape"] for f in all_features})>=30
    assert len({f["layout"] for f in all_features})>=180
    assert {f["count"] for f in all_features}==set(range(2,8))
    assert {f["radius"] for f in all_features}=={2,3,4,5}
    assert {w for f in all_features for w in f["widths"]}=={1,2,3}
    assert {f["background"] for f in all_features}==set(range(10))
    assert {f["carrier"] for f in all_features}==set(range(10))
    assert {p for f in all_features for p in f["phases"]}=={0,1,2,3}
    assert {d for f in all_features for d in f["directions"]}==set(_DIRECTIONS)
    assert any(f["ties"] for f in all_features)
    assert any(f["euclidean_disagrees"] for f in all_features)
    for puzzle in puzzles:
        for pair in puzzle["train"]+puzzle["test"]:
            assert np.array_equal(solve(pair["input"].copy()),pair["output"])
    again = generate_puzzle(0)
    assert all(_key(a[k])==_key(b[k]) for s in ("train","test")
               for a,b in zip(puzzles[0][s],again[s]) for k in ("input","output"))
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(100+10*nt+nq,nt,nq)
            assert len(p["train"])==nt and len(p["test"])==nq
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Omission of A (leave cuff intervals allocated as in input) must fail.
    # Omission of B (leave the sparse carrier uncompleted) must fail.
    base = puzzles[0]
    base_pair = next(p for p in base["train"] if _features(p)["changed"])
    a = _parse(base_pair["input"])
    no_a = np.where(_carrier_mask(a["radius"],a["shape"],a["center"]),a["carrier"],a["background"]).astype(np.int8)
    for slot in a["slots"]:
        for y,x in _slot_cells(slot,slot["span"]):
            no_a[y,x]=slot["color"]
    no_b = base_pair["output"].copy()
    no_b[(no_b==a["carrier"]) & (base_pair["input"]==a["background"])]=a["background"]
    for wrong in (no_a,no_b):
        assert not np.array_equal(wrong,base_pair["output"])
        try:
            _check_pair({"input":base_pair["input"],"output":wrong})
        except AssertionError:
            pass
        else:
            raise AssertionError("parent-omission mutation accepted")
    # Local normals must rotate with the entire scene, not stay attached to
    # global rows/columns. These witnesses include moved one-sided intervals.
    for puzzle in puzzles[:3]:
        pair = next(p for p in puzzle["train"] if _features(p)["rotated_signed_transfer"])
        for turns in (1,2,3):
            assert np.array_equal(solve(np.rot90(pair["input"],turns)),np.rot90(pair["output"],turns))
    # Mutations are painted without calling solve or the output allocator.
    for mode in ("area","reversal","raster","recenter"):
        field = {"area":"area_disagrees","reversal":"reversal_disagrees",
                 "raster":"raster_disagrees","recenter":"rotated_signed_transfer"}[mode]
        pair = next(p for p in base["train"] if _features(p)[field] and _features(p)["changed"])
        a = _parse(pair["input"])
        slots = a["slots"]
        source = [s["span"] for s in slots]
        ranked = sorted(source,key=lambda p:p[1]-p[0])
        if mode=="area":
            intervals = [s["span"] for s in sorted(slots,key=lambda s:s["width"]*(s["span"][1]-s["span"][0]+1))]
        elif mode=="reversal":
            intervals = source[::-1]
        elif mode=="raster":
            intervals = [None]*len(slots)
            for i,p in zip(sorted(range(len(slots)),key=lambda i:slots[i]["position"]),ranked):
                intervals[i]=p
        else:
            intervals = [(-(n//2),-(n//2)+n-1) for n in (hi-lo+1 for lo,hi in ranked)]
        wrong = np.where(_carrier_mask(a["radius"],a["shape"],a["center"]),a["carrier"],a["background"]).astype(np.int8)
        for slot,span in zip(slots,intervals):
            for y,x in _slot_cells(slot,span):
                wrong[y,x]=slot["color"]
        assert not np.array_equal(wrong,pair["output"])
        try:
            _check_pair({"input":pair["input"],"output":wrong})
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{mode} shortcut mutation accepted")
    f=all_features
    return {"seeds":42,"pairs":len(f),"shapes":len({v["shape"] for v in f}),
            "layouts":len({v["layout"] for v in f}),
            "radii":sorted({v["radius"] for v in f}),"counts":sorted({v["count"] for v in f}),
            "area_counterexamples":sum(v["area_disagrees"] for v in f),
            "raster_counterexamples":sum(v["raster_disagrees"] for v in f),
            "euclidean_counterexamples":sum(v["euclidean_disagrees"] for v in f)}
