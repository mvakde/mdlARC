"""Decode exact-span lattice rectangles, then complete anchored mirror copies.

All inference uses only the supplied grid. No parent code or files are loaded.
Generation and validation keep only one puzzle at a time and write no artifacts.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


HOST_SEEDS = (0,335753363,552904126,150669495,511743310,922642701,337785851,
              25326692,442196207,623433763,972881931,526941348,943795302,
              593787830,445603913,91701795,908349449,988718418,5748555,
              960448950,143750993,944332833,128016116,909755852,174306779,
              927705633,851513116,53096724,544718245,593225137,215207937,
              402475931,11707981,819918645,922685624,949823768,339916178,
              161499606,384213505,847276587,352750044,594082791)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= x <= 30 for x in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _bounds(h, w, row, col):
    return ((0,row,0,col),(0,row,col+1,w),
            (row+1,h,0,col),(row+1,h,col+1,w))


def _square(panel, color):
    points = np.argwhere(panel == color)
    assert len(points), "missing anchor"
    top, left = map(int, points.min(0))
    bottom, right = map(int, points.max(0) + 1)
    side = bottom - top
    assert side == right-left and len(points) == side*side, "not one solid square"
    return top, left, side


def _runs(row, bg):
    result, c = [], 0
    while c < len(row):
        color = int(row[c])
        if color == bg:
            c += 1
            continue
        end = c+1
        while end < len(row) and int(row[end]) == color:
            end += 1
        result.append((color, c, end))
        c = end
    return result


def _rectangles(coarse, bg):
    """Half-open pixel rectangles; horizontal first, exact vertical spans second."""
    rows = [_runs(row, bg) for row in coarse]
    rectangles = []
    for r, runs in enumerate(rows):
        previous = set(rows[r-1]) if r else set()
        for color, left, right in runs:
            if (color,left,right) in previous:
                continue
            end = r+1
            while end < len(rows) and (color,left,right) in rows[end]:
                end += 1
            rectangles.append((color,1+2*r,2*end,1+2*left,2*right))
    return rectangles


def _features(coarse, bg):
    rows = [_runs(row,bg) for row in coarse]
    horizontal = any(b-a >= 2 for row in rows for _,a,b in row)
    exact = any(set(a) & set(b) for a,b in zip(rows,rows[1:]))
    partial = any(ca == cb and (la,ra) != (lb,rb) and max(la,lb) < min(ra,rb)
                  for upper,lower in zip(rows,rows[1:])
                  for ca,la,ra in upper for cb,lb,rb in lower)
    return bool(horizontal),bool(exact),bool(partial)


def _decode(value):
    grid = _grid(value)
    h,w = grid.shape
    rows = [r for r in range(1,h-1) if np.all(grid[r] == grid[r,0])]
    cols = [c for c in range(1,w-1) if np.all(grid[:,c] == grid[0,c])]
    crosses = [(r,c,int(grid[r,c])) for r in rows for c in cols
               if grid[r,0] == grid[0,c]]
    assert len(crosses) == 1, "expected a unique full cross"
    row,col,divider = crosses[0]
    bounds = _bounds(h,w,row,col)
    panels = [grid[t:b,l:r] for t,b,l,r in bounds]
    colors = [set(map(int,np.unique(p))) for p in panels]
    common = set.intersection(*colors)
    assert len(common) == 2, "only background and anchor occur in every panel"
    candidates = []
    for color in sorted(common):
        try:
            squares = [_square(p,color) for p in panels]
        except AssertionError:
            continue
        candidates.append((color,squares))
    assert len(candidates) == 1, "anchor must be unambiguous by square geometry"
    anchor,anchors = candidates[0]
    bg = next(iter(common-{anchor}))
    assert len({bg,anchor,divider}) == 3
    sources = [i for i,s in enumerate(colors) if s-{bg,anchor}]
    assert len(sources) == 1, "expected exactly one coded reference panel"
    ref = sources[0]
    for p in range(4):
        if p != ref:
            assert colors[p] == {bg,anchor}, "unexpected destination detail"
    source = panels[ref]
    assert min(source.shape) >= 3 and all(n % 2 == 1 for n in source.shape)
    raw = source.copy()
    raw[raw == anchor] = bg
    assert np.all(raw[::2,:] == bg) and np.all(raw[:,::2] == bg), "broken lattice"
    coarse = raw[1::2,1::2].copy()
    assert np.any(coarse != bg)
    a = anchors[ref][2]
    assert 1 <= a <= 3
    assert all(s % a == 0 and 1 <= s//a <= 3 for _,_,s in anchors)
    rectangles = _rectangles(coarse,bg)
    ar,ac,_ = anchors[ref]
    for _,t,b,l,r in rectangles:
        assert not (max(t,ar) < min(b,ar+a) and max(l,ac) < min(r,ac+a)), "payload crosses anchor"
    return {"grid":grid,"bounds":bounds,"panels":panels,"background":bg,
            "divider":divider,"anchor":anchor,"reference":ref,"anchors":anchors,
            "coarse":coarse,"rectangles":rectangles}


def _mapped(rect, source_anchor, destination_anchor, vertical, horizontal):
    color,t,b,l,r = rect
    sy,sx,a = source_anchor
    dy,dx,d = destination_anchor
    scale = d//a
    t,b,l,r = t-sy,b-sy,l-sx,r-sx
    if vertical:
        t,b = a-b,a-t
    if horizontal:
        l,r = a-r,a-l
    return color,dy+scale*t,dy+scale*b,dx+scale*l,dx+scale*r


def _paint(panel, rectangle):
    color,t,b,l,r = rectangle
    t,b = max(0,t),min(panel.shape[0],b)
    l,r = max(0,l),min(panel.shape[1],r)
    if t < b and l < r:
        panel[t:b,l:r] = color


def _completion(data):
    out = data["grid"].copy()
    ref = data["reference"]
    for p,(t,b,l,r) in enumerate(data["bounds"]):
        for rect in data["rectangles"]:
            mapped = _mapped(rect,data["anchors"][ref],data["anchors"][p],
                             ref//2 != p//2,ref%2 != p%2)
            _paint(out[t:b,l:r],mapped)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete the reference lattice and its scaled, panel-relative reflections."""
    return _completion(_decode(input_grid))


def _bridge_pixels(data, vertical="exact", horizontal=True):
    """Validation/ablation model: paint sites and bridges without stacking runs."""
    coarse,bg = data["coarse"],data["background"]
    pixels = np.full(data["panels"][data["reference"]].shape,bg,dtype=np.int8)
    pixels[1::2,1::2] = coarse
    if horizontal:
        for y,row in enumerate(coarse):
            for x in range(len(row)-1):
                if row[x] != bg and row[x] == row[x+1]:
                    pixels[1+2*y,2+2*x] = row[x]
    if vertical in ("exact","loose"):
        for y in range(len(coarse)-1):
            for color,l,r in set(_runs(coarse[y],bg)) & set(_runs(coarse[y+1],bg)):
                pixels[2+2*y,1+2*l:2*r] = color
    if vertical == "loose":
        for y in range(len(coarse)-1):
            for x in range(coarse.shape[1]):
                if coarse[y,x] != bg and coarse[y,x] == coarse[y+1,x]:
                    pixels[2+2*y,1+2*x] = coarse[y,x]
    elif vertical not in ("none","exact"):
        raise ValueError("unknown vertical policy")
    return pixels


def _pixel_prediction(data, pixels, reflect=True, scale=True, destinations=True):
    """Check transport by expanded individual pixels, not merged rectangles."""
    out = data["grid"].copy()
    ref,bg = data["reference"],data["background"]
    sy,sx,a = data["anchors"][ref]
    points = [(int(y),int(x),int(pixels[y,x])) for y,x in np.argwhere(pixels != bg)]
    for p,(t,b,l,r) in enumerate(data["bounds"]):
        if not destinations and p != ref:
            continue
        dy,dx,d = data["anchors"][p]
        k = d//a if scale else 1
        v = reflect and ref//2 != p//2
        h = reflect and ref%2 != p%2
        panel = out[t:b,l:r]
        for y,x,color in points:
            yy = a-1-(y-sy) if v else y-sy
            xx = a-1-(x-sx) if h else x-sx
            _paint(panel,(color,dy+k*yy,dy+k*(yy+1),dx+k*xx,dx+k*(xx+1)))
    return out


def _destination_difference(data, a, b):
    return any(not np.array_equal(a[t:e,l:r],b[t:e,l:r])
               for p,(t,e,l,r) in enumerate(data["bounds"]) if p != data["reference"])


def _copy_records(data):
    ref = data["reference"]
    records = []
    for p,panel in enumerate(data["panels"]):
        if p == ref:
            continue
        total,visible,sides = 0,0,set()
        for rect in data["rectangles"]:
            _,t,b,l,r = _mapped(rect,data["anchors"][ref],data["anchors"][p],
                               ref//2 != p//2,ref%2 != p%2)
            total += (b-t)*(r-l)
            visible += max(0,min(b,panel.shape[0])-max(0,t))*max(0,min(r,panel.shape[1])-max(0,l))
            sides.update(name for name,flag in (("top",t<0),("bottom",b>panel.shape[0]),
                                                 ("left",l<0),("right",r>panel.shape[1])) if flag)
        records.append({"panel":p,"scale":data["anchors"][p][2]//data["anchors"][ref][2],
                        "total":total,"visible":visible,"sides":sorted(sides)})
    return records


def _set_run(coarse,row,left,right,color,bg):
    coarse[row,left:right] = color
    if left and coarse[row,left-1] == color:
        coarse[row,left-1] = bg
    if right < coarse.shape[1] and coarse[row,right] == color:
        coarse[row,right] = bg


def _sample_coarse(rng,shape,bg,details,role):
    coarse = np.full(shape,bg,dtype=np.int8)
    mask = rng.random(shape) < float(rng.uniform(.20,.85))
    colors = rng.choice(details,size=shape)
    coarse[mask] = colors[mask]
    rows,cols = shape
    kinds = [role] if role in ("exact","partial") else []
    kinds += [str(rng.choice(("horizontal","exact","partial","return")))
              for _ in range(int(rng.integers(3)))] if role == "random" else []
    for kind in kinds:
        color = int(rng.choice(details))
        if kind == "horizontal" and cols >= 2:
            l = int(rng.integers(cols-1)); r = int(rng.integers(l+2,cols+1))
            _set_run(coarse,int(rng.integers(rows)),l,r,color,bg)
        elif kind == "exact" and min(shape) >= 2:
            l = int(rng.integers(cols-1)); r = int(rng.integers(l+2,cols+1))
            depth = int(rng.integers(2,rows+1)); top = int(rng.integers(rows-depth+1))
            for y in range(top,top+depth):
                _set_run(coarse,y,l,r,color,bg)
        elif kind == "partial" and min(shape) >= 2:
            spans = [(l,r) for l in range(cols) for r in range(l+1,cols+1)]
            pairs = [(a,b) for a in spans for b in spans
                     if a != b and max(a[0],b[0]) < min(a[1],b[1])]
            a,b = pairs[int(rng.integers(len(pairs)))]
            top = int(rng.integers(rows-1))
            _set_run(coarse,top,*a,color,bg)
            _set_run(coarse,top+1,*b,color,bg)
        elif kind == "return" and rows >= 3:
            top = int(rng.integers(rows-2)); end = int(rng.integers(top+2,rows))
            l = int(rng.integers(cols)); r = int(rng.integers(l+1,cols+1))
            for y in (top,end):
                _set_run(coarse,y,l,r,color,bg)
            coarse[top+1,l:r] = bg
    if not np.any(coarse != bg):
        coarse[int(rng.integers(rows)),int(rng.integers(cols))] = int(rng.choice(details))
    return coarse


def _axis(rng,minimum,source_index,source_length):
    other = 1-source_index
    if minimum[source_index] > source_length or source_length+minimum[other]+1 > 30:
        return None
    result = [0,0]
    result[source_index] = source_length
    result[other] = int(rng.integers(minimum[other],31-source_length))
    return result


def _build_pair(rng,ref,role,shape,witness,shift=False):
    palette = list(map(int,rng.permutation(10)))
    bg,divider,anchor = palette[:3]
    details = palette[3:3+int(rng.integers(1,6))]
    if rng.random() < .35:
        details[0] = divider
    coarse = _sample_coarse(rng,shape,bg,details,role)
    if role == "exact" and not all(_features(coarse,bg)[:2]):
        return None
    if role == "partial" and not _features(coarse,bg)[2]:
        return None
    source_shape = (2*shape[0]+1,2*shape[1]+1)
    rectangles = _rectangles(coarse,bg)
    rendered = np.full(source_shape,bg,dtype=np.int8)
    for rect in rectangles:
        _paint(rendered,rect)
    side = int(rng.integers(1,min(3,min(source_shape)-2)+1))
    places = [(y,x) for y in range(source_shape[0]-side+1)
              for x in range(source_shape[1]-side+1)
              if np.all(rendered[y:y+side,x:x+side] == bg)]
    if not places:
        return None
    ay,ax = places[int(rng.integers(len(places)))]
    source_anchor = (ay,ax,side)
    # One equal and one enlarged copy are compulsory in both teaching witnesses.
    scales = [int(rng.choice((1,1,2,3))) for _ in range(4)]
    scales[ref] = 1
    diagonal = 3-ref
    if witness:
        scales = [1]*4
        scales[diagonal] = 2
    elif shift:
        scales[diagonal] = 3
    transformed = {}
    for p in range(4):
        d = side*scales[p]
        rs = [_mapped(rect,source_anchor,(0,0,d),ref//2 != p//2,ref%2 != p%2)
              for rect in rectangles]
        top = min([0]+[r[1] for r in rs]); bottom = max([d]+[r[2] for r in rs])
        left = min([0]+[r[3] for r in rs]); right = max([d]+[r[4] for r in rs])
        transformed[p] = rs,(top,bottom,left,right)
    # Each witness keeps at least two complete copies; the second teaches clipping.
    clips = [False if witness or p == ref else bool(rng.random() < .65)
             for p in range(4)]
    if witness and role == "partial":
        # Teach clipping even at the minimum two demonstrations. The enlarged
        # diagonal copy and one unchanged-size copy still remain complete.
        adjacent = [p for p in range(4) if p not in (ref,diagonal)]
        clips[int(rng.choice(adjacent))] = True
    if shift:
        clips[diagonal] = True
    min_rows,min_cols = [3,3],[3,3]
    for p in range(4):
        d = side*scales[p]
        _,(t,b,l,r) = transformed[p]
        ph,pw = (d+1,d+1) if clips[p] else (b-t,r-l)
        if p == ref:
            ph,pw = source_shape
        min_rows[p//2] = max(min_rows[p//2],ph)
        min_cols[p%2] = max(min_cols[p%2],pw)
    heights = _axis(rng,min_rows,ref//2,source_shape[0])
    widths = _axis(rng,min_cols,ref%2,source_shape[1])
    if heights is None or widths is None:
        return None
    bounds = _bounds(sum(heights)+1,sum(widths)+1,heights[0],widths[0])
    inp = np.full((sum(heights)+1,sum(widths)+1),bg,dtype=np.int8)
    inp[heights[0],:] = divider; inp[:,widths[0]] = divider
    for p,(t,b,l,r) in enumerate(bounds):
        panel = inp[t:b,l:r]
        d = side*scales[p]
        if p == ref:
            panel[1::2,1::2] = coarse
            y,x = ay,ax
        else:
            rs,(mt,mb,ml,mr) = transformed[p]
            places = []
            for y in range(panel.shape[0]-d+1):
                for x in range(panel.shape[1]-d+1):
                    total = sum((b0-t0)*(r0-l0) for _,t0,b0,l0,r0 in rs)
                    visible = sum(max(0,min(panel.shape[0],y+b0)-max(0,y+t0))*
                                  max(0,min(panel.shape[1],x+r0)-max(0,x+l0))
                                  for _,t0,b0,l0,r0 in rs)
                    if bool(visible < total) == clips[p] and visible > 0:
                        places.append((y,x))
            if not places:
                return None
            y,x = places[int(rng.integers(len(places)))]
        panel[y:y+d,x:x+d] = anchor
    try:
        data = _decode(inp)
    except AssertionError:
        return None
    out = _completion(data)
    if not all(rec["visible"] for rec in _copy_records(data)):
        return None
    if witness:
        pixels = _bridge_pixels(data)
        if not _destination_difference(data,out,_pixel_prediction(data,pixels,reflect=False)):
            return None
        if not _destination_difference(data,out,_pixel_prediction(data,pixels,scale=False)):
            return None
    return {"input":inp,"output":out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed,"seed",0)
    nt = _integer(num_train,"num_train",2,10)
    ns = _integer(num_test,"num_test",1,4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    refs = list(map(int,rng.permutation(4)))
    # The held-out axis count is never used in training, even with ten examples.
    axis = int(rng.integers(2))
    held = int(rng.choice((5,6)))
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",nt),("test",ns)):
        for index in range(count):
            witness = split == "train" and index < 2
            role = ("exact" if index == 0 else "partial") if witness else "random"
            ref = refs[index] if witness else int(rng.integers(4))
            for attempt in range(256):
                shape = [int(rng.integers(2 if witness else 1,5)) for _ in range(2)]
                if split == "test":
                    shape[axis] = held
                pair = _build_pair(rng,ref,role,tuple(shape),witness,split == "test")
                if pair is None or _key(pair["input"]) in seen:
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError(f"bounded construction exhausted for {split}/{index}")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    """Check actual pixels, teaching evidence, shift, and causal ablations."""
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    nt,ns = len(puzzle["train"]),len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen,analyses = set(),{"train":[],"test":[]}
    evidence = {"horizontal":False,"exact":False,"partial":False,"transport":False}
    witness_refs = set()
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            data = _decode(inp)
            pixels = _bridge_pixels(data)
            assert np.array_equal(out,_pixel_prediction(data,pixels)), "wrong rectangle completion or transport"
            assert np.array_equal(out,solve(inp)), "solve mismatch"
            ref = data["reference"]
            # Existing evidence, anchors and divider must survive exactly.
            assert np.all(out[inp != data["background"]] == inp[inp != data["background"]])
            assert set(map(int,np.unique(out))) == set(map(int,np.unique(inp)))
            copies = _copy_records(data)
            assert all(rec["visible"] > 0 for rec in copies), "empty destination completion"
            data["copies"] = copies
            analyses[split].append(data)
            if split == "train":
                for name,variant in (("horizontal",_bridge_pixels(data,"exact",False)),
                                     ("exact",_bridge_pixels(data,"none")),
                                     ("partial",_bridge_pixels(data,"loose"))):
                    alternative = _pixel_prediction(data,variant)
                    evidence[name] |= _destination_difference(data,out,alternative)
                evidence["transport"] |= _destination_difference(data,out,_pixel_prediction(data,pixels,destinations=False))
                complete_count = sum(rec["visible"] == rec["total"] for rec in copies)
                reflected = _destination_difference(data,out,_pixel_prediction(data,pixels,reflect=False))
                scaled = _destination_difference(data,out,_pixel_prediction(data,pixels,scale=False))
                if complete_count >= 2 and reflected and scaled:
                    witness_refs.add(ref)
    assert all(evidence.values()), "demonstrations must expose both parents"
    assert len(witness_refs) >= 2, "show scale/reflection from two reference quadrants"
    assert any(any(r["visible"] < r["total"] for r in a["copies"])
               for a in analyses["train"]), "teach clipping in training"
    train_shapes = [a["coarse"].shape for a in analyses["train"]]
    for data in analyses["test"]:
        shape = data["coarse"].shape
        assert any(shape[d] not in {s[d] for s in train_shapes} for d in (0,1)), "missing held-out axis"
        assert any(r["visible"] < r["total"] for r in data["copies"]), "test must combine new extent and clipping"


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for name in ("input","output"):
                a = np.asarray(pair[name],dtype=np.uint8)
                digest.update(bytes(a.shape)); digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeded puzzles, regression geometry and all legal counts; no disk writes."""
    shapes,coarse_shapes,roles,refs,sides,scales,clipped,layouts = (set() for _ in range(8))
    digests = set()
    empty_rows = deep = repeated = zero_payload = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = _puzzle_digest(puzzle)
        assert digest == _puzzle_digest(generate_puzzle(seed))
        digests.add(digest)
        for split in ("train","test"):
            for pair in puzzle[split]:
                data = _decode(pair["input"])
                coarse,bg = data["coarse"],data["background"]
                mapping = {bg:0}
                labels = tuple(mapping.setdefault(int(v),len(mapping)) for v in coarse.flat)
                layouts.add((coarse.shape,labels))
                shapes.add(pair["input"].shape); coarse_shapes.add(coarse.shape)
                roles.add((bg,data["divider"],data["anchor"]))
                refs.add(data["reference"]); sides.add(data["anchors"][data["reference"]][2])
                records = _copy_records(data)
                scales.update(r["scale"] for r in records)
                clipped.add(sum(r["visible"] < r["total"] for r in records))
                empty_rows += any(np.all(row == bg) for row in coarse)
                deep += any(b-t >= 5 for _,t,b,_,_ in data["rectangles"])
                colors = [r[0] for r in data["rectangles"]]
                repeated += len(colors) != len(set(colors))
                zero_payload += bg != 0 and np.any(coarse == 0)
    assert len(digests) == 42 and len(shapes) >= 35 and len(layouts) >= 100
    assert len(coarse_shapes) >= 20 and len(roles) >= 80
    assert refs == {0,1,2,3} and sides == {1,2,3} and scales == {1,2,3}
    assert {0,1,2,3} <= clipped and empty_rows > 0 and deep > 0 and repeated > 0 and zero_payload > 0
    # Unequal overlapping runs cannot bridge, even if two equal cells touch vertically.
    coarse = np.array([[4,4,0],[4,4,0],[0,4,4]],dtype=np.int8)
    assert _rectangles(coarse,0) == [(4,1,4,1,4),(4,5,6,3,6)]
    assert _features(coarse,0) == (True,True,True)
    # Reflection is after horizontal decoding: transposition is not an invariant.
    assert len(_rectangles(coarse.T,0)) != len(_rectangles(coarse,0))
    for nt in range(2,11):
        for ns in range(1,min(4,nt-1)+1):
            p = generate_puzzle(12345,nt,ns)
            assert (len(p["train"]),len(p["test"])) == (nt,ns)
    for nt,ns in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0,nt,ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(31415,2,1)
    broken["test"][0]["output"][0,0] = (int(broken["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corruption accepted")
    return {"seeds":42,"pairs":210,"distinct_puzzles":len(digests),
            "canvas_shapes":len(shapes),"coarse_shapes":len(coarse_shapes),
            "normalized_coarse_layouts":len(layouts),"color_role_assignments":len(roles),
            "reference_sides":sorted(sides),"scales":sorted(scales),
            "clipped_destination_counts":sorted(clipped),"empty_row_pairs":int(empty_rows),
            "deep_stack_pairs":int(deep),"repeated_color_rectangle_pairs":int(repeated),
            "zero_payload_pairs":int(zero_payload)}


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",",":")))
