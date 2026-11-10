"""Cavity-anchored replication around a color-coordinate frame.

All solving conventions are visible in the input. No parent generator, source
fixture, seed, or training example participates in solve().
"""
from __future__ import annotations

import copy
import json
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
# Rotate a right-facing canonical mask into the indicated inward direction.
_TURNS = {"left": 0, "bottom": 1, "right": 2, "top": 3}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in _STEPS:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    todo.append(p)
                    cells.append(p)
        result.append(np.array(cells, dtype=int))
    return result


def _profile(mask, side):
    """Four-connected negative space; eligibility uses the whole silhouette."""
    k = _TURNS[side]
    canonical = np.rot90(mask, -k)
    h, w = canonical.shape
    holes, active, open_rows = [], [], set()
    for cells in _components(~canonical):
        rr, cc = cells.T
        boundary = bool(np.any((rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)))
        if boundary:
            open_rows.update(int(r) for r, c in cells if c == w - 2)
            continue
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        _require(bool(np.all((hi - lo + 1) % 2 == 1)), "cavities require odd bounding boxes")
        rows = set(map(int, rr))
        contacts = {int(r) for r, c in cells if c == w - 2 and canonical[r, w - 1]}
        _require(not contacts or contacts == rows, "partial facing-wall contact is outside this family")
        # Map the exact negative-space center back to the original local mask.
        center = (lo + hi) // 2
        _require(not canonical[tuple(center)], "cavity center must be hollow")
        marker = np.zeros_like(canonical, dtype=bool)
        marker[tuple(center)] = True
        anchor = np.argwhere(np.rot90(marker, k))[0]
        hole = dict(cells=cells, anchor=anchor, shape=tuple(map(int, hi - lo + 1)),
                    active=contacts == rows)
        holes.append(hole)
        if hole["active"]:
            active.append(anchor)
    return dict(holes=holes, anchors=active, open_rows=open_rows,
                decoys=sum(not h["active"] for h in holes))


def _side_of(lo, hi, frame):
    t, l, b, r = frame
    sides = []
    if hi[1] < l and t < lo[0] <= hi[0] < b:
        sides.append("left")
    if lo[1] > r and t < lo[0] <= hi[0] < b:
        sides.append("right")
    if hi[0] < t and l < lo[1] <= hi[1] < r:
        sides.append("top")
    if lo[0] > b and l < lo[1] <= hi[1] < r:
        sides.append("bottom")
    _require(len(sides) == 1, "source must lie directly outside exactly one frame side")
    return sides[0]


def _parse_candidate(a, frame_color):
    cells = np.argwhere(a == frame_color)
    lo, hi = cells.min(axis=0), cells.max(axis=0)
    t, l = map(int, lo)
    b, r = map(int, hi)
    _require(b - t >= 4 and r - l >= 4, "frame is too small")
    _require(all(a[y, x] == frame_color for y, x in ((t,l),(t,r),(b,l),(b,r))), "frame corners")
    inside = a[t+1:b, l+1:r]
    bg = int(inside[0, 0])
    _require(bg != frame_color and bool(np.all(inside == bg)), "target must be blank")
    _require(bool(np.all((cells[:,0] == t) | (cells[:,0] == b) |
                        (cells[:,1] == l) | (cells[:,1] == r))), "frame color only on border")
    _require(np.array_equal(a[t,l+1:r], a[b,l+1:r]), "column labels must be paired")
    _require(np.array_equal(a[t+1:b,l], a[t+1:b,r]), "row labels must be paired")
    border = np.concatenate((a[t,l:r+1], a[b,l:r+1], a[t+1:b,l], a[t+1:b,r]))
    _require(not np.any(border == bg), "background is not a label")
    votes = np.bincount(border, minlength=10)
    _require(votes[frame_color] > max(votes[c] for c in range(10) if c != frame_color), "dominant frame")
    rows = {int(c): list(map(int, np.flatnonzero(a[t+1:b,l] == c) + t + 1))
            for c in np.unique(border) if c != frame_color}
    cols = {int(c): list(map(int, np.flatnonzero(a[t,l+1:r] == c) + l + 1))
            for c in np.unique(border) if c != frame_color}
    outside = np.ones(a.shape, dtype=bool)
    outside[t:b+1,l:r+1] = False
    objects = []
    source_colors = [int(c) for c in np.unique(a[outside]) if c != bg]
    _require(1 <= len(source_colors) <= 6, "need one to six source colors")
    for color in source_colors:
        source = np.argwhere((a == color) & outside)
        p, q = source.min(axis=0), source.max(axis=0)
        side = _side_of(p, q, (t,l,b,r))
        region = a[p[0]:q[0]+1,p[1]:q[1]+1]
        _require(bool(np.all((region == color) | (region == bg))), "source boxes may not contain other colors")
        mask = region == color
        _require(len(_components(mask)) == 1, "one connected source per color")
        profile = _profile(mask, side)
        obj = dict(color=color, cells=source, lo=p, hi=q, mask=mask,
                   side=side, rows=rows.get(color, []), cols=cols.get(color, []), **profile)
        obj["eligible"] = bool(obj["rows"] and obj["cols"] and obj["anchors"])
        if obj["eligible"]:
            offsets = np.concatenate([np.argwhere(mask) - anchor for anchor in obj["anchors"]])
            for y in obj["rows"]:
                for x in obj["cols"]:
                    dest = offsets + (y, x)
                    _require(bool(np.all((dest[:,0] > t) & (dest[:,0] < b) &
                                         (dest[:,1] > l) & (dest[:,1] < r))), "stamp crosses frame")
        objects.append(obj)
    # This requirement disambiguates a plain hollow source from the real frame.
    _require(any(o["rows"] and o["cols"] for o in objects), "frame needs matching complete coordinates")
    objects.sort(key=lambda o: tuple(map(int, o["lo"])))
    return dict(frame=(t,l,b,r), frame_color=int(frame_color), bg=bg, objects=objects,
                rows=rows, cols=cols)


def _parse(a):
    candidates = []
    for color in np.unique(a):
        try:
            candidates.append(_parse_candidate(a, int(color)))
        except (AssertionError, IndexError, ValueError):
            pass
    _require(len(candidates) == 1, "frame and source geometry must be uniquely interpretable")
    return candidates[0]


def _stamp_sets(o, mode="correct"):
    if not o["rows"] or not o["cols"]:
        return []
    anchors = o["anchors"]
    if mode == "all_holes":
        anchors = [h["anchor"] for h in o["holes"]]
    elif mode == "center":
        anchors = [(np.array(o["mask"].shape) - 1) // 2]
    elif mode == "first_anchor":
        anchors = anchors[:1]
    elif mode == "wall_pockets":
        # Deliberately wrong: promote open pockets beside the extreme wall too.
        k = _TURNS[o["side"]]
        canonical = np.rot90(o["mask"], -k)
        anchors = []
        for region in _components(~canonical):
            if not any(c == canonical.shape[1] - 2 and canonical[r, -1]
                       for r, c in region):
                continue
            center = (region.min(axis=0) + region.max(axis=0)) // 2
            marker = np.zeros_like(canonical)
            marker[tuple(center)] = True
            anchors.append(np.argwhere(np.rot90(marker, k))[0])
    if not anchors:
        return []
    rs, cs = o["rows"], o["cols"]
    if mode == "first_coordinate":
        rs, cs = rs[:1], cs[:1]
    if mode == "zip":
        destinations = list(zip(rs, cs))
    else:
        destinations = [(r,c) for r in rs for c in cs]
    pixels = np.argwhere(o["mask"])
    return [set(map(tuple, pixels - anchor + (r,c)))
            for anchor in anchors for r,c in destinations]


def _render(a, p, mode="correct", order="source"):
    out = a.copy()
    objects = list(p["objects"])
    if order == "reverse":
        objects.reverse()
    elif order == "destination":
        objects.sort(key=lambda o: (min(o["rows"]) if o["rows"] else 99,
                                    min(o["cols"]) if o["cols"] else 99))
    occupancy = set()
    t,l,b,r = p["frame"]
    for o in objects:
        stamps = _stamp_sets(o, mode)
        if not stamps:
            continue
        out[o["cells"][:,0],o["cells"][:,1]] = p["bg"]
        for stamp in stamps:
            for y,x in stamp - occupancy:
                # Wrong-rule diagnostics clip solely to permit comparison.
                if t < y < b and l < x < r:
                    out[y,x] = o["color"]
            occupancy.update(stamp)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the frame, inward direction, cavities, coordinates, and priority."""
    a = _grid(input_grid)
    return _render(a, _parse(a))


def _intervals(rng, height, count):
    # Odd cavity heights and at least one foreground row between cavities.
    spare = height - (2 * count + 1)
    gaps = rng.multinomial(spare, np.full(count + 1, 1 / (count + 1)))
    row, result = 1 + int(gaps[0]), []
    for i in range(count):
        result.append(row)
        row += 2 + int(gaps[i+1])
    return result


def _mask(rng, kind, max_size=7):
    count = 2 if kind == "multi" else 3 if kind == "triple" else 1
    min_h = 2 * count + 1
    h = int(rng.integers(min_h, max(max_size, min_h) + 1))
    w = int(rng.integers(3 if kind not in ("mixed", "recessed") else 5, max_size + 1))
    m = np.ones((h,w), dtype=bool)
    if kind == "solid":
        if h > 3 and w > 3:
            m[0,:int(rng.integers(1,w-1))] = False
        return m
    if kind == "open":
        m[1:-1,1:-1] = False
        # Keep the rightmost wall, but connect its pocket to the exterior.
        if rng.random() < .5:
            m[1:-1,0] = False
        else:
            m[0,w-2] = False
        return m
    if kind == "recessed":
        # A thin forward tail makes an otherwise thin-walled hole inactive.
        body_w = w - 1
        m[:,body_w:] = False
        d = int(rng.choice([v for v in (1,3,5) if v <= body_w-2]))
        m[1:2,body_w-1-d:body_w-1] = False
        m[int(rng.choice([0,h-1])),body_w:] = True
        return m
    if kind == "decoy":
        w = max(w,4)
        m = np.ones((h,w), dtype=bool)
        m[1,1] = False
        return m
    ys = _intervals(rng,h,count)
    for y in ys:
        d = int(rng.choice([v for v in (1,3,5) if v <= w-2]))
        m[y,w-1-d:w-1] = False
    if count == 1 and h >= 5 and rng.random() < .65:
        # Grow the cavity to odd height three while keeping its center hollow.
        y = min(max(2,ys[0]),h-3)
        m = np.ones((h,w),dtype=bool)
        d = int(rng.choice([v for v in (1,3,5) if v <= w-2]))
        m[y-1:y+2,w-1-d:w-1] = False
        if d >= 3 and rng.random() < .6:
            m[y-1,w-1-d] = True  # stepped back edge, same odd bounding box
    if kind == "mixed":
        # Ensure a separate recessed hole without changing the facing extreme.
        m = np.ones((h,max(w,7)),dtype=bool)
        w = m.shape[1]
        for y in ys:
            m[y,w-2] = False
        m[ys[0],1] = False
    # Irregular exterior without changing holes or their active classification.
    if rng.random() < .55:
        trial = m.copy()
        trial[int(rng.choice([0,h-1])),0] = False
        if len(_components(trial)) == 1:
            m = trial
    return m


def _offset_bounds(mask, side):
    anchors = _profile(mask, side)["anchors"]
    # Inactive templates still receive coordinates that would fit a copy
    # centered at their box center; this makes decoy comparisons legible.
    if not anchors:
        anchors = [(np.array(mask.shape)-1)//2]
    points = np.concatenate([np.argwhere(mask) - x for x in anchors])
    return points.min(axis=0), points.max(axis=0)


def _construct(rng, kind, palette, shift, bank=()):
    """Build sources first, then a frame large enough for their anchor offsets."""
    if kind == "anchors":
        kinds, counts = ["multi", "mixed"], [(2,2),(1,2)]
    elif kind == "contrasts":
        kinds, counts = ["active", "open", "recessed", "active"], [(1,1),(1,1),(1,1),(0,1)]
        if rng.random() < .5:
            counts[-1] = (1,0)
    else:
        n = int(rng.integers(1,6))
        pool = ["active","multi","mixed","open","recessed","solid","decoy"]
        if kind == "test" and shift == "extent":
            pool += ["triple","triple"]
        kinds = [str(rng.choice(pool)) for _ in range(n)]
        kinds[0] = "triple" if kind == "test" and shift == "extent" else "active"
        counts = []
        for i in range(n):
            limit = 3 if kind == "test" and shift == "multiplicity" else 2
            nr,nc = map(int,rng.integers(1,limit+1,size=2))
            if i and rng.random() < .23:
                nr,nc = [(0,0),(0,nc),(nr,0)][int(rng.integers(3))]
            counts.append((nr,nc))
        if kind == "test" and shift == "multiplicity":
            counts[0] = (3, int(rng.integers(1,3))) if rng.random() < .5 else (int(rng.integers(1,3)),3)
    canonical = []
    for name in kinds:
        if bank:
            canonical.append(bank[int(rng.integers(len(bank)))].copy())
        else:
            canonical.append(_mask(rng,name,7 if kind in ("test","random") else 6))
    sides = ["left","top","right","bottom"]
    rng.shuffle(sides)
    assignments = [sides[i % 4] for i in range(len(kinds))]
    if kind not in ("anchors","contrasts") and rng.random() < .35:
        assignments = [sides[0]] * len(kinds)
    objects = []
    for i,(mask,side,axes) in enumerate(zip(canonical,assignments,counts)):
        rotated = np.rot90(mask,_TURNS[side]).copy()
        low,high = _offset_bounds(rotated,side)
        objects.append(dict(mask=rotated,side=side,counts=axes,low=low,high=high,
                            anchors=_profile(rotated,side)["anchors"],index=i))
    depth = {s: max([o["mask"].shape[1 if s in ("left","right") else 0] + 1
                     for o in objects if o["side"] == s] or [0]) for s in sides}
    span = {s: sum(o["mask"].shape[0 if s in ("left","right") else 1]+1
                   for o in objects if o["side"] == s)-1 for s in sides}
    min_h = max(9,span["left"],span["right"],max(int(o["high"][0]-o["low"][0])+4 for o in objects))
    min_w = max(9,span["top"],span["bottom"],max(int(o["high"][1]-o["low"][1])+4 for o in objects))
    # 2 border cells; source margins and random empty margin slack.
    max_h,max_w = 28-depth["top"]-depth["bottom"],28-depth["left"]-depth["right"]
    if min_h > max_h or min_w > max_w:
        return None
    ih = int(rng.integers(min_h,min(max_h,max(min_h+2,20))+1))
    iw = int(rng.integers(min_w,min(max_w,max(min_w+2,20))+1))
    extra_h = int(rng.integers(min(3,max_h-ih)+1))
    extra_w = int(rng.integers(min(3,max_w-iw)+1))
    mt = depth["top"] + int(rng.integers(extra_h+1))
    mb = depth["bottom"] + extra_h - (mt-depth["top"])
    ml = depth["left"] + int(rng.integers(extra_w+1))
    mr = depth["right"] + extra_w - (ml-depth["left"])
    t,l,b,r = mt,ml,mt+ih+1,ml+iw+1
    height,width = b+1+mb,r+1+mr
    bg,fc = palette[:2]
    a = np.full((height,width),bg,dtype=np.int8)
    a[[t,b],l:r+1] = fc
    a[t:b+1,[l,r]] = fc
    rng.shuffle(objects)
    for s in sides:
        group = [o for o in objects if o["side"] == s]
        if not group:
            continue
        length = ih if s in ("left","right") else iw
        sizes = [o["mask"].shape[0 if s in ("left","right") else 1] for o in group]
        spare = length - sum(sizes) - len(group) + 1
        gaps = rng.multinomial(spare,np.full(len(group)+1,1/(len(group)+1)))
        cursor = (t if s in ("left","right") else l) + 1 + int(gaps[0])
        for i,o in enumerate(group):
            h,w = o["mask"].shape
            if s == "left":
                pos = (cursor,int(rng.integers(0,l-w)))
            elif s == "right":
                pos = (cursor,int(rng.integers(r+2,width-w+1)))
            elif s == "top":
                pos = (int(rng.integers(0,t-h)),cursor)
            else:
                pos = (int(rng.integers(b+2,height-h+1)),cursor)
            o["lo"] = np.array(pos)
            cursor += sizes[i]+1+int(gaps[i+1])
    objects.sort(key=lambda o:tuple(o["lo"]))
    for o,color in zip(objects,palette[2:]):
        o["color"] = color
        y,x = o["lo"]
        h,w = o["mask"].shape
        a[y:y+h,x:x+w][o["mask"]] = color
    used_rows,used_cols = set(),set()
    # Larger footprints reserve coordinates first; all choices are input-visible.
    for o in sorted(objects,key=lambda o:-o["mask"].size):
        for axis,(start,stop,used) in enumerate(((t+1,b,used_rows),(l+1,r,used_cols))):
            choices = [v for v in range(start-int(o["low"][axis]),stop-int(o["high"][axis])) if v not in used]
            need = o["counts"][axis]
            if len(choices) < need:
                return None
            coords = sorted(map(int,rng.choice(choices,need,replace=False)))
            used.update(coords)
            o["rows" if axis == 0 else "cols"] = coords
        for y in o["rows"]:
            a[y,l] = a[y,r] = o["color"]
        for x in o["cols"]:
            a[t,x] = a[b,x] = o["color"]
    out = a.copy()
    # Construction records paint back-to-front; solver instead tracks ownership.
    for o in objects:
        if o["anchors"] and o["rows"] and o["cols"]:
            y,x = o["lo"]
            h,w = o["mask"].shape
            out[y:y+h,x:x+w][o["mask"]] = bg
    for o in reversed(objects):
        for anchor in o["anchors"]:
            pixels = np.argwhere(o["mask"]) - anchor
            for y in o["rows"]:
                for x in o["cols"]:
                    dest = pixels + (y,x)
                    out[dest[:,0],dest[:,1]] = o["color"]
    return {"input":a,"output":out}


def _features(a, p=None):
    p = _parse(a) if p is None else p
    expected = _render(a,p)
    os = p["objects"]
    stamp_unions = [set().union(*_stamp_sets(o)) if _stamp_sets(o) else set() for o in os]
    conflicts = sum(len(s & q) for i,s in enumerate(stamp_unions) for q in stamp_unions[i+1:])
    return dict(shape=a.shape,count=len(os),sides=tuple(sorted({o["side"] for o in os})),
                masks=tuple((o["mask"].shape,o["mask"].tobytes()) for o in os),
                active_holes=tuple(len(o["anchors"]) for o in os),
                hole_shapes=tuple(h["shape"] for o in os for h in o["holes"]),
                open=sum(bool(o["open_rows"]) for o in os),decoys=sum(o["decoys"] for o in os),
                missing=sum(bool(o["anchors"]) and not (o["rows"] and o["cols"]) for o in os),
                products=tuple((len(o["rows"]),len(o["cols"])) for o in os if o["eligible"]),
                conflicts=conflicts,
                differences={mode:int(np.count_nonzero(expected != _render(a,p,mode=mode)))
                             for mode in ("center","all_holes","wall_pockets","first_anchor","first_coordinate","zip")},
                reverse=int(np.count_nonzero(expected != _render(a,p,order="reverse"))),
                destination=int(np.count_nonzero(expected != _render(a,p,order="destination"))))


def _make_pair(rng,kind,palette,shift,bank=()):
    for _ in range(300):
        pair = _construct(rng,kind,palette,shift,bank)
        if pair is None:
            continue
        try:
            p = _parse(pair["input"])
        except AssertionError:
            continue
        f = _features(pair["input"],p)
        if kind == "test" and not f["products"]:
            # Recombination can draw several inactive masks. Keep held-out
            # cases positive, while solve still supports wholly silent scenes.
            continue
        if kind == "anchors" and not (all(f["differences"][k] for k in
                                               ("center","first_anchor","first_coordinate","zip"))
                                       and f["reverse"] and f["destination"]):
            continue
        if kind == "contrasts" and not (f["open"] and f["decoys"] and f["missing"]
                                         and f["differences"]["all_holes"]
                                         and f["differences"]["wall_pockets"]):
            continue
        if not np.array_equal(pair["output"],_render(pair["input"],p)):
            raise AssertionError("construction and visible-input prediction disagree")
        return pair
    raise RuntimeError(f"bounded construction exhausted for {kind}")


def _integer(value,name,low=None,high=None):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if low is not None and not low <= value <= high:
        raise ValueError(f"{name} outside allowed range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed,"seed") % (1 << 64)
    nt = _integer(num_train,"num_train",2,10)
    ne = _integer(num_test,"num_test",1,4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base_palette = list(map(int,rng.permutation(10)))
    vary_palette = bool(rng.integers(2))
    shift = str(rng.choice(["multiplicity","extent","recombination"]))
    train,tests,seen = [],[],set()
    for i in range(nt):
        palette = list(map(int,rng.permutation(10))) if vary_palette else base_palette
        kind = "anchors" if i == 0 else "contrasts" if i == 1 else "random"
        pair = _make_pair(rng,kind,palette,shift)
        # Geometry resampling retains the puzzle's palette convention.
        for _ in range(20):
            if _key(pair["input"]) not in seen:
                break
            pair = _make_pair(rng,kind,palette,shift)
        else:
            raise RuntimeError("duplicate input after bounded resampling")
        seen.add(_key(pair["input"]))
        train.append(pair)
    bank = []
    if shift == "recombination":
        for pair in train:
            for o in _parse(pair["input"])["objects"]:
                bank.append(np.rot90(o["mask"],-_TURNS[o["side"]]).copy())
    for _ in range(ne):
        palette = list(map(int,rng.permutation(10))) if vary_palette else base_palette
        pair = _make_pair(rng,"test",palette,shift,bank)
        for _ in range(20):
            if _key(pair["input"]) not in seen:
                break
            pair = _make_pair(rng,"test",palette,shift,bank)
        else:
            raise RuntimeError("duplicate input after bounded resampling")
        seen.add(_key(pair["input"]))
        tests.append(pair)
    rng.shuffle(train)
    result = {"train":train,"test":tests}
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s],list) for s in ("train","test")), "split schema")
    nt,ne = len(puzzle["train"]),len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid counts")
    seen,training = set(),[]
    for split in ("train","test"):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {"input","output"}, "pair schema")
            a,out = _grid(pair["input"]),_grid(pair["output"])
            _require(a.shape == out.shape, "shape must be preserved")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            p = _parse(a)
            _require(np.array_equal(_render(a,p),out), "wrong stamping or priority")
            t,l,b,r = p["frame"]
            inside = np.zeros(a.shape,dtype=bool)
            inside[t+1:b,l+1:r] = True
            consumed = np.zeros(a.shape,dtype=bool)
            for o in p["objects"]:
                if o["eligible"]:
                    consumed[o["cells"][:,0],o["cells"][:,1]] = True
                    _require(bool(np.all(out[o["cells"][:,0],o["cells"][:,1]] == p["bg"])), "source not consumed")
                else:
                    _require(np.array_equal(out[o["cells"][:,0],o["cells"][:,1]],
                                            a[o["cells"][:,0],o["cells"][:,1]]), "inactive source changed")
            _require(np.array_equal(a[~(inside|consumed)],out[~(inside|consumed)]), "frame or unaffected pixels changed")
            _require(set(map(int,np.unique(out))) <= set(map(int,np.unique(a))), "unexpected color")
            if split == "train":
                training.append(_features(a,p))
    # Minimal two-example evidence budget still refutes these simpler rules.
    for mode in ("center","all_holes","wall_pockets","first_anchor","first_coordinate","zip"):
        _require(any(f["differences"][mode] for f in training), f"training does not distinguish {mode}")
    _require(any(f["reverse"] and f["destination"] for f in training), "priority underdemonstrated")
    _require(any(f["open"] for f in training), "missing enclosure contrast")
    _require(any(f["decoys"] for f in training), "missing global-extreme contrast")
    _require(any(f["missing"] for f in training), "missing coordinate eligibility contrast")


def self_test():
    """Bounded geometry, evidence, metamorphic, and corruption checks: 42 seeds."""
    shapes,masks,counts,sides,products,hole_shapes,signatures = set(),set(),set(),set(),set(),set(),set()
    all_features = []
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = tuple(_key(x[k]) for s in ("train","test") for x in p[s] for k in ("input","output"))
        signatures.add(signature)
        q = generate_puzzle(seed)
        _require(signature == tuple(_key(x[k]) for s in ("train","test") for x in q[s] for k in ("input","output")), "nondeterministic")
        for split in ("train","test"):
            for pair in p[split]:
                a,out = pair["input"],pair["output"]
                f = _features(a)
                all_features.append(f)
                shapes.add(a.shape); masks.update(f["masks"]); counts.add(f["count"])
                sides.update(f["sides"]); products.update(f["products"]); hole_shapes.update(f["hole_shapes"])
                # Color equality is the convention, including nonblack background.
                lookup = np.arange(9,-1,-1,dtype=np.int8)
                _require(np.array_equal(solve(lookup[a]),lookup[out]), "palette equivariance")
        if seed == 0:
            bad = copy.deepcopy(p)
            bad["test"][0]["output"][0,0] = (int(bad["test"][0]["output"][0,0])+1)%10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError("corrupt answer accepted")
            for mode in ("center","all_holes","wall_pockets","first_anchor","zip"):
                bad = copy.deepcopy(p)
                index = next(i for i,pair in enumerate(p["train"]) if _features(pair["input"])["differences"][mode])
                a = bad["train"][index]["input"]
                bad["train"][index]["output"] = _render(a,_parse(a),mode=mode)
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("shortcut answer accepted")
    _require(len(signatures) == 42, "cross-seed duplication")
    _require(len(shapes) >= 35 and len(masks) >= 120, "structural variation collapsed")
    _require({1,2,3,4,5} <= counts, "object count range missing")
    _require(sides == set(_TURNS), "missing inward direction")
    _require({(1,1),(1,2),(2,1),(2,2)} <= products, "coordinate products missing")
    _require(any(max(p) == 3 for p in products), "multiplicity shift missing")
    _require(any(max(f["active_holes"]) >= 3 for f in all_features), "cavity count shift missing")
    _require(len(hole_shapes) >= 5, "cavity sizes collapsed")
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            generate_puzzle(731+nt*7+ne,nt,ne)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return dict(seeds=42,pairs=len(all_features),distinct_puzzles=len(signatures),
                grid_shapes=len(shapes),source_masks=len(masks),object_counts=sorted(counts),
                directions=sorted(sides),coordinate_products=sorted(products),hole_shapes=sorted(hole_shapes),
                checks="authored geometry, ablations, palette equivariance and mutation rejection; no independent solver")


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",",":")))
