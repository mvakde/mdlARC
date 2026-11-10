"""Keyed cavity filling followed by whole-canvas consecutive-vector collapse.

Only the input determines solve(). Generation keeps construction-derived labels
separate from the input parser. No execution traces or per-seed payloads are saved.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


HOST_SEEDS = (0,953450941,318623210,274617717,41366658,544668412,492521000,
              187562747,65279898,123642500,41760789,262838876,913207997,
              798578657,286499724,145568808,479020864,644931488,724883810,
              556196348,915758767,184387314,641086447,281218305,893357663,
              312697082,21804726,348457810,17629856,640531722,159599881,
              32554437,337149138,140175268,757191497,512576106,806449790,
              928355131,430324489,12244037,429258163,3627944)
STEPS = ((-1,0),(1,0),(0,-1),(0,1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer rectangle with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    h,w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    steps = STEPS + ((-1,-1),(-1,1),(1,-1),(1,1)) if diagonal else STEPS
    result = []
    for r,c in np.argwhere(mask):
        r,c = int(r),int(c)
        if seen[r,c]:
            continue
        cells, stack = [], [(r,c)]
        seen[r,c] = True
        while stack:
            x,y = stack.pop()
            cells.append((x,y))
            for dx,dy in steps:
                xx,yy = x+dx,y+dy
                if 0 <= xx < h and 0 <= yy < w and mask[xx,yy] and not seen[xx,yy]:
                    seen[xx,yy] = True
                    stack.append((xx,yy))
        result.append(cells)
    return result


def _scene(input_grid):
    """Find the isolated local edge strip, then original four-connected cavities."""
    a = _grid(input_grid)
    bg = int(a[0,0])
    if not all(int(a[q]) == bg for q in ((0,-1),(-1,0),(-1,-1))):
        raise ValueError("the four exterior corners must show the background")
    h,w = a.shape
    candidates = []
    for cells in _components(a != bg, diagonal=True):
        rr,cc = zip(*cells)
        r0,r1,c0,c1 = min(rr),max(rr),min(cc),max(cc)
        sh,sw = r1-r0+1,c1-c0+1
        if len(cells) != sh*sw or len({int(a[q]) for q in cells}) < 2:
            continue
        lanes = []
        if sh == 2 and 3 <= sw <= 6:
            if r0 == 0:
                lanes.append(("top",a[r0,c0:c1+1],a[r1,c0:c1+1]))
            if r1 == h-1:
                lanes.append(("bottom",a[r1,c0:c1+1],a[r0,c0:c1+1]))
        if sw == 2 and 3 <= sh <= 6:
            if c0 == 0:
                lanes.append(("left",a[r0:r1+1,c0],a[r0:r1+1,c1]))
            if c1 == w-1:
                lanes.append(("right",a[r0:r1+1,c1],a[r0:r1+1,c0]))
        for side,outer,inner in lanes:
            if len(set(map(int,outer))) == len(outer):
                candidates.append({"side":side,"mapping":dict(zip(map(int,outer),map(int,inner))),
                                   "cells":set(cells),"background":bg})
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous two-lane edge key")
    key = candidates[0]
    cavities = []
    for cells in _components(a == bg):
        if any(r in (0,h-1) or c in (0,w-1) for r,c in cells):
            continue
        border = set()
        for r,c in cells:
            for dr,dc in STEPS:
                rr,cc = r+dr,c+dc
                if 0 <= rr < h and 0 <= cc < w and a[rr,cc] != bg:
                    border.add((rr,cc))
        colors = {int(a[q]) for q in border}
        if len(colors) != 1 or border & key["cells"]:
            raise ValueError("cavity must have one original contour color")
        cavities.append((cells,next(iter(colors))))
    return a,key,cavities


def _fill(a, key, cavities, reverse=False, recursive=False):
    mapping = key["mapping"]
    if reverse:
        # Only a diagnostic competing rule; ambiguous inverses are ignored.
        mapping = {v:k for k,v in mapping.items()}
    out = a.copy()
    for cells,color in cavities:
        if color not in mapping:
            continue
        dest = mapping[color]
        if recursive:
            seen = {color}
            while dest in mapping and dest not in seen:
                seen.add(dest)
                dest = mapping[dest]
        for q in cells:
            out[q] = dest
    return out


def _keep(a):
    rows = np.r_[True,np.any(a[1:] != a[:-1],axis=1)]
    cols = np.r_[True,np.any(a[:,1:] != a[:,:-1],axis=0)]
    return rows,cols


def _collapse(a):
    rows,cols = _keep(a)
    return a[np.ix_(rows,cols)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    a,key,cavities = _scene(input_grid)
    return _collapse(_fill(a,key,cavities))


def _object(rng, kind, holes=1):
    """Return original ink and the intended enclosed background cells."""
    if kind == "multi":
        nr = int(rng.integers(1, min(3,holes)+1))
        nc = (holes+nr-1)//nr
        # A long one-row partition and a compact two/three-row partition coexist.
        hs = list(map(int,rng.integers(1,3,size=nr)))
        ws = list(map(int,rng.integers(1,4,size=nc)))
        h,w = sum(hs)+nr+1,sum(ws)+nc+1
        hole = np.zeros((h,w),dtype=bool)
        slots = list(map(int,rng.permutation(nr*nc)[:holes]))
        for index in slots:
            i,j = divmod(index,nc)
            r,c = 1+sum(hs[:i])+i,1+sum(ws[:j])+j
            hole[r:r+hs[i],c:c+ws[j]] = True
        return ~hole,hole
    if kind == "diamond":
        radius = int(rng.integers(1,4))
        rr,cc = np.indices((2*radius+1,2*radius+1))
        d = abs(rr-radius)+abs(cc-radius)
        return d == radius,d < radius
    if kind == "profile":
        h,w = int(rng.integers(2,5)),int(rng.integers(2,5))
        hole = np.zeros((h+2,w+2),dtype=bool)
        hole[1:h+1,1:w+1] = True
        # Concave L-shaped cavity: bounding-box filling would repaint this wall.
        hole[1:1+int(rng.integers(1,h)),1:1+int(rng.integers(1,w))] = False
        expanded = hole.copy()
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                expanded |= np.roll(np.roll(hole,dr,axis=0),dc,axis=1)
        ink = expanded & ~hole
        rr,cc = np.nonzero(ink)
        sl = np.s_[rr.min():rr.max()+1,cc.min():cc.max()+1]
        return ink[sl].copy(),hole[sl].copy()
    h,w = int(rng.integers(3,8)),int(rng.integers(3,8))
    ink = np.ones((h,w),dtype=bool)
    ink[1:-1,1:-1] = False
    hole = ~ink
    if kind == "open":
        ink[0,int(rng.integers(1,w-1))] = False
        hole = np.zeros_like(ink)
    elif kind != "rectangle":
        raise ValueError("unknown object kind")
    return ink,hole


def _expand(rng, a, out, key_length, key_start):
    h,w = a.shape
    row_runs,col_runs = np.ones(h,dtype=int),np.ones(w,dtype=int)
    for runs,allowed in ((row_runs,list(range(2,h))),
                         (col_runs,[c for c in range(w) if not key_start <= c < key_start+key_length])):
        extra = int(rng.integers(0, min(30-len(runs),9)+1))
        for _ in range(extra):
            runs[int(rng.choice(allowed))] += 1
    return tuple(np.repeat(np.repeat(x,row_runs,axis=0),col_runs,axis=1) for x in (a,out))


def _safe_pair(role, turn, variant=0):
    """Constructive fallback: disjoint supports guarantee productive equality.

    Ten distinct corner colors supply more uniqueness variants than the maximum
    five pairs on any one key side in a legal generated puzzle.
    """
    a = np.zeros((28,28),dtype=np.int8)
    a[:2,3:6] = [[1,2,3],[1,3,7]]
    out = a.copy()
    nh = 1 if role == "first" else 2 if role == "second" else 3+variant%3 if role == "test" else 1+variant%2
    hole = np.zeros((3,2*nh+1),dtype=bool)
    hole[1,1::2] = True
    shapes = [(20,15,1,~hole,hole)]
    rr,cc = np.indices((5,5))
    if role == "second":
        distance = abs(rr-2)+abs(cc-2)
        ink,hole = distance == 2,distance < 2
    else:
        hole = np.zeros((5,5),dtype=bool)
        hole[1:4,1:4] = True
        hole[1,1] = False
        ink = ~hole
    shapes.append((5,2,2,ink,hole))
    ring = np.ones((5,5),dtype=bool)
    ring[1:4,1:4] = False
    if role == "first":
        shapes.append((5,9,8,ring,~ring))
    if role == "second":
        opened = ring.copy()
        opened[0,2] = False
        shapes.extend([(11,2,2,opened,np.zeros_like(opened)),(11,9,7,ring,~ring)])
    mapping = {1:1,2:3,3:7}
    for r,c,color,ink,hole in shapes:
        h,w = ink.shape
        a[r:r+h,c:c+w][ink] = color
        out[r:r+h,c:c+w][ink] = color
        if color in mapping:
            out[r:r+h,c:c+w][hole] = mapping[color]
    a,out = (a+variant)%10,(out+variant)%10
    return {"input":np.rot90(a,turn).copy(),"output":_collapse(np.rot90(out,turn))}


def _make_pair(rng, role, turn, extra_count=None):
    """Bounded random packing; labels come from the object hole masks."""
    for attempt in range(96):
        n = int(rng.integers(3,7))
        mapping = {1:1,2:3,3:7}
        for c in range(4,n+1):
            mapping[c] = int(rng.choice([1,2,3,4,5,6,7,9]))
        order = list(map(int,rng.permutation(list(mapping))))
        if role == "first":
            specs = [(1,"rectangle",1),(2,"profile",1),(8,"rectangle",1)]
        elif role == "second":
            specs = [(1,"multi",2),(2,"diamond",1),(2,"open",0),(7,"rectangle",1)]
        else:
            nh = int(rng.integers(3,6)) if role == "test" else int(rng.integers(1,3))
            specs = [(1,"multi" if nh > 1 else "rectangle",nh),
                     (2,str(rng.choice(["rectangle","profile","diamond"])),1)]
            count = int(rng.integers(2,7)) if extra_count is None else extra_count
            while len(specs) < count:
                color = int(rng.choice(list(mapping)+[7,8]))
                kind = str(rng.choice(["rectangle","profile","diamond","open","multi"]))
                specs.append((color,kind,int(rng.integers(1,3))))
        objects = [(c,*_object(rng,kind,nh)) for c,kind,nh in specs]
        # Larger logical canvases make the bounded packing progressively easier.
        low = min(24,14+len(specs))
        h = int(rng.integers(low,27)) if attempt < 72 else 27
        w = int(rng.integers(low,27)) if attempt < 72 else 27
        if max(x[1].shape[0] for x in objects) > h-2 or max(x[1].shape[1] for x in objects) > w-2:
            continue
        a = np.zeros((h,w),dtype=np.int8)
        out = a.copy()
        start = int(rng.integers(1,w-n))
        a[0,start:start+n] = order
        a[1,start:start+n] = [mapping[c] for c in order]
        out[:] = a
        blocked = np.zeros((h,w),dtype=bool)
        blocked[:3,max(0,start-1):min(w,start+n+1)] = True
        placed = True
        for i in sorted(range(len(objects)),key=lambda i:-objects[i][1].size):
            color,ink,hole = objects[i]
            oh,ow = ink.shape
            positions = [(r,c) for r in range(1,h-oh) for c in range(1,w-ow)
                         if not np.any(blocked[r:r+oh,c:c+ow])]
            if not positions:
                placed = False
                break
            r,c = positions[int(rng.integers(len(positions)))]
            a[r:r+oh,c:c+ow][ink] = color
            out[r:r+oh,c:c+ow][ink] = color
            if color in mapping:
                out[r:r+oh,c:c+ow][hole] = mapping[color]
            blocked[r-1:r+oh+1,c-1:c+ow+1] = True
        if not placed:
            continue
        before,after = _keep(a),_keep(out)
        # Every pair demonstrates new equality on both complete-vector axes.
        if not all(np.any(b & ~f) for b,f in zip(before,after)):
            continue
        a,out = _expand(rng,a,out,n,start)
        if rng.random() < .5:
            a,out = np.fliplr(a),np.fliplr(out)
        palette = rng.permutation(10).astype(np.int8)
        a,out = palette[a],palette[out]
        a,out = np.rot90(a,turn).copy(),np.rot90(out,turn).copy()
        return {"input":a,"output":_collapse(out)}
    return _safe_pair(role,turn,int(rng.integers(10)))


def _count(value, name, low, high):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape,a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt,ne = _count(num_train,"num_train",2,10),_count(num_test,"num_test",1,4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    rotation = int(rng.integers(4))
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",nt),("test",ne)):
        for i in range(count):
            role = ("first" if i == 0 else "second" if i == 1 else "other") if split == "train" else "test"
            turn = (rotation+(i % 2)) % 4 if split == "train" else (rotation+2+(i % 2)) % 4
            for _ in range(8):
                pair = _make_pair(rng,role,turn)
                k = _key(pair["input"])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                for variant in range(10):
                    pair = _safe_pair(role,turn,variant)
                    k = _key(pair["input"])
                    if k not in seen:
                        seen.add(k)
                        puzzle[split].append(pair)
                        break
                else:
                    raise AssertionError("constructive uniqueness bound violated")
    return puzzle


def _features(pair):
    a,key,cavities = _scene(pair["input"])
    filled = _fill(a,key,cavities)
    out = _grid(pair["output"])
    rows,cols = _keep(filled)
    expected = filled[np.ix_(rows,cols)]
    assert np.array_equal(out,expected), "wrong fill/collapse answer"
    # Reconstruct all filled pixels from the retained representatives and runs.
    ri,ci = np.cumsum(rows)-1,np.cumsum(cols)-1
    assert np.array_equal(out[np.ix_(ri,ci)],filled), "loss in run contraction"
    assert np.array_equal(filled[a != key["background"]],a[a != key["background"]]), "ink changed"
    assert all(filled[q] == a[q] for q in key["cells"]), "key repainted"
    assert out.shape[0] < a.shape[0] and out.shape[1] < a.shape[1]
    assert np.all(np.any(out[1:] != out[:-1],axis=1))
    assert np.all(np.any(out[:,1:] != out[:,:-1],axis=0))
    before = _keep(a)
    created = tuple(int(np.count_nonzero(b & ~f)) for b,f in zip(before,(rows,cols)))
    assert min(created) > 0, "fills must create full-vector equality on both axes"
    assert not np.array_equal(_collapse(a),out), "A-only shortcut survived"
    assert not np.array_equal(filled,out), "B-only shortcut survived"
    # Fixing representatives before filling cannot remove the newly equal bands.
    assert filled[np.ix_(*before)].shape != out.shape, "wrong-order shortcut survived"
    bg = key["background"]
    obj = []
    nonkey = a != bg
    for q in key["cells"]:
        nonkey[q] = False
    ids = np.full(a.shape,-1,dtype=int)
    for cells in _components(nonkey,diagonal=True):
        colors = {int(a[q]) for q in cells}
        assert len(colors) == 1
        index = len(obj)
        for q in cells:
            ids[q] = index
        obj.append({"color":next(iter(colors)),"holes":0,"cells":cells})
    irregular = diagonal = False
    for cells,color in cavities:
        owners = set()
        for r,c in cells:
            for dr,dc in STEPS:
                rr,cc = r+dr,c+dc
                if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1] and ids[rr,cc] >= 0:
                    owners.add(int(ids[rr,cc]))
        assert len(owners) == 1
        owner = obj[next(iter(owners))]
        owner["holes"] += 1
        rr,cc = zip(*cells)
        irregular |= len(cells) != (max(rr)-min(rr)+1)*(max(cc)-min(cc)+1)
        mask = ids == next(iter(owners))
        diagonal |= len(_components(mask)) > 1
    mapping = key["mapping"]
    active = [(cells,c) for cells,c in cavities if c in mapping]
    assert any(mapping[c] == c for cells,c in active), "missing self-map witness"
    assert any(mapping[c] != c for cells,c in active), "missing directed fill witness"
    # Adjacent-only compression retains repetitions separated by other vectors.
    separated = tuple(len({tuple(map(int,v)) for v in vectors}) < len(vectors) for vectors in (out,out.T))
    assert any(separated), "missing separated-repeat witness"
    return {"side":key["side"],"background":bg,"key_length":len(mapping),
            "objects":len(obj),"max_holes":max(o["holes"] for o in obj),
            "self_holes":max(o["holes"] for o in obj if mapping.get(o["color"]) == o["color"]),
            "created":created,"irregular":bool(irregular),"diagonal":bool(diagonal),
            "open_active":any(o["color"] in mapping and not o["holes"] for o in obj),
            "inert":any(o["color"] not in mapping and o["holes"] for o in obj),
            "chain":any(mapping[c] in mapping and mapping[mapping[c]] != mapping[c] for cells,c in active),
            "reverse_differs":not np.array_equal(_collapse(_fill(a,key,cavities,reverse=True)),out),
            "recursive_differs":not np.array_equal(_collapse(_fill(a,key,cavities,recursive=True)),out),
            "separated":separated,"input_shape":a.shape,"output_shape":out.shape}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    _count(len(puzzle["train"]),"num_train",2,10)
    _count(len(puzzle["test"]),"num_test",1,4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen,features = set(),{"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            a = _grid(pair["input"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            features[split].append(_features(pair))
    train,test = features["train"],features["test"]
    for feature in ("irregular","diagonal","open_active","inert","chain","reverse_differs","recursive_differs"):
        assert any(f[feature] for f in train), "missing training witness: "+feature
    assert {0,1} <= {axis for f in train for axis,present in enumerate(f["separated"]) if present}
    assert len({f["side"] for f in train}) == 2
    assert not {f["side"] for f in train} & {f["side"] for f in test}
    assert max(f["self_holes"] for f in train) <= 2
    assert all(f["self_holes"] >= 3 for f in test)


def self_test():
    """42 seeds, streaming metrics, explicit semantic witnesses, no disk writes."""
    shapes,normalized,digests,sides,backgrounds,keys,counts = set(),set(),set(),set(),set(),set(),set()
    totals = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train","test"):
            for p in puzzle[split]:
                f = _features(p)
                for name in ("irregular","diagonal","open_active","inert","chain"):
                    totals[name] += bool(f[name])
                shapes.add(f["input_shape"])
                sides.add(f["side"])
                backgrounds.add(f["background"])
                keys.add(f["key_length"])
                counts.add(f["objects"])
                roles = {}
                a = p["input"]
                norm = bytes(roles.setdefault(int(c),len(roles)) for c in a.flat)
                normalized.add((a.shape,hashlib.sha256(norm).digest()))
                digest.update(str(a.shape).encode())
                digest.update(a.tobytes())
                digest.update(p["output"].tobytes())
                assert np.array_equal(solve(a.copy()),p["output"])
        digests.add(digest.digest())
    assert len(digests) == 42 and len(normalized) >= 200
    assert len(shapes) >= 45 and sides == {"top","bottom","left","right"}
    assert backgrounds == set(range(10)) and keys == {3,4,5,6}
    assert {2,3,4,5,6} <= counts
    assert all(totals[n] >= 42 for n in totals)
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(42,nt,ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
            validate_generated_puzzle(p)
    safe = {"train":[_safe_pair("first" if i == 0 else "second" if i == 1 else "other",i%2,i)
                     for i in range(10)],
            "test":[_safe_pair("test",2+i%2,i) for i in range(4)]}
    validate_generated_puzzle(safe)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(3.0,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p,q = generate_puzzle(91,2,1),generate_puzzle(91,2,1)
    assert all(np.array_equal(x[k],y[k]) for s in p for x,y in zip(p[s],q[s]) for k in x)
    # Hand-calculated semantic regression. Filling the 1-ring merges all three
    # rows and columns; key columns 1 and 2 remain distinct complete vectors.
    a = np.zeros((9,13),dtype=np.int8)
    a[:2,1:4] = [[1,2,3],[1,3,7]]
    a[4:7,7:10] = [[1,1,1],[1,0,1],[1,1,1]]
    a[4:7,2:5] = [[2,2,2],[2,0,2],[2,2,2]]
    filled = a.copy()
    filled[5,8],filled[5,3] = 1,3
    assert np.array_equal(solve(a),_collapse(filled))
    hand_expected = np.array([list(map(int,row)) for row in (
        "01230000","01370000","00000000","00222010",
        "00232010","00222010","00000000")],dtype=np.int8)
    assert np.array_equal(solve(a),hand_expected)
    # The distant key blocks local shrinking of a solid self-filled rectangle.
    b = np.zeros((9,15),dtype=np.int8)
    b[:2,2:5] = [[1,2,3],[1,3,7]]
    b[4:7,1:7] = 1
    b[5,2:6] = 0
    b[4:7,10:13] = [[2,2,2],[2,0,2],[2,2,2]]
    hand_global = np.array([list(map(int,row)) for row in (
        "00123000000","00137000000","00000000000",
        "01111102220","01111102320","01111102220","00000000000")],dtype=np.int8)
    assert np.array_equal(solve(b),hand_global)
    # A deliberately wrong output must be rejected by the family validator.
    broken = {s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0,0] = (int(broken["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong answer was accepted")
    return {"seeds":42,"pairs":210,"distinct_puzzles":len(digests),
            "input_shapes":len(shapes),"normalized_inputs":len(normalized),
            "key_lengths":sorted(keys),"object_counts":sorted(counts),
            "feature_pairs":dict(totals),"legal_count_combinations":30,
            "fallback_pairs_checked":14,"hand_regressions":2}
