"""Synchronized stencil cuts: a separate row/column test at every inset pixel.

Tagged frames make stencil coordinates and palette roles entirely input-visible.
There are no imports, calls, or stored pairs from either parent generator.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


HOST_SEEDS = (0,26368074,897812782,865748037,657183522,616723021,
    505093018,216604419,628335548,970421449,565070290,340062104,
    392455572,184803142,164454568,218737783,669709007,609141795,
    756806334,658316481,540962956,372283709,240361356,215062882,
    964317059,474373816,292796049,296049801,635329260,492622045,
    397535864,375106422,102542833,341374429,920675327,401034212,
    657624346,805036189,90630839,377350073,140922372,124801694)
SHAPES = ((2,2),(2,3),(3,2),(2,4),(4,2))


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _digest(a):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.uint8).tobytes()).hexdigest()


def _d4(a):
    """Transform only the last two (local stencil) axes."""
    return [np.rot90(a, n, axes=(-2,-1)) for n in range(4)] + [
        np.rot90(np.flip(a, -1), n, axes=(-2,-1)) for n in range(4)]


def _cuts(masks):
    """R,C,K,K votes -> ink belonging to any complete macro row/column."""
    rows = np.all(masks, axis=1, keepdims=True)
    cols = np.all(masks, axis=0, keepdims=True)
    return masks & (rows | cols)


def _support(rng, rows, cols, kind):
    a = np.zeros((rows, cols), dtype=bool)
    r, c = int(rng.integers(rows)), int(rng.integers(cols))
    if kind == "row":
        a[r] = True
    elif kind == "col":
        a[:, c] = True
    elif kind == "cross":
        a[r] = True
        a[:, c] = True
    elif kind == "loose":
        # A completed line plus one inked tile outside every completed line.
        if rows >= 3:
            a[r] = True
            a[(r + int(rng.integers(1, rows))) % rows, c] = True
        elif cols >= 3:
            a[:, c] = True
            a[r, (c + int(rng.integers(1, cols))) % cols] = True
        else:
            return _support(rng, rows, cols, "diagonal")
    elif kind == "diagonal":
        rr, cc = rng.permutation(rows), rng.permutation(cols)
        for i in range(min(rows, cols)):
            a[int(rr[i]), int(cc[i])] = True
    elif kind == "near":
        if rng.random() < .5:
            a[r] = True
        else:
            a[:, c] = True
        a[r, c] = False
    elif kind == "full":
        a[:] = True
    elif kind == "empty":
        pass
    elif kind == "multi":
        a[rng.choice(rows, int(rng.integers(1, rows + 1)), replace=False)] = True
        if rng.random() < .5:
            a[:, rng.choice(cols, int(rng.integers(1, cols + 1)), replace=False)] = True
    else:
        a = rng.random((rows, cols)) < float(rng.uniform(.15, .85))
    return a


def _masks(rng, rows, cols, k, lesson):
    kinds = ("row","col","cross","loose","diagonal","near",
             "full","empty","multi","random","random")
    for attempt in range(32):
        cube = np.empty((rows, cols, k, k), dtype=bool)
        for y, x in np.ndindex(k, k):
            cube[:,:,y,x] = _support(rng, rows, cols, _pick(rng, kinds))
        if lesson:
            # Five independent local coordinates expose both axes, negatives,
            # extra votes and simultaneous lines in each minimal demonstration.
            cells = rng.permutation(k*k)[:5]
            for cell, kind in zip(cells, ("row","col","diagonal","loose","cross")):
                y, x = divmod(int(cell), k)
                cube[:,:,y,x] = _support(rng, rows, cols, kind)
        cut = _cuts(cube)
        orientation_unique = all(not np.array_equal(cut, t) for t in _d4(cut)[1:])
        if (cut.any() and np.any(cube & ~cut)
                and np.any(~cut) and (not lesson or orientation_unique)):
            return cube
    # Constructive finite fallback; the three distinct signatures at three
    # corners establish local orientation. Tags allow all other cells empty.
    cube = np.zeros((rows, cols, k, k), dtype=bool)
    cube[0,:,0,0] = True
    cube[:,0,0,k-1] = True
    if k >= 3:
        cube[0,:,k-1,0] = True
        cube[:,0,k-1,0] = True
        cube[:,:,1,1] = _support(rng, rows, cols, "loose")
        cube[:,:,1,0] = _support(rng, rows, cols, "diagonal")
    else:
        cube[0,0,k-1,0] = True
    return cube


def _bank_options(rows, cols, s, count):
    ah, aw = rows*(s+1)+1, cols*(s+1)+1
    options = []
    for nc in range(1, count+1):
        minimum = (count+nc-1)//nc
        for nr in (minimum, minimum+1):
            for gy, gx in itertools.product((1,2), repeat=2):
                bh, bw = nr*s+(nr-1)*gy, nc*s+(nc-1)*gx
                if max(ah,bh+2) <= 30 and aw+bw+2 <= 30:
                    options.append(("side",nr,nc,gy,gx,bh,bw,max(ah,bh+2),aw+bw+2))
                if ah+bh+2 <= 30 and max(aw,bw+2) <= 30:
                    options.append(("above",nr,nc,gy,gx,bh,bw,ah+bh+2,max(aw,bw+2)))
    return options


def _bank(rng, rows, cols, s, count):
    ah, aw = rows*(s+1)+1, cols*(s+1)+1
    mode,nr,nc,gy,gx,bh,bw,h,w = _pick(rng, _bank_options(rows,cols,s,count))
    if mode == "side":
        ay, ax = int(rng.integers(h-ah+1)), 0
        by, bx = int(rng.integers(1,h-bh)), aw+1
    else:
        ay, ax = 0, int(rng.integers(w-aw+1))
        by, bx = ah+1, int(rng.integers(1,w-bw))
    slots = [(by+r*(s+gy),bx+c*(s+gx)) for r in range(nr) for c in range(nc)]
    slots = [slots[int(i)] for i in rng.choice(len(slots),count,replace=False)]
    if rng.random() < .5:
        ay = h-ah-ay
        slots = [(h-s-y,x) for y,x in slots]
    if rng.random() < .5:
        ax = w-aw-ax
        slots = [(y,w-s-x) for y,x in slots]
    return h,w,ay,ax,slots


def _layout(rng, rows, cols, s, count, scatter):
    ah, aw = rows*(s+1)+1, cols*(s+1)+1
    if scatter:
        # Eight attempts, each scanning at most 26^2 positions once. No logs
        # or candidate payloads are retained. Bank fallback always fits.
        for _ in range(8):
            h = int(rng.integers(max(24,ah),31))
            w = int(rng.integers(max(24,aw),31))
            ay, ax = int(rng.integers(h-ah+1)), int(rng.integers(w-aw+1))
            busy = np.zeros((h,w),bool)
            busy[ay:ay+ah,ax:ax+aw] = True
            candidates = [(y,x) for y in range(1,h-s) for x in range(1,w-s)]
            slots = []
            for index in rng.permutation(len(candidates)):
                y,x = candidates[int(index)]
                if not busy[y:y+s,x:x+s].any():
                    slots.append((y,x))
                    busy[y-1:y+s+1,x-1:x+s+1] = True
                    if len(slots) == count:
                        return h,w,ay,ax,slots
    return _bank(rng,rows,cols,s,count)


def _render(rng, rows, cols, k, bg, marker, lesson, scatter):
    s, n = k+2, rows*cols
    possible = [d for d in range(9-n) if _bank_options(rows,cols,s,n+d)]
    distractors = _pick(rng,possible)
    colors = list(map(int,rng.choice([c for c in range(10) if c not in (bg,marker)],
                                     n+distractors,replace=False)))
    cube = _masks(rng,rows,cols,k,lesson)
    patterns = list(cube.reshape(n,k,k))
    for _ in range(distractors):
        patterns.append(rng.random((k,k)) < float(rng.uniform(.1,.9)))
    h,w,ay,ax,slots = _layout(rng,rows,cols,s,n+distractors,scatter)
    inp = np.full((h,w),bg,dtype=np.int8)
    out = np.full((rows*(s+1)+1,cols*(s+1)+1),bg,dtype=np.int8)
    for index,color in enumerate(colors[:n]):
        r,c = divmod(index,cols)
        y,x = 1+r*(s+1),1+c*(s+1)
        inp[ay+y:ay+y+s,ax+x:ax+x+s] = color
        out[y:y+s,x:x+s] = color
    # Construction uses explicit support sets and line membership. The input
    # solver below instead reconstructs a tensor and reduces its visible axes.
    for u,v in np.ndindex(k,k):
        active = {(r,c) for r,c in np.ndindex(rows,cols) if cube[r,c,u,v]}
        winning = set()
        for r in range(rows):
            line = {(r,c) for c in range(cols)}
            if line <= active:
                winning.update(line)
        for c in range(cols):
            line = {(r,c) for r in range(rows)}
            if line <= active:
                winning.update(line)
        for r,c in winning:
            out[2+r*(s+1)+u,2+c*(s+1)+v] = bg
    for slot,index in zip(slots,rng.permutation(n+distractors)):
        y,x = slot
        color = colors[int(index)]
        block = inp[y:y+s,x:x+s]
        block[:] = marker
        block[1:-1,1:-1] = bg
        block[0,0] = color
        block[1:-1,1:-1][patterns[int(index)]] = color
    extra = []
    for length in inp.shape:
        total = int(rng.integers(min(3,30-length)+1))
        a = int(rng.integers(total+1))
        extra.append((a,total-a))
    inp = np.pad(inp,extra,constant_values=bg)
    # Whole-scene symmetries transform macro positions and stencil coordinates
    # together. Each displayed tag may occupy any corner after this transform.
    rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
    if reflect:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {"input":np.rot90(inp,rotation).copy(),"output":np.rot90(out,rotation).copy()}


def _components(grid,bg):
    seen = np.zeros(grid.shape,bool)
    components = []
    for yy,xx in np.argwhere(grid != bg):
        y,x = int(yy),int(xx)
        if seen[y,x]:
            continue
        color = int(grid[y,x])
        todo,cells = [(y,x)],[]
        seen[y,x] = True
        while todo:
            r,c = todo.pop()
            cells.append((r,c))
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                a,b = r+dr,c+dc
                if (0 <= a < grid.shape[0] and 0 <= b < grid.shape[1]
                        and not seen[a,b] and grid[a,b] == color):
                    seen[a,b] = True
                    todo.append((a,b))
        points = np.array(cells,dtype=int)
        y0,x0 = points.min(axis=0)
        y1,x1 = points.max(axis=0)+1
        components.append((color,int(y0),int(x0),int(y1),int(x1),points))
    return components


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0,0])
    assert np.all(a[[0,-1],:] == bg) and np.all(a[:,[0,-1]] == bg)
    components = _components(a,bg)
    solids = [(y1-y0,y0,x0,color) for color,y0,x0,y1,x1,p in components
              if y1-y0 == x1-x0 and y1-y0 >= 4 and len(p) == (y1-y0)**2]
    assert solids, "missing solid tile array"
    s = max(t[0] for t in solids)
    assert 4 <= s <= 6
    tiles = sorted(t for t in solids if t[0] == s)
    ys,xs = sorted({t[1] for t in tiles}),sorted({t[2] for t in tiles})
    rows,cols,k = len(ys),len(xs),s-2
    assert (rows,cols) in SHAPES
    assert all(b-a == s+1 for q in (ys,xs) for a,b in zip(q,q[1:]))
    assert {(t[1],t[2]) for t in tiles} == set(itertools.product(ys,xs))
    colors = [t[3] for t in tiles]
    assert len(set(colors)) == len(colors)
    y0,x0,y1,x1 = ys[0]-1,xs[0]-1,ys[-1]+s+1,xs[-1]+s+1
    assert y0 >= 0 and x0 >= 0 and y1 <= len(a) and x1 <= a.shape[1]
    expected = np.full(a.shape,bg,dtype=np.int8)
    for _,y,x,color in tiles:
        expected[y:y+s,x:x+s] = color
    ring = np.ones((s,s),bool)
    ring[1:-1,1:-1] = False
    corners = ((0,0),(0,s-1),(s-1,0),(s-1,s-1))
    patterns,frames,markers = {},[],set()
    occupied = np.zeros(a.shape,bool)
    occupied[y0:y1,x0:x1] = True
    for color,fy,fx,by,bx,points in components:
        if (by-fy,bx-fx) != (s,s) or len(points) != 4*s-5:
            continue
        block = a[fy:by,fx:bx]
        matches = []
        for cy,cx in corners:
            border = ring.copy()
            border[cy,cx] = False
            if np.all(block[border] == color) and block[cy,cx] not in (bg,color):
                matches.append((cy,cx))
        if not matches:
            continue
        assert len(matches) == 1 and not occupied[fy:by,fx:bx].any()
        cy,cx = matches[0]
        tag = int(block[cy,cx])
        assert tag not in patterns and color not in (bg,*colors)
        assert np.all((block[1:-1,1:-1] == bg) | (block[1:-1,1:-1] == tag))
        occupied[fy-1:by+1,fx-1:bx+1] = True
        patterns[tag] = block[1:-1,1:-1] == tag
        frames.append((fy,fx,tag))
        markers.add(color)
        expected[fy:by,fx:bx] = block
    assert len(markers) == 1 and set(colors) <= set(patterns)
    assert 4 <= len(patterns) <= 8 and len(patterns) + 2 <= 10
    marker = next(iter(markers))
    assert marker not in patterns and np.array_equal(a,expected), "unexpected ink or malformed frame"
    cube = np.stack([patterns[c] for c in colors]).reshape(rows,cols,k,k)
    crop = a[y0:y1,x0:x1].copy()
    return {"input":a,"background":bg,"marker":marker,"size":s,
            "rows":rows,"cols":cols,"colors":colors,"patterns":patterns,
            "frames":sorted(frames),"cube":cube,"crop":crop}


def _output(info,cuts):
    result = info["crop"].copy()
    s,bg = info["size"],info["background"]
    for r,c in np.ndindex(info["rows"],info["cols"]):
        y,x = 2+r*(s+1),2+c*(s+1)
        result[y:y+s-2,x:x+s-2][cuts[r,c]] = bg
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only displayed color links, registers and the solid array lattice."""
    info = _parse(input_grid)
    return _output(info,_cuts(info["cube"]))


def _features(info):
    cube = info["cube"]
    cut = _cuts(cube)
    rows,cols = info["rows"],info["cols"]
    result = {"row_only":False,"col_only":False,"cross":False,
              "loose_extra":False,"failed_multiple":False,"diagonal":False}
    for y,x in np.ndindex(cube.shape[-2:]):
        active,carved = cube[:,:,y,x],cut[:,:,y,x]
        rr,cc = active.all(axis=1),active.all(axis=0)
        result["row_only"] |= bool(rr.any() and not cc.any())
        result["col_only"] |= bool(cc.any() and not rr.any())
        result["cross"] |= bool(rr.any() and cc.any())
        result["loose_extra"] |= bool(carved.any() and (active & ~carved).any())
        result["failed_multiple"] |= bool(active.sum() >= 2 and not carved.any())
        # Includes embedded partial diagonals on rectangular boards.
        result["diagonal"] |= bool(active.sum() == min(rows,cols)
            and np.all(active.sum(axis=0) <= 1) and np.all(active.sum(axis=1) <= 1)
            and not carved.any())
    listed = [tag for _,_,tag in info["frames"] if tag in info["colors"]]
    wrong = np.stack([info["patterns"][tag] for tag in listed]).reshape(cube.shape)
    result["order_matters_counterexample"] = not np.array_equal(_cuts(wrong),cut)
    result["omitting_B_changes"] = bool(np.any(cube & ~cut))
    result["omitting_A_changes"] = bool(cut.any())
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    nt,ne = len(puzzle["train"]),len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen,train_infos = set(),[]
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            cube = info["cube"]
            cut = _cuts(cube)
            assert np.array_equal(out,_output(info,cut)), "wrong cut, perimeter, gutter or crop"
            assert cut.any() and np.any(cube & ~cut), "both mechanisms must affect each pair"
            assert np.any(~cut), "carve geometry must matter"
            # Every cut is authorized by its own same-colored stencil, and its
            # support contains a complete line through this particular tile.
            assert np.all(~cut | cube)
            for r,c,u,v in np.argwhere(cut):
                assert cube[r,:,u,v].all() or cube[:,c,u,v].all()
            if split == "train":
                train_infos.append(info)
    features = [_features(i) for i in train_infos]
    for name in ("row_only","col_only","cross","loose_extra","failed_multiple",
                 "diagonal","order_matters_counterexample"):
        assert any(f[name] for f in features), f"training lacks {name} evidence"
    candidates = set(range(8))
    for info in train_infos:
        cuts = _cuts(info["cube"])
        candidates &= {i for i,t in enumerate(_d4(cuts)) if np.array_equal(cuts,t)}
    assert candidates == {0}, "training leaves stencil orientation ambiguous"
    assert {2,3} <= {d for i in train_infos for d in (i["rows"],i["cols"])}, "teach variable line lengths"


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train,"num_train",2,10)
    ne = _count(num_test,"num_test",1,4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = _pick(rng,("longer","finer","smaller"))
    varying_palette = bool(rng.random() < .8)
    fixed_bg,fixed_marker = map(int,rng.choice(10,2,replace=False))
    scatter_rate = float(rng.uniform(.05,.55))
    first_shape = _pick(rng,((2,3),(3,2)))
    second_shape = first_shape[::-1]
    puzzle,seen = {"train":[],"test":[]},set()
    for index in range(nt+ne):
        training = index < nt
        if index == 0:
            shape = first_shape
        elif index == 1:
            shape = second_shape
        elif training:
            shape = _pick(rng,((2,2),(2,3),(3,2)) if regime == "longer"
                          else ((2,3),(3,2)))
        elif regime == "longer":
            shape = _pick(rng,((2,4),(4,2)))
        elif regime == "smaller":
            shape = (2,2)
        else:
            shape = _pick(rng,SHAPES)
        if regime == "finer":
            k = 3 if training else 4
        else:
            k = 3 if index == 0 else (4 if index == 1 else int(rng.integers(2,5)))
        if varying_palette:
            bg,marker = map(int,rng.choice(10,2,replace=False))
        else:
            bg,marker = fixed_bg,fixed_marker
        # Retry only distinctness / didactic color-order counterexamples.
        # Sixteen attempts; no intermediate grids or traces are saved.
        for _ in range(16):
            pair = _render(rng,*shape,k,bg,marker,index < 2,rng.random() < scatter_rate)
            info = _parse(pair["input"])
            if (_key(pair["input"]) not in seen
                    and (index >= 2 or _features(info)["order_matters_counterexample"])):
                break
        else:
            raise RuntimeError("bounded construction retries exhausted")
        seen.add(_key(pair["input"]))
        puzzle["train" if training else "test"].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    puzzle["test"] = [puzzle["test"][int(i)] for i in rng.permutation(ne)]
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """42 supplied seeds, bounded summaries, symmetry and semantic mutations."""
    fingerprints,shapes,sizes,bgs,markers,dimensions = set(),set(),set(),set(),set(),set()
    patterns,supports,distractors,partial_masks = set(),set(),set(),set()
    feature_counts = {}
    mixed_bg = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        repeat = generate_puzzle(seed)
        fingerprint = tuple(_digest(p[k]) for split in ("train","test")
                            for p in puzzle[split] for k in ("input","output"))
        assert fingerprint == tuple(_digest(p[k]) for split in ("train","test")
                                   for p in repeat[split] for k in ("input","output"))
        fingerprints.add(fingerprint)
        local_bgs = set()
        for split in ("train","test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                assert np.array_equal(solve(pair["input"]),pair["output"])
                shapes.add((info["rows"],info["cols"]))
                sizes.add(info["size"])
                bgs.add(info["background"]); local_bgs.add(info["background"])
                markers.add(info["marker"])
                dimensions.add(pair["input"].shape)
                distractors.add(len(info["patterns"])-len(info["colors"]))
                for m in info["patterns"].values():
                    patterns.add((m.shape,m.tobytes()))
                    partial_masks.add(int(m.sum()))
                for u,v in np.ndindex(info["cube"].shape[-2:]):
                    supports.add((info["rows"],info["cols"],info["cube"][:,:,u,v].tobytes()))
                for key,value in _features(info).items():
                    feature_counts[key] = feature_counts.get(key,0)+int(value)
        mixed_bg += len(local_bgs) > 1
    assert len(fingerprints) == 42 and shapes == set(SHAPES)
    assert sizes == {4,5,6} and bgs == markers == set(range(10))
    assert len(dimensions) >= 30 and len(patterns) >= 120 and len(supports) >= 60
    assert {0,1,2,3,4} <= distractors and 0 in partial_masks
    assert 10 <= mixed_bg < 42
    assert all(feature_counts[k] >= 42 for k in feature_counts)
    base = generate_puzzle(0)
    for orientation in range(8):
        transformed = {s:[{k:_d4(v)[orientation].copy() for k,v in p.items()}
                          for p in base[s]] for s in base}
        validate_generated_puzzle(transformed)
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(314159,nt,ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Pixel damage, unsynchronized A-only cuts and duplicate inputs must fail.
    for defect in ("pixel","A_only","duplicate"):
        p = {s:[{k:v.copy() for k,v in pair.items()} for pair in base[s]] for s in base}
        if defect == "pixel":
            p["test"][0]["output"][0,0] = (int(p["test"][0]["output"][0,0])+1)%10
        elif defect == "A_only":
            info = _parse(p["test"][0]["input"])
            p["test"][0]["output"] = _output(info,info["cube"])
        else:
            p["train"][1] = p["train"][0]
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted "+defect)
    return {"seeds":42,"distinct_puzzles":len(fingerprints),
            "board_shapes":sorted(shapes),"tile_sizes":sorted(sizes),
            "input_dimensions":len(dimensions),"distinct_stencils":len(patterns),
            "distinct_supports":len(supports),"distractor_counts":sorted(distractors),
            "varying_background_puzzles":mixed_bg,"features":feature_counts,
            "independent_solver":False,"blind_prediction":False}


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",",":")))
