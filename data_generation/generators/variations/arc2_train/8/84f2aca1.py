import numpy as np


_SOURCE_INPUT_ROWS = {
    ("000000", "444000", "404000", "404000", "444000", "000000"),
    ("000000", "088800", "080800", "088800", "000000", "000000"),
    (
        "000000000000",
        "044440000000",
        "040040033300",
        "044440030300",
        "000000030300",
        "000000033300",
        "000000000000",
        "000000000000",
        "000000000000",
        "000333002222",
        "000303002002",
        "000333002222",
        "000000000000",
        "000002220000",
        "000002020000",
        "000002220000",
    ),
    (
        "0000000000000",
        "0444000222000",
        "0404000202000",
        "0444000222000",
        "0000000000000",
        "0003333000000",
        "0003003000000",
        "0003333000000",
        "0000000000000",
        "0000000333000",
        "8888000303000",
        "8008000333000",
        "8888000000000",
    ),
    (
        "000000000000",
        "044400000000",
        "040400222200",
        "044400200200",
        "000000222200",
        "000000000000",
        "000000000000",
        "333022200000",
        "303020200000",
        "303022200333",
        "333000000303",
        "000000000333",
        "000000000000",
    ),
}

_KIND_SHAPES = {
    "S": (3, 3),
    "H": (3, 4),
    "V": (4, 3),
}


def _resolve_counts(rng, num_train, num_test):
    for name, val, lo, hi in (("num_train", num_train, 2, 10),
                              ("num_test", num_test, 1, 4)):
        if val is not None and (isinstance(val, (bool, np.bool_)) or
                                not isinstance(val, (int, np.integer)) or
                                not lo <= int(val) <= hi):
            raise ValueError(name)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _signature(grid):
    return grid.shape, grid.tobytes()


def _rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _parse_for_bg(grid, bg):
    h, w = grid.shape
    objects = []
    for color in (int(x) for x in np.unique(grid) if int(x) != bg):
        seen = np.zeros(grid.shape, bool)
        for sr in range(h):
            for sc in range(w):
                if seen[sr, sc] or int(grid[sr, sc]) != color:
                    continue
                stack, cells = [(sr, sc)], set()
                seen[sr, sc] = True
                while stack:
                    r, c = stack.pop()
                    cells.add((r, c))
                    for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                        if (0 <= rr < h and 0 <= cc < w and not seen[rr,cc]
                                and int(grid[rr,cc]) == color):
                            seen[rr,cc] = True
                            stack.append((rr,cc))
                r0 = min(r for r,c in cells)
                r1 = max(r for r,c in cells)
                c0 = min(c for r,c in cells)
                c1 = max(c for r,c in cells)
                bh, bw = r1-r0+1, c1-c0+1
                if (bh,bw) not in _KIND_SHAPES.values():
                    return None
                border = {(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)
                          if r in (r0,r1) or c in (c0,c1)}
                if cells != border:
                    return None
                cavity = tuple((r,c) for r in range(r0+1,r1) for c in range(c0+1,c1))
                if any(int(grid[r,c]) != bg for r,c in cavity):
                    return None
                kind = "S" if len(cavity) == 1 else ("H" if bh == 3 else "V")
                objects.append((kind, color, (r0,c0,bh,bw), cavity))
    return sorted(objects, key=lambda o:o[2][:2]) if objects else None


def _parse(grid):
    parses = [(bg, obj) for bg in (int(x) for x in np.unique(grid))
              if (obj := _parse_for_bg(grid, bg)) is not None]
    assert len(parses) == 1, "ambiguous frames/background"
    return parses[0]


def _compatible(a, ca, b, cb):
    r,c,h,w = a
    rr,cc,hh,ww = b
    if not (r+h <= rr or rr+hh <= r or c+w <= cc or cc+ww <= c):
        return False
    return ca != cb or (r+h < rr or rr+hh < r or c+w < cc or cc+ww < c)


def _sample_kinds(rng, count):
    modes = (("S",),("H",),("V",),("S","H"),("S","V"),("H","V"),("S","H","V"))
    pool = modes[int(rng.integers(len(modes)))]
    kinds = list(pool) if count >= len(pool) else []
    kinds += [str(rng.choice(pool if kinds else ("S","H","V")))
              for _ in range(count-len(kinds))]
    rng.shuffle(kinds)
    return kinds


def _sample_count(rng, regime):
    weights = ([4,3,3,3,3,2,1,1] if regime == "source_train"
               else [2,3,3,3,4,3,2,1])
    return int(rng.choice(np.arange(1,9), p=np.array(weights)/sum(weights)))


def _line(rng, kinds, colors, horizontal):
    boxes = []
    cursor = 0
    for i,(kind,color) in enumerate(zip(kinds, colors)):
        bh,bw = _KIND_SHAPES[kind]
        if i:
            cursor += int(rng.integers(1 if color == colors[i-1] else 0, 3))
        offset = int(rng.integers(0,4))
        boxes.append((offset,cursor,bh,bw) if horizontal
                     else (cursor,offset,bh,bw))
        cursor += bw if horizontal else bh
    height = max(r+h for r,c,h,w in boxes)
    width = max(c+w for r,c,h,w in boxes)
    if height > 27 or width > 27:
        return None
    top,left,bottom,right = (int(rng.integers(0,3)) for _ in range(4))
    height += top+bottom
    width += left+right
    if height > 30 or width > 30:
        return None
    boxes = [(r+top,c+left,h,w) for r,c,h,w in boxes]
    if any(not _compatible(boxes[i],colors[i],boxes[j],colors[j])
           for i in range(len(boxes)) for j in range(i+1,len(boxes))):
        return None
    return height,width,boxes


def _canvas(rng, mode, count):
    if mode == "compact":
        low, high = (5,12) if count == 1 else (8,17)
        return int(rng.integers(low,high+1)), int(rng.integers(low,high+1))
    if mode == "medium":
        return int(rng.integers(10,24)),int(rng.integers(10,24))
    if mode == "spacious":
        return int(rng.integers(18,31)),int(rng.integers(18,31))
    if mode == "wide":
        return int(rng.integers(5,15)),int(rng.integers(17,31))
    return int(rng.integers(17,31)),int(rng.integers(5,15))


def _scatter(rng, kinds, colors, mode, boundary, near):
    order = list(rng.permutation(len(kinds)))
    for attempt in range(65):
        h,w = _canvas(rng, mode if attempt < 45 else "spacious", len(kinds))
        boxes = [None]*len(kinds)
        for step,i in enumerate(order):
            bh,bw = _KIND_SHAPES[kinds[i]]
            positions = [(r,c,bh,bw) for r in range(h-bh+1)
                         for c in range(w-bw+1)]
            if boundary == "none":
                positions = [b for b in positions if
                             0 < b[0] < h-bh and 0 < b[1] < w-bw]
            elif step == 0 and boundary in ("edge","corner"):
                positions = [b for b in positions if
                             (b[0] in (0,h-bh) or b[1] in (0,w-bw)) and
                             (boundary != "corner" or
                              (b[0] in (0,h-bh) and b[1] in (0,w-bw)))]
            rng.shuffle(positions)
            valid = [b for b in positions if all(
                _compatible(b,colors[i],old,colors[j])
                for j,old in enumerate(boxes) if old is not None)]
            if not valid:
                break
            if near and step:
                def dist(b):
                    return min(max(0,old[0]-(b[0]+b[2]),b[0]-(old[0]+old[2]))+
                               max(0,old[1]-(b[1]+b[3]),b[1]-(old[1]+old[3]))
                               for old in boxes if old is not None)
                valid.sort(key=dist)
                boxes[i] = valid[int(rng.integers(min(5,len(valid))))]
            else:
                boxes[i] = valid[0]
        if all(b is not None for b in boxes):
            if len(boxes) >= 3:
                span_h = max(b[0]+b[2] for b in boxes)-min(b[0] for b in boxes)
                span_w = max(b[1]+b[3] for b in boxes)-min(b[1] for b in boxes)
                if mode == "spacious" and (span_h < .5*h or span_w < .5*w):
                    continue
                if mode == "wide" and span_w < .6*w:
                    continue
                if mode == "tall" and span_h < .6*h:
                    continue
            return h,w,boxes
    raise AssertionError("layout failed")


def _layout(rng, kinds, colors, regime):
    modes = ("compact","medium","spacious","wide","tall","row","column")
    weights = [3,3,2,2,2,1,1] if regime == "source_train" else [3,2,2,2,2,1,1]
    mode = str(rng.choice(modes,p=np.array(weights)/sum(weights)))
    if mode in ("row","column"):
        lined = _line(rng,kinds,colors,mode == "row")
        if lined is not None:
            return lined
        mode = "wide" if mode == "row" else "tall"
    boundary = str(rng.choice(("none","edge","corner","free"),p=(.30,.25,.12,.33)))
    return _scatter(rng,kinds,colors,mode,boundary,
                    mode in ("compact","medium") and rng.random() < .4)


def _make_pair(rng, kinds, colors, bg, fills, regime):
    for _ in range(25):
        h,w,boxes = _layout(rng,kinds,colors,regime)
        inp = np.full((h,w),bg,dtype=np.int8)
        for (r,c,bh,bw),color in zip(boxes,colors):
            inp[r,c:c+bw] = color
            inp[r+bh-1,c:c+bw] = color
            inp[r:r+bh,c] = color
            inp[r:r+bh,c+bw-1] = color
        try:
            parsed_bg, objects = _parse(inp)
        except AssertionError:
            continue
        if parsed_bg != bg or len(objects) != len(kinds):
            continue
        out = inp.copy()
        for kind,color,box,cavity in objects:
            for r,c in cavity:
                out[r,c] = fills[len(cavity)]
        return {"input":inp,"output":out}
    raise AssertionError("ambiguous pair construction")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain,ntest = _resolve_counts(rng,num_train,num_test)
    perm = [int(x) for x in rng.permutation(10)]
    fills = {1:perm[0],2:perm[1]}
    available = perm[2:]
    prs = [np.random.default_rng(int(x)) for x in
           rng.integers(0,np.iinfo(np.int64).max,size=ntrain+ntest)]
    kinds = [_sample_kinds(pr,_sample_count(pr,"source_train")) for pr in prs[:ntrain]]
    first,second = (0,1) if rng.random() < .5 else (1,0)
    kinds[first][0] = "S"
    kinds[second][0] = str(rng.choice(("H","V")))
    # Both pair orientations appear in demonstrations, including the
    # two-training-pair boundary case.
    missing = {"H","V"} - {kind for row in kinds for kind in row}
    for kind in sorted(missing):
        slot = int(rng.integers(ntrain))
        kinds[slot].append(kind)
    total = sum(map(len,kinds))
    palette_size = int(rng.integers(1,min(5,max(2,total-1))))
    palette = available[:palette_size]
    anchor = palette[0]
    frame_colors = [[int(pr.choice(palette)) for _ in ks]
                    for pr,ks in zip(prs,kinds)]
    slots = {1:[],2:[]}
    for i,ks in enumerate(kinds):
        for j,k in enumerate(ks):
            slots[1 if k == "S" else 2].append((i,j))
    assert slots[1] and slots[2]
    reserved = set()
    for size in (1,2):
        i,j = slots[size][int(rng.integers(len(slots[size])))]
        frame_colors[i][j] = anchor
        reserved.add((i,j))
    other_slots = [(i,j) for i,row in enumerate(frame_colors)
                   for j in range(len(row)) if (i,j) not in reserved]
    rng.shuffle(other_slots)
    for color,(i,j) in zip(palette[1:],other_slots):
        frame_colors[i][j] = color
    common_bg = int(rng.choice([c for c in available if c != anchor]))
    varied_bg = rng.random() < .6
    bgs = [common_bg]
    for i in range(1,ntrain+ntest):
        if not varied_bg:
            bgs.append(common_bg)
        elif i == 1:
            bgs.append(int(rng.choice([c for c in available
                                       if c not in (anchor,common_bg)])))
        else:
            bgs.append(int(rng.choice([c for c in available if c != anchor])))
    puzzle = {"train":[],"test":[]}
    seen = set()
    for i,(pr,ks,cs) in enumerate(zip(prs[:ntrain],kinds,frame_colors)):
        bg = bgs[i]
        cs = [c if c != bg else int(pr.choice([x for x in palette if x != bg]))
              for c in cs]
        for _ in range(25):
            pair = _make_pair(pr,ks,cs,bg,fills,"source_train")
            sig = _signature(pair["input"])
            if sig not in seen and _rows(pair["input"]) not in _SOURCE_INPUT_ROWS:
                seen.add(sig)
                puzzle["train"].append(pair)
                break
        else:
            raise AssertionError("duplicate training pair")
    trained = {o[1] for p in puzzle["train"] for o in _parse(p["input"])[1]}
    for j,pr in enumerate(prs[ntrain:]):
        bg = bgs[ntrain+j]
        ks = _sample_kinds(pr,_sample_count(pr,"target_test"))
        allowed = [c for c in palette if c != bg]
        novel = [c for c in available if c not in trained and c != bg]
        if novel and pr.random() < .35:
            allowed.append(int(pr.choice(novel)))
        if pr.random() < .32:
            cs = [int(pr.choice(allowed))]*len(ks)
        else:
            cs = [int(pr.choice(allowed)) for _ in ks]
        for _ in range(25):
            pair = _make_pair(pr,ks,cs,bg,fills,"target_test")
            sig = _signature(pair["input"])
            if sig not in seen and _rows(pair["input"]) not in _SOURCE_INPUT_ROWS:
                seen.add(sig)
                puzzle["test"].append(pair)
                break
        else:
            raise AssertionError("duplicate test pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert isinstance(puzzle["train"],list) and isinstance(puzzle["test"],list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(),set()
    learned, observed = {1:set(),2:set()},{1:set(),2:set()}
    color_sizes, input_colors = {},set()
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            inp,out = pair["input"],pair["output"]
            for grid in (inp,out):
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype,np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            sig = _signature(inp)
            both = sig,_signature(out)
            assert sig not in seen_inputs and both not in seen_pairs
            seen_inputs.add(sig)
            seen_pairs.add(both)
            assert _rows(inp) not in _SOURCE_INPUT_ROWS
            bg,objects = _parse(inp)
            mask = np.zeros(inp.shape,bool)
            colors_here = {int(x) for x in np.unique(inp)}
            for kind,color,box,cavity in objects:
                values = {int(out[r,c]) for r,c in cavity}
                assert len(values) == 1
                fill = next(iter(values))
                assert fill not in colors_here and color != bg
                for r,c in cavity:
                    mask[r,c] = True
                size = len(cavity)
                observed[size].add(fill)
                if split == "train":
                    learned[size].add(fill)
                    color_sizes.setdefault(color,set()).add(size)
            assert np.array_equal(inp[~mask],out[~mask])
            input_colors.update(colors_here)
    assert all(len(learned[s]) == 1 and learned[s] == observed[s] for s in (1,2))
    fills = {s:next(iter(learned[s])) for s in (1,2)}
    assert fills[1] != fills[2] and all(f not in input_colors for f in fills.values())
    assert any(sizes == {1,2} for sizes in color_sizes.values())
    for split in ("train","test"):
        for pair in puzzle[split]:
            expected = pair["input"].copy()
            for kind,color,box,cavity in _parse(pair["input"])[1]:
                for r,c in cavity:
                    expected[r,c] = fills[len(cavity)]
            assert np.array_equal(expected,pair["output"])


def self_test() -> None:
    a,b = generate_puzzle(314159,7,3),generate_puzzle(314159,7,3)
    assert all(np.array_equal(x[key],y[key]) for split in ("train","test")
               for x,y in zip(a[split],b[split]) for key in ("input","output"))
    c = generate_puzzle(314160,7,3)
    assert any(not np.array_equal(x["input"],y["input"])
               for x,y in zip(a["train"],c["train"]))
    for nt,ns in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        p = generate_puzzle(9137+(nt or 0)+(ns or 0),nt,ns)
        validate_generated_puzzle(p)
        if nt is not None: assert len(p["train"]) == nt
        if ns is not None: assert len(p["test"]) == ns
    for nt,ns in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4)):
        try: generate_puzzle(1,nt,ns)
        except ValueError: pass
        else: raise AssertionError("illegal counts accepted")
    train_counts,test_counts,shapes,layouts,fill_pairs = set(),set(),set(),set(),set()
    counts,bgs,motifs = set(),set(),set()
    mixed_bg = edge_free = edge_touch = novel = mono = compact5 = 0
    wide = tall = zero_role = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        trained = {o[1] for pair in p["train"] for o in _parse(pair["input"])[1]}
        local_bgs, local_fills = set(),set()
        for split in ("train","test"):
            for pair in p[split]:
                inp,out = pair["input"],pair["output"]
                bg,objects = _parse(inp)
                local_bgs.add(bg); bgs.add(bg); shapes.add(inp.shape)
                counts.add(len(objects))
                layouts.add((inp.shape,tuple((o[0],o[2]) for o in objects)))
                wide += inp.shape[1] >= 2*inp.shape[0]
                tall += inp.shape[0] >= 2*inp.shape[1]
                compact5 += len(objects) == 5 and max(inp.shape) <= 14
                for o in objects: local_fills.add(int(out[o[3][0]]))
                if split == "test":
                    motifs.add(frozenset(o[0] for o in objects))
                    free = all(0 < o[2][0] and 0 < o[2][1] and
                               o[2][0]+o[2][2] < inp.shape[0] and
                               o[2][1]+o[2][3] < inp.shape[1] for o in objects)
                    edge_free += free; edge_touch += not free
                    novel += any(o[1] not in trained for o in objects)
                    mono += len(trained) > 1 and len({o[1] for o in objects}) == 1
        mixed_bg += len(local_bgs) > 1
        fill_pairs.add(tuple(sorted(local_fills)))
        zero_role += 0 in local_bgs or 0 in local_fills or any(
            o[1] == 0 for pair in p["train"]+p["test"]
            for o in _parse(pair["input"])[1])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 70 and len(layouts) >= 150 and len(fill_pairs) >= 10
    assert counts >= set(range(1,7)) and len(bgs) >= 8 and mixed_bg >= 12
    assert zero_role >= 15 and edge_free >= 5 and edge_touch >= 5
    assert novel >= 5 and mono >= 5 and len(motifs) >= 5
    assert compact5 >= 1 and wide >= 5 and tall >= 5
