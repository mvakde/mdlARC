"""Solved ARC puzzles: crop the rectangle with the fewest accent cells."""
import hashlib
import numpy as np

_AUTO_COUNTS = ((2,1),(3,1),(4,1),(5,2),(6,2),(7,2),
                (4,3),(6,3),(8,3),(5,4),(8,4),(10,4))
_SHAPES = tuple((h,w) for h in range(3,11) for w in range(3,11)
                if h*w <= 80)
_SOURCE_INPUT_HASHES = {
    ((12,11),"fe8f2500fcd7a7cfb1cd04390b05039e86275d5141945c089f81167350f8807b"),
    ((22,20),"399ab12aeaab1fc06c7113fb8941e3ad579f21ac451ca2e08931cd3f226807dc"),
    ((22,21),"e1b2b384129b3931afd9ed1083454bb293457b10b26b2e42a12238e309bef0bd"),
    ((22,15),"5a9ba7c6f03af836f2f10fe5695ac9b1f1aef610350cfd876d3e99b7d9efd623"),
    ((21,18),"70966e99e4c6dc0463eaa5a09082c47ded4901c577823c4ccb2634feeab4db0e"),
    ((22,25),"56086892ad7ad9fd6d2956dbc6d7479e5f7cbebd04775e3effc655c6ab9134e7"),
}


def _integer(x):
    return isinstance(x, (int,np.integer)) and not isinstance(x, (bool,np.bool_))


def _counts(rng, seed, nt, ns):
    if nt is not None:
        if not _integer(nt) or not 2 <= int(nt) <= 10:
            raise ValueError("num_train must be 2..10")
        nt = int(nt)
    if ns is not None:
        if not _integer(ns) or not 1 <= int(ns) <= 4:
            raise ValueError("num_test must be 1..4")
        ns = int(ns)
    if nt is None and ns is None:
        nt,ns = _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    elif nt is None:
        nt = int(rng.integers(max(2,ns+1),11))
    elif ns is None:
        ns = int(rng.integers(1,min(4,nt-1)+1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt,ns


def _boxes(grid, bg):
    mask = grid != bg
    seen = np.zeros(grid.shape,dtype=bool)
    H,W = grid.shape
    boxes = []
    for r in range(H):
        for c in range(W):
            if seen[r,c] or not mask[r,c]:
                continue
            seen[r,c] = True
            stack = [(r,c)]
            r0=r1=r
            c0=c1=c
            while stack:
                y,x = stack.pop()
                r0,r1 = min(r0,y),max(r1,y)
                c0,c1 = min(c0,x),max(c1,x)
                for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0 <= yy < H and 0 <= xx < W and mask[yy,xx] and not seen[yy,xx]:
                        seen[yy,xx] = True
                        stack.append((yy,xx))
            boxes.append((r0,r1+1,c0,c1+1))
    return boxes


def _crop(grid,bg,carrier):
    boxes = _boxes(grid,bg)
    counts = [int(np.count_nonzero(grid[r0:r1,c0:c1] != carrier))
              for r0,r1,c0,c1 in boxes]
    if not counts or counts.count(min(counts)) != 1:
        raise RuntimeError("no unique accent minimum")
    r0,r1,c0,c1 = boxes[int(np.argmin(counts))]
    return grid[r0:r1,c0:c1].copy()


def _shape_choices(area,relation,minimum):
    pool = [s for s in _SHAPES if (s[0]*s[1]-1)//2 >= minimum+1]
    if relation == "small":
        return [s for s in pool if s[0]*s[1] < area]
    if relation == "large":
        return [s for s in pool if s[0]*s[1] > area]
    if relation == "equal":
        return [s for s in pool if s[0]*s[1] == area]
    if relation == "double":
        return [s for s in pool if s[0]*s[1] == 2*area]
    return pool


def _shapes(rng,n,minimum,profile,orientation):
    def target_ok(s):
        h,w=s
        area=h*w
        if orientation == "landscape" and not h < w:
            return False
        if orientation == "portrait" and not h > w:
            return False
        if orientation == "square" and not h == w:
            return False
        if profile == "smallest":
            return area <= 32
        if profile == "largest":
            return 25 <= area <= 60
        if profile == "middle":
            return 16 <= area <= 42
        if profile == "tie":
            return 12 <= area <= 30 and bool(_shape_choices(area,"double",minimum))
        return 12 <= area <= 52
    targets=[s for s in _SHAPES if target_ok(s)]
    if not targets:
        return None
    target=targets[int(rng.integers(len(targets)))]
    area=target[0]*target[1]
    shapes=[target]
    counts=[minimum]
    for i in range(1,n):
        relation = ("large" if profile == "smallest" else
                    "small" if profile == "largest" else
                    "small" if profile == "middle" and i == 1 else
                    "large" if profile == "middle" and i == 2 else
                    "equal" if profile == "equal" and i == 1 else
                    "double" if profile == "tie" and i == 1 else "any")
        choices=_shape_choices(area,relation,minimum)
        if profile == "smallest" and i == 1:
            # A larger rival with more accents but lower density makes
            # absolute count, rather than accent fraction, observable.
            choices=[s for s in choices
                     if s[0]*s[1]*minimum > area*(minimum+1)]
        if not choices:
            return None
        shape=choices[int(rng.integers(len(choices)))]
        maximum=min(8,(shape[0]*shape[1]-1)//2)
        k=(minimum+1 if profile == "smallest" and i == 1 else
           2*minimum if relation == "double" else
           int(rng.integers(minimum+1,maximum+1)))
        shapes.append(shape)
        counts.append(k)
    order=rng.permutation(n)
    return ([shapes[int(i)] for i in order],
            [counts[int(i)] for i in order],
            int(np.where(order == 0)[0][0]))


def _apart(rect,placed):
    r,c,h,w=rect
    return all(r+h+1 <= y or y+hh+1 <= r or c+w+1 <= x or x+ww+1 <= c
               for y,x,hh,ww in placed)


def _pack(rng,shapes,regime,shift,train_width):
    n=len(shapes)
    area=sum(h*w for h,w in shapes)
    max_h=max(h for h,w in shapes)
    max_w=max(w for h,w in shapes)
    for _ in range(70):
        base=min(26,12+2*(n-2))
        if regime == "target_test" and shift == "wide":
            lo_h,hi_h=max(14,max_h+2),27
            lo_w,hi_w=max(25,train_width+1,max_w+2),30
        else:
            lo_h,hi_h=max(base,max_h+2),(30 if regime == "target_test" else 28)
            lo_w,hi_w=max(base,max_w+2),(30 if regime == "target_test" else 28)
        if lo_h > hi_h or lo_w > hi_w:
            return None
        H=int(rng.integers(lo_h,hi_h+1))
        W=int(rng.integers(lo_w,hi_w+1))
        if area > .43*H*W:
            continue
        order=sorted(range(n),key=lambda i:shapes[i][0]*shapes[i][1],reverse=True)
        placed=[]
        rectangles=[None]*n
        for i in order:
            h,w=shapes[i]
            coords=[(r,c) for r in range(H-h+1) for c in range(W-w+1)]
            rng.shuffle(coords)
            if rng.random() < .05:
                edge=[p for p in coords if p[0] == 0 or p[1] == 0 or
                      p[0]+h == H or p[1]+w == W]
                coords=edge+coords
            found=None
            for r,c in coords:
                candidate=(r,c,h,w)
                if _apart(candidate,placed):
                    found=candidate
                    break
            if found is None:
                break
            rectangles[i]=found
            placed.append(found)
        if all(x is not None for x in rectangles):
            return H,W,rectangles
    return None


def _make_pair(rng,regime,conventions,plan,shift,train_width):
    n,minimum,profile,orientation,palette,contrast=plan
    bg,carrier=conventions[:2]
    for _ in range(100):
        chosen=_shapes(rng,n,minimum,profile,orientation)
        if chosen is None:
            continue
        shapes,counts,target=chosen
        packed=_pack(rng,shapes,regime,shift,train_width)
        if packed is None:
            continue
        H,W,rectangles=packed
        grid=np.full((H,W),bg,dtype=np.int8)
        slots=[]
        for i,(r,c,h,w) in enumerate(rectangles):
            grid[r:r+h,c:c+w]=carrier
            positions=rng.choice(h*w,size=counts[i],replace=False)
            slots.append([(r+int(p)//w,c+int(p)%w) for p in positions])
        hues=[[palette[int(rng.integers(len(palette)))] for _ in cells] for cells in slots]
        if contrast and len(palette) >= 2 and minimum >= 2:
            hues[target]=[palette[0]]*len(slots[target])
            rival=next(i for i in range(n) if i != target)
            hues[rival]=[palette[0]]+[palette[1]]*(len(slots[rival])-1)
        # Ensure every pair palette is visible; the first training pair teaches
        # the full puzzle palette, even when later pairs use only a subset.
        flat=[(i,j) for i in range(n) for j in range(len(slots[i]))]
        if contrast and len(palette) >= 2 and minimum >= 2:
            reserved=[(target,0),(rival,1)]
            for hue in palette[2:]:
                choices=[p for p in flat if p[0] != target and p not in reserved]
                if choices:
                    i,j=choices.pop(0)
                    hues[i][j]=hue
                    reserved.append((i,j))
        else:
            chosen_slots=rng.choice(len(flat),size=len(palette),replace=False)
            for hue,slot_index in zip(palette,chosen_slots):
                i,j=flat[int(slot_index)]
                hues[i][j]=hue
        for i,cells in enumerate(slots):
            for (r,c),color in zip(cells,hues[i]):
                grid[r,c]=color
        output=_crop(grid,bg,carrier)
        return {"input":grid,"output":output}
    raise RuntimeError("could not construct pair")


def _key(array):
    return array.shape,tuple(int(v) for v in array.flat)


def _analyze_pair(input_grid,output_grid):
    values,frequency=np.unique(input_grid,return_counts=True)
    bi=int(np.argmax(frequency))
    bg=int(values[bi])
    assert int(frequency[bi]) > input_grid.size/2
    boxes=_boxes(input_grid,bg)
    assert len(boxes) >= 2
    records=[]
    carriers=[]
    for box in boxes:
        r0,r1,c0,c1=box
        patch=input_grid[r0:r1,c0:c1]
        assert np.all(patch != bg)
        colors,counts=np.unique(patch,return_counts=True)
        assert np.count_nonzero(counts == counts.max()) == 1
        carrier=int(colors[int(np.argmax(counts))])
        carriers.append(carrier)
        accents=int(np.count_nonzero(patch != carrier))
        assert 1 <= accents < patch.size/2
        records.append((box,patch.size,accents))
    assert len(set(carriers)) == 1
    carrier=carriers[0]
    accents=tuple(sorted(set(int(v) for v in values)-{bg,carrier}))
    assert 1 <= len(accents) <= 3
    nums=[record[2] for record in records]
    assert nums.count(min(nums)) == 1
    selected=int(np.argmin(nums))
    r0,r1,c0,c1=records[selected][0]
    assert np.array_equal(output_grid,input_grid[r0:r1,c0:c1])
    assert output_grid.shape != input_grid.shape
    return {"background":bg,"carrier":carrier,"accents":accents,
            "records":records,"selected":selected,"minimum":min(nums)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert isinstance(puzzle["train"],list) and isinstance(puzzle["test"],list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs,seen_pairs=set(),set()
    analyses={"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            for grid in pair.values():
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype,np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            key=_key(pair["input"])
            both=(key,_key(pair["output"]))
            assert key not in seen_inputs and both not in seen_pairs
            source_hash=hashlib.sha256(np.asarray(pair["input"],dtype=np.int8).tobytes()).hexdigest()
            assert (pair["input"].shape,source_hash) not in _SOURCE_INPUT_HASHES
            seen_inputs.add(key)
            seen_pairs.add(both)
            a=_analyze_pair(pair["input"],pair["output"])
            analyses[split].append(a)
            assert 2 <= len(a["records"]) <= (6 if split == "train" else 9)
            assert 1 <= a["minimum"] <= 3
    all_a=analyses["train"]+analyses["test"]
    assert len({(a["background"],a["carrier"]) for a in all_a}) == 1
    shown=set().union(*(set(a["accents"]) for a in analyses["train"]))
    assert all(set(a["accents"]).issubset(shown) for a in analyses["test"])
    def relation(a,kind):
        values=[r[1] for r in a["records"]]
        selected=values[a["selected"]]
        others=[x for i,x in enumerate(values) if i != a["selected"]]
        return any(x < selected for x in others) if kind == "smaller" else any(x > selected for x in others)
    assert any(relation(a,"smaller") for a in analyses["train"])
    assert any(relation(a,"larger") for a in analyses["train"])
    assert any(any(other[2]*winner[1] < winner[2]*other[1]
                   for i,other in enumerate(a["records"]) if i != a["selected"])
               for a in analyses["train"] for winner in [a["records"][a["selected"]]])
    if len(shown)>1:
        hue_contrast=False
        for pair,a in zip(puzzle["train"],analyses["train"]):
            carrier=a["carrier"]
            patches=[pair["input"][r0:r1,c0:c1]
                     for (r0,r1,c0,c1),_,_ in a["records"]]
            chosen=patches[a["selected"]]
            for i,other in enumerate(patches):
                if i==a["selected"]:
                    continue
                if any(int(np.count_nonzero(chosen==hue)) >
                       int(np.count_nonzero(other==hue)) for hue in shown):
                    hue_contrast=True
        assert hue_contrast


def generate_puzzle(seed: int,num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be nonnegative integer")
    seed=int(seed)
    rng=np.random.default_rng(seed)
    nt,ns=_counts(rng,seed,num_train,num_test)
    colors=[int(x) for x in rng.permutation(10)]
    bg,carrier=colors[:2]
    pool=tuple(colors[2:2+1+seed%3])
    variable=(seed//3)%2 == 1 and len(pool)>1
    shift=("expanded","wide","orientation","matched")[seed%4]
    conventions=(bg,carrier,pool)
    first_profiles=["smallest","largest"]
    rng.shuffle(first_profiles)
    first_orientations=["landscape","portrait"]
    rng.shuffle(first_orientations)
    first_minima=[0,0]
    contrast_index=first_profiles.index("smallest")
    first_minima[contrast_index]=2
    first_minima[1-contrast_index]=(1,3)[int(rng.integers(2))]
    puzzle={"train":[],"test":[]}
    train_object_counts=[]
    for index in range(nt+ns):
        training=index<nt
        regime="source_train" if training else "target_test"
        if training:
            count=int(rng.integers(2,7))
            minimum=first_minima[index] if index<2 else int(rng.integers(1,4))
            profile=first_profiles[index] if index<2 else ("smallest","largest","middle","equal","tie")[int(rng.integers(5))]
            if index<2:
                orientation=first_orientations[index]
            elif shift=="orientation":
                orientation=("landscape","portrait")[int(rng.integers(2))]
            else:
                orientation=("landscape","portrait","square")[int(rng.integers(3))]
        else:
            count=(int(rng.integers(7,10)) if shift=="expanded" else
                   int(rng.integers(2,max(train_object_counts)+1)) if shift=="orientation" else
                   int(rng.integers(2,7)))
            minimum=int(rng.integers(1,4))
            profile=("smallest","largest","middle","equal","tie")[int(rng.integers(5))]
            orientation=("landscape","portrait","square")[int(rng.integers(3))]
            if shift=="orientation":
                orientation="square"
        if count==2 and profile=="middle":
            profile="tie"
        if variable and index != 0:
            k=int(rng.integers(1,len(pool)+1))
            palette=tuple(int(x) for x in rng.choice(pool,size=k,replace=False))
        else:
            palette=pool
        if training and index==contrast_index and len(pool)>=2:
            minimum=2
            palette=pool
        train_width=max((p["input"].shape[1] for p in puzzle["train"]),default=0)
        plan=(count,minimum,profile,orientation,palette,
              training and index==contrast_index and len(pool)>=2)
        pair_rng=np.random.default_rng(int(rng.integers(0,2**63,dtype=np.int64)))
        pair=_make_pair(pair_rng,regime,conventions,plan,shift,train_width)
        puzzle["train" if training else "test"].append(pair)
        if training:
            train_object_counts.append(count)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles=[generate_puzzle(s) for s in range(42)]
    assert len({len(p["train"]) for p in puzzles})>=4
    assert len({len(p["test"]) for p in puzzles})>=3
    shapes,layouts,roles,counts,inputs=set(),set(),set(),set(),set()
    area_roles,density_roles,palette_sizes,edges=set(),set(),set(),set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train","test"):
            for pair in puzzle[split]:
                grid,output=pair["input"],pair["output"]
                a=_analyze_pair(grid,output)
                assert np.array_equal(_crop(grid,a["background"],a["carrier"]),output)
                shapes.add((grid.shape,output.shape))
                layouts.add((grid.shape,tuple(r[0] for r in a["records"])))
                roles.add((a["background"],a["carrier"],a["accents"]))
                counts.add((len(a["records"]),a["minimum"]))
                inputs.add(_key(grid))
                palette_sizes.add(len(a["accents"]))
                areas=[r[1] for r in a["records"]]
                densities=[r[2]/r[1] for r in a["records"]]
                j=a["selected"]
                area_roles.add("smallest" if areas[j]<min(x for i,x in enumerate(areas) if i!=j)
                               else "largest" if areas[j]>max(x for i,x in enumerate(areas) if i!=j)
                               else "other")
                density_roles.add("least" if densities[j]<min(x for i,x in enumerate(densities) if i!=j)
                                  else "most" if densities[j]>max(x for i,x in enumerate(densities) if i!=j)
                                  else "other")
                edges.add(any(r0==0 or c0==0 or r1==grid.shape[0] or c1==grid.shape[1]
                              for (r0,r1,c0,c1),_,_ in a["records"]))
    assert len(shapes)>=30 and len(layouts)>=100 and len(roles)>=25
    assert len(counts)>=12 and len(inputs)>=300
    assert area_roles=={"smallest","largest","other"}
    assert density_roles=={"least","most","other"}
    assert palette_sizes=={1,2,3} and edges=={False,True}
    a,b=generate_puzzle(123456,7,3),generate_puzzle(123456,7,3)
    assert all(np.array_equal(x[k],y[k]) for split in ("train","test")
               for x,y in zip(a[split],b[split]) for k in ("input","output"))
    assert _key(generate_puzzle(111)["train"][0]["input"]) != _key(generate_puzzle(112)["train"][0]["input"])
    for seed,nt,ns in ((800,2,1),(801,10,1),(802,10,4),
                       (900,None,1),(901,None,4),(902,2,None),(903,10,None)):
        p=generate_puzzle(seed,nt,ns)
        assert nt is None or len(p["train"])==nt
        assert ns is None or len(p["test"])==ns
    for kwargs in ({"num_train":1},{"num_train":11},{"num_test":0},
                   {"num_test":5},{"num_train":2,"num_test":2},
                   {"num_train":3,"num_test":4},{"num_train":True}):
        try:
            generate_puzzle(1000,**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
