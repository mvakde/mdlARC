"""Corner triangles encoded by singleton tokens."""
import numpy as np

SOURCE_INPUTS = {
    ("040000200","000002000","000020200","000000020","000400000","000000003"),
    ("00000000000","00000000000","00000007000","00001000700","00200000770","00000007007","00030070000","00000000000","00070000000","00002000000","80000000000"),
    ("00000000000000","00000000000006","00000000888000","00000008000000","00000088000600","00000800800000","00000800080000","00000800008000","00000000000800","00000000000080","00000000000008","00000000000000","00000000000000","00000000000000","00030006000006"),
    ("9000000000","0000000000","0000000000","0008000000","0000808000","0000080000","0000800070","0700000000","0000000000","0400800030"),
    ("04070000002","00000000000","20000000000","80400000000","00000000000","00030000000","30300000000","33000000000","03300000000","00000000000","00000000000","00000000000"),
}

def _corners(h, w):
    return {"TL": (0, 0), "TR": (0, w-1), "BL": (h-1, 0), "BR": (h-1, w-1)}

def _triangle(h, w, corner, leg):
    r0, c0 = _corners(h, w)[corner]
    return {(r, c) for r in range(h) for c in range(w)
            if abs(r-r0) + abs(c-c0) < leg}

def _neighbors(p, h, w):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            q = (p[0]+dr, p[1]+dc)
            if (dr or dc) and 0 <= q[0] < h and 0 <= q[1] < w:
                yield q

def _touch(p, q):
    return max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1

def _counts(seed, rng, num_train, num_test):
    for value, lo, hi, name in ((num_train,2,10,"num_train"),(num_test,1,4,"num_test")):
        if value is not None and (isinstance(value,(bool,np.bool_))
            or not isinstance(value,(int,np.integer)) or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    if num_train is None and num_test is None:
        k = abs(int(seed))
        num_train = 2+k%9
        num_test = 1+(k//9)%min(4,num_train-1)
    elif num_train is None:
        num_train = int(rng.integers(max(2,int(num_test)+1),11))
    elif num_test is None:
        num_test = int(rng.integers(1,min(4,int(num_train)-1)+1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)

def _component(rng, h, w, triangle, size, style):
    allowed = {(r,c) for r in range(h) for c in range(w)}
    allowed -= triangle | set(_corners(h,w).values())
    if style == "diagonal":
        directions = [(1,1),(1,-1),(-1,1),(-1,-1)]
        rng.shuffle(directions)
        for dr,dc in directions:
            starts = [(r,c) for r,c in sorted(allowed)
                      if all((r+i*dr,c+i*dc) in allowed for i in range(size))]
            if starts:
                r,c = starts[int(rng.integers(len(starts)))]
                return {(r+i*dr,c+i*dc) for i in range(size)}
        return None
    starts = sorted(allowed)
    if not starts:
        return None
    last = starts[int(rng.integers(len(starts)))]
    group = {last}
    direction = None
    while len(group) < size:
        origins = [last] if style == "path" else sorted(group)
        frontier = sorted({q for p in origins for q in _neighbors(p,h,w)
                           if q in allowed and q not in group})
        if not frontier:
            return None
        if style == "path" and direction is not None and rng.random() < .55:
            straight = (last[0]+direction[0],last[1]+direction[1])
            nxt = straight if straight in frontier else frontier[int(rng.integers(len(frontier)))]
        else:
            nxt = frontier[int(rng.integers(len(frontier)))]
        direction = (nxt[0]-last[0],nxt[1]-last[1])
        group.add(nxt)
        last = nxt
    return group

def _decode(inp):
    h,w = inp.shape
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    assert int(np.max(counts)) > h*w//2
    foreground = {tuple(map(int,p)) for p in np.argwhere(inp != background)}
    unseen = set(foreground)
    groups = []
    while unseen:
        start = min(unseen)
        color = int(inp[start])
        group,stack = {start},[start]
        unseen.remove(start)
        while stack:
            for q in _neighbors(stack.pop(),h,w):
                if q in unseen and int(inp[q]) == color:
                    unseen.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    large = [g for g in groups if len(g)>1]
    assert len(large)==1
    component = large[0]
    tokens = foreground-component
    assert len(tokens)==len(groups)-1
    hits = [(name,p) for name,p in _corners(h,w).items() if p in tokens]
    assert len(hits)==1
    corner,locator = hits[0]
    counted = tokens-{locator}
    leg = len(counted)
    assert 2 <= leg <= min(h,w)
    triangle = _triangle(h,w,corner,leg)
    assert component.isdisjoint(triangle)
    frequency = {}
    for p in counted:
        color = int(inp[p])
        frequency[color] = frequency.get(color,0)+1
    repeats = [color for color,n in frequency.items() if n>1]
    assert len(repeats)==1
    fill = repeats[0]
    out = np.full_like(inp,background)
    for p in component:
        out[p] = inp[p]
    for p in triangle:
        out[p] = fill
    locator_color = int(inp[locator])
    return out, {
        "background":background, "corner":corner, "leg":leg, "fill":fill,
        "repeat":frequency[fill], "locator_color":locator_color,
        "tie":locator_color != fill and frequency.get(locator_color,0)==1,
        "component":component, "component_color":int(inp[min(component)]),
        "counted":counted, "inside":len(counted & triangle), "frequency":frequency,
    }

def _pair(rng, spec, parameters):
    corner,leg,mode = spec
    for _ in range(240):
        source_sizes = [(6,9),(11,11),(15,14),(10,10),(12,11)]
        eligible_sizes = [shape for shape in source_sizes if min(shape)>leg]
        if eligible_sizes and rng.random()<.18:
            h,w = eligible_sizes[int(rng.integers(len(eligible_sizes)))]
        else:
            high = 14 if rng.random()<.36 else 24
            h = int(rng.integers(max(6,leg+1),high))
            w = int(rng.integers(max(6,leg+1),high))
        base = parameters["base_background"]
        background = base if rng.random()<.62 else int(rng.choice([x for x in range(10) if x!=base]))
        palette = [int(x) for x in parameters["color_order"] if x!=background]
        triangle = _triangle(h,w,corner,leg)
        style = str(rng.choice(["diagonal","path","branch","blob"],p=[.23,.39,.18,.20]))
        upper = min(h,w,18) if style=="diagonal" else min(22,max(3,h*w//5))
        if upper<3:
            continue
        size = int(rng.integers(15,upper+1)) if style!="diagonal" and upper>=15 and rng.random()<.27 else int(rng.integers(3,min(upper,12)+1))
        component = _component(rng,h,w,triangle,size,style)
        if component is None:
            continue
        repeat = 2 if mode=="tie" else int(rng.integers(2,min(4,leg)+1))
        singles_count = leg-repeat
        if mode=="tie" and singles_count<1:
            continue
        fill = int(rng.choice(palette))
        others = [x for x in palette if x!=fill]
        rng.shuffle(others)
        needed = singles_count+(mode=="unique")
        if len(others)<needed:
            continue
        singles = others[:singles_count]
        counted_colors = [fill]*repeat+singles
        locator_color = fill if mode=="equal" else int(rng.choice(singles)) if mode=="tie" else others[singles_count]
        component_color = int(rng.choice(counted_colors)) if rng.random()<.44 else int(rng.choice(palette))
        locator = _corners(h,w)[corner]
        if component_color==locator_color and any(_touch(locator,p) for p in component):
            continue
        placed = [(locator,locator_color)]
        available = {(r,c) for r in range(h) for c in range(w)}
        available -= component | set(_corners(h,w).values())
        rng.shuffle(counted_colors)
        inside_target = int(rng.integers(1,min(3,leg)+1)) if rng.random()<.40 else 0
        for color in counted_colors:
            candidates = [p for p in sorted(available)
                          if (color!=component_color or all(not _touch(p,q) for q in component))
                          and all(color!=old_color or not _touch(p,old_p) for old_p,old_color in placed)]
            if not candidates:
                break
            inside = [p for p in candidates if p in triangle]
            outside = [p for p in candidates if p not in triangle]
            if inside_target and inside:
                choices=inside
                inside_target-=1
            else:
                choices=outside if outside else inside
            p=choices[int(rng.integers(len(choices)))]
            placed.append((p,color))
            available.remove(p)
        else:
            inp=np.full((h,w),background,dtype=np.int8)
            for p in component:
                inp[p]=component_color
            for p,color in placed:
                inp[p]=color
            try:
                out,info=_decode(inp)
            except AssertionError:
                continue
            found_mode = "equal" if info["locator_color"]==info["fill"] else "tie" if info["tie"] else "unique"
            if info["corner"]==corner and info["leg"]==leg and info["fill"]==fill and found_mode==mode:
                return inp,out
    raise RuntimeError("Could not construct a valid pair")

def _specs(rng, train_count, test_count):
    names=["TL","TR","BL","BR"]
    first_corners=list(map(str,rng.choice(names,size=2,replace=False)))
    first_legs=list(map(int,rng.choice(np.arange(3,9),size=2,replace=False)))
    train=[(first_corners[0],first_legs[0],"tie"),(first_corners[1],first_legs[1],"equal")]
    for _ in range(train_count-2):
        mode=str(rng.choice(["tie","equal","unique"]))
        leg=int(rng.integers(3 if mode=="tie" else 2,9))
        train.append((str(rng.choice(names)),leg,mode))
    rng.shuffle(train)
    shown_corners={x[0] for x in train}
    shown_legs={x[1] for x in train}
    test=[]
    for _ in range(test_count):
        if rng.random()<.38:
            corner="TR"
        elif rng.random()<.48 and len(shown_corners)<4:
            corner=str(rng.choice(sorted(set(names)-shown_corners)))
        else:
            corner=str(rng.choice(sorted(shown_corners)))
        novel=sorted(set(range(2,9))-shown_legs)
        legs=novel if novel and rng.random()<.55 else list(range(2,9))
        mode=str(rng.choice(["tie","equal","unique"]))
        legal=[x for x in legs if x>=3] if mode=="tie" else legs
        leg=int(rng.choice(legal if legal else list(range(3,9))))
        test.append((corner,leg,mode))
    return train,test

def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng=np.random.default_rng(seed)
    train_count,test_count=_counts(seed,rng,num_train,num_test)
    parameters={"base_background":int(rng.integers(10)),
                "color_order":tuple(map(int,rng.permutation(10)))}
    train_specs,test_specs=_specs(rng,train_count,test_count)
    puzzle={"train":[],"test":[]}
    seen=set()
    for key,specs in (("train",train_specs),("test",test_specs)):
        for spec in specs:
            for _ in range(12):
                inp,out=_pair(rng,spec,parameters)
                signature=(inp.shape,inp.tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[key].append({"input":inp,"output":out})
                    break
            else:
                raise RuntimeError("Could not construct unique inputs")
    validate_generated_puzzle(puzzle)
    return puzzle

def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={"train","test"}
    assert isinstance(puzzle["train"],list) and isinstance(puzzle["test"],list)
    nt,ns=len(puzzle["train"]),len(puzzle["test"])
    assert 2<=nt<=10 and 1<=ns<=4 and nt>ns
    seen_inputs,seen_pairs=set(),set()
    metadata={"train":[],"test":[]}
    for key in ("train","test"):
        for pair in puzzle[key]:
            assert isinstance(pair,dict) and set(pair)=={"input","output"}
            inp,out=pair["input"],pair["output"]
            for grid in (inp,out):
                assert isinstance(grid,np.ndarray) and grid.ndim==2
                assert 1<=grid.shape[0]<=30 and 1<=grid.shape[1]<=30
                assert np.issubdtype(grid.dtype,np.integer)
                assert np.all((grid>=0)&(grid<=9))
            assert inp.shape==out.shape and not np.array_equal(inp,out)
            rows=tuple("".join(map(str,row)) for row in inp)
            assert rows not in SOURCE_INPUTS
            signature=(inp.shape,inp.astype(np.int8).tobytes())
            pair_signature=(signature,out.astype(np.int8).tobytes())
            assert signature not in seen_inputs and pair_signature not in seen_pairs
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            predicted,info=_decode(inp)
            assert np.array_equal(predicted,out)
            metadata[key].append(info)
    train=metadata["train"]
    assert len({x["corner"] for x in train})>=2
    assert len({x["leg"] for x in train})>=2
    assert any(x["tie"] for x in train)
    assert any(x["locator_color"]==x["fill"] for x in train)
    for info in metadata["test"]:
        assert info["corner"] in {"TL","TR","BL","BR"}
        assert 2<=info["leg"]<=8

def _same_puzzle(a,b):
    return all(len(a[key])==len(b[key]) and
               all(np.array_equal(p[field],q[field])
                   for p,q in zip(a[key],b[key]) for field in ("input","output"))
               for key in ("train","test"))

def self_test() -> None:
    train_counts,test_counts=set(),set()
    shapes,sizes,patterns,colors=set(),set(),set(),set()
    backgrounds,corners,legs,modes,repeats=set(),set(),set(),set(),set()
    test_corners,test_legs,test_modes,test_repeats,palette_sizes=set(),set(),set(),set(),set()
    inside=diagonal=component_token=source_shape_hits=0
    different_backgrounds=unseen_corner=novel_leg=zero_visible=combined=0
    test_inside=test_diagonal=test_component_token=0
    source_dimensions={(6,9),(11,11),(15,14),(10,10),(12,11)}
    signatures=set()
    for seed in range(42):
        puzzle=generate_puzzle(seed)
        assert _same_puzzle(puzzle,generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info=[_decode(p["input"])[1] for p in puzzle["train"]]
        test_info=[_decode(p["input"])[1] for p in puzzle["test"]]
        shown_corners={x["corner"] for x in train_info}
        shown_legs={x["leg"] for x in train_info}
        different_backgrounds+=len({x["background"] for x in train_info+test_info})>1
        unseen_corner+=any(x["corner"] not in shown_corners for x in test_info)
        novel_leg+=any(x["leg"] not in shown_legs for x in test_info)
        for key in ("train","test"):
            for pair in puzzle[key]:
                inp,out=pair["input"],pair["output"]
                predicted,info=_decode(inp)
                assert np.array_equal(predicted,out)
                shapes.add(inp.shape)
                source_shape_hits+=inp.shape in source_dimensions
                backgrounds.add(info["background"])
                corners.add(info["corner"])
                legs.add(info["leg"])
                repeats.add(info["repeat"])
                mode="equal" if info["locator_color"]==info["fill"] else "tie" if info["tie"] else "unique"
                modes.add(mode)
                palette_sizes.add(len(np.unique(inp)))
                if key=="test":
                    test_corners.add(info["corner"])
                    test_legs.add(info["leg"])
                    test_modes.add(mode)
                    test_repeats.add(info["repeat"])
                sizes.add(len(info["component"]))
                colors.update(int(x) for x in np.unique(inp))
                signatures.add((inp.shape,inp.tobytes()))
                group=info["component"]
                r0,c0=min(r for r,_ in group),min(c for _,c in group)
                patterns.add(tuple(sorted((r-r0,c-c0) for r,c in group)))
                inside+=info["inside"]>0
                component_token+=info["component_color"] in info["frequency"]
                orthogonal=any(abs(a[0]-b[0])+abs(a[1]-b[1])==1
                               for a in group for b in group if a!=b)
                diagonal+=not orthogonal
                zero_visible+=info["background"]!=0 and np.any(inp==0)
                combined+=info["background"]!=0 and info["inside"]>0 and info["component_color"] in info["frequency"]
                if key=="test":
                    test_inside+=info["inside"]>0
                    test_diagonal+=not orthogonal
                    test_component_token+=info["component_color"] in info["frequency"]
    assert len(train_counts)>=4 and len(test_counts)>=3
    assert len(shapes)>=45 and len(patterns)>=40 and len(signatures)>=120
    assert max(sizes)>=15 and min(sizes)<=5
    assert backgrounds==set(range(10)) and colors==set(range(10))
    assert corners=={"TL","TR","BL","BR"} and legs==set(range(2,9))
    assert modes=={"tie","equal","unique"} and max(repeats)>=3
    assert inside>=10 and diagonal>=10 and component_token>=10
    assert source_dimensions<=shapes and source_shape_hits>=20
    assert test_corners==corners and test_legs==legs
    assert test_modes==modes and max(test_repeats)>=3
    assert palette_sizes==set(range(2,10))
    assert different_backgrounds>=20 and unseen_corner>=5 and novel_leg>=15
    assert zero_visible>=20 and combined>=10
    assert test_inside>=5 and test_diagonal>=5 and test_component_token>=5
    for i,(nt,ns) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        puzzle=generate_puzzle(1000+i,nt,ns)
        assert _same_puzzle(puzzle,generate_puzzle(1000+i,nt,ns))
        if nt is not None:
            assert len(puzzle["train"])==nt
        if ns is not None:
            assert len(puzzle["test"])==ns
    for nt,ns in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4),(10,10)):
        try:
            generate_puzzle(123,nt,ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {nt}, {ns}")
