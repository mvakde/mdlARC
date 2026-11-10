"""Independent cyclic transfers of bar colors and bar lengths."""
import numpy as np

DIRECTIONS = ("up", "down", "right", "left")
SOURCE = ((9, (9,8,6,1), (2,3,8,3)), (7, (2,8,5), (6,3,4)),
          (11, (1,2,5,4,8), (4,5,3,8,6)),
          (11, (0,2,8,9,6), (10,1,4,4,5)))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(seed, nt, ns):
    for value, name, low, high in ((nt,"num_train",2,10),(ns,"num_test",1,4)):
        if value is not None and (not _integer(value) or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if nt is None and ns is None:
        return [(a,b) for a in range(2,11) for b in range(1,min(a,5))][seed % 30]
    if nt is None:
        choices = list(range(max(2,int(ns)+1),11))
        nt = choices[seed % len(choices)]
    if ns is None:
        choices = list(range(1,min(4,int(nt)-1)+1))
        ns = choices[seed % len(choices)]
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return int(nt), int(ns)


def _canonical(grid, direction):
    if direction == "up": return grid
    if direction == "down": return grid[::-1,:]
    if direction == "right": return grid.T
    return grid[:,::-1].T


def _uncanonical(grid, direction):
    if direction == "up": return grid
    if direction == "down": return grid[::-1,:]
    if direction == "right": return grid.T
    return grid.T[:,::-1]


def _draw(shape, direction, bg, colors, lengths, thick, gap, before, inset):
    h,w = shape
    span,cross = (h,w) if direction in ("up","down") else (w,h)
    canvas = np.full((span,cross), bg, dtype=np.int8)
    end = span-inset
    for i,(color,length) in enumerate(zip(colors,lengths)):
        start = before+i*(thick+gap)
        canvas[end-int(length):end,start:start+thick] = int(color)
    return _uncanonical(canvas,direction).copy()


def _key(grid):
    return grid.shape, grid.tobytes()


def _source_keys():
    return {_key(_draw((side,side),"up",7,colors,lengths,1,1,1,0))
            for side,colors,lengths in SOURCE}


def _orientations(rng, nt, ns):
    if rng.random() < .44:
        d = str(rng.choice(DIRECTIONS))
        return [d]*nt, [d]*ns
    train = [str(rng.choice(("up","down"))),
             str(rng.choice(("right","left")))]
    train += [str(rng.choice(DIRECTIONS)) for _ in range(nt-2)]
    train = [train[int(i)] for i in rng.permutation(nt)]
    test = [str(rng.choice(train)) for _ in range(ns)]
    return train,test


def _geometry(rng,n,direction,anchor,source_like):
    if source_like and direction == "up" and n in (3,4,5):
        side=2*n+1
        return (side,side),1,1,1,0,side-1
    feasible = [(t,g) for t in (1,2,3) for g in (1,2,3,4)
                if n*t+(n-1)*g <= 29]
    weights=np.array([5 if t==g==1 else 3 if t==1 or g==1 else 1
                      for t,g in feasible],dtype=float)
    weights/=weights.sum()
    thick,gap=feasible[int(rng.choice(len(feasible),p=weights))]
    occupied=n*thick+(n-1)*gap
    spare=30-occupied
    before=int(rng.integers(0,min(6,spare)+1))
    after=int(rng.integers(0,min(6,spare-before)+1))
    cross=occupied+before+after
    for _ in range(32):
        span=int(rng.integers(max(7,n+2 if anchor else 7),31))
        inset=int(rng.choice((0,0,0,1,1,2,3)))
        far=int(rng.integers(1,5))
        maximum=span-inset-far
        if maximum >= (n if anchor else 3): break
    else:
        span,inset,maximum=max(12,n+2),0,max(11,n+1)
    shape=(span,cross) if direction in ("up","down") else (cross,span)
    return shape,thick,gap,before,inset,maximum


def _lengths(rng,n,maximum,anchor,test):
    if anchor:
        return [int(x) for x in rng.choice(np.arange(1,maximum+1),size=n,replace=False)]
    mode=str(rng.choice(("distinct","random","pair","triple","groups","equal"),
                        p=(.18,.25,.21,.14,.12,.10)))
    if mode=="distinct" and maximum>=n:
        result=[int(x) for x in rng.choice(np.arange(1,maximum+1),size=n,replace=False)]
    else:
        result=[int(x) for x in rng.integers(1,maximum+1,size=n)]
        if mode=="pair":
            i,j=(int(x) for x in rng.choice(n,size=2,replace=False))
            result[j]=result[i]
        elif mode=="triple":
            idx=[int(x) for x in rng.choice(n,size=3,replace=False)]
            for j in idx[1:]: result[j]=result[idx[0]]
        elif mode=="groups":
            values=[int(x) for x in rng.integers(1,maximum+1,size=2)]
            result=[values[int(x)] for x in rng.integers(0,2,size=n)]
        elif mode=="equal": result=[result[0]]*n
    if test and mode not in ("equal","groups") and rng.random()<.48:
        result[int(rng.integers(n))]=int(rng.choice((1,maximum)))
    return result


def _colors(rng,n,bg,anchor,seen,test):
    available=[c for c in range(10) if c!=bg]
    if anchor:
        return [int(x) for x in rng.choice(available,size=n,replace=False)]
    if rng.random()<.38:
        result=[int(x) for x in rng.choice(available,size=n,replace=False)]
    else:
        size=int(rng.integers(1,min(9,n+2)+1))
        pool=[int(x) for x in rng.choice(available,size=size,replace=False)]
        result=[int(x) for x in rng.choice(pool,size=n)]
    if test and rng.random()<.53:
        novel=[c for c in available if c not in seen and c not in result]
        if novel: result[int(rng.integers(n))]=int(rng.choice(novel))
    return result


def _make_pair(rng,direction,n,anchor,test,seen,source_like):
    shape,thick,gap,before,inset,maximum=_geometry(rng,n,direction,anchor,source_like)
    bg=int(rng.integers(10))
    colors=_colors(rng,n,bg,anchor,seen,test)
    lengths=_lengths(rng,n,maximum,anchor,test)
    if len(set(colors))==len(set(lengths))==1:
        lengths[0]=1 if lengths[0]!=1 else maximum
    input_grid=_draw(shape,direction,bg,colors,lengths,thick,gap,before,inset)
    output_grid=_draw(shape,direction,bg,np.roll(colors,1),np.roll(lengths,-1),
                      thick,gap,before,inset)
    return {"input":input_grid,"output":output_grid},bg,colors


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _integer(seed) or seed<0: raise ValueError("seed must be a non-negative integer")
    seed=int(seed)
    nt,ns=_counts(seed,num_train,num_test)
    rng=np.random.default_rng(seed)
    train_dirs,test_dirs=_orientations(rng,nt,ns)
    puzzle={"train":[],"test":[]}
    seen_inputs=set()
    source_inputs=_source_keys()
    train_colors=set()
    anchored=set()
    for split,directions in (("train",train_dirs),("test",test_dirs)):
        for direction in directions:
            anchor=split=="train" and direction not in anchored
            if anchor: anchored.add(direction)
            for _ in range(128):
                n=int(rng.integers(3,10))
                if anchor and rng.random()<.60: n=int(rng.integers(3,6))
                source_like=direction=="up" and rng.random()<.22
                pair,bg,colors=_make_pair(rng,direction,n,anchor,split=="test",
                                          train_colors,source_like)
                key=_key(pair["input"])
                if (key not in seen_inputs and key not in source_inputs
                    and not np.array_equal(pair["input"],pair["output"])): break
            else: raise RuntimeError("could not construct a fresh bar pair")
            seen_inputs.add(key)
            puzzle[split].append(pair)
            if split=="train": train_colors.update(colors)
    validate_generated_puzzle(puzzle)
    return puzzle


def _parse_candidates(grid):
    candidates=[]
    for direction in DIRECTIONS:
        canvas=_canonical(grid,direction)
        span,cross=canvas.shape
        for bg in range(10):
            active=np.any(canvas!=bg,axis=0)
            starts=np.flatnonzero(active & ~np.r_[False,active[:-1]])
            ends=np.flatnonzero(active & ~np.r_[active[1:],False])+1
            n=len(starts)
            if not 3<=n<=9: continue
            widths=ends-starts
            if len(set(int(x) for x in widths))!=1: continue
            gaps=starts[1:]-ends[:-1]
            if len(set(int(x) for x in gaps))!=1 or not 1<=int(gaps[0])<=4: continue
            thick=int(widths[0])
            if not 1<=thick<=3 or starts[0]>6 or cross-ends[-1]>6: continue
            colors=[];lengths=[];common_end=None;valid=True
            for start,end in zip(starts,ends):
                block=canvas[:,start:end]
                rows=np.flatnonzero(np.any(block!=bg,axis=1))
                if len(rows)==0: valid=False;break
                first,last=int(rows[0]),int(rows[-1])+1
                color=int(block[first,0])
                if (color==bg or not np.all(block[first:last,:]==color)
                    or not np.all(block[:first,:]==bg)
                    or not np.all(block[last:,:]==bg)
                    or (common_end is not None and common_end!=last)):
                    valid=False;break
                common_end=last
                colors.append(color);lengths.append(last-first)
            if not valid or common_end is None: continue
            inset=span-common_end
            if not 0<=inset<=3 or max(lengths)>common_end-1: continue
            candidates.append({"direction":direction,"background":bg,"colors":tuple(colors),
                               "lengths":tuple(lengths),"count":n,"thickness":thick,
                               "gap":int(gaps[0]),"before":int(starts[0]),
                               "inset":inset,"shape":grid.shape})
    return candidates


def _pair_matches(pair):
    matches=[]
    for info in _parse_candidates(pair["input"]):
        prediction=_draw(info["shape"],info["direction"],info["background"],
                         np.roll(info["colors"],1),np.roll(info["lengths"],-1),
                         info["thickness"],info["gap"],info["before"],info["inset"])
        if np.array_equal(prediction,pair["output"]): matches.append(info)
    return matches


def _observable(info):
    c=np.asarray(info["colors"]);l=np.asarray(info["lengths"])
    return (not np.array_equal(np.roll(c,1),np.roll(c,-1))
            and not np.array_equal(np.roll(l,1),np.roll(l,-1))
            and not np.array_equal(c,np.roll(c,1))
            and not np.array_equal(l,np.roll(l,-1)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={"train","test"}
    assert isinstance(puzzle["train"],list) and isinstance(puzzle["test"],list)
    assert 2<=len(puzzle["train"])<=10 and 1<=len(puzzle["test"])<=4
    assert len(puzzle["train"])>len(puzzle["test"])
    source=_source_keys();seen=set();metadata={"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={"input","output"}
            for grid in (pair["input"],pair["output"]):
                assert isinstance(grid,np.ndarray) and grid.ndim==2
                assert all(1<=side<=30 for side in grid.shape)
                assert np.issubdtype(grid.dtype,np.integer)
                assert np.all((grid>=0)&(grid<=9))
            assert pair["input"].shape==pair["output"].shape
            key=_key(pair["input"])
            assert key not in seen and key not in source
            seen.add(key)
            assert not np.array_equal(pair["input"],pair["output"])
            matches=_pair_matches(pair)
            assert matches
            metadata[split].append(matches)
    demonstrated={i["direction"] for possibilities in metadata["train"]
                  for i in possibilities if _observable(i)}
    assert demonstrated
    for possibilities in metadata["test"]:
        assert any(i["direction"] in demonstrated for i in possibilities)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]),_key(p["output"])) for p in puzzle[s])
                 for s in ("train","test"))


def self_test() -> None:
    assert _puzzle_key(generate_puzzle(12345))==_puzzle_key(generate_puzzle(12345))
    assert _puzzle_key(generate_puzzle(987,7,3))==_puzzle_key(generate_puzzle(987,7,3))
    for i,(nt,ns) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),
                                (2,None),(10,None))):
        p=generate_puzzle(20000+i,nt,ns)
        if nt is not None: assert len(p["train"])==nt
        if ns is not None: assert len(p["test"])==ns
    for i,(nt,ns) in enumerate(((1,None),(11,None),(None,0),(None,5),
                                (2,2),(2,4),(4,4))):
        try: generate_puzzle(21000+i,nt,ns)
        except ValueError: pass
        else: raise AssertionError("illegal counts accepted")
    train_counts=set();test_counts=set();keys=set();shapes=set();counts=set()
    directions=set();backgrounds=set();thicknesses=set();gaps=set();length_types=set()
    measured={name:0 for name in (
        "train_one","test_no_extremes","test_novel","test_familiar",
        "four_repeat","triple","multi_groups","all_equal","repeated_color",
        "source_layout","thick","inset","large_rect","combined",
        "mixed_orientation","mixed_background","mixed_count","mixed_spacing")}
    for seed in range(42):
        p=generate_puzzle(seed);validate_generated_puzzle(p)
        train_counts.add(len(p["train"]));test_counts.add(len(p["test"]));keys.add(_puzzle_key(p))
        infos={s:[_pair_matches(pair)[0] for pair in p[s]] for s in ("train","test")}
        together=infos["train"]+infos["test"]
        for name,feature in (("mixed_orientation","direction"),("mixed_background","background"),
                             ("mixed_count","count"),("mixed_spacing","gap")):
            measured[name]+=len({i[feature] for i in together})>1
        train_palette={color for i in infos["train"] for color in i["colors"]}
        for split in ("train","test"):
            for pair,info in zip(p[split],infos[split]):
                shapes.add(pair["input"].shape);counts.add(info["count"])
                directions.add(info["direction"]);backgrounds.add(info["background"])
                thicknesses.add(info["thickness"]);gaps.add(info["gap"])
                lengths=info["lengths"]
                length_types.add((len(set(lengths)),len(lengths)))
                h,w=info["shape"]
                span=h if info["direction"] in ("up","down") else w
                maximum=span-info["inset"]-1
                measured["train_one"]+=split=="train" and 1 in lengths
                measured["test_no_extremes"]+=split=="test" and 1 not in lengths and maximum not in lengths
                measured["test_novel"]+=split=="test" and bool(set(info["colors"])-train_palette)
                measured["test_familiar"]+=split=="test" and not (set(info["colors"])-train_palette)
                measured["four_repeat"]+=info["count"]==4 and len(set(lengths))<4
                measured["triple"]+=max(lengths.count(x) for x in set(lengths))>=3
                measured["multi_groups"]+=sum(lengths.count(x)>=2 for x in set(lengths))>=2
                measured["all_equal"]+=len(set(lengths))==1
                measured["repeated_color"]+=len(set(info["colors"]))<info["count"]
                measured["source_layout"]+=(info["direction"]=="up" and h==w==2*info["count"]+1
                    and info["thickness"]==info["gap"]==info["before"]==1 and info["inset"]==0)
                measured["thick"]+=info["thickness"]>1
                measured["inset"]+=info["inset"]>0
                measured["large_rect"]+=h!=w and max(h,w)>=24
                measured["combined"]+=(info["thickness"]>1 and info["inset"]>0 and h!=w
                    and (len(set(lengths))<len(lengths) or len(set(info["colors"]))<info["count"]))
    assert len(train_counts)>=4 and len(test_counts)>=3 and len(keys)==42
    assert len(shapes)>=25 and counts==set(range(3,10)) and directions==set(DIRECTIONS)
    assert len(backgrounds)==10 and len(thicknesses)>=2 and len(gaps)>=2
    assert any(u<n for u,n in length_types) and any(u==n for u,n in length_types)
    for name,minimum in (("train_one",5),("test_no_extremes",5),("test_novel",3),
                         ("test_familiar",5),("four_repeat",3),("triple",5),
                         ("multi_groups",3),("all_equal",3),("repeated_color",5),
                         ("source_layout",2),("thick",5),("inset",5),
                         ("large_rect",5),("combined",3),("mixed_orientation",5),
                         ("mixed_background",10),("mixed_count",10),("mixed_spacing",10)):
        assert measured[name]>=minimum, (name,measured[name])
