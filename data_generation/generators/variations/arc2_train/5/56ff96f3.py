import numpy as np


_SOURCE_PAIR_SIGNATURES = {
    (
        (
            "0000000000",
            "0000000000",
            "0400000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000040000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000000000",
            "0444440000",
            "0444440000",
            "0444440000",
            "0444440000",
            "0444440000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "00000000",
            "03000000",
            "00000000",
            "00000007",
            "00307000",
            "00000000",
            "00000000",
        ),
        (
            "00000000",
            "03300000",
            "03300000",
            "03307777",
            "03307777",
            "00000000",
            "00000000",
        ),
    ),
    (
        (
            "00000000000",
            "07000000000",
            "00000000000",
            "00000007000",
            "00000000000",
            "00000000000",
        ),
        (
            "00000000000",
            "07777777000",
            "07777777000",
            "07777777000",
            "00000000000",
            "00000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0000001000",
            "0000000000",
            "0100000000",
            "0000000000",
            "0002000000",
            "0000000000",
            "0000000200",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0111111000",
            "0111111000",
            "0111111000",
            "0000000000",
            "0002222200",
            "0002222200",
            "0002222200",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "80000000",
            "00800000",
            "00000000",
            "00000000",
            "00000600",
            "00000000",
            "00000000",
            "06000000",
            "00000000",
        ),
        (
            "88800000",
            "88800000",
            "00000000",
            "00000000",
            "06666600",
            "06666600",
            "06666600",
            "06666600",
            "00000000",
        ),
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _array_signature(grid: np.ndarray) -> tuple:
    return (
        grid.shape,
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_signature(pair: dict) -> tuple:
    return (
        _array_signature(pair["input"]),
        _array_signature(pair["output"]),
    )


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        (
            split_name,
            tuple(_pair_signature(pair) for pair in puzzle[split_name]),
        )
        for split_name in ("train", "test")
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    signature = (_grid_rows(input_grid), _grid_rows(output_grid))
    return signature in _SOURCE_PAIR_SIGNATURES


def _check_optional_count(
    name: str,
    value: int | None,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    return int(value)


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _check_optional_count("num_train", num_train)
    num_test = _check_optional_count("num_test", num_test)

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")

    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        resolved_test = int(rng.integers(1, 5))
        minimum_train = max(2, resolved_test + 1)
        resolved_train = int(rng.integers(minimum_train, 11))
        return resolved_train, resolved_test

    if num_train is None:
        minimum_train = max(2, num_test + 1)
        resolved_train = int(rng.integers(minimum_train, 11))
        return resolved_train, num_test

    maximum_test = min(4, num_train - 1)
    resolved_test = int(rng.integers(1, maximum_test + 1))
    return num_train, resolved_test


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    """Only color preferences are shared; geometry and backgrounds are pair local."""
    palette = [int(c) for c in rng.permutation(10)]
    return {"base_background": palette[0], "train_palette": tuple(palette[1:7])}


def _canvas_size(rng: np.random.Generator) -> tuple[int, int]:
    def dimension() -> int:
        band = int(rng.choice(3, p=(0.30, 0.43, 0.27)))
        bounds = ((6, 12), (12, 21), (21, 31))[band]
        return int(rng.integers(*bounds))
    return dimension(), dimension()


def _mode(rng: np.random.Generator, regime: str) -> str:
    names = ("interior", "top", "bottom", "left", "right", "corner", "span", "free")
    weights = ((.30, .09, .09, .09, .09, .09, .08, .17)
               if regime == "source_train" else
               (.23, .11, .11, .11, .11, .10, .12, .11))
    return str(rng.choice(names, p=weights))


def _regions(rng: np.random.Generator, bounds: tuple[int, int, int, int],
             count: int, strip_axis: int | None) -> list[tuple[int, int, int, int]]:
    if strip_axis is not None:
        r0, c0, r1, c1 = bounds
        length = (r1-r0+1) if strip_axis == 0 else (c1-c0+1)
        extras = rng.multinomial(length - (3*count-1), [1/count]*count)
        lengths = [2+int(x) for x in extras]
        rng.shuffle(lengths)
        out = []
        cursor = r0 if strip_axis == 0 else c0
        for size in lengths:
            if strip_axis == 0:
                out.append((cursor, c0, cursor+size-1, c1))
            else:
                out.append((r0, cursor, r1, cursor+size-1))
            cursor += size+1
        return out

    regions = [bounds]
    while len(regions) < count:
        choices = []
        for i, (r0, c0, r1, c1) in enumerate(regions):
            if r1-r0+1 >= 5:
                choices.append((i, 0))
            if c1-c0+1 >= 5:
                choices.append((i, 1))
        if not choices:
            raise AssertionError("canvas cannot hold requested rectangles")
        i, axis = choices[int(rng.integers(len(choices)))]
        r0, c0, r1, c1 = regions.pop(i)
        if axis == 0:
            cut = int(rng.integers(r0+2, r1-1))
            regions.extend(((r0,c0,cut-1,c1), (cut+1,c0,r1,c1)))
        else:
            cut = int(rng.integers(c0+2, c1-1))
            regions.extend(((r0,c0,r1,cut-1), (r0,cut+1,r1,c1)))
    rng.shuffle(regions)
    return regions


def _length(rng: np.random.Generator, maximum: int) -> int:
    roll = rng.random()
    if roll < .13:
        return maximum
    if roll < .28:
        return 2
    return int(rng.integers(2, maximum+1))


def _box_in_region(rng: np.random.Generator,
                   region: tuple[int, int, int, int],
                   forced_edge: str | None = None,
                   forced_size: tuple[int, int] | None = None,
                   span_axis: int | None = None) -> tuple[int, int, int, int]:
    r0,c0,r1,c1 = region
    hmax,wmax = r1-r0+1,c1-c0+1
    h,w = forced_size or (_length(rng,hmax), _length(rng,wmax))
    if span_axis == 0:
        h = hmax
    elif span_axis == 1:
        w = wmax
    top = (r0 if forced_edge in ("top", "top_left", "top_right") else
           r1-h+1 if forced_edge in ("bottom", "bottom_left", "bottom_right") else
           int(rng.integers(r0, r1-h+2)))
    left = (c0 if forced_edge in ("left", "top_left", "bottom_left") else
            c1-w+1 if forced_edge in ("right", "top_right", "bottom_right") else
            int(rng.integers(c0, c1-w+2)))
    return top,left,top+h-1,left+w-1


def _place_boxes(rng: np.random.Generator, count: int, mode: str,
                 shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    height,width = shape
    if mode == "interior":
        bounds = (1,1,height-2,width-2)
    else:
        bounds = (0,0,height-1,width-1)
    strip_axis = int(rng.integers(2)) if mode == "span" else None
    regions = _regions(rng, bounds, count, strip_axis)
    target_index = None
    target_edge = None
    if mode in ("top", "bottom", "left", "right", "corner"):
        if mode == "corner":
            target_edge = str(rng.choice(("top_left", "top_right", "bottom_left", "bottom_right")))
            needed = set(target_edge.split("_"))
        else:
            target_edge = mode
            needed = {mode}
        candidates = []
        for i,(r0,c0,r1,c1) in enumerate(regions):
            edges = set()
            if r0 == 0: edges.add("top")
            if r1 == height-1: edges.add("bottom")
            if c0 == 0: edges.add("left")
            if c1 == width-1: edges.add("right")
            if needed <= edges:
                candidates.append(i)
        target_index = int(rng.choice(candidates))
    elif mode == "span":
        target_index = int(rng.integers(count))

    forced_equal = None
    if count >= 2 and rng.random() < .28:
        pair = rng.choice(count, 2, replace=False)
        a,b = int(pair[0]),int(pair[1])
        hmax = min(regions[a][2]-regions[a][0]+1, regions[b][2]-regions[b][0]+1)
        wmax = min(regions[a][3]-regions[a][1]+1, regions[b][3]-regions[b][1]+1)
        if mode != "span" or target_index not in (a,b):
            forced_equal = (a,b,(_length(rng,hmax),_length(rng,wmax)))
    boxes = []
    for i,region in enumerate(regions):
        size = forced_equal[2] if forced_equal and i in forced_equal[:2] else None
        boxes.append(_box_in_region(rng, region,
                     target_edge if i == target_index else None,
                     size,
                     1-strip_axis if mode == "span" and i == target_index else None))
    return boxes


def _background(rng: np.random.Generator, params: dict, first: bool,
                regime: str) -> int:
    base = params["base_background"]
    if first or rng.random() < (.58 if regime == "source_train" else .46):
        return base
    return int(rng.choice([c for c in range(10) if c != base]))


def _colors(rng: np.random.Generator, count: int, background: int,
            params: dict, regime: str, train_colors: set[int]) -> tuple[int, ...]:
    if regime == "source_train":
        pool = [c for c in params["train_palette"] if c != background]
        if len(pool) < count:
            pool += [c for c in range(10) if c != background and c not in pool]
        return tuple(int(c) for c in rng.choice(pool, count, replace=False))
    reused = [c for c in sorted(train_colors) if c != background]
    fresh = [c for c in range(10) if c != background and c not in train_colors]
    reuse_count = min(int(rng.integers(0,count+1)),len(reused))
    reuse_count = max(reuse_count, count-len(fresh))
    selected = [int(c) for c in rng.choice(reused, reuse_count, replace=False)]
    selected += [int(c) for c in rng.choice(fresh, count-reuse_count, replace=False)]
    rng.shuffle(selected)
    return tuple(selected)


def _render_pair(shape: tuple[int, int], background: int,
                 colors: tuple[int, ...],
                 boxes: list[tuple[int, int, int, int]],
                 orientations: tuple[int, ...]) -> dict:
    inp = np.full(shape, background, dtype=np.int8)
    out = np.full(shape, background, dtype=np.int8)
    for color,(r0,c0,r1,c1),diagonal in zip(colors,boxes,orientations):
        out[r0:r1+1,c0:c1+1] = color
        if diagonal == 0:
            inp[r0,c0] = inp[r1,c1] = color
        else:
            inp[r0,c1] = inp[r1,c0] = color
    return {"input":inp,"output":out}


def _pair_features(pair: dict) -> dict:
    inp = pair["input"]
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    boxes = []
    orientations = []
    for color in colors:
        if int(color) == background:
            continue
        pts = np.argwhere(inp == color)
        (ra,ca),(rb,cb) = pts
        boxes.append((min(ra,rb),min(ca,cb),max(ra,rb),max(ca,cb)))
        orientations.append(0 if (ra-rb)*(ca-cb)>0 else 1)
    return {"background":background,"colors":set(int(c) for c in colors if c != background),
            "boxes":boxes,"orientations":orientations,
            "shapes":set((r1-r0+1,c1-c0+1) for r0,c0,r1,c1 in boxes)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train,n_test = _resolve_counts(rng,num_train,num_test)
    params = sample_puzzle_parameters(rng)
    puzzle = {"train":[],"test":[]}
    seen_inputs = set()
    train_colors: set[int] = set()
    train_shapes: set[tuple[int,int]] = set()
    test_palettes = set()
    for regime,amount in (("source_train",n_train),("target_test",n_test)):
        for index in range(amount):
            for attempt in range(80):
                count = (2 if regime == "source_train" and index == 0 else
                         int(rng.choice((1,2,3,4),p=(.27,.49,.20,.04))) if regime == "source_train" else
                         int(rng.choice((1,2,3,4),p=(.12,.39,.32,.17))))
                mode = _mode(rng,regime)
                height,width = _canvas_size(rng)
                if attempt >= 20:
                    height,width = max(height,22),max(width,22)
                if mode == "interior" and count > 1:
                    height,width = max(height,9),max(width,9)
                if mode == "span":
                    if rng.random() < .5:
                        height = max(height,3*count-1)
                    else:
                        width = max(width,3*count-1)
                    # The strip axis is chosen in _place_boxes; make either feasible.
                    height,width = max(height,3*count-1),max(width,3*count-1)
                boxes = _place_boxes(rng,count,mode,(height,width))
                if regime == "source_train" and index == 0:
                    orientations = (0,1) if rng.random()<.5 else (1,0)
                else:
                    orientations = tuple(int(x) for x in rng.integers(0,2,size=count))
                bg = _background(rng,params,regime == "source_train" and index == 0,regime)
                colors = _colors(rng,count,bg,params,regime,train_colors)
                pair = _render_pair((height,width),bg,colors,boxes,orientations)
                features = _pair_features(pair)
                signature = _array_signature(pair["input"])
                if signature in seen_inputs or _is_source_pair(pair["input"],pair["output"]):
                    continue
                if regime == "target_test":
                    if not (features["shapes"]-train_shapes):
                        continue
                    palette = frozenset(features["colors"])
                    if palette in test_palettes:
                        continue
                break
            else:
                raise AssertionError("bounded pair construction exhausted")
            puzzle["train" if regime == "source_train" else "test"].append(pair)
            seen_inputs.add(signature)
            if regime == "source_train":
                train_colors.update(features["colors"])
                train_shapes.update(features["shapes"])
            else:
                test_palettes.add(palette)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle,dict) and set(puzzle)=={"train","test"},"puzzle schema")
    train,test = puzzle["train"],puzzle["test"]
    _require(isinstance(train,list) and isinstance(test,list),"pair lists")
    _require(2<=len(train)<=10 and 1<=len(test)<=4 and len(train)>len(test),"pair counts")
    seen = set()
    train_shapes = set()
    train_orientations = set()
    test_palettes = set()
    for split,pairs in (("train",train),("test",test)):
        for pair in pairs:
            _require(isinstance(pair,dict) and set(pair)=={"input","output"},"pair schema")
            inp,out = pair["input"],pair["output"]
            for grid in (inp,out):
                _require(isinstance(grid,np.ndarray) and grid.ndim==2 and
                         grid.dtype.kind in "iu" and 1<=grid.shape[0]<=30 and
                         1<=grid.shape[1]<=30,"grid type/shape")
                _require(np.all((grid>=0)&(grid<=9)),"ARC colors")
            _require(inp.shape==out.shape and not np.array_equal(inp,out),"output dimensions/change")
            signature = _array_signature(inp)
            _require(signature not in seen,"duplicate input")
            seen.add(signature)
            _require(not _is_source_pair(inp,out),"source copy")
            values,counts=np.unique(inp,return_counts=True)
            _require(len(values)>=2,"missing markers")
            background=int(values[int(np.argmax(counts))])
            _require(int(np.max(counts)) > inp.size//2,"ambiguous background")
            marker_mask = inp != background
            _require(not np.any(marker_mask[1:,:] & marker_mask[:-1,:]) and
                     not np.any(marker_mask[:,1:] & marker_mask[:,:-1]),
                     "markers must be isolated")
            predicted=np.full(inp.shape,background,dtype=np.int8)
            occupied=np.zeros(inp.shape,dtype=bool)
            shapes=set()
            colors=set()
            for color in values:
                color=int(color)
                if color==background: continue
                pts=np.argwhere(inp==color)
                _require(len(pts)==2,"marker count")
                (ra,ca),(rb,cb)=pts
                _require(ra!=rb and ca!=cb,"opposite corners")
                r0,r1=sorted((int(ra),int(rb)))
                c0,c1=sorted((int(ca),int(cb)))
                _require(not np.any(occupied[r0:r1+1,c0:c1+1]),"overlapping rectangles")
                occupied[r0:r1+1,c0:c1+1]=True
                predicted[r0:r1+1,c0:c1+1]=color
                shapes.add((r1-r0+1,c1-c0+1))
                colors.add(color)
                if split=="train":
                    train_orientations.add(0 if (ra-rb)*(ca-cb)>0 else 1)
            _require(np.array_equal(predicted,out),"inclusive bounding-box rule")
            if split=="train":
                train_shapes.update(shapes)
            else:
                _require(bool(shapes-train_shapes),"test has no novel rectangle size")
                palette=frozenset(colors)
                _require(palette not in test_palettes,"repeated test palette")
                test_palettes.add(palette)
    _require(any(len(_pair_features(p)["boxes"])>=2 for p in train),"no multicolor demonstration")
    _require(train_orientations=={0,1},"both diagonals absent in training")


def _same_puzzle(a: dict,b: dict) -> bool:
    return _puzzle_signature(a)==_puzzle_signature(b)


def self_test() -> None:
    samples=[generate_puzzle(seed) for seed in range(42)]
    for p in samples: validate_generated_puzzle(p)
    assert _same_puzzle(samples[0],generate_puzzle(0))
    assert len({_puzzle_signature(p) for p in samples})==42
    assert len({len(p["train"]) for p in samples})>=4
    assert len({len(p["test"]) for p in samples})>=3
    for n,m in ((2,1),(10,1),(10,4),(3,2),(5,4)):
        p=generate_puzzle(1234,n,m)
        assert len(p["train"])==n and len(p["test"])==m
    for n,m in ((None,1),(None,4),(2,None),(10,None)):
        p=generate_puzzle(4321,n,m)
        assert (n is None or len(p["train"])==n) and (m is None or len(p["test"])==m)
    for n,m in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4)):
        try: generate_puzzle(0,n,m)
        except ValueError: pass
        else: raise AssertionError("invalid count accepted")
    features=[_pair_features(pair) for p in samples for pair in p["train"]+p["test"]]
    sizes={(pair["input"].shape) for p in samples for pair in p["train"]+p["test"]}
    shapes={shape for f in features for shape in f["shapes"]}
    assert len(sizes)>=35 and len(shapes)>=45
    assert {len(f["boxes"]) for f in features}>={1,2,3,4}
    assert len({f["background"] for f in features})>=7
    assert any(len({f["background"] for f in [_pair_features(pair) for pair in p["train"]+p["test"]]})>1 for p in samples)
    assert any(any(shape not in {s for pair in p["train"] for s in _pair_features(pair)["shapes"]}
                   for shape in _pair_features(test)["shapes"])
               for p in samples for test in p["test"])
    review = {name: 0 for name in (
        "train_edge", "test_interior", "single_top", "single_bottom",
        "single_left", "single_right", "span", "equal_size",
        "same_diagonal", "cross_color_nearer", "reuse_none",
        "reuse_partial", "reuse_all", "high_count_test",
        "different_backgrounds", "different_test_palettes")}
    for puzzle in samples:
        train_features = [_pair_features(pair) for pair in puzzle["train"]]
        test_features = [_pair_features(pair) for pair in puzzle["test"]]
        train_colors = set().union(*(f["colors"] for f in train_features))
        if len({f["background"] for f in train_features+test_features}) > 1:
            review["different_backgrounds"] += 1
        if len(test_features) > 1 and len({frozenset(f["colors"]) for f in test_features}) > 1:
            review["different_test_palettes"] += 1
        for split, pairs, pair_features in (("train",puzzle["train"],train_features),
                                             ("test",puzzle["test"],test_features)):
            for pair,f in zip(pairs,pair_features):
                height,width = pair["input"].shape
                edges = []
                sizes = []
                for r0,c0,r1,c1 in f["boxes"]:
                    edge = set()
                    if r0 == 0: edge.add("top")
                    if r1 == height-1: edge.add("bottom")
                    if c0 == 0: edge.add("left")
                    if c1 == width-1: edge.add("right")
                    edges.append(edge)
                    sizes.append((r1-r0+1,c1-c0+1))
                    for side in ("top","bottom","left","right"):
                        if edge == {side}: review["single_"+side] += 1
                    if {"top","bottom"} <= edge or {"left","right"} <= edge:
                        review["span"] += 1
                if split == "train" and any(edges): review["train_edge"] += 1
                if split == "test" and not any(edges): review["test_interior"] += 1
                if len(sizes) >= 2:
                    if len(set(sizes)) < len(sizes): review["equal_size"] += 1
                    if len(set(f["orientations"])) == 1: review["same_diagonal"] += 1
                    markers = {int(c): [tuple(p) for p in np.argwhere(pair["input"]==c)]
                               for c in f["colors"]}
                    crossed = False
                    for color,(a,b) in markers.items():
                        mate_distance = abs(a[0]-b[0])+abs(a[1]-b[1])
                        for other,points in markers.items():
                            if color != other and any(abs(a[0]-p[0])+abs(a[1]-p[1]) < mate_distance
                                                      for p in points):
                                crossed = True
                    review["cross_color_nearer"] += crossed
                if split == "test":
                    review["high_count_test"] += len(sizes) >= 3
                    reused = len(f["colors"] & train_colors)
                    key = ("reuse_none" if reused == 0 else
                           "reuse_all" if reused == len(f["colors"]) else "reuse_partial")
                    review[key] += 1
    assert all(value >= 5 for value in review.values()), review
