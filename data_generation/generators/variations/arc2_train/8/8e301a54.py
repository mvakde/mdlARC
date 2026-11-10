"""Generate ARC scenes in which each component falls by its input area."""
import numpy as np

_SOURCE_ROWS = (
    ("7777777777","7777777797","7222777797","7777777797","7777577797","7777777777","7777777777","7777777777","7777777777","7777777777"),
    ("7777777777","7777777777","7757777997","7777777797","7777227777","7777777777","7777777777","7777777777","7777777777","7777777777"),
    ("7777777777","7999977777","7777977777","7777772777","7777772777","7777772777","7777777777","7777777775","7777777775","7777777777"),
    ("7777777777","7777277777","7772227777","7777277777","7777777997","7777777777","5777777777","5777777777","5777777777","7777777777"),
)

def _key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))

_SOURCE_KEYS = {((10,10),tuple(int(c) for row in rows for c in row)) for rows in _SOURCE_ROWS}

def _normal(cells):
    top = min(r for r,c in cells)
    left = min(c for r,c in cells)
    return tuple(sorted((r-top,c-left) for r,c in cells))

def _components(grid, background):
    unseen = {(int(r),int(c)) for r,c in np.argwhere(grid != background)}
    answer = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        cells = []
        while stack:
            r,c = stack.pop()
            cells.append((r,c))
            for neighbor in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        answer.append(tuple(sorted(cells)))
    return tuple(sorted(answer))

def _background(grid):
    counts = np.bincount(grid.ravel().astype(int),minlength=10)
    ranked = np.sort(counts)
    assert ranked[-1] > ranked[-2]
    return int(np.argmax(counts))

def _rule(grid, background):
    result = np.full_like(grid,background)
    occupied = set()
    for cells in _components(grid,background):
        color = int(grid[cells[0]])
        for row,col in cells:
            dest = (row+len(cells),col)
            if dest[0] < grid.shape[0]:
                assert dest not in occupied
                occupied.add(dest)
                result[dest] = color
    return result

def _shape(rng, area, special=None):
    if special == "L":
        cells = [(0,0),(1,0),(1,1)]
    elif special == "square":
        cells = [(0,0),(0,1),(1,0),(1,1)]
    elif special == "line":
        cells = [(0,c) for c in range(area)]
    elif area == 1:
        cells = [(0,0)]
    else:
        kind = int(rng.integers(0,10))
        if kind < 2:
            cells = [(0,c) for c in range(area)]
        elif area == 8 and kind == 2:
            cells = [(r,c) for r in range(3) for c in range(3) if (r,c)!=(1,1)]
        elif area == 5 and kind == 2:
            cells = [(0,1),(1,0),(1,1),(1,2),(2,1)]
        elif area == 4 and kind == 2:
            cells = [(0,0),(0,1),(1,0),(1,1)]
        else:
            cells = {(0,0)}
            while len(cells) < area:
                ordered = sorted(cells)
                base = ordered[int(rng.integers(0,len(ordered)))]
                dr,dc = ((1,0),(-1,0),(0,1),(0,-1))[int(rng.integers(0,4))]
                cells.add((base[0]+dr,base[1]+dc))
    cells = list(cells)
    if int(rng.integers(0,2)):
        cells = [(r,-c) for r,c in cells]
    for _ in range(int(rng.integers(0,4))):
        cells = [(c,-r) for r,c in cells]
    return _normal(cells)

def _candidates(shape,height,width,mode):
    sh = max(r for r,c in shape)+1
    sw = max(c for r,c in shape)+1
    if sh > height or sw > width:
        return []
    area = len(shape)
    choices = []
    for top in range(height-sh+1):
        base = tuple((top+r,c) for r,c in shape)
        survivors = sum(r+area<height for r,c in base)
        if mode == "none" and survivors != area: continue
        if mode == "partial" and not 0<survivors<area: continue
        if mode == "full" and survivors != 0: continue
        if mode == "severe" and not 0<survivors*2<area: continue
        for left in range(width-sw+1):
            cells = tuple((r,c+left) for r,c in base)
            shifted = tuple((r+area,c) for r,c in cells if r+area<height)
            choices.append((cells,shifted))
    return choices

def _place(rng,shapes,modes,height,width):
    entries = []
    for i,(shape,mode) in enumerate(zip(shapes,modes)):
        choices = _candidates(shape,height,width,mode)
        if not choices:
            raise RuntimeError("No placement")
        order = rng.permutation(len(choices))
        entries.append((i,[choices[int(j)] for j in order]))
    entries.sort(key=lambda item:len(item[1]))
    chosen = [None]*len(shapes)
    source_used = set()
    target_used = set()
    def visit(depth):
        if depth == len(entries):
            return True
        index,choices = entries[depth]
        for cells,shifted in choices:
            if any((r,c) in source_used or (r-1,c) in source_used or
                   (r+1,c) in source_used or (r,c-1) in source_used or
                   (r,c+1) in source_used for r,c in cells):
                continue
            if any(cell in target_used for cell in shifted):
                continue
            chosen[index] = cells
            source_used.update(cells)
            target_used.update(shifted)
            if visit(depth+1): return True
            source_used.difference_update(cells)
            target_used.difference_update(shifted)
        return False
    if not visit(0):
        raise RuntimeError("No nonconflicting scene")
    return chosen

def _count(name,value,lower,upper):
    if not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)):
        raise ValueError(name+" must be an integer")
    value = int(value)
    if not lower<=value<=upper:
        raise ValueError(name+" out of range")
    return value

def _counts(rng,seed,num_train,num_test):
    train = None if num_train is None else _count("num_train",num_train,2,10)
    test = None if num_test is None else _count("num_test",num_test,1,4)
    if train is None and test is None:
        test = 1+seed%4
        train = test+1+(seed//4)%(10-test)
    elif train is None:
        train = int(rng.integers(test+1,11))
    elif test is None:
        test = int(rng.integers(1,min(4,train-1)+1))
    if train<=test:
        raise ValueError("num_train must exceed num_test")
    return train,test

def _modes(rng,areas,split,force_partial=False):
    modes = ["none"]*len(areas)
    eligible = [i for i,a in enumerate(areas) if a>=2]
    if force_partial or (eligible and int(rng.integers(0,10)) < (5 if split=="train" else 7)):
        if eligible:
            first = eligible[int(rng.integers(0,len(eligible)))]
            if force_partial:
                modes[first] = "partial"
            else:
                options = ["partial","partial","full"]
                if areas[first]>=3: options.append("severe")
                modes[first] = options[int(rng.integers(0,len(options)))]
            if not force_partial and len(areas)>=3 and int(rng.integers(0,5))==0:
                others = [i for i in range(len(areas)) if i!=first]
                second = others[int(rng.integers(0,len(others)))]
                modes[second] = "partial" if areas[second]>=2 else "full"
    if all(m!="none" for m in modes):
        modes[0] = "none"
    return modes

def _make_pair(rng,areas,specials,colors,modes,background):
    for _ in range(60):
        height = int(rng.integers(max(9,max(areas)+4),21))
        width = int(rng.integers(9,21))
        if max(areas)<=6 and int(rng.integers(0,9))==0:
            height = width = 10
        shapes = [_shape(rng,a,s) for a,s in zip(areas,specials)]
        try:
            placed = _place(rng,shapes,modes,height,width)
        except RuntimeError:
            continue
        grid = np.full((height,width),background,dtype=np.int8)
        for color,cells in zip(colors,placed):
            for cell in cells:
                grid[cell] = color
        if len(_components(grid,background)) != len(areas):
            continue
        output = _rule(grid,background)
        if np.any(output!=background) and not np.array_equal(grid,output):
            return {"input":grid,"output":output}
    raise RuntimeError("Could not construct pair")

def generate_puzzle(seed: int,num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not isinstance(seed,(int,np.integer)) or isinstance(seed,(bool,np.bool_)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed<0: raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count,test_count = _counts(rng,seed,num_train,num_test)
    fixed_palette = bool(rng.integers(0,2))
    shared_palette = tuple(int(x) for x in rng.choice(10,size=4,replace=False))
    max_area = int(rng.integers(6,9)) if train_count>=3 else 5

    train_counts = [int(rng.integers(2,5)) for _ in range(train_count)]
    if train_counts[0]+train_counts[1]<5:
        train_counts[int(rng.integers(0,2))] = 3
    slots = [int(x) for x in rng.permutation([1,2,3,4,5])]
    area_lists = []
    cursor = 0
    for count in train_counts:
        areas = []
        for _ in range(count):
            if cursor<len(slots):
                areas.append(slots[cursor]); cursor+=1
            else:
                areas.append(int(rng.integers(1,max_area+1)))
        area_lists.append(areas)
    if train_count>=3:
        large_pair = int(rng.integers(2,train_count))
        area_lists[large_pair][int(rng.integers(0,train_counts[large_pair]))] = max_area
    same_color_demo = int(rng.integers(0,2))
    clip_options = [i for i,areas in enumerate(area_lists) if any(a>=2 for a in areas)]
    clip_demo = clip_options[int(rng.integers(0,len(clip_options)))]

    puzzle = {"train":[],"test":[]}
    seen = set()
    for split in ("train","test"):
        if split=="train":
            counts = train_counts
        else:
            counts = [train_counts[int(rng.integers(0,len(train_counts)))]
                      for _ in range(test_count)]
        for index,count in enumerate(counts):
            accepted = False
            for _ in range(120):
                areas = (list(area_lists[index]) if split=="train"
                         else [int(rng.integers(1,max_area+1)) for _ in range(count)])
                specials = [None]*count
                if split=="train" and index<2 and 3 in areas and 4 in areas:
                    specials[areas.index(3)] = "L"
                    specials[areas.index(4)] = "square"
                elif split=="train" and index<2 and areas.count(4)>=2:
                    fours = [i for i,a in enumerate(areas) if a==4]
                    specials[fours[0]],specials[fours[1]] = "square","line"
                palette = shared_palette if fixed_palette else tuple(
                    int(x) for x in rng.choice(10,size=4,replace=False))
                background = palette[0]
                foreground = list(palette[1:])
                rng.shuffle(foreground)
                colors = foreground[:count]
                if count==4:
                    colors.append(foreground[int(rng.integers(0,3))])
                if split=="train" and index==same_color_demo:
                    colors[1] = colors[0]
                elif count>=2 and int(rng.integers(0,3))==0:
                    colors[1] = colors[0]
                modes = _modes(rng,areas,split,split=="train" and index==clip_demo)
                try:
                    pair = _make_pair(rng,areas,specials,colors,modes,background)
                except RuntimeError:
                    continue
                key = _key(pair["input"])
                if key in seen or key in _SOURCE_KEYS:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                accepted = True
                break
            if not accepted:
                raise RuntimeError("Could not generate distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle

def _facts(grid,background):
    facts = []
    for cells in _components(grid,background):
        colors = {int(grid[cell]) for cell in cells}
        assert len(colors)==1
        area = len(cells)
        survivors = sum(r+area<grid.shape[0] for r,c in cells)
        facts.append((area,survivors,next(iter(colors)),_normal(cells)))
    return facts

def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={"train","test"}
    assert isinstance(puzzle["train"],list) and isinstance(puzzle["test"],list)
    assert 2<=len(puzzle["train"])<=10 and 1<=len(puzzle["test"])<=4
    assert len(puzzle["train"])>len(puzzle["test"])
    seen_inputs,seen_pairs = set(),set()
    records = {"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={"input","output"}
            src,dst = pair["input"],pair["output"]
            for grid in (src,dst):
                assert isinstance(grid,np.ndarray) and grid.ndim==2
                assert np.issubdtype(grid.dtype,np.integer)
                assert 1<=grid.shape[0]<=30 and 1<=grid.shape[1]<=30
                assert 0<=int(grid.min())<=int(grid.max())<=9
            assert src.shape==dst.shape and not np.array_equal(src,dst)
            input_key = _key(src)
            pair_key = (input_key,_key(dst))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert input_key not in _SOURCE_KEYS
            seen_inputs.add(input_key); seen_pairs.add(pair_key)
            bg = _background(src)
            facts = _facts(src,bg)
            assert 2<=len(facts)<=4
            assert all(1<=area<=8 for area,survivors,color,shape in facts)
            assert any(survivors>0 for area,survivors,color,shape in facts)
            assert np.array_equal(dst,_rule(src,bg))
            records[split].append(facts)
    train = [f for pair in records["train"] for f in pair]
    test = [f for pair in records["test"] for f in pair]
    assert {1,2,3,4,5}<={area for area,survivors,color,shape in train}
    assert any(survivors<area for area,survivors,color,shape in train)
    assert max(f[0] for f in test)<=max(f[0] for f in train)
    train_counts = {len(pair) for pair in records["train"]}
    assert all(len(pair) in train_counts for pair in records["test"])
    train_repeated = any(len({f[2] for f in pair})<len(pair) for pair in records["train"])
    test_repeated = any(len({f[2] for f in pair})<len(pair) for pair in records["test"])
    assert train_repeated or not test_repeated

def _equal(a,b):
    return all(len(a[split])==len(b[split]) and
               all(np.array_equal(pa[side],pb[side])
                   for pa,pb in zip(a[split],b[split])
                   for side in ("input","output"))
               for split in ("train","test"))

def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles})>=4
    assert len({len(p["test"]) for p in puzzles})>=3
    assert _equal(generate_puzzle(17),generate_puzzle(17))
    assert _equal(generate_puzzle(20260827,6,3),generate_puzzle(20260827,6,3))
    assert len({tuple(_key(pair["input"]) for split in ("train","test")
                      for pair in p[split]) for p in puzzles})>=40
    for seed,train,test in ((101,2,1),(102,10,1),(103,10,4),
                            (201,None,1),(202,None,4),
                            (203,2,None),(204,10,None)):
        puzzle = generate_puzzle(seed,train,test)
        validate_generated_puzzle(puzzle)
        if train is not None: assert len(puzzle["train"])==train
        if test is not None: assert len(puzzle["test"])==test
    for kwargs in ({"num_train":1},{"num_train":11},{"num_test":0},
                   {"num_test":5},{"num_train":2,"num_test":2},
                   {"num_train":3,"num_test":4}):
        try: generate_puzzle(300,**kwargs)
        except ValueError: pass
        else: raise AssertionError("Illegal counts accepted")
    sizes,counts,shapes,clips,palettes,layouts = set(),set(),set(),set(),set(),set()
    repeated = clear_test = full_clip = severe_train = 0
    rectangular = large = multiple_clips = unlike = same_box = 0
    color_area_contrast = combined = ring_count = 0
    varied_sizes = varied_counts = varied_backgrounds = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_sizes,local_counts,local_backgrounds = set(),set(),set()
        for split in ("train","test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                bg = _background(grid)
                facts = _facts(grid,bg)
                local_sizes.add(grid.shape)
                local_counts.add(len(facts))
                local_backgrounds.add(bg)
                sizes.add(grid.shape); counts.add(len(facts))
                palettes.add((bg,tuple(sorted(set(int(c) for c in grid.flat)-{bg}))))
                layouts.add(tuple(sorted((shape,area) for area,surv,color,shape in facts)))
                shapes.update(shape for area,surv,color,shape in facts)
                clips.update((area,surv) for area,surv,color,shape in facts)
                repeated += len({color for area,surv,color,shape in facts})<len(facts)
                full_clip += any(surv==0 for area,surv,color,shape in facts)
                rectangular += grid.shape[0]!=grid.shape[1]
                large += any(area>5 for area,surv,color,shape in facts)
                multiple_clips += sum(surv<area for area,surv,color,shape in facts)>=2
                area_shapes,box_areas = {},{}
                color_areas = {}
                for area,surv,color,shape in facts:
                    area_shapes.setdefault(area,set()).add(shape)
                    color_areas.setdefault(color,set()).add(area)
                    box = (max(r for r,c in shape)+1,max(c for r,c in shape)+1)
                    box_areas.setdefault(box,set()).add(area)
                unlike += any(len(values)>1 for values in area_shapes.values())
                same_box += any(len(values)>1 for values in box_areas.values())
                contrast = any(len(values)>1 for values in color_areas.values())
                color_area_contrast += contrast
                combined += (contrast and grid.shape[0]!=grid.shape[1] and
                             any(area>5 for area,surv,color,shape in facts) and
                             any(surv<area for area,surv,color,shape in facts))
                ring = tuple(sorted((r,c) for r in range(3) for c in range(3)
                                    if (r,c)!=(1,1)))
                ring_count += any(shape==ring for area,surv,color,shape in facts)
                if split=="train":
                    severe_train += any(0<2*surv<area for area,surv,color,shape in facts)
                else:
                    clear_test += all(surv==area for area,surv,color,shape in facts)
        varied_sizes += len(local_sizes)>1
        varied_counts += len(local_counts)>1
        varied_backgrounds += len(local_backgrounds)>1
    assert len(sizes)>=35 and len(palettes)>=20
    assert counts=={2,3,4} and len(shapes)>=30
    assert len(clips)>=12 and len(layouts)>=100
    assert repeated>=30 and clear_test>=5 and full_clip>=5 and severe_train>=1
    assert rectangular>=30 and large>=10 and multiple_clips>=1
    assert unlike>=10 and same_box>=5
    assert color_area_contrast>=20 and combined>=1 and ring_count>=1
    assert varied_sizes>=30 and varied_counts>=20 and varied_backgrounds>=5
