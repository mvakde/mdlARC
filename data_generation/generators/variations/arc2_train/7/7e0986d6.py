"""Sparse two-way noise on unions of filled axis-aligned rectangles."""

import numpy as np


_SOURCE_INPUTS = (
    ("0000000000000000", "0222201000000000", "0222200002122222",
     "0221200002222222", "0222200002222222", "0222210002222212",
     "0000000002222222", "0000000000000000", "0000000000001000",
     "0102222221222000", "0001222222222001", "0002222221221000",
     "0002222222222000"),
    ("80008033333800", "00000033333000", "00000033838000",
     "00333033333000", "00333038333000", "00338000000000",
     "00000000033333", "08033383033383", "00038333033333",
     "00033333033333", "33338333800000", "33300000000000",
     "38308000000008"),
    ("00000000000055450", "04000040000055550", "05455555500055550",
     "05555555500055554", "05555555500055550", "05555554500055550",
     "05555555500055450", "05555455500040000", "00000000000000000",
     "00000000000000040", "00000055555400000", "00040055545500000"),
)


def _count(value, name, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


def _counts(seed, rng, num_train, num_test):
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt is None and ne is None:
        ne = 1 + seed % 4
        nt = ne + 1 + (seed // 4) % (10 - ne)
    elif nt is None:
        nt = int(rng.integers(ne + 1, 11))
    elif ne is None:
        ne = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    return nt, ne


def _runs(mask, axis):
    """All maximal foreground runs of length at least two."""
    h, w = mask.shape
    found = []
    if axis == 0:
        for r in range(h):
            c = 0
            while c < w:
                if not mask[r, c]:
                    c += 1
                    continue
                start = c
                while c < w and mask[r, c]:
                    c += 1
                if c - start >= 2:
                    found.append((r, start, c))
    else:
        for c in range(w):
            r = 0
            while r < h:
                if not mask[r, c]:
                    r += 1
                    continue
                start = r
                while r < h and mask[r, c]:
                    r += 1
                if r - start >= 2:
                    found.append((c, start, r))
    return found


def _reconstruct(input_grid, background, foreground, corruption):
    """Cross intact horizontal and vertical runs into supported rectangles.

    Every admitted rectangle has a complete foreground row and column and
    contains no observed background. A noisy gap cannot become a bridge just
    because foreground lies on both sides of one noisy pixel.
    """
    fg = input_grid == foreground
    bg = input_grid == background
    inferred = fg.copy()
    horizontal = _runs(fg, 0)
    vertical = _runs(fg, 1)
    for r, left, right in horizontal:
        for c, top, bottom in vertical:
            if left <= c < right and top <= r < bottom and not np.any(bg[top:bottom, left:right]):
                inferred[top:bottom, left:right] = True
    answer = input_grid.copy()
    noisy = input_grid == corruption
    answer[noisy & inferred] = foreground
    answer[noisy & ~inferred] = background
    return answer


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    count = 0
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        count += 1
        seen[r, c] = True
        stack = [(r, c)]
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
    return count


def _has_composite(mask):
    """Whether a connected foreground component bends beyond one rectangle."""
    seen = np.zeros(mask.shape, bool)
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r,c]:
            continue
        cells = []
        seen[r,c] = True
        stack = [(r,c)]
        while stack:
            y,x = stack.pop()
            cells.append((y,x))
            for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=yy<mask.shape[0] and 0<=xx<mask.shape[1] and mask[yy,xx] and not seen[yy,xx]:
                    seen[yy,xx] = True
                    stack.append((yy,xx))
        ys = [y for y,_ in cells]
        xs = [x for _,x in cells]
        if len(cells) < (max(ys)-min(ys)+1)*(max(xs)-min(xs)+1):
            return True
    return False


def _span(mask):
    s = np.zeros(mask.shape, bool)
    s[:,1:-1] |= mask[:,:-2] & mask[:,2:]
    s[1:-1,:] |= mask[:-2,:] & mask[2:,:]
    return s


def _neighbors(mask):
    n = np.zeros(mask.shape, bool)
    n[1:,:] |= mask[:-1,:]
    n[:-1,:] |= mask[1:,:]
    n[:,1:] |= mask[:,:-1]
    n[:,:-1] |= mask[:,1:]
    return n


def _paint(shape, rectangles):
    mask = np.zeros(shape, bool)
    for a, b, c, d in rectangles:
        if not (0 <= a < b <= shape[0] and 0 <= c < d <= shape[1] and b-a >= 2 and d-c >= 2):
            raise ValueError("bad rectangle")
        mask[a:b,c:d] = True
    return mask


def _rect(rng, h, w, thin=False):
    if thin:
        if int(rng.integers(2)):
            rh, rw = 2, int(rng.integers(4, min(11, w+1)))
        else:
            rh, rw = int(rng.integers(4, min(11, h+1))), 2
    else:
        rh = int(rng.integers(3, min(9, h+1)))
        rw = int(rng.integers(3, min(10, w+1)))
    r = int(rng.integers(0, h-rh+1))
    c = int(rng.integers(0, w-rw+1))
    return (r, r+rh, c, c+rw)


def _separate(first, second):
    a,b,c,d = first
    e,f,g,k = second
    return b+1 < e or f+1 < a or d+1 < g or k+1 < c


def _attach(rng, base, h, w):
    a,b,c,d = base
    side = int(rng.integers(4))
    if side < 2:
        rh = int(rng.integers(2, min(6, h+1)))
        rw = int(rng.integers(2, min(7, w+1)))
        lo = max(0, a-rh+2)
        hi = min(h-rh, b-2)
        if lo > hi:
            return None
        r = int(rng.integers(lo, hi+1))
        c = d - 1 if side == 0 else c - rw + 1
    else:
        rh = int(rng.integers(2, min(7, h+1)))
        rw = int(rng.integers(2, min(6, w+1)))
        lo = max(0, c-rw+2)
        hi = min(w-rw, d-2)
        if lo > hi:
            return None
        c = int(rng.integers(lo, hi+1))
        r = b - 1 if side == 2 else a - rh + 1
    if r < 0 or c < 0 or r+rh > h or c+rw > w:
        return None
    return (r,r+rh,c,c+rw)


def _layout(rng, shape, mode):
    h,w = shape
    rectangles = []
    if mode == "gap":
        rh = int(rng.integers(3, min(7,h-1)))
        lw = int(rng.integers(2, min(6,(w-3)//2+1)))
        rw = int(rng.integers(2, min(6,(w-3)//2+1)))
        top = int(rng.integers(0,h-rh+1))
        left = int(rng.integers(0,w-lw-rw))
        rectangles = [(top,top+rh,left,left+lw),
                      (top,top+rh,left+lw+1,left+lw+1+rw)]
        if int(rng.integers(2)) and h >= 13:
            for _ in range(20):
                extra = _rect(rng,h,w)
                if all(_separate(extra, old) for old in rectangles):
                    rectangles.append(extra)
                    break
        return rectangles
    base = _rect(rng,h,w,thin=bool(rng.integers(4)==0))
    rectangles.append(base)
    if mode in ("connected", "mixed"):
        for _ in range(int(rng.integers(1,4))):
            parent = rectangles[int(rng.integers(len(rectangles)))]
            branch = _attach(rng,parent,h,w)
            if branch is not None and np.count_nonzero(_paint(shape,rectangles+[branch])) > np.count_nonzero(_paint(shape,rectangles))+3:
                rectangles.append(branch)
        if len(rectangles) < 2:
            return None
    if mode in ("separated", "mixed"):
        target = int(rng.integers(2,5)) if mode == "separated" else int(rng.integers(1,3))
        for _ in range(target):
            for _ in range(70):
                candidate = _rect(rng,h,w,thin=bool(rng.integers(4)==0))
                if all(_separate(candidate, old) for old in rectangles):
                    rectangles.append(candidate)
                    break
        if mode == "separated" and len(rectangles) < 2:
            return None
        if mode == "mixed" and len(rectangles) < 3:
            return None
    return rectangles


def _palette(rng, convention):
    if convention["palette_mode"] == "stable":
        return convention["palette"]
    if convention["palette_mode"] == "pool" and int(rng.integers(4)):
        return convention["palette"]
    colors = [int(x) for x in rng.permutation(10)]
    bg = convention["palette"][0] if convention["fixed_background"] else colors[0]
    remaining = [x for x in colors if x != bg]
    return bg, remaining[0], remaining[1]


def _noise(rng, clean, colors, hard):
    bg,fg,no = colors
    area = clean.size
    maximum = min(22, max(6, int(area*0.115)))
    total = int(rng.integers(4, maximum+1))
    inside_count = int(rng.integers(2,total-1))
    outside_count = total-inside_count
    inside = list(map(tuple,np.argwhere(clean)))
    outside = list(map(tuple,np.argwhere(~clean)))
    chosen_inside = []
    chosen_outside = []
    unspanned = list(map(tuple,np.argwhere(clean & ~_span(clean))))
    gap = list(map(tuple,np.argwhere(~clean & _span(clean))))
    if hard and unspanned and int(rng.integers(4)) != 0:
        chosen_inside.append(unspanned[int(rng.integers(len(unspanned)))])
    if hard and gap and int(rng.integers(3)) != 0:
        chosen_outside.append(gap[int(rng.integers(len(gap)))])
    if hard and int(rng.integers(3)) == 0:
        for a in inside:
            if a not in chosen_inside:
                neighbors = [b for b in inside if b not in chosen_inside
                             and abs(a[0]-b[0])+abs(a[1]-b[1]) == 1]
                if neighbors:
                    chosen_inside.extend((a,neighbors[int(rng.integers(len(neighbors)))]))
                    break
    for pool, selected, count in ((inside,chosen_inside,inside_count),(outside,chosen_outside,outside_count)):
        for i in rng.permutation(len(pool)):
            cell = pool[int(i)]
            if cell not in selected:
                selected.append(cell)
            if len(selected) >= count:
                break
        del selected[count:]
    output = np.where(clean,fg,bg).astype(np.int8)
    input_grid = output.copy()
    for r,c in chosen_inside+chosen_outside:
        input_grid[r,c] = no
    # A noisy cell between two blocks must have an intact gap row elsewhere.
    for r,c in chosen_outside:
        if 0<c<clean.shape[1]-1 and clean[r,c-1] and clean[r,c+1]:
            if not any(input_grid[rr,c]==bg and clean[rr,c-1] and clean[rr,c+1]
                       for rr in range(clean.shape[0]) if rr!=r):
                return None
    if np.count_nonzero(input_grid==fg) <= np.count_nonzero(input_grid==no):
        return None
    if np.count_nonzero(input_grid==bg) <= np.count_nonzero(input_grid==fg):
        return None
    if not np.array_equal(_reconstruct(input_grid,bg,fg,no),output):
        return None
    return input_grid,output


def _make_pair(rng, convention, regime, mode):
    for _ in range(250):
        if regime == "source_train":
            h = int(rng.integers(11,21))
            w = int(rng.integers(12,23))
        else:
            h = int(rng.integers(10,23))
            w = int(rng.integers(12,26))
        if mode in ("mixed","gap"):
            h = max(h,13)
            w = max(w,17)
        shape = h,w
        rectangles = _layout(rng,shape,mode)
        if rectangles is None:
            continue
        clean = _paint(shape,rectangles)
        if mode in ("connected","mixed") and not _has_composite(clean):
            continue
        fill = np.count_nonzero(clean)/clean.size
        if not 0.12 <= fill <= 0.43:
            continue
        colors = _palette(rng,convention)
        pair = _noise(rng,clean,colors,hard=(mode in ("gap","connected","mixed") or bool(rng.integers(2))))
        if pair is not None:
            return {"input":pair[0],"output":pair[1]}
    raise RuntimeError(f"could not construct {mode} pair")


def _roles(pair):
    a,b = pair["input"],pair["output"]
    vals,counts = np.unique(a,return_counts=True)
    assert len(vals)==3 and len(set(map(int,counts)))==3
    order = np.argsort(counts)
    no,fg,bg = (int(vals[i]) for i in order)
    assert counts[order[0]] < counts[order[1]] < counts[order[2]]
    return bg,fg,no


def _signature(array):
    return array.shape,array.tobytes()


def _source_rows(a):
    return tuple("".join(str(int(v)) for v in row) for row in a)


def _inspect_pair(pair):
    assert isinstance(pair,dict) and set(pair)=={"input","output"}
    a,b = pair["input"],pair["output"]
    for g in (a,b):
        assert isinstance(g,np.ndarray) and g.ndim==2
        assert 1<=g.shape[0]<=30 and 1<=g.shape[1]<=30
        assert np.issubdtype(g.dtype,np.integer)
        assert 0<=int(np.min(g)) and int(np.max(g))<=9
    assert a.shape==b.shape and not np.array_equal(a,b)
    assert _source_rows(a) not in _SOURCE_INPUTS
    bg,fg,no = _roles(pair)
    assert set(map(int,np.unique(b)))=={bg,fg}
    noisy = a==no
    assert np.array_equal(a[~noisy],b[~noisy])
    assert np.count_nonzero(noisy & (b==fg))>=2
    assert np.count_nonzero(noisy & (b==bg))>=2
    assert np.count_nonzero(noisy)<=max(22,int(a.size*0.115))
    assert np.array_equal(_reconstruct(a,bg,fg,no),b)
    mask = b==fg
    assert 0.12 <= np.count_nonzero(mask)/mask.size <= 0.43
    assert _runs(a==fg,0) and _runs(a==fg,1)
    return (bg,fg,no,_components(mask),_has_composite(mask))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle)=={"train","test"}
    assert isinstance(puzzle["train"],list) and isinstance(puzzle["test"],list)
    nt,ne = len(puzzle["train"]),len(puzzle["test"])
    assert 2<=nt<=10 and 1<=ne<=4 and nt>ne
    seen = set()
    train_roles = []
    test_roles = []
    for part,roles in (("train",train_roles),("test",test_roles)):
        for pair in puzzle[part]:
            roles.append(_inspect_pair(pair))
            h,w = pair["input"].shape
            if part == "train":
                assert 11<=h<=20 and 12<=w<=22
            else:
                assert 10<=h<=22 and 12<=w<=25
            key = _signature(pair["input"])
            assert key not in seen
            seen.add(key)
    assert any(role[3]>=2 for role in train_roles)
    assert any(role[4] for role in train_roles)
    if any(role[4] for role in test_roles):
        assert any(role[4] for role in train_roles)
    # Each training example itself demonstrates both possible repairs.
    for pair in puzzle["train"]:
        bg,fg,no = _roles(pair)
        noisy = pair["input"]==no
        assert np.any(noisy & (pair["output"]==fg))
        assert np.any(noisy & (pair["output"]==bg))
    # All pair palettes obey the same observable semantic convention.
    assert all(len({r[0],r[1],r[2]})==3 for r in train_roles+test_roles)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or int(seed)<0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt,ne = _counts(seed,rng,num_train,num_test)
    colors = tuple(int(x) for x in rng.permutation(10)[:3])
    convention = {"palette":colors,
                  "palette_mode":("stable","pool","free")[int(rng.integers(3))],
                  "fixed_background":bool(rng.integers(2))}
    train_modes = ["separated","connected"]
    if bool(rng.integers(2)):
        train_modes[0] = "gap"
    if bool(rng.integers(2)):
        train_modes[1] = "mixed"
    options = ("single","separated","connected","mixed","gap")
    train_modes.extend(options[int(rng.integers(len(options)))] for _ in range(nt-2))
    test_modes = [options[int(rng.integers(len(options)))] for _ in range(ne)]
    puzzle = {"train":[],"test":[]}
    seen = set()
    for part,modes,regime in (("train",train_modes,"source_train"),("test",test_modes,"target_test")):
        for mode in modes:
            pair_rng = np.random.default_rng(int(rng.integers(0,2**63-1)))
            for _ in range(12):
                pair = _make_pair(pair_rng,convention,regime,mode)
                key = _signature(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("duplicate puzzle input")
            seen.add(key)
            puzzle[part].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((part,tuple((_signature(p["input"]),_signature(p["output"])) for p in puzzle[part])) for part in ("train","test"))


def self_test() -> None:
    for nt,ne in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None),(7,3)):
        p = generate_puzzle(300+int(nt or 0)*10+int(ne or 0),nt,ne)
        validate_generated_puzzle(p)
        if nt is not None: assert len(p["train"])==nt
        if ne is not None: assert len(p["test"])==ne
    for nt,ne in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4)):
        try:
            generate_puzzle(42,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts,test_counts = set(),set()
    shapes,areas,layouts,corruptions,palettes,components = set(),set(),set(),set(),set(),set()
    signatures = set()
    hard_inside=hard_gap=adjacent_loss=thin=0
    noisy_counts = set()
    interior_counts = set()
    stable_palettes = changing_backgrounds = repeated_palettes = 0
    palette_overlap = connected_tests = separate_tests = similar_test_sizes = 0
    composite_tests = test_composite_covered = border_contacts = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = _puzzle_key(p)
        assert signature == _puzzle_key(generate_puzzle(seed))
        signatures.add(signature)
        train_counts.add(len(p["train"])); test_counts.add(len(p["test"]))
        pair_palettes = []
        train_shapes = [q["input"].shape for q in p["train"]]
        test_shapes = [q["input"].shape for q in p["test"]]
        similar_test_sizes += int(any(abs(h-th)<=2 and abs(w-tw)<=2
                                      for h,w in test_shapes for th,tw in train_shapes))
        train_composite = any(_has_composite(q["output"]==_roles(q)[1]) for q in p["train"])
        connected_tests += int(any(_components(q["output"]==_roles(q)[1])==1 for q in p["test"]))
        separate_tests += int(any(_components(q["output"]==_roles(q)[1])>=2 for q in p["test"]))
        test_has_composite = any(_has_composite(q["output"]==_roles(q)[1]) for q in p["test"])
        composite_tests += int(test_has_composite)
        test_composite_covered += int(test_has_composite and train_composite)
        for part in ("train","test"):
            for pair in p[part]:
                a,b = pair["input"],pair["output"]
                bg,fg,no = _roles(pair)
                clean = b==fg
                noisy = a==no
                assert np.array_equal(_reconstruct(a,bg,fg,no),b)
                shapes.add(a.shape); areas.add(int(np.count_nonzero(clean)))
                layouts.add((a.shape,clean.tobytes()))
                corruptions.add((a.shape,noisy.tobytes()))
                palettes.add((bg,fg,no)); components.add(_components(clean))
                pair_palettes.append((bg,fg,no))
                noisy_counts.add(int(noisy.sum()))
                interior_counts.add(int((noisy&clean).sum()))
                hard_inside += int(np.any(noisy & clean & ~_span(a==fg)))
                hard_gap += int(np.any(noisy & ~clean & _span(clean)))
                adjacent_loss += int(np.any((noisy & clean) & _neighbors(noisy & clean)))
                thin += int(np.any(np.sum(clean,axis=0)==2) or np.any(np.sum(clean,axis=1)==2))
                border_contacts += int(np.any(clean[0,:]) or np.any(clean[-1,:])
                                       or np.any(clean[:,0]) or np.any(clean[:,-1]))
        stable_palettes += int(len(set(pair_palettes))==1)
        changing_backgrounds += int(len({x[0] for x in pair_palettes})>1)
        repeated_palettes += int(len(set(pair_palettes))<len(pair_palettes))
        palette_overlap += int(bool(set(pair_palettes[:len(p["train"])])
                                    & set(pair_palettes[len(p["train"]):])))
    assert len(train_counts)>=4 and len(test_counts)>=3
    assert len(signatures)==42 and len(shapes)>=35 and len(layouts)>=100
    assert len(corruptions)>=100 and len(areas)>=35 and len(palettes)>=30
    assert {1,2,3}.issubset(components)
    assert hard_inside>=12 and hard_gap>=3 and adjacent_loss>=3 and thin>=8
    assert min(noisy_counts)<=5 and max(noisy_counts)>=16
    assert min(interior_counts)==2 and max(interior_counts)>=10
    assert stable_palettes>=3 and changing_backgrounds>=3 and repeated_palettes>=5
    assert palette_overlap>=4 and connected_tests>=5 and separate_tests>=5
    assert similar_test_sizes>=5 and composite_tests>=5
    assert test_composite_covered==composite_tests and border_contacts>=20
