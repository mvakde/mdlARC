"""Contour casting: a stripe stencil acts on shape-matched Voronoi fronts.

The blueprint's structural color is the stencil's 'inherit owner' symbol.
No solving convention is hidden in a seed or a train/test schedule.
"""
from __future__ import annotations

import numpy as np

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _bbox(cells):
    return min(r for r, _ in cells), max(r for r, _ in cells) + 1, min(c for _, c in cells), max(c for _, c in cells) + 1


def _normalise(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rotate(cells, turns):
    for _ in range(turns % 4):
        cells = tuple((c, -r) for r, c in cells)
    return _normalise(cells)


def _orientations(cells, reflect=True):
    cells = _normalise(cells)
    bases = (cells, tuple((r, -c) for r, c in cells)) if reflect else (cells,)
    return tuple(sorted({_rotate(b, q) for b in bases for q in range(4)}))


def _canonical(cells):
    return _orientations(cells)[0]


def _mask(cells):
    cells = _normalise(cells)
    a = np.zeros((max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1), dtype=bool)
    for r, c in cells:
        a[r, c] = True
    return a


def _expand(a, scale):
    return np.repeat(np.repeat(a, scale, 0), scale, 1)


def _decode(cells, scale):
    r0, r1, c0, c1 = _bbox(cells)
    if (r1-r0) % scale or (c1-c0) % scale or len(cells) % (scale*scale):
        return None
    blocks = {}
    for r, c in cells:
        p = (r-r0)//scale, (c-c0)//scale
        blocks[p] = blocks.get(p, 0) + 1
    if any(n != scale*scale for n in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _dilate(mask):
    """One king-move step: the eight-neighbor square metric."""
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([p[r:r+h, c:c+w] for r in range(3) for c in range(3)])


def _cast_by_fronts(shape, holes, colors, code, structural):
    """Solve using simultaneous dilation and first arrival, with color consensus."""
    layers = code[:-1]  # the terminal duplicate is a closing convention
    masks = []
    for cells in holes:
        m = np.zeros(shape, dtype=bool)
        for r, c in cells:
            m[r, c] = True
        masks.append(m)
    out = np.full(shape, structural, dtype=np.int8)
    visited = np.zeros(shape, dtype=bool)
    for distance, token in enumerate(reversed(layers)):
        arrivals = np.stack(masks)
        reached = np.any(arrivals, axis=0)
        frontier = reached & ~visited
        if token != structural:
            out[frontier] = token
        else:
            # Multiple shapes carrying the same color agree, even on a tie.
            lo = np.min(np.where(arrivals, np.array(colors)[:, None, None], 10), axis=0)
            hi = np.max(np.where(arrivals, np.array(colors)[:, None, None], -1), axis=0)
            agree = frontier & (lo == hi)
            out[agree] = lo[agree]
        visited |= reached
        if distance + 1 < len(layers):
            masks = [_dilate(m) for m in masks]
    return out


def _cast_by_distance(shape, holes, colors, code, structural, metric="chebyshev"):
    """Generation/check construction: direct minimum distances to hole cells."""
    out = np.full(shape, structural, dtype=np.int8)
    layers = code[:-1]
    for r in range(shape[0]):
        for c in range(shape[1]):
            distances = [min((max(abs(r-x), abs(c-y)) if metric == "chebyshev" else abs(r-x)+abs(c-y))
                             for x, y in hole) for hole in holes]
            d = min(distances)
            if d >= len(layers):
                continue
            token = int(layers[-1-d])
            owners = {color for color, dist in zip(colors, distances) if dist == d}
            out[r, c] = token if token != structural else next(iter(owners)) if len(owners) == 1 else structural
    return out


def _recover(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= a.size//2:
        raise ValueError("background must occupy more than half the canvas")
    components = _components(a != bg)
    codes = []
    for cells in components:
        r0, r1, c0, c1 = _bbox(cells)
        sub = a[r0:r1, c0:c1]
        if not (4 <= sub.shape[0] <= 8 and 2 <= sub.shape[1] <= 5):
            continue
        if len(cells) != sub.size or len(np.unique(sub)) < 2:
            continue
        code = sub[:, 0]
        if np.all(sub == code[:, None]) and code[-1] == code[-2]:
            codes.append((tuple(map(int, code)), (r0, r1, c0, c1)))
    solutions = []
    for code, code_bounds in codes:
        structural = code[-1]
        for component in _components(a == structural):
            bounds = _bbox(component)
            r0, r1, c0, c1 = bounds
            board = a[r0:r1, c0:c1]
            if min(board.shape) < 6:
                continue
            if not (np.all(board[0] == structural) and np.all(board[-1] == structural)
                    and np.all(board[:, 0] == structural) and np.all(board[:, -1] == structural)):
                continue
            if not np.all((board == structural) | (board == bg)):
                continue
            holes = _components(board == bg)
            if not 2 <= len(holes) <= 5:
                continue
            outside = np.ones(a.shape, dtype=bool)
            outside[r0:r1, c0:c1] = False
            cr0, cr1, cc0, cc1 = code_bounds
            if np.any(~outside[cr0:cr1, cc0:cc1]):
                continue
            outside[cr0:cr1, cc0:cc1] = False
            # All source objects have a background moat, including diagonals.
            moat_ok = True
            for x0, x1, y0, y1 in (bounds, code_bounds):
                moat = np.zeros(a.shape, dtype=bool)
                moat[max(0,x0-1):min(a.shape[0],x1+1), max(0,y0-1):min(a.shape[1],y1+1)] = True
                moat[x0:x1, y0:y1] = False
                if np.any((a != bg) & moat):
                    moat_ok = False
            if not moat_ok:
                continue
            external = _components((a != bg) & outside)
            large = [cells for cells in external if len(cells) >= 4]
            noise = [cells for cells in external if len(cells) < 4]
            literal_colors = set(code) - {structural}
            if len(large) != len(holes):
                continue
            if any(len({int(a[r,c]) for r,c in cells}) != 1 for cells in external):
                continue
            piece_colors = [int(a[cells[0]]) for cells in large]
            if set(piece_colors) & ({structural, bg} | literal_colors):
                continue
            hole_groups = {}
            for i, cells in enumerate(holes):
                hole_groups.setdefault(_canonical(cells), []).append(i)
            for scale in range(2, 5):
                if max(board.shape)*scale > 30:
                    continue
                shapes = [_decode(cells, scale) for cells in large]
                if any(s is None for s in shapes):
                    continue
                groups = {}
                for s, color in zip(shapes, piece_colors):
                    groups.setdefault(_canonical(s), []).append((s, color))
                if set(groups) != set(hole_groups) or any(len(groups[k]) != len(hole_groups[k]) for k in groups):
                    continue
                if any(len({color for _, color in group}) != 1 for group in groups.values()):
                    raise ValueError("congruent differently colored pieces are ambiguous")
                colors = [groups[_canonical(cells)][0][1] for cells in holes]
                logical = _cast_by_fronts(board.shape, holes, colors, code, structural)
                reflection, rotation = False, False
                for key, indices in hole_groups.items():
                    if len(indices) == 1:
                        dest, source = _normalise(holes[indices[0]]), groups[key][0][0]
                        reflection |= dest not in _orientations(source, False)
                        rotation |= dest != source and dest in _orientations(source, False)
                info = dict(background=bg, structural=structural, code=code, code_width=cc1-cc0,
                            code_bounds=code_bounds, piece_cells=large, piece_colors=piece_colors,
                            board_shape=board.shape, board_bounds=bounds, holes=holes, colors=colors,
                            pieces=shapes, scale=scale, reflection=reflection, rotation=rotation,
                            noise_sizes=tuple(map(len, noise)))
                solutions.append((_expand(logical, scale), info))
    if len(solutions) != 1:
        raise ValueError("expected one stripe code, blueprint and scale interpretation")
    return solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _recover(input_grid)[0]


def _grow(rng, area, span=4):
    cells = {(0,0)}
    while len(cells) < area:
        choices = sorted({(r+dr,c+dc) for r,c in cells for dr,dc in _NEIGHBORS} - cells)
        choices = [p for p in choices if max(_mask(tuple(cells | {p})).shape) <= span]
        cells.add(choices[int(rng.integers(len(choices)))])
    return _normalise(cells)


def _place_holes(rng, shapes, bh, bw):
    for _ in range(20):
        occupied, holes = set(), [None]*len(shapes)
        order = sorted(range(len(shapes)), key=lambda i: -len(shapes[i]))
        for i in order:
            variants = _orientations(shapes[i])
            s = variants[int(rng.integers(len(variants)))]
            sh, sw = _mask(s).shape
            origins = [(r,c) for r in range(1,bh-sh) for c in range(1,bw-sw)]
            for j in rng.permutation(len(origins)):
                r0,c0 = origins[int(j)]
                cells = tuple((r+r0,c+c0) for r,c in s)
                if any(p in occupied or any((p[0]+dr,p[1]+dc) in occupied for dr,dc in _NEIGHBORS) for p in cells):
                    continue
                holes[i] = cells
                occupied.update(cells)
                break
            if holes[i] is None:
                break
        else:
            return holes
    return None


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0,2), dtype=int)
    s = np.pad(blocked.astype(int), ((1,0),(1,0))).cumsum(0).cumsum(1)
    totals = s[h:,w:] - s[:-h,w:] - s[h:,:-w] + s[:-h,:-w]
    return np.argwhere(totals == 0)


def _pack(rng, board, code, code_width, loose, colors, scale, bg, noise_color):
    strip = np.repeat(np.array(code,dtype=np.int8)[:,None], code_width, 1)
    masks = [np.ones(board.shape,dtype=bool), np.ones(strip.shape,dtype=bool)]
    masks += [_expand(_mask(s), scale) for s in loose]
    for attempt in range(35):
        h,w = (int(rng.integers(24,31)),int(rng.integers(24,31))) if attempt < 15 else (30,30)
        canvas = np.full((h,w),bg,dtype=np.int8)
        blocked = np.zeros((h,w),dtype=bool)
        order = sorted(range(len(masks)), key=lambda i: (-masks[i].size, float(rng.random())))
        for i in order:
            mask = masks[i]
            mh,mw = mask.shape
            origins = _free_origins(blocked,mh,mw)
            if not len(origins):
                break
            top,left = map(int,origins[int(rng.integers(len(origins)))])
            view = canvas[top:top+mh,left:left+mw]
            if i < 2:
                view[:] = board if i == 0 else strip
            else:
                view[mask] = colors[i-2]
            blocked[max(0,top-1):min(h,top+mh+1),max(0,left-1):min(w,left+mw+1)] = True
        else:
            for _ in range(int(rng.integers(0,5))):
                noise_shape = (((0,0),), ((0,0),(0,1)), ((0,0),(0,1),(1,0)))[int(rng.integers(3))]
                noise_shape = _rotate(noise_shape,int(rng.integers(4)))
                nh,nw = _mask(noise_shape).shape
                origins = _free_origins(blocked,nh,nw)
                if not len(origins):
                    break
                r0,c0 = map(int,origins[int(rng.integers(len(origins)))])
                color = noise_color if rng.random() < .6 else colors[int(rng.integers(len(colors)))]
                for r,c in noise_shape:
                    canvas[r0+r,c0+c] = color
                blocked[max(0,r0-1):min(h,r0+nh+1),max(0,c0-1):min(w,c0+nw+1)] = True
            if np.count_nonzero(canvas == bg) <= canvas.size//2:
                continue
            if rng.random() < .7:
                r0,r1,c0,c1 = _bbox(tuple(map(tuple,np.argwhere(canvas != bg).tolist())))
                r0,c0 = max(0,r0-int(rng.integers(4))),max(0,c0-int(rng.integers(4)))
                r1,c1 = min(h,r1+int(rng.integers(4))),min(w,c1+int(rng.integers(4)))
                crop = canvas[r0:r1,c0:c1]
                if np.count_nonzero(crop == bg) > crop.size//2:
                    canvas = crop.copy()
            return canvas
    return None


def _features(info):
    shape,holes,colors,code,s = (info[k] for k in ("board_shape","holes","colors","code","structural"))
    h,w = shape
    carry_tie = literal_tie = owner_band = literal_band = agreement_tie = False
    max_distance = 0
    for r in range(h):
        for c in range(w):
            ds = [min(max(abs(r-x),abs(c-y)) for x,y in hole) for hole in holes]
            d = min(ds)
            max_distance = max(max_distance, d)
            if d == 0 or d >= len(code)-1:
                continue
            nearest = {color for color,dist in zip(colors,ds) if dist == d}
            nearest_count = sum(dist == d for dist in ds)
            carry = code[-2-d] == s
            carry_tie |= carry and len(nearest) > 1
            agreement_tie |= carry and nearest_count > 1 and len(nearest) == 1
            literal_tie |= not carry and len(nearest) > 1
            literal_band |= not carry
            owner_band |= carry and len(nearest) == 1
    equal_area = any(len(a) == len(b) and _canonical(a) != _canonical(b) and colors[i] != colors[j]
                     for i,a in enumerate(holes) for j,b in enumerate(holes) if i < j)
    base = np.full(shape,s,dtype=np.int8)
    for hole,color in zip(holes,colors):
        for r,c in hole:
            base[r,c] = color
    correct = _cast_by_fronts(shape,holes,colors,code,s)
    collapsed = tuple(v for i,v in enumerate(code[:-1]) if i == 0 or v != code[i-1])
    collapsed += (collapsed[-1],)
    return dict(carry_tie=carry_tie,literal_tie=literal_tie,owner_band=owner_band,
                literal_band=literal_band,agreement_tie=agreement_tie,max_distance=max_distance,
                equal_area=equal_area, no_a=not np.array_equal(base,correct),
                collapse_diff=not np.array_equal(_cast_by_fronts(shape,holes,colors,collapsed,s),correct),
                metric_diff=not np.array_equal(_cast_by_distance(shape,holes,colors,code,s,"manhattan"),correct))


def _make_pair(rng, scale, length, count, witness=None, width=None, forbidden_roles=(), full_reach=False):
    for _ in range(180):
        palette = list(map(int,rng.permutation(10)))
        bg,s = palette[:2]
        if forbidden_roles and (bg == forbidden_roles[0] or s == forbidden_roles[1]):
            continue
        globals_ = palette[2:4]
        available = palette[4:9]
        noise_color = palette[9]
        if witness == "reflection":
            # The second equal-area shape is grown, never chosen by position.
            first = ((0,0),(1,0),(2,0),(2,1))
            second = _grow(rng,4)
            if _canonical(first) == _canonical(second):
                continue
            shapes = [first,second]
            for _i in range(count-2):
                shapes.append(_grow(rng,int(rng.integers(1,5))))
            if any(_canonical(x) == _canonical(first) for x in shapes[1:]):
                continue
            code_layers = [globals_[0],globals_[0],s,globals_[1],s]
            if length == 6:
                code_layers.insert(3,globals_[0])
        else:
            shapes = []
            for i in range(count):
                if shapes and rng.random() < .18 and not (witness and i == 0):
                    shapes.append(shapes[int(rng.integers(len(shapes)))])
                else:
                    max_area = 4 if scale == 4 or count >= 4 else 8
                    shapes.append(_grow(rng,int(rng.integers(1,max_area+1))))
            if witness == "rotation" and (len(_orientations(shapes[0],False)) < 2 or
                                           any(_canonical(x) == _canonical(shapes[0]) for x in shapes[1:])):
                continue
            code_layers = [globals_[int(rng.integers(2))] for _ in range(length)]
            code_layers[-1] = s
            code_layers[int(rng.integers(1,length-1))] = s
            for i in range(1,length-1):
                if rng.random() < .25:
                    code_layers[i] = s
        code = tuple(code_layers + [s])
        limit = 30//scale
        low = max(8 if witness == "reflection" else 6, length+1 if full_reach else 6)
        bh,bw = int(rng.integers(low,limit+1)),int(rng.integers(low,limit+1))
        if rng.random() < .2:
            bw = bh
        holes = _place_holes(rng,shapes,bh,bw)
        if holes is None:
            continue
        keys = list(dict.fromkeys(_canonical(x) for x in shapes))
        color_count = len(keys) if witness == "reflection" else 1 if witness == "rotation" else int(rng.integers(1,len(keys)+1))
        assignment = {key:available[i % color_count] for i,key in enumerate(keys)}
        colors = [assignment[_canonical(x)] for x in shapes]
        board = np.full((bh,bw),s,dtype=np.int8)
        prefeatures = _features(dict(board_shape=board.shape,holes=holes,colors=colors,code=code,structural=s))
        if not prefeatures["literal_band"] or (full_reach and prefeatures["max_distance"] < length-1):
            continue
        if witness == "rotation" and not prefeatures["agreement_tie"]:
            continue
        loose = []
        for i,cells in enumerate(holes):
            dest = _normalise(cells)
            variants = _orientations(dest)
            if i == 0 and witness == "reflection":
                variants = tuple(x for x in variants if x not in _orientations(dest,False))
            elif i == 0 and witness == "rotation":
                variants = tuple(x for x in _orientations(dest,False) if x != dest)
            loose.append(variants[int(rng.integers(len(variants)))])
            for r,c in cells:
                board[r,c] = bg
        code_width = int(rng.integers(2,6)) if width is None else width
        canvas = _pack(rng,board,code,code_width,loose,colors,scale,bg,noise_color)
        if canvas is None:
            continue
        output = _expand(_cast_by_distance(board.shape,holes,colors,code,s),scale)
        try:
            recovered,info = _recover(canvas)
        except ValueError:
            continue
        assert np.array_equal(recovered,output)
        f = _features(info)
        if not f["literal_band"] or (full_reach and f["max_distance"] < length-1):
            continue
        if witness == "reflection" and not (info["reflection"] and f["equal_area"] and f["carry_tie"]
                                             and f["literal_tie"] and f["owner_band"] and f["collapse_diff"] and f["metric_diff"]):
            continue
        if witness == "rotation" and not info["rotation"]:
            continue
        return {"input":canvas,"output":output},info
    raise RuntimeError("bounded feasible contour-casting construction exhausted")


def _check_count(value, name, low, high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _check_count(num_train,"num_train",2,10)
    ntest = _check_count(num_test,"num_test",1,4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))
    first_scale = int(rng.integers(2,4))
    first_width = int(rng.integers(2,6))
    puzzle,seen = {"train":[],"test":[]},set()
    first_roles = ()
    for split,n in (("train",ntrain),("test",ntest)):
        for i in range(n):
            witness = ("reflection" if i == 0 else "rotation" if i == 1 else None) if split == "train" else None
            scale = (first_scale if i == 0 else 5-first_scale if i == 1 else int(rng.integers(2,4))) if split == "train" else (4 if shift == 1 else int(rng.integers(2,4)))
            if witness == "reflection":
                length = 5 if shift == 0 else int(rng.integers(5,7))
            else:
                length = int(rng.integers(3,6 if shift == 0 and split == "train" else 8))
            if split == "test" and shift == 0:
                length = int(rng.integers(6,8))
            if witness == "rotation":
                length = int(rng.integers(3,5))
            count = int(rng.integers(2,4 if shift == 2 and split == "train" else 6))
            if split == "test" and shift == 2:
                count = int(rng.integers(4,6))
            if split == "test" and shift == 0:
                count = int(rng.integers(2,4))
            if scale == 4:
                count = int(rng.integers(2,4))
            width = first_width if witness == "reflection" else next(w for w in rng.permutation(np.arange(2,6)) if w != first_width) if witness == "rotation" else None
            for _ in range(12):
                pair,info = _make_pair(rng,scale,length,count,witness,width,first_roles if witness == "rotation" else (),
                                       full_reach=split == "test" and shift == 0)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if witness == "reflection":
                first_roles = info["background"],info["structural"]
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen,infos = set(),{"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted,info = _recover(inp)
            assert np.array_equal(predicted,out), "wrong contour-casting answer"
            logical = _cast_by_distance(info["board_shape"],info["holes"],info["colors"],info["code"],info["structural"])
            assert np.array_equal(_expand(logical,info["scale"]),out)
            assert info["code"][-1] == info["code"][-2] == info["structural"]
            assert info["code"][0] != info["structural"]
            assert info["structural"] in info["code"][1:-2]
            assert all(n <= 3 for n in info["noise_sizes"])
            assert _features(info)["no_a"] and _features(info)["literal_band"]
            # Matched holes always retain their donor color, without ambiguity.
            for cells,color in zip(info["holes"],info["colors"]):
                for r,c in cells:
                    assert np.all(out[r*info["scale"]:(r+1)*info["scale"],c*info["scale"]:(c+1)*info["scale"]] == color)
            infos[split].append(info)
    training = infos["train"]
    fs = [_features(i) for i in training]
    assert any(i["reflection"] for i in training) and any(i["rotation"] for i in training)
    assert any(f["agreement_tie"] for f in fs)
    assert any(all(f[k] for k in ("equal_area","carry_tie","literal_tie","owner_band","collapse_diff","metric_diff")) for f in fs)
    for role in ("background","structural","scale","code_width"):
        assert len({i[role] for i in training}) >= 2
    assert len({len(i["code"]) for i in training}) >= 2
    # At least one visible, semantic train/test extrapolation, without metadata.
    assert (any(len(i["code"]) > max(len(j["code"]) for j in training)
                and _features(i)["max_distance"] >= len(i["code"])-2 for i in infos["test"])
            or any(i["scale"] not in {j["scale"] for j in training} for i in infos["test"])
            or min(len(i["holes"]) for i in infos["test"]) > max(len(i["holes"]) for i in training))
    return None


def self_test():
    """Bounded fixed-seed coverage plus negative and counterfactual checks."""
    signatures,dimensions,classes,codes,palettes,scales,counts = set(),set(),set(),set(),set(),set(),set()
    collision,reflection,rotation,shared,noise = 0,0,0,0,0
    repeated_fixture = None
    donor_fixture = None
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = tuple((_key(pair["input"]),_key(pair["output"])) for split in ("train","test") for pair in p[split])
        signatures.add(signature)
        for split in ("train","test"):
            for pair in p[split]:
                out,i = _recover(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()),pair["output"])
                dimensions.add((pair["input"].shape,out.shape))
                classes.update(_canonical(h) for h in i["holes"])
                rolemap = {i["structural"]:0}
                codes.add(tuple(rolemap.setdefault(c,len(rolemap)) for c in i["code"]))
                palettes.add((i["background"],i["structural"]))
                scales.add(i["scale"])
                counts.add(len(i["holes"]))
                f = _features(i)
                collision += f["carry_tie"]
                reflection += i["reflection"]
                rotation += i["rotation"]
                shared += len(set(i["colors"])) < len(i["colors"])
                noise += bool(i["noise_sizes"])
                if i["reflection"] and f["equal_area"]:
                    donor_fixture = pair, i
                piece_keys = [_canonical(s) for s in i["pieces"]]
                if len(set(piece_keys)) < len(piece_keys):
                    repeated_fixture = pair, i
                # Omitting B's donor assignment changes both core and carry bands.
                changed_colors = [((c+1)%10) for c in i["colors"]]
                changed = _cast_by_fronts(i["board_shape"],i["holes"],changed_colors,i["code"],i["structural"])
                assert not np.array_equal(_expand(changed,i["scale"]),out)
    assert len(signatures) == 42
    assert len(dimensions) >= 80 and len(classes) >= 35 and len(codes) >= 20
    assert len(palettes) >= 40 and scales == {2,3,4} and counts == {2,3,4,5}
    assert min(collision,reflection,rotation,shared,noise) >= 20
    p = generate_puzzle(918273,7,3)
    q = generate_puzzle(918273,7,3)
    assert all(np.array_equal(a[k],b[k]) for s in p for a,b in zip(p[s],q[s]) for k in a)
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            generate_puzzle(1000+a*10+b,a,b)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3,False),(2.5,1),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0,0] = (int(bad["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Change donor colors in a valid input: matching changes the inherited bands,
    # while geometry and literal stripe colors are exactly the same.
    assert donor_fixture is not None and repeated_fixture is not None
    pair, info = donor_fixture
    a, b = sorted(set(info["colors"]))[:2]
    swapped = pair["input"].copy()
    swapped[pair["input"] == a] = b
    swapped[pair["input"] == b] = a
    replacement = solve(swapped)
    assert not np.array_equal(replacement, pair["output"])
    # A partial macrocell is not a piece, even when its color is legitimate.
    broken = pair["input"].copy()
    broken[info["piece_cells"][0][0]] = info["background"]
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("broken macrocell accepted")
    # Recolor just one of two congruent pieces: there is no positional tie-break.
    pair, info = repeated_fixture
    ks = [_canonical(s) for s in info["pieces"]]
    index = next(i for i, k in enumerate(ks) if ks.count(k) > 1)
    used = {info["background"], info["structural"]} | set(info["code"]) | set(info["piece_colors"])
    new_color = next(c for c in range(10) if c not in used)
    ambiguous = pair["input"].copy()
    for r, c in info["piece_cells"][index]:
        ambiguous[r, c] = new_color
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("different-colored congruent pieces accepted")
    # Exact corner, finite horizon, consensus tie and marker-dependent seam fixture.
    holes = [((2,2),),((2,6),)]
    code = (7,7,9,8,9,9)
    ans = _cast_by_fronts((9,9),holes,[1,2],code,9)
    assert ans[2,4] == 9 and ans[1,4] == 9  # d=2, competing owner colors
    assert ans[2,3] == 8 and ans[0,2] == 1  # literal d=1; owner d=2
    assert ans[0,0] == 1  # square, not diamond, distance 2
    agree = _cast_by_fronts((9,9),holes,[1,1],code,9)
    assert agree[2,4] == 1
    assert np.array_equal(ans,_cast_by_distance((9,9),holes,[1,2],code,9))
    # Literal bands still paint a tie; beyond the finite code stays structural.
    literal = _cast_by_fronts((13,13),holes,[1,2],(7,8,9,9,9),9)
    assert literal[2,4] == 8 and literal[12,12] == 9
