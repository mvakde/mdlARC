import numpy as np


# Original inputs are used only for exact-copy exclusion; dimensions remain legal.
_SOURCE_INPUTS = (
    ("00000000000000", "00000007213000", "00000007213000", "00000007213000",
     "00000000000000", "00000000000000", "00000000000000", "00000000000000",
     "00020020000000", "00222222000000", "00002200000000", "00010010000000",
     "00111111000000", "00001100000000", "00000000000000", "00000000000000",
     "00000000000000", "00000000000000"),
    ("0000000000000", "0041600000000", "0041600000000", "0041600000000",
     "0000000000000", "0000000000000", "0000000000000", "0001111000000",
     "0001001000000", "0111001110000", "0011001100000", "0011111100000",
     "0006666000000", "0006006000000", "0666006660000", "0066006600000",
     "0066666600000"),
    ("000000000000000", "000000000000000", "000000000000000", "000000000000000",
     "000000000000000", "000000000000000", "000001100000000", "000010010000000",
     "000111111000000", "000000000000000", "000000000000000", "000000000000000",
     "000000000000000", "000000000000000", "000000002318700", "000000002318700",
     "000000002318700", "000000000000000"),
)


def _check_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    low, high = (2, 10) if name == "num_train" else (1, 4)
    if not low <= value <= high:
        raise ValueError(f"{name} is outside {low}..{high}")
    return value


def _choose_counts(rng, num_train, num_test):
    n = _check_count(num_train, "num_train")
    t = _check_count(num_test, "num_test")
    if n is None and t is None:
        t = int(rng.integers(1, 5))
        n = int(rng.integers(t + 1, 11))
    elif n is None:
        n = int(rng.integers(max(2, t + 1), 11))
    elif t is None:
        t = int(rng.integers(1, min(4, n - 1) + 1))
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    return n, t


def _grid_key(grid):
    return (grid.shape, grid.astype(np.int8, copy=False).tobytes())


def _is_source_input(grid):
    for rows in _SOURCE_INPUTS:
        if grid.shape == (len(rows), len(rows[0])):
            if all("".join(map(str, row)) == expected for row, expected in zip(grid, rows)):
                return True
    return False


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    sizes = []
    while unseen:
        start = unseen.pop()
        todo = [start]
        size = 1
        while todo:
            r, c = todo.pop()
            for p in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if p in unseen:
                    unseen.remove(p)
                    todo.append(p)
                    size += 1
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _has_hole(mask):
    empty = set(map(tuple, np.argwhere(~mask)))
    height, width = mask.shape
    reached = {p for p in empty if p[0] in (0, height-1) or p[1] in (0, width-1)}
    todo = list(reached)
    while todo:
        r, c = todo.pop()
        for p in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
            if p in empty and p not in reached:
                reached.add(p)
                todo.append(p)
    return bool(empty-reached)


def _sample_mask(rng, height, width):
    """Compose compact outlined, stepped, branched and lightly detached forms."""
    family = int(rng.integers(0, 4))
    mask = np.zeros((height, width), dtype=bool)
    if family == 0:  # smoothly changing row spans, possibly with a small hole
        left = int(rng.integers(0, max(1, width // 3)))
        right = int(rng.integers(max(left + 2, 2 * width // 3), width + 1))
        for r in range(height):
            left = int(np.clip(left + rng.integers(-1, 2), 0, width - 2))
            right = int(np.clip(right + rng.integers(-1, 2), left + 2, width))
            mask[r, left:right] = True
        if height >= 4 and width >= 5 and rng.random() < .7:
            r = int(rng.integers(1, height - 1))
            occupied = np.flatnonzero(mask[r])
            if len(occupied) >= 4:
                mask[r, int(occupied[len(occupied)//2])] = False
    elif family == 1:  # loop or open outline, with irregular edges
        mask[0, 1:width-1] = True
        mask[-1, 1:width-1] = True
        mask[1:height-1, 0] = True
        mask[1:height-1, -1] = True
        mask[1, 1] = True
        mask[1, -2] = True
        mask[-2, 1] = True
        mask[-2, -2] = True
        if height >= 5 and width >= 5:
            side = int(rng.integers(0, 2))
            mask[int(rng.integers(1, height-1)), 0 if side == 0 else width-1] = False
            mask[int(rng.integers(1, height-1)), 1 if side == 0 else width-2] = True
        if rng.random() < .55:
            mask[int(rng.integers(1, height-1)), 1:width-1] = True
    elif family == 2:  # spine and several unequal arms
        mid = int(rng.integers(1, width-1))
        mask[:, mid] = True
        for r in range(height):
            if rng.random() < .85:
                reach = int(rng.integers(1, width))
                lo = max(0, mid - int(rng.integers(0, reach+1)))
                hi = min(width, mid + reach + 1)
                mask[r, lo:hi] = True
        if rng.random() < .5:
            mask[int(rng.integers(0, height)), :] = True
    else:  # one coherent body and a restrained detached mark
        core_right = max(2, width-2)
        mid = min(core_right-1, max(0, core_right//2))
        mask[:, mid] = True
        for r in range(height):
            lo = int(rng.integers(0, mid+1))
            hi = int(rng.integers(mid+1, core_right+1))
            mask[r, lo:hi] = True
        mask[0, -1] = True
        if height >= 5 and rng.random() < .35:
            mask[-1, -1] = True
    rr, cc = np.nonzero(mask)
    mask = mask[rr.min():rr.max()+1, cc.min():cc.max()+1]
    if mask.all():
        mask[0, 0] = False
    return mask


def _draw_mask(grid, top, left, mask, color):
    view = grid[top:top+mask.shape[0], left:left+mask.shape[1]]
    view[mask] = color


def _choose_run(rng, length, regime, index):
    if regime == "source_train" and index == 0:
        # The first demonstration always gives an adjacent pair and both phases.
        return 1, 2
    if regime == "target_test":
        count = 1 if rng.random() < .5 else int(rng.integers(2, length))
    else:
        count = 1 if rng.random() < .35 else int(rng.integers(2, length))
    return int(rng.integers(0, length-count+1)), count


def _placement(rng, mask, length, start, count, style):
    h, w = mask.shape
    stack_h = h * length
    visible = np.zeros((stack_h, w), dtype=bool)
    full = np.tile(mask, (length, 1))
    for i in range(start, start+count):
        visible[i*h:(i+1)*h] = mask
    candidates = []
    for y in range(-4, stack_h+3):
        for x in range(-length-4, w+4):
            if max(stack_h, y+3)-min(0, y) > 28 or max(w, x+length)-min(0, x) > 28:
                continue
            overlap = 0
            blocked = False
            for dy in range(3):
                yy = y+dy
                if 0 <= yy < stack_h:
                    for dx in range(length):
                        xx = x+dx
                        if 0 <= xx < w:
                            if visible[yy, xx]:
                                blocked = True
                                break
                            overlap += int(full[yy, xx])
                    if blocked:
                        break
            if blocked:
                continue
            beside = x+length <= 0 or x >= w
            straddle = (y < 0 < y+3 or y < stack_h < y+3 or
                        x < 0 < x+length or x < w < x+length)
            if style == 0 and overlap == 0 and beside:
                candidates.append((y,x,overlap))
            elif style == 1 and overlap == 0 and not beside:
                candidates.append((y,x,overlap))
            elif style == 2 and overlap == 1:
                candidates.append((y,x,overlap))
            elif style == 3 and 2 <= overlap <= 4:
                candidates.append((y,x,overlap))
            elif style == 4 and overlap >= 5:
                candidates.append((y,x,overlap))
            elif style == 5 and straddle and overlap >= 1:
                candidates.append((y,x,overlap))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _generate_pair(rng, regime, orientation, background, index):
    if regime == "source_train" and index == 0:
        length = int(rng.integers(4, 7))
    elif regime == "target_test":
        length = int(rng.choice(np.array([3,4,5,5,6,6])))
    else:
        length = int(rng.integers(3, 7))
    start, count = _choose_run(rng, length, regime, index)
    motif_h = int(rng.integers(3, min(8, 26//length + 1)))
    motif_w = int(rng.integers(3, 12))
    if motif_w == 3 and rng.random() < .5:
        motif_h = max(motif_h, 4)
    mask = _sample_mask(rng, motif_h, motif_w)
    colors = rng.choice(np.array([c for c in range(10) if c != background], dtype=np.int8),
                        size=length, replace=False)
    if regime == "source_train" and index == 0:
        styles = (2,3,4,5)
    elif regime == "source_train" and index == 1:
        styles = (0,1)
    else:
        styles = (0,1,2,3,4,5)
    style = int(rng.choice(np.array(styles)))
    placed = _placement(rng, mask, length, start, count, style)
    if placed is None:
        for alt in (3,2,5,4,1,0):
            placed = _placement(rng, mask, length, start, count, alt)
            if placed is not None:
                break
    if placed is None:
        raise ValueError("no palette placement")
    py, px, _ = placed
    stack_h, w = length*mask.shape[0], mask.shape[1]
    min_y, min_x = min(0,py), min(0,px)
    span_h = max(stack_h,py+3)-min_y
    span_w = max(w,px+length)-min_x
    top_pad = int(rng.integers(1, min(5,30-span_h)))
    left_pad = int(rng.integers(1, min(5,30-span_w)))
    extra_h = int(rng.integers(1, min(6,31-span_h-top_pad)))
    extra_w = int(rng.integers(1, min(6,31-span_w-left_pad)))
    sh, sw = top_pad-min_y, left_pad-min_x
    ph, pw = py+sh, px+sw
    grid_h, grid_w = span_h+top_pad+extra_h, span_w+left_pad+extra_w
    inp = np.full((grid_h,grid_w), background, dtype=np.int8)
    out = np.full_like(inp, background)
    for i,color in enumerate(colors):
        _draw_mask(out, sh+i*mask.shape[0], sw, mask, int(color))
        if start <= i < start+count:
            _draw_mask(inp, sh+i*mask.shape[0], sw, mask, int(color))
    for i,color in enumerate(colors):
        inp[ph:ph+3,pw+i] = int(color)
    return np.rot90(inp, orientation), np.rot90(out, orientation)


def _palette_candidates(grid, bg):
    height, width = grid.shape
    found = []
    for top in range(height-2):
        same = np.all(grid[top:top+3] == grid[top], axis=0)
        for left in range(width-2):
            if not same[left] or grid[top,left] == bg:
                continue
            if left and same[left-1] and grid[top,left-1] != bg:
                continue
            colors = []
            x = left
            while x < width and same[x] and grid[top,x] != bg and int(grid[top,x]) not in colors:
                colors.append(int(grid[top,x]))
                x += 1
            if 3 <= len(colors) <= 6:
                found.append((top,left,tuple(colors)))
    return found


def _parse_canonical(inp, top, left, colors, bg):
    length = len(colors)
    clean = inp.copy()
    clean[top:top+3,left:left+length] = bg
    if not set(map(int,np.unique(clean))) <= {bg,*colors}:
        return None
    active = [i for i,color in enumerate(colors) if np.any(clean == color)]
    if not active or len(active) == length or active != list(range(active[0],active[-1]+1)):
        return None
    copies = []
    for i in active:
        positions = np.argwhere(clean == colors[i])
        r0,c0 = positions.min(axis=0)
        r1,c1 = positions.max(axis=0)+1
        region = clean[r0:r1,c0:c1]
        if np.any((region != bg) & (region != colors[i])):
            return None
        copies.append((i,int(r0),int(c0),region == colors[i]))
    anchor_i, anchor_r, anchor_c, mask = copies[0]
    mh,mw = mask.shape
    if mh < 3 or mw < 2 or np.count_nonzero(mask) < 5 or np.all(mask):
        return None
    for i,r,c,other in copies:
        if r-anchor_r != (i-anchor_i)*mh or c != anchor_c or not np.array_equal(mask,other):
            return None
    stack_top = anchor_r-anchor_i*mh
    if stack_top < 0 or stack_top+length*mh > inp.shape[0]:
        return None
    observation = np.full_like(inp,bg)
    output = np.full_like(inp,bg)
    for i,color in enumerate(colors):
        _draw_mask(output,stack_top+i*mh,anchor_c,mask,color)
        if i in active:
            _draw_mask(observation,stack_top+i*mh,anchor_c,mask,color)
    if not np.array_equal(clean,observation):
        return None
    if np.array_equal(output,inp):
        return None
    overlap = int(np.count_nonzero(output[top:top+3,left:left+length] != bg))
    info = {"palette_length":length,"visible_start":active[0],"visible_count":len(active),
            "palette_colors":colors,"background":bg,"motif_height":mh,"motif_width":mw,
            "mask":mask,"components":len(_components(mask)),"full_row":bool(np.any(np.all(mask,axis=1))),
            "symmetric":bool(np.array_equal(mask,np.fliplr(mask))),"overlap":overlap,
            "palette_rect":(top,left,3,length),"stack_top":stack_top,"stack_left":anchor_c,
            "canvas":inp.shape}
    return output,info


def _analyze_pair(inp, out=None):
    bg = int(inp[0,0])
    if any(int(inp[r,c]) != bg for r,c in ((0,-1),(-1,0),(-1,-1))):
        raise AssertionError("background corners differ")
    interpretations = {}
    for rotation in range(4):
        canonical = np.rot90(inp,-rotation)
        for top,left,colors in _palette_candidates(canonical,bg):
            parsed = _parse_canonical(canonical,top,left,colors,bg)
            if parsed is None:
                continue
            prediction,info = parsed
            prediction = np.rot90(prediction,rotation)
            key = _grid_key(prediction)
            info["orientation"] = rotation
            interpretations.setdefault(key,(prediction,info))
    if len(interpretations) != 1:
        raise AssertionError(f"input has {len(interpretations)} possible completions")
    prediction,info = next(iter(interpretations.values()))
    if out is not None:
        assert np.array_equal(out,prediction)
    return info,prediction


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n,t = _choose_counts(rng,num_train,num_test)
    orientation = int(rng.integers(0,4))
    backgrounds = rng.permutation(10)
    child_seeds = rng.integers(0,np.iinfo(np.uint64).max,size=n+t,dtype=np.uint64)
    puzzle = {"train":[],"test":[]}
    seen = set()
    for serial,(section,index) in enumerate([("train",i) for i in range(n)]+[("test",i) for i in range(t)]):
        pair_rng = np.random.default_rng(int(child_seeds[serial]))
        regime = "source_train" if section == "train" else "target_test"
        background = int(backgrounds[serial%10])
        for attempt in range(80):
            inp,out = _generate_pair(pair_rng,regime,orientation,background,index)
            key = _grid_key(inp)
            if key in seen or _is_source_input(inp):
                continue
            try:
                _analyze_pair(inp,out)
            except AssertionError:
                continue
            seen.add(key)
            puzzle[section].append({"input":inp,"output":out})
            break
        else:
            raise ValueError(f"could not construct {section}[{index}]")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert isinstance(puzzle["train"],list) and isinstance(puzzle["test"],list)
    n,t = len(puzzle["train"]),len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen,infos = set(),{"train":[],"test":[]}
    for section in ("train","test"):
        for pair in puzzle[section]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            for grid in pair.values():
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype,np.integer) and np.all((0 <= grid)&(grid <= 9))
            inp,out = pair["input"],pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            assert not _is_source_input(inp)
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            info,_ = _analyze_pair(inp,out)
            infos[section].append(info)
    # Rotation parity fixes the stack/palette axis for the entire puzzle.
    assert len({info["orientation"]%2 for v in infos.values() for info in v}) == 1
    assert infos["train"][0]["visible_count"] >= 2
    assert infos["train"][0]["visible_start"] > 0
    assert infos["train"][0]["visible_start"]+infos["train"][0]["visible_count"] < infos["train"][0]["palette_length"]
    assert infos["train"][0]["overlap"] > 0
    assert infos["train"][1]["overlap"] == 0
    assert all(info["visible_count"] < info["palette_length"] for v in infos.values() for info in v)


def _puzzles_equal(a,b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k],y[k]) for x,y in zip(a[s],b[s]) for k in ("input","output")) for s in ("train","test"))


def self_test() -> None:
    puzzles = [generate_puzzle(i) for i in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_grid_key(p["train"][0]["input"]) for p in puzzles}) == 42
    shapes, masks, lengths, runs, orientations, overlaps, backgrounds = set(),set(),set(),set(),set(),set(),set()
    observations = {"train":[], "test":[]}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_backgrounds = set()
        for section in ("train","test"):
            for pair in puzzle[section]:
                info,predicted = _analyze_pair(pair["input"])
                assert np.array_equal(predicted,pair["output"])
                shapes.add(pair["input"].shape)
                masks.add((info["mask"].shape,info["mask"].tobytes()))
                lengths.add(info["palette_length"])
                runs.add((info["visible_start"],info["visible_count"],info["palette_length"]))
                orientations.add(info["orientation"]%2)
                overlaps.add(min(info["overlap"],5))
                backgrounds.add(info["background"])
                local_backgrounds.add(info["background"])
                py,px,_,length = info["palette_rect"]
                sy,sx = info["stack_top"],info["stack_left"]
                height = info["palette_length"]*info["motif_height"]
                width = info["motif_width"]
                observations[section].append({
                    "phase": "begin" if info["visible_start"] == 0 else
                             "end" if info["visible_start"]+info["visible_count"] == info["palette_length"] else "middle",
                    "count": info["visible_count"],
                    "overlap": info["overlap"],
                    "full_row": info["full_row"],
                    "components": info["components"],
                    "symmetric": info["symmetric"],
                    "slim": info["motif_width"] < info["palette_length"],
                    "tall": info["motif_height"] > info["motif_width"],
                    "hole": _has_hole(info["mask"]),
                    "beside": px+length <= sx or px >= sx+width,
                    "straddle": (py < sy < py+3 or py < sy+height < py+3 or
                                 px < sx < px+length or px < sx+width < px+length),
                })
        assert len(local_backgrounds) >= 2
    assert len(shapes) >= 25 and len(masks) >= 80 and len(runs) >= 12
    assert lengths == {3,4,5,6} and orientations == {0,1}
    assert {0,1,5} <= overlaps and len(backgrounds) >= 8
    for section, records in observations.items():
        assert {r["phase"] for r in records} == {"begin","middle","end"}
        assert any(r["count"] == 1 for r in records)
        assert any(r["count"] >= 3 for r in records)
        assert all(1 <= r["components"] <= 3 for r in records)
        assert {r["full_row"] for r in records} == {False,True}
        assert {r["symmetric"] for r in records} == {False,True}
        assert any(r["slim"] for r in records) and any(r["tall"] for r in records)
        assert any(r["hole"] for r in records)
        assert any(r["beside"] for r in records) and any(r["straddle"] for r in records)
        assert any(r["overlap"] == 0 for r in records)
        assert any(r["overlap"] == 1 for r in records)
        assert any(r["overlap"] >= 5 for r in records)
    for i,(n,t) in enumerate(((2,1),(10,1),(10,4),(6,2),(None,1),(None,4),(2,None),(10,None))):
        p = generate_puzzle(1000+i,n,t)
        assert n is None or len(p["train"]) == n
        assert t is None or len(p["test"]) == t
    a,b,c = (generate_puzzle(20260827,7,3),generate_puzzle(20260827,7,3),generate_puzzle(20260828,7,3))
    assert _puzzles_equal(a,b) and not _puzzles_equal(a,c)
    for n,t in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4)):
        try:
            generate_puzzle(0,n,t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
