"""Pens that read the walls: contour depth addresses a colored stroke program.

Only NumPy and the standard library are used. No parent module is imported.
The four glyph meanings are fixed; all literal color roles are input-local.
"""
from collections import deque
import hashlib
import json
import numpy as np


# Occupancy masks inherited from A, in order R3, L2, D2, L4.
MASKS = np.array([[[1,1,0],[1,0,1],[0,1,0]],
                 [[1,0,1],[1,0,1],[1,1,1]],
                 [[1,0,1],[0,1,0],[0,1,0]],
                 [[1,1,1],[0,1,0],[1,0,1]]], dtype=np.int8)
STROKES = ((0,1,3),(0,-1,2),(1,0,2),(0,-1,4))
D4 = ((-1,0),(1,0),(0,-1),(0,1))
D8 = tuple((r,c) for r in (-1,0,1) for c in (-1,0,1) if r or c)
HOST_SEEDS = (0,502107782,243299664,869300479,589191762,988760411,
              454570841,262068982,787239614,82801680,363420886,390718864,
              539762507,525868190,74444175,394663365,952368781,197936593,
              300688814,400185143,904217128,787387662,959221236,486238931,
              679892912,233939090,536279650,145050781,275417997,185115988,
              3026072,730461622,107153589,289767838,766914745,970654717,
              585781273,596732764,400851670,559300235,764187649,983894437)


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool,np.bool_)) or not isinstance(value, (int,np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(name + " out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _inside(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r,c])


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    for r,c in np.argwhere(mask):
        r,c = int(r),int(c)
        if seen[r,c]:
            continue
        part = np.zeros(mask.shape, bool)
        queue = [(r,c)]
        seen[r,c] = True
        for y,x in queue:
            part[y,x] = True
            for dy,dx in D4:
                yy,xx = y+dy,x+dx
                if _inside(mask,yy,xx) and not seen[yy,xx]:
                    seen[yy,xx] = True
                    queue.append((yy,xx))
        result.append(part)
    return result


def _depth(mask, diagonal=True):
    """Repeated simultaneous erosion. Outside the board counts as absent."""
    current = mask.copy()
    depth = np.full(mask.shape,-1,np.int8)
    h,w = mask.shape
    level = 0
    while current.any():
        padded = np.pad(current,1)
        inner = current.copy()
        for dr,dc in (D8 if diagonal else D4):
            inner &= padded[1+dr:1+dr+h,1+dc:1+dc+w]
        depth[current & ~inner] = level
        current = inner
        level += 1
    return depth


def _bfs_depth(mask):
    """A second depth calculation for validation, not used by solve."""
    result = np.full(mask.shape,-1,np.int8)
    queue = deque()
    for r,c in np.argwhere(mask):
        r,c = int(r),int(c)
        if any(not _inside(mask,r+dr,c+dc) for dr,dc in D8):
            result[r,c] = 0
            queue.append((r,c))
    while queue:
        r,c = queue.popleft()
        for dr,dc in D8:
            rr,cc = r+dr,c+dc
            if _inside(mask,rr,cc) and result[rr,cc] < 0:
                result[rr,cc] = result[r,c]+1
                queue.append((rr,cc))
    return result


def _corner_paths(mask, depth):
    paths = []
    for r,c in np.argwhere(mask & (depth == 0)):
        r,c = int(r),int(c)
        for dr,dc in ((1,1),(1,-1),(-1,1),(-1,-1)):
            if (_inside(mask,r-dr,c) or _inside(mask,r,c-dc)
                or not _inside(mask,r+dr,c) or not _inside(mask,r,c+dc)):
                continue
            points = []
            for d in range(15):
                rr,cc = r+d*dr,c+d*dc
                if not _inside(mask,rr,cc) or depth[rr,cc] != d:
                    break
                points.append((rr,cc))
            if points:
                paths.append((tuple(points),(dr,dc)))
    return paths


def _read(input_grid):
    inp = _grid(input_grid)
    assert inp.shape[0] >= 15 and inp.shape[1] >= 17
    wall,bg = int(inp[0,0]),int(inp[4,1])
    assert wall != bg
    assert np.all(inp[0] == wall) and np.all(inp[:,0] == wall)
    assert np.all(inp[:,-1] == wall) and np.all(inp[5] == wall)
    assert np.all(inp[4,1:-1] == bg)
    panel = inp[1:4,1:-1]
    occupied = np.flatnonzero(np.any(panel != bg,axis=0))
    runs = []
    for col in occupied:
        col = int(col)
        if not runs or col != runs[-1][-1]+1:
            runs.append([col])
        else:
            runs[-1].append(col)
    assert len(runs) == 4 and all(len(run) == 3 for run in runs)
    lookup, glyphs = {}, set()
    for run in runs:
        block = panel[:,run[0]:run[-1]+1]
        ink = np.unique(block[block != bg])
        assert len(ink) == 1 and int(ink[0]) not in (bg,wall)
        matches = [k for k,m in enumerate(MASKS) if np.array_equal(block != bg,m)]
        assert len(matches) == 1
        k,color = matches[0],int(ink[0])
        assert k not in glyphs and color not in lookup
        glyphs.add(k)
        lookup[color] = k
    assert glyphs == set(range(4))
    board = inp[5:].copy()  # Keep separator as the board's top frame.
    assert np.all(board[-1] == wall)
    extra = set(map(int,np.unique(board))) - {bg,wall} - set(lookup)
    assert len(extra) == 1, "one visible pen-marker color is required"
    marker = extra.pop()
    wall_mask = board == wall
    assert len(_components(wall_mask)) == 1
    assert not np.any(wall_mask[:-1,:-1] & wall_mask[1:,:-1]
                      & wall_mask[:-1,1:] & wall_mask[1:,1:])
    regions = []
    for mask in _components(~wall_mask):
        seeds = np.argwhere(mask & (board == marker))
        assert seeds.shape == (1,2), "each region has exactly one pen"
        seed = tuple(map(int,seeds[0]))
        depth = _depth(mask)
        cues = np.argwhere(mask & (board != bg) & (board != marker))
        assert len(cues) > 0
        assert all(int(board[r,c]) in lookup for r,c in cues)
        length = 1+max(int(depth[r,c]) for r,c in cues)
        assert 1 <= length <= 4
        candidates = []
        for points,direction in _corner_paths(mask,depth):
            if len(points) >= length and set(map(tuple,cues)) == set(points[:length]):
                period = tuple(int(board[q]) for q in points[:length])
                candidates.append((period,direction))
        assert candidates, "key must be a complete inward corner diagonal"
        periods = {period for period,_ in candidates}
        assert len(periods) == 1
        period = periods.pop()
        assert len(set(period)) == length, "period entries are distinct command colors"
        regions.append({"mask":mask,"depth":depth,"period":period,"seed":seed,
                        "directions":tuple(sorted({d for _,d in candidates}))})
    assert 1 <= len(regions) <= 4
    return {"board":board,"bg":bg,"wall":wall,"marker":marker,
            "lookup":lookup,"regions":regions}


def _walk(mask, depth, period, lookup, seed, *, fault=None):
    """Return paint and aggregate diagnostics; no execution trace is stored."""
    ink = np.full(mask.shape,-1,np.int8)
    r,c = seed
    used,full,queries = set(),set(),set()
    clipped = False
    steps = 0
    for _ in range(mask.shape[0]):
        r += 1
        if not _inside(mask,r,c):
            break
        d = int(depth[r,c])
        index = steps if fault == "clock" else 0 if fault == "flat" else d
        color = period[index % len(period)]
        command = lookup[color]
        dr,dc,length = STROKES[2 if fault == "no_glyph" else command]
        used.add(command)
        queries.add(d)
        steps += 1
        cells = []
        for j in range(length):
            rr,cc = r+j*dr,c+j*dc
            if not _inside(mask,rr,cc):
                clipped = True
                break
            cells.append((rr,cc))
            ink[rr,cc] = color
        if clipped:
            break
        full.add(command)
        r,c = cells[-1]
    else:
        raise AssertionError("downward progress failed")
    return ink,{"used":used,"full":full,"queries":queries,
                "steps":steps,"clipped":clipped}


def _render(info, *, fault=None):
    board = info["board"]
    out = np.full(board.shape,info["bg"],np.int8)
    out[board == info["wall"]] = info["wall"]
    for region in info["regions"]:
        mask,depth = region["mask"],region["depth"]
        if fault == "manhattan":
            depth = _depth(mask,False)
        elif fault == "box":
            rr,cc = np.indices(mask.shape)
            ys,xs = np.nonzero(mask)
            depth = np.minimum.reduce((rr-ys.min(),ys.max()-rr,cc-xs.min(),xs.max()-cc))
        ink,_ = _walk(mask,depth,region["period"],info["lookup"],region["seed"],fault=fault)
        out[ink >= 0] = ink[ink >= 0]
        out[region["seed"]] = info["marker"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only the given input; no seed, examples or generation state."""
    return _render(_read(input_grid))


def _geometry(rng, h, w, count, notches, long_notches):
    wall = np.zeros((h,w),bool)
    wall[[0,-1],:] = True
    wall[:,[0,-1]] = True
    rooms = [(1,h-2,1,w-2)]
    while len(rooms) < count:
        options = [(i,axis) for i,(a,b,c,d) in enumerate(rooms)
                   for axis,n in ((0,b-a+1),(1,d-c+1)) if n >= 15]
        if not options:
            return None
        i,axis = options[int(rng.integers(len(options)))]
        a,b,c,d = rooms.pop(i)
        if axis == 0:
            q = int(rng.integers(a+7,b-6))
            wall[q,c-1:d+2] = True
            rooms += [(a,q-1,c,d),(q+1,b,c,d)]
        else:
            q = int(rng.integers(c+7,d-6))
            wall[a-1:b+2,q] = True
            rooms += [(a,b,c,q-1),(a,b,q+1,d)]
    for _ in range(notches):
        for _attempt in range(16):
            a,b,c,d = rooms[int(rng.integers(len(rooms)))]
            side = int(rng.integers(4))
            if side < 2:
                x = int(rng.integers(c+2,d-1))
                y,dy,dx = (a,1,0) if side == 0 else (b,-1,0)
            else:
                y = int(rng.integers(a+2,b-1))
                x,dy,dx = (c,0,1) if side == 2 else (d,0,-1)
            length = int(rng.integers(1,7 if long_notches else 4))
            points = [(y+j*dy,x+j*dx) for j in range(length)]
            if rng.random() < .65:
                sign = int(rng.choice((-1,1)))
                ey,ex = points[-1]
                points += [(ey+j*dx*sign,ex+j*dy*sign)
                           for j in range(1,int(rng.integers(2,5)))]
            candidate = wall.copy()
            valid = True
            for j,(yy,xx) in enumerate(points):
                if not (a <= yy <= b and c <= xx <= d) or candidate[yy,xx]:
                    valid = False
                    break
                neighbors = sum(_inside(candidate,yy+dr,xx+dc) for dr,dc in D4)
                if neighbors != 1:
                    valid = False
                    break
                candidate[yy,xx] = True
            if valid and len(_components(~candidate)) == count:
                wall = candidate
                break
    return wall


def _make_pair(rng, index, test, roles, seen):
    bg,wall_color,marker,*colors = roles
    lookup = {int(color):k for k,color in enumerate(colors)}
    count = 2 if not test and index == 0 else 1 if not test and index == 1 else int(rng.integers(1,4))
    for _attempt in range(48):
        h = int(rng.integers(23,26) if test else rng.integers(19,25))
        w = int(rng.integers(24,31) if test else rng.integers(23,31))
        wall = _geometry(rng,h,w,count,int(rng.integers(3,7) if test else rng.integers(1,4)),test)
        if wall is None:
            continue
        masks = _components(~wall)
        depths = [_depth(mask) for mask in masks]
        lengths = ([2,3] if not test and index == 0 else
                   [4]+[int(rng.integers(1,5)) for _ in masks[1:]] if test else
                   [3] if index == 1 else [int(rng.integers(1,4)) for _ in masks])
        paths = [[p for p in _corner_paths(mask,depth) if len(p[0]) >= length]
                 for mask,depth,length in zip(masks,depths,lengths)]
        if not all(paths):
            continue
        board = np.full((h,w),bg,np.int8)
        board[wall] = wall_color
        permutation = [int(x) for x in rng.permutation(colors)]
        regions = []
        failed = False
        for i,(mask,depth,length,options) in enumerate(zip(masks,depths,lengths,paths)):
            if not test and index == 0:
                values = permutation[:2] if i == 0 else permutation[1:]
                period = tuple(int(x) for x in rng.permutation(values))
            else:
                period = tuple(int(x) for x in rng.choice(colors,length,replace=False))
            points,direction = options[int(rng.integers(len(options)))]
            key = points[:length]
            for q,color in zip(key,period):
                board[q] = color
            candidates = [(int(r),int(c)) for r,c in np.argwhere(mask & (depth >= 1))
                          if r <= h//2 and (int(r),int(c)) not in key]
            if not candidates:
                failed = True
                break
            best = None
            box = np.full(mask.shape,-1,np.int8)
            rr,cc = np.indices(mask.shape)
            ys,xs = np.nonzero(mask)
            box[mask] = np.minimum.reduce((rr-ys.min(),ys.max()-rr,cc-xs.min(),xs.max()-cc))[mask]
            l1 = _depth(mask,False)
            for j in rng.permutation(len(candidates))[:80]:
                seed = candidates[int(j)]
                ink,stats = _walk(mask,depth,period,lookup,seed)
                required = {lookup[color] for color in period}
                if not test and index < 2 and stats["full"] != required:
                    continue
                # First demonstration must execute a genuinely repeated cycle.
                if not test and index == 0 and length == 3 and max(stats["queries"],default=-1) < 3:
                    continue
                witness = sum(not np.array_equal(ink,_walk(mask,alternate,period,lookup,seed)[0])
                              for alternate in (box,l1))
                score = (len(stats["full"]),witness,len(stats["queries"]),stats["steps"])
                if best is None or score > best[0]:
                    best = (score,seed)
            if best is None:
                failed = True
                break
            seed = best[1]
            board[seed] = marker
            regions.append({"mask":mask,"depth":depth,"period":period,"seed":seed,
                            "directions":(direction,)})
        if failed:
            continue
        # Expose prefix painting at a collision even with only two demos.
        if not test and index == 0 and not any(
                _walk(r["mask"],r["depth"],r["period"],lookup,r["seed"])[1]["clipped"]
                for r in regions):
            continue
        # Header position and order are deliberately irrelevant to command order.
        inp = np.full((h+5,w),bg,np.int8)
        inp[0,:] = wall_color
        inp[:,[0,-1]] = wall_color
        gap = int(rng.integers(1,3))
        width = 12+3*gap
        left = int(rng.integers(1,w-width))
        for j,k in enumerate(rng.permutation(4)):
            block = inp[1:4,left+j*(3+gap):left+j*(3+gap)+3]
            block[MASKS[int(k)].astype(bool)] = colors[int(k)]
        inp[5:] = board
        info = {"board":board,"bg":bg,"wall":wall_color,"marker":marker,
                "lookup":lookup,"regions":regions}
        output = _render(info)
        if not test and index == 0 and any(np.array_equal(output,_render(info,fault=f))
                                          for f in ("no_glyph","flat","clock","box","manhattan")):
            continue
        identity = (inp.shape,inp.tobytes())
        if identity in seen:
            continue
        seen.add(identity)
        return {"input":inp,"output":output}
    raise RuntimeError("bounded geometry/pen search exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed,"seed",0)
    nt = _integer(num_train,"num_train",2,10)
    nq = _integer(num_test,"num_test",1,4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train":[],"test":[]}
    seen = set()
    # Rotate a random role permutation so the first two demonstrations change
    # background, wall, pen and every glyph's color, without hidden conventions.
    palette = [int(x) for x in rng.permutation(10)]
    for i in range(nt+nq):
        offset = i % 10
        roles = (palette[offset:]+palette[:offset])[:7]
        test = i >= nt
        pair = _make_pair(rng,i-nt if test else i,test,roles,seen)
        puzzle["test" if test else "train"].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen,infos = set(),{"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            key = (inp.shape,inp.tobytes())
            assert key not in seen
            seen.add(key)
            info = _read(inp)
            assert np.array_equal(out,_render(info)), "wrong pen trajectory or color"
            assert out.shape == (inp.shape[0]-5,inp.shape[1])
            assert np.array_equal(out == info["wall"],info["board"] == info["wall"])
            assert np.array_equal(out == info["marker"],info["board"] == info["marker"])
            assert np.any((out != info["bg"]) & (out != info["wall"]) & (out != info["marker"]))
            for r in info["regions"]:
                assert np.array_equal(r["depth"],_bfs_depth(r["mask"]))
                ink,stats = _walk(r["mask"],r["depth"],r["period"],info["lookup"],r["seed"])
                assert stats["steps"] > 0 and np.all(r["mask"][ink >= 0])
                assert set(map(int,np.unique(ink[ink >= 0]))) <= set(r["period"])
                if split == "train":
                    assert len(r["period"]) <= 3
            if split == "test":
                assert any(len(r["period"]) == 4 for r in info["regions"])
            infos[split].append(info)
    first = infos["train"][0]
    assert len(first["regions"]) == 2 and sorted(len(r["period"]) for r in first["regions"]) == [2,3]
    full = set()
    clipped_demo = False
    for region in first["regions"]:
        _,stats = _walk(region["mask"],region["depth"],region["period"],first["lookup"],region["seed"])
        full |= stats["full"]
        clipped_demo |= stats["clipped"]
        if len(region["period"]) == 3:
            assert max(stats["queries"]) >= 3
    assert full == set(range(4)), "demonstrations must expose all stroke meanings"
    assert clipped_demo, "demonstration must show a partially painted collision"
    assert len(infos["train"][1]["regions"]) == 1
    for feature in ("bg","wall","marker"):
        assert first[feature] != infos["train"][1][feature]
    for fault in ("no_glyph","flat","clock","box","manhattan"):
        assert not np.array_equal(_render(first),_render(first,fault=fault)), "missing shortcut witness: "+fault


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train","test"):
        for pair in puzzle[split]:
            for role in ("input","output"):
                grid = np.asarray(pair[role],np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, streaming summaries, count boundaries and semantic mutations."""
    shapes,geometries,puzzles,periods,counts,directions,palettes = (set() for _ in range(7))
    pairs,clipped,concave,steps = 0,0,0,set()
    last = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        puzzles.add(_fingerprint(puzzle))
        for split in ("train","test"):
            for pair in puzzle[split]:
                pairs += 1
                info = _read(pair["input"])
                shapes.add(tuple(pair["input"].shape))
                wall = info["board"] == info["wall"]
                geometries.add((wall.shape,wall.tobytes()))
                palettes.add((info["bg"],info["wall"],info["marker"]))
                counts.add(len(info["regions"]))
                for r in info["regions"]:
                    periods.add(len(r["period"]))
                    directions.update(r["directions"])
                    yy,xx = np.nonzero(r["mask"])
                    concave += int(len(yy) < (int(yy.max()-yy.min())+1)*(int(xx.max()-xx.min())+1))
                    _,stats = _walk(r["mask"],r["depth"],r["period"],info["lookup"],r["seed"])
                    clipped += int(stats["clipped"])
                    steps.add(stats["steps"])
        last = puzzle
    assert len(puzzles) == 42 and len(geometries) >= 100 and len(shapes) >= 25
    assert periods == {1,2,3,4} and counts >= {1,2,3}
    assert directions == {(1,1),(1,-1),(-1,1),(-1,-1)}
    assert concave >= 100 and clipped >= 20 and len(palettes) >= 25 and len(steps) >= 8
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            validate_generated_puzzle(generate_puzzle(71,nt,nq))
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(2.5,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Destroy actual trail paint, a preserved wall, and a key cue in turn.
    for damage in ("trail","wall","key"):
        broken = {s:[{k:v.copy() for k,v in pair.items()} for pair in last[s]] for s in last}
        p = broken["test"][0]
        info = _read(p["input"])
        if damage == "wall":
            p["output"][0,0] = info["bg"]
        elif damage == "trail":
            ink = (p["output"] != info["bg"]) & (p["output"] != info["wall"]) & (p["output"] != info["marker"])
            q = tuple(np.argwhere(ink)[0])
            p["output"][q] = info["bg"]
        else:
            region = next(r for r in info["regions"] if len(r["period"]) == 4)
            cues = region["mask"] & np.isin(info["board"],list(info["lookup"]))
            q = tuple(np.argwhere(cues & (region["depth"] == 1))[0])
            p["input"][q[0]+5,q[1]] = info["bg"]
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("semantic mutation accepted: "+damage)
    return {"seeds":42,"pairs":pairs,"distinct_puzzles":len(puzzles),
            "input_shapes":len(shapes),"wall_geometries":len(geometries),
            "period_lengths":sorted(periods),"region_counts":sorted(counts),
            "key_directions":sorted(directions),"role_assignments":len(palettes),
            "concave_regions":concave,"clipped_strokes":clipped,
            "stroke_counts":sorted(steps),"all_legal_count_combinations":30,
            "semantic_mutations_rejected":3,"independent_solver":False}


if __name__ == "__main__":
    print(json.dumps(self_test(),separators=(",", ":")))
