"""Synthetic puzzles for ordered, spatially scoped ARC color rewriting."""

import hashlib
import numpy as np

_SOURCE_PAIR_DIGESTS = {
    "61b261b1f699bd5acfdb0974344f0e2427908a3e0465d6dabd9f46f433a",
    "3c8f59d8185fc524d6a097f1a0955436ca84a3b4287abd492038e3dc6444c8e8",
    "7a2e0281c1dc7fd04e920f7158efabab00832dca7a8f4646c1469e1576fb1006",
    "a9c0db9a91f9f10cf67c9157777ce032f9b739f4d68258e4506193a163a196e5",
    "897a7e5582423fc44e7d4b6ed1feb0c5c2f24618049fd5b5c3e73a81328705aa",
}


def _counts(seed, rng, num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not lo <= value <= hi):
            raise ValueError(name)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = max(2, num_test + 1) + (seed // 4) % (11 - max(2, num_test + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _apply(grid, region, rules):
    out = grid.copy()
    for source, destination in rules:
        out[region & (out == source)] = destination
    return out


def _program(rng, palette, count, inert, cyclic):
    # A directed chain is always present, but other edges and its displayed order vary.
    a, b, c, d, e, f = palette
    rules = [(a, b), (b, c), (c, d)]
    if cyclic and count >= 4:
        rules.append((d, a))
    if inert and count >= 4:
        # This row precedes all other rows and will have no matching input mark.
        rules.append((f, e))
    while len(rules) < count:
        if cyclic:
            source = (a, b, c, d, e)[int(rng.integers(0, 5))]
            dest = (a, b, c, d, e, f)[int(rng.integers(0, 6))]
        else:
            i = int(rng.integers(0, 5))
            source = palette[i]
            dest = palette[int(rng.integers(i + 1, 6))]
        if source != dest:
            rules.append((source, dest))
    rng.shuffle(rules)
    if inert:
        rules = [(f, e)] + [r for r in rules if r != (f, e)]
    return tuple(rules[:count]), (f if inert else None)


def _divider(rng, height, width, orientation, strength):
    along = width if orientation == "h" else height
    across = height if orientation == "h" else width
    segments = 1 if strength == 0 else int(rng.integers(2, 4 if strength == 1 else 7))
    cuts = sorted(int(x) for x in rng.choice(np.arange(3, along - 2),
                                               size=segments - 1, replace=False))
    limits = (max(5, across // 3), min(across - 6, 2 * across // 3)) if strength == 1 else (4, across - 5)
    levels = [int(rng.integers(limits[0], limits[1] + 1))]
    for _ in cuts:
        jump = int(rng.integers(2, 5 if strength == 1 else 9))
        sign = -1 if rng.integers(0, 2) else 1
        levels.append(int(np.clip(levels[-1] + sign * jump, *limits)))
    mask = np.zeros((height, width), dtype=bool)
    starts = [0] + cuts
    ends = cuts + [along - 1]
    for lo, hi, level in zip(starts, ends, levels):
        if orientation == "h":
            mask[level, lo:hi + 1] = True
        else:
            mask[lo:hi + 1, level] = True
    for cut, left, right in zip(cuts, levels, levels[1:]):
        if orientation == "h":
            mask[min(left, right):max(left, right) + 1, cut] = True
        else:
            mask[cut, min(left, right):max(left, right) + 1] = True
    if strength == 2 and rng.random() < .4:
        # A one-cell side contact mirrors the local degree-three contacts in
        # the source winding divider without joining the two payload regions.
        candidates = []
        for r, c in np.argwhere(~mask):
            if not (1 < r < height - 2 and 1 < c < width - 2):
                continue
            neighbors = ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
            attached = [(rr,cc) for rr,cc in neighbors if mask[rr,cc]]
            if len(attached) == 1:
                rr,cc = attached[0]
                degree = sum(mask[y,x] for y,x in
                             ((rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)))
                if degree == 2:
                    candidates.append((r,c))
        if candidates:
            r,c = candidates[int(rng.integers(len(candidates)))]
            mask[r,c] = True
    return mask


def _regions(barrier):
    h, w = barrier.shape
    seen = barrier.copy()
    regions = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            component = np.zeros_like(barrier)
            todo = [(r, c)]
            seen[r, c] = True
            while todo:
                y, x = todo.pop()
                component[y, x] = True
                for yy, xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx]:
                        seen[yy, xx] = True
                        todo.append((yy, xx))
            regions.append(component)
    return regions


def _table_sites(region, barrier, length):
    h, w = region.shape
    sites = []
    for row in range(1, h - length):
        for col in range(1, w - 2):
            if not np.all(region[row:row+length, col:col+2]):
                continue
            # A background moat keeps the colored divider and table separate.
            if np.any(barrier[row-1:row+length+1, col-1:col+3]):
                continue
            sites.append((row, col))
    return sites


def _shape(rng, style):
    if style == "dot":
        return [(0, 0)]
    if style == "bar":
        n = int(rng.integers(3, 11))
        return [(0, i) for i in range(n)] if rng.integers(0, 2) else [(i, 0) for i in range(n)]
    if style == "block":
        hh, ww = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        return [(r, c) for r in range(hh) for c in range(ww)]
    if style == "ell":
        hh, ww = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        return list(set([(r, 0) for r in range(hh)] + [(hh-1, c) for c in range(ww)]))
    size = int(rng.integers(7, 20))
    cells = {(0, 0)}
    for _ in range(size - 1):
        frontier = [(r+dr, c+dc) for r, c in cells
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))
                    if (r+dr, c+dc) not in cells and -3 <= r+dr <= 4 and -3 <= c+dc <= 5]
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(len(frontier)))])
    lo_r = min(r for r, _ in cells)
    lo_c = min(c for _, c in cells)
    return [(r-lo_r, c-lo_c) for r, c in cells]


def _place_payload(rng, grid, region, barrier, rules, inert_source, mode,
                   mandatory=None):
    h, w = grid.shape
    valid = region.copy()
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    for r, c in np.argwhere(barrier):
        valid[max(0,r-1):min(h,r+2), max(0,c-1):min(w,c+2)] = False
    occupied = np.zeros_like(valid)
    active = [color for color in range(10) if color != int(grid[0, 0]) and
              color != inert_source and
              _apply(np.array([[color]], dtype=np.int8), np.array([[True]]), rules)[0,0] != color]
    if mandatory is not None and mandatory not in active:
        raise AssertionError("shared probe is not active")
    unused = [color for color in range(10) if color != int(grid[0, 0]) and
              color not in {s for s, _ in rules} and color != inert_source]
    if not active:
        raise AssertionError("program lacks an active probe")
    count = int(rng.integers(3, 7)) if mode == "simple" else int(rng.integers(5, 12))
    styles = (["dot"] * count if mode == "simple" else
              ["dot", "dot", "bar", "block", "ell", "irregular"] if mode == "mixed" else
              ["dot", "bar", "block", "ell", "irregular", "irregular"])
    placed = 0
    for index in range(count):
        color = (mandatory if index == 0 and mandatory is not None else
                 active[int(rng.integers(len(active)))] if index == 0 or not unused or
                 (index > 1 and rng.random() < .72) else
                 unused[int(rng.integers(len(unused)))])
        style = styles[int(rng.integers(len(styles)))]
        if mode == "large" and index == 0:
            style = "irregular"
        anchors = np.argwhere(valid)
        placed_before = placed
        for chosen_style in (style, "dot"):
            cells = _shape(rng, chosen_style)
            for anchor_index in rng.permutation(len(anchors))[:130]:
                ar, ac = map(int, anchors[int(anchor_index)])
                points = [(ar+r, ac+c) for r, c in cells]
                if any(r >= h or c >= w or not valid[r, c] or
                       np.any(occupied[max(0,r-1):min(h,r+2), max(0,c-1):min(w,c+2)])
                       for r, c in points):
                    continue
                for r, c in points:
                    grid[r, c] = color
                    occupied[r, c] = True
                placed += 1
                break
            if placed > placed_before:
                break
    if placed < 2:
        raise AssertionError("payload placement failed")
    return occupied


def _make_pair(rng, split, index, core, previous_background, seed, layout_mode):
    witness = split == "train" and index < 2
    count = (4 if index == 0 else 3) if witness else int(rng.integers(3, 6 if split == "train" else 9))
    if witness:
        a, b, c, d, e, f = core
        rules = ((a,b),(b,c),(c,d),(e,f)) if index == 0 else ((a,b),(c,d),(b,c))
        inert_source = None
        bg = int(rng.choice([x for x in range(10) if x not in core]))
    else:
        bg = previous_background if rng.random() < .38 else int(rng.integers(0,10))
        palette = tuple(int(x) for x in rng.choice([x for x in range(10) if x != bg], 6, replace=False))
        rules, inert_source = _program(rng, palette, count,
                                       bool(rng.random() < .35 and count >= 4),
                                       bool(rng.random() < .45))
    divider_options = [x for x in range(10) if x != bg]
    divider_color = (int(rng.choice([d for _, d in rules if d != bg]))
                     if rng.random() < .3 else int(rng.choice(divider_options)))
    if layout_mode == 0:
        orientation = "v"
    elif layout_mode == 1:
        orientation = "h"
    elif layout_mode == 2:
        orientation = "v" if split == "train" else "h"
    else:
        orientation = ("v" if index == 0 else "h") if witness else (
            "v" if rng.integers(0,2) else "h")
    strength = 0 if witness else int(rng.choice([0,1,2], p=[.28,.37,.35] if split == "test" else [.48,.37,.15]))
    mode = ("simple" if index == 0 and witness else "mixed" if witness else
            ("simple", "mixed", "large")[(seed + index + (1 if split == "test" else 0)) % 3])
    for attempt in range(24):
        h = int(rng.integers(15 if split == "train" else 20, 27 if split == "train" else 31))
        w = int(rng.integers(14 if split == "train" else 13, 27 if split == "train" else 31))
        if orientation == "h" and count >= 7:
            h = max(h, 23)
        barrier = _divider(rng, h, w, orientation, strength)
        regions = _regions(barrier)
        if len(regions) != 2:
            continue
        sides = list(rng.permutation(2))
        for table_side in sides:
            sites = _table_sites(regions[int(table_side)], barrier, count)
            if not sites:
                continue
            row, col = sites[int(rng.integers(len(sites)))]
            grid = np.full((h,w), bg, dtype=np.int8)
            grid[barrier] = divider_color
            for k, (source, dest) in enumerate(rules):
                grid[row+k, col] = source
                grid[row+k, col+1] = dest
            payload_side = regions[1-int(table_side)]
            try:
                _place_payload(rng, grid, payload_side, barrier, rules, inert_source,
                               mode, core[0] if witness else None)
            except AssertionError:
                continue
            output = _apply(grid, payload_side & (grid != bg), rules)
            if np.any(output != grid):
                return {"input": grid, "output": output}
    raise AssertionError("bounded pair construction failed")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, nq = _counts(seed, rng, num_train, num_test)
    core = tuple(int(x) for x in rng.choice(10, 6, replace=False))
    layout_mode = int(rng.integers(0, 4))
    puzzle = {"train": [], "test": []}
    previous_bg = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            pair = _make_pair(rng, split, index, core, previous_bg, seed, layout_mode)
            previous_bg = int(np.bincount(pair["input"].ravel(), minlength=10).argmax())
            puzzle[split].append(pair)
    return puzzle


def _component(grid, color, start):
    h, w = grid.shape
    seen = np.zeros((h,w), dtype=bool)
    todo = [start]
    seen[start] = True
    while todo:
        r,c = todo.pop()
        for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if 0 <= rr < h and 0 <= cc < w and not seen[rr,cc] and grid[rr,cc] == color:
                seen[rr,cc] = True
                todo.append((rr,cc))
    return seen


def _parse(grid):
    h,w = grid.shape
    bg = int(np.bincount(grid.ravel(), minlength=10).argmax())
    barriers = []
    for color in range(10):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        for r,c in coords:
            if r not in (0,h-1) and c not in (0,w-1):
                continue
            comp = _component(grid, color, (int(r),int(c)))
            if (np.any(comp[0]) and np.any(comp[-1])) or (np.any(comp[:,0]) and np.any(comp[:,-1])):
                if not any(np.array_equal(comp, old) for old in barriers):
                    barriers.append(comp)
            break
    assert len(barriers) == 1, "unique edge-to-edge divider"
    barrier = barriers[0]
    regions = _regions(barrier)
    assert len(regions) == 2, "divider separates two regions"
    tables = []
    for col in range(w-1):
        row = 0
        while row < h:
            if grid[row,col] == bg or grid[row,col+1] == bg or grid[row,col] == grid[row,col+1] or barrier[row,col] or barrier[row,col+1]:
                row += 1
                continue
            start = row
            while row < h and grid[row,col] != bg and grid[row,col+1] != bg and grid[row,col] != grid[row,col+1] and not barrier[row,col] and not barrier[row,col+1]:
                row += 1
            if row-start >= 3:
                tables.append((start,col,row))
    assert len(tables) == 1, "unique readable table"
    start,col,end = tables[0]
    rules = tuple((int(grid[r,col]),int(grid[r,col+1])) for r in range(start,end))
    table_side = next((i for i,region in enumerate(regions) if region[start,col]), None)
    assert table_side is not None and np.all(regions[table_side][start:end,col:col+2])
    payload = regions[1-table_side] & (grid != bg)
    return {"background":bg, "barrier":barrier, "regions":regions,
            "table_side":table_side, "rules":rules, "payload":payload,
            "table":(start,col,end)}


def _digest(inp, out):
    return hashlib.sha256(bytes(inp.shape) + inp.astype(np.uint8).tobytes() +
                          bytes(out.shape) + out.astype(np.uint8).tobytes()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp,out):
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype,np.integer) and np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and np.any(inp != out)
            key = (inp.shape,inp.astype(np.uint8).tobytes())
            pairkey = (key,out.astype(np.uint8).tobytes())
            assert key not in inputs and pairkey not in pairs
            inputs.add(key); pairs.add(pairkey)
            assert _digest(inp,out) not in _SOURCE_PAIR_DIGESTS
            info = _parse(inp)
            assert np.array_equal(out,_apply(inp,info["payload"],info["rules"]))
            assert np.array_equal(inp[~info["payload"]],out[~info["payload"]])
            assert info["background"] not in {s for s, _ in info["rules"]}
            assert np.any(info["payload"] & (inp != out))
            assert np.any(info["payload"] &
                          np.isin(inp, [c for c in range(10)
                                        if c not in {s for s, _ in info["rules"]}]))
            assert len(info["rules"]) <= (5 if split == "train" else 8)
            if split == "train":
                assert 15 <= inp.shape[0] <= 26 and 14 <= inp.shape[1] <= 26
            else:
                assert 20 <= inp.shape[0] <= 30 and 13 <= inp.shape[1] <= 30
            metadata[split].append(info)
    # The first two training demonstrations contrast the same input probe under reordered rows.
    first,second = puzzle["train"][:2]
    a,b = metadata["train"][:2]
    common = set(np.unique(first["input"][a["payload"]])) & set(np.unique(second["input"][b["payload"]]))
    assert any(_apply(np.array([[int(c)]],dtype=np.int8),np.array([[True]]),a["rules"])[0,0] !=
               _apply(np.array([[int(c)]],dtype=np.int8),np.array([[True]]),b["rules"])[0,0]
               for c in common)


def _equal(a,b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k],y[k]) for x,y in zip(a[s],b[s]) for k in ("input","output")) for s in ("train","test"))


def _cycle(rules):
    edges = {}
    for source, destination in rules:
        edges.setdefault(source, set()).add(destination)
    def walk(color, path):
        if color in path:
            return True
        return any(walk(next_color, path | {color})
                   for next_color in edges.get(color, ()))
    return any(walk(color, set()) for color in edges)


def _object_sizes(grid, payload):
    seen = np.zeros_like(payload)
    sizes = []
    h,w = grid.shape
    for r,c in np.argwhere(payload):
        r,c = int(r),int(c)
        if seen[r,c]:
            continue
        color = grid[r,c]
        seen[r,c] = True
        todo = [(r,c)]
        size = 0
        while todo:
            y,x = todo.pop()
            size += 1
            for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0 <= yy < h and 0 <= xx < w and payload[yy,xx] and not seen[yy,xx] and grid[yy,xx] == color:
                    seen[yy,xx] = True
                    todo.append((yy,xx))
        sizes.append(size)
    return sizes


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _equal(generate_puzzle(123),generate_puzzle(123))
    assert not _equal(generate_puzzle(123),generate_puzzle(124))
    all_pairs = [pair for p in puzzles for split in ("train","test") for pair in p[split]]
    shapes = {pair["input"].shape for pair in all_pairs}
    palettes = {tuple(np.unique(pair["input"])) for pair in all_pairs}
    orientations, lengths, depths, backgrounds, dividers = set(),set(),set(),set(),set()
    sides, combinations, object_counts, object_sizes = set(),set(),set(),set()
    inert_rows = cyclic = acyclic = repeated = double_repeated = 0
    singleton = mixed = large = degree_three = divider_result = 0
    for pair in all_pairs:
        info = _parse(pair["input"])
        barrier = info["barrier"]
        orientation = "h" if np.any(barrier[:,0]) and np.any(barrier[:,-1]) else "v"
        orientations.add(orientation)
        lengths.add(len(info["rules"]))
        backgrounds.add(info["background"])
        divider_color = int(pair["input"][barrier][0])
        dividers.add(divider_color)
        coords = np.argwhere(barrier)
        depth = int(np.ptp(coords[:,0 if orientation == "h" else 1]))
        depths.add(depth)
        tr,tc,_ = info["table"]
        side = ("top" if tr < np.mean(coords[:,0]) else "bottom") if orientation == "h" else ("left" if tc < np.mean(coords[:,1]) else "right")
        sides.add(side)
        sizes = _object_sizes(pair["input"],info["payload"])
        object_counts.add(len(sizes)); object_sizes.update(sizes)
        singleton += max(sizes) == 1
        mixed += min(sizes) == 1 and max(sizes) > 1
        large += max(sizes) >= 8
        combinations.add((orientation,side,depth//3,len(info["rules"]),max(sizes)//4))
        for r,c in coords:
            if sum(barrier[y,x] for y,x in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                   if 0 <= y < barrier.shape[0] and 0 <= x < barrier.shape[1]) >= 3:
                degree_three += 1
                break
        cyclic += _cycle(info["rules"])
        acyclic += not _cycle(info["rules"])
        source_counts = [sum(s == source for s,_ in info["rules"])
                         for source in {s for s,_ in info["rules"]}]
        repeated += max(source_counts) >= 2
        double_repeated += sum(n >= 2 for n in source_counts) >= 2
        current = pair["input"].copy()
        hits = []
        for src,dst in info["rules"]:
            hit = info["payload"] & (current == src)
            hits.append(int(np.count_nonzero(hit)))
            current[hit] = dst
        inert_rows += any(x == 0 for x in hits)
        divider_result += any(dst == divider_color and hit > 0
                              for (_,dst),hit in zip(info["rules"],hits))
    assert len(shapes) >= 35 and len(palettes) >= 25
    assert orientations == {"h","v"} and len(lengths) >= 5
    assert sides == {"top","bottom","left","right"}
    assert len(backgrounds) >= 8 and len(dividers) >= 8 and inert_rows >= 10
    assert max(depths) >= 8 and 0 in depths and degree_three >= 5
    assert cyclic >= 10 and acyclic >= 30 and repeated >= 20 and double_repeated >= 5
    assert singleton >= 20 and mixed >= 20 and large >= 20
    assert len(object_counts) >= 6 and max(object_sizes) >= 15
    assert divider_result >= 10 and len(combinations) >= 40
    for nt,nq in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        p = generate_puzzle(701+int(nt or 0)*11+int(nq or 0),nt,nq)
        assert (nt is None or len(p["train"]) == nt) and (nq is None or len(p["test"]) == nq)
        validate_generated_puzzle(p)
    for nt,nq in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4)):
        try:
            generate_puzzle(907,nt,nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
