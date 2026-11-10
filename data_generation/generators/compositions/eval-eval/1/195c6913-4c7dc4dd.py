"""Route-map analogy: infer a cyclic program, then register and paint new routes.

Only NumPy and the standard library are used. Nothing is read from disk. The
constructor uses explicit vertices; the solver walks rendered pixels one step
at a time. self_test keeps only small signatures and aggregate measurements.
"""
from __future__ import annotations

import hashlib
from itertools import permutations

import numpy as np


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(rng, a, b):
    for x, low, high in ((a, 2, 10), (b, 1, 4)):
        if x is not None and (not _integer(x) or not low <= x <= high):
            raise ValueError("illegal example count")
    if b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1 if a is not None else 5))
    if a is None:
        a = int(rng.integers(max(2, b + 1), 11))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _digest(a):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.uint8).tobytes()).hexdigest()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    while remaining:
        todo = [remaining.pop()]
        cells = []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        yield cells


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _walk(mask, row):
    """Input-reading walker. A corner is visited once; edge exits are terminal."""
    p, direction = (row, 0), 0
    path, collisions, legs = [], [], [0]
    assert mask[p]
    for _ in range(mask.size + 1):
        path.append(p)
        legs[-1] += 1
        r, c = p
        ahead = (r, c+1) if direction == 0 else (r-1, c)
        if not _inside(mask.shape, ahead):
            return dict(path=tuple(path), collisions=tuple(collisions),
                        legs=tuple(legs), end="edge")
        if mask[ahead]:
            p = ahead
            continue
        collisions.append(ahead)
        turn = (r-1, c) if direction == 0 else (r, c+1)
        if not _inside(mask.shape, turn) or not mask[turn]:
            return dict(path=tuple(path), collisions=tuple(collisions),
                        legs=tuple(legs), end="blocked")
        p, direction = turn, 1-direction
        legs.append(0)
    raise AssertionError("route failed to terminate")


def _cards(grid):
    """Detect disconnected fence outlines, including one/two entry gaps."""
    by_color = {}
    for color in map(int, np.unique(grid)):
        found = []
        for cells in _components(grid == color):
            rr, cc = zip(*cells)
            r, c = min(rr), min(cc)
            h, w = max(rr)-r+1, max(cc)-c+1
            if not (8 <= h <= 13 and 8 <= w <= 13):
                continue
            tile = grid[r:r+h, c:c+w]
            interior = tile[1:-1, 1:-1]
            if np.any(interior == color):
                continue
            if any(int(tile[p]) != color for p in ((0,0),(0,w-1),(h-1,0),(h-1,w-1))):
                continue
            border = np.zeros(tile.shape, bool)
            border[[0,-1], :] = True
            border[:, [0,-1]] = True
            holes = np.argwhere(border & (tile != color))
            if not 1 <= len(holes) <= 2:
                continue
            # Exactly one rotation makes every gate a noncorner left-edge cell.
            candidates = []
            for k in range(4):
                t = np.rot90(tile, k)
                if not (np.all(t[0] == color) and np.all(t[-1] == color)
                        and np.all(t[:, -1] == color)):
                    continue
                rows = np.flatnonzero(t[1:-1, 0] != color)
                if len(rows) != len(holes):
                    continue
                a = t[1:-1, 1:-1].copy()
                corners = {int(a[p]) for p in ((0,0),(0,-1),(-1,0),(-1,-1))}
                if len(corners) != 1:
                    continue
                wall = next(iter(corners))
                gates = tuple((int(y), int(t[y+1, 0])) for y in rows)
                if wall == color or any(v in (wall, color) or int(a[y,0]) != v for y,v in gates):
                    continue
                if any(b[0]-a0[0] < 3 for a0,b in zip(gates,gates[1:])):
                    continue
                candidates.append(dict(array=a, gates=gates, wall=wall,
                                       rotation=k, box=(r,c,h,w), frame=color))
            if len(candidates) == 1:
                found.append(candidates[0])
        if len(found) == 3 and len({t["wall"] for t in found}) == 1:
            # The fence color is reserved; no unnoticed fence fragments allowed.
            covered = np.zeros(grid.shape, bool)
            for t in found:
                r,c,h,w = t["box"]
                covered[r:r+h,c:c+w] = True
            if not np.any((grid == color) & ~covered):
                by_color[color] = found
    assert len(by_color) == 1, "expected three uniquely detectable route maps"
    result = next(iter(by_color.values()))
    for a,b in permutations(result, 2):
        r,c,h,w = a["box"]
        y,x,dh,dw = b["box"]
        assert r+h <= y or y+dh <= r or c+w <= x or x+dw <= c
    outside = np.ones(grid.shape, bool)
    for t in result:
        r,c,h,w = t["box"]
        outside[r:r+h,c:c+w] = False
    assert np.all(grid[outside] == result[0]["wall"])
    return result


def _period(values):
    for p in range(2, 7):
        if len(values) >= 2*p and len(set(values[:p])) >= 2:
            word = tuple(values[:p])
            if all(x == word[i % p] for i,x in enumerate(values)):
                return word
    raise AssertionError("worked route does not exhibit two full cycles")


def _phases(word, prefix):
    return [q for q in range(len(word))
            if all(v == word[(q+i) % len(word)] for i,v in enumerate(prefix))]


def _paint(terrain, records, word, marker, phases):
    out = terrain.copy()
    for record, q in zip(records, phases):
        for i,p in enumerate(record["path"]):
            out[p] = word[(q+i) % len(word)]
        for p in record["collisions"]:
            out[p] = marker
    return out


def _interpret(cards, ib, ia, it):
    before, after, target = [cards[i] for i in (ib,ia,it)]
    a,b,t = [x["array"] for x in (before,after,target)]
    wall, frame = before["wall"], before["frame"]
    assert a.shape == b.shape and before["gates"] == after["gates"]
    assert len(before["gates"]) == 1
    row, first = before["gates"][0]
    track_options = set(map(int,np.unique(a))) - {wall, first}
    assert len(track_options) == 1
    track = track_options.pop()
    assert track not in (wall,frame,first) and np.count_nonzero(a == first) == 1
    terrain = a.copy()
    terrain[row,0] = track
    mask = terrain == track
    record = _walk(mask,row)
    assert len(record["collisions"]) >= 1
    values = [int(b[p]) for p in record["path"]]
    word = _period(values)
    assert len(set(word)) <= 4
    assert word[0] == first and not set(word) & {wall,track,frame}
    markers = {int(b[p]) for p in record["collisions"]}
    assert len(markers) == 1
    marker = markers.pop()
    assert marker not in set(word) | {wall,track,frame}
    assert np.array_equal(b,_paint(terrain,[record],word,marker,[0]))
    assert set(map(int,np.unique(t))) <= {wall,track} | set(word)
    target_terrain = t.copy()
    target_terrain[np.isin(t,word)] = track
    assert a.shape != t.shape or not np.array_equal(mask,target_terrain == track)
    records, phases, prefixes, used = [], [], [], set()
    for y, gate_color in target["gates"]:
        rec = _walk(target_terrain == track,y)
        assert int(t[y,0]) == gate_color and len(rec["path"]) >= 3
        footprint = set(rec["path"]) | set(rec["collisions"])
        assert not used & footprint
        used |= footprint
        prefix = []
        ended = False
        for p in rec["path"]:
            if int(t[p]) == track:
                ended = True
            else:
                assert not ended and int(t[p]) in word
                prefix.append(int(t[p]))
        assert 1 <= len(prefix) <= len(word) and len(prefix) < len(rec["path"])-1
        alignments = _phases(word,prefix)
        assert len(alignments) == 1, "painted prefix must identify one cycle phase"
        phases.append(alignments[0])
        prefixes.append(tuple(prefix))
        records.append(rec)
    assert all(tuple(p) in used for p in np.argwhere(np.isin(t,word)))
    out = _paint(target_terrain,records,word,marker,phases)
    return dict(before=before,after=after,target=target,track=track,wall=wall,
                marker=marker,word=word,reference=record,terrain=target_terrain,
                records=records,phases=tuple(phases),prefixes=prefixes,
                canonical_output=out,output=np.rot90(out,-target["rotation"]).copy())


def _analyze(grid):
    cards = _cards(_grid(grid))
    hypotheses = []
    for ib,ia,it in permutations(range(3)):
        try:
            hypotheses.append(_interpret(cards,ib,ia,it))
        except (AssertionError,IndexError):
            pass
    assert len(hypotheses) == 1, "worked pair and target must have unique content correspondence"
    return hypotheses[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the completed target interior in the target's visible orientation."""
    return _analyze(input_grid)["output"]


def _split_positive(rng, total, n):
    if not n:
        return []
    cuts = sorted(map(int,rng.choice(np.arange(1,total),n-1,replace=False)))
    return list(np.diff([0] + cuts + [total]))


def _make_route(rng,h,w,lo,hi,minimum,bent):
    """Construct vertices directly; no invocation of the input-reading walker."""
    for _ in range(160):
        nh = int(rng.integers(1,min(3,hi-lo+2)+1))
        nv = nh - int(rng.integers(2))
        if nv > hi-lo or (bent and nv == 0):
            continue
        dx = int(rng.integers(max(3,nh),w))
        dy = int(rng.integers(nv,hi-lo+1)) if nv else 0
        if dx+dy+1 < minimum or (nv == nh and dx == w-1):
            continue
        hs,vs = _split_positive(rng,dx,nh), _split_positive(rng,dy,nv)
        r,c = hi,0
        path, collisions, legs = [(r,c)], [], []
        for j in range(nh+nv):
            horizontal = j % 2 == 0
            distance = hs[j//2] if horizontal else vs[j//2]
            leg = distance + (j == 0)
            for _ in range(distance):
                r,c = (r,c+1) if horizontal else (r-1,c)
                path.append((r,c))
            legs.append(int(leg))
            ahead = (r,c+1) if horizontal else (r-1,c)
            if _inside((h,w),ahead):
                collisions.append(ahead)
            elif j != nh+nv-1:
                break
        else:
            horizontal = (nh+nv) % 2 == 1
            ahead = (r,c+1) if horizontal else (r-1,c)
            end = "blocked" if _inside((h,w),ahead) else "edge"
            stop = (r-1,c) if horizontal else (r,c+1)
            blockers = set(collisions)
            if end == "blocked" and _inside((h,w),stop):
                blockers.add(stop)
            if set(path) & blockers:
                continue
            return dict(path=tuple(path),collisions=tuple(collisions),legs=tuple(legs),
                        end=end,blockers=blockers)
    raise AssertionError("route construction exhausted")


def _map(rng,h,w,nroutes,minimum,bent=True):
    for _ in range(40):
        bands = [(0,h-2)] if nroutes == 1 else [(0,h//2-1),(h//2+1,h-2)]
        records = [_make_route(rng,h,w,lo,hi,minimum,bent) for lo,hi in bands]
        paths = set(p for rec in records for p in rec["path"])
        blockers = set(p for rec in records for p in rec["blockers"])
        footprint = set()
        good = not paths & blockers
        for rec in records:
            cells = set(rec["path"]) | set(rec["collisions"])
            good &= not bool(footprint & cells)
            footprint |= cells
        if not good:
            continue
        # Add broad, uneven terrain away from the guard cells that define turns.
        mask = np.zeros((h,w),bool)
        for _ in range(int(rng.integers(3,10))):
            y,z = sorted(map(int,rng.choice(h+1,2,replace=False)))
            x,v = sorted(map(int,rng.choice(w+1,2,replace=False)))
            mask[y:z,x:v] = True
        for p in paths:
            mask[p] = True
        for p in blockers | {(0,0),(0,w-1),(h-1,0),(h-1,w-1)}:
            mask[p] = False
        # Remove floating terrain components without inventing more starts.
        for comp in _components(mask):
            if not paths.intersection(comp):
                for p in comp:
                    mask[p] = False
        if np.count_nonzero(mask) - len(paths) < 3:
            continue
        return mask,records
    raise AssertionError("map construction exhausted")


def _primitive(word):
    return not any(len(word) % n == 0 and all(x == word[i % n] for i,x in enumerate(word))
                   for n in range(1,len(word)))


def _word(rng,p,informative):
    for _ in range(100):
        distinct = int(rng.integers(2,min(4,p)+1))
        word = list(range(distinct)) + list(map(int,rng.integers(distinct,size=p-distinct)))
        rng.shuffle(word)
        if not _primitive(word):
            continue
        unique_phases = [i for i,x in enumerate(word) if word.count(x) == 1 and i != 0]
        if informative and (distinct == p or not unique_phases):
            continue
        return tuple(word),unique_phases
    raise AssertionError("word construction exhausted")


def _tile(interior,rows,colors,frame,rotation):
    tile = np.full((interior.shape[0]+2,interior.shape[1]+2),frame,dtype=np.int8)
    tile[1:-1,1:-1] = interior
    for r,color in zip(rows,colors):
        tile[r+1,0] = color
    return np.rot90(tile,rotation).copy()


def _pack(rng,tiles,wall):
    """Three maps in variable shelves or a row/column, with no semantic slots."""
    options = []
    for order in permutations(range(3)):
        for transpose in (False,True):
            sizes = [(t.shape[1],t.shape[0]) if transpose else t.shape for t in tiles]
            a,b,c = [sizes[i] for i in order]
            for kind in (0,1):
                mh = max(a[0],b[0],c[0]) if kind == 0 else a[0]+max(b[0],c[0])+1
                mw = a[1]+b[1]+c[1]+2 if kind == 0 else max(a[1],b[1]+c[1]+1)
                if mh <= 30 and mw <= 30:
                    options.append((order,transpose,kind,mh,mw))
    order,transpose,kind,mh,mw = options[int(rng.integers(len(options)))]
    work = [t.T if transpose else t for t in tiles]
    a,b,c = [work[i] for i in order]
    h = int(rng.integers(mh,min(30,mh+5)+1))
    w = int(rng.integers(mw,min(30,mw+5)+1))
    grid = np.full((h,w),wall,dtype=np.int8)
    if kind == 0:
        slack = w - a.shape[1]-b.shape[1]-c.shape[1]-2
        gaps = rng.multinomial(slack,[.25]*4)
        xs = [int(gaps[0]),int(gaps[0]+a.shape[1]+1+gaps[1])]
        xs.append(xs[-1]+b.shape[1]+1+int(gaps[2]))
        pos = [(int(rng.integers(h-t.shape[0]+1)),x) for t,x in zip((a,b,c),xs)]
    else:
        lower_h = max(b.shape[0],c.shape[0])
        slack = h-a.shape[0]-lower_h-1
        top,gap,_ = map(int,rng.multinomial(slack,[1/3]*3))
        below = top+a.shape[0]+1+gap
        x = int(rng.integers(w-b.shape[1]-c.shape[1]))
        pos = [(top,int(rng.integers(w-a.shape[1]+1))),
               (below+int(rng.integers(lower_h-b.shape[0]+1)),x),
               (below+int(rng.integers(lower_h-c.shape[0]+1)),x+b.shape[1]+1)]
    for tile,(r,c0) in zip((a,b,c),pos):
        grid[r:r+tile.shape[0],c0:c0+tile.shape[1]] = tile
    return grid.T.copy() if transpose else grid


def _evidence(info):
    word,records,phases = info["word"],info["records"],info["phases"]
    good = info["canonical_output"]
    evidence = {"reference_cycles"}
    if any(q != 0 for q in phases):
        if not np.array_equal(good,_paint(info["terrain"],records,word,info["marker"],[0]*len(records))):
            evidence.add("cue_phase")
    dedup = tuple(dict.fromkeys(word))
    if len(dedup) != len(word):
        wrong = _paint(info["terrain"],records,dedup,info["marker"],
                       [dedup.index(word[q]) for q in phases])
        if not np.array_equal(good,wrong):
            evidence.add("multiplicity")
    for label,consume in (("bend_phase",False),("marker_phase",True)):
        wrong = info["terrain"].copy()
        for rec,q in zip(records,phases):
            offset = 0
            for j,n in enumerate(rec["legs"]):
                for i,p in enumerate(rec["path"][offset:offset+n]):
                    phase = q+offset+i+j if consume else q+i
                    wrong[p] = word[phase % len(word)]
                offset += n
            for p in rec["collisions"]:
                wrong[p] = info["marker"]
        if not np.array_equal(good,wrong):
            evidence.add(label)
    if len(records) > 1:
        global_phases,offset = [],phases[0]
        for rec in records:
            global_phases.append(offset % len(word))
            offset += len(rec["path"])
        if not np.array_equal(good,_paint(info["terrain"],records,word,info["marker"],global_phases)):
            evidence.add("independent_phase")
    return evidence


_REQUIRED = {"reference_cycles","cue_phase","multiplicity","bend_phase",
             "marker_phase","independent_phase"}


def _make_pair(rng,index,split,style):
    informative = split == "train" and index < 2
    for _ in range(160):
        p = int(rng.integers(3,6)) if informative else int(rng.integers(2,7))
        logical,unique_phases = _word(rng,p,informative)
        palette = list(map(int,rng.permutation(10)))
        wall,track,frame,marker = palette[:4]
        word = tuple(palette[4+x] for x in logical)
        rh = int(rng.integers(max(7,p+2),12))
        rw = int(rng.integers(max(7,p+2),12))
        count = (1 if index == 0 else 2) if informative else int(rng.integers(1,3))
        th = int(rng.integers(10 if count == 2 else 7,12))
        tw = int(rng.integers(10,12)) if split == "test" and style["stretch"] else int(rng.integers(8,10 if style["stretch"] else 12))
        try:
            rm,rr = _map(rng,rh,rw,1,2*p+1)
            tm,tr = _map(rng,th,tw,count,p+3,bent=informative or bool(rng.integers(2)))
        except AssertionError:
            continue
        if rm.shape == tm.shape and np.array_equal(rm,tm):
            continue
        ref_terrain = np.where(rm,track,wall).astype(np.int8)
        before = ref_terrain.copy()
        before[rr[0]["path"][0]] = word[0]
        after = _paint(ref_terrain,rr,word,marker,[0])
        terrain = np.where(tm,track,wall).astype(np.int8)
        target,phases = terrain.copy(),[]
        for j,rec in enumerate(tr):
            q = int(rng.choice(unique_phases)) if informative and index == 0 else int(rng.integers(p))
            k = next(k for k in range(1,p+1) if len(_phases(word,[word[(q+i)%p] for i in range(k)])) == 1)
            if not informative and k < p and rng.random() < .25:
                k += 1
            if k >= len(rec["path"])-1:
                break
            for i,point in enumerate(rec["path"][:k]):
                target[point] = word[(q+i)%p]
            phases.append(q)
        if len(phases) != count:
            continue
        output = _paint(terrain,tr,word,marker,phases)
        rotations = [style["orientation"]]*3
        if split == "test":
            rotations = list(map(int,rng.integers(4,size=3)))
            rotations[2] = (style["orientation"]+int(rng.integers(1,4))) % 4
        elif not style["fixed"]:
            rotations = list(map(int,rng.integers(4,size=3)))
        tiles = [_tile(a,[r["path"][0][0] for r in recs],colors,frame,k)
                 for a,recs,colors,k in ((before,rr,[word[0]],rotations[0]),
                                        (after,rr,[word[0]],rotations[1]),
                                        (target,tr,[word[q] for q in phases],rotations[2]))]
        inp = _pack(rng,tiles,wall)
        pair = dict(input=inp,output=np.rot90(output,rotations[2]).copy())
        try:
            info = _analyze(inp)
            assert np.array_equal(info["output"],pair["output"])
            if informative:
                needs = {"multiplicity","bend_phase","marker_phase","cue_phase"} if index == 0 else {"independent_phase"}
                assert needs <= _evidence(info)
        except AssertionError:
            continue
        return pair
    raise AssertionError("pair construction exhausted bounded attempts")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    nt,ne = _counts(rng,num_train,num_test)
    regime = int(rng.integers(3))
    style = dict(orientation=int(rng.integers(4)),fixed=regime in (0,2),
                 stretch=regime in (1,2))
    puzzle,seen = {"train":[],"test":[]},set()
    for split,n in (("train",nt),("test",ne)):
        for i in range(n):
            for _ in range(8):
                pair = _make_pair(rng,i,split,style)
                key = _digest(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    a,b = len(puzzle["train"]),len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen,evidence = set(),set()
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            key = _digest(inp)
            assert key not in seen
            seen.add(key)
            info = _analyze(inp)
            assert np.array_equal(info["output"],out)
            # Every nonroute cell is preserved; collision cells are originally wall.
            mask = np.zeros(info["terrain"].shape,bool)
            for rec,prefix,q in zip(info["records"],info["prefixes"],info["phases"]):
                assert len(set(rec["path"])) == len(rec["path"])
                assert sum(rec["legs"]) == len(rec["path"])
                assert _phases(info["word"],prefix) == [q]
                for p in rec["path"]:
                    mask[p] = True
                for p in rec["collisions"]:
                    assert info["terrain"][p] == info["wall"]
                    mask[p] = True
            assert np.array_equal(info["canonical_output"][~mask],info["terrain"][~mask])
            if split == "train":
                evidence |= _evidence(info)
    assert _REQUIRED <= evidence, _REQUIRED-evidence


def _counterfactual_reference(inp):
    """Change only one unobserved phase in the completed exemplar."""
    info = _analyze(inp)
    assert len(info["records"]) == 1 and len(info["prefixes"][0]) == 1
    word = info["word"]
    spare = next(c for c in range(10) if c not in set(map(int,np.unique(inp))))
    chosen = next(i for i in range(1,len(word)) if i != info["phases"][0]
                  and (len(set(word)) < 4 or word.count(word[i]) == 1))
    changed = inp.copy()
    tile = info["after"]
    canonical = tile["array"].copy()
    for i,p in enumerate(info["reference"]["path"]):
        if i % len(word) == chosen:
            canonical[p] = spare
    r,c,h,w = tile["box"]
    changed[r+1:r+h-1,c+1:c+w-1] = np.rot90(canonical,-tile["rotation"])
    other = _analyze(changed)
    assert np.array_equal(info["target"]["array"],other["target"]["array"])
    assert not np.array_equal(info["output"],other["output"])
    return changed


def _replace_tile(inp,card,interior,gates=None):
    changed = inp.copy()
    r,c,h,w = card["box"]
    tile = np.rot90(changed[r:r+h,c:c+w],card["rotation"]).copy()
    tile[1:-1,1:-1] = interior
    if gates is not None:
        for y,color in gates:
            tile[y+1,0] = color
    changed[r:r+h,c:c+w] = np.rot90(tile,-card["rotation"])
    return changed


def _counterfactual_terrain(inp):
    """A changed obstacle changes the answer even with the worked pair fixed."""
    info = _analyze(inp)
    for point in info["records"][0]["collisions"]:
        if point in ((0,0),(0,info["terrain"].shape[1]-1),
                     (info["terrain"].shape[0]-1,0),
                     (info["terrain"].shape[0]-1,info["terrain"].shape[1]-1)):
            continue
        target = info["target"]["array"].copy()
        target[point] = info["track"]
        changed = _replace_tile(inp,info["target"],target)
        try:
            other = _analyze(changed)
        except AssertionError:
            continue
        assert not np.array_equal(info["output"],other["output"])
        assert np.array_equal(info["before"]["array"],other["before"]["array"])
        assert np.array_equal(info["after"]["array"],other["after"]["array"])
        return changed
    raise AssertionError("no valid obstacle counterfactual")


def _must_reject_input(inp):
    try:
        solve(inp)
    except AssertionError:
        return
    raise AssertionError("invalid input accepted")


def self_test():
    shapes,words,geometries,palettes,rotations,counts,signatures = [set() for _ in range(7)]
    end_types,leg_counts,prefix_lengths,phases,layouts = [set() for _ in range(5)]
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_digest(p["input"]) for s in ("train","test") for p in puzzle[s]))
        _counterfactual_reference(puzzle["train"][0]["input"])
        _counterfactual_terrain(puzzle["train"][0]["input"])
        for split in ("train","test"):
            for pair in puzzle[split]:
                pairs += 1
                info = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                roles = {}
                words.add(tuple(roles.setdefault(c,len(roles)) for c in info["word"]))
                geometries.add(_digest((info["terrain"] == info["track"]).astype(np.int8)))
                palettes.add((info["wall"],info["track"],info["marker"],info["word"]))
                rotations.add(info["target"]["rotation"])
                counts.add(len(info["records"]))
                layouts.add(tuple(info[k]["box"] for k in ("before","after","target")))
                prefix_lengths.update(map(len,info["prefixes"]))
                phases.update(info["phases"])
                for rec in [info["reference"]]+info["records"]:
                    end_types.add(rec["end"])
                    leg_counts.add(len(rec["legs"]))
    assert len(signatures) == 42 and pairs == 210
    assert len(shapes) >= 30 and len(geometries) >= 150 and len(words) >= 15
    assert len(palettes) >= 180 and len(layouts) >= 180
    assert rotations == {0,1,2,3} and counts == {1,2}
    assert end_types == {"edge","blocked"} and {1,2,3,4,5} <= leg_counts
    assert {1,2,3} <= prefix_lengths and {0,1,2,3,4} <= phases
    # Explicit/partial counts, deterministic replay, and malformed count types.
    for a,b in ((2,1),(10,4),(None,4),(10,None)):
        p = generate_puzzle(773,a,b)
        q = generate_puzzle(773,a,b)
        assert all(_key(x[k]) == _key(y[k]) for s in p for x,y in zip(p[s],q[s]) for k in x)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(2.0,1),(4,True),(False,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Edge regression: available terrain above a right exit does not allow a turn.
    m = np.zeros((5,8),bool)
    m[3,:] = True
    m[1:4,-1] = True
    r = _walk(m,3)
    assert r["end"] == "edge" and len(r["path"]) == 8 and not r["collisions"]
    m = np.zeros((7,9),bool)
    m[5,:4] = True
    m[:6,3] = True
    m[0,3:8] = True
    r = _walk(m,5)
    assert r["end"] == "edge" and r["path"][-1] == (0,3)
    assert (0,4) not in r["path"] and r["collisions"] == ((5,4),)
    # A wrong output, a disconnected colored cue, and an ambiguous phase all fail.
    p = generate_puzzle(47,2,1)
    bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0,0] = (int(bad["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    info = _analyze(p["train"][0]["input"])
    target = info["target"]["array"].copy()
    unused = set(map(tuple,np.argwhere(target == info["track"])))
    unused -= {x for rec in info["records"] for x in rec["path"]}
    assert unused
    target[sorted(unused)[0]] = info["word"][0]
    _must_reject_input(_replace_tile(p["train"][0]["input"],info["target"],target))
    repeated = next(x for x in info["word"] if info["word"].count(x) > 1)
    target = info["target"]["array"].copy()
    row = info["target"]["gates"][0][0]
    target[row,0] = repeated
    _must_reject_input(_replace_tile(p["train"][0]["input"],info["target"],target,[(row,repeated)]))
    assert len(_phases((1,1,2),(1,))) == 2
    return dict(seeds=42,pairs=pairs,canvas_shapes=len(shapes),word_patterns=len(words),
                target_geometries=len(geometries),palettes=len(palettes),layouts=len(layouts),
                gate_orientations=len(rotations),route_counts=sorted(counts),
                leg_counts=sorted(leg_counts),end_types=sorted(end_types),
                prefix_lengths=sorted(prefix_lengths),phase_offsets=sorted(phases),
                reference_dependence_twins=42,terrain_dependence_twins=42)
