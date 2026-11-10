"""Porous dams: conserved liquid, inferred masonry, and defect halos.

Only standard Python and NumPy are used. No parent module or evidence is read.
The border identifies masonry, 0 is air, and 2 is the diagnostic highlight.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0,394766073,725180984,448970993,105730537,521568224,
              457824600,226187020,671711982,942949106,295885984,267661544,
              690822342,878820881,902786373,722321606,845067040,152586692,
              187064026,322373874,46562873,657266549,646695255,903158208,
              170547018,839902938,76949888,533724386,681521230,650236864,
              234086166,554123188,450500780,222130604,694277223,331725291,
              791736173,14018660,565717798,801849002,19826968,480749524)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and
            1 <= b <= 4 and a > b):
        raise ValueError("require integer counts 2..10 / 1..4 with train > test")
    return int(a), int(b)


def _components(mask):
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        todo, cells = [p], [p]
        while todo:
            r, c = todo.pop()
            for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
                    cells.append(q)
        part = np.zeros((h,w), dtype=bool)
        rr, cc = zip(*cells)
        part[rr,cc] = True
        result.append(part)
    return result


def _halo(holes):
    h, w = holes.shape
    padded = np.pad(holes, 1)
    result = np.zeros_like(holes)
    for dr in range(3):
        for dc in range(3):
            result |= padded[dr:dr+h,dc:dc+w]
    return result & ~holes


def _decode(input_grid):
    a = np.asarray(input_grid)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer) or
            not all(1 <= n <= 30 for n in a.shape) or
            np.any((a < 0) | (a > 9))):
        raise ValueError("invalid ARC grid")
    h, w = a.shape
    if h < 10 or w < 12:
        raise ValueError("not a porous-dam canvas")
    solid = int(a[0,0])
    colors = set(map(int, np.unique(a)))
    water_colors = colors - {0,solid}
    if solid in (0,2) or len(water_colors) != 1 or 2 in colors:
        raise ValueError("air, masonry, liquid and highlight roles conflict")
    water = next(iter(water_colors))
    if not (np.all(a[0] == solid) and np.all(a[-1] == solid) and
            np.all(a[:,0] == solid) and np.all(a[:,-1] == solid)):
        raise ValueError("the intact perimeter must identify masonry")
    observed = a == solid
    base = np.zeros_like(observed)
    base[[0,-1],:] = True
    base[:,[0,-1]] = True
    # Dams span the cavity. Sparse defects never remove 40% of a row;
    # grounded teeth plus specks occupy strictly less than 60% elsewhere.
    dam_rows = observed[1:-1,1:-1].sum(axis=1) * 5 >= 3*(w-2)
    base[1:-1,1:-1] = dam_rows[:,None]
    # Floor teeth are contiguous with the floor and have no corruption.
    # Detached specks cannot extend them in the family grammar.
    for c in range(1,w-1):
        r = h-2
        while r > 0 and observed[r,c] and not base[r,c]:
            base[r,c] = True
            r -= 1
    holes = base & (a == 0)
    extras = ~base & observed
    if np.any(base & (a == water)):
        raise ValueError("liquid cannot replace intended masonry")
    return a, solid, water, base, holes, extras


def _pour(walls, source, reverse=False, top_first=False):
    """Unit pours: descend; spread on support; descend through an outlet.

    Orders and outlet ties are challenged during construction/validation.
    Accepted inputs have the same complete-level result for all four orders.
    """
    h, w = walls.shape
    wet = np.zeros_like(walls)
    points = sorted(map(tuple, np.argwhere(source)),
                    key=lambda p: (p[0] if top_first else -p[0],
                                   -p[1] if reverse else p[1]))
    for rr, cc in points:
        r, c = int(rr), int(cc)
        while r > 0 and wet[r,c]:
            r -= 1
        if walls[r,c] or wet[r,c]:
            raise ValueError("overfull source chamber")
        for _ in range(h+1):
            if r+1 < h and not walls[r+1,c] and not wet[r+1,c]:
                r += 1
                continue
            lo = hi = c
            while lo > 0 and not walls[r,lo-1] and not wet[r,lo-1]:
                lo -= 1
            while hi+1 < w and not walls[r,hi+1] and not wet[r,hi+1]:
                hi += 1
            exits = [x for x in range(lo,hi+1)
                     if r+1 < h and not walls[r+1,x] and not wet[r+1,x]]
            if not exits:
                wet[r,c] = True
                break
            c = min(exits, key=lambda x: (abs(x-c), -x if reverse else x))
            r += 1
        else:
            raise AssertionError("downward pour did not terminate")
    return wet


def _routes(walls, pools):
    """All dry downward destinations, treating each final pool as absorbing.

    Labels are bit sets; an unassigned dry trap gets its own unknown bit.
    This checks source allocation separately from finite-volume pouring.
    """
    h, w = walls.shape
    fixed = np.zeros((h,w), dtype=np.int32)
    for k, pool in enumerate(pools):
        fixed[pool] = 1 << k
    labels = fixed.copy()
    unknown = 1 << len(pools)
    for r in range(h-1,-1,-1):
        c = 0
        while c < w:
            if walls[r,c]:
                c += 1
                continue
            end = c
            while end+1 < w and not walls[r,end+1]:
                end += 1
            outlets = 0
            for x in range(c,end+1):
                outlets |= int(fixed[r,x])
                if r+1 < h and not walls[r+1,x]:
                    outlets |= int(labels[r+1,x])
            for x in range(c,end+1):
                if fixed[r,x]:
                    labels[r,x] = fixed[r,x]
                elif r+1 < h and not walls[r+1,x]:
                    labels[r,x] = labels[r+1,x]
                else:
                    labels[r,x] = outlets or unknown
            c = end+1
    return labels


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, solid, water, base, holes, extras = _decode(input_grid)
    wet = _pour(base & ~holes, a == water)
    halo = _halo(holes)
    if np.any(wet & (halo | holes)):
        raise ValueError("family requires dry defects and dry diagnostic halos")
    out = np.where(base & ~holes, solid, 0).astype(np.int8)
    out[wet] = water
    out[halo] = 2
    return out


def _branches(pool):
    top = int(np.nonzero(pool)[0].min())
    return max([1] + [len(_components(pool &
               (np.arange(pool.shape[0])[:,None] >= r)))
               for r in range(top+1,pool.shape[0])])


def _geometry(rng, kind):
    # Anchor 0: one perforated dam; anchor 1: closed above open.
    n = (1 if kind == 0 else 2 if kind == 1 else
         int(rng.integers(1,3)) if kind == 2 else 3)
    width = int(rng.integers(14,29))
    thick = rng.integers(3,5,size=n)
    gaps = np.full(n+1,4,dtype=int)
    gaps[-1] = int(rng.integers(7,10))
    spare = 28-int(thick.sum()+gaps.sum())
    for _ in range(int(rng.integers(0,max(0,spare)+1))):
        gaps[int(rng.integers(n+1))] += 1
    height = int(thick.sum()+gaps.sum()+2)
    # Three thick dams fit by reducing upper gaps, retaining a deep basin.
    while height > 30:
        j = int(np.argmax(gaps[:-1]))
        if gaps[j] > 3:
            gaps[j] -= 1
        else:
            j = int(np.argmax(thick))
            thick[j] -= 1
        height = int(thick.sum()+gaps.sum()+2)
    base = np.zeros((height,width),dtype=bool)
    base[[0,-1],:] = True
    base[:,[0,-1]] = True
    holes = np.zeros_like(base)
    dams = []
    row = 1
    for k in range(n):
        row += int(gaps[k])
        t = int(thick[k])
        base[row:row+t,1:-1] = True
        opened = kind == 0 or k == n-1 or (kind != 1 and rng.random() < .7)
        # One connected through pore per open dam. It can turn sideways.
        c = int(rng.integers(3,width-4))
        pore_width = int(rng.integers(1,3))
        bend = (k == n-1 and kind == 1) or rng.random() < .6
        if opened:
            for j in range(t):
                shift = int(bend and j >= t//2)
                holes[row+j,c+shift:c+shift+pore_width] = True
                if bend and j == t//2:
                    holes[row+j,c:c+1+pore_width] = True
        # One or two buried voids; no connection to either dam surface.
        # Their halos may meet the through pore's halo without joining it.
        for _ in range(int(rng.integers(1,3))):
            options = [x for x in range(2,width-2)
                       if not holes[row:row+t,max(0,x-1):x+2].any()]
            if options:
                x = int(rng.choice(options))
                holes[row+int(rng.integers(1,t-1)),x] = True
        dams.append((row,row+t,opened))
        row += t
    # Grounded rectangular teeth produce several lower pockets and spill rims.
    teeth = 1 if kind < 2 else int(rng.integers(1,3))
    centers = ([width//2 + int(rng.integers(-2,3))] if teeth == 1 else
               [width//3,2*width//3])
    for c in centers:
        tw = int(rng.integers(1,3))
        th = int(rng.integers(2,5))
        base[height-1-th:height-1,c:c+tw] = True
    return base, holes, dams


def _pool_options(walls, halo):
    h, w = walls.shape
    rows = np.arange(h)[:,None]
    options = []
    for top in range(2,h-1):
        for pool in _components(~walls & (rows >= top)):
            volume = int(pool.sum())
            if (2 <= volume <= min(135,walls.size//3) and pool[top].any()
                    and not np.any(pool & halo)):
                options.append(pool)
    return options


def _pick_source(rng, eligible, target, high, style):
    points = np.argwhere(eligible)
    outside = np.argwhere(eligible & ~target & high)
    count = int(target.sum())
    if len(points) < count or not len(outside):
        raise ValueError("insufficient visible supply space")
    anchor = outside[int(rng.integers(len(outside)))]
    dr, dc = np.abs(points[:,0]-anchor[0]), np.abs(points[:,1]-anchor[1])
    jitter = rng.random(len(points))
    if style == 0:
        score = dr+dc+jitter
    elif style == 1:
        score = jitter
    elif style == 2:
        score = 4*dr+.2*dc+jitter
    else:
        score = 4*dc+.2*dr+jitter
    chosen = points[np.argsort(score)[:count]].copy()
    if not np.any(np.all(chosen == anchor,axis=1)):
        chosen[-1] = anchor
    source = np.zeros_like(eligible)
    source[chosen[:,0],chosen[:,1]] = True
    return source


def _construct(rng, water, kind):
    for _ in range(120):
        base, holes, dams = _geometry(rng,kind)
        walls = base & ~holes
        halo = _halo(holes)
        options = _pool_options(walls,halo | holes)
        h, w = base.shape
        rows = np.arange(h)[:,None]
        bottom = [p for p in options if p[h-2].any() and
                  (kind != 0 or _branches(p) == 1) and
                  (kind != 1 or _branches(p) >= 2)]
        rng.shuffle(bottom)
        if not bottom:
            continue
        pools = [bottom[0]]
        # Each closed dam can retain a separate supply above it. No new rule:
        # buried defects do not connect its upper and lower faces.
        for top, end, opened in dams:
            if not opened:
                upper = [p for p in options if p[top-1].any() and
                         int(np.nonzero(p)[0].max()) == top-1]
                if upper:
                    pools.append(upper[int(rng.integers(len(upper)))])
        labels = _routes(walls,pools)
        source = np.zeros_like(base)
        try:
            for k, pool in enumerate(pools):
                if k == 0:
                    # At least one source must cross the lowest open dam.
                    limit = dams[-1][0]
                    if kind == 3 and dams[-2][2]:
                        limit = dams[-2][0]
                    high = rows < limit
                else:
                    high = rows < int(np.nonzero(pool)[0].min())
                eligible = ((labels == 1 << k) & ~walls & ~holes & ~halo)
                source |= _pick_source(rng,eligible,pool,high,int(rng.integers(4)))
            wet = np.logical_or.reduce(pools)
            if any(not np.array_equal(_pour(walls,source,rev,order),wet)
                   for rev in (False,True) for order in (False,True)):
                continue
            if np.array_equal(_pour(base,source),wet):
                continue
            if np.array_equal(source.sum(axis=0),wet.sum(axis=0)):
                continue
            solid = int(rng.choice([c for c in range(1,10) if c not in (2,water)]))
            # Specks do not touch the floor teeth; hence no hidden extension
            # of a legitimate grounded rectangle can masquerade as corruption.
            near_ground = np.zeros_like(base)
            ground = base.copy()
            for top,end,_ in dams:
                ground[top:end,1:-1] = False
            near_ground |= _halo(ground) | ground
            available = ~base & ~source & ~near_ground & ~halo & ~wet
            extras = np.zeros_like(base)
            points = np.argwhere(available)
            rng.shuffle(points)
            wanted = int(rng.integers(1,6))
            for r,c in points:
                if extras[r].sum() >= 2:
                    continue
                extras[r,c] = True
                if extras.sum() == wanted:
                    break
            if not extras.any():
                continue
            inp = np.where(base,solid,0).astype(np.int8)
            inp[holes] = 0
            inp[source] = water
            inp[extras] = solid
            # Ground truth is painted from constructed masks, not solve().
            out = np.where(walls,solid,0).astype(np.int8)
            out[wet] = water
            for r,c in np.argwhere(holes):
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        rr,cc = int(r+dr),int(c+dc)
                        if 0 <= rr < h and 0 <= cc < w and not holes[rr,cc]:
                            out[rr,cc] = 2
            decoded = _decode(inp)
            if not (np.array_equal(decoded[3],base) and
                    np.array_equal(decoded[4],holes) and
                    np.array_equal(decoded[5],extras)):
                continue
            return {"input":inp,"output":out}
        except ValueError:
            continue
    raise RuntimeError("bounded porous-dam construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train,num_test = _counts(num_train,num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    water = int(rng.choice([1,3,4,5,6,7,8,9]))
    puzzle = {"train":[],"test":[]}
    seen = set()
    for split,count in (("train",num_train),("test",num_test)):
        for i in range(count):
            kind = (i if i < 2 else 2) if split == "train" else 3
            for _ in range(12):
                pair = _construct(rng,water,kind)
                key = (pair["input"].shape,pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain distinct inputs")
    return puzzle


def _inspect(pair, challenge=False):
    a, solid, water, base, holes, extras = _decode(pair["input"])
    out = np.asarray(pair["output"])
    walls, halo = base & ~holes, _halo(holes)
    source, wet = a == water, out == water
    assert out.shape == a.shape and np.any(a != out)
    assert holes.any() and extras.any()
    assert np.all(out[holes] == 0), "missing centers must stay empty"
    assert np.array_equal(out == 2,halo), "exact union of eight-neighbor halos"
    assert not np.any(wet & (walls | holes | halo))
    assert int(wet.sum()) == int(source.sum()) > 0, "conserved volume"
    assert np.all(out[walls & ~halo] == solid), "preserve intact masonry"
    assert np.all(out[extras] == 0), "erase far-away extra specks"
    assert np.all(out[~walls & ~wet & ~halo] == 0)
    assert np.argwhere(wet)[:,0].sum() > np.argwhere(source)[:,0].sum()
    pools = _components(wet)
    rows = np.arange(a.shape[0])[:,None]
    for pool in pools:
        top = int(np.nonzero(pool)[0].min())
        closed = [p for p in _components(~walls & (rows >= top))
                  if np.any(p & pool)]
        assert len(closed) == 1 and np.array_equal(closed[0],pool), "full stable levels"
    routes = _routes(walls,pools)
    assigned = np.zeros_like(source)
    for k,pool in enumerate(pools):
        supply = source & (routes == 1 << k)
        assert int(supply.sum()) == int(pool.sum()), "unique dry catchment allocation"
        assigned |= supply
    assert np.array_equal(assigned,source)
    assert np.array_equal(solve(a.copy()),out)
    if challenge:
        for rev in (False,True):
            for order in (False,True):
                assert np.array_equal(_pour(walls,source,rev,order),wet)
    assert not np.array_equal(_pour(base,source),wet), "sealing defects must change flow"
    assert not np.array_equal(source,wet), "liquid motion must matter"
    assert not np.array_equal(source.sum(axis=0),wet.sum(axis=0)), "teach lateral flow"
    dam_rows = (base[1:-1,1:-1].sum(axis=1) == a.shape[1]-2)
    groups = []
    for r in np.flatnonzero(dam_rows)+1:
        if not groups or r != groups[-1][-1]+1:
            groups.append([int(r)])
        else:
            groups[-1].append(int(r))
    openings = []
    buried = 0
    for rr in groups:
        mask = np.zeros_like(holes)
        mask[rr] = holes[rr]
        opened = False
        for part in _components(mask):
            opened |= bool(part[rr[0]].any() and part[rr[-1]].any())
            buried += int(not part[rr[0]].any() and not part[rr[-1]].any())
        openings.append(opened)
    crossed = sum(bool(source[:rr[0]].sum() > wet[:rr[0]].sum())
                  for rr,opened in zip(groups,openings) if opened)
    return {"shape":a.shape,"water":water,"solid":solid,"dams":len(groups),
            "open_dams":sum(openings),"closed_dams":len(groups)-sum(openings),
            "crossed_open_dams":crossed,
            "holes":int(holes.sum()),"buried_voids":buried,"extras":int(extras.sum()),
            "volume":int(wet.sum()),"pools":len(pools),
            "branches":max(_branches(p) for p in pools),
            "levels":len({int(np.nonzero(p)[0].min()) for p in pools}),
            "source_components":len(_components(source)),
            "structure":hashlib.sha256(np.stack((base,holes,source)).tobytes()).hexdigest()}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen, waters, records = set(),set(),{"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            for value in pair.values():
                a = np.asarray(value)
                assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
                assert np.issubdtype(a.dtype,np.integer) and ((0 <= a) & (a <= 9)).all()
            a = np.asarray(pair["input"])
            key = (a.shape,a.tobytes())
            assert key not in seen
            seen.add(key)
            f = _inspect(pair)
            records[split].append(f)
            waters.add(f["water"])
    assert len(waters) == 1
    first,second = records["train"][:2]
    assert first["dams"] == 1 and first["open_dams"] == 1 and first["branches"] == 1
    assert first["buried_voids"] >= 1
    assert second["dams"] == 2 and second["closed_dams"] == 1
    assert second["pools"] >= 2 and second["branches"] >= 2
    assert all(f["dams"] == 3 for f in records["test"])
    assert all(f["dams"] <= 2 for f in records["train"])


def self_test():
    """42 host seeds, all count pairs, and bounded symmetry/ablation checks.

    No files or traces are written. Only hashes and scalar diversity summaries
    survive each iteration; full generated puzzles are released immediately.
    """
    shapes, structures, volumes, holes, palettes, signatures = (set() for _ in range(6))
    features = []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train","test"):
            for pair in puzzle[split]:
                f = _inspect(pair,challenge=True)
                features.append({k:v for k,v in f.items() if k != "structure"})
                shapes.add(f["shape"])
                structures.add(f["structure"])
                volumes.add(f["volume"])
                holes.add(f["holes"])
                palettes.add((f["water"],f["solid"]))
                for role in ("input","output"):
                    digest.update(pair[role].tobytes())
                # Reflection keeps gravity down; arbitrary rotation does not.
                assert np.array_equal(solve(pair["input"][:,::-1]),pair["output"][:,::-1])
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(structures) >= 180
    assert len(volumes) >= 30 and len(holes) >= 12 and len(palettes) >= 20
    assert {f["dams"] for f in features} == {1,2,3}
    assert max(f["pools"] for f in features) >= 3
    assert max(f["branches"] for f in features) >= 3
    assert max(f["open_dams"] for f in features) >= 2
    assert sum(f["levels"] >= 2 for f in features) >= 42
    assert sum(f["source_components"] > 1 for f in features) >= 80
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            puzzle = generate_puzzle(20261022,a,b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
            validate_generated_puzzle(puzzle)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p,q = generate_puzzle(0),generate_puzzle(0)
    assert all(np.array_equal(x[k],y[k]) for s in p
               for x,y in zip(p[s],q[s]) for k in x)
    return {"seeds":42,"pairs":len(features),"shapes":len(shapes),
            "structures":len(structures),"volumes":len(volumes),
            "hole_counts":len(holes),"palettes":len(palettes)}
