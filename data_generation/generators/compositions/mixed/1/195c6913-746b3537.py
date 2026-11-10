"""Cyclic staircase painting followed by adjacent full-vector run collapse.

No parent module is imported.  The input-only decoder, unit-cell route walker,
and two-axis projection implement the rule described in task_spec.md.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0,267818639,260847847,271021902,848316480,435950755,449069427,
207375853,838954589,319628233,162905588,447724307,848665441,154096135,
303113090,836330524,896641167,291420565,497437991,157169819,953691363,
980211422,377213288,6633229,525648012,183592690,955471253,728330277,
243437845,136438047,282810655,938383739,998632254,212838114,560860799,
443854856,105390508,869368085,439124739,594990315,846609293,725954998)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid of colors 0..9")
    return a.astype(np.int8, copy=True)


def _orient(a, code):
    return np.rot90(a[:, ::-1] if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -(code % 4))
    return (b[:, ::-1] if code >= 4 else b).copy()


def _inverse_point(p, shape, code):
    r, c = p
    h, w = shape
    r, c = ((r,c), (c,h-1-r), (h-1-r,w-1-c), (w-1-c,r))[code % 4]
    cw = h if code % 2 else w
    return int(r), int(cw-1-c if code >= 4 else c)


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(a, color):
    remaining = set(map(tuple, np.argwhere(a == color)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for p in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _decode(a):
    colors, counts = np.unique(a, return_counts=True)
    assert len(colors) >= 5, "two terrains, two program colors and a marker required"
    terrain_colors = set(map(int, colors[np.argsort(counts)[-2:]]))
    swatches, starts = [], []
    for color in map(int, colors):
        if color in terrain_colors:
            continue
        for cells in _components(a, color):
            rs, cs = sorted({p[0] for p in cells}), sorted({p[1] for p in cells})
            if len(cells) == 4 and len(rs) == len(cs) == 2 and rs[1]-rs[0] == cs[1]-cs[0] == 1:
                swatches.append((color, cells))
            else:
                assert len(cells) == 1, "annotations must be 2x2; starts must be singletons"
                starts.append(cells[0])
    assert 3 <= len(swatches) <= 7 and 1 <= len(starts) <= 3
    firsts = {int(a[p]) for p in starts}
    assert len(firsts) == 1
    first = next(iter(firsts))
    annotation_cells = {p for _, cells in swatches for p in cells}
    rings = {int(a[q]) for p in annotation_cells
             for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1))
             if _inside(a.shape,q) and q not in annotation_cells}
    assert len(rings) == 1 and rings <= terrain_colors
    bg = next(iter(rings))
    route = next(iter(terrain_colors - {bg}))
    restored = a.copy()
    for p in annotation_cells:
        restored[p] = bg
    for p in starts:
        restored[p] = route
    assert set(map(int, np.unique(restored))) == terrain_colors
    candidates = []
    for code in range(8):
        terrain = _unorient(restored, code)
        mask = terrain == route
        ss = tuple(sorted(_inverse_point(p,a.shape,code) for p in starts))
        patches = []
        for color, cells in swatches:
            cells = tuple(_inverse_point(p,a.shape,code) for p in cells)
            patches.append((min(p[0] for p in cells), min(p[1] for p in cells), color, cells))
        for marker in patches:
            legend = sorted((s for s in patches if s is not marker), key=lambda s:s[1])
            if len({s[0] for s in legend}) != 1:
                continue
            if abs(marker[0]-legend[0][0]) < 3:
                continue
            if any(b[1]-x[1] < 3 for x,b in zip(legend,legend[1:])):
                continue
            word = tuple(s[2] for s in legend)
            if word[0] != first or marker[2] in word:
                continue
            if not all(r >= legend[0][0]+4 and c+1 < mask.shape[1] and mask[r,c+1]
                       and (c == 0 or not mask[r,c-1]) for r,c in ss):
                continue
            candidates.append(dict(code=code, terrain=terrain, mask=mask, starts=ss,
                                   word=word, marker=marker[2], legend=legend,
                                   patches=patches, bg=bg, route=route))
    assert len(candidates) == 1, "scene must identify one legend and movement orientation"
    return candidates[0]


def _walk(mask, start):
    """Unit-cell walk: right and up alternate; a bend cell is counted once."""
    p, axis = start, 0
    path, collisions, legs = [], [], [[]]
    while True:
        assert _inside(mask.shape,p) and mask[p] and p not in path
        path.append(p)
        legs[-1].append(p)
        dr, dc = ((0,1),(-1,0))[axis]
        ahead = (p[0]+dr,p[1]+dc)
        if not _inside(mask.shape,ahead):
            end = "edge"
            break
        if mask[ahead]:
            p = ahead
            continue
        collisions.append(ahead)
        dr, dc = ((-1,0),(0,1))[axis]
        turn = (p[0]+dr,p[1]+dc)
        if not _inside(mask.shape,turn) or not mask[turn]:
            end = "blocked"
            break
        axis = 1-axis
        p = turn
        legs.append([])
    return dict(path=tuple(path), collisions=tuple(collisions),
                legs=tuple(tuple(x) for x in legs), end=end)


def _paint(info, mode="correct"):
    out = info["terrain"].copy()
    word = info["word"]
    if mode == "dedup_word":
        word = tuple(dict.fromkeys(word))
    offset = 0
    for record in info["records"]:
        phase = offset if mode == "global_phase" else 0
        for leg in record["legs"]:
            if mode == "reset_bends":
                phase = 0
            for p in leg:
                out[p] = word[phase % len(word)]
                phase += 1
            if mode == "count_markers":
                phase += 1
        if mode != "no_markers":
            for p in record["collisions"]:
                out[p] = info["marker"]
        offset += len(record["path"])
    return out


def _keep(a):
    """Compute both masks from the SAME uncompressed array."""
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:,1:] != a[:,:-1], axis=0)]
    return rows, cols


def _collapse(a):
    rows, cols = _keep(a)
    return a[np.ix_(rows,cols)].copy()


def _analyze(a):
    info = _decode(a)
    info["records"] = [_walk(info["mask"],s) for s in info["starts"]]
    occupied = set()
    annotations = {p for s in info["patches"] for p in s[3]}
    for record in info["records"]:
        footprint = set(record["path"]) | set(record["collisions"])
        assert not footprint & occupied, "routes and collisions may not intersect"
        assert not footprint & annotations, "annotations may not mask a route or collision"
        occupied |= footprint
    info["painted"] = _paint(info)
    info["canonical_output"] = _collapse(info["painted"])
    info["output"] = _orient(info["canonical_output"],info["code"])
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode, paint at input resolution, then collapse exact vector runs."""
    return _analyze(_grid(input_grid))["output"]


def _same(a,b):
    return a.shape == b.shape and np.array_equal(a,b)


def _global_unique(a):
    def firsts(vectors):
        seen, indices = set(), []
        for i,v in enumerate(vectors):
            key = v.tobytes()
            if key not in seen:
                indices.append(i)
                seen.add(key)
        return indices
    return a[np.ix_(firsts(a),firsts(a.T))]


def _evidence(info, a):
    g, target = info["painted"], info["canonical_output"]
    rows, cols = _keep(g)
    evidence = set()
    for name in ("dedup_word","reset_bends","count_markers","global_phase","no_markers"):
        if not _same(target,_collapse(_paint(info,name))):
            evidence.add(name)
    if any(len(r["path"]) >= 2*len(info["word"]) for r in info["records"]):
        evidence.add("cycling")
    if np.any(~rows):
        evidence.add("collapse_rows")
    if np.any(~cols):
        evidence.add("collapse_cols")
    terrain_rows, terrain_cols = _keep(info["terrain"])
    if not np.array_equal(rows,terrain_rows):
        evidence.add("route_controls_rows")
    if not np.array_equal(cols,terrain_cols):
        evidence.add("route_controls_cols")
    if any(not rows[i] and np.any(g[i] != info["bg"]) for i in range(len(rows))):
        evidence.add("nonblank_rows")
    if any(not cols[i] and np.any(g[:,i] != info["bg"]) for i in range(len(cols))):
        evidence.add("nonblank_cols")
    if len({r.tobytes() for r in target}) < len(target):
        evidence.add("separated_rows")
    if len({c.tobytes() for c in target.T}) < target.shape[1]:
        evidence.add("separated_cols")
    if not _same(target,_collapse(info["terrain"])):
        evidence.add("painting_necessary")
    canonical_input = _unorient(a,info["code"])
    ri, ci = _keep(canonical_input)
    if not _same(target,g[np.ix_(ri,ci)]):
        evidence.add("paint_before_collapse")
    return evidence


REQUIRED_EVIDENCE = {"dedup_word","reset_bends","count_markers","global_phase",
                     "no_markers","cycling","collapse_rows","collapse_cols",
                     "nonblank_rows","nonblank_cols","separated_rows",
                     "separated_cols","painting_necessary","paint_before_collapse",
                     "route_controls_rows","route_controls_cols"}


def _word(rng, length, informative):
    d = int(rng.integers(2,min(6,length)+1))
    if informative:
        d = min(d,length-1)
    word = list(range(d)) + list(map(int,rng.integers(0,d,length-d)))
    rng.shuffle(word)
    if informative and len(set(word)) == len(word):
        word[1] = word[0]
    return tuple(word)


def _partition(rng, total, n, minimum=1):
    values = np.full(n,minimum,dtype=int)
    for _ in range(total-int(values.sum())):
        values[int(rng.integers(n))] += 1
    return values


def _terrain(rng, h, w, top, test, shift):
    """Independent block profiles create thick ledges, bays and unpainted lobes."""
    bottom = int(rng.integers(2,4))
    left_margin, right_margin = int(rng.integers(2,4)), int(rng.integers(2,4))
    ah, aw = h-top-bottom, w-left_margin-right_margin
    if shift == "geometry":
        low, high = (5,7) if test else (3,4)
    else:
        low, high = 3, 7 if test else 6
    nb = int(rng.integers(low,min(high,ah//2)+1))
    nc = int(rng.integers(5,min(12,aw)+1))
    heights = _partition(rng,ah,nb,2)
    widths = _partition(rng,aw,nc)
    bounds = np.r_[left_margin,left_margin+np.cumsum(widths)]
    spans = []
    previous = None
    # Top-to-bottom boundaries need not be monotone or constant-width.
    for i in range(nb):
        if rng.random() < .55:
            center = int(round((nb-1-i)*(nc-3)/max(1,nb-1)))
            l = int(np.clip(center+int(rng.integers(-1,2)),0,nc-3))
        else:
            l = int(rng.integers(0,nc-2))
        r = int(rng.integers(l+2,nc+1))
        if previous is not None:
            # Consecutive spans overlap: one connected terrain, no permanent bridge.
            l = min(l,previous[1]-1)
            r = max(r,previous[0]+1)
            if (l,r) == previous:
                if r < nc:
                    r += 1
                elif l > 0:
                    l -= 1
                else:
                    l += 1
        spans.append((l,r))
        previous = l,r
    mask = np.zeros((h,w),dtype=bool)
    row = top
    for height,(l,r) in zip(heights,spans):
        mask[row:row+height,bounds[l]:bounds[r]] = True
        row += int(height)
    return mask


def _select(rng, mask, count, bent):
    candidates = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]-1):
            if mask[r,c] and mask[r,c+1] and (c == 0 or not mask[r,c-1]):
                record = _walk(mask,(r,c))
                if len(record["path"]) >= 4:
                    candidates.append(((r,c),record))
    rng.shuffle(candidates)
    if bent:
        candidates.sort(key=lambda item:(len(item[1]["legs"]),len(item[1]["path"])),reverse=True)
    chosen, used = [], set()
    for start,record in candidates:
        footprint = set(record["path"]) | set(record["collisions"])
        if footprint & used:
            continue
        if any(abs(start[0]-s[0])+abs(start[1]-s[1]) <= 2 for s,_ in chosen):
            continue
        chosen.append((start,record))
        used |= footprint
        if len(chosen) == count:
            return chosen
    return []


def _candidate(rng, code, test=False, witness=False, restart=False, shift="program"):
    for attempt in range(500):
        lo, hi = (5,6) if test and shift == "program" else (2,4) if shift == "program" else (2,6 if test else 5)
        length = int(rng.integers(max(lo,3 if witness else 2),hi+1))
        abstract = _word(rng,length,witness)
        palette = tuple(map(int,rng.permutation(10)))
        bg, route, marker = palette[:3]
        word = tuple(palette[3+i] for i in abstract)
        minimum_width = max(18,3*length+2)
        w = int(rng.integers(minimum_width,31))
        hlo, hhi = ((26,30) if test else (18,23)) if shift == "size" else ((23,30) if test else (18,27))
        h = int(rng.integers(hlo,hhi+1))
        legend_row = int(rng.integers(0,3))
        top = legend_row+int(rng.integers(4,6))
        mask = _terrain(rng,h,w,top,test,shift)
        if min(int(mask.sum()),int((~mask).sum())-4*(length+1)) < 30:
            continue
        count = 2 if restart else int(rng.choice([1,2,3],p=[.50,.42,.08]))
        chosen = _select(rng,mask,count,witness)
        if not chosen:
            continue
        records = [r for _,r in chosen]
        if witness and max(len(r["legs"]) for r in records) < 3:
            continue
        # Place a translated, unevenly spaced ordered program in the clear header.
        slack = w-(3*length-1)
        legend_left = int(rng.integers(0,min(3,slack)+1))
        columns = [legend_left]
        remaining = w-legend_left-(3*length-1)
        for _ in range(length-1):
            extra = int(rng.integers(0,min(2,remaining)+1))
            columns.append(columns[-1]+3+extra)
            remaining -= extra
        terrain = np.where(mask,route,bg).astype(np.int8)
        a = terrain.copy()
        annotations = set()
        for c,color in zip(columns,word):
            a[legend_row:legend_row+2,c:c+2] = color
            annotations |= {(r,s) for r in range(legend_row,legend_row+2) for s in range(c,c+2)}
        # A separate swatch has a one-cell background ring and no route interference.
        forbidden = annotations.copy()
        for _,record in chosen:
            forbidden |= set(record["path"]) | set(record["collisions"])
        marker_positions = []
        for r in range(legend_row+3,h-2):
            for c in range(1,w-2):
                cells = {(rr,cc) for rr in range(r-1,r+3) for cc in range(c-1,c+3)}
                if not cells & forbidden and not np.any(mask[r-1:r+3,c-1:c+3]):
                    marker_positions.append((r,c))
        if not marker_positions:
            continue
        mr,mc = marker_positions[int(rng.integers(len(marker_positions)))]
        a[mr:mr+2,mc:mc+2] = marker
        for start,_ in chosen:
            a[start] = word[0]
        a = _orient(a,code)
        try:
            info = _analyze(a)
        except AssertionError:
            continue
        evidence = _evidence(info,a)
        need = {"nonblank_rows","nonblank_cols","dedup_word","reset_bends","count_markers","cycling"} if witness else set()
        if restart:
            need.add("global_phase")
        if not need <= evidence:
            continue
        return {"input":a,"output":info["output"]}
    raise RuntimeError("bounded geometry sampler exhausted")


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _digest(a):
    return hashlib.sha256(bytes(a.shape)+a.astype(np.uint8).tobytes()).hexdigest()


def _band_count(mask):
    return sum(bool(mask[i].any()) and (i == 0 or not np.array_equal(mask[i],mask[i-1]))
               for i in range(len(mask)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be in 1..4 and less than num_train")
    rng = np.random.default_rng(seed)
    code = int(rng.integers(8))
    shift = str(rng.choice(["program","geometry","size"]))
    # Random positions for the two complementary training witnesses.
    witness, restart = map(int,rng.choice(int(num_train),size=2,replace=False))
    result, seen = {"train":[],"test":[]}, set()
    for split,n in (("train",int(num_train)),("test",int(num_test))):
        for i in range(n):
            for _ in range(20):
                pair = _candidate(rng,code,split == "test",split == "train" and i == witness,
                                  split == "train" and i == restart,shift)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("unable to construct distinct inputs")
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence, codes = set(), set(), set()
    shift_features = {"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            shift_features[split].append((len(info["word"]),info["mask"].shape[0],_band_count(info["mask"])))
            codes.add(info["code"])
            assert _same(out,info["output"]), "output disagrees with input-only rule"
            assert len(_components(info["mask"].astype(np.int8),1)) == 1
            rows,cols = _keep(info["painted"])
            assert not rows.all() and not cols.all(), "both axes must actually collapse"
            assert out.shape[0] < a.shape[0] and out.shape[1] < a.shape[1]
            # Projection retains every collision/paint color and background restoration.
            assert set(map(int,np.unique(out))) == set(map(int,np.unique(info["painted"])))
            assert np.all(np.any(out[1:] != out[:-1],axis=1))
            assert np.all(np.any(out[:,1:] != out[:,:-1],axis=0))
            assert not _same(info["canonical_output"],_global_unique(info["painted"]))
            if split == "train":
                evidence |= _evidence(info,a)
    assert len(codes) == 1, "one scene orientation convention per generated puzzle"
    assert REQUIRED_EVIDENCE <= evidence, REQUIRED_EVIDENCE-evidence
    assert any(min(v[i] for v in shift_features["test"]) > max(v[i] for v in shift_features["train"])
               for i in range(3)), "test must extend program length, canonical height or terrain bands"


def self_test():
    """Bounded 42-seed audit, count contract, regression and negative-label checks."""
    # Adjacent equality is of complete vectors, not individual cell runs.
    fixture = np.array([[1,1,2,1,1],[1,1,2,1,1],[3,3,4,3,3],
                        [1,1,2,1,1],[1,1,2,1,1]],dtype=np.int8)
    assert _same(_collapse(fixture),np.array([[1,2,1],[3,4,3],[1,2,1]]))
    # A difference in a removed duplicate row cannot be the only column distinction.
    rows,cols = _keep(fixture)
    assert _same(_collapse(fixture),_collapse(fixture[rows])[:, :])
    assert _same(_collapse(fixture),_collapse(fixture[:,cols]))
    # Canvas exits terminate immediately, even with available terrain round a corner.
    mask = np.zeros((5,9),bool)
    mask[3,:] = True
    mask[1:4,-1] = True
    rec = _walk(mask,(3,0))
    assert len(rec["legs"]) == 1 and not rec["collisions"] and rec["end"] == "edge"
    shapes, patterns, geometries, palette_roles, counts, codes, fingerprints = (set() for _ in range(7))
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        fingerprints.add(tuple(_digest(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for p,q in zip(puzzle[split],again[split]):
                assert _same(p["input"],q["input"]) and _same(p["output"],q["output"])
                info = _analyze(p["input"])
                shapes.add(p["input"].shape)
                labels = {}
                patterns.add(tuple(labels.setdefault(c,len(labels)) for c in info["word"]))
                geometries.add(_digest(info["mask"].astype(np.int8)))
                palette_roles.add((info["bg"],info["route"],info["marker"]))
                counts.add(len(info["starts"]))
                codes.add(info["code"])
        # Reject a label mutation without changing its input.
        first = puzzle["train"][0]
        bad = {s:[dict(p) for p in puzzle[s]] for s in puzzle}
        bad["train"][0] = dict(first,output=(first["output"]+1)%10)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("mutated label accepted")
    assert len(fingerprints) == 42
    assert len(shapes) >= 25 and len(patterns) >= 15 and len(geometries) >= 150
    assert len(palette_roles) >= 75 and codes == set(range(8)) and {1,2,3} <= counts
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(12345,nt,ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(4.0,1),(True,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
