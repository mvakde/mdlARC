"""Repaired paint stencils. Standard library + NumPy; no I/O during generation.

The solver reads its argument only. Generation constructs intended walls first;
labels composite those walls, whereas solve recovers walls and visits pixels in
priority order. No stored examples or parent modules are used.
"""
from collections import deque
import hashlib
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _strong(mask):
    result = np.zeros_like(mask)
    for a, b in ((mask, result), (mask.T, result.T)):
        if a.shape[1] < 3:
            continue
        runs = a[:, :-2] & a[:, 1:-1] & a[:, 2:]
        for k in range(3):
            b[:, k:k + runs.shape[1]] |= runs
    return result


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _gaps(mask):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        a = mask if dc else mask.T
        if a.shape[1] < 5:
            continue
        hit = a[:, :-4] & a[:, 4:] & ~(a[:, 1:-3] | a[:, 2:-2] | a[:, 3:-1])
        for y, x in np.argwhere(hit):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        component, todo = [], [start]
        while todo:
            r, c = todo.pop()
            component.append((r, c))
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in cells:
                    cells.remove(p)
                    todo.append(p)
        result.append(component)
    return result


def _diagonal(cells):
    cells = sorted(cells)
    return (len(cells) in (2, 3)
            and all(cells[i+1][0] - cells[i][0] == 1 for i in range(len(cells)-1))
            and len({cells[i+1][1] - cells[i][1] for i in range(len(cells)-1)}) == 1
            and abs(cells[1][1] - cells[0][1]) == 1)


def _repair(raw):
    strong = _strong(raw)
    added = np.zeros_like(raw)
    evidence = []
    for r, c, dr, dc in _gaps(raw):
        if not (_at(raw, r-dr, c-dc) or _at(raw, r+5*dr, c+5*dc)):
            continue
        matches = []
        for endpoint in (0, 4):
            ar, ac = r + endpoint*dr, c + endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                spur = tuple((ar + k*sr, ac + k*sc) for k in (1, 2))
                if (all(_at(raw, rr, cc) and not strong[rr, cc] for rr, cc in spur)
                        and not _at(raw, ar+3*sr, ac+3*sc)):
                    matches.append(spur)
        if len(matches) > 1:
            raise ValueError("ambiguous spur attachment")
        if matches:
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            for p in gap:
                added[p] = True
            evidence.append((gap, matches[0]))
    repaired = _strong(raw | added)
    removed = raw & ~repaired
    if any(not _diagonal(x) for x in _components(removed, True)):
        raise ValueError("unsupported cells are not diagonal scraps")
    if np.any(added & ~repaired):
        raise ValueError("unsupported repair")
    for gap, spur in evidence:
        if not all(removed[p] for p in spur):
            raise ValueError("spur was not removed")
    return repaired, evidence, removed


def _interior(wall):
    """Flood a padded canvas; the canvas edge is never an implicit wall."""
    barrier = np.pad(wall, 1)
    outside = np.zeros_like(barrier)
    outside[0, 0] = True
    todo = deque([(0, 0)])
    h, w = barrier.shape
    while todo:
        r, c = todo.popleft()
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < h and 0 <= cc < w and not barrier[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                todo.append((rr, cc))
    return ~outside[1:-1, 1:-1] & ~wall


def _layers(walls):
    return [w | _interior(w) for w in walls]


def _compose(bg, cues, fills, walls, supports, legend_grid=None, transparent=False):
    """Construction path: whole masks, bottom to top."""
    out = np.full(walls[0].shape, bg, dtype=np.int8)
    for cue, fill, wall, support in reversed(list(zip(cues, fills, walls, supports))):
        if not (transparent and fill == bg):
            out[support] = fill
        out[wall] = cue
    if legend_grid is not None:
        r, c, n, values = legend_grid
        out[r:r+n, c:c+2] = values
    return out


def _predict(parsed):
    """Solver path: choose the first covering legend row at each scene pixel."""
    walls, supports = parsed["walls"], parsed["supports"]
    out = np.full(walls[0].shape, parsed["bg"], dtype=np.int8)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            for i in range(len(walls)):
                if supports[i][r, c]:
                    out[r, c] = parsed["cues"][i] if walls[i][r, c] else parsed["fills"][i]
                    break
    r, c, n, values = parsed["legend"]
    out[r:r+n, c:c+2] = values
    return out


def _parse(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if np.sum(counts == counts[bg]) != 1:
        raise ValueError("background is not uniquely most frequent")
    h, w = grid.shape
    candidates = []
    for c in range(w-1):
        rows = np.flatnonzero(grid[:, c] != bg)
        for run in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1):
            n = len(run)
            if not 1 <= n <= 3:
                continue
            r = int(run[0])
            cues = list(map(int, grid[r:r+n, c]))
            if len(set(cues)) != n:
                continue
            moat = grid[max(0, r-1):min(h, r+n+1), max(0, c-1):min(w, c+3)].copy()
            moat[r-max(0, r-1):r+n-max(0, r-1), c-max(0, c-1):c+2-max(0, c-1)] = bg
            if np.any(moat != bg):
                continue
            scene = grid.copy()
            scene[r:r+n, c:c+2] = bg
            colors = set(map(int, np.unique(scene))) - {bg}
            if colors != set(cues):
                continue
            try:
                raw = [scene == cue for cue in cues]
                recovered = [_repair(m) for m in raw]
                walls = [x[0] for x in recovered]
                if any(not m.any() for m in walls):
                    continue
                supports = _layers(walls)
                if any(s[r:r+n, c:c+2].any() for s in supports):
                    continue
                if not any(x[1] for x in recovered):
                    continue
                candidates.append(dict(bg=bg, cues=cues, fills=list(map(int, grid[r:r+n, c+1])),
                    raw=raw, walls=walls, supports=supports,
                    evidence=[x[1] for x in recovered], removed=[x[2] for x in recovered],
                    legend=(r, c, n, grid[r:r+n, c:c+2].copy())))
            except ValueError:
                continue
    if len(candidates) != 1:
        raise ValueError(f"expected one readable legend, found {len(candidates)}")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from this input; no seed, examples, or state."""
    return _predict(_parse(_grid(input_grid)))


def _bbox_support(wall):
    cells = np.argwhere(wall)
    support = np.zeros_like(wall)
    lo, hi = cells.min(axis=0), cells.max(axis=0)
    support[lo[0]:hi[0]+1, lo[1]:hi[1]+1] = True
    return support


def _all_closed(wall):
    result = wall.copy()
    for r, c, dr, dc in _gaps(wall):
        for k in (1, 2, 3):
            result[r+k*dr, c+k*dc] = True
    return result


def _witnesses(p):
    walls, supports = p["walls"], p["supports"]
    bg, cues, fills, legend = p["bg"], p["cues"], p["fills"], p["legend"]
    actual = _predict(p)
    found = set()
    def render(ws=walls, ss=supports, cs=cues, fs=fills, transparent=False):
        return _compose(bg, cs, fs, ws, ss, legend, transparent)
    unhealed = [_strong(raw) for raw in p["raw"]]
    unhealed_support = _layers(unhealed)
    no_repair = render(unhealed, unhealed_support)
    affected_interior = np.logical_or.reduce([s & ~w & ~u for s, w, u in
                                            zip(supports, walls, unhealed_support)])
    if np.any((actual != no_repair) & affected_interior):
        found.add("repair_area")
    if not np.array_equal(actual, render(ss=[_bbox_support(w) for w in walls])):
        found.add("bbox")
    if not np.array_equal(actual, render(ws=walls[::-1], ss=supports[::-1], cs=cues[::-1], fs=fills[::-1])):
        found.add("priority")
    if not np.array_equal(actual, render(fs=cues)):
        found.add("lookup")
    if not np.array_equal(actual, render(transparent=True)):
        found.add("opaque")
    overclosed = [_all_closed(w) for w in walls]
    if not np.array_equal(actual, render(overclosed, _layers(overclosed))):
        # This also requires a true enclosure change, not merely a painted gap.
        if any(np.any(_interior(a) & ~s) for a, s in zip(overclosed, supports)):
            found.add("leak")
    for i, wall in enumerate(walls):
        if np.any(wall & (actual == cues[i])):
            found.add("retained_wall")
        if i and np.any(wall & (actual != cues[i]) & np.logical_or.reduce(supports[:i])):
            found.add("covered_wall")
    return found


def _halo(mask):
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[r:r+mask.shape[0], c:c+mask.shape[1]]
                                 for r in range(3) for c in range(3)])


def _outline(shape, vertices):
    mask = np.zeros(shape, dtype=bool)
    for (r, c), (rr, cc) in zip(vertices, vertices[1:] + vertices[:1]):
        if r == rr:
            mask[r, min(c, cc):max(c, cc)+1] = True
        elif c == cc:
            mask[min(r, rr):max(r, rr)+1, c] = True
        else:
            raise ValueError("nonorthogonal polygon")
    return mask


def _polygon(rng, shape, bounds, kind=None, divide=True):
    r0, c0, r1, c1 = bounds
    h, w = r1-r0+1, c1-c0+1
    kind = kind or str(rng.choice(["rectangle", "elbow", "skyline", "notch"]))
    if kind == "notch" and w < 10:
        kind = "elbow"
    if kind == "rectangle":
        vertices = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
    elif kind == "elbow":
        rm = int(rng.integers(r0+3, r1-2))
        cm = int(rng.integers(c0+3, c1-2))
        vertices = [(r0, c0), (r0, c1), (rm, c1), (rm, cm), (r1, cm), (r1, c0)]
    elif kind == "notch":
        ca = int(rng.integers(c0+3, c1-5))
        cb = int(rng.integers(ca+3, c1-2))
        rm = int(rng.integers(r0+3, r1-2))
        vertices = [(r0,c0), (r0,ca), (rm,ca), (rm,cb), (r0,cb), (r0,c1), (r1,c1), (r1,c0)]
    else:
        bands = int(rng.integers(2, min(4, (w-1)//3)+1))
        widths = np.full(bands, 3, dtype=int)
        widths += rng.multinomial(w-1-3*bands, np.full(bands, 1/bands))
        edges = [c0] + list(c0 + np.cumsum(widths))
        levels = list(range(r0, r1-2, 3))
        heights = list(map(int, rng.choice(levels, size=bands)))
        heights[int(rng.integers(bands))] = r0
        vertices = [(r1,c0), (heights[0],c0)]
        for i in range(bands):
            vertices.append((heights[i], int(edges[i+1])))
            if i+1 < bands:
                vertices.append((heights[i+1], int(edges[i+1])))
        vertices.append((r1,c1))
    mask = _outline(shape, vertices)
    if kind == "rectangle" and divide and rng.random() < .35:
        columns = list(range(c0+3, c1-2))
        for _ in range(int(rng.integers(1, 3))):
            if not columns:
                break
            c = int(rng.choice(columns))
            mask[r0:r1+1, c] = True
            columns = [x for x in columns if abs(x-c) >= 3]
    if rng.integers(2):
        local = mask[r0:r1+1, c0:c1+1].copy()
        mask[r0:r1+1, c0:c1+1] = local[::-1]
    if rng.integers(2):
        local = mask[r0:r1+1, c0:c1+1].copy()
        mask[r0:r1+1, c0:c1+1] = local[:, ::-1]
    return mask


def _damage_candidates(wall):
    candidates = []
    for dr, dc in ((0, 1), (1, 0)):
        for r, c in np.argwhere(wall):
            r, c = int(r), int(c)
            if not all(_at(wall, r+k*dr, c+k*dc) for k in range(5)):
                continue
            if not (_at(wall, r-dr, c-dc) or _at(wall, r+5*dr, c+5*dc)):
                continue
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            if any(_at(wall, rr+dc, cc+dr) or _at(wall, rr-dc, cc-dr) for rr, cc in gap):
                continue
            for endpoint in (0, 4):
                ar, ac = r+endpoint*dr, c+endpoint*dc
                toward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    spur = tuple((ar+k*(toward*dr+side*dc), ac+k*(toward*dc+side*dr)) for k in (1, 2))
                    if all(0 <= rr < wall.shape[0] and 0 <= cc < wall.shape[1] and not wall[rr,cc]
                           for rr, cc in spur):
                        candidates.append((gap, spur))
    return candidates


def _open(rng, wall):
    candidates = _damage_candidates(wall)
    for j in rng.permutation(len(candidates)):
        gap, _ = candidates[int(j)]
        trial = wall.copy()
        for p in gap:
            trial[p] = False
        # Only a genuine opening that releases all this component's paint.
        if np.array_equal(trial, _strong(trial)) and not _interior(trial).any():
            return trial
    raise ValueError("cannot place an intentional leak")


def _bounds(rng, region, minimum=9):
    r0, c0, r1, c1 = region
    h, w = r1-r0+1, c1-c0+1
    if min(h,w) < minimum:
        raise ValueError("region too small")
    hh, ww = int(rng.integers(minimum, min(h,22)+1)), int(rng.integers(minimum, min(w,22)+1))
    r, c = int(rng.integers(r0, r1-hh+2)), int(rng.integers(c0, c1-ww+2))
    return r,c,r+hh-1,c+ww-1


def _geometry(rng, n, teach, shift, test):
    h, w = int(rng.integers(18,31)), int(rng.integers(18,31))
    if test and shift == "components":
        w = int(rng.integers(24,31))
    if test and shift == "damage":
        h, w = int(rng.integers(24,31)), int(rng.integers(24,31))
    shape, walls = (h,w), []
    # Margins are sampled; closed walls may touch any edge.
    margin = int(rng.choice([0,0,1,1,2]))
    region = (margin,margin,h-1-margin,w-1-margin)
    for i in range(n):
        if test and shift == "components" and i == 0:
            regions = [(0,0,h-1,w//2-2), (0,w//2+1,h-1,w-1)]
            parts = [_polygon(rng, shape, _bounds(rng, reg), divide=False) for reg in regions]
            parts[1] = _open(rng, parts[1])
            walls.append(parts[0] | parts[1])
            continue
        minimum = 14 if test and shift == "damage" and i == 0 else 9
        b = _bounds(rng, region, minimum)
        kind = "elbow" if teach == 1 and i == 0 else None
        if test and shift == "damage" and i == 0:
            kind = "rectangle"
        is_open = (teach == 0 and i == 1) or (teach < 0 and i > 0 and rng.random() < .22)
        wall = _polygon(rng, shape, b, kind, divide=not is_open)
        if is_open:
            wall = _open(rng, wall)
        walls.append(wall)
    if any(not np.array_equal(m, _strong(m)) for m in walls):
        raise ValueError("short unsupported wall")
    return walls


def _corrupt(rng, walls, main_target):
    raw = [m.copy() for m in walls]
    candidates = [_damage_candidates(m) for m in walls]
    occupied_clean = np.logical_or.reduce(walls)
    footprints = [np.zeros_like(walls[0]) for _ in walls]
    counts = [0]*len(walls)
    def damage(i, target=None):
        choices = [x for x in candidates[i] if target is None or target in x[0]]
        occupied = np.logical_or.reduce(raw)
        forbidden = _halo(footprints[i])
        for j in rng.permutation(len(choices)):
            gap, spur = choices[int(j)]
            if (not all(raw[i][p] for p in gap)
                    or any(occupied_clean[p] or occupied[p] for p in spur)
                    or any(forbidden[p] for p in gap+spur)):
                continue
            trial = raw[i].copy()
            for p in gap:
                trial[p] = False
            for p in spur:
                trial[p] = True
            try:
                clean, evidence, _ = _repair(trial)
            except ValueError:
                continue
            if np.array_equal(clean, walls[i]) and len(evidence) == counts[i]+1:
                raw[i] = trial
                counts[i] += 1
                for p in gap+spur:
                    footprints[i][p] = True
                return True
        return False
    # Crossings must be visible for both contours: a licensed tear recovers the
    # occluded one, so no hidden unmarked crossing convention is introduced.
    crossings = np.argwhere(np.sum(raw, axis=0) > 1)
    for j in rng.permutation(len(crossings)):
        p = tuple(map(int, crossings[int(j)]))
        while sum(m[p] for m in raw) > 1:
            owners = [i for i, m in enumerate(raw) if m[p]]
            if not any(damage(int(i), p) for i in rng.permutation(owners)):
                raise ValueError("crossing cannot be visibly reconstructed")
    for i in range(len(walls)):
        target = main_target if i == 0 else int(rng.integers(0,2))
        while counts[i] < target:
            if not damage(i):
                raise ValueError("not enough separated tears")
    # A bounded number of free scraps, each independently checked for ambiguity.
    for _ in range(int(rng.integers(3))):
        i = int(rng.integers(len(walls)))
        for _ in range(16):
            length = int(rng.choice([2,3]))
            r, c = int(rng.integers(walls[0].shape[0])), int(rng.integers(walls[0].shape[1]))
            dr, dc = int(rng.choice([-1,1])), int(rng.choice([-1,1]))
            cells = [(r+k*dr,c+k*dc) for k in range(length)]
            occupied = _halo(np.logical_or.reduce(raw) | occupied_clean)
            if not all(0 <= rr < occupied.shape[0] and 0 <= cc < occupied.shape[1] and not occupied[rr,cc]
                       for rr, cc in cells):
                continue
            trial = raw[i].copy()
            for p in cells:
                trial[p] = True
            try:
                clean, evidence, _ = _repair(trial)
                if np.array_equal(clean,walls[i]) and len(evidence) == counts[i]:
                    raw[i] = trial
                    break
            except ValueError:
                pass
    if np.any(np.sum(raw,axis=0) > 1):
        raise ValueError("input color collision")
    return raw


def _make_colored(rng, walls, raw, teach, fallback=False):
    n, shape = len(walls), walls[0].shape
    palette = list(map(int, rng.permutation(10)))
    bg, cues = palette[0], palette[1:n+1]
    fills = list(map(int, rng.integers(10,size=n)))
    if teach == 0 or fallback:
        fills = [palette[n+1+i] for i in range(n)]
    order = list(map(int, rng.permutation(n))) if teach != 0 and not fallback else list(range(n))
    walls, raw, cues, fills = ([items[i] for i in order] for items in (walls,raw,cues,fills))
    if teach == 1:
        fills[0] = bg
        fills[1] = int(rng.choice([x for x in range(10) if x not in (bg,cues[1])]))
    elif teach < 0 and n > 1 and not fallback and rng.random() < .30:
        fills[0] = bg
    supports = _layers(walls)
    grid = np.full(shape,bg,dtype=np.int8)
    for cue, m in zip(cues,raw):
        grid[m] = cue
    blocked = np.logical_or.reduce(raw+supports)
    options = []
    h,w = shape
    for r in range(h-n+1):
        for c in range(w-1):
            if not blocked[max(0,r-1):min(h,r+n+1),max(0,c-1):min(w,c+3)].any():
                options.append((r,c))
    if not options:
        raise ValueError("no isolated legend location")
    r,c = options[int(rng.integers(len(options)))]
    values = np.array(list(zip(cues,fills)),dtype=np.int8)
    grid[r:r+n,c:c+2] = values
    legend = (r,c,n,values)
    output = _compose(bg,cues,fills,walls,supports,legend)
    parsed = _parse(grid)
    if any(not np.array_equal(a,b) for a,b in zip(parsed["walls"],walls)):
        raise ValueError("constructed walls not uniquely recoverable")
    if not np.array_equal(output,_predict(parsed)):
        raise ValueError("construction/solver disagreement")
    return {"input":grid,"output":output},parsed


def _fallback(rng, teach, shift, test):
    """Readable same-rule geometry, used only after finite spatial retries."""
    shape = (29,29)
    if test and shift == "components":
        a = _outline(shape,[(2,1),(2,10),(12,10),(12,1)])
        b = _outline(shape,[(16,1),(16,10),(27,10),(27,1)])
        b[20:23,1] = False
        first = a | b
        second = _outline(shape,[(5,6),(5,23),(24,23),(24,6)])
        walls = [first,second]
    else:
        first = _outline(shape,[(2,3),(2,21),(11,21),(11,14),(24,14),(24,3)])
        second = _outline(shape,[(6,7),(6,25),(26,25),(26,7)])
        if teach == 0:
            second[26,15:18] = False
        walls = [first,second]
        if test and shift == "layers":
            walls.append(_outline(shape,[(15,17),(15,23),(23,23),(23,17)]))
    if test and shift == "damage":
        walls[0] = _outline(shape,[(2,3),(2,21),(24,21),(24,3)])
    # Reflections preserve legend reading direction because only scene masks move.
    if rng.integers(2):
        walls = [x[::-1].copy() for x in walls]
    if rng.integers(2):
        walls = [x[:,::-1].copy() for x in walls]
    raw = _corrupt(rng,walls,3 if test and shift == "damage" else 1)
    return _make_colored(rng,walls,raw,teach,fallback=True)


def _key(grid):
    a = np.asarray(grid,dtype=np.int8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("counts must be integers in 2..10 / 1..4 with train > test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["layers","components","damage"]))
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",int(num_train)),("test",int(num_test))):
        for index in range(count):
            test = split == "test"
            teach = index if split == "train" and index < 2 else -1
            required = ({"repair_area","leak","lookup"} if teach == 0 else
                        {"bbox","priority","opaque","covered_wall","repair_area"} if teach == 1 else {"repair_area"})
            n = 2 if teach >= 0 else (3 if test and shift == "layers" else
                                     2 if test and shift == "components" else int(rng.integers(1,3)))
            for attempt in range(128):
                try:
                    if attempt < 112:
                        walls = _geometry(rng,n,teach,shift,test)
                        raw = _corrupt(rng,walls,3 if test and shift == "damage" else 1)
                        pair,parsed = _make_colored(rng,walls,raw,teach)
                    else:
                        pair,parsed = _fallback(rng,teach,shift,test)
                    if not required <= _witnesses(parsed) or _key(pair["input"]) in seen:
                        continue
                    if not test and shift == "damage" and max(map(len,parsed["evidence"])) > 2:
                        continue
                    if test and shift == "damage" and max(map(len,parsed["evidence"])) < 3:
                        continue
                    if test and shift == "components" and not any(len(_components(w)) >= 2 for w in parsed["walls"]):
                        continue
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded geometry attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    n,t = len(puzzle["train"]),len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen,teaching = set(),set()
    features = {"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            key = _key(inp)
            assert key not in seen,"duplicate input"
            seen.add(key)
            p = _parse(inp)
            assert np.array_equal(out,_predict(p)),"incomplete or incorrect label"
            assert any(e for e in p["evidence"])
            assert any(x.any() for x in p["removed"])
            for wall,support in zip(p["walls"],p["supports"]):
                assert np.array_equal(wall,_strong(wall))
                assert not (wall[:-1,:-1] & wall[1:,:-1] & wall[:-1,1:] & wall[1:,1:]).any()
                assert np.array_equal(support,wall | _interior(wall))
            witness = _witnesses(p)
            assert "repair_area" in witness,"repairs must affect visible enclosed paint"
            features[split].append((len(p["cues"]),max(len(_components(w)) for w in p["walls"]),
                                    max(map(len,p["evidence"]))))
            if split == "train":
                teaching |= witness
    assert {"repair_area","leak","lookup","bbox","priority","opaque","covered_wall","retained_wall"} <= teaching, \
        "training does not distinguish required alternatives"
    # A visible compositional shift, checked without the sampled regime label.
    assert any(min(f[i] for f in features["test"]) > max(f[i] for f in features["train"])
               for i in range(3)), "test does not extend layer, component, or repair count"


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ("train","test"):
        for p in puzzle[split]:
            for role in ("input","output"):
                a = np.asarray(p[role],dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded sweep; one puzzle in memory at a time, no artifacts or logs."""
    signatures,shapes,palettes,structures = set(),set(),set(),set()
    repairs,components,layers,legend_positions = set(),set(),set(),set()
    free_lengths,axes,spurs = set(),set(),set()
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train","test"):
            for pair in puzzle[split]:
                total_pairs += 1
                p = _parse(pair["input"])
                shapes.add(pair["input"].shape)
                palettes.add(tuple([p["bg"]]+p["cues"]+p["fills"]))
                layers.add(len(p["cues"]))
                legend_positions.add(p["legend"][:2])
                for wall,evidence,removed in zip(p["walls"],p["evidence"],p["removed"]):
                    structures.add((wall.shape,wall.tobytes()))
                    components.add(len(_components(wall)))
                    repairs.add(len(evidence))
                    free_lengths.update(len(c) for c in _components(removed,True))
                    for gap,spur in evidence:
                        axes.add("h" if gap[0][0] == gap[1][0] else "v")
                        spurs.add((spur[1][0]-spur[0][0],spur[1][1]-spur[0][1]))
                assert np.array_equal(solve(pair["input"]),pair["output"])
        if seed == 0:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
            # Geometrically plausible wrong labels: miss the sealing repairs,
            # reverse the layers, or make background paint transparent.
            for mode in ("repair", "priority", "transparent"):
                bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in puzzle[s]] for s in puzzle}
                index = 0 if mode == "repair" else 1
                target = bad["train"][index]
                p = _parse(target["input"])
                ws,cs,fs = p["walls"],p["cues"],p["fills"]
                if mode == "repair":
                    ws = [_strong(raw) for raw in p["raw"]]
                elif mode == "priority":
                    ws,cs,fs = ws[::-1],cs[::-1],fs[::-1]
                wrong = _compose(p["bg"],cs,fs,ws,_layers(ws),p["legend"],mode == "transparent")
                assert not np.array_equal(wrong,target["output"])
                target["output"] = wrong
                try:
                    validate_generated_puzzle(bad)
                except (AssertionError,ValueError):
                    pass
                else:
                    raise AssertionError("plausible wrong label accepted")
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(palettes) >= 150 and len(structures) >= 250
    assert layers == {1,2,3} and max(components) >= 2 and max(repairs) >= 3
    assert free_lengths == {2,3} and axes == {"h","v"}
    assert spurs == {(-1,-1),(-1,1),(1,-1),(1,1)}
    assert len(legend_positions) >= 60
    # Border behavior and a same-cue divider are topological, not bbox rules.
    wall = np.zeros((12,12),bool)
    wall[0,:] = wall[-1,:] = wall[:,0] = wall[:,-1] = True
    assert np.count_nonzero(_interior(wall)) == 100
    wall[0,4:7] = False
    assert not _interior(wall).any()
    wall[:,6] = True
    assert np.count_nonzero(_interior(wall)) == 40
    # Exercise the finite-retry geometry directly, even if the sweep did not
    # need it. Fallbacks obey the same teaching and visible-shift contracts.
    for i,shift in enumerate(("layers","components","damage")):
        rng = np.random.default_rng(714+i)
        a,_ = _fallback(rng,0,shift,False)
        b,_ = _fallback(rng,1,shift,False)
        c,_ = _fallback(rng,-1,shift,True)
        validate_generated_puzzle({"train":[a,b],"test":[c]})
    for a,b in ((2,1),(10,1),(10,4)):
        p = generate_puzzle(128+a,a,b)
        assert len(p["train"]) == a and len(p["test"]) == b
    for a,b in ((1,1),(11,1),(4,0),(5,5),(2,2),(3,4),(True,1),(3.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds":42,"pairs":total_pairs,"distinct_puzzles":len(signatures),
            "shapes":len(shapes),"palettes":len(palettes),"wall_structures":len(structures),
            "layer_counts":sorted(layers),"component_counts":sorted(components),
            "repair_counts":sorted(repairs),"legend_positions":len(legend_positions),
            "scrap_lengths":sorted(free_lengths),"repair_axes":sorted(axes),
            "negative_label_regressions":3,"fallback_puzzles":3,"independent_solver":False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(),separators=(",",":")))
