"""Counted enclosures: sparse translated contours control a boundary route.

Only visible input geometry determines roles, topology, containment and routing.
The constructor paints a geometric plan; solve reconstructs it from the pixels.
"""
from functools import lru_cache
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _sign(x):
    return (x > 0) - (x < 0)


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr or dc) and (not dr or not dc or abs(dr) == abs(dc)), "unaligned segment")
    return [(a[0] + j * _sign(dr), a[1] + j * _sign(dc))
            for j in range(max(abs(dr), abs(dc)) + 1)]


def _stroke(vertices):
    return set().union(*(set(_segment(a, b))
                         for a, b in zip(vertices, vertices[1:] + vertices[:1])))


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


@lru_cache(maxsize=2048)
def _contour(points):
    """Unique simple full cycle, with no preference based on a sampled template."""
    n = len(points)
    _require(4 <= n <= 10, "contours have 4..10 landmarks")
    vertices = set(points)
    edges = []
    for i, a in enumerate(points):
        for j in range(i + 1, n):
            b = points[j]
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            if (dr == 0 or dc == 0 or dr == dc) and not vertices.intersection(_segment(a, b)[1:-1]):
                edges.append((i, j))
    neighbors = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        neighbors[i].append((j, k))
        neighbors[j].append((i, k))
    _require(all(len(x) >= 2 for x in neighbors), "landmarks do not form a contour")
    conflicts = [set() for _ in edges]
    for k, (i, j) in enumerate(edges):
        for m, (s, t) in enumerate(edges[:k]):
            if not {i, j}.intersection((s, t)) and _intersects(points[i], points[j], points[s], points[t]):
                conflicts[k].add(m)
                conflicts[m].add(k)
    solutions, budget = {}, [0]

    def visit(order, used, chosen):
        budget[0] += 1
        _require(budget[0] <= 50000, "topology search limit")
        if len(solutions) > 1:
            return
        if len(order) == n:
            closing = [k for j, k in neighbors[order[-1]] if j == 0]
            if closing and not conflicts[closing[0]].intersection(chosen):
                signature = tuple(sorted(chosen + [closing[0]]))
                solutions[signature] = tuple(points[j] for j in order)
            return
        for j, k in neighbors[order[-1]]:
            if j not in used and not conflicts[k].intersection(chosen):
                visit(order + [j], used | {j}, chosen + [k])

    visit([0], {0}, [])
    _require(len(solutions) == 1, "ambiguous or missing closed contour")
    selected, order = next(iter(solutions.items()))
    turns = [_sign(_cross(order[j-1], order[j], order[(j+1) % n])) for j in range(n)]
    return order, len(edges) - len(selected), -1 in turns and 1 in turns


def _inside(p, polygon):
    """Strict lattice-point containment; contour pixels are not interior."""
    if p in _stroke(list(polygon)):
        return False
    r, c = p
    result = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[0] > r) != (b[0] > r):
            crossing = a[1] + (r - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
            if c < crossing:
                result = not result
    return result


def _adjacent(points):
    ps = set(points)
    return any((r + 1, c) in ps or (r, c + 1) in ps for r, c in ps)


def _to_forward(a, direction):
    if direction == 0:
        return a.copy()
    if direction == 1:
        return a.T.copy()
    if direction == 2:
        return a[::-1].copy()
    return a.T[::-1].copy()


def _from_forward(a, direction):
    if direction in (0, 2):
        return _to_forward(a, direction)
    if direction == 1:
        return a.T.copy()
    return a[::-1].T.copy()


def _bounds(pixels):
    return (min(r for r, c in pixels), max(r for r, c in pixels),
            min(c for r, c in pixels), max(c for r, c in pixels))


def _read(input_grid):
    grid = _grid(input_grid)
    values, frequencies = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(frequencies)])
    _require(int(max(frequencies)) > grid.size // 2, "background must occupy a majority")
    groups = {int(c): [tuple(map(int, p)) for p in np.argwhere(grid == c)]
              for c in values if c != bg}
    h, w = grid.shape
    boundary_singletons = [(c, ps[0]) for c, ps in groups.items() if len(ps) == 1
                           and (ps[0][0] in (0, h-1) or ps[0][1] in (0, w-1))]
    _require(len(boundary_singletons) == 1, "one boundary seed is required")
    route, (r, c) = boundary_singletons[0]
    boundary = [r == 0, c == 0, r == h-1, c == w-1]
    _require(sum(boundary) == 1, "seed cannot be at a corner")
    direction = boundary.index(True)
    a = _to_forward(grid, direction)
    groups = {col: [tuple(map(int, p)) for p in np.argwhere(a == col)] for col in groups}
    dot_colors = [col for col, ps in groups.items() if _adjacent(ps)]
    _require(len(dot_colors) == 1, "counters are the unique color with an orthogonally adjacent pair")
    dots = dot_colors[0]
    source_colors = [col for col, ps in groups.items() if col not in (route, dots) and len(ps) > 1]
    _require(1 <= len(source_colors) <= 3, "one to three sources required")
    marker_colors = [col for col, ps in groups.items() if col != route and len(ps) == 1]
    _require(len(marker_colors) == len(source_colors), "one companion per source")
    sources = []
    for col in source_colors:
        vertices = groups[col]
        _require(not _adjacent(vertices), "source vertices must have no orthogonal adjacency")
        order, chords, concave = _contour(tuple(sorted(vertices)))
        stroke = _stroke(list(order))
        sources.append({"color": col, "vertices": vertices, "order": order,
                        "stroke": stroke, "chords": chords, "concave": concave})
    used = set()
    for marker_color in marker_colors:
        marker = groups[marker_color][0]
        associations = [(i, v) for i, s in enumerate(sources) for v in s["vertices"]
                        if max(abs(v[0]-marker[0]), abs(v[1]-marker[1])) == 1]
        _require(len(associations) == 1, "companion anchor must be unique")
        i, v = associations[0]
        _require(i not in used, "multiple companions for one source")
        used.add(i)
        df, dl = marker[0] - v[0], marker[1] - v[1]
        _require(dl in (-1, 1) and df in (-1, 0, 1), "companion must have a lateral displacement")
        s = sources[i]
        copy = {(r + df, c + dl) for r, c in s["stroke"]}
        _require(marker not in s["stroke"], "marker occluded by its source")
        _require(all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] for r, c in copy), "copy is clipped")
        s.update(marker=marker, companion=marker_color, offset=(df, dl), copy=copy,
                 box=_bounds(s["stroke"] | copy))
        inside = {p for p in groups[dots] if _inside(p, s["order"])}
        _require(1 <= len(inside) <= 3, "each source encloses 1..3 counters")
        s["inside"] = inside
        s["count"] = len(inside)
        translated_order = tuple((r+df,c+dl) for r,c in s["order"])
        s["copy_inside"] = {p for p in groups[dots] if _inside(p,translated_order)}
        r0, r1, c0, c1 = _bounds(s["stroke"])
        s["box_decoys"] = sum(r0 < r < r1 and c0 < c < c1 and p not in inside
                              for p in groups[dots] for r, c in [p])
        _require(not set(groups[dots]).intersection(s["stroke"] | copy), "counter lies on an outline")
    sources.sort(key=lambda s: s["box"][0])
    occupied = set()
    previous_end = -1
    for s in sources:
        r0, r1, _, _ = s["box"]
        _require(0 < r0 <= r1 < a.shape[0]-1 and r0 > previous_end, "overlapping forward bands")
        _require(not occupied.intersection(s["stroke"] | s["copy"]), "different obstacles overlap")
        occupied.update(s["stroke"] | s["copy"])
        previous_end = r1
    path = set()
    initial = groups[route][0][1]
    current, start, previous_end = initial, 0, -1
    for s in sources:
        r0, r1, c0, c1 = s["box"]
        d = s["count"] + 1
        turn = r0 - d
        sign = -s["offset"][1]
        target = c1 + d if sign == 1 else c0 - d
        _require(1 <= turn and turn > previous_end and turn >= start, "turn too early")
        _require(c0 < current < c1, "obstacle must intercept the incoming ray")
        _require(0 <= target < a.shape[1], "bypass outside canvas")
        path.update((r, current) for r in range(start, turn + 1))
        path.update((turn, c) for c in range(min(current, target), max(current, target) + 1))
        s.update(turn=turn, target=target, side=sign, incoming=current)
        current, start, previous_end = target, turn, r1
    path.update((r, current) for r in range(start, a.shape[0]))
    _require(not path.intersection(occupied | set(groups[dots])), "route collides with an object or counter")
    active = set().union(*(s["inside"] for s in sources))
    exterior = set(groups[dots]) - active
    _require(bool(exterior), "at least one exterior counter is required")
    return a, {"background": bg, "route": route, "dots": dots, "direction": direction,
               "sources": sources, "path": path, "active": active, "exterior": exterior}


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, info = _read(input_grid)
    out = a.copy()
    for p in info["active"]:
        out[p] = info["background"]
    for s in info["sources"]:
        for p in s["copy"]:
            if a[p] == info["background"]:
                out[p] = s["companion"]
    for s in info["sources"]:
        for p in s["stroke"]:
            out[p] = s["color"]
    for p in info["path"]:
        out[p] = info["route"]
    return _from_forward(out, info["direction"])


def _shape(rng, family, height_limit=10):
    """Variable contours, accepted only when the landmarks recover the plan."""
    for _ in range(160):
        h = int(rng.integers(4, height_limit + 1))
        w = int(rng.integers(5, 13))
        if family == "notch":
            if height_limit < 8:
                continue
            h = int(rng.integers(8, height_limit + 1))
            w = int(rng.integers(7, 13))
            depth = int(rng.integers(3, w-2))
            top, bottom = 2, h-2
            ps = [(0,0),(0,w),(top,w),(top,w-depth),(bottom,w-depth),
                  (bottom,w),(h,w),(h,0)]
        elif family == "diamond":
            if height_limit < 6:
                continue
            q = int(rng.integers(3, height_limit // 2 + 1))
            ps = [(0,q),(q,2*q),(2*q,q),(q,0)]
        elif family == "clipped":
            cut = int(rng.integers(1, min(h,w)//2 + 1))
            ps = [(0,0),(0,w-cut),(cut,w),(h,w),(h,cut),(h-cut,0)]
        elif family == "step":
            cut = int(rng.integers(2, w-1))
            mid = h//2
            ps = [(0,0),(0,w-cut),(mid,w-cut),(mid,w),(h,w),(h,0)]
        else:
            if family == "square":
                w = h
            ps = [(0,0),(0,w),(h,w),(h,0)]
        if rng.random() < .5:
            ps = [(r, max(c for r,c in ps)-c) for r,c in ps]
        if rng.random() < .5:
            ps = [(max(r for r,c in ps)-r, c) for r,c in ps]
        if _adjacent(ps):
            continue
        try:
            order, _, concave = _contour(tuple(sorted(ps)))
            if _stroke(ps) != _stroke(list(order)) or (family in ("notch", "step") and not concave):
                continue
            return ps
        except ValueError:
            continue
    if family == "notch":
        raise ValueError("could not sample a unique notched contour")
    return [(0,0),(0,7),(4,7),(4,0)]


def _plan(rng, counts, sides, notch=False, diagonal=False):
    k = len(counts)
    shapes, offsets = [], []
    # Reserve enough forward extent for all later objects before sampling height.
    remaining = 28 - sum(n+1 for n in counts) - k
    for j in range(k):
        limit = min(10, remaining - 4*(k-j-1))
        families = ["rectangle", "square", "clipped", "step"]
        if limit >= 6:
            families += ["diamond"]
        if limit >= 8:
            families += ["notch"]
        family = ("notch" if notch and j == 0 else
                  "clipped" if diagonal and j == 0 else str(rng.choice(families)))
        df = -1 if notch and j == 0 else int(rng.integers(-1,2))
        if diagonal and j == 0:
            df = int(rng.choice((-1,1)))
        # A 3-object scene can use zero forward offset when the budget is tight.
        if limit <= 4:
            df = 0
        ps = _shape(rng, family, max(4, limit-abs(df)))
        span = max(r for r,c in ps) + abs(df)
        remaining -= span
        shapes.append(ps)
        offsets.append((df, sides[j]))
    minimum = sum(n+1 + max(r for r,c in ps) + abs(df)
                  for n, ps, (df, dl) in zip(counts, shapes, offsets)) + k + 2
    _require(minimum <= 30, "plan exceeds canvas")
    height = int(rng.integers(minimum, 31))
    slack = rng.multinomial(height-minimum, np.full(k+1, 1/(k+1)))
    previous_end, specifications = -1, []
    for j, (ps, n, offset) in enumerate(zip(shapes, counts, offsets)):
        turn = (1 if j == 0 else previous_end+1) + int(slack[j])
        begin = turn + n+1
        shift = begin - min(0, offset[0])
        moved = [(r+shift,c) for r,c in ps]
        stroke = _stroke(moved)
        copy = {(r+offset[0],c+offset[1]) for r,c in stroke}
        previous_end = max(r for r,c in stroke | copy)
        specifications.append({"vertices": moved, "stroke": stroke, "copy": copy,
                               "offset": offset, "count": n, "turn": turn,
                               "width": max(c for r,c in ps)})
    # Search a finite lateral placement space. No assumed alternation.
    width = int(rng.integers(8 if k == 1 else 16, 31))
    widths = list(dict.fromkeys((width, min(30,width+5), 30)))
    for width in widths:
        failed = set()

        def search(index, current):
            if index == k:
                return []
            if (index,current) in failed:
                return None
            s = specifications[index]
            df, dl = s["offset"]
            d, span = s["count"]+1, s["width"]
            choices = []
            for lo in range(0, width-span):
                hi = lo+span
                target = lo-d if dl == 1 else hi+d
                if (lo+min(0,dl) >= 0 and hi+max(0,dl) < width
                        and lo < current < hi and 0 <= target < width):
                    choices.append((lo,target))
            rng.shuffle(choices)
            for lo,target in choices:
                rest = search(index+1,target)
                if rest is not None:
                    return [(lo,target)] + rest
            failed.add((index,current))
            return None

        for initial in rng.permutation(np.arange(1,width-1)):
            placements = search(0,int(initial))
            if placements is None:
                continue
            objects = []
            for s,(lo,target) in zip(specifications,placements):
                obj = dict(s)
                for key in ("vertices", "stroke", "copy"):
                    cells = [(r,c+lo) for r,c in s[key]]
                    obj[key] = cells if key == "vertices" else set(cells)
                obj["target"] = target
                objects.append(obj)
            return (height,width), int(initial), objects
    raise ValueError("no feasible lateral placement")


def _build_pair(rng, counts, sides, direction, notch=False, diagonal=False):
    for _ in range(160):
        try:
            shape, initial, objects = _plan(rng, counts, sides, notch, diagonal)
            palette = list(map(int,rng.permutation(10)))
            bg, route, dots = palette[:3]
            a = np.full(shape,bg,dtype=np.int8)
            a[0,initial] = route
            b = a.copy()
            path = set()
            previous, current = 0, initial
            for i,s in enumerate(objects):
                col, companion = palette[3+2*i:5+2*i]
                s.update(color=col, companion=companion)
                for p in s["vertices"]:
                    a[p] = col
                df,dl = s["offset"]
                candidates = [(r+df,c+dl) for r,c in s["vertices"]
                              if (r+df,c+dl) not in s["stroke"]
                              and sum(max(abs(r+df-v[0]),abs(c+dl-v[1])) == 1
                                      for v in s["vertices"]) == 1
                              and 0 < c+dl < shape[1]-1]
                _require(bool(candidates), "no unique interior companion marker")
                marker = candidates[int(rng.integers(len(candidates)))]
                a[marker] = companion
                # Constructor paints planned bends using slices, unlike solve's point trace.
                turn,target = s["turn"],s["target"]
                b[previous:turn+1,current] = route
                b[turn,min(current,target):max(current,target)+1] = route
                path.update((r,current) for r in range(previous,turn+1))
                path.update((turn,c) for c in range(min(current,target),max(current,target)+1))
                current,previous = target,turn
            b[previous:,current] = route
            path.update((r,current) for r in range(previous,shape[0]))
            complete = set().union(*(s["stroke"] | s["copy"] for s in objects))
            counters = []
            for object_index,s in enumerate(objects):
                r0,r1,c0,c1 = _bounds(s["stroke"])
                candidates = [(r,c) for r in range(r0+1,r1) for c in range(c0+1,c1)
                              if (r,c) not in complete and _inside((r,c),s["vertices"])]
                _require(len(candidates) >= s["count"], "insufficient counter space")
                # Connected unary clusters coexist with scattered counters.
                rng.shuffle(candidates)
                if diagonal and object_index == 0:
                    df,dl = s["offset"]
                    shifted = [(r+df,c+dl) for r,c in s["vertices"]]
                    preferred = [p for p in candidates if not _inside(p,shifted)]
                    _require(bool(preferred), "source-only counter scope not visible")
                    chosen = [preferred[int(rng.integers(len(preferred)))]]
                elif notch:
                    candidate_set = set(candidates)
                    pairs = [(p,(p[0]+dr,p[1]+dc)) for p in candidates
                             for dr,dc in ((0,1),(1,0))
                             if (p[0]+dr,p[1]+dc) in candidate_set]
                    _require(bool(pairs), "no connected active counter pair")
                    chosen = list(pairs[int(rng.integers(len(pairs)))])
                else:
                    chosen = [candidates[0]]
                while len(chosen) < s["count"]:
                    nearby = [p for p in candidates if p not in chosen
                              and any(abs(p[0]-q[0])+abs(p[1]-q[1]) == 1 for q in chosen)]
                    pool = nearby if nearby and rng.random() < .8 else [p for p in candidates if p not in chosen]
                    chosen.append(pool[int(rng.integers(len(pool)))])
                counters.extend(chosen)
            exterior = [(r,c) for r in range(1,shape[0]-1) for c in range(shape[1])
                        if (r,c) not in complete | path and a[r,c] == bg
                        and not any(_inside((r,c),s["vertices"]) for s in objects)]
            if notch:
                s = objects[0]
                r0,r1,c0,c1 = _bounds(s["stroke"])
                cavity = [p for p in exterior if r0 < p[0] < r1 and c0 < p[1] < c1]
                _require(bool(cavity), "no notch counter")
                counters.append(cavity[int(rng.integers(len(cavity)))])
            # The domino identifies the counter role even when all active counts are one.
            exterior_set = set(exterior)
            dominoes = [(p,(p[0]+dr,p[1]+dc)) for p in exterior
                        for dr,dc in ((0,1),(1,0)) if (p[0]+dr,p[1]+dc) in exterior_set]
            _require(bool(dominoes), "no exterior counter domino")
            pair = dominoes[int(rng.integers(len(dominoes)))]
            counters.extend(pair)
            spare = [p for p in exterior if p not in counters]
            rng.shuffle(spare)
            counters.extend(spare[:int(rng.integers(0,3))])
            for p in set(counters):
                a[p] = dots
            # Expected drawing uses the proposed geometry, not solve(input).
            nonroute = b != route
            b[nonroute] = a[nonroute]
            for s in objects:
                for p in counters:
                    if _inside(p,s["vertices"]):
                        b[p] = bg
                for p in s["copy"]:
                    if a[p] == bg:
                        b[p] = s["companion"]
            for s in objects:
                for p in s["stroke"]:
                    b[p] = s["color"]
            inp,out = _from_forward(a,direction),_from_forward(b,direction)
            _,info = _read(inp)
            _require(all(s["count"] == n for s,n in zip(info["sources"],counts)), "plan count mismatch")
            _require(np.array_equal(solve(inp),out), "plan and decoded geometry differ")
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise ValueError("bounded pair construction exhausted")


def _count(name,value,lo,hi):
    _require(not isinstance(value,(bool,np.bool_)) and isinstance(value,(int,np.integer))
             and lo <= int(value) <= hi, f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(grid):
    a = np.asarray(grid,dtype=np.int8)
    return a.shape,a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed,(bool,np.bool_)) and isinstance(seed,(int,np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    nt, nq = _count("num_train",num_train,2,10), _count("num_test",num_test,1,4)
    _require(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    directions = list(map(int,rng.permutation(4)))
    witness_side = int(rng.choice((-1,1)))
    requirements = [([2],[witness_side],True,False),
                    ([1,3],[-witness_side,-witness_side],False,True)]
    for _ in range(nt-2):
        k = int(rng.integers(1,4))
        requirements.append((list(map(int,rng.integers(1,4,size=k))),
                             list(map(int,rng.choice((-1,1),size=k))),False,False))
    rng.shuffle(requirements)
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",nt),("test",nq)):
        for j in range(count):
            if split == "train":
                ns,sides,notch,diagonal = requirements[j]
                direction = directions[int(rng.integers(3))]
            else:
                k = int(rng.integers(1,4))
                ns = list(map(int,rng.integers(1,4,size=k)))
                sides = list(map(int,rng.choice((-1,1),size=k)))
                notch,diagonal,direction = False,False,directions[3]
            for _ in range(30):
                pair = _build_pair(rng,ns,sides,direction,notch,diagonal)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {"train","test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "splits must be lists")
    nt,nq = len(puzzle["train"]),len(puzzle["test"])
    _count("num_train",nt,2,10)
    _count("num_test",nq,1,4)
    _require(nt > nq, "illegal split counts")
    seen,features = set(),{"train":[],"test":[]}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {"input","output"}, "invalid pair keys")
            inp,out = _grid(pair["input"]),_grid(pair["output"])
            _require(inp.shape == out.shape and _key(inp) not in seen, "duplicate input or shape mismatch")
            seen.add(_key(inp))
            a,info = _read(inp)
            b = _to_forward(out,info["direction"])
            _require(np.array_equal(solve(inp),out), "incorrect contour, count, or route")
            _require(all(b[p] == info["background"] for p in info["active"]), "active counters not consumed")
            _require(all(b[p] == info["dots"] for p in info["exterior"]), "exterior counters changed")
            _require(len(info["path"]) > 1, "B must add a nontrivial route")
            for s in info["sources"]:
                _require(any(a[p] == info["background"] for p in s["stroke"]), "A must complete missing strokes")
                _require(len(s["copy"] - s["stroke"] - {s["marker"]}) >= 2, "companion must visibly grow")
                _require(all(b[p] == s["color"] for p in s["stroke"]), "source loses overlap priority")
            features[split].append(info)
    training = features["train"]
    _require({s["count"] for i in training for s in i["sources"]} == {1,2,3}, "counts not demonstrated")
    _require({s["side"] for i in training for s in i["sources"]} == {-1,1}, "both bypass sides not demonstrated")
    _require(any(s["concave"] and s["box_decoys"] and s["chords"] for i in training for s in i["sources"]),
             "missing containment and chord rejection witness")
    _require(any(s["count"] == 2 and _adjacent(s["inside"]) for i in training for s in i["sources"]),
             "cell counting versus component counting not demonstrated")
    _require(any(s["inside"] != s["copy_inside"] for i in training for s in i["sources"]),
             "source versus companion counter scope not demonstrated")
    _require(any(any(a["side"] == b["side"] for a,b in zip(i["sources"],i["sources"][1:]))
                 for i in training), "repeated-side routing not demonstrated")
    _require(any(s["offset"][0] == -1 for i in training for s in i["sources"]), "forward extent of copy not demonstrated")
    _require(any(a[0] != b[0] and a[1] != b[1] for i in training for s in i["sources"]
                 for a,b in zip(s["order"],s["order"][1:] + s["order"][:1])),
             "diagonal completion not demonstrated")
    _require(any(s["copy"] & s["stroke"] - set(s["vertices"])
                 for i in training for s in i["sources"]), "source priority not demonstrated")
    _require({i["direction"] for i in features["test"]}.isdisjoint({i["direction"] for i in training}),
             "test direction is not held out")


def self_test():
    _literal_regression()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(all(_key(x[k]) == _key(y[k]) for split in puzzles[7]
                 for x,y in zip(puzzles[7][split],generate_puzzle(7)[split]) for k in ("input","output")),
             "nondeterminism")
    shapes,structures,directions,offsets,counts,vertices,transitions = set(),set(),set(),set(),set(),set(),set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in puzzle:
            for pair in puzzle[split]:
                a,info = _read(pair["input"])
                shapes.add(tuple(pair["input"].shape))
                roles = {}
                structures.add((a.shape,tuple(roles.setdefault(int(c),len(roles)) for c in a.flat)))
                directions.add(info["direction"])
                for s in info["sources"]:
                    offsets.add(s["offset"])
                    counts.add(s["count"])
                    vertices.add(len(s["vertices"]))
                transitions.update((x["side"],y["side"]) for x,y in zip(info["sources"],info["sources"][1:]))
    _require(len(shapes) >= 45 and len(structures) >= 180, "insufficient structural variation")
    _require(directions == set(range(4)) and len(offsets) == 6, "orientation variation missing")
    _require(counts == {1,2,3} and len(vertices) >= 3, "count or contour variation missing")
    _require(transitions == {(-1,-1),(-1,1),(1,-1),(1,1)}, "side transitions missing")
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            validate_generated_puzzle(generate_puzzle(100+nt*10+nq,nt,nq))
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Color and physical orientation equivariance, including nested-list input.
    palette = np.array([3,8,1,6,0,9,4,7,5,2],dtype=np.int8)
    pair = puzzles[0]["train"][0]
    for rotation in range(4):
        for reflection in (False,True):
            a,b = np.rot90(pair["input"],rotation),np.rot90(pair["output"],rotation)
            if reflection:
                a,b = np.fliplr(a),np.fliplr(b)
            _require(np.array_equal(solve(palette[a].tolist()),palette[b]), "equivariance failure")
    # Validator rejects a removed route cell and a changed preserved exterior counter.
    for feature in ("path","exterior"):
        bad = {s:[{k:v.copy() for k,v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        pair = bad["test"][0]
        a,info = _read(pair["input"])
        out = _to_forward(pair["output"],info["direction"])
        candidates = info[feature] - {(0,int(np.argwhere(a == info["route"])[0,1]))}
        p = sorted(candidates)[0]
        out[p] = info["background"]
        pair["output"] = _from_forward(out,info["direction"])
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError("damaged answer accepted")
    return {"seeds":42,"pairs":210,"shapes":len(shapes),"structures":len(structures),
            "directions":len(directions),"relative_offsets":len(offsets),
            "vertex_counts":sorted(vertices),"counter_counts":sorted(counts),
            "side_transitions":sorted(transitions)}


def _literal_regression():
    """Hand-specified bends/priority, rather than output obtained from solve."""
    a = np.zeros((18,18),dtype=np.int8)
    a[0,8] = 1
    a[8,5] = a[8,11] = a[12,5] = a[12,11] = 3
    a[7,4] = 4
    a[9,7] = a[10,7] = a[13,7] = a[14,7] = 2
    expected = np.zeros_like(a)
    expected[13,7] = expected[14,7] = 2
    expected[7,4:11] = expected[11,4:11] = 4
    expected[7:12,4] = expected[7:12,10] = 4
    expected[8,5:12] = expected[12,5:12] = 3
    expected[8:13,5] = expected[8:13,11] = 3
    expected[:5,8] = expected[4,8:15] = expected[4:,14] = 1
    _require(np.array_equal(solve(a),expected), "literal contour/route regression failed")
    # Omitting copying moves the turn from row 4 to row 5. Counting exterior
    # counters moves it to row 2 and the bypass from column 14 to column 16.
    _,info = _read(a)
    s = info["sources"][0]
    _require(s["count"] == 2 and s["turn"] == 4 and s["target"] == 14,
             "literal decoded features disagree")
    _require(_bounds(s["stroke"])[0] - 3 == 5, "copy must affect approach geometry")
    all_counter_turn = s["box"][0] - (int(np.count_nonzero(a == 2)) + 1)
    _require(all_counter_turn == 2 and all_counter_turn != s["turn"], "containment ablation ineffective")
