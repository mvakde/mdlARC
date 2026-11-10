"""Color-passport beams: local color containment controls an emitting network.

Only standard-library modules and NumPy are used. The solver parses its input,
then resolves spatial encounters and obstacles; construction metadata is never
an argument to the solver. Public validation measures returned grids.
"""
from __future__ import annotations

import math
from itertools import combinations
import numpy as np


PORTS = {"NW": (-1, -1), "N0": (-1, 0), "N1": (-1, 1),
         "NE": (-1, 2), "W0": (0, -1), "W1": (1, -1),
         "E0": (0, 2), "E1": (1, 2), "SW": (2, -1),
         "S0": (2, 0), "S1": (2, 1), "SE": (2, 2)}


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _direction(offset):
    return tuple(-1 if n == -1 else 1 if n == 2 else 0 for n in offset)


def _box_cells(box):
    r0, r1, c0, c1 = box
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _apart(a, b):
    # Half-open bounds: require at least one empty row or column between boxes.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _parse(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _need(counts[bg] * 2 > grid.size, "background must occupy more than half the input")
    h, w = grid.shape
    launchers, covered, footprints = [], set(), []
    for r in range(1, h - 2):
        for c in range(1, w - 2):
            color = int(grid[r, c])
            if color == bg or not np.all(grid[r:r+2, c:c+2] == color):
                continue
            ring = [(r+dr, c+dc, int(grid[r+dr, c+dc]), (dr, dc))
                    for dr, dc in PORTS.values()]
            values = {v for _, _, v, _ in ring if v != bg}
            holes = sum(v == bg for _, _, v, _ in ring)
            if len(values) != 1 or color in values or not 1 <= holes <= 4:
                continue
            structure = next(iter(values))
            box = (r-1, r+3, c-1, c+3)
            _need(all(_apart(box, b) for b in footprints), "touching emitter footprints")
            footprints.append(box)
            cells = _box_cells(box)
            covered |= {p for p in cells if int(grid[p]) != bg}
            launchers.append({"core": (r, c), "box": box, "color": color,
                              "structure": structure,
                              "ports": [((rr, cc), _direction(off))
                                        for rr, cc, v, off in ring if v == bg]})
    _need(launchers, "no emitters")
    structures = {l["structure"] for l in launchers}
    _need(len(structures) == 1, "rings must share a structural color")
    structure = next(iter(structures))
    remaining = set(zip(*np.nonzero(grid != bg))) - covered
    records = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells = [start]
        for r, c in cells:
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    cells.append(point)
        rows, cols = zip(*cells)
        box = (min(rows), max(rows)+1, min(cols), max(cols)+1)
        r0, r1, c0, c1 = box
        _need(r1-r0 >= 3 and c1-c0 >= 3, "record too thin")
        _need(len(cells) == (r1-r0)*(c1-c0), "record is not solid")
        border = np.concatenate((grid[r0, c0:c1], grid[r1-1, c0:c1],
                                 grid[r0:r1, c0], grid[r0:r1, c1-1]))
        filler = int(border[0])
        _need(np.all(border == filler), "record border is not uniform")
        _need(filler not in (bg, structure), "record filler conflicts with a role")
        _need(all(_apart(box, b) for b in footprints), "touching objects")
        footprints.append(box)
        vals, ns = np.unique(grid[r0:r1, c0:c1], return_counts=True)
        cmap = {int(v): int(n) for v, n in zip(vals, ns)}
        labels = set(cmap) - {filler}
        _need(not labels & {bg, structure}, "invalid label color")
        records.append({"box": box, "filler": filler, "labels": frozenset(labels),
                        "counts": cmap, "marks": sum(cmap[v] for v in labels)})
    if records:
        _need(len({g["filler"] for g in records}) == 1, "different record fillers")
        _need(all(l["color"] != records[0]["filler"] for l in launchers),
              "payload cannot be filler")
    return {"background": bg, "structure": structure, "launchers": launchers,
            "records": records, "footprints": footprints}


def _walk(origin, velocity, shape):
    major = max(map(abs, velocity))
    points = []
    for k in range(3 * max(shape) + 1):
        p = tuple(origin[j] + (1 if velocity[j] >= 0 else -1)
                  * (k * abs(velocity[j]) // major) for j in range(2))
        if not _inside(p, shape):
            break
        points.append(p)
    return points


def _encounter(a, b):
    hits = []
    for ai, pa in enumerate(a):
        cells = {p: k for k, p in enumerate(pa)}
        mid = {(p[0]+q[0], p[1]+q[1]): k
               for k, (p, q) in enumerate(zip(pa, pa[1:]))
               if p[0] != q[0] and p[1] != q[1]}
        for bi, pb in enumerate(b):
            for j, p in enumerate(pb):
                if p in cells:
                    hits.append((cells[p]-1, j-1, ai, bi, "cell"))
            for j, (p, q) in enumerate(zip(pb, pb[1:])):
                if p[0] == q[0] or p[1] == q[1]:
                    continue
                doubled = (p[0]+q[0], p[1]+q[1])
                if doubled in mid:
                    i = mid[doubled]
                    if {pa[i], pa[i+1]} != {p, q}:
                        hits.append((i, j, ai, bi, "between"))
    if not hits:
        return None
    return min(hits, key=lambda x: (x[0]+x[1], max(x[:2]), x[2], x[3]))


def _collision_kind(a, b):
    _need(max(map(abs, a)) == max(map(abs, b)) == 1,
          "staircase re-encounters are outside this family")
    dot = a[0]*b[0] + a[1]*b[1]
    cross = a[0]*b[1] - a[1]*b[0]
    _need(cross != 0, "parallel overlap or head-on encounter")
    if sum(map(abs, a)) == sum(map(abs, b)) == 1:
        return "perpendicular"
    if sum(map(abs, a)) == sum(map(abs, b)) == 2:
        return "opposing_diagonals"
    _need(dot > 0, "obtuse cardinal/diagonal encounter is outside this family")
    return "staircase"


def _trace(grid, info=None, *, containment=True, global_query=False):
    info = _parse(grid) if info is None else info
    bg = info["background"]
    out = grid.copy()
    records = info["records"]
    record_at = {p: i for i, g in enumerate(records) for p in _box_cells(g["box"])}
    fixed = {p for l in info["launchers"] for p in _box_cells(l["box"])
             if int(grid[p]) != bg}
    owners, groups = [], []
    for l in info["launchers"]:
        for origin, velocity in l["ports"]:
            sid = len(owners)
            owners.append(l["color"])
            groups.append({"v": velocity, "members": [(sid, origin)], "depth": 0})
    all_query = frozenset(owners)
    erased, stopped, segments, events, visits = set(), set(), [], [], []
    painted, last, diagonals = {}, {}, set()
    touches = {}

    def query(group):
        return frozenset(owners[s] for s, _ in group["members"])

    def draw(group, paths, cut=None):
        used = [p if cut is None else p[:cut+1] for p in paths]
        segments.append({"query": query(group), "v": group["v"], "paths": used,
                         "colors": tuple(owners[s] for s, _ in group["members"])})
        for (sid, _), path in zip(group["members"], used):
            for k, p in enumerate(path):
                _need(p not in fixed and (p not in record_at or record_at[p] in erased),
                      "a strand overwrites an intact object")
                _need(p not in painted or (k == 0 and painted[p] == sid and last.get(sid) == p),
                      "trail overlap or self-intersection")
                painted[p], last[sid], out[p] = sid, p, owners[sid]
            for p, q in zip(path, path[1:]):
                if p[0] != q[0] and p[1] != q[1]:
                    midpoint = (p[0]+q[0], p[1]+q[1])
                    slope = 1 if (q[0]-p[0])*(q[1]-p[1]) > 0 else -1
                    _need((midpoint, -slope) not in diagonals, "crossing an earlier trail")
                    diagonals.add((midpoint, slope))

    for iteration in range(len(records) + 3*len(owners) + 5):
        if not groups:
            break
        forecasts, barriers = [], []
        for group in groups:
            paths, blocked = [], []
            for _, origin in group["members"]:
                path = _walk(origin, group["v"], grid.shape)
                obstruction = None
                for k, p in enumerate(path):
                    if p in fixed:
                        obstruction = (k-1, -1)
                        path = path[:k]
                        break
                    if p in record_at and record_at[p] not in erased:
                        obstruction = (k-1, record_at[p])
                        path = path[:k]
                        break
                paths.append(path)
                if obstruction is not None:
                    blocked.append(obstruction)
            barrier = min(blocked) if blocked else None
            if barrier is not None:
                cut = barrier[0]
                _need(len({g for k, g in blocked if k == cut}) == 1,
                      "bundle meets different objects at once")
                _need(all(len(p) > cut for p in paths), "bundle clipped before an obstacle")
                paths = [p[:cut+1] for p in paths]
            forecasts.append(paths)
            barriers.append(barrier)
        nearest = [b[0] if b else 10000 for b in barriers]
        candidates = []
        for i, j in combinations(range(len(groups)), 2):
            hit = _encounter(forecasts[i], forecasts[j])
            if hit is not None:
                _need(min(hit[:2]) >= 0, "encounter at an existing origin")
                candidates.append((i, j, hit))
                nearest[i], nearest[j] = min(nearest[i], hit[0]), min(nearest[j], hit[1])
        ready = []
        for i, barrier in enumerate(barriers):
            if barrier is not None and barrier[0] == nearest[i]:
                _need(not any((i == a and hit[0] == nearest[i]) or
                              (i == b and hit[1] == nearest[i]) for a, b, hit in candidates),
                      "obstacle/collision tie")
                ready.append((barrier[0], 0, i, -1, barrier))
        for i, j, hit in candidates:
            if hit[0] == nearest[i] and hit[1] == nearest[j]:
                _need(not any((a, b) != (i, j) and
                              ((a in (i, j) and x[0] == nearest[a]) or
                               (b in (i, j) and x[1] == nearest[b]))
                              for a, b, x in candidates), "multiway encounter")
                ready.append((max(hit[:2]), 1, i, j, hit))
        if not ready:
            terminals = [i for i, n in enumerate(nearest) if n == 10000]
            _need(terminals, "cyclic event order")
            for i in terminals:
                draw(groups[i], forecasts[i])
            groups = [g for i, g in enumerate(groups) if i not in terminals]
            continue
        _, kind, i, j, data = min(ready)
        if kind == 0:
            cut, gid = data
            _need(gid >= 0, "ray hits an emitter")
            _need(not any(k != i and barrier is not None and barrier[1] == gid
                          and nearest[k] == barrier[0]
                          for k, barrier in enumerate(barriers)),
                  "different bundles share an intact gate")
            q = query(groups[i])
            # Each intact record has one incident bundle. This avoids an
            # unspecified restart law for previously stopped streams.
            signature = frozenset(s for s, _ in groups[i]["members"])
            _need(gid not in touches or touches[gid] == signature, "shared gate")
            touches[gid] = signature
            tested = all_query if global_query else q
            opened = tested <= records[gid]["labels"] if containment else True
            visits.append({"record": gid, "query": q, "opened": opened,
                           "labels": records[gid]["labels"], "cut": cut,
                           "bundle_size": len(groups[i]["members"])})
            if opened:
                erased.add(gid)
                r0, r1, c0, c1 = records[gid]["box"]
                out[r0:r1, c0:c1] = bg
                # Do not reset the origin/raster phase when a gate opens.
            else:
                stopped.add(gid)
                draw(groups[i], forecasts[i], cut)
                groups.pop(i)
        else:
            a, b = groups[i], groups[j]
            collision = _collision_kind(a["v"], b["v"])
            velocity = tuple(a["v"][k]+b["v"][k] for k in range(2))
            divisor = math.gcd(*map(abs, velocity))
            _need(divisor > 0, "canceling headings")
            velocity = tuple(v//divisor for v in velocity)
            members = []
            for index, cut in ((i, data[0]), (j, data[1])):
                _need(all(len(p) > cut for p in forecasts[index]), "partly clipped bundle")
                draw(groups[index], forecasts[index], cut)
                members.extend((sid, p[cut]) for (sid, _), p
                               in zip(groups[index]["members"], forecasts[index]))
            events.append({"kind": collision, "intersection": data[4],
                           "unequal": data[0] != data[1], "outgoing": velocity,
                           "query": query(a) | query(b),
                           "depth": 1+max(a["depth"], b["depth"])})
            groups = [g for k, g in enumerate(groups) if k not in (i, j)]
            groups.append({"v": velocity, "members": members,
                           "depth": 1+max(a["depth"], b["depth"])})
    else:
        raise AssertionError("event resolution did not terminate")
    return out, dict(info, events=events, visits=visits, segments=segments,
                     erased=erased, stopped=stopped)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all conventions from the input and return a fresh solved array."""
    grid = _grid(input_grid)
    return _trace(grid)[0]


def _make_emitters(shape, specs, bg=0, ring=8):
    grid = np.full(shape, bg, dtype=np.int8)
    occupied = []
    for spec in specs:
        r, c = spec["core"]
        box = (r-1, r+3, c-1, c+3)
        _need(0 <= box[0] < box[1] <= shape[0] and 0 <= box[2] < box[3] <= shape[1],
              "emitter outside canvas")
        _need(all(_apart(box, b) for b in occupied), "touching emitters")
        occupied.append(box)
        grid[r:r+2, c:c+2] = spec["color"]
        for name, (dr, dc) in PORTS.items():
            if name not in spec["ports"]:
                grid[r+dr, c+dc] = ring
    return grid


def _at_port(point, name, color):
    dr, dc = PORTS[name]
    return {"core": (point[0]-dr, point[1]-dc), "ports": (name,), "color": color}


def _primary(rng, collision):
    for _ in range(120):
        shape = (int(rng.integers(24, 31)), int(rng.integers(22, 31)))
        row = int(rng.integers(15, shape[0]-5))
        col = int(rng.integers(8, max(9, shape[1]-12)))
        a, b = (int(x) for x in rng.integers(3, 9, 2))
        if a == b:
            b += 1
        if collision == "opposing_diagonals":
            points, names = ((row+a, col-a), (row+b, col+b+1)), ("NE", "NW")
        elif collision == "perpendicular":
            points, names = ((row, col-a), (row+b, col)), (str(rng.choice(("E0", "E1"))),
                                                                       str(rng.choice(("N0", "N1"))))
        else:
            points, names = ((row+a, col-a), (row+b, col)), ("NE", str(rng.choice(("N0", "N1"))))
        specs = [_at_port(p, n, k+1) for k, (p, n) in enumerate(zip(points, names))]
        try:
            grid = _make_emitters(shape, specs)
            _, info = _trace(grid)
            _need(len(info["events"]) == 1 and info["events"][0]["kind"] == collision, "wrong primary")
            return grid, specs
        except AssertionError:
            continue
    raise AssertionError("primary construction exhausted")


def _candidate_boxes(grid, info, segment, min_capacity=2):
    own = set().union(*(set(p) for p in segment["paths"]))
    other = set().union(*(set(p) for s in info["segments"] if s is not segment for p in s["paths"]))
    other -= {p[0] for p in segment["paths"] if p}
    boxes = set()
    shortest = min(map(len, segment["paths"]))
    for tick in range(2, shortest-1):
        points = [p[tick] for p in segment["paths"]]
        lowr, highr = min(p[0] for p in points), max(p[0] for p in points)
        lowc, highc = min(p[1] for p in points), max(p[1] for p in points)
        for h, w in ((3, 4), (4, 3), (4, 4), (3, 5), (5, 3), (4, 5), (5, 4), (5, 5), (3, 6), (6, 3)):
            if (h-2)*(w-2) < min_capacity:
                continue
            for r0 in (lowr, lowr-1, highr-h+1):
                for c0 in (lowc, lowc-1, highc-w+1):
                    box = (r0, r0+h, c0, c0+w)
                    if not (0 <= r0 and r0+h <= grid.shape[0] and
                            0 <= c0 and c0+w <= grid.shape[1]):
                        continue
                    cells = _box_cells(box)
                    if not all(any(p in cells for p in path[2:]) for path in segment["paths"]):
                        continue
                    if cells & other or any(not _apart(box, b) for b in info["footprints"]):
                        continue
                    if not np.all(grid[r0:r0+h, c0:c0+w] == info["background"]):
                        continue
                    # A gate must be wholly downstream of the segment origin.
                    if any(p in cells for path in segment["paths"] for p in path[:2]):
                        continue
                    boxes.add(box)
    return sorted(boxes)


def _pack_boxes(rng, candidates, count):
    _need(len(candidates) >= count, "too few gate positions")
    for _ in range(180):
        chosen = []
        for index in rng.permutation(len(candidates)):
            box = candidates[int(index)]
            if all(_apart(box, b) for b in chosen):
                chosen.append(box)
                if len(chosen) == count:
                    return chosen
    raise AssertionError("gate packing exhausted")


def _paint_record(rng, grid, box, marks, filler=9):
    r0, r1, c0, c1 = box
    _need(len(marks) <= (r1-r0-2)*(c1-c0-2), "marks exceed interior")
    grid[r0:r1, c0:c1] = filler
    interior = [(r, c) for r in range(r0+1, r1-1) for c in range(c0+1, c1-1)]
    for k, color in zip(rng.permutation(len(interior)), rng.permutation(marks)):
        grid[interior[int(k)]] = color


def _add_pre_gate(rng, grid, color, opened):
    _, info = _trace(grid)
    possible = [s for s in info["segments"] if s["query"] == {color}]
    rng.shuffle(possible)
    for segment in possible:
        boxes = _candidate_boxes(grid, info, segment, 2)
        if boxes:
            box = boxes[int(rng.integers(len(boxes)))]
            marks = [color, 4] if opened else [3, 4]
            _paint_record(rng, grid, box, marks)
            return True
    return False


def _add_third(rng, grid, specs):
    _, info = _trace(grid)
    # An opposing-diagonal merge produces a cardinal bundle. A third
    # perpendicular strand can join it without a staircase re-encounter.
    segment = next(s for s in info["segments"] if len(s["query"]) == 2)
    for _ in range(80):
        if min(map(len, segment["paths"])) < 10:
            return None
        tick = int(rng.integers(2, 5))
        target = segment["paths"][0][tick]
        distance = int(rng.integers(5, 10))
        new = _at_port((target[0], target[1]-distance), "E0", 3)
        try:
            candidate = _make_emitters(grid.shape, specs+[new])
            _, details = _trace(candidate)
            _need(len(details["events"]) == 2 and details["events"][-1]["query"] == {1, 2, 3},
                  "third source did not join")
            return candidate, specs+[new]
        except AssertionError:
            continue
    return None


def _scene(rng, collision, package, gate_count=3, chain=False):
    last_error = None
    for attempt in range(100):
        try:
            grid, specs = _primary(rng, collision)
            if package == "blocked":
                _need(not chain and _add_pre_gate(rng, grid, 1, False),
                      "no pre-collision blocker")
            if chain:
                extended = _add_third(rng, grid, specs)
                _need(extended is not None, "no chain")
                grid, specs = extended
            _, info = _trace(grid)
            q = frozenset({2}) if package == "blocked" else frozenset(range(1, 4 if chain else 3))
            segment = next(s for s in info["segments"] if s["query"] == q)
            # Rich labels need bigger interiors, but only one or two records
            # need them. Footprints themselves never encode qualification.
            candidates = _candidate_boxes(grid, info, segment,
                                          5 if package == "extras" else max(2, len(q)))
            boxes = _pack_boxes(rng, candidates, gate_count)
            for box in boxes:
                _paint_record(rng, grid, box, sorted(q))
            _, passing = _trace(grid)
            order = [v["record"] for v in passing["visits"] if v["query"] == q]
            _need(len(order) == gate_count, "gate chain not reached as planned")
            ordered = [passing["records"][gid]["box"] for gid in order]
            if package in ("identity", "extras"):
                blocker = 1
            elif package == "blocked":
                blocker = None
            else:
                blocker = int(rng.integers(gate_count+1))
                if blocker == gate_count:
                    blocker = None
            missing = 1 if package == "identity" else 2 if package == "extras" else int(rng.choice(sorted(q)))
            for index, box in enumerate(ordered):
                capacity = (box[1]-box[0]-2)*(box[3]-box[2]-2)
                marks = sorted(q)
                if index == blocker:
                    marks = [c for c in sorted(q) if c != missing]+[4]
                if package == "extras":
                    if index == blocker:
                        marks = [c for c in sorted(q) if c != missing]+[4, 5, 1, 4]
                    else:
                        marks = sorted(q)+[4, 1, 4]
                elif package == "free" and rng.random() < .6:
                    allowed = marks+[5, 6]
                    # Repetition and extra colors occur in both classes.
                    while len(marks) < min(capacity, int(rng.integers(len(marks)+1, len(marks)+5))):
                        marks.append(int(rng.choice(allowed)))
                _paint_record(rng, grid, box, marks)
            if package == "extras":
                _need(_add_pre_gate(rng, grid, 1, True), "no local singleton teaching gate")
            elif package == "blocked":
                # Block color 1 before its hypothetical merge. Downstream
                # gates ask for color 2 only, which now suffices to open them.
                for box in ordered:
                    _paint_record(rng, grid, box, [2, 4])
            out, final = _trace(grid)
            _need(final["erased"], "no gate opened")
            if package in ("identity", "extras"):
                _need(len(final["stopped"]) == 1 and len(final["events"]) == (2 if chain else 1),
                      "lost teaching interaction")
                _need(len(final["records"])-len(final["erased"])-len(final["stopped"]) >= 1,
                      "no shadowed record")
            if package == "blocked":
                _need(not final["events"] and len(final["stopped"]) == 1,
                      "blocker did not prevent the collision")
                _need(any(v["opened"] and v["query"] == {2} and 1 not in v["labels"]
                          for v in final["visits"]), "downstream query did not shrink")
            return grid, out
        except (AssertionError, StopIteration) as error:
            last_error = str(error)
            continue
    raise AssertionError(f"scene construction exhausted: {collision}/{package}/{chain}: {last_error}")


def _transform(grid, rotation, reflection):
    result = np.rot90(grid, rotation)
    if reflection:
        result = np.fliplr(result)
    return np.ascontiguousarray(result)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    kinds = ["opposing_diagonals", "perpendicular", "staircase"]
    rng.shuffle(kinds)
    taught_kinds = kinds[:max(1, min(3, nt-2))]
    # Every puzzle demonstrates identity and occlusion even at the minimum
    # count. Additional examples teach a prevented collision and optional chain.
    chain_enabled = nt >= 4 and "opposing_diagonals" in taught_kinds and rng.random() < .45
    puzzle, seen = {"train": [], "test": []}, set()
    base_palette = rng.permutation(10).astype(np.int8)
    for split, count in (("train", nt), ("test", ne)):
        if split == "test":
            # A chain can consume the last free teaching slot. Derive the
            # actual taught vocabulary from returned grids, not the plan.
            returned = [_trace(p["input"])[1] for p in puzzle["train"]]
            taught_kinds = sorted({e["kind"] for info in returned for e in info["events"]})
            chain_enabled = any(e["depth"] > 1 for info in returned for e in info["events"])
        for i in range(count):
            if split == "train":
                package = "identity" if i == 0 else "extras" if i == 1 else "blocked" if i == 2 else "free"
                collision = taught_kinds[0] if i < 3 else taught_kinds[(i-2) % len(taught_kinds)]
                is_chain = chain_enabled and i == nt-1
                if is_chain:
                    collision = "opposing_diagonals"
            else:
                package = "blocked" if nt >= 3 and rng.random() < .30 else "free"
                collision = str(rng.choice(taught_kinds))
                is_chain = chain_enabled and package != "blocked" and rng.random() < .5
                if is_chain:
                    collision = "opposing_diagonals"
            for retry in range(12):
                number = 3 if package in ("identity", "extras") else int(rng.integers(2, 5 if split == "test" else 4))
                if is_chain:
                    number = 2
                inp, out = _scene(rng, collision, package, number, is_chain)
                palette = base_palette if rng.random() < .25 else rng.permutation(10).astype(np.int8)
                rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
                inp = _transform(palette[inp], rotation, reflection)
                out = _transform(palette[out], rotation, reflection)
                if _key(inp) not in seen:
                    break
            else:
                raise AssertionError("distinct-input construction exhausted")
            seen.add(_key(inp))
            puzzle[split].append({"input": inp, "output": out})
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal counts")
    seen, train_infos, test_infos = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "shape changed")
            _need(_key(inp) not in seen, "duplicate inputs")
            seen.add(_key(inp))
            expected, info = _trace(inp)
            _need(np.array_equal(expected, out), "wrong passport-beam output")
            _need(info["records"] and 2 <= len(info["launchers"]) <= 3, "missing family objects")
            _need(np.any(inp != out), "identity pair")
            for l in info["launchers"]:
                for p in _box_cells(l["box"]):
                    if inp[p] != info["background"]:
                        _need(inp[p] == out[p], "emitter modified")
            visited = {v["record"] for v in info["visits"]}
            for gid, record in enumerate(info["records"]):
                r0, r1, c0, c1 = record["box"]
                if gid not in info["erased"]:
                    _need(np.array_equal(inp[r0:r1, c0:c1], out[r0:r1, c0:c1]),
                          "intact or shadowed record changed")
                else:
                    colors = next(v["query"] for v in info["visits"] if v["record"] == gid)
                    _need(set(np.unique(out[r0:r1, c0:c1])) <= set(colors) | {info["background"]},
                          "record was not fully dissolved")
            info["unvisited"] = set(range(len(info["records"]))) - visited
            (train_infos if split == "train" else test_infos).append(info)
    visits = [(info, v, info["records"][v["record"]]) for info in train_infos for v in info["visits"]]
    positives = [(i, v, g) for i, v, g in visits if v["opened"]]
    negatives = [(i, v, g) for i, v, g in visits if not v["opened"]]
    _need(positives and negatives, "missing both gate outcomes")
    _need(any(len(v["query"]) >= 2 and g["labels"] == v["query"]
              and g["marks"] == len(v["query"]) for _, v, g in positives), "no exact identity witness")
    _need(any(len(v["query"]) >= 2 and len(g["labels"]) >= len(v["query"])
              and g["marks"] >= len(v["query"]) for _, v, g in negatives), "no count-shortcut negative")
    _need(any(len(v["query"]) == 1 and v["opened"] and
              not frozenset(l["color"] for l in i["launchers"]) <= g["labels"]
              for i, v, g in positives), "no locally formed query witness")
    for group in (positives, negatives):
        _need(any(g["labels"]-v["query"] and
                  any(g["counts"].get(c, 0) > 1 for c in v["query"]) and
                  any(g["counts"].get(c, 0) > 1 for c in g["labels"]-v["query"])
                  for _, v, g in group), "missing repeated query/extra-label witness")
    _need(any(any(frozenset(l["color"] for l in i["launchers"]) <= i["records"][gid]["labels"]
                       for gid in i["unvisited"]) for i in train_infos), "no qualifying shadow")
    # Omitting A changes propagation; omitting B changes which records disappear.
    _need(any(not np.array_equal(_trace(p["input"], containment=False)[0], p["output"])
              for p, info in zip(puzzle["train"], train_infos)
              if info["events"] and any(not v["opened"] and len(v["query"]) >= 2
                                        for v in info["visits"])),
          "containment is dispensable")
    _need(any(any(v["opened"] for v in i["visits"]) for i in train_infos), "emission is dispensable")
    taught = {e["kind"] for i in train_infos for e in i["events"]}
    _need({e["kind"] for i in test_infos for e in i["events"]} <= taught, "untaught collision mechanism")
    if any(any(e["depth"] > 1 for e in i["events"]) for i in test_infos):
        _need(any(any(e["depth"] > 1 for e in i["events"]) for i in train_infos), "untaught chain")
    if any(not i["events"] for i in test_infos):
        _need(any(not i["events"] for i in train_infos), "untaught prevented collision")


def _geometry_fixtures():
    """Explicit expected pixels for three unequal-distance collisions.

    These expectations are constructed from manually chosen last separate
    cells, rather than from _walk, _encounter, or _trace.
    """
    cases = [
        ("perpendicular", (20, 26), ((8, 5), (14, 13)), ("E0", "N0"),
         ((8, 12), (9, 13)), (-1, 1)),
        ("opposing_diagonals", (20, 26), ((13, 5), (12, 17)), ("NE", "NW"),
         ((7, 11), (7, 12)), (-1, 0)),
        ("staircase", (20, 24), ((14, 5), (12, 14)), ("NE", "N0"),
         ((6, 13), (6, 14)), (-2, 1)),
    ]
    for name, shape, origins, names, lasts, velocity in cases:
        specs = [_at_port(p, n, i+1) for i, (p, n) in enumerate(zip(origins, names))]
        inp = _make_emitters(shape, specs)
        expected = inp.copy()
        for color, origin, port, last in zip((1, 2), origins, names, lasts):
            dr, dc = _direction(PORTS[port])
            r, c = origin
            while (r, c) != last:
                expected[r, c] = color
                r, c = r+dr, c+dc
            expected[r, c] = color
            step = 0
            while True:
                step += 1
                if name == "staircase":
                    # N, NE, N, NE... (first step has no east component).
                    r -= 1
                    if step % 2 == 0:
                        c += 1
                else:
                    r, c = r+velocity[0], c+velocity[1]
                if not _inside((r, c), shape):
                    break
                expected[r, c] = color
        out, info = _trace(inp)
        _need(np.array_equal(out, expected), f"{name} fixture pixels disagree")
        _need(len(info["events"]) == 1 and info["events"][0]["unequal"],
              "fixture did not expose spatial scheduling")
    # A single emitter with two opposite side ports illustrates multiplicity
    # independently of the sampled one-port vocabulary. One ray has a gate.
    inp = _make_emitters((12, 15), [{"core": (5, 3), "color": 1,
                                  "ports": ("E0", "W1")}])
    rng = np.random.default_rng(810)
    _paint_record(rng, inp, (4, 8, 8, 12), [1, 1, 4])
    out, info = _trace(inp)
    _need(info["erased"] == {0}, "two-port record should open")
    _need(np.all(out[5, 5:] == 1) and np.all(out[6, :3] == 1),
          "both opposite-port rays should continue")
    # An empty, off-route record must remain, and an empty reached one blocks.
    inp = _make_emitters((14, 18), [{"core": (6, 2), "color": 1, "ports": ("E0",)}])
    _paint_record(rng, inp, (5, 9, 8, 12), [])
    _paint_record(rng, inp, (1, 4, 12, 16), [])
    out, info = _trace(inp)
    _need(not info["erased"] and len(info["stopped"]) == 1,
          "filler-only blocker or off-route record behaved incorrectly")
    _need(np.array_equal(out[5:9, 8:12], inp[5:9, 8:12]), "empty blocker modified")


def _membership_counterfactual(puzzle):
    """Keep geometry fixed, repair one missing label, and check propagation."""
    for pair in puzzle["train"]:
        _, info = _trace(pair["input"])
        for visit in info["visits"]:
            if visit["opened"] or len(visit["query"]) < 2:
                continue
            gid = visit["record"]
            record = info["records"][gid]
            changed = pair["input"].copy()
            r0, r1, c0, c1 = record["box"]
            interior = [(r, c) for r in range(r0+1, r1-1) for c in range(c0+1, c1-1)]
            changed[r0:r1, c0:c1] = record["filler"]
            for point, color in zip(interior, sorted(visit["query"])):
                changed[point] = color
            out, after = _trace(changed)
            _need(gid in after["erased"] and not np.array_equal(out, pair["output"]),
                  "repairing membership did not open the route")
            # Relocating labels within the same record cannot change its query
            # or resulting dissolved pixels.
            relocated = changed.copy()
            for point in interior:
                relocated[point] = record["filler"]
            for point, color in zip(reversed(interior), sorted(visit["query"])):
                relocated[point] = color
            _need(np.array_equal(solve(relocated), out), "mark position affected containment")
            return
    raise AssertionError("no missing-identity counterfactual")


def self_test() -> dict:
    """Bounded semantic fixtures, counterfactuals, and 42-seed variation audit."""
    _geometry_fixtures()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _need(len({_puzzle_key(p) for p in puzzles}) == 42, "puzzles repeat")
    shapes, roles, layouts, kinds, velocities, queries, gate_counts, depths = (set() for _ in range(8))
    blocked, shadows, extras, repeats = 0, 0, 0, 0
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        _membership_counterfactual(puzzle)
        if seed < 3:
            _need(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                _, info = _trace(inp)
                shapes.add(inp.shape)
                roles.add((info["background"], info["structure"], info["records"][0]["filler"]))
                layouts.add((inp.shape, tuple(l["core"] for l in info["launchers"]),
                             tuple(g["box"] for g in info["records"])))
                kinds.update(e["kind"] for e in info["events"])
                velocities.update(e["outgoing"] for e in info["events"])
                depths.update(e["depth"] for e in info["events"])
                gate_counts.add(len(info["records"]))
                blocked += not info["events"]
                shadows += len(info["records"])-len(info["visits"])
                for v in info["visits"]:
                    queries.add(len(v["query"]))
                    g = info["records"][v["record"]]
                    extras += bool(g["labels"]-v["query"])
                    repeats += g["marks"] > len(g["labels"])
                # D4 covariance tests the staircase phase and local directions.
                rotated = np.rot90(inp)
                _need(np.array_equal(solve(rotated), np.rot90(out)), "rotation changed the law")
                _need(np.array_equal(solve(np.fliplr(inp)), np.fliplr(out)), "reflection changed the law")
    _need(len(shapes) >= 20 and len(layouts) >= 150, "geometry diversity collapsed")
    _need(len(roles) >= 80 and all({p[k] for p in roles} == set(range(10)) for k in range(3)),
          "palette roles collapsed")
    _need(kinds == {"opposing_diagonals", "perpendicular", "staircase"}, "collision kinds missing")
    _need(len(velocities) >= 12 and queries >= {1, 2}, "direction or local-query diversity collapsed")
    _need(blocked > 0 and shadows > 0 and extras > 0 and repeats > 0, "semantic variation missing")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(143, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    return {"seeds": 42, "shapes": len(shapes), "layouts": len(layouts),
            "role_assignments": len(roles), "collision_kinds": sorted(kinds),
            "outgoing_headings": len(velocities), "query_sizes": sorted(queries),
            "record_counts": sorted(gate_counts), "collision_depths": sorted(depths),
            "prevented_collision_pairs": blocked, "unreached_records": shadows,
            "extra_label_visits": extras, "repeated_label_visits": repeats}
