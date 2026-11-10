"""Dissolving searchlights: cap keys control endpoint-ray permeability.

Only standard library and NumPy are used. No parent code, example lookup,
generation metadata, or random state participates in solve().
"""
from __future__ import annotations

from collections import Counter
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _field(shape, background):
    return np.asarray(background, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _ring(center):
    r, c = center
    return [(r + a, c + b) for a in range(-2, 3) for b in range(-2, 3)
            if abs(a) + abs(b) == 2]


def _cap(center, direction):
    r, c = center
    a, b = direction
    return [(r + a, c + 3*b), (r + 2*a, c + 2*b), (r + 3*a, c + b)]


def _path(start, direction, shape):
    r, c = start
    a, b = direction
    result = []
    r, c = r + a, c + b
    while _inside((r, c), shape):
        result.append((r, c))
        r, c = r + a, c + b
    return result


def _gap(a, b):
    # Half-open boxes: strict comparison requires a full empty separator.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _cells(box):
    r0, r1, c0, c1 = box
    return [(r, c) for r in range(r0, r1) for c in range(c0, c1)]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _array(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _geometry_info(shape, emitters):
    zones, caps, owners = set(), [], {}
    for e in emitters:
        r, c = e["center"]
        zone = set(_cells((r-3, r+4, c-3, c+4)))
        assert all(_inside(p, shape) for p in zone)
        assert not zones & zone, "Emitter neighborhoods overlap"
        zones.update(zone)
        for direction, colors in sorted(e["caps"].items()):
            pts = _cap(e["center"], direction)
            cap = {"center": e["center"], "direction": direction,
                   "points": pts, "colors": tuple(colors), "query": frozenset(colors),
                   "paths": [_path(pts[0], direction, shape),
                             _path(pts[2], direction, shape)]}
            cap_id = len(caps)
            caps.append(cap)
            for path in cap["paths"]:
                for p in path:
                    assert p not in owners, "Potential endpoint paths cross"
                    owners[p] = cap_id
    assert not zones.intersection(owners), "A ray enters an emitter neighborhood"
    return zones, caps, owners


def _analyze(grid):
    """Recover all roles and objects solely from the input pixels."""
    grid = _array(grid)
    h, w = grid.shape
    parity = np.indices(grid.shape).sum(axis=0) % 2
    background = []
    for i in (0, 1):
        values = grid[parity == i]
        assert len(values)
        counts = np.bincount(values, minlength=10)
        color = int(counts.argmax())
        assert 2*int(counts[color]) > len(values), "Background lacks a parity majority"
        background.append(color)
    assert background[0] != background[1]
    background = tuple(background)
    base = _field(grid.shape, background)
    emitters = []
    for r in range(3, h-3):
        for c in range(3, w-3):
            ring = _ring((r, c))
            s = int(grid[ring[0]])
            if s in background or any(int(grid[p]) != s for p in ring):
                continue
            interior = [(r+a, c+b) for a in range(-1, 2) for b in range(-1, 2)
                        if abs(a)+abs(b) < 2]
            if any(grid[p] != base[p] for p in interior):
                continue
            caps = {}
            for d in DIRECTIONS:
                pts = _cap((r, c), d)
                colors = tuple(int(grid[p]) for p in pts)
                if all(grid[p] == base[p] for p in pts):
                    continue
                assert len(set(colors)) == 3 and not set(colors) & set(background+(s,)), \
                    "Active caps need three complete, distinct label cells"
                caps[d] = colors
            assert caps, "Every emitter needs an active cap"
            emitters.append({"center": (r, c), "struct": s, "caps": caps})
    assert emitters, "No complete hollow diamond"
    assert len({e["struct"] for e in emitters}) == 1
    structural = emitters[0]["struct"]
    zones, caps, owners = _geometry_info(grid.shape, emitters)
    expected = base.copy()
    for e in emitters:
        for p in _ring(e["center"]):
            expected[p] = structural
        for d, colors in e["caps"].items():
            for p, color in zip(_cap(e["center"], d), colors):
                expected[p] = color
    assert all(grid[p] == expected[p] for p in zones), "Unexplained emitter pixels"
    remaining = set(map(tuple, np.argwhere(grid != base))) - zones
    records, lookup = [], {}
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = [start]
        for r, c in queue:
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
        rows, cols = zip(*queue)
        box = (min(rows), max(rows)+1, min(cols), max(cols)+1)
        assert len(queue) == (box[1]-box[0])*(box[3]-box[2]), "Record has a hole"
        counts = Counter(int(grid[p]) for p in queue)
        assert structural in counts and not set(counts) & set(background)
        assert all(_gap(box, other["box"]) for other in records)
        own = {owners[p] for p in queue if p in owners}
        assert len(own) <= 1, "A record meets competing cap queries"
        record = {"box": box, "counts": counts,
                  "labels": frozenset(set(counts)-{structural}),
                  "marks": len(queue)-counts[structural],
                  "owner": next(iter(own)) if own else None}
        rec_id = len(records)
        records.append(record)
        lookup.update({p: rec_id for p in queue})
    return {"grid": grid, "background": background, "base": base,
            "struct": structural, "emitters": emitters, "zones": zones,
            "caps": caps, "owners": owners, "records": records, "lookup": lookup}


def _trace(info, transparent=False, middle=False):
    """Immutable records determine each independent ray's visibility prefix."""
    deleted, paint, traces = set(), {}, []
    for cap_id, cap in enumerate(info["caps"]):
        paths = list(cap["paths"])
        colors = [cap["colors"][0], cap["colors"][2]]
        if middle:
            paths.append(_path(cap["points"][1], cap["direction"], info["grid"].shape))
            colors.append(cap["colors"][1])
        for lane, (path, color) in enumerate(zip(paths, colors)):
            traveled, passed, blocker = [], set(), None
            for p in path:
                # Used by the deliberately wrong middle-lane alternative.
                if p in info["zones"] and (not middle or info["grid"][p] != info["base"][p]):
                    break
                rec = info["lookup"].get(p)
                if rec is not None:
                    if transparent or cap["query"] <= info["records"][rec]["labels"]:
                        passed.add(rec)
                        deleted.add(rec)
                    else:
                        blocker = rec
                        break
                traveled.append(p)
                assert p not in paint or middle
                paint[p] = color
            traces.append({"cap": cap_id, "lane": lane, "path": path,
                           "traveled": traveled, "passed": passed, "blocker": blocker})
    return deleted, paint, traces


def _from_trace(info, deleted, paint):
    result = info["grid"].copy()
    for rec_id in deleted:
        r0, r1, c0, c1 = info["records"][rec_id]["box"]
        result[r0:r1, c0:c1] = info["base"][r0:r1, c0:c1]
    for p, color in paint.items():
        result[p] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    deleted, paint, _ = _trace(info)
    return _from_trace(info, deleted, paint)


def _closed_form(info):
    """Threshold calculation checks the ray walker without its control flow."""
    deleted, paint = set(), {}
    for cap in info["caps"]:
        for path, color in zip(cap["paths"], (cap["colors"][0], cap["colors"][2])):
            contact = {}
            for t, p in enumerate(path):
                rec = info["lookup"].get(p)
                if rec is not None:
                    contact.setdefault(rec, t)
            cut = min((t for rec, t in contact.items()
                       if not cap["query"] <= info["records"][rec]["labels"]),
                      default=len(path))
            deleted.update(rec for rec, t in contact.items() if t < cut)
            paint.update({p: color for p in path[:cut]})
    return _from_trace(info, deleted, paint)


def _valid_box(box, shape, zones, owners, records, owner=None, off_ray=False):
    r0, r1, c0, c1 = box
    if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
        return False
    if any(not _gap(box, rec["box"]) for rec in records):
        return False
    pts = _cells(box)
    if zones.intersection(pts):
        return False
    own = {owners[p] for p in pts if p in owners}
    return len(own) <= 1 and (not off_ray or not own) and (owner is None or own == {owner})


def _on_lane(rng, info, records, cap_id, lane, interval, shapes):
    path = info["caps"][cap_id]["paths"][lane]
    low, high = interval
    candidates = set()
    for t in range(max(0, low-1), min(high, len(path))):
        row, col = path[t]
        for h, w in shapes:
            for r0 in range(row-h+1, row+1):
                for c0 in range(col-w+1, col+1):
                    box = (r0, r0+h, c0, c0+w)
                    if box in candidates:
                        continue
                    if not _valid_box(box, info["shape"], info["zones"], info["owners"],
                                      records, cap_id):
                        continue
                    first = next((k+1 for k, p in enumerate(path)
                                  if r0 <= p[0] < r0+h and c0 <= p[1] < c0+w), None)
                    if first is not None and low <= first <= high:
                        candidates.add(box)
    if not candidates:
        raise RuntimeError("No compatible lane-contact rectangle")
    options = sorted(candidates)
    return options[int(rng.integers(len(options)))]


def _middle_box(rng, info, records, cap_id, size, interval, bridge):
    cap = info["caps"][cap_id]
    r, c = cap["points"][1]
    a, b = cap["direction"]
    options = []
    for t in range(interval[0], interval[1]+1):
        rr, cc = r+a*t, c+b*t
        if size == 2:
            box = (min(rr, rr+a), max(rr, rr+a)+1,
                   min(cc, cc+b), max(cc, cc+b)+1)
        else:
            h, w = size if isinstance(size, tuple) else (size, size)
            box = (rr-h//2, rr-h//2+h, cc-w//2, cc-w//2+w)
        if not _valid_box(box, info["shape"], info["zones"], info["owners"], records,
                          cap_id if bridge else None, off_ray=not bridge):
            continue
        if bridge and not all(set(_cells(box)).intersection(p) for p in cap["paths"]):
            continue
        options.append(box)
    if not options:
        raise RuntimeError("No compatible middle/bridge record")
    return options[int(rng.integers(len(options)))]


def _record(rng, box, marks, structural):
    h, w = box[1]-box[0], box[3]-box[2]
    assert len(marks) < h*w
    body = np.full((h, w), structural, dtype=np.int8)
    for pos, color in zip(rng.choice(h*w, len(marks), replace=False), rng.permutation(marks)):
        body[int(pos)//w, int(pos)%w] = color
    return {"box": box, "body": body, "labels": frozenset(marks)}


def _random_marks(rng, query, labels, area):
    q = list(query)
    extras = [x for x in labels if x not in q]
    if rng.random() < .12:
        return []
    positive = area >= 4 and rng.random() < .55
    identities = q if positive else list(rng.permutation(q)[:int(rng.integers(3))])
    identities += list(rng.permutation(extras)[:int(rng.integers(len(extras)+1))])
    identities = identities[:area-1]
    if not identities:
        return []
    target = int(rng.integers(len(identities), max(len(identities), min(area-1, int(area*.8)))+1))
    return identities + [int(rng.choice(identities)) for _ in range(target-len(identities))]


def _make_geometry(rng, colors, kind, count):
    background, structural, labels = tuple(colors[:2]), colors[2], colors[3:]
    if kind == "identity":
        shape = tuple(int(rng.integers(27, 31)) for _ in range(2))
        center = tuple(int(rng.integers(10, n-10)) for n in shape)
        query = [int(x) for x in rng.choice(labels, 3, replace=False)]
        emitters = [{"center": center, "struct": structural,
                     "caps": {d: tuple(map(int, rng.permutation(query))) for d in DIRECTIONS}}]
    elif kind == "cascade":
        shape = tuple(int(rng.integers(28, 31)) for _ in range(2))
        center = tuple(int(rng.integers(5, 8)) for _ in range(2))
        mask = int(rng.integers(1, 16)) | 4
        emitters = [{"center": center, "struct": structural,
                     "caps": {d: tuple(map(int, rng.choice(labels, 3, replace=False)))
                              for j, d in enumerate(DIRECTIONS) if mask & (1 << j)}}]
    else:
        low = 18 if count == 1 else 25
        shape = tuple(int(rng.integers(low, 31)) for _ in range(2))
        emitters = []
        for _ in range(count):
            for attempt in range(120):
                center = tuple(int(rng.integers(3, n-3)) for n in shape)
                mask = int(rng.integers(1, 16))
                e = {"center": center, "struct": structural,
                     "caps": {d: tuple(map(int, rng.choice(labels, 3, replace=False)))
                              for j, d in enumerate(DIRECTIONS) if mask & (1 << j)}}
                try:
                    _geometry_info(shape, emitters+[e])
                except AssertionError:
                    continue
                emitters.append(e)
                break
            else:
                raise RuntimeError("Incompatible emitter placement")
    zones, caps, owners = _geometry_info(shape, emitters)
    return {"shape": shape, "background": background, "struct": structural,
            "labels": labels, "emitters": emitters, "zones": zones,
            "caps": caps, "owners": owners}


def _free_records(rng, info, records, count, off_ray=False, large=False):
    shapes = [(1, 4), (4, 1), (2, 2), (2, 3), (3, 2), (3, 3),
              (2, 5), (5, 2), (4, 4), (3, 5), (5, 3)]
    if large:
        shapes += [(3, 7), (7, 3), (5, 5), (2, 7), (7, 2)]
    for _ in range(count):
        box = None
        cap_id = int(rng.integers(len(info["caps"])))
        if not off_ray and rng.random() < .70:
            lane = int(rng.integers(2))
            try:
                box = _on_lane(rng, info, records, cap_id, lane,
                               (1, len(info["caps"][cap_id]["paths"][lane])), shapes)
            except RuntimeError:
                pass
        if box is None:
            for attempt in range(160):
                h, w = shapes[int(rng.integers(len(shapes)))]
                r = int(rng.integers(info["shape"][0]-h+1))
                c = int(rng.integers(info["shape"][1]-w+1))
                candidate = (r, r+h, c, c+w)
                if _valid_box(candidate, info["shape"], info["zones"], info["owners"],
                              records, off_ray=off_ray):
                    box = candidate
                    break
        if box is not None:
            area = (box[1]-box[0])*(box[3]-box[2])
            records.append(_record(rng, box, _random_marks(rng, info["caps"][cap_id]["query"],
                                                         info["labels"], area), info["struct"]))


def _render(info, records):
    """Paint from specifications; labels are not obtained by calling solve."""
    grid = _field(info["shape"], info["background"])
    for e in info["emitters"]:
        for p in _ring(e["center"]):
            grid[p] = info["struct"]
        for d, colors in e["caps"].items():
            for p, color in zip(_cap(e["center"], d), colors):
                grid[p] = color
    lookup = {}
    for i, rec in enumerate(records):
        r0, r1, c0, c1 = rec["box"]
        grid[r0:r1, c0:c1] = rec["body"]
        lookup.update({p: i for p in _cells(rec["box"])})
    authored = dict(info, grid=grid, base=_field(grid.shape, info["background"]),
                    records=records, lookup=lookup)
    out = _closed_form(authored)
    # Free rotations/reflections change both contact geometry and phase.
    return {"input": grid, "output": out}


def _make_pair(rng, colors, kind, count=1, contact_shift=False):
    info = _make_geometry(rng, colors, kind, count)
    records = []
    structural = info["struct"]
    if kind == "identity":
        # Sampling-only reservation for clear hollow-strip witnesses. These
        # background cells are not special markers or solver conventions.
        for cap in info["caps"]:
            r, c = cap["points"][1]
            a, b = cap["direction"]
            info["zones"].update((r+a*k, c+b*k) for k in (1, 2))
        # Sorted cap order is NW, NE, SW, SE; all four are present.
        cap = info["caps"][0]
        lane = int(rng.integers(2))
        box = _on_lane(rng, info, records, 0, lane, (2, 3), [(2, 2)])
        records.append(_record(rng, box, list(cap["colors"]), structural))
        for cap_id, missing in ((0, 0), (1, 1), (2, 2)):
            cap = info["caps"][cap_id]
            q = list(cap["colors"])
            x = [v for v in info["labels"] if v not in q]
            marks = [q[j] for j in range(3) if j != missing] + [x[0]]
            if cap_id == 1:
                marks += x[1:3]
            elif cap_id == 2:
                marks += [marks[0], x[0]]
            dims = [(2, 2)] if cap_id == 0 else [(3, 3)]
            box = _on_lane(rng, info, records, cap_id,
                           lane if cap_id == 0 else int(rng.integers(2)),
                           (6, 8) if cap_id == 0 else (4, 7), dims)
            records.append(_record(rng, box, marks, structural))
        cap_id = 3
        box = _middle_box(rng, info, records, cap_id, 3, (3, 7), True)
        records.append(_record(rng, box, list(info["caps"][cap_id]["colors"]), structural))
        # A matching record entirely between the two endpoint lines survives.
        choices = list(map(int, rng.permutation(4)))
        for cap_id in choices:
            try:
                box = _middle_box(rng, info, records, cap_id, 2, (3, 12), False)
                records.append(_record(rng, box, list(info["caps"][cap_id]["colors"]), structural))
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError("Missing between-lane teaching record")
        _free_records(rng, info, records, int(rng.integers(3)), off_ray=True)
    elif kind == "cascade":
        cap_id = next(i for i, c in enumerate(info["caps"]) if c["direction"] == (1, 1))
        cap = info["caps"][cap_id]
        q = list(cap["colors"])
        x = next(v for v in info["labels"] if v not in q)
        lane = int(rng.integers(2))
        requests = [((2, 3), (2, 2), q),
                    ((6, 7), (3, 3), q+[x, q[0], x]),
                    ((11, 12), (2, 2), [q[0], q[2], x]),
                    ((15, 17), (2, 2), q)]
        for interval, shape, marks in requests:
            box = _on_lane(rng, info, records, cap_id, lane, interval, [shape])
            records.append(_record(rng, box, marks, structural))
        _free_records(rng, info, records, int(rng.integers(1, 5)), off_ray=True)
    else:
        if contact_shift:
            options = list(map(int, rng.permutation(len(info["caps"]))))
            for cap_id in options:
                try:
                    shape = (3, 7) if rng.random() < .5 else (7, 3)
                    box = _middle_box(rng, info, records, cap_id, shape, (3, 10), True)
                    q = list(info["caps"][cap_id]["colors"])
                    x = next(v for v in info["labels"] if v not in q)
                    records.append(_record(rng, box, q+[x, q[0]], structural))
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError("No wide two-lane test gate")
        _free_records(rng, info, records, int(rng.integers(3, 12)), large=contact_shift)
    pair = _render(info, records)
    recovered = _analyze(pair["input"])
    assert len(recovered["emitters"]) == len(info["emitters"])
    assert len(recovered["records"]) == len(records)
    assert np.array_equal(pair["output"], solve(pair["input"]))
    assert not np.array_equal(pair["input"], pair["output"]), "Resample an uninformative scene"
    turn, flip = int(rng.integers(4)), bool(rng.integers(2))
    for role in pair:
        a = np.rot90(pair[role], turn)
        pair[role] = (a[:, ::-1] if flip else a).copy()
    return pair


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable = bool(rng.integers(2))
    base_colors = list(map(int, rng.permutation(10)))
    shift = str(rng.choice(("more_emitters", "contact_geometry")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            kind = ("identity" if index == 0 else "cascade" if index == 1 else "free") \
                if split == "train" else "free"
            colors = base_colors if stable else list(map(int, rng.permutation(10)))
            n = (1 if kind != "free" else int(rng.integers(2, 4))
                 if split == "test" and shift == "more_emitters" else int(rng.integers(1, 3)))
            for attempt in range(80):
                try:
                    pair = _make_pair(rng, colors, kind, n,
                                      split == "test" and shift == "contact_geometry")
                    key = _key(pair["input"])
                    if key in seen:
                        continue
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
                except (AssertionError, RuntimeError):
                    continue
            else:
                raise RuntimeError(f"Bounded scene construction exhausted: {kind}/{n}")
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(info, output):
    deleted, paint, traces = _trace(info)
    records, caps = info["records"], info["caps"]
    result = set()
    for t in traces:
        cap = caps[t["cap"]]
        middle = cap["points"][1]
        a, b = cap["direction"]
        probes = [(middle[0]+a*k, middle[1]+b*k) for k in (1, 2)]
        sibling = next(s for s in traces if s["cap"] == t["cap"] and s["lane"] != t["lane"])
        if min(len(t["traveled"]), len(sibling["traveled"])) >= 2 and all(_inside(p, output.shape)
                and info["grid"][p] == output[p] == info["base"][p] for p in probes):
            result.add(("direction", cap["direction"]))
        if len(t["passed"]) >= 2:
            result.add(("cascade",))
        if t["blocker"] is not None:
            rec = records[t["blocker"]]
            for j, color in enumerate(cap["colors"]):
                if rec["labels"] & cap["query"] == cap["query"]-{color}:
                    result.add(("missing", j))
            if rec["marks"] == 3 and len(rec["labels"]) == 3:
                result.add(("three_negative",))
            if len(rec["labels"]) > 3:
                result.add(("colorful_negative",))
            if len(sibling["traveled"]) > len(t["traveled"])+2:
                result.add(("independent_stopping",))
            after = set(t["path"][len(t["traveled"])+1:])
            for rec_id, candidate in enumerate(records):
                if rec_id not in deleted and cap["query"] <= candidate["labels"] \
                        and after.intersection(_cells(candidate["box"])):
                    result.add(("shadow_match",))
        for rec_id in t["passed"]:
            rec = records[rec_id]
            if rec["marks"] == 3:
                result.add(("exact_positive",))
            extras = rec["labels"]-cap["query"]
            if extras and any(rec["counts"].get(q, 0) > 1 for q in cap["query"]) \
                    and any(rec["counts"].get(x, 0) > 1 for x in extras):
                result.add(("repeated_extra_positive",))
            pts = set(_cells(rec["box"]))
            if all(pts.intersection(path) for path in cap["paths"]):
                result.add(("two_lane_gate",))
            if pts-set(paint):
                result.add(("whole_record",))
    for rec_id, rec in enumerate(records):
        if rec_id not in deleted and rec["owner"] is None \
                and any(cap["query"] <= rec["labels"] for cap in caps):
            result.add(("off_ray_match",))
    positive_shapes = {(rec["box"][1]-rec["box"][0], rec["box"][3]-rec["box"][2])
                       for i, rec in enumerate(records) if i in deleted and rec["marks"] == 3
                       and len(rec["labels"]) == 3}
    negative_shapes = {(records[t["blocker"]]["box"][1]-records[t["blocker"]]["box"][0],
                        records[t["blocker"]]["box"][3]-records[t["blocker"]]["box"][2])
                       for t in traces if t["blocker"] is not None
                       and records[t["blocker"]]["marks"] == 3
                       and len(records[t["blocker"]]["labels"]) == 3}
    if positive_shapes & negative_shapes:
        result.add(("matched_counts_geometry",))
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence = set(), set()
    wrong_transparent = wrong_global = wrong_middle = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            deleted, paint, traces = _trace(info)
            assert np.array_equal(out, _from_trace(info, deleted, paint))
            assert np.array_equal(out, _closed_form(info))
            assert not np.array_equal(inp, out)
            for p in info["zones"]:
                assert inp[p] == out[p], "An emitter was changed"
            for rec_id, rec in enumerate(info["records"]):
                pts = _cells(rec["box"])
                if rec_id in deleted:
                    assert all(out[p] == paint.get(p, int(info["base"][p])) for p in pts)
                else:
                    assert all(out[p] == inp[p] for p in pts), "A survivor was damaged"
            for t in traces:
                if t["blocker"] is not None:
                    # No paint reaches the contact point or passes its barrier.
                    assert not set(t["path"][len(t["traveled"]):]).intersection(paint)
            if split == "train":
                evidence.update(_evidence(info, out))
                d, p, _ = _trace(info, transparent=True)
                wrong_transparent |= not np.array_equal(out, _from_trace(info, d, p))
                global_deleted = {i for i, rec in enumerate(info["records"])
                                  if any(cap["query"] <= rec["labels"] for cap in info["caps"])}
                wrong_global |= not np.array_equal(out, _from_trace(info, global_deleted, paint))
                if len(info["emitters"]) == 1:
                    d, p, _ = _trace(info, middle=True)
                    wrong_middle |= not np.array_equal(out, _from_trace(info, d, p))
    required = {("direction", d) for d in DIRECTIONS} | {("missing", i) for i in range(3)}
    required |= {(name,) for name in ("exact_positive", "three_negative", "colorful_negative",
                  "repeated_extra_positive", "cascade", "independent_stopping", "shadow_match",
                  "off_ray_match", "two_lane_gate", "whole_record", "matched_counts_geometry")}
    assert required <= evidence, f"Missing teaching evidence: {required-evidence}"
    assert wrong_transparent and wrong_global and wrong_middle, "A parent-omitting shortcut survived"


def _fingerprint(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded, substantive checks across 42 seeds and API boundaries."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    shapes, geometry, masks, counts, palettes, densities = set(), set(), set(), set(), set(), set()
    aspects, lengths, contacts = set(), [], set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_structures, local_shapes = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                shapes.add(info["grid"].shape)
                local_shapes.add(info["grid"].shape)
                local_structures.add((info["background"], info["struct"]))
                counts.add(len(info["emitters"]))
                palettes.add((info["background"], info["struct"]))
                geometry.add((info["grid"].shape,
                              tuple((e["center"], tuple(sorted(e["caps"]))) for e in info["emitters"]),
                              tuple(rec["box"] for rec in info["records"])))
                for e in info["emitters"]:
                    masks.add(tuple(sorted(e["caps"])))
                for cap in info["caps"]:
                    lengths.extend(map(len, cap["paths"]))
                for rec in info["records"]:
                    h, w = rec["box"][1]-rec["box"][0], rec["box"][3]-rec["box"][2]
                    aspects.add("square" if h == w else "wide" if w > h else "tall")
                    densities.add(round(rec["marks"]/(h*w), 3))
                    if rec["owner"] is not None:
                        pts = set(_cells(rec["box"]))
                        contacts.add(sum(bool(pts.intersection(path))
                                         for path in info["caps"][rec["owner"]]["paths"]))
        assert len(local_shapes) >= 2
    assert len(shapes) >= 35 and len(geometry) >= 180 and len(palettes) >= 30
    assert counts == {1, 2, 3} and len(masks) >= 12
    assert aspects == {"square", "wide", "tall"} and contacts == {1, 2}
    assert min(lengths) == 0 and max(lengths) >= 20
    assert min(densities) == 0 and max(densities) >= .7
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(271828, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Dihedral symmetry and literal color permutations preserve the rule.
    pair = puzzles[0]["train"][0]
    for turn in range(4):
        for flip in (False, True):
            inp = np.rot90(pair["input"], turn)
            out = np.rot90(pair["output"], turn)
            if flip:
                inp, out = inp[:, ::-1], out[:, ::-1]
            assert np.array_equal(solve(inp), out)
    color_map = np.array([9, 5, 1, 7, 3, 0, 8, 2, 6, 4], dtype=np.int8)
    assert np.array_equal(solve(color_map[pair["input"]]), color_map[pair["output"]])
    # Keep geometry and both endpoint colors fixed; only the middle key changes.
    key_witness = False
    for pair in puzzles[0]["train"]:
        info = _analyze(pair["input"])
        for cap in info["caps"]:
            # Reordering the key may change paint colors, never membership.
            permuted = pair["input"].copy()
            for p, color in zip(cap["points"], cap["colors"][1:]+cap["colors"][:1]):
                permuted[p] = color
            assert _trace(_analyze(permuted))[0] == _trace(info)[0]
            replacement = next(c for c in range(10)
                               if c not in set(info["background"]+(info["struct"],)) | cap["query"])
            changed = pair["input"].copy()
            changed[cap["points"][1]] = replacement
            actual = solve(changed)
            expected = pair["output"].copy()
            expected[cap["points"][1]] = replacement
            key_witness |= not np.array_equal(actual, expected)
    assert key_witness, "Middle key is not causally necessary"
    # Both an omitted ray cell and partial record deletion must be rejected.
    for mode in ("missing_ray", "partial_clear", "changed_survivor"):
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
        changed = False
        for pair in bad["train"]:
            info = _analyze(pair["input"])
            deleted, paint, _ = _trace(info)
            if mode == "missing_ray" and paint:
                p = next(iter(paint))
                pair["output"][p] = pair["input"][p]
                changed = pair["output"][p] != paint[p]
            elif mode == "partial_clear":
                pts = [p for i in deleted for p in _cells(info["records"][i]["box"]) if p not in paint]
                if pts:
                    p = pts[0]
                    pair["output"][p] = pair["input"][p]
                    changed = True
            elif mode == "changed_survivor":
                remaining = [rec for i, rec in enumerate(info["records"]) if i not in deleted]
                if remaining:
                    p = _cells(remaining[0]["box"])[0]
                    pair["output"][p] = info["base"][p]
                    changed = True
            if changed:
                break
        assert changed
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Corruption accepted: {mode}")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, count boundaries, geometry, membership and mutations")
