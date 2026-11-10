"""Symmetry-certified sleds: repair outlines, then propagate shape contacts.

Standard library + NumPy only. No parent generator imports, stored examples,
seed-dependent solving, or mutable puzzle convention.
"""
from collections import deque
import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(np.all((g >= 0) & (g <= 9)), "invalid colors")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _neighbors(cells, offsets=EIGHT):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in offsets}


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, todo = {start}, [start]
        while todo:
            p = todo.pop()
            near = {(p[0] + dr, p[1] + dc) for dr, dc in EIGHT} & remaining
            remaining -= near
            part |= near
            todo.extend(sorted(near))
        result.append(part)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs), min(cs), max(cs)


def _boundary(body):
    return {p for p in body if any((p[0] + dr, p[1] + dc) not in body for dr, dc in FOUR)}


def _enclosed(frame):
    r0, r1, c0, c1 = _box(frame)
    domain = {(r, c) for r in range(r0 - 1, r1 + 2) for c in range(c0 - 1, c1 + 2)}
    outside, todo = {(r0 - 1, c0 - 1)}, deque([(r0 - 1, c0 - 1)])
    while todo:
        r, c = todo.popleft()
        for dr, dc in FOUR:
            p = r + dr, c + dc
            if p in domain and p not in frame and p not in outside:
                outside.add(p)
                todo.append(p)
    return domain - outside


def _axis(cells, width):
    scored = [(len({(r, c) for r, c in cells if (r, a - c) in cells}), a)
              for a in range(2 * width - 1)]
    best = max(s for s, _ in scored)
    winners = [a for s, a in scored if s == best]
    _require(len(winners) == 1, "ambiguous frame reflection axis")
    a = winners[0]
    return a, {p for p in cells if (p[0], a - p[1]) in cells}


def _canonical(g):
    horizontal = (np.all(g[:, 0] == g[0, 0]) and np.all(g[:, -1] == g[0, -1])
                  and g[0, 0] != g[0, -1])
    vertical = (np.all(g[0] == g[0, 0]) and np.all(g[-1] == g[-1, 0])
                and g[0, 0] != g[-1, 0])
    _require(horizontal != vertical, "expected one pair of opposite colored rails")
    return (g if horizontal else g.T.copy()), not horizontal


def _rows(body):
    result = {}
    for r, c in body:
        if r not in result:
            result[r] = [c, c]
        else:
            result[r][0] = min(result[r][0], c)
            result[r][1] = max(result[r][1], c)
    return result


def _decode(input_grid):
    g, transposed = _canonical(_grid(input_grid))
    h, w = g.shape
    low, high = int(g[0, 0]), int(g[0, -1])
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] * 2 > g.size and bg not in (low, high), "background must be a third-color majority")
    objects, accounted = [], {(r, c) for r in range(h) for c in (0, w - 1)}
    for color in (low, high):
        all_frame = {(int(r), int(c)) for r, c in np.argwhere(g == color) if 0 < c < w - 1}
        for raw in _components(all_frame):
            a, frame = _axis(raw, w)
            body = _enclosed(frame)
            _require(body - frame and _boundary(body) == frame, "repaired frame must close a body")
            rows = _rows(body)
            _require(len(rows) == max(rows) - min(rows) + 1, "body has missing rows")
            for r, (l, u) in rows.items():
                _require({(r, c) for c in range(l, u + 1)} <= body, "body must be row convex")
                _require(l + u == a, "body must be mirror symmetric")
            _require(all(0 <= r < h and 0 < c < w - 1 for r, c in raw | body), "sled touches a rail or is cropped")
            _require(not (body & accounted), "overlapping bodies")
            _require(not ((raw - frame) & body), "noise must be exterior")
            interior = body - frame
            _require(all(int(g[p]) not in (low, high) for p in interior), "rail-colored cargo is ambiguous")
            _require(any(g[p] != bg for p in interior), "each sled must carry visible cargo")
            pixels = {p: color if p in frame else int(g[p]) for p in body}
            objects.append(dict(body=body, frame=frame, raw=raw, rows=rows,
                                pixels=pixels, axis=a, color=color, direction=int(color == high)))
            accounted |= body | raw
    foreground = set(map(tuple, np.argwhere(g != bg).tolist()))
    _require(objects and foreground <= accounted, "unassigned foreground pixels")
    return g, transposed, bg, (low, high), objects


def _row_order(objects):
    """No contradictory initial ordering on overlapping row spans."""
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            if a["direction"] != b["direction"]:
                continue
            shared = a["rows"].keys() & b["rows"].keys()
            if shared:
                left, right = (a, b) if a["axis"] < b["axis"] else (b, a)
                _require(left["axis"] != right["axis"] and all(
                    left["rows"][r][1] < right["rows"][r][0] for r in shared),
                    "same-class order must be consistent on shared rows")


def _settle(objects, width, mode="shape"):
    """Boundary recurrence; alternatives are used only for witness checks."""
    _row_order(objects)
    profiles = []
    for o in objects:
        if mode == "raw":
            profiles.append(_rows(o["body"] | o["raw"]))
        elif mode == "box":
            r0, r1, c0, c1 = _box(o["body"])
            profiles.append({r: [c0, c1] for r in range(r0, r1 + 1)})
        else:
            profiles.append(o["rows"])
    offsets, depths = [0] * len(objects), [0] * len(objects)
    for direction in (0, 1):
        indices = [i for i, o in enumerate(objects) if o["direction"] == direction]
        indices.sort(key=lambda i: ((1 if direction == 0 else -1) * objects[i]["axis"],
                                    min(objects[i]["rows"])))
        prior = []
        for i in indices:
            rows = profiles[i]
            if direction == 0:
                d = 1 - min(v[0] for v in rows.values())
            else:
                d = width - 2 - max(v[1] for v in rows.values())
            depth = 1
            for j in prior:
                shared = rows.keys() & profiles[j].keys()
                if shared:
                    depth = max(depth, depths[j] + 1)
                    if direction == 0:
                        d = max(d, max(profiles[j][r][1] + offsets[j] - rows[r][0] + 1 for r in shared))
                    else:
                        d = min(d, min(profiles[j][r][0] + offsets[j] - rows[r][1] - 1 for r in shared))
            offsets[i], depths[i] = d, depth
            prior.append(i)
    return offsets, depths


def _translated(body, d):
    return {(r, c + d) for r, c in body}


def _render(shape, bg, rails, objects, offsets, raw=False):
    h, w = shape
    grid = np.full(shape, bg, dtype=np.int8)
    grid[:, 0], grid[:, -1] = rails
    covered = set()
    for o, d in zip(objects, offsets):
        cells = _translated(o["body"], d)
        _require(all(0 <= r < h and 0 < c < w - 1 for r, c in cells), "packed sled outside rails")
        _require(not cells & covered, "final opposite classes overlap")
        covered |= cells
        for (r, c), value in o["pixels"].items():
            grid[r, c + d] = value
        if raw:
            for r, c in o["raw"] - o["frame"]:
                _require(0 < c + d < w - 1, "noise on rail")
                _require((r, c + d) not in covered, "noise overlaps another body")
                grid[r, c + d] = o["color"]
                covered.add((r, c + d))
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    g, transposed, bg, rails, objects = _decode(input_grid)
    offsets, _ = _settle(objects, g.shape[1])
    _require(all(d <= 0 if o["direction"] == 0 else d >= 0 for o, d in zip(objects, offsets)),
             "packing would move away from a destination")
    output = _render(g.shape, bg, rails, objects, offsets)
    return output.T.copy() if transposed else output


def _unit_offsets(objects, shape):
    """Separate occupancy simulation: single-cell moves cannot jump blockers."""
    offsets = [0] * len(objects)
    for direction, step in ((0, -1), (1, 1)):
        group = [i for i, o in enumerate(objects) if o["direction"] == direction]
        occupied = set().union(*(objects[i]["body"] for i in group)) if group else set()
        for _ in range(shape[1] * max(1, len(group)) + 1):
            moved = False
            for i in group:
                here = _translated(objects[i]["body"], offsets[i])
                occupied -= here
                candidate = _translated(objects[i]["body"], offsets[i] + step)
                if all(0 < c < shape[1] - 1 for _, c in candidate) and not candidate & occupied:
                    offsets[i] += step
                    here = candidate
                    moved = True
                occupied |= here
            if not moved:
                break
        else:
            raise AssertionError("unit-step simulation failed to converge")
    return offsets


def _repack_offsets(objects, width):
    """Competing edge-first final-location algorithm that can overtake."""
    offsets = [0] * len(objects)
    for direction in (0, 1):
        group = [i for i, o in enumerate(objects) if o["direction"] == direction]
        group.sort(key=lambda i: (1 if direction == 0 else -1) * objects[i]["axis"])
        placed = set()
        for i in group:
            _, _, c0, c1 = _box(objects[i]["body"])
            ds = range(1 - c0, width - 1 - c1)
            if direction:
                ds = reversed(ds)
            for d in ds:
                body = _translated(objects[i]["body"], d)
                if not body & placed:
                    offsets[i] = d
                    placed |= body
                    break
            else:
                raise ValueError("repacking has no free location")
    return offsets


def _profile(rng, h, w, style=None):
    limit = (w - 2) // 2
    mode = int(rng.integers(5)) if style is None else style
    ds = []
    d = int(rng.integers(limit + 1))
    for r in range(h):
        if mode == 0:
            d = 0
        elif mode == 1:
            d = min(limit, r, h - 1 - r)
        elif mode == 2:
            d = min(limit, abs(r - h // 2))
        elif mode == 3:
            d = int(np.clip(d + rng.integers(-1, 2), 0, limit))
        else:
            d = min(limit, r)
        ds.append(d)
    # Attain the declared width, and retain a genuinely enclosed cargo pixel.
    j = int(rng.integers(h))
    ds[j] = 0
    for r in range(j - 1, -1, -1):
        ds[r] = min(ds[r], ds[r + 1] + 1)
    for r in range(j + 1, h):
        ds[r] = min(ds[r], ds[r - 1] + 1)
    if w == 3:
        ds = [0] * h
    return ds


def _local_body(h, w, ds):
    return {(r, c) for r in range(h) for c in range(ds[r], w - ds[r])}


def _make_object(rng, r, c, h, w, direction, rails, bg, ds=None, cargo=None):
    ds = _profile(rng, h, w) if ds is None else ds
    local = _local_body(h, w, ds)
    local_frame = _boundary(local)
    inside = sorted(local - local_frame)
    _require(inside and _enclosed(local_frame) == local, "profile has no enclosed cargo")
    palette = [x for x in range(10) if x not in (*rails, bg)]
    if cargo is None:
        colors = list(map(int, rng.choice(palette, int(rng.integers(1, 4)), replace=False)))
        mode = int(rng.integers(4))
        cargo = {}
        for rr, cc in inside:
            k = 0 if mode == 0 else rr if mode == 1 else rr + cc if mode == 2 else int(rng.integers(len(colors)))
            cargo[rr, cc] = colors[k % len(colors)]
            if rng.random() < .13:
                cargo[rr, cc] = bg
        cargo[inside[0]] = colors[0]
        # Where possible, show that cargo symmetry is irrelevant.
        off_axis = [p for p in inside if (p[0], w - 1 - p[1]) in inside and 2 * p[1] != w - 1]
        if off_axis:
            p = off_axis[0]
            cargo[p] = palette[0]
            cargo[p[0], w - 1 - p[1]] = palette[1]
    body = {(rr + r, cc + c) for rr, cc in local}
    frame = {(rr + r, cc + c) for rr, cc in local_frame}
    color = rails[direction]
    pixels = {(rr + r, cc + c): color if (rr, cc) in local_frame else cargo[rr, cc] for rr, cc in local}
    return dict(body=body, frame=frame, raw=set(frame), rows=_rows(body), pixels=pixels,
                axis=2 * c + w - 1, color=color, direction=direction, cargo=cargo)


def _is_input_safe(objects, shape):
    occupied = set()
    for o in objects:
        if occupied & o["body"]:
            return False
        occupied |= o["body"]
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            if a["color"] == b["color"] and _neighbors(a["raw"]) & b["raw"]:
                return False
    try:
        ds, _ = _settle(objects, shape[1])
        if not all(d <= 0 if o["direction"] == 0 else d >= 0 for o, d in zip(objects, ds)):
            return False
        cells = set()
        for o, d in zip(objects, ds):
            final = _translated(o["body"], d)
            if final & cells or not all(0 < c < shape[1] - 1 for _, c in final):
                return False
            cells |= final
    except ValueError:
        return False
    return True


def _noise(rng, objects, shape, force=None):
    h, w = shape
    bodies = set().union(*(o["body"] for o in objects))
    for i, o in enumerate(objects):
        others = set().union(*(p["raw"] for j, p in enumerate(objects) if j != i and p["color"] == o["color"]))
        blocked = bodies | _neighbors(others) | others
        forced = force if i == 0 else None
        budget = int(rng.integers(0, max(2, min(7, len(o["frame"]) // 3)) + 1))
        if forced is not None:
            budget = max(1, budget)
        for k in range(budget):
            candidates = sorted(p for p in _neighbors(o["raw"]) - blocked - o["raw"]
                                if 0 <= p[0] < h and 1 < p[1] < w - 2)
            if k == 0 and forced is not None:
                candidates = [forced] if forced in candidates else []
            else:
                rng.shuffle(candidates)
            for p in candidates:
                try:
                    a, supported = _axis(o["raw"] | {p}, w)
                    if a == o["axis"] and supported == o["frame"]:
                        o["raw"].add(p)
                        break
                except ValueError:
                    pass
            else:
                if k == 0 and forced is not None:
                    return False
                break
    return True


def _layout(rng, purpose, rails, bg, simple=False):
    if purpose == "nest":
        ha, hb = (5, 5) if simple else (int(rng.integers(4, 8)), int(rng.integers(4, 8)))
        wa, wb = (5, 4) if simple else (int(rng.choice((5, 7))), int(rng.choice((4, 6))))
        overlap = int(rng.integers(1, min(3, ha - 1, hb - 1)))
        step = ha - overlap
        h = int(rng.integers(ha + step + hb + 3, 31))
        w = int(rng.integers(wa + wb + 7, 31))
        direction = int(rng.integers(2))
        da = [0] * (ha - overlap - 1) + [1] * (overlap + 1)
        db = [1] * (overlap + 1) + [0] * (hb - overlap - 1)
        c0 = int(rng.integers(3, w - wa - wb - 2))
        c1 = c0 + wa + 2
        if direction:
            c0, c1 = w - c0 - wa, w - c1 - wb
        r0 = int(rng.integers(0, h - (ha + step + hb + 1) + 1))
        a = _make_object(rng, r0, c0, ha, wa, direction, rails, bg, da)
        b = _make_object(rng, r0 + step, c1, hb, wb, direction, rails, bg, db)
        r2 = r0 + step + hb + 1
        c2 = int(rng.integers(2, w - wa - 1))
        z = _make_object(rng, r2, c2, ha, wa, 1 - direction, rails, bg, da, a["cargo"])
        row = r0 + step
        forced = row, a["rows"][row][1] + 1 if direction == 0 else a["rows"][row][0] - 1
        return (h, w), [a, b, z], forced
    if purpose in ("chain", "test"):
        n = 3 if purpose == "chain" else 4
        hbody = 5 if simple else int(rng.integers(4, 7))
        step = 3 if simple else int(rng.integers((hbody + 1) // 2, hbody))
        ws = [4] * n if simple else list(map(int, rng.integers(3, 7, size=n)))
        minw = sum(ws) + n + 3
        if minw > 30:
            ws = [4] * n
            minw = sum(ws) + n + 3
        h = int(rng.integers(max(20 if simple else 10, hbody + (n - 1) * step + 1), 31))
        w = int(rng.integers(max(24 if simple else 14, minw), 31))
        direction = int(rng.integers(2))
        r0 = int(rng.integers(h - hbody - (n - 1) * step + 1))
        reverse = bool(rng.integers(2))
        c = int(rng.integers(2, w - sum(ws) - n))
        objects = []
        for i, wi in enumerate(ws):
            ri = r0 + (n - 1 - i if reverse else i) * step
            ci = c if direction == 0 else w - c - wi
            objects.append(_make_object(rng, ri, ci, hbody, wi, direction, rails, bg,
                                        [0] * hbody if simple else None))
            c += wi + 1
        # Extra opposite-class cargo can share independent bands. It does not
        # change the queried four-link dependency depth.
        if purpose == "test" and not simple and rng.random() < .65:
            spans = [(0, r0), (r0 + hbody + (n - 1) * step, h)]
            for lo, hi in spans:
                if hi - lo < 5 or rng.random() < .35:
                    continue
                hh, ww = int(rng.integers(3, min(7, hi - lo - 1) + 1)), int(rng.integers(3, 8))
                rr = int(rng.integers(lo, hi - hh + 1))
                cc = int(rng.integers(2, w - ww - 1))
                try:
                    o = _make_object(rng, rr, cc, hh, ww, 1 - direction, rails, bg)
                    if _is_input_safe(objects + [o], (h, w)):
                        objects.append(o)
                except ValueError:
                    pass
        return (h, w), objects, None
    h, w = int(rng.integers(10, 31)), int(rng.integers(14, 31))
    objects = []
    requested = int(rng.integers(1, 7))
    for _ in range(requested):
        for _ in range(35):
            hh, ww = int(rng.integers(3, min(9, h) + 1)), int(rng.integers(3, min(8, w - 4) + 1))
            rr, cc = int(rng.integers(h - hh + 1)), int(rng.integers(2, w - ww - 1))
            try:
                o = _make_object(rng, rr, cc, hh, ww, int(rng.integers(2)), rails, bg)
            except ValueError:
                continue
            proposed = objects + [o]
            if _is_input_safe(proposed, (h, w)) and max(_settle(proposed, w)[1]) <= 3:
                objects = proposed
                break
    _require(objects, "could not place a free sled")
    return (h, w), objects, None


def _features(g):
    canonical, transposed, bg, rails, objects = _decode(g)
    ds, depths = _settle(objects, canonical.shape[1])
    raw, _ = _settle(objects, canonical.shape[1], "raw")
    boxes, _ = _settle(objects, canonical.shape[1], "box")
    repack = _repack_offsets(objects, canonical.shape[1])
    causal = False
    signatures = []
    asymmetric_cargo = False
    for i, o in enumerate(objects):
        if o["raw"] != o["frame"]:
            one_raw = [dict(p, raw=p["raw"] if i == j else p["frame"]) for j, p in enumerate(objects)]
            changed, _ = _settle(one_raw, canonical.shape[1], "raw")
            causal |= any(changed[j] != ds[j] for j in range(len(objects)) if j != i)
        r0, _, c0, _ = _box(o["body"])
        signature = (tuple(sorted((r - r0, c - c0) for r, c in o["body"])),
                     tuple(sorted((r - r0, c - c0, o["pixels"][r, c]) for r, c in o["body"] - o["frame"])))
        signatures.append(signature)
        asymmetric_cargo |= any(o["pixels"][r, c] != o["pixels"][r, o["axis"] - c]
                                for r, c in o["body"] - o["frame"])
    shared_cargo = any(signatures[i] == signatures[j] and a["direction"] != b["direction"]
                       for i, a in enumerate(objects) for j, b in enumerate(objects) if i < j)
    return dict(orientation=int(transposed), shape=list(g.shape), background=bg,
                rails=list(rails), count=len(objects), depth=max(depths),
                parities=sorted({o["axis"] % 2 for o in objects}),
                noise=sum(len(o["raw"] - o["frame"]) for o in objects),
                noise_changes_positions=raw != ds, box_changes_positions=boxes != ds,
                noise_affects_follower=causal, shared_cargo=shared_cargo,
                asymmetric_cargo=asymmetric_cargo,
                repack_changes_positions=repack != ds,
                moved_classes=sorted({o["direction"] for o, d in zip(objects, ds) if d}),
                footprints=sorted({(_box(o["body"])[1] - _box(o["body"])[0] + 1,
                                    _box(o["body"])[3] - _box(o["body"])[2] + 1) for o in objects}),
                travels=[abs(d) for d in ds])


def _pair(rng, purpose, transposed):
    for attempt in range(120):
        colors = list(map(int, rng.choice(10, 3, replace=False)))
        rails, bg = tuple(colors[:2]), colors[2]
        try:
            shape, objects, force = _layout(rng, purpose, rails, bg, simple=attempt >= 90)
            if not _is_input_safe(objects, shape) or not _noise(rng, objects, shape, force):
                continue
            if not _is_input_safe(objects, shape):
                continue
            ds, _ = _settle(objects, shape[1])
            inp = _render(shape, bg, rails, objects, [0] * len(objects), raw=True)
            out = _render(shape, bg, rails, objects, ds)
            if np.count_nonzero(inp == bg) * 2 <= inp.size:
                continue
            decoded = _decode(inp)[4]
            _require(len(decoded) == len(objects), "frame segmentation changed")
            recovered = {frozenset(o["body"]) for o in decoded}
            _require(recovered == {frozenset(o["body"]) for o in objects}, "repair changed intended body")
            f = _features(inp)
            if purpose == "nest":
                if not (f["box_changes_positions"] and f["noise_changes_positions"]
                        and f["moved_classes"] == [0, 1] and f["parities"] == [0, 1]):
                    continue
                # The leading object's compulsory spur must affect its follower.
                clean_ds, _ = _settle(objects, shape[1])
                raw_ds, _ = _settle(objects, shape[1], "raw")
                if raw_ds[1] == clean_ds[1]:
                    continue
            if purpose == "chain" and not (f["repack_changes_positions"] and f["noise"] > 0):
                continue
            if purpose == "test" and f["depth"] != 4:
                continue
            if transposed:
                inp, out = inp.T.copy(), out.T.copy()
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError(f"bounded generation failed for {purpose}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    transposed = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            purpose = "test" if split == "test" else "nest" if i == 0 else "chain" if i == 1 else "free"
            for _ in range(20):
                pair = _pair(rng, purpose, transposed)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            g, transposed, bg, rails, objects = _decode(inp)
            ds, _ = _settle(objects, g.shape[1])
            assert ds == _unit_offsets(objects, g.shape), "contact recurrence differs from physical sliding"
            assert np.array_equal(solve(inp), out), "incorrect label"
            actual = out.T if transposed else out
            assert np.array_equal(actual[:, 0], g[:, 0]) and np.array_equal(actual[:, -1], g[:, -1])
            for o, d in zip(objects, ds):
                assert all(actual[r, c + d] == v for (r, c), v in o["pixels"].items()), "cargo/frame changed"
            features[split].append(_features(inp))
    assert len({f["orientation"] for s in features.values() for f in s}) == 1
    training = features["train"]
    assert any(f["noise_changes_positions"] for f in training), "A must change a packing position"
    assert any(f["noise_affects_follower"] for f in training), "A must change another sled's destination"
    assert any(f["box_changes_positions"] for f in training), "shape contact must be taught"
    assert any(f["repack_changes_positions"] for f in training), "stable order must be taught"
    assert set().union(*(set(f["moved_classes"]) for f in training)) == {0, 1}
    assert set().union(*(set(f["parities"]) for f in training)) == {0, 1}
    assert any(f["shared_cargo"] for f in training), "frame instruction must be separated from cargo/size"
    assert any(f["asymmetric_cargo"] for f in training), "cargo is carried without symmetry filtering"
    assert all(f["depth"] <= 3 for f in training)
    assert all(f["depth"] == 4 for f in features["test"])
    assert any(f["noise"] for f in training)
    return None


def self_test():
    records, fingerprints = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]))
        for split in ("train", "test"):
            for pair in p[split]:
                f = _features(pair["input"])
                f["split"] = split
                records.append(f)
        if seed < 2:
            q = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in ("input", "output"))
    assert len(fingerprints) == 42
    assert {f["orientation"] for f in records} == {0, 1}
    assert len({tuple(f["shape"]) for f in records}) >= 60
    assert {f["background"] for f in records} == set(range(10))
    assert len({tuple(f["rails"]) for f in records}) >= 40
    assert {1, 2, 3, 4} <= {f["count"] for f in records}
    assert len({tuple(v) for f in records for v in f["footprints"]}) >= 15
    assert any(f["noise"] == 0 for f in records)
    assert all(any(f[name] for f in records) for name in ("noise_changes_positions", "box_changes_positions", "repack_changes_positions"))
    for counts in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(8301, *counts))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    p = generate_puzzle(1901, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds_checked": 42, "pairs_checked": len(records),
            "canvas_shapes": len({tuple(f["shape"]) for f in records}),
            "rail_assignments": len({tuple(f["rails"]) for f in records}),
            "object_counts": sorted({f["count"] for f in records}),
            "footprints": len({tuple(v) for f in records for v in f["footprints"]}),
            "noise_position_witnesses": sum(f["noise_changes_positions"] for f in records),
            "noise_follower_witnesses": sum(f["noise_affects_follower"] for f in records),
            "shape_contact_witnesses": sum(f["box_changes_positions"] for f in records),
            "stable_order_witnesses": sum(f["repack_changes_positions"] for f in records)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
