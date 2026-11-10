"""Phase-gated plates: local periodic apertures control whole-object deletion.

Only NumPy and the standard library are used. Generation paints truth from
chosen geometry; the public solver reads the plate grammar from input cells.
No file access, parent imports, stored examples, or seed-dependent solver state.
"""
from __future__ import annotations

import numpy as np


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _reflect(a, corner):
    if corner & 2:
        a = a[::-1, :]
    if corner & 1:
        a = a[:, ::-1]
    return a


def _decode_rail(values, frame):
    locations = np.flatnonzero(values != frame)
    counts = {int(c): int(np.count_nonzero(values == c))
              for c in np.unique(values[locations])}
    repeated = [c for c, n in counts.items() if n > 1]
    if len(repeated) != 1 or counts[repeated[0]] != 2:
        raise ValueError("a calibration rail needs one unique repeated pair")
    ref = repeated[0]
    a, b = map(int, np.flatnonzero(values == ref))
    d = b - a
    extras = [int(x) for x in locations if int(x) not in (a, b)]
    aligned = [x for x in extras if (x-a) % d == 0]
    if len(aligned) > 1:
        raise ValueError("multiple on-phase terminals")
    t = aligned[0] if aligned else None
    if t is not None and a < t < b:
        raise ValueError("a terminal must lie beyond the pair")
    full = tuple(x for x in range(len(values)) if (x-a) % d == 0)
    lo, hi = (0, len(values)-1) if t is None else (min(a, t), max(b, t))
    active = tuple(x for x in full if lo <= x <= hi)
    return {"a": a, "b": b, "period": d, "terminal": t,
            "reference": ref, "full": full, "active": active,
            "noise": tuple(x for x in extras if x != t)}


def _analyze(input_grid):
    """Read local roles, segmented plates, and ruler geometry from cells."""
    grid = _grid(input_grid)
    if min(grid.shape) < 4:
        raise ValueError("no room for the selector")
    candidates = []
    for corner in range(4):
        v = _reflect(grid, corner)
        frame = int(v[3, 3])
        if not (np.all(v[3, :4] == frame) and np.all(v[:4, 3] == frame)):
            continue
        colors, counts = np.unique(v[:3, :3], return_counts=True)
        padding = colors[counts == 6]
        if len(padding) != 1 or len(colors) != 4:
            continue
        bg = int(padding[0])
        q = frozenset(int(c) for c in colors if c != bg)
        if frame != bg and frame not in q and len(q) == 3:
            candidates.append((corner, bg, frame, q))
    if len(candidates) != 1:
        raise ValueError("selector must be unique")
    corner, bg, frame, q = candidates[0]
    view = _reflect(grid, corner)
    mask = (view != bg).copy()
    mask[:4, :4] = False
    remaining = set(zip(*np.nonzero(mask)))
    boxes = [(0, 4, 0, 4)]
    plates = []
    filler = None
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        points = [start]
        for r, c in points:
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    points.append(point)
        rows, cols = zip(*points)
        box = tuple(map(int, (min(rows), max(rows)+1, min(cols), max(cols)+1)))
        r0, r1, c0, c1 = box
        tile = view[r0:r1, c0:c1]
        if len(points) != tile.size or min(tile.shape) < 5:
            raise ValueError("plates must be solid rectangles with interior >=3")
        if not all(r1 < a or b < r0 or c1 < c or d < c0 for a, b, c, d in boxes):
            raise ValueError("plate bounding boxes need a background gap")
        boxes.append(box)
        if not np.all(tile[np.ix_([0, tile.shape[0]-1], [0, tile.shape[1]-1])] == frame):
            raise ValueError("outer corner pads must be frame-colored")
        field = tile[1:-1, 1:-1]
        pads = field[np.ix_([0, field.shape[0]-1], [0, field.shape[1]-1])]
        local_filler = int(pads[0, 0])
        if (not np.all(pads == local_filler) or local_filler in q | {bg, frame}
                or np.any((field == bg) | (field == frame))):
            raise ValueError("interior filler pads or labels are invalid")
        if filler is None:
            filler = local_filler
        if filler != local_filler:
            raise ValueError("plates must share one visible filler role")
        horizontal = [(0, tile[0, 1:-1]), (1, tile[-1, 1:-1])]
        vertical = [(0, tile[1:-1, 0]), (1, tile[1:-1, -1])]
        hs = [(side, rail) for side, rail in horizontal if np.any(rail != frame)]
        vs = [(side, rail) for side, rail in vertical if np.any(rail != frame)]
        if len(hs) != 1 or len(vs) != 1:
            raise ValueError("one horizontal and one vertical calibration rail required")
        if any(np.any((rail == bg) | (rail == filler)) for _, rail in hs + vs):
            raise ValueError("ruler ticks must be label identities")
        x = _decode_rail(hs[0][1], frame)
        y = _decode_rail(vs[0][1], frame)
        sample = field[np.ix_(y["active"], x["active"])]
        labels = frozenset(int(c) for c in np.unique(sample) if c != filler)
        plates.append({"box": box, "field": field, "x": x, "y": y,
                       "rail_corner": 2*vs[0][0] + hs[0][0],
                       "sample_labels": labels, "positive": q <= labels})
    if not plates:
        raise ValueError("no plates")
    return {"corner": corner, "background": bg, "frame": frame,
            "filler": filler, "query": q, "plates": plates, "view": view}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase qualifying calibrated plates, deriving everything from input."""
    info = _analyze(input_grid)
    out = np.asarray(input_grid).copy()
    view = _reflect(out, info["corner"])
    for plate in info["plates"]:
        if plate["positive"]:
            r0, r1, c0, c1 = plate["box"]
            view[r0:r1, c0:c1] = info["background"]
    return out


def _cuts(box, minimum):
    r0, r1, c0, c1 = box
    for gap in (1, 2):
        for cut in range(r0+minimum, r1-gap-minimum+1):
            yield ((r0, cut, c0, c1), (cut+gap, r1, c0, c1))
        for cut in range(c0+minimum, c1-gap-minimum+1):
            yield ((r0, r1, c0, cut), (r0, r1, cut+gap, c1))


def _capacity(box, minimum):
    a, b, c, d = box
    return ((b-a+1)//(minimum+1))*((d-c+1)//(minimum+1))


def _pack(rng, count, minimum, teaching, bias):
    """Random guillotine packing around a variable-sized corner reservation."""
    low = 21 if count >= 4 else 16
    first_h, first_w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    if rng.random() < .2:
        first_w = first_h
    leaves = None
    for attempt in range(50):
        h, w = min(30, first_h+attempt//5), min(30, first_w+attempt//5)
        # The selector's reserved region is irregular in size. Both L-shaped
        # complements are available; no class receives a reserved object slot.
        options = []
        for cut_r in range(4, h-minimum):
            for cut_c in range(4, w-minimum):
                for vertical in (False, True):
                    candidates = ([(0, h, cut_c+1, w), (cut_r+1, h, 0, cut_c)]
                                  if vertical else
                                  [(0, cut_r, cut_c+1, w), (cut_r+1, h, 0, w)])
                    candidates = [b for b in candidates
                                  if min(b[1]-b[0], b[3]-b[2]) >= minimum]
                    if candidates and sum(_capacity(b, minimum) for b in candidates) >= count:
                        options.append(candidates)
        if not options:
            continue
        chosen = [b for b in _pick(rng, options)]
        if count == 1:
            chosen = [_pick(rng, chosen)]
        while len(chosen) < count:
            alternatives = [(i, cut) for i, b in enumerate(chosen)
                            for cut in _cuts(b, minimum)
                            if sum(_capacity(x, minimum) for x in cut)
                            + sum(_capacity(x, minimum) for j, x in enumerate(chosen) if j != i) >= count]
            if not alternatives:
                break
            i, cut = _pick(rng, alternatives)
            chosen.pop(i)
            chosen.extend(cut)
        if len(chosen) == count:
            leaves = chosen
            break
    if leaves is None:
        # A 30x30 reservation admits at least eight minimum-nine plates.
        # Capacity-preserving cuts give a finite fallback, not lucky retries.
        h = w = 30
        leaves = [(0, minimum, 5, 30), (minimum+1, 30, 0, 30)]
        if count == 1:
            leaves = leaves[:1]
        while len(leaves) < count:
            possibilities = [(i, cut) for i, b in enumerate(leaves) for cut in _cuts(b, minimum)]
            if not possibilities:
                raise RuntimeError("packing capacity exhausted")
            i, cut = max(possibilities, key=lambda item:
                         (sum(_capacity(b, minimum) for b in item[1]),
                          min((b[1]-b[0])*(b[3]-b[2]) for b in item[1])))
            leaves.pop(i)
            leaves.extend(cut)
    rng.shuffle(leaves)
    boxes = []
    for r0, r1, c0, c1 in leaves:
        max_h, max_w = r1-r0, c1-c0
        if teaching:
            max_h, max_w = min(13, max_h), min(13, max_w)
        options = [(a, b) for a in range(minimum, max_h+1)
                   for b in range(minimum, max_w+1)]
        kind = _pick(rng, ("compact", "full", "large", "square", bias, bias))
        if kind == "full":
            options = [(max_h, max_w)]
        elif kind == "large":
            options = [(a, b) for a, b in options if a*b >= .65*max_h*max_w]
        elif kind == "compact":
            options = [(a, b) for a, b in options if a+b <= 2*minimum+3]
        elif kind == "square":
            delta = min(abs(a-b) for a, b in options)
            options = [(a, b) for a, b in options if abs(a-b) == delta]
        elif kind in ("wide", "tall"):
            target = max((b/a if kind == "wide" else a/b) for a, b in options)
            options = [(a, b) for a, b in options
                       if (b/a if kind == "wide" else a/b) >= target*.8]
        oh, ow = _pick(rng, options)
        rr = r0 + int(_pick(rng, (0, r1-r0-oh, int(rng.integers(r1-r0-oh+1)))))
        cc = c0 + int(_pick(rng, (0, c1-c0-ow, int(rng.integers(c1-c0-ow+1)))))
        boxes.append((rr, rr+oh, cc, cc+ow))
    return h, w, boxes


def _axis_options(length, period, terminal, needs_stop=False, both_sides=False):
    result = []
    for a in range(length-period):
        b = a + period
        full = tuple(range(a % period, length, period))
        if len(full) < 3:
            continue
        if not terminal:
            if not both_sides or (full[0] < a and full[-1] > b):
                result.append({"a": a, "b": b, "period": period, "terminal": None,
                               "full": full, "active": full})
            continue
        for t in full:
            if a <= t <= b:
                continue
            lo, hi = min(a, t), max(b, t)
            active = tuple(x for x in full if lo <= x <= hi)
            if len(active) >= 3 and (not needs_stop or len(active) < len(full)):
                result.append({"a": a, "b": b, "period": period, "terminal": t,
                               "full": full, "active": active})
    return result


def _geometry(rng, length, kind, teaching):
    max_period = min(3 if teaching else 10, (length-1)//2)
    if kind == "dense":
        periods, terminal = [1], False
    elif kind == "boundary":
        periods, terminal = list(range(2, max_period+1)), False
    elif kind == "phase":
        periods, terminal = list(range(2, max_period+1)), False
    elif kind == "offset":
        periods = list(range(2, max_period+1))
        terminal = bool(rng.integers(2))
    else:
        periods = list(range(1, max_period+1))
        terminal = True if kind == "stop" else bool(rng.integers(2))
    possibilities = [(p, _axis_options(length, p, terminal, kind == "stop", kind == "boundary"))
                     for p in periods]
    if kind == "offset":
        possibilities = [(p, [v for v in values if v["a"] % p != 0]) for p, values in possibilities]
    possibilities = [(p, values) for p, values in possibilities if values]
    if not possibilities:
        raise RuntimeError("ruler geometry infeasible")
    _, options = _pick(rng, possibilities)
    return dict(_pick(rng, options))


def _rail(rng, length, geometry, frame, labels, force_noise=False, reference=None):
    rail = np.full(length, frame, dtype=np.int8)
    reference = int(_pick(rng, labels)) if reference is None else reference
    rail[[geometry["a"], geometry["b"]]] = reference
    available = [c for c in labels if c != reference]
    rng.shuffle(available)
    t = geometry["terminal"]
    if t is not None:
        rail[t] = available.pop()
    off = [x for x in range(length) if (x-geometry["a"]) % geometry["period"]]
    n = int(rng.integers(min(2, len(off), len(available))+1)) if rng.random() < .55 else 0
    if force_noise:
        n = max(n, 1)
    if n > min(len(off), len(available)):
        raise RuntimeError("no off-phase tick position")
    for x, c in zip(rng.choice(off, n, replace=False), available[:n]):
        rail[int(x)] = c
    return rail


def _record(rng, height, width, roles, query, labels, kind, rail_corner, teaching):
    bg, frame, filler = roles
    h, w = height-2, width-2
    row_kind = "phase" if kind == "phase_row" else "stop" if kind == "stop_row" else "random"
    col_kind = "phase" if kind == "phase_col" else "stop" if kind == "stop_col" else "random"
    if kind == "boundary_positive":
        row_kind, col_kind = "dense", "boundary"
    elif kind == "extra_positive":
        row_kind = "dense"  # Enough sampled room for repetition and extras.
        col_kind = "offset"  # Teach phase, rather than an origin-zero lattice.
    elif kind == "cross_negative":
        row_kind = str(_pick(rng, ("phase", "stop")))
        col_kind = str(_pick(rng, ("phase", "stop")))
    y = _geometry(rng, h, row_kind, teaching)
    x = _geometry(rng, w, col_kind, teaching)
    pads = {(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)}
    sample = [(r, c) for r in y["active"] for c in x["active"] if (r, c) not in pads]
    marks = {}
    q = list(query)
    extras = [c for c in labels if c not in q]

    def put(color, positions):
        eligible = [p for p in positions if p not in marks and p not in pads]
        if not eligible:
            raise RuntimeError("no available label position")
        point = _pick(rng, eligible)
        marks[point] = int(color)

    if kind == "boundary_positive":
        put(q[0], [(r, c) for r, c in sample if c < x["a"]])
        put(q[1], [(r, c) for r, c in sample if c > x["b"]])
        put(q[2], sample)
    elif kind in ("phase_row", "phase_col", "stop_row", "stop_col"):
        missing = 0 if kind.endswith("row") else 1
        for i in range(3):
            if i != missing:
                put(q[i], sample)
        if kind == "phase_row":
            positions = [(r, c) for r in range(h) if r not in y["full"] for c in x["active"]]
        elif kind == "phase_col":
            positions = [(r, c) for r in y["active"] for c in range(w) if c not in x["full"]]
        elif kind == "stop_row":
            positions = [(r, c) for r in y["full"] if r not in y["active"] for c in x["active"]]
        else:
            positions = [(r, c) for r in y["active"] for c in x["full"] if c not in x["active"]]
        put(q[missing], positions)
    elif kind == "cross_negative":
        # The held-out structural recombination: two different identities fail
        # the two independently demonstrated apertures inside the same plate.
        row_out = ([r for r in range(h) if r not in y["full"]] if row_kind == "phase"
                   else [r for r in y["full"] if r not in y["active"]])
        col_out = ([c for c in range(w) if c not in x["full"]] if col_kind == "phase"
                   else [c for c in x["full"] if c not in x["active"]])
        put(q[0], [(r, c) for r in row_out for c in x["active"]])
        put(q[1], [(r, c) for r in y["active"] for c in col_out])
        put(q[2], sample)
    elif kind == "identity_negative":
        for c in (q[0], q[1], extras[0]):
            put(c, sample)
    elif kind == "extra_positive":
        for c in q + [extras[0], q[0], extras[0]]:
            put(c, sample)
        for c in extras[1:]:
            if rng.random() < .5:
                put(c, sample)
        # Unsampled content, when space permits, is erased with the object.
        sample_sites = set(sample)
        other = [(r, c) for r in range(h) for c in range(w) if (r, c) not in sample_sites]
        other = [p for p in other if p not in pads]
        if other:
            put(_pick(rng, labels), other)
    elif kind in ("positive", "negative"):
        if kind == "positive":
            extra_limit = min(len(extras), len(sample)-3)
            used = q + list(rng.permutation(extras)[:int(rng.integers(extra_limit+1))])
            for c in used:
                put(c, sample)
        else:
            used = list(rng.permutation(q)[:int(rng.integers(3))])
            used += list(rng.permutation(extras)[:int(rng.integers(len(extras)+1))])
        all_sites = [(r, c) for r in range(h) for c in range(w) if (r, c) not in pads]
        if used:
            target = int(rng.integers(max(len(marks), len(used)),
                                      max(max(len(marks), len(used)), int(len(all_sites)*rng.uniform(.08, .72)))+1))
            for c in used:
                if c not in marks.values():
                    put(c, all_sites)
            while len(marks) < target:
                put(_pick(rng, used), all_sites)
    elif kind != "empty":
        raise ValueError("unknown construction kind")
    tile = np.full((height, width), frame, dtype=np.int8)
    tile[1:-1, 1:-1] = filler
    for (r, c), color in marks.items():
        tile[r+1, c+1] = color
    horizontal = _rail(rng, w, x, frame, labels, kind == "boundary_positive",
                       q[2] if kind == "identity_negative" else None)
    vertical = _rail(rng, h, y, frame, labels)
    # Independent side choice; coordinates still align with the interior.
    tile[-1 if rail_corner & 1 else 0, 1:-1] = horizontal
    tile[1:-1, -1 if rail_corner & 2 else 0] = vertical
    observed = {color for point, color in marks.items()
                if point[0] in y["active"] and point[1] in x["active"]}
    return tile, set(q) <= observed


def _scene(rng, colors, query, corner, package, bias):
    roles, labels = colors[:3], colors[3:]
    bg, frame, _ = roles
    teaching = package in ("identity", "bounds")
    if package == "identity":
        kinds = ["boundary_positive", "phase_row", "phase_col", "identity_negative"]
    elif package == "bounds":
        kinds = ["extra_positive", "stop_row", "stop_col", "empty"]
    elif package == "test":
        kinds = [str(_pick(rng, ("positive", "extra_positive"))),
                 "cross_negative"]
    else:
        kinds = []
    if teaching:
        if rng.random() < .4:
            kinds.append(str(_pick(rng, ("positive", "negative", "empty"))))
    else:
        count = int(rng.integers(2 if package == "test" else 1, 7))
        mode = str(_pick(rng, ("mixed", "mixed", "all", "none")))
        while len(kinds) < count:
            kinds.append("positive" if mode == "all" or (mode == "mixed" and rng.random() < .5)
                         else "negative")
    rng.shuffle(kinds)
    h, w, boxes = _pack(rng, len(kinds), 9 if teaching or package == "test" else 7, teaching, bias)
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[3, :4] = frame
    grid[:4, 3] = frame
    for position, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
        grid[int(position)//3, int(position)%3] = color
    out = grid.copy()
    corners = list(map(int, rng.permutation(4)))
    corners += [int(rng.integers(4)) for _ in range(max(0, len(kinds)-4))]
    for box, kind, rail_corner in zip(boxes, kinds, corners):
        r0, r1, c0, c1 = box
        tile, positive = _record(rng, r1-r0, c1-c0, roles, query, labels, kind, rail_corner, teaching)
        grid[r0:r1, c0:c1] = tile
        out[r0:r1, c0:c1] = bg if positive else tile
    return {"input": _reflect(grid, corner).copy(), "output": _reflect(out, corner).copy()}


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base = list(map(int, rng.permutation(10)))
    base = base[:3+int(rng.integers(4, 8))]
    varied_roles = bool(rng.random() < .65)
    corners = list(map(int, rng.choice(4, 2, replace=False)))
    bias = str(_pick(rng, ("wide", "tall")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            package = ("identity" if i == 0 else "bounds" if i == 1 else "free") if split == "train" else "test"
            for _ in range(12):
                if varied_roles:
                    colors = list(map(int, rng.permutation(10)))
                    colors = colors[:3+int(rng.integers(4 if package != "free" else 3, 8))]
                else:
                    colors = base
                query = list(map(int, rng.choice(colors[3:], 3, replace=False)))
                corner = corners[i] if split == "train" and i < 2 else int(_pick(rng, corners))
                pair = _scene(rng, colors, query, corner, package, bias if split == "test" else "square")
                k = _key(pair["input"])
                if k not in seen:
                    break
            else:
                raise RuntimeError("duplicate-input retries exhausted")
            seen.add(k)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _labels(field, rows, cols, filler):
    return set(map(int, np.unique(field[np.ix_(rows, cols)]))) - {filler}


def _evidence(info, plate):
    f, q, filler = plate["field"], info["query"], info["filler"]
    x, y = plate["x"], plate["y"]
    ar, ac = y["active"], x["active"]
    sr, sc = range(f.shape[0]), range(f.shape[1])
    actual = set(plate["sample_labels"])
    full = _labels(f, sr, sc, filler)
    sample = f[np.ix_(ar, ac)]
    counts = {int(c): int(np.count_nonzero(sample == c)) for c in actual}
    mark_count = int(np.count_nonzero(f != filler))
    def origin_phase(axis, length):
        t = axis["terminal"]
        lo, hi = (0, length-1) if t is None else (min(axis["a"], t), max(axis["b"], t))
        return [v for v in range(lo, hi+1) if v % axis["period"] == 0]
    return {"positive": plate["positive"], "actual": actual, "full": full,
            "marks": mark_count, "sample_marks": sum(counts.values()),
            "sample_counts": counts,
            "row_only": _labels(f, ar, sc, filler),
            "col_only": _labels(f, sr, ac, filler),
            "infinite_row": _labels(f, y["full"], ac, filler),
            "infinite_col": _labels(f, ar, x["full"], filler),
            "zero_phase": _labels(f, origin_phase(y, f.shape[0]), origin_phase(x, f.shape[1]), filler),
            "before": _labels(f, ar, [c for c in ac if c < x["a"]], filler),
            "after": _labels(f, ar, [c for c in ac if c > x["b"]], filler),
            "info": info, "plate": plate, "query": q}


def _training_coverage(infos):
    es = [_evidence(i, p) for i in infos for p in i["plates"]]
    positives, negatives = [e for e in es if e["positive"]], [e for e in es if not e["positive"]]
    assert positives and negatives, "teach deletion and copying"
    assert any(e["marks"] == 3 and e["actual"] == e["query"] for e in positives), "exact positive"
    assert any(e["sample_marks"] == 3 and len(e["actual"]) == 3
               and len(e["actual"] & e["query"]) == 2 for e in negatives), "B count counterexample"
    assert any(not e["full"] for e in negatives), "empty survivor"
    assert any(e["actual"]-e["query"]
               and any(e["sample_counts"].get(c, 0) >= 2 for c in e["query"])
               and any(e["sample_counts"][c] >= 2 for c in e["actual"]-e["query"])
               for e in positives), "extra identities and repetition"
    assert any(e["query"] <= e["full"] for e in negatives), "omitting A must change an answer"
    assert any(not e["query"] <= e["zero_phase"] for e in positives), "reference phase must differ from origin zero"
    assert any(e["query"] <= e["col_only"] and not e["query"] <= e["infinite_row"]
               for e in negatives), "row phase must matter"
    assert any(e["query"] <= e["row_only"] and not e["query"] <= e["infinite_col"]
               for e in negatives), "column phase must matter"
    assert any(e["plate"]["y"]["terminal"] is not None and e["query"] <= e["infinite_row"]
               for e in negatives), "row terminal must matter"
    assert any(e["plate"]["x"]["terminal"] is not None and e["query"] <= e["infinite_col"]
               for e in negatives), "column terminal must matter"
    assert any(e["plate"]["x"]["terminal"] is None and e["before"] & e["query"]
               and e["after"] & e["query"] and e["marks"] == 3 for e in positives), "two-sided extension"
    assert any(all(any(not e["positive"] and e["actual"] & i["query"] == i["query"]-{q}
                      for e in es if e["info"] is i) for q in i["query"]) for i in infos), "each query hue matters"
    assert any(p["x"]["noise"] or p["y"]["noise"] for i in infos for p in i["plates"]), "off-phase ticks"
    assert any(e["query"]-e["full"] and
               (e["query"]-e["full"]) & {e["plate"]["x"]["reference"], e["plate"]["y"]["reference"]}
               for e in negatives), "rail colors are not content"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            k = _key(inp)
            assert k not in seen, "duplicate input"
            seen.add(k)
            info = _analyze(inp)
            view_out = _reflect(out, info["corner"])
            assert np.array_equal(view_out[:4, :4], info["view"][:4, :4]), "selector changed"
            untouched = np.ones(inp.shape, dtype=bool)
            for plate in info["plates"]:
                r0, r1, c0, c1 = plate["box"]
                if plate["positive"]:
                    assert np.all(view_out[r0:r1, c0:c1] == info["background"]), "partial deletion"
                    untouched[r0:r1, c0:c1] = False
                else:
                    assert np.array_equal(view_out[r0:r1, c0:c1], info["view"][r0:r1, c0:c1]), "survivor changed"
                assert plate["x"]["active"] and plate["y"]["active"]
            assert np.array_equal(view_out[untouched], info["view"][untouched]), "unrelated cells changed"
            assert np.array_equal(solve(inp), out), "input-only solver disagrees"
            infos[split].append(info)
    _training_coverage(infos["train"])
    corners = {i["corner"] for i in infos["train"]}
    assert len(corners) >= 2 and all(i["corner"] in corners for i in infos["test"])
    for info in infos["test"]:
        es = [_evidence(info, p) for p in info["plates"]]
        assert any(e["positive"] for e in es)
        assert any(not e["positive"] and e["query"] <= e["full"]
                   and len(e["actual"] & e["query"]) == 1
                   and len(e["row_only"] & e["query"]) == 2
                   and len(e["col_only"] & e["query"]) == 2 for e in es), "held-out two-axis recombination"


def _counterfactuals(puzzle):
    query_done = phase_done = terminal_done = False
    for pair in puzzle["train"]:
        info = _analyze(pair["input"])
        for plate in info["plates"]:
            e = _evidence(info, plate)
            if not query_done and not plate["positive"] and len(e["actual"]) == 3:
                changed = pair["input"].copy()
                v = _reflect(changed, info["corner"])
                pts = list(zip(*np.nonzero(v[:3, :3] != info["background"])))
                for p, c in zip(pts, sorted(e["actual"])):
                    v[p] = c
                result = _reflect(solve(changed), info["corner"])
                r0, r1, c0, c1 = plate["box"]
                assert np.all(result[r0:r1, c0:c1] == info["background"])
                query_done = True
            if not phase_done and not plate["positive"] and info["query"] <= e["full"]:
                missing = info["query"]-e["actual"]
                if len(missing) == 1:
                    c = next(iter(missing))
                    source = tuple(map(int, np.argwhere(plate["field"] == c)[0]))
                    candidates = [(r, col) for r in plate["y"]["active"] for col in plate["x"]["active"]
                                  if plate["field"][r, col] == info["filler"]
                                  and (r not in (0, plate["field"].shape[0]-1)
                                       or col not in (0, plate["field"].shape[1]-1))]
                    if candidates:
                        changed = pair["input"].copy()
                        v = _reflect(changed, info["corner"])
                        r0, r1, c0, c1 = plate["box"]
                        field = v[r0+1:r1-1, c0+1:c1-1]
                        field[source], field[candidates[0]] = info["filler"], c
                        result = _reflect(solve(changed), info["corner"])
                        assert np.all(result[r0:r1, c0:c1] == info["background"])
                        assert np.count_nonzero(field != info["filler"]) == e["marks"]
                        phase_done = True
            if not terminal_done and not plate["positive"]:
                axis = "y" if info["query"] <= e["infinite_row"] else "x" if info["query"] <= e["infinite_col"] else None
                if axis and plate[axis]["terminal"] is not None:
                    changed = pair["input"].copy()
                    v = _reflect(changed, info["corner"])
                    r0, r1, c0, c1 = plate["box"]
                    tile = v[r0:r1, c0:c1]
                    t = plate[axis]["terminal"]
                    rc = plate["rail_corner"]
                    if axis == "x":
                        tile[-1 if rc & 1 else 0, t+1] = info["frame"]
                    else:
                        tile[t+1, -1 if rc & 2 else 0] = info["frame"]
                    result = _reflect(solve(changed), info["corner"])
                    assert np.all(result[r0:r1, c0:c1] == info["background"])
                    terminal_done = True
    assert query_done and phase_done and terminal_done


def self_test():
    """Bounded 42-seed variation, counterfactual, symmetry, and mutation audit."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, plate_shapes, periods, modes, phases, corners, roles, layouts = (set() for _ in range(8))
    candidate_counts, deletion_counts, density = set(), set(), set()
    total_plates = 0
    signatures = set()
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        _counterfactuals(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g = pair["input"]
                info = _analyze(g)
                shapes.add(g.shape)
                roles.add(tuple(info[k] for k in ("background", "frame", "filler")))
                corners.add(info["corner"])
                candidate_counts.add(len(info["plates"]))
                deletion_counts.add(sum(p["positive"] for p in info["plates"]))
                layouts.add((g.shape, info["corner"], tuple(p["box"] for p in info["plates"])))
                for p in info["plates"]:
                    total_plates += 1
                    plate_shapes.add(p["field"].shape)
                    f = p["field"]
                    density.add(round(np.count_nonzero(f != info["filler"])/f.size, 3))
                    for axis in ("x", "y"):
                        a = p[axis]
                        periods.add(a["period"])
                        phases.add((a["period"], a["a"] % a["period"]))
                        modes.add("unbounded" if a["terminal"] is None else "before" if a["terminal"] < a["a"] else "after")
                if seed < 4:
                    for turns in range(4):
                        assert np.array_equal(solve(np.rot90(g, turns)), np.rot90(pair["output"], turns))
                    permutation = np.roll(np.arange(10), seed+1)
                    assert np.array_equal(solve(permutation[g]), permutation[pair["output"]])
                    assert np.array_equal(solve(g[:, ::-1]), pair["output"][:, ::-1])
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(plate_shapes) >= 25 and len(layouts) >= 150
    assert {1, 2, 3, 4} <= periods and len(phases) >= 8
    assert modes == {"unbounded", "before", "after"} and corners == {0, 1, 2, 3}
    assert len(roles) >= 30 and len(candidate_counts) >= 5 and len(deletion_counts) >= 4
    assert 0 in density and max(density) > .4
    for counts in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1701, *counts)
        assert (len(p["train"]), len(p["test"])) == counts
        _counterfactuals(p)
    assert tuple(_key(p["input"]) for p in generate_puzzle(17)["train"]) == tuple(_key(p["input"]) for p in puzzles[17]["train"])
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (3.2, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for corruption in ("partial_deletion", "survivor", "selector"):
        bad = {s: [{k: np.asarray(g).copy() for k, g in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        pair = next(p for p in bad["train"] if any(x["positive"] for x in _analyze(p["input"])["plates"])
                    and any(not x["positive"] for x in _analyze(p["input"])["plates"]))
        info = _analyze(pair["input"])
        v = _reflect(pair["output"], info["corner"])
        if corruption == "selector":
            v[3, 3] = info["background"]
        else:
            plate = next(x for x in info["plates"] if x["positive"] == (corruption == "partial_deletion"))
            r0, _, c0, _ = plate["box"]
            v[r0, c0] = info["frame"] if corruption == "partial_deletion" else info["background"]
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "plates": total_plates,
            "canvas_shapes": len(shapes), "interior_shapes": len(plate_shapes),
            "layouts": len(layouts), "periods": sorted(periods),
            "period_phase_combinations": len(phases), "roles": len(roles),
            "candidate_counts": sorted(candidate_counts), "deletion_counts": sorted(deletion_counts),
            "max_mark_density": float(max(density))}
