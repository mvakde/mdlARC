"""Perforated diagonal tracks. Only standard-library Python and NumPy.

The top two-chamber tool visibly supplies a stencil, its ink color, and a
component count/secondary ink. solve() reads one input and nothing else.
"""
from __future__ import annotations

import copy
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    group.append(point)
                    stack.append(point)
        groups.append(group)
    return groups


def _crop(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "empty stencil")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _segment(a, b):
    slope = 1 if b[1] > a[1] else -1
    return tuple((a[0] + t, a[1] + slope*t) for t in range(b[0] - a[0] + 1))


def _line(point, slope, shape):
    r, c = point
    lo = max(-r, -c if slope == 1 else c - shape[1] + 1)
    hi = min(shape[0] - r - 1, shape[1] - c - 1 if slope == 1 else c)
    return {(r + t, c + slope*t) for t in range(lo, hi + 1)}


def _stamps(tile, count, a, slope):
    h, w = tile.shape
    step = max(h, w) + 1
    for k in range(count):
        r = a[0] + 1 + k*step
        c = a[1] + 1 + k*step if slope == 1 else a[1] - w - k*step
        yield r, c


def _mask(tile, count, a, slope, shape):
    result = np.zeros(shape, dtype=bool)
    h, w = tile.shape
    for r, c in _stamps(tile, count, a, slope):
        _require(0 <= r and r+h <= shape[0] and 0 <= c and c+w <= shape[1],
                 "stamp falls outside field")
        result[r:r+h, c:c+w] |= tile
    return result


def _decode(input_grid):
    """Unique input-only tool parse; field colors are named by chamber order."""
    grid = _grid(input_grid)
    candidates = []
    for cut in range(1, grid.shape[0] - 1):
        if not np.all(grid[cut] == grid[cut, 0]):
            continue
        frame = int(grid[cut, 0])
        header, field = grid[:cut], grid[cut+1:]
        if np.any(field == frame):
            continue
        bars = np.flatnonzero(np.all(header == frame, axis=0))
        if len(bars) != 1 or int(np.sum(header == frame)) != cut:
            continue
        bar = int(bars[0])
        if not 0 < bar < grid.shape[1]-1:
            continue
        left, right = header[:, :bar], header[:, bar+1:]
        lc, rc = set(map(int, np.unique(left))), set(map(int, np.unique(right)))
        common = lc & rc
        if len(lc) != 2 or len(rc) != 2 or len(common) != 1:
            continue
        bg = next(iter(common))
        primary, secondary = next(iter(lc - common)), next(iter(rc - common))
        if len({bg, frame, primary, secondary}) != 4:
            continue
        if not set(map(int, np.unique(field))) <= {bg, primary, secondary}:
            continue
        tile, counter = _crop(left == primary), right == secondary
        groups = _components(counter)
        if not groups or len(groups) != len(_components(counter, True)):
            continue
        candidates.append(dict(field=field.copy(), tile=tile, count=len(groups),
                               counter=counter, component_sizes=tuple(len(g) for g in groups),
                               palette=(bg, frame, primary, secondary), cut=cut, bar=bar,
                               stencil_panel=left == primary))
    _require(len(candidates) == 1, "input has no unique two-chamber tool")
    d = candidates[0]
    field, tile, count = d["field"], d["tile"], d["count"]
    primary, secondary = d["palette"][2:]
    dots = sorted(map(tuple, np.argwhere(field == primary)))
    degrees = {p: 0 for p in dots}
    rails = []
    for i, a in enumerate(dots):
        for b in dots[i+1:]:
            if a[0] != b[0] and abs(a[0]-b[0]) == abs(a[1]-b[1]):
                degrees[a] += 1
                degrees[b] += 1
                slope = 1 if b[1] > a[1] else -1
                _require(b[0]-a[0] >= count*(max(tile.shape)+1), "track too short for count")
                rails.append(dict(a=a, b=b, slope=slope, cells=set(_segment(a, b)),
                                  mask=_mask(tile, count, a, slope, field.shape)))
    _require(rails and all(v <= 1 for v in degrees.values()), "ambiguous endpoint pairing")
    raw = np.logical_or.reduce([rail["mask"] for rail in rails])
    _require(not any(raw[p] for p in dots), "printing would obscure a primary marker")
    secondary_dots = set(map(tuple, np.argwhere(field == secondary)))
    gates, beams = [], set()
    for point in sorted(secondary_dots):
        hosts = [i for i, rail in enumerate(rails) if point in rail["cells"]]
        _require(len(hosts) <= 1, "secondary dot belongs to multiple tracks")
        if hosts:
            i = hosts[0]
            # Exclude accidental cross-support at a hole: the mask belongs to
            # the same track whose slope launches the beam.
            _require(bool(raw[point]) == bool(rails[i]["mask"][point]),
                     "secondary dot has conflicting track ownership")
            if rails[i]["mask"][point]:
                line = _line(point, -rails[i]["slope"], field.shape)
                _require(not line.intersection(dots), "beam would obscure an endpoint")
                gates.append((point, i))
                beams.update(line)
    d.update(rails=rails, primary_dots=set(dots), secondary_dots=secondary_dots,
             raw=raw, gates=gates, beams=beams,
             unpaired={p for p, v in degrees.items() if v == 0})
    return d


def _render(d, mode="correct"):
    """Counterfactual modes deliberately omit one part of the joint gate."""
    field = d["field"]
    primary, secondary = d["palette"][2:]
    out = field.copy()
    out[d["raw"]] = primary
    beams = set(d["beams"])
    if mode != "correct":
        beams = set()
        for point in d["secondary_dots"]:
            if mode == "solid_track":  # Parent B alone: ignore the perforated mask.
                owners = [i for i, rail in enumerate(d["rails"]) if point in rail["cells"]]
            elif mode == "any_printed_ink":  # Ignore B's centerline membership.
                owners = [i for i, rail in enumerate(d["rails"]) if rail["mask"][point]]
            else:
                raise ValueError("unknown rendering mode")
            if owners:
                beams.update(_line(point, -d["rails"][owners[0]]["slope"], field.shape))
    for point in beams:
        out[point] = secondary
    for point in d["secondary_dots"]:
        out[point] = secondary
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the single input, including the visible tool conventions."""
    return _render(_decode(input_grid))


def _cellwise_expected(d):
    """Validate labels by coordinates, rather than by the mask-blitting renderer."""
    field, tile = d["field"], d["tile"]
    primary, secondary = d["palette"][2:]
    out = field.copy()
    h, w = tile.shape
    origins = [(r, c) for rail in d["rails"]
               for r, c in _stamps(tile, d["count"], rail["a"], rail["slope"])]
    for r in range(field.shape[0]):
        for c in range(field.shape[1]):
            if any(0 <= r-a < h and 0 <= c-b < w and tile[r-a, c-b]
                   for a, b in origins):
                out[r, c] = primary
            if any(c-point[1] == -d["rails"][i]["slope"]*(r-point[0])
                   for point, i in d["gates"]):
                out[r, c] = secondary
            if field[r, c] == secondary:
                out[r, c] = secondary
    return out


def _endpoint_budget_ablation(d):
    """Wrong alternative: infer copies from endpoint separation, omit counter."""
    altered = copy.copy(d)
    altered["rails"] = []
    for rail in d["rails"]:
        n = (rail["b"][0]-rail["a"][0]) // (max(d["tile"].shape)+1)
        altered["rails"].append(dict(rail, mask=_mask(d["tile"], n, rail["a"], rail["slope"], d["field"].shape)))
    altered["raw"] = np.logical_or.reduce([r["mask"] for r in altered["rails"]])
    altered["gates"], altered["beams"] = [], set()
    for point in d["secondary_dots"]:
        for i, rail in enumerate(altered["rails"]):
            if point in rail["cells"] and rail["mask"][point]:
                altered["gates"].append((point, i))
                altered["beams"].update(_line(point, -rail["slope"], d["field"].shape))
    return _render(altered)


def _features(d):
    kinds = {k: set() for k in ("hole", "gap", "tail", "off_track_ink", "beyond")}
    tile, n = d["tile"], d["count"]
    h, w = tile.shape
    step = max(h, w) + 1
    union = set().union(*(rail["cells"] for rail in d["rails"]))
    for point in d["secondary_dots"]:
        if d["raw"][point] and point not in union:
            kinds["off_track_ink"].add(point)
        for rail in d["rails"]:
            a, b, s = rail["a"], rail["b"], rail["slope"]
            t = point[0] - a[0]
            if point[1] - a[1] != s*t:
                continue
            if t < 0 or t > b[0]-a[0]:
                kinds["beyond"].add(point)
            elif 0 < t < b[0]-a[0]:
                k, offset = divmod(t-1, step)
                col = offset if s == 1 else w-1-offset
                if k >= n:
                    kinds["tail"].add(point)
                elif offset == step-1 and k < n-1:
                    kinds["gap"].add(point)
                elif offset < h and 0 <= col < w and not tile[offset, col]:
                    kinds["hole"].add(point)
    return kinds


def _asymmetric(tile):
    return all(not np.array_equal(tile, np.rot90(tile[:, ::-1] if i >= 4 else tile, i % 4))
               for i in range(1, 8))


def _make_tile(rng, h, w, witness=False):
    for _ in range(80):
        tile = rng.random((h, w)) < rng.uniform(.3, .85)
        if not tile.any() or _crop(tile).shape != (h, w):
            continue
        if witness:
            if not _asymmetric(tile):
                continue
            good = True
            for slope in (-1, 1):
                coords = [(t, t if slope == 1 else w-1-t) for t in range(min(h, w))]
                values = [tile[p] for p in coords]
                if not any(values) or all(values):
                    good = False
                if not any(tile[p] and sum(tile[q] for q in _line(p, -slope, tile.shape)) >= 2
                           for p in coords):
                    good = False
                active_keys = {c+slope*r for r, c in coords if tile[r, c]}
                if not any(tile[r, c] and (r, c) not in coords and c+slope*r not in active_keys
                           for r in range(h) for c in range(w)):
                    good = False
            if not good:
                continue
        return tile
    if witness:
        tile = np.ones((h, w), dtype=bool)
        tile[0, 0] = tile[0, -1] = tile[-1, 1] = tile[1, 0] = False
        _require(_asymmetric(tile), "witness fallback must preserve orientation")
        return tile
    tile = np.ones((h, w), dtype=bool)
    if min(h, w) >= 2:
        tile[int(rng.integers(h)), int(rng.integers(w))] = False
    return tile


def _grow(rng, h, w, area):
    mask = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        p = choices[int(rng.integers(len(choices)))]
        frontier.remove(p)
        mask[p] = True
        r, c = p
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= q[0] < h and 0 <= q[1] < w and not mask[q]:
                frontier.add(q)
    return mask


def _counter(rng, h, w, count, area=None):
    _require(w >= 2*count-1, "counter chamber too narrow")
    widths = [1]*count
    for _ in range(w - (2*count-1)):
        widths[int(rng.integers(count))] += 1
    capacities = [h*x for x in widths]
    if area is None:
        upper = sum(capacities) - (1 if count == 1 else 0)
        area = int(rng.integers(count, min(upper, count+30)+1))
    _require(count <= area <= sum(capacities), "counter ink budget does not fit")
    areas = [1]*count
    for _ in range(area-count):
        choices = [i for i, cap in enumerate(capacities) if areas[i] < cap]
        areas[int(rng.choice(choices))] += 1
    result = np.zeros((h, w), dtype=bool)
    offset = 0
    for width, amount in zip(widths, areas):
        result[:, offset:offset+width] = _grow(rng, h, width, amount)
        offset += width+1
    _require(len(_components(result)) == count == len(_components(result, True)),
             "counter objects touched")
    _require(not result.all(), "counter needs visible background")
    return result


def _place_rails(rng, shape, tile, count, number, slope, witness):
    h, w = shape
    step = max(tile.shape)+1
    endpoints, rails = [], []
    margin = 1 if witness else 0
    for index in range(number):
        for _ in range(160):
            s = slope if index == 0 else int(rng.choice((-1, 1)))
            max_slack = min(8, min(shape)-1-2*margin-count*step)
            min_slack = step+2 if witness and count == 2 else (2 if witness else 0)
            if max_slack < min_slack:
                return None
            slack = int(rng.integers(min_slack, max_slack+1))
            length = count*step + slack
            a = (int(rng.integers(margin, h-length-margin)),
                 int(rng.integers(margin, w-length-margin)))
            if s == -1:
                a = a[0], a[1]+length
            b = a[0]+length, a[1]+s*length
            if any(abs(p[0]-q[0]) == abs(p[1]-q[1])
                   for p in (a, b) for q in endpoints):
                continue
            mask = _mask(tile, count, a, s, shape)
            if any(mask[p] for p in endpoints) or any(rail["mask"][p] for rail in rails for p in (a, b)):
                continue
            endpoints.extend((a, b))
            rails.append(dict(a=a, b=b, slope=s, cells=set(_segment(a, b)), mask=mask))
            break
        else:
            return None
    return rails, set(endpoints)


def _choose_marks(rng, shape, rails, endpoints, tile, count, witness):
    raw = np.logical_or.reduce([r["mask"] for r in rails])
    spans = set().union(*(r["cells"] for r in rails))
    gates, beams, qdots = [], set(), set()
    for i, rail in enumerate(rails):
        eligible = [p for p in sorted(rail["cells"] - endpoints) if rail["mask"][p]
                    and sum(p in r["cells"] for r in rails) == 1
                    and not (_line(p, -rail["slope"], shape) & endpoints)]
        amount = (2 if witness else int(rng.choice((0, 0, 1, 1, 2, 3)))) if i == 0 else int(rng.integers(3))
        if witness and len(eligible) < amount:
            return None
        chosen = [eligible[int(j)] for j in rng.permutation(len(eligible))[:min(amount, len(eligible))]]
        if witness:
            cutting = [p for p in eligible if sum(raw[q] for q in _line(p, -rail["slope"], shape)) >= 2]
            if not cutting:
                return None
            chosen[0] = cutting[int(rng.integers(len(cutting)))]
            chosen[1] = eligible[int(rng.choice([j for j, p in enumerate(eligible) if p != chosen[0]]))]
        for p in chosen:
            qdots.add(p)
            gates.append((p, i))
            beams.update(_line(p, -rail["slope"], shape))
    # Teaching dots distinguish holes, inter-copy gaps, the unused finite tail,
    # diagonal extensions past endpoints, and printed ink off the centerline.
    if witness:
        rail = rails[0]
        a, b, s = rail["a"], rail["b"], rail["slope"]
        step, (h, w) = max(tile.shape)+1, tile.shape
        pools = []
        holes = []
        for k in range(count):
            for t in range(min(h, w)):
                col = t if s == 1 else w-1-t
                if not tile[t, col]:
                    holes.append((a[0]+1+k*step+t, a[1]+s*(1+k*step+t)))
        pools.append(holes)
        pools.append([(a[0]+step, a[1]+s*step)])
        tail = [(a[0]+count*step+1+t, a[1]+s*(count*step+1+t))
                for t in range(min(h, w)) if tile[t, t if s == 1 else w-1-t]
                and count*step+1+t < b[0]-a[0]]
        pools.append(tail or [(a[0]+count*step+1, a[1]+s*(count*step+1))])
        pools.append([(a[0]-1, a[1]-s), (b[0]+1, b[1]+s)])
        off_ink = [tuple(p) for p in np.argwhere(raw)
                   if tuple(p) not in spans
                   and not (_line(tuple(p), -s, shape) & endpoints)
                   and bool(_line(tuple(p), -s, shape)-beams)]
        pools.append(off_ink)
        for pool in pools:
            valid = [p for p in pool if 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]
                     and p not in endpoints and p not in qdots]
            if not valid:
                return None
            qdots.add(valid[int(rng.integers(len(valid)))])
    # Optional inert secondary dots, including ones already on a beam.
    free = [(r, c) for r in range(shape[0]) for c in range(shape[1])
            if (r, c) not in endpoints and (r, c) not in qdots and (r, c) not in spans]
    for j in rng.permutation(len(free))[:int(rng.integers(4))]:
        qdots.add(free[int(j)])
    pdots = set(endpoints)
    for _ in range(int(rng.integers(3))):
        candidates = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                      if not raw[r, c] and (r, c) not in beams and (r, c) not in qdots
                      and all(abs(r-p[0]) != abs(c-p[1]) for p in pdots)]
        if candidates:
            pdots.add(candidates[int(rng.integers(len(candidates)))])
    return pdots, qdots


def _parallel_rails(shape, tile, count, number, slope, witness):
    """Finite safe packing of the same requested rule, not a worked-puzzle bank."""
    step = max(tile.shape)+1
    length = count*step + (step+2 if witness and count == 2 else 2)
    rails, endpoints = [], set()
    for i in range(number):
        a = (1, 1+i if slope == 1 else shape[1]-2-i)
        b = (1+length, a[1]+slope*length)
        _require(0 <= b[0] < shape[0] and 0 <= b[1] < shape[1], "safe packing exceeds field")
        rails.append(dict(a=a, b=b, slope=slope, cells=set(_segment(a, b)),
                          mask=_mask(tile, count, a, slope, shape)))
        endpoints.update((a, b))
    return rails, endpoints


def _make_pair(rng, palette, tile, count, number, slope, witness, used, force_safe=False):
    bg, frame, primary, secondary = palette
    head_h = tile.shape[0] + int(rng.integers(1 if witness else 0, 3))
    if witness and count == 3 and tile.shape[0] == 5:
        head_h = 6  # The three-copy tall witness spends the full 30-row budget.
    head_h = max(3, head_h)
    max_field_h = 29-head_h
    step = max(tile.shape)+1
    extra = step+5 if witness and count == 2 else (5 if witness else 1)
    minimum = count*step + extra
    _require(minimum <= max_field_h, "requested printing geometry cannot fit")
    for attempt in range(91):
        safe = force_safe or attempt == 90
        fh = max_field_h if safe else int(rng.integers(max(minimum, 10), max_field_h+1))
        fw = 30 if safe else int(rng.integers(max(minimum, 10), 31))
        geometry = (_parallel_rails((fh, fw), tile, count, number, slope, witness) if safe else
                    _place_rails(rng, (fh, fw), tile, count, number, slope, witness))
        if geometry is None:
            continue
        marks = _choose_marks(rng, (fh, fw), *geometry, tile, count, witness)
        if marks is None:
            continue
        pdots, qdots = marks
        if safe:
            pdots = geometry[1]  # The safe first row is reserved for uniqueness.
        left_w = tile.shape[1] + int(rng.integers(1 if witness else 0, 4))
        if tile.all() and head_h == tile.shape[0] and left_w == tile.shape[1]:
            left_w += 1
        if fw-left_w-1 < 2*count-1:
            continue
        header = np.full((head_h, fw), bg, dtype=np.int8)
        header[:, left_w] = frame
        tr = int(rng.integers(head_h-tile.shape[0]+1))
        tc = int(rng.integers(left_w-tile.shape[1]+1))
        header[tr:tr+tile.shape[0], tc:tc+tile.shape[1]][tile] = primary
        counter = _counter(rng, head_h, fw-left_w-1, count, 12 if witness else None)
        header[:, left_w+1:][counter] = secondary
        field = np.full((fh, fw), bg, dtype=np.int8)
        for p in pdots:
            field[p] = primary
        for p in qdots:
            field[p] = secondary
        inp = np.concatenate((header, np.full((1, fw), frame, dtype=np.int8), field))
        if safe:
            # All tracks begin in field row 1; row 0 cannot launch a beam.
            # At most 13 prior inputs and at least 26 eligible single-dot
            # variants give a deterministic finite uniqueness guarantee.
            for c in range(fw):
                if field[0, c] != bg:
                    continue
                candidate = inp.copy()
                candidate[head_h+1, c] = secondary
                if _key(candidate) not in used:
                    inp = candidate
                    break
            else:
                raise AssertionError("safe uniqueness capacity exhausted")
        if _key(inp) in used:
            continue
        try:
            d = _decode(inp)
        except AssertionError:
            continue
        if witness:
            f = _features(d)
            if not all(f.values()):
                continue
            if np.array_equal(_render(d), _render(d, "solid_track")) or np.array_equal(_render(d), _render(d, "any_printed_ink")):
                continue
            if not any(sum(d["raw"][p] for p in _line(point, -d["rails"][i]["slope"], field.shape)) >= 2
                       for point, i in d["gates"]):
                continue
        return {"input": inp, "output": _render(d)}
    raise AssertionError("safe packing did not satisfy the family invariants")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palettes = [tuple(map(int, rng.choice(10, 4, replace=False))) for _ in range(int(rng.integers(1, 4)))]
    shift = int(rng.integers(3))
    slopes = list(map(int, rng.permutation((-1, 1))))
    witness_shapes = ((3, 4), (5, 4)) if rng.random() < .5 else ((4, 3), (4, 5))
    witness_tiles = [_make_tile(rng, *shape, witness=True) for shape in witness_shapes]
    puzzle, used = {"train": [], "test": []}, set()
    for split, amount in (("train", nt), ("test", nq)):
        for i in range(amount):
            witness = split == "train" and i < 2
            if witness:
                tile, count, number, slope = witness_tiles[i].copy(), 2+i, 1, slopes[i]
            elif split == "train":
                h, w = map(int, rng.integers(1, 5, size=2))
                tile, count, number, slope = _make_tile(rng, h, w), int(rng.integers(1, 4)), 1, int(rng.choice((-1, 1)))
            elif shift == 0:  # More copies, independently of track number and marker density.
                count = int(rng.integers(4, 6))
                maximum = 2 if count == 5 else 3
                h, w = map(int, rng.integers(1, maximum+1, size=2))
                tile, number, slope = _make_tile(rng, h, w), 1, int(rng.choice((-1, 1)))
            elif shift == 1:  # Joint field of several tracks, with familiar copy budgets.
                h, w = map(int, rng.integers(1, 5, size=2))
                tile, count = _make_tile(rng, h, w), int(rng.integers(1, 4))
                number, slope = int(rng.integers(2, 4)), int(rng.choice((-1, 1)))
            else:  # Larger stencil, with fewer copies so the grid still fits.
                h, w = int(rng.integers(2, 8)), int(rng.integers(6, 8))
                if rng.random() < .5:
                    h, w = w, h
                tile, count, number, slope = _make_tile(rng, h, w), int(rng.integers(1, 3)), 1, int(rng.choice((-1, 1)))
            palette = palettes[int(rng.integers(len(palettes)))]
            pair = _make_pair(rng, palette, tile, count, number, slope, witness, used)
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    try:
        _counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            _require(out.shape == d["field"].shape, "output must contain only the field")
            _require(np.array_equal(out, _cellwise_expected(d)), "incorrect solved output")
            _require(all(out[p] == d["field"][p] for p in d["primary_dots"] | d["secondary_dots"]),
                     "original markers changed")
            analyses[split].append(d)
    train, test = analyses["train"], analyses["test"]
    _require({rail["slope"] for d in train for _, i in d["gates"] for rail in [d["rails"][i]]} == {-1, 1},
             "training must demonstrate both active slopes")
    _require(any(len(d["gates"]) >= 2 for d in train), "missing multiple-gate demonstration")
    _require(any(sum(d["component_sizes"]) > d["count"] for d in train), "pixel counting is not distinguished")
    _require(any(a["count"] != b["count"] and sum(a["component_sizes"]) == sum(b["component_sizes"])
                 for a in train for b in train), "missing equal-area/different-count witness")
    _require(any(_asymmetric(d["tile"]) and d["stencil_panel"].shape != d["tile"].shape for d in train),
             "missing crop and orientation demonstration")
    for kind in ("hole", "gap", "tail", "off_track_ink", "beyond"):
        _require(any(_features(d)[kind] for d in train), "missing gate distinction: " + kind)
    for mode in ("solid_track", "any_printed_ink"):
        _require(any(not np.array_equal(_render(d), _render(d, mode)) for d in train),
                 "training does not distinguish omitted parent: " + mode)
    _require(any(sum(d["raw"][p] for p in d["beams"]) > len(d["gates"]) for d in train),
             "secondary overprint is not demonstrated")
    _require(any(rail["b"][0]-rail["a"][0] >= (d["count"]+1)*(max(d["tile"].shape)+1)
                 for d in train for rail in d["rails"]),
             "endpoint spacing could substitute for the counter")
    _require(any(not np.array_equal(_render(d), _endpoint_budget_ablation(d)) for d in train),
             "omitting the counter gives the same demonstrated answers")
    _require(any(d["tile"].shape[0] != d["tile"].shape[1] and d["count"] > 1 for d in train),
             "rectangular registration spacing is not demonstrated")
    _require(len({max(d["tile"].shape) for d in train if d["count"] > 1}) >= 2
             and any(d["tile"].shape[0] > d["tile"].shape[1] for d in train)
             and any(d["tile"].shape[0] < d["tile"].shape[1] for d in train),
             "training cannot distinguish fixed or single-axis registration spacing")
    # The supported sampling shifts change a single structural axis while
    # retaining the same semantics. Inspect the grids, not a saved mode label.
    _require(all(d["count"] > max(t["count"] for t in train)
                 or len(d["rails"]) > max(len(t["rails"]) for t in train)
                 or max(d["tile"].shape) > max(max(t["tile"].shape) for t in train)
                 for d in test), "test lacks the promised structural shift")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed diversity, ablation, determinism and corruption checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows = [_decode(p["input"]) for puzzle in puzzles for s in ("train", "test") for p in puzzle[s]]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "puzzles do not vary")
    _require(len({d["field"].shape for d in rows}) >= 40, "field shapes do not vary")
    _require(len({_key(d["tile"]) for d in rows}) >= 45, "stencil structures do not vary")
    _require(len({d["palette"] for d in rows}) >= 30, "palette roles do not vary")
    _require({1, 2, 3, 4, 5} <= {d["count"] for d in rows}, "copy counts do not vary")
    _require({1, 2, 3} <= {len(d["rails"]) for d in rows}, "track counts do not vary")
    _require(any(not d["gates"] for d in rows), "no-gate members missing")
    _require(any(d["unpaired"] for d in rows), "unpaired dots missing")
    _require(any(len(_components(d["tile"])) > 1 for d in rows), "fragmented stencils missing")
    _require(any(d["tile"].all() for d in rows), "simple solid stencils missing")
    _require(len({d["component_sizes"] for d in rows}) >= 45, "counter geometry does not vary")
    # Exercise the finite packing, including its 14-input uniqueness capacity.
    safe_rng = np.random.default_rng(9871)
    safe_tiles = [_make_tile(safe_rng, *shape, witness=True) for shape in ((3, 4), (5, 4))]
    safe, used = {"train": [], "test": []}, set()
    for split, count in (("train", 10), ("test", 4)):
        for i in range(count):
            witness = split == "train" and i < 2
            pair = _make_pair(safe_rng, (0, 8, 2, 6), safe_tiles[i] if witness else safe_tiles[0], 2+i if witness else 2,
                              1 if split == "train" else 3, -1 if i % 2 else 1,
                              witness, used, force_safe=True)
            safe[split].append(pair)
            used.add(_key(pair["input"]))
    validate_generated_puzzle(safe)
    for seed in (0, 17, 41):
        _require(_puzzle_key(puzzles[seed]) == _puzzle_key(generate_puzzle(seed)), "nondeterministic puzzle")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(7319, nt, nq)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == nq, "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = copy.deepcopy(puzzles[0])
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted label")
    # Mutating the counter quantity must affect printing/gate eligibility, not
    # just an unused legend. The finite-track-only and ink-only ablations are
    # also required in every training puzzle by the validator.
    d = _decode(puzzles[0]["train"][0]["input"])
    reduced = copy.copy(d)
    reduced["count"] = 1
    reduced["rails"] = [dict(rail, mask=_mask(d["tile"], 1, rail["a"], rail["slope"], d["field"].shape))
                        for rail in d["rails"]]
    reduced["raw"] = np.logical_or.reduce([r["mask"] for r in reduced["rails"]])
    reduced["gates"] = [(p, i) for p, i in d["gates"] if reduced["rails"][i]["mask"][p]]
    reduced["beams"] = set().union(*(_line(p, -reduced["rails"][i]["slope"], d["field"].shape)
                                     for p, i in reduced["gates"]))
    _require(not np.array_equal(_render(d), _render(reduced)), "counter quantity has no effect")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
