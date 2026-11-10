"""Stable cargo docking with persistent, payload-colored ray barriers.

Only standard-library modules and NumPy are used. No parent code, examples,
seed metadata, or puzzle-wide hidden convention is needed by solve().
"""
from __future__ import annotations

import copy
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= g) & (g <= 9)), "invalid colors")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _decode(g):
    """Read separated eight-connected cargo; aperture size identifies motion."""
    h, w = g.shape
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] * 2 > g.size, "background must occupy a strict majority")
    visited = g == bg
    objects = []
    ring_colors, axes = set(), set()
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            visited[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and not visited[yy, xx]:
                            visited[yy, xx] = True
                            stack.append((yy, xx))
            r0, r1 = min(y for y, _ in cells), max(y for y, _ in cells) + 1
            c0, c1 = min(x for _, x in cells), max(x for _, x in cells) + 1
            patch = g[r0:r1, c0:c1].copy()
            ph, pw = patch.shape
            _require(ph >= 4 and pw >= 4, "cargo needs an interior of at least 2 by 2")
            ring, payload = int(patch[0, 0]), int(patch[1, 1])
            _require(len({bg, ring, payload}) == 3, "cargo colors erase its geometry")
            _require(np.all(patch[1:-1, 1:-1] == payload), "core must be solid")
            _require(all(int(patch[y, x]) == ring for y, x in
                         ((0, 0), (0, pw-1), (ph-1, 0), (ph-1, pw-1))), "open corner")
            faces = {
                "W": [(y, 0) for y in range(1, ph-1)],
                "E": [(y, pw-1) for y in range(1, ph-1)],
                "N": [(0, x) for x in range(1, pw-1)],
                "S": [(ph-1, x) for x in range(1, pw-1)],
            }
            directions = {"W": (0, -1), "E": (0, 1), "N": (-1, 0), "S": (1, 0)}
            gaps, doubles = {}, []
            for name, face in faces.items():
                _require(all(int(patch[p]) in (bg, ring) for p in face), "mixed ring")
                indices = [i for i, p in enumerate(face) if int(patch[p]) == bg]
                gaps[name] = indices
                if len(indices) == 2 and indices[1] == indices[0] + 1:
                    doubles.append(name)
            _require(len(doubles) == 1, "need exactly one double travel aperture")
            command = doubles[0]
            opposite = {"W": "E", "E": "W", "N": "S", "S": "N"}[command]
            _require(not gaps[opposite], "opposite travel face must be complete")
            for name, indices in gaps.items():
                if name != command:
                    _require(all(b-a > 1 for a, b in zip(indices, indices[1:])),
                             "perpendicular ports must be isolated singletons")
            ports = [(faces[name][i], directions[name]) for name in faces for i in gaps[name]]
            axis = 1 if command in ("W", "E") else 0
            axes.add(axis)
            ring_colors.add(ring)
            objects.append(dict(r=r0, c=c0, h=ph, w=pw, patch=patch,
                                payload=payload, direction=directions[command], ports=ports))
    _require(objects and len(axes) == 1 and len(ring_colors) == 1, "inconsistent cargo grammar")
    for i, a in enumerate(objects):
        for b in objects[i+1:]:
            _require(_separated(a, b), "input cargo envelopes must be separated")
    return bg, next(iter(axes)), objects


def _separated(a, b):
    return (a["r"] + a["h"] < b["r"] or b["r"] + b["h"] < a["r"] or
            a["c"] + a["w"] < b["c"] or b["c"] + b["w"] < a["c"])


def _rows_overlap(a, b):
    return max(a["r"], b["r"]) < min(a["r"]+a["h"], b["r"]+b["h"])


def _canonical(g):
    bg, axis, objects = _decode(g)
    if axis == 0:
        g = g.T.copy()
        bg, _, objects = _decode(g)
    return g, bg, axis, objects


def _emit(canvas, obj, column, bg, owner=None, ident=-1):
    """Color every port, then extend until settled material or the boundary."""
    stops, lengths = 0, []
    for (pr, pc), (dr, dc) in obj["ports"]:
        r, c = obj["r"] + pr, column + pc
        _require(canvas[r, c] == bg, "port was overwritten")
        length = 0
        while 0 <= r < canvas.shape[0] and 0 <= c < canvas.shape[1]:
            if canvas[r, c] != bg:
                stops += 1
                break
            canvas[r, c] = obj["payload"]
            if owner is not None:
                owner[r, c] = ident
            length += 1
            r, c = r + dr, c + dc
        lengths.append(length)
    return stops, lengths


def _settle(g):
    canonical, bg, axis, objects = _canonical(g)
    shape = canonical.shape
    result = np.full(shape, bg, dtype=np.int8)
    union = np.zeros(shape, dtype=bool)
    destinations = [None] * len(objects)
    depths = [0] * len(objects)
    dependencies, stops, lengths = [], 0, []
    for sign in (-1, 1):
        canvas = np.full(shape, bg, dtype=np.int8)
        owner = np.full(shape, -1, dtype=int)
        group = [i for i, o in enumerate(objects) if o["direction"][1] == sign]
        group.sort(key=lambda i: (objects[i]["c"] if sign == -1 else
                                 -objects[i]["c"]-objects[i]["w"], objects[i]["r"]))
        for i in group:
            o = objects[i]
            r, h, width = o["r"], o["h"], o["w"]
            yy, xx = np.where(canvas[r:r+h] != bg)
            blockers = set()
            if len(xx):
                boundary = int(xx.max()) if sign == -1 else int(xx.min())
                col = boundary + 1 if sign == -1 else boundary - width
                blockers = {int(owner[r+y, x]) for y, x in zip(yy, xx) if x == boundary}
            else:
                col = 0 if sign == -1 else shape[1] - width
            _require(0 <= col <= shape[1]-width, "cargo queue exceeds the canvas")
            _require(col <= o["c"] if sign == -1 else col >= o["c"],
                     "a new barrier would displace cargo against its aperture")
            _require(np.all(canvas[r:r+h, col:col+width] == bg), "settled envelopes overlap")
            destinations[i] = col
            depths[i] = 1 + max((depths[j] for j in blockers), default=0)
            dependencies.extend((j, i) for j in sorted(blockers))
            canvas[r:r+h, col:col+width] = o["patch"]
            owner[r:r+h, col:col+width] = i
            count, ray_lengths = _emit(canvas, o, col, bg, owner, i)
            stops += count
            lengths.extend(ray_lengths)
        mask = canvas != bg
        _require(not np.any(mask & union), "opposite directional material overlaps")
        result[mask] = canvas[mask]
        union |= mask
    info = dict(background=bg, axis=axis, objects=objects, destinations=destinations,
                depths=depths, dependencies=dependencies, stops=stops, ray_lengths=lengths)
    return (result if axis == 1 else result.T.copy()), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from this input's visible cargo and apertures."""
    return _settle(_grid(input_grid))[0]


def _box_destinations(objects, width, edge_first=False):
    """Explicit competing law: ignore all emitted barriers."""
    positions = [None] * len(objects)
    for sign in (-1, 1):
        ids = [i for i, o in enumerate(objects) if o["direction"][1] == sign]
        ids.sort(key=lambda i: (objects[i]["c"] if sign == -1 else
                               -objects[i]["c"]-objects[i]["w"], objects[i]["r"]))
        previous = []
        for i in ids:
            o = objects[i]
            blockers = [j for j in previous if _rows_overlap(o, objects[j])]
            if not edge_first:
                col = (max((positions[j]+objects[j]["w"] for j in blockers), default=0)
                       if sign == -1 else min((positions[j] for j in blockers), default=width)-o["w"])
            else:
                candidates = range(width-o["w"]+1)
                if sign == 1:
                    candidates = reversed(candidates)
                col = next((x for x in candidates if all(
                    x+o["w"] <= positions[j] or positions[j]+objects[j]["w"] <= x
                    for j in blockers)), None)
                _require(col is not None, "alternative does not fit")
            _require(0 <= col <= width-o["w"], "alternative does not fit")
            positions[i] = col
            previous.append(i)
    return positions


def _emit_on_fixed_cargo(g, positions):
    canonical, bg, axis, objects = _canonical(g)
    out = np.full_like(canonical, bg)
    # All final rectangles are fixed blockers for this competing emission law.
    for o, col in zip(objects, positions):
        r, h, w = o["r"], o["h"], o["w"]
        _require(np.all(out[r:r+h, col:col+w] == bg), "alternative rectangles overlap")
        out[r:r+h, col:col+w] = o["patch"]
        for (pr, pc), _ in o["ports"]:
            out[r+pr, col+pc] = o["payload"]
    for o, col in zip(objects, positions):
        for (pr, pc), (dr, dc) in o["ports"]:
            r, c = o["r"]+pr+dr, col+pc+dc
            while 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and out[r, c] == bg:
                out[r, c] = o["payload"]
                r, c = r+dr, c+dc
    return out if axis == 1 else out.T.copy()


def _metrics(g):
    out, info = _settle(g)
    canonical, bg, axis, objects = _canonical(g)
    boxes = _box_destinations(objects, canonical.shape[1])
    edge = _box_destinations(objects, canonical.shape[1], True)
    sensitive = [i for i, (a, b) in enumerate(zip(info["destinations"], boxes)) if a != b]
    info.update(beam_sensitive=sensitive, stable_witness=boxes != edge,
                nonlocal_dependencies=sum(not _rows_overlap(objects[a], objects[b])
                             for a, b in info["dependencies"]),
                stationary=sum(o["c"] == c for o, c in zip(objects, info["destinations"])),
                count=len(objects), shape=g.shape,
                footprints=tuple(sorted({(o["h"], o["w"]) for o in objects})),
                port_counts=tuple(len(o["ports"]) for o in objects),
                classes=tuple(sum(o["direction"][1] == d for o in objects) for d in (-1, 1)),
                roles=(bg, int(objects[0]["patch"][0, 0])))
    return info


def _has_tie(objects):
    for sign in (-1, 1):
        keys = [o["c"] if sign == -1 else o["c"]+o["w"]
                for o in objects if o["direction"][1] == sign]
        if len(keys) != len(set(keys)):
            return True
    return False


def _patch(h, w, sign, notch, top, bottom, bg, ring, color):
    p = np.full((h, w), ring, dtype=np.int8)
    p[1:-1, 1:-1] = color
    p[notch:notch+2, 0 if sign == -1 else w-1] = bg
    for x in top:
        p[0, x] = bg
    for x in bottom:
        p[-1, x] = bg
    return p


def _object(rng, r, c, h, w, sign, top, bottom, roles, color):
    bg, ring = roles
    return dict(r=int(r), c=int(c), h=int(h), w=int(w),
                patch=_patch(h, w, sign, int(rng.integers(1, h-2)), top, bottom, bg, ring, color))


def _render(shape, objects, bg):
    g = np.full(shape, bg, dtype=np.int8)
    for o in objects:
        r, c, h, w = (o[k] for k in ("r", "c", "h", "w"))
        g[r:r+h, c:c+w] = o["patch"]
    return g


def _colors(rng, p, changed=False):
    roles = p["roles"] if not changed and rng.random() < .55 else tuple(
        int(x) for x in rng.choice(10, 2, replace=False))
    if changed and roles == p["roles"]:
        roles = roles[::-1]
    available = [c for c in range(10) if c not in roles]
    rng.shuffle(available)
    return roles, available


def _dimensions(rng, p, cap=7):
    if p["size"] is not None:
        return p["size"], p["size"]
    h, w = (int(x) for x in rng.integers(4, cap+1, size=2))
    return h, h if rng.random() < .35 else w


def _orient(g, rng, axis=None):
    if bool(rng.integers(2)):
        g = np.fliplr(g)
    if axis is None:
        axis = int(rng.integers(2))
    return g.copy() if axis == 1 else g.T.copy()


def _curtain(rng, p, roles, colors):
    ha, wa = _dimensions(rng, p)
    hb, wb = _dimensions(rng, p)
    # Same payload and footprint under opposite commands separates color/size
    # from the visible travel-aperture instruction.
    ho, wo = ha, wa
    rb = int(rng.integers(0, 3))
    ra = rb + hb + int(rng.integers(1, 4))
    height = ra + ha + int(rng.integers(1, 5))
    width = int(rng.integers(max(18, wo+wb+5), 31))
    port = int(rng.integers(1, wa-2))
    ca = int(rng.integers(0, min(6, width-wa)))
    co = int(rng.integers(0, 2))
    cb = int(rng.integers(max(ca+1, co+wo+1), width-wb+1))
    objects = [
        _object(rng, ra, ca, ha, wa, -1, [port], [], roles, colors[0]),
        _object(rng, rb, cb, hb, wb, -1, [], [1], roles, colors[1]),
        _object(rng, rb, co, ho, wo, 1, [], [], roles, colors[0]),
    ]
    # The opposite object can be taller than its band. Put it above the lower
    # object without depending on an accidental empty-region overlap.
    if not all(_separated(a, b) for i, a in enumerate(objects) for b in objects[i+1:]):
        objects[2]["r"] = 0
        objects[0]["r"] = max(ra, ho+1)
        height = max(height, objects[0]["r"]+ha+1)
    if rng.random() < .35 and height + ho + 1 <= 30:
        r = height
        objects.append(_object(rng, r, co+2, ho, wo, 1, [], [], roles, colors[2]))
        height += ho + int(rng.integers(1, min(4, 31-height-ho)))
    return _render((height, width), objects, roles[0])


def _chain(rng, p, roles, colors):
    h, first_w = _dimensions(rng, p, 6)
    widths = [first_w] + [(_dimensions(rng, p, 6)[1]) for _ in range(2)]
    step = int(rng.integers((h+1)//2, h))
    base = int(rng.integers(0, 3))
    heights = base + h + 2*step + int(rng.integers(0, 4))
    gaps = [int(x) for x in rng.integers(1, 4, size=2)]
    start = int(rng.integers(0, 4))
    minimum_width = start + sum(widths) + sum(gaps)
    width = int(rng.integers(minimum_width, 31))
    # Extra background keeps even large common-size chains unambiguous.
    heights = max(heights, (2*sum(h*w-2 for w in widths))//width + 1)
    objects, c = [], start
    reverse = bool(rng.integers(2))
    for i, w in enumerate(widths):
        r = base + (2-i if reverse else i)*step
        objects.append(_object(rng, r, c, h, w, -1, [], [], roles, colors[i]))
        c += w + (gaps[i] if i < 2 else 0)
    return _render((heights, width), objects, roles[0])


def _relay(rng, p, roles, colors):
    max_count = min(5, 31//((p["size"] or 4)+1))
    n = int(rng.integers(3, max_count+1))
    cap = min(7, (29-(n-1))//n)
    dimensions = [_dimensions(rng, p, cap) for _ in range(n)]
    row_gaps = [1]*(n-1)
    col_gaps = [1]*(n-1)
    rmargin, cmargin = int(rng.integers(0, 2)), int(rng.integers(0, 2))
    min_h = sum(h for h, _ in dimensions) + sum(row_gaps) + rmargin
    min_w = sum(w for _, w in dimensions) + sum(col_gaps) + cmargin
    height, width = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
    rows, r = [], rmargin
    for i, (h, _) in enumerate(dimensions):
        rows.append(r)
        r += h + (row_gaps[i] if i < n-1 else 0)
    order = [int(i) for i in rng.permutation(n)]
    objects, c = [], cmargin
    for k, band in enumerate(order):
        h, w = dimensions[band]
        top, bottom = [], []
        if k < n-1:
            (top if rows[order[k+1]] < rows[band] else bottom).append(w-2)
        # Additional singleton ports change occlusion and dependency reach,
        # while retaining the long relay's guaranteed outer curtain.
        if rng.random() < .45:
            opposite = bottom if top else top
            opposite.append(int(rng.integers(1, w-1)))
        objects.append(_object(rng, rows[band], c, h, w, -1, top, bottom,
                               roles, colors[k % len(colors)]))
        c += w + (col_gaps[k] if k < n-1 else 0)
    return _render((height, width), objects, roles[0])


def _singletons(rng, width):
    selected = []
    for x in rng.permutation(np.arange(1, width-1)):
        if rng.random() < .35 and all(abs(int(x)-y) > 1 for y in selected):
            selected.append(int(x))
    return sorted(selected)


def _free(rng, p, roles, colors):
    for _ in range(50):
        height, width = (int(x) for x in rng.integers(11, 31, size=2))
        target = int(rng.integers(1, 9))
        objects = []
        for _ in range(target):
            for attempt in range(40):
                h, w = _dimensions(rng, p)
                r, c = int(rng.integers(height-h+1)), int(rng.integers(width-w+1))
                sign = -1 if rng.random() < .5 else 1
                top, bottom = _singletons(rng, w), _singletons(rng, w)
                o = _object(rng, r, c, h, w, sign, top, bottom, roles,
                            colors[int(rng.integers(min(len(colors), target)))])
                if all(_separated(o, a) for a in objects):
                    objects.append(o)
                    break
        if not objects:
            continue
        g = _render((height, width), objects, roles[0])
        try:
            _, info = _settle(g)
            if not _has_tie(info["objects"]):
                return g
        except ValueError:
            continue
    return _relay(rng, p, roles, colors)


def _counts(num_train, num_test):
    for name, n, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        _require(isinstance(n, (int, np.integer)) and not isinstance(n, (bool, np.bool_))
                 and low <= n <= high, f"illegal {name}")
    _require(num_train > num_test, "num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    p = dict(size=int(rng.integers(4, 7)) if rng.random() < .35 else None,
             roles=tuple(int(x) for x in rng.choice(10, 2, replace=False)))
    shared_axis = int(rng.integers(2)) if rng.random() < .35 else None
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            mode = "curtain" if split == "train" and i == 0 else (
                "chain" if split == "train" and i == 1 else
                "relay" if split == "test" and i == 0 else
                "free" if rng.random() < .75 else "relay")
            for attempt in range(80):
                roles, colors = _colors(rng, p, changed=split == "train" and i == 1)
                try:
                    builder = {"curtain": _curtain, "chain": _chain,
                               "relay": _relay, "free": _free}[mode]
                    g = _orient(builder(rng, p, roles, colors), rng, shared_axis)
                    out, info = _settle(g)
                    m = _metrics(g)
                    if _has_tie(info["objects"]) or _key(g) in seen:
                        continue
                    if mode == "curtain" and not (m["beam_sensitive"] and m["nonlocal_dependencies"] and m["stops"]):
                        continue
                    if mode == "chain" and not m["stable_witness"]:
                        continue
                    if split == "test" and not m["beam_sensitive"]:
                        continue
                    if mode == "relay" and max(m["depths"]) < 3:
                        continue
                except ValueError:
                    continue
                puzzle[split].append({"input": g, "output": out})
                seen.add(_key(g))
                break
            else:
                raise RuntimeError(f"bounded construction exhausted for {split}/{mode}")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_metrics = set(), []
    witnesses = dict(no_a=False, no_b=False, order=False, curtain=False, occlusion=False)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            g, actual = _grid(pair["input"]), _grid(pair["output"])
            _require(g.shape == actual.shape, "canvas changed")
            _require(_key(g) not in seen, "duplicate input")
            seen.add(_key(g))
            expected, info = _settle(g)
            _require(np.array_equal(actual, expected), "incorrect docking or emission")
            _require(not _has_tie(info["objects"]), "generated curricula exclude primary-order ties")
            canonical, bg, axis, objects = _canonical(g)
            answer = actual if axis == 1 else actual.T
            for i, (o, col) in enumerate(zip(objects, info["destinations"])):
                settled_patch = answer[o["r"]:o["r"]+o["h"], col:col+o["w"]]
                mask = o["patch"] != bg
                _require(np.array_equal(settled_patch[mask], o["patch"][mask]), "cargo pattern changed")
                _require(all(int(settled_patch[p]) == o["payload"] for p, _ in o["ports"]),
                         "port was not colored with its cargo")
                for j, b in enumerate(objects):
                    if o["direction"] != b["direction"] or not _rows_overlap(o, b):
                        continue
                    if o["c"] < b["c"]:
                        _require(col+o["w"] <= info["destinations"][j], "same-class overtaking")
            m = _metrics(g)
            if split == "train":
                train_metrics.append(m)
                witnesses["no_a"] |= not np.array_equal(
                    actual, _emit_on_fixed_cargo(g, [o["c"] for o in objects]))
                try:
                    alt = _emit_on_fixed_cargo(g, _box_destinations(objects, canonical.shape[1]))
                    witnesses["no_b"] |= not np.array_equal(actual, alt) and bool(m["beam_sensitive"])
                    alt_order = _emit_on_fixed_cargo(g, _box_destinations(objects, canonical.shape[1], True))
                    witnesses["order"] |= m["stable_witness"] and not np.array_equal(actual, alt_order)
                except ValueError:
                    # Alternatives may be infeasible in free scenes; mandatory
                    # witnesses must have concrete legal competing outputs.
                    pass
                witnesses["curtain"] |= bool(m["beam_sensitive"] and m["nonlocal_dependencies"])
                witnesses["occlusion"] |= m["stops"] > 0
            else:
                _require(bool(m["beam_sensitive"]), "test must exercise ray-based docking")
    _require(all(witnesses.values()), f"insufficient demonstrations: {witnesses}")
    _require(any(min(m["classes"]) > 0 for m in train_metrics), "training must show both commands")
    _require(max(max(_metrics(_grid(p["input"]))["depths"]) for p in puzzle["test"]) >= 3,
             "test must extend the persistent dependency chain")


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(pair["input"])), _key(np.asarray(pair["output"])))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded, generator-owned regressions plus 42-seed structural coverage."""
    # Handcrafted curtain: lower 4x4 cargo's north beam is at column 1.
    # A disjoint upper 4x4 cargo therefore docks at column 2, not column 0.
    rng = np.random.default_rng(123)
    objects = [_object(rng, 7, 2, 4, 4, -1, [1], [], (0, 8), 3),
               _object(rng, 1, 9, 4, 4, -1, [], [1], (0, 8), 6)]
    g = _render((12, 17), objects, 0)
    out, m = _settle(g)
    by_row = {o["r"]: col for o, col in zip(m["objects"], m["destinations"])}
    assert by_row == {7: 0, 1: 2}
    assert np.all(out[:7, 1] == 3)
    assert out[5, 3] == out[6, 3] == 6 and out[7, 3] == 8
    # Pure stable chain: endpoint spans are disjoint, but cannot overtake.
    chain = [_object(rng, r, c, 4, 4, -1, [], [], (0, 8), color)
             for r, c, color in ((0, 1, 2), (2, 6, 3), (4, 11, 4))]
    _, info = _settle(_render((10, 17), chain, 0))
    assert info["destinations"] == [0, 4, 8]
    assert _box_destinations(info["objects"], 17, True) == [0, 4, 0]
    # Decode a feasible larger rectangle beyond the sampler's size range.
    large = _object(rng, 1, 4, 8, 9, -1, [2, 5], [], (0, 7), 4)
    large_g = _render((18, 20), [large], 0)
    assert solve(large_g).shape == large_g.shape
    assert np.array_equal(solve(g.T), out.T)
    assert np.array_equal(solve(np.fliplr(g)), np.fliplr(out))
    corpus, records = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        corpus.append(puzzle)
        records.extend(_metrics(pair["input"]) for s in puzzle.values() for pair in s)
    assert len({_fingerprint(p) for p in corpus}) == 42
    assert _fingerprint(corpus[0]) == _fingerprint(generate_puzzle(0))
    assert {m["axis"] for m in records} == {0, 1}
    assert len({m["shape"] for m in records}) >= 40
    assert len({m["footprints"] for m in records}) >= 30
    assert 1 in {m["count"] for m in records} and max(m["count"] for m in records) >= 6
    assert max(max(m["depths"]) for m in records) >= 4
    assert any(m["stationary"] for m in records)
    assert any(min(m["classes"]) == 0 for m in records)
    assert any(min(m["classes"]) > 0 for m in records)
    assert any(len(set(o["payload"] for o in m["objects"])) < m["count"] for m in records)
    assert max(max(m["port_counts"]) for m in records) >= 5
    assert max(m["nonlocal_dependencies"] for m in records) >= 2
    assert max(m["stops"] for m in records) >= 3
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(1000+nt*10+ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = copy.deepcopy(corpus[0])
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": 42,
            "shapes": len({m["shape"] for m in records}),
            "footprint_sets": len({m["footprints"] for m in records}),
            "object_counts": sorted({m["count"] for m in records}),
            "max_dependency_depth": max(max(m["depths"]) for m in records),
            "nonlocal_dependencies": sum(m["nonlocal_dependencies"] for m in records)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
