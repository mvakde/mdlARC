"""Reclaimed thread: mirror-supported rigid material, surplus becomes a strand.

Only NumPy and the standard library are used. No files, parent code, latent
generation state, or demonstrations are read by solve(). All diagnostics stream
in memory; this module writes no artifacts or execution traces.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json

import numpy as np


_D8 = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _box(cells):
    return (min(p[0] for p in cells), max(p[0] for p in cells),
            min(p[1] for p in cells), max(p[1] for p in cells))


def _global(p, origin, forward, lateral):
    return (origin[0] + p[0] * forward[0] + p[1] * lateral[0],
            origin[1] + p[0] * forward[1] + p[1] * lateral[1])


def _local(p, origin, forward, lateral):
    dr, dc = p[0] - origin[0], p[1] - origin[1]
    return dr * forward[0] + dc * forward[1], dr * lateral[0] + dc * lateral[1]


def _connected(cells, offsets=_D8):
    if not cells:
        return False
    remaining = set(cells)
    stack = [min(remaining)]
    remaining.remove(stack[0])
    while stack:
        r, c = stack.pop()
        for dr, dc in offsets:
            q = r + dr, c + dc
            if q in remaining:
                remaining.remove(q)
                stack.append(q)
    return not remaining


def _supported_core(material):
    """Score ordered within-row pairs; axis coordinates are doubled integers."""
    rows = defaultdict(list)
    for u, v in material:
        rows[u].append(v)
    scores = Counter()
    for vs in rows.values():
        for v in vs:
            for w in vs:
                scores[v + w] += 1
    best = max(scores.values())
    axes = [s for s, score in scores.items() if score == best]
    _require(len(axes) == 1, "reflection-axis tie")
    s = axes[0]
    core = {p for p in material if (p[0], s - p[1]) in material}
    _require(len(core) == best, "support score mismatch")
    return s, core


def _clearance(core, length):
    peak = max(v for u, v in core)
    first_peak = min(u for u, v in core if v == peak)
    return [(i, 1 if peak == 0 and i <= first_peak else
             max(v + 1 - abs(i - u) for u, v in core)) for i in range(length)]


def _route(core, length):
    _require(min(u for u, v in core) >= 2, "core must be forward of support")
    _require(len({v for u, v in core}) >= 2, "core is too narrow")
    _require(max(v for u, v in core) == 0 or max(v-u for u, v in core) == 0,
             "unsupported incoming tangent")
    route = _clearance(core, max(length, max(u for u, v in core) + 5))
    slopes = [b[1] - a[1] for a, b in zip(route, route[1:])]
    _require(route[0] == (0, 1), "route lost its attachment")
    _require(all(s in (-1, 0, 1) for s in slopes) and slopes == sorted(slopes, reverse=True),
             "exposed clearance must be convex with digital tangents")
    _require(not set(route) & core, "route intersects core")
    return route[:length]


def _construction_route(core, length):
    """Ground truth uses clearance sweeps, rather than prediction's envelope."""
    n = max(length, max(u for u, v in core) + 5)
    profile = [-1000] * n
    for u, v in core:
        profile[u] = max(profile[u], v + 1)
    for i in range(1, n):
        profile[i] = max(profile[i], profile[i-1] - 1)
    for i in range(n-2, -1, -1):
        profile[i] = max(profile[i], profile[i+1] - 1)
    if max(v for u, v in core) == 0:
        first = min(u for u, v in core if v == 0)
        profile[:first+1] = [1] * (first+1)
    return list(enumerate(profile[:length]))


def _analyze(input_grid):
    inp = _grid(input_grid)
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 3, "expected background, support, and material")
    bg = int(colors[counts.argmax()])
    _require(int(counts.max()) > inp.size / 2, "background must be a strict majority")
    lines = []
    for color in colors:
        if color == bg:
            continue
        cells = set(map(tuple, np.argwhere(inp == color).tolist()))
        if 4 <= len(cells) <= 8:
            r0, r1, c0, c1 = _box(cells)
            if (r0 == r1 or c0 == c1) and len(cells) == (r1-r0+1)*(c1-c0+1):
                lines.append((int(color), cells))
    _require(len(lines) == 1, "expected one straight support")
    support_color, support = lines[0]
    material_color = next(int(c) for c in colors if c not in (bg, support_color))
    material_global = set(map(tuple, np.argwhere(inp == material_color).tolist()))
    positive = (1, 0) if len({c for r, c in support}) == 1 else (0, 1)
    frames = []
    for origin, f in ((min(support), (-positive[0], -positive[1])), (max(support), positive)):
        for l in ((f[1], -f[0]), (-f[1], f[0])):
            anchor = origin[0] + l[0], origin[1] + l[1]
            if anchor in material_global:
                frames.append((origin, f, l))
    _require(len(frames) == 1, "attachment must identify one endpoint and side")
    origin, f, l = frames[0]
    material = {_local(p, origin, f, l) for p in material_global}
    axis, core = _supported_core(material)
    surplus = material - core
    _require((0, 1) in surplus and len(surplus) >= 3, "missing loose attachment")
    slack = {p for p in surplus if p[0] <= 0}
    _require(all(-len(support)+2 <= u <= 0 and v >= 1 for u, v in slack), "invalid slack region")
    _require(_connected(slack), "slack must be eight-connected")
    route = _route(core, len(surplus))
    return {"background": bg, "support_color": support_color, "material_color": material_color,
            "support": support, "origin": origin, "forward": f, "lateral": l,
            "axis": axis, "material": material, "core": core, "surplus": surplus,
            "slack": slack, "route": route}


def _paint(inp, info, core, route):
    out = np.full_like(inp, info["background"])
    for p in info["support"]:
        out[p] = info["support_color"]
    for p in core:
        q = _global(p, info["origin"], info["forward"], info["lateral"])
        if _inside(q, out.shape):
            out[q] = info["material_color"]
    for p in route:
        q = _global(p, info["origin"], info["forward"], info["lateral"])
        if _inside(q, out.shape):
            _require(out[q] in (info["background"], info["material_color"]), "support collision")
            out[q] = info["material_color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pure input-only recovery. Does not modify or retain the supplied grid."""
    inp = _grid(input_grid)
    info = _analyze(inp)
    return _paint(inp, info, info["core"], info["route"])


def _sample_core(rng, approach, parity, lesson):
    radius = int(rng.integers(3 if parity or lesson == "visible" else 2, 5))
    height = int(rng.integers(5 if lesson == "visible" else 4, 9))
    family = int(rng.integers(3))
    max_left = radius - (3 if parity else 2) if approach == "diagonal" else radius-1
    left = int(rng.integers(0, max_left+1))
    right = int(rng.integers(0, radius))
    if family == 0:
        left = right = 0
    elif family == 1:
        left = 0
    if lesson == "visible":
        right = max(1, right)  # Bounding-box routing must give a different answer.
    if left + right >= height:
        right = height - left - 1
    s = -parity
    core = set()
    mode = 1 if lesson == "visible" else int(rng.integers(3))
    for u in range(height):
        rad = radius - max(0, left-u) - max(0, u-(height-1-right))
        upper = s//2 + rad
        lower = s - upper
        for v in range(lower, upper+1):
            edge = u in (0, height-1) or v in (lower, upper)
            if mode == 0 or edge:
                core.add((u, v))
        if mode == 2:
            # Symmetric perforations, with pair choices sampled only once.
            for v in range((s+1)//2, upper):
                if rng.random() < .45:
                    core.update(((u, v), (u, s-v)))
    if mode == 1 and height >= 7 and radius >= 4 and rng.random() < .5:
        # A supported island inside a hollow core remains rigid material.
        u, v = height//2, (s+1)//2
        core.update(((u, v), (u, s-v)))
    if approach == "parallel":
        shift = -max(v for u, v in core)
        gap = int(rng.integers(2, 6))
        core = {(u+gap, v+shift) for u, v in core}
        s += 2*shift
    else:
        gap = max(v-u for u, v in core)
        core = {(u+gap, v) for u, v in core}
    _route(core, 3)
    return core, s


def _slack(rng, amount, depth, width, diagonal=False):
    cells = {(0, 1)}
    allowed = {(u, v) for u in range(-depth, 0) for v in range(1, width+1)}
    if diagonal:
        allowed.discard((-1, 1))
    style = int(rng.integers(3))
    while len(cells) < amount:
        frontier = sorted({(u+du, v+dv) for u, v in cells for du, dv in _D8} & allowed - cells)
        _require(bool(frontier), "slack capacity exhausted")
        weights = []
        for u, v in frontier:
            degree = sum((u+du, v+dv) in cells for du, dv in _D8)
            weights.append((degree+1)**(2 if style == 0 else -2 if style == 1 else 0))
        weights = np.asarray(weights, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))])
    return cells


def _additions(rng, core, axis, amount, hole_required):
    u0, u1, v0, v1 = _box(core)
    all_candidates = {(u, v) for u in range(max(1, u0-1), u1+3)
                      for v in range(v0-2, v1+4)
                      if (u, v) not in core and (u, axis-v) not in core and 2*v != axis}
    selected = set()
    # A visible protrusion changes clearance if mistaken for rigid material.
    outer = [(u, max(v for t, v in core if t == u)+1) for u in sorted({t for t, v in core})]
    selected.add(outer[int(rng.integers(len(outer)))])
    if hole_required:
        holes = sorted(p for p in all_candidates if u0 < p[0] < u1 and
                       min(v for u, v in core if u == p[0]) < p[1] <
                       max(v for u, v in core if u == p[0]))
        _require(bool(holes) and amount >= 2, "lesson needs an asymmetric hole addition")
        selected.add(holes[int(rng.integers(len(holes)))])
    while len(selected) < amount:
        candidates = sorted(p for p in all_candidates - selected if (p[0], axis-p[1]) not in selected)
        _require(bool(candidates), "no asymmetric addition available")
        if rng.random() < .5:
            near = [p for p in candidates if any((p[0]+a, p[1]+b) in core | selected for a, b in _D8)]
            candidates = near or candidates
        selected.add(candidates[int(rng.integers(len(candidates)))])
    return selected


def _render(rng, core, surplus, support, length, bg, colors, orientation, lesson):
    sign, side = (int(rng.choice((-1, 1))) for _ in range(2))
    f, l = ((sign, 0), (0, side)) if orientation == 0 else ((0, sign), (side, 0))
    convert = lambda cells: {_global(p, (0, 0), f, l) for p in cells}
    kc, uc, sc = convert(core), convert(surplus), convert(support)
    route = _construction_route(core, length)
    _require(route == _route(core, length), "clearance implementations disagree")
    pc = convert(route)
    input_cells = kc | uc | sc
    bounds = list(_box(input_cells | pc))
    ib = _box(input_cells)
    clip = lesson == "clipped" or (lesson is None and rng.random() < .3)
    edge = None
    if clip:
        choices = [i for i in range(4) if (bounds[i] < ib[i] if i % 2 == 0 else bounds[i] > ib[i])]
        _require(bool(choices), "no real boundary can clip this route")
        edge = int(rng.choice(choices))
        bounds[edge] = ib[edge]
    margins = [int(rng.integers(1 if lesson == "visible" else 0, 4)) for _ in range(4)]
    if edge is not None:
        margins[edge] = 0
    bounds = [bounds[0]-margins[0], bounds[1]+margins[1],
              bounds[2]-margins[2], bounds[3]+margins[3]]
    h, w = bounds[1]-bounds[0]+1, bounds[3]-bounds[2]+1
    _require(max(h, w) <= 30, "canvas too large")
    if rng.random() < .2 and h != w:
        # Add padding opposite a clipped edge so clipping is not undone.
        if h < w:
            i = 1 if edge == 0 else 0
            bounds[i] += (w-h) * (1 if i == 1 else -1)
        else:
            i = 3 if edge == 2 else 2
            bounds[i] += (h-w) * (1 if i == 3 else -1)
        h = w = max(h, w)
    shift = lambda p: (p[0]-bounds[0], p[1]-bounds[2])
    inp = np.full((h, w), bg, dtype=np.int8)
    out = inp.copy()
    for p in sc:
        inp[shift(p)] = out[shift(p)] = colors[0]
    for p in kc:
        inp[shift(p)] = out[shift(p)] = colors[1]
    for p in uc:
        inp[shift(p)] = colors[1]
    for p in pc:
        q = shift(p)
        if _inside(q, out.shape):
            _require(out[q] == bg, "thread collides with rigid material")
            out[q] = colors[1]
    _require(np.count_nonzero(inp == bg) > inp.size/2, "background not a majority")
    return {"input": inp, "output": out}


def _counterfactuals(inp, info):
    core, surplus = info["core"], info["surplus"]
    n = len(surplus)
    rigid_noise = core | {p for p in surplus if p[0] > 0}
    u0, u1, v0, v1 = _box(core)
    rectangle = {(u, v) for u in range(u0, u1+1) for v in range(v0, v1+1)}
    def alternative(obstacle, path):
        try:
            return _paint(inp, info, obstacle, path)
        except ValueError:
            # A shortcut whose route enters the support is already refuted.
            return None
    return {
        "delete_only": _paint(inp, info, core, []),
        "slack_only": _paint(inp, info, core, _clearance(core, len(info["slack"]))),
        "raw_obstacle": alternative(rigid_noise, _clearance(rigid_noise, len(info["slack"]))),
        "all_material": _paint(inp, info, core, _clearance(core, len(info["material"]))),
        "bounding_box": _paint(inp, info, core, _clearance(rectangle, n)),
    }


def _make_pair(rng, bg, colors, split, lesson, parity, orientation):
    for attempt in range(160):
        approach = ("diagonal" if lesson == "visible" else "parallel") if lesson else str(rng.choice(("parallel", "diagonal")))
        try:
            core, axis = _sample_core(rng, approach, parity, lesson)
            near, far, _, _ = _box(core)
            added = int(rng.integers(4, 7)) if split == "test" else int(rng.integers(2 if lesson == "visible" else 1, 4))
            if lesson:
                length = far + int(rng.integers(4, 8))
            else:
                stage = int(rng.integers(4))
                low, high = ((3, max(4, near+1)), (near+1, far+2), (far+1, far+4), (far+4, far+9))[stage]
                length = int(rng.integers(low, high))
            length = max(length, added+3)
            _require(len(core) >= length+3, "core needs stronger reflected evidence")
            n = length-added
            width = int(rng.integers(4, 8))
            depth = max(int(rng.integers(2, 6)), (n+width-2)//width)
            _require(depth <= 5, "slack too deep")
            support_length = int(rng.integers(depth+2, 9))
            support = {(-i, 0) for i in range(support_length)}
            slack = _slack(rng, n, depth, width, lesson == "visible")
            extra = _additions(rng, core, axis, added, lesson == "visible")
            surplus = slack | extra
            recovered_axis, recovered_core = _supported_core(core | surplus)
            _require(recovered_axis == axis and recovered_core == core, "corruption changes the recovered core")
            pair = _render(rng, core, surplus, support, length, bg, colors, orientation, lesson)
            info = _analyze(pair["input"])
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "construction mismatch")
            if lesson == "visible":
                wrong = _counterfactuals(pair["input"], info)
                _require(all(not np.array_equal(a, pair["output"]) for a in wrong.values()),
                         "lesson does not distinguish a shortcut")
            return pair
        except ValueError:
            continue
    return _fallback_pair(rng, bg, colors, split, lesson, parity, orientation)


def _fallback_pair(rng, bg, colors, split, lesson, parity, orientation):
    """Closed geometric fallback; no unbounded retry or rule change."""
    # Hollow 7-row cap: enough support even with the maximum noise budget.
    core = set()
    s = -parity
    for u in range(7):
        upper = 4-parity-max(0, u-4)
        lower = s-upper
        for v in range(lower, upper+1):
            if u in (0, 6) or v in (lower, upper):
                core.add((u, v))
    approach = "parallel" if lesson == "clipped" else "diagonal"
    dv = -max(v for u, v in core) if approach == "parallel" else 0
    du = 3 if approach == "parallel" else max(v-u for u, v in core)
    core = {(u+du, v+dv) for u, v in core}
    axis = s+2*dv
    extra = {(du, 5-parity+dv), (du+2, 1+dv)}
    if split == "test":
        extra |= {(du-1, 5-parity+dv), (du+1, 5-parity+dv), (du+3, 2+dv)}
    far = max(u for u, v in core)
    length = far+6
    amount = length-len(extra)
    slack = {(0, 1)} | {(-1-i//5, 2+i%5) for i in range(amount-1)}
    support = {(-i, 0) for i in range(7)}
    found_axis, found_core = _supported_core(core | extra | slack)
    _require(found_axis == axis and found_core == core, "invalid fallback symmetry")
    return _render(rng, core, slack | extra, support, length, bg, colors, orientation, lesson)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                 and low <= value <= high, f"illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    parity0, orientation0 = int(rng.integers(2)), int(rng.integers(2))
    backgrounds = list(map(int, rng.permutation(10)))
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            lesson = None
            parity, orientation = int(rng.integers(2)), int(rng.integers(2))
            if split == "train" and index in slots:
                position = slots.index(index)
                lesson = ("visible", "clipped")[position]
                parity, orientation = parity0 ^ position, orientation0 ^ position
            bg = backgrounds[(index + (num_train if split == "test" else 0)) % 10]
            colors = tuple(map(int, rng.choice([c for c in range(10) if c != bg], 2, replace=False)))
            for retry in range(20):
                pair = _make_pair(rng, bg, colors, split, lesson, parity, orientation)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    inp = _grid(pair["input"])
    a = _analyze(inp)
    core, surplus = a["core"], a["surplus"]
    visible = [p for p in a["route"] if _inside(_global(p, a["origin"], a["forward"], a["lateral"]), inp.shape)]
    u0, u1, v0, v1 = _box(core)
    holes = {(u, v) for u in range(u0+1, u1) for v in range(v0+1, v1)} - core
    # Interior row gaps, including supported islands, are the useful hole sites.
    holes = {p for p in holes if min(v for u, v in core if u == p[0]) < p[1] <
             max(v for u, v in core if u == p[0])}
    return {"shape": list(inp.shape), "background": a["background"], "axis_parity": a["axis"] % 2,
            "orientation": "vertical" if a["forward"][0] else "horizontal",
            "approach": "parallel" if max(v for u, v in core) == 0 else "diagonal",
            "core_height": u1-u0+1, "core_width": v1-v0+1, "core_cells": len(core),
            "surplus": len(surplus), "reservoir": len(a["slack"]),
            "forward_surplus": len(surplus-a["slack"]), "visible_thread": len(visible),
            "clipped": len(visible) < len(surplus), "holes": len(holes),
            "surplus_in_holes": len(surplus & holes),
            "diagonal_slack": not _connected(a["slack"], ((1, 0), (-1, 0), (0, 1), (0, -1))),
            "stage": "approach" if len(surplus) <= u0 else "contour" if len(surplus) <= u1+1 else "outgoing",
            "silhouette": tuple(sorted((u-u0, max(v for t, v in core if t == u)-v0)
                                        for u in sorted({t for t, v in core}))),
            "core_structure": tuple(sorted((u-u0, v-v0) for u, v in core))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, lessons, train_metrics = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            assert np.array_equal(solve(inp), out), "output disagrees with input-only recovery"
            assert not np.array_equal(inp, out)
            assert all(out[p] == a["support_color"] for p in a["support"])
            assert all(out[_global(p, a["origin"], a["forward"], a["lateral"])] == a["material_color"] for p in a["core"])
            m = measure_pair(pair)
            assert np.count_nonzero(out == a["material_color"]) == m["core_cells"]+m["visible_thread"]
            assert m["core_cells"]+m["surplus"] == np.count_nonzero(inp == a["material_color"])
            assert set(np.unique(inp)) == set(np.unique(out))
            assert not np.array_equal(_paint(inp, a, a["core"], []), out), "A contributes no relocation"
            assert (1 <= m["forward_surplus"] <= 3) if split == "train" else (4 <= m["forward_surplus"] <= 6)
            if split == "train":
                train_metrics.append(m)
                wrong = _counterfactuals(inp, a)
                if not m["clipped"] and m["stage"] == "outgoing":
                    lessons.append(all(not np.array_equal(pred, out) for pred in wrong.values()) and
                                   m["surplus_in_holes"] > 0 and m["diagonal_slack"])
    assert any(lessons), "missing complete conservation/contour/denoising lesson"
    assert any(m["clipped"] for m in train_metrics), "missing clipping lesson"
    assert {m["axis_parity"] for m in train_metrics} == {0, 1}
    assert {m["orientation"] for m in train_metrics} == {"vertical", "horizontal"}
    assert {m["approach"] for m in train_metrics} == {"diagonal", "parallel"}
    assert len({m["background"] for m in train_metrics}) >= 2


def self_test():
    """Bounded 42-seed structural audit and targeted semantic regressions."""
    shapes, cores, profiles, backgrounds, puzzle_keys = set(), set(), set(), set(), set()
    orientations, parities, stages, approaches = set(), set(), set(), set()
    records, changed, clipped, hole_cases = 0, 0, 0, 0
    budgets, reservoirs, core_sizes, noise_counts = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzle_keys.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = measure_pair(pair)
                records += 1
                changed += not np.array_equal(pair["input"], pair["output"])
                clipped += m["clipped"]
                hole_cases += m["surplus_in_holes"] > 0
                shapes.add(tuple(m["shape"]))
                cores.add(m["core_structure"])
                profiles.add(m["silhouette"])
                backgrounds.add(m["background"])
                orientations.add(m["orientation"])
                parities.add(m["axis_parity"])
                stages.add(m["stage"])
                approaches.add(m["approach"])
                budgets.add(m["surplus"])
                reservoirs.add(m["reservoir"])
                core_sizes.add(m["core_cells"])
                noise_counts.add(m["forward_surplus"])
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s]) for k in p)
            for pair in puzzle["train"]:
                original = pair["input"].copy()
                for turn in range(4):
                    for reflect in (False, True):
                        transform = lambda g: np.fliplr(np.rot90(g, turn)) if reflect else np.rot90(g, turn)
                        assert np.array_equal(solve(transform(original)), transform(pair["output"]))
                permutation = np.random.default_rng(100+seed).permutation(10)
                assert np.array_equal(solve(permutation[original]), permutation[pair["output"]])
                assert np.array_equal(pair["input"], original), "solve modified its input"
        # Remove one unmatched forward cell: the rigid core is unchanged and
        # a fully visible thread becomes exactly one cell shorter.
        lesson = next(p for p in puzzle["train"] if not measure_pair(p)["clipped"] and
                      measure_pair(p)["stage"] == "outgoing")
        a = _analyze(lesson["input"])
        point = min(p for p in a["surplus"] if p[0] > 0)
        modified = lesson["input"].copy()
        modified[_global(point, a["origin"], a["forward"], a["lateral"])] = a["background"]
        b = _analyze(modified)
        assert b["core"] == a["core"] and b["route"] == a["route"][:-1]
        assert np.count_nonzero(solve(modified) == a["material_color"]) == np.count_nonzero(lesson["output"] == a["material_color"])-1
    assert len(puzzle_keys) == 42 and changed == records == 210
    assert len(shapes) >= 70 and len(cores) >= 50 and len(profiles) >= 25
    assert backgrounds == set(range(10)) and parities == {0, 1}
    assert orientations == {"vertical", "horizontal"} and approaches == {"parallel", "diagonal"}
    assert stages == {"approach", "contour", "outgoing"}
    assert noise_counts == set(range(1, 7)) and len(budgets) >= 12 and len(reservoirs) >= 10
    assert clipped >= 42 and hole_cases >= 42 and len(core_sizes) >= 20
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12345, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Direct fallback checks, including both axis parities and split budgets.
    fallback_rng = np.random.default_rng(712)
    for parity in (0, 1):
        for lesson in ("visible", "clipped", None):
            for split in ("train", "test"):
                p = _fallback_pair(fallback_rng, 0, (2, 7), split, lesson, parity, parity)
                assert np.array_equal(solve(p["input"]), p["output"])
    damaged = generate_puzzle(901, 2, 1)
    a = _analyze(damaged["test"][0]["input"])
    damaged["test"][0]["output"][min(a["support"])] = a["background"]
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("validator accepted damaged support")
    return {"seeds": 42, "pairs": records, "distinct_puzzles": len(puzzle_keys), "grid_shapes": len(shapes),
            "core_structures": len(cores), "exposed_profiles": len(profiles), "backgrounds": sorted(backgrounds),
            "thread_budgets": sorted(budgets), "reservoir_counts": sorted(reservoirs), "core_cell_counts": len(core_sizes),
            "forward_surplus_counts": sorted(noise_counts), "clipped_pairs": clipped, "hole_surplus_pairs": hole_cases,
            "stopping_stages": sorted(stages), "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
