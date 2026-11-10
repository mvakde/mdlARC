"""Registered stamps: boundary projection induces parity classes of map sites.

Only NumPy and the standard library are used. Decoding and prediction consume
pixels, not construction records. See task_spec.md for the complete convention.
"""
from __future__ import annotations

import json

import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= g) & (g <= 9))), "invalid grid color")
    return g.astype(np.int8, copy=True)


def _components(g, bg):
    remaining = {tuple(map(int, p)) for p in np.argwhere(g != bg)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _decode(g):
    """Identify a marked connected glyph, isolated edge guides, and its map."""
    hgrid, wgrid = g.shape
    colors, counts = np.unique(g, return_counts=True)
    _require(len(colors) == 3, "exactly three local color roles are required")
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > g.size / 2, "background must occupy over half the input")
    components = _components(g, bg)
    glyphs = [x for x in components if len(x) > 1]
    _require(len(glyphs) == 1, "there must be one connected nonsingleton glyph")
    glyph = glyphs[0]
    local_colors, local_counts = np.unique([int(g[p]) for p in sorted(glyph)], return_counts=True)
    _require(len(local_colors) == 2 and sorted(map(int, local_counts)) == [1, len(glyph) - 1]
             and len(glyph) >= 3, "glyph needs one handle and at least two body cells")
    marker = int(local_colors[np.argmin(local_counts)])
    ink = int(local_colors[np.argmax(local_counts)])
    handle = next(p for p in glyph if int(g[p]) == marker)
    top, left = min(r for r, _ in glyph), min(c for _, c in glyph)
    th, tw = max(r for r, _ in glyph) - top + 1, max(c for _, c in glyph) - left + 1
    _require(th >= 2 and tw >= 2, "glyph bounding box must span both axes")
    cells = tuple((r - handle[0], c - handle[1], int(g[r, c])) for r, c in sorted(glyph))
    mask = tuple(sorted((r - top, c - left) for r, c in glyph))
    anchors, points = {}, []
    for component in components:
        if len(component) != 1:
            continue
        r, c = next(iter(component))
        if r in (0, hgrid - 1) or c in (0, wgrid - 1):
            _require(not (r in (0, hgrid - 1) and c in (0, wgrid - 1)), "corner guide is ambiguous")
            anchors.setdefault(int(g[r, c]), []).append((r, c))
        else:
            points.append((r, c))
    _require(set(anchors) == {ink, marker}, "both glyph colors need edge guides")
    systems = {}
    for color, positions in anchors.items():
        horizontal = len({r for r, _ in positions}) == 1 and all(c in (0, wgrid - 1) for _, c in positions)
        vertical = len({c for _, c in positions}) == 1 and all(r in (0, hgrid - 1) for r, _ in positions)
        _require(horizontal != vertical and 1 <= len(positions) <= 2, "inconsistent edge guide")
        axis = 0 if horizontal else 1  # changed coordinate
        guide = positions[0][axis]
        systems[color] = {"axis": axis, "guide": guide, "anchors": tuple(sorted(positions))}
    exceptional = [p for p in points if int(g[p]) == ink]
    _require(len(exceptional) == 1 and len(points) >= 2, "map needs one body-colored anchor and ordinary sites")
    anchor = exceptional[0]
    _require(all((r - anchor[0]) % 2 == 0 and (c - anchor[1]) % 2 == 0 for r, c in points),
             "map must use a pitch-two lattice")
    pr, pc = min(r for r, _ in points), min(c for _, c in points)
    br, bc = max(r for r, _ in points), max(c for _, c in points)
    _require(br + 1 < top or top + th < pr or bc + 1 < left or left + tw < pc,
             "map and glyph boxes need a blank separating row or column")
    coarse = tuple(sorted(((r - pr) // 2, (c - pc) // 2) for r, c in points))
    return {"shape": g.shape, "bg": bg, "ink": ink, "marker": marker,
            "top": top, "left": left, "th": th, "tw": tw, "mask": mask,
            "handle": handle, "handle_offset": (handle[0] - top, handle[1] - left),
            "cells": cells, "systems": systems, "points": tuple(sorted(points)),
            "anchor": anchor, "coarse": coarse,
            "map_dims": ((br - pr) // 2 + 1, (bc - pc) // 2 + 1),
            "map_anchor": ((anchor[0] - pr) // 2, (anchor[1] - pc) // 2)}


def _records(m, *, align=True, center=False, pitch=None, compress=False, reference=None):
    """Projection creates equivalence classes; no intermediate expanded grid."""
    groups = {}
    h, w = (m["th"], m["tw"]) if pitch is None else pitch
    reference = m["anchor"] if reference is None else reference
    ranks = [{v: i for i, v in enumerate(sorted({p[axis] for p in m["points"]}))} for axis in (0, 1)]
    for point in m["points"]:
        special = point == m["anchor"]
        color = m["ink"] if special else m["marker"]
        if compress:
            dr, dc = (ranks[axis][point[axis]] - ranks[axis][m["anchor"][axis]] for axis in (0, 1))
        else:
            dr, dc = ((point[axis] - reference[axis]) // 2 for axis in (0, 1))
        raw = [m["handle"][0] + dr * h, m["handle"][1] + dc * w]
        position = raw.copy()
        system = m["systems"][color]
        axis = system["axis"]
        if align:
            alignment_cell = raw[axis]
            if center:
                alignment_cell += ((m["th"], m["tw"])[axis] - 1) // 2 - m["handle_offset"][axis]
            position[axis] += system["guide"] - alignment_cell
        key = color, tuple(position)
        groups[key] = groups.get(key, 0) + 1
    return tuple((color, p, n, color == m["ink"]) for (color, p), n in sorted(groups.items()))


def _render(m, records, *, parity=True, singles=False, inversion=True, strict=True):
    out = np.full(m["shape"], m["bg"], dtype=np.int8)
    claimed = set()
    for color, system in m["systems"].items():
        for p in system["anchors"]:
            out[p] = color
            claimed.add(p)
    for color, position, multiplicity, special in records:
        if (singles and multiplicity != 1) or (parity and multiplicity % 2 == 0):
            continue
        axis = m["systems"][color]["axis"]
        if strict:
            # Only the projected, perpendicular coordinate may be clipped.
            offsets = [cell[1 - axis] for cell in m["cells"]]
            _require(0 <= position[1 - axis] + min(offsets)
                     and position[1 - axis] + max(offsets) < m["shape"][1 - axis],
                     "stamp extends beyond its preserved coordinate")
        for dr, dc, value in m["cells"]:
            r, c = position[0] + dr, position[1] + dc
            if not (0 <= r < m["shape"][0] and 0 <= c < m["shape"][1]):
                continue
            if strict:
                _require((r, c) not in claimed, "surviving stamps or fixed anchors overlap")
            claimed.add((r, c))
            out[r, c] = (m["marker"] if value == m["ink"] else m["ink"]) if special and inversion else value
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    m = _decode(g)
    return _render(m, _records(m))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _mask(rng, h, w, informative):
    """Connected tight masks from spanning paths and finite frontier growth."""
    cells = {(0, 0)}
    r = c = 0
    steps = [0] * (h - 1) + [1] * (w - 1)
    rng.shuffle(steps)
    for axis in steps:
        r += axis == 0
        c += axis == 1
        cells.add((r, c))
    goal = int(rng.integers(len(cells), h * w + (0 if informative else 1)))
    while len(cells) < goal:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS
                           if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
        cells.add(_pick(rng, frontier))
    if rng.integers(2):
        cells = {(r, w - 1 - c) for r, c in cells}
    if rng.integers(2):
        cells = {(h - 1 - r, c) for r, c in cells}
    return sorted(cells)


def _layout(rng, u, v, teaching, test, gap, simple=False, previous=None):
    """Rows are collapsed; columns are retained in canonical coordinates."""
    if teaching:
        counts = [1, 2, 3] + [int(rng.integers(1, min(3, u) + 1)) for _ in range(v - 3)]
    elif test:
        counts = [1, 2, 4] + [int(rng.integers(1, u + 1)) for _ in range(v - 3)]
    else:
        counts = [int(rng.integers(1, min(3, u) + 1)) for _ in range(v)]
    if v == 1:
        counts[0] = int(rng.integers(1, min(3, u - 1) + 1))
    rng.shuffle(counts)
    if gap:
        # Retain all parity witnesses, with an internal empty physical band.
        if teaching or test:
            if teaching:
                # Odd witnesses on opposite sides make coordinate compression
                # observably wrong even if the anchor is next to the gap.
                middle = [2] + [int(rng.integers(1, min(3, u) + 1)) for _ in range(v - 4)]
                rng.shuffle(middle)
                counts = [1] + middle + [3]
                if rng.integers(2):
                    counts.reverse()
            else:
                counts = [1, 2, 4] + [int(rng.integers(1, u + 1)) for _ in range(v - 4)]
                rng.shuffle(counts)
            counts.insert(int(rng.integers(1, v - 1)), 0)
        else:
            counts[int(rng.integers(1, v - 1))] = 0
    options = [c for c in range(v) if 0 < counts[c] < u]
    if not options:  # A simple one-row map can put its exceptional point at an end.
        ac = int(rng.integers(v))
        counts[ac] = 0
    else:
        candidates = [(r, c) for r in range(u) for c in options
                      if (previous is None or (r != previous[0] and c != previous[1]))
                      and (not teaching or (r, c) != ((u - 1) // 2, (v - 1) // 2))]
        ar, ac = _pick(rng, candidates)
    if not options:
        ar = int(rng.integers(u))
    occupied = {(ar, ac)}
    tight_column = next((c for c, n in enumerate(counts) if n >= 2 and c != ac), None)
    for c, n in enumerate(counts):
        choices = [r for r in range(u) if (r, c) != (ar, ac)]
        required_rows = [0, u - 1] if c == tight_column and u > 1 else []
        for r in required_rows:
            occupied.add((r, c))
        choices = [r for r in choices if r not in required_rows]
        for r in rng.choice(choices, size=n - len(required_rows), replace=False):
            occupied.add((int(r), c))
    return occupied, (ar, ac)


def _scene(rng, colors, h, w, ordinary_axis, mixed, teaching, test, gap, clip, previous_anchor=None):
    # All geometry below is canonical: ordinary guide is horizontal. Transpose
    # the entire scene at the end when it is vertical.
    u = int(rng.integers(4, 7) if test else rng.integers(3, 6))
    min_v = 4 if gap or test else 3
    max_v = min(6 if test else 5, 26 // w)
    v = int(rng.integers(min_v, max_v + 1))
    simple = not teaching and not test and not gap and rng.random() < .18
    if simple:
        if rng.integers(2):
            v = 1
        else:
            u = 1
            v = int(rng.integers(2, min(5, 26 // w) + 1))
    previous = previous_anchor[::-1] if previous_anchor is not None and ordinary_axis else previous_anchor
    occupied, anchor = _layout(rng, u, v, teaching, test, gap, simple, previous)
    ph, pw = 2 * u - 1, 2 * v - 1
    height_min = max(ph + h + 3, 2 * h + 2)
    # Even one-column maps leave a guide-handle coordinate outside the source
    # glyph's column span, so clipped mixed scenes always have an isolated end.
    width_min = max(v * w + 2, pw + 5, 2 * w + 2)
    gh = int(rng.integers(height_min, min(30, height_min + 7) + 1))
    gw = int(rng.integers(width_min, min(30, width_min + 7) + 1))
    cells = _mask(rng, h, w, informative=teaching)
    # Exclude the integer center used by the explicit centering shortcut,
    # including its floor choice for even-height masks.
    marks = [p for p in cells if p[0] != (h - 1) // 2]
    if clip:
        marks = [p for p in marks if p[0] >= 2 or h - 1 - p[0] >= 2]
    mark = _pick(rng, marks)
    mr, mc = mark
    guides = ([1] if mr >= 2 else []) + ([gh - 2] if h - 1 - mr >= 2 else []) if clip else list(range(1 + mr, gh - h + mr))
    # Sample feasible source/guide pairs; the separate source band guarantees
    # no contradictory mixed-guide overlap. This is packing, not a solving cue.
    source_options = []
    for st in range(1, gh - h):
        map_rows = [p for p in range(1, gh - ph) if p + ph < st or st + h < p]
        if not map_rows:
            continue
        for guide in guides:
            ot = guide - mr
            if ot + h <= st or st + h <= ot:
                source_options.append((st, guide, map_rows))
    st, guide, map_rows = _pick(rng, source_options)
    sl = int(rng.integers(1 + anchor[1] * w, gw - (v - anchor[1]) * w))
    if mixed:
        # Exceptional copy is aligned vertically. Its retained row stays in
        # the source band, disjoint from ordinary horizontal stamps.
        ink_guides = list(range(1 + mc, gw - w + mc))
    else:
        ink_guides = [p for p in range(1 + mr, gh - h + mr)
                      if abs(p - guide) >= h and abs(p - guide) >= 2]
    bg, ink, marker = colors
    inp = np.full((gh, gw), bg, dtype=np.int8)
    for r, c in cells:
        inp[st + r, sl + c] = marker if (r, c) == mark else ink
    ordinary_counts = {c: sum((r, c) in occupied and (r, c) != anchor for r in range(u)) for c in range(v)}
    target_cells = {(guide + r - mr, sl + (column - anchor[1]) * w + c)
                    for column, n in ordinary_counts.items() if n % 2 for r, c in cells}
    source_cells = {(st + r, sl + c) for r, c in cells}
    source_blocked = source_cells | {(r + dr, c + dc) for r, c in source_cells for dr, dc in _DIRS}
    ink_guides = [p for p in ink_guides if any(q not in source_blocked and q not in target_cells
                  for q in (((0, p), (gh - 1, p)) if mixed else ((p, 0), (p, gw - 1))))]
    ink_guide = _pick(rng, ink_guides)
    special_top, special_left = (st, ink_guide - mc) if mixed else (ink_guide - mr, sl)
    target_cells.update((special_top + r, special_left + c) for r, c in cells)
    # Choose one or both ends independently for each guide.
    for color, axis, coordinate in ((marker, 0, guide), (ink, 1 if mixed else 0, ink_guide)):
        positions = [(coordinate, 0 if end == 0 else gw - 1) if axis == 0
                     else (0 if end == 0 else gh - 1, coordinate) for end in (0, 1)]
        allowed = [end for end, (r, c) in enumerate(positions)
                   if (r, c) not in target_cells and inp[r, c] == bg
                   and all(not (0 <= r + dr < gh and 0 <= c + dc < gw)
                                             or inp[r + dr, c + dc] == bg for dr, dc in _DIRS)]
        ends = tuple(allowed) if rng.random() < .35 else (_pick(rng, allowed),)
        for end in ends:
            inp[positions[end]] = color
    # Enumerate valid isolated-map placements, then sample. Never repair a
    # rendered answer or choose an output rule using a latent generation flag.
    candidates = []
    foreground = {tuple(map(int, p)) for p in np.argwhere(inp != bg)}
    blocked = foreground | {(r + dr, c + dc) for r, c in foreground for dr, dc in _DIRS}
    for mt in map_rows:
        for ml in range(2, gw - pw - 1):
            if all((mt + 2 * r, ml + 2 * c) not in blocked for r, c in occupied):
                candidates.append((mt, ml))
    mt, ml = _pick(rng, candidates)
    for r, c in sorted(occupied):
        inp[mt + 2 * r, ml + 2 * c] = ink if (r, c) == anchor else marker
    if ordinary_axis:
        inp = inp.T.copy()
    return {"input": inp, "output": solve(inp)}


def _signature(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, size=3, replace=False)))
    stable_palette = bool(rng.random() < .25)
    clipping = bool(rng.random() < .5)
    gaps = bool(rng.random() < .55)
    base_axis = int(rng.integers(2))
    first_mixed = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    previous_dims = None
    previous_anchor = None
    for i in range(nt + nq):
        test, teaching = i >= nt, i < 2
        split = "test" if test else "train"
        axis = base_axis if i == 0 else 1 - base_axis if i == 1 else int(rng.integers(2))
        mixed = first_mixed if i == 0 else not first_mixed if i == 1 else bool(rng.integers(2))
        gap = gaps and (i == 1 or (not teaching and rng.random() < .45))
        clip = clipping and (i == 1 or (not teaching and rng.random() < .45))
        dimensions = [(h, w) for h in range(2, 6) for w in range(2, 6)
                      if (not teaching or h != w)
                      and (not teaching or w >= 3)
                      and (not clip or h >= 3)
                      and (not teaching or i != 1 or
                           ((w if axis else h) != previous_dims[0]
                            and (h if axis else w) != previous_dims[1]))]
        for _ in range(40):
            h, w = _pick(rng, dimensions)
            colors = palette if stable_palette else tuple(map(int, rng.choice(10, size=3, replace=False)))
            pair = _scene(rng, colors, h, w, axis, mixed, teaching, test, gap, clip,
                          previous_anchor if i == 1 else None)
            signature = _signature(pair["input"])
            if signature not in seen:
                break
        else:
            # 720 visible role assignments exceed the 13 previous inputs. This
            # fallback changes presentation only, while retaining the same rule.
            for bg in range(10):
                for ink in range(10):
                    for marker in range(10):
                        if len({bg, ink, marker}) != 3:
                            continue
                        remap = np.arange(10, dtype=np.int8)
                        remap[list(colors)] = (bg, ink, marker)
                        candidate = {k: remap[v] for k, v in pair.items()}
                        signature = _signature(candidate["input"])
                        if signature not in seen:
                            pair = candidate
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
        seen.add(signature)
        puzzle[split].append(pair)
        if i == 0:
            previous_dims = (w, h) if axis else (h, w)
            previous_anchor = _decode(pair["input"])["map_anchor"]
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(m, name):
    if name == "no_A":
        return _render(m, _records(m, align=False), strict=False)
    if name == "no_B":
        p = list(m["handle"])
        system = m["systems"][m["marker"]]
        p[system["axis"]] = system["guide"]
        return _render(m, ((m["marker"], tuple(p), 1, False),), strict=False)
    if name == "union":
        return _render(m, _records(m), parity=False, strict=False)
    if name == "singletons_only":
        return _render(m, _records(m), singles=True, strict=False)
    if name == "no_inversion":
        return _render(m, _records(m), inversion=False, strict=False)
    if name == "center":
        return _render(m, _records(m, center=True), strict=False)
    if name == "pitch_two":
        return _render(m, _records(m, pitch=(2, 2)), strict=False)
    if name == "compress":
        return _render(m, _records(m, compress=True), strict=False)
    if name in ("map_center", "map_corner"):
        origin = tuple(min(p[axis] for p in m["points"]) for axis in (0, 1))
        reference = tuple(origin[axis] + (2 * ((m["map_dims"][axis] - 1) // 2)
                                         if name == "map_center" else 0) for axis in (0, 1))
        return _render(m, _records(m, reference=reference), strict=False)
    raise ValueError("unknown alternative")


def pair_features(pair):
    g, out = _grid(pair["input"]), _grid(pair["output"])
    m = _decode(g)
    records = _records(m)
    ordinary = [x for x in records if not x[3]]
    axis = m["systems"][m["marker"]]["axis"]
    retained = 1 - axis
    active = [x for x in records if x[2] % 2]
    clipped = sum(any(not (0 <= p[0] + dr < g.shape[0] and 0 <= p[1] + dc < g.shape[1])
                      for dr, dc, _ in m["cells"]) for _, p, _, _ in active)
    used = {p[retained] for p in m["coarse"]}
    return {"shape": g.shape, "roles": (m["bg"], m["ink"], m["marker"]),
            "template_dims": (m["th"], m["tw"]), "mask": m["mask"],
            "handle_offset": m["handle_offset"], "map_dims": m["map_dims"],
            "map_anchor": m["map_anchor"], "layout": m["coarse"], "sites": len(m["points"]),
            "ordinary_axis": axis, "mixed": axis != m["systems"][m["ink"]]["axis"],
            "multiplicities": tuple(sorted(x[2] for x in ordinary)),
            "survivors": len(active), "canceled_groups": sum(x[2] % 2 == 0 for x in ordinary),
            "clipped": clipped, "gap": len(used) < m["map_dims"][retained],
            "anchors": tuple((color, p) for color, s in sorted(m["systems"].items()) for p in s["anchors"]),
            "matches": np.array_equal(_render(m, records), out)}


def validate_generated_puzzle(puzzle) -> None:
    """Semantic checks from input pixels, plus demonstration sufficiency checks."""
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
        _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
        _counts(len(puzzle["train"]), len(puzzle["test"]))
        seen, features = set(), {"train": [], "test": []}
        rejected = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
                g, out = _grid(pair["input"]), _grid(pair["output"])
                _require(g.shape == out.shape, "output canvas changed")
                signature = _signature(g)
                _require(signature not in seen, "duplicate input")
                seen.add(signature)
                m, f = _decode(g), pair_features(pair)
                _require(f["matches"], "incorrect registered-parity output")
                _require(not np.array_equal(g, out), "unchanged pair")
                # Independent counting invariant checks clipping, parity, and
                # fixed markers without relying solely on whole-grid equality.
                expected_count = sum(len(s["anchors"]) for s in m["systems"].values())
                for _, p, n, _ in _records(m):
                    if n % 2:
                        expected_count += sum(0 <= p[0] + dr < g.shape[0] and 0 <= p[1] + dc < g.shape[1]
                                              for dr, dc, _ in m["cells"])
                _require(int(np.count_nonzero(out != m["bg"])) == expected_count, "wrong parity foreground count")
                for color, s in m["systems"].items():
                    _require(all(int(out[p]) == color for p in s["anchors"]), "fixed anchor changed")
                features[split].append(f)
                if split == "train":
                    for name in ("no_A", "no_B", "union", "singletons_only", "no_inversion", "center",
                                 "pitch_two", "compress", "map_center", "map_corner"):
                        if not np.array_equal(_alternative(m, name), out):
                            rejected.add(name)
        required = {"no_A", "no_B", "union", "singletons_only", "no_inversion", "center", "pitch_two",
                    "map_center", "map_corner"}
        _require(required <= rejected, "demonstrations admit a specified shortcut")
        train = features["train"]
        _require({f["ordinary_axis"] for f in train} == {0, 1}, "training must establish both projected axes")
        _require(len({f["template_dims"][0] for f in train}) >= 2
                 and len({f["template_dims"][1] for f in train}) >= 2, "training must vary both tight dimensions")
        _require(len({f["map_anchor"][0] for f in train}) >= 2
                 and len({f["map_anchor"][1] for f in train}) >= 2, "vary both exceptional-anchor coordinates")
        _require(any(f["mixed"] for f in train) and any(not f["mixed"] for f in train), "teach mixed and parallel guides")
        if any(f["gap"] for f in features["test"]):
            _require("compress" in rejected and any(f["gap"] for f in train), "test gaps need a distinguishing demonstration")
        if any(f["clipped"] for f in features["test"]):
            _require(any(f["clipped"] for f in train), "test clipping needs a demonstration")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise AssertionError(str(exc)) from exc


def _fingerprint(puzzle):
    return tuple((_signature(p["input"]), _signature(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def _fixture():
    """Hand specified counterexample: ordinary column counts 1, 2, and 3."""
    inp = np.zeros((15, 19), dtype=np.int8)
    cells = ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2))
    for r, c in cells:
        inp[9 + r, 10 + c] = 7 if (r, c) == (2, 0) else 2
    inp[3, 0], inp[11, 18], inp[1, 2] = 7, 2, 2
    for p in ((5, 2), (1, 4), (3, 4), (1, 6), (3, 6), (5, 6)):
        inp[p] = 7
    expected = np.zeros_like(inp)
    expected[3, 0], expected[11, 18] = 7, 2
    for r, c in cells:
        expected[9 + r, 10 + c] = 2 if (r, c) == (2, 0) else 7
        for left in (10, 16):
            expected[1 + r, left + c] = 7 if (r, c) == (2, 0) else 2
    return inp, expected, cells


def self_test() -> dict:
    puzzles, fs = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        assert all(max(pair_features(pair)["multiplicities"], default=0) <= 3 for pair in p["train"])
        assert all(max(pair_features(pair)["multiplicities"]) >= 4
                   and pair_features(pair)["survivors"] >= 2 for pair in p["test"])
        fs.extend(pair_features(pair) for split in ("train", "test") for pair in p[split])
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    shapes = {f["shape"] for f in fs}
    masks = {(f["template_dims"], f["mask"]) for f in fs}
    maps = {(f["map_dims"], f["layout"], f["map_anchor"]) for f in fs}
    assert len(shapes) >= 50 and len(masks) >= 60 and len(maps) >= 100
    assert {f["ordinary_axis"] for f in fs} == {0, 1}
    assert {f["mixed"] for f in fs} == {False, True}
    assert {1, 2, 3, 4, 5} <= {n for f in fs for n in f["multiplicities"]}
    assert {2, 3, 4, 5} <= {d for f in fs for d in f["template_dims"]}
    assert {f["roles"][0] for f in fs} == set(range(10))
    assert sum(f["clipped"] > 0 for f in fs) >= 10
    assert sum(f["gap"] for f in fs) >= 10
    assert len({f["map_anchor"] for f in fs}) >= 15
    sides = set()
    for f in fs:
        for _, (r, c) in f["anchors"]:
            sides.update(s for s, hit in (("top", r == 0), ("bottom", r == f["shape"][0] - 1),
                                         ("left", c == 0), ("right", c == f["shape"][1] - 1)) if hit)
    assert sides == {"top", "bottom", "left", "right"}
    for counts in ((2, 1), (10, 4), (7, 3)):
        p = generate_puzzle(320 + sum(counts), *counts)
        assert (len(p["train"]), len(p["test"])) == counts
        assert _fingerprint(p) == _fingerprint(generate_puzzle(320 + sum(counts), *counts))
    # Regression: two even-sided teaching glyphs formerly let floor-centering
    # coincide with their handles despite geometric eccentricity.
    validate_generated_puzzle(generate_puzzle(17748233315884, 2, 1))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Palette and D4 equivariance are checked on input-only predictions.
    permutation = np.random.default_rng(818).permutation(10)
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        for turns in (1, 2, 3):
            g = permutation[np.rot90(pair["input"], turns)]
            expected = permutation[np.rot90(pair["output"], turns)]
            assert np.array_equal(solve(g), expected)
    # A manually painted answer, not a second call to the construction loop.
    inp, expected, cells = _fixture()
    assert np.array_equal(solve(inp), expected)
    four = inp.copy()
    four[7, 6] = 7
    four_answer = expected.copy()
    four_answer[1:4, 16:19] = 0
    assert np.array_equal(solve(four), four_answer)
    five = four.copy()
    five[9, 6] = 7
    assert np.array_equal(solve(five), expected)
    swapped = inp.copy()
    swapped[1, 2], swapped[3, 4] = 7, 2
    assert not np.array_equal(solve(swapped), expected)
    clipped = inp.copy()
    clipped[3, 0], clipped[1, 0] = 0, 7
    clip_answer = expected.copy()
    clip_answer[1:4, 10:19] = 0
    clip_answer[3, 0], clip_answer[1, 0] = 0, 7
    for r, c in cells:
        if r >= 1:
            for left in (10, 16):
                clip_answer[r - 1, left + c] = 7 if (r, c) == (2, 0) else 2
    assert np.array_equal(solve(clipped), clip_answer)
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": len(fs), "canvas_shapes": len(shapes),
            "template_masks": len(masks), "anchored_maps": len(maps),
            "clipped_pairs": sum(f["clipped"] > 0 for f in fs),
            "gap_pairs": sum(f["gap"] for f in fs), "anchor_positions": len({f["map_anchor"] for f in fs})}


if __name__ == "__main__":
    # Optional bounded diagnostics. All stored JSON is compact.
    print(json.dumps(self_test(), separators=(",", ":")))
