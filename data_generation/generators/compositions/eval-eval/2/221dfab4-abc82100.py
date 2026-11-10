"""Phase-addressed embroidery: a clocked lane reads registered stencil fragments.

Only NumPy and the standard library are used. All solving conventions are visible
in one input. No parent generator, seed, or training output is used by solve().
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SIDES = ("top", "right", "bottom", "left")
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=True)


def _canonical(g, side):
    if side == "top":
        return g.copy()
    if side == "bottom":
        return g[::-1].copy()
    if side == "left":
        return g.T.copy()
    return g.T[::-1].copy()


def _orient(g, side):
    if side in ("top", "bottom"):
        return _canonical(g, side)
    if side == "left":
        return g.T.copy()
    return g[::-1].T.copy()


def _phase(depth):
    return -1 if depth % 2 else (1 if depth % 6 == 4 else 0)


def _components(points):
    todo = set(points)
    result = []
    while todo:
        root = min(todo)
        todo.remove(root)
        stack, cells = [root], {root}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in todo:
                        todo.remove(p)
                        cells.add(p)
                        stack.append(p)
        result.append(cells)
    return result


def _edge(g):
    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[counts.argmax()])
    if counts.max() <= g.size / 2:
        raise ValueError("background must occupy more than half the input")
    bars = []
    for color in colors:
        if color == bg:
            continue
        rr, cc = np.where(g == color)
        for side, fixed, boundary, lateral in (
                ("top", rr, 0, cc), ("bottom", rr, g.shape[0] - 1, cc),
                ("left", cc, 0, rr), ("right", cc, g.shape[1] - 1, rr)):
            if np.all(fixed == boundary):
                start = int(lateral.min())
                if np.array_equal(np.sort(lateral), np.arange(start, start + len(lateral))):
                    bars.append((int(color), side, start, len(lateral)))
    if len(bars) != 1:
        raise ValueError("need a unique noncorner edge bar in its own color")
    marker, side, start, length = bars[0]
    return bg, marker, side, start, length


def _instructions(g, bg, marker, neutral):
    """Read every exact neutral component; no template catalogue is consulted."""
    comps = _components(map(tuple, np.argwhere(g == neutral).tolist()))
    if not comps:
        raise ValueError("no stencil")
    records = []
    for cells in comps:
        near = {(r + dr, c + dc) for r, c in cells
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        candidates = []
        for qr, qc in sorted(near):
            if not (0 <= qr < g.shape[0] and 0 <= qc < g.shape[1]):
                continue
            if int(g[qr, qc]) not in (bg, neutral):
                continue
            for dr, dc in DIRS:
                sl, tl = (qr - 2 * dr, qc - 2 * dc), (qr - dr, qc - dc)
                if not (0 <= sl[0] < g.shape[0] and 0 <= sl[1] < g.shape[1]):
                    continue
                source, target = int(g[sl]), int(g[tl])
                if (source in (bg, marker, neutral) or target in (bg, marker, neutral)
                        or source == target):
                    continue
                candidates.append(dict(mask=frozenset(cells), q=(qr, qc), sl=sl, tl=tl,
                                       source=source, target=target, direction=(dr, dc)))
        if len(candidates) != 1:
            raise ValueError("stencil label association is not unique")
        records.append(candidates[0])
    labels = [p for rec in records for p in (rec["sl"], rec["tl"])]
    if len(set(labels)) != len(labels) or len({r["source"] for r in records}) != len(records):
        raise ValueError("shared labels or repeated source instructions")
    return records


def _decode(input_grid):
    original = _grid(input_grid)
    bg, marker, side, start, length = _edge(original)
    g = _canonical(original, side)
    options = []
    for neutral in sorted(set(map(int, np.unique(g))) - {bg, marker}):
        try:
            recs = _instructions(g, bg, marker, neutral)
        except ValueError:
            continue
        options.append((neutral, recs))
    if len(options) != 1:
        raise ValueError("input does not identify one stencil color")
    neutral, records = options[0]
    labels = {p for rec in records for p in (rec["sl"], rec["tl"])}
    for rec in records:
        rec["parts"] = {
            phase: tuple(sorted((r - rec["q"][0], c - rec["q"][1])
                                for r, c in rec["mask"] if _phase(r) == phase))
            for phase in (0, 1)}
        rec["anchors"] = [tuple(map(int, p)) for p in np.argwhere(g == rec["source"])
                          if tuple(p) not in labels]
        rec["active"] = [(p, _phase(p[0])) for p in rec["anchors"]
                         if start <= p[1] < start + length and _phase(p[0]) >= 0]
    return dict(grid=g, bg=bg, marker=marker, side=side, start=start, length=length,
                neutral=neutral, records=records, labels=labels)


def _footprint(anchor, offsets, shape):
    return {(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets
            if 0 <= anchor[0] + dr < shape[0] and 0 <= anchor[1] + dc < shape[1]}


def _render(info):
    g, bg = info["grid"], info["bg"]
    out = g.copy()
    out[g == info["neutral"]] = bg
    for p in info["labels"]:
        out[p] = bg
    paint = {}
    for rec in info["records"]:
        for anchor, phase in rec["active"]:
            out[anchor] = bg
            for p in _footprint(anchor, rec["parts"][phase], g.shape):
                if p in paint and paint[p] != rec["target"]:
                    raise ValueError("incompatible stamp overlap")
                paint[p] = rec["target"]
    # Original anchors are all collected before any painting; no recursion.
    for p, color in paint.items():
        out[p] = color
    return _orient(out, info["side"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone, including its bar, labels, and exact masks."""
    return _render(_decode(input_grid))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _shape(rng, hole=False):
    h, w = int(rng.integers(5, 12)), int(rng.integers(3, 9))
    family = "c" if hole else _choose(rng, ("c", "outline", "comb", "tree", "walk", "wedge"))
    if family in ("c", "outline"):
        cells = {(r, w - 1) for r in range(h)}
        cells |= {(r, c) for r in (0, h - 1) for c in range(w)}
        if family == "outline":
            cells |= {(r, 0) for r in range(h)}
    elif family == "comb":
        cells = {(r, 0) for r in range(h)}
        for r in rng.choice(h, size=int(rng.integers(2, h)), replace=False):
            cells |= {(int(r), c) for c in range(int(rng.integers(2, w + 1)))}
    elif family == "wedge":
        cells = {(r, c) for r in range(h) for c in range(1 + (r * (w - 1)) // (h - 1))}
        if rng.random() < .5:
            cells = {(r, c) for r, c in cells if c == 0 or c == (r * (w - 1)) // (h - 1)}
    else:
        c = int(rng.integers(w))
        cells = set()
        for r in range(h):
            c = int(np.clip(c + rng.integers(-1, 2), 0, w - 1))
            cells.add((r, c))
            if family == "tree" or rng.random() < .4:
                end = int(rng.integers(w))
                cells |= {(r, x) for x in range(min(c, end), max(c, end) + 1)}
    if rng.random() < .5:
        cells = {(r, w - 1 - c) for r, c in cells}
    if rng.random() < .5:
        cells = {(h - 1 - r, c) for r, c in cells}
    return cells


def _registration_options(cells, hole):
    near = {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    options = []
    for q in sorted(near):
        if hole is True and q in cells or hole is False and q not in cells:
            continue
        for dr, dc in DIRS:
            sl, tl = (q[0] - 2 * dr, q[1] - 2 * dc), (q[0] - dr, q[1] - dc)
            if sl in cells or tl in cells:
                continue
            # The reverse label ray must not find another registration.
            reverse_q = (q[0] - 3 * dr, q[1] - 3 * dc)
            if reverse_q in near:
                continue
            options.append((q, (dr, dc), sl, tl))
    return options


def _place_masks(rng, shape, bg, marker, neutral, mappings, hole_kind, q_parity):
    g = np.full(shape, bg, dtype=np.int8)
    occupied, halo, records = set(), set(), []
    for i, (source, target) in enumerate(mappings):
        wanted = hole_kind if i == 0 else None
        placed = False
        for _ in range(120):
            cells = _shape(rng, hole=wanted is True)
            registrations = _registration_options(cells, wanted)
            if not registrations:
                continue
            q, direction, sl, tl = _choose(rng, registrations)
            if i == 0 and q_parity == 1:
                center = ((min(r for r, c in cells) + max(r for r, c in cells)) // 2,
                          (min(c for r, c in cells) + max(c for r, c in cells)) // 2)
                if q == center:
                    continue
            all_cells = cells | {q, sl, tl}
            r0, r1 = min(r for r, c in all_cells), max(r for r, c in all_cells)
            c0, c1 = min(c for r, c in all_cells), max(c for r, c in all_cells)
            if r1 - r0 >= shape[0] - 2 or c1 - c0 >= shape[1] - 2:
                continue
            rshift = int(rng.integers(1 - r0, shape[0] - r1))
            cshift = int(rng.integers(1 - c0, shape[1] - c1))
            move = lambda p: (p[0] + rshift, p[1] + cshift)
            mask = {move(p) for p in cells}
            atq, ats, att = move(q), move(sl), move(tl)
            used = mask | {atq, ats, att}
            if (used & occupied or mask & halo or i == 0 and q_parity is not None
                    and atq[0] % 2 != q_parity):
                continue
            if any(sum(_phase(r) == k for r, c in mask) < (2 if k == 0 else 1)
                   for k in (0, 1)) or not any(_phase(r) == -1 for r, c in mask):
                continue
            candidate = g.copy()
            for p in mask:
                candidate[p] = neutral
            candidate[ats], candidate[att] = source, target
            try:
                parsed = _instructions(candidate, bg, marker, neutral)
            except ValueError:
                continue
            if len(parsed) != i + 1:
                continue
            rec = next(r for r in parsed if r["source"] == source)
            rec["parts"] = {k: tuple(sorted((r - atq[0], c - atq[1])
                                            for r, c in mask if _phase(r) == k)) for k in (0, 1)}
            records.append(rec)
            occupied |= used
            halo |= {(r + dr, c + dc) for r, c in mask
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            g, placed = candidate, True
            break
        if not placed:
            raise ValueError("mask placement failed")
    return g, records, occupied


def _signature(records):
    return tuple((r["source"], r["target"], r["q"], r["mask"], r["sl"], r["tl"])
                 for r in records)


def _make_pair(rng, palette, mappings, side, teaching, first, hole):
    bg, marker, neutral = palette[:3]
    for attempt in range(180):
        # Larger teaching canvases give unmerged complete copies of both fragments.
        low = 24 if teaching else (19 if len(mappings) > 1 else 14)
        shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        try:
            g, records, occupied = _place_masks(rng, shape, bg, marker, neutral,
                                               mappings, hole, (1 if first else 0) if teaching else None)
        except ValueError:
            continue
        length = int(rng.integers(5 if teaching else 1, max(7, shape[1] * 2 // 3)))
        start = int(rng.integers(0, shape[1] - length + 1))
        if length == 1:
            start = int(rng.integers(1, shape[1] - 1))
        g[0, start:start + length] = marker
        bar = {(0, c) for c in range(start, start + length)}
        occupied |= bar
        baseline = _signature(records)
        anchors, paint, planned, dormant = set(), {}, [], []

        def same_parse():
            try:
                return _signature(_instructions(g, bg, marker, neutral)) == baseline
            except ValueError:
                return False

        def place(rec, p, active, complete=False, separate=False):
            if p in occupied or p in anchors or g[p] != bg:
                return False
            phase = _phase(p[0])
            live = start <= p[1] < start + length and phase >= 0
            if live != active:
                return False
            foot = _footprint(p, rec["parts"].get(phase, ()), shape) if active else set()
            if active and (not foot or foot & occupied or foot & set(dormant)):
                return False
            if complete and len(foot) != len(rec["parts"][phase]):
                return False
            if separate and (foot & set(paint) or p in paint):
                return False
            if any(q in paint and paint[q] != rec["target"] for q in foot):
                return False
            if not active and p in paint:
                return False
            g[p] = rec["source"]
            if not same_parse():
                g[p] = bg
                return False
            anchors.add(p)
            if active:
                planned.append((rec, p, phase, foot))
                paint.update({q: rec["target"] for q in foot})
            else:
                dormant.append(p)
            return True

        failed = False
        for i, rec in enumerate(records):
            if teaching and i == 0:
                residues = [0, 2, 4, 4] if first else [0, 2, 4]
            else:
                residues = [int(_choose(rng, (0, 2, 4))) for _ in range(int(rng.integers(1, 7)))]
            for j, residue in enumerate(residues):
                candidates = [(r, c) for r in range(2, shape[0]) if r % 6 == residue
                              for c in range(start, start + length)]
                rng.shuffle(candidates)
                if first and j == 3:
                    prev = [p[0] for rr, p, ph, ff in planned if rr is rec and ph == 1]
                    candidates = [p for p in candidates if p[0] not in prev]
                # First two examples give readable complete separated fragments.
                for p in candidates:
                    if place(rec, p, True, complete=teaching,
                             separate=teaching):
                        break
                else:
                    failed = True
                    break
            if failed:
                break
            # Inactive lane gaps and off-lane even ticks teach both selection gates.
            modes = ("gap", "outside") if teaching else tuple(
                _choose(rng, ("gap", "outside")) for _ in range(int(rng.integers(4))))
            for mode in modes:
                positions = [(r, c) for r in range(1, shape[0]) for c in range(shape[1])
                             if (r % 2 == 1 and start <= c < start + length if mode == "gap"
                                 else r % 2 == 0 and not start <= c < start + length)]
                rng.shuffle(positions)
                if not any(place(rec, p, False) for p in positions):
                    failed = True
                    break
        if failed:
            continue
        if teaching and not first:
            # Keep the reference copies intact, but also demonstrate truncation
            # rather than introducing it for the first time in a two-train test.
            clipped = False
            for rec in records:
                candidates = [(r, c) for r in range(2, shape[0], 2)
                              for c in range(start, start + length)
                              if 0 < len(_footprint((r, c), rec["parts"][_phase(r)], shape))
                              < len(rec["parts"][_phase(r)])]
                rng.shuffle(candidates)
                if any(place(rec, p, True, separate=True) for p in candidates):
                    clipped = True
                    break
            if not clipped:
                continue
        # Optional unrelated pixels include target-only colors, rather than a fixed noise palette.
        inactive = [c for c in palette[3:] if c not in {r["source"] for r in records}]
        for _ in range(int(rng.integers(0, 6))):
            p = (int(rng.integers(1, shape[0])), int(rng.integers(shape[1])))
            if g[p] == bg and p not in occupied and p not in paint:
                g[p] = _choose(rng, inactive)
                if not same_parse():
                    g[p] = bg
        # Build the expected result from sampled geometry and independently decode the input.
        out = g.copy()
        for rec in records:
            for p in rec["mask"] | {rec["sl"], rec["tl"]}:
                out[p] = bg
        for rec, anchor, phase, foot in planned:
            out[anchor] = bg
        for p, color in paint.items():
            out[p] = color
        inp, answer = _orient(g, side), _orient(out, side)
        try:
            info = _decode(inp)
            if not np.array_equal(_render(info), answer):
                continue
        except ValueError:
            continue
        feature = _features(info, answer)
        if teaching and not (feature["without_A_differs"] and feature["without_B_differs"]
                             and feature["transport"] and feature["whole_copy_differs"]):
            continue
        if teaching and not first and not feature["nonrecursive_differs"]:
            continue
        return {"input": inp, "output": answer}
    raise RuntimeError("bounded scene construction exhausted")


def _counts(num_train, num_test):
    for name, n, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
            raise ValueError(f"{name} is out of range or not an integer")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(grid):
    return tuple(grid.shape), grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    colors = palette[3:]
    base = int(rng.integers(4))
    orientation_shift = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen, train_geometries = set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = split == "train" and index < 2
            first = teaching and index == 0
            if first:
                mappings = [(colors[0], colors[1])]
            elif teaching:
                # A changed local target plus a color chain, evaluated simultaneously.
                mappings = [(colors[0], colors[2]), (colors[2], colors[3])]
            else:
                order = list(map(int, rng.permutation(colors)))
                number = int(_choose(rng, (1, 1, 2)))
                shared = number == 2 and rng.random() < .3
                mappings = [(order[i], order[2] if shared else order[i + number]) for i in range(number)]
            side = SIDES[(base + (2 * int(rng.integers(2)) if split == "train" and orientation_shift
                                  else (1 if split == "test" and orientation_shift else int(rng.integers(4))))) % 4]
            if teaching:
                side = SIDES[base]
            hole = False if first else True if teaching else _choose(rng, (False, True, None))
            for _ in range(20):
                pair = _make_pair(rng, palette, mappings, side, teaching, first, hole)
                info = _decode(pair["input"])
                geometries = {_geometry(r["mask"]) for r in info["records"]}
                if _key(pair["input"]) in seen:
                    continue
                if split == "test" and not geometries - train_geometries:
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if split == "train":
                    train_geometries |= geometries
                break
            else:
                raise RuntimeError("could not obtain a fresh distinct scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def _geometry(cells):
    """Translation, rotation and reflection invariant geometry fingerprint."""
    variants = []
    for swap in (False, True):
        for sr in (-1, 1):
            for sc in (-1, 1):
                points = [(sr * (c if swap else r), sc * (r if swap else c)) for r, c in cells]
                r0, c0 = min(r for r, c in points), min(c for r, c in points)
                variants.append(tuple(sorted((r - r0, c - c0) for r, c in points)))
    return min(variants)


def _counterfactual(info, kind):
    """Explicit ablations, compared by pixels rather than by parameter counts."""
    if kind == "recursive":
        out = _canonical(_render(info), info["side"])
        additional, new_anchors = {}, []
        for rec in info["records"]:
            for rr, cc in np.argwhere(out == rec["source"]):
                p = int(rr), int(cc)
                phase = _phase(p[0])
                if info["start"] <= p[1] < info["start"] + info["length"] and phase >= 0:
                    new_anchors.append(p)
                    additional.update({q: rec["target"] for q in
                                       _footprint(p, rec["parts"][phase], out.shape)})
        for p in new_anchors:
            out[p] = info["bg"]
        for p, color in additional.items():
            out[p] = color
        return _orient(out, info["side"])
    out, bg = info["grid"].copy(), info["bg"]
    out[out == info["neutral"]] = bg
    for p in info["labels"]:
        out[p] = bg
    paint = {}
    for rec in info["records"]:
        active = rec["active"] if kind != "without_A" else [(p, 0) for p in rec["anchors"]]
        for p, phase in active:
            out[p] = bg
            if kind == "without_B":
                offsets = ((0, 0),)
            elif kind in ("whole", "without_A"):
                offsets = tuple((r - rec["q"][0], c - rec["q"][1]) for r, c in rec["mask"])
            else:
                raise ValueError("unknown ablation")
            paint.update({q: rec["target"] for q in _footprint(p, offsets, out.shape)})
    for p, color in paint.items():
        out[p] = color
    return _orient(out, info["side"])


def _features(info, answer):
    g, records = info["grid"], info["records"]
    visits, clips, transport = {}, 0, False
    residues, dormant_gap, dormant_outside = set(), 0, 0
    for rec in records:
        for p, phase in rec["active"]:
            residues.add(p[0] % 6)
            foot = _footprint(p, rec["parts"][phase], g.shape)
            clips += len(foot) < len(rec["parts"][phase])
            transport |= any(_phase(q[0]) != phase for q in foot)
            for q in foot:
                visits[q] = visits.get(q, 0) + 1
        live = {p for p, phase in rec["active"]}
        for p in set(rec["anchors"]) - live:
            dormant_gap += info["start"] <= p[1] < info["start"] + info["length"] and p[0] % 2 == 1
            dormant_outside += not info["start"] <= p[1] < info["start"] + info["length"] and p[0] % 2 == 0
    return dict(side=info["side"], shape=tuple(g.shape), length=info["length"],
                sources=len(records), residues=residues, dormant_gap=int(dormant_gap),
                dormant_outside=int(dormant_outside), transport=bool(transport), clips=int(clips),
                overlaps=sum(n > 1 for n in visits.values()),
                holes=sum(rec["q"] not in rec["mask"] for rec in records),
                active=sum(len(rec["active"]) for rec in records),
                without_A_differs=not np.array_equal(answer, _counterfactual(info, "without_A")),
                without_B_differs=not np.array_equal(answer, _counterfactual(info, "without_B")),
                nonrecursive_differs=not np.array_equal(answer, _counterfactual(info, "recursive")),
                whole_copy_differs=not np.array_equal(answer, _counterfactual(info, "whole")))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, teaching, geometries = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(solve(inp), out), "incorrect output"
            assert not np.array_equal(inp, out)
            assert np.array_equal(out[inp == info["marker"]], inp[inp == info["marker"]]), "bar changed"
            feature = _features(info, out)
            # A mask must really contain distinct sampled and discarded material.
            for rec in info["records"]:
                assert all(rec["parts"][k] for k in (0, 1))
                assert any(_phase(r) == -1 for r, c in rec["mask"])
                assert rec["anchors"] and rec["active"]
            if split == "train":
                teaching.append(feature)
                geometries |= {_geometry(rec["mask"]) for rec in info["records"]}
            else:
                assert {_geometry(rec["mask"]) for rec in info["records"]} - geometries, "test needs new geometry"
    assert set.union(*(f["residues"] for f in teaching)) == {0, 2, 4}
    assert all(any(f[key] for f in teaching) for key in (
        "holes", "dormant_gap", "dormant_outside", "transport", "without_A_differs",
        "without_B_differs", "whole_copy_differs", "nonrecursive_differs", "clips")), "missing teaching evidence"


def self_test():
    """Bounded 42-seed structural, semantic-witness, equivariance and API audit."""
    # Hand-computed hole registration: ordinary offsets (0,1),(2,1),
    # special offset (-2,1). The last ordinary copy clips its lower cell.
    g = np.zeros((15, 18), dtype=np.int8)
    g[0, 10:15] = 1
    g[3, 2:7] = g[9, 2:7] = 8
    g[3:10, 6] = 8
    g[6, 3:5] = (2, 3)
    for p in ((6, 11), (10, 13), (2, 10), (14, 14), (5, 12), (4, 16)):
        g[p] = 2
    expected = np.zeros_like(g)
    expected[0, 10:15] = 1
    expected[5, 12] = expected[4, 16] = 2
    for p in ((6, 12), (8, 12), (8, 14), (2, 11), (4, 11), (14, 15)):
        expected[p] = 3
    assert np.array_equal(solve(g), expected), "manual registration/phase/clipping regression"
    signatures, shapes, masks, sides, lengths = set(), set(), set(), set(), set()
    features, role_colors = [], [set(), set(), set()]
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(pair["input"]) for s in p for pair in p[s]))
        for s in p:
            for pair in p[s]:
                info = _decode(pair["input"])
                features.append(_features(info, pair["output"]))
                shapes.add(tuple(pair["input"].shape))
                masks |= {_geometry(rec["mask"]) for rec in info["records"]}
                sides.add(info["side"])
                lengths.add(info["length"])
                for collection, role in zip(role_colors, ("bg", "marker", "neutral")):
                    collection.add(info[role])
        # Direction is geometric, including reflection, rather than a stored seed parameter.
        pair = p["test"][0]
        assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
        assert np.array_equal(solve(pair["input"][:, ::-1]), pair["output"][:, ::-1])
    assert len(signatures) == 42 and len(shapes) >= 50 and len(masks) >= 70
    assert sides == set(SIDES) and len(lengths) >= 10
    assert all(len(c) >= 8 for c in role_colors)
    assert any(f["clips"] for f in features) and any(f["overlaps"] for f in features)
    assert {f["sources"] for f in features} == {1, 2}
    assert any(not f["holes"] for f in features) and any(f["holes"] for f in features)
    assert any(f["active"] == 1 for f in features) and max(f["active"] for f in features) >= 7
    # Determinism, every explicit count combination, and bad types/ranges.
    a, b = generate_puzzle(103, 2, 1), generate_puzzle(103, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            q = generate_puzzle(1000 + nt * 10 + ns, nt, ns)
            assert len(q["train"]) == nt and len(q["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(77, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "stencil_geometries": len(masks), "sides": sorted(sides),
            "clipped_pairs": sum(bool(f["clips"]) for f in features),
            "overlapping_pairs": sum(bool(f["overlaps"]) for f in features)}


if __name__ == "__main__":
    report = self_test()
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self_test.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report, separators=(",", ":")))
