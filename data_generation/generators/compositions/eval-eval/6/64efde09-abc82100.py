"""Programmable stencil projectors. Only NumPy and the standard library.

The constructor knows sampled placements. The decoder/solve rediscover bars,
cartridges, boundary code and receivers from input pixels alone.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

NEUTRAL = 8
RECEIVER = 5


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((a >= 0) & (a <= 9))), "bad colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _vector(v, t):
    r, c = v
    if t >= 4:
        c = -c
    for _ in range(t % 4):
        r, c = -c, r
    return r, c


def _inverse(v, t):
    r, c = v
    for _ in range(t % 4):
        r, c = c, -r
    return (r, -c) if t >= 4 else (r, c)


def _add(a, b, factor=1):
    return a[0] + factor * b[0], a[1] + factor * b[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _components8(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in remaining:
                        remaining.remove(q)
                        group.add(q)
                        stack.append(q)
        result.append(group)
    return result


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    _need(np.count_nonzero(counts == counts.max()) == 1, "ambiguous background")
    b = int(counts.argmax())
    _need(b not in (NEUTRAL, RECEIVER), "reserved background")
    return b


def _bar(template, t, r, c):
    arr = _orient(template, t)
    return {"arr": arr, "t": t, "r": r, "c": c,
            "labels": _orient(np.arange(template.size).reshape(template.shape), t),
            "cells": {(r + x, c + y) for x in range(arr.shape[0])
                      for y in range(arr.shape[1])}}


def _edge(bar, side, slot, length):
    x, y = np.argwhere(bar["labels"] == side * length + slot)[0]
    u, v = np.argwhere(bar["labels"] == (1 - side) * length + slot)[0]
    return (bar["r"] + int(x), bar["c"] + int(y)), (int(x-u), int(y-v))


def _ray(bar, side, slot, length, shape):
    p, direction = _edge(bar, side, slot, length)
    p = _add(p, direction)
    cells = []
    while _inside(p, shape):
        cells.append(p)
        p = _add(p, direction)
    return tuple(cells)


def _find_bars(a, bg):
    # Maximal two-cell-wide runs; cartridges are only one cell wide at labels.
    eligible = (a != bg) & (a != NEUTRAL) & (a != RECEIVER)
    found = []
    for transpose in (False, True):
        b = eligible.T if transpose else eligible
        for r in range(b.shape[0] - 1):
            run = b[r] & b[r+1]
            start = 0
            while start < len(run):
                if not run[start]:
                    start += 1
                    continue
                end = start
                while end < len(run) and run[end]:
                    end += 1
                if 4 <= end-start <= 10:
                    rr, cc = (start, r) if transpose else (r, start)
                    hh, ww = (end-start, 2) if transpose else (2, end-start)
                    found.append({"r": rr, "c": cc,
                                  "arr": a[rr:rr+hh, cc:cc+ww].copy(),
                                  "cells": {(x, y) for x in range(rr, rr+hh)
                                            for y in range(cc, cc+ww)}})
                start = end
    found.sort(key=lambda x: (x["r"], x["c"]))
    _need(2 <= len(found) <= 6, "expected two to six bars")
    all_cells = set()
    for e in found:
        _need(not all_cells.intersection(e["cells"]), "overlapping bar candidates")
        all_cells |= e["cells"]
    first = found[0]["arr"]
    canonical = first if first.shape[0] == 2 else np.rot90(first)
    length = canonical.shape[1]
    forms = [_orient(canonical, t) for t in range(8)]
    _need(len({_key(x) for x in forms}) == 8, "bar must be asymmetric")
    structural = set(map(int, canonical.flat))
    _need(2 <= len(structural) <= 3, "expected two or three structural colors")
    labels = np.arange(canonical.size).reshape(canonical.shape)
    for e in found:
        matches = [t for t, form in enumerate(forms) if np.array_equal(form, e["arr"])]
        _need(len(matches) == 1, "bar correspondence is not unique")
        e["t"] = matches[0]
        e["labels"] = _orient(labels, e["t"])
    return found, length, structural


def _decode(input_grid):
    a = _grid(input_grid)
    bg = _background(a)
    bars, length, structural = _find_bars(a, bg)
    bar_cells = set().union(*(e["cells"] for e in bars))
    masks = _components8(map(tuple, np.argwhere(a == NEUTRAL)))
    _need(1 <= len(masks) <= 3, "expected one to three cartridges")
    options = [[] for _ in masks]
    for i, e in enumerate(bars):
        for side in (0, 1):
            for slot in range(length):
                edge, d = _edge(e, side, slot, length)
                sl, tl, q = [_add(edge, d, k) for k in (1, 2, 3)]
                if not all(_inside(p, a.shape) for p in (sl, tl, q)):
                    continue
                source, target = int(a[sl]), int(a[tl])
                if source in structural | {bg, NEUTRAL, RECEIVER}:
                    continue
                if target in structural | {bg, NEUTRAL, RECEIVER, source}:
                    continue
                if int(a[q]) not in (bg, NEUTRAL):
                    continue
                for j, mask in enumerate(masks):
                    if q not in _halo(mask):
                        continue
                    offsets = tuple(sorted(_inverse((r-q[0], c-q[1]), e["t"])
                                           for r, c in mask))
                    options[j].append({"donor": i, "side": side, "slot": slot,
                                       "source": source, "target": target, "q": q,
                                       "mask": mask, "labels": {sl, tl}, "offsets": offsets})
    _need(all(len(x) == 1 for x in options), "ambiguous cartridge association")
    records = [x[0] for x in options]
    _need(len({x["source"] for x in records}) == len(records), "duplicate program address")
    sources = {x["source"] for x in records}
    targets = {x["target"] for x in records}
    _need(sources.isdisjoint(targets), "sources and targets must be separate roles")
    labels = set().union(*(rec["labels"] for rec in records))
    _need(len(labels) == 2*len(records), "shared labels")
    instruction_cells = labels | set().union(*masks)
    _need(not instruction_cells.intersection(bar_cells), "cartridge touches bar interior")
    holes = {rec["q"] for rec in records if rec["q"] not in rec["mask"]}
    protected = bar_cells | instruction_cells | holes
    leftover = {tuple(p) for p in np.argwhere(a != bg)} - protected
    receivers = {p for p in leftover if int(a[p]) == RECEIVER}
    clues = leftover - receivers
    _need(bool(clues), "missing boundary code")
    for p in clues:
        _need(p[0] in (0, a.shape[0]-1) or p[1] in (0, a.shape[1]-1),
              "unexplained interior content")
        _need(int(a[p]) in sources, "boundary code has no program")
    code, active_sides, observations = {}, set(), []
    for p in sorted(clues):
        matches = []
        for i, e in enumerate(bars):
            for side in (0, 1):
                for slot in range(length):
                    ray = _ray(e, side, slot, length, a.shape)
                    if (ray and ray[-1] == p and not protected.intersection(ray)
                            and all(int(a[q]) in (bg, RECEIVER) for q in ray[:-1])):
                        matches.append((i, side, slot))
        _need(len(matches) == 1, "ambiguous boundary clue")
        i, side, slot = matches[0]
        source = int(a[p])
        _need(slot not in code or code[slot] == source, "conflicting slot code")
        code[slot] = source
        active_sides.add(side)
        observations.append((i, slot, source))
    _need(len(active_sides) == 1 and 1 <= len(code) <= 4, "inconsistent active side/code")
    side = active_sides.pop()
    _need(all(rec["side"] == 1-side for rec in records), "cartridges must face backward")
    _need(set(code.values()) == sources, "each cartridge must have a boundary observation")
    ray_map, rays = {}, {}
    for i, e in enumerate(bars):
        for slot, source in code.items():
            ray = _ray(e, side, slot, length, a.shape)
            _need(len(ray) >= 2 and not protected.intersection(ray), "blocked active path")
            _need(all(int(a[p]) in (bg, RECEIVER, source) for p in ray), "path crosses other code")
            rays[i, slot] = ray
            for p in ray:
                ray_map.setdefault(p, []).append((i, slot, source))
    by_source = {rec["source"]: rec for rec in records}
    hits, inactive = [], set()
    for p in sorted(receivers):
        entries = ray_map.get(p, [])
        if not entries:
            inactive.add(p)
            continue
        _need(len(entries) == 1, "receiver has multiple projectors")
        i, slot, source = entries[0]
        hits.append({"p": p, "bar": i, "slot": slot, "record": by_source[source]})
    _need(bool(hits), "no activated receivers")
    _need({h["record"]["source"] for h in hits} == sources, "unused cartridge")
    paint, visits, clipped = {}, Counter(), set()
    hit_cells = {h["p"] for h in hits}
    for h in hits:
        e, rec, anchor = bars[h["bar"]], h["record"], h["p"]
        visible = 0
        for offset in rec["offsets"]:
            p = _add(anchor, _vector(offset, e["t"]))
            if not _inside(p, a.shape):
                if p[0] < 0: clipped.add("top")
                if p[0] >= a.shape[0]: clipped.add("bottom")
                if p[1] < 0: clipped.add("left")
                if p[1] >= a.shape[1]: clipped.add("right")
                continue
            _need(p not in protected | clues | inactive, "stamp covers retained content")
            _need(p not in paint or paint[p] == rec["target"], "incompatible stamp overlap")
            paint[p] = rec["target"]
            visits[p] += 1
            visible += 1
        _need(visible > 0, "entire stamp is clipped")
    out = a.copy()
    for p in instruction_cells | hit_cells:
        out[p] = bg
    for p, color in paint.items():
        out[p] = color
    return {"input": a, "output": out, "background": bg, "bars": bars,
            "length": length, "structural": structural, "records": records,
            "code": code, "side": side, "clues": clues, "observations": observations,
            "receivers": receivers, "hits": hits, "inactive": inactive,
            "protected": protected, "instruction_cells": instruction_cells,
            "rays": rays, "paint": paint, "visits": visits, "clipped": clipped}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode this input only; no puzzle, seed or generation state is consulted."""
    return _decode(input_grid)["output"]


def _shape_key(offsets):
    variants = []
    for t in range(8):
        points = [_vector(p, t) for p in offsets]
        r0, c0 = min(r for r, c in points), min(c for r, c in points)
        variants.append(tuple(sorted((r-r0, c-c0) for r, c in points)))
    return min(variants)


def _template(rng, length, colors):
    for _ in range(40):
        a = rng.choice(colors, (2, length)).astype(np.int8)
        if rng.random() < .55:
            a[0] = colors[0]
            a[1] = colors[1]
            a[1, :int(rng.integers(1, length))] = colors[0]
            if len(colors) == 3:
                a[1, int(rng.integers(length))] = colors[2]
        if set(map(int, a.flat)) == set(colors) and len({_key(_orient(a, t)) for t in range(8)}) == 8:
            return a
    a = np.full((2, length), colors[0], dtype=np.int8)
    a[1, 1:] = colors[1]
    if len(colors) == 3:
        a[1, -1] = colors[2]
    return a


def _mask(rng, hole=False, teaching=False):
    # Coordinates (passive outward distance, longitudinal distance), relative to q.
    mode = int(rng.integers(4))
    if hole:
        cells = {(-1, -1), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)}
        if rng.random() < .8:
            cells.add((2, int(rng.choice((-1, 0, 1)))))
        if rng.random() < .5:
            cells.add((1, 2))
    elif mode == 0:
        cells = {(0, c) for c in range(int(rng.integers(2, 5)))}
        if teaching or rng.random() < .6:
            cells.add((1, int(rng.integers(0, max(c for r, c in cells)+1))))
    elif mode == 1:
        n, w = int(rng.integers(2, 5)), int(rng.integers(1, 4))
        cells = {(r, 0) for r in range(n)} | {(n-1, c) for c in range(1, w+1)}
    elif mode == 2:
        n, w = int(rng.integers(2, 4)), int(rng.integers(2, 5))
        cells = {(r, c) for r in range(n) for c in range(w)
                 if r in (0, n-1) or c in (0, w-1)}
        cells.discard((0, w-1))
    else:
        cells = {(0, 0)}
        desired = int(rng.integers(3, 13))
        for _ in range(100):
            if len(cells) >= desired:
                break
            r, c = sorted(cells)[int(rng.integers(len(cells)))]
            dr, dc = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
            p = r+dr, c+dc
            if -1 <= p[0] <= 3 and -3 <= p[1] <= 3 and p != (-1, 0):
                cells.add(p)
    if rng.integers(2):
        cells = {(r, -c) for r, c in cells}
    return tuple(sorted(cells))


def _place_bars(rng, template, transforms, shape):
    placed, cells = [], set()
    for t in transforms:
        ah, aw = _orient(template, t).shape
        for _ in range(70):
            r = int(rng.integers(0, shape[0]-ah+1))
            c = int(rng.integers(0, shape[1]-aw+1))
            e = _bar(template, t, r, c)
            if not _halo(e["cells"]).intersection(cells):
                placed.append(e)
                cells |= e["cells"]
                break
        else:
            raise ValueError("bars did not fit")
    return placed


def _mount(rng, grid, bars, side, sources, targets, masks):
    length = max(bars[0]["arr"].shape)
    occupied = set().union(*(e["cells"] for e in bars))
    neutral_cells, mounted, used_slots = set(), [], set()
    for source, target, offsets_nt in zip(sources, targets, masks):
        choices = [(i, s) for i in range(len(bars)) for s in range(length)]
        rng.shuffle(choices)
        for i, slot in choices:
            # Neighboring one-cell labels must not form a false two-wide bar.
            if any(j == i and abs(slot-s) <= 2 for j, s in used_slots):
                continue
            edge, d = _edge(bars[i], 1-side, slot, length)
            tangent = _vector((0, 1), bars[i]["t"])
            sl, tl, q = [_add(edge, d, k) for k in (1, 2, 3)]
            cells = {_add(_add(q, d, n), tangent, t) for n, t in offsets_nt}
            footprint = cells | {sl, tl, q}
            if (not all(_inside(p, grid.shape) for p in footprint)
                    or footprint.intersection(occupied)
                    or _halo(cells).intersection(neutral_cells)):
                continue
            # Keep all registration holes reserved, too.
            for p in cells:
                grid[p] = NEUTRAL
            grid[sl], grid[tl] = source, target
            canonical_offsets = tuple(sorted(_inverse((r-q[0], c-q[1]), bars[i]["t"])
                                             for r, c in cells))
            mounted.append({"source": source, "target": target, "mask": cells,
                            "labels": {sl, tl}, "q": q, "offsets": canonical_offsets,
                            "donor": i})
            occupied |= footprint
            neutral_cells |= cells
            used_slots.add((i, slot))
            break
        else:
            raise ValueError("cartridges did not fit")
    return mounted, occupied


def _choose_clues(rng, bars, side, code, protected, shape, mode):
    length = max(bars[0]["arr"].shape)
    endpoints = {}
    for i, bar in enumerate(bars):
        for local_side in (0, 1):
            for s in range(length):
                ray = _ray(bar, local_side, s, length, shape)
                if len(ray) >= 2 and not protected.intersection(ray):
                    endpoints.setdefault(ray[-1], []).append((i, local_side, s))
    rays = {(i, s): _ray(e, side, s, length, shape)
            for i, e in enumerate(bars) for s in code}
    _need(all(len(ray) >= 2 and not protected.intersection(ray) for ray in rays.values()),
          "blocked projected path")
    choices = {s: [i for i in range(len(bars))
                   if endpoints.get(rays[i, s][-1]) == [(i, side, s)]] for s in code}
    _need(all(choices.values()), "no readable clue endpoint")
    slots = list(code)
    rng.shuffle(slots)
    allocations = []
    if mode == "full":
        full = [i for i in range(len(bars)) if all(i in choices[s] for s in slots)]
        _need(bool(full), "cannot put whole code on one bar")
        donor = full[int(rng.integers(len(full)))]
        allocations = [(donor, s) for s in slots]
    else:
        for s in slots:
            options = choices[s][:]
            rng.shuffle(options)
            if allocations:
                options.sort(key=lambda i: sum(i == j for j, _ in allocations))
            allocations.append((options[0], s))
        if mode == "split":
            _need(len({i for i, s in allocations}) >= 2, "need distributed clues")
    # A clue cannot sit on another active path, even when its color agrees.
    for i, s in allocations:
        p = rays[i, s][-1]
        _need(sum(p in ray for ray in rays.values()) == 1, "clue crosses another path")
    if mode == "spread" and rng.random() < .3:
        extra = [(i, s) for s in slots for i in choices[s]
                 if (i, s) not in allocations and
                 sum(rays[i, s][-1] in ray for ray in rays.values()) == 1]
        if extra:
            allocations.append(extra[int(rng.integers(len(extra)))])
    return allocations, rays


def _receivers(rng, grid, bg, bars, records, code, clues, rays, protected, teaching):
    clue_cells = {rays[i, s][-1] for i, s in clues}
    records_by_source = {rec["source"]: rec for rec in records}
    membership = {}
    for (i, s), ray in rays.items():
        for p in ray:
            membership.setdefault(p, []).append((i, s))
    active, paint, hits = set(), {}, []

    def place(i, s, p, complete=False, clipped=False):
        if p in protected | clue_cells | active or membership.get(p) != [(i, s)]:
            return False
        rec = records_by_source[code[s]]
        full = {_add(p, _vector(offset, bars[i]["t"])) for offset in rec["offsets"]}
        footprint = {q for q in full if _inside(q, grid.shape)}
        if not footprint or footprint.intersection(protected | clue_cells):
            return False
        if complete and len(full) != len(footprint):
            return False
        if clipped and len(full) == len(footprint):
            return False
        if any(q in paint and paint[q] != rec["target"] for q in footprint):
            return False
        grid[p] = RECEIVER
        active.add(p)
        hits.append((i, s, p, rec))
        for q in footprint:
            paint[q] = rec["target"]
        return True

    possibilities = [(i, s, p) for (i, s), ray in rays.items() for p in ray]
    rng.shuffle(possibilities)
    # All programs have a complete example; at least one event requires transfer.
    for source in records_by_source:
        options = [(i, s, p) for i, s, p in possibilities if code[s] == source]
        options.sort(key=lambda x: (x[0], x[1]) in clues)
        _need(any(place(i, s, p, complete=teaching is not None)
                  for i, s, p in options), "cannot activate program")
    if teaching == "hole":
        # The first cartridge is hole registered and has a forward tip, so a
        # border receiver can demonstrate clipping without losing registration.
        source = records[0]["source"]
        edge_options = [(i, s, p) for i, s, p in possibilities if code[s] == source]
        _need(any(place(i, s, p, clipped=True) for i, s, p in edge_options),
              "cannot demonstrate clipped hole stamp")
    desired = int(rng.integers(max(3, len(active)), max(4, len(active)+7)))
    if rng.random() < .5:
        # Neighboring original receivers produce simultaneous overlapping stamps.
        possibilities.sort(key=lambda x: min(abs(x[2][0]-p[0])+abs(x[2][1]-p[1]) for p in active))
    for i, s, p in possibilities:
        if len(active) >= desired:
            break
        place(i, s, p)
    _need(any((i, s) not in clues for i, s, p, rec in hits), "no unhinted receiver event")
    # Inert receivers include exact alignment with unobserved slots when possible.
    all_rays = set(membership)
    candidates = [p for p in np.ndindex(grid.shape)
                  if p not in all_rays | protected | clue_cells | set(paint)
                  and int(grid[p]) == bg]
    rng.shuffle(candidates)
    length = max(bars[0]["arr"].shape)
    unobserved = set().union(*(_ray(e, side, s, length, grid.shape)
                              for e in bars for side in (0, 1) for s in range(length)
                              if s not in code))
    candidates.sort(key=lambda p: p not in unobserved)
    _need(bool(candidates), "no inert receiver location")
    for p in candidates[:int(rng.integers(1, 5))]:
        grid[p] = RECEIVER
    return active, paint


def _pair_once(rng, teaching=None, test=False, forbidden_shapes=frozenset()):
    length = int(rng.integers(5 if teaching else 4, 11))
    count = int(rng.choice((2, 2, 3, 3, 4, 5)))
    if teaching == "pool":
        transforms = [0, 4] + [0]*(count-2)
    elif teaching == "hole":
        transforms = [0, 1] + [1]*(count-2)
    elif test:
        transforms = [0, 5] + list(map(int, rng.integers(8, size=count-2)))
    else:
        transforms = list(map(int, rng.integers(8, size=count)))
    if count >= 4:
        shape = tuple(map(int, rng.integers(24, 31, size=2)))
    else:
        shape = tuple(map(int, rng.integers(15, 31, size=2)))
    if rng.random() < .2:
        shape = shape[0], shape[0]
    program_count = 2 if teaching else int(rng.integers(1, 4))
    structural_count = int(rng.integers(2, 4))
    palette = [int(x) for x in rng.permutation([c for c in range(10) if c not in (NEUTRAL, RECEIVER)])]
    bg = palette[0]
    structural = palette[1:1+structural_count]
    sources = palette[1+structural_count:1+structural_count+program_count]
    available = palette[1+structural_count+program_count:]
    target_count = int(rng.integers(1, min(program_count, len(available))+1))
    targets = [available[i % target_count] for i in range(program_count)]
    rng.shuffle(targets)
    template = _template(rng, length, structural)
    bars = _place_bars(rng, template, transforms, shape)
    side = int(rng.integers(2))
    masks = [_mask(rng, hole=(teaching == "hole" and i == 0)
                   or (teaching != "pool" and rng.random() < .35),
                   teaching=teaching is not None) for i in range(program_count)]
    if test:
        _need(any(_shape_key(m) not in forbidden_shapes for m in masks), "test needs a fresh mask")
    grid = np.full(shape, bg, dtype=np.int8)
    for e in bars:
        ah, aw = e["arr"].shape
        grid[e["r"]:e["r"]+ah, e["c"]:e["c"]+aw] = e["arr"]
    records, protected = _mount(rng, grid, bars, side, sources, targets, masks)
    slot_count = program_count if teaching else int(rng.integers(program_count, min(4, length)+1))
    slots = sorted(map(int, rng.choice(length, slot_count, replace=False)))
    values = sources + [sources[int(rng.integers(program_count))] for _ in range(slot_count-program_count)]
    rng.shuffle(values)
    code = dict(zip(slots, values))
    mode = "split" if teaching == "pool" else "full" if teaching == "hole" else str(rng.choice(("full", "spread")))
    clues, rays = _choose_clues(rng, bars, side, code, protected, shape, mode)
    for i, s in clues:
        grid[rays[i, s][-1]] = code[s]
    active, paint = _receivers(rng, grid, bg, bars, records, code, clues, rays, protected, teaching)
    # Construct output from sampled geometry, before asking the input decoder.
    out = grid.copy()
    for rec in records:
        for p in rec["mask"] | rec["labels"]:
            out[p] = bg
    for p in active:
        out[p] = bg
    for p, color in paint.items():
        out[p] = color
    decoded = _decode(grid)
    _need(np.array_equal(out, decoded["output"]), "construction and visible rule disagree")
    f = _features(decoded)
    _need(f["nonpoint"] and f["transferred"] > 0, "parent concept is unobservable")
    _need(f["without_transfer_different"], "unhinted stamps must change the answer")
    _need(f["unpainted_path"] > 0, "need visible evidence that paths are not painted")
    if teaching == "pool":
        _need(f["pooled"] and f["reversed"] and f["offcenter"], "weak pooling demonstration")
    if teaching == "hole":
        _need(f["mixed_axes"] and f["holes"] and f["clipped"], "weak hole/rotation demonstration")
    if teaching:
        _need(f["addressed_not_opposite"] > 0, "program address must differ from mounting slot")
    # No compass direction or reference-bar placement is privileged.
    overall = int(rng.integers(8))
    return {"input": _orient(grid, overall), "output": _orient(out, overall)}


def _features(d):
    allocations = [set() for _ in d["bars"]]
    directly_observed = set()
    for i, slot, source in d["observations"]:
        allocations[i].add(slot)
        directly_observed.add((i, slot))
    point = d["input"].copy()
    for p in d["instruction_cells"]:
        point[p] = d["background"]
    for h in d["hits"]:
        point[h["p"]] = h["record"]["target"]
    local_only = d["input"].copy()
    for p in d["instruction_cells"]:
        local_only[p] = d["background"]
    local_hits = [h for h in d["hits"] if (h["bar"], h["slot"]) in directly_observed]
    for h in local_hits:
        local_only[h["p"]] = d["background"]
    for h in local_hits:
        for offset in h["record"]["offsets"]:
            p = _add(h["p"], _vector(offset, d["bars"][h["bar"]]["t"]))
            if _inside(p, local_only.shape):
                local_only[p] = h["record"]["target"]
    # Relative transformations are measured from the first decoded bar.
    transforms = [e["t"] for e in d["bars"]]
    offsets = [rec["offsets"] for rec in d["records"]]
    return {"shape": d["input"].shape, "background": d["background"],
            "length": d["length"], "bar_count": len(d["bars"]),
            "program_count": len(d["records"]), "slots": tuple(sorted(d["code"])),
            "slot_count": len(d["code"]), "allocations": tuple(map(len, allocations)),
            "pooled": all(len(x) < len(d["code"]) for x in allocations),
            "holes": sum((0, 0) not in x for x in offsets),
            "offcenter": any((0, 0) in x and (sum(r for r, c in x), sum(c for r, c in x)) != (0, 0)
                             for x in offsets),
            "reversed": any(t >= 4 for t in transforms),
            "mixed_axes": len({min(e["arr"].shape) == e["arr"].shape[0] for e in d["bars"]}) > 1,
            "transforms": tuple(transforms),
            "transferred": sum((h["bar"], h["slot"]) not in directly_observed for h in d["hits"]),
            "without_transfer_different": not np.array_equal(local_only, d["output"]),
            "addressed_not_opposite": sum(h["slot"] != h["record"]["slot"] for h in d["hits"]),
            "nonpoint": not np.array_equal(point, d["output"]),
            "hits": len(d["hits"]), "inactive": len(d["inactive"]),
            "clipped": tuple(sorted(d["clipped"])),
            "overlap": sum(n > 1 for n in d["visits"].values()),
            "covered_receivers": len(set(d["paint"]).intersection(h["p"] for h in d["hits"])),
            "mask_sizes": tuple(map(len, offsets)),
            "geometries": tuple(_shape_key(x) for x in offsets),
            "targets": tuple(rec["target"] for rec in d["records"]),
            "unpainted_path": sum(p not in d["paint"] and int(d["input"][p]) == d["background"]
                                  for ray in d["rays"].values() for p in ray)}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    _need(low <= int(value) <= high, name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    _need(num_train > num_test, "num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen, familiar = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = ("pool", "hole")[index] if split == "train" and index < 2 else None
            for _ in range(240):
                try:
                    pair = _pair_once(rng, teaching, split == "test", familiar)
                    k = _key(pair["input"])
                    if k in seen:
                        continue
                    seen.add(k)
                    puzzle[split].append(pair)
                    if split == "train":
                        familiar.update(_features(_decode(pair["input"]))["geometries"])
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded projector construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _count(len(puzzle["train"]), 2, 10, "training count")
    _count(len(puzzle["test"]), 1, 4, "test count")
    _need(len(puzzle["train"]) > len(puzzle["test"]), "counts must decrease")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "shape changed")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            _need(np.array_equal(out, d["output"]), "incorrect projector output")
            _need(not np.array_equal(inp, out), "identity pair")
            _need(bool(d["inactive"]) and all(out[p] == RECEIVER for p in d["inactive"]),
                  "need a preserved off-path receiver")
            _need(all(out[p] == inp[p] for p in d["clues"] | set().union(*(e["cells"] for e in d["bars"]))),
                  "bar or boundary observation was overwritten")
            f = _features(d)
            _need(f["nonpoint"] and f["transferred"] > 0, "need both parent mechanisms")
            _need(f["without_transfer_different"], "ignoring transfer accidentally gives the answer")
            _need(f["unpainted_path"] > 0, "paths must remain invisible")
            features[split].append(f)
    tr = features["train"]
    _need(any(f["pooled"] and f["reversed"] and f["offcenter"] for f in tr),
          "training must show pooling, reflection and off-center registration")
    _need(any(f["mixed_axes"] and f["holes"] and f["clipped"] for f in tr),
          "training must show rotation, hole registration and clipping")
    _need(any(f["addressed_not_opposite"] > 0 for f in tr), "training must show color addressing")
    known = {g for f in tr for g in f["geometries"]}
    _need(all(set(f["geometries"])-known for f in features["test"]), "test must read a fresh mask")
    _need(all(f["mixed_axes"] and f["reversed"] for f in features["test"]),
          "test must combine a quarter turn with reflection")
    return True


def _regression_fixture():
    """Small hand-specified geometry with explicit expected paint coordinates."""
    template = np.array([[1, 1, 1, 1, 1], [2, 1, 1, 2, 2]], dtype=np.int8)
    a = np.zeros((20, 20), dtype=np.int8)
    a[8:10, 2:7] = template
    a[8:10, 12:17] = np.fliplr(template)
    a[10, 2], a[11, 2] = 3, 6
    for p in ((12, 2), (12, 3), (13, 3)):
        a[p] = NEUTRAL
    a[10, 13], a[11, 13] = 4, 7
    for p in ((11, 12), (11, 14), (12, 12), (12, 14), (13, 12), (13, 13), (13, 14)):
        a[p] = NEUTRAL
    a[0, 3], a[0, 13] = 3, 4
    for p in ((3, 15), (4, 15), (0, 5), (2, 5), (4, 8)):
        a[p] = RECEIVER
    out = np.zeros_like(a)
    out[8:10, 2:7] = template
    out[8:10, 12:17] = np.fliplr(template)
    out[0, 3], out[0, 13], out[4, 8] = 3, 4, RECEIVER
    for p in ((3, 15), (3, 14), (4, 14), (4, 15), (5, 14)):
        out[p] = 6
    for p in ((0, 4), (0, 6), (1, 4), (1, 5), (1, 6),
              (2, 4), (2, 6), (3, 4), (3, 5), (3, 6)):
        out[p] = 7
    return {"input": a, "output": out}


def self_test():
    fixture = _regression_fixture()
    _need(np.array_equal(solve(fixture["input"]), fixture["output"]),
          "registered reflection/hole/clipping regression failed")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = [_features(_decode(pair["input"])) for p in puzzles for pair in p["train"]+p["test"]]
    _need(len({tuple(_key(x["input"]) for x in p["train"]+p["test"]) for p in puzzles}) == 42,
          "42 seeds must give distinct puzzles")
    _need(len({f["shape"] for f in features}) >= 40, "too few canvases")
    _need({f["length"] for f in features} == set(range(4, 11)), "missing bar lengths")
    _need({f["program_count"] for f in features} == {1, 2, 3}, "missing program counts")
    _need({f["slot_count"] for f in features} >= {1, 2, 3, 4}, "missing code counts")
    _need(len({g for f in features for g in f["geometries"]}) >= 35, "too few stencil geometries")
    _need(len({f["background"] for f in features}) >= 7, "too few background roles")
    _need(any(f["overlap"] for f in features), "missing compatible overlap")
    _need(any(f["clipped"] for f in features) and any(not f["clipped"] for f in features),
          "missing clipping variation")
    _need(any(f["bar_count"] >= 4 for f in features), "missing many-bar cases")
    _need(any(len(set(f["targets"])) < f["program_count"] for f in features), "missing shared targets")
    # Determinism and input immutability.
    p = generate_puzzle(7001, 5, 3)
    q = generate_puzzle(7001, 5, 3)
    for x, y in zip(p["train"]+p["test"], q["train"]+q["test"]):
        _need(np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"]),
              "nondeterministic")
        saved = x["input"].copy()
        _need(np.array_equal(solve(x["input"]), x["output"]), "solve disagrees")
        _need(np.array_equal(saved, x["input"]), "solve modified its input")
        for t in range(8):
            _need(np.array_equal(solve(_orient(saved, t)), _orient(x["output"], t)),
                  "dihedral equivariance failed")
    # Recolor every non-reserved role, including the background, input-only.
    perm = np.arange(10, dtype=np.int8)
    roles = [c for c in range(10) if c not in (NEUTRAL, RECEIVER)]
    perm[roles] = np.roll(roles, 3)
    x = p["train"][0]
    _need(np.array_equal(solve(perm[x["input"]]), perm[x["output"]]), "role recoloring failed")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in ("train", "test")}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    for a, b in ((2, 1), (10, 4), (7, 2)):
        test = generate_puzzle(8100+a+b, a, b)
        _need(len(test["train"]) == a and len(test["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(features), "canvases": len({f["shape"] for f in features}),
            "mask_geometries": len({g for f in features for g in f["geometries"]}),
            "overlap_pairs": sum(bool(f["overlap"]) for f in features)}


if __name__ == "__main__":
    print(self_test())
