"""Finite-volume spillway networks. Only the standard library and NumPy are used.

The input-only solver recognizes its own visible palette and gravity direction.
Generation constructs terrain; it does not import either parent generator.
"""
from __future__ import annotations

from itertools import combinations
import json
import numpy as np


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("a grid must be an integer matrix of size 1..30 with colors 0..9")
    return a.astype(np.int8, copy=True)


def _recognize(grid):
    """Liquid is the unique color forming separated inward boundary prefixes."""
    a = _grid(grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) < 3 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely dominant")
    bg = int(colors[np.argmax(counts)])
    choices = []
    for k in range(4):
        c = np.rot90(a, k)
        h, w = c.shape
        for water in map(int, colors):
            if water == bg:
                continue
            entries = np.flatnonzero(c[0] == water)
            if not len(entries) or np.any(np.diff(entries) < 2):
                continue
            mask = np.zeros(c.shape, dtype=bool)
            sources = {}
            for col in entries:
                rows = np.flatnonzero(c[:, col] == water)
                if not 2 <= len(rows) < h or not np.array_equal(rows, np.arange(len(rows))):
                    break
                mask[:len(rows), col] = True
                sources[int(col)] = int(len(rows))
            else:
                if np.array_equal(mask, c == water):
                    choices.append((k, bg, water, c.copy(), sources))
    if len(choices) != 1:
        raise ValueError("liquid role and inward direction must have exactly one interpretation")
    return choices[0]


def _runs(solid):
    for r in range(solid.shape[0]):
        c = 0
        while c < solid.shape[1]:
            if not solid[r, c]:
                c += 1
                continue
            a = c
            while c + 1 < solid.shape[1] and solid[r, c + 1]:
                c += 1
            yield r, a, c
            c += 1


def _cups(solid):
    """Detect open rectangular cavities on maximal floor runs, including edges.

    Colors of touching solid cells do not split a run. An ordinary shelf has
    capacity zero; forward wall fragments are therefore ordinary singleton runs.
    """
    h, w = solid.shape
    cups = {}
    occupied = np.zeros(solid.shape, dtype=bool)
    for r, a, b in _runs(solid):
        da = db = 0
        while r - da - 1 >= 0 and solid[r - da - 1, a]:
            da += 1
        while r - db - 1 >= 0 and solid[r - db - 1, b]:
            db += 1
        if b - a >= 2 and da == db and da > 0:
            d, lo, hi = da, a + 1, b
        elif a == 0 and b > a and da == 0 and db > 0:
            d, lo, hi = db, a, b
        elif b == w - 1 and b > a and db == 0 and da > 0:
            d, lo, hi = da, a + 1, b + 1
        else:
            continue
        if np.any(solid[r-d:r, lo:hi]):
            continue
        cavity = np.zeros(solid.shape, dtype=bool)
        cavity[r-d:r, lo:hi] = True
        if np.any(cavity & occupied):
            raise ValueError("overlapping cup interiors")
        occupied |= cavity
        cups[(r, a, b)] = {"floor": r, "left": a, "right": b,
                           "depth": d, "lo": lo, "hi": hi,
                           "width": hi-lo, "capacity": d*(hi-lo)}
    return cups


def _simulate(canonical, bg, water, sources, *, no_storage=False,
              no_detours=False, no_addition=False, outline_all=False,
              no_initial_retention=False):
    """Counted streams, with optional faulty rules used only for witness checks."""
    c = canonical
    solid = (c != bg) & (c != water)
    cups = _cups(solid)
    h, w = c.shape
    out = c.copy()
    out[c == water] = bg
    active = dict(sources)
    events = []
    incoming_layers = []
    merge_count = simultaneous = 0
    held = 0
    encountered_cup = False

    def add(target, col, amount):
        nonlocal merge_count
        if amount <= 0:
            return
        if col in target:
            merge_count += 1
        target[col] = (max(target.get(col, 0), amount) if no_addition
                       else target.get(col, 0) + amount)

    for r in range(h):
        incoming_layers.append(dict(active))
        hits = {}
        following = {}
        for col, amount in sorted(active.items()):
            if not solid[r, col]:
                add(following, col, amount)
                continue
            a = b = col
            while a > 0 and solid[r, a-1]:
                a -= 1
            while b + 1 < w and solid[r, b+1]:
                b += 1
            hits.setdefault((r, a, b), []).append((col, amount))
        simultaneous += len(hits) > 1
        for key, arrivals in sorted(hits.items()):
            _, a, b = key
            amount = sum(v for _, v in arrivals)
            if no_addition:
                amount = max(v for _, v in arrivals)
            cup = cups.get(key)
            capacity = 0 if cup is None or no_storage else cup["capacity"]
            if cup is not None:
                if no_initial_retention and not encountered_cup:
                    capacity = 0
                encountered_cup = True
            stored = min(amount, capacity)
            if stored:
                iw = cup["width"]
                if stored % iw:
                    raise ValueError("the supply would partially fill a horizontal level")
                out[r-stored//iw:r, cup["lo"]:cup["hi"]] = water
                held += stored
            excess = amount - stored
            flanks = [x for x in (a-1, b+1) if 0 <= x < w]
            if excess:
                if no_detours:
                    # A gravity-only shortcut traps excess above its impact.
                    col = arrivals[0][0]
                    free = [y for y in range(r-1, -1, -1)
                            if out[y, col] == bg]
                    if len(free) < excess:
                        raise ValueError("shortcut cannot store its excess")
                    out[free[:excess], col] = water
                    held += excess
                else:
                    if not flanks or excess % len(flanks):
                        raise ValueError("uncaught or fractional spill")
                    for flank in flanks:
                        add(following, flank, excess // len(flanks))
            events.append({"run": key, "arrivals": tuple(arrivals), "amount": amount,
                           "capacity": capacity, "stored": stored, "excess": excess,
                           "flanks": tuple(flanks), "cup": cup is not None})
        active = following
    if active:
        raise ValueError("positive flow reaches the far boundary without a receiving cup")
    if not (no_storage or no_addition or no_detours or no_initial_retention) and held != sum(sources.values()):
        raise AssertionError("volume accounting failed")
    if outline_all:
        # The wrong noncausal interpretation wets every dry cup too.
        for cup in cups.values():
            if not np.any(out[cup["floor"]-cup["depth"]:cup["floor"],
                              cup["lo"]:cup["hi"]] == water):
                out[cup["floor"]-1, cup["lo"]:cup["hi"]] = water
    return out, {"cups": cups, "events": events, "incoming": incoming_layers,
                 "merges": merge_count, "simultaneous": simultaneous,
                 "sources": sources, "background": bg, "water": water}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone; no generation settings or examples are read."""
    k, bg, water, c, sources = _recognize(input_grid)
    out, _ = _simulate(c, bg, water, sources)
    return np.ascontiguousarray(np.rot90(out, -k), dtype=np.int8)


def _cup_object(a, b, r, d, edge=None):
    cells = {(r, x) for x in range(a, b+1)}
    if edge != "left":
        cells.update((y, a) for y in range(r-d, r))
    if edge != "right":
        cells.update((y, b) for y in range(r-d, r))
    lo, hi = a + (edge != "left"), b + (edge == "right")
    return {"kind": "cup", "a": a, "b": b, "r": r, "d": d,
            "edge": edge, "lo": int(lo), "hi": int(hi), "cells": cells}


def _bar_object(a, b, r):
    return {"kind": "bar", "a": a, "b": b, "r": r, "d": 0,
            "cells": {(r, x) for x in range(a, b+1)}}


def _draw(c, obj):
    for r, x in obj["cells"]:
        c[r, x] = 2


def _source_lengths(rng, total, count, limit=12):
    values = [2] * count
    for _ in range(total - 2*count):
        possible = [i for i in range(count) if values[i] < limit]
        if not possible:
            raise ValueError("source amount exceeds prefix capacity")
        values[int(rng.choice(possible))] += 1
    rng.shuffle(values)
    return values


def _entry_columns(rng, lo, hi, count):
    options = [p for p in combinations(range(lo, hi), count)
               if all(b-a >= 2 for a, b in zip(p, p[1:]))]
    if not options:
        raise ValueError("not enough separated source positions")
    return options[int(rng.integers(len(options)))]


def _canonical_sources(c):
    return {int(x): int(np.count_nonzero(c[:, x] == 1))
            for x in np.flatnonzero(c[0] == 1)}


def _flow_after(c, last):
    """Construction's arithmetic through the already drawn layers only."""
    # A temporary terminal floor makes the ordinary simulator terminate; its
    # trace just before that floor records the remaining amounts, not labels.
    t = c[:last+2].copy()
    t[last+1] = 2
    # Simulate explicitly only to the next layer; this handles uncaught flow
    # without giving a special boundary convention to the public solver.
    solid = t == 2
    cups = _cups(solid)
    active = _canonical_sources(c)
    for r in range(last+1):
        hits, nxt = {}, {}
        for col, amount in active.items():
            if not solid[r, col]:
                nxt[col] = nxt.get(col, 0) + amount
            else:
                a = b = col
                while a > 0 and solid[r, a-1]:
                    a -= 1
                while b+1 < c.shape[1] and solid[r, b+1]:
                    b += 1
                hits[(r, a, b)] = hits.get((r, a, b), 0) + amount
        for key, amount in hits.items():
            r0, a, b = key
            cup = cups.get(key)
            capacity = 0 if cup is None else cup["capacity"]
            stored = min(amount, capacity)
            if stored and stored % cup["width"]:
                raise ValueError("incomplete level")
            excess = amount-stored
            flanks = [x for x in (a-1, b+1) if 0 <= x < c.shape[1]]
            if excess:
                if not flanks or excess % len(flanks):
                    raise ValueError("fractional construction")
                for x in flanks:
                    nxt[x] = nxt.get(x, 0) + excess//len(flanks)
        active = nxt
    return active


def _partitions(columns, width):
    """Reserve a background column between independently drawn floor runs."""
    result = []
    for i, c in enumerate(columns):
        lo = 0 if i == 0 else (columns[i-1]+c)//2 + 1
        hi = width-1 if i+1 == len(columns) else (c+columns[i+1])//2 - 1
        if not lo <= c <= hi:
            raise ValueError("streams too close for separate devices")
        result.append((lo, hi))
    return result


def _cup_options(col, amount, lo, hi, width, terminal=False):
    options = []
    for a in range(lo, col+1):
        for b in range(col, hi+1):
            variants = []
            if a < col < b:
                variants.append(None)
            if a == 0 and col < b:
                variants.append("left")
            if b == width-1 and a < col:
                variants.append("right")
            for edge in variants:
                iw = b-a-1 if edge is None else b-a
                if not 1 <= iw <= 6:
                    continue
                for d in range(1, 11 if terminal else 4):
                    capacity = iw*d
                    stored = min(amount, capacity)
                    excess = amount-stored
                    nf = int(a > 0) + int(b < width-1)
                    if stored % iw or (excess and (nf == 0 or excess % nf)):
                        continue
                    if terminal and excess:
                        continue
                    if terminal and capacity > amount + 2*iw:
                        continue
                    options.append((a, b, d, edge))
    return options


def _terminal(rng, c, objects, active, last):
    w = c.shape[1]
    if not active:
        return last
    columns = sorted(active)
    # When paths are too close, a reached shelf groups their incident amounts
    # and sends their sum to its outer flanks before terminal collection.
    if any(b-a < 4 for a, b in zip(columns, columns[1:])):
        a, b = columns[0], columns[-1]
        if a == 0 and b == w-1:
            raise ValueError("no outer flanks for consolidation")
        last += int(rng.integers(1, 4))
        if last >= 29:
            raise ValueError("too deep")
        obj = _bar_object(a, b, last)
        _draw(c, obj)
        objects.append(obj)
        active = _flow_after(c, last)
        columns = sorted(active)
    chosen = []
    for col, (lo, hi) in zip(columns, _partitions(columns, w)):
        options = _cup_options(col, active[col], lo, hi, w, True)
        if not options:
            raise ValueError("no complete terminal levels")
        # Favor shorter, wider cups, while retaining deeper narrow wells.
        depths = np.array([o[2] for o in options], dtype=float)
        p = 1/(depths*depths)
        option = options[int(rng.choice(len(options), p=p/p.sum()))]
        chosen.append(option)
    floor = last + max(o[2] for o in chosen) + int(rng.integers(2, 4))
    if floor >= 30:
        raise ValueError("terminal floor exceeds canvas")
    for a, b, d, edge in chosen:
        obj = _cup_object(a, b, floor, d, edge)
        _draw(c, obj)
        objects.append(obj)
    return floor


def _witness(rng):
    """One terrain, two volumes: retention versus repeated spill and confluence."""
    iw = int(rng.integers(3, 7))
    n = min(3, (iw+1)//2)
    d = 3 if iw == 3 else int(rng.integers(2, 4))
    cap = iw*d
    lows = [v for v in range(2*n, cap) if v % iw == 0]
    if not lows:
        raise ValueError("no partial witness level")
    low, high = int(rng.choice(lows)), cap+8
    ll, hl = _source_lengths(rng, low, n), _source_lengths(rng, high, n)
    w = int(rng.integers(iw+10, 31))
    # Some controlled demonstrations also teach a clipped downstream outlet.
    a = 2 if rng.random() < .4 else int(rng.integers(3, w-iw-4))
    b = a+iw+1
    entries = _entry_columns(rng, a+1, b, n)
    c = np.zeros((30, w), dtype=np.int8)
    for col, amount in zip(entries, hl):
        c[:amount, col] = 1
    objects = []
    r = max(hl)+d+int(rng.integers(1, 3))
    first = _cup_object(a, b, r, d)
    _draw(c, first)
    objects.append(first)
    p, q = a-1, b+1
    m = p + int(rng.choice([2, 3]))
    left_width = m-p-1
    ld = 2//left_width
    r2 = r + ld + int(rng.integers(2, 4))
    left = _cup_object(p-1, m-1, r2, ld)
    right_width = q-m-1
    if right_width in (1, 2) and rng.random() < .65:
        rd = 2//right_width
        r2 = r + max(ld, rd) + int(rng.integers(2, 4))
        left = _cup_object(p-1, m-1, r2, ld)
        right = _cup_object(m+1, q+1, r2, rd)
    else:
        right = _bar_object(m+1, q+1, r2)
    for obj in (left, right):
        _draw(c, obj)
        objects.append(obj)
    if r2 >= 27:
        raise ValueError("witness too deep")
    active = _flow_after(c, r2)
    # The outer, central, outer streams are separated; the central amount is
    # the sum of two surviving inner branches, not one path's amount.
    floor = _terminal(rng, c, objects, active, r2)
    h = min(30, floor+int(rng.integers(1, 4)))
    c = c[:h].copy()
    small = c.copy()
    small[small == 1] = 0
    for col, amount in zip(entries, ll):
        small[:amount, col] = 1
    if rng.random() < .5:
        c, small = c[:, ::-1].copy(), small[:, ::-1].copy()
        for obj in objects:
            obj["cells"] = {(r0, w-1-x) for r0, x in obj["cells"]}
    return [small, c], objects


def _random_scene(rng, test=False, allow_clipped=True):
    iw = int(rng.integers(1, 7))
    d = int(rng.integers(1, 4))
    n = int(rng.integers(1, min(3, (iw+1)//2)+1))
    w = int(rng.integers(max(10, iw+(10 if test and not allow_clipped else 5)), 31))
    edge = ("none" if test and not allow_clipped else
            str(rng.choice(["none", "left", "right"], p=[.55, .225, .225])))
    if edge == "left":
        a, b, lo, hi = 0, iw, 0, iw
    elif edge == "right":
        a, b, lo, hi = w-iw-1, w-1, w-iw, w
    else:
        a = (int(rng.integers(3, w-iw-4)) if test and not allow_clipped
             else int(rng.integers(1, w-iw-2)))
        b, lo, hi = a+iw+1, a+1, a+iw+1
    cap = iw*d
    nf = int(a > 0) + int(b < w-1)
    possible = [v for v in range(2*n, min(12*n, 26)+1)
                if (v <= cap and v % iw == 0 and not test)
                or (v > cap and (8 if test and nf == 2 else 4 if test else 1) <= v-cap <= 10
                    and (v-cap) % nf == 0)]
    if not possible:
        raise ValueError("no source quantity")
    volume = int(rng.choice(possible))
    lengths = _source_lengths(rng, volume, n)
    entries = _entry_columns(rng, lo, hi, n)
    c = np.zeros((30, w), dtype=np.int8)
    for col, amount in zip(entries, lengths):
        c[:amount, col] = 1
    r = max(lengths)+d+int(rng.integers(1, 3))
    first = _cup_object(a, b, r, d, None if edge == "none" else edge)
    _draw(c, first)
    objects = [first]
    active = _flow_after(c, r)
    stages = int(rng.choice([1, 2], p=[.8, .2])) if test else int(rng.integers(0, 3))
    for stage in range(stages):
        if not active:
            break
        columns = sorted(active)
        if (len(columns) > 1 and not (test and stage == 0) and rng.random() < .3) or any(
                y-x < 3 for x, y in zip(columns, columns[1:])):
            a, b = columns[0], columns[-1]
            if a == 0 and b == w-1:
                raise ValueError("no flanks")
            r += int(rng.integers(1, 4))
            obj = _bar_object(a, b, r)
            _draw(c, obj)
            objects.append(obj)
        else:
            nodes = []
            for j, (col, (lo, hi)) in enumerate(zip(columns, _partitions(columns, w))):
                cups = _cup_options(col, active[col], lo, hi, w)
                if test and not allow_clipped:
                    cups = [o for o in cups if active[col] <= (o[1]-o[0]-1 if o[3] is None
                            else o[1]-o[0])*o[2] or (o[0] > 0 and o[1] < w-1)]
                bars = []
                for a in range(lo, col+1):
                    for b in range(col, hi+1):
                        nf = int(a > 0) + int(b < w-1)
                        if nf and active[col] % nf == 0 and (allow_clipped or nf == 2):
                            bars.append((a, b, 0, None))
                if test and stage == 0 and j == 0:
                    options = [o for o in cups if (o[1]-o[0]-1 if o[3] is None
                               else o[1]-o[0])*o[2] < active[col]]
                else:
                    options = cups if cups and (not bars or rng.random() < .7) else bars
                if not options:
                    raise ValueError("no valid intermediate node")
                nodes.append(options[int(rng.integers(len(options)))])
            r += max(o[2] for o in nodes) + int(rng.integers(2, 4))
            if r >= 28:
                raise ValueError("too many stages")
            for a, b, depth, ed in nodes:
                obj = (_cup_object(a, b, r, depth, ed) if depth
                       else _bar_object(a, b, r))
                _draw(c, obj)
                objects.append(obj)
        if r >= 28:
            raise ValueError("scene too deep")
        active = _flow_after(c, r)
    floor = _terminal(rng, c, objects, active, r)
    c = c[:min(30, floor+int(rng.integers(1, 4)))].copy()
    out, stats = _simulate(c, 0, 1, _canonical_sources(c))
    if test and sum(e["cup"] and e["excess"] > 0 for e in stats["events"]) < 2:
        raise ValueError("test needs a fresh combination of repeated saturating nodes")
    return [c], objects


def _add_dry_scenery(rng, scenes, objects):
    """Inert geometry is accepted only if both witness allocations stay fixed."""
    baseline = [_simulate(c, 0, 1, _canonical_sources(c))[0] == 1 for c in scenes]
    h, w = scenes[0].shape
    wanted = int(rng.integers(0, 5))
    added = 0
    for _ in range(50):
        if added >= wanted:
            break
        r = int(rng.integers(1, h))
        width = int(rng.integers(1, min(6, w-2)+1))
        if rng.random() < .65 and r >= 2:
            d = int(rng.integers(1, min(5, r)+1))
            a = int(rng.integers(0, w-width-1))
            obj = _cup_object(a, a+width+1, r, d)
            region = [(y, x) for y in range(r-d, r+1)
                      for x in range(a, a+width+2)]
        else:
            a = int(rng.integers(0, w-width+1))
            obj = _bar_object(a, a+width-1, r)
            region = list(obj["cells"])
        if any(c[y, x] != 0 for c in scenes for y, x in region):
            continue
        proposed = [c.copy() for c in scenes]
        for c in proposed:
            _draw(c, obj)
        try:
            actual = [_simulate(c, 0, 1, _canonical_sources(c))[0] == 1 for c in proposed]
        except ValueError:
            continue
        if not all(np.array_equal(x, y) for x, y in zip(actual, baseline)):
            continue
        scenes[:] = proposed
        objects.append(obj)
        added += 1


def _paint(rng, scenes, objects, palette, rotation):
    bg, water = palette[:2]
    available = palette[2:]
    cup_indices = [i for i, o in enumerate(objects) if o["kind"] == "cup"]
    rng.shuffle(cup_indices)
    ncolors = min(len(available), len(cup_indices))
    colors = available[:ncolors]
    assignments = {i: colors[j % ncolors] for j, i in enumerate(cup_indices)}
    pairs = []
    painted = []
    for c in scenes:
        a = np.full(c.shape, bg, dtype=np.int8)
        a[c == 1] = water
        painted.append(a)
    for i, obj in enumerate(objects):
        color = assignments.get(i)
        if color is None:
            color = int(rng.choice(colors))
        for c in painted:
            for r, x in obj["cells"]:
                c[r, x] = color
    for a in painted:
        inp = np.ascontiguousarray(np.rot90(a, rotation), dtype=np.int8)
        out = solve(inp)
        pairs.append({"input": inp, "output": out})
    return pairs


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _fallback(rng, palette, rotation, seen, witness):
    """Finite clean construction, with more distinct canvases than legal pairs.

    It uses the same capacity and endpoint rule. Only scene construction has a
    fallback; solving never relies on an example lookup or stored output.
    """
    for w in range(14, 31):
        for h in range(21, 31):
            c = np.zeros((h, w), dtype=np.int8)
            c[:8, 5] = c[:8, 7] = 1
            objects = [_cup_object(4, 9, 10, 2), _cup_object(2, 5, 13, 1),
                       _bar_object(7, 11, 13), _cup_object(0, 2, 19, 2),
                       _cup_object(5, 7, 19, 4), _cup_object(11, 13, 19, 3)]
            for obj in objects:
                _draw(c, obj)
            scenes = [c]
            if witness:
                low = c.copy()
                low[low == 1] = 0
                low[:2, 5] = low[:2, 7] = 1
                scenes.insert(0, low)
            pairs = _paint(rng, scenes, objects, palette, rotation)
            if not any(_signature(p["input"]) in seen for p in pairs):
                return pairs
    raise AssertionError("more unique canvases excluded than the maximum puzzle size")


def _make(rng, palette, rotation, seen, witness=False, test=False, allow_clipped=True):
    for _ in range(140):
        try:
            scenes, objects = (_witness(rng) if witness
                               else _random_scene(rng, test, allow_clipped))
            _add_dry_scenery(rng, scenes, objects)
            pairs = _paint(rng, scenes, objects, palette, rotation)
            if any(_signature(p["input"]) in seen for p in pairs):
                continue
            if test and not allow_clipped and any(_stats(p)["clipped"] for p in pairs):
                continue
            if witness:
                stats = [_stats(p) for p in pairs]
                if not (stats[0]["retention_only"] and stats[1]["overflow_cups"] >= 2
                        and stats[1]["merges"] > 0 and stats[1]["simultaneous"] > 0):
                    continue
            return pairs
        except (ValueError, IndexError):
            continue
    return _fallback(rng, palette, rotation, seen, witness)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 training, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    rotation = int(rng.integers(4))
    mixed = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    first = _make(rng, palette, rotation, seen, witness=True)
    puzzle["train"].extend(first)
    seen.update(_signature(p["input"]) for p in first)
    for split, count in (("train", num_train-2), ("test", num_test)):
        for i in range(count):
            local_palette = (tuple(map(int, rng.choice(10, 6, replace=False)))
                             if mixed and rng.random() < .7 else palette)
            local_rotation = int(rng.integers(4)) if mixed else rotation
            pair = _make(rng, local_palette, local_rotation, seen,
                         test=split == "test",
                         allow_clipped=any(_stats(p)["clipped"] for p in puzzle["train"]))[0]
            puzzle[split].append(pair)
            seen.add(_signature(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _stats(pair):
    k, bg, water, c, sources = _recognize(pair["input"])
    expected, trace = _simulate(c, bg, water, sources)
    events = trace["events"]
    positive = [e for e in events if e["stored"]]
    trace.update(rotation=k, shape=c.shape, volume=sum(sources.values()),
                 overflow_cups=sum(e["cup"] and e["excess"] > 0 for e in events),
                 partial_cups=sum(0 < e["stored"] < e["capacity"] for e in events),
                 filled_cups=len(positive),
                 retention_only=len(positive) == 1 and not any(e["excess"] for e in events),
                 multi_hit=sum(len(e["arrivals"]) > 1 for e in events),
                 clipped=sum(len(e["flanks"]) == 1 and e["excess"] > 0 for e in events),
                 dry_cups=len(trace["cups"])-len(positive))
    return trace


def _wrong_differs(pair, **shortcut):
    k, bg, water, c, sources = _recognize(pair["input"])
    try:
        wrong, _ = _simulate(c, bg, water, sources, **shortcut)
        return not np.array_equal(np.rot90(wrong, -k), pair["output"])
    except ValueError:
        return True


def validate_generated_puzzle(puzzle):
    """Reconstruct all checks from returned grids, without construction metadata."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _signature(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            k, bg, water, c, sources = _recognize(inp)
            o = np.rot90(out, k)
            solid = (c != bg) & (c != water)
            assert np.array_equal(c[solid], o[solid]), "terrain changed"
            assert np.array_equal(solid, (o != bg) & (o != water))
            assert np.count_nonzero(c == water) == np.count_nonzero(o == water)
            assert np.array_equal(solve(inp), out), "solver disagreement"
            stats = _stats(pair)
            wet = np.zeros(c.shape, dtype=bool)
            for cup in stats["cups"].values():
                r, d, lo, hi = cup["floor"], cup["depth"], cup["lo"], cup["hi"]
                pool = o[r-d:r, lo:hi] == water
                rows = pool.sum(axis=1)
                assert all(v in (0, hi-lo) for v in rows), "partial meniscus"
                assert np.all(np.diff(rows) >= 0), "water above an empty layer"
                wet[r-d:r, lo:hi] |= pool
            assert np.array_equal(wet, o == water), "water outside receiving cups"
            assert all(o[r+1, x] != bg for r, x in np.argwhere(o == water)
                       if r+1 < o.shape[0]), "unsupported liquid"
            for e in stats["events"]:
                assert e["amount"] == sum(v for _, v in e["arrivals"])
                assert e["stored"] + e["excess"] == e["amount"]
                assert not e["excess"] or (e["flanks"] and e["excess"] % len(e["flanks"]) == 0)
            assert sum(e["stored"] for e in stats["events"]) == stats["volume"]
            records[split].append(stats)
    first, second = records["train"][:2]
    assert first["retention_only"] and first["partial_cups"] > 0
    assert second["overflow_cups"] >= 2 and second["merges"] > 0
    assert second["simultaneous"] > 0 and second["multi_hit"] > 0
    a, b = puzzle["train"][:2]
    ka, ba, wa, ca, _ = _recognize(a["input"])
    kb, bb, wb, cb, _ = _recognize(b["input"])
    assert (ka, ba, wa) == (kb, bb, wb) and ca.shape == cb.shape
    assert np.array_equal((ca != ba) & (ca != wa), (cb != bb) & (cb != wb))
    assert first["volume"] < second["volume"]
    assert all(s["overflow_cups"] >= 2 for s in records["test"])
    assert not any(s["clipped"] for s in records["test"]) or any(
        s["clipped"] for s in records["train"]), "clipped test spill lacks training evidence"
    # Removing either inherited mechanism visibly contradicts training. Failure
    # to allocate a conserved integer output also constitutes a contradiction.
    for shortcut in ({"no_storage": True}, {"no_initial_retention": True}, {"no_detours": True},
                     {"no_addition": True}, {"outline_all": True}):
        assert any(_wrong_differs(p, **shortcut) for p in puzzle["train"])


def self_test():
    """Bounded, 42-seed diversity audit plus exact arithmetic fixtures."""
    puzzles, records, signatures, layouts = [], [], set(), set()
    role_colors = [set(), set()]
    widths, depths, volumes, shapes, rotations, source_counts = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p[k]) for s in ("train", "test")
                             for p in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                stats = _stats(pair)
                records.append(stats)
                k, bg, water, c, _ = _recognize(pair["input"])
                normalized = np.where(c == bg, 0, np.where(c == water, 1, 2)).astype(np.int8)
                layouts.add(_signature(normalized))
                role_colors[0].add(bg)
                role_colors[1].add(water)
                volumes.add(stats["volume"])
                shapes.add(pair["input"].shape)
                rotations.add(k)
                source_counts.add(len(stats["sources"]))
                for cup in stats["cups"].values():
                    widths.add(cup["width"])
                    depths.add(cup["depth"])
    assert len(signatures) == 42 and len(layouts) >= 150
    assert len(shapes) >= 45 and len(volumes) >= 15
    assert rotations == set(range(4)) and source_counts >= {1, 2, 3}
    assert all(len(x) >= 8 for x in role_colors)
    assert len(widths) >= 5 and len(depths) >= 5
    assert any(s["clipped"] for s in records)
    assert any(s["dry_cups"] for s in records)
    assert any(s["merges"] and s["partial_cups"] and s["overflow_cups"] >= 2 for s in records)
    # Source length is quantity, rather than distance, and confluences add.
    c = np.zeros((21, 13), dtype=np.int8)
    c[:8, 5] = c[:8, 7] = 1
    for obj in (_cup_object(4, 9, 10, 2),
                _cup_object(2, 5, 13, 1), _bar_object(7, 11, 13),
                _cup_object(0, 2, 19, 2),
                _cup_object(5, 7, 19, 4), _cup_object(11, 12, 19, 3, "right")):
        _draw(c, obj)
    out, trace = _simulate(c, 0, 1, _canonical_sources(c))
    assert np.count_nonzero(out[8:10, 5:9] == 1) == 8
    assert np.count_nonzero(out[12:13, 3:5] == 1) == 2
    assert np.count_nonzero(out[:, 1] == 1) == 1
    assert np.count_nonzero(out[15:19, 6] == 1) == 3
    assert np.count_nonzero(out[:, 12] == 1) == 2
    assert trace["merges"] > 0
    expected = c.copy()
    expected[expected == 1] = 0
    expected[8:10, 5:9] = 1
    expected[12, 3:5] = 1
    expected[18, 1] = 1
    expected[16:19, 6] = 1
    expected[17:19, 12] = 1
    assert np.array_equal(out, expected), "explicit finite-volume arithmetic regression"
    # Both ablations can yield a conserved but different answer, rather than
    # merely being rejected for fractional division or leaving the canvas.
    low = c.copy()
    low[low == 1] = 0
    low[:2, 5] = low[:2, 7] = 1
    correct_low, _ = _simulate(low, 0, 1, _canonical_sources(low))
    ignore_a, _ = _simulate(low, 0, 1, _canonical_sources(low), no_initial_retention=True)
    ignore_b, _ = _simulate(c, 0, 1, _canonical_sources(c), no_detours=True)
    assert (ignore_a == 1).sum() == (correct_low == 1).sum() == 4
    assert not np.array_equal(ignore_a, correct_low)
    assert (ignore_b == 1).sum() == (out == 1).sum() == 16
    assert not np.array_equal(ignore_b, out)
    for k in range(4):
        assert np.array_equal(solve(np.rot90(c, k)), np.rot90(out, k))
    # Input-only solving also works beyond the sampler's source/width/depth
    # ceilings. These are compatible members of the same documented grammar.
    extensions = []
    wide = np.zeros((12, 28), dtype=np.int8)
    wide[:2, 1:26:2] = 1
    _draw(wide, _cup_object(0, 27, 8, 2))
    wide_expected = wide.copy()
    wide_expected[wide_expected == 1] = 0
    wide_expected[7, 1:27] = 1
    extensions.append((wide, wide_expected))
    deep = np.zeros((28, 9), dtype=np.int8)
    deep[:6, 3] = 1
    _draw(deep, _cup_object(2, 4, 25, 12))
    deep_expected = deep.copy()
    deep_expected[deep_expected == 1] = 0
    deep_expected[19:25, 3] = 1
    extensions.append((deep, deep_expected))
    for a, b in extensions:
        for k in range(4):
            assert np.array_equal(solve(np.rot90(a, k)), np.rot90(b, k))
    fallback_rng = np.random.default_rng(2026)
    fallback_seen = set()
    fallback = {"train": _fallback(fallback_rng, (0, 1, 2, 3, 4, 5), 0, fallback_seen, True),
                "test": []}
    fallback_seen.update(_signature(p["input"]) for p in fallback["train"])
    for split, count in (("train", 8), ("test", 4)):
        for _ in range(count):
            pair = _fallback(fallback_rng, (0, 1, 2, 3, 4, 5), 0, fallback_seen, False)[0]
            fallback[split].append(pair)
            fallback_seen.add(_signature(pair["input"]))
    validate_generated_puzzle(fallback)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(314159, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    assert all(np.array_equal(a[k], b[k]) for s in ("train", "test")
               for a, b in zip(puzzles[0][s], generate_puzzle(0)[s])
               for k in ("input", "output"))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (2.5, 1), (True, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(records), "distinct_puzzles": len(signatures),
            "normalized_layouts": len(layouts), "shapes": len(shapes),
            "volumes": sorted(volumes), "cup_widths": sorted(widths),
            "cup_depths": sorted(depths), "source_counts": sorted(source_counts),
            "rotations": sorted(rotations), "background_colors": sorted(role_colors[0]),
            "liquid_colors": sorted(role_colors[1])}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
