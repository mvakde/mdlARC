"""Projected chamber keys: pooled activation, transformed lookup, cavity capture.

Only standard-library facilities and NumPy are used. The solver reconstructs
all geometry and conventions from the input; generation never supplies it a
template, palette, activation code, or receiver identity.
"""

from __future__ import annotations

import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _fingerprint(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _grid(a):
    a = np.asarray(a)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = EIGHT if diagonal else FOUR
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(cells))
    return result


def _regions(a, bg):
    h, w = a.shape
    result = []
    for cells in _components(a == bg):
        boundary = set()
        edge = False
        for r, c in cells:
            edge |= r in (0, h - 1) or c in (0, w - 1)
            for dr, dc in FOUR:
                q = (r + dr, c + dc)
                if 0 <= q[0] < h and 0 <= q[1] < w and a[q] != bg:
                    boundary.add(q)
        result.append((cells, boundary, edge))
    return result


def _bar(template, t, r, c):
    a = _orient(template, t)
    return {"a": a, "r": r, "c": c, "t": t,
            "labels": _orient(np.arange(template.size).reshape(template.shape), t),
            "cells": frozenset((r + i, c + j)
                               for i in range(a.shape[0]) for j in range(a.shape[1]))}


def _ray(bar, side, slot, length, shape):
    labels = bar["labels"]
    p = np.argwhere(labels == side * length + slot)[0]
    q = np.argwhere(labels == (1 - side) * length + slot)[0]
    dr, dc = map(int, p - q)
    r, c = int(p[0]) + bar["r"] + dr, int(p[1]) + bar["c"] + dc
    cells = []
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        cells.append((r, c))
        r, c = r + dr, c + dc
    return tuple(cells)


def _edge(q, shape):
    r, c = q
    if r == 0:
        return "top"
    if r == shape[0] - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == shape[1] - 1:
        return "right"
    return None


def _decode(a):
    """Input-only parsing, template matching, clue pooling and region capture."""
    colors, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, "ambiguous background"
    bg = int(colors[np.argmax(counts)])
    bars, signals, objects = [], [], []
    for cells in _components(a != bg, diagonal=True):
        rr, cc = zip(*cells)
        r, c = min(rr), min(cc)
        patch = a[r:max(rr) + 1, c:max(cc) + 1]
        # A clue can be a singleton or part of a contiguous edge stripe.
        sides = {_edge(q, a.shape) for q in cells}
        if len(sides) == 1 and None not in sides and min(patch.shape) == 1:
            assert all(q[0] not in (0, a.shape[0] - 1)
                       or q[1] not in (0, a.shape[1] - 1) for q in cells), "corner clue"
            signals.extend((r0, c0, int(a[r0, c0])) for r0, c0 in cells)
        elif (patch.size == len(cells) and min(patch.shape) == 2
              and 3 <= max(patch.shape) <= 7 and len(np.unique(patch)) > 1):
            bars.append({"a": patch.copy(), "r": r, "c": c,
                         "cells": frozenset(cells)})
        else:
            assert len({int(a[q]) for q in cells}) == 1, "unrecognized multicolor object"
            objects.append({"cells": cells, "color": int(a[cells[0]]), "holes": []})
    assert 2 <= len(bars) <= 5 and signals and objects
    canonical = bars[0]["a"]
    if canonical.shape[0] != 2:
        canonical = np.rot90(canonical)
    length = canonical.shape[1]
    forms = [_orient(canonical, t) for t in range(8)]
    assert len({_fingerprint(p) for p in forms}) == 8, "ambiguous key orientation"
    labels = np.arange(canonical.size).reshape(canonical.shape)
    for bar in bars:
        matches = [t for t, p in enumerate(forms) if np.array_equal(p, bar["a"])]
        assert len(matches) == 1, "keys do not share one asymmetric template"
        bar["t"] = matches[0]
        bar["labels"] = _orient(labels, matches[0])
    occupied = set().union(*(b["cells"] for b in bars))
    alignments = {}
    for i, bar in enumerate(bars):
        for side in (0, 1):
            for slot in range(length):
                ray = _ray(bar, side, slot, length, a.shape)
                if not ray or occupied.intersection(ray):
                    continue
                local = np.argwhere(bar["labels"] == side * length + slot)[0]
                source = int(bar["a"][tuple(local)])
                alignments.setdefault((ray[-1], source), []).append((i, side, slot))
    observations, sides, slots = [], set(), set()
    allocations = [set() for _ in bars]
    for r, c, color in signals:
        matches = alignments.get(((r, c), color), [])
        assert len(matches) == 1, "ambiguous or unmatched boundary activation"
        i, side, slot = matches[0]
        observations.append((i, slot))
        allocations[i].add(slot)
        sides.add(side)
        slots.add(slot)
    assert len(sides) == 1, "activation lanes disagree"
    side = sides.pop()
    sources = tuple(map(int, canonical[side]))
    fills = tuple(map(int, canonical[1 - side]))
    assert len(set(sources)) == length, "source lane must have unique identities"
    mapping = dict(zip(sources, fills))
    channels = {color: set() for color in sources}
    rays = {}
    for i, bar in enumerate(bars):
        for slot in sorted(slots):
            ray = _ray(bar, side, slot, length, a.shape)
            assert ray and not occupied.intersection(ray), "channel crosses a key"
            rays[i, slot] = set(ray)
            channels[sources[slot]].update(ray)
    ids = {q: i for i, obj in enumerate(objects) for q in obj["cells"]}
    out = a.copy()
    captured = []
    for cells, boundary, edge in _regions(a, bg):
        if edge:
            continue
        wall_colors = {int(a[q]) for q in boundary}
        owners = {ids.get(q, -1) for q in boundary}
        assert len(wall_colors) == 1 and len(owners) == 1 and -1 not in owners, "mixed cavity ownership"
        color = next(iter(wall_colors))
        hit = bool(set(cells).intersection(channels.get(color, set())))
        hole = {"cells": cells, "hit": hit, "color": color}
        objects[next(iter(owners))]["holes"].append(hole)
        if hit:
            for q in cells:
                out[q] = mapping[color]
            captured.append(hole)
    assert captured, "no chamber receives a channel"
    # Reconstruct pedagogical and structural features from pixels, not metadata.
    transferred = any(set(hole["cells"]).intersection(ray)
                      and slot not in allocations[i]
                      for hole in captured for (i, slot), ray in rays.items()
                      if sources[slot] == hole["color"])
    missed = any(not h["hit"] and h["color"] in mapping
                 for o in objects for h in o["holes"])
    active_colors = {sources[s] for s in slots}
    open_on_channel = False
    for obj in objects:
        rr, cc = zip(*obj["cells"])
        obj["box"] = (min(rr), min(cc), max(rr), max(cc))
        obj["diagonal"] = len(_components(_object_mask(obj, a.shape))) > 1
        if not obj["holes"] and obj["color"] in active_colors:
            r0, c0, r1, c1 = obj["box"]
            open_on_channel |= any(r0 < r < r1 and c0 < c < c1
                                   for r, c in channels[obj["color"]])
    features = {
        "background": bg, "length": length, "shape": a.shape,
        "bars": bars, "objects": objects, "sources": sources, "fills": fills,
        "mapping": mapping, "slots": tuple(sorted(slots)), "side": side,
        "allocations": tuple(len(s) for s in allocations), "channels": channels,
        "transforms": tuple(b["t"] for b in bars), "key_count": len(bars),
        "object_count": len(objects), "pooled": all(len(s) < len(slots) for s in allocations),
        "transferred": transferred, "missed_mapped": missed,
        "open_on_channel": open_on_channel,
        "inactive_closed": any(o["holes"] and o["color"] in set(sources) - active_colors for o in objects),
        "unlisted_closed": any(o["holes"] and o["color"] not in mapping for o in objects),
        "partial_object": any(any(h["hit"] for h in o["holes"])
                              and any(not h["hit"] for h in o["holes"]) for o in objects),
        "diagonal_capture": any(o["diagonal"] and any(h["hit"] for h in o["holes"]) for o in objects),
        "chain_capture": any(mapping[h["color"]] in mapping
                             and mapping[mapping[h["color"]]] != mapping[h["color"]] for h in captured),
        "self_capture": any(mapping[h["color"]] == h["color"] for h in captured),
        "irregular_capture": any(_irregular(h["cells"]) for h in captured),
        "captured_count": len(captured), "changed_cells": int(np.count_nonzero(a != out)),
    }
    return out, features


def _object_mask(obj, shape):
    mask = np.zeros(shape, dtype=bool)
    for q in obj["cells"]:
        mask[q] = True
    return mask


def _irregular(cells):
    rr, cc = zip(*cells)
    return len(cells) != (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _decode(_grid(input_grid))[0]


def _dilate(mask):
    out = mask.copy()
    for dr, dc in FOUR:
        rs = slice(max(0, dr), mask.shape[0] + min(0, dr))
        cs = slice(max(0, dc), mask.shape[1] + min(0, dc))
        src_r = slice(max(0, -dr), mask.shape[0] - max(0, dr))
        src_c = slice(max(0, -dc), mask.shape[1] - max(0, dc))
        out[rs, cs] |= mask[src_r, src_c]
    return out


def _profile(rng, h, w):
    cavity = np.zeros((h, w), dtype=bool)
    left, right = 0, w - 1
    for r in range(h):
        if r:
            anchor = int(rng.integers(left, right + 1))
            left = int(rng.integers(anchor + 1))
            right = int(rng.integers(anchor, w))
        cavity[r, left:right + 1] = True
    if np.all(cavity) and h > 1 and w > 1:
        cavity[-1, -1] = False
    return cavity


def _shape(rng, kind):
    """Wall mask plus nominal interior used for geometric placement.

    Actual closed regions are separately reconstructed in the local patch.
    Thus an open receiver has a useful placement interior but no fill regions.
    """
    if kind == "diamond":
        radius = int(rng.integers(1, 4))
        rr, cc = np.indices((2 * radius + 1, 2 * radius + 1))
        dist = abs(rr - radius) + abs(cc - radius)
        wall, nominal = dist == radius, dist < radius
    elif kind == "multi":
        n = int(rng.integers(2, 4))
        hh = int(rng.integers(1, 4))
        widths = list(map(int, rng.integers(1, 4, size=n)))
        nominal = np.zeros((hh + 2, sum(widths) + n + 1), dtype=bool)
        col = 1
        for width in widths:
            hole = _profile(rng, hh, width) if rng.random() < .5 else np.ones((hh, width), bool)
            nominal[1:hh + 1, col:col + width] = hole
            col += width + 1
        wall = ~nominal
    else:
        h, w = map(int, rng.integers(2, 5, size=2))
        cavity = _profile(rng, h, w) if kind == "profile" else np.ones((h, w), bool)
        thickness = 2 if kind != "open" and rng.random() < .18 else 1
        nominal = np.pad(cavity, thickness)
        wall = nominal.copy()
        for _ in range(thickness):
            wall = _dilate(wall)
        wall &= ~nominal
        if kind == "open":
            # Break a complete radial wall, creating four-connected leakage.
            wall[:thickness, thickness + w // 2] = False
    t = int(rng.integers(8))
    wall, nominal = _orient(wall, t), _orient(nominal, t)
    assert len(_components(wall, diagonal=True)) == 1
    holes = [cells for cells, _, edge in _regions(wall.astype(np.int8), 0) if not edge]
    assert (len(holes) == 0) == (kind == "open")
    return wall, nominal, holes


def _block_box(blocked, r, c, h, w):
    blocked[max(0, r - 1):min(blocked.shape[0], r + h + 1),
            max(0, c - 1):min(blocked.shape[1], c + w + 1)] = True


def _choose_clues(rng, bars, template, slots, shape, pool):
    occupied = set().union(*(b["cells"] for b in bars))
    alignments = {}
    length = template.shape[1]
    for i, bar in enumerate(bars):
        for side in (0, 1):
            for slot in range(length):
                ray = _ray(bar, side, slot, length, shape)
                if ray and not occupied.intersection(ray):
                    color = int(template[side, slot])
                    alignments.setdefault((ray[-1], color), []).append((i, side, slot))
    options = {}
    for slot in slots:
        options[slot] = []
        for i, bar in enumerate(bars):
            ray = _ray(bar, 0, slot, length, shape)
            if (ray and alignments.get((ray[-1], int(template[0, slot]))) == [(i, 0, slot)]
                    and all(max(abs(ray[-1][0] - r), abs(ray[-1][1] - c)) > 1 for r, c in occupied)):
                options[slot].append((i, slot, ray[-1]))
        rng.shuffle(options[slot])
        if not options[slot]:
            return None
    answer = []
    preferred = int(rng.integers(len(bars)))
    nodes = 0

    def search(index):
        nonlocal nodes
        nodes += 1
        if nodes > 200:
            return False
        if index == len(slots):
            return not pool or len({x[0] for x in answer}) >= 2
        choices = sorted(options[slots[index]], key=lambda x: x[0] != preferred)
        for entry in choices:
            q = entry[2]
            if any(q == old[2] or (max(abs(q[0] - old[2][0]), abs(q[1] - old[2][1])) <= 1
                                  and _edge(q, shape) != _edge(old[2], shape)) for old in answer):
                continue
            answer.append(entry)
            if search(index + 1):
                return True
            answer.pop()
        return False

    return answer if search(0) else None


def _place_keys(rng, template, transforms, slots, shape, pool):
    length = template.shape[1]
    for _ in range(12):
        placed = []
        for t in transforms:
            ah, aw = _orient(template, t).shape
            for _ in range(80):
                r = int(rng.integers(1, shape[0] - ah))
                c = int(rng.integers(1, shape[1] - aw))
                bar = _bar(template, t, r, c)
                rays = set().union(*(_ray(bar, 0, s, length, shape) for s in slots))
                if min(len(_ray(bar, 0, s, length, shape)) for s in slots) < 5:
                    continue
                if any(any(max(abs(x - y), abs(z - v)) <= 1
                           for y, v in b["cells"]) for x, z in bar["cells"] for b in placed):
                    continue
                if any(rays.intersection(b["cells"]) or bar["cells"].intersection(
                       set().union(*(_ray(b, 0, s, length, shape) for s in slots))) for b in placed):
                    continue
                placed.append(bar)
                break
            else:
                break
        if len(placed) == len(transforms):
            clues = _choose_clues(rng, placed, template, slots, shape, pool)
            if clues is not None:
                return placed, clues
    return None


def _put_receiver(rng, a, blocked, color, kind, focus, channel, mode):
    """Free-placement receiver with a specified observable hit/miss relation."""
    for _ in range(7):
        wall, nominal, holes = _shape(rng, kind)
        h, w = wall.shape
        if h > a.shape[0] or w > a.shape[1]:
            continue
        if mode == "miss":
            positions = [(r, c) for r in range(a.shape[0] - h + 1)
                         for c in range(a.shape[1] - w + 1)]
        else:
            interior = np.argwhere(nominal)
            positions = sorted({(r - int(i), c - int(j)) for r, c in focus for i, j in interior
                                if 0 <= r - i <= a.shape[0] - h and 0 <= c - j <= a.shape[1] - w})
        for index in rng.permutation(len(positions)):
            r, c = positions[int(index)]
            if np.any(blocked[r:r + h, c:c + w]):
                continue
            global_holes = [set((r + i, c + j) for i, j in cells) for cells in holes]
            hits = [bool(cells.intersection(channel)) for cells in global_holes]
            if mode == "miss" and any(hits):
                continue
            if mode == "hit" and not any(cells.intersection(focus) for cells in global_holes):
                continue
            if mode == "partial" and (not any(hits) or all(hits)
                                      or not any(cells.intersection(focus) for cells in global_holes)):
                continue
            a[r:r + h, c:c + w][wall] = color
            _block_box(blocked, r, c, h, w)
            return global_holes
    return None


def _make_pair(rng, lesson, size_bias):
    for attempt in range(70):
        palette = list(map(int, rng.permutation(10)))
        bg = palette[0]
        length = int(rng.integers(4, 8))
        sources = palette[1:length + 1]
        spare = palette[length + 1:]
        fills = list(map(int, rng.choice(sources + spare[:1], length)))
        fills[0], fills[1] = sources[1], spare[0]  # One-step chain, not recursion.
        if lesson == "rotate" or rng.random() < .3:
            fills[2] = sources[2]
        elif lesson == "pool":
            fills[2] = spare[0]
        order = list(map(int, rng.permutation(length)))
        template = np.array([[sources[i] for i in order], [fills[i] for i in order]], dtype=np.int8)
        if len({_fingerprint(_orient(template, t)) for t in range(8)}) != 8:
            continue
        by_color = {int(c): i for i, c in enumerate(template[0])}
        if lesson == "pool":
            slots = [by_color[sources[0]], by_color[sources[2]]]
        elif lesson == "rotate":
            slots = [by_color[sources[i]] for i in range(3)]
        else:
            k = int(rng.integers(1, min(4, length - 1) + 1))
            slots = list(map(int, rng.choice(length, k, replace=False)))
        slots.sort()
        base = int(rng.integers(8))
        if lesson == "pool":
            transforms = [base, base ^ 4]
        elif lesson == "rotate":
            transforms = [base, (base // 4) * 4 + (base + 1) % 4]
        elif lesson == "test":
            transforms = [base, base ^ 4, (base // 4) * 4 + (base + 1) % 4]
            if rng.random() < .25:
                transforms.append(int(rng.integers(8)))
        else:
            n = int(rng.integers(2, 5))
            transforms = list(map(int, rng.integers(8, size=n)))
        low = 23 if lesson in ("pool", "rotate", "test") else 17
        low = min(28, max(low, size_bias))
        h, w = map(int, rng.integers(low, 31, size=2))
        if rng.random() < .2:
            h = w = max(h, w)
        if attempt >= 45:
            h = w = 30
        shape = h, w
        placed = _place_keys(rng, template, transforms, slots, shape, lesson == "pool")
        if placed is None:
            continue
        bars, clues = placed
        a = np.full(shape, bg, dtype=np.int8)
        blocked = np.zeros(shape, dtype=bool)
        for b in bars:
            r, c = b["r"], b["c"]
            ah, aw = b["a"].shape
            a[r:r + ah, c:c + aw] = b["a"]
            _block_box(blocked, r, c, ah, aw)
        allocations = [set() for _ in bars]
        for i, slot, q in clues:
            a[q] = template[0, slot]
            allocations[i].add(slot)
            _block_box(blocked, q[0], q[1], 1, 1)
        rays = {(i, s): set(_ray(b, 0, s, length, shape))
                for i, b in enumerate(bars) for s in range(length)}
        channels = {c: set() for c in sources}
        for s in slots:
            for i in range(len(bars)):
                channels[int(template[0, s])].update(rays[i, s])
        # Prioritize the informative transferred channel in every demonstration.
        first_slot = by_color[sources[0]] if lesson in ("pool", "rotate") else int(rng.choice(slots))
        missing = [i for i in range(len(bars)) if first_slot not in allocations[i]]
        owner = int(rng.choice(missing or list(range(len(bars)))))
        second_slot = by_color[sources[2]] if lesson in ("pool", "rotate") else int(rng.choice(slots))
        inert = spare[-1] if rng.random() < .5 else spare[0]
        inactive = [s for s in range(length) if s not in slots]
        # Each recipe is (source color, shape, focus ray, placement relation).
        recipes = []
        if lesson == "pool":
            recipes = [(sources[0], "profile", rays[owner, first_slot], "hit"),
                       (sources[2], "rectangle", rays[1 - owner, second_slot], "hit"),
                       (sources[0], "rectangle", set(), "miss"),
                       (sources[0], "open", channels[sources[0]], "nominal"),
                       (int(template[0, inactive[0]]), "rectangle", set().union(*(rays[i, inactive[0]] for i in range(len(bars)))), "nominal"),
                       (inert, "rectangle", channels[sources[0]], "nominal")]
        elif lesson == "rotate":
            recipes = [(sources[0], "multi", rays[owner, first_slot], "partial"),
                       (sources[2], "diamond", rays[1 - owner, second_slot], "hit"),
                       (sources[1], "profile", rays[owner, by_color[sources[1]]], "hit"),
                       (sources[0], "open", channels[sources[0]], "nominal"),
                       (inert, "rectangle", channels[sources[0]], "nominal")]
        else:
            first_color = int(template[0, first_slot])
            recipes = [(first_color, "multi" if lesson == "test" else str(rng.choice(["profile", "diamond", "rectangle"])),
                        rays[owner, first_slot], "partial" if lesson == "test" else "hit")]
            for _ in range(int(rng.integers(1, 4))):
                s, i = int(rng.choice(slots)), int(rng.integers(len(bars)))
                recipes.append((int(template[0, s]), str(rng.choice(["profile", "diamond", "rectangle"])), rays[i, s], "hit"))
            recipes.append((first_color, str(rng.choice(["open", "rectangle"])),
                            channels[first_color], "nominal"))
            if rng.random() < .7:
                recipes.append((inert, str(rng.choice(["diamond", "rectangle"])), channels[first_color], "nominal"))
            if rng.random() < .5:
                s = int(rng.choice(inactive))
                recipes.append((int(template[0, s]), "rectangle", rays[0, s], "nominal"))
        receiver_records = []
        for color, kind, focus, mode in recipes:
            holes = _put_receiver(rng, a, blocked, color, kind, focus, channels.get(color, set()), mode)
            if holes is None:
                break
            receiver_records.append((color, holes))
        else:
            # Construction labels come from explicit local cavities and known
            # projected coordinates; they are not obtained by calling solve.
            out = a.copy()
            mapping = dict(zip(map(int, template[0]), map(int, template[1])))
            for color, holes in receiver_records:
                for hole in holes:
                    if hole.intersection(channels.get(color, set())):
                        for q in hole:
                            out[q] = mapping[color]
            try:
                expected, f = _decode(a)
                assert np.array_equal(out, expected)
                if lesson == "pool":
                    assert f["pooled"] and f["transferred"] and f["chain_capture"]
                    assert f["missed_mapped"] and f["open_on_channel"] and f["inactive_closed"] and f["irregular_capture"]
                elif lesson == "rotate":
                    assert f["partial_object"] and f["diagonal_capture"] and f["self_capture"] and f["chain_capture"]
                elif lesson == "test":
                    assert f["partial_object"]
                return {"input": a, "output": out}
            except AssertionError:
                continue
    raise RuntimeError("bounded scene construction exhausted: " + lesson)


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    size_bias = int(rng.integers(17, 25))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            lesson = ("pool", "rotate")[i] if split == "train" and i < 2 else ("test" if split == "test" else "free")
            for _ in range(10):
                pair = _make_pair(rng, lesson, size_bias)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            key = _fingerprint(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            expected, f = _decode(a)
            assert np.array_equal(b, expected), "incorrect chamber capture"
            assert np.array_equal(a[a != f["background"]], b[a != f["background"]]), "original ink changed"
            assert f["changed_cells"] > 0
            # Explicit ablation: removing activation/projection fills more holes.
            no_a = a.copy()
            for obj in f["objects"]:
                if obj["color"] in f["mapping"]:
                    for hole in obj["holes"]:
                        for q in hole["cells"]:
                            no_a[q] = f["mapping"][obj["color"]]
            f["a_ablation_differs"] = not np.array_equal(no_a, b)
            # Removing region capture paints exterior scan cells instead.
            no_b = a.copy()
            for source, cells in f["channels"].items():
                for q in cells:
                    if a[q] == f["background"]:
                        no_b[q] = f["mapping"][source]
            f["b_ablation_differs"] = not np.array_equal(no_b, b)
            features[split].append(f)
    train = features["train"]
    for name in ("pooled", "transferred", "missed_mapped", "open_on_channel", "inactive_closed",
                 "unlisted_closed", "partial_object", "diagonal_capture", "irregular_capture",
                 "chain_capture", "self_capture", "a_ablation_differs", "b_ablation_differs"):
        assert any(f[name] for f in train), "missing training witness: " + name
    assert any(any(t >= 4 for t in f["transforms"]) for f in train), "missing reflected transfer"
    assert any(len({t % 2 for t in f["transforms"]}) > 1 for f in train), "missing quarter-turn transfer"
    return None


def _same(p, q):
    return all(len(p[s]) == len(q[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(p[s], q[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed diversity, symmetry and family-specific regression checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({tuple(_fingerprint(x["input"]) for x in p["train"] + p["test"]) for p in puzzles}) == 42
    records = [_decode(x["input"])[1] for p in puzzles for x in p["train"] + p["test"]]
    assert len({f["shape"] for f in records}) >= 30
    assert {f["length"] for f in records} == {4, 5, 6, 7}
    assert len({f["slots"] for f in records}) >= 15
    assert {len(f["slots"]) for f in records} >= {1, 2, 3, 4}
    assert {f["key_count"] for f in records} >= {2, 3, 4}
    assert len({f["object_count"] for f in records}) >= 4
    assert len({f["background"] for f in records}) == 10
    assert sum(f["partial_object"] for f in records) >= 42
    assert sum(f["diagonal_capture"] for f in records) >= 42
    assert _same(puzzles[0], generate_puzzle(0))
    # Whole-grid D4 and color permutations must commute with the rule.
    for i, pair in enumerate(puzzles[0]["train"] + puzzles[0]["test"]):
        recolor = np.roll(np.arange(10, dtype=np.int8), i + 3)
        for t in range(8):
            a, b = (_orient(recolor[pair[k]], t) for k in ("input", "output"))
            assert np.array_equal(solve(a), b)
    # A hand-sized scene: two reflected keys, split observations, one-step
    # chain, a diagonal cavity, an open ring, and an inactive-source ring.
    a = np.zeros((16, 22), dtype=np.int8)
    key = np.array([[1, 2, 3, 4], [2, 5, 3, 5]], dtype=np.int8)
    a[11:13, 2:6] = key
    a[11:13, 15:19] = np.fliplr(key)
    a[0, 2], a[0, 16] = 1, 3
    a[3:6, 1:4] = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    a[3:6, 15:18] = [[0, 3, 0], [3, 0, 3], [0, 3, 0]]
    a[7:10, 17:20] = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
    a[7:10, 2:5] = [[2, 2, 2], [2, 0, 2], [2, 2, 2]]
    expected = a.copy()
    expected[4, 2], expected[4, 16] = 2, 3
    assert np.array_equal(solve(a), expected)
    # Two matching chambers separated by an unlisted receiver. The nearer
    # chamber and foreign ink do not occlude the farther chamber's capture.
    a = np.zeros((20, 24), dtype=np.int8)
    a[16:18, 2:6] = key
    a[16:18, 17:21] = np.fliplr(key)
    a[0, 2], a[0, 18] = 1, 3
    ring = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)
    a[2:5, 1:4] = ring
    a[6:9, 1:4] = 7 * ring
    a[10:13, 1:4] = ring
    a[4:7, 17:20] = [[0, 3, 0], [3, 0, 3], [0, 3, 0]]
    expected = a.copy()
    expected[3, 2] = expected[11, 2] = 2
    expected[5, 18] = 3
    assert np.array_equal(solve(a), expected)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (3, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len({f["shape"] for f in records}),
            "slot_patterns": len({(f["length"], f["slots"]) for f in records}),
            "key_counts": sorted({f["key_count"] for f in records}),
            "receiver_counts": sorted({f["object_count"] for f in records})}
