"""Registered arrays of reference motifs, combined by parity overprinting.

Only NumPy and the standard library are used. The decoder reads every role
from one input; sampling records are used only to construct a separate label.
"""
from collections import Counter
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _check(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
           "grids must contain integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(cells, diagonal=False):
    remaining = set(cells)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in D8 if diagonal else D4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        groups.append(component)
    return groups


def _bounds(cells):
    rows, cols = zip(*[(p[0], p[1]) for p in cells])
    return min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _pitch(template):
    return _bounds(template)[2:]


def _card(grid, cells, bg, neutral):
    """A specimen card is a rectangular neutral border with a blank inner rim."""
    r, c, h, w = _bounds(cells)
    if h < 5 or w < 5:
        return None
    border = {(y, x) for y in range(r, r + h) for x in range(c, c + w)
              if y in (r, r + h - 1) or x in (c, c + w - 1)}
    if cells != border:
        return None
    patch = grid[r + 1:r + h - 1, c + 1:c + w - 1]
    if not (np.all(patch[0] == bg) and np.all(patch[-1] == bg)
            and np.all(patch[:, 0] == bg) and np.all(patch[:, -1] == bg)):
        return None
    objects = _components(zip(*np.nonzero(patch != bg)))
    if not objects:
        return None  # An empty rectangular stencil is still a stencil.
    signatures = []
    for obj in objects:
        colors = Counter(int(patch[p]) for p in obj)
        singletons = [ink for ink, n in colors.items() if n == 1]
        if len(obj) < 3 or len(singletons) != 1 or neutral in colors:
            return None
        key = singletons[0]
        ar, ac = next(p for p in obj if int(patch[p]) == key)
        signatures.append((key, tuple(sorted((y - ar, x - ac, int(patch[y, x]))
                                             for y, x in obj))))
    if any(sig != signatures[0] for sig in signatures):
        return None
    return {"box": (r, c, h, w), "key": signatures[0][0],
            "template": signatures[0][1], "repetitions": len(signatures),
            "cells": {(y, x) for y in range(r, r + h) for x in range(c, c + w)}}


def _interpret(grid, bg, neutral):
    comps = _components(zip(*np.nonzero(grid == neutral)), diagonal=True)
    cards, masks = [], []
    for comp in comps:
        card = _card(grid, comp, bg, neutral)
        (cards if card is not None else masks).append(card if card is not None else comp)
    _check(cards and masks, "both specimens and site stencils are required")
    dictionary, repetitions = {}, Counter()
    guides = set()
    for card in cards:
        key, template = card["key"], card["template"]
        _check(key not in dictionary or dictionary[key] == template, "inconsistent specimens")
        dictionary[key] = template
        repetitions[key] += card["repetitions"]
        guides |= card["cells"]
    decorations = {ink for key, t in dictionary.items() for r, c, ink in t if (r, c) != (0, 0)}
    _check(not decorations & set(dictionary), "a decoration cannot be a specimen key")
    records = []
    for mask in masks:
        nearby = {(r + dr, c + dc) for r, c in mask
                  for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        choices = []
        for qr, qc in sorted(nearby):
            if not (0 <= qr < grid.shape[0] and 0 <= qc < grid.shape[1]):
                continue
            if (qr, qc) in guides or grid[qr, qc] not in (bg, neutral):
                continue
            for dr, dc in D4:
                sl, tl = (qr - 2 * dr, qc - 2 * dc), (qr - dr, qc - dc)
                if any(not (0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1])
                       or p in guides for p in (sl, tl)):
                    continue
                source, key = int(grid[sl]), int(grid[tl])
                if source in (bg, neutral, key) or key not in dictionary or source in decorations:
                    continue
                choices.append({"source": source, "key": key, "q": (qr, qc),
                                "direction": (dr, dc), "mask": mask,
                                "labels": (sl, tl),
                                "sites": tuple(sorted((r - qr, c - qc) for r, c in mask))})
        _check(len(choices) == 1, "each site stencil must have one label reading")
        records.append(choices[0])
    sources = [rec["source"] for rec in records]
    _check(len(sources) == len(set(sources)), "a source has more than one instruction")
    labels = [p for rec in records for p in rec["labels"]]
    _check(len(labels) == len(set(labels)), "instructions share labels")
    for rec in records:
        guides |= rec["mask"] | set(rec["labels"])
    for rec in records:
        rec["anchors"] = tuple((int(r), int(c)) for r, c in np.argwhere(grid == rec["source"])
                               if (r, c) not in guides)
        _check(rec["anchors"], "an instruction has no original source pixels")
    return {"bg": bg, "neutral": neutral, "dictionary": dictionary,
            "repetitions": repetitions, "cards": cards, "records": records, "guides": guides}


def _decode(input_grid):
    grid = _grid(input_grid)
    frequencies = np.bincount(grid.ravel(), minlength=10)
    bg = int(frequencies.argmax())
    _check(frequencies[bg] > grid.size / 2, "background must occupy more than half the input")
    valid = []
    for neutral in sorted(int(c) for c in np.unique(grid) if c != bg):
        try:
            valid.append(_interpret(grid, bg, neutral))
        except ValueError:
            pass
    _check(len(valid) == 1, "input must identify one neutral/dictionary/instruction interpretation")
    return grid, valid[0]


def _render(grid, info, variant="full"):
    """Count original print events; never convert an intermediate painted pixel to a site."""
    output = grid.copy()
    bg = info["bg"]
    for p in info["guides"]:
        output[p] = bg
    contributions = {}
    clipped = 0
    for rec in info["records"]:
        template = info["dictionary"][rec["key"]]
        h, w = _pitch(template)
        sites = ((0, 0),) if variant == "no_stencil" else rec["sites"]
        ink_template = ((0, 0, rec["key"]),) if variant == "no_motif" else template
        if variant == "opaque":
            lr, lc, _, _ = _bounds(template)
            occupied = {(r, c) for r, c, ink in template}
            ink_template = template + tuple((r, c, bg) for r in range(lr, lr + h)
                                             for c in range(lc, lc + w) if (r, c) not in occupied)
        for ar, ac in rec["anchors"]:
            output[ar, ac] = bg
            for sr, sc in sites:
                for tr, tc, ink in ink_template:
                    p = ar + sr * h + tr, ac + sc * w + tc
                    if not (0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]):
                        clipped += 1
                        continue
                    previous, count = contributions.get(p, (ink, 0))
                    _check(previous == ink, "different ink colors would overlap")
                    contributions[p] = (ink, count + 1)
    for p, (ink, count) in contributions.items():
        output[p] = ink if count % 2 else bg
    return output, contributions, clipped


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, info = _decode(input_grid)
    return _render(grid, info)[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _geometry_key(template):
    cells = [(r, c) for r, c, *_ in template]
    return min(tuple(sorted(_rotate((sign * r, c), turns) for r, c in cells))
               for sign in (-1, 1) for turns in range(4))


def _motif(rng, key, colors, small=False, teaching=False):
    """Connected growth, followed by choosing an occupied registration key."""
    for _ in range(60):
        h, w = int(rng.integers(1, 4 if small else 5)), int(rng.integers(1, 4 if small else 5))
        if h * w < 3:
            continue
        n = int(rng.integers(3, min(9, h * w) + 1))
        if teaching and n < 5:
            continue
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        compact = bool(rng.integers(2))
        while len(cells) < n:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in D4
                               if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
            weights = np.array([sum((r + dr, c + dc) in cells for dr, dc in D4)
                                for r, c in frontier], dtype=float)
            weights = weights ** 2 if compact else 1 / weights
            cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
        if teaching and _bounds(cells)[2] * _bounds(cells)[3] == n:
            continue  # Visible blank tile cells teach transparent copying.
        ar, ac = _pick(rng, sorted(cells))
        offsets = sorted((r - ar, c - ac) for r, c in cells if (r, c) != (ar, ac))
        inks = [colors[0]] * len(offsets)
        if len(offsets) >= 4 and (teaching or rng.random() < .6):
            inks[:2] = [colors[1]] * 2
            rng.shuffle(inks)
        return tuple(sorted([(0, 0, key)] + [(r, c, ink) for (r, c), ink in zip(offsets, inks)]))
    if teaching:
        return ((0, 0, key), (0, 1, colors[0]), (0, 2, colors[0]),
                (1, 0, colors[1]), (1, 1, colors[1]))
    return ((0, 0, key), (1, 0, colors[0]), (1, 1, colors[0]))


def _specimen_item(template, neutral, bg, copies, vertical):
    lr, lc, h, w = _bounds(template)
    shape = ((h + 1) * copies + 3, w + 4) if vertical else (h + 4, (w + 1) * copies + 3)
    patch = np.full(shape, bg, dtype=np.int8)
    patch[0, :] = patch[-1, :] = neutral
    patch[:, 0] = patch[:, -1] = neutral
    for j in range(copies):
        ar = 2 - lr + (j * (h + 1) if vertical else 0)
        ac = 2 - lc + (0 if vertical else j * (w + 1))
        for r, c, ink in template:
            patch[ar + r, ac + c] = ink
    return {"kind": "card", "shape": shape, "patch": patch,
            "reserve": {(r, c) for r in range(shape[0]) for c in range(shape[1])}}


def _site_shape(rng, mode, small):
    if mode == "double":
        return ((0, 0), (0, 1)), (0, 1), ((0, 0), (0, 1))
    if mode == "triple_hole":
        return ((-1, 0), (-1, 1), (0, 1), (1, 0), (1, 1)), (0, 1), ((0, 0), (1, 0), (2, 0))
    if mode == "clip":
        return ((0, 0), (0, 1)), (0, 1), ((0, 0), (0, 1))
    for _ in range(100):
        family = _pick(rng, ("growth", "growth", "corner", "ring") if mode == "new_array"
                       else ("line", "growth", "growth", "corner", "ring", "single"))
        if family == "single":
            cells = {(0, 0)}
        elif family == "line":
            cells = {(0, c) for c in range(int(rng.integers(2, 4 if small else 5)))}
        elif family == "ring":
            cells = {(r, c) for r in range(3) for c in range(3)
                     if r in (0, 2) or c in (0, 2)}
            cells.remove((1, 0))  # Opening for the registration labels.
        elif family == "corner":
            cells = {(0, c) for c in range(3)} | {(r, 2) for r in range(3)}
        else:
            cells = {(0, 0)}
            for _ in range(int(rng.integers(2, 6 if small else 8))):
                frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in D8
                                   if 0 <= r + dr < 3 and 0 <= c + dc < 4} - cells)
                if frontier:
                    cells.add(_pick(rng, frontier))
        if mode == "new_array" and (len(cells) < 3 or min(_bounds(cells)[2:]) < 2):
            continue
        near = {(r + dr, c + dc) for r, c in cells
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        choices = []
        for qr, qc in sorted(near):
            if (qr, qc - 1) in cells or (qr, qc - 2) in cells:
                continue
            # Avoid the reverse interpretation of the same label pair.
            if (qr, qc - 3) in near and (qr, qc - 3) not in cells:
                continue
            choices.append(tuple(sorted((r - qr, c - qc) for r, c in cells)))
        if choices:
            hole_choices = [s for s in choices if (0, 0) not in s]
            sites = _pick(rng, hole_choices if hole_choices and rng.random() < .45 else choices)
            count = int(rng.integers(1, 5 if small else 6))
            style = _pick(rng, ("run", "cluster", "scatter"))
            if style == "run":
                anchors = tuple((0, i) for i in range(count))
            else:
                available = [(r, c) for r in range(2 if style == "cluster" else 3)
                             for c in range(3)]
                anchors = tuple(sorted(available[int(i)] for i in rng.choice(len(available), count, replace=False)))
            return sites, (0, 1), anchors
    if mode == "new_array":
        return ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2)), (0, 1), ((0, 0),)
    return ((0, 0), (0, 1)), (0, 1), ((0, 0),)


def _instruction_item(source, key, sites, direction, neutral, bg):
    dr, dc = direction
    labels = {(-2 * dr, -2 * dc): source, (-dr, -dc): key}
    reserve = set(sites) | set(labels) | {(0, 0)}
    lr, lc, h, w = _bounds(reserve)
    patch = np.full((h, w), bg, dtype=np.int8)
    for r, c in sites:
        patch[r - lr, c - lc] = neutral
    for (r, c), ink in labels.items():
        patch[r - lr, c - lc] = ink
    return {"kind": "instruction", "shape": (h, w), "patch": patch,
            "reserve": {(r - lr, c - lc) for r, c in reserve}, "q_local": (-lr, -lc)}


def _work_item(rec, template, coarse_anchors, clip):
    h, w = _pitch(template)
    anchors = tuple((r * h, c * w) for r, c in coarse_anchors)
    counts = Counter()
    colors = {}
    origins = Counter()
    for ar, ac in anchors:
        for sr, sc in rec["sites"]:
            origins[ar + sr * h, ac + sc * w] += 1
            for tr, tc, ink in template:
                p = ar + sr * h + tr, ac + sc * w + tc
                _check(p not in colors or colors[p] == ink, "construction ink collision")
                colors[p] = ink
                counts[p] += 1
    reserve = set(counts) | set(anchors)
    lr, lc, bh, bw = _bounds(reserve)
    tr0, tc0, _, _ = _bounds(template)
    tile_cells = {(r, c) for r, c, ink in template}
    holes = {(ar + r, ac + c) for (ar, ac), n in origins.items() if n % 2
             for r in range(tr0, tr0 + h) for c in range(tc0, tc0 + w)
             if (r, c) not in tile_cells and (ar + r, ac + c) not in reserve}
    return {"kind": "work", "shape": (bh, bw),
            "reserve": {(r - lr, c - lc) for r, c in reserve},
            "anchors": tuple((r - lr, c - lc) for r, c in anchors),
            "colors": {(r - lr, c - lc): ink for (r, c), ink in colors.items()},
            "counts": {(r - lr, c - lc): n for (r, c), n in counts.items()},
            "transparent_holes": {(r - lr, c - lc) for r, c in holes},
            "source": rec["source"], "clip": clip}


def _pack(rng, items, shape):
    """Free placement of exact footprints, with a blank cell between guide objects."""
    height, width = shape
    occupied = np.zeros(shape, dtype=bool)
    placed = []
    order = sorted(range(len(items)), key=lambda i: (items[i].get("clip", False),
                                                    len(items[i]["reserve"])), reverse=True)
    for i in order:
        item = items[i]
        h, w = item["shape"]
        clip = item.get("clip", False)
        if clip:
            ar0, ac0, ah, aw = _bounds(item["anchors"])
            candidates = [(r, c) for r in range(-ar0, height - ar0 - ah + 1)
                          for c in range(-ac0, width - ac0 - aw + 1)
                          if r < 0 or c < 0 or r + h > height or c + w > width]
        else:
            candidates = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
        rng.shuffle(candidates)
        found = False
        for y, x in candidates:
            cells = {(y + r, x + c) for r, c in item["reserve"]
                     if 0 <= y + r < height and 0 <= x + c < width}
            if not cells or any(occupied[p] for p in cells):
                continue
            if clip:
                visible = [p for p in item["colors"] if 0 <= y + p[0] < height and 0 <= x + p[1] < width]
                if len(visible) < 3 or len(visible) == len(item["colors"]):
                    continue
            halo = {(r + dr, c + dc) for r, c in cells
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if 0 <= r + dr < height and 0 <= c + dc < width}
            for p in halo:
                occupied[p] = True
            placed.append((item, y, x))
            found = True
            break
        if not found:
            return None
    return placed


def _make_pair(rng, palette, stage, index, first_pitch=None, training_shapes=frozenset()):
    bg, neutral = palette[:2]
    pool, decor = palette[2:8], palette[8:]
    for attempt in range(100):
        small = stage == "train" and index == 1 or attempt >= 30
        teaching = stage == "train" and index < 2
        if stage == "train" and index == 0:
            keys, sources, targets, modes = [pool[0]], [pool[3]], [pool[0]], ["double"]
        elif stage == "train" and index == 1:
            keys, sources, targets = [pool[1], pool[0]], [pool[3], pool[1]], [pool[1], pool[0]]
            modes = ["triple_hole", "clip"]
        else:
            count = int(rng.choice((1, 2, 2, 3)))
            order = [int(c) for c in rng.permutation(pool)]
            nkeys = max(1, count - 1) if rng.random() < .3 else count
            keys = order[:nkeys]
            targets = keys + [_pick(rng, keys) for _ in range(count - nkeys)]
            rng.shuffle(targets)
            sources = []
            for target in targets:
                sources.append(_pick(rng, [c for c in order if c != target and c not in sources]))
            modes = ["new_array" if stage == "test" and i == 0 else "free" for i in range(count)]
        templates = {key: _motif(rng, key, decor, small, teaching and i == 0)
                     for i, key in enumerate(keys)}
        if stage == "train" and index == 1:
            # A width-one specimen makes the two original source pixels adjacent.
            # This teaches per-pixel triggering instead of one print per component.
            length = int(rng.integers(3, 6))
            ar = int(rng.integers(length))
            templates[keys[1]] = tuple((r - ar, 0, keys[1] if r == ar else decor[0])
                                       for r in range(length))
        if stage == "train" and index == 1 and tuple(sorted(_pitch(templates[keys[0]]))) == first_pitch:
            continue
        if stage == "test" and all(_geometry_key(t) in training_shapes for t in templates.values()):
            continue
        items, records, work_items = [], [], []
        for key, template in templates.items():
            copies = 2 if teaching else int(rng.choice((1, 1, 2)))
            items.append(_specimen_item(template, neutral, bg, copies, rng.random() < .35))
        for i, (source, key, mode) in enumerate(zip(sources, targets, modes)):
            sites, direction, coarse = _site_shape(rng, mode, small)
            turns = 0 if mode == "clip" else int(rng.integers(4))
            sites = tuple(sorted(_rotate(p, turns) for p in sites))
            direction = _rotate(direction, turns)
            coarse = tuple(sorted(_rotate(p, turns) for p in coarse))
            rec = {"source": source, "key": key, "sites": sites, "direction": direction}
            records.append(rec)
            items.append(_instruction_item(source, key, sites, direction, neutral, bg))
            clip = mode == "clip" or not teaching and rng.random() < .35
            work = _work_item(rec, templates[key], coarse, clip)
            if max(work["shape"]) > (24 if len(records) == 1 else 22):
                break
            work_items.append(work)
            items.append(work)
        if len(work_items) != len(records) or len(records) != len(sources):
            continue
        minimum = 14 if len(records) == 1 else 23 if len(records) == 2 else 27
        if attempt >= 15:
            shape = (30, 30)
        else:
            shape = (int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31)))
        placed = None
        for _ in range(12):
            placed = _pack(rng, items, shape)
            if placed is not None:
                break
        if placed is None:
            continue
        grid, label = np.full(shape, bg, dtype=np.int8), np.full(shape, bg, dtype=np.int8)
        guides, all_prints, transparent_holes = set(), set(), set()
        for item, y, x in placed:
            if item["kind"] == "work":
                transparent_holes |= {(y + r, x + c) for r, c in item["transparent_holes"]
                                      if 0 <= y + r < shape[0] and 0 <= x + c < shape[1]}
                for r, c in item["anchors"]:
                    grid[y + r, x + c] = item["source"]
                for (r, c), n in item["counts"].items():
                    if 0 <= y + r < shape[0] and 0 <= x + c < shape[1]:
                        all_prints.add((y + r, x + c))
                        label[y + r, x + c] = item["colors"][r, c] if n % 2 else bg
            else:
                h, w = item["shape"]
                grid[y:y + h, x:x + w] = item["patch"]
                guides |= {(y + r, x + c) for r, c in item["reserve"]}
        # Optional inactive pixels teach that a target color is not automatically a source.
        inactive = [c for c in pool if c not in sources]
        desired = 1 if stage == "train" and index == 1 else int(rng.integers(4))
        free = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                if grid[r, c] == bg and (r, c) not in guides and (r, c) not in all_prints
                and all((r + dr, c + dc) not in guides for dr, dc in D8)]
        rng.shuffle(free)
        if stage == "train" and index == 0:
            # A visible inert pixel inside a surviving tile's blank cell proves
            # transparency, rather than copying a background-filled rectangle.
            options = sorted(transparent_holes & set(free))
            if not options:
                continue
            marker = _pick(rng, options)
            free = [marker] + [p for p in free if p != marker]
            desired = max(1, desired)
        for p in free[:desired]:
            ink = targets[-1] if teaching and index == 1 else _pick(rng, inactive)
            grid[p] = label[p] = ink
        try:
            decoded_grid, info = _decode(grid)
            predicted, counts, clipped = _render(decoded_grid, info)
            if not np.array_equal(predicted, label):
                continue
            if teaching and index == 0 and (clipped or not any(n == 2 for _, n in counts.values())):
                continue
            if teaching and index == 1 and (not clipped or not any(n >= 3 and n % 2 for _, n in counts.values())):
                continue
            if stage == "test" and np.array_equal(_render(decoded_grid, info, "no_motif")[0], label):
                continue
            # Whole-scene symmetry augmentation moves specimens, labels and prints together.
            turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
            grid, label = np.rot90(grid, turns), np.rot90(label, turns)
            if reflect:
                grid, label = np.fliplr(grid), np.fliplr(label)
            return {"input": grid.copy(), "output": label.copy()}
        except ValueError:
            continue
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + " out of range")
    return value


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    puzzle, seen = {"train": [], "test": []}, set()
    first_pitch, training_shapes = None, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(8):
                pair = _make_pair(rng, palette, split, index, first_pitch, training_shapes)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        _, info = _decode(pair["input"])
                        if index == 0:
                            first_pitch = tuple(sorted(_pitch(next(iter(info["dictionary"].values())))))
                        training_shapes.update(_geometry_key(t) for t in info["dictionary"].values())
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate labels, local recoverability, original-source behavior and teaching evidence."""
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _count(len(puzzle["train"]), "num_train", 2, 10)
    _count(len(puzzle["test"]), "num_test", 1, 4)
    _check(len(puzzle["train"]) > len(puzzle["test"]), "invalid split counts")
    seen, roles, train_info = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, info = _decode(pair["input"])
            label = _grid(pair["output"])
            _check(grid.shape == label.shape, "the output is the whole canvas")
            key = _signature(grid)
            _check(key not in seen, "duplicate input")
            seen.add(key)
            roles.add((info["bg"], info["neutral"]))
            output, counts, clipped = _render(grid, info)
            _check(np.array_equal(output, label), "incorrect parity print")
            _check(not np.array_equal(grid, label), "unchanged pair")
            _check(not set(counts) & info["guides"], "sampled prints must avoid the guide objects")
            # Every local source is taken from the original canvas, including connected groups.
            anchors = {p for rec in info["records"] for p in rec["anchors"]}
            inert = {(int(r), int(c)) for r, c in np.argwhere(grid != info["bg"])
                     if (r, c) not in info["guides"] and (r, c) not in anchors and (r, c) not in counts}
            _check(all(output[p] == grid[p] for p in inert), "unassigned pixels were not preserved")
            if split == "train":
                train_info.append((grid, info, counts, clipped))
    _check(len(roles) == 1, "background and neutral conventions must be shared within a puzzle")
    _check(any(any(n % 2 == 0 for _, n in counts.values()) for _, _, counts, _ in train_info),
           "training must show cancellation of a repeated print")
    _check(any(any(n >= 3 and n % 2 for _, n in counts.values()) for _, _, counts, _ in train_info),
           "training must distinguish parity from deleting every overlap")
    _check(any((0, 0) not in rec["sites"] for _, info, _, _ in train_info for rec in info["records"]),
           "training must show a hole registration")
    _check(any(clipped for _, _, _, clipped in train_info), "training must show boundary clipping")
    _check(any(not clipped for _, _, _, clipped in train_info), "training must show complete copies")
    _check(any(len({ink for r, c, ink in t if (r, c) != (0, 0)}) > 1
               for _, info, _, _ in train_info for t in info["dictionary"].values()),
           "training must expose copying multiple motif colors")
    _check(any(any((r + dr, c + dc) in set(rec["anchors"]) for r, c in rec["anchors"] for dr, dc in D4)
               for _, info, _, _ in train_info for rec in info["records"]),
           "training must distinguish pixels from source components")
    for variant in ("no_stencil", "no_motif", "opaque"):
        _check(any(not np.array_equal(_render(grid, info, variant)[0], _render(grid, info)[0])
                   for grid, info, _, _ in train_info), "training permits omitting " + variant)


def self_test():
    """Bounded 42-seed audit plus count, mutation and direct parity regressions."""
    # Hand-specified expected coordinates, using a 2x2 L specimen. Two source
    # pixels and two sites have event multiplicities 1,2,1 at tile origins.
    # With three of each they have 1,2,3,2,1. No decoder constructs these labels.
    for count in (2, 3):
        grid = np.zeros((12, 20), dtype=np.int8)
        card = _specimen_item(((0, 0, 2), (1, 0, 4), (1, 1, 4)), 8, 0, 2, False)["patch"]
        grid[1:1 + card.shape[0], 1:1 + card.shape[1]] = card
        grid[1, 12:14] = (1, 2)
        grid[1, 14:14 + count] = 8
        for c in range(2, 2 + count * 2, 2):
            grid[8, c] = 1
        grid[11, 18] = 2  # Target-only ink is inactive.
        expected = np.zeros_like(grid)
        for c in ((2, 6) if count == 2 else (2, 6, 10)):
            expected[8, c], expected[9, c], expected[9, c + 1] = 2, 4, 4
        expected[11, 18] = 2
        assert np.array_equal(solve(grid), expected), "manual scale/parity/inert regression"
        bad = grid.copy()
        bad[4, 7] = 3  # Disagreement between the two specimen copies.
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("inconsistent specimens accepted")
    shapes, motifs, site_shapes, pitches, directions, key_counts, anchor_counts = set(), set(), set(), set(), set(), set(), set()
    palettes, puzzles, holes, clipped_pairs, triple_pairs, mixed = set(), set(), 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        training_geometry = set()
        puzzles.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = pair["input"].copy()
                grid, info = _decode(before)
                prediction, counts, clips = _render(grid, info)
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutated its input"
                shapes.add(grid.shape)
                palettes.add((info["bg"], info["neutral"]))
                key_counts.add(len(info["dictionary"]))
                mixed += len({r["direction"] for r in info["records"]}) > 1
                clipped_pairs += bool(clips)
                triple_pairs += any(n >= 3 and n % 2 for _, n in counts.values())
                for key, template in info["dictionary"].items():
                    motifs.add(tuple((r, c) for r, c, ink in template))
                    pitches.add(_pitch(template))
                if split == "train":
                    training_geometry.update(_geometry_key(t) for t in info["dictionary"].values())
                else:
                    assert any(_geometry_key(t) not in training_geometry for t in info["dictionary"].values())
                    assert any(len(rec["sites"]) >= 3 and min(_bounds(rec["sites"])[2:]) >= 2
                               for rec in info["records"])
                for rec in info["records"]:
                    site_shapes.add(rec["sites"])
                    directions.add(rec["direction"])
                    anchor_counts.add(len(rec["anchors"]))
                    holes += (0, 0) not in rec["sites"]
                # Count parity, rather than merely checking the output against solve.
                for p, (ink, n) in counts.items():
                    assert prediction[p] == (ink if n % 2 else info["bg"])
                for card in info["cards"]:
                    assert info["repetitions"][card["key"]] >= 1
    assert len(puzzles) == 42
    assert len(shapes) >= 20 and len(motifs) >= 60 and len(site_shapes) >= 40
    assert len(pitches) >= 8 and len(palettes) >= 20
    assert key_counts == {1, 2, 3} and {1, 2, 3, 4} <= anchor_counts
    assert directions == set(D4) and holes >= 42 and clipped_pairs >= 42 and triple_pairs >= 42 and mixed >= 20
    # Explicit counts include every legal combination; verification also does this externally.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(77, 2, 1), generate_puzzle(77, 2, 1)
    assert all(np.array_equal(pa[k], pb[k]) for s in a for pa, pb in zip(a[s], b[s]) for k in pa)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return {"seeds": 42, "input_shapes": len(shapes), "motif_geometries": len(motifs),
            "registered_site_shapes": len(site_shapes), "tile_pitches": len(pitches),
            "palette_roles": len(palettes), "hole_instructions": holes,
            "clipped_pairs": clipped_pairs, "odd_multiple_pairs": triple_pairs,
            "mixed_direction_pairs": mixed}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
