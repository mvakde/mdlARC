"""Census wavefronts: component counts address diagonals of labeled lattices.

Only NumPy and the standard library are used. No parent module is imported.
The parser and solver use visible input cells; sampling records are used only
to construct labels. See task_spec.md for the complete family convention.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((0 <= a) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(a, bg, diagonal=False):
    unseen = {(int(r), int(c)) for r, c in np.argwhere(a != bg)}
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while unseen:
        start = min(unseen)
        color = int(a[start])
        cells, todo = {start}, [start]
        unseen.remove(start)
        while todo:
            r, c = todo.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in unseen and int(a[p]) == color:
                    unseen.remove(p)
                    cells.add(p)
                    todo.append(p)
        result.append((color, cells))
    return result


def _box(cells):
    return min(r for r, c in cells), max(r for r, c in cells), min(c for r, c in cells), max(c for r, c in cells)


def _perimeter(box):
    r0, r1, c0, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}


def _canonical(a):
    """Find the right-hand chart, full-height divider and exceptional corner."""
    found = []
    for turn in range(4):
        g = np.rot90(a, turn)
        h, w = g.shape
        for d in range(5, w - 6):
            div = int(g[0, d])
            if not np.all(g[:, d] == div):
                continue
            inside = g[1:-1, d + 2:-1]
            if not inside.size or not np.all(inside == inside[0, 0]):
                continue
            bg = int(inside[0, 0])
            corners = [(0, d + 1), (0, w - 1), (h - 1, d + 1), (h - 1, w - 1)]
            counts = Counter(int(g[p]) for p in corners)
            if sorted(counts.values()) != [1, 3]:
                continue
            frame = next(c for c, n in counts.items() if n == 3)
            origin = next(c for c, n in counts.items() if n == 1)
            if len({bg, div, frame, origin}) != 4 or np.count_nonzero(g == origin) != 1:
                continue
            if not np.array_equal(g[0, d + 2:-1], g[-1, d + 2:-1]):
                continue
            if not np.array_equal(g[1:-1, d + 1], g[1:-1, -1]):
                continue
            stage = g[:, :d]
            if not np.any(stage != bg) or np.any(np.isin(stage, [div, frame, origin])):
                continue
            edge = np.concatenate((g[0, d + 2:-1], g[-1, d + 2:-1],
                                   g[1:-1, d + 1], g[1:-1, -1]))
            if np.any(np.isin(edge, [bg, div, origin])):
                continue
            corner = next(p for p in corners if int(g[p]) == origin)
            found.append((g.copy(), turn, d, bg, div, frame, origin, corner))
    _require(len(found) == 1, "chart layout must be uniquely visible")
    return found[0]


def _parse(value):
    a = _grid(value)
    g, turn, d, bg, div, frame, origin, corner = _canonical(a)
    h, w = g.shape
    stage = g[:, :d]
    parts = _components(stage, bg)
    capsules = []
    for color, cells in parts:
        box = _box(cells)
        r0, r1, c0, c1 = box
        if min(r1 - r0 + 1, c1 - c0 + 1) >= 5 and cells == _perimeter(box):
            _require((r1 - r0) % 2 == (c1 - c0) % 2 == 0, "odd capsule dimensions")
            capsules.append(dict(color=color, box=box, outline=cells))
    # A template can itself be an outlined odd rectangle. Only outermost rings
    # are capsules; rings inside them are reference-mask pixels.
    capsules = [t for t in capsules if not any(
        u is not t and u['box'][0] < t['box'][0] <= t['box'][1] < u['box'][1]
        and u['box'][2] < t['box'][2] <= t['box'][3] < u['box'][3]
        for u in capsules)]
    _require(1 <= len(capsules) <= 3, "one through three template capsules")
    _require(len({t['color'] for t in capsules}) == len(capsules), "one capsule per category")
    excluded = set()
    for t in capsules:
        r0, r1, c0, c1 = t['box']
        rectangle = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        _require(not excluded & rectangle, "capsules cannot overlap")
        excluded |= rectangle
        interior = g[r0 + 1:r1, c0 + 1:c1]
        _require(np.all(np.isin(interior, [bg, t['color']])), "capsule contents")
        _require(np.all(interior[[0, -1], :] == bg) and np.all(interior[:, [0, -1]] == bg), "template moat")
        pixels = {(int(r), int(c)) for r, c in np.argwhere(g == t['color'])
                  if r0 < r < r1 and c0 < c < c1}
        _require(bool(pixels), "empty template")
        mr0, mr1, mc0, mc1 = _box(pixels)
        _require((mr1 - mr0) % 2 == (mc1 - mc0) % 2 == 0, "odd template bounding box")
        cr, cc = (mr0 + mr1) // 2, (mc0 + mc1) // 2
        t['offsets'] = {(r - cr, c - cc) for r, c in pixels}
        mask = np.full((mr1 - mr0 + 1, mc1 - mc0 + 1), bg, np.int8)
        for r, c in pixels:
            mask[r - mr0, c - mc0] = t['color']
        _require(len(_components(mask, bg, True)) == 1, "one eight-connected template")
        t['shape'] = mask.shape
        t['mask'] = tuple(sorted(t['offsets']))
        rs = [r for r in range(1, h - 1) if int(g[r, d + 1]) == t['color']]
        cs = [c for c in range(d + 2, w - 1) if int(g[0, c]) == t['color']]
        _require(rs and cs, "every capsule has both coordinate axes")
        t['rows'] = sorted(rs, reverse=corner[0] != 0)
        t['cols'] = sorted(cs, reverse=corner[1] == w - 1)
    external = []
    for color, cells in parts:
        if cells & excluded:
            _require(cells <= excluded, "object touches a capsule")
        else:
            _require(len(cells) <= 8, "external objects have at most eight cells")
            external.append((color, cells))
    labels = np.full(stage.shape, -1, int)
    for i, t in enumerate(capsules):
        r0, r1, c0, c1 = t['box']
        labels[r0:r1 + 1, c0:c1 + 1] = i
    for i, (color, cells) in enumerate(external, len(capsules)):
        for cell in cells:
            labels[cell] = i
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        left = labels[max(0, dr):stage.shape[0] + min(0, dr),
                      max(0, dc):stage.shape[1] + min(0, dc)]
        right = labels[max(0, -dr):stage.shape[0] + min(0, -dr),
                       max(0, -dc):stage.shape[1] + min(0, -dc)]
        _require(not np.any((left >= 0) & (right >= 0) & (left != right)), "source objects or capsules touch")
    counts = Counter(color for color, cells in external)
    areas = Counter()
    for color, cells in external:
        areas[color] += len(cells)
    possible = set()
    for t in capsules:
        t['count'], t['area'] = counts[t['color']], areas[t['color']]
        _require(t['count'] <= len(t['rows']) + len(t['cols']) - 1, "count exceeds last wavefront")
        union = set()
        for r in t['rows']:
            for c in t['cols']:
                for dr, dc in t['offsets']:
                    p = r + dr, c + dc
                    _require(1 <= p[0] < h - 1 and d + 2 <= p[1] < w - 1, "stamp crosses chart border")
                    union.add(p)
        _require(not possible & union, "different categories' possible stamps overlap")
        possible |= union
    return dict(grid=g, turn=turn, d=d, bg=bg, divider=div, frame=frame,
                origin=origin, corner=corner, capsules=capsules, external=external)


def _prediction(p, policy="wave"):
    g, bg = p['grid'], p['bg']
    out = g.copy()
    for color, cells in p['external']:
        for cell in cells:
            out[cell] = bg
    for t in p['capsules']:
        rows, cols, n = t['rows'], t['cols'], t['count']
        if policy == "fixed_corner":
            rows, cols = sorted(rows), sorted(cols)
        elif policy.startswith("screen_corner_"):
            corner = int(policy[-1])
            for _ in range(p['turn']):
                corner = [2, 0, 3, 1][corner]
            rows = sorted(rows, reverse=corner >= 2)
            cols = sorted(cols, reverse=corner % 2 == 1)
        if policy == "area":
            n = t['area']
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                selected = n > 0 and i + j == n - 1
                if policy == "all":
                    selected = n > 0
                elif policy == "queue":
                    selected = i * len(cols) + j < n
                elif policy == "cumulative":
                    selected = i + j < n
                elif policy == "metric":
                    selected = n > 0 and abs(r - rows[0]) + abs(c - cols[0]) == n - 1
                if selected:
                    offsets = {(0, 0)} if policy == "dots" else t['offsets']
                    if policy == "solid_box":
                        bh, bw = t['shape']
                        offsets = {(dr, dc) for dr in range(-bh // 2 + 1, bh // 2 + 1)
                                   for dc in range(-bw // 2 + 1, bw // 2 + 1)}
                    for dr, dc in offsets:
                        out[r + dr, c + dc] = t['color']
    return np.rot90(out, -p['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution; count objects, then stamp the addressed frontier."""
    return _prediction(_parse(input_grid))


def _mask(rng, h, w):
    m = np.zeros((h, w), bool)
    kind = int(rng.integers(6))
    if min(h, w) == 1 or kind == 0:
        m[:] = True
    elif kind == 1:
        m[[0, -1], :] = True
        m[:, [0, -1]] = True
    elif kind == 2 and h == w:
        m = np.eye(h, dtype=bool)
    elif kind == 3 and h == w:
        yy, xx = np.indices((h, w))
        m = abs(yy - h // 2) + abs(xx - w // 2) == h // 2
    else:
        # Random eight-connected paths joining all four bounding-box extremes.
        points = [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                  (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
        rng.shuffle(points)
        r, c = points[0]
        m[r, c] = True
        for rr, cc in points[1:]:
            while (r, c) != (rr, cc):
                r += int(np.sign(rr - r))
                c += int(np.sign(cc - c))
                m[r, c] = True
        for r, c in rng.permutation([(r, c) for r in range(h) for c in range(w)]):
            if rng.random() < .35 and m[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
                m[r, c] = True
    if rng.random() < .5:
        m = m[::-1]
    if rng.random() < .5:
        m = m[:, ::-1]
    return m.copy()


def _object(rng, area):
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = sorted({(r + dr, c + dc) for r, c in cells
                           for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _partition(rng, total, count):
    values = [1] * count
    for _ in range(total - count):
        choices = [i for i, v in enumerate(values) if v < 8]
        values[int(rng.choice(choices))] += 1
    return values


def _place_stage(rng, h, d, bg, categories, masks, specifications, keep_area_colors=()):
    """Place capsule rectangles and freely scattered separated objects."""
    for attempt in range(90):
        stage = np.full((h, d), bg, np.int8)
        blocked = np.zeros((h, d), bool)
        rectangles = []
        # The fallback retains every count and mandatory teaching area. Other
        # objects become singletons, so crowded optional geometry cannot fail.
        fallback = attempt == 89
        top = 0
        order = list(map(int, rng.permutation(len(categories))))
        good = True
        for idx in order:
            m, color = masks[idx], categories[idx]
            bh, bw = m.shape
            sh, sw = bh + 4, bw + 4
            if fallback:
                r, c = top, 0
                top += sh + 1
                if r + sh > h:
                    good = False
                    break
            else:
                choices = [(r, c) for r in range(h - sh + 1) for c in range(d - sw + 1)
                           if not blocked[r:r + sh, c:c + sw].any()]
                if not choices:
                    good = False
                    break
                r, c = choices[int(rng.integers(len(choices)))]
            box = r, r + sh - 1, c, c + sw - 1
            for cell in _perimeter(box):
                stage[cell] = color
            view = stage[r + 2:r + 2 + bh, c + 2:c + 2 + bw]
            view[m] = color
            blocked[max(0, r - 1):min(h, r + sh + 1), max(0, c - 1):min(d, c + sw + 1)] = True
            rectangles.append(box)
        if not good:
            continue
        ordered = sorted(specifications, key=lambda item: -len(item[1]))
        for color, original in ordered:
            if fallback:
                area = len(original) if color in keep_area_colors else 1
                shape = tuple((i, 0) for i in range(area))
            else:
                shape = original
            bh = 1 + max(r for r, c in shape)
            bw = 1 + max(c for r, c in shape)
            choices = [(r, c) for r in range(h - bh + 1) for c in range(d - bw + 1)
                       if all(not blocked[r + dr, c + dc] for dr, dc in shape)]
            if not choices:
                good = False
                break
            r, c = choices[0] if fallback else choices[int(rng.integers(len(choices)))]
            for dr, dc in shape:
                rr, cc = r + dr, c + dc
                stage[rr, cc] = color
                blocked[max(0, rr - 1):min(h, rr + 2), max(0, cc - 1):min(d, cc + 2)] = True
        if good:
            return stage, rectangles
    raise AssertionError("bounded stage packing failed")


def _coordinates(rng, h, fw, masks, multiplicities, witness=False):
    """Random independent axes, with a guaranteed row-band packing fallback."""
    for attempt in range(121):
        rows, cols, used_r, used_c, unions = [], [], set(), set(), set()
        band_top = 1
        spacing = 2 if witness else 1
        band_slack = h - 2 - sum(m.shape[0] + spacing * (nr - 1) for m, (nr, nc) in zip(masks, multiplicities)) - (len(masks) - 1)
        order = list(map(int, rng.permutation(len(masks))))
        records = {}
        good = True
        for idx in order:
            m = masks[idx]
            bh, bw = m.shape
            nr, nc = multiplicities[idx]
            available_r = [r for r in range(1 + bh // 2, h - 1 - bh // 2) if r not in used_r]
            available_c = [c for c in range(1 + bw // 2, fw - 1 - bw // 2) if c not in used_c]
            if attempt == 120:
                gap = int(rng.integers(band_slack + 1)) if band_slack > 0 else 0
                band_slack -= gap
                band_top += gap
                rs = [band_top + bh // 2 + spacing * i for i in range(nr)]
                band_top += bh + spacing * (nr - 1) + 1
            else:
                rs = sorted(map(int, rng.choice(available_r, nr, replace=False)))
                if witness and any(b - a < 2 for a, b in zip(rs, rs[1:])):
                    good = False
                    break
            if len(available_c) < nc:
                good = False
                break
            cs = sorted(map(int, rng.choice(available_c, nc, replace=False)))
            cells = {(r + int(dr) - bh // 2, c + int(dc) - bw // 2)
                     for r in rs for c in cs for dr, dc in np.argwhere(m)}
            if unions & cells:
                good = False
                break
            unions |= cells
            used_r.update(rs)
            used_c.update(cs)
            records[idx] = rs, cs
        if good:
            for idx in range(len(masks)):
                rows.append(records[idx][0])
                cols.append(records[idx][1])
            return rows, cols
    raise AssertionError("coordinate packing failed")


def _make_pair(rng, categories, counts, forced_areas, distractors, bg, corner_index,
               rotation, witness=None, test=False):
    k = len(categories)
    # Narrow source capsules leave room for objects; one-category scenes can
    # also use wider 5x5 motifs. Dimensions do not encode any wavefront value.
    dimensions = [(1, 1), (1, 3), (3, 1), (3, 3), (5, 1), (5, 3)]
    if k == 1:
        dimensions += [(3, 5), (5, 5)]
    masks = [_mask(rng, *dimensions[int(rng.integers(len(dimensions)))]) for _ in categories]
    if witness is not None:
        masks[0] = witness.copy()
        masks[1] = _mask(rng, 3, 3)
    if k == 3:
        masks = [_mask(rng, int(rng.choice([1, 3])), int(rng.choice([1, 3]))) for _ in categories]
    if test and counts[0] == 4:
        masks[0] = _mask(rng, 3, 3)
    multiplicities = []
    for n in counts:
        options = [(r, c) for r in range(1, 5) for c in range(1, 4) if r + c - 1 >= n]
        if test and n == 4:
            options = [(r, c) for r, c in options if r >= 2 and c >= 2]
        if witness is not None:
            multiplicities = [(2, 3), (3, 2)]
            break
        multiplicities.append(options[int(rng.integers(len(options)))])
    h = int(rng.integers(24 if k == 3 else 21, 31))
    h = max(h, sum(m.shape[0] + 5 for m in masks) - 1)
    d = int(rng.integers(11, 13))
    min_fw = max(11, sum(c for r, c in multiplicities) + 2 * max(m.shape[1] // 2 for m in masks) + 2)
    fw = int(rng.integers(min_fw, 30 - d))
    used_colors = set(categories) | set(distractors) | {bg}
    structural = list(map(int, rng.choice([c for c in range(10) if c not in used_colors], 3, replace=False)))
    divider, frame, origin = structural
    specs = []
    for idx, (color, n) in enumerate(zip(categories, counts)):
        areas = forced_areas.get(idx, [int(rng.integers(1, 7)) for _ in range(n)])
        specs.extend((color, _object(rng, area)) for area in areas)
    specs.extend((c, _object(rng, int(rng.integers(1, 7)))) for c in distractors)
    stage, boxes = _place_stage(rng, h, d, bg, categories, masks, specs,
                                keep_area_colors={categories[i] for i in forced_areas})
    rows, cols = _coordinates(rng, h, fw, masks, multiplicities, witness=witness is not None)
    inp = np.full((h, d + 1 + fw), bg, np.int8)
    inp[:, :d] = stage
    inp[:, d] = divider
    inp[[0, -1], d + 1:] = frame
    inp[:, [d + 1, -1]] = frame
    corner = [(0, d + 1), (0, inp.shape[1] - 1), (h - 1, d + 1), (h - 1, inp.shape[1] - 1)][corner_index]
    inp[corner] = origin
    for color, rs, cs in zip(categories, rows, cols):
        for r in rs:
            inp[r, d + 1] = inp[r, -1] = color
        for c in cs:
            inp[0, d + 1 + c] = inp[-1, d + 1 + c] = color
    # Output construction uses sampled masks, counts and coordinate records,
    # rather than calling solve or recovering components from the input.
    out = inp.copy()
    out[:, :d] = bg
    for r0, r1, c0, c1 in boxes:
        # Capsule boxes are in random placement order; copy their visible cells.
        out[r0:r1 + 1, c0:c1 + 1] = inp[r0:r1 + 1, c0:c1 + 1]
    for color, n, m, rs, cs in zip(categories, counts, masks, rows, cols):
        rs = sorted(rs, reverse=corner[0] != 0)
        cs = sorted(cs, reverse=corner[1] == inp.shape[1] - 1)
        bh, bw = m.shape
        for i, r in enumerate(rs):
            for j, c in enumerate(cs):
                if n > 0 and i + j + 1 == n:
                    view = out[r - bh // 2:r + bh // 2 + 1,
                               d + 1 + c - bw // 2:d + 1 + c + bw // 2 + 1]
                    view[m] = color
    return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, nq = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed % (1 << 64))
    anchor, dropped, replacement, novel = map(int, rng.choice(10, 4, replace=False))
    initial_corner, rotation = int(rng.integers(4)), int(rng.integers(4))
    collision_area = int(rng.integers(4, 7))
    witness = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)
    if rng.random() < .5:
        witness = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], bool)
    puzzle, seen = {"train": [], "test": []}, set()
    previous_bg = None
    for index in range(nt + nq):
        training = index < nt
        local = index if training else index - nt
        split = "train" if training else "test"
        if index == 0:
            colors, counts, noise = [anchor, dropped], [1, 2], [novel]
            forced = {0: [collision_area], 1: _partition(rng, collision_area + 3, 2)}
        elif index == 1:
            colors, counts, noise = [anchor, replacement], [3, 2], [dropped]
            forced = {0: _partition(rng, collision_area, 3), 1: _partition(rng, collision_area + 5, 2)}
        else:
            k = int(rng.choice([1, 2, 3], p=[.25, .55, .20]))
            pool = [c for c in range(10) if c != novel or (not training and local != 0)]
            colors = list(map(int, rng.choice(pool, k, replace=False)))
            counts = list(map(int, rng.integers(0, 4, k)))
            if not training and local == 0:
                colors[0] = novel
                counts[0] = 4  # new ordinal layer, including finite-axis truncation
            noise_pool = [c for c in range(10) if c not in colors]
            noise = list(map(int, rng.choice(noise_pool, int(rng.integers(0, 3)), replace=True)))
            forced = {}
        bg_pool = [c for c in range(10) if c not in set(colors + noise) and c != previous_bg]
        bg = int(rng.choice(bg_pool))
        # A one-quarter-turn change to both the canonical corner and scene
        # makes the displayed origin opposite, preventing cancellation.
        corner = initial_corner if index == 0 else [2, 0, 3, 1][initial_corner] if index == 1 else int(rng.integers(4))
        turn = rotation if index == 0 else (rotation + 1) % 4 if index == 1 else int(rng.integers(4))
        if not training and local == 0:
            turn = (rotation + int(rng.choice([2, 3]))) % 4
        for attempt in range(20):
            pair = _make_pair(rng, colors, counts, forced, noise, bg, corner, turn,
                              witness=witness if index < 2 else None, test=not training)
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            raise AssertionError("failed to generate distinct inputs")
        seen.add(key)
        previous_bg = bg
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, details = set(), {"train": [], "test": []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape and _key(inp) not in seen, "dimensions or duplicate")
            seen.add(_key(inp))
            p = _parse(inp)
            canonical_out = np.rot90(out, p['turn'])
            g, d, bg = p['grid'], p['d'], p['bg']
            _require(np.array_equal(canonical_out[:, d:d + 2], g[:, d:d + 2]), "divider or left frame changed")
            _require(np.array_equal(canonical_out[[0, -1], d + 1:], g[[0, -1], d + 1:])
                     and np.array_equal(canonical_out[:, -1], g[:, -1]), "border changed")
            # Reconstruct destination by inverse mask membership at each pixel.
            expected = g.copy()
            for color, cells in p['external']:
                for cell in cells:
                    expected[cell] = bg
            for r in range(1, g.shape[0] - 1):
                for c in range(d + 2, g.shape[1] - 1):
                    votes = [t['color'] for t in p['capsules']
                             for i, rr in enumerate(t['rows']) for j, cc in enumerate(t['cols'])
                             if t['count'] > 0 and i + j + 1 == t['count']
                             and (r - rr, c - cc) in t['offsets']]
                    _require(len(set(votes)) <= 1, "ambiguous destination ownership")
                    expected[r, c] = votes[0] if votes else bg
            _require(np.array_equal(expected, canonical_out), "wrong frontier, mask, preservation or erasure")
            _require(np.array_equal(solve(inp), out), "solve disagrees")
            details[split].append(p)
    train = details['train']
    records = [(t['color'], t['count'], t['area']) for p in train for t in p['capsules']]
    _require(any(c == cc and a == aa and n != nn for c, n, a in records for cc, nn, aa in records), "area/count collision missing")
    _require(any(n == nn and a != aa for c, n, a in records for cc, nn, aa in records), "area independence missing")
    _require({1, 2, 3} <= {n for c, n, a in records}, "first three wavefronts must be taught")
    for policy in ('all', 'dots', 'queue', 'cumulative', 'metric', 'fixed_corner', 'solid_box', 'area',
                   'screen_corner_0', 'screen_corner_1', 'screen_corner_2', 'screen_corner_3'):
        _require(any(not np.array_equal(_prediction(p, policy), _prediction(p)) for p in train), "training permits shortcut: " + policy)
    categories = {t['color'] for p in train for t in p['capsules']}
    discarded = {c for p in train for c, cells in p['external'] if c not in {t['color'] for t in p['capsules']}}
    _require(bool(categories & discarded), "local color roles not taught")
    _require(any({t['color'] for t in p['capsules']} - categories for p in details['test']), "novel test category missing")
    _require(any(t['count'] >= 4 for p in details['test'] for t in p['capsules']), "test layer shift missing")


def self_test():
    """Bounded 42-seed variation audit, count coverage, and semantic mutations."""
    shapes, motifs, object_masks, palettes, count_values, products, corners = (set() for _ in range(7))
    signatures, category_numbers = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_key(pair[k]) for s in ('train', 'test') for pair in puzzle[s] for k in ('input', 'output'))
        signatures.add(signature)
        local_shapes = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _parse(pair['input'])
                shapes.add(pair['input'].shape)
                local_shapes.add(pair['input'].shape)
                palettes.add((p['bg'], p['divider'], p['frame'], p['origin']))
                corners.add((p['corner'][0] == 0, p['corner'][1] == p['grid'].shape[1] - 1))
                category_numbers.add(len(p['capsules']))
                for t in p['capsules']:
                    motifs.add((t['shape'], t['mask']))
                    count_values.add(t['count'])
                    products.add((len(t['rows']), len(t['cols'])))
                for color, cells in p['external']:
                    r0, r1, c0, c1 = _box(cells)
                    object_masks.add(tuple(sorted((r - r0, c - c0) for r, c in cells)))
        _require(len(local_shapes) > 1, "within-puzzle geometry frozen")
    _require(len(signatures) == 42, "duplicate seed puzzles")
    _require(len(shapes) >= 30 and len(palettes) >= 100, "canvas/palette variation")
    _require(len(motifs) >= 25 and len(object_masks) >= 40, "shape vocabulary")
    _require(category_numbers == {1, 2, 3} and count_values >= {0, 1, 2, 3, 4}, "category/count variation")
    _require(len(products) >= 8 and len(corners) == 4, "lattice variation")
    # Hand-built boundary member: an outlined 5x5 template inside an outer
    # capsule, six tokens, and the last layer of a nonuniform 4x3 lattice.
    fixture = np.zeros((23, 29), np.int8)
    fixture[:, 11] = 8
    fixture[[0, -1], 12:] = 9
    fixture[:, [12, -1]] = 9
    fixture[0, 12] = 7
    for cell in _perimeter((0, 8, 0, 8)) | _perimeter((2, 6, 2, 6)):
        fixture[cell] = 2
    token_cells = [(r, 0) for r in (11, 13, 15, 17, 19, 21)]
    for cell in token_cells:
        fixture[cell] = 2
    for r in (3, 8, 14, 19):
        fixture[r, 12] = fixture[r, -1] = 2
    for c in (15, 20, 25):
        fixture[0, c] = fixture[-1, c] = 2
    expected = fixture.copy()
    for cell in token_cells:
        expected[cell] = 0
    for r, c in _perimeter((17, 21, 23, 27)):
        expected[r, c] = 2
    for turn in range(4):
        for shift in (0, 2, 5):
            _require(np.array_equal(solve(np.rot90((fixture + shift) % 10, turn)),
                                    np.rot90((expected + shift) % 10, turn)),
                     "outlined template, last frontier, rotation or recoloring")
    empty = fixture.copy()
    for cell in token_cells:
        empty[cell] = 0
    _require(np.array_equal(solve(empty), empty), "empty census should draw no frontier")
    # Determinism, every legal explicit count and invalid-type handling.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(901 + nt, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "explicit counts")
    a, b = generate_puzzle(887), generate_puzzle(887)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), "determinism")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    for policy in ('all', 'dots', 'queue', 'cumulative', 'area', 'solid_box'):
        bad = {s: [{k: np.array(v, copy=True) for k, v in x.items()} for x in p[s]] for s in p}
        pair = bad['train'][1]
        pair['output'] = _prediction(_parse(pair['input']), policy)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted a semantic mutation: " + policy)
    return {"seeds": 42, "distinct_puzzles": len(signatures), "canvas_shapes": len(shapes),
            "template_masks": len(motifs), "object_masks": len(object_masks),
            "palettes": len(palettes), "coordinate_products": len(products)}


if __name__ == '__main__':
    print(self_test())
