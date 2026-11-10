"""Perforated jigsaw mosaic: corner materials select globally phased stencils.

Only Python's standard library and NumPy are used. No file I/O or persistent
generation state is needed. All temporary puzzles and diagnostics stay in RAM.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import math

import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "invalid grid shape")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "invalid color")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.uint8).tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    components = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _N4:
                yy, xx = y + dy, x + dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        components.append(np.array(cells, dtype=int))
    return components


def _shape(cells):
    coords = cells - cells.min(axis=0)
    result = np.zeros(tuple(coords.max(axis=0) + 1), dtype=bool)
    result[coords[:, 0], coords[:, 1]] = True
    return result


def _bits(shape, side, y=0, x=0):
    result = 0
    for r, c in np.argwhere(shape):
        result |= 1 << ((int(r) + y) * side + int(c) + x)
    return result


def _placements(shape, side, corner):
    result = {}
    corners = (0, side - 1, side * (side - 1), side * side - 1)
    for k in range(4):
        rotated = np.rot90(shape, k)
        h, w = rotated.shape
        if h > side or w > side:
            continue
        y, x = (0 if corner < 2 else side - h), (0 if corner % 2 == 0 else side - w)
        mask = _bits(rotated, side, y, x)
        hits = tuple(bool(mask & (1 << q)) for q in corners)
        if sum(hits) == 1 and hits[corner]:
            result.setdefault(mask, k)
    return list(result.items())


def _pack(shapes, side, limit=2):
    """Exhaustive corner placements; distinct owner maps, not piece identities."""
    if len(shapes) != 4 or sum(int(s.sum()) for s in shapes) != side * side:
        return []
    if max(shapes[0].shape) > side or not shapes[0][0, 0]:
        return []
    anchor = _bits(shapes[0], side)
    corner_bits = (0, side - 1, side * (side - 1), side * side - 1)
    if sum(bool(anchor & (1 << q)) for q in corner_bits) != 1:
        return []
    choices = {(i, c): _placements(shapes[i], side, c)
               for i in range(1, 4) for c in range(1, 4)}
    full = (1 << (side * side)) - 1
    solutions, signatures = [], set()

    def search(corner, remaining, occupied, masks, pieces, rotations):
        if len(solutions) >= limit:
            return
        if corner == 4:
            if occupied == full and masks not in signatures:
                signatures.add(masks)
                solutions.append({"masks": masks, "pieces": pieces, "rotations": rotations})
            return
        for i in remaining:
            for mask, k in choices[i, corner]:
                if occupied & mask:
                    continue
                search(corner + 1, tuple(j for j in remaining if j != i),
                       occupied | mask, masks + (mask,), pieces + (i,), rotations + (k,))

    search(1, (1, 2, 3), anchor, (anchor,), (0,), (0,))
    return solutions


def _owner(solution, side):
    result = np.full((side, side), -1, dtype=np.int8)
    for corner, mask in enumerate(solution["masks"]):
        for q in range(side * side):
            if (mask >> q) & 1:
                result[q // side, q % side] = corner
    _require(bool(np.all(result >= 0)), "packing did not cover square")
    return result


def _separated(a, b):
    # Half-open bounding boxes, with at least one blank row or column between.
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _decode(input_grid):
    """Read only pixels. The output, RNG and generator are never consulted."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(all(bool(np.all(edge == bg)) for edge in (g[0], g[-1], g[:, 0], g[:, -1])),
             "input needs a uniform background border")
    components = _components(g != bg)
    anchor_candidates = []
    for index, cells in enumerate(components):
        colors, counts = np.unique(g[cells[:, 0], cells[:, 1]], return_counts=True)
        if len(colors) != 5 or np.count_nonzero(counts == 1) != 4 or np.count_nonzero(counts > 1) != 1:
            continue
        carrier = int(colors[counts > 1][0])
        points = cells[g[cells[:, 0], cells[:, 1]] != carrier]
        y, x = (int(v) for v in points.min(axis=0))
        if set(map(tuple, points)) != {(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)}:
            continue
        anchor_candidates.append((index, carrier, g[y:y + 2, x:x + 2].copy()))
    _require(len(anchor_candidates) == 1, "exactly one readable keyed anchor required")
    ai, carrier, legend = anchor_candidates[0]
    keys = tuple(int(v) for v in legend.flat)
    _require(len(set(keys)) == 4 and bg not in keys and carrier not in keys, "ambiguous key palette")
    plain = [i for i, cells in enumerate(components)
             if np.all(g[cells[:, 0], cells[:, 1]] == carrier)]
    _require(len(plain) == 3, "exactly three plain carrier pieces required")
    pieces = [components[i] for i in [ai] + plain]
    piece_area = np.zeros(g.shape, dtype=bool)
    boxes = []
    for cells in pieces:
        piece_area[cells[:, 0], cells[:, 1]] = True
        y, x = cells.min(axis=0)
        yy, xx = cells.max(axis=0) + 1
        boxes.append(tuple(int(v) for v in (y, x, yy, xx)))
    _require(all(_separated(a, b) for a, b in itertools.combinations(boxes, 2)),
             "piece bounding boxes need a blank-cell gap")
    external = (g != bg) & ~piece_area
    colors = [int(v) for v in np.unique(g[external])]
    _require(set(keys) <= set(colors) and carrier not in colors and 4 <= len(colors) <= 6,
             "one stencil per key color and at most two distractors required")
    patterns, stencil_boxes = {}, []
    for color in colors:
        cells = np.argwhere(external & (g == color))
        y, x = (int(v) for v in cells.min(axis=0))
        yy, xx = (int(v) for v in cells.max(axis=0) + 1)
        box = (y, x, yy, xx)
        register = g[y:yy, x:xx]
        _require(all(2 <= d <= 4 for d in register.shape), "stencil register axes must be 2..4")
        _require(not piece_area[y:yy, x:xx].any() and np.all((register == bg) | (register == color)),
                 "stencil register overlaps other content")
        mask = register == color
        _require(0 < int(mask.sum()) < mask.size, "stencils must have ink and empty cells")
        _require(all(edge.any() for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1])),
                 "stencil ink must span its register")
        patterns[color] = mask
        stencil_boxes.append(box)
    all_boxes = boxes + stencil_boxes
    _require(all(_separated(a, b) for a, b in itertools.combinations(all_boxes, 2)),
             "all register/piece bounding boxes need a blank-cell gap")
    _require(len({m.shape for m in patterns.values()}) == 1, "register dimensions must agree within input")
    shapes = [_shape(cells) for cells in pieces]
    area = sum(len(cells) for cells in pieces)
    side = math.isqrt(area)
    _require(side * side == area and 4 <= side <= 30, "piece area must determine a square")
    solutions = _pack(shapes, side)
    _require(len(solutions) == 1, "visible geometry must have one corner-owner tiling")
    solution = solutions[0]
    return {"background": bg, "carrier": carrier, "key": keys, "patterns": patterns,
            "side": side, "shapes": shapes, "boxes": boxes, "pieces": pieces,
            "solution": solution, "owner": _owner(solution, side)}


def _transform(mask, orientation):
    return np.rot90(np.fliplr(mask) if orientation >= 4 else mask, orientation % 4)


def _render(info, *, orientation=0, positive=False, local=False, once=False,
            union=False, mask_order=None, rim=True):
    """Rule renderer; optional switches are explicit counterfactual diagnostics."""
    owner, side = info["owner"], info["side"]
    keys = info["key"]
    patterns = [_transform(info["patterns"][color], orientation) for color in keys]
    if mask_order is not None:
        patterns = [patterns[i] for i in mask_order]
    out = np.asarray(keys, dtype=np.int8)[owner].copy()
    origins = [np.argwhere(owner == i).min(axis=0) for i in range(4)]
    for r in range(side):
        for c in range(side):
            if rim and (r in (0, side - 1) or c in (0, side - 1)):
                continue
            piece = int(owner[r, c])
            y, x = r - 1, c - 1
            if local:
                y, x = y - int(origins[piece][0]), x - int(origins[piece][1])
            masks = patterns if union else [patterns[piece]]
            hit = False
            for mask in masks:
                h, w = mask.shape
                if not once or (0 <= y < h and 0 <= x < w):
                    hit |= bool(mask[y % h, x % w])
            if hit != positive:
                out[r, c] = info["background"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_decode(input_grid))


def _key_locations(owner):
    return [(r, c) for r in range(len(owner) - 1) for c in range(len(owner) - 1)
            if np.all(owner[r:r + 2, c:c + 2] == 0)]


def _grow(rng, side):
    """Connected weighted growth with independent corner speeds and local costs."""
    owner = np.full((side, side), -1, dtype=np.int8)
    speed = np.exp(rng.uniform(-.65, .65, 4))
    weights = rng.uniform(.25, 2.8, (side, side))
    expanded = np.zeros((side, side), dtype=bool)
    queue = []
    for i, (y, x) in enumerate(((0, 0), (0, side - 2), (side - 2, 0), (side - 2, side - 2))):
        owner[y:y + 2, x:x + 2] = i
        for r in range(y, y + 2):
            for c in range(x, x + 2):
                heapq.heappush(queue, (0., float(rng.random()), i, r, c))
    while queue:
        distance, _, i, r, c = heapq.heappop(queue)
        if expanded[r, c] or owner[r, c] not in (-1, i):
            continue
        expanded[r, c] = True
        if owner[r, c] == -1:
            owner[r, c] = i
        for dr, dc in _N4:
            y, x = r + dr, c + dc
            if 0 <= y < side and 0 <= x < side and owner[y, x] == -1:
                heapq.heappush(queue, (distance + float(weights[y, x]) * speed[i],
                                       float(rng.random()), i, y, x))
    return owner


def _seams(rng, side):
    owner = np.empty((side, side), dtype=np.int8)
    split = int(rng.integers(2, side - 1))
    cuts = [int(rng.integers(2, side - 1)), int(rng.integers(2, side - 1))]
    for r in range(side):
        band = int(r >= split)
        if 1 < r < side - 2 and rng.random() < .7:
            cuts[band] = int(np.clip(cuts[band] + rng.integers(-1, 2), 2, side - 2))
        x = cuts[band]
        owner[r, :x], owner[r, x:] = 2 * band, 2 * band + 1
    return owner


def _exchange(rng, owner):
    """Move seam cells only when source and destination stay connected."""
    side = len(owner)
    for _ in range(int(rng.integers(side, 3 * side + 1))):
        r, c = (int(v) for v in rng.integers(1, side - 1, 2))
        old = int(owner[r, c])
        new = int(_pick(rng, [int(owner[r + dr, c + dc]) for dr, dc in _N4]))
        if old == new:
            continue
        owner[r, c] = new
        if (any(len(_components(owner == i)) != 1 for i in (old, new))
                or not _key_locations(owner)):
            owner[r, c] = old
    return owner


def _acceptable(owner, demonstrate):
    side = len(owner)
    corners = tuple(int(owner[r, c]) for r, c in ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)))
    if corners != (0, 1, 2, 3) or not _key_locations(owner):
        return None
    shapes = []
    for i in range(4):
        cells = np.argwhere(owner == i)
        if len(cells) < (7 if i == 0 else 5) or np.count_nonzero(owner[1:-1, 1:-1] == i) < 3:
            return None
        if len(_components(owner == i)) != 1:
            return None
        shapes.append(_shape(cells))
    if demonstrate:
        canonical = []
        for shape in shapes:
            versions = {(np.rot90(shape, k).shape, np.rot90(shape, k).tobytes()) for k in range(4)}
            if len(versions) != 4:
                return None
            canonical.append(min(versions))
        if len(set(canonical)) != 4:
            return None
    solutions = _pack(shapes, side)
    expected = tuple(_bits(owner == i, side) for i in range(4))
    if len(solutions) != 1 or solutions[0]["masks"] != expected:
        return None
    return shapes, solutions[0]


def _partition(rng, side, demonstrate):
    for attempt in range(180):
        owner = _grow(rng, side) if rng.random() < .6 else _seams(rng, side)
        if rng.random() < .75:
            owner = _exchange(rng, owner)
        accepted = _acceptable(owner, demonstrate)
        if accepted is not None:
            return owner, accepted[0], accepted[1]
    # Deterministic tabbed seams are a finite last resort, never an example bank.
    for y in range(2, side - 2):
        for x in range(2, side - 2):
            owner = np.empty((side, side), dtype=np.int8)
            owner[:y, :x], owner[:y, x:] = 0, 1
            owner[y:, :x], owner[y:, x:] = 2, 3
            owner[1, x], owner[y, x + 1] = 0, 1
            owner[y + 1, x - 1], owner[y - 1, 1] = 3, 2
            accepted = _acceptable(owner, demonstrate)
            if accepted is not None:
                return owner, accepted[0], accepted[1]
    raise RuntimeError("no valid partition within bounded search")


def _mask(rng, h, w):
    for _ in range(60):
        style = int(rng.integers(8))
        m = np.zeros((h, w), dtype=bool)
        if style == 0:
            m[0, 0] = m[-1, -1] = True
        elif style == 1:
            m[0] = m[-1] = True
        elif style == 2:
            m[int(rng.integers(h)), :] = True
            m[:, int(rng.integers(w))] = True
        elif style == 3:
            m[:] = True
            m.flat[int(rng.integers(m.size))] = False
        else:
            m = rng.random((h, w)) < float(rng.uniform(.2, .8))
            if style == 4:
                m |= np.fliplr(m)
        if (0 < m.sum() < m.size and all(edge.any() for edge in (m[0], m[-1], m[:, 0], m[:, -1]))):
            return m
    m = np.zeros((h, w), dtype=bool)
    m[0, 0] = m[-1, -1] = True
    return m


def _material_ok(info, out):
    interior = np.ones(out.shape, dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    return all(np.any(interior & (info["owner"] == i) & (out == info["background"]))
               and np.any(interior & (info["owner"] == i) & (out == info["key"][i])) for i in range(4))


def _evidence(info, out):
    side = info["side"]
    rr, cc = np.indices((side, side))
    quadrants = (2 * (rr >= side // 2) + (cc >= side // 2)).astype(np.int8)
    variants = {"local": _render(info, local=True), "once": _render(info, once=True),
                "union": _render(info, union=True), "rim": _render(info, rim=False),
                "quadrants": _render(dict(info, owner=quadrants))}
    flags = {name: not np.array_equal(other, out) for name, other in variants.items()}
    holes = out == info["background"]
    flags["selection"] = any(not np.array_equal(
        _render(info, mask_order=p) == info["background"], holes)
        for p in ((1, 0, 2, 3), (0, 2, 1, 3), (3, 1, 2, 0)))
    return flags


def _carried_output(info, turns):
    patterns = dict(info["patterns"])
    for color, k in zip(info["key"], turns):
        patterns[color] = np.rot90(patterns[color], k)
    return _render(dict(info, patterns=patterns))


def _hypotheses(info, out):
    return {(i, positive) for i in range(8) for positive in (False, True)
            if np.array_equal(_render(info, orientation=i, positive=positive), out)}


def _layout(rng, images, side):
    """Varied shelf groups and axis; deterministic fitting fallback."""
    candidates = []
    for attempt in range(55):
        transpose = bool(rng.integers(2))
        dims = [im.shape[::-1] if transpose else im.shape for im in images]
        width = int(rng.integers(max(16, max(w for h, w in dims)), 29))
        order = list(rng.permutation(len(images)))
        if attempt % 3 == 0:
            order.sort(key=lambda i: -dims[i][0])
        shelves, offsets = [], [None] * len(images)
        for i in order:
            h, w = dims[i]
            fits = [s for s in range(len(shelves)) if h <= shelves[s][1] and shelves[s][2] + w <= width]
            if fits:
                s = int(_pick(rng, fits))
                y, sh, x = shelves[s]
                offsets[i] = (y + int(rng.integers(sh - h + 1)), x)
                shelves[s][2] += w + int(rng.integers(1, 4))
            else:
                y = 0 if not shelves else shelves[-1][0] + shelves[-1][1] + int(rng.integers(1, 4))
                if y + h > 28:
                    break
                offsets[i] = (y, 0)
                shelves.append([y, h, w + int(rng.integers(1, 4))])
        else:
            for s, (y, h, _) in enumerate(shelves):
                members = [i for i, pos in enumerate(offsets) if y <= pos[0] < y + h]
                right = max(offsets[i][1] + dims[i][1] for i in members)
                shift = int(rng.integers(width - right + 1))
                for i in members:
                    offsets[i] = (offsets[i][0], offsets[i][1] + shift)
            if transpose:
                offsets = [(x, y) for y, x in offsets]
            candidates.append(offsets)
    if candidates:
        offsets = _pick(rng, candidates)
    else:
        # Four <=N-square pieces, then a row of <=5 small stencils; when N<=8
        # six stencils can instead use two rows. At N<=11 this fits in 28x28.
        order = list(rng.permutation(4))
        offsets = [None] * len(images)
        y = 0
        for group in (order[:2], order[2:]):
            x = 0
            for i in group:
                offsets[i] = (y, x)
                x += images[i].shape[1] + 1
            y += max(images[i].shape[0] for i in group) + 1
        cards = list(rng.permutation(np.arange(4, len(images))))
        groups = [cards] if len(cards) <= 5 else [cards[:3], cards[3:]]
        for group in groups:
            x = 0
            for i in group:
                offsets[i] = (y, x)
                x += images[i].shape[1] + 1
            y += max(images[i].shape[0] for i in group) + 1
    mins = np.min(offsets, axis=0)
    offsets = [(int(y - mins[0]), int(x - mins[1])) for y, x in offsets]
    extent = [max(offsets[i][a] + im.shape[a] for i, im in enumerate(images)) for a in (0, 1)]
    pads = []
    for length in extent:
        room = 30 - length
        _require(room >= 2, "layout cannot retain its background border")
        before = int(rng.integers(1, min(4, room)))
        after = int(rng.integers(1, min(4, room - before) + 1))
        pads.append((before, after))
    offsets = [(r + pads[0][0], c + pads[1][0]) for r, c in offsets]
    canvas = tuple(extent[a] + sum(pads[a]) for a in (0, 1))
    return offsets, canvas


def _make_pair(rng, side, register, first=False, evidence=False, previous_key=None):
    owner, shapes, solution = _partition(rng, side, first)
    rotations = [0] + ([int(v) for v in rng.permutation([1, 2, 3])] if first
                       else [int(rng.integers(4)) for _ in range(3)])
    distractors = int(rng.integers(0, 3 if side <= 8 else 2))
    palette = [int(v) for v in rng.choice(10, 6 + distractors, replace=False)]
    bg, carrier, keys = palette[0], palette[1], tuple(palette[2:6])
    if previous_key == keys:
        keys = keys[1:] + keys[:1]
    colors = list(keys) + palette[6:]
    info = {"side": side, "background": bg, "key": keys, "owner": owner}
    for _ in range(160):
        patterns = {color: _mask(rng, *register) for color in colors}
        info["patterns"] = patterns
        out = _render(info)
        if not _material_ok(info, out):
            continue
        if evidence and not all(_evidence(info, out).values()):
            continue
        if first and _hypotheses(info, out) != {(0, False)}:
            continue
        if first and np.array_equal(_carried_output(info, [(-k) % 4 for k in rotations]), out):
            continue
        break
    else:
        # Retry the complete geometry rather than inventing a special rule.
        return None
    y, x = _pick(rng, _key_locations(owner))
    anchor_origin = np.argwhere(owner == 0).min(axis=0)
    images = []
    for i, shape in enumerate(shapes):
        im = np.full(shape.shape, bg, dtype=np.int8)
        im[shape] = carrier
        if i == 0:
            yy, xx = y - int(anchor_origin[0]), x - int(anchor_origin[1])
            im[yy:yy + 2, xx:xx + 2] = np.asarray(keys).reshape(2, 2)
        images.append(np.rot90(im, rotations[i]).copy())
    order = [int(v) for v in rng.permutation(len(colors))]
    for index in order:
        color = colors[index]
        images.append(np.where(patterns[color], color, bg).astype(np.int8))
    for _ in range(12):
        offsets, canvas = _layout(rng, images, side)
        reading = sorted(range(4), key=lambda i: offsets[i])
        if not first or reading != [0, 1, 2, 3]:
            break
    else:
        return None
    inp = np.full(canvas, bg, dtype=np.int8)
    for im, (r, c) in zip(images, offsets):
        inp[r:r + im.shape[0], c:c + im.shape[1]] = im
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen, first_key = {"train": [], "test": []}, set(), None
    for index in range(nt + ne):
        split = "train" if index < nt else "test"
        side = (8 if index == 0 else int(rng.integers(7, 9)) if index == 1
                else int(rng.integers(6, 9)) if split == "train" else int(rng.integers(9, 12)))
        register = ((3, 3) if index == 0 else _pick(rng, [(2, 3), (3, 2)]) if index == 1
                    else (int(rng.integers(2, 5)), int(rng.integers(2, 5))))
        for _ in range(32):
            pair = _make_pair(rng, side, register, first=index == 0,
                              evidence=index < 2, previous_key=first_key if index == 1 else None)
            if pair is not None and _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if index == 0:
                    first_key = tuple(int(v) for v in _decode(pair["input"])["key"])
                break
        else:
            raise RuntimeError("could not construct a distinct evidenced pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_infos, train_sides, test_sides = set(), [], [], []
    candidates = set(itertools.product(range(8), (False, True)))
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _decode(inp)
            _require(np.array_equal(out, _render(info)), "wrong keyed periodic perforation")
            _require(_material_ok(info, out), "each material needs holes and surviving interior")
            _require(not np.array_equal(inp, out), "identity pair")
            side = info["side"]
            rim = np.zeros(out.shape, dtype=bool)
            rim[0] = rim[-1] = True
            rim[:, 0] = rim[:, -1] = True
            _require(not np.any(out[rim] == info["background"]), "rim must remain intact")
            for i in range(4):
                _require(np.count_nonzero(info["owner"][1:-1, 1:-1] == i) >= 3, "insufficient interior material")
            if split == "train":
                train_infos.append(info)
                train_sides.append(side)
                candidates &= _hypotheses(info, out)
            else:
                test_sides.append(side)
    _require(candidates == {(0, False)}, "training leaves stencil orientation/polarity ambiguous")
    _require(train_infos[0]["key"] != train_infos[1]["key"], "key must visibly change")
    periods = [next(iter(a["patterns"].values())).shape for a in train_infos[:2]]
    _require(periods[0] != periods[1], "initial demonstrations must expose variable visible periods")
    for i in range(2):
        info = train_infos[i]
        _require(all(_evidence(info, np.asarray(puzzle["train"][i]["output"])).values()),
                 "initial demonstrations lack a required counterexample")
    first = train_infos[0]
    _require(set(first["solution"]["rotations"][1:]) == {1, 2, 3}, "first demonstration must cover nonzero turns")
    _require(not np.array_equal(_carried_output(first, first["solution"]["rotations"]),
                                np.asarray(puzzle["train"][0]["output"])),
             "first demonstration must reject rotating stencils with pieces")
    shape_versions = [{(np.rot90(s, k).shape, np.rot90(s, k).tobytes()) for k in range(4)}
                      for s in first["shapes"]]
    _require(all(len(v) == 4 for v in shape_versions) and len({min(v) for v in shape_versions}) == 4,
             "initial rotation evidence requires noncongruent asymmetric pieces")
    reading = sorted(range(4), key=lambda i: first["boxes"][i][:2])
    corners = {piece: c for c, piece in enumerate(first["solution"]["pieces"])}
    _require([corners[i] for i in reading] != [0, 1, 2, 3], "scatter order must not predict corners")
    _require(min(test_sides) > max(train_sides), "generated tests must transfer to a larger square")


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(bytes((len(puzzle[split]),)))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded streaming audit. Returns small measurements, writes nothing."""
    signatures, canvases, cuts, motif_shapes, backgrounds, carrier_colors = set(), set(), set(), set(), set(), set()
    sides, distractions, turns, areas, occupancies = set(), set(), set(), set(), set()
    connected = disconnected = symmetric = asymmetric = rectangular = 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _fingerprint(puzzle)
        _require(signature == _fingerprint(generate_puzzle(seed)), "nondeterministic generation")
        signatures.add(signature)
        local_canvases, local_cuts = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                info = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "solve mismatch")
                canvases.add(pair["input"].shape)
                cuts.add((info["side"], info["owner"].tobytes()))
                local_canvases.add(pair["input"].shape)
                local_cuts.add((info["side"], info["owner"].tobytes()))
                sides.add(info["side"])
                backgrounds.add(info["background"])
                carrier_colors.add(info["carrier"])
                distractions.add(len(info["patterns"]) - 4)
                turns.update(info["solution"]["rotations"])
                areas.add(tuple(int(np.count_nonzero(info["owner"] == i)) for i in range(4)))
                rectangular += any(bool(s.all()) for s in info["shapes"])
                for mask in info["patterns"].values():
                    motif_shapes.add(mask.shape)
                    occupancies.add((mask.size, int(mask.sum())))
                    conn = len(_components(mask)) == 1
                    connected += int(conn)
                    disconnected += int(not conn)
                    asym = len({(_transform(mask, i).shape, _transform(mask, i).tobytes()) for i in range(8)}) == 8
                    asymmetric += int(asym)
                    symmetric += int(not asym)
        _require(len(local_canvases) >= 3 and len(local_cuts) >= 3, "within-puzzle geometry collapsed")
    _require(len(signatures) == 42, "seed diversity collapsed")
    _require(len(canvases) >= 40 and len(cuts) >= 150 and len(areas) >= 100, "geometric diversity collapsed")
    _require(sides == set(range(6, 12)) and len(motif_shapes) == 9, "scale/period coverage missing")
    _require(backgrounds == set(range(10)) and carrier_colors == set(range(10)), "palette roles collapsed")
    _require(distractions == {0, 1, 2} and turns == {0, 1, 2, 3}, "count/rotation coverage missing")
    _require(min(connected, disconnected, symmetric, asymmetric) >= 20 and rectangular > 0,
             "motif connectivity/symmetry or simple geometry missing")
    # Count coverage is sequential and in memory, never archived per request.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(618863187, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "count request ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4., 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(123, 2, 1)
    # All ten colors, including background, can be relabeled without a new rule.
    permutation = np.array([3, 7, 0, 9, 2, 6, 8, 1, 5, 4], dtype=np.int8)
    relabeled = {s: [{k: permutation[v] for k, v in pair.items()} for pair in p[s]]
                 for s in ("train", "test")}
    validate_generated_puzzle(relabeled)
    for pair in relabeled["test"]:
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "palette equivariance failed")
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "canvas_shapes": len(canvases), "owner_maps": len(cuts), "area_profiles": len(areas),
            "sides": sorted(sides), "motif_shapes": sorted(motif_shapes),
            "backgrounds": sorted(backgrounds), "carrier_colors": sorted(carrier_colors),
            "distractors": sorted(distractions), "turns": sorted(turns),
            "connected_masks": connected, "disconnected_masks": disconnected,
            "symmetric_masks": symmetric, "asymmetric_masks": asymmetric,
            "pairs_with_rectangular_piece": rectangular, "occupancy_profiles": len(occupancies)}
