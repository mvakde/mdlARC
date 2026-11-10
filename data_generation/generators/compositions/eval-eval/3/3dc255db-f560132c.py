"""Counted jigsaw teeth: decode local reservoirs, then assemble and recolor.

Only NumPy and the standard library are used. No files, seed, examples, or
generation records are read by solve(). No diagnostics are written here.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np

_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must be integral")
    _require(bool(np.all((a >= 0) & (a <= 9))), "bad grid colors")
    return a.astype(np.int16, copy=True)


def _components(cells, diagonal=False):
    left = set(cells)
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1))
    answer = []
    while left:
        start = min(left)
        left.remove(start)
        comp, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in left:
                    left.remove(p)
                    comp.add(p)
                    stack.append(p)
        answer.append(comp)
    return answer


def _normal(cells):
    cells = {(int(r), int(c)) for r, c in cells}
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _rotate(cells, turns):
    result = set(cells)
    for _ in range(turns % 4):
        result = {(c, -r) for r, c in result}  # clockwise; translate afterwards
    return _normal(result)


def _dims(cells):
    return max(r for r, c in cells) + 1, max(c for r, c in cells) + 1


def _mask(cells, n, offset=(0, 0)):
    r0, c0 = offset
    return sum(1 << ((int(r) + r0) * n + int(c) + c0) for r, c in cells)


def _pack(shapes, n):
    """Exhaust all corner placements; stop at two different colored tilings."""
    if len(shapes) != 4 or sum(map(len, shapes)) != n * n:
        return []
    corners = (0, n - 1, n * (n - 1), n * n - 1)
    if (0, 0) not in shapes[0] or max(_dims(shapes[0])) > n:
        return []
    anchor = _mask(shapes[0], n)
    if sum(bool(anchor & (1 << j)) for j in corners) != 1:
        return []
    candidates = {}
    for i in range(1, 4):
        for corner in range(1, 4):
            unique = {}
            for k in range(4):
                cells = _rotate(shapes[i], k)
                h, w = _dims(cells)
                if h > n or w > n:
                    continue
                offset = (0 if corner < 2 else n - h, 0 if corner % 2 == 0 else n - w)
                bits = _mask(cells, n, offset)
                if bits & (1 << corners[corner]) and sum(bool(bits & (1 << j)) for j in corners) == 1:
                    unique.setdefault(bits, k)
            candidates[i, corner] = tuple(unique.items())
    answers, seen = [], set()

    def visit(corner, remaining, occupied, masks, ids, rotations):
        if len(answers) == 2:
            return
        if corner == 4:
            if occupied == (1 << (n * n)) - 1 and masks not in seen:
                seen.add(masks)
                answers.append({"masks": masks, "pieces": ids, "rotations": rotations})
            return
        for i in remaining:
            for bits, k in candidates[i, corner]:
                if bits & occupied == 0:
                    visit(corner + 1, tuple(j for j in remaining if j != i), occupied | bits,
                          masks + (bits,), ids + (i,), rotations + (k,))

    visit(1, (1, 2, 3), anchor, (anchor,), (0,), (0,))
    return answers


def _canonical(p, direction):
    r, c = p
    return ((r, c), (-c, r), (-r, -c), (c, -r))[direction]


def _world(p, direction):
    y, x = p
    return ((y, x), (x, -y), (-y, -x), (-x, y))[direction]


def _read_outline(cells, direction):
    """Read thin pointed envelopes from cross-sections, without a size bank."""
    local = {_canonical(p, direction) for p in cells}
    y0 = min(y for y, x in local)
    y1 = max(y for y, x in local)
    rows = [sorted(x for y, x in local if y == level) for level in range(y0, y1 + 1)]
    if len(rows) < 3 or len(rows[0]) != 1:
        return None
    tip_x = rows[0][0]
    lo = hi = tip_x
    cavity, split = set(), False
    for depth, row in enumerate(rows[1:], 1):
        if len(row) == 1 and not split and row[0] == tip_x:
            continue
        if len(row) != 2 or row[0] > lo or row[1] < hi or row[1] - row[0] < 2:
            return None
        lo, hi = row
        split = True
        cavity.update(_world((y0 + depth, x), direction) for x in range(lo + 1, hi))
    if not split or not lo < tip_x < hi:
        return None
    return {"frame": set(cells), "cavity": cavity,
            "tip": _world((y0, tip_x), direction), "direction": direction}


def _parse(input_grid):
    grid = _grid(input_grid)
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    bg = int(border[0])
    _require(bool(np.all(border == bg)), "a blank uniform border is required")
    hist = np.bincount(grid.ravel(), minlength=10)
    keys = []
    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1] - 1):
            block = grid[r:r + 2, c:c + 2]
            colors = set(map(int, block.flat))
            if len(colors) == 4 and bg not in colors and all(hist[v] == 1 for v in colors):
                keys.append((r, c, block.copy()))
    _require(len(keys) == 1, "one four-color singleton key is required")
    kr, kc, key = keys[0]
    key_cells = {(kr + r, kc + c) for r in range(2) for c in range(2)}
    colors = set(map(int, np.unique(grid))) - {bg} - set(map(int, key.flat))
    _require(len(colors) == 3, "need carrier, reservoir outline and payload colors")
    hypotheses = []
    for ink, token, carrier in itertools.permutations(sorted(colors)):
        stamps, assigned, valid = [], set(), True
        for component in _components(map(tuple, np.argwhere(grid == ink)), diagonal=True):
            candidates = []
            for direction in range(4):
                stamp = _read_outline(component, direction)
                if stamp is None or any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1])
                                        for r, c in stamp["cavity"]):
                    continue
                values = {int(grid[p]) for p in stamp["cavity"]}
                payload = {p for p in stamp["cavity"] if grid[p] == token}
                if values <= {bg, token} and payload and len(payload) < len(stamp["cavity"]):
                    stamp.update(payload=payload, quantity=len(payload))
                    candidates.append(stamp)
            if len(candidates) != 1:
                valid = False
                break
            stamp = candidates[0]
            if not assigned.isdisjoint(stamp["payload"]):
                valid = False
                break
            assigned.update(stamp["payload"])
            stamps.append(stamp)
        if not valid or not stamps or assigned != set(map(tuple, np.argwhere(grid == token))):
            continue
        restored = grid.copy()
        for stamp in stamps:
            for p in stamp["frame"] | stamp["cavity"]:
                restored[p] = carrier
        comps = _components(map(tuple, np.argwhere(restored != bg)))
        if len(comps) != 4:
            continue
        anchors = [i for i, comp in enumerate(comps) if key_cells <= comp]
        if len(anchors) != 1:
            continue
        comps = [comps[anchors[0]]] + [v for i, v in enumerate(comps) if i != anchors[0]]
        if any({int(restored[p]) for p in comp - key_cells} != {carrier} for comp in comps):
            continue
        okay = True
        grown = [set(comp) for comp in comps]
        for stamp in stamps:
            owners = [i for i, comp in enumerate(comps) if stamp["frame"] | stamp["cavity"] <= comp]
            if len(owners) != 1:
                okay = False
                break
            i = owners[0]
            dr, dc = _DIRECTIONS[stamp["direction"]]
            tr, tc = stamp["tip"]
            ray = {(tr + dr * j, tc + dc * j) for j in range(1, stamp["quantity"] + 1)}
            if not grown[i].isdisjoint(ray):
                okay = False
                break
            stamp.update(piece=i, ray=ray)
            grown[i].update(ray)
        if not okay:
            continue
        shapes = [_normal(comp) for comp in grown]
        area = sum(map(len, shapes))
        n = math.isqrt(area)
        if n * n != area or not 1 <= n <= 30:
            continue
        hypotheses.append({"background": bg, "ink": ink, "token": token, "carrier": carrier,
                           "key": key, "key_cells": key_cells, "base": comps, "grown": grown,
                           "shapes": shapes, "side": n, "stamps": stamps})
    _require(len(hypotheses) == 1, "reservoir roles or ownership are ambiguous")
    return hypotheses[0]


def _render_solution(analysis, solution):
    n = analysis["side"]
    out = np.empty((n, n), dtype=np.int8)
    for corner, bits in enumerate(solution["masks"]):
        for index in range(n * n):
            if bits & (1 << index):
                out[index // n, index % n] = int(analysis["key"].flat[corner])
    return out


def _analyze(input_grid):
    a = _parse(input_grid)
    answers = _pack(a["shapes"], a["side"])
    _require(len(answers) == 1, "completed tiles need one unique corner-colored packing")
    a["solution"] = answers[0]
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _analyze(input_grid)
    return _render_solution(a, a["solution"])


def _profile(radius, tail, stem=0):
    frame, cavity = set(), set()
    for y in range(stem + radius + tail + 1):
        reach = min(radius, max(0, y - stem))
        if reach == 0:
            frame.add((y, radius))
        else:
            frame.update(((y, radius - reach), (y, radius + reach)))
            cavity.update((y, x) for x in range(radius - reach + 1, radius + reach))
    return frame, cavity


def _connected_partition(owner, stamps):
    n = len(owner)
    if tuple(int(owner[p]) for p in ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1))) != (0, 1, 2, 3):
        return False
    for label in range(4):
        cells = set(map(tuple, np.argwhere(owner == label)))
        removed = set().union(*(s["ray"] for s in stamps if s["owner"] == label))
        if len(cells) < 8 or len(_components(cells)) != 1 or len(_components(cells - removed)) != 1:
            return False
    for s in stamps:
        if any(owner[p] != s["owner"] for p in s["frame"] | s["cavity"] | s["ray"]):
            return False
        dr, dc = _DIRECTIONS[s["direction"]]
        projection = lambda p: p[0] * dr + p[1] * dc
        base = set(map(tuple, np.argwhere(owner == s["owner"]))) - s["ray"]
        if max(map(projection, base)) != projection(s["tip"]):
            return False
    return True


def _key_locations(owner, stamps):
    excluded = set().union(*(s["frame"] | s["cavity"] | s["ray"] for s in stamps))
    return [(r, c) for r in range(len(owner) - 1) for c in range(len(owner) - 1)
            if np.all(owner[r:r + 2, c:c + 2] == 0)
            and excluded.isdisjoint({(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)})]


def _make_plan(rng, n, max_quantities, witness=False):
    cut_row = int(rng.integers(5, n - 4))
    owner = np.empty((n, n), dtype=np.int8)
    stamps = []
    bands = list(rng.permutation(2))[:len(max_quantities)]
    for band in range(2):
        start, stop = (0, cut_row) if band == 0 else (cut_row, n)
        left, right = (0, 1) if band == 0 else (2, 3)
        cut = int(rng.integers(3, n - 2))
        for r in range(start, stop):
            cut = int(np.clip(cut + rng.choice([-1, 0, 0, 1]), 2, n - 2))
            owner[r, :cut], owner[r, cut:] = left, right
        if band not in bands:
            continue
        q = max_quantities[bands.index(band)]
        options = []
        for radius in (1, 2):
            for tail in range(2, 6):
                for stem in (0, 1):
                    h = radius + tail + stem + 1
                    capacity = radius * radius + tail * (2 * radius - 1)
                    if (2 * radius + 3 <= stop - start and q < capacity and h + q + 1 <= n
                            and (not witness or (radius, tail, stem) == (1, 3, 0))):
                        options.append((radius, tail, stem))
        if not options:
            return None
        radius, tail, stem = options[int(rng.integers(len(options)))]
        h = radius + tail + stem + 1
        tip_r = int(rng.integers(start + radius + 1, stop - radius - 1))
        tip_c = int(rng.integers(h - 1, n - q - 1))
        reverse = bool(rng.integers(2))
        label, donor = (right, left) if reverse else (left, right)
        # The nose is a convex extension of an otherwise wandering seam.
        for r in range(start, stop):
            old_cut = int(np.count_nonzero(owner[r] == left))
            limited = min(old_cut, tip_c - radius)
            owner[r, :limited], owner[r, limited:] = left, right
        for lateral in range(-radius, radius + 1):
            reach_depth = 0 if lateral == 0 else abs(lateral) + stem
            boundary = tip_c - reach_depth + 1
            r = tip_r + lateral
            owner[r, :boundary], owner[r, boundary:] = left, right
        owner[tip_r, tip_c + 1:tip_c + q + 1] = left
        f, hollow = _profile(radius, tail, stem)
        transform = lambda p: (tip_r + p[1] - radius, tip_c - p[0])
        frame, cavity = set(map(transform, f)), set(map(transform, hollow))
        tip = (tip_r, tip_c)
        ray = {(tip_r, tip_c + j) for j in range(1, q + 1)}
        if reverse:
            # Mirror this band's entire cut, exchanging its left/right labels.
            rowcopy = owner[start:stop, ::-1].copy()
            owner[start:stop] = np.where(rowcopy == left, right, left)
            flip = lambda p: (p[0], n - 1 - p[1])
            frame, cavity, ray, tip = set(map(flip, frame)), set(map(flip, cavity)), set(map(flip, ray)), flip(tip)
        stamps.append({"frame": frame, "cavity": cavity, "tip": tip, "direction": 3 if reverse else 1,
                       "ray": ray, "owner": label, "donor": donor, "quantity": q,
                       "slot": bands.index(band)})
    if not _connected_partition(owner, stamps):
        return None
    protected = {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}
    protected.update(set().union(*(s["frame"] | s["cavity"] | s["ray"] for s in stamps)))
    for _ in range(3 * n):
        r, c = int(rng.integers(n)), int(rng.integers(n))
        if (r, c) in protected:
            continue
        dr, dc = _STEPS[int(rng.integers(4))]
        if not (0 <= r + dr < n and 0 <= c + dc < n):
            continue
        before, after = int(owner[r, c]), int(owner[r + dr, c + dc])
        if before == after:
            continue
        owner[r, c] = after
        if not _connected_partition(owner, stamps):
            owner[r, c] = before
    # Rotate the complete design, then relabel by actual corner ownership.
    turns = int(rng.integers(4))
    def spin(p):
        r, c = p
        for _ in range(turns):
            r, c = n - 1 - c, r
        return r, c
    owner = np.rot90(owner, turns).copy()
    mapping = {int(owner[p]): i for i, p in enumerate(((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)))}
    owner = np.array([[mapping[int(v)] for v in row] for row in owner], dtype=np.int8)
    for s in stamps:
        for field in ("frame", "cavity", "ray"):
            s[field] = set(map(spin, s[field]))
        s["tip"] = spin(s["tip"])
        s["direction"] = (s["direction"] - turns) % 4
        s["owner"], s["donor"] = mapping[s["owner"]], mapping[s["donor"]]
    return owner, sorted(stamps, key=lambda s: s["slot"])


def _materialize(plan, quantities):
    owner, source = plan
    owner = owner.copy()
    stamps = []
    for s, q in zip(source, quantities):
        s = {k: set(v) if isinstance(v, set) else v for k, v in s.items()}
        dr, dc = _DIRECTIONS[s["direction"]]
        r, c = s["tip"]
        ray = {(r + dr * j, c + dc * j) for j in range(1, q + 1)}
        for p in s["ray"] - ray:
            owner[p] = s["donor"]
        s.update(ray=ray, quantity=q)
        stamps.append(s)
    if not _connected_partition(owner, stamps) or not _key_locations(owner, stamps):
        return None
    shapes = [_normal(map(tuple, np.argwhere(owner == i))) for i in range(4)]
    answers = _pack(shapes, len(owner))
    expected = tuple(_mask(set(map(tuple, np.argwhere(owner == i))), len(owner)) for i in range(4))
    if len(answers) != 1 or answers[0]["masks"] != expected:
        return None
    return owner, stamps


def _payload(rng, cavity, q, sparse):
    cells = sorted(cavity)
    chosen = {cells[int(rng.integers(len(cells)))]}
    while len(chosen) < q:
        remaining = sorted(cavity - chosen)
        if sparse:
            score = [min(max(abs(r - y), abs(c - x)) for y, x in chosen) for r, c in remaining]
            best = max(score)
            pool = [p for p, distance in zip(remaining, score) if distance == best]
        else:
            pool = [p for p in remaining if any(abs(p[0] - y) + abs(p[1] - x) == 1 for y, x in chosen)]
        chosen.add(pool[int(rng.integers(len(pool)))])
    return chosen


def _scatter(rng, dims):
    layouts = []
    for transpose in (False, True):
        ds = [(w, h) if transpose else (h, w) for h, w in dims]
        for _ in range(24):
            order = list(map(int, rng.permutation(4)))
            groups, group = [], []
            for j, i in enumerate(order):
                group.append(i)
                if j == 3 or rng.random() < .45:
                    groups.append(group)
                    group = []
            gap = int(rng.integers(1, 4))
            heights = [max(ds[i][0] for i in g) for g in groups]
            widths = [sum(ds[i][1] for i in g) + gap * (len(g) - 1) for g in groups]
            h, w = sum(heights) + gap * (len(groups) - 1), max(widths)
            if h > 28 or w > 28:
                continue
            offsets, y = [None] * 4, 0
            for g, height, width in zip(groups, heights, widths):
                x = int(rng.integers(w - width + 1))
                for i in g:
                    r = y + int(rng.integers(height - ds[i][0] + 1))
                    offsets[i] = (x, r) if transpose else (r, x)
                    x += ds[i][1] + gap
                y += height + gap
            layouts.append((offsets, (w, h) if transpose else (h, w)))
    if not layouts:
        # Each footprint is at most 13x13: two shelves plus a border fit.
        order = list(map(int, rng.permutation(4)))
        groups = (order[:2], order[2:])
        offsets, y = [None] * 4, 0
        width = 0
        for g in groups:
            x = 0
            for i in g:
                offsets[i] = (y, x)
                x += dims[i][1] + 1
            width = max(width, x - 1)
            y += max(dims[i][0] for i in g) + 1
        layouts.append((offsets, (y - 1, width)))
    offsets, extent = layouts[int(rng.integers(len(layouts)))]
    pads = []
    for size in extent:
        before = int(rng.integers(1, min(4, 30 - size)))
        after = int(rng.integers(1, min(5, 31 - size - before)))
        pads.append((before, after))
    return ([(r + pads[0][0], c + pads[1][0]) for r, c in offsets],
            tuple(extent[j] + sum(pads[j]) for j in range(2)))


def _paint(rng, material, palette, sparse, rotations, force_edge=False):
    owner, stamps = material
    bg, carrier, ink, token = palette[:4]
    key = np.array(palette[4:8], dtype=np.int8).reshape(2, 2)
    base = [set(map(tuple, np.argwhere(owner == i))) for i in range(4)]
    for s in stamps:
        base[s["owner"]] -= s["ray"]
    local = []
    for i, cells in enumerate(base):
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        h, w = max(r for r, c in cells) - r0 + 1, max(c for r, c in cells) - c0 + 1
        tile = np.full((h, w), bg, dtype=np.int8)
        for r, c in cells:
            tile[r - r0, c - c0] = carrier
        for s in stamps:
            if s["owner"] != i:
                continue
            for r, c in s["frame"]:
                tile[r - r0, c - c0] = ink
            for r, c in s["cavity"]:
                tile[r - r0, c - c0] = bg
            for r, c in _payload(rng, s["cavity"], s["quantity"], sparse):
                tile[r - r0, c - c0] = token
        if i == 0:
            locations = _key_locations(owner, stamps)
            r, c = locations[int(rng.integers(len(locations)))]
            tile[r - r0:r - r0 + 2, c - c0:c - c0 + 2] = key
        local.append(np.rot90(tile, -rotations[i]).copy())
    offsets, canvas = _scatter(rng, [a.shape for a in local])
    inp = np.full(canvas, bg, dtype=np.int8)
    for tile, (r, c) in zip(local, offsets):
        inp[r:r + tile.shape[0], c:c + tile.shape[1]] = tile
    if force_edge:
        # A visible witness that source-canvas clipping is the wrong convention.
        bounds = (min(r for r, c in offsets),
                  max(c + tile.shape[1] - 1 for tile, (r, c) in zip(local, offsets)),
                  max(r + tile.shape[0] - 1 for tile, (r, c) in zip(local, offsets)),
                  min(c for r, c in offsets))
        possible = []
        for s in stamps:
            i = s["owner"]
            r, c = offsets[i]
            direction = (s["direction"] + rotations[i]) % 4
            edge = (r, c + local[i].shape[1] - 1, r + local[i].shape[0] - 1, c)[direction]
            if edge == bounds[direction]:
                possible.append(direction)
        if not possible:
            return None
        direction = possible[int(rng.integers(len(possible)))]
        r0, c0, r1, c1 = 0, 0, inp.shape[0], inp.shape[1]
        if direction == 0:
            r0 = bounds[0] - 1
        elif direction == 1:
            c1 = bounds[1] + 2
        elif direction == 2:
            r1 = bounds[2] + 2
        else:
            c0 = bounds[3] - 1
        inp = inp[r0:r1, c0:c1].copy()
    return {"input": inp, "output": key.ravel()[owner].astype(np.int8)}


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return tuple(a.shape), a.tobytes()


def _counts(a, b):
    for value, lo, hi in ((a, 2, 10), (b, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError("illegal example count")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # The only shared witness data are visible, matched reservoir properties.
    matched = None
    for _ in range(160):
        plan = _make_plan(rng, int(rng.integers(11, 14)), (3, 3), witness=True)
        if plan is None:
            continue
        first = _materialize(plan, (2, 3))
        second = _materialize(plan, (3, 2))
        if first is not None and second is not None:
            witness_shapes = [_normal(map(tuple, np.argwhere(first[0] == i))) for i in range(4)]
            orientation_sets = [{_rotate(shape, k) for k in range(4)} for shape in witness_shapes]
            if any(len(views) != 4 for views in orientation_sets):
                continue
            if any(orientation_sets[i] == orientation_sets[j] for i in range(4) for j in range(i)):
                continue
            matched = (first, second)
            break
    _require(matched is not None, "could not make matched count witnesses")
    witness_palette = list(map(int, rng.choice(10, 8, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for attempt in range(160):
                if split == "train" and index < 2:
                    material = matched[index]
                    palette = witness_palette.copy()
                    if index == 1:
                        palette[4:] = palette[5:] + palette[4:5]
                else:
                    number = int(rng.integers(1, 3))
                    quantities = [int(rng.integers(1, 4)) for _ in range(number)]
                    if split == "test":
                        quantities[0] = int(rng.integers(4, 6))
                        quantities[1:] = [int(rng.integers(1, 6)) for _ in quantities[1:]]
                    n = int(rng.integers(13, 15) if split == "test" else rng.integers(10, 15))
                    plan = _make_plan(rng, n, quantities)
                    material = _materialize(plan, quantities) if plan is not None else None
                    if material is None:
                        continue
                    palette = list(map(int, rng.choice(10, 8, replace=False)))
                rotations = [0] + ([1, 2, 3] if split == "train" and index == 0
                                    else list(map(int, rng.integers(0, 4, size=3))))
                pair = _paint(rng, material, palette, sparse=(index % 2 == 1), rotations=rotations,
                              force_edge=(split == "train" and index == 0))
                if pair is None:
                    continue
                signature = _signature(pair["input"])
                if signature in seen:
                    continue
                try:
                    a = _analyze(pair["input"])
                except ValueError:
                    continue
                if not np.array_equal(_render_solution(a, a["solution"]), pair["output"]):
                    continue
                # Source reading order must not encode destination corners.
                reading = sorted(range(4), key=lambda i: min(a["base"][i]))
                if tuple(a["solution"]["pieces"]) == tuple(reading):
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_counts, test_counts, evidence = set(), set(), [], {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _signature(inp)
            assert sig not in seen, "duplicate input"
            seen.add(sig)
            a = _analyze(inp)
            assert np.array_equal(out, _render_solution(a, a["solution"])), "incorrect assembled output"
            assert set(map(int, out.flat)) == set(map(int, a["key"].flat)), "wrong corner palette"
            assert 1 <= len(a["stamps"]) <= 2
            assert 10 <= a["side"] <= 14
            # Without A, even the inferred reservoir-filled base areas cannot tile a square.
            base_area = sum(map(len, a["base"]))
            assert math.isqrt(base_area) ** 2 != base_area, "count decoding is redundant in area"
            assert sum(s["quantity"] for s in a["stamps"]) == a["side"] ** 2 - base_area
            # Without B's spatial key, source reading order gives a different color map.
            reading = sorted(range(4), key=lambda i: min(a["base"][i]))
            assert tuple(reading) != a["solution"]["pieces"]
            for stamp in a["stamps"]:
                comp = a["base"][stamp["piece"]]
                dr, dc = _DIRECTIONS[stamp["direction"]]
                proj = lambda p: p[0] * dr + p[1] * dc
                assert max(map(proj, comp)) == proj(stamp["tip"]), "tooth must extend the silhouette"
                assert len(stamp["ray"]) == stamp["quantity"] and comp.isdisjoint(stamp["ray"])
                if split == "train":
                    train_counts.add(stamp["quantity"])
                    # Normalize orientation as well as translation for matched evidence.
                    profile = _normal({_canonical(p, stamp["direction"]) for p in stamp["frame"]})
                    sig = (profile, a["ink"], a["token"])
                    evidence.setdefault(sig, []).append((stamp["quantity"], len(_components(stamp["payload"], True))))
            if split == "test":
                test_counts.append(max(s["quantity"] for s in a["stamps"]))
    assert {2, 3} <= train_counts, "two visible counts required"
    assert any({2, 3} <= {q for q, components in rows} for rows in evidence.values()), "matched count evidence absent"
    assert any(len({components for q, components in rows if q == 2}) >= 2 for rows in evidence.values()), "no arrangement counterexample"
    assert any(components < q for rows in evidence.values() for q, components in rows), "component-count shortcut unchallenged"
    assert all(q > max(train_counts) for q in test_counts), "missing longer-tooth test shift"
    first = _analyze(puzzle["train"][0]["input"])
    assert set(first["solution"]["rotations"][1:]) == {1, 2, 3}, "rotation witnesses missing"
    h, w = np.asarray(puzzle["train"][0]["input"]).shape
    assert any(not (0 <= r < h and 0 <= c < w) for s in first["stamps"] for r, c in s["ray"]), "no local-coordinate boundary witness"


def self_test():
    """42 seeds, streamed; all count combinations; metamorphic and negative checks."""
    signatures, cuts, shapes, sides, directions, quantities, bgs, stamp_counts = (set() for _ in range(8))
    rotations, profiles = set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                inp, out = pair["input"], pair["output"]
                digest.update(inp.tobytes()); digest.update(out.tobytes())
                a = _analyze(inp)
                assert np.array_equal(solve(inp), out)
                shapes.add(inp.shape); sides.add(a["side"]); bgs.add(a["background"])
                stamp_counts.add(len(a["stamps"]))
                normalized = np.empty(out.shape, dtype=np.int8)
                for i, color in enumerate(a["key"].flat):
                    normalized[out == color] = i
                cuts.add(_signature(normalized))
                rotations.update(a["solution"]["rotations"][1:])
                for s in a["stamps"]:
                    directions.add(s["direction"]); quantities.add(s["quantity"])
                    profiles.add(_normal({_canonical(p, s["direction"]) for p in s["frame"]}))
        signatures.add(digest.hexdigest())
        if seed == 0:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for split in puzzle for a, b in zip(puzzle[split], repeat[split]) for k in a)
            # Reversible palette relabeling and blank padding cannot change the rule.
            pair = puzzle["test"][0]
            perm = np.array([3, 8, 0, 9, 1, 7, 4, 2, 6, 5])
            assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
            a = _parse(pair["input"])
            padded = np.pad(pair["input"], ((0, min(1, 30 - pair["input"].shape[0])), (0, min(1, 30 - pair["input"].shape[1]))), constant_values=a["background"])
            assert np.array_equal(solve(padded), pair["output"])
            broken = {split: [{k: v.copy() for k, v in p.items()} for p in puzzle[split]] for split in puzzle}
            broken["test"][0]["output"][0, 0] = a["key"].flat[1]
            try:
                validate_generated_puzzle(broken)
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError("bad output accepted")
    assert len(signatures) == 42 and len(cuts) >= 100 and len(shapes) >= 30
    assert {10, 11, 12, 13, 14} <= sides and directions == rotations == {0, 1, 2, 3}
    assert {1, 2, 3, 4, 5} <= quantities and len(bgs) == 10 and stamp_counts == {1, 2}
    assert len(profiles) >= 8
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(2000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures), "cuts": len(cuts),
            "input_shapes": len(shapes), "square_sides": sorted(sides), "directions": sorted(directions),
            "quantities": sorted(quantities), "backgrounds": len(bgs), "reservoir_profiles": len(profiles)}
