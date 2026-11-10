"""Rigid assembly constrained by the topology of an emergent material region.

Only the standard library and NumPy are used. The input parser and exact-cover
solver know no construction state, example labels, or seeds.
"""
from __future__ import annotations

import json
from itertools import product

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
AROUND = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
               if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "Grids must be two dimensional with axes 1..30.")
    _require(np.issubdtype(a.dtype, np.integer), "Integer colors required.")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Colors must be 0..9.")
    return a.astype(np.int8, copy=True)


def _components(mask, neighbors=EDGE):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _trim(mask):
    p = np.argwhere(mask)
    _require(len(p) > 0, "Empty object.")
    low, high = p.min(axis=0), p.max(axis=0) + 1
    return mask[low[0]:high[0], low[1]:high[1]].copy()


def _connected(mask):
    return len(_components(mask)) == 1


def _holes(mask):
    """Count complementary four-components with a true exterior padding."""
    return len(_components(~np.pad(np.asarray(mask, dtype=bool), 1))) - 1


def _rotations(a):
    result, seen = [], set()
    for k in range(4):
        b = np.rot90(a, k).copy()
        key = b.shape, b.tobytes()
        if key not in seen:
            seen.add(key)
            result.append(b)
    return result


def _shape_key(a):
    return min((b.shape, b.tobytes()) for b in _rotations(_trim(a)))


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _piece(cells, grid):
    p = np.asarray(cells, dtype=int)
    lo, hi = p.min(axis=0), p.max(axis=0) + 1
    a = np.full(tuple(hi - lo), -1, dtype=np.int8)
    for r, c in cells:
        a[r - lo[0], c - lo[1]] = grid[r, c]
    return a


def _parse(input_grid):
    a = _grid(input_grid)
    corners = (int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1]))
    _require(len(set(corners)) == 1, "Four corners identify the background.")
    bg = corners[0]
    colors, counts = np.unique(a, return_counts=True)
    singleton = [int(c) for c, n in zip(colors, counts) if n == 1 and c != bg]
    _require(len(singleton) == 1, "Exactly one singleton marker is required.")
    marker = singleton[0]
    lines = []
    for r in range(a.shape[0]):
        if a[r, 0] != bg and np.all(a[r] == a[r, 0]):
            lines.append((0, r, int(a[r, 0])))
    for c in range(a.shape[1]):
        if a[0, c] != bg and np.all(a[:, c] == a[0, c]):
            lines.append((1, c, int(a[0, c])))
    _require(len(lines) == 1, "One complete separator identifies the panel.")
    axis, p, sep = lines[0]
    _require(0 < p < a.shape[axis] - 1, "Separator must be internal.")
    if axis == 0:
        panel_side = 'bottom' if np.any(a[:p] == marker) else 'top'
        b = a
    else:
        panel_side = 'right' if np.any(a[:, :p] == marker) else 'left'
        # A transpose would reflect chiral fragments. Rotate the full input.
        b = np.rot90(a)
        p = a.shape[1] - 1 - p
    left, right = b[:p], b[p + 1:]
    if np.any(left == marker):
        work, panel = left, right
    else:
        work, panel = right, left
    _require(np.sum(work == marker) == 1 and not np.any(panel == marker),
             "The marker must identify the work side.")
    panel_colors = set(map(int, np.unique(panel))) - {bg}
    _require(len(panel_colors) == 1, "One colored prototype is required.")
    target_color = next(iter(panel_colors))
    glyph = panel == target_color
    _require(_connected(glyph), "Prototype must be edge connected.")
    target = _holes(glyph)
    _require(0 <= target <= 4, "Prototype class must be 0..4.")
    pieces = [_piece(cells, work) for cells in _components(work != bg)]
    _require(3 <= len(pieces) <= 8, "Three to eight fragments are required.")
    _require(len(_components(work != bg, AROUND)) == len(pieces),
             "Different fragments must be separated even diagonally.")
    marked = [p for p in pieces if np.any(p == marker)]
    _require(len(marked) == 1, "Marker must attach to one fragment.")
    carrier = set(map(int, marked[0][marked[0] >= 0])) - {marker}
    _require(len(carrier) == 1, "Marked fragment has one material.")
    query = next(iter(carrier))
    work_colors = set(map(int, np.unique(work))) - {bg, marker, query}
    _require(len(work_colors) == 1, "One complementary material is required.")
    other = next(iter(work_colors))
    _require(len({bg, marker, query, other, sep, target_color}) == 6,
             "All six visible roles must have different colors.")
    qparts = sparts = 0
    for piece in pieces:
        values = set(map(int, piece[piece >= 0])) - {marker}
        _require(values in ({query}, {other}), "Each fragment has one material.")
        _require(np.sum(piece >= 0) >= 3, "Each fragment has at least three cells.")
        qparts += query in values
        sparts += other in values
    _require(qparts >= 2 and sparts >= 1, "The query must be fragmented.")
    area = sum(int(np.sum(piece >= 0)) for piece in pieces)
    side = int(round(area ** .5))
    _require(side * side == area and 5 <= side <= 16, "Area must define a square of side 5..16.")
    return dict(pieces=pieces, side=side, marker=marker, query=query, other=other,
                background=bg, separator=sep, target_color=target_color,
                target=target, panel_side=panel_side, qparts=qparts,
                sparts=sparts, work=work.copy())


def _assemblies(facts, target=None, limit=2):
    """Enumerate distinct colored anchored covers, optionally topology filtered.

    Identical fragment permutations and rotations are deduplicated by the final
    complementary-material mask. No generation state is consulted. No search
    budget is imposed on solving: limit means number of distinct answers only.
    """
    pieces, side = facts['pieces'], facts['side']
    marker, other = facts['marker'], facts['other']
    full = (1 << (side * side)) - 1
    candidates = []
    anchor = next(i for i, p in enumerate(pieces) if np.any(p == marker))
    for i, piece in enumerate(pieces):
        seen, options = set(), []
        for b in _rotations(piece):
            h, w = b.shape
            if h > side or w > side:
                continue
            if i == anchor and b[0, 0] != marker:
                continue
            base = sum(1 << (int(r) * side + int(c)) for r, c in np.argwhere(b >= 0))
            sbits = sum(1 << (int(r) * side + int(c)) for r, c in np.argwhere(b == other))
            positions = [(0, 0)] if i == anchor else product(range(side - h + 1), range(side - w + 1))
            for r, c in positions:
                offset = r * side + c
                item = base << offset, sbits << offset
                if item not in seen:
                    seen.add(item)
                    options.append(item)
        candidates.append(options)
    answers, seen_answers = [], set()
    # Repeated shapes/materials are interchangeable, so avoid identical siblings.
    identities = [_shape_key(p >= 0) + (int(np.any(p == other)), int(np.any(p == marker)))
                  for p in pieces]
    cache = set()

    def visit(occupied, sbits, remaining):
        if len(answers) >= limit:
            return
        state = (occupied, sbits, tuple(sorted(identities[i] for i in remaining)))
        if state in cache:
            return
        cache.add(state)
        if not remaining:
            if occupied != full or sbits in seen_answers:
                return
            seen_answers.add(sbits)
            smask = np.fromiter(((sbits >> p) & 1 for p in range(side * side)),
                                dtype=bool).reshape(side, side)
            qmask = ~smask
            if not _connected(qmask):
                return
            holes = _holes(qmask)
            if target is not None and holes != target:
                return
            out = np.where(smask, other, facts['query']).astype(np.int8)
            out[0, 0] = marker
            answers.append(out)
            return
        # Place the most constrained remaining whole fragment first. In these
        # families this usually fixes the large scaffold before considering
        # interchangeable small plugs. Every compatible placement is searched.
        used_identities, choices = set(), []
        for i in remaining:
            if identities[i] in used_identities:
                continue
            used_identities.add(identities[i])
            fitting = [(bits, ss) for bits, ss in candidates[i] if not bits & occupied]
            if not fitting:
                return
            choices.append((len(fitting), -int(np.sum(pieces[i] >= 0)), i, fitting))
        _, _, i, fitting = min(choices, key=lambda item: item[:3])
        rest = tuple(j for j in remaining if j != i)
        for bits, ss in fitting:
            visit(occupied | bits, sbits | ss, rest)
            if len(answers) >= limit:
                return

    for bits, ss in candidates[anchor]:
        visit(bits, ss, tuple(i for i in range(len(pieces)) if i != anchor))
        if len(answers) >= limit:
            break
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _parse(input_grid)
    answers = _assemblies(facts, target=facts['target'], limit=2)
    _require(len(answers) == 1, "Topology must select exactly one colored assembly.")
    answer = answers[0].copy()
    answer[answer == facts['query']] = facts['target_color']
    return answer


def _grow_shape(rng, area):
    """Small connected, hole-free silhouettes, with unequal boxes and arms."""
    for _ in range(50):
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        if h * w < area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
        while mask.sum() < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in EDGE:
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        frontier.add((rr, cc))
            options = sorted(frontier)
            mask[options[int(rng.integers(len(options)))]] = True
        if _holes(mask) == 0:
            return _trim(mask)
    # Connected staircase with at most four cells per row, no cavities.
    h = (area + 3) // 4
    mask = np.ones((h, 4), dtype=bool)
    mask[-1, area - 4 * (h - 1):] = False
    return _trim(mask)


def _socket_positions(shape, side, edge):
    h, w = shape.shape
    positions = []
    for r in range(side - h + 1):
        for c in range(side - w + 1):
            cells = np.argwhere(shape) + (r, c)
            on_edge = bool(np.any((cells == 0) | (cells == side - 1)))
            if on_edge == edge and not np.any(np.all(cells == (0, 0), axis=1)):
                positions.append((r, c))
    return positions


def _dilate(mask):
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr:dr + mask.shape[0], dc:dc + mask.shape[1]]
    return out


def _scaffold(rng, side, active, extra, shards):
    areas = list(map(int, rng.choice(np.arange(3, 10), active + extra, replace=False)))
    shapes = [_grow_shape(rng, area) for area in areas]
    slots, blocked = [], np.zeros((side, side), dtype=bool)
    # Each active silhouette appears at an interior and a perimeter socket.
    requests = [(i, edge) for i in range(active) for edge in (False, True)]
    requests += [(i, False) for i in range(active, active + extra)]
    for i, edge in requests:
        shape = np.rot90(shapes[i], int(rng.integers(4))).copy()
        options = _socket_positions(shape, side, edge)
        rng.shuffle(options)
        found = False
        for r, c in options:
            mask = np.zeros((side, side), dtype=bool)
            mask[r:r + shape.shape[0], c:c + shape.shape[1]] = shape
            if np.any(mask & blocked):
                continue
            slots.append((i, edge, mask))
            blocked |= _dilate(mask)
            found = True
            break
        if not found:
            return None
    holes = np.logical_or.reduce([mask for _, _, mask in slots])
    frame = ~holes
    if not _connected(frame) or _holes(frame) != active + extra or frame.sum() < side * side * .5:
        return None
    fragments = [frame]
    if shards >= 2:
        interior = np.logical_or.reduce([mask for _, edge, mask in slots if not edge])
        cuts = [(axis, p) for axis in (0, 1) for p in range(2, side - 1)]
        rng.shuffle(cuts)
        for axis, p in cuts:
            # Split through a cavity, so its rim has to close across a seam.
            before = interior[:p] if axis == 0 else interior[:, :p]
            after = interior[p:] if axis == 0 else interior[:, p:]
            if not before.any() or not after.any():
                continue
            across = (interior[p - 1] & interior[p]).any() if axis == 0 else (interior[:, p - 1] & interior[:, p]).any()
            if not across:
                continue
            a = frame.copy()
            if axis == 0:
                a[p:] = False
            else:
                a[:, p:] = False
            b = frame & ~a
            if min(a.sum(), b.sum()) >= 6 and _connected(a) and _connected(b):
                fragments = [a, b]
                break
        if len(fragments) != 2:
            return None
    if shards == 3:
        possibilities = []
        for i, mask in enumerate(fragments):
            for axis in (0, 1):
                for p in range(2, side - 1):
                    a = mask.copy()
                    if axis == 0:
                        a[p:] = False
                    else:
                        a[:, p:] = False
                    b = mask & ~a
                    if min(a.sum(), b.sum()) >= 6 and _connected(a) and _connected(b):
                        possibilities.append((i, a, b))
        if not possibilities:
            return None
        i, a, b = possibilities[int(rng.integers(len(possibilities)))]
        fragments = fragments[:i] + [a, b] + fragments[i + 1:]
    return fragments, slots


def _build_pieces(fragments, slots, query, other, marker, high):
    """One query plug per active pair; all extra plugs use other material."""
    side = fragments[0].shape[0]
    out = np.full((side, side), query, dtype=np.int8)
    active = {i for i, edge, _ in slots if edge}
    masks = [(mask, query) for mask in fragments]
    for i, edge, mask in slots:
        # Low class: query fills interior; high class: query fills perimeter.
        color = query if i in active and edge == high else other
        out[mask] = color
        masks.append((mask, color))
    out[0, 0] = marker
    # Both low and high contain the same colored fragment multiset.
    pieces = []
    for mask, color in masks:
        colored = np.full(mask.shape, color, dtype=np.int8)
        if mask[0, 0]:
            colored[0, 0] = marker
        pieces.append(_piece(_components(mask)[0], colored))
    return pieces, out


def _scatter(rng, pieces, bg, anchor_rotation):
    anchor = next(i for i, p in enumerate(pieces) if np.any(p >= 0) and len(np.unique(p[p >= 0])) == 2)
    rotations = list(map(int, rng.integers(4, size=len(pieces))))
    rotations[anchor] = anchor_rotation
    # The small, asymmetric pieces actually move and rotate independently.
    oriented = [np.rot90(p, k).copy() for p, k in zip(pieces, rotations)]
    for attempt in range(16):
        canvas = np.full((23, 28), -1, dtype=np.int8)
        blocked = np.zeros(canvas.shape, dtype=bool)
        order = sorted(range(len(pieces)), key=lambda i: -oriented[i].size)
        if attempt >= 8:
            order = list(map(int, rng.permutation(len(pieces))))
        maxr = maxc = 0
        okay = True
        aspect = float(rng.uniform(.65, 1.45))
        for i in order:
            piece = oriented[i]
            mask = piece >= 0
            h, w = piece.shape
            options = []
            for r in range(24 - h):
                for c in range(29 - w):
                    if np.any(blocked[r:r + h, c:c + w] & mask):
                        continue
                    hr, wc = max(maxr, r + h), max(maxc, c + w)
                    score = hr * wc + abs(hr - aspect * wc) * 3 + float(rng.random()) * 22
                    options.append((score, r, c))
            if not options:
                okay = False
                break
            _, r, c = min(options)
            canvas[r:r + h, c:c + w][mask] = piece[mask]
            occupied = np.zeros(canvas.shape, dtype=bool)
            occupied[r:r + h, c:c + w] = mask
            blocked |= _dilate(occupied)
            maxr, maxc = max(maxr, r + h), max(maxc, c + w)
        if not okay:
            continue
        points = np.argwhere(canvas >= 0)
        lo, hi = points.min(axis=0), points.max(axis=0) + 1
        crop = canvas[lo[0]:hi[0], lo[1]:hi[1]]
        # Mandatory blank margins preserve corner background and parsing.
        top, bottom = 1, 1
        left, right = 1, 1
        if crop.shape[0] < 22:
            top += int(rng.integers(min(3, 22 - crop.shape[0]) + 1))
        if crop.shape[1] < 28:
            left += int(rng.integers(min(3, 28 - crop.shape[1]) + 1))
        result = np.full((crop.shape[0] + top + bottom, crop.shape[1] + left + right), bg, dtype=np.int8)
        result[top:top + crop.shape[0], left:left + crop.shape[1]] = np.where(crop >= 0, crop, bg)
        if result.shape[0] <= 24 and result.shape[1] <= 30:
            return result
    return None


def _prototype(k, style):
    # A single connected glyph, no ordering code. All prototypes have >=6 cells.
    if k == 0:
        return np.ones((3, 3 if style == 0 else 4), dtype=bool)
    rows = 1 if style == 0 or k < 3 else 2
    cols = (k + rows - 1) // rows
    mask = np.ones((2 * rows + 1, 2 * cols + 1), dtype=bool)
    for i in range(k):
        mask[1 + 2 * (i // cols), 1 + 2 * (i % cols)] = False
    if style == 2:
        mask = np.repeat(mask, 2, axis=1)
    return mask


def _attach(work, k, roles, style=0, orientation=0):
    bg, marker, query, other, sep, color = roles
    glyph = _prototype(k, style)
    # Tall prototypes fit the side orientation; choose canonical top dimensions.
    width = max(work.shape[1], glyph.shape[1] + 2)
    height = glyph.shape[0] + 2 + 1 + work.shape[0]
    if height > 30 or width > 30:
        return None
    a = np.full((height, width), bg, dtype=np.int8)
    c = (width - glyph.shape[1]) // 2
    a[1:1 + glyph.shape[0], c:c + glyph.shape[1]][glyph] = color
    separator = glyph.shape[0] + 2
    a[separator] = sep
    c = (width - work.shape[1]) // 2
    a[separator + 1:, c:c + work.shape[1]] = work
    return np.rot90(a, orientation).copy()


def _facts_for_pieces(pieces, side, roles):
    bg, marker, query, other, sep, color = roles
    return dict(pieces=pieces, side=side, marker=marker, query=query, other=other)


def _construction(rng, side, active, extra, shards, roles, anchor_rotation):
    for _ in range(160):
        design = _scaffold(rng, side, active, extra, shards)
        if design is None:
            continue
        fragments, slots = design
        pieces, low = _build_pieces(fragments, slots, roles[2], roles[3], roles[1], False)
        _, high = _build_pieces(fragments, slots, roles[2], roles[3], roles[1], True)
        facts = _facts_for_pieces(pieces, side, roles)
        lows = _assemblies(facts, extra, 2)
        if len(lows) != 1 or not np.array_equal(lows[0], low):
            continue
        highs = _assemblies(facts, extra + active, 2)
        if len(highs) != 1 or not np.array_equal(highs[0], high):
            continue
        work = _scatter(rng, pieces, roles[0], anchor_rotation)
        if work is not None:
            return work, low, high
    return None


def _fallback(rng, roles, anchor_rotation, ordinal=0):
    # Finite safety net in the SAME family: congruent L inserts, interior and
    # perimeter, one forced 2x2 interior patch; split the scaffold across the L.
    side = 10
    a = np.zeros((side, side), dtype=bool)
    a[3, 3:6] = True
    a[4, 3] = True
    b = np.zeros_like(a)
    b[8, 0:3] = True
    b[9, 0] = True
    c = np.zeros_like(a)
    c[6:8, 7:9] = True
    frame = ~(a | b | c)
    f1 = frame.copy()
    f1[4:] = False
    f2 = frame & ~f1
    slots = [(0, False, a), (0, True, b), (1, False, c)]
    pieces, low = _build_pieces([f1, f2], slots, roles[2], roles[3], roles[1], False)
    _, high = _build_pieces([f1, f2], slots, roles[2], roles[3], roles[1], True)
    facts = _facts_for_pieces(pieces, side, roles)
    _require(len(_assemblies(facts, 1, 2)) == 1 and len(_assemblies(facts, 2, 2)) == 1,
             "Fallback must have unique topology-selected answers.")
    # Tight shelves provide a guaranteed <=23x23 fit, rather than relying on
    # successful random packing for the safety net itself. The ordinal changes
    # visible margins and a chiral scaffold's rotation for distinct examples.
    oriented = [np.rot90(pieces[0], anchor_rotation).copy(),
                np.rot90(pieces[1], ordinal % 4).copy()]
    oriented += [np.rot90(p, int(rng.integers(4))).copy() for p in pieces[2:]]
    margin = 1 + ordinal // 4
    top_height = max(p.shape[0] for p in oriented[:2])
    width = 3 + sum(p.shape[1] for p in oriented[:2])
    height = margin + top_height + 1 + max(p.shape[0] for p in oriented[2:]) + 1
    work = np.full((height, width), roles[0], dtype=np.int8)
    col = 1
    for p in oriented[:2]:
        work[margin:margin + p.shape[0], col:col + p.shape[1]][p >= 0] = p[p >= 0]
        col += p.shape[1] + 1
    col = 1
    row = margin + top_height + 1
    for p in oriented[2:]:
        work[row:row + p.shape[0], col:col + p.shape[1]][p >= 0] = p[p >= 0]
        col += p.shape[1] + 1
    return work, low, high, 1, 2


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}.")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    seed = int(seed)
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test.")
    rng = np.random.default_rng(seed)
    base_roles = tuple(map(int, rng.permutation(10)[:6]))
    # Marker remains stable; the other five roles may change visibly per pair.
    varying = bool(rng.integers(2)) and num_train >= 3
    regime = int(rng.integers(3))  # larger tests, more swaps, or combined
    first_side = int(rng.integers(9, 12))
    first_orientation = int(rng.integers(4))
    anchor_rotation = int(rng.integers(4))
    first = _construction(rng, first_side, 1, 1, 2, base_roles, anchor_rotation)
    if first is None:
        work, low, high, k0, k1 = _fallback(rng, base_roles, anchor_rotation)
    else:
        work, low, high = first
        k0, k1 = 1, 2
    puzzle, seen = {'train': [], 'test': []}, set()

    def add(split, work, raw, k, roles, style, orientation):
        inp = _attach(work, k, roles, style, orientation)
        if inp is None or _key(inp) in seen:
            return False
        out = raw.copy()
        out[out == roles[2]] = roles[5]
        # Parsing and solving validate recoverability after every scatter.
        prediction = solve(inp)
        if not np.array_equal(prediction, out):
            return False
        seen.add(_key(inp))
        puzzle[split].append({'input': inp, 'output': out})
        return True

    _require(add('train', work, low, k0, base_roles, 0, first_orientation), "First teaching pair failed.")
    _require(add('train', work, high, k1, base_roles, 0, first_orientation), "Causal teaching pair failed.")
    for split, start, count in [('train', 2, num_train), ('test', 0, num_test)]:
        for index in range(start, count):
            accepted = False
            for attempt in range(24):
                if varying:
                    marker = base_roles[1]
                    available = list(map(int, rng.permutation([c for c in range(10) if c != marker])))
                    roles = (available[0], marker, *available[1:5])
                else:
                    roles = base_roles
                if split == 'test':
                    side = int(rng.integers(12, 15)) if regime != 1 else int(rng.integers(10, 13))
                    active = 1 if regime == 0 else 2
                    extra = int(rng.integers(0, 3))
                    shards = int(rng.integers(1, min(3, 8 - 2 * active - extra) + 1))
                    rotation = (anchor_rotation + 2 + index) % 4
                else:
                    side = int(rng.integers(9, 12 if regime != 1 else 13))
                    active = 2 if index % 3 == 2 else 1
                    extra = 0 if active == 2 else int(rng.integers(0, 3))
                    shards = int(rng.integers(1, 4))
                    rotation = int(rng.integers(4))
                design = _construction(rng, side, active, extra, shards, roles, rotation)
                if design is None:
                    continue
                work, low, high = design
                # The four-cavity endpoint deserves explicit coverage; the
                # lower endpoint is already represented by many simpler cases.
                high_class = bool(rng.integers(2)) or (active == 2 and extra == 2)
                k = extra + (active if high_class else 0)
                raw = high if high_class else low
                style = int(rng.integers(3))
                orientation = int(rng.integers(4))
                if add(split, work, raw, k, roles, style, orientation):
                    accepted = True
                    break
            if not accepted:
                work, low, high, k0, k1 = _fallback(rng, roles, rotation, len(seen))
                # Vary visible margins and orientation to make all inputs distinct.
                for style in (0, 1, 2):
                    for orientation in range(4):
                        if add(split, work, high, k1, roles, style, orientation):
                            accepted = True
                            break
                    if accepted:
                        break
                _require(accepted, "Could not create a distinct valid pair.")
    return puzzle


def _features(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    f = _parse(inp)
    alternatives = _assemblies(f, target=None, limit=8)
    f.update(shape=inp.shape, count=len(f['pieces']),
             classes=tuple(sorted({_holes((a == f['query']) | (a == f['marker'])) for a in alternatives})),
             geometric_answers=len(alternatives),
             input_fragment_holes=tuple(_holes(p >= 0) for p in f['pieces'] if np.any(p == f['query'])),
             rotation_witness=bool(next(p for p in f['pieces'] if np.any(p == f['marker']))[0, 0]
                                   != f['marker']),
             material_area=int(np.sum((out == f['target_color']) | (out == f['marker']))))
    return f


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle schema.")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists.")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal example counts.")
    seen, markers, training = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair schema.")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "Inputs must be distinct.")
            seen.add(_key(inp))
            f = _parse(inp)
            markers.add(f['marker'])
            _require(out.shape == (f['side'], f['side']), "Area must determine output size.")
            _require(int(out[0, 0]) == f['marker'] and np.sum(out == f['marker']) == 1,
                     "Preserve the unique corner marker.")
            _require(set(map(int, np.unique(out))) == {f['marker'], f['other'], f['target_color']},
                     "Output contains only the two materials and marker.")
            qmask = (out == f['target_color']) | (out == f['marker'])
            _require(_connected(qmask) and _holes(qmask) == f['target'], "Wrong emergent topology.")
            qarea = sum(int(np.sum((p == f['query']) | (p == f['marker']))) for p in f['pieces'])
            sarea = sum(int(np.sum(p == f['other'])) for p in f['pieces'])
            _require(qarea == int(qmask.sum()) and sarea == int(np.sum(out == f['other'])),
                     "Preserve both material areas before recoloring.")
            _require(np.array_equal(solve(inp), out), "Wrong rigid topology-constrained assembly.")
            alternatives = _assemblies(f, target=None, limit=2)
            _require(len(alternatives) >= 2, "Geometry alone must leave a choice.")
            if split == 'train':
                training.append((f, inp, out))
    _require(len(markers) == 1, "Marker color is stable within a puzzle.")
    _require({1, 2} <= {f['target'] for f, _, _ in training}, "Teach a count distinction beyond hole presence.")
    # First two are a causal comparison: same scattered work, changed prototype.
    f0, i0, o0 = training[0]
    f1, i1, o1 = training[1]
    signatures = lambda f: sorted(_shape_key(p >= 0) + (int(np.any(p == f['other'])),
        int(np.any(p == f['marker']))) for p in f['pieces'])
    _require(signatures(f0) == signatures(f1) and f0['side'] == f1['side'], "First two must share fragments.")
    _require(np.array_equal(f0['work'], f1['work']), "First two must share actual scattered work pixels.")
    _require(f0['target'] != f1['target'] and not np.array_equal(o0 == f0['other'], o1 == f1['other']),
             "Prototype intervention must change the physical assembly.")
    _require(f1['target'] > max(_holes(p >= 0) for p in f1['pieces'] if np.any(p == f1['query'])),
             "Demonstrations must contain closure across fragment seams.")


def self_test():
    """Bounded 42-seed audit plus topology, count, and corruption regressions."""
    diagonal = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    _require(_holes(diagonal) == 1, "Diagonal escape remains enclosed.")
    diagonal[0, 1] = False
    _require(_holes(diagonal) == 0, "Orthogonal opening reaches exterior.")
    _require(_holes(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)) == 0,
             "A perimeter notch is not a cavity.")
    chiral = np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)
    _require(_shape_key(chiral) != _shape_key(chiral[:, ::-1]), "Rotations must not include reflections.")
    shapes, sides, counts, targets, panels, roles, structures, class_sets = [set() for _ in range(8)]
    closure_witnesses = swaps = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(f['shape'])
                sides.add(f['side'])
                counts.add(f['count'])
                targets.add(f['target'])
                panels.add(f['panel_side'])
                roles.add(tuple(f[k] for k in ('background', 'marker', 'query', 'other', 'separator', 'target_color')))
                norm, values = {}, []
                for v in pair['output'].flat:
                    values.append(norm.setdefault(int(v), len(norm)))
                structures.add((pair['output'].shape, tuple(values)))
                class_sets.add(f['classes'])
                closure_witnesses += f['target'] > max(f['input_fragment_holes'])
                swaps += f['geometric_answers'] >= 4
    _require(len(shapes) >= 25 and len(sides) >= 4 and len(counts) >= 4, "Vary actual geometry and counts.")
    _require(targets == {0, 1, 2, 3, 4} and len(class_sets) >= 5, "Vary topological counts and alternatives.")
    _require(len(panels) == 4 and len(roles) >= 30, "Vary panel sides and palette roles.")
    _require(len(structures) >= 160 and closure_witnesses >= 30 and swaps >= 20,
             "Exercise emergent closure and multiple swaps, not only color changes.")
    a, b = generate_puzzle(707, 2, 1), generate_puzzle(707, 2, 1)
    encode = lambda p: json.dumps({s: [{k: np.asarray(v).tolist() for k, v in pair.items()}
                          for pair in p[s]] for s in p}, separators=(',', ':'))
    _require(encode(a) == encode(b), "Seed determinism failed.")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted.")
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong output accepted.")
    # The procedural safety net is checked without invoking a different rule.
    rng = np.random.default_rng(42)
    for rotation in range(4):
        roles6 = (0, 9, 1, 2, 3, 4)
        work, low, high, k0, k1 = _fallback(rng, roles6, rotation)
        for raw, k in ((low, k0), (high, k1)):
            inp = _attach(work, k, roles6, 0, rotation)
            out = raw.copy()
            out[out == roles6[2]] = roles6[5]
            _require(np.array_equal(solve(inp), out), "Fallback rotation failed.")


if __name__ == '__main__':
    self_test()
    print('self_test passed')
