"""Rejoin a cut stencil, then classify restored objects by their holes.

Only NumPy and the standard library are used. The solver reads pixels alone;
it does not call construction routines or consult seeds or saved examples.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np

_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


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
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grids required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a.astype(np.int8, copy=False)


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for dr, dc in _D4:
                x, y = rr + dr, cc + dc
                if 0 <= x < h and 0 <= y < w and mask[x, y] and not seen[x, y]:
                    seen[x, y] = True
                    stack.append((x, y))
        result.append(np.asarray(cells, dtype=int))
    return result


def _cell_mask(cells):
    cells = np.asarray(cells, dtype=int)
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    mask = np.zeros(tuple(hi - lo), dtype=bool)
    mask[tuple((cells - lo).T)] = True
    return mask, tuple(int(x) for x in lo)


def _holes(mask):
    # The padding ring supplies the exterior, even at a canvas boundary.
    inverse = ~np.pad(np.asarray(mask, dtype=bool), 1)
    h, w = inverse.shape
    return [cells - 1 for cells in _components(inverse)
            if not np.any((cells[:, 0] == 0) | (cells[:, 0] == h - 1)
                          | (cells[:, 1] == 0) | (cells[:, 1] == w - 1))]


def _hole_count(mask):
    return len(_holes(mask))


def _border(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _parse(input_grid):
    a = _grid(input_grid)
    border = _border(a)
    bg = int(np.bincount(border.astype(int), minlength=10).argmax())
    _require(np.count_nonzero(border == bg) * 2 > len(border), "background not readable")
    cells = _components(a != bg)
    _require(len(cells) == 4, "exactly four tiles required")
    for x, y in itertools.combinations(cells, 2):
        distance = np.abs(x[:, None, :] - y[None, :, :]).max(axis=2).min()
        _require(distance >= 2, "tiles need a blank separating cell")
    total = sum(map(len, cells))
    side = math.isqrt(total)
    _require(side * side == total, "tile area is not a square")
    patches = []
    for r in range(a.shape[0] - 1):
        for c in range(a.shape[1] - 2):
            patch = a[r:r + 2, c:c + 3]
            if len(np.unique(patch)) != 6 or np.any(patch == bg):
                continue
            carrier, ink = int(patch[0, 0]), int(patch[1, 0])
            key = patch[:, 1:].copy().reshape(-1)
            if not all(np.count_nonzero(a == color) == 1 for color in key):
                continue
            if np.any(~np.isin(a, [bg, carrier, ink, *key])):
                continue
            anchor = [i for i, comp in enumerate(cells)
                      if any((comp == (r, c)).all(axis=1))]
            if len(anchor) == 1:
                patches.append((r, c, carrier, ink, key, anchor[0]))
    _require(len(patches) == 1, "one unambiguous 2-by-3 key required")
    r, c, carrier, ink, key, ai = patches[0]
    ordered = [cells[ai]] + [comp for i, comp in enumerate(cells) if i != ai]
    pieces = []
    for i, comp in enumerate(ordered):
        footprint, (rr, cc) = _cell_mask(comp)
        ink_mask = (a[rr:rr + footprint.shape[0], cc:cc + footprint.shape[1]] == ink) & footprint
        if i == 0:
            kr, kc = r - rr, c - cc
            _require(footprint[kr:kr + 2, kc:kc + 3].all(), "key not inside anchor")
            ink_mask[kr:kr + 2, kc:kc + 3] = False
            _require(footprint[0, 0], "anchor lacks upper-left corner")
        pieces.append((footprint, ink_mask))
    return dict(background=bg, carrier=carrier, ink_color=ink, key=key,
                side=side, pieces=pieces, key_location=(r, c), cells=ordered,
                canvas=tuple(a.shape))


def _placement(mask, content, side, corner, rotation):
    mask, content = np.rot90(mask, rotation), np.rot90(content, rotation)
    h, w = mask.shape
    if h > side or w > side:
        return None
    r = 0 if corner < 2 else side - h
    c = 0 if corner % 2 == 0 else side - w
    coords = np.argwhere(mask) + (r, c)
    hits = {tuple(x) for x in coords} & {(0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)}
    target = (0 if corner < 2 else side - 1, 0 if corner % 2 == 0 else side - 1)
    if hits != {target}:
        return None
    bits = sum(1 << (int(x) * side + int(y)) for x, y in coords)
    ink_coords = np.argwhere(content) + (r, c)
    return bits, ink_coords, coords, rotation


def _pack(pieces, side, limit=2):
    """Enumerate every corner motion, deduplicating by reconstructed ink.

    Symmetric footprints do not suppress rotations of asymmetric contents.
    Corner occupancy fixes translation; no possible offset is skipped.
    """
    if len(pieces) != 4 or sum(int(x[0].sum()) for x in pieces) != side * side:
        return []
    anchor = _placement(*pieces[0], side, 0, 0)
    if anchor is None:
        return []
    table = {}
    for i in range(1, 4):
        for corner in range(1, 4):
            options, seen = [], set()
            for k in range(4):
                p = _placement(*pieces[i], side, corner, k)
                if p is None:
                    continue
                sig = (p[0], tuple(map(tuple, p[1])))
                if sig not in seen:
                    seen.add(sig)
                    options.append(p)
            table[i, corner] = options
    full = (1 << (side * side)) - 1
    answers, seen_ink = [], set()

    def visit(corner, remaining, occupied, choices):
        if len(answers) >= limit:
            return
        if corner == 4:
            if occupied != full:
                return
            ink = np.zeros((side, side), dtype=bool)
            owner = np.zeros((side, side), dtype=np.int8)
            rotations = [0] * 4
            piece_corners = [0] * 4
            for co, (pi, p) in enumerate(choices):
                if len(p[1]):
                    ink[tuple(p[1].T)] = True
                owner[tuple(p[2].T)] = pi
                rotations[pi], piece_corners[pi] = p[3], co
            sig = ink.tobytes()
            if sig not in seen_ink:
                seen_ink.add(sig)
                answers.append(dict(ink=ink, owner=owner, rotations=tuple(rotations),
                                    piece_corners=tuple(piece_corners)))
            return
        for pi in remaining:
            for p in table[pi, corner]:
                if p[0] & occupied:
                    continue
                visit(corner + 1, tuple(i for i in remaining if i != pi),
                      occupied | p[0], choices + [(pi, p)])

    visit(1, (1, 2, 3), anchor[0], [(0, anchor)])
    return answers


def _classify(ink, key, background):
    output = np.full(ink.shape, background, dtype=np.int8)
    for cells in _components(ink):
        mask, _ = _cell_mask(cells)
        k = _hole_count(mask)
        output[tuple(cells.T)] = int(key[k]) if k < 4 else background
    return output


def _analyze(input_grid):
    parsed = _parse(input_grid)
    candidates = _pack(parsed['pieces'], parsed['side'])
    _require(len(candidates) == 1, "tiles must reconstruct one unique ink mask")
    parsed.update(candidates[0])
    parsed['output'] = _classify(parsed['ink'], parsed['key'], parsed['background'])
    return parsed


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from the visible tile geometry and key."""
    return _analyze(input_grid)['output'].copy()


def _early_coloring(ink, owner, key, background):
    """Ablation: classify fragments within tiles before reconnecting them."""
    early = np.full(ink.shape, background, dtype=np.int8)
    for pi in range(4):
        fragment_output = _classify(ink & (owner == pi), key, background)
        early[ink & (owner == pi)] = fragment_output[ink & (owner == pi)]
    return early


def _compact(k):
    if k == 0:
        return np.ones((2, 3), dtype=bool)
    rows = 2 if k == 4 else 1
    cols = (k + rows - 1) // rows
    mask = np.ones((2 * rows + 1, 2 * cols + 1), dtype=bool)
    for j in range(k):
        mask[1 + 2 * (j // cols), 1 + 2 * (j % cols)] = False
    return mask


def _motif(k, rng, rich=False):
    """Create topology, then make only verified contour/cavity changes."""
    if k == 0:
        h, w = int(rng.integers(2, 5)), int(rng.integers(3, 6))
        mask = np.ones((h, w), dtype=bool)
        if rng.random() < .6:
            mask[:-1, 1:-1] = False  # open U, including short crooked strokes
    else:
        mask = _compact(k)
        if (rich or rng.random() < .25) and k in (1, 2):
            # One bent cavity alongside an optional smaller cavity. This is
            # area/shape variation of the same topological class, not noise.
            mask = np.ones((4, 4 if k == 1 else 6), dtype=bool)
            mask[1, 1] = mask[1, 2] = mask[2, 1] = False
            if k == 2:
                mask[1, 4] = False
        # Multi-row three-hole motifs prevent shape width being a class code.
        if k == 3 and rng.random() < .5:
            mask = np.ones((5, 5), dtype=bool)
            mask[1, 1] = mask[1, 3] = mask[3, 1] = False
        if rich or rng.random() < .35:
            axis = int(rng.integers(2))
            index = int(rng.integers(1, mask.shape[axis] - 1))
            mask = np.insert(mask, index, np.take(mask, index, axis=axis), axis=axis)
    # Growing exterior lobes and cavity bays changes area and silhouette.
    for _ in range(18 if rich else 6):
        candidate = mask.copy()
        r, c = int(rng.integers(mask.shape[0])), int(rng.integers(mask.shape[1]))
        candidate[r, c] = ~candidate[r, c]
        if candidate.any() and len(_components(candidate)) == 1 and _hole_count(candidate) == k:
            mask = candidate
    mask, _ = _cell_mask(np.argwhere(mask))
    return np.rot90(mask, int(rng.integers(4))).copy()


def _key_locations(owner):
    return [(r, c) for r in range(owner.shape[0] - 1)
            for c in range(owner.shape[1] - 2)
            if np.all(owner[r:r + 2, c:c + 3] == 0)]


def _partition(rng, side):
    """Wandering seams plus independently checked boundary transfers."""
    y = int(rng.integers(3, side - 2))
    x1, x2 = int(rng.integers(3, side - 2)), int(rng.integers(2, side - 1))
    owner = np.empty((side, side), dtype=np.int8)
    rough = float(rng.choice([.2, .5, .85]))
    for r in range(side):
        if r < y:
            if r > 1 and rng.random() < rough:
                x1 = int(np.clip(x1 + rng.choice([-1, 1]), 2, side - 2))
            owner[r, :x1], owner[r, x1:] = 0, 1
        else:
            if rng.random() < rough:
                x2 = int(np.clip(x2 + rng.choice([-1, 1]), 1, side - 1))
            owner[r, :x2], owner[r, x2:] = 2, 3
    corners = {(0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)}
    for _ in range(side * 2):
        r, c = int(rng.integers(side)), int(rng.integers(side))
        if (r, c) in corners:
            continue
        dr, dc = _D4[int(rng.integers(4))]
        rr, cc = r + dr, c + dc
        if not (0 <= rr < side and 0 <= cc < side):
            continue
        before, after = int(owner[r, c]), int(owner[rr, cc])
        if before == after:
            continue
        owner[r, c] = after
        if any(len(_components(owner == j)) != 1 for j in (before, after)):
            owner[r, c] = before
    return owner


def _asymmetric(mask):
    return len({(np.rot90(mask, k).shape, np.rot90(mask, k).tobytes()) for k in range(4)}) == 4


def _place_stencil(rng, owner, key_pos, counts, rich, test):
    side = len(owner)
    ink = np.zeros((side, side), dtype=bool)
    blocked = np.zeros_like(ink)
    kr, kc = key_pos
    blocked[max(0, kr - 1):kr + 3, max(0, kc - 1):kc + 4] = True
    # Larger motifs first; mandatory motifs are never silently dropped.
    motifs = [(k, _motif(k, rng, rich)) for k in counts]
    motifs.sort(key=lambda x: x[1].size, reverse=True)
    for k, mask in motifs:
        h, w = mask.shape
        locations = []
        for r in range(side - h + 1):
            for c in range(side - w + 1):
                if blocked[r:r + h, c:c + w].any():
                    continue
                participation = len(np.unique(owner[r:r + h, c:c + w][mask]))
                locations.append((r, c, participation))
        if not locations:
            return None
        # Favor closure across cuts, without using piece positions as a class.
        cross = [v for v in locations if v[2] >= (3 if test else 2)]
        pool = cross if cross and k in (1, 2, 3) and rng.random() < .85 else locations
        r, c, _ = pool[int(rng.integers(len(pool)))]
        ink[r:r + h, c:c + w] |= mask
        blocked[max(0, r - 1):min(side, r + h + 1), max(0, c - 1):min(side, c + w + 1)] = True
    return ink


def _pieces_from_square(owner, ink):
    result = []
    positions = []
    for i in range(4):
        footprint, (r, c) = _cell_mask(np.argwhere(owner == i))
        h, w = footprint.shape
        result.append((footprint, ink[r:r + h, c:c + w] & footprint))
        positions.append((r, c))
    return result, positions


def _scatter(rng, tiles, background):
    """Choose among all ordered shelf groupings on either axis."""
    dims = [t.shape for t in tiles]
    gap = int(rng.choice([1, 1, 2, 3]))
    layouts = []
    for axis in (0, 1):
        d = dims if axis == 0 else [v[::-1] for v in dims]
        for order in itertools.permutations(range(4)):
            for cuts in range(8):
                groups, group = [], []
                for j, i in enumerate(order):
                    group.append(i)
                    if j == 3 or (cuts >> j) & 1:
                        groups.append(group)
                        group = []
                hs = [max(d[i][0] for i in g) for g in groups]
                ws = [sum(d[i][1] for i in g) + gap * (len(g) - 1) for g in groups]
                height, width = sum(hs) + gap * (len(groups) - 1), max(ws)
                if height <= 30 and width <= 30:
                    layouts.append((axis, groups, hs, ws, height, width))
    if not layouts:
        # Two shelves always fit four pieces whose axes are at most fourteen.
        gap = 1
        groups = [[0, 1], [2, 3]]
        hs = [max(dims[i][0] for i in g) for g in groups]
        ws = [sum(dims[i][1] for i in g) + 1 for g in groups]
        layouts = [(0, groups, hs, ws, sum(hs) + 1, max(ws))]
    for _ in range(24):
        axis, groups, hs, ws, height, width = layouts[int(rng.integers(len(layouts)))]
        offsets, row = [None] * 4, 0
        d = dims if axis == 0 else [v[::-1] for v in dims]
        for g, h, w in zip(groups, hs, ws):
            col = int(rng.integers(width - w + 1))
            for i in g:
                rr = row + int(rng.integers(h - d[i][0] + 1))
                offsets[i] = (rr, col) if axis == 0 else (col, rr)
                col += d[i][1] + gap
            row += h + gap
        extent = (height, width) if axis == 0 else (width, height)
        pads = []
        for n in extent:
            room = 30 - n
            before = int(rng.integers(min(3, room) + 1))
            after = int(rng.integers(min(3, room - before) + 1))
            pads.append((before, after))
        canvas = tuple(extent[j] + sum(pads[j]) for j in range(2))
        a = np.full(canvas, background, dtype=np.int8)
        for i, (r, c) in enumerate(offsets):
            r, c = r + pads[0][0], c + pads[1][0]
            h, w = tiles[i].shape
            region = a[r:r + h, c:c + w]
            nonbg = tiles[i] != background
            region[nonbg] = tiles[i][nonbg]
        border = _border(a)
        if np.count_nonzero(border == background) * 2 > len(border):
            return a
    return None


def _features(a):
    ink, owner = a['ink'], a['owner']
    components = _components(ink)
    counts, spans, hole_areas, sizes = [], [], [], []
    irregular = 0
    for cells in components:
        mask, _ = _cell_mask(cells)
        holes = _holes(mask)
        counts.append(len(holes))
        spans.append(len(np.unique(owner[tuple(cells.T)])))
        sizes.append(len(cells))
        for hole in holes:
            hole_mask, _ = _cell_mask(hole)
            hole_areas.append(len(hole))
            irregular += int(not hole_mask.all())
    early = _early_coloring(ink, owner, a['key'], a['background'])
    valid_cross = [s for k, s in zip(counts, spans) if 0 < k < 4]
    return dict(side=a['side'], canvas=a['canvas'],
                counts=tuple(sorted(counts)), spans=tuple(spans), sizes=tuple(sorted(sizes)),
                max_retained_span=max(valid_cross, default=0),
                early_wrong=int(np.count_nonzero(early != a['output'])),
                hole_areas=tuple(sorted(hole_areas)), irregular_holes=irregular,
                rotations=a['rotations'], areas=tuple(int(p[0].sum()) for p in a['pieces']),
                background=a['background'], carrier=a['carrier'], ink_color=a['ink_color'],
                key=tuple(int(x) for x in a['key']), key_location=a['key_location'])


def _fallback_pair(rng, index, training, palette=None):
    """Construct a small guaranteed fitting witness, without saved grid data.

    A ring straddles the four-way seam. Independent one-cell tabs make all
    footprints asymmetric. These finite recipes are tested for both sizes and
    both teaching populations; they retain the exact same family semantics.
    """
    side = 12 if training and index == 0 else 13
    middle = side // 2
    owner = np.empty((side, side), dtype=np.int8)
    owner[:middle, :middle], owner[:middle, middle:] = 0, 1
    owner[middle:, :middle], owner[middle:, middle:] = 2, 3
    for r, c, label in ((2, middle, 0), (middle, 3, 0), (4, middle - 1, 1),
                         (side - 4, middle, 2), (middle, side - 4, 1),
                         (middle + 1, middle - 1, 3)):
        owner[r, c] = label
    ink = np.zeros((side, side), dtype=bool)
    second = training and index == 1
    top = _compact(3 if second else 0)
    ink[:top.shape[0], 4:4 + top.shape[1]] = top
    ink[5:8, 5:8] = _compact(1)
    k = 4 if second else 2
    bottom = np.ones((3, 2 * k + 1), dtype=bool)
    bottom[1, 1:2 * k:2] = False
    ink[-3:, 1:1 + bottom.shape[1]] = bottom
    pieces, _ = _pieces_from_square(owner, ink)
    packings = _pack(pieces, side)
    _require(len(packings) == 1 and np.array_equal(packings[0]['ink'], ink), "fallback reconstruction")
    colors = list(map(int, rng.permutation(10)[:7])) if palette is None else palette
    bg, carrier, ink_color, *keys = colors
    turns = [0, 1, 2, 3] if training and index == 0 else [0, *map(int, rng.integers(0, 4, 3))]
    tiles = []
    for pi, (footprint, content) in enumerate(pieces):
        tile = np.full(footprint.shape, bg, dtype=np.int8)
        tile[footprint], tile[content] = carrier, ink_color
        if pi == 0:
            tile[:2, :3] = [[carrier, keys[0], keys[1]], [ink_color, keys[2], keys[3]]]
        tiles.append(np.rot90(tile, turns[pi]).copy())
    grid = _scatter(rng, tiles, bg)
    if grid is None:
        # Four footprints fit two shelves with a completely blank outer border.
        dims = [t.shape for t in tiles]
        r2 = 2 + max(dims[0][0], dims[1][0])
        offsets = [(1, 1), (1, dims[0][1] + 2), (r2, 1), (r2, dims[2][1] + 2)]
        shape = (r2 + max(dims[2][0], dims[3][0]) + 1,
                 max(dims[0][1] + dims[1][1], dims[2][1] + dims[3][1]) + 3)
        grid = np.full(shape, bg, dtype=np.int8)
        for (r, c), tile in zip(offsets, tiles):
            h, w = tile.shape
            grid[r:r + h, c:c + w] = tile
    return {'input': grid, 'output': _classify(ink, keys, bg)}


def _make_pair(rng, index, training, emphasis, palette=None):
    if index == 0 and training:
        counts, side = [0, 1, 2], 12
    elif index == 1 and training:
        counts, side = [1, 3, 4], 13
    else:
        counts = [int(rng.integers(1, 4)), int(rng.integers(5))]
        counts += [int(rng.integers(5)) for _ in range(int(rng.integers(1, 3)))]
        side = int(rng.integers(11, 14) if training else rng.integers(13, 15))
        if emphasis != 0 and not training:
            side = 13
    rich = emphasis == 2 and not training or index > 1 and rng.random() < .45
    for attempt in range(160):
        if attempt and attempt % 16 == 0 and not (training and index < 2):
            # Optional populations are proposals, not mandatory latent classes.
            # Resample an oversized proposal instead of retrying an impossible
            # set of bounding boxes for the entire finite attempt budget.
            counts = [int(rng.integers(1, 4)), int(rng.integers(5))]
            counts += [int(rng.integers(5)) for _ in range(int(rng.integers(1, 3)))]
            if attempt >= 96:
                counts = [1, int(rng.integers(4)), int(rng.integers(5))]
                side = 13
        owner = _partition(rng, side)
        key_positions = _key_locations(owner)
        if not key_positions:
            continue
        key_pos = key_positions[int(rng.integers(len(key_positions)))]
        ink = _place_stencil(rng, owner, key_pos, counts, rich and attempt < 120, not training)
        if ink is None:
            continue
        pieces, starts = _pieces_from_square(owner, ink)
        if index == 0 and training and not all(_asymmetric(p[0]) for p in pieces):
            continue
        # Uniqueness includes transported content and symmetric motions.
        candidates = _pack(pieces, side)
        if len(candidates) != 1 or not np.array_equal(candidates[0]['ink'], ink):
            continue
        colors = list(map(int, rng.permutation(10)[:7])) if palette is None else palette
        bg, carrier, ink_color, *keys = colors
        answer = _classify(ink, keys, bg)
        early = _early_coloring(ink, owner, keys, bg)
        if np.array_equal(early, answer):
            continue
        spans = []
        for comp in _components(ink):
            mask, _ = _cell_mask(comp)
            if 0 < _hole_count(mask) < 4:
                spans.append(len(np.unique(owner[tuple(comp.T)])))
        if max(spans, default=0) < (2 if training else 3):
            continue
        turns = [0] + [int(x) for x in rng.integers(0, 4, 3)]
        if index == 0 and training:
            turns[1:] = list(map(int, rng.permutation([1, 2, 3])))
        elif index == 1 and training:
            turns = [0, 0, 0, 0]
        elif rng.random() < .25:
            turns = [0, 0, 0, 0]
        tiles = []
        for pi, (footprint, contents) in enumerate(pieces):
            tile = np.full(footprint.shape, bg, dtype=np.int8)
            tile[footprint] = carrier
            tile[contents] = ink_color
            if pi == 0:
                kr, kc = key_pos[0] - starts[0][0], key_pos[1] - starts[0][1]
                tile[kr:kr + 2, kc:kc + 3] = [[carrier, keys[0], keys[1]],
                                             [ink_color, keys[2], keys[3]]]
            tiles.append(np.rot90(tile, turns[pi]).copy())
        grid = _scatter(rng, tiles, bg)
        if grid is None:
            continue
        # Parsing returned pixels is mandatory, and construction supplies labels.
        try:
            a = _analyze(grid)
        except AssertionError:
            continue
        f = _features(a)
        if (np.array_equal(a['output'], answer) and f['early_wrong'] > 0
                and f['max_retained_span'] >= (2 if training else 3)):
            return {'input': grid, 'output': answer}
    return _fallback_pair(rng, index, training, palette)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ntrain, ntest = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    emphasis = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    palettes = []
    for split, count in (('train', ntrain), ('test', ntest)):
        for i in range(count):
            palette = None
            if len(palettes) >= 2 and rng.random() < .2:
                palette = palettes[int(rng.integers(len(palettes)))].copy()
            for _ in range(8):
                pair = _make_pair(rng, i, split == 'train', emphasis, palette)
                sig = _signature(pair['input'])
                if sig in seen:
                    continue
                seen.add(sig)
                puzzle[split].append(pair)
                a = _parse(pair['input'])
                palettes.append([a['background'], a['carrier'], a['ink_color'], *map(int, a['key'])])
                break
            else:
                raise RuntimeError("distinct-input bound exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    try:
        _counts(len(puzzle['train']), len(puzzle['test']))
    except ValueError as exc:
        raise AssertionError("illegal puzzle counts") from exc
    seen, taught, rotations = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            a = _analyze(inp)
            _require(np.array_equal(a['output'], out), "wrong restored-object coloring")
            f = _features(a)
            _require(f['early_wrong'] > 0, "no witness against fragment-first classification")
            _require(f['max_retained_span'] >= (2 if split == 'train' else 3), "missing seam participation")
            assembled_only = np.where(a['ink'], a['ink_color'], a['background'])
            _require(not np.array_equal(assembled_only, out), "A ablation unchanged")
            _require(out.shape == (a['side'], a['side']), "wrong square extent")
            _require(np.isin(out, [a['background'], *a['key']]).all(), "carrier/key artifact remains")
            if split == 'train':
                taught.update(f['counts'])
                rotations.update(f['rotations'])
    _require({0, 1, 2, 3, 4} <= taught, "training must teach every slot and erasure")
    _require({0, 1, 2, 3} <= rotations, "training must demonstrate quarter turns")


def _equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(np.array_equal(a[k], b[k])
               for a, b in zip(first[s], second[s]) for k in ('input', 'output'))
               for s in ('train', 'test'))


def self_test():
    """Bounded semantic regressions and actual feature coverage on 42 seeds."""
    diagonal = np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    _require(_hole_count(diagonal) == 1, "diagonal escape is not an opening")
    diagonal[0, 1] = False
    _require(_hole_count(diagonal) == 0, "orthogonal opening ignored")
    u = np.asarray([[1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool)
    _require(_hole_count(u) == 0, "boundary manufactured a hole")
    # Footprint symmetry must never hide a distinct transported payload.
    square = np.ones((3, 3), dtype=bool)
    blank = np.zeros((3, 3), dtype=bool)
    off_center = blank.copy()
    off_center[0, 1] = True
    symmetric = [(square, blank), (square, off_center), (square, blank), (square, blank)]
    _require(len(_pack(symmetric, 6)) == 2, "symmetric payload ambiguity missed")
    _require(len(_pack([(square, blank)] * 4, 6)) == 1, "equivalent motions rejected")
    key = np.asarray([2, 3, 4, 6])
    for k in range(5):
        mask = _compact(k)
        out = _classify(mask, key, 0)
        _require(np.all(out[mask] == (key[k] if k < 4 else 0)), "class regression")
    features, signatures, canvases = [], set(), set()
    within = Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        local = []
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _analyze(pair['input'])
                f = _features(a)
                f['canvas'] = tuple(pair['input'].shape)
                features.append(f)
                local.append(f)
                signatures.add(_signature(pair['input']))
                canvases.add(pair['input'].shape)
        for name in ('areas', 'spans', 'sizes', 'canvas', 'key_location'):
            within[name] += len({f[name] for f in local}) > 1
    _require(len(signatures) == 42 * 5, "cross-seed inputs duplicated")
    _require(len(canvases) >= 35, "canvas geometry collapsed")
    _require(len({f['areas'] for f in features}) >= 100, "piece area profiles collapsed")
    _require(len({f['counts'] for f in features}) >= 12, "object populations collapsed")
    _require(len({f['sizes'] for f in features}) >= 100, "object geometry collapsed")
    _require({f['background'] for f in features} == set(range(10)), "background roles collapsed")
    _require({f['carrier'] for f in features} == set(range(10)), "carrier roles collapsed")
    _require({f['ink_color'] for f in features} == set(range(10)), "ink roles collapsed")
    _require(any(f['max_retained_span'] == 4 for f in features), "four-tile object missing")
    _require(sum(f['irregular_holes'] > 0 for f in features) >= 10, "irregular cavities missing")
    _require(len({f['hole_areas'] for f in features}) >= 20, "cavity geometry collapsed")
    _require(all(within[k] >= 38 for k in within), "within-puzzle structures collapsed")
    a = generate_puzzle(731, 2, 1)
    _require(_equal(a, generate_puzzle(731, 2, 1)), "nondeterministic")
    _require(not _equal(a, generate_puzzle(732, 2, 1)), "seed ignored")
    fallback_rng = np.random.default_rng(991)
    fallback = {'train': [_fallback_pair(fallback_rng, i, True) for i in range(2)],
                'test': [_fallback_pair(fallback_rng, 0, False)]}
    validate_generated_puzzle(fallback)
    permutation = np.asarray([8, 5, 0, 7, 1, 9, 3, 6, 4, 2], dtype=np.int8)
    for pair in a['train'] + a['test']:
        _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
                 "visible palette roles were not respected")
    for tr, te in ((10, 4), (2, 1), (5, 4)):
        p = generate_puzzle(1000 + tr, tr, te)
        _require((len(p['train']), len(p['test'])) == (tr, te), "count ignored")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds and semantic regressions')
