"""Nested stencil jigsaw: topology defines material, complementarity defines fit.

The sampler constructs abstract stencils, not parent puzzles.  Its canonical
labels are checked against reconstruction from the independently drawn input.
No files, training pairs, random state, or generation metadata enter solve().
"""
from __future__ import annotations

from hashlib import sha256
import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 942865257, 533466816, 51429936, 928210799, 744499130,
              272639790, 111789981, 916443732, 754158176, 814307520,
              226274113, 349580309, 522248771, 824061192, 829936883,
              385465240, 828081872, 569405325, 21103286, 488577965,
              133124107, 217896538, 275270326, 190698917, 93013518,
              593862515, 964256013, 485796526, 201081438, 83510367,
              122582045, 803870311, 568223505, 904680941, 841137918,
              152917731, 831591467, 48316446, 534915893, 168777806,
              658872501)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _mask_bits(mask):
    return int.from_bytes(np.packbits(mask, bitorder='little').tobytes(), 'little')


def _bits_mask(bits, shape):
    count = shape[0]*shape[1]
    packed = np.frombuffer(bits.to_bytes((count+7)//8, 'little'), dtype=np.uint8)
    return np.unpackbits(packed, count=count, bitorder='little').reshape(shape).astype(bool)


def _flood_bits(seed, allowed, width, first_column, last_column):
    """Exact four-neighbor reachability, with row wrapping explicitly masked."""
    region = seed & allowed
    while True:
        expanded = (region | ((region & ~first_column) >> 1)
                    | ((region & ~last_column) << 1)
                    | (region >> width) | (region << width)) & allowed
        if expanded == region:
            return region
        region = expanded


def _components(mask):
    h, w = mask.shape
    if not mask.size:
        return []
    remaining = _mask_bits(mask)
    full = (1 << (h*w))-1
    first = full // ((1 << w)-1)
    last = first << (w-1)
    result = []
    while remaining:
        component = _flood_bits(remaining & -remaining, remaining, w, first, last)
        # The least occupied bit preserves the previous row-major component
        # ordering. Only within-component coordinate order changes.
        result.append(np.argwhere(_bits_mask(component, mask.shape)))
        remaining ^= component
    return result


def _holes(mask):
    if not mask.size:
        return np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    full = (1 << (h*w))-1
    first = full // ((1 << w)-1)
    last = first << (w-1)
    top = (1 << w)-1
    boundary = first | last | top | (top << ((h-1)*w))
    empty = full ^ _mask_bits(mask)
    # Flooding all empty edge cells equals padding with an empty exterior and
    # flooding from its corner, including contours that touch the grid edge.
    outside = _flood_bits(boundary, empty, w, first, last)
    return _bits_mask(empty ^ outside, mask.shape)


def _tile(diagram, material, leaves):
    return dict(diagram=diagram, material=material, leaves=leaves)


def _point(key=0):
    return _tile(np.ones((1, 1), bool), np.ones((1, 1), bool), [(key, 0, 0)])


def _join(parts, axis, offsets, gaps):
    """Groups add no contour. Keep both the drawing and its parity-filled mask."""
    low = min(offsets)
    shape = [0, 0]
    starts = []
    position = 0
    for i, part in enumerate(parts):
        start = [0, 0]
        start[axis], start[1-axis] = position, offsets[i] - low
        starts.append(start)
        position += part['diagram'].shape[axis]
        if i < len(gaps):
            position += gaps[i]
        shape[1-axis] = max(shape[1-axis], start[1-axis] + part['diagram'].shape[1-axis])
    shape[axis] = position
    diagram = np.zeros(shape, bool)
    material = np.zeros(shape, bool)
    leaves = []
    for part, (r, c) in zip(parts, starts):
        h, w = part['diagram'].shape
        diagram[r:r+h, c:c+w] |= part['diagram']
        material[r:r+h, c:c+w] |= part['material']
        leaves.extend((i, y+r, x+c) for i, y, x in part['leaves'])
    return _tile(diagram, material, leaves)


def _wrap(content):
    diagram = np.pad(content['diagram'], 2)
    diagram[0, :] = diagram[-1, :] = True
    diagram[:, 0] = diagram[:, -1] = True
    # Every boundary toggles ownership INCLUDING its normalized boundary cells.
    material = np.ones_like(diagram)
    material[2:-2, 2:-2] ^= content['material']
    return _tile(diagram, material, [(i, r+2, c+2) for i, r, c in content['leaves']])


def _parse_diagram(mask):
    objects = []
    for coords in _components(mask):
        local = np.zeros_like(mask)
        local[coords[:, 0], coords[:, 1]] = True
        hole = _holes(local)
        _require(len(_components(hole)) <= 1, 'A contour cannot have multiple holes')
        objects.append(dict(coords=coords, lo=coords.min(0), hi=coords.max(0),
                            hole=hole, contour=bool(hole.any())))
    _require(objects, 'Empty stencil')
    parents = []
    for i, child in enumerate(objects):
        candidates = []
        for j, obj in enumerate(objects):
            if i == j or not obj['contour']:
                continue
            inside = obj['hole'][child['coords'][:, 0], child['coords'][:, 1]]
            _require(inside.all() or not inside.any(), 'Crossing objects')
            if inside.all():
                candidates.append(j)
        parents.append(min(candidates, key=lambda j: int(objects[j]['hole'].sum()))
                       if candidates else None)
    roots = [i for i, p in enumerate(parents) if p is None]
    _require(len(roots) == 1, 'Each carrier has one root object')
    children = {i: [] for i in range(len(objects))}
    for i, p in enumerate(parents):
        if p is not None:
            children[p].append(i)
    for i, obj in enumerate(objects):
        _require(bool(children[i]) == obj['contour'], 'Contours must enclose content')
    return objects, children, roots[0], parents


def _normalize(mask):
    """Normalize clear cardinal layouts using descendant blob alignments.

    A whitespace cut separates siblings. Aligned blob projections determine
    cross-axis offsets, even across nesting boundaries. Both possible slicing
    directions must agree. This deliberately rejects arbitrary diagonals.
    """
    objects, children, root, parents = _parse_diagram(mask)
    cache = {}

    def aligned(parts, axis):
        cross = 1-axis
        edges = []
        for i in range(len(parts)):
            for j in range(i+1, len(parts)):
                shifts = set()
                for a, ay, ax in parts[i]['leaves']:
                    for b, by, bx in parts[j]['leaves']:
                        if max(objects[a]['lo'][cross], objects[b]['lo'][cross]) <= min(objects[a]['hi'][cross], objects[b]['hi'][cross]):
                            shifts.add((ay, ax)[cross] - (by, bx)[cross])
                _require(len(shifts) <= 1, 'Competing blob alignments')
                if shifts:
                    edges.append((i, j, shifts.pop()))
        offsets = {0: 0}
        for _ in parts:
            for i, j, delta in edges:
                if i in offsets and j in offsets:
                    _require(offsets[j] == offsets[i] + delta, 'Inconsistent alignment')
                elif i in offsets:
                    offsets[j] = offsets[i] + delta
                elif j in offsets:
                    offsets[i] = offsets[j] - delta
        _require(len(offsets) == len(parts), 'Unanchored diagonal layout')
        return [offsets[i] for i in range(len(parts))]

    def group(ids):
        ids = tuple(sorted(ids))
        if ids in cache:
            return cache[ids]
        if len(ids) == 1:
            i = ids[0]
            result = _wrap(group(children[i])) if objects[i]['contour'] else _point(i)
        else:
            possibilities = []
            for axis in (0, 1):
                ordered = sorted(ids, key=lambda i: int(objects[i]['lo'][axis]))
                sections = []
                end = -2
                for i in ordered:
                    if int(objects[i]['lo'][axis]) > end + 1:
                        sections.append([])
                    sections[-1].append(i)
                    end = max(end, int(objects[i]['hi'][axis]))
                if len(sections) < 2:
                    continue
                try:
                    parts = [group(s) for s in sections]
                    possibilities.append(_join(parts, axis, aligned(parts, axis), [1]*(len(parts)-1)))
                except ValueError:
                    continue
            _require(possibilities, 'No clear cardinal decomposition')
            result = possibilities[0]
            for other in possibilities[1:]:
                _require(_key(result['diagram']) == _key(other['diagram']) and
                         _key(result['material']) == _key(other['material']) and
                         sorted(result['leaves']) == sorted(other['leaves']), 'Ambiguous normalization')
        cache[ids] = result
        return result

    tile = group([root])
    depth = 0
    for i in range(len(objects)):
        d, p = 0, i
        while p is not None:
            d += int(objects[p]['contour'])
            p = parents[p]
        depth = max(depth, d)
    return tile, dict(contours=sum(o['contour'] for o in objects),
                      blobs=sum(not o['contour'] for o in objects), depth=depth,
                      root=objects[root], objects=objects)


def _read_input(input_grid):
    grid = np.asarray(input_grid)
    _require(grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer), 'Integer 2D input required')
    _require(all(1 <= d <= 30 for d in grid.shape) and np.all((0 <= grid) & (grid <= 9)), 'Invalid input bounds')
    _require(np.all(grid[0] == grid[0, 0]) and np.all(grid[-1] == grid[0, 0]) and
             np.all(grid[:, 0] == grid[0, 0]) and np.all(grid[:, -1] == grid[0, 0]), 'Blank border required')
    bg = int(grid[0, 0])
    colors, counts = np.unique(grid, return_counts=True)
    singletons = [int(c) for c, n in zip(colors, counts) if c != bg and n == 1]
    _require(len(singletons) == 1, 'One unique singleton anchor color required')
    marker = singletons[0]
    mr, mc = map(int, np.argwhere(grid == marker)[0])
    neighbors = {int(grid[r, c]) for dr, dc in EDGE
                 if 0 <= (r := mr+dr) < grid.shape[0] and 0 <= (c := mc+dc) < grid.shape[1]
                 and grid[r, c] not in (bg, marker)}
    _require(len(neighbors) == 1, 'Marker attaches to one carrier contour')
    anchor = neighbors.pop()
    carriers = [int(c) for c in colors if c not in (bg, marker)]
    _require(2 <= len(carriers) <= 4, 'Two to four stencils required')
    pieces, features = [], []
    anchor_corner = None
    for color in carriers:
        mask = grid == color
        if color == anchor:
            mask = mask | (grid == marker)
        tile, info = _normalize(mask)
        material = tile['material'].astype(np.int8) * color
        material[~tile['material']] = -1
        if color == anchor:
            obj = info['root']
            corners = [tuple(obj['lo']), (int(obj['lo'][0]), int(obj['hi'][1])),
                       tuple(obj['hi']), (int(obj['hi'][0]), int(obj['lo'][1]))]
            _require((mr, mc) in corners and obj['contour'], 'Marker must be at a root contour bounding-box corner')
            corner = corners.index((mr, mc))
            anchor_corner = corner
            h, w = material.shape
            material[[(0, 0), (0, w-1), (h-1, w-1), (h-1, 0)][corner]] = marker
        pieces.append(material)
        features.append(dict(info, color=color, tile=tile))
    area = sum(int(np.sum(p >= 0)) for p in pieces)
    side = int(np.sqrt(area))
    _require(side * side == area, 'Normalized material area must be square')
    return dict(background=bg, marker=marker, anchor=anchor, carriers=carriers,
                pieces=pieces, side=side, features=features, anchor_corner=anchor_corner)


def _pack(pieces, side, marker, limit=2):
    """Enumerate colored rotation-only exact covers; stop after two answers."""
    full = (1 << (side*side)) - 1
    choices = []
    anchor = next(i for i, p in enumerate(pieces) if np.any(p == marker))
    for i, p in enumerate(pieces):
        options, rotations = {}, set()
        for k in range(4):
            q = np.rot90(p, k)
            if _key(q) in rotations:
                continue
            rotations.add(_key(q))
            h, w = q.shape
            if h > side or w > side:
                continue
            if i == anchor and q[0, 0] != marker:
                continue
            cells = [(int(r)*side + int(c), int(q[r, c])) for r, c in np.argwhere(q >= 0)]
            base = sum(1 << pos for pos, _ in cells)
            offsets = [0] if i == anchor else [r*side+c for r in range(side-h+1) for c in range(side-w+1)]
            for offset in offsets:
                mask = base << offset
                # In this family each piece has one carrier, except the anchor.
                options.setdefault(mask, (mask, offset, cells))
        if not options:
            return []
        choices.append(list(options.values()))
    solutions, signatures = [], set()
    placed = [None]*len(pieces)

    def search(occupied, remaining):
        if len(solutions) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            out = np.empty((side, side), np.int8)
            for _, offset, cells in placed:
                for pos, color in cells:
                    out.flat[pos+offset] = color
            key = _key(out)
            if key not in signatures:
                signatures.add(key)
                solutions.append(out)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            for option in choices[i]:
                if option[0] & first and not option[0] & occupied:
                    placed[i] = option
                    search(occupied | option[0], tuple(j for j in remaining if j != i))
                    if len(solutions) >= limit:
                        return
            placed[i] = None

    for option in choices[anchor]:
        placed[anchor] = option
        search(option[0], tuple(i for i in range(len(pieces)) if i != anchor))
        if len(solutions) >= limit:
            break
    return solutions


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _read_input(input_grid)
    answers = _pack(facts['pieces'], facts['side'], facts['marker'])
    _require(len(answers) == 1, 'Require one colored assembly')
    return answers[0]


# Abstract sampler nodes: B = filled blob, C = contour, G = spatial grouping.
def _leafcount(node):
    if node[0] == 'B':
        return 1
    if node[0] == 'C':
        return _leafcount(node[1])
    return sum(_leafcount(c) for c in node[2])


def _group(rng, children, axis=None):
    if len(children) == 1:
        return children[0]
    if axis is None:
        axis = int(rng.integers(2))
    return ('G', axis, tuple(children), tuple(int(rng.integers(_leafcount(c))) for c in children))


def _canonical(node, serial=None):
    if serial is None:
        serial = [0]
    if node[0] == 'B':
        key = serial[0]
        serial[0] += 1
        return _point(key)
    if node[0] == 'C':
        return _wrap(_canonical(node[1], serial))
    parts = [_canonical(c, serial) for c in node[2]]
    offsets = [-p['leaves'][i][2-node[1]] for p, i in zip(parts, node[3])]
    return _join(parts, node[1], offsets, [1]*(len(parts)-1))


def _swap(n):
    if n[0] == 'B':
        return n
    if n[0] == 'C':
        return ('C', _swap(n[1]))
    return ('G', 1-n[1], tuple(_swap(c) for c in n[2]), n[3])


def _profile_plate(rng, nrows, ncols):
    # Variable-length rows share one visible anchor column. At least one row
    # reaches each horizontal extreme, so dimensions are known to construction
    # without making blob COUNT determine the complete two-dimensional shape.
    anchor = int(rng.integers(ncols))
    intervals = [(int(rng.integers(anchor+1)), int(rng.integers(anchor, ncols))) for _ in range(nrows)]
    left_row, right_row = map(int, rng.integers(nrows, size=2))
    intervals[left_row] = (0, intervals[left_row][1])
    intervals[right_row] = (intervals[right_row][0], ncols-1)
    rows = [_group(rng, [('B',)]*(b-a+1), 1) for a, b in intervals]
    content = rows[0] if nrows == 1 else ('G', 0, tuple(rows), tuple(anchor-a for a, b in intervals))
    return ('C', content)


def _random_plate(rng, rich=False):
    node = _profile_plate(rng, int(rng.integers(2 if rich else 1, 6)), int(rng.integers(1, 6)))
    return _swap(node) if rng.random() < .5 else node


def _equal_offset_group(rng, nodes, axis):
    parts = [_canonical(n) for n in nodes]
    common = set(p[2-axis] for p in parts[0]['leaves'])
    for part in parts[1:]:
        common &= {p[2-axis] for p in part['leaves']}
    _require(common, 'No common construction anchor')
    coordinate = int(rng.choice(sorted(common)))
    anchors = tuple(next(i for i, leaf in enumerate(p['leaves']) if leaf[2-axis] == coordinate) for p in parts)
    return ('G', axis, tuple(nodes), anchors)


def _three_cavities(rng):
    # A dimension equation, not a saved mask: H = b+c+1 = A+W+1.
    # Three independently drawn pin profiles occupy the resulting L layout.
    height = int(rng.choice([13, 15]))
    widths = [(7, 5), (5, 7)] if height == 13 else [(9, 5), (7, 7), (5, 9)]
    a, w = widths[int(rng.integers(len(widths)))]
    tall = _profile_plate(rng, (height-3)//2, (a-3)//2)
    column = [_profile_plate(rng, 1, (w-3)//2),
              _profile_plate(rng, (height-9)//2, (w-3)//2)]
    if rng.random() < .5:
        column.reverse()
    nodes = [tall] + column
    groups = [tall, _equal_offset_group(rng, column, 0)]
    if rng.random() < .5:
        groups.reverse()
    root = ('C', _equal_offset_group(rng, groups, 1))
    if rng.random() < .5:
        root, nodes = _swap(root), [_swap(n) for n in nodes]
    return root, nodes


def _nested_witness(rng):
    """A varied, small depth-three witness; not the multi-cavity sampler.

    A 5x5 inner contour sits beside a point column and above a point row.
    Their lengths, sides, axes and descendant alignment anchors vary. Only
    arrangements whose minimal content is 7x7 are retained, giving a plate
    of side 11 and a surrounding stencil of side 15, which fit on the input.
    """
    for _ in range(300):
        inner = ('C', ('B',))
        column = _group(rng, [('B',)] * int(rng.integers(1, 4)), 0)
        top = [inner, column]
        if rng.random() < .5:
            top.reverse()
        row = _group(rng, [('B',)] * int(rng.integers(1, 5)), 1)
        rows = [_group(rng, top, 1), row]
        if rng.random() < .5:
            rows.reverse()
        child = ('C', _group(rng, rows, 0))
        tile = _canonical(child)
        if tile['material'].shape == (11, 11) and not np.array_equal(tile['material'], tile['material'].T):
            return ('C', child), [child]
    raise RuntimeError('Bounded nested-witness search exhausted')


def _abstract_pair(rng, count, rich):
    if count == 1:
        return _nested_witness(rng)
    if count == 3:
        return _three_cavities(rng)
    for _ in range(2400):
        children = [_random_plate(rng, rich) for _ in range(count)]
        tiles = [_canonical(c) for c in children]
        if any(max(t['diagram'].shape) > 13 for t in tiles):
            continue
        # Each cavity must be distinguishable by its material shape, not color.
        classes = [min(_key(np.rot90(t['material'], k)) for k in range(4)) for t in tiles]
        if len(set(classes)) != count:
            continue
        if count >= 3 and rng.random() < .75:
            cut = int(rng.integers(1, count))
            axis = int(rng.integers(2))
            content = _group(rng, [_group(rng, children[:cut], 1-axis),
                                   _group(rng, children[cut:], 1-axis)], axis)
        else:
            content = _group(rng, children)
        root = ('C', content)
        tile = _canonical(root)
        h, w = tile['diagram'].shape
        if h != w or not 9 <= h <= 19:
            continue
        # Require an asymmetric pin/socket pattern. Even a two-piece example
        # therefore distinguishes reflection and unguided rotation.
        if np.array_equal(tile['material'], tile['material'].T):
            continue
        return root, children
    raise RuntimeError('Bounded abstract stencil search exhausted')


def _draw(node, rng, compact=False):
    """New metric realization of a fixed topology, with guaranteed corner cells."""
    serial = [0]

    def visit(n):
        if n[0] == 'B':
            key = serial[0]
            serial[0] += 1
            h, w = (1, 1) if compact else ((int(rng.integers(1, 3)), int(rng.integers(2, 4))))
            if rng.random() < .5:
                h, w = w, h
            mask = np.ones((h, w), bool)
            if h > 1 and w > 1 and rng.random() < .45:
                mask[0, 0] = False
            return _tile(mask, mask, [(key, (h-1)//2, (w-1)//2)])
        if n[0] == 'G':
            parts = [visit(c) for c in n[2]]
            offsets = [-p['leaves'][i][2-n[1]] for p, i in zip(parts, n[3])]
            gap = [1 if compact else int(rng.integers(1, 4)) for _ in parts[:-1]]
            return _join(parts, n[1], offsets, gap)
        part = visit(n[1])
        pad = [2 if compact else int(rng.integers(2, 4)) for _ in range(4)]
        top, bottom, left, right = pad
        h, w = part['diagram'].shape
        H, W = h+top+bottom, w+left+right
        outline = np.zeros((H, W), bool)
        outline[0] = outline[-1] = True
        outline[:, 0] = outline[:, -1] = True
        # Corner-preserving outward shoulders/inward bays: replace a straight
        # run by a one-cell-inward orthogonal detour in an empty side corridor.
        # The larger clearance above keeps the path disjoint from the contents.
        if not compact:
            for side, clearance, length in ((0, top, W), (1, bottom, W),
                                             (2, left, H), (3, right, H)):
                if clearance < 3 or length < 7 or rng.random() < .30:
                    continue
                a = int(rng.integers(2, length-3))
                b = int(rng.integers(a+1, length-1))
                if side < 2:
                    row, inner = (0, 1) if side == 0 else (H-1, H-2)
                    outline[row, a:b+1] = False
                    outline[inner, a-1:b+2] = True
                else:
                    col, inner = (0, 1) if side == 2 else (W-1, W-2)
                    outline[a:b+1, col] = False
                    outline[a-1:b+2, inner] = True
        outline[top:top+h, left:left+w] |= part['diagram']
        return _tile(outline, outline, [(i, y+top, x+left) for i, y, x in part['leaves']])

    return visit(node)['diagram']


def _scatter(drawings, colors, rotations, background, marker, rng):
    oriented = []
    for i, (mask, color, rotation) in enumerate(zip(drawings, colors, rotations)):
        arr = np.full(mask.shape, -1, np.int8)
        arr[mask] = color
        if i == 0:
            arr[0, 0] = marker
        oriented.append(np.rot90(arr, rotation))
    # Tight-box packing is conservative here: different stencil diagrams must
    # not be confused by enclosure or overlap in their input presentation.
    for attempt in range(20):
        canvas = np.full((28, 28), -1, np.int8)
        boxes = []
        order = sorted(range(len(oriented)), key=lambda i: -oriented[i].size) if attempt == 0 else list(map(int, rng.permutation(len(oriented))))
        okay = True
        for i in order:
            p = oriented[i]
            h, w = p.shape
            options = []
            for r in range(29-h):
                for c in range(29-w):
                    if any(not (r+h < a or a+hh < r or c+w < b or b+ww < c) for a, b, hh, ww in boxes):
                        continue
                    H = max([r+h] + [a+hh for a, b, hh, ww in boxes])
                    W = max([c+w] + [b+ww for a, b, hh, ww in boxes])
                    options.append((H*W + float(rng.random())*20, r, c))
            if not options:
                okay = False
                break
            _, r, c = min(options)
            canvas[r:r+h, c:c+w] = p
            boxes.append((r, c, h, w))
        if okay:
            rr, cc = np.nonzero(canvas >= 0)
            core = canvas[rr.min():rr.max()+1, cc.min():cc.max()+1]
            h, w = core.shape
            H = int(rng.integers(h+2, min(30, h+5)+1))
            W = int(rng.integers(w+2, min(30, w+5)+1))
            r, c = int(rng.integers(1, H-h)), int(rng.integers(1, W-w))
            grid = np.full((H, W), background, np.int8)
            grid[r:r+h, c:c+w] = np.where(core >= 0, core, background)
            return grid
    return None


def _metric_variation(grid, rng):
    """Spend available canvas space on real, topology-preserving variation.

    Compact placement is often essential for fitting all diagrams. After it
    succeeds, nonuniform row/column stretches enlarge selected blobs; small
    contour bays use existing empty corridors. Neither changes a logical mask.
    Only the current grid and at most one candidate copy are retained.
    """
    facts = _read_input(grid)
    bg, marker = facts['background'], facts['marker']
    occupied = np.argwhere(grid != bg)
    lo, hi = occupied.min(0), occupied.max(0)+1
    grid = np.pad(grid[lo[0]:hi[0], lo[1]:hi[1]], 1, constant_values=bg)
    for axis in map(int, rng.permutation(2)):
        if grid.shape[axis] >= 30:
            continue
        facts = _read_input(grid)
        marker_index = int(np.argwhere(grid == marker)[0, axis])
        indices = sorted({int(p[axis]) for f in facts['features'] for obj in f['objects']
                          if not obj['contour'] for p in obj['coords'] if p[axis] != marker_index})
        if indices:
            repeats = np.ones(grid.shape[axis], int)
            repeats[int(rng.choice(indices))] = 2
            grid = np.repeat(grid, repeats, axis=axis)
    facts = _read_input(grid)
    baseline = {f['color']: _key(f['tile']['material']) for f in facts['features']}
    accepted = 0
    for _ in range(18):
        f = facts['features'][int(rng.integers(len(facts['features'])))]
        contours = [o for o in f['objects'] if o['contour']]
        obj = contours[int(rng.integers(len(contours)))]
        side = int(rng.integers(4))
        lo, hi = obj['lo'], obj['hi']
        along = 1 if side < 2 else 0
        length = int(hi[along]-lo[along]+1)
        if length < 7:
            continue
        a = int(lo[along]) + int(rng.integers(2, length-3))
        b = min(a+int(rng.integers(2)), int(hi[along])-2)
        if side < 2:
            edge = int(lo[0] if side == 0 else hi[0])
            inside = edge + (1 if side == 0 else -1)
            old = [(edge, c) for c in range(a, b+1)]
            new = [(inside, c) for c in range(a-1, b+2)]
        else:
            edge = int(lo[1] if side == 2 else hi[1])
            inside = edge + (1 if side == 2 else -1)
            old = [(r, edge) for r in range(a, b+1)]
            new = [(r, inside) for r in range(a-1, b+2)]
        if any(grid[p] != f['color'] for p in old) or any(grid[p] != bg for p in new):
            continue
        candidate = grid.copy()
        for p in old:
            candidate[p] = bg
        for p in new:
            candidate[p] = f['color']
        mask = candidate == f['color']
        if f['color'] == facts['anchor']:
            mask |= candidate == marker
        try:
            tile, _ = _normalize(mask)
            if _key(tile['material']) != baseline[f['color']]:
                continue
        except ValueError:
            continue
        grid = candidate
        accepted += 1
        if accepted == 2:
            break
    # Restore independent margins without exceeding the canvas budget.
    extra = [int(rng.integers(min(3, 30-d)+1)) for d in grid.shape]
    top, left = [int(rng.integers(e+1)) for e in extra]
    return np.pad(grid, ((top, extra[0]-top), (left, extra[1]-left)), constant_values=bg)


def _make_pair(rng, count, rich, colors, background, marker, forced_rotation=None):
    # Bounded retries, only the accepted pair is retained. No execution traces.
    for attempt in range(80):
        try:
            root, children = _abstract_pair(rng, count, rich)
        except RuntimeError:
            continue
        nodes = [root] + children
        tiles = [_canonical(n) for n in nodes]
        side = tiles[0]['material'].shape[0]
        # No rectangle placement can put a square and a diagram with both
        # dimensions this large in the 28x28 usable interior. Reject early.
        if any(side+1+min(t['diagram'].shape) > 28 for t in tiles[1:]):
            continue
        pieces = []
        for i, (t, color) in enumerate(zip(tiles, colors)):
            p = np.where(t['material'], color, -1).astype(np.int8)
            if i == 0:
                p[0, 0] = marker
            pieces.append(p)
        side = pieces[0].shape[0]
        answers = _pack(pieces, side, marker)
        if len(answers) != 1:
            continue
        rotations = list(map(int, rng.integers(4, size=len(nodes))))
        rotations[0] = int(rng.integers(1, 4)) if forced_rotation is None else forced_rotation
        # Redraw topology at independently sampled metric scales. Compact mode
        # is a fit strategy, never a different rule or canned topology fallback.
        for redraw in range(6):
            drawings = [_draw(n, rng, compact=redraw >= 2) for n in nodes]
            if any(max(d.shape) > 26 for d in drawings):
                continue
            try:
                if any(_key(_normalize(d)[0]['material']) != _key(t['material']) for d, t in zip(drawings, tiles)):
                    continue
            except ValueError:
                continue
            inp = _scatter(drawings, colors, rotations, background, marker, rng)
            if inp is None:
                continue
            try:
                inp = _metric_variation(inp, rng)
                facts = _read_input(inp)
                # Rotation equivariance is checked after restoring the marker.
                expected = answers[0]
                actual = _pack(facts['pieces'], facts['side'], marker)
                if (len(actual) == 1 and np.array_equal(actual[0], expected)
                        and np.sum(inp != background) != expected.size):
                    return dict(input=inp, output=expected)
            except ValueError:
                continue
    raise RuntimeError('Bounded metric/scatter search exhausted')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name + ' is outside its legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    background, marker = palette[:2]
    puzzle = dict(train=[], test=[])
    seen = set()
    # Small and multi-cavity demonstrations both occur even with two examples.
    # Test holds out one anchor orientation and combines previously shown
    # parity nesting with independently drawn multi-cavity arrangements.
    held = int(rng.integers(4))
    train_rotations = [k for k in range(4) if k != held]
    for split, number in (('train', num_train), ('test', num_test)):
        for i in range(number):
            count = 1 if split == 'train' and i == 0 else 2
            rich = split == 'test' or i % 2 == 0
            if (split == 'test' or i >= 2) and rng.random() < .35:
                count = 3
            colors = list(map(int, rng.permutation(palette[2:])))[:count+1]
            rotation = held if split == 'test' else train_rotations[i % 3]
            for _ in range(8):
                pair = _make_pair(rng, count, rich, colors, background, marker, rotation)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not draw distinct inputs')
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, facts_by_split = set(), {}
    for split in ('train', 'test'):
        facts_by_split[split] = []
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair schema')
            for value in pair.values():
                a = np.asarray(value)
                _require(a.ndim == 2 and np.issubdtype(a.dtype, np.integer), 'Integer grids required')
                _require(all(1 <= d <= 30 for d in a.shape) and np.all((0 <= a) & (a <= 9)), 'Grid bounds')
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            _require(_key(inp) not in seen, 'Duplicate input')
            seen.add(_key(inp))
            f = _read_input(inp)
            _require(out.shape == (f['side'], f['side']) and out[0, 0] == f['marker'], 'Square/anchor invariant')
            _require(not np.any(out == f['background']), 'No empty square cells')
            _require(np.array_equal(solve(inp), out), 'Incorrect assembly')
            for color, p in zip(f['carriers'], f['pieces']):
                _require(np.sum(out == color) == np.sum(p == color), 'Normalized material conservation')
            _require(np.sum(out == f['marker']) == 1, 'One output marker')
            anchor = next(x for x in f['features'] if x['color'] == f['anchor'])
            _require(anchor['depth'] >= 2, 'Containment must affect material')
            marked_piece = next(p for p in f['pieces'] if np.any(p == f['marker']))
            _require(marked_piece.shape == out.shape, 'Outer stencil spans the assembled square')
            # A real hole in at least one insert must receive material from
            # another stencil. Filling every contour as a solid plate fails.
            _require(any(np.any(x['tile']['diagram'] & ~x['tile']['material']) or
                         np.any(~x['tile']['material'][1:-1, 1:-1])
                         for x in f['features'] if x['color'] != f['anchor']), 'Expose sockets')
            # Literal raw-pixel packing cannot reproduce normalized area.
            _require(np.sum(inp != f['background']) != out.size, 'Expose geometric normalization')
            facts_by_split[split].append(f)
    allfacts = facts_by_split['train'] + facts_by_split['test']
    _require(len({(f['background'], f['marker']) for f in allfacts}) == 1, 'Stable puzzle conventions')
    _require(len({len(f['pieces']) for f in facts_by_split['train']}) >= 2, 'Show changing stencil count')
    _require(max(max(x['depth'] for x in f['features']) for f in facts_by_split['train']) >= 3,
             'Show three alternating contour depths')
    _require(len({f['anchor_corner'] for f in facts_by_split['train']}) >= 2, 'Show orientation changes')
    _require(not ({f['anchor_corner'] for f in facts_by_split['train']} &
                  {f['anchor_corner'] for f in facts_by_split['test']}), 'Held-out anchor orientation')


def _puzzle_digest(puzzle):
    h = sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _geometry_metrics(inp, facts):
    changed, nonrect = False, 0
    blob_shapes = set()
    for f in facts['features']:
        mask = (inp == f['color']) | ((inp == facts['marker']) if f['color'] == facts['anchor'] else False)
        coords = np.argwhere(mask)
        lo, hi = coords.min(0), coords.max(0)+1
        changed |= not np.array_equal(mask[lo[0]:hi[0], lo[1]:hi[1]], f['tile']['diagram'])
        for obj in f['objects']:
            if not obj['contour']:
                blob_shapes.add(tuple(map(int, obj['hi']-obj['lo']+1)))
            else:
                local = np.zeros(obj['hi']-obj['lo']+1, bool)
                q = obj['coords']-obj['lo']
                local[q[:, 0], q[:, 1]] = True
                nonrect += not (local[0].all() and local[-1].all() and local[:, 0].all() and local[:, -1].all())
    return bool(changed), int(nonrect), blob_shapes


def self_test():
    """Bounded streaming tests: 42 host seeds, metamorphisms, negative checks."""
    # Equal diagram boxes, equal contour/blob counts, equal material area,
    # different pin positions and different colored answers. Counts alone fail.
    b = ('B',)
    line2 = ('G', 1, (b, b), (0, 0))
    line3 = ('G', 1, (b, b, b), (0, 0, 0))
    profiles = [('C', ('G', 0, (line3, b, b), (0, 0, 0))),
                ('C', ('G', 0, (line2, line2, b), (1, 0, 0)))]
    counter_answers, classes = [], []
    for i, child in enumerate(profiles):
        nodes = [('C', child), child]
        tiles = [_canonical(n) for n in nodes]
        _require(tiles[1]['material'].shape == (9, 9) and np.sum(~tiles[1]['material']) == 5,
                 'Counterexample dimensions and counts')
        classes.append(min(_key(np.rot90(tiles[1]['material'], k)) for k in range(4)))
        inp = _scatter([t['diagram'] for t in tiles], [2, 3], [1, 3], 0, 9, np.random.default_rng(550+i))
        _require(inp is not None, 'Counterexample fits')
        counter_answers.append(_key(solve(inp)))
    _require(classes[0] != classes[1] and counter_answers[0] != counter_answers[1], 'Arrangement matters')
    signatures, shapes, outputs, sizes, depths, blob_counts, palettes = (set() for _ in range(7))
    metric_pairs = bay_pairs = varied_blob_pairs = 0
    observed_blob_shapes = set()
    first = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        if first is None:
            first = puzzle
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                f = _read_input(inp)
                shapes.add(inp.shape)
                sizes.add(f['side'])
                depths.add(max(x['depth'] for x in f['features']))
                blob_counts.add(sum(x['blobs'] for x in f['features']))
                palettes.add((f['background'], f['marker']))
                changed, nonrect, actual_shapes = _geometry_metrics(inp, f)
                metric_pairs += changed
                bay_pairs += nonrect > 0
                varied_blob_pairs += len(actual_shapes) > 1
                observed_blob_shapes.update(actual_shapes)
                mapping = {}
                normalized = np.array([mapping.setdefault(int(c), len(mapping)) for c in out.flat], np.int8).reshape(out.shape)
                outputs.add(_key(normalized))
        # Recoloring and independent rigid reorientation of the whole INPUT
        # leave the anchored answer unchanged, except for the color map.
        pair = puzzle['test'][0]
        permutation = np.random.default_rng(seed+911).permutation(10)
        _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]), 'Palette equivariance')
        _require(np.array_equal(solve(np.rot90(pair['input'])), pair['output']), 'Input rotation invariance')
        # Stretch a non-marker row/column, while staying within ARC bounds.
        stretched = pair['input'].copy()
        marker = _read_input(stretched)['marker']
        for axis in (0, 1):
            if stretched.shape[axis] < 30:
                marker_index = int(np.argwhere(stretched == marker)[0, axis])
                repeat = np.ones(stretched.shape[axis], dtype=int)
                repeat[(marker_index+1) % len(repeat)] = 2
                stretched = np.repeat(stretched, repeat, axis=axis)
        _require(np.array_equal(solve(stretched), pair['output']), 'Metric stretch invariance')
    _require(len(signatures) == 42, 'Seeds must differ')
    _require(len(shapes) >= 15 and len(outputs) >= 50, 'Structural variety')
    _require(len(sizes) >= 3 and len(depths) >= 2 and len(blob_counts) >= 10, 'Topology/scale variety')
    _require(len(palettes) >= 20, 'Palette variety')
    _require(metric_pairs >= 150 and bay_pairs >= 30 and varied_blob_pairs >= 100,
             'Geometric variation must survive the fit fallback')
    _require(len(observed_blob_shapes) >= 3, 'Actual blob shapes must vary')
    _require(_puzzle_digest(first) == _puzzle_digest(generate_puzzle(0)), 'Determinism')
    # Every legal explicit count is exercised by verify.py. Keep self_test
    # representative rather than retaining the full Cartesian product.
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1700+a+b, a, b)
        _require(len(p['train']) == a and len(p['test']) == b, 'Count contract')
        validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.2), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad['test'][0]['output'][0, 0] = int(bad['test'][0]['output'][0, 1])
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Corrupted answer accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                normalized_outputs=len(outputs), sides=sorted(sizes), depths=sorted(depths),
                blob_count_variants=len(blob_counts), palettes=len(palettes),
                metric_changed_pairs=metric_pairs, nonrectangular_pairs=bay_pairs,
                varied_blob_pairs=varied_blob_pairs, blob_shapes=sorted(observed_blob_shapes))
