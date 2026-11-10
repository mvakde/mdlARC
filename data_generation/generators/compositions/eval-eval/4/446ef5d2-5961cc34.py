"""Tractor-beam jigsaws: first-hit selection and keyed docking interact online.

Only standard Python and NumPy are used. All solving conventions are visible in
one input. Generation keeps intended placements only to construct/check labels;
solve decodes fresh objects and uses no construction state.
"""
from __future__ import annotations

import copy
import json
import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= x <= 30 for x in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((g >= 0) & (g <= 9))
    return g.astype(np.int8, copy=True)


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack = [p]
        for r, c in stack:
            for dr, dc in _DIRS:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(set(stack))
    return result


def _add(p, q):
    return p[0] + q[0], p[1] + q[1]


def _absolute(obj, pos=None):
    pos = obj['pos'] if pos is None else pos
    return {(r + pos[0], c + pos[1]) for r, c in obj['cells']}


def _inside(cells, shape, margin=0):
    return all(margin <= r < shape[0] - margin and margin <= c < shape[1] - margin
               for r, c in cells)


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _separated(a, b):
    return not a & _halo(b)


def _placement_field(forbidden, shape):
    """A reusable summed-area table for one fixed search scene."""
    blocked = np.zeros(shape, dtype=np.int16)
    for r, c in forbidden:
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            blocked[r, c] = 1
    prefix = np.zeros((shape[0] + 1, shape[1] + 1), dtype=np.int16)
    prefix[1:, 1:] = blocked.cumsum(0).cumsum(1)
    return blocked, prefix


def _free_positions(obj, forbidden, shape, bounds=None):
    """Exact mask collisions, in the same row-major candidate order as before.

    Count a rectangle with four table slices, then subtract its missing cells.
    Bounds discard translations already excluded by the beam geometry.
    """
    h, w = obj['matrix'].shape
    r0, r1, c0, c1 = 1, shape[0] - h - 1, 1, shape[1] - w - 1
    if bounds is not None:
        r0, r1 = max(r0, bounds[0]), min(r1, bounds[1])
        c0, c1 = max(c0, bounds[2]), min(c1, bounds[3])
    if r0 > r1 or c0 > c1:
        return []
    blocked, prefix = (forbidden if isinstance(forbidden, tuple)
                       else _placement_field(forbidden, shape))
    nr, nc = r1 - r0 + 1, c1 - c0 + 1
    counts = (prefix[r0+h:r1+h+1, c0+w:c1+w+1]
              - prefix[r0:r1+1, c0+w:c1+w+1]
              - prefix[r0+h:r1+h+1, c0:c1+1]
              + prefix[r0:r1+1, c0:c1+1])
    for r, c in zip(*np.nonzero(obj['matrix'] < 0)):
        counts -= blocked[r0+r:r0+r+nr, c0+c:c0+c+nc]
    return [(int(r + r0), int(c + c0)) for r, c in np.argwhere(counts == 0)]


def _connected_cells(cells, width):
    """Exact bit flood fill; the unused column prevents horizontal wrap."""
    if not cells:
        return False
    stride = width + 1
    mask = sum(1 << (r * stride + c) for r, c in cells)
    reached = mask & -mask
    while True:
        expanded = (reached | (reached << 1) | (reached >> 1)
                    | (reached << stride) | (reached >> stride)) & mask
        if expanded == reached:
            return reached == mask
        reached = expanded


def _seam_cells(shape, side, start, length):
    h, w = shape
    if side == 0:
        return [(0, start + k) for k in range(length)]
    if side == 1:
        return [(start + k, w - 1) for k in range(length)]
    if side == 2:
        return [(h - 1, start + k) for k in range(length)]
    return [(start + k, 0) for k in range(length)]


def _marker(point, corner):
    dr = -1 if corner[0] == 'T' else 1
    dc = -1 if corner[1] == 'L' else 1
    r, c = point
    return {(r + dr, c), (r, c + dc), (r + dr, c + dc)}


def _decode(g):
    """Infer local color roles, the external L, pieces, sockets and live ports."""
    g = _grid(g)
    bg = int(g[0, 0])
    assert np.all(g[0] == bg) and np.all(g[-1] == bg)
    assert np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)
    assert np.count_nonzero(g == bg) > g.size // 2
    candidates = []
    for color in np.unique(g):
        if color == bg:
            continue
        for comp in _components(map(tuple, np.argwhere(g == color).tolist())):
            if len(comp) != 3:
                continue
            r, c = min(y for y, x in comp), min(x for y, x in comp)
            square = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if not comp < square:
                continue
            point = next(iter(square - comp))
            if int(g[point]) in (bg, int(color)):
                continue
            corner = {(1, 1): 'TL', (1, 0): 'TR', (0, 1): 'BL', (0, 0): 'BR'}[
                point[0] - r, point[1] - c]
            candidates.append((int(color), comp, point, corner))
    assert len(candidates) == 1, 'one external L must identify the anchor'
    mark, marker_cells, point, corner = candidates[0]
    payload = set(map(tuple, np.argwhere(g != bg).tolist())) - marker_cells
    pieces, anchor = [], None
    for comp in _components(payload):
        r, c = min(y for y, x in comp), min(x for y, x in comp)
        h, w = max(y for y, x in comp) - r + 1, max(x for y, x in comp) - c + 1
        assert h >= 4 and w >= 4
        local = {(y - r, x - c) for y, x in comp}
        matrix = np.full((h, w), -1, dtype=np.int8)
        for y, x in comp:
            matrix[y - r, x - c] = g[y, x]
        values, counts = np.unique(matrix[matrix >= 0], return_counts=True)
        body = int(values[np.argmax(counts)])
        assert int(counts.max()) > len(local) / 2 and body not in (bg, mark)
        dots = [p for p in local if int(matrix[p]) == mark]
        assert len(dots) == 1 and 0 < dots[0][0] < h - 1 and 0 < dots[0][1] < w - 1
        seams = []
        accounted = {dots[0]}
        for side in range(4):
            length = w if side % 2 == 0 else h
            run = []
            for k in range(length + 1):
                cell = _seam_cells((h, w), side, k, 1)[0] if k < length else None
                color = int(matrix[cell]) if cell is not None else -1
                if color >= 0 and color not in (body, mark):
                    assert 0 < k < length - 1, 'keys do not use corners'
                    run.append(cell)
                elif run:
                    key = tuple(int(matrix[p]) for p in run)
                    assert 2 <= len(run) <= 4 and len(set(key)) == 2
                    assert key != key[::-1], 'ordered keys are asymmetric'
                    seams.append({'side': side, 'coords': tuple(run), 'key': key})
                    accounted.update(run)
                    run = []
        special = {p for p in local if int(matrix[p]) != body}
        assert special == accounted
        live = [s for s in seams if dots[0] in {
            _add(p, _DIRS[(s['side'] + 2) % 4]) for p in s['coords']}]
        assert len(live) == 1, 'one dot activates exactly one edge key'
        obj = {'matrix': matrix, 'cells': local, 'pos': (r, c),
               'body': body, 'dot': dots[0], 'seams': seams, 'out': live[0]}
        if point in comp:
            anchor = len(pieces)
            expected = (r + (h - 1 if corner[0] == 'B' else 0),
                        c + (w - 1 if corner[1] == 'R' else 0))
            assert point == expected
        pieces.append(obj)
    assert anchor is not None and len(pieces) >= 2
    assert all(_separated(_absolute(pieces[i]), _absolute(pieces[j]))
               for i in range(len(pieces)) for j in range(i))
    assert len(pieces[anchor]['seams']) == 1
    assert all(len(p['seams']) in (2, 3) for i, p in enumerate(pieces) if i != anchor)
    return bg, mark, marker_cells, point, corner, pieces, anchor


def _hit_project(objects, positions, current, shape):
    """First occupied cell over the emitted strip; partial-width hits are valid."""
    out = objects[current]['out']
    side = out['side']
    dr, dc = _DIRS[side]
    ports = [_add(p, positions[current]) for p in out['coords']]
    lanes = {r if dc else c for r, c in ports}
    front = ports[0][0] if dr else ports[0][1]
    distance = (front + 1 if side in (0, 3) else
                shape[0 if dr else 1] - front)
    hits = set()
    for i, obj in enumerate(objects):
        if i == current:
            continue
        for r, c in _absolute(obj, positions[i]):
            if (r if dc else c) not in lanes:
                continue
            d = (r - ports[0][0]) * dr + (c - ports[0][1]) * dc
            if 0 < d < distance:
                distance, hits = d, {i}
            elif d == distance:
                hits.add(i)
    assert len(hits) <= 1, 'simultaneous contact with different fragments'
    strip = {_add(p, (k * dr, k * dc)) for p in ports for k in range(1, distance)}
    return (next(iter(hits)) if hits else None), strip, distance


def _hit_front(objects, positions, current, shape):
    """Construction check uses an advancing front, independently of projection."""
    out = objects[current]['out']
    direction = _DIRS[out['side']]
    front = [_add(_add(p, positions[current]), direction) for p in out['coords']]
    owner = {}
    for i, obj in enumerate(objects):
        if i != current:
            for p in _absolute(obj, positions[i]):
                assert p not in owner
                owner[p] = i
    strip = set()
    for distance in range(1, 31):
        if not _inside(front, shape):
            return None, strip, distance
        hits = {owner[p] for p in front if p in owner}
        assert len(hits) <= 1
        if hits:
            return next(iter(hits)), strip, distance
        strip.update(front)
        front = [_add(p, direction) for p in front]
    raise AssertionError('a straight beam must leave a <=30 canvas')


def _matching_sockets(source, target):
    key, side = source['out']['key'], source['out']['side']
    return [s for s in target['seams'] if s is not target['out']
            and s['side'] == (side + 2) % 4 and s['key'] == key]


def _dock_position(source, source_pos, socket):
    # Natural top-to-bottom / left-to-right sequence order is never reversed.
    port = _add(source['out']['coords'][0], source_pos)
    point = _add(port, _DIRS[source['out']['side']])
    return point[0] - socket['coords'][0][0], point[1] - socket['coords'][0][1]


def _walk(g, details=False):
    bg, mark, marker_cells, point, corner, objects, root = _decode(g)
    positions = [o['pos'] for o in objects]
    active = [root]
    history, changes, transverse, dual, partial, registration = [], [], [], [], [], []
    initial_candidates = sum(bool(_matching_sockets(objects[root], o))
                             for i, o in enumerate(objects) if i != root)
    for _ in range(len(objects)):
        current = active[-1]
        hit, strip, distance = _hit_project(objects, positions, current, g.shape)
        history.append(strip)
        if hit is None:
            break
        assert hit not in active, 'live beam may not re-enter the docked assembly'
        source, target = objects[current], objects[hit]
        sockets = _matching_sockets(source, target)
        assert len(sockets) == 1, 'the struck fragment needs one matching facing socket'
        socket = sockets[0]
        old = positions[hit]
        new = _dock_position(source, positions[current], socket)
        cells = _absolute(target, new)
        assert _inside(cells, g.shape, 1), 'docking must fit the canvas interior'
        assert not cells & marker_cells
        assert all(not cells & _absolute(o, positions[i]) for i, o in enumerate(objects) if i != hit)
        out = source['out']
        front = {_add(_add(p, positions[current]),
                      (distance * _DIRS[out['side']][0], distance * _DIRS[out['side']][1]))
                 for p in out['coords']}
        partial.append(not front <= _absolute(target, old))
        dual.append(sum(s['side'] == socket['side'] and s is not target['out']
                        for s in target['seams']) > 1)
        changes.append((new[0] - old[0], new[1] - old[1]))
        transverse.append((new[1] - old[1]) if out['side'] % 2 == 0 else (new[0] - old[0]))
        axis = 1 if out['side'] % 2 == 0 else 0
        registration.append(new[axis] != positions[current][axis])
        positions[hit] = new
        active.append(hit)
    else:
        # At most one activation per object; query the last emitter once more.
        hit, strip, _ = _hit_project(objects, positions, active[-1], g.shape)
        assert hit is None
        history.append(strip)
    output = np.full(g.shape, bg, dtype=np.int8)
    for i in active:
        for p in objects[i]['cells']:
            output[_add(p, positions[i])] = objects[i]['matrix'][p]
    if not details:
        return output
    static, cur = [root], root
    original = [o['pos'] for o in objects]
    for _ in objects:
        try:
            hit, _, _ = _hit_project(objects, original, cur, g.shape)
        except AssertionError:
            static.append(-1)  # The no-movement alternative can itself be ambiguous.
            break
        if hit is None or hit in static:
            break
        static.append(hit)
        cur = hit
    root_out = objects[root]['out']
    dr, dc = _DIRS[root_out['side']]
    root_ports = [_add(p, original[root]) for p in root_out['coords']]
    lanes = {r if dc else c for r, c in root_ports}
    behind_decoy = False
    if len(active) > 1:
        far = max(r * dr + c * dc for r, c in _absolute(objects[active[1]], original[active[1]]))
        behind_decoy = any(i not in active and any(
            (r if dc else c) in lanes and r * dr + c * dc > far
            for r, c in _absolute(obj, original[i])) for i, obj in enumerate(objects))
    facts = {'shape': tuple(g.shape), 'corner': corner, 'root': root,
             'objects': objects, 'positions': positions, 'active': active,
             'moves': len(active) - 1, 'decoys': len(objects) - len(active),
             'changes': changes, 'transverse': transverse, 'dual': dual,
             'partial': partial, 'offset_registration': registration,
             'behind_decoy': behind_decoy, 'dynamic_selection': active != static,
             'compatible_first': initial_candidates, 'history': history,
             'directions': [objects[i]['out']['side'] for i in active],
             'widths': [len(objects[i]['out']['key']) for i in active],
             'stepped': sum(len(o['cells']) != o['matrix'].size for o in objects),
             'palette': (bg, mark, tuple(sorted({o['body'] for o in objects}))),
             'marker': marker_cells, 'bg': bg, 'mark': mark}
    return output, facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode, select, dock, and re-emit using only this input."""
    return _walk(_grid(input_grid))


def _key(rng, colors, length):
    for _ in range(128):
        values = tuple(colors[int(x)] for x in rng.integers(len(colors), size=length))
        if len(set(values)) == 2 and values != values[::-1]:
            return values
    raise RuntimeError('bounded asymmetric-key sampling failed')


def _make_piece(rng, roles, out_side, out_key, in_side=None, in_key=None,
                dual=False, root=False, cuts=True):
    bg, mark, key_colors, bodies = roles
    h, w = map(int, rng.integers(4, 8, size=2))
    for side, length in [(out_side, len(out_key))] + ([] if root else [(in_side, len(in_key))]):
        minimum = (2 * length + 3) if dual and side == in_side else length + 2
        if side % 2:
            h = max(h, minimum)
        else:
            w = max(w, minimum)
    body = bodies[int(rng.integers(len(bodies)))]
    matrix = np.full((h, w), body, dtype=np.int8)
    seams = []
    if not root:
        length = h if in_side % 2 else w
        if dual:
            starts = [1, length - 1 - len(in_key)]
            if rng.integers(2):
                starts.reverse()
        else:
            starts = [int(rng.integers(1, length - len(in_key)))]
        for j, start in enumerate(starts):
            key = in_key if j == 0 else in_key[::-1]
            coords = tuple(_seam_cells((h, w), in_side, start, len(key)))
            seams.append({'side': in_side, 'coords': coords, 'key': key})
    length = h if out_side % 2 else w
    start = int(rng.integers(1, length - len(out_key)))
    coords = tuple(_seam_cells((h, w), out_side, start, len(out_key)))
    out = {'side': out_side, 'coords': coords, 'key': out_key}
    seams.append(out)
    for s in seams:
        for p, color in zip(s['coords'], s['key']):
            matrix[p] = color
    dot = _add(coords[len(coords) // 2], _DIRS[(out_side + 2) % 4])
    # Reject ambiguous inner dot adjacency and perpendicular seam interference.
    if matrix[dot] != body or sum(dot in {_add(p, _DIRS[(s['side'] + 2) % 4])
                                         for p in s['coords']} for s in seams) != 1:
        return None
    matrix[dot] = mark
    special = {dot} | {p for seam in seams for p in seam['coords']}
    cells = {(r, c) for r in range(h) for c in range(w)}
    if cuts:
        protected = {dot} | {p for s in seams for p in s['coords']}
        protected |= {_add(p, _DIRS[(s['side'] + 2) % 4]) for s in seams for p in s['coords']}
        # Corner clips/short steps leave straight key faces intact.
        for k in rng.permutation(4)[:int(rng.integers(3))]:
            y, x = (0 if k < 2 else h - 1), (0 if k % 2 == 0 else w - 1)
            removed = {(y, x)}
            if rng.random() < .4:
                removed.add((y, x + (1 if x == 0 else -1)))
            if not root and not removed & protected:
                for p in removed:
                    matrix[p] = -1
                    cells.discard(p)
    body_cells = cells - special
    if len(body_cells) <= len(cells) / 2 or not _connected_cells(body_cells, w):
        return None
    extent = (min(r for r, c in cells), max(r for r, c in cells),
              min(c for r, c in cells), max(c for r, c in cells))
    return {'matrix': matrix, 'cells': cells, 'pos': (0, 0), 'out': out,
            'seams': seams, 'dot': dot, 'body': body, 'extent': extent}


def _render(objects, positions, shape, bg):
    g = np.full(shape, bg, dtype=np.int8)
    for obj, pos in zip(objects, positions):
        for p in obj['cells']:
            q = _add(p, pos)
            assert int(g[q]) == bg
            g[q] = obj['matrix'][p]
    return g


def _layout(rng, roles, moves, dual, straight, widths, decoys, behind):
    """Bounded recursive geometry search keeps the requested move/decoy counts."""
    size_min = 22 if behind else 18 if moves <= 2 else 23 if moves == 3 else 26
    shape = tuple(map(int, rng.integers(size_min, 31, size=2)))
    if moves <= 1 and rng.random() < .5:
        shape = (int(rng.integers(12, 19)), int(rng.integers(22, 31)))
        if rng.integers(2):
            shape = shape[::-1]
    direction = int(rng.integers(4))
    keys = [_key(rng, roles[2], int(widths[i % len(widths)])) for i in range(moves + 1)]
    root = _make_piece(rng, roles, direction, keys[0], root=True)
    if root is None:
        return None
    h, w = root['matrix'].shape
    if behind:
        # Reserve room beyond the first target for the required occluded decoy.
        pos = [int(rng.integers(max(2, (shape[i] - size) // 2 - 3),
                                 min(shape[i] - size - 1, (shape[i] - size) // 2 + 4)))
               for i, size in enumerate((h, w))]
        axis = 1 if direction % 2 else 0
        inset = int(rng.integers(2, 6))
        pos[axis] = inset if direction in (1, 2) else shape[axis] - (h if axis == 0 else w) - inset
        root['pos'] = tuple(pos)
    else:
        root['pos'] = (int(rng.integers(2, shape[0] - h - 1)),
                       int(rng.integers(2, shape[1] - w - 1)))
    corner = ('TL', 'TR', 'BL', 'BR')[int(rng.integers(4))]
    point = _add(root['pos'], (h - 1 if corner[0] == 'B' else 0,
                              w - 1 if corner[1] == 'R' else 0))
    markers = _marker(point, corner)
    budget = [240]

    def grow(objects, originals, finals, history, index):
        if index == moves:
            try:
                hit, strip, _ = _hit_front(objects, finals, index, shape)
            except AssertionError:
                return None
            if hit is not None:
                return None
            return objects, originals, finals, history + [strip]
        source = objects[-1]
        side = source['out']['side']
        dr, dc = _DIRS[side]
        transverse_dir = (0, 1) if dr else (1, 0)
        all_final = set().union(*(_absolute(o, p) for o, p in zip(objects, finals)))
        final_blocked = all_final | markers
        all_original = set().union(*(_absolute(o, p) for o, p in zip(objects, originals)))
        old_rays = set().union(*history) if history else set()
        field = _placement_field(_halo(all_original | markers) | all_final | old_rays, shape)
        for _ in range(35):
            budget[0] -= 1
            if budget[0] < 0:
                return None
            turn = 0 if straight and index == 0 else int(rng.integers(-1, 2))
            obj = _make_piece(rng, roles, (side + turn) % 4, keys[index + 1],
                              (side + 2) % 4, source['out']['key'],
                              dual=dual and index == 0)
            if obj is None:
                continue
            socket = _matching_sockets(source, obj)[0]
            final = _dock_position(source, finals[-1], socket)
            oh, ow = obj['matrix'].shape
            min_r, max_r, min_c, max_c = obj['extent']
            if not (1 - min_r <= final[0] <= shape[0] - max_r - 2
                    and 1 - min_c <= final[1] <= shape[1] - max_c - 2):
                continue
            if _absolute(obj, final) & final_blocked:
                continue
            cross = ow if dr else oh
            if dr:
                r0, r1 = sorted((final[0] + dr, final[0] + 12 * dr))
                c0, c1 = final[1] - cross + 2, final[1] + cross - 2
            else:
                r0, r1 = final[0] - cross + 2, final[0] + cross - 2
                c0, c1 = sorted((final[1] + dc, final[1] + 12 * dc))
            if behind and index == 0:
                if dr == 1:
                    r1 = min(r1, shape[0] - max_r - 9)
                elif dr == -1:
                    r0 = max(r0, 8 - min_r)
                elif dc == 1:
                    c1 = min(c1, shape[1] - max_c - 9)
                else:
                    c0 = max(c0, 8 - min_c)
            candidates = _free_positions(obj, field, shape, (r0, r1, c0, c1))
            if not candidates:
                continue
            for k in rng.permutation(len(candidates))[:14]:
                pos = candidates[int(k)]
                try:
                    hit, strip, _ = _hit_front(objects + [obj], finals + [pos], index, shape)
                except AssertionError:
                    continue
                if hit != index + 1:
                    continue
                obj['pos'] = pos
                result = grow(objects + [obj], originals + [pos], finals + [final],
                              history + [strip], index + 1)
                if result is not None:
                    return result
        return None

    result = grow([root], [root['pos']], [root['pos']], [], 0)
    if result is None:
        return None
    objects, originals, finals, history = result
    original_used = set().union(*(_absolute(o, p) for o, p in zip(objects, originals)))
    final_used = set().union(*(_absolute(o, p) for o, p in zip(objects, finals)))
    rays = set().union(*history)
    for j in range(decoys):
        field = _placement_field(_halo(original_used | markers) | final_used | rays, shape)
        found = False
        for _ in range(16):
            # Compatible keys are deliberately repeated on off-route fragments.
            in_side = (direction + 2) % 4
            out_side = (direction + int(rng.integers(-1, 2))) % 4
            obj = _make_piece(rng, roles, out_side, _key(rng, roles[2], widths[int(rng.integers(len(widths)))]),
                              in_side, keys[0])
            if obj is None:
                continue
            positions = _free_positions(obj, field, shape)
            if behind and j == 0:
                ports = [_add(p, originals[0]) for p in root['out']['coords']]
                lanes = {y if direction % 2 else x for y, x in ports}
                first = _absolute(objects[1], originals[1])
                far = max(y * _DIRS[direction][0] + x * _DIRS[direction][1] for y, x in first)
                # Project once per local lane, rather than translating every mask
                # for every possible origin. This is the same existential test.
                extents = {}
                dr, dc = _DIRS[direction]
                for y, x in obj['cells']:
                    cross, along = (y, x * dc) if dc else (x, y * dr)
                    extents[cross] = max(extents.get(cross, -100), along)
                positions = [pos for pos in positions if any(
                    lane - (pos[0] if dc else pos[1]) in extents and
                    pos[0] * dr + pos[1] * dc +
                    extents[lane - (pos[0] if dc else pos[1])] > far
                    for lane in lanes)]
            if positions:
                obj['pos'] = positions[int(rng.integers(len(positions)))]
                objects.append(obj)
                originals.append(obj['pos'])
                original_used |= _absolute(obj)
                found = True
                break
        if not found:
            return None
    bg, mark = roles[:2]
    inp = _render(objects, originals, shape, bg)
    for p in markers:
        inp[p] = mark
    out = _render(objects[:moves + 1], finals, shape, bg)
    if np.count_nonzero(inp == bg) <= inp.size // 2:
        return None
    return {'input': inp, 'output': out}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError('counts require 2..10 training, 1..4 tests, and train > test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    varied_palette = bool(rng.integers(2))
    palette = list(map(int, rng.permutation(10)))
    body_count = int(rng.integers(1, 4))
    max_train_moves = int(rng.choice([2, 3], p=[.6, .4]))
    puzzle = {'train': [], 'test': []}
    seen, train_facts = set(), []
    train_plan = [(2, True, False, 1, True),
                  (max_train_moves, False, True, int(rng.integers(1, 3)), False)]
    for _ in range(num_train - 2):
        train_plan.append((int(rng.integers(1, max_train_moves + 1)), bool(rng.random() < .3),
                           False, int(rng.integers(0, 4)), False))
    # The first two construction obligations are shuffled after construction.
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if varied_palette:
                palette = list(map(int, rng.permutation(10)))
            roles = (palette[0], palette[1], tuple(palette[2:4]), tuple(palette[4:4 + body_count]))
            if split == 'train':
                moves, dual, straight, decoys, behind = train_plan[index]
            else:
                moves, dual, straight, decoys, behind = (
                    max_train_moves + 1, bool(rng.random() < .5), False,
                    int(rng.integers(1, 4)), False)
            widths = list(map(int, rng.integers(2, 4, size=moves + 1)))
            if split == 'train' and index < 2:
                # Establish both widths, then allow repeated or changed widths freely.
                widths[:2] = list(map(int, rng.permutation([2, 3])))
            for attempt in range(160):
                pair = _layout(rng, roles, moves, dual, straight, widths, decoys, behind)
                if pair is None:
                    continue
                try:
                    predicted, facts = _walk(pair['input'], True)
                except AssertionError:
                    continue
                if not np.array_equal(predicted, pair['output']):
                    continue
                if split == 'train' and index == 0 and not (
                        facts['dynamic_selection'] and any(facts['transverse']) and any(facts['dual'])
                        and len(set(facts['directions'])) > 1
                        and any(facts['offset_registration'])
                        and any(len(facts['objects'][i]['cells']) != facts['objects'][i]['matrix'].size
                                for i in facts['active'])):
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key in seen:
                    continue
                break
            else:
                raise RuntimeError(f'bounded geometry search failed: {moves} moves, {decoys} decoys')
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train':
                train_facts.append(facts)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible decoding, online geometry, conservation, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            predicted, f = _walk(inp, True)
            assert np.array_equal(predicted, out), 'output differs from input-derived docking'
            objects, positions, active = f['objects'], f['positions'], f['active']
            assert positions[f['root']] == objects[f['root']]['pos']
            assert all(out[p] == f['bg'] for p in f['marker'])
            used = set()
            histogram = np.zeros(10, dtype=int)
            for i in active:
                obj = objects[i]
                cells = _absolute(obj, positions[i])
                assert not used & cells
                used |= cells
                for p in obj['cells']:
                    assert int(out[_add(p, positions[i])]) == int(obj['matrix'][p])
                    histogram[int(obj['matrix'][p])] += 1
            assert set(map(tuple, np.argwhere(out != f['bg']).tolist())) == used
            assert len(_components(used)) == 1, 'all surviving pieces dock into one connected assembly'
            assert all(np.count_nonzero(out == c) == histogram[c] for c in range(10) if c != f['bg'])
            # Independent front traversal agrees at every actual intermediate scene.
            current_positions = [o['pos'] for o in objects]
            for k, current in enumerate(active):
                a = _hit_front(objects, current_positions, current, inp.shape)
                b = _hit_project(objects, current_positions, current, inp.shape)
                assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
                assert a[0] == (active[k + 1] if k + 1 < len(active) else None)
                if k + 1 < len(active):
                    nxt = active[k + 1]
                    current_positions[nxt] = positions[nxt]
            records[split].append(f)
    train, test = records['train'], records['test']
    assert any(f['moves'] >= 2 and f['dynamic_selection'] and any(f['transverse'])
               and any(f['dual']) and f['compatible_first'] >= 2
               and f['behind_decoy'] and any(f['offset_registration']) for f in train)
    assert any(f['decoys'] for f in train)
    assert any(a == b for f in train for a, b in zip(f['directions'], f['directions'][1:]))
    assert any(a != b for f in train for a, b in zip(f['directions'], f['directions'][1:]))
    widths = {w for f in train for w in f['widths']}
    assert {2, 3} <= widths
    assert all(set(f['widths']) <= widths for f in test)
    assert min(f['moves'] for f in test) > max(f['moves'] for f in train)
    return None


def _fingerprint(puzzle):
    return tuple((np.asarray(pair[k]).shape, np.asarray(pair[k], dtype=np.int8).tobytes())
                 for split in ('train', 'test') for pair in puzzle[split] for k in ('input', 'output'))


def self_test():
    """Bounded 42-seed structural and semantic checks; returns compact metrics."""
    puzzles, records = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                output, facts = _walk(pair['input'], True)
                assert np.array_equal(output, pair['output'])
                records.append((split, facts))
    all_facts = [f for _, f in records]
    structures = set()
    for f in all_facts:
        root_pos = f['positions'][f['root']]
        structures.add(tuple((f['objects'][i]['matrix'].shape,
                              tuple(sorted(f['objects'][i]['cells'])),
                              (f['positions'][i][0] - root_pos[0],
                               f['positions'][i][1] - root_pos[1]),
                              f['objects'][i]['out']['side']) for i in f['active']))
    metrics = {
        'seeds': 42, 'pairs': len(records),
        'distinct_puzzles': len({_fingerprint(p) for p in puzzles}),
        'canvas_shapes': len({f['shape'] for f in all_facts}),
        'assembly_structures': len(structures),
        'piece_dimensions': len({o['matrix'].shape for f in all_facts for o in f['objects']}),
        'move_counts': sorted({f['moves'] for f in all_facts}),
        'decoy_counts': sorted({f['decoys'] for f in all_facts}),
        'corners': sorted({f['corner'] for f in all_facts}),
        'directions': sorted({d for f in all_facts for d in f['directions']}),
        'dual_socket_pairs': sum(any(f['dual']) for f in all_facts),
        'feedback_pairs': sum(f['dynamic_selection'] for f in all_facts),
        'partial_hit_pairs': sum(any(f['partial']) for f in all_facts),
        'behind_decoy_pairs': sum(f['behind_decoy'] for f in all_facts),
        'stepped_pairs': sum(f['stepped'] > 0 for f in all_facts),
        'palettes': len({str(f['palette']) for f in all_facts}),
    }
    assert metrics['distinct_puzzles'] == 42
    assert metrics['assembly_structures'] >= 100
    assert metrics['canvas_shapes'] >= 40 and metrics['piece_dimensions'] >= 15
    assert set(metrics['move_counts']) >= {1, 2, 3, 4}
    assert set(metrics['decoy_counts']) >= {0, 1, 2, 3}
    assert len(metrics['corners']) == 4 and metrics['directions'] == list(range(4))
    assert metrics['dual_socket_pairs'] >= 42 and metrics['feedback_pairs'] >= 42
    assert metrics['behind_decoy_pairs'] >= 42
    assert metrics['partial_hit_pairs'] >= 5 and metrics['stepped_pairs'] >= 50
    assert metrics['palettes'] >= 20
    for split in ('train', 'test'):
        rows = [f for s, f in records if s == split]
        assert any(a == b for f in rows for a, b in zip(f['widths'], f['widths'][1:]))
        assert any(a != b for f in rows for a, b in zip(f['widths'], f['widths'][1:]))
    assert _fingerprint(generate_puzzle(170, 2, 1)) == _fingerprint(generate_puzzle(170, 2, 1))
    # Corrupt preserved pixels and anchoring; duplicate inputs must also fail.
    for kind in ('pixel', 'shift', 'duplicate'):
        damaged = copy.deepcopy(puzzles[0])
        if kind == 'duplicate':
            damaged['train'][1] = copy.deepcopy(damaged['train'][0])
        elif kind == 'shift':
            damaged['test'][0]['output'] = np.roll(damaged['test'][0]['output'], 1, axis=0)
        else:
            pair = damaged['test'][0]
            bg = int(pair['input'][0, 0])
            p = tuple(np.argwhere(pair['output'] != bg)[0])
            pair['output'][p] = bg
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('corrupted puzzle accepted')
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.2)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return metrics


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
