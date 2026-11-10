"""Boundary-anchored wallpaper: simultaneous, component-local texture transfers.

Only standard-library modules and NumPy are used. The public solver reads one
input, with fixed geometric conventions and no generation state or training data.
"""
from __future__ import annotations

import json
import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(a):
    """Original four-connected monochromatic components, independent of color IDs."""
    labels = np.full(a.shape, -1, dtype=np.int16)
    colors, cells = [], []
    h, w = a.shape
    for r, c in np.ndindex(a.shape):
        if labels[r, c] >= 0:
            continue
        index, color = len(colors), int(a[r, c])
        colors.append(color)
        points, stack = [], [(r, c)]
        labels[r, c] = index
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy, dx in DIRECTIONS:
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w and labels[yy, xx] < 0
                        and a[yy, xx] == color):
                    labels[yy, xx] = index
                    stack.append((yy, xx))
        cells.append(points)
    return labels, colors, cells


def _connected(mask):
    p = np.argwhere(mask)
    if not len(p):
        return False
    remaining = set(map(tuple, p))
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRECTIONS:
            q = r + dr, c + dc
            if q in remaining:
                remaining.remove(q)
                stack.append(q)
    return not remaining


def _box(points):
    p = np.asarray(points)
    return (int(p[:, 0].min()), int(p[:, 0].max()) + 1,
            int(p[:, 1].min()), int(p[:, 1].max()) + 1)


def _geometry(center, direction, shape):
    r, c = center
    dr, dc = direction
    vr, vc = -dc, dr
    arms = ((r + vr, c + vc), (r - vr, c - vc))
    tip, entry = (r + dr, c + dc), (r + 2 * dr, c + 2 * dc)
    sides = ((tip[0] + vr, tip[1] + vc), (tip[0] - vr, tip[1] - vc))
    write = frozenset((center, tip, *arms))
    context = frozenset((entry, *sides))
    h, w = shape
    if not all(0 <= y < h and 0 <= x < w for y, x in write | context):
        return None
    return dict(center=center, d=direction, arms=arms, tip=tip, q=entry,
                sides=sides, write=write, context=context)


def _parse(input_grid):
    """Recover the legend, restore arrows, then segment the original territories."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _need(all(int(grid[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))),
          "canvas corners must show the background")
    frames = []
    for color in np.unique(grid):
        if color == bg:
            continue
        points = np.argwhere(grid == color)
        r, e, c, f = _box(points)
        h, w = e - r, f - c
        if (h >= 8 and w >= 8 and len(points) == 2 * h + 2 * w - 4
                and np.all(grid[r, c:f] == color) and np.all(grid[e-1, c:f] == color)
                and np.all(grid[r:e, c] == color) and np.all(grid[r:e, f-1] == color)):
            frames.append(((r, e, c, f), int(color)))
    _need(len(frames) == 1, "need one exclusive main frame")
    frame, frame_color = frames[0]
    r, e, c, f = frame
    exterior = np.ones(grid.shape, dtype=bool)
    exterior[r:e, c:f] = False
    exterior &= grid != bg
    # Group all exterior nonbackground pixels, including touching key colors.
    ext_ids, _, ext_cells = _components(exterior.astype(np.int8))
    del ext_ids
    keys, key_mask, key_boxes = {}, np.zeros(grid.shape, bool), []
    for points in ext_cells:
        if not exterior[points[0]]:
            continue
        values = {int(grid[p]) for p in points}
        if len(values) == 1:
            continue
        a, b, x, y = _box(points)
        _need((b-a, y-x) == (2, 2) and len(points) == 4, "unknown legend key")
        tile = grid[a:b, x:y]
        src, paint = int(tile[0, 0]), int(tile[1, 1])
        _need(src != paint and tile[0, 1] == src and tile[1, 0] == src,
              "key convention is [[source,source],[source,paint]]")
        _need(src not in keys, "duplicate source key")
        keys[src] = paint
        key_mask[a:b, x:y] = True
        key_boxes.append((a, b, x, y))
    _need(2 <= len(keys) <= 6, "need two to six legend entries")
    remaining = exterior & ~key_mask
    _need(set(map(int, np.unique(grid[remaining]))) == set(keys), "unexplained exterior color")
    tiles, tile_boxes = {}, {}
    for source in keys:
        a, b, x, y = _box(np.argwhere(remaining & (grid == source)))
        _need(2 <= b-a <= 4 and 2 <= y-x <= 4, "invalid tile period")
        patch = grid[a:b, x:y]
        _need(bool(np.all(np.isin(patch, [source, bg]))), "overlapping stencil clues")
        mask = patch == source
        _need(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any(),
              "stencil bounds are not visible")
        tiles[source], tile_boxes[source] = mask.copy(), (a, b, x, y)
    boxes = list(tile_boxes.values()) + key_boxes
    for i, (a, b, x, y) in enumerate(boxes):
        _need(b < r or a > e or y < c or x > f, "clue must have a gap from the board")
        for aa, bb, xx, yy in boxes[i+1:]:
            _need(b < aa or bb < a or y < xx or yy < x, "clues require a blank-cell gap")
    board = grid[r+1:e-1, c+1:f-1]
    extra = set(map(int, np.unique(board))) - set(keys)
    _need(len(extra) == 1, "arrow ink must be the only non-source board color")
    ink = extra.pop()
    _need(ink not in (bg, frame_color, *keys.values()), "arrow ink role overlaps the legend")
    arrows = []
    for center in np.ndindex(board.shape):
        for direction in DIRECTIONS:
            g = _geometry(center, direction, board.shape)
            if g is None or not all(board[p] == ink for p in (*g['arms'], g['tip'])):
                continue
            if board[center] != ink and int(board[center]) not in keys:
                continue
            a, b = g['sides']
            host = int(board[a])
            if host not in keys or board[b] != host:
                continue
            if int(board[g['q']]) not in keys or board[g['q']] == host:
                continue
            selector = host if board[center] == ink else int(board[center])
            arrows.append(dict(g, host=host, selector=selector,
                               explicit=bool(board[center] != ink)))
    _need(bool(arrows), "no recognizable T arrows")
    writes, contexts = set(), set()
    clean = board.copy()
    covered = set()
    for a in arrows:
        _need(not (a['write'] & (writes | contexts)) and not (a['context'] & writes),
              "overlapping or occluded arrows")
        writes.update(a['write'])
        contexts.update(a['context'])
        covered.update(p for p in a['write'] if board[p] == ink)
        for p in a['write']:
            clean[p] = a['host']
    _need(covered == set(map(tuple, np.argwhere(board == ink))), "unexplained arrow ink")
    _need(set(map(int, np.unique(clean))) <= set(keys), "invalid underlying fill")
    _need(set(keys) == set(map(int, np.unique(clean))) | {a['selector'] for a in arrows},
          "legend entry has no territory or selector role")
    labels, colors, cells = _components(clean)
    _need(2 <= len(colors) <= 7, "invalid territory count")
    assignments = {}
    for a in arrows:
        src, target = int(labels[a['center']]), int(labels[a['q']])
        _need(src != target and all(labels[p] == src for p in a['write']), "arrow crosses its host")
        a.update(source=src, target=target)
        paint = keys[a['selector']]
        _need(target not in assignments or assignments[target] == paint, "conflicting incoming paints")
        assignments[target] = paint
    return dict(grid=grid, bg=bg, frame=frame, frame_color=frame_color, ink=ink,
                mapping=keys, tiles=tiles, tile_boxes=tile_boxes, key_boxes=key_boxes,
                board=board, clean=clean, labels=labels, colors=colors, cells=cells,
                arrows=arrows, assignments=assignments)


def _render(info, alternative=None):
    """Vectorized input-derived rendering; alternatives are only ablation probes."""
    clean, labels, arrows = info['clean'], info['labels'], info['arrows']
    result = clean.copy()
    rr, cc = np.indices(clean.shape)
    unions, paints, targets = {}, {}, {}
    if alternative == 'last':
        last = {a['target']: a for a in arrows}
        arrows = list(last.values())
    for a in arrows:
        target = a['target']
        selector = (info['colors'][target] if alternative == 'target_selector' else
                    a['host'] if alternative == 'no_explicit' else a['selector'])
        if alternative == 'relay' and not a['explicit']:
            # Wrong rule: a receiving host forwards the incoming texture instead
            # of selecting from its original fill. This has no ordering dependency.
            incoming = [b for b in arrows if b['target'] == a['source']]
            if incoming:
                selector = incoming[0]['selector']
        tile = info['tiles'][selector]
        dr, dc = (1, 0) if alternative == 'no_orientation' else a['d']
        qr, qc = a['q']
        if alternative == 'bbox':
            qr, _, qc, _ = _box(info['cells'][target])
        forward_raw = (rr-qr) * dr + (cc-qc) * dc
        sideways_raw = -(rr-qr) * dc + (cc-qc) * dr
        forward = forward_raw % tile.shape[0]
        sideways = sideways_raw % tile.shape[1]
        active = np.ones(clean.shape, bool) if alternative == 'solid' else tile[forward, sideways]
        if alternative == 'single_stamp':
            active &= ((0 <= forward_raw) & (forward_raw < tile.shape[0])
                       & (0 <= sideways_raw) & (sideways_raw < tile.shape[1]))
        domain = clean == info['colors'][target] if alternative == 'color_wide' else labels == target
        if alternative == 'bbox_domain':
            top, bottom, left, right = _box(info['cells'][target])
            domain = (rr >= top) & (rr < bottom) & (cc >= left) & (cc < right)
        paint = info['mapping'][selector]
        if alternative == 'remap':
            paint = info['mapping'].get(paint, paint)
        unions[target] = unions.get(target, np.zeros(clean.shape, bool)) | (active & domain)
        targets[target], paints[target] = domain, paint
    for target in unions:
        result[targets[target]] = info['bg']
    for target in unions:
        result[unions[target]] = paints[target]
    r, e, c, f = info['frame']
    cropped = info['grid'][r:e, c:f].copy()
    cropped[1:-1, 1:-1] = result
    return cropped


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the inclusive main frame after all texture transfers, using input only."""
    return _render(_parse(input_grid))


def _partition(rng, h, w, count, rough):
    labels = np.zeros((h, w), dtype=np.int16)
    rr, cc = np.indices(labels.shape)
    for new_id in range(1, count):
        accepted = False
        for _ in range(70):
            sizes = np.bincount(labels.ravel(), minlength=new_id).astype(float)
            old = int(rng.choice(new_id, p=sizes/sizes.sum()))
            mask = labels == old
            a, b, x, y = _box(np.argwhere(mask))
            if rough and b-a >= 8 and y-x >= 8 and rng.random() < .22:
                top = int(rng.integers(a+1, b-4))
                left = int(rng.integers(x+1, y-4))
                bottom = int(rng.integers(top+3, b))
                right = int(rng.integers(left+3, y))
                section = (rr >= top) & (rr < bottom) & (cc >= left) & (cc < right)
            else:
                axis = int(rng.integers(2))
                lo, hi = (a, b) if axis == 0 else (x, y)
                if hi-lo < 6:
                    continue
                cut = int(rng.integers(lo+3, hi-2))
                path = np.full(w if axis == 0 else h, cut)
                if rough and rng.random() < .85:
                    value = cut
                    for start in range(0, len(path), int(rng.integers(2, 5))):
                        value = int(np.clip(value + rng.integers(-2, 3), lo+2, hi-2))
                        path[start:start+4] = value
                coordinate, along = (rr, cc) if axis == 0 else (cc, rr)
                section = coordinate < path[along]
            part, other = mask & section, mask & ~section
            if min(int(part.sum()), int(other.sum())) < 16:
                continue
            if not _connected(part) or not _connected(other):
                continue
            labels[part] = new_id
            accepted = True
            break
        if not accepted:
            return None
    return labels


def _candidates(labels):
    candidates = []
    for center in np.ndindex(labels.shape):
        for direction in DIRECTIONS:
            g = _geometry(center, direction, labels.shape)
            if g is None:
                continue
            src, target = int(labels[center]), int(labels[g['q']])
            if src != target and all(labels[p] == src for p in g['write'] | set(g['sides'])):
                candidates.append(dict(g, source=src, target=target))
    return candidates


def _plan_geometry(rng, candidates, kind, allow_union):
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    selected, blocked, writes, contexts = [], set(), set(), set()

    def legal(a):
        return not (a['write'] & (blocked | contexts) or a['context'] & writes)

    def append(a):
        selected.append(dict(a))
        writes.update(a['write'])
        contexts.update(a['context'])
        blocked.update(a['write'])
        for r, c in a['write']:
            blocked.update((r+dr, c+dc) for dr, dc in DIRECTIONS)

    if kind in ('chain', 'union', 'cycle'):
        found = None
        for first in candidates:
            for second in candidates:
                if kind == 'chain':
                    good = first['target'] == second['source'] and first['source'] != second['target']
                elif kind == 'cycle':
                    good = first['target'] == second['source'] and first['source'] == second['target']
                else:
                    good = first['target'] == second['target']
                halo = first['write'] | {(r+dr, c+dc) for r, c in first['write'] for dr, dc in DIRECTIONS}
                if (good and not (second['write'] & (halo | first['context']))
                        and not (second['context'] & first['write'])):
                    found = first, second
                    break
            if found:
                break
        if not found:
            return None
        for a in found:
            append(a)
    if kind in ('chain', 'union'):
        return selected
    desired = 1 if kind == 'explicit' else int(rng.integers(max(1, len(selected)), 7))
    for a in candidates:
        if len(selected) >= desired:
            break
        if not legal(a):
            continue
        if any(b['target'] == a['target'] for b in selected) and (not allow_union or rng.random() > .22):
            continue
        append(a)
    return selected or None


def _tile(rng, shape, informative=False):
    h, w = shape
    rr, cc = np.indices(shape)
    for _ in range(80):
        kind = 0 if informative else int(rng.integers(8))
        if kind == 0:
            mask = rng.random(shape) < rng.uniform(.25, .7)
        elif kind == 1:
            mask = (rr == 0) | (rr == h-1) | (cc == 0) | (cc == w-1)
        elif kind == 2:
            mask = (rr + cc) % 2 == 0
        elif kind == 3:
            mask = (cc == w//2) | (rr == h//2)
        elif kind == 4:
            mask = rr % 2 == 0
        elif kind == 5:
            mask = cc == (rr % w)
        elif kind == 6:
            mask = rng.random(shape) < .85
        else:
            mask = np.ones(shape, bool)
        for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
            if not edge.any():
                edge[int(rng.integers(len(edge)))] = True
        if informative:
            if mask.all() or (h == w and any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))):
                continue
        return mask
    # A bounded, asymmetric, non-solid tile with visible bounds.
    mask = np.zeros(shape, bool)
    mask[0, :] = mask[:, 0] = True
    mask[-1, -1] = True
    return mask


def _draw_frame(grid, r, c, h, w, color):
    grid[r, c:c+w] = grid[r+h-1, c:c+w] = color
    grid[r:r+h, c] = grid[r:r+h, c+w-1] = color


def _available(occupied, h, w):
    summed = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    total = summed[h:, w:] - summed[:-h, w:] - summed[h:, :-w] + summed[:-h, :-w]
    return np.argwhere(total == 0)


def _reserve(occupied, r, c, h, w):
    occupied[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True


def _pack(rng, board, expected, tiles, mapping, bg, frame_color):
    h, w = board.shape[0] + 2, board.shape[1] + 2
    sources = list(tiles)
    objects = []
    for source in sources:
        objects.append(('tile', source, np.where(tiles[source], source, bg).astype(np.int8)))
        objects.append(('key', source, np.array([[source, source], [source, mapping[source]]], np.int8)))
    positions = None
    # Free packing can distribute the two clue banks independently around the frame.
    for _ in range(16):
        ch = int(rng.integers(max(24, h+5), 31))
        cw = int(rng.integers(max(24, w+5), 31))
        fr, fc = int(rng.integers(ch-h+1)), int(rng.integers(cw-w+1))
        occupied = np.zeros((ch, cw), bool)
        occupied[0, 0] = occupied[0, -1] = occupied[-1, 0] = occupied[-1, -1] = True
        if occupied[fr:fr+h, fc:fc+w].any():
            continue
        _reserve(occupied, fr, fc, h, w)
        order = list(rng.permutation(len(objects)))
        order.sort(key=lambda i: -objects[int(i)][2].size)
        positions = {}
        for i in order:
            _, source, patch = objects[int(i)]
            ph, pw = patch.shape
            available = _available(occupied, ph, pw)
            if not len(available):
                positions = None
                break
            # Sometimes favor a side, while still associating clues by color.
            if rng.random() < .5:
                side = int(rng.integers(4))
                favored = [(r, c) for r, c in available if
                           (r+ph < fr if side == 0 else c > fc+w if side == 1 else
                            r > fr+h if side == 2 else c+pw < fc)]
                if favored:
                    available = favored
            pr, pc = map(int, available[int(rng.integers(len(available)))])
            positions[int(i)] = pr, pc
            _reserve(occupied, pr, pc, ph, pw)
        if positions is not None:
            break
    if positions is None:
        # Two-shelf fallback; reject if this particular combination will not fit.
        mh = max(t.shape[0] for t in tiles.values())
        fr, ch, cw = mh+4, mh+4+h, max(w+2, sum(t.shape[1]+1 for t in tiles.values())+1)
        fc = int(rng.integers(1, cw-w))
        positions = {}
        x = 1
        for source in rng.permutation(sources):
            i = next(i for i, obj in enumerate(objects) if obj[:2] == ('tile', source))
            positions[i] = 0, x
            x += tiles[int(source)].shape[1]+1
        for j, source in enumerate(rng.permutation(sources)):
            i = next(i for i, obj in enumerate(objects) if obj[:2] == ('key', source))
            positions[i] = mh+1, 1+3*j
    if ch > 30 or cw > 30:
        return None
    grid = np.full((ch, cw), bg, np.int8)
    _draw_frame(grid, fr, fc, h, w, frame_color)
    grid[fr+1:fr+h-1, fc+1:fc+w-1] = board
    for i, (_, _, patch) in enumerate(objects):
        pr, pc = positions[i]
        ph, pw = patch.shape
        grid[pr:pr+ph, pc:pc+pw] = patch
    out = np.full((h, w), frame_color, np.int8)
    out[1:-1, 1:-1] = expected
    # Trim unused margins only when all four remaining corners still expose bg.
    a, b, x, y = _box(np.argwhere(grid != bg))
    aa, bb = max(0, a-int(rng.integers(1, 4))), min(ch, b+int(rng.integers(1, 4)))
    xx, yy = max(0, x-int(rng.integers(1, 4))), min(cw, y+int(rng.integers(1, 4)))
    trimmed = grid[aa:bb, xx:yy]
    if all(trimmed[p] == bg for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))):
        grid = trimmed.copy()
    return dict(input=grid, output=out)


def _construct(rng, params, split, kind):
    shift = params['shift']
    if shift == 'extent':
        lo, hi = (9, 15) if split == 'train' else (15, 22)
    else:
        lo, hi = 10, 22
    h, w = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
    count = 4 if kind == 'chain' else int(rng.integers(2 if kind == 'free' else 3, 7))
    # The mandatory chain includes a nonrectangular target, teaching clipping.
    # In boundary mode, tests hold out enclosed targets; training already shows
    # irregular targets, but every training target touches the board boundary.
    rough = kind == 'chain' or shift == 'boundary' or bool(rng.random() < .55)
    labels = _partition(rng, h, w, count, rough)
    if labels is None:
        return None
    candidates = _candidates(labels)
    arrows = _plan_geometry(rng, candidates, kind, params['union'])
    if arrows is None:
        return None
    # Repeat one target fill in a disconnected, untargeted territory when teaching locality.
    repeat = None
    adjacency = {i: set() for i in range(count)}
    for a, b in zip(labels[:-1].ravel(), labels[1:].ravel()):
        if a != b:
            adjacency[int(a)].add(int(b)); adjacency[int(b)].add(int(a))
    for a, b in zip(labels[:, :-1].ravel(), labels[:, 1:].ravel()):
        if a != b:
            adjacency[int(a)].add(int(b)); adjacency[int(b)].add(int(a))
    targets = {a['target'] for a in arrows}
    choices = [(t, other) for t in targets for other in range(count)
               if other not in targets and other != t and other not in adjacency[t]]
    if kind == 'chain':
        if not choices:
            return None
        repeat = choices[int(rng.integers(len(choices)))]
    elif choices and rng.random() < .35:
        repeat = choices[int(rng.integers(len(choices)))]
    palette = list(map(int, rng.permutation(10)))
    bg, frame_color, ink = palette[:3]
    colors = palette[3:3+count]
    if repeat:
        colors[repeat[1]] = colors[repeat[0]]
    sources = list(dict.fromkeys(colors))
    if len(sources) > 6:
        return None
    available = palette[3:]
    mapping = {s: int(rng.choice([p for p in available if p != s])) for s in sources}
    if kind == 'chain':
        a, b = arrows
        ca, cb, cc = colors[a['source']], colors[a['target']], colors[b['target']]
        if len({ca, cb, cc}) < 3:
            return None
        mapping.update({ca: cc, cc: cb, cb: ca})
    tiles = {}
    for i, s in enumerate(sources):
        if shift == 'period':
            shape = (int(rng.integers(2, 4)), int(rng.integers(2, 4))) if split == 'train' else (
                (4, int(rng.integers(2, 5))) if rng.random() < .5 else (int(rng.integers(2, 5)), 4))
        else:
            shape = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        if kind in ('chain', 'union') and min(shape) < 3:
            shape = (3, 3) if shift != 'period' or split == 'train' else shape
        if i and kind not in ('chain', 'union') and rng.random() < .15:
            tiles[s] = tiles[sources[int(rng.integers(i))]].copy()
        else:
            tiles[s] = _tile(rng, shape, informative=kind in ('chain', 'union', 'explicit'))
    target_paint = {}
    for a in arrows:
        host = colors[a['source']]
        if a['target'] in target_paint:
            selectors = [s for s in sources if mapping[s] == target_paint[a['target']]]
            selector = int(rng.choice(selectors))
        elif kind in ('explicit', 'union'):
            selector = int(rng.choice([s for s in sources if s != host]))
        elif kind == 'chain' or rng.random() < .6:
            selector = host
        else:
            selector = int(rng.choice(sources))
        a.update(host=host, selector=selector, explicit=selector != host)
        target_paint[a['target']] = mapping[selector]
    clean = np.asarray(colors, np.int8)[labels]
    annotated = clean.copy()
    for a in arrows:
        for p in a['write']:
            annotated[p] = ink
        if a['explicit']:
            annotated[a['center']] = a['selector']
    # Constructor uses original labels and scalar cell arithmetic, not solve().
    expected = clean.copy()
    for target, paint in target_paint.items():
        instructions = [a for a in arrows if a['target'] == target]
        for r, c in np.argwhere(labels == target):
            active = False
            for a in instructions:
                tile = tiles[a['selector']]
                dr, dc = a['d']
                y, x = int(r)-a['q'][0], int(c)-a['q'][1]
                active |= bool(tile[(y*dr+x*dc) % tile.shape[0], (-y*dc+x*dr) % tile.shape[1]])
            expected[r, c] = paint if active else bg
    pair = _pack(rng, annotated, expected, tiles, mapping, bg, frame_color)
    if pair is None:
        return None
    try:
        info = _parse(pair['input'])
    except ValueError:
        return None
    if not np.array_equal(_render(info), pair['output']):
        return None
    if kind == 'chain':
        if not all(_differs(info, alt) for alt in ('solid', 'target_selector', 'bbox', 'no_orientation',
                                                  'remap', 'color_wide', 'bbox_domain',
                                                  'single_stamp', 'relay')):
            return None
    if kind in ('explicit', 'union') and not _differs(info, 'no_explicit'):
        return None
    if kind == 'union':
        # Every port must contribute pixels, with some inactive target cells left.
        target = info['arrows'][0]['target']
        combined = pair['output']
        for a in info['arrows']:
            single = dict(info, arrows=[a])
            if np.array_equal(_render(single), combined):
                return None
        if not np.any(combined[1:-1, 1:-1][info['labels'] == target] == bg):
            return None
    if shift == 'boundary':
        enclosed = _features(info)['enclosed_targets']
        if (split == 'train' and enclosed) or (split == 'test' and not enclosed):
            return None
    return pair


def _differs(info, alternative):
    return not np.array_equal(_render(info), _render(info, alternative))


def _union_witness(info):
    """At least two ports each add exclusive pixels to one non-solid result."""
    complete = _render(info)
    for target in info['assignments']:
        ports = [a for a in info['arrows'] if a['target'] == target]
        if len(ports) < 2:
            continue
        if not np.any(complete[1:-1, 1:-1][info['labels'] == target] == info['bg']):
            continue
        if all(not np.array_equal(complete, _render(dict(
                info, arrows=[b for b in info['arrows'] if b is not a]))) for a in ports):
            return True
    return False


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate distinct examples with one fixed rule; default counts are 4/1."""
    _need(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    _need(_integer(num_train) and 2 <= num_train <= 10, "num_train must be an integer in 2..10")
    _need(_integer(num_test) and 1 <= num_test <= 4, "num_test must be an integer in 1..4")
    _need(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    params = dict(shift=str(rng.choice(['extent', 'boundary', 'period'])), union=bool(rng.integers(2)))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            kind = ('chain' if index == 0 else 'union' if params['union'] else 'explicit') if split == 'train' and index < 2 else (
                'cycle' if rng.random() < .25 else 'free')
            for _ in range(180):
                pair = _construct(rng, params, split, kind)
                if pair is not None and _key(pair['input']) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    break
            else:
                raise RuntimeError('bounded construction exhausted for ' + kind)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    labels, cells, arrows = info['labels'], info['cells'], info['arrows']
    areas = [len(p) for p in cells]
    irregular = any(len(p) != (b-a)*(y-x) for p in cells for a, b, x, y in [_box(p)])
    chain = any(a['target'] == b['source'] and a['source'] != b['target'] and not a['explicit']
                and not b['explicit'] for a in arrows for b in arrows if a is not b)
    h, w = labels.shape
    enclosed_targets = sum(all(0 < r < h-1 and 0 < c < w-1 for r, c in cells[t])
                           for t in info['assignments'])
    return dict(shape=labels.shape, territories=len(cells), sources=len(info['tiles']), arrows=len(arrows),
                explicit=sum(a['explicit'] for a in arrows), chain=chain, irregular=irregular,
                enclosed_targets=enclosed_targets,
                union=len({a['target'] for a in arrows}) < len(arrows),
                locality=_differs(info, 'color_wide'), areas=areas,
                directions={a['d'] for a in arrows},
                periods={m.shape for m in info['tiles'].values()},
                disconnected=sum(not _connected(m) for m in info['tiles'].values()),
                solids=sum(bool(m.all()) for m in info['tiles'].values()),
                repeated_geometry=len(info['tiles'])-len({(m.shape, m.tobytes()) for m in info['tiles'].values()}),
                bg=info['bg'], ink=info['ink'], frame_color=info['frame_color'])


def validate_generated_puzzle(puzzle):
    """Schema, family semantics, original-component locality and curriculum checks."""
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid pair counts")
    seen, analyses = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            _need(np.array_equal(_render(info), out), "incorrect texture transfer")
            _need(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "invented output color")
            _need(np.all(out[0] == info['frame_color']) and np.all(out[-1] == info['frame_color'])
                  and np.all(out[:, 0] == info['frame_color']) and np.all(out[:, -1] == info['frame_color']),
                  "frame not preserved")
            inner = out[1:-1, 1:-1]
            untouched = ~np.isin(info['labels'], list(info['assignments']))
            _need(np.array_equal(inner[untouched], info['clean'][untouched]), "untargeted territory changed")
            for target, paint in info['assignments'].items():
                _need(set(map(int, np.unique(inner[info['labels'] == target]))) <= {paint, info['bg']},
                      "target has more than its direct paint and background")
            analyses[split].append(info)
    train = analyses['train']
    _need(any(_features(i)['chain'] for i in train), "missing simultaneous-chain demonstration")
    for alt in ('target_selector', 'solid', 'bbox', 'no_orientation', 'remap', 'color_wide',
                'no_explicit', 'bbox_domain', 'single_stamp', 'relay'):
        _need(any(_differs(i, alt) for i in train), "training does not distinguish " + alt)
    if any(_features(i)['union'] for i in analyses['test']):
        _need(any(_union_witness(i) for i in train), "contributing union not demonstrated")
    _need(bool(_visible_shifts(analyses)), "missing an input-visible train/test shift")
    return None


def _visible_shifts(analyses):
    """Measure holdouts from inputs, without sampled mode flags or output labels."""
    tr = [_features(i) for i in analyses['train']]
    te = [_features(i) for i in analyses['test']]
    result = []
    if max(max(f['shape']) for f in tr) < min(min(f['shape']) for f in te):
        result.append('extent')
    if all(not f['enclosed_targets'] for f in tr) and all(f['enclosed_targets'] for f in te):
        result.append('boundary')
    if (all(max(max(p) for p in f['periods']) <= 3 for f in tr)
            and all(all(max(p) == 4 for p in f['periods']) for f in te)):
        result.append('period')
    return result


def self_test():
    """Bounded 42-seed variation, ablations and semantic metamorphic checks."""
    puzzles, features, structures = set(), [], set()
    first_info = union_info = None
    shifts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])
        puzzles.add(fingerprint)
        if seed in (0, 17):
            again = generate_puzzle(seed)
            _need(fingerprint == tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in again[s]),
                  "nondeterministic generation")
        analyses = {s: [_parse(p['input']) for p in puzzle[s]] for s in ('train', 'test')}
        shifts.update(_visible_shifts(analyses))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                saved = pair['input'].copy()
                _need(np.array_equal(solve(saved), pair['output']) and np.array_equal(saved, pair['input']),
                      "solver disagreement or mutation")
                info = _parse(saved)
                f = _features(info)
                f['split'] = split
                features.append(f)
                structures.add(_key(info['labels']))
                if first_info is None:
                    first_info = info
                if union_info is None and _differs(info, 'last'):
                    union_info = info
        if seed < 3:
            # Arbitrary color names, including background and arrow ink, are irrelevant.
            permutation = np.random.default_rng(seed+900).permutation(10)
            for pair in puzzle['test']:
                _need(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
                      "palette permutation changed the rule")
    _need(len(puzzles) == 42 and len(structures) > 100, "collapsed structural variation")
    _need(shifts == {'extent', 'boundary', 'period'}, "missing curriculum shift")
    _need(len({f['shape'] for f in features}) > 35, "collapsed board dimensions")
    _need({d for f in features for d in f['directions']} == set(DIRECTIONS), "missing arrow direction")
    _need({p for f in features for p in f['periods']} == {(h, w) for h in range(2, 5) for w in range(2, 5)},
          "missing rectangular tile periods")
    _need({f['sources'] for f in features} >= {2, 3, 4, 5}, "collapsed source counts")
    _need({f['arrows'] for f in features} >= {1, 2, 3, 4, 5}, "collapsed transfer counts")
    for name in ('irregular', 'union', 'locality', 'disconnected', 'solids', 'repeated_geometry'):
        _need(sum(bool(f[name]) for f in features) >= 4, "missing variation: " + name)
    for name in ('bg', 'ink', 'frame_color'):
        _need({f[name] for f in features} == set(range(10)), "palette role missing: " + name)
    # Canvas translation: shift every object together inside background margins.
    info = first_info
    r, e, c, f = info['frame']
    padded = np.pad(info['grid'], ((1, 1), (1, 1)), constant_values=info['bg'])
    if max(padded.shape) <= 30:
        _need(np.array_equal(solve(padded), _render(info)), "canvas translation changed the crop")
    # A deliberate conflicting second instruction must be rejected, not ordered.
    _need(union_info is not None, "no contributing union example")
    info = union_info
    a = next(a for a in info['arrows'] if sum(b['target'] == a['target'] for b in info['arrows']) > 1)
    other = next((s for s, p in info['mapping'].items() if p != info['mapping'][a['selector']]), None)
    if other is not None:
        bad = info['grid'].copy()
        r, _, c, _ = info['frame']
        y, x = a['center']
        bad[r+1+y, c+1+x] = other
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise ValueError("conflicting incoming paint accepted")
    # A valid family input beyond the sampler's 2..6-component distribution:
    # seven alternating bands, with an explicit selector present only in its
    # legend. The rightward 3x2 tile has these hand-constructed alternating rows.
    clean = np.tile(np.repeat([3, 4, 3, 4, 3, 4, 3], 3), (8, 1)).astype(np.int8)
    board = clean.copy()
    arrow = _geometry((3, 1), (0, 1), board.shape)
    for point in arrow['write']:
        board[point] = 2
    board[3, 1] = 5
    expected = clean.copy()
    expected[::2, 3:6] = [0, 7, 7]
    expected[1::2, 3:6] = [7, 0, 7]
    tiles = {3: np.array([[1, 0], [0, 1]], bool),
             4: np.array([[0, 1], [1, 0]], bool),
             5: np.array([[1, 0], [0, 1], [1, 1]], bool)}
    pair = _pack(np.random.default_rng(918), board, expected, tiles, {3: 6, 4: 3, 5: 7}, 0, 1)
    _need(pair is not None and np.array_equal(solve(pair['input']), pair['output']),
          "seven components or selector-only legend failed")
    analysis = _parse(pair['input'])
    _need(len(analysis['cells']) == 7 and 5 not in analysis['colors'], "extension fixture collapsed")
    for tr, te in ((2, 1), (10, 4)):
        p = generate_puzzle(9100+tr, tr, te)
        _need(len(p['train']) == tr and len(p['test']) == te, "ignored legal counts")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise ValueError("illegal count accepted")
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(puzzles),
                original_partitions=len(structures), board_shapes=len({f['shape'] for f in features}),
                directions=len({d for f in features for d in f['directions']}),
                period_shapes=len({p for f in features for p in f['periods']}),
                features={name: sum(bool(f[name]) for f in features) for name in
                          ('irregular', 'union', 'locality', 'disconnected', 'solids', 'repeated_geometry')},
                arrow_counts=sorted({f['arrows'] for f in features}),
                source_counts=sorted({f['sources'] for f in features}),
                visible_shifts=sorted(shifts),
                extended_input_cases=['seven_components', 'selector_only_legend'],
                semantic_verification='generator-owned checks only')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
