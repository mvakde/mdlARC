"""Cavity tally waves: color-matched component inventories drive geodesic bands.

No parent code or data is imported. All output conventions are visible in input.
"""
from collections import deque
import numpy as np

N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 413593356, 381977201, 542081129, 873886359, 845315346,
              78535591, 812676042, 666404934, 30741541, 419016884, 667097149,
              195908470, 559150697, 329784989, 843908539, 912234858, 463700191,
              384852391, 619091179, 914864486, 51588472, 267535316, 457380027,
              669293666, 893413372, 879400184, 977930561, 656456795, 858001438,
              934851199, 680032580, 871066596, 237038734, 28573574, 272282497,
              866311823, 404655685, 383409335, 547433781, 825548975, 9761352)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer grid required')
    _require(bool(((0 <= a) & (a <= 9)).all()), 'ARC colors required')
    return a.astype(np.int8, copy=False)


def _regions(mask, steps=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _mask(shape, cells):
    m = np.zeros(shape, bool)
    if cells:
        r, c = np.array(sorted(cells)).T
        m[r, c] = True
    return m


def _holes(barrier, steps=N4):
    h, w = barrier.shape
    return [cells for cells in _regions(~barrier, steps)
            if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells)]


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[1 + r:1 + r + h, 1 + c:1 + c + w] for r, c in N8])


def _distances(cells, source):
    """Orthogonal shortest paths in one supplied pocket, including its key."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        for dr, dc in N4:
            q = (r + dr, c + dc)
            if q in cells and q not in dist:
                dist[q] = dist[(r, c)] + 1
                queue.append(q)
    _require(len(dist) == len(cells), 'disconnected pocket')
    return dist


def _analyze(input_grid):
    g = _grid(input_grid)
    frequencies = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    _require(frequencies[bg] > g.size / 2, 'background must occupy over half of input')
    candidates = []
    for wall in sorted(set(map(int, g.ravel())) - {bg}):
        holes = _holes(g == wall)
        keys = []
        for cells in holes:
            markers = [(r, c) for r, c in sorted(cells) if g[r, c] not in (bg, wall)]
            if len(markers) != 1:
                break
            keys.append(markers[0])
        else:
            if holes:
                candidates.append((wall, holes, keys))
    _require(len(candidates) == 1, 'exactly one wall color must enclose singleton keys')
    wall, holes, keys = candidates[0]
    body = g == wall
    objects = _regions(body)
    _require(len(objects) == len(_regions(body, N8)), 'diagonal wall contact')
    _require({frozenset(c) for c in holes} == {frozenset(c) for c in _holes(body, N8)},
             'diagonal-only wall leak or cavity join')
    hole_mask = _mask(g.shape, set().union(*holes))
    exterior = ~body & ~hole_mask
    labels = np.full(g.shape, -1, int)
    owners, local_holes = [], set()
    for i, cells in enumerate(objects):
        labels[_mask(g.shape, cells)] = i
        # A body by itself must account for its holes; collective enclosures
        # and nested wall components are excluded from the intended family.
        own = _holes(_mask(g.shape, cells))
        for pocket in own:
            _require(not any(g[r, c] == wall for r, c in pocket), 'nested wall body')
            _require(not (pocket & local_holes), 'overlapping cavity ownership')
            local_holes |= pocket
        owners.append(own)
    _require(local_holes == set().union(*holes), 'collective enclosure')
    tokens, counts = {}, {}
    token_labels = np.full(g.shape, -1, int)
    serial = 0
    for color in sorted(set(map(int, g.ravel())) - {bg, wall}):
        tokens[color] = _regions((g == color) & exterior, N8)
        counts[color] = len(tokens[color])
        for cells in tokens[color]:
            token_labels[_mask(g.shape, cells)] = serial
            serial += 1
    wall_halo = _dilate(body)
    for r, c in np.argwhere(token_labels >= 0):
        neighborhood = token_labels[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        _require(np.all((neighborhood < 0) | (neighborhood == token_labels[r, c])),
                 'different exterior tokens must remain separate')
        _require(not wall_halo[r, c], 'exterior token touches wall')
    pockets = []
    for cells, key in zip(holes, keys):
        color = int(g[key])
        n = counts.get(color, 0)
        _require(n > 0, 'every key requires a positive exterior inventory')
        distances = _distances(cells, key)
        _require(max(distances.values()) >= 2 * n - 1, 'pocket needs a blank terminal band')
        neighbor_ids = {int(labels[r + dr, c + dc]) for r, c in cells for dr, dc in N4
                        if labels[r + dr, c + dc] >= 0}
        _require(len(neighbor_ids) == 1, 'pocket has ambiguous owner')
        pockets.append(dict(cells=cells, key=key, color=color, count=n,
                            distances=distances, owner=next(iter(neighbor_ids))))
    outline = _dilate(body) & exterior & (g == bg)
    return dict(grid=g, background=bg, wall=wall, body=body, objects=objects,
                pockets=pockets, holes=hole_mask, exterior=exterior,
                tokens=tokens, counts=counts, outline=outline, owners=owners)


def _paint(info, counts=None):
    g, bg, wall = info['grid'], info['background'], info['wall']
    out = g.copy()
    out[info['outline']] = wall
    for pocket in info['pockets']:
        n = pocket['count'] if counts is None else counts[pocket['color']]
        for cell, distance in pocket['distances'].items():
            out[cell] = pocket['color'] if distance % 2 == 0 and distance < 2 * n else bg
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only: recognize walls/keys, count exterior components, route waves."""
    return _paint(_analyze(input_grid))


def _orient(rng, a):
    a = np.rot90(a, int(rng.integers(4)))
    if rng.integers(2):
        a = np.fliplr(a)
    return a.copy()


def _pocket(rng, count, style, interior_key=False):
    # Rectangles, bends and detours are independent of wall silhouette.
    for attempt in range(20):
        h = int(rng.integers(2, 6))
        w = int(rng.integers(max(4, count + 2), max(5, count + 5)))
        if style in ('u', 'witness_u'):
            h = max(3, h)
        if style == 'z':
            h = max(4, h)
        if style == 'witness_u':
            w = 5
        m = np.zeros((h, w), bool)
        if style == 'rect':
            m[:] = True
        elif style == 'l':
            m[:int(rng.integers(1, min(3, h))), :] = True
            m[:, :int(rng.integers(1, 3))] = True
        elif style in ('u', 'witness_u'):
            m[:, 0] = m[:, -1] = True
            m[-1, :] = True
        elif style == 'z':
            # A serpentine pocket has short visual distances across walls.
            m[0, :] = m[-1, :] = True
            m[:h // 2 + 1, -1] = True
            m[h // 2:, 0] = True
            m[h // 2, :] = True
        else:
            raise ValueError('unknown pocket style')
        cells = set(map(tuple, np.argwhere(m).tolist()))
        viable = []
        for source in sorted(cells):
            d = _distances(cells, source)
            if max(d.values()) >= 2 * count - 1:
                viable.append(source)
        if viable:
            if style in ('u', 'witness_u') and not interior_key:
                key = (0, 0) if rng.integers(2) else (0, w - 1)
            elif interior_key:
                center = np.array(m.shape) / 2
                viable.sort(key=lambda q: float(np.abs(np.array(q) - center).sum()))
                key = viable[int(rng.integers(min(3, len(viable))))]
            else:
                key = viable[int(rng.integers(len(viable)))]
            return m, key
    # Finite fallback still has a detour and enough capacity for all counts.
    m = np.zeros((4, 2 * count + 1), bool)
    m[:, 0] = m[:, -1] = True
    m[-1, :] = True
    return m, (0, 0)


def _wall_item(rng, pocket_specs, bg, wall, interior_key=False):
    pockets = [_pocket(rng, n, style, interior_key) for color, n, style in pocket_specs]
    thick = int(rng.choice([1, 1, 2]))
    gap = int(rng.choice([1, 2]))
    # Place 1..3 pockets within one connected mass, in an independently
    # chosen arrangement. Geometry is not a chart with category rows.
    if len(pockets) <= 2:
        offsets = [(0, 0)]
        if len(pockets) == 2:
            offsets.append((int(rng.integers(0, 3)), pockets[0][0].shape[1] + gap))
    else:
        offsets = [(0, 0), (0, pockets[0][0].shape[1] + gap),
                   (max(pockets[0][0].shape[0], pockets[1][0].shape[0]) + gap,
                    int(rng.integers(0, 3)))]
    h = max(r + p[0].shape[0] for (r, c), p in zip(offsets, pockets)) + 2 * thick
    w = max(c + p[0].shape[1] for (r, c), p in zip(offsets, pockets)) + 2 * thick
    a = np.full((h, w), wall, np.int8)
    reserved = np.zeros((h, w), bool)
    for (r, c), (m, key), (color, n, style) in zip(offsets, pockets, pocket_specs):
        r, c = r + thick, c + thick
        a[r:r + m.shape[0], c:c + m.shape[1]][m] = bg
        reserved[r:r + m.shape[0], c:c + m.shape[1]] |= m
        a[r + key[0], c + key[1]] = color
    # Off-center lobes/corner cuts vary the exterior without changing holes.
    protect = _dilate(reserved) | reserved
    for _ in range(int(rng.integers(0, 5))):
        candidate = a.copy()
        side = int(rng.integers(4))
        depth = int(rng.integers(1, 3))
        if side == 0:
            cut = (slice(0, depth), slice(0, int(rng.integers(1, w))))
        elif side == 1:
            cut = (slice(h - depth, h), slice(int(rng.integers(w)), w))
        elif side == 2:
            cut = (slice(0, int(rng.integers(1, h))), slice(w - depth, w))
        else:
            cut = (slice(int(rng.integers(h)), h), slice(0, depth))
        view = candidate[cut]
        view[~protect[cut]] = bg
        barrier = candidate == wall
        if len(_regions(barrier)) == 1 and len(_regions(barrier, N8)) == 1:
            if {frozenset(x) for x in _holes(np.pad(barrier, 1))} == {
                    frozenset((r + 1, c + 1) for r, c in x) for x in _regions(reserved)}:
                a = candidate
    # A short body tail makes the sealed-hole owner visibly more than a ring.
    if rng.random() < .4:
        a = np.pad(a, ((0, 2), (0, 0)), constant_values=bg)
        choices = np.flatnonzero(a[-3] == wall)
        if len(choices):
            c = int(rng.choice(choices))
            a[-2:, c] = wall
    return _orient(rng, a)


def _open_item(rng, color, bg, wall):
    h, w = int(rng.integers(5, 8)), int(rng.integers(5, 8))
    a = np.full((h, w), wall, np.int8)
    a[1:-1, 1:-1] = bg
    a[0, w // 2] = bg
    a[h // 2, w // 2] = color
    return _orient(rng, a)


def _token(rng, color, bg, special=None):
    if special == 'diamond':
        mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)
    elif special == 'square':
        mask = np.ones((2, 2), bool)
    elif special == 'single':
        mask = np.ones((1, 1), bool)
    else:
        mode = int(rng.integers(6))
        if mode == 0:
            mask = np.ones((1, int(rng.integers(1, 6))), bool)
        elif mode == 1:
            mask = np.eye(int(rng.integers(2, 5)), dtype=bool)
        elif mode == 2:
            mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], bool)
        elif mode == 3:
            mask = np.ones((2, int(rng.integers(2, 5))), bool)
        elif mode == 4:
            mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
        else:
            side = int(rng.integers(2, 5))
            cells = {(side // 2, side // 2)}
            for _ in range(int(rng.integers(1, 8))):
                frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in N8
                                   if 0 <= r + dr < side and 0 <= c + dc < side} - cells)
                if frontier:
                    cells.add(frontier[int(rng.integers(len(frontier)))])
            mask = _mask((side, side), cells)
            rr, cc = np.nonzero(mask)
            mask = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    return np.where(_orient(rng, mask), color, bg).astype(np.int8)


def _pack(rng, items, bg):
    area = sum(a.size for a in items)
    # Largest-first free rectangle placement, without fixed quadrants/slots.
    order = sorted(range(len(items)), key=lambda i: -items[i].size)
    for attempt in range(18):
        if attempt < 12:
            ratio = float(np.exp(rng.uniform(-.45, .45)))
            target = max(225, area / float(rng.uniform(.26, .40)))
            h = int(np.clip(np.sqrt(target * ratio), 15, 30))
            w = int(np.clip(np.sqrt(target / ratio), 15, 30))
            h = min(30, max(h, max(a.shape[0] for a in items) + 2))
            w = min(30, max(w, max(a.shape[1] for a in items) + 2))
        else:
            h = w = 30
        margin = int(rng.choice([0, 0, 1, 2])) if attempt < 12 else 0
        blocked = np.zeros((h, w), bool)
        g = np.full((h, w), bg, np.int8)
        for i in order:
            a = items[i]
            sh, sw = a.shape
            candidates = [(r, c) for r in range(margin, h - sh - margin + 1)
                          for c in range(margin, w - sw - margin + 1)
                          if not blocked[r:r + sh, c:c + sw].any()]
            if not candidates:
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            g[r:r + sh, c:c + sw] = a
            blocked[max(0, r - 1):r + sh + 1, max(0, c - 1):c + sw + 1] = True
        else:
            if np.count_nonzero(g == bg) > g.size / 2:
                return g
    return None


def _make_pair(rng, palette, kind, shift):
    bg, wall, categories = palette[0], palette[1], palette[2:]
    a, b, unused = categories[:3]
    if kind == 'count_anchor':
        counts = {a: 3}
        specs = [(a, 3, 'witness_u')]
        open_color = a
        distractors = {b: int(rng.integers(1, 4))}
    elif kind == 'group_anchor':
        counts = {a: 2, b: int(rng.integers(1, 4))}
        specs = [(a, counts[a], 'l'), (b, counts[b], 'rect')]
        open_color = None
        distractors = {unused: int(rng.integers(0, 3))}
    else:
        k = (3 if kind == 'test' and shift == 'group' else
             int(rng.integers(1, 3 if shift in ('count', 'group') else 4)))
        # Optional examples can have several pockets of the same category.
        colors = list(map(int, rng.choice(categories[:4], k, replace=True)))
        if kind == 'test' and shift == 'palette':
            colors[0] = a
        counts = {c: int(rng.integers(1, 5)) for c in colors}
        if kind == 'test' and shift == 'count':
            counts[colors[0]] = 5
        elif kind == 'test' and max(counts.values()) == 1:
            # Each held-out scene requires extending at least one wave.
            counts[colors[int(rng.integers(len(colors)))]] = 2
        while sum(counts.values()) > 11:
            c = max(counts, key=counts.get)
            counts[c] -= 1
        specs = [(c, counts[c], str(rng.choice(['rect', 'l', 'u', 'z']))) for c in colors]
        open_color = int(rng.choice(colors)) if rng.random() < .45 else None
        free = [c for c in categories if c not in counts]
        distractors = {int(rng.choice(free)): int(rng.integers(0, 4))} if free else {}
    interior = kind == 'group_anchor' or (kind == 'test' and shift == 'palette')
    # A global inventory can drive pockets owned by different wall bodies.
    # Teaching anchors retain the shared-body and open-bay contrasts.
    groups = [specs]
    if kind not in ('count_anchor', 'group_anchor') and len(specs) > 1:
        order = list(map(int, rng.permutation(len(specs))))
        pieces = int(rng.integers(1, len(specs) + 1))
        cuts = sorted(map(int, rng.choice(np.arange(1, len(specs)),
                                          pieces - 1, replace=False)))
        stops = [0] + cuts + [len(specs)]
        groups = [[specs[order[j]] for j in range(lo, hi)]
                  for lo, hi in zip(stops, stops[1:])]
    for attempt in range(24):
        items = [_wall_item(rng, group, bg, wall, interior) for group in groups]
        if open_color is not None:
            items.append(_open_item(rng, open_color, bg, wall))
        for color, n in list(counts.items()) + list(distractors.items()):
            external = n - int(color == open_color)
            for j in range(external):
                special = None
                if kind == 'count_anchor' and color == a:
                    special = ['diamond', 'square'][j]
                items.append(_token(rng, color, bg, special))
        g = _pack(rng, items, bg)
        if g is None:
            continue
        try:
            info = _analyze(g)
            _require(info['background'] == bg and info['wall'] == wall, 'wrong role recovery')
            _require(all(info['counts'].get(c, 0) == n for c, n in counts.items()), 'inventory placement error')
        except AssertionError:
            continue
        # Labels are constructed from requested inventories, not solve's
        # recovered counts. Geometry is read from placed, visible pockets.
        out = _paint(info, counts)
        return dict(input=g, output=out)
    return _fallback_pair(rng, specs, counts, distractors, open_color, bg, wall, kind)



def _fallback_pair(rng, specs, counts, distractors, open_color, bg, wall, kind):
    """Guaranteed 30x30 shelf scene, preserving the two teaching witnesses.

    At most three pockets (maximum count five) and fourteen free tokens fit.
    This simplifies shapes only when random free packing is exhausted.
    """
    g = np.full((30, 30), bg, np.int8)
    if kind == 'count_anchor':
        pocket = np.zeros((3, 5), bool)
        pocket[:, 0] = pocket[:, -1] = True
        pocket[-1] = True
        g[:5, :7] = wall
        g[1:4, 1:6][pocket] = bg
        g[1, 1] = specs[0][0]
        bottom = 5
    else:
        # Each thin corridor is a legitimate sealed cavity with independent
        # capacity and key. The connecting mass owns all the corridors.
        bottom = 2 * len(specs) + 1
        width = 2 * max(n for c, n, style in specs) + 2
        g[:bottom, :width] = wall
        for i, (color, n, style) in enumerate(specs):
            r = 1 + 2 * i
            g[r, 1:2 * n + 1] = bg
            g[r, 1] = color
    if open_color is not None:
        g[:5, 17:22] = wall
        g[1:4, 18:21] = bg
        g[0, 19] = bg
        g[2, 19] = open_color
    slots = [(r, c) for r in (max(bottom + 2, 8), max(bottom + 2, 8) + 4)
             for c in range(1, 27, 4)]
    number = 0
    for color, n in list(counts.items()) + list(distractors.items()):
        for j in range(n - int(color == open_color)):
            r, c = slots[number]
            number += 1
            if kind == 'count_anchor' and color == specs[0][0]:
                token = _token(rng, color, bg, ['diamond', 'square'][j])
                g[r:r + token.shape[0], c:c + token.shape[1]] = token
            else:
                g[r, c] = color
    info = _analyze(g)
    _require(all(info['counts'][c] == n for c, n in counts.items()), 'fallback inventories')
    return dict(input=g, output=_paint(info, counts))

def _legal_integer(value, low, high):
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
            and low <= value <= high)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _legal_integer(num_train, 2, 10) or not _legal_integer(num_test, 1, 4) or num_train <= num_test:
        raise ValueError('require integer counts 2..10/1..4 with num_train > num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(['count', 'palette', 'group']))
    base_palette = list(map(int, rng.permutation(10)))
    fixed_palette = bool(rng.random() < .25)
    novel = base_palette[-1] if shift == 'palette' else None
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', num_train), ('test', num_test)):
        for index in range(number):
            kind = ('count_anchor' if index == 0 else 'group_anchor' if index == 1 else 'ordinary') if split == 'train' else 'test'
            for retry in range(16):
                # Roles are local and are recoverable without a learned map.
                palette = base_palette.copy() if fixed_palette else list(map(int, rng.permutation(10)))
                if novel is not None:
                    palette.remove(novel)
                    if split == 'test':
                        palette.insert(2, novel)
                pair = _make_pair(rng, palette, kind, shift)
                signature = _signature(pair['input'])
                if signature not in seen:
                    break
            else:
                raise RuntimeError('bounded uniqueness construction exhausted')
            seen.add(signature)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(g):
    return tuple(g.shape), np.ascontiguousarray(g, dtype=np.int8).tobytes()


def _features(info):
    shapes = []
    mixed = False
    diagonal = False
    for c in {p['color'] for p in info['pockets']}:
        objects = info['tokens'][c]
        mixed |= len({len(o) for o in objects}) > 1
        diagonal |= any(len(_regions(_mask(info['grid'].shape, o))) > 1 for o in objects)
    for objects in info['tokens'].values():
        for cells in objects:
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            shapes.append(tuple(sorted((r - r0, c - c0) for r, c in cells)))
    detour = any(any(d > abs(r - p['key'][0]) + abs(c - p['key'][1])
                    for (r, c), d in p['distances'].items()) for p in info['pockets'])
    leak_token = False
    for cells in info['objects']:
        r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
        c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
        if not _holes(_mask(info['grid'].shape, cells)):
            leak_token |= any(r0 < r < r1 and c0 < c < c1
                              for color in {p['color'] for p in info['pockets']}
                              for o in info['tokens'][color] for r, c in o)
    key_counts = tuple(sorted(p['count'] for p in info['pockets']))
    return dict(shapes=tuple(sorted(shapes)), mixed=mixed, diagonal=diagonal, detour=detour,
                leak_token=leak_token, key_counts=key_counts,
                pockets=len(info['pockets']), bodies=len(info['objects']),
                sealed_bodies=sum(bool(o) for o in info['owners']),
                multi=any(len(o) > 1 for o in info['owners']),
                edge=bool(info['body'][0].any() or info['body'][-1].any()
                          or info['body'][:, 0].any() or info['body'][:, -1].any()))


def _manhattan_prediction(info):
    out = info['grid'].copy()
    out[info['outline']] = info['wall']
    for p in info['pockets']:
        for r, c in p['cells']:
            d = abs(r - p['key'][0]) + abs(c - p['key'][1])
            out[r, c] = p['color'] if d % 2 == 0 and d < 2 * p['count'] else info['background']
    return out


def _box_exclusion_counts(info):
    """Shortcut: exclude every wall bounding box instead of sealed pockets."""
    outside = np.ones(info['grid'].shape, bool)
    for cells in info['objects']:
        r0, r1 = min(r for r, c in cells), max(r for r, c in cells) + 1
        c0, c1 = min(c for r, c in cells), max(c for r, c in cells) + 1
        outside[r0:r1, c0:c1] = False
    return {p['color']: len(_regions((info['grid'] == p['color']) & outside, N8))
            for p in info['pockets']}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            g, out = _grid(pair['input']), _grid(pair['output'])
            _require(g.shape == out.shape, 'preserve canvas dimensions')
            sig = _signature(g)
            _require(sig not in seen, 'inputs must be distinct')
            seen.add(sig)
            info = _analyze(g)
            _require(np.array_equal(out, _paint(info)), 'incorrect tally/topology output')
            _require(info['outline'].any(), 'outline evidence required')
            unchanged = ~info['holes'] & ~info['outline']
            _require(np.array_equal(g[unchanged], out[unchanged]), 'walls/tokens/distant exterior changed')
            for p in info['pockets']:
                # Exactly n nonempty colored distance layers; none missing,
                # no odd layer or distant layer painted, key not counted.
                occupied = {d for cell, d in p['distances'].items() if out[cell] == p['color']}
                _require(occupied == set(range(0, 2 * p['count'], 2)), 'wrong unary distance layers')
            if split == 'train':
                training.append((info, _features(info), out))
            else:
                _require(any(p['count'] >= 2 for p in info['pockets']),
                         'each test must require an extended inventory wave')
                testing.append((info, _features(info), out))
    _require(any(f['diagonal'] and f['mixed'] and f['leak_token'] for i, f, o in training),
             'training must distinguish diagonal counting, areas and open/closed markers')
    _require(any(f['multi'] for i, f, o in training), 'training must demonstrate shared-body pockets')
    _require(any(max(f['key_counts']) >= 3 for i, f, o in training), 'need extended alternating tally')
    _require(any(not np.array_equal(_manhattan_prediction(i), o) for i, f, o in training),
             'training must distinguish geodesic from Manhattan distance')
    _require(any(not np.array_equal(_paint(i, {p['color']: 1 for p in i['pockets']}), o)
                 for i, f, o in training), 'training must require object inventories')
    _require(any(not np.array_equal(_paint(i, {
        p['color']: len(_regions((i['grid'] == p['color']) & i['exterior'], N4))
        for p in i['pockets']}), o) for i, f, o in training),
        'training must reject four-neighbor token counts')
    _require(any(not np.array_equal(_paint(i, {
        p['color']: len(_regions(i['grid'] == p['color'], N8))
        for p in i['pockets']}), o) for i, f, o in training),
        'training must require excluding sealed keys')
    _require(any(not np.array_equal(_paint(i, _box_exclusion_counts(i)), o)
                 for i, f, o in training), 'training must count open-bay tokens')
    _require(any(not np.array_equal(_paint(i, {
        p['color']: int(np.count_nonzero((i['grid'] == p['color']) & i['exterior']))
        for p in i['pockets']}), o) for i, f, o in training),
        'training must reject pixel inventories')
    _require(any(not np.array_equal(_paint(i, {
        p['color']: sum(i['counts'].values()) for p in i['pockets']}), o)
        for i, f, o in training), 'training must reject pooling all token colors')
    # Removing enclosure counts singleton keys as objects and changes the
    # requested extent (or overflows the explicit blank capacity).
    _require(any(any(len(_regions(i['grid'] == p['color'], N8)) != p['count']
                         for p in i['pockets']) for i, f, o in training),
             'training must reject counting keys as exterior tokens')
    # Infer the sampling shift from pixels; no mode or seed metadata is needed.
    train_colors = set().union(*(set(map(int, i['grid'].ravel())) for i, f, o in training))
    count_shift = max(max(f['key_counts']) for i, f, o in testing) > max(
        max(f['key_counts']) for i, f, o in training)
    pocket_shift = max(f['pockets'] for i, f, o in testing) > max(
        f['pockets'] for i, f, o in training)
    palette_shift = any(p['color'] not in train_colors for i, f, o in testing for p in i['pockets'])
    _require(count_shift or pocket_shift or palette_shift, 'missing visible train/test shift')


def _regressions():
    """Small pixel witnesses for detours and a door closed by the outline."""
    g = np.zeros((15, 18), np.int8)
    g[1:6, 1:8] = 6
    path = [(2, 2), (3, 2), (4, 2), (4, 3), (4, 4),
            (4, 5), (4, 6), (3, 6), (2, 6)]
    for q in path:
        g[q] = 0
    g[2, 2] = 2
    g[1:6, 11:16] = 6
    g[2:5, 12:15] = 0
    g[1, 13] = 0  # An orthogonal door; the bay is exterior in the INPUT.
    g[3, 13] = 2
    for q in ((8, 3), (9, 2), (9, 4), (10, 3)):
        g[q] = 2
    g[8:10, 8:10] = 2
    g[11, 14] = 4  # An unkeyed category never contributes to a wave.
    out = solve(g)
    _require([int(out[q]) for q in path] == [2, 0, 2, 0, 2, 0, 0, 0, 0],
             'three object-count bands must follow the bent path')
    _require(out[1, 13] == 6 and out[3, 13] == 2, 'original open-bay key must remain exterior')
    _require(np.all(out[8:11] == g[8:11]) and out[11, 14] == 4, 'inventory/distractor preservation')
    # Dihedral transforms and a full palette permutation preserve semantics.
    palette = np.array([9, 5, 1, 7, 0, 3, 8, 4, 2, 6], np.int8)
    for turn in range(4):
        for reflected in (False, True):
            a, b = np.rot90(g, turn), np.rot90(out, turn)
            if reflected:
                a, b = np.fliplr(a), np.fliplr(b)
            _require(np.array_equal(solve(a.copy()), b), 'rotation/reflection equivariance')
            _require(np.array_equal(solve(palette[a]), palette[b]), 'local palette equivariance')


def self_test():
    """Bounded returned-pixel checks and structural measurements on 42 seeds."""
    _regressions()
    puzzles = [generate_puzzle(s) for s in HOST_SEEDS]
    shapes, dimensions, inventories, roles, layouts = set(), set(), set(), set(), set()
    records = []
    for q in puzzles:
        validate_generated_puzzle(q)
        for split in q:
            for p in q[split]:
                info = _analyze(p['input'])
                f = _features(info)
                records.append(f)
                dimensions.add(p['input'].shape)
                shapes.update(f['shapes'])
                inventories.add(f['key_counts'])
                roles.add((info['background'], info['wall']))
                layouts.add(_signature(info['body']))
                _require(np.array_equal(solve(p['input'].copy()), p['output']), 'input-only prediction')
    _require(len(dimensions) >= 30 and len(shapes) >= 30, 'geometry diversity')
    _require(len(inventories) >= 15 and len(roles) >= 25, 'inventory/palette diversity')
    _require(len(layouts) >= 190, 'wall geometry/layout diversity')
    _require({f['pockets'] for f in records} >= {1, 2, 3}, 'pocket count diversity')
    _require({f['sealed_bodies'] for f in records} >= {1, 2, 3}, 'independent cavity ownership variation')
    _require({f['edge'] for f in records} == {True, False}, 'boundary variation')
    _require(any(5 in f['key_counts'] for f in records), 'count extrapolation coverage')
    _require(len({_signature(q['train'][0]['input']) for q in puzzles}) == 42, 'seed diversity')
    repeat = generate_puzzle(HOST_SEEDS[1])
    _require(all(np.array_equal(a[k], b[k]) for s in repeat
                 for a, b in zip(repeat[s], puzzles[1][s]) for k in a), 'determinism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            q = generate_puzzle(12345 + 10 * nt + nq, nt, nq)
            _require(len(q['train']) == nt and len(q['test']) == nq, 'explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Mutate several disjoint semantic output zones and require rejection.
    base = puzzles[0]
    info = _analyze(base['train'][0]['input'])
    zones = [info['outline'], info['body'], info['holes'],
             info['exterior'] & ~info['outline'] & (info['grid'] != info['background'])]
    for zone in zones:
        if not zone.any():
            continue
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        r, c = np.argwhere(zone)[0]
        out = bad['train'][0]['output']
        out[r, c] = (int(out[r, c]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted semantic zone accepted')
    return dict(seeds=42, pairs=len(records), input_shapes=len(dimensions),
                token_shapes=len(shapes), inventory_signatures=len(inventories),
                palette_roles=len(roles), wall_layouts=len(layouts),
                sealed_body_counts=sorted({f['sealed_bodies'] for f in records}))
