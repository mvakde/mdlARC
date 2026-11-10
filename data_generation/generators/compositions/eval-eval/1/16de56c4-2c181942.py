"""Terminal-switched combs: cadence controls color-indexed rigid assembly.

Only NumPy and the standard library are used. Prediction reads the input alone;
sampling and puzzle-level distribution choices are not available to solve().
"""

import numpy as np


_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((0 <= g) & (g <= 9))), "invalid colors")
    return g.astype(np.int8, copy=True)


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(g, background):
    """Eight-connectivity across colors: a rail is one multicolored object."""
    unseen = set(map(tuple, np.argwhere(g != background)))
    result = []
    for start in sorted(unseen):
        if start not in unseen:
            continue
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        coords = np.array(cells, dtype=int)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        mask = np.zeros(tuple(hi - lo + 1), dtype=bool)
        mask[coords[:, 0] - lo[0], coords[:, 1] - lo[1]] = True
        result.append(dict(coords=coords, origin=lo, mask=mask))
    return result


def _connected(mask, diagonal=True):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    stack = [cells.pop()]
    neighbors = _NEIGHBORS if diagonal else ((-1, 0), (1, 0), (0, -1), (0, 1))
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            q = (r + dr, c + dc)
            if q in cells:
                cells.remove(q)
                stack.append(q)
    return not cells


def _attachments(mask):
    results = {}
    for t in range(4):
        a = np.rot90(mask, t).copy()
        h, w = a.shape
        if h < 2 or w < 2 or w % 2:
            continue
        neck = np.zeros(w, dtype=bool)
        neck[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(a[-1], neck):
            k = _key(a)
            if k not in results:
                results[k] = [a, []]
            results[k][1].append(t)
    return list(results.values())


def _canonical(mask):
    _require(_connected(mask), "disconnected template")
    candidates = _attachments(mask)
    _require(len(candidates) == 1, "missing or competing attachment results")
    return candidates[0]


def _edge_cells(rail, start):
    r, c, side = rail['row'], rail['col'], rail['side']
    if side == 0:
        return np.array(((r, c + start), (r, c + start + 1)))
    if side == 2:
        return np.array(((r + 1, c + start), (r + 1, c + start + 1)))
    if side == 3:
        return np.array(((r + start, c), (r + start + 1, c)))
    return np.array(((r + start, c + 1), (r + start + 1, c + 1)))


def _dock(mask, rail, start):
    side = rail['side']
    marker = np.zeros_like(mask)
    w = mask.shape[1]
    marker[-1, w // 2 - 1:w // 2 + 1] = True
    oriented = np.rot90(mask, -side)
    neck = np.argwhere(np.rot90(marker, -side))
    normals = ((-1, 0), (0, 1), (1, 0), (0, -1))
    target = _edge_cells(rail, start) + normals[side]
    return np.argwhere(oriented) + target.min(axis=0) - neck.min(axis=0)


def _decode_rail(block, origin, horizontal):
    edges = (block[0], block[1]) if horizontal else (block[:, 0], block[:, 1])
    backing = [i for i, edge in enumerate(edges) if np.all(edge == edge[0])]
    _require(len(backing) == 1, "rail needs one uniform backing edge")
    inner = backing[0]
    outer = 1 - inner
    material = int(edges[inner][0])
    values = edges[outer]
    sockets, p = [], 0
    while p < len(values):
        if values[p] == material:
            p += 1
            continue
        color, end = int(values[p]), p + 1
        while end < len(values) and values[end] == color:
            end += 1
        _require(end - p == 2, "socket must be a domino")
        sockets.append((p, color))
        p = end
    _require(len(sockets) >= 2, "missing reference sockets")
    _require(all(b[0] - a[0] >= 3 for a, b in zip(sockets, sockets[1:])),
             "input sockets need a material separator")
    counts = {}
    for pos, color in sockets:
        counts.setdefault(color, []).append(pos)
    repeats = [c for c, ps in counts.items() if len(ps) > 1]
    _require(len(repeats) == 1 and len(counts[repeats[0]]) == 2,
             "rail needs exactly one repeated socket color")
    reference = repeats[0]
    a, b = counts[reference]
    period = b - a
    extras = [(pos, color) for pos, color in sockets if color != reference]
    aligned = [(pos, color) for pos, color in extras if (pos - a) % period == 0]
    _require(len(aligned) <= 1, "multiple on-phase terminals")
    terminal = aligned[0] if aligned else None
    _require(terminal is None or terminal[0] < a or terminal[0] > b,
             "terminal must lie beyond the pair")
    full = list(range(a % period, len(values) - 1, period))
    starts = full if terminal is None else [x for x in full if min(a, terminal[0]) <= x <= max(b, terminal[0])]
    inert = [(pos, color) for pos, color in extras if (pos - a) % period]
    _require(all(abs(pos - x) >= 2 for pos, _ in inert for x in full),
             "inert socket overlaps a phase domino")
    side = (0 if outer == 0 else 2) if horizontal else (3 if outer == 0 else 1)
    return dict(row=int(origin[0]), col=int(origin[1]), side=side,
                length=len(values), material=material, sockets=sockets,
                reference=reference, a=a, b=b, period=period,
                phase=a % period, terminal=terminal, inert=inert, full=full,
                starts=starts, selected=reference if terminal is None else terminal[1])


def _read_scene(value):
    g = _grid(value)
    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > g.size // 2, "background must be a strict majority")
    rails, templates = [], {}
    for comp in _components(g, bg):
        mask = comp['mask']
        h, w = mask.shape
        r, c = comp['origin']
        block = g[r:r + h, c:c + w]
        is_rail = ((h == 2 and w >= 9) or (w == 2 and h >= 9)) and bool(mask.all())
        if is_rail:
            rail = _decode_rail(block, (r, c), h == 2)
            rail['coords'] = comp['coords']
            rails.append(rail)
        else:
            _require(min(h, w) >= 2 and max(h, w) <= 6, "invalid library silhouette dimensions")
            palette = set(map(int, g[comp['coords'][:, 0], comp['coords'][:, 1]]))
            _require(len(palette) == 1, "library objects must be monochrome")
            color = palette.pop()
            _require(color not in templates, "one template per color is required")
            canonical, turns = _canonical(mask)
            templates[color] = dict(mask=canonical, input_mask=mask, undo_turns=turns,
                                    coords=comp['coords'])
    _require(1 <= len(rails) <= 3 and bool(templates), "need 1..3 rails and a library")
    _require(len({r['material'] for r in rails}) == 1, "rails must share material color")
    _require(rails[0]['material'] not in templates, "material is not a template label")
    for rail in rails:
        _require(all(color in templates for _, color in rail['sockets']), "socket has no template")
        mask = templates[rail['selected']]['mask']
        _require(rail['period'] >= mask.shape[1] + 1, "cadence too short for selected teeth")
    return g, bg, rails, templates


def _reconstruct(g, bg, rails, templates, *, socket_only=False, reference_sites=False,
                 paint_reference_shape=False):
    result = np.full_like(g, bg)
    for rail in rails:
        rr, cc = rail['coords'].T
        result[rr, cc] = g[rr, cc]
    for rail in rails:
        color = rail['selected']
        template = rail['reference'] if paint_reference_shape else color
        mask = templates[template]['mask']
        starts = [rail['a'], rail['b']] if reference_sites else rail['starts']
        for x in starts:
            rr, cc = _edge_cells(rail, x).T
            result[rr, cc] = color
            if socket_only:
                continue
            coords = _dock(mask, rail, x)
            rr, cc = coords.T
            _require(bool(np.all((rr >= 0) & (rr < g.shape[0]) & (cc >= 0) & (cc < g.shape[1]))),
                     "tooth would be clipped")
            _require(bool(np.all(result[rr, cc] == bg)), "assembled teeth/supports overlap")
            result[rr, cc] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all output pixels solely from the visible input."""
    g, bg, rails, templates = _read_scene(input_grid)
    return _reconstruct(g, bg, rails, templates)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _mask(*rows):
    return np.array([[c == '#' for c in row] for row in rows], dtype=bool)


def _shape(rng, width=None, directional=False, rich=True, asymmetric=False):
    """Procedural silhouettes; accept geometry, not a secretly selected neck."""
    for _ in range(100):
        w = int(width if width is not None else _pick(rng, (2, 4, 4, 6)))
        h = int(rng.integers(3 if asymmetric else 2, 6 if rich else 5))
        kind = 3 if asymmetric else int(rng.integers(4))
        middle = w // 2
        a = np.zeros((h, w), dtype=bool)
        a[:, middle - 1:middle + 1] = True
        stem = int(rng.integers(1, h))
        body = h - stem
        if w == 2:
            pass
        elif kind <= 1:
            a[:body] = True
        elif kind == 2:
            wing = min(w // 2 - 1, h - 1)
            a[:] = False
            a[wing:, middle - 1:middle + 1] = True
            for r in range(wing):
                reach = wing - r
                a[r, middle - 1 - reach] = True
                a[r, middle + reach] = True
                if rng.random() < .35:
                    a[r, middle - 1 - reach:middle + reach + 1] = True
        else:
            for r in range(body):
                left = int(rng.integers(middle))
                right = int(rng.integers(middle + 1, w + 1))
                a[r, left:right] = True
            a[int(rng.integers(body)), :middle] = True
            a[int(rng.integers(body)), middle:] = True
            for r, c in np.argwhere(a[:body]):
                if c not in (middle - 1, middle) and rng.random() < .2:
                    a[r, c] = False
        if not a[:, 0].any() or not a[:, -1].any() or not _connected(a):
            continue
        candidates = _attachments(a)
        if len(candidates) != 1:
            continue
        canonical, turns = candidates[0]
        if width is not None and canonical.shape[1] != width:
            continue
        if directional and len(turns) != 1:
            continue
        if asymmetric and np.array_equal(canonical, np.fliplr(canonical)):
            continue
        return canonical
    if asymmetric:
        return _mask('###.', '####', '.##.')
    if width == 2 and not directional:
        return _mask('##', '##', '##')
    w = 4 if width is None or width == 2 else width
    return _mask('#' * w, '.' * (w // 2 - 1) + '##' + '.' * (w // 2 - 1))


def _sample_unit(rng, side, ref, terminal_color, library, material,
                 *, witness=False, direction=1, mixed=False, force_inert=None):
    selected = ref if terminal_color is None else terminal_color
    width = library[selected]['mask'].shape[1]
    # Witness references and selected bodies have widths 2 and 4; nonwitness
    # geometry may use any valid period without a puzzle-wide cap.
    minimum = width + 1
    lo = max(12 if mixed else 14, 2 * minimum + 2)
    if witness:
        lo = max(lo, 4 * minimum + 2)
    high = 18 if mixed else 26
    length = int(rng.integers(lo, high + 1))
    max_period = (length - 2) // (4 if witness else 2)
    period = int(rng.integers(minimum, max_period + 1))
    phases = [phase for phase in range(period)
              if len(range(phase, length - 1, period)) >= (5 if witness else 3)]
    phase = int(_pick(rng, phases))
    sites = list(range(phase, length - 1, period))
    n = len(sites)
    if terminal_color is None:
        indices = list(range(1, n - 2)) if witness else list(range(n - 1))
        k = int(_pick(rng, indices))
        a, b, t = sites[k], sites[k + 1], None
    else:
        options = []
        for k in range(n - 1):
            for j in range(n):
                if (direction > 0 and j <= k + 1) or (direction < 0 and j >= k):
                    continue
                low, high_index = min(k, j), max(k + 1, j)
                if witness and (high_index - low < 3 or (low == 0 and high_index == n - 1)):
                    continue
                options.append((k, j))
        k, j = _pick(rng, options)
        a, b, t = sites[k], sites[k + 1], sites[j]
    sockets = [(a, ref), (b, ref)]
    if t is not None:
        sockets.append((t, terminal_color))
    options = [x for x in range(length - 1)
               if (x - phase) % period and all(abs(x - y) >= 2 for y in sites)]
    extra_colors = [c for c in library if c not in (ref, terminal_color)]
    rng.shuffle(extra_colors)
    if force_inert is not None:
        extra_colors.remove(force_inert)
        extra_colors.insert(0, force_inert)
    extra_count = min(len(extra_colors), int(rng.integers(3)))
    if force_inert is not None:
        extra_count = max(1, extra_count)
    for color in extra_colors[:extra_count]:
        available = [x for x in options if all(abs(x - y) >= 3 for y, _ in sockets)]
        if not available:
            break
        sockets.append((int(_pick(rng, available)), color))
    if force_inert is not None and not any(c == force_inert for _, c in sockets):
        return None
    rail = dict(row=0, col=0, side=side, length=length)
    shape = (2, length) if side % 2 == 0 else (length, 2)
    inp = {(r, c): material for r in range(shape[0]) for c in range(shape[1])}
    for x, color in sockets:
        for r, c in _edge_cells(rail, x):
            inp[(int(r), int(c))] = color
    out = dict(inp)
    starts = sites if t is None else [x for x in sites if min(a, t) <= x <= max(b, t)]
    for x in starts:
        for r, c in _edge_cells(rail, x):
            out[(int(r), int(c))] = selected
        for r, c in _dock(library[selected]['mask'], rail, x):
            q = (int(r), int(c))
            if q in out:
                raise RuntimeError('sampled teeth overlap')
            out[q] = selected
    coords = np.array(list(out), dtype=int)
    low, high_coord = coords.min(axis=0), coords.max(axis=0)
    def normalize(cells):
        return {(r - int(low[0]), c - int(low[1])): color for (r, c), color in cells.items()}
    return dict(input=normalize(inp), output=normalize(out),
                shape=tuple(map(int, high_coord - low + 1)), horizontal=side % 2 == 0)


def _box_positions(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    windows = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(windows == 0)


def _shelf_positions(units):
    """Guaranteed packing: parallel strips or short perpendicular groups."""
    horizontal = [i for i, u in enumerate(units) if u['horizontal']]
    vertical = [i for i, u in enumerate(units) if not u['horizontal']]
    positions = [None] * len(units)
    if not vertical:
        row = 0
        for i in horizontal:
            positions[i] = (row, 0)
            row += units[i]['shape'][0] + 1
    elif not horizontal:
        col = 0
        for i in vertical:
            positions[i] = (0, col)
            col += units[i]['shape'][1] + 1
    elif len(vertical) == 1:
        row = 0
        for i in horizontal:
            positions[i] = (row, 0)
            row += units[i]['shape'][0] + 1
        col = max(units[i]['shape'][1] for i in horizontal) + 1
        positions[vertical[0]] = (0, col)
    else:
        col = 0
        for i in vertical:
            positions[i] = (0, col)
            col += units[i]['shape'][1] + 1
        row = max(units[i]['shape'][0] for i in vertical) + 1
        positions[horizontal[0]] = (row, 0)
    _require(all(r + units[i]['shape'][0] <= 30 and c + units[i]['shape'][1] <= 30
                 for i, (r, c) in enumerate(positions)), "shelf construction exceeded canvas")
    return positions


def _pack_units(rng, units, force_large=False):
    if force_large:
        return (30, 30), _shelf_positions(units)
    minimum = np.max(np.array([u['shape'] for u in units]), axis=0)
    shape = tuple(int(rng.integers(max(19, x), 31)) for x in minimum)
    if rng.random() < .2:
        shape = (max(shape), max(shape))
    order = sorted(range(len(units)), key=lambda i: -np.prod(units[i]['shape']))
    for _ in range(18):
        blocked = np.zeros(shape, dtype=bool)
        positions = [None] * len(units)
        for i in order:
            h, w = units[i]['shape']
            available = _box_positions(blocked, h, w)
            if not len(available):
                break
            r, c = map(int, available[int(rng.integers(len(available)))])
            positions[i] = (r, c)
            blocked[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True
        else:
            return shape, positions
    return (30, 30), _shelf_positions(units)


def _block_neighbors(blocked, coords):
    for r, c in coords:
        blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True


def _place_library(rng, g, bg, library):
    blocked = np.zeros(g.shape, dtype=bool)
    _block_neighbors(blocked, np.argwhere(g != bg))
    order = list(library)
    rng.shuffle(order)
    order.sort(key=lambda c: -library[c]['mask'].size)
    for color in order:
        entry = library[color]
        mask = np.rot90(entry['mask'], entry['input_turns'])
        h, w = mask.shape
        local = np.argwhere(mask)
        if h > g.shape[0] or w > g.shape[1]:
            return False
        tops, lefts = np.indices((g.shape[0] - h + 1, g.shape[1] - w + 1))
        rr = tops.ravel()[:, None] + local[:, 0]
        cc = lefts.ravel()[:, None] + local[:, 1]
        available = np.flatnonzero(~blocked[rr, cc].any(axis=1))
        if not len(available):
            return False
        index = int(_pick(rng, available))
        g[rr[index], cc[index]] = color
        _block_neighbors(blocked, np.column_stack((rr[index], cc[index])))
    return np.count_nonzero(g == bg) > g.size // 2


def _build_pair(rng, palette, rich, *, witness_group=None, test=False):
    bg, material, *labels = palette
    count = 2 if witness_group is not None else int(_pick(rng, (2, 2, 3) if test else (1, 2, 2, 3)))
    if witness_group == 0:
        sides = [0, 2]
    elif witness_group == 1:
        sides = [1, 3]
    elif test:
        sides = [int(_pick(rng, (0, 2))), int(_pick(rng, (1, 3)))]
        if count == 3:
            sides.append(int(rng.integers(4)))
        # Preserve a shared-label pair on different axes; vary its order.
        if rng.random() < .5:
            sides[0], sides[1] = sides[1], sides[0]
    else:
        axis = int(rng.integers(2))
        sides = [int(_pick(rng, (0, 2) if axis == 0 else (1, 3))) for _ in range(count)]
    num_colors = int(rng.integers(4, 7))
    library = {}
    for i, color in enumerate(labels[:num_colors]):
        forced = 4 if witness_group is not None and i in (0, 2) else 2 if witness_group is not None and i == 1 else None
        library[color] = dict(mask=_shape(rng, forced, directional=forced == 4, rich=rich,
                                          asymmetric=witness_group is not None and i == 2),
                              input_turns=int(rng.integers(4)))
    if witness_group is not None:
        for i, label_index in enumerate((0, 2)):
            required_turn = i + 2 * witness_group
            library[labels[label_index]]['input_turns'] = (-sides[i] - required_turn) % 4
    units = []
    for i, side in enumerate(sides):
        witness = witness_group is not None
        if witness:
            ref = labels[0] if i == 0 else labels[1]
            terminal_color = None if i == 0 else labels[2]
            direction = 1 if witness_group == 0 else -1
        elif test and i < 2:
            ref = labels[0]
            terminal_color = None if i == 0 else labels[2]
            direction = int(_pick(rng, (-1, 1)))
        else:
            ref = int(_pick(rng, list(library)))
            terminal_color = None if rng.random() < .5 else int(_pick(rng, [c for c in library if c != ref]))
            direction = int(_pick(rng, (-1, 1)))
        inert = labels[3] if (witness and i == 0) or (test and i == 0) else None
        unit = _sample_unit(rng, side, ref, terminal_color, library, material,
                            witness=witness, direction=direction, mixed=test, force_inert=inert)
        if unit is None:
            return None
        units.append(unit)
    for attempt in range(6):
        shape, positions = _pack_units(rng, units, force_large=attempt >= 3)
        inp, out = (np.full(shape, bg, dtype=np.int8) for _ in range(2))
        for unit, (row, col) in zip(units, positions):
            for key, array in (('input', inp), ('output', out)):
                for (r, c), color in unit[key].items():
                    array[r + row, c + col] = color
        if not _place_library(rng, inp, bg, library):
            continue
        # Labels were rendered from sampled geometry, not by solve().
        _require(np.array_equal(solve(inp), out), "sampled answer differs from input reconstruction")
        return dict(input=inp, output=out)
    return None


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .3)
    base_palette = list(map(int, rng.permutation(10)[:8]))
    rich = bool(rng.random() < .7)
    witnesses = list(map(int, rng.choice(num_train, 2, replace=False)))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            group = witnesses.index(i) if split == 'train' and i in witnesses else None
            for _ in range(80):
                palette = base_palette if stable else list(map(int, rng.permutation(10)[:8]))
                pair = _build_pair(rng, palette, rich, witness_group=group, test=split == 'test')
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    g, bg, rails, library = _read_scene(pair['input'])
    expected = _reconstruct(g, bg, rails, library)
    if not np.array_equal(expected, _grid(pair['output'])):
        raise AssertionError('incorrect socket completion, template switch, or rigid assembly')
    rotations = set()
    for rail in rails:
        template = library[rail['selected']]
        if len(template['undo_turns']) == 1:
            rotations.update((t - rail['side']) % 4 for t in template['undo_turns'])
        for x, color in rail['inert']:
            rr, cc = _edge_cells(rail, x).T
            assert np.all(expected[rr, cc] == color), 'inert socket was repainted'
        if rail['terminal'] is not None:
            for x in set(rail['full']) - set(rail['starts']):
                rr, cc = _edge_cells(rail, x).T
                assert np.all(expected[rr, cc] == rail['material']), 'terminal failed to stop'
    return dict(grid=g, background=bg, rails=rails, library=library, rotations=rotations,
                sides={r['side'] for r in rails},
                boundary=any(r['terminal'] is None for r in rails),
                stopped=any(r['terminal'] is not None and set(r['full']) - set(r['starts']) for r in rails),
                switched=any(r['terminal'] is not None and _key(library[r['reference']]['mask']) != _key(library[r['selected']]['mask']) for r in rails),
                both_ends=any(r['terminal'] is None and min(r['starts']) < r['a'] and max(r['starts']) > r['b'] for r in rails),
                inert=sum(len(r['inert']) for r in rails),
                added=sum(len(set(r['starts']) - {x for x, _ in r['sockets']}) for r in rails),
                unused=set(library) - {r['selected'] for r in rails})


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    features, seen = {'train': [], 'test': []}, set()
    for split in features:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape and not np.array_equal(g, out)
            assert _key(g) not in seen, 'duplicate input'
            seen.add(_key(g))
            f = _features(pair)
            assert f['unused'], 'unused legitimate templates should demonstrate erasure'
            assert not np.array_equal(out, _reconstruct(g, f['background'], f['rails'], f['library'], socket_only=True)), 'B has no effect'
            features[split].append(f)
    train = features['train']
    assert set().union(*(f['sides'] for f in train)) == set(range(4)), 'directions not demonstrated'
    assert set().union(*(f['rotations'] for f in train)) == set(range(4)), 'rotations not demonstrated'
    for name in ('boundary', 'stopped', 'switched', 'both_ends', 'inert', 'added'):
        assert any(f[name] for f in train), f'missing training distinction: {name}'
    assert all(len({r['side'] % 2 for r in f['rails']}) == 1 for f in train)
    assert any(not np.array_equal(_grid(p['output']), _reconstruct(f['grid'], f['background'], f['rails'], f['library'], reference_sites=True))
               for p, f in zip(puzzle['train'], train)), 'A cadence ablation has no effect'
    assert any(f['switched'] for f in train), 'terminal type override has no geometric effect'
    assert any(not np.array_equal(f['library'][r['selected']]['mask'],
                                 np.fliplr(f['library'][r['selected']]['mask']))
               for f in train for r in f['rails']), 'handedness must be demonstrated'
    for f in features['test']:
        rails = f['rails']
        assert {r['side'] % 2 for r in rails} == {0, 1}, 'missing mixed-axis transfer'
        assert any(a['reference'] == b['reference'] and a['side'] % 2 != b['side'] % 2
                   and (a['terminal'] is None) != (b['terminal'] is None)
                   for a in rails for b in rails), 'missing shared-label mode contrast'
        assert f['inert'], 'test needs an inert available-template socket'


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def _manual_fixture():
    """Hand-specified answer: period five, bounded switched diagonal wings."""
    g = np.zeros((12, 29), dtype=np.int8)
    g[7:9, 3:25] = 8
    # Reference starts 5/10, terminal 20; slot 0 stays material, slot 15 is new.
    g[7, 8:10] = 1
    g[7, 13:15] = 1
    g[7, 23:25] = 2
    g[7, 5:7] = 3  # Start 2 is off phase, even though color 3 has a template.
    g[0, 1:5] = 1
    g[1, 2:4] = 1
    # Clockwise orientation of #..# / .##. / .##., entered as coordinates.
    g[0, 10] = 2
    g[1:3, 8:10] = 2
    g[3, 10] = 2
    g[0:2, 15:17] = 3
    out = np.zeros_like(g)
    out[7:9, 3:25] = 8
    out[7, 5:7] = 3
    for x in (8, 13, 18, 23):
        out[7, x:x + 2] = 2
        out[5:7, x:x + 2] = 2
        out[4, x - 1] = out[4, x + 2] = 2
    return g, out


def self_test() -> dict:
    g, answer = _manual_fixture()
    original = g.copy()
    assert np.array_equal(solve(g), answer), 'hand-specified switched-wing fixture'
    assert np.array_equal(g, original), 'solve mutated its input'
    scene, bg, rails, library = _read_scene(g)
    for ablation in ('socket_only', 'reference_sites', 'paint_reference_shape'):
        assert not np.array_equal(_reconstruct(scene, bg, rails, library, **{ablation: True}), answer), ablation
    for t in range(4):
        for invert in (False, True):
            a, b = np.rot90(g, t), np.rot90(answer, t)
            if invert:
                a, b = 9 - a, 9 - b
            assert np.array_equal(solve(a), b), 'rotation/palette equivariance'
    assert len(_canonical(_mask('##', '##'))[1]) == 4
    assert len(_canonical(_mask('##', '##', '##'))[1]) == 2
    assert _connected(_mask('#..#', '.##.')) and not _connected(_mask('#..#', '.##.'), False)
    assert len(_attachments(_mask('.###', '####', '#.##', '.##.'))) > 1
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    shapes, masks, periods, phases, counts, backgrounds, modes, directions = set(), set(), set(), set(), set(), set(), set(), set()
    diagonal = asymmetric = shared = 0
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(pair['input'].shape)
                counts.add(len(f['rails']))
                backgrounds.add(f['background'])
                refs = [r['reference'] for r in f['rails']]
                shared += len(refs) != len(set(refs))
                modes.add(tuple(sorted(r['terminal'] is not None for r in f['rails'])))
                for r in f['rails']:
                    periods.add(r['period'])
                    phases.add(r['phase'])
                    directions.add(0 if r['terminal'] is None else 1 if r['terminal'][0] > r['b'] else -1)
                for entry in f['library'].values():
                    m = entry['mask']
                    masks.add(_key(m))
                    diagonal += not _connected(m, False)
                    asymmetric += not np.array_equal(m, np.fliplr(m))
    assert len(shapes) >= 25 and len(masks) >= 35
    assert {3, 5, 6, 7} <= periods and len(phases) >= 5
    assert counts == {1, 2, 3} and len(backgrounds) == 10
    assert directions == {-1, 0, 1} and len(modes) >= 4
    assert diagonal >= 10 and asymmetric >= 5 and shared >= 42
    assert _puzzle_key(generate_puzzle(123, 5, 2)) == _puzzle_key(generate_puzzle(123, 5, 2))
    # Bounded API checks; the host verifier additionally covers every legal count.
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(900 + a + b, a, b)
        assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    damaged = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('damaged answer accepted')
    return dict(seeds=42, distinct_puzzles=42, canvas_shapes=len(shapes), canonical_silhouettes=len(masks),
                periods=sorted(periods), phases=sorted(phases), rail_counts=sorted(counts),
                diagonal_templates=diagonal, asymmetric_templates=asymmetric, shared_label_scenes=shared)
