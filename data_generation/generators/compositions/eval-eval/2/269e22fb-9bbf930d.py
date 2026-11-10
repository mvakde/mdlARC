"""Couriers between registered, recolored views of one visible rail map.

Only standard-library modules and NumPy are used. No parent module, puzzle,
seed table, learned bitmap, or generation state is consulted by solve().
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

OUTSIDE, TOKEN, FRAME = 0, 8, 9
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


class _Retry(Exception):
    pass


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _need(np.issubdtype(a.dtype, np.integer), 'integer grid required')
    _need(bool(np.all((a >= 0) & (a <= 9))), 'colors must be 0..9')
    return a.astype(np.int8, copy=True)


def _d4(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _flanks(g, p, d, bg):
    n = (-d[1], d[0])
    a, b = _add(p, n), _add(p, (-n[0], -n[1]))
    if not (_inside(g, a) and _inside(g, b)):
        return None
    x, y = int(g[a]), int(g[b])
    return (x, y) if x != bg and y != bg else None


def _walk(g, start, direction, bg):
    p, d = start, direction
    path, seen, turns = [p], set(), 0
    while True:
        _need((p, d) not in seen, 'route cycle')
        seen.add((p, d))
        q = _add(p, d)
        if not _inside(g, q):
            break
        if g[q] == bg:
            p = q
            path.append(p)
            continue
        choices = []
        for nd in ((-d[1], d[0]), (d[1], -d[0])):
            q = _add(p, nd)
            if _inside(g, q) and g[q] == bg:
                sides = _flanks(g, q, nd, bg)
                if sides is not None and sides[0] == sides[1]:
                    choices.append(nd)
        _need(len(choices) <= 1, 'two supported turns')
        if not choices:
            break
        d = choices[0]
        turns += 1
    _need(len(path) == len(set(path)), 'self-intersection')
    return dict(terminal=p, path=path, turns=turns,
                open=p[0] in (0, len(g) - 1) or p[1] in (0, g.shape[1] - 1))


def _route(g, start, direction, bg, ignore_equality=False):
    """First two-sided opening selects; then motion retains its heading."""
    _need(g[start] == bg, 'token annotation must cover background')
    p, opening = start, None
    while _inside(g, p) and g[p] == bg:
        sides = _flanks(g, p, direction, bg)
        if sides is not None:
            opening = sides
            break
        p = _add(p, direction)
    active = opening is not None and (opening[0] == opening[1] or ignore_equality)
    result = dict(start=start, direction=direction, opening=opening, active=active)
    # Reading the hypothetical inactive route also checks the generated decoys.
    if opening is not None:
        result.update(_walk(g, start, direction, bg))
    return result


def _panels(g):
    """Read isolated rectangular 9-frames; no assumed page layout."""
    remaining = set(map(tuple, np.argwhere(g == FRAME)))
    panels = []
    while remaining:
        first = remaining.pop()
        stack, cells = [first], {first}
        while stack:
            p = stack.pop()
            for d in DIRECTIONS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        rs, cs = zip(*cells)
        r, c, R, C = min(rs), min(cs), max(rs), max(cs)
        _need(R - r >= 3 and C - c >= 3, 'frame too small')
        boundary = {(y, x) for y in range(r, R + 1) for x in range(c, C + 1)
                    if y in (r, R) or x in (c, C)}
        _need(cells == boundary, 'frames must be complete rectangles')
        inner = g[r + 1:R, c + 1:C].copy()
        _need(bool(np.all((inner >= 1) & (inner <= TOKEN))), 'panel colors')
        panels.append(dict(origin=(r + 1, c + 1), grid=inner))
    _need(len(panels) >= 3, 'a map and at least two observation windows are required')
    panels.sort(key=lambda p: (-p['grid'].size, p['origin']))
    _need(panels[0]['grid'].size > panels[1]['grid'].size, 'unique largest reference')
    mask = np.zeros(g.shape, dtype=bool)
    for panel in panels:
        r, c = panel['origin']
        h, w = panel['grid'].shape
        _need(not np.any(mask[r - 1:r + h + 1, c - 1:c + w + 1]), 'overlapping panels')
        mask[r - 1:r + h + 1, c - 1:c + w + 1] = True
    _need(bool(np.all(g[~mask] == OUTSIDE)), 'outside the frames must be black')
    return panels


def _registrations(master, view, bg):
    """Exact color-isomorphic crops under D4; 8 annotates background only.

    Equality classes are compared directly, rather than trying color names.
    Each result supplies an original-map cell ID for every local view cell.
    """
    h, w = view.shape
    groups = [np.flatnonzero(view.ravel() == color) for color in np.unique(view)
              if color != TOKEN]
    if len(groups) < 2:
        return []
    annotations = np.flatnonzero(view.ravel() == TOKEN)
    identifiers = np.arange(master.size).reshape(master.shape)
    found = []
    for t in range(8):
        full = _d4(master, t)
        if h > full.shape[0] or w > full.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
        flat = windows.reshape(windows.shape[:2] + (h * w,))
        valid = np.ones(flat.shape[:2], dtype=bool)
        representatives = []
        for indices in groups:
            rep = flat[..., indices[0]]
            valid &= np.all(flat[..., indices] == rep[..., None], axis=-1)
            for previous in representatives:
                valid &= rep != previous
            representatives.append(rep)
        if len(annotations):
            valid &= np.all(flat[..., annotations] == bg, axis=-1)
        ids = _d4(identifiers, t)
        for r, c in np.argwhere(valid):
            r, c = int(r), int(c)
            patch = full[r:r + h, c:c + w]
            mapping = {int(patch.ravel()[ix[0]]): int(view.ravel()[ix[0]]) for ix in groups}
            # Background must be visibly represented, not an unseen color role.
            if bg not in mapping:
                continue
            found.append(dict(ids=ids[r:r + h, c:c + w].copy(), mapping=mapping,
                              transform=t, offset=(r, c)))
    return found


def _read(input_grid, ignore_equality=False):
    g = _grid(input_grid)
    panels = _panels(g)
    master = panels[0]['grid']
    _need(not np.any(master == TOKEN), 'reference must be unmarked')
    corners = [master[0, 0], master[0, -1], master[-1, 0], master[-1, -1]]
    _need(len(set(map(int, corners))) == 1, 'reference corners identify its background')
    bg = int(corners[0])
    _need(len(np.unique(master)) >= 3, 'two guide identities are required')
    for panel in panels[1:]:
        matches = _registrations(master, panel['grid'], bg)
        _need(len(matches) == 1, 'observation must have exactly one registration')
        panel.update(matches[0])
    token_panels = [i for i in range(1, len(panels)) if np.any(panels[i]['grid'] == TOKEN)]
    _need(len(token_panels) == 1, 'one departure window is required')
    source_index = token_panels[0]
    source = panels[source_index]
    routes = []
    for r, c in np.argwhere(source['grid'] == TOKEN):
        r, c = int(r), int(c)
        h, w = source['grid'].shape
        dirs = [d for yes, d in ((c == 0, (0, 1)), (c == w - 1, (0, -1)),
                                (r == 0, (1, 0)), (r == h - 1, (-1, 0))) if yes]
        _need(len(dirs) == 1, 'tokens must occupy non-corner boundary cells')
        nr, nc = r + dirs[0][0], c + dirs[0][1]
        start_id, next_id = int(source['ids'][r, c]), int(source['ids'][nr, nc])
        start = divmod(start_id, master.shape[1])
        nxt = divmod(next_id, master.shape[1])
        direction = (nxt[0] - start[0], nxt[1] - start[1])
        route = _route(master, start, direction, bg, ignore_equality)
        route.update(local_start=(r, c), source=source_index)
        if route.get('terminal') is not None:
            terminal_id = np.ravel_multi_index(route['terminal'], master.shape)
            landings = [(i, (int(y), int(x))) for i, p in enumerate(panels[1:], 1)
                        for y, x in np.argwhere(p['ids'] == terminal_id)]
            _need(len(landings) == 1, 'each terminal must be visible in exactly one observation')
            route['landing'] = landings[0]
            _need(landings[0][0] != source_index, 'terminal must be outside departure view')
        routes.append(route)
    _need(any(r['active'] for r in routes), 'at least one mover required')
    return dict(grid=g, panels=panels, master=master, background=bg, routes=routes)


def _answer(a):
    out = a['grid'].copy()
    for route in a['routes']:
        if not route['active']:
            continue
        source = a['panels'][route['source']]
        r, c = route['local_start']
        y, x = source['origin']
        out[y + r, x + c] = source['mapping'][a['background']]
        index, (r, c) = route['landing']
        y, x = a['panels'][index]['origin']
        out[y + r, x + c] = TOKEN
    return out


def _features(a):
    features = {'matching', 'mismatch'}
    for route in a['routes']:
        if not route['active']:
            continue
        features.add('open' if route['open'] else 'closed')
        if route['turns']:
            features.add('turn')
        path = route['path']
        for p, q, r in zip(path, path[1:], path[2:]):
            d = (q[0] - p[0], q[1] - p[1])
            if (r[0] - q[0], r[1] - q[1]) != d:
                continue
            before = _flanks(a['master'], p, d, a['background'])
            middle = _flanks(a['master'], q, d, a['background'])
            after = _flanks(a['master'], r, d, a['background'])
            if (before is not None and before[0] == before[1] and middle is None
                    and after is not None and after[0] == after[1]):
                features.add('gap')
    return features


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, crop registrations, headings and terminals from input."""
    return _answer(_read(input_grid))


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = (int(np.sign(target[0] - p[0])), int(np.sign(target[1] - p[1])))
        if (d[0] == 0) == (d[1] == 0):
            return None
        while p != target:
            p = _add(p, d)
            path.append(p)
    cells = set(path)
    if len(cells) != len(path):
        return None
    if any(sum(_add(p, d) in cells for d in DIRECTIONS) !=
           (1 if i in (0, len(path) - 1) else 2) for i, p in enumerate(path)):
        return None
    return path


def _random_path(rng, H, W, start, width, bends, closed):
    points, d = [start], (0, 1)
    lower_col = start[1] + width + 1
    for segment in range(bends + 1):
        p = points[-1]
        if segment == bends and not closed and d != (0, -1):
            target = ((H - 1 if d[0] > 0 else 0), p[1]) if d[0] else (p[0], W - 1)
        else:
            options = []
            for length in range(2, max(H, W)):
                q = (p[0] + length * d[0], p[1] + length * d[1])
                if 1 <= q[0] < H - 1 and lower_col <= q[1] < W - 1:
                    options.append(q)
            if not options:
                return None
            target = options[int(rng.integers(len(options)))]
        if target == p:
            return None
        points.append(target)
        d = (-d[1], d[0]) if rng.integers(2) else (d[1], -d[0])
    return _polyline(points)


def _walls(rng, path, start_col, color, inactive, H, W, guide_count):
    cells = set(path)
    walls = {}
    for p in path:
        for d in DIRECTIONS:
            q = _add(p, d)
            if 0 <= q[0] < H and start_col + 1 <= q[1] < W and q not in cells:
                walls[q] = color
    # Independent approach lengths make the crop register at a real offset.
    for sign in (-1, 1):
        begin = start_col + int(rng.integers(1, 3))
        for q in list(walls):
            if q[0] == path[0][0] + sign and q[1] < begin:
                del walls[q]
    if inactive:
        other = int(rng.choice([c for c in range(1, guide_count + 1) if c != color]))
        for q in walls:
            if q[0] > path[0][0]:
                walls[q] = other
    else:
        steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        eligible = [i for i in range(3, len(path) - 3)
                    if len(set(steps[i - 2:i + 2])) == 1 and path[i][1] > start_col + 2]
        if eligible and rng.random() < .45:
            i = int(rng.choice(eligible))
            d = steps[i]
            n = (-d[1], d[0])
            for q in (_add(path[i], n), _add(path[i], (-n[0], -n[1]))):
                walls.pop(q, None)
        # A closed cap can have a different identity; it is still a blocker.
        cap = _add(path[-1], steps[-1])
        if cap in walls and rng.random() < .4:
            walls[cap] = int(rng.integers(1, guide_count + 1))
    # The four visible reference corners always retain background.
    for corner in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        walls.pop(corner, None)
    return walls


def _make_map(rng, active_count, inactive_count, source_width, bent):
    count = active_count + inactive_count
    H = 12 if count == 4 else int(rng.integers(10, 13))
    W = int(rng.integers(12, 17))
    s = int(rng.integers(1, 3))
    guide_count = int(rng.integers(2, 4))
    slots = list(range(1, H - 1, 3))
    if len(slots) < count:
        raise _Retry()
    rows = sorted(map(int, rng.choice(slots, count, replace=False)))
    flags = [True] * active_count + [False] * inactive_count
    rng.shuffle(flags)
    plan = list(zip(rows, flags))
    # Pack bends first, but do not allocate a separate band for their motion.
    order = sorted(range(count), key=lambda i: not flags[i])
    g = np.zeros((H, W), dtype=np.int8)
    reserved, records = set(), {}
    for ordinal, i in enumerate(order):
        row, active = plan[i]
        force = active and bent and ordinal == 0
        for attempt in range(100):
            bends = int(rng.integers(1, 4)) if force else (int(rng.integers(0, 4)) if active and rng.random() < .4 else 0)
            closed = bool(rng.integers(2))
            path = _random_path(rng, H, W, (row, s), source_width, bends, closed)
            if path is None or any(p in reserved or g[p] != 0 for p in path):
                continue
            walls = _walls(rng, path, s, int(rng.integers(1, guide_count + 1)),
                           not active, H, W, guide_count)
            if any(q in reserved or (g[q] != 0 and g[q] != v) for q, v in walls.items()):
                continue
            candidate = g.copy()
            for q, value in walls.items():
                candidate[q] = value
            wanted = dict(records)
            wanted[i] = dict(start=(row, s), active=active, path=path, terminal=path[-1])
            try:
                for record in wanted.values():
                    actual = _route(candidate, record['start'], (0, 1), 0)
                    if actual['active'] != record['active'] or actual.get('path') != record['path']:
                        raise _Retry()
            except (ValueError, _Retry):
                continue
            g, reserved, records = candidate, reserved | set(path), wanted
            break
        else:
            raise _Retry()
    records = [records[i] for i in range(count)]
    if len({r['terminal'] for r in records}) != count:
        raise _Retry()
    if len(_registrations(g, g, 0)) != 1:
        raise _Retry()
    top, bottom = rows[0] - 1, rows[-1] + 2
    top = int(rng.integers(0, top + 1))
    bottom = int(rng.integers(bottom, H + 1))
    source = g[top:bottom, s:s + source_width].copy()
    ids = np.arange(g.size).reshape(g.shape)[top:bottom, s:s + source_width].copy()
    for row in rows:
        source[row - top, 0] = TOKEN
    if len(_registrations(g, source, 0)) != 1:
        raise _Retry()
    return g, source, ids, records


def _receivers(rng, master, records, desired):
    """Crop charts with a unique owner for every real or hypothetical terminal."""
    H, W = master.shape
    terminals = [r['terminal'] for r in records]
    groups = [list(range(len(terminals)))]
    if desired == 2:
        choices = []
        for axis in (0, 1):
            values = sorted({p[axis] for p in terminals})
            for v in values[:-1]:
                left = [i for i, p in enumerate(terminals) if p[axis] <= v]
                right = [i for i in range(len(terminals)) if i not in left]
                choices.append([left, right])
        if not choices:
            raise _Retry()
        groups = choices[int(rng.integers(len(choices)))]
    ids = np.arange(master.size).reshape(master.shape)
    crops = []
    for group in groups:
        ys, xs = zip(*(terminals[i] for i in group))
        others = [terminals[i] for i in range(len(terminals)) if i not in group]
        candidates = []
        for attempt in range(100):
            # Context width varies, and may grow to full-span strips.
            r = int(rng.integers(0, min(ys) + 1))
            R = int(rng.integers(max(ys) + 1, H + 1))
            c = int(rng.integers(0, min(xs) + 1))
            C = int(rng.integers(max(xs) + 1, W + 1))
            if R - r < 2 or C - c < 2 or (R - r) * (C - c) == master.size:
                continue
            if any(r <= y < R and c <= x < C for y, x in others):
                continue
            candidates.append((r, R, c, C))
        # Smallest unique contexts are easier to fit and harder to identify by size.
        candidates = sorted(set(candidates), key=lambda b: (b[1] - b[0]) * (b[3] - b[2]))
        for r, R, c, C in candidates:
            patch = master[r:R, c:C]
            if len(_registrations(master, patch, 0)) == 1:
                crops.append((patch.copy(), ids[r:R, c:C].copy()))
                break
        else:
            raise _Retry()
    return crops


def _pack(rng, inputs, outputs):
    """Place rectangles without a fixed panel order or row/column template."""
    sizes = [(a.shape[0] + 2, a.shape[1] + 2) for a in inputs]
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0] * sizes[i][1])
    for attempt in range(60):
        occupied = np.zeros((30, 30), dtype=bool)
        origins = {}
        for index in order:
            h, w = sizes[index]
            options = []
            for r in range(1, 30 - h):
                for c in range(1, 30 - w):
                    if not np.any(occupied[r - 1:r + h + 1, c - 1:c + w + 1]):
                        options.append((r, c))
            if not options:
                break
            r, c = options[int(rng.integers(len(options)))]
            occupied[r:r + h, c:c + w] = True
            origins[index] = (r, c)
        if len(origins) == len(inputs):
            break
    else:
        raise _Retry()
    a, b = np.zeros((30, 30), dtype=np.int8), np.zeros((30, 30), dtype=np.int8)
    for index, (r, c) in origins.items():
        h, w = sizes[index]
        for canvas, content in ((a, inputs[index]), (b, outputs[index])):
            canvas[r:r + h, c:c + w] = FRAME
            canvas[r + 1:r + h - 1, c + 1:c + w - 1] = content
    ys, xs = np.nonzero(a)
    bounds = (slice(max(0, int(ys.min()) - 1), min(30, int(ys.max()) + 2)),
              slice(max(0, int(xs.min()) - 1), min(30, int(xs.max()) + 2)))
    a, b = a[bounds].copy(), b[bounds].copy()
    rotation = int(rng.integers(4))
    return {'input': np.rot90(a, rotation).copy(), 'output': np.rot90(b, rotation).copy()}


def _make_pair(rng, index, test, stable_palette):
    active = int(rng.integers(2, 4)) if test else (1 if index == 0 else 2 if index == 1 else int(rng.integers(1, 4)))
    inactive = 1 if active > 1 else int(rng.integers(1, 3))
    width = 3 if test else int(rng.integers(4, 6))
    bent = index == 1 if not test else bool(rng.integers(2))
    receiver_count = 2 if not test and index == 1 else int(rng.integers(1, 3))
    if not test and index == 0:
        receiver_count = 1
    for attempt in range(160):
        try:
            master, source, source_ids, records = _make_map(rng, active, inactive, width, bent)
            receivers = _receivers(rng, master, records, receiver_count)
            abstract = [(master, np.arange(master.size).reshape(master.shape)), (source, source_ids), *receivers]
            panels, ids, palettes = [], [], []
            for i, (patch, coordinates) in enumerate(abstract):
                t = 0 if i == 0 else int(rng.integers(8))
                if not test and index == 0 and i == 2:
                    t = 4  # demonstrate reflection even with only two pairs
                if not test and index == 1 and i == 1:
                    t = 1  # demonstrate a quarter turn independently
                palette = stable_palette if stable_palette is not None else np.asarray(rng.permutation(np.arange(1, 8)))
                colored = np.full(patch.shape, TOKEN, dtype=np.int8)
                known = patch != TOKEN
                colored[known] = palette[patch[known]]
                panels.append(_d4(colored, t))
                ids.append(_d4(coordinates, t))
                palettes.append(palette)
            answers = [p.copy() for p in panels]
            for record in records:
                if not record['active']:
                    continue
                start_id = np.ravel_multi_index(record['start'], master.shape)
                terminal_id = np.ravel_multi_index(record['terminal'], master.shape)
                starts = np.argwhere(ids[1] == start_id)
                _need(len(starts) == 1, 'source construction')
                r, c = starts[0]
                answers[1][r, c] = palettes[1][0]
                locations = [(i, y, x) for i in range(1, len(ids)) for y, x in np.argwhere(ids[i] == terminal_id)]
                if len(locations) != 1 or locations[0][0] == 1:
                    raise _Retry()
                i, r, c = locations[0]
                answers[i][r, c] = TOKEN
            pair = _pack(rng, panels, answers)
            a = _read(pair['input'])
            if not np.array_equal(_answer(a), pair['output']):
                raise _Retry()
            paths = [p for route in a['routes'] for p in route['path']]
            if len(paths) != len(set(paths)):
                raise _Retry()
            # Every selected decoy has a visible, unique hypothetical landing.
            if any(route['opening'] is None for route in a['routes']):
                raise _Retry()
            if bent and not any(r['active'] and r['turns'] for r in a['routes']):
                raise _Retry()
            features = _features(a)
            if not test and index == 0 and 'gap' not in features:
                raise _Retry()
            if not test and index == 1 and not {'turn', 'open', 'closed'} <= features:
                raise _Retry()
            return pair
        except (_Retry, ValueError):
            continue
    raise RuntimeError('bounded map/view construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        _need(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
              and low <= int(value) <= high, name + ' is outside its legal integer range')
    _need(num_train > num_test, 'num_train must exceed num_test')
    _need(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
          'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    stable = np.asarray(rng.permutation(np.arange(1, 8))) if rng.random() < .3 else None
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for retry in range(8):
                pair = _make_pair(rng, index, split == 'test', stable)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('duplicate-input retry bound exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, train, coverage = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'canvas dimensions must be preserved')
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            a = _read(inp)
            _need(np.array_equal(_answer(a), out), 'wrong transfers')
            moving = [r for r in a['routes'] if r['active']]
            still = [r for r in a['routes'] if not r['active']]
            _need(bool(moving and still), 'matched and mismatched couriers required')
            _need(all(r['opening'] is not None for r in still), 'true mismatches required')
            _need(np.count_nonzero(inp == TOKEN) == np.count_nonzero(out == TOKEN), 'token conservation')
            _need(np.count_nonzero(inp != out) == 2 * len(moving), 'only two cells per mover change')
            _need(np.array_equal(inp[inp == FRAME], out[inp == FRAME]), 'frame preservation')
            p = a['panels'][0]
            y, x = p['origin']
            h, w = p['grid'].shape
            _need(np.array_equal(out[y:y + h, x:x + w], p['grid']), 'reference preservation')
            endpoints = [r['terminal'] for r in a['routes']]
            _need(len(endpoints) == len(set(endpoints)), 'unique terminals')
            paths = [p for r in a['routes'] for p in r['path']]
            _need(len(paths) == len(set(paths)), 'separated centerlines')
            _need(not np.array_equal(_answer(_read(inp, True)), out), 'rail equality must affect output')
            # A tempting shortcut routes onto the large reference itself. It
            # must fail: landing coordinates belong to registered small views.
            wrong = inp.copy()
            my, mx = a['panels'][0]['origin']
            for route in moving:
                sy, sx = a['panels'][route['source']]['origin']
                r, c = route['local_start']
                wrong[sy + r, sx + c] = a['panels'][route['source']]['mapping'][a['background']]
                r, c = route['terminal']
                wrong[my + r, mx + c] = TOKEN
            _need(not np.array_equal(wrong, out), 'view registration must affect output')
            if split == 'train':
                train.append(a)
                coverage.update(_features(a))
            else:
                _need(_features(a) <= coverage, 'test routing convention absent from training')
    _need(any(r['turns'] for a in train for r in a['routes'] if r['active']), 'training must show a bend')
    _need({1, 2} <= {len(a['panels']) - 2 for a in train}, 'one and two landing windows must be demonstrated')
    _need({1, 2} <= {sum(r['active'] for r in a['routes']) for a in train}, 'single and multiple transfers')
    _need({'gap', 'turn', 'open', 'closed'} <= coverage, 'demonstrated routing primitives')


def _signature(puzzle):
    return tuple(_key(pair[k]) for s in ('train', 'test') for pair in puzzle[s] for k in ('input', 'output'))


def self_test():
    """Stream 42 seeds, retaining only small signatures and feature sets."""
    signatures, shapes, maps, registrations, palettes, geometries = set(), set(), set(), set(), set(), set()
    token_counts, receiver_counts, openings, turns, terminal_types = set(), set(), set(), set(), set()
    checked, bent, reflected, shifted = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(hashlib.sha256(repr(_signature(puzzle)).encode()).hexdigest())
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _read(pair['input'])
                checked += 1
                shapes.add(pair['input'].shape)
                m = a['master']
                # Color- and D4-normalized reference content, not literal hashes.
                canonical = []
                for t in range(8):
                    roles = {}
                    flat = bytes(roles.setdefault(int(v), len(roles)) for v in _d4(m, t).flat)
                    canonical.append((_d4(m, t).shape, flat))
                maps.add(min(canonical))
                token_counts.add(len(a['routes']))
                receiver_counts.add(len(a['panels']) - 2)
                for p in a['panels'][1:]:
                    registrations.add(p['transform'])
                    palettes.add(tuple(sorted(p['mapping'].items())))
                    reflected += p['transform'] >= 4
                    shifted += p['offset'] != (0, 0)
                for route in a['routes']:
                    openings.add(route['opening'][0] == route['opening'][1])
                    if route['active']:
                        turns.add(route['turns'])
                        bent += route['turns'] > 0
                        terminal_types.add(route['open'])
                geometries.add(tuple(sorted((r['active'], len(r['path']), r['turns'], r['open']) for r in a['routes'])))
                _need(np.array_equal(solve(pair['input'].tolist()), pair['output']), 'nested-list solve')
        if seed == 0:
            _need(_signature(puzzle) == _signature(generate_puzzle(seed)), 'determinism')
            bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            bad['test'][0]['output'][0, 0] = FRAME
            try:
                validate_generated_puzzle(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('output corruption accepted')
    _need(len(signatures) == 42 and len(maps) >= 150, 'reference/content diversity')
    _need(len(shapes) >= 15 and len(geometries) >= 100, 'structural diversity')
    _need(registrations == set(range(8)) and reflected and shifted, 'registration diversity')
    _need(len(palettes) >= 50 and openings == {False, True}, 'role/equality diversity')
    _need({2, 3, 4} <= token_counts and receiver_counts == {1, 2}, 'count diversity')
    _need({0, 1, 2} <= turns and terminal_types == {False, True} and bent, 'route diversity')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(20261022, nt, nq)
            _need(len(p['train']) == nt and len(p['test']) == nq, 'explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    return {'seeds': 42, 'pairs': checked, 'distinct_puzzles': len(signatures),
            'reference_classes': len(maps), 'canvas_shapes': len(shapes),
            'route_signatures': len(geometries), 'view_transforms': sorted(registrations),
            'token_counts': sorted(token_counts), 'landing_window_counts': sorted(receiver_counts),
            'active_turn_counts': sorted(turns), 'terminal_types': sorted(terminal_types)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
