"""Routed color couriers: remote, simultaneous component-local transfers.

The public solver reads only its argument. Construction plans geometric links;
the reader restores glyphs and traces the actual visible field independently of
those plans. Neither parent module is imported or called.
"""
from __future__ import annotations

import json

import numpy as np

D = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid grid color")
    return a.astype(np.int8, copy=True)


def _add(p, d, n=1):
    return p[0] + D[d][0] * n, p[1] + D[d][1] * n


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(grid, excluded=()):
    """Orthogonal components; equal colors never imply equal identities."""
    labels = np.full(grid.shape, -1, dtype=np.int16)
    colors, cells = [], []
    excluded = set(excluded)
    for start in np.ndindex(grid.shape):
        if labels[start] >= 0 or int(grid[start]) in excluded:
            continue
        color, label = int(grid[start]), len(colors)
        colors.append(color)
        labels[start] = label
        todo, points = [start], []
        while todo:
            p = todo.pop()
            points.append(p)
            for d in range(4):
                q = _add(p, d)
                if _inside(q, grid.shape) and labels[q] < 0 and int(grid[q]) == color:
                    labels[q] = label
                    todo.append(q)
        cells.append(tuple(points))
    return labels, colors, cells


def _parse_header(grid):
    ink = int(grid[0, 0])
    _need(bool(np.all(grid[0] == ink)), "missing top key frame")
    rows = np.flatnonzero(np.all(grid[1:] == ink, axis=1)) + 1
    _need(len(rows) > 0, "missing bottom key frame")
    bottom = int(rows[0])
    _need(2 <= bottom < len(grid) - 1, "invalid key band")
    band = grid[1:bottom]
    _need(bool(np.all(band == band[0])), "key bars must fill the panel height")
    row = band[0]
    _need(int(row[0]) == int(row[-1]) == ink, "missing key side walls")
    runs, c = [], 0
    while c < len(row):
        if int(row[c]) == ink:
            c += 1
            continue
        start = c
        while c < len(row) and int(row[c]) != ink:
            c += 1
        runs.append((start, c))
    _need(bool(runs) and all(b - a >= 3 for a, b in runs), "ambiguous key interiors")
    background = int(row[runs[0][0] + 1])
    _need(background != ink, "background aliases arrow ink")
    turns = {}
    for a, b in runs:
        _need(bool(np.all(row[a + 1:b - 1] == background)), "inconsistent panel background")
        left, right = int(row[a]), int(row[b - 1])
        _need((left == background) != (right == background), "ambiguous turn bar")
        color, side = (right, 1) if left == background else (left, -1)
        _need(color not in turns and color not in (ink, background), "conflicting key")
        turns[color] = side
    return bottom + 1, ink, background, turns, tuple(b - a + 2 for a, b in runs)


def _glyph(center, d, shape):
    arms = (_add(center, (d + 1) % 4), _add(center, (d - 1) % 4))
    tip = _add(center, d)
    flanks = (_add(tip, (d + 1) % 4), _add(tip, (d - 1) % 4))
    back = _add(center, (d + 2) % 4)
    exit_cell = _add(tip, d)
    write = {center, tip, *arms}
    support = {back, *flanks}
    if not all(_inside(p, shape) for p in write | support | {exit_cell}):
        return None
    return dict(center=center, d=d, tip=tip, arms=arms, write=write,
                support=support, exit=exit_cell)


def _trace(field, labels, background, turns, tip, direction):
    """A one-cell courier turns BEFORE entering a switch; paths are invisible."""
    p, d = tip, direction
    seen, path, events = set(), [], []
    while True:
        state = (*p, d)
        _need(state not in seen, "cyclic courier trajectory")
        seen.add(state)
        q = _add(p, d)
        if not _inside(q, field.shape):
            return None, tuple(path), tuple(events)
        color = int(field[q])
        if color == background:
            p = q
            path.append(p)
        elif color in turns:
            new_d = (d + turns[color]) % 4
            events.append((color, d, new_d, p))
            d = new_d
        else:
            _need(int(labels[q]) >= 0, "courier encountered an unexplained cell")
            return int(labels[q]), tuple(path), tuple(events)


def _read(input_grid):
    grid = _grid(input_grid)
    crop, ink, background, turns, widths = _parse_header(grid)
    field = grid[crop:].copy()
    reserved = {ink, background, *turns}
    arrows = []
    centers = {_add(tuple(map(int, p)), d) for p in np.argwhere(field == ink) for d in range(4)}
    for center in sorted(centers):
        for d in range(4):
            a = _glyph(center, d, field.shape)
            if a is None or not all(int(field[p]) == ink for p in (*a['arms'], a['tip'])):
                continue
            fills = {int(field[p]) for p in a['support']}
            if len(fills) != 1 or next(iter(fills)) in reserved:
                continue
            host_color = next(iter(fills))
            junction = int(field[center])
            if junction != ink and junction in reserved:
                continue
            if int(field[a['exit']]) != background:
                continue
            arrows.append(dict(a, host_color=host_color, explicit=junction != ink,
                               payload=junction if junction != ink else host_color))
    _need(bool(arrows), "no readable courier arrows")
    used, covered = set(), set()
    for a in arrows:
        _need(not (used & a['write']), "overlapping or ambiguous T arrows")
        used.update(a['write'])
        covered.update(p for p in a['write'] if int(field[p]) == ink)
    actual = set(map(tuple, np.argwhere(field == ink)))
    _need(covered == actual, "unexplained arrow ink")
    clean = field.copy()
    for a in arrows:
        _need(not (a['support'] & used), "annotation hides required host evidence")
        for p in a['write']:
            clean[p] = a['host_color']
    switch_ids, switch_colors, switch_cells = _components(clean, excluded=set(range(10)) - set(turns))
    for points in switch_cells:
        rr, cc = zip(*points)
        _need(len(points) == (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1),
              "switch components must be solid rectangles")
    labels, colors, cells = _components(clean, excluded={background, *turns})
    _need(ink not in colors, "arrow ink survived restoration")
    assignments = {}
    for a in arrows:
        host = int(labels[a['center']])
        _need(host >= 0 and all(int(labels[p]) == host for p in a['write'] | a['support']),
              "arrow does not have one recoverable host")
        target, path, events = _trace(clean, labels, background, turns, a['tip'], a['d'])
        a.update(host=host, target=target, path=path, events=events)
        if target is not None:
            _need(target not in assignments or assignments[target] == a['payload'],
                  "conflicting deliveries to one original territory")
            assignments[target] = a['payload']
    output = clean.copy()
    for target, color in assignments.items():
        output[labels == target] = color
    info = dict(crop=crop, ink=ink, background=background, turns=turns, widths=widths,
                field=field, clean=clean, labels=labels, colors=colors, cells=cells,
                arrows=arrows, assignments=assignments, switches=switch_cells,
                switch_colors=switch_colors, switch_ids=switch_ids)
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the result entirely from the visible key, glyphs and components."""
    return _read(input_grid)[0]


def _features(info):
    colors, arrows, assigned = info['colors'], info['arrows'], info['assignments']
    final = [assigned.get(i, c) for i, c in enumerate(colors)]
    implicit = [a for a in arrows if not a['explicit']]
    chain = any(a['target'] == b['host'] and colors[a['host']] != colors[b['host']]
                and b['target'] is not None and a is not b for a in implicit for b in implicit)
    locality = any(colors[a] == colors[b] and final[a] != final[b]
                   for a in range(len(colors)) for b in range(a))
    bent_witness = False
    headings = {-1: set(), 1: set()}
    for a in arrows:
        for color, d, _, _ in a['events']:
            headings[info['turns'][color]].add(d)
        if a['target'] is not None and a['events']:
            # Removing switches entirely is a deliberately simpler alternative.
            bare = info['clean'].copy()
            bare[np.isin(bare, list(info['turns']))] = info['background']
            straight = _trace(bare, info['labels'], info['background'], {}, a['tip'], a['d'])[0]
            bent_witness |= straight != a['target']
    masks, oriented = [], []
    for a in arrows:
        masks.append(set(a['path']))
        axes = [set(), set()]
        prev = a['tip']
        for p in a['path']:
            axes[0 if p[0] != prev[0] else 1].add(p)
            prev = p
        for _, old, new, corner in a['events']:
            axes[old % 2].add(corner)
            axes[new % 2].add(corner)
        oriented.append(axes)
    crossing = any(oriented[i][0] & oriented[j][1] or oriented[i][1] & oriented[j][0]
                   for i in range(len(arrows)) for j in range(i))
    sharing = any(oriented[i][axis] & oriented[j][axis] for axis in (0, 1)
                  for i in range(len(arrows)) for j in range(i))
    return dict(chain=chain, locality=locality, bent_witness=bent_witness,
                explicit=any(a['explicit'] for a in arrows),
                singleton=any(a['explicit'] and a['payload'] not in colors for a in arrows),
                recursive=any(len(a['events']) >= 2 and a['target'] is not None for a in arrows),
                escaped=any(a['target'] is None for a in arrows),
                arrows=len(arrows), territories=len(colors), switches=len(info['switches']),
                repeated_switches=len(info['switch_colors']) > len(set(info['switch_colors'])),
                unused_keys=len(set(info['turns']) - set(info['switch_colors'])),
                max_turns=max(len(a['events']) for a in arrows),
                crossing=crossing, sharing=sharing,
                overlap=any(masks[i] & masks[j] for i in range(len(masks)) for j in range(i)),
                fanout=any(sum(a['host'] == h for a in arrows) >= 2 for h in range(len(colors))),
                nonrectangular=sum(len(s) != (max(p[0] for p in s) - min(p[0] for p in s) + 1) *
                                   (max(p[1] for p in s) - min(p[1] for p in s) + 1) for s in info['cells']),
                headings=headings, shape=info['clean'].shape)


def _variant(info, mode):
    """Counterfactual outputs for concrete shortcut checks, never for solving.

    Conflicts or cycles invalidate a hypothesis. For 'local', deliveries paint
    just their impact cell. For 'host', explicit junction payloads are ignored.
    The four turn hypotheses assign L/R meanings to the two visible bar sides.
    """
    field, turns = info['clean'].copy(), info['turns'].copy()
    if mode == 'straight':
        field[np.isin(field, list(turns))] = info['background']
        turns = {}
    elif isinstance(mode, tuple):
        mapping = {-1: mode[0], 1: mode[1]}
        turns = {c: mapping[side] for c, side in turns.items()}
    assignments, pixels = {}, {}
    for a in info['arrows']:
        target, path, events = _trace(field, info['labels'], info['background'], turns, a['tip'], a['d'])
        if target is None:
            continue
        payload = info['colors'][a['host']] if mode == 'host' else a['payload']
        if mode == 'local':
            direction = events[-1][2] if events else a['d']
            point = _add(path[-1] if path else a['tip'], direction)
            _need(point not in pixels or pixels[point] == payload, 'conflicting local hypothesis')
            pixels[point] = payload
        else:
            _need(target not in assignments or assignments[target] == payload, 'conflicting shortcut hypothesis')
            assignments[target] = payload
    out = info['clean'].copy()
    for target, color in assignments.items():
        out[info['labels'] == target] = color
    for point, color in pixels.items():
        out[point] = color
    return out


def _hypothesis_fits(info, output, mode):
    try:
        return bool(np.array_equal(_variant(info, mode), output))
    except AssertionError:
        return False


def _shape(rng, height, width, kind):
    mask = np.ones((height, width), dtype=bool)
    if kind == 'notch':
        mask[:int(rng.integers(1, height - 2)), :int(rng.integers(1, width - 2))] = False
    elif kind == 'step':
        for r in range(height):
            mask[r, :min(width - 3, r // 2)] = False
    elif kind == 'cross':
        mask[:] = False
        mask[1:-1, :] = True
        mask[:, 1:-1] = True
    elif kind == 'bevel':
        mask[0, 0] = mask[-1, -1] = False
    return np.rot90(mask, int(rng.integers(4))).copy()


def _scene(rng, n, test, style):
    h = int(rng.integers(22 if test else 18, 28 if test else 25))
    w = int(rng.integers(23 if test else 18, 31 if test else 28))
    labels = np.full((h, w), -1, dtype=np.int16)
    kinds = ['rectangle', 'notch', 'step', 'cross', 'bevel']
    for label in range(n):
        placed = False
        for _ in range(160):
            size = int(rng.integers(4, 7 if test else 6))
            other = int(rng.integers(4, 7 if test else 6))
            kind = str(rng.choice(kinds if style != 'rectilinear' else kinds[:3]))
            mask = _shape(rng, size, other, kind)
            mh, mw = mask.shape
            r = int(rng.integers(0, h - mh + 1))
            c = int(rng.integers(0, w - mw + 1))
            if np.any(labels[max(0, r - 1):min(h, r + mh + 1), max(0, c - 1):min(w, c + mw + 1)] >= 0):
                continue
            patch = labels[r:r + mh, c:c + mw]
            patch[mask] = label
            placed = True
            break
        if not placed:
            return None
    candidates, arrivals = [[] for _ in range(n)], [[] for _ in range(n)]
    for p in np.ndindex(labels.shape):
        host = int(labels[p])
        if host < 0:
            continue
        for d in range(4):
            a = _glyph(p, d, labels.shape)
            if a is not None and all(int(labels[q]) == host for q in a['write'] | a['support']) and labels[a['exit']] < 0:
                candidates[host].append(a)
            before = _add(p, (d + 2) % 4)
            if _inside(before, labels.shape) and labels[before] < 0:
                arrivals[host].append((p, d))
    if any(not a for a in candidates):
        return None
    return labels, candidates, arrivals


def _line(a, b):
    if a[0] != b[0] and a[1] != b[1]:
        return None
    length = abs(a[0] - b[0]) + abs(a[1] - b[1])
    if length == 0:
        return None
    dr = (b[0] > a[0]) - (b[0] < a[0])
    dc = (b[1] > a[1]) - (b[1] < a[1])
    d = D.index((dr, dc))
    return d, [(a[0] + dr * t, a[1] + dc * t) for t in range(1, length + 1)]


def _connect(rng, field, labels, candidates, arrivals, source, target, bends,
             turns, chosen, protected, background):
    """Construct a Manhattan connection by placing switches outside its corners.

    The points are only a construction device. The public reader does not know
    them, and each proposed connection is checked by actual cell tracing.
    """
    writes = set().union(*(a['write'] for a in chosen)) if chosen else set()
    contexts = set().union(*(a['support'] for a in chosen)) if chosen else set()
    halo = writes | {_add(p, d) for p in writes for d in range(4)}
    sources = [a for a in candidates[source] if not (a['write'] & (halo | contexts) or a['support'] & writes)]
    if not sources:
        return None
    for _ in range(280):
        a = sources[int(rng.integers(len(sources)))]
        end, last = arrivals[target][int(rng.integers(len(arrivals[target])))]
        start, first = a['tip'], a['d']
        if bends == 0:
            corners = []
        elif bends == 1:
            if last % 2 == first % 2:
                continue
            corners = [(start[0], end[1])] if first % 2 else [(end[0], start[1])]
        else:
            if last % 2 != first % 2:
                continue
            if first % 2:
                x = int(rng.integers(field.shape[1]))
                corners = [(start[0], x), (end[0], x)]
            else:
                y = int(rng.integers(field.shape[0]))
                corners = [(y, start[1]), (y, end[1])]
        points = [start, *corners, end]
        legs = [_line(p, q) for p, q in zip(points, points[1:])]
        if any(leg is None for leg in legs):
            continue
        directions = [leg[0] for leg in legs]
        if directions[0] != first or directions[-1] != last or any((y - x) % 4 not in (1, 3) for x, y in zip(directions, directions[1:])):
            continue
        path = set(p for _, cells in legs for p in cells)
        path.remove(end)
        if any(int(field[p]) != background for p in path):
            continue
        trial = field.copy()
        successful = True
        for i, corner in enumerate(corners):
            front = _add(corner, directions[i])
            side = 1 if (directions[i + 1] - directions[i]) % 4 == 1 else -1
            if not _inside(front, field.shape):
                successful = False
                break
            if int(trial[front]) in turns:
                if turns[int(trial[front])] != side:
                    successful = False
                    break
                continue
            if int(trial[front]) != background or front in protected | path:
                successful = False
                break
            color = int(rng.choice([color for color, turn in turns.items() if turn == side]))
            depth, span = int(rng.integers(1, 3)), int(rng.integers(1, 4))
            if directions[i] % 2:
                r0 = max(0, front[0] - int(rng.integers(span)))
                r1 = min(field.shape[0], r0 + span)
                c0, c1 = (front[1], min(field.shape[1], front[1] + depth)) if directions[i] == 1 else (max(0, front[1] - depth + 1), front[1] + 1)
            else:
                c0 = max(0, front[1] - int(rng.integers(span)))
                c1 = min(field.shape[1], c0 + span)
                r0, r1 = (front[0], min(field.shape[0], front[0] + depth)) if directions[i] == 2 else (max(0, front[0] - depth + 1), front[0] + 1)
            rect = {(r, c) for r in range(r0, r1) for c in range(c0, c1)}
            if (front not in rect or rect & (protected | path) or any(int(trial[p]) != background for p in rect)
                    or any(_inside(_add(p, d), trial.shape) and int(trial[_add(p, d)]) == color for p in rect for d in range(4))):
                # A one-cell rectangle is still a meaningful switch.
                rect = {front}
                if any(_inside(_add(front, d), trial.shape) and int(trial[_add(front, d)]) == color for d in range(4)):
                    successful = False
                    break
            for p in rect:
                trial[p] = color
        if not successful:
            continue
        try:
            hit, actual_path, events = _trace(trial, labels, background, turns, a['tip'], a['d'])
        except AssertionError:
            continue
        if hit != target or len(events) != bends:
            continue
        # Old routes cannot silently be changed by a new switch.
        stable = all(_trace(trial, labels, background, turns, b['tip'], b['d'])[0] == b['target'] for b in chosen)
        if stable:
            return trial, dict(a, host=source, target=target, path=actual_path, events=events)
    return None


def _header(rng, width, ink, background, turns, height):
    n = len(turns)
    widths = np.full(n, 5, dtype=int)
    widths += rng.multinomial(width - 5 * n, np.full(n, 1 / n))
    result = np.full((height, width), background, dtype=np.int8)
    result[[0, -1], :] = ink
    c = 0
    for color, size in zip(rng.permutation(list(turns)), widths):
        color, size = int(color), int(size)
        result[:, c] = result[:, c + size - 1] = ink
        result[1:-1, c + 1 if turns[color] == -1 else c + size - 2] = color
        c += size
    return result


def _make_pair(rng, test, lesson, params, table=None):
    for _ in range(100):
        n = int(rng.integers(6, 8) if test else rng.integers(4, 6))
        scene = _scene(rng, n, test, params['style'])
        if scene is None:
            continue
        labels, candidates, arrivals = scene
        if table is None:
            palette = list(map(int, rng.permutation(10)))
            ink, background = palette[:2]
            keys = palette[2:2 + int(rng.integers(2, 4))]
            turns = {color: (-1 if i == 0 else 1 if i == 1 else int(rng.choice([-1, 1]))) for i, color in enumerate(keys)}
        else:
            ink, background, turns = table
            keys = list(turns)
        available = [c for c in range(10) if c not in {ink, background, *keys}]
        rng.shuffle(available)
        singleton = available[-1]
        colors = [available[i % (len(available) - 1)] for i in range(n)]
        rng.shuffle(colors)
        order = list(map(int, rng.permutation(n)))
        a, b, c, untouched = order[:4]
        colors[a], colors[b], colors[c], colors[untouched] = available[0], available[1], available[2], available[1]
        if test and params['network'] == 'branch':
            # The two remote receivers begin alike but receive different fills.
            # Every mandatory branch delivery also changes its recipient.
            colors[untouched] = available[2]
        field = np.full(labels.shape, background, dtype=np.int8)
        for i, color in enumerate(colors):
            field[labels == i] = color
        chosen, protected = [], set()
        requests = []
        if lesson == 'chain':
            bends = [1, 2]
            rng.shuffle(bends)
            requests = [(a, b, bends[0], False), (b, c, bends[1], False)]
        elif lesson == 'payload':
            requests = [(a, b, 2, True), (a, c, 1, False)]
        elif test:
            if params['network'] == 'cycle':
                requests = [(a, b, 2, False), (b, c, 1, False), (c, a, 2, False)]
            else:
                requests = [(a, b, 1, False), (a, c, 2, False), (b, untouched, 1, False)]
        else:
            requests = [(a, b, int(rng.integers(0, 3)), bool(rng.random() < .45))]
        failed = False
        assignments = {}
        for source, target, bends, explicit in requests:
            connection = _connect(rng, field, labels, candidates, arrivals, source, target, bends,
                                  turns, chosen, protected, background)
            if connection is None:
                failed = True
                break
            field, arrow = connection
            payload = singleton if explicit else colors[source]
            if target in assignments and assignments[target] != payload:
                failed = True
                break
            assignments[target] = payload
            chosen.append(dict(arrow, payload=payload, explicit=explicit))
            protected.update(arrow['path'])
        if failed:
            continue
        desired = int(rng.integers(4, 7) if test else rng.integers(max(2, len(chosen)), 5))
        for _ in range(35):
            if len(chosen) >= desired:
                break
            source, target = map(int, rng.choice(n, size=2, replace=False))
            if target == untouched and lesson == 'chain':
                continue
            explicit = bool(rng.random() < params['explicit_rate'])
            payload = int(rng.choice(available)) if explicit else colors[source]
            if payload == colors[target] or (target in assignments and assignments[target] != payload):
                continue
            connection = _connect(rng, field, labels, candidates, arrivals, source, target,
                                  int(rng.choice([0, 1, 2], p=[.15, .45, .4])), turns, chosen, protected, background)
            if connection is None:
                continue
            field, arrow = connection
            assignments[target] = payload
            chosen.append(dict(arrow, payload=payload, explicit=explicit))
            protected.update(arrow['path'])
        # Sometimes an outward arrow delivers nothing: border exit is not a flood instruction.
        if lesson != 'chain' and rng.random() < .35:
            writes = set().union(*(x['write'] for x in chosen))
            contexts = set().union(*(x['support'] for x in chosen))
            halo = writes | {_add(p, d) for p in writes for d in range(4)}
            escaped = []
            for host in range(n):
                for arrow in candidates[host]:
                    if arrow['write'] & (halo | contexts) or arrow['support'] & writes:
                        continue
                    try:
                        hit, path, events = _trace(field, labels, background, turns, arrow['tip'], arrow['d'])
                    except AssertionError:
                        continue
                    if hit is None:
                        escaped.append(dict(arrow, host=host, target=None, path=path, events=events,
                                            payload=colors[host], explicit=False))
            if escaped:
                chosen.append(escaped[int(rng.integers(len(escaped)))])
        inp, out = field.copy(), field.copy()
        for arrow in chosen:
            for p in arrow['write']:
                inp[p] = ink
            if arrow['explicit']:
                inp[arrow['center']] = arrow['payload']
        for target, color in assignments.items():
            out[labels == target] = color
        height = int(rng.integers(3, min(5, 30 - field.shape[0]) + 1))
        inp = np.vstack((_header(rng, field.shape[1], ink, background, turns, height), inp))
        try:
            predicted, info = _read(inp)
        except AssertionError:
            continue
        if not np.array_equal(predicted, out) or not np.array_equal(info['clean'], field):
            continue
        features = _features(info)
        if not features['bent_witness']:
            continue
        if lesson == 'chain' and not (features['chain'] and features['locality']):
            continue
        if lesson == 'payload' and not (features['singleton'] and features['recursive'] and features['fanout']):
            continue
        if test and any(_hypothesis_fits(info, out, mode) for mode in ('straight', 'local')):
            continue
        if test and any(features[name] and not params.get(name + '_taught', True) for name in ('crossing', 'escaped')):
            continue
        return dict(input=inp, output=out), info
    raise AssertionError('bounded courier construction exhausted')


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _teaching(infos):
    fs = [_features(q) for q in infos]
    headings = {side: set().union(*(f['headings'][side] for f in fs)) for side in (-1, 1)}
    return (all(any(f[name] for f in fs) for name in ('chain', 'locality', 'singleton', 'recursive', 'bent_witness', 'fanout'))
            and all(len(headings[side]) >= 2 for side in (-1, 1)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _integer(num_train) or not _integer(num_test) or not 2 <= num_train <= 10
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('counts require 2..10 train, 1..4 test, and num_train > num_test')
    rng = np.random.default_rng(int(seed))
    params = dict(style=str(rng.choice(['rectilinear', 'sculpted'])),
                  network=str(rng.choice(['cycle', 'branch'])), explicit_rate=float(rng.uniform(.15, .65)))
    # Shared colors in the first two demonstrations change meaning in the visible key.
    palette = list(map(int, rng.permutation(10)))
    table = palette[0], palette[1], {palette[2]: -1, palette[3]: 1}
    first, first_info = _make_pair(rng, False, 'chain', params, table)
    second_table = table[0], table[1], {color: -side for color, side in table[2].items()}
    for _ in range(24):
        second, second_info = _make_pair(rng, False, 'payload', params, second_table)
        fitting = [mode for mode in ((-1, -1), (-1, 1), (1, -1), (1, 1))
                   if _hypothesis_fits(first_info, first['output'], mode) and
                   _hypothesis_fits(second_info, second['output'], mode)]
        if _teaching([first_info, second_info]) and fitting == [(-1, 1)]:
            break
    else:
        raise AssertionError('could not demonstrate both relative turn meanings')
    puzzle = {'train': [first, second], 'test': []}
    seen = {_signature(p['input']) for p in puzzle['train']}
    for split, count in (('train', int(num_train) - 2), ('test', int(num_test))):
        if split == 'test':
            taught = [_features(_read(p['input'])[1]) for p in puzzle['train']]
            params.update({name + '_taught': any(f[name] for f in taught) for name in ('crossing', 'escaped')})
        for _ in range(count):
            for _ in range(16):
                pair, info = _make_pair(rng, split == 'test', '', params)
                key = _signature(pair['input'])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise AssertionError('unique input construction exhausted')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _signature(inp)
            _need(key not in seen, "duplicate input")
            seen.add(key)
            predicted, info = _read(inp)
            _need(np.array_equal(predicted, out), "incorrect delivery result")
            _need(out.shape == info['clean'].shape, "incorrect key-band crop")
            structural = (info['labels'] < 0)
            _need(np.array_equal(out[structural], info['clean'][structural]), "a path or switch was painted")
            for i, points in enumerate(info['cells']):
                expected = info['assignments'].get(i, info['colors'][i])
                _need(all(int(out[p]) == expected for p in points), "territory was only partly recolored")
            _need(bool(info['assignments']), "example has no delivery")
            _need(any(color != info['colors'][i] for i, color in info['assignments'].items()), "no region changes")
            infos[split].append(info)
    _need(_teaching(infos['train']), "training lacks necessary mechanism contrasts")
    _need(all(_features(q)['bent_witness'] for q in infos['test']), "test can ignore routing")
    _need(all(not _hypothesis_fits(q, np.asarray(p['output']), mode)
              for q, p in zip(infos['test'], puzzle['test']) for mode in ('straight', 'local')),
          'omitting a parent mechanism leaves a test answer unchanged')
    for feature in ('crossing', 'escaped'):
        _need(not any(_features(q)[feature] for q in infos['test']) or
              any(_features(q)[feature] for q in infos['train']), 'test mechanism is undemonstrated: ' + feature)
    tables = [q['turns'] for q in infos['train']]
    _need(any(len({t[c] for t in tables if c in t}) > 1 for c in range(10)), "training can memorize key colors")
    fitting = [mode for mode in ((-1, -1), (-1, 1), (1, -1), (1, 1))
               if all(_hypothesis_fits(q, np.asarray(p['output']), mode)
                      for q, p in zip(infos['train'], puzzle['train']))]
    _need(fitting == [(-1, 1)], 'relative bar meanings are not uniquely demonstrated')
    for mode in ('straight', 'local', 'host'):
        _need(any(not _hypothesis_fits(q, np.asarray(p['output']), mode)
                  for q, p in zip(infos['train'], puzzle['train'])),
              'training cannot distinguish shortcut: ' + mode)
    return None


def _encode(puzzle):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
                       for s in ('train', 'test')}, separators=(',', ':'))


def self_test():
    """Bounded semantic and diversity checks over 42 seeds, plus count boundaries."""
    # Hand-constructed, three-turn route, beyond the connector's two-turn plans.
    # A second disconnected receiver of the same fill is deliberately untouched.
    field = np.zeros((14, 16), dtype=np.int8)
    field[6:10, :4] = 7
    field[10:14, 4:8] = field[:4, 12:16] = 6
    field[7, 9] = field[3, 3] = 2
    field[2, 8] = 4
    expected = field.copy()
    expected[10:14, 4:8] = 7
    inp = field.copy()
    for p in _glyph((7, 2), 1, field.shape)['write']:
        inp[p] = 9
    header = np.array([[9] * 16, [9, 2, 0, 0, 0, 0, 0, 9, 9, 4, 0, 0, 0, 0, 0, 9], [9] * 16], dtype=np.int8)
    routed = np.vstack((header, inp))
    out, info = _read(routed)
    _need(np.array_equal(out, expected) and len(info['arrows'][0]['events']) == 3,
          'manual recursive/locality fixture failed')
    cyclic = routed.copy()
    cyclic[3 + 8, 4] = 4
    try:
        solve(cyclic)
    except AssertionError:
        pass
    else:
        raise AssertionError('cyclic route accepted')
    # Consistent repeated arrivals are allowed; inconsistent arrivals are not.
    field = np.zeros((12, 18), dtype=np.int8)
    field[4:8, :4] = field[4:8, 14:] = 7
    field[4:8, 7:11] = 6
    expected = field.copy()
    expected[4:8, 7:11] = 7
    inp = field.copy()
    for center, direction in (((5, 2), 1), ((6, 15), 3)):
        for p in _glyph(center, direction, field.shape)['write']:
            inp[p] = 9
    header = _header(np.random.default_rng(999), 18, 9, 0, {2: -1, 4: 1}, 3)
    incoming = np.vstack((header, inp))
    _need(np.array_equal(solve(incoming), expected), 'consistent duplicate arrivals rejected')
    contradictory = incoming.copy()
    contradictory[3 + 6, 15] = 8
    try:
        solve(contradictory)
    except AssertionError:
        pass
    else:
        raise AssertionError('conflicting literal delivery accepted')
    features, puzzles, structures, shapes = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_encode(puzzle))
        if seed < 2:
            _need(_encode(puzzle) == _encode(generate_puzzle(seed)), "nondeterminism")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                out, info = _read(pair['input'])
                _need(np.array_equal(solve(np.asarray(pair['input']).copy()), out), "input-only solve failure")
                f = _features(info)
                features.append(f)
                shapes.add(f['shape'])
                structures.add(_signature(info['labels']))
                # A geometry-preserving palette permutation must commute with solving.
                permutation = np.roll(np.arange(10), 3)
                _need(np.array_equal(solve(permutation[pair['input']]), permutation[out]), "literal-color dependency")
    _need(len(puzzles) == 42 and len(structures) > 100 and len(shapes) > 35, "collapsed structural diversity")
    for name in ('chain', 'locality', 'singleton', 'recursive', 'fanout', 'bent_witness',
                 'repeated_switches', 'nonrectangular', 'escaped', 'crossing'):
        _need(sum(bool(f[name]) for f in features) >= 3, 'missing variation: ' + name)
    _need(len({f['arrows'] for f in features}) >= 4, "arrow count does not vary")
    _need(len({f['switches'] for f in features}) >= 5, "switch count does not vary")
    # The host verifier exhausts all 30 legal combinations. Keep this local
    # 42-seed check bounded by adding only the meaningful count boundaries.
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(700 + 10 * nt + nq, nt, nq)
        _need(len(p['train']) == nt and len(p['test']) == nq, "counts were ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return None


if __name__ == '__main__':
    self_test()
    print('self_test passed')
