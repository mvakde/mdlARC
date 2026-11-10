"""First-contact harvest: cap endpoints address unary stacks.

One collector, arbitrary staggered receiver stacks, and simultaneous first-hit
selection. No parent imports, stored puzzles, or generation state in solve().
"""
from __future__ import annotations

from collections import defaultdict
import json
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _outline(center, radius):
    y, x = center
    return {(y + dy, x + dx) for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1) if abs(dy) + abs(dx) == radius}


def _cap(center, radius, direction):
    y, x = center
    dy, dx = DIRECTIONS[direction]
    return [(y + dy * (k + 1), x + dx * (radius - k + 1))
            for k in range(radius + 1)]


def _lines(center, radius, caps, shape, whole=False):
    result = []
    for direction in sorted(caps):
        cap = _cap(center, radius, direction)
        origins = cap if whole else (cap[0], cap[-1])
        dy, dx = DIRECTIONS[direction]
        for origin in origins:
            p = (origin[0] + dy, origin[1] + dx)
            path = []
            while _inside(p, shape):
                path.append(p)
                p = (p[0] + dy, p[1] + dx)
            result.append((direction, path))
    return result


def _trace(lines, bars):
    paths, contacts = [], []
    for direction, line in lines:
        path, hit = [], None
        for p in line:
            if p in bars:
                hit = bars[p]
                break
            path.append(p)
        paths.append((direction, path))
        contacts.append(hit)
    return paths, contacts


def _token_cells(stack, count=None):
    count = stack['count'] if count is None else count
    h = stack['mask'].shape[0]
    offsets = np.argwhere(stack['mask']).tolist()
    return {(stack['bottom'] - k * (h + 1) + y, stack['x'] + x)
            for k in range(count) for y, x in offsets}


def _bar_cells(stack):
    return {(stack['bar'], stack['x'] + x) for x in range(stack['mask'].shape[1])}


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, queue = {start}, [start]
        while queue:
            y, x = queue.pop()
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    queue.append(p)
        result.append(component)
    return result


def _parse(input_grid):
    """Recover shapes, controls, destination, and pitch from this input only."""
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > a.size // 2, 'background must have a strict majority'
    fg = set(map(tuple, np.argwhere(a != bg)))
    emitters = []
    for radius in (2, 3):
        for y in range(radius + 1, a.shape[0] - radius - 1):
            for x in range(radius + 1, a.shape[1] - radius - 1):
                outline = _outline((y, x), radius)
                color = int(a[next(iter(outline))])
                if color == bg or any(int(a[p]) != color for p in outline):
                    continue
                caps = {}
                for d in range(4):
                    cells = _cap((y, x), radius, d)
                    values = {int(a[p]) for p in cells}
                    if len(values) == 1 and next(iter(values)) not in (bg, color):
                        caps[d] = next(iter(values))
                if caps:
                    emitters.append(((y, x), radius, color, caps))
    assert len(emitters) == 1, 'exactly one recognizable collector'
    center, radius, structural, caps = emitters[0]
    emitter_cells = _outline(center, radius)
    for d in caps:
        emitter_cells.update(_cap(center, radius, d))
    token_groups, controls = defaultdict(list), {}
    for cells in _components(fg - emitter_cells):
        ys, xs = zip(*cells)
        y, end_y, x, end_x = min(ys), max(ys) + 1, min(xs), max(xs) + 1
        colors = {int(a[p]) for p in cells}
        assert len(colors) == 1
        color = colors.pop()
        if color == structural:
            assert end_y - y == 1 and 2 <= end_x - x <= 7
            assert len(cells) == end_x - x
            assert (x, end_x) not in controls
            controls[x, end_x] = y
        else:
            assert 2 <= end_y - y <= 3 and 2 <= end_x - x <= 7
            mask = np.zeros((end_y - y, end_x - x), dtype=bool)
            for yy, xx in cells:
                mask[yy - y, xx - x] = True
            token_groups[x, end_x].append((y, color, mask))
    assert token_groups and set(controls) <= set(token_groups)
    stacks, destinations = [], []
    for (x, end_x), tokens in sorted(token_groups.items()):
        tokens.sort(key=lambda t: t[0])
        h = tokens[0][2].shape[0]
        mask, color = tokens[0][2], tokens[0][1]
        assert all(c == color and np.array_equal(m, mask) for y, c, m in tokens)
        assert all(b[0] - t[0] == h + 1 for t, b in zip(tokens, tokens[1:]))
        s = dict(x=x, bottom=tokens[-1][0], color=color, mask=mask,
                 count=len(tokens), bar=controls.get((x, end_x)))
        if s['bar'] is not None:
            assert s['bar'] == s['bottom'] + h + 1, 'one blank row above control'
        else:
            gap = center[0] - radius - (s['bottom'] + h)
            if end_x - x in (3, 5) and x + (end_x - x) // 2 == center[1] and 1 <= gap <= 3:
                destinations.append(len(stacks))
        stacks.append(s)
    assert len(destinations) == 1
    dst = destinations[0]
    assert all(s['bar'] is not None or i == dst or
               s['x'] + s['mask'].shape[1] // 2 != center[1]
               for i, s in enumerate(stacks)), 'unmarked stacks cannot imitate the destination'
    bars = {p: i for i, s in enumerate(stacks) if s['bar'] is not None for p in _bar_cells(s)}
    lines = _lines(center, radius, caps, a.shape)
    paths, contacts = _trace(lines, bars)
    selected = set(contacts) - {None}
    assert selected and dst not in selected
    assert not ({p for d, path in paths for p in path} & fg), 'rays encounter bars, never payloads'
    total = sum(stacks[i]['count'] for i in selected)
    new_cells = _token_cells(stacks[dst], stacks[dst]['count'] + total) - _token_cells(stacks[dst])
    assert all(_inside(p, a.shape) for p in new_cells)
    assert not new_cells & fg
    assert not new_cells & {p for d, path in paths for p in path}
    return dict(grid=a, background=bg, center=center, radius=radius, structural=structural,
                caps=caps, stacks=stacks, destination=dst, bars=bars, lines=lines,
                paths=paths, contacts=contacts, selected=selected, total=total)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All decisions come from visible geometry and fixed family conventions."""
    p = _parse(input_grid)
    output = p['grid'].copy()
    for direction, path in p['paths']:
        for cell in path:
            output[cell] = p['caps'][direction]
    for i in p['selected']:
        s = p['stacks'][i]
        for cell in _bar_cells(s):
            output[cell] = p['background']
        for cell in _token_cells(s):
            output[cell] = p['structural']
    dst = p['stacks'][p['destination']]
    for cell in _token_cells(dst, dst['count'] + p['total']):
        output[cell] = dst['color']
    return output


def _mask(rng, width=None, height=None):
    h = int(rng.integers(2, 4)) if height is None else height
    w = int(rng.integers(2, 8)) if width is None else width
    mode = int(rng.integers(4))
    m = np.ones((h, w), dtype=bool)
    if mode == 1:
        m[:] = False
        m[:, int(rng.integers(w))] = True
        m[int(rng.integers(h)), :] = True
    elif mode >= 2:
        m[:] = False
        m[:, 0] = True
        m[0, :] = True
        for y in range(1, h):
            m[y, :int(rng.integers(1, w + 1))] = True
        if rng.integers(2):
            m = m[:, ::-1]
        if rng.integers(2):
            m = m[::-1]
    return m.copy()


def _halo(cells):
    return cells | {(y + dy, x + dx) for y, x in cells
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))}


def _candidate_ok(s, state, required, n_selected=None):
    key = (s['x'], s['x'] + s['mask'].shape[1])
    if key in state['columns']:
        return False
    cells = _token_cells(s)
    if s['bar'] is not None:
        cells |= _bar_cells(s)
    if not all(_inside(p, state['shape']) for p in cells):
        return False
    if _halo(cells) & state['fg'] or cells & state['reserved']:
        return False
    bars = dict(state['bars'])
    i = len(state['sources'])
    if s['bar'] is not None:
        bars.update({p: i for p in _bar_cells(s)})
    paths, contacts = _trace(state['lines'], bars)
    selected = set(contacts) - {None}
    if required == 'selected' and i not in selected:
        return False
    if required in ('dormant', 'interior') and i in selected:
        return False
    if required == 'interior':
        _, whole_contacts = _trace(state['whole_lines'], bars)
        if i not in whole_contacts:
            return False
    if not state['selected'] <= selected:
        return False
    if n_selected is not None and len(selected) != n_selected:
        return False
    if {p for d, path in paths for p in path} & (state['fg'] | cells):
        return False
    state['sources'].append(s)
    state['bars'] = bars
    state['fg'].update(cells)
    state['columns'].add(key)
    state['selected'] = selected
    return True


def _place(rng, state, count, mode, duplicate=False):
    """Sample receivers from geometric incidences, then vary their silhouettes."""
    proposals = []
    r = state['radius']
    if mode == 'interior':
        lines = state['whole_lines']
        endpoints = {p for d, path in state['lines'] for p in path}
        points = [(d, y, x) for d, path in lines for y, x in path if (y, x) not in endpoints]
    else:
        points = [(d, y, x) for d, path in state['lines'] for y, x in path]
    rng.shuffle(points)
    for d, y, x in points:
        dy, dx = DIRECTIONS[d]
        if duplicate:
            if dy != -1:
                continue
            # At a shared row the two parallel endpoint lanes are 2*r apart.
            xs = [xx for dd, line in state['lines'] if dd == d for yy, xx in line if yy == y]
            if len(xs) != 2:
                continue
            # Reserve room for an interior receiver before this wide blocker
            # and for a shadow receiver beyond its outer lane. A greedy near-cap
            # placement can be valid by itself yet make the teaching witnesses
            # geometrically impossible later.
            if state['center'][0] - r - 1 - y < 4:
                continue
            if dx < 0 and min(xs) < 4:
                continue
            if dx > 0 and max(xs) > state['shape'][1] - 5:
                continue
            widths = [2 * r + 1]
            starts = [min(xs)]
        else:
            widths = [2, 3, int(rng.integers(2, 8))]
            starts = None
        for w in widths:
            offsets = rng.permutation(w).tolist() if dy < 0 or mode == 'interior' else [0 if dx > 0 else w - 1]
            for left in (starts if starts is not None else [x - offset for offset in offsets]):
                proposals.append((y, left, w))
    rng.shuffle(proposals)
    for y, x, w in proposals[:900]:
        masks = [_mask(rng, width=w, height=2 if count >= 3 else None)]
        if mode == 'interior' and w in (2, 3):
            # Connected corners can sit between rays without touching either.
            corner = np.zeros((2, w), bool)
            corner[0, :] = True
            corner[:, 0] = True
            masks.extend([corner, corner[:, ::-1].copy(), corner[::-1].copy()])
        for mask in masks:
            s = dict(x=x, bottom=y - 1 - mask.shape[0], bar=y, count=count,
                     mask=mask, color=int(rng.choice(state['payloads'])))
            required = 'selected' if mode == 'source' else ('interior' if mode == 'interior' else 'dormant')
            if mode == 'shadow':
                # The original endpoint line reaches this bar, but a nearer bar stops it.
                if not any(p in _bar_cells(s) for dd, line in state['lines'] for p in line):
                    continue
            expected = len(state['selected']) + (mode == 'source')
            before = len(state['sources'])
            if _candidate_ok(s, state, required, expected):
                if duplicate:
                    _, hits = _trace(state['lines'], state['bars'])
                    assert hits.count(before) >= 2
                return True
    return False


def _place_free(rng, state, marked):
    for _ in range(140):
        mask = _mask(rng)
        count = int(rng.integers(1, 4))
        y = int(rng.integers(4, state['shape'][0]))
        x = int(rng.integers(0, state['shape'][1] - mask.shape[1] + 1))
        s = dict(x=x, bottom=y - mask.shape[0] - 1, count=count, mask=mask,
                 bar=y if marked else None, color=int(rng.choice(state['payloads'])))
        if x + mask.shape[1] // 2 == state['center'][1] and not marked:
            continue
        if _candidate_ok(s, state, 'dormant', len(state['selected'])):
            return True
    return False


def _alternative_sets(p):
    whole_lines = _lines(p['center'], p['radius'], p['caps'], p['grid'].shape, whole=True)
    _, whole_hits = _trace(whole_lines, p['bars'])
    through = {p['bars'][cell] for d, line in p['lines'] for cell in line if cell in p['bars']}
    return set(whole_hits) - {None}, through


def _build_pair(rng, radius, cap_directions, source_counts, destination_count,
                double=False, teaching=False, destination_height=None):
    total = sum(source_counts)
    for attempt in range(100):
        tight_height_witness = destination_height == 3 and destination_count + total == 5
        h = 30 if tight_height_witness else int(rng.integers(24 if teaching else 19, 31))
        w = int(rng.integers(25 if teaching else 19, 31))
        token_height = destination_height if destination_height is not None else (
            3 if destination_count + total <= 5 and rng.random() < .35 else 2)
        token_width = 3 if tight_height_witness else int(rng.choice([3, 5]))
        destination_mask = _mask(rng, width=token_width, height=token_height)
        gap = 1 if tight_height_witness else int(rng.integers(destination_mask.shape[1] // 2, 4))
        min_y = radius + gap + token_height + (destination_count + total - 1) * (token_height + 1)
        if double:
            min_y = max(min_y, 3 * source_counts[0] + radius + 5)
        max_y = h - radius - (4 if teaching else 2)
        if min_y > max_y:
            continue
        cy = int(rng.integers(min_y, max_y + 1))
        cx = int(rng.integers(radius + 4, w - radius - 4))
        center = (cy, cx)
        palette = rng.permutation(10).tolist()
        bg, structural = palette[:2]
        colors = palette[2:]
        cap_bank = rng.choice(colors, int(rng.integers(1, min(4, len(cap_directions)) + 1)), replace=False).tolist()
        caps = {d: int(rng.choice(cap_bank)) for d in cap_directions}
        dst = dict(x=cx - destination_mask.shape[1] // 2,
                   bottom=cy - radius - gap - token_height, mask=destination_mask,
                   count=destination_count, bar=None, color=int(rng.choice(colors)))
        fixed = _outline(center, radius) | _token_cells(dst)
        for d in caps:
            fixed.update(_cap(center, radius, d))
        state = dict(shape=(h, w), center=center, radius=radius, caps=caps,
                     lines=_lines(center, radius, caps, (h, w)), fg=fixed,
                     whole_lines=_lines(center, radius, caps, (h, w), whole=True),
                     reserved=_token_cells(dst, destination_count + total),
                     bars={}, sources=[], selected=set(), payloads=colors,
                     columns={(dst['x'], dst['x'] + dst['mask'].shape[1])})
        if not all(_inside(p, (h, w)) for p in fixed | state['reserved']):
            continue
        if any(p in fixed for d, line in state['lines'] for p in line):
            continue
        if not all(_place(rng, state, n, 'source', double and i == 0)
                   for i, n in enumerate(source_counts)):
            continue
        if double:
            if not _place(rng, state, int(rng.integers(1, 3)), 'interior'):
                continue
            if not _place(rng, state, int(rng.integers(1, 3)), 'shadow'):
                continue
        else:
            if rng.random() < .7:
                _place(rng, state, int(rng.integers(1, 3)), 'shadow')
            if rng.random() < .7:
                _place(rng, state, 1, 'interior')
        for _ in range(int(rng.integers(0, 3))):
            _place_free(rng, state, marked=bool(rng.integers(2)))
        a = np.full((h, w), bg, dtype=np.int8)
        for p in _outline(center, radius):
            a[p] = structural
        for d, color in caps.items():
            for p in _cap(center, radius, d):
                a[p] = color
        for s in [dst] + state['sources']:
            for p in _token_cells(s):
                a[p] = s['color']
            if s['bar'] is not None:
                for p in _bar_cells(s):
                    a[p] = structural
        # Construct the label from the construction's explicit objects.
        b = a.copy()
        paths, hits = _trace(state['lines'], state['bars'])
        selected = set(hits) - {None}
        assert len(selected) == len(source_counts)
        if teaching and any(max(len(path) for dd, path in paths if dd == d) < 2 for d in caps):
            continue
        for d, path in paths:
            for p in path:
                b[p] = caps[d]
        for i in selected:
            for p in _bar_cells(state['sources'][i]):
                b[p] = bg
            for p in _token_cells(state['sources'][i]):
                b[p] = structural
        for p in _token_cells(dst, destination_count + total):
            b[p] = dst['color']
        try:
            parsed = _parse(a)
            assert parsed['total'] == total and len(parsed['selected']) == len(source_counts)
            if double:
                whole, through = _alternative_sets(parsed)
                assert whole != parsed['selected'] and through != parsed['selected']
                assert any(parsed['contacts'].count(i) >= 2 for i in parsed['selected'])
            assert np.array_equal(solve(a), b)
        except AssertionError:
            continue
        return {'input': a, 'output': b}
    raise RuntimeError('bounded geometric construction exhausted')


def _check_counts(num_train, num_test):
    for n, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    max_radius = int(rng.choice([2, 3]))
    # Jointly choose informative operands; the two witnesses have shuffled roles.
    options = [(a, b, d1, d2) for a in range(2, 5) for b in range(3, 6)
               for d1 in range(1, 4) for d2 in range(1, 4)
               if a != b and d1 != d2 and a * d2 != b * d1
               and a + d1 != b + d2
               and a + d1 <= 6 and b + d2 <= 6
               and not (a * d1 == a + d1 and b * d2 == b + d2)]
    a, b, d1, d2 = options[int(rng.integers(len(options)))]
    cuts = [v for v in range(1, b) if v != b - v]
    cut = int(rng.choice(cuts))
    counts2 = [cut, b - cut]
    rng.shuffle(counts2)
    first_mask = int(rng.choice([m for m in range(1, 16) if m & 3]))
    second_mask = (15 ^ first_mask) | int(rng.integers(1, 16)) | int(rng.choice([1, 2]))
    plans = [(max_radius, [d for d in range(4) if first_mask & (1 << d)], [a], d1, True),
             (max_radius, [d for d in range(4) if second_mask & (1 << d)], counts2, d2, False)]
    taller = int(rng.choice([i for i, plan in enumerate(plans) if sum(plan[2]) + plan[3] <= 5]))
    plans = [(*plan, 3 if i == taller else 2) for i, plan in enumerate(plans)]
    rng.shuffle(plans)
    for i in range(2, num_train + num_test):
        test = i >= num_train
        n = int(rng.choice([1, 2, 3], p=[.15, .35, .50] if test else [.35, .4, .25]))
        total = int(rng.integers(max(2, n), 5))
        if i == num_train:
            # Transfer to three singleton contributors and a singleton destination.
            n, total = 3, 3
        cuts = sorted(rng.choice(np.arange(1, total), n - 1, replace=False).tolist())
        heights = np.diff([0] + cuts + [total]).tolist()
        d = int(rng.integers(1, min(3, 6 - total) + 1))
        if i == num_train:
            d = 1
        elif not test and total == n and d == 1:
            d = 2
        mask_options = [m for m in range(1, 16) if m.bit_count() * 2 >= n
                        and (total + d <= 4 or m & 3)]
        mask = int(rng.choice(mask_options))
        plans.append((int(rng.choice([2, max_radius])),
                      [d for d in range(4) if mask & (1 << d)], heights, d, False, None))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, (r, directions, heights, d, double, destination_height) in enumerate(plans):
        for _ in range(12):
            pair = _build_pair(rng, r, directions, heights, d, double, i < 2, destination_height)
            key = (pair['input'].shape, pair['input'].tobytes())
            if key not in seen:
                seen.add(key)
                puzzle['train' if i < num_train else 'test'].append(pair)
                break
        else:
            raise RuntimeError('unique input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _closed_form_output(p):
    """Validate selection by collinearity and distance, rather than walking rays.

    This shares segmentation with solve; it is not an independent semantic oracle.
    """
    a = p['grid']
    b = a.copy()
    hits = set()
    yy, xx = np.indices(a.shape)
    for direction, color in p['caps'].items():
        dy, dx = DIRECTIONS[direction]
        cap = _cap(p['center'], p['radius'], direction)
        for oy, ox in (cap[0], cap[-1]):
            distances = [(dy * (y - oy), i) for (y, x), i in p['bars'].items()
                         if dy * (y - oy) == dx * (x - ox) and dy * (y - oy) > 0]
            limit, hit = min(distances) if distances else (100, None)
            if hit is not None:
                hits.add(hit)
            distance = dy * (yy - oy)
            mask = (distance > 0) & (distance < limit) & (distance == dx * (xx - ox))
            assert np.all(a[mask] == p['background'])
            b[mask] = color
    total = sum(p['stacks'][i]['count'] for i in hits)
    assert hits == p['selected'] and total == p['total']
    for i in hits:
        for cell in _bar_cells(p['stacks'][i]):
            b[cell] = p['background']
        for cell in _token_cells(p['stacks'][i]):
            b[cell] = p['structural']
    dst = p['stacks'][p['destination']]
    for cell in _token_cells(dst, dst['count'] + total):
        b[cell] = dst['color']
    return b


def _features(p):
    dst = p['stacks'][p['destination']]
    whole, through = _alternative_sets(p)
    return dict(radius=p['radius'], caps=tuple(sorted(p['caps'])), shape=p['grid'].shape,
                center=p['center'], source_counts=tuple(sorted(p['stacks'][i]['count'] for i in p['selected'])),
                sources=len(p['selected']), total=p['total'], destination=dst['count'],
                destination_height=dst['mask'].shape[0],
                receivers=len({i for i in p['bars'].values()}),
                unmarked=sum(s['bar'] is None for i, s in enumerate(p['stacks']) if i != p['destination']),
                double=any(p['contacts'].count(i) > 1 for i in p['selected']),
                interior=whole != p['selected'], shadow=through != p['selected'],
                selected_directions=tuple(sorted({d for (d, _), hit in zip(p['lines'], p['contacts']) if hit is not None})),
                shown_directions=tuple(sorted({d for d, path in p['paths'] if len(path) >= 2})),
                masks=tuple(sorted((s['mask'].shape, s['mask'].tobytes()) for s in p['stacks'])),
                palette=(p['background'], p['structural']),
                layout=(p['grid'].shape, p['center'], p['radius'], tuple(sorted(p['caps'])),
                        tuple((s['x'], s['bottom'], s['count'], s['bar'], s['mask'].shape, s['mask'].tobytes())
                              for s in p['stacks'])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and not np.array_equal(a, b)
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            p = _parse(a)
            assert np.array_equal(b, solve(a))
            assert np.array_equal(b, _closed_form_output(p))
            # Check parent B's whole-stack recoloring, deduplicated sum, and exact copy.
            for i, s in enumerate(p['stacks']):
                expected = p['structural'] if i in p['selected'] else s['color']
                assert all(b[cell] == expected for cell in _token_cells(s))
                if s['bar'] is not None:
                    expected_bar = p['background'] if i in p['selected'] else p['structural']
                    assert all(b[cell] == expected_bar for cell in _bar_cells(s))
            features[split].append(_features(p))
    train = features['train']
    assert set(d for f in train for d in f['caps']) == set(range(4))
    assert set(d for f in train for d in f['shown_directions']) == set(range(4))
    assert max(f['radius'] for f in features['test']) <= max(f['radius'] for f in train)
    assert {f['destination_height'] for f in train[:2]} == {2, 3}
    assert any(f['double'] and f['interior'] and f['shadow'] for f in train)
    assert any(f['sources'] >= 2 and len(set(f['source_counts'])) > 1 for f in train)
    totals, dests = [f['total'] for f in train], [f['destination'] for f in train]
    assert len(set(totals)) > 1 and len(set(dests)) > 1
    assert len({s + d for s, d in zip(totals, dests)}) > 1
    assert any(totals[0] * d != s * dests[0] for s, d in zip(totals, dests))
    rivals = (lambda f: 2 * f['destination'], lambda f: 2 * f['total'],
              lambda f: max(f['destination'], f['total']), lambda f: f['total'],
              lambda f: f['total'] * f['destination'],
              lambda f: f['destination'] + f['sources'],
              lambda f: f['destination'] + max(f['source_counts']))
    for rival in rivals:
        assert any(rival(f) != f['total'] + f['destination'] for f in train)
    first_test = features['test'][0]
    assert first_test['source_counts'] == (1, 1, 1) and first_test['destination'] == 1
    assert all(f['destination'] > 1 or any(n > 1 for n in f['source_counts']) for f in train)


def _fingerprint(puzzle):
    return tuple((s, np.asarray(pair['input']).shape, np.asarray(pair['input']).tobytes(),
                  np.asarray(pair['output']).tobytes())
                 for s in ('train', 'test') for pair in puzzle[s])


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    rows = [_features(_parse(pair['input'])) for p in puzzles for split in ('train', 'test') for pair in p[split]]
    assert {f['radius'] for f in rows} == {2, 3}
    assert {f['sources'] for f in rows} >= {1, 2, 3}
    assert len({f['shape'] for f in rows}) >= 25
    assert len({f['layout'] for f in rows}) >= 180
    assert len({m for f in rows for m in f['masks']}) >= 25
    assert len({f['palette'] for f in rows}) >= 30
    assert set(d for f in rows for d in f['selected_directions']) == set(range(4))
    assert any(f['unmarked'] for f in rows)
    assert any(f['receivers'] == f['sources'] for f in rows)
    assert any(f['shape'] == (30, 30) for f in rows)
    for p in puzzles:
        local = [_features(_parse(pair['input'])) for s in ('train', 'test') for pair in p[s]]
        assert len({f['layout'] for f in local}) == len(local)
        assert len({f['palette'] for f in local}) > 1
    for n in range(2, 11):
        for t in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(100 + n * 5 + t, n, t)
            assert len(p['train']) == n and len(p['test']) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (None, 1), (4, np.bool_(True))):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted illegal counts')
    # Labels missing a ray, highlight, bar erasure, or an added token must fail.
    original = puzzles[0]
    p = _parse(original['train'][0]['input'])
    source = p['stacks'][min(p['selected'])]
    dst = p['stacks'][p['destination']]
    new = _token_cells(dst, dst['count'] + p['total']) - _token_cells(dst)
    ray = next(cell for d, path in p['paths'] for cell in path)
    cells = [ray, min(_token_cells(source)), min(_bar_cells(source)), min(new)]
    for cell in cells:
        bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in original[s]]
               for s in ('train', 'test')}
        bad['train'][0]['output'][cell] = bad['train'][0]['input'][cell]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('accepted a damaged output')
    return {'seeds': 42, 'pairs': len(rows), 'shapes': len({f['shape'] for f in rows}),
            'layouts': len({f['layout'] for f in rows}),
            'token_masks': len({m for f in rows for m in f['masks']}),
            'selected_directions': sorted({d for f in rows for d in f['selected_directions']})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
