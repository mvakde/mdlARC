"""Cross-shutter census: ranked apertures determine an object-counting window.

Only NumPy and the standard library are used. The decoder sees pixels only;
construction records are used separately to label generated examples.
"""
import numpy as np


_STEPS8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_SMALL = (((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 1)),
          ((0, 0), (0, 1), (1, 0)), _SQUARE,
          ((0, 0), (1, 1), (2, 2)), _DIAMOND,
          ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
          ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)))
_SHORTCUTS = ('all_objects', 'area', 'four_connected', 'any_overlap',
              'horizontal_only', 'vertical_only', 'longest', 'shortest',
              'ascending', 'spatial_forward', 'spatial_reverse', 'key_value')


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'bad grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'bad grid colors')
    return a.astype(np.int8, copy=False)


def _signature(a):
    a = np.asarray(a)
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _puzzle_signature(p):
    return tuple(tuple((_signature(x['input']), _signature(x['output']))
                       for x in p[s]) for s in ('train', 'test'))


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = _STEPS8 if diagonal else ((0, 1), (0, -1), (1, 0), (-1, 0))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(tuple(sorted(cells)))
    return result


def _field_rectangles(a, bg):
    """A field color occurs only within its rectangle, with a complete border."""
    result = []
    for color in sorted(set(map(int, np.unique(a))) - {bg}):
        rr, cc = np.where(a == color)
        bounds = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        r0, r1, c0, c1 = bounds
        p = a[r0:r1, c0:c1]
        if min(p.shape) < 5 or np.mean(p == color) <= .55:
            continue
        if not all(np.all(edge == color) for edge in (p[0], p[-1], p[:, 0], p[:, -1])):
            continue
        ring = np.zeros(a.shape, bool)
        ring[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
        ring[r0:r1, c0:c1] = False
        if np.all(a[ring] == bg):
            result.append((color, bounds))
    return result


def _chart(a, field, bounds):
    r0, r1, c0, c1 = bounds
    physical = a[r0:r1, c0:c1]
    candidates = []
    for rotation in range(4):
        p = np.rot90(physical, -rotation)
        colors = sorted(set(map(int, np.unique(p))) - {field})
        if not 1 <= len(colors) <= 4:
            continue
        keys = []
        good = True
        for color in colors:
            pos = np.argwhere(p == color)
            rows, cols = pos[:, 0], pos[:, 1]
            if (not 1 <= len(pos) <= 4 or len(set(rows)) != 1 or
                    not np.array_equal(cols, np.arange(cols.min(), cols.max()+1))):
                good = False
                break
            keys.append(dict(color=color, row=int(rows[0]), start=int(cols[0]), rank=len(pos)))
        if not good:
            continue
        keys.sort(key=lambda k: k['row'])
        starts = {k['start'] for k in keys}
        if len(starts) != 1 or any(b['row'] - x['row'] < 2 for x, b in zip(keys, keys[1:])):
            continue
        inset = keys[0]['start']
        # The start line is uniquely nearest this panel side, even for a
        # single singleton key. This also makes inward tally growth visible.
        distances = (keys[0]['row'], p.shape[0]-1-keys[-1]['row'],
                     p.shape[1]-1-max(k['start']+k['rank']-1 for k in keys))
        if not (inset >= 1 and all(inset < d for d in distances)):
            continue
        candidates.append(dict(rotation=rotation, canonical=p.copy(), keys=keys,
                               field=field, bounds=bounds))
    _require(len(candidates) == 1, 'chart needs a unique inward direction')
    return candidates[0]


def _runs(indices):
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1) if len(indices) else []


def _parse_scene(a, bg, garden, chart):
    field, (r0, r1, c0, c1) = garden
    cr0, cr1, cc0, cc1 = chart['bounds']
    _require(r1 <= cr0 or cr1 <= r0 or c1 <= cc0 or cc1 <= c0, 'overlapping panels')
    controls = {k['color'] for k in chart['keys']}
    _require(not controls & {bg, field, chart['field']}, 'overlapping color roles')
    patch = a[r0:r1, c0:c1]
    _require(not np.any(np.isin(patch, [bg, chart['field']])), 'field palette leaks')
    objects = []
    owner = np.full(patch.shape, -1, int)
    for color in sorted(set(map(int, np.unique(patch))) - {field}):
        for cells in _components(patch == color):
            for r, c in cells:
                owner[r, c] = len(objects)
            objects.append(dict(color=color, cells=cells))
    for r, c in np.argwhere(owner >= 0):
        neighbors = owner[max(0, r-1):r+2, max(0, c-1):c+2]
        _require(bool(np.all((neighbors < 0) | (neighbors == owner[r, c]))),
                 'separate garden objects must not touch, even diagonally')
    _require(controls <= {o['color'] for o in objects}, 'key has no garden objects')
    exterior = a != bg
    exterior = exterior.copy()
    exterior[r0:r1, c0:c1] = False
    exterior[cr0:cr1, cc0:cc1] = False
    consumed = np.zeros(a.shape, bool)
    gates = []
    for axis in ('H', 'V'):
        count = a.shape[0] if axis == 'H' else a.shape[1]
        for line in range(count):
            within_garden_axis = (r0 <= line < r1) if axis == 'H' else (c0 <= line < c1)
            if within_garden_axis:
                continue
            mask = exterior[line, :] if axis == 'H' else exterior[:, line]
            for run in _runs(np.flatnonzero(mask)):
                lo, hi = int(run[0]), int(run[-1])
                if axis == 'H':
                    if not c0 < lo <= hi < c1-1:
                        continue
                    values = a[line, lo:hi+1]
                    cells = [(line, j) for j in range(lo, hi+1)]
                    start, end = lo-c0, hi-c0
                else:
                    if not r0 < lo <= hi < r1-1:
                        continue
                    values = a[lo:hi+1, line]
                    cells = [(j, line) for j in range(lo, hi+1)]
                    start, end = lo-r0, hi-r0
                _require(len(values) >= 3, 'shutter needs two caps and an interior')
                cap, payload = int(values[0]), int(values[1])
                _require(cap in controls and int(values[-1]) == cap, 'shutter cap mismatch')
                _require(payload not in controls | {bg, field, chart['field']} and
                         np.all(values[1:-1] == payload), 'invalid shutter interior')
                _require(not any(consumed[r, c] for r, c in cells), 'overlapping shutters')
                for r, c in cells:
                    consumed[r, c] = True
                gates.append(dict(color=cap, axis=axis, start=start, end=end,
                                  length=end-start+1, line=line, payload=payload))
    _require(np.array_equal(exterior, consumed), 'unexplained exterior pixels')
    families = {}
    for key in chart['keys']:
        for axis in ('H', 'V'):
            family = [g for g in gates if g['color'] == key['color'] and g['axis'] == axis]
            _require(len(family) >= 2 and key['rank'] <= len(family), 'missing requested shutter')
            _require(len({g['length'] for g in family}) == len(family), 'tied shutter lengths')
            families[key['color'], axis] = sorted(family, key=lambda g: -g['length'])
    return dict(background=bg, garden_field=field, garden_bounds=(r0, r1, c0, c1),
                garden=patch.copy(), chart=chart, objects=objects, gates=gates, families=families)


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             'canvas needs a complete background border')
    rectangles = _field_rectangles(a, bg)
    candidates = []
    for chart_field, chart_bounds in rectangles:
        try:
            chart = _chart(a, chart_field, chart_bounds)
        except AssertionError:
            continue
        for garden in rectangles:
            if garden[0] == chart_field:
                continue
            try:
                candidates.append(_parse_scene(a, bg, garden, chart))
            except AssertionError:
                continue
    _require(len(candidates) == 1, 'input must have a unique garden, chart, and shutter interpretation')
    return candidates[0]


def _window(meta, key, alternative=None):
    chosen = []
    for axis in ('H', 'V'):
        family = meta['families'][key['color'], axis]
        if alternative == 'longest':
            gate = family[0]
        elif alternative == 'shortest':
            gate = family[-1]
        elif alternative == 'ascending':
            gate = family[-key['rank']]
        elif alternative in ('spatial_forward', 'spatial_reverse'):
            ordered = sorted(family, key=lambda g: g['line'], reverse=alternative == 'spatial_reverse')
            gate = ordered[key['rank']-1]
        else:
            gate = family[key['rank']-1]
        chosen.append(gate)
    horizontal, vertical = chosen
    return vertical['start'], vertical['end'], horizontal['start'], horizontal['end']


def _inside(cells, window, mode=None):
    top, bottom, left, right = window
    if mode == 'horizontal_only':
        return all(left <= c <= right for r, c in cells)
    if mode == 'vertical_only':
        return all(top <= r <= bottom for r, c in cells)
    tests = [top <= r <= bottom and left <= c <= right for r, c in cells]
    return any(tests) if mode == 'any_overlap' else all(tests)


def _counts(meta, alternative=None):
    counts = []
    for key in meta['chart']['keys']:
        if alternative == 'key_value':
            counts.append(key['rank'])
            continue
        window = _window(meta, key, alternative)
        total = 0
        for obj in meta['objects']:
            if obj['color'] != key['color']:
                continue
            if alternative == 'four_connected':
                mask = np.zeros(meta['garden'].shape, bool)
                for r, c in obj['cells']:
                    mask[r, c] = True
                total += sum(_inside(cells, window) for cells in _components(mask, False))
            elif alternative == 'all_objects' or _inside(obj['cells'], window, alternative):
                total += len(obj['cells']) if alternative == 'area' else 1
        counts.append(int(total))
    return tuple(counts)


def _output(chart, counts):
    out = np.full(chart['canonical'].shape, chart['field'], dtype=np.int8)
    for key, count in zip(chart['keys'], counts):
        _require(count >= 1, 'selected census must be positive')
        last = key['start'] + 2*(count-1)
        _require(last < out.shape[1]-1, 'tally exceeds chart capacity')
        out[key['row'], key['start']:last+1:2] = key['color']
    return np.ascontiguousarray(np.rot90(out, chart['rotation']))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select both shutters, count whole eight-connected objects, crop the chart."""
    meta = _decode(input_grid)
    return _output(meta['chart'], _counts(meta))


def _shape(rng):
    if rng.random() < .72:
        cells = _SMALL[int(rng.choice(len(_SMALL), p=[.18, .15, .12, .15, .14, .07, .08, .06, .05]))]
    else:
        cells = {(1, 1)}
        target = int(rng.integers(2, 8))
        while len(cells) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _STEPS8
                               if 0 <= r+dr < 3 and 0 <= c+dc < 3} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    cells = list(cells)
    if rng.integers(2):
        cells = [(r, -c) for r, c in cells]
    for _ in range(int(rng.integers(4))):
        cells = [(-c, r) for r, c in cells]
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _pack(rng, height, width, records):
    blocked = np.zeros((height, width), bool)
    blocked[0] = blocked[-1] = True
    blocked[:, 0] = blocked[:, -1] = True
    placed = []
    order = list(range(len(records)))
    rng.shuffle(order)
    order.sort(key=lambda i: -len(records[i][1]))
    for index in order:
        color, shape = records[index]
        sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
        nh, nw = height-sh+1, width-sw+1
        allowed = np.ones((nh, nw), bool)
        for r, c in shape:
            allowed &= ~blocked[r:r+nh, c:c+nw]
        positions = np.argwhere(allowed)
        if not len(positions):
            return None
        r0, c0 = map(int, positions[int(rng.integers(len(positions)))])
        cells = tuple((r0+r, c0+c) for r, c in shape)
        for r, c in cells:
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
        placed.append(dict(color=color, cells=cells))
    return placed


def _intervals(side, low, high):
    return [(start, start+length-1) for length in range(low, high+1)
            for start in range(1, side-length)]


def _choose_window(rng, objects, color, height, width, hlimits, vlimits, witness=False):
    """Enumerate compatible apertures; objects remain independently shaped."""
    objs = [o for o in objects if o['color'] == color]
    horizontal = _intervals(width, *hlimits)
    vertical = _intervals(height, *vlimits)
    if not horizontal or not vertical:
        return None
    bounds = np.array([(min(r for r, c in o['cells']), max(r for r, c in o['cells']),
                        min(c for r, c in o['cells']), max(c for r, c in o['cells'])) for o in objs])
    hh, vv = np.array(horizontal), np.array(vertical)
    hm = (bounds[None, :, 2] >= hh[:, None, 0]) & (bounds[None, :, 3] <= hh[:, None, 1])
    vm = (bounds[None, :, 0] >= vv[:, None, 0]) & (bounds[None, :, 1] <= vv[:, None, 1])
    intersections = hm.astype(int) @ vm.astype(int).T
    valid = intersections >= 1
    if witness:
        valid &= (intersections <= 2) & (hm.sum(axis=1)[:, None] > intersections)
        valid &= vm.sum(axis=1)[None, :] > intersections
    choices = np.argwhere(valid)
    if not len(choices):
        return None
    for index in rng.permutation(len(choices)):
        hi, vi = map(int, choices[index])
        left, right = horizontal[hi]
        top, bottom = vertical[vi]
        window = top, bottom, left, right
        if witness and not any(_inside(o['cells'], window, 'any_overlap') and
                               not _inside(o['cells'], window) for o in objs):
            continue
        return window
    return None


def _gate_family(rng, color, axis, side, count, rank, interval, payloads):
    lo, hi = interval
    selected_length = hi-lo+1
    larger = list(map(int, rng.choice(np.arange(selected_length+1, side-1), rank-1, replace=False)))
    smaller = list(map(int, rng.choice(np.arange(3, selected_length), count-rank, replace=False)))
    gates = [dict(color=color, axis=axis, start=lo, end=hi, length=selected_length,
                  payload=int(rng.choice(payloads)))]
    for length in larger+smaller:
        start = int(rng.integers(1, side-length))
        gates.append(dict(color=color, axis=axis, start=start, end=start+length-1,
                          length=length, payload=int(rng.choice(payloads))))
    rng.shuffle(gates)
    return gates


def _palette(rng, families):
    p = list(map(int, rng.permutation(10)))
    return dict(bg=p[0], garden=p[1], chart=p[2], controls=p[3:3+families],
                payloads=p[3+families:5+families], distractor=p[5+families])


def _make_world(rng, families, ranks, capacity, anchor=False, force_five=False):
    """Joint geometry, independently counted labels, bounded resampling."""
    for attempt in range(180):
        palette = _palette(rng, families)
        chart_h = 5+2*(families-1)
        chart_w = 2*capacity+1+int(rng.integers(2))
        hpop = [int(rng.integers(max(2, r), 6 if families < 3 else 4)) for r in ranks]
        vpop = [int(rng.integers(max(2, r), 6 if families < 3 else 4)) for r in ranks]
        if anchor:
            hpop = [int(rng.integers(3, 5)) for _ in ranks]
            vpop = [int(rng.integers(3, 5)) for _ in ranks]
        top_strip = max(chart_h+2, sum(hpop)+1)
        left_strip = max(chart_w+2, sum(vpop)+1)
        max_h, max_w = min(18, 28-top_strip), min(18, 28-left_strip)
        if min(max_h, max_w) < 12:
            continue
        height = int(rng.integers(max(9, max_h-7), max_h+1))
        width = int(rng.integers(max(9, max_w-7), max_w+1))
        records = []
        populations = [int(rng.integers(3, capacity+1)) for _ in ranks]
        if anchor:
            populations[0] = 4
        if force_five:
            populations[0] = 5
        for i, color in enumerate(palette['controls']):
            for j in range(populations[i]):
                shape = (_DIAMOND, _SQUARE, ((0, 0),))[j] if anchor and i == 0 and j < 3 else _shape(rng)
                records.append((color, shape))
        for _ in range(int(rng.integers(1, 4))):
            records.append((palette['distractor'], _shape(rng)))
        objects = _pack(rng, height, width, records)
        if objects is None:
            continue
        gates, good = [], True
        for i, (color, rank) in enumerate(zip(palette['controls'], ranks)):
            nh, nv = hpop[i], vpop[i]
            if anchor:
                # Rank two is a non-extreme translated aperture. Rank one
                # spans the entire garden interior in both axes.
                window = _choose_window(rng, objects, color, height, width,
                                        (nh+1, width-3), (nv+1, height-3), i == 0)
                selected_rank = 2
            elif force_five and i == 0:
                window = (1, height-2, 1, width-2)
                selected_rank = 1
            else:
                window = _choose_window(rng, objects, color, height, width,
                                        (3+nh-rank, width-2-(rank-1)),
                                        (3+nv-rank, height-2-(rank-1)))
                selected_rank = rank
            if window is None:
                good = False
                break
            top, bottom, left, right = window
            hg = _gate_family(rng, color, 'H', width, nh, selected_rank, (left, right), palette['payloads'])
            vg = _gate_family(rng, color, 'V', height, nv, selected_rank, (top, bottom), palette['payloads'])
            if anchor:
                for family, side in ((hg, width), (vg, height)):
                    longest = max(family, key=lambda g: g['length'])
                    longest.update(start=1, end=side-2, length=side-2)
            gates.extend(hg+vg)
        if not good:
            continue
        return dict(palette=palette, height=height, width=width, objects=objects, gates=gates,
                    chart_h=chart_h, chart_w=chart_w, top_strip=top_strip, left_strip=left_strip)
    raise RuntimeError('bounded world construction exhausted')


def _world_counts(world, ranks):
    result = []
    for color, rank in zip(world['palette']['controls'], ranks):
        chosen = [sorted([g for g in world['gates'] if g['color'] == color and g['axis'] == axis],
                         key=lambda g: -g['length'])[rank-1] for axis in ('H', 'V')]
        h, v = chosen
        window = v['start'], v['end'], h['start'], h['end']
        result.append(sum(o['color'] == color and _inside(o['cells'], window) for o in world['objects']))
    return tuple(result)


def _render_pair(rng, world, ranks, geometry=None):
    p, ph, pw = world['palette'], world['height'], world['width']
    ts, ls = world['top_strip'], world['left_strip']
    if geometry is None:
        gates = [dict(g) for g in world['gates']]
        hindices = [i for i, g in enumerate(gates) if g['axis'] == 'H']
        vindices = [i for i, g in enumerate(gates) if g['axis'] == 'V']
        bh = max(0, 28-ts-ph-1)
        rw = max(0, 28-ls-pw-1)
        bottom_count = int(rng.integers(0, min(bh, len(hindices))+1))
        right_count = int(rng.integers(0, min(rw, len(vindices))+1))
        bottom = set(map(int, rng.choice(hindices, bottom_count, replace=False)))
        right = set(map(int, rng.choice(vindices, right_count, replace=False)))
        bs, rs = (bottom_count+1 if bottom_count else 0), (right_count+1 if right_count else 0)
        height, width = 2+ts+ph+bs, 2+ls+pw+rs
        gr, gc = 1+ts, 1+ls
        available = {'top': list(range(1, gr-1)), 'bottom': list(range(gr+ph+1, height-1)),
                     'left': list(range(1, gc-1)), 'right': list(range(gc+pw+1, width-1))}
        for values in available.values():
            rng.shuffle(values)
        for i, gate in enumerate(gates):
            side = ('bottom' if i in bottom else 'top') if gate['axis'] == 'H' else ('right' if i in right else 'left')
            gate['line'] = available[side].pop()
        chart_r = int(rng.integers(1, gr-world['chart_h']-1))
        chart_c = int(rng.integers(1, gc-world['chart_w']-1))
        key_order = list(map(int, rng.permutation(len(ranks))))
        geometry = dict(gates=gates, shape=(height, width), gr=gr, gc=gc,
                        chart_r=chart_r, chart_c=chart_c, key_order=key_order,
                        rotation=int(rng.integers(4)), reflect=bool(rng.integers(2)))
    gr, gc = geometry['gr'], geometry['gc']
    a = np.full(geometry['shape'], p['bg'], dtype=np.int8)
    a[gr:gr+ph, gc:gc+pw] = p['garden']
    for obj in world['objects']:
        for r, c in obj['cells']:
            a[gr+r, gc+c] = obj['color']
    for gate in geometry['gates']:
        line, lo, hi = gate['line'], gate['start'], gate['end']
        if gate['axis'] == 'H':
            a[line, gc+lo:gc+hi+1] = gate['payload']
            a[line, gc+lo] = a[line, gc+hi] = gate['color']
        else:
            a[gr+lo:gr+hi+1, line] = gate['payload']
            a[gr+lo, line] = a[gr+hi, line] = gate['color']
    chart = np.full((world['chart_h'], world['chart_w']), p['chart'], dtype=np.int8)
    output = chart.copy()
    counts = _world_counts(world, ranks)
    for j, i in enumerate(geometry['key_order']):
        row, color = 2+2*j, p['controls'][i]
        chart[row, 1:1+ranks[i]] = color
        _require(1 <= counts[i] <= (chart.shape[1]-1)//2, 'constructed count exceeds capacity')
        output[row, 1:2*counts[i]:2] = color
    r, c = geometry['chart_r'], geometry['chart_c']
    a[r:r+chart.shape[0], c:c+chart.shape[1]] = chart
    if geometry['reflect']:
        a, output = np.fliplr(a), np.fliplr(output)
    rotation = geometry['rotation']
    pair = dict(input=np.ascontiguousarray(np.rot90(a, rotation)),
                output=np.ascontiguousarray(np.rot90(output, rotation)))
    return pair, geometry


def _training_evidence(metas):
    _require({1, 2} <= {k['rank'] for m in metas for k in m['chart']['keys']},
             'training must show ranks one and two')
    _require(any(k['rank'] == 2 and all(len(m['families'][k['color'], ax]) >= 3 for ax in ('H', 'V'))
                 for m in metas for k in m['chart']['keys']), 'training needs an intermediate rank')
    histories = {}
    for m in metas:
        for k, n in zip(m['chart']['keys'], _counts(m)):
            histories.setdefault(k['color'], []).append((k['rank'], n))
    _require(any(any(r == 1 and s == 2 and n != q for r, n in values for s, q in values)
                 for values in histories.values()), 'same cap must change rank and census')
    for alternative in _SHORTCUTS:
        _require(any(_counts(m, alternative) != _counts(m) for m in metas),
                 'training leaves shortcut: '+alternative)
    counted = [o for m in metas for k in m['chart']['keys'] for o in m['objects']
               if o['color'] == k['color'] and _inside(o['cells'], _window(m, k))]
    _require(any(len(o['cells']) == 1 for o in counted), 'training needs a counted singleton')
    _require(any(len(o['cells']) >= 4 for o in counted), 'training needs a larger counted glyph')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or
                not lo <= value <= hi):
            raise ValueError(name+' must be a legal integer count')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(('families', 'rank', 'census')))
    puzzle = {'train': [], 'test': []}
    # A controlled contrast establishes causality: identical garden/shutters,
    # changed unary instructions, and changed counts. Placement is resampled
    # only if a spatial ranking shortcut accidentally explains both answers.
    for attempt in range(40):
        world = _make_world(rng, 2, [1, 2], 4, anchor=True)
        first, geom = _render_pair(rng, world, [1, 2])
        second, _ = _render_pair(rng, world, [2, 1], geom)
        metas = [_decode(first['input']), _decode(second['input'])]
        try:
            _training_evidence(metas)
        except AssertionError:
            continue
        puzzle['train'] = [first, second]
        break
    else:
        raise RuntimeError('bounded teaching contrast exhausted')
    seen = {_signature(pair['input']) for pair in puzzle['train']}
    for split, count in (('train', num_train-2), ('test', num_test)):
        for index in range(count):
            for attempt in range(40):
                target = split == 'test'
                families = 3 if target and shift == 'families' else int(rng.integers(1, 3))
                ranks = list(map(int, rng.integers(1, 3, families)))
                capacity = 5 if target and shift == 'census' else 4
                if target and shift == 'rank':
                    ranks[int(rng.integers(families))] = 3
                if target and shift == 'census':
                    ranks[0] = 1
                world = _make_world(rng, families, ranks, capacity,
                                    force_five=target and shift == 'census')
                pair, _ = _render_pair(rng, world, ranks)
                signature = _signature(pair['input'])
                if signature not in seen:
                    break
            else:
                raise RuntimeError('bounded unique example construction exhausted')
            seen.add(signature)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'bad pair counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = _signature(inp)
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            m = _decode(inp)
            counts = _counts(m)
            _require(np.array_equal(out, _output(m['chart'], counts)), 'wrong window census or chart')
            _require(np.array_equal(solve(inp.copy()), out), 'input-only solver disagreement')
            if split == 'train':
                training.append(m)
    _training_evidence(training)
    demonstrated = {k['rank'] for m in training for k in m['chart']['keys']}
    for pair in puzzle['test']:
        for key in _decode(pair['input'])['chart']['keys']:
            _require(key['rank'] in demonstrated or set(range(1, key['rank'])) <= demonstrated,
                     'test ordinal lacks a demonstrated progression')


def self_test() -> dict:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, windows, structures = set(), set(), set(), set()
    backgrounds, fields, orientations, family_counts, ranks, counts, populations = (set() for _ in range(7))
    glyphs, offsets, ignored, boundary = set(), 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _decode(pair['input'])
                shapes.add(np.asarray(pair['input']).shape)
                backgrounds.add(m['background'])
                fields.add((m['garden_field'], m['chart']['field']))
                orientations.add(m['chart']['rotation'])
                family_counts.add(len(m['chart']['keys']))
                ranks.update(k['rank'] for k in m['chart']['keys'])
                counts.update(_counts(m))
                populations.update(len(v) for v in m['families'].values())
                roles = {}
                structures.add((np.asarray(pair['input']).shape,
                                tuple(roles.setdefault(int(c), len(roles)) for c in np.asarray(pair['input']).flat)))
                for obj in m['objects']:
                    r0 = min(r for r, c in obj['cells'])
                    c0 = min(c for r, c in obj['cells'])
                    glyphs.add(tuple(sorted((r-r0, c-c0) for r, c in obj['cells'])))
                for key in m['chart']['keys']:
                    window = _window(m, key)
                    windows.add(window)
                    offsets += window[0] > 1 and window[2] > 1
                    boundary += any(o['color'] == key['color'] and _inside(o['cells'], window, 'any_overlap')
                                    and not _inside(o['cells'], window) for o in m['objects'])
                ignored += any(o['color'] not in {k['color'] for k in m['chart']['keys']} for o in m['objects'])
    _require(len(signatures) == 42, 'seed variation')
    _require(len(shapes) >= 35 and len(structures) >= 190, 'structural variation')
    _require(backgrounds == set(range(10)) and len(fields) >= 40, 'semantic palette variation')
    _require(orientations == set(range(4)), 'chart direction variation')
    _require(family_counts == {1, 2, 3} and ranks == {1, 2, 3}, 'category/ordinal variation')
    _require(counts == {1, 2, 3, 4, 5} and populations == {2, 3, 4, 5}, 'census/population variation')
    _require(len(glyphs) >= 35 and len(windows) >= 90, 'glyph/window variation')
    _require(offsets >= 25 and boundary >= 42 and ignored >= 190, 'spatial and distractor witnesses')
    _require(_puzzle_signature(puzzles[0]) == _puzzle_signature(generate_puzzle(0)), 'determinism')
    for nt, nq in ((2, 1), (5, 4), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(4100+nt+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # All rigid transforms must retain input-visible rank, containment and
    # inward growth. No training set or generation record is passed to solve.
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        for rotation in range(4):
            for reflect in (False, True):
                inp = np.fliplr(pair['input']) if reflect else pair['input']
                out = np.fliplr(pair['output']) if reflect else pair['output']
                _require(np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(out, rotation)),
                         'rotation/reflection covariance')
    # An unsampled but admissible scene exercises four controls, rank four,
    # an eight-connected 5x5 glyph, adjacent lanes, and exact window edges.
    # Each chosen window contains exactly its one matching object.
    extended_rng = np.random.default_rng(22552)
    controls = [3, 4, 5, 6]
    large = tuple((r, c) for r in range(1, 6) for c in range(1, 6)
                  if abs(r-3)+abs(c-3) == 2)
    objects = [dict(color=3, cells=large), dict(color=4, cells=((8, 1),)),
               dict(color=5, cells=((8, 4),)), dict(color=6, cells=((8, 8),))]
    gates = []
    for i, color in enumerate(controls):
        for axis in ('H', 'V'):
            for length in (9, 8, 7, 6):
                start = 1 if i < 2 else 10-length
                gates.append(dict(color=color, axis=axis, start=start,
                                  end=start+length-1, length=length, payload=7+i % 2))
    extended_world = dict(palette=dict(bg=0, garden=1, chart=2, controls=controls),
                          height=11, width=11, objects=objects, gates=gates,
                          chart_h=11, chart_w=9, top_strip=17, left_strip=17)
    extended, geometry = _render_pair(extended_rng, extended_world, [1, 2, 3, 4])
    expected = np.full((11, 9), 2, dtype=np.int8)
    for j, i in enumerate(geometry['key_order']):
        expected[2+2*j, 1] = controls[i]
    if geometry['reflect']:
        expected = np.fliplr(expected)
    expected = np.rot90(expected, geometry['rotation'])
    _require(np.array_equal(solve(extended['input']), expected), 'extended family support')
    mapping = np.array([8, 2, 0, 9, 1, 6, 3, 5, 4, 7], dtype=np.int8)
    _require(np.array_equal(solve(mapping[extended['input']]), mapping[expected]),
             'local color-role covariance')
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    return dict(seeds=42, pairs=210, distinct_input_shapes=len(shapes),
                color_normalized_inputs=len(structures), glyphs=len(glyphs),
                windows=len(windows), boundary_witnesses=boundary,
                shortcuts_rejected=list(_SHORTCUTS), extended_fixture=True,
                independent_solver=False)
