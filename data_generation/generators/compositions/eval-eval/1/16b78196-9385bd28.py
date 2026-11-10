"""Interlocking printing plates: infer exact assembly, then print its envelopes.

Only NumPy and the standard library are used. No evidence files or generation
state are read by solve(). See task_spec.md for the full geometric domain.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _components(g, bg):
    remaining = set(map(tuple, np.argwhere(g != bg)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color = int(g[start])
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining and int(g[p]) == color:
                    remaining.remove(p)
                    stack.append(p)
        result.append((np.asarray(sorted(cells), dtype=int), color))
    return result


def _dilate(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([padded[r:r+h, c:c+w]
                                 for r in range(3) for c in range(3)])


def _bbox(cells):
    r, c = cells.min(axis=0)
    rr, cc = cells.max(axis=0) + 1
    return int(r), int(c), int(rr), int(cc)


def _parse(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert np.count_nonzero(counts == counts[bg]) == 1, "background must be unique mode"
    components = _components(g, bg)
    spanning = []
    for cells, color in components:
        for axis in (0, 1):
            if len(np.unique(cells[:, 1-axis])) == g.shape[1-axis]:
                spanning.append((cells, color, axis))
    assert len(spanning) == 1, "one spanning band required"
    band_cells, band, axis = spanning[0]
    band_mask = np.zeros_like(g, dtype=bool)
    band_mask[tuple(band_cells.T)] = True
    # The only other component in this structural color is the table's ring.
    rings = [(cells, col) for cells, col in components
             if col == band and not band_mask[tuple(cells[0])]]
    assert len(rings) == 1, "one detached table frame required"
    ring = rings[0][0]
    r0, c0, r1, c1 = _bbox(ring)
    length = r1-r0-2
    assert 2 <= length <= 6 and c1-c0 == 4
    border = np.ones((length+2, 4), dtype=bool)
    border[1:-1, 1:-1] = False
    assert set(map(tuple, ring-(r0, c0))) == set(map(tuple, np.argwhere(border)))
    keys = list(map(int, g[r0+1:r1-1, c0+1]))
    inks = list(map(int, g[r0+1:r1-1, c0+2]))
    assert len(set(keys)) == length and bg not in keys and band not in keys
    assert band not in inks
    table_mask = np.zeros_like(band_mask)
    table_mask[r0:r1, c0:c1] = True
    fixed = band_mask | table_mask
    assert not np.any(_dilate(band_mask) & table_mask)
    scene = g.copy()
    scene[fixed] = bg
    pieces = _components(scene, bg)
    assert 2 <= len(pieces) <= 8
    assert {color for _, color in pieces} == set(keys), "each key must name a plate"
    occupied = fixed.copy()
    for cells, _ in pieces:
        assert not np.any(_dilate(occupied)[tuple(cells.T)]), "plates need clearance"
        occupied[tuple(cells.T)] = True
    base = g.copy()
    for cells, _ in pieces:
        base[tuple(cells.T)] = bg
    canonical_band = band_mask if axis == 0 else band_mask.T
    edges = []
    for c in range(canonical_band.shape[1]):
        rows = np.flatnonzero(canonical_band[:, c])
        assert len(rows) and np.array_equal(rows, np.arange(rows[0], rows[-1]+1))
        edges.append((int(rows[0]), int(rows[-1])+1))
    edges = np.asarray(edges)
    assert np.all(edges[:-1, 0] < edges[1:, 1])
    assert np.all(edges[1:, 0] < edges[:-1, 1])
    return dict(bg=bg, band=band, axis=axis, band_mask=band_mask,
                table_mask=table_mask, table=(r0, c0, r1, c1), fixed=fixed,
                keys=keys, inks=inks, pieces=pieces, base=base, edges=edges)


def _profile_description(cells, sign):
    cross = cells[:, 1] - cells[:, 1].min()
    outward = sign * cells[:, 0]
    lower, upper = [], []
    for c in range(int(cross.max())+1):
        vals = np.sort(outward[cross == c])
        assert len(vals) and np.array_equal(vals, np.arange(vals[0], vals[-1]+1))
        lower.append(int(vals[0]))
        upper.append(int(vals[-1])+1)
    offset = min(lower)
    return (tuple(v-offset for v in lower), tuple(v-offset for v in upper),
            np.column_stack((outward-offset, cross)))


def _recover(g, p=None):
    """Exhaustive placements in the specified chain domain; deduplicate mosaics.

    Both sides, every cross-axis window, every piece subset and chain order are
    searched. A root is nonflat and an outer cap is flat. No maximal-notch or
    width-to-site assignment is assumed. Frame cells are excluded from packing.
    """
    p = _parse(g) if p is None else p
    axis, bg = p['axis'], p['bg']
    base = p['base'] if axis == 0 else p['base'].T
    pieces = [(cells if axis == 0 else cells[:, ::-1], color)
              for cells, color in p['pieces']]
    h, w = base.shape
    candidates = []
    for sign in (-1, 1):
        descriptions = [_profile_description(cells, sign) for cells, _ in pieces]
        lookup = {}
        for i, desc in enumerate(descriptions):
            lookup.setdefault(desc[0], []).append(i)
        boundary = 1-p['edges'][:, 0] if sign < 0 else p['edges'][:, 1]
        for width in sorted({len(desc[0]) for desc in descriptions}):
            for start in range(w-width+1):
                root = tuple(map(int, boundary[start:start+width]))
                if len(set(root)) == 1:
                    continue

                def extend(profile, used, placements):
                    if used and len(set(profile)) == 1:
                        cells = frozenset((int(r), int(c))
                                          for _, a in placements for r, c in a)
                        candidates.append(dict(bits=used, cells=cells, placements=placements,
                                               side=sign, start=start, width=width,
                                               depth=profile[0]-min(root)))
                    offset = min(profile)
                    normalized = tuple(v-offset for v in profile)
                    for i in lookup.get(normalized, ()):
                        if used & (1 << i):
                            continue
                        _, upper, local = descriptions[i]
                        dest = np.column_stack((sign*(local[:, 0]+offset), local[:, 1]+start))
                        if np.any(dest[:, 0] < 0) or np.any(dest[:, 0] >= h):
                            continue
                        if np.any(base[tuple(dest.T)] != bg):
                            continue
                        if np.array_equal(dest.min(axis=0), pieces[i][0].min(axis=0)):
                            continue  # Every physical plate is translated.
                        extend(tuple(offset+v for v in upper), used | (1 << i),
                               placements+[(i, dest)])

                extend(root, 0, [])
    by_piece = [[t for t in candidates if t['bits'] & (1 << i)] for i in range(len(pieces))]
    full, solutions = (1 << len(pieces))-1, {}

    def cover(used, occupied, tabs):
        if len(solutions) >= 2:
            return
        if used == full:
            mosaic = base.copy()
            envelopes = {color: np.zeros_like(base, dtype=bool) for color in p['keys']}
            for t in tabs:
                for i, dest in t['placements']:
                    mosaic[tuple(dest.T)] = pieces[i][1]
                    r0, c0, r1, c1 = _bbox(dest)
                    envelopes[pieces[i][1]][r0:r1, c0:c1] = True
            # A colored mosaic alone could hide a same-color decomposition.
            # Stamping also requires unique envelope unions for every key.
            signature = mosaic.tobytes()+b''.join(envelopes[c].tobytes() for c in p['keys'])
            solutions.setdefault(signature, (mosaic, tabs))
            return
        options = min(([t for t in by_piece[i] if not (t['bits'] & used)
                        and not (t['cells'] & occupied)]
                       for i in range(len(pieces)) if not used & (1 << i)), key=len)
        for t in options:
            cover(used | t['bits'], occupied | t['cells'], tabs+[t])

    cover(0, frozenset(), [])
    assert len(solutions) == 1, "physical assembly must have one colored mosaic"
    mosaic, tabs = next(iter(solutions.values()))
    placements = {}
    for t in tabs:
        for i, dest in t['placements']:
            placements[i] = dest if axis == 0 else dest[:, ::-1]
    # A whole stamp box may meet the band but must never reach the table.
    for dest in placements.values():
        r0, c0, r1, c1 = _bbox(dest)
        assert not p['table_mask'][r0:r1, c0:c1].any()
    return dict(p, placements=placements, tabs=tabs,
                virtual=mosaic if axis == 0 else mosaic.T,
                candidates=len(candidates))


def _predict(p, *, reverse=False, masks=False, stationary=False, transparent=False):
    """Input-derived pixel traversal; does not call the construction compositor."""
    result = p['base'].copy()
    positions = {i: (cells if stationary else p['placements'][i])
                 for i, (cells, _) in enumerate(p['pieces'])}
    boxes = {i: _bbox(a) for i, a in positions.items()}
    cellsets = {i: set(map(tuple, a)) for i, a in positions.items()} if masks else {}
    order = list(zip(p['keys'], p['inks']))
    if reverse:
        order.reverse()
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            if p['fixed'][r, c]:
                continue
            for key, ink in order:
                if transparent and ink == p['bg']:
                    continue
                hit = False
                for i, (_, color) in enumerate(p['pieces']):
                    if color != key:
                        continue
                    r0, c0, r1, c1 = boxes[i]
                    if (r0 <= r < r1 and c0 <= c < c1
                            and (not masks or (r, c) in cellsets[i])):
                        hit = True
                        break
                if hit:
                    result[r, c] = ink
                    break
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover geometry and all paint conventions solely from the input."""
    g = _grid(input_grid)
    return _predict(_recover(g))


def _contour(rng, width, amplitude):
    mode = int(rng.integers(4))
    if mode == 0:
        a = rng.integers(0, amplitude+1, width)
    elif mode == 1:
        a = np.rint(np.linspace(0, amplitude, width)).astype(int)
        if rng.random() < .5:
            a = a[::-1]
    elif mode == 2:
        a = np.cumsum(rng.integers(-1, 2, width))
        a -= a.min()
        if a.max() > amplitude:
            a = np.rint(a*amplitude/a.max()).astype(int)
    else:
        a = np.zeros(width, dtype=int)
        cut = int(rng.integers(1, width))
        a[cut:] = amplitude
    a -= a.min()
    if not a.max():
        a[int(rng.integers(width))] = amplitude
    return a


def _cuts(rng, width, count, rich):
    cuts = [_contour(rng, width, int(rng.integers(1, 3)))]
    for _ in range(count-1):
        lo = cuts[-1]
        shape = _contour(rng, width, int(rng.integers(1, 3+int(rich))))
        offset = max(int(np.max(lo+1-shape)),
                     int(np.max(np.maximum(lo[:-1], lo[1:])+1-
                                np.minimum(shape[:-1], shape[1:]))))
        cuts.append(shape+offset+int(rng.integers(0, 2)))
    cuts.append(np.full(width, int(cuts[-1].max())+int(rng.integers(1, 3))))
    return cuts


def _stamp(base, plans, keys, inks, band_mask):
    """Ground truth from construction: whole-box overwrites, bottom to top."""
    result = base.copy()
    for key, ink in reversed(list(zip(keys, inks))):
        for cells, color in plans:
            if color == key:
                r0, c0, r1, c1 = _bbox(cells)
                result[r0:r1, c0:c1] = ink
    result[band_mask] = base[band_mask]
    return result


def _scatter(rng, base, plans, bg):
    h, w = base.shape
    shapes = [cells-cells.min(axis=0) for cells, _ in plans]
    for _ in range(16):
        result = base.copy()
        forbidden = _dilate(base != bg)
        order = list(rng.permutation(len(plans)))
        order.sort(key=lambda i: -len(shapes[i]))
        for i in order:
            shape = shapes[i]
            hh, ww = shape.max(axis=0)+1
            valid = np.ones((h-hh+1, w-ww+1), dtype=bool)
            for r, c in shape:
                valid &= ~forbidden[r:r+h-hh+1, c:c+w-ww+1]
            old = plans[i][0].min(axis=0)
            valid[tuple(old)] = False
            choices = np.argwhere(valid)
            if not len(choices):
                break
            dest = shape+choices[int(rng.integers(len(choices)))]
            result[tuple(dest.T)] = plans[i][1]
            mask = np.zeros_like(forbidden)
            mask[tuple(dest.T)] = True
            forbidden |= _dilate(mask)
        else:
            return result
    return None


def _candidate(rng, axis, counts, clearing, rich, equal_width,
               required_direction=None, teach_repeat=False):
    tab_count = len(counts)
    widths = [int(rng.integers(3, 8)) for _ in counts]
    if equal_width:
        widths = [widths[0]]*tab_count
    sides = list(map(int, rng.choice([-1, 1], tab_count)))
    cuts = [_cuts(rng, width, count, rich) for width, count in zip(widths, counts)]
    # Independently irregular unused boundary segments; they are never commands.
    min_cross = max(14, max(sum(width for width, s in zip(widths, sides) if s == sign)
                           + 2*sum(s == sign for s in sides)+2 for sign in (-1, 1)))
    if min_cross > 30:
        return None
    w = int(rng.integers(min_cross, 31))
    starts, occupied = [], {-1: set(), 1: set()}
    for width, sign in zip(widths, sides):
        choices = [x for x in range(w-width+1)
                   if not set(range(x-1, x+width+1)) & occupied[sign]]
        if not choices:
            return None
        x = int(rng.choice(choices))
        starts.append(x)
        occupied[sign].update(range(x, x+width))
    edge = {-1: np.zeros(w, dtype=int), 1: np.zeros(w, dtype=int)}
    for sign in (-1, 1):
        for _ in range(int(rng.integers(1, 5))):
            x = int(rng.integers(w))
            edge[sign][x:min(w, x+int(rng.integers(1, 4)))] = int(rng.integers(1, 3))
    extent = {-1: 2, 1: 2}
    for sign, x, width, cc in zip(sides, starts, widths, cuts):
        edge[sign][x:x+width] = cc[0]
        extent[sign] = max(extent[sign], int(cc[-1][0]))
    core = int(rng.integers(2, 5))
    # Reserve some outside space even for one-sided scenes, to fit the table.
    required = max(18, core+max(7, extent[-1])+max(7, extent[1]))
    if required > 30:
        return None
    h = int(rng.integers(required, 31))
    low = max(7, extent[-1])
    high = h-core-max(7, extent[1])
    center = int(rng.integers(low, high+1))
    bg, band = map(int, rng.choice(10, 2, replace=False))
    n = sum(counts)
    k = 2 if teach_repeat else int(rng.integers(2, min(6, n)+1))
    available = [c for c in range(10) if c not in (bg, band)]
    colors = list(map(int, rng.choice(available, k, replace=False)))
    assignments = colors+[int(rng.choice(colors)) for _ in range(n-k)]
    rng.shuffle(assignments)
    keys = list(map(int, rng.permutation(colors)))
    paint = [c for c in range(10) if c not in (bg, band)]
    inks = list(map(int, rng.choice(paint, k)))
    if clearing:
        inks[0] = bg
    # Guarantee at least two distinguishable paints; other rows can share ink.
    inks[-1] = int(rng.choice([c for c in paint if c != inks[0]]))
    base = np.full((h, w), bg, dtype=np.int8)
    for c in range(w):
        base[center-edge[-1][c]:center+core+edge[1][c], c] = band
    band_mask = base == band
    plans = []
    for sign, x, width, cc in zip(sides, starts, widths, cuts):
        for lo, hi in zip(cc, cc[1:]):
            cells = np.asarray([(center-1-u if sign < 0 else center+core+u, x+c)
                                for c in range(width) for u in range(int(lo[c]), int(hi[c]))])
            assert np.all(base[tuple(cells.T)] == bg)
            plans.append((cells, assignments[len(plans)]))
    # Reflect the geometric scene before inserting the always-upright table.
    if rng.random() < .5:
        base = base[:, ::-1].copy()
        band_mask = band_mask[:, ::-1].copy()
        plans = [(np.column_stack((a[:, 0], w-1-a[:, 1])), c) for a, c in plans]
    if axis:
        base, band_mask = base.T.copy(), band_mask.T.copy()
        plans = [(a[:, ::-1], c) for a, c in plans]
    blocked = band_mask.copy()
    for a, _ in plans:
        r0, c0, r1, c1 = _bbox(a)
        blocked[r0:r1, c0:c1] = True
    blocked = _dilate(blocked)
    h, w = base.shape
    locations = [(r, c) for r in range(h-k-1) for c in range(w-3)
                 if not blocked[r:r+k+2, c:c+4].any()]
    if not locations:
        return None
    tr, tc = locations[int(rng.integers(len(locations)))]
    base[tr:tr+k+2, tc:tc+4] = band
    base[tr+1:tr+k+1, tc+1] = keys
    base[tr+1:tr+k+1, tc+2] = inks
    inp = _scatter(rng, base, plans, bg)
    if inp is None:
        return None
    histogram = np.bincount(inp.ravel(), minlength=10)
    if histogram[bg] <= max(np.delete(histogram, bg)):
        return None
    out = _stamp(base, plans, keys, inks, band_mask)
    try:
        p = _recover(inp)
    except AssertionError:
        return None
    virtual = base.copy()
    for a, color in plans:
        virtual[tuple(a.T)] = color
    if not np.array_equal(p['virtual'], virtual):
        return None
    pair = {'input': inp.copy(), 'output': out.copy()}
    witnesses = _witnesses(p, out)
    if not {'priority', 'envelope', 'assembly'} <= witnesses:
        return None
    if clearing and 'clearing' not in witnesses:
        return None
    if required_direction is not None and required_direction not in witnesses:
        return None
    if teach_repeat and 'separate_objects' not in witnesses:
        return None
    return pair


def _witnesses(p, out):
    result = set()
    for name, kwargs in (('priority', {'reverse': True}),
                         ('envelope', {'masks': True}),
                         ('assembly', {'stationary': True}),
                         ('clearing', {'transparent': True})):
        if not np.array_equal(out, _predict(p, **kwargs)):
            result.add(name)
    if len(p['pieces']) > len(p['keys']):
        merged = [np.vstack([p['placements'][i] for i, (_, color) in enumerate(p['pieces'])
                             if color == key]) for key in p['keys']]
        alternative = dict(p, pieces=list(zip(merged, p['keys'])),
                           placements=dict(enumerate(merged)))
        if not np.array_equal(out, _predict(alternative)):
            result.add('separate_objects')
    rank = {key: i for i, key in enumerate(p['keys'])}
    ink = dict(zip(p['keys'], p['inks']))
    for tab in p['tabs']:
        for (i, _), (j, _) in zip(tab['placements'], tab['placements'][1:]):
            a, b = p['pieces'][i][1], p['pieces'][j][1]
            if ink[a] != ink[b]:
                result.add('inward_top' if rank[a] < rank[b] else 'outward_top')
    return result


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError('counts must be legal integers')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    # These choices change coverage, never the interpretation of an input.
    orientation_mode = int(rng.integers(5))
    train_axis = int(rng.integers(2))
    first_direction = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            direction, teach_repeat = None, False
            if orientation_mode < 2:
                axis = orientation_mode
            elif orientation_mode == 2:
                axis = train_axis if split == 'train' else 1-train_axis
            else:
                axis = (index+train_axis) % 2 if split == 'train' else int(rng.integers(2))
            if split == 'train' and index == 0:
                counts, clearing, rich, equal = [2], False, False, False
                direction = ('inward_top', 'outward_top')[first_direction]
            elif split == 'train' and index == 1:
                counts, clearing, rich, equal = [2, 1], True, False, True
                direction = ('outward_top', 'inward_top')[first_direction]
                teach_repeat = True
            elif split == 'test':
                recipes = ([3, 1], [3, 2], [2, 1, 2], [3, 3], [4], [4, 1], [3, 2, 1])
                counts = list(recipes[int(rng.integers(len(recipes)))])
                clearing, rich, equal = bool(rng.integers(2)), bool(rng.integers(2)), bool(rng.integers(2))
            else:
                counts = list(map(int, rng.integers(1, 5, int(rng.integers(1, 4)))))
                while sum(counts) > 8:
                    i = int(np.argmax(counts))
                    counts[i] -= 1
                if sum(counts) < 2:
                    counts = [2]
                if max(counts) < 2:
                    counts[0] = 2  # At least one jagged joint must affect paint.
                clearing, rich, equal = bool(rng.integers(2)), bool(rng.integers(2)), bool(rng.integers(2))
            for _ in range(240):
                pair = _candidate(rng, axis, counts, clearing, rich, equal,
                                  direction, teach_repeat)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded construction exhausted: no unvalidated fallback')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            p = _recover(inp)
            assert np.array_equal(_predict(p), out), 'incorrect print'
            assert np.array_equal(inp[p['fixed']], out[p['fixed']]), 'fixed cells changed'
            used = set()
            cover = np.zeros_like(inp, dtype=bool)
            for t in p['tabs']:
                for i, _ in t['placements']:
                    assert i not in used
                    used.add(i)
            assert used == set(range(len(p['pieces'])))
            for i, (source, color) in enumerate(p['pieces']):
                dest = p['placements'][i]
                delta = dest.min(axis=0)-source.min(axis=0)
                assert np.any(delta)
                assert set(map(tuple, source+delta)) == set(map(tuple, dest))
                assert np.all(p['virtual'][tuple(dest.T)] == color)
                r0, c0, r1, c1 = _bbox(dest)
                cover[r0:r1, c0:c1] = True
            assert np.array_equal(out[~cover], p['base'][~cover]), 'paint escaped stamp boxes'
            # Reconstruct every final rectangular tab and verify gaplessness.
            virtual = p['virtual'] if p['axis'] == 0 else p['virtual'].T
            band = p['band_mask'] if p['axis'] == 0 else p['band_mask'].T
            for t in p['tabs']:
                a = np.vstack([dest for _, dest in t['placements']])
                r0, c0, r1, c1 = _bbox(a)
                assert np.all((virtual[r0:r1, c0:c1] != p['bg']) | band[r0:r1, c0:c1])
            witnesses = _witnesses(p, out)
            assert {'assembly', 'envelope', 'priority'} <= witnesses
            if split == 'train':
                training |= witnesses
            elif p['bg'] in p['inks']:
                assert 'clearing' in training, 'opaque background paint was not demonstrated'
    assert {'assembly', 'envelope', 'priority', 'clearing', 'inward_top',
            'outward_top', 'separate_objects'} <= training


def self_test():
    """Bounded 42-seed semantic, variation, API and metamorphic checks."""
    shapes, structures, palettes, roots, counts, axes, depths = set(), set(), set(), set(), set(), set(), set()
    repeated = equal_width = multi = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for pair in puzzle['train']+puzzle['test']:
            p = _recover(pair['input'])
            shapes.add(pair['input'].shape)
            roles = {}
            norm = tuple(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)
            structures.add((pair['input'].shape, norm))
            palettes.add((p['bg'], p['band'], tuple(p['keys']), tuple(p['inks'])))
            axes.add(p['axis'])
            counts.add(len(p['pieces']))
            widths = [t['width'] for t in p['tabs']]
            multi += len(widths) > 1
            equal_width += len(set(widths)) < len(widths)
            repeated += len(p['pieces']) > len(p['keys'])
            depths.update(t['depth'] for t in p['tabs'])
            for t in p['tabs']:
                edge = 1-p['edges'][:, 0] if t['side'] < 0 else p['edges'][:, 1]
                a = edge[t['start']:t['start']+t['width']]
                roots.add(tuple(map(int, a-a.min())))
    assert len(shapes) >= 30 and len(structures) == 210
    assert len(palettes) >= 180 and len(roots) >= 35
    assert axes == {0, 1} and counts >= {2, 3, 4, 5, 6}
    assert multi >= 70 and equal_width >= 42 and repeated >= 42 and len(depths) >= 6
    first, again = generate_puzzle(0), generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in first
               for a, b in zip(first[s], again[s]) for k in ('input', 'output'))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(10000+10*a+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Literal colors are irrelevant, including black. Permute an entire pair.
    permutation = np.array([7, 2, 9, 0, 6, 1, 8, 4, 3, 5])
    pair = first['test'][0]
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # Changing just the table order changes the joint without changing packing.
    pair = first['train'][0]
    p = _recover(pair['input'])
    reordered = pair['input'].copy()
    r0, c0, r1, c1 = p['table']
    reordered[r0+1:r1-1, c0+1:c1-1] = reordered[r0+1:r1-1, c0+1:c1-1][::-1]
    expected = _predict(p, reverse=True)
    expected[p['table_mask']] = reordered[p['table_mask']]
    assert np.array_equal(solve(reordered), expected)
    assert not np.array_equal(expected[~p['fixed']], pair['output'][~p['fixed']])
    # Reject two indistinguishable roots that permit two colored assignments.
    ambiguous = np.zeros((20, 24), dtype=np.int8)
    ambiguous[7:10, :] = 8
    ambiguous[10, [5, 15]] = 8
    cap = np.asarray([(u, c) for c, low in enumerate((0, 1, 0)) for u in range(low, 4)])
    ambiguous[tuple(cap.T)] = 2
    ambiguous[cap[:, 0], cap[:, 1]+8] = 3
    ambiguous[14:18, 10:14] = 8
    ambiguous[15:17, 11] = (2, 3)
    ambiguous[15:17, 12] = (4, 5)
    try:
        _recover(ambiguous)
    except AssertionError as error:
        assert 'physical assembly' in str(error)
    else:
        raise AssertionError('ambiguous assembly accepted')
    broken = {s: [{k: v.copy() for k, v in x.items()} for x in first[s]] for s in first}
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect labels accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, all legal explicit counts)')
