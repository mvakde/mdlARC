"""Complete marker-associated shapes, then pack the whole resulting inventory.

No files, RNG, training examples or generation metadata are used by solve().
The intermediate symmetry geometry is deliberately not observable after packing.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_POOLS = (('H',), ('V',), ('P',), ('H', 'V'), ('H', 'P'), ('V', 'P'), ('H', 'V', 'P'))


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9))):
        raise ValueError('Expected an integer ARC grid with dimensions 1..30 and colors 0..9')
    return a.astype(np.int8, copy=False)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        q = min(remaining)
        remaining.remove(q)
        stack, group = [q], []
        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for dr, dc in _NEIGHBORS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(sorted(group))
    return result


def _cells(mask):
    return [tuple(map(int, q)) for q in np.argwhere(mask)]


def _kind(cells):
    if len(cells) == 1:
        return 'P'
    rs, cs = zip(*cells)
    if min(rs) == max(rs) and len(cells) == max(cs) - min(cs) + 1:
        return 'H'
    if min(cs) == max(cs) and len(cells) == max(rs) - min(rs) + 1:
        return 'V'
    return None


def _hist(a):
    return np.bincount(a.ravel(), minlength=10)


def _rank(hist):
    return tuple(sorted((c for c in range(10) if hist[c]), key=lambda c: (-int(hist[c]), c)))


def _distinct(hist):
    x = hist[hist > 0]
    return len(set(map(int, x))) == len(x)


def _pack(hist, shape):
    # Explicit raster construction; a run may cross any number of column edges.
    h, w = shape
    result = np.empty(shape, dtype=np.int8)
    i = 0
    for color in _rank(hist):
        for _ in range(int(hist[color])):
            result[h - 1 - i % h, i // h] = color
            i += 1
    assert i == h * w
    return result


def _read_scene(value):
    a = _array(value)
    hist = _hist(a)
    bg = int(np.argmax(hist))
    if hist[bg] * 2 <= a.size:
        raise ValueError('Background must occupy more than half of the input')
    # The marker is the unique non-background color whose every 8-component is
    # a straight line or singleton. Payloads and unmarked stock are non-linear.
    by_color = {int(c): _components(_cells(a == c)) for c in np.unique(a) if c != bg}
    candidates = [c for c, comps in by_color.items() if all(_kind(x) for x in comps)]
    if len(candidates) != 1:
        raise ValueError('Marker color must be uniquely recognizable by component geometry')
    marker = candidates[0]
    regions = _components(_cells(a != bg))
    groups, stock, destinations = [], [], set()
    for region in regions:
        marks = [q for q in region if int(a[q]) == marker]
        payload = [q for q in region if int(a[q]) != marker]
        if not marks:
            if len({int(a[q]) for q in region}) != 1 or _kind(region) is not None:
                raise ValueError('Unmarked stock must be a single non-linear monochrome shape')
            stock.append(region)
            continue
        kind = _kind(marks)
        if kind is None or not payload or len(_components(payload)) != 1:
            raise ValueError('Each marked region needs one straight/point marker and one payload')
        if len({int(a[q]) for q in payload}) != 1 or _kind(payload) is not None:
            raise ValueError('Payload must be a non-linear monochrome 8-connected shape')
        ar, ac = marks[0]
        rs, cs = zip(*payload)
        if kind == 'H':
            if not (max(rs) < ar or min(rs) > ar):
                raise ValueError('Horizontal payload must be one-sided')
            dest = [(2 * ar - r, c) for r, c in payload]
        elif kind == 'V':
            if not (max(cs) < ac or min(cs) > ac):
                raise ValueError('Vertical payload must be one-sided')
            dest = [(r, 2 * ac - c) for r, c in payload]
        else:
            if not ((max(rs) <= ar or min(rs) >= ar) and (max(cs) <= ac or min(cs) >= ac)):
                raise ValueError('Point payload must lie in one closed quadrant')
            dest = [(2 * ar - r, 2 * ac - c) for r, c in payload]
        h, w = a.shape
        if any(not (0 <= r < h and 0 <= c < w) or a[r, c] != bg for r, c in dest):
            raise ValueError('A copy is clipped or overwrites an input cell')
        if destinations.intersection(dest):
            raise ValueError('Copies collide')
        destinations.update(dest)
        groups.append({'kind': kind, 'marker': marks, 'source': payload,
                       'dest': dest, 'color': int(a[payload[0]])})
    if not groups:
        raise ValueError('At least one marked group is required')
    return a, bg, marker, groups, stock


def _complete(value):
    a, bg, marker, groups, stock = _read_scene(value)
    completed = a.copy()
    for group in groups:
        for source, dest in zip(group['source'], group['dest']):
            completed[dest] = a[source]
    return completed, (a, bg, marker, groups, stock)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles locally, add the prescribed copies, then rank and pack all cells.

    Tied positive completed counts are outside this family and raise ValueError;
    no literal-color tie breaker is part of the intended transformation.
    """
    completed, _ = _complete(input_grid)
    hist = _hist(completed)
    if not _distinct(hist):
        raise ValueError('Tied completed color frequencies are outside this family')
    return _pack(hist, completed.shape)


def _nonlinear_shape(rng, cap=5):
    """Small procedural silhouettes with density, topology and extent variation."""
    for _ in range(30):
        h, w = map(int, rng.integers(2, cap + 1, size=2))
        style = int(rng.integers(6))
        cells = set()
        if style == 0:  # filled rectangle, sometimes with an open corner cutout
            cells = {(r, c) for r in range(h) for c in range(w)}
            if rng.random() < .65:
                rh, cw = int(rng.integers(1, h)), int(rng.integers(1, w))
                cells -= {(r, c) for r in range(rh) for c in range(cw)}
        elif style == 1:  # hollow perimeter
            cells = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h - 1) or c in (0, w - 1)}
        elif style == 2:  # diagonal zigzag, connected in eight-neighbor geometry
            col, step = 0, 1
            for row in range(max(h, w)):
                cells.add((row, col))
                if not 0 <= col + step < w:
                    step = -step
                col += step
        elif style == 3:  # unequal L arms
            cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
        else:  # eight-neighbor growth, including sparse diagonal bridges
            cells = {(0, 0)}
            target = int(rng.integers(3, h * w + 1))
            while len(cells) < target:
                frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _NEIGHBORS
                                   if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
                cells.add(frontier[int(rng.integers(len(frontier)))])
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        cells = {(r - r0, c - c0) for r, c in cells}
        rh, cw = max(r for r, _ in cells), max(c for _, c in cells)
        if rng.random() < .5:
            cells = {(rh - r, c) for r, c in cells}
        if rng.random() < .5:
            cells = {(r, cw - c) for r, c in cells}
        if _kind(sorted(cells)) is None and len(_components(cells)) == 1:
            return sorted(cells)
    return [(0, 0), (1, 0), (1, 1)]


def _prototype(rng, kind, cap):
    source = _nonlinear_shape(rng, cap)
    if kind == 'P':
        # Connect a corner so point centers can touch diagonally or share a row.
        cells = set(source)
        target = min(cells)
        r = c = 0
        while (r, c) != target:
            cells.add((r, c))
            r += int(np.sign(target[0] - r))
            c += int(np.sign(target[1] - c))
        cells.add((0, 0))
        dr, dc = ((1, 1), (0, 1), (1, 0), (0, 0))[int(rng.integers(4))]
        cells = {(r + dr, c + dc) for r, c in cells} - {(0, 0)}
        if _kind(sorted(cells)) is not None or len(_components(cells)) != 1:
            cells = {(1, 1), (1, 2), (2, 1)}
        sr, sc = map(int, rng.choice([-1, 1], size=2))
        source = sorted((sr * r, sc * c) for r, c in cells)
        marker = [(0, 0)]
        dest = [(-r, -c) for r, c in source]
    else:
        contact = int(rng.choice([c for r, c in source if r == 0]))
        length = int(rng.integers(2, min(9, max(c for _, c in source) + 5)))
        start = contact - int(rng.integers(length))
        marker = [(0, c) for c in range(start, start + length)]
        sign = int(rng.choice([-1, 1]))
        source = [(sign * (r + 1), c) for r, c in source]
        dest = [(-r, c) for r, c in source]
        if kind == 'V':
            marker, source, dest = ([(c, r) for r, c in qs] for qs in (marker, source, dest))
    result = {'kind': kind, 'marker': marker, 'source': source, 'dest': dest}
    footprint = marker + source + dest
    top, left = min(r for r, _ in footprint), min(c for _, c in footprint)
    for role in ('marker', 'source', 'dest'):
        result[role] = [(r - top, c - left) for r, c in result[role]]
    return result


def _place(rng, prototypes, shape, style):
    h, w = shape
    blocked, placed = set(), [None] * len(prototypes)
    footprints = [p['marker'] + p['source'] + p['dest'] for p in prototypes]
    anchor = (int(rng.integers(h)), int(rng.integers(w)))
    for i in sorted(range(len(prototypes)), key=lambda j: -len(footprints[j])):
        p, cells = prototypes[i], footprints[i]
        gh, gw = 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)
        if gh > h or gw > w:
            return None
        origins = [(r, c) for r in range(h - gh + 1) for c in range(w - gw + 1)]
        rng.shuffle(origins)
        if style == 'edge':
            origins.sort(key=lambda q: min(q[0], q[1], h - gh - q[0], w - gw - q[1]))
        elif style == 'cluster':
            origins.sort(key=lambda q: abs(q[0] + gh / 2 - anchor[0]) + abs(q[1] + gw / 2 - anchor[1]))
        for top, left in origins:
            moved = [(r + top, c + left) for r, c in cells]
            if not blocked.intersection(moved):
                placed[i] = {role: [(r + top, c + left) for r, c in p[role]]
                             for role in ('marker', 'source', 'dest')}
                placed[i]['kind'], placed[i]['color'] = p['kind'], p['color']
                blocked.update((r + dr, c + dc) for r, c in moved
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1))
                break
        else:
            return None
    return placed


def _signature(a):
    return (a.shape, a.tobytes())


def _payload_inversion(before, after, bg, marker):
    colors = [c for c in range(10) if before[c] and c not in (bg, marker)]
    return any((before[a] - before[b]) * (after[a] - after[b]) < 0
               for i, a in enumerate(colors) for b in colors[i + 1:])


def _pair(rng, bg, marker, pool, kinds, require_inversion, stock_count, shape_hint=None):
    # Retry bounds apply to one live candidate. No candidate traces are retained.
    for attempt in range(320):
        cap = int(rng.choice([3, 4, 5], p=[.3, .45, .25]))
        active = [_prototype(rng, kind, cap) for kind in kinds]
        stocks = [{'kind': 'stock', 'marker': [], 'source': _nonlinear_shape(rng, cap), 'dest': []}
                  for _ in range(stock_count)]
        k = int(rng.integers(2, min(5, len(active) + stock_count) + 1))
        palette = [int(x) for x in rng.choice(pool, size=k, replace=False)]
        colors = [palette[i % k] for i in range(len(active) + stock_count)]
        rng.shuffle(colors)
        # Sharing an active color with a static shape makes global color doubling wrong.
        colors[len(active)] = colors[0]
        if len(set(colors)) < 2:
            colors[1] = palette[1]
        for p, color in zip(active + stocks, colors):
            p['color'] = color
        if shape_hint is not None and attempt % 4 == 0:
            shape = shape_hint
        else:
            shape = tuple(map(int, rng.integers(11, 29, size=2)))
        placed = _place(rng, active + stocks, shape, str(rng.choice(['free', 'edge', 'cluster'])))
        if placed is None:
            continue
        inp = np.full(shape, bg, dtype=np.int8)
        completed = inp.copy()
        for p in placed:
            for q in p['marker']:
                inp[q] = completed[q] = marker
            for q in p['source']:
                inp[q] = completed[q] = p['color']
            for q in p['dest']:
                completed[q] = p['color']
        before, after = _hist(inp), _hist(completed)
        if before[bg] * 2 <= inp.size or not _distinct(before) or not _distinct(after):
            continue
        if _rank(before) == _rank(after):
            continue
        if require_inversion and not _payload_inversion(before, after, bg, marker):
            continue
        # A nontrivial partial column is required in every candidate, in either split.
        boundaries = np.cumsum([after[c] for c in _rank(after)])[:-1] % shape[0]
        if not np.any((boundaries >= 2) & (boundaries <= shape[0] - 2)):
            continue
        pair = {'input': inp, 'output': _pack(after, shape)}
        # Re-read observable geometry rather than trusting the placement plan.
        try:
            parsed, _ = _complete(inp)
        except ValueError:
            continue
        if np.array_equal(parsed, completed):
            return pair
    raise RuntimeError('Bounded scene sampler exhausted its 320 candidates')


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer from {low} to {high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    order = list(map(int, rng.permutation(10)))
    marker, bg = order[:2]
    pool = order[2:]
    kinds = _POOLS[int(rng.integers(len(_POOLS)))]
    plans = [[] for _ in range(nt)]
    for i, kind in enumerate(rng.permutation(kinds)):
        plans[i % 2].append(str(kind))
    seen, puzzle = set(), {'train': [], 'test': []}
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            # Tests recombine trained operations and carry more unmarked stock;
            # familiar palettes and canvas sizes are deliberately still possible.
            specs = list(plans[i]) if split == 'train' else []
            n = max(len(specs), int(rng.integers(2, 6 if split == 'train' else 7)))
            while len(specs) < n:
                specs.append(str(rng.choice(kinds)))
            rng.shuffle(specs)
            stocks = (int(rng.integers(1, 3)) if split == 'train' else
                      (3 if i == 0 else int(rng.integers(2, 4))))
            hint = puzzle['train'][0]['input'].shape if split == 'test' else None
            for _ in range(12):
                pair = _pair(rng, bg, marker, pool, specs, True, stocks, hint)
                key = _signature(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('Could not generate distinct inputs')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _traversals(hist, shape):
    """Eight elementary rasters, used only to check demonstration ambiguity."""
    h, w = shape
    stream = np.repeat(np.array(_rank(hist)), [hist[c] for c in _rank(hist)])
    for major in ('column', 'row'):
        for rev_major in (False, True):
            for rev_minor in (False, True):
                xs = list(range(w if major == 'column' else h))[::(-1 if rev_major else 1)]
                ys = list(range(h if major == 'column' else w))[::(-1 if rev_minor else 1)]
                positions = [(y, x) if major == 'column' else (x, y) for x in xs for y in ys]
                out = np.empty(shape, dtype=np.int8)
                for q, color in zip(positions, stream):
                    out[q] = color
                yield (major, rev_major, rev_minor), out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, conventions, trained = set(), set(), set()
    rasters = None
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _array(pair['input']), _array(pair['output'])
            assert inp.shape == out.shape
            sig = _signature(inp)
            assert sig not in seen
            seen.add(sig)
            completed, (_, bg, marker, groups, stock) = _complete(inp)
            conventions.add((bg, marker))
            assert 2 <= len(groups) <= 6 and 1 <= len(stock) <= 3
            active_colors = {g['color'] for g in groups}
            assert any(int(inp[s[0]]) in active_colors for s in stock)
            before, after = _hist(inp), _hist(completed)
            assert _distinct(before) and _distinct(after)
            assert _payload_inversion(before, after, bg, marker), 'No payload-color rank reversal'
            # Independent count accounting: static shapes and marker cells count once.
            added = np.zeros(10, dtype=int)
            for g in groups:
                added[g['color']] += len(g['source'])
            formula = before + added
            formula[bg] -= int(added.sum())
            assert np.array_equal(formula, after)
            assert after[marker] == before[marker] and after[bg] > 0
            assert np.array_equal(_hist(out), after)
            assert np.array_equal(solve(inp), out)
            assert not np.array_equal(inp, out) and not np.array_equal(completed, out)
            assert not np.array_equal(_pack(before, inp.shape), out), 'Omitting A changes nothing'
            doubled = before.copy()
            for c in active_colors:
                doubled[c] *= 2
            doubled[bg] -= int(doubled.sum() - inp.size)
            assert not np.array_equal(doubled, after), 'Stock fails to refute color-wide doubling'
            if split == 'train':
                trained.update(g['kind'] for g in groups)
                matches = {key for key, a in _traversals(after, inp.shape) if np.array_equal(a, out)}
                rasters = matches if rasters is None else rasters & matches
            else:
                assert {g['kind'] for g in groups} <= trained
    assert len(conventions) == 1
    assert rasters == {('column', False, True)}, 'Training does not expose packing direction'


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, one puzzle at a time; bounded memory and no artifact writes."""
    shapes, counts, markers, backgrounds, silhouettes, palettes, mixes = (set() for _ in range(7))
    hashes = set()
    total = reversals = stock_reuse = edge = diagonal = holes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                complete, (_, bg, marker, groups, stock) = _complete(inp)
                shapes.add(inp.shape)
                counts.add(len(groups))
                markers.add(marker)
                backgrounds.add(bg)
                palettes.add(len(np.unique(inp)))
                mixes.add(tuple(sorted({g['kind'] for g in groups})))
                reversals += _payload_inversion(_hist(inp), _hist(complete), bg, marker)
                stock_reuse += any(int(inp[s[0]]) in {g['color'] for g in groups} for s in stock)
                edge += any(np.any(side != bg) for side in (complete[0], complete[-1], complete[:, 0], complete[:, -1]))
                # A separate crop-flip check exercises marker formulas before packing.
                reference = inp.copy()
                for g in groups:
                    rs, cs = zip(*g['source'])
                    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
                    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
                    for r, c in g['source']:
                        mask[r - r0, c - c0] = True
                    silhouettes.add((mask.shape, mask.tobytes()))
                    # Multiple 4-components detect genuinely diagonal connectivity.
                    foreground = set(_cells(mask))
                    stack = [min(foreground)]
                    reached = set(stack)
                    while stack:
                        r, c = stack.pop()
                        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                            q = (r + dr, c + dc)
                            if q in foreground and q not in reached:
                                reached.add(q)
                                stack.append(q)
                    diagonal += len(reached) < len(foreground)
                    holes += bool(mask.shape[0] >= 3 and mask.shape[1] >= 3
                                  and mask[0].all() and mask[-1].all()
                                  and mask[:, 0].all() and mask[:, -1].all()
                                  and not mask.all())
                    ar, ac = g['marker'][0]
                    if g['kind'] == 'H':
                        transformed, top, left = np.flipud(mask), 2 * ar - r1, c0
                    elif g['kind'] == 'V':
                        transformed, top, left = np.fliplr(mask), r0, 2 * ac - c1
                    else:
                        transformed, top, left = np.rot90(mask, 2), 2 * ar - r1, 2 * ac - c1
                    for r, c in _cells(transformed):
                        reference[top + r, left + c] = g['color']
                assert np.array_equal(reference, complete)
                # Count using a dictionary, then reshape a stream (different raster code).
                hist = {int(c): int(np.sum(reference == c)) for c in np.unique(reference)}
                stream = [c for c in sorted(hist, key=lambda c: -hist[c]) for _ in range(hist[c])]
                expected = np.array(stream, dtype=np.int8).reshape(inp.shape[1], inp.shape[0])[:, ::-1].T
                assert np.array_equal(expected, out)
                total += 1
    assert len(hashes) == 42 and len(shapes) >= 45 and len(silhouettes) >= 80
    assert counts >= {2, 3, 4, 5, 6} and len(markers) >= 8 and len(backgrounds) >= 8
    assert len(palettes) >= 3 and len(mixes) == 7
    assert reversals == stock_reuse == total and edge >= 20 and diagonal >= 20 and holes >= 5
    assert _fingerprint(generate_puzzle(160, 2, 1)) == _fingerprint(generate_puzzle(160, 2, 1))
    for nt, ne in ((2, 1), (3, 2), (10, 4)):
        validate_generated_puzzle(generate_puzzle(4321, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted illegal counts')
    # Wrong count, tie, and overwritten-destination regressions must fail closed.
    p = generate_puzzle(123, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['train'][0]['output'][0, 0] = (int(bad['train'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('Accepted corrupted output')
    tied = np.zeros((9, 11), dtype=np.int8)
    tied[3, 2:8] = 2
    tied[4, 3] = tied[5, 3] = tied[5, 4] = 3
    tied[7:9, 7:9] = 4
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted tied completed frequencies')
    complete, (_, bg, marker, groups, _) = _complete(p['train'][0]['input'])
    bad_input = p['train'][0]['input'].copy()
    bad_input[groups[0]['dest'][0]] = groups[0]['color']
    try:
        solve(bad_input)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted occupied copy destination')
    return {'seeds': 42, 'pairs': total, 'shapes': len(shapes), 'silhouettes': len(silhouettes),
            'group_counts': sorted(counts), 'palette_sizes': sorted(palettes), 'operation_mixtures': len(mixes),
            'marker_colors': len(markers), 'background_colors': len(backgrounds), 'rank_reversals': reversals,
            'stock_reuse': stock_reuse, 'border_contact': edge, 'diagonal_payloads': diagonal, 'hollow_payloads': holes}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
