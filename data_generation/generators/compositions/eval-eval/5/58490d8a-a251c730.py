"""Census blooms: counted objects control iterations of visible growth stencils.

Only the standard library and NumPy are used. No filesystem access is needed by
generation, solving, validation, or self_test. See task_spec.md for the rule.
"""
from __future__ import annotations

import hashlib
import numpy as np


_D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a


def _components(cells, diagonal=True):
    remaining, result = set(cells), []
    directions = _D8 if diagonal else _D4
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, component = [p], set()
        while stack:
            r, c = stack.pop()
            component.add((r, c))
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(component)
    return result


def _normal(cells):
    lr = min(r for r, c in cells)
    lc = min(c for r, c in cells)
    return tuple(sorted((r-lr, c-lc) for r, c in cells))


def _grow(stencil, rounds):
    """Union of all stencil walks of length at most rounds; round zero is root."""
    reached = {(0, 0)}
    for _ in range(rounds):
        reached |= {(r+dr, c+dc) for r, c in reached for dr, dc in stencil}
    return reached


def _bounds(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr)+1, max(cc)+1


def _border(a, color):
    return all(np.all(edge == color) for edge in (a[0], a[-1], a[:, 0], a[:, -1]))


def _parse(input_grid):
    """Recover panel, local dictionary, seeds, and census from input cells only."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(_border(g, bg), "canvas must have an uninterrupted background boundary")
    candidates = []
    for field in map(int, np.unique(g)):
        if field == bg:
            continue
        coords = [tuple(map(int, p)) for p in np.argwhere(g == field)]
        r0, c0, r1, c1 = _bounds(coords)
        if min(r1-r0, c1-c0) < 3 or r0 < 2 or c0 < 2 or r1+1 >= g.shape[0] or c1+1 >= g.shape[1]:
            continue
        inner = g[r0:r1, c0:c1]
        if not _border(inner, field):
            continue
        panel = g[r0-1:r1+1, c0-1:c1+1]
        ink = int(panel[0, 0])
        if ink in (bg, field) or not _border(panel, ink):
            continue
        positions = [tuple(map(int, p)) for p in np.argwhere(inner != field)]
        if not positions or any(len(comp) != 1 for comp in _components(positions)):
            continue
        seeds = [(r+1, c+1, int(inner[r, c])) for r, c in positions]
        if any(k in (bg, field, ink) for r, c, k in seeds):
            continue
        ring = g[r0-2:r1+2, c0-2:c1+2].copy()
        ring[1:-1, 1:-1] = bg
        if not np.all(ring == bg):
            continue
        candidates.append((r0-1, c0-1, r1+1, c1+1, field, ink, seeds))
    _require(len(candidates) == 1, "one visibly unique target panel is required")
    r0, c0, r1, c1, field, ink, seeds = candidates[0]
    exterior = g != bg
    exterior[r0:r1, c0:c1] = False
    components = _components(tuple(map(tuple, np.argwhere(exterior))))
    templates, repetitions, counters = {}, {}, {}
    for comp in components:
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) == 1:
            color = next(iter(colors))
            _require(color not in (field, ink), "exterior monochrome objects must be census colors")
            counters.setdefault(color, []).append(comp)
            continue
        _require(len(colors) == 2 and ink in colors and field not in colors,
                 "a reference must have one key color and the common ink")
        key = next(iter(colors - {ink}))
        roots = [(r, c) for r, c in comp if int(g[r, c]) == key]
        _require(len(roots) == 1 and len(comp) >= 3, "a reference needs one anchor and at least two ink cells")
        _require(len(_components(comp, False)) == 1, "reference motifs must be four-connected")
        ar, ac = roots[0]
        stencil = tuple(sorted((r-ar, c-ac) for r, c in comp if (r, c) != (ar, ac)))
        _require(key not in templates or templates[key] == stencil, "conflicting stencil definitions")
        templates[key] = stencil
        repetitions[key] = repetitions.get(key, 0) + 1
    keys = {k for r, c, k in seeds}
    _require(keys <= set(templates) and keys <= set(counters), "every seed needs a reference and a positive census")
    counts = {k: len(v) for k, v in counters.items()}
    # Bound malformed input work before any iterative growth. The generated
    # family uses 1..4, but up to 12 is safe for an input fitting the ARC limit.
    _require(all(counts[k] <= 12 for k in keys), "census exceeds supported family bounds")
    return dict(grid=g, background=bg, field=field, ink=ink, box=(r0, c0, r1, c1),
                panel=g[r0:r1, c0:c1].copy(), seeds=seeds, keys=keys,
                templates=templates, repetitions=repetitions, counters=counters, counts=counts)


def _paint(panel, seeds, templates, counts, ink):
    result = panel.copy()
    footprints = []
    for ar, ac, key in seeds:
        cells = {(ar+r, ac+c) for r, c in _grow(templates[key], counts[key])}
        _require(all(0 < r < panel.shape[0]-1 and 0 < c < panel.shape[1]-1 for r, c in cells),
                 "growth would cross the panel frame")
        for r, c in cells:
            result[r, c] = ink
        footprints.append(cells)
    # Roots are permanent markers, even if another bloom passes through them.
    for r, c, key in seeds:
        result[r, c] = key
    return result, footprints


def solve(input_grid: np.ndarray) -> np.ndarray:
    m = _parse(input_grid)
    return _paint(m['panel'], m['seeds'], m['templates'], m['counts'], m['ink'])[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _stencil(rng):
    """Connected procedural masks, with varied occupied anchors and holes."""
    height, width = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    n = int(rng.integers(3, min(7, height*width)+1))
    cells = {(int(rng.integers(height)), int(rng.integers(width)))}
    while len(cells) < n:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4
                           if 0 <= r+dr < height and 0 <= c+dc < width} - cells)
        cells.add(_pick(rng, frontier))
    # Avoid a filled bounding rectangle: transparent gaps are informative.
    a, b, d, e = _bounds(cells)
    if (d-a)*(e-b) == len(cells):
        corners = [(a, b), (a, e-1), (d-1, b), (d-1, e-1)]
        options = [p for p in corners if p in cells and len(_components(cells-{p}, False)) == 1]
        if len(cells) > 3 and options:
            cells.remove(_pick(rng, options))
        else:
            cells = {(0, 0), (0, 1), (1, 0)}
    anchor = _pick(rng, sorted(cells))
    return tuple(sorted((r-anchor[0], c-anchor[1]) for r, c in cells if (r, c) != anchor))


def _counter_shape(rng):
    mode = int(rng.integers(5))
    if mode == 0:
        return _DIAMOND
    if mode == 1:
        return _SQUARE
    if mode == 2:
        return ((0, 0),)
    if mode == 3:
        return tuple((0, c) for c in range(int(rng.integers(2, 4))))
    cells = {(int(rng.integers(3)), int(rng.integers(3)))}
    n = int(rng.integers(2, 8))
    while len(cells) < n:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D8
                           if 0 <= r+dr < 3 and 0 <= c+dc < 3} - cells)
        cells.add(_pick(rng, frontier))
    return _normal(cells)


def _layout(rng, templates, key_counts, mode, repeated=False):
    """Make a target around actual growth footprints; no clipping or wrapping."""
    ids = list(key_counts)
    rng.shuffle(ids)
    if repeated:
        ids.append(ids[0])
    shapes = [_grow(templates[k], key_counts[k]) for k in ids]
    boxes = [_bounds(s) for s in shapes]
    roots = []
    if mode == 'separate':
        horizontal = bool(rng.integers(2))
        cursor = 0
        cross = max((d-a if horizontal else e-b) for a, b, d, e in boxes)
        for a, b, d, e in boxes:
            other = int(rng.integers(cross-(d-a if horizontal else e-b)+1))
            roots.append((other-a, cursor-b) if horizontal else (cursor-a, other-b))
            cursor += (e-b if horizontal else d-a) + int(rng.integers(1, 3))
    else:
        # Align two real occupied cells of each pair, making a compatible
        # overlap. Keep roots distinct and nonadjacent in the sparse input.
        roots = [(0, 0)]
        for i in range(1, len(ids)):
            candidates = set()
            for r, c in shapes[0]:
                for dr, dc in shapes[i]:
                    ar, ac = r-dr, c-dc
                    if all(max(abs(ar-y), abs(ac-x)) > 1 for y, x in roots):
                        candidates.add((ar, ac))
            if not candidates:
                return None
            roots.append(_pick(rng, sorted(candidates)))
    absolute = [{(ar+r, ac+c) for r, c in cells} for (ar, ac), cells in zip(roots, shapes)]
    root_set = set(roots)
    # Every source must contribute visible ink of its own. A merge witness
    # must share ink, not merely pass through a permanent root marker.
    if any(len(cells-set().union(*(s for j, s in enumerate(absolute) if j != i))-root_set) < 2
           for i, cells in enumerate(absolute)):
        return None
    if mode == 'merge' and not any((a & b)-root_set for i, a in enumerate(absolute) for b in absolute[i+1:]):
        return None
    lr, lc, hr, hc = _bounds(set().union(*absolute))
    top, left = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    bottom, right = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    h, w = hr-lr+top+bottom, hc-lc+left+right
    # Exterior packing has a guaranteed fallback for these maximum sizes.
    if h > 19 or w > 21:
        return None
    seeds = [(ar-lr+top, ac-lc+left, k) for (ar, ac), k in zip(roots, ids)]
    return (h, w), seeds


def _place_exterior(rng, grid, box, objects, fallback=False):
    r0, c0, r1, c1 = box
    blocked = np.zeros(grid.shape, dtype=bool)
    blocked[0] = blocked[-1] = True
    blocked[:, 0] = blocked[:, -1] = True
    blocked[r0-1:r1+1, c0-1:c1+1] = True
    if fallback:
        slots = [(r, c) for r in range(1, 27, 4) for c in range(1, 27, 4)
                 if not blocked[r:r+3, c:c+3].any()]
        if len(slots) < len(objects):
            return False
        rng.shuffle(slots)
        for cells, (r, c) in zip(objects, slots):
            for dr, dc, color in cells:
                grid[r+dr, c+dc] = color
        return True
    order = list(objects)
    rng.shuffle(order)
    order.sort(key=lambda s: -(1+max(r for r, c, k in s))*(1+max(c for r, c, k in s)))
    for cells in order:
        sh, sw = 1+max(r for r, c, k in cells), 1+max(c for r, c, k in cells)
        nh, nw = grid.shape[0]-sh+1, grid.shape[1]-sw+1
        allowed = np.ones((nh, nw), dtype=bool)
        for r, c, color in cells:
            allowed &= ~blocked[r:r+nh, c:c+nw]
        locations = np.argwhere(allowed)
        if not len(locations):
            return False
        ar, ac = map(int, locations[int(rng.integers(len(locations)))])
        for dr, dc, color in cells:
            r, c = ar+dr, ac+dc
            grid[r, c] = color
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return True


def _make_pair(rng, kind, shift, ignored_color=None):
    witness = kind == 'census'
    nkeys = 3 if kind == 'test' and shift == 'types' else 2
    pool = list(map(int, rng.permutation(10)))
    if ignored_color is not None:
        pool.remove(ignored_color)
        pool.append(ignored_color)
    bg, field, ink = pool[:3]
    keys = pool[3:3+nkeys]
    distractor_pool = pool[3+nkeys:]
    if ignored_color is not None:
        distractors = [ignored_color]
    else:
        distractors = distractor_pool[:int(rng.integers(0, 2))]
    counts = {k: int(rng.integers(1, 4)) for k in keys}
    if witness:
        counts[keys[0]], counts[keys[1]] = 3, 1
    elif kind == 'merge':
        counts[keys[0]], counts[keys[1]] = 2, 3
    elif kind == 'test' and shift == 'rounds':
        counts[_pick(rng, keys)] = 4
    # Bounded total population leaves room for references and distractors.
    while sum(counts.values()) > 9:
        eligible = [k for k in keys if counts[k] > 1 and counts[k] < 4]
        counts[_pick(rng, eligible)] -= 1
    templates = {k: _stencil(rng) for k in keys}
    if witness:
        templates[keys[1]] = templates[keys[0]]
    if kind == 'merge' and len(set(templates.values())) == 1:
        # The two core training examples must not imply one universal stencil.
        old = templates[keys[1]]
        for _ in range(16):
            templates[keys[1]] = _stencil(rng)
            if templates[keys[1]] != old:
                break
        if templates[keys[1]] == old:
            templates[keys[1]] = ((0, -1), (1, 0)) if old != ((0, -1), (1, 0)) else ((-1, 0), (0, 1))
    mode = 'separate' if witness else ('merge' if kind == 'merge' or nkeys == 3 or rng.random() < .65 else 'separate')
    repeated = kind not in ('census', 'merge') and nkeys == 2 and rng.random() < .23
    target = _layout(rng, templates, counts, mode, repeated)
    if target is None:
        return None
    (ph, pw), seeds = target
    panel = np.full((ph, pw), field, dtype=np.int8)
    panel[0] = panel[-1] = ink
    panel[:, 0] = panel[:, -1] = ink
    for r, c, k in seeds:
        panel[r, c] = k
    output, _ = _paint(panel, seeds, templates, counts, ink)
    objects = []
    for k in keys:
        for j in range(counts[k]):
            shape = (_DIAMOND, _SQUARE, ((0, 0),))[j] if witness and k == keys[0] else _counter_shape(rng)
            objects.append(tuple((r, c, k) for r, c in shape))
        stencil = templates[k]
        raw = [(0, 0, k)] + [(r, c, ink) for r, c in stencil]
        lr, lc, hr, hc = _bounds([(r, c) for r, c, color in raw])
        glyph = tuple((r-lr, c-lc, color) for r, c, color in raw)
        for _ in range(2 if witness else int(rng.integers(1, 3))):
            objects.append(glyph)
    for k in distractors:
        for _ in range(int(rng.integers(1, 3))):
            objects.append(tuple((r, c, k) for r, c in _counter_shape(rng)))
    for attempt in range(7):
        fallback = attempt == 6
        if fallback:
            h = w = 30
            r0 = c0 = 1
        else:
            h = int(rng.integers(max(18, ph+4), 31))
            w = int(rng.integers(max(18, pw+4), 31))
            r0 = int(rng.integers(2, h-ph-1))
            c0 = int(rng.integers(2, w-pw-1))
        g = np.full((h, w), bg, dtype=np.int8)
        g[r0:r0+ph, c0:c0+pw] = panel
        box = (r0, c0, r0+ph, c0+pw)
        if _place_exterior(rng, g, box, objects, fallback):
            break
    else:
        return None
    # Whole-scene isometries preserve literal orientation between stencil and
    # growth. They also change the panel's position and aspect without a code.
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    if reflect:
        g, output = np.fliplr(g), np.fliplr(output)
    g, output = np.rot90(g, turns).copy(), np.rot90(output, turns).copy()
    _require(np.array_equal(solve(g), output), "construction disagrees with visible input")
    return {'input': g, 'output': output}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(name+' must be an integer in the supported range')
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = _pick(rng, ('rounds', 'types'))
    puzzle, seen = {'train': [], 'test': []}, set()
    first_key = None
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            kind = ('census' if index == 0 else 'merge' if index == 1 else 'other') if split == 'train' else 'test'
            for _ in range(80):
                pair = _make_pair(rng, kind, shift, first_key if kind == 'merge' else None)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if kind == 'census':
                first_key = min(_parse(pair['input'])['keys'])
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(m):
    output, footprints = _paint(m['panel'], m['seeds'], m['templates'], m['counts'], m['ink'])
    shapes = {k: [_normal(c) for c in m['counters'][k]] for k in m['keys']}
    witness = any(_DIAMOND in ss and _SQUARE in ss and ((0, 0),) in ss for ss in shapes.values())
    roots = {(r, c) for r, c, k in m['seeds']}
    overlap = any((a & b)-roots for i, a in enumerate(footprints) for b in footprints[i+1:])
    no_a_diff = any(_grow(m['templates'][k], m['counts'][k]) != _grow(m['templates'][k], 1) for k in m['keys'])
    bbox_diff = False
    for k in m['keys']:
        stencil = m['templates'][k]
        a, b, d, e = _bounds(set(stencil) | {(0, 0)})
        filled = tuple((r, c) for r in range(a, d) for c in range(b, e) if (r, c) != (0, 0))
        bbox_diff |= _grow(stencil, m['counts'][k]) != _grow(filled, m['counts'][k])
    return dict(expected=output, witness=witness, overlap=overlap, no_a_diff=no_a_diff,
                bbox_diff=bbox_diff, shapes=shapes, footprints=footprints)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "wrong pair keys")
            g, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(g) not in seen, "duplicate input")
            seen.add(_key(g))
            m = _parse(g)
            f = _features(m)
            _require(np.array_equal(out, f['expected']), "wrong bloom or crop")
            _require(all(1 <= m['counts'][k] <= 4 for k in m['keys']), "generated census range")
            _require(all(_bounds(set(s) | {(0, 0)})[2]-_bounds(set(s) | {(0, 0)})[0] <= 3
                         and _bounds(set(s) | {(0, 0)})[3]-_bounds(set(s) | {(0, 0)})[1] <= 3
                         for s in m['templates'].values()), "generated stencil extent")
            for r, c, k in m['seeds']:
                _require(out[r, c] == k, "root overwritten")
            roots = {(r, c) for r, c, k in m['seeds']}
            _require(all(len(cells-set().union(*(s for j, s in enumerate(f['footprints']) if j != i))-roots) >= 2
                         for i, cells in enumerate(f['footprints'])), "a bloom has no distinctive visible ink")
            _require(_border(out, m['ink']), "frame changed")
            if split == 'train':
                train.append((m, f))
    _require(any(f['witness'] for m, f in train), "training lacks census connectivity/area witness")
    _require(any(f['overlap'] for m, f in train), "training lacks compatible merge")
    _require(any(f['no_a_diff'] and f['bbox_diff'] for m, f in train), "both parent mechanisms must matter")
    _require(any(1 in [m['counts'][k] for k in m['keys']] for m, f in train), "training lacks one-round copying")
    _require(any(len({m['templates'][k] for k in m['keys']}) > 1 for m, f in train),
             "training lacks key-specific geometry")
    _require(any(a['keys'] & (set(b['counters'])-b['keys']) for a, f in train for b, ff in train),
             "training lacks counted/ignored color transfer")


def self_test():
    """Stream 42 seeds; store only bounded feature sets and hashes, no grids."""
    signatures, inputs, outputs, stencils, object_shapes = set(), set(), set(), set(), set()
    roles, seed_totals, overlaps, max_counts, key_counts = set(), set(), set(), set(), set()
    shifts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _puzzle_digest(puzzle)
        signatures.add(digest)
        if seed in (0, 17, 41):
            assert digest == _puzzle_digest(generate_puzzle(seed))
        by_split = {}
        for split in ('train', 'test'):
            records = []
            for pair in puzzle[split]:
                g = pair['input']
                before = g.copy()
                predicted = solve(g)
                assert np.array_equal(g, before) and np.array_equal(predicted, pair['output'])
                m, f = _parse(g), _features(_parse(g))
                inputs.add(g.shape)
                outputs.add(predicted.shape)
                stencils.update(m['templates'].values())
                object_shapes.update(s for ss in f['shapes'].values() for s in ss)
                roles.add((m['background'], m['field'], m['ink']))
                seed_totals.add(len(m['seeds']))
                overlaps.add(f['overlap'])
                max_counts.add(max(m['counts'][k] for k in m['keys']))
                key_counts.add(len(m['keys']))
                # A's connectivity distinction is measured on the counted
                # components themselves, not generic foreground statistics.
                if f['witness']:
                    k = next(k for k, ss in f['shapes'].items() if _DIAMOND in ss and _SQUARE in ss and ((0, 0),) in ss)
                    n4 = sum(len(_components(comp, False)) for comp in m['counters'][k])
                    area = sum(map(len, m['counters'][k]))
                    assert n4 != m['counts'][k] and area != m['counts'][k]
                    assert _grow(m['templates'][k], n4) != _grow(m['templates'][k], m['counts'][k])
                records.append(m)
            by_split[split] = records
        tm = max(m['counts'][k] for m in by_split['train'] for k in m['keys'])
        qm = max(m['counts'][k] for m in by_split['test'] for k in m['keys'])
        tk = max(len(m['keys']) for m in by_split['train'])
        qk = max(len(m['keys']) for m in by_split['test'])
        shifts.add((qm > tm, qk > tk))
    assert len(signatures) == 42
    assert len(inputs) >= 30 and len(outputs) >= 35
    assert len(stencils) >= 60 and len(object_shapes) >= 35
    assert len(roles) >= 100 and seed_totals >= {2, 3}
    assert overlaps == {False, True} and max_counts >= {2, 3, 4} and key_counts == {2, 3}
    assert shifts == {(True, False), (False, True)}
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(4200+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal example counts accepted')
    # Semantic regression independent of generated labels: an L stencil grows
    # triangles, not repeated disjoint copies or a filled bounding rectangle.
    assert _grow(((0, 1), (1, 0)), 3) == {(r, c) for r in range(4) for c in range(4-r)}
    assert len(_components(_DIAMOND)) == 1 and len(_components(_DIAMOND, False)) == 4
    p = generate_puzzle(0)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds and all legal count combinations)')
