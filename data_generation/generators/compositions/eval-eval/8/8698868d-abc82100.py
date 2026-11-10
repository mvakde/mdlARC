"""Self-addressed printing plates: topology selects a registered pinhole print.

No files are read or written. Construction and input decoding are separate.
All temporary grids are <=30x30; self_test retains only aggregate measurements.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = DIRS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else DIRS
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.add(p)
        result.append(cells)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _holes(mask):
    h, w = mask.shape
    return [c for c in _components(~mask)
            if all(0 < r < h - 1 and 0 < s < w - 1 for r, s in c)]


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid shape')
    _require(np.issubdtype(a.dtype, np.integer), 'noninteger grid')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'invalid color')
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _decode(input_grid):
    """Infer every role, plate, cavity count and registration from one input."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(bool(np.all(_perimeter(g) == bg)), 'family has a blank outer border')
    h, w = g.shape
    records = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for cells in _components(g == color):
            r0, r1, c0, c1 = _box(cells)
            bh, bw = r1 - r0, c1 - c0
            if not (4 <= bh <= 7 and 4 <= bw <= 7):
                continue
            plaque = g[r0:r1, c0:c1]
            if not np.all(_perimeter(plaque) == color):
                continue
            if not set(map(int, np.unique(plaque))) <= {bg, color}:
                continue
            rays = []
            # Labels always approach perpendicular to an edge. The next cell
            # is on that edge, or is the one-cell exterior registration hole.
            edges = [((r, c0), (0, 1)) for r in range(r0, r1)]
            edges += [((r, c1 - 1), (0, -1)) for r in range(r0, r1)]
            edges += [((r0, c), (1, 0)) for c in range(c0, c1)]
            edges += [((r1 - 1, c), (-1, 0)) for c in range(c0, c1)]
            for (er, ec), (dr, dc) in edges:
                for gap in (0, 1):
                    q = er - gap * dr, ec - gap * dc
                    tl = q[0] - dr, q[1] - dc
                    sl = q[0] - 2 * dr, q[1] - 2 * dc
                    if not all(0 <= r < h and 0 <= c < w for r, c in (q, tl, sl)):
                        continue
                    source, target = int(g[sl]), int(g[tl])
                    if source in (bg, color) or target in (bg, color, source):
                        continue
                    if gap and int(g[q]) != bg:
                        continue
                    rays.append((q, sl, tl, source, target, (dr, dc), gap))
            if not rays:
                continue
            _require(len(rays) == 1, 'ambiguous stencil label ray')
            q, sl, tl, source, target, direction, gap = rays[0]
            cavities = _components(plaque == bg)
            _require(1 <= len(cavities) <= 4, 'stencil must have 1..4 cavities')
            records.append({'cells': cells, 'box': (r0, r1, c0, c1), 'q': q,
                            'source_label': sl, 'target_label': tl,
                            'source': source, 'target': target, 'neutral': color,
                            'direction': direction, 'gap': gap, 'count': len(cavities),
                            'cavities': cavities, 'cavity_mask': plaque == bg,
                            'offsets': tuple(sorted((r - q[0], c - q[1]) for r, c in cells))})
    _require(2 <= len(records) <= 3, 'expected two or three stencil instructions')
    _require(len({r['neutral'] for r in records}) == 1, 'stencils must share a color')
    _require(len({r['source'] for r in records}) == 1, 'stencils must address one carrier color')
    _require(len({r['count'] for r in records}) == len(records), 'cavity count tie')
    neutral, source = records[0]['neutral'], records[0]['source']
    by_count = {r['count']: r for r in records}
    labels = {p for r in records for p in (r['source_label'], r['target_label'])}
    _require(len(labels) == 2 * len(records), 'shared labels')
    _require(set().union(*(r['cells'] for r in records)) ==
             set(map(tuple, np.argwhere(g == neutral).tolist())), 'unassigned neutral pixels')
    plates = []
    for cells in _components(g == source):
        if cells & labels:
            _require(len(cells) == 1 and cells <= labels, 'source label touches a plate')
            continue
        _require(len(cells) >= 16, 'unassigned source pixels')
        r0, r1, c0, c1 = _box(cells)
        block = g[r0:r1, c0:c1]
        _require(set(map(int, np.unique(block))) <= {bg, source}, 'foreign content in plate box')
        cavities = _holes(block == source)
        _require(all(len(c) == 1 for c in cavities), 'plate holes must be single pixels')
        anchors = sorted((r0 + next(iter(c))[0], c0 + next(iter(c))[1]) for c in cavities)
        _require(len(anchors) in by_count, 'no stencil for plate pinhole count')
        support = cells | set(anchors)
        plates.append({'support': support, 'anchors': anchors, 'count': len(anchors),
                       'box': (r0, r1, c0, c1), 'record': by_count[len(anchors)]})
    _require(1 <= len(plates) <= 3, 'expected one to three plates')
    # Complete accounting prevents unknown pixels from being silently ignored.
    accounted = labels | set().union(*(r['cells'] for r in records))
    accounted |= set().union(*(p['support'] - set(p['anchors']) for p in plates))
    _require(accounted == set(map(tuple, np.argwhere(g != bg).tolist())), 'unassigned input content')
    boxes = [r['box'] for r in records] + [p['box'] for p in plates]
    # Plate boxes and plaques are disjoint even when their colors repeat.
    for i, (a, b, c, d) in enumerate(boxes):
        for e, f, j, k in boxes[i + 1:]:
            _require(b < e or f < a or d < j or k < c, 'objects need a blank box gap')
    return {'grid': g, 'bg': bg, 'source': source, 'neutral': neutral,
            'records': records, 'plates': plates}


def _render(decoded, variant=None):
    """Input-derived printing, plus explicit competing rules for validation."""
    g, bg = decoded['grid'], decoded['bg']
    out = np.full(g.shape, bg, dtype=np.int8)
    records = decoded['records']
    for plate in decoded['plates']:
        for p in plate['support']:
            out[p] = decoded['source']
        record = plate['record']
        if variant == 'ignore_count':
            record = min(records, key=lambda r: r['count'])
        elif variant in ('area_rank', 'foreground_rank', 'box_rank'):
            keys = sorted(r['count'] for r in records)
            def area(r):
                if variant == 'foreground_rank':
                    return len(r['cells'])
                if variant == 'box_rank':
                    return r['cavity_mask'].size
                return int(r['cavity_mask'].sum())
            record = sorted(records, key=lambda r: (area(r), r['count']))[keys.index(plate['count'])]
        offsets = record['offsets']
        if variant == 'center':
            a, b, c, d = record['box']
            offsets = tuple((r - (a + b - 1) // 2, s - (c + d - 1) // 2)
                            for r, s in record['cells'])
        elif variant == 'top_left':
            a, b, c, d = record['box']
            offsets = tuple((r - a, s - c) for r, s in record['cells'])
        elif variant == 'ignore_registration_hole':
            dr, dc = record['direction']
            offsets = tuple((r - record['gap'] * dr, c - record['gap'] * dc) for r, c in offsets)
        anchors = plate['anchors']
        if variant == 'one_stamp':
            anchors = anchors[:1]
        for ar, ac in anchors:
            if variant == 'opaque':
                # Competing opaque copying would erase earlier ink at the
                # stencil's background pixels. The actual rule unions ink.
                a, b, c, d = record['box']
                qr, qc = record['q']
                for r in range(a - qr, b - qr):
                    for s in range(c - qc, d - qc):
                        if (ar + r, ac + s) in plate['support']:
                            out[ar + r, ac + s] = decoded['source']
            for dr, dc in offsets:
                p = ar + dr, ac + dc
                if p in plate['support'] or (variant == 'canvas_clip' and
                        0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]):
                    out[p] = record['target']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pure input-only solution; no puzzle, seed, examples or files are needed."""
    return _render(_decode(input_grid))


def _choice(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _cavity_mask(rng, h, w, count):
    if count == 1 and rng.random() < .38:
        return np.pad(np.ones((h - 2, w - 2), dtype=bool), 1)
    for _ in range(32):
        mask = np.zeros((h, w), dtype=bool)
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        for index in rng.permutation(len(cells)):
            p = cells[int(index)]
            if any(0 <= p[0] + dr < h and 0 <= p[1] + dc < w and
                   mask[p[0] + dr, p[1] + dc] for dr, dc in DIRS):
                continue
            mask[p] = True
            if len(_components(~mask)) != 1:
                mask[p] = False
            if int(mask.sum()) == count:
                break
        if int(mask.sum()) == count:
            break
    else:
        mask[:] = False
        for p in [(1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)][:count]:
            mask[p] = True
    wanted = int(rng.integers(count, (h - 2) * (w - 2) + 1))
    for index in rng.permutation((h - 2) * (w - 2)):
        if int(mask.sum()) >= wanted:
            break
        r, c = divmod(int(index), w - 2)
        p = r + 1, c + 1
        if mask[p]:
            continue
        trial = mask.copy()
        trial[p] = True
        if len(_components(trial)) == count and len(_components(~trial)) == 1:
            mask = trial
    return mask


def _stencil(rng, count, bg, neutral, source, target, witness=False, size=None):
    if witness:
        h = w = 5
        holes = np.zeros((5, 5), dtype=bool)
        if count == 1:
            holes[1:-1, 1:-1] = True
        else:
            holes[1, 1] = holes[2, 2] = holes[2, 3] = True
            holes = np.rot90(holes, int(rng.integers(4))).copy()
            if rng.random() < .5:
                holes = holes.T.copy()
    else:
        h, w = size if size is not None else (int(rng.integers(5, 8)), int(rng.integers(5, 8)))
        holes = _cavity_mask(rng, h, w, count)
    direction = _choice(rng, DIRS)
    dr, dc = direction
    # First witness includes both occupied and exterior-hole registrations.
    gap = int(count == 2) if witness else int(rng.random() < .4)
    if dr == 0:
        er, ec = int(rng.integers(1, h - 1)), 0 if dc == 1 else w - 1
    else:
        er, ec = 0 if dr == 1 else h - 1, int(rng.integers(1, w - 1))
    q = er - gap * dr, ec - gap * dc
    sl, tl = (q[0] - 2 * dr, q[1] - 2 * dc), (q[0] - dr, q[1] - dc)
    low_r, low_c = min(0, sl[0]), min(0, sl[1])
    high_r, high_c = max(h - 1, sl[0]), max(w - 1, sl[1])
    tile = np.full((high_r - low_r + 1, high_c - low_c + 1), bg, dtype=np.int8)
    r0, c0 = -low_r, -low_c
    tile[r0:r0 + h, c0:c0 + w] = np.where(holes, bg, neutral)
    tile[sl[0] - low_r, sl[1] - low_c] = source
    tile[tl[0] - low_r, tl[1] - low_c] = target
    cells = set(map(tuple, np.argwhere(~holes).tolist()))
    offsets = tuple(sorted((r - q[0], c - q[1]) for r, c in cells))
    return {'tile': tile, 'count': count, 'offsets': offsets, 'target': target,
            'mask': holes, 'gap': gap, 'direction': direction}


def _plate_shape(rng, h, w, kind):
    mask = np.ones((h, w), dtype=bool)
    if kind == 'bevel':
        depth = int(rng.integers(2, 4))
        for r in range(depth):
            mask[r, :depth - r] = False
            if rng.random() < .6:
                mask[-1 - r, w - depth + r:] = False
    elif kind == 'ell':
        mask[:int(rng.integers(3, h - 3)), int(rng.integers(4, w - 2)):] = False
    elif kind == 'cross':
        a, b = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        mask[:a, :b] = mask[:a, -b:] = False
        mask[-a:, :b] = mask[-a:, -b:] = False
    elif kind == 'step':
        for r in range(h // 2):
            mask[r, w - 1 - r // 2:] = False
    if rng.random() < .5:
        mask = np.flipud(mask)
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return mask.copy()


def _plate(rng, stencil, bg, source, kind, complete=False, clipping=False, clustered=False):
    h, w = int(rng.integers(9, 12)), int(rng.integers(9, 12))
    mask = _plate_shape(rng, h, w, kind)
    support = set(map(tuple, np.argwhere(mask).tolist()))
    eligible = [p for p in sorted(support) if all((p[0] + dr, p[1] + dc) in support for dr, dc in DIRS)]
    full = [p for p in eligible if all((p[0] + dr, p[1] + dc) in support for dr, dc in stencil['offsets'])]
    clipped = [p for p in eligible if p not in full]
    count = stencil['count']
    for _ in range(40):
        pool = full if complete else eligible
        if clipping:
            pool = clipped
        if not pool:
            return None
        anchors = [_choice(rng, pool)]
        for _ in range(count - 1):
            options = [p for p in (full if complete else eligible) if p not in anchors and
                       all(abs(p[0] - r) + abs(p[1] - c) != 1 for r, c in anchors)]
            if not options:
                break
            if clustered:
                diagonal = [p for p in options if any(abs(p[0] - r) == abs(p[1] - c) == 1 for r, c in anchors)]
                if diagonal:
                    options = diagonal
            anchors.append(_choice(rng, options))
        if len(anchors) != count:
            continue
        visible = mask.copy()
        for p in anchors:
            visible[p] = False
        if len(_components(visible)) != 1 or len(_holes(visible)) != count:
            continue
        tile = np.where(visible, source, bg).astype(np.int8)
        out = np.where(mask, source, bg).astype(np.int8)
        # Direct construction uses sampled geometry, not the input decoder.
        for ar, ac in anchors:
            for dr, dc in stencil['offsets']:
                if (ar + dr, ac + dc) in support:
                    out[ar + dr, ac + dc] = stencil['target']
        return tile, out
    return None


def _pack(rng, objects, bg):
    """Free bounding-box placement in a bounded 28x28 inner workspace."""
    order = sorted(range(len(objects)), key=lambda i: objects[i][0].size, reverse=True)
    for _ in range(40):
        boxes, locations = [], {}
        for index in order:
            h, w = objects[index][0].shape
            rr, cc = np.indices((29 - h, 29 - w))
            free = np.ones(rr.shape, dtype=bool)
            for r, c, bh, bw in boxes:
                free &= (rr + h < r) | (rr > r + bh) | (cc + w < c) | (cc > c + bw)
            choices = np.argwhere(free)
            if not len(choices):
                break
            r, c = map(int, choices[int(rng.integers(len(choices)))])
            locations[index] = r, c
            boxes.append((r, c, h, w))
        if len(locations) == len(objects):
            break
    else:
        return None
    low_r, low_c = min(r for r, c, h, w in boxes), min(c for r, c, h, w in boxes)
    high_r, high_c = max(r + h for r, c, h, w in boxes), max(c + w for r, c, h, w in boxes)
    # Whole objects move together; their visible stencil orientation is retained.
    top = int(rng.integers(1, min(4, 30 - (high_r - low_r))))
    left = int(rng.integers(1, min(4, 30 - (high_c - low_c))))
    bottom = int(rng.integers(1, min(4, 31 - top - (high_r - low_r))))
    right = int(rng.integers(1, min(4, 31 - left - (high_c - low_c))))
    shape = high_r - low_r + top + bottom, high_c - low_c + left + right
    inp = np.full(shape, bg, dtype=np.int8)
    out = inp.copy()
    for i, (a, b) in enumerate(objects):
        r, c = locations[i]
        r, c = r - low_r + top, c - low_c + left
        h, w = a.shape
        inp[r:r + h, c:c + w] = a
        out[r:r + h, c:c + w] = b
    return {'input': inp, 'output': out}


def _geometry(record):
    return record['cavity_mask'].shape, record['cavity_mask'].tobytes()


def _make_pair(rng, stage, forced_source, old_target, forbidden):
    for _ in range(100):
        palette = list(map(int, rng.permutation(10)))
        source = forced_source if forced_source is not None else palette[0]
        rest = [c for c in palette if c != source]
        bg, neutral = rest[:2]
        inks = rest[2:]
        if stage == 0:
            counts, nplates = [1, 2], 2
        elif stage == 1:
            counts, nplates = [1, 2], 2
        elif stage == 'test':
            counts = sorted(map(int, rng.choice(np.arange(1, 5), int(rng.integers(2, 4)), replace=False)))
            nplates = _choice(rng, (1, 2, 3, 3))
        else:
            counts, nplates = sorted(map(int, rng.choice(np.arange(1, 5), 2, replace=False))), 2
            if rng.random() < .22:
                nplates, counts = 1, [1, 2]
        stencils = []
        for i, count in enumerate(counts):
            choices = [c for c in inks if not (stage == 1 and count == 2 and c == old_target)]
            target = _choice(rng, choices) if stage != 0 else inks[i]
            # The second teaching pair reverses both foreground and box area
            # ranking: a 7x7 one-cavity mask has >=24 foreground cells, whereas
            # a 5x5 two-cavity mask has <=23. This fixes a measured 2-train gap.
            size = ((7, 7) if count == 1 else (5, 5)) if stage == 1 else None
            stencils.append(_stencil(rng, count, bg, neutral, source, target,
                                      witness=stage == 0, size=size))
        if stage == 'test' and any((s['mask'].shape, s['mask'].tobytes()) in forbidden for s in stencils):
            continue
        chosen = list(stencils)[:nplates]
        while len(chosen) < nplates:
            chosen.append(_choice(rng, stencils))
        objects = [(s['tile'], np.full_like(s['tile'], bg)) for s in stencils]
        for i, stencil in enumerate(chosen):
            kind = 'rectangle' if stage == 0 else _choice(rng, ('rectangle', 'bevel', 'ell', 'cross', 'step'))
            if stage == 1 and i == 1:
                kind = 'ell'
            plate = _plate(rng, stencil, bg, source, kind, complete=stage == 0,
                           clipping=stage == 1, clustered=stage == 0 or rng.random() < .4)
            if plate is None:
                break
            objects.append(plate)
        if len(objects) != len(stencils) + nplates:
            continue
        pair = _pack(rng, objects, bg)
        if pair is None:
            continue
        try:
            decoded = _decode(pair['input'])
        except AssertionError:
            continue
        _require(np.array_equal(_render(decoded), pair['output']), 'construction disagrees with input decoding')
        if stage == 0:
            if any(np.array_equal(_render(decoded, v), pair['output']) for v in
                   ('ignore_count', 'area_rank', 'center', 'top_left', 'one_stamp',
                    'opaque', 'ignore_registration_hole')):
                continue
        if stage == 1 and np.array_equal(_render(decoded, 'canvas_clip'), pair['output']):
            continue
        if stage == 1 and any(np.array_equal(_render(decoded, v), pair['output'])
                              for v in ('foreground_rank', 'box_rank')):
            continue
        return pair
    raise RuntimeError('bounded scene construction exhausted')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= value <= high:
            raise ValueError(name + ' is outside the legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    source = int(rng.integers(10))
    puzzle, used, forbidden = {'train': [], 'test': []}, set(), set()
    old_target = None
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            stage = i if split == 'train' else 'test'
            for _ in range(8):
                pair = _make_pair(rng, stage, source if split == 'train' and i < 2 else None,
                                  old_target, forbidden)
                key = _key(pair['input'])
                if key not in used:
                    break
            else:
                raise RuntimeError('bounded uniqueness retries exhausted')
            used.add(key)
            puzzle[split].append(pair)
            decoded = _decode(pair['input'])
            if split == 'train':
                forbidden.update(_geometry(r) for r in decoded['records'])
                if i == 0:
                    old_target = next(r['target'] for r in decoded['records'] if r['count'] == 2)
    # Teaching pairs are evidence, not a positional convention.
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    try:
        _counts(len(puzzle['train']), len(puzzle['test']))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, decoded = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'printing preserves canvas dimensions')
            key = _key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            d = _decode(inp)
            _require(np.array_equal(_render(d), out), 'incorrect registered plate printing')
            _require(not np.array_equal(inp, out), 'identity pair')
            # No output ink may escape a plate, and every original hole is filled.
            support = set().union(*(p['support'] for p in d['plates']))
            _require(all(tuple(p) in support for p in np.argwhere(out != d['bg'])), 'ink outside plates')
            _require(all(int(out[p]) != d['bg'] for p in support), 'unfilled plate pinhole')
            decoded[split].append(d)
    train = decoded['train']
    # These checks reject teaching sets that only happen to fit the intended rule.
    for variant in ('ignore_count', 'area_rank', 'foreground_rank', 'box_rank', 'center', 'top_left', 'one_stamp',
                    'canvas_clip', 'opaque', 'ignore_registration_hole'):
        _require(any(not np.array_equal(_render(d, variant), _render(d)) for d in train),
                 'training does not distinguish ' + variant)
    _require(any(any(len(_components(r['cavity_mask'], True)) < r['count'] for r in d['records'])
                 for d in train), 'missing four/eight-connectivity witness')
    _require(any(any(len(c) > 1 for r in d['records'] for c in r['cavities']) for d in train),
             'missing multi-pixel cavity')
    _require(any(r['gap'] for d in train for r in d['records']) and
             any(not r['gap'] for d in train for r in d['records']), 'missing registration kinds')
    # Local recoloring must actually change for an already seen carrier/count.
    mappings = {}
    for d in train:
        for r in d['records']:
            mappings.setdefault((r['source'], r['count']), set()).add(r['target'])
    _require(any(len(v) > 1 for v in mappings.values()), 'training permits a memorized ink dictionary')
    train_masks = {_geometry(r) for d in train for r in d['records']}
    _require(all(all(_geometry(r) not in train_masks for r in d['records']) for d in decoded['test']),
             'test must require reading a fresh visible stencil')
    return None


def self_test():
    """42 puzzles, streamed metrics and focused failures; no disk artifacts."""
    shapes, geometries, counts, directions, backgrounds, outlines, targets = (set() for _ in range(7))
    signatures = set()
    test_plate_counts = set()
    clipping, overlap, pairs = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                _require(np.array_equal(solve(inp.copy()), out), 'input-only solve failed')
                digest.update(inp.tobytes())
                pairs += 1
                d = _decode(inp)
                if split == 'test':
                    test_plate_counts.add(len(d['plates']))
                shapes.add(inp.shape)
                backgrounds.add(d['bg'])
                for r in d['records']:
                    geometries.add(_geometry(r))
                    counts.add(r['count'])
                    directions.add(r['direction'])
                    targets.add(r['target'])
                for p in d['plates']:
                    r0, r1, c0, c1 = p['box']
                    support = np.zeros((r1 - r0, c1 - c0), dtype=np.int8)
                    for r, c in p['support']:
                        support[r - r0, c - c0] = 1
                    outlines.add(_key(support))
                    visits = []
                    for ar, ac in p['anchors']:
                        raw = {(ar + dr, ac + dc) for dr, dc in p['record']['offsets']}
                        clipping += bool(raw - p['support'])
                        visits.append(raw & p['support'])
                    overlap += sum(len(a & b) > 0 for i, a in enumerate(visits) for b in visits[i + 1:])
        signatures.add(digest.hexdigest())
    _require(len(signatures) == 42, 'seed diversity')
    _require(len(shapes) >= 20 and len(geometries) >= 70 and len(outlines) >= 35, 'structural diversity')
    _require(counts == {1, 2, 3, 4} and directions == set(DIRS), 'counts or orientations missing')
    _require(len(backgrounds) == len(targets) == 10 and clipping > 42 and overlap > 42, 'role/print coverage')
    _require(test_plate_counts == {1, 2, 3}, 'test multiplicity variation missing')
    # Minimum/maximum counts, all legal combinations, invalid type/range cases.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(918001190, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(731, 2, 1), generate_puzzle(731, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s])
                 for k in ('input', 'output')), 'nondeterminism')
    b['test'][0]['output'][0, 0] = (int(b['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(b)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': pairs, 'distinct_puzzles': len(signatures),
            'input_shapes': len(shapes), 'stencil_geometries': len(geometries),
            'plate_outlines': len(outlines), 'clipped_stamps': clipping,
            'overlapping_stamp_pairs': overlap, 'counts': sorted(counts),
            'directions': sorted(directions), 'backgrounds': len(backgrounds),
            'ink_colors': len(targets), 'test_plate_counts': sorted(test_plate_counts)}
