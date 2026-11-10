"""Registered blind spots: recognize a window, then stamp its unseen remainder.

No files, parent modules, training examples, or generation state are used by solve.
All sampling uses the one explicitly seeded NumPy generator. Diagnostics are
streamed in memory; this module never saves puzzles or execution traces.
"""
from itertools import combinations
import hashlib
import json

import numpy as np


DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return (a.shape, a.tobytes())


def _d4(a, k):
    """Four rotations, followed by four rotations of a left/right reflection."""
    return np.rot90(np.fliplr(a) if k >= 4 else a, k % 4).copy()


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(a):
    a = np.asarray(a)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(cells)
    return result


def _rectangles(grid, colors):
    rectangles = []
    for cells in _components(np.isin(grid, colors)):
        rows, cols = zip(*cells)
        r, c = int(min(rows)), int(min(cols))
        h, w = int(max(rows) - r + 1), int(max(cols) - c + 1)
        if min(h, w) < 2 or len(cells) != h * w:
            return None
        patch = grid[r:r + h, c:c + w]
        if len(np.unique(patch)) != 2:
            return None
        rectangles.append((r, c, h, w))
    return rectangles


def _label_options(grid, rect, excluded):
    r, c, h, w = rect
    result = []
    for dr, dc in DIRS:
        edge = ([(i, 0 if dc == 1 else w - 1) for i in range(h)] if dc else
                [(0 if dr == 1 else h - 1, j) for j in range(w)])
        for qr, qc in edge:
            sl = (r + qr - 2 * dr, c + qc - 2 * dc)
            tl = (r + qr - dr, c + qc - dc)
            if not (0 <= sl[0] < grid.shape[0] and 0 <= sl[1] < grid.shape[1]):
                continue
            source, target = int(grid[sl]), int(grid[tl])
            if source in excluded or target in excluded or source == target:
                continue
            result.append((source, target, (qr, qc), (dr, dc), sl, tl))
    return result


def _matches(reference, patch):
    h, w = patch.shape
    matches = []
    for k in range(8):
        full = _d4(reference, k)
        if h > full.shape[0] or w > full.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
        for swap in (0, 1):
            rr, cc = np.nonzero(np.all(windows == (1 - patch if swap else patch), axis=(-2, -1)))
            matches.extend((k, swap, int(r), int(c)) for r, c in zip(rr, cc))
    return matches


def _registered(reference, patch, q):
    matches = _matches(reference, patch)
    _require(len(matches) == 1, "window must uniquely identify orientation, role exchange, and location")
    k, swap, r, c = matches[0]
    full = _d4(reference, k)
    h, w = patch.shape
    reg = (r + q[0], c + q[1])
    outside = full.copy()
    outside[r:r + h, c:c + w] = 0
    _require(bool(outside.any()) and bool(full[r:r + h, c:c + w].any()), "trivial window")
    offsets = tuple((int(a - reg[0]), int(b - reg[1])) for a, b in np.argwhere(outside))
    return dict(full=full, orientation=k, swap=swap, origin=(r, c),
                window_shape=(h, w), registration=reg, offsets=offsets,
                occupied=bool(full[reg]), fraction=h * w / full.size)


def _decode_palette(grid, bg, colors):
    rects = _rectangles(grid, colors)
    _require(rects is not None and 2 <= len(rects) <= 4, "no swatch collection")
    labeled, masters = [], []
    for rect in rects:
        options = _label_options(grid, rect, {bg, *colors})
        _require(len(options) <= 1, "ambiguous labels")
        (labeled if options else masters).append((rect, options))
    _require(len(masters) == 1 and labeled, "need one unlabeled master")
    master_rect = masters[0][0]
    r, c, h, w = master_rect
    _require(min(h, w) >= 5, "master too small")
    master = grid[r:r + h, c:c + w]
    counts = [int(np.sum(master == color)) for color in colors]
    _require(counts[0] != counts[1], "master color roles tied")
    ink = colors[int(counts[1] < counts[0])]
    paper = next(v for v in colors if v != ink)
    reference = (master == ink).astype(np.int8)
    _require(len({_key(_d4(reference, k)) for k in range(8)}) == 8, "symmetric master")
    records, labels = [], set()
    for rect, options in labeled:
        r, c, hh, ww = rect
        _require(hh * ww < h * w, "window must be proper")
        source, target, q, direction, sl, tl = options[0]
        _require(sl not in labels and tl not in labels, "shared labels")
        labels.update((sl, tl))
        patch = (grid[r:r + hh, c:c + ww] == ink).astype(np.int8)
        record = _registered(reference, patch, q)
        record.update(rect=rect, source=source, target=target, q=q,
                      direction=direction, labels=(sl, tl))
        records.append(record)
    _require(len({v['source'] for v in records}) == len(records), "duplicate source")
    swatches = {(a, b) for r, c, hh, ww in rects for a in range(r, r + hh) for b in range(c, c + ww)}
    _require(not labels & swatches, "labels overlap swatches")
    for rec in records:
        rec['anchors'] = tuple((int(a), int(b)) for a, b in np.argwhere(grid == rec['source'])
                               if (a, b) not in labels and (a, b) not in swatches)
        _require(bool(rec['anchors']), "instruction without anchors")
    return dict(background=bg, ink=ink, paper=paper, reference=reference,
                master_rect=master_rect, rectangles=rects, swatches=swatches,
                labels=labels, records=records)


def _decode(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background is tied")
    bg = int(counts.argmax())
    # Filled two-color rectangles identify the swatch palette without a legend.
    colors = [int(v) for v in np.unique(grid) if v != bg]
    answers = []
    for palette in combinations(colors, 2):
        try:
            answers.append(_decode_palette(grid, bg, palette))
        except ValueError:
            continue
    _require(len(answers) == 1, "swatch palette is missing or ambiguous")
    return answers[0]


def _offsets(rec, variant):
    if variant == 'correct':
        return rec['offsets']
    full = rec['full'].copy()
    r, c = rec['origin']
    h, w = rec['window_shape']
    if variant == 'fragment':
        keep = np.zeros_like(full)
        keep[r:r + h, c:c + w] = full[r:r + h, c:c + w]
        full = keep
    elif variant == 'center':
        full[r:r + h, c:c + w] = 0
    elif variant != 'whole':
        raise ValueError('unknown counterfactual')
    qr, qc = ((full.shape[0] // 2, full.shape[1] // 2) if variant == 'center'
              else rec['registration'])
    return tuple((int(a - qr), int(b - qc)) for a, b in np.argwhere(full))


def _render(grid, data, variant='correct'):
    out = grid.copy()
    bg = data['background']
    anchors = {p for rec in data['records'] for p in rec['anchors']}
    for p in data['swatches'] | data['labels'] | anchors:
        out[p] = bg
    paint = {}
    for rec in data['records']:
        for ar, ac in rec['anchors']:
            for dr, dc in _offsets(rec, variant):
                p = (ar + dr, ac + dc)
                if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]:
                    _require(p not in paint or paint[p] == rec['target'], "conflicting paint")
                    paint[p] = rec['target']
    for p, color in paint.items():
        out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    return _render(grid, _decode(grid))


def _bank(reference):
    """At most 12,544 small window keys for an 8x8 reference, held in RAM."""
    counts = {}
    for k in range(8):
        full = _d4(reference, k)
        H, W = full.shape
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                if not .18 <= h * w / full.size <= .8:
                    continue
                for r in range(H - h + 1):
                    for c in range(W - w + 1):
                        patch = full[r:r + h, c:c + w]
                        if not (patch.any() and (1 - patch).any()):
                            continue
                        for swap in (0, 1):
                            key = _key(1 - patch if swap else patch)
                            counts[key] = counts.get(key, 0) + 1
    H, W = reference.shape
    result = []
    for h in range(2, H + 1):
        for w in range(2, W + 1):
            if not .18 <= h * w / reference.size <= .8:
                continue
            for r in range(H - h + 1):
                for c in range(W - w + 1):
                    patch = reference[r:r + h, c:c + w]
                    if counts.get(_key(patch)) == 1 and 0 < patch.sum() < reference.sum():
                        result.append((r, c, h, w))
    return result


def _pattern(rng, H, W):
    a = np.zeros((H, W), dtype=np.int8)
    # Combine independently placed strokes, loops, teeth and staircase segments.
    for _ in range(int(rng.integers(2, 5))):
        kind = int(rng.integers(4))
        r0, r1 = sorted(int(v) for v in rng.choice(H, 2, replace=False))
        c0, c1 = sorted(int(v) for v in rng.choice(W, 2, replace=False))
        if kind == 0:
            a[r0:r1 + 1, c0] = 1
            a[r1, c0:c1 + 1] = 1
            if rng.random() < .6:
                a[r0, c0:c1 + 1] = 1
                a[r0:r1 + 1, c1] = 1
        elif kind == 1:
            a[r0, c0:c1 + 1] = 1
            for c in range(c0, c1 + 1, 2):
                a[r0:int(rng.integers(r0 + 1, r1 + 2)), c] = 1
        elif kind == 2:
            n = max(r1 - r0, c1 - c0) + 1
            rr = np.rint(np.linspace(r0, r1, n)).astype(int)
            cc = np.rint(np.linspace(c0, c1, n)).astype(int)
            a[rr, cc] = 1
            a[r1, c0:c1 + 1] = 1
        else:
            a[r0:r1 + 1, c0] = 1
            m = int(rng.integers(r0, r1 + 1))
            a[m, c0:c1 + 1] = 1
            a[m:r1 + 1, c1] = 1
    return a


def _reference(rng):
    H = int(rng.integers(5, 9))
    W = H if rng.random() < .35 else int(rng.integers(5, 9))
    for _ in range(48):
        a = _pattern(rng, H, W)
        if not .20 <= a.mean() < .48:
            continue
        if len({_key(_d4(a, k)) for k in range(8)}) < 8:
            continue
        bank = _bank(a)
        if (sum(.45 <= h * w / a.size <= .8 for r, c, h, w in bank) >= 8 and
                sum(.18 <= h * w / a.size <= .4 for r, c, h, w in bank) >= 8 and
                _teaching_capable(a, bank)):
            return a, bank
    return _fallback_reference()


def _teaching_capable(a, bank):
    """A reference must admit spatial remainders and both boundary roles."""
    for r, c, h, w in bank:
        if not .45 <= h * w / a.size <= .8:
            continue
        outside = a.copy()
        outside[r:r + h, c:c + w] = 0
        rr, cc = np.nonzero(outside)
        patch = a[r:r + h, c:c + w]
        boundary = {(u, v) for u in range(h) for v in range(w)
                    if u in (0, h - 1) or v in (0, w - 1)}
        role_counts = [sum(int(patch[p]) == role for p in boundary) for role in (0, 1)]
        # With two positions for each role, excluding the full-image center
        # cannot eliminate all registrations, in any transformed orientation.
        if len(rr) >= 5 and len(set(rr)) >= 2 and len(set(cc)) >= 2 and min(role_counts) >= 2:
            return True
    return False


def _fallback_reference():
    # Bounded, structured fallback; still matched exhaustively, never seed lookup.
    a = np.array([[int(c) for c in row] for row in
                  ('0100000', '0111100', '0100100', '0000110', '1110000', '0011000')], dtype=np.int8)
    return a, _bank(a)


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _command(rng, reference, bank, training, orientation=None, swap=None, wanted=None):
    candidates = [b for b in bank if (.45 <= b[2] * b[3] / reference.size <= .8 if training
                                     else .18 <= b[2] * b[3] / reference.size <= .4)]
    for index in rng.permutation(len(candidates)):
        r, c, h, w = candidates[int(index)]
        k = int(rng.integers(8)) if orientation is None else orientation
        s = int(rng.integers(2)) if swap is None else swap
        full = _d4(reference, k)
        support = np.zeros_like(reference)
        support[r:r + h, c:c + w] = 1
        rr, cc = np.nonzero(_d4(support, k))
        r, c, h, w = int(rr.min()), int(cc.min()), int(rr.max() - rr.min() + 1), int(cc.max() - cc.min() + 1)
        binary = full[r:r + h, c:c + w]
        choices = []
        for direction in DIRS:
            dr, dc = direction
            edge = ([(i, 0 if dc == 1 else w - 1) for i in range(h)] if dc else
                    [(0 if dr == 1 else h - 1, j) for j in range(w)])
            for q in edge:
                occupied = bool(binary[q])
                if wanted == 'hole' and occupied or wanted == 'ink' and not occupied:
                    continue
                if (r + q[0], c + q[1]) == (full.shape[0] // 2, full.shape[1] // 2):
                    continue
                choices.append((q, direction))
        if choices:
            q, direction = _pick(rng, choices)
            patch = 1 - binary if s else binary
            rec = _registered(reference, patch, q)
            if wanted is not None:
                # The first two examples must show a spatial shape, rather
                # than leaving only a tiny line after window subtraction.
                rows, cols = zip(*rec['offsets'])
                if len(rows) < 5 or len(set(rows)) < 2 or len(set(cols)) < 2:
                    continue
            rec.update(patch=patch.copy(), q=q, direction=direction)
            return rec
    raise ValueError('no suitable registered crop')


def _halo(cells, distance, shape):
    return {(r + dr, c + dc) for r, c in cells
            for dr in range(-distance, distance + 1) for dc in range(-distance, distance + 1)
            if 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]}


def _place_cards(rng, reference, commands, palette, shape):
    bg, paper, ink = palette[:3]
    grid = np.full(shape, bg, dtype=np.int8)
    occupied, reserved = set(), set()
    # Master and commands have no fixed row, side, or order.
    items = [(reference, None)] + [(rec['patch'], rec) for rec in commands]
    order = [int(v) for v in rng.permutation(len(items))]
    for i in order:
        patch, rec = items[i]
        h, w = patch.shape
        candidates = [(r, c) for r in range(shape[0] - h + 1) for c in range(shape[1] - w + 1)]
        for index in rng.permutation(len(candidates)):
            r, c = candidates[int(index)]
            cells = {(a, b) for a in range(r, r + h) for b in range(c, c + w)}
            labels = []
            if rec is not None:
                qr, qc = rec['q']
                dr, dc = rec['direction']
                labels = [(r + qr - t * dr, c + qc - t * dc) for t in (2, 1)]
                if any(not (0 <= a < shape[0] and 0 <= b < shape[1]) for a, b in labels):
                    continue
            footprint = cells | set(labels)
            if footprint & reserved:
                continue
            grid[r:r + h, c:c + w] = np.where(patch, ink, paper)
            if rec is not None:
                rec['rect'], rec['labels'] = (r, c, h, w), tuple(labels)
                grid[labels[0]], grid[labels[1]] = rec['source'], rec['target']
            occupied |= footprint
            reserved |= _halo(footprint, 1, shape)
            break
        else:
            raise ValueError('cards do not fit')
    return grid, occupied


def _footprint(rec, anchor, shape):
    return {(anchor[0] + dr, anchor[1] + dc) for dr, dc in rec['offsets']
            if 0 <= anchor[0] + dr < shape[0] and 0 <= anchor[1] + dc < shape[1]}


def _make_pair(rng, reference, bank, palette, training, index):
    bg, paper, ink = palette[:3]
    pool = palette[3:]
    teaching = training and index < 2
    count = (1 if index == 0 else 2) if teaching else int(_pick(rng, (1, 1, 2, 2, 3)))
    commands = []
    sources = list(rng.permutation(pool)[:count])
    targets = [_pick(rng, [v for v in pool if v != s]) for s in sources]
    if teaching:
        sources[0], targets[0] = pool[0], pool[1 if index == 0 else 2]
        if index == 1:
            sources[1], targets[1] = pool[2], pool[0]
    for j, (source, target) in enumerate(zip(sources, targets)):
        k = (0 if index == 0 else 1 if j == 0 else 4) if teaching else None
        s = (1 if index == 1 and j == 0 else 0) if teaching else None
        wanted = ('ink' if index == 0 else 'hole' if j == 0 else 'ink') if teaching else None
        rec = _command(rng, reference, bank, training, k, s, wanted)
        rec.update(source=int(source), target=int(target))
        commands.append(rec)
    failures = {}
    for attempt in range(28):
        shape = (30, 30) if attempt >= 12 else (int(rng.integers(21, 31)), int(rng.integers(21, 31)))
        try:
            grid, occupied = _place_cards(rng, reference, commands, palette, shape)
            guard = _halo(occupied, 2, shape)
            paint, anchors = {}, set()
            for rec in commands:
                # Every teaching instruction has a complete stamp plus an
                # actually clipped stamp. Other examples vary these independently.
                policy = 'mixed' if teaching else _pick(rng, ('interior', 'edge', 'mixed', 'free'))
                desired = max(2, int(rng.integers(1, 5))) if teaching else int(rng.integers(1, 5))
                positions = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
                positions = [positions[int(i)] for i in rng.permutation(len(positions))]
                selected = []
                for j in range(desired):
                    need_full = j == 0 or policy == 'interior'
                    need_clip = j > 0 and (teaching and j == 1 or policy == 'edge')
                    found = None
                    # Adjacent original anchors are legal; compatible paint may
                    # cover them, since collection precedes erasure and painting.
                    nearby = [(a + dr, b + dc) for a, b in selected for dr, dc in DIRS]
                    for p in (nearby if j and rng.random() < .5 else []) + positions:
                        if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]):
                            continue
                        if p in guard or p in anchors or grid[p] != bg:
                            continue
                        footprint = _footprint(rec, p, shape)
                        if not footprint or footprint & occupied:
                            continue
                        complete = len(footprint) == len(rec['offsets'])
                        if need_full and not complete or need_clip and complete:
                            continue
                        if j == 0:
                            qr, qc = rec['registration']
                            hh, ww = rec['full'].shape
                            if not (0 <= p[0] - qr <= shape[0] - hh and
                                    0 <= p[1] - qc <= shape[1] - ww):
                                continue
                        if any(q in paint and paint[q] != rec['target'] for q in footprint):
                            continue
                        found = p
                        for q in footprint:
                            paint[q] = rec['target']
                        break
                    if found is None:
                        if teaching or not selected:
                            raise ValueError('anchors do not fit')
                        break
                    grid[found] = rec['source']
                    anchors.add(found)
                    selected.append(found)
                rec['anchors'] = tuple(selected)
            # Inactive colors, including target-only colors, can be inert.
            inactive = [v for v in pool if v not in {rec['source'] for rec in commands}]
            for _ in range(int(rng.integers(5))):
                candidates = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                              if (r, c) not in guard and (r, c) not in paint and grid[r, c] == bg]
                if candidates:
                    grid[_pick(rng, candidates)] = int(_pick(rng, inactive))
            # Reconstruct every role and location from the finished input.
            data = _decode(grid)
            if data['ink'] != ink or data['paper'] != paper:
                continue
            output = _render(grid, data)
            if any(_same_variant(grid, data, output, v) for v in ('fragment', 'center', 'whole')):
                continue
            return {'input': grid, 'output': output}
        except ValueError as error:
            reason = str(error)
            failures[reason] = failures.get(reason, 0) + 1
            continue
    raise ValueError('bounded card/anchor placement exhausted: ' + str(failures))


def _same_variant(grid, data, output, variant):
    try:
        return np.array_equal(_render(grid, data, variant), output)
    except ValueError:
        return False


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(name + ' outside allowed range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = [int(v) for v in rng.permutation(10)]
    reference, bank = _reference(rng)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            last_error = None
            for _ in range(8):
                try:
                    pair = _make_pair(rng, reference, bank, palette, split == 'train', index)
                except ValueError as error:
                    last_error = error
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError('bounded pair retries exhausted: ' + str(last_error))
    return puzzle


def _canonical(reference):
    return min(_key(_d4(reference, k)) for k in range(8))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'invalid splits')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'invalid pair counts')
    seen, references, roles, teaching = set(), set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            grid, output = _grid(pair['input']), _grid(pair['output'])
            _require(grid.shape == output.shape, 'changed canvas dimensions')
            _require(_key(grid) not in seen, 'duplicate input')
            seen.add(_key(grid))
            data = _decode(grid)
            _require(np.array_equal(_render(grid, data), output), 'incorrect output')
            _require(not np.array_equal(grid, output), 'identity pair')
            references.add(_canonical(data['reference']))
            roles.add((data['background'], data['paper'], data['ink']))
            anchors = {p for rec in data['records'] for p in rec['anchors']}
            paints = set()
            for rec in data['records']:
                _require(.45 <= rec['fraction'] <= .8 if split == 'train' else .18 <= rec['fraction'] <= .4,
                         'wrong crop scale for split')
                for p in rec['anchors']:
                    paints |= _footprint(rec, p, grid.shape)
                qr, qc = rec['registration']
                hh, ww = rec['full'].shape
                _require(any(0 <= a - qr <= grid.shape[0] - hh and
                             0 <= b - qc <= grid.shape[1] - ww for a, b in rec['anchors']),
                         'instruction needs one entire in-bounds reference box')
            # All administrative swatches and labels disappear unless stamped;
            # unrelated input cells survive exactly when outside the new paint.
            erased = data['swatches'] | data['labels'] | anchors
            _require(all(output[p] == data['background'] for p in erased - paints), 'incomplete erasure')
            _require(all(output[r, c] == grid[r, c] for r in range(grid.shape[0]) for c in range(grid.shape[1])
                         if (r, c) not in erased | paints), 'inert content changed')
            for variant in ('fragment', 'center', 'whole'):
                _require(not _same_variant(grid, data, output, variant), 'shortcut gives same answer: ' + variant)
            if split == 'train':
                teaching.append((grid, data))
    _require(len(references) == 1 and len(roles) == 1, 'puzzle-wide reference/roles changed')
    # Teaching obligations are reconstructed from grids, not latent metadata.
    first, second = teaching[:2]
    _require(len(first[1]['records']) == 1 and len(second[1]['records']) == 2, 'missing teaching counts')
    r0 = first[1]['records'][0]
    _require(r0['orientation'] == 0 and r0['occupied'], 'missing direct occupied registration')
    _require({r['orientation'] for r in second[1]['records']} == {1, 4}, 'missing geometric evidence')
    _require(any(r['swap'] and not r['occupied'] for r in second[1]['records']), 'missing exchanged hole registration')
    local0 = {r['source']: r['target'] for r in first[1]['records']}
    local1 = {r['source']: r['target'] for r in second[1]['records']}
    _require(any(s in local1 and local1[s] != t for s, t in local0.items()), 'mapping did not change')
    for grid, data in teaching[:2]:
        for rec in data['records']:
            rows, cols = zip(*rec['offsets'])
            _require(len(rows) >= 5 and len(set(rows)) >= 2 and len(set(cols)) >= 2,
                     'teaching remainder is too small or one-dimensional')
            sizes = [len(_footprint(rec, p, grid.shape)) for p in rec['anchors']]
            _require(len(rec['offsets']) in sizes and min(sizes) < len(rec['offsets']), 'need complete and clipped teaching stamps')


def self_test():
    """42 streaming seeds, bounded fingerprints/sets, no retained grid corpus."""
    signatures, references, shapes, windows, geometries = set(), set(), set(), set(), set()
    directions, orientations, swaps, occupied, bg_colors = set(), set(), set(), set(), set()
    anchor_counts, instruction_counts = set(), set()
    overlaps = covered = inert = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        serial = json.dumps({s: [{k: v.tolist() for k, v in p.items()} for p in puzzle[s]]
                             for s in ('train', 'test')}, separators=(',', ':'))
        signatures.add(hashlib.sha256(serial.encode()).hexdigest())
        if seed == 0:
            replay = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], replay[s]) for k in p)
            # Metamorphic checks use the visible grids and their transformed
            # answers; they do not introduce another solving implementation.
            color_map = np.random.default_rng(729).permutation(10)
            for split in puzzle:
                for pair in puzzle[split]:
                    assert np.array_equal(solve(color_map[pair['input']]), color_map[pair['output']])
                    for k in range(8):
                        assert np.array_equal(solve(_d4(pair['input'], k)), _d4(pair['output'], k))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid = pair['input']
                data = _decode(grid)
                references.add(_canonical(data['reference']))
                shapes.add(grid.shape)
                bg_colors.add(data['background'])
                instruction_counts.add(len(data['records']))
                anchors = {p for rec in data['records'] for p in rec['anchors']}
                visits = {}
                for rec in data['records']:
                    directions.add(rec['direction'])
                    orientations.add(rec['orientation'])
                    swaps.add(rec['swap'])
                    occupied.add(rec['occupied'])
                    windows.add(rec['window_shape'])
                    geometries.add(rec['offsets'])
                    anchor_counts.add(len(rec['anchors']))
                    for p in rec['anchors']:
                        fp = _footprint(rec, p, grid.shape)
                        covered += len((fp & anchors) - {p})
                        for q in fp:
                            visits[q] = visits.get(q, 0) + 1
                overlaps += sum(n > 1 for n in visits.values())
                erased = data['swatches'] | data['labels'] | anchors
                inert += sum(grid[r, c] != data['background'] and (r, c) not in erased and (r, c) not in visits
                             for r in range(grid.shape[0]) for c in range(grid.shape[1]))
        if seed == 0:
            broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
            try:
                validate_generated_puzzle(broken)
            except ValueError:
                pass
            else:
                raise AssertionError('corrupt output accepted')
    assert len(signatures) == 42 and len(references) >= 35
    assert len(shapes) >= 35 and len(windows) >= 20 and len(geometries) >= 100
    assert directions == set(DIRS) and orientations == set(range(8))
    assert swaps == {0, 1} and occupied == {False, True}
    assert len(bg_colors) >= 8 and instruction_counts == {1, 2, 3}
    assert {1, 2, 3, 4} <= anchor_counts and overlaps > 20 and covered > 10 and inert > 30
    fallback, fallback_bank = _fallback_reference()
    assert _teaching_capable(fallback, fallback_bank)
    rng = np.random.default_rng(9876)
    palette = list(range(10))
    regression = {'train': [_make_pair(rng, fallback, fallback_bank, palette, True, i) for i in range(2)],
                  'test': [_make_pair(rng, fallback, fallback_bank, palette, False, 0)]}
    validate_generated_puzzle(regression)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(300 + 10 * nt + nq, nt, nq)
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
