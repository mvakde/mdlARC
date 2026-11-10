"""Collision-cohort fusion: geometric grouping, unary addition, stable packing.

Two opposite colored rails visibly identify the frame-to-edge instructions.
No puzzle parameters, examples, seeds, or parent generators are used by solve.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0, 745703340, 307442988, 523031296, 350898330, 758044832,
              775271369, 184996775, 136329287, 947742973, 623673341,
              849416460, 343229055, 494160823, 949188468, 108713114,
              518913500, 247815700, 540020411, 900955448, 827815172,
              725019195, 145121852, 53314400, 346988983, 426595274,
              67709256, 853861289, 84685223, 428858210, 75524001,
              129133895, 507499887, 969463052, 188508203, 447767949,
              3501736, 73270521, 88245164, 623980377, 753756296, 315202028)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid colors must be integer values 0..9")
    return a.astype(np.int8, copy=True)


def _canonical(grid):
    a = _grid(grid)
    _require(min(a.shape) >= 5, "a rail scene needs at least five cells per side")
    horizontal = (np.all(a[:, 0] == a[0, 0]) and
                  np.all(a[:, -1] == a[0, -1]) and a[0, 0] != a[0, -1])
    vertical = (np.all(a[0] == a[0, 0]) and
                np.all(a[-1] == a[-1, 0]) and a[0, 0] != a[-1, 0])
    _require(horizontal != vertical, "exactly one opposite pair of rails is required")
    return (a if horizontal else a.T.copy()), not horizontal


def _decode_canonical(a):
    """Read full rectangular frames and repeated, one-row interior tokens."""
    height, width = a.shape
    rails = (int(a[0, 0]), int(a[0, -1]))
    bg = int(a[0, 1])
    _require(bg not in rails and np.all(a[0, 1:-1] == bg),
             "the clear edge between rails must reveal background")
    _require(np.all(a[:, 0] == rails[0]) and np.all(a[:, -1] == rails[1]),
             "rails must be complete")
    covered = a == bg
    covered[:, (0, -1)] = True
    objects = []
    for r in range(1, height):
        for c in range(1, width - 1):
            if covered[r, c]:
                continue
            frame = int(a[r, c])
            _require(frame in rails and r + 2 < height and c + 2 < width - 1,
                     "unrecognized or incomplete tray")
            bottom = r + 1
            while bottom < height and a[bottom, c + 1] != frame:
                bottom += 1
            right = c + 1
            while right < width - 1 and a[r + 1, right] != frame:
                right += 1
            _require(bottom < height and right < width - 1,
                     "tray frame is open")
            h, w = bottom - r + 1, right - c + 1
            patch = a[r:bottom + 1, c:right + 1]
            _require(h >= 3 and h % 2 == 1 and 3 <= w <= 7,
                     "tray must have an odd height and a nonempty interior")
            _require(np.all(patch[0] == frame) and np.all(patch[-1] == frame)
                     and np.all(patch[:, 0] == frame) and np.all(patch[:, -1] == frame),
                     "tray must have a complete one-cell frame")
            interior = patch[1:-1, 1:-1]
            token = interior[0].copy()
            _require(np.any(token != bg) and not np.any(interior == frame),
                     "tokens cannot erase their own frame")
            _require(len(np.unique(token[token != bg])) == 1,
                     "one tray repeats a monochrome token")
            _require(all(np.array_equal(row, token) for row in interior[::2])
                     and np.all(interior[1::2] == bg),
                     "tokens must repeat at pitch two with clear gaps")
            _require(not np.any(covered[r:bottom + 1, c:right + 1] &
                                 (a[r:bottom + 1, c:right + 1] != bg)),
                     "tray frames overlap")
            covered[r:bottom + 1, c:right + 1] = True
            objects.append(dict(r=r, c=c, h=h, w=w, n=(h - 1) // 2,
                                frame=frame, token=token, bottom=bottom))
    _require(objects and np.all(covered), "scene must contain only rails and trays")
    return bg, rails, objects


def _band_overlap(a, b):
    return max(a['r'], b['r']) < min(a['r'] + a['h'], b['r'] + b['h'])


def _overlap(a, b):
    return (_band_overlap(a, b) and
            max(a['c'], b['c']) < min(a['c'] + a['w'], b['c'] + b['w']))


def _order(o, left):
    return (o['c'] if left else -o['c'] - o['w'], o['r'], o['c'])


def _cohorts(objects):
    """Connected components of same-frame row-interval overlap, from INPUT."""
    unseen = set(range(len(objects)))
    result = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        component, queue = [root], [root]
        while queue:
            i = queue.pop()
            adjacent = [j for j in sorted(unseen)
                        if objects[i]['frame'] == objects[j]['frame']
                        and _band_overlap(objects[i], objects[j])]
            for j in adjacent:
                unseen.remove(j)
                component.append(j)
                queue.append(j)
        result.append(sorted(component))
    return result


def _fuse(objects, rails, *, add=True):
    grown, records = [], []
    for component in _cohorts(objects):
        left = objects[component[0]]['frame'] == rails[0]
        receiver = min(component, key=lambda i: _order(objects[i], left))
        source = objects[receiver]
        total = sum(objects[i]['n'] for i in component)
        o = dict(source)
        o['n'] = total if add else source['n']
        o['h'] = 2 * o['n'] + 1
        o['r'] = source['bottom'] - o['h'] + 1
        o['start_c'] = source['c']
        o['start_r'] = source['r']
        o['members'] = tuple(component)
        o['receiver'] = receiver
        grown.append(o)
        records.append(dict(component=component, receiver=receiver, total=total,
                            initial=source['n'],
                            sources=[objects[i]['n'] for i in component if i != receiver]))
    return grown, records


def _pack(grown, width, rails, *, edge_first=False):
    """Stable order uses receivers' original edge distances; grown spans block."""
    result = [dict(o) for o in grown]
    for frame in rails:
        left = frame == rails[0]
        indices = [i for i, o in enumerate(grown) if o['frame'] == frame]
        indices.sort(key=lambda i: (grown[i]['start_c'] if left else
                                   -grown[i]['start_c'] - grown[i]['w'],
                                   grown[i]['start_r'], grown[i]['start_c']))
        settled = []
        for i in indices:
            o = result[i]
            blockers = [p for p in settled if _band_overlap(o, p)]
            if edge_first:
                positions = range(1, width - o['w'])
                if not left:
                    positions = reversed(positions)
                for c in positions:
                    o['c'] = c
                    if not any(_overlap(o, p) for p in settled):
                        break
                else:
                    raise ValueError("packing capacity exceeded")
            elif left:
                o['c'] = max((p['c'] + p['w'] for p in blockers), default=1)
            else:
                o['c'] = min((p['c'] for p in blockers), default=width - 1) - o['w']
            _require(1 <= o['c'] and o['c'] + o['w'] <= width - 1,
                     "packing capacity exceeded")
            settled.append(o)
    _require(not any(_overlap(a, b) for i, a in enumerate(result) for b in result[i + 1:]),
             "opposite packed groups collide")
    return result


def _render(shape, bg, rails, objects):
    a = np.full(shape, bg, dtype=np.int8)
    a[:, 0], a[:, -1] = rails
    occupied = np.zeros(shape, bool)
    for o in objects:
        r, c, h, w = (o[k] for k in ('r', 'c', 'h', 'w'))
        _require(r >= 1 and r + h <= shape[0] and c >= 1 and c + w <= shape[1] - 1,
                 "a tray would be clipped or would cover a rail")
        _require(not np.any(occupied[r:r + h, c:c + w]), "tray footprints collide")
        occupied[r:r + h, c:c + w] = True
        patch = np.full((h, w), bg, dtype=np.int8)
        patch[0] = patch[-1] = o['frame']
        patch[:, 0] = patch[:, -1] = o['frame']
        patch[1:-1:2, 1:-1] = o['token']
        a[r:r + h, c:c + w] = patch
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, transposed = _canonical(input_grid)
    bg, rails, objects = _decode_canonical(a)
    grown, _ = _fuse(objects, rails)
    result = _render(a.shape, bg, rails, _pack(grown, a.shape[1], rails))
    return result.T.copy() if transposed else result


def _features(grid):
    a, axis = _canonical(grid)
    bg, rails, objects = _decode_canonical(a)
    grown, records = _fuse(objects, rails)
    settled = _pack(grown, a.shape[1], rails)
    old = _pack(grown, a.shape[1], rails, edge_first=True)
    indirect = any(len(rec['component']) >= 3 and any(
        not _band_overlap(objects[rec['receiver']], objects[j])
        for j in rec['component']) for rec in records)
    new_edges = sum(a['frame'] == b['frame'] and _band_overlap(a, b)
                    for i, a in enumerate(grown) for b in grown[i + 1:])
    return dict(shape=tuple(grid.shape), axis=axis, background=bg, rails=rails,
                objects=objects, grown=grown, records=records, settled=settled,
                count=len(objects), cohorts=len(grown),
                cardinalities=tuple(sorted(len(r['component']) for r in records)),
                totals=tuple(sorted(r['total'] for r in records)),
                widths=tuple(sorted({o['w'] for o in objects})),
                indirect=indirect, new_edges=int(new_edges),
                stable_witness=[o['c'] for o in settled] != [o['c'] for o in old],
                motif_widths=tuple(sorted({int(np.count_nonzero(o['token'] != bg))
                                          for o in objects})),
                travel=tuple(abs(a['start_c'] - b['c']) for a, b in zip(grown, settled)))


def _evidence(features):
    fusions = [r for f in features for r in f['records'] if r['sources']]
    assert len({r['initial'] for r in fusions}) >= 2, "vary the initial receiver count"
    assert len({sum(r['sources']) for r in fusions}) >= 2, "vary the source total"
    assert len({(sum(r['sources']), r['initial']) for r in fusions}) >= 2
    first = fusions[0]
    assert any(sum(r['sources']) * first['initial'] !=
               sum(first['sources']) * r['initial'] for r in fusions), "avoid constant multipliers"
    rivals = (lambda r: 2 * r['initial'],
              lambda r: r['initial'] + len(r['sources']),
              lambda r: r['initial'] + max(r['sources']),
              lambda r: sum(r['sources']),
              lambda r: r['initial'] * sum(r['sources']))
    assert all(any(rival(r) != r['total'] for r in fusions) for rival in rivals)
    assert any(f['indirect'] for f in features), "teach transitive membership"
    assert any(f['new_edges'] and f['stable_witness'] for f in features), "teach grown-span stable packing"
    assert any(len({o['frame'] for o in f['objects']}) == 1 and f['cohorts'] >= 3
               for f in features), "teach separate local sums and one-shot grouping"
    assert any(len(set(r['sources'])) > 1 for r in fusions), "teach token counts rather than equal sources"
    assert any(max(r['sources']) > r['initial'] for r in fusions), "nearest receiver need not be tallest"
    rails = features[0]['rails']
    moved = {a['frame'] for f in features for a in f['settled'] if a['c'] != a['start_c']}
    assert moved == set(rails), "both frame instructions must move"
    assert any(a['frame'] != b['frame'] and np.array_equal(a['token'], b['token'])
               for f in features for a in f['objects'] for b in f['objects']), "payload does not select direction"
    assert any(any(not np.array_equal(f['objects'][r['receiver']]['token'], f['objects'][j]['token'])
                       for j in r['component'] if j != r['receiver'])
               for f in features for r in f['records'] if r['sources']), "copy the receiver's token"
    assert any(any(f['objects'][j]['w'] > f['objects'][r['receiver']]['w']
                       for j in r['component'])
               for f in features for r in f['records']), "nearest receiver need not be widest"
    assert any(r['initial'] + sum(f['objects'][j]['n'] * int(np.count_nonzero(
                   f['objects'][j]['token'] != f['background']))
                   for j in r['component'] if j != r['receiver']) != r['total']
               for f in features for r in f['records'] if r['sources']), "count tokens, not colored pixels"


def _layout_constraints(objects, grown, records, rails):
    for rec in records:
        left = objects[rec['receiver']]['frame'] == rails[0]
        distances = [_order(objects[i], left)[0] for i in rec['component']]
        _require(distances.count(min(distances)) == 1, "cohort needs a unique nearest receiver")
    for i, a in enumerate(grown):
        for b in grown[i + 1:]:
            if a['frame'] == b['frame'] and _band_overlap(a, b):
                left = a['frame'] == rails[0]
                da = a['start_c'] if left else -a['start_c'] - a['w']
                db = b['start_c'] if left else -b['start_c'] - b['w']
                _require(da != db, "interacting receivers need a strict original edge order")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_features, convention = set(), [], None
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            f = _features(inp)
            current = (f['axis'], f['rails'])
            convention = current if convention is None else convention
            assert current == convention, "puzzle conventions must be consistent"
            _layout_constraints(f['objects'], f['grown'], f['records'], f['rails'])
            canonical_out, out_axis = _canonical(out)
            out_bg, out_rails, out_objects = _decode_canonical(canonical_out)
            assert out_axis == f['axis'] and out_bg == f['background'] and out_rails == f['rails']
            assert len(out_objects) == f['cohorts'], "one output tray per original cohort"
            for frame in f['rails']:
                assert sum(o['n'] for o in f['objects'] if o['frame'] == frame) == \
                       sum(o['n'] for o in out_objects if o['frame'] == frame), "token conservation"
            # Check receiver patterns and baselines directly at their settled boxes.
            for o in f['settled']:
                matches = [p for p in out_objects if p['r'] == o['r'] and p['c'] == o['c']]
                assert len(matches) == 1
                p = matches[0]
                assert (p['bottom'], p['w'], p['n'], p['frame']) == \
                       (o['bottom'], o['w'], o['n'], o['frame'])
                assert np.array_equal(p['token'], o['token']), "receiver silhouette and color must be copied"
            assert np.array_equal(solve(inp), out), "solve disagrees"
            if split == 'train':
                train_features.append(f)
    _evidence(train_features)
    assert len({f['shape'] for f in train_features}) >= 2, "demonstrations vary geometry"


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _tray(rng, bg, rails, frame, n, w, bottom):
    allowed = [c for c in range(10) if c not in (bg, frame)]
    color = int(_pick(rng, allowed))
    mask = np.zeros(w - 2, bool)
    length = int(rng.integers(1, w - 1))
    start = int(rng.integers(0, w - 1 - length))
    mask[start:start + length] = True
    token = np.full(w - 2, bg, np.int8)
    token[mask] = color
    h = 2 * n + 1
    return dict(r=bottom - h + 1, c=0, h=h, w=w, n=n,
                frame=int(frame), token=token, bottom=bottom)


def _columns(rng, objects, precedence=(), right=False):
    """Random topological order, random gaps, optional reflection to the right."""
    remaining = set(range(len(objects)))
    order = []
    while remaining:
        available = sorted(i for i in remaining if not any(b == i and a in remaining
                                                          for a, b in precedence))
        _require(bool(available), "cyclic column constraints")
        i = int(_pick(rng, available))
        remaining.remove(i)
        order.append(i)
    minimum = sum(o['w'] for o in objects) + 2
    _require(minimum <= 30, "input does not fit columns")
    width = int(rng.integers(minimum, 31))
    gaps = rng.multinomial(width - minimum, np.full(len(objects) + 1, 1 / (len(objects) + 1)))
    c = 1 + int(gaps[0])
    for k, i in enumerate(order):
        o = objects[i]
        o['c'] = width - 1 - c - o['w'] + 1 if right else c
        c += o['w'] + int(gaps[k + 1])
    return width


def _finish(rng, p, bg, shape, objects):
    rails = p['rails']
    grown, records = _fuse(objects, rails)
    _layout_constraints(objects, grown, records, rails)
    inp = _render(shape, bg, rails, objects)
    out = _render(shape, bg, rails, _pack(grown, shape[1], rails))
    if p['axis']:
        inp, out = inp.T.copy(), out.T.copy()
    return {'input': inp, 'output': out}


def _anchor_chain(rng, p):
    rails, w = p['rails'], p['width']
    bg = int(_pick(rng, [c for c in range(10) if c not in rails]))
    counts = _pick(rng, [(d, a, b) for d in range(1, 4) for a in range(1, 4)
                        for b in range(1, 4) if 4 <= d + a + b <= 6 and a != b])
    frame = int(_pick(rng, rails))
    bottoms = [0]
    # Two one-row contacts make a transitive chain with disjoint endpoints.
    for n in counts[1:]:
        bottoms.append(bottoms[-1] - (2 * counts[len(bottoms) - 1] + 1) + 1)
    widths = [w - 1, w, int(_pick(rng, (w - 1, w)))]
    objects = [_tray(rng, bg, rails, frame, n, ow, b)
               for n, ow, b in zip(counts, widths, bottoms)]
    # Reverse the row geometry sometimes while keeping endpoint zero as receiver.
    if rng.integers(2):
        for o in objects:
            o['r'] = -o['bottom']
            o['bottom'] = o['r'] + o['h'] - 1
    grown, _ = _fuse([dict(o, c=i * w + 1) for i, o in enumerate(objects)], (frame, -1))
    lowest = min([o['r'] for o in objects] + [o['r'] for o in grown])
    shift = 1 - lowest + int(rng.integers(0, 4))
    for o in objects:
        o['r'] += shift
        o['bottom'] += shift
    # Same token under the opposite frame isolates material instruction from payload.
    opposite = rails[1] if frame == rails[0] else rails[0]
    other = _tray(rng, bg, rails, opposite, int(rng.integers(1, 4)), widths[0],
                  max(o['bottom'] for o in objects))
    other['token'] = objects[0]['token'].copy()
    if np.any(other['token'] == opposite):
        replacement = int(_pick(rng, [c for c in range(10) if c not in (*rails, bg)]))
        for o in (objects[0], other):
            o['token'][o['token'] != bg] = replacement
    objects.append(other)
    # Ensure at least one donor has a different appearance and a nontrivial mask.
    donor = objects[1]
    donor['token'][:] = int(_pick(rng, [c for c in range(10) if c not in
                                       (bg, frame, int(objects[0]['token'][objects[0]['token'] != bg][0]))]))
    width = _columns(rng, objects, ((0, 1), (1, 2)), frame == rails[1])
    minimum_h = max(o['bottom'] for o in objects) + 1
    _require(minimum_h <= 30, "anchor height exceeded")
    height = int(rng.integers(max(12, minimum_h), 31))
    return _finish(rng, p, bg, (height, width), objects)


def _anchor_growth(rng, p):
    rails, w = p['rails'], p['width']
    bg = int(_pick(rng, [c for c in range(10) if c not in rails]))
    frame = int(_pick(rng, rails))
    upper = int(rng.integers(1, 4))
    pairs = [_pick(rng, ((1, 2), (1, 3), (2, 2), (2, 3))) for _ in range(2)]
    bottom = 2 * upper
    descriptions = [(upper, bottom)]
    for d, s in pairs:
        gap = int(rng.integers(1, 2 * min(d, s)))
        bottom += gap + 2 * max(d, s) + 1
        descriptions += [(d, bottom), (s, bottom)]
    objects = [_tray(rng, bg, rails, frame, n, w, b) for n, b in descriptions]
    # Receivers are 0,1,3. Their three GROWN intervals form a chain, not a clique.
    order = [0, 1, 3] if rng.integers(2) else [3, 1, 0]
    precedence = [(order[0], order[1]), (order[1], order[2]), (1, 2), (3, 4)]
    width = _columns(rng, objects, precedence, frame == rails[1])
    grown, _ = _fuse(objects, rails)
    lowest = min([o['r'] for o in objects] + [o['r'] for o in grown])
    extent = max(o['bottom'] for o in objects) - lowest + 1
    _require(extent + 1 <= 30, "growth anchor exceeds canvas")
    shift = 1 - lowest + int(rng.integers(0, min(3, 29 - extent) + 1))
    for o in objects:
        o['r'] += shift
        o['bottom'] += shift
    height = int(rng.integers(max(o['bottom'] for o in objects) + 1, 31))
    pair = _finish(rng, p, bg, (height, width), objects)
    _require(_features(pair['input'])['stable_witness'], "anchor must expose stable order")
    return pair


def _random_training(rng, p):
    rails = p['rails']
    for _ in range(250):
        bg = int(_pick(rng, [c for c in range(10) if c not in rails]))
        height, width = int(rng.integers(12, 31)), int(rng.integers(9, 31))
        requested = int(rng.integers(1, 9))
        objects = []
        for _ in range(requested):
            for _ in range(45):
                n = int(rng.integers(1, 4))
                w = int(rng.integers(3, 6))
                bottom = int(rng.integers(2 * n + 1, height))
                o = _tray(rng, bg, rails, int(_pick(rng, rails)), n, w, bottom)
                o['c'] = int(rng.integers(1, width - w))
                if not any(_overlap(o, b) for b in objects):
                    objects.append(o)
                    break
        try:
            grown, records = _fuse(objects, rails)
            if any(len(r['component']) > 3 or r['total'] > 6 for r in records):
                continue
            return _finish(rng, p, bg, (height, width), objects)
        except ValueError:
            continue
    return _anchor_chain(rng, p)


def _partition(rng, total, count):
    cuts = sorted(rng.choice(np.arange(1, total), count - 1, replace=False).tolist())
    return np.diff([0] + cuts + [total]).tolist()


def _query(rng, p):
    rails = p['rails']
    for _ in range(350):
        bg = int(_pick(rng, [c for c in range(10) if c not in rails]))
        frame = int(_pick(rng, rails))
        count = int(rng.integers(4, 6)) if p['shift'] == 'cardinality' else int(rng.integers(3, 6))
        total = int(rng.integers(max(count, 7 if p['shift'] == 'arithmetic' else count + 1), 11))
        counts = _partition(rng, total, count)
        if max(counts) > 4:
            continue
        widths = rng.integers(3, 6, size=count).tolist()
        if sum(widths) > 28:
            continue
        height = int(rng.integers(2 * total + 2, 31))
        bottom = int(rng.integers(2 * total + 1, height))
        objects = [_tray(rng, bg, rails, frame, counts[0], widths[0], bottom)]
        for n, w in zip(counts[1:], widths[1:]):
            neighbor = _pick(rng, objects)
            lo = max(1, neighbor['r'] - 2 * n)
            hi = min(height - (2 * n + 1), neighbor['bottom'])
            if lo > hi:
                break
            r = int(rng.integers(lo, hi + 1))
            objects.append(_tray(rng, bg, rails, frame, n, w, r + 2 * n))
        if len(objects) != count:
            continue
        # Optional extra cohorts/classes; no fixed two-sided population.
        for _ in range(int(rng.integers(0, 3))):
            w, n = int(rng.integers(3, 6)), int(rng.integers(1, 4))
            if sum(o['w'] for o in objects) + w > 28:
                break
            b = int(rng.integers(2 * n + 1, height))
            extra = _tray(rng, bg, rails, int(_pick(rng, rails)), n, w, b)
            if extra['frame'] == frame and any(_band_overlap(extra, o) for o in objects):
                extra['frame'] = rails[1] if frame == rails[0] else rails[0]
                if np.any(extra['token'] == extra['frame']):
                    extra['token'][extra['token'] != bg] = int(_pick(
                        rng, [c for c in range(10) if c not in (*rails, bg)]))
            objects.append(extra)
        try:
            width = _columns(rng, objects, [(0, i) for i in range(1, count)], frame == rails[1])
            return _finish(rng, p, bg, (height, width), objects)
        except ValueError:
            continue
    raise RuntimeError("bounded query construction exhausted")


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    p = dict(rails=tuple(int(c) for c in rng.choice(10, 2, replace=False)),
             axis=bool(rng.integers(2)), width=int(rng.integers(4, 6)),
             shift=str(_pick(rng, ('cardinality', 'arithmetic'))))
    for _ in range(80):
        try:
            anchors = [_anchor_chain(rng, p), _anchor_growth(rng, p)]
            _evidence([_features(pair['input']) for pair in anchors])
            if anchors[0]['input'].shape == anchors[1]['input'].shape:
                continue
            break
        except (ValueError, AssertionError):
            continue
    else:
        raise RuntimeError('bounded demonstration construction exhausted')
    if rng.integers(2):
        anchors.reverse()
    puzzle, seen = {'train': anchors, 'test': []}, set()
    for pair in anchors:
        seen.add((pair['input'].shape, pair['input'].tobytes()))
    for split, count in (('train', num_train - 2), ('test', num_test)):
        for _ in range(count):
            for _ in range(80):
                pair = _random_training(rng, p) if split == 'train' else _query(rng, p)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded uniqueness construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    return tuple((np.asarray(p['input']).shape, np.asarray(p['input']).tobytes(),
                  np.asarray(p['output']).tobytes())
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """Bounded semantic, negative, count, determinism, and 42-seed checks."""
    corpus = [generate_puzzle(seed) for seed in HOST_SEEDS]
    assert len({_fingerprint(p) for p in corpus}) == 42
    rows = [_features(np.asarray(pair['input'])) for p in corpus
            for split in ('train', 'test') for pair in p[split]]
    assert {f['axis'] for f in rows} == {False, True}
    assert len({f['rails'] for f in rows}) >= 20
    assert len({f['shape'] for f in rows}) >= 60
    assert len({f['count'] for f in rows}) >= 5
    assert len({f['cardinalities'] for f in rows}) >= 12
    assert len({f['widths'] for f in rows}) >= 5
    assert {1, 2, 3} <= {n for f in rows for n in f['motif_widths']}
    assert any(f['count'] == 1 for f in rows)
    assert any(max(f['totals']) >= 9 for f in rows)
    assert any(f['shape'] == (30, 30) for f in rows)
    assert any(min(f['travel']) == 0 and max(f['travel']) > 0 for f in rows)
    for p in corpus:
        fs = [_features(pair['input']) for pair in p['train'] + p['test']]
        assert len({f['count'] for f in fs}) >= 2
        assert any(f['indirect'] for f in fs[:4])
        assert any(f['stable_witness'] for f in fs[:4])
        # At least one advertised shift is visible in the query pixels.
        tr, te = fs[:4], fs[4:]
        assert (max(max(f['cardinalities']) for f in te) > max(max(f['cardinalities']) for f in tr)
                or max(max(f['totals']) for f in te) > max(max(f['totals']) for f in tr))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(0, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    assert _fingerprint(corpus[0]) == _fingerprint(generate_puzzle(0))
    # Exercise valid family geometry beyond the sampler's usual 3..5 widths:
    # thirteen source tokens plus one receiver reach the exact 29-row capacity.
    token = np.array([0, 8, 8, 0, 0], dtype=np.int8)
    receiver = dict(r=27, c=2, h=3, w=7, n=1, frame=2, token=token, bottom=29)
    donor = dict(r=3, c=13, h=27, w=7, n=13, frame=2,
                 token=np.array([3, 3, 3, 3, 3], dtype=np.int8), bottom=29)
    boundary = _render((30, 30), 0, (2, 6), [receiver, donor])
    expected_boundary = _render((30, 30), 0, (2, 6),
                                [dict(receiver, r=1, c=1, h=29, n=14)])
    assert np.array_equal(solve(boundary), expected_boundary)
    assert np.array_equal(solve(boundary.T), expected_boundary.T)
    a_changes, b_changes = [], []
    for pair in corpus[0]['train']:
        a, axis = _canonical(pair['input'])
        bg, rails, objects = _decode_canonical(a)
        grown, _ = _fuse(objects, rails)
        unadded, _ = _fuse(objects, rails, add=False)
        without_a = _render(a.shape, bg, rails, _pack(unadded, a.shape[1], rails))
        without_motion = _render(a.shape, bg, rails, grown)
        expected = pair['output'].T if axis else pair['output']
        a_changes.append(not np.array_equal(expected, without_a))
        b_changes.append(not np.array_equal(expected, without_motion))
    assert any(a_changes), 'A must affect the answer'
    assert any(b_changes), 'B must affect the answer'
    # Wrong edge-first placement preserves counts, frames and copied patterns;
    # the validator must still reject same-class overtaking.
    order_broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in corpus[0][s]]
                    for s in ('train', 'test')}
    for pair in order_broken['train']:
        f = _features(pair['input'])
        if f['stable_witness']:
            a, axis = _canonical(pair['input'])
            old = _render(a.shape, f['background'], f['rails'],
                          _pack(f['grown'], a.shape[1], f['rails'], edge_first=True))
            pair['output'] = old.T.copy() if axis else old
            break
    try:
        validate_generated_puzzle(order_broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('same-class overtaking accepted')
    # A pixel mutation must be rejected even when schema is intact.
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in corpus[0][s]]
              for s in ('train', 'test')}
    inp = broken['test'][0]['input']
    broken['test'][0]['output'] = inp.copy()
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('incorrect output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 host seeds, all legal counts, semantic witnesses')
