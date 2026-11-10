"""Registered mirror sheets: complementary damage plus reflection-orbit recovery.

Only standard-library modules and NumPy are used. No files are read at runtime.
All diagnostics are returned in memory; self_test writes nothing.
"""
from collections import Counter
import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = np.array(((-1, 0), (0, 1), (1, 0), (0, -1)), dtype=int)
ROT = np.array(((0, -1), (1, 0)), dtype=int)
MATRICES = tuple(np.linalg.matrix_power(ROT, k) @
                 (np.diag((1, -1)) if flip else np.eye(2, dtype=int))
                 for flip in (False, True) for k in range(4))


def _transform(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _side_map(code, side):
    normal = MATRICES[code] @ NORMALS[side]
    return next(i for i, n in enumerate(NORMALS) if np.array_equal(n, normal))


def _move(a, matrix):
    """Solver's coordinate placement, separate from construction's array views."""
    target = matrix @ np.indices(a.shape).reshape(2, -1)
    target -= target.min(axis=1, keepdims=True)
    out = np.empty(tuple(target.max(axis=1) + 1), dtype=a.dtype)
    out[target[0], target[1]] = a.ravel()
    return out


def _edge(a, side):
    return (a[0, 1:-1], a[1:-1, -1], a[-1, 1:-1], a[1:-1, 0])[side]


def _segment(a, box, side):
    r, c, h, w = box
    if side == 0:
        return a[r - 1, c:c + w]
    if side == 1:
        return a[r:r + h, c + w]
    if side == 2:
        return a[r + h, c:c + w]
    return a[r:r + h, c - 1]


def _components(a, color):
    seen = set()
    result = []
    H, W = a.shape
    for rr, cc in np.argwhere(a == color):
        start = int(rr), int(cc)
        if start in seen:
            continue
        seen.add(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= p[0] < H and 0 <= p[1] < W and
                        p not in seen and a[p] == color):
                    seen.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _grid(value):
    a = np.asarray(value)
    if not (a.ndim == 2 and all(1 <= d <= 30 for d in a.shape) and
            np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _decode(input_grid):
    """Read controls, isolated bars, exact rectangles and local coordinate frames."""
    a = _grid(input_grid)
    H, W = a.shape
    bg = int(a[0, 0])
    assert all(int(a[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    outer = {}
    for side in range(4):
        line = _edge(a, side)
        assert len(line) and np.all(line == line[0]), "Broken outer control"
        if int(line[0]) != bg:
            outer[side] = int(line[0])
    assert len(outer) == 2 and len(set(outer.values())) == 2
    assert (list(outer)[0] - list(outer)[1]) % 2 == 1
    bars = []
    for color in sorted(outer.values()):
        for cells in _components(a, color):
            if any(r in (0, H - 1) or c in (0, W - 1) for r, c in cells):
                continue
            rows, cols = zip(*cells)
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            assert len(cells) >= 3
            assert r0 == r1 or c0 == c1, "Non-straight guide"
            assert len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)
            bars.append((color, r0, r1, c0, c1))
    candidates = []
    for hi, horizontal in enumerate(bars):
        hc, hr, hr1, c0, c1 = horizontal
        if hr != hr1:
            continue
        for vi, vertical in enumerate(bars):
            vc, r0, r1, vv, vv1 = vertical
            if vv != vv1 or hc == vc:
                continue
            hs = 0 if hr == r0 - 1 else 2 if hr == r1 + 1 else None
            vs = 3 if vv == c0 - 1 else 1 if vv == c1 + 1 else None
            if hs is None or vs is None:
                continue
            box = r0, c0, r1 - r0 + 1, c1 - c0 + 1
            r, c, h, w = box
            if not (r >= 2 and c >= 2 and r + h <= H - 2 and c + w <= W - 2):
                continue
            payload = a[r:r + h, c:c + w].copy()
            if set(np.unique(payload)) & set(outer.values()):
                continue
            guides = {hs: hc, vs: vc}
            if any(not np.all(_segment(a, box, s) == guides.get(s, bg)) for s in range(4)):
                continue
            # A whole background moat separates this envelope from every other.
            ring = np.concatenate((a[r - 2, c - 2:c + w + 2], a[r + h + 1, c - 2:c + w + 2],
                                   a[r - 1:r + h + 1, c - 2], a[r - 1:r + h + 1, c + w + 1]))
            if not np.all(ring == bg):
                continue
            matrix = np.zeros((2, 2), dtype=int)
            for source, color in guides.items():
                dest = next(s for s in outer if outer[s] == color)
                matrix += np.outer(NORMALS[dest], NORMALS[source])
            aligned = _move(payload, matrix)
            candidates.append(dict(box=box, guides=guides, matrix=matrix,
                                   payload=payload, aligned=aligned, bars=(hi, vi)))
    assert 2 <= len(candidates) <= 4, "Need two to four unambiguous sheets"
    used = [b for info in candidates for b in info['bars']]
    assert sorted(used) == list(range(len(bars))), "Ambiguous or unmatched bars"
    assert len({info['aligned'].shape for info in candidates}) == 1
    occupied = np.zeros(a.shape, dtype=bool)
    expected = np.full(a.shape, bg, dtype=np.int8)
    for s, color in outer.items():
        _edge(expected, s)[:] = color
    for info in candidates:
        r, c, h, w = info['box']
        assert not occupied[r - 1:r + h + 1, c - 1:c + w + 1].any()
        occupied[r - 1:r + h + 1, c - 1:c + w + 1] = True
        expected[r:r + h, c:c + w] = info['payload']
        for s, color in info['guides'].items():
            _segment(expected, info['box'], s)[:] = color
    assert np.array_equal(expected, a), "Unexplained material outside sheets"
    return dict(background=bg, outer=outer, sheets=candidates)


def _recover(layers, hole, axis):
    """Equate same-position observations and the two members of each mirror orbit."""
    stack = np.stack(layers)
    visible = stack != hole
    lo = np.where(visible, stack, 10).min(axis=0)
    hi = np.where(visible, stack, -1).max(axis=0)
    low = np.minimum(lo, np.flip(lo, axis=axis))
    high = np.maximum(hi, np.flip(hi, axis=axis))
    if not (np.all(high >= 0) and np.array_equal(low, high)):
        return None
    # Substantial visible mirror evidence; neither an arbitrary completion nor
    # a one-cell accidental agreement establishes a reflection.
    known = hi >= 0
    witnesses = known & np.flip(known, axis=axis)
    coords = np.indices(hi.shape)[axis]
    witnesses &= coords != hi.shape[axis] - 1 - coords
    if witnesses.sum() < max(8, hi.size // 4):
        return None
    if len(np.unique(high)) < 2:
        return None
    return high.astype(np.int8)


def _infer(input_grid):
    info = _decode(input_grid)
    layers = [sheet['aligned'] for sheet in info['sheets']]
    colors = sorted(set(np.unique(np.stack(layers))) - {info['background']})
    hypotheses = []
    for hole in colors:
        if not all(np.any(a == hole) for a in layers):
            continue
        for axis in (0, 1):
            answer = _recover(layers, hole, axis)
            if answer is not None:
                hypotheses.append((int(hole), axis, answer))
    assert hypotheses, "No recoverable, witnessed mirror model"
    assert len({h for h, _, _ in hypotheses}) == 1, "Ambiguous erasure identity"
    answer = hypotheses[0][2]
    assert all(np.array_equal(answer, p) for _, _, p in hypotheses), "Conflicting completions"
    hole, axis, _ = hypotheses[0]
    info.update(hole=hole, axis=axis, hypotheses=hypotheses, answer=answer)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every role, registration and surviving reflection orbit from this input."""
    return _infer(input_grid)['answer'].copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for value, lower, upper in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lower <= int(value) <= upper):
            raise ValueError("Counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _pattern(rng, m, w, colors, style):
    colors = np.asarray(colors, dtype=np.int8)
    if style == 0:
        base = rng.choice(colors, size=(m, w))
    elif style == 1:
        base = np.full((m, w), colors[0], dtype=np.int8)
        for _ in range(int(rng.integers(3, 8))):
            r, c = int(rng.integers(m)), int(rng.integers(w))
            rh, cw = int(rng.integers(1, m + 1)), int(rng.integers(1, w + 1))
            base[r:r + rh, c:c + cw] = int(rng.choice(colors))
    elif style == 2:
        period = int(rng.integers(2, 5))
        tile = rng.choice(colors, size=(min(2, m), period))
        base = np.tile(tile, (m, w))[:m, :w].copy()
        for _ in range(3):
            base[int(rng.integers(m)), int(rng.integers(w))] = int(rng.choice(colors))
    else:
        base = np.full((m, w), colors[0], dtype=np.int8)
        for color in colors[1:]:
            r, c = int(rng.integers(m)), int(rng.integers(w))
            for _ in range(int(rng.integers(3, m * w))):
                base[r, c] = color
                dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[int(rng.integers(4))]
                r, c = min(m - 1, max(0, r + dr)), min(w - 1, max(0, c + dc))
    for color, index in zip(colors, rng.choice(m * w, size=len(colors), replace=False)):
        base.flat[int(index)] = color
    return base.astype(np.int8)


def _damage(rng, full, n, hole):
    h, w = full.shape
    top_half = h // 2
    # A one-sided wound shared by all sheets remains absent after direct overlay.
    r = int(rng.integers(top_half))
    rh = int(rng.integers(1, top_half - r + 1))
    cw = int(rng.integers(1, min(4, w) + 1))
    c = int(rng.integers(w - cw + 1))
    cuts = [0] + sorted(int(v) for v in rng.choice(np.arange(1, w), n - 1, replace=False)) + [w]
    intervals = list(zip(cuts[:-1], cuts[1:]))
    rng.shuffle(intervals)
    masks = []
    for a, b in intervals:
        mask = np.zeros(full.shape, dtype=bool)
        mask[r:r + rh, c:c + cw] = True
        left = int(rng.integers(a, b))
        right = int(rng.integers(left + 1, b + 1))
        start = int(rng.integers((h + 1) // 2))
        # This symmetric wound deletes a complete orbit from this sheet alone.
        mask[start:h - start, left:right] = True
        masks.append(mask)
    # Optional extra wounds are accepted only if collective orbit support survives.
    for i in range(n):
        if rng.random() < .45:
            r0, c0 = int(rng.integers(h)), int(rng.integers(w))
            rr, cc = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            trial = masks[i].copy()
            trial[r0:r0 + rr, c0:c0 + cc] = True
            old = masks[i]
            masks[i] = trial
            jointly_missing = np.logical_and.reduce(masks)
            if np.any(jointly_missing & jointly_missing[::-1]):
                masks[i] = old
    return [np.where(mask, hole, full).astype(np.int8) for mask in masks]


def _conflict_colors(layers, hole):
    alternatives = set()
    for i, x in enumerate(layers):
        for y in layers[i + 1:]:
            for values in (x[(x != y) & (y == hole)], y[(x != y) & (x == hole)]):
                alternatives.update(int(v) for v in values)
    return alternatives


def _pack(rng, shapes):
    """Shelf packing with random order, orientation, independent jitter and margins.

    At most four 12x12 envelopes plus moat/margins fit in a 30x30 canvas.
    This is placement only, not a semantic partition of the input.
    """
    n = len(shapes)
    order = [int(i) for i in rng.permutation(n)]
    layouts = [(2, 2)] if n >= 3 else [(1, 2), (2, 1), (2, 2)]
    if n == 3 and max(max(s) for s in shapes) <= 6:
        layouts += [(1, 3), (3, 1)]
    nr, nc = layouts[int(rng.integers(len(layouts)))]
    slots = [(i // nc, i % nc) for i in range(n)]
    row_sizes = [max([shapes[order[i]][0] + 2 for i in range(n) if slots[i][0] == r] or [0]) for r in range(nr)]
    col_sizes = [max([shapes[order[i]][1] + 2 for i in range(n) if slots[i][1] == c] or [0]) for c in range(nc)]
    min_h, min_w = sum(row_sizes) + nr - 1 + 4, sum(col_sizes) + nc - 1 + 4
    assert min_h <= 30 and min_w <= 30
    H, W = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
    rmargin, cmargin = int(rng.integers(H - min_h + 1)), int(rng.integers(W - min_w + 1))
    result = [None] * n
    for i, index in enumerate(order):
        rr, cc = slots[i]
        h, w = shapes[index]
        r = 3 + rmargin + sum(row_sizes[:rr]) + rr + int(rng.integers(row_sizes[rr] - h - 1))
        c = 3 + cmargin + sum(col_sizes[:cc]) + cc + int(rng.integers(col_sizes[cc] - w - 1))
        result[index] = (r, c, h, w)
    return (H, W), result


def _generate_pair(rng, target_corner, n, style, palette=None, teach_blank=False):
    for _ in range(100):
        roles = [int(x) for x in rng.permutation(10)] if palette is None else palette
        bg, m0, m1, hole = roles[:4]
        colors = roles[4:4 + int(rng.integers(2, 6))]
        use_background = teach_blank or rng.random() < .35
        if use_background:
            colors = [bg] + colors
        m, w = int(rng.integers(3, 6)), int(rng.integers(max(n, 5), 11))
        base = _pattern(rng, m, w, colors, style)
        if use_background and (teach_blank or rng.random() < .7):
            # A full blank strip gives guide-defined extents a visible consequence.
            base[:, 0 if rng.random() < .5 else -1] = bg
        through_cell = bool(rng.integers(2))
        full = np.concatenate((base, base[-2::-1] if through_cell else base[::-1]), axis=0)
        # Only the intended axial reflection is compulsory; accidental ones are
        # excluded here to keep the marker teaching contrast visibly informative.
        if len({_key(_transform(full, k)) for k in range(8)}) != 4:
            continue
        layers = _damage(rng, full, n, hole)
        if len(_conflict_colors(layers, hole)) < 2:
            continue
        masks = np.stack(layers) == hole
        joint = np.logical_and.reduce(masks)
        if not np.any(full[joint] != bg):
            continue
        if any(not np.any(full[mask & mask[::-1]] != bg) for mask in masks):
            continue
        codes = [int(rng.integers(8))]
        # Both local axis orientations are visible even with only two sheets.
        choices = [k for k in range(8) if (k % 2) != (codes[0] % 2)]
        codes.append(int(rng.choice(choices)))
        codes += [int(rng.integers(8)) for _ in range(n - 2)]
        local = [_transform(a, k) for a, k in zip(layers, codes)]
        try:
            shape, boxes = _pack(rng, [a.shape for a in local])
        except AssertionError:
            continue
        # The corner constrains only occupied sides. Their identities remain free.
        eligible = [k for k in range(8) if {_side_map(k, 0), _side_map(k, 3)} ==
                    {target_corner, (target_corner + 1) % 4}]
        global_code = int(rng.choice(eligible))
        a = np.full(shape, bg, dtype=np.int8)
        for side, color in ((_side_map(global_code, 0), m0), (_side_map(global_code, 3), m1)):
            _edge(a, side)[:] = color
        for values, code, box in zip(local, codes, boxes):
            r, c, h, w = box
            a[r:r + h, c:c + w] = values
            for side, color in ((_side_map(code, 0), m0), (_side_map(code, 3), m1)):
                _segment(a, box, side)[:] = color
        output = _transform(full, global_code)
        try:
            info = _infer(a)
        except AssertionError:
            continue
        if np.array_equal(info['answer'], output):
            return {'input': a, 'output': output}
    raise RuntimeError("Bounded mirror-sheet construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    held_corner = int(rng.integers(4))
    familiar = [c for c in range(4) if c != held_corner]
    preferred_style = int(rng.integers(4))
    stable_palette = [int(x) for x in rng.permutation(10)] if rng.random() < .2 else None
    first_corner = int(rng.choice(familiar))
    first = _generate_pair(rng, first_corner, int(rng.integers(2, 5)), preferred_style,
                           stable_palette, teach_blank=True)
    # A controlled intervention makes matching identities essential in training.
    second = {k: v.copy() for k, v in first.items()}
    outer = _decode(first['input'])['outer']
    sides, colors = list(outer), list(outer.values())
    for side, color in zip(sides, colors[::-1]):
        _edge(second['input'], side)[:] = color
    # Swapping the output-side normals is a diagonal reflection of the answer.
    swap = np.outer(NORMALS[sides[1]], NORMALS[sides[0]]) + np.outer(NORMALS[sides[0]], NORMALS[sides[1]])
    second['output'] = _move(first['output'], swap)
    puzzle = {'train': [first, second], 'test': []}
    seen = {_key(first['input']), _key(second['input'])}
    for index in range(2, nt + ne):
        split = 'train' if index < nt else 'test'
        corner = int(rng.choice(familiar)) if split == 'train' else held_corner
        for _ in range(30):
            n = int(rng.integers(2, 5))
            style = preferred_style if rng.random() < .55 else int(rng.integers(4))
            pair = _generate_pair(rng, corner, n, style, stable_palette)
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError("Distinct-input budget exhausted")
        puzzle[split].append(pair)
        seen.add(_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def _family_features(info):
    layers = [x['aligned'] for x in info['sheets']]
    masks = np.stack(layers) == info['hole']
    axis = info['axis']
    return dict(sheets=len(layers), shape=layers[0].shape, hole=info['hole'], axis=axis,
                shared_missing=int(np.logical_and.reduce(masks).sum()),
                private_losses=tuple(int((x & np.flip(x, axis=axis)).sum()) for x in masks),
                blank_edges=sum(bool(np.all(line == info['background'])) for line in
                                (info['answer'][0], info['answer'][-1], info['answer'][:, 0], info['answer'][:, -1])),
                control_corner=tuple(sorted(info['outer'])),
                registrations=tuple(next(k for k, m in enumerate(MATRICES) if np.array_equal(m, x['matrix']))
                                    for x in info['sheets']))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _infer(a)
            assert np.array_equal(info['answer'], b), "Incorrect reconstructed sheet"
            assert info['hole'] not in b and not (set(info['outer'].values()) & set(np.unique(b)))
            features = _family_features(info)
            # These are generation evidence requirements, not solver assumptions.
            assert features['shared_missing'] > 0, "Direct overlay would suffice"
            assert all(x > 0 for x in features['private_losses']), "An individual sheet would suffice"
            masks = np.stack([x['aligned'] == info['hole'] for x in info['sheets']])
            joint = np.logical_and.reduce(masks)
            assert not np.any(joint & np.flip(joint, axis=info['axis'])), "Erased entire collective orbit"
            assert np.any(b[joint] != info['background'])
            assert all(np.any(b[mask & np.flip(mask, axis=info['axis'])] != info['background'])
                       for mask in masks)
            # At least two differently colored cross-copy conflicts identify the
            # common eraser rather than a content color mistaken for missing data.
            layers = [x['aligned'] for x in info['sheets']]
            assert len(_conflict_colors(layers, info['hole'])) >= 2
            records[split].append((info, features))
    # The same inner evidence, same bar positions, opposite outer identities.
    a, b = (np.asarray(p['input']) for p in puzzle['train'][:2])
    assert a.shape == b.shape and np.array_equal(a[1:-1, 1:-1], b[1:-1, 1:-1])
    oa, ob = (r[0]['outer'] for r in records['train'][:2])
    assert set(oa) == set(ob) and all(oa[s] != ob[s] for s in oa)
    assert not np.array_equal(puzzle['train'][0]['output'], puzzle['train'][1]['output'])
    train_corners = {r[1]['control_corner'] for r in records['train']}
    assert all(r[1]['control_corner'] not in train_corners for r in records['test'])
    assert any(r[1]['blank_edges'] for r in records['train'])


def self_test():
    """42 seeds, all count combinations, equivariance and intentional mutations.

    Streaming aggregation keeps at most two puzzles resident. No saved grids,
    logs, traces or JSON files; runtime is bounded by finite construction loops.
    """
    features, signatures, input_shapes, output_shapes = [], set(), set(), set()
    registrations, palette_sizes, sheet_counts, holes, blanks = set(), set(), Counter(), set(), 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(x['input']) for s in p.values() for x in s))
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _infer(pair['input'])
                f = _family_features(info)
                features.append((f['axis'], f['shape'], f['shared_missing']))
                input_shapes.add(pair['input'].shape)
                output_shapes.add(pair['output'].shape)
                registrations.update(f['registrations'])
                sheet_counts[f['sheets']] += 1
                holes.add(f['hole'])
                palette_sizes.add(len(np.unique(pair['output'])))
                blanks += f['blank_edges'] > 0
                assert np.array_equal(solve(pair['input']), pair['output'])
        if seed in (0, 17, 41):
            q = generate_puzzle(seed)
            assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    assert len(signatures) == 42 and len(input_shapes) >= 15 and len(output_shapes) >= 15
    assert registrations == set(range(8)) and set(sheet_counts) == {2, 3, 4}
    assert len(holes) >= 8 and len(palette_sizes) >= 3 and blanks > 0
    assert {f[0] for f in features} == {0, 1}
    assert {min(f[1]) % 2 for f in features} == {0, 1}
    for nt in range(2, 11):
        for ne in range(1, min(nt - 1, 4) + 1):
            q = generate_puzzle(123 + nt * 11 + ne, nt, ne)
            assert len(q['train']) == nt and len(q['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    p = generate_puzzle(817, 2, 1)
    # Joint rotations and color bijections preserve the family rule.
    for pair in p['train'] + p['test']:
        permutation = np.random.default_rng(71).permutation(10)
        for k in pair:
            pair[k] = permutation[np.rot90(pair[k])].astype(np.int8)
    validate_generated_puzzle(p)
    q = generate_puzzle(1, 2, 1)
    q['test'][0]['output'][0, 0] = (int(q['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong output accepted")
    # Remove all observations of one orbit, then introduce a conflicting witness.
    info = _infer(generate_puzzle(7, 2, 1)['train'][0]['input'])
    layers = [x['aligned'].copy() for x in info['sheets']]
    for x in layers:
        x[0, 0] = info['hole']
        mate = (x.shape[0] - 1, 0) if info['axis'] == 0 else (0, x.shape[1] - 1)
        x[mate] = info['hole']
    assert _recover(layers, info['hole'], info['axis']) is None
    conflicting = [info['answer'].copy(), info['answer'].copy()]
    replacement = next(c for c in range(10) if c not in (info['hole'], int(conflicting[0][0, 0])))
    conflicting[0][0, 0] = replacement
    assert _recover(conflicting, info['hole'], info['axis']) is None
    damaged = generate_puzzle(2, 2, 1)
    pair = damaged['test'][0]
    decoded = _decode(pair['input'])
    sheet = decoded['sheets'][0]
    side = next(iter(sheet['guides']))
    line = _segment(pair['input'], sheet['box'], side)
    line[len(line) // 2] = decoded['background']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Broken guide accepted")
    report = dict(seeds=42, pairs=len(features), distinct_input_shapes=len(input_shapes),
                  distinct_output_shapes=len(output_shapes), registrations=sorted(registrations),
                  sheet_counts=dict(sorted(sheet_counts.items())), hole_colors=sorted(holes),
                  output_palette_sizes=sorted(palette_sizes), blank_edge_pairs=blanks)
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
