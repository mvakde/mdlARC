from __future__ import annotations

import hashlib
import numpy as np

# Exact source input/output fingerprints. Source dimensions remain available.
_SOURCE = {
    '940741714ecbda2c5dd62dca1a49aab98d36a802a6e12351d2bdda30d423e2d7',
    'e751a1ae740969407623a9d4fe5966f2fc9900a6d57a8aa60c178c0a02774f7a',
    '2e80f23dc1ce4c474e16ab300b3336e3bc742f06942087662549415fd4fe6a91',
    '6e9c94ffd8bc374ef5ee3a6c47893ba29c3adfd369b46345b31dced2171fa6f8',
}


def _fingerprint(a, b):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.uint8).tobytes()
                          + b.astype(np.uint8).tobytes()).hexdigest()


def _counts(seed, rng, num_train, num_test):
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train is None and num_test is None:
        nt = 1 + seed % 4
        nr = max(nt + 1, 2 + (seed // 4) % 9)
    elif num_train is None:
        nt = int(num_test)
        nr = int(rng.integers(nt + 1, 11))
    elif num_test is None:
        nr = int(num_train)
        nt = int(rng.integers(1, min(4, nr - 1) + 1))
    else:
        nr, nt = int(num_train), int(num_test)
    if nr <= nt:
        raise ValueError('num_train must exceed num_test')
    return nr, nt


def _scale(rng, seed, section, index, training):
    mode = seed % 4
    if section == 'train':
        if mode == 0:
            pool = ((2, 2), (5, 5), (3, 3))
        elif mode == 1:
            pool = ((4, 4), (2, 3), (3, 5))
        elif mode == 2:
            held = (3, 5, 6)[(seed // 4) % 3]
            pool = tuple((s, s) for s in (2, 3, 4, 5, 6) if s != held)
        else:
            pool = ((2, 4), (4, 3), (3, 5), (5, 2))
        return pool[index] if index < 2 else pool[int(rng.integers(len(pool)))]
    if mode == 0:
        return 4, 4
    if mode == 2:
        held = (3, 5, 6)[(seed // 4) % 3]
        return held, held
    if mode == 3 and index % 2 == 0:
        return 3, 5
    return training[int(rng.integers(len(training)))]


def _geometry(rng, rh, cw, force_tail):
    forced_axis = ('row', 'col')[int(rng.integers(2))] if force_tail else None
    result = []
    for scale, forced in ((rh, forced_axis == 'row'), (cw, forced_axis == 'col')):
        max_full = min(10, 30 // (scale + 1))
        if forced and max_full * (scale + 1) == 30:
            max_full -= 1
        full = int(rng.integers(4, max_full + 1))
        room = min(scale - 1, 30 - full * (scale + 1))
        tail = int(rng.integers(1, room + 1)) if forced else (
            int(rng.integers(room + 1)) if room and int(rng.integers(3)) else 0)
        result.append([scale] * full + ([tail] if tail else []))
    return result[0], result[1], forced_axis


def _three(rng, size):
    return sorted(int(v) for v in rng.choice(size, 3, replace=False))


def _markers(rng, rows, cols, kind, tail_axis):
    if kind == 'multi':
        r0, r1, r2, r3 = sorted(int(v) for v in rng.choice(rows, 4, replace=False))
        c0, cm, c2 = _three(rng, cols)
        a, b = {(r1, c0), (r1, c2), (r2, c0), (r2, c2)}, {(r0, cm), (r3, cm)}
    elif kind == 'long':
        r0, rm, r2 = 0, int(rng.integers(1, rows - 1)), rows - 1
        c0, cm, c2 = 0, int(rng.integers(1, cols - 1)), cols - 1
        a = {(rm, c0), (rm, c2), (r0, c0), (r2, c0), (r0, c2)}
        b = {(r0, cm), (r2, cm)}
    elif kind == 'disjoint':
        ra, rb = sorted(int(v) for v in rng.choice(rows, 2, replace=False))
        ca = sorted(int(v) for v in rng.choice(cols, 2, replace=False))
        cb = sorted(int(v) for v in rng.choice(cols, 2, replace=False))
        a, b = {(ra, ca[0]), (ra, ca[1])}, {(rb, cb[0]), (rb, cb[1])}
    else:
        r0, rm, r2 = _three(rng, rows)
        c0, cm, c2 = _three(rng, cols)
        if kind == 'cross':
            a, b = {(rm, c0), (rm, c2)}, {(r0, cm), (r2, cm)}
        elif kind == 'marker_cross':
            a, b = {(rm, c0), (rm, c2)}, {(r0, c0), (r2, c0)}
        elif kind == 'L':
            a, b = {(rm, c0), (rm, c2), (r2, c0)}, {(r0, cm), (r2, cm)}
        elif kind == 'branch':
            a, b = {(rm, c0), (rm, c2), (r0, c0), (r2, c0)}, {(r0, cm), (r2, cm)}
        else:
            raise ValueError('unknown topology')
    if tail_axis is not None:
        axis = 0 if tail_axis == 'row' else 1
        size = rows if axis == 0 else cols
        used = sorted({p[axis] for p in a | b})
        lower = sorted(int(v) for v in rng.choice(size - 1, len(used) - 1, replace=False))
        mapping = dict(zip(used, lower + [size - 1]))
        if axis == 0:
            a, b = ({(mapping[r], c) for r, c in group} for group in (a, b))
        else:
            a, b = ({(r, mapping[c]) for r, c in group} for group in (a, b))
    if int(rng.integers(2)):
        a, b = b, a
    return a, b


def _segments(sizes):
    spans, lines, pos = [], [], 0
    for i, size in enumerate(sizes):
        spans.append((pos, pos + size))
        pos += size
        if i < len(sizes) - 1 or size == sizes[0]:
            lines.append(pos)
            pos += 1
    return spans, lines, pos


def _coverage(points):
    covered = set(points)
    for r1, c1 in points:
        for r2, c2 in points:
            if r1 == r2:
                covered.update((r1, c) for c in range(min(c1, c2), max(c1, c2) + 1))
            if c1 == c2:
                covered.update((r, c1) for r in range(min(r1, r2), max(r1, r2) + 1))
    return covered


def _draw(rng, row_sizes, col_sizes, routes, palette, mismatch):
    bg, divider, color_a, color_b, crossing = palette
    rr, rlines, height = _segments(row_sizes)
    cc, clines, width = _segments(col_sizes)
    inp = np.full((height, width), bg, dtype=np.int8)
    inp[rlines, :] = divider
    inp[:, clines] = divider
    for color, points in zip((color_a, color_b), routes):
        ordered = sorted(points)
        if not mismatch:
            mh = min(row_sizes[r] for r, _ in points)
            mw = min(col_sizes[c] for _, c in points)
            common = int(rng.integers(mh)), int(rng.integers(mw))
        for i, (r, c) in enumerate(ordered):
            if mismatch:
                offset = ((0, 0) if i == 0 else
                          (row_sizes[r] - 1, col_sizes[c] - 1) if i == 1 else
                          (int(rng.integers(row_sizes[r])), int(rng.integers(col_sizes[c]))))
            else:
                offset = common
            inp[rr[r][0] + offset[0], cc[c][0] + offset[1]] = color
    out = inp.copy()
    a, b = (_coverage(points) for points in routes)
    for r, c in a | b:
        color = crossing if (r, c) in a and (r, c) in b else (color_a if (r, c) in a else color_b)
        out[rr[r][0]:rr[r][1], cc[c][0]:cc[c][1]] = color
    return {'input': inp, 'output': out}


def _inspect(pair):
    inp = pair['input']
    candidates = []
    for color in np.unique(inp):
        rl = [r for r in range(inp.shape[0]) if np.all(inp[r] == color)]
        cl = [c for c in range(inp.shape[1]) if np.all(inp[:, c] == color)]
        if len(rl) >= 2 and len(cl) >= 2:
            candidates.append((int(color), rl, cl))
    assert len(candidates) == 1, 'unclear full-line divider'
    divider, rlines, clines = candidates[0]
    def spans(size, lines):
        starts = [0] + [v + 1 for v in lines if v + 1 < size]
        ends = lines + ([size] if lines[-1] < size - 1 else [])
        assert len(starts) == len(ends)
        return list(zip(starts, ends))
    rr, cc = spans(inp.shape[0], rlines), spans(inp.shape[1], clines)
    assert min(hi - lo for lo, hi in rr + cc) >= 1
    values, counts = np.unique(inp[inp != divider], return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    colors = sorted(set(int(v) for v in np.unique(inp)) - {divider, bg})
    assert len(colors) == 2, 'two marker identities required'
    routes, offsets = [], []
    for color in colors:
        points, off = set(), []
        for row, col in np.argwhere(inp == color):
            rs = [i for i, (lo, hi) in enumerate(rr) if lo <= row < hi]
            cs = [i for i, (lo, hi) in enumerate(cc) if lo <= col < hi]
            assert len(rs) == len(cs) == 1
            r, c = rs[0], cs[0]
            assert (r, c) not in points
            points.add((r, c))
            off.append((int(row) - rr[r][0], int(col) - cc[c][0]))
        assert len(points) >= 2
        assert all(any(q != p and (q[0] == p[0] or q[1] == p[1]) for q in points) for p in points)
        routes.append(points)
        offsets.append(off)
    assert not routes[0] & routes[1]
    a, b = (_coverage(points) for points in routes)
    return {'divider': divider, 'background': bg, 'colors': colors,
            'rr': rr, 'cc': cc, 'rlines': rlines, 'clines': clines,
            'routes': routes, 'coverage': (a, b), 'crossings': a & b, 'offsets': offsets}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {'train', 'test'}
    train, test = puzzle['train'], puzzle['test']
    assert type(train) is type(test) is list
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, crossing_colors, observations = set(), set(), []
    training_tail, training_offset, training_marker_cross = False, False, False
    for section in (train, test):
        for pair in section:
            assert type(pair) is dict and set(pair) == {'input', 'output'}
            a, b = pair['input'], pair['output']
            assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
            assert a.ndim == b.ndim == 2 and a.shape == b.shape
            assert 1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30
            assert np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer)
            assert np.all((0 <= a) & (a <= 9)) and np.all((0 <= b) & (b <= 9))
            key = a.shape, a.astype(np.uint8).tobytes()
            assert key not in seen, 'duplicate input'
            seen.add(key)
            assert _fingerprint(a, b) not in _SOURCE, 'source copy'
            assert not np.array_equal(a, b)
            info = _inspect(pair)
            observations.append((pair, info))
            tail_marked = _tail_marked(info, a.shape)
            offset_varied = any(len(set(off)) > 1 for off in info['offsets'])
            marker_cross = bool(info['crossings'] & (info['routes'][0] | info['routes'][1]))
            if section is train:
                training_tail |= tail_marked
                training_offset |= offset_varied
                training_marker_cross |= marker_cross
            else:
                assert not tail_marked or training_tail, 'partial endpoint absent from training'
                assert not offset_varied or training_offset, 'offset variation absent from training'
                assert not marker_cross or training_marker_cross, 'marker crossing absent from training'
            if section is train and info['crossings']:
                extra = set(int(v) for v in np.unique(b)) - set(int(v) for v in np.unique(a))
                assert len(extra) == 1
                crossing_colors |= extra
    assert len(crossing_colors) == 1, 'crossing color must be taught consistently'
    train_shapes = {(info['rr'][0][1] - info['rr'][0][0],
                     info['cc'][0][1] - info['cc'][0][0]) for _, info in observations[:len(train)]}
    assert len(train_shapes) >= 2, 'training must demonstrate different compartment scales'
    crossing = next(iter(crossing_colors))
    for pair, info in observations:
        a, b = pair['input'], pair['output']
        assert crossing not in set(int(v) for v in np.unique(a))
        expected = a.copy()
        route_a, route_b = info['coverage']
        for r, c in route_a | route_b:
            color = crossing if (r, c) in route_a and (r, c) in route_b else (
                info['colors'][0] if (r, c) in route_a else info['colors'][1])
            lo_r, hi_r = info['rr'][r]
            lo_c, hi_c = info['cc'][c]
            expected[lo_r:hi_r, lo_c:hi_c] = color
        assert np.array_equal(expected, b), 'route rule violation'
        assert np.all(b[info['rlines'], :] == info['divider'])
        assert np.all(b[:, info['clines']] == info['divider'])


def _tail_marked(info, shape):
    row_tail = info['rr'][-1][1] == shape[0] and info['rr'][-1][1] - info['rr'][-1][0] < info['rr'][0][1] - info['rr'][0][0]
    col_tail = info['cc'][-1][1] == shape[1] and info['cc'][-1][1] - info['cc'][-1][0] < info['cc'][0][1] - info['cc'][0][0]
    return any((row_tail and r == len(info['rr']) - 1) or
               (col_tail and c == len(info['cc']) - 1)
               for route in info['routes'] for r, c in route)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nr, nt = _counts(seed, rng, num_train, num_test)
    mode = seed % 4
    crossing = int(rng.integers(10))
    available = [v for v in range(10) if v != crossing]
    fixed_palette = bool(rng.integers(2))
    common = tuple(int(v) for v in rng.choice(available, 4, replace=False))
    tail_enabled = bool(rng.integers(3))
    offset_enabled = bool(rng.integers(3))
    training_scales = []
    puzzle = {'train': [], 'test': []}
    seen = set()
    for section, count in (('train', nr), ('test', nt)):
        for index in range(count):
            scale = _scale(rng, seed, section, index, training_scales)
            if section == 'train':
                training_scales.append(scale)
                if index == 0:
                    kind = ('branch', 'L', 'multi', 'marker_cross')[mode]
                elif index == 1:
                    kind = ('disjoint', 'multi', 'marker_cross', 'branch')[mode]
                else:
                    options = (('cross', 'disjoint', 'branch', 'multi'),
                               ('cross', 'disjoint', 'L', 'branch', 'multi', 'marker_cross'),
                               ('cross', 'disjoint', 'L', 'branch', 'long', 'multi', 'marker_cross'),
                               ('cross', 'disjoint', 'L', 'branch', 'multi', 'marker_cross'))[mode]
                    kind = options[int(rng.integers(len(options)))]
            else:
                choices = (('L', 'cross', 'disjoint', 'branch'),
                           ('L', 'multi', 'cross', 'disjoint'),
                           ('multi', 'long', 'cross', 'marker_cross', 'disjoint'),
                           ('cross', 'branch', 'L', 'marker_cross', 'disjoint'))[mode]
                kind = choices[0] if index == 0 else choices[int(rng.integers(len(choices)))]
            force_tail = tail_enabled and ((section == 'train' and index == 1) or
                                           (section == 'test' and index == 0))
            mismatch = offset_enabled and ((section == 'train' and index == 0) or
                                            (section == 'test' and index == 0) or
                                            int(rng.integers(3)) == 0)
            roles = common if fixed_palette else tuple(int(v) for v in rng.choice(available, 4, replace=False))
            for _ in range(32):
                row_sizes, col_sizes, tail_axis = _geometry(rng, *scale, force_tail)
                rows = len(row_sizes) - (tail_axis != 'row' and row_sizes[-1] < scale[0])
                cols = len(col_sizes) - (tail_axis != 'col' and col_sizes[-1] < scale[1])
                routes = _markers(rng, rows, cols, kind, tail_axis)
                pair = _draw(rng, row_sizes, col_sizes, routes, (*roles, crossing), mismatch)
                key = pair['input'].shape, pair['input'].tobytes()
                if key not in seen and _fingerprint(pair['input'], pair['output']) not in _SOURCE:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise AssertionError('could not construct distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for seed, puzzle in enumerate(puzzles):
        again = generate_puzzle(seed)
        assert all(np.array_equal(a[key], b[key]) for part in ('train', 'test')
                   for a, b in zip(puzzle[part], again[part]) for key in ('input', 'output'))
        validate_generated_puzzle(puzzle)
    assert len({p['train'][0]['input'].tobytes() for p in puzzles}) == 42
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    for nr, nt in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(927, nr, nt)
        assert nr is None or len(p['train']) == nr
        assert nt is None or len(p['test']) == nt
    for nr, nt in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(2, nr, nt)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    all_info = [[_inspect(pair) for part in ('train', 'test') for pair in p[part]] for p in puzzles]
    assert len({pair['input'].shape for p in puzzles for part in p.values() for pair in part}) >= 30
    assert any(len(info['crossings']) >= 2 for infos in all_info for info in infos)
    assert any(info['crossings'] & (info['routes'][0] | info['routes'][1])
               for infos in all_info for info in infos)
    assert any(len(cover) >= 12 for infos in all_info for info in infos for cover in info['coverage'])
    assert any(len(set(off)) > 1 for infos in all_info for info in infos for off in info['offsets'])
    palettes = [{(i['background'], i['divider'], tuple(i['colors'])) for i in infos} for infos in all_info]
    assert any(len(p) > 1 for p in palettes) and any(len(p) == 1 for p in palettes)
    assert all(len({(i['rr'][0][1], i['cc'][0][1]) for i in infos[:len(p['train'])]}) >= 2
               for p, infos in zip(puzzles, all_info))
    assert any(any(_tail_marked(i, pair['input'].shape)
                   for pair, i in zip(p[part], infos[:len(p[part])] if part == 'train' else infos[len(p['train']):]))
               for p, infos in zip(puzzles, all_info) for part in ('train', 'test'))
    assert any(any(i['rr'][0][1] != i['cc'][0][1] for i in infos) for infos in all_info)
    assert any(any(len(i['rr']) >= 8 and len(i['cc']) >= 8 for i in infos) for infos in all_info)
    assert any(any(pair['input'].shape in ((22, 22), (24, 24), (26, 26), (28, 28))
                   for part in ('train', 'test') for pair in p[part]) for p in puzzles)
    assert any((4, 4) in {(i['rr'][0][1], i['cc'][0][1]) for i in infos[:len(p['train'])]}
               for p, infos in zip(puzzles, all_info))
    assert any((4, 4) in {(i['rr'][0][1], i['cc'][0][1]) for i in infos[len(p['train']):]}
               and (4, 4) not in {(i['rr'][0][1], i['cc'][0][1]) for i in infos[:len(p['train'])]}
               for p, infos in zip(puzzles, all_info))
