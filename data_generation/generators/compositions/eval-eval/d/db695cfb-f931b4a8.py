"""Diagonal shutter weave: input-only geometry gates a two-scale fabric overlay.

Only standard-library modules and NumPy are used. The two cards in every input
encode roles; no color inference from training, seed, or saved labels is needed.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid of size 1..30 and colors 0..9")
    return a.astype(np.int64, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _normal(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _segment(a, b):
    # Endpoints are sorted by row; no horizontal or vertical pairing is used.
    slope = 1 if b[1] > a[1] else -1
    return tuple((a[0] + k, a[1] + slope * k)
                 for k in range(b[0] - a[0] + 1)), slope


def _geometry(board, primary, secondary):
    points = [tuple(map(int, p)) for p in np.argwhere(board == primary)]
    dots = [tuple(map(int, p)) for p in np.argwhere(board == secondary)]
    degree = {p: 0 for p in points}
    segments, slopes = [], []
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            if a[0] != b[0] and abs(a[0] - b[0]) == abs(a[1] - b[1]):
                cells, slope = _segment(a, b)
                if len(cells) < 3:
                    raise ValueError("A bridge must contain at least three cells")
                segments.append(cells)
                slopes.append(slope)
                degree[a] += 1
                degree[b] += 1
    if not segments or any(n > 1 for n in degree.values()):
        raise ValueError("Endpoint partners must be unique, with at least one pair")
    launches, counts, inactive = [], [0] * len(segments), []
    for p in dots:
        owners = [i for i, s in enumerate(segments) if p in s[1:-1]]
        if len(owners) > 1:
            raise ValueError("A secondary marker cannot belong to two bridges")
        if owners:
            i = owners[0]
            launches.append((p[0], p[1], -slopes[i]))
            counts[i] += 1
        else:
            inactive.append(p)
    return dict(segments=segments, slopes=slopes, launches=launches,
                triggers=counts, inactive=inactive,
                unpaired=[p for p, d in degree.items() if d == 0])


def _decode(input_grid):
    a = _grid(input_grid)
    primary = int(a[0, 0])
    first_width = 0
    while first_width < a.shape[1] and a[0, first_width] == primary:
        first_width += 1
    if first_width + 1 >= a.shape[1]:
        raise ValueError("Missing the two framed swatches")
    background = int(a[0, first_width])
    start = first_width + 1
    secondary = int(a[0, start])
    second_width = 0
    while start + second_width < a.shape[1] and a[0, start + second_width] == secondary:
        second_width += 1
    if len({primary, secondary, background}) != 3:
        raise ValueError("Frame and transparency colors must be distinct")
    heights = []
    for column, ink in ((0, primary), (start, secondary)):
        h = 0
        while h < a.shape[0] and a[h, column] == ink:
            h += 1
        heights.append(h)
    ch, cw = heights[0] - 2, first_width - 2
    sh, sw = heights[1] - 2, second_width - 2
    if not all(2 <= d <= 6 for d in (ch, cw, sh, sw)):
        raise ValueError("Swatch sides must be 2..6")
    header = max(heights)
    if header + 3 >= a.shape[0]:
        raise ValueError("Working rectangle must have at least three rows")
    carrier = a[1:ch + 1, 1:cw + 1]
    stencil = a[1:sh + 1, start + 1:start + sw + 1]
    if np.any(np.isin(carrier, [primary, secondary, background])):
        raise ValueError("Carrier cells must be payload colors")
    if np.any(np.isin(stencil, [primary, secondary])):
        raise ValueError("Control colors cannot occur in the stencil payload")
    expected = np.full((header + 1, a.shape[1]), background, dtype=int)
    expected[:ch + 2, :cw + 2] = primary
    expected[1:ch + 1, 1:cw + 1] = carrier
    expected[:sh + 2, start:start + sw + 2] = secondary
    expected[1:sh + 1, start + 1:start + sw + 1] = stencil
    if not np.array_equal(a[:header + 1], expected):
        raise ValueError("Malformed card frames or separator")
    board = a[header + 1:]
    h, w = board.shape
    if h % ch or w % cw:
        raise ValueError("Carrier replication factors must be integers")
    if not np.all(np.isin(board, [primary, secondary, background])):
        raise ValueError("Working rectangle may contain only control markers and background")
    geom = _geometry(board, primary, secondary)
    return dict(primary=primary, secondary=secondary, background=background,
                carrier=carrier, stencil=stencil, board=board, geometry=geom)


def _expanded_launches(d, mode):
    g = d['geometry']
    launches = list(g['launches'])
    if mode == 'same_slope':
        return [(r, c, -s) for r, c, s in launches]
    if mode in ('activate_all', 'infinite_activation'):
        for r, c in g['inactive']:
            collinear = [s for cells, s in zip(g['segments'], g['slopes'])
                         if c - cells[0][1] == s * (r - cells[0][0])]
            if mode == 'infinite_activation' and not collinear:
                continue
            if collinear:
                slope = collinear[0]
            else:
                i = min(range(len(g['segments'])), key=lambda j: min(
                    abs(r - x) + abs(c - y) for x, y in g['segments'][j]))
                slope = g['slopes'][i]
            launches.append((r, c, -slope))
    return launches


def _render(d, mode='correct'):
    """Vectorized rule, plus explicit counterfactuals used for teaching checks."""
    board, carrier, stencil = d['board'], d['carrier'], d['stencil']
    bg, p, s = d['background'], d['primary'], d['secondary']
    h, w = board.shape
    ch, cw = carrier.shape
    sh, sw = stencil.shape
    rr, cc = np.indices(board.shape)
    br, bc = rr * ch // h, cc * cw // w
    if mode == 'carrier_tile_rows':
        br = rr % ch
    if mode == 'carrier_tile_columns':
        bc = cc % cw
    base = carrier[br, bc]
    tr, tc = rr % sh, cc % sw
    if mode == 'scale_stencil':
        tr, tc = rr * sh // h, cc * sw // w
    if mode == 'phase_rows':
        tr = (tr + 1) % sh
    if mode == 'phase_columns':
        tc = (tc + 1) % sw
    tiled = stencil[tr, tc]
    gate = np.zeros(board.shape, dtype=bool)
    seam = np.zeros(board.shape, dtype=bool)
    launches = _expanded_launches(d, mode)
    for r, c, slope in launches:
        distance = rr - slope * cc - (r - slope * c)
        below = distance < 0 if mode == 'above_gate' else distance >= 0
        if mode == 'union_gate':
            gate |= below
        else:
            gate ^= below
        seam |= distance == 0
    if mode == 'always_overlay':
        gate[:] = True
    if mode == 'no_overlay':
        gate[:] = False
    opaque = gate if mode == 'opaque_stencil' else gate & (tiled != bg)
    out = np.where(opaque, tiled, base)
    if mode == 'ignore_fabrics':
        out[:] = bg
    if mode != 'no_bridges':
        for cells, slope in zip(d['geometry']['segments'], d['geometry']['slopes']):
            if mode == 'full_bridges':
                r, c = cells[0]
                out[cc - c == slope * (rr - r)] = p
            else:
                for r, c in cells:
                    out[r, c] = p
    out[seam] = s
    if mode == 'primary_above_seams':
        for cells in d['geometry']['segments']:
            for r, c in cells:
                out[r, c] = p
    # Original controls are annotations above the two fabrics and added lines.
    original = board != bg
    if mode != 'ignore_original':
        out[original] = board[original]
    return out.astype(np.int64)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read cards, pair original endpoints, and construct the shutter weave."""
    return _render(_decode(input_grid))


_ALTERNATIVES = (
    'activate_all', 'infinite_activation', 'same_slope', 'union_gate',
    'above_gate', 'always_overlay', 'no_overlay', 'no_bridges', 'full_bridges',
    'carrier_tile_rows', 'carrier_tile_columns', 'scale_stencil',
    'opaque_stencil', 'phase_rows', 'phase_columns', 'ignore_fabrics',
    'primary_above_seams', 'ignore_original',
)
_REQUIRED = frozenset(_ALTERNATIVES) | {
    'active_positive', 'active_negative', 'multiple_triggers', 'untriggered',
    'unpaired', 'beyond_endpoint', 'mixed_active', 'visible_carrier', 'overwrite',
}


def _evidence(pair, decoded=None):
    d = _decode(pair['input']) if decoded is None else decoded
    out = np.asarray(pair['output'])
    cover = {m for m in _ALTERNATIVES if not np.array_equal(out, _render(d, m))}
    g = d['geometry']
    active_slopes = {-s for _, _, s in g['launches']}
    if 1 in active_slopes:
        cover.add('active_positive')
    if -1 in active_slopes:
        cover.add('active_negative')
    if active_slopes == {-1, 1}:
        cover.add('mixed_active')
    if max(g['triggers']) >= 2:
        cover.add('multiple_triggers')
    if 0 in g['triggers']:
        cover.add('untriggered')
    if g['unpaired']:
        cover.add('unpaired')
    if any(c - cells[0][1] == slope * (r - cells[0][0])
           for r, c in g['inactive'] for cells, slope in zip(g['segments'], g['slopes'])):
        cover.add('beyond_endpoint')
    # Reconstruct the annotation-free layer to certify both texture behaviors.
    rr, cc = np.indices(d['board'].shape)
    ch, cw = d['carrier'].shape
    sh, sw = d['stencil'].shape
    base = d['carrier'][rr * ch // rr.shape[0], cc * cw // rr.shape[1]]
    tiled = d['stencil'][rr % sh, cc % sw]
    gate = np.zeros(rr.shape, dtype=bool)
    guide = d['board'] != d['background']
    for r, c, slope in g['launches']:
        dist = rr - slope * cc - (r - slope * c)
        gate ^= dist >= 0
        guide |= dist == 0
    for cells in g['segments']:
        for r, c in cells:
            guide[r, c] = True
    if np.any(~guide & (~gate | (tiled == d['background'])) & (out == base)):
        cover.add('visible_carrier')
    if np.any(~guide & gate & (tiled != d['background']) & (tiled != base)):
        cover.add('overwrite')
    return cover


def _configuration(d):
    ch, cw = d['carrier'].shape
    return (*d['carrier'].shape, *d['stencil'].shape,
            d['board'].shape[0] // ch, d['board'].shape[1] // cw)


def _pack(board, carrier, stencil, roles):
    bg, p, s, _ = roles
    ch, cw = carrier.shape
    sh, sw = stencil.shape
    header = max(ch, sh) + 2
    start = cw + 3
    a = np.full((header + 1 + board.shape[0], board.shape[1]), bg, dtype=np.int64)
    a[:ch + 2, :cw + 2] = p
    a[1:ch + 1, 1:cw + 1] = carrier
    a[:sh + 2, start:start + sw + 2] = s
    a[1:sh + 1, start + 1:start + sw + 1] = stencil
    a[header + 1:] = board
    return dict(input=a, output=solve(a))


def _orient(a, rng):
    # Swatch geometry is unchanged by these reflections and half-turns.
    if rng.random() < .5:
        a = a[::-1]
    if rng.random() < .5:
        a = a[:, ::-1]
    return a.copy()


def _fabric(shape, inks, rng):
    h, w = shape
    rr, cc = np.indices(shape)
    k = int(rng.integers(1, min(5, len(inks)) + 1))
    colors = rng.choice(inks, k, replace=False)
    kind = int(rng.integers(7))
    if kind == 0 or k == 1:
        labels = np.zeros(shape, dtype=int)
    elif kind in (1, 2):
        axis = int(rng.integers(2))
        n = shape[axis]
        if kind == 1:
            line = np.arange(n) % k
        else:
            cuts = sorted(rng.choice(np.arange(1, n), int(rng.integers(1, n)), replace=False))
            line = rng.integers(k, size=len(cuts) + 1)[np.searchsorted(cuts, np.arange(n), side='right')]
        labels = np.broadcast_to(line[:, None] if axis == 0 else line[None, :], shape)
    elif kind == 3:
        labels = (rr // int(rng.integers(1, h)) + cc // int(rng.integers(1, w))) % k
    elif kind == 4:
        ar, ac = int(rng.integers(1, h)), int(rng.integers(1, w))
        blocks = rng.integers(k, size=(2, 2))
        labels = blocks[(rr >= ar).astype(int), (cc >= ac).astype(int)]
    elif kind == 5:
        labels = np.zeros(shape, dtype=int)
        for ink in range(1, k):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            labels[abs(rr - r) + abs(cc - c) <= int(rng.integers(1, max(h, w)))] = ink
    else:
        labels = rng.integers(k, size=shape)
    return _orient(colors[labels], rng)


def _stencil(shape, inks, bg, rng):
    h, w = shape
    rr, cc = np.indices(shape)
    kind = int(rng.integers(9))
    if kind == 0:
        mask = np.zeros(shape, dtype=bool)
    elif kind == 1:
        mask = np.ones(shape, dtype=bool)
    elif kind == 2:
        mask = (rr + cc + int(rng.integers(2))) % 2 == 0
    elif kind == 3:
        mask = np.minimum.reduce([rr, cc, h - 1 - rr, w - 1 - cc]) < int(rng.integers(1, max(2, min(h, w) // 2 + 1)))
    elif kind == 4:
        mask = (rr if rng.random() < .5 else cc) % int(rng.integers(2, max(h, w) + 1)) == 0
    elif kind == 5:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        mask = abs(rr - r) + abs(cc - c) <= int(rng.integers(0, min(h, w)))
    else:
        mask = rng.random(shape) < float(rng.uniform(.12, .88))
    mask = _orient(mask, rng)
    tile = np.full(shape, bg, dtype=int)
    if mask.any():
        k = int(rng.integers(1, min(5, len(inks), int(mask.sum())) + 1))
        colors = rng.choice(inks, k, replace=False)
        tile[mask] = rng.choice(colors, int(mask.sum()))
    return tile


def _place_geometry(shape, count, roles, rng, test=False):
    h, w = shape
    bg, p, s, _ = roles
    segments, endpoints = [], []
    for _ in range(count):
        for attempt in range(80):
            length = int(rng.integers(3, min(h, w) + 1))
            slope = int(rng.choice((-1, 1)))
            r = int(rng.integers(h - length + 1))
            c = int(rng.integers(w - length + 1)) + (length - 1 if slope == -1 else 0)
            cells, _ = _segment((r, c), (r + length - 1, c + slope * (length - 1)))
            ends = cells[0], cells[-1]
            if any(abs(a[0] - b[0]) == abs(a[1] - b[1]) for a in ends for b in endpoints):
                continue
            segments.append(cells)
            endpoints.extend(ends)
            break
        else:
            return None
    board = np.full(shape, bg, dtype=int)
    for point in endpoints:
        board[point] = p
    segment_union = set().union(*(set(cells) for cells in segments))
    for cells in segments:
        eligible = [a for a in cells[1:-1] if sum(a in b for b in segments) == 1]
        # Leave a visible gap in every bridge; activation counts remain free.
        cap = min(4, len(eligible), len(cells) - 3)
        k = int(rng.integers(cap + 1)) if cap else 0
        for index in rng.permutation(len(eligible))[:k]:
            board[eligible[int(index)]] = s
    primary_count = int(rng.choice((0, 0, 1, 1, 2, 3)))
    for index in rng.permutation(h * w):
        if primary_count == 0:
            break
        point = int(index) // w, int(index) % w
        if point in segment_union or any(abs(point[0] - b[0]) == abs(point[1] - b[1]) for b in endpoints):
            continue
        board[point] = p
        endpoints.append(point)
        primary_count -= 1
    free = [a for a in np.ndindex(shape) if a not in segment_union and board[a] == bg]
    n = int(rng.choice((0, 0, 1, 1, 2, 3, 4, 5)))
    # A dot beyond an endpoint is especially useful evidence of finite activation.
    beyond = [a for a in free if any(a[1] - cells[0][1] ==
              (1 if cells[-1][1] > cells[0][1] else -1) * (a[0] - cells[0][0]) for cells in segments)]
    if n and beyond and rng.random() < .75:
        point = beyond[int(rng.integers(len(beyond)))]
        board[point] = s
        free.remove(point)
        n -= 1
    for index in rng.permutation(len(free))[:n]:
        board[free[int(index)]] = s
    geom = _geometry(board, p, s)
    if test and (len(geom['launches']) < 3 or len({x[2] for x in geom['launches']}) != 2):
        return None
    return board


def _sample_pair(roles, rng, test=False):
    bg, _, _, inks = roles
    for _ in range(120):
        ch, cw, sh, sw = map(int, rng.integers(2, 7, size=4))
        header = max(ch, sh) + 2
        heights = [h for h in range(ch, 30 - header, ch) if h >= 6]
        widths = [w for w in range(cw, 31, cw) if w >= cw + sw + 5]
        h, w = int(rng.choice(heights)), int(rng.choice(widths))
        count = int(rng.choice((2, 3, 3, 4, 5))) if test else int(rng.choice((1, 1, 2, 2, 3, 4, 5)))
        board = _place_geometry((h, w), count, roles, rng, test)
        if board is None:
            continue
        return _pack(board, _fabric((ch, cw), inks, rng),
                     _stencil((sh, sw), inks, bg, rng), roles)
    return None


def _fallback_training(roles):
    """Constructive teaching witnesses used only if the random pool lacks evidence."""
    bg, p, s, inks = roles
    a, b, c = map(int, inks[:3])
    board1 = np.full((12, 18), bg, dtype=int)
    for point in ((2, 2), (8, 8), (1, 7), (4, 10), (0, 15)):
        board1[point] = p
    for point in ((4, 4), (6, 6), (10, 10), (0, 0)):
        board1[point] = s
    first = _pack(board1, np.array([[a, b, a], [c, a, b]]),
                  np.array([[bg, c], [b, bg]]), roles)
    board2 = np.full((15, 16), bg, dtype=int)
    for point in ((1, 12), (8, 5), (4, 0), (10, 6), (0, 0)):
        board2[point] = p
    for point in ((3, 10), (6, 7), (7, 3), (0, 13), (14, 15)):
        board2[point] = s
    second = _pack(board2, np.array([[b, c], [a, b], [c, a]]),
                   np.array([[c, bg], [bg, a], [b, bg]]), roles)
    return [first, second]


def _training_diverse(pairs):
    ds = [_decode(p['input']) for p in pairs]
    return (len({_configuration(d) for d in ds}) >= 2
            and len({_normal(d['carrier']) for d in ds}) >= 2
            and len({_normal(d['stencil']) for d in ds}) >= 2)


def _fallback_extra(roles, used, forbidden_configurations=()):
    """Finite bank of 32 valid sizes, exceeding all 14-pair requests.

    It keeps the mixed-slope three-seam geometry of the second constructive
    witness, extending only blank board space. Test configurations remain novel.
    """
    original = _decode(_fallback_training(roles)[1]['input'])
    for h in (15, 18, 21, 24):
        for w in (16, 18, 20, 22, 24, 26, 28, 30):
            configuration = (3, 2, 3, 2, h // 3, w // 2)
            if configuration in forbidden_configurations:
                continue
            board = np.full((h, w), roles[0], dtype=int)
            board[:15, :16] = original['board']
            pair = _pack(board, original['carrier'], original['stencil'], roles)
            if _key(pair['input']) not in used:
                return pair
    raise AssertionError("The finite fallback bank exceeds maximum puzzle capacity")


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    roles = (palette[0], palette[1], palette[2], palette[3:])
    candidates, covers, seen = [], [], set()
    train = None
    # Choose an informative pair of demonstrations, not a fixed lesson schedule.
    for _ in range(128):
        pair = _sample_pair(roles, rng)
        if pair is None or _key(pair['input']) in seen:
            continue
        cover = _evidence(pair)
        for j in rng.permutation(len(candidates)):
            j = int(j)
            if _REQUIRED <= cover | covers[j] and _training_diverse([pair, candidates[j]]):
                train = [pair, candidates[j]]
                break
        if train is not None:
            break
        seen.add(_key(pair['input']))
        candidates.append(pair)
        covers.append(cover)
    if train is None:
        train = _fallback_training(roles)
    puzzle = dict(train=train, test=[])
    seen = {_key(p['input']) for p in train}
    for pair in candidates:
        if len(puzzle['train']) == num_train:
            break
        if _key(pair['input']) not in seen:
            puzzle['train'].append(pair)
            seen.add(_key(pair['input']))
    for _ in range(256):
        if len(puzzle['train']) == num_train:
            break
        pair = _sample_pair(roles, rng)
        if pair is not None and _key(pair['input']) not in seen:
            puzzle['train'].append(pair)
            seen.add(_key(pair['input']))
    for _ in range(num_train - len(puzzle['train'])):
        pair = _fallback_extra(roles, seen)
        puzzle['train'].append(pair)
        seen.add(_key(pair['input']))
    configurations = {_configuration(_decode(p['input'])) for p in puzzle['train']}
    for _ in range(num_test):
        for attempt in range(256):
            pair = _sample_pair(roles, rng, test=True)
            if (pair is not None and _key(pair['input']) not in seen
                    and _configuration(_decode(pair['input'])) not in configurations):
                puzzle['test'].append(pair)
                seen.add(_key(pair['input']))
                break
        else:
            pair = _fallback_extra(roles, seen, configurations)
            puzzle['test'].append(pair)
            seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, role_keys, evidence, train_configs = set(), set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "Duplicate input"
            seen.add(_key(inp))
            d = _decode(inp)
            role_keys.add((d['background'], d['primary'], d['secondary']))
            assert out.shape == d['board'].shape
            assert np.array_equal(solve(inp), out), "Incorrect output"
            original = d['board'] != d['background']
            assert np.array_equal(out[original], d['board'][original]), "Original markers changed"
            if split == 'train':
                evidence |= _evidence(pair, d)
                train_configs.add(_configuration(d))
            else:
                assert _configuration(d) not in train_configs, "Test scale combination was demonstrated"
                assert len(d['geometry']['launches']) >= 3
                assert {x[2] for x in d['geometry']['launches']} == {-1, 1}
    assert len(role_keys) == 1, "Puzzle conventions changed"
    assert _REQUIRED <= evidence, "Missing teaching evidence: " + str(_REQUIRED - evidence)
    assert _training_diverse(puzzle['train'])


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded checks of semantics, counterfactuals, legal counts, and 42-seed diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    rows = {s: [_decode(pair['input']) for p in puzzles for pair in p[s]]
            for s in ('train', 'test')}
    metrics = {}
    for split, ds in rows.items():
        metrics[split] = dict(
            pairs=len(ds), shapes=len({d['board'].shape for d in ds}),
            configurations=len({_configuration(d) for d in ds}),
            carriers=len({_normal(d['carrier']) for d in ds}),
            stencils=len({_normal(d['stencil']) for d in ds}),
            segment_counts=sorted({len(d['geometry']['segments']) for d in ds}),
            trigger_counts=sorted({k for d in ds for k in d['geometry']['triggers']}),
            backgrounds=sorted({d['background'] for d in ds}))
        assert metrics[split]['shapes'] >= 15
        assert metrics[split]['configurations'] >= 25
        assert metrics[split]['carriers'] >= 25
        assert metrics[split]['stencils'] >= 25
        assert len(metrics[split]['backgrounds']) >= 8
        assert len(metrics[split]['segment_counts']) >= 3
        assert {0, 1, 2} <= set(metrics[split]['trigger_counts'])
    assert any(np.all(d['stencil'] == d['background']) for d in rows['train'])
    assert any(np.all(d['stencil'] == d['background']) for d in rows['test'])
    assert any(0 in d['carrier'] and d['background'] != 0 for d in rows['train'] + rows['test'])
    assert any(len({s for s in d['geometry']['slopes']}) == 1 and
               len(d['geometry']['launches']) >= 2 for d in rows['train'])
    assert any(set(a) & set(b) for d in rows['train'] + rows['test']
               for i, a in enumerate(d['geometry']['segments'])
               for b in d['geometry']['segments'][i + 1:])
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    # Check the deterministic fallback independently of whether sampling needs it.
    roles = (0, 1, 2, [3, 4, 5, 6, 7, 8, 9])
    fallback = _fallback_training(roles)
    assert _REQUIRED <= set().union(*(_evidence(p) for p in fallback))
    assert _training_diverse(fallback)
    # Hand-calculated cell probes: two /-seams have constants r+c=8 and 12.
    # Between them the shutter is open; below both it is closed.
    out = fallback[0]['output']
    assert out[5, 4] == 4  # Open + opaque stencil cell (1,0) overwrites carrier 3.
    assert out[5, 6] == 4  # Same native stencil phase despite another carrier block.
    assert out[8, 7] == 3  # Below both seams: enlarged carrier, not stencil 5.
    assert out[3, 2] == 3  # Above both: enlarged carrier.
    assert out[4, 4] == 2  # Original trigger preserved.
    assert out[3, 3] == 1  # Finite bridge gap completed.
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(9128 + train * 10 + test, train, test)
            assert len(p['train']) == train and len(p['test']) == test
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Palette permutation must be read from input, including nonblack background.
    for pair in fallback:
        perm = np.array([8, 5, 0, 2, 9, 1, 6, 3, 7, 4])
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    # Small valid members beyond the normal six-row sampling floor. No dots of
    # the secondary role: the answer is explicitly replicated carrier + bridge.
    board = np.zeros((4, 10), dtype=int)
    for point in ((0, 0), (2, 2), (3, 8)):
        board[point] = 1
    carrier = np.array([[3, 4], [5, 6]])
    stencil = np.array([[0, 7], [8, 0]])
    query = _pack(board, carrier, stencil, roles)
    expected = np.repeat(np.repeat(carrier, 2, axis=0), 5, axis=1)
    for point in ((0, 0), (1, 1), (2, 2), (3, 8)):
        expected[point] = 1
    assert np.array_equal(query['output'], expected)
    # One center trigger in a length-three bridge; /-seam r+c=2 reaches edges.
    board[1, 1] = 2
    query = _pack(board, carrier, stencil, roles)
    out = query['output']
    assert (out[0, 3], out[0, 4], out[3, 0], out[1, 0]) == (7, 3, 8, 3)
    assert out[0, 2] == out[2, 0] == 2 and out[2, 2] == 1
    # A secondary marker at the two-slope crossing has no unique launch slope.
    crossing = np.zeros((12, 12), dtype=int)
    for point in ((1, 1), (8, 8), (2, 10), (9, 3)):
        crossing[point] = 1
    _pack(crossing, carrier, stencil, roles)  # Unmarked bridge crossings are valid.
    crossing[6, 6] = 2
    try:
        _pack(crossing, carrier, stencil, roles)
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous trigger accepted")
    used = set()
    for _ in range(14):
        pair = _fallback_extra(roles, used)
        used.add(_key(pair['input']))
        assert len(_decode(pair['input'])['geometry']['launches']) == 3
    assert len(used) == 14
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted label accepted")
    return dict(seeds=42, distinct_puzzles=42, diversity=metrics,
                semantic_check='generator-owned checks and explicit cell probes')


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
