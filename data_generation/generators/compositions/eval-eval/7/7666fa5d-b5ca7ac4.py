"""Rigid parcels dock in a diagonal inlet inferred from finite guide strokes.

All conventions are visible in the input. Only standard library and NumPy are
used. ``solve`` never reads demonstrations, random state, or generated labels.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    """Four-connected components; guides are isolated in this connectivity."""
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _profile(o, slope):
    rr, cc = np.indices((o['h'], o['w']))
    rr, cc = rr.ravel()+o['r'], cc.ravel()+o['c']
    q, t = cc-slope*rr, rr+slope*cc
    o = dict(o, rr=rr, cc=cc, q=q, t=t)
    o['profile'] = {int(v): (int(q[t == v].min()), int(q[t == v].max()))
                    for v in np.unique(t)}
    o['midq'] = int(q.min()+q.max())
    return o


def _decode(value):
    g = _grid(value)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg]*2 <= g.size:
        raise ValueError("background must occupy more than half the input")
    objects, guide_cells = [], []
    for cells in _components(g != bg):
        if len(cells) == 1:
            guide_cells.extend(cells)
            continue
        rs, cs = zip(*cells)
        r, c, h, w = min(rs), min(cs), max(rs)-min(rs)+1, max(cs)-min(cs)+1
        if h < 3 or w < 3 or h*w != len(cells):
            raise ValueError("input parcels must be isolated complete rectangles")
        patch = g[r:r+h, c:c+w].copy()
        f = int(patch[0, 0])
        if not (np.all(patch[0] == f) and np.all(patch[-1] == f)
                and np.all(patch[:, 0] == f) and np.all(patch[:, -1] == f)
                and np.all(patch[1:-1, 1:-1] != f)):
            raise ValueError("a parcel needs a uniform frame and contrasting interior")
        objects.append(dict(r=r, c=c, h=h, w=w, frame=f, patch=patch))
    if len(guide_cells) < 4 or not objects:
        raise ValueError("need guides and at least one parcel")
    candidates = []
    for s in (-1, 1):
        lines = {}
        for r, c in guide_cells:
            lines.setdefault(c-s*r, []).append(r)
        strokes = []
        valid = True
        for q, rs in lines.items():
            rs.sort()
            if (len(rs) < 2 or rs != list(range(rs[0], rs[-1]+1))
                    or len({int(g[r, q+s*r]) for r in rs}) != 1):
                valid = False
                break
            strokes.append(dict(q=q, lo=2*rs[0]+s*q, hi=2*rs[-1]+s*q,
                                color=int(g[rs[0], q+s*rs[0]])))
        if valid and len(strokes) >= 2:
            strokes.sort(key=lambda z: z['q'])
            candidates.append((s, strokes))
    if len(candidates) != 1:
        raise ValueError("guides must have a unique parallel diagonal slope")
    slope, strokes = candidates[0]
    if any(b['q']-a['q'] < 3 for a, b in zip(strokes, strokes[1:])):
        raise ValueError("guide lines need transverse separation of at least three")
    low, high = strokes[0]['color'], strokes[-1]['color']
    if low == high or any(o['frame'] not in (low, high) for o in objects):
        raise ValueError("extreme guides must identify two distinct frame instructions")
    objects = [_profile(o, slope) for o in objects]
    info = dict(grid=g, bg=bg, slope=slope, strokes=strokes, objects=objects,
                low=low, high=high, guide_cells=guide_cells)
    for frame in (low, high):
        ids = _ordered(info, frame)
        for rank, i in enumerate(ids):
            for j in ids[:rank]:
                a, b = objects[j]['profile'], objects[i]['profile']
                for t in a.keys() & b.keys():
                    if not (a[t][1] < b[t][0] if frame == low else b[t][1] < a[t][0]):
                        raise ValueError('source parcels have ambiguous destination-side order')
    field = _field(info)
    if any(not np.all(field[o['rr'], o['cc']]) for o in objects):
        raise ValueError("each input parcel must fit entirely inside the inferred inlet")
    return info


def _field(info, mode='full'):
    rr, cc = np.indices(info['grid'].shape)
    q, t = cc-info['slope']*rr, rr+info['slope']*cc
    strokes = info['strokes']
    field = np.zeros(q.shape, dtype=bool)
    if mode == 'canvas':
        field[:] = True
    elif mode == 'infinite':
        field = (q > strokes[0]['q']) & (q < strokes[-1]['q'])
    else:
        pairs = [(i, j) for i in range(len(strokes)) for j in range(i+1, len(strokes))]
        if mode == 'consecutive':
            pairs = list(zip(range(len(strokes)-1), range(1, len(strokes))))
        for i, j in pairs:
            a, b = strokes[i], strokes[j]
            m = ((t >= max(a['lo'], b['lo'])) & (t <= min(a['hi'], b['hi']))
                 & (q > a['q']) & (q < b['q']))
            if mode == 'centers':
                m &= ((t-a['q']) % 2 == 0) & ((t-b['q']) % 2 == 0)
            field |= m
    for r, c in info['guide_cells']:
        field[r, c] = False
    return field


def _ordered(info, frame):
    low = frame == info['low']
    return sorted([i for i, o in enumerate(info['objects']) if o['frame'] == frame],
                  key=lambda i: ((1 if low else -1)*info['objects'][i]['midq'],
                                 min(info['objects'][i]['profile']), i))


def _dock(info, mode='full', stable=True):
    """Input-only solver: enumerate rigid translations in each direction."""
    field = _field(info, mode)
    h, w = field.shape
    s = info['slope']
    shifts = np.zeros(len(info['objects']), dtype=int)
    for frame in (info['low'], info['high']):
        low = frame == info['low']
        settled = []
        for i in _ordered(info, frame):
            o = info['objects'][i]
            local_field = field
            if mode == 'center_slice':
                center = (min(o['profile'])+max(o['profile']))/2
                active = [z['q'] for z in info['strokes'] if z['lo'] <= center <= z['hi']]
                if len(active) < 2:
                    raise ValueError('unbracketed parcel center')
                gr, gc = np.indices(field.shape)
                qq = gc-s*gr
                local_field = (qq > min(active)) & (qq < max(active))
                for r, c in info['guide_cells']:
                    local_field[r, c] = False
            ns = np.arange(-29, 30)
            rr = o['rr'][None, :] - s*ns[:, None]
            cc = o['cc'][None, :] + ns[:, None]
            inside = np.all((rr >= 0) & (rr < h) & (cc >= 0) & (cc < w), axis=1)
            valid = inside & np.all(local_field[np.clip(rr, 0, h-1), np.clip(cc, 0, w-1)], axis=1)
            for j in settled:
                p, n = info['objects'][j], shifts[j]
                shared = o['profile'].keys() & p['profile'].keys()
                for t in shared:
                    a, b = o['profile'][t]
                    x, y = p['profile'][t]
                    x, y = x+2*n, y+2*n
                    if stable:
                        valid &= (a+2*ns > y) if low else (b+2*ns < x)
                    else:
                        valid &= (b+2*ns < x) | (a+2*ns > y)
            choices = ns[valid]
            if choices.size == 0:
                raise ValueError("inlet lacks capacity for this stable directional group")
            shifts[i] = int(choices[0] if low else choices[-1])
            settled.append(i)
    return shifts


def _draw(info, shifts, check=True):
    g = np.full_like(info['grid'], info['bg'])
    occupied = np.zeros(g.shape, dtype=bool)
    for r, c in info['guide_cells']:
        g[r, c] = info['grid'][r, c]
        occupied[r, c] = True
    for o, n in zip(info['objects'], shifts):
        r, c = o['r']-info['slope']*int(n), o['c']+int(n)
        if check and (r < 0 or c < 0 or r+o['h'] > g.shape[0] or c+o['w'] > g.shape[1]
                      or np.any(occupied[r:r+o['h'], c:c+o['w']])):
            raise ValueError("final parcels or guides collide")
        g[r:r+o['h'], c:c+o['w']] = o['patch']
        occupied[r:r+o['h'], c:c+o['w']] = True
    return g


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    return _draw(info, _dock(info))


def _analytic_shifts(info):
    """Separate scalar constraint check, not candidate enumeration or _field."""
    s, g = info['slope'], info['grid']
    shifts = [0]*len(info['objects'])
    for frame in (info['low'], info['high']):
        settled = []
        low = frame == info['low']
        for i in _ordered(info, frame):
            o = info['objects'][i]
            lo, hi, forbidden = -29, 29, set()
            for r, c, q, t in zip(o['rr'], o['cc'], o['q'], o['t']):
                active = [z for z in info['strokes'] if z['lo'] <= t <= z['hi']]
                if len(active) < 2:
                    raise ValueError("parcel extends beyond a bracketed interval")
                lo = max(lo, (active[0]['q']-int(q))//2+1, -int(c))
                hi = min(hi, (active[-1]['q']-int(q)-1)//2, g.shape[1]-1-int(c))
                if s == 1:
                    lo, hi = max(lo, int(r)-g.shape[0]+1), min(hi, int(r))
                else:
                    lo, hi = max(lo, -int(r)), min(hi, g.shape[0]-1-int(r))
                for z in active:
                    if (z['q']-int(q)) % 2 == 0:
                        forbidden.add((z['q']-int(q))//2)
            for j in settled:
                p, n = info['objects'][j], shifts[j]
                for t in o['profile'].keys() & p['profile'].keys():
                    if low:
                        lo = max(lo, (p['profile'][t][1]+2*n-o['profile'][t][0])//2+1)
                    else:
                        hi = min(hi, (p['profile'][t][0]+2*n-o['profile'][t][1]-1)//2)
            possibilities = [n for n in range(lo, hi+1) if n not in forbidden]
            if not possibilities:
                raise ValueError("no feasible extremal placement")
            shifts[i] = possibilities[0] if low else possibilities[-1]
            settled.append(i)
    return shifts


def _alternative_difference(info, actual, mode='full', stable=True):
    try:
        return not np.array_equal(_draw(info, _dock(info, mode, stable)), actual)
    except ValueError:
        # A shortcut with no legal placement cannot explain the displayed output.
        return True


def _features(pair):
    info = _decode(pair['input'])
    shifts = _analytic_shifts(info)
    objects, strokes = info['objects'], info['strokes']
    guide_mask = np.zeros(info['grid'].shape, dtype=bool)
    for r, c in info['guide_cells']:
        guide_mask[r, c] = True
    crosses_guide = False
    for o, shift in zip(objects, shifts):
        ns = np.arange(min(0, shift), max(0, shift)+1)
        rr = o['rr'][None, :]-info['slope']*ns[:, None]
        cc = o['cc'][None, :]+ns[:, None]
        crosses_guide |= bool(np.any(guide_mask[rr, cc]))
    return dict(shape=list(info['grid'].shape), slope=info['slope'],
                guides=len(strokes), parcels=len(objects),
                footprints=sorted({(o['h'], o['w']) for o in objects}),
                lengths=[(z['hi']-z['lo'])//2+1 for z in strokes],
                gaps=[b['q']-a['q'] for a, b in zip(strokes, strokes[1:])],
                travels=list(map(abs, shifts)), background=info['bg'],
                frames=(info['low'], info['high']),
                classes=[sum(o['frame'] == f for o in objects) for f in (info['low'], info['high'])],
                patterned=any(len(np.unique(o['patch'][1:-1, 1:-1])) > 1 for o in objects),
                guide_crossing=crosses_guide,
                ablate_a=_alternative_difference(info, pair['output'], 'canvas'),
                ablate_b=not np.array_equal(pair['input'], pair['output']),
                endpoints=_alternative_difference(info, pair['output'], 'infinite'),
                bottleneck=_alternative_difference(info, pair['output'], 'center_slice'),
                skip=_alternative_difference(info, pair['output'], 'consecutive'),
                parity=_alternative_difference(info, pair['output'], 'centers'),
                stable=_alternative_difference(info, pair['output'], stable=False))


def _stroke_pixels(q, lo, hi):
    return [((t-q)//2, (t+q)//2) for t in range(lo, hi+1, 2)]


def _sample_guides(rng, style, colors, wide=False):
    low, high, neutral = colors
    if style == 'plain':
        q0, q1 = -2*int(rng.integers(5, 9)), 2*int(rng.integers(5, 9))
        length = int(rng.integers(7, 13))
        specs = [(q0, -length+(length % 2), length-(length % 2), low),
                 (q1, -length+(length % 2), length-(length % 2), high)]
    elif style in ('stepped', 'chain'):
        # Longer inner guides remain active beyond the finite outer guides.
        inner = 2*int(rng.integers(4, 7))
        outer = inner+2*int(rng.integers(4, 7) if wide else rng.integers(2, 4))
        length = 2*int(rng.integers(4, 7))
        specs = [(-outer, -int(rng.integers(1, 4))*2, int(rng.integers(1, 4))*2, low),
                 (-inner, -length, length, neutral),
                 (inner, -length, length, neutral),
                 (outer, -int(rng.integers(1, 4))*2, int(rng.integers(1, 4))*2, high)]
        if rng.random() < .55:
            t0 = int(rng.choice([-1, 1]))*(length-2)
            specs.append((0, t0, t0+2, neutral))
    elif style == 'skip':
        outer = 2*int(rng.integers(6, 10))
        length = 2*int(rng.integers(4, 7))
        t0 = int(rng.choice([-1, 1]))*(length-2)
        specs = [(-outer, -length, length, low), (0, t0, t0+2, neutral),
                 (outer, -length, length, high)]
    else:
        n = int(rng.integers(3, 7))
        qs = [0]
        for _ in range(n-1):
            qs.append(qs[-1]+int(rng.integers(3, 8)))
        shift = qs[-1]//2
        specs = []
        for i, q in enumerate(qs):
            q -= shift
            lo = int(rng.integers(-11, 0))
            lo -= (lo-q) % 2
            hi = lo+2*int(rng.integers(2, 11))
            specs.append((q, lo, hi, low if i == 0 else high if i == n-1 else neutral))
    return sorted(specs)


def _blank_inlet(rng, specs, bg):
    cells = [(r, c, color) for q, lo, hi, color in specs for r, c in _stroke_pixels(q, lo, hi)]
    r0, c0 = min(r for r, c, z in cells), min(c for r, c, z in cells)
    span_h, span_w = max(r for r, c, z in cells)-r0+1, max(c for r, c, z in cells)-c0+1
    if max(span_h, span_w) > 30:
        raise ValueError("guide arrangement exceeds the canvas")
    eh, ew = int(rng.integers(0, min(5, 30-span_h)+1)), int(rng.integers(0, min(5, 30-span_w)+1))
    top, left = int(rng.integers(eh+1)), int(rng.integers(ew+1))
    g = np.full((span_h+eh, span_w+ew), bg, dtype=np.int8)
    strokes = []
    for q, lo, hi, color in specs:
        qp, tp = q+left-top-c0+r0, lo+top+left-r0-c0
        strokes.append(dict(q=qp, lo=tp, hi=tp+hi-lo, color=color))
    guide_cells = []
    for r, c, color in cells:
        r, c = r-r0+top, c-c0+left
        g[r, c] = color
        guide_cells.append((r, c))
    return dict(grid=g, bg=bg, slope=1, strokes=strokes, guide_cells=guide_cells,
                low=specs[0][3], high=specs[-1][3], objects=[])


def _place_parcels(rng, info, colors, purpose):
    g = info['grid'].copy()
    field = _field(info)
    occupied = g != info['bg']
    count = (2 if purpose == 'teach' else int(rng.integers(3, 7)) if purpose == 'witness'
             else int(rng.integers(2, 8)) if purpose == 'transfer' else int(rng.integers(1, 7)))
    shared = int(rng.choice(colors))
    common_size = int(rng.integers(3, 5))
    bias = float(rng.choice([0, .2, .5, .8, 1]))
    for i in range(count):
        if purpose == 'teach':
            h = w = common_size
        else:
            upper = 7 if purpose == 'transfer' else 6
            h, w = int(rng.integers(3, upper)), int(rng.integers(3, upper))
            if purpose == 'transfer' and i == 0:
                if rng.integers(2):
                    h = 6
                else:
                    w = 6
            if purpose == 'witness' and i < 3:
                h = w = 3
        if h > g.shape[0] or w > g.shape[1]:
            raise ValueError('parcel exceeds canvas')
        windows = np.lib.stride_tricks.sliding_window_view(field, (h, w))
        clear = np.lib.stride_tricks.sliding_window_view(np.pad(occupied, 1), (h+2, w+2))
        possible = np.argwhere(windows.all(axis=(-2, -1)) & ~clear.any(axis=(-2, -1)))
        if len(possible) == 0:
            if i < (2 if purpose == 'teach' else 3 if purpose == 'witness' else 1):
                raise ValueError("not enough clear initial parcel positions")
            break
        r, c = map(int, possible[int(rng.integers(len(possible)))])
        frame = info['low'] if (i == 0 if purpose == 'teach' else rng.random() < bias) else info['high']
        if purpose == 'witness' and i < 3:
            frame = info['low']
        patch = np.full((h, w), frame, dtype=np.int8)
        if purpose == 'teach':
            patch[1:-1, 1:-1] = shared
        else:
            patch[1:-1, 1:-1] = int(rng.choice(colors))
            if patch[1:-1, 1:-1].size > 1 and rng.random() < .55:
                patch[1:-1, 1:-1] = rng.choice(colors, size=(h-2, w-2))
        g[r:r+h, c:c+w] = patch
        occupied[r:r+h, c:c+w] = True
    return g


def _make_pair(rng, palette, purpose, seen, reflected):
    for attempt in range(64 if purpose == 'witness' else 100):
        low, high, neutral = palette[:3]
        bg = int(rng.choice(palette[3:5]))
        if purpose == 'teach':
            style = 'plain'
        elif purpose == 'witness':
            style = str(rng.choice(['stepped', 'skip', 'staggered'], p=[.65, .20, .15]))
        elif purpose == 'transfer':
            style = str(rng.choice(['stepped', 'staggered'], p=[.5, .5]))
        else:
            style = str(rng.choice(['plain', 'skip', 'stepped', 'staggered']))
        try:
            info = _blank_inlet(rng, _sample_guides(rng, style, (low, high, neutral), purpose == 'transfer'), bg)
            payloads = palette[5:] if purpose == 'transfer' else palette[5:9]
            g = _place_parcels(rng, info, payloads, purpose)
            if reflected & 1:
                g = g[:, ::-1].copy()
            if reflected & 2:
                g = g[::-1].copy()
            if _key(g) in seen:
                continue
            decoded = _decode(g)
            shifts = _analytic_shifts(decoded)
            out = _draw(decoded, shifts)
            if purpose == 'teach':
                if len(decoded['objects']) != 2 or any(n == 0 for n in shifts):
                    continue
            if purpose == 'witness':
                features = _features(dict(input=g, output=out))
                if not all(features[k] for k in ('endpoints', 'bottleneck', 'skip', 'stable', 'guide_crossing')):
                    continue
                if len(features['footprints']) < 2 or not features['patterned']:
                    continue
            if purpose == 'transfer':
                features = _features(dict(input=g, output=out))
                if not all(features[k] for k in ('ablate_a', 'ablate_b', 'endpoints')):
                    continue
            return dict(input=g, output=out)
        except ValueError:
            continue
    return _constructed_witness(rng, palette, seen, reflected, purpose)


def _constructed_witness(rng, palette, seen, reflected, purpose='witness'):
    """A feasible inlet/chain with all the required teaching distinctions.

    Unlike a latent rule exception, this fallback follows exactly the same
    docking rule. Geometry and color still come entirely from visible pixels.
    At least sixteen different canvas shapes guarantee a fresh input after at
    most thirteen earlier examples, irrespective of appearance choices.
    """
    low, high, neutral, bg = palette[:4]
    inner = int(rng.choice([10, 12]))
    outer = inner+10
    specs = [(-outer, -4, 4, low), (-inner, -12, 14, neutral),
             (0, 12, 14, neutral), (inner, -12, 14, neutral), (outer, -4, 4, high)]
    guide_cells = [(r, c, color) for q, lo, hi, color in specs
                   for r, c in _stroke_pixels(q, lo, hi)]
    r0, c0 = min(r for r, c, z in guide_cells), min(c for r, c, z in guide_cells)
    span_h = max(r for r, c, z in guide_cells)-r0+1
    span_w = max(c for r, c, z in guide_cells)-c0+1
    variants_w = 31-span_w
    for variant in rng.permutation((31-span_h)*variants_w):
        h, w = span_h+int(variant)//variants_w, span_w+int(variant) % variants_w
        g = np.full((h, w), bg, dtype=np.int8)
        for r, c, color in guide_cells:
            g[r-r0, c-c0] = color
        k = (inner-10)//2
        fourth = (2, 5, 3, 6, high) if purpose == 'transfer' else (7+k, -9-k, 3, 3 if purpose == 'teach' else 4, high)
        for r, c, oh, ow, f in [(0, -6, 3, 3, low), (-1, -1, 3, 3, low),
                                (-2, 4, 3, 3, low), fourth]:
            patch = np.full((oh, ow), f, dtype=np.int8)
            patch[1:-1, 1:-1] = palette[5] if purpose == 'teach' else int(rng.choice(palette[5:9]))
            if ow >= 4:
                patch[1, 1:3] = palette[5:7]
            if purpose == 'transfer' and f == high:
                patch[1, -2] = palette[9]
            g[r-r0:r-r0+oh, c-c0:c-c0+ow] = patch
        if reflected & 1:
            g = g[:, ::-1].copy()
        if reflected & 2:
            g = g[::-1].copy()
        if _key(g) in seen:
            continue
        info = _decode(g)
        pair = dict(input=g, output=_draw(info, _analytic_shifts(info)))
        f = _features(pair)
        assert all(f[k] for k in ('endpoints', 'bottleneck', 'skip', 'stable', 'parity'))
        assert f['ablate_a'] and f['ablate_b']
        assert purpose == 'transfer' or f['guide_crossing']
        assert purpose == 'teach' or f['patterned']
        return pair
    raise AssertionError('finite fresh-shape construction exhausted')


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    reflected = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            purpose = ('transfer' if split == 'test' else 'teach' if i == 0
                       else 'witness' if i == 1 else 'free')
            pair = _make_pair(rng, palette, purpose, seen, reflected)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, conventions, records = set(), set(), {'train': [], 'test': []}
    moved_frames, shared_payload = set(), False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert np.array_equal(np.bincount(inp.ravel(), minlength=10),
                                  np.bincount(out.ravel(), minlength=10)), 'rigid relocation must conserve every color count'
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert all(out[r, c] == inp[r, c] for r, c in info['guide_cells']), 'guides must remain fixed'
            conventions.add((info['slope'], info['low'], info['high']))
            shifts = _analytic_shifts(info)
            assert np.array_equal(_draw(info, shifts), out), 'wrong rigid docking or payload'
            assert np.array_equal(solve(inp), out), 'enumerated solver disagreement'
            for o, n in zip(info['objects'], shifts):
                assert n <= 0 if o['frame'] == info['low'] else n >= 0
                if split == 'train' and n:
                    moved_frames.add(o['frame'])
            if split == 'train':
                for a in info['objects']:
                    for b in info['objects']:
                        shared_payload |= (a['frame'] != b['frame'] and a['patch'].shape == b['patch'].shape
                                           and np.array_equal(a['patch'][1:-1, 1:-1], b['patch'][1:-1, 1:-1]))
            records[split].append(_features(pair))
    assert len(conventions) == 1, 'conventions must stay consistent within a puzzle'
    assert moved_frames == set(next(iter(conventions))[1:]) and shared_payload
    assert all(any(r[k] for r in records['train']) for k in ('ablate_a', 'ablate_b'))
    for feature in ('endpoints', 'bottleneck', 'skip', 'parity', 'stable', 'guide_crossing'):
        assert not any(r[feature] for r in records['test']) or any(r[feature] for r in records['train']), f'missing {feature} training witness'
    if any(r['patterned'] for r in records['test']):
        assert any(r['patterned'] for r in records['train']), 'missing patterned-payload demonstration'
    if any(len(r['footprints']) > 1 for r in records['test']):
        assert any(len(r['footprints']) > 1 for r in records['train']), 'missing mixed-footprint demonstration'
    assert all(all(r[k] for k in ('ablate_a', 'ablate_b', 'endpoints'))
               for r in records['test']), 'held-out examples must exercise both parents and finite shores'


def self_test():
    """Bounded 42-seed corpus, geometric fixture, mutation and API checks.

    These checks are self-authored and do not claim independent ARC approval.
    The returned measurements describe actual solved grids, not RNG settings.
    """
    # Hand calculation in an unshifted diagonal coordinate system. For inner
    # shores +/-10 the three low parcels settle at q centers -6,-4,-2; the
    # high 3x4 parcel goes from -15.5 to +16.5, crossing the guide lines.
    specs = [(-20, -4, 4, 0), (-10, -12, 14, 2), (0, 12, 14, 2),
             (10, -12, 14, 2), (20, -4, 4, 1)]
    fixture = np.full((25, 25), 3, dtype=np.int8)
    for q, lo, hi, color in specs:
        for r, c in _stroke_pixels(q, lo, hi):
            fixture[r+12, c+12] = color
    source_objects = [(0, -6, 3, 3, 0, 0), (-1, -1, 3, 3, 0, -2),
                      (-2, 4, 3, 3, 0, -4), (7, -9, 3, 4, 1, 16)]
    expected = fixture.copy()
    for r, c, h, w, frame, shift in source_objects:
        patch = np.full((h, w), frame, dtype=np.int8)
        patch[1:-1, 1:-1] = 5
        if w == 4:
            patch[1, 2] = 6
        fixture[r+12:r+12+h, c+12:c+12+w] = patch
        expected[r+12-shift:r+12-shift+h, c+12+shift:c+12+shift+w] = patch
    assert np.array_equal(solve(fixture), expected)
    assert np.array_equal(_draw(_decode(fixture), _analytic_shifts(_decode(fixture))), expected)
    for horizontal, vertical in ((True, False), (False, True), (True, True)):
        a, b = fixture.copy(), expected.copy()
        if horizontal:
            a, b = a[:, ::-1], b[:, ::-1]
        if vertical:
            a, b = a[::-1], b[::-1]
        assert np.array_equal(solve(a), b)
    record = _features(dict(input=fixture, output=expected))
    assert all(record[k] for k in ('endpoints', 'bottleneck', 'skip', 'parity',
                                  'stable', 'guide_crossing', 'ablate_a', 'ablate_b'))
    # An opposite frame color can be carried as data without selecting direction.
    reused, reused_expected = fixture.copy(), expected.copy()
    reused[13, 7] = reused_expected[13, 7] = 1
    assert np.array_equal(solve(reused), reused_expected)
    fingerprints, shapes, normalized, rows = set(), set(), set(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(tuple((_key(x['input']), _key(x['output'])) for s in p for x in p[s]))
        for split in p:
            for pair in p[split]:
                rows.append(_features(pair))
                shapes.add(pair['input'].shape)
                role_map = {}
                normalized.add((pair['input'].shape,
                                tuple(role_map.setdefault(int(c), len(role_map)) for c in pair['input'].ravel())))
        for pair in p['test']:
            assert any(max(o['h'], o['w']) == 6 for o in _decode(pair['input'])['objects'])
            f = _features(pair)
            assert all(f[k] for k in ('ablate_a', 'ablate_b', 'endpoints'))
    assert len(fingerprints) == 42
    assert len(shapes) >= 20 and len(normalized) >= 140
    assert {r['slope'] for r in rows} == {-1, 1}
    assert len({tuple(r['frames']) for r in rows}) >= 25
    assert len({r['background'] for r in rows}) >= 8
    assert {2, 3, 4, 5} <= {r['guides'] for r in rows}
    assert {1, 2, 3, 4} <= {r['parcels'] for r in rows}
    assert len({hw for r in rows for hw in r['footprints']}) >= 12
    assert any(min(r['classes']) == 0 for r in rows)
    assert any(0 in r['travels'] and max(r['travels']) > 0 for r in rows)
    assert any(max(r['shape']) == 30 for r in rows)
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(104729+a*10+b, a, b)
        assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (3, False), (3, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    q = generate_puzzle(104833, 10, 4)
    assert tuple((_key(x['input']), _key(x['output'])) for s in p for x in p[s]) == tuple(
        (_key(x['input']), _key(x['output'])) for s in q for x in q[s])
    # The last legal loop uses seed 104833 and counts 10/4, so this also checks
    # whole-puzzle reproducibility, including the bounded fallback path.
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in q[s]] for s in q}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    # Correct labels can still be an inadequate demonstration set. Two plain
    # inlets do not teach finite stepped shores or the stable chain distinction.
    teaching = generate_puzzle(0, 2, 1)
    simple = teaching['train'][0]
    bg = _decode(simple['input'])['bg']
    replacement = next(c for c in range(10) if not np.any(simple['input'] == c))
    recolored = {k: np.where(a == bg, replacement, a).astype(np.int8)
                 for k, a in simple.items()}
    inadequate = dict(train=[simple, recolored], test=[teaching['train'][1]])
    try:
        validate_generated_puzzle(inadequate)
    except AssertionError as e:
        assert 'training witness' in str(e)
    else:
        raise AssertionError('inadequate teaching accepted')
    return dict(seeds=42, pairs=len(rows), shapes=len(shapes),
                normalized_structures=len(normalized),
                guide_counts=sorted({r['guides'] for r in rows}),
                parcel_counts=sorted({r['parcels'] for r in rows}),
                footprints=sorted({hw for r in rows for hw in r['footprints']}),
                witnesses={k: sum(r[k] for r in rows) for k in
                           ('endpoints', 'bottleneck', 'skip', 'parity', 'stable', 'guide_crossing', 'ablate_a', 'ablate_b')})
