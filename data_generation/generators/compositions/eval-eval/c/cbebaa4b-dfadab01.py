"""Input-defined, registered XOR stencils commissioned by isolated code dots.

Only standard-library modules and NumPy are used. Generation has finite retry
budgets; no arrays, traces, or per-seed samples are written by this module.
"""
from __future__ import annotations

import hashlib
import numpy as np


N8 = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in N8:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _normal(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("a grid must be an integer 1..30 rectangle in colors 0..9")
    return a.astype(np.int8, copy=False)


def _analyze(value):
    """Read every convention from one input; no generation metadata is used."""
    inp = _grid(value)
    bg = int(np.bincount(inp.ravel(), minlength=10).argmax())
    assert np.count_nonzero(inp == bg) > inp.size // 2, "background must be a majority"
    objects = _components(map(tuple, np.argwhere(inp != bg).tolist()))
    palettes = [{int(inp[q]) for q in obj} for obj in objects]
    roots = [i for i, colors in enumerate(palettes) if len(colors) == 3]
    assert 2 <= len(roots) <= 3, "two or three labeled stencils are required"
    common = set.intersection(*(palettes[i] for i in roots))
    assert len(common) == 1, "registration hue must be shared by all labeled stencils"
    pin = common.pop()
    book, catalog, inks = {}, set(), set()
    for index in roots:
        obj = objects[index]
        t, l, _, _ = _box(obj)
        origin = (t, l)
        assert origin in obj, "the label occupies the actual top-left corner"
        code = int(inp[origin])
        assert code != pin and sum(inp[q] == code for q in obj) == 1
        ink_set = palettes[index] - {pin, code}
        assert len(ink_set) == 1
        ink = ink_set.pop()
        assert code not in book and ink not in inks
        pins = {q for q in obj if inp[q] == pin}
        p = {q for q in obj if inp[q] == ink}
        candidates = [j for j, colors in enumerate(palettes) if colors == {pin, ink}]
        assert len(candidates) == 1, "one unlabeled partner per ink"
        partner_index = candidates[0]
        partner = objects[partner_index]
        other_pins = {q for q in partner if inp[q] == pin}
        q_native = {q for q in partner if inp[q] == ink}
        # Lexicographic minima fix a translation; the complete sets must agree.
        a, b = min(pins), min(other_pins)
        delta = a[0] - b[0], a[1] - b[1]
        assert _shift(other_pins, delta) == pins, "all registration pins must coincide"
        q = _shift(q_native, delta)
        p = _shift(p, (-t, -l))
        q = _shift(q, (-t, -l))
        local_pins = _shift(pins, (-t, -l))
        assert len(pins) in (2, 3) and len(p) >= 4 and len(q) >= 4
        assert not (p | q) & local_pins
        assert len(p & q) >= 2 and len(p - q) >= 2 and len(q - p) >= 2
        stamp = p ^ q
        assert len(_components(stamp)) == 1
        qt, ql, _, _ = _box(q | local_pins)
        wrong = p ^ _shift(q, (-qt, -ql))
        assert wrong != stamp, "bounding-box alignment must differ from pin alignment"
        book[code] = dict(ink=ink, stamp=stamp, p=p, q=q, pins=local_pins,
                          origin=origin, root=index, partner=partner_index,
                          wrong=wrong, partner_offset=(qt, ql))
        inks.add(ink)
        catalog.update((index, partner_index))
    codes = set(book)
    assert codes.isdisjoint(inks | {pin, bg}) and inks.isdisjoint({pin, bg})
    assert len(catalog) == 2 * len(book)
    live, bare, attached = [], [], []
    for i, obj in enumerate(objects):
        if i in catalog:
            continue
        colors = palettes[i]
        assert pin not in colors and colors <= codes | inks
        points = [q for q in obj if inp[q] in codes]
        if len(obj) == 1 and len(points) == 1:
            r, c = points[0]
            live.append((r, c, int(inp[r, c])))
        elif not points:
            assert len(colors) == 1 and colors <= inks
            bare.append(obj)
        else:
            assert len(points) == 1, "nonlibrary material has at most one code dot"
            point = points[0]
            code = int(inp[point])
            ink = book[code]['ink']
            assert colors == {code, ink}
            body = obj - {point}
            last = max(body)
            assert point == (last[0] + 1, last[1] + 1), "decoy dot is lower-right attached"
            attached.append((point, code))
    assert codes <= {code for _, _, code in live}, "each class has a live request"
    return inp, bg, pin, book, live, bare, attached


def _render(shape, bg, book, live, field='stamp'):
    out = np.full(shape, bg, dtype=np.int8)
    for r, c, code in live:
        info = book[code]
        for a, b in info[field]:
            x, y = r + a, c + b
            if 0 <= x < shape[0] and 0 <= y < shape[1]:
                out[x, y] = info['ink']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive registered masks, classify input dots, and print their requests."""
    inp, bg, _, book, live, _, _ = _analyze(input_grid)
    return _render(inp.shape, bg, book, live)


def _stencils(rng, pin_count, forbidden_key=None):
    """Sample two independent perforated plates in a shared registration frame."""
    for _ in range(2000):
        ph, pw, qh, qw = map(int, rng.integers(3, 6, size=4))
        dr, dc = map(int, rng.integers(-2, 3, size=2))
        if dr == dc == 0:
            continue
        pd = {(r, c) for r in range(ph) for c in range(pw)} - {(0, 0)}
        qd = {(r, c) for r in range(dr, dr + qh) for c in range(dc, dc + qw)}
        choices = sorted(pd & qd)
        if len(choices) < pin_count + 3:
            continue
        selected = rng.choice(len(choices), pin_count, replace=False)
        pins = {choices[int(i)] for i in selected}
        key = _normal(pins)
        if key == forbidden_key:
            continue
        p = {z for z in sorted(pd - pins) if rng.random() < .76}
        q = {z for z in sorted(qd - pins) if rng.random() < .76}
        if min(len(p), len(q)) < 4:
            continue
        if len(_components(p | pins | {(0, 0)})) != 1 or len(_components(q | pins)) != 1:
            continue
        if len(p & q) < 2 or len(p - q) < 2 or len(q - p) < 2:
            continue
        stamp = p ^ q
        if (len(_components(stamp)) != 1 or max(r for r,c in stamp) < 1
                or max(c for r,c in stamp) < 1):
            continue
        t, l, _, _ = _box(q | pins)
        if p ^ _shift(q, (-t, -l)) == stamp:
            continue
        return dict(p=p, q=q, pins=pins, stamp=stamp, key=key)
    raise RuntimeError("finite stencil sampling budget exhausted")


def _object(mapping, print_mapping=None):
    print_mapping = print_mapping or {}
    t, l, b, r = _box(set(mapping) | set(print_mapping))
    return dict(cells={(a-t, c-l): v for (a, c), v in mapping.items()},
                printed={(a-t, c-l): v for (a, c), v in print_mapping.items()},
                shape=(b-t+1, r-l+1))


def _empty_positions(forbidden, h, w):
    H, W = forbidden.shape
    if h > H or w > W:
        return np.empty((0, 2), dtype=int)
    integral = np.pad(forbidden.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(sums == 0)


def _reserve(forbidden, t, l, h, w):
    H, W = forbidden.shape
    forbidden[max(0,t-1):min(H,t+h+1), max(0,l-1):min(W,l+w+1)] = True


def _pack(rng, objects, clipped, shape, random_mode):
    H, W = shape
    forbidden = np.zeros(shape, dtype=bool)
    placements = []
    c_input, c_print = clipped
    t, l, b, r = _box(set(c_input) | set(c_print))
    _reserve(forbidden, t, l, b-t+1, r-l+1)
    priorities = rng.random(len(objects))
    order = sorted(range(len(objects)), key=lambda i: (-max(objects[i]['shape']),
                   -np.prod(objects[i]['shape']), priorities[i]))
    for i in order:
        obj = objects[i]
        h, w = obj['shape']
        positions = _empty_positions(forbidden, h, w)
        if not len(positions):
            return None
        if random_mode:
            index = int(rng.integers(len(positions)))
        else:
            # Compact corner packing is a bounded backup. Stencil contents
            # always keep their original orientation.
            score = positions[:, 0] * (W + 1) + positions[:, 1]
            index = int(np.argmin(score))
        t, l = map(int, positions[index])
        placements.append((obj, t, l))
        _reserve(forbidden, t, l, h, w)
    return placements


def _clipped_request(rng, shape, code, ink, stamp, side):
    H, W = shape
    t, l, b, r = _box(stamp | {(0, 0)})
    # Put the last extreme row/column just outside. Keep at least two ink cells.
    if (side in ('bottom','corner') and b < 1) or (side in ('right','corner') and r < 1):
        return None
    ar = H - max(1, b) if side in ('bottom', 'corner') else int(rng.integers(-t, H-b))
    ac = W - max(1, r) if side in ('right', 'corner') else int(rng.integers(-l, W-r))
    ar, ac = min(H-1, ar), min(W-1, ac)
    ink_map = {(ar+a, ac+c): ink for a, c in stamp
               if 0 <= ar+a < H and 0 <= ac+c < W}
    if len(ink_map) < 2 or len(ink_map) == len(stamp):
        return None
    return {(ar, ac): code}, ink_map


def _make_pair(rng, params, split, index):
    pin, codes, inks = params['pin'], params['codes'], params['inks']
    bg = params['backgrounds'][index % 2]
    k = len(codes)
    book = {}
    first_key = None
    for j, (code, ink) in enumerate(zip(codes, inks)):
        count = 2 if j < 2 else int(rng.integers(2, 4))
        st = _stencils(rng, count, first_key if j == 1 else None)
        if j == 0:
            first_key = st['key']
        book[code] = dict(st, ink=ink)
    objects = []
    for code, st in book.items():
        ink = st['ink']
        root = {q: ink for q in st['p']}
        root.update({q: pin for q in st['pins']})
        root[(0, 0)] = code
        other = {q: ink for q in st['q']}
        other.update({q: pin for q in st['pins']})
        objects.extend((_object(root), _object(other)))
    # Every class has a full print. Early demonstrations teach one repetition;
    # tests ask for two, in addition to their boundary print.
    extras = 1 if split == 'test' else (0 if index < 2 else int(rng.integers(2)))
    request_codes = codes + [int(rng.choice(codes)) for _ in range(extras)]
    for code in request_codes:
        st = book[code]
        objects.append(_object({(0, 0): code}, {q: st['ink'] for q in st['stamp']}))
    kinds = []
    if split == 'train' and index == 0:
        kinds = ['attached']
    elif split == 'train' and index == 1:
        kinds = ['bare']
    else:
        kinds = [kind for kind in ('bare', 'attached') if rng.random() < .35]
    for kind in kinds:
        code = int(rng.choice(codes))
        st = book[code]
        material = {q: st['ink'] for q in st['stamp']}
        if kind == 'attached':
            r, c = max(material)
            material[(r+1, c+1)] = code
        objects.append(_object(material))
    side = ('bottom' if index == 0 else 'right') if split == 'train' else 'corner'
    for attempt in range(48):
        shape = (30, 30) if attempt >= 24 else tuple(map(int, rng.integers(23, 31, size=2)))
        code = int(rng.choice(codes))
        st = book[code]
        clipped = _clipped_request(rng, shape, code, st['ink'], st['stamp'], side)
        if clipped is None:
            continue
        placements = _pack(rng, objects, clipped, shape, attempt < 12)
        if placements is None:
            continue
        inp = np.full(shape, bg, dtype=np.int8)
        out = np.full(shape, bg, dtype=np.int8)
        for mapping, target in zip(clipped, (inp, out)):
            for q, color in mapping.items():
                target[q] = color
        for obj, t, l in placements:
            for q, color in obj['cells'].items():
                inp[t+q[0], l+q[1]] = color
            for q, color in obj['printed'].items():
                out[t+q[0], l+q[1]] = color
        # Relocate whole packed rectangles, never rotate or reflect their contents.
        # Random packing already supplies free scene positions, so no scene
        # reflection is needed (it would invalidate the top-left label convention).
        if np.count_nonzero(inp == bg) <= inp.size // 2:
            continue
        parsed = _analyze(inp)
        if not np.array_equal(_render(shape, bg, parsed[3], parsed[4]), out):
            raise AssertionError("construction disagrees with input-derived registration")
        return {'input': inp, 'output': out}
    raise RuntimeError("finite rectangle packing budget exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    k = int(rng.integers(2, 4))
    colors = list(map(int, rng.permutation(10)))
    params = dict(pin=colors[0], codes=colors[1:1+k], inks=colors[1+k:1+2*k],
                  backgrounds=colors[1+2*k:3+2*k])
    puzzle, seen = {'train': [], 'test': []}, set()
    first_stamps = None
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            for _ in range(16):
                pair = _make_pair(rng, params, split, i)
                key = (pair['input'].shape, pair['input'].tobytes())
                book = _analyze(pair['input'])[3]
                stamps = {c: tuple(sorted(info['stamp'])) for c, info in book.items()}
                if key in seen or (split == 'train' and i == 1 and
                                   any(stamps[c] == first_stamps[c] for c in stamps)):
                    continue
                if split == 'train' and i == 0:
                    first_stamps = stamps
                break
            else:
                raise RuntimeError('finite distinct-example budget exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _features(pair):
    inp, bg, pin, book, live, bare, attached = _analyze(pair['input'])
    records = []
    for code, info in book.items():
        record = dict(code=code, ink=info['ink'], key=_normal(info['pins']),
                      mask=tuple(sorted(info['stamp'])),
                      master_box=_box(info['p'] | info['pins'] | {(0,0)}),
                      partner_box=_box(info['q'] | info['pins']),
                      offset=info['partner_offset'], common=len(info['p'] & info['q']),
                      exclusive=(len(info['p']-info['q']), len(info['q']-info['p'])))
        records.append(record)
    clipped = []
    full = []
    sides = set()
    for r, c, code in live:
        stamp = book[code]['stamp']
        inside = sum(0 <= r+a < inp.shape[0] and 0 <= c+b < inp.shape[1] for a,b in stamp)
        (full if inside == len(stamp) else clipped).append(code)
        if any(r+a < 0 for a,b in stamp):
            sides.add('top')
        if any(c+b < 0 for a,b in stamp):
            sides.add('left')
        if any(r+a >= inp.shape[0] for a,b in stamp):
            sides.add('bottom')
        if any(c+b >= inp.shape[1] for a,b in stamp):
            sides.add('right')
    return dict(shape=inp.shape, bg=bg, pin=pin, records=records, live=len(live),
                bare=len(bare), attached=len(attached), clipped=clipped, full=full,
                clip_sides=sorted(sides))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, convention, taught, taught_masks = set(), None, set(), {}
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            parsed = _analyze(inp)
            _, bg, pin, book, live, bare, attached = parsed
            roles = (pin, tuple(sorted((c, d['ink']) for c,d in book.items())))
            convention = roles if convention is None else convention
            assert convention == roles, 'code, ink and registration roles are stable within a puzzle'
            expected = _render(inp.shape, bg, book, live)
            assert np.array_equal(expected, out), 'output differs from input-only registration printing'
            assert set(map(int, np.unique(out))) <= {bg} | {d['ink'] for d in book.values()}
            occupied = set()
            for r,c,code in live:
                # Bounding windows, including the reference point, do not overlap.
                t,l,b,z = _box(book[code]['stamp'] | {(0,0)})
                window = {(a,d) for a in range(max(0,r+t),min(inp.shape[0],r+b+1))
                          for d in range(max(0,c+l),min(inp.shape[1],c+z+1))}
                assert not occupied & window
                occupied |= window
            assert not np.array_equal(_render(inp.shape,bg,book,live,'p'), out), 'omitting cancellation must change the answer'
            assert not np.array_equal(_render(inp.shape,bg,book,live,'wrong'), out), 'ignoring pins must change the answer'
            for operation in ('union', 'intersection'):
                alternative = {c:dict(d,stamp=(d['p'] | d['q'] if operation == 'union'
                                             else d['p'] & d['q'])) for c,d in book.items()}
                assert not np.array_equal(_render(inp.shape,bg,alternative,live),out)
            library_origins = [(d['origin'][0],d['origin'][1],c) for c,d in book.items()]
            assert not np.array_equal(_render(inp.shape,bg,book,library_origins),out), 'omitting commissioned stamping must change the answer'
            all_codes = [(int(r),int(c),int(inp[r,c]))
                         for r,c in np.argwhere(np.isin(inp,list(book)))]
            assert not np.array_equal(_render(inp.shape,bg,book,all_codes),out), 'library labels and attached dots are not requests'
            f = _features(pair)
            assert f['clipped'] and set(book) <= set(f['full'])
            two_pin_keys = {rec['key'] for rec in f['records'] if len(rec['key']) == 2}
            assert len(two_pin_keys) >= 2
            features[split].append(f)
            if split == 'train':
                taught.update(('bare',) if bare else ())
                taught.update(('attached',) if attached else ())
                for c,d in book.items():
                    taught_masks.setdefault(c,set()).add(tuple(sorted(d['stamp'])))
            else:
                assert not bare or 'bare' in taught
                assert not attached or 'attached' in taught
                assert set(f['clip_sides']) == {'bottom','right'}
    assert all(len(masks) >= 2 for masks in taught_masks.values()), 'current stencils, not a fixed mask lookup'
    assert len({f['bg'] for f in features['train']}) == 2
    assert {'bottom','right'} <= set().union(*(set(f['clip_sides']) for f in features['train']))
    assert min(f['live'] for f in features['test']) > min(f['live'] for f in features['train'])
    return None


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train','test'):
        for pair in puzzle[split]:
            for role in ('input','output'):
                g = np.asarray(pair[role],dtype=np.uint8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, bounded summary sets, counterfactuals, counts and corruption."""
    shapes, masks, keys, offsets, colors, classes, requests, fingerprints = (set() for _ in range(8))
    pair_count = bare_count = attached_count = 0
    retained = None
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(_fingerprint(p))
        if seed == 0:
            retained = p
        for split in ('train','test'):
            for pair in p[split]:
                f = _features(pair)
                shapes.add(f['shape'])
                colors.add(f['bg'])
                classes.add(len(f['records']))
                requests.add(f['live'])
                bare_count += bool(f['bare'])
                attached_count += bool(f['attached'])
                pair_count += 1
                for rec in f['records']:
                    masks.add(rec['mask'])
                    keys.add(rec['key'])
                    offsets.add(rec['offset'])
    assert len(fingerprints) == 42 and classes == {2,3}
    assert len(shapes) >= 15 and len(masks) >= 200
    assert len(keys) >= 20 and len(offsets) >= 8 and len(colors) >= 8
    assert len(requests) >= 3 and bare_count >= 42 and attached_count >= 42
    assert _fingerprint(retained) == _fingerprint(generate_puzzle(0))
    for nt,nq in ((2,1),(10,1),(10,4)):
        validate_generated_puzzle(generate_puzzle(917,nt,nq))
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in retained[s]] for s in ('train','test')}
    inp, bg, _, book, live, _, _ = _analyze(bad['test'][0]['input'])
    r,c,code = live[0]
    for a,b in book[code]['stamp']:
        if 0 <= r+a < inp.shape[0] and 0 <= c+b < inp.shape[1]:
            bad['test'][0]['output'][r+a,c+b] = bg
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('erasing an entire commissioned print was accepted')
    return dict(seeds=42,pairs=pair_count,distinct_puzzles=len(fingerprints),
                canvas_shapes=len(shapes),registered_masks=len(masks),pin_keys=len(keys),
                registration_offsets=len(offsets),background_colors=len(colors),
                class_counts=sorted(classes),request_counts=sorted(requests),
                pairs_with_bare=bare_count,pairs_with_attached=attached_count)
