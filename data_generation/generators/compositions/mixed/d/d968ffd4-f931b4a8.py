"""Counted canvas, post-resize opposed-pad partition, native stencil overlay.

Only standard library and NumPy; no files, examples, seed, or puzzle state are
read by solve(). Gray (5) is the fixed, newly painted odd-gap separator ink.
"""
from __future__ import annotations

import hashlib
import numpy as np


SEPARATOR = 5
HOST_SEEDS = (0,447868136,824010598,345090075,492655006,416185080,872654919,
              892010790,19459348,40156240,581264180,405490159,687415042,
              414736859,882436739,986985756,951745826,709530827,626868346,
              468754430,530224305,826022853,404680127,280862768,248939498,
              935974202,823965692,36120069,881549316,738138116,342067975,
              503184052,200086197,179045811,899905568,61115758,410098531,
              218878314,264311745,403830225,596373085,702114827)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _normalized(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(x), len(roles)) for x in a.flat)


def _pads(a):
    """Read two solid rectangles and their unique opposing axis from a grid."""
    bg = int(a[0, 0])
    assert bg != SEPARATOR
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)
    colors = [int(x) for x in np.unique(a) if x != bg]
    assert len(colors) == 2 and SEPARATOR not in colors
    boxes = []
    for color in colors:
        points = np.argwhere(a == color)
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0) + 1
        box = tuple(int(x) for x in (r0, r1, c0, c1))
        assert len(points) == (r1-r0)*(c1-c0)
        boxes.append(box)
    candidates = []
    for axis in (0, 1):
        k, t = 2*axis, 2*(1-axis)
        order = sorted(range(2), key=lambda i: boxes[i][k])
        first, last = (boxes[i] for i in order)
        start, stop = first[k+1], last[k]
        if (stop-start >= 2 and
                max(first[t], last[t]) < min(first[t+1], last[t+1])):
            view = a if axis == 0 else a.T
            assert np.all(view[start:stop] == bg)
            candidates.append(dict(axis=axis, start=start, stop=stop,
                                   gap=stop-start, background=bg,
                                   colors=tuple(colors[i] for i in order),
                                   boxes=tuple(boxes[i] for i in order)))
    assert len(candidates) == 1, "opposing pads must define one axis"
    return candidates[0]


def _decode(input_grid):
    a = _grid(input_grid)
    assert a.shape[0] == a.shape[1] and a.shape[0] % 2 == 0
    n = a.shape[0] // 2
    assert 6 <= n <= 15 and not np.any(a == SEPARATOR)
    left, right = a[:n, :n], a[:n, n:]
    carrier, stencil = a[n:, :n], a[n:, n:]
    source = _pads(carrier)
    bg = source['background']
    h, w = int(np.sum(left != bg)), int(np.sum(right != bg))
    assert n <= h <= 30 and n <= w <= 30
    assert h % n == 0 and w % n == 0
    base = np.repeat(np.repeat(carrier, h//n, axis=0), w//n, axis=1)
    info = _pads(base)
    tiled = stencil[np.arange(h)[:, None] % n, np.arange(w)[None, :] % n]
    return dict(n=n, h=h, w=w, left=left, right=right, carrier=carrier,
                stencil=stencil, base=base, tiled=tiled, info=info,
                source=source, background=bg, factors=(h//n, w//n))


def _partition(base, info):
    result = base.copy()
    v = result if info['axis'] == 0 else result.T
    start, stop, half = info['start'], info['stop'], info['gap']//2
    v[start:start+half] = info['colors'][0]
    v[stop-half:stop] = info['colors'][1]
    if info['gap'] % 2:
        v[start+half] = SEPARATOR
    return result


def _overlay(base, stencil, background):
    return np.where(stencil != background, stencil, base).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve one input without demonstrations or generation parameters."""
    d = _decode(input_grid)
    partition = _partition(d['base'], d['info'])
    return _overlay(partition, d['tiled'], d['background'])


def _evidence(input_grid, output):
    """Observable training counterexamples to concrete simpler rules."""
    d = _decode(input_grid)
    bg, i = d['background'], d['info']
    part = _partition(d['base'], i)
    covers = set()

    def compare(name, alternative):
        if not np.array_equal(output, alternative):
            covers.add(name)

    compare('omit_pad_split', _overlay(d['base'], d['tiled'], bg))
    compare('ignore_stencil', part)
    compare('opaque_background', d['tiled'])
    kr, kc = d['factors']
    source_part = _partition(d['carrier'], d['source'])
    early = np.repeat(np.repeat(source_part, kr, axis=0), kc, axis=1)
    compare('fill_before_resize', _overlay(early, d['tiled'], bg))
    compare('tile_carrier', _overlay(np.tile(source_part, (kr, kc)), d['tiled'], bg))
    scaled_stencil = np.repeat(np.repeat(d['stencil'], kr, axis=0), kc, axis=1)
    compare('resize_stencil', _overlay(part, scaled_stencil, bg))
    for axis, name in ((0, 'phase_rows'), (1, 'phase_columns')):
        rr = (np.arange(d['h'])[:, None] + (axis == 0)) % d['n']
        cc = (np.arange(d['w'])[None, :] + (axis == 1)) % d['n']
        compare(name, _overlay(part, d['stencil'][rr, cc], bg))
    clipped = d['base'].copy()
    cv = clipped if i['axis'] == 0 else clipped.T
    pv = part if i['axis'] == 0 else part.T
    t = 2*(1-i['axis'])
    lo = max(b[t] for b in i['boxes'])
    hi = min(b[t+1] for b in i['boxes'])
    cv[i['start']:i['stop'], lo:hi] = pv[i['start']:i['stop'], lo:hi]
    compare('full_cross_section', _overlay(clipped, d['tiled'], bg))
    if i['gap'] % 2:
        center = part == SEPARATOR
        old_midpoint = part.copy()
        old_midpoint[center] = bg
        compare('midpoint_ink', _overlay(old_midpoint, d['tiled'], bg))
        protected = np.asarray(output).copy()
        protected[center] = SEPARATOR
        compare('stencil_over_separator', protected)
    if d['h'] != d['w']:
        covers.add('counter_swap')
    bbox, max_ink, color_count = [], [], []
    for counter in (d['left'], d['right']):
        r, c = np.nonzero(counter != bg)
        bbox.append(int((r.max()-r.min()+1)*(c.max()-c.min()+1)))
        _, frequency = np.unique(counter[counter != bg], return_counts=True)
        max_ink.append(int(frequency.max()))
        color_count.append(len(frequency))
    for name, dims in (('counter_bbox', bbox), ('counter_one_ink', max_ink),
                       ('counter_color_count', color_count)):
        if tuple(dims) != (d['h'], d['w']):
            covers.add(name)
    return covers


REQUIRED_EVIDENCE = frozenset((
    'omit_pad_split', 'ignore_stencil', 'opaque_background', 'fill_before_resize',
    'tile_carrier', 'resize_stencil', 'phase_rows', 'phase_columns',
    'full_cross_section', 'midpoint_ink', 'stencil_over_separator',
    'counter_swap', 'counter_bbox', 'counter_one_ink', 'counter_color_count'))


def _visible(d, require_crossing=False):
    """Keep each pad and each propagation half observable through the stencil."""
    i, bg = d['info'], d['background']
    transparent = d['tiled'] == bg
    v = transparent if i['axis'] == 0 else transparent.T
    half = i['gap']//2
    if not (v[i['start']:i['start']+half].any() and
            v[i['stop']-half:i['stop']].any()):
        return False
    # A visible fill beyond BOTH pad spans rules out footprint-only propagation.
    t = 2*(1-i['axis'])
    outside = np.ones(v.shape[1], dtype=bool)
    for b in i['boxes']:
        outside[b[t]:b[t+1]] = False
    if not v[i['start']:i['start']+half, outside].any():
        return False
    if not v[i['stop']-half:i['stop'], outside].any():
        return False
    for r0, r1, c0, c1 in i['boxes']:
        if not transparent[r0:r1, c0:c1].any():
            return False
    if i['gap'] % 2:
        center = v[i['start']+half]
        if not center.any() or (require_crossing and center.all()):
            return False
    return True


def _counter(n, count, bg, palette, rng):
    """Exact occupancy; position and ink frequency carry no extra meaning."""
    style = int(rng.integers(4))
    if style == 0:
        # A compact row-major patch, freely rotated/reflected/translated on torus.
        width = int(rng.integers((count+n-1)//n, n+1))
        positions = [(j//width, j % width) for j in range(count)]
        mask = np.zeros((n, n), dtype=bool)
        for r, c in positions:
            mask[r, c] = True
    elif style == 1:
        mask = np.zeros((n, n), dtype=bool)
        rows, cols = rng.permutation(n), rng.permutation(n)
        j = np.arange(count)
        mask[rows[j//n], cols[j % n]] = True
    elif style == 2:
        points, frontier = set(), {(int(rng.integers(n)), int(rng.integers(n)))}
        while len(points) < count:
            choices = sorted(frontier)
            r, c = choices[int(rng.integers(len(choices)))]
            frontier.remove((r, c))
            points.add((r, c))
            for q in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if 0 <= q[0] < n and 0 <= q[1] < n and q not in points:
                    frontier.add(q)
        mask = np.zeros((n, n), dtype=bool)
        for r, c in points:
            mask[r, c] = True
    else:
        mask = np.zeros((n, n), dtype=bool)
        mask.flat[rng.choice(n*n, count, replace=False)] = True
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = mask[:, ::-1]
    mask = np.roll(mask, tuple(int(x) for x in rng.integers(n, size=2)), axis=(0,1))
    inks = rng.choice(palette, 1 if rng.random() < .4 else int(rng.integers(2,5)), replace=False)
    values = np.r_[inks, rng.choice(inks, count-len(inks), p=rng.dirichlet(np.ones(len(inks))))]
    rng.shuffle(values)
    result = np.full((n, n), bg, dtype=np.int8)
    result[mask] = values
    return result


def _carrier(n, axis, parity, colors, rng):
    # Enumerate a tiny geometry space; no unbounded rejection or payload cache.
    options = [(g, d0, d1, e0, n-g-d0-d1-e0)
               for g in range(2, n-3) if g % 2 == parity
               for d0 in range(1,4) for d1 in range(1,4) for e0 in range(1,4)
               if 1 <= n-g-d0-d1-e0 <= 3]
    g, d0, d1, e0, _ = options[int(rng.integers(len(options)))]
    s0 = int(rng.integers(1, n-2))
    z0 = int(rng.integers(s0+1, n))
    s1, z1 = s0, z0
    if rng.random() < .45:
        s1 = max(1, min(n-2, s0+int(rng.integers(-1,2))))
        z1 = max(s1+1, min(n-1, z0+int(rng.integers(-1,2))))
        if max(s0,s1) >= min(z0,z1):
            s1, z1 = s0, z0
    bg, first, last = colors
    a = np.full((n, n), bg, dtype=np.int8)
    a[e0:e0+d0, s0:z0] = first
    a[e0+d0+g:e0+d0+g+d1, s1:z1] = last
    return a if axis == 0 else a.T.copy()


def _stencil(n, bg, palette, rng, allow_empty):
    result = np.full((n, n), bg, dtype=np.int8)
    if allow_empty and rng.random() < .12:
        return result
    rr, cc = np.indices((n, n))
    kind = int(rng.integers(7))
    if kind == 0:
        mask = rng.random((n, n)) < rng.uniform(.06,.48)
    elif kind == 1:
        period = int(rng.integers(2, n))
        mask = (rr + int(rng.integers(period))) % period == 0
    elif kind == 2:
        k = int(rng.integers(1,4))
        mask = (rr//k + cc//k + int(rng.integers(2))) % 2 == 0
    elif kind == 3:
        thick = int(rng.integers(1, max(2,n//4)))
        mask = np.minimum.reduce((rr,cc,n-1-rr,n-1-cc)) < thick
    elif kind == 4:
        mask = np.zeros((n,n), dtype=bool)
        for _ in range(int(rng.integers(1,5))):
            r, c = rng.integers(n, size=2)
            mask |= abs(rr-r)+abs(cc-c) <= int(rng.integers(0,max(2,n//3)))
    elif kind == 5:
        r, c = rng.integers(n, size=2)
        mask = (abs(rr-r) <= int(rng.integers(1,max(2,n//3)))) & (abs(cc-c) <= 1)
    else:
        mask = (rr + int(rng.integers(1,4))*cc + int(rng.integers(n))) % n < 2
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = mask[:, ::-1]
    if not mask.any():
        mask.flat[int(rng.integers(n*n))] = True
    if mask.all():
        mask.flat[int(rng.integers(n*n))] = False
    count = int(mask.sum())
    k = 1
    if count > 1 and rng.random() >= .55:
        k = int(rng.integers(2,min(4,count)+1))
    inks = rng.choice(palette, k, replace=False)
    values = np.r_[inks, rng.choice(inks, count-k)]
    rng.shuffle(values)
    result[mask] = values
    return result


def _make_pair(rng, axis, mode, stable_bg, avoid_n=(), avoid_shapes=()):
    for _ in range(128):
        ns = [n for n in range(7 if mode != 'free' else 6,16) if n not in avoid_n]
        n = int(rng.choice(ns))
        maximum = 30//n
        if mode == 'odd':
            axial = int(rng.choice([k for k in range(1,maximum+1) if k % 2]))
            cross = int(rng.choice([k for k in range(1,maximum+1) if k != axial]))
            parity = 1
        elif mode == 'even':
            axial = int(rng.choice([k for k in range(2,maximum+1,2)]))
            cross = int(rng.integers(2,maximum+1))
            parity = 1
        else:
            axial, cross = (int(x) for x in rng.integers(1,maximum+1,size=2))
            parity = 0 if n == 6 else int(rng.integers(2))
        kr, kc = (axial,cross) if axis == 0 else (cross,axial)
        if (n*kr,n*kc) in avoid_shapes:
            continue
        bg = int(rng.choice([c for c in range(10) if c != SEPARATOR])) if stable_bg is None else stable_bg
        palette = [c for c in range(10) if c not in (bg, SEPARATOR)]
        pad_colors = [int(x) for x in rng.choice(palette, 2, replace=False)]
        carrier = _carrier(n, axis, parity, (bg,*pad_colors), rng)
        stencil = _stencil(n, bg, palette, rng, mode == 'free')
        left = _counter(n, n*kr, bg, palette, rng)
        right = _counter(n, n*kc, bg, palette, rng)
        a = np.block([[left,right], [carrier,stencil]])
        d = _decode(a)
        if not _visible(d, require_crossing=mode == 'odd'):
            continue
        return {'input': a, 'output': solve(a)}
    raise RuntimeError('bounded pair construction exhausted')


def _integer(value, low, high, name):
    if isinstance(value, (bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(name+' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name+' out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt = _integer(num_train,2,10,'num_train')
    ns = _integer(num_test,1,4,'num_test')
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    bg = None if rng.random() < .55 else int(rng.choice([c for c in range(10) if c != SEPARATOR]))
    axis = int(rng.integers(2))
    odd = _make_pair(rng, axis, 'odd', bg)
    # The two witnesses vary geometry freely, but are selected for observable
    # evidence even at the minimum of two demonstrations.
    cover = _evidence(odd['input'], odd['output'])
    for _ in range(64):
        even = _make_pair(rng,1-axis,'even',bg,avoid_n=(odd['input'].shape[0]//2,))
        if REQUIRED_EVIDENCE <= cover | _evidence(even['input'],even['output']):
            break
    else:
        raise RuntimeError('bounded demonstration selection exhausted')
    puzzle = {'train':[odd,even], 'test':[]}
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, count in (('train',nt), ('test',ns)):
        while len(puzzle[split]) < count:
            train_shapes = {p['output'].shape for p in puzzle['train']}
            for _ in range(64):
                pair = _make_pair(rng,int(rng.integers(2)),'free',bg,
                                  avoid_shapes=train_shapes if split == 'test' else ())
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded uniqueness selection exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check arrays, explicit semantics, visible concepts and training evidence."""
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, covers, train_info, train_shapes = set(), set(), [], set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _decode(inp)
            assert out.shape == (d['h'],d['w'])
            assert _visible(d), 'stencil conceals a required inherited feature'
            i = d['info']
            # Cellwise distance check, separate from the vectorized fill path.
            for r in range(d['h']):
                for c in range(d['w']):
                    ink = int(d['stencil'][r % d['n'],c % d['n']])
                    k = r if i['axis'] == 0 else c
                    if ink != d['background']:
                        expected = ink
                    elif i['start'] <= k < i['stop']:
                        near, far = k-i['start']+1, i['stop']-k
                        expected = i['colors'][0] if near < far else i['colors'][1] if far < near else SEPARATOR
                    else:
                        expected = int(d['carrier'][r//d['factors'][0],c//d['factors'][1]])
                    assert int(out[r,c]) == expected, 'cell violates partition/overlay rule'
            assert np.array_equal(solve(inp),out)
            # A is observable in every pair, including test pairs.
            assert not np.array_equal(out,_overlay(d['base'],d['tiled'],d['background']))
            if split == 'train':
                covers |= _evidence(inp,out)
                train_info.append(d)
                train_shapes.add(out.shape)
            else:
                assert out.shape not in train_shapes, 'missing count/shape recombination shift'
    assert REQUIRED_EVIDENCE <= covers, 'missing demonstration evidence: '+str(REQUIRED_EVIDENCE-covers)
    assert {d['info']['axis'] for d in train_info} == {0,1}
    assert {d['info']['gap'] % 2 for d in train_info} == {0,1}
    assert len({d['n'] for d in train_info}) >= 2


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train','test'):
        for pair in puzzle[split]:
            for role in ('input','output'):
                shape, payload = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(payload)
    return digest.hexdigest()


def self_test():
    """Streaming 42-seed audit; no grids, logs, or traces are saved."""
    signatures, answers, shapes, structures = set(), set(), set(), set()
    panels, backgrounds, scales, test_cases, pad_geometry, stencil_inks = (set() for _ in range(6))
    source_parities, post_parities, source_odd_to_even = set(), set(), 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        answers.add(_normalized(puzzle['test'][0]['output']))
        for split in ('train','test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                panels.add(d['n'])
                backgrounds.add(d['background'])
                scales.add(d['factors'])
                shapes.add(pair['output'].shape)
                structures.add(_normalized(d['carrier']))
                stencil_inks.add(len(np.unique(d['stencil'][d['stencil'] != d['background']])))
                pad_geometry.add((d['source']['axis'],d['source']['gap'],d['source']['boxes']))
                source_parities.add(d['source']['gap'] % 2)
                post_parities.add(d['info']['gap'] % 2)
                source_odd_to_even += d['source']['gap'] % 2 == 1 and d['info']['gap'] % 2 == 0
                if split == 'test':
                    test_cases.add((d['info']['axis'],d['info']['gap'] % 2))
    assert len(signatures) == 42 and len(answers) >= 35
    assert len(shapes) >= 30 and len(structures) >= 100 and len(pad_geometry) >= 100
    assert panels == set(range(6,16)) and len(backgrounds) >= 8
    assert len(scales) >= 8 and test_cases == {(0,0),(0,1),(1,0),(1,1)}
    assert {0,1,2,3,4} <= stencil_inks
    assert source_parities == post_parities == {0,1} and source_odd_to_even >= 42
    assert _puzzle_digest(generate_puzzle(7)) == _puzzle_digest(generate_puzzle(7))
    # Metamorphic checks: rearranging counter cells changes no dimensions;
    # changing the axial count changes the actual gap parity after replication.
    witness = generate_puzzle(0)
    pair = next(p for p in witness['train'] if _decode(p['input'])['info']['gap'] % 2)
    original = pair['input'].copy()
    d = _decode(original)
    rearranged = original.copy()
    n = d['n']
    rearranged[:n,:n] = original[:n,:n].ravel()[::-1].reshape(n,n)
    rearranged[:n,n:] = original[:n,n:].T
    untouched = rearranged.copy()
    assert np.array_equal(solve(rearranged),pair['output'])
    assert np.array_equal(rearranged,untouched), 'solve mutated its input'
    changed = original.copy()
    counter = np.full((n,n),d['background'],dtype=np.int8)
    counter.flat[:2*n] = d['source']['colors'][0]
    if d['info']['axis'] == 0:
        changed[:n,:n] = counter
    else:
        changed[:n,n:] = counter
    assert _decode(changed)['info']['gap'] % 2 == 0
    assert np.any(pair['output'] == SEPARATOR) and not np.any(solve(changed) == SEPARATOR)
    for nt in range(2,11):
        for ns in range(1,min(4,nt-1)+1):
            generate_puzzle(1000+nt*10+ns,nt,ns)
    for counts in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,*counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(0)
    out = damaged['test'][0]['output']
    out[0,0] = (int(out[0,0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt label accepted')
    return dict(seeds=42,distinct_puzzles=len(signatures),distinct_first_answers=len(answers),
                output_shapes=len(shapes),carrier_structures=len(structures),
                pad_geometries=len(pad_geometry),panel_sides=sorted(panels),
                backgrounds=sorted(backgrounds),scale_pairs=len(scales),
                stencil_ink_counts=sorted(stencil_inks),source_odd_to_even=source_odd_to_even)
