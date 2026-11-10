"""A travelling stencil is read, front to back, by a sequence of gate bands.

Only NumPy and the standard library are used.  The decoder reads geometry and
the visible key; generated outputs are built separately from construction data.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=False)


def _signature(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _groups(positions):
    """Maximal contiguous integer intervals, with exclusive upper bounds."""
    result = []
    for v in positions:
        v = int(v)
        if result and result[-1][1] == v:
            result[-1] = result[-1][0], v + 1
        else:
            result.append((v, v + 1))
    return result


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells = [(y, x)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                        and mask[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        yield np.asarray(cells)


def _bbox(mask):
    p = np.argwhere(mask)
    if not len(p):
        raise ValueError("empty object")
    lo, hi = p.min(axis=0), p.max(axis=0) + 1
    return tuple(map(int, (lo[0], hi[0], lo[1], hi[1])))


def _decode(input_grid):
    """Normalize to a vertical lane and recover every role from this input."""
    raw = _grid(input_grid)
    bg = int(raw[0, 0])
    border = np.concatenate((raw[0], raw[-1], raw[:, 0], raw[:, -1]))
    if np.any(border != bg):
        raise ValueError("the family has a uniform one-cell outer border")
    found = []
    for transposed in (False, True):
        a = raw.T if transposed else raw
        for comp in _components(a != bg):
            r0, c0 = comp.min(axis=0)
            r1, c1 = comp.max(axis=0) + 1
            r0, r1, c0, c1 = map(int, (r0, r1, c0, c1))
            width = c1 - c0
            if r1-r0 != 3 or width < 3 or len(comp) != 3 * width:
                continue
            key = a[r0:r1, c0:c1]
            seq, fill = key[0].copy(), int(key[2, 0])
            if (not np.array_equal(key[0], key[1])
                    or not np.all(key[2] == fill) or len(np.unique(seq)) < 2
                    or fill in seq or bg in key):
                continue
            rest = a.copy()
            rest[r0:r1, c0:c1] = bg
            colors = [int(v) for v in np.unique(rest) if v != bg]
            if len(colors) != 3 or set(colors) & set(map(int, key.ravel())):
                continue
            for goal in colors:
                gy0, gy1, gx0, gx1 = _bbox(rest == goal)
                depth = gy1 - gy0
                if (depth < 2 or gx1-gx0 != width
                        or not np.all(rest[gy0:gy1, gx0:gx1] == goal)):
                    continue
                for stencil in [v for v in colors if v != goal]:
                    sy0, sy1, sx0, sx1 = _bbox(rest == stencil)
                    if (sy1-sy0 != depth or (sx0, sx1) != (gx0, gx1)
                            or not (sy1 < gy0 or gy1 < sy0)):
                        continue
                    local = rest[sy0:sy1, sx0:sx1]
                    mask = local == stencil
                    if np.all(mask) or np.any((local != bg) & ~mask):
                        continue
                    obstacle = next(v for v in colors if v not in (goal, stencil))
                    if gx0 < 1 or gx1 >= a.shape[1]:
                        continue
                    gate_rows, arms = [], []
                    good = True
                    for r in range(a.shape[0]):
                        p = np.flatnonzero(rest[r] == obstacle)
                        if not len(p):
                            continue
                        low, high = p[p < gx0], p[p >= gx1]
                        if (not len(low) or not len(high) or len(low)+len(high) != len(p)
                                or not np.array_equal(low, np.arange(low[0], gx0))
                                or not np.array_equal(high, np.arange(gx1, high[-1]+1))):
                            good = False
                            break
                        gate_rows.append(r)
                        arms.append((len(low), len(high)))
                    if not good or not gate_rows:
                        continue
                    # There is no third object or lane noise hidden outside these roles.
                    occupied = rest != bg
                    occupied[sy0:sy1, sx0:sx1] = False
                    occupied[gy0:gy1, gx0:gx1] = False
                    occupied[rest == obstacle] = False
                    if np.any(occupied):
                        continue
                    bands = _groups(gate_rows)
                    low_end, high_start = (sy1, gy0) if sy0 < gy0 else (gy1, sy0)
                    crossed = [b for b in bands if low_end <= b[0] and b[1] <= high_start]
                    if len(crossed) != depth:
                        continue
                    # A band cannot straddle an endpoint boundary in this family.
                    regions = ((0, min(sy0, gy0)), (sy0, sy1), (gy0, gy1),
                               (low_end, high_start), (max(sy1, gy1), a.shape[0]))
                    if any(not any(l <= b[0] and b[1] <= h for l, h in regions) for b in bands):
                        continue
                    down = sy0 < gy0
                    order = crossed if down else crossed[::-1]
                    indices = list(range(depth-1, -1, -1)) if down else list(range(depth))
                    found.append(dict(a=a, transposed=transposed, background=bg,
                                      sequence=seq, fill=fill, stencil=stencil, goal=goal,
                                      obstacle=obstacle, mask=mask.copy(), source=(sy0, sy1),
                                      target=(gy0, gy1), cross=(gx0, gx1), bands=bands,
                                      crossed=crossed, order=order, indices=indices,
                                      key_box=(r0, r1, c0, c1), arms=arms,
                                      depth=depth, width=width, down=down))
    if len(found) != 1:
        raise ValueError(f"expected one readable scene, found {len(found)}")
    return found[0]


def _render(info):
    out = info['a'].copy()
    s0, s1 = info['source']
    g0, g1 = info['target']
    c0, c1 = info['cross']
    inked = np.where(info['mask'], info['sequence'][None, :], info['fill'])
    out[s0:s1, c0:c1] = info['background']
    for (r0, r1), row in zip(info['order'], info['indices']):
        out[r0:r1, c0:c1] = inked[row]
    out[g0:g1, c0:c1] = inked
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the key, move the stencil, and print its slices at crossed bands."""
    info = _decode(input_grid)
    out = _render(info)
    return (out.T if info['transposed'] else out).copy()


def _mask(rng, depth, width, style):
    m = np.zeros((depth, width), dtype=bool)
    if style == 'walk':
        x = int(rng.integers(width))
        for r in range(depth):
            nx = int(rng.integers(width))
            m[r, min(x, nx):max(x, nx)+1] = True
            x = nx
    elif style == 'blocks':
        for _ in range(int(rng.integers(2, 7))):
            r, c = int(rng.integers(depth)), int(rng.integers(width))
            m[r:min(depth, r+int(rng.integers(1, 4))),
              c:min(width, c+int(rng.integers(1, 5)))] = True
    elif style == 'frame':
        m[0] = m[-1] = True
        m[:, 0] = m[:, -1] = True
        m |= rng.random(m.shape) < .15
    else:
        m[:] = rng.random(m.shape) < rng.uniform(.3, .75)
    # Tight bounding boxes, while empty internal scanlines/columns remain legal.
    m[0, int(rng.integers(width))] = True
    m[-1, int(rng.integers(width))] = True
    m[int(rng.integers(depth)), 0] = True
    m[int(rng.integers(depth)), -1] = True
    if np.all(m):
        m[int(rng.integers(depth)), int(rng.integers(width))] = False
        if not (m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any()):
            return _mask(rng, depth, width, 'random')
    return m


def _sequence(rng, width, available):
    chosen = rng.choice(available, int(rng.integers(2, min(5, len(available), width)+1)),
                        replace=False)
    mode = int(rng.integers(3))
    if mode == 0:
        seq = rng.choice(chosen, width)
    elif mode == 1:
        seq = np.array([chosen[i % len(chosen)] for i in range(width)])
    else:
        cuts = sorted(map(int, rng.choice(np.arange(1, width), len(chosen)-1, replace=False)))
        seq = np.repeat(chosen, np.diff([0] + cuts + [width]))
    seq[0], seq[-1] = chosen[0], chosen[1]
    return seq.astype(np.int8)


def _make_pair(rng, depth, width, transposed, down, teaching=False,
               spacing=False, preferred_layout=None):
    # Build canonical vertical geometry; transposition changes every object,
    # including the direction of the key, without rotating the moving stencil.
    lengths = rng.integers(1, 3, size=depth)
    if teaching:
        lengths[int(rng.integers(depth))] = 2
    spaces = rng.integers(1, 4 if spacing else 3, size=depth-1)
    if spacing:
        spaces[int(rng.integers(depth-1))] = 3
    core = int(lengths.sum() + spaces.sum())
    before = int(rng.integers(0, 2))
    after = int(rng.integers(0, 2))
    lead = 3 if teaching else int(rng.integers(1, 5))
    trail = 3 if teaching else int(rng.integers(1, 5))
    lo = lead
    hi = lo + depth + before + core + after
    height = hi + depth + trail
    if height > 30:
        return None
    cross_left, cross_right = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    c0 = cross_left + 1
    c1 = c0 + width
    span = c1 + cross_right + 1
    if span + width + 2 > 30:
        return None
    bands = []
    p = lo + depth + before
    for i, thick in enumerate(lengths):
        bands.append((p, p + int(thick)))
        p += int(thick)
        if i < depth - 1:
            p += int(spaces[i])
    crossed = bands.copy()
    extras = []
    # Kept away from the first/last crossed band by at least one blank slice.
    if teaching or rng.random() < .35:
        extras.append((lo, lo+1))
    if teaching or rng.random() < .4:
        extras.append((hi+depth-1, hi+depth))
    if lead >= 3 and (teaching or rng.random() < .45):
        extras.append((1, 2))
    if trail >= 3 and (teaching or rng.random() < .45):
        extras.append((height-2, height-1))
    bands = sorted(bands + extras)
    # Endpoint bands must not merge with an adjacent outside band.
    if any(a[1] >= b[0] for a, b in zip(bands, bands[1:])):
        return None
    palette = list(map(int, rng.permutation(10)))
    bg, obstacle, stencil, goal, fill = palette[:5]
    seq = _sequence(rng, width, palette[5:])
    style = str(rng.choice(('walk', 'blocks', 'frame', 'random')))
    mask = _mask(rng, depth, width, style)
    # Training must show order, not just identical slices or a symmetric stencil.
    if teaching and np.array_equal(mask, mask[::-1]):
        return None
    src, dst = (lo, hi) if down else (hi, lo)
    a = np.full((height, span), bg, dtype=np.int8)
    for r0, r1 in bands:
        left = int(rng.integers(1, cross_left+1))
        right = int(rng.integers(1, cross_right+1))
        a[r0:r1, c0-left:c0] = obstacle
        a[r0:r1, c1:c1+right] = obstacle
    a[src:src+depth, c0:c1][mask] = stencil
    a[dst:dst+depth, c0:c1] = goal
    # Construct the label without calling solve or _decode.
    out = a.copy()
    out[src:src+depth, c0:c1] = bg
    colored = np.where(mask, seq[None, :], fill)
    for rank, (r0, r1) in enumerate(crossed if down else crossed[::-1]):
        row = depth-1-rank if down else rank
        out[r0:r1, c0:c1] = colored[row]
    out[dst:dst+depth, c0:c1] = colored
    layouts = ['left', 'right'] + (['above', 'below'] if height <= 26 else [])
    layout = preferred_layout if preferred_layout in layouts else str(rng.choice(layouts))
    if layout in ('left', 'right'):
        shape = height, span + width + 2
        scene_y, scene_x = 0, width+2 if layout == 'left' else 0
        ky = int(rng.integers(1, height-3))
        kx = 1 if layout == 'left' else span+1
    else:
        shape = height+4, span
        scene_y, scene_x = (4 if layout == 'above' else 0), 0
        ky = 1 if layout == 'above' else height
        kx = int(rng.integers(1, span-width))
    inp, final = (np.full(shape, bg, dtype=np.int8) for _ in range(2))
    for canvas, scene in ((inp, a), (final, out)):
        canvas[scene_y:scene_y+height, scene_x:scene_x+span] = scene
        canvas[ky:ky+2, kx:kx+width] = seq
        canvas[ky+2, kx:kx+width] = fill
    if transposed:
        inp, final = inp.T.copy(), final.T.copy()
    return {'input': inp, 'output': final}


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    first_axis, first_sign = bool(rng.integers(2)), bool(rng.integers(2))
    shift = str(rng.choice(('depth', 'width', 'spacing', 'recombination')))
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            for _ in range(300):
                teaching = split == 'train' and i < 2
                depth = (3 if i == 0 else int(rng.choice((2, 4)))) if teaching else int(rng.integers(2, 5))
                width = int(rng.integers(3, 8))
                axis, down = bool(rng.integers(2)), bool(rng.integers(2))
                if teaching:
                    axis, down = (first_axis, first_sign) if i == 0 else (not first_axis, not first_sign)
                elif split == 'train' and shift == 'recombination':
                    axis, down = ((first_axis, first_sign) if rng.random() < .5
                                  else (not first_axis, not first_sign))
                if split == 'test':
                    if shift == 'depth':
                        depth = 5
                    elif shift == 'width':
                        width = int(rng.integers(8, 11))
                    elif shift == 'spacing':
                        depth = int(rng.integers(2, 4))
                    else:
                        # Test the unseen axis/sign combination of the two core demos.
                        axis, down = first_axis, not first_sign
                pair = _make_pair(rng, depth, width, axis, down, teaching,
                                  spacing=split == 'test' and shift == 'spacing',
                                  preferred_layout='right' if teaching and i == 0 else
                                  'above' if teaching and i == 1 else None)
                if pair is None or _signature(pair['input']) in seen:
                    continue
                info = _decode(pair['input'])
                if teaching and not _witnesses(info):
                    continue
                seen.add(_signature(pair['input']))
                result[split].append(pair)
                break
            else:
                raise RuntimeError('could not construct a distinct readable example')
    validate_generated_puzzle(result)
    return result


def _witnesses(info):
    """The two core demonstrations reject the main competing mechanisms."""
    m, s = info['mask'], info['sequence']
    basic = bool(np.any(~m) and not np.array_equal(m, m[::-1])
                 and np.any(m & (s != s[::-1])[None, :]))
    # A thick band only teaches grouping when advancing on each constituent
    # slice would actually change the displayed profile.  Saturate at the final
    # stencil slice so even that charitable shortcut is distinguished.
    profiles = np.where(m, s[None, :], info['fill'])[info['indices']]
    per_slice_differs = False
    n = 0
    for rank, (a, b) in enumerate(info['order']):
        for _ in range(b-a):
            per_slice_differs |= not np.array_equal(profiles[rank], profiles[min(n, len(profiles)-1)])
            n += 1
    return basic and per_slice_differs


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            sig = _signature(inp)
            assert sig not in seen
            seen.add(sig)
            info = _decode(inp)
            canonical = out.T if info['transposed'] else out
            assert np.array_equal(canonical, _render(info))
            # Gates and the key are untouched, including irrelevant gate openings.
            assert np.array_equal(out[inp == info['obstacle']], inp[inp == info['obstacle']])
            r0, r1, c0, c1 = info['key_box']
            assert np.array_equal(canonical[r0:r1, c0:c1], info['a'][r0:r1, c0:c1])
            s0, s1 = info['source']
            g0, g1 = info['target']
            x0, x1 = info['cross']
            assert np.all(canonical[s0:s1, x0:x1] == info['background'])
            assert np.all(canonical[g0:g1, x0:x1][~info['mask']] == info['fill'])
            for band, row in zip(info['order'], info['indices']):
                for r in range(*band):
                    profile = canonical[r, x0:x1]
                    assert np.array_equal(profile[info['mask'][row]], info['sequence'][info['mask'][row]])
                    assert np.all(profile[~info['mask'][row]] == info['fill'])
            for b in info['bands']:
                if b not in info['crossed'] and not (b[0] < s1 and s0 < b[1]) and not (b[0] < g1 and g0 < b[1]):
                    assert np.array_equal(canonical[b[0]:b[1], x0:x1], info['a'][b[0]:b[1], x0:x1])
            if split == 'train':
                train.append(info)
    assert {x['transposed'] for x in train[:2]} == {False, True}
    assert {x['down'] for x in train[:2]} == {False, True}
    assert train[0]['depth'] != train[1]['depth']
    for info in train[:2]:
        assert _witnesses(info), 'demonstrations must distinguish scan/key reversal and solid printing'
        assert any(b-a > 1 for a, b in info['crossed']), 'teach gate-band grouping'
        assert len(info['bands']) > len(info['crossed']), 'teach finite traversal filtering'
        s0, s1 = info['source']
        g0, g1 = info['target']
        assert any(s0 <= a < b <= s1 for a, b in info['bands'])
        assert any(g0 <= a < b <= g1 for a, b in info['bands'])
        assert any(b <= min(s0, g0) for a, b in info['bands'])
        assert any(a >= max(s1, g1) for a, b in info['bands'])


def _fingerprint(p):
    return tuple((split, tuple((_signature(x['input']), _signature(x['output']))
                               for x in p[split])) for split in ('train', 'test'))


def self_test():
    """Bounded checks over 42 seeds; return descriptive diversity measurements."""
    signatures, shapes, masks, palettes, directions, depths, widths = (set() for _ in range(7))
    thick = holes = blank = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_fingerprint(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                inp = pair['input']
                before = inp.copy()
                assert np.array_equal(solve(inp), pair['output'])
                assert np.array_equal(inp, before), 'solve must not mutate input'
                info = _decode(inp)
                if seed < 6:
                    # Visible-role relabeling and a whole-scene transposition
                    # must commute with solving, independent of generated labels.
                    permutation = np.random.default_rng(500+seed).permutation(10)
                    assert np.array_equal(solve(permutation[inp]), permutation[pair['output']])
                    assert np.array_equal(solve(inp.T), pair['output'].T)
                shapes.add(inp.shape)
                masks.add(_signature(info['mask']))
                palettes.add(tuple(map(int, info['sequence'])))
                directions.add((info['transposed'], info['down']))
                depths.add(info['depth'])
                widths.add(info['width'])
                thick += any(b-a > 1 for a, b in info['crossed'])
                holes += int((~info['mask']).sum())
                blank += bool(np.any(~info['mask'].any(axis=1)))
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(masks) >= 100 and len(palettes) >= 100
    assert len(directions) == 4 and depths == {2, 3, 4, 5}
    assert min(widths) == 3 and max(widths) == 10
    assert thick > 100 and holes > 100 and blank > 0
    assert _fingerprint(generate_puzzle(47, 10, 4)) == _fingerprint(generate_puzzle(47, 10, 4))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(83, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                masks=len(masks), sequences=len(palettes), directions=len(directions),
                depths=sorted(depths), widths=sorted(widths), thick_band_pairs=thick,
                hole_cells=holes, blank_scanline_pairs=blank)
