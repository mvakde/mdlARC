"""Self-similar alarm mosaics. No file access, hidden state, or parent imports.

The glyph's ink is the coarse scaffold. Diagnose signed tile errors against its
unique nearest translation; erase extras and stamp alarm glyphs around holes.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


HOST_SEEDS = (0, 63907944, 157799249, 164484512, 539526461, 182419992,
              580353797, 570188412, 610638204, 335283920, 83744620, 975960071,
              683588583, 994806958, 798971109, 327201999, 723852123, 7388799,
              425278397, 987696427, 814596947, 244602410, 185883195, 558872400,
              847319917, 325652846, 662378361, 819913306, 645092915, 687133622,
              959024807, 822081975, 490270176, 689403523, 596719954, 274096090,
              320373389, 991341315, 80458596, 481403080, 875384373, 820788968)
_SHAPES = tuple((h, w) for h in range(2, 5) for w in range(2, 5) if h*w >= 6)


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10),
                                 (num_test, 'num_test', 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected a 1..30 integer ARC grid with colors 0..9')
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _runs(indices):
    result = []
    for raw in indices:
        i = int(raw)
        if result and result[-1][1] == i:
            result[-1] = (result[-1][0], i+1)
        else:
            result.append((i, i+1))
    return result


def _lattice(a):
    d = int(a[0, 0])
    yy = _runs(np.flatnonzero(np.all(a == d, axis=1)))
    xx = _runs(np.flatnonzero(np.all(a == d, axis=0)))
    if (len(yy) < 3 or len(xx) < 3 or yy[0][0] != 0 or xx[0][0] != 0 or
            yy[-1][1] != a.shape[0] or xx[-1][1] != a.shape[1]):
        raise ValueError('missing complete enclosing divider lattice')
    ts = {b-y for y, b in yy+xx}
    hs = {yy[i+1][0]-yy[i][1] for i in range(len(yy)-1)}
    ws = {xx[i+1][0]-xx[i][1] for i in range(len(xx)-1)}
    if len(ts) != 1 or len(hs) != 1 or len(ws) != 1:
        raise ValueError('irregular divider lattice')
    t, h, w = ts.pop(), hs.pop(), ws.pop()
    if t not in (1, 2) or (h, w) not in _SHAPES:
        raise ValueError('unsupported tile geometry')
    return (h, w, t, len(yy)-1, len(xx)-1), d


def _slice(geometry, r, c):
    h, w, t, _, _ = geometry
    y, x = t+r*(h+t), t+c*(w+t)
    return np.s_[y:y+h, x:x+w]


def _neighbors(points, rows, cols):
    return {(r+dy, c+dx) for r, c in points
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if 0 <= r+dy < rows and 0 <= c+dx < cols} - set(points)


def _templates(mask, rows, cols):
    points = [tuple(map(int, p)) for p in np.argwhere(mask)]
    h, w = mask.shape
    for r in range(rows-h+1):
        for c in range(cols-w+1):
            yield (r, c), {(r+y, c+x) for y, x in points}


def _register(mask, rows, cols, observed):
    ranked = sorted((len(target ^ observed), origin, target)
                    for origin, target in _templates(mask, rows, cols))
    if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        raise ValueError('coarse stencil translation is ambiguous')
    _, origin, target = ranked[0]
    return origin, target


def _infer(input_grid):
    a = _grid(input_grid)
    geometry, divider = _lattice(a)
    h, w, t, rows, cols = geometry
    records = {}
    for r in range(rows):
        for c in range(cols):
            tile = a[_slice(geometry, r, c)]
            if np.any(tile == divider):
                raise ValueError('divider color must be absent from input tile interiors')
            records.setdefault(_key(tile), [tile.copy(), set()])[1].add((r, c))
    if len(records) != 2:
        raise ValueError('expected exactly ordinary and glyph input tiles')
    ordinary_record, glyph_record = sorted(records.values(), key=lambda x: -len(x[1]))
    ordinary, ordinary_positions = ordinary_record
    glyph, observed = glyph_record
    if not len(ordinary_positions) > len(observed) >= 3:
        raise ValueError('ordinary tiles must be a strict majority, with several glyphs')
    ordinary_colors = set(map(int, np.unique(ordinary)))
    novel = set(map(int, np.unique(glyph))) - ordinary_colors
    if len(ordinary_colors) != 2 or len(novel) != 1:
        raise ValueError('glyph ink must be the unique novel interior color')
    ink = novel.pop()
    mask = glyph == ink
    if mask.all() or not mask.any() or len(np.unique(glyph[~mask])) != 1:
        raise ValueError('glyph must have a constant non-ink background')
    background = int(glyph[~mask][0])
    if background not in ordinary_colors:
        raise ValueError('glyph background must be an ordinary-tile color')
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        raise ValueError('ink stencil must span its tile in both axes')
    origin, target = _register(mask, rows, cols, observed)
    holes, extras = target-observed, observed-target
    if not holes or not extras or len(target & observed) <= len(extras):
        raise ValueError('expected both signed errors and a majority of correct glyph clues')
    if len(target & observed) < 2:
        raise ValueError('at least two correct glyph copies are required')
    halo = _neighbors(holes, rows, cols)
    alarm = glyph.copy()
    alarm[mask] = divider
    return dict(geometry=geometry, divider=divider, ordinary=ordinary, glyph=glyph,
                ink=ink, background=background, mask=mask, origin=origin,
                target=target, observed=observed, holes=holes, extras=extras,
                halo=halo, alarm=alarm)


def _prediction(a, info, target=None, *, fill_holes=False, cardinal=False,
                ink_only=False):
    """Input reconstruction, with explicit counterfactuals for self-tests."""
    g = info['geometry']
    rows, cols = g[-2:]
    target = info['target'] if target is None else target
    holes = target-info['observed']
    extras = info['observed']-target
    halo = _neighbors(holes, rows, cols)
    if cardinal:
        halo = {(r+dy, c+dx) for r, c in holes
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r+dy < rows and 0 <= c+dx < cols} - holes
    out = a.copy()
    for r, c in extras:
        out[_slice(g, r, c)] = info['ordinary']
    for r, c in halo:
        tile = out[_slice(g, r, c)]
        if ink_only:
            tile[info['mask']] = info['divider']
        else:
            tile[:] = info['alarm']
    if fill_holes:
        for r, c in holes:
            out[_slice(g, r, c)] = info['glyph']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use the visible lattice, tile patterns, and uniquely nearest stencil only."""
    a = _grid(input_grid)
    return _prediction(a, _infer(a)).astype(int)


def _paint(geometry, divider, ordinary, glyph, target, holes, extras, output):
    """Construction renderer uses supplied sets; never invokes input inference."""
    h, w, t, rows, cols = geometry
    grid = np.full((rows*(h+t)+t, cols*(w+t)+t), divider, dtype=np.int8)
    alarm = glyph.copy()
    # Background belongs to the ordinary palette; ink is its sole novel color.
    alarm[~np.isin(glyph, np.unique(ordinary))] = divider
    halo = _neighbors(holes, rows, cols) if output else set()
    for r in range(rows):
        for c in range(cols):
            p = r, c
            if output:
                tile = alarm if p in halo else glyph if p in target-holes else ordinary
            else:
                tile = glyph if p in (target-holes) | extras else ordinary
            grid[_slice(geometry, r, c)] = tile
    return grid


def _asymmetric(mask):
    # All nonidentity dihedral maps differ, including maps swapping the axes.
    transforms = [np.rot90(mask, k) for k in range(4)]
    transforms += [np.fliplr(m) for m in transforms]
    return all(not np.array_equal(mask, m) for m in transforms[1:])


def _motif_key(tile):
    roles = {}
    return tile.shape, tuple(roles.setdefault(int(v), len(roles)) for v in tile.flat)


def _compatible_orientations(pair, info):
    """Challenge orientation against actual outputs, not mask asymmetry alone.

    Even a tied alternative registration is considered compatible if any of
    its best matches gives the same answer. Thus ambiguity is not hidden by a
    scan-order or unique-registration assumption in the counterfactual.
    """
    transforms = [np.rot90(info['mask'], k) for k in range(4)]
    transforms += [np.fliplr(m) for m in transforms]
    inp, out = _grid(pair['input']), _grid(pair['output'])
    rows, cols = info['geometry'][-2:]
    result = set()
    for index, mask in enumerate(transforms):
        candidates = list(_templates(mask, rows, cols))
        if not candidates:
            continue
        best = min(len(target ^ info['observed']) for _, target in candidates)
        if any(len(target ^ info['observed']) == best and np.array_equal(
                _prediction(inp, info, target=target), out)
                for _, target in candidates):
            result.add(index)
    return result


def _make_pair(rng, hole_count, palette, teaching, forbidden_motifs=(),
               forbidden_masks=(), seen=(), prior_orientations=None):
    for attempt in range(900):
        h, w = _SHAPES[int(rng.integers(len(_SHAPES)))]
        t = int(rng.choice((1, 2), p=(.8, .2)))
        max_rows, max_cols = (30-t)//(h+t), (30-t)//(w+t)
        rows = int(rng.integers(max(3, h), max_rows+1))
        cols = int(rng.integers(max(3, w), max_cols+1))
        geometry = h, w, t, rows, cols
        extra_count = int(rng.integers(1, 4))
        low = hole_count+max(2, extra_count+1)
        high = min(h*w-1, (rows*cols-1)//2+hole_count-extra_count)
        if low > high:
            continue
        count = int(rng.integers(low, high+1))
        mask = np.zeros((h, w), dtype=bool)
        mask.flat[rng.choice(h*w, count, replace=False)] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if _key(mask) in forbidden_masks or (teaching == 'single' and not _asymmetric(mask)):
            continue
        origin = int(rng.integers(rows-h+1)), int(rng.integers(cols-w+1))
        target = {(origin[0]+int(y), origin[1]+int(x)) for y, x in np.argwhere(mask)}
        ordered = sorted(target)
        if teaching == 'cluster':
            choices = [(p, q) for p, q in combinations(ordered, 2)
                       if max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1 and
                       any(y in (0, rows-1) or x in (0, cols-1) for y, x in (p, q))]
            if not choices:
                continue
            holes = set(choices[int(rng.integers(len(choices)))])
        else:
            indices = rng.choice(len(ordered), hole_count, replace=False)
            holes = {ordered[int(i)] for i in indices}
        halo = _neighbors(holes, rows, cols)
        available = {(r, c) for r in range(rows) for c in range(cols)} - target
        if teaching == 'single':
            available -= halo  # Visible erasure separate from any alarm.
        elif teaching == 'cluster':
            available &= halo  # Teach that an alarm overwrites an adjacent extra.
        if len(available) < extra_count:
            continue
        ordered_extra = sorted(available)
        extras = {ordered_extra[int(i)] for i in rng.choice(len(ordered_extra), extra_count, replace=False)}
        observed = (target-holes) | extras
        if not len(observed)*2 < rows*cols:
            continue
        try:
            inferred_origin, inferred_target = _register(mask, rows, cols, observed)
        except ValueError:
            continue
        if inferred_origin != origin or inferred_target != target:
            continue
        divider, first, second, ink = palette
        labels = rng.integers(2, size=(h, w))
        if labels.min() == labels.max():
            continue
        ordinary = np.where(labels, first, second).astype(np.int8)
        if _motif_key(ordinary) in forbidden_motifs:
            continue
        backgrounds = [v for v in (first, second) if np.any(ordinary[~mask] != v)]
        background = int(rng.choice(backgrounds))
        glyph = np.where(mask, ink, background).astype(np.int8)
        inp = _paint(geometry, divider, ordinary, glyph, target, holes, extras, False)
        if _key(inp) in seen:
            continue
        out = _paint(geometry, divider, ordinary, glyph, target, holes, extras, True)
        info = _infer(inp)
        # Every pair demonstrates whole-tile stamping, not ink-only painting.
        if np.array_equal(_prediction(inp, info, ink_only=True), out):
            continue
        if teaching == 'single':
            rectangle = {(origin[0]+y, origin[1]+x) for y in range(h) for x in range(w)}
            if np.array_equal(_prediction(inp, info, target=rectangle), out):
                continue
        if not np.array_equal(_prediction(inp, info), out):
            raise AssertionError('construction disagrees with input-only reconstruction')
        pair = {'input': inp.astype(int), 'output': out.astype(int)}
        if prior_orientations is not None and (
                prior_orientations & _compatible_orientations(pair, info)) != {0}:
            continue
        return pair
    raise RuntimeError('bounded sampler could not construct a readable alarm mosaic')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.random() < .35)
    base_palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    puzzle = {'train': [], 'test': []}
    seen, motifs, masks = set(), set(), set()
    orientations = set(range(8))
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            if split == 'train':
                hole_count = 1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3))
                teaching = 'single' if i == 0 else 'cluster' if i == 1 else 'free'
            else:
                hole_count = int(rng.integers(3, 5))
                teaching = 'free'
            palette = base_palette if stable_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
            pair = _make_pair(rng, hole_count, palette, teaching,
                              motifs if split == 'train' and i == 1 else (),
                              masks if split == 'test' or i == 1 else (), seen,
                              orientations if split == 'train' and i == 1 else None)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
            if split == 'train':
                info = _infer(pair['input'])
                motifs.add(_motif_key(info['ordinary']))
                masks.add(_key(info['mask']))
                if i < 2:
                    orientations &= _compatible_orientations(pair, info)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _infer(inp)
            records[split].append(info)
            g = info['geometry']
            assert 1 <= len(info['extras']) <= 3
            assert 1 <= len(info['holes']) <= 4
            assert np.array_equal(solve(inp.copy()), out)
            # Check semantic tile classes directly, including ordinary holes.
            rows, cols = g[-2:]
            interiors = np.zeros_like(inp, dtype=bool)
            for r in range(rows):
                for c in range(cols):
                    sl = _slice(g, r, c)
                    interiors[sl] = True
                    p = r, c
                    tile = info['alarm'] if p in info['halo'] else info['glyph'] if p in info['target']-info['holes'] else info['ordinary']
                    assert np.array_equal(out[sl], tile)
            assert np.array_equal(out[~interiors], inp[~interiors])
            assert not np.array_equal(out, _prediction(inp, info, fill_holes=True)), 'A: holes must remain'
            assert not np.array_equal(out, _prediction(inp, info, ink_only=True)), 'B: copy the whole glyph'
    train = records['train']
    assert len(train[0]['holes']) == 1 and _asymmetric(train[0]['mask'])
    first_input = _grid(puzzle['train'][0]['input'])
    first_output = _grid(puzzle['train'][0]['output'])
    r, c = train[0]['origin']
    h, w = train[0]['mask'].shape
    rectangle = {(r+y, c+x) for y in range(h) for x in range(w)}
    assert not np.array_equal(first_output, _prediction(first_input, train[0], target=rectangle)), 'B: the stencil is not a filled rectangle'
    assert not np.array_equal(first_output, _prediction(first_input, train[0], cardinal=True)), 'A: diagonal neighbors must matter'
    assert len(train[1]['holes']) == 2
    assert train[0]['extras'].isdisjoint(train[0]['halo'])
    assert train[1]['extras'] <= train[1]['halo']
    rows, cols = train[1]['geometry'][-2:]
    assert any(y in (0, rows-1) or x in (0, cols-1) for y, x in train[1]['holes'])
    p, q = sorted(train[1]['holes'])
    assert max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1
    assert len({_motif_key(f['ordinary']) for f in train}) >= 2
    train_masks = {_key(f['mask']) for f in train}
    assert len(train_masks) >= 2
    orientations = set(range(8))
    for pair, f in zip(puzzle['train'][:2], train[:2]):
        orientations &= _compatible_orientations(pair, f)
    assert orientations == {0}, 'the demonstrations must distinguish stencil orientation'
    for f in records['test']:
        assert len(f['holes']) in (3, 4) and _key(f['mask']) not in train_masks


def _equal(p, q):
    return all(len(p[s]) == len(q[s]) and all(np.array_equal(a[k], b[k])
               for a, b in zip(p[s], q[s]) for k in ('input', 'output'))
               for s in ('train', 'test'))


def self_test():
    """42 streamed seeds; bounded retention; semantic and metamorphic checks."""
    # Hand-worked new-rule fixture. The expected tile positions below are
    # literal coordinates, independent of the sampler, registration and halo
    # helpers. It includes an extra in a halo, a remote extra, and two touching
    # hole neighborhoods. Both missing centers must survive as ordinary tiles.
    g = (3, 3, 1, 6, 6)
    ordinary = np.array([[2, 2, 3], [3, 2, 3], [2, 3, 2]])
    glyph = np.array([[4, 2, 4], [4, 4, 2], [2, 4, 4]])
    alarm = np.array([[8, 2, 8], [8, 8, 2], [2, 8, 8]])
    supplied = {(1, 4), (2, 2), (3, 3), (3, 4), (5, 0), (2, 4)}
    outlined = {(0, 1), (0, 2), (0, 3), (1, 1), (1, 3), (1, 4),
                (2, 1), (2, 2), (2, 4), (3, 2), (3, 3), (3, 4)}
    manual_in = np.full((25, 25), 8, dtype=int)
    manual_out = manual_in.copy()
    for r in range(6):
        for c in range(6):
            sl = np.s_[1+4*r:4+4*r, 1+4*c:4+4*c]
            manual_in[sl] = glyph if (r, c) in supplied else ordinary
            manual_out[sl] = alarm if (r, c) in outlined else ordinary
    f = _infer(manual_in)
    assert f['origin'] == (1, 2) and f['holes'] == {(1, 2), (2, 3)}
    assert f['extras'] == {(5, 0), (2, 4)} and f['halo'] == outlined
    assert np.array_equal(solve(manual_in), manual_out)
    shapes, geometries, designs, motifs, palettes, origins = set(), set(), set(), set(), set(), set()
    holes, extras, normalized = set(), set(), set()
    max_active = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                f = _infer(inp)
                shapes.add(inp.shape)
                geometries.add(f['geometry'])
                designs.add(_key(f['mask']))
                motifs.add(_motif_key(f['ordinary']))
                palettes.add(tuple(map(int, np.unique(inp))))
                origins.add(f['origin'])
                holes.add(len(f['holes']))
                extras.add(len(f['extras']))
                normalized.add(_motif_key(inp))
                max_active = max(max_active, int(f['mask'].sum()))
        if seed == 0:
            assert _equal(p, generate_puzzle(seed))
            nested = {s: [{k: value.tolist() for k, value in pair.items()}
                          for pair in p[s]] for s in p}
            validate_generated_puzzle(nested)
            a, b = p['train'][0], p['train'][1]
            f = _infer(a['input'])
            assert not np.array_equal(a['output'], _prediction(a['input'], f, cardinal=True))
            r, c = f['origin']
            h, w = f['mask'].shape
            rectangle = {(r+y, c+x) for y in range(h) for x in range(w)}
            assert not np.array_equal(a['output'], _prediction(a['input'], f, target=rectangle))
            # Coarse geometry and micro glyph transform together under symmetry.
            for pair in (a, b, p['test'][0]):
                for k in range(4):
                    for reflected in (False, True):
                        ii, oo = pair['input'], pair['output']
                        if reflected:
                            ii, oo = np.fliplr(ii), np.fliplr(oo)
                        assert np.array_equal(solve(np.rot90(ii, k)), np.rot90(oo, k))
                permutation = np.random.default_rng(100).permutation(10)
                assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
            bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
            f = _infer(bad['test'][0]['input'])
            hole = min(f['holes'])
            bad['test'][0]['output'][_slice(f['geometry'], *hole)] = f['glyph']
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('filled defect center accepted')
    assert len(shapes) >= 30 and len(geometries) >= 50
    assert len(designs) >= 90 and len(motifs) >= 80
    assert len(palettes) >= 30 and len(origins) >= 8
    assert holes == {1, 2, 3, 4} and extras == {1, 2, 3}
    assert len(normalized) >= 190 and max_active >= 11
    # Deliberate registration tie; never break it by scan order.
    ambiguous_mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    try:
        _register(ambiguous_mask, 5, 6, {(2, 2)})
    except ValueError:
        pass
    else:
        raise AssertionError('ambiguous registration accepted')
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': len(HOST_SEEDS), 'pairs': 5*len(HOST_SEEDS),
            'canvas_shapes': len(shapes), 'geometries': len(geometries),
            'stencils': len(designs), 'ordinary_motifs': len(motifs),
            'palettes': len(palettes), 'origins': len(origins),
            'color_normalized_inputs': len(normalized),
            'hole_counts': sorted(holes), 'extra_counts': sorted(extras)}
