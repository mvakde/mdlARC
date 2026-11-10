"""Endpoint printing: diamond cap endpoints emit diagonal trains of motif tiles.

All conventions are geometric and fixed. The solver reads only its input.
The validator's walking and coordinate renderers share segmentation; neither is
an independent semantic oracle. See task_spec.md for the supported family.
"""
from itertools import permutations
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
RADII = (3, 4, 5)
ORDERS = tuple((t, r, c) for t in (False, True)
               for r in permutations(range(3)) for c in permutations(range(3)))


def _full_box(mask):
    return (mask.shape == (3, 3) and 2 <= int(mask.sum()) <= 8
            and mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any())


def _ordered(mask, order):
    t, rows, cols = order
    return (mask.T if t else mask)[np.ix_(rows, cols)]


MASKS = np.array([m for code in range(1, 511)
                  if _full_box(m := np.array([(code >> k) & 1 for k in range(9)],
                                             dtype=bool).reshape(3, 3))])
DENSITIES = MASKS.sum(axis=(1, 2))
AGREEMENT = np.array([[np.array_equal(m, _ordered(m, order)) for order in ORDERS]
                     for m in MASKS])
ASYMMETRIC = np.array([all(not np.array_equal(m, other)
                          for other in [np.rot90(m, k) for k in (1, 2, 3)]
                          + [np.rot90(m[:, ::-1], k) for k in range(4)])
                       for m in MASKS])


def _inside(cell, shape):
    return all(0 <= int(x) < n for x, n in zip(cell, shape))


def _outline(center, radius):
    r, c = center
    return [(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius]


def _cap(center, radius, direction):
    r, c = center
    vr, vc = DIRECTIONS[direction]
    return [(r + vr * (k + 1), c + vc * (radius - k + 1))
            for k in range(radius + 1)]


def _boxes(endpoint, direction, shape):
    """Top-left origins of 3x3 tiles; their near corner follows the endpoint."""
    if not _inside(endpoint, shape):
        return []
    vr, vc = DIRECTIONS[direction]
    top = endpoint[0] + (1 if vr > 0 else -3)
    left = endpoint[1] + (1 if vc > 0 else -3)
    boxes = []
    while top < shape[0] and left < shape[1] and top + 3 > 0 and left + 3 > 0:
        boxes.append((top, left))
        top += 3 * vr
        left += 3 * vc
    return boxes


def _box_cells(origin, shape):
    top, left = origin
    return {(r, c) for r in range(max(0, top), min(shape[0], top + 3))
            for c in range(max(0, left), min(shape[1], left + 3))}


def _protected(e, shape):
    cells = set(_outline(e['center'], e['radius']))
    r, c = e['center']
    cells.update((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    for d in e['caps']:
        cells.update(p for p in _cap(e['center'], e['radius'], d) if _inside(p, shape))
    return cells


def _streams(e, shape, whole_cap=False):
    for d, color in e['caps'].items():
        cap = _cap(e['center'], e['radius'], d)
        endpoints = cap if whole_cap else (cap[0], cap[-1])
        for endpoint in endpoints:
            yield d, color, endpoint, _boxes(endpoint, d, shape)


def _print_footprint(e, shape):
    cells = set()
    for _, _, _, boxes in _streams(e, shape):
        for origin in boxes:
            part = _box_cells(origin, shape)
            if cells & part:
                raise ValueError('printing boxes overlap')
            cells.update(part)
    return cells


def _as_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError('grid must be a 1..30 integer rectangle in colors 0..9')
    return a.astype(np.int8, copy=False)


def _analyze(value):
    grid = _as_grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[int(counts.argmax())])
    if counts.max() <= grid.size // 2:
        raise ValueError('carrier must be a strict input majority')
    h, w = grid.shape
    emitters = []
    for radius in RADII:
        if min(h, w) < 2 * radius + 1:
            continue
        shape = (h - 2 * radius, w - 2 * radius)
        dr, dc = _outline((0, 0), radius)[0]
        struct = grid[radius + dr:radius + dr + shape[0],
                      radius + dc:radius + dc + shape[1]]
        matches = struct != bg
        for dr, dc in _outline((0, 0), radius)[1:]:
            matches &= grid[radius + dr:radius + dr + shape[0],
                            radius + dc:radius + dc + shape[1]] == struct
        for rr, cc in np.argwhere(matches):
            r, c = int(rr) + radius, int(cc) + radius
            patch = grid[r - 1:r + 2, c - 1:c + 2]
            accents = set(map(int, np.unique(patch))) - {bg}
            if len(accents) != 1:
                continue
            accent = accents.pop()
            structural = int(struct[rr, cc])
            mask = patch == accent
            if accent == structural or not _full_box(mask):
                continue
            caps = {}
            for d in range(4):
                visible = [p for p in _cap((r, c), radius, d) if _inside(p, grid.shape)]
                if len(visible) >= 2:
                    values = {int(grid[p]) for p in visible}
                    if len(values) == 1 and not values & {bg, structural}:
                        caps[d] = values.pop()
            if caps:
                emitters.append(dict(center=(r, c), radius=radius, struct=structural,
                                     accent=accent, mask=mask.copy(), caps=caps))
    if not 1 <= len(emitters) <= 3:
        raise ValueError('expected one to three unambiguous emitters')
    protected, foreground, printed = set(), set(), set()
    for e in emitters:
        p = _protected(e, grid.shape)
        if p & protected:
            raise ValueError('input objects overlap')
        protected.update(p)
        foreground.update(_outline(e['center'], e['radius']))
        r, c = e['center']
        foreground.update((r - 1 + int(dr), c - 1 + int(dc))
                          for dr, dc in np.argwhere(e['mask']))
        for d in e['caps']:
            foreground.update(p for p in _cap(e['center'], e['radius'], d)
                              if _inside(p, grid.shape))
    if foreground != set(map(tuple, np.argwhere(grid != bg))):
        raise ValueError('unexplained input pixels')
    for e in emitters:
        p = _print_footprint(e, grid.shape)
        if p & (printed | protected):
            raise ValueError('printing boxes overlap a source or another stream')
        printed.update(p)
    return dict(background=bg, emitters=sorted(emitters, key=lambda e: e['center']),
                protected=protected, footprint=printed)


def _walk_render(grid, info, alternative=None):
    result = grid.copy()
    for e in info['emitters']:
        for d, color, endpoint, boxes in _streams(e, grid.shape, alternative == 'whole_cap'):
            if alternative == 'accent_color':
                color = e['accent']
            if alternative == 'thin_rays':
                if _inside(endpoint, grid.shape):
                    vr, vc = DIRECTIONS[d]
                    r, c = endpoint[0] + vr, endpoint[1] + vc
                    while _inside((r, c), grid.shape):
                        result[r, c] = color
                        r, c = r + vr, c + vc
                continue
            mask = np.ones((3, 3), dtype=bool) if alternative == 'solid_tiles' else e['mask']
            if alternative == 'single_tile':
                boxes = boxes[:1]
            for top, left in boxes:
                for dr, dc in np.argwhere(mask):
                    p = (top + int(dr), left + int(dc))
                    if _inside(p, grid.shape):
                        result[p] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _as_grid(input_grid)
    return _walk_render(grid, _analyze(grid))


def _coordinate_render(grid, info):
    """Quotient/remainder membership rather than stepping/stamping boxes."""
    result = grid.copy()
    rows, cols = np.indices(grid.shape)
    used = np.zeros(grid.shape, dtype=bool)
    for e in info['emitters']:
        for d, color in e['caps'].items():
            vr, vc = DIRECTIONS[d]
            cap = _cap(e['center'], e['radius'], d)
            for er, ec in (cap[0], cap[-1]):
                if not _inside((er, ec), grid.shape):
                    continue
                u, v = (rows - er) * vr - 1, (cols - ec) * vc - 1
                box = (u >= 0) & (v >= 0) & (u // 3 == v // 3)
                dr = u % 3 if vr > 0 else 2 - u % 3
                dc = v % 3 if vc > 0 else 2 - v % 3
                assert not np.any(used & box)
                used |= box
                result[box & e['mask'][dr, dc]] = color
    assert set(map(tuple, np.argwhere(used))) == info['footprint']
    return result


def _witnesses(grid, info):
    evidence = set()
    for e in info['emitters']:
        for d in e['caps']:
            cap = _cap(e['center'], e['radius'], d)
            if not all(_inside(p, grid.shape) for p in cap):
                continue
            streams = [_boxes(p, d, grid.shape) for p in (cap[0], cap[-1])]
            if all(len(bs) >= 2 and all(0 <= r <= grid.shape[0] - 3
                                       and 0 <= c <= grid.shape[1] - 3
                                       for r, c in bs[:2]) for bs in streams):
                evidence.add((d, e['radius']))
    return evidence


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer between {low} and {high}')
    return int(value)


def _choose_mask(rng, allowed):
    ids = np.flatnonzero(allowed)
    if not len(ids):
        raise ValueError('empty motif selection')
    density = int(rng.choice(np.unique(DENSITIES[ids])))
    return int(rng.choice(ids[DENSITIES[ids] == density]))


def _render_input(shape, bg, specs):
    grid = np.full(shape, bg, dtype=np.int8)
    for e in specs:
        for p in _outline(e['center'], e['radius']):
            grid[p] = e['struct']
        r, c = e['center']
        grid[r - 1:r + 2, c - 1:c + 2][e['mask']] = e['accent']
        for d, color in e['caps'].items():
            for p in _cap(e['center'], e['radius'], d):
                if _inside(p, shape):
                    grid[p] = color
    return grid


def _sample_pair(rng, max_radius, density_allowed, n, witness=None, witness_mask=None,
                 fixed_bg=None):
    """Bounded free-placement search; retries resample scene data, not the rule."""
    for restart in range(100):
        radii = [int(rng.integers(3, max_radius + 1)) for _ in range(n)]
        if witness is not None:
            radii[0] = max_radius
        if n == 3:
            radii = [min(r, 4) for r in radii]
        low = max(2 * max(radii) + 5, (11, 22, 27)[n - 1])
        if witness is not None:
            low = max(low, 2 * max_radius + 15)
        shape = tuple(int(rng.integers(low, 31)) for _ in range(2))
        bg = int(rng.integers(10)) if fixed_bg is None else fixed_bg
        palette = [c for c in range(10) if c != bg]
        specs, occupied, prints = [], set(), set()
        for i, radius in enumerate(radii):
            if i == 0 and witness is not None:
                directions = list(witness)
                mask = MASKS[witness_mask].copy()
            else:
                count = int(rng.choice((1, 2, 3, 4), p=(.44, .29, .17, .10)))
                directions = sorted(map(int, rng.choice(4, count, replace=False)))
                mask = MASKS[_choose_mask(rng, density_allowed)].copy()
            structural, accent = map(int, rng.choice(palette, 2, replace=False))
            choices = [c for c in palette if c != structural]
            bank_size = int(rng.integers(1, len(directions) + 1))
            bank = list(map(int, rng.choice(choices, bank_size, replace=False)))
            payloads = bank + [int(rng.choice(bank)) for _ in range(len(directions) - bank_size)]
            rng.shuffle(payloads)
            caps = dict(zip(directions, payloads))
            if witness is not None and i == 0 and all(c == accent for c in caps.values()):
                caps[directions[0]] = int(rng.choice([c for c in choices if c != accent]))
            candidates = [(r, c) for r in range(radius, shape[0] - radius)
                          for c in range(radius, shape[1] - radius)]
            # Clipped caps arise from ordinary near-edge coordinates in either split.
            rng.shuffle(candidates)
            if witness is None and rng.random() < .32:
                candidates.sort(key=lambda p: min(p[0], p[1], shape[0]-1-p[0], shape[1]-1-p[1]))
            for center in candidates:
                e = dict(center=center, radius=radius, struct=structural,
                         accent=accent, mask=mask, caps=caps)
                if any(sum(_inside(p, shape) for p in _cap(center, radius, d)) < 2 for d in caps):
                    continue
                prot = _protected(e, shape)
                fp = _print_footprint(e, shape)
                if not fp or prot & (occupied | prints) or fp & (occupied | prints | prot):
                    continue
                if i == 0 and witness is not None:
                    fake = dict(emitters=[e])
                    if not {(d, max_radius) for d in witness} <= _witnesses(np.empty(shape), fake):
                        continue
                specs.append(e)
                occupied |= prot
                prints |= fp
                break
            else:
                break
        if len(specs) != n:
            continue
        grid = _render_input(shape, bg, specs)
        try:
            info = _analyze(grid)
        except ValueError:
            continue
        if len(info['emitters']) != n:
            continue
        out = _walk_render(grid, info)
        if np.array_equal(grid, out):
            continue
        if witness is not None and any(np.array_equal(out, _walk_render(grid, info, alt))
                                       for alt in ('whole_cap', 'thin_rays', 'solid_tiles', 'single_tile', 'accent_color')):
            continue
        return {'input': grid, 'output': out}
    raise RuntimeError('bounded compatible scene search exhausted')


def _grid_key(grid):
    a = _as_grid(grid)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, 'num_train', 2, 10)
    ntest = _count(num_test, 'num_test', 1, 4)
    if ntrain <= ntest:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    maximum = int(rng.choice(RADII))
    regime = str(rng.choice(('denser', 'sparser', 'multiple_sources')))
    train_allowed = np.ones(len(MASKS), dtype=bool)
    test_allowed = train_allowed.copy()
    if regime == 'denser':
        train_allowed &= DENSITIES <= 5
        test_allowed &= DENSITIES >= 6
    elif regime == 'sparser':
        train_allowed &= DENSITIES >= 5
        test_allowed &= DENSITIES <= 4
    # Each directional witness independently distinguishes rotations/reflections.
    # Together the two masks also distinguish all 72 arbitrary axis orders.
    first = _choose_mask(rng, train_allowed & ASYMMETRIC)
    eligible = train_allowed & ASYMMETRIC & (DENSITIES != DENSITIES[first])
    eligible &= (AGREEMENT & AGREEMENT[first]).sum(axis=1) == 1
    second = _choose_mask(rng, eligible)
    first_directions = int(rng.integers(1, 15))
    second_directions = (15 ^ first_directions) | int(rng.integers(16))
    witness_data = [(tuple(d for d in range(4) if bits & (1 << d)), mask)
                    for bits, mask in ((first_directions, first), (second_directions, second))]
    slots = list(map(int, rng.choice(ntrain, 2, replace=False)))
    witnesses = dict(zip(slots, witness_data))
    # Carrier may remain stable, but never acts as an arbitrary output color code.
    fixed_bg = int(rng.integers(10)) if rng.random() < .3 else None
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for index in range(count):
            wd = witnesses.get(index) if split == 'train' else None
            if wd is not None or (regime == 'multiple_sources' and split == 'train'):
                n = 1
            elif regime == 'multiple_sources':
                n = int(rng.integers(2, 4))
            else:
                n = int(rng.choice((1, 2, 3), p=(.5, .33, .17)))
            allowed = train_allowed if split == 'train' else test_allowed
            for _ in range(12):
                pair = _sample_pair(rng, maximum, allowed, n,
                                    None if wd is None else wd[0],
                                    None if wd is None else wd[1], fixed_bg)
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('unique input search exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence = set(), set()
    surviving_orders = set(range(len(ORDERS)))
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _as_grid(pair['input']), _as_grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            info = _analyze(inp)
            assert np.array_equal(out, _walk_render(inp, info))
            assert np.array_equal(out, _coordinate_render(inp, info))
            bg = info['background']
            assert np.array_equal(out[inp != bg], inp[inp != bg])
            changed = set(map(tuple, np.argwhere(inp != out)))
            assert changed and changed <= info['footprint']
            # Every copied carrier gap is preserved, including clipped patches.
            for e in info['emitters']:
                for _, color, _, boxes in _streams(e, inp.shape):
                    for top, left in boxes:
                        for dr in range(3):
                            for dc in range(3):
                                p = top + dr, left + dc
                                if _inside(p, inp.shape):
                                    assert inp[p] == bg
                                    assert out[p] == (color if e['mask'][dr, dc] else bg)
            features[split].append(info)
            if split == 'train':
                for e in info['emitters']:
                    ew = _witnesses(inp, dict(emitters=[e]))
                    asymmetric = all(not np.array_equal(e['mask'], other)
                                     for other in [np.rot90(e['mask'], k) for k in (1, 2, 3)]
                                     + [np.rot90(e['mask'][:, ::-1], k) for k in range(4)])
                    if asymmetric:
                        evidence |= ew
                    if ew:
                        surviving_orders &= {k for k in surviving_orders
                                             if np.array_equal(e['mask'], _ordered(e['mask'], ORDERS[k]))}
    assert {d for d, _ in evidence} == set(range(4)), 'all directions need two complete copies per endpoint'
    test_max = max(e['radius'] for i in features['test'] for e in i['emitters'])
    assert all(max(r for dd, r in evidence if dd == d) >= test_max for d in range(4))
    assert len(surviving_orders) == 1, 'training must disambiguate motif orientation'
    for alt in ('whole_cap', 'thin_rays', 'solid_tiles', 'single_tile', 'accent_color'):
        assert any(not np.array_equal(_as_grid(p['output']),
                                     _walk_render(_as_grid(p['input']), info, alt))
                   for p, info in zip(puzzle['train'], features['train'])), alt
    train_density = [int(e['mask'].sum()) for i in features['train'] for e in i['emitters']]
    test_density = [int(e['mask'].sum()) for i in features['test'] for e in i['emitters']]
    assert len(set(train_density)) >= 2
    source_shift = (all(len(i['emitters']) == 1 for i in features['train'])
                    and all(len(i['emitters']) >= 2 for i in features['test']))
    assert (min(test_density) > max(train_density)
            or max(test_density) < min(train_density) or source_shift), 'missing visible test shift'


def _fingerprint(puzzle):
    return tuple((_grid_key(p['input']), _grid_key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def _measure(puzzle):
    rows = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            grid = _as_grid(pair['input'])
            info = _analyze(grid)
            boxes, clipped, zero = 0, 0, 0
            for e in info['emitters']:
                for d in e['caps']:
                    cap = _cap(e['center'], e['radius'], d)
                    clipped += any(not _inside(p, grid.shape) for p in cap)
                    zero += all(not bs for _, _, _, bs in _streams(dict(e, caps={d:e['caps'][d]}), grid.shape))
                boxes += sum(len(bs) for _, _, _, bs in _streams(e, grid.shape))
            rows.append(dict(split=split, shape=grid.shape, background=info['background'],
                             count=len(info['emitters']), radii=tuple(e['radius'] for e in info['emitters']),
                             densities=tuple(int(e['mask'].sum()) for e in info['emitters']),
                             subsets=tuple(sum(1 << d for d in e['caps']) for e in info['emitters']),
                             masks=tuple(e['mask'].tobytes() for e in info['emitters']),
                             boxes=boxes, clipped_caps=int(clipped), zero_caps=int(zero),
                             layout=(grid.shape, tuple((e['center'], e['radius'], e['mask'].tobytes(),
                                                       tuple(e['caps'])) for e in info['emitters']))))
    return rows


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    rows = [r for p in puzzles for r in _measure(p)]
    assert {r['count'] for r in rows} == {1, 2, 3}
    assert {n for r in rows for n in r['radii']} == set(RADII)
    assert {n for r in rows for n in r['densities']} == set(range(2, 9))
    assert {m for r in rows for m in r['subsets']} == set(range(1, 16))
    assert len({r['shape'] for r in rows}) >= 35
    assert len({r['layout'] for r in rows}) >= 200
    assert len({m for r in rows for m in r['masks']}) >= 90
    assert any(r['clipped_caps'] for r in rows) and any(r['zero_caps'] for r in rows)
    assert min(min(r['shape']) for r in rows) <= 14
    assert max(r['boxes'] for r in rows) >= 16
    assert {r['background'] for r in rows} == set(range(10))
    # Exercise every legal explicit count combination and invalid types/ranges.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + a * 10 + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Label mutation: a missing motif pixel and a filled carrier hole both fail.
    for fill_hole in (False, True):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ('train', 'test')}
        pair = bad['test'][0]
        info = _analyze(pair['input'])
        if fill_hole:
            cells = [p for p in info['footprint'] if pair['output'][p] == info['background']]
            assert cells
            cell = min(cells)
            pair['output'][cell] = next(iter(info['emitters'][0]['caps'].values()))
        else:
            cell = tuple(np.argwhere(pair['input'] != pair['output'])[0])
            pair['output'][cell] = pair['input'][cell]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('damaged motif label accepted')
