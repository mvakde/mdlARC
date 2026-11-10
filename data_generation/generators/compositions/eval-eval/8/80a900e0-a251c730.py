"""Keyed stencil cartridges printing inside endpoint-defined diagonal lanes.

Only NumPy and the standard library are used. No files or generation metadata
are consulted by solve. The self-test streams its seeds without saving grids.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
HOST_SEEDS = (0,144689643,386604314,56115810,861264606,145309674,158158804,383898981,14044603,919765913,783381871,28238077,762624523,396779819,634197512,406468737,476496050,266891264,440630848,413526889,214113889,464432334,414702663,12045437,102945135,119775555,452151115,998373711,615601119,722614603,608408789,839224121,376357708,515091663,486903580,137710906,538039612,223084018,135059038,386701279,407907546,38022583)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _outline(center, radius):
    r, c = center
    return {(r+dr, c+dc) for dr in range(-radius, radius+1)
            for dc in range(-radius, radius+1) if abs(dr)+abs(dc) == radius}


def _cap(center, radius, direction):
    r, c = center
    vr, vc = direction
    return [(r+vr*(k+1), c+vc*(radius-k+1)) for k in range(radius+1)]


def _base(shape, background):
    return np.asarray(background, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _connected(cells):
    if not cells:
        return False
    todo = set(cells)
    stack = [todo.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            p = r+dr, c+dc
            if p in todo:
                todo.remove(p)
                stack.append(p)
    return not todo


def _lane(point, emitter, direction):
    """Open strip beyond the cap, including both checkerboard parities."""
    vr, vc = direction
    u = (point[0]-emitter['center'][0])*vr
    v = (point[1]-emitter['center'][1])*vc
    return abs(u-v) < emitter['radius'] and u+v > emitter['radius']+2


def _direction_at(point, emitter):
    matches = [d for d in emitter['directions'] if _lane(point, emitter, d)]
    assert len(matches) <= 1
    return matches[0] if matches else None


def _rail_map(emitters, shape):
    claims = {}
    for e in emitters:
        for d in e['directions']:
            cap = _cap(e['center'], e['radius'], d)
            for endpoint in (cap[0], cap[-1]):
                p = endpoint[0]+d[0], endpoint[1]+d[1]
                while _inside(p, shape):
                    assert p not in claims, 'rails cannot intersect'
                    claims[p] = e['key']
                    p = p[0]+d[0], p[1]+d[1]
    return claims


def _footprint(anchor, emitter, direction, shape, clip=True):
    result = {}
    for dr, dc, color in emitter['template']:
        p = anchor[0]+dr, anchor[1]+dc
        if _inside(p, shape) and (not clip or _lane(p, emitter, direction)):
            result[p] = color
    return result


def _lateral_witness(full, emitter, direction, shape):
    """An unclipped canvas and no cap-plane cut isolate the side-wall effect."""
    vr, vc = direction
    uvs = [((r-emitter['center'][0])*vr, (c-emitter['center'][1])*vc) for r, c in full]
    return (all(_inside(p, shape) for p in full)
            and all(u+v > emitter['radius']+2 for u, v in uvs)
            and any(abs(u-v) >= emitter['radius'] for u, v in uvs))


def _read_input(input_grid):
    """Recover cartridges, their dictionaries, and all loose keys from pixels."""
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
    parity = np.indices(grid.shape).sum(axis=0) % 2
    background = []
    for bit in (0, 1):
        counts = np.bincount(grid[parity == bit].astype(int), minlength=10)
        color = int(counts.argmax())
        assert counts[color] > np.count_nonzero(parity == bit)/2
        background.append(color)
    assert background[0] != background[1]
    base = _base(grid.shape, background)
    foreground = set(map(tuple, np.argwhere(grid != base)))
    emitters = []
    # Complete uniform outlines. A core has fewer cells than a radius-3 outline.
    for radius in (3, 4):
        if min(grid.shape) < 2*radius+1:
            continue
        sh = grid.shape[0]-2*radius, grid.shape[1]-2*radius
        offsets = sorted(_outline((0, 0), radius))
        dr, dc = offsets[0]
        colors = grid[radius+dr:radius+dr+sh[0], radius+dc:radius+dc+sh[1]]
        mask = (colors != background[0]) & (colors != background[1])
        for dr, dc in offsets[1:]:
            mask &= grid[radius+dr:radius+dr+sh[0], radius+dc:radius+dc+sh[1]] == colors
        for rr, cc in np.argwhere(mask):
            emitters.append({'center': (int(rr)+radius, int(cc)+radius),
                             'radius': radius, 'struct': int(colors[rr, cc])})
    assert 1 <= len(emitters) <= 2
    covered = set()
    for e in emitters:
        cells = _outline(e['center'], e['radius'])
        assert not covered & cells
        covered |= cells
        directions, cap_colors = [], []
        for d in DIRECTIONS:
            cap = _cap(e['center'], e['radius'], d)
            if not all(_inside(p, grid.shape) for p in cap):
                continue
            values = [int(grid[p]) for p in cap]
            if len(set(values)) == 1 and values[0] not in (*background, e['struct']):
                assert not covered.intersection(cap)
                directions.append(d)
                cap_colors.append(values[0])
                covered.update(cap)
        assert directions and len(set(cap_colors)) == 1
        e['directions'], e['key'] = tuple(directions), cap_colors[0]
        cr, cc = e['center']
        core = {p for p in foreground if abs(p[0]-cr)+abs(p[1]-cc) < e['radius']}
        assert 3 <= len(core) <= 10 and _connected(core)
        keys = [p for p in core if int(grid[p]) == e['key']]
        assert len(keys) == 1
        ar, ac = keys[0]
        e['source_anchor'] = (ar, ac)
        e['template'] = tuple(sorted((r-ar, c-ac, int(grid[r, c])) for r, c in core))
        assert not covered & core
        covered |= core
    assert len({e['key'] for e in emitters}) == len(emitters)
    key_set = {e['key'] for e in emitters}
    structs = {e['struct'] for e in emitters}
    assert not key_set & structs
    for e in emitters:
        assert all(color not in key_set | structs | set(background)
                   for dr, dc, color in e['template'] if (dr, dc) != (0, 0))
    anchors = sorted((r, c, int(grid[r, c])) for r, c in foreground-covered)
    assert anchors and all(k in key_set for _, _, k in anchors)
    points = {(r, c) for r, c, _ in anchors}
    for r, c in points:
        assert not any((r+dr, c+dc) in points | covered
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                       if (dr, dc) != (0, 0))
    return {'background': tuple(background), 'base': base, 'emitters': emitters,
            'anchors': anchors, 'fixed': covered}


def _predict(grid, info):
    rails = _rail_map(info['emitters'], grid.shape)
    output = grid.copy()
    for p, key in rails.items():
        assert p not in info['fixed']
        assert grid[p] == info['base'][p] or int(grid[p]) == key
        output[p] = key
    by_key = {e['key']: e for e in info['emitters']}
    written = set()
    for r, c, key in info['anchors']:
        e = by_key[key]
        d = _direction_at((r, c), e)
        if d is None:
            continue
        footprint = _footprint((r, c), e, d, grid.shape)
        assert len(footprint) >= 2, 'an active anchor retains some decoration'
        for p, color in footprint.items():
            assert p not in info['fixed'] and p not in rails and p not in written
            assert p == (r, c) or grid[p] == info['base'][p]
            output[p] = color
            written.add(p)
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Extend rails and print the visible dictionary through their apertures."""
    grid = np.asarray(input_grid)
    return _predict(grid, _read_input(grid))


def _closed_prediction(grid, info):
    """Pixel/offset formulation; shares parsing, but does not walk endpoint rays."""
    out = grid.copy()
    rows, cols = np.indices(grid.shape)
    for e in info['emitters']:
        for vr, vc in e['directions']:
            u, v = (rows-e['center'][0])*vr, (cols-e['center'][1])*vc
            out[(np.abs(u-v) == e['radius']) & (np.minimum(u, v) >= 2)] = e['key']
            aperture = (np.abs(u-v) < e['radius']) & (u+v > e['radius']+2)
            for r, c, key in info['anchors']:
                if key != e['key'] or not aperture[r, c]:
                    continue
                for dr, dc, color in e['template']:
                    rr, cc = r+dr, c+dc
                    if _inside((rr, cc), grid.shape) and aperture[rr, cc]:
                        out[rr, cc] = color
    return out


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _sample_core(rng, radius, key, decors):
    domain = {(r, c) for r in range(1-radius, radius)
              for c in range(1-radius, radius) if abs(r)+abs(c) < radius}
    count = int(rng.integers(3, min(10, len(domain))+1))
    cells = {_pick(rng, sorted(domain))}
    while len(cells) < count:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in N4} & domain - cells)
        cells.add(_pick(rng, frontier))
    anchor = _pick(rng, sorted(cells))
    # All colors are visible offsets; decoration may be shared by cartridges.
    colored = {p: int(_pick(rng, decors)) for p in sorted(cells)}
    colored[anchor] = key
    template = tuple(sorted((r-anchor[0], c-anchor[1], color) for (r, c), color in colored.items()))
    return anchor, colored, template


def _asymmetric_with_gap(template):
    cells = {(r, c) for r, c, _ in template}
    rows, cols = zip(*cells)
    gap = (max(rows)-min(rows)+1)*(max(cols)-min(cols)+1) > len(cells)
    return gap and cells != {(-c, r) for r, c in cells} and cells != {(r, -c) for r, c in cells}


def _specs(rng, n, shape, max_radius, stage, palette):
    background = tuple(map(int, palette[:2]))
    structs = [int(palette[2]), int(palette[2 if rng.random() < .5 else 3])]
    keys = list(map(int, palette[4:4+n]))
    decors = list(map(int, palette[6:int(rng.integers(7, 10))+1]))
    emitters, fixed = [], {}
    for i in range(n):
        radius = max_radius if stage == 'four_way' else int(rng.integers(3, max_radius+1))
        if stage == 'four_way':
            center = (int(rng.integers(11, shape[0]-10)), int(rng.integers(11, shape[1]-10)))
            directions = DIRECTIONS
        else:
            center = tuple(int(rng.integers(radius+1, extent-radius-1)) for extent in shape)
            count = 1 if stage == 'two_keys' else int(rng.integers(1, 4))
            if stage == 'combine':
                count = int(rng.integers(2, 4))
            directions = tuple(DIRECTIONS[int(j)] for j in sorted(rng.choice(4, count, replace=False)))
        anchor, core, template = _sample_core(rng, radius, keys[i], decors)
        if stage == 'four_way':
            assert _asymmetric_with_gap(template) and anchor != (0, 0)
        cr, cc = center
        local = {p: structs[i] for p in _outline(center, radius)}
        local.update({(cr+r, cc+c): color for (r, c), color in core.items()})
        for d in directions:
            local.update({p: keys[i] for p in _cap(center, radius, d)})
        assert not set(local) & set(fixed)
        fixed.update(local)
        emitters.append({'center': center, 'radius': radius, 'struct': structs[i],
                         'key': keys[i], 'directions': directions, 'template': template,
                         'source_anchor': (cr+anchor[0], cc+anchor[1])})
    rails = _rail_map(emitters, shape)
    assert not set(rails) & set(fixed)
    assert len(rails) >= 8
    return background, emitters, fixed, rails


def _construct(rng, n, max_radius, stage, palette):
    """Bounded rejection of incompatible geometry, never altered answer rules."""
    for attempt in range(240):
        shape = tuple(int(rng.integers(25 if n == 2 else 24, 31)) for _ in range(2))
        try:
            background, emitters, fixed, rails = _specs(rng, n, shape, max_radius, stage, palette)
            grid = _base(shape, background)
            for p, color in fixed.items():
                grid[p] = color
            anchors, writes = {}, {}
            fixed_set = set(fixed)
            near_fixed = {(r+dr, c+dc) for r, c in fixed_set
                          for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            candidates = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
            rng.shuffle(candidates)

            def place(e, kind, direction=None):
                for p in candidates:
                    if p in near_fixed or p in writes or p in anchors:
                        continue
                    if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1 for q in anchors):
                        continue
                    d = _direction_at(p, e)
                    if kind in ('full', 'partial', 'active'):
                        if d is None or (direction is not None and d != direction) or p in rails:
                            continue
                        foot = _footprint(p, e, d, shape)
                        full = {(p[0]+dr, p[1]+dc) for dr, dc, _ in e['template']}
                        if len(foot) < 2 or set(foot) & (fixed_set | set(rails) | set(writes) | set(anchors)):
                            continue
                        if kind == 'full' and len(foot) != len(full):
                            continue
                        if kind == 'full' and stage == 'four_way' and (sum(p)-sum(e['source_anchor'])) % 2 != 1:
                            continue
                        if kind == 'partial' and not _lateral_witness(full, e, d, shape):
                            continue
                    else:
                        if d is not None:
                            continue
                        if kind == 'wall':
                            if rails.get(p) != e['key']:
                                continue
                        elif p in rails:
                            continue
                        if kind == 'foreign' and not any(other['key'] != e['key'] and
                                _direction_at(p, other) is not None for other in emitters):
                            continue
                        foot = {}
                    anchors[p] = e['key']
                    writes.update(foot)
                    return True
                return False

            # These two witnesses expose copying and clipping even at 2/1 counts.
            for e in emitters:
                assert place(e, 'full')
            assert place(emitters[0], 'partial')
            # Each active cap has at least one active target.
            for e in emitters:
                for d in e['directions']:
                    if not any(k == e['key'] and _lane(p, e, d) for p, k in anchors.items()):
                        assert place(e, 'active', d)
                    if stage == 'combine' and rng.random() < .4:
                        place(e, 'active', d)
            assert place(emitters[0], 'wall')
            assert place(emitters[0], 'outside')
            if n == 2:
                assert place(_pick(rng, emitters), 'foreign')
            for _ in range(int(rng.integers(3))):
                place(_pick(rng, emitters), 'outside')
            for p, key in anchors.items():
                grid[p] = key
            output = grid.copy()
            for p, key in rails.items():
                output[p] = key
            for p, color in writes.items():
                output[p] = color
            info = _read_input(grid)
            assert np.array_equal(_predict(grid, info), output)
            features = _features(grid, info)
            assert features['full'] and features['rail_clipped'] and features['inactive'] and features['wall']
            assert features['unprinted_interior'] >= 2
            if stage == 'four_way':
                assert features['transparency'] and features['endpoint_witnesses'] == set(DIRECTIONS)
            if n == 2:
                assert features['foreign'] and len(features['geometries']) == 2
            return {'input': grid, 'output': output}
        except AssertionError:
            continue
    raise RuntimeError('bounded cartridge layout search exhausted')


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    max_radius = int(rng.integers(3, 5))
    stable_palette = bool(rng.random() < .3)
    palette = rng.permutation(10)
    stages = [('four_way', 1), ('two_keys', 2)]
    for _ in range(num_train-2):
        stages.append(('free', 1) if rng.random() < .5 else ('two_keys', 2))
    rng.shuffle(stages)
    result, seen = {'train': [], 'test': []}, set()
    for split, specs in (('train', stages), ('test', [('combine', 2)]*num_test)):
        for stage, n in specs:
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for _ in range(6):
                pair_palette = palette if stable_palette else pair_rng.permutation(10)
                pair = _construct(pair_rng, n, max_radius, stage, pair_palette)
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded unique-input search exhausted')
    validate_generated_puzzle(result)
    return result


def _features(grid, info=None):
    info = _read_input(grid) if info is None else info
    rails = _rail_map(info['emitters'], grid.shape)
    out = _predict(grid, info)
    by_key = {e['key']: e for e in info['emitters']}
    counts = Counter()
    geometries, sizes, radii, masks, directions, endpoint_witnesses = set(), set(), set(), [], set(), set()
    for e in info['emitters']:
        geometries.add(tuple((r, c) for r, c, _ in e['template']))
        sizes.add(len(e['template']))
        radii.add(e['radius'])
        masks.append(sum(1 << DIRECTIONS.index(d) for d in e['directions']))
        directions.update(e['directions'])
    for r, c, key in info['anchors']:
        p = (r, c)
        e = by_key[key]
        d = _direction_at(p, e)
        if d is None:
            counts['inactive'] += 1
            counts['wall'] += int(rails.get(p) == key)
            counts['foreign'] += int(any(other['key'] != key and _direction_at(p, other) is not None
                                         for other in info['emitters']))
        else:
            counts['active'] += 1
            full = {(r+dr, c+dc) for dr, dc, _ in e['template']}
            foot = _footprint(p, e, d, grid.shape)
            counts['full'] += int(len(foot) == len(full))
            counts['rail_clipped'] += int(_lateral_witness(full, e, d, grid.shape))
            counts['canvas_clipped'] += int(not all(_inside(q, grid.shape) for q in full))
            if len(foot) == len(full) and (r+c-sum(e['source_anchor'])) % 2 == 1:
                low_r, high_r = min(p[0] for p in full), max(p[0] for p in full)
                low_c, high_c = min(p[1] for p in full), max(p[1] for p in full)
                gaps = {(rr, cc) for rr in range(low_r, high_r+1) for cc in range(low_c, high_c+1)}-full
                counts['transparency'] += int(any(out[p] == grid[p] == info['base'][p] for p in gaps))
    # Unextended interior-cap lanes distinguish two endpoint rails from a beam.
    for e in info['emitters']:
        for d in e['directions']:
            unchanged = 0
            for r, c in _cap(e['center'], e['radius'], d)[1:-1]:
                for step in (1, 2):
                    p = r+step*d[0], c+step*d[1]
                    if _inside(p, grid.shape) and grid[p] == info['base'][p] and out[p] == grid[p]:
                        counts['unprinted_interior'] += 1
                        unchanged += 1
            cap = _cap(e['center'], e['radius'], d)
            if unchanged and all(_inside((p[0]+2*d[0], p[1]+2*d[1]), grid.shape) for p in (cap[0], cap[-1])):
                endpoint_witnesses.add(d)
    result = dict(counts)
    result.update({'emitters': len(info['emitters']), 'anchors': len(info['anchors']), 'rails': len(rails),
                   'copied_cells': int(np.count_nonzero((out != grid) & ~np.array(
                       [[(r, c) in rails for c in range(grid.shape[1])] for r in range(grid.shape[0])]))),
                   'geometries': geometries, 'sizes': sizes, 'radii': radii,
                   'masks': masks, 'directions': directions, 'endpoint_witnesses': endpoint_witnesses,
                   'background': info['background']})
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, demonstrated, teaching = set(), set(), Counter()
    max_train_radius, max_test_radius = 0, 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            info = _read_input(inp)
            assert np.array_equal(out, _predict(inp, info))
            assert np.array_equal(out, _closed_prediction(inp, info))
            foreground = inp != info['base']
            assert np.array_equal(inp[foreground], out[foreground]), 'preserve every input foreground cell'
            f = _features(inp, info)
            assert f['full'] and f['rail_clipped'] and f['inactive'] and f['wall']
            assert f['copied_cells'] >= 2 and f['rails'] >= 8 and f['unprinted_interior'] >= 2
            # No-copy ablation is precisely the rails-only image.
            no_b = inp.copy()
            for p, color in _rail_map(info['emitters'], inp.shape).items():
                no_b[p] = color
            assert np.any(no_b != out)
            # No-aperture ablation: at least one omitted on-grid motif pixel
            # beside a clipped active anchor would be added by unrestricted B.
            clip_witness = False
            for r, c, k in info['anchors']:
                e = next(e for e in info['emitters'] if e['key'] == k)
                d = _direction_at((r, c), e)
                if d is not None:
                    clip_witness |= any(_inside((r+dr, c+dc), inp.shape)
                        and not _lane((r+dr, c+dc), e, d) and out[r+dr, c+dc] != color
                        for dr, dc, color in e['template'])
            assert clip_witness, 'aperture clipping must visibly affect the answer'
            if split == 'train':
                demonstrated |= f['endpoint_witnesses']
                for name in ('full', 'rail_clipped', 'inactive', 'foreign', 'wall', 'transparency'):
                    teaching[name] += f.get(name, 0)
                teaching['two_distinct_keys'] += int(f['emitters'] == 2 and len(f['geometries']) == 2)
                max_train_radius = max(max_train_radius, max(f['radii']))
            else:
                max_test_radius = max(max_test_radius, max(f['radii']))
    assert demonstrated == set(DIRECTIONS)
    assert all(teaching[name] for name in ('full', 'rail_clipped', 'inactive', 'foreign', 'wall', 'transparency', 'two_distinct_keys'))
    assert max_train_radius >= max_test_radius


def self_test():
    """Stream 42 seeds; retain only hashes and small feature sets (<1 MB)."""
    signatures, geometries, shapes, palettes, masks, sizes, radii = set(), set(), set(), set(), set(), set(), set()
    totals, split_counts = Counter(), {'train': set(), 'test': set()}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _read_input(pair['input'])
                f = _features(pair['input'], info)
                split_counts[split].add(f['emitters'])
                geometries |= f['geometries']
                masks.update(f['masks'])
                sizes |= f['sizes']
                radii |= f['radii']
                shapes.add(pair['input'].shape)
                palettes.add(tuple(map(int, np.unique(pair['input']))))
                for name in ('full', 'rail_clipped', 'canvas_clipped', 'inactive', 'foreign', 'wall', 'copied_cells'):
                    totals[name] += f.get(name, 0)
                digest.update(str(pair['input'].shape).encode())
                digest.update(pair['input'].tobytes())
                # Same transformation is equivariant under whole-scene D4.
                if seed == 0:
                    for turns in range(4):
                        for flip in (False, True):
                            inp = np.rot90(pair['input'], turns)
                            out = np.rot90(pair['output'], turns)
                            if flip:
                                inp, out = inp[:, ::-1], out[:, ::-1]
                            assert np.array_equal(solve(inp), out)
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(geometries) >= 100
    assert len(shapes) >= 25 and len(palettes) >= 15
    assert radii == {3, 4} and set(range(3, 11)) <= sizes
    assert set(range(1, 16)) <= masks
    assert split_counts['train'] == {1, 2} and split_counts['test'] == {2}
    assert all(totals[name] >= 42 for name in ('full', 'rail_clipped', 'inactive', 'foreign', 'wall'))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(17, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    first, again = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    assert all(_grid_key(x[k]) == _grid_key(y[k]) for s in first
               for x, y in zip(first[s], again[s]) for k in ('input', 'output'))
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('accepted illegal counts')
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    p = broken['test'][0]
    changed = tuple(np.argwhere(p['input'] != p['output'])[0])
    p['output'][changed] = p['input'][changed]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a deleted output pixel')
    return {'seeds': 42, 'pairs': 210, 'geometries': len(geometries), 'shapes': len(shapes),
            'palettes': len(palettes), 'masks': sorted(masks), 'sizes': sorted(sizes),
            'radii': sorted(radii), 'witness_totals': dict(totals)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
