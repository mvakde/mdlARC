"""Ranked melting cascade: geometry addresses conserved liquid charges.

The solver reads only pixels. The rimmed chambers stay in their scattered
positions; source squares melt. Ordinal overflow channels are a fixed rule,
not inferred physical paths through the surrounding background.
"""
from itertools import combinations

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _parts(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], []
        while stack:
            r, c = stack.pop()
            part.append((int(r), int(c)))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(part)
    return result


def _chamber(size, color, bg, raised=()):
    tile = np.full((size, size), color, dtype=np.int8)
    tile[1:-1, 1:-1] = bg
    if raised:
        for col in raised:
            tile[-3:-1, 1+col] = color
    return tile


def _analyze(value):
    grid = _grid(value)
    bg = int(grid[0, 0])
    if not (np.all(grid[0] == bg) and np.all(grid[-1] == bg)
            and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)):
        raise ValueError("the entire outer border must be background")
    sources, cups = [], []
    for part in _parts(grid != bg):
        top, left = min(r for r, _ in part), min(c for _, c in part)
        bottom, right = max(r for r, _ in part)+1, max(c for _, c in part)+1
        size = bottom-top
        if size != right-left:
            raise ValueError("each object has a square bounding box")
        tile = grid[top:bottom, left:right].copy()
        color = int(tile[0, 0])
        if color == bg or np.any((tile != bg) & (tile != color)):
            raise ValueError("objects must be monochrome")
        neighborhood = grid[top-1:bottom+1, left-1:right+1].copy()
        neighborhood[1:-1, 1:-1] = bg
        if np.any(neighborhood != bg):
            raise ValueError("objects require a background gap, including diagonals")
        obj = dict(top=top, left=left, size=size, color=color, tile=tile)
        if np.all(tile == color):
            if size not in (2, 4):
                raise ValueError("solid charges have side 2 or 4")
            obj['volume'] = size*size
            sources.append(obj)
        else:
            if size not in (4, 6):
                raise ValueError("chambers have side 4 or 6")
            profiles = [()] + (list(combinations(range(4), 2)) if size == 6 else [])
            matches = [p for p in profiles if np.array_equal(tile, _chamber(size, color, bg, p))]
            if len(matches) != 1:
                raise ValueError("invalid rim or internal step profile")
            obj['raised'] = tuple(matches[0])
            obj['capacity'] = int(np.count_nonzero(tile == bg))
            cups.append(obj)
    if not sources or not cups or len(sources) > len(cups):
        raise ValueError("there must be a receiving rank for every charge")
    for queue in (sources, cups):
        if len({o['top'] for o in queue}) != len(queue):
            raise ValueError("top-row ties within a class are outside the family")
        queue.sort(key=lambda o: o['top'])
    liquids = {o['color'] for o in sources}
    if len(liquids) != 1:
        raise ValueError("all solid charges share one liquid color")
    return dict(grid=grid, background=bg, liquid=next(iter(liquids)), sources=sources, cups=cups)


def _cascade(a, arrivals=None, cups=None, require_contained=True):
    """Allocate complete bottom-up levels, passing only excess to later ranks."""
    cups = a['cups'] if cups is None else cups
    if arrivals is None:
        arrivals = [o['volume'] for o in a['sources']]
    out = a['grid'].copy()
    for o in a['sources']:
        r, c, s = o['top'], o['left'], o['size']
        out[r:r+s, c:c+s] = a['background']
    carry = 0
    records = []
    for rank, cup in enumerate(cups):
        injection = int(arrivals[rank]) if rank < len(arrivals) else 0
        incoming = carry + injection
        volume = min(incoming, cup['capacity'])
        carry = incoming-volume
        cavity = cup['tile'] == a['background']
        mask = np.zeros(cavity.shape, dtype=bool)
        remaining = volume
        for row in range(cup['size']-2, 0, -1):
            width = int(cavity[row].sum())
            if remaining >= width:
                mask[row] = cavity[row]
                remaining -= width
            elif remaining:
                raise ValueError("a partial horizontal level would be ambiguous")
        if remaining:
            raise ValueError("volume exceeds cavity")
        r, c, s = cup['top'], cup['left'], cup['size']
        target = out[r:r+s, c:c+s]
        target[mask] = a['liquid']
        records.append(dict(volume=volume, injected=injection, incoming=incoming,
                            overflow=carry, mask=mask))
    if carry and require_contained:
        raise ValueError("the last chamber cannot contain the remaining overflow")
    return out, records, carry


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve every valid family member without seed, examples, or metadata."""
    return _cascade(_analyze(input_grid))[0]


def _alternative(a, mode):
    """Concrete shortcut diagnostics, not an independent reference solver."""
    cups = a['cups']
    sources = a['sources']
    volumes = [o['volume'] for o in sources]
    if mode == 'pooled':
        volumes = [sum(volumes)]
    elif mode.startswith('source_x'):
        sources = sorted(sources, key=lambda o: (o['left'], o['top']), reverse=mode.endswith('reverse'))
        volumes = [o['volume'] for o in sources]
    elif mode.startswith('cup_x'):
        cups = sorted(cups, key=lambda o: (o['left'], o['top']), reverse=mode.endswith('reverse'))
    elif mode == 'nearest':
        volumes = [0]*len(cups)
        unused = set(range(len(cups)))
        for source in sources:
            sr = source['top'] + (source['size']-1)/2
            sc = source['left'] + (source['size']-1)/2
            idx = min(unused, key=lambda i: (
                abs(sr-cups[i]['top']-(cups[i]['size']-1)/2)
                + abs(sc-cups[i]['left']-(cups[i]['size']-1)/2), i))
            unused.remove(idx)
            volumes[idx] = source['volume']
    return _cascade(a, volumes, cups, require_contained=False)[0]


def _witness(a, out):
    return all(not np.array_equal(out, _alternative(a, mode)) for mode in
               ('pooled', 'source_x', 'source_x_reverse', 'cup_x', 'cup_x_reverse', 'nearest'))


def _layout(rng, objects):
    """Make a feasible packing, then relocate each square over free pixels."""
    order = sorted(range(len(objects)), key=lambda i: (-objects[i]['size'], float(rng.random())))
    for attempt in range(25):
        width = int(rng.integers(8, 31)) if attempt < 24 else 30
        if max(o['size'] for o in objects) > width-2:
            continue
        positions = [None]*len(objects)
        row, col, row_count, bottom = 1, 1, 0, 1
        for idx in order:
            s = objects[idx]['size']
            if col+s > width-1:
                row, col, row_count = bottom+1, 1, 0
            r = row+row_count
            positions[idx] = (r, col)
            bottom = max(bottom, r+s)
            col += s+1
            row_count += 1
        if bottom+1 <= 30:
            height = int(rng.integers(bottom+1, 31))
            break
    else:
        raise AssertionError("bounded packing failed")
    for _ in range(3):
        for idx in rng.permutation(len(objects)):
            idx = int(idx)
            s = objects[idx]['size']
            rr, cc = np.indices((height-s-1, width-s-1))
            rr, cc = rr+1, cc+1
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(positions):
                if j == idx:
                    continue
                t = objects[j]['size']
                legal &= ((rr+s < r) | (rr > r+t) | (cc+s < c) | (cc > c+t))
                if objects[idx]['kind'] == objects[j]['kind']:
                    legal &= rr != r
            sites = np.argwhere(legal)
            if not len(sites):
                raise AssertionError("current location should remain legal")
            r, c = sites[int(rng.integers(len(sites)))]
            positions[idx] = (int(r)+1, int(c)+1)
    # Check rank uniqueness after free placement, not only at shelf slots.
    for kind in ('cup', 'source'):
        indices = [i for i, o in enumerate(objects) if o['kind'] == kind]
        # Objects of different sizes cannot simply swap rectangles. Their
        # requested order is checked by the caller or supplied in a fallback.
        assert len({positions[i][0] for i in indices}) == len(indices)
    return (height, width), positions


def _draw(bg, liquid, cup_specs, source_sizes, cup_colors, positions, shape):
    grid = np.full(shape, bg, dtype=np.int8)
    specs = [('cup', s, profile, color) for (s, profile), color in zip(cup_specs, cup_colors)]
    specs += [('source', s, (), liquid) for s in source_sizes]
    for (kind, s, profile, color), (r, c) in zip(specs, positions):
        grid[r:r+s, c:c+s] = (_chamber(s, color, bg, profile) if kind == 'cup'
                             else np.full((s, s), color, dtype=np.int8))
    return grid


def _place(rng, bg, liquid, cup_specs, source_sizes, cup_colors, ordered=False):
    objects = [dict(size=s, kind='cup') for s, _ in cup_specs]
    objects += [dict(size=s, kind='source') for s in source_sizes]
    if ordered:
        # Reserve equal bounding squares so assigning freely scattered slots
        # in each queue's desired rank cannot create a collision. Actual
        # objects may be smaller; their sizes never change in the drawn input.
        reserve = [dict(size=6, kind=o['kind']) for o in objects]
        shape, positions = _layout(rng, reserve)
        for kind in ('cup', 'source'):
            indices = [i for i, o in enumerate(objects) if o['kind'] == kind]
            slots = sorted([positions[i] for i in indices])
            for i, slot in zip(indices, slots):
                positions[i] = slot
    else:
        shape, positions = _layout(rng, objects)
    grid = _draw(bg, liquid, cup_specs, source_sizes, cup_colors, positions, shape)
    a = _analyze(grid)
    if ordered:
        if ([(o['size'], o['raised']) for o in a['cups']] != cup_specs
                or [o['size'] for o in a['sources']] != source_sizes):
            return None
    return grid


def _colors(rng, bg, liquid, count, shared=False):
    palette = [c for c in range(10) if c != bg]
    result = [int(c) for c in rng.choice(palette, size=count)]
    if shared:
        result[0] = liquid
    return result


def _first_pair(rng, bg, liquid):
    # Volumes [4,16,16], capacities [16,4,16,16]. Result [4,4,16,12]:
    # two spills, a partially filled first cup, and an unpaired wet last cup.
    specs, sizes = [(6, ()), (4, ()), (6, ()), (6, ())], [2, 4, 4]
    colors = _colors(rng, bg, liquid, 4, shared=True)
    # Free layouts need the two geometry ranks and observable x alternatives.
    for _ in range(80):
        grid = _place(rng, bg, liquid, specs, sizes, colors, ordered=True)
        if grid is None:
            continue
        a = _analyze(grid)
        out = _cascade(a)[0]
        if _witness(a, out):
            return dict(input=grid, output=out)
    # Guaranteed geometry witness; recolored per puzzle, never a parent pair.
    positions = [(1, 1), (2, 10), (9, 16), (16, 2), (8, 10), (15, 24), (23, 9)]
    grid = _draw(bg, liquid, specs, sizes, colors, positions, (28, 29))
    a = _analyze(grid)
    out = _cascade(a)[0]
    assert _witness(a, out)
    return dict(input=grid, output=out)


def _second_pair(rng, bg, liquid):
    # The equal four-cell charges expose the difference between a flat floor
    # and a step; a third chamber remains dry. No overflow obscures the lesson.
    profile = tuple(map(int, rng.choice(4, size=2, replace=False)))
    profile = tuple(sorted(profile))
    specs = [(6, profile), (6, ()), (int(rng.choice([4, 6])), ())]
    colors = _colors(rng, bg, liquid, 3)
    for _ in range(40):
        grid = _place(rng, bg, liquid, specs, [2, 2], colors, ordered=True)
        if grid is not None:
            return dict(input=grid, output=solve(grid))
    positions = [(1, 1), (9, 10), (17, 2), (2, 12), (20, 14)]
    grid = _draw(bg, liquid, specs, [2, 2], colors, positions, (25, 24))
    return dict(input=grid, output=solve(grid))


def _random_pair(rng, bg, liquid, combined):
    profiles = list(combinations(range(4), 2))
    for _ in range(100):
        nc = int(rng.integers(3 if combined else 1, 6))
        ns = int(rng.integers(2 if combined else 1, min(nc, 4)+1))
        cup_specs = []
        for _ in range(nc):
            size = 4 if rng.random() < .25 else 6
            raised = profiles[int(rng.integers(6))] if size == 6 and rng.random() < .55 else ()
            cup_specs.append((size, raised))
        sizes = [int(s) for s in rng.choice([2, 4], size=ns, p=[.65, .35])]
        if combined and (len(set(sizes)) != 2 or not any(p for _, p in cup_specs)):
            continue
        colors = _colors(rng, bg, liquid, nc)
        grid = _place(rng, bg, liquid, cup_specs, sizes, colors)
        a = _analyze(grid)
        try:
            out, records, _ = _cascade(a)
        except ValueError:
            continue
        if combined and not any(c['raised'] and r['overflow']
                                for c, r in zip(a['cups'], records)):
            continue
        return dict(input=grid, output=out)
    # A combined construction which does not depend on rejection succeeding.
    specs = [(6, ()), (4, ()), (6, (0, 1)), (6, ())]
    colors = _colors(rng, bg, liquid, 4)
    positions = [(1, 1), (2, 10), (9, 16), (16, 2), (8, 10), (15, 24), (23, 9)]
    grid = _draw(bg, liquid, specs, [2, 4, 4], colors, positions, (28, 29))
    # Capacities [16,4,12,16] receive [4,4,12,16], with no lost cells.
    return dict(input=grid, output=solve(grid))


def _key(value):
    a = np.asarray(value, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if not _integer(value) or not lo <= int(value) <= hi:
            raise ValueError("illegal pair count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    bg, liquid = map(int, rng.choice(10, size=2, replace=False))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for idx in range(count):
            for _ in range(20):
                if split == 'train' and idx == 0:
                    pair = _first_pair(rng, bg, liquid)
                elif split == 'train' and idx == 1:
                    pair = _second_pair(rng, bg, liquid)
                else:
                    pair = _random_pair(rng, bg, liquid, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded duplicate resampling failed")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    a = _analyze(inp)
    expected, records, carry = _cascade(a)
    assert np.array_equal(out, expected), 'incorrect cascade output'
    assert inp.shape == out.shape and np.any(inp != out)
    assert carry == 0
    movable = np.zeros(inp.shape, dtype=bool)
    occupied = np.zeros(inp.shape, dtype=bool)
    for source in a['sources']:
        r, c, s = source['top'], source['left'], source['size']
        movable[r:r+s, c:c+s] = True
        assert np.all(out[r:r+s, c:c+s] == a['background'])
    total = 0
    for cup, record in zip(a['cups'], records):
        r, c, s = cup['top'], cup['left'], cup['size']
        tile = out[r:r+s, c:c+s]
        terrain = cup['tile'] != a['background']
        assert np.array_equal(tile[terrain], cup['tile'][terrain]), 'terrain moved'
        water = (tile == a['liquid']) & ~terrain
        assert int(water.sum()) == record['volume']
        total += int(water.sum())
        occupied[r:r+s, c:c+s] |= water
        # Flat levels, complete low-to-high filling, no wall penetration.
        last_wet = False
        for row in range(1, s-1):
            free = ~terrain[row]
            if free.any():
                count = int(water[row].sum())
                assert count in (0, int(free.sum()))
                assert not last_wet or count == int(free.sum())
                last_wet |= bool(count)
        assert not np.any(water & terrain)
    assert total == sum(o['volume'] for o in a['sources']), 'lost liquid volume'
    assert int((inp == a['liquid']).sum()) == int((out == a['liquid']).sum())
    assert np.array_equal(inp[~(movable | occupied)], out[~(movable | occupied)])
    assert np.array_equal(solve(inp), out)
    return a, records


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles, train = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a, records = _check_pair(pair)
            key = _key(pair['input'])
            assert key not in seen, 'duplicate input'
            seen.add(key)
            roles.add((a['background'], a['liquid']))
            if split == 'train':
                train.append((a, records, np.asarray(pair['output'])))
            else:
                assert any(c['raised'] for c in a['cups'])
                assert any(c['raised'] and r['overflow'] for c, r in zip(a['cups'], records))
                assert len({s['size'] for s in a['sources']}) == 2
    assert len(roles) == 1, 'puzzle color roles must be coherent'
    assert any(_witness(a, out) for a, _, out in train), 'missing rank-order evidence'
    assert any(any(r['volume'] and i >= len(a['sources']) for i, r in enumerate(records))
               for a, records, _ in train), 'missing overflow into an unpaired chamber'
    assert any(any(c['raised'] and 0 < r['volume'] < c['capacity'] for c, r in zip(a['cups'], records))
               and not any(r['overflow'] for r in records) for a, records, _ in train), 'missing step lesson'
    assert any(any(c['color'] == a['liquid'] for c in a['cups']) for a, _, _ in train)


def self_test():
    """Bounded authored checks over 42 seeds, counts, and metamorphic inputs."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, layouts, palettes, profiles, counts, signatures = set(), set(), set(), set(), set(), set()
    wet_steps = spills = unpaired = 0
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[k]) for split in ('train', 'test')
                          for p in puzzle[split] for k in ('input', 'output'))
        signatures.add(signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, records = _check_pair(pair)
                shapes.add(a['grid'].shape)
                palettes.add((a['background'], a['liquid']))
                counts.add((len(a['cups']), len(a['sources'])))
                layout = tuple((o['top'], o['left'], o['size'], o.get('raised', 'solid'))
                               for o in a['cups']+a['sources'])
                layouts.add((a['grid'].shape, layout))
                profiles.update(c['raised'] for c in a['cups'])
                wet_steps += sum(bool(c['raised']) and r['volume'] > 0 for c, r in zip(a['cups'], records))
                spills += sum(r['overflow'] > 0 for r in records)
                unpaired += sum(r['volume'] > 0 and i >= len(a['sources']) for i, r in enumerate(records))
                # Input-only metamorphism: padding translates geometry without
                # changing ranks; global palette permutations preserve meaning.
                permutation = np.roll(np.arange(10, dtype=np.int8), seed % 9 + 1)
                assert np.array_equal(solve(permutation[a['grid']]), permutation[pair['output']])
                assert np.array_equal(solve(a['grid'][:, ::-1]), pair['output'][:, ::-1])
                if max(a['grid'].shape) <= 28:
                    padded = np.pad(a['grid'], 1, constant_values=a['background'])
                    expected = np.pad(pair['output'], 1, constant_values=a['background'])
                    assert np.array_equal(solve(padded), expected)
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(_key(p[k]) == _key(q[k]) for split in ('train', 'test')
                       for p, q in zip(puzzle[split], again[split]) for k in ('input', 'output'))
    assert len(signatures) == 42 and len(layouts) >= 120
    assert len(shapes) >= 25 and len(palettes) >= 25 and len(counts) >= 8
    assert len(profiles) == 7 and wet_steps >= 80 and spills >= 100 and unpaired >= 42
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(8100+10*nt+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (2.0, 1), (True, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Geometry counterfactuals directly show that neither parent is dispensable.
    first = puzzles[0]['train'][0]
    a = _analyze(first['input'])
    assert _witness(a, first['output'])
    second = puzzles[0]['train'][1]
    a = _analyze(second['input'])
    changed = a['grid'].copy()
    for c in a['cups']:
        if c['raised']:
            r, col, s = c['top'], c['left'], c['size']
            changed[r:r+s, col:col+s] = _chamber(s, c['color'], a['background'])
            break
    assert not np.array_equal(solve(changed), second['output'])
    # An input with more objects than the sampler's usual count range. Only
    # visible rank and cavity geometry are needed by solve.
    cup_specs = [(4, ())]*6
    positions = [(1, 1), (2, 7), (3, 13), (8, 1), (9, 7), (10, 13),
                 (16, 1), (17, 7), (18, 13), (21, 19), (22, 25)]
    beyond = _draw(0, 1, cup_specs, [2]*5, [2]*6, positions, (26, 29))
    answer, allocation, remaining = _cascade(_analyze(beyond))
    assert [r['volume'] for r in allocation] == [4, 4, 4, 4, 4, 0]
    assert remaining == 0 and np.array_equal(solve(beyond), answer)
    damaged = {split: [dict(input=np.asarray(p['input']).copy(), output=np.asarray(p['output']).copy())
                       for p in puzzles[0][split]] for split in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds plus legal counts and counterfactuals)')
