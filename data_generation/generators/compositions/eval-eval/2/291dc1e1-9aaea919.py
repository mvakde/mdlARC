"""Cyclic ribbon collection: keyed traversal determines unary transfers.

No parent module is imported. Generation draws answers from explicit ownership;
the solver reconstructs ownership from the input's cyclic reading order.
"""
from __future__ import annotations

from itertools import product
import hashlib
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer arrays, sides 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _body(tile, count, bg):
    w = tile.shape[1]
    a = np.full((2, (w + 1) * count - 1), bg, dtype=np.int8)
    for i in range(count):
        a[:, i * (w + 1):i * (w + 1) + w] = tile
    return a


def _center(bodies, bg):
    width = max(b.shape[1] for b in bodies)
    a = np.full((2 * len(bodies), width), bg, dtype=np.int8)
    for i, b in enumerate(bodies):
        assert (width - b.shape[1]) % 2 == 0
        x = (width - b.shape[1]) // 2
        a[2*i:2*i+2, x:x+b.shape[1]] = b
    return a


def _asymmetric(tile):
    return all(not np.array_equal(tile, other)
               for other in (tile[::-1], tile[:, ::-1], tile[::-1, ::-1]))


def _keyed_view(grid):
    """Find the proper rotation pointing the solid (fast) arm rightward."""
    candidates = []
    for rotation in range(4):
        a = np.rot90(grid, rotation)
        h, w = a.shape
        if h < 5 or w < 5:
            continue
        for bottom in (False, True):
            row = h - 1 if bottom else 0
            far = 0 if bottom else h - 1
            highlight, source = int(a[row, 0]), int(a[row, 1])
            bg = int(a[far, -1])
            direction = -1 if bottom else 1
            dest = int(a[row + direction, 0])
            if len({highlight, source, dest, bg}) != 4:
                continue
            distances = range(1, h)
            expected = [dest if d % 2 else bg for d in distances]
            observed = [int(a[row + direction*d, 0]) for d in distances]
            right_edge = a[:-1, -1] if bottom else a[1:, -1]
            if (np.all(a[row, 1:] == source)
                    and np.all(a[far, 1:] == bg)
                    and np.all(right_edge == bg)):
                if observed == expected:
                    candidates.append(dict(view=a.copy(), rotation=rotation,
                                           bottom=bottom, background=bg,
                                           source=source, destination=dest,
                                           highlight=highlight))
    if len(candidates) != 1:
        raise ValueError("input must have one solid/dotted L key")
    return candidates[0]


def _read(input_grid):
    """Input-only geometric parser; neither seeds nor output grids are read."""
    d = _keyed_view(_grid(input_grid))
    a, bg = d['view'], d['background']
    content = a.copy()
    content[-1 if d['bottom'] else 0, :] = bg
    content[:, 0] = bg
    lanes = _runs(np.any(content != bg, axis=1))
    if not lanes or any(end - start != 2 for start, end in lanes):
        raise ValueError("ribbons must occupy separated two-row lanes")
    if d['bottom']:
        lanes.reverse()
    ribbons = []
    for lane, (y, end) in enumerate(lanes):
        near = 1 if d['bottom'] else 2
        far = a.shape[0] - (2 if d['bottom'] else 1)
        if y < near or end > far:
            raise ValueError("payload must be separated from both borders")
        strip = content[y:end]
        runs = _runs(np.any(strip != bg, axis=0))
        groups = []
        for run in runs:
            if not groups or run[0] - groups[-1][-1][1] >= 3:
                groups.append([run])
            elif run[0] - groups[-1][-1][1] == 1:
                groups[-1].append(run)
            else:
                raise ValueError("one-cell tile gaps and >=3-cell ribbon gaps required")
        for group in groups:
            x, stop = group[0][0], group[-1][1]
            if x < 2 or stop >= a.shape[1]:
                raise ValueError("payload must have a background margin")
            role = 'n'
            if group[0][1] - group[0][0] == 1:
                p, q = group.pop(0)
                cap = strip[:, p:q]
                if not np.all(cap == cap[0, 0]):
                    raise ValueError("control cap must be uniform")
                color = int(cap[0, 0])
                if color not in (d['source'], d['destination']):
                    raise ValueError("control cap must match a guide")
                role = 's' if color == d['source'] else 'd'
            if not group:
                raise ValueError("empty ribbon")
            p, q = group[0]
            tile = strip[:, p:q].copy()
            if q-p not in (3, 5) or np.any(tile == bg):
                raise ValueError("tokens are full 2x3 or 2x5 rectangles")
            if any(v-u != q-p or not np.array_equal(strip[:, u:v], tile)
                   for u, v in group):
                raise ValueError("each ribbon repeats one exact token")
            if role == 's' and np.all(tile == d['highlight']):
                raise ValueError("source highlighting must be visible")
            ribbons.append(dict(role=role, count=len(group), tile=tile,
                                lane=lane, y=y, x=x))
    if not 2 <= len(ribbons) <= 15:
        raise ValueError("ribbon count outside output budget")
    if not any(b['role'] == 's' for b in ribbons) or not any(b['role'] == 'd' for b in ribbons):
        raise ValueError("sources and destinations are both required")
    d['ribbons'] = ribbons
    d['lane_sizes'] = tuple(sum(b['lane'] == i for b in ribbons) for i in range(len(lanes)))
    d['configuration'] = ((-d['rotation']) % 4, d['bottom'])
    return d


def _counts(d, rival=None):
    """Assign every source to the next destination on a circular route."""
    bs = d['ribbons']
    order = list(range(len(bs)))
    if rival == 'previous':
        order.reverse()
    assigned = {i: [] for i, b in enumerate(bs) if b['role'] == 'd'}
    for pos, i in enumerate(order):
        if bs[i]['role'] != 's':
            continue
        if rival == 'global':
            for values in assigned.values():
                values.append(bs[i]['count'])
            continue
        for step in range(1, len(bs) + 1):
            if rival in ('no_wrap', 'lane_reset') and pos + step >= len(bs):
                break
            j = order[(pos + step) % len(bs)]
            if rival == 'lane_reset' and bs[j]['lane'] != bs[i]['lane']:
                break
            if bs[j]['role'] == 'd':
                assigned[j].append(1 if rival == 'bars' else bs[i]['count'])
                break
    counts = [b['count'] for b in bs]
    for i, values in assigned.items():
        contribution = max(values, default=0) if rival == 'maximum' else sum(values)
        counts[i] += contribution
        if rival == 'double':
            counts[i] = 2 * bs[i]['count']
        if rival == 'replace':
            counts[i] = contribution
    return counts, assigned


def _answer(d, counts=None):
    if counts is None:
        counts, _ = _counts(d)
    bodies = []
    for b, count in zip(d['ribbons'], counts):
        tile = (np.full_like(b['tile'], d['highlight'])
                if b['role'] == 's' else b['tile'])
        bodies.append(_body(tile, count, d['background']))
    return _center(bodies, d['background'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _read(input_grid)
    counts, _ = _counts(d)
    if any((b['tile'].shape[1]+1)*n-1 > 30
           for b, n in zip(d['ribbons'], counts)):
        raise ValueError("complete transfer exceeds the 30-cell output budget")
    return _answer(d, counts)


def _tile(rng, width, bg, role, highlight, witness=False):
    pool = [c for c in range(10) if c != bg]
    if witness:
        colors = rng.choice(pool, 3, replace=False)
        tile = np.full((2, width), colors[0], dtype=np.int8)
        tile[0, int(rng.integers(width))] = colors[1]
        tile[1, int(rng.integers(width))] = colors[2]
        if not _asymmetric(tile):
            tile[0, 0], tile[1, -1] = colors[1], colors[2]
        assert _asymmetric(tile)
        return tile
    local = rng.choice(pool, int(rng.integers(1, 6)), replace=False)
    mode = int(rng.integers(4))
    if mode == 0:
        tile = np.full((2, width), rng.choice(local), dtype=np.int8)
    elif mode == 1:
        tile = np.repeat(rng.choice(local, (2, 1)), width, axis=1).astype(np.int8)
    elif mode == 2:
        tile = np.repeat(rng.choice(local, (1, width)), 2, axis=0).astype(np.int8)
    else:
        tile = rng.choice(local, (2, width)).astype(np.int8)
    if role == 's' and np.all(tile == highlight):
        tile[0, 0] = rng.choice([c for c in pool if c != highlight])
    return tile


def _partition(rng, total, n):
    if n == 0:
        assert total == 0
        return []
    cuts = sorted(rng.choice(np.arange(1, total), n-1, replace=False).tolist())
    return np.diff([0] + cuts + [total]).tolist()


def _group_options():
    return [(n, total, dest) for n in range(4)
            for total in (range(n, 7) if n else [0])
            for dest in range(1, min(6, 7-total)+1)]


_OPTIONS = _group_options()
_ANCHORS = [(a, b, base1, base2, g2) for a, b in product(range(1, 4), repeat=2)
            if a != b and a+b <= 5
            for base1, base2 in product(range(1, 4), repeat=2)
            for g2 in ((1,), (2,), (3,), (1, 1), (1, 2), (2, 1))
            if base1 != base2 and a+b != sum(g2)
            and a+b+base1 <= 7 and sum(g2)+base2 <= 7
            and (a+b)*base2 != sum(g2)*base1]


def _scene(rng, colors, kind):
    source, destination, highlight = colors
    bg = int(rng.choice([c for c in range(10) if c not in colors]))
    if kind == 0:
        a, b, base1, base2, g2 = _ANCHORS[int(rng.integers(len(_ANCHORS)))]
        groups = [([a, b], base1), (list(g2), base2)]
        neutral_count = int(rng.integers(1, 3))
    elif kind == 1:
        # An empty collecting interval teaches that destinations reset the carry.
        group_count = int(rng.integers(2, 4))
        groups = []
        empty = int(rng.integers(group_count))
        for i in range(group_count):
            choices = [o for o in _OPTIONS if (o[0] == 0) == (i == empty)
                       and o[0] <= (2 if group_count == 3 else 3)]
            n, total, base = choices[int(rng.integers(len(choices)))]
            groups.append((_partition(rng, total, n), base))
        neutral_count = int(rng.integers(0, min(2, 9-group_count-sum(len(s) for s, _ in groups))+1))
    else:
        group_count = int(rng.integers(1, 4))
        groups = []
        remaining = 9 - group_count
        for i in range(group_count):
            choices = [o for o in _OPTIONS if o[0] <= min(3, remaining)
                       and (i != group_count-1 or any(s for s, _ in groups) or o[0] > 0)]
            n, total, base = choices[int(rng.integers(len(choices)))]
            groups.append((_partition(rng, total, n), base))
            remaining -= n
        neutral_count = int(rng.integers(0, min(3, remaining)+1))
    bands = []
    for target, (sources, base) in enumerate(groups):
        for count in sources:
            bands.append(dict(role='s', count=count, target=target, final=count))
        bands.append(dict(role='d', count=base, target=target, final=base+sum(sources)))
    # Cutting within the unequal source batch guarantees cyclic wrap evidence.
    cut = 1 if kind == 0 else int(rng.integers(len(bands)))
    bands = bands[cut:] + bands[:cut]
    for _ in range(neutral_count):
        count = int(rng.integers(1, 7))
        bands.insert(int(rng.integers(len(bands)+1)),
                     dict(role='n', count=count, final=count, target=None))
    witness_index = next(i for i, b in enumerate(bands) if b['role'] == 'd') if kind < 2 else -1
    source_width = int(rng.choice([3, 5]))
    for i, b in enumerate(bands):
        widths = [w for w in (3, 5) if (w+1)*b['final']-1 <= 29
                  and (w+1)*b['count']-1 + (2 if b['role'] != 'n' else 0) <= 25]
        width = int(rng.choice(widths))
        if kind == 0 and b['role'] == 's' and b['target'] == 0:
            width = source_width
            source_width = 8 - source_width
        b['tile'] = _tile(rng, width, bg, b['role'], highlight, i == witness_index)
        b['witness'] = i == witness_index
    if kind == 0:
        # Identical payloads with different controls expose spatial role semantics.
        copied = next(b for b in bands if b['role'] == 's' and b['tile'].shape[1] == 3)
        neutral = next(b for b in bands if b['role'] == 'n')
        neutral['tile'] = copied['tile'].copy()
    return dict(bands=bands, background=bg, source=source,
                destination=destination, highlight=highlight)


def _input_band(b, scene):
    body = _body(b['tile'], b['count'], scene['background'])
    if b['role'] == 'n':
        return body
    result = np.full((2, body.shape[1]+2), scene['background'], dtype=np.int8)
    result[:, 0] = scene['source'] if b['role'] == 's' else scene['destination']
    result[:, 2:] = body
    return result


def _layouts(lengths, force_multiple):
    """At most 2**8 partitions: every scene fits even as nine single lanes."""
    result = []
    n = len(lengths)
    for bits in range(1 << (n-1)):
        groups, start = [], 0
        for i in range(n-1):
            if bits & (1 << i):
                groups.append(list(range(start, i+1)))
                start = i+1
        groups.append(list(range(start, n)))
        if (not force_multiple or len(groups) >= 2) and len(groups) <= 9:
            if all(sum(lengths[i] for i in g) + 3*(len(g)-1) + 2 <= 29 for g in groups):
                result.append(groups)
    return result


def _gaps(rng, minimums, extra):
    return (np.array(minimums) + rng.multinomial(extra, np.full(len(minimums), 1/len(minimums)))).tolist()


def _draw(scene, rng, configuration, force_multiple=False):
    physical = [_input_band(b, scene) for b in scene['bands']]
    lengths = [b.shape[1] for b in physical]
    partitions = _layouts(lengths, force_multiple)
    assert partitions
    groups = partitions[int(rng.integers(len(partitions)))]
    nlanes = len(groups)
    min_h = 3*nlanes + 2
    min_w = max(sum(lengths[i] for i in g) + 3*(len(g)-1) + 3 for g in groups)
    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))
    rotation, bottom = configuration
    bg = scene['background']
    a = np.full((h, w), bg, dtype=np.int8)
    keyrow = h-1 if bottom else 0
    direction = -1 if bottom else 1
    a[keyrow, 0] = scene['highlight']
    a[keyrow, 1:] = scene['source']
    for distance in range(1, h):
        a[keyrow + direction*distance, 0] = scene['destination'] if distance % 2 else bg
    vertical = _gaps(rng, [1]*(nlanes+1), h-min_h)
    y = 1 + vertical[0]
    for lane, group in enumerate(groups):
        occupied = sum(lengths[i] for i in group)
        horizontal = _gaps(rng, [1] + [3]*(len(group)-1) + [1],
                           w-1-occupied-3*(len(group)-1)-2)
        x = 1 + horizontal[0]
        for j, i in enumerate(group):
            actual_y = h-y-2 if bottom else y
            a[actual_y:actual_y+2, x:x+lengths[i]] = physical[i]
            x += lengths[i] + horizontal[j+1]
        y += 2 + vertical[lane+1]
    # Expected answer uses the sampled group ownership, not the input parser.
    bodies = []
    for b in scene['bands']:
        tile = np.full_like(b['tile'], scene['highlight']) if b['role'] == 's' else b['tile']
        bodies.append(_body(tile, b['final'], bg))
    return {'input': np.rot90(a, rotation).copy(), 'output': _center(bodies, bg)}


def _key(grid):
    return grid.shape, grid.tobytes()


def _unique(scene, pair, rng, configuration, used, multiple):
    if _key(pair['input']) not in used:
        return pair
    # Sixteen distinct motif variants exceed the thirteen earlier input keys.
    candidates = [b for b in scene['bands'] if b['role'] == 'd' and not b['witness']]
    if not candidates:
        candidates = [b for b in scene['bands'] if b['role'] == 'd']
    band = candidates[0]
    original = band['tile'].copy()
    colors = [c for c in range(10) if c != scene['background']][:2]
    for code in range(16):
        band['tile'] = original.copy()
        for bit in range(4):
            band['tile'].flat[bit] = colors[(code >> bit) & 1]
        pair = _draw(scene, rng, configuration, multiple)
        if _key(pair['input']) not in used:
            return pair
    raise AssertionError("finite uniqueness fallback exhausted")


def _evidence(descriptors):
    bs_all = [b for d in descriptors for b in d['ribbons']]
    assert any(b['role'] == 'n' for b in bs_all), "teach ignoring unmarked ribbons"
    assert any(any(s['role'] == 's' and n['role'] == 'n'
                   and np.array_equal(s['tile'], n['tile'])
                   for s in d['ribbons'] for n in d['ribbons'])
               for d in descriptors), "equal payloads can have different roles"
    assert any(b['role'] != 's' and np.any(np.isin(b['tile'], [d['source'], d['destination']]))
               for d in descriptors for b in d['ribbons']), "payload colors are not controls"
    assert any(b['role'] == 'd' and _asymmetric(b['tile']) for b in bs_all)
    assert len({d['configuration'] for d in descriptors}) >= 2
    assert any(len(d['lane_sizes']) > 1 and max(d['lane_sizes']) > 1 for d in descriptors), "teach lane-major reading"
    assert any(any(not values for values in _counts(d)[1].values()) for d in descriptors), "teach empty batches"
    assert any(_mixed_source_widths(d) for d in descriptors), "count tokens, not pixel width"
    for rival in ('global', 'bars', 'maximum', 'double', 'replace', 'no_wrap', 'lane_reset', 'previous'):
        assert any(_counts(d)[0] != _counts(d, rival)[0] for d in descriptors), rival
    assert any(len({(b['tile'].shape[1]+1)*c-1 for b, c in zip(d['ribbons'], _counts(d)[0])}) > 1
               for d in descriptors), "teach symmetric centering"
    for mode in ('fast_major', 'reverse_lanes'):
        distinguished = False
        for d in descriptors:
            bs = d['ribbons']
            if mode == 'fast_major':
                order = sorted(range(len(bs)), key=lambda i: (bs[i]['x'], bs[i]['lane']))
            else:
                order = sorted(range(len(bs)), key=lambda i: (-bs[i]['lane'], bs[i]['x']))
            other = dict(d, ribbons=[bs[i] for i in order])
            distinguished |= not np.array_equal(_answer(d), _answer(other))
        assert distinguished, mode


def _mixed_source_widths(d):
    bs = d['ribbons']
    destinations = [i for i, b in enumerate(bs) if b['role'] == 'd']
    for dest in destinations:
        widths = set()
        for distance in range(1, len(bs)):
            b = bs[(dest-distance) % len(bs)]
            if b['role'] == 'd':
                break
            if b['role'] == 's':
                widths.add(b['tile'].shape[1])
        if len(widths) > 1:
            return True
    return False


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    colors = tuple(map(int, rng.choice(10, 3, replace=False)))
    r, h = int(rng.integers(4)), bool(rng.integers(2))
    anchors = [(r, h), ((r+1) % 4, not h)]
    heldout = [(r, not h), ((r+1) % 4, h)]
    pool = [(i, j) for i in range(4) for j in (False, True) if (i, j) not in heldout]
    # Both anchors together must visibly disambiguate all documented rivals.
    for _ in range(128):
        pairs = [_draw(_scene(rng, colors, i), rng, anchors[i], True) for i in range(2)]
        try:
            _evidence([_read(p['input']) for p in pairs])
        except AssertionError:
            continue
        break
    else:
        raise AssertionError("informative anchor construction exhausted")
    puzzle = {'train': pairs, 'test': []}
    used = {_key(p['input']) for p in pairs}
    assert len(used) == 2  # the two fast axes differ
    for split, count in (('train', num_train-2), ('test', num_test)):
        for _ in range(count):
            configurations = pool if split == 'train' else heldout
            config = configurations[int(rng.integers(len(configurations)))]
            scene = _scene(rng, colors, 2)
            pair = _draw(scene, rng, config)
            pair = _unique(scene, pair, rng, config, used, False)
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles, training, testing = set(), set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            d = _read(a)
            counts, assigned = _counts(d)
            expected = _answer(d, counts)
            assert expected.shape == b.shape and np.array_equal(expected, b), "incorrect cyclic transfer / normalized rendering"
            assert b.shape[0] == 2*len(d['ribbons']) and b.shape[1] % 2 == 1
            assert sum(sum(v) for v in assigned.values()) == sum(x['count'] for x in d['ribbons'] if x['role'] == 's'), "every source counted once"
            for ribbon, count in zip(d['ribbons'], counts):
                if ribbon['role'] != 'd':
                    assert count == ribbon['count']
            roles.add((d['source'], d['destination'], d['highlight']))
            (training if split == 'train' else testing).append(d)
    assert len(roles) == 1, "guide/highlight identities stay fixed within a puzzle"
    _evidence(training)
    assert not {d['configuration'] for d in training} & {d['configuration'] for d in testing}, "held-out direction/side combinations"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for name in ('input', 'output'):
                a = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, finite feature summaries, count boundaries and label mutations."""
    puzzles, shapes, outputs, configs, palettes, structures = set(), set(), set(), set(), set(), set()
    widths, populations, totals, bases, nsource, ndest, nneutral = set(), set(), set(), set(), set(), set(), set()
    empty = wrap = patterned = repeated_colors = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for pair in puzzle['train'] + puzzle['test']:
            d = _read(pair['input'])
            a = pair['input'].copy()
            assert np.array_equal(solve(a), pair['output']) and np.array_equal(a, pair['input'])
            shapes.add(a.shape)
            outputs.add(pair['output'].shape)
            configs.add(d['configuration'])
            palettes.add((d['source'], d['destination'], d['highlight']))
            populations.add(d['lane_sizes'])
            bs = d['ribbons']
            counts, assigned = _counts(d)
            structures.add((d['lane_sizes'], tuple((b['role'], b['count'], b['tile'].shape[1]) for b in bs)))
            nsource.add(sum(b['role'] == 's' for b in bs))
            ndest.add(sum(b['role'] == 'd' for b in bs))
            nneutral.add(sum(b['role'] == 'n' for b in bs))
            for i, b in enumerate(bs):
                widths.add(b['tile'].shape[1])
                patterned += not np.all(b['tile'] == b['tile'].flat[0])
                if b['role'] == 'd':
                    totals.add(counts[i]-b['count'])
                    bases.add(b['count'])
            empty += any(not v for v in assigned.values())
            wrap += _counts(d)[0] != _counts(d, 'no_wrap')[0]
            flat_colors = [int(b['tile'].flat[0]) for b in bs]
            repeated_colors += len(flat_colors) != len(set(flat_colors))
    assert len(puzzles) == 42 and len(palettes) >= 25
    assert len(shapes) >= 40 and len(outputs) >= 30 and len(structures) >= 150
    assert len(configs) == 8 and widths == {3, 5}
    assert {1, 2, 3} <= ndest and {0, 1, 2, 3} <= nneutral
    assert max(nsource) >= 5 and max(totals) >= 5 and max(bases) >= 5
    assert len(populations) >= 30 and empty >= 42 and wrap >= 42 and repeated_colors >= 100
    for train, test in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(42+train+test, train, test)
        assert len(p['train']) == train and len(p['test']) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A boundary member outside the sampler's nine-ribbon population: 15 bodies.
    # One one-token source feeds a one-token destination; thirteen neutrals stay.
    scene = dict(background=0, source=1, destination=2, highlight=3)
    member = np.zeros((18, 25), dtype=np.int8)
    member[0, 0], member[0, 1:] = 3, 1
    member[1::2, 0] = 2
    expected_bodies = []
    for i in range(15):
        role = 's' if i == 0 else 'd' if i == 1 else 'n'
        tile = np.array([[4, 5, 6], [7, 4, 8]], dtype=np.int8)
        band = dict(role=role, count=1, tile=tile)
        body = _input_band(band, scene)
        y, x = 2+3*(i//3), 2+8*(i % 3)
        member[y:y+2, x:x+body.shape[1]] = body
        expected_bodies.append(_body(np.full_like(tile, 3) if role == 's' else tile,
                                     2 if role == 'd' else 1, 0))
    expected = _center(expected_bodies, 0)
    assert expected.shape == (30, 7)
    recolor = np.array([9, 4, 7, 0, 2, 8, 3, 6, 1, 5], dtype=np.int8)
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(member, rotation)), expected)
        assert np.array_equal(solve(recolor[np.rot90(member, rotation)]), recolor[expected])
    damaged = member.copy()
    damaged[2, 4] = 0  # a hole in the source token
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("partial token accepted")
    # Reject wrong labels for each operational obligation, not just schema errors.
    original = generate_puzzle(0, 2, 1)
    for defect in ('transfer', 'highlight', 'motif', 'centering', 'order'):
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in ('train', 'test')}
        p = broken['test'][0]
        d = _read(p['input'])
        if defect == 'transfer':
            p['output'] = _answer(d, [b['count'] for b in d['ribbons']])
        elif defect == 'order':
            p['output'] = p['output'][::-1].copy()
        else:
            role = 's' if defect == 'highlight' else 'd'
            i = next(i for i, b in enumerate(d['ribbons']) if b['role'] == role)
            if defect == 'centering':
                p['output'][2*i:2*i+2] = np.roll(p['output'][2*i:2*i+2], 1, axis=1)
            else:
                x = int(np.flatnonzero(p['output'][2*i] != d['background'])[0])
                p['output'][2*i, x] = d['background']
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("bad " + defect + " accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(puzzles), input_shapes=len(shapes),
                output_shapes=len(outputs), structural_signatures=len(structures), configurations=len(configs),
                palettes=len(palettes), lane_populations=len(populations), token_widths=sorted(widths),
                source_counts=sorted(nsource), destination_counts=sorted(ndest), neutral_counts=sorted(nneutral),
                contributions=sorted(totals), destination_bases=sorted(bases), empty_batch_pairs=empty,
                wrap_pairs=wrap, patterned_tokens=patterned, repeated_color_pairs=repeated_colors,
                independent_solver=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
