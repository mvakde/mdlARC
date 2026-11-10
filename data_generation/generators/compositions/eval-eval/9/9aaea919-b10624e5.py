"""Mirror collectors: geometric equivalence gates unary stack addition.

Only standard-library modules and NumPy are used. Generation paints its planned
answer; solve reconstructs roles and eligibility solely from input pixels.
Nothing is written by this module, including self_test.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np


HOST_SEEDS = (0, 64453905, 230911285, 692305947, 419167433, 371256367,
              848460295, 472530239, 693707649, 435415129, 932450696,
              497771978, 339915792, 671591916, 329778695, 29451450,
              227351240, 6859943, 19930053, 659215804, 522104121,
              671702164, 30655611, 732067828, 756039558, 710212736,
              636411885, 519979142, 763465185, 265315731, 324348099,
              470841035, 151346187, 546877471, 44498304, 82498287,
              251940802, 150043628, 459804104, 399257832, 95238425,
              26668448)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + ' is out of range')
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= s <= 30 for s in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _bounds(height, width, row, col):
    return ((0, row, 0, col), (0, row, col + 1, width),
            (row + 1, height, 0, col), (row + 1, height, col + 1, width))


def _flip(array, reference, panel):
    if reference // 2 != panel // 2:
        array = array[::-1]
    if reference % 2 != panel % 2:
        array = array[:, ::-1]
    return array.copy()


def _components(panel, background):
    remaining = set(map(tuple, np.argwhere(panel != background).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, points = [start], [start]
        while todo:
            y, x = todo.pop()
            for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if point in remaining:
                    remaining.remove(point)
                    todo.append(point)
                    points.append(point)
        ys, xs = zip(*points)
        y, x, end_y, end_x = min(ys), min(xs), max(ys) + 1, max(xs) + 1
        result.append(dict(y=y, x=x, end_y=end_y, end_x=end_x,
                           tile=panel[y:end_y, x:end_x].copy()))
    return sorted(result, key=lambda c: (c['x'], c['y']))


def _square(tile, color):
    points = np.argwhere(tile == color)
    assert len(points)
    top, left = points.min(0)
    bottom, right = points.max(0) + 1
    side = int(bottom - top)
    assert side == right - left and len(points) == side * side
    assert np.all(tile[top:bottom, left:right] == color)
    return int(top), int(left), side


def _normalize(tile, anchor, background):
    _, _, side = _square(tile, anchor)
    assert side >= 2 and side % 2 == 0
    unit = side // 2
    h, w = tile.shape
    assert h % unit == 0 and w % unit == 0
    blocks = tile.reshape(h // unit, unit, w // unit, unit)
    small = blocks[:, 0, :, 0]
    assert np.all(blocks == small[:, None, :, None])
    assert np.all(np.any(small != background, axis=0))
    assert np.all(np.any(small != background, axis=1))
    return small.copy(), unit


def _decode(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    rows = [r for r in range(1, grid.shape[0] - 1)
            if grid[r, 0] != background and np.all(grid[r] == grid[r, 0])]
    cols = [c for c in range(1, grid.shape[1] - 1)
            if grid[0, c] != background and np.all(grid[:, c] == grid[0, c])]
    assert len(rows) == len(cols) == 1, 'one full cross required'
    row, col = rows[0], cols[0]
    divider = int(grid[row, col])
    assert grid[row, 0] == grid[0, col]
    bounds = _bounds(*grid.shape, row, col)
    lanes, bars = [], []
    for panel, (r0, r1, c0, c1) in enumerate(bounds):
        objects = _components(grid[r0:r1, c0:c1], background)
        groups = defaultdict(list)
        for obj in objects:
            tile = obj['tile']
            if tile.shape[0] == 1:
                assert tile.shape[1] >= 2 and np.all(tile == tile[0, 0])
                bars.append(dict(obj, panel=panel, color=int(tile[0, 0])))
            else:
                groups[(obj['x'], obj['end_x'])].append(obj)
        assert groups, 'every panel contains a stack'
        for (x, end_x), tokens in sorted(groups.items()):
            tokens.sort(key=lambda c: c['y'])
            tile = tokens[0]['tile']
            assert all(np.array_equal(o['tile'], tile) for o in tokens)
            lanes.append(dict(panel=panel, x=x, end_x=end_x, tokens=tokens,
                              tile=tile, count=len(tokens)))
    assert len(bars) == 1, 'one isolated one-row collector bar required'
    bar = bars[0]
    targets = [s for s in lanes if (s['panel'], s['x'], s['end_x']) ==
               (bar['panel'], bar['x'], bar['end_x'])]
    assert len(targets) == 1
    target = targets[0]
    colors, sizes = np.unique(target['tile'], return_counts=True)
    foreground = [(int(n), int(c)) for c, n in zip(colors, sizes) if c != background]
    assert len(foreground) >= 3
    foreground.sort(reverse=True)
    assert foreground[0][0] > foreground[1][0], 'anchor is uniquely largest color'
    anchor = foreground[0][1]
    prototype, target_unit = _normalize(target['tile'], anchor, background)
    assert np.count_nonzero(prototype == anchor) == 4
    assert all(np.count_nonzero(prototype == c) <= 3
               for c in np.unique(prototype) if c not in (background, anchor))
    assert len({background, divider, anchor, bar['color']}) == 4
    for lane in lanes:
        small, unit = _normalize(lane['tile'], anchor, background)
        lane.update(small=small, unit=unit)
        assert bar['color'] not in lane['tile']
        pitch = lane['tile'].shape[0] + unit
        assert all(b['y'] - a['y'] == pitch
                   for a, b in zip(lane['tokens'], lane['tokens'][1:]))
        lane['eligible'] = (lane is not target and np.array_equal(
            small, _flip(prototype, target['panel'], lane['panel'])))
    # The bar is always on the side of its stack facing the horizontal divider.
    if target['panel'] // 2 == 0:
        inward_token = target['tokens'][-1]
        assert bar['y'] - inward_token['end_y'] == target_unit
    else:
        inward_token = target['tokens'][0]
        assert inward_token['y'] - bar['end_y'] == target_unit
    target['inward_y'] = inward_token['y']
    return dict(grid=grid, background=background, divider=divider, bounds=bounds,
                bar=bar, anchor=anchor, target=target, prototype=prototype,
                lanes=lanes, sources=[s for s in lanes if s['eligible']])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the collector, compare normalized mirrored motifs, and add counts."""
    data = _decode(input_grid)
    out = data['grid'].copy()
    bg, target, bar = data['background'], data['target'], data['bar']
    r0, r1, c0, c1 = data['bounds'][target['panel']]
    region = out[r0:r1, c0:c1]
    region[bar['y']:bar['end_y'], bar['x']:bar['end_x']] = bg
    total = sum(s['count'] for s in data['sources'])
    tile = target['tile']
    height, width = tile.shape
    pitch = height + target['unit']
    direction = -1 if target['panel'] // 2 == 0 else 1
    for k in range(target['count'], target['count'] + total):
        y = target['inward_y'] + direction * k * pitch
        assert 0 <= y and y + height <= region.shape[0], 'collector must fit completely'
        region[y:y + height, target['x']:target['x'] + width] = tile
    for source in data['sources']:
        a, b, c, d = data['bounds'][source['panel']]
        for token in source['tokens']:
            block = out[a + token['y']:a + token['end_y'],
                        c + token['x']:c + token['end_x']]
            block[token['tile'] != bg] = bar['color']
    return out


def _motif(rng, bg, anchor, details, final, fallback=False, preferred_height=None):
    # A 2x2 anchor plus a grown, thin colored polyomino. No detail color may
    # rival the anchor's four base cells. This makes the anchor input-visible.
    if fallback:
        cells = {(0, 0): anchor, (0, 1): anchor, (1, 0): anchor,
                 (1, 1): anchor, (0, 2): details[0], (1, 2): details[1]}
    else:
        max_h = min(5, (24 - final) // (final + 1))
        max_h = max(2, max_h)
        if preferred_height is not None:
            max_h = preferred_height
        elif rng.random() < .48:
            max_h = 2
        max_w = int(rng.integers(3, 7))
        cells = {(r, c): anchor for r in range(2) for c in range(2)}
        detail_count = int(rng.integers(2, 9))
        usage = {c: 0 for c in details}
        initial = 0
        if preferred_height is not None:
            for distance in range(1, preferred_height - 1):
                color = details[initial % len(details)]
                cells[-distance, 0] = color
                usage[color] += 1
                initial += 1
            detail_count = max(detail_count, initial)
        for step in range(initial, detail_count):
            frontier = sorted({(r + dr, c + dc) for r, c in cells
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                               if (r + dr, c + dc) not in cells})
            legal = []
            for y, x in frontier:
                ys = [p[0] for p in cells] + [y]
                xs = [p[1] for p in cells] + [x]
                if max(ys) - min(ys) < max_h and max(xs) - min(xs) < max_w:
                    legal.append((y, x))
            pool = [c for c in details if usage[c] < 3]
            if not legal or not pool:
                break
            point = legal[int(rng.integers(len(legal)))]
            color = details[step] if step < len(details) else int(rng.choice(pool))
            cells[point] = color
            usage[color] += 1
    ys, xs = zip(*cells)
    tile = np.full((max(ys) - min(ys) + 1, max(xs) - min(xs) + 1), bg, np.int8)
    for (r, c), color in cells.items():
        tile[r - min(ys), c - min(xs)] = color
    if rng.integers(2):
        tile = tile[::-1]
    if rng.integers(2):
        tile = tile[:, ::-1]
    return tile.copy()


def _decoy(rng, expected, bg, anchor):
    if rng.random() < .65:
        alternatives = [expected[::-1], expected[:, ::-1], expected[::-1, ::-1]]
        alternatives = [a for a in alternatives if not np.array_equal(a, expected)]
        return alternatives[int(rng.integers(len(alternatives)))].copy()
    colors = [int(c) for c in np.unique(expected) if c not in (bg, anchor)]
    a, b = rng.choice(colors, 2, replace=False)
    result = expected.copy()
    result[expected == a], result[expected == b] = b, a
    return result


def _partition(rng, total, number, unequal=False):
    # Tiny exhaustive compositions avoid any unbounded retry.
    options = []
    def visit(prefix, remaining):
        if len(prefix) == number - 1:
            if 1 <= remaining <= 3:
                values = prefix + [remaining]
                if not unequal or len(set(values)) > 1:
                    options.append(values)
        else:
            for value in range(1, min(3, remaining - (number - len(prefix) - 1)) + 1):
                visit(prefix + [value], remaining - value)
    visit([], total)
    assert options
    return options[int(rng.integers(len(options)))].copy()


def _plan(rng, reference, total, dest, matches, relations, unequal=False, extras=True):
    # XOR quadrant bits: 1 horizontal, 2 vertical, 3 both reflections.
    grouped = {reference ^ relation: relations.count(relation) for relation in (1, 2, 3)}
    counts = _partition(rng, total, matches, unequal)
    lanes = [dict(panel=reference, target=True, match=False, count=dest)]
    cursor = 0
    for panel in sorted(grouped):
        n = grouped[panel]
        local = []
        for _ in range(n):
            local.append(dict(panel=panel, target=False, match=True, count=counts[cursor]))
            cursor += 1
        if n == 0:
            local.append(dict(panel=panel, target=False, match=False,
                              count=int(rng.choice([1, 1, 1, 2]))))
        if extras and len(local) == 1 and rng.random() < .28:
            local.append(dict(panel=panel, target=False, match=False,
                              count=int(rng.choice([1, 1, 2, 3]))))
        rng.shuffle(local)
        lanes.extend(local)
    assert cursor == matches
    return lanes


def _axis(rng, requirements):
    minimum = sum(requirements) + 1
    if minimum > 30:
        return None
    size = 30 if rng.random() < .2 else int(rng.integers(minimum, 31))
    extra = size - minimum
    before = int(rng.integers(extra + 1))
    return [requirements[0] + before, requirements[1] + extra - before]


def _attempt(rng, reference, total, dest, matches, relations, unequal, fallback, require_scale,
             preferred_height=None):
    colors = list(map(int, rng.permutation(10)))
    bg, divider, anchor, marker = colors[:4]
    details = colors[4:4 + int(rng.integers(2, 5))]
    if rng.random() < .35:
        details[-1] = divider  # divider-colored content remains part of tokens
    prototype = _motif(rng, bg, anchor, details, total + dest, fallback,
                       None if fallback else preferred_height)
    if any(np.array_equal(prototype, a) for a in
           (prototype[::-1], prototype[:, ::-1], prototype[::-1, ::-1])):
        return None
    lanes = _plan(rng, reference, total, dest, matches, relations, unequal, not fallback)
    target_unit = 1 if fallback else int(rng.choice([1, 1, 1, 1, 2]))
    # The aggregation witness always shows another scale. Other examples may
    # legitimately use equal sizes, especially when a tall opposite stack fits
    # only at unit scale. Scale is not a split-wide requirement.
    selected = [s for s in lanes if s['match']]
    scale_witness = min(selected, key=lambda s: s['count']) if fallback else selected[int(rng.integers(len(selected)))]
    if not require_scale and (fallback or rng.random() < .4):
        scale_witness = None
    for lane in lanes:
        unit = target_unit if lane['target'] else (1 if fallback else int(rng.choice([1, 1, 1, 2, 3])))
        if lane is scale_witness:
            unit = 2 if target_unit == 1 else 1
        small = _flip(prototype, reference, lane['panel'])
        if not lane['target'] and not lane['match']:
            small = _decoy(rng, small, bg, anchor)
        tile = np.repeat(np.repeat(small, unit, axis=0), unit, axis=1)
        lane.update(unit=unit, tile=tile, margin=1 if fallback else int(rng.integers(1, 4)))
    row_needs, col_needs = [0, 0], [0, 0]
    for panel in range(4):
        local = [s for s in lanes if s['panel'] == panel]
        width = sum(s['tile'].shape[1] for s in local) + len(local) - 1 + 2
        col_needs[panel % 2] = max(col_needs[panel % 2], width)
        for lane in local:
            h = lane['tile'].shape[0]
            count = dest + total if lane['target'] else lane['count']
            height = lane['margin'] + count * h + (count - 1) * lane['unit'] + 1
            if lane['target']:
                height += 1 + lane['unit']
            row_needs[panel // 2] = max(row_needs[panel // 2], height)
    heights, widths = _axis(rng, row_needs), _axis(rng, col_needs)
    if heights is None or widths is None:
        return None
    bounds = _bounds(sum(heights) + 1, sum(widths) + 1, heights[0], widths[0])
    inp = np.full((sum(heights) + 1, sum(widths) + 1), bg, np.int8)
    inp[heights[0]] = divider
    inp[:, widths[0]] = divider
    out = inp.copy()
    for panel, (r0, r1, c0, c1) in enumerate(bounds):
        local = [s for s in lanes if s['panel'] == panel]
        minimum = sum(s['tile'].shape[1] for s in local) + len(local) - 1 + 2
        extra = rng.multinomial(c1 - c0 - minimum, np.full(len(local) + 1, 1 / (len(local) + 1)))
        x = 1 + int(extra[0])
        for index, lane in enumerate(local):
            tile, unit = lane['tile'], lane['unit']
            h, w = tile.shape
            inward = lane['margin'] + (1 + unit if lane['target'] else 0)
            base = r1 - r0 - inward - h if panel // 2 == 0 else inward
            direction = -1 if panel // 2 == 0 else 1
            final = dest + total if lane['target'] else lane['count']
            for k in range(final):
                y = base + direction * k * (h + unit)
                out_tile = tile.copy()
                if lane['match']:
                    out_tile[tile != bg] = marker
                out[r0 + y:r0 + y + h, c0 + x:c0 + x + w] = out_tile
                if k < lane['count']:
                    inp[r0 + y:r0 + y + h, c0 + x:c0 + x + w] = tile
            if lane['target']:
                bar_y = r1 - r0 - lane['margin'] - 1 if panel // 2 == 0 else lane['margin']
                inp[r0 + bar_y, c0 + x:c0 + x + w] = marker
            x += w + 1 + int(extra[index + 1])
    return {'input': inp, 'output': out}


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _make_pair(rng, reference, total, dest, matches, relations, seen, unequal=False):
    # Persist a feasible height preference across attempts, rather than silently
    # biasing every accepted scene toward whichever tiny glyph fits first.
    final = total + dest
    preferred_height = None
    if final <= 4 and rng.random() < .60:
        maximum = 3 if final == 4 else 4 if total > 1 else 5
        preferred_height = int(rng.integers(3, maximum + 1))
    for attempt in range(192):
        pair = _attempt(rng, reference, total, dest, matches, relations,
                        unequal, fallback=attempt >= 128, require_scale=unequal,
                        preferred_height=preferred_height)
        if pair is not None and _key(pair['input']) not in seen:
            # Decoder is an audit, never the construction of the answer.
            assert np.array_equal(solve(pair['input']), pair['output'])
            return pair
    raise AssertionError('bounded scene construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    # Jointly informative operands, randomized order, shapes, and placement.
    options = [(s1, d1, s2, d2) for s1 in (3, 4) for d1 in (1, 2)
               for s2 in (1, 2) for d2 in (2, 3)
               if s1 + d1 <= 5 and s2 + d2 <= 5 and d1 != d2
               and s1 + d1 != s2 + d2 and s1 * d2 != s2 * d1
               and not (s1 * d1 == s1 + d1 and s2 * d2 == s2 + d2)]
    s1, d1, s2, d2 = options[int(rng.integers(len(options)))]
    references = list(map(int, rng.permutation(4)))
    axis = int(rng.choice([1, 2]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(num_train + num_test):
        reference = references[index] if index < 2 else int(rng.integers(4))
        if index == 0:
            total, dest, matches, relations = s1, d1, 2, [axis, axis]
        elif index == 1:
            total, dest, matches, relations = s2, d2, 1, [3 ^ axis]
        elif index < num_train:
            if rng.random() < .45:
                matches, total, dest = 1, 1, int(rng.integers(1, 3))
            else:
                matches = int(rng.integers(1, 3))
                total = int(rng.integers(matches, (3 if matches == 1 else 4) + 1))
                dest = int(rng.integers(1, min(3, 5 - total) + 1))
            relations = list(map(int, rng.integers(1, 4, size=matches)))
        else:
            matches = int(rng.choice([3, 3, 3, 4]))
            total = matches + int(matches == 3 and rng.integers(2))
            dest = int(rng.integers(1, min(2, 6 - total) + 1))
            a, b = list(map(int, rng.choice([1, 2, 3], 2, replace=False)))
            relations = [a, a, b] + ([b] if matches == 4 else [])
        pair = _make_pair(rng, reference, total, dest, matches, relations,
                          seen, unequal=index == 0)
        seen.add(_key(pair['input']))
        puzzle['train' if index < num_train else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(data):
    target, sources = data['target'], data['sources']
    source_counts = tuple(s['count'] for s in sources)
    grouped = defaultdict(list)
    for s in sources:
        grouped[s['panel']].append(s['count'])
    ratios = set()
    for s in sources:
        divisor = math.gcd(s['unit'], target['unit'])
        ratios.add((s['unit'] // divisor, target['unit'] // divisor))
    return dict(reference=target['panel'], destination=target['count'],
                total=sum(source_counts), source_counts=source_counts,
                matches=len(sources), decoys=len(data['lanes']) - 1 - len(sources),
                lanes=len(data['lanes']), ratios=tuple(sorted(ratios)),
                target_unit=target['unit'], motif_shape=data['prototype'].shape,
                motif=tuple(map(tuple, data['prototype'].tolist())),
                relations=tuple(sorted({target['panel'] ^ s['panel'] for s in sources})),
                same_panel_pool=any(len(v) > 1 for v in grouped.values()),
                one_per_panel=sum(max(v) for v in grouped.values()),
                area_total=sum(s['count'] * s['unit'] ** 2 for s in sources),
                divider_content=data['divider'] in data['prototype'],
                palette=(data['background'], data['divider'], data['anchor'], data['bar']['color']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            data = _decode(inp)
            assert data['sources']
            target = data['target']
            for panel in range(4):
                n = sum(s['panel'] == panel for s in data['lanes'])
                assert n == 1 if panel == target['panel'] else 1 <= n <= 2
            assert np.array_equal(solve(inp), out), 'wrong completion or preservation'
            assert not np.array_equal(inp, out)
            f = _features(data)
            assert f['decoys'] >= 1
            assert all(not np.array_equal(data['prototype'], a) for a in
                       (data['prototype'][::-1], data['prototype'][:, ::-1],
                        data['prototype'][::-1, ::-1]))
            features[split].append(f)
            # Direct preservation/recoloring invariants supplement whole-grid equality.
            r0, r1, c0, c1 = data['bounds'][target['panel']]
            for lane in data['lanes']:
                a, b, c, d = data['bounds'][lane['panel']]
                for token in lane['tokens']:
                    block = out[a + token['y']:a + token['end_y'],
                                c + token['x']:c + token['end_x']]
                    wanted = token['tile'].copy()
                    if lane['eligible']:
                        wanted[wanted != data['background']] = data['bar']['color']
                    assert np.array_equal(block, wanted)
            assert np.all(out[r0 + data['bar']['y'],
                              c0 + data['bar']['x']:c0 + data['bar']['end_x']] == data['background'])
    train, test = features['train'], features['test']
    assert len({f['reference'] for f in train}) >= 2
    assert {1, 2} <= {r for f in train for r in f['relations']}
    assert all(1 <= f['matches'] <= 2 for f in train)
    assert all(3 <= f['matches'] <= 4 and f['same_panel_pool'] for f in test)
    assert any(f['same_panel_pool'] and len(set(f['source_counts'])) > 1 for f in train)
    assert len({f['total'] for f in train}) > 1
    assert len({f['destination'] for f in train}) > 1
    assert len({f['total'] + f['destination'] for f in train}) > 1
    assert len({(f['total'] // math.gcd(f['total'], f['destination']),
                 f['destination'] // math.gcd(f['total'], f['destination']))
                for f in train}) > 1, 'a fixed destination multiplier must fail'
    assert any(n != d for f in train for n, d in f['ratios'])
    rivals = (lambda f: 2 * f['destination'], lambda f: 2 * f['total'],
              lambda f: f['destination'] * f['total'],
              lambda f: max(f['destination'], f['total']), lambda f: f['total'],
              lambda f: f['destination'] + f['matches'],
              lambda f: f['destination'] + max(f['source_counts']),
              lambda f: f['destination'] + f['one_per_panel'])
    assert all(any(rival(f) != f['total'] + f['destination'] for f in train) for rival in rivals)
    assert any(f['area_total'] != f['total'] * f['target_unit'] ** 2 for f in train)
    return None


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 fixed seeds, all legal counts, counterfactuals, and mutations; no I/O."""
    shapes, motifs, palettes, references, ratios, counts, fingerprints = (set() for _ in range(7))
    pair_count = same_panel = divider_content = tall_motifs = 0
    motif_heights = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprint = _puzzle_digest(puzzle)
        assert fingerprint == _puzzle_digest(generate_puzzle(seed))
        fingerprints.add(fingerprint)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _decode(pair['input'])
                f = _features(data)
                shapes.add(pair['input'].shape)
                # Normalize literal colors, retaining every cell and its geometry.
                roles = {data['background']: 0, data['anchor']: 1}
                motifs.add(tuple(tuple(roles.setdefault(int(c), len(roles)) for c in row)
                                 for row in data['prototype']))
                palettes.add(f['palette'])
                references.add(f['reference'])
                ratios.update(f['ratios'])
                counts.add((f['matches'], f['total'], f['destination'], f['lanes']))
                same_panel += f['same_panel_pool']
                divider_content += f['divider_content']
                tall_motifs += data['prototype'].shape[0] > 2
                motif_heights.add(data['prototype'].shape[0])
                pair_count += 1
                # Ignoring B collects decoys; ignoring A leaves a short collector.
                all_count = sum(s['count'] for s in data['lanes'] if s is not data['target'])
                assert all_count > f['total'] > 0
                # Multiplying object count by geometric area is a different rule.
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
    assert len(fingerprints) == 42 and pair_count == 210
    assert len(shapes) >= 25 and len(motifs) >= 35 and len(palettes) >= 100
    assert references == {0, 1, 2, 3} and {(1, 1), (2, 1), (1, 2)} <= ratios
    assert len(counts) >= 25 and same_panel >= 84 and divider_content >= 20
    assert {2, 3, 4, 5} <= motif_heights and tall_motifs >= 25
    # Exercise the compact path explicitly, across all sampled witness operands,
    # both reflection axes, and all collector quadrants. No payload is retained.
    fallback_count = 0
    fallback_rng = np.random.default_rng(8675309)
    for ref in range(4):
        for relation in (1, 2):
            for total, dest in ((3, 1), (3, 2), (4, 1)):
                p = _attempt(fallback_rng, ref, total, dest, 2,
                             [relation, relation], True, True, True)
                assert p is not None and np.array_equal(solve(p['input']), p['output'])
                fallback_count += 1
    for total, dest, relations in ((1, 3, [2]), (2, 3, [2]), (3, 2, [3]),
                                   (4, 1, [2, 2]), (3, 2, [1, 1, 2]),
                                   (4, 2, [2, 2, 3]), (4, 1, [1, 1, 3, 3])):
        p = _attempt(fallback_rng, 0, total, dest, len(relations),
                     relations, False, True, False)
        assert p is not None and np.array_equal(solve(p['input']), p['output'])
        fallback_count += 1
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(7301 + 10 * train + test, train, test)
            assert len(p['train']) == train and len(p['test']) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    original = generate_puzzle(0, 2, 1)
    # Valid family symmetries and a complete palette bijection check that no
    # absolute quadrant, growth half, or literal color is accidentally special.
    lookup = np.random.default_rng(2048).permutation(10).astype(np.int8)
    for pair in original['train'] + original['test']:
        for transform in (lambda a: np.rot90(a, 2).copy(), lambda a: a[:, ::-1].copy(),
                          lambda a: lookup[a]):
            assert np.array_equal(solve(transform(pair['input'])), transform(pair['output']))
    data = _decode(original['train'][0]['input'])
    for defect in ('highlight', 'bar', 'sum', 'decoy'):
        broken = {s: [{r: g.copy() for r, g in p.items()} for p in original[s]] for s in original}
        pair = broken['train'][0]
        target = data['target']
        if defect in ('highlight', 'decoy'):
            lane = next(s for s in data['lanes'] if s['eligible']) if defect == 'highlight' else next(
                s for s in data['lanes'] if s is not target and not s['eligible'])
            token = lane['tokens'][0]
            r0, r1, c0, c1 = data['bounds'][lane['panel']]
            point = np.argwhere(token['tile'] != data['background'])[0]
            y, x = r0 + token['y'] + int(point[0]), c0 + token['x'] + int(point[1])
        elif defect == 'bar':
            r0, r1, c0, c1 = data['bounds'][target['panel']]
            y, x = r0 + data['bar']['y'], c0 + data['bar']['x']
        else:
            r0, r1, c0, c1 = data['bounds'][target['panel']]
            sign = -1 if target['panel'] // 2 == 0 else 1
            y = r0 + target['inward_y'] + sign * target['count'] * (target['tile'].shape[0] + target['unit'])
            point = np.argwhere(target['tile'] != data['background'])[0]
            y, x = y + int(point[0]), c0 + target['x'] + int(point[1])
        pair['output'][y, x] = (int(pair['output'][y, x]) + 1) % 10
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('undetected ' + defect)
    return {'seeds': 42, 'pairs': pair_count, 'shapes': len(shapes),
            'motifs': len(motifs), 'palettes': len(palettes),
            'count_configurations': len(counts), 'ratios': sorted(ratios),
            'same_panel_pool_pairs': same_panel, 'divider_content_pairs': divider_content,
            'tall_motifs': tall_motifs, 'fallback_cases': fallback_count}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
