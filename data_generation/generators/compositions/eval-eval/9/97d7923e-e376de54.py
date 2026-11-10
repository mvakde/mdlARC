"""Ranked mirror calibration: unary instructions rank errors about a middle tip.

Only the standard library and NumPy are used. Sampling records draw labels;
the solver independently reconstructs the visible objects from input pixels.
"""
from __future__ import annotations

from collections import deque
from numbers import Integral

import numpy as np


AXES = ((0, 1), (1, 0), (1, 1), (1, -1))
HOST_SEEDS = (0, 604244500, 472168554, 744161995, 319358240, 96790628,
              912001480, 787537447, 178604819, 202265691, 957260908,
              615330970, 367863200, 461446195, 427522185, 528940277,
              665896193, 465233191, 954004807, 818856318, 779621965,
              714956435, 431822120, 20756244, 808301520, 704765776,
              777501830, 180614795, 546485627, 250497985, 895726285,
              886641684, 337651809, 773825409, 852954455, 219554218,
              554336177, 869460989, 300644032, 661102477, 27852596, 345926384)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid colors")
    return grid.astype(np.int8, copy=False)


def _signature(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.tobytes()


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p['input']), _signature(p['output']))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _components(grid, background):
    """Eight-neighbor components; spacing is a visible family convention."""
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        queue, component = deque([first]), []
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    point = (r + dr, c + dc)
                    if point in remaining:
                        remaining.remove(point)
                        queue.append(point)
        result.append(component)
    return result


def _straight(component, direction):
    dr, dc = direction
    points = sorted(component, key=lambda p: p[0] * dr + p[1] * dc)
    return (all((b[0] - a[0], b[1] - a[1]) == direction
                for a, b in zip(points, points[1:])), points)


def _analyze(input_grid):
    """Fresh input-only parse, with no random state or generation metadata."""
    grid = _grid(input_grid)
    frequencies = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    _require(frequencies[bg] > grid.size / 2, "background must occupy over half the grid")
    keys, raw_bars = [], []
    for component in _components(grid, bg):
        colors = {int(grid[p]) for p in component}
        if len(colors) == 1:
            keys.append({'color': next(iter(colors)), 'rank': len(component),
                         'cells': component})
            continue
        _require(len(component) >= 3 and len(colors) == 2, "malformed capped bar")
        possibilities = []
        for axis in AXES:
            straight, points = _straight(component, axis)
            if not straight:
                continue
            cap, payload = int(grid[points[0]]), int(grid[points[1]])
            if (int(grid[points[-1]]) == cap and payload != cap
                    and all(int(grid[p]) == payload for p in points[1:-1])):
                possibilities.append((axis, points, cap, payload))
        _require(len(possibilities) == 1, "bar axis or color pattern is ambiguous")
        axis, points, cap, payload = possibilities[0]
        raw_bars.append({'axis': axis, 'cells': points, 'cap': cap,
                         'payload': payload, 'length': len(points)})
    _require(len(raw_bars) >= 3 and len(raw_bars) % 2 == 1, "need an odd bar array")
    _require(len({b['axis'] for b in raw_bars}) == 1, "bars must be parallel")
    axis = raw_bars[0]['axis']
    dr, dc = axis
    low = [b['cells'][0][0] * dr + b['cells'][0][1] * dc for b in raw_bars]
    high = [b['cells'][-1][0] * dr + b['cells'][-1][1] * dc for b in raw_bars]
    minimum_common, maximum_common = len(set(low)) == 1, len(set(high)) == 1
    _require(minimum_common != maximum_common, "anchored side must be unique")
    direction = axis if minimum_common else (-dr, -dc)
    dr, dc = direction
    perpendicular = (dc, -dr)
    bars = []
    for bar in raw_bars:
        cells = bar['cells'] if minimum_common else bar['cells'][::-1]
        bars.append(dict(bar, cells=cells, anchor=cells[0],
                         position=cells[0][0] * dc - cells[0][1] * dr))
    bars.sort(key=lambda b: b['position'])
    unit = dr * dr + dc * dc
    gaps = [b['position'] - a['position'] for a, b in zip(bars, bars[1:])]
    _require(all(g >= 2 * unit and g % unit == 0 for g in gaps), "bars are too close")
    reference = bars[len(bars) // 2]['length']
    _require(reference >= 3, "reference needs two caps and an interior")
    controls = {b['cap'] for b in bars}
    _require(1 <= len(controls) <= 8, "invalid family count")
    _require(len(keys) == len(controls) and {k['color'] for k in keys} == controls,
             "one unary key is required per cap color")
    _require(all(b['payload'] not in controls for b in bars), "cap and payload roles overlap")
    baseline = bars[0]['anchor'][0] * dr + bars[0]['anchor'][1] * dc
    key_tips = []
    key_positions = []
    for key in keys:
        straight, points = _straight(key['cells'], direction)
        _require(straight, "unary keys must be straight and parallel to the bars")
        tip = points[-1][0] * dr + points[-1][1] * dc
        key_tips.append(tip)
        key_positions.append(points[-1][0] * dc - points[-1][1] * dr)
        _require(tip <= baseline - 3 * unit, "key must be behind the anchor baseline")
    _require(len(set(key_tips)) == 1, "unary key tips must align")
    _require(max(key_positions) - min(key_positions) <= 3 * len(keys) * unit,
             "unary key bank must be compact")
    ranks = {k['color']: k['rank'] for k in keys}
    families, selected = {}, []
    for cap in sorted(controls):
        family = [b for b in bars if b['cap'] == cap]
        _require(len(family) >= 2, "every family needs at least two competitors")
        errors = [abs(b['length'] - reference) for b in family]
        _require(len(set(errors)) == len(errors), "error magnitudes must not tie within a family")
        _require(1 <= ranks[cap] <= len(family), "requested rank does not exist")
        for b in family:
            target = 2 * reference - b['length']
            _require(target >= 3, "a reflected bar must retain two caps and an interior")
            endpoint = (b['anchor'][0] + (target - 1) * dr,
                        b['anchor'][1] + (target - 1) * dc)
            _require(0 <= endpoint[0] < grid.shape[0] and 0 <= endpoint[1] < grid.shape[1],
                     "reflected tip is out of bounds")
        families[cap] = family
        selected.append(sorted(family, key=lambda b: abs(b['length'] - reference),
                               reverse=True)[ranks[cap] - 1])
    return {'grid': grid, 'background': bg, 'direction': direction,
            'perpendicular': perpendicular, 'bars': bars, 'keys': keys,
            'ranks': ranks, 'families': families, 'reference': reference,
            'gaps': tuple(g // unit for g in gaps), 'selected': selected}


def _render_answer(analysis, selected=None, reference=None):
    result = analysis['grid'].copy()
    dr, dc = analysis['direction']
    reference = analysis['reference'] if reference is None else reference
    selected = analysis['selected'] if selected is None else selected
    for bar in selected:
        for p in bar['cells']:
            result[p] = analysis['background']
        span = 2 * reference - bar['length']
        _require(span >= 3, "invalid reflected length")
        r, c = bar['anchor']
        for t in range(span):
            p = (r + t * dr, c + t * dc)
            _require(0 <= p[0] < result.shape[0] and 0 <= p[1] < result.shape[1],
                     "reflected tip is out of bounds")
            result[p] = bar['cap'] if t in (0, span - 1) else bar['payload']
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reflect each color family's keyed error-rank bar about the middle tip."""
    return _render_answer(_analyze(input_grid))


def _direction(index):
    dr, dc = AXES[index // 2]
    sign = 1 if index % 2 == 0 else -1
    return dr * sign, dc * sign


def _sequence(rng, counts, mode):
    order = [int(i) for i in rng.permutation(len(counts))]
    if mode == 'grouped':
        return [f for f in order for _ in range(counts[f])]
    if mode == 'shuffle':
        sequence = [f for f in order for _ in range(counts[f])]
        rng.shuffle(sequence)
        return sequence
    chunks = []
    for f in order:
        remaining = counts[f]
        while remaining:
            size = int(rng.integers(1, remaining + 1))
            chunks.append([f] * size)
            remaining -= size
    rng.shuffle(chunks)
    return [f for chunk in chunks for f in chunk]


def _case(rng, direction, counts_allowed, rank_limit, witness=0,
          force_rank_three=False, caps_fixed=None, avoid_reference=None):
    """Draw directly from geometric records; labels never call solve."""
    diagonal = all(direction)
    # A bounded fit search combines population, spacing, key and axial budgets.
    for attempt in range(160):
        n = 5 if witness else int(rng.choice(counts_allowed))
        f = 2 if witness else int(rng.integers(1, min(4, n // 2) + 1))
        counts = [2] * f
        for _ in range(n - 2 * f):
            counts[int(rng.integers(f))] += 1
        if witness:
            counts = [3, 2]
            sequence = [0, 1, 0, 1, 0]
            if bool(rng.integers(2)):
                sequence = [1, 0, 0, 0, 1]
        else:
            sequence = _sequence(rng, counts, str(rng.choice(('grouped', 'chunks', 'shuffle'))))
        central_family = sequence[n // 2]
        ranks = [witness] * f if witness else [int(rng.integers(1, min(rank_limit, c) + 1)) for c in counts]
        if force_rank_three:
            eligible = [i for i, c in enumerate(counts) if c - (i == central_family) >= 3]
            if not eligible:
                continue
            ranks[int(rng.choice(eligible))] = 3
        key_gap = 3 if witness else int(rng.integers(3, 6))
        rear = key_gap + max(ranks) - 1
        needed = 4 if witness else max(c - (i == central_family) for i, c in enumerate(counts))
        minimum_span = 2 * (n - 1)
        max_span = min(29, 30 - rear - (2 * needed + 3)) if diagonal else 29
        if minimum_span + (3 if witness else 0) > max_span:
            continue
        gaps = [2] * (n - 1)
        if witness:
            extras = [e for e in (3, 5, 7, 9) if e <= max_span - minimum_span]
            gaps[0 if bool(rng.integers(2)) else -1] += int(rng.choice(extras))
        elif rng.random() < .35:
            gap = int(rng.integers(2, min(4, max_span // (n - 1)) + 1))
            gaps = [gap] * (n - 1)
        else:
            extra = int(rng.integers(0, min(10, max_span - minimum_span) + 1))
            for _ in range(extra):
                gaps[int(rng.integers(n - 1))] += 1
        positions = [0] + [int(x) for x in np.cumsum(gaps)]
        capacity = 30 - rear - (positions[-1] if diagonal else 0)
        max_deviation = min(10, (capacity - 3) // 2)
        if max_deviation < needed:
            continue
        deviation_ceiling = int(rng.integers(needed, max_deviation + 1))
        choices = [r for r in range(3 + deviation_ceiling, capacity - deviation_ceiling + 1)
                   if r != avoid_reference]
        if not choices:
            continue
        reference = int(rng.choice(choices))
        break
    else:
        raise RuntimeError('no compatible geometry found in bounded sampling')

    lengths = [reference] * n
    if witness:
        # Three short bars and one long bar make the middle nonmedian.
        # Four distinct magnitudes a < b < c < d, with c=b+1, make the
        # second family's ranking reverse when the reference grows one cell.
        b = int(rng.integers(2, deviation_ceiling - 1))
        c = b + 1
        a = int(rng.integers(1, b))
        d = int(rng.integers(c + 1, deviation_ceiling + 1))
        noncentral0 = [i for i, s in enumerate(sequence) if s == 0 and i != n // 2]
        noncentral1 = [i for i, s in enumerate(sequence) if s == 1]
        rng.shuffle(noncentral0)
        rng.shuffle(noncentral1)
        lengths[noncentral0[0]] = reference - d
        lengths[noncentral0[1]] = reference - a
        lengths[noncentral1[0]] = reference + c
        lengths[noncentral1[1]] = reference - b
    else:
        pattern = str(rng.choice(('mixed', 'extend', 'trim'), p=(.6, .2, .2)))
        for family in range(f):
            members = [i for i, s in enumerate(sequence) if s == family and i != n // 2]
            deviations = [int(d) for d in rng.choice(np.arange(1, deviation_ceiling + 1), len(members), replace=False)]
            if family != central_family and rng.random() < .25:
                deviations[0] = 0
            for i, deviation in zip(members, deviations):
                sign = -1 if pattern == 'extend' else 1 if pattern == 'trim' else int(rng.choice((-1, 1)))
                lengths[i] = reference + sign * deviation

    def chosen_indices():
        return [sorted([i for i, s in enumerate(sequence) if s == family],
                       key=lambda i: abs(lengths[i] - reference), reverse=True)[ranks[family] - 1]
                for family in range(f)]

    if all(lengths[i] == reference for i in chosen_indices()):
        ranks[central_family] = 1
    selected = set(chosen_indices())
    colors = [int(c) for c in rng.permutation(10)]
    caps = list(caps_fixed) if witness else colors[1:f + 1]
    if witness:
        bg = int(rng.choice([c for c in range(10) if c not in caps]))
    else:
        bg = colors[0]
    payload_pool = [c for c in range(10) if c not in caps and c != bg]
    rng.shuffle(payload_pool)
    style = 'common' if witness else str(rng.choice(('common', 'family', 'bar')))
    if style == 'common':
        payloads = [payload_pool[0]] * n
    elif style == 'family':
        classes = int(rng.integers(1, min(f, len(payload_pool)) + 1))
        payloads = [payload_pool[s % classes] for s in sequence]
    else:
        classes = int(rng.integers(1, min(4, len(payload_pool)) + 1))
        payloads = [int(rng.choice(payload_pool[:classes])) for _ in range(n)]
    dr, dc = direction
    pr, pc = dc, -dr
    key_order = [int(i) for i in rng.permutation(f)]
    key_start = int(rng.integers(0, positions[-1] - 2 * (f - 1) + 1))
    input_cells, output_cells = {}, {}
    reserve = []
    for i, p in enumerate(positions):
        anchor = (p * pr, p * pc)
        output_length = 2 * reference - lengths[i] if i in selected else lengths[i]
        # Reserve every possible reflection, including currently unselected bars.
        for t in (0, max(lengths[i], 2 * reference - lengths[i]) - 1):
            reserve.append((anchor[0] + t * dr, anchor[1] + t * dc))
        for cells, span in ((input_cells, lengths[i]), (output_cells, output_length)):
            for t in range(span):
                point = (anchor[0] + t * dr, anchor[1] + t * dc)
                _require(point not in cells, 'construction overlap')
                cells[point] = caps[sequence[i]] if t in (0, span - 1) else payloads[i]
    for j, family in enumerate(key_order):
        p = key_start + 2 * j
        for t in range(ranks[family]):
            point = (p * pr - (key_gap + t) * dr, p * pc - (key_gap + t) * dc)
            _require(point not in input_cells and point not in output_cells, 'key overlap')
            input_cells[point] = output_cells[point] = caps[family]
            reserve.append(point)
    low = tuple(min(p[k] for p in reserve) for k in (0, 1))
    box = tuple(max(p[k] for p in reserve) - low[k] + 1 for k in (0, 1))
    shapes = [(h, w) for h in range(box[0], 31) for w in range(box[1], 31)
              if h * w > 2 * max(len(input_cells), len(output_cells))]
    _require(shapes, 'no canvas can fit scene')
    if rng.random() < .5:
        preferred = [s for s in shapes if s[0] <= box[0] + 4 and s[1] <= box[1] + 4] or shapes
    else:
        preferred = shapes
    shape = preferred[int(rng.integers(len(preferred)))]
    margins = [shape[k] - box[k] for k in (0, 1)]
    offset = [int(rng.integers(m + 1)) - low[k] for k, m in enumerate(margins)]
    inp, out = np.full(shape, bg, dtype=np.int8), np.full(shape, bg, dtype=np.int8)
    for grid, cells in ((inp, input_cells), (out, output_cells)):
        for (r, c), color in cells.items():
            grid[r + offset[0], c + offset[1]] = color
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(('rank', 'population', 'direction')))
    base = int(rng.integers(4 if shift == 'population' else 8))
    direction = _direction(base)
    cap_pair = [int(c) for c in rng.choice(10, 2, replace=False)]
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_reference = None
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = index + 1 if split == 'train' and index < 2 else 0
            pair_direction = direction
            if split == 'test' and shift == 'direction':
                # Reverse or turn to an unseen axis; all keys remain visibly behind anchors.
                target = (base ^ 1) if index % 2 == 0 else (base + 2) % 8
                pair_direction = _direction(target)
            counts = (5, 7) if split == 'train' and shift == 'population' else (5, 7, 9, 11)
            if split == 'test' and shift == 'population':
                counts = (9, 11)
            force_three = split == 'test' and shift == 'rank'
            for attempt in range(40):
                pair = _case(rng, pair_direction, counts,
                             2 if split == 'train' else 3, witness,
                             force_three, cap_pair, previous_reference if witness == 2 else None)
                signature = _signature(pair['input'])
                if signature not in seen:
                    break
            else:
                raise RuntimeError('failed to draw a distinct input')
            if witness == 1:
                previous_reference = _analyze(pair['input'])['reference']
            seen.add(signature)
            puzzle[split].append(pair)
    # Witness order is varied without removing the controlled 1/2 contrast.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(analysis):
    bars, ref = analysis['bars'], analysis['reference']
    positions = np.array([b['position'] for b in bars])
    distances = abs(2 * positions - positions[0] - positions[-1])
    nearest = np.flatnonzero(distances == distances.min())
    middle = len(bars) // 2
    raw_selected = [sorted(family, key=lambda b: b['length'], reverse=True)[analysis['ranks'][cap] - 1]
                    for cap, family in analysis['families'].items()]
    extensions = sum(b['length'] < ref for b in analysis['selected'])
    trims = sum(b['length'] > ref for b in analysis['selected'])
    ordered_caps = list(dict.fromkeys(b['cap'] for b in bars))
    return {'count': len(bars), 'families': len(analysis['families']), 'reference': ref,
            'ranks': tuple(sorted(analysis['ranks'].values())),
            'gaps': analysis['gaps'], 'direction': analysis['direction'],
            'background': analysis['background'], 'shape': analysis['grid'].shape,
            'extensions': extensions, 'trims': trims,
            'nonmedian': sorted(b['length'] for b in bars)[middle] != ref,
            'geometric_witness': len(nearest) == 1 and int(nearest[0]) != middle
                                 and bars[int(nearest[0])]['length'] != ref,
            'raw_rank_differs': {b['anchor'] for b in raw_selected}
                                != {b['anchor'] for b in analysis['selected']},
            'structure': (len(bars), tuple(b['length'] for b in bars),
                          tuple(analysis['gaps']),
                          tuple(tuple(i for i, b in enumerate(bars) if b['cap'] == cap)
                                for cap in ordered_caps),
                          tuple(analysis['ranks'][cap] for cap in ordered_caps))}


def _check_unsampled_members():
    """Small explicit witnesses outside the generator's sampled count limits."""
    fixtures = (
        ([3, 5, 6], [1, 1, 1], {1: 2}, (2,)),
        ([4, 8, 9, 12, 15], [1] * 5, {1: 4}, (1,)),
        ([4, 8, 9, 12, 15], [1] * 5, {1: 5}, (2,)),
        ([3, 10, 8, 4, 3, 7, 5, 11, 9, 8, 6],
         [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
         {1: 2, 2: 1, 3: 2, 4: 1, 5: 1}, (10, 1, 2, 3, 4)),
    )
    for lengths, caps, ranks, selected in fixtures:
        reference = lengths[len(lengths) // 2]
        anchor = max(ranks.values()) + 2
        capacity = max(max(lengths), 2 * reference - min(lengths))
        inp = np.zeros((anchor + capacity + 1, 2 * len(lengths) + 1), dtype=np.int8)
        out = inp.copy()
        for i, (length, cap) in enumerate(zip(lengths, caps)):
            for grid, span in ((inp, length), (out, 2 * reference - length if i in selected else length)):
                grid[anchor:anchor + span, 2 * i] = 8
                grid[anchor, 2 * i] = grid[anchor + span - 1, 2 * i] = cap
        for j, (cap, rank) in enumerate(ranks.items()):
            inp[anchor - rank - 2:anchor - 2, 2 * j + 1] = cap
            out[anchor - rank - 2:anchor - 2, 2 * j + 1] = cap
        for turns in range(4):
            for mirror in (False, True):
                x, y = np.rot90(inp, turns), np.rot90(out, turns)
                if mirror:
                    x, y = np.fliplr(x), np.fliplr(y)
                _require(np.array_equal(solve(x), y), 'unsampled legal family member failed')


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'invalid example counts')
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'canvas must be preserved')
            signature = _signature(inp)
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            analysis = _analyze(inp)
            _require(np.array_equal(out, _render_answer(analysis)), 'incorrect mirror repair')
            _require(np.any(inp != out), 'each generated example needs a repair')
            # Invariants checked apart from whole-output equality.
            ref, selected = analysis['reference'], {x['anchor'] for x in analysis['selected']}
            for key in analysis['keys']:
                _require(all(inp[p] == out[p] for p in key['cells']), 'key changed')
            for bar in analysis['bars']:
                _require(out[bar['anchor']] == bar['cap'], 'anchor cap moved')
                if bar['anchor'] not in selected:
                    _require(all(inp[p] == out[p] for p in bar['cells']), 'unselected bar changed')
                else:
                    dr, dc = analysis['direction']
                    span = 2 * ref - bar['length']
                    tip = (bar['anchor'][0] + (span - 1) * dr,
                           bar['anchor'][1] + (span - 1) * dc)
                    _require(out[tip] == bar['cap'], 'reflected free cap missing')
                    _require(span - ref == -(bar['length'] - ref), 'signed error did not reverse')
            center = analysis['bars'][len(analysis['bars']) // 2]
            _require(all(inp[p] == out[p] for p in center['cells']), 'middle reference changed')
            analyses[split].append(analysis)
    train = analyses['train']
    features = [_features(x) for x in train]
    _require(len({x['reference'] for x in features}) >= 2, 'training needs variable mirror positions')
    _require(any(x['nonmedian'] for x in features), 'numerical median shortcut survives')
    _require(any(x['geometric_witness'] for x in features), 'geometric midpoint shortcut survives')
    _require(any(x['raw_rank_differs'] for x in features), 'raw length ranking shortcut survives')
    _require(any(x['extensions'] for x in features) and any(x['trims'] for x in features),
             'training must show both directions of reflection')
    history = {}
    for analysis in train:
        for cap, rank in analysis['ranks'].items():
            history.setdefault(cap, set()).add(rank)
    _require(any({1, 2} <= ranks for ranks in history.values()), 'missing unary key contrast')
    _require(any(analysis['ranks'][cap] == 2 and len(family) >= 3
                 for analysis in train for cap, family in analysis['families'].items()),
             'training needs an intermediate rank')
    # A-only repair means selecting by raw length; B-only means repairing all bars.
    # Also check that merging cap families into a global pool changes a label.
    grouping_witness = False
    for analysis in train:
        ref = analysis['reference']
        for cap, chosen in zip(analysis['families'], analysis['selected']):
            global_order = sorted(analysis['bars'], key=lambda x: abs(x['length'] - ref), reverse=True)
            grouping_witness |= global_order[analysis['ranks'][cap] - 1]['anchor'] != chosen['anchor']
    _require(grouping_witness, 'cap grouping is not demonstrated')
    _require(any(not np.array_equal(_render_answer(x, selected=x['bars']), _render_answer(x))
                 for x in train), 'omitting keyed selection does not change training answers')
    demonstrated = {r for x in train for r in x['ranks'].values()}
    for analysis in analyses['test']:
        _require(all(r in demonstrated or set(range(1, r)) <= demonstrated
                     for r in analysis['ranks'].values()), 'test ordinal lacks demonstrated progression')


def self_test() -> dict:
    """Bounded 42-seed semantic, mutation, equivariance and diversity checks."""
    records, signatures = [], set()
    key_mutations, reference_mutations, reference_reorderings = 0, 0, 0
    rng = np.random.default_rng(97531)
    _check_unsampled_members()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed)), 'nondeterminism')
        signatures.add(_puzzle_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                analysis = _analyze(pair['input'])
                records.append((split, _features(analysis)))
            # Both reflections, all rotations, and a palette permutation commute.
            pair = puzzle[split][0]
            for turns in range(4):
                for mirror in (False, True):
                    x, y = np.rot90(pair['input'], turns), np.rot90(pair['output'], turns)
                    if mirror:
                        x, y = np.fliplr(x), np.fliplr(y)
                    _require(np.array_equal(solve(x), y), 'rotation/reflection equivariance failed')
            palette = rng.permutation(10)
            _require(np.array_equal(solve(palette[pair['input']]), palette[pair['output']]),
                     'palette equivariance failed')
        # Causal interventions: change only a visible instruction or the central
        # exemplar, then reparse. Reference changes must also be able to reorder
        # another family's candidates, not merely change a drawing constant.
        key_changed = reference_changed = False
        for pair in puzzle['train']:
            a = _analyze(pair['input'])
            if not key_changed:
                for key in a['keys']:
                    if key['rank'] == 2:
                        altered = pair['input'].copy()
                        _, cells = _straight(key['cells'], a['direction'])
                        altered[cells[0]] = a['background']
                        parsed = _analyze(altered)
                        _require(parsed['reference'] == a['reference'], 'key altered the reference')
                        if not np.array_equal(solve(altered), pair['output']):
                            key_mutations += 1
                            key_changed = True
                            break
            if not reference_changed and len(a['bars']) == 5 and len(a['families']) == 2:
                center = a['bars'][2]
                dr, dc = a['direction']
                tip = (center['cells'][-1][0] + dr, center['cells'][-1][1] + dc)
                if not (0 <= tip[0] < pair['input'].shape[0] and 0 <= tip[1] < pair['input'].shape[1]):
                    continue
                altered = pair['input'].copy()
                altered[center['cells'][-1]] = center['payload']
                altered[tip] = center['cap']
                try:
                    parsed = _analyze(altered)
                except AssertionError:
                    continue
                _require(parsed['ranks'] == a['ranks'], 'reference altered the keys')
                _require(parsed['reference'] == a['reference'] + 1, 'reference mutation was not read')
                _require(not np.array_equal(solve(altered), pair['output']), 'reference is not causally used')
                reference_mutations += 1
                reference_changed = True
                reference_reorderings += ({b['anchor'] for b in parsed['selected']}
                                          != {b['anchor'] for b in a['selected']})
        _require(key_changed, 'no key-sensitive training witness')
        # Remove exactly one erroneous output cell and require rejection.
        bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
        p = bad['test'][0]
        r, c = np.argwhere(p['input'] != p['output'])[0]
        p['output'][r, c] = p['input'][r, c]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('damaged answer was accepted')
    _require(len(signatures) == 42, 'cross-seed puzzle diversity failed')
    _require(reference_mutations >= 10 and reference_reorderings >= 10,
             'central-reference causal checks lack coverage')
    rows = [r for _, r in records]
    _require({5, 7, 9, 11} <= {r['count'] for r in rows}, 'bar population coverage')
    _require({1, 2, 3, 4} <= {r['families'] for r in rows}, 'family count coverage')
    _require(len({r['direction'] for r in rows}) == 8, 'directed axis coverage')
    _require({1, 2, 3} <= {k for r in rows for k in r['ranks']}, 'ordinal coverage')
    _require(len({r['reference'] for r in rows}) >= 8, 'reference length diversity')
    _require(len({r['shape'] for r in rows}) >= 40, 'canvas diversity')
    _require(len({r['structure'] for r in rows}) >= 150, 'geometric structure diversity')
    _require({r['background'] for r in rows} == set(range(10)), 'background color coverage')
    _require(any(len(set(r['gaps'])) == 1 for r in rows)
             and any(len(set(r['gaps'])) > 1 for r in rows), 'spacing diversity')
    for num_train in range(2, 11):
        for num_test in range(1, min(4, num_train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(0, num_train, num_test))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'pairs': len(rows), 'distinct_puzzles': len(signatures),
            'bar_counts': sorted({r['count'] for r in rows}),
            'family_counts': sorted({r['families'] for r in rows}),
            'directions': len({r['direction'] for r in rows}),
            'references': sorted({r['reference'] for r in rows}),
            'shapes': len({r['shape'] for r in rows}),
            'structures': len({r['structure'] for r in rows}),
            'key_mutations': key_mutations, 'reference_mutations': reference_mutations,
            'reference_reorderings': reference_reorderings,
            'independent_semantic_verification': False}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
