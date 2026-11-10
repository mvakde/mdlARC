"""Counted jigsaw stamps: geometric exact cover defines unary addition groups.

Only the standard library and NumPy are used. Prediction reads a single grid;
construction metadata is never supplied to the parser or assembly search.
"""
from functools import lru_cache
from itertools import product

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _runs(values):
    d = np.diff(np.r_[False, values, False].astype(int))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _connected(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    if not remaining:
        return False
    pending = [remaining.pop()]
    while pending:
        r, c = pending.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in remaining:
                remaining.remove(p)
                pending.append(p)
    return not remaining


def _layer(lo, hi):
    """A full-width, column-convex layer in outward/cross coordinates."""
    return np.array([(u, c) for c, (a, b) in enumerate(zip(lo, hi))
                     for u in range(int(a), int(b))], dtype=int)


def _layer_connected(lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    return bool(np.all(hi > lo) and
                np.all(np.maximum(lo[:-1], lo[1:]) < np.minimum(hi[:-1], hi[1:])))


def _description(cells):
    # Canonical receivers grow upward, so the outward coordinate is minus row.
    cross = cells[:, 1] - cells[:, 1].min()
    outward = -cells[:, 0]
    lo, hi = [], []
    for c in range(int(cross.max()) + 1):
        values = np.sort(outward[cross == c])
        _require(len(values) > 0 and
                 np.array_equal(values, np.arange(values[0], values[-1] + 1)),
                 "tokens must be column-convex and full-width")
        lo.append(int(values[0]))
        hi.append(int(values[-1]) + 1)
    origin = min(lo)
    lo = tuple(v - origin for v in lo)
    hi = tuple(v - origin for v in hi)
    _require(_layer_connected(lo, hi), "token is disconnected")
    return lo, hi, np.column_stack((outward - origin, cross))


def _parse(input_grid):
    original = _grid(input_grid)
    histogram = np.bincount(original.ravel(), minlength=10)
    bg = int(np.argmax(histogram))
    _require(int(histogram[bg]) > max(int(v) for c, v in enumerate(histogram) if c != bg),
             "background must be the unique most frequent input color")
    bands = []
    for transposed in (False, True):
        a = original.T if transposed else original
        rows = np.flatnonzero(np.all(a == a[:, :1], axis=1) & (a[:, 0] != bg))
        if len(rows):
            _require(np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)),
                     "spanning band must be contiguous")
            _require(len(np.unique(a[rows, 0])) == 1, "band must be monochrome")
            bands.append((transposed, int(rows[0]), int(rows[-1]) + 1))
    _require(len(bands) == 1, "expected exactly one spanning band axis")
    transposed, start, stop = bands[0]
    a = (original.T if transposed else original).copy()
    flipped = start < a.shape[0] - stop
    if flipped:
        a = a[::-1].copy()
        start, stop = a.shape[0] - stop, a.shape[0] - start
    band = int(a[start, 0])
    _require(1 <= stop - start <= 3 and 0 <= a.shape[0] - stop <= 2,
             "invalid band thickness or outer margin")
    _require(bool(np.all(a[stop:] == bg)), "objects must be on the inner side of the band")
    _require(bool(np.all(a[0] == bg)), "inputs must have a blank inner edge")
    _require(start >= 6, "insufficient working area")
    lanes = _runs(np.any(a[:start] != bg, axis=0))
    sources, receivers, unmarked = [], [], []
    for x, end in lanes:
        strip = a[:start, x:end]
        groups = _runs(np.any(strip != bg, axis=1))
        marker_cells = np.argwhere(strip == band)
        marked = len(marker_cells) > 0
        if marked:
            _require(np.array_equal(marker_cells,
                     np.array([(start - 2, c) for c in range(end - x)])),
                     "source control must span its lane one row before the band")
            _require(groups[-1] == (start - 2, start - 1), "source bar is not isolated")
            groups = groups[:-1]
        _require(bool(groups), "empty token lane")
        colors, masks, coordinates = [], [], []
        for top, bottom in groups:
            block = strip[top:bottom]
            mask = block != bg
            values = np.unique(block[mask])
            _require(len(values) == 1 and int(values[0]) != band,
                     "each token must be monochrome and distinct from the band")
            _require(_connected(mask), "disconnected token")
            _require(bool(np.all(np.any(mask, axis=0))), "token does not span its lane")
            colors.append(int(values[0]))
            masks.append(mask)
            coordinates.append(np.argwhere(mask) + np.array([top, x]))
        _require(len(set(colors)) == 1 and
                 all(np.array_equal(masks[0], m) for m in masks),
                 "a stack must repeat one unchanged token")
        bottoms = [b for _, b in groups]
        pitch_values = set(b - c for c, b in zip(bottoms, bottoms[1:]))
        _require(len(pitch_values) <= 1, "irregular repetition")
        lo, hi, local = _description(coordinates[-1])
        token = dict(x=x, width=end - x, color=colors[0], count=len(groups),
                     cells=np.concatenate(coordinates), lo=lo, hi=hi, local=local,
                     height=masks[0].shape[0], mask=masks[0])
        if marked:
            _require(bottoms[-1] == start - 3, "source needs one blank row above its bar")
            _require(not pitch_values or pitch_values == {token['height'] + 1},
                     "sources use a one-cell token gap")
            token['marker'] = marker_cells + np.array([0, x])
            sources.append(token)
        elif bottoms[-1] == start:
            _require(set(lo) == {0} and len(set(hi)) > 1,
                     "receiver foundation must be bottom-aligned and jagged")
            token['root'] = hi
            token['pitch'] = next(iter(pitch_values), None)
            _require(token['pitch'] is None or token['pitch'] > token['height'],
                     "receiver foundations must be separated")
            receivers.append(token)
        else:
            _require(bottoms[-1] <= start - 1, "unmarked stack touches band")
            _require(not pitch_values or pitch_values == {token['height'] + 1},
                     "unmarked stack has irregular token gaps")
            unmarked.append(token)
    _require(bool(sources) and bool(receivers), "missing source or receiver")
    return dict(grid=a, original=original, bg=bg, band=band, baseline=start,
                band_stop=stop, transposed=transposed, flipped=flipped,
                sources=sources, receivers=receivers, unmarked=unmarked)


def _restore(a, scene):
    if scene['flipped']:
        a = a[::-1]
    if scene['transposed']:
        a = a.T
    return a.copy()


def _put(a, baseline, x, cells, color, offset=0):
    rows = baseline - 1 - (cells[:, 0] + offset)
    columns = cells[:, 1] + x
    _require(bool(np.all((rows >= 0) & (rows < baseline))), "completion clips the canvas")
    a[rows, columns] = color


def _candidate_groups(scene, receiver):
    sources, limit = scene['sources'], scene['baseline']
    options = []

    def visit(profile, bits, placements):
        if bits and len(set(profile)) == 1:
            length = profile[0]
            pitch = length + 1
            total = sum(s['count'] for i, s in enumerate(sources) if bits & (1 << i))
            final = receiver['count'] + total
            if ((receiver['pitch'] is None or receiver['pitch'] == pitch) and
                    final * pitch - 1 <= limit):
                options.append(dict(bits=bits, length=length, pitch=pitch,
                                    total=total, final=final, placements=placements))
        offset = min(profile)
        normalized = tuple(v - offset for v in profile)
        for i, source in enumerate(sources):
            if bits & (1 << i) or source['width'] != receiver['width']:
                continue
            if source['lo'] != normalized:
                continue
            next_profile = tuple(offset + v for v in source['hi'])
            if max(next_profile) >= limit:
                continue
            visit(next_profile, bits | (1 << i), placements + [(i, offset)])

    visit(tuple(receiver['root']), 0, [])
    return options


def _render(scene, chosen):
    a = scene['grid'].copy()
    for source in scene['sources']:
        cells = source['cells']
        a[cells[:, 0], cells[:, 1]] = scene['band']
        cells = source['marker']
        a[cells[:, 0], cells[:, 1]] = scene['bg']
    for receiver, option in zip(scene['receivers'], chosen):
        foundation = _layer([0] * receiver['width'], receiver['root'])
        for k in range(option['final']):
            unit_offset = k * option['pitch']
            _put(a, scene['baseline'], receiver['x'], foundation, receiver['color'], unit_offset)
            for i, offset in option['placements']:
                source = scene['sources'][i]
                _put(a, scene['baseline'], receiver['x'], source['local'], source['color'],
                     unit_offset + offset)
    return a


def _analyze(input_grid):
    scene = _parse(input_grid)
    options = [_candidate_groups(scene, r) for r in scene['receivers']]
    all_bits = (1 << len(scene['sources'])) - 1
    answers = {}
    scene['nearest_compatible'] = False
    centers = [r['x'] + (r['width'] - 1) / 2 for r in scene['receivers']]

    def combine(index, used, chosen):
        if len(answers) > 1:
            return
        if index == len(options):
            if used == all_bits:
                a = _render(scene, chosen)
                answers.setdefault(a.tobytes(), (a, chosen))
                near = True
                for center, option in zip(centers, chosen):
                    for i, _ in option['placements']:
                        source = scene['sources'][i]
                        source_center = source['x'] + (source['width'] - 1) / 2
                        near &= abs(source_center - center) <= min(abs(source_center - c) for c in centers)
                scene['nearest_compatible'] |= near
            return
        for option in options[index]:
            if not option['bits'] & used:
                combine(index + 1, used | option['bits'], chosen + [option])

    combine(0, 0, [])
    _require(len(answers) == 1, "input needs exactly one distinct completed color grid")
    answer, chosen = next(iter(answers.values()))
    scene['chosen'] = chosen
    return _restore(answer, scene), scene


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from pixels alone, including singleton sources and receivers."""
    return _analyze(input_grid)[0]


@lru_cache(maxsize=64)
def _profile_paths(width, length, pieces, rich):
    """Count connected column-state paths with every interface nonflat.

    Local interval intersections imply global connectivity for full-width layers.
    Dynamic programming avoids rejection of independently drawn incompatible cuts.
    This cache contains only geometric possibilities, never examples or answers.
    """
    root_max = min(3 if rich else 2, length - pieces)
    _require(root_max >= 2, "no room for a jagged foundation")
    ranges = [range(1, root_max + 1)] + [range(i + 1, length - (pieces - i) + 1)
                                               for i in range(1, pieces)]
    states = [s for s in product(*ranges) if all(a < b for a, b in zip(s, s[1:]))]
    edges = []
    for state in states:
        a = (0,) + state + (length,)
        targets = []
        for j, other in enumerate(states):
            b = (0,) + other + (length,)
            if all(max(x, y) < min(z, t) for x, y, z, t in zip(a, b, a[1:], b[1:])):
                changed = sum(1 << k for k, (x, y) in enumerate(zip(state, other)) if x != y)
                targets.append((j, changed))
        edges.append(targets)
    full = (1 << pieces) - 1

    @lru_cache(maxsize=None)
    def ways(index, remaining, changed):
        if remaining == 0:
            return int(changed == full)
        return sum(ways(j, remaining - 1, changed | bits) for j, bits in edges[index])

    starts = [ways(i, width - 1, 0) for i in range(len(states))]
    _require(sum(starts) > 0, "no connected nonflat chain at these dimensions")
    return states, edges, ways, starts


def _weighted_index(rng, weights):
    draw = int(rng.integers(sum(weights)))
    for i, weight in enumerate(weights):
        if draw < weight:
            return i
        draw -= weight
    raise AssertionError("weighted choice failed")


def _profiles(rng, width, length, pieces, rich):
    states, edges, ways, starts = _profile_paths(width, length, pieces, rich)
    index = _weighted_index(rng, starts)
    columns = [states[index]]
    changed = 0
    for remaining in range(width - 1, 0, -1):
        targets = edges[index]
        weights = [ways(j, remaining - 1, changed | bits) for j, bits in targets]
        index, bits = targets[_weighted_index(rng, weights)]
        changed |= bits
        columns.append(states[index])
    cuts = [p.copy() for p in np.array(columns, dtype=int).T]
    cuts.append(np.full(width, length, dtype=int))
    return cuts


def _palette(rng):
    bg = 0 if rng.random() < .55 else int(rng.integers(10))
    band = int(rng.choice([c for c in range(10) if c != bg]))
    pool = [int(c) for c in rng.permutation([c for c in range(10) if c not in (bg, band)])]
    return bg, band, pool


def _operands(rng, pieces, length, require_unequal=False):
    # Reserve up to four rows for the band and outer margin, without clipping.
    capacity = 27 // (length + 1)
    options = [(d, tuple(weights)) for d in range(1, 4)
               for weights in product(range(1, 4), repeat=pieces)
               if d + sum(weights) <= capacity and
               (not require_unequal or len(set(weights)) > 1)]
    _require(bool(options), "operands do not fit")
    return options[int(rng.integers(len(options)))]


def _training_specs(rng):
    # Jointly choose informative comparisons; their counts are not fixed templates.
    length = int(rng.choice([4, 5]))
    d1, weights1 = _operands(rng, 2, length, require_unequal=True)
    first = dict(widths=[int(rng.integers(3, 7))], lengths=[length],
                 counts=[2], operands=[(d1, weights1)], unmarked=1, rich=False)
    choices = [(d, w) for d in range(1, 4) for w in product(range(1, 4), repeat=2)
               if d + sum(w) <= 5]
    distinguished = [(d, w) for d, w in choices
                     if d != d1 and sum(w) != sum(weights1) and
                     d + sum(w) != d1 + sum(weights1) and
                     d * sum(weights1) != d1 * sum(w)]
    d2, weights2 = distinguished[int(rng.integers(len(distinguished)))]
    other = choices[int(rng.integers(len(choices)))]
    width = int(rng.choice([3, 4]))
    second = dict(widths=[width, width], lengths=[4, 4], counts=[2, 2],
                  operands=[(d2, weights2), other], unmarked=0, rich=False)
    specs = [first, second]
    rng.shuffle(specs)
    return specs


def _other_spec(rng, test, shift):
    if test and shift == 0:
        # A third interlocking fragment, with the same addition/completion rule.
        return dict(widths=[int(rng.integers(4, 7))], lengths=[5], counts=[3],
                    operands=[(1, (1, 1, 1))], unmarked=int(rng.integers(2)), rich=False)
    if test and shift == 1:
        count = int(rng.choice([1, 2, 3]))
        return dict(widths=[int(rng.integers(3 if count < 3 else 4, 7))],
                    lengths=[6], counts=[count], operands=[(1, (1,) * count)],
                    unmarked=int(rng.integers(2)), rich=True)
    if test and shift == 2:
        width = int(rng.choice([3, 4, 5]))
        return dict(widths=[width, width], lengths=[3, 4], counts=[1, 2],
                    operands=[(1, (1,)), (1, (1, 1))], unmarked=0, rich=False)
    receivers = 1 if rng.random() < .7 else 2
    if receivers == 1:
        count = int(rng.choice([1, 2]))
        length = int(rng.choice([3, 4, 5] if count == 1 else [4, 5]))
        width = int(rng.integers(3, 7))
        return dict(widths=[width], lengths=[length], counts=[count],
                    operands=[_operands(rng, count, length)],
                    unmarked=int(rng.integers(2)), rich=length >= 5)
    counts = [int(rng.choice([1, 2])) for _ in range(2)]
    widths = [int(rng.choice([3, 4])) for _ in range(2)]
    lengths = [int(rng.choice([3, 4, 5] if k == 1 else [4, 5])) for k in counts]
    return dict(widths=widths, lengths=lengths, counts=counts,
                operands=[_operands(rng, k, l) for k, l in zip(counts, lengths)],
                unmarked=0, rich=False)


def _make_pair(rng, spec, palette, orientation, require_far):
    bg, band, pool = palette
    cuts = [_profiles(rng, w, l, n, spec['rich'])
            for w, l, n in zip(spec['widths'], spec['lengths'], spec['counts'])]
    color_count = sum(spec['counts']) + len(cuts)
    if rng.random() < .35:
        local_pool = pool[:int(rng.integers(2, min(5, color_count) + 1))]
        colors = [int(rng.choice(local_pool)) for _ in range(color_count)]
    else:
        colors = pool[:color_count]
    cursor, receivers, sources = 0, [], []
    for group, profiles in enumerate(cuts):
        d, weights = spec['operands'][group]
        root_color = colors[cursor]
        cursor += 1
        receiver = dict(kind='receiver', group=group, width=spec['widths'][group],
                        color=root_color, root=profiles[0], count=d,
                        length=spec['lengths'][group], weights=weights)
        receivers.append(receiver)
        for lo, hi, weight in zip(profiles, profiles[1:], weights):
            cells = _layer(lo, hi)
            local = cells - np.array([int(cells[:, 0].min()), 0])
            sources.append(dict(kind='source', group=group, width=receiver['width'],
                                color=colors[cursor], count=int(weight), local=local,
                                original=cells))
            cursor += 1
        if all(s['color'] == root_color for s in sources if s['group'] == group):
            raise ValueError("mosaic must visibly contain a fragment color")
    lanes = receivers + sources
    if spec['unmarked']:
        # A shape/color-identical unmarked stack teaches that controls select donors.
        prototype = sources[int(rng.integers(len(sources)))]
        lanes.append(dict(kind='unmarked', group=-1, width=prototype['width'],
                          color=prototype['color'], count=int(rng.integers(1, 4)),
                          local=prototype['local'].copy()))
    minimum_width = sum(l['width'] for l in lanes) + len(lanes) - 1
    _require(minimum_width <= 30, "lanes do not fit")
    width = int(rng.integers(minimum_width, 31))
    rng.shuffle(lanes)
    extras = rng.multinomial(width - minimum_width, np.full(len(lanes) + 1, 1 / (len(lanes) + 1)))
    x = int(extras[0])
    for i, lane in enumerate(lanes):
        lane['x'] = x
        x += lane['width'] + 1 + int(extras[i + 1])
    if require_far:
        centers = {r['group']: r['x'] + (r['width'] - 1) / 2 for r in receivers}
        _require(any(abs(s['x'] + (s['width'] - 1) / 2 - centers[s['group']]) >
                     min(abs(s['x'] + (s['width'] - 1) / 2 - c) for c in centers.values())
                     for s in sources), "training must disprove nearest-receiver routing")
    outward = max((r['count'] + sum(r['weights'])) * (r['length'] + 1) - 1
                  for r in receivers)
    for lane in lanes:
        if lane['kind'] != 'receiver':
            h = int(lane['local'][:, 0].max()) + 1
            lane['height'] = h
            lane['base'] = 3 if lane['kind'] == 'source' else int(rng.integers(1, 4))
            outward = max(outward, lane['base'] + lane['count'] * (h + 1))
    thickness = int(rng.integers(1, 4))
    margin = int(rng.integers(0, min(2, 30 - outward - thickness) + 1))
    minimum_height = outward + thickness + margin
    _require(minimum_height <= 30, "outward growth does not fit")
    height = int(rng.integers(minimum_height, 31))
    baseline = height - thickness - margin
    a = np.full((height, width), bg, dtype=np.int8)
    a[baseline:baseline + thickness] = band
    b = a.copy()
    for lane in lanes:
        if lane['kind'] == 'receiver':
            foundation = _layer([0] * lane['width'], lane['root'])
            for k in range(lane['count']):
                _put(a, baseline, lane['x'], foundation, lane['color'], k * (lane['length'] + 1))
            for k in range(lane['count'] + sum(lane['weights'])):
                offset = k * (lane['length'] + 1)
                _put(b, baseline, lane['x'], foundation, lane['color'], offset)
                for s in sources:
                    if s['group'] == lane['group']:
                        _put(b, baseline, lane['x'], s['original'], s['color'], offset)
        else:
            for k in range(lane['count']):
                offset = lane['base'] + k * (lane['height'] + 1)
                _put(a, baseline, lane['x'], lane['local'], lane['color'], offset)
                _put(b, baseline, lane['x'], lane['local'],
                     band if lane['kind'] == 'source' else lane['color'], offset)
            if lane['kind'] == 'source':
                a[baseline - 2, lane['x']:lane['x'] + lane['width']] = band
    # Whole-scene reflection/transposition changes no input-to-output piece mask.
    if orientation & 1:
        a, b = a[::-1], b[::-1]
    if orientation & 2:
        a, b = a.T, b.T
    if rng.random() < .5:
        a, b = a[:, ::-1], b[:, ::-1]
    pair = dict(input=a.copy(), output=b.copy())
    prediction, parsed = _analyze(pair['input'])
    _require(np.array_equal(prediction, pair['output']), "construction disagrees with unique assembly")
    _require(not require_far or not parsed['nearest_compatible'],
             "training has a nearest-compatible equivalent assignment")
    return pair


def _checked_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _checked_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    shared_palette = _palette(rng)
    stable_palette = bool(rng.integers(2))
    orientation_mode = int(rng.integers(3))
    shared_orientation = int(rng.integers(4))
    shift = int(rng.integers(3))
    specs = _training_specs(rng)
    specs.extend(_other_spec(rng, False, shift) for _ in range(num_train - 2))
    specs.extend(_other_spec(rng, True, (shift + i) % 3) for i in range(num_test))
    for spec in specs:
        # Optional unmarked stacks are selected only when their extra lane fits.
        widths = [w for w, n in zip(spec['widths'], spec['counts']) for _ in range(n + 1)]
        if spec['unmarked'] and sum(widths) + len(widths) + spec['widths'][0] > 30:
            spec['unmarked'] = 0
    puzzle, seen = {'train': [], 'test': []}, set()
    for index, spec in enumerate(specs):
        is_test = index >= num_train
        if orientation_mode == 0:
            orientation = shared_orientation
        elif orientation_mode == 1:
            orientation = shared_orientation ^ (2 if is_test else 0)
        else:
            orientation = int(rng.integers(4))
        palette = shared_palette if stable_palette else _palette(rng)
        for _ in range(180):
            try:
                pair = _make_pair(rng, spec, palette, orientation,
                                  require_far=index < 2 and len(spec['widths']) == 2)
            except ValueError:
                continue
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle['test' if is_test else 'train'].append(pair)
                break
        else:
            raise RuntimeError("bounded unique-scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(scenes):
    operands = [(r['count'], o['total'], len(o['placements']))
                for s in scenes for r, o in zip(s['receivers'], s['chosen'])]
    ds, ss = [d for d, s, n in operands], [s for d, s, n in operands]
    _require(len(set(ds)) > 1 and len(set(ss)) > 1, "training must vary both operands")
    _require(len({d + s for d, s, n in operands}) > 1, "constant-answer shortcut")
    _require(any(ds[0] * s != ss[0] * d for d, s, n in operands), "fixed-multiplier shortcut")
    rivals = (lambda d, s, n: 2 * d, lambda d, s, n: d * s,
              lambda d, s, n: max(d, s), lambda d, s, n: s,
              lambda d, s, n: d + n)
    for rival in rivals:
        _require(any(rival(d, s, n) != d + s for d, s, n in operands),
                 "training does not distinguish simple arithmetic alternative")
    _require(any(len(s['receivers']) > 1 and
                 len({r['width'] for r in s['receivers']}) < len(s['receivers'])
                 for s in scenes), "training must teach shape routing at equal widths")
    _require(any(not s['nearest_compatible'] for s in scenes),
             "training must distinguish geometric matching from nearest-receiver routing")
    _require(any(s['unmarked'] for s in scenes), "training must teach unmarked preservation")
    _require(any(o['total'] > len(o['placements']) for s in scenes for o in s['chosen']),
             "training must distinguish token counts from bar counts")


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    _checked_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "output dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            expected, scene = _analyze(inp)
            _require(np.array_equal(expected, out), "wrong output")
            a = out.T if scene['transposed'] else out
            a = a[::-1] if scene['flipped'] else a
            base = scene['baseline']
            _require(np.array_equal(a[base:], scene['grid'][base:]), "fixed band or outer margin changed")
            for source in scene['sources']:
                cells = source['cells']
                _require(bool(np.all(a[cells[:, 0], cells[:, 1]] == scene['band'])), "source not highlighted")
                cells = source['marker']
                _require(bool(np.all(a[cells[:, 0], cells[:, 1]] == scene['bg'])), "control not erased")
            for token in scene['unmarked']:
                x, width = token['x'], token['width']
                _require(np.array_equal(a[:base, x:x + width], scene['grid'][:base, x:x + width]),
                         "unmarked lane changed")
            for receiver, option in zip(scene['receivers'], scene['chosen']):
                width, length, x = receiver['width'], option['length'], receiver['x']
                units = []
                for k in range(option['final']):
                    bottom = base - k * option['pitch']
                    block = a[bottom - length:bottom, x:x + width]
                    _require(block.shape == (length, width) and bool(np.all(block != scene['bg'])),
                             "mosaic must be a gapless rectangle")
                    units.append(block)
                    if k < option['final'] - 1:
                        _require(bool(np.all(a[bottom - length - 1, x:x + width] == scene['bg'])),
                                 "receiver gap changed")
                _require(all(np.array_equal(units[0], u) for u in units), "mosaic copies differ")
                _require(option['final'] == receiver['count'] + option['total'], "wrong addition")
                for i, offset in option['placements']:
                    source = scene['sources'][i]
                    rows = base - 1 - (source['local'][:, 0] + offset)
                    cols = x + source['local'][:, 1]
                    _require(bool(np.all(a[rows, cols] == source['color'])), "fragment mask/color changed")
                cells = receiver['cells']
                _require(bool(np.all(a[cells[:, 0], cells[:, 1]] == receiver['color'])),
                         "existing foundations changed")
            if split == 'train':
                training.append(scene)
    _evidence(training)


def _features(scene):
    return dict(shape=scene['original'].shape, roles=(scene['bg'], scene['band']),
                direction=(scene['transposed'], scene['flipped']),
                receivers=len(scene['receivers']), sources=len(scene['sources']),
                unmarked=len(scene['unmarked']),
                widths=tuple(r['width'] for r in scene['receivers']),
                pieces=tuple(len(o['placements']) for o in scene['chosen']),
                lengths=tuple(o['length'] for o in scene['chosen']),
                source_counts=tuple(s['count'] for s in scene['sources']),
                destination_counts=tuple(r['count'] for r in scene['receivers']),
                masks=tuple((s['lo'], s['hi']) for s in scene['sources']))


def self_test():
    """Bounded structural diversity, arithmetic witnesses, API and mutation checks."""
    # Two identical foundations and two matching caps: colors and multiplicities
    # matter to uniqueness, whereas permutations of indistinguishable donors do not.
    fixture = np.zeros((16, 15), dtype=np.int8)
    fixture[14:] = 8
    root = (1, 2, 1)
    foundation = _layer((0, 0, 0), root)
    cap = _layer(root, (3, 3, 3)) - np.array([1, 0])
    for x, color in ((0, 6), (4, 7)):
        _put(fixture, 14, x, foundation, color)
    for x, color in ((8, 2), (12, 3)):
        _put(fixture, 14, x, cap, color, 3)
        fixture[12, x:x + 3] = 8
    try:
        solve(fixture)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous colored assignments accepted")
    fixture[fixture == 3] = 2
    solve(fixture)  # Same-colored, same-count caps have one actual output.
    _put(fixture, 14, 12, cap, 2, 6)
    try:
        solve(fixture)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous count assignments accepted")
    features, structures, signatures = [], set(), set()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed == 0:
            first = puzzle
        fingerprint = tuple(_key(p['input']) for p in puzzle['train'] + puzzle['test'])
        signatures.add(fingerprint)
        for pair in puzzle['train'] + puzzle['test']:
            expected, scene = _analyze(pair['input'].copy())
            assert np.array_equal(expected, pair['output'])
            if pair is puzzle['train'][0] or pair is puzzle['train'][1]:
                # Removing A leaves sparse copies instead of completed mosaics.
                without_a = _render(scene, [dict(o, placements=[]) for o in scene['chosen']])
                # Removing B completes the visible foundations without adding units.
                only_completion = [dict(o, final=r['count'])
                                   for r, o in zip(scene['receivers'], scene['chosen'])]
                without_b = _render(scene, only_completion)
                actual = expected.T if scene['transposed'] else expected
                actual = actual[::-1] if scene['flipped'] else actual
                assert not np.array_equal(without_a, actual)
                assert not np.array_equal(without_b, actual)
            features.append(_features(scene))
            roles, normalized = {}, []
            for value in pair['input'].flat:
                normalized.append(roles.setdefault(int(value), len(roles)))
            structures.add((pair['input'].shape, tuple(normalized)))
    assert len(signatures) == 42 and len(structures) >= 190
    assert len({f['shape'] for f in features}) >= 35
    assert len({f['roles'] for f in features}) >= 20
    assert len({f['direction'] for f in features}) == 4
    assert {f['receivers'] for f in features} == {1, 2}
    assert {n for f in features for n in f['pieces']} == {1, 2, 3}
    assert {n for f in features for n in f['lengths']} >= {3, 4, 5, 6}
    assert {n for f in features for n in f['source_counts']} >= {1, 2, 3}
    assert {n for f in features for n in f['destination_counts']} >= {1, 2, 3}
    assert len({m for f in features for m in f['masks']}) >= 50
    again = generate_puzzle(0)
    assert all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
               for p, q in zip(first[s], again[s]) for k in ('input', 'output'))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(100 + 10 * a + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Distinct defects target A's filled shape, B's count and processing, and preservation.
    for defect in ('fragment', 'sum', 'marker', 'highlight', 'unmarked', 'band'):
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        changed = False
        for pair in broken['train']:
            _, scene = _analyze(pair['input'])
            out = pair['output'].T if scene['transposed'] else pair['output']
            out = out[::-1] if scene['flipped'] else out
            r, o = scene['receivers'][0], scene['chosen'][0]
            if defect == 'fragment':
                i, offset = o['placements'][0]
                u, c = scene['sources'][i]['local'][0]
                out[scene['baseline'] - 1 - offset - u, r['x'] + c] = scene['bg']
            elif defect == 'sum':
                top = scene['baseline'] - (o['final'] - 1) * o['pitch'] - o['length']
                out[top:top + o['length'], r['x']:r['x'] + r['width']] = scene['bg']
            elif defect in ('marker', 'highlight'):
                s = scene['sources'][0]
                cells = s['marker'] if defect == 'marker' else s['cells']
                out[tuple(cells[0])] = scene['band'] if defect == 'marker' else s['color']
            elif defect == 'unmarked':
                if not scene['unmarked']:
                    continue
                out[tuple(scene['unmarked'][0]['cells'][0])] = scene['band']
            else:
                out[scene['baseline'], 0] = scene['bg']
            changed = True
            break
        assert changed
        try:
            validate_generated_puzzle(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted bad " + defect)
    return {'seeds': 42, 'pairs': len(features), 'distinct_structures': len(structures),
            'grid_shapes': len({f['shape'] for f in features}),
            'source_masks': len({m for f in features for m in f['masks']})}
