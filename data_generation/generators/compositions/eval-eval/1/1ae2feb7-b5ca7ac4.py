"""Resonant freight: stable framed-cart compaction onto periodic color berths.

Only NumPy and the Python standard library are used. solve reads one input;
all direction, phase, key, and program information is visible in that input.
"""
from __future__ import annotations

import copy
import json
import numpy as np


SELF_TEST_SEEDS = (0, 107331996, 18242286, 320508122, 322192330, 113352670,
                  351068766, 69572211, 495802839, 241878175, 53154671,
                  133282904, 710078616, 687398970, 665509253, 600971023,
                  357510744, 664489255, 504723729, 990870157, 743687788,
                  249641613, 31387553, 430283524, 827415590, 174183440,
                  713392171, 576616799, 390987872, 8013382, 622699059,
                  320222209, 582696790, 666412198, 499656500, 210822182,
                  245870758, 713795405, 177144728, 440694438, 191707506,
                  504519967)
THREE_PROGRAMS = ((4, 3, 2), (5, 3, 2), (3, 4, 2),
                  (2, 3, 1), (3, 2, 1), (4, 2, 3))
RIVALS = ('no_period', 'no_blockers', 'restart', 'overtake', 'short', 'long',
          'far', 'color_low', 'color_high', 'plus_one', 'minus_one',
          'strip_width', 'source_gap', 'positive_far')
REQUIRED_RIVALS = ('no_period', 'no_blockers', 'restart', 'overtake',
                   'short', 'long', 'far', 'color_low', 'color_high',
                   'positive_far')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(grid):
    a = np.asarray(grid)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             'grids must be 2D with dimensions 1..30')
    _require(np.issubdtype(a.dtype, np.integer) and
             np.all((0 <= a) & (a <= 9)), 'grid colors must be integers 0..9')
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _overlap_rows(a, b):
    return max(a['r'], b['r']) < min(a['r'] + a['h'], b['r'] + b['h'])


def _overlap(a, b):
    return (_overlap_rows(a, b) and
            max(a['c'], b['c']) < min(a['c'] + a['w'], b['c'] + b['w']))


def _nose(o, width):
    return o['c'] if o['d'] == 0 else width - o['c'] - o['w']


def _column(o, q, width):
    return q if o['d'] == 0 else width - q - o['w']


def _read_program(row, background, direction):
    values = row if direction == 0 else row[::-1]
    runs, i = [], 0
    while i < len(values):
        color = int(values[i])
        if color == background:
            i += 1
            continue
        start = i
        while i < len(values) and int(values[i]) == color:
            i += 1
        runs.append((color, i - start, start))
    return runs


def _decode(input_grid):
    grid = _array(input_grid)
    candidates = []
    for axis in (1, 0):
        a = grid if axis == 1 else grid.T
        if a.shape[0] < 7 or a.shape[1] < 6:
            continue
        low, high = int(a[0, 0]), int(a[0, -1])
        background = int(a[0, 1])
        if len({low, high, background}) != 3:
            continue
        if (np.all(a[:, 0] == low) and np.all(a[:, -1] == high)
                and np.all(a[0, 1:-1] == background)
                and np.all(a[-1, 1:-1] == background)):
            candidates.append((axis, a, background, (low, high)))
    _require(len(candidates) == 1, 'need one unambiguous pair of opposite rails')
    axis, canonical, background, rails = candidates[0]
    body = canonical[:, 1:-1]
    occupied = np.zeros(body.shape, dtype=bool)
    covered = body == background
    objects = []
    height, width = body.shape
    for r in range(height):
        for c in range(width):
            if covered[r, c]:
                continue
            frame = int(body[r, c])
            _require(frame in rails, 'unexplained non-frame mark')
            _require(r + 4 < height and c + 3 < width, 'incomplete cart')
            # Interior never uses a rail color. The next frame down this inner
            # column is the bottom border; the middle interior row finds width.
            bottom = r + 1
            while bottom < height and body[bottom, c + 1] != frame:
                bottom += 1
            right = c + 1
            while right < width and body[r + 2, right] != frame:
                right += 1
            _require(bottom < height and right < width, 'open frame')
            h, w = bottom - r + 1, right - c + 1
            _require(h >= 5 and w >= 4, 'cart is too small')
            patch = body[r:bottom + 1, c:right + 1].copy()
            _require(np.all(patch[0] == frame) and np.all(patch[-1] == frame)
                     and np.all(patch[:, 0] == frame)
                     and np.all(patch[:, -1] == frame), 'incomplete rectangular frame')
            interior = patch[1:-1, 1:-1]
            _require(not np.any(np.isin(interior, rails)), 'interior uses a rail color')
            _require(np.all(interior[1:-1] == background), 'intermediate rows must be blank')
            counts = [int(np.count_nonzero(row != background))
                      for row in (interior[0], interior[-1])]
            _require((counts[0] == 1 and counts[1] >= 2) or
                     (counts[1] == 1 and counts[0] >= 2), 'need singleton key and run strip')
            key_at_top = counts[0] == 1
            key_row = interior[0] if key_at_top else interior[-1]
            program_row = interior[-1] if key_at_top else interior[0]
            direction = rails.index(frame)
            key_col = 0 if direction == 0 else w - 3
            _require(key_row[key_col] != background, 'key must face the destination')
            key_color = int(key_row[key_col])
            runs = _read_program(program_row, background, direction)
            _require(key_color in {x[0] for x in runs}, 'key is absent from program')
            _require(r >= 1 and bottom < height - 1, 'cart touches a clear margin')
            _require(not np.any(occupied[r:bottom + 1, c:right + 1]), 'cart rectangles overlap')
            occupied[r:bottom + 1, c:right + 1] = True
            covered[r:bottom + 1, c:right + 1] = True
            objects.append(dict(r=r, c=c, h=h, w=w, frame=frame, d=direction,
                                patch=patch, key=key_color, runs=runs,
                                key_at_top=key_at_top))
    _require(objects and np.all(covered), 'empty scene or unexplained pixels')
    return dict(axis=axis, canonical=canonical, background=background,
                rails=rails, width=width, objects=objects)


def _winner(runs, q):
    for color, length, _ in runs:
        if q % length == 0:
            return color
    return None


def _settle(objects, width):
    result = [dict(o) for o in objects]
    for direction in (0, 1):
        indices = [i for i, o in enumerate(objects) if o['d'] == direction]
        indices.sort(key=lambda i: (_nose(objects[i], width), objects[i]['r']))
        predecessors = []
        for i in indices:
            original = objects[i]
            frontier = max((_nose(p, width) + p['w'] for p in predecessors
                            if _overlap_rows(original, p)), default=0)
            bound = min(_nose(original, width), width - original['w'])
            q = frontier
            while q <= bound and _winner(original['runs'], q) != original['key']:
                q += 1
            _require(q <= bound, 'no matching berth reachable toward the destination')
            result[i]['c'] = _column(original, q, width)
            predecessors.append(result[i])
    _require(not any(_overlap(a, b) for i, a in enumerate(result)
                     for b in result[i + 1:]), 'final opposite classes collide')
    return result


def _render(shape, background, rails, objects):
    grid = np.full(shape, background, dtype=np.int8)
    grid[:, 0], grid[:, -1] = rails
    for o in objects:
        grid[o['r']:o['r'] + o['h'], 1 + o['c']:1 + o['c'] + o['w']] = o['patch']
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique stable resonant arrangement from this input alone."""
    info = _decode(input_grid)
    settled = _settle(info['objects'], info['width'])
    out = _render(info['canonical'].shape, info['background'], info['rails'], settled)
    return out if info['axis'] == 1 else out.T.copy()


def _audit_expected(input_grid, mode='intended'):
    """Separate candidate-position enumeration and slice-overlay pulse check.

    Also implements explicit simpler rival rules for curriculum auditing.
    A rival may produce collisions; its painted answer is still compared.
    Infeasible rivals return None, rather than borrowing the intended answer.
    """
    info = _decode(input_grid)
    original, width = info['objects'], info['width']
    result = [dict(o) for o in original]
    for direction in (0, 1):
        indices = sorted((i for i, o in enumerate(original) if o['d'] == direction),
                         key=lambda i: (_nose(original[i], width), original[i]['r']))
        placed = []
        for i in indices:
            o = original[i]
            before = [p for p in placed if
                      o['r'] < p['r'] + p['h'] and p['r'] < o['r'] + o['h']]
            frontier = max((_nose(p, width) + p['w'] for p in before), default=0)
            if mode in ('no_blockers', 'overtake'):
                frontier = 0
            limit = min(_nose(o, width), width - o['w'])
            # Values are filled by superimposing full periodic streams. This
            # does not call solve's first-divisible-run winner or _settle.
            pulse = np.full(width, -1, dtype=int)
            runs = list(o['runs'])
            if mode == 'short':
                runs.sort(key=lambda x: (x[1], x[2]))
            elif mode == 'long':
                runs.sort(key=lambda x: (-x[1], x[2]))
            elif mode in ('far', 'positive_far'):
                runs.reverse()
            elif mode in ('color_low', 'color_high'):
                runs.sort(key=lambda x: x[0], reverse=mode == 'color_high')
            for color, period, gap_start in reversed(runs):
                if mode == 'plus_one':
                    period += 1
                elif mode == 'minus_one':
                    period = max(1, period - 1)
                elif mode == 'strip_width':
                    period = o['w'] - 2
                pulse[::period] = color
            if mode == 'positive_far':
                pulse[0] = o['runs'][0][0]
            positions = []
            for q in range(frontier, limit + 1):
                phase = q - frontier if mode == 'restart' else q
                if mode == 'source_gap':
                    phase = q - o['runs'][0][2]
                    if phase < 0:
                        continue
                accepts = mode == 'no_period' or pulse[phase] == o['key']
                if not accepts:
                    continue
                if mode == 'overtake':
                    # Any non-overlapping hole, including one before a prior
                    # overlapping cart: the old edge-first rival to stable order.
                    if any(not (q + o['w'] <= _nose(p, width)
                                or _nose(p, width) + p['w'] <= q) for p in before):
                        continue
                positions.append(q)
            if not positions:
                return None
            result[i]['c'] = _column(o, positions[0], width)
            placed.append(result[i])
    if mode == 'intended' and any(_overlap(a, b) for i, a in enumerate(result)
                                  for b in result[i + 1:]):
        return None
    out = _render(info['canonical'].shape, info['background'], info['rails'], result)
    return out if info['axis'] == 1 else out.T.copy()


def _matches(pair, mode):
    guess = _audit_expected(pair['input'], mode)
    return guess is not None and np.array_equal(guess, pair['output'])


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                    ('num_test', num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_))
                 and isinstance(value, (int, np.integer)) and low <= value <= high,
                 name + ' is outside its legal integer range')
    _require(num_train > num_test, 'num_train must exceed num_test')
    return int(num_train), int(num_test)


def _cart(rng, r, q, h, lengths, colors, key, direction, rails, background,
          width, gap_budget=None, force_leading_gap=False):
    lengths = tuple(int(x) for x in lengths)
    colors = tuple(int(x) for x in colors)
    if gap_budget is None:
        gap_budget = int(rng.integers(0, 4))
    # Compartments: nose padding, inter-run gaps, tail padding.
    gaps = rng.multinomial(gap_budget, np.full(len(lengths) + 1, 1 / (len(lengths) + 1)))
    if force_leading_gap:
        gaps[0] += 1
    for i in range(1, len(colors)):
        if colors[i] == colors[i - 1] and gaps[i] == 0:
            gaps[i] = 1
    program = [background] * int(gaps[0])
    for i, (length, color) in enumerate(zip(lengths, colors)):
        program.extend([color] * length)
        program.extend([background] * int(gaps[i + 1]))
    _require(len(program) >= 2 and key in colors, 'invalid constructed program')
    w = len(program) + 2
    patch = np.full((h, w), rails[direction], dtype=np.int8)
    patch[1:-1, 1:-1] = background
    key_at_top = bool(rng.integers(2))
    key_row, program_row = (1, h - 2) if key_at_top else (h - 2, 1)
    patch[program_row, 1:-1] = program if direction == 0 else program[::-1]
    patch[key_row, 1 if direction == 0 else w - 2] = key
    runs = _read_program(patch[program_row, 1:-1], background, direction)
    return dict(r=r, c=q if direction == 0 else width - q - w, h=h, w=w,
                frame=rails[direction], d=direction, patch=patch,
                key=key, runs=runs, key_at_top=key_at_top)


def _first_berth(o, bound, width):
    for q in range(bound, width - o['w'] + 1):
        if _winner(o['runs'], q) == o['key']:
            return q
    raise ValueError('constructed key has no berth')


def _pair_from_objects(shape, background, rails, objects, axis):
    _require(not any(_overlap(a, b) for i, a in enumerate(objects)
                     for b in objects[i + 1:]), 'constructed input collision')
    output_objects = _settle(objects, shape[1] - 2)
    inp = _render(shape, background, rails, objects)
    out = _render(shape, background, rails, output_objects)
    if axis == 0:
        inp, out = inp.T.copy(), out.T.copy()
    return {'input': inp, 'output': out}


def _priority_scene(rng, p, background):
    width = 28
    x, y = p['palette'][:2]
    first_direction = int(rng.integers(2))
    prototypes = [((2, 3), (x, y), y, first_direction),
                  ((2, 3), (x, y), y, 1 - first_direction),
                  ((3, 2), (y, x), x, int(rng.integers(2)))]
    objects = [_cart(rng, 0, 0, int(rng.integers(5, 8)), ls, cs, key, d,
                     p['rails'], background, width,
                     force_leading_gap=i == 0)
               for i, (ls, cs, key, d) in enumerate(prototypes)]
    rng.shuffle(objects)
    minimum = max(_first_berth(o, 0, width) + o['w'] for o in objects)
    width = int(rng.integers(max(12, minimum + 2), 29))
    required_height = sum(o['h'] for o in objects) + 2
    height = int(rng.integers(required_height, 31))
    gaps = rng.multinomial(height - sum(o['h'] for o in objects) - 2,
                          np.full(4, .25))
    r = 1 + int(gaps[0])
    for i, o in enumerate(objects):
        o['r'] = r
        first = _first_berth(o, 0, width)
        q = int(rng.integers(first, width - o['w'] + 1))
        o['c'] = _column(o, q, width)
        r += o['h'] + int(gaps[i + 1])
    return _pair_from_objects((height, width + 2), background, p['rails'], objects, p['axis'])


def _chain_scene(rng, p, background):
    width = 28
    direction = int(rng.integers(2))
    h = int(rng.integers(5, 7))
    step = int(rng.integers((h + 1) // 2, h))
    x, y, z = p['palette'][:3]
    variant = int(rng.choice(4, p=[.35, .25, .20, .20]))
    if variant == 0:
        # Third frontier is 17 or 18: the key's q=18 pulse loses to period 3,
        # so the cart must continue to q=20. Farthest priority would stop at 18.
        padding = int(rng.integers(2))
        specs = [((3, 2), (x, y), y, padding)] * 3
    elif variant == 1:
        # Two period-5 leaders leave a frontier of 17 or 18. The nearest
        # period-2 run wins the nonzero collision at 18.
        specs = [((5,), (x,), x, int(rng.integers(2))) for _ in range(2)]
        specs.append(((2, 3), (x, y), x, int(rng.integers(2))))
    elif variant == 2:
        # Width-8 period-4 leaders settle at 0 and 8; the next frontier is 16.
        # Period 3 wins the collision at 18, independent of footprint width.
        specs = [((4,), (x,), x, 2), ((4,), (x,), x, 2),
                 ((3, 2), (y, x), y, int(rng.integers(2)))]
    else:
        # Compact mixed-width chain: 0,6,10 (or 12). The last code's near
        # period-1 run wins a nonzero collision, while the second shows rounding.
        specs = [((3,), (x,), x, 0), ((2,), (y,), y, int(rng.choice([0, 2]))),
                 ((1, 2), (x, y), x, int(rng.integers(2)))]
    prototypes = [_cart(rng, 1, 0, h, ls, cs, key, direction, p['rails'],
                        background, width, gap_budget=padding)
                  for ls, cs, key, padding in specs]
    berth = []
    frontier = 0
    for o in prototypes:
        berth.append(_first_berth(o, frontier, width))
        frontier = berth[-1] + o['w']
    _require(any(berth[i + 1] > berth[i] + prototypes[i]['w'] for i in (0, 1)),
             'chain needs a skipped frontier')
    minimum_width = berth[-1] + prototypes[-1]['w']
    chain_span = h + 2 * step
    extra_h = int(rng.integers(5, 8))
    extra_direction = int(rng.integers(2))
    ls = THREE_PROGRAMS[int(rng.integers(len(THREE_PROGRAMS)))]
    extra = _cart(rng, chain_span + 2, 0, extra_h, ls, (x, y, z), z,
                  extra_direction, p['rails'], background, width,
                  gap_budget=int(rng.integers(0, 3)))
    extra_berth = _first_berth(extra, 0, width)
    minimum_width = max(minimum_width, extra_berth + extra['w'])
    width = int(rng.integers(minimum_width, 29))
    band_gap = int(rng.integers(0, 3))
    minimum_height = chain_span + extra_h + band_gap + 2
    height = int(rng.integers(minimum_height, 31))
    margin = int(rng.integers(height - minimum_height + 1))
    start = 1 + margin
    reverse_rows = bool(rng.integers(2))
    chain = []
    source = [0, 0, int(rng.integers(berth[2], width - prototypes[2]['w'] + 1))]
    for i in (1, 0):
        source[i] = int(rng.integers(berth[i], source[i + 1] - prototypes[i]['w'] + 1))
    for i in range(3):
        o = dict(prototypes[i])
        o['r'] = start + (2 - i if reverse_rows else i) * step
        o['c'] = _column(o, source[i], width)
        o['patch'] = prototypes[i]['patch'].copy()
        if rng.random() < .5:
            o['patch'] = o['patch'][::-1].copy()
            o['key_at_top'] = not o['key_at_top']
        chain.append(o)
    extra['r'] = start + chain_span + band_gap
    extra['c'] = _column(extra, int(rng.integers(extra_berth, width - extra['w'] + 1)), width)
    objects = chain + [extra]
    return _pair_from_objects((height, width + 2), background, p['rails'], objects, p['axis'])


def _random_cart(rng, p, background, width, height, force_three=False):
    count = 3 if force_three else int(rng.choice([1, 2, 3], p=[.35, .45, .20]))
    if count == 1:
        lengths = (int(rng.integers(2, 6)),)
    else:
        lengths = tuple(int(x) for x in rng.integers(1, 6, size=count))
        if sum(lengths) > 10:
            lengths = THREE_PROGRAMS[int(rng.integers(len(THREE_PROGRAMS)))] if count == 3 else (2, 3)
    colors = [int(rng.choice(p['palette'])) for _ in lengths]
    # Most mixtures use distinct colors; separated repeated colors also occur.
    if count > 1 and rng.random() < .8:
        colors = list(map(int, rng.choice(p['palette'], count, replace=False)))
    h = int(rng.integers(5, min(8, height - 2) + 1))
    o = _cart(rng, 1, 0, h, lengths, colors, colors[0], int(rng.integers(2)),
              p['rails'], background, width, gap_budget=int(rng.integers(0, 4)))
    _require(o['w'] <= min(width, 14), 'sampled cart is too wide')
    reachable = sorted({c for q in range(width - o['w'] + 1)
                        if (c := _winner(o['runs'], q)) is not None})
    o['key'] = int(rng.choice(reachable))
    key_row = 1 if o['key_at_top'] else h - 2
    o['patch'][key_row, 1 if o['d'] == 0 else o['w'] - 2] = o['key']
    o['r'] = int(rng.integers(1, height - h))
    q = int(rng.integers(0, width - o['w'] + 1))
    o['c'] = _column(o, q, width)
    return o


def _add_random(rng, p, background, shape, objects, target):
    height, width = shape[0], shape[1] - 2
    for _ in range(target - len(objects)):
        for _ in range(45):
            try:
                o = _random_cart(rng, p, background, width, height)
                if any(_overlap(o, a) for a in objects):
                    continue
                _settle(objects + [o], width)
            except ValueError:
                continue
            objects.append(o)
            break
        else:
            break
    return objects


def _free_scene(rng, p, background):
    width, height = int(rng.integers(12, 29)), int(rng.integers(9, 31))
    objects = []
    target = int(rng.integers(1, 9))
    _add_random(rng, p, background, (height, width + 2), objects, target)
    if not objects:
        o = _cart(rng, 1, 1, 5, (2,), (p['palette'][0],), p['palette'][0],
                  0, p['rails'], background, width, gap_budget=0)
        objects = [o]
    return _pair_from_objects((height, width + 2), background, p['rails'], objects, p['axis'])


def _test_scene(rng, p, background):
    # Transfer the three-run syntax from an independent training band into a
    # blocking interaction, always putting a three-run cart second in its class.
    for _ in range(100):
        width = 28
        direction = int(rng.integers(2))
        x, y, z = p['palette'][:3]
        h1, h2 = int(rng.integers(5, 9)), int(rng.integers(5, 9))
        first = _cart(rng, 1, 0, h1, (int(rng.integers(2, 5)),), (x,), x,
                      direction, p['rails'], background, width, gap_budget=int(rng.integers(2)))
        ls = THREE_PROGRAMS[int(rng.integers(len(THREE_PROGRAMS)))]
        second = _cart(rng, 1, 0, h2, ls, (x, y, z), z, direction,
                       p['rails'], background, width, gap_budget=int(rng.integers(3)))
        try:
            q1 = _first_berth(first, 0, width)
            q2 = _first_berth(second, q1 + first['w'], width)
        except ValueError:
            continue
        width = int(rng.integers(max(16, q2 + second['w']), 29))
        offset = int(rng.integers(-(h2 - 1), h1))
        base = 1 + max(0, -offset)
        span = max(h1, offset + h2) - min(0, offset)
        height = int(rng.integers(max(9, span + 2), 31))
        base += int(rng.integers(height - span - 1))
        first['r'], second['r'] = base, base + offset
        source2 = int(rng.integers(q2, width - second['w'] + 1))
        source1 = int(rng.integers(q1, source2 - first['w'] + 1))
        first['c'], second['c'] = _column(first, source1, width), _column(second, source2, width)
        objects = [first, second]
        _add_random(rng, p, background, (height, width + 2), objects, int(rng.integers(2, 9)))
        return _pair_from_objects((height, width + 2), background, p['rails'], objects, p['axis'])
    raise RuntimeError('bounded three-run interaction construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and
             isinstance(seed, (int, np.integer)), 'seed must be an integer')
    rng = np.random.default_rng(seed)
    palette_order = list(map(int, rng.permutation(10)))
    base_background = palette_order[0]
    rails = tuple(palette_order[1:3])
    payloads = palette_order[3:3 + int(rng.integers(3, 6))]
    backgrounds = [c for c in range(10) if c not in (*rails, *payloads)]
    p = dict(rails=rails, palette=payloads, axis=int(rng.integers(2)))
    varying_background = bool(rng.random() < .7)
    priority_slot, chain_slot = map(int, rng.choice(num_train, 2, replace=False))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            purpose = ('priority' if split == 'train' and i == priority_slot else
                       'chain' if split == 'train' and i == chain_slot else
                       'free' if split == 'train' else 'test')
            for _ in range(60):
                background = int(rng.choice(backgrounds)) if varying_background else base_background
                factory = {'priority': _priority_scene, 'chain': _chain_scene,
                           'free': _free_scene, 'test': _test_scene}[purpose]
                pair = factory(rng, p, background)
                ik = _key(pair['input'])
                if ik not in seen:
                    seen.add(ik)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded unique-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_three_run_overlap(info):
    objects = info['objects']
    for o in objects:
        if len(o['runs']) >= 3:
            for p in objects:
                if p is not o and p['d'] == o['d'] and _overlap_rows(p, o):
                    return True
    return False


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, conventions = set(), set()
    three_taught = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _array(pair['input']), _array(pair['output'])
            _require(inp.shape == out.shape, 'dimensions changed')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _decode(inp)
            conventions.add((info['axis'], info['rails']))
            settled = _settle(info['objects'], info['width'])
            expected = _audit_expected(inp)
            _require(expected is not None and np.array_equal(expected, out),
                     'wrong first berth, stable order, footprint, phase, patch, or rails')
            # Exact complete-patch copying plus distinct footprints, not only
            # the number or color of foreground cells.
            actual = out if info['axis'] == 1 else out.T
            for old, new in zip(info['objects'], settled):
                patch = actual[new['r']:new['r'] + new['h'],
                               new['c'] + 1:new['c'] + 1 + new['w']]
                _require(np.array_equal(old['patch'], patch), 'cart patch was changed')
                _require(_nose(new, info['width']) <= _nose(old, info['width']), 'cart moved inward')
            if split == 'train':
                three_taught |= any(len(o['runs']) >= 3 for o in info['objects'])
            else:
                _require(_has_three_run_overlap(info), 'missing test interaction shift')
    _require(len(conventions) == 1, 'axis and rail identities must be consistent')
    _require(three_taught, 'training must expose three-run syntax')
    for rival in RIVALS:
        fits_train = all(_matches(pair, rival) for pair in puzzle['train'])
        if rival in REQUIRED_RIVALS:
            _require(not fits_train, 'missing teaching witness against ' + rival)
        _require(not fits_train or all(_matches(pair, rival) for pair in puzzle['test']),
                 'training fits a rival that changes a test: ' + rival)


def measure_puzzle(puzzle):
    """Visible-grid family statistics, suitable for compact scratch reports."""
    records = []
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            info = _decode(pair['input'])
            objects, width = info['objects'], info['width']
            settled = _settle(objects, width)
            gaps, edges = [], 0
            for d in (0, 1):
                indices = sorted((i for i, o in enumerate(objects) if o['d'] == d),
                                 key=lambda i: (_nose(objects[i], width), objects[i]['r']))
                before = []
                for i in indices:
                    frontier = max((_nose(p, width) + p['w'] for p in before
                                    if _overlap_rows(objects[i], p)), default=0)
                    gaps.append(_nose(settled[i], width) - frontier)
                    before.append(settled[i])
            edges = sum(a['d'] == b['d'] and _overlap_rows(a, b)
                        for i, a in enumerate(objects) for b in objects[i + 1:])
            records.append(dict(split=split, index=index, shape=list(np.asarray(pair['input']).shape),
                                axis=info['axis'], rails=list(info['rails']), background=info['background'],
                                count=len(objects), classes=[sum(o['d'] == d for o in objects) for d in (0, 1)],
                                periods=sorted({L for o in objects for _, L, _ in o['runs']}),
                                run_counts=sorted({len(o['runs']) for o in objects}),
                                footprints=sorted({(o['h'], o['w']) for o in objects}),
                                overlap_edges=int(edges), skipped_frontiers=gaps,
                                stationary=sum(_nose(a, width) == _nose(b, width)
                                               for a, b in zip(objects, settled)),
                                three_run_overlap=_has_three_run_overlap(info),
                                overtaking_witness=not _matches(pair, 'overtake'),
                                source_gaps=sorted({o['runs'][0][2] for o in objects}),
                                repeated_colors=any(len({c for c, _, _ in o['runs']}) < len(o['runs'])
                                                    for o in objects)))
    return records


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                 for s in ('train', 'test') for p in puzzle[s])


def _must_fail(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('invalid data was accepted')


def self_test():
    """Bounded deterministic regression, ablation, mutation and diversity checks."""
    rng = np.random.default_rng(7788)
    rails, background = (1, 2), 0
    # Hand-computed two-period first berths, in both directions.
    a = _cart(rng, 1, 8, 5, (2, 3), (6, 7), 7, 0, rails, background, 24, gap_budget=0)
    b = _cart(rng, 8, 8, 6, (3, 2), (7, 6), 6, 1, rails, background, 24, gap_budget=0)
    inp = _render((16, 26), background, rails, [a, b])
    expected = _render((16, 26), background, rails, [dict(a, c=3), dict(b, c=24 - b['w'] - 2)])
    assert np.array_equal(solve(inp), expected)
    # A period-3 chain has width 5, so its berths are 0,6,12, not 0,5,10
    # (tight packing) or 0,6,0 (same-class overtaking).
    chain = [_cart(rng, r, q, 5, (3,), (6,), 6, 0, rails, background, 22, gap_budget=0)
             for r, q in ((1, 2), (4, 9), (7, 16))]
    chain_input = _render((14, 24), background, rails, chain)
    chain_expected = _render((14, 24), background, rails,
                             [dict(o, c=q) for o, q in zip(chain, (0, 6, 12))])
    assert np.array_equal(solve(chain_input), chain_expected)
    assert not np.array_equal(_audit_expected(chain_input, 'overtake'), chain_expected)
    assert not np.array_equal(_audit_expected(chain_input, 'restart'), chain_expected)
    # Three sources behind a blocker: q=6 and q=8 are occluded ticks of the
    # key-8 period-2 run; q=10 is the first permitted tick after frontier 5.
    blocker = _cart(rng, 1, 2, 5, (3,), (6,), 6, 0, rails, background, 28, gap_budget=0)
    follower = _cart(rng, 3, 15, 7, (4, 3, 2), (6, 7, 8), 8, 0,
                     rails, background, 28, gap_budget=0)
    transfer = _render((13, 30), background, rails, [blocker, follower])
    transfer_expected = _render((13, 30), background, rails,
                                [dict(blocker, c=0), dict(follower, c=10)])
    assert np.array_equal(solve(transfer), transfer_expected)
    # Grammar extensions beyond the sampler: higher periods, four runs,
    # taller frames, and repeated colors separated into distinct runs.
    extensions = [
        ((4, 6), (6, 7), 7, 6, 9),
        ((5, 4, 3, 2), (5, 6, 7, 8), 8, 2, 9),
        ((2, 3, 2), (6, 7, 6), 7, 3, 10),
        ((13,), (6,), 6, 0, 9),
    ]
    for lengths, colors, key, berth, h in extensions:
        o = _cart(rng, 1, 10, h, lengths, colors, key, 0,
                  rails, background, 28, gap_budget=0)
        before = _render((h + 3, 30), 0, rails, [o])
        after = _render((h + 3, 30), 0, rails, [dict(o, c=berth)])
        assert np.array_equal(solve(before), after)
        assert np.array_equal(_audit_expected(before), after)
    for before, after in ((inp, expected), (chain_input, chain_expected),
                          (transfer, transfer_expected)):
        assert np.array_equal(_audit_expected(before), after)
        for transform in (lambda g: g.T.copy(), lambda g: g[::-1].copy(),
                          lambda g: g[:, ::-1].copy()):
            assert np.array_equal(solve(transform(before)), transform(after))
        permutation = np.array([9, 4, 0, 1, 2, 3, 5, 8, 7, 6], dtype=np.int8)
        assert np.array_equal(solve(permutation[before]), permutation[after])
    corpus = [generate_puzzle(seed) for seed in SELF_TEST_SEEDS]
    assert len({_fingerprint(p) for p in corpus}) == 42
    assert _fingerprint(corpus[0]) == _fingerprint(generate_puzzle(SELF_TEST_SEEDS[0]))
    records = [row for p in corpus for row in measure_puzzle(p)]
    assert {r['axis'] for r in records} == {0, 1}
    assert len({tuple(r['rails']) for r in records}) >= 20
    assert len({tuple(r['shape']) for r in records}) >= 70
    assert len({tuple(f) for r in records for f in r['footprints']}) >= 20
    assert {1, 2, 3} <= {n for r in records for n in r['run_counts']}
    assert {1, 2, 3, 4, 5} <= {n for r in records for n in r['periods']}
    assert len({r['count'] for r in records}) >= 5
    assert {False, True} == {bool(r['overlap_edges']) for r in records}
    assert any(r['repeated_colors'] for r in records)
    assert any(max(r['source_gaps']) >= 2 for r in records)
    assert any(r['stationary'] for r in records)
    for p in corpus:
        rows = measure_puzzle(p)
        assert len({tuple(r['shape']) for r in rows}) > 1
        assert any(r['overtaking_witness'] for r in rows if r['split'] == 'train')
        for split in ('train', 'test'):
            for pair in p[split]:
                assert np.array_equal(solve(pair['input']), pair['output'])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(617, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (None, 1), (4, None)):
        _must_fail(lambda counts=counts: generate_puzzle(0, *counts))
    for seed in SELF_TEST_SEEDS:
        p = generate_puzzle(seed, 2, 1)
        if seed in SELF_TEST_SEEDS[::7]:
            assert _fingerprint(p) == _fingerprint(generate_puzzle(seed, 2, 1))
    broken = copy.deepcopy(corpus[0])
    output = broken['test'][0]['output']
    output[0, 0] = (int(output[0, 0]) + 1) % 10
    _must_fail(lambda: validate_generated_puzzle(broken))
    # A permanently masked far key is not silently docked somewhere arbitrary.
    hidden = _cart(rng, 1, 10, 5, (1, 2), (6, 7), 7, 0,
                   rails, background, 20, gap_budget=0)
    _must_fail(lambda: solve(_render((9, 22), 0, rails, [hidden])))
    crossing = [_cart(rng, 1, 9, 5, ls, cs, key, d, rails, 0, 16, gap_budget=0)
                for ls, cs, key, d in (((2, 3), (6, 7), 7, 0),
                                        ((3, 2), (7, 6), 6, 1))]
    _must_fail(lambda: solve(_render((9, 18), 0, rails, crossing)))


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': len(SELF_TEST_SEEDS)}, separators=(',', ':')))
