"""Programmable baffles: stable packing constructs a self-modifying paint route.

The input contains every convention: two matching edge rails, an initial word,
a separate collision swatch, cartridge words, and visible probe starts. No
generation metadata is used by solve. Only standard library and NumPy are used.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4
            and num_train > num_test):
        raise ValueError("Require integer counts: 2..10 train, 1..4 test, train > test")
    return int(num_train), int(num_test)


def _grid(value):
    g = np.asarray(value)
    if not (g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
            and np.issubdtype(g.dtype, np.integer)
            and np.all((0 <= g) & (g <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _rows_overlap(a, b):
    return max(a['r'], b['r']) < min(a['r'] + a['h'], b['r'] + b['h'])


def _overlap(a, b):
    return (_rows_overlap(a, b)
            and max(a['c'], b['c']) < min(a['c'] + a['w'], b['c'] + b['w']))


def _pack(objects, width):
    """Destination-side recurrence, including transitive stable chains."""
    result = [dict(o) for o in objects]
    for direction in (0, 1):
        order = [i for i, o in enumerate(objects) if o['d'] == direction]
        order.sort(key=lambda i: ((objects[i]['c'] if direction == 0 else
                                  -objects[i]['c'] - objects[i]['w']), objects[i]['r']))
        settled = []
        for i in order:
            o = result[i]
            blockers = [b for b in settled if _rows_overlap(o, b)]
            o['c'] = (max((b['c'] + b['w'] for b in blockers), default=1)
                      if direction == 0 else
                      min((b['c'] for b in blockers), default=width - 1) - o['w'])
            if not 1 <= o['c'] <= width - 1 - o['w']:
                raise ValueError("A stable chain exceeds the arena")
            settled.append(o)
    if any(_overlap(a, b) for i, a in enumerate(result) for b in result[i + 1:]):
        raise ValueError("Opposite packed classes collide")
    return result


def _physical_pack(objects, width):
    """Validator's independent one-cell sliding simulation."""
    result = [dict(o) for o in objects]
    for direction, delta in ((0, -1), (1, 1)):
        group = [o for o in result if o['d'] == direction]
        for _ in range(width * max(1, len(group)) + 1):
            moved = False
            for o in group:
                trial = dict(o, c=o['c'] + delta)
                if (1 <= trial['c'] <= width - 1 - o['w']
                        and all(b is o or not _overlap(trial, b) for b in group)):
                    o['c'] = trial['c']
                    moved = True
            if not moved:
                break
        else:
            raise AssertionError("Sliding failed to converge")
    assert not any(_overlap(a, b) for i, a in enumerate(result) for b in result[i + 1:])
    return result


def _edge_first_pack(objects, width):
    """Deliberately wrong alternative: teleport into the first final free slot."""
    result = [dict(o) for o in objects]
    for direction in (0, 1):
        indices = [i for i, o in enumerate(objects) if o['d'] == direction]
        indices.sort(key=lambda i: objects[i]['c'] if direction == 0 else
                     -objects[i]['c'] - objects[i]['w'])
        placed = []
        for i in indices:
            o = result[i]
            positions = range(1, width - o['w'])
            if direction == 1:
                positions = reversed(positions)
            for c in positions:
                o['c'] = c
                if all(not _overlap(o, b) for b in placed):
                    placed.append(o)
                    break
            else:
                raise ValueError("No repacking slot")
    if any(_overlap(a, b) for i, a in enumerate(result) for b in result[i + 1:]):
        raise ValueError("Alternative packed groups collide")
    return result


def _arena(shape, bg, frames, objects):
    grid = np.full(shape, bg, dtype=np.int8)
    grid[:, 0], grid[:, -1] = frames
    owner = np.full(shape, -1, dtype=int)
    for i, o in enumerate(objects):
        r, c, h, w = (o[k] for k in ('r', 'c', 'h', 'w'))
        grid[r:r+h, c:c+w] = o['patch']
        owner[r:r+h, c:c+w] = i
    return grid, owner


def _decode_canonical(g):
    h, w = g.shape
    if h < 8 or w < 12:
        raise ValueError("Too small for header and arena")
    bg = int(g[2, 0])
    if not np.all(g[2] == bg):
        raise ValueError("Missing header separator")
    frames = (int(g[3, 0]), int(g[3, -1]))
    if len({bg, *frames}) != 3 or not all(np.all(g[3:, c] == color)
                                         for c, color in ((0, frames[0]), (-1, frames[1]))):
        raise ValueError("Missing two uniform edge rails")
    # Header symbols are exact 2x2 blocks with 1..2-cell legend gaps;
    # the final block has a >=3-cell gap and supplies the collision color.
    swatches = []
    c = 0
    while c < w:
        if g[0, c] == bg:
            if g[1, c] != bg:
                raise ValueError("Malformed header")
            c += 1
            continue
        color = int(g[0, c])
        if c + 1 >= w or not np.all(g[:2, c:c+2] == color):
            raise ValueError("Malformed swatch")
        swatches.append((c, color))
        c += 2
    if not 3 <= len(swatches) <= 6:
        raise ValueError("Expected 2..5 initial entries plus marker")
    if swatches[0][0] not in (1, 2) or swatches[-1][0] != w - 3:
        raise ValueError("Header end margins identify reading direction")
    if any(b[0] - a[0] not in (3, 4) for a, b in zip(swatches[:-2], swatches[1:-1])):
        raise ValueError("Bad legend spacing")
    if swatches[-1][0] - swatches[-2][0] < 5:
        raise ValueError("Marker must be separate from legend")
    word = tuple(color for _, color in swatches[:-1])
    marker = swatches[-1][1]
    if marker in (*word, bg, *frames) or set(word) & {bg, *frames}:
        raise ValueError("Conflicting header color roles")
    arena = g[3:].copy()
    covered = arena == bg
    covered[:, 0] = covered[:, -1] = True
    objects = []
    ah = arena.shape[0]
    for r in range(ah):
        for c in range(1, w - 1):
            if covered[r, c] or int(arena[r, c]) not in frames:
                continue
            frame = int(arena[r, c])
            if r + 2 >= ah or c + 2 >= w - 1:
                raise ValueError("Incomplete cartridge")
            bottom, right = r + 2, c + 2
            while bottom < ah and arena[bottom, c + 1] != frame:
                bottom += 1
            while right < w - 1 and arena[r + 1, right] != frame:
                right += 1
            if bottom >= ah or right >= w - 1:
                raise ValueError("Open cartridge frame")
            patch = arena[r:bottom+1, c:right+1]
            if not (np.all(patch[0] == frame) and np.all(patch[-1] == frame)
                    and np.all(patch[:, 0] == frame) and np.all(patch[:, -1] == frame)):
                raise ValueError("Nonrectangular frame")
            interior = patch[1:-1, 1:-1]
            if (not 1 <= interior.shape[1] <= 5 or not 1 <= interior.shape[0] <= 4
                    or np.any(np.isin(interior, [bg, *frames, marker]))
                    or not np.all(interior == interior[0])):
                raise ValueError("Payload must repeat an ordered 1..5-entry row")
            if np.any(covered[r:bottom+1, c:right+1]):
                raise ValueError("Overlapping cartridges")
            covered[r:bottom+1, c:right+1] = True
            objects.append(dict(r=r, c=c, h=bottom-r+1, w=right-c+1,
                                d=frames.index(frame), frame=frame,
                                word=tuple(map(int, interior[0])), patch=patch.copy()))
    starts = tuple(map(tuple, np.argwhere(~covered)))
    if not objects or not 1 <= len(starts) <= 3:
        raise ValueError("Expected cartridges and 1..3 starts")
    if any(c != 1 or int(arena[r, c]) != word[0] for r, c in starts):
        raise ValueError("Starts are first-word singletons beside the entry rail")
    if any(abs(a[0] - b[0]) < 2 for i, a in enumerate(starts) for b in starts[i+1:]):
        raise ValueError("Starts must be isolated")
    return dict(grid=g, arena=arena, bg=bg, frames=frames, word=word, marker=marker,
                objects=objects, starts=starts, swatches=swatches)


def _decode(input_grid):
    g = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(g, rotation).copy()
        try:
            info = _decode_canonical(canonical)
        except ValueError:
            continue
        info['rotation'] = rotation
        candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("Input must have one rail/header interpretation")
    return candidates[0]


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _trace(mask, owner, objects, start, initial_word, *, variant=None, offset=0):
    """Maximal-leg implementation; frame contact loads a word for future cells."""
    if not mask[start]:
        raise ValueError("A settled cartridge covers a start")
    p, direction, phase, word = start, 0, offset, initial_word
    path, paint, collisions, legs, loaded = [], [], [], [], []
    for _ in range(sum(mask.shape) + 1):
        dr, dc = ((0, 1), (-1, 0))[direction]
        leg = []
        while _inside(mask.shape, p) and mask[p]:
            leg.append(p)
            path.append(p)
            active = tuple(dict.fromkeys(word)) if variant == 'deduplicate' else word
            paint.append(active[phase % len(active)])
            phase += 1
            p = (p[0] + dr, p[1] + dc)
        assert leg
        legs.append(len(leg))
        if not _inside(mask.shape, p):
            return dict(path=tuple(path), paint=tuple(paint), collisions=tuple(collisions),
                        legs=tuple(legs), loaded=tuple(loaded), end='edge')
        collisions.append(p)
        who = int(owner[p])
        if who >= 0 and variant != 'ignore_payload':
            if variant == 'wrapped_clock':
                phase %= len(word)
            word = objects[who]['word']
            if variant == 'reverse_payload':
                word = tuple(reversed(word))
            if variant == 'reset_load':
                phase = 0
            loaded.append(who)
        if variant == 'consume_marker':
            phase += 1
        if variant == 'reset_bend':
            phase = 0
        direction = 1 - direction
        dr, dc = ((0, 1), (-1, 0))[direction]
        p = (leg[-1][0] + dr, leg[-1][1] + dc)
        if not _inside(mask.shape, p) or not mask[p]:
            return dict(path=tuple(path), paint=tuple(paint), collisions=tuple(collisions),
                        legs=tuple(legs), loaded=tuple(loaded), end='blocked')
    raise AssertionError("Monotone route did not terminate")


def _step_trace(mask, owner, objects, start, initial_word):
    """Validator's separate cell walker, maintaining one global paint clock."""
    p, up, phase, word = start, False, 0, initial_word
    path, paint, collisions, lengths, loaded = [], [], [], [0], []
    assert mask[p]
    while True:
        path.append(p)
        paint.append(word[phase % len(word)])
        phase += 1
        lengths[-1] += 1
        ahead = (p[0] - 1, p[1]) if up else (p[0], p[1] + 1)
        if not _inside(mask.shape, ahead):
            end = 'edge'
            break
        if mask[ahead]:
            p = ahead
            continue
        collisions.append(ahead)
        who = int(owner[ahead])
        if who >= 0:
            word = objects[who]['word']
            loaded.append(who)
        next_p = (p[0], p[1] + 1) if up else (p[0] - 1, p[1])
        if not _inside(mask.shape, next_p) or not mask[next_p]:
            end = 'blocked'
            break
        up = not up
        lengths.append(0)
        p = next_p
    return dict(path=tuple(path), paint=tuple(paint), collisions=tuple(collisions),
                legs=tuple(lengths), loaded=tuple(loaded), end=end)


def _evaluate(info, *, variant=None, independent=False):
    objects = info['objects']
    width = info['arena'].shape[1]
    if variant == 'no_packing':
        settled = [dict(o) for o in objects]
    elif variant == 'edge_first':
        settled = _edge_first_pack(objects, width)
    elif independent:
        settled = _physical_pack(objects, width)
    else:
        settled = _pack(objects, width)
    base, owner = _arena(info['arena'].shape, info['bg'], info['frames'], settled)
    mask = base == info['bg']
    output = info['grid'].copy()
    output[:3] = info['bg']
    output[3:] = base
    occupied, records, offset = set(), [], 0
    for start in info['starts']:
        if independent:
            record = _step_trace(mask, owner, settled, start, info['word'])
        else:
            record = _trace(mask, owner, settled, start, info['word'], variant=variant,
                            offset=offset if variant == 'shared_phase' else 0)
        footprint = set(record['path']) | set(record['collisions'])
        if footprint & occupied:
            raise ValueError("Probe footprints interfere")
        occupied |= footprint
        for (r, c), color in zip(record['path'], record['paint']):
            output[r + 3, c] = color
        for r, c in record['collisions']:
            output[r + 3, c] = info['marker']
        records.append(record)
        offset += len(record['path'])
    return output, settled, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the visible header, rails, frames, payloads and starts."""
    info = _decode(input_grid)
    output = _evaluate(info)[0]
    return np.rot90(output, -info['rotation']).copy()


def _evidence(info):
    expected, settled, records = _evaluate(info)
    found = set()
    for wrong in ('no_packing', 'edge_first', 'ignore_payload', 'deduplicate',
                  'reset_bend', 'reset_load', 'wrapped_clock', 'reverse_payload',
                  'consume_marker', 'shared_phase'):
        try:
            alternative = _evaluate(info, variant=wrong)[0]
        except (ValueError, AssertionError):
            # An invalid competing scene is not counted as a labeled witness.
            continue
        if not np.array_equal(expected, alternative):
            found.add(wrong)
    if any(len(r['path']) >= 2 * len(info['word']) and r['legs'][0] >= 2 * len(info['word'])
           for r in records):
        found.add('initial_cycling')
    if any(len(r['legs']) >= 3 and len(r['loaded']) >= 2 for r in records):
        found.add('multiple_loads')
    moved = {a['d'] for a, b in zip(info['objects'], settled) if a['c'] != b['c']}
    found |= {f'move_{d}' for d in moved}
    if any(r['end'] == 'edge' and r['legs'][-1] >= 2 for r in records):
        found.add('edge_exit')
    return found


_REQUIRED = {'no_packing', 'edge_first', 'ignore_payload', 'deduplicate',
             'reset_bend', 'reset_load', 'wrapped_clock', 'reverse_payload',
             'consume_marker', 'shared_phase', 'initial_cycling',
             'multiple_loads', 'move_0', 'move_1', 'edge_exit'}


def _sample_word(rng, colors, length, repeated=False):
    if repeated:
        distinct = int(rng.integers(2, min(length - 1, len(colors)) + 1))
        values = list(map(int, rng.choice(colors, distinct, replace=False)))
        word = values + [values[int(rng.integers(distinct))] for _ in range(length-distinct)]
        rng.shuffle(word)
    else:
        word = list(map(int, rng.choice(colors, length, replace=True)))
    return tuple(word)


def _appear(rng, objects, colors, frames):
    for o in objects:
        word = _sample_word(rng, colors, o['w'] - 2)
        o['word'], o['frame'] = word, frames[o['d']]
        patch = np.full((o['h'], o['w']), o['frame'], dtype=np.int8)
        patch[1:-1, 1:-1] = word
        o['patch'] = patch


def _spread(rng, widths, inner_width):
    """Random gaps with a clear margin on both sides and preserved order."""
    free = inner_width - sum(widths)
    if free < 2:
        raise ValueError("No margin for a moving witness")
    gaps = rng.multinomial(free - 2, np.full(len(widths)+1, 1/(len(widths)+1)))
    gaps[0] += 1
    gaps[-1] += 1
    positions, c = [], 1 + int(gaps[0])
    for i, width in enumerate(widths):
        positions.append(c)
        c += width + int(gaps[i + 1])
    return positions


def _anchor_layout(rng, chain=True, test=False):
    """Interacting teeth, with variable programs, heights and stable overlap graph."""
    for _ in range(100):
        nl = int(rng.integers(2, 5)) if test else 3
        nr = int(rng.integers(2, 4)) if test else 2
        left_w = list(map(int, rng.integers(3, 7, nl)))
        right_w = list(map(int, rng.integers(3, 8 if test else 7, nr)))
        width = sum(left_w) + sum(right_w) + 2
        if not 19 <= width <= 30:
            continue
        lh = int(rng.integers(3, 6))
        step = int(rng.integers((lh+1)//2, lh))
        top = int(rng.integers(1, 4))
        cap_bottom = top + lh - 1
        objects = []
        # Destination order and vertical order differ in a stable chain.
        for i, ow in enumerate(left_w):
            row = top + (nl - 1 - i) * step if chain else top
            objects.append(dict(r=row, c=0, h=lh, w=ow, d=0))
        row = cap_bottom + 1
        right_objects = []
        for ow in right_w:
            oh = int(rng.integers(3, 6))
            right_objects.append(dict(r=row, c=0, h=oh, w=ow, d=1))
            row += oh - 1  # One-row overlap creates successive packed layers.
        start_row = row
        height = max(start_row + int(rng.integers(3, 6)), max(o['r']+o['h'] for o in objects) + 1)
        if height > 27:
            continue
        # Left chains advance right; right chains advance left in destination order.
        left_c = _spread(rng, left_w, width - 2)
        reversed_c = _spread(rng, list(reversed(right_w)), width - 2)
        right_c = list(reversed(reversed_c))
        for o, c in zip(objects, left_c):
            o['c'] = c
        for o, c in zip(right_objects, right_c):
            o['c'] = c
        objects += right_objects
        if any(_overlap(a, b) for i, a in enumerate(objects) for b in objects[i+1:]):
            continue
        try:
            settled = _pack(objects, width)
        except ValueError:
            continue
        if not all(o['c'] > 1 for o in objects if o['r'] <= start_row < o['r']+o['h']):
            continue
        return (height, width), objects, start_row
    raise ValueError("Could not construct interacting teeth")


def _free_layout(rng, min_width, test):
    height = int(rng.integers(10, 28))
    width = int(rng.integers(max(17, min_width), 31))
    requested = int(rng.integers(1, 11))
    objects = []
    for _ in range(requested):
        for _ in range(45):
            oh, ow = int(rng.integers(3, 7)), int(rng.integers(3, 8 if test else 7))
            o = dict(r=int(rng.integers(0, height-oh+1)),
                     c=int(rng.integers(1, width-ow)), h=oh, w=ow,
                     d=int(rng.integers(2)))
            if any(_overlap(o, b) for b in objects):
                continue
            try:
                _pack(objects + [o], width)
            except ValueError:
                continue
            objects.append(o)
            break
    if not objects:
        objects = [dict(r=2, c=3, h=3, w=3, d=1)]
    return (height, width), objects


def _render_input(rng, shape, objects, starts, palette, word):
    bg, low, high, marker = palette[:4]
    arena = _arena(shape, bg, (low, high), objects)[0]
    for start in starts:
        if arena[start] != bg:
            raise ValueError("Input cartridge covers a start")
        arena[start] = word[0]
    header = np.full((3, shape[1]), bg, dtype=np.int8)
    slack = shape[1] - 3 - (1 + 3 * len(word) - 1) - 3
    # One-cell legend gaps always fit; extra spacing varies when there is slack.
    left = int(rng.integers(1, min(2, 1 + max(0, slack)) + 1))
    columns, c = [], left
    for i in range(len(word)):
        columns.append(c)
        c += 2
        if i + 1 < len(word):
            remaining = len(word) - i - 1
            spare = shape[1] - 3 - 3 - c - (3 * remaining - 1)
            c += 1 + int(spare > 0 and rng.random() < .35)
    if columns[-1] + 2 > shape[1] - 6:
        raise ValueError("Legend does not fit beside separate marker")
    for c, color in zip(columns, word):
        header[:2, c:c+2] = color
    header[:2, -3:-1] = marker
    return np.vstack((header, arena))


def _select_starts(rng, shape, objects, bg, frames, initial_word, desired, preferred=None):
    settled = _pack(objects, shape[1])
    arena, owner = _arena(shape, bg, frames, settled)
    raw = _arena(shape, bg, frames, objects)[0]
    mask = arena == bg
    rows = list(map(int, rng.permutation(shape[0])))
    if preferred is not None:
        rows = [preferred] + [r for r in rows if r != preferred]
    starts, occupied = [], set()
    for r in rows:
        p = (r, 1)
        if not mask[p] or raw[p] != bg or any(abs(r-s[0]) < 2 for s in starts):
            continue
        record = _trace(mask, owner, settled, p, initial_word)
        footprint = set(record['path']) | set(record['collisions'])
        if len(record['path']) < 4 or footprint & occupied:
            continue
        starts.append(p)
        occupied |= footprint
        if len(starts) == desired:
            break
    if not starts or (preferred is not None and starts[0] != (preferred, 1)):
        raise ValueError("No useful unobstructed start")
    return tuple(sorted(starts))


def _make_pair(rng, kind, rotation, test=False):
    # Five-entry initial programs are held out, with no new instruction colors.
    length = 5 if test else int(rng.integers(3, 5) if kind == 'witness' else rng.integers(2, 5))
    for _ in range(250):
        palette = tuple(map(int, rng.permutation(10)))
        bg, low, high, marker = palette[:4]
        colors = palette[4:]
        word = _sample_word(rng, colors, length, repeated=kind == 'witness')
        min_width = 3 * length + 6
        try:
            if kind == 'witness' or (test and rng.random() < .6):
                shape, objects, preferred = _anchor_layout(rng, chain=True, test=test)
                if shape[1] < min_width:
                    continue
                desired = 2 if kind == 'witness' else int(rng.integers(1, 4))
            else:
                shape, objects = _free_layout(rng, min_width, test)
                preferred = None
                desired = int(rng.integers(1, 4))
            _appear(rng, objects, colors, (low, high))
            starts = _select_starts(rng, shape, objects, bg, (low, high), word, desired, preferred)
            inp = _render_input(rng, shape, objects, starts, palette, word)
            info = _decode_canonical(inp)
            if kind == 'witness' and not _REQUIRED <= _evidence(info):
                continue
            output = _evaluate(info)[0]
            return {'input': np.rot90(inp, rotation).copy(),
                    'output': np.rot90(output, rotation).copy()}
        except (ValueError, AssertionError):
            continue
    raise RuntimeError("Bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    base_rotation = int(rng.integers(4))
    witness = int(rng.integers(nt))
    mixed = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            kind = 'witness' if split == 'train' and i == witness else 'free'
            rotation = int(rng.integers(4)) if mixed else base_rotation
            for _ in range(20):
                pair = _make_pair(rng, kind, rotation, test=split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check the visible rule, separate simulations, and demonstration evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, train_lengths, test_lengths = set(), set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            expected, settled, records = _evaluate(info, independent=True)
            assert np.array_equal(np.rot90(expected, -info['rotation']), out), "Wrong route or packing"
            fast, recurrence_settled, fast_records = _evaluate(info)
            assert np.array_equal(fast, expected) and fast_records == records
            assert [o['c'] for o in settled] == [o['c'] for o in recurrence_settled]
            # Every carried interior remains intact; only blocked frame pixels
            # can change to the collision marker after translation.
            marked = {p for r in records for p in r['collisions']}
            for before, after in zip(info['objects'], settled):
                assert all(before[k] == after[k] for k in ('r', 'h', 'w', 'd', 'word'))
                r, c, h, w = (after[k] for k in ('r', 'c', 'h', 'w'))
                patch = expected[r+3:r+3+h, c:c+w]
                assert np.array_equal(patch[1:-1, 1:-1], before['patch'][1:-1, 1:-1])
                for rr in range(h):
                    for cc in range(w):
                        if (r+rr, c+cc) not in marked:
                            assert patch[rr, cc] == before['patch'][rr, cc]
            assert np.all(expected[:3] == info['bg'])
            if split == 'train':
                evidence |= _evidence(info)
                train_lengths.add(len(info['word']))
            else:
                test_lengths.add(len(info['word']))
    assert _REQUIRED <= evidence, f"Missing training witnesses: {_REQUIRED - evidence}"
    assert max(train_lengths) < min(test_lengths), "Missing visible-program length shift"


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def measure_puzzle(puzzle):
    """Descriptive family features, recovered from pixels rather than sample tags."""
    rows = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _decode(pair['input'])
            _, settled, records = _evaluate(info)
            objects = info['objects']
            rows.append(dict(split=split, shape=list(np.shape(pair['input'])),
                rotation=info['rotation'], roles=[info['bg'], *info['frames'], info['marker']],
                objects=len(objects), classes=[sum(o['d'] == d for o in objects) for d in (0, 1)],
                starts=len(info['starts']), initial_length=len(info['word']),
                word_pattern=_pattern(info['word']),
                payload_lengths=sorted({len(o['word']) for o in objects}),
                footprints=sorted({(o['h'], o['w']) for o in objects}),
                destinations=sorted({o['c'] for o in settled}),
                legs=[list(r['legs']) for r in records],
                loads=[len(r['loaded']) for r in records],
                ends=[r['end'] for r in records],
                travel=[abs(a['c']-b['c']) for a, b in zip(objects, settled)],
                evidence=sorted(_evidence(info))))
    return rows


def _pattern(word):
    mapping = {}
    return tuple(mapping.setdefault(c, len(mapping)) for c in word)


def self_test():
    """Bounded 42-seed corpus, all count combinations, and semantic regressions."""
    corpus = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in corpus}) == 42
    assert _fingerprint(corpus[0]) == _fingerprint(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(7000 + 10*a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    rows = [r for p in corpus for r in measure_puzzle(p)]
    assert len({tuple(r['shape']) for r in rows}) >= 35
    assert {r['rotation'] for r in rows} == {0, 1, 2, 3}
    assert len({tuple(r['roles']) for r in rows}) >= 80
    assert len({r['objects'] for r in rows}) >= 6
    assert {r['starts'] for r in rows} >= {1, 2}
    assert len({tuple(r['word_pattern']) for r in rows}) >= 15
    assert {n for r in rows for n in r['payload_lengths']} == {1, 2, 3, 4, 5}
    assert len({tuple(f) for r in rows for f in r['footprints']}) >= 15
    assert {'edge', 'blocked'} <= {e for r in rows for e in r['ends']}
    assert max(n for r in rows for n in r['loads']) >= 3
    assert any(0 in r['travel'] for r in rows)
    assert any(min(r['classes']) == 0 for r in rows)
    # Explicit route clock: phase continues across a word replacement, and
    # a marker does not increment it. This fixture does not use generation.
    mask = np.ones((7, 10), bool)
    mask[:, 0] = mask[:, -1] = False
    mask[3:6, 6:9] = False
    owner = np.full(mask.shape, -1, int)
    owner[3:6, 6:9] = 0
    objects = [dict(word=(7, 8, 9))]
    r = _trace(mask, owner, objects, (5, 1), (2, 2, 4))
    assert r == _step_trace(mask, owner, objects, (5, 1), (2, 2, 4))
    assert r['paint'][:5] == (2, 2, 4, 2, 2)
    assert r['paint'][5] == 9  # New word index 5 modulo 3, not index zero.
    assert r['end'] == 'edge' and r['path'][-1] == (0, 5)
    # No turn after top exit, even though free cells exist to the right.
    assert (0, 6) not in r['path']
    stable = [dict(r=8, c=3, h=4, w=5, d=0),
              dict(r=6, c=10, h=4, w=4, d=0),
              dict(r=4, c=17, h=4, w=3, d=0)]
    assert [o['c'] for o in _pack(stable, 25)] == [1, 6, 10]
    assert [o['c'] for o in _physical_pack(stable, 25)] == [1, 6, 10]
    assert [o['c'] for o in _edge_first_pack(stable, 25)] == [1, 6, 1]
    # Every explicit incorrect model has an actual differing training output.
    # Mutating one output also proves the validator does not merely check schema.
    p = corpus[0]
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return dict(seeds=42, pairs=len(rows), shapes=len({tuple(r['shape']) for r in rows}),
                object_counts=sorted({r['objects'] for r in rows}),
                initial_patterns=len({tuple(r['word_pattern']) for r in rows}),
                route_legs=dict(Counter(len(leg) for r in rows for leg in r['legs'])))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
