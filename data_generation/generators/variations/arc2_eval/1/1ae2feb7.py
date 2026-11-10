"""Separator-relative pulse trains with visible, discriminating curricula.

All semantic conventions are invariant: maximal run length is the period,
phase starts at the separator, nearer runs win, and colors are copied locally.
Puzzle-level sampling controls the curriculum and distribution, never a hidden
rule. Construction paints pulses from far to near; validation reconstructs
geometry from the input and checks each cell by modular arithmetic.
"""
import itertools
import numpy as np


# Exact source contents serve regression tests and copy exclusion, not templates.
_SOURCE_ROWS = ((('0000020000000000', '0000020000000000', '0000020000000000', '1111020000000000', '0000020000000000', '6666820000000000', '0000020000000000', '3333320000000000', '0000020000000000', '0555520000000000', '0000020000000000'), ('0000020000000000', '0000020000000000', '0000020000000000', '1111021000100010', '0000020000000000', '6666828888888888', '0000020000000000', '3333323000030000', '0000020000000000', '0555525000500050', '0000020000000000')), (('0000020000000000', '0000020000000000', '0000020000000000', '0211020000000000', '0000020000000000', '0000020000000000', '0000020000000000', '3334420000000000', '0000020000000000', '0000020000000000', '0000020000000000'), ('0000020000000000', '0000020000000000', '0000020000000000', '0211021212121212', '0000020000000000', '0000020000000000', '0000020000000000', '3334424043404043', '0000020000000000', '0000020000000000', '0000020000000000')), (('000200000000000', '011200000000000', '000200000000000', '000200000000000', '000200000000000', '005200000000000', '000200000000000', '000200000000000', '000200000000000', '777200000000000', '000200000000000', '000200000000000', '000200000000000'), ('000200000000000', '011210101010101', '000200000000000', '000200000000000', '000200000000000', '005255555555555', '000200000000000', '000200000000000', '000200000000000', '777270070070070', '000200000000000', '000200000000000', '000200000000000')), (('05555400000000000000', '00011400000000000000', '00002400000000000000', '00000400000000000000', '00000400000000000000', '00766400000000000000', '00000400000000000000', '66888400000000000000', '00000400000000000000', '00000000000000000000'), ('05555450005000500050', '00011410101010101010', '00002422222222222222', '00000400000000000000', '00000400000000000000', '00766467676767676767', '00000400000000000000', '66888480686080686080', '00000400000000000000', '00000000000000000000')), (('000000000000300', '000000000002300', '000000000000300', '000000000044300', '000000000000300', '000000006666300', '000000000000300', '000000000097300'), ('000000000000300', '000000000002322', '000000000000300', '000000000044340', '000000000000300', '000000006666360', '000000000000300', '000000000097377')), (('000000000000300', '000000000000340', '000000000000300', '000000000000377', '000000000000300', '000000000000360', '000000000000300', '000000000000355'), ('000000000000300', '444444444444340', '000000000000300', '070707070707377', '000000000000300', '666666666666360', '000000000000300', '050505050505355')))


def _validate_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_train = int(num_train)
        largest_test_count = min(4, num_train - 1)
        num_test = int(rng.integers(1, largest_test_count + 1))
    else:
        num_train = int(num_train)
        num_test = int(num_test)

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _grid_key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output']))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _source_pairs():
    return [dict(zip(('input', 'output'),
                     (np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
                      for rows in pair))) for pair in _SOURCE_ROWS]


_SOURCE_KEYS = {(_grid_key(p['input']), _grid_key(p['output'])) for p in _source_pairs()}


def _analyze_input(grid):
    """Find a unique geometric interpretation, allowing all four directions.

    A candidate separator spans the occupied lanes, sits on the outer edge of
    the foreground, and has an empty drawing half-plane. This does not assume
    separated lanes or that the separator is longer than a payload run.
    """
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1, 'ambiguous background'
    background = int(counts.argmax())
    candidates = []
    for rotation in range(4):
        a = np.rot90(grid, rotation)
        rr, cc = np.nonzero(a != background)
        if not len(rr):
            continue
        start, end, col = int(rr.min()), int(rr.max()), int(cc.max())
        if end - start + 1 < 3 or not 0 < col < a.shape[1] - 1:
            continue
        separator = int(a[start, col])
        if separator == background or not np.all(a[start:end + 1, col] == separator):
            continue
        blocks = {}
        for row in range(start, end + 1):
            lane, cursor = [], col - 1
            while cursor >= 0:
                color = int(a[row, cursor])
                if color == background:
                    cursor -= 1
                    continue
                near = col - cursor
                length = 0
                while cursor >= 0 and a[row, cursor] == color:
                    length += 1
                    cursor -= 1
                lane.append({'color': color, 'length': length, 'proximity': near})
            if lane:
                blocks[row] = lane
        if blocks:
            candidates.append({'rotation': rotation, 'canonical': a,
                               'background': background, 'separator_color': separator,
                               'separator_col': col, 'separator_start': start,
                               'separator_end': end, 'instruction_width': col,
                               'draw_length': a.shape[1] - col - 1,
                               'row_blocks': blocks, 'active_rows': tuple(blocks)})
    assert len(candidates) == 1, 'separator and populated side must be unique'
    return candidates[0]


def _predict(grid, priority='near'):
    """Input-only cell-wise check, separate from construction's slice painter."""
    info = _analyze_input(grid)
    out = info['canonical'].copy()
    for row, blocks in info['row_blocks'].items():
        for offset in range(info['draw_length']):
            firing = [b for b in blocks if offset % b['length'] == 0]
            if firing:
                if priority == 'short':
                    winner = min(firing, key=lambda b: (b['length'], b['proximity']))
                elif priority == 'long':
                    winner = min(firing, key=lambda b: (-b['length'], b['proximity']))
                elif priority == 'far':
                    winner = firing[-1]
                else:
                    winner = firing[0]
                out[row, info['separator_col'] + 1 + offset] = winner['color']
    return np.rot90(out, -info['rotation']).copy()


def _visibility(lengths, draw_length):
    seen, collisions = set(), []
    for t in range(draw_length):
        hits = [i for i, length in enumerate(lengths) if t % length == 0]
        if hits:
            seen.add(hits[0])
        if t and len(hits) > 1:
            collisions.append(t + 1)
    return seen, collisions


def _mixture_options(count):
    # Enumerated arithmetic possibilities, not a bank of lane-layout recipes.
    result = []
    for lengths in itertools.product(range(2, 9), repeat=count):
        room = 29 - sum(lengths)
        if room < 6:
            continue
        visible, collisions = _visibility(lengths, room)
        if len(visible) != count or not collisions:
            continue
        for draw in range(max(lengths) + 1, room + 1):
            visible, collisions = _visibility(lengths, draw)
            if len(visible) == count and collisions:
                result.append((lengths, draw))
                break
    return tuple(result)


_VISIBLE_MIXTURES = {k: _mixture_options(k) for k in (3, 4)}
_WITNESS_PERIODS = tuple((a, b) for a in range(2, 9) for b in range(a + 1, 9)
                         if b % a and int(np.lcm(a, b)) + 1 <= 17
                         and a + b + int(np.lcm(a, b)) + 2 <= 30)


def _spec(lengths, colors=None, min_draw=1, gap=0):
    return {'lengths': tuple(lengths), 'colors': colors,
            'min_draw': min_draw, 'gap': gap}


def _random_spec(rng, simple, max_period, regime, min_draw, max_payload):
    count = 1 if simple else int(rng.choice([1, 2, 3, 4], p=[.43, .28, .19, .10]))
    count = min(count, max_payload)
    if count >= 3 and rng.random() < .78:
        options = [(ls, d) for ls, d in _VISIBLE_MIXTURES[count]
                   if max(ls) <= max_period and sum(ls) <= max_payload
                   and max(d, min_draw) + sum(ls) <= 29]
        if options:
            lengths, draw = options[int(rng.integers(len(options)))]
            return _spec(lengths, min_draw=draw)
    # A bounded composition allows equal, divisible, coprime, and occluded runs.
    lengths = []
    for index in range(count):
        room = max_payload - sum(lengths) - (count - index - 1)
        if regime == 'source_train':
            room = min(room, 28 - min_draw)
        lengths.append(int(rng.integers(1, min(max_period, room) + 1)))
    draw = max(lengths) + 1 if regime == 'source_train' else 1
    if max(draw, min_draw) + sum(lengths) > 29:
        return _spec([int(rng.integers(1, min(max_period, max_payload, 12) + 1))])
    return _spec(lengths, min_draw=draw)


def _sample_parameters(rng, num_train, num_test):
    """Shared distributions; the phase/priority/copy conventions never change."""
    simple = bool(rng.random() < .20)
    max_period = int(rng.integers(2, 9) if simple else rng.integers(3, 13))
    base = int(rng.choice(4, p=[.36, .14, .36, .14]))
    direction_mode = int(rng.choice(4, p=[.34, .25, .20, .21]))
    if direction_mode == 0:
        train_directions, test_directions = [base] * num_train, [base] * num_test
    elif direction_mode == 1:
        train_directions = [base, (base + 2) % 4]
        train_directions += [int(rng.choice([base, (base + 2) % 4])) for _ in range(num_train - 2)]
        test_directions = [int(rng.choice([base, (base + 2) % 4])) for _ in range(num_test)]
    elif direction_mode == 2:
        train_directions = [base, (base + int(rng.choice([1, 3]))) % 4]
        train_directions += [int(rng.integers(4)) for _ in range(num_train - 2)]
        test_directions = [int(rng.integers(4)) for _ in range(num_test)]
    else:
        train_directions, test_directions = [base] * num_train, [(base + 2) % 4] * num_test
    background = int(rng.integers(10))
    separator = int(rng.choice([c for c in range(10) if c != background]))
    plans = [[] for _ in range(num_train)]
    # Teach a sampled numerical range, rather than reciting five fixed lengths.
    periods = [max_period] + [int(v) for v in rng.choice(
        range(1, max_period), size=int(rng.integers(1, min(4, max_period))), replace=False)]
    slots = rng.permutation(num_train)
    for index, period in enumerate(periods):
        plans[int(slots[index % num_train])].append(
            _spec([period], min_draw=period + 1, gap=1 if index == 0 else 0))
    witness_colors = tuple(int(c) for c in rng.choice(
        [c for c in range(10) if c != background], size=2, replace=False))
    if not simple:
        options = [(a, b) for a, b in _WITNESS_PERIODS if b <= max_period]
        a, b = options[int(rng.integers(len(options)))]
        draw = int(np.lcm(a, b)) + 1
        slots = rng.choice(num_train, size=2, replace=True)
        for slot, ls, cs in zip(slots, [(a, b), (b, a)],
                                [witness_colors, witness_colors[::-1]]):
            plans[int(slot)].append(_spec(ls, cs, draw))
    return {'simple': simple, 'max_period': max_period, 'plans': plans,
            'directions': {'source_train': train_directions, 'target_test': test_directions},
            'background': background, 'separator': separator,
            'vary_background': bool(rng.random() < .72),
            'vary_separator': bool(rng.random() < .75)}


def _choose_rows(rng, height, count, fallback=False):
    # One bare separator lane prevents dense payloads from looking like a panel.
    if fallback:
        return 0, height, list(range(1, count + 1))
    segment = height if rng.random() < .35 else int(rng.integers(max(3, count + 1), height + 1))
    start = int(rng.integers(height - segment + 1))
    mode = int(rng.integers(4))
    if mode == 0 and segment >= 2 * count - 1:
        compressed = np.sort(rng.choice(segment - count + 1, count, replace=False))
        rows = [start + int(v) + i for i, v in enumerate(compressed)]
    elif mode == 1:
        first = start + int(rng.integers(segment - count + 1))
        rows = list(range(first, first + count))
    elif mode == 2 and count >= 3:
        cluster = int(rng.integers(2, count))
        first = int(rng.integers(segment - cluster + 1))
        local = list(range(first, first + cluster))
        other = [i for i in range(segment) if i not in local]
        rows = sorted(start + int(v) for v in local + list(rng.choice(other, count - cluster, replace=False)))
    else:
        rows = sorted(int(v) + start for v in rng.choice(segment, count, replace=False))
    return start, segment, rows


def _pair_geometry(rng, specs, regime, draw_range):
    payload = max(sum(s['lengths']) + s['gap'] for s in specs)
    minimum = max(s['min_draw'] for s in specs)
    if regime == 'source_train':
        minimum = max(minimum, max(max(s['lengths']) + 1 for s in specs))
    room = 29 - payload
    assert minimum <= room, 'incompatible teaching constraints'
    compact = rng.random() < .35
    if regime == 'target_test' and draw_range == 'clip':
        draw = int(rng.integers(1, min(3, room) + 1))
    elif regime == 'target_test' and draw_range == 'long':
        draw = int(rng.integers(max(minimum, room - 4), room + 1))
    elif isinstance(draw_range, tuple):
        low, high = max(minimum, min(draw_range[0], room)), min(draw_range[1], room)
        draw = int(rng.integers(low, max(low, high) + 1))
    else:
        draw = int(rng.integers(minimum, min(room, minimum + 4) + 1 if compact else room + 1))
    upper = min(29 - draw, payload + 4) if compact else 29 - draw
    instruction = int(rng.integers(payload, upper + 1))
    if not compact and rng.random() < .14:
        instruction = 29 - draw
    count = len(specs)
    height = int(rng.integers(max(3, count + 1), 31))
    if count <= 4 and rng.random() < .32:
        height = int(rng.integers(max(3, count + 1), min(11, max(3, count + 1) + 5)))
    return instruction, draw, height


def _paint_pair(rng, specs, instruction, draw, height, colors, rotation, fallback=False):
    background, separator = colors
    if fallback:
        height, instruction = max(height, len(specs) + 2), 29 - draw
    start, segment, rows = _choose_rows(rng, height, len(specs), fallback)
    inp = np.full((height, instruction + 1 + draw), background, dtype=np.int8)
    inp[start:start + segment, instruction] = separator
    out = inp.copy()
    palette = [c for c in range(10) if c != background]
    palette = [int(v) for v in rng.choice(palette, int(rng.integers(2, 10)), replace=False)]
    if fallback:
        palette = [c for c in range(10) if c not in (background, separator)]
    for index, (row, spec) in enumerate(zip(rows, specs)):
        lengths = spec['lengths']
        chosen = spec['colors']
        if chosen is None:
            if fallback:
                chosen = [palette[(index + i) % len(palette)] for i in range(len(lengths))]
            else:
                chosen = [int(rng.choice(palette)) for _ in lengths]
                # Keep most rich mixtures visibly multicolored, but retain repeats.
                if len(lengths) >= 3 and rng.random() < .65:
                    chosen = list(map(int, rng.choice([c for c in range(10) if c != background],
                                                      len(lengths), replace=False)))
        gaps = [spec['gap']] + [int(chosen[i] == chosen[i - 1]) for i in range(1, len(lengths))]
        spare = instruction - sum(lengths) - sum(gaps)
        if spare < 0:
            # No space to split a repeated color: use a different adjacent color.
            chosen = list(chosen)
            for i in range(1, len(chosen)):
                if chosen[i] == chosen[i - 1]:
                    chosen[i] = next(c for c in range(10) if c not in (background, chosen[i - 1]))
            gaps = [spec['gap']] + [0] * (len(lengths) - 1)
            spare = instruction - sum(lengths) - sum(gaps)
        assert spare >= 0
        # Include the outer margin as an independent gap compartment.
        budget = int(rng.integers(spare + 1))
        additions = rng.multinomial(budget, [1 / (len(gaps) + 1)] * (len(gaps) + 1))
        gaps = [g + int(a) for g, a in zip(gaps, additions)]
        cursor = instruction
        for length, color, gap in zip(lengths, chosen, gaps):
            cursor -= gap
            inp[row, cursor - length:cursor] = color
            cursor -= length
        out[row, :instruction] = inp[row, :instruction]
        # This construction uses specified runs and overwriting, not the parser.
        for length, color in reversed(list(zip(lengths, chosen))):
            out[row, instruction + 1::length] = color
    return {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(out, rotation).copy()}


def _generate_pair(rng, regime, parameters, index, previous_count, train_draws):
    specs = list(parameters['plans'][index]) if regime == 'source_train' else []
    shift = str(rng.choice(['same', 'clip', 'long', 'dense'], p=[.40, .25, .20, .15]))
    dense = rng.random() < (.48 if regime == 'target_test' and shift == 'dense' else .20)
    if dense:
        count = int(rng.integers(max(6, len(specs)), 29))
    else:
        count = max(len(specs), int(rng.choice([1, 2, 3, 4, 5, 6, 8], p=[.19, .21, .20, .16, .12, .08, .04])))
    if count == previous_count:
        count = count + 1 if count < 28 else count - 1
    minimum = max([s['min_draw'] for s in specs] + [1])
    payload = max([sum(s['lengths']) + s['gap'] for s in specs] + [1])
    while len(specs) < count:
        budget = min(16, 29 - minimum)
        if regime == 'target_test' and shift == 'long':
            # Reserve room for an actual extension beyond every demonstration,
            # rather than merely sampling a wide region that training exceeds.
            budget = min(budget, max(1, 28 - max(train_draws)))
        spec = _random_spec(rng, parameters['simple'], parameters['max_period'], regime,
                            minimum, budget)
        candidate_min = max(minimum, spec['min_draw'])
        candidate_payload = max(payload, sum(spec['lengths']) + spec['gap'])
        if candidate_min + candidate_payload > 29:
            spec = _spec([int(rng.integers(1, min(parameters['max_period'], payload) + 1))])
            candidate_min = max(minimum, max(spec['lengths']) + (regime == 'source_train'))
            candidate_payload = payload
        specs.append(spec)
        minimum, payload = candidate_min, candidate_payload
    specs = [specs[int(i)] for i in rng.permutation(len(specs))]
    draw_range = None
    if regime == 'target_test':
        draw_range = ('clip' if shift == 'clip' else (max(train_draws) + 1, 28)
                      if shift == 'long' else (min(train_draws), max(train_draws)))
    instruction, draw, height = _pair_geometry(rng, specs, regime, draw_range)
    forbidden_bg = {c for s in specs if s['colors'] is not None for c in s['colors']}
    background = parameters['background']
    if parameters['vary_background'] and (index > 0 or regime == 'target_test'):
        available = [c for c in range(10) if c not in forbidden_bg and
                     (c != parameters['background'] if regime == 'source_train' and index == 1 else True)]
        background = int(rng.choice(available))
    separator = parameters['separator']
    if separator == background or (parameters['vary_separator'] and rng.random() < .70):
        separator = int(rng.choice([c for c in range(10) if c != background]))
    rotation = parameters['directions'][regime][index]
    # A few geometric trials, then a guaranteed recognizable two-anchor layout.
    for attempt in range(9):
        pair = _paint_pair(rng, specs, instruction, draw, height, (background, separator),
                           rotation, fallback=attempt == 8)
        try:
            _analyze_input(pair['input'])
            return pair
        except AssertionError:
            if attempt == 8:
                raise
    raise AssertionError('unreachable')


def _teaching_evidence(train):
    singles, shifted, short_near, long_near, edges = set(), False, False, False, set()
    for pair in train:
        info = _analyze_input(pair['input'])
        output = np.rot90(pair['output'], info['rotation'])
        draw, col = info['draw_length'], info['separator_col']
        for row, blocks in info['row_blocks'].items():
            if len(blocks) == 1:
                block = blocks[0]
                length, color = block['length'], block['color']
                if draw > length and output[row, col + 1] == color and output[row, col + length + 1] == color:
                    if length == 1 or np.all(output[row, col + 2:col + length + 1] == info['background']):
                        singles.add(length)
                        shifted |= block['proximity'] > 1
            for i, near in enumerate(blocks):
                for far in blocks[i + 1:]:
                    if near['color'] == far['color']:
                        continue
                    # Count only observed collisions won by this near run, and a
                    # separate unoccluded pulse of this specific farther run.
                    collision, survives = False, False
                    for t in range(draw):
                        firing = [b for b in blocks if t % b['length'] == 0]
                        if not firing:
                            continue
                        winner = firing[0]
                        if winner is near and far in firing and output[row, col + t + 1] == near['color']:
                            collision = True
                        if winner is far and output[row, col + t + 1] == far['color']:
                            survives = True
                    if collision and survives:
                        short_near |= near['length'] < far['length']
                        long_near |= near['length'] > far['length']
                        edges.add((near['color'], far['color']))
    return {'single_periods': singles, 'offset_phase': shifted,
            'short_near': short_near, 'long_near': long_near,
            'color_cycle': any((b, a) in edges for a, b in edges)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs, infos = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            ikey, okey = _grid_key(inp), _grid_key(out)
            assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
            assert (ikey, okey) not in _SOURCE_KEYS, 'exact source copy'
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            info = _analyze_input(inp)
            assert np.array_equal(out, _predict(inp)), 'phase, segmentation, priority, clipping, or preservation'
            for blocks in info['row_blocks'].values():
                assert 1 <= len(blocks) <= 4
                assert all(1 <= b['length'] <= 12 for b in blocks)
                if split == 'train':
                    assert info['draw_length'] > max(b['length'] for b in blocks), 'training must allow recurrence'
            infos[split].append(info)
    evidence = _teaching_evidence(puzzle['train'])
    assert len(evidence['single_periods']) >= 2, 'need two numerical recurrence examples'
    assert evidence['offset_phase'], 'phase must be independent of source-side gap'
    # Rich examples anywhere in the puzzle are accompanied by a full priority lesson.
    rich = any(len(bs) > 1 for split in infos for i in infos[split] for bs in i['row_blocks'].values())
    if rich:
        assert evidence['short_near'] and evidence['long_near'] and evidence['color_cycle'], 'priority teaching gap'
        for alternative in ('short', 'long', 'far'):
            assert any(not np.array_equal(p['output'], _predict(p['input'], alternative))
                       for p in puzzle['train']), 'a rival priority rule fits all training'
    largest_taught = max(evidence['single_periods'])
    assert all(b['length'] <= largest_taught for i in infos['test'] for bs in i['row_blocks'].values() for b in bs)
    assert len({len(i['active_rows']) for split in infos for i in infos[split]}) >= 2
    # Colors and directions are geometric roles, not hidden palette or side rules.
    # Clipped tests may omit recurrence; training supplies it before that shift.


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    parameters = _sample_parameters(rng, num_train, num_test)
    puzzle, seen, previous_count, train_draws = {'train': [], 'test': []}, set(), None, []
    for split, regime, count in [('train', 'source_train', num_train), ('test', 'target_test', num_test)]:
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _generate_pair(pair_rng, regime, parameters, index, previous_count, train_draws)
            # Deterministic finite recoloring fallback: fix every witness color so
            # opposite precedence demonstrations still refer to the same colors.
            protected = {c for plan in parameters['plans'] for s in plan
                         if s['colors'] is not None for c in s['colors']}
            available = [c for c in range(10) if c not in protected]
            original = pair
            for order in itertools.permutations(available):
                key = (_grid_key(pair['input']), _grid_key(pair['output']))
                if key[0] not in seen and key not in _SOURCE_KEYS:
                    break
                mapping = np.arange(10, dtype=np.int8)
                mapping[available] = order
                pair = {role: mapping[grid] for role, grid in original.items()}
            else:
                raise AssertionError('exhausted distinct colorings')
            seen.add(_grid_key(pair['input']))
            puzzle[split].append(pair)
            info = _analyze_input(pair['input'])
            previous_count = len(info['active_rows'])
            if split == 'train':
                train_draws.append(info['draw_length'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_features(pair):
    """Review measurements reconstructed from actual returned grids."""
    info = _analyze_input(pair['input'])
    out = np.rot90(pair['output'], info['rotation'])
    blocks = list(info['row_blocks'].values())
    rows = info['active_rows']
    gaps, margins, visible_rich, rich, repeated, divisibility, equal, coprime = [], [], 0, 0, False, False, False, False
    for row, lane in info['row_blocks'].items():
        previous = 1
        for block in lane:
            gaps.append(block['proximity'] - previous)
            previous = block['proximity'] + block['length']
        margins.append(info['instruction_width'] - previous + 1)
        repeated |= len({b['color'] for b in lane}) < len(lane)
        lengths = [b['length'] for b in lane]
        for a, b in itertools.combinations(lengths, 2):
            equal |= a == b
            divisibility |= a != b and max(a, b) % min(a, b) == 0
            coprime |= a > 1 and b > 1 and int(np.gcd(a, b)) == 1
        if len(lane) >= 3:
            rich += 1
            seen, collisions = _visibility(lengths, info['draw_length'])
            # Distinct visible colors, not just different hidden run indices.
            visible_rich += int(len(seen) == len(lane) and len({b['color'] for b in lane}) == len(lane)
                                and bool(collisions) and all(
                                    b['color'] in out[row, info['separator_col'] + 1:] for b in lane))
    row_gaps = tuple(b - a for a, b in zip(rows, rows[1:]))
    return {'shape': tuple(pair['input'].shape), 'direction': (-info['rotation']) % 4,
            'background': info['background'], 'separator': info['separator_color'],
            'lane_count': len(rows), 'adjacent': 1 in row_gaps,
            'consecutive': len(rows) > 1 and all(d == 1 for d in row_gaps),
            'irregular': len(set(row_gaps)) >= 2, 'single_only': all(len(b) == 1 for b in blocks),
            'max_blocks': max(map(len, blocks)), 'periods': tuple(sorted({b['length'] for bs in blocks for b in bs})),
            'separator_length': info['separator_end'] - info['separator_start'] + 1,
            'full_separator': info['separator_start'] == 0 and info['separator_end'] + 1 == info['canonical'].shape[0],
            'instruction_width': info['instruction_width'], 'draw_length': info['draw_length'],
            'max_gap': max(gaps), 'gap_budget': max(sum(b['proximity'] - (lane[i - 1]['proximity'] + lane[i - 1]['length'] if i else 1)
                                                       for i, b in enumerate(lane)) for lane in blocks),
            'max_margin': max(margins), 'repeated_color': repeated,
            'separator_reuse': any(b['color'] == info['separator_color'] for bs in blocks for b in bs),
            'rich_lanes': rich, 'visible_rich_lanes': visible_rich,
            'divisible': divisibility, 'equal': equal, 'coprime': coprime,
            'layout': (info['instruction_width'], info['draw_length'], rows,
                       tuple(tuple((b['length'], b['proximity']) for b in bs) for bs in blocks))}


def self_test() -> None:
    # Exact regressions are fixed evidence, not values regenerated by a solver.
    for pair in _source_pairs():
        assert np.array_equal(_predict(pair['input']), pair['output'])
    # Hand-computed diagnostics also cover extensions absent from the source.
    # Tuples are (instruction row, separator color, expected drawing cells).
    diagnostics = [
        ('022', '4', '202020'),                  # Compact 5 x 10 geometry.
        ('003', '4', '333333'),
        ('44333222211111', '8', '104321302310204'),  # Four visible contributors.
        ('33222111111', '8', '103230103230103230'),  # Divisible 6, 3, 2.
        ('0220111022', '8', '202120202120'),      # Separate repeated colors.
        ('0033330000', '8', '300030003000'),      # Gap does not shift phase.
        ('22111', '8', '1'),                     # One-cell clipping.
        ('222211', '8', '101010101010'),         # Equal/divisible occlusion.
    ]
    for source, separator, pulses in diagnostics:
        blank = '0' * len(source) + separator + '0' * len(pulses)
        before = [blank] * 5
        after = [blank] * 5
        before[2] = source + separator + '0' * len(pulses)
        after[2] = source + separator + pulses
        inp = np.array([list(map(int, row)) for row in before], dtype=np.int8)
        expected = np.array([list(map(int, row)) for row in after], dtype=np.int8)
        for rotation in range(4):
            assert np.array_equal(_predict(np.rot90(inp, rotation)), np.rot90(expected, rotation))
    automatic = [generate_puzzle(seed) for seed in range(42)]
    features = []
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                features.append(_pair_features(pair))
                assert np.array_equal(_predict(pair['input']), pair['output'])
    for counts in [(2, 1), (10, 1), (10, 4), (7, 3), (None, 1), (None, 4), (2, None), (10, None)]:
        a = generate_puzzle(901, *counts)
        assert _puzzle_key(a) == _puzzle_key(generate_puzzle(901, *counts))
        for split, requested in zip(('train', 'test'), counts):
            assert requested is None or len(a[split]) == requested
    assert _puzzle_key(generate_puzzle(902)) == _puzzle_key(generate_puzzle(902))
    assert _puzzle_key(generate_puzzle(903)) != _puzzle_key(generate_puzzle(904))
    for counts in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 3), (3, 4), (4, 4), (True, 1), (2.0, 1)]:
        try:
            generate_puzzle(1200, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # The reported two-demo counterexamples and a wider two-demo suite.
    for seed in range(200):
        p = generate_puzzle(seed, 2, 1)
        for alternative in ('short', 'long', 'far'):
            fits = all(np.array_equal(_predict(q['input'], alternative), q['output']) for q in p['train'])
            if fits:
                assert all(np.array_equal(_predict(q['input'], alternative), q['output']) for q in p['test'])
    assert len({len(p['train']) for p in automatic}) >= 4
    assert len({len(p['test']) for p in automatic}) >= 3
    assert len({_puzzle_key(p) for p in automatic}) == 42
    assert len({f['shape'] for f in features}) >= 90
    assert len({f['layout'] for f in features}) >= 150
    assert {f['direction'] for f in features} == set(range(4))
    assert len({f['background'] for f in features}) == 10
    assert len({f['lane_count'] for f in features}) >= 15
    assert set(range(1, 13)) <= {v for f in features for v in f['periods']}
    for field in ('adjacent', 'consecutive', 'irregular', 'single_only', 'full_separator',
                  'repeated_color', 'separator_reuse', 'divisible', 'equal', 'coprime'):
        assert {f[field] for f in features} == {False, True}, field
    assert {f['max_blocks'] for f in features} == {1, 2, 3, 4}
    assert min(f['separator_length'] for f in features) <= 4
    assert max(f['max_gap'] for f in features) >= 5
    assert max(f['gap_budget'] for f in features) >= 6
    assert max(f['max_margin'] for f in features) >= 6
    assert min(f['draw_length'] for f in features) == 1
    assert max(f['draw_length'] for f in features) >= 20
    assert min(min(f['shape']) for f in features) <= 5
    assert max(max(f['shape']) for f in features) == 30
    rich = sum(f['rich_lanes'] for f in features)
    visible = sum(f['visible_rich_lanes'] for f in features)
    assert visible >= .40 * rich and visible >= 30, (visible, rich)
    groups = [[_pair_features(q) for split in ('train', 'test') for q in p[split]] for p in automatic]
    assert all(len({f['lane_count'] for f in group}) > 1 for group in groups)
    for field in ('direction', 'background', 'separator', 'max_blocks', 'adjacent', 'draw_length', 'shape'):
        assert sum(len({f[field] for f in group}) > 1 for group in groups) >= 12, field
    assert any(len({f['direction'] for f in group}) == 1 for group in groups)
    assert any(all(f['single_only'] for f in group) for group in groups)
    assert any(f['adjacent'] and f['max_blocks'] == 4 and f['separator_reuse'] and f['max_gap'] > 0 for f in features)
    # Mutation and metamorphic checks target errors that shared generation state
    # would miss: a changed preserved cell, wrong phase, and every orientation.
    original = automatic[0]
    changed = {s: [{r: a.copy() for r, a in p.items()} for p in original[s]] for s in original}
    changed['test'][0]['output'][0, 0] = (int(changed['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(changed)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    for pair in _source_pairs():
        for turn in range(4):
            assert np.array_equal(_predict(np.rot90(pair['input'], turn)), np.rot90(pair['output'], turn))
