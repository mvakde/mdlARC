"""Pressure-programmed pin circuits. Only stdlib and NumPy are used.

solve reads the visible key, arrow and capped registers. It has no access to
sampling parameters, seed, demonstration outputs or cached examples.
"""
from __future__ import annotations

import json
import numpy as np

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0,102026823,552233901,297764544,810198489,503328707,338135665,
530342462,659137272,137270696,554533652,777294000,361585020,507872840,
648817459,580798697,298779948,738175928,834696875,732416567,671171551,
122155061,855224916,11720532,408294850,169220669,547410956,380278545,
556825284,557297004,554981081,108275163,334235247,96405841,232687639,
278118353,520886116,774785090,821371800,204587422,741417494,887357482)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), 'grid size')
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             'integer colors 0..9 required')
    return a.astype(np.int8, copy=True)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _inside(shape, point):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _add(point, direction, amount=1):
    dr, dc = DIRECTIONS[direction]
    return point[0] + amount * dr, point[1] + amount * dc


def _arrow(tip, direction):
    # Three-cell chevron: the two wings are one cell behind and to either side.
    behind = _add(tip, direction, -1)
    return (tip, _add(behind, (direction + 1) % 4),
            _add(behind, (direction - 1) % 4))


def _cap_cells(cap, axis):
    return (cap, _add(cap, (axis + 1) % 4), _add(cap, (axis - 1) % 4))


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            point = stack.pop()
            cells.append(point)
            for d in range(4):
                q = _add(point, d)
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _read(input_grid):
    a = _grid(input_grid)
    structure = int(a[0, 0])
    _require(np.all(a[0] == structure), 'missing key top frame')
    bottoms = np.flatnonzero(np.all(a[1:] == structure, axis=1)) + 1
    _require(len(bottoms) > 0, 'missing key bottom frame')
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(a) - 1, 'bad key height')
    band = a[1:bottom]
    _require(np.all(band == band[0]), 'inconsistent key rows')
    row = band[0]
    _require(row[0] == row[-1] == structure, 'key side walls')
    runs, c = [], 0
    while c < len(row):
        if row[c] == structure:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != structure:
            c += 1
        runs.append((start, c))
    _require(2 <= len(runs) <= 4 and all(b - x >= 3 for x, b in runs),
             'key panels must have clear interiors')
    background = int(row[runs[0][0] + 1])
    turns = {}
    for left, right in runs:
        _require(np.all(row[left + 1:right - 1] == background), 'key interior')
        x, y = int(row[left]), int(row[right - 1])
        _require((x == background) != (y == background), 'ambiguous bar')
        color, turn = (y, 1) if x == background else (x, -1)
        _require(color not in turns and color not in (structure, background),
                 'conflicting key color')
        turns[color] = turn
    field = a[bottom + 1:].copy()
    caps, registers = set(), []
    reconstruction = np.full(field.shape, background, dtype=np.int8)
    for cells in _components(field == structure):
        _require(len(cells) == 3, 'each cap must be a three-cell bar')
        rs, cs = zip(*cells)
        if len(set(rs)) == 1 and max(cs) - min(cs) == 2:
            cap, axes = (rs[0], min(cs) + 1), (0, 2)
        elif len(set(cs)) == 1 and max(rs) - min(rs) == 2:
            cap, axes = (min(rs) + 1, cs[0]), (1, 3)
        else:
            raise ValueError('cap is not a straight bar')
        candidates = [d for d in axes if _inside(field.shape, _add(cap, d))
                      and int(field[_add(cap, d)]) in turns]
        _require(len(candidates) == 1, 'cap must have one attached pin')
        axis = candidates[0]
        seq, pos = [], _add(cap, axis)
        while _inside(field.shape, pos) and int(field[pos]) in turns:
            seq.append(int(field[pos]))
            pos = _add(pos, axis)
        _require(len(seq) >= 2, 'input pins must have at least two cells')
        reg = dict(cap=cap, axis=axis, seq=tuple(seq))
        registers.append(reg)
        for p in cells:
            _require(reconstruction[p] == background, 'overlapping caps')
            reconstruction[p] = structure
            caps.add(p)
        for j, color in enumerate(seq, 1):
            p = _add(cap, axis, j)
            _require(reconstruction[p] == background, 'overlapping registers')
            reconstruction[p] = color
    _require(1 <= len(registers) <= 7, 'one through seven registers required')
    unkeyed = set(map(int, np.unique(field))) - set(turns) - {structure, background}
    _require(len(unkeyed) == 1, 'one unkeyed arrow color required')
    source = unkeyed.pop()
    source_cells = set(map(tuple, np.argwhere(field == source).tolist()))
    candidates = [(p, d) for p in source_cells for d in range(4)
                  if set(_arrow(p, d)) == source_cells]
    _require(len(candidates) == 1, 'arrow orientation is ambiguous')
    tip, direction = candidates[0]
    for p in source_cells:
        _require(reconstruction[p] == background, 'source overlaps a register')
        reconstruction[p] = source
    _require(np.array_equal(reconstruction, field), 'extra or malformed field cells')
    # Registers cannot touch: their ownership and individual sequences are visible.
    owners = {}
    for i, reg in enumerate(registers):
        cells = (*_cap_cells(reg['cap'], reg['axis']),
                 *(_add(reg['cap'], reg['axis'], j)
                   for j in range(1, len(reg['seq']) + 1)))
        for p in cells:
            owners[p] = i
    for p, owner in owners.items():
        for d in range(4):
            q = _add(p, d)
            _require(q not in owners or owners[q] == owner, 'registers touch')
            _require(q not in source_cells, 'arrow touches a register')
    return dict(field=field, structure=structure, background=background,
                source=source, tip=tip, direction=direction, turns=turns,
                registers=registers, caps=caps, crop=bottom + 1)


def _run(info, readout='exposed', turn_override=None, push=True):
    """Cell simulation, reading current registers on each encounter.

    Optional switches are explicit counterfactuals for authored checks. solve
    always uses exposed readout, pressure and the visible relative-turn table.
    """
    field = info['field']
    output = field.copy()
    seqs = [list(r['seq']) for r in info['registers']]
    position, direction = info['tip'], info['direction']
    ray = np.zeros(field.shape, dtype=bool)
    ray[position] = True
    segments, events, seen = [], [], set()
    origin, length = position, 0
    max_steps = 4 * field.size * (1 + sum(map(len, seqs)))
    for _ in range(max_steps):
        state = (position, direction, tuple(map(tuple, seqs)))
        _require(state not in seen, 'cyclic trajectory')
        seen.add(state)
        q = _add(position, direction)
        if not _inside(field.shape, q) or q in info['caps']:
            reason = 'edge' if not _inside(field.shape, q) else 'cap'
            segments.append((origin, direction, length, reason))
            break
        hit = None
        for i, reg in enumerate(info['registers']):
            for j in range(1, len(seqs[i]) + 1):
                if q == _add(reg['cap'], reg['axis'], j):
                    hit = (i, j)
                    break
            if hit is not None:
                break
        if hit is None:
            output[q] = info['source']
            ray[q] = True
            position, length = q, length + 1
            continue
        i, distance = hit
        reg, before = info['registers'][i], tuple(seqs[i])
        if direction % 2 == reg['axis'] % 2:
            segments.append((origin, direction, length, 'pin_end'))
            break
        capacity = distance - 1
        _require(capacity >= 1, 'side collision would empty the register')
        shift = len(before) - capacity
        _require(shift >= 1, 'side collision must overlap the pin')
        after = before[shift:] if push else before
        if push:
            for j in range(1, len(before) + 1):
                output[_add(reg['cap'], reg['axis'], j)] = info['background']
            for j, color in enumerate(after, 1):
                output[_add(reg['cap'], reg['axis'], j)] = color
            seqs[i] = list(after)
        color = after[0]
        if readout == 'original_root':
            color = before[0]
        elif readout == 'contact':
            color = before[distance - 1]
        turn = info['turns'][color]
        if turn_override == 'stop':
            new_direction = None
        elif turn_override == 'absolute':
            new_direction = 3 if turn == -1 else 1
        elif turn_override == 'reverse':
            new_direction = (direction - turn) % 4
        else:
            new_direction = (direction + turn) % 4
        events.append(dict(register=i, strike=q, incoming=direction,
                           outgoing=new_direction, distance=distance,
                           capacity=capacity, shift=shift, before=before,
                           after=after, selected=color))
        segments.append((origin, direction, length, 'actuation'))
        if new_direction is None:
            break
        direction, origin, length = new_direction, position, 0
    else:
        raise ValueError('trajectory did not terminate')
    return output, dict(events=events, segments=segments, ray=ray,
                        final_sequences=tuple(map(tuple, seqs)))


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _run(_read(input_grid))[0]


def _render_field(shape, tip, direction, registers):
    field = np.zeros(shape, dtype=np.int8)
    for p in _arrow(tip, direction):
        field[p] = 1
    for reg in registers:
        for p in _cap_cells(reg['cap'], reg['axis']):
            field[p] = 8
        for j, color in enumerate(reg['seq'], 1):
            field[_add(reg['cap'], reg['axis'], j)] = color
    return field


def _make_header(rng, width, turns, height):
    order = list(map(int, rng.permutation(list(turns))))
    widths = np.full(len(order), 5, dtype=int)
    widths += rng.multinomial(width - int(widths.sum()),
                             np.full(len(order), 1 / len(order)))
    header = np.zeros((height, width), dtype=np.int8)
    header[[0, -1]] = 8
    start = 0
    for color, size in zip(order, widths):
        header[:, start] = header[:, start + size - 1] = 8
        col = start + 1 if turns[color] == -1 else start + size - 2
        header[1:-1, col] = color
        start += size
    return header


def _info(field, tip, direction, registers, turns):
    return dict(field=field, background=0, source=1, structure=8,
                tip=tip, direction=direction, registers=registers, turns=turns,
                caps=set(p for r in registers for p in _cap_cells(r['cap'], r['axis'])))


def _clear_register(field, cap, axis, length):
    cells = (*_cap_cells(cap, axis), *(_add(cap, axis, j) for j in range(1, length + 1)))
    if not all(_inside(field.shape, p) for p in cells):
        return False
    # Keep a cell of separation even diagonally, making caps easy to segment.
    for r, c in cells:
        if np.any(field[max(0, r - 1):r + 2, max(0, c - 1):c + 2] != 0):
            return False
    return True


def _grow(rng, turns, direction, target, length_range, witness, last_toward):
    """Grow a circuit on its actual final free segment, validating every insertion."""
    for scene_attempt in range(45):
        h = int(rng.integers(max(17, length_range[1] + 9), 28))
        w = int(rng.integers(max(20, 5 * len(turns)), 31))
        r, c = int(rng.integers(4, h - 4)), int(rng.integers(4, w - 4))
        if direction == 0:
            r = int(rng.integers(h // 2, h - 3))
        elif direction == 1:
            c = int(rng.integers(2, w // 2))
        elif direction == 2:
            r = int(rng.integers(2, h // 2))
        else:
            c = int(rng.integers(w // 2, w - 3))
        tip, registers = (r, c), []
        field = _render_field((h, w), tip, direction, registers)
        trace = _run(_info(field, tip, direction, registers, turns))[1]
        for gate in range(target):
            success = False
            origin, heading, travel, _ = trace['segments'][-1]
            if travel < 3:
                break
            for attempt in range(120):
                # A register belongs to the forward route, not a radial template.
                along = int(rng.integers(2, travel + 1))
                strike = _add(origin, heading, along)
                pin_length = int(rng.integers(length_range[0], length_range[1] + 1))
                distance = int(rng.integers(2, pin_length + 1))
                shift = pin_length - distance + 1
                teaching = witness and gate == 0
                if teaching and (distance < 3 or shift == distance - 1):
                    continue
                selected = 2 if teaching else int(rng.choice(list(turns)))
                turn = turns[selected]
                outgoing = (heading + turn) % 4
                toward = gate == target - 1 and last_toward
                axis = (outgoing + 2) % 4 if toward else outgoing
                cap = _add(strike, axis, -distance)
                if not _clear_register(field, cap, axis, pin_length):
                    continue
                seq = list(map(int, rng.choice(list(turns), pin_length)))
                if teaching:
                    opposite = [color for color, value in turns.items() if value != turn]
                    other = int(rng.choice(opposite))
                    seq[0] = seq[distance - 1] = other
                    seq[-1] = other  # at least two distinct surviving colors
                seq[shift] = selected
                reg = dict(cap=cap, axis=axis, seq=tuple(seq))
                candidate = registers + [reg]
                grid = _render_field(field.shape, tip, direction, candidate)
                try:
                    new_trace = _run(_info(grid, tip, direction, candidate, turns))[1]
                except ValueError:
                    continue
                old_ids = [e['register'] for e in trace['events']]
                new_ids = [e['register'] for e in new_trace['events']]
                if new_ids != old_ids + [len(registers)]:
                    continue
                registers, field, trace, success = candidate, grid, new_trace, True
                break
            if not success:
                break
        if len(registers) != target:
            continue
        # Optional genuinely unused registers, not simply unused key entries.
        for _ in range(int(rng.integers(0, min(2, 7 - target) + 1))):
            for attempt in range(80):
                axis = int(rng.integers(4))
                cap = (int(rng.integers(h)), int(rng.integers(w)))
                n = int(rng.integers(2, length_range[1] + 1))
                if not _clear_register(field, cap, axis, n):
                    continue
                seq = tuple(map(int, rng.choice(list(turns), n)))
                candidate = registers + [dict(cap=cap, axis=axis, seq=seq)]
                grid = _render_field(field.shape, tip, direction, candidate)
                try:
                    updated = _run(_info(grid, tip, direction, candidate, turns))[1]
                except ValueError:
                    continue
                if not np.array_equal(updated['ray'], trace['ray']):
                    continue
                if updated['events'] != trace['events']:
                    continue
                registers, field, trace = candidate, grid, updated
                break
        return field, _run(_info(field, tip, direction, registers, turns))[0]
    raise RuntimeError('bounded circuit construction exhausted')


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10),
                                ('num_test', num_test, 1, 4)):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and lo <= value <= hi, name + ' out of range')
    _require(num_train > num_test, 'num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, 'seed must be a nonnegative integer')
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    mode = str(rng.choice(('depth', 'capacity', 'orientation')))
    base_direction = int(rng.integers(4))
    palette = rng.permutation(10)
    initial_turn = int(rng.choice((-1, 1)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            witness = split == 'train' and index < 2
            if witness:
                t = initial_turn if index == 0 else -initial_turn
                turns = {2: t, 3: -t}
                # Recurring literal colors change their key meanings visibly.
                local_palette = palette
            else:
                n = int(rng.integers(2, 5))
                turns = {color: int(rng.choice((-1, 1))) for color in range(2, 2 + n)}
                local_palette = palette if rng.random() < .2 else rng.permutation(10)
            direction = base_direction
            if mode == 'orientation':
                direction = (base_direction + (2 * (index % 2) if split == 'train'
                                               else 1 + 2 * (index % 2))) % 4
            else:
                direction = (base_direction + index) % 4
            target = 2 if witness else int(rng.integers(1, 3 if mode == 'depth' else 5))
            if mode == 'depth' and split == 'test':
                target = int(rng.integers(3, 6))
            lengths = (4, 6)
            if mode == 'capacity' and split == 'test':
                lengths = (7, 9)
            toward = (index == 1) if witness else bool(rng.random() < .3)
            for attempt in range(24):
                field, answer = _grow(rng, turns, direction, target, lengths, witness, toward)
                hh = int(rng.integers(3, min(5, 31 - field.shape[0])))
                header = _make_header(rng, field.shape[1], turns, hh)
                pair = {'input': local_palette[np.vstack((header, field))].astype(np.int8),
                        'output': local_palette[answer].astype(np.int8)}
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('unique input construction exhausted')
        if split == 'train':
            rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _counterfactual_differs(info, actual, **options):
    try:
        other, _ = _run(info, **options)
    except ValueError:
        return True
    return not np.array_equal(other, actual)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists required')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_info = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(a)
            _require(sig not in seen, 'duplicate input')
            seen.add(sig)
            info = _read(a)
            expected, trace = _run(info)
            _require(b.shape == info['field'].shape and np.array_equal(b, expected), 'incorrect solution')
            _require(np.all(b[info['field'] == info['structure']] == info['structure']), 'caps changed')
            for p in _arrow(info['tip'], info['direction']):
                _require(b[p] == info['source'], 'source changed')
            for event in trace['events']:
                _require(event['after'] == event['before'][event['shift']:], 'sequence order changed')
                _require(len(event['after']) == event['capacity'] and event['shift'] >= 1,
                         'wrong contact/clipping capacity')
                _require(event['outgoing'] == (event['incoming'] + info['turns'][event['after'][0]]) % 4,
                         'turn does not use exposed color')
            for reg, seq in zip(info['registers'], trace['final_sequences']):
                for j, color in enumerate(seq, 1):
                    _require(int(b[_add(reg['cap'], reg['axis'], j)]) == color, 'surviving pin changed')
            if split == 'train':
                train_info.append((info, trace, b))
    events = [e for _, trace, _ in train_info for e in trace['events']]
    _require(any(len(t['events']) >= 2 for _, t, _ in train_info), 'recursion undemonstrated')
    _require(any(len(set(e['after'])) >= 2 for e in events), 'ordered multicolor survivors undemonstrated')
    sides = {info['turns'][e['after'][0]] for info, t, _ in train_info for e in t['events']}
    _require(sides == {-1, 1}, 'both relative turns must be demonstrated')
    _require(any(info['turns'][e['after'][0]] != info['turns'][e['before'][0]]
                 and info['turns'][e['after'][0]] != info['turns'][e['before'][e['distance'] - 1]]
                 for info, t, _ in train_info for e in t['events']),
             'pressure must change the routing instruction')
    for options in ({'readout': 'original_root'}, {'readout': 'contact'},
                    {'push': False, 'readout': 'original_root'}, {'turn_override': 'stop'},
                    {'turn_override': 'reverse'}, {'turn_override': 'absolute'}):
        _require(any(_counterfactual_differs(info, b, **options) for info, _, b in train_info),
                 'training fails to distinguish shortcut: ' + str(options))


def _fixed_regressions():
    # Five cells, side contact at depth three: clip three cells and retain (2,3).
    regs = [dict(cap=(2, 9), axis=2, seq=(3, 2, 3, 2, 3))]
    field = _render_field((12, 16), (5, 3), 1, regs)
    turns = {2: -1, 3: 1}
    info = _info(field, (5, 3), 1, regs, turns)
    actual, t = _run(info)
    expected = field.copy()
    expected[3:8, 9] = 0
    expected[3, 9], expected[4, 9] = 2, 3
    expected[5, 4:9] = 1
    expected[3:5, 8] = 1  # left turn toward cap; stops at row-two cap.
    assert np.array_equal(actual, expected)
    assert t['events'][0]['shift'] == 3 and t['events'][0]['selected'] == 2
    # A mirrored instruction sends the same mechanical result away to the edge.
    info['turns'] = {2: 1, 3: -1}
    actual, t = _run(info)
    expected[3:5, 8] = 0
    expected[6:, 8] = 1
    assert np.array_equal(actual, expected)
    # End-on impact must preserve the complete pin and stop the ray.
    end_field = _render_field((12, 16), (10, 9), 0, regs)
    end_info = _info(end_field, (10, 9), 0, regs, turns)
    result, t = _run(end_info)
    assert not t['events'] and np.array_equal(result[3:8, 9], [3, 2, 3, 2, 3])
    # Round-trip parser and rotations with unchanged relative key meanings.
    header = _make_header(np.random.default_rng(0), 16, turns, 3)
    a = np.vstack((header, field))
    assert np.array_equal(solve(a), _run(_info(field, (5, 3), 1, regs, turns))[0])
    for rotation in range(4):
        f = np.rot90(field, rotation).copy()
        hdr = _make_header(np.random.default_rng(rotation), f.shape[1], turns, 3)
        assert np.array_equal(solve(np.vstack((hdr, f))),
                              np.rot90(_run(_info(field, (5, 3), 1, regs, turns))[0], rotation))
    mapping = np.array([7, 0, 8, 4, 1, 3, 2, 6, 9, 5])
    assert np.array_equal(solve(mapping[a]), mapping[solve(a)])
    # Reflecting the complete input also swaps the key's left/right bars.
    assert np.array_equal(solve(np.fliplr(a)), np.fliplr(solve(a)))
    malformed = a.copy()
    malformed[-1, -1] = 2
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError('detached key-colored noise was accepted')


def self_test():
    """Bounded authored checks and measured diversity over the 42 host seeds."""
    _fixed_regressions()
    shapes, layouts, palettes, lengths, capacities, axes, headings, key_counts = (set() for _ in range(8))
    signatures, turn_flips, untouched, toward, max_events = set(), 0, 0, 0, 0
    regimes = {name: 0 for name in ('depth', 'capacity', 'orientation')}
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        repeat = generate_puzzle(seed)
        sig = tuple(_signature(q[n]) for s in ('train', 'test') for q in p[s] for n in ('input', 'output'))
        assert sig == tuple(_signature(q[n]) for s in ('train', 'test') for q in repeat[s] for n in ('input', 'output'))
        signatures.add(sig)
        tables = []
        by_split = {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in p[split]:
                a, b = pair['input'], pair['output']
                assert np.array_equal(solve(a.copy()), b)
                info, trace = _read(a), _run(_read(a))[1]
                by_split[split].append((info, trace))
                tables.append(info['turns'])
                shapes.add(info['field'].shape)
                palettes.add((info['background'], info['structure'], info['source']))
                headings.add(info['direction'])
                key_counts.add(len(info['turns']))
                layouts.add((info['field'].shape, info['tip'], info['direction'],
                             tuple((r['cap'], r['axis'], len(r['seq'])) for r in info['registers'])))
                for reg in info['registers']:
                    lengths.add(len(reg['seq']))
                    axes.add(reg['axis'])
                hit = {e['register'] for e in trace['events']}
                untouched += len(info['registers']) - len(hit)
                max_events = max(max_events, len(trace['events']))
                for e in trace['events']:
                    capacities.add(e['capacity'])
                    reg = info['registers'][e['register']]
                    toward += e['outgoing'] == (reg['axis'] + 2) % 4
        turn_flips += any(len({table[c] for table in tables if c in table}) > 1 for c in range(10))
        # Generation schedules are checked separately from semantic validity.
        # This seed is used only by self-test to identify the sampled schedule.
        mode = str(np.random.default_rng(seed).choice(('depth', 'capacity', 'orientation')))
        regimes[mode] += 1
        if mode == 'depth':
            assert all(1 <= len(t['events']) <= 2 for _, t in by_split['train'])
            assert all(3 <= len(t['events']) <= 5 for _, t in by_split['test'])
        elif mode == 'capacity':
            for split, allowed in (('train', range(4, 7)), ('test', range(7, 10))):
                assert all(len(info['registers'][e['register']]['seq']) in allowed
                           for info, trace in by_split[split] for e in trace['events'])
        else:
            train_d = {info['direction'] for info, _ in by_split['train']}
            test_d = {info['direction'] for info, _ in by_split['test']}
            assert len(train_d) == 2 and train_d.isdisjoint(test_d)
    assert len(signatures) == 42 and len(shapes) >= 35 and len(layouts) >= 150
    assert len(palettes) >= 30 and headings == axes == set(range(4))
    assert key_counts == {2, 3, 4} and lengths.issuperset({2, 3, 4, 5, 6, 7, 8, 9})
    assert len(capacities) >= 4 and max_events >= 4 and untouched >= 20 and toward >= 42
    assert turn_flips == 42
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(17, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=210, field_shapes=len(shapes), layouts=len(layouts),
                role_palettes=len(palettes), pin_lengths=sorted(lengths),
                surviving_capacities=sorted(capacities), max_actuations=max_events,
                untouched_registers=untouched, turns_toward_cap=toward,
                puzzles_with_instruction_color_flip=turn_flips,
                transfer_regimes=regimes,
                verification='authored checks; no independent or blind solver')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
