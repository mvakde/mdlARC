"""Parallel arithmetic progressions with phase-sensitive terminal recoloring.

Colors have local roles; geometry alone defines the operation. Distribution
profiles affect difficulty, never the answer. Labels are painted from sampled
geometry and then checked by a separate input-only reconstruction.
"""

import numpy as np


# Exact source contents are regression fixtures, never generation templates.
_SOURCE_ROWS = ((('000000000',
   '100100000',
   '000000000',
   '000000000',
   '110000000',
   '000000000',
   '000000000',
   '000000000',
   '909000600',
   '000000000',
   '000000000',
   '880000030'),
  ('000000000',
   '100100100',
   '000000000',
   '000000000',
   '111111111',
   '000000000',
   '000000000',
   '000000000',
   '606060600',
   '000000000',
   '000000000',
   '333333330')),
 (('00000000000000000000',
   '00000000000000000000',
   '00400000000000000000',
   '00000000000000000050',
   '00000000000000010000',
   '00300000000000000000',
   '00000000000000000090',
   '00000000030000000000',
   '00300000030000010090'),
  ('00000000030000010090',
   '00000000030000000000',
   '00400000030000000090',
   '00000000030000000050',
   '00000000030000010090',
   '00400000030000000000',
   '00000000030000000090',
   '00000000030000000000',
   '00400000030000010090')),
 (('000000000000000',
   '000000000000000',
   '000000000000000',
   '000000000000020',
   '020000000000000',
   '000002000600000',
   '020002000600020'),
  ('020002000600020',
   '000002000600000',
   '020002000600000',
   '000002000600020',
   '020002000600000',
   '000002000600000',
   '020002000600020')),
 (('000000000000000000000',
   '000000000700000002002',
   '000000900000000000055',
   '000600000000000000022',
   '000000000000000000000',
   '000000000000010600006',
   '000000000000000000000',
   '000000000000000008008',
   '000000040000000000707'),
  ('000000000000000000000',
   '002002002702002002002',
   '000000999999999999999',
   '000666666666666666666',
   '000000000000000000000',
   '600006000060010600006',
   '000000000000000000000',
   '008008008008008008008',
   '707070747070707070707')),
 (('020030040607070902070',
   '020000000007000002000',
   '000030000000070000000',
   '000000040000000900070',
   '000020000600000000000',
   '000000000000000000000',
   '040000000300060000090',
   '000000000000000000000',
   '000000000000000000000'),
  ('040020040607060902090',
   '040000000007000002000',
   '040020000007060002000',
   '040000040007000902090',
   '040020000607060002000',
   '040000000007000002000',
   '040000040307060902090',
   '000000000007000002000',
   '000000000607000002000')))


def _source_pairs():
    return [{"input": np.array([list(map(int, r)) for r in a], dtype=np.int8),
             "output": np.array([list(map(int, r)) for r in b], dtype=np.int8)}
            for a, b in _SOURCE_ROWS]


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if counts.max() <= grid.size // 2:
        raise ValueError("input needs a distinguishable majority background")
    return int(colors[np.argmax(counts)])


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _nonbackground_colors(grid: np.ndarray) -> frozenset[int]:
    return frozenset(
        int(value)
        for value in np.unique(grid)
        if int(value) != _background(grid)
    )


def _parse_signal_line(values: np.ndarray, background: int) -> dict | None:
    positions = np.flatnonzero(values != background)
    if len(positions) < 2:
        return None

    counts: dict[int, int] = {}
    for position in positions:
        color = int(values[position])
        counts[color] = counts.get(color, 0) + 1

    repeated_colors = [
        color for color, count in counts.items()
        if count >= 2
    ]
    if len(repeated_colors) != 1:
        return None

    reference_color = repeated_colors[0]
    if counts[reference_color] != 2:
        return None

    references = [
        int(position)
        for position in positions
        if int(values[position]) == reference_color
    ]
    a, b = references
    period = b - a

    aligned_extras = [
        int(position)
        for position in positions
        if int(position) not in (a, b)
        and (int(position) - a) % period == 0
    ]
    if len(aligned_extras) > 1:
        return None

    terminal = aligned_extras[0] if aligned_extras else None
    if terminal is not None and a < terminal < b:
        return None

    distractors = [
        int(position)
        for position in positions
        if int(position) not in (a, b)
        and int(position) != terminal
    ]
    if any((position - a) % period == 0 for position in distractors):
        return None

    return {
        "a": a,
        "b": b,
        "period": period,
        "reference_color": reference_color,
        "terminal": terminal,
        "terminal_color": (
            None if terminal is None else int(values[terminal])
        ),
        "distractors": tuple(distractors),
    }


def _collect_signals(
    grid: np.ndarray,
    orientation: str,
) -> list[dict] | None:
    signals: list[dict] = []
    background = _background(grid)

    if orientation == "H":
        active_lines = [
            row for row in range(grid.shape[0])
            if np.any(grid[row] != background)
        ]
        for row in active_lines:
            signal = _parse_signal_line(grid[row], background)
            if signal is None:
                return None
            signal["line"] = row
            signals.append(signal)
    else:
        active_lines = [
            column for column in range(grid.shape[1])
            if np.any(grid[:, column] != background)
        ]
        for column in active_lines:
            signal = _parse_signal_line(grid[:, column], background)
            if signal is None:
                return None
            signal["line"] = column
            signals.append(signal)

    return signals if signals else None


def _infer_signals(grid: np.ndarray) -> tuple[str, list[dict]]:
    candidates: list[tuple[str, list[dict]]] = []

    for orientation in ("H", "V"):
        signals = _collect_signals(grid, orientation)
        if signals is not None:
            candidates.append((orientation, signals))

    if len(candidates) != 1:
        raise ValueError(
            "the active axis or reference-pair association is ambiguous"
        )

    return candidates[0]


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    orientation, signals = _infer_signals(grid)
    result = grid.copy()
    axis_length = (
        grid.shape[1] if orientation == "H"
        else grid.shape[0]
    )

    for signal in signals:
        a = signal["a"]
        b = signal["b"]
        period = signal["period"]
        terminal = signal["terminal"]

        if terminal is None:
            coordinates = [
                position
                for position in range(axis_length)
                if (position - a) % period == 0
            ]
            paint_color = signal["reference_color"]
        else:
            low = min(a, terminal)
            high = max(b, terminal)
            coordinates = [
                position
                for position in range(low, high + 1)
                if (position - a) % period == 0
            ]
            paint_color = signal["terminal_color"]

        for position in coordinates:
            if orientation == "H":
                result[signal["line"], position] = paint_color
            else:
                result[position, signal["line"]] = paint_color

    return result


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)

    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )

    return value


def _choose_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_optional_count(
        "num_train",
        num_train,
        2,
        10,
    )
    num_test = _validate_optional_count(
        "num_test",
        num_test,
        1,
        4,
    )

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    elif num_train is None:
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    elif num_test is None:
        num_test = int(
            rng.integers(
                1,
                min(4, num_train - 1) + 1,
            )
        )

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


# Geometry options use the entire feasible extent, not a puzzle-wide period cap.
def _geometries(length, period, terminal, direction, stopping=False):
    result = []
    for a in range(length - period):
        b = a + period
        phase = list(range(a % period, length, period))
        if len(phase) < 3:
            continue
        if not terminal:
            result.append((a, b, None))
            continue
        for t in phase:
            if (direction < 0 and t < a) or (direction > 0 and t > b):
                low, high = min(a, t), max(b, t)
                if not stopping or any(x < low or x > high for x in phase):
                    result.append((a, b, t))
    return result


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _build_pair(rng, orientation, regime, settings, witness=None, rescue=False):
    # Size, density, spacing, mode balance and noise are separate choices.
    scale = _pick(rng, settings['scales'])
    low, high = ((5, 15), (8, 24), (12, 30))[scale]
    length = int(rng.integers(low, high + 1))
    if witness == 'boundary':
        length = max(7, length)
    cross = int(rng.integers(3, high + 1))
    if rng.random() < .25:
        cross = length
    if rescue:
        length = max(length, 12)
    density = _pick(rng, ('few', 'medium', 'dense'))
    maximum = min(14, cross)
    if density == 'few':
        count = int(rng.integers(1, min(4, maximum) + 1))
    elif density == 'medium':
        count = int(rng.integers(min(3, maximum), min(8, maximum) + 1))
    else:
        count = int(rng.integers(max(1, maximum - 4), maximum + 1))
    if settings['transfer'] == 'source' and regime == 'target_test':
        count = max(count, min(cross, int(rng.integers(6, 13))))
    if rescue:
        cross = max(cross, count + 2)
    lines = [int(x) for x in rng.choice(cross, count, replace=False)]

    mode = _pick(rng, ('boundary', 'terminal', 'mixed', 'mixed'))
    terminals = 0 if mode == 'boundary' else count if mode == 'terminal' else int(rng.integers(count + 1))
    flags = [True] * terminals + [False] * (count - terminals)
    rng.shuffle(flags)
    if witness == 'stop':
        flags[0] = True
    elif witness == 'boundary':
        flags[0] = False

    constant = settings['constant_palette']
    background = settings['background'] if constant else int(rng.integers(10))
    if constant:
        palette = settings['palette']
    else:
        minimum = 2 if any(flags) or (witness == 'boundary' and settings['noise']) else 1
        size = int(_pick(rng, (minimum, minimum, max(minimum, 3), int(rng.integers(minimum, 10)))))
        palette = [int(x) for x in rng.choice([c for c in range(10) if c != background], size, replace=False)]
    # Small palettes intentionally identify several signals with the same hue.
    refs = [int(_pick(rng, palette)) for _ in range(count)]
    if count > 1 and rng.random() < .7:
        refs[1] = refs[0]
    grid = np.full((cross, length), background, dtype=np.int8)
    output = grid.copy()
    noise_level = _pick(rng, (0., 0., .3, .6, 1.)) if settings['noise'] else 0.
    adjacent = _pick(rng, ('none', 'some', 'many'))
    used = set()

    for index, (line, has_terminal, reference) in enumerate(zip(lines, flags, refs)):
        direction = int(_pick(rng, (-1, 1)))
        if settings['transfer'] == 'source':
            direction = 1 if orientation == 'H' else -1
            if regime == 'target_test':
                direction *= -1
        stopping = witness == 'stop' and index == 0
        if stopping:
            period = 1  # One training witness teaches solid as well as dotted completion.
        elif witness == 'boundary' and index == 0:
            period = int(rng.integers(2, (length - 1) // 3 + 1))
        elif adjacent == 'many' and rng.random() < .65:
            period = 1
        else:
            minimum = 2 if adjacent == 'none' else 1
            period = int(rng.integers(minimum, (length - 1) // 2 + 1))
        candidates = _geometries(length, period, has_terminal, direction, stopping)
        if witness == 'boundary' and index == 0:
            # Teach extrapolation on both sides, so a two-demo puzzle cannot
            # accidentally support a forward-only or backward-only rule.
            candidates = [g for g in candidates if g[0] >= period and g[1] + period < length]
        if rescue:
            candidates = [g for g in candidates if (0 in g) == (index == 0)]
        if not candidates:
            return None
        if settings['transfer'] == 'source' and regime == 'target_test':
            packed = [g for g in candidates if (g[0] >= length // 2 if orientation == 'H' else g[1] <= length // 2)]
            if packed:
                candidates = packed
        # First choose distance, then placement, so long terminals are not
        # crowded out by the many possible placements of short spans.
        if has_terminal:
            distances = sorted({min(abs(g[2]-g[0]), abs(g[2]-g[1])) // period for g in candidates})
            distance = _pick(rng, distances)
            candidates = [g for g in candidates if min(abs(g[2]-g[0]), abs(g[2]-g[1])) // period == distance]
        placement = _pick(rng, ('left', 'interior', 'right', 'any'))
        preferred = [g for g in candidates if
                     (placement == 'left' and g[0] < length / 3) or
                     (placement == 'right' and g[1] >= 2 * length / 3) or
                     (placement == 'interior' and g[0] >= length / 3 and g[1] < 2 * length / 3)]
        a, b, terminal = _pick(rng, preferred or candidates)
        row = grid[line]
        row[a] = row[b] = reference
        used.add(reference)
        available = [c for c in palette if c != reference]
        rng.shuffle(available)
        # For constant palettes only, fill missing color roles opportunistically.
        if constant:
            available.sort(key=lambda c: c in used)
        if terminal is not None:
            row[terminal] = available.pop(0)
            used.add(int(row[terminal]))
        options = [x for x in range(length) if (x - a) % period and (not rescue or x != 0)]
        noise_count = 0
        if options and available and rng.random() < noise_level:
            noise_count = int(rng.integers(1, min(4, len(options), len(available)) + 1))
        if witness == 'boundary' and index == 0 and settings['noise']:
            noise_count = max(noise_count, 1)
        if constant:
            noise_count = max(noise_count, min(len(set(available) - used), len(options)))
        if noise_count > min(len(options), len(available)):
            return None
        for x, color in zip(rng.choice(options, noise_count, replace=False), available):
            row[int(x)] = color
            used.add(color)
        output[line] = row
        start, end = (a % period, length - 1) if terminal is None else (min(a, terminal), max(b, terminal))
        color = reference if terminal is None else int(row[terminal])
        output[line, start:end + 1:period] = color

    if constant and used != set(palette):
        return None
    if np.count_nonzero(grid == background) <= grid.size // 2:
        return None
    if orientation == 'V':
        grid, output = grid.T.copy(), output.T.copy()
    try:
        axis, signals = _infer_signals(grid)
    except ValueError:
        return None
    if axis != orientation or len(signals) != count:
        return None
    return {'input': grid, 'output': output}


def _training_evidence(analyses):
    return {
        'boundary': any(a['signal_count'] > a['terminal_count'] for a in analyses),
        'terminal': any(a['terminal_count'] for a in analyses),
        'stop': any(a['stopping_witnesses'] for a in analyses),
        'noise': any(a['distractor_count'] for a in analyses),
        'adjacent': any(1 in a['periods'] for a in analyses),
        'dotted': any(max(a['periods']) > 1 for a in analyses),
        'left': any(a['left_extensions'] for a in analyses),
        'right': any(a['right_extensions'] for a in analyses),
    }


def _fallback_pairs(settings, orientation, regime, witness):
    """Finite constructive fallback with a provably unique common axis.

    Coordinate zero belongs only to the first signal, so the perpendicular
    interpretation contains a singleton line and cannot explain the input.
    Enumerating shapes and counts also supplies more than fourteen distinct
    choices for any requested puzzle, without relying on further lucky draws.
    """
    background = settings['background']
    first, second = settings['palette']
    direction = 1
    if settings['transfer'] == 'source':
        direction = 1 if orientation == 'H' else -1
        if regime == 'target_test':
            direction *= -1
    for count in range(3, 15):
        for length in range(20, 31):
            for period in range(2, (length - 2) // 3 + 1):
                grid = np.full((count + 2, length), background, dtype=np.int8)
                output = grid.copy()
                for line in range(count):
                    reference, paint = (first, second) if line % 2 == 0 else (second, first)
                    terminal = line == 1 or (line == 0 and witness == 'stop')
                    if terminal:
                        a, b, t = ((0, 1, 5) if line == 0 else (2, 3, 6)) if direction > 0 else ((4, 5, 0) if line == 0 else (4, 5, 1))
                        grid[line, [a, b]] = reference
                        grid[line, t] = paint
                        output[line, min(a, t):max(b, t) + 1] = paint
                    else:
                        a = 0 if line == 0 else period + 1 if line == count - 1 else 1
                        grid[line, [a, a + period]] = reference
                        output[line, a % period::period] = reference
                        if settings['noise'] and witness == 'boundary' and line == 0:
                            grid[line, 1] = output[line, 1] = paint
                if orientation == 'V':
                    grid, output = grid.T.copy(), output.T.copy()
                yield {'input': grid, 'output': output}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    # No hidden answer-affecting convention is sampled. These are presentation
    # and transfer regimes, with the same visible geometric rule throughout.
    profile = int(rng.integers(3))
    background = int(rng.integers(10))
    settings = {
        'scales': ((0, 0, 1, 2), (0, 1, 1, 2), (0, 1, 2, 2))[profile],
        'noise': bool(rng.random() < .75),
        'constant_palette': bool(rng.random() < .2),
        'background': background,
        'palette': [int(c) for c in rng.choice([c for c in range(10) if c != background], 2, replace=False)],
        'transfer': _pick(rng, ('mixed', 'mixed', 'HH', 'VV', 'HV', 'VH', 'source')),
    }
    train_axes = [str(_pick(rng, ('H', 'V'))) for _ in range(num_train)]
    test_axes = [str(_pick(rng, ('H', 'V'))) for _ in range(num_test)]
    transfer = settings['transfer']
    if transfer in ('HH', 'VV', 'HV', 'VH'):
        train_axes = [transfer[0]] * num_train
        test_axes = [transfer[1]] * num_test
    elif transfer == 'source':
        train_axes[:2] = ['H', 'V']
        rng.shuffle(train_axes)
    witnesses = [None] * num_train
    first, second = [int(x) for x in rng.choice(num_train, 2, replace=False)]
    witnesses[first], witnesses[second] = 'stop', 'boundary'
    puzzle = {'train': [], 'test': []}
    seen = set()
    source_keys = {(_grid_key(p['input']), _grid_key(p['output'])) for p in _source_pairs()}
    for split, axes in (('train', train_axes), ('test', test_axes)):
        regime = 'source_train' if split == 'train' else 'target_test'
        for index, axis in enumerate(axes):
            witness = witnesses[index] if split == 'train' else None
            pair_rng = np.random.default_rng(int(rng.integers(2**63)))
            def candidates():
                for attempt in range(40):
                    yield _build_pair(pair_rng, axis, regime, settings, witness, rescue=attempt >= 30)
                yield from _fallback_pairs(settings, axis, regime, witness)

            for pair in candidates():
                if pair is None:
                    continue
                key = _grid_key(pair['input'])
                if key in seen or (key, _grid_key(pair['output'])) in source_keys:
                    continue
                if split == 'train' and len(puzzle['train']) == num_train - 1:
                    analyses = [_analyze_pair(p) for p in puzzle['train'] + [pair]]
                    # Avoid accidental within-puzzle collapse without requiring
                    # every individual pair to exhibit all forms of completion.
                    if len({a['signal_count'] for a in analyses}) == 1:
                        continue
                    if len({max(a['periods']) for a in analyses}) == 1:
                        continue
                break
            else:
                raise RuntimeError('bounded construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    grid, output = pair['input'], pair['output']
    try:
        orientation, signals = _infer_signals(grid)
        expected = _solve_grid(grid)
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    assert np.array_equal(expected, output), 'incorrect phase completion or recoloring'
    length = grid.shape[1] if orientation == 'H' else grid.shape[0]
    stop = 0
    left = right = 0
    for s in signals:
        phase = range(s['a'] % s['period'], length, s['period'])
        if s['terminal'] is None:
            assert len(phase) >= 3, 'boundary signal must add a cell'
            left += phase[0] < s['a']
            right += phase[-1] > s['b']
        else:
            assert s['terminal_color'] not in (_background(grid), s['reference_color'])
            low, high = min(s['a'], s['terminal']), max(s['b'], s['terminal'])
            stop += any(x < low or x > high for x in phase)
    return {'orientation': orientation, 'signals': signals, 'signal_count': len(signals),
            'terminal_count': sum(s['terminal'] is not None for s in signals),
            'distractor_count': sum(len(s['distractors']) for s in signals),
            'stopping_witnesses': stop, 'left_extensions': left, 'right_extensions': right,
            'periods': tuple(s['period'] for s in signals),
            'colors': _nonbackground_colors(grid), 'background': _background(grid)}


def _review_features(pair):
    """Diagnostics reconstructed from grids, with no construction metadata."""
    a = _analyze_pair(pair)
    grid = pair['input']
    rows = grid if a['orientation'] == 'H' else grid.T
    length = rows.shape[1]
    signals = a['signals']
    refs = [s['reference_color'] for s in signals]
    terminals = [s for s in signals if s['terminal'] is not None]
    steps = [min(abs(s['terminal']-s['a']), abs(s['terminal']-s['b'])) // s['period'] for s in terminals]
    directions = sorted({-1 if s['terminal'] < s['a'] else 1 for s in terminals})
    foreground = grid != a['background']
    rr, cc = np.nonzero(foreground)
    locations = set()
    role_colors = {'reference': set(refs),
                   'terminal': {s['terminal_color'] for s in terminals},
                   'distractor': {int(rows[s['line'], x]) for s in signals for x in s['distractors']}}
    for s in signals:
        for x in s['distractors']:
            locations.add('between' if s['a'] < x < s['b'] else 'beyond')
            if s['terminal'] is not None:
                lo, hi = min(s['a'], s['terminal']), max(s['b'], s['terminal'])
                locations.add('inside_span' if lo < x < hi else 'outside_span')
    return {
        'orientation': a['orientation'], 'shape': list(grid.shape),
        'aspect': 'square' if grid.shape[0] == grid.shape[1] else 'portrait' if grid.shape[0] > grid.shape[1] else 'landscape',
        'background': a['background'], 'palette': sorted(int(c) for c in np.unique(grid)),
        'signal_count': len(signals), 'terminal_count': len(terminals),
        'mode': 'boundary' if not terminals else 'terminal' if len(terminals) == len(signals) else 'mixed',
        'periods': sorted(a['periods']), 'max_period': max(a['periods']),
        'adjacent_count': a['periods'].count(1),
        'repeated_periods': len(set(a['periods'])) < len(signals),
        'repeated_references': len(set(refs)) < len(refs),
        'repeated_reference_different_period_mode': any(
            s['reference_color'] == t['reference_color'] and s['period'] != t['period']
            and (s['terminal'] is None) != (t['terminal'] is None)
            for s in signals for t in signals),
        'perpendicular_repeat': any(len(v := [int(c) for c in row if c != a['background']]) != len(set(v)) for row in rows.T),
        'terminal_steps': steps, 'max_terminal_steps': max(steps, default=0),
        'terminal_directions': directions, 'stopping_witnesses': a['stopping_witnesses'],
        'distractors': a['distractor_count'],
        'clean_dotted_signals': sum(s['period'] > 1 and not s['distractors'] for s in signals),
        'clean_and_noisy': 0 < sum(bool(s['distractors']) for s in signals) < len(signals),
        'distractor_locations': sorted(locations),
        'black_roles': sorted(role for role, colors in role_colors.items() if 0 in colors),
        'cross_role_reuse': any(role_colors[r] & role_colors[t] for r in role_colors for t in role_colors if r != t),
        'boundary_phase_gap': any(s['terminal'] is None and (s['a'] % s['period'] != 0 or (length - 1 - s['a']) % s['period'] != 0) for s in signals),
        'reference_locations': sorted({'left' if s['b'] < length / 3 else 'right' if s['a'] >= 2 * length / 3 else 'interior' for s in signals}),
        'margins': [int(rr.min()), int(grid.shape[0]-1-rr.max()), int(cc.min()), int(grid.shape[1]-1-cc.max())],
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    source_keys = {(_grid_key(p['input']), _grid_key(p['output'])) for p in _source_pairs()}
    inputs, pairs = set(), set()
    analyses = {'train': [], 'test': []}
    for split in analyses:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert pair['input'].shape == pair['output'].shape
            assert not np.array_equal(pair['input'], pair['output'])
            key = (_grid_key(pair['input']), _grid_key(pair['output']))
            assert key[0] not in inputs and key not in pairs and key not in source_keys
            inputs.add(key[0])
            pairs.add(key)
            analyses[split].append(_analyze_pair(pair))
    evidence = _training_evidence(analyses['train'])
    assert all(evidence[k] for k in ('boundary', 'terminal', 'stop', 'adjacent', 'dotted', 'left', 'right'))
    if any(a['distractor_count'] for a in analyses['test']):
        assert evidence['noise'], 'training must demonstrate off-phase preservation'
    assert len({a['signal_count'] for a in analyses['train']}) > 1
    assert len({max(a['periods']) for a in analyses['train']}) > 1


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test() -> None:
    # Original labels supply evidence independent of this implementation.
    for pair in _source_pairs():
        _analyze_pair(pair)
        for turns in range(4):
            for reverse_colors in (False, True):
                inp, out = np.rot90(pair['input'], turns), np.rot90(pair['output'], turns)
                if reverse_colors:
                    inp, out = 9 - inp, 9 - out
                assert np.array_equal(_solve_grid(inp), out)
    for orientation in ('H', 'V'):
        for regime in ('source_train', 'target_test'):
            for witness in (None, 'stop', 'boundary'):
                settings = {'background': 8, 'palette': [0, 3], 'noise': True, 'transfer': 'source'}
                pair = next(_fallback_pairs(settings, orientation, regime, witness))
                a = _analyze_pair(pair)
                assert a['stopping_witnesses'] and a['terminal_count'] < a['signal_count']
    puzzles = [generate_puzzle(10000 + i) for i in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    features = []
    signal_features = []
    constant_palettes = 0
    regimes = set()
    measured = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        axes = []
        palettes = []
        for split in ('train', 'test'):
            split_axes = set()
            for pair in puzzle[split]:
                a = _analyze_pair(pair)
                measured.append(_review_features(pair))
                split_axes.add(a['orientation'])
                g = pair['input']
                palettes.append(tuple(np.unique(g)))
                shape = g.shape
                repeated = len({s['reference_color'] for s in a['signals']}) < a['signal_count']
                mode = 'boundary' if not a['terminal_count'] else 'terminal' if a['terminal_count'] == a['signal_count'] else 'mixed'
                features.append((shape, a['background'], len(a['colors']) + 1, a['signal_count'],
                                 mode, a['distractor_count'], a['periods'].count(1), repeated, split))
                length = shape[1] if a['orientation'] == 'H' else shape[0]
                for s in a['signals']:
                    t = s['terminal']
                    steps = 0 if t is None else min(abs(t-s['a']), abs(t-s['b'])) // s['period']
                    direction = 0 if t is None else (-1 if t < s['a'] else 1)
                    signal_features.append((s['period'], steps, direction, len(s['distractors']), s['a'], s['b'], length))
            axes.append(''.join(sorted(split_axes)))
        regimes.add(tuple(axes))
        constant_palettes += len(set(palettes)) == 1
    assert len({f[0] for f in features}) >= 100
    assert {np.sign(f[0][0] - f[0][1]) for f in features} == {-1, 0, 1}
    assert max(max(f[0]) for f in features) == 30
    assert len({f[1] for f in features}) == 10
    assert min(f[2] for f in features) == 2 and max(f[2] for f in features) >= 8
    assert min(f[3] for f in features) == 1 and max(f[3] for f in features) >= 9
    assert {f[4] for f in features} == {'boundary', 'terminal', 'mixed'}
    assert min(f[5] for f in features) == 0 and max(f[5] for f in features) >= 5
    assert {min(2, f[6]) for f in features} == {0, 1, 2}
    assert all(any(f[7] and f[8] == split for f in features) for split in ('train', 'test'))
    assert {1, 2, 3, 4, 5, 6} <= {s[0] for s in signal_features}
    assert max(s[1] for s in signal_features) >= 16
    assert {s[2] for s in signal_features} == {-1, 0, 1}
    assert constant_palettes >= 2
    assert {('H', 'H'), ('V', 'V'), ('H', 'V'), ('V', 'H')} <= regimes
    for field in ('repeated_periods', 'repeated_reference_different_period_mode',
                  'perpendicular_repeat', 'clean_and_noisy', 'cross_role_reuse',
                  'boundary_phase_gap'):
        assert sum(bool(f[field]) for f in measured) >= 10, field
    assert sum(f['terminal_directions'] == [-1, 1] for f in measured) >= 10
    assert {x for f in measured for x in f['distractor_locations']} == {'between', 'beyond', 'inside_span', 'outside_span'}
    assert {x for f in measured for x in f['black_roles']} == {'reference', 'terminal', 'distractor'}
    assert {x for f in measured for x in f['reference_locations']} == {'left', 'right', 'interior'}
    assert len({tuple(f['margins']) for f in measured}) >= 50
    assert any(f['aspect'] == 'portrait' and f['repeated_references']
               and f['max_terminal_steps'] >= 5 and f['distractors']
               and f['terminal_directions'] == [-1, 1] for f in measured)
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(20000 + index, *counts)
        for split, requested in zip(('train', 'test'), counts):
            assert requested is None or len(p[split]) == requested
        validate_generated_puzzle(p)
    validate_generated_puzzle(generate_puzzle(14, 2, 1))
    assert _puzzle_key(generate_puzzle(314159, 7, 3)) == _puzzle_key(generate_puzzle(314159, 7, 3))
    assert _puzzle_key(generate_puzzle(314159, 7, 3)) != _puzzle_key(generate_puzzle(271828, 7, 3))
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, None), (2.5, 1)):
        try:
            generate_puzzle(1, *counts)
        except ValueError:
            continue
        raise AssertionError('invalid counts accepted')
