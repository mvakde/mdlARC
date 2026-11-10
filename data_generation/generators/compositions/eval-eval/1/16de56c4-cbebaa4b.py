"""Phase-locked docking: periodic rails jointly register rigid toothed objects.

Only 0 (empty space) and 8 (stationary open bracket) have fixed color roles.
Rail reference, terminal, and payload hues are read from each input. Generation
paints labels from sampled coordinates; solve reconstructs them from the raster.
"""
from itertools import product

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]): v for (r, c), v in cells.items()}


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 integer ARC grid")
    return grid.astype(np.int8, copy=True)


def _bracket(box):
    t, l, b, r = box
    return ({(t, c) for c in range(l, r + 1)}
            | {(a, c) for a in range(t, b + 1) for c in (l, r)})


def _canonical(grid):
    choices = []
    for k in range(4):
        candidate = np.rot90(grid, k).copy()
        frame = set(map(tuple, np.argwhere(candidate == 8).tolist()))
        if not frame:
            continue
        box = _box(frame)
        t, l, b, r = box
        if b - t >= 3 and r - l >= 9 and frame == _bracket(box):
            choices.append((candidate, k, box))
    if len(choices) != 1:
        raise ValueError("expected one open color-8 bracket")
    return choices[0]


def _parse(input_grid):
    grid, turns, box = _canonical(_grid(input_grid))
    t, l, b, r = box
    raw = {q: 8 for q in _bracket(box)}
    rack = dict(raw)
    rails, role_colors = [], {0, 8}
    for row in range(t + 1, b + 1):
        positions = [c for c in range(l + 1, r) if grid[row, c] != 0]
        if not positions:
            continue
        colors = [int(grid[row, c]) for c in positions]
        repeated = [v for v in set(colors) if colors.count(v) == 2]
        if len(repeated) != 1 or any(colors.count(v) > 2 for v in colors):
            raise ValueError("each rail needs exactly one equal-colored pair")
        ref = repeated[0]
        a, z = [c for c in positions if grid[row, c] == ref]
        period = z - a
        extras = [c for c in positions if c not in (a, z)]
        aligned = [c for c in extras if (c - a) % period == 0]
        if len(aligned) > 1 or len(extras) > 2:
            raise ValueError("ambiguous terminal or too much rail clutter")
        terminal = aligned[0] if aligned else None
        if terminal is not None and a < terminal < z:
            raise ValueError("terminal must be beyond the reference pair")
        paint = ref if terminal is None else int(grid[row, terminal])
        lo, hi = (l + 1, r - 1) if terminal is None else (min(a, terminal), max(z, terminal))
        points = [c for c in range(lo, hi + 1) if (c - a) % period == 0]
        if len(points) < 3:
            raise ValueError("a rail must expose at least three completed teeth")
        for c in positions:
            raw[row, c] = rack[row, c] = int(grid[row, c])
        for c in points:
            rack[row, c] = paint
        rails.append(dict(row=row, a=a, z=z, period=period, terminal=terminal,
                          paint=paint, points=points,
                          noise=[c for c in extras if c != terminal]))
        role_colors.update(colors)
    if len(rails) not in (2, 3) or rails[-1]['row'] != b - 1:
        raise ValueError("expected two or three rails and a one-cell end margin")
    foreground = set(map(tuple, np.argwhere(grid != 0).tolist()))
    inside = {(a, c) for a in range(t, b + 1) for c in range(l, r + 1)}
    if foreground & inside != set(raw):
        raise ValueError("unrecognized cells in the bracket")
    pieces = []
    for cells in _components(foreground - inside):
        colors = {int(grid[q]) for q in cells}
        body_colors = colors - role_colors
        if len(body_colors) != 1:
            raise ValueError("a piece needs one payload hue distinct from rail roles")
        body_color = body_colors.pop()
        body = {q: body_color for q in cells if grid[q] == body_color}
        pins = {q: int(grid[q]) for q in cells if grid[q] != body_color}
        if (len(pins) not in (2, 3) or len({q[0] for q in pins}) != len(pins)
                or len({q[1] for q in pins}) != len(pins)
                or any((q[0] + 1, q[1]) not in body for q in pins)
                or not set(pins.values()) <= {s['paint'] for s in rails}
                or len(_components(body)) != 1):
            raise ValueError("expected connected payload with staggered colored tips")
        pieces.append(dict(cells={q: int(grid[q]) for q in cells}, body=body,
                           pins=pins, color=body_color))
    if not 1 <= len(pieces) <= 4:
        raise ValueError("expected one through four movable objects")
    return dict(grid=grid, turns=turns, box=box, raw=raw, rack=rack,
                rails=rails, pieces=pieces)


def _candidates(piece, rails, rack, shape, *, outer_only=False):
    """Enumerate translations, testing all tips and all solid prong cells."""
    outer = max(piece['pins'])  # tips occupy distinct rows
    bottom = rails[-1]
    rail_by_row = {s['row']: s for s in rails}
    dr = bottom['row'] - outer[0]
    choices = []
    for c in bottom['points']:
        delta = dr, c - outer[1]
        pins = _shift(piece['pins'], delta)
        if outer_only:
            if rack.get((bottom['row'], c)) != piece['pins'][outer]:
                continue
            # This alternative still respects existing solid obstacles.
            if any(q in rack and rack[q] != v for q, v in pins.items()):
                continue
        elif any(q[0] not in rail_by_row or q[1] not in rail_by_row[q[0]]['points']
                 or v != rail_by_row[q[0]]['paint'] or rack.get(q) != v
                 for q, v in pins.items()):
            continue
        body = _shift(piece['body'], delta)
        moved = {**body, **pins}
        if any(not (0 <= a < shape[0] and 0 <= x < shape[1]) for a, x in moved):
            continue
        if set(body) & set(rack):
            continue
        choices.append((delta, moved))
    return choices


def _assemblies(candidates, rack, shape):
    """Complete placement search; distinct output grids, not piece labels, count."""
    if any(not options for options in candidates):
        return []
    answers = {}
    positions = {}

    def visit(remaining, occupied):
        if not remaining:
            result = np.zeros(shape, dtype=np.int8)
            for q, v in rack.items():
                result[q] = v
            for i, option in positions.items():
                for q, v in candidates[i][option][1].items():
                    result[q] = v
            answers.setdefault(result.tobytes(), (result, dict(positions)))
            return
        viable = {i: [j for j, (_, moved) in enumerate(candidates[i])
                      if not occupied.intersection(moved)] for i in remaining}
        i = min(remaining, key=lambda x: (len(viable[x]), x))
        for j in viable[i]:
            positions[i] = j
            visit(remaining - {i}, occupied | set(candidates[i][j][1]))
            del positions[i]
            if len(answers) > 1:
                return

    visit(set(range(len(candidates))), set())
    return list(answers.values())


def _recover(input_grid):
    parsed = _parse(input_grid)
    candidates = [_candidates(p, parsed['rails'], parsed['rack'], parsed['grid'].shape)
                  for p in parsed['pieces']]
    answers = _assemblies(candidates, parsed['rack'], parsed['grid'].shape)
    if len(answers) != 1:
        raise ValueError("docking must have exactly one output")
    result, positions = answers[0]
    parsed.update(candidates=candidates, positions=positions, output=result)
    return parsed


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only visible bracket geometry, signal colors, and rigid objects."""
    parsed = _recover(input_grid)
    return np.rot90(parsed['output'], -parsed['turns']).copy()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _rail(rng, length, row, palette, terminal, max_period, force_noise=False, allow_noise=True):
    periods = list(range(2 if force_noise else 1, min(max_period, (length - 1) // 3) + 1))
    period = int(_pick(rng, periods))
    phase = int(rng.integers(period)) + 1
    phase_points = list(range(phase, length + 1, period))
    geometries = []
    for j in range(len(phase_points) - 1):
        a, z = phase_points[j:j + 2]
        if terminal:
            for k, end in enumerate(phase_points):
                if k in (j, j + 1):
                    continue
                low, high = min(j, k), max(j + 1, k)
                if low == 0 and high == len(phase_points) - 1:
                    continue  # visible stopping witness
                geometries.append((a, z, end))
        else:
            geometries.append((a, z, None))
    if not geometries:
        return None
    a, z, end = _pick(rng, geometries)
    reference = int(_pick(rng, palette))
    paint = int(_pick(rng, [v for v in palette if v != reference])) if terminal else reference
    raw = {(row, a): reference, (row, z): reference}
    if terminal:
        raw[row, end] = paint
    noise_options = [c for c in range(1, length + 1) if (c - a) % period]
    noise_colors = [v for v in palette if v not in (reference, paint)]
    if noise_options and noise_colors and (force_noise or rng.random() < .12) and allow_noise:
        raw[row, int(_pick(rng, noise_options))] = int(_pick(rng, noise_colors))
    lo, hi = (1, length) if end is None else (min(a, end), max(z, end))
    points = [c for c in range(lo, hi + 1) if (c - a) % period == 0]
    completed = dict(raw)
    completed.update({(row, c): paint for c in points})
    return dict(row=row, period=period, points=points, paint=paint,
                terminal=end, raw=raw, completed=completed)


def _body(rng, top, left, width, height):
    """A solid top bar supports variable solid, hollow, open, or ragged bodies."""
    cells = {(top, c) for c in range(left, left + width)}
    kind = int(rng.integers(5))
    for a in range(1, height):
        if kind == 0:
            columns = range(width)
        elif kind == 1:
            columns = range(width) if a == height - 1 else (0, width - 1)
        elif kind == 2:
            columns = (0, width - 1)
        elif kind == 3:
            columns = range(int(rng.integers(1, width + 1)))
        else:
            columns = (int(rng.integers(width)),)
            # Join the irregular protrusion back to the top bar.
            for y in range(top + 1, top + a):
                cells.add((y, columns[0] + left))
        cells.update((top + a, left + c) for c in columns)
    return cells


def _plug(rng, rails, rack, length, color, three=False):
    selected = rails if three else [int(_pick(rng, list(range(len(rails) - 1)))), len(rails) - 1]
    if not three:
        selected = [rails[i] for i in selected]
    configurations = []
    for cols in product(*(s['points'] for s in selected)):
        if len(set(cols)) != len(cols) or not 1 <= max(cols) - min(cols) <= 6:
            continue
        pins = {(s['row'], c): s['paint'] for s, c in zip(selected, cols)}
        top = rails[-1]['row'] + 2
        fingers = {(a, c) for s, c in zip(selected, cols) for a in range(s['row'] + 1, top)}
        if fingers & set(rack):
            continue
        configurations.append((pins, fingers))
    if not configurations:
        return None
    pins, fingers = _pick(rng, configurations)
    lo, hi = min(c for a, c in pins), max(c for a, c in pins)
    left = max(1, lo - int(rng.integers(3)))
    right = min(length, max(left + 2, hi + int(rng.integers(3))))
    if right < hi:
        return None
    body = _body(rng, rails[-1]['row'] + 2, left, right - left + 1, int(rng.integers(1, 5))) | fingers
    return dict(body={q: color for q in body}, pins=pins,
                cells={**{q: color for q in body}, **pins}, color=color)


def _scatter(rng, shape, box, pieces, raw):
    """Scatter whole objects with blank separation and no ordering hint."""
    t, l, b, r = box
    blocked_frame = {(a, c) for a in range(t - 1, b + 2) for c in range(l - 1, r + 2)}
    for _ in range(12):
        blocked = set(blocked_frame)
        scattered = []
        order = sorted(rng.permutation(len(pieces)), key=lambda i: -len(pieces[i]['cells']))
        for i in order:
            cells = pieces[i]['cells']
            a, c, bottom, right = _box(cells)
            h, w = bottom - a + 1, right - c + 1
            origins = [(y, x) for y in range(shape[0] - h + 1) for x in range(shape[1] - w + 1)]
            rng.shuffle(origins)
            for y, x in origins:
                delta = y - a, x - c
                moved = _shift(cells, delta)
                if delta == (0, 0) or set(moved) & blocked:
                    continue
                scattered.append(moved)
                blocked.update((u + dy, v + dx) for u, v in moved
                               for dy in (-1, 0, 1) for dx in (-1, 0, 1))
                break
            else:
                break
        if len(scattered) == len(pieces):
            inp = np.zeros(shape, dtype=np.int8)
            for cells in [raw] + scattered:
                for q, v in cells.items():
                    inp[q] = v
            return inp
    return None


def _make_pair(rng, settings, split, index):
    mode = settings['mode']
    if split == 'train':
        count = 1 if index == 0 else 2 if index == 1 else int(rng.integers(1, 3 if mode == 0 else 4))
    else:
        count = int(rng.integers(3, 5)) if mode == 0 else int(rng.integers(1, 4))
    rail_count = 2 if split == 'train' and index == 0 else 3 if (split == 'train' and index == 1) else int(rng.integers(2, 4))
    force_three = rail_count == 3 and ((split == 'train' and index == 1) or (split == 'test' and mode == 2))
    for _ in range(360):
        length = int(rng.integers(22 if count >= 3 else 14, 29))
        rows = [int(rng.integers(1, 3))]
        for j in range(rail_count - 1):
            rows.append(rows[-1] + int(rng.integers(1, 4)))
        bottom = rows[-1] + 1
        if split == 'train' and index == 0:
            flags = [True, False]
        elif split == 'train' and index == 1:
            flags = [False, False, True]
        else:
            flags = [bool(rng.integers(2)) for _ in rows]
        palette = settings['signals']
        cap = 4 if mode == 1 and split == 'train' else 7
        rails = [_rail(rng, length, row, palette, terminal, cap,
                       force_noise=settings['noise'] and split == 'train' and index == 0 and j == 0,
                       allow_noise=settings['noise'])
                 for j, (row, terminal) in enumerate(zip(rows, flags))]
        if any(s is None for s in rails):
            continue
        if split == 'test' and mode == 1 and max(s['period'] for s in rails) < 5:
            continue
        if split == 'train' and index == 0 and rails[0]['period'] == rails[1]['period']:
            continue
        box = (0, 0, bottom, length + 1)
        raw = {q: 8 for q in _bracket(box)}
        rack = dict(raw)
        for s in rails:
            raw.update(s['raw'])
            rack.update(s['completed'])
        pieces, occupied = [], set()
        for i in range(count):
            for attempt in range(35):
                three = rail_count == 3 and ((i == 0 and force_three) or rng.random() < .35)
                p = _plug(rng, rails, rack, length, int(_pick(rng, settings['payloads'])), three)
                if p is not None and not set(p['cells']) & occupied:
                    pieces.append(p)
                    occupied.update(p['cells'])
                    break
            else:
                break
        if len(pieces) != count:
            continue
        h = int(rng.integers(16 if count == 1 else 20 if count == 2 else 23, 31))
        w = int(rng.integers(length + 2, 31))
        full_height = max(a for a, c in occupied) + 1
        offset = int(rng.integers(min(5, h - full_height + 1))), int(rng.integers(w - length - 1))
        raw, rack = _shift(raw, offset), _shift(rack, offset)
        shifted_rails = [{**s, 'row': s['row'] + offset[0],
                          'points': [c + offset[1] for c in s['points']]} for s in rails]
        pieces = [{**p, **{name: _shift(p[name], offset) for name in ('cells', 'body', 'pins')}} for p in pieces]
        candidates = [_candidates(p, shifted_rails, rack, (h, w)) for p in pieces]
        answers = _assemblies(candidates, rack, (h, w))
        if len(answers) != 1:
            continue
        out = np.zeros((h, w), dtype=np.int8)
        for mapping in [rack] + [p['cells'] for p in pieces]:
            for q, v in mapping.items():
                out[q] = v
        if not np.array_equal(out, answers[0][0]):
            continue
        if not any(not _candidates(p, shifted_rails, raw, (h, w)) for p in pieces):
            continue  # raw reference pixels alone must not suffice for assembly
        if split == 'train' and index == 0:
            if not any(len(_candidates(p, shifted_rails, rack, (h, w), outer_only=True)) > len(options)
                       for p, options in zip(pieces, candidates)):
                continue  # teach matching multiple rails, not just one tip
        final_box = tuple(x + offset[j % 2] for j, x in enumerate(box))
        inp = _scatter(rng, (h, w), final_box, pieces, raw)
        if inp is None:
            continue
        turns = int(rng.integers(4))
        inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
        if rng.random() < .5:
            inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        return {'input': inp, 'output': out}
    raise RuntimeError("bounded phase-docking construction exhausted")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    signals = [int(v) for v in rng.choice([1, 2, 3, 4, 5, 6, 7, 9], 3, replace=False)]
    settings = dict(signals=signals, payloads=[c for c in range(1, 10) if c not in signals + [8]],
                    mode=int(rng.integers(3)), noise=bool(rng.integers(2)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(16):
                pair = _make_pair(rng, settings, split, index)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(parsed):
    pieces, rails = parsed['pieces'], parsed['rails']
    moves = [parsed['candidates'][i][parsed['positions'][i]][0] for i in range(len(pieces))]
    return dict(canvas=tuple(parsed['grid'].shape), frame=tuple(parsed['box']),
                rail_count=len(rails), object_count=len(pieces), periods=tuple(s['period'] for s in rails),
                terminals=sum(s['terminal'] is not None for s in rails),
                noise=sum(len(s['noise']) for s in rails),
                tips=tuple(len(p['pins']) for p in pieces),
                body_boxes=tuple((b - t + 1, r - l + 1) for t, l, b, r in (_box(p['body']) for p in pieces)),
                colors=tuple(sorted(np.unique(parsed['grid']).tolist())), moves=tuple(moves),
                local_options=tuple(len(c) for c in parsed['candidates']),
                payload_reuse=len(pieces) - len({p['color'] for p in pieces}))


def validate_generated_puzzle(puzzle) -> None:
    """Raster-derived invariant and teaching-evidence checks; no sampler metadata."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            parsed = _recover(inp)
            assert np.array_equal(np.rot90(out, parsed['turns']), parsed['output'])
            assert not np.array_equal(inp, out)
            rack, raw, rails = parsed['rack'], parsed['raw'], parsed['rails']
            expected = dict(rack)
            occupied, witnesses = set(), []
            for i, p in enumerate(parsed['pieces']):
                delta, moved = parsed['candidates'][i][parsed['positions'][i]]
                assert delta != (0, 0), 'each loose object must move'
                assert not occupied.intersection(moved), 'no three-owner connector or object collision'
                occupied.update(moved)
                for q, v in _shift(p['pins'], delta).items():
                    assert rack[q] == v
                assert not set(_shift(p['body'], delta)) & set(rack)
                expected.update(moved)
                witnesses.append(not _candidates(p, rails, raw, parsed['grid'].shape))
            redraw = np.zeros(parsed['grid'].shape, dtype=np.int8)
            for q, v in expected.items():
                redraw[q] = v
            assert np.array_equal(redraw, parsed['output'])
            assert any(witnesses), 'omitting progression completion must prevent a docking'
            f = _features(parsed)
            f['multi_rail_witness'] = any(
                len(_candidates(p, rails, rack, parsed['grid'].shape, outer_only=True)) > len(options)
                for p, options in zip(parsed['pieces'], parsed['candidates']))
            f['stop_witness'] = any(s['terminal'] is not None and any(
                c not in s['points'] for c in range(parsed['box'][1] + 1, parsed['box'][3])
                if (c - s['a']) % s['period'] == 0) for s in rails)
            if split == 'train':
                training.append(f)
            else:
                testing.append(f)
    assert any(f['terminals'] for f in training)
    assert any(f['terminals'] < f['rail_count'] for f in training)
    assert any(f['stop_witness'] for f in training)
    assert any(f['multi_rail_witness'] for f in training)
    assert {1, 2} <= {f['object_count'] for f in training}
    assert {2, 3} <= {f['rail_count'] for f in training}
    assert any(3 in f['tips'] for f in training)
    if any(f['noise'] for f in testing):
        assert any(f['noise'] for f in training), 'off-phase preservation needs a demonstration'


def self_test() -> dict:
    """Bounded 42-seed variation checks, symmetries, and semantic corruptions."""
    # Hand-painted semantic fixture: t = 2 (mod 3), t+1 = 4 (mod 5),
    # with the first rail stopped at 14. Only t=8 survives both constraints.
    fixture = np.zeros((24, 30), dtype=np.int8)
    for q in _bracket((0, 0, 6, 29)):
        fixture[q] = 8
    fixture[2, 2] = fixture[2, 5] = 3
    fixture[2, 14] = 6
    fixture[2, 3] = 9  # off-phase, same hue as the other rail
    fixture[5, 19] = fixture[5, 24] = 9
    payload = ({(a, 8) for a in range(3, 7)} | {(6, 9)}
               | {(a, c) for a in (7, 9) for c in (8, 9, 10)}
               | {(8, 8), (8, 10)})
    target = {**{q: 4 for q in payload}, (2, 8): 6, (5, 9): 9}
    for q, v in _shift(target, (10, 5)).items():
        fixture[q] = v
    expected = np.zeros_like(fixture)
    expected[:7] = fixture[:7]
    expected[2, [2, 5, 8, 11, 14]] = 6
    expected[5, [4, 9, 14, 19, 24]] = 9
    for q, v in target.items():
        expected[q] = v
    assert np.array_equal(solve(fixture), expected)
    parsed = _recover(fixture)
    p = parsed['pieces'][0]
    assert len(_candidates(p, parsed['rails'], parsed['rack'], fixture.shape, outer_only=True)) == 4
    assert not _candidates(p, parsed['rails'], parsed['raw'], fixture.shape)
    # Removing the stopping condition creates a second placement, not merely
    # extra paint. Completion colors and the other rail stay the same.
    extended_rails = [{**s} for s in parsed['rails']]
    extended_rails[0]['points'] = list(range(2, 29, 3))
    extended_rack = dict(parsed['rack'])
    extended_rack.update({(2, c): 6 for c in extended_rails[0]['points']})
    assert len(_candidates(p, extended_rails, extended_rack, fixture.shape)) == 2
    # A preserved singleton never becomes a socket, even with matching hue.
    wrong = {**p, 'pins': {q: 9 for q in p['pins']}}
    assert not _candidates(wrong, parsed['rails'], parsed['rack'], fixture.shape)
    records, puzzle_keys = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzle_keys.add(tuple(_key(q['input']) for s in ('train', 'test') for q in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                parsed = _recover(pair['input'])
                f = _features(parsed)
                f['orientation'] = parsed['turns']
                records.append(f)
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
        if seed in (0, 17):
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(puzzle[s][j][k], repeat[s][j][k])
                       for s in puzzle for j in range(len(puzzle[s])) for k in ('input', 'output'))
    assert len(puzzle_keys) == 42
    assert {r['object_count'] for r in records} == {1, 2, 3, 4}
    assert {r['orientation'] for r in records} == {0, 1, 2, 3}
    assert {1, 2, 3, 4, 5, 6, 7} <= {p for r in records for p in r['periods']}
    assert len({r['canvas'] for r in records}) >= 25
    assert len({b for r in records for b in r['body_boxes']}) >= 20
    assert any(r['payload_reuse'] for r in records)
    assert any(r['noise'] for r in records)
    assert any(r['terminals'] == 0 for r in records)
    assert any(r['terminals'] == r['rail_count'] for r in records)
    assert any(max(r['local_options']) > 1 for r in records), 'global object exclusion must sometimes matter'
    pair = generate_puzzle(0)['train'][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k))
    assert np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1])
    # Permuting all nonstructural colors preserves local semantic roles.
    palette = np.arange(10, dtype=np.int8)
    colors = [1, 2, 3, 4, 5, 6, 7, 9]
    palette[colors] = colors[::-1]
    assert np.array_equal(solve(palette[pair['input']]), palette[pair['output']])
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in generate_puzzle(0)[s]]
           for s in ('train', 'test')}
    location = tuple(np.argwhere(bad['test'][0]['output'] != 0)[0])
    bad['test'][0]['output'][location] = 0
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('an erased rack/object cell was accepted')
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=len(records), canvases=len({r['canvas'] for r in records}),
                body_boxes=len({b for r in records for b in r['body_boxes']}),
                object_counts=sorted({r['object_count'] for r in records}),
                periods=sorted({p for r in records for p in r['periods']}))
