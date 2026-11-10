"""Completed lookup lines act as unary counters for rotating board color layers.

Only NumPy and the standard library are used. Generation, solving, validation,
and self_test do not write files or retain puzzles between calls.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


HOST_SEEDS = (0,110977039,833774944,405563262,737347851,152750216,60430238,597046744,253228063,252783409,164786193,917734319,578398727,358515792,332568981,928642282,141534993,457645560,409830378,972688760,74961555,124253116,364478280,380440395,904782360,881344892,178223209,560082605,251444372,616920495,423139283,19704489,376935427,578186512,21090664,264253614,793783853,234396008,353769785,570972996,214838198,60875752)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _digest(a):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.int8).tobytes()).hexdigest()


def _shape_bank(s):
    """All nonempty, trimmed then consistently centered binary glyphs."""
    bank = {}
    for code in range(1, 1 << (s * s)):
        mask = np.array([(code >> j) & 1 for j in range(s * s)], bool).reshape(s, s)
        rr, cc = np.where(mask)
        crop = mask[rr.min():rr.max()+1, cc.min():cc.max()+1]
        centered = np.zeros((s, s), bool)
        r, c = (s-crop.shape[0])//2, (s-crop.shape[1])//2
        centered[r:r+crop.shape[0], c:c+crop.shape[1]] = crop
        bank[_key(centered)] = centered
    return list(bank.values())


def _macro_orbits(n):
    remaining = set(itertools.product(range(n), repeat=2))
    orbits = []
    while remaining:
        r, c = min(remaining)
        orbit = []
        for _ in range(4):
            if (r, c) not in orbit:
                orbit.append((r, c))
            r, c = c, n-1-r
        remaining.difference_update(orbit)
        orbits.append(orbit)
    return orbits


def _lines(selected, n):
    """Row terminals precede column terminals. Each complete line counts once."""
    a = np.zeros((n, n), bool)
    for r, c in selected:
        a[r, c] = True
    return np.concatenate((a.all(axis=1), a.all(axis=0)))


def _powers(selected, n, terminals, colors):
    complete = _lines(selected, n)
    return {c: sum(bool(complete[i]) and terminals[i] == c for i in range(2*n)) for c in colors}


def _rotate_layers(source, bg, divider, powers):
    result = np.full(source.shape, bg, np.int8)
    result[source == divider] = divider
    occupied = result != bg
    for color in powers:
        mask = np.rot90(source == color, -powers[color] % 4)
        assert not np.any(mask & occupied), "layer collision"
        result[mask] = color
        occupied |= mask
    return result


def _coordinate_output(info, powers=None):
    """Coordinate arithmetic check, separate from array rotation in solve."""
    source, bg, div = info['source'], info['bg'], info['divider']
    powers = info['powers'] if powers is None else powers
    length = len(source)
    result = np.full(source.shape, bg, np.int8)
    result[source == div] = div
    for r, c in np.argwhere((source != bg) & (source != div)):
        color = int(source[r, c])
        y, x = int(r), int(c)
        for _ in range(powers[color] % 4):
            y, x = x, length-1-y
        assert result[y, x] == bg, "rotated ink overlaps"
        result[y, x] = color
    return result


def _parse_canonical(a):
    """Infer all geometry and conventions from the displayed input."""
    bg = int(a[0, 0])
    bars = [r for r in range(len(a)) if a[r, 0] != bg and np.all(a[r] == a[r, 0])]
    assert len(bars) == 1, "one full-width separator required"
    length = bars[0]
    assert length in (10, 13)
    div = int(a[length, 0])
    start = length+3
    assert a.shape[0] in (2*length+3, 2*length+4)
    assert length+2 <= a.shape[1] <= length+5
    source = a[start:start+length, 2:2+length].copy()
    assert source.shape == (length, length)
    rows = np.flatnonzero(np.all(source == div, axis=1))
    cols = np.flatnonzero(np.all(source == div, axis=0))
    assert len(rows) >= 4 and np.array_equal(rows, cols)
    pitch = int(rows[1])
    assert pitch in (3, 4) and np.array_equal(rows, np.arange(0, length, pitch))
    s, n = pitch-1, (length-1)//pitch
    assert n in (3, 4) and length == n*pitch+1
    expected = np.full(a.shape, bg, np.int8)
    expected[length] = div
    template = np.full(source.shape, bg, np.int8)
    template[::pitch] = div
    template[:, ::pitch] = div
    board, lookup, colors, macro = {}, {}, set(), np.full((n, n), -1, int)
    for r, c in itertools.product(range(n), repeat=2):
        rs, cs = slice(1+r*pitch, 1+r*pitch+s), slice(1+c*pitch, 1+c*pitch+s)
        token = source[rs, cs].copy()
        inks = np.unique(token[token != bg])
        assert len(inks) == 1 and int(inks[0]) != div
        color = int(inks[0])
        key = _key(token)
        assert key not in lookup, "exact colored tokens must be unique"
        lookup[key] = (r, c)
        board[r, c] = token
        colors.add(color)
        macro[r, c] = color
        template[rs, cs] = token
    assert np.array_equal(template, source), "bad board lattice"
    assert 2 <= len(colors) <= 4
    # A color owns whole macro-cell rotation orbits: any independent turn is safe.
    assert np.array_equal(macro, np.rot90(macro)), "color territories must be rotation invariant"
    selected, query_order, query_positions = set(), [], []
    for r, c in itertools.product(range(n), repeat=2):
        rs, cs = slice(1+r*pitch, 1+r*pitch+s), slice(3+c*pitch, 3+c*pitch+s)
        token = a[rs, cs]
        if np.any(token != bg):
            assert _key(token) in lookup, "query must match one exact colored token"
            pos = lookup[_key(token)]
            assert pos not in selected, "query tokens cannot repeat"
            selected.add(pos)
            query_order.append(pos)
            query_positions.append((r, c))
        expected[rs, cs] = token
    assert selected, "nonempty query required"
    terminals = []
    for r in range(n):
        y = start+1+r*pitch+(s-1)//2
        terminals.append(int(a[y, 0]))
        expected[y, 0] = a[y, 0]
    for c in range(n):
        x = 3+c*pitch+(s-1)//2
        terminals.append(int(a[length+1, x]))
        expected[length+1, x] = a[length+1, x]
    assert set(terminals) <= colors, "terminals must name visible layer colors"
    expected[start:start+length, 2:2+length] = source
    assert np.array_equal(expected, a), "unexpected cells outside the defined regions"
    colors = tuple(sorted(colors))
    return {'bg': bg, 'divider': div, 'n': n, 's': s, 'length': length,
            'source': source, 'board': board, 'colors': colors, 'selected': selected,
            'query_order': query_order, 'query_positions': query_positions,
            'terminals': terminals, 'powers': _powers(selected, n, terminals, colors)}


def _analyze(input_grid):
    a = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        try:
            info = _parse_canonical(np.rot90(a, rotation))
        except (AssertionError, IndexError):
            continue
        info['rotation'] = rotation
        candidates.append(info)
    assert len(candidates) == 1, "scene must have a unique gallery/terminal/board orientation"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input only: exact lookup -> completed-line counts -> independent rotation."""
    info = _analyze(input_grid)
    result = _rotate_layers(info['source'], info['bg'], info['divider'], info['powers'])
    return np.rot90(result, -info['rotation']).copy()


def _selection(rng, n, mode):
    rows = list(map(int, rng.permutation(n)))
    cols = list(map(int, rng.permutation(n)))
    if mode == 'lesson':
        return {(r, c) for r in range(n) for c in range(n) if r in rows[:2] or c == cols[0]}
    if mode == 'full':
        return set(itertools.product(range(n), repeat=2))
    if mode == 'negative':
        # A whole diagonal and a row missing one cell: no terminal line is complete.
        chosen = {(i, i) for i in range(n)}
        chosen.update((rows[0], c) for c in cols[:n-2])
        if _lines(chosen, n).any():
            chosen = {(i, i) for i in range(n)}
        return chosen
    if mode == 'hole':
        nr = int(rng.integers(2, n))
        nc = int(rng.integers(1, n-1))
        chosen = {(r, c) for r in range(n) for c in range(n) if r in rows[:nr] or c in cols[:nc]}
        chosen.remove((rows[0], cols[0]))
        return chosen
    if mode == 'cross':
        nr, nc = int(rng.integers(1, n)), int(rng.integers(1, n))
        return {(r, c) for r in range(n) for c in range(n) if r in rows[:nr] or c in cols[:nc]}
    count = int(rng.integers(1, n*n+1))
    return {divmod(int(i), n) for i in rng.choice(n*n, count, replace=False)}


def _make_board(rng, n, s, k, bg, div):
    colors = list(map(int, rng.choice([c for c in range(10) if c not in (bg, div)], k, replace=False)))
    orbits = _macro_orbits(n)
    rng.shuffle(orbits)
    owner = np.full((n, n), -1, int)
    # Round-robin allocation gives every color 1..2 orbits (at most 8 tokens).
    for i, orbit in enumerate(orbits):
        for r, c in orbit:
            owner[r, c] = colors[i % k]
    bank = _shape_bank(s)
    length, pitch = n*(s+1)+1, s+1
    source = np.full((length, length), bg, np.int8)
    source[::pitch] = div
    source[:, ::pitch] = div
    board = {}
    for color in colors:
        positions = list(map(tuple, np.argwhere(owner == color)))
        ids = rng.choice(len(bank), len(positions), replace=False)
        for (r, c), index in zip(positions, ids):
            token = np.full((s, s), bg, np.int8)
            token[bank[int(index)]] = color
            source[1+r*pitch:1+r*pitch+s, 1+c*pitch:1+c*pitch+s] = token
            board[r, c] = token
    return source, board, colors


def _render(rng, source, board, selected, terminals, bg, div, orientation):
    length = len(source)
    n = int(round(len(board)**.5))
    s = next(iter(board.values())).shape[0]
    pitch = s+1
    a = np.full((2*length+3+int(rng.integers(2)), length+2+int(rng.integers(4))), bg, np.int8)
    a[length] = div
    a[length+3:2*length+3, 2:2+length] = source
    for r in range(n):
        a[length+4+r*pitch+(s-1)//2, 0] = terminals[r]
    for c in range(n):
        a[length+1, 3+c*pitch+(s-1)//2] = terminals[n+c]
    slots = list(map(int, rng.choice(n*n, len(selected), replace=False)))
    order = sorted(selected)
    rng.shuffle(order)
    for slot, pos in zip(slots, order):
        r, c = divmod(slot, n)
        a[1+r*pitch:1+r*pitch+s, 3+c*pitch:3+c*pitch+s] = board[pos]
    powers = _powers(selected, n, terminals, sorted({int(t[t != bg][0]) for t in board.values()}))
    info = {'source': source, 'bg': bg, 'divider': div, 'powers': powers}
    # Labels are constructed by coordinate arithmetic; solve uses array masks.
    out = _coordinate_output(info)
    return {'input': np.rot90(a, orientation).copy(), 'output': np.rot90(out, orientation).copy()}


def _shortcuts(info):
    """Concrete alternatives for teaching-coverage checks, never for solving."""
    bg, source, colors = info['bg'], info['source'], info['colors']
    n, selected, board = info['n'], info['selected'], info['board']
    shapes = {_key(board[p] != bg) for p in selected}
    shape_selection = {p for p in board if _key(board[p] != bg) in shapes}
    query_counts = {c: sum(np.any(board[p] == c) for p in selected) for c in colors}
    all_terminal_counts = {c: info['terminals'].count(c) for c in colors}
    positional = set(info['query_positions'])
    variants = {
        'omit_rotation': {c: 0 for c in colors},
        'query_count': query_counts,
        'ignore_completion': all_terminal_counts,
        'shape_only': _powers(shape_selection, n, info['terminals'], colors),
        'query_position': _powers(positional, n, info['terminals'], colors),
        'binary_activation': {c: int(info['powers'][c] > 0) for c in colors},
        'counterclockwise': {c: -info['powers'][c] for c in colors},
    }
    expected = _coordinate_output(info)
    return {name: not np.array_equal(_coordinate_output(info, powers), expected) for name, powers in variants.items()}


def _generate_pair(rng, index, training, orientation, allow_zero):
    if training and index < 2:
        n, s, k = 4, 2, 4
        mode = 'lesson' if index == 0 else 'full'
    else:
        n, s = ((3, 3), (3, 2), (4, 2))[int(rng.integers(3))]
        k = int(rng.integers(2, min(4, len(_macro_orbits(n)))+1))
        if training:
            mode = 'negative' if index == 2 else str(rng.choice(['cross', 'full', 'scatter']))
        else:
            mode = 'hole' if index == 0 else str(rng.choice(['hole', 'full', 'cross', 'scatter']))
    # No files or histories in retries. At most one 30x18 scene is live per attempt.
    for attempt in range(160):
        bg, div = map(int, rng.choice(10, 2, replace=False))
        source, board, colors = _make_board(rng, n, s, k, bg, div)
        if any(any(np.array_equal(source == c, np.rot90(source == c, q)) for q in (1, 2, 3)) for c in colors):
            continue
        selected = _selection(rng, n, mode)
        complete = _lines(selected, n)
        if not training and not allow_zero and not complete.any():
            continue
        terminals = list(map(int, rng.choice(colors, 2*n)))
        if training and index < 2:
            chosen = list(map(int, rng.permutation(colors)))
            credits = [chosen[0], chosen[0], chosen[1]] if index == 0 else [chosen[0]]*4+[chosen[1]]*3+[chosen[2]]
            rng.shuffle(credits)
            for line, color in zip(np.flatnonzero(complete), credits):
                terminals[int(line)] = color
        elif complete.any():
            # Broad independent label assignments, including repeated colors.
            if rng.random() < .25:
                color = int(rng.choice(colors))
                for line in np.flatnonzero(complete):
                    terminals[int(line)] = color
        pair = _render(rng, source, board, selected, terminals, bg, div, orientation)
        info = _analyze(pair['input'])
        if training and index == 0:
            checks = _shortcuts(info)
            if not all(checks[name] for name in ('omit_rotation', 'query_count', 'ignore_completion', 'shape_only', 'query_position', 'binary_activation')):
                continue
        return pair
    raise RuntimeError('bounded construction failed; no semantic fallback is substituted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    orientation = int(rng.integers(4))
    mixed_orientation = bool(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    used = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(8):
                turn = int(rng.integers(4)) if mixed_orientation and index >= 2 else orientation
                pair = _generate_pair(rng, index, split == 'train', turn, num_train >= 3)
                key = _key(pair['input'])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input budget exhausted')
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_infos, tests = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _analyze(inp)
            expected = np.rot90(_coordinate_output(info), -info['rotation'])
            assert np.array_equal(out, expected), 'wrong rotation, count, or extraction'
            canonical = np.rot90(out, info['rotation'])
            assert canonical.shape == info['source'].shape
            assert np.array_equal(canonical == info['divider'], info['source'] == info['divider'])
            for color in info['colors']:
                mask = info['source'] == color
                assert np.sum(canonical == color) == np.sum(mask), 'ink area must be conserved'
                assert all(not np.array_equal(mask, np.rot90(mask, q)) for q in (1, 2, 3)), 'informative layers required'
            (train_infos if split == 'train' else tests).append(info)
    # The two required lessons teach all four residues, unchanged zero and four,
    # independent turns, direction, completion, matching, and order invariance.
    powers = {v for info in train_infos for v in info['powers'].values()}
    assert {0, 1, 2, 3, 4} <= powers, 'missing counter teaching evidence'
    witnesses = [_shortcuts(i) for i in train_infos]
    assert all(any(w[name] for w in witnesses) for name in witnesses[0]), 'an advertised shortcut remains compatible'
    assert any(_lines(i['selected'], i['n']).all() for i in train_infos)
    assert any(0 < sum(_lines(i['selected'], i['n'])) < 2*i['n'] for i in train_infos)
    if any(not _lines(i['selected'], i['n']).any() for i in tests):
        assert any(not _lines(i['selected'], i['n']).any() for i in train_infos)
    for pair in puzzle['test']:
        assert np.array_equal(solve(np.asarray(pair['input'])), pair['output'])


def self_test():
    """42 supplied seeds, bounded memory, pixel-derived diversity and damage tests."""
    stats = {'seeds': 42, 'pairs': 0}
    fingerprints, layouts, backgrounds, palettes, selections, power_vectors, sources = (set() for _ in range(7))
    raw_counts, orientations, layer_counts = set(), set(), set()
    changed = unchanged = zero_line = holes = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        repeat = generate_puzzle(seed)
        fingerprints.add(tuple(_digest(pair['input']) for split in puzzle for pair in puzzle[split]))
        for split in puzzle:
            for pair, duplicate in zip(puzzle[split], repeat[split]):
                assert all(np.array_equal(pair[k], duplicate[k]) for k in pair), 'seed nondeterminism'
                assert np.array_equal(solve(pair['input']), pair['output'])
                info = _analyze(pair['input'])
                stats['pairs'] += 1
                layouts.add((info['n'], info['s']))
                backgrounds.add(info['bg'])
                palettes.add(info['colors'])
                selections.add((info['n'], tuple(sorted(info['selected']))))
                power_vectors.add(tuple(sorted(info['powers'].values())))
                raw_counts.update(info['powers'].values())
                orientations.add(info['rotation'])
                layer_counts.add(len(info['colors']))
                sources.add(_digest(info['source']))
                canonical = np.rot90(pair['output'], info['rotation'])
                changed += not np.array_equal(canonical, info['source'])
                unchanged += np.array_equal(canonical, info['source'])
                zero_line += not _lines(info['selected'], info['n']).any()
                for r, c in itertools.product(range(info['n']), repeat=2):
                    if (r, c) not in info['selected']:
                        augmented = info['selected'] | {(r, c)}
                        holes += int(sum(_lines(augmented, info['n']))-sum(_lines(info['selected'], info['n'])) == 2)
    assert len(fingerprints) == 42 and len(sources) >= 180
    assert layouts == {(3, 2), (3, 3), (4, 2)}
    assert backgrounds == set(range(10)) and len(palettes) >= 30
    assert layer_counts == {2, 3, 4} and orientations == {0, 1, 2, 3}
    assert len(selections) >= 70 and len(power_vectors) >= 12
    assert {0, 1, 2, 3, 4, 5, 6} <= raw_counts
    assert changed >= 150 and unchanged >= 42 and zero_line >= 42 and holes >= 42
    # Every legal explicit count combination, plus strict type/range rejection.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(700+a*10+b, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,True),(3.0,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    baseline = generate_puzzle(0)
    for q in range(4):
        transformed = {s: [{k: np.rot90(v, q).copy() for k, v in pair.items()} for pair in baseline[s]] for s in baseline}
        validate_generated_puzzle(transformed)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in baseline[s]] for s in baseline}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    stats.update({'distinct_puzzles': len(fingerprints), 'distinct_sources': len(sources),
                  'layouts': sorted(layouts), 'backgrounds': sorted(backgrounds),
                  'palettes': len(palettes), 'selection_sets': len(selections),
                  'power_vectors': len(power_vectors), 'raw_counts': sorted(raw_counts),
                  'layer_counts': sorted(layer_counts), 'changed': int(changed),
                  'unchanged': int(unchanged), 'zero_line': int(zero_line),
                  'two_line_holes': int(holes)})
    return stats
