"""Stencil tapes: identity-specific queues consumed in a keyed raster traversal.

Only Python and NumPy are used. No files, seed state, or training labels are
consulted by solve(). See task_spec.md for the visual grammar.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


CONFIGS = tuple((a, b, r) for a in (0, 1) for b in (False, True)
                for r in (False, True))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _turn(config):
    axis, bottom, right = config
    return (3 if bottom else 1) if axis else (2 if right else 0)


def _sites(shape, config):
    axis, bottom, right = config
    rr = list(range(shape[0]))[::(-1 if bottom else 1)]
    cc = list(range(shape[1]))[::(-1 if right else 1)]
    return ([(r, c) for c in cc for r in rr] if axis else
            [(r, c) for r in rr for c in cc])


def _border(shape, config, frame, fast, slow):
    """Closed board: solid fast arm, alternating slow arm, two frame arms."""
    rows, cols = shape
    a = np.full((rows + 2, cols + 2), frame, dtype=np.int8)
    axis, bottom, right = config
    origin_r = rows + 1 if bottom else 0
    origin_c = cols + 1 if right else 0
    if axis == 0:
        a[origin_r, 1:-1] = fast
        for i, r in enumerate(range(rows, 0, -1) if bottom else range(1, rows + 1)):
            a[r, origin_c] = slow if i % 2 == 0 else frame
    else:
        a[1:-1, origin_c] = fast
        for i, c in enumerate(range(cols, 0, -1) if right else range(1, cols + 1)):
            a[origin_r, c] = slow if i % 2 == 0 else frame
    return a


def _card(mask, label, tape, bg, frame):
    n, length = mask.shape[0], len(tape)
    w = max(n, length)
    a = np.full((n + 5, w + 2), frame, dtype=np.int8)
    a[1:-1, 1:-1] = bg
    a[1:n + 1, 1:n + 1] = np.where(mask, label, bg)
    a[n + 1, :] = frame
    a[n + 2, 1:length + 1] = label
    a[n + 3, 1:length + 1] = tape
    return a


def _runs(a, bg):
    runs = set()
    for r, row in enumerate(a):
        c = 0
        while c < len(row):
            end = c + 1
            while end < len(row) and row[end] == row[c]:
                end += 1
            if row[c] != bg and 4 <= end - c <= 8:
                runs.add((r, c, end, int(row[c])))
            c = end
    return runs


def _cards(a, bg):
    """Find closed cards from three horizontal rules, never from their order."""
    runs = _runs(a, bg)
    found = []
    for r, c, end, frame in sorted(runs):
        w = end - c - 2
        for n in range(2, 6):
            bottom = r + n + 4
            if bottom >= a.shape[0] or w < n:
                continue
            if ((r + n + 1, c, end, frame) not in runs or
                    (bottom, c, end, frame) not in runs):
                continue
            card = a[r:bottom + 1, c:end]
            if not (np.all(card[:, 0] == frame) and np.all(card[:, -1] == frame)):
                continue
            mask = card[1:n + 1, 1:n + 1]
            colors = np.unique(mask[mask != bg])
            if len(colors) != 1 or int(colors[0]) == frame:
                continue
            label = int(colors[0])
            rail, tokens = card[n + 2, 1:-1], card[n + 3, 1:-1]
            length = int(np.sum(rail != bg))
            if not 1 <= length <= 6 or w != max(n, length):
                continue
            if not (np.all(rail[:length] == label) and np.all(rail[length:] == bg)
                    and np.all(tokens[:length] != bg) and np.all(tokens[length:] == bg)
                    and np.all(card[1:n + 1, n + 1:-1] == bg)):
                continue
            found.append(dict(box=(r, bottom + 1, c, end), frame=frame, n=n,
                              label=label, mask=mask == label,
                              tape=tokens[:length].copy()))
    return found


def _read_board(a, frame, bg, labels):
    assert 4 <= a.shape[0] <= 8 and 4 <= a.shape[1] <= 8
    board = a[1:-1, 1:-1]
    assert set(map(int, np.unique(board))) <= {bg, *labels}
    edge = np.ones(a.shape, dtype=bool)
    edge[1:-1, 1:-1] = False
    interpretations = []
    for config in CONFIGS:
        axis, bottom, right = config
        r0, c0 = (-1 if bottom else 0), (-1 if right else 0)
        if axis == 0:
            fast, slow = int(a[r0, 1]), int(a[-2 if bottom else 1, c0])
        else:
            fast, slow = int(a[1, c0]), int(a[r0, -2 if right else 1])
        if len({bg, frame, fast, slow}) != 4 or {fast, slow} & set(labels):
            continue
        expected = _border(board.shape, config, frame, fast, slow)
        if np.array_equal(expected[edge], a[edge]):
            interpretations.append(dict(board=board.copy(), config=config,
                                        fast=fast, slow=slow))
    assert len(interpretations) == 1, 'Board needs one solid/alternating key'
    return interpretations[0]


def _parse(input_grid):
    original = _grid(input_grid)
    assert min(original.shape) >= 4
    bg = int(original[0, 0])
    assert np.all(original[[0, -1], :] == bg) and np.all(original[:, [0, -1]] == bg)
    candidates = []
    for rotation in range(4):
        a = np.rot90(original, rotation)
        possible = _cards(a, bg)
        groups = sorted({(p['frame'], p['n']) for p in possible})
        for frame, n in groups:
            cards = [p for p in possible if (p['frame'], p['n']) == (frame, n)]
            if not 2 <= len(cards) <= 4:
                continue
            labels = [p['label'] for p in cards]
            if len(set(labels)) != len(labels):
                continue
            covered = np.zeros(a.shape, dtype=bool)
            overlap = False
            for p in cards:
                r, end, c, stop = p['box']
                overlap |= bool(covered[r:end, c:stop].any())
                covered[r:end, c:stop] = True
            if overlap:
                continue
            rr, cc = np.nonzero((a != bg) & ~covered)
            if not len(rr):
                continue
            r, end, c, stop = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
            if covered[r:end, c:stop].any():
                continue
            try:
                parsed = _read_board(a[r:end, c:stop], frame, bg, labels)
                for p in cards:
                    assert int(np.sum(parsed['board'] == p['label'])) == len(p['tape'])
                # A non-self-colored token also makes upside-down n=2 cards
                # unambiguous. This is a visible grammar condition.
                assert any(np.any(p['tape'] != p['label']) for p in cards)
            except AssertionError:
                continue
            parsed.update(cards=cards, n=n, bg=bg, frame=frame, rotation=rotation,
                          board_box=(r, end, c, stop), input_shape=original.shape)
            candidates.append(parsed)
    assert len(candidates) == 1, f'Expected one complete visual grammar, got {len(candidates)}'
    return candidates[0]


def _render(d, alternative=None):
    board, n, bg = d['board'], d['n'], d['bg']
    config = d['config']
    scan = config
    if alternative == 'raster':
        scan = (0, False, False)
    elif alternative == 'fast_reverse':
        scan = (config[0], config[1] if not config[0] else not config[1],
                not config[2] if not config[0] else config[2])
    elif alternative == 'slow_reverse':
        scan = (config[0], not config[1] if not config[0] else config[1],
                config[2] if not config[0] else not config[2])
    elif alternative == 'other_axis':
        scan = (1 - config[0], config[1], config[2])
    lookup = {p['label']: p for p in d['cards']}
    count = dict.fromkeys(lookup, 0)
    out = np.full((board.shape[0] * n, board.shape[1] * n), bg, dtype=np.int8)
    # A positional shortcut assigns stencils in spatial card order to labels
    # in first-encounter board order. Tapes remain correctly identity keyed.
    first_labels = list(dict.fromkeys(int(board[r, c]) for r, c in _sites(board.shape, config)
                                     if board[r, c] != bg))
    positional = dict(zip(first_labels, d['cards']))
    lane_counts = {}
    for r, c in _sites(board.shape, scan):
        label = int(board[r, c])
        if label == bg:
            continue
        card = lookup[label]
        index = count[label]
        if alternative == 'reset_lane':
            lane = c if config[0] else r
            index = lane_counts.get((label, lane), 0)
            lane_counts[label, lane] = index + 1
        elif alternative == 'constant':
            index = 0
        elif alternative == 'reverse_tape':
            index = len(card['tape']) - 1 - index
        target = label if alternative == 'identity_color' else int(card['tape'][index])
        mask = positional[label]['mask'] if alternative == 'position' else card['mask']
        if alternative == 'full_blocks':
            mask = np.ones((n, n), dtype=bool)
        elif alternative == 'mirror_stamp':
            mask = np.fliplr(mask)
        elif alternative == 'macro_only':
            mask = np.rot90(mask, -_turn(config))
        out[r*n:(r+1)*n, c*n:(c+1)*n] = np.where(mask, target, bg)
        count[label] += 1
    assert all(count[x] == len(lookup[x]['tape']) for x in lookup)
    return out.copy() if alternative == 'no_rotation' else np.rot90(out, _turn(config)).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the input alone, consume one independent tape per identity."""
    return _render(_parse(input_grid))


ALTERNATIVES = ('raster', 'fast_reverse', 'slow_reverse', 'other_axis',
                'reverse_tape', 'constant', 'reset_lane', 'position',
                'full_blocks', 'identity_color', 'mirror_stamp', 'macro_only',
                'no_rotation')


def _evidence(descriptors):
    return {name: any(not np.array_equal(_render(d), _render(d, name)) for d in descriptors)
            for name in ALTERNATIVES}


def _mask(rng, n):
    style = int(rng.integers(8))
    rr, cc = np.indices((n, n))
    if style == 0:
        mask = np.ones((n, n), dtype=bool)
    elif style == 1:
        mask = (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)
    elif style == 2:
        mask = (rr == cc) | ((rr + cc == n - 1) & bool(rng.integers(2)))
    elif style == 3:
        mask = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
    elif style == 4:
        mask = (rr + cc) % 2 == int(rng.integers(2))
    elif style == 5:
        mask = (rr if rng.integers(2) else cc) == int(rng.integers(n))
    elif style == 6:
        r0, r1 = sorted(map(int, rng.choice(n + 1, 2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(n + 1, 2, replace=False)))
        mask = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    else:
        mask = rng.random((n, n)) < rng.uniform(.2, .8)
    if not mask.any():
        mask.flat[int(rng.integers(n*n))] = True
    return mask


def _pack(rng, pieces, bg):
    """Random shelf packing; bounded fallback fits <=5 pieces of size <=10x8."""
    for attempt in range(17):
        gap = int(rng.integers(1, 4)) if attempt < 16 else 1
        limit = int(rng.integers(16, 29)) if attempt < 16 else 28
        order = rng.permutation(len(pieces)).tolist()
        rows, row, width = [], [], 0
        for index in order:
            piece = pieces[index]
            if row and width + gap + piece.shape[1] > limit:
                rows.append(row)
                row, width = [], 0
            row.append(index)
            width += (gap if len(row) > 1 else 0) + piece.shape[1]
        rows.append(row)
        heights = [max(pieces[i].shape[0] for i in row) for row in rows]
        widths = [sum(pieces[i].shape[1] for i in row) + gap * (len(row) - 1) for row in rows]
        h, w = sum(heights) + gap * (len(rows) - 1), max(widths)
        if h <= 28 and w <= 28:
            break
    else:
        raise AssertionError('Packing bound violated')
    top = int(rng.integers(1, min(3, 29 - h) + 1))
    left = int(rng.integers(1, min(3, 29 - w) + 1))
    bottom = int(rng.integers(1, min(3, 30 - h - top) + 1))
    right = int(rng.integers(1, min(3, 30 - w - left) + 1))
    scene = np.full((h + top + bottom, w + left + right), bg, dtype=np.int8)
    y = top
    for row, height, width in zip(rows, heights, widths):
        x = left + int(rng.integers(w - width + 1))
        for index in row:
            p = pieces[index]
            dy = int(rng.integers(height - p.shape[0] + 1))
            scene[y + dy:y + dy + p.shape[0], x:x + p.shape[1]] = p
            x += p.shape[1] + gap
        y += height + gap
    return np.rot90(scene, int(rng.integers(4))).copy()


def _make_pair(rng, config, witness=False):
    bg, frame, fast, slow = map(int, rng.choice(10, 4, replace=False))
    n = int(rng.integers(2, 5 if witness else 6))
    k = int(rng.integers(2, 4 if witness else 5))
    labels = list(map(int, rng.choice([x for x in range(10) if x not in (bg, frame, fast, slow)],
                                    k, replace=False)))
    if witness:
        counts = rng.integers(3, 6, size=k)
    else:
        counts = rng.integers(1, 7, size=k)
        if rng.random() < .18:
            counts[:] = 1
    total = int(counts.sum())
    low = 3 if witness else 2
    sizes = [(r, c) for r in range(low, 7) for c in range(low, 7) if r*c >= total]
    rows, cols = sizes[int(rng.integers(len(sizes)))]
    board = np.full((rows, cols), bg, dtype=np.int8)
    style = int(rng.integers(5))
    coords = [(r, c) for r in range(rows) for c in range(cols)]
    if style == 0:
        coords = [coords[int(i)] for i in rng.permutation(rows*cols)]
    elif style == 1:
        coords.sort(key=lambda p: (sum(p), p[0]))
    elif style == 2:
        cr, cc = int(rng.integers(rows)), int(rng.integers(cols))
        coords.sort(key=lambda p: (abs(p[0]-cr) + abs(p[1]-cc), p))
    elif style == 3:
        coords.sort(key=lambda p: (min(p[0], p[1], rows-1-p[0], cols-1-p[1]), p))
    else:
        coords.sort(key=lambda p: ((sum(p) % 2), p))
    assignment = np.repeat(labels, counts)
    rng.shuffle(assignment)
    for pos, label in zip(coords, assignment):
        board[pos] = label
    masks = [_mask(rng, n) for _ in labels]
    if not witness and rng.random() < .25:
        masks[-1] = masks[0].copy()
    targets = [x for x in range(10) if x != bg]
    tapes = []
    for label, count in zip(labels, counts):
        tape = rng.choice(targets, int(count), replace=not witness).astype(np.int8)
        if not witness and rng.random() < .28:
            tape[:] = rng.choice(targets)
        tapes.append(tape)
    if not any(np.any(t != label) for t, label in zip(tapes, labels)):
        tapes[0][0] = frame
    cards = [_card(m, label, tape, bg, frame) for m, label, tape in zip(masks, labels, tapes)]
    keyed = _border(board.shape, config, frame, fast, slow)
    keyed[1:-1, 1:-1] = board
    scene = _pack(rng, cards + [keyed], bg)
    # Construction uses per-identity ranked macro color arrays and Kronecker
    # masks. The input-only renderer instead consumes queues block by block.
    out = np.full((rows*n, cols*n), bg, dtype=np.int8)
    axis, bottom, right = config
    for label, mask, tape in zip(labels, masks, tapes):
        positions = [(r, c) for r, c in coords[:total] if board[r, c] == label]
        positions.sort(key=lambda p: ((-p[1] if right else p[1]), (-p[0] if bottom else p[0]))
                       if axis else ((-p[0] if bottom else p[0]), (-p[1] if right else p[1])))
        colors = np.full(board.shape, bg, dtype=np.int8)
        for (r, c), token in zip(positions, tape):
            colors[r, c] = token
        pixels = np.kron(board == label, mask)
        expanded = np.kron(colors, np.ones((n, n), dtype=np.int8))
        out[pixels] = expanded[pixels]
    return {'input': scene, 'output': np.rot90(out, _turn(config)).copy()}


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f'{name} is out of range')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    first = (0, bool(rng.integers(2)), bool(rng.integers(2)))
    second = (1, not first[1], not first[2])
    held = [x for x in CONFIGS if x not in (first, second)]
    rng.shuffle(held)
    held = held[:int(rng.integers(1, 3))]
    pool = [x for x in CONFIGS if x not in held]
    training = [first, second] + [pool[int(rng.integers(len(pool)))] for _ in range(num_train-2)]
    tests = [held[int(rng.integers(len(held)))] for _ in range(num_test)]
    puzzle, seen = {'train': [], 'test': []}, set()
    required = ('fast_reverse', 'slow_reverse', 'other_axis', 'reverse_tape',
                'constant', 'reset_lane', 'position', 'full_blocks', 'identity_color', 'mirror_stamp')
    for split, schedule in (('train', training), ('test', tests)):
        for index, config in enumerate(schedule):
            witness = split == 'train' and index < 2
            for _ in range(64):
                pair = _make_pair(rng, config, witness)
                try:
                    d = _parse(pair['input'])
                except AssertionError:
                    continue
                if witness and not all(_evidence([d])[name] for name in required):
                    continue
                if split == 'test' and not all(_evidence([d])[name] for name in
                                               ('constant', 'full_blocks')):
                    continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('Bounded pair construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, descriptors = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen
            seen.add(_key(a))
            d = _parse(a)
            predicted = _render(d)
            assert np.array_equal(predicted, out), 'Wrong ranked stencil rendering'
            n = d['n']
            expected_shape = tuple(x*n for x in d['board'].shape)
            if _turn(d['config']) % 2:
                expected_shape = expected_shape[::-1]
            assert out.shape == expected_shape
            assert set(map(int, np.unique(out))) <= {d['bg'], *(int(x) for p in d['cards'] for x in p['tape'])}
            assert sum(len(p['tape']) for p in d['cards']) == int(np.sum(d['board'] != d['bg']))
            # Per-target foreground mass, checked before output rotation, is
            # the sum of stencil areas for all occurrences using that token.
            for color in set(int(x) for p in d['cards'] for x in p['tape']):
                mass = sum(int(p['mask'].sum()) * int(np.sum(p['tape'] == color)) for p in d['cards'])
                assert int(np.sum(out == color)) == mass
            descriptors[split].append(d)
    train, test = descriptors['train'], descriptors['test']
    assert {d['config'][0] for d in train} == {0, 1}
    assert {d['config'][1] for d in train} == {False, True}
    assert {d['config'][2] for d in train} == {False, True}
    assert not {d['config'] for d in train} & {d['config'] for d in test}
    evidence = _evidence(train)
    assert all(evidence.values()), f'Undistinguished alternatives: {evidence}'
    assert all(all(_evidence([d])[name] for name in ('constant', 'full_blocks')) for d in test)


def self_test():
    """Bounded, streamed 42-seed suite; returns small descriptive measurements."""
    shapes, outputs, configs, scales, libraries, lengths = set(), set(), set(), set(), set(), set()
    signatures, masks, boards, palettes = set(), set(), set(), set()
    duplicate_shapes = singleton_tapes = constant_tapes = changing_tapes = holes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, out = pair['input'], pair['output']
                digest.update(bytes(a.shape) + a.tobytes() + bytes(out.shape) + out.tobytes())
                d = _parse(a)
                assert np.array_equal(solve(a.copy()), out)
                # Rotating the whole encoding transports cards and key
                # together; normalized output must be exactly invariant.
                if split == 'test':
                    for rotation in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(a, rotation)), out)
                shapes.add(a.shape)
                outputs.add(out.shape)
                configs.add(d['config'])
                scales.add(d['n'])
                libraries.add(len(d['cards']))
                normalized = np.full(d['board'].shape, -1, dtype=np.int8)
                for i, p in enumerate(d['cards']):
                    normalized[d['board'] == p['label']] = i
                    lengths.add(len(p['tape']))
                    masks.add(_key(p['mask']))
                    singleton_tapes += len(p['tape']) == 1
                    constant_tapes += len(set(map(int, p['tape']))) == 1
                    changing_tapes += len(set(map(int, p['tape']))) > 1
                boards.add(_key(normalized))
                palettes.add((d['bg'], d['frame'], d['fast'], d['slow']))
                holes += bool(np.any(d['board'] == d['bg']))
                duplicate_shapes += len({p['mask'].tobytes() for p in d['cards']}) < len(d['cards'])
        signatures.add(digest.hexdigest())
        if seed in (0, 41):
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for split in ('train', 'test')
                       for p, q in zip(puzzle[split], again[split]) for k in ('input', 'output'))
    assert len(signatures) == 42 and len(shapes) >= 30 and len(outputs) >= 20
    assert configs == set(CONFIGS) and scales == {2, 3, 4, 5} and libraries == {2, 3, 4}
    assert lengths == set(range(1, 7)) and len(masks) >= 50 and len(boards) >= 150
    assert len(palettes) >= 100 and duplicate_shapes >= 5 and singleton_tapes >= 5
    assert constant_tapes >= 20 and changing_tapes >= 100 and holes >= 100
    for a in range(2, 11):
        for b in range(1, min(4, a-1) + 1):
            p = generate_puzzle(7000 + 10*a + b, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    for seed in (-1, True, 0.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal seed accepted')
    original = generate_puzzle(8000, 2, 1)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Bad label accepted')
    # Input grammar mutation: delete an occurrence without shortening its tape.
    d = _parse(original['test'][0]['input'])
    a = np.rot90(original['test'][0]['input'], d['rotation']).copy()
    r, _, c, _ = d['board_box']
    x, y = next((x, y) for x, y in np.ndindex(d['board'].shape) if d['board'][x, y] != d['bg'])
    a[r + 1 + x, c + 1 + y] = d['bg']
    try:
        solve(a)
    except AssertionError:
        pass
    else:
        raise AssertionError('Tape count mismatch accepted')
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), output_shapes=len(outputs),
                configurations=len(configs), stencil_scales=sorted(scales), library_counts=sorted(libraries),
                tape_lengths=sorted(lengths), stencil_bitmaps=len(masks), macro_layouts=len(boards),
                palettes=len(palettes), pairs_with_duplicate_stencils=int(duplicate_shapes),
                singleton_tapes=int(singleton_tapes), constant_tapes=int(constant_tapes),
                changing_tapes=int(changing_tapes))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
