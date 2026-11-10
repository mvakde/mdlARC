"""Repair mirror seals using a four-connected cavity-count checksum.

Only Python's standard library and NumPy are used. Generation labels boards
directly; solve receives only an input and intersects symmetry with topology.
No examples, seeds, parent code, caches or filesystem data enter prediction.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import product

import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _FOUR
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
        result.append(cells)
    return result


def _holes(walls, diagonal=False):
    # The intact perimeter is wall, so every background component is enclosed.
    return len(_components(~walls, diagonal))


def _orbits(shape):
    h, w = shape
    return [tuple(sorted({(r, c), (h - 1 - r, c),
                          (r, w - 1 - c), (h - 1 - r, w - 1 - c)}))
            for r in range((h + 1) // 2) for c in range((w + 1) // 2)]


def _interior_orbits(shape):
    h, w = shape
    return [o for o in _orbits(shape)
            if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in o)]


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('grid must be an integer ARC grid with dimensions 1..30')
    return a.astype(np.int8, copy=True)


def _read_input(value):
    a = _grid(value)
    bg = int(a[0, 0])
    if not np.all(_perimeter(a) == bg):
        raise ValueError('canvas must have a clean background margin')
    interpretations = []
    for cells in _components(a != bg):
        if len(cells) < 24:
            continue
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr) + 1, min(cc), max(cc) + 1
        board = a[r0:r1, c0:c1]
        if min(board.shape) < 7:
            continue
        wall = int(board[0, 0])
        if wall == bg or not np.all(_perimeter(board) == wall):
            continue
        extras = set(map(int, np.unique(board))) - {bg, wall}
        if len(extras) != 1:
            continue
        erase = extras.pop()
        outside = a.copy()
        outside[r0:r1, c0:c1] = bg
        pips = _components(outside != bg)
        if not 1 <= len(pips) <= 12 or any(len(p) != 1 for p in pips):
            continue
        if any(int(outside[p[0]]) != wall for p in pips):
            continue
        # A complete blank moat separates the frame and the external pips.
        if any(r0 - 1 <= r <= r1 and c0 - 1 <= c <= c1 for p in pips for r, c in p):
            continue
        interpretations.append((board.copy(), bg, wall, erase, len(pips),
                                (r0, r1, c0, c1)))
    if len(interpretations) != 1:
        raise ValueError('input does not identify one framed seal and pip checksum')
    return interpretations[0]


def _candidates(board, bg, wall, erase):
    repaired = board.copy()
    blind, hard, values_seen = [], 0, set()
    for orbit in _orbits(board.shape):
        visible = {int(board[p]) for p in orbit if board[p] != erase}
        if len(visible) > 1 or not visible <= {bg, wall}:
            raise ValueError('contradictory reflected witnesses')
        if not visible:
            blind.append(orbit)
            continue
        color = next(iter(visible))
        for p in orbit:
            repaired[p] = color
        if len(orbit) == 4 and sum(board[p] == erase for p in orbit) == 3:
            hard += 1
        if any(board[p] == erase for p in orbit):
            values_seen.add(color)
    if not 1 <= len(blind) <= 2:
        raise ValueError('family has one or two completely missing interior orbits')
    options = []
    for bits in product((0, 1), repeat=len(blind)):
        result = repaired.copy()
        for bit, orbit in zip(bits, blind):
            for p in orbit:
                result[p] = wall if bit else bg
        options.append((bits, _holes(result == wall), result))
    return options, blind, hard, values_seen


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use the intact frame, reflected evidence and visible pip count only."""
    board, bg, wall, erase, target, _ = _read_input(input_grid)
    options, _, _, _ = _candidates(board, bg, wall, erase)
    selected = [out for _, count, out in options if count == target]
    if len(selected) != 1:
        raise ValueError('checksum must select exactly one symmetric completion')
    return selected[0].copy()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _symmetric_walls(rng, shape):
    h, w = shape
    qh, qw = (h + 1) // 2, (w + 1) // 2
    style = int(rng.integers(5))
    if style == 0:
        quarter = rng.random((qh, qw)) < rng.uniform(0.40, 0.70)
    elif style == 1:
        bh, bw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        blocks = rng.random((qh, qw)) < rng.uniform(0.35, 0.70)
        rr, cc = np.indices((qh, qw))
        quarter = blocks[rr // bh, cc // bw]
    elif style == 2:
        quarter = np.ones((qh, qw), dtype=bool)
        for _ in range(int(rng.integers(2, 8))):
            r0, r1 = sorted(map(int, rng.choice(np.arange(1, qh + 1), 2, replace=False)))
            c0, c1 = sorted(map(int, rng.choice(np.arange(1, qw + 1), 2, replace=False)))
            quarter[r0:r1, c0:c1] = False
    elif style == 3:
        # Crossing full walls with independently chosen symmetric gate openings.
        quarter = np.zeros((qh, qw), dtype=bool)
        for r in range(1, qh):
            if rng.random() < 0.45:
                quarter[r, :] = True
        for c in range(1, qw):
            if rng.random() < 0.45:
                quarter[:, c] = True
        quarter[rng.random((qh, qw)) < 0.10] ^= True
    else:
        rr, cc = np.indices((qh, qw))
        bh, bw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        quarter = ((rr // bh + (cc // bw) // 2) % 2) == int(rng.integers(2))
        quarter[rng.random((qh, qw)) < 0.12] ^= True
    rr = np.minimum(np.arange(h), h - 1 - np.arange(h))
    cc = np.minimum(np.arange(w), w - 1 - np.arange(w))
    walls = quarter[rr[:, None], cc[None, :]].copy()
    walls[[0, -1], :] = True
    walls[:, [0, -1]] = True
    return walls


def _topologies(rng, shape, nblind=1, witness=False):
    """Find a genuinely topological gate, never a seed-to-answer convention."""
    orbits = _interior_orbits(shape)
    for _ in range(96):
        walls = _symmetric_walls(rng, shape)
        groups = rng.permutation(len(orbits))
        for j in range(min(40, len(orbits))):
            selected = [orbits[int(groups[j])]]
            if nblind == 2:
                selected.append(orbits[int(groups[(j + 1) % len(groups)])])
            variants = []
            for bits in product((0, 1), repeat=nblind):
                candidate = walls.copy()
                for bit, orbit in zip(bits, selected):
                    for p in orbit:
                        candidate[p] = bool(bit)
                variants.append((bits, _holes(candidate), candidate))
            if nblind == 1:
                lo, hi = variants
                if not (1 <= lo[1] <= 12 and 1 <= hi[1] <= 12 and lo[1] != hi[1]):
                    continue
                if witness and not (hi[1] > lo[1] and any(
                        _holes(v[2], True) < v[1] for v in variants)):
                    continue
                return selected, variants
            counts = [v[1] for v in variants]
            eligible = [v for i, v in enumerate(variants) if 1 <= v[1] <= 12
                        and counts.count(v[1]) == 1 and 0 < sum(v[0]) < nblind
                        and all(counts[i ^ (1 << k)] != v[1] for k in range(nblind))]
            if eligible:
                chosen = _pick(rng, eligible)
                return selected, [chosen]
    raise RuntimeError('bounded topological gate construction exhausted')


def _damage(rng, clean_walls, blind, colors, dense=False):
    bg, wall, erase = colors
    h, w = clean_walls.shape
    style = int(rng.integers(5))
    mask = np.zeros((h, w), dtype=bool)
    if style == 0:
        mask = rng.random((h, w)) < (rng.uniform(0.25, 0.55) if dense else rng.uniform(0.06, 0.28))
    else:
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
            rh, cw = int(rng.integers(1, max(3, h // 2))), int(rng.integers(1, max(3, w // 2)))
            if style == 1:
                rh = 1
            if style == 2:
                cw = 1
            for dr in range(min(rh, h - 1 - r)):
                width = cw if style != 4 else 1 + dr * cw // max(1, rh)
                mask[r + dr, c:min(w - 1, c + width)] = True
    if dense:
        # The extensive-damage curriculum applies to every geometric mask style.
        mask |= rng.random((h, w)) < rng.uniform(0.48, 0.68)
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    blind_set = set(blind)
    for orbit in _interior_orbits((h, w)):
        if orbit in blind_set:
            for p in orbit:
                mask[p] = True
        elif all(mask[p] for p in orbit):
            mask[_pick(rng, orbit)] = False
    # Two protected triple-erased orbits, one per payload value, demonstrate
    # perpendicular and diagonal lookup and refute uniform-color gap filling.
    for value in (False, True):
        eligible = [o for o in _interior_orbits((h, w))
                    if o not in blind_set and len(o) == 4 and bool(clean_walls[o[0]]) == value]
        if not eligible:
            return None
        orbit = _pick(rng, eligible)
        survivor = _pick(rng, orbit)
        for p in orbit:
            mask[p] = p != survivor
    board = np.where(clean_walls, wall, bg).astype(np.int8)
    board[mask] = erase
    return board


def _count_only_alternative(board, out, bg, wall, erase):
    """A nonmirror repair with the correct checksum witnesses the need for B."""
    target = _holes(out == wall)
    for r, c in np.argwhere(board == erase):
        trial = out.copy()
        trial[r, c] = wall if trial[r, c] == bg else bg
        if (not (np.array_equal(trial, trial[::-1]) and np.array_equal(trial, trial[:, ::-1]))
                and _holes(trial == wall) == target):
            return trial
    return None


def _isolated_pips(rng, cells, count):
    for _ in range(24):
        chosen = []
        occupied = set()
        for i in rng.permutation(len(cells)):
            r, c = cells[int(i)]
            if any((r + dr, c + dc) in occupied for dr, dc in _FOUR):
                continue
            occupied.add((r, c))
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    # A 3 by at least 9 band has capacity >=14, exceeding the maximum 12.
    choices = [p for p in cells if sum(p) % 2 == 0]
    return [choices[int(i)] for i in rng.permutation(len(choices))[:count]]


def _layout(rng, shape):
    h, w = shape
    side = int(rng.integers(4))
    mh, mw = (h + 6, w + 2) if side < 2 else (h + 2, w + 6)
    H, W = mh + int(rng.integers(min(3, 30 - mh) + 1)), mw + int(rng.integers(min(3, 30 - mw) + 1))
    r = 1 + int(rng.integers(H - mh + 1))
    c = 1 + int(rng.integers(W - mw + 1))
    if side == 0:
        cells = [(r + dr, c + dc) for dr in range(3) for dc in range(w)]
        r += 4
    elif side == 1:
        cells = [(r + h + 1 + dr, c + dc) for dr in range(3) for dc in range(w)]
    elif side == 2:
        cells = [(r + dr, c + dc) for dr in range(h) for dc in range(3)]
        c += 4
    else:
        cells = [(r + dr, c + w + 1 + dc) for dr in range(h) for dc in range(3)]
    return (H, W), (r, c), cells


def _package(rng, damaged, clean, count, colors, layout):
    bg, wall, _ = colors
    shape, (r, c), band = layout
    a = np.full(shape, bg, dtype=np.int8)
    h, w = damaged.shape
    a[r:r + h, c:c + w] = damaged
    for p in _isolated_pips(rng, band, count):
        a[p] = wall
    return {'input': a, 'output': np.where(clean, wall, bg).astype(np.int8)}


def _new_pair_group(rng, shape, colors, *, contrast=False, nblind=1, dense=False):
    for _ in range(24):
        blind, variants = _topologies(rng, shape, nblind, contrast)
        chosen = variants if contrast else [_pick(rng, variants)]
        board = _damage(rng, chosen[0][2], blind, colors, dense)
        if board is None:
            continue
        if any(_count_only_alternative(board, np.where(v[2], colors[1], colors[0]),
                                       *colors) is None for v in chosen):
            continue
        layout = _layout(rng, shape)
        return [_package(rng, board, clean, count, colors, layout)
                for _, count, clean in chosen]
    raise RuntimeError('bounded reflected-witness construction exhausted')


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(name + ' is an illegal example count')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.permutation(10)[:3]))
    shift = int(rng.integers(3))
    shape = tuple(map(int, rng.integers(9, 18, size=2)))
    puzzle = {'train': _new_pair_group(rng, shape, colors, contrast=True), 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, count in (('train', nt - 2), ('test', ne)):
        for _ in range(count):
            for attempt in range(32):
                # Roles remain visible even when a pair changes its literal palette.
                palette = tuple(map(int, rng.permutation(10)[:3])) if rng.random() < 0.65 else colors
                large = split == 'test' and shift == 0
                dims = tuple(map(int, rng.integers(18, 25, size=2) if large else rng.integers(9, 18, size=2)))
                nblind = 2 if split == 'test' and shift == 1 else 1
                dense = split == 'test' and shift == 2
                pair = _new_pair_group(rng, dims, palette, nblind=nblind, dense=dense)[0]
                key = _key(pair['input'])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    a, out = _grid(pair['input']), _grid(pair['output'])
    board, bg, wall, erase, target, box = _read_input(a)
    options, blind, hard, seen_values = _candidates(board, bg, wall, erase)
    chosen = [(bits, result) for bits, count, result in options if count == target]
    assert len(chosen) == 1 and np.array_equal(chosen[0][1], out), 'wrong or ambiguous repair'
    assert np.array_equal(out, out[::-1]) and np.array_equal(out, out[:, ::-1])
    assert np.all(out[board != erase] == board[board != erase]), 'visible cell changed'
    assert set(map(int, np.unique(out))) == {bg, wall} and erase not in out
    assert _holes(out == wall) == target
    assert hard >= 2 and seen_values == {bg, wall}, 'both folds and payload values need witnesses'
    assert np.count_nonzero(out != board) == np.count_nonzero(board == erase)
    assert _count_only_alternative(board, out, bg, wall, erase) is not None, 'checksum alone must be insufficient'
    # Every lost orbit has a causal effect on the selected chamber count.
    bits = chosen[0][0]
    counts = {b: k for b, k, _ in options}
    for i in range(len(blind)):
        flipped = tuple(1 - b if j == i else b for j, b in enumerate(bits))
        assert counts[flipped] != target, 'redundant silent orbit'
    return {'board': board, 'bg': bg, 'wall': wall, 'erase': erase, 'target': target,
            'box': box, 'blind': blind, 'bits': bits, 'hard': hard,
            'options': options, 'area': int((out == bg).sum()),
            'diag': _holes(out == wall, True) < target,
            'shape': out.shape, 'input_shape': a.shape}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    analyses, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            analyses[split].append(_analyze(pair))
    contrast = []
    for i, a in enumerate(analyses['train']):
        for b in analyses['train'][i + 1:]:
            if (np.array_equal(a['board'], b['board']) and a['bg'] == b['bg']
                    and a['wall'] == b['wall'] and a['erase'] == b['erase']
                    and a['target'] != b['target']):
                contrast.append((a, b))
    assert contrast, 'training must expose the symmetry-only ambiguity'
    assert any((a['target'] - b['target']) * (a['area'] - b['area']) < 0
               for a, b in contrast), 'training must refute monotonic background-area matching'
    assert any(a['diag'] for a in analyses['train']), 'training must distinguish four/eight connectivity'
    assert any(a['area'] > a['target'] for a in analyses['train']), 'cavities must not all be singleton pixels'
    return None


def _digest(puzzle):
    h = sha256()
    for split in ('train', 'test'):
        for p in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(p[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Streaming 42-seed study; no disk writes or retained puzzle archive."""
    fingerprints, shapes, canvases, colors, counts, parities, masks, topologies = (set() for _ in range(8))
    multi = large = dense = pairs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        assert _digest(p) == _digest(generate_puzzle(seed))
        fingerprints.add(_digest(p))
        training_shapes = [np.asarray(q['output']).shape for q in p['train']]
        large += all(min(q['output'].shape) > max(max(s) for s in training_shapes) for q in p['test'])
        for split in ('train', 'test'):
            for q in p[split]:
                pairs += 1
                a = _analyze(q)
                assert np.array_equal(solve(q['input'].copy()), q['output'])
                shapes.add(a['shape'])
                canvases.add(a['input_shape'])
                colors.add((a['bg'], a['wall'], a['erase']))
                counts.add(a['target'])
                parities.add(tuple(n % 2 for n in a['shape']))
                masks.add((a['shape'], (a['board'] == a['erase']).tobytes()))
                # Geometry is retained as a small digest, not a duplicate board.
                topologies.add((a['shape'], sha256((q['output'] == a['wall']).tobytes()).hexdigest()))
                multi += len(a['blind']) == 2
                dense += int((a['board'] == a['erase']).mean() > 0.25)
                # Reflection alone always leaves >=2 candidate answers.
                assert len(a['options']) >= 2
                # Filling every gap with one literal payload value is wrong.
                for fill in (a['bg'], a['wall']):
                    naive = np.where(a['board'] == a['erase'], fill, a['board'])
                    assert not np.array_equal(naive, q['output'])
                alternative = _count_only_alternative(a['board'], q['output'],
                                                     a['bg'], a['wall'], a['erase'])
                assert alternative is not None and _holes(alternative == a['wall']) == a['target']
    assert len(fingerprints) == 42 and len(shapes) >= 45 and len(canvases) >= 65
    assert len(colors) >= 80 and len(counts) >= 8 and len(parities) == 4
    assert len(topologies) >= 120 and len(masks) >= 150
    assert multi >= 5 and large >= 5 and dense >= 5
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(382733781, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0, 2, 1)
    broken = {s: [{k: v.copy() for k, v in q.items()} for q in p[s]] for s in p}
    broken['test'][0]['output'][1, 1] = (int(broken['test'][0]['output'][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('validator accepted a wrong repair')
    return {'seeds': 42, 'pairs': pairs, 'puzzles': len(fingerprints), 'board_shapes': len(shapes),
            'canvas_shapes': len(canvases), 'palettes': len(colors), 'cavity_counts': sorted(counts),
            'fold_parities': len(parities), 'wall_topologies': len(topologies),
            'damage_patterns': len(masks), 'two_blind_orbit_pairs': multi,
            'larger_test_puzzles': large, 'dense_damage_pairs': dense}
