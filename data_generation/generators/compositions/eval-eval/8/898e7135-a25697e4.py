"""Registered exclusive stamps: macro-scale fitting, parity ink, shared overprints.

Only standard Python and NumPy are used. No files, seeds, or demonstrations are
consulted by solve. Generation retains one candidate at a time; self_test streams
42 seeds and stores only compact signatures and aggregate feature sets.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


_DIR = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIR:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(frozenset(component))
    return result


def _cells(mask):
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _normal(cells):
    r0, _, c0, _ = _bounds(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _transform(regions, op):
    """Reflect left/right for op>=4, then rotate; normalize jointly."""
    raw = []
    for region in regions:
        points = []
        for r, c in region:
            if op >= 4:
                c = -c
            for _ in range(op % 4):
                r, c = c, -r
            points.append((r, c))
        raw.append(points)
    all_points = [p for region in raw for p in region]
    r0, _, c0, _ = _bounds(all_points)
    return tuple(frozenset((r - r0, c - c0) for r, c in region) for region in raw)


def _canonical(cells):
    return min(tuple(sorted(_transform((cells,), op)[0])) for op in range(8))


def _orbit_size(cells):
    return len({tuple(sorted(_transform((cells,), op)[0])) for op in range(8)})


def _span(cells):
    r0, r1, c0, c1 = _bounds(cells)
    return r1 - r0, c1 - c0


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected an integer 1..30 by 1..30 grid with colors 0..9')
    return a.astype(np.int8, copy=False)


def _expand(a, scale):
    return np.repeat(np.repeat(a, scale, axis=0), scale, axis=1)


def _decode_key(inp, component, scale, bg):
    r0, r1, c0, c1 = _bounds(component)
    h, w = r1 - r0, c1 - c0
    if h % scale or w % scale:
        return None
    crop = inp[r0:r1, c0:c1]
    # No disconnected neighbors can enter this crop in accepted inputs.
    # Mask them explicitly anyway, so this decoder describes only this key.
    isolated = np.full(crop.shape, bg, dtype=np.int8)
    for r, c in component:
        isolated[r - r0, c - c0] = inp[r, c]
    logical = isolated[::scale, ::scale]
    if not np.array_equal(_expand(logical, scale), isolated):
        return None
    colors = tuple(int(c) for c in np.unique(logical) if c != bg)
    if len(colors) != 2:
        return None
    regions = tuple(_cells(logical == color) for color in colors)
    if any(len(_components(region)) != 1 for region in regions):
        return None
    return {'regions': regions, 'colors': colors, 'bounds': (r0, r1, c0, c1)}


def _render(board, moves, policy=(0, 1), overlap='exclusive', plugs_only=False):
    """Simultaneous printing, independent of key/slot enumeration order."""
    out = board.copy()
    coverage = Counter(p for move in moves for p in move['arm'])
    for move in moves:
        colors = move['colors'][::-1] if policy[move['parity']] else move['colors']
        for p in move['plug']:
            out[p] = colors[move['plug_index']]
        if not plugs_only:
            ink = colors[1 - move['plug_index']]
            for p in move['arm']:
                if overlap == 'union' or coverage[p] == 1:
                    out[p] = ink
                elif overlap == 'background':
                    # Deliberately wrong alternative used only in diagnostics.
                    out[p] = move['background']
    return out


def _recover(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= inp.size // 2:
        raise ValueError('background must occupy more than half the input')
    objects = _components(_cells(inp != bg))
    boards, external = [], []
    for obj in objects:
        colors = {int(inp[p]) for p in obj}
        if len(colors) == 1:
            color = next(iter(colors))
            r0, r1, c0, c1 = _bounds(obj)
            board = inp[r0:r1, c0:c1]
            if (min(board.shape) >= 5
                    and np.all(board[0] == color) and np.all(board[-1] == color)
                    and np.all(board[:, 0] == color) and np.all(board[:, -1] == color)
                    and np.all((board == color) | (board == bg))):
                holes = _components(_cells(board == bg))
                if 2 <= len(holes) <= 3:
                    boards.append((board.copy(), color, holes, obj))
                    continue
            raise ValueError('unrecognized monochrome object')
        if len(colors) != 2:
            raise ValueError('each loose stamp must have exactly two inks')
        external.append(obj)
    if len(boards) != 1:
        raise ValueError('expected one connected rectangular registration board')
    board, structural, holes, board_obj = boards[0]
    if len(external) != len(holes) or not 2 <= len(external) <= 3:
        raise ValueError('one whole stamp is required per slot')
    if any(structural in {int(inp[p]) for p in obj} for obj in external):
        raise ValueError('structure color is reserved for the board')
    structure = _cells(board == structural)
    answers = {}
    for scale in range(2, min(30 // n for n in board.shape) + 1):
        keys = [_decode_key(inp, obj, scale, bg) for obj in external]
        if any(key is None for key in keys):
            continue
        candidates = []
        for index, key in enumerate(keys):
            options, seen = [], set()
            for hi, hole in enumerate(holes):
                hr, _, hc, _ = _bounds(hole)
                for op in range(8):
                    regions = _transform(key['regions'], op)
                    for pi in (0, 1):
                        pr, _, pc, _ = _bounds(regions[pi])
                        placed = tuple(_shift(region, hr - pr, hc - pc) for region in regions)
                        if placed[pi] != hole or not placed[1 - pi] <= structure:
                            continue
                        move = {'hole_index': hi, 'key_index': index,
                                'plug': placed[pi], 'arm': placed[1 - pi],
                                'plug_index': pi, 'colors': key['colors'],
                                'parity': int(op >= 4), 'op': op, 'background': bg}
                        signature = (hi, placed, pi, int(op >= 4))
                        if signature not in seen:
                            seen.add(signature)
                            options.append(move)
            candidates.append(options)
        if any(not options for options in candidates):
            continue

        def search(index, used, chosen, coverage):
            if len(answers) > 1:
                return
            if index == len(keys):
                logical = _render(board, chosen)
                out = _expand(logical, scale)
                signature = _key(out)
                if signature not in answers:
                    answers[signature] = (out, {'board': board, 'structural': structural,
                        'background': bg, 'scale': scale, 'holes': holes,
                        'moves': tuple(chosen), 'keys': keys, 'coverage': coverage.copy(),
                        'input_shape': inp.shape})
                return
            for move in candidates[index]:
                if move['hole_index'] in used:
                    continue
                updated = coverage.copy()
                updated.update(move['arm'])
                if max(updated.values(), default=0) > 2:
                    continue
                search(index + 1, used | {move['hole_index']}, chosen + [move], updated)

        search(0, set(), [], Counter())
        if len(answers) > 1:
            break
    if len(answers) != 1:
        raise ValueError('expected one scale and one final colored registration')
    return next(iter(answers.values()))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the unique cropped, enlarged print from the input alone."""
    return _recover(input_grid)[0].copy()


def _grow_shape(rng, area, max_side):
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _DIR} - cells)
        frontier = [p for p in frontier if max(_span(cells | {p})) <= max_side]
        if not frontier:
            return None
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return _normal(cells)


def _sample_board(rng, count, scale, equal_area):
    low, high = ((8, 13) if scale == 2 else (7, 10))
    if count == 3:
        low = 9
    h, w = int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))
    holes, occupied, classes = [], set(), set()
    for _ in range(count):
        for attempt in range(60):
            area = 5 if equal_area else int(rng.integers(4, 8 if scale == 2 else 7))
            side = 3 if scale == 3 or rng.random() < .7 else 4
            shape = _grow_shape(rng, area, side)
            if shape is None or _orbit_size(shape) != 8:
                continue
            signature = _canonical(shape)
            if signature in classes:
                continue
            sh, sw = _span(shape)
            r, c = int(rng.integers(1, h - sh)), int(rng.integers(1, w - sw))
            placed = _shift(shape, r, c)
            if any((r + dr, c + dc) in occupied for r, c in placed for dr, dc in ((0, 0),) + _DIR):
                continue
            holes.append(placed)
            occupied.update(placed)
            classes.add(signature)
            break
        else:
            return None
    structure = frozenset((r, c) for r in range(h) for c in range(w)) - occupied
    if len(_components(structure)) != 1:
        return None
    return (h, w), holes, structure


def _handle_pool(rng, hole, others, structure, scale, classes):
    span_limit = 6 if scale == 2 else 4
    starts = sorted({(r + dr, c + dc) for r, c in hole for dr, dc in _DIR} & structure)
    pool, seen = [], set()
    for attempt in range(42):
        handle = {starts[int(rng.integers(len(starts)))]}
        target = int(rng.integers(3, 10 if scale == 2 else 7))
        for _ in range(target - 1):
            frontier = sorted(({(r + dr, c + dc) for r, c in handle for dr, dc in _DIR}
                               & structure) - handle)
            frontier = [p for p in frontier if max(_span(hole | handle | {p})) <= span_limit]
            if not frontier:
                break
            distances = [min(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in others) for p in frontier]
            weights = np.array([1 / (d + 1) ** 2 for d in distances])
            p = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))]
            handle.add(p)
        frozen = frozenset(handle)
        if (len(handle) >= 3 and frozen not in seen
                and max(_span(hole | handle)) <= span_limit
                and _canonical(handle) not in classes):
            seen.add(frozen)
            pool.append(frozen)
    return pool


def _sample_handles(rng, holes, structure, scale):
    classes = {_canonical(hole) for hole in holes}
    pools = [_handle_pool(rng, hole, set().union(*(h for h in holes if h is not hole)),
                          structure, scale, classes) for hole in holes]
    if any(not pool for pool in pools):
        return None
    for _ in range(100):
        chosen = [pools[0][int(rng.integers(len(pools[0])))]]
        for pool in pools[1:]:
            union = set().union(*chosen)
            options = [arm for arm in pool if arm & union
                       and not any(len(arm & a & b) for i, a in enumerate(chosen) for b in chosen[i + 1:])]
            if not options:
                break
            chosen.append(options[int(rng.integers(len(options)))])
        if len(chosen) != len(holes):
            continue
        coverage = Counter(p for arm in chosen for p in arm)
        if max(coverage.values()) > 2 or not any(n == 2 for n in coverage.values()):
            continue
        if not all(any(coverage[p] == 1 for p in arm) for arm in chosen):
            continue
        return chosen
    return None


def _pack(rng, board, stamps, bg):
    objects = [board] + stamps
    for attempt in range(30):
        h, w = ((int(rng.integers(23, 31)), int(rng.integers(23, 31))) if attempt < 12 else (30, 30))
        inp = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, float(rng.random())))
        for index in order:
            obj = objects[index]
            oh, ow = obj.shape
            if oh > h or ow > w:
                break
            summed = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            totals = summed[oh:, ow:] - summed[:-oh, ow:] - summed[oh:, :-ow] + summed[:-oh, :-ow]
            options = np.argwhere(totals == 0)
            if not len(options):
                break
            r, c = map(int, options[int(rng.integers(len(options)))])
            inp[r:r + oh, c:c + ow] = obj
            blocked[max(0, r - 1):min(h, r + oh + 1), max(0, c - 1):min(w, c + ow + 1)] = True
        else:
            if np.count_nonzero(inp == bg) <= inp.size // 2:
                continue
            if rng.random() < .8:
                r0, r1, c0, c1 = _bounds(_cells(inp != bg))
                r0, c0 = max(0, r0 - int(rng.integers(4))), max(0, c0 - int(rng.integers(4)))
                r1, c1 = min(h, r1 + int(rng.integers(4))), min(w, c1 + int(rng.integers(4)))
                crop = inp[r0:r1, c0:c1]
                if np.count_nonzero(crop == bg) > crop.size // 2:
                    inp = crop.copy()
            return inp
    return None


def _make_pair(rng, count, scale, witness=None):
    for attempt in range(240):
        sampled = _sample_board(rng, count, scale, equal_area=witness == 0)
        if sampled is None:
            continue
        shape, holes, structure = sampled
        arms = _sample_handles(rng, holes, structure, scale)
        if arms is None:
            continue
        palette = [int(c) for c in rng.permutation(10)]
        bg, structural, inks = palette[0], palette[1], palette[2:]
        board = np.full(shape, structural, dtype=np.int8)
        for hole in holes:
            for p in hole:
                board[p] = bg
        moves, stamps = [], []
        shared_ink = inks[0]
        for i, (hole, arm) in enumerate(zip(holes, arms)):
            parity = int(rng.integers(2))
            if witness is not None and i < 2:
                parity = (i + witness) % 2
            op = int(rng.integers(4)) + 4 * parity
            if witness == 0 and i == 0:
                op = int(rng.integers(1, 4))
            if witness == 1 and i == 1:
                op = 0
            final_arm = shared_ink if witness == 0 else inks[i] if witness == 1 else int(rng.choice(inks))
            final_plug = int(rng.choice([c for c in inks if c != final_arm]))
            # Input orientation is obtained from the destination by a D4 map.
            # The inverse map has the same parity, which controls local inks.
            regions = _transform((hole, arm), op)
            source_colors = (final_plug, final_arm) if not parity else (final_arm, final_plug)
            oh, ow = _span(regions[0] | regions[1])
            stamp = np.full((oh, ow), bg, dtype=np.int8)
            for region, color in zip(regions, source_colors):
                for p in region:
                    stamp[p] = color
            stamps.append(_expand(stamp, scale))
            moves.append({'plug': hole, 'arm': arm, 'plug_index': 0,
                          'colors': source_colors, 'parity': parity, 'op': op, 'background': bg})
        inp = _pack(rng, board, stamps, bg)
        if inp is None:
            continue
        expected = _expand(_render(board, moves), scale)
        try:
            actual, info = _recover(inp)
        except ValueError:
            continue
        if not np.array_equal(actual, expected):
            raise AssertionError('construction and input-only fitting disagree')
        return {'input': inp, 'output': expected}
    raise RuntimeError('bounded registration construction exhausted')


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('counts must satisfy 2<=train<=10, 1<=test<=4, train>test')
    rng = np.random.default_rng(seed)
    # Shifts change a visible structural demand, never the transformation.
    schedules = ((2, 2, 3, 2), (2, 2, 2, 3), (2, 3, 2, 2), (3, 2, 2, 2))
    tr_count, tr_scale, te_count, te_scale = schedules[int(rng.integers(len(schedules)))]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n, count, scale in (('train', num_train, tr_count, tr_scale), ('test', num_test, te_count, te_scale)):
        for i in range(n):
            for _ in range(8):
                pair = _make_pair(rng, count, scale, i if split == 'train' and i < 2 else None)
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
    rng.shuffle(puzzle['train'])
    return puzzle


def _features(info):
    moves, coverage = info['moves'], info['coverage']
    overlap = [p for p, n in coverage.items() if n == 2]
    post_inks = []
    for move in moves:
        colors = move['colors'][::-1] if move['parity'] else move['colors']
        post_inks.append(colors[1 - move['plug_index']])
    equal = any(len(a) == len(b) and _canonical(a) != _canonical(b)
                for i, a in enumerate(info['holes']) for b in info['holes'][i + 1:])
    same = any(len({post_inks[i] for i, m in enumerate(moves) if p in m['arm']}) == 1 for p in overlap)
    different = any(len({post_inks[i] for i, m in enumerate(moves) if p in m['arm']}) == 2 for p in overlap)
    return {'scale': info['scale'], 'count': len(moves), 'board_shape': info['board'].shape,
            'input_shape': info['input_shape'], 'background': info['background'],
            'structural': info['structural'], 'operations': tuple(m['op'] for m in moves),
            'classes': tuple(_canonical(h) for h in info['holes']),
            'arm_classes': tuple(_canonical(m['arm']) for m in moves),
            'areas': tuple(len(m['plug']) for m in moves),
            'arm_areas': tuple(len(m['arm']) for m in moves),
            'overlap_cells': len(overlap), 'overlap_components': len(_components(overlap)),
            'equal_area_decoy': equal, 'same_ink_overlap': same, 'different_ink_overlap': different,
            'plug_indices': tuple(m['plug_index'] for m in moves)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(expected, out), 'wrong output'
            board, scale, moves, coverage = info['board'], info['scale'], info['moves'], info['coverage']
            assert out.shape == tuple(n * scale for n in board.shape)
            assert scale in (2, 3)
            assert all(4 <= len(h) <= 7 and _orbit_size(h) == 8 for h in info['holes'])
            assert all(3 <= len(m['arm']) <= 9 for m in moves)
            assert any(n == 2 for n in coverage.values()), 'stamps must interact'
            assert all(n <= 2 for n in coverage.values())
            assert all(any(coverage[p] == 1 for p in m['arm']) for m in moves), 'each arm needs visible ink'
            linked = {0}
            for _ in moves:
                linked.update(i for i, move in enumerate(moves)
                              if any(move['arm'] & moves[j]['arm'] for j in tuple(linked)))
            assert len(linked) == len(moves), 'every stamp must join the overlap interaction'
            logical = out[::scale, ::scale]
            assert all(logical[p] == info['structural'] for p, n in coverage.items() if n == 2)
            assert not np.any(out == info['background']), 'all slots must be filled'
            assert not np.array_equal(_render(board, moves, plugs_only=True), logical), 'arms must matter'
            assert not np.array_equal(_render(board, moves, overlap='union'), logical), 'overlap must matter'
            # Both inherited mechanisms change the answer on actual examples.
            assert scale > 1 and board.shape != out.shape
            if split == 'train':
                train.append(info)
    assert {m['parity'] for info in train for m in info['moves']} == {0, 1}
    assert any(_features(info)['same_ink_overlap'] for info in train)
    assert any(_features(info)['different_ink_overlap'] for info in train)
    assert any(_features(info)['equal_area_decoy'] for info in train)
    # Check three plausible wrong color policies against the actual training
    # labels; plugs make parity observable even where arms cancel.
    for policy in ((0, 0), (1, 0), (1, 1)):
        assert any(not np.array_equal(_render(info['board'], info['moves'], policy),
                                     _render(info['board'], info['moves'])) for info in train)


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed structural and mechanism checks; no files are written."""
    puzzles, inputs, boards, slots, arms, operations = set(), set(), set(), set(), set(), set()
    dimensions, scales, counts, schedules, relations, plug_indices = set(), set(), set(), set(), set(), set()
    metrics = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_digest(puzzle))
        split_features = {}
        for split in ('train', 'test'):
            features = []
            for pair in puzzle[split]:
                inp = np.asarray(pair['input'])
                expected, info = _recover(inp)
                assert np.array_equal(expected, pair['output'])
                feature = _features(info)
                features.append(feature)
                inputs.add(_key(inp))
                boards.add((info['board'].shape, (info['board'] == info['structural']).tobytes()))
                dimensions.add(info['input_shape'])
                slots.update(feature['classes'])
                arms.update(feature['arm_classes'])
                operations.update(feature['operations'])
                scales.add(feature['scale'])
                counts.add(feature['count'])
                plug_indices.update(feature['plug_indices'])
                relations.update(int(np.sign(a - b)) for a, b in zip(feature['areas'], feature['arm_areas']))
                for name in ('equal_area_decoy', 'same_ink_overlap', 'different_ink_overlap'):
                    metrics[name] += int(feature[name])
                metrics['disconnected_overlap'] += feature['overlap_components'] > 1
                metrics['pairs'] += 1
                assert not np.array_equal(_render(info['board'], info['moves'], overlap='background'),
                                          expected[::info['scale'], ::info['scale']])
            split_features[split] = features
        schedules.add(tuple((split_features[s][0]['count'], split_features[s][0]['scale']) for s in ('train', 'test')))
    assert len(puzzles) == 42 and len(inputs) == 210
    assert len(boards) >= 150 and len(slots) >= 25 and len(arms) >= 70
    assert len(dimensions) >= 25 and len(schedules) == 4
    assert scales == {2, 3} and counts == {2, 3} and operations == set(range(8))
    assert relations == {-1, 0, 1} and plug_indices == {0, 1}
    assert all(metrics[name] > 0 for name in metrics)
    p = generate_puzzle(597919047, 2, 1)
    assert _puzzle_digest(p) == _puzzle_digest(generate_puzzle(597919047, 2, 1))
    assert _puzzle_digest(p) != _puzzle_digest(generate_puzzle(519509674, 2, 1))
    # Negative label and macrocell diagnostics. Copies are tiny and ephemeral.
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect label accepted')
    inp = np.asarray(p['train'][0]['input']).copy()
    _, info = _recover(inp)
    r0, _, c0, _ = info['keys'][0]['bounds']
    # Modify a known key cell (its component need not start at the bbox corner).
    first_key = _decode_key(inp, next(obj for obj in _components(_cells(inp != info['background']))
                                     if _bounds(obj) == info['keys'][0]['bounds']), info['scale'], info['background'])
    region = first_key['regions'][0]
    rr, cc = min(region)
    inp[r0 + rr * info['scale'], c0 + cc * info['scale']] = info['background']
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError('broken macrocell accepted')
    # A symmetric slot in open board space admits distinct rigid placements.
    # Rejection must come from competing colored solutions, not a slot tie-break.
    ambiguous = np.zeros((23, 25), dtype=np.int8)
    ambiguous[:9, :9] = 2
    slot_a = frozenset(((3, 2), (3, 3)))
    slot_b = frozenset(((1, 6), (2, 6), (3, 6), (3, 7)))
    for p0 in slot_a | slot_b:
        ambiguous[p0] = 0
    region_pairs = ((frozenset(((0, 0), (0, 1))),
                     frozenset(((1, 1), (2, 1), (2, 2)))),
                    (frozenset(((0, 0), (1, 0), (2, 0), (2, 1))),
                     frozenset(((0, 1), (0, 2), (1, 2)))))
    for regions, (r0, c0) in zip(region_pairs, ((0, 12), (12, 0))):
        stamp = np.zeros((3, 3), dtype=np.int8)
        for region, ink in zip(regions, (3, 4)):
            for p0 in region:
                stamp[p0] = ink
        ambiguous[r0:r0 + 6, c0:c0 + 6] = _expand(stamp, 2)
    try:
        solve(ambiguous)
    except ValueError as error:
        assert 'one scale and one final colored registration' in str(error)
    else:
        raise AssertionError('competing colored registrations accepted')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    return {'seeds': 42, 'pairs': metrics['pairs'], 'unique_puzzles': len(puzzles),
            'unique_inputs': len(inputs), 'board_patterns': len(boards),
            'slot_classes': len(slots), 'arm_classes': len(arms),
            'input_dimensions': len(dimensions), 'schedules': len(schedules),
            'operations': sorted(operations), 'mechanisms': dict(metrics)}
