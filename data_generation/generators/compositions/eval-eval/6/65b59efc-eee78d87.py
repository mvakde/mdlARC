"""Resonant rooms: keyed, phase-anchored motif broadcasts and collisions.

The entire rule, palette and output extent are recoverable from one input.
No parent module, stored puzzle, training output or seed is used by solve.
"""
from __future__ import annotations

from collections import deque
import copy
import numpy as np


def _full_box(mask):
    return (mask.shape == (3, 3) and 2 <= int(mask.sum()) <= 8
            and mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any())


# All non-solid full-bounding-box 3x3 motifs, rather than a small vocabulary.
_MASKS = tuple(m for code in range(1, 511)
               if _full_box(m := np.array([(code >> i) & 1 for i in range(9)],
                                           dtype=bool).reshape(3, 3)))
_DENSITIES = np.array([int(m.sum()) for m in _MASKS])
_ALTERNATIVES = ("first_motif", "source_color", "solid", "local_stamp",
                 "global_phase", "corner_anchor", "card_phase", "ignore_walls",
                 "beacon_count", "target_count", "exactly_two") + tuple(
                     f'motif_turn_{t}' for t in range(1, 8))


def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _components(passable):
    """Four-neighbor connectivity; diagonal contact never opens a wall."""
    h, w = passable.shape
    ids = np.full((h, w), -1, dtype=np.int16)
    cells = []
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or ids[r, c] >= 0:
                continue
            index = len(cells)
            ids[r, c] = index
            queue = deque([(r, c)])
            region = []
            while queue:
                y, x = queue.popleft()
                region.append((y, x))
                for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= ny < h and 0 <= nx < w and passable[ny, nx]
                            and ids[ny, nx] == -1):
                        ids[ny, nx] = index
                        queue.append((ny, nx))
            cells.append(region)
    return ids, cells


def _parse(input_grid):
    g = _grid(input_grid)
    height, width = g.shape
    assert 16 <= height <= 30 and 16 <= width <= 30
    wall, background, collision = int(g[0, 0]), int(g[7, 1]), int(g[7, -2])
    assert len({wall, background, collision}) == 3
    assert all(np.all(g[r] == wall) for r in (0, 6, 8, height-1))
    assert np.all(g[:, 0] == wall) and np.all(g[:, -1] == wall)
    # The key line alone identifies the card count. Its final swatch is the
    # collision color, after a wall delimiter and at least one carrier gap.
    delimiters = np.flatnonzero(g[7, :-1] == wall).tolist()
    assert delimiters and delimiters[0] == 0
    k = len(delimiters) - 1
    assert 2 <= k <= 4 and delimiters == [6*j for j in range(k+1)]
    assert width >= 6*k + 4
    expected_key = np.full(width, background, dtype=np.int8)
    expected_key[delimiters] = wall
    expected_key[-1] = wall
    expected_key[-2] = collision
    labels, motifs, targets, origins = [], [], [], []
    for j in range(k):
        start = 6*j + 1
        assert np.all(g[1:6, start-1] == wall)
        assert np.all(g[1:6, start+5] == wall)
        card = g[1:6, start:start+5]
        colors = np.unique(card[card != background])
        assert len(colors) == 1
        label = int(colors[0])
        assert label not in {background, wall, collision, *labels}
        coords = np.argwhere(card == label)
        top, left = coords.min(axis=0)
        bottom, right = coords.max(axis=0)
        assert (int(bottom-top), int(right-left)) == (2, 2)
        motif = card[top:bottom+1, left:right+1] == label
        assert _full_box(motif)
        target = int(g[7, start+2])
        assert target not in (background, wall, collision)
        expected_key[start+2] = target
        labels.append(label)
        motifs.append(motif.copy())
        targets.append(target)
        origins.append((int(top), int(left)))
    assert np.array_equal(g[7], expected_key)
    assert np.all(g[1:6, 6*k+1:-1] == background)
    field = g[9:-1, 1:-1].copy()
    assert set(map(int, np.unique(field))) <= {background, wall, *labels}
    ids, rooms = _components(field != wall)
    assert len(rooms) >= 3
    beacons = [np.argwhere(field == label) for label in labels]
    assert all(len(b) > 0 for b in beacons)
    occupied = {int(ids[r, c]) for b in beacons for r, c in b}
    assert len(occupied) < len(rooms), "keep a visible unseeded room"
    return dict(background=background, wall=wall, collision=collision,
                labels=labels, motifs=motifs, targets=targets, origins=origins,
                field=field, ids=ids, rooms=rooms, beacons=beacons)


def _render(info, alternative=None):
    """Vectorized signal union per identity, then identity-count collisions."""
    field, ids = info['field'], info['ids']
    rows, cols = np.indices(field.shape)
    signals = []
    owners = []
    for index, (mask, beacons) in enumerate(zip(info['motifs'], info['beacons'])):
        if alternative == 'first_motif':
            mask = info['motifs'][0]
        if alternative == 'solid':
            mask = np.ones((3, 3), dtype=bool)
        if alternative and alternative.startswith('motif_turn_'):
            turn = int(alternative.rsplit('_', 1)[-1])
            mask = np.rot90(np.fliplr(mask) if turn >= 4 else mask, turn % 4)
        union = np.zeros(field.shape, dtype=bool)
        for br, bc in beacons:
            domain = ids == ids[br, bc]
            if alternative == 'ignore_walls':
                domain = ids >= 0
            dy, dx = rows-br+1, cols-bc+1
            if alternative == 'global_phase':
                dy, dx = rows+1, cols+1
            elif alternative == 'corner_anchor':
                dy, dx = rows-br, cols-bc
            elif alternative == 'card_phase':
                top, left = info['origins'][index]
                dy, dx = dy+top, dx+left
            active = domain & mask[dy % 3, dx % 3]
            if alternative == 'local_stamp':
                active &= (dy >= 0) & (dy < 3) & (dx >= 0) & (dx < 3)
            if alternative == 'beacon_count':
                signals.append(active)
                owners.append(index)
            else:
                union |= active
        if alternative != 'beacon_count':
            signals.append(union)
            owners.append(index)
    if alternative == 'target_count':
        grouped = {}
        grouped_owners = {}
        for signal, owner in zip(signals, owners):
            target = info['targets'][owner]
            grouped[target] = grouped.get(target, np.zeros(field.shape, dtype=bool)) | signal
            grouped_owners[target] = owner
        signals = list(grouped.values())
        owners = [grouped_owners[t] for t in grouped]
    stack = np.stack(signals)
    count = stack.sum(axis=0)
    result = np.full(field.shape, info['background'], dtype=np.int8)
    result[field == info['wall']] = info['wall']
    for signal, owner in zip(signals, owners):
        color = (info['labels'][owner] if alternative == 'source_color'
                 else info['targets'][owner])
        result[signal & (count == 1)] = color
    result[count >= 2] = info['collision']
    if alternative == 'exactly_two':
        result[count > 2] = info['background']
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the supplied input, with no learned hidden conventions."""
    return _render(_parse(input_grid))


def _cellwise_expected(info):
    """Authored validation oracle: loop over rooms and pixels, without _render.

    This checks renderer mechanics; it is not independent semantic approval.
    """
    field = info['field']
    out = np.full(field.shape, info['background'], dtype=np.int8)
    out[field == info['wall']] = info['wall']
    for room_id, cells in enumerate(info['rooms']):
        emitters = []
        for index, beacons in enumerate(info['beacons']):
            sites = [(int(r), int(c)) for r, c in beacons
                     if info['ids'][r, c] == room_id]
            if sites:
                emitters.append((index, sites))
        for r, c in cells:
            active = []
            for index, sites in emitters:
                mask = info['motifs'][index]
                if any(mask[(r-br+1) % 3, (c-bc+1) % 3] for br, bc in sites):
                    active.append(index)
            if len(active) == 1:
                out[r, c] = info['targets'][active[0]]
            elif len(active) > 1:
                out[r, c] = info['collision']
    return out


def _partition(rng, h, w, many=False):
    """Three connected rooms of varied geometry, or six rooms for a shift."""
    walls = np.zeros((h, w), dtype=bool)
    style = int(rng.integers(4))
    if many:
        c1 = int(rng.integers(3, w//2-1))
        c2 = int(rng.integers(w//2+1, w-3))
        r = int(rng.integers(3, h-3))
        walls[:, [c1, c2]] = True
        walls[r, :] = True
    elif style in (0, 1):
        # Two stepped walls; horizontal links make them impermeable under
        # four-neighbor connectivity while leaving nonrectangular rooms.
        transpose = style == 1
        hh, ww = (w, h) if transpose else (h, w)
        work = np.zeros((hh, ww), dtype=bool)
        a, b = max(2, ww//3), min(ww-3, 2*ww//3)
        for base in (a, b):
            turn = int(rng.integers(2, hh-2))
            delta = int(rng.choice([-1, 0, 1]))
            work[:turn+1, base] = True
            work[turn:, base+delta] = True
            work[turn, min(base, base+delta):max(base, base+delta)+1] = True
        walls = work.T.copy() if transpose else work
    elif style == 2:
        # A T partition with short internal obstacles. These do not close
        # a passage, so connectivity cannot be replaced by rectangular boxes.
        c = int(rng.integers(4, w-4))
        r = int(rng.integers(3, h-3))
        walls[:, c] = True
        left = bool(rng.integers(2))
        walls[r, :c] = left
        walls[r, c+1:] = not left
        if c >= 6:
            walls[1:3, c-2] = True
    else:
        # Two closed wall rings create inside rooms and one common outside
        # room that routes around both obstacles.
        top = int(rng.integers(1, h-4))
        bottom = int(rng.integers(top+3, h))
        for left, right in ((1, w//2-2), (w//2+1, w-2)):
            walls[top, left:right+1] = True
            walls[bottom, left:right+1] = True
            walls[top:bottom+1, left] = True
            walls[top:bottom+1, right] = True
    ids, rooms = _components(~walls)
    assert len(rooms) == (6 if many else 3)
    return walls, ids, rooms


def _choose_site(rng, room, used, previous_phases=()):
    candidates = [(r, c) for r, c in room if (r, c) not in used
                  and (r % 3, c % 3) not in previous_phases]
    if not candidates:
        raise ValueError('room too small for distinct beacon phases')
    site = candidates[int(rng.integers(len(candidates)))]
    used.add(site)
    return site


def _sample_motif(rng, rich=False):
    # Balance density classes. Dense masks and classic symmetric patterns
    # remain valid; witness checks handle accidental ambiguity.
    density = int(rng.integers(2, 9 if rich else 7))
    candidates = np.flatnonzero(_DENSITIES == density)
    return _MASKS[int(rng.choice(candidates))].copy()


def _make_pair(rng, h, k, many, witness):
    palette = list(map(int, rng.permutation(10)))
    background, wall, collision = palette[:3]
    labels = palette[3:3+k]
    target_pool = palette[3:]
    targets = list(map(int, rng.choice(target_pool, k, replace=False)))
    if witness:
        # Explicit mapping should be visible, including where templates have
        # the same silhouette. A label derangement is a finite fallback.
        for _ in range(24):
            if all(s != t for s, t in zip(labels, targets)):
                break
            targets = list(map(int, rng.choice(target_pool, k, replace=False)))
        else:
            targets = labels[1:] + labels[:1]
        if witness == 2:
            # Equal keys must still collide when distinct source identities
            # meet. Teach this even in the smallest two-training puzzle.
            common = int(rng.choice([c for c in target_pool if c not in labels]))
            targets[0] = targets[1] = common
    elif rng.random() < .23:
        targets[-1] = targets[0]
    w = int(rng.integers(6*k+2, 29))
    masks = [_sample_motif(rng, rich=witness == 0) for _ in labels]
    if not witness and rng.random() < .20:
        masks[-1] = masks[0].copy()
    walls, ids, rooms = _partition(rng, h, w, many)
    eligible = [i for i, cells in enumerate(rooms) if len(cells) >= 18]
    if not eligible:
        raise ValueError('no room large enough for teaching overlap')
    primary = int(rng.choice(eligible))
    others = [i for i in range(len(rooms)) if i != primary and len(rooms[i]) >= 4]
    if len(others) < 2:
        raise ValueError('need one secondary room and one empty room')
    secondary = int(rng.choice(others))
    field = np.full((h, w), background, dtype=np.int8)
    field[walls] = wall
    beacons = [[] for _ in labels]
    used = set()
    # First witness exposes three distinct identities in one room when k=3.
    # Otherwise reserve another identity in the secondary room, if available.
    main_count = k if witness == 1 else min(2, k)
    for index in range(k):
        room_id = primary if index < main_count else secondary
        site = _choose_site(rng, rooms[room_id], used)
        beacons[index].append(site)
    # A repeat with a distinct phase creates genuine same-identity overlap.
    # Longer examples can exercise more phases, without changing the rule.
    if witness != 1:
        repeats = 1 if witness == 2 else int(rng.integers(0, 3))
        for _ in range(repeats):
            phases = [(r % 3, c % 3) for r, c in beacons[0]]
            site = _choose_site(rng, rooms[primary], used, phases)
            beacons[0].append(site)
    # Show an identity in a separate room too; its phase restarts at the beacon.
    if witness == 1 or (k == 2 and rng.random() < .7):
        site = _choose_site(rng, rooms[secondary], used)
        beacons[0].append(site)
    # Extra rooms can be seeded in richer tests, but always leave one empty.
    if many:
        empty = next(i for i in range(len(rooms)) if i not in (primary, secondary))
        for room_id in range(len(rooms)):
            if room_id not in (primary, secondary, empty) and rng.random() < .65:
                index = int(rng.integers(k))
                site = _choose_site(rng, rooms[room_id], used)
                beacons[index].append(site)
    for label, sites in zip(labels, beacons):
        for site in sites:
            field[site] = label
    width = w+2
    g = np.full((h+10, width), background, dtype=np.int8)
    g[[0, 6, 8, h+9], :] = wall
    g[:, [0, width-1]] = wall
    origins = []
    for j, (label, mask, target) in enumerate(zip(labels, masks, targets)):
        start = 6*j+1
        g[1:8, start-1] = wall
        g[1:8, start+5] = wall
        top, left = map(int, rng.integers(0, 3, size=2))
        origins.append((top, left))
        patch = g[1+top:4+top, start+left:start+left+3]
        patch[mask] = label
        g[7, start+2] = target
    g[7, -2] = collision
    g[9:-1, 1:-1] = field
    # Construct expected output from sampled command data; parsing is later
    # checked by both solve and the cellwise validation calculation.
    info = dict(background=background, wall=wall, collision=collision,
                labels=labels, motifs=masks, targets=targets, origins=origins, field=field,
                ids=ids, rooms=rooms,
                beacons=[np.array(b, dtype=int) for b in beacons])
    out = _render(info)
    assert np.any(out == collision)
    assert all(np.any(out == t) for t in targets)
    if witness == 1:
        required = [a for a in _ALTERNATIVES if a not in ('beacon_count', 'target_count')]
    elif witness == 2:
        required = ['beacon_count', 'target_count', 'global_phase', 'local_stamp']
    else:
        required = []
    assert all(not np.array_equal(out, _render(info, a)) for a in required)
    return {'input': g, 'output': out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, 'num_train', 2, 10)
    ntest = _count(num_test, 'num_test', 1, 4)
    if ntrain <= ntest:
        raise ValueError('num_train must be greater than num_test')
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    # The first two demonstrations are sufficient even for 2/1 counts.
    # Regime controls observable shifts only, never a hidden solving convention.
    schedule = []
    for index in range(ntrain):
        h = int(rng.integers(8, 14 if regime == 0 else 19))
        k = 3 if index == 0 else int(rng.integers(2, 4 if regime == 1 else 5))
        schedule.append(('train', h, k, False, index+1 if index < 2 else 0))
    for _ in range(ntest):
        h = int(rng.integers(14, 21) if regime == 0 else rng.integers(8, 21))
        k = 4 if regime == 1 else int(rng.integers(2, 5))
        schedule.append(('test', h, k, regime == 2, 0))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, h, k, many, witness in schedule:
        for _ in range(200):
            try:
                pair = _make_pair(rng, h, k, many, witness)
            except (AssertionError, ValueError):
                continue
            ik = _key(pair['input'])
            if split == 'train' and len(puzzle['train']) == 1:
                first = puzzle['train'][0]['input']
                now = pair['input']
                if (int(first[0, 0]), int(first[7, 1]), int(first[7, -2])) == (
                        int(now[0, 0]), int(now[7, 1]), int(now[7, -2])):
                    continue
            if ik not in seen:
                seen.add(ik)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError('bounded construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    infos = {'train': [], 'test': []}
    seen = set()
    refuted = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            ik = _key(inp)
            assert ik not in seen
            seen.add(ik)
            info = _parse(inp)
            assert out.shape == info['field'].shape
            assert np.array_equal(out, _cellwise_expected(info))
            assert np.array_equal(out == info['wall'], info['field'] == info['wall'])
            assert set(map(int, np.unique(out))) <= {
                info['background'], info['wall'], info['collision'], *info['targets']}
            assert np.any(out == info['collision'])
            assert all(np.any(out == t) for t in info['targets'])
            for room_id, cells in enumerate(info['rooms']):
                if not any(info['ids'][r, c] == room_id
                           for b in info['beacons'] for r, c in b):
                    assert all(out[r, c] == info['background'] for r, c in cells)
            # Within each room, signals have period three even around holes.
            for axis in (0, 1):
                a, b = [slice(None), slice(None)], [slice(None), slice(None)]
                a[axis], b[axis] = slice(None, -3), slice(3, None)
                same = (info['ids'][tuple(a)] == info['ids'][tuple(b)])
                same &= info['ids'][tuple(a)] >= 0
                assert np.all(out[tuple(a)][same] == out[tuple(b)][same])
            if split == 'train':
                refuted.update(a for a in _ALTERNATIVES
                               if not np.array_equal(out, _render(info, a)))
            infos[split].append(info)
    assert refuted == set(_ALTERNATIVES), ('missing teaching witness', refuted)
    training, tests = infos['train'], infos['test']
    assert len({(p['background'], p['wall'], p['collision']) for p in training}) >= 2
    # A shift must be evident in dimensions, library count, or room count.
    assert (min(p['field'].shape[0] for p in tests) > max(p['field'].shape[0] for p in training)
            or min(len(p['labels']) for p in tests) > max(len(p['labels']) for p in training)
            or min(len(p['rooms']) for p in tests) > max(len(p['rooms']) for p in training))


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(pair['input'])), _key(np.asarray(pair['output'])))
                 for split in ('train', 'test') for pair in puzzle[split])


def self_test():
    """Bounded checks of 42 seeds, all count combinations, and rule mutations.

    Return compact descriptive measurements useful to the subsequent reviewer.
    """
    assert len(_MASKS) == 321
    # Hand-calculated boundary/overlap fixture. The corner motif's center is
    # empty; a duplicate beacon three rows away must NOT create a collision.
    field = np.zeros((6, 9), dtype=np.int8)
    field[:, 4] = 5
    field[0, 0] = field[3, 0] = 2
    field[2, 1] = 3
    ids, rooms = _components(field != 5)
    fixture = dict(background=0, wall=5, collision=9, labels=[2, 3],
                   targets=[4, 6], origins=[(0, 0), (0, 0)], field=field,
                   motifs=[np.array([[1,0,1],[0,0,0],[1,0,1]], dtype=bool),
                           np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)],
                   ids=ids, rooms=rooms,
                   beacons=[np.array([[0,0],[3,0]]), np.array([[2,1]])])
    expected = np.array([[int(c) for c in row] for row in
                         ('060050000', '094050000', '699650000') * 2], dtype=np.int8)
    assert np.array_equal(_render(fixture), expected)
    assert np.array_equal(_cellwise_expected(fixture), expected)
    assert not np.array_equal(_render(fixture, 'beacon_count'), expected)
    shared = dict(fixture, targets=[4, 4])
    same_keys = _render(shared)
    assert np.array_equal(same_keys == 9, expected == 9)
    assert np.all(same_keys[(expected == 4) | (expected == 6)] == 4)
    opened = copy.deepcopy(fixture)
    opened['field'][2, 4] = 0
    opened['ids'], opened['rooms'] = _components(opened['field'] != 5)
    assert np.any(_render(opened)[:, 5:] != 0)
    assert np.array_equal(_render(opened), _cellwise_expected(opened))
    motifs, shapes, layouts, palettes, outputs, phases, signatures = (set() for _ in range(7))
    channels, densities, room_counts, collisions, origins = (set() for _ in range(5))
    shifts = [0, 0, 0]
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        train_infos, test_infos = [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                (train_infos if split == 'train' else test_infos).append(info)
                shapes.add((pair['input'].shape, pair['output'].shape))
                palettes.add((info['background'], info['wall'], info['collision']))
                outputs.add(_key(pair['output']))
                normalized = np.full(info['field'].shape, -1, dtype=np.int8)
                normalized[info['field'] == info['wall']] = -2
                for index, label in enumerate(info['labels']):
                    normalized[info['field'] == label] = index
                layouts.add(_key(normalized))
                channels.add(len(info['labels']))
                room_counts.add(len(info['rooms']))
                collisions.add(int(np.sum(pair['output'] == info['collision'])))
                for mask, origin, beacons in zip(info['motifs'], info['origins'], info['beacons']):
                    motifs.add(mask.tobytes())
                    densities.add(int(mask.sum()))
                    origins.add(origin)
                    phases.update((int(r) % 3, int(c) % 3) for r, c in beacons)
        shifts[0] += min(p['field'].shape[0] for p in test_infos) > max(p['field'].shape[0] for p in train_infos)
        shifts[1] += min(len(p['labels']) for p in test_infos) > max(len(p['labels']) for p in train_infos)
        shifts[2] += min(len(p['rooms']) for p in test_infos) > max(len(p['rooms']) for p in train_infos)
    assert len(signatures) == 42
    assert len(motifs) >= 90 and len(shapes) >= 35 and len(layouts) >= 170
    assert len(palettes) >= 100 and len(outputs) >= 190 and len(collisions) >= 25
    assert channels == {2, 3, 4} and densities == set(range(2, 9))
    assert room_counts == {3, 6} and len(phases) == len(origins) == 9
    assert all(x >= 3 for x in shifts)
    legal = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
    for index, (a, b) in enumerate(legal):
        p = generate_puzzle(1000+index, a, b)
        assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (2.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    original = generate_puzzle(700, 2, 1)
    translated = original['train'][0]['input'].copy()
    original_info = _parse(translated)
    for j, (mask, origin, label) in enumerate(zip(original_info['motifs'],
            original_info['origins'], original_info['labels'])):
        start = 6*j+1
        translated[1:6, start:start+5] = original_info['background']
        top, left = ((x+1) % 3 for x in origin)
        patch = translated[1+top:4+top, start+left:start+left+3]
        patch[mask] = label
    assert not np.array_equal(translated, original['train'][0]['input'])
    assert np.array_equal(solve(translated), original['train'][0]['output'])
    # Reject wrong answers, including every tested parent-ablation shortcut.
    info = _parse(original['train'][1]['input'])
    for alternative in _ALTERNATIVES:
        wrong = _render(info, alternative)
        if np.array_equal(wrong, original['train'][1]['output']):
            continue
        bad = copy.deepcopy(original)
        bad['train'][1]['output'] = wrong
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError('shortcut answer accepted')
    bad = copy.deepcopy(original)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged answer accepted')
    for mutation in ('broken_frame', 'invisible_collision', 'empty_card'):
        damaged = original['test'][0]['input'].copy()
        if mutation == 'broken_frame':
            damaged[8, 2] = int(damaged[7, 1])
        elif mutation == 'invisible_collision':
            damaged[7, -2] = int(damaged[7, 1])
        else:
            damaged[1:6, 1:6] = int(damaged[7, 1])
        try:
            solve(damaged)
        except AssertionError:
            continue
        raise AssertionError('malformed input accepted')
    return dict(seeds=42, pairs=210, motifs=len(motifs), shapes=len(shapes),
                layouts=len(layouts), palettes=len(palettes), outputs=len(outputs),
                densities=sorted(densities), channels=sorted(channels),
                room_counts=sorted(room_counts), collision_counts=len(collisions),
                shifts=shifts, semantic_verification='generator-owned checks only')


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
