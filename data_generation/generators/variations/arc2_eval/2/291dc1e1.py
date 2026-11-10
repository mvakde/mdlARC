"""Coordinate-key strip traversal, proper rotation, and centered stacking."""

import numpy as np


def _count_value(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(rng, seed, num_train, num_test):
    train_count = _count_value("num_train", num_train)
    test_count = _count_value("num_test", num_test)

    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        low = max(2, test_count + 1)
        train_count = low + (seed // 4) % (11 - low)
    elif train_count is None:
        low = max(2, test_count + 1)
        train_count = int(rng.integers(low, 11))
    elif test_count is None:
        high = min(4, train_count - 1)
        if high < 1:
            raise ValueError("no legal num_test for num_train")
        test_count = int(rng.integers(1, high + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


# Exact staged source contents: regression fixtures, never generation templates.
_SOURCE = ((('02222222', '18888888', '18448998', '18448998', '18888888', '18888888', '18168448', '18268448', '18668888', '18668888', '18668478', '18668748', '18888778', '18888778', '18368778', '18368778', '18338888', '18338888', '18888888', '18888888'), ('884488', '884488', '666666', '126666', '866338', '833338', '889988', '889988', '884488', '884488', '747777', '477777')), (('01111111111', '28888888888', '28338844448', '28338844448', '28888888888', '28668899888', '28668899888', '28888888888'), ('8338', '8338', '4444', '4444', '8668', '8668', '8998', '8998')), (('0111111111111111111111111111', '2888888888888888888888888888', '2833333333889988777788444488', '2833333333889988777788444488', '2888888888888888888888888888', '2899886666666688448888888888', '2899886666666688448888888888', '2888888888888888888888888888'), ('33333333', '33333333', '88899888', '88899888', '88777788', '88777788', '88444488', '88444488', '88899888', '88899888', '66666666', '66666666', '88844888', '88844888')), (('11111111111111111111110', '88888888888888888888882', '83333338877883388444482', '83333338877883388444482', '88888888888888888888882', '88888333388666666889982', '88888333388666666889982', '88888888888888888888882'), ('844448', '844448', '883388', '883388', '887788', '887788', '333333', '333333', '889988', '889988', '666666', '666666', '833338', '833338')), (('22222220', '88888881', '83384481', '83384481', '83388881', '83388881', '83386381', '83786381', '87383381', '87383381', '88888881', '88888881', '85586681', '85586681', '88886681', '88886681', '87786681', '87986681', '89788881', '87788881', '88889981', '88889981', '89988881', '89988881', '89984481', '89984481', '88888881', '88888881'), ('88844888', '88844888', '88333388', '88663388', '86666668', '86666668', '88899888', '88899888', '88844888', '88844888', '33333733', '33333377', '88855888', '88855888', '88797788', '88779788', '88999988', '88999988')))

def _grid_key(grid):
    return grid.shape, tuple(map(int, grid.flat))


def _source_pairs():
    return [{name: np.array([list(map(int, row)) for row in rows], dtype=np.int8)
             for name, rows in zip(('input', 'output'), fixture)} for fixture in _SOURCE]


def _is_source_input(grid):
    # Compare contents, including shape; no color or dimension is banned.
    rows = tuple(''.join(map(str, row)) for row in grid.tolist())
    return any(rows == fixture[0] for fixture in _SOURCE)


def _stack(blocks, background):
    width = max(b.shape[1] for b in blocks)
    result = np.full((2 * len(blocks), width), background, dtype=np.int8)
    for i, block in enumerate(blocks):
        offset = (width - block.shape[1]) // 2
        result[2*i:2*i+2, offset:offset+block.shape[1]] = block
    return result


def _gaps(rng, occupied, count, extent=None):
    minimum = occupied + count + 1
    if extent is None:
        spare = 29 - minimum
        upper = min(spare, 6) if rng.random() < .55 else spare
        extent = minimum + int(rng.integers(upper + 1))
    extra = extent - minimum
    weights = rng.dirichlet(np.full(count + 1, rng.choice([.35, 1., 3.])))
    return (1 + rng.multinomial(extra, weights)).tolist(), extent


def _layout(rng, witness=False, ordered_witness=False):
    base = 3 if rng.random() < .27 else 2
    n = int(rng.integers(4, 13) if witness else rng.integers(1, 16))
    capacity = 28 // (base + 1)
    low = max(2 if witness else 1, (n + capacity - 1) // capacity)
    high = min(4 if witness else 9, n // 2 if witness else n)
    choices = np.arange(low, high + 1)
    weights = np.array([2.5 if 2 <= k <= 4 else 1. for k in choices])
    lanes = int(rng.choice(choices, p=weights / weights.sum()))
    counts = [2 if witness else 1] * lanes
    balanced = rng.random() < .35
    preferences = rng.dirichlet(np.ones(lanes))
    for _ in range(n - sum(counts)):
        eligible = [i for i, c in enumerate(counts) if c < capacity]
        if balanced:
            least = min(counts[i] for i in eligible)
            eligible = [i for i in eligible if counts[i] == least]
        p = preferences[eligible]
        lane = int(rng.choice(eligible, p=p / p.sum()))
        counts[lane] += 1
    # A shared length ceiling is only a proposal: actual lane budgets decide fits.
    ceiling = int(rng.choice([base + 2*k for k in range((27-base)//2 + 1)],
                            p=None))
    equal = not witness and rng.random() < .23
    common_max = min((28-c)//c for c in counts)
    common = base + 2 * int(rng.integers((min(ceiling, common_max)-base)//2 + 1))
    lengths = []
    for c in counts:
        lane = [common] * c if equal else rng.choice(
            np.arange(base, ceiling + 1, 2), size=c).astype(int).tolist()
        if c == 1 and not equal and rng.random() < .35:
            lane[0] = ceiling
        budget = 28 - c  # one-cell margins and separators
        while sum(lane) > budget:
            indices = [i for i, length in enumerate(lane) if length > base]
            lane[int(rng.choice(indices))] -= 2
        rng.shuffle(lane)
        lengths.extend(lane)
    duplicate = None
    if witness:
        pivot = n - counts[-1] if ordered_witness else int(rng.integers(n))
        a, b = map(int, rng.choice([i for i in range(n) if i != pivot], size=2, replace=False))
        lengths[a] = lengths[b] = base
        duplicate = (a, b, pivot)
        if len(set(lengths)) == 1:
            eligible, start = [], 0
            for count in counts:
                if count * base + count + 1 + 2 <= 29:
                    eligible.extend(i for i in range(start, start+count) if i not in (a, b))
                start += count
            lengths[int(rng.choice(eligible))] = base + 2
    groups, start = [], 0
    for c in counts:
        groups.append(lengths[start:start+c])
        start += c
    # Witness lanes have enough room for the sole possible forced +2 above.
    minimum_fast = max(sum(group) + len(group) + 1 for group in groups)
    spare = 29 - minimum_fast
    extra = int(rng.integers(min(spare, 6) + 1) if rng.random() < .55
                else rng.integers(spare + 1))
    fast_extent = minimum_fast + extra
    slow_gaps, slow_extent = _gaps(rng, 2*lanes, lanes)
    placements = []
    v = slow_gaps[0]
    for lane_index, group in enumerate(groups):
        gaps, _ = _gaps(rng, sum(group), len(group), fast_extent)
        if ordered_witness and lane_index == lanes-1:
            # A uniquely patterned block here must move under fast-major order.
            gaps[-1] += gaps[0] - 1
            gaps[0] = 1
        u = gaps[0]
        for i, length in enumerate(group):
            placements.append((v, u, length))
            u += length + gaps[i+1]
        v += 2 + slow_gaps[lane_index+1]
    return lengths, placements, (slow_extent, fast_extent), duplicate


def _block(rng, length, palette, mode):
    local = rng.choice(palette, size=int(rng.integers(1, len(palette)+1)), replace=False)
    if mode == 0:
        return np.full((2, length), rng.choice(local), dtype=np.int8)
    if mode == 1:
        if rng.random() < .5:
            return np.repeat(rng.choice(local, size=(2, 1)), length, axis=1).astype(np.int8)
        return np.repeat(rng.choice(local, size=(1, length)), 2, axis=0).astype(np.int8)
    if mode == 2:
        motif = rng.choice(local, size=(2, int(rng.integers(1, min(4, length)+1))))
        return np.tile(motif, (1, length))[:, :length].astype(np.int8)
    return rng.choice(local, size=(2, length)).astype(np.int8)


def _asymmetric(block):
    return (not np.array_equal(block, block[::-1])
            and not np.array_equal(block, block[:, ::-1])
            and not np.array_equal(block, block[::-1, ::-1]))


def _payloads(rng, lengths, pool, duplicate=None, witness=False):
    size = int(rng.integers(3 if witness else 1, len(pool)+1))
    palette = rng.choice(pool, size=size, replace=False)
    solid = not witness and rng.random() < .30
    blocks = []
    repeat_rate = rng.uniform(.05, .6)
    for length in lengths:
        previous = [b for b in blocks if b.shape[1] == length]
        if previous and rng.random() < repeat_rate:
            blocks.append(previous[int(rng.integers(len(previous)))].copy())
        else:
            blocks.append(_block(rng, length, palette, 0 if solid else int(rng.integers(4))))
    if duplicate is not None:
        a, b, chosen = duplicate
        blocks[b] = blocks[a].copy()
    if witness:
        # Random location and random contents, with no shared signature.
        for _ in range(32):
            block = rng.choice(palette, size=blocks[chosen].shape).astype(np.int8)
            if _asymmetric(block) and all(not np.array_equal(block, b) for b in blocks):
                break
        else:
            # Enumerating four varying cells suffices for 15 possible payloads.
            for code in range(3**4):
                block = np.full(blocks[chosen].shape, palette[0], dtype=np.int8)
                digits = code
                for row, col in ((0, 0), (0, -1), (1, 0), (1, -1)):
                    block[row, col] = palette[digits % 3]
                    digits //= 3
                if _asymmetric(block) and all(not np.array_equal(block, b) for b in blocks):
                    break
            else:
                raise AssertionError('No asymmetric witness')
        blocks[chosen] = block
    return blocks


def _paint(blocks, placements, shape, configuration, roles):
    axis, bottom, right = configuration
    fast, slow, background, origin = roles
    interior = np.full(shape, background, dtype=np.int8)
    # Correct the thickness basis before the coordinate placement. The net
    # transform on each payload is a proper rotation, including bottom origins.
    reverse_thickness = (bottom != right) if axis == 0 else (bottom == right)
    for block, (v, u, length) in zip(blocks, placements):
        interior[v:v+2, u:u+length] = block[::-1] if reverse_thickness else block
    grid = np.full((shape[0]+1, shape[1]+1), background, dtype=np.int8)
    grid[1:, 1:] = interior
    grid[0, 0], grid[0, 1:], grid[1:, 0] = origin, fast, slow
    if axis == 1:
        grid = grid.T
    if bottom:
        grid = grid[::-1]
    if right:
        grid = grid[:, ::-1]
    return {'input': grid.copy(), 'output': _stack(blocks, background)}


def _key(grid):
    assert grid.ndim == 2 and min(grid.shape) >= 5, 'Input is too small for the key and separated payload'
    candidates = []
    for bottom in (False, True):
        for right in (False, True):
            aligned = grid[::-1] if bottom else grid
            aligned = aligned[:, ::-1] if right else aligned
            origin = int(aligned[0, 0])
            horizontal, vertical, bg = map(int, (aligned[0, 1], aligned[1, 0], aligned[-1, -1]))
            if len({origin, horizontal, vertical, bg}) != 4:
                continue
            if (np.all(aligned[0, 1:] == horizontal) and np.all(aligned[1:, 0] == vertical)
                    and np.all(aligned[-1, 1:] == bg) and np.all(aligned[1:, -1] == bg)):
                candidates.append((bottom, right, origin, horizontal, vertical, bg))
    assert len(candidates) == 1, 'The border must determine one coordinate key'
    return candidates[0]


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, points = [first], [first]
        while todo:
            r, c = todo.pop()
            for q in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    points.append(q)
        r0 = min(p[0] for p in points)
        r1 = max(p[0] for p in points) + 1
        c0 = min(p[1] for p in points)
        c1 = max(p[1] for p in points) + 1
        assert len(points) == (r1-r0)*(c1-c0), 'Payload must be a full rectangle'
        result.append((int(r0), int(r1), int(c0), int(c1)))
    return result


def _read_input(grid, fast_color):
    """Parse only visible input geometry; never consult an output label."""
    bottom, right, origin, horizontal, vertical, bg = _key(grid)
    assert fast_color in (horizontal, vertical)
    axis = int(fast_color == vertical)
    mask = grid != bg
    mask[-1 if bottom else 0, :] = False
    mask[:, -1 if right else 0] = False
    boxes = _components(mask)
    assert 1 <= len(boxes) <= 15
    lanes = {}
    for r0, r1, c0, c1 in boxes:
        assert 0 < r0 < r1 < grid.shape[0] and 0 < c0 < c1 < grid.shape[1]
        assert (c1-c0 if axis else r1-r0) == 2
        length = r1-r0 if axis else c1-c0
        assert length >= 2
        v = c0 if axis else r0
        lanes.setdefault(v, []).append((r0, r1, c0, c1))
    sorted_v = sorted(lanes)
    assert all(b-a >= 3 for a, b in zip(sorted_v, sorted_v[1:])), 'Staggered or overlapping lane bands'
    slow_reverse = right if axis else bottom
    fast_reverse = bottom if axis else right
    lane_order = sorted(lanes, reverse=slow_reverse)
    # np.rot90 maps the actual physical fast vector to right, with no reflection.
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    blocks, ordered_boxes, counts, strip_gaps = [], [], [], []
    for v in lane_order:
        boxes_in_lane = sorted(lanes[v], key=lambda b: b[0] if axis else b[2], reverse=fast_reverse)
        counts.append(len(boxes_in_lane))
        intervals = sorted((b[0], b[1]) if axis else (b[2], b[3]) for b in boxes_in_lane)
        assert all(b[0]-a[1] >= 1 for a, b in zip(intervals, intervals[1:]))
        strip_gaps.extend(b[0]-a[1] for a, b in zip(intervals, intervals[1:]))
        for r0, r1, c0, c1 in boxes_in_lane:
            blocks.append(np.rot90(grid[r0:r1, c0:c1], rotation).copy())
            ordered_boxes.append((r0, r1, c0, c1))
    lengths = [b.shape[1] for b in blocks]
    assert len({length % 2 for length in lengths}) == 1, 'Inexact centering'
    return {'axis': axis, 'bottom': bottom, 'right': right, 'origin': origin,
            'background': bg, 'fast': fast_color, 'slow': horizontal if axis else vertical,
            'blocks': blocks, 'boxes': ordered_boxes, 'lengths': lengths,
            'lane_counts': counts, 'strip_gaps': strip_gaps,
            'lane_gaps': [b-a-2 for a, b in zip(sorted_v, sorted_v[1:])],
            'payload_colors': set(map(int, np.concatenate(blocks, axis=1).flat))}


def _infer_convention(train_pairs):
    """Learn fast/slow color identities exclusively from training pairs."""
    first = _key(train_pairs[0]['input'])
    candidates = []
    for fast in first[3:5]:
        try:
            descriptors = [_read_input(p['input'], fast) for p in train_pairs]
            if len({d['slow'] for d in descriptors}) != 1:
                continue
            if all(np.array_equal(_stack(d['blocks'], d['background']), p['output'])
                   for d, p in zip(descriptors, train_pairs)):
                candidates.append((fast, descriptors[0]['slow']))
        except AssertionError:
            continue
    assert len(candidates) == 1, 'Training does not uniquely teach the guide identities'
    return candidates[0]


def solve(train_pairs, test_input):
    fast, slow = _infer_convention(train_pairs)
    descriptor = _read_input(test_input, fast)
    assert descriptor['slow'] == slow
    return _stack(descriptor['blocks'], descriptor['background'])


def _evidence(descriptors):
    assert {d['axis'] for d in descriptors} == {0, 1}
    assert {d['bottom'] for d in descriptors} == {False, True}
    assert {d['right'] for d in descriptors} == {False, True}
    assert any(len(set(d['lengths'])) > 1 for d in descriptors), 'Centering needs a demonstration'
    assert any(len({_grid_key(b) for b in d['blocks']}) < len(d['blocks']) for d in descriptors)
    for axis in (0, 1):
        assert any(d['axis'] == axis and any(_asymmetric(b) for b in d['blocks']) for d in descriptors)
    # Distinguish reasonable competing traversal rules using observed payloads.
    for alternative in ('fast_reverse', 'slow_reverse', 'fast_major'):
        distinguished = False
        for d in descriptors:
            groups, start = [], 0
            for count in d['lane_counts']:
                groups.append(list(range(start, start+count)))
                start += count
            if alternative == 'fast_reverse':
                order = [i for group in groups for i in group[::-1]]
            elif alternative == 'slow_reverse':
                order = [i for group in groups[::-1] for i in group]
            else:
                reverse = d['bottom'] if d['axis'] else d['right']
                coord = (0 if d['axis'] else 2) + int(reverse)
                order = sorted(range(len(d['blocks'])), key=lambda i: d['boxes'][i][coord], reverse=reverse)
            expected = _stack(d['blocks'], d['background'])
            alternative_output = _stack([d['blocks'][i] for i in order], d['background'])
            distinguished |= not np.array_equal(expected, alternative_output)
        assert distinguished, 'Training leaves a competing traversal rule unresolved'


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _grid_key(pair['input'])
            assert key not in seen and not _is_source_input(pair['input'])
            seen.add(key)
            assert not np.array_equal(pair['input'], pair['output'])
    fast, slow = _infer_convention(puzzle['train'])
    training = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            d = _read_input(pair['input'], fast)
            assert d['slow'] == slow
            expected = _stack(d['blocks'], d['background'])
            assert np.array_equal(expected, pair['output']), 'Incorrect copied, rotated, centered sequence'
            if split == 'train':
                training.append(d)
    _evidence(training)
    # Background and marker colors are locally geometric roles, not learned maps.
    all_keys = [_key(p['input']) for s in ('train', 'test') for p in puzzle[s]]
    assert len({k[5] for k in all_keys}) >= 2
    assert len({k[2] for k in all_keys}) >= 2


def _configurations(rng, train_count, test_count):
    configurations = [(axis, bottom, right) for axis in (0, 1)
                      for bottom in (False, True) for right in (False, True)]
    first = configurations[int(rng.integers(8))]
    second = (1-first[0], not first[1], not first[2])
    mode = int(rng.integers(5))
    train = [first, second]
    held = [c for c in configurations if c not in train]
    for _ in range(train_count-2):
        pool = train[:2] if mode == 1 else configurations
        if mode in (3, 4):
            pool = [c for c in configurations if c[0] == mode-3]
        train.append(pool[int(rng.integers(len(pool)))])
    if mode == 0:  # familiar configurations
        test = [train[int(rng.integers(len(train)))] for _ in range(test_count)]
    elif mode == 1:  # new combinations of demonstrated directions
        test = [held[int(rng.integers(len(held)))] for _ in range(test_count)]
    elif mode == 2:  # mixed query orientations when multiple queries are requested
        start_axis = int(rng.integers(2))
        test = [(int((start_axis+i) % 2), bool(rng.integers(2)), bool(rng.integers(2)))
                for i in range(test_count)]
    else:  # shift toward the less common training axis, in either direction
        pool = [c for c in configurations if c[0] == 4-mode]
        test = [pool[int(rng.integers(len(pool)))] for _ in range(test_count)]
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    fast, slow = map(int, rng.choice(10, size=2, replace=False))
    reserve_count = int(rng.choice([0, 1, 2, 3], p=[.50, .22, .18, .10]))
    reserved = set(map(int, rng.choice(10, size=reserve_count, replace=False)))
    configs = _configurations(rng, train_count, test_count)
    puzzle = {'train': [], 'test': []}
    used, familiar, local_roles = set(), set(), []
    for split, regime, config_list in zip(('train', 'test'), ('source_train', 'target_test'), configs):
        for i, configuration in enumerate(config_list):
            witness = regime == 'source_train' and i < 2
            local_pool = [c for c in range(10) if c not in (fast, slow)]
            if local_roles and i == 1 and split == 'train':
                local_pool = [c for c in local_pool if c not in local_roles[0]]
            background, origin = map(int, rng.choice(local_pool, size=2, replace=False))
            pool = [c for c in range(10) if c != background and
                    (regime != 'source_train' or c not in reserved)]
            if regime == 'target_test' and rng.random() < .45:
                pool = sorted(familiar - {background})
            lengths, placements, shape, duplicate = _layout(rng, witness)
            blocks = _payloads(rng, lengths, pool, duplicate, witness)
            if regime == 'target_test':
                novel = sorted(set(pool) - familiar)
                if novel and rng.random() < .7:
                    for color in rng.choice(novel, size=int(rng.integers(1, len(novel)+1)), replace=False):
                        block = blocks[int(rng.integers(len(blocks)))]
                        block[int(rng.integers(2)), int(rng.integers(block.shape[1]))] = color
            pair = _paint(blocks, placements, shape, configuration, (fast, slow, background, origin))
            # Deterministic content fallback, without banning source role colors.
            # Earlier inputs use at most 13 markers and the source adds at most 5.
            # Marker plus non-background recolorings offer 8*8 alternatives.
            if _grid_key(pair['input']) in used or _is_source_input(pair['input']):
                found = False
                for marker in [c for c in local_pool if c != background]:
                    for replacement in [c for c in range(10) if c != background]:
                        original = int(blocks[0][0, 0])
                        changed = [np.where(b == original, replacement,
                                            np.where(b == replacement, original, b)).astype(np.int8)
                                   for b in blocks]
                        candidate = _paint(changed, placements, shape, configuration,
                                           (fast, slow, background, marker))
                        if _grid_key(candidate['input']) not in used and not _is_source_input(candidate['input']):
                            pair, blocks, found = candidate, changed, True
                            break
                    if found:
                        break
                assert found
            used.add(_grid_key(pair['input']))
            puzzle[split].append(pair)
            local_roles.append((background, origin))
            if split == 'train':
                familiar.update(map(int, np.concatenate(blocks, axis=1).flat))
        if split == 'train':
            try:
                _evidence([_read_input(p['input'], fast) for p in puzzle['train']])
            except AssertionError:
                # Only an accidental traversal ambiguity triggers this bounded
                # fallback. A unique last-lane first block at the near margin
                # necessarily moves under each alternative traversal.
                old = _read_input(puzzle['train'][1]['input'], fast)
                lengths, placements, shape, duplicate = _layout(rng, True, True)
                pool = [c for c in range(10) if c != old['background'] and c not in reserved]
                blocks = _payloads(rng, lengths, pool, duplicate, True)
                pair = _paint(blocks, placements, shape, configs[0][1],
                              (fast, slow, old['background'], old['origin']))
                puzzle['train'][1] = pair
                familiar = set().union(*(_read_input(p['input'], fast)['payload_colors']
                                         for p in puzzle['train']))
                used = {_grid_key(p['input']) for p in puzzle['train']}
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def puzzle_features(puzzle):
    """Review measurements reconstructed from the returned grids."""
    fast, slow = _infer_convention(puzzle['train'])
    train_descriptors = [_read_input(p['input'], fast) for p in puzzle['train']]
    familiar = set().union(*(d['payload_colors'] for d in train_descriptors))
    training_configs = {(d['axis'], d['bottom'], d['right']) for d in train_descriptors}
    result = []
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            d = _read_input(pair['input'], fast)
            lengths, blocks = d['lengths'], d['blocks']
            maximum = max(lengths)
            identities = [_grid_key(b) for b in blocks]
            solids = [int(b.flat[0]) if np.all(b == b.flat[0]) else None for b in blocks]
            first_max = lengths.index(maximum)
            configuration = (d['axis'], d['bottom'], d['right'])
            # Distances include near/far clear space, measured in physical input.
            margins = []
            for r0, r1, c0, c1 in d['boxes']:
                margins.append([r0-int(not d['bottom']), pair['input'].shape[0]-r1-int(d['bottom']),
                                c0-int(not d['right']), pair['input'].shape[1]-c1-int(d['right'])])
            outer_margins = tuple(map(int, np.min(margins, axis=0)))
            patterns = []
            for block, solid in zip(blocks, solids):
                if solid is not None:
                    patterns.append('solid')
                elif np.array_equal(block[0], block[1]) or np.all(block == block[:, :1]):
                    patterns.append('striped')
                elif any(np.array_equal(block, np.tile(block[:, :period], (1, block.shape[1]))[:, :block.shape[1]])
                         for period in range(1, block.shape[1]//2+1)):
                    patterns.append('motif')
                else:
                    patterns.append('multicolor')
            result.append({
                'split': split, 'pair': index, 'guides': (fast, slow),
                'configuration': configuration, 'familiar_configuration': configuration in training_configs,
                'background': d['background'], 'origin': d['origin'],
                'input_shape': pair['input'].shape, 'output_shape': pair['output'].shape,
                'strip_count': len(blocks), 'lane_counts': tuple(d['lane_counts']),
                'lengths': tuple(lengths), 'maximum': maximum, 'parity': maximum % 2,
                'first_max': first_max, 'max_occurrences': lengths.count(maximum),
                'last_longest': lengths[-1] == maximum,
                'late_last_lane_max': first_max >= sum(d['lane_counts'][:-1]) and first_max > 0,
                'duplicate_occurrences': len(blocks)-len(set(identities)),
                'repeated_squares': len([k for k in identities if k[0] == (2, 2)]) >
                                    len({k for k in identities if k[0] == (2, 2)}),
                'same_color_different_lengths': any(solids[i] is not None and solids[i] == solids[j]
                                                    and lengths[i] != lengths[j]
                                                    for i in range(len(blocks)) for j in range(i)),
                'all_solid': all(c is not None for c in solids), 'equal_lengths': len(set(lengths)) == 1,
                'all_square': maximum == 2, 'palette_size': len(d['payload_colors']),
                'key_payload_colors': len(d['payload_colors'] & {fast, slow, d['origin']}),
                'new_colors': len(d['payload_colors']-familiar),
                'novel_positions': tuple(i for i, b in enumerate(blocks)
                                         if set(map(int, b.flat))-familiar),
                'asymmetric_positions': tuple(i for i, b in enumerate(blocks) if _asymmetric(b)),
                'patterns': tuple(patterns), 'outer_margins': outer_margins,
                'strip_gaps': tuple(d['strip_gaps']), 'lane_gaps': tuple(d['lane_gaps']),
                'largest_clear_margin': max(outer_margins),
                'layout': (configuration, pair['input'].shape, tuple(d['boxes']), tuple(lengths)),
            })
    return result


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _expect_invalid(puzzle):
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return
    raise AssertionError('An invalid puzzle was accepted')


def self_test() -> None:
    source = _source_pairs()
    # Predictions are inferred from the source training set, never the test label.
    assert _infer_convention(source[:4]) == (1, 2)
    for pair in source:
        for rotation in range(4):
            prediction = solve(source[:4], np.rot90(pair['input'], rotation).copy())
            assert np.array_equal(prediction, pair['output'])
    assert _read_input(source[2]['input'], 1)['lane_counts'] == [4, 3]
    assert _read_input(source[4]['input'], 1)['lane_counts'] == [5, 4]
    assert _read_input(source[4]['input'], 1)['lengths'].index(8) == 5
    recolor = np.array([8, 1, 2, 3, 4, 5, 6, 7, 0, 9], dtype=np.int8)
    combined = recolor[np.rot90(source[4]['input'], 2)]
    assert np.array_equal(solve(source[:4], combined), recolor[source[4]['output']])
    # Direct all-square, odd-length, large-width, and repeated-payload checks.
    for lengths in ([2, 2, 2, 2], [3, 7, 3], [4, 12, 26], [2]*15):
        if len(lengths) == 15:
            counts = [5, 5, 5]
        elif len(lengths) == 4:
            counts = [2, 2]
        else:
            counts = [1]*len(lengths)
        placements, start, v = [], 0, 1
        for count in counts:
            u = 1
            for length in lengths[start:start+count]:
                placements.append((v, u, length))
                u += length + 1
            start += count
            v += 3
        blocks = [np.full((2, length), 3, dtype=np.int8) for length in lengths]
        shape = (v, max(u+length+1 for _, u, length in placements))
        for axis in (0, 1):
            for bottom in (False, True):
                for right in (False, True):
                    pair = _paint(blocks, placements, shape, (axis, bottom, right), (1, 2, 8, 0))
                    assert np.array_equal(solve(source[:4], pair['input']), pair['output'])
    features, keys, train_counts, test_counts = [], set(), set(), set()
    within = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        keys.add(_puzzle_key(puzzle))
        rows = puzzle_features(puzzle)
        features.extend(rows)
        within.append({name: len({repr(r[name]) for r in rows}) for name in
                       ('background', 'origin', 'configuration', 'strip_count', 'maximum', 'palette_size', 'lane_counts')})
        for pair in puzzle['test']:
            predicted = solve([{'input': p['input'].copy(), 'output': p['output'].copy()}
                               for p in puzzle['train']], pair['input'].copy())
            assert np.array_equal(predicted, pair['output'])
    assert len(keys) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({r['layout'] for r in features}) >= 200
    assert len({r['guides'] for r in features}) >= 25
    assert len({r['configuration'] for r in features}) == 8
    assert len({r['input_shape'] for r in features}) >= 100
    assert len({r['output_shape'] for r in features}) >= 80
    for split in ('train', 'test'):
        rows = [r for r in features if r['split'] == split]
        assert min(r['strip_count'] for r in rows) <= 3 and max(r['strip_count'] for r in rows) == 15
        assert min(r['palette_size'] for r in rows) == 1 and max(r['palette_size'] for r in rows) >= 8
        for feature in ('all_solid', 'equal_lengths', 'all_square', 'repeated_squares',
                        'same_color_different_lengths', 'late_last_lane_max'):
            assert sum(bool(r[feature]) for r in rows) >= 3, (split, feature)
        assert any(len(r['lane_counts']) == 1 for r in rows)
        assert max(max(r['lane_counts']) for r in rows) >= 6
        assert sum(r['first_max'] > 0 for r in rows) > .35 * len(rows)
    for name in ('background', 'origin', 'configuration'):
        assert all(w[name] >= 2 for w in within)
    for name in ('strip_count', 'maximum', 'palette_size', 'lane_counts'):
        assert sum(w[name] >= 2 for w in within) >= 36
    assert {r['background'] for r in features} == set(range(10))
    assert {r['origin'] for r in features} == set(range(10))
    assert max(r['maximum'] for r in features) >= 25
    assert sum(r['parity'] for r in features) >= 40
    assert max(g for r in features for g in r['strip_gaps']) >= 8
    assert max(g for r in features for g in r['lane_gaps']) >= 8
    assert max(r['largest_clear_margin'] for r in features) >= 8
    assert {p for r in features for p in r['patterns']} == {'solid', 'striped', 'motif', 'multicolor'}
    assert any(r['max_occurrences'] >= 4 for r in features)
    tests = [r for r in features if r['split'] == 'test']
    assert any(r['new_colors'] >= 2 for r in tests) and sum(r['new_colors'] == 0 for r in tests) >= 20
    assert any(r['novel_positions'] and 0 not in r['novel_positions'] for r in tests)
    assert any(r['familiar_configuration'] for r in tests)
    assert any(not r['familiar_configuration'] for r in tests)
    assert sum(r['key_payload_colors'] > 0 for r in features) >= 100
    cases = [(2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None), (6, 2)]
    for i, (a, b) in enumerate(cases):
        puzzle = generate_puzzle(2000+i, a, b)
        validate_generated_puzzle(puzzle)
        assert a is None or len(puzzle['train']) == a
        assert b is None or len(puzzle['test']) == b
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(2000+i, a, b))
        assert _puzzle_key(puzzle) != _puzzle_key(generate_puzzle(3000+i, a, b))
    for a, b in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3, 1.5)]:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('Invalid counts accepted')
    damaged = generate_puzzle(700, 3, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    _expect_invalid(damaged)
    damaged = generate_puzzle(701, 3, 1)
    damaged['test'][0] = damaged['train'][0]
    _expect_invalid(damaged)
    # Reject mixed parity and holes without looking at a label.
    bad = _paint([np.ones((2, 2), dtype=np.int8), np.ones((2, 3), dtype=np.int8)],
                 [(1, 1, 2), (4, 1, 3)], (7, 6), (0, False, False), (1, 2, 8, 0))
    try:
        _read_input(bad['input'], 1)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mixed parity accepted')
    hole = source[0]['input'].copy()
    hole[2, 2] = 8
    try:
        _read_input(hole, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError('Incomplete rectangle accepted')
