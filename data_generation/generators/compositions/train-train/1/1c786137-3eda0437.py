"""Framed mosaics fill the largest compatible maximal vacancy.

Only NumPy and the standard library are used. No parent generator is imported.
Coordinates are half-open (top, left, height, width), except component boxes.
"""
from __future__ import annotations

from collections import deque
from math import gcd
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer in 2..10')
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError('num_test must be an integer in 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    if grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape):
        raise ValueError('grid must be two-dimensional with dimensions 1..30')
    if not np.issubdtype(grid.dtype, np.integer) or not np.all((0 <= grid) & (grid <= 9)):
        raise ValueError('grid must contain integer ARC colors')
    return grid


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _ratio(shape):
    divisor = gcd(*shape)
    return shape[0] // divisor, shape[1] // divisor


def _sources(grid):
    """Recognize isolated, filled, framed nonzero components geometrically.

    Terrain can contain the frame colors. A source is a whole 4-connected
    nonzero component, rather than an arbitrary monochromatic loop in terrain.
    """
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for start_r in range(height):
        for start_c in range(width):
            if seen[start_r, start_c] or grid[start_r, start_c] == 0:
                continue
            queue = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if (0 <= nr < height and 0 <= nc < width and
                            not seen[nr, nc] and grid[nr, nc] != 0):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells) + 1
            left = min(c for _, c in cells)
            right = max(c for _, c in cells) + 1
            h, w = bottom-top, right-left
            if h < 3 or w < 3 or len(cells) != h*w:
                continue
            tile = grid[top:bottom, left:right]
            border = int(tile[0, 0])
            if not (np.all(tile[0] == border) and np.all(tile[-1] == border)
                    and np.all(tile[:, 0] == border) and np.all(tile[:, -1] == border)):
                continue
            payload = tile[1:-1, 1:-1]
            if np.any(payload == border) or len(np.unique(payload)) < 2:
                continue
            result.append({'box': (top, left, h, w), 'color': border,
                           'payload': payload.copy()})
    return result


def _runs(row):
    padded = np.concatenate(([False], row, [False])).astype(np.int8)
    edges = np.diff(padded)
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _maximal_vacancies(grid):
    """All inclusion-maximal zero rectangles, with both dimensions >= 2.

    Horizontal runs already forbid left/right expansion. Checking the next
    rows forbids up/down expansion; these four tests imply full maximality.
    """
    mask = grid == 0
    height, width = grid.shape
    result = []
    for top in range(height-1):
        common = np.ones(width, dtype=bool)
        for bottom in range(top, height):
            common &= mask[bottom]
            if bottom == top:
                continue
            for left, right in _runs(common):
                left, right = int(left), int(right)
                if right-left < 2:
                    continue
                if top and np.all(mask[top-1, left:right]):
                    continue
                if bottom+1 < height and np.all(mask[bottom+1, left:right]):
                    continue
                result.append((top, left, bottom-top+1, right-left))
    return result


def _match(rect, source):
    h, w = rect[2:]
    ph, pw = source['payload'].shape
    if h % ph or w % pw:
        return None
    scale = h // ph
    return scale if scale >= 2 and w // pw == scale else None


def _interpret(grid):
    sources = _sources(grid)
    if not sources:
        raise ValueError('input has no framed mosaic')
    # Duplicate aspect ratios would leave an ambiguity about the content.
    ratios = [_ratio(s['payload'].shape) for s in sources]
    if len(ratios) != len(set(ratios)):
        raise ValueError('source aspect ratios must be distinct')
    vacancies = _maximal_vacancies(grid)
    eligible = []
    for rect in vacancies:
        matches = [(index, _match(rect, source)) for index, source in enumerate(sources)]
        matches = [(index, k) for index, k in matches if k is not None]
        if len(matches) > 1:
            raise ValueError('ambiguous source for a vacancy')
        if matches:
            index, scale = matches[0]
            eligible.append((rect[2]*rect[3], rect, index, scale))
    if not eligible:
        raise ValueError('input has no compatible maximal vacancy')
    best_area = max(item[0] for item in eligible)
    winners = [item for item in eligible if item[0] == best_area]
    if len(winners) != 1:
        raise ValueError('largest compatible vacancy must be unique')
    return sources, vacancies, eligible, winners[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only geometry, matching and copying; no generation metadata."""
    grid = _grid(input_grid)
    sources, _, _, (_, rect, index, scale) = _interpret(grid)
    top, left, height, width = rect
    payload = sources[index]['payload']
    enlarged = np.repeat(np.repeat(payload, scale, axis=0), scale, axis=1)
    output = grid.copy()
    output[top:top+height, left:left+width] = enlarged
    return output


_SHAPES = tuple((h, w) for h in range(1, 5) for w in range(1, 5) if h*w >= 2)


def _motif(rng, shape, border):
    colors = [c for c in range(1, 10) if c != border]
    count = int(rng.integers(2, min(5, shape[0]*shape[1])+1))
    palette = rng.choice(colors, count, replace=False)
    h, w = shape
    rows, cols = np.indices(shape)
    mode = int(rng.integers(4))
    if mode == 0:
        payload = rng.choice(palette, size=shape)
    elif mode == 1:
        payload = palette[(rows+cols+int(rng.integers(count))) % count].copy()
    elif mode == 2:
        payload = np.full(shape, palette[0], dtype=int)
        payload[rng.random(shape) < .4] = palette[1]
    else:
        payload = palette[(rows*max(1, count-1)+cols//2) % count].copy()
    # Two separated color samples make uniform filling, rotation and tiling
    # visibly different from copying enlarged pixels.
    payload.flat[0] = palette[0]
    payload.flat[-1] = palette[1]
    return payload.astype(np.int8)


def _pack(rng, height, width, shapes):
    """Random free placement of source-moat zones and empty rectangles."""
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][0]*shapes[i][1], reverse=True)
    occupied = np.zeros((height, width), dtype=bool)
    positions = [None] * len(shapes)
    for index in order:
        h, w = shapes[index]
        if h > height or w > width:
            return None
        blocked = np.lib.stride_tricks.sliding_window_view(
            np.pad(occupied, 1), (3, 3)).any(axis=(-2, -1))
        prefix = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        totals = prefix[h:, w:] - prefix[:-h, w:] - prefix[h:, :-w] + prefix[:-h, :-w]
        options = np.argwhere(totals == 0)
        if len(options) == 0:
            return None
        top, left = (int(v) for v in options[int(rng.integers(len(options)))])
        positions[index] = (top, left, h, w)
        occupied[top:top+h, left:left+w] = True
    return positions


def _draw_source(grid, rect, payload, color):
    top, left, h, w = rect
    # The packed zone includes a zero moat on all four sides.
    grid[top:top+h, left:left+w] = 0
    grid[top+1:top+h-1, left+1:left+w-1] = color
    grid[top+2:top+h-2, left+2:left+w-2] = payload


def _put_overlap(rng, grid, rect, protected):
    top, left, h, w = rect
    options = []
    if w >= 3:
        for length in range(2, w):
            for offset in range(w-length+1):
                for row in (top-1, top+h):
                    if 0 <= row < grid.shape[0]:
                        options.append((row, left+offset, 1, length))
    if h >= 3:
        for length in range(2, h):
            for offset in range(h-length+1):
                for col in (left-1, left+w):
                    if 0 <= col < grid.shape[1]:
                        options.append((top+offset, col, length, 1))
    rng.shuffle(options)
    for y, x, rh, rw in options:
        if not np.any(protected[y:y+rh, x:x+rw]):
            grid[y:y+rh, x:x+rw] = 0
            return True
    return False


def _put_line(rng, grid, minimum_length, protected):
    options = []
    h, w = grid.shape
    for axis, limit in ((0, w), (1, h)):
        if minimum_length > limit:
            continue
        length = int(rng.integers(minimum_length, min(limit, minimum_length+5)+1))
        rh, rw = (1, length) if axis == 0 else (length, 1)
        for y in range(h-rh+1):
            for x in range(w-rw+1):
                # Keep a one-cell moat of terrain around the line so that it
                # cannot turn into an extra two-dimensional candidate.
                section = (slice(max(0, y-1), min(h, y+rh+1)),
                           slice(max(0, x-1), min(w, x+rw+1)))
                if not np.any(protected[section]) and np.all(grid[section] != 0):
                    options.append((y, x, rh, rw))
    if not options:
        return False
    y, x, rh, rw = options[int(rng.integers(len(options)))]
    grid[y:y+rh, x:x+rw] = 0
    return True


def _features(grid, interpretation=None):
    sources, vacancies, eligible, winner = interpretation or _interpret(grid)
    area, rect, index, scale = winner
    top, left, h, w = rect
    overlap = any(other != rect and other[0] < top+h and top < other[0]+other[2]
                  and other[1] < left+w and left < other[1]+other[3]
                  for other in vacancies)
    mask = grid == 0
    mask[top:top+h, left:left+w] = False
    longest = max([int(b-a) for row in mask for a, b in _runs(row)] +
                  [int(b-a) for col in mask.T for a, b in _runs(col)] + [0])
    max_area = max(r[2]*r[3] for r in vacancies)
    source_areas = [s['payload'].size for s in sources]
    return {'rect': rect, 'scale': scale, 'source_shape': sources[index]['payload'].shape,
            'source_count': len(sources), 'source_areas': source_areas,
            'unused_larger_source': max(source_areas) > source_areas[index],
            'eligible_count': len(eligible), 'incompatible_larger': max_area > area,
            'overlap': overlap, 'long_line': longest > area,
            'boundary': top == 0 or left == 0 or top+h == grid.shape[0] or left+w == grid.shape[1],
            'frame_colors': tuple(s['color'] for s in sources)}


def _make_pair(rng, scale, profile):
    """Construct, then accept according to the finished pixels, not planting."""
    for attempt in range(240):
        # Required demonstrations select a middle-size source: both an unused
        # larger frame and a smaller compatible vacancy are present.
        if profile in ('fit', 'line_overlap'):
            target_pool = [s for s in _SHAPES if 3 <= s[0]*s[1] <= (6 if profile == 'line_overlap' else 12)]
        else:
            target_pool = [s for s in _SHAPES if 3 <= s[0]*s[1] <= 12]
        shape = target_pool[int(rng.integers(len(target_pool)))]
        smaller = [s for s in _SHAPES if s[0]*s[1] < shape[0]*shape[1] and _ratio(s) != _ratio(shape)]
        if not smaller:
            continue
        small = smaller[int(rng.integers(len(smaller)))]
        larger = [s for s in _SHAPES if s[0]*s[1] > shape[0]*shape[1]
                  and _ratio(s) not in (_ratio(shape), _ratio(small))]
        shapes = [shape, small]
        if profile in ('fit', 'line_overlap') or rng.random() < .55:
            if not larger:
                continue
            shapes.append(larger[int(rng.integers(len(larger)))])
        rh, rw = shape[0]*scale, shape[1]*scale
        rival_scale = 2
        rival_shape = (small[0]*rival_scale, small[1]*rival_scale)
        vacancy_shapes = [(rh, rw)]
        # Some examples are simpler: there may be only one eligible vacancy.
        want_rival = profile in ('fit', 'line_overlap') or rng.random() < .65
        if want_rival:
            vacancy_shapes.append(rival_shape)
        want_incompatible = profile == 'fit' or (profile == 'mixed' and rng.random() < .65)
        if want_incompatible:
            decoys = [(h, w) for h in range(2, 15) for w in range(2, 15)
                      if rh*rw < h*w <= int(rh*rw*1.65)+3
                      and not any(h % s[0] == 0 and w % s[1] == 0
                                  and h//s[0] == w//s[1] and h//s[0] >= 2 for s in shapes)]
            if not decoys:
                continue
            vacancy_shapes.append(decoys[int(rng.integers(len(decoys)))])
        zones = [(h+4, w+4) for h, w in shapes]
        all_shapes = zones + vacancy_shapes
        min_side = 23 if scale >= 3 else 19
        height, width = int(rng.integers(min_side, 31)), int(rng.integers(min_side, 31))
        if profile == 'line_overlap':
            # A run longer than the winner area must fit along one axis.
            if rh*rw >= 29:
                continue
            if rng.random() < .5:
                height = max(height, rh*rw+2)
            else:
                width = max(width, rh*rw+2)
        if max(height, width) > 30:
            continue
        positions = _pack(rng, height, width, all_shapes)
        if positions is None:
            continue
        colors = rng.choice(np.arange(1, 10), int(rng.integers(1, 7)), replace=False)
        rows, cols = np.indices((height, width))
        if rng.random() < .35:
            grid = colors[(rows+2*cols) % len(colors)].astype(np.int8)
        else:
            grid = rng.choice(colors, size=(height, width)).astype(np.int8)
        protected = np.zeros(grid.shape, dtype=bool)
        for i, zone in enumerate(positions[:len(zones)]):
            color = int(rng.integers(1, 10))
            _draw_source(grid, zone, _motif(rng, shapes[i], color), color)
            y, x, h, w = zone
            protected[max(0, y-1):min(height, y+h+1), max(0, x-1):min(width, x+w+1)] = True
        target = positions[len(zones)]
        for y, x, h, w in positions[len(zones):]:
            grid[y:y+h, x:x+w] = 0
        if profile in ('line_overlap', 'overlap', 'mixed'):
            if not _put_overlap(rng, grid, target, protected):
                continue
        if profile == 'line_overlap' and not _put_line(rng, grid, rh*rw+1, protected):
            continue
        # Sparse zero flecks can create irregular connected regions and new
        # maximal rectangles. No recipe-based assumption survives this check.
        if profile != 'line_overlap':
            noise = (rng.random(grid.shape) < rng.uniform(.015, .075)) & ~protected
            grid[noise] = 0
        try:
            info = _interpret(grid)
        except ValueError:
            continue
        if len(info[0]) != len(shapes) or info[3][1] != target or info[3][3] != scale:
            continue
        features = _features(grid, info)
        if profile == 'fit' and not (features['incompatible_larger'] and features['eligible_count'] >= 2
                                     and features['unused_larger_source']):
            continue
        if profile == 'line_overlap' and not (features['overlap'] and features['long_line']
                                              and features['eligible_count'] >= 2
                                              and features['unused_larger_source']):
            continue
        if profile in ('overlap', 'mixed') and not features['overlap']:
            continue
        output = grid.copy()
        source = info[0][info[3][2]]['payload']
        y, x, h, w = target
        output[y:y+h, x:x+w] = np.repeat(np.repeat(source, scale, axis=0), scale, axis=1)
        return {'input': grid, 'output': output}
    raise RuntimeError(f'failed to construct {profile} at scale {scale}')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    puzzle = {'train': [], 'test': []}
    used = set()
    train_profiles = ['fit', 'line_overlap'] + [str(rng.choice(('simple', 'overlap', 'mixed')))
                                              for _ in range(num_train-2)]
    rng.shuffle(train_profiles)
    for split, profiles in (('train', train_profiles),
                            ('test', [str(rng.choice(('simple', 'overlap', 'mixed')))
                                      for _ in range(num_test)])):
        for index, profile in enumerate(profiles):
            scale = 2 if split == 'train' else 3 if index == 0 else int(rng.integers(2, 4))
            for _ in range(10):
                pair = _make_pair(rng, scale, profile)
                signature = _key(pair['input'])
                if signature not in used:
                    used.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not produce distinct input')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check rule invariants and the promised discriminating demonstrations."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, output = _grid(pair['input']), _grid(pair['output'])
            assert grid.shape == output.shape
            assert _key(grid) not in seen, 'duplicate input'
            seen.add(_key(grid))
            info = _interpret(grid)
            sources, vacancies, eligible, (area, rect, index, scale) = info
            assert len(sources) >= 2, 'comparison needs multiple sources'
            top, left, h, w = rect
            changed = grid != output
            expected_mask = np.zeros(grid.shape, dtype=bool)
            expected_mask[top:top+h, left:left+w] = True
            assert np.array_equal(changed, expected_mask), 'only selected zeros may change'
            assert np.all(grid[expected_mask] == 0) and np.all(output[expected_mask] != 0)
            payload = sources[index]['payload']
            # Check each enlarged pixel, including all repetitions, separately.
            for r in range(payload.shape[0]):
                for c in range(payload.shape[1]):
                    block = output[top+r*scale:top+(r+1)*scale, left+c*scale:left+(c+1)*scale]
                    assert np.all(block == payload[r, c]), 'incorrect pixel expansion'
            assert len(np.unique(output[expected_mask])) >= 2, 'cannot be uniform highlighting'
            assert np.array_equal(solve(grid), output)
            assert area == h*w
            features[split].append(_features(grid, info))
    train = features['train']
    assert any(f['incompatible_larger'] and f['eligible_count'] >= 2 for f in train), 'missing compatibility/area witness'
    assert any(f['unused_larger_source'] for f in train), 'largest-source shortcut unchallenged'
    assert any(f['overlap'] and f['long_line'] for f in train), 'missing maximal-rectangle and thin-run witness'
    assert all(f['scale'] == 2 for f in train)
    assert any(f['scale'] == 3 for f in features['test']), 'missing scale extrapolation'
    return None


def _puzzle_key(puzzle):
    return tuple((split, _key(pair['input']), _key(pair['output']))
                 for split in ('train', 'test') for pair in puzzle[split])


def self_test():
    """Bounded 42-seed coverage, count validation and semantic regressions."""
    shapes, target_shapes, source_shapes, palettes, frame_colors = set(), set(), set(), set(), set()
    source_counts, scales, structures, puzzle_keys = set(), set(), set(), set()
    coverage = {'boundary': 0, 'overlap': 0, 'incompatible_larger': 0, 'long_line': 0,
                'unused_larger_source': 0, 'multiple_eligible': 0, 'thin_source': 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzle_keys.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid = pair['input']
                info = _interpret(grid)
                f = _features(grid, info)
                shapes.add(grid.shape)
                target_shapes.add(f['rect'][2:])
                source_counts.add(f['source_count'])
                scales.add(f['scale'])
                palettes.add(tuple(int(c) for c in np.unique(grid)))
                frame_colors.update(f['frame_colors'])
                source_shapes.update(s['payload'].shape for s in info[0])
                roles = {}
                structures.add((grid.shape, tuple(roles.setdefault(int(c), len(roles)) for c in grid.flat)))
                for name in ('boundary', 'overlap', 'incompatible_larger', 'long_line', 'unused_larger_source'):
                    coverage[name] += int(f[name])
                coverage['multiple_eligible'] += int(f['eligible_count'] >= 2)
                coverage['thin_source'] += int(1 in f['source_shape'])
                # A solid fill ablation must differ on every example.
                top, left, h, w = f['rect']
                assert len(np.unique(pair['output'][top:top+h, left:left+w])) >= 2
                # Copying the source periodically is different from magnifying.
                payload = info[0][info[3][2]]['payload']
                tiled = np.tile(payload, (f['scale'], f['scale']))
                actual = pair['output'][top:top+h, left:left+w]
                assert not np.array_equal(tiled, actual)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), 'not deterministic'
    assert len(puzzle_keys) == 42 and len(structures) >= 200
    assert len(shapes) >= 40 and len(target_shapes) >= 20 and len(source_shapes) >= 12
    assert len(palettes) >= 15 and len(frame_colors) == 9
    assert source_counts == {2, 3} and scales == {2, 3}
    assert all(count >= 10 for count in coverage.values()), coverage
    for num_train in range(2, 11):
        for num_test in range(1, min(4, num_train-1)+1):
            puzzle = generate_puzzle(1717+num_train*10+num_test, num_train, num_test)
            assert (len(puzzle['train']), len(puzzle['test'])) == (num_train, num_test)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted illegal counts {a}/{b}')
    # A hand-built family member extends beyond the sampler: scale 4 and a
    # source touching two canvas edges. Its expected content is explicit.
    edge = np.full((16, 25), 8, dtype=int)
    edge[:4, :5] = 0
    edge[:3, :4] = 3
    edge[1, 1:3] = [1, 2]
    edge[:6, 7:12] = 0
    edge[1:5, 8:11] = 4
    edge[2:4, 9] = [5, 6]
    edge[8:12, :8] = 0
    edge[10:14, 16:18] = 0
    expected = edge.copy()
    expected[8:12, :4] = 1
    expected[8:12, 4:8] = 2
    assert np.array_equal(solve(edge), expected)
    # An equal-area compatible opening must not be resolved by reading order.
    edge[8:12, 11:19] = 0
    try:
        solve(edge)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted a greatest-area tie')
    # Inputs without framed mosaics are outside the family.
    for bad in (np.zeros((5, 5), dtype=int), np.ones((5, 5), dtype=int)):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted input without a framed source')
    return {'seeds': 42, 'pairs': 210, 'input_shapes': len(shapes),
            'target_shapes': len(target_shapes), 'source_shapes': len(source_shapes),
            'palettes': len(palettes), 'normalized_structures': len(structures),
            'coverage': coverage}
