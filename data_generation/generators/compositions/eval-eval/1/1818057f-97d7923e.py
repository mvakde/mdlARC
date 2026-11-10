"""Junction bridges: rank consecutive-cross intervals within cap-color families.

Only standard-library modules and NumPy are used. No files are read or written.
All inference, including the palette and orientation, comes from the input.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid grid color")
    return a


def _count(value, low, high, name):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    _require(low <= int(value) <= high, name + " out of range")
    return int(value)


def _runs(indices):
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if len(indices) else []


def _crosses(mask):
    result = np.zeros_like(mask, dtype=bool)
    result[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                           & mask[1:-1, :-2] & mask[1:-1, 2:])
    return result


def _parse(g):
    """Recognize a canonical scene: keys above vertical, bottom-aligned rails."""
    frequencies = np.bincount(g.ravel().astype(int), minlength=10)
    bg = int(np.argmax(frequencies))
    _require(np.count_nonzero(frequencies == frequencies[bg]) == 1, "ambiguous background")
    rails, remaining = [], []
    for col in range(g.shape[1]):
        for run in _runs(np.flatnonzero(g[:, col] != bg)):
            top, bottom = int(run[0]), int(run[-1])
            v = g[top:bottom + 1, col]
            if len(v) >= 3 and v[0] == v[-1] and v[0] != v[1] and np.all(v[1:-1] == v[1]):
                rails.append(dict(col=col, top=top, bottom=bottom, cap=int(v[0]), payload=int(v[1])))
            else:
                remaining.append((top, bottom, col, v))
    _require(bool(rails), "missing capped rails")
    controls = sorted({b['cap'] for b in rails})
    payloads = {b['payload'] for b in rails}
    _require(not payloads.intersection(controls), "overlapping cap and payload roles")
    _require(len({b['bottom'] for b in rails}) == 1, "rails need a common baseline")
    columns = sorted(b['col'] for b in rails)
    _require(all(b - a >= 4 for a, b in zip(columns, columns[1:])), "rails too close")
    _require(columns[0] >= 1 and columns[-1] < g.shape[1] - 1, "missing arm space")
    keys = {}
    for top, bottom, col, v in remaining:
        if int(v[0]) in controls:
            cap = int(v[0])
            _require(np.all(v == cap) and cap not in keys, "invalid unary key")
            keys[cap] = dict(top=top, col=col, rank=bottom - top + 1)
    _require(set(keys) == set(controls), "one key per cap required")
    _require(len({k['top'] for k in keys.values()}) == 1, "key starts must align")
    _require(all(1 <= k['rank'] <= 4 for k in keys.values()), "key rank outside 1..4")
    _require(max(k['top'] + k['rank'] for k in keys.values()) < min(b['top'] for b in rails),
             "key and rail band need a blank row")
    allowed = np.full(g.shape, bg, dtype=g.dtype)
    for cap, k in keys.items():
        allowed[k['top']:k['top'] + k['rank'], k['col']] = cap
    for b in rails:
        c, t, z, p = b['col'], b['top'], b['bottom'], b['payload']
        allowed[t:z + 1, c] = g[t:z + 1, c]
        # The only decorations are one-cell transverse half-arms. A complete
        # left/right pair at a payload cell is a station; a T is a near miss.
        for side in (c - 1, c + 1):
            v = g[t + 1:z, side]
            _require(np.all((v == bg) | (v == p)), "wrong transverse payload")
            allowed[t + 1:z, side] = v
        centers = _crosses(g == p)
        b['stations'] = [int(r) for r in np.flatnonzero(centers[:, c]) if t < r < z]
    _require(np.array_equal(g, allowed), "unrecognized marks")
    groups = {cap: [] for cap in controls}
    for b in rails:
        for a, z in zip(b['stations'], b['stations'][1:]):
            groups[b['cap']].append(dict(a=a, z=z, col=b['col'], distance=z - a,
                                         cap=b['cap'], rail=b))
    for cap, candidates in groups.items():
        distances = [x['distance'] for x in candidates]
        _require(len(distances) == len(set(distances)), "tied bridge lengths within a family")
        _require(len(candidates) >= keys[cap]['rank'], "requested bridge rank absent")
    return dict(bg=bg, rails=rails, keys=keys, groups=groups)


def _decode(g):
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(g, -rotation)
        try:
            data = _parse(canonical)
        except ValueError:
            continue
        candidates.append((rotation, canonical, data))
    _require(len(candidates) == 1, "scene needs a unique key/rail orientation")
    return candidates[0]


def _cells(bridge):
    a, z, c = bridge['a'], bridge['z'], bridge['col']
    cells = {(r, c) for r in range(a - 1, z + 2)}
    cells.update((r, c + side) for r in (a, z) for side in (-1, 1))
    return cells


def _chosen(data):
    return [sorted(candidates, key=lambda b: b['distance'], reverse=True)
            [data['keys'][cap]['rank'] - 1] for cap, candidates in data['groups'].items()]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill one virtual bridge per cap identity, including both endpoint crosses."""
    g = _grid(input_grid)
    rotation, canonical, data = _decode(g)
    out = canonical.copy()
    # All stations and all choices were determined before any cells change.
    for bridge in _chosen(data):
        for point in _cells(bridge):
            out[point] = bridge['cap']
    return np.ascontiguousarray(np.rot90(out, rotation))


def _signature(g):
    a = np.asarray(g, dtype=np.uint8)
    return a.shape, a.tobytes()


def _render(rng, profiles, ranks, bg):
    """Build geometry only. Profiles use positions relative to their top cap."""
    n = len(profiles)
    base_width = 3 + 4 * (n - 1)
    width = int(rng.integers(max(9, base_width), min(30, max(9, base_width) + 6) + 1))
    spare = width - base_width
    gaps = [4] * (n - 1)
    for i in rng.permutation(n - 1):
        extra = int(rng.integers(min(2, spare) + 1))
        gaps[int(i)] += extra
        spare -= extra
    left = 1 + int(rng.integers(spare + 1))
    columns = [left] + [left + int(v) for v in np.cumsum(gaps)]
    maximum = max(b['length'] for b in profiles)
    key_height = max(ranks.values())
    spare_y = 30 - (maximum + key_height + 1)
    _require(spare_y >= 0, "axis budget exceeded")
    padding = int(rng.integers(min(4, spare_y) + 1))
    key_y = int(rng.integers(padding + 1))
    baseline = key_y + key_height + maximum
    height = maximum + key_height + 1 + padding
    grid = np.full((height, width), bg, dtype=np.int8)
    key_order = list(rng.permutation(list(ranks)))
    key_step = int(rng.integers(1, 4))
    key_width = 1 + (len(key_order) - 1) * key_step
    key_x = int(rng.integers(width - key_width + 1))
    for index, cap in enumerate(key_order):
        grid[key_y:key_y + ranks[int(cap)], key_x + index * key_step] = cap
    order = rng.permutation(n)
    for col, index in zip(columns, order):
        b = profiles[int(index)]
        top = baseline - b['length'] + 1
        grid[top, col] = grid[baseline, col] = b['cap']
        grid[top + 1:baseline, col] = b['payload']
        for r in b['centers']:
            grid[top + r, col - 1] = grid[top + r, col + 1] = b['payload']
        for r, side in b['halves']:
            grid[top + r, col + side] = b['payload']
    return grid


def _profile(rng, cap, payload, length, centers, extra_halves=True):
    centers = sorted(map(int, centers))
    halves = []
    if extra_halves:
        available = [r for r in range(1, length - 1) if r not in centers]
        for r in rng.permutation(available)[:int(rng.integers(min(7, len(available)) + 1))]:
            halves.append((int(r), int(rng.choice((-1, 1)))))
    return dict(cap=cap, payload=payload, length=int(length), centers=centers, halves=halves)


def _anchors(rng, palette):
    bg, cap0, cap1, payload = palette[:4]
    profiles = []
    for cap in (cap0, cap1):
        largest = int(rng.integers(5, 10))
        gaps = [1, largest]
        if rng.integers(2):
            gaps.reverse()
        start = int(rng.integers(2, 4))
        centers = [start, start + gaps[0], start + sum(gaps)]
        short_length = centers[-1] + int(rng.integers(3, 5))
        long_length = int(rng.integers(max(short_length + 2, largest + 10), 25))
        short = _profile(rng, cap, payload, short_length, centers)
        long = _profile(rng, cap, payload, long_length, [2, 4], extra_halves=False)
        # A remote T would create a spurious bridge longer than the true winner
        # if one ignored the requirement for both transverse arms.
        remote = long_length - 3
        long['halves'] = [(r, s) for r, s in long['halves'] if r != remote]
        long['halves'].append((remote, int(rng.choice((-1, 1)))))
        profiles.extend((short, long))
    ranks = {cap0: 1, cap1: 2}
    first = _render(rng, profiles, ranks, bg)
    data = _parse(first)
    second = first.copy()
    for cap, key in data['keys'].items():
        r, c = key['top'], key['col']
        second[r:r + 2, c] = bg
        second[r:r + (3 - key['rank']), c] = cap
    return first, second


def _sample_family(rng, cap, payloads, rank, rail_limit, minimum, maximum):
    for _ in range(80):
        minimum_rails = 2 if minimum >= 5 else 1
        n = int(rng.integers(minimum_rails, rail_limit + 1))
        count = int(rng.integers(max(rank, minimum), maximum + 1))
        gaps = [int(x) for x in rng.choice(np.arange(1, 11), count, replace=False)]
        allocation = rng.multinomial(count, np.full(n, 1 / n))
        result, cursor = [], 0
        for amount in allocation:
            values = gaps[cursor:cursor + int(amount)]
            cursor += int(amount)
            if sum(values) > 19:
                break
            start = int(rng.integers(2, 23 - sum(values) - 1))
            centers = [start] + [start + int(v) for v in np.cumsum(values)]
            if not values and rng.random() < .4:
                centers = []
            length = int(rng.integers(max(5, (centers[-1] + 3) if centers else 5), 25))
            result.append(_profile(rng, cap, int(rng.choice(payloads)), length, centers))
        if len(result) == n:
            return result
    # Finite constructive fallback: round-robin small unique gaps, two or three
    # rails when required. At most six gaps need total capacity 21.
    count = max(rank, minimum)
    n = min(rail_limit, 3)
    _require(n > 1 or count <= 5, "fallback capacity")
    result = []
    for i in range(n):
        values = list(range(1, count + 1))[i::n]
        centers = [2] + [2 + int(v) for v in np.cumsum(values)]
        result.append(_profile(rng, cap, int(rng.choice(payloads)), max(8, centers[-1] + 3), centers))
    return result


def _random_scene(rng, palette, family_count, ranks, interval_range):
    bg = palette[0]
    controls = palette[1:1 + family_count]
    payloads = palette[4:4 + int(rng.integers(1, 4))]
    profiles = []
    for i, cap in enumerate(controls):
        minimum, maximum = interval_range
        minimum, maximum = max(minimum, ranks[i]), max(maximum, ranks[i])
        # Five or six distinct intervals need at least two short rails. Reserve
        # this capacity for every subsequent family, six rails in all.
        reserve = 2 if minimum >= 5 else 1
        limit = min(3, 6 - len(profiles) - reserve * (family_count - i - 1))
        # Six distinct positive distances cannot fit one rail of length <=24.
        family = _sample_family(rng, cap, payloads, ranks[i], limit, minimum, maximum)
        profiles.extend(family)
    return _render(rng, profiles, dict(zip(controls, ranks)), bg)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, 'seed must be a nonnegative integer')
    seed = int(seed)
    nt = _count(num_train, 2, 10, 'num_train')
    nv = _count(num_test, 1, 4, 'num_test')
    _require(nt > nv, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = [int(v) for v in rng.permutation(10)]
    rotation = int(rng.integers(4))
    shift = str(rng.choice(('interval_count', 'orientation')))
    mixed_palette = bool(rng.integers(2))
    train = list(_anchors(rng, palette))
    for _ in range(nt - 2):
        families = int(rng.integers(1, 4))
        ranks = [int(rng.integers(1, 5)) for _ in range(families)]
        train.append(_random_scene(rng, palette, families, ranks, (2, 4)))
    shown = {k['rank'] for g in train for k in _parse(g)['keys'].values()}
    max_families = max(len(_parse(g)['keys']) for g in train)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nv)):
        for index in range(count):
            for attempt in range(32):
                if split == 'train':
                    canonical = train[index].copy()
                else:
                    families = int(rng.integers(1, max_families + 1))
                    ranks = [int(rng.choice(sorted(shown))) for _ in range(families)]
                    interval_range = (5, 6) if shift == 'interval_count' else (2, 4)
                    canonical = _random_scene(rng, palette, families, ranks, interval_range)
                turn = (rotation + 2 * int(rng.integers(2))
                        + (int(split == 'test') if shift == 'orientation' else 0)) % 4
                g = np.rot90(canonical, turn)
                if rng.integers(2):
                    g = np.fliplr(g)
                if mixed_palette and (split == 'test' or index >= 2):
                    g = np.asarray(rng.permutation(10), dtype=np.int8)[g]
                g = np.ascontiguousarray(g)
                if _signature(g) not in seen:
                    break
            else:
                raise RuntimeError('bounded uniqueness attempts exhausted')
            seen.add(_signature(g))
            puzzle[split].append({'input': g, 'output': solve(g)})
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape, "dimensions changed")
    rotation, canonical, data = _decode(inp)
    answer = np.rot90(out, -rotation)
    expected_changes = set()
    for chosen in _chosen(data):
        cells = _cells(chosen)
        _require(len(cells) == chosen['distance'] + 7, "bridge cell-count invariant")
        _require(not expected_changes.intersection(cells), "selected bridges intersect")
        expected_changes.update(cells)
        _require(all(answer[p] == chosen['cap'] for p in cells), "wrong bridge fill")
        _require(all(canonical[p] == chosen['rail']['payload'] for p in cells), "non-payload changed")
        _require(not any(chosen['a'] < r < chosen['z'] for r in chosen['rail']['stations']),
                 "bridge skips a station")
    actual = {tuple(map(int, p)) for p in np.argwhere(canonical != answer)}
    _require(actual == expected_changes, "extra or omitted changed cells")
    return canonical, data


def _shortcut_witnesses(g, data):
    """Visible teaching evidence, not generator metadata or a second solver."""
    chosen = _chosen(data)
    no_key = any(data['keys'][x['cap']]['rank'] != 1 for x in chosen)
    physical = any(x['rail']['bottom'] - x['rail']['top'] < max(
        b['bottom'] - b['top'] for b in data['rails'] if b['cap'] == x['cap']) for x in chosen)
    skip = any(len(x['rail']['stations']) >= 3 and x['distance'] <
               x['rail']['stations'][-1] - x['rail']['stations'][0] for x in chosen)
    near_miss = False
    for x in chosen:
        if data['keys'][x['cap']]['rank'] != 1:
            continue
        for b in data['rails']:
            if b['cap'] != x['cap']:
                continue
            c, p = b['col'], b['payload']
            branches = [r for r in range(b['top'] + 2, b['bottom'] - 1)
                        if g[r, c - 1] == p or g[r, c + 1] == p]
            near_miss |= any(z - a > x['distance'] for a, z in zip(branches, branches[1:]))
    return dict(key=no_key, physical_rail=physical, consecutive=skip, plus=near_miss,
                overlap=any(x['distance'] <= 2 for x in chosen))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _count(len(puzzle['train']), 2, 10, 'num_train')
    _count(len(puzzle['test']), 1, 4, 'num_test')
    _require(len(puzzle['train']) > len(puzzle['test']), "invalid counts")
    seen, ranks, witnesses = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            key = _signature(_grid(pair['input']))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            g, data = _analyze(pair)
            current_ranks = {k['rank'] for k in data['keys'].values()}
            if split == 'train':
                ranks.update(current_ranks)
                witnesses.append(_shortcut_witnesses(g, data))
            else:
                _require(current_ranks <= ranks, "test ordinal not demonstrated")
    _require({1, 2} <= ranks, "training needs ranks one and two")
    for name in ('key', 'physical_rail', 'consecutive', 'overlap', 'plus'):
        _require(any(w[name] for w in witnesses), 'missing teaching witness: ' + name)


def self_test():
    """42 seeds streamed in memory; no artifacts, unbounded searches or traces."""
    signatures, shapes, distances, populations, ranks, rail_counts = set(), set(), set(), set(), set(), set()
    plus_witnesses = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                g, data = _analyze(pair)
                digest.update(np.asarray(pair['input'], dtype=np.uint8).tobytes())
                shapes.add(g.shape)
                rail_counts.add(len(data['rails']))
                for cap, candidates in data['groups'].items():
                    distances.update(x['distance'] for x in candidates)
                    populations.add(len(candidates))
                    ranks.add(data['keys'][cap]['rank'])
                if split == 'train':
                    plus_witnesses += _shortcut_witnesses(g, data)['plus']
        signatures.add(digest.digest())
        again = generate_puzzle(seed)
        assert all(np.array_equal(a[k], b[k]) for s in puzzle for a, b in zip(puzzle[s], again[s])
                   for k in ('input', 'output'))
    assert len(signatures) == 42 and len(shapes) >= 35
    assert {1, 2, 3, 4} <= ranks and {1, 2, 3, 4, 5, 6} <= rail_counts
    assert {2, 3, 4, 5, 6} <= populations and len(distances) >= 9
    assert plus_witnesses >= 20
    # Metamorphic checks cover every reflection/rotation and an arbitrary
    # palette permutation without any generation settings entering solve.
    pair = generate_puzzle(0)['train'][0]
    mapping = np.array([7, 0, 9, 2, 5, 1, 8, 4, 6, 3])
    for turns in range(4):
        for reflect in (False, True):
            a, b = (np.rot90(np.asarray(pair[k]), turns) for k in ('input', 'output'))
            if reflect:
                a, b = np.fliplr(a), np.fliplr(b)
            assert np.array_equal(solve(mapping[a]), mapping[b])
    # A false cross has exactly one missing cardinal cell; diagonals cannot
    # repair it. All 512 local neighborhoods exercise this literal condition.
    for bits in range(512):
        a = np.array([(bits >> j) & 1 for j in range(9)], dtype=bool).reshape(3, 3)
        assert bool(_crosses(a)[1, 1]) == all(a[p] for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)))
    # Explicit coordinate fixture: distance-five bridge on the shorter rail,
    # distance-two bridge on the longer rail, and a remote one-sided T.
    fixture = np.zeros((20, 11), dtype=np.int8)
    fixture[0, 0] = 2
    for col, top, stations in ((3, 5, (8, 9, 14)), (8, 3, (5, 7))):
        fixture[top, col] = fixture[19, col] = 2
        fixture[top + 1:19, col] = 5
        for r in stations:
            fixture[r, col - 1] = fixture[r, col + 1] = 5
    fixture[16, 7] = 5
    expected = fixture.copy()
    expected[8:16, 3] = 2
    expected[np.ix_([9, 14], [2, 4])] = 2
    assert np.array_equal(solve(fixture), expected)
    second = fixture.copy()
    second[1, 0] = 2
    expected_second = second.copy()
    expected_second[4:9, 8] = 2
    expected_second[np.ix_([5, 7], [7, 9])] = 2
    assert np.array_equal(solve(second), expected_second)
    assert not np.array_equal(expected != fixture, expected_second != second)
    broken = fixture.copy()
    broken[9, 2] = 0
    assert not np.array_equal(solve(broken) != broken, expected != fixture)
    padded = np.pad(fixture, ((1, 2), (2, 1)), constant_values=0)
    assert np.array_equal(solve(padded), np.pad(expected, ((1, 2), (2, 1)), constant_values=0))
    for nt in range(2, 11):
        for nv in range(1, min(4, nt - 1) + 1):
            generate_puzzle(8191, nt, nv)
    for nt, nv in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nv)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(0)
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
