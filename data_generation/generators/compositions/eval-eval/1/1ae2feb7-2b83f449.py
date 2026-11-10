"""Competing run-length clocks on locally switched, nonbranching cables.

Only NumPy and the standard library are used. No file IO occurs in generation,
validation, solving, or self_test. Every semantic parameter is input-visible.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


_DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))
_SEEDS = (0, 7482550, 243214643, 426476001, 37356118, 184212085,
          671134014, 583788815, 240947520, 216316618, 888567094,
          52454996, 92668880, 376892157, 79818653, 405988096,
          395889095, 678039812, 535065019, 176414016, 901355823,
          785961730, 256122753, 141096464, 794743089, 389552351,
          731073584, 852914724, 978126852, 816152030, 760268403,
          834139691, 601099584, 572348747, 591583450, 604307009,
          970595676, 171972076, 699408340, 678195573, 275190275,
          296528167)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _neighbors(mask, cell):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in _DIRS
            if 0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1]
            and mask[r + dr, c + dc]]


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, todo = [], [start]
        while todo:
            cell = todo.pop()
            component.append(cell)
            for neighbor in _neighbors(mask, cell):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    todo.append(neighbor)
        result.append(component)
    return result


def _paths(mask):
    result = []
    for cells in _components(mask):
        degrees = {cell: len(_neighbors(mask, cell)) for cell in cells}
        assert all(d in (1, 2) for d in degrees.values()), 'branch or isolated cell'
        ends = sorted(cell for cell, degree in degrees.items() if degree == 1)
        assert len(ends) == 2, 'closed cable'
        path, previous = [ends[0]], None
        while path[-1] != ends[1]:
            choices = [p for p in _neighbors(mask, path[-1]) if p != previous]
            assert len(choices) == 1
            previous = path[-1]
            path.append(choices[0])
        assert len(path) == len(cells)
        result.append(path)
    return result


def _switches(core, bg, track, gate):
    """Simultaneous writes; shared tile cells must agree."""
    centers, writes, overlap = [], {}, 0
    for r in range(1, core.shape[0], 2):
        xs = np.flatnonzero(core[r] != bg).tolist()
        assert all(core[r, c] == gate for c in xs)
        for _, group in itertools.groupby(enumerate(xs), key=lambda v: v[1] - v[0]):
            run = [v for _, v in group]
            assert len(run) == 3 and 0 < run[0] and run[-1] < core.shape[1] - 1
            c = run[1]
            assert np.all(core[r - 1, c - 1:c + 2] == track)
            assert np.all(core[r + 1, c - 1:c + 2] == track)
            centers.append((r, c))
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    value = gate if cc == c else track
                    if (rr, cc) in writes:
                        assert writes[rr, cc] == value, 'conflicting switches'
                        overlap += 1
                    writes[rr, cc] = value
    assert centers
    return centers, writes, overlap


def _decode_core(core, bg):
    h, w = core.shape
    assert h >= 3 and h % 2 == 1 and w >= 7
    gap_colors = set(map(int, np.unique(core[1::2]))) - {bg}
    assert len(gap_colors) == 1
    gate = next(iter(gap_colors))
    r, c = map(int, np.argwhere(core[1::2] == gate)[0])
    track = int(core[2 * r, c + 1])  # Above the center of the first three-bar.
    assert len({bg, gate, track}) == 3
    assert not np.any(core[::2] == gate)
    assert np.all(core[::2, 0] != bg) and np.all(core[::2, -1] != bg)
    centers, writes, overlap = _switches(core, bg, track, gate)
    structural = np.full(core.shape, bg, dtype=np.int8)
    structural[::2] = np.where(core[::2] == bg, bg, track)
    for cell, value in writes.items():
        structural[cell] = value
    paths = _paths(structural == track)
    packets = []
    packet_cells = set()
    for r in range(0, h, 2):
        for color, group in itertools.groupby(range(w), key=lambda x: int(core[r, x])):
            cols = list(group)
            if color in (bg, track):
                continue
            assert 1 <= len(cols) <= 6
            cells = [(r, x) for x in cols]
            assert all(cell not in writes for cell in cells), 'packet in a switch tile'
            packets.append((int(color), cells))
            packet_cells.update(cells)
    assert packets
    sources = {}
    for i, path in enumerate(paths):
        ends = [None, None]
        for color, cells in packets:
            count = len(cells)
            occupied = set(cells)
            if occupied <= set(path):
                if occupied == set(path[:count]):
                    assert ends[0] is None
                    ends[0] = (count, color)
                elif occupied == set(path[-count:]):
                    assert ends[1] is None
                    ends[1] = (count, color)
                else:
                    raise AssertionError('packet must occupy an exposed straight end')
        total = sum(s[0] for s in ends if s)
        assert total < len(path), 'no cable between packets'
        sources[i] = ends
        for end in (path[0], path[-1]):
            rr, cc = end
            assert rr % 2 == 0
            assert cc in (0, w - 1) or core[rr, cc - 1] == bg or core[rr, cc + 1] == bg
    reconstructed = {cell for i, p in enumerate(paths) for k, s in enumerate(sources[i]) if s
                     for cell in (p[:s[0]] if k == 0 else p[-s[0]:])}
    assert reconstructed == packet_cells
    return {'core': core, 'background': bg, 'track': track, 'gate': gate,
            'structural': structural, 'paths': paths, 'sources': sources,
            'packet_cells': packet_cells, 'centers': centers, 'overlap': overlap,
            'breaks': int(np.count_nonzero(core[::2] == bg))}


def _inspect(value):
    a = _grid(value)
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), 'background frame required'
    cells = np.argwhere(a != bg)
    assert len(cells)
    r0, c0 = map(int, cells.min(axis=0))
    r1, c1 = map(int, cells.max(axis=0) + 1)
    candidates = []
    crop = a[r0:r1, c0:c1]
    for axis in (0, 1):
        try:
            info = _decode_core(crop if axis == 0 else crop.T, bg)
        except (AssertionError, IndexError):
            continue
        info.update(axis=axis, box=(r0, r1, c0, c1), shape=a.shape)
        candidates.append(info)
    assert len(candidates) == 1, 'cable axis must be unique'
    return candidates[0]


def _restore(core, info):
    a = np.full(info['shape'], info['background'], dtype=np.int8)
    r0, r1, c0, c1 = info['box']
    a[r0:r1, c0:c1] = core if info['axis'] == 0 else core.T
    return a


def _paint(structural, paths, sources, track, variant='rule'):
    """Construction/synthesis: emit each source, then resolve wave overlaps."""
    out = structural.copy()
    for number, path in enumerate(paths):
        left, right = sources[number]
        lo, hi = (left[0] if left else 0), len(path) - (right[0] if right else 0)
        for i, cell in enumerate(path):
            if i < lo:
                out[cell] = left[1]
            elif i >= hi:
                out[cell] = right[1]
        arrivals = {}
        for side, source in enumerate((left, right)):
            if source is None:
                continue
            period, color = source
            launch = period if side == 0 else len(path) - 1 - period
            delta = 1 if side == 0 else -1
            spacing = 1 if variant == 'solid' else period + (variant == 'period_plus_one')
            if variant == 'manhattan':
                origin = path[period - 1] if side == 0 else path[-period]
                for index in range(lo, hi):
                    target = path[index]
                    distance = abs(origin[0] - target[0]) + abs(origin[1] - target[1])
                    if distance > 0 and (distance - 1) % period == 0:
                        arrivals.setdefault(index, []).append((distance, period, color, side))
                continue
            for offset in range(0, len(path), spacing):
                index = launch + delta * offset
                if lo <= index < hi:
                    distance = offset + 1
                    arrivals.setdefault(index, []).append((distance, period, color, side))
        for index, hits in arrivals.items():
            if variant == 'nearest_owner':
                distances = ([index - left[0] + 1] if left else []) + (
                    [len(path) - right[0] - index] if right else [])
                hits = [hit for hit in hits if hit[0] == min(distances)]
                if not hits:
                    continue
            if variant == 'shortest':
                chosen = min(hits, key=lambda h: (h[1], h[3]))
            elif variant == 'longest':
                chosen = min(hits, key=lambda h: (-h[1], h[3]))
            elif variant == 'fixed_end':
                chosen = min(hits, key=lambda h: h[3])
            else:
                hits.sort()
                if len(hits) == 2 and hits[0][0] == hits[1][0] and variant != 'no_cancel':
                    continue
                chosen = hits[0]
            if structural[path[index]] == track:
                out[path[index]] = chosen[2]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _inspect(input_grid)
    return _restore(_paint(info['structural'], info['paths'], info['sources'], info['track']), info)


def _no_switch(info):
    """Rival: preserve switch rendering but send clocks along original rows."""
    core, paths, sources = info['core'], [], {}
    for r in range(0, core.shape[0], 2):
        for occupied, group in itertools.groupby(range(core.shape[1]), key=lambda c: core[r, c] != info['background']):
            cols = list(group)
            if not occupied:
                continue
            path = [(r, c) for c in cols]
            ends = []
            for p in (path, path[::-1]):
                color = int(core[p[0]])
                if color == info['track']:
                    ends.append(None)
                else:
                    count = 0
                    for cell in p:
                        if core[cell] != color:
                            break
                        count += 1
                    ends.append((count, color))
            sources[len(paths)] = ends
            paths.append(path)
    return _paint(info['structural'], paths, sources, info['track'])


def _base(lanes, width, centers, breaks, palette):
    bg, track, gate = palette[:3]
    core = np.full((2 * lanes - 1, width), bg, dtype=np.int8)
    core[::2] = track
    for cell in breaks:
        core[cell] = bg
    for r, c in centers:
        core[r, c - 1:c + 2] = gate
    return core


def _network(rng, lanes, width, palette, rich):
    centers, previous = [], []
    for r in range(1, 2 * lanes - 1, 2):
        wanted = int(rng.integers(0, 4 if rich else 3))
        chosen = []
        for c in rng.permutation(np.arange(3, width - 3)):
            c = int(c)
            if len(chosen) == wanted:
                break
            if all(abs(c - x) >= 4 for x in chosen) and all(abs(c - x) >= 2 for x in previous):
                chosen.append(c)
        centers.extend((r, c) for c in sorted(chosen))
        previous = chosen
    if not centers:
        centers = [(1, int(rng.integers(5, width - 5)))]
    core = _base(lanes, width, centers, [], palette)
    _, writes, _ = _switches(core, *palette[:3])
    structural = core.copy()
    for cell, value in writes.items():
        structural[cell] = value
    breaks = []
    # A cycle can be opened at an ordinary straight cable cell. Unlike B's
    # gravity model, this family also permits winding, nonmonotone paths.
    for component in _components(structural == palette[1]):
        degrees = [len(_neighbors(structural == palette[1], cell)) for cell in component]
        assert all(d in (1, 2) for d in degrees)
        if all(d == 2 for d in degrees):
            options = [(r, c) for r, c in component if r % 2 == 0 and 1 < c < width - 2
                       and (r, c) not in writes
                       and structural[r, c - 1] == structural[r, c + 1] == palette[1]]
            if not options:
                return None
            cell = options[int(rng.integers(len(options)))]
            structural[cell] = palette[0]
            breaks.append(cell)
    for _ in range(int(rng.integers(0, 5 if rich else 3))):
        mask = structural == palette[1]
        options = [(r, c) for r in range(0, core.shape[0], 2) for c in range(2, width - 2)
                   if (r, c) not in writes and mask[r, c] and mask[r, c - 1] and mask[r, c + 1]
                   and len(_neighbors(mask, (r, c - 1))) == len(_neighbors(mask, (r, c + 1))) == 2]
        if options:
            cell = options[int(rng.integers(len(options)))]
            structural[cell] = palette[0]
            breaks.append(cell)
    core = _base(lanes, width, centers, breaks, palette)
    return core, structural, _paths(structural == palette[1]), set(writes)


def _capacity(path, forbidden):
    result = 0
    row = path[0][0]
    for cell in path:
        if cell[0] != row or cell in forbidden:
            break
        result += 1
    return min(6, result)


def _events(n, left, right):
    result = {'left': 0, 'right': 0, 'tie': 0, 'alone_left': 0, 'alone_right': 0, 'far_alone': 0}
    for i in range(left, n - right):
        a, b = i % left == 0, (n - 1 - i) % right == 0
        if a and b:
            d1, d2 = i - left + 1, n - right - i
            result['left' if d1 < d2 else 'right' if d2 < d1 else 'tie'] += 1
        elif a:
            result['alone_left'] += 1
            result['far_alone'] += i - left + 1 > n - right - i
        elif b:
            result['alone_right'] += 1
            result['far_alone'] += n - right - i > i - left + 1
    return result


def _decorate(rng, network, palette, lesson, periods, dense):
    core, structural, paths, forbidden = network
    colors = list(palette[3:])
    sources = {i: [None, None] for i in range(len(paths))}
    capacities = [(_capacity(p, forbidden), _capacity(p[::-1], forbidden)) for p in paths]
    curved = [i for i, p in enumerate(paths) if len({r for r, c in p}) > 1]
    used = set()
    if lesson == 'competition':
        options = []
        for i in curved:
            a, b = capacities[i]
            for x in periods:
                for y in periods:
                    if min(x, y) < 2 or x == y or x > a or y > b:
                        continue
                    e = _events(len(paths[i]), x, y)
                    if all(e[k] for k in ('left', 'right', 'alone_left', 'alone_right', 'far_alone')):
                        options.append((i, x, y))
        if not options:
            return None
        i, x, y = options[int(rng.integers(len(options)))]
        sources[i] = [(x, colors[0]), (y, colors[1])]
        used.add(i)
    elif lesson == 'cancellation':
        options = [(i, x) for i in curved for x in periods
                   if 2 <= x <= min(capacities[i]) and _events(len(paths[i]), x, x)['tie']]
        if not options:
            return None
        i, x = options[int(rng.integers(len(options)))]
        sources[i] = [(x, colors[0]), (x, colors[1])]
        used.add(i)
    # At least one visibly recurring lone clock on a curved path in the
    # competition lesson; all other pairs also get an active curved path.
    options = [(i, side, x) for i in curved if i not in used
               for side in (0, 1) for x in periods
               if 2 <= x <= capacities[i][side] and len(paths[i]) >= 3 * x + 1]
    if options:
        i, side, x = options[int(rng.integers(len(options)))]
        sources[i][side] = (x, colors[int(rng.integers(len(colors)))])
        used.add(i)
    elif lesson == 'competition' or not used:
        return None
    probability = .83 if dense else float(rng.uniform(.25, .65))
    for i, path in enumerate(paths):
        if i in used:
            continue
        for side in (0, 1):
            available = [x for x in periods if x <= capacities[i][side]]
            if available and rng.random() < probability:
                x = int(rng.choice(available))
                other = sources[i][1 - side]
                if x + (other[0] if other else 0) + 2 < len(path):
                    sources[i][side] = (x, int(rng.choice(colors)))
    inp = core.copy()
    for i, path in enumerate(paths):
        for side, source in enumerate(sources[i]):
            if source:
                length, color = source
                for cell in (path[:length] if side == 0 else path[-length:]):
                    inp[cell] = color
    out = _paint(structural, paths, sources, palette[1])
    if lesson == 'competition' and np.array_equal(
            out, _paint(structural, paths, sources, palette[1], 'manhattan')):
        return None
    return inp, out


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 7, replace=False)))
    shift = int(rng.integers(3))
    base_turn = int(rng.integers(4))
    train_max = int(rng.integers(3, 7))
    lesson_slots = list(map(int, rng.permutation(num_train)[:2]))
    lessons = {lesson_slots[0]: 'competition', lesson_slots[1]: 'cancellation'}
    puzzle, seen, taught = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            lesson = lessons.get(index, 'ordinary') if split == 'train' else 'ordinary'
            periods = list(range(1, 7)) if split == 'train' else sorted(taught)
            for attempt in range(240):
                lanes = int(rng.integers(2, train_max + 1))
                if split == 'test' and shift == 0:
                    lanes = int(rng.integers(train_max + 1, min(12, train_max + 5)))
                width = int(rng.integers(17, 29))
                network = _network(rng, lanes, width, palette,
                                   rich=split == 'test' or rng.random() < .4)
                if network is None:
                    continue
                pair = _decorate(rng, network, palette, lesson, periods,
                                 dense=split == 'test' and shift == 1)
                if pair is None:
                    continue
                inp, out = pair
                # All presentation changes are input-visible; no gravity bit.
                if rng.random() < .5:
                    inp, out = np.fliplr(inp), np.fliplr(out)
                turn = base_turn if shift == 2 and split == 'train' else (
                    (base_turn + 1) % 4 if shift == 2 else int(rng.integers(4)))
                margins = []
                for size in inp.shape:
                    total = int(rng.integers(2, min(5, 30 - size) + 1))
                    before = int(rng.integers(1, total))
                    margins.append((before, total - before))
                inp, out = (np.rot90(np.pad(a, margins, constant_values=palette[0]), turn).copy()
                            for a in (inp, out))
                key = (inp.shape, inp.tobytes())
                if key in seen:
                    continue
                info = _inspect(inp)
                r0, r1, c0, c1 = info['box']
                canonical_out = out[r0:r1, c0:c1]
                if info['axis']:
                    canonical_out = canonical_out.T
                if np.array_equal(_no_switch(info), canonical_out):
                    continue
                seen.add(key)
                puzzle[split].append({'input': inp, 'output': out})
                if split == 'train':
                    taught.update(s[0] for ss in info['sources'].values() for s in ss if s)
                break
            else:
                raise RuntimeError('bounded construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _cell_check(info, output):
    """Validate direct congruences, separately from the emitter painter."""
    expected = info['structural'].copy()
    stats = {'single': 0, 'dual': 0, 'left': 0, 'right': 0, 'tie': 0,
             'alone_left': 0, 'alone_right': 0, 'empty': 0}
    for number, path in enumerate(info['paths']):
        a, b = info['sources'][number]
        stats['single' if bool(a) != bool(b) else 'dual' if a else 'empty'] += 1
        lo, hi = a[0] if a else 0, len(path) - (b[0] if b else 0)
        for i, cell in enumerate(path):
            if i < lo:
                value = a[1]
            elif i >= hi:
                value = b[1]
            else:
                firing_a = a is not None and i % a[0] == 0
                firing_b = b is not None and (len(path) - 1 - i) % b[0] == 0
                value = info['track']
                if firing_a and firing_b:
                    da, db = i - a[0] + 1, len(path) - b[0] - i
                    stats['left' if da < db else 'right' if db < da else 'tie'] += 1
                    if da != db:
                        value = a[1] if da < db else b[1]
                elif firing_a:
                    stats['alone_left'] += 1
                    value = a[1]
                elif firing_b:
                    stats['alone_right'] += 1
                    value = b[1]
            expected[cell] = value
    assert np.array_equal(output, expected), 'switch, period, phase, competition, cancellation, or preservation'
    for cell in info['packet_cells']:
        assert output[cell] == info['core'][cell], 'packet changed'
    assert set(map(int, np.unique(output))) <= set(map(int, np.unique(info['core'])))
    return stats


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles, train_periods, test_periods, evidence = set(), set(), set(), set(), set()
    rivals = {'solid', 'period_plus_one', 'shortest', 'longest', 'fixed_end', 'no_cancel',
              'manhattan', 'nearest_owner', 'no_switch'}
    refuted = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            info = _inspect(inp)
            r0, r1, c0, c1 = info['box']
            crop = out[r0:r1, c0:c1]
            canonical = crop if info['axis'] == 0 else crop.T
            assert np.array_equal(_restore(canonical, info), out), 'padding changed'
            stats = _cell_check(info, canonical)
            roles.add((info['background'], info['track'], info['gate']))
            periods = {s[0] for ss in info['sources'].values() for s in ss if s}
            (train_periods if split == 'train' else test_periods).update(periods)
            assert not np.array_equal(_no_switch(info), canonical), 'rewiring must affect signals'
            if split == 'train':
                evidence.update(k for k, n in stats.items() if n)
                for rival in rivals:
                    alternative = _no_switch(info) if rival == 'no_switch' else _paint(
                        info['structural'], info['paths'], info['sources'], info['track'], rival)
                    if not np.array_equal(canonical, alternative):
                        refuted.add(rival)
    assert len(roles) == 1, 'structural palette must be consistent within a puzzle'
    assert {'single', 'dual', 'left', 'right', 'tie', 'alone_left', 'alone_right'} <= evidence
    assert test_periods <= train_periods
    assert rivals <= refuted, f'insufficient demonstrations: {rivals - refuted}'


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    # A hand-worked two-lane fixture: one switch creates two U paths.
    # Left periods 2/3 change collision winner; right 2/2 cancel.
    inp = np.array([[int(c) for c in row] for row in (
        '2288888888888888844', '0000000555000000000', '3338888888888888811')], dtype=np.int8)
    inp = np.pad(inp, 1)
    # Fixed expected values in path order, not recomputed by solve.
    left = [2, 2, 2, 8, 2, 8, 2, 3, 2, 8, 3, 8, 2, 3, 3, 3, 3]
    right = [4, 4, 4, 8, 4, 8, 4, 8, 4, 8, 8, 8, 1, 8, 1, 8, 1, 8, 1, 1, 1]
    expected = np.full((3, 19), 0, dtype=np.int8)
    expected[::2] = 8
    expected[:, 8] = 5
    lp = [(0, c) for c in range(8)] + [(1, 7)] + [(2, c) for c in range(7, -1, -1)]
    rp = [(0, c) for c in range(18, 8, -1)] + [(1, 9)] + [(2, c) for c in range(9, 19)]
    for cells, colors in ((lp, left), (rp, right)):
        for cell, color in zip(cells, colors):
            expected[cell] = color
    expected = np.pad(expected, 1)
    for turn in range(4):
        for flip in (False, True):
            a, b = (np.fliplr(v) if flip else v for v in (inp, expected))
            assert np.array_equal(solve(np.rot90(a, turn)), np.rot90(b, turn))
    # Equal source colors still cancel; a missing source instead permits a
    # single clock to continue all the way to the unoccupied terminal.
    assert np.array_equal(solve(np.where(inp == 1, 4, inp)), np.where(expected == 1, 4, expected))
    lone_in, lone_out = inp.copy(), expected.copy()
    lone_in[lone_in == 1] = 8
    for i, (r, c) in enumerate(rp):
        lone_out[r + 1, c + 1] = 4 if i < 2 or i % 2 == 0 else 8
    assert np.array_equal(solve(lone_in), lone_out)
    # Unsupported middle packets, source-covered switch tiles and closed
    # circuits are rejected by the input parser, independent of output labels.
    for bad_core in (
        _base(2, 19, [(1, 6), (1, 12)], [], (0, 8, 5)),
        np.array([[int(c) for c in row] for row in (
            '8882888888888888888', '0000000555000000000', '3338888888888888811')], dtype=np.int8),
        np.array([[int(c) for c in row] for row in (
            '2288888288888888844', '0000000555000000000', '3338888888888888811')], dtype=np.int8),
    ):
        try:
            solve(np.pad(bad_core, 1))
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid input geometry accepted')
    signatures, shapes, layouts, palettes = set(), set(), set(), set()
    axes, periods, lane_counts, divider_counts, break_counts, occupancies = set(), set(), set(), set(), set(), set()
    overlaps = winding = internal = same_color = 0
    for seed in _SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _inspect(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                shapes.add(pair['input'].shape)
                axes.add(info['axis'])
                palettes.add((info['background'], info['track'], info['gate']))
                lane_counts.add((info['core'].shape[0] + 1) // 2)
                divider_counts.add(len(info['centers']))
                break_counts.add(info['breaks'])
                overlaps += info['overlap'] > 0
                normalized = {info['background']: 0, info['track']: 1, info['gate']: 2}
                norm = bytes(normalized.setdefault(int(v), len(normalized)) for v in info['core'].flat)
                layouts.add((info['core'].shape, hashlib.sha256(norm).hexdigest()))
                for i, path in enumerate(info['paths']):
                    ss = info['sources'][i]
                    periods.update(s[0] for s in ss if s)
                    occupancies.add(sum(s is not None for s in ss))
                    same_color += bool(ss[0] and ss[1] and ss[0][1] == ss[1][1])
                    internal += sum(s is not None and p[1] not in (0, info['core'].shape[1] - 1)
                                    for p, s in zip((path[0], path[-1]), ss))
                    vertical_steps = {b[0] - a[0] for a, b in zip(path, path[1:]) if a[0] != b[0]}
                    winding += len(vertical_steps) == 2
    assert len(signatures) == 42
    assert len(shapes) >= 60 and len(layouts) >= 180 and len(palettes) >= 35
    assert axes == {0, 1} and periods == set(range(1, 7))
    assert len(lane_counts) >= 7 and len(divider_counts) >= 10 and len(break_counts) >= 8
    assert occupancies == {0, 1, 2} and min(overlaps, winding, internal, same_color) > 0
    # Counts, corruption, color relabeling, and illegal geometry are bounded.
    for n in range(2, 11):
        for m in range(1, min(n - 1, 4) + 1):
            generate_puzzle(0, n, m)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    mapping = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
    assert np.array_equal(solve(mapping[inp]), mapping[expected])
    bad = generate_puzzle(0, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
