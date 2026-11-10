"""Sort mobile spans within classes defined by attached badge cavity counts.

Only standard-library modules and NumPy are used. The public solver reads the
input geometry; construction parameters are never consulted by it.
"""
from collections import defaultdict
import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _NEIGHBORS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _NEIGHBORS
    result = []
    while remaining:
        cell = min(remaining)
        remaining.remove(cell)
        stack, cells = [cell], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                other = (r + dr, c + dc)
                if other in remaining:
                    remaining.remove(other)
                    stack.append(other)
        result.append(cells)
    return result


def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid


def _key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _perimeter(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _analyze(value):
    """Parse a scene and prove that all cells follow its visible grammar."""
    grid = _grid(value)
    bg = int(grid[0, 0])
    assert all(int(grid[r, c]) == bg for r, c in
               ((0, -1), (-1, 0), (-1, -1))), 'corners must identify background'
    rows = np.flatnonzero(np.all(grid != bg, axis=1))
    cols = np.flatnonzero(np.all(grid != bg, axis=0))
    assert len(rows) + len(cols) == 1, 'unique full carrier required'
    vertical = bool(len(cols))
    raw = grid.T if vertical else grid
    row = int(cols[0] if vertical else rows[0])
    carrier = int(raw[row, 0])
    assert carrier != bg and int(raw[row, -1]) == carrier
    assert 0 < row < raw.shape[0] - 1
    slots = []
    c = 0
    while c < raw.shape[1]:
        color = int(raw[row, c])
        end = c + 1
        while end < raw.shape[1] and raw[row, end] == color:
            end += 1
        if color != carrier:
            assert color != bg and 1 <= end - c <= 3
            slots.append((c, end, color))
        c = end
    assert 2 <= len(slots) <= 7
    assert all(b[0] > a[1] for a, b in zip(slots, slots[1:]))
    objects = _components((raw != bg) & (raw != carrier))
    assert len(objects) == len(slots)
    matched = []
    side_flags = []
    for start, end, color in slots:
        candidates = [cells for cells in objects if (row, start) in cells]
        assert len(candidates) == 1
        cells = candidates[0]
        assert all(int(raw[r, c]) == color for r, c in cells)
        crossing = sorted(c for r, c in cells if r == row)
        assert crossing == list(range(start, end))
        broad = [(r, c) for r, c in cells if c < start or c >= end]
        assert broad, 'badge must visibly widen beyond stem'
        flags = {r > row for r, c in broad}
        assert len(flags) == 1 and all(r != row for r, c in broad)
        side_flags.append(next(iter(flags)))
        matched.append(cells)
    assert len(set(side_flags)) == 1, 'all badges share one side'
    flipped = not side_flags[0]
    canonical = raw[::-1] if flipped else raw
    if flipped:
        row = raw.shape[0] - 1 - row
    rebuilt = np.full(canonical.shape, bg, dtype=canonical.dtype)
    rebuilt[row, :] = carrier
    records = []
    for (start, end, color), cells in zip(slots, matched):
        if flipped:
            cells = [(raw.shape[0] - 1 - r, c) for r, c in cells]
        below = [(r, c) for r, c in cells if r > row]
        assert below
        b0 = min(c for r, c in below)
        b1 = max(c for r, c in below) + 1
        wide = [r for r, c in below if c < start or c >= end]
        top, bottom = min(wide), max(r for r, c in below) + 1
        badge = canonical[top:bottom, b0:b1]
        assert 3 <= badge.shape[0] <= 9 and 3 <= badge.shape[1] <= 9
        assert b0 < start < end < b1
        assert b1 - b0 >= end - start + 2
        assert 0 <= top - row - 1 <= 3
        assert set(map(int, np.unique(badge))) <= {bg, color}
        assert np.all(_perimeter(badge) == color)
        assert len(_components(badge != bg)) == 1
        cavities = _components(badge == bg)
        top_bar = min(r for r, c in cells)
        assert top_bar <= row
        length = row - top_bar + 1
        rebuilt[top_bar:row + 1, start:end] = color
        rebuilt[row + 1:top, start:end] = color
        rebuilt[top:bottom, b0:b1] = badge
        records.append({'start': start, 'end': end, 'color': color,
                        'length': length, 'count': len(cavities),
                        'badge_box': (top, bottom, b0, b1),
                        'badge': badge.copy(),
                        'hole_area': sum(map(len, cavities)),
                        'cavity_sizes': tuple(sorted(map(len, cavities))),
                        'count8': len(_components(badge == bg, diagonal=True))})
    assert np.array_equal(rebuilt, canonical), 'unexplained cells or nonsolid bar/stem'
    assert all(b['badge_box'][2] > a['badge_box'][3]
               for a, b in zip(records, records[1:])), 'badge boxes must be separated'
    return {'canonical': canonical, 'row': row, 'background': bg,
            'carrier': carrier, 'vertical': vertical, 'flipped': flipped,
            'records': records}


def _restore(canonical, analysis):
    result = canonical[::-1] if analysis['flipped'] else canonical
    return (result.T if analysis['vertical'] else result).copy()


def _assigned_lengths(records, grouping='count', ordering='length'):
    # This helper is for solving and explicit competing rules, not validation.
    groups = defaultdict(list)
    for i, record in enumerate(records):
        group = 0 if grouping == 'global' else record[grouping]
        groups[group].append(i)
    lengths = [r['length'] for r in records]
    for indices in groups.values():
        if ordering == 'reverse':
            source = indices[::-1]
        else:
            source = sorted(indices, key=lambda i: records[i]['length'] *
                            (records[i]['end'] - records[i]['start'] if ordering == 'area' else 1))
        for target, origin in zip(indices, source):
            lengths[target] = records[origin]['length']
    return lengths


def _rewrite(analysis, lengths):
    out = analysis['canonical'].copy()
    row = analysis['row']
    out[:row] = analysis['background']
    for record, length in zip(analysis['records'], lengths):
        assert 1 <= length <= row + 1
        out[row - length + 1:row, record['start']:record['end']] = record['color']
    return _restore(out, analysis)


def solve(input_grid: np.ndarray) -> np.ndarray:
    analysis = _analyze(input_grid)
    return _rewrite(analysis, _assigned_lengths(analysis['records']))


def _check_pair(pair):
    """Relational invariants specify the answer without sorting/painting it."""
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    a, b = _analyze(inp), _analyze(out)
    for field in ('row', 'background', 'carrier', 'vertical', 'flipped'):
        assert a[field] == b[field]
    row = a['row']
    assert np.array_equal(a['canonical'][row:], b['canonical'][row:]), 'fixed side changed'
    assert len(a['records']) == len(b['records'])
    before, after = defaultdict(list), defaultdict(list)
    for x, y in zip(a['records'], b['records']):
        for field in ('start', 'end', 'color', 'count', 'badge_box'):
            assert x[field] == y[field], 'slot address or destination geometry changed'
        assert np.array_equal(x['badge'], y['badge'])
        before[x['count']].append(x['length'])
        after[y['count']].append(y['length'])
    for group in before:
        # Explicit multiset removal and pairwise ordering; no assignment helper.
        remaining = list(before[group])
        previous = 0
        for length in after[group]:
            assert length >= previous, 'class subsequence is not increasing'
            assert length in remaining, 'length crossed a class boundary'
            remaining.remove(length)
            previous = length
        assert not remaining
    actual = [r['length'] for r in b['records']]
    records = a['records']
    alternatives = {
        'omit_A': [r['length'] for r in records],
        'omit_B': _assigned_lengths(records, grouping='global'),
        'bar_area': _assigned_lengths(records, ordering='area'),
        'blank_area': _assigned_lengths(records, grouping='hole_area'),
        'eight_connected': _assigned_lengths(records, grouping='count8'),
        'color_classes': _assigned_lengths(records, grouping='color'),
        'class_reverse': _assigned_lengths(records, ordering='reverse'),
    }
    features = {name: lengths != actual for name, lengths in alternatives.items()}
    features.update({'slots': len(records), 'vertical': a['vertical'], 'flipped': a['flipped'],
                     'shape': inp.shape, 'background': a['background'], 'carrier': a['carrier'],
                     'counts': tuple(r['count'] for r in records),
                     'class_sizes': tuple(sorted(len(v) for v in before.values())),
                     'widths': tuple(r['end'] - r['start'] for r in records),
                     'lengths': tuple(r['length'] for r in records),
                     'badge_shapes': tuple(r['badge'].shape for r in records),
                     'cavity_sizes': tuple(r['cavity_sizes'] for r in records),
                     'geometry': (a['vertical'], a['flipped'], inp.shape, row,
                                  tuple((r['start'], r['end'], r['badge_box'], r['length']) for r in records)),
                     'masks': tuple((r['badge'].shape, (r['badge'] == a['background']).tobytes()) for r in records),
                     'palette_size': len(np.unique(inp))})
    return features


def _capacity(height, width):
    return ((height - 1) // 2) * ((width - 1) // 2)


def _cavity_mask(rng, height, width, count):
    assert 0 <= count <= _capacity(height, width)
    mask = np.zeros((height, width), dtype=bool)
    if count == 0:
        return mask
    if count == 1 and rng.random() < 0.28:
        mask[1:-1, 1:-1] = True
        return mask
    if count == 4 and min(height, width) >= 5 and rng.random() < 0.24:
        for r in (1, height - 2):
            for c in (1, width - 2):
                mask[r, c] = True
    else:
        # Arbitrary isolated seeds first, with a separated-grid guaranteed fallback.
        for _ in range(10):
            mask[:] = False
            for index in rng.permutation((height - 2) * (width - 2)):
                r, c = divmod(int(index), width - 2)
                r, c = r + 1, c + 1
                if any(mask[r + dr, c + dc] for dr, dc in _NEIGHBORS):
                    continue
                mask[r, c] = True
                if len(_components(~mask)) != 1:
                    mask[r, c] = False
                if int(mask.sum()) == count:
                    break
            if int(mask.sum()) == count:
                break
        else:
            mask[:] = False
            cells = [(r, c) for r in range(1, height - 1, 2) for c in range(1, width - 1, 2)]
            for i in rng.permutation(len(cells))[:count]:
                mask[cells[int(i)]] = True
    target = int(rng.integers(count, (height - 2) * (width - 2) + 1))
    for _ in range(3):
        grew = False
        for index in rng.permutation((height - 2) * (width - 2)):
            if int(mask.sum()) >= target:
                break
            r, c = divmod(int(index), width - 2)
            r, c = r + 1, c + 1
            if mask[r, c] or not any(mask[r + dr, c + dc] for dr, dc in _NEIGHBORS):
                continue
            mask[r, c] = True
            if len(_components(mask)) != count or len(_components(~mask)) != 1:
                mask[r, c] = False
            else:
                grew = True
        if not grew:
            break
    assert len(_components(mask)) == count and len(_components(~mask)) == 1
    return mask


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _sample_data(rng, kind):
    if kind == 'first':
        counts = [1, 2, 1, 2]
        short = int(rng.integers(3, 5))
        long = int(rng.integers(max(7, short + 3), min(9, 3 * short - 1) + 1))
        lengths = [long, int(rng.integers(5, 7)), short, int(rng.integers(1, 3))]
        widths = [1, int(rng.integers(1, 3)), 3, int(rng.integers(1, 3))]
        head_widths = list(map(int, rng.integers(5, 7, size=4)))
    elif kind == 'second':
        counts = [0, 3, 0, 3, 0]
        lengths = [int(rng.integers(6, 8)), int(rng.integers(8, 11)), 1,
                   int(rng.integers(2, 4)), int(rng.integers(4, 6))]
        widths = list(map(int, rng.integers(1, 4, size=5)))
        while sum(w + 2 for w in widths) + len(widths) + 1 > 30:
            widths[_choose(rng, [i for i, w in enumerate(widths) if w > 1])] -= 1
        head_widths = [int(rng.integers(w + 2, 8)) for w in widths]
    else:
        n = int(rng.integers(6, 8)) if kind == 'test' else int(rng.integers(2, 6))
        classes = list(map(int, rng.choice(5, 2 if n >= 3 else 1, replace=False)))
        if kind == 'test':
            counts = [classes[0]] * (4 if n == 6 else int(rng.integers(4, 6)))
            counts += [classes[1]] * (n - len(counts))
            if n == 7 and len(counts) - counts.count(classes[0]) == 3 and rng.random() < 0.35:
                counts[-1] = _choose(rng, [k for k in range(5) if k not in classes])
        else:
            counts = [classes[i % len(classes)] for i in range(n)]
        rng.shuffle(counts)
        lengths = list(map(int, rng.integers(1, 13, size=n)))
        widths = list(map(int, rng.integers(1, 4 if n <= 5 else 3, size=n)))
        if n == 7:
            widths = [1] * n
        # Reduce stem widths only where packing makes their minimum heads too wide.
        while sum(w + 2 for w in widths) + n + 1 > 30:
            choices = [i for i, w in enumerate(widths) if w > 1]
            widths[_choose(rng, choices)] -= 1
        head_widths = [int(rng.integers(w + 2, 8)) for w in widths]
        if kind == 'extra' and rng.random() < 0.14:
            temporary = [{'count': k, 'length': length} for k, length in zip(counts, lengths)]
            lengths = _assigned_lengths(temporary)
    n = len(counts)
    minimums = [5] * n if kind == 'first' else [w + 2 for w in widths]
    while sum(head_widths) + n + 1 > 30:
        choices = [i for i, w in enumerate(head_widths) if w > minimums[i]]
        assert choices
        head_widths[_choose(rng, choices)] -= 1
    heights = []
    for width, count in zip(head_widths, counts):
        legal = [h for h in range(3, 10) if _capacity(h, width) >= count]
        heights.append(_choose(rng, legal))
    if kind == 'first':
        heights = list(map(int, rng.integers(5, 8, size=n)))
    masks = [_cavity_mask(rng, h, w, k) for h, w, k in zip(heights, head_widths, counts)]
    if kind == 'first':
        # Distinguish blank area, four/eight connectivity, and shape equality.
        masks[0][:] = False
        masks[0][1:-1, 1:-1] = True
        masks[1][:] = False
        masks[1][1, 1] = masks[1][2, 2] = masks[1][2, 3] = True
        masks[2][:] = False
        masks[2][1, 1:3] = True
        masks[3][:] = False
        masks[3][1, 1] = masks[3][-2, -2] = True
        # Transverse/parallel reflections vary the address shapes; never labels.
        for i in range(n):
            if rng.random() < 0.5:
                masks[i] = masks[i][::-1].copy()
            if rng.random() < 0.5:
                masks[i] = masks[i][:, ::-1].copy()
    return counts, lengths, widths, head_widths, heights, masks


def _make_pair(rng, kind, vertical, flipped, richness):
    counts, lengths, widths, head_widths, heights, masks = _sample_data(rng, kind)
    n = len(counts)
    palette = list(map(int, rng.permutation(10)))
    bg, carrier = palette[:2]
    pool = palette[2:2 + richness]
    colors = list(map(int, rng.choice(pool, n)))
    if kind == 'first':
        colors = [pool[0], pool[1], pool[2], pool[0]]
    stems = list(map(int, rng.integers(0, 4, size=n)))
    needed_width = sum(head_widths) + n + 1
    canvas_width = int(rng.integers(needed_width, 31))
    extra = canvas_width - needed_width
    gaps = [1 + int(x) for x in rng.multinomial(extra, [1 / (n + 1)] * (n + 1))]
    badge_starts, starts = [], []
    position = gaps[0]
    for i, width in enumerate(head_widths):
        badge_starts.append(position)
        starts.append(position + int(rng.integers(1, width - widths[i])))
        position += width + gaps[i + 1]
    depth = max(s + h for s, h in zip(stems, heights))
    minimum_height = max(lengths) + depth
    spare = min(6, 30 - minimum_height)
    assert spare >= 1
    top_margin = int(rng.integers(1, min(4, spare) + 1))
    bottom_margin = int(rng.integers(0, spare - top_margin + 1))
    canvas_height = minimum_height + top_margin + bottom_margin
    row = max(lengths) - 1 + top_margin
    inp = np.full((canvas_height, canvas_width), bg, dtype=np.int8)
    inp[row] = carrier
    for i in range(n):
        x, w, color = starts[i], widths[i], colors[i]
        inp[row - lengths[i] + 1:row + 1, x:x + w] = color
        badge_top = row + stems[i] + 1
        inp[row + 1:badge_top, x:x + w] = color
        badge = np.where(masks[i], bg, color).astype(np.int8)
        bx = badge_starts[i]
        inp[badge_top:badge_top + heights[i], bx:bx + head_widths[i]] = badge
    # Ground truth uses known construction attributes, independent of parsing.
    out = inp.copy()
    out[:row] = bg
    for count in set(counts):
        destinations = [i for i, k in enumerate(counts) if k == count]
        ordered_lengths = sorted(lengths[i] for i in destinations)
        for i, length in zip(destinations, ordered_lengths):
            out[row - length + 1:row, starts[i]:starts[i] + widths[i]] = colors[i]
    if flipped:
        inp, out = inp[::-1].copy(), out[::-1].copy()
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {'input': inp, 'output': out}


def _fresh_palette(pair, seen):
    # 90 different images of two present colors exceed the 13 other inputs.
    present = list(map(int, np.unique(pair['input'])))
    original = present[:2] + [c for c in range(10) if c not in present[:2]]
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            target = [a, b] + [c for c in range(10) if c not in (a, b)]
            lookup = np.empty(10, dtype=np.int8)
            lookup[original] = target
            candidate = {k: lookup[v] for k, v in pair.items()}
            if _key(candidate['input']) not in seen:
                return candidate
    raise AssertionError('finite distinct-palette fallback exhausted')


def _resolve_counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + ' must be an integer')
        if not low <= int(value) <= high:
            raise ValueError(name + ' is outside the legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _resolve_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    vertical = bool(rng.integers(2))
    flipped = bool(rng.integers(2))
    richness = int(rng.integers(3, 9))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', nt), ('test', ne)):
        for i in range(number):
            kind = ('first' if i == 0 else 'second' if i == 1 else 'extra') if split == 'train' else 'test'
            v = vertical if kind == 'first' else not vertical if kind == 'second' else bool(rng.integers(2))
            f = flipped if kind == 'first' else not flipped if kind == 'second' else bool(rng.integers(2))
            for _ in range(8):
                pair = _make_pair(rng, kind, v, f, richness)
                if _key(pair['input']) not in seen:
                    break
            else:
                pair = _fresh_palette(pair, seen)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _resolve_counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp = _grid(pair['input'])
            _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            features[split].append(_check_pair(pair))
    for alternative in ('omit_A', 'omit_B', 'bar_area', 'blank_area', 'eight_connected',
                        'color_classes', 'class_reverse'):
        assert any(f[alternative] for f in features['train']), 'training permits ' + alternative
    assert {f['vertical'] for f in features['train']} == {False, True}
    assert {f['flipped'] for f in features['train']} == {False, True}
    assert sum(f['omit_A'] for f in features['train']) >= 2
    largest_train = max(f['slots'] for f in features['train'])
    for f in features['test']:
        assert f['slots'] > largest_train, 'slot-count growth missing'
        assert max(f['class_sizes']) >= 4, 'test must generalize class cardinality'
    return features


def self_test():
    """Bounded 42-seed sweep plus interface and semantic mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = []
    signatures = set()
    for puzzle in puzzles:
        analyses = validate_generated_puzzle(puzzle)
        signature = tuple(_key(pair['input']) for split in ('train', 'test') for pair in puzzle[split])
        signatures.add(signature)
        features.extend(analyses['train'] + analyses['test'])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                before = pair['input'].copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input']), 'solve mutated its input'
    assert len(signatures) == 42
    repeated = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(puzzles[0][s], repeated[s]) for k in ('input', 'output'))
    assert len({f['shape'] for f in features}) >= 35
    assert len({f['geometry'] for f in features}) >= 190
    assert len({mask for f in features for mask in f['masks']}) >= 100
    assert {f['slots'] for f in features} >= {2, 3, 4, 5, 6, 7}
    assert {k for f in features for k in f['counts']} == set(range(5))
    assert {w for f in features for w in f['widths']} == {1, 2, 3}
    assert {n for f in features for n in f['lengths']} == set(range(1, 13))
    assert {f['background'] for f in features} == set(range(10))
    assert {f['carrier'] for f in features} == set(range(10))
    assert len({shape for f in features for shape in f['badge_shapes']}) >= 25
    assert max(size for f in features for sizes in f['cavity_sizes'] for size in sizes) >= 16
    assert any(len(set(f['lengths'])) < len(f['lengths']) for f in features)
    assert any(not f['omit_A'] for f in features), 'identity class-sorted examples missing'
    # A legal family member outside the sampler's 0--4 count range. Expected
    # lengths [1, 8, 6] are written explicitly, not obtained from a sorter.
    fixture = np.zeros((21, 23), dtype=np.int8)
    fixture[9] = 1
    hole_mask = np.zeros((7, 5), dtype=bool)
    for r, c in ((1, 1), (1, 3), (3, 1), (3, 3), (5, 1)):
        hole_mask[r, c] = True
    for x, width, color, length, count in ((3, 1, 2, 6, 5), (11, 2, 3, 8, 0), (19, 1, 4, 1, 5)):
        fixture[9 - length + 1:11, x:x + width] = color
        fixture[11:18, x - 2:x + 3] = np.where(hole_mask if count else False, 0, color)
    expected = fixture.copy()
    expected[:9] = 0
    for x, width, color, length in ((3, 1, 2, 1), (11, 2, 3, 8), (19, 1, 4, 6)):
        expected[9 - length + 1:9, x:x + width] = color
    for flipped in (False, True):
        for vertical in (False, True):
            inp = fixture[::-1] if flipped else fixture
            out = expected[::-1] if flipped else expected
            inp = inp.T if vertical else inp
            out = out.T if vertical else out
            assert np.array_equal(solve(inp), out)
            _check_pair({'input': inp, 'output': out})
    # Minimum curriculum must reject every competing rule, not just default counts.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(7000 + 10 * nt + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Reject both structurally valid competing labels and arbitrary corruption.
    p = generate_puzzle(123, 2, 1)
    rejected = set()
    for pair in p['train']:
        a = _analyze(pair['input'])
        for name, grouping, ordering in (('global', 'global', 'length'),
                                         ('area', 'count', 'area'),
                                         ('eight', 'count8', 'length'),
                                         ('reverse', 'count', 'reverse')):
            bad = _rewrite(a, _assigned_lengths(a['records'], grouping, ordering))
            if np.array_equal(bad, pair['output']):
                continue
            try:
                _check_pair({'input': pair['input'], 'output': bad})
            except AssertionError:
                rejected.add(name)
            else:
                raise AssertionError('incorrect length assignment accepted')
    assert rejected == {'global', 'area', 'eight', 'reverse'}
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in ('train', 'test')}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    # Correct labels with an uninformative training set are also insufficient.
    weak = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in ('train', 'test')}
    for pair in weak['train']:
        a = _analyze(pair['input'])
        canonical = a['canonical'].copy()
        for r in a['records']:
            top, bottom, left, right = r['badge_box']
            canonical[top:bottom, left:right] = r['color']
        pair['input'] = _restore(canonical, a)
        pair['output'] = solve(pair['input'])
    try:
        validate_generated_puzzle(weak)
    except AssertionError:
        pass
    else:
        raise AssertionError('training without topology distinctions accepted')
    return {'seeds': 42, 'pairs': len(features),
            'shapes': len({f['shape'] for f in features}),
            'geometries': len({f['geometry'] for f in features}),
            'badge_masks': len({mask for f in features for mask in f['masks']}),
            'badge_shapes': len({shape for f in features for shape in f['badge_shapes']})}


if __name__ == '__main__':
    print(self_test())
