"""Rank-paired square queues followed by full-vector run contraction.

Only standard Python and NumPy are used. No parent module, seed-dependent
solver state, filesystem access, or stored examples are needed by solve().
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


HOLLOW, SOLID = 0, 1


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _tile(size, color, background, kind):
    a = np.full((size, size), color, dtype=np.int8)
    if kind == HOLLOW:
        a[1:-1, 1:-1] = background
    return a


def _parse(input_grid):
    """Read complete square footprints against the visible background border."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("the input must have an uninterrupted background border")
    todo = a != bg
    objects = []
    common_size = None
    while np.any(todo):
        r, c = map(int, np.argwhere(todo)[0])
        color = int(a[r, c])
        s = 0
        while c + s < a.shape[1] and a[r, c + s] == color:
            s += 1
        if s < 3 or r + s > a.shape[0] or c + s > a.shape[1]:
            raise ValueError("incomplete square")
        tile = a[r:r+s, c:c+s]
        if np.all(tile == color):
            kind = SOLID
        elif np.array_equal(tile, _tile(s, color, bg, HOLLOW)):
            kind = HOLLOW
        else:
            raise ValueError("expected a monochrome solid or one-cell square rim")
        if common_size is not None and s != common_size:
            raise ValueError("square sizes must agree within an input")
        common_size = s
        # This also prevents touching instances of different or repeated colors.
        neighborhood = a[max(0, r-1):r+s+1, max(0, c-1):c+s+1].copy()
        rr, cc = r - max(0, r-1), c - max(0, c-1)
        neighborhood[rr:rr+s, cc:cc+s] = bg
        if not np.all(neighborhood == bg):
            raise ValueError("square bounding boxes need a background-cell gap")
        objects.append(dict(top=r, left=c, kind=kind, color=color, tile=tile.copy()))
        todo[r:r+s, c:c+s] = False
    if not objects:
        raise ValueError("no square tiles")
    queues = []
    for kind in (HOLLOW, SOLID):
        q = sorted((o for o in objects if o['kind'] == kind), key=lambda o: o['top'])
        if not q or len({o['top'] for o in q}) != len(q):
            raise ValueError("both classes are required, without within-class top ties")
        queues.append(q)
    if common_size * max(map(len, queues)) > 30 or 2 * common_size > 30:
        raise ValueError("rank layout exceeds the ARC bounds")
    return dict(background=bg, size=common_size, objects=objects, queues=queues)


def _layout(analysis, queues=None):
    s, bg = analysis['size'], analysis['background']
    queues = analysis['queues'] if queues is None else queues
    a = np.full((s * max(map(len, queues)), 2*s), bg, dtype=np.int8)
    for col, q in enumerate(queues):
        for rank, obj in enumerate(q):
            if obj is not None:
                a[rank*s:(rank+1)*s, col*s:(col+1)*s] = obj['tile']
    return a


def _run_indices(a, axis):
    vectors = a if axis == 0 else a.T
    return np.r_[0, np.flatnonzero(np.any(vectors[1:] != vectors[:-1], axis=1)) + 1]


def _collapse(a):
    # Both sets of representatives are determined on the same full layout.
    rows, cols = _run_indices(a, 0), _run_indices(a, 1)
    return a[np.ix_(rows, cols)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Parse, rank hollow/solid queues, pad, then contract equal adjacent vectors."""
    return _collapse(_layout(_parse(input_grid)))


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _alternatives(a, source):
    """Concrete wrong rules used only to audit demonstration evidence."""
    ranked = _layout(a)
    qs = a['queues']
    alternatives = {'omit_A': _collapse(source), 'omit_B': ranked}
    for reverse in (False, True):
        q = [sorted(v, key=lambda o: (o['left'], o['top']), reverse=reverse) for v in qs]
        alternatives['horizontal_' + str(reverse)] = _collapse(_layout(a, q))
    alternatives['reverse_vertical'] = _collapse(_layout(a, [q[::-1] for q in qs]))
    alternatives['swapped_columns'] = _collapse(_layout(a, qs[::-1]))
    alternatives['numeric_color_order'] = _collapse(_layout(
        a, [sorted(q, key=lambda o: (o['color'], o['top'])) for q in qs]))
    alternatives['numeric_color_reverse'] = _collapse(_layout(
        a, [sorted(q, key=lambda o: (-o['color'], o['top'])) for q in qs]))
    greedy_rows = [[None, None]]
    for obj in sorted(a['objects'], key=lambda o: (o['top'], o['left'])):
        col = obj['kind']
        if greedy_rows[-1][col] is not None:
            greedy_rows.append([None, None])
        greedy_rows[-1][col] = obj
    greedy_queues = [[row[col] for row in greedy_rows] for col in (0, 1)]
    alternatives['greedy'] = _collapse(_layout(a, greedy_queues))
    # Compress each square before packing it into fixed 3-by-3/3-by-1 slots.
    # This misses shared boundaries and matching consecutive rank pairs.
    n, bg = max(map(len, qs)), a['background']
    local = np.full((3*n, 4), bg, dtype=np.int8)
    for col, q in enumerate(qs):
        for rank, obj in enumerate(q):
            if col == 0:
                local[3*rank:3*rank+3, :3] = _tile(3, obj['color'], bg, HOLLOW)
            else:
                local[3*rank:3*rank+3, 3] = obj['color']
    alternatives['tile_local'] = local
    unique_rows, unique_cols = [], []
    for i in range(ranked.shape[0]):
        if not any(np.array_equal(ranked[i], ranked[j]) for j in unique_rows):
            unique_rows.append(i)
    for i in range(ranked.shape[1]):
        if not any(np.array_equal(ranked[:, i], ranked[:, j]) for j in unique_cols):
            unique_cols.append(i)
    alternatives['global_deduplication'] = ranked[np.ix_(unique_rows, unique_cols)]
    alternatives['rows_only'] = ranked[_run_indices(ranked, 0)]
    alternatives['columns_only'] = ranked[:, _run_indices(ranked, 1)]
    return alternatives


def _positions(rng, size, count):
    """Variable shelf initialization followed by free legal square relocation."""
    extent = 28  # Leave an explicit one-cell background border on every side.
    capacity = (extent + 1) // (size + 1)
    low = (count + capacity - 1) // capacity
    high = min(count, (extent + 1 - count) // size)
    bands = int(rng.integers(low, high + 1))
    amounts = [1] * bands
    for _ in range(count - bands):
        candidates = [i for i, n in enumerate(amounts) if n < capacity]
        amounts[int(rng.choice(candidates))] += 1
    min_h, min_w = count + bands*size - 1, max(amounts)*(size+1) - 1
    mode = int(rng.integers(5))
    if mode == 0:
        h = int(rng.integers(min_h, min(28, min_h+3)+1))
        w = int(rng.integers(min_w, min(28, min_w+3)+1))
    elif mode == 1:
        h = w = int(rng.integers(max(min_h, min_w), 29))
    else:
        h = 28 if mode == 2 else int(rng.integers(min_h, 29))
        w = 28 if mode == 3 else int(rng.integers(min_w, 29))
    gaps = rng.multinomial(h-min_h, np.ones(bands+1)/(bands+1))
    positions = []
    top = int(gaps[0])
    for band, n in enumerate(amounts):
        blanks = rng.multinomial(w-(n*(size+1)-1), np.ones(n+1)/(n+1))
        left = int(blanks[0])
        for offset, extra in zip(rng.permutation(n), blanks[1:]):
            positions.append((top+int(offset), left))
            left += size+1+int(extra)
        top += size+n+int(gaps[band+1])
    rr, cc = np.indices((h-size+1, w-size+1))
    for _ in range(2):
        for idx in rng.permutation(count):
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(positions):
                if j != idx:
                    legal &= ((abs(rr-r) > size) | (abs(cc-c) > size))
                    legal &= rr != r
            sites = np.argwhere(legal)
            r, c = sites[int(rng.integers(len(sites)))]
            positions[int(idx)] = (int(r), int(c))
    return (h+2, w+2), sorted((r+1, c+1) for r, c in positions)


def _draw(rng, size, background, colors, anchor=False, fallback=False):
    """Colors are two top-ordered queues, not a mapping from color to class."""
    n = sum(map(len, colors))
    if fallback:
        # Six squares: grouped HHHSSS, both horizontal sort directions wrong.
        p = size+1
        positions = [(1, 1+2*p), (2, 1), (3, 1+p),
                     (size+4, 1+2*p), (size+5, 1), (size+6, 1+p)]
        shape = (2*size+7, 3*size+4)
        kinds = [HOLLOW]*3 + [SOLID]*3
    else:
        shape, positions = _positions(rng, size, n)
        kinds = list(map(int, rng.permutation([0]*len(colors[0]) + [1]*len(colors[1]))))
        if anchor and all(x != y for x, y in zip(kinds, kinds[1:])):
            # An early duplicate class can expose greedy global-scan packing.
            kinds = [HOLLOW]*3 + [SOLID]*3
    inp = np.full(shape, background, dtype=np.int8)
    ranks = [0, 0]
    for (r, c), kind in zip(positions, kinds):
        inp[r:r+size, c:c+size] = _tile(size, colors[kind][ranks[kind]], background, kind)
        ranks[kind] += 1
    return {'input': inp, 'output': solve(inp)}


def _colors(rng, counts, background, style):
    palette = list(map(int, rng.permutation([c for c in range(10) if c != background])))
    h, s = counts
    if style == 'anchor':
        # Shared seam merges; an adjacent pair of rank signatures also merges.
        return [[palette[0], palette[0], palette[1]]] * 2
    if style == 'seam':
        # Same counts as the anchor, but only the middle rank matches colors.
        # The first and last rank signatures recur with a different one between.
        return [[palette[0], palette[1], palette[0]],
                [palette[2], palette[1], palette[2]]]
    if style == 'padding':
        small, large = min(counts), max(counts)
        short = [palette[i % 9] for i in range(small)]
        # c,d,c,... refutes either numeric color direction; for four or more
        # members the repeated-color trailing ranks also share a boundary.
        long = [palette[2], palette[3]] + [palette[2]] * (large-2)
        return [short, long] if h < s else [long, short]
    mode = int(rng.integers(5))
    if mode == 0 and h == s:
        q = list(map(int, rng.choice(palette[:3], size=h)))
        return [q, q.copy()]
    if mode == 1 and h+s <= 9:
        return [palette[:h], palette[h:h+s]]
    pool = palette[:int(rng.integers(1, 6))]
    queues = [list(map(int, rng.choice(pool, size=n))) for n in counts]
    if mode == 2:
        # Repeated adjacent rank signatures, if both queues are long enough.
        for q in queues:
            if len(q) >= 2:
                q[1] = q[0]
    elif mode == 3:
        # Separated repeats must survive; a blanket uniqueness rule is wrong.
        for q in queues:
            if len(q) >= 3:
                q[0], q[1], q[2] = palette[0], palette[1], palette[0]
    return queues


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    # The first test uses a visibly new scale; no hidden parameter changes rules.
    novel_size = int(rng.integers(4, 9))
    train_sizes = [s for s in range(3, 9) if s != novel_size]
    first_size = int(rng.choice(train_sizes))
    second_size = int(rng.choice([s for s in train_sizes if s != first_size]))
    base_bg = int(rng.integers(10))
    varying_bg = bool(rng.integers(2))
    padding_side = int(rng.integers(2))  # Index of the longer queue.
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(num_train + num_test):
        training = index < num_train
        if index == 0:
            size, counts, style = first_size, (3, 3), 'anchor'
        elif index == 1:
            size, counts, style = second_size, (3, 3), 'seam'
        elif training and index in (2, 3):
            size = int(rng.choice(train_sizes))
            large = int(rng.integers(3, min(5, 30//size)+1))
            counts = [2, 2]
            counts[padding_side if index == 2 else 1-padding_side] = large
            counts, style = tuple(counts), 'padding'
        else:
            size = novel_size if index == num_train else int(rng.choice(
                train_sizes if training else range(3, 9)))
            limit = 30//size
            if rng.random() < 0.25:
                n = int(rng.integers(1, limit+1))
                counts = (n, n)
            else:
                counts = tuple(map(int, rng.integers(1, limit+1, size=2)))
            # Two demos teach balance and both shared-seam outcomes. Additional
            # demos teach padding directions before tests can require them.
            if not training and num_train == 2:
                counts = (counts[0], counts[0])
            elif not training and num_train == 3:
                counts = tuple(sorted(counts, reverse=padding_side == 0))
            style = 'random'
        bg = int(rng.integers(10)) if varying_bg else base_bg
        for attempt in range(48):
            colors = _colors(rng, counts, bg, style)
            pair = _draw(rng, size, bg, colors, anchor=index == 0)
            if index == 0:
                a = _parse(pair['input'])
                wrong = _alternatives(a, pair['input'])
                required = ('greedy', 'horizontal_False', 'horizontal_True',
                            'reverse_vertical', 'global_deduplication', 'tile_local')
                if any(np.array_equal(wrong[k], pair['output']) for k in required):
                    continue
            if _key(pair['input']) not in seen:
                break
        else:
            if index == 0:
                pair = _draw(rng, size, bg, _colors(rng, counts, bg, 'anchor'), fallback=True)
            else:
                # Ninety distinct bijective palette encodings of this input;
                # at most thirteen inputs precede it. Preserve every geometry,
                # equality and padding witness instead of changing counts.
                original = pair['input']
                old_bg = int(original[0, 0])
                fg = int(original[original != old_bg][0])
                accepted = False
                for new_bg in range(10):
                    for new_fg in range(10):
                        if new_bg == new_fg:
                            continue
                        mapping = np.zeros(10, dtype=np.int8)
                        mapping[old_bg], mapping[fg] = new_bg, new_fg
                        old = [c for c in range(10) if c not in (old_bg, fg)]
                        new = [c for c in range(10) if c not in (new_bg, new_fg)]
                        mapping[old] = new
                        inp = mapping[original]
                        if _key(inp) not in seen:
                            pair = {'input': inp, 'output': solve(inp)}
                            accepted = True
                            break
                    if accepted:
                        break
            if _key(pair['input']) in seen:
                raise RuntimeError('bounded distinct-input construction failed')
        seen.add(_key(pair['input']))
        puzzle['train' if training else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_invariants(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    source, output = _grid(pair['input']), _grid(pair['output'])
    a = _parse(source)
    ranked = _layout(a)
    rows, cols = _run_indices(ranked, 0), _run_indices(ranked, 1)
    assert np.array_equal(output, ranked[np.ix_(rows, cols)]), 'wrong contraction'
    # Re-expansion recovers EVERY cell, including all row/column crossings.
    row_lengths = np.diff(np.r_[rows, ranked.shape[0]])
    col_lengths = np.diff(np.r_[cols, ranked.shape[1]])
    expanded = np.repeat(np.repeat(output, row_lengths, axis=0), col_lengths, axis=1)
    assert np.array_equal(expanded, ranked), 'lost crossing data'
    assert len(_run_indices(output, 0)) == output.shape[0]
    assert len(_run_indices(output, 1)) == output.shape[1]
    h, s = map(len, a['queues'])
    # Whole-vector column equality has a tight, family-specific consequence.
    aligned = h == s and all(x['color'] == y['color'] for x, y in zip(*a['queues']))
    assert output.shape[1] == (3 if aligned else 4)
    assert ranked.shape[1] > output.shape[1], 'B must change every answer'
    # Closed-form height check independent of the generic run-index function.
    signatures = [(a['queues'][0][i]['color'] if i < h else a['background'],
                   a['queues'][1][i]['color'] if i < s else a['background'])
                  for i in range(max(h, s))]
    height = 0
    for i, sig in enumerate(signatures):
        height += 3 if i < h else 1
        if i and sig == signatures[i-1]:
            height -= 1
    assert output.shape[0] == height, 'wrong cross-rank row merging'
    assert np.array_equal(solve(source), output)
    return a


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, evidence, analyses = set(), {}, {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _pair_invariants(pair)
            analyses[split].append(a)
            key = _key(np.asarray(pair['input'], dtype=np.int8))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            if split == 'train':
                for name, wrong in _alternatives(a, np.asarray(pair['input'])).items():
                    evidence[name] = evidence.get(name, False) or not np.array_equal(wrong, pair['output'])
    # This checks real answers after compression, which can hide differences
    # that were observable in the uncompressed A layout.
    assert all(evidence.values()), 'missing training counterexample: ' + str(
        [k for k, v in evidence.items() if not v])
    widths = {np.asarray(p['output']).shape[1] for p in puzzle['train']}
    assert widths == {3, 4}, 'teach equality and inequality across the shared seam'
    train_sizes = {a['size'] for a in analyses['train']}
    assert analyses['test'][0]['size'] not in train_sizes, 'missing test scale shift'
    differences = [len(a['queues'][0]) - len(a['queues'][1]) for a in analyses['train']]
    for a in analyses['test']:
        d = len(a['queues'][0]) - len(a['queues'][1])
        assert d == 0 or any(d*x > 0 for x in differences), 'unshown padding side'


def self_test():
    """Bounded, in-memory 42-seed audit; no files or execution traces are saved."""
    # Literal regression answers make shared seams and separated recurrences
    # explicit, independently of the generic contraction and height checks.
    expected = (
        ([[2, 2, 7], [2, 2, 7]],
         ['222', '202', '222', '202', '222', '777', '707', '777']),
        ([[2, 3, 2], [7, 3, 7]],
         ['2227', '2027', '2227', '3333', '3033', '3333', '2227', '2027', '2227']),
        ([[2, 3], [7, 4, 7, 7]],
         ['2227', '2027', '2227', '3334', '3034', '3334', '0007']),
        ([[7, 4, 7, 7], [2, 3]],
         ['7772', '7072', '7772', '4443', '4043', '4443',
          '7770', '7070', '7770', '7070', '7770']),
    )
    regression_rng = np.random.default_rng(918)
    for colors, rows in expected:
        answer = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
        for size in (3, 4, 6):
            pair = _draw(regression_rng, size, 0, colors)
            assert np.array_equal(pair['output'], answer), 'literal boundary regression'
    # With the visible border and two separated squares, 13 is the largest
    # feasible size. Opposite-class top ties need no sorting convention.
    for size in range(3, 14):
        inp = np.full((size+2, 2*size+3), 5, dtype=np.int8)
        inp[1:size+1, 1:size+1] = _tile(size, 0, 5, HOLLOW)
        inp[1:size+1, size+2:2*size+2] = _tile(size, 9, 5, SOLID)
        answer = np.array([[0, 0, 0, 9], [0, 5, 0, 9], [0, 0, 0, 9]], dtype=np.int8)
        assert np.array_equal(solve(inp), answer), 'opposite top tie or feasible scale'
    shapes, layouts, sizes, backgrounds, counts, outputs, hashes = (set() for _ in range(7))
    black_tiles = repeated_ranks = separated_colors = total = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _pair_invariants(pair)
                total += 1
                shapes.add(pair['input'].shape)
                outputs.add(pair['output'].shape)
                sizes.add(a['size'])
                backgrounds.add(a['background'])
                counts.add(tuple(map(len, a['queues'])))
                layouts.add((pair['input'].shape, a['size'], tuple(
                    (o['top'], o['left'], o['kind']) for o in a['objects'])))
                black_tiles += any(o['color'] == 0 for o in a['objects'])
                qs = [[o['color'] for o in q] for q in a['queues']]
                repeated_ranks += any(x == y for x, y in zip(list(zip(*qs)), list(zip(*qs))[1:]))
                separated_colors += any(any(q[i] == q[j] and any(c != q[i] for c in q[i:j])
                                           for i in range(len(q)) for j in range(i+2, len(q))) for q in qs)
                for role in ('input', 'output'):
                    digest.update(bytes(pair[role].shape))
                    digest.update(pair[role].tobytes())
                # Sequential axis orders commute with simultaneous contraction.
                ranked = _layout(a)
                reduced = ranked[_run_indices(ranked, 0)]
                assert np.array_equal(reduced[:, _run_indices(reduced, 1)], pair['output'])
                reduced = ranked[:, _run_indices(ranked, 1)]
                assert np.array_equal(reduced[_run_indices(reduced, 0)], pair['output'])
        hashes.add(digest.digest())
    assert len(hashes) == 42
    assert len(shapes) >= 50 and len(layouts) >= 180
    assert sizes == set(range(3, 9)) and backgrounds == set(range(10))
    assert len(counts) >= 25 and len(outputs) >= 12
    assert black_tiles >= 10 and repeated_ranks >= 42 and separated_colors >= 10
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(314159, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for bad_seed in (-1, 1.5, True):
        try:
            generate_puzzle(bad_seed)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal seed accepted')
    first, second = generate_puzzle(271828, 10, 4), generate_puzzle(271828, 10, 4)
    for split in ('train', 'test'):
        for p, q in zip(first[split], second[split]):
            assert all(np.array_equal(p[k], q[k]) for k in ('input', 'output'))
    # Input-only solver works at every scale used by generation and at a
    # larger legal scale not sampled by generate_puzzle.
    rng = np.random.default_rng(999)
    for size in range(3, 10):
        p = _draw(rng, size, 0, [[1, 1], [2, 3]])
        _pair_invariants(p)
    corrupted = generate_puzzle(0, 2, 1)
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'pairs': total, 'input_shapes': len(shapes),
            'layouts': len(layouts), 'count_combinations': len(counts),
            'output_shapes': len(outputs), 'black_tile_pairs': black_tiles,
            'adjacent_repeated_rank_pairs': repeated_ranks,
            'separated_color_repeat_pairs': separated_colors}
