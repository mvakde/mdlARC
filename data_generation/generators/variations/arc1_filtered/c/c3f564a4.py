"""Completion of one grid-wide diagonal color cycle behind a distinct mask."""

import numpy as np


def _key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    sizes = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _reconstructions(inp):
    """Enumerate fully evidenced cycles using only an input grid."""
    h, w = inp.shape
    residue_base = np.add.outer(np.arange(h), np.arange(w))
    found = []
    for mask_color in np.unique(inp):
        visible = inp != mask_color
        if not np.any(~visible):
            continue
        for period in range(2, 10):
            residues = residue_base % period
            cycle = []
            for k in range(period):
                values = inp[visible & (residues == k)]
                if values.size == 0 or np.any(values != values[0]):
                    break
                cycle.append(int(values[0]))
            if len(cycle) != period or len(set(cycle)) != period:
                continue
            if int(mask_color) in cycle:
                continue
            output = np.asarray(cycle, dtype=np.int8)[residues]
            if np.array_equal(inp[visible], output[visible]):
                found.append((int(mask_color), period, output))
    return found


def _shape(rng):
    choice = int(rng.integers(0, 10))
    if choice == 0:
        return 16, 16
    if choice in (1, 2):
        short = int(rng.integers(9, 14))
        long = int(rng.integers(18, 28))
        return (short, long) if rng.random() < .5 else (long, short)
    return int(rng.integers(10, 26)), int(rng.integers(10, 26))


def _patch(mask, rng, small=False):
    h, w = mask.shape
    if small:
        rh = int(rng.integers(1, min(5, h) + 1))
        rw = int(rng.integers(1, min(6, w) + 1))
    else:
        rh = int(rng.integers(2, min(8, h) + 1))
        rw = int(rng.integers(2, min(9, w) + 1))
    r = int(rng.integers(0, h - rh + 1))
    c = int(rng.integers(0, w - rw + 1))
    mask[r:r + rh, c:c + rw] = True


def _block(mask, rng, edge=False):
    h, w = mask.shape
    rh = int(rng.integers(max(3, h // 4), max(4, 3 * h // 5) + 1))
    rw = int(rng.integers(max(3, w // 4), max(4, 3 * w // 5) + 1))
    rh, rw = min(rh, h), min(rw, w)
    r = int(rng.choice((0, h - rh))) if edge and rng.random() < .5 else int(rng.integers(0, h - rh + 1))
    c = int(rng.choice((0, w - rw))) if edge and rng.random() < .5 else int(rng.integers(0, w - rw + 1))
    for y in range(r, r + rh):
        left = max(0, c + int(rng.integers(-1, 2)))
        right = min(w, c + rw + int(rng.integers(-1, 2)))
        mask[y, left:right] = True


def _stair(mask, rng):
    h, w = mask.shape
    vertical = bool(rng.integers(0, 2))
    if vertical:
        y = int(rng.integers(0, max(1, h // 3)))
        x = int(rng.integers(0, w))
        length = int(rng.integers(max(4, h // 2), h + 1))
        for step in range(length):
            yy = min(h - 1, y + step)
            if step and step % int(rng.integers(2, 5)) == 0:
                x = max(0, min(w - 1, x + int(rng.choice((-1, 1)))))
            mask[yy, max(0, x - 1):min(w, x + int(rng.integers(1, 3)))] = True
    else:
        x = int(rng.integers(0, max(1, w // 3)))
        y = int(rng.integers(0, h))
        length = int(rng.integers(max(4, w // 2), w + 1))
        for step in range(length):
            xx = min(w - 1, x + step)
            if step and step % int(rng.integers(2, 5)) == 0:
                y = max(0, min(h - 1, y + int(rng.choice((-1, 1)))))
            mask[max(0, y - 1):min(h, y + int(rng.integers(1, 3))), xx] = True


def _thread(mask, rng):
    h, w = mask.shape
    if rng.random() < .5:
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, max(1, w // 2)))
        span = int(rng.integers(max(3, w // 3), w - c + 1))
        mask[r:min(h, r + int(rng.integers(1, 3))), c:c + span] = True
    else:
        c = int(rng.integers(0, w))
        r = int(rng.integers(0, max(1, h // 2)))
        span = int(rng.integers(max(3, h // 3), h - r + 1))
        mask[r:r + span, c:min(w, c + int(rng.integers(1, 3)))] = True


def _make_mask(rng, shape, regime):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    modes = ('patches', 'block', 'stair', 'threads', 'edge', 'scanlines')
    weights = ((25, 17, 16, 15, 13, 14) if regime == 'source_train'
               else (17, 24, 17, 12, 15, 15))
    mode = str(rng.choice(modes, p=np.asarray(weights) / sum(weights)))
    if mode == 'patches':
        for _ in range(int(rng.integers(2, 7))):
            _patch(mask, rng)
    elif mode in ('block', 'edge'):
        _block(mask, rng, mode == 'edge')
    elif mode == 'stair':
        _stair(mask, rng)
        if rng.random() < .5:
            _stair(mask, rng)
    elif mode == 'threads':
        for _ in range(int(rng.integers(2, 5))):
            _thread(mask, rng)
    else:
        mask[int(rng.integers(0, h)), :] = True
        if rng.random() < .65:
            mask[:, int(rng.integers(0, w))] = True
        _patch(mask, rng)

    # These edits compose with every base geometry, including train blocks.
    if rng.random() < .17:
        mask[:, int(rng.integers(0, w))] = True
    if rng.random() < .11:
        mask[int(rng.integers(0, h)), :] = True
    if rng.random() < .19:
        for y in range(h):
            mask[y, int(rng.integers(0, w))] = True
    for _ in range(int(rng.integers(0, 3))):
        if rng.random() < .65:
            _patch(mask, rng, small=True)
        else:
            _thread(mask, rng)
    if rng.random() < .24 and np.any(mask):
        # Visible pinholes within an otherwise continuous gap.
        for _ in range(int(rng.integers(1, 5))):
            points = np.argwhere(mask)
            y, x = points[int(rng.integers(0, len(points)))]
            mask[int(y), int(x)] = False
    return mask


def _source_output(out):
    if out.shape != (16, 16):
        return False
    rows, cols = np.indices(out.shape)
    return any(np.array_equal(out, (rows + cols) % p + 1) for p in (5, 6, 7, 8))


def _pair(rng, period, common, regime):
    for _ in range(32):
        shape = _shape(rng)
        if common is None:
            permutation = [int(x) for x in rng.permutation(10)]
            mask_color, palette = permutation[0], permutation[1:]
        else:
            mask_color, palette = common
        phase = int(rng.integers(0, period))
        rows, cols = np.indices(shape)
        out = np.asarray(palette[:period], dtype=np.int8)[(rows + cols + phase) % period]
        mask = _make_mask(rng, shape, regime)
        area = int(mask.sum())
        if not .035 * out.size <= area <= .52 * out.size:
            continue
        inp = out.copy()
        inp[mask] = mask_color
        if _source_output(out):
            continue
        solutions = _reconstructions(inp)
        if len(solutions) == 1 and np.array_equal(solutions[0][2], out):
            return {'input': inp, 'output': out}
    # Deterministic simple fallback for unusually unlucky masks.
    shape = _shape(rng)
    if common is None:
        permutation = [int(x) for x in rng.permutation(10)]
        mask_color, palette = permutation[0], permutation[1:]
    else:
        mask_color, palette = common
    rows, cols = np.indices(shape)
    phase = int(rng.integers(0, period))
    out = np.asarray(palette[:period], dtype=np.int8)[(rows + cols + phase) % period]
    inp = out.copy()
    inp[1:min(4, shape[0]), 2:min(6, shape[1])] = mask_color
    assert len(_reconstructions(inp)) == 1 and not _source_output(out)
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if num_train is not None and (type(num_train) is not int or not 2 <= num_train <= 10):
        raise ValueError('num_train must be an integer from 2 through 10')
    if num_test is not None and (type(num_test) is not int or not 1 <= num_test <= 4):
        raise ValueError('num_test must be an integer from 1 through 4')
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    # Only the one-step diagonal rule is universal. A shared local codebook is
    # sometimes present, but every pair remains solvable from its own input.
    shared = rng.random() < .38
    permutation = [int(x) for x in rng.permutation(10)]
    common = (permutation[0], tuple(permutation[1:])) if shared else None
    source_like = rng.random() < .19
    if source_like:
        train_periods = [5, 7] if num_train == 2 else [5, 6, 7]
        train_periods += [int(rng.integers(3, 10)) for _ in range(num_train - len(train_periods))]
        test_periods = [8] * num_test
    else:
        train_periods = [int(rng.integers(3, 10)) for _ in range(num_train)]
        test_periods = [int(rng.choice(train_periods)) if rng.random() < .42
                        else int(rng.integers(3, 10)) for _ in range(num_test)]
    puzzle = {'train': [], 'test': []}
    seen_inputs, seen_outputs = set(), set()
    for split, periods in (('train', train_periods), ('test', test_periods)):
        regime = 'source_train' if split == 'train' else 'target_test'
        for period in periods:
            for _ in range(16):
                pair = _pair(rng, period, common, regime)
                ik, ok = _key(pair['input']), _key(pair['output'])
                if ik not in seen_inputs and ok not in seen_outputs:
                    break
            else:
                raise AssertionError('could not create a distinct completed grid')
            puzzle[split].append(pair)
            seen_inputs.add(ik)
            seen_outputs.add(ok)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs, outputs, pairs = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            ik, ok = _key(inp), _key(out)
            assert ik not in inputs and ok not in outputs and (ik, ok) not in pairs
            inputs.add(ik)
            outputs.add(ok)
            pairs.add((ik, ok))
            changed = inp != out
            assert np.any(changed)
            mask_colors = np.unique(inp[changed])
            assert len(mask_colors) == 1
            mask_color = int(mask_colors[0])
            assert np.array_equal(changed, inp == mask_color)
            assert not np.any(out == mask_color)
            solutions = _reconstructions(inp)
            assert len(solutions) == 1
            color, period, reconstructed = solutions[0]
            assert color == mask_color and 3 <= period <= 9
            assert np.array_equal(out, reconstructed)
            assert len(np.unique(out)) == period
            assert len(np.unique(inp)) == period + 1
            assert not _source_output(out)
            sizes = _components(changed)
            assert sizes and .035 * inp.size <= changed.sum() <= .52 * inp.size
    # The complete training examples establish a common diagonal increment,
    # while the unique visible-cell reconstruction determines each test result.
    assert len(puzzle['train']) >= 2


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
        for s in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, periods, mask_colors, areas, components = set(), set(), set(), set(), set()
    puzzle_keys = set()
    no_clear_rows = fully_masked_rows = fully_masked_cols = 0
    broad_train = modest_test = shared_puzzles = mixed_puzzles = 0
    equal_period_puzzles = nine_period_pairs = 0
    edge_cases = pinholes = similar_components = nonsquare = nonmultiple = 0
    changed_sets = changed_orders = source_like = combined = 0
    within_shapes = within_periods = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        puzzle_key = tuple((_key(pair['input']), _key(pair['output']))
                           for split in ('train', 'test') for pair in puzzle[split])
        puzzle_keys.add(puzzle_key)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        records = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                mask = inp != out
                p = len(np.unique(out))
                rows, cols = np.indices(out.shape)
                assert np.array_equal(out, out[0, (rows + cols) % p])
                assert np.array_equal(inp[~mask], out[~mask])
                shapes.add(inp.shape)
                periods.add(p)
                mask_colors.add(int(inp[mask][0]))
                areas.add(int(mask.sum()))
                sizes = _components(mask)
                components.add(len(sizes))
                edge_cases += bool(mask[0].any() or mask[-1].any()
                                   or mask[:, 0].any() or mask[:, -1].any())
                pinholes += bool(np.any((~mask[1:-1, 1:-1]) & mask[:-2, 1:-1]
                                        & mask[2:, 1:-1] & mask[1:-1, :-2]
                                        & mask[1:-1, 2:]))
                similar_components += len(sizes) > 1 and sizes[1] * 2 >= sizes[0]
                nonsquare += inp.shape[0] != inp.shape[1]
                nonmultiple += inp.shape[1] % p != 0
                no_clear_rows += not np.any(~mask.any(axis=1))
                fully_masked_rows += bool(np.any(mask.all(axis=1)))
                fully_masked_cols += bool(np.any(mask.all(axis=0)))
                nine_period_pairs += p == 9
                if split == 'train':
                    broad_train += sizes[0] >= 25
                else:
                    modest_test += mask.sum() / mask.size < .22
                row = tuple(int(x) for x in out[0, :p])
                canonical = min(row[k:] + row[:k] for k in range(p))
                records.append((split, p, int(inp[mask][0]), canonical,
                                frozenset(int(x) for x in np.unique(out)), inp.shape))
        train_records = records[:len(puzzle['train'])]
        test_records = records[len(puzzle['train']):]
        equal_period_puzzles += any(a[1] == b[1] for a in train_records for b in test_records)
        shared_puzzles += len({x[2] for x in records}) == 1
        mixed_puzzles += len({x[2] for x in records}) > 1
        changed_sets += any(a[1] == b[1] and a[4] != b[4] for a in records for b in records)
        changed_orders += any(a[1] == b[1] and a[3] != b[3] for a in records for b in records)
        source_like += (5 in {x[1] for x in train_records}
                        and 7 in {x[1] for x in train_records}
                        and all(x[1] == 8 for x in test_records))
        within_shapes += len({x[5] for x in records}) > 1
        within_periods += len({x[1] for x in records}) > 1
        combined += any((pair['input'] != pair['output']).any(axis=1).all()
                        and (pair['input'] != pair['output'])[0].any()
                        for split in ('train', 'test') for pair in puzzle[split])
        if len(records) >= 5:
            assert len({_key(x['output']) for s in ('train', 'test') for x in puzzle[s]}) == len(records)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 55 and len(periods) == 7
    assert len(mask_colors) >= 8 and len(areas) >= 35
    assert min(components) == 1 and max(components) >= 3
    assert no_clear_rows > 0 and fully_masked_rows > 0 and fully_masked_cols > 0
    assert broad_train > 0 and modest_test > 0
    assert shared_puzzles > 0 and mixed_puzzles > 0
    assert equal_period_puzzles > 0 and nine_period_pairs > 0
    assert len(puzzle_keys) >= 40 and within_shapes >= 35 and within_periods >= 30
    assert changed_sets > 0 and changed_orders > 0 and source_like > 0
    assert edge_cases > 0 and pinholes > 0 and similar_components > 0
    assert nonsquare > 0 and nonmultiple > 0 and combined > 0
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4),
                                   (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1000 + i, nt, ns)
        validate_generated_puzzle(p)
        assert (nt is None or len(p['train']) == nt) and (ns is None or len(p['test']) == ns)
        assert _same_puzzle(p, generate_puzzle(1000 + i, nt, ns))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(2000, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
