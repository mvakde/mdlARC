"""Exception panels form one mosaic; collapse its consecutive vector runs.

Only NumPy and the standard library are used. No evidence files are loaded at
runtime. Generated labels are computed from unexpanded logical panels, while
solve recovers the mosaic and its vector boundaries from input pixels.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


SELF_TEST_SEEDS = (0, 931712904, 652171058, 324611483, 792483926, 940129423,
                  841221927, 928094929, 284490387, 876862839, 888088147,
                  919005353, 345863159, 145532916, 172806332, 183169387,
                  571164455, 857551408, 976353058, 296301680, 412279955,
                  443556958, 2404777, 690184651, 158843370, 331717954,
                  620151260, 872303133, 560522548, 484879572, 347320854,
                  984837157, 232638772, 212056932, 578410559, 384740745,
                  900663793, 788918747, 753921459, 514189799, 151984235,
                  462901196)


def _integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _segments(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(padded) == 1),
                    np.flatnonzero(np.diff(padded) == -1)))


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (np.all(a[[0, -1], :] == bg) and np.all(a[:, [0, -1]] == bg)):
        raise ValueError("a uniform background frame is required")
    occupied = a != bg
    row_mask, col_mask = np.any(occupied, axis=1), np.any(occupied, axis=0)
    rows, cols = _segments(row_mask), _segments(col_mask)
    if not rows or len(cols) < 3:
        raise ValueError("need bands with at least three panel copies")
    if not np.array_equal(occupied, row_mask[:, None] & col_mask[None, :]):
        raise ValueError("panels must be filled rectangles with background dividers")
    if len({int(y-x) for x, y in rows}) != 1 or len({int(y-x) for x, y in cols}) != 1:
        raise ValueError("all panels must have equal dimensions")
    if any(int(rows[i][0]-rows[i-1][1]) != 1 for i in range(1, len(rows))):
        raise ValueError("horizontal dividers must be one pixel thick")
    if any(int(cols[i][0]-cols[i-1][1]) != 1 for i in range(1, len(cols))):
        raise ValueError("vertical dividers must be one pixel thick")
    panels = [[a[r:s, c:d] for c, d in cols] for r, s in rows]
    counts, exemplars = {}, {}
    for band in panels:
        for p in band:
            k = _key(p)
            counts[k] = counts.get(k, 0) + 1
            exemplars[k] = p
    best = max(counts.values())
    winners = [k for k, v in counts.items() if v == best]
    if len(winners) != 1:
        raise ValueError("master must have a unique frequency maximum")
    master = exemplars[winners[0]]
    selected, positions = [], []
    for band in panels:
        exceptions = [i for i, p in enumerate(band) if not np.array_equal(p, master)]
        if len(exceptions) != 1:
            raise ValueError("every band must contain exactly one visible exception")
        positions.append(exceptions[0])
        selected.append(band[exceptions[0]])
    return dict(background=bg, panels=panels, master=master, selected=selected,
                positions=tuple(positions), rows=rows, cols=cols,
                margins=(int(rows[0][0]), int(a.shape[0]-rows[-1][1]),
                         int(cols[0][0]), int(a.shape[1]-cols[-1][1])))


def _collapse_pixels(mosaic):
    # Boundaries are measured independently on the complete joined mosaic.
    rs = np.r_[0, np.flatnonzero(np.any(mosaic[1:] != mosaic[:-1], axis=1)) + 1]
    cs = np.r_[0, np.flatnonzero(np.any(mosaic[:, 1:] != mosaic[:, :-1], axis=0)) + 1]
    return (mosaic[np.ix_(rs, cs)].copy(),
            np.diff(np.r_[rs, mosaic.shape[0]]),
            np.diff(np.r_[cs, mosaic.shape[1]]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select actual nonmaster bitmaps, join in band order, and compress."""
    info = _parse(input_grid)
    mosaic = np.concatenate(info['selected'], axis=0)
    return _collapse_pixels(mosaic)[0]


def _logical_label(variants):
    """Construct labels before expansion, using consecutive tuple comparison."""
    rows = [tuple(map(int, row)) for panel in variants for row in panel]
    kept_rows = [row for i, row in enumerate(rows) if i == 0 or row != rows[i-1]]
    columns = list(zip(*kept_rows))
    kept_columns = [col for i, col in enumerate(columns)
                    if i == 0 or col != columns[i-1]]
    return np.array(list(zip(*kept_columns)), dtype=np.int8)


def _profile(rng, logical_size, maximum):
    # At least one repeated row/column, with both uniform and uneven options.
    if maximum < logical_size + 1:
        raise ValueError("insufficient room for a repeated run")
    if maximum >= 2 * logical_size and rng.random() < .22:
        return np.full(logical_size, int(rng.integers(2, maximum // logical_size + 1)))
    sizes = np.ones(logical_size, dtype=int)
    additions = int(rng.integers(1, maximum - logical_size + 1))
    for _ in range(additions):
        sizes[int(rng.integers(logical_size))] += 1
    return sizes


def _joint_panels(rng, bands, h, w, colors):
    """One exception erases a seam; another is its single-row witness."""
    a, b = map(int, colors[:2])
    master = rng.choice(colors, size=(h, w)).astype(np.int8)
    master[-1] = master[0]
    seam = int(rng.integers(1, w))
    witness = int(rng.integers(1, h-1))
    master[:, seam] = master[:, seam-1]
    old = int(master[witness, seam])
    master[witness, seam] = int(rng.choice([c for c in colors if c != old]))
    spare = int(rng.choice([c for c in range(w) if c not in (seam-1, seam)]))
    # A count-preserving exception can be made away from the seam.
    master[0, spare] = master[-1, spare] = a
    master[witness, spare] = b
    erased = master.copy()
    erased[witness, seam] = erased[witness, seam-1]
    exposed = master.copy()
    exposed[0, spare], exposed[witness, spare] = b, a
    # Put the sole seam witness at an end. A central witness in three bands
    # would form a palindrome and fail to teach top-to-bottom stacking.
    variants = [exposed] + [erased.copy() for _ in range(1, bands)]
    if rng.random() < .5:
        variants = variants[::-1]
    if rng.random() < .5:
        master = master[:, ::-1].copy()
        variants = [p[:, ::-1].copy() for p in variants]
    return master, variants


def _join_panels(rng, bands, h, w, colors):
    """Common edge rows merge across joins; separated edge columns survive."""
    a, b = map(int, colors[:2])
    edge = rng.choice(colors, size=w).astype(np.int8)
    edge[0] = edge[-1] = a
    edge[int(rng.integers(1, w-1))] = b
    middle = np.where(edge == a, b, a).astype(np.int8)
    master = np.tile(edge, (h, 1))
    pivot = int(rng.integers(1, h-1))
    master[pivot] = middle
    variants = []
    for i in range(bands):
        p = master.copy()
        p[pivot] = a if i % 2 == 0 else b
        variants.append(p)
    # Both differ from the nonsolid master middle; outer rows are untouched.
    if rng.random() < .5:
        variants = variants[::-1]
    return master, variants


def _ordinary_panels(rng, bands, h, w, colors):
    style = int(rng.integers(5))
    rr, cc = np.indices((h, w))
    if style == 0:
        master = rng.choice(colors, (h, w)).astype(np.int8)
    elif style == 1:
        master = colors[(rr + cc) % len(colors)].astype(np.int8)
    elif style == 2:
        master = colors[((rr == h//2) | (cc == w//2)).astype(int)].astype(np.int8)
    elif style == 3:
        master = colors[((rr == 0) | (rr == h-1) | (cc == 0) | (cc == w-1)).astype(int)].astype(np.int8)
    else:
        master = rng.choice(colors, (h, w)).astype(np.int8)
        master[-1] = master[0]
    if len(np.unique(master)) < 2:
        master[0, 0], master[-1, -1] = colors[:2]
    variants = []
    for _ in range(bands):
        if variants and rng.random() < .18:
            variants.append(variants[int(rng.integers(len(variants)))].copy())
            continue
        p = master.copy()
        changes = int(rng.integers(1, max(2, h*w//3 + 1)))
        cells = rng.choice(h*w, changes, replace=False)
        for cell in cells:
            r, c = divmod(int(cell), w)
            p[r, c] = int(rng.choice([v for v in colors if v != p[r, c]]))
        variants.append(p)
    return master, variants


def _make_pair(rng, bands, copies, kind, palette):
    bg, colors = palette[0], np.array(palette[1:], dtype=np.int8)
    max_h, max_w = 29 // bands - 1, 29 // copies - 1
    if kind in ('joint', 'join'):
        h = int(rng.integers(3, min(4, max_h-1) + 1))
        w = int(rng.integers(3, min(4, max_w-1) + 1))
        builder = _joint_panels if kind == 'joint' else _join_panels
    else:
        h = int(rng.integers(2, min(5, max_h-1) + 1))
        w = int(rng.integers(2, min(5, max_w-1) + 1))
        builder = _ordinary_panels
    master, variants = builder(rng, bands, h, w, colors)
    heights, widths = _profile(rng, h, max_h), _profile(rng, w, max_w)
    if kind == 'joint':
        # The exchanged logical cells have equal pixel area, so counting colors
        # cannot identify this exception even before compression. The final row
        # is outside the exchange and provides a nonuniform expansion witness.
        options = [(common, last) for common in range(1, max_h+1)
                   for last in range(1, max_h+1)
                   if common != last and (h-1)*common+last <= max_h]
        common, last = options[int(rng.integers(len(options)))]
        heights[:] = common
        heights[-1] = last
        if len(set(widths)) == 1:
            widths[:] = 1
            widths[int(rng.integers(w))] = 2
    expand = lambda p: np.repeat(np.repeat(p, heights, axis=0), widths, axis=1)
    expanded_master = expand(master)
    expanded_variants = [expand(p) for p in variants]
    ph, pw = expanded_master.shape
    positions = rng.integers(copies, size=bands)
    if kind in ('joint', 'join') and len(set(map(int, positions))) == 1:
        positions[-1] = (int(positions[0]) + int(rng.integers(1, copies))) % copies
    base_h, base_w = bands * (ph+1) + 1, copies * (pw+1) + 1
    extra_h = int(rng.integers(0, min(4, 30-base_h) + 1))
    extra_w = int(rng.integers(0, min(4, 30-base_w) + 1))
    top_extra = int(rng.integers(extra_h+1))
    left_extra = int(rng.integers(extra_w+1))
    grid = np.full((base_h+extra_h, base_w+extra_w), bg, dtype=np.int8)
    for band in range(bands):
        for col in range(copies):
            p = expanded_variants[band] if col == positions[band] else expanded_master
            r0, c0 = top_extra+1+band*(ph+1), left_extra+1+col*(pw+1)
            grid[r0:r0+ph, c0:c0+pw] = p
    return {'input': grid, 'output': _logical_label(variants)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(('bands', 'copies')))
    cap = int(rng.integers(2, 4)) if shift == 'bands' else int(rng.integers(3, 5))
    shared = bool(rng.random() < .25)
    shared_palette = tuple(map(int, rng.choice(10, int(rng.integers(3, 6)), replace=False)))
    witness_indices = rng.choice(nt, size=2, replace=False)
    witnesses = {int(witness_indices[0]): 'joint', int(witness_indices[1]): 'join'}
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            kind = witnesses.get(index, 'ordinary') if split == 'train' else str(rng.choice(('ordinary', 'joint', 'join')))
            bands = (cap + 1 if split == 'test' else int(rng.integers(1, cap+1))) if shift == 'bands' else int(rng.integers(1, 6))
            copies = (cap + 1 if split == 'test' else int(rng.integers(3, cap+1))) if shift == 'copies' else int(rng.integers(3, 6))
            if kind in ('joint', 'join'):
                bands = max(2, min(bands, 4))
            for _ in range(32):
                palette = shared_palette if shared else tuple(map(int, rng.choice(10, int(rng.integers(3, 6)), replace=False)))
                pair = _make_pair(rng, bands, copies, kind, palette)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input in 32 attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def _recurrence(a, axis):
    vectors = a if axis == 0 else a.T
    keys = [tuple(map(int, v)) for v in vectors]
    return len(set(keys)) < len(keys)


def _analyze(pair):
    input_grid, output = _grid(pair['input']), _grid(pair['output'])
    info = _parse(input_grid)
    mosaic = np.concatenate(info['selected'], axis=0)
    recovered, row_runs, col_runs = _collapse_pixels(mosaic)
    assert np.array_equal(recovered, output), 'incorrect compact exception mosaic'
    assert info['background'] not in output, 'background must be cropped'
    assert np.array_equal(np.repeat(np.repeat(output, row_runs, axis=0), col_runs, axis=1), mosaic), 'lossless run reconstruction failed'
    assert not np.any(np.all(output[1:] == output[:-1], axis=1))
    assert not np.any(np.all(output[:, 1:] == output[:, :-1], axis=0))
    assert len(np.unique(input_grid)) >= 3
    # These are observable counterfactuals, not latent generation mode labels.
    global_boundaries = np.any(mosaic[:, 1:] != mosaic[:, :-1], axis=0)
    joint_seam = any(np.any(global_boundaries &
                           ~np.any(p[:, 1:] != p[:, :-1], axis=0))
                     for p in info['selected'])
    cross_join = any(np.array_equal(a[-1], b[0])
                     for a, b in zip(info['selected'], info['selected'][1:]))
    master_stack = np.concatenate([info['master']] * len(info['selected']), axis=0)
    all_panels = np.concatenate([np.concatenate(row, axis=1)
                                 for row in info['panels']], axis=0)
    balanced = any(not np.array_equal(p, info['master']) and
                   np.array_equal(np.bincount(p.ravel(), minlength=10),
                                  np.bincount(info['master'].ravel(), minlength=10))
                   for p in info['selected'])
    reversed_bands = _collapse_pixels(np.concatenate(info['selected'][::-1], axis=0))[0]
    return dict(bands=len(info['rows']), copies=len(info['cols']),
                panel_shape=info['master'].shape, background=info['background'],
                positions=info['positions'], row_runs=tuple(map(int, row_runs)),
                col_runs=tuple(map(int, col_runs)),
                joint_seam=bool(joint_seam), cross_join=cross_join,
                row_recurrence=_recurrence(output, 0), col_recurrence=_recurrence(output, 1),
                balanced=balanced,
                order_sensitive=not np.array_equal(reversed_bands, output),
                omit_a=not np.array_equal(_collapse_pixels(master_stack)[0], output),
                keep_all=not np.array_equal(_collapse_pixels(all_panels)[0], output),
                omit_b=not np.array_equal(mosaic, output))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train, test = set(), [], []
    for split, target in (('train', train), ('test', test)):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a = _grid(pair['input'])
            key = _key(a)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            m = _analyze(pair)
            assert max(m['row_runs']) > 1 and max(m['col_runs']) > 1
            assert m['omit_b'] and m['keep_all']
            target.append(m)
    assert any(m['joint_seam'] for m in train), 'show globally shared column boundaries'
    assert any(m['cross_join'] for m in train), 'show merging across a band join'
    assert any(m['row_recurrence'] for m in train), 'show separated row recurrence'
    assert any(m['col_recurrence'] for m in train), 'show separated column recurrence'
    assert any(m['balanced'] for m in train), 'show full-bitmap selection beyond color counts'
    assert any(m['order_sensitive'] for m in train), 'show top-to-bottom band order'
    assert any(len(set(m['positions'])) > 1 for m in train), 'reject fixed-column extraction'
    assert any(m['omit_a'] for m in train), 'master substitution must change the answer'
    assert (min(m['bands'] for m in test) > max(m['bands'] for m in train)
            or min(m['copies'] for m in test) > max(m['copies'] for m in train)), 'missing count extrapolation'


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test() -> dict:
    """Bounded: 42 fixed seeds, all legal counts, and small mutation checks.

    Puzzles are processed one at a time; no files or execution traces are saved.
    Returned counters describe observed coverage, not an independent oracle.
    """
    # Hand-worked outputs, independent of the sampled logical-label constructor.
    fixtures = (
        ([[1,1,2],[2,1,1],[1,1,2]],
         [[[1,1,2],[2,2,1],[1,1,2]], [[1,1,1],[2,1,2],[1,1,2]]],
         [[1,1,2],[2,2,1],[1,1,2],[1,1,1],[2,1,2],[1,1,2]]),
        ([[1,2,1],[2,1,2],[1,2,1]],
         [[[1,2,1],[1,1,1],[1,2,1]], [[1,2,1],[2,2,2],[1,2,1]]],
         [[1,2,1],[1,1,1],[1,2,1],[2,2,2],[1,2,1]]),
    )
    for master, variants, expected in fixtures:
        expand = lambda p: np.repeat(np.repeat(np.array(p), [1,1,2], axis=0),
                                     [2,1,1], axis=1)
        canvas = np.zeros((11, 16), dtype=np.int8)
        for b in range(2):
            for c in range(3):
                p = variants[b] if c == (2, 0)[b] else master
                canvas[1+b*5:5+b*5, 1+c*5:5+c*5] = expand(p)
        assert np.array_equal(solve(canvas), np.array(expected))
    signatures, shapes, geometries, profiles, structures = set(), set(), set(), set(), set()
    counts = {name: 0 for name in ('pairs', 'joint_seam', 'cross_join', 'balanced', 'order_sensitive',
                                  'row_recurrence', 'col_recurrence')}
    backgrounds, bands, copies = set(), set(), set()
    for seed in SELF_TEST_SEEDS:
        p = generate_puzzle(seed)
        assert _puzzle_digest(p) == _puzzle_digest(generate_puzzle(seed))
        signatures.add(_puzzle_digest(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                m = _analyze(pair)
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                counts['pairs'] += 1
                for name in counts:
                    if name != 'pairs':
                        counts[name] += int(m[name])
                shapes.add(pair['input'].shape)
                geometries.add((m['bands'], m['copies'], m['panel_shape']))
                profiles.add((m['row_runs'], m['col_runs']))
                backgrounds.add(m['background'])
                bands.add(m['bands'])
                copies.add(m['copies'])
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in pair['input'].ravel())
                structures.add(hashlib.sha256(bytes(pair['input'].shape)+normalized).hexdigest())
    assert len(signatures) == 42 and len(backgrounds) == 10
    assert len(shapes) >= 30 and len(geometries) >= 40
    assert len(profiles) >= 60 and len(structures) >= 150
    assert len(bands) >= 4 and copies == {3, 4, 5}
    assert all(counts[n] >= 42 for n in counts if n != 'pairs')
    legal_calls = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12000 + 10*nt + ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
            legal_calls += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None), (None, 1), ('4', 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    base = generate_puzzle(0, 2, 1)
    for defect in ('missing', 'label', 'divider', 'order'):
        bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()}
                    for p in base[s]] for s in base}
        p = bad['train'][0]
        info = _parse(p['input'])
        if defect == 'missing':
            r, s = info['rows'][0]
            c, d = info['cols'][info['positions'][0]]
            p['input'][r:s, c:d] = info['master']
        elif defect == 'label':
            p['output'][0, 0] = (int(p['output'][0, 0])+1) % 10
        elif defect == 'divider':
            r = int(info['rows'][0][1])
            p['input'][r, int(info['cols'][0][0])] = int(info['master'][0, 0])
        else:
            reversed_output = _collapse_pixels(np.concatenate(info['selected'][::-1], axis=0))[0]
            if np.array_equal(p['output'], reversed_output):
                p = bad['train'][1]
                info = _parse(p['input'])
                reversed_output = _collapse_pixels(np.concatenate(info['selected'][::-1], axis=0))[0]
            assert not np.array_equal(p['output'], reversed_output)
            p['output'] = reversed_output
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('validator accepted '+defect)
    return dict(seeds=42, legal_count_calls=legal_calls, distinct_puzzles=len(signatures),
                input_shapes=len(shapes), geometries=len(geometries),
                run_profiles=len(profiles), color_normalized_inputs=len(structures),
                background_colors=len(backgrounds), band_counts=sorted(bands),
                copy_counts=sorted(copies), **counts)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
