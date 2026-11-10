"""Generate lattices of panels whose binary interiors are complemented."""

import numpy as np


# Exact source inputs.  Matching contents, rather than banning their palette or
# dimensions, keeps the original 0/8 color roles available to new puzzles.
_SOURCE_INPUTS = {
    tuple(s.split('|')) for s in (
        '0000000000000|0808080808000|0080088808800|0808080800880|0000000000000|0888088808800|0808000000080|0080088808800|0000000000000',
        '0000000000000|0808080008800|0808008000800|0888088800880|0000000000000',
        '000000000|088808080|000000880|008008080|000000000|088008000|008808880|080000800|000000000',
        '0000000000000|0800080800800|0808088800880|0888080008080|0000000000000|0808080008880|0888008008080|0088080800800|0000000000000|0888080800080|0880088008880|0080008808000|0000000000000',
    )
}
_MODES = ('random', 'sparse', 'dense', 'repeat', 'orbit', 'alternating', 'uniform', 'extremes', 'mixed')


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _checked_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a non-negative integer')
    return int(seed)


def _checked_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    return int(value)


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _checked_count(num_train, 'num_train')
    num_test = _checked_count(num_test, 'num_test')
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError('num_train must be between 2 and 10')
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError('num_test must be between 1 and 4')
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _sample_geometry(rng):
    # Geometry is a puzzle-level convention.  The source geometry remains common.
    if rng.random() < 0.43:
        return (3, 3, 1, 1)  # panel height, width, separator, border
    h = int(rng.choice((2, 3, 4, 5), p=(.2, .35, .3, .15)))
    w = int(rng.choice((2, 3, 4, 5), p=(.2, .35, .3, .15)))
    s = int(rng.choice((1, 2), p=(.7, .3)))
    b = int(rng.choice((1, 2), p=(.72, .28)))
    return h, w, s, b


def _max_panels(d, s, b):
    return (30 - 2 * b + s) // (d + s)


def _shape(layout, geometry):
    rows, cols = layout
    h, w, s, b = geometry
    return (2*b + rows*h + (rows-1)*s, 2*b + cols*w + (cols-1)*s)


def _random_mask(rng, h, w, probability=None):
    if probability is None:
        probability = float(rng.uniform(.2, .8))
    mask = rng.random((h, w)) < probability
    if not mask.any():
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
    if mask.all():
        mask[int(rng.integers(h)), int(rng.integers(w))] = False
    return mask


def _symmetries(mask):
    variants = [mask, mask[::-1, :], mask[:, ::-1], mask[::-1, ::-1]]
    if mask.shape[0] == mask.shape[1]:
        transposed = mask.T
        variants += [transposed, transposed[::-1, :], transposed[:, ::-1], transposed[::-1, ::-1]]
    return variants


def _sample_motifs(rng, n, h, w, mode, columns):
    area = h*w
    if mode == 'random':
        p = float(rng.uniform(.15, .85))
        motifs = [rng.random((h, w)) < p for _ in range(n)]
    elif mode in ('sparse', 'dense'):
        motifs = []
        for _ in range(n):
            k = int(rng.integers(1, max(2, area//3 + 1)))
            chosen = rng.choice(area, size=k, replace=False)
            mask = np.zeros(area, dtype=bool)
            mask[chosen] = True
            motifs.append((~mask if mode == 'dense' else mask).reshape(h, w))
    elif mode == 'repeat':
        base = _random_mask(rng, h, w)
        motifs = [base.copy() for _ in range(n)]
        for j in range(2, n):
            if rng.random() < .45:
                flips = rng.choice(area, size=int(rng.integers(1, min(3, area)+1)), replace=False)
                motifs[j].flat[flips] = ~motifs[j].flat[flips]
    elif mode == 'orbit':
        base = _random_mask(rng, h, w)
        variants = _symmetries(base)
        motifs = [variants[int(rng.integers(len(variants)))].copy() for _ in range(n)]
        if n >= 2:
            motifs[0], motifs[1] = base.copy(), variants[1].copy()
    elif mode == 'alternating':
        a = _random_mask(rng, h, w)
        b = _random_mask(rng, h, w)
        if np.array_equal(a, b):
            b = ~a
        motifs = [a.copy() if (j//columns + j%columns) % 2 == 0 else b.copy() for j in range(n)]
    elif mode == 'uniform':
        base = _random_mask(rng, h, w)
        motifs = [base.copy() for _ in range(n)]
    elif mode == 'extremes':
        motifs = [_random_mask(rng, h, w) for _ in range(n)]
        if n >= 2:
            motifs[0] = np.zeros((h, w), dtype=bool)
            motifs[1] = np.ones((h, w), dtype=bool)
    elif mode == 'mixed':
        motifs = []
        for _ in range(n):
            p = float(rng.uniform(.06, .94))
            motifs.append(rng.random((h, w)) < p)
    else:
        raise ValueError('unknown motif mode')
    # A pair must show source ink and an interior blank.  Empty/full panels
    # themselves remain allowed whenever another panel supplies the other state.
    if not any(m.any() for m in motifs):
        motifs[-1].flat[int(rng.integers(area))] = True
    if not any((~m).any() for m in motifs):
        motifs[-1].flat[int(rng.integers(area))] = False
    return motifs


def _make_pair(rng, layout, geometry, colors, mode):
    h, w, s, b = geometry
    background, source, result = colors
    height, width = _shape(layout, geometry)
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    motifs = _sample_motifs(rng, layout[0]*layout[1], h, w, mode, layout[1])
    for j, mask in enumerate(motifs):
        pr, pc = divmod(j, layout[1])
        r = b + pr*(h+s)
        c = b + pc*(w+s)
        inp[r:r+h, c:c+w] = np.where(mask, source, background)
        out[r:r+h, c:c+w] = np.where(mask, background, result)
    return {'input': inp, 'output': out}


def _layout_plan(rng, seed, geometry, num_train, num_test):
    h, w, s, b = geometry
    mr, mc = _max_panels(h, s, b), _max_panels(w, s, b)
    kind = seed % 6
    # The six regimes cover two-axis growth, familiar size, smaller tests,
    # height-only and width-only growth, and unrestricted mixed layouts.
    if kind == 0:
        capr, capc = int(rng.integers(2, mr)), int(rng.integers(2, mc))
        anchor = (capr, capc)
        test_first = (int(rng.integers(capr+1, mr+1)), int(rng.integers(capc+1, mc+1)))
    elif kind == 1:
        capr, capc = mr, mc
        anchor = (int(rng.integers(2, mr+1)), int(rng.integers(2, mc+1)))
        test_first = anchor
    elif kind == 2:
        capr, capc = mr, mc
        anchor = (int(rng.integers(3, mr+1)), int(rng.integers(3, mc+1)))
        test_first = (int(rng.integers(1, anchor[0])), int(rng.integers(1, anchor[1])))
    elif kind == 3:
        capr, capc = int(rng.integers(2, mr)), int(rng.integers(2, mc+1))
        anchor = (capr, capc)
        test_first = (int(rng.integers(capr+1, mr+1)), capc)
    elif kind == 4:
        capr, capc = int(rng.integers(2, mr+1)), int(rng.integers(2, mc))
        anchor = (capr, capc)
        test_first = (capr, int(rng.integers(capc+1, mc+1)))
    else:
        capr, capc = mr, mc
        anchor = (int(rng.integers(2, mr+1)), int(rng.integers(2, mc+1)))
        test_first = (int(rng.integers(1, mr+1)), int(rng.integers(1, mc+1)))
    train = [anchor]
    for _ in range(num_train-1):
        if rng.random() < .3:
            train.append((int(rng.integers(1, min(3, capr)+1)), int(rng.integers(1, min(3, capc)+1))))
        else:
            train.append((int(rng.integers(1, capr+1)), int(rng.integers(1, capc+1))))
    test = [test_first]
    for _ in range(num_test-1):
        if rng.random() < .35:
            test.append(train[int(rng.integers(len(train)))])
        else:
            test.append((int(rng.integers(1, mr+1)), int(rng.integers(1, mc+1))))
    return train, test


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _source_copy(inp):
    return tuple(''.join(str(int(x)) for x in row) for row in inp) in _SOURCE_INPUTS


def _panel_counts(length, d, s, b):
    remaining = length - 2*b + s
    if remaining < d+s or remaining % (d+s):
        return None
    count = remaining // (d+s)
    return count if count >= 1 else None


def _predict(inp, result, geometry):
    h, w, s, b = geometry
    rows = _panel_counts(inp.shape[0], h, s, b)
    cols = _panel_counts(inp.shape[1], w, s, b)
    if rows is None or cols is None:
        return None
    bg = int(inp[0, 0])
    colors = set(int(x) for x in np.unique(inp))
    if len(colors) != 2 or bg not in colors or result in colors:
        return None
    source = next(x for x in colors if x != bg)
    out = np.full(inp.shape, bg, dtype=np.int8)
    used = np.zeros(inp.shape, dtype=bool)
    for pr in range(rows):
        for pc in range(cols):
            r, c = b+pr*(h+s), b+pc*(w+s)
            block = inp[r:r+h, c:c+w]
            out[r:r+h, c:c+w] = np.where(block == bg, result, bg)
            used[r:r+h, c:c+w] = True
    if not np.all(inp[~used] == bg):
        return None
    return out


def _learn_candidates(train):
    result_colors = set()
    for pair in train:
        bg = int(pair['input'][0, 0])
        result_colors.update(int(x) for x in np.unique(pair['output']) if int(x) != bg)
    if len(result_colors) != 1:
        return []
    result = next(iter(result_colors))
    candidates = []
    for h in range(2, 6):
        for w in range(2, 6):
            for s in (1, 2):
                for b in (1, 2):
                    geom = (h, w, s, b)
                    if all((prediction := _predict(p['input'], result, geom)) is not None
                           and np.array_equal(prediction, p['output']) for p in train):
                        candidates.append((result, geom))
    return candidates


def _learn_geometry_model(train):
    """Recover the result color and lattice choices solely from training pairs."""
    individual = [_learn_candidates([pair]) for pair in train]
    if not individual or any(not choices for choices in individual):
        return [], False
    common = set(individual[0])
    for choices in individual[1:]:
        common.intersection_update(choices)
    if common:
        return sorted(common), False
    # When geometry changes between pairs, each training input/output must
    # identify its own partition.  Test inputs must do so as well.
    if any(len(choices) != 1 for choices in individual):
        return [], True
    return sorted({choices[0] for choices in individual}), True


def _validate_observed(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'splits must be lists')
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, 'pair counts')
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer), 'grid type')
                _require(all(1 <= size <= 30 for size in grid.shape), 'grid dimensions')
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, 'grid colors')
            _require(inp.shape == out.shape and not np.array_equal(inp, out), 'pair dimensions or identity')
            key = _grid_key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            _require(not _source_copy(inp), 'exact original input copy')
    candidates, variable_geometry = _learn_geometry_model(puzzle['train'])
    _require(len(candidates) >= 1, 'training does not reveal a consistent rule')
    _require(len({result for result, _ in candidates}) == 1, 'result color varies within puzzle')
    for pair in puzzle['test']:
        predictions = []
        for result, geom in candidates:
            predicted = _predict(pair['input'], result, geom)
            if predicted is not None:
                predictions.append(predicted)
        _require(predictions, 'test partition is not supported by training')
        if variable_geometry:
            _require(len(predictions) == 1, 'test input does not expose one partition')
        _require(all(np.array_equal(predictions[0], p) for p in predictions), 'test output is geometrically ambiguous')
        _require(np.array_equal(predictions[0], pair['output']), 'test output violates learned rule')


def validate_generated_puzzle(puzzle: dict) -> None:
    _validate_observed(puzzle)


def _puzzles_equal(a, b):
    if a.keys() != b.keys():
        return False
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[split], b[split]) for k in ('input', 'output'))
        for split in ('train', 'test'))


def _assemble_puzzle(rng, seed, base_geometry, other_geometry, num_train, num_test,
                     result, fixed_palette, fixed_background, fixed_source):
    if other_geometry is None:
        bound = base_geometry
    else:
        bound = tuple(max(x, y) for x, y in zip(base_geometry, other_geometry))
    train_layouts, test_layouts = _layout_plan(rng, seed, bound, num_train, num_test)
    if other_geometry is not None:
        train_layouts[1] = (2, 2)
    if len(set(train_layouts + test_layouts)) == 1:
        train_layouts[-1] = (1, train_layouts[-1][1])
    mode_order = list(rng.permutation(_MODES))
    puzzle = {'train': [], 'test': []}
    seen = set()
    first_colors = None
    for j, (split, layout) in enumerate([('train', l) for l in train_layouts] + [('test', l) for l in test_layouts]):
        geometry = base_geometry if other_geometry is None or j == 0 else (
            other_geometry if j == 1 or j == num_train else
            (base_geometry if rng.random() < .5 else other_geometry)
        )
        if fixed_palette:
            bg, source = fixed_background, fixed_source
        elif j == 0:
            bg, source = fixed_background, fixed_source
            first_colors = (bg, source)
        else:
            options = [x for x in range(10) if x != result]
            if j == 1:
                bg = int(rng.choice([x for x in options if x != first_colors[0]]))
                source = int(rng.choice([x for x in options if x not in (bg, first_colors[1])]))
            else:
                bg, source = [int(x) for x in rng.choice(options, size=2, replace=False)]
        mode = 'random' if j == 0 or (j == 1 and other_geometry is not None) else mode_order[(j-1) % len(mode_order)]
        colors = bg, source, result
        # Equal layouts and palettes are legal; redraw their motifs if needed.
        for attempt in range(256):
            sampled_mode = mode if attempt < 16 else 'random'
            pair = _make_pair(rng, layout, geometry, colors, sampled_mode)
            key = _grid_key(pair['input'])
            if key not in seen and not _source_copy(pair['input']):
                break
        else:
            raise RuntimeError('could not construct distinct pair')
        seen.add(key)
        puzzle[split].append(pair)
    return puzzle


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _checked_seed(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)
    geometry = _sample_geometry(rng)
    result = int(rng.integers(10))
    fixed_palette = bool(rng.random() < .34)
    if rng.random() < .12:
        fixed_background, fixed_source = 0, 8
        if result in (0, 8):
            result = int(rng.choice([x for x in range(10) if x not in (0, 8)]))
    else:
        fixed_background, fixed_source = [int(x) for x in rng.choice([x for x in range(10) if x != result], size=2, replace=False)]
    variable = seed % 7 == 5
    if variable:
        for _ in range(24):
            alternative = _sample_geometry(rng)
            if alternative == geometry:
                continue
            try:
                puzzle = _assemble_puzzle(rng, seed, geometry, alternative, num_train, num_test,
                                          result, fixed_palette, fixed_background, fixed_source)
                validate_generated_puzzle(puzzle)
            except (AssertionError, RuntimeError):
                continue
            return puzzle
    # A bounded fixed-geometry fallback is always available for legal counts.
    puzzle = _assemble_puzzle(rng, seed, geometry, None, num_train, num_test,
                              result, fixed_palette, fixed_background, fixed_source)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    # These are measured from returned arrays, rather than sampling metadata.
    axes = {name: 0 for name in (
        'canonical', 'rectangular', 'thick_separator', 'thick_border',
        'varying_palette', 'single_panel', 'single_row', 'single_column',
        'tall', 'wide', 'large', 'repeated', 'uniform', 'empty', 'full',
        'sparse', 'dense', 'extra_blank_line', 'original_palette',
        'grow_both', 'grow_rows', 'grow_columns', 'familiar', 'smaller',
        'related', 'combined', 'variable_geometry',
    )}
    geometries, layouts = set(), set()
    for puzzle in puzzles:
        candidates, varying_geometry = _learn_geometry_model(puzzle['train'])
        _require(bool(candidates), 'training geometry is not learned')
        axes['variable_geometry'] += varying_geometry
        puzzle_geometries = {geom for _, geom in candidates}
        geometries.update(puzzle_geometries)
        axes['canonical'] += (3, 3, 1, 1) in puzzle_geometries
        axes['rectangular'] += any(h != w for h, w, s, b in puzzle_geometries)
        axes['thick_separator'] += any(s == 2 for h, w, s, b in puzzle_geometries)
        axes['thick_border'] += any(b == 2 for h, w, s, b in puzzle_geometries)
        train_layouts, test_layouts, palettes = [], [], set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                if varying_geometry:
                    compatible = ([geom for _, geom in _learn_candidates([pair])]
                                  if split == 'train' else
                                  [geom for result, geom in candidates
                                   if _predict(inp, result, geom) is not None])
                    _require(len(compatible) == 1, 'individual pair geometry is ambiguous')
                    geom = compatible[0]
                else:
                    _require(len(candidates) == 1, 'fixed geometry is not uniquely learned')
                    geom = candidates[0][1]
                h, w, s, b = geom
                bg = int(inp[0, 0])
                source = next(int(x) for x in np.unique(inp) if int(x) != bg)
                palettes.add((bg, source))
                axes['original_palette'] += (bg, source) == (0, 8)
                nr = _panel_counts(inp.shape[0], h, s, b)
                nc = _panel_counts(inp.shape[1], w, s, b)
                layout = (nr, nc)
                layouts.add(layout)
                (train_layouts if split == 'train' else test_layouts).append(layout)
                axes['single_panel'] += nr*nc == 1
                axes['single_row'] += nr == 1
                axes['single_column'] += nc == 1
                axes['tall'] += nr > nc
                axes['wide'] += nc > nr
                axes['large'] += max(inp.shape) >= 25
                masks = [inp[b+r*(h+s):b+r*(h+s)+h,
                             b+c*(w+s):b+c*(w+s)+w] != bg
                         for r in range(nr) for c in range(nc)]
                amounts = [int(mask.sum()) for mask in masks]
                unique = len({mask.tobytes() for mask in masks})
                axes['repeated'] += unique < len(masks)
                axes['uniform'] += unique == 1 and len(masks) > 1
                axes['empty'] += amounts.count(0)
                axes['full'] += amounts.count(h*w)
                axes['sparse'] += sum(k <= max(1, h*w//3) for k in amounts) > len(amounts)/2
                axes['dense'] += sum(k >= h*w-max(1, h*w//3) for k in amounts) > len(amounts)/2
                axes['extra_blank_line'] += (
                    any(np.all(inp[i, :] == bg) for i in range(b, inp.shape[0]-b)
                        if (i-b) % (h+s) < h)
                    or any(np.all(inp[:, i] == bg) for i in range(b, inp.shape[1]-b)
                           if (i-b) % (w+s) < w)
                )
                if len(masks) >= 2:
                    variants = _symmetries(masks[0])
                    axes['related'] += any(
                        any(np.array_equal(m, variant) for variant in variants)
                        and not np.array_equal(m, masks[0]) for m in masks[1:]
                    )
                axes['combined'] += h != w and s == 2 and unique < len(masks)
        axes['varying_palette'] += len(palettes) > 1
        _require(len(set(train_layouts+test_layouts)) >= 2, 'no within-puzzle layout variation')
        mr = max(x[0] for x in train_layouts)
        mc = max(x[1] for x in train_layouts)
        axes['grow_both'] += any(r > mr and c > mc for r, c in test_layouts)
        axes['grow_rows'] += any(r > mr and c <= mc for r, c in test_layouts)
        axes['grow_columns'] += any(c > mc and r <= mr for r, c in test_layouts)
        axes['familiar'] += any(x in train_layouts for x in test_layouts)
        axes['smaller'] += any(r < mr and c < mc for r, c in test_layouts)
    _require(len(geometries) >= 12 and len(layouts) >= 35, 'geometry or layout diversity')
    for name in axes:
        _require(axes[name] > 0, f'missing review variation: {name}')
    _require(len({len(p['train']) for p in puzzles}) >= 4, 'train-count diversity')
    _require(len({len(p['test']) for p in puzzles}) >= 3, 'test-count diversity')
    _require(len({tuple(_grid_key(x['input']) for x in p['train']) for p in puzzles}) == 42, 'cross-seed difference')
    _require(len({x['input'].shape for p in puzzles for split in ('train', 'test') for x in p[split]}) >= 15, 'shape diversity')
    _require(any(x['input'].shape[0] >= 25 or x['input'].shape[1] >= 25 for p in puzzles for split in ('train', 'test') for x in p[split]), 'large grids')
    _require(any(x['input'].shape[0] <= 6 and x['input'].shape[1] <= 6 for p in puzzles for split in ('train', 'test') for x in p[split]), 'single small panels')
    _require(any(len({int(y) for y in x['input'].ravel()}) == 2 and int(x['input'][0,0]) == 0 and 8 in x['input'] for p in puzzles for split in ('train','test') for x in p[split]), 'original palette available')
    _require(_puzzles_equal(generate_puzzle(314159), generate_puzzle(314159)), 'automatic determinism')
    _require(_puzzles_equal(generate_puzzle(271828, 6, 3), generate_puzzle(271828, 6, 3)), 'explicit determinism')
    for seed, tr, te in ((1001,2,1),(1002,10,1),(1003,10,4),(2001,None,1),(2002,None,4),(2003,2,None),(2004,10,None)):
        p = generate_puzzle(seed, tr, te)
        _require((tr is None or len(p['train']) == tr) and (te is None or len(p['test']) == te), 'requested counts')
    for request in ({'num_train':1},{'num_train':11},{'num_test':0},{'num_test':5},{'num_train':2,'num_test':2},{'num_train':2,'num_test':3},{'num_train':True},{'num_train':3.5}):
        try:
            generate_puzzle(9999, **request)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
