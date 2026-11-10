import numpy as np


def _text_grid_key(text):
    rows = text.split("/")
    return (
        (len(rows), len(rows[0])),
        bytes(int(ch) for row in rows for ch in row),
    )


def _grid_key(grid):
    array = np.asarray(grid)
    return (
        tuple(array.shape),
        bytes(np.asarray(array, dtype=np.uint8).ravel().tolist()),
    )


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


_SOURCE_PAIR_TEXTS = (
    (
        "11113333/11110000/11110000/11110000/"
        "55556060/55550606/55556060/55550606",
        "6565/5656/6565/5656/6565/5656/6565/5656/"
        "6565/5656/6565/5656/6565/5656/6565/5656",
    ),
    (
        "88886666/88886666/88886666/88886666/"
        "11114444/55554004/11114004/55554444",
        "4444444444444444/"
        "4114411441144114/"
        "4114411441144114/"
        "4444444444444444/"
        "4444444444444444/"
        "4554455445544554/"
        "4554455445544554/"
        "4444444444444444/"
        "4444444444444444/"
        "4114411441144114/"
        "4114411441144114/"
        "4444444444444444/"
        "4444444444444444/"
        "4554455445544554/"
        "4554455445544554/"
        "4444444444444444",
    ),
    (
        "6600000000044444/"
        "6600000000044444/"
        "6600000000000000/"
        "6600000000000000/"
        "6600000000000000/"
        "0000000000000000/"
        "0000000000000000/"
        "0000000000000000/"
        "1111111180800606/"
        "1111111108000060/"
        "1111111180800606/"
        "1111111100000000/"
        "1111111100000000/"
        "1111111160600808/"
        "1111111106000080/"
        "1111111160600808",
        "8181161681/"
        "1811116118/"
        "8181161681/"
        "1111111111/"
        "1111111111/"
        "6161181861/"
        "1611118116/"
        "6161181861/"
        "8181161681/"
        "1811116118",
    ),
    (
        "44444444/00000000/00000000/00000000/"
        "25250000/25250000/25250000/25250000",
        "2525/2525/2525/2525",
    ),
    (
        "80804040/00000000/00804004/00000000/"
        "66661111/66660000/66661111/66660000",
        "1111/6666/1111",
    ),
    (
        "20051010/03050131/55551010/00520030/"
        "22220333/22223230/22223005/22223310",
        "23332333/32323232/32253225/33123312/"
        "23332333/32323232/32253225/33123312/"
        "23332333/32323232",
    ),
    (
        "2255/2255/7370/6807",
        "7773/7737/7678/6787",
    ),
)

_SOURCE_PAIR_KEYS = frozenset(
    (_text_grid_key(input_text), _text_grid_key(output_text))
    for input_text, output_text in _SOURCE_PAIR_TEXTS
)


def _validate_optional_count(value, name, low, high):
    if value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(rng, num_train, num_test):
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)

    train = None if num_train is None else int(num_train)
    test = None if num_test is None else int(num_test)

    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must be greater than num_test")
        return train, test

    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
        return train, test

    if train is None:
        low = max(2, test + 1)
        if low > 10:
            raise ValueError("no legal num_train is greater than num_test")
        return int(rng.integers(low, 11)), test

    high = min(4, train - 1)
    if high < 1:
        raise ValueError("no legal num_test is less than num_train")
    return train, int(rng.integers(1, high + 1))


def _sample_puzzle_parameters(rng):
    # Only transparency is a learned color convention. Complexity and size
    # support change sampling weights, never the interpretation of a panel.
    background = 0 if rng.random() < .25 else int(rng.integers(10))
    complexity = int(rng.integers(3))
    palette_size = int(rng.integers(3 + complexity, 5 + 2 * complexity))
    palette = rng.permutation([c for c in range(10) if c != background])[:palette_size]
    sizes = np.array([2, 3, 4, 5, 6, 8] if complexity == 0 else range(2, 16))
    train_sizes = rng.choice(sizes, size=min(len(sizes), 3 + complexity), replace=False)
    return dict(background=background, palette=palette, complexity=complexity,
                sizes=sizes, train_sizes=train_sizes,
                test_regime=int(rng.integers(3)))


def _orient(array, rng):
    array = np.rot90(array, int(rng.integers(4)))
    if rng.random() < .5:
        array = array[:, ::-1]
    return array.copy()


def _inks(rng, palette, maximum, mono=.4):
    k = 1 if maximum == 1 or rng.random() < mono else int(rng.integers(2, maximum + 1))
    return rng.choice(palette, size=k, replace=False)


def _color_positions(rng, inks, count):
    # Every selected ink appears, but frequencies need not be balanced.
    rest = rng.choice(inks, size=count - len(inks),
                      p=rng.dirichlet(np.full(len(inks), .7)))
    return rng.permutation(np.concatenate((inks, rest)))


def _grow(n, count, rng):
    start = (int(rng.integers(n)), int(rng.integers(n)))
    points, frontier = set(), {start}
    while len(points) < count:
        choices = sorted(frontier)
        p = choices[int(rng.integers(len(choices)))]
        frontier.remove(p)
        points.add(p)
        r, c = p
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= q[0] < n and 0 <= q[1] < n and q not in points:
                frontier.add(q)
    mask = np.zeros((n, n), dtype=bool)
    for p in points:
        mask[p] = True
    return mask


def _counter_panel(n, count, palette, background, rng):
    style = int(rng.integers(4))
    mask = np.zeros((n, n), dtype=bool)
    if style == 0:  # Compact rectangles, or a rectangle with a partial edge.
        factors = [(h, count // h) for h in range(1, n+1)
                   if count % h == 0 and count // h <= n]
        if factors:
            h, w = factors[int(rng.integers(len(factors)))]
        else:
            w = int(rng.integers((count+n-1)//n, n+1))
            h = (count+w-1)//w
        r, c = int(rng.integers(n-h+1)), int(rng.integers(n-w+1))
        for i in range(count):
            mask[r+i//w, c+i%w] = True
    elif style == 1:  # Full rows/columns, with an independently placed remainder.
        rows, cols = rng.permutation(n), rng.permutation(n)
        mask[rows[np.arange(count)//n], cols[np.arange(count)%n]] = True
    elif style == 2:
        mask = _grow(n, count, rng)
    else:
        mask.ravel()[rng.choice(n*n, count, replace=False)] = True
    mask = _orient(mask, rng)
    inks = _inks(rng, palette, min(4, len(palette), count), mono=.48)
    panel = np.full((n, n), background, dtype=np.int64)
    panel[mask] = _color_positions(rng, inks, count)
    return panel


def _carrier(n, params, rng):
    palette, level = params['palette'], params['complexity']
    kinds = ['solid', 'stripes', 'runs', 'checker', 'blocks', 'motif', 'mosaic', 'diagonal']
    weights = ([.23, .20, .18, .15, .12, .06, .03, .03] if level == 0 else
               [.10, .12, .15, .12, .15, .14, .17, .05])
    kind = str(rng.choice(kinds, p=weights))
    maximum = min(len(palette), 6, n if kind in ('stripes', 'runs') else n*n)
    inks = _inks(rng, palette, maximum, mono=.08)
    k = len(inks)
    rr, cc = np.indices((n, n))
    if kind == 'solid' or k == 1:
        labels = np.zeros((n, n), dtype=int)
    elif kind in ('stripes', 'runs'):
        if kind == 'stripes':
            line = (np.arange(n) + int(rng.integers(k))) % k
        else:
            cuts = sorted(rng.choice(np.arange(1, n), k-1, replace=False))
            line = np.searchsorted(cuts, np.arange(n), side='right')
            line = rng.permutation(k)[line]
        labels = np.tile(line, (n, 1))
    elif kind in ('checker', 'diagonal'):
        a, b = (int(rng.integers(1, max(2, n//2+1))) for _ in range(2))
        labels = ((rr//a + cc//b) if kind == 'checker' else
                  rr + int(rng.integers(1, k+1))*cc) % k
    elif kind == 'blocks':
        cuts_r = sorted(rng.choice(np.arange(1, n), int(rng.integers(1, n)), replace=False))
        cuts_c = sorted(rng.choice(np.arange(1, n), int(rng.integers(1, n)), replace=False))
        block_colors = rng.integers(k, size=(len(cuts_r)+1, len(cuts_c)+1))
        labels = block_colors[np.searchsorted(cuts_r, rr, side='right'),
                              np.searchsorted(cuts_c, cc, side='right')]
    elif kind == 'motif':
        labels = np.zeros((n, n), dtype=int)
        for color in range(1, k):
            r, c = int(rng.integers(n)), int(rng.integers(n))
            h, w = int(rng.integers(1, n)), int(rng.integers(1, n))
            if rng.random() < .5:
                mask = (abs(rr-r)/h + abs(cc-c)/w) <= 1
            else:
                mask = (abs(rr-r) < h) & (abs(cc-c) < w)
            labels[mask] = color
    else:
        if rng.random() < .5:
            labels = rng.integers(k, size=(n, n))
        else:
            centers = rng.choice(n*n, size=min(n*n, int(rng.integers(k, 2*k+1))), replace=False)
            distances = abs(rr[..., None] - centers//n) + abs(cc[..., None] - centers % n)
            labels = rng.integers(k, size=len(centers))[np.argmin(distances, axis=2)]
    return _orient(inks[labels].astype(np.int64), rng)


def _stencil(n, params, rng):
    background, palette = params['background'], params['palette']
    stencil = np.full((n, n), background, dtype=np.int64)
    if rng.random() < .20:
        return stencil
    density = float(rng.uniform(.04, .94))
    count = max(1, min(n*n-1, round(density*n*n)))
    rr, cc = np.indices((n, n))
    kind = str(rng.choice(['scatter', 'blob', 'lines', 'frame', 'checker', 'motifs', 'holes']))
    mask = np.zeros((n, n), dtype=bool)
    if kind == 'scatter':
        mask.ravel()[rng.choice(n*n, count, replace=False)] = True
    elif kind == 'blob':
        mask = _grow(n, count, rng)
    elif kind == 'lines':
        period = int(rng.integers(2, n+1))
        thickness = int(rng.integers(1, period))
        mask = (rr + int(rng.integers(period))) % period < thickness
    elif kind == 'frame':
        thickness = int(rng.integers(1, max(2, (n+1)//2)))
        mask = np.minimum.reduce([rr, cc, n-1-rr, n-1-cc]) < thickness
        if rng.random() < .35:
            mask &= (rr >= int(rng.integers(max(1, n//2))))
    elif kind == 'checker':
        a, b = int(rng.integers(1, max(2, n//2+1))), int(rng.integers(1, max(2, n//2+1)))
        mask = (rr//a + cc//b + int(rng.integers(2))) % 2 == 0
    elif kind == 'motifs':
        for _ in range(int(rng.integers(1, max(2, n)))):
            r, c = int(rng.integers(n)), int(rng.integers(n))
            mask |= abs(rr-r) + abs(cc-c) <= int(rng.integers(0, max(2, n//3)))
    else:
        mask[:] = True
        mask[_grow(n, n*n-count, rng)] = False
        if rng.random() < .5:
            mask &= rng.random((n, n)) > .12
    # Correct degenerate masks at randomly chosen cells, never fixed corners.
    if not mask.any():
        mask.ravel()[int(rng.integers(n*n))] = True
    if mask.all():
        mask.ravel()[int(rng.integers(n*n))] = False
    mask = _orient(mask, rng)
    inks = _inks(rng, palette, min(5, len(palette), int(mask.sum())), mono=.46)
    stencil[mask] = _color_positions(rng, inks, int(mask.sum()))
    return stencil


def _varies_vertically(carrier):
    return not np.all(carrier == carrier[:1, :])


def _varies_horizontally(carrier):
    return not np.all(carrier == carrier[:, :1])


def _dimension(n, varies, rng):
    cap = min(n*n, 30)
    choices = np.arange(n, cap+1, n) if varies else np.arange(1, cap+1)
    if rng.random() < .35:
        return int(rng.choice([choices[0], n, choices[-1]]))
    return int(rng.choice(choices))


def _resize_and_overlay(carrier, stencil, height, width, background=0):
    n = carrier.shape[0]
    base = carrier[(np.arange(height)*n//height)[:, None],
                   (np.arange(width)*n//width)[None, :]]
    tiled = stencil[np.arange(height)[:, None] % n, np.arange(width)[None, :] % n]
    return np.where(tiled == background, base, tiled).astype(np.int64)


def _assemble_pair(left, right, carrier, stencil, background):
    height, width = int(np.sum(left != background)), int(np.sum(right != background))
    return dict(input=np.block([[left, right], [carrier, stencil]]),
                output=_resize_and_overlay(carrier, stencil, height, width, background))


def _sample_pair(rng, params, regime, train_sizes=()):
    sizes = params['train_sizes']
    if regime == 'target_test':
        if params['test_regime'] == 0:
            sizes = sorted(train_sizes)
        elif params['test_regime'] == 1:
            sizes = [n for n in params['sizes'] if n not in train_sizes]
            if not len(sizes):
                sizes = params['sizes']
        else:
            sizes = params['sizes']
    n = int(rng.choice(sizes))
    carrier = _carrier(n, params, rng)
    stencil = _stencil(n, params, rng)
    height = _dimension(n, _varies_vertically(carrier), rng)
    width = _dimension(n, _varies_horizontally(carrier), rng)
    if rng.random() < .17 and (not _varies_horizontally(carrier) or height % n == 0):
        width = height
    left = _counter_panel(n, height, params['palette'], params['background'], rng)
    right = _counter_panel(n, width, params['palette'], params['background'], rng)
    return _assemble_pair(left, right, carrier, stencil, params['background'])


def _decode(grid, background):
    assert grid.ndim == 2 and grid.shape[0] == grid.shape[1] and grid.shape[0] % 2 == 0
    n = grid.shape[0]//2
    assert 2 <= n <= 15
    left, right = grid[:n, :n], grid[:n, n:]
    carrier, stencil = grid[n:, :n], grid[n:, n:]
    h, w = int(np.sum(left != background)), int(np.sum(right != background))
    assert 1 <= h <= min(n*n, 30) and 1 <= w <= min(n*n, 30)
    assert np.all(carrier != background)
    assert not _varies_vertically(carrier) or (h >= n and h % n == 0)
    assert not _varies_horizontally(carrier) or (w >= n and w % n == 0)
    return n, h, w, left, right, carrier, stencil


def _predict(grid, background):
    # A second implementation reads the actual counters and expands constant
    # runs; it neither sees generation parameters nor calls the constructor.
    n, h, w, _, _, carrier, stencil = _decode(grid, background)
    if _varies_vertically(carrier):
        base = np.repeat(carrier, h//n, axis=0)
    else:
        base = np.repeat(carrier[:1], h, axis=0)
    if _varies_horizontally(carrier):
        base = np.repeat(base, w//n, axis=1)
    else:
        base = np.repeat(base[:, :1], w, axis=1)
    for r in range(h):
        for c in range(w):
            ink = stencil[r % n, c % n]
            if ink != background:
                base[r, c] = ink
    return base


def _background_candidates(train_pairs):
    candidates = []
    for color in range(10):
        matches = True
        for pair in train_pairs:
            try:
                n, h, w, _, _, _, _ = _decode(pair['input'], color)
                matches = pair['output'].shape == (h, w) and np.array_equal(
                    _predict(pair['input'], color), pair['output'])
            except AssertionError:
                matches = False
            if not matches:
                break
        if matches:
            candidates.append(color)
    return candidates


def solve(train_pairs, test_input):
    candidates = _background_candidates(train_pairs)
    assert len(candidates) == 1, 'training must uniquely reveal transparency color'
    return _predict(test_input, candidates[0])


_REQUIRED_EVIDENCE = frozenset((
    'counter_swap', 'counter_bbox', 'counter_one_ink', 'counter_color_count',
    'carrier_tile_rows', 'carrier_tile_columns', 'stencil_resize',
    'overlay_before_resize', 'opaque_background', 'ignore_stencil',
    'phase_rows', 'phase_columns', 'swap_lower_panels',
    'visible_carrier', 'visible_overwrite',
))


def _evidence(pair, background):
    n, h, w, left, right, carrier, stencil = _decode(pair['input'], background)
    out = pair['output']
    rr, cc = np.arange(h)[:, None], np.arange(w)[None, :]
    sr, sc = rr*n//h, cc*n//w
    base, tiled = carrier[sr, sc], stencil[rr % n, cc % n]
    cover = set()
    def compare(name, alternative):
        if not np.array_equal(out, alternative):
            cover.add(name)
    if h != w:
        cover.add('counter_swap')
    bounds, maxima, color_counts = [], [], []
    for counter in (left, right):
        r, c = np.nonzero(counter != background)
        bounds.append(int((r.max()-r.min()+1)*(c.max()-c.min()+1)))
        _, frequencies = np.unique(counter[counter != background], return_counts=True)
        maxima.append(int(frequencies.max()))
        color_counts.append(len(frequencies))
    for name, dimensions in (('counter_bbox', bounds), ('counter_one_ink', maxima),
                             ('counter_color_count', color_counts)):
        if tuple(dimensions) != (h, w):
            cover.add(name)
    compare('carrier_tile_rows', np.where(tiled == background, carrier[rr % n, sc], tiled))
    compare('carrier_tile_columns', np.where(tiled == background, carrier[sr, cc % n], tiled))
    scaled_stencil = stencil[sr, sc]
    compare('stencil_resize', np.where(scaled_stencil == background, base, scaled_stencil))
    fused = np.where(stencil == background, carrier, stencil)
    compare('overlay_before_resize', fused[sr, sc])
    compare('opaque_background', tiled)
    compare('ignore_stencil', base)
    for name, shifted in (('phase_rows', stencil[(rr+1) % n, cc % n]),
                          ('phase_columns', stencil[rr % n, (cc+1) % n])):
        compare(name, np.where(shifted == background, base, shifted))
    compare('swap_lower_panels', carrier[rr % n, cc % n])
    if np.any(tiled == background):
        cover.add('visible_carrier')
    if np.any((tiled != background) & (tiled != base)):
        cover.add('visible_overwrite')
    return cover


def _normalized(grid):
    palette, result = {}, []
    for cell in grid.flat:
        value = int(cell)
        if value not in palette:
            palette[value] = len(palette)
        result.append(palette[value])
    return (grid.shape, bytes(result))


def _training_diverse(pairs, background):
    decoded = [_decode(p['input'], background) for p in pairs]
    return (len({a[0] for a in decoded}) >= 2
            and len({(a[1], a[2]) for a in decoded}) >= 2
            and len({_normalized(a[5]) for a in decoded}) >= 2
            and len({_normalized(a[6]) for a in decoded}) >= 2
            and len({_grid_key(a[3] != background) for a in decoded}) >= 2
            and len({_counter_layout(counter != background) for a in decoded
                     for counter in a[3:5]}) >= 2)


def _fallback_training(params):
    # Only used if a bounded random candidate pool lacks a teaching pair.
    # These are constructive witnesses, not a normal pair-index schedule.
    bg = params['background']
    a, b, c = (int(x) for x in params['palette'][:3])
    carrier1 = np.array([[a, b, a], [c, a, b], [b, c, c]])
    left1 = np.array([[a, bg, b], [bg, c, a], [b, bg, a]])
    right1 = np.array([[bg, a, bg], [b, bg, bg], [bg, bg, c]])
    first = _assemble_pair(left1, right1, carrier1, np.full((3, 3), bg), bg)
    carrier2 = np.array([[b, b, a, c], [a, c, c, b], [c, a, b, a], [a, b, c, c]])
    stencil = np.full((4, 4), bg)
    stencil[0, 2], stencil[1, 0], stencil[2, 3], stencil[3, 1] = b, a, b, c
    left2 = np.array([[a, bg, b, bg], [bg, c, bg, a],
                      [b, bg, a, bg], [bg, c, bg, b]])
    right2 = np.array([[a, b, bg, c], [b, bg, c, a],
                       [bg, c, b, a], [a, c, a, bg]])
    return [first, _assemble_pair(left2, right2, carrier2, stencil, bg)]


def _usable(pair, seen):
    return (_grid_key(pair['input']) not in seen and _pair_key(pair) not in _SOURCE_PAIR_KEYS
            and not np.array_equal(pair['input'], pair['output']))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    params = _sample_puzzle_parameters(rng)
    bg = params['background']
    candidates, covers, seen = [], [], set()
    train = None
    # Select for observable evidence, then shuffle. No index fixes a picture,
    # scale, mask, ink count, or split-specific counter style.
    for _ in range(64):
        pair = _sample_pair(rng, params, 'source_train')
        if not _usable(pair, seen):
            continue
        cover = _evidence(pair, bg)
        for j in rng.permutation(len(candidates)):
            proposed = [candidates[int(j)], pair]
            if (_REQUIRED_EVIDENCE <= (covers[int(j)] | cover)
                    and _training_diverse(proposed, bg)
                    and _background_candidates(proposed) == [bg]):
                train = proposed
                break
        candidates.append(pair)
        covers.append(cover)
        seen.add(_grid_key(pair['input']))
        if train is not None:
            break
    if train is None:
        train = _fallback_training(params)
    seen = {_grid_key(p['input']) for p in train}
    puzzle = {'train': train, 'test': []}
    train_sizes = {p['input'].shape[0]//2 for p in train}
    for split, regime, count in (('train', 'source_train', train_count),
                                 ('test', 'target_test', test_count)):
        while len(puzzle[split]) < count:
            for attempt in range(40):
                pair = _sample_pair(rng, params, regime, train_sizes)
                if _usable(pair, seen):
                    break
            else:
                # There are vastly more than 14 distinct counter colorings.
                # A finite enumeration guarantees a unique fallback input.
                pair = _fallback_training(params)[0]
                original = pair['input'].copy()
                locations = np.flatnonzero(original[:3, :3] != bg)
                for serial in range(3**len(locations)):
                    trial = original.copy()
                    for digit, loc in enumerate(locations):
                        trial[int(loc)//3, int(loc)%3] = params['palette'][(serial//3**digit) % 3]
                    pair = {'input': trial, 'output': _predict(trial, bg)}
                    if _usable(pair, seen):
                        break
            puzzle[split].append(pair)
            seen.add(_grid_key(pair['input']))
            if split == 'train':
                train_sizes.add(pair['input'].shape[0]//2)
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert _usable(pair, seen), 'duplicate, source copy, or identity pair'
            seen.add(_grid_key(pair['input']))
    backgrounds = _background_candidates(puzzle['train'])
    assert len(backgrounds) == 1, 'ambiguous training background'
    bg = backgrounds[0]
    evidence = set()
    for split in puzzle:
        for pair in puzzle[split]:
            _decode(pair['input'], bg)
            assert np.array_equal(_predict(pair['input'], bg), pair['output']), 'incorrect output'
            if split == 'train':
                evidence |= _evidence(pair, bg)
            else:
                # In particular, background is never inferred from test labels.
                assert np.array_equal(solve(puzzle['train'], pair['input']), pair['output'])
    assert _REQUIRED_EVIDENCE <= evidence, 'training does not distinguish ' + str(_REQUIRED_EVIDENCE-evidence)
    assert _training_diverse(puzzle['train'], bg), 'insufficient within-puzzle variation'


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    sizes = []
    while remaining:
        todo, size = [remaining.pop()], 0
        while todo:
            r, c = todo.pop()
            size += 1
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        sizes.append(size)
    return tuple(sorted(sizes))


def _counter_layout(mask):
    r, c = np.nonzero(mask)
    bbox = (r.max()-r.min()+1)*(c.max()-c.min()+1)
    if bbox == mask.sum():
        return 'rectangle'
    if np.any(mask.all(axis=0)) or np.any(mask.all(axis=1)):
        return 'rows_or_columns'
    return 'connected_irregular' if len(_components(mask)) == 1 else 'scattered'


def _features(pair, background):
    # Measurements are reconstructed from returned arrays, not sampler labels.
    n, h, w, left, right, carrier, stencil = _decode(pair['input'], background)
    mask = stencil != background
    ci = len(np.unique(carrier))
    si = len(np.unique(stencil[mask]))
    vertical, horizontal = _varies_vertically(carrier), _varies_horizontally(carrier)
    axes = ('both' if vertical and horizontal else 'rows' if vertical else
            'columns' if horizontal else 'solid')
    counts, layouts, unequal, counter_colors = [], [], [], set()
    for counter in (left, right):
        colors, freq = np.unique(counter[counter != background], return_counts=True)
        counter_colors.update(int(x) for x in colors)
        counts.append(len(colors))
        unequal.append(bool(len(freq) > 1 and np.ptp(freq) > 0))
        layouts.append(_counter_layout(counter != background))
    edges = int(np.sum(carrier[1:] != carrier[:-1]) + np.sum(carrier[:, 1:] != carrier[:, :-1]))
    overlap = bool(set(np.unique(carrier)) & set(np.unique(stencil[mask])))
    runs = []
    if axes in ('rows', 'columns'):
        line = carrier[:, 0] if axes == 'rows' else carrier[0]
        cuts = np.r_[0, np.flatnonzero(line[1:] != line[:-1])+1, n]
        runs = np.diff(cuts).tolist()
    rr, cc = np.indices((n, n))
    distance = np.minimum.reduce([rr, cc, n-rr-1, n-cc-1])
    frame = next((k for k in range(1, (n+1)//2)
                  if np.array_equal(mask, distance < k)), 0)
    return dict(n=n, shape=(h, w), background=background, carrier_inks=ci,
                carrier_axes=axes, carrier_structure=_normalized(carrier),
                carrier_edge_density=round(edges/(2*n*(n-1)), 2),
                stencil_inks=si, stencil_structure=_normalized(stencil),
                mask_structure=_grid_key(mask), density=round(float(mask.mean()), 4),
                stencil_components=_components(mask), frame_thickness=frame,
                stencil_lines=si > 0 and (np.all(mask == mask[:1]) or np.all(mask == mask[:, :1])),
                checker_mask=np.array_equal(mask, (rr+cc) % 2 == 0) or
                np.array_equal(mask, (rr+cc) % 2 == 1),
                same_ink_diagonal=n == 2 and si == 1 and (
                    np.array_equal(mask, np.eye(2, dtype=bool)) or
                    np.array_equal(mask, np.fliplr(np.eye(2, dtype=bool)))),
                unequal_stripe_runs=len(set(runs)) > 1,
                two_color_stripes=bool(runs) and ci == 2,
                corners=tuple(int(x) for x in mask[[0, 0, -1, -1], [0, -1, 0, -1]]),
                counter_inks=tuple(counts), counter_layouts=tuple(layouts),
                counter_unequal=any(unequal), equal_count_different_layout=h == w and
                not np.array_equal(left != background, right != background),
                overlap=overlap, counter_reuse=bool(counter_colors & set(np.unique(np.vstack((carrier, stencil))))),
                black_payload=background != 0 and bool(np.any(carrier == 0) or np.any(stencil == 0)),
                scale=(h/n if vertical else 'uniform', w/n if horizontal else 'uniform'),
                partial_tile=h % n != 0 or w % n != 0,
                reduced=h < n or w < n,
                combination=(axes, ci, si, overlap, h != w, tuple(counts)))


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        _pair_key(a) == _pair_key(b) for a, b in zip(first[s], second[s])) for s in ('train', 'test'))


def self_test() -> None:
    source = [dict(input=np.array([[int(c) for c in row] for row in a.split('/')]),
                   output=np.array([[int(c) for c in row] for row in b.split('/')]))
              for a, b in _SOURCE_PAIR_TEXTS]
    for pair in source:
        assert np.array_equal(_predict(pair['input'], 0), pair['output'])
    for pair in source[5:]:
        assert np.array_equal(solve(source[:5], pair['input']), pair['output'])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_pair_key(p['test'][0]) for p in puzzles}) == 42
    assert _puzzles_equal(generate_puzzle(123456), generate_puzzle(123456))
    assert _puzzles_equal(generate_puzzle(98765, 7, 3), generate_puzzle(98765, 7, 3))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    features = {'train': [], 'test': []}
    first_answers = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        bg = _background_candidates(puzzle['train'])[0]
        first_answers.add(_normalized(puzzle['test'][0]['output']))
        for split in features:
            features[split].extend(_features(p, bg) for p in puzzle[split])
    all_features = features['train'] + features['test']
    assert len({f['background'] for f in all_features}) >= 8
    assert len(first_answers) >= 38
    assert any(f['black_payload'] for f in all_features)
    assert any(f['frame_thickness'] >= 2 for f in all_features)
    assert len({f['carrier_edge_density'] for f in all_features}) >= 35
    assert len({f['corners'] for f in all_features}) == 16
    for split, fs in features.items():
        assert len({f['n'] for f in fs}) >= 10
        assert len({f['shape'] for f in fs}) >= 35
        assert len({f['carrier_structure'] for f in fs}) >= 40
        assert len({f['stencil_structure'] for f in fs}) >= 40
        assert len({f['mask_structure'] for f in fs}) >= 35
        assert {0, 1, 4} <= {f['stencil_inks'] for f in fs}
        assert {1, 2, 3, 4} <= {f['carrier_inks'] for f in fs}
        assert {'solid', 'rows', 'columns', 'both'} <= {f['carrier_axes'] for f in fs}
        assert len({f['corners'] for f in fs}) >= 10
        assert any(f['counter_inks'] == (1, 1) for f in fs)
        assert any(min(f['counter_inks']) > 1 for f in fs)
        assert any(f['counter_inks'][0] != f['counter_inks'][1] for f in fs)
        assert any(f['counter_unequal'] for f in fs)
        assert any(f['equal_count_different_layout'] for f in fs)
        assert any(f['two_color_stripes'] for f in fs)
        assert any(f['unequal_stripe_runs'] for f in fs)
        assert any(f['checker_mask'] and f['stencil_inks'] == 1 for f in fs)
        assert any(f['stencil_lines'] for f in fs)
        assert any(1 in f['stencil_components'] for f in fs)
        assert any(len(f['stencil_components']) > 1 for f in fs)
        for mono in (True, False):
            assert len({int(f['density']*5) for f in fs if f['stencil_inks'] > 0
                        and (f['stencil_inks'] == 1) == mono}) >= 4
        assert {'rectangle', 'rows_or_columns', 'connected_irregular', 'scattered'} <= {
            layout for f in fs for layout in f['counter_layouts']}
        assert any(f['partial_tile'] for f in fs) and any(f['reduced'] for f in fs)
        assert any(1 in f['shape'] for f in fs)
        assert any(f['shape'] == (30, 30) for f in fs)
        assert len({f['combination'] for f in fs}) >= 40
        for axes in ('solid', 'rows', 'columns', 'both'):
            assert {True, False} == {f['stencil_inks'] == 0 for f in fs if f['carrier_axes'] == axes}
    for offset, request in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                      (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000+offset, *request)
        validate_generated_puzzle(p)
        for split, count in zip(('train', 'test'), request):
            assert count is None or len(p[split]) == count
    # Exercise the hardest teaching boundary across many different palettes.
    for seed in range(20):
        validate_generated_puzzle(generate_puzzle(3000+seed, 2, 1))
    for seed in range(10):
        rng = np.random.default_rng(5000+seed)
        params = _sample_puzzle_parameters(rng)
        fallback = _fallback_training(params)
        assert _background_candidates(fallback) == [params['background']]
        assert _training_diverse(fallback, params['background'])
        assert _REQUIRED_EVIDENCE <= set().union(*(
            _evidence(p, params['background']) for p in fallback))
    rng = np.random.default_rng(71234)
    asymmetric = np.arange(9).reshape(3, 3)
    assert len({_grid_key(_orient(asymmetric, rng)) for _ in range(64)}) == 8
    # Small diagonal masks are rare in a 42-puzzle mixture of fourteen sizes;
    # probe the actual small-panel sampler as well as measuring the mixture.
    params = _sample_puzzle_parameters(rng)
    small_masks = [_stencil(2, params, rng) for _ in range(100)]
    for diagonal in (np.eye(2, dtype=bool), np.fliplr(np.eye(2, dtype=bool))):
        assert any(np.array_equal(s != params['background'], diagonal)
                   and len(np.unique(s[diagonal])) == 1 for s in small_masks)
    for request in ((1, None), (11, None), (None, 0), (None, 5), (2, 2),
                    (4, 4), (3, 4), (2, 4), (True, 1), (3, 1.5)):
        try:
            generate_puzzle(0, *request)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # The validator must reject a corrupted label, independently of metadata.
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
