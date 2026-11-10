"""Color-matched subtractive stencils, with visible, full-span registers.

Only the local rule is shared across a puzzle. Background, scale and array
geometry are observable pair properties. Layout regimes control the curriculum,
never the interpretation of an input.
"""
import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    '9a6e383e6d6966dc809e59d08b530faedd28077cacec07209646e294faee0f96',
    'e01c2262d68853cf47468a1d2428eb64a5c69406fe4a139cd4991ae3d7667238',
    '41b72ae09735f485ca1ae969128d14dbbd8f3233fa7f3855c50420cad07d82ce',
    '915d12e9a5ad0eeb5c0b28deac2e7cbba466c37704f0b16bfd9f5c65b82a44d4',
}


def _checked_count(
    name: str,
    value: int,
    low: int,
    high: int,
) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = (
        None
        if num_train is None
        else _checked_count("num_train", num_train, 2, 10)
    )
    test_count = (
        None
        if num_test is None
        else _checked_count("num_test", num_test, 1, 4)
    )

    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif train_count is None:
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, maximum_test + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _transforms(mask):
    return [np.rot90(mask, i) for i in range(4)] + [
        np.rot90(np.fliplr(mask), i) for i in range(4)]


def _full_span(mask):
    return all((mask[0].any(), mask[-1].any(),
                mask[:, 0].any(), mask[:, -1].any()))


def _mask(rng, k, asymmetric=False):
    """Compositional motifs, symmetric masks, sparse and dense irregular ink."""
    for _ in range(100):
        family = int(rng.integers(10))
        m = np.zeros((k, k), dtype=bool)
        if family == 0:  # diagonal, sometimes an X
            np.fill_diagonal(m, True)
            if rng.random() < .5:
                m |= np.fliplr(m)
        elif family == 1:  # a U or T, in any displayed orientation
            m[0] = True
            if rng.random() < .5:
                m[:, 0] = m[:, -1] = True
            else:
                m[:, int(rng.integers(k))] = True
        elif family == 2:  # crossing bars
            m[int(rng.integers(k)), :] = True
            m[:, int(rng.integers(k))] = True
        elif family == 3:  # parallel bars spanning both axes
            m[0] = m[-1] = True
        elif family == 4:  # near full, including eight of nine
            m[:] = True
            m.flat[int(rng.integers(k*k))] = False
        elif family == 5:  # sparse opposite corners, with optional extra ink
            m[0, 0] = m[-1, -1] = True
            if rng.random() < .5:
                m[int(rng.integers(k)), int(rng.integers(k))] = True
        else:
            m = rng.random((k, k)) < float(rng.uniform(.18, .86))
            if family == 6:
                m |= np.fliplr(m)
            elif family == 7:
                m |= m.T
        m = _pick(rng, _transforms(m)).copy()
        if not (0 < m.sum() < k*k and _full_span(m)):
            continue
        if asymmetric and (len({x.tobytes() for x in _transforms(m)}) != 8
                           or 2*m.sum() == m.size):
            continue
        return m
    # Finite, full-span fallback; for k>=3 this has no D4 symmetry.
    m = np.zeros((k, k), dtype=bool)
    m[0, 0] = m[0, -1] = m[-1, 0] = True
    if k >= 3:
        m[1, -1] = True
    return m


def _bank_options(rows, cols, size, n):
    h, w, k = rows*(size+1)+1, cols*(size+1)+1, size-2
    options = []
    for nc in range(1, n+1):
        nr = (n+nc-1)//nc
        bh, bw = nr*(k+1)-1, nc*(k+1)-1
        if max(h, bh+2) <= 30 and w+bw+2 <= 30:
            options.append(('side', nr, nc))
        if h+bh+2 <= 30 and max(w, bw+2) <= 30:
            options.append(('above', nr, nc))
    return options


def _configurations():
    return [(r, c, s) for r in range(1, 6) for c in range(1, 6)
            for s in range(4, 9) if r*c <= 9
            and _bank_options(r, c, s, r*c)]


_CONFIGS = _configurations()


def _bank_layout(rng, r, c, s, n):
    """Enumerate fitting banks, then add variable gaps and empty slots."""
    ah, aw, k = r*(s+1)+1, c*(s+1)+1, s-2
    choices = []
    for mode, nr, nc in _bank_options(r, c, s, n):
        for extra in range(2):
            for gy in range(1, 4):
                for gx in range(1, 4):
                    bh = (nr+extra)*k+(nr+extra-1)*gy
                    bw = nc*k+(nc-1)*gx
                    h, w = ((max(ah, bh+2), aw+bw+2) if mode == 'side'
                            else (ah+bh+2, max(aw, bw+2)))
                    if h <= 30 and w <= 30:
                        choices.append((mode, nr+extra, nc, gy, gx, bh, bw, h, w))
    mode, nr, nc, gy, gx, bh, bw, h, w = _pick(rng, choices)
    if mode == 'side':
        ay, ax = int(rng.integers(h-ah+1)), 0
        by, bx = int(rng.integers(1, h-bh)), aw+1
    else:
        ay, ax = 0, int(rng.integers(w-aw+1))
        by, bx = ah+1, int(rng.integers(1, w-bw))
    slots = [(by+i*(k+gy), bx+j*(k+gx)) for i in range(nr) for j in range(nc)]
    slots = [slots[int(i)] for i in rng.choice(len(slots), n, replace=False)]
    if rng.random() < .5:
        ay = h-ah-ay
        slots = [(h-k-y, x) for y, x in slots]
    if rng.random() < .5:
        ax = w-aw-ax
        slots = [(y, w-k-x) for y, x in slots]
    return h, w, ay, ax, slots


def _scatter_layout(rng, r, c, s, n):
    """Random separated registers on any side; bounded bank fallback."""
    ah, aw, k = r*(s+1)+1, c*(s+1)+1, s-2
    for _ in range(16):
        h = int(rng.integers(max(ah, 23), 31))
        w = int(rng.integers(max(aw, 23), 31))
        ay, ax = int(rng.integers(h-ah+1)), int(rng.integers(w-aw+1))
        occupied = np.zeros((h, w), dtype=bool)
        occupied[ay:ay+ah, ax:ax+aw] = True
        slots = []
        # All possible top-left coordinates are visited once, without rejection.
        candidates = [(y, x) for y in range(1, h-k) for x in range(1, w-k)]
        for index in rng.permutation(len(candidates)):
            y, x = candidates[int(index)]
            if occupied[y:y+k, x:x+k].any():
                continue
            slots.append((y, x))
            occupied[y-1:y+k+1, x-1:x+k+1] = True
            if len(slots) == n:
                return h, w, ay, ax, slots
    return _bank_layout(rng, r, c, s, n)


def _make_pair(rng, config, background, distractors, scatter, witness=None):
    r, c, s = config
    k, n = s-2, r*c+distractors
    colors = [int(x) for x in rng.choice(
        [x for x in range(10) if x != background], n, replace=False)]
    patterns = [_mask(rng, k) for _ in colors]
    if witness is not None:
        if k >= 3:
            patterns[0] = _mask(rng, k, asymmetric=True)
        else:
            patterns[0][:] = True
            patterns[0].flat[witness] = False
    # Duplication and one-cell near matches remain color-disambiguated.
    if n >= 2 and rng.random() < .35:
        target = int(rng.integers(1, n))
        base = patterns[int(rng.integers(target))].copy()
        if rng.random() < .55:
            patterns[target] = base
        else:
            valid = []
            for cell in range(k*k):
                changed = base.copy()
                changed.flat[cell] ^= True
                if _full_span(changed) and 0 < changed.sum() < k*k:
                    valid.append(changed)
            if valid:
                patterns[target] = _pick(rng, valid)
    if witness is not None and n >= 2 and all(
            np.array_equal(patterns[0], m) for m in patterns[1:]):
        # At least one visible alternative makes color selection consequential.
        alternatives = [m for m in _transforms(patterns[0])
                        if not np.array_equal(m, patterns[0])]
        patterns[-1] = _pick(rng, alternatives).copy()
    layout = _scatter_layout if scatter else _bank_layout
    h, w, ay, ax, slots = layout(rng, r, c, s, n)
    inp = np.full((h, w), background, dtype=np.int8)
    out = np.full((r*(s+1)+1, c*(s+1)+1), background, dtype=np.int8)
    for index, color in enumerate(colors[:r*c]):
        ty, tx = 1+(index//c)*(s+1), 1+(index % c)*(s+1)
        inp[ay+ty:ay+ty+s, ax+tx:ax+tx+s] = color
        out[ty:ty+s, tx:tx+s] = color
        out[ty+1:ty+s-1, tx+1:tx+s-1][patterns[index]] = background
    for slot, index in zip(slots, rng.permutation(n)):
        y, x = slot
        inp[y:y+k, x:x+k][patterns[int(index)]] = colors[int(index)]
    # Remove incidental blank space, retaining the mandatory border; then vary
    # extra whitespace independently. Source dimensions are explicitly allowed.
    yy, xx = np.nonzero(inp != background)
    inp = inp[max(0, yy.min()-1):min(h, yy.max()+2),
              max(0, xx.min()-1):min(w, xx.max()+2)]
    extra = []
    for dim in inp.shape:
        total = int(rng.integers(min(5, 30-dim)+1))
        if rng.random() < .18:
            total = 30-dim
        first = int(rng.integers(total+1))
        extra.append((first, total-first))
    inp = np.pad(inp, extra, constant_values=background)
    # Preserve a dominant background even for a single large dense tile.
    while np.count_nonzero(inp == background) <= max(
            np.count_nonzero(inp == color) for color in colors):
        axis = 0 if inp.shape[0] < 30 else 1
        assert inp.shape[axis] < 30
        padding = [(0, 0), (0, 0)]
        padding[axis] = (0, 1)
        inp = np.pad(inp, padding, constant_values=background)
    return {'input': inp, 'output': out}


def _components(grid, background):
    visited = np.zeros(grid.shape, dtype=bool)
    result = []
    for y, x in np.argwhere(grid != background):
        y, x = int(y), int(x)
        if visited[y, x]:
            continue
        color = int(grid[y, x])
        stack, cells = [(y, x)], []
        visited[y, x] = True
        while stack:
            yy, xx = stack.pop()
            cells.append((yy, xx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = yy+dy, xx+dx
                if (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]
                        and not visited[ny, nx] and grid[ny, nx] == color):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        result.append((color, np.array(cells)))
    return result


def _decode(inp):
    """Recognize only input geometry, without the output or generation state."""
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert np.count_nonzero(counts == counts.max()) == 1
    squares = []
    for color, cells in _components(inp, bg):
        y, x = cells.min(axis=0)
        bottom, right = cells.max(axis=0)+1
        if bottom-y == right-x and bottom-y >= 4 and len(cells) == (bottom-y)**2:
            squares.append((int(bottom-y), int(y), int(x), color))
    assert squares, 'no solid tile'
    s = max(square[0] for square in squares)
    tiles = sorted(square for square in squares if square[0] == s)
    ys, xs = sorted({t[1] for t in tiles}), sorted({t[2] for t in tiles})
    assert all(b-a == s+1 for coords in (ys, xs) for a, b in zip(coords, coords[1:]))
    assert {(t[1], t[2]) for t in tiles} == {(y, x) for y in ys for x in xs}
    tile_colors = [t[3] for t in tiles]
    assert len(set(tile_colors)) == len(tiles)
    y0, x0, y1, x1 = ys[0]-1, xs[0]-1, ys[-1]+s+1, xs[-1]+s+1
    assert y0 >= 0 and x0 >= 0 and y1 <= inp.shape[0] and x1 <= inp.shape[1]
    tile_area = np.zeros(inp.shape, dtype=bool)
    for _, y, x, color in tiles:
        tile_area[y:y+s, x:x+s] = True
    crop_area = np.zeros(inp.shape, dtype=bool)
    crop_area[y0:y1, x0:x1] = True
    assert np.all(inp[crop_area & ~tile_area] == bg)
    external = ~crop_area & (inp != bg)
    patterns, boxes = {}, {}
    k = s-2
    for color in sorted(int(x) for x in np.unique(inp[external])):
        cells = np.argwhere(external & (inp == color))
        top, left = (int(x) for x in cells.min(axis=0))
        bottom, right = (int(x) for x in cells.max(axis=0)+1)
        assert (bottom-top, right-left) == (k, k), 'unregistered stencil'
        assert not crop_area[top:bottom, left:right].any()
        register = inp[top:bottom, left:right]
        assert np.all((register == bg) | (register == color))
        mask = register == color
        assert _full_span(mask) and 0 < mask.sum() < k*k
        patterns[color] = mask
        boxes[color] = (top, left, bottom, right)
    assert set(tile_colors) <= set(patterns)
    for a, first in enumerate(list(boxes.values())):
        for second in list(boxes.values())[a+1:]:
            assert (first[2] < second[0] or second[2] < first[0]
                    or first[3] < second[1] or second[3] < first[1])
    expected = inp[y0:y1, x0:x1].copy()
    for _, y, x, color in tiles:
        for dy, dx in np.argwhere(patterns[color]):
            expected[y-y0+1+dy, x-x0+1+dx] = bg
    return expected, {'background': bg, 'tile_size': s,
        'array_shape': (len(ys), len(xs)), 'array_origin': (y0, x0),
        'tile_colors': tuple(tile_colors), 'patterns': patterns, 'boxes': boxes,
        'distractors': len(patterns)-len(tiles)}


def _digest(pair):
    return hashlib.sha256(b''.join(bytes(pair[role].shape)
        + pair[role].astype(np.uint8).tobytes() for role in ('input', 'output'))).hexdigest()


def _grid_key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _orientation_candidates(infos):
    candidates = set(range(8))
    for info in infos:
        for color in info['tile_colors']:
            mask = info['patterns'][color]
            candidates &= {i for i, other in enumerate(_transforms(mask))
                           if np.array_equal(mask, other)}
    return candidates


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_info = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _grid_key(pair['input'])
            assert key not in seen
            seen.add(key)
            assert _digest(pair) not in _SOURCE_DIGESTS
            expected, info = _decode(pair['input'])
            assert np.array_equal(expected, pair['output']), 'incorrect carving or crop'
            assert not np.array_equal(pair['input'], pair['output'])
            if split == 'train':
                train_info.append(info)
    # All sizes and shapes are visible. The curriculum adds no hidden split rule.
    assert _orientation_candidates(train_info) == {0}, 'training leaves orientation ambiguous'
    # A non-half-full matched mask also rejects complemented/rotated copying by
    # cell count. No arbitrary per-color dictionary is shared across pairs.
    assert any(2*m.sum() != m.size for info in train_info
               for color, m in info['patterns'].items() if color in info['tile_colors'])
    assert any(len({m.tobytes() for m in info['patterns'].values()}) > 1
               for info in train_info), 'training does not demonstrate selection'


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    nt, ne = _resolve_counts(rng, num_train, num_test)
    # These are sampling conventions, not hidden semantic parameters.
    mixed_bg, mixed_size = rng.random() < .82, rng.random() < .78
    fixed_bg, fixed_size = int(rng.integers(10)), int(rng.integers(4, 7))
    regime = _pick(rng, ('growth', 'shared', 'mixed', 'mixed', 'row_to_col', 'col_to_row'))
    richness = float(rng.uniform(.12, .9))
    configs = [q for q in _CONFIGS if mixed_size or q[2] == fixed_size]
    shared_shape = _pick(rng, sorted({q[:2] for q in configs if q[0]*q[1] <= 6}))
    backgrounds = [fixed_bg]* (nt+ne)
    if mixed_bg:
        backgrounds = [int(rng.integers(10)) for _ in backgrounds]
        backgrounds[1] = (backgrounds[0]+int(rng.integers(1, 10))) % 10
    witness_corner = int(rng.integers(4))
    puzzle, seen, sizes = {'train': [], 'test': []}, set(), []
    for index in range(nt+ne):
        split = 'train' if index < nt else 'test'
        role = 'source_train' if split == 'train' else 'target_test'
        choices = configs
        if regime == 'growth':
            choices = [q for q in configs if (q[0]*q[1] <= 4 if role == 'source_train'
                                              else q[0]*q[1] >= 6)]
        elif regime == 'shared':
            choices = [q for q in configs if q[:2] == shared_shape]
        elif regime in ('row_to_col', 'col_to_row'):
            horizontal = (regime == 'row_to_col') == (role == 'source_train')
            choices = [q for q in configs if (q[0] == 1 if horizontal else q[1] == 1)
                       and q[0]*q[1] >= 2]
        if mixed_size and index == 1:
            alternate = [q for q in choices if q[2] != sizes[0]]
            if alternate:
                choices = alternate
        config = _pick(rng, choices)
        sizes.append(config[2])
        ntiles = config[0]*config[1]
        possible_d = [d for d in range(10-ntiles)
                      if _bank_options(*config, ntiles+d)]
        # On one-tile demonstrations include color-selection evidence.
        if index < 2 and ntiles == 1:
            possible_d = [d for d in possible_d if d > 0]
        distractors = _pick(rng, possible_d)
        witness = (witness_corner if index == 0 else witness_corner ^ 1) if index < 2 else None
        for attempt in range(32):
            pair = _make_pair(rng, config, backgrounds[index], distractors,
                              rng.random() < richness, witness)
            if _grid_key(pair['input']) not in seen and _digest(pair) not in _SOURCE_DIGESTS:
                seen.add(_grid_key(pair['input']))
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError('exhausted distinct configurations')
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    return tuple(tuple(_digest(p) for p in puzzle[s]) for s in ('train', 'test'))


def _connected(mask):
    return len(_components(mask.astype(np.int8), 0)) == 1


def self_test() -> None:
    train_counts, test_counts, fingerprints = set(), set(), set()
    shapes, test_shapes, sizes, dimensions, distractors, backgrounds = set(), set(), set(), set(), set(), set()
    symmetric = asymmetric = connected = disconnected = near_full = sparse = duplicates = near_matches = 0
    mixed_background = mixed_scale = multi_side = 0
    regimes = {name: 0 for name in ('growth', 'shared', 'mixed_train', 'mixed_test',
                                   'row_to_col', 'col_to_row', 'not_larger')}
    bank_rows, bank_cols, bank_gaps = set(), set(), set()
    vacancies = scattered = combined = color_role_reuse = 0
    compact_case = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        fingerprints.add(_fingerprint(puzzle))
        bs, ss, puzzle_infos = set(), set(), {'train': [], 'test': []}
        complex_pairs = 0
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                expected, info = _decode(pair['input'].copy())
                assert np.array_equal(expected, pair['output'])
                puzzle_infos[split].append(info)
                if split == 'train' and max(pair['input'].shape) <= 19:
                    compact_case = (puzzle, pair)
                bs.add(info['background']); ss.add(info['tile_size'])
                backgrounds.add(info['background']); sizes.add(info['tile_size'])
                shapes.add(info['array_shape']); distractors.add(info['distractors'])
                dimensions.add(pair['input'].shape)
                if split == 'test':
                    test_shapes.add(info['array_shape'])
                masks = list(info['patterns'].values())
                for m in masks:
                    asym = len({x.tobytes() for x in _transforms(m)}) == 8
                    asymmetric += asym; symmetric += not asym
                    conn = _connected(m)
                    connected += conn; disconnected += not conn
                    near_full += m.sum() == m.size-1
                    sparse += m.sum() <= max(2, m.size//3)
                duplicates += any(np.array_equal(a, b) for i, a in enumerate(masks) for b in masks[i+1:])
                near_matches += any(np.count_nonzero(a != b) == 1 for i, a in enumerate(masks) for b in masks[i+1:])
                ay, ax = info['array_origin']
                ah, aw = expected.shape
                sides = set()
                for y, x, bottom, right in info['boxes'].values():
                    if bottom <= ay: sides.add('above')
                    if y >= ay+ah: sides.add('below')
                    if right <= ax: sides.add('left')
                    if x >= ax+aw: sides.add('right')
                multi_side += len(sides) > 1
                yy = sorted({box[0] for box in info['boxes'].values()})
                xx = sorted({box[1] for box in info['boxes'].values()})
                tidy = (len(yy)*len(xx) <= 2*len(masks)
                        and all(b-a >= info['tile_size']-1
                                for coords in (yy, xx) for a, b in zip(coords, coords[1:])))
                if tidy:
                    bank_rows.add(len(yy)); bank_cols.add(len(xx))
                    bank_gaps.update(b-a-(info['tile_size']-2)
                                     for coords in (yy, xx) for a, b in zip(coords, coords[1:]))
                    vacancies += len(yy)*len(xx) > len(masks)
                else:
                    scattered += 1
                symmetries = [len({x.tobytes() for x in _transforms(m)}) for m in masks]
                complex_pairs += (not tidy and min(symmetries) < 8 and max(symmetries) == 8
                                  and info['distractors'] > 0)
        mixed_background += len(bs) > 1; mixed_scale += len(ss) > 1
        tr = {i['array_shape'] for i in puzzle_infos['train']}
        te = {i['array_shape'] for i in puzzle_infos['test']}
        regimes['growth'] += min(r*c for r, c in te) > max(r*c for r, c in tr)
        regimes['shared'] += bool(tr & te)
        regimes['mixed_train'] += len(tr) > 1
        regimes['mixed_test'] += len(te) > 1
        regimes['not_larger'] += min(r*c for r, c in te) <= max(r*c for r, c in tr)
        regimes['row_to_col'] += any(r == 1 and c > 1 for r, c in tr) and any(r > 1 and c == 1 for r, c in te)
        regimes['col_to_row'] += any(r > 1 and c == 1 for r, c in tr) and any(r == 1 and c > 1 for r, c in te)
        combined += complex_pairs if len(bs) > 1 and len(ss) > 1 and len(tr | te) > 1 else 0
        color_role_reuse += bool(bs & {color for infos in puzzle_infos.values()
                                      for info in infos for color in info['patterns']})
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(fingerprints) == 42
    assert len(shapes) >= 12 and len(test_shapes) >= 10 and len(dimensions) >= 60
    assert sizes == set(range(4, 9)) and backgrounds == set(range(10))
    assert {0, 1, 2, 3, 4, 5} <= distractors
    assert 20 <= mixed_background < 42 and 18 <= mixed_scale < 42
    assert min(symmetric, asymmetric, connected, disconnected, near_full, sparse) > 50
    assert duplicates > 20 and near_matches > 20 and multi_side > 40
    assert min(regimes.values()) >= 3
    assert min(len(bank_rows), len(bank_cols)) >= 4 and {1, 2, 3} <= bank_gaps
    assert vacancies > 10 and scattered > 40 and combined > 10 and color_role_reuse > 20
    # Source dimensions are legal: only exact source contents are prohibited.
    assert compact_case is not None
    original, selected = compact_case
    for shape in ((19, 19), (19, 23)):
        copied = {split: [{role: grid.copy() for role, grid in pair.items()}
                          for pair in original[split]] for split in ('train', 'test')}
        index = next(i for i, pair in enumerate(original['train']) if pair is selected)
        grid = selected['input']
        copied['train'][index]['input'] = np.pad(grid,
            ((0, shape[0]-grid.shape[0]), (0, shape[1]-grid.shape[1])),
            constant_values=int(grid[0, 0]))
        validate_generated_puzzle(copied)
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        a = generate_puzzle(314159, *counts)
        assert _fingerprint(a) == _fingerprint(generate_puzzle(314159, *counts))
        assert _fingerprint(a) != _fingerprint(generate_puzzle(314160, *counts))
        for split, count in zip(('train', 'test'), counts):
            if count is not None:
                assert len(a[split]) == count
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (3, 1.5)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('accepted illegal pair count')
    # A mutation must be rejected by geometry and output checks.
    p = generate_puzzle(123, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('accepted a damaged crop margin')
