"""Copy the complete local motif at isolated cells of its anchor color."""
import hashlib
from collections import Counter
import numpy as np

_SOURCE = {
    ((9, 22), 'e9849f8a0bc746df846a6859ed579b1282169ed301cd7c9ae9d82f1fd5e80a14'),
    ((16, 14), '3afb3c991da1b1ac47ff1de7651bf36d4321d660c561b0a6fb7e946d38c90e0a'),
    ((13, 13), 'f59a3a0b51ae688b6d2c7144b49f64ecbd340a24f95b5aad4677267a1b48dc27'),
    ((14, 10), 'd9195495c14cbec8319418ab7a576738a8f04953d75ad467d0c9913df397a5a4'),
    ((6, 20), 'd88e5ce16f4cf97ef137ae2f8ebbf95c58cfefbaa50f70cbe61c29e7f2ee267b'),
}
_STYLES = ('solid', 'two_color', 'hollow', 'stripe', 'wide', 'asymmetric', 'silhouette')


def _counts(rng, train, test):
    for name, value, lo, hi in (('num_train', train, 2, 10), ('num_test', test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f'invalid {name}')
    if test is None:
        test = int(rng.integers(1, min(4, (train or 10) - 1) + 1))
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return int(train), int(test)


def _motif(rng, style, bg, colors):
    a, b, c = colors[0], colors[min(1, len(colors)-1)], colors[min(2, len(colors)-1)]
    if style == 'solid':
        return np.full((int(rng.integers(2, 7)), int(rng.integers(2, 8))), a, np.int8)
    if style == 'wide':
        w = int(rng.integers(6, 10))
        m = np.full((2, w), a, np.int8)
        start = int(rng.integers(1, w-2))
        m[1, start:start+int(rng.integers(1, min(4, w-start)))] = bg
        return m
    h, w = int(rng.integers(3, 8)), int(rng.integers(3, 9))
    if style == 'two_color':
        h, w = max(h, 4), max(w, 4)
        m = np.full((h, w), a, np.int8)
        m[1:-1, 1:-1] = b
    elif style == 'hollow':
        m = np.full((h, w), a, np.int8)
        m[1:-1, 1:-1] = bg
    elif style == 'stripe':
        h, w = (4, 5) if rng.random() < 0.35 else (max(h, 4), max(w, 5))
        m = np.full((h, w), a, np.int8)
        m[1:-1, 1:-1:2] = b
    elif style == 'asymmetric':
        h, w = max(h, 4), max(w, 4)
        m = np.full((h, w), a, np.int8)
        split = int(rng.integers(2, w-1))
        m[1:-1, 1:split] = b
        m[1:-1, split:-1] = c
        if rng.random() < 0.6:
            m[int(rng.integers(1, h-1)), int(rng.integers(1, w-1))] = bg
    elif style == 'silhouette':
        m = np.full((h, w), bg, np.int8)
        m[0, :] = a
        m[:, 0] = a
        kind = int(rng.integers(3))
        if kind == 0:
            m[-1, :] = a
        elif kind == 1:
            m[:, -1] = a
        else:
            for y in range(1, h):
                m[y, 1:1+max(1, (w-1)*(h-y)//h)] = a
        if b != a and rng.random() < 0.5:
            m[1:min(h, 3), 0] = b
    else:
        raise AssertionError('unknown motif')
    return m


def _separate(box, boxes, pad=1):
    y, x, h, w = box
    return all(y >= oy+oh+pad or oy >= y+h+pad or x >= ox+ow+pad or ox >= x+w+pad
               for oy, ox, oh, ow in boxes)


def _sites(rng, mh, mw, markers, arrangement):
    n = markers+1
    if arrangement in ('row', 'column'):
        span_size = mw if arrangement == 'row' else mh
        gap = min(int(rng.integers(1, 5)), max(1, (30-n*span_size)//(n-1)))
        span = n*span_size+(n-1)*gap
        if span > 30:
            return None
        if arrangement == 'row':
            h = int(rng.integers(mh+1, min(30, mh+10)+1))
            w = int(rng.integers(span, min(30, span+8)+1))
            y, x0 = int(rng.integers(h-mh+1)), int(rng.integers(w-span+1))
            coords = [(y, x0+i*(mw+gap)) for i in range(n)]
        else:
            h = int(rng.integers(span, min(30, span+8)+1))
            w = int(rng.integers(mw+1, min(30, mw+10)+1))
            y0, x = int(rng.integers(h-span+1)), int(rng.integers(w-mw+1))
            coords = [(y0+i*(mh+gap), x) for i in range(n)]
        if rng.random() < 0.5:
            coords.reverse()
        return h, w, coords
    for _ in range(35):
        h = int(rng.integers(max(9, mh+2), 31))
        w = int(rng.integers(max(9, mw+2), 31))
        if h*w < n*(mh+1)*(mw+1):
            continue
        boxes = []
        for _ in range(n):
            found = None
            for _ in range(100):
                box = (int(rng.integers(h-mh+1)), int(rng.integers(w-mw+1)), mh, mw)
                if _separate(box, boxes):
                    found = box
                    break
            if found is None:
                break
            boxes.append(found)
        if len(boxes) == n:
            return h, w, [(y, x) for y, x, _, _ in boxes]
    return None


def _distractor(rng, color, bg):
    kind = int(rng.integers(5))
    h, w = int(rng.integers(2, 8)), int(rng.integers(2, 8))
    if kind == 0:
        h = 1
    elif kind == 1:
        w = 1
    m = np.full((h, w), color, np.int8)
    if kind == 2:
        m[:-1, 1:] = bg
    elif kind == 3 and h >= 3 and w >= 3:
        m[1:-1, 1:-1] = bg
    elif kind == 4:
        for y in range(1, h):
            m[y, max(1, w-y):] = bg
    return m


def _components(grid, bg):
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    result = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or grid[y, x] == bg:
                continue
            seen[y, x] = True
            stack, cells = [(y, x)], []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] != bg:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            result.append(cells)
    return result


def _place_distractor(rng, grid, bg, shape, reserved):
    h, w = grid.shape
    sh, sw = shape.shape
    if sh > h or sw > w:
        return False
    options = [(y, x) for y in range(h-sh+1) for x in range(w-sw+1)]
    for index in rng.permutation(len(options)):
        y, x = options[int(index)]
        if not _separate((y, x, sh, sw), reserved, 0):
            continue
        if np.any(grid[max(0, y-1):min(h, y+sh+1), max(0, x-1):min(w, x+sw+1)] != bg):
            continue
        mask = shape != bg
        grid[y:y+sh, x:x+sw][mask] = shape[mask]
        return True
    return False


def _place_contact(rng, grid, bg, motif, coords, reserved, color):
    """Put a distinct bar beside a future copy, away from its input marker."""
    h, w = grid.shape
    mh, mw = motif.shape
    options = []
    for y, x in coords[1:]:
        for off in range(1, mh):
            if motif[off, -1] != bg and x+mw < w:
                options.append((y+off, x+mw, 'right'))
        for off in range(1, mw):
            if motif[-1, off] != bg and y+mh < h:
                options.append((y+mh, x+off, 'below'))
    for index in rng.permutation(len(options)):
        y, x, direction = options[int(index)]
        if not _separate((y, x, 1, 1), reserved, 0):
            continue
        if np.any(grid[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)] != bg):
            continue
        # Extend along the boundary when space permits; a point plus a line
        # makes the unchanged object visually distinct from a marker.
        extension = [(y+1, x), (y-1, x)] if direction == 'right' else [(y, x+1), (y, x-1)]
        for ey, ex in extension:
            if not (0 <= ey < h and 0 <= ex < w):
                continue
            if not _separate((ey, ex, 1, 1), reserved, 0):
                continue
            if np.any(grid[max(0, ey-1):min(h, ey+2), max(0, ex-1):min(w, ex+2)] != bg):
                continue
            grid[y, x] = color
            grid[ey, ex] = color
            return True
    return False


def _pair(rng, style, marker_count, arrangement, color_pool, fresh=None):
    for _ in range(100):
        bg_choices = [0] + list(color_pool)
        bg = int(rng.choice([c for c in bg_choices if c != fresh])) if rng.random() < 0.4 else 0
        available = [c for c in color_pool if c != bg]
        if fresh is not None:
            available = [fresh] + [c for c in available if c != fresh]
        else:
            available = list(rng.permutation(available))
        if len(available) < 2:
            continue
        colors = available[:min(len(available), int(rng.integers(2, 4)))]
        motif = _motif(rng, style, bg, colors)
        mh, mw = motif.shape
        sites = _sites(rng, mh, mw, marker_count, arrangement)
        if sites is None:
            continue
        h, w, coords = sites
        grid = np.full((h, w), bg, np.int8)
        ry, rx = coords[0]
        grid[ry:ry+mh, rx:rx+mw] = motif
        for y, x in coords[1:]:
            grid[y, x] = motif[0, 0]
        reserved = [(y, x, mh, mw) for y, x in coords]
        distractor_colors = [c for c in color_pool if c != bg and c not in set(np.unique(motif))]
        wanted = int(rng.integers(5)) if distractor_colors else 0
        if distractor_colors and rng.random() < 0.24:
            _place_contact(rng, grid, bg, motif, coords, reserved,
                           int(rng.choice(distractor_colors)))
        for _ in range(wanted):
            color = int(rng.choice(distractor_colors))
            _place_distractor(rng, grid, bg, _distractor(rng, color, bg), reserved)
        if np.count_nonzero(grid == bg) <= grid.size//2:
            continue
        out = grid.copy()
        for y, x in coords[1:]:
            out[y:y+mh, x:x+mw] = motif
        try:
            _inspect_pair(grid, out)
        except AssertionError:
            continue
        return grid, out
    raise RuntimeError('pair construction exhausted')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    # Alignment and copying are shared. Colors, backgrounds, geometry are local.
    pool = list(map(int, rng.choice(np.arange(1, 10), size=int(rng.integers(4, 8)), replace=False)))
    holdout = [c for c in range(1, 10) if c not in pool]
    classic = rng.random() < 0.22
    train_styles = ['solid', 'two_color'] if classic else list(rng.choice(_STYLES, size=2, replace=False))
    puzzle = {'train': [], 'test': []}
    styles_seen = set()
    for side, count in (('train', nt), ('test', ns)):
        for i in range(count):
            prng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
            if side == 'train':
                style = train_styles[i] if i < 2 else str(prng.choice(_STYLES))
                markers = 2 if classic and i < 2 else int(prng.integers(1, 4))
                colors, fresh = pool, None
            else:
                choices = [s for s in _STYLES if s not in styles_seen]
                style = str(prng.choice(choices if choices and prng.random() < 0.7 else _STYLES))
                markers = int(prng.integers(1, 4))
                colors = list(range(1, 10))
                fresh = int(prng.choice(holdout)) if holdout and (i == 0 or prng.random() < 0.55) else None
            arrangement = str(prng.choice(('row', 'column', 'scatter'), p=(0.32, 0.32, 0.36)))
            pair = None
            for option in (arrangement, 'scatter', 'row', 'column'):
                try:
                    pair = _pair(prng, style, markers, option, colors, fresh)
                    break
                except RuntimeError:
                    pass
            if pair is None:
                raise RuntimeError('cannot construct puzzle pair')
            styles_seen.add(style)
            puzzle[side].append({'input': pair[0], 'output': pair[1]})
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(inp, out):
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert counts.max() > inp.size//2
    components = _components(inp, bg)
    singletons = [c[0] for c in components if len(c) == 1]
    assert 1 <= len(singletons) <= 3
    anchor = int(inp[singletons[0]])
    assert all(inp[y, x] == anchor for y, x in singletons)
    candidates = [c for c in components if len(c) > 1 and any(inp[y, x] == anchor for y, x in c)]
    assert len(candidates) == 1
    ref = candidates[0]
    y0, x0 = min(y for y, _ in ref), min(x for _, x in ref)
    y1, x1 = max(y for y, _ in ref)+1, max(x for _, x in ref)+1
    motif = inp[y0:y1, x0:x1]
    assert motif[0, 0] == anchor and 2 <= motif.shape[0] <= 9 and 2 <= motif.shape[1] <= 9
    assert np.count_nonzero(motif != bg) == len(ref)
    assert all(anchor not in {int(inp[y, x]) for y, x in c}
               for c in components if c is not ref and len(c) > 1)
    predicted = inp.copy()
    for y, x in singletons:
        assert y+motif.shape[0] <= inp.shape[0] and x+motif.shape[1] <= inp.shape[1]
        patch = inp[y:y+motif.shape[0], x:x+motif.shape[1]]
        assert np.count_nonzero(patch != bg) == 1 and patch[0, 0] == anchor
        predicted[y:y+motif.shape[0], x:x+motif.shape[1]] = motif
    assert np.array_equal(predicted, out) and not np.array_equal(inp, out)
    return bg, motif, singletons, ref, components


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    inputs, pairs = set(), set()
    for side in ('train', 'test'):
        for pair in puzzle[side]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = (inp.shape, inp.astype(np.int8).tobytes())
            digest = hashlib.sha256(inp.astype(np.int8).tobytes()+out.astype(np.int8).tobytes()).hexdigest()
            assert key not in inputs and (inp.shape, digest) not in pairs
            assert (inp.shape, digest) not in _SOURCE
            inputs.add(key)
            pairs.add((inp.shape, digest))
            _inspect_pair(inp, out)
    # All training pairs demonstrate the same visible copy and alignment rule.
    assert all(np.any(p['input'] != p['output']) for p in puzzle['train'])


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts, shapes, palettes, arrangements, marker_counts = (set() for _ in range(6))
    signatures = set()
    coverage = Counter()
    source_shapes = {shape for shape, _ in _SOURCE}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(seed))
        train_colors = {int(c) for pair in puzzle['train'] for c in np.unique(pair['input'])}
        train_markers = set()
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        signatures.add(tuple((p['input'].shape, p['input'].tobytes())
                             for side in ('train', 'test') for p in puzzle[side]))
        for side in ('train', 'test'):
            for pair in puzzle[side]:
                inp, out = pair['input'], pair['output']
                bg, motif, markers, ref, components = _inspect_pair(inp, out)
                shapes.add(inp.shape)
                palettes.add(tuple(sorted(set(np.unique(inp).tolist()))))
                marker_counts.add(len(markers))
                coverage['nonzero_bg'] += bg != 0
                coverage['small'] += inp.shape[0] <= 9
                coverage['source_shape'] += inp.shape in source_shapes
                coverage['source_stripe'] += (motif.shape == (4, 5) and
                    np.array_equal(motif[0], motif[-1]) and np.array_equal(motif[1], motif[2]) and
                    motif[1, 1] != motif[1, 2] and motif[1, 3] != motif[1, 2])
                distractors = [c for c in components if c is not ref and len(c) > 1]
                coverage[f'distractors_{len(distractors)}'] += 1
                dcolors = [int(inp[c[0]]) for c in distractors]
                coverage['repeated_distractor_color'] += len(dcolors) > len(set(dcolors))
                for component in distractors:
                    dh = max(y for y, _ in component)-min(y for y, _ in component)+1
                    dw = max(x for _, x in component)-min(x for _, x in component)+1
                    coverage['thin_distractor'] += min(dh, dw) == 1
                    coverage['large_distractor'] += max(dh, dw) >= 6
                dpos = {cell for component in distractors for cell in component}
                coverage['contact'] += any((y+int(dy)+vy, x+int(dx)+vx) in dpos
                    for y, x in markers for dy, dx in np.argwhere(motif != bg)
                    for vy, vx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                ry, rx = min(y for y, _ in ref), min(x for _, x in ref)
                arrangement = ('row' if all(y == ry for y, _ in markers) else
                               'column' if all(x == rx for _, x in markers) else 'scatter')
                arrangements.add(arrangement)
                if len(markers) > 1 and arrangement != 'scatter':
                    axis = [x for y, x in markers]+[rx] if arrangement == 'row' else [y for y, x in markers]+[ry]
                    coverage['regular_run'] += len(set(np.diff(sorted(axis)))) == 1
                if side == 'train':
                    train_markers.add(len(markers))
                    coverage['training_gap'] += np.any(motif == bg)
                    coverage['training_pattern'] += len(set(np.unique(motif))-{bg}) > 1
                else:
                    coverage['fresh_test_color'] += bool(set(np.unique(inp))-train_colors)
                    coverage['simple_test'] += np.all(motif == motif[0, 0])
                assert all(np.array_equal(out[y:y+motif.shape[0], x:x+motif.shape[1]], motif)
                           for y, x in markers)
        coverage['within_train_marker_variation'] += len(train_markers) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 40 and len(palettes) >= 20
    assert marker_counts == {1, 2, 3} and arrangements == {'row', 'column', 'scatter'}
    assert all(coverage[k] > 0 for k in (
        'nonzero_bg', 'small', 'source_shape', 'source_stripe',
        'distractors_0', 'distractors_1', 'distractors_2', 'distractors_3',
        'repeated_distractor_color', 'thin_distractor', 'large_distractor',
        'contact', 'regular_run', 'training_gap', 'training_pattern',
        'fresh_test_color', 'simple_test'))
    assert coverage['within_train_marker_variation'] >= 20
    for seed, nt, ns in ((103, 2, 1), (104, 10, 1), (105, 10, 4),
                         (106, None, 1), (107, None, 4), (108, 2, None), (109, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        assert _same(p, generate_puzzle(seed, nt, ns))
        assert nt is None or len(p['train']) == nt
        assert ns is None or len(p['test']) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(123, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid pair count accepted')
