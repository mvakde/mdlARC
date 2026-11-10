"""Ghost stencil interference: recover hidden loops, then cancel shared ink.

Only NumPy and the standard library are used. All semantic data are in the input.
The scene renderer and the input reader share the stated perimeter definition,
but truth is built from sampled complete objects, not by calling solve.
"""
from __future__ import annotations

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer rectangles of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, c) for c in range(l, r + 1)}
            | {(b, c) for c in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _key(a):
    return a.shape, a.tobytes()


def _dihedral(a, index):
    return np.rot90(np.fliplr(a) if index >= 4 else a, index % 4).copy()


def _read_upright(g, n):
    """Read a bottom library and reconstruct every source-colored rectangle."""
    split = g.shape[0] - n - 4
    assert split >= 8
    bg, sep = int(g[0, 0]), int(g[split, 0])
    assert bg != sep and np.all(g[split] == sep)
    bottom = np.flatnonzero(g[-1] == sep)
    assert len(bottom) and np.array_equal(bottom, np.arange(len(bottom)))
    core = len(bottom)
    assert (core - 1) % (n + 1) == 0
    k = (core - 1) // (n + 1)
    assert 2 <= k <= 4
    assert np.all(g[split + 1:, core:] == bg)
    assert np.all(g[split + n + 1, :core] == sep)
    labels, masks, targets = [], [], []
    for j in range(k + 1):
        assert np.all(g[split + 1:, j * (n + 1)] == sep)
    for j in range(k):
        c = j * (n + 1) + 1
        tile = g[split + 1:split + n + 1, c:c + n]
        colors = np.unique(tile[tile != bg])
        assert len(colors) == 1
        label = int(colors[0])
        assert label != sep and label not in labels
        swatch = g[split + n + 2, c:c + n]
        assert np.all(swatch == swatch[0]) and swatch[0] != bg
        labels.append(label)
        masks.append(tile == label)
        targets.append(int(swatch[0]))
    scene = g[:split]
    assert (np.all(scene[0] == bg) and np.all(scene[-1] == bg)
            and np.all(scene[:, 0] == bg) and np.all(scene[:, -1] == bg))
    assert not np.any(scene == sep)
    possible = set(map(int, np.unique(scene))) - {bg, *labels}
    frames = []
    for color in sorted(possible):
        cells = set(map(tuple, np.argwhere(scene == color).tolist()))
        if not cells:
            continue
        box = _bbox(cells)
        t, b, l, r = box
        if b - t < 5 or r - l < 5 or cells != _perimeter(box):
            continue
        interior = scene[t + 1:b, l + 1:r]
        fill = int(interior[0, 0])
        if (np.all(interior == fill) and fill not in {sep, color, *labels}
                and (b - t - 1) * n <= 30 and (r - l - 1) * n <= 30):
            frames.append((box, color, fill))
    assert len(frames) == 1, "expected one opaque scene window"
    window, frame, fill = frames[0]
    assert possible == ({frame, fill} - {bg})
    assert all(target != fill for target in targets)
    covered = _area(window)
    boxes, layers = [], []
    t, b, l, r = window
    counts = np.zeros(scene.shape, dtype=np.int8)
    for label in labels:
        visible = set(map(tuple, np.argwhere(scene == label).tolist()))
        assert visible, "every library identity must have visible scene evidence"
        box = _bbox(visible)
        assert box[1] - box[0] >= 2 and box[3] - box[2] >= 2
        full = _perimeter(box)
        assert visible == full - covered, "incomplete or ambiguous rectangular evidence"
        layer = np.zeros((b - t - 1, r - l - 1), dtype=bool)
        for y, x in full:
            counts[y, x] += 1
            if t < y < b and l < x < r:
                layer[y - t - 1, x - l - 1] = True
        boxes.append(box)
        layers.append(layer)
    assert counts.max() <= 2, "three-way intersections are outside this family"
    assert all(counts[y, x] <= 1 for y, x in _area((0, split - 1, 0, g.shape[1] - 1))
               if (y, x) not in covered), "visible outlines must not overwrite one another"
    return dict(n=n, k=k, bg=bg, sep=sep, frame=frame, fill=fill,
                labels=labels, masks=masks, targets=targets, boxes=boxes,
                layers=layers, window=window, scene_shape=scene.shape)


def _read(input_grid):
    g = _grid(input_grid)
    candidates = []
    for turns in range(4):
        upright = np.rot90(g, turns)
        for n in (2, 3, 4):
            # A whole-scene reflection can move the library's unused padding
            # to the left. With no padding either reading is equivalent.
            for reflected in (False, True):
                try:
                    p = _read_upright(np.fliplr(upright) if reflected else upright, n)
                except (AssertionError, IndexError, ValueError):
                    continue
                p.update(turns=turns, reflected=reflected)
                candidates.append(p)
                break
    if len(candidates) != 1:
        raise ValueError(f"expected one scene/library interpretation, found {len(candidates)}")
    return candidates[0]


def _render(p, mode='normal', stamp_transform=0):
    """Input-derived block renderer, also used for explicit ablation checks."""
    n = p['n']
    h, w = p['layers'][0].shape
    result = np.full((h * n, w * n), p['fill'], dtype=np.int8)
    count = np.zeros(result.shape, dtype=np.int8)
    ink = result.copy()
    layers = p['layers']
    if mode == 'position':
        layers = [p['layers'][i] for i in sorted(range(p['k']), key=lambda i: p['boxes'][i])]
    for label, mask, target, layer in zip(p['labels'], p['masks'], p['targets'], layers):
        if mode == 'no_completion':
            continue
        tile = np.ones_like(mask) if mode == 'solid_tiles' else _dihedral(mask, stamp_transform)
        color = label if mode == 'no_key' else target
        for y, x in zip(*np.nonzero(layer)):
            sl = np.s_[y * n:(y + 1) * n, x * n:(x + 1) * n]
            count[sl] += tile
            ink[sl][tile] = color
    if mode == 'last_wins':
        result[count > 0] = ink[count > 0]
    else:
        result[count == 1] = ink[count == 1]
    if p['reflected']:
        result = np.fliplr(result)
    return np.rot90(result, -p['turns']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the answer from this input alone; no training or seed is needed."""
    return _render(_read(input_grid))


def _motif(rng, n):
    style = int(rng.integers(9))
    a = np.zeros((n, n), dtype=bool)
    yy, xx = np.indices(a.shape)
    if style == 0:
        a[:] = True
    elif style == 1:
        a[0] = a[-1] = True
        a[:, 0] = a[:, -1] = True
    elif style == 2:
        a = yy == xx
    elif style == 3:
        a = (yy + xx) % 2 == int(rng.integers(2))
    elif style == 4:
        a[int(rng.integers(n)), :] = True
        a[:, int(rng.integers(n))] = True
    elif style == 5:
        a = (yy if rng.integers(2) else xx) % 2 == int(rng.integers(2))
    elif style == 6:
        ys, xs = sorted(rng.choice(n + 1, 2, replace=False)), sorted(rng.choice(n + 1, 2, replace=False))
        a[ys[0]:ys[1], xs[0]:xs[1]] = True
    else:
        a = rng.random((n, n)) < rng.uniform(.2, .8)
    if not a.any():
        a.flat[int(rng.integers(n * n))] = True
    return _dihedral(a, int(rng.integers(8)))


def _sample_masks(rng, n, k, demo):
    if demo == 0:
        # Identical silhouettes with different inks make complete cancellation visible.
        a = rng.random((n, n)) < .55
        if not a.any() or a.all():
            a[:] = False
            a[0, :2] = True
        masks = [a.copy(), a.copy()]
    elif demo == 1:
        for _ in range(100):
            a, b = rng.random((2, n, n)) < .55
            if (a & b).any() and (a & ~b).any() and (b & ~a).any():
                break
        else:
            a = np.zeros((n, n), dtype=bool)
            b = a.copy()
            a[0, :2] = True
            b[:2, 0] = True
        masks = [a, b]
    else:
        masks = [_motif(rng, n) for _ in range(2)]
    masks.extend(_motif(rng, n) for _ in range(k - 2))
    return masks


def _random_box(rng, h, w):
    ys = sorted(map(int, rng.choice(np.arange(1, h - 1), 2, replace=False)))
    xs = sorted(map(int, rng.choice(np.arange(1, w - 1), 2, replace=False)))
    return ys[0], ys[1], xs[0], xs[1]


def _geometry(rng, n, k, style, outside_extra):
    """Crossings are hidden, but each loop's four extrema remain visible."""
    oh = int(rng.integers(4, min(9, 30 // n) + 1))
    ow = int(rng.integers(4, min(10, 30 // n) + 1))
    h = int(rng.integers(max(15, oh + 8), min(24, 30 - n - 4) + 1))
    w = int(rng.integers(max(16, ow + 8, k * (n + 1) + 1), 31))
    t = int(rng.integers(3, h - oh - 4))
    l = int(rng.integers(3, w - ow - 4))
    b, r = t + oh + 1, l + ow + 1
    window = t, b, l, r
    covered = _area(window)
    boxes, occupied = [], set()
    counts = np.zeros((h, w), dtype=np.int8)

    def accept(box, require_hidden=None):
        y0, y1, x0, x1 = box
        if y1 - y0 < 2 or x1 - x0 < 2:
            return False
        full = _perimeter(box)
        visible = full - covered
        hidden = {(y, x) for y, x in full if t < y < b and l < x < r}
        if not visible or _bbox(visible) != box or (full - covered) & (occupied - covered):
            return False
        if require_hidden is not None and bool(hidden) != require_hidden:
            return False
        if any(counts[p] == 2 for p in full):
            return False
        boxes.append(box)
        occupied.update(full)
        for p in full:
            counts[p] += 1
        return True

    rows = sorted(map(int, rng.choice(np.arange(t + 1, b), 2, replace=False)))
    cols = sorted(map(int, rng.choice(np.arange(l + 1, r), 2, replace=False)))
    if style == 'corner':
        first = (int(rng.integers(1, t)), rows[1], int(rng.integers(1, l)), cols[1])
        second = (rows[0], int(rng.integers(b + 1, h - 1)), cols[0], int(rng.integers(r + 1, w - 1)))
        assert accept(first, True) and accept(second, True)
    elif style == 'spanning':
        if rows[1] - rows[0] < 2:
            rows = [t + 1, b - 1]
        if cols[1] - cols[0] < 2:
            cols = [l + 1, r - 1]
        first = (int(rng.integers(1, t)), int(rng.integers(b + 1, h - 1)), cols[0], cols[1])
        second = (rows[0], rows[1], int(rng.integers(1, l)), int(rng.integers(r + 1, w - 1)))
        assert accept(first, True) and accept(second, True)
    for index in range(len(boxes), k):
        requirement = False if outside_extra and index == k - 1 else True
        for attempt in range(700):
            box = _random_box(rng, h, w)
            if accept(box, requirement):
                break
        else:
            return None
    # Reflect the scene without changing the library location; stamps keep their own orientation.
    flip_y, flip_x = bool(rng.integers(2)), bool(rng.integers(2))
    def reflect(box):
        a, z, c, d = box
        return (h - 1 - z, h - 1 - a, w - 1 - d, w - 1 - c) if flip_y and flip_x else (
            (h - 1 - z, h - 1 - a, c, d) if flip_y else
            (a, z, w - 1 - d, w - 1 - c) if flip_x else box)
    return h, w, reflect(window), [reflect(box) for box in boxes]


def _make_pair(rng, n, k, demo, style, presentation, previous_bg, allow_outside=True):
    colors = list(map(int, rng.permutation(10)))
    if colors[0] == previous_bg:
        colors[0], colors[1] = colors[1], colors[0]
    bg, sep, frame = colors[:3]
    fill = bg if demo == 0 or (demo == 2 and rng.random() < .4) else colors[3]
    labels = colors[4:4 + k]
    masks = _sample_masks(rng, n, k, demo)
    pool = [c for c in range(10) if c not in (bg, fill)]
    for _ in range(100):
        targets = list(map(int, rng.choice(pool, k, replace=False)))
        if demo == 2 or all(a != b for a, b in zip(labels, targets)):
            break
    else:
        targets = labels[1:] + labels[:1]
    if demo == 2 and rng.random() < .3:
        targets[1] = targets[0]
    outside = allow_outside and k >= 3 and (demo == 1 or (demo == 2 and rng.random() < .35))
    for _ in range(40):
        geometry = _geometry(rng, n, k, style, outside)
        if geometry is not None:
            break
    else:
        raise RuntimeError('bounded geometry construction exhausted')
    h, w, window, boxes = geometry
    scene = np.full((h, w), bg, dtype=np.int8)
    for label, box in zip(labels, boxes):
        for p in _perimeter(box):
            scene[p] = label
    t, b, l, r = window
    # Truth from complete per-identity layers; overlap is retained, not overwritten.
    coverage = np.zeros(((b - t - 1) * n, (r - l - 1) * n), dtype=np.int8)
    ink = np.full(coverage.shape, fill, dtype=np.int8)
    for box, mask, target in zip(boxes, masks, targets):
        layer = np.zeros((b - t - 1, r - l - 1), dtype=bool)
        for y, x in _perimeter(box):
            if t < y < b and l < x < r:
                layer[y - t - 1, x - l - 1] = True
        expanded = np.kron(layer, mask)
        coverage += expanded
        ink[expanded] = target
    answer = np.full(coverage.shape, fill, dtype=np.int8)
    answer[coverage == 1] = ink[coverage == 1]
    scene[t:b + 1, l:r + 1] = fill
    for p in _perimeter(window):
        scene[p] = frame
    library = np.full((n + 4, w), bg, dtype=np.int8)
    library[0] = sep
    core = k * (n + 1) + 1
    library[n + 1, :core] = library[n + 3, :core] = sep
    for j in range(k + 1):
        library[1:, j * (n + 1)] = sep
    order = rng.permutation(k)
    for j, identity in enumerate(order):
        c = j * (n + 1) + 1
        library[1:n + 1, c:c + n] = np.where(masks[identity], labels[identity], bg)
        library[n + 2, c:c + n] = targets[identity]
    grid = np.concatenate([scene, library])
    # Targets have no scene role. Only the card attached to a label gives its key.
    return {'input': _dihedral(grid, presentation), 'output': _dihedral(answer, presentation)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    num_train, num_test = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 training, 1..4 test, and num_train > num_test')
    rng = np.random.default_rng(seed)
    n, presentation = int(rng.integers(2, 5)), int(rng.integers(8))
    first_style = str(rng.choice(['corner', 'spanning']))
    training_counts = [2, 3 if rng.random() < .55 else 2]
    training_counts += list(map(int, rng.integers(2, 4, size=num_train - 2)))
    minimum_test = max(training_counts) + 1
    test_counts = list(map(int, rng.integers(minimum_test, 5, size=num_test)))
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_bg = None
    for split, schedule in (('train', training_counts), ('test', test_counts)):
        for index, k in enumerate(schedule):
            demo = index if split == 'train' and index < 2 else 2
            style = first_style if demo == 0 else (
                'spanning' if first_style == 'corner' else 'corner') if demo == 1 else str(
                    rng.choice(['corner', 'spanning', 'free'], p=[.3, .3, .4]))
            for _ in range(80):
                pair = _make_pair(rng, n, k, demo, style, presentation, previous_bg,
                                  allow_outside=split == 'train')
                p = _read(pair['input'])
                if split == 'test' and any(not layer.any() for layer in p['layers']):
                    continue
                if split == 'test' and any(np.array_equal(_render(p), _render(p, mode))
                                          for mode in ('solid_tiles', 'last_wins', 'no_completion')):
                    continue
                if _key(pair['input']) in seen:
                    continue
                if demo == 1:
                    interpretations = [_read(puzzle['train'][0]['input']), p]
                    if any(all(np.array_equal(_render(q), _render(q, stamp_transform=turn))
                               for q in interpretations) for turn in range(1, 8)):
                        continue
                    if all(np.array_equal(_render(q), _render(q, 'position')) for q in interpretations):
                        continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                previous_bg = p['bg']
                break
            else:
                raise RuntimeError('bounded pair construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _pixel_check(p, output):
    """Check each pixel's contributors, including transparent and cancelled cells."""
    upright = np.rot90(output, p['turns'])
    if p['reflected']:
        upright = np.fliplr(upright)
    n = p['n']
    t, b, l, r = p['window']
    assert upright.shape == ((b - t - 1) * n, (r - l - 1) * n)
    perimeters = [_perimeter(box) for box in p['boxes']]
    for y in range(upright.shape[0]):
        for x in range(upright.shape[1]):
            position = (t + 1 + y // n, l + 1 + x // n)
            contributors = [target for full, mask, target in zip(perimeters, p['masks'], p['targets'])
                            if position in full and mask[y % n, x % n]]
            expected = contributors[0] if len(contributors) == 1 else p['fill']
            assert int(upright[y, x]) == expected, 'incorrect interference pixel'


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, parsed = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen
            seen.add(_key(a))
            p = _read(a)
            _pixel_check(p, out)
            assert set(map(int, np.unique(out))) <= {p['fill'], *p['targets']}
            assert any(layer.any() for layer in p['layers'])
            parsed[split].append(p)
    train, test = parsed['train'], parsed['test']
    assert len({p['turns'] for p in train + test}) == 1
    assert len({p['n'] for p in train + test}) == 1
    assert train[0]['fill'] == train[0]['bg'] and train[1]['fill'] != train[1]['bg']
    assert train[0]['bg'] != train[1]['bg']
    assert min(p['k'] for p in test) > max(p['k'] for p in train)
    assert all(all(layer.any() for layer in p['layers']) for p in test)
    assert all(not np.array_equal(_render(p), _render(p, mode)) for p in test
               for mode in ('solid_tiles', 'last_wins', 'no_completion'))
    # Actual answer differences, rather than unobservable generator mode labels.
    for mode in ('solid_tiles', 'last_wins', 'no_key', 'no_completion', 'position'):
        assert any(not np.array_equal(_render(p), _render(p, mode)) for p in train), mode
    assert all(any(not np.array_equal(_render(p), _render(p, stamp_transform=turn))
                   for p in train) for turn in range(1, 8)), 'stamp orientation not demonstrated'
    overlap_examples, partial_examples = 0, 0
    for p in train:
        for i in range(p['k']):
            for j in range(i + 1, p['k']):
                if not (p['layers'][i] & p['layers'][j]).any():
                    continue
                a, b = p['masks'][i], p['masks'][j]
                if np.array_equal(a, b) and p['targets'][i] != p['targets'][j]:
                    overlap_examples += 1
                if (a & b).any() and (a & ~b).any() and (b & ~a).any():
                    partial_examples += 1
    assert overlap_examples and partial_examples, 'complete and partial cancellation need demonstrations'


def self_test():
    """Bounded 42-seed structural, semantic-ablation, count and mutation checks."""
    shapes, motifs, layouts, palettes, scales, counts, presentations = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _read(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                motifs.update(_key(a) for a in p['masks'])
                layouts.add((p['scene_shape'], p['window'], tuple(sorted(p['boxes']))))
                palettes.add((p['bg'], p['sep'], p['frame'], p['fill']))
                scales.add(p['n'])
                counts.add(p['k'])
                presentations.add(p['turns'])
    assert scales == {2, 3, 4} and counts == {2, 3, 4} and presentations == {0, 1, 2, 3}
    assert len(shapes) >= 50 and len(motifs) >= 80 and len(layouts) >= 150 and len(palettes) >= 100
    # Every legal explicit request, plus defaults and rejection of invalid types/counts.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(5000 + nt * 10 + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    one, two = generate_puzzle(701, 2, 1), generate_puzzle(701, 2, 1)
    assert all(_key(a[name]) == _key(b[name]) for s in one for a, b in zip(one[s], two[s])
               for name in ('input', 'output'))
    bad = {s: [{name: grid.copy() for name, grid in pair.items()} for pair in one[s]] for s in one}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupted answer accepted')
    # Damaging one exposed corner must fail completion, not create a cached answer.
    a = one['train'][0]['input'].copy()
    p = _read(a)
    upright = np.rot90(a, p['turns']).copy()
    if p['reflected']:
        upright = np.fliplr(upright).copy()
    full = _perimeter(p['boxes'][0]) - _area(p['window'])
    y, x = min(full)
    upright[y, x] = p['bg']
    try:
        solve(np.rot90(np.fliplr(upright) if p['reflected'] else upright, -p['turns']))
    except ValueError:
        pass
    else:
        raise AssertionError('broken outline accepted')
    return {'seeds_checked': 42, 'shape_pairs': len(shapes), 'motifs': len(motifs),
            'layouts': len(layouts), 'role_palettes': len(palettes)}
