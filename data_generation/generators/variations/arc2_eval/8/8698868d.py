"""Count four-connected motif cavities, transplant motifs, and crop the mosaic.

Literal colors and geometry are visible pair-level properties.  Puzzle-level
sampling policies control the curriculum, never the transformation itself.
"""
import hashlib
import numpy as np


# Content fingerprints include shapes; original dimensions remain available.
_SOURCE_PAIRS = {
    ((20, 15), 'ddc2777c34d8d30575a6c9952e434ae6434366ee4f148e132c192121d648ce94', (6, 12), 'd17f57a591ffc83660790dc1e87bdfb67c504ef571df2a74c754d973fda01f0b'),
    ((30, 30), 'fee59fc68306f1b979b5bd42cc7e2db02bb8a2e6687247a07c4435aba1bc552f', (16, 16), '475d16d15d2c827a7caaf5fd739692a782633d6efbc70eeb312417be34c8c2d5'),
    ((30, 30), '5c90849b8d7863d90d135311998eee399fe9b29738d1204540469570f82888ce', (14, 21), 'eb921f4ac0fe972f97f070f96d4ec9377de9d8fb97828be8ce2fb3130ff2ef48'),
}
_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = _NEIGHBORS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _NEIGHBORS
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                neighbor = (r + dr, c + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        result.append(cells)
    return result


def _count_components(mask):
    return len(_components(mask))


def _transforms(mask):
    return [np.rot90(mask, k) for k in range(4)] + [np.rot90(mask.T, k) for k in range(4)]


def _is_directional(mask):
    return any(not np.array_equal(mask, other) for other in _transforms(mask)[1:])


def _asymmetric(mask):
    return all(not np.array_equal(mask, other) for other in _transforms(mask)[1:])


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _isolated(rng, size, count):
    """Random independent sets, with a capacity-guaranteed fallback."""
    for _ in range(16):
        mask = np.zeros((size, size), dtype=bool)
        for index in rng.permutation(size * size):
            r, c = divmod(int(index), size)
            if any(0 <= r + dr < size and 0 <= c + dc < size and mask[r + dr, c + dc]
                   for dr, dc in _NEIGHBORS):
                continue
            mask[r, c] = True
            if _count_components(np.pad(~mask, 1, constant_values=True)) != 1:
                mask[r, c] = False
                continue
            if int(mask.sum()) == count:
                return mask
    if size <= 3:
        cells = [(r, c) for r in range(size) for c in range(size) if (r + c) % 2 == 0]
    elif size == 4:
        # Seven isolated holes with connected foreground; eight cannot keep
        # the foreground connected in a four-by-four interior.
        cells = [(0, 0), (0, 2), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)]
    else:
        cells = ([(r, c) for r in (0, size - 1) for c in range(0, size, 2)]
                 + [(r, c) for r in range(2, size - 2, 2) for c in (0, size - 1)])
    assert len(cells) >= count
    mask = np.zeros((size, size), dtype=bool)
    for i in rng.permutation(len(cells))[:count]:
        mask[cells[int(i)]] = True
    assert _count_components(np.pad(~mask, 1, constant_values=True)) == 1
    return mask


def _capacity(tile_size):
    # Capacity of isolated cavities while retaining connected foreground.
    n = tile_size - 4
    return {2: 2, 3: 5, 4: 7}.get(n, 8)


def _rectangle_cavities(rng, size, count):
    """Recursively cut rectangles with a one-cell separating wall."""
    boxes = [(0, size, 0, size)]
    while len(boxes) < count:
        cuts = []
        for i, (r0, r1, c0, c1) in enumerate(boxes):
            cuts.extend((i, 0, p) for p in range(r0 + 1, r1 - 1))
            cuts.extend((i, 1, p) for p in range(c0 + 1, c1 - 1))
        if not cuts:
            return None
        i, axis, p = _choice(rng, cuts)
        r0, r1, c0, c1 = boxes.pop(i)
        boxes.extend([(r0, p, c0, c1), (p + 1, r1, c0, c1)] if axis == 0
                     else [(r0, r1, c0, p), (r0, r1, p + 1, c1)])
    mask = np.zeros((size, size), dtype=bool)
    for r0, r1, c0, c1 in boxes:
        if rng.random() < 0.4:
            r0 = int(rng.integers(r0, r1))
            r1 = int(rng.integers(r0 + 1, r1 + 1))
            c0 = int(rng.integers(c0, c1))
            c1 = int(rng.integers(c0 + 1, c1 + 1))
        mask[r0:r1, c0:c1] = True
    return mask


def _make_cavity_mask(rng, motif_size, cavity_count):
    n, k = motif_size - 2, cavity_count
    assert n >= 2 and 0 <= k <= _capacity(motif_size + 2)
    if k == 0:
        return np.zeros((motif_size, motif_size), dtype=bool)
    mode = int(rng.integers(5))
    if k == 1 and mode <= 1:
        # Ordinary hollow squares are a substantial part of the distribution.
        return np.pad(np.ones((n, n), dtype=bool), 1)
    if k == 4 and n >= 3 and mode == 0:
        mask = np.zeros((n, n), dtype=bool)
        q = int(rng.integers(1, (n - 1) // 2 + 1))
        for r in (0, n - q):
            for c in (0, n - q):
                mask[r:r + q, c:c + q] = True
        return np.pad(mask, 1)
    if mode <= 2:
        mask = _rectangle_cavities(rng, n, k)
        if mask is not None:
            return np.pad(_choice(rng, _transforms(mask)), 1).copy()
    mask = np.pad(_isolated(rng, n, k), 1)
    target = int(rng.integers(k, n * n + 1))
    # Growth has no artificial area ceiling and cannot merge cavities or isolate
    # foreground islands. Multiple passes allow long and irregular cavities.
    for _ in range(n * n):
        if int(mask.sum()) >= target:
            break
        candidates = []
        for r in range(1, n + 1):
            for c in range(1, n + 1):
                if mask[r, c] or not any(mask[r + dr, c + dc] for dr, dc in _NEIGHBORS):
                    continue
                trial = mask.copy()
                trial[r, c] = True
                if _count_components(trial) == k and _count_components(~trial) == 1:
                    candidates.append((r, c))
        if not candidates:
            break
        mask[_choice(rng, candidates)] = True
    return mask


def _make_clue_mask(rng, size, count, cavities):
    if count == 0:
        return np.zeros((size, size), dtype=bool)
    if cavities.sum() == count and rng.random() < 0.22:
        return cavities.copy()
    return _isolated(rng, size, count)


def _free_origins(boxes, size):
    rr, cc = np.indices((31 - size, 31 - size))
    free = np.ones(rr.shape, dtype=bool)
    for r, c, h, w in boxes:
        free &= (rr + size < r) | (rr > r + h) | (cc + size < c) | (cc > c + w)
    return np.argwhere(free)


def _fallback_pack(tile_size, rows, cols):
    size, count = tile_size - 2, rows * cols
    boxes = [(0, 0, rows * tile_size, cols * tile_size)]
    for _ in range(count):
        free = _free_origins(boxes, size)
        if not len(free):
            return None
        r, c = map(int, free[0])
        boxes.append((r, c, size, size))
    return boxes


# Feasibility is geometric, not a split-specific catalog of count sets.
_CONFIGS = tuple((t, r, c) for t in range(6, 11)
                 for r, c in ((1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1), (2, 2), (2, 3), (3, 2))
                 if r * t <= 30 and c * t <= 30
                 and r * c <= _capacity(t) + 1
                 and _fallback_pack(t, r, c) is not None)


def _place_scene(rng, background, mosaic, output, motifs):
    h, w = mosaic.shape
    size = len(motifs[0])
    boxes = None
    prefer_interior = rng.random() < 0.40 and h <= 26 and w <= 26
    # Independent offsets in a full workspace, then random crop and margins.
    # Tight scenes have a bounded, proven feasible shelf fallback.
    for attempt in range(32):
        if prefer_interior and attempt < 24:
            r = int(rng.integers(2, 29 - h))
            c = int(rng.integers(2, 29 - w))
        else:
            r = _choice(rng, [0, 30 - h, int(rng.integers(31 - h))])
            c = _choice(rng, [0, 30 - w, int(rng.integers(31 - w))])
        trial = [(r, c, h, w)]
        for _ in motifs:
            free = _free_origins(trial, size)
            if not len(free):
                break
            rr, cc = map(int, free[int(rng.integers(len(free)))])
            trial.append((rr, cc, size, size))
        if len(trial) == len(motifs) + 1:
            boxes = trial
            break
    if boxes is None:
        tile_size = size + 2
        boxes = _fallback_pack(tile_size, h // tile_size, w // tile_size)
    assert boxes is not None
    canvas = np.full((30, 30), background, dtype=np.int8)
    objects = [mosaic] + [motifs[int(i)] for i in rng.permutation(len(motifs))]
    for (r, c, bh, bw), obj in zip(boxes, objects):
        canvas[r:r + bh, c:c + bw] = obj
    low_r = min(b[0] for b in boxes)
    low_c = min(b[1] for b in boxes)
    high_r = max(b[0] + b[2] for b in boxes)
    high_c = max(b[1] + b[3] for b in boxes)
    if rng.random() >= 0.22:
        low_r = _choice(rng, [low_r, int(rng.integers(low_r + 1))])
        low_c = _choice(rng, [low_c, int(rng.integers(low_c + 1))])
        high_r = _choice(rng, [high_r, int(rng.integers(high_r, 31))])
        high_c = _choice(rng, [high_c, int(rng.integers(high_c, 31))])
        if prefer_interior:
            r, c = boxes[0][:2]
            low_r, low_c = min(low_r, max(0, r - 2)), min(low_c, max(0, c - 2))
            high_r, high_c = max(high_r, min(30, r + h + 2)), max(high_c, min(30, c + w + 2))
        canvas = canvas[low_r:high_r, low_c:high_c]
    # No scene rotation: copied orientation is exactly what is visible in input.
    return canvas.copy(), output.copy()


def _generate_pair(rng, config, counts, background, witness=False):
    t, rows, cols = config
    s, count = t - 2, rows * cols
    order = list(map(int, rng.permutation(counts)))
    palette = list(map(int, rng.permutation([c for c in range(10) if c != background])))
    pool = palette[:int(rng.integers(2, 10))]
    repeat_carriers = rng.random() < 0.38 or len(pool) < count
    carriers = list(map(int, rng.choice(pool, count, replace=repeat_carriers)))
    motifs, masks = {}, {}
    for index, k in enumerate(order):
        mask = _make_cavity_mask(rng, s, k)
        if witness and k == 1:
            mask = np.pad(np.ones((s - 2, s - 2), dtype=bool), 1)
        if witness and k == 2:
            # Two four-components but one eight-component; no D4 symmetry.
            mask = np.zeros((s, s), dtype=bool)
            mask[1, 1] = mask[2, 2] = mask[2, 3] = True
            mask = _choice(rng, _transforms(mask)).copy()
            for _ in range(12):
                candidate = _make_cavity_mask(rng, s, 2)
                if (candidate.sum() < (s - 2) ** 2 and _asymmetric(candidate)
                        and len(_components(candidate, diagonal=True)) == 1):
                    mask = candidate
                    break
        color = _choice(rng, [c for c in pool if c != carriers[index]])
        motif = np.full((s, s), color, dtype=np.int8)
        motif[mask] = background
        motifs[k], masks[k] = motif, mask
    mosaic = np.empty((rows * t, cols * t), dtype=np.int8)
    output = np.empty_like(mosaic)
    for i, k in enumerate(order):
        r, c = (i // cols) * t, (i % cols) * t
        tile = np.full((t, t), carriers[i], dtype=np.int8)
        clues = _make_clue_mask(rng, s, k, masks[k])
        tile[1:-1, 1:-1][clues] = background
        mosaic[r:r + t, c:c + t] = tile
        tile[:] = carriers[i]
        tile[1:-1, 1:-1] = np.where(masks[k], carriers[i], motifs[k])
        output[r:r + t, c:c + t] = tile
    inp, out = _place_scene(rng, background, mosaic, output, list(motifs.values()))
    return {"input": inp, "output": out}


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _perimeter(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _decode_background(grid, background):
    # A full blank row/column gap separates every object's box. The foreground
    # and the cavities both use ordinary four-neighbor connectivity.
    components = sorted(_components(grid != background), key=len, reverse=True)
    assert len(components) >= 3
    boxes = [_box(cells) for cells in components]
    mr0, mr1, mc0, mc1 = boxes[0]
    shapes = {(r1 - r0, c1 - c0) for r0, r1, c0, c1 in boxes[1:]}
    assert len(shapes) == 1
    s, sw = next(iter(shapes))
    assert 4 <= s <= 8 and s == sw
    t = s + 2
    mosaic = grid[mr0:mr1, mc0:mc1]
    assert mosaic.shape[0] % t == mosaic.shape[1] % t == 0
    rows, cols = mosaic.shape[0] // t, mosaic.shape[1] // t
    n = rows * cols
    assert n == len(boxes) - 1 and n in (2, 3, 4, 6)
    for i, (a, b, c, d) in enumerate(boxes):
        for e, f, g, h in boxes[i + 1:]:
            assert b < e or f < a or d < g or h < c
    motifs, motif_colors, cavity_sizes = {}, [], []
    for r0, r1, c0, c1 in boxes[1:]:
        motif = grid[r0:r1, c0:c1]
        foregrounds = np.unique(motif[motif != background])
        assert len(foregrounds) == 1
        color = int(foregrounds[0])
        assert np.all(_perimeter(motif) == color)
        cavities = _components(motif == background)
        k = len(cavities)
        assert k not in motifs
        motifs[k] = motif.copy()
        motif_colors.append(color)
        cavity_sizes.append((k, tuple(sorted(map(len, cavities)))))
    counts, carriers, clues = [], [], []
    expected = np.empty_like(mosaic)
    for r in range(rows):
        for c in range(cols):
            tile = mosaic[r * t:(r + 1) * t, c * t:(c + 1) * t]
            carrier = int(tile[0, 0])
            assert carrier != background and np.all(_perimeter(tile) == carrier)
            assert set(map(int, np.unique(tile))) <= {background, carrier}
            clue = tile[1:-1, 1:-1] == background
            assert not np.any(clue[:-1] & clue[1:])
            assert not np.any(clue[:, :-1] & clue[:, 1:])
            k = int(clue.sum())
            assert k in motifs
            matched = motifs[k]
            assert int(matched[0, 0]) != carrier
            result = np.full((t, t), carrier, dtype=np.int8)
            result[1:-1, 1:-1] = np.where(matched == background, carrier, matched)
            expected[r * t:(r + 1) * t, c * t:(c + 1) * t] = result
            counts.append(k)
            carriers.append(carrier)
            clues.append(clue.copy())
    assert len(set(counts)) == n and set(counts) == set(motifs)
    patterns = [(k, motif == background) for k, motif in sorted(motifs.items())]
    areas = {k: int(mask.sum()) for k, mask in patterns}
    sides = set()
    for a, b, c, d in boxes[1:]:
        if b < mr0:
            sides.add('top')
        if a > mr1:
            sides.add('bottom')
        if d < mc0:
            sides.add('left')
        if c > mc1:
            sides.add('right')
    analysis = {
        'background': background, 'tile_size': t, 'tile_rows': rows, 'tile_cols': cols,
        'object_count': n, 'counts': tuple(sorted(counts)), 'carrier_colors': tuple(carriers),
        'motif_colors': tuple(motif_colors), 'motif_patterns': patterns, 'cavity_sizes': cavity_sizes,
        'area_inversion': any(a < b and areas[a] > areas[b] for a in areas for b in areas),
        'equal_area': any(a != b and areas[a] == areas[b] for a in areas for b in areas),
        'connectivity_witness': any(len(_components(mask, True)) < k for k, mask in patterns),
        'asymmetric': any(_asymmetric(mask) for k, mask in patterns),
        'symmetric_nonzero': sum(not _is_directional(mask) for k, mask in patterns if k),
        'nonpositional_count': sum(k > 0 and not np.array_equal(clue, motifs[k] == background)
                                   for k, clue in zip(counts, clues)),
        'coincidences': sum(k > 0 and np.array_equal(clue, motifs[k] == background)
                            for k, clue in zip(counts, clues)),
        'mixed_parity_clues': sum(len({int((r + c) % 2) for r, c in np.argwhere(clue)}) == 2
                                  for clue in clues),
        'mosaic_box': boxes[0], 'motif_boxes': boxes[1:], 'sides': tuple(sorted(sides)),
        'edge_mosaic': mr0 == 0 or mc0 == 0 or mr1 == grid.shape[0] or mc1 == grid.shape[1],
        'edge_motif': any(a == 0 or c == 0 or b == grid.shape[0] or d == grid.shape[1]
                          for a, b, c, d in boxes[1:]),
        'interior_mosaic': min(mr0, mc0, grid.shape[0] - mr1, grid.shape[1] - mc1) >= 2,
        'palette_size': len(np.unique(grid)), 'shape': grid.shape,
    }
    return expected, analysis


def _decode_input(grid):
    candidates = []
    for background in map(int, np.unique(grid)):
        try:
            candidates.append(_decode_background(grid, background))
        except (AssertionError, IndexError, ValueError):
            continue
    assert len(candidates) == 1, 'input must have exactly one background and mosaic interpretation'
    return candidates[0]


def _analyze_pair(input_grid, output_grid):
    expected, analysis = _decode_input(input_grid)
    assert np.array_equal(expected, output_grid), 'incorrect motif transplant or mosaic crop'
    return analysis


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _source_copy(pair):
    inp, out = pair['input'], pair['output']
    return (inp.shape, hashlib.sha256(np.asarray(inp, dtype=np.int8).tobytes()).hexdigest(),
            out.shape, hashlib.sha256(np.asarray(out, dtype=np.int8).tobytes()).hexdigest()) in _SOURCE_PAIRS


def _fresh_recolor(pair, used):
    # At least background and one foreground are present. Specifying their two
    # images yields 90 distinct inputs, more than the 13 previous and 3 source
    # inputs combined. This makes the duplicate fallback finite and guaranteed.
    present = list(map(int, np.unique(pair['input'])))
    order = present[:2] + [c for c in range(10) if c not in present[:2]]
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            target = [a, b] + [c for c in range(10) if c not in (a, b)]
            lookup = np.empty(10, dtype=np.int8)
            lookup[order] = target
            recolored = {k: lookup[v] for k, v in pair.items()}
            if _grid_key(recolored['input']) not in used and not _source_copy(recolored):
                return recolored
    raise AssertionError('color-permutation uniqueness bound violated')


def _training_coverage(analyses):
    assert any(a['area_inversion'] for a in analyses), 'training permits area-rank matching'
    assert any(any(size > 1 for k, sizes in a['cavity_sizes'] for size in sizes) for a in analyses)
    assert any(a['connectivity_witness'] for a in analyses), 'four/eight connectivity not distinguished'
    assert any(a['asymmetric'] for a in analyses), 'orientation not fully demonstrated'
    assert any(a['nonpositional_count'] for a in analyses), 'clue geometry not refuted'


def _validate_generated_puzzle_impl(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    keys = set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert not _source_copy(pair)
            assert not np.array_equal(pair['input'], pair['output'])
            key = _grid_key(pair['input'])
            assert key not in keys
            keys.add(key)
            analyses[split].append(_analyze_pair(pair['input'], pair['output']))
    _training_coverage(analyses['train'])
    # All scenes share the same count rule. Background identity, palette, scale,
    # layout and zero presence are visible, and impose no hidden conventions.
    return analyses


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError('malformed generated puzzle') from exc


def predict(train_pairs, test_input):
    """Prediction uses demonstrations and input only, never a label or seed."""
    analyses = [_analyze_pair(p['input'], p['output']) for p in train_pairs]
    _training_coverage(analyses)
    return _decode_input(test_input)[0]


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(name + ' has an invalid count')
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1 if num_train is not None else 5))
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _select_counts(rng, config, include_zero, witness):
    t, r, c = config
    cap = _capacity(t)
    required = ([0] if include_zero else []) + ([1, 2] if witness else [])
    allowed = [k for k in range(1, cap + 1) if k not in required]
    remaining = r * c - len(required)
    if rng.random() < 0.35 and not witness:
        # Familiar consecutive count ranges coexist with arbitrary subsets.
        start = 0 if include_zero else int(rng.integers(1, cap - r * c + 2))
        return list(range(start, start + r * c))
    return sorted(required + list(map(int, rng.choice(allowed, remaining, replace=False))))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    nt, ne = _resolve_counts(rng, num_train, num_test)
    # Curriculum policies are sampled once. They never encode an output map.
    zero_policy = _choice(rng, ('unseen', 'both', 'none', 'train_only'))
    relationship = _choice(rng, ('growth', 'familiar', 'smaller', 'new_layout'))
    orientation = _choice(rng, ('mixed', 'mixed', 'horizontal', 'vertical'))
    varied_background = rng.random() < 0.85
    base_background = int(rng.integers(10))
    puzzle = {'train': [], 'test': []}
    used = set()
    train_configs, train_counts = [], []
    previous_background = base_background
    for split, number, regime in (('train', nt, 'source_train'), ('test', ne, 'target_test')):
        for i in range(number):
            witness = regime == 'source_train' and i == 0
            include_zero = zero_policy in (('both', 'train_only') if split == 'train' else ('both', 'unseen'))
            if include_zero and i > 0:
                include_zero = rng.random() < 0.70
            eligible = [conf for conf in _CONFIGS
                        if conf[1] * conf[2] <= _capacity(conf[0]) + int(include_zero)
                        and (not witness or (conf[0] >= 7 and conf[1] * conf[2] >= 2 + int(include_zero)))
                        and (orientation == 'mixed' or (conf[1] <= conf[2] if orientation == 'horizontal' else conf[1] >= conf[2]))]
            if split == 'train':
                if relationship == 'growth':
                    eligible = [conf for conf in eligible if conf[1] * conf[2] <= 4]
                if relationship == 'smaller':
                    eligible = [conf for conf in eligible if conf[1] * conf[2] >= 3]
                if i == 1:
                    varied = [conf for conf in eligible if conf[0] != train_configs[0][0]
                              and conf[1:] != train_configs[0][1:]]
                    if varied:
                        eligible = varied
            elif relationship == 'growth':
                eligible = [conf for conf in eligible if conf[1] * conf[2] == 6]
            elif relationship == 'smaller':
                eligible = [conf for conf in eligible if conf[1] * conf[2] == 2]
            elif relationship == 'familiar':
                shared = [conf for conf in eligible if conf in train_configs]
                if shared:
                    eligible = shared
            else:
                novel = [conf for conf in eligible if conf[1:] not in [x[1:] for x in train_configs]]
                if novel:
                    eligible = novel
            config = _choice(rng, eligible)
            counts = _select_counts(rng, config, include_zero, witness)
            if split == 'test' and relationship == 'familiar':
                shared = [ks for conf, ks in zip(train_configs, train_counts)
                          if conf == config and (0 in ks) == include_zero]
                if shared:
                    counts = list(_choice(rng, shared))
            background = (_choice(rng, [c for c in range(10) if c != previous_background])
                          if varied_background else base_background)
            previous_background = background
            for _ in range(16):
                pair = _generate_pair(rng, config, counts, background, witness)
                key = _grid_key(pair['input'])
                if key not in used and not _source_copy(pair):
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                pair = _fresh_recolor(pair, used)
                used.add(_grid_key(pair['input']))
                puzzle[split].append(pair)
            if split == 'train':
                train_configs.append(config)
                train_counts.append(counts)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def measure_puzzles(puzzles):
    """Review metrics reconstructed from returned grids, not sampler metadata."""
    hist = {name: {} for name in ('canvas', 'layout', 'layout_counts', 'counts', 'tile_size',
                                  'palette', 'cavity_area', 'cavity_sizes', 'zero_regime',
                                  'object_shift', 'placement_sides', 'train_count', 'test_count',
                                  'palette_train', 'palette_test', 'layout_counts_train', 'layout_counts_test')}
    totals = {name: 0 for name in ('pairs', 'motifs', 'cavities', 'rings', 'long_cavities',
                                   'irregular_cavities', 'unequal_cavities', 'substantial_cavities',
                                   'equal_area_pairs', 'symmetry_mixed_pairs', 'symmetric_nonzero',
                                   'asymmetric_motifs', 'mixed_parity_clues', 'coincidences',
                                   'nonpositional_tiles', 'nonzero_tiles', 'repeated_carriers',
                                   'repeated_motif_colors', 'cross_role_colors', 'nonconsecutive',
                                   'edge_mosaic', 'edge_motif', 'interior_mosaic', 'combined_pairs',
                                   'adjacent_repeated_carriers', 'uneven_motif_offsets', 'three_tile_strips',
                                   'six_without_zero', 'full_size_canvas')}
    within = []
    shapes = set()
    def add(name, value):
        key = str(value)
        hist[name][key] = hist[name].get(key, 0) + 1
    for puzzle in puzzles:
        analyses = _validate_generated_puzzle_impl(puzzle)
        all_a = analyses['train'] + analyses['test']
        add('train_count', len(puzzle['train']))
        add('test_count', len(puzzle['test']))
        z = tuple(any(0 in a['counts'] for a in analyses[s]) for s in ('train', 'test'))
        add('zero_regime', z)
        tr = max(a['object_count'] for a in analyses['train'])
        te = max(a['object_count'] for a in analyses['test'])
        add('object_shift', 'larger' if te > tr else 'smaller' if te < tr else 'same')
        row = {name: len({a[name] for a in all_a}) for name in
               ('background', 'tile_size', 'counts', 'palette_size', 'object_count', 'shape')}
        row['layouts'] = len({(a['tile_rows'], a['tile_cols']) for a in all_a})
        row['training_area_inversion'] = any(a['area_inversion'] for a in analyses['train'])
        row['training_connectivity'] = any(a['connectivity_witness'] for a in analyses['train'])
        row['training_orientation'] = any(a['asymmetric'] for a in analyses['train'])
        row['single_orientation'] = (all(a['tile_rows'] <= a['tile_cols'] for a in all_a)
                                     or all(a['tile_rows'] >= a['tile_cols'] for a in all_a))
        orientations = {('h' if a['tile_rows'] == 1 else 'v') for a in all_a
                        if 1 in (a['tile_rows'], a['tile_cols'])}
        row['mixed_strip_orientations'] = orientations == {'h', 'v'}
        train_layouts = {(a['tile_rows'], a['tile_cols']) for a in analyses['train']}
        row['test_new_layout'] = any((a['tile_rows'], a['tile_cols']) not in train_layouts for a in analyses['test'])
        row['test_familiar_size_counts'] = any((a['tile_size'], a['counts']) in
                [(b['tile_size'], b['counts']) for b in analyses['train']] for a in analyses['test'])
        row['unseen_zero_growth'] = not z[0] and z[1] and te > tr
        for split in ('train', 'test'):
            for a in analyses[split]:
                add('palette_' + split, a['palette_size'])
                add('layout_counts_' + split, (a['tile_size'], a['tile_rows'], a['tile_cols'], a['counts']))
        within.append(row)
        for a in all_a:
            totals['pairs'] += 1
            layout = (a['tile_size'], a['tile_rows'], a['tile_cols'])
            for name, value in (('canvas', a['shape']), ('layout', layout),
                                ('layout_counts', (layout, a['counts'])), ('counts', a['counts']),
                                ('tile_size', a['tile_size']), ('palette', a['palette_size']),
                                ('placement_sides', a['sides'])):
                add(name, value)
            asym = 0
            for k, mask in a['motif_patterns']:
                totals['motifs'] += 1
                totals['nonzero_tiles'] += k > 0
                shapes.add((mask.shape, min(m.tobytes() for m in _transforms(mask))))
                asym += _asymmetric(mask)
                totals['rings'] += k == 1 and bool(np.all(mask[1:-1, 1:-1]))
                components = _components(mask)
                sizes = sorted(map(len, components))
                add('cavity_sizes', tuple(sizes))
                totals['unequal_cavities'] += len(set(sizes)) > 1
                totals['substantial_cavities'] += sum(n >= 3 for n in sizes) >= 2
                for cells in components:
                    totals['cavities'] += 1
                    add('cavity_area', len(cells))
                    r0, r1, c0, c1 = _box(cells)
                    totals['long_cavities'] += min(r1 - r0, c1 - c0) == 1 and len(cells) >= 3
                    totals['irregular_cavities'] += len(cells) < (r1 - r0) * (c1 - c0)
            totals['asymmetric_motifs'] += asym
            totals['symmetry_mixed_pairs'] += a['symmetric_nonzero'] > 0 and asym > 0
            totals['equal_area_pairs'] += a['equal_area']
            for name in ('symmetric_nonzero', 'mixed_parity_clues', 'coincidences', 'edge_mosaic',
                         'edge_motif', 'interior_mosaic'):
                totals[name] += int(a[name])
            totals['nonpositional_tiles'] += a['nonpositional_count']
            repeated = len(set(a['carrier_colors'])) < a['object_count']
            totals['repeated_carriers'] += repeated
            colors = np.array(a['carrier_colors']).reshape(a['tile_rows'], a['tile_cols'])
            totals['adjacent_repeated_carriers'] += int(np.any(colors[:-1] == colors[1:])
                                                       or np.any(colors[:, :-1] == colors[:, 1:]))
            offsets = [sorted({b[i] for b in a['motif_boxes']}) for i in (0, 2)]
            totals['uneven_motif_offsets'] += any(len(v) >= 3 and len(set(np.diff(v))) > 1 for v in offsets)
            totals['three_tile_strips'] += a['object_count'] == 3
            totals['six_without_zero'] += a['object_count'] == 6 and 0 not in a['counts']
            totals['full_size_canvas'] += a['shape'] == (30, 30)
            totals['repeated_motif_colors'] += len(set(a['motif_colors'])) < a['object_count']
            totals['cross_role_colors'] += bool(set(a['carrier_colors']) & set(a['motif_colors']))
            nonconsecutive = max(a['counts']) - min(a['counts']) + 1 > a['object_count']
            totals['nonconsecutive'] += nonconsecutive
            totals['combined_pairs'] += (nonconsecutive and repeated and a['mixed_parity_clues'] > 0
                                         and any(n > 4 for k, sizes in a['cavity_sizes'] for n in sizes))
    return {'histograms': hist, 'totals': totals, 'within_puzzle': within,
            'dihedral_cavity_shapes': len(shapes)}


def _area_rank_prediction(inp):
    """The precise competing rule from the seed-2523 review."""
    _, a = _decode_input(inp)
    bg, t = a['background'], a['tile_size']
    r0, r1, c0, c1 = a['mosaic_box']
    mosaic = inp[r0:r1, c0:c1]
    motifs = [inp[r:s, c:d] for r, s, c, d in a['motif_boxes']]
    areas = [int((m == bg).sum()) for m in motifs]
    if len(set(areas)) != len(areas):
        return None
    assigned = dict(zip(a['counts'], [motifs[int(i)] for i in np.argsort(areas)]))
    result = mosaic.copy()
    for r in range(a['tile_rows']):
        for c in range(a['tile_cols']):
            tile = mosaic[r*t:(r+1)*t, c*t:(c+1)*t]
            k = int((tile == bg).sum())
            result[r*t:(r+1)*t, c*t:(c+1)*t] = tile[0, 0]
            result[r*t+1:(r+1)*t-1, c*t+1:(c+1)*t-1] = np.where(assigned[k] == bg, tile[0, 0], assigned[k])
    return result


def _source_shape_tests():
    # Exact masks transcribed from the source; seeds only exercise the public
    # shape sampler here, never prediction. These are tests, not a template bank.
    cases = (
        (1, ('0000', '0110', '0110', '0000'), (4,)),
        (1, ('000000', '011110', '011110', '011110', '011110', '000000'), (16,)),
        (31, ('000000', '011010', '011010', '000010', '011010', '000000'), (2, 4, 4)),
        (21, ('000000', '011010', '011010', '000000', '011010', '000000'), (1, 2, 2, 4)),
        (1, ('00000', '01110', '01110', '01110', '00000'), (9,)),
        (11, ('00000', '01010', '01010', '01010', '00000'), (3, 3)),
        (1, ('00000', '01010', '00000', '01010', '00000'), (1, 1, 1, 1)),
    )
    for seed, rows, sizes in cases:
        mask = np.array([[int(x) for x in row] for row in rows], dtype=bool)
        assert tuple(sorted(map(len, _components(mask)))) == sizes
        actual = _make_cavity_mask(np.random.default_rng(seed), len(mask), len(sizes))
        assert any(np.array_equal(actual, variant) for variant in _transforms(mask))
    small, large = set(), set()
    for seed in range(42):
        for s, collection in ((5, small), (6, large)):
            mask = _make_cavity_mask(np.random.default_rng(seed), s, 5)
            assert _count_components(mask) == 5
            collection.add(min(m.tobytes() for m in _transforms(mask)))
    assert len(small) == 1 and len(large) >= 10
    rng = np.random.default_rng(8)
    pair = _generate_pair(rng, (7, 1, 3), [0, 2, 4], 8)
    assert _analyze_pair(pair['input'], pair['output'])['counts'] == (0, 2, 4)


def self_test() -> None:
    _source_shape_tests()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = measure_puzzles(puzzles)
    hist, totals = metrics['histograms'], metrics['totals']
    assert len(hist['train_count']) >= 4 and len(hist['test_count']) >= 3
    assert len(hist['canvas']) >= 35 and len(hist['layout']) >= 20
    assert len(hist['layout_counts']) >= 65 and len(hist['counts']) >= 25
    assert len(hist['tile_size']) == 5 and len(hist['palette']) >= 6
    assert len(hist['palette_train']) >= 6 and len(hist['palette_test']) >= 5
    assert len(hist['layout_counts_train']) >= 60 and len(hist['layout_counts_test']) >= 25
    assert len(hist['zero_regime']) == 4 and len(hist['object_shift']) == 3
    assert metrics['dihedral_cavity_shapes'] >= 100
    assert max(map(int, hist['cavity_area'])) >= 16
    for name in ('rings', 'long_cavities', 'irregular_cavities', 'unequal_cavities',
                 'substantial_cavities', 'equal_area_pairs', 'symmetry_mixed_pairs',
                 'coincidences', 'repeated_carriers', 'mixed_parity_clues',
                 'interior_mosaic', 'edge_motif', 'edge_mosaic', 'combined_pairs',
                 'adjacent_repeated_carriers', 'uneven_motif_offsets', 'three_tile_strips',
                 'six_without_zero', 'full_size_canvas'):
        assert totals[name] >= 3, (name, totals[name])
    assert totals['nonpositional_tiles'] > 0.7 * totals['nonzero_tiles']
    assert sum(row['background'] > 1 for row in metrics['within_puzzle']) >= 25
    assert sum(row['layouts'] > 1 and row['tile_size'] > 1 for row in metrics['within_puzzle']) >= 35
    assert sum(row['single_orientation'] for row in metrics['within_puzzle']) >= 5
    for name in ('mixed_strip_orientations', 'test_new_layout', 'test_familiar_size_counts', 'unseen_zero_growth'):
        assert sum(row[name] for row in metrics['within_puzzle']) >= 3, name
    assert any(str((side,)) in hist['placement_sides'] for side in ('left', 'right', 'top', 'bottom'))
    assert str(('left', 'right')) in hist['placement_sides'] or str(('bottom', 'top')) in hist['placement_sides']
    # Predictor receives copies of demonstrations and only the test input.
    for puzzle in puzzles:
        train = [{k: grid.copy() for k, grid in pair.items()} for pair in puzzle['train']]
        for pair in puzzle['test']:
            predicted = predict(train, pair['input'].copy())
            assert np.array_equal(predicted, pair['output'])
    assert _puzzles_equal(generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    for nt, ne in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2523, nt, ne)
        validate_generated_puzzle(puzzle)
        assert nt is None or len(puzzle['train']) == nt
        assert ne is None or len(puzzle['test']) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count request accepted')
    # Regression: even minimum training must reject area-rank matching.
    for seed in (2523, 0, 1, 123):
        puzzle = generate_puzzle(seed, 2, 1)
        assert any(_area_rank_prediction(p['input']) is None
                   or not np.array_equal(_area_rank_prediction(p['input']), p['output'])
                   for p in puzzle['train'])
        _training_coverage([_analyze_pair(p['input'], p['output']) for p in puzzle['train']])
    # Corrupted substitutions and missing training witnesses must fail.
    sample = generate_puzzle(2523, 2, 1)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
    broken['test'][0]['output'][1, 1] = (int(broken['test'][0]['output'][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted transplant accepted')
    for p in broken['train']:
        _, a = _decode_input(p['input'])
        bg = a['background']
        for r0, r1, c0, c1 in a['motif_boxes']:
            motif = p['input'][r0:r1, c0:c1]
            k = _count_components(motif == bg)
            color = int(motif[0, 0])
            motif[:] = color
            if k:
                motif[1:-1, 1:-1][_isolated(np.random.default_rng(k), len(motif) - 2, k)] = bg
        p['output'] = _decode_input(p['input'])[0]
    broken['test'] = sample['test']
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('all-single-cell training accepted')
