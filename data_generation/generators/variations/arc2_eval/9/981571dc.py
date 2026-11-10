"""Repair one puzzle-wide erasure color using two input-visible mirror folds."""
from functools import lru_cache
from hashlib import sha256
import numpy as np

# Exact int8 source input+output fingerprints; 30x30 itself remains legal.
_SOURCE_HASHES = frozenset((
    'd88279b117d9667110079a24c51079e0bf10262370e3839a198f1ddf11c8d552',
    '81a50c704316d1dafb0305b930fb085e2d286ca8499f6aba06afa92217b8af36',
    '9bc3a2d7834f4bbb1d490f581c7d580924194d26b23050c272a67ab2765d3963',
    'f90961c34a200812d51e7a201b60c1c51ad47caae986d4e8d06c8740dd0f5975',
    'ac1ef6736991198b496aa7caf4f3adf77d4aed26db5044323022946e1dbca412',
))


def _integer_argument(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f'{name} must be between {lower} and {upper}')
    return value


def _choose_counts(seed_value, rng, num_train, num_test):
    nt = _integer_argument('num_train', num_train, 2, 10)
    ns = _integer_argument('num_test', num_test, 1, 4)
    if nt is not None and ns is not None:
        if nt <= ns:
            raise ValueError('num_train must be greater than num_test')
    elif nt is None and ns is None:
        ns = 1 + seed_value % 4
        nt = ns + 1 + (seed_value // 4) % (10 - ns)
    elif nt is None:
        nt = int(rng.integers(ns + 1, 11))
    else:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    return nt, ns


def _orbit(row, col, height, width, row_sum, col_sum):
    # Outside one fold's overlap, partners under the other fold still count.
    rr, cc = row_sum - row, col_sum - col
    rows = {row, rr} if 0 <= rr < height else {row}
    cols = {col, cc} if 0 <= cc < width else {col}
    return tuple(sorted((r, c) for r in rows for c in cols))


@lru_cache(maxsize=512)
def _all_orbits(height, width, row_sum, col_sum):
    seen, result = set(), []
    for row in range(height):
        for col in range(width):
            if (row, col) not in seen:
                orbit = _orbit(row, col, height, width, row_sum, col_sum)
                seen.update(orbit)
                result.append(orbit)
    return tuple(result)


@lru_cache(maxsize=64)
def _axis_indices(length):
    result = []
    for fold in range(1, 2 * length - 2):
        left = np.array([x for x in range(length) if x < fold - x < length])
        result.append((fold, left, fold - left))
    return tuple(result)


def _axis_candidates(grid, erase, dimension):
    a = grid if dimension == 0 else grid.T
    candidates = []
    for fold, left, right in _axis_indices(a.shape[0]):
        first, second = a[left], a[right]
        visible = (first != erase) & (second != erase)
        if np.count_nonzero(visible) < max(4, a.shape[1] // 2):
            continue
        if np.all(first[visible] == second[visible]):
            candidates.append(fold)
    return candidates


def _infer_axes(grid, erase):
    rows = _axis_candidates(grid, erase, 0)
    cols = _axis_candidates(grid, erase, 1)
    assert len(rows) == len(cols) == 1, 'visible evidence does not identify unique folds'
    return rows[0], cols[0]


def _restore(grid, erase):
    axes = _infer_axes(grid, erase)
    result = grid.copy()
    for orbit in _all_orbits(*grid.shape, *axes):
        witnesses = {int(grid[p]) for p in orbit if grid[p] != erase}
        assert len(witnesses) == 1, 'empty or contradictory reflection orbit'
        color = next(iter(witnesses))
        for point in orbit:
            if result[point] == erase:
                result[point] = color
    return result


def _learn_marker(train_pairs):
    markers = set()
    for pair in train_pairs:
        changed = pair['input'] != pair['output']
        assert changed.any(), 'a demonstration must reveal the erasure role'
        values = np.unique(pair['input'][changed])
        assert len(values) == 1
        markers.add(int(values[0]))
    assert len(markers) == 1, 'inconsistent erasure role'
    return next(iter(markers))


def solve(train_pairs, test_input):
    """No test label, seed or generation metadata enters prediction."""
    return _restore(test_input, _learn_marker(train_pairs))


def _sample_shape(rng):
    mode = int(rng.integers(10))
    if mode == 0:
        return 30, 30
    low, high = (8, 15) if mode < 4 else ((23, 31) if mode < 7 else (8, 31))
    height = int(rng.integers(low, high))
    width = height if rng.random() < 0.30 else int(rng.integers(low, high))
    return height, width


def _sample_geometry(rng, shape=None):
    height, width = _sample_shape(rng) if shape is None else shape
    offsets = []
    for size in (height, width):
        limit = min(6, size // 3)
        offset = 0 if rng.random() < 0.38 else int(rng.integers(1, limit + 1))
        offsets.append(offset if rng.random() < 0.5 else -offset)
    return height, width, height - 1 + offsets[0], width - 1 + offsets[1]


def _rotate_geometry(geometry, turns):
    h, w, a, b = geometry
    for _ in range(turns):
        h, w, a, b = w, h, 2 * (w - 1) - b, a
    return h, w, a, b


def _draw_field(rng, geometry, colors, style, noise):
    h, w, a, b = geometry
    orbits = _all_orbits(h, w, a, b)
    r, c = np.array([orbit[0] for orbit in orbits]).T
    k = len(colors)
    bh, bw = (int(x) for x in rng.integers(2, 6, size=2))
    shift = int(rng.integers(k))
    weights = rng.dirichlet(np.full(k, 1.5))
    if style == 0:
        values = rng.choice(k, len(orbits), p=weights)
    elif style == 1:
        coarse = rng.integers(k, size=(h, w))
        values = coarse[r // bh, c // bw]
    elif style == 2:  # Stepped bands.
        values = (r // bh + (c // bw) // 2 + shift) % k
    elif style == 3:  # Diagonal strokes.
        values = ((int(rng.choice([-2, -1, 1, 2])) * r + c) // bw + shift) % k
    elif style == 4:  # Nested outlines in a fragment, then reflected.
        cr, cc = int(rng.integers(h)), int(rng.integers(w))
        values = (np.maximum(abs(r - cr), abs(c - cc)) // bh + shift) % k
    else:  # Larger uniform areas with asymmetric motif fragments.
        values = np.full(len(orbits), shift)
        for _ in range(int(rng.integers(2, 7))):
            r0, c0 = int(rng.integers(h)), int(rng.integers(w))
            rh, cw = int(rng.integers(2, h // 2 + 2)), int(rng.integers(2, w // 2 + 2))
            shape = int(rng.integers(3))
            if shape == 0:
                region = (r >= r0) & (r < r0 + rh) & (c >= c0) & (c < c0 + cw)
            elif shape == 1:
                region = abs((r - r0) - (c - c0)) <= int(rng.integers(1, 3))
            else:
                region = (r >= r0) & (r < r0 + rh) & (c >= c0) & (c - c0 <= (r - r0) // 2 + cw)
            values[region] = int(rng.integers(k))
    speckles = rng.random(len(orbits)) < noise
    values[speckles] = rng.integers(k, size=int(speckles.sum()))
    values[rng.permutation(len(orbits))[:k]] = np.arange(k)
    output = np.empty((h, w), dtype=np.int8)
    for value, orbit in zip(values, orbits):
        for point in orbit:
            output[point] = colors[int(value)]
    return output


def _place_patch(rng, mask, rh, cw, interior, kind):
    h, w = mask.shape
    margin = int(interior)
    rh, cw = min(rh, h - 2 * margin), min(cw, w - 2 * margin)
    r0 = int(rng.integers(margin, h - rh - margin + 1))
    c0 = int(rng.integers(margin, w - cw - margin + 1))
    if not interior:
        if rng.random() < 0.4:
            r0 = 0 if rng.random() < 0.5 else h - rh
        if rng.random() < 0.4:
            c0 = 0 if rng.random() < 0.5 else w - cw
    if kind == 0:
        mask[r0:r0 + rh, c0:c0 + cw] = True
    elif kind == 1:
        reverse = rng.random() < 0.5
        for dr in range(rh):
            reach = 1 + (dr * max(1, cw - 1)) // max(1, rh - 1)
            start = c0 + cw - reach if reverse else c0
            mask[r0 + dr, start:start + reach] = True
    else:
        row, col = r0, c0
        for _ in range(rh * cw):
            mask[row, col] = True
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            row = int(np.clip(row + dr, margin, h - margin - 1))
            col = int(np.clip(col + dc, margin, w - margin - 1))


def _raw_mask(rng, geometry, family, interior):
    h, w, a, b = geometry
    mask = np.zeros((h, w), dtype=bool)
    if family == 5:
        mask = rng.random((h, w)) < rng.uniform(0.015, 0.13)
        if interior:
            mask[[0, -1], :] = False
            mask[:, [0, -1]] = False
    elif family == 7:  # Source strip-plus-corner and other corner contacts.
        rh = int(rng.integers(4, min(6, h - 3) + 1))
        left = max(1, b - w + 1)
        right = min(w - 2, (b - 1) // 2)
        cw = int(rng.integers(1, min(3, right - left + 1) + 1))
        r0 = int(np.clip(a // 2 - rh // 2 + 1, 1, h - rh - 1))
        c0 = int(rng.integers(left, right - cw + 2))
        mask[r0:r0 + rh, c0:c0 + cw] = True
        bh = int(rng.integers(2, min(4, h // 3) + 1))
        bw = int(rng.integers(3, min(6, max(3, w // 3)) + 1))
        br = h - bh if rng.random() < 0.8 else 0
        bc = w - bw if rng.random() < 0.8 else 0
        mask[br:br + bh, bc:bc + bw] = True
    else:
        count = 1 if family in (0, 2, 3, 4) else int(rng.integers(2, 6))
        for _ in range(count):
            kind = 0 if family < 4 else (1 if family == 4 else int(rng.integers(3)))
            rh = int(rng.integers(1, max(3, h // 3) + 1))
            cw = int(rng.integers(1, max(3, w // 3) + 1))
            if family == 2:
                rh, cw = int(rng.integers(1, 3)), int(rng.integers(3, max(4, w // 2 + 1)))
            elif family == 3:
                rh, cw = int(rng.integers(3, max(4, h // 2 + 1))), int(rng.integers(1, 3))
            _place_patch(rng, mask, rh, cw, interior, kind)
    return mask


def _make_mask(rng, geometry, family, interior, difficulty):
    mask = _raw_mask(rng, geometry, family, interior)
    h, w, a, b = geometry
    orbits = _all_orbits(h, w, a, b)
    four = [o for o in orbits if len(o) == 4 and
            (not interior or all(0 < r < h - 1 and 0 < c < w - 1 for r, c in o))]
    protected = np.zeros_like(mask)
    if difficulty:
        chosen = rng.permutation(len(four))[:int(rng.integers(2, min(5, len(four)) + 1))]
        for index, selected in enumerate(chosen):
            orbit = four[int(selected)]
            survivor = orbit[int(rng.integers(4))]
            for point in orbit:
                mask[point] = False
            if difficulty == 2 and index == 0:
                points = [p for p in orbit if p != survivor]
            elif index % 2:
                points = [p for p in orbit if p[0] != survivor[0]]
            else:
                points = [p for p in orbit if p[1] != survivor[1]]
            for point in points:
                mask[point] = protected[point] = True
    for orbit in orbits:
        if all(mask[p] for p in orbit):
            mask[orbit[int(rng.integers(len(orbit)))]] = False
    return mask, protected


def _visible_mask(output, erase, mask, protected, geometry):
    """Expose clean witnesses for false folds without undoing hard erasures."""
    inp = output.copy()
    inp[mask] = erase
    for dimension, true_axis in enumerate(geometry[2:]):
        for false_axis in _axis_candidates(inp, erase, dimension):
            if false_axis == true_axis:
                continue
            candidates = []
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    q = (false_axis - r, c) if dimension == 0 else (r, false_axis - c)
                    if (0 <= q[0] < inp.shape[0] and 0 <= q[1] < inp.shape[1]
                            and not protected[r, c] and not protected[q]
                            and output[r, c] != output[q]):
                        candidates.append(((r, c), q))
            if not candidates:
                return None
            p, q = min(candidates, key=lambda pq: int(mask[pq[0]]) + int(mask[pq[1]]))
            inp[p], inp[q] = output[p], output[q]
    if not np.any(inp == erase):
        return None
    if any(_axis_candidates(inp, erase, d) != [geometry[d + 2]] for d in (0, 1)):
        return None
    return inp


def _fallback(colors, erase):
    # Deterministic termination guard; never a normal texture or mask family.
    fragment = np.array([[0, 0, 1, 0], [1, 0, 1, 1],
                         [0, 1, 1, 0], [1, 1, 0, 1]], dtype=np.int8)
    output = np.block([[fragment, fragment[:, ::-1]],
                       [fragment[::-1], fragment[::-1, ::-1]]])
    output = np.asarray(colors, dtype=np.int8)[output]
    inp = output.copy()
    inp[1, 2] = inp[6, 2] = inp[1, 5] = erase
    assert _infer_axes(inp, erase) == (7, 7)
    return {'input': inp, 'output': output}


def _generate_pair(rng, geometry, colors, erase, regime, must_demonstrate=False):
    assert regime in ('source_train', 'target_test')
    style, family = int(rng.integers(6)), int(rng.integers(8))
    noise = float(rng.uniform(0.04, 0.22))
    interior = bool(rng.random() < 0.34 and family != 7)
    hard_probability = 0.68 if regime == 'target_test' else 0.32
    difficulty = 2 if must_demonstrate else (
        int(rng.integers(1, 3)) if rng.random() < hard_probability else 0)
    if family == 7 and not must_demonstrate and rng.random() < 0.8:
        difficulty = 0
    for attempt in range(64):
        # Random texture is a fallback for uninformative, small two-color motifs.
        output = _draw_field(rng, geometry, colors, style if attempt < 24 else 0, noise)
        if any(_axis_candidates(output, erase, d) != [geometry[d + 2]] for d in (0, 1)):
            continue
        mask, protected = _make_mask(rng, geometry, family, interior, difficulty)
        inp = _visible_mask(output, erase, mask, protected, geometry)
        if inp is not None:
            return {'input': inp, 'output': output}
    return _fallback(colors, erase)


def _array_signature(array):
    return array.shape, array.astype(np.int8).tobytes()


def _is_source(pair):
    return pair['input'].shape == pair['output'].shape == (30, 30) and sha256(
        pair['input'].astype(np.int8).tobytes() + pair['output'].astype(np.int8).tobytes()
    ).hexdigest() in _SOURCE_HASHES


def _puzzle_signature(puzzle):
    return tuple(tuple((_array_signature(p['input']), _array_signature(p['output']))
                       for p in puzzle[split]) for split in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed_value = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_value)
    nt, ns = _choose_counts(seed_value, rng, num_train, num_test)
    erase = int(rng.integers(10))
    payload = [c for c in range(10) if c != erase]
    # Optional unseen test payloads have surviving witnesses within that test.
    held_out = list(rng.choice(payload, size=int(rng.integers(1, 3)), replace=False)) if rng.random() < 0.38 else []
    training_pool = [c for c in payload if c not in held_out]
    regime = int(rng.choice(5, p=[0.22, 0.16, 0.42, 0.10, 0.10]))
    base = _sample_geometry(rng)
    if rng.random() < 0.13:
        base = (30, 30, 31, 31)
    demonstration = int(rng.integers(nt))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            if regime == 0:  # Shared geometry, unfamiliar masks and payload.
                geometry = base
            elif regime == 1:  # Shared offsets, variable dimensions.
                h, w = _sample_shape(rng)
                dr, dc = base[2] - base[0] + 1, base[3] - base[1] + 1
                h, w = max(h, abs(dr) + 6), max(w, abs(dc) + 6)
                geometry = h, w, h - 1 + dr, w - 1 + dc
            elif regime == 3:  # Held-out rotation with folds visible in the input.
                geometry = base if split == 'train' else _rotate_geometry(base, int(rng.integers(1, 4)))
            elif regime == 4:  # Larger test canvas.
                lo, hi = (23, 31) if split == 'test' else (8, 19)
                geometry = _sample_geometry(rng, tuple(int(x) for x in rng.integers(lo, hi, size=2)))
            else:
                geometry = _sample_geometry(rng)
            pool = training_pool if split == 'train' else payload
            k = int(rng.integers(2, len(pool) + 1))
            colors = list(rng.choice(pool, size=k, replace=False))
            if split == 'test' and held_out and not set(colors) & set(held_out):
                colors[int(rng.integers(k))] = int(rng.choice(held_out))
            for _ in range(32):
                pair = _generate_pair(rng, geometry, colors, erase,
                                      'source_train' if split == 'train' else 'target_test',
                                      split == 'train' and index == demonstration)
                signature = _array_signature(pair['input'])
                if signature not in seen and not _is_source(pair):
                    break
                rng.shuffle(colors)
            else:
                raise AssertionError('distinct-pair construction exhausted')
            seen.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        point = remaining.pop()
        stack, component = [point], [point]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    component.append(q)
        result.append(component)
    return result


def _mask_features(mask, axes):
    h, w = mask.shape
    a, b = axes
    hn = vn = diagonal_only = fringe = 0
    for r, c in zip(*np.nonzero(mask)):
        rr, cc = a - r, b - c
        lr = 0 <= cc < w and cc != c and not mask[r, cc]
        tb = 0 <= rr < h and rr != r and not mask[rr, c]
        diagonal = 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]
        hn += int(not lr)
        vn += int(not tb)
        diagonal_only += int(not lr and not tb and diagonal)
        fringe += int(not 0 <= rr < h or not 0 <= cc < w)
    components = _components(mask)
    boxes = [(max(r for r, c in comp) - min(r for r, c in comp) + 1,
              max(c for r, c in comp) - min(c for r, c in comp) + 1) for comp in components]
    edges = (bool(mask[0].any()), bool(mask[-1].any()), bool(mask[:, 0].any()), bool(mask[:, -1].any()))
    source_mask = False
    if len(components) == 2 and all(len(comp) == bh * bw for comp, (bh, bw) in zip(components, boxes)):
        for corner, strip in (components, components[::-1]):
            bh, bw = max(r for r, c in corner) - min(r for r, c in corner) + 1, max(c for r, c in corner) - min(c for r, c in corner) + 1
            sr0, sr1 = min(r for r, c in strip), max(r for r, c in strip)
            sc0, sc1 = min(c for r, c in strip), max(c for r, c in strip)
            source_mask |= ((h - 1, w - 1) in corner and 2 <= bh <= 4 and 3 <= bw <= 6
                            and 0 < sr0 <= a / 2 <= sr1 < h - 1 and 0 < sc0 <= sc1 < w - 1
                            and 4 <= sr1 - sr0 + 1 <= 6 and 1 <= sc1 - sc0 + 1 <= 3)
    return {
        'erased': int(mask.sum()), 'components': len(components), 'edges': edges,
        'interior': not any(edges), 'corner': bool(mask[-1, -1]), 'single': len(components) == 1,
        'irregular': any(len(comp) != bh * bw for comp, (bh, bw) in zip(components, boxes)),
        'horizontal_strip': any(bh <= 2 and bw >= 3 * bh for bh, bw in boxes),
        'vertical_strip': any(bw <= 2 and bh >= 3 * bw for bh, bw in boxes),
        'scatter': sum(len(comp) == 1 for comp in components) >= 3,
        'horizontal_needed': hn, 'vertical_needed': vn, 'diagonal_only': diagonal_only,
        'fringe_erased': fringe,
        'source_mask': bool(source_mask),
    }


def _field_features(grid):
    h, w = grid.shape
    adjacency = float(((grid[:-1] == grid[1:]).sum() +
                       (grid[:, :-1] == grid[:, 1:]).sum()) / ((h - 1) * w + h * (w - 1)))
    largest = max(len(comp) for color in np.unique(grid) for comp in _components(grid == color))
    return {'adjacency': adjacency, 'largest_uniform': largest}


def _check_prediction_by_mirrors(demos, grid):
    """Second reconstruction implementation: three direct reflected lookups."""
    erase = _learn_marker(demos)
    a, b = _infer_axes(grid, erase)
    h, w = grid.shape
    result = grid.copy()
    for r, c in zip(*np.nonzero(grid == erase)):
        positions = ((a - r, c), (r, b - c), (a - r, b - c))
        values = {int(grid[y, x]) for y, x in positions
                  if 0 <= y < h and 0 <= x < w and grid[y, x] != erase}
        assert len(values) == 1
        result[r, c] = values.pop()
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert all(1 <= n <= 30 for n in grid.shape)
            assert np.all((grid >= 0) & (grid <= 9))
        assert pair['input'].shape == pair['output'].shape
        signature = _array_signature(pair['input'])
        assert signature not in seen and not _is_source(pair)
        seen.add(signature)
    erase = _learn_marker(puzzle['train'])
    horizontal_demo = vertical_demo = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            inp, out = pair['input'], pair['output']
            mask = inp == erase
            assert mask.any() and not np.any(out == erase)
            assert np.array_equal(mask, inp != out), 'non-erased cells changed'
            assert 2 <= len(np.unique(out)) <= 9
            predicted = solve(puzzle['train'], inp.copy())
            assert np.array_equal(predicted, out), 'incorrect or ambiguous repair'
            axes = _infer_axes(inp, erase)
            assert _infer_axes(out, erase) == axes
            if split == 'train':
                features = _mask_features(mask, axes)
                horizontal_demo |= features['horizontal_needed'] > 0
                vertical_demo |= features['vertical_needed'] > 0
    assert horizontal_demo and vertical_demo, 'training must demonstrate both folds'


def self_test() -> None:
    for erase in range(10):
        colors = [c for c in range(10) if c != erase][:2]
        fallback = _fallback(colors, erase)
        assert np.array_equal(_restore(fallback['input'], erase), fallback['output'])
    sample = generate_puzzle(73)
    assert _puzzle_signature(sample) == _puzzle_signature(generate_puzzle(73))
    assert _puzzle_signature(sample) != _puzzle_signature(generate_puzzle(74))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3),
                                 (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000 + i, nt, ns)
        validate_generated_puzzle(p)
        assert nt is None or len(p['train']) == nt
        assert ns is None or len(p['test']) == ns
        assert _puzzle_signature(p) == _puzzle_signature(generate_puzzle(2000 + i, nt, ns))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    rows, counts, markers, shapes, offsets, parity, palettes = [], set(), set(), set(), set(), set(), set()
    mixed_geometry = fixed_geometry = mixed_offsets = novel_colors = 0
    held_rotations = larger_tests = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        counts.add((len(p['train']), len(p['test'])))
        erase = _learn_marker(p['train'])
        markers.add(erase)
        training_colors = set(int(x) for pair in p['train'] for x in pair['output'].flat)
        geometry_set, palette_set, offset_set = set(), set(), set()
        split_geometry = {'train': set(), 'test': set()}
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                demos = [q for q in p['train'] if q is not pair]
                assert np.array_equal(solve(demos, inp.copy()), out)
                assert np.array_equal(_check_prediction_by_mirrors(demos, inp.copy()), out)
                a, b = _infer_axes(inp, erase)
                h, w = inp.shape
                geometry_set.add((h, w, a, b))
                split_geometry[split].add((h, w, a, b))
                offset_set.add((a - h + 1, b - w + 1))
                shapes.add((h, w))
                offsets.add((int(np.sign(a - h + 1)), int(np.sign(b - w + 1))))
                parity.add((a % 2, b % 2))
                colors = tuple(int(x) for x in np.unique(out))
                palettes.add(len(colors))
                palette_set.add(colors)
                if split == 'test' and set(colors) - training_colors:
                    novel_colors += 1
                f = _mask_features(inp == erase, (a, b))
                f.update(_field_features(out))
                f.update(split=split, shape=(h, w), palette=len(colors))
                f['offset'] = (a - h + 1, b - w + 1)
                f['parity'] = (a % 2, b % 2)
                rows.append(f)
        mixed_geometry += len(geometry_set) > 1
        fixed_geometry += len(geometry_set) == 1
        mixed_offsets += len(offset_set) > 1
        tr, te = split_geometry['train'], split_geometry['test']
        if len(tr) == 1:
            base = next(iter(tr))
            rotated = {_rotate_geometry(base, turns) for turns in (1, 2, 3)}
            held_rotations += bool(te - tr) and te <= rotated
        larger_tests += min(h * w for h, w, a, b in te) > max(h * w for h, w, a, b in tr)
        assert len(palette_set) > 1, 'pair palettes collapsed'
    assert len({x[0] for x in counts}) >= 4 and len({x[1] for x in counts}) >= 3
    assert len(markers) >= 8 and len(shapes) >= 60
    assert (30, 30) in shapes and any(max(s) <= 12 for s in shapes)
    assert any(h >= 2 * w for h, w in shapes) and any(w >= 2 * h for h, w in shapes)
    assert len(offsets) == 9 and len(parity) == 4
    assert palettes == set(range(2, 10))
    assert mixed_geometry >= 15 and fixed_geometry >= 3 and mixed_offsets >= 15 and novel_colors >= 8
    assert held_rotations >= 2 and larger_tests >= 3
    assert sum(r['adjacency'] > 0.6 and r['largest_uniform'] >= 30 for r in rows) >= 20
    assert sum(r['adjacency'] < 0.25 for r in rows) >= 10
    assert len({(r['shape'], r['offset'], r['palette'], r['edges'], r['components']) for r in rows}) >= 250
    assert sum(r['irregular'] and r['diagonal_only'] > 0 and any(r['offset']) for r in rows) >= 15
    assert sum(r['parity'] == (0, 0) and max(r['shape']) <= 14 for r in rows) >= 5
    for split in ('train', 'test'):
        group = [r for r in rows if r['split'] == split]
        assert len({r['edges'] for r in group}) >= 10
        for feature in ('interior', 'single', 'irregular', 'horizontal_strip', 'vertical_strip', 'scatter', 'diagonal_only', 'fringe_erased'):
            assert sum(bool(r[feature]) for r in group) >= 3, (split, feature)
        assert any(r['corner'] for r in group) and any(not r['corner'] for r in group)
        assert any(r['horizontal_needed'] == r['vertical_needed'] == 0 for r in group)
        assert sum(r['horizontal_needed'] > 0 and r['vertical_needed'] > 0 for r in group) >= 10
        assert any(r['source_mask'] for r in group), (split, 'source strip-plus-corner missing')
    for point_kind in ('gap', 'visible'):
        bad = {s: [{k: a.copy() for k, a in q.items()} for q in sample[s]] for s in sample}
        pair = bad['test'][0]
        erase = _learn_marker(bad['train'])
        eligible = pair['input'] == erase if point_kind == 'gap' else pair['input'] != erase
        point = tuple(np.argwhere(eligible)[0])
        pair['output'][point] = (int(pair['output'][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('validator accepted a corrupted output')
