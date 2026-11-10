"""Backwards-blooming compass: radial continuation and a sampled hidden hub.

Only NumPy and the standard library are used. No files, seeds, demonstrations,
or generation metadata are consulted by solve(). All arrays are small; generation
retains at most one puzzle and one candidate pair and never writes artifacts.
"""

import hashlib
import numpy as np


# Clockwise, starting north. Antipodal support keeps the center observable even
# when the entire inner generation and hub have been erased.
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _corners(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0 + 1, c0), (r0, c0 + 1),
            (r1, c1), (r1 - 1, c1), (r1, c1 - 1)}


def _annotations(a):
    candidates = []
    for value in np.unique(a):
        points = np.argwhere(a == value)
        if len(points) != 6:
            continue
        lo, hi = points.min(axis=0), points.max(axis=0)
        bounds = (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]))
        if min(hi - lo) < 2:
            continue
        if set(map(tuple, points)) == _corners(bounds):
            candidates.append((int(value), bounds))
    _require(len(candidates) == 1, "expected one canonical pair of L markers")
    return candidates[0]


def _views(a):
    return tuple(np.rot90(a, k) for k in range(4)) + tuple(
        np.rot90(a[:, ::-1], k) for k in range(4))


def _tile_slice(center, radius, direction):
    y = center[0] + radius * direction[0] - 1
    x = center[1] + radius * direction[1] - 1
    return slice(y, y + 3), slice(x, x + 3)


def _inside(bounds, y, x):
    r0, r1, c0, c1 = bounds
    return r0 <= y <= r1 and c0 <= x <= c1


def _analyze(input_grid):
    """Recover geometry and a color permutation from visible outer generations."""
    a = _grid(input_grid)
    marker, bounds = _annotations(a)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(counts.max()) > a.size // 2, "background must be a strict majority")
    _require(bg != marker, "marker is not background")
    clean = a.copy()
    clean[clean == marker] = bg
    r0, r1, c0, c1 = bounds
    _require(bool(np.all(clean[r0:r1 + 1, c0:c1 + 1] == bg)), "window is not erased")
    points = np.argwhere(clean != bg)
    _require(len(points) > 0, "no stamp evidence")
    lo, hi = points.min(axis=0), points.max(axis=0)
    _require(bool(np.all((lo + hi) % 2 == 0)), "nonintegral center")
    center = tuple(map(int, (lo + hi) // 2))
    _require(int(hi[0] - lo[0]) == int(hi[1] - lo[1]), "outer support must be square")
    outer_radius = int((hi[0] - lo[0]) // 2 - 1)
    cy, cx = center
    _require(r0 <= cy - 1 and r1 >= cy + 1 and c0 <= cx - 1 and c1 >= cx + 1,
             "window must hide the entire hub")
    candidates = []
    for step, depth in ((3, 3), (3, 4), (4, 3)):
        if step * depth != outer_radius:
            continue
        reach = step + 1
        if not (cy - reach <= r0 <= r1 <= cy + reach and
                cx - reach <= c0 <= c1 <= cx + reach):
            continue
        support = tuple(i for i, d in enumerate(DIRECTIONS)
                        if np.all(clean[_tile_slice(center, outer_radius, d)] != bg))
        if len(support) not in (4, 6, 8):
            continue
        if any((i + 4) % 8 not in support for i in support):
            continue
        if not any(i % 2 == 0 for i in support):
            continue
        occupied = np.zeros(a.shape, dtype=bool)
        tiles = {}
        for i in support:
            for k in range(1, depth + 1):
                sl = _tile_slice(center, step * k, DIRECTIONS[i])
                occupied[sl] = True
                if k >= 2:
                    tiles[i, k] = clean[sl].copy()
        occupied[r0:r1 + 1, c0:c1 + 1] = False
        if not np.array_equal(occupied, clean != bg):
            continue
        observed = np.concatenate([t.ravel() for t in tiles.values()])
        vals, nums = np.unique(observed, return_counts=True)
        carrier = int(vals[np.argmax(nums)])
        if int(nums.max()) <= len(observed) // 2:
            continue
        forward = {}
        good = True
        for i in support:
            for k in range(2, depth):
                source = tiles[i, k][::-1, ::-1]
                target = tiles[i, k + 1]
                for old, new in zip(source.flat, target.flat):
                    old, new = int(old), int(new)
                    if old in forward and forward[old] != new:
                        good = False
                    forward[old] = new
        colors = set(map(int, vals))
        details = colors - {carrier}
        if not good or set(forward) != colors or set(forward.values()) != colors:
            continue
        if forward[carrier] != carrier or not 2 <= len(details) <= 7:
            continue
        # Every detail color participates in ONE cycle; its order is visible,
        # never the numerical order of ARC colors or a hidden puzzle convention.
        visited, color = set(), min(details)
        while color not in visited:
            visited.add(color)
            color = forward[color]
        if visited != details or color != min(details):
            continue
        inverse = np.arange(10, dtype=np.int8)
        for old, new in forward.items():
            inverse[new] = old
        inner = {i: inverse[tiles[i, 2]][::-1, ::-1].copy() for i in support}
        witness = False
        for i, tile in inner.items():
            dr, dc = DIRECTIONS[i]
            if tile[1 - dr, 1 - dc] == carrier or tile[1 + dr, 1 + dc] != carrier:
                good = False
            if np.count_nonzero(tile == carrier) < 5:
                good = False
            if i % 2 == 0 and len({v.tobytes() for v in _views(tile == carrier)}) == 8:
                witness = True
            sl = _tile_slice(center, step, DIRECTIONS[i])
            actual = clean[sl]
            for y in range(3):
                for x in range(3):
                    gy, gx = sl[0].start + y, sl[1].start + x
                    if not _inside(bounds, gy, gx) and actual[y, x] != tile[y, x]:
                        good = False
        if not good or not witness:
            continue
        candidates.append(dict(background=bg, marker=marker, carrier=carrier,
                               center=center, step=step, depth=depth, bounds=bounds,
                               support=support, forward=forward, inverse=inverse,
                               inner=inner, outer_tiles=tiles))
    _require(len(candidates) == 1, "inconsistent or ambiguous radial continuation")
    return a, candidates[0]


def _reconstruct(a, info, ablation=None):
    """Crop after reversing the visible recurrence and sampling its hidden input."""
    bg, carrier = info['background'], info['carrier']
    center, step, bounds = info['center'], info['step'], info['bounds']
    cy, cx = center
    result = a.copy()
    r0, r1, c0, c1 = bounds
    result[r0:r1 + 1, c0:c1 + 1] = bg
    hub = np.full((3, 3), bg, dtype=np.int8)
    hub[1, 1] = carrier
    for i in info['support']:
        dr, dc = DIRECTIONS[i]
        tile = info['inner'][i]
        if ablation == 'no_color_continuation':
            tile = info['outer_tiles'][i, 2][::-1, ::-1]
        elif ablation == 'no_half_turn':
            tile = info['inverse'][info['outer_tiles'][i, 2]]
        elif ablation == 'radial_reflection':
            tile = info['inverse'][info['outer_tiles'][i, 2]]
            tile = tile[::-1 if dr else 1, ::-1 if dc else 1]
        sl = _tile_slice(center, step, DIRECTIONS[i])
        result[sl] = tile
        if ablation == 'no_sampling':
            hub[1 + dr, 1 + dc] = carrier
        elif ablation == 'outward_sampling':
            hub[1 + dr, 1 + dc] = tile[1 + dr, 1 + dc]
        else:
            hub[1 + dr, 1 + dc] = tile[1 - dr, 1 - dc]
    result[cy - 1:cy + 2, cx - 1:cx + 2] = hub
    return result[r0:r1 + 1, c0:c1 + 1].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution, including geometry, color phase, and hub sampling."""
    a, info = _analyze(input_grid)
    return _reconstruct(a, info)


def _stamp_masks(rng, support):
    masks = {}
    witness = int(rng.choice([i for i in support if i % 2 == 0]))
    for i in support:
        dr, dc = DIRECTIONS[i]
        inward, outward = (1 - dr, 1 - dc), (1 + dr, 1 + dc)
        places = [(y, x) for y in range(3) for x in range(3)
                  if (y, x) not in (inward, outward)]
        if i == witness:
            options = []
            for place in places:
                mask = np.zeros((3, 3), dtype=bool)
                mask[inward] = mask[place] = True
                if len({v.tobytes() for v in _views(mask)}) == 8:
                    options.append(mask)
            masks[i] = options[int(rng.integers(len(options)))].copy()
        else:
            rng.shuffle(places)
            count = int(rng.integers(1, 5))
            mask = np.zeros((3, 3), dtype=bool)
            mask[inward] = True
            for place in places[:count - 1]:
                mask[place] = True
            masks[i] = mask
    return masks


def _crop_offsets(rng, reach, test, full_axis=0):
    # Training clips at least one end of BOTH axes. Tests span both ends of
    # at least one axis, joining opposite rays inside a single missing window.
    def partial():
        choices = [(low, high) for low in range(-reach, 0)
                   for high in range(1, reach + 1)
                   if (low, high) != (-reach, reach)]
        return choices[int(rng.integers(len(choices)))]
    row, col = partial(), partial()
    if test:
        if full_axis == 0:
            row = (-reach, reach)
        else:
            col = (-reach, reach)
        if rng.random() < .3:
            row = col = (-reach, reach)
    return row[0], row[1], col[0], col[1]


def _make_pair(rng, test, preferred_step):
    step = preferred_step if rng.random() < .65 else int(rng.choice([3, 4]))
    depth = int(rng.choice([3, 4])) if step == 3 else 3
    count = int(rng.integers(2, 5))
    pairs = sorted(map(int, rng.choice(4, size=count, replace=False)))
    if not any(i % 2 == 0 for i in pairs):
        pairs[0] = int(rng.choice([0, 2]))
        pairs.sort()
    support = tuple(sorted(pairs + [i + 4 for i in pairs]))
    masks = _stamp_masks(rng, support)
    slots = [(i, int(y), int(x)) for i in support for y, x in np.argwhere(masks[i])]
    p = int(rng.integers(2, min(7, len(slots)) + 1))
    roles = list(map(int, rng.permutation(10)))
    bg, marker, carrier = roles[:3]
    palette = roles[3:3 + p]
    forward = np.arange(10, dtype=np.int8)
    for old, new in zip(palette, palette[1:] + palette[:1]):
        forward[old] = new
    inner = {i: np.full((3, 3), carrier, dtype=np.int8) for i in support}
    # Expose every color at EVERY generation; the complete one-step permutation
    # is therefore recoverable from the two untouched outer generations.
    rng.shuffle(slots)
    for j, (i, y, x) in enumerate(slots):
        inner[i][y, x] = palette[j] if j < p else int(rng.choice(palette))
    footprint = 2 * step * depth + 3
    h, w = map(int, rng.integers(footprint, 31, size=2))
    top = int(rng.integers(h - footprint + 1))
    left = int(rng.integers(w - footprint + 1))
    center = (top + step * depth + 1, left + step * depth + 1)
    cy, cx = center
    field = np.full((h, w), bg, dtype=np.int8)
    # Ground truth uses forward, scalar coordinate painting of latent stamps.
    # It never calls solve(), _analyze(), or the reverse reconstruction helper.
    for i in support:
        dr, dc = DIRECTIONS[i]
        for k in range(1, depth + 1):
            for y in range(3):
                for x in range(3):
                    yy, xx = (2 - y, 2 - x) if k % 2 == 0 else (y, x)
                    value = int(inner[i][yy, xx])
                    for _ in range(k - 1):
                        value = int(forward[value])
                    field[cy + step * k * dr - 1 + y,
                          cx + step * k * dc - 1 + x] = value
        field[cy + dr, cx + dc] = inner[i][1 - dr, 1 - dc]
    field[cy, cx] = carrier
    reach = step + 1
    full_axis = int(rng.choice([i // 2 for i in pairs if i % 2 == 0]))
    for _ in range(64):
        offsets = _crop_offsets(rng, reach, test, full_axis)
        bounds = (cy + offsets[0], cy + offsets[1], cx + offsets[2], cx + offsets[3])
        hidden = 0
        for i in support:
            dr, dc = DIRECTIONS[i]
            for y, x in np.argwhere(masks[i]):
                hidden += _inside(bounds, cy + step * dr - 1 + int(y),
                                  cx + step * dc - 1 + int(x))
        if hidden:
            break
    else:
        # Always hides the inward color of the chosen active cardinal witness,
        # while preserving the same partial/full-axis split convention.
        i = next(i for i in support if i % 2 == 0)
        dr, dc = DIRECTIONS[i]
        row = (-reach, reach) if test else ((-reach, 1) if dr < 0 else (-1, reach) if dr > 0 else (-1, 1))
        col = (-reach, 1) if dc < 0 else (-1, reach) if dc > 0 else (-1, 1)
        bounds = (cy + row[0], cy + row[1], cx + col[0], cx + col[1])
    r0, r1, c0, c1 = bounds
    output = field[r0:r1 + 1, c0:c1 + 1].copy()
    a = field.copy()
    a[r0:r1 + 1, c0:c1 + 1] = bg
    for y, x in _corners(bounds):
        a[y, x] = marker
    return dict(input=a, output=output)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' out of range')
    return int(value)


def _distinct(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    # Ninety distinct role recolorings, with at most thirteen preceding inputs.
    # Every input visibly contains both background and carrier, so the 90 images
    # really are distinct. This proves that finite duplicate repair cannot fail.
    _, info = _analyze(pair['input'])
    old = [info['background'], info['carrier']] + [c for c in range(10)
            if c not in (info['background'], info['carrier'])]
    for bg in range(10):
        for carrier in range(10):
            if bg == carrier:
                continue
            new = [bg, carrier] + [c for c in range(10) if c not in (bg, carrier)]
            mapping = np.empty(10, dtype=np.int8)
            mapping[old] = new
            a = mapping[pair['input']]
            if _key(a) not in seen:
                return dict(input=a, output=mapping[pair['output']])
    raise AssertionError('impossible: all 90 recolorings excluded')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    preferred_step = int(rng.choice([3, 4]))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for _ in range(count):
            pair = _distinct(_make_pair(rng, split == 'test', preferred_step), seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    return puzzle


def measure_pair(pair):
    a, info = _analyze(pair['input'])
    cy, cx = info['center']
    r0, r1, c0, c1 = info['bounds']
    reach = info['step'] + 1
    full_axes = int((r0, r1) == (cy - reach, cy + reach)) + int(
        (c0, c1) == (cx - reach, cx + reach))
    hidden = 0
    samples, masks = [], []
    for i in info['support']:
        dr, dc = DIRECTIONS[i]
        tile = info['inner'][i]
        samples.append(int(tile[1 - dr, 1 - dc]))
        masks.append(tuple(map(int, (tile != info['carrier']).flat)))
        for y, x in np.argwhere(tile != info['carrier']):
            hidden += _inside(info['bounds'], cy + info['step'] * dr - 1 + int(y),
                              cx + info['step'] * dc - 1 + int(x))
    return dict(step=info['step'], depth=info['depth'], support=info['support'],
                background=info['background'], marker=info['marker'], carrier=info['carrier'],
                cycle_length=len(info['forward']) - 1, full_axes=full_axes,
                hidden_detail_cells=hidden, samples=tuple(samples), masks=tuple(masks),
                input_shape=a.shape, output_shape=(r1 - r0 + 1, c1 - c0 + 1),
                offsets=(r0 - cy, r1 - cy, c0 - cx, c1 - cx))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            _, info = _analyze(a)
            expected = _reconstruct(a, info)
            _require(np.array_equal(b, expected), "wrong continuation, crop, or hub")
            _require(info['marker'] not in b, "marker leaked")
            _require(set(map(int, np.unique(b))) <= set(map(int, np.unique(a))), "unseen output color")
            features = measure_pair(pair)
            _require(features['hidden_detail_cells'] > 0, "no hidden stamp content")
            _require((features['full_axes'] > 0) == (split == 'test'), "occlusion shift missing")
            if split == 'test':
                cy, cx = info['center']
                reach = info['step'] + 1
                r0, r1, c0, c1 = info['bounds']
                _require((0 in info['support'] and (r0, r1) == (cy - reach, cy + reach)) or
                         (2 in info['support'] and (c0, c1) == (cx - reach, cx + reach)),
                         'test must hide a complete active antipodal cardinal pair')
            for ablation in ('no_color_continuation', 'no_half_turn', 'no_sampling', 'outward_sampling'):
                _require(not np.array_equal(b, _reconstruct(a, info, ablation)),
                         'answer does not depend on ' + ablation)
    # The fully visible outer transitions also distinguish half-turn from all
    # other D4 operations via a cardinal carrier-mask witness in EVERY input.
    # A training crop need not happen to cut the particular off-axis witness.


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, content = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(content)
    return digest.hexdigest()


def self_test():
    """Bounded, in-memory checks across 42 seeds; returns small measurements."""
    features, signatures, normalized = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                features.append(measure_pair(pair))
                a = pair['input']
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                normalized.add((a.shape, hashlib.sha256(norm).digest()))
                before = a.copy()
                _require(np.array_equal(solve(a.tolist()), pair['output']), 'list input failed')
                _require(np.array_equal(a, before), 'solver mutated input')
        if seed < 3:
            _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), 'nondeterminism')
            # Recolor all ten roles together; solve must not use their identities.
            permutation = np.random.default_rng(9000 + seed).permutation(10)
            for pair in puzzle['test']:
                _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
                         'color equivariance failed')
                marker = measure_pair(pair)['marker']
                a = np.rot90(pair['input']).copy()
                points = np.argwhere(a == marker)
                lo, hi = points.min(axis=0), points.max(axis=0)
                bounds = (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]))
                # Rotations transform the construction; redraw the fixed L convention.
                a[a == marker] = measure_pair(pair)['background']
                for y, x in _corners(bounds):
                    a[y, x] = marker
                _require(np.array_equal(solve(a), np.rot90(pair['output'])), 'rotation equivariance failed')
    _require(len(signatures) == 42 and len(normalized) >= 180, 'structural diversity')
    _require({f['step'] for f in features} == {3, 4}, 'spacing variation')
    _require({f['depth'] for f in features} == {3, 4}, 'depth variation')
    _require({len(f['support']) for f in features} == {4, 6, 8}, 'spoke count variation')
    _require({f['cycle_length'] for f in features} == set(range(2, 8)), 'cycle variation')
    _require(len({f['support'] for f in features}) >= 8, 'spoke geometry variation')
    _require(len({f['input_shape'] for f in features}) >= 30, 'canvas variation')
    _require(len({f['output_shape'] for f in features}) >= 25, 'crop variation')
    _require({f['background'] for f in features} == set(range(10)), 'background variation')
    _require({f['marker'] for f in features} == set(range(10)), 'marker variation')
    _require(len({f['masks'] for f in features}) >= 180, 'stamp geometry variation')
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            validate_generated_puzzle(generate_puzzle(4100 + n * 10 + m, n, m))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(999, 2, 1)
    b = broken['test'][0]['output']
    b[0, 0] = (int(b[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    # Exact duplicate repair is exercised deliberately, rather than waiting for
    # a practically impossible random collision.
    base = generate_puzzle(998, 2, 1)['train'][0]
    seen = set()
    for _ in range(14):
        repaired = _distinct(base, seen)
        seen.add(_key(repaired['input']))
        _require(np.array_equal(solve(repaired['input']), repaired['output']), 'duplicate repair')
    _require(len(seen) == 14, 'duplicate repair did not provide distinct inputs')
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(signatures),
                normalized_inputs=len(normalized), input_shapes=len({f['input_shape'] for f in features}),
                output_shapes=len({f['output_shape'] for f in features}),
                supports=len({f['support'] for f in features}),
                cycles=sorted({f['cycle_length'] for f in features}))
