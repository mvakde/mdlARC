"""Generate reflected mosaics with a single erased rectangular patch."""

import hashlib

import numpy as np


_SOURCE_INPUT_DIGESTS = frozenset({
    "2a72a5f7905f42287fa2c28454e851c7a59f41cc94664bb30207a2a60db3bfb5",
    "e91b211eb3f21048413644c6c71591f80b5cd3e08974884b9248796271105ca9",
    "8dbadad1f43c938837098fbf818dd70e68311f3ae6855caa94e94c3b2ce0fb85",
    "c3ee6b100d8ecdb1bc14e52273676cfd09c440732ba3b0177bbe9ad5bf611fa8",
    "77da49ff6ddabed6eeff85db99ebfabeb43a22a40cad60b595254b9d4506161b",
})


def _checked_int(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    ntr = _checked_int(num_train, "num_train")
    nte = _checked_int(num_test, "num_test")
    if ntr is not None and not 2 <= ntr <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if nte is not None and not 1 <= nte <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if ntr is None and nte is None:
        nte = 1 + seed % 4
        low = max(2, nte + 1)
        ntr = low + ((seed + seed // 4) % (11 - low))
    elif ntr is None:
        ntr = int(rng.integers(nte + 1, 11))
    elif nte is None:
        nte = int(rng.integers(1, min(4, ntr - 1) + 1))
    if ntr <= nte:
        raise ValueError("num_train must exceed num_test")
    return ntr, nte


def _reflect(r, c, kind, k):
    if kind == "H":
        return k - r, c
    if kind == "V":
        return r, k - c
    if kind == "D":
        return c + k, r - k
    raise AssertionError("unknown reflection")


def _symmetrize(grid, kind, k):
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            rr, cc = _reflect(r, c, kind, k)
            if 0 <= rr < height and 0 <= cc < width and (r, c) < (rr, cc):
                grid[rr, cc] = grid[r, c]


def _make_field(rng, shape, palette, axes):
    values = np.asarray(palette, dtype=np.int8)
    weights = rng.integers(1, 8, size=len(values)).astype(float)
    weights /= weights.sum()
    grid = rng.choice(values, size=shape, p=weights).astype(np.int8)
    height, width = shape
    texture = int(rng.integers(0, 4))

    # Independent blocks and bands provide coherent motifs without replacing
    # the irregular cells that make reflection easy to detect.
    block_count = (2, 6, 9, 5)[texture]
    for _ in range(block_count):
        rh = int(rng.integers(1, min(8, height) + 1))
        rw = int(rng.integers(1, min(8, width) + 1))
        r = int(rng.integers(0, height - rh + 1))
        c = int(rng.integers(0, width - rw + 1))
        color = values[int(rng.integers(len(values)))]
        grid[r:r + rh, c:c + rw] = color
    if texture in (1, 3):
        for _ in range(int(rng.integers(2, 6))):
            if bool(rng.integers(0, 2)):
                r = int(rng.integers(height))
                c = int(rng.integers(width - 4))
                end = min(width, c + int(rng.integers(5, 14)))
                grid[r:r + int(rng.integers(1, 3)), c:end] = values[int(rng.integers(len(values)))]
            else:
                c = int(rng.integers(width))
                r = int(rng.integers(height - 4))
                end = min(height, r + int(rng.integers(5, 14)))
                grid[r:end, c:c + int(rng.integers(1, 3))] = values[int(rng.integers(len(values)))]
    if texture == 3:
        r = int(rng.integers(0, height - 8))
        c = int(rng.integers(0, width - 8))
        for inset in range(3):
            color = values[int(rng.integers(len(values)))]
            grid[r + inset, c + inset:c + 9 - inset] = color
            grid[r + 8 - inset, c + inset:c + 9 - inset] = color
            grid[r + inset:r + 9 - inset, c + inset] = color
            grid[r + inset:r + 9 - inset, c + 8 - inset] = color

    for kind, k in axes:
        _symmetrize(grid, kind, k)
    return grid


def _axis_layout(rng, length, extent, force_gap=None, force_side=None, offcenter=False):
    max_gap = min(9, (length - 2 * extent) // 2)
    if max_gap < 0:
        return None
    gaps = [force_gap] if force_gap is not None else list(range(max_gap + 1))
    choices = []
    for gap in gaps:
        if gap is None or gap > max_gap:
            continue
        low = extent + gap - 1
        high = length - extent - gap - 1
        for a in range(low, high + 1):
            k = 2 * a + 1
            if offcenter and k == length - 1:
                continue
            for side in ("before", "after"):
                if force_side is not None and side != force_side:
                    continue
                start = a - gap - extent + 1 if side == "before" else a + 1 + gap
                choices.append((start, k, gap))
    if not choices:
        return None
    # Favor boundary starts and the widest feasible gaps in some pairs.
    if force_gap is None and rng.random() < 0.35:
        extreme = [item for item in choices if item[0] in (0, length - extent)]
        if extreme:
            choices = extreme
    elif force_gap is None and rng.random() < 0.3:
        far = max(item[2] for item in choices)
        choices = [item for item in choices if item[2] >= max(0, far - 2)]
    return choices[int(rng.integers(len(choices)))][:2]


def _sample_dimensions(rng, profile):
    if profile == "large":
        return int(rng.integers(8, 12)), int(rng.integers(8, 12))
    if profile == "narrow_v":
        return int(rng.integers(7, 13)), int(rng.integers(1, 4))
    if profile == "narrow_h":
        return int(rng.integers(1, 4)), int(rng.integers(7, 13))
    return int(rng.integers(3, 10)), int(rng.integers(3, 10))


def _sample_geometry(rng, kind, profile, multi, source_shift):
    for _ in range(120):
        if rng.random() < 0.18:
            height = width = 30
        else:
            height = int(rng.integers(24 if source_shift else 20, 31))
            width = int(rng.integers(20, 31))
        h, w = _sample_dimensions(rng, profile)
        if h > height or w > width:
            continue
        if multi:
            row = _axis_layout(rng, height, h,
                               force_gap=0 if source_shift else None,
                               force_side="after" if source_shift else None,
                               offcenter=source_shift)
            col = _axis_layout(rng, width, w)
            if row is None or col is None:
                continue
            r0, kr = row
            c0, kc = col
            return (height, width), (r0, c0, h, w), [("H", kr), ("V", kc)]
        if kind in ("H", "V"):
            axis = _axis_layout(rng, height if kind == "H" else width,
                                h if kind == "H" else w,
                                force_gap=0 if source_shift else None,
                                force_side="after" if source_shift else None,
                                offcenter=source_shift)
            if axis is None:
                continue
            start, k = axis
            if kind == "H":
                r0 = start
                c0 = int(rng.integers(width - w + 1))
                if rng.random() < 0.25:
                    c0 = 0 if bool(rng.integers(2)) else width - w
            else:
                c0 = start
                r0 = int(rng.integers(height - h + 1))
                if rng.random() < 0.25:
                    r0 = 0 if bool(rng.integers(2)) else height - h
            return (height, width), (r0, c0, h, w), [(kind, k)]

        # Transposition about r-c=k. Either side and the displacement of the
        # diagonal can vary; source and mask occupy disjoint square regions.
        gap = int(rng.integers(0, 9))
        required = h + w + gap
        if required > min(height, width):
            continue
        size = int(rng.integers(required, min(height, width) + 1))
        origin_r = int(rng.integers(height - size + 1))
        origin_c = int(rng.integers(width - size + 1))
        start = int(rng.integers(size - required + 1))
        if bool(rng.integers(2)):
            r0 = origin_r + start + w + gap
            c0 = origin_c + start
        else:
            r0 = origin_r + start
            c0 = origin_c + start + h + gap
        return (height, width), (r0, c0, h, w), [("D", origin_r - origin_c)]
    raise RuntimeError("no legal geometry found")


def _mapped_patch(grid, bbox, kind, k, forbidden=None):
    r0, c0, h, w = bbox
    rr, cc = np.indices((h, w))
    rr += r0
    cc += c0
    if kind == "H":
        rr = k - rr
    elif kind == "V":
        cc = k - cc
    else:
        rr, cc = cc + k, rr - k
    if (np.any(rr < 0) or np.any(rr >= grid.shape[0]) or
            np.any(cc < 0) or np.any(cc >= grid.shape[1])):
        return None
    patch = grid[rr, cc]
    if forbidden is not None and np.any(patch == forbidden):
        return None
    return patch


def _find_marker(grid):
    candidates = []
    for color in range(10):
        positions = np.argwhere(grid == color)
        if len(positions) < 6:
            continue
        r0, c0 = positions.min(axis=0)
        r1, c1 = positions.max(axis=0)
        h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
        if len(positions) == h * w and np.all(grid[r0:r1 + 1, c0:c1 + 1] == color):
            candidates.append((color, (int(r0), int(c0), h, w)))
    assert len(candidates) == 1, "expected one dedicated rectangular marker"
    return candidates[0]


def _reflection_support(grid, marker, kind, k):
    height, width = grid.shape
    if kind == "H":
        rows = [r for r in range(height) if 0 <= k - r < height and r < k - r]
        if not rows:
            return 0, 0
        left, right = grid[rows, :], grid[[k - r for r in rows], :]
    elif kind == "V":
        cols = [c for c in range(width) if 0 <= k - c < width and c < k - c]
        if not cols:
            return 0, 0
        left, right = grid[:, cols], grid[:, [k - c for c in cols]]
    else:
        rr, cc = np.indices(grid.shape)
        qr, qc = cc + k, rr - k
        valid = ((qr >= 0) & (qr < height) & (qc >= 0) & (qc < width) &
                 ((rr < qr) | ((rr == qr) & (cc < qc))))
        if not np.any(valid):
            return 0, 0
        left, right = grid[rr[valid], cc[valid]], grid[qr[valid], qc[valid]]
    usable = (left != marker) & (right != marker)
    return int(np.count_nonzero(usable)), int(np.count_nonzero(usable & (left == right)))


def _supported_candidates(grid, marker, bbox):
    height, width = grid.shape
    minimum = max(24, (bbox[2] * bbox[3] + 1) // 2)
    specs = ([('H', k) for k in range(2 * height - 1)] +
             [('V', k) for k in range(2 * width - 1)] +
             [('D', k) for k in range(-(width - 1), height)])
    candidates = []
    for kind, k in specs:
        patch = _mapped_patch(grid, bbox, kind, k, forbidden=marker)
        if patch is None:
            continue
        compared, matched = _reflection_support(grid, marker, kind, k)
        if compared >= minimum and matched == compared:
            candidates.append((kind, k, patch, compared))
    return candidates


def _unique_patch(candidates):
    if not candidates:
        return None
    first = candidates[0][2]
    if any(not np.array_equal(first, item[2]) for item in candidates[1:]):
        return None
    return first


def _generate_pair(rng, marker, kind, profile, multi, source_shift):
    for _ in range(80):
        shape, bbox, axes = _sample_geometry(rng, kind, profile, multi, source_shift)
        available = np.asarray([c for c in range(10) if c != marker], dtype=np.int8)
        palette = tuple(int(x) for x in rng.choice(
            available, size=int(rng.integers(6, 10)), replace=False))
        full = _make_field(rng, shape, palette, axes)
        output = _mapped_patch(full, bbox, *axes[0])
        if output is None or len(np.unique(output)) < 3:
            continue
        inp = full.copy()
        r0, c0, h, w = bbox
        inp[r0:r0 + h, c0:c0 + w] = marker
        try:
            found_marker, found_bbox = _find_marker(inp)
        except AssertionError:
            continue
        if found_marker != marker or found_bbox != bbox:
            continue
        candidates = _supported_candidates(inp, marker, bbox)
        recovered = _unique_patch(candidates)
        if (recovered is not None and np.array_equal(recovered, output) and
                all(any(item[0] == axis[0] and item[1] == axis[1]
                        for item in candidates) for axis in axes)):
            return {'input': inp.astype(np.int8), 'output': output.astype(np.int8)}
    raise RuntimeError("could not generate an unambiguous pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _checked_int(seed, 'seed')
    if seed is None or seed < 0:
        raise ValueError('seed must be nonnegative')
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(seed, rng, num_train, num_test)
    marker = int(rng.integers(10))

    # The only shared convention is the marker's role and color. Geometry,
    # content palette, and texture can vary independently in each pair.
    kinds = list(rng.permutation(['H', 'V', 'D']))
    if n_train == 2:
        kinds = kinds[:2]
    else:
        kinds += [str(rng.choice(['H', 'V', 'D'])) for _ in range(n_train - 3)]
        rng.shuffle(kinds)
    profiles = [str(rng.choice(['balanced', 'large', 'narrow_v', 'narrow_h']))
                for _ in range(n_train)]
    if n_train >= 3:
        spots = rng.choice(n_train, size=2, replace=False)
        profiles[int(spots[0])] = 'large'
        profiles[int(spots[1])] = 'narrow_v'

    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', n_train), ('test', n_test)):
        for i in range(count):
            if split == 'train':
                kind, profile = kinds[i], profiles[i]
            else:
                kind = str(rng.choice(kinds))
                profile = str(rng.choice(
                    ['narrow_v', 'balanced', 'narrow_h', 'large'],
                    p=[0.4, 0.27, 0.2, 0.13]))
            multi = kind != 'D' and rng.random() < 0.48
            source_shift = (split == 'test' and kind == 'H' and
                            profile == 'narrow_v' and rng.random() < 0.7)
            for _ in range(12):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _generate_pair(child, marker, kind, profile, multi, source_shift)
                signature = (pair['input'].shape, pair['input'].tobytes())
                if signature not in seen:
                    break
            else:
                raise RuntimeError('duplicate pair sampling')
            seen.add(signature)
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    ntr, nte = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= ntr <= 10 and 1 <= nte <= 4 and ntr > nte
    markers, train_kinds, test_kinds = [], set(), []
    signatures = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            signature = (inp.shape, inp.tobytes())
            assert signature not in signatures
            signatures.add(signature)
            if inp.shape == (30, 30):
                assert hashlib.sha256(inp.astype(np.uint8).tobytes()).hexdigest() not in _SOURCE_INPUT_DIGESTS
            marker, bbox = _find_marker(inp)
            markers.append(marker)
            assert out.shape == bbox[2:]
            assert marker not in out and len(np.unique(out)) >= 3
            assert len(set(int(v) for v in np.unique(inp)) - {marker}) >= 6
            candidates = _supported_candidates(inp, marker, bbox)
            recovered = _unique_patch(candidates)
            assert recovered is not None and np.array_equal(recovered, out)
            pair_kinds = {item[0] for item in candidates}
            if split == 'train':
                train_kinds.update(pair_kinds)
            else:
                test_kinds.append(pair_kinds)
    assert len(set(markers)) == 1
    assert all(kinds & train_kinds for kinds in test_kinds)
    if ntr >= 3:
        assert train_kinds == {'H', 'V', 'D'}


def _puzzles_equal(a, b):
    return (all(len(a[s]) == len(b[s]) for s in ('train', 'test')) and
            all(np.array_equal(x[key], y[key])
                for s in ('train', 'test') for x, y in zip(a[s], b[s])
                for key in ('input', 'output')))


def self_test() -> None:
    a = generate_puzzle(314159, 6, 2)
    b = generate_puzzle(314159, 6, 2)
    assert _puzzles_equal(a, b)
    assert not _puzzles_equal(a, generate_puzzle(314160, 6, 2))
    for i, (ntr, nte) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        p = generate_puzzle(500 + i, ntr, nte)
        assert (len(p['train']), len(p['test'])) == (ntr, nte)
        validate_generated_puzzle(p)
    for i, (ntr, nte) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(700 + i, ntr, nte)
        validate_generated_puzzle(p)
        assert ntr is None or len(p['train']) == ntr
        assert nte is None or len(p['test']) == nte
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                   {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                   {'num_train': 4, 'num_test': 4}):
        try:
            generate_puzzle(900, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts = set(), set()
    shapes, masks, layouts, markers = set(), set(), set(), set()
    axis_sets, test_kinds, train_palettes = set(), set(), set()
    multi_pairs = narrow_train = narrow_test = wide_test = source_sizes = 0
    boundary = long_gap = source_like = multi_edge = large_edge = 0
    for p in puzzles:
        validate_generated_puzzle(p)
        train_counts.add(len(p['train']))
        test_counts.add(len(p['test']))
        local_palettes = set()
        for split in ('train', 'test'):
            for pair in p[split]:
                inp = pair['input']
                marker, bbox = _find_marker(inp)
                cands = _supported_candidates(inp, marker, bbox)
                assert np.array_equal(_unique_patch(cands), pair['output'])
                kinds = frozenset(item[0] for item in cands)
                axis_sets.add(kinds)
                multi_pairs += len(kinds) > 1
                r, c, h, w = bbox
                shapes.add(inp.shape)
                masks.add((h, w))
                layouts.add((inp.shape, bbox, tuple((x[0], x[1]) for x in cands)))
                markers.add(marker)
                boundary += r == 0 or c == 0 or r + h == inp.shape[0] or c + w == inp.shape[1]
                multi_edge += (len(kinds) > 1 and
                               (r == 0 or c == 0 or r + h == inp.shape[0] or c + w == inp.shape[1]))
                large_edge += (h >= 9 and w >= 9 and
                               (r == 0 or c == 0 or r + h == inp.shape[0] or c + w == inp.shape[1]))
                source_sizes += inp.shape == (30, 30)
                long_gap += any((x[0] == 'H' and abs(2 * r + h - 1 - x[1]) >= h + 6)
                                or (x[0] == 'V' and abs(2 * c + w - 1 - x[1]) >= w + 6)
                                for x in cands)
                if split == 'test':
                    source_like += any(x[0] == 'H' and h >= 7 and w <= 3 and
                                       r == (x[1] + 1) // 2 and x[1] != inp.shape[0] - 1
                                       for x in cands)
                local_palettes.add(tuple(sorted(int(v) for v in np.unique(inp) if v != marker)))
                if split == 'train':
                    narrow_train += h >= 3 * w or w >= 3 * h
                    train_palettes.add(tuple(sorted(int(v) for v in np.unique(inp) if v != marker)))
                else:
                    narrow_test += h >= 3 * w
                    wide_test += w >= 3 * h
                    test_kinds.update(kinds)
        assert len(local_palettes) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(masks) >= 30 and len(layouts) >= 80
    assert len(markers) >= 6 and len(train_palettes) >= 20
    assert multi_pairs >= 15 and narrow_train >= 10 and narrow_test >= 10
    assert wide_test >= 3 and source_sizes >= 10 and boundary >= 10
    assert long_gap >= 10 and test_kinds == {'H', 'V', 'D'}
    assert source_like >= 2 and multi_edge >= 10 and large_edge >= 5
