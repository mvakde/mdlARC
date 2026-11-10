import hashlib
import numpy as np


# Exact solved source contents, not a ban on their shapes or palettes.
_SOURCE_FINGERPRINTS = {
    'f8cac9aae125b7952b08d591a632b9579a0cf1593829c318b50bd7921db72c59',
    '51945bda5bd69ed61be09539c4e5a6731ac0eeecff2f0b81530ddc8278b2a4ab',
    '6c17d405c3cd9b4d678050ac26599702565c466c7414d54490df3da8c5a54c3a',
    'e71488af687c2d69200f11274d6d7b6b5f1119b0a74f6355bea1c06fb2f11a91',
    'b4babd6726931ea1dca1ccbc200b8ff2efc883deada50ca8924f94719de6737e',
}
_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _require_optional_int(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(seed, num_train, num_test):
    _require_optional_int("num_train", num_train, 2, 10)
    _require_optional_int("num_test", num_test, 1, 4)

    supplied_train = num_train is not None
    supplied_test = num_test is not None
    s = int(seed)

    if supplied_train and supplied_test:
        nt = int(num_train)
        ne = int(num_test)
        if nt <= ne:
            raise ValueError("num_train must be greater than num_test")
        return nt, ne

    if not supplied_train and not supplied_test:
        ne = 1 + (s % 4)
        candidate = 2 + ((s // 4) % 9)
        nt = max(candidate, ne + 1)
        return nt, ne

    if not supplied_train:
        ne = int(num_test)
        low = max(2, ne + 1)
        nt = low + (s % (11 - low))
        return nt, ne

    nt = int(num_train)
    high = min(4, nt - 1)
    ne = 1 + (s % high)
    return nt, ne


def _inside(mask, r, c):
    return (
        0 <= r < mask.shape[0]
        and 0 <= c < mask.shape[1]
        and bool(mask[r, c])
    )


def _is_connected(mask):
    starts = np.argwhere(mask)
    if len(starts) == 0:
        return False

    seen = np.zeros(mask.shape, dtype=bool)
    sr, sc = (int(x) for x in starts[0])
    queue = [(sr, sc)]
    seen[sr, sc] = True
    head = 0

    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if _inside(mask, nr, nc) and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return int(np.count_nonzero(seen)) == int(np.count_nonzero(mask))


def _components(open_mask):
    h, w = open_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []

    for sr in range(h):
        for sc in range(w):
            if not open_mask[sr, sc] or seen[sr, sc]:
                continue

            queue = [(sr, sc)]
            seen[sr, sc] = True
            coords = []
            head = 0

            while head < len(queue):
                r, c = queue[head]
                head += 1
                coords.append((r, c))

                for dr, dc in _DIR4:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and open_mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            component = np.zeros((h, w), dtype=bool)
            for r, c in coords:
                component[r, c] = True
            result.append(component)

    return result


def _depths(component):
    """Eight-neighbor erosion depth, with off-canvas positions outside."""
    h, w = component.shape
    depth = np.full((h, w), -1, dtype=np.int16)
    queue = []

    for r, c in np.argwhere(component):
        r, c = int(r), int(c)
        boundary = False
        for dr, dc in _DIR8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w and component[nr, nc]):
                boundary = True
                break
        if boundary:
            depth[r, c] = 0
            queue.append((r, c))

    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        next_depth = int(depth[r, c]) + 1

        for dr, dc in _DIR8:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and component[nr, nc]
                and depth[nr, nc] < 0
            ):
                depth[nr, nc] = next_depth
                queue.append((nr, nc))

    assert np.all(depth[component] >= 0)
    return depth


def _candidate_paths(component, depth, allowed_row_direction=None):
    """
    Return convex-corner diagonal paths.

    Each result is (row, col, dr, dc, usable_length), where the position
    at offset d has contour depth d.
    """
    paths = []

    for r, c in np.argwhere(component & (depth == 0)):
        r, c = int(r), int(c)

        for dr in (-1, 1):
            if allowed_row_direction is not None and dr != allowed_row_direction:
                continue

            for dc in (-1, 1):
                # Inward orthogonal and diagonal cells must be in the region.
                if not (
                    _inside(component, r + dr, c)
                    and _inside(component, r, c + dc)
                    and _inside(component, r + dr, c + dc)
                ):
                    continue

                # The two outward orthogonal cells must be outside the region.
                if _inside(component, r - dr, c):
                    continue
                if _inside(component, r, c - dc):
                    continue

                length = 0
                while True:
                    nr, nc = r + length * dr, c + length * dc
                    if not _inside(component, nr, nc):
                        break
                    if int(depth[nr, nc]) != length:
                        break
                    length += 1

                if length >= 1:
                    paths.append((r, c, dr, dc, length))

    return paths


def _infer_roles(input_grid):
    counts = np.bincount(input_grid.ravel(), minlength=10)
    order = sorted(range(10), key=lambda color: (-int(counts[color]), color))

    neutral = int(order[0])
    divider = int(order[1])
    third = int(order[2])

    assert counts[neutral] > counts[divider] > 0
    assert counts[divider] > counts[third]
    return neutral, divider


def _edge_touch_count(mask):
    return sum(
        (
            bool(np.any(mask[0, :])),
            bool(np.any(mask[-1, :])),
            bool(np.any(mask[:, 0])),
            bool(np.any(mask[:, -1])),
        )
    )


def _divider_has_cycle(divider_mask):
    nodes = int(np.count_nonzero(divider_mask))
    horizontal_edges = int(
        np.count_nonzero(divider_mask[:, :-1] & divider_mask[:, 1:])
    )
    vertical_edges = int(
        np.count_nonzero(divider_mask[:-1, :] & divider_mask[1:, :])
    )
    return horizontal_edges + vertical_edges >= nodes


def _grid_key(grid):
    canonical = grid.astype(np.int8, copy=False)
    return tuple(grid.shape), canonical.tobytes()


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _puzzle_key(puzzle):
    return tuple(
        (
            split,
            tuple(_pair_key(pair) for pair in puzzle[split]),
        )
        for split in ("train", "test")
    )


def _erode_depths(component, diagonal=True):
    """Independent, vectorized erosion used to check the BFS construction."""
    h, w = component.shape
    current = component.copy()
    result = np.full(component.shape, -1, dtype=np.int16)
    level = 0
    neighbors = _DIR8 if diagonal else _DIR4
    while np.any(current):
        padded = np.pad(current, 1)
        interior = current.copy()
        for dr, dc in neighbors:
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[current & ~interior] = level
        current = interior
        level += 1
    return result


def _partition(rng, shape, target, reserve_deep=False):
    """Recursive splits, retaining rectangles for optional attached boxes."""
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    vertical = bool(rng.integers(2))
    extent = w if vertical else h
    cut = int(rng.integers(7, min(12, extent - 6) + 1)) if reserve_deep else int(rng.integers(3, extent - 3))
    if vertical:
        mask[:, cut] = True
        rooms = [(0, h - 1, 0, cut - 1), (0, h - 1, cut + 1, w - 1)]
    else:
        mask[cut, :] = True
        rooms = [(0, cut - 1, 0, w - 1), (cut + 1, h - 1, 0, w - 1)]
    reserved = rooms[0] if reserve_deep else None
    while len(rooms) < target:
        choices = []
        weights = []
        for i, (a, b, c, d) in enumerate(rooms):
            if rooms[i] == reserved:
                continue
            for axis, length in ((0, b - a + 1), (1, d - c + 1)):
                if length >= 7:
                    choices.append((i, axis))
                    weights.append((b - a + 1) * (d - c + 1))
        if not choices:
            break
        weights = np.asarray(weights, dtype=float)
        i, axis = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
        a, b, c, d = rooms.pop(i)
        if axis == 0:
            q = int(rng.integers(a + 3, b - 2))
            mask[q, max(0, c - 1):min(w, d + 2)] = True
            rooms.extend(((a, q - 1, c, d), (q + 1, b, c, d)))
        else:
            q = int(rng.integers(c + 3, d - 2))
            mask[max(0, a - 1):min(h, b + 2), q] = True
            rooms.extend(((a, b, c, q - 1), (a, b, q + 1, d)))
    return mask, rooms


def _attach_boxes(rng, mask, rooms, budget):
    """Add U-shaped walls against existing walls, making holes and concavities."""
    added = 0
    for i in rng.permutation(len(rooms)):
        if added >= budget:
            break
        a, b, c, d = rooms[int(i)]
        options = []
        # Work in either coordinate frame so all four attachment sides occur.
        for transpose in (False, True):
            aa, bb, cc, dd = (c, d, a, b) if transpose else (a, b, c, d)
            work = mask.T if transpose else mask
            if bb - aa + 1 < 9 or dd - cc + 1 < 6:
                continue
            for side in (-1, 1):
                wall = cc - 1 if side == 1 else dd + 1
                if 0 <= wall < work.shape[1] and np.all(work[aa:bb + 1, wall]):
                    options.append((transpose, aa, bb, cc, dd, side, wall))
        if not options:
            continue
        transpose, a, b, c, d, side, wall = options[int(rng.integers(len(options)))]
        work = mask.T if transpose else mask
        top = int(rng.integers(a + 2, b - 5))
        bottom = int(rng.integers(top + 4, b - 1))
        width = int(rng.integers(3, d - c - 1))
        far = wall + side * (width + 1)
        lo, hi = sorted((wall, far))
        work[top, lo:hi + 1] = True
        work[bottom, lo:hi + 1] = True
        work[top:bottom + 1, far] = True
        added += 1


def _step_walls(rng, mask, budget):
    """Displace line intervals without changing segmentation or wall thickness."""
    for _ in range(budget):
        for _attempt in range(70):
            work = mask.T if rng.integers(2) else mask
            points = np.argwhere(work)
            r, c = map(int, points[int(rng.integers(len(points)))])
            end = r + int(rng.integers(4, 12))
            shift = int(rng.choice((-3, -2, -1, 1, 2, 3)))
            lo, hi = sorted((c, c + shift))
            if r < 1 or end + 1 >= work.shape[0] or lo < 1 or hi + 1 >= work.shape[1]:
                continue
            block = work[r - 1:end + 2, lo - 1:hi + 2]
            expected = np.zeros(block.shape, dtype=bool)
            expected[:, c - lo + 1] = True
            if not np.array_equal(block, expected):
                continue
            work[r + 1:end, c] = False
            work[r, lo:hi + 1] = True
            work[end, lo:hi + 1] = True
            work[r:end + 1, c + shift] = True
            break


def _layout(rng, shape, target, regime, first=False):
    box_budget = int(rng.integers(0, min(3, target - 2) + 1))
    mask, rooms = _partition(rng, shape, max(2, target - box_budget), first)
    _attach_boxes(rng, mask, rooms, box_budget)
    _step_walls(rng, mask, int(rng.integers(1 if first else 0, 4 if regime == 'source_train' else 6)))
    # Reflect after composing the network, without changing the requested shape.
    if rng.integers(2):
        mask = mask[::-1].copy()
    if rng.integers(2):
        mask = mask[:, ::-1].copy()
    return mask


def _fallback_mask(shape, first):
    """Guaranteed construction after a bounded search for a richer layout."""
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if first:
        mask[:, 8] = True
        mask[6, 8:] = True
        mask[11:h - 3, 8] = False
        mask[10, 6:9] = True
        mask[h - 3, 6:9] = True
        mask[10:h - 2, 6] = True
    else:
        mask[:, w // 2] = True
    return mask


def _sample_period(rng, neutral, divider, length, gap=None):
    palette = [c for c in range(10) if c not in (neutral, divider)]
    period = [int(c) for c in rng.choice(palette, size=length, replace=False)]
    if gap is not None:
        period[gap] = neutral
    elif length >= 2 and rng.random() < 0.3:
        period[int(rng.integers(length - 1))] = neutral
    # The final visible cue unambiguously terminates the period.
    return tuple(period)


def _is_concave(component):
    rows, cols = np.nonzero(component)
    return int(component.sum()) < (int(rows.max() - rows.min()) + 1) * (int(cols.max() - cols.min()) + 1)


def _fill_pair(rng, mask, neutral, divider, regime, index):
    components = _components(~mask)
    depths = [_depths(c) for c in components]
    paths = [_candidate_paths(c, d, 1 if regime == 'source_train' else None)
             for c, d in zip(components, depths)]
    assert all(paths)
    capacity = [max(p[4] for p in ps) for ps in paths]
    specifications = {}
    if regime == 'source_train' and index == 0:
        assert 3 <= len(components) <= 6
        big = max(range(len(components)), key=lambda i: capacity[i])
        assert capacity[big] >= 4
        specifications[big] = (3, 1, None)
        others = [i for i in range(len(components)) if i != big and capacity[i] >= 2]
        assert others
        two = max(others, key=lambda i: (_is_concave(components[i]), capacity[i]))
        specifications[two] = (2, None, None)
        one = next(i for i in range(len(components)) if i not in specifications)
        specifications[one] = (1, None, None)
    elif regime == 'source_train' and index == 1:
        lead = max(range(len(components)), key=lambda i: capacity[i])
        assert capacity[lead] >= 2
        specifications[lead] = (2, 0, None)
        blank = next(i for i in range(len(components)) if i != lead)
        specifications[blank] = (0, None, None)
    elif regime == 'target_test':
        upward = [(i, p) for i, ps in enumerate(paths) for p in ps if p[2] == -1 and p[4] >= 2]
        assert upward
        i, path = upward[int(rng.integers(len(upward)))]
        specifications[i] = (int(rng.integers(2, min(4, path[4]) + 1)), -1, path)

    inp = np.full(mask.shape, neutral, dtype=np.int8)
    out = inp.copy()
    inp[mask] = out[mask] = divider
    for i, (component, depth, options) in enumerate(zip(components, depths, paths)):
        if i in specifications:
            length, gap, path = specifications[i]
        else:
            length = 0 if rng.random() < 0.15 else int(rng.integers(1, min(3 if regime == 'source_train' else 4, capacity[i]) + 1))
            gap, path = None, None
        if length == 0:
            continue
        available = [p for p in options if p[4] >= length]
        path = path or available[int(rng.integers(len(available)))]
        # -1 requests an entirely explicit key to demonstrate upward reading.
        if gap == -1:
            colors = [c for c in range(10) if c not in (neutral, divider)]
            period = tuple(int(c) for c in rng.choice(colors, length, replace=False))
        else:
            period = _sample_period(rng, neutral, divider, length, gap)
        r, c, dr, dc, _ = path
        for d, color in enumerate(period):
            inp[r + d * dr, c + d * dc] = color
        out[component] = np.asarray(period, dtype=np.int8)[depth[component] % length]
    return {'input': inp, 'output': out}


def _decode_region_key(inp, component, depth, neutral):
    cues = [(int(r), int(c)) for r, c in np.argwhere(component & (inp != neutral))]
    if not cues:
        return {'period': (neutral,), 'directions': (), 'marked': False,
                'component': component, 'depth': depth, 'concave': _is_concave(component)}
    length = max(int(depth[r, c]) for r, c in cues) + 1
    candidates = []
    for r, c, dr, dc, capacity in _candidate_paths(component, depth):
        if capacity >= length and all((rr, cc) == (r + int(depth[rr, cc]) * dr, c + int(depth[rr, cc]) * dc) for rr, cc in cues):
            period = tuple(int(inp[r + d * dr, c + d * dc]) for d in range(length))
            candidates.append((period, dr, dc))
    assert candidates, 'cues must form one inward corner diagonal'
    periods = {x[0] for x in candidates}
    assert len(periods) == 1, 'key must determine exactly one period'
    period = next(iter(periods))
    assert period[-1] != neutral
    return {'period': period, 'directions': tuple(sorted({(x[1], x[2]) for x in candidates})),
            'marked': True, 'component': component, 'depth': depth, 'concave': _is_concave(component)}


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    neutral, divider = _infer_roles(inp)
    mask = inp == divider
    assert _is_connected(mask)
    assert _edge_touch_count(mask) >= 2
    assert not np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:])
    assert np.array_equal(out == divider, mask)
    predicted = inp.copy()
    regions = []
    for component in _components(~mask):
        depth = _erode_depths(component)
        info = _decode_region_key(inp, component, depth, neutral)
        period = np.asarray(info['period'], dtype=np.int8)
        predicted[component] = period[depth[component] % len(period)]
        regions.append(info)
    assert np.array_equal(predicted, out), 'output disagrees with input-only contour decoding'
    assert not np.array_equal(inp, out)
    return {'neutral': neutral, 'divider': divider, 'divider_mask': mask,
            'regions': regions, 'shape': inp.shape}


def _contour_witness(analysis):
    """A colored concavity must distinguish erosion from rectangular or L1 rings."""
    for region in analysis['regions']:
        period = np.asarray(region['period'])
        if not region['concave'] or len(period) < 2:
            continue
        mask, depth = region['component'], region['depth']
        rr, cc = np.nonzero(mask)
        box_depth = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
        manhattan = _erode_depths(mask, diagonal=False)[mask]
        actual = period[depth[mask] % len(period)]
        if np.any(actual != period[box_depth % len(period)]) and np.any(actual != period[manhattan % len(period)]):
            return True
    return False


def _validate_regime(analysis, regime):
    regions = analysis['regions']
    assert 2 <= len(regions) <= (6 if regime == 'source_train' else 10)
    for info in regions:
        assert len(info['period']) <= (3 if regime == 'source_train' else 4)
        if regime == 'source_train' and info['marked']:
            assert any(dr == 1 for dr, _ in info['directions'])
    if regime == 'target_test':
        assert any(len(info['period']) >= 2 and info['period'][0] != analysis['neutral']
                   and info['directions'] and all(dr == -1 for dr, _ in info['directions'])
                   for info in regions)


def _source_fingerprint(pair):
    inp, out = pair['input'], pair['output']
    return hashlib.sha256(bytes(inp.shape) + inp.astype(np.int8).tobytes() + out.astype(np.int8).tobytes()).hexdigest()


def _role_schedule(rng, count):
    neutral = [int(c) for c in rng.permutation(10)]
    divider = None
    for _ in range(24):
        candidate = [int(c) for c in rng.permutation(10)]
        if all(a != b for a, b in zip(neutral, candidate)):
            divider = candidate
            break
    if divider is None:
        divider = neutral[1:] + neutral[:1]
    return [(neutral[i % 10], divider[i % 10]) for i in range(count)]


def _shape_schedule(rng, train_count, test_count):
    shapes = []
    for i in range(train_count + test_count):
        if i == 0:
            candidates = [(h, w) for h in range(18, 29) for w in range(18, 29)]
        elif i < train_count:
            candidates = [(h, w) for h in range(9, 29) for w in range(9, 29)]
        else:
            candidates = [(h, w) for h in range(12, 31) for w in range(12, 31)]
            if (30, 30) not in shapes and rng.random() < 0.25:
                shapes.append((30, 30))
                continue
        candidates = [s for s in candidates if s not in shapes]
        shapes.append(candidates[int(rng.integers(len(candidates)))])
    return shapes


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt, ne = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Shared convention: roles are local; corner keys encode erosion-depth cycles.
    # These schedules balance visible examples, not hidden output conventions.
    roles = _role_schedule(rng, nt + ne)
    shapes = _shape_schedule(rng, nt, ne)
    puzzle = {'train': [], 'test': []}
    first_count = None
    for i, (shape, (neutral, divider)) in enumerate(zip(shapes, roles)):
        regime = 'source_train' if i < nt else 'target_test'
        index = i if i < nt else i - nt
        first = i == 0
        targets = list(range(3 if first else 2, 7 if i < nt else 11))
        if i == 1:
            targets.remove(first_count)
        target = int(rng.choice(targets))
        # Search is bounded; the final geometry is constructively guaranteed.
        for attempt in range(25):
            mask = _layout(rng, shape, target, regime, first) if attempt < 24 else _fallback_mask(shape, first)
            try:
                pair = _fill_pair(rng, mask, neutral, divider, regime, index)
                analysis = _analyze_pair(pair['input'], pair['output'])
                _validate_regime(analysis, regime)
                if first:
                    assert _contour_witness(analysis)
                if i == 1:
                    assert len(analysis['regions']) != first_count
                assert _source_fingerprint(pair) not in _SOURCE_FINGERPRINTS
            except AssertionError:
                if attempt == 24:
                    raise
                continue
            break
        if first:
            first_count = len(analysis['regions'])
        puzzle['train' if i < nt else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            ikey, pkey = _grid_key(pair['input']), _pair_key(pair)
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            assert _source_fingerprint(pair) not in _SOURCE_FINGERPRINTS
            analysis = _analyze_pair(pair['input'], pair['output'])
            _validate_regime(analysis, 'source_train' if split == 'train' else 'target_test')
            analyses[split].append(analysis)
    all_analyses = analyses['train'] + analyses['test']
    for feature in ('neutral', 'divider', 'shape'):
        assert len({a[feature] for a in all_analyses}) >= 3
    assert len({len(a['regions']) for a in all_analyses}) >= 2
    train_regions = [(a['neutral'], r) for a in analyses['train'] for r in a['regions']]
    for length in (1, 2, 3):
        assert any(r['marked'] and len(r['period']) == length for _, r in train_regions)
    assert any(not r['marked'] for _, r in train_regions)
    assert any(r['marked'] and len(r['period']) >= 2 and r['period'][0] == n for n, r in train_regions)
    assert any(n in r['period'][1:-1] for n, r in train_regions)
    assert any(len(r['period']) >= 3 and int(r['depth'].max()) >= len(r['period']) for _, r in train_regions)
    assert any(_contour_witness(a) for a in analyses['train'])


def self_test() -> None:
    counts_train, counts_test, shapes, roles, geometry, combinations = (set() for _ in range(6))
    test_regions, train_regions, sizes, directions, lengths = (set() for _ in range(5))
    complete, cycles, concavities, combined_complex = set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle['train']))
        counts_test.add(len(puzzle['test']))
        complete.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze_pair(pair['input'], pair['output'])
                mask, regions = a['divider_mask'], a['regions']
                shapes.add(a['shape'])
                sizes.add(int(pair['input'].size))
                roles.add((a['neutral'], a['divider']))
                signature = _grid_key(mask.astype(np.int8))
                geometry.add(signature)
                combinations.add((a['shape'], len(regions), signature))
                (test_regions if split == 'test' else train_regions).add(len(regions))
                cyclic = _divider_has_cycle(mask)
                concave = sum(r['concave'] for r in regions)
                cycles += cyclic
                concavities += concave
                combined_complex += bool(cyclic and concave and len(regions) >= 5)
                for region in regions:
                    assert np.array_equal(_depths(region['component']), region['depth'])
                    directions.update(region['directions'])
                    lengths.add(len(region['period']))
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(complete) == 42 and len(roles) >= 40
    assert len(shapes) >= 80 and (30, 30) in shapes and min(sizes) <= 120
    assert len(geometry) >= 100 and len(combinations) >= 100
    assert len(test_regions) >= 7 and len(train_regions) >= 4
    assert cycles >= 15 and concavities >= 80 and combined_complex >= 10
    assert directions == set((dr, dc) for dr in (-1, 1) for dc in (-1, 1))
    assert lengths == {1, 2, 3, 4}
    for nt, ne in ((None, None), (2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        first = generate_puzzle(9173, nt, ne)
        second = generate_puzzle(9173, nt, ne)
        assert _puzzle_key(first) == _puzzle_key(second)
        assert _puzzle_key(first) != _puzzle_key(generate_puzzle(9174, nt, ne))
        if nt is not None:
            assert len(first['train']) == nt
        if ne is not None:
            assert len(first['test']) == ne
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(300, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Corruptions test the returned-data validator rather than generator metadata.
    original = generate_puzzle(81, 2, 1)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    a = _analyze_pair(broken['test'][0]['input'], broken['test'][0]['output'])
    r, c = map(int, np.argwhere(a['divider_mask'])[0])
    broken['test'][0]['output'][r, c] = a['neutral']
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        return
    raise AssertionError('changed divider accepted')
