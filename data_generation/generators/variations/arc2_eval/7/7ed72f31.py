import numpy as np


def _source_signature(rows):
    return f"{len(rows)}x{len(rows[0])}:" + "".join(rows)


_SOURCE_INPUT_SIGNATURES = frozenset(
    {
        _source_signature(
            (
                "111111111111111111",
                "111111111111111111",
                "111131111111111111",
                "111331111111111111",
                "111112111111111111",
                "111111111114412111",
                "111111111111442111",
                "111111111111142111",
                "111111111111142111",
                "111111111111111111",
                "111111111111111111",
                "111111251111111111",
                "111111551111111111",
                "111111115111181111",
                "111111111118818811",
                "111111111112222211",
                "111111111111111111",
                "111111111111111111",
            )
        ),
        _source_signature(
            (
                "888888888888888888",
                "888888888888888888",
                "888888888888888888",
                "888824448881128888",
                "888824848888128888",
                "888824448888888888",
                "888824888888888888",
                "888888888888888888",
                "888888888838888888",
                "888888888383888888",
                "888888883383888888",
                "888888882222888888",
                "888888888888888888",
                "888888888888888888",
                "888888888888888888",
                "888888888888888888",
                "888888888888888888",
                "888888888888888888",
            )
        ),
        _source_signature(
            (
                "888888888888888888",
                "888888888888448888",
                "888888888888444888",
                "888888888888444888",
                "888888888882888888",
                "888888888888888888",
                "888888888888888888",
                "888288888888888888",
                "111888888888888888",
                "881888888888888888",
                "881888888888888888",
                "888888888888888888",
                "888888888888888888",
                "888888822222288888",
                "888888833333388888",
                "888888883383888888",
                "888888888338888888",
                "888888888883888888",
            )
        ),
        _source_signature(
            (
                "3333333333333333311133333333",
                "3333333333333333311333333333",
                "3333333333333333313233333333",
                "3333333344433333333333333333",
                "3333334443433333333333333333",
                "3333334334333333333333333333",
                "3333332222233333333333333333",
                "3333333333333333333333333333",
                "3333333333333333333333333333",
                "3333333333333333333283333333",
                "3333333333333333333288333333",
                "3333333333333333333283833333",
                "3333333333323333333288833333",
                "3333333338833333333233833333",
                "3333333388833333333288333333",
                "3333333338333333333333333333",
                "3333333333333333333333333333",
                "3333231333333333333336333333",
                "3333213333333333333363633333",
                "3333213333333333333336633333",
                "3333211333333333333333323333",
                "3333331333333333333333333333",
                "3333333333333333333333333333",
                "3333333333333333334333333333",
                "3333333333333333343333333333",
                "3333333333333333233333333333",
                "3333333333333333333333333333",
                "3333333333333333333333333333",
            )
        ),
    }
)


def _array_text_signature(grid):
    rows, cols = grid.shape
    body = "".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )
    return f"{rows}x{cols}:{body}"


def _grid_fingerprint(grid):
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _puzzles_equal(first, second):
    if set(first) != {"train", "test"}:
        return False
    if set(second) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for pair_a, pair_b in zip(first[split], second[split]):
            if not np.array_equal(pair_a["input"], pair_b["input"]):
                return False
            if not np.array_equal(pair_a["output"], pair_b["output"]):
                return False
    return True


def _validate_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _validate_optional_count(
        num_train, "num_train", 2, 10
    )
    num_test = _validate_optional_count(
        num_test, "num_test", 1, 4
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        choices = 10 - num_test
        num_train = num_test + 1 + (seed // 4) % choices
    elif num_train is None:
        choices = 10 - num_test
        num_train = num_test + 1 + seed % choices
    elif num_test is None:
        largest_test = min(4, num_train - 1)
        num_test = 1 + seed % largest_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(sorted(component))
    return result


def _bbox(cells):
    a = np.array(cells)
    return int(a[:, 0].min()), int(a[:, 0].max()), int(a[:, 1].min()), int(a[:, 1].max())


def _infer_background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    return int(colors[counts.argmax()])


def _classify_marker(cells):
    r0, r1, c0, c1 = _bbox(cells)
    if len(cells) == 1:
        return 'P'
    if r0 == r1 and len(cells) == c1 - c0 + 1:
        return 'H'
    if c0 == c1 and len(cells) == r1 - r0 + 1:
        return 'V'
    raise AssertionError('Marker must be a singleton or contiguous straight segment')


def _candidate_mapping(marker, kind, payload, grid, background):
    # Eligibility uses observable contact, geometry, and a clear destination.
    # No nearest-marker tie-breaking or generation-time ownership is consulted.
    distance = min(max(abs(r - a), abs(c - b))
                   for r, c in payload for a, b in marker)
    if distance != 1:
        return None
    r0, r1, c0, c1 = _bbox(payload)
    ar, ac = marker[0]
    if kind == 'H':
        if not (r1 < ar or r0 > ar):
            return None
        if c1 < min(c for _, c in marker) or c0 > max(c for _, c in marker):
            return None
        dest = [(2 * ar - r, c) for r, c in payload]
    elif kind == 'V':
        if not (c1 < ac or c0 > ac):
            return None
        if r1 < min(r for r, _ in marker) or r0 > max(r for r, _ in marker):
            return None
        dest = [(r, 2 * ac - c) for r, c in payload]
    else:
        # Closed quadrants include the source's row/column-aligned cells.
        if not ((r1 <= ar or r0 >= ar) and (c1 <= ac or c0 >= ac)):
            return None
        dest = [(2 * ar - r, 2 * ac - c) for r, c in payload]
    h, w = grid.shape
    if any(not (0 <= r < h and 0 <= c < w) or grid[r, c] != background
           for r, c in dest):
        return None
    return dest


def _read_input(grid, marker_color):
    background = _infer_background(grid)
    assert marker_color != background
    markers = _components(grid == marker_color)
    payloads = _components((grid != background) & (grid != marker_color))
    assert markers and len(markers) == len(payloads)
    assert all(len({int(grid[r, c]) for r, c in p}) == 1 for p in payloads)
    groups, used, destinations = [], set(), set()
    for marker in markers:
        kind = _classify_marker(marker)
        candidates = []
        for index, payload in enumerate(payloads):
            dest = _candidate_mapping(marker, kind, payload, grid, background)
            if dest is not None:
                candidates.append((index, dest))
        assert len(candidates) == 1, 'Marker has ambiguous or missing payload ownership'
        index, dest = candidates[0]
        assert index not in used, 'A payload belongs to multiple markers'
        assert not destinations.intersection(dest), 'Copies collide'
        used.add(index)
        destinations.update(dest)
        groups.append({'kind': kind, 'marker': marker, 'payload': payloads[index],
                       'destination': dest, 'color': int(grid[payloads[index][0]])})
    assert len(used) == len(payloads)
    return background, groups


def _predict_with_marker(grid, marker):
    _, groups = _read_input(grid, marker)
    predicted = grid.copy()
    for group in groups:
        for (r, c), (rr, cc) in zip(group['payload'], group['destination']):
            predicted[rr, cc] = grid[r, c]
    return predicted


def _infer_marker_from_train(puzzle):
    candidates = set(range(10))
    for pair in puzzle['train']:
        inp, out = pair['input'], pair['output']
        assert inp.shape == out.shape
        bg = _infer_background(inp)
        assert np.array_equal(inp[inp != bg], out[inp != bg]), 'Input cells changed'
        added = set(map(int, np.unique(out[inp != out])))
        # All payload colors are copied; markers alone are preserved without additions.
        candidates &= set(map(int, np.unique(inp))) - {bg} - added
    good = []
    for marker in sorted(candidates):
        try:
            if all(np.array_equal(_predict_with_marker(p['input'], marker), p['output'])
                   for p in puzzle['train']):
                good.append(marker)
        except AssertionError:
            continue
    assert len(good) == 1, 'Training does not identify one marker role'
    return good[0]


def solve_from_training(train_pairs, test_input):
    """Prediction boundary deliberately excludes test labels, RNG, and metadata."""
    marker = _infer_marker_from_train({'train': train_pairs})
    return _predict_with_marker(test_input, marker)


def _translation_destination(group):
    r0, r1, c0, c1 = _bbox(group['payload'])
    a0, _, b0, _ = _bbox(group['destination'])
    return [(r + a0 - r0, c + b0 - c0) for r, c in group['payload']]


def _orientation_witness(group):
    return set(_translation_destination(group)) != set(group['destination'])


def _trim(mask):
    r, c = np.nonzero(mask)
    return mask[r.min():r.max()+1, c.min():c.max()+1].copy()


def _silhouette(rng, h, w, witness=False):
    """Compositional eight-connected silhouettes, including intentionally simple ones."""
    family = int(rng.choice(6, p=[.12, .22, .20, .15, .15, .16]))
    if family != 0:
        h, w = max(2, h), max(2, w)
    if family in (2, 5):
        h, w = max(3, h), max(3, w)
    if witness:
        h, w = max(2, h), max(2, w)
    mask = np.zeros((h, w), dtype=bool)
    if family == 0:
        mask[:] = True
    elif family == 1:
        # An eight-neighbor growth process can form concavities and diagonal bridges.
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
        target = int(rng.integers(max(2, (h*w)//4), h*w + 1)) if h*w > 1 else 1
        for _ in range(target - 1):
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        a, b = int(r+dr), int(c+dc)
                        if 0 <= a < h and 0 <= b < w and not mask[a, b]:
                            frontier.add((a, b))
            if not frontier:
                break
            choices = sorted(frontier)
            mask[choices[int(rng.integers(len(choices)))]] = True
    elif family == 2:
        # Three or four separated diagonal arms with independent lengths form
        # a sparse eight-connected branch without dense king-neighbor chords.
        row, col = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        mask[row, col] = True
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        rng.shuffle(directions)
        for dr, dc in directions[:int(rng.integers(3, 5))]:
            reach = min(row if dr < 0 else h-1-row, col if dc < 0 else w-1-col)
            for step in range(1, int(rng.integers(1, reach+1))+1):
                mask[row+dr*step, col+dc*step] = True
    elif family == 3:
        # A thin zigzag spine; each step touches diagonally.
        if h < w:
            mask = mask.T
        hh, ww = mask.shape
        col = int(rng.integers(ww))
        for row in range(hh):
            mask[row, col] = True
            step = int(rng.choice([-1, 1]))
            if not 0 <= col + step < ww:
                step = -step
            col += step
        if h < w:
            mask = mask.T
    elif family == 4:
        mask[:] = True
        if h > 1 and w > 1:
            # Unequal arms of an L/U-like concavity; the cut is open to an edge.
            cut_h = int(rng.integers(1, h))
            cut_w = int(rng.integers(1, w))
            mask[:cut_h, :cut_w] = False
    else:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        if h > 3 and w > 3 and rng.random() < .5:
            mask[1, int(rng.integers(1, w-1))] = True
    if rng.random() < .5:
        mask = mask[::-1, :]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    mask = _trim(mask)
    if witness and np.array_equal(mask, mask[::-1, ::-1]):
        # A variable-size asymmetric concavity supplies a visible orientation witness.
        mask = np.ones((max(2, h), max(2, w)), dtype=bool)
        mask[0, 0] = False
        if rng.random() < .5:
            mask = mask[::-1, :]
    return mask.copy()


def _prototype(rng, kind, scale, witness=False, compact=False):
    if compact:
        mask = np.array([[1, 1], [1, 0]], dtype=bool)
    else:
        h = int(rng.integers(1, scale + 1))
        w = int(rng.integers(1, min(9, scale + 3)))
        mask = _silhouette(rng, h, w, witness)
    if kind == 'P':
        # Both strict diagonal centers and centers aligned with a bbox edge occur.
        offsets = [(1, 1), (0, 1), (1, 0), (0, 0)]
        dr, dc = offsets[int(rng.integers(4))]
        cells = set(map(tuple, np.argwhere(mask).tolist()))
        cells.add((0, 0))
        # Attach the corner to the existing eight-connected shape along a short path.
        target = min(cells - {(0, 0)}, default=(0, 0))
        r = c = 0
        while (r, c) != target:
            r += int(np.sign(target[0] - r))
            c += int(np.sign(target[1] - c))
            cells.add((r, c))
        payload = {(r + dr, c + dc) for r, c in cells} - {(0, 0)}
        if not payload:
            payload = {(1, 1), (1, 2)}
        # Removing the center can split a very thin corner; use a local diagonal bridge.
        arr = np.zeros((max(r for r, _ in payload)+1, max(c for _, c in payload)+1), bool)
        for q in payload:
            arr[q] = True
        if len(_components(arr)) != 1:
            payload.add((1, 1))
        sr, sc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
        payload = sorted((sr*r, sc*c) for r, c in payload)
        marker = [(0, 0)]
        destination = [(-r, -c) for r, c in payload]
    else:
        # Construct horizontally, then transpose for vertical axes.
        h, w = mask.shape
        near = np.flatnonzero(mask[0])
        if not len(near):
            raise AssertionError('Trimmed mask must reach its nearest row')
        contact = int(rng.choice(near))
        a = int(rng.integers(-2, contact+1))
        b = int(rng.integers(max(a+1, contact), max(a+2, w+2)))
        marker = [(0, c) for c in range(a, b+1)]
        sign = int(rng.choice([-1, 1]))
        payload = [(sign*(int(r)+1), int(c)) for r, c in np.argwhere(mask)]
        destination = [(-r, c) for r, c in payload]
        if kind == 'V':
            marker = [(c, r) for r, c in marker]
            payload = [(c, r) for r, c in payload]
            destination = [(c, r) for r, c in destination]
    group = {'kind': kind, 'marker': marker, 'payload': payload, 'destination': destination}
    if witness and not _orientation_witness(group):
        # Retry boundedly with a guaranteed asymmetric mask and strict diagonal point.
        if kind == 'P':
            group = {'kind': kind, 'marker': [(0, 0)],
                     'payload': [(1, 1), (1, 2), (2, 1)],
                     'destination': [(-1, -1), (-1, -2), (-2, -1)]}
        else:
            group = {'kind': 'H', 'marker': [(0, 0), (0, 1)],
                     'payload': [(1, 0), (1, 1), (2, 0)],
                     'destination': [(-1, 0), (-1, 1), (-2, 0)]}
            if kind == 'V':
                group = {key: (kind if key == 'kind' else [(c, r) for r, c in val])
                         for key, val in group.items()}
    footprint = group['marker'] + group['payload'] + group['destination']
    r0, _, c0, _ = _bbox(footprint)
    for key in ('marker', 'payload', 'destination'):
        group[key] = [(r-r0, c-c0) for r, c in group[key]]
    return group


def sample_puzzle_parameters(rng):
    order = [int(c) for c in rng.permutation(10)]
    modes = [('P',), ('H',), ('V',), ('H', 'V'), ('H', 'P'), ('V', 'P'), ('H', 'V', 'P')]
    kinds = modes[int(rng.choice(len(modes), p=[.18, .06, .06, .20, .12, .12, .26]))]
    palette_size = int(rng.integers(1, 7))
    return {'marker': order[0], 'backgrounds': order[1:],
            'payloads': order[3:3+palette_size], 'novel': order[9],
            'kinds': kinds, 'steady_background': bool(rng.random() < .30),
            'steady_palette': bool(rng.random() < .27),
            'familiar_shape': ((18, 18) if rng.random() < .25 else
                               (int(rng.integers(10, 27)), int(rng.integers(10, 27)))),
            'richness': int(rng.integers(3))}


def _pack(rng, prototypes, height, width, style, deterministic=False):
    blocked = np.zeros((height, width), bool)
    placed = []
    anchor = (int(rng.integers(height)), int(rng.integers(width)))
    order = sorted(range(len(prototypes)), key=lambda i: -len(
        prototypes[i]['marker'] + prototypes[i]['payload'] + prototypes[i]['destination']))
    for index in order:
        g = prototypes[index]
        footprint = sorted(set(g['marker'] + g['payload'] + g['destination']))
        a = np.array(footprint)
        gh, gw = a.max(axis=0) + 1
        if gh > height or gw > width:
            return None
        coords = [(r, c) for r in range(height-gh+1) for c in range(width-gw+1)]
        if not deterministic:
            rng.shuffle(coords)
            if style == 'edge':
                coords.sort(key=lambda q: min(q[0], q[1], height-gh-q[0], width-gw-q[1]))
            elif style == 'cluster':
                coords.sort(key=lambda q: abs(q[0]+gh/2-anchor[0]) + abs(q[1]+gw/2-anchor[1]))
        accepted = None
        for r, c in coords:
            rr, cc = a[:, 0]+r, a[:, 1]+c
            if not blocked[rr, cc].any():
                accepted = (r, c)
                break
        if accepted is None:
            return None
        r, c = accepted
        moved = {key: (val if key == 'kind' else [(x+r, y+c) for x, y in val])
                 for key, val in g.items()}
        moved['index'] = index
        placed.append(moved)
        for x, y in footprint:
            x, y = x+r, y+c
            blocked[max(0,x-1):min(height,x+2), max(0,y-1):min(width,y+2)] = True
    return placed


def generate_pair(rng, regime, parameters, specs, force_shape=None, seen_colors=()):
    assert regime in ('source_train', 'target_test')
    if force_shape is not None:
        height, width = force_shape
    elif rng.random() < .25:
        height, width = parameters['familiar_shape']
    else:
        size_class = int(rng.choice(3, p=[.25, .45, .30]))
        low, high = [(8, 17), (14, 25), (21, 31)][size_class]
        height, width = int(rng.integers(low, high)), int(rng.integers(low, high))
    style = str(rng.choice(['free', 'cluster', 'edge'], p=[.45, .25, .30]))
    scale = int(rng.integers(2, 8))
    # Packing is bounded; reduce scale and finally use a compact 30x30 free scan.
    placed = None
    for attempt in range(16):
        compact = attempt == 15
        if attempt in (5, 10, 15):
            height, width = min(30, height+5), min(30, width+5)
        if compact:
            height = width = 30
        cap = max(2, scale - attempt//4)
        prototypes = [_prototype(rng, kind, cap, witness, compact) for kind, witness in specs]
        placed = _pack(rng, prototypes, height, width, style, deterministic=compact)
        if placed is not None:
            break
    assert placed is not None, 'Compact deterministic packing must fit'
    background = parameters['backgrounds'][0] if parameters['steady_background'] else int(rng.choice(parameters['backgrounds']))
    pool = list(parameters['payloads'])
    if regime == 'target_test' and seen_colors and rng.random() < .55:
        pool = sorted(seen_colors)
    elif regime == 'target_test' and rng.random() < .65:
        pool = sorted(set(pool + [parameters['novel']]))
    if pool == [background]:
        background = parameters['backgrounds'][0]
    pool = [c for c in pool if c != background]
    if not parameters['steady_palette']:
        pool = [int(c) for c in rng.choice(pool, size=int(rng.integers(1, len(pool)+1)), replace=False)]
    colors = [pool[i % len(pool)] for i in range(len(specs))]
    rng.shuffle(colors)
    grid = np.full((height, width), background, dtype=np.int8)
    output = grid.copy()
    for g in placed:
        color = colors[g['index']]
        for r, c in g['marker']:
            grid[r, c] = output[r, c] = parameters['marker']
        for r, c in g['payload']:
            grid[r, c] = output[r, c] = color
        for r, c in g['destination']:
            output[r, c] = color
    return {'input': grid, 'output': output}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert pair['input'].shape == pair['output'].shape
            assert not np.array_equal(pair['input'], pair['output'])
            key = _grid_fingerprint(pair['input'])
            assert key not in seen
            seen.add(key)
            assert _array_text_signature(pair['input']) not in _SOURCE_INPUT_SIGNATURES
    marker = _infer_marker_from_train(puzzle)
    training_kinds, witnessed = set(), set()
    shapes, counts = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            inp, out = pair['input'], pair['output']
            bg, groups = _read_input(inp, marker)
            shapes.add(inp.shape)
            counts.add(len(groups))
            assert np.array_equal(inp[inp != bg], out[inp != bg])
            assert np.array_equal(_predict_with_marker(inp, marker), out)
            assert np.count_nonzero(inp != out) == sum(len(g['payload']) for g in groups)
            if split == 'train':
                training_kinds.update(g['kind'] for g in groups)
                witnessed.update(g['kind'] for g in groups if _orientation_witness(g))
            else:
                assert {g['kind'] for g in groups} <= training_kinds
    # Training alone must refute copying an unturned payload to its opposite bbox.
    assert training_kinds <= witnessed, 'Training conceals an operation behind symmetric payloads'
    assert len(shapes) >= 2 and len(counts) >= 2, 'Scene composition must vary within a puzzle'


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    parameters = sample_puzzle_parameters(rng)
    kinds = parameters['kinds']
    plans = [[] for _ in range(train_count)]
    # Distribute demonstrations without pinning a kind to a particular training row.
    for kind in rng.permutation(kinds):
        plans[int(rng.integers(train_count))].append((str(kind), True))
    puzzle = {'train': [], 'test': []}
    seen, seen_colors = set(), set()
    observed_counts, observed_shapes = set(), set()
    for split, count, regime in [('train', train_count, 'source_train'), ('test', test_count, 'target_test')]:
        for i in range(count):
            if split == 'train':
                specs = list(plans[i])
            else:
                specs = []
            max_count = (4, 7, 10)[parameters['richness']]
            desired = max(len(specs), int(rng.integers(1, max_count)))
            last_pair = split == 'test' and i == count - 1
            if last_pair and observed_counts == {desired}:
                desired = desired + 1 if desired < max_count - 1 else desired - 1
            pair_kinds = kinds if rng.random() < .60 else (str(rng.choice(kinds)),)
            while len(specs) < desired:
                specs.append((str(rng.choice(pair_kinds)), False))
            rng.shuffle(specs)
            force_shape = None
            if split == 'test' and rng.random() < .4:
                force_shape = puzzle['train'][int(rng.integers(train_count))]['input'].shape
            for attempt in range(24):
                pair = generate_pair(rng, regime, parameters, specs, force_shape, seen_colors)
                key = _grid_fingerprint(pair['input'])
                if last_pair and observed_shapes == {pair['input'].shape}:
                    h, w = pair['input'].shape
                    force_shape = (30 if h < 30 else 29, 30 if w < 30 else 29)
                    continue
                if key in seen or _array_text_signature(pair['input']) in _SOURCE_INPUT_SIGNATURES:
                    continue
                # Inspect actual geometry before accepting; no private ownership used.
                try:
                    _read_input(pair['input'], parameters['marker'])
                except AssertionError:
                    continue
                break
            else:
                raise AssertionError('Could not construct a unique readable scene')
            puzzle[split].append(pair)
            seen.add(key)
            observed_counts.add(len(specs))
            observed_shapes.add(pair['input'].shape)
            if split == 'train':
                seen_colors.update(g['color'] for g in _read_input(pair['input'], parameters['marker'])[1])
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_hole(mask):
    # Four-connected empty regions not touching the padded exterior are holes.
    padded = np.pad(mask, 1)
    return len(_components(~padded, diagonal=False)) > 1


def measure_pair(pair, marker):
    inp, out = pair['input'], pair['output']
    bg, groups = _read_input(inp, marker)
    details = []
    for g in groups:
        cells = g['payload']
        r0, r1, c0, c1 = _bbox(cells)
        mask = np.zeros((r1-r0+1, c1-c0+1), bool)
        for r, c in cells:
            mask[r-r0, c-c0] = True
        ar, ac = g['marker'][0]
        overhang = longer = before = after = False
        offset = False
        if g['kind'] == 'P':
            offset = any(r == ar or c == ac for r, c in cells)
            side = (int(np.sign(sum(r-ar for r, c in cells))), int(np.sign(sum(c-ac for r, c in cells))))
            depth = max(max(abs(r-ar), abs(c-ac)) for r, c in cells)
        else:
            coord = 1 if g['kind'] == 'H' else 0
            lo, hi = min(p[coord] for p in cells), max(p[coord] for p in cells)
            ml, mh = min(p[coord] for p in g['marker']), max(p[coord] for p in g['marker'])
            overhang, longer = lo < ml or hi > mh, ml < lo or mh > hi
            before, after = lo < ml, hi > mh
            depth = max(abs((r-ar) if coord else (c-ac)) for r, c in cells)
            side = int(np.sign((cells[0][0]-ar) if coord else (cells[0][1]-ac)))
        neighbors = [[q for q in cells if q != p and max(abs(q[0]-p[0]), abs(q[1]-p[1])) == 1]
                     for p in cells]
        degrees = [len(ns) for ns in neighbors]
        chain = len(cells) >= 3 and max(degrees) <= 2 and degrees.count(1) == 2
        bent = any(len(ns) == 2 and (ns[0][0]+ns[1][0], ns[0][1]+ns[1][1]) != (2*p[0], 2*p[1])
                   for p, ns in zip(cells, neighbors))
        details.append({'kind': g['kind'], 'side': side, 'size': len(cells),
                        'height': mask.shape[0], 'width': mask.shape[1], 'depth': depth,
                        'marker_length': len(g['marker']), 'diagonal': len(_components(mask, False)) > 1,
                        'hole': _has_hole(mask), 'rectangle': bool(mask.all()),
                        'offset_center': offset, 'overhang': overhang, 'longer_marker': longer,
                        'overhang_before': before, 'overhang_after': after,
                        'thin_chain': chain, 'zigzag': chain and bent,
                        'branch': degrees.count(1) >= 3 and max(degrees) >= 3,
                        'sparse': len(cells) <= .45 * mask.size,
                        'witness': _orientation_witness(g),
                        'silhouette': (mask.shape, tuple(map(int, mask.flat)))})
    def edge(grid):
        return bool(np.any(grid[0] != bg) or np.any(grid[-1] != bg)
                    or np.any(grid[:, 0] != bg) or np.any(grid[:, -1] != bg))
    return {'shape': tuple(inp.shape), 'background': bg, 'marker': marker,
            'group_count': len(groups), 'kinds': tuple(sorted({g['kind'] for g in groups})),
            'palette': tuple(sorted({g['color'] for g in groups})),
            'reused_color': len({g['color'] for g in groups}) < len(groups),
            'input_edge': edge(inp), 'output_edge': edge(out), 'groups': details}


def self_test() -> None:
    puzzles, measurements = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        marker = _infer_marker_from_train(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                measurements.append(measure_pair(pair, marker))
                # Check pixel mappings against flips/rotations of cropped masks.
                expected = pair['input'].copy()
                bg, groups = _read_input(pair['input'], marker)
                for g in groups:
                    r0, r1, c0, c1 = _bbox(g['payload'])
                    local = np.zeros((r1-r0+1, c1-c0+1), bool)
                    for r, c in g['payload']:
                        local[r-r0, c-c0] = True
                    ar, ac = g['marker'][0]
                    top, left = r0, c0
                    if g['kind'] == 'H':
                        local, top = np.flipud(local), 2*ar-r1
                    elif g['kind'] == 'V':
                        local, left = np.fliplr(local), 2*ac-c1
                    else:
                        local, top, left = np.rot90(local, 2), 2*ar-r1, 2*ac-c1
                    for r, c in np.argwhere(local):
                        q = (top+int(r), left+int(c))
                        assert pair['input'][q] == bg
                        expected[q] = g['color']
                assert np.array_equal(expected, pair['output'])
        train = [{k: v.copy() for k, v in p.items()} for p in p['train']]
        for pair in p['test']:
            predicted = solve_from_training(train, pair['input'].copy())
            assert np.array_equal(predicted, pair['output'])
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({m['marker'] for m in measurements}) >= 8
    assert len({m['background'] for m in measurements}) >= 8
    assert len({m['shape'] for m in measurements}) >= 60
    assert len({m['group_count'] for m in measurements}) >= 7
    assert len({m['kinds'] for m in measurements}) == 7
    groups = [g for m in measurements for g in m['groups']]
    assert len({g['silhouette'] for g in groups}) >= 150
    for name in ('diagonal', 'hole', 'offset_center', 'overhang', 'longer_marker',
                 'overhang_before', 'overhang_after', 'thin_chain', 'zigzag', 'sparse'):
        assert sum(g[name] for g in groups) >= 15, name
    assert sum(g['branch'] for g in groups) >= 40
    assert sum(g['rectangle'] for g in groups) < .40 * len(groups)
    assert max(g['depth'] for g in groups) >= 6
    assert max(g['marker_length'] for g in groups) >= 8
    assert {g['side'] for g in groups if g['kind'] == 'H'} == {-1, 1}
    assert {g['side'] for g in groups if g['kind'] == 'V'} == {-1, 1}
    assert {(-1, -1), (-1, 1), (1, -1), (1, 1)} <= {
        g['side'] for g in groups if g['kind'] == 'P'}
    for name in ('input_edge', 'output_edge', 'reused_color'):
        assert sum(m[name] for m in measurements) >= 30, name
    assert sum(any(g['diagonal'] for g in m['groups']) and m['reused_color']
               and m['output_edge'] for m in measurements) >= 10
    # Within-puzzle variations, familiar and novel tests, and homogeneous regimes.
    stats = {'familiar': 0, 'novel': 0, 'same_palette': 0, 'same_bg': 0,
             'mixed': 0, 'homogeneous': 0}
    for p in puzzles:
        marker = _infer_marker_from_train(p)
        train = [measure_pair(x, marker) for x in p['train']]
        test = [measure_pair(x, marker) for x in p['test']]
        all_m = train + test
        assert len({m['shape'] for m in all_m}) >= 2
        assert len({m['group_count'] for m in all_m}) >= 2
        assert len({g['silhouette'] for m in all_m for g in m['groups']}) >= 2
        known = set(c for m in train for c in m['palette'])
        stats['familiar'] += sum(set(m['palette']) <= known for m in test)
        stats['novel'] += sum(bool(set(m['palette']) - known) for m in test)
        stats['same_palette'] += len({m['palette'] for m in train}) == 1
        stats['same_bg'] += len({m['background'] for m in all_m}) == 1
        kinds = set(k for m in all_m for k in m['kinds'])
        stats['mixed' if 'P' in kinds and len(kinds) > 1 else 'homogeneous'] += 1
    assert all(v > 0 for v in stats.values()), stats
    assert _puzzles_equal(generate_puzzle(12345, 6, 3), generate_puzzle(12345, 6, 3))
    assert not _puzzles_equal(generate_puzzle(54321, 5, 2), generate_puzzle(54322, 5, 2))
    for i, (nt, ns) in enumerate([(2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)]):
        p = generate_puzzle(7000+i, nt, ns)
        assert nt is None or len(p['train']) == nt
        assert ns is None or len(p['test']) == ns
        validate_generated_puzzle(p)
    for nt, ns in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)]:
        try:
            generate_puzzle(9000, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # A translation-corrupted label must be rejected; only training chooses the witness.
    for p in puzzles:
        marker = _infer_marker_from_train(p)
        found = False
        for i, pair in enumerate(p['train']):
            for g in _read_input(pair['input'], marker)[1]:
                if g['kind'] == 'P' and _orientation_witness(g):
                    bad = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
                    out = bad['train'][i]['output']
                    bg = _infer_background(pair['input'])
                    for q in g['destination']:
                        out[q] = bg
                    for q in _translation_destination(g):
                        out[q] = g['color']
                    try:
                        validate_generated_puzzle(bad)
                    except AssertionError:
                        found = True
                    else:
                        raise AssertionError('Accepted competing point translation')
                    break
            if found:
                break
        if found:
            break
    assert found
    # Exercise the original geometry regressions directly from all four staged
    # source input signatures. Labels are checked separately by the review harness.
    assert len(_SOURCE_INPUT_SIGNATURES) == 4
    source_groups = []
    for signature in sorted(_SOURCE_INPUT_SIGNATURES):
        dims, body = signature.split(':')
        h, w = map(int, dims.split('x'))
        inp = np.array([int(c) for c in body], dtype=np.int8).reshape(h, w)
        source_groups.extend(_read_input(inp, 2)[1])
    assert len(source_groups) == 17
    ambiguous = np.zeros((8, 9), dtype=np.int8)
    ambiguous[3, 3] = ambiguous[3, 5] = 2
    ambiguous[4, 4] = 4
    try:
        _read_input(ambiguous, 2)
    except AssertionError:
        pass
    else:
        raise AssertionError('Accepted a payload adjacent to competing centers')
