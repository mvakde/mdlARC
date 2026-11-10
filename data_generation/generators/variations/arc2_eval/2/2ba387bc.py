import numpy as np


_HOLLOW = "hollow"
_SOLID = "solid"

# Lossless descriptions of the five source inputs, for content comparisons.
# The source uses size-four tiles on zero; its dimensions are not forbidden.
_SOURCE_TILES = (
    ((23, 24), ((1, 2, 1, "s"), (4, 9, 2, "s"), (6, 17, 8, "h"),
                (12, 1, 7, "s"), (13, 8, 3, "h"), (15, 17, 6, "s"), (18, 3, 4, "h"))),
    ((20, 21), ((1, 4, 2, "h"), (3, 11, 1, "s"), (9, 1, 4, "h"),
                (10, 7, 8, "s"), (14, 14, 9, "s"))),
    ((20, 23), ((1, 2, 8, "h"), (3, 11, 4, "s"), (6, 4, 2, "s"), (14, 17, 7, "h"))),
    ((22, 22), ((1, 1, 2, "h"), (3, 14, 8, "h"), (7, 3, 3, "s"),
                (9, 10, 6, "h"), (15, 10, 4, "s"))),
    ((24, 19), ((1, 1, 6, "s"), (4, 8, 4, "h"), (7, 14, 7, "s"),
                (11, 2, 1, "s"), (14, 8, 2, "h"), (19, 3, 8, "s"))),
)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed_value


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed_value: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    test_count = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    # These schedules are deterministic from the seed. Across consecutive
    # seeds they cover all four legal test counts and many train counts.
    phase = seed_value

    if train_count is None and test_count is None:
        test_count = 1 + phase % 4
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = minimum_train + (phase // 4) % span
    elif train_count is None:
        assert test_count is not None
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = minimum_train + phase % span
    else:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count for num_train")
        test_count = 1 + phase % maximum_test

    assert train_count is not None
    assert test_count is not None
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")
    return train_count, test_count


def _tile(size, color, background, kind):
    tile = np.full((size, size), color, dtype=np.int8)
    if kind == _HOLLOW:
        tile[1:-1, 1:-1] = background
    return tile


def _grid_key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _pair_key(pair):
    return _grid_key(pair['input']), _grid_key(pair['output'])


def _source_copy(grid):
    for shape, tiles in _SOURCE_TILES:
        if tuple(grid.shape) != shape:
            continue
        source = np.zeros(shape, dtype=np.int8)
        for r, c, color, kind in tiles:
            source[r:r+4, c:c+4] = _tile(
                4, color, 0, _HOLLOW if kind == 'h' else _SOLID)
        if np.array_equal(grid, source):
            return True
    return False


def _parse_background(grid, background):
    """Segment instances, including repeated colors, by connectivity."""
    height, width = grid.shape
    mask = grid != background
    remaining = set(zip(*np.nonzero(mask)))
    objects = []
    size = None
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = [start]
        while stack:
            r, c = stack.pop()
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
                    component.append(point)
        top = int(min(p[0] for p in component))
        left = int(min(p[1] for p in component))
        bottom = int(max(p[0] for p in component)) + 1
        right = int(max(p[1] for p in component)) + 1
        s = bottom - top
        assert s == right - left and s >= 3
        assert size is None or size == s
        size = s
        tile = grid[top:bottom, left:right].copy()
        color = int(tile[0, 0])
        assert color != background
        if np.all(tile == color):
            kind = _SOLID
        else:
            kind = _HOLLOW
            assert np.array_equal(tile, _tile(s, color, background, kind))
        # Require a complete background gap, including diagonal neighbors.
        neighborhood = grid[max(0, top-1):min(height, bottom+1),
                            max(0, left-1):min(width, right+1)].copy()
        neighborhood[top-max(0, top-1):bottom-max(0, top-1),
                     left-max(0, left-1):right-max(0, left-1)] = background
        assert np.all(neighborhood == background)
        objects.append(dict(top=top, left=left, size=s, color=color,
                            kind=kind, tile=tile))
    assert objects
    for kind in (_HOLLOW, _SOLID):
        queue = [o for o in objects if o['kind'] == kind]
        assert queue
        tops = [o['top'] for o in queue]
        assert len(tops) == len(set(tops)), 'same-class vertical tie'
        assert size * len(queue) <= 30
    assert 2 * size <= 30
    objects.sort(key=lambda o: (o['top'], o['left']))
    return dict(background=int(background), tile_size=size, objects=objects)


def _analyze_input(grid):
    # Try all visible colors: background need not be black, at a corner, or
    # the largest component. Require a unique valid geometric interpretation.
    interpretations = []
    for background in np.unique(grid):
        try:
            interpretations.append(_parse_background(grid, int(background)))
        except AssertionError:
            continue
    assert len(interpretations) == 1, 'ambiguous or malformed input'
    return interpretations[0]


def _output_from_analysis(analysis, dtype=np.int8):
    s = analysis['tile_size']
    queues = [[o for o in analysis['objects'] if o['kind'] == kind]
              for kind in (_HOLLOW, _SOLID)]
    out = np.full((s * max(map(len, queues)), 2*s),
                  analysis['background'], dtype=dtype)
    for col, queue in enumerate(queues):
        for rank, obj in enumerate(queue):
            out[rank*s:(rank+1)*s, col*s:(col+1)*s] = obj['tile']
    return out


def _greedy_output(analysis):
    """The review's competing scan rule; it never fills an earlier hole."""
    rows = [[None, None]]
    for obj in analysis['objects']:
        col = int(obj['kind'] == _SOLID)
        if rows[-1][col] is not None:
            rows.append([None, None])
        rows[-1][col] = obj
    s = analysis['tile_size']
    out = np.full((len(rows)*s, 2*s), analysis['background'], dtype=np.int8)
    for row, slots in enumerate(rows):
        for col, obj in enumerate(slots):
            if obj is not None:
                out[row*s:(row+1)*s, col*s:(col+1)*s] = obj['tile']
    return out


def _greedy_witness(analysis):
    return not np.array_equal(_greedy_output(analysis),
                              _output_from_analysis(analysis))


def _horizontal_witness(analysis, reverse):
    alternative = dict(analysis)
    alternative['objects'] = sorted(analysis['objects'],
                                    key=lambda o: (o['left'], o['top']),
                                    reverse=reverse)
    return not np.array_equal(_output_from_analysis(analysis),
                              _output_from_analysis(alternative))


def _fallback_witness(size, background, demonstration, sign):
    # A guaranteed five-tile counterexample to greedy, horizontal ascending,
    # and horizontal descending packing. Used only if bounded sampling misses
    # a witness. Two tiles descend leftward; three follow a zigzag.
    positions = [(0, size+1), (1, 0), (size+2, 0),
                 (size+3, size+1), (2*size+4, 0)]
    kinds = [_HOLLOW]*2 + [_SOLID]*3
    if sign < 0:
        kinds = [_SOLID if k == _HOLLOW else _HOLLOW for k in kinds]
    colors = [demonstration] + [c for c in range(10)
                                if c not in (background, demonstration)][:4]
    hollow = kinds.index(_HOLLOW)
    colors[0], colors[hollow] = colors[hollow], colors[0]
    grid = np.full((3*size+4, 2*size+1), background, dtype=np.int8)
    for (r, c), color, kind in zip(positions, colors, kinds):
        grid[r:r+size, c:c+size] = _tile(size, color, background, kind)
    analysis = _parse_background(grid, background)
    return dict(input=grid, output=_output_from_analysis(analysis))


def _sequence(rng, hollow, solid, witness=False):
    mode = int(rng.integers(4))
    if mode == 0:
        kinds = [_HOLLOW]*hollow + [_SOLID]*solid
        if rng.integers(2):
            kinds.reverse()
    elif mode == 1:
        first = _HOLLOW if rng.integers(2) else _SOLID
        counts = {_HOLLOW: hollow, _SOLID: solid}
        kinds = []
        while sum(counts.values()):
            for kind in (first, _SOLID if first == _HOLLOW else _HOLLOW):
                if counts[kind]:
                    kinds.append(kind)
                    counts[kind] -= 1
    else:
        kinds = list(rng.permutation([_HOLLOW]*hollow + [_SOLID]*solid))
    # Compare rank assignments, not the number of transitions. The fallback
    # has an early same-class run and therefore necessarily exposes a hole.
    occupied = set()
    row = 0
    ranks = {_HOLLOW: 0, _SOLID: 0}
    disagrees = False
    for kind in kinds:
        if kind in occupied:
            occupied = set()
            row += 1
        disagrees |= row != ranks[kind]
        ranks[kind] += 1
        occupied.add(kind)
    if witness and not disagrees:
        first = _HOLLOW if rng.integers(2) else _SOLID
        rest = [_HOLLOW]*(hollow-(2 if first == _HOLLOW else 0))
        rest += [_SOLID]*(solid-(2 if first == _SOLID else 0))
        kinds = [first, first] + list(rng.permutation(rest))
    return kinds


def _partition(rng, total, parts, capacity):
    counts = [1]*parts
    for _ in range(total-parts):
        choices = [i for i, c in enumerate(counts) if c < capacity]
        counts[int(rng.choice(choices))] += 1
    return counts


def _sample_positions(rng, size, count):
    """Construct a feasible packing, then move tiles over the free canvas.

    Variable shelf counts are only an initialization. Free relocation removes
    the shelf lattice, with every accepted move checked against all rectangles.
    There is always a legal starting layout, even at maximum output counts.
    """
    margin = int(rng.random() < 0.45)
    extent = 30 - 2*margin
    capacity = (extent+1)//(size+1)
    bands = list(range((count+capacity-1)//capacity,
                       min(count, (extent+1-count)//size)+1))
    assert bands
    band_count = int(rng.choice(bands))
    counts = _partition(rng, count, band_count, capacity)
    min_h = count + band_count*size - 1
    min_w = max(counts)*(size+1)-1
    # Shape is sampled identically in both splits, independently of counts.
    mode = int(rng.integers(6))
    if mode == 0:  # tight cluster, with small and uneven blank margins
        h = int(rng.integers(min_h, min(extent, min_h+3)+1))
        w = int(rng.integers(min_w, min(extent, min_w+3)+1))
    elif mode == 1:
        h = w = int(rng.integers(max(min_h, min_w), extent+1))
    elif mode == 2:
        h = extent
        w = int(rng.integers(min_w, extent+1))
    elif mode == 3:
        w = extent
        h = int(rng.integers(min_h, extent+1))
    elif mode == 4:
        h = w = extent
    else:
        h = int(rng.integers(min_h, extent+1))
        w = int(rng.integers(min_w, extent+1))
    vertical = rng.multinomial(h-min_h, np.full(band_count+1, 1/(band_count+1)))
    positions = []
    top = int(vertical[0])
    for band, amount in enumerate(counts):
        horizontal = rng.multinomial(w-(amount*(size+1)-1),
                                    np.full(amount+1, 1/(amount+1)))
        left = int(horizontal[0])
        offsets = rng.permutation(amount)
        for j in range(amount):
            positions.append((top+int(offsets[j]), left))
            left += size+1+int(horizontal[j+1])
        top += size+amount+int(vertical[band+1])
    rr, cc = np.indices((h-size+1, w-size+1))
    # Three full sweeps; current coordinates are always a valid fallback.
    for _ in range(3):
        for idx in rng.permutation(count):
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(positions):
                if j != idx:
                    legal &= ((np.abs(rr-r) > size) | (np.abs(cc-c) > size))
                    legal &= rr != r
            sites = np.argwhere(legal)
            r, c = sites[int(rng.integers(len(sites)))]
            positions[int(idx)] = (int(r), int(c))
    if margin == 0 and rng.random() < 0.65:
        # A rigid translation reaches an edge without altering separation.
        side = int(rng.integers(4))
        dr = -min(r for r, _ in positions) if side == 0 else (
            h-size-max(r for r, _ in positions) if side == 1 else 0)
        dc = -min(c for _, c in positions) if side == 2 else (
            w-size-max(c for _, c in positions) if side == 3 else 0)
        positions = [(r+dr, c+dc) for r, c in positions]
    return (h+2*margin, w+2*margin), sorted(
        (r+margin, c+margin) for r, c in positions)


def _opposite_tie(rng, positions, kinds, size, shape):
    """Optionally align opposite classes; never invent a within-class tie."""
    for idx in rng.permutation(len(positions)):
        options = [p[0] for j, p in enumerate(positions) if kinds[j] != kinds[idx]]
        for top in rng.permutation(options):
            for left in rng.permutation(shape[1]-size+1):
                if all(j == idx or (
                    (kinds[j] != kinds[idx] or top != r) and
                    (abs(int(top)-r) > size or abs(int(left)-c) > size))
                       for j, (r, c) in enumerate(positions)):
                    positions[int(idx)] = (int(top), int(left))
                    return


def _sample_counts(rng, size, simple=False, sign=None, witness=False):
    limit = 30//size
    if simple:
        limit = min(limit, 4)
    minimum = 2 if witness else 1
    if sign is not None:
        small = int(rng.integers(minimum, limit))
        gap = int(rng.integers(1, limit-small+1))
        return (small, small+gap) if sign > 0 else (small+gap, small)
    if rng.random() < 0.25:
        same = int(rng.integers(1, limit+1))
        return same, same
    if rng.random() < 0.65:
        limit = min(limit, 5)
    return int(rng.integers(1, limit+1)), int(rng.integers(1, limit+1))


def _colors(rng, kinds, background, simple, demonstration, forced_kind, witness):
    palette = list(rng.permutation([c for c in range(10) if c != background]))
    n = len(kinds)
    if n <= 9 and (simple or rng.random() < 0.4):
        colors = palette[:n]
    else:
        pool_size = int(rng.integers(1, min(n, 7)+1))
        colors = [int(c) for c in rng.choice(palette[:pool_size], size=n)]
    if forced_kind is not None:
        idx = kinds.index(forced_kind)
        # Swap rather than duplicate in a distinct-color demonstration.
        if demonstration in colors:
            other = colors.index(demonstration)
            colors[idx], colors[other] = colors[other], colors[idx]
        else:
            colors[idx] = demonstration
    if witness:
        # Both queues visibly teach direction even if other tiles repeat.
        for kind in (_HOLLOW, _SOLID):
            indices = [i for i, k in enumerate(kinds) if k == kind]
            first, last = indices[0], indices[-1]
            if colors[first] == colors[last]:
                choices = [c for c in palette if c != colors[first]]
                colors[last] = int(rng.choice(choices))
    return colors


def _generate_pair(rng, size, counts, background, simple, demonstration,
                   forced_kind=None, witness=False):
    hollow, solid = counts
    shape, positions = _sample_positions(rng, size, hollow+solid)
    kinds = _sequence(rng, hollow, solid, witness)
    if not witness and rng.random() < 0.3:
        _opposite_tie(rng, positions, kinds, size, shape)
    colors = _colors(rng, kinds, background, simple, demonstration, forced_kind, witness)
    grid = np.full(shape, background, dtype=np.int8)
    for (r, c), kind, color in zip(positions, kinds, colors):
        grid[r:r+size, c:c+size] = _tile(size, color, background, kind)
    # Ground truth is reconstructed from the visible pixels, including ties
    # across opposite classes, instead of relying on construction order.
    analysis = _parse_background(grid, background)
    return dict(input=grid, output=_output_from_analysis(analysis))


def _sample_sizes(rng, train_count, test_count, simple):
    sizes = np.arange(3, 9)
    probabilities = [0.30, 0.24, 0.20, 0.12, 0.08, 0.06]
    first = int(rng.choice(sizes, p=probabilities)) if not simple else int(rng.integers(3, 6))
    if simple or rng.random() < 0.22:
        return [first]*(train_count+test_count)
    result = [first]
    result.append(int(rng.choice([s for s in sizes if s != first])))
    result += [int(rng.choice(sizes, p=probabilities)) for _ in range(train_count-2)]
    novel = [s for s in sizes if s not in result]
    for i in range(test_count):
        choices = novel if i == 0 and novel and rng.random() < 0.5 else sizes
        result.append(int(rng.choice(choices)))
    return result


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # All semantic conventions are fixed by this family: hollow left, solid
    # right, ascending top rows. These sampled settings control diversity only.
    simple = bool(rng.random() < 0.18)
    demonstration = int(rng.integers(1, 10))
    varying_background = not simple and rng.random() < 0.8
    background = 0 if simple or rng.random() < 0.6 else int(
        rng.choice([c for c in range(10) if c != demonstration]))
    sizes = _sample_sizes(rng, num_train, num_test, simple)
    first_sign = 1 if rng.integers(2) else -1
    test_style = int(rng.integers(6))
    puzzle = dict(train=[], test=[])
    seen = set()
    train_counts = []
    backgrounds = []
    for index, size in enumerate(sizes):
        training = index < num_train
        witness = index == 0
        if index < 2:
            counts = _sample_counts(rng, size, simple,
                                    first_sign if index == 0 else -first_sign, witness)
        elif training:
            counts = _sample_counts(rng, size, simple)
        else:
            # Shared ranges, familiar counts, either larger tail, increases and
            # decreases all remain the same visible rule. Multiple tests can
            # use different regimes inside the same puzzle.
            mode = test_style if index == num_train or rng.random() < 0.55 else int(rng.integers(6))
            limit = min(30//size, 4) if simple else 30//size
            familiar = [p for p in train_counts if max(p) <= limit]
            if mode == 0 and familiar:
                counts = familiar[int(rng.integers(len(familiar)))]
            elif mode in (1, 2) and limit >= 3:
                small = int(rng.integers(1, limit-1))
                large = int(rng.integers(small+2, limit+1))
                counts = (small, large) if mode == 1 else (large, small)
            elif mode == 3:
                same = int(rng.integers(1, limit+1))
                counts = (same, same)
            elif mode == 4 and familiar:
                counts = list(familiar[int(rng.integers(len(familiar)))])
                col = int(rng.integers(2))
                counts[col] = int(rng.integers(1, limit+1))
                counts = tuple(counts)
            else:
                counts = _sample_counts(rng, size, simple)
        if varying_background:
            choices = [c for c in range(10) if c != demonstration]
            if index == 1:
                choices.remove(backgrounds[0])
            bg = 0 if index != 1 and rng.random() < 0.35 else int(rng.choice(choices))
        else:
            bg = background
        forced = _HOLLOW if index == 0 else (_SOLID if index == 1 else None)
        # The constructor itself is bounded and always feasible. Check the
        # actual visible first demonstration against horizontal shortcuts too.
        for attempt in range(32):
            pair = _generate_pair(rng, size, counts, bg, simple,
                                  demonstration, forced, witness)
            key = _grid_key(pair['input'])
            if witness:
                analysis = _parse_background(pair['input'], bg)
                if not all(_horizontal_witness(analysis, reverse) for reverse in (False, True)):
                    continue
            if key not in seen and not _source_copy(pair['input']):
                break
        else:
            if witness:
                pair = _fallback_witness(size, bg, demonstration, first_sign)
                counts = (2, 3) if first_sign > 0 else (3, 2)
                key = _grid_key(pair['input'])
            # Deterministic content variation is enough to break a collision
            # without changing any geometry or learned color convention.
            for shift in range(1, 10):
                if key not in seen and not _source_copy(pair['input']):
                    break
                for name in ('input', 'output'):
                    pair[name] = ((pair[name].astype(np.int16)+1) % 10).astype(np.int8)
                key = _grid_key(pair['input'])
                if key not in seen and not _source_copy(pair['input']):
                    break
            assert key not in seen and not _source_copy(pair['input'])
        seen.add(key)
        puzzle['train' if training else 'test'].append(pair)
        backgrounds.append(bg)
        if training:
            train_counts.append(counts)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    training = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert _grid_key(inp) not in seen_inputs
            assert _pair_key(pair) not in seen_pairs
            seen_inputs.add(_grid_key(inp))
            seen_pairs.add(_pair_key(pair))
            assert not _source_copy(inp)
            assert not np.array_equal(inp, out)
            analysis = _analyze_input(inp)
            assert np.array_equal(out, _output_from_analysis(analysis))
            if split == 'train':
                training.append(analysis)
    # Even two demonstrations teach compaction, copying, both padding sides,
    # direction with visible colors in both queues, and shape-based roles.
    assert any(_greedy_witness(a) for a in training), 'missing compaction witness'
    for reverse in (False, True):
        assert any(_horizontal_witness(a, reverse) for a in training), 'horizontal-order shortcut'
    differences = []
    roles = {}
    for a in training:
        counts = [sum(o['kind'] == k for o in a['objects']) for k in (_HOLLOW, _SOLID)]
        differences.append(counts[1]-counts[0])
        for o in a['objects']:
            roles.setdefault(o['color'], set()).add(o['kind'])
    assert min(differences) < 0 < max(differences), 'demonstrate both padding sides'
    assert any(len(v) == 2 for v in roles.values()), 'missing color-role switch'
    for kind in (_HOLLOW, _SOLID):
        assert any(len({o['color'] for o in a['objects'] if o['kind'] == kind}) >= 2
                   for a in training), 'ordering must be observable in both classes'


def _scan_prediction(grid):
    """Second implementation: consume complete square footprints, no BFS.

    This authored check is useful for implementation mistakes; it is not a
    protected reference solver or independent evidence for the semantic rule.
    """
    answers = []
    for bg in np.unique(grid):
        todo = grid != bg
        objects = []
        valid = True
        while np.any(todo):
            r, c = np.argwhere(todo)[0]
            s = 0
            while c+s < grid.shape[1] and todo[r, c+s]:
                s += 1
            tile = grid[r:r+s, c:c+s]
            if s < 3 or tile.shape != (s, s):
                valid = False
                break
            color = grid[r, c]
            rim = np.concatenate((tile[0], tile[-1], tile[:, 0], tile[:, -1]))
            if not np.all(rim == color):
                valid = False
                break
            interior = tile[1:-1, 1:-1]
            if np.all(interior == bg):
                col = 0
            elif np.all(interior == color):
                col = 1
            else:
                valid = False
                break
            objects.append((int(r), col, s, tile.copy()))
            todo[r:r+s, c:c+s] = False
        if valid and objects and {o[1] for o in objects} == {0, 1} and len({o[2] for o in objects}) == 1:
            s = objects[0][2]
            columns = [[o[3] for o in sorted(objects, key=lambda t: t[0]) if o[1] == col]
                       for col in (0, 1)]
            length = max(map(len, columns))
            if length*s > 30:
                continue
            for col in columns:
                col.extend([np.full((s, s), bg, dtype=np.int8)]*(length-len(col)))
            answers.append(np.concatenate([np.concatenate(col, axis=0) for col in columns], axis=1))
    assert len(answers) == 1
    return answers[0]


def _features(pair):
    a = _analyze_input(pair['input'])
    objects, s = a['objects'], a['tile_size']
    queues = [[o for o in objects if o['kind'] == k] for k in (_HOLLOW, _SOLID)]
    h, w = pair['input'].shape
    colors = [{o['color'] for o in q} for q in queues]
    horizontal = any([o['color'] for o in q] != [o['color'] for o in sorted(q, key=lambda o: (o['left'], o['top']))]
                     for q in queues)
    nearest = any(min(queues[1], key=lambda p: (abs(o['top']-p['top'])+abs(o['left']-p['left']), p['top'])) is not queues[1][i]
                  for i, o in enumerate(queues[0][:len(queues[1])]))
    def bands_fit(values):
        return any(all(any(0 <= v-start-band*(s+3) <= 2 for band in range(3)) for v in values)
                   for start in range(min(values)-2, min(values)+1))
    old_lattice = bands_fit([o['top'] for o in objects]) and bands_fit([o['left'] for o in objects])
    return dict(size=s, background=a['background'], hollow=len(queues[0]), solid=len(queues[1]),
                total=len(objects), imbalance=len(queues[1])-len(queues[0]),
                aspect=int(np.sign(h-w)), shape=(h, w),
                edge=any(o['top'] == 0 or o['left'] == 0 or o['top']+s == h or o['left']+s == w for o in objects),
                repeat=len(set.union(*colors)) < len(objects),
                repeat_within=any(len(c) < len(q) for c, q in zip(colors, queues)),
                repeat_across=bool(colors[0] & colors[1]),
                black_tile=any(o['color'] == 0 for o in objects),
                cross_tie=len({o['top'] for o in objects}) < len(objects),
                sequence=''.join(o['kind'][0].upper() for o in objects),
                greedy=_greedy_witness(a), horizontal=horizontal, nearest=nearest,
                old_lattice=old_lattice,
                density=float(np.mean(pair['input'] != a['background'])),
                layout=(s, tuple((o['top'], o['left'], o['kind']) for o in objects)))


def _puzzle_signature(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[s]) for s in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = []
    within = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        per = {s: [_features(p) for p in puzzle[s]] for s in ('train', 'test')}
        within.append(per)
        for split in ('train', 'test'):
            for pair, f in zip(puzzle[split], per[split]):
                assert np.array_equal(_scan_prediction(pair['input'].copy()), pair['output'])
                features.append((split, f))
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    all_f = [f for _, f in features]
    assert len({f['shape'] for f in all_f}) >= 50
    assert len({f['layout'] for f in all_f}) >= 180
    assert {f['size'] for f in all_f} == set(range(3, 9))
    assert {f['background'] for f in all_f} == set(range(10))
    for split in ('train', 'test'):
        fs = [f for s, f in features if s == split]
        assert {f['aspect'] for f in fs} == {-1, 0, 1}
        assert len({(f['hollow'], f['solid']) for f in fs}) >= 20
        assert any(f['imbalance'] >= 2 for f in fs)
        assert any(f['imbalance'] <= -2 for f in fs)
        assert any(f['imbalance'] == 0 for f in fs)
        assert any(f['hollow'] == 1 for f in fs) and any(f['solid'] == 1 for f in fs)
        assert any(f['total'] <= 3 for f in fs) and any(f['total'] > 9 for f in fs)
        assert any(max(f['shape']) <= 16 for f in fs)
        assert any(f['shape'] == (30, 30) for f in fs)
    for property_name in ('edge', 'repeat', 'repeat_within', 'repeat_across', 'black_tile', 'cross_tie'):
        assert sum(f[property_name] for f in all_f) >= 10
        assert sum(not f[property_name] for f in all_f) >= 10
    assert sum(not f['old_lattice'] for f in all_f) >= len(all_f)*0.8
    assert sum(f['background'] != 0 and f['repeat'] and f['edge'] and f['imbalance'] < 0 for f in all_f) >= 5
    assert sum(f['horizontal'] for s, f in features if s == 'train') >= 30
    assert sum(f['nearest'] for s, f in features if s == 'train') >= 30
    assert sum(all(a != b for a, b in zip(f['sequence'], f['sequence'][1:])) for f in all_f) >= 10
    assert sum(f['sequence'].count('HS')+f['sequence'].count('SH') == 1 for f in all_f) >= 10
    for property_name in ('size', 'background', 'aspect', 'total', 'imbalance'):
        assert sum(len({f[property_name] for f in p['train']+p['test']}) > 1 for p in within) >= 15
    assert any(len({f['size'] for f in p['train']+p['test']}) == 1 for p in within)
    assert sum(any(f['size'] not in {t['size'] for t in p['train']} for f in p['test']) for p in within) >= 5
    assert sum(len({f['size'] for f in p['test']}) > 1 for p in within) >= 5
    # Visible familiar counts as well as increases/decreases in either class.
    assert any((t['hollow'], t['solid']) == (f['hollow'], f['solid']) for p in within for t in p['train'] for f in p['test'])
    for kind in ('hollow', 'solid'):
        assert any(f[kind] > max(t[kind] for t in p['train']) for p in within for f in p['test'])
        assert any(f[kind] < min(t[kind] for t in p['train']) for p in within for f in p['test'])
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        a = generate_puzzle(1001, train, test)
        b = generate_puzzle(1001, train, test)
        assert _puzzle_signature(a) == _puzzle_signature(b)
        assert train is None or len(a['train']) == train
        assert test is None or len(a['test']) == test
    assert _puzzle_signature(generate_puzzle(123)) == _puzzle_signature(generate_puzzle(123))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2.0, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Negative controls exercise the actual review correction and validator.
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt copying accepted')
    kinds = [_SOLID, _HOLLOW, _SOLID, _HOLLOW, _SOLID]
    objects = [dict(top=i, left=i*5, kind=k, color=i+1, size=3, tile=_tile(3, i+1, 0, k))
               for i, k in enumerate(kinds)]
    a = dict(objects=objects, tile_size=3, background=0)
    assert not _greedy_witness(a)
    objects[0]['kind'], objects[3]['kind'] = _HOLLOW, _SOLID
    objects[0]['tile'] = _tile(3, 1, 0, _HOLLOW)
    objects[3]['tile'] = _tile(3, 4, 0, _SOLID)
    assert _greedy_witness(a)
    shortcut = dict(train=[], test=puzzles[0]['test'])
    for first in (_HOLLOW, _SOLID):
        grid = np.zeros((12, 26), dtype=np.int8)
        second = _SOLID if first == _HOLLOW else _HOLLOW
        for i, kind in enumerate([first, second, first, second, first]):
            grid[i:i+3, i*5:i*5+3] = _tile(3, i+1, 0, kind)
        shortcut['train'].append(dict(input=grid, output=_scan_prediction(grid)))
    try:
        validate_generated_puzzle(shortcut)
    except AssertionError as exc:
        assert 'compaction witness' in str(exc)
    else:
        raise AssertionError('alternating-only training accepted')
    for size in range(3, 9):
        for sign in (-1, 1):
            p = _fallback_witness(size, 0, 5, sign)
            a = _analyze_input(p['input'])
            assert _greedy_witness(a)
            assert all(_horizontal_witness(a, reverse) for reverse in (False, True))
            assert np.array_equal(_scan_prediction(p['input']), p['output'])
    # Source dimensions remain legal; only the original cell contents are
    # excluded. Exercise every formerly banned shape with a new arrangement.
    for shape, tiles in _SOURCE_TILES:
        first = _fallback_witness(4, 0, 5, 1)
        inset = np.zeros(shape, dtype=np.int8)
        h, w = first['input'].shape
        inset[1:h+1, 1:w+1] = first['input']
        first['input'] = inset
        validate_generated_puzzle(dict(
            train=[first, _fallback_witness(4, 0, 5, -1)],
            test=[_fallback_witness(3, 1, 2, 1)]))
        source = np.zeros(shape, dtype=np.int8)
        for r, c, color, kind in tiles:
            source[r:r+4, c:c+4] = _tile(4, color, 0, _HOLLOW if kind == 'h' else _SOLID)
        assert _source_copy(source) and not _source_copy(inset)
