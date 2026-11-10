import numpy as np


_DIRECTIONS = ("right", "up", "left", "down")
_ROTATION_FOR_DIRECTION = {
    "right": 0,
    "up": 1,
    "left": 2,
    "down": 3,
}

# Exact source pairs, used only to reject accidental copies.
_SOURCE_PAIR_TEXTS = frozenset(
    {
        (
            "111111111111111\n"
            "111111111111111\n"
            "111111111111111\n"
            "111111333333111\n"
            "111113119111311\n"
            "111131111333111\n"
            "111131113111111\n"
            "111113113111111\n"
            "111111311311111\n"
            "111113111131111\n"
            "111131111113111\n"
            "111111155111111\n"
            "111111111111111\n"
            "111111111111111\n"
            "111111111111111",
            "111111111111111\n"
            "111111111111111\n"
            "111111111111111\n"
            "111111333333111\n"
            "111113999999311\n"
            "111139999333111\n"
            "111139993111111\n"
            "111113993111111\n"
            "111111399311111\n"
            "111113999931111\n"
            "111139999993111\n"
            "111199955999111\n"
            "111999911999911\n"
            "119999911999991\n"
            "199999911999999",
        ),
        (
            "000000000000000\n"
            "000000000000000\n"
            "000000050005550\n"
            "000000005050050\n"
            "000000000500450\n"
            "000000000000050\n"
            "000000000500050\n"
            "000000005055550",
            "444440000000000\n"
            "444444000000000\n"
            "444444450005550\n"
            "444444445054450\n"
            "444444444544450\n"
            "444444444444450\n"
            "444444444544450\n"
            "444444445055550",
        ),
        (
            "7777777777777777\n"
            "7777767777777777\n"
            "7777777777777777\n"
            "7777977777977777\n"
            "7777977677977777\n"
            "7777797779777777\n"
            "7777797779777777\n"
            "7777779797777777\n"
            "7777779797777777\n"
            "7777797799977777\n"
            "7777977773977777\n"
            "7777977333977777\n"
            "7777797339777777\n"
            "7777779997777777\n"
            "7777777777777677\n"
            "7767777777777777\n"
            "7777777778777777\n"
            "7777777777777777",
            "7773373733337777\n"
            "7777363733377777\n"
            "7777333733377777\n"
            "7777933733977777\n"
            "7777933633977777\n"
            "7777793339777777\n"
            "7777793339777777\n"
            "7777779397777777\n"
            "7777779397777777\n"
            "7777793399977777\n"
            "7777933333977777\n"
            "7777933333977777\n"
            "7777793339777777\n"
            "7777779997777777\n"
            "7777777777777677\n"
            "7767777777777777\n"
            "7777777778777777\n"
            "7777777777777777",
        ),
        (
            "1111111111111111111\n"
            "1111111111111111111\n"
            "1111111111111111111\n"
            "1111111111111111111\n"
            "1111111111111111111\n"
            "1111111144111144411\n"
            "1111311111441411411\n"
            "1111111111114111411\n"
            "1181111111111121411\n"
            "1111113111114411411\n"
            "1111111111441444111\n"
            "1111111144111111111\n"
            "1111111111111111111\n"
            "1111111111111111111\n"
            "1111111111111111111",
            "1111111111111111111\n"
            "1111111111111111111\n"
            "2211111111111111111\n"
            "2222111111111111111\n"
            "2222221111111111111\n"
            "2222222244111144411\n"
            "1111322222441422411\n"
            "2222222222224222411\n"
            "1182222222222222411\n"
            "1111113222224422411\n"
            "2222222222441444111\n"
            "2222222244111111111\n"
            "2222221111111111111\n"
            "2222111111111111111\n"
            "2211111111111111111",
        ),
    }
)


def _grid_text(grid: np.ndarray) -> str:
    return "\n".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _is_source_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    return (_grid_text(inp), _grid_text(out)) in _SOURCE_PAIR_TEXTS


def _array_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(v) for v in grid.shape),
        tuple(int(v) for v in grid.ravel()),
    )


def _puzzle_key(puzzle: dict) -> tuple:
    result = []
    for section in ("train", "test"):
        result.append(
            tuple(
                (_array_key(pair["input"]), _array_key(pair["output"]))
                for pair in puzzle[section]
            )
        )
    return tuple(result)


def _puzzles_equal(first: dict, second: dict) -> bool:
    if tuple(first.keys()) != tuple(second.keys()):
        return False
    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False
        for a, b in zip(first[section], second[section]):
            if set(a) != {"input", "output"}:
                return False
            if set(b) != {"input", "output"}:
                return False
            if not np.array_equal(a["input"], b["input"]):
                return False
            if not np.array_equal(a["output"], b["output"]):
                return False
    return True


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _validate_count_argument(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}"
        )
    return value


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_count_argument(
        num_train, "num_train", 2, 10
    )
    num_test = _validate_count_argument(
        num_test, "num_test", 1, 4
    )

    if num_train is None and num_test is None:
        # This deterministic schedule guarantees count diversity over
        # consecutive seeds while remaining entirely seed-controlled.
        num_test = 1 + seed % 4
        num_train = (
            num_test
            + 1
            + (seed // 4) % (10 - num_test)
        )
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError(
            "counts must satisfy 2 <= num_train <= 10, "
            "1 <= num_test <= 4, and num_train > num_test"
        )

    return num_train, num_test


def _fit_flank(xs: np.ndarray, ys: np.ndarray, sign: int) -> tuple:
    """Read the longest terminal periodic stair; prefer its least period."""
    candidates = []
    end, value = int(xs[-1]), int(ys[-1])
    for run in (1, 2, 3):
        for phase in range(run):
            predicted = value + sign * ((xs - end + phase) // run)
            length = 0
            for same in (predicted == ys)[::-1]:
                if not same:
                    break
                length += 1
            if length >= 2 and len(set(ys[-length:])) >= 2:
                candidates.append((length, -run, phase))
    if not candidates:
        raise ValueError("mouth lacks a readable widening cadence")
    length, negative_run, phase = max(candidates)
    return -negative_run, phase, end, value, length


def _inspect_input(inp: np.ndarray) -> dict:
    """Recover roles and geometry from the input, never from its answer.

    Each axial section has one selected interior between the inner faces of
    the two wall runs. A closed rear and outward terminal stairs determine
    orientation. The upstream interior patch supplies the payload; other
    interior colors are permitted only downstream of that patch.
    """
    counts = np.bincount(inp.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > inp.size // 2, "background must dominate input"
    matches = []
    for rotation in range(4):
        a = np.rot90(inp, -rotation)
        for color in np.unique(a):
            wall = int(color)
            if wall == background:
                continue
            positions = np.argwhere(a == wall)
            if len(positions) < 9:
                continue
            rear, mouth = int(positions[:, 1].min()), int(positions[:, 1].max())
            cap = np.flatnonzero(a[:, rear] == wall)
            if len(cap) < 3 or np.any(np.diff(cap) != 1) or mouth - rear < 3:
                continue
            lows, highs, low_x, high_x = [], [], [], []
            valid = True
            for x in range(rear + 1, mouth + 1):
                rows = np.flatnonzero(a[:, x] == wall)
                gaps = np.flatnonzero(np.diff(rows) > 1)
                if len(gaps) == 1:
                    j = int(gaps[0])
                    lows.append(int(rows[j]))
                    highs.append(int(rows[j + 1]))
                    low_x.append(x)
                    high_x.append(x)
                elif len(rows) == 1 and x == mouth and lows:
                    # The compact source example clips one terminal rail.
                    if abs(int(rows[0]) - lows[-1]) <= 1:
                        low_x.append(x)
                        lows.append(int(rows[0]))
                    elif abs(int(rows[0]) - highs[-1]) <= 1:
                        high_x.append(x)
                        highs.append(int(rows[0]))
                    else:
                        valid = False
                        break
                else:
                    valid = False
                    break
            if not valid:
                continue
            try:
                fits = (_fit_flank(np.array(low_x), np.array(lows), -1),
                        _fit_flank(np.array(high_x), np.array(highs), 1))
            except ValueError:
                continue
            footprint = np.zeros(a.shape, dtype=bool)
            bounds = []
            for x in range(rear + 1, a.shape[1]):
                limits = []
                for xs, ys, fit, sign in zip((low_x, high_x), (lows, highs), fits, (-1, 1)):
                    run, phase, end, value, _ = fit
                    limit = (ys[xs.index(x)] if x in xs else
                             value + sign * ((x - end + phase) // run))
                    limits.append(limit)
                if limits[1] - limits[0] < 2:
                    valid = False
                    break
                bounds.append(tuple(limits))
                footprint[:, x] = ((np.arange(a.shape[0]) > limits[0]) &
                                   (np.arange(a.shape[0]) < limits[1]))
            if not valid or np.any(footprint & (a == wall)):
                continue
            wall_cells = set(map(tuple, positions))
            pending = [min(wall_cells)]
            wall_cells.remove(pending[0])
            while pending:
                y, x = pending.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        point = (y + dy, x + dx)
                        if point in wall_cells:
                            wall_cells.remove(point)
                            pending.append(point)
            if wall_cells:
                continue
            # Any visible single rail must be explained by actual edge clipping.
            if len(low_x) != len(high_x):
                lo, hi = bounds[mouth - rear - 1]
                if 0 <= lo < a.shape[0] and 0 <= hi < a.shape[0]:
                    continue
            chamber = footprint.copy()
            chamber[:, mouth + 1:] = False
            if len(_components(chamber)) != 1:
                continue
            markers = np.argwhere(chamber & (a != background) & (a != wall))
            if not len(markers):
                continue
            first = markers[markers[:, 1] == markers[:, 1].min()]
            colors = {int(a[y, x]) for y, x in first}
            if len(colors) != 1:
                continue
            payload = colors.pop()
            patch = a == payload
            if np.any(patch & ~chamber) or len(_components(patch)) != 1:
                continue
            extras = (a != background) & (a != wall) & ~patch
            internal_extras = extras & chamber
            if np.any(internal_extras):
                if np.argwhere(internal_extras)[:, 1].min() <= np.argwhere(patch)[:, 1].max():
                    continue
            matches.append(dict(canonical=a, rotation=rotation, background=background,
                                structure=wall, payload=payload, rear=rear, mouth=mouth,
                                fits=fits, footprint=footprint, chamber=chamber,
                                bounds=tuple(bounds), extras=extras))
    assert len(matches) == 1, "input must identify one chamber, payload and opening"
    return matches[0]


def _components(mask: np.ndarray) -> list:
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, queue = [start], [start]
        while queue:
            y, x = queue.pop()
            for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if point in remaining:
                    remaining.remove(point)
                    group.append(point)
                    queue.append(point)
        groups.append(group)
    return groups


def _predict_from_input(inp: np.ndarray) -> np.ndarray:
    analysis = _inspect_input(inp)
    a = analysis['canonical']
    expected = a.copy()
    # A lane scan is deliberately separate from construction's shadow mask.
    for y in range(a.shape[0]):
        blocked = False
        for x in range(analysis['rear'] + 1, a.shape[1]):
            if not analysis['footprint'][y, x]:
                continue
            if analysis['extras'][y, x]:
                blocked = True
            elif a[y, x] == analysis['background'] and not blocked:
                expected[y, x] = analysis['payload']
    return np.rot90(expected, analysis['rotation']).copy()


def predict(train_pairs: list, test_input: np.ndarray) -> np.ndarray:
    """Prediction interface accepts only demonstrations and a query input.

    Color values, orientation and cadence are visible per pair. The shared
    rule is thin-boundary continuation with straight opaque lane shadows.
    """
    assert train_pairs
    clean, occluded = False, False
    for pair in train_pairs:
        a = _inspect_input(pair['input'])
        assert np.array_equal(_predict_from_input(pair['input']), pair['output'])
        effective = a['extras'] & a['footprint']
        clean |= not np.any(effective)
        occluded |= _features_from_analysis(a)['shadow_count'] > 0
    query = _inspect_input(test_input)
    assert clean, "training must demonstrate clean chamber and cone filling"
    if np.any(query['extras'] & query['footprint']):
        assert occluded, "training must demonstrate opacity"
    return _predict_from_input(test_input)


def _apply_rule_canonical(inp, background, structure, payload, footprint):
    extra = (inp != background) & (inp != structure) & (inp != payload)
    blockers = extra & footprint
    shadow = np.zeros(inp.shape, dtype=bool)
    for y, x in np.argwhere(blockers):
        shadow[int(y), int(x) + 1:] = True
    out = inp.copy()
    out[footprint & (inp == background) & ~shadow] = payload
    return out


def _sample_geometry(rng, complexity, projection_regime):
    # Thin rails: one pixel in every axial section, including at stair corners.
    # The rear is a separate chamber; only the terminal flanks are extrapolated.
    compact = rng.random() < 0.20
    simple = rng.random() < (0.65 if complexity == 0 else 0.2)
    runs = [int(rng.integers(1, 4))] * 2
    if rng.random() < 0.3:
        runs[1] = int(rng.integers(1, 4))
    if compact:
        runs = [int(rng.integers(1, 3))] * 2
    phases = [int(rng.integers(run)) for run in runs]
    if simple:
        length = int(rng.integers(2, 5))
        starts = rng.integers(1, 3 if compact else 4, size=2)
        profiles = [np.full(length, int(v), dtype=int) for v in starts]
    else:
        length = int(rng.integers(5, 7) if compact else rng.integers(6, 10))
        middle = int(rng.integers(2, length - 2))
        profiles = []
        for _ in range(2):
            start, neck = (int(rng.integers(1, 3 if compact else 4)) for _ in range(2))
            peak = int(rng.integers(max(start, neck), min(start + middle, neck + length - 1 - middle, 3 if compact else 5) + 1))
            profile = np.rint(np.interp(np.arange(length), [0, middle, length - 1],
                                       [start, peak, neck])).astype(int)
            profiles.append(profile)
    if rng.random() < 0.25:
        profiles[1] = profiles[0].copy()
        runs[1], phases[1] = runs[0], phases[0]
    flare_length = 2 * max(runs) + (0 if compact else int(rng.integers(0, 3)))
    extended = []
    for profile, run, phase in zip(profiles, runs, phases):
        extension = [int(profile[-1]) + (t + phase) // run
                     for t in range(1, flare_length + 1)]
        extended.append(np.array(list(profile) + extension, dtype=int))
    axial_length = len(extended[0]) + 1
    minimum_projection = 10 if projection_regime == 'long' else 2
    rear_max = min(3 if compact else 8, 30 - axial_length - minimum_projection)
    rear = int(rng.integers(rear_max + 1))
    max_projection = 30 - rear - axial_length
    if projection_regime == 'short':
        max_projection = min(9, max_projection)
    elif compact and projection_regime != 'long':
        max_projection = min(8, max_projection)
    projection = int(rng.integers(minimum_projection, max_projection + 1))
    width = rear + axial_length + projection
    span = int(extended[0].max() + extended[1].max() + 1)
    height = int(rng.integers(max(8, span), max(8, span) + 4 if compact else 31))
    if rng.random() < 0.08 and projection_regime != 'short':
        height, width = 30, 30
    center = int(rng.integers(int(extended[0].max()), height - int(extended[1].max())))
    mouth = rear + axial_length - 1
    wall = np.zeros((height, width), dtype=bool)
    footprint = np.zeros_like(wall)
    wall[center - profiles[0][0]:center + profiles[1][0] + 1, rear] = True
    for j, (upper, lower) in enumerate(zip(*extended)):
        x = rear + 1 + j
        wall[center - upper, x] = True
        wall[center + lower, x] = True
        footprint[center - upper + 1:center + lower, x] = True
    for x in range(mouth + 1, width):
        t = x - (rear + length)
        upper = center - int(profiles[0][-1]) - (t + phases[0]) // runs[0]
        lower = center + int(profiles[1][-1]) + (t + phases[1]) // runs[1]
        footprint[max(0, upper + 1):min(height, lower), x] = True
    return wall, footprint, rear, mouth, rear + length


def _grow_patch(rng, candidates, target):
    cells = set(map(tuple, np.argwhere(candidates)))
    start = sorted(cells)[int(rng.integers(len(cells)))]
    patch = {start}
    for _ in range(target - 1):
        frontier = sorted({q for y, x in patch for q in
                           ((y-1, x), (y+1, x), (y, x-1), (y, x+1))
                           if q in cells and q not in patch})
        if not frontier:
            break
        patch.add(frontier[int(rng.integers(len(frontier)))])
    return sorted(patch)


def _place_group(rng, grid, footprint, background, color, min_x, mouth, kind, shape):
    height, width = grid.shape
    offsets = ((0, 0),) if shape == 'single' else (
        ((0, 0), (1, 0)) if shape == 'bar' else (
        ((0, 0), (0, 1), (1, 0), (1, 1)) if shape == 'rectangle' else
        ((0, 0), (1, 0), (1, 1))))
    candidates = []
    for x in range(min_x, width):
        for y in range(height):
            points = [(y + dy, x + dx) for dy, dx in offsets]
            if any(yy >= height or xx >= width or grid[yy, xx] != background for yy, xx in points):
                continue
            inside = [footprint[yy, xx] for yy, xx in points]
            if not any(inside):
                continue
            if kind == 'internal' and not all(xx <= mouth for _, xx in points):
                continue
            if kind == 'mouth' and x not in (mouth, mouth + 1):
                continue
            if kind == 'far' and x < mouth + 2:
                continue
            if kind == 'edge' and max(xx for _, xx in points) < width - 2:
                continue
            if kind == 'straddle' and (all(inside) or x <= mouth):
                continue
            candidates.append(points)
    if not candidates:
        return False
    points = candidates[int(rng.integers(len(candidates)))]
    for y, x in points:
        grid[y, x] = color
    return True


def _build_pair(rng, palette, rotation, complexity, style, projection_regime, teaching=False):
    background, structure, payload = map(int, palette[:3])
    wall, footprint, rear, mouth, flare_start = _sample_geometry(rng, complexity, projection_regime)
    grid = np.full(wall.shape, background, dtype=np.int8)
    grid[wall] = structure
    patch_candidates = footprint.copy()
    patch_candidates[:, flare_start:] = False
    # Leave at least one axial section before the mouth for internal obstacles.
    area = 1 if rng.random() < 0.35 else int(rng.integers(2, 9))
    patch = _grow_patch(rng, patch_candidates, area)
    for y, x in patch:
        grid[y, x] = payload
    min_x = max(x for _, x in patch) + 1
    extra_colors = [int(v) for v in palette[3:3 + (2 if teaching else int(rng.integers(1, 4)))]]
    if style != 'clean':
        groups = 2 if teaching else (int(rng.integers(1, 3)) if style == 'light' else int(rng.integers(3, 6)))
        for index in range(groups):
            kind = str(rng.choice(['internal', 'mouth', 'far', 'edge', 'straddle', 'repeat']))
            shape = str(rng.choice(['single', 'bar', 'rectangle', 'irregular']))
            color = extra_colors[index % len(extra_colors)]
            if teaching:
                # Separate colors and lanes make color-independent opacity visible.
                shape, kind = 'single', 'visible'
                occupied_rows = set(np.argwhere((grid != background) & (grid != structure) &
                                                (grid != payload))[:, 0])
                options = [(int(y), int(x)) for y, x in np.argwhere(footprint & (grid == background))
                           if x >= min_x and x <= grid.shape[1] - 3 and y not in occupied_rows
                           and np.any(footprint[y, x + 1:] & (grid[y, x + 1:] == background))]
                if options:
                    y, x = options[int(rng.integers(len(options)))]
                    grid[y, x] = color
                    continue
            if kind == 'repeat':
                effective = np.argwhere(footprint & (grid != background) & (grid != structure) & (grid != payload))
                options = [(int(y), xx) for y, x in effective for xx in range(int(x) + 1, grid.shape[1])
                           if footprint[y, xx] and grid[y, xx] == background]
                if options:
                    y, x = options[int(rng.integers(len(options)))]
                    grid[y, x] = color
                    continue
                kind = 'far'
            placed = _place_group(rng, grid, footprint, background, color, min_x, mouth, kind, shape)
            if not placed:
                _place_group(rng, grid, footprint, background, color, min_x, mouth, 'any', 'single')
    # Exterior distractions can share any blocker color, but never color roles.
    for _ in range(int(rng.integers(0, 4))):
        options = np.argwhere(~footprint & ~wall & (grid == background))
        if len(options):
            y, x = options[int(rng.integers(len(options)))]
            grid[y, x] = extra_colors[int(rng.integers(len(extra_colors)))]
    output = _apply_rule_canonical(grid, background, structure, payload, footprint)
    return {'input': np.rot90(grid, rotation).copy(), 'output': np.rot90(output, rotation).copy()}


def _features_from_analysis(a):
    grid, footprint = a['canonical'], a['footprint']
    inside = a['extras'] & footprint
    outside = a['extras'] & ~footprint
    shadow = np.zeros(grid.shape, dtype=bool)
    for y, x in np.argwhere(inside):
        shadow[y, x + 1:] = True
    shadow &= footprint & (grid == a['background'])
    patch = np.argwhere(grid == a['payload'])
    wall = grid == a['structure']
    rows, cols = np.nonzero(wall)
    crop = wall[rows.min():rows.max()+1, cols.min():cols.max()+1]
    low = np.array([v[0] for v in a['bounds'][:a['mouth']-a['rear']]])
    high = np.array([v[1] for v in a['bounds'][:a['mouth']-a['rear']]])
    groups = []
    for color in np.unique(grid[a['extras']]):
        groups.extend(_components(grid == color))
    effective_groups = [g for g in groups if any(footprint[y, x] for y, x in g)]
    patch_crop = grid[patch[:,0].min():patch[:,0].max()+1, patch[:,1].min():patch[:,1].max()+1] == a['payload']
    lateral_offsets = [2*int(y)-sum(a['bounds'][int(x)-a['rear']-1]) for y,x in patch]
    terminal_start = min(f[2] - f[4] + 1 for f in a['fits'])
    return dict(direction=_DIRECTIONS[a['rotation']], palette=(a['background'], a['structure'], a['payload']),
                shape=tuple(grid.shape), outline=_array_key(crop.astype(np.int8)),
                runs=tuple(f[0] for f in a['fits']), phases=tuple(f[1] for f in a['fits']),
                visible_length=a['mouth']-a['rear'], rear=a['rear'],
                projection=grid.shape[1]-a['mouth']-1, chamber_area=int(a['chamber'].sum()),
                mouth_width=int(high[-1]-low[-1]-1), min_channel_width=int((high-low-1).min()),
                neck_position=int(np.argmin(high-low-1)) + 1,
                lateral_margin=int(rows.min()),
                nonmonotone=bool(np.any(np.diff(high-low)<0)),
                asymmetric=bool(np.any(low+high != low[0]+high[0])),
                payload_area=len(patch), payload_shape=_array_key(patch_crop.astype(np.int8)),
                payload_irregular=bool(min(patch_crop.shape)>1 and patch_crop.sum()<patch_crop.size),
                payload_offset=bool(any(v != 0 for v in lateral_offsets)),
                single_offcenter=bool(len(patch)==1 and lateral_offsets[0]!=0),
                payload_axial=int(patch[:,1].min()-a['rear']),
                inside_count=int(inside.sum()), outside_count=int(outside.sum()),
                blocker_colors=len(np.unique(grid[inside])), shadow_count=int(shadow.sum()),
                grouped=any(len(g)>1 for g in effective_groups),
                transverse_bar=any(len(g)==2 and len({x for y,x in g})==1 for g in effective_groups),
                rectangle=any(len(g)>=4 and len(g)==(max(y for y,x in g)-min(y for y,x in g)+1)*(max(x for y,x in g)-min(x for y,x in g)+1) for g in effective_groups),
                irregular_group=any(len(g)>=3 and len(g)<(max(y for y,x in g)-min(y for y,x in g)+1)*(max(x for y,x in g)-min(x for y,x in g)+1) for g in effective_groups),
                straddling=any(any(not footprint[y,x] for y,x in g) for g in effective_groups),
                internal=bool(np.any(inside & a['chamber'])),
                reservoir_blocker=bool(np.any(inside[:,:terminal_start])),
                mouth_blocker=bool(np.any(inside[:,a['mouth']:a['mouth']+2])),
                far_blocker=bool(np.any(inside[:,a['mouth']+2:])),
                edge_blocker=bool(np.any(inside[:,-2:])),
                repeated_lane=bool(np.any(inside.sum(axis=1)>1)),
                shared_extra_color=bool(set(grid[inside]) & set(grid[outside])),
                clipped=bool(np.any(footprint[0]) or np.any(footprint[-1])))


def measure_pair(inp: np.ndarray) -> dict:
    """Task measurements reconstruct actual returned inputs, not sampler metadata."""
    return _features_from_analysis(_inspect_input(inp))


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    a = _inspect_input(inp)
    assert np.array_equal(_predict_from_input(inp), out), 'input-only prediction mismatch'
    changed = inp != out
    assert np.any(changed)
    assert np.all(inp[changed] == a['background'])
    assert np.all(out[changed] == a['payload'])
    assert np.array_equal(inp[inp != a['background']], out[inp != a['background']])
    return _features_from_analysis(a)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair['input'].shape == pair['output'].shape
            key = _array_key(pair['input'])
            assert key not in seen
            seen.add(key)
            assert not _is_source_pair(pair['input'], pair['output'])
            features[split].append(_analyze_pair(pair['input'], pair['output']))
    assert any(f['inside_count'] == 0 for f in features['train'])
    assert any(f['shadow_count'] > 0 and f['blocker_colors'] >= 2 for f in features['train'])
    assert len({f['outline'] for f in features['train']}) >= 2
    # No palette, compass direction, length or obstacle count is a split invariant.
    # The invariant convention is demonstrated clean filling plus color-independent
    # opacity, including every cadence used in either split being locally readable.
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert np.array_equal(_predict_from_input(pair['input']), pair['output'])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)
    complexity = int(rng.integers(3))
    shared_palette = rng.random() < 0.25
    palette = rng.permutation(10)
    orientation_regime = str(rng.choice(['mixed', 'mixed', 'shared', 'held_test', 'held_train', 'recombine']))
    preferred = int(rng.integers(4))
    alternatives = [k for k in range(4) if k != preferred]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, regime, count in (('train', 'source_train', num_train),
                                 ('test', 'target_test', num_test)):
        for index in range(count):
            teaching = regime == 'source_train' and index == 1
            style = ('clean' if regime == 'source_train' and index == 0 else
                     str(rng.choice(['clean', 'light', 'dense'], p=[0.28, 0.40, 0.32])))
            if teaching:
                style = 'light'
            if orientation_regime == 'shared':
                rotation = preferred
            elif orientation_regime == 'held_test':
                rotation = preferred if split == 'test' else int(rng.choice(alternatives))
            elif orientation_regime == 'held_train':
                rotation = preferred if split == 'train' else int(rng.choice(alternatives))
            elif orientation_regime == 'recombine' and split == 'test':
                rotation = preferred
                style = str(rng.choice(['light', 'dense']))
            elif orientation_regime == 'recombine' and index == 0:
                rotation = preferred
            else:
                rotation = int(rng.integers(4))
            colors = palette if shared_palette and orientation_regime != 'recombine' else rng.permutation(10)
            projection_regime = (('long' if split == 'test' else 'short')
                                 if orientation_regime == 'recombine' else 'any')
            accepted = False
            for attempt in range(40):
                pair = _build_pair(rng, colors, rotation, complexity, style, projection_regime, teaching)
                try:
                    f = _analyze_pair(pair['input'], pair['output'])
                except (AssertionError, ValueError):
                    continue
                if teaching and not (f['shadow_count'] > 0 and f['blocker_colors'] >= 2):
                    continue
                key = _array_key(pair['input'])
                if key in seen or _is_source_pair(pair['input'], pair['output']):
                    continue
                if split == 'train' and index == 1 and f['outline'] == measure_pair(puzzle['train'][0]['input'])['outline']:
                    continue
                puzzle[split].append(pair)
                seen.add(key)
                accepted = True
                break
            if not accepted:
                # Deterministic clean-geometry fallback; retain palette, direction
                # and the required teaching role instead of failing a legal seed.
                fallback = np.random.default_rng(7919 + 101 * index + (split == 'test'))
                for _ in range(40):
                    pair = _build_pair(fallback, colors, rotation, 1, style, projection_regime, teaching)
                    try:
                        f = _analyze_pair(pair['input'], pair['output'])
                    except (AssertionError, ValueError):
                        continue
                    key = _array_key(pair['input'])
                    if key in seen or _is_source_pair(pair['input'], pair['output']):
                        continue
                    if teaching and not (f['shadow_count'] > 0 and f['blocker_colors'] >= 2):
                        continue
                    puzzle[split].append(pair)
                    seen.add(key)
                    accepted = True
                    break
            assert accepted, 'bounded construction exhausted'
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    # Fixed source fixtures are regressions, never a prediction lookup table.
    sources = []
    for inp, out in sorted(_SOURCE_PAIR_TEXTS):
        pair = {role: np.array([[int(c) for c in row] for row in text.splitlines()], dtype=np.int8)
                for role, text in (('input', inp), ('output', out))}
        sources.append(pair)
        assert np.array_equal(_predict_from_input(pair['input']), pair['output']), 'source regression'
    source_query = next(p for p in sources if p['input'].shape == (15, 19))
    source_train = [p for p in sources if p is not source_query]
    assert np.array_equal(predict(source_train, source_query['input'].copy()), source_query['output'])
    # Hand-written phase regressions: virtual thin corner cells stay background;
    # the adjacent inner cells fill. No construction footprint helper is used.
    for lower in ((5,4,3,2), (5,5,4,4,3,3), (5,4,4,3,3,2),
                  (6,6,6,5,5,5,4,4,4)):
        mouth = len(lower)
        grid = np.zeros((15, mouth + 4), dtype=np.int8)
        grid[lower[0]:15-lower[0],0] = 1
        for x,y in enumerate(lower,1):
            grid[y,x] = grid[14-y,x] = 1
        grid[7,1] = 2
        predicted = _predict_from_input(grid)
        next_boundary = {4:1, 6:2, 9:3}[mouth]
        assert predicted[next_boundary,mouth+1] == 0
        assert predicted[next_boundary+1,mouth+1] == 2
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    records = []
    for p in puzzles:
        validate_generated_puzzle(p)
        train = [{k: v.copy() for k,v in pair.items()} for pair in p['train']]
        for pair in p['test']:
            assert np.array_equal(predict(train, pair['input'].copy()), pair['output'])
        records.extend((split, measure_pair(pair['input'])) for split in ('train','test') for pair in p[split])
    features = [f for _,f in records]
    assert len({f['outline'] for f in features}) >= 100
    assert len({f['shape'] for f in features}) >= 70
    assert len({f['payload_shape'] for f in features}) >= 25
    assert len({f['palette'] for f in features}) >= 80
    for split in ('train','test'):
        fs = [f for s,f in records if s == split]
        assert {f['direction'] for f in fs} == set(_DIRECTIONS)
        assert any(f['inside_count'] == 0 for f in fs)
        assert any(f['inside_count'] >= 5 for f in fs)
        assert min(f['projection'] for f in fs) <= 3 and max(f['projection'] for f in fs) >= 15
    assert {r for f in features for r in f['runs']} == {1,2,3}
    assert any(f['runs'][0] != f['runs'][1] for f in features)
    assert max(f['rear'] for f in features) >= 7
    assert any(f['shape'] == (30,30) for f in features)
    assert min(min(f['shape']) for f in features) <= 9
    for flag in ('nonmonotone','asymmetric','payload_offset','single_offcenter','payload_irregular',
                 'grouped','transverse_bar','rectangle','irregular_group',
                 'straddling','internal','reservoir_blocker','mouth_blocker','far_blocker','edge_blocker',
                 'repeated_lane','shared_extra_color','clipped'):
        assert sum(bool(f[flag]) for f in features) >= 4, flag
    varied = sum(len({measure_pair(pair['input'])['palette'] for pair in p['train']+p['test']})>1 for p in puzzles)
    shared = sum(len({measure_pair(pair['input'])['palette'] for pair in p['train']+p['test']})==1 for p in puzzles)
    assert varied >= 20 and shared >= 3
    for name in ('neck_position','lateral_margin','mouth_width','min_channel_width','payload_axial'):
        assert len({f[name] for f in features}) >= 4, name
    assert sum(f['runs'][0]==f['runs'][1] and f['phases'][0]!=f['phases'][1] for f in features)>=10
    assert all({f['palette'][role] for f in features} == set(range(10)) for role in range(3))
    assert sum(f['nonmonotone'] and f['asymmetric'] and f['grouped'] and f['payload_offset'] for f in features)>=15
    regimes = dict(shared=0, held_test=0, held_train=0, longer=0, comparable=0, clean_first=0)
    for p in puzzles:
        tf = [measure_pair(pair['input']) for pair in p['train']]
        qf = [measure_pair(pair['input']) for pair in p['test']]
        td, qd = {f['direction'] for f in tf}, {f['direction'] for f in qf}
        regimes['shared'] += len(td | qd) == 1
        regimes['held_test'] += bool(qd - td)
        regimes['held_train'] += bool(td - qd)
        regimes['longer'] += min(f['projection'] for f in qf)>max(f['projection'] for f in tf)
        regimes['comparable'] += max(f['projection'] for f in qf)>=min(f['projection'] for f in tf) and min(f['projection'] for f in qf)<=max(f['projection'] for f in tf)
        regimes['clean_first'] += tf[0]['inside_count'] == 0
    assert all(regimes[name]>=3 for name in ('shared','held_test','held_train','longer','comparable'))
    assert 5<=regimes['clean_first']<=35, 'demonstration order collapsed'
    for name in ('runs','outline','payload_shape'):
        assert sum(len({measure_pair(pair['input'])[name] for pair in p['train']+p['test']})>1 for p in puzzles)>=25
    for seed in (0,13,41):
        assert _puzzles_equal(puzzles[seed], generate_puzzle(seed))
    for i,(a,b) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None),(6,3))):
        p = generate_puzzle(1000+i,a,b)
        validate_generated_puzzle(p)
        assert a is None or len(p['train']) == a
        assert b is None or len(p['test']) == b
        assert _puzzles_equal(p,generate_puzzle(1000+i,a,b))
    for a,b in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4),(True,1),(3,1.0)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    # Deliberate label corruption must be rejected by the input-only validator.
    p = generate_puzzle(83,2,1)
    damaged = {s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in ('train','test')}
    pair = damaged['test'][0]
    y,x = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][y,x] = pair['input'][y,x]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError('corrupted output accepted')
