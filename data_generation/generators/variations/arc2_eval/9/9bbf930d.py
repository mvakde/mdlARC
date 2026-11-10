import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_count_value(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _check_count_value(num_train, "num_train", 2, 10)
    _check_count_value(num_test, "num_test", 1, 4)

    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        largest_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, largest_test + 1))

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError("could not select a legal count combination")

    return num_train, num_test



def _validate_basic_grid(
    grid: np.ndarray,
    label: str,
) -> None:
    _require(
        isinstance(grid, np.ndarray),
        f"{label} must be a NumPy array",
    )
    _require(
        grid.ndim == 2,
        f"{label} must be two-dimensional",
    )
    _require(
        1 <= grid.shape[0] <= 30
        and 1 <= grid.shape[1] <= 30,
        f"{label} dimensions are outside 1..30",
    )
    _require(
        np.issubdtype(grid.dtype, np.integer),
        f"{label} must have an integer dtype",
    )
    _require(
        bool(np.all((grid >= 0) & (grid <= 9))),
        f"{label} contains a non-ARC color",
    )



_SOURCE_INPUTS = (('6711111111111777', '6777777777777777', '6711111111111474', '6777777777777474', '6744444444444474', '6777777777777774', '6744444444444444', '6777777777777777', '6700000000000077', '6777777777777777', '6779999999999999', '6777777777777779', '6779999999999979', '6777777777777979', '6778888888888877', '6777777777777777', '6733333333333377'), ('67333333333377', '67777777777777', '67888888888877', '67777777777777', '67111111111177', '67777777777777', '67111111111177', '67777777777777', '67555555555577', '67777777777777', '67000000000077', '67777777777777', '67000000000077', '67777777777777', '67999999999977', '67777777777777', '67333333333377', '67777777777777', '67333333333377'), ('6744444444444444', '6777777774777774', '6744444474744474', '6777777474747474', '6722222277777474', '6777777777777777', '6722222275757970', '6777777775757970', '6755555555757970', '6777777777757970', '6755555555557970', '6777777777777777', '6733333777333333', '6777777777777773', '6733333777333373', '6777777777777373', '6700000000000077'), ('679999999777777799999', '677777777777777777779', '679999999777707099979', '677777777777707077979', '670000000777007077979', '677777777777777077979', '670000000777000077979', '677777777777770777777', '670000000777070888877', '677777771717070777777', '678888871717070888877', '677777771717070777777', '671111111717070777777', '677777777717070007777', '671111111117077777777', '677777777777000007777', '674444444477744447777', '677777777777777747777', '674444444427244747777', '677777777727274747979', '672222222227274749979', '677777777777277777779', '672222222222277779999'))


def _signature(a):
    return a.shape, a.tobytes()


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _plus(p, d):
    return p[0] + d[0], p[1] + d[1]


def _normalise(inp):
    """Infer visible roles, and rotate the distinguished strip to the left."""
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        token = int(g[0, 0])
        if np.all(g[:, 0] == token) and np.count_nonzero(g == token) == len(g):
            candidates.append((g, k, token))
    _require(len(candidates) == 1, 'one unique full-edge token strip is required')
    g, k, token = candidates[0]
    # The unpainted approach column identifies background without output labels.
    background = int(g[0, 1])
    _require(background != token and np.all(g[:, 1] == background),
             'the approach next to the strip must be clear')
    return g, k, token, background


def _paired(g, p, d, bg, token):
    normal = (-d[1], d[0])
    a = _plus(p, normal)
    b = _plus(p, (-normal[0], -normal[1]))
    if not (_inside(g, a) and _inside(g, b)):
        return None
    color = int(g[a])
    if color == int(g[b]) and color not in (bg, token):
        return color
    return None


def _analyse(inp, ignore_matching=False):
    """Read rails directionally. No components, path templates, or output labels.

    Inertia across absent or interrupted rails is established by the source.
    A blocked mover turns only if the next free cell has two matching flanks.
    Both possible turns are examined; there is no arbitrary turn priority.
    A later turn uses local equality, not a remembered entrance identity.
    """
    g, rotation, token, bg = _normalise(inp)
    h, w = g.shape
    routes, mismatches = [], []
    for row in range(1, h - 1):
        entrance = None
        for col in range(1, w):
            if int(g[row, col]) != bg:
                break
            a, b = int(g[row - 1, col]), int(g[row + 1, col])
            if a not in (bg, token) and b not in (bg, token):
                if a == b or ignore_matching:
                    entrance = (row, col)
                else:
                    mismatches.append((row, col, a, b))
                # The first visible opening decides selection, even if unrelated
                # matching strands occur farther along this inactive row.
                break
        if entrance is None:
            continue
        p, direction = (row, 1), (0, 1)
        path, directions, turns, seen = [p], [], [], set()
        while True:
            state = p, direction
            _require(state not in seen, 'guide route cycles')
            seen.add(state)
            nxt = _plus(p, direction)
            if not _inside(g, nxt):
                break
            if int(g[nxt]) == bg:
                directions.append(direction)
                p = nxt
                path.append(p)
                continue
            choices = []
            for d in ((-direction[1], direction[0]), (direction[1], -direction[0])):
                q = _plus(p, d)
                if (_inside(g, q) and int(g[q]) == bg
                        and _paired(g, q, d, bg, token) is not None):
                    choices.append(d)
            _require(len(choices) <= 1, 'matching rails offer two possible turns')
            if not choices:
                break
            turns.append((p, direction, choices[0]))
            direction = choices[0]
        _require(p[1] != 0 and g[p] == bg, 'invalid terminal')
        _require(len(path) == len(set(path)), 'route revisits itself')
        # Measure missing rail evidence from the reconstructed route itself.
        support = [_paired(g, q, d, bg, token) is not None
                   for q, d in zip(path[1:], directions)]
        first = next((i for i, yes in enumerate(support) if yes), len(support))
        last = max((i for i, yes in enumerate(support) if yes), default=-1)
        gaps, run_start = [], None
        for i in range(first, last + 1):
            if support[i]:
                if run_start is not None:
                    # Count interruptions of a straight guide, not the
                    # naturally unsupported cell at an orthogonal corner.
                    if len(set(directions[run_start - 1:i + 1])) == 1:
                        gaps.append(i - run_start)
                    run_start = None
            elif run_start is None:
                run_start = i
        terminal_open = p[0] in (0, h - 1) or p[1] == w - 1
        segments = []
        for d in directions:
            if segments and segments[-1][0] == d:
                segments[-1] = (d, segments[-1][1] + 1)
            else:
                segments.append((d, 1))
        routes.append(dict(start=(row, 0), destination=p, path=path,
                           rail=int(g[row - 1, entrance[1]]),
                           approach=entrance[1] - 1, turns=turns,
                           segments=segments, gaps=gaps,
                           tail=len(support) - 1 - last, open=terminal_open,
                           exit_direction=direction))
    destinations = [r['destination'] for r in routes]
    _require(len(destinations) == len(set(destinations)), 'destinations conflict')
    occupied = {}
    for r in routes:
        corners = {p for p, before, after in r['turns']}
        for i, p in enumerate(r['path']):
            before = r['path'][max(0, i - 1)]
            after = r['path'][min(len(r['path']) - 1, i + 1)]
            axis = (int(before[0] != after[0]), int(before[1] != after[1]))
            if p in occupied:
                other_axis, other_corner = occupied[p]
                # Source rays cross in unmarked space while retaining their
                # directions. They do not merge or turn at the intersection.
                _require(axis != other_axis and p not in corners and not other_corner,
                         'routes merge or intersect at a bend')
            occupied[p] = axis, p in corners
    return dict(grid=g, rotation=rotation, token=token, background=bg,
                guides=set(map(int, np.unique(g))) - {token, bg},
                routes=routes, mismatches=mismatches)


def _answer(analysis):
    out = analysis['grid'].copy()
    for route in analysis['routes']:
        out[route['start']] = analysis['background']
        out[route['destination']] = analysis['token']
    return np.rot90(out, -analysis['rotation']).copy()


def solve(train_pairs, test_input):
    """Prediction uses only visible training pairs and the supplied input."""
    for pair in train_pairs:
        _require(np.array_equal(_answer(_analyse(pair['input'])), pair['output']),
                 'training contradicts directional matching-rail movement')
    return _answer(_analyse(test_input))


def _features(analysis):
    routes = analysis['routes']
    return {name for name, present in (
        ('matching', bool(routes)), ('mismatch', bool(analysis['mismatches'])),
        ('turn', any(r['turns'] for r in routes)),
        ('gap', any(r['gaps'] for r in routes)),
        ('tail', any(r['tail'] > 0 for r in routes)),
        ('open', any(r['open'] for r in routes)),
        ('closed', any(not r['open'] for r in routes))) if present}


def _sample_parameters(rng):
    # These are distribution regimes, not hidden commands. Roles are visible.
    return dict(tier=int(rng.choice([0, 1, 2], p=[.18, .36, .46])),
                mixing=int(rng.integers(0, 3)),
                orientation_mode=int(rng.integers(0, 3)),
                rotation=int(rng.integers(0, 4)),
                colors=tuple(map(int, rng.permutation(10))),
                palette_size=int(rng.integers(2, 9)))


def _pair_roles(rng, params, index, is_test):
    colors = list(params['colors'])
    if params['mixing'] == 1:
        # Keep token recognizable across pairs while changing other roles.
        colors[1:] = list(map(int, rng.permutation(colors[1:])))
    elif params['mixing'] == 2:
        colors = list(map(int, rng.permutation(10)))
    size = params['palette_size'] if params['mixing'] == 0 else int(rng.integers(2, 9))
    k = params['rotation']
    if params['orientation_mode'] == 1:
        k = (k + index) % 4
    elif params['orientation_mode'] == 2 and is_test:
        k = (k + int(rng.integers(1, 4))) % 4
    return colors[0], colors[1], colors[2:2 + size], k


def _straight_scene(rng, h, w, token, bg, palette, margin, demonstration):
    """A variable rail stack: adjacent openings may share a rail.

    This reproduces the source's selection challenge, without assigning an
    inactive meaning to a particular color. Different rail lengths, gaps and
    independent caps make active and inactive geometries comparable.
    """
    maximum = (h - 1) // 2
    minimum = len(palette) + 2 if maximum >= len(palette) + 2 and rng.random() < .7 else 4
    count = int(rng.integers(minimum, maximum + 1))
    top = int(rng.integers(0, h - (2 * count + 1) + 1))
    rail_colors = [int(rng.choice(palette))]
    flags = list(map(bool, rng.random(count) < rng.uniform(.3, .8)))
    # Two matched openings of a repeated identity, plus a true mismatch.
    flags[0:3] = [True, True, False]
    if count >= len(palette) + 2:
        # Leave enough mismatched boundaries to display a rich actual palette,
        # instead of merely sampling colors that are never painted.
        for i in range(2, len(palette) + 1):
            flags[i] = False
    rng.shuffle(flags)
    for same in flags:
        options = [v for v in palette if v not in rail_colors]
        if not options:
            options = [v for v in palette if v != rail_colors[-1]]
        rail_colors.append(rail_colors[-1] if same else int(rng.choice(options)))
    g = np.full((h, w), bg, dtype=np.int8)
    g[:, 0] = token
    max_tail = min(5, w - margin - 5)
    ends = [w - 1 - int(rng.integers(1, max_tail + 1)) for _ in rail_colors]
    staggered = rng.random() < .6
    starts = [int(rng.integers(2, max(2, min(5, end - 2)) + 1)) if staggered else margin
              for end in ends]
    for i, (color, end, start) in enumerate(zip(rail_colors, ends, starts)):
        g[top + 2 * i, start:end + 1] = color
    active = [i for i, same in enumerate(flags) if same]
    rng.shuffle(active)
    for i in range(count):
        # Closed/open choices are unrelated to match status or guide identity.
        closed = bool(rng.integers(0, 2))
        if i == active[0]:
            closed = False
        if i == active[1]:
            closed = True
        if closed:
            end = int(rng.integers(max(max(starts[i:i + 2]) + 3, min(ends[i:i + 2]) + 1), w))
            g[top + 2 * i + 1, end] = int(rng.choice(palette))
    # An aligned interruption is visible before and after the gap.
    i = int(rng.choice(active))
    end = min(ends[i:i + 2])
    start = max(starts[i:i + 2])
    if end - start >= 5:
        begin = int(rng.integers(start + 2, end - 1))
        length = int(rng.integers(1, min(4, end - begin) + 1))
        g[top + 2 * i, begin:begin + length] = bg
        g[top + 2 * i + 2, begin:begin + length] = bg
    return g


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = (int(np.sign(target[0] - p[0])), int(np.sign(target[1] - p[1])))
        if (d[0] == 0) == (d[1] == 0):
            return None
        while p != target:
            p = _plus(p, d)
            path.append(p)
    cells = set(path)
    if len(cells) != len(path):
        return None
    # Prevent self contacts, while permitting row/column ranges to overlap.
    for i, p in enumerate(path):
        neighbors = sum(_plus(p, d) in cells for d in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if neighbors != (1 if i in (0, len(path) - 1) else 2):
            return None
    return path


def _random_path(rng, h, w, margin, bends, closed=None):
    r = int(rng.integers(1, h - 1))
    points, direction = [(r, 1)], (0, 1)
    if closed is None:
        closed = bool(rng.integers(0, 2))
    for j in range(bends + 1):
        p = points[-1]
        if j == bends and not closed and direction != (0, -1):
            target = ((h - 1 if direction[0] > 0 else 0), p[1]) if direction[0] else (p[0], w - 1)
        else:
            choices = []
            for length in range(2, max(h, w)):
                q = p[0] + length * direction[0], p[1] + length * direction[1]
                if not (1 <= q[0] <= h - 2 and margin + 2 <= q[1] <= w - 2):
                    continue
                if j == 0 and q[1] < margin + 3:
                    continue
                choices.append(q)
            if not choices:
                return None
            target = choices[int(rng.integers(len(choices)))]
        points.append(target)
        direction = ((-direction[1], direction[0]) if rng.integers(0, 2)
                     else (direction[1], -direction[0]))
    return _polyline(points)


def _route_walls(path, h, w, margin, color):
    cells, walls = set(path), {}
    for p in path:
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = _plus(p, d)
            if 0 <= q[0] < h and margin <= q[1] < w and q not in cells:
                walls[q] = color
    return walls


def _decorate_walls(rng, path, walls, bg, palette, margin, tier):
    directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    # Only remove straight-run flanks. The forward blocker and outgoing pair at
    # every bend are retained, so continuation remains visible and unique.
    eligible = []
    for i in range(3, len(path) - 3):
        if path[i][1] < margin + 2 or len(set(directions[i - 2:i + 2])) != 1:
            continue
        d = directions[i]
        n = (-d[1], d[0])
        flanks = (_plus(path[i], n), _plus(path[i], (-n[0], -n[1])))
        if all(q in walls for q in flanks):
            eligible.append((i, flanks))
    if eligible and rng.random() < (.6 if tier else .35):
        start, _ = eligible[int(rng.integers(len(eligible)))]
        length = int(rng.integers(1, 2 + tier))
        for i, flanks in eligible:
            if start <= i < start + length:
                for q in flanks:
                    walls.pop(q, None)
    # Independently remove support before an edge exit or an internal cap.
    if len(path) > 5 and rng.random() < .8:
        d = directions[-1]
        n = (-d[1], d[0])
        tail = int(rng.integers(1, 5))
        for i in range(len(path) - 1, max(2, len(path) - 1 - tail), -1):
            if directions[i - 1] != d or (i >= 2 and directions[i - 2] != d):
                break
            for q in (_plus(path[i], n), _plus(path[i], (-n[0], -n[1]))):
                walls.pop(q, None)
    cap = _plus(path[-1], directions[-1])
    if cap in walls:
        walls[cap] = int(rng.choice(palette))


def _packed_scene(rng, h, w, token, bg, palette, margin, tier, force_bent):
    g = np.full((h, w), bg, dtype=np.int8)
    g[:, 0] = token
    intended, reserved = {}, set()
    desired = int(rng.integers(1, min(10, (h - 1) // 2) + 1))
    max_bends = 3 if tier == 1 else 7

    def place(path, color, inactive=False):
        nonlocal g, reserved
        if path is None or len(path) < margin + 3:
            return False
        cells = set(path)
        if cells.intersection(reserved) or any(g[q] != bg for q in cells):
            return False
        first_turn = next((a[1] for a, b in zip(path, path[1:]) if a[0] != b[0]), path[-1][1])
        local_margin = int(rng.integers(2, max(2, min(5, first_turn - 2)) + 1))
        walls = _route_walls(path, h, w, local_margin, color)
        if inactive:
            other = int(rng.choice([v for v in palette if v != color]))
            for q in walls:
                if q[0] > path[0][0]:
                    walls[q] = other
        else:
            _decorate_walls(rng, path, walls, bg, palette, local_margin, tier)
        if any(q in reserved or (g[q] != bg and g[q] != v) for q, v in walls.items()):
            return False
        candidate = g.copy()
        for q, v in walls.items():
            candidate[q] = v
        wanted = dict(intended)
        if not inactive:
            wanted[(path[0][0], 0)] = path[-1]
        try:
            a = _analyse(candidate)
        except AssertionError:
            return False
        actual = {r['start']: r['destination'] for r in a['routes']}
        if actual != wanted or (inactive and not a['mismatches']):
            return False
        if not inactive:
            # The sampled geometry is checked against the visible route, but
            # is never used to compute an output.
            route = next(r for r in a['routes'] if r['start'] == (path[0][0], 0))
            if route['path'] != path:
                return False
        g, reserved = candidate, reserved | cells
        intended.clear()
        intended.update(wanted)
        return True

    # Put the complicated route down first, then pack other paths around it.
    for attempt in range(180):
        bends = int(rng.integers(1, max_bends + 1)) if force_bent else int(rng.integers(0, max_bends + 1))
        if place(_random_path(rng, h, w, margin, bends), int(rng.choice(palette))):
            break
    if not intended:
        # Deterministic, finite fallback for a completely empty scene.
        path = _polyline([(h // 2, 1), (h // 2, margin + 4), (0, margin + 4)])
        _require(place(path, palette[0]), 'empty-board elbow must fit')
    # Short mismatched openings consume little of the area used by later bends.
    for row in map(int, rng.permutation(np.arange(1, h - 1))):
        path = _polyline([(row, 1), (row, int(rng.integers(margin + 3, w)))])
        if place(path, int(rng.choice(palette)), inactive=True):
            break
    if not _analyse(g)['mismatches']:
        return None
    for attempt in range(200):
        if len(intended) >= desired:
            break
        # Zero through many turns are sampled independently for each route.
        bends = int(rng.choice(np.arange(max_bends + 1), p=np.array([5] + [2] * max_bends) / (5 + 2 * max_bends)))
        place(_random_path(rng, h, w, margin, bends), int(rng.choice(palette)))
    # Additional inactive structures vary the selection ratio independently.
    for _ in range(int(rng.integers(0, 4))):
        place(_random_path(rng, h, w, margin, 0), int(rng.choice(palette)), inactive=True)
    return g


def _construct_pair(rng, regime, index, params, roles):
    token, bg, palette, rotation = roles
    is_test = regime == 'target_test'
    straight = params['tier'] == 0 or (not is_test and index == 0) or rng.random() < .22
    if not is_test and index == 1 and params['tier']:
        straight = False
    # Tests favor more area, but both regimes overlap and neither prescribes
    # route counts, a size list, or a turn-count composition.
    lower = 13 if is_test else 11
    h = int(rng.integers(lower, 31))
    w = int(rng.integers(12 if is_test else 10, 31))
    if rng.random() < .07:
        h = w = 30
    margin = int(rng.integers(2, min(6, w - 6)))
    if straight:
        g = _straight_scene(rng, h, w, token, bg, palette, margin, not is_test and index == 0)
    else:
        g = _packed_scene(rng, h, w, token, bg, palette, margin, params['tier'], True)
        if g is None:
            return None
    inp = np.rot90(g, rotation).copy()
    try:
        analysis = _analyse(inp)
    except AssertionError:
        return None
    if not analysis['routes'] or not analysis['mismatches']:
        return None
    return dict(input=inp, output=_answer(analysis))


def _fallback_elbow(h, w, token, bg, palette):
    """Guaranteed spare layout, used only if random packing is exhausted."""
    g = np.full((h, w), bg, dtype=np.int8)
    g[:, 0] = token
    path = _polyline([(h // 2, 1), (h // 2, w // 2), (0, w // 2)])
    for q, color in _route_walls(path, h, w, 2, palette[0]).items():
        g[q] = color
    g[h - 3, 2:w - 1] = palette[0]
    g[h - 1, 2:w - 1] = palette[1]
    return g


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    params = _sample_parameters(rng)
    puzzle, seen, coverage = {'train': [], 'test': []}, set(), set()
    for key, count, regime in (('train', num_train, 'source_train'), ('test', num_test, 'target_test')):
        for index in range(count):
            roles = _pair_roles(rng, params, index, key == 'test')
            for attempt in range(80):
                pair = _construct_pair(rng, regime, index, params, roles)
                if pair is None or _signature(pair['input']) in seen or _is_source(pair['input']):
                    continue
                a = _analyse(pair['input'])
                features = _features(a)
                if key == 'test' and not features <= coverage:
                    continue
                if key == 'train' and index == 0 and not {'matching', 'mismatch', 'open', 'closed', 'tail', 'gap'} <= features:
                    continue
                if key == 'train' and index == 1 and params['tier'] and 'turn' not in features:
                    continue
                if key == 'train' and index == 1:
                    prev = _analyse(puzzle['train'][0]['input'])
                    if len(a['routes']) == len(prev['routes']) or a['grid'].shape == prev['grid'].shape:
                        continue
                    if len({r['rail'] for r in a['routes'] + prev['routes']}) < 2:
                        continue
                seen.add(_signature(pair['input']))
                puzzle[key].append(pair)
                if key == 'train':
                    coverage.update(features)
                break
            else:
                # A finite fallback retains all demonstrated conventions and
                # can always use fresh dimensions/rail-stack contents.
                for serial in range(200):
                    h, w = 15 + serial % 16, 13 + (serial // 16) % 16
                    if key == 'train' and index == 1 and params['tier']:
                        previous_colors = {r['rail'] for r in _analyse(puzzle['train'][0]['input'])['routes']}
                        palette = sorted(roles[2], key=lambda c: c in previous_colors)
                        g = _fallback_elbow(h, w, roles[0], roles[1], palette)
                    else:
                        g = _straight_scene(rng, h, w, roles[0], roles[1], roles[2], 2, True)
                    inp = np.rot90(g, roles[3]).copy()
                    a = _analyse(inp)
                    if _signature(inp) in seen or _is_source(inp):
                        continue
                    if key == 'test' and not _features(a) <= coverage:
                        continue
                    if key == 'train' and index == 0 and not {'open', 'closed', 'gap', 'tail'} <= _features(a):
                        continue
                    if key == 'train' and index == 1:
                        first = _analyse(puzzle['train'][0]['input'])
                        if len(a['routes']) == len(first['routes']) or a['grid'].shape == first['grid'].shape:
                            continue
                        if len({r['rail'] for r in a['routes'] + first['routes']}) < 2:
                            continue
                    puzzle[key].append(dict(input=inp, output=_answer(a)))
                    seen.add(_signature(inp))
                    coverage.update(_features(a))
                    break
                else:
                    raise AssertionError('bounded construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _is_source(inp):
    rows = tuple(''.join(map(str, row.tolist())) for row in inp)
    return rows in _SOURCE_INPUTS


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[k], list) for k in puzzle), 'split lists required')
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, 'pair counts')
    seen, train_features, train_analyses = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = pair['input'], pair['output']
            _validate_basic_grid(inp, 'input')
            _validate_basic_grid(out, 'output')
            _require(inp.shape == out.shape, 'dimensions must be preserved')
            signature = _signature(inp)
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            _require(not _is_source(inp), 'exact source input copied')
            a = _analyse(inp)
            _require(a['routes'] and a['mismatches'], 'matched and mismatched openings required')
            paths = [p for r in a['routes'] for p in r['path']]
            _require(len(paths) == len(set(paths)), 'generated routes must be separated')
            _require(np.array_equal(_answer(a), out), 'output differs from visible rail tracing')
            token, bg = a['token'], a['background']
            _require(np.count_nonzero(inp == token) == np.count_nonzero(out == token), 'token conservation')
            guides = (inp != token) & (inp != bg)
            _require(np.array_equal(inp[guides], out[guides]), 'guide preservation')
            _require(np.count_nonzero(inp != out) == 2 * len(a['routes']), 'only starts and destinations change')
            if split == 'train':
                train_features.update(_features(a))
                train_analyses.append(a)
            else:
                _require(_features(a) <= train_features, 'test convention absent from training')
    _require({'matching', 'mismatch', 'open', 'closed', 'gap', 'tail'} <= train_features,
             'training must distinguish matching, inertia, and stopping')
    _require(len({len(a['routes']) for a in train_analyses}) > 1, 'training counts must vary')
    _require(len({a['grid'].shape for a in train_analyses}) > 1, 'training dimensions must vary')
    _require(len({r['rail'] for a in train_analyses for r in a['routes']}) >= 2,
             'training must demonstrate matching across guide identities')


def measurements(puzzle):
    """Task metrics reconstructed from returned inputs, normalized for rotation."""
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _analyse(pair['input'])
            routes = a['routes']
            boxes = [(min(p[0] for p in r['path']), max(p[0] for p in r['path']),
                      min(p[1] for p in r['path']), max(p[1] for p in r['path'])) for r in routes]
            overlap = sum(max(x[0], y[0]) <= min(x[1], y[1]) and max(x[2], y[2]) <= min(x[3], y[3])
                          for i, x in enumerate(boxes) for y in boxes[i + 1:])
            nested = sum((x[0] <= y[0] and y[1] <= x[1] and y[3] <= x[3])
                         or (y[0] <= x[0] and x[1] <= y[1] and x[3] <= y[3])
                         for i, x in enumerate(boxes) for y in boxes[i + 1:])
            interleaved = sum(x[0] < y[0] <= x[1] < y[1] or y[0] < x[0] <= y[1] < x[1]
                              for i, x in enumerate(boxes) for y in boxes[i + 1:])
            active_colors = [r['rail'] for r in routes]
            used_guides = set()
            for route in routes:
                for p in route['path']:
                    for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        q = _plus(p, d)
                        if _inside(a['grid'], q) and int(a['grid'][q]) in a['guides']:
                            used_guides.add(int(a['grid'][q]))
            records.append(dict(split=split, shape=list(a['grid'].shape), rotation=a['rotation'],
                                token=a['token'], background=a['background'],
                                guides=len(a['guides']), active=len(routes), inactive=len(a['mismatches']),
                                inactive_only=len(a['guides'] - used_guides),
                                repeated=len(set(active_colors)) < len(active_colors),
                                turns=[len(r['turns']) for r in routes],
                                lengths=[len(r['path']) for r in routes],
                                approaches=[r['approach'] for r in routes],
                                gaps=[n for r in routes for n in r['gaps']],
                                tails=[r['tail'] for r in routes],
                                open=sum(r['open'] for r in routes),
                                closed=sum(not r['open'] for r in routes),
                                adjacent=sum(r['open'] and r['destination'][0] in (0, a['grid'].shape[0] - 1) for r in routes),
                                transverse=[d[0] * n for r in routes for d, n in r['segments'] if d[0]],
                                returns=sum(any(d == (0, -1) for d, n in r['segments']) for r in routes),
                                both_transverse=sum({d[0] for d, n in r['segments'] if d[0]} == {-1, 1} for r in routes),
                                overlap=overlap,
                                nested=nested, interleaved=interleaved,
                                palette=sorted(a['guides']),
                                inactive_rows=[r for r, c, x, y in a['mismatches']],
                                selection_spacing=min(abs(r['start'][0] - row) for r in routes
                                                      for row, col, x, y in a['mismatches']),
                                closed_unflanked=sum(not r['open'] and _paired(a['grid'], r['destination'], r['exit_direction'], a['background'], a['token']) is None for r in routes),
                                foreign_caps=sum(not r['open'] and int(a['grid'][_plus(r['destination'], r['exit_direction'])]) != r['rail'] for r in routes),
                                terminals=[r['exit_direction'] for r in routes]))
    return records


def self_test() -> None:
    puzzles, records, counts = [], [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        records.extend(measurements(p))
        counts.add((len(p['train']), len(p['test'])))
        for pair in p['test']:
            prediction = solve([{'input': x['input'].copy(), 'output': x['output'].copy()}
                                for x in p['train']], pair['input'].copy())
            _require(np.array_equal(prediction, pair['output']), 'input-only prediction')
    _require(len({n for n, m in counts}) >= 4 and len({m for n, m in counts}) >= 3, 'automatic counts collapse')
    for n, m in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(173, n, m)
        q = generate_puzzle(173, n, m)
        _require(all(np.array_equal(a[k], b[k]) for split in p for a, b in zip(p[split], q[split])
                     for k in ('input', 'output')), 'determinism')
        _require(n is None or len(p['train']) == n, 'explicit training count')
        _require(m is None or len(p['test']) == m, 'explicit test count')
    for n, m in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.2, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    _require(len({_signature(p['train'][0]['input']) for p in puzzles}) == 42, 'cross-seed difference')
    # Source regressions include open margins, the green gap, local color
    # handoff in the compact bends, and stops against unrelated guide colors.
    expected = [({1, 5, 11}, {(1, 15), (0, 14), (16, 14)}),
                ({5, 11, 17}, {(5, 13), (11, 13), (17, 13)}),
                ({1, 5, 9, 13}, {(7, 8), (5, 15), (11, 14), (16, 14)}),
                ({1, 5, 7, 13, 17, 21}, {(21, 13), (0, 14), (14, 20), (0, 9), (22, 15), (0, 11)})]
    for rows, (starts, ends) in zip(_SOURCE_INPUTS, expected):
        inp = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
        a = _analyse(inp)
        _require({r['start'][0] for r in a['routes']} == starts, 'source selection')
        _require({r['destination'] for r in a['routes']} == ends, 'source destinations')
    _require(len({tuple(r['shape']) for r in records}) > 100, 'dimension diversity')
    _require(set(range(1, 9)) <= {r['active'] for r in records}, 'active count diversity')
    _require(set(range(2, 9)) <= {r['guides'] for r in records}, 'guide palette diversity')
    _require({0, 1, 2, 3} == {r['rotation'] for r in records}, 'orientation diversity')
    _require(set(range(10)) == {r['token'] for r in records}, 'all token literals permitted')
    _require(set(range(5)) <= {n for r in records for n in r['turns']}, 'bend diversity')
    _require(all(any(r[field] for r in records) for field in
                 ('gaps', 'adjacent', 'returns', 'both_transverse', 'overlap', 'inactive_only', 'repeated')), 'geometry/selection diversity')
    _require(any(r['closed'] == 0 for r in records) and any(r['open'] == 0 for r in records), 'terminal choices collapse')
    _require(any(r['gaps'] and r['overlap'] and r['closed'] and r['adjacent'] for r in records), 'combined rich scene absent')
    _require(set(range(1, 5)) <= {n for r in records for n in r['approaches']}, 'approach margins collapse')
    _require(any(len(set(r['approaches'])) > 1 for r in records), 'approaches must also vary within scenes')
    _require(set(range(1, 5)) <= {n for r in records for n in r['gaps']}, 'straight interruption lengths collapse')
    _require(max(n for r in records for n in r['tails']) >= 4, 'unsupported final runs absent')
    transverse = {n for r in records for n in r['transverse']}
    _require(min(transverse) < -8 and max(transverse) > 8 and len(transverse) > 15, 'transverse lengths/directions collapse')
    for field in ('nested', 'interleaved', 'closed_unflanked', 'foreign_caps'):
        _require(any(r[field] for r in records), field + ' coverage absent')
    _require(len({r['selection_spacing'] for r in records}) >= 5, 'inactive/active spacing collapse')
    for field in ('rotation', 'token', 'background', 'palette'):
        distinct = [len({str(r[field]) for r in measurements(p)}) for p in puzzles]
        _require(any(n == 1 for n in distinct) and sum(n > 1 for n in distinct) >= 8,
                 field + ' must have both homogeneous and mixed puzzle regimes')
    _require(any(r['shape'] == [30, 30] for r in records), 'full-size square absent')
    for h, w in ((15, 13), (30, 28), (16, 14)):
        a = _analyse(_fallback_elbow(h, w, 6, 7, [1, 2]))
        _require(len(a['routes']) == 1 and len(a['routes'][0]['turns']) == 1
                 and a['mismatches'], 'deterministic packing fallback')
    # Mutation check: changing a mismatched rail into a matching one changes
    # selection. A geometry-only reader must not predict the same transfers.
    simple = next(p['train'][0]['input'] for p in puzzles)
    a = _analyse(simple)
    wrong = _analyse(simple, ignore_matching=True)
    _require(len(wrong['routes']) > len(a['routes']), 'colors must affect selection')
    # Removing a section of aligned straight rails leaves the answer unchanged.
    straight = np.full((9, 15), 7, dtype=np.int8)
    straight[:, 0] = 6
    straight[2, 2:12] = straight[4, 2:12] = 3
    before = _answer(_analyse(straight))
    straight[2, 6:9] = straight[4, 6:9] = 7
    after = _answer(_analyse(straight))
    _require(np.array_equal(before == 6, after == 6), 'interrupted rails must preserve transfer')
    bad = {'train': [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0]['train']],
           'test': [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0]['test']]}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('invalid transfer accepted')
