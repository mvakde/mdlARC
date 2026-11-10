"""Clockwork flipbook: periodic clocks stamp independently rotating motifs.

Only NumPy and standard Python are used. The solver reads its input alone.
Generation paints from sampled geometry; solving rotates parsed coordinates.
"""

import numpy as np


HOST_SEEDS = (
    0, 296154851, 136888478, 502438766, 242399493, 673798977, 276157055,
    859704690, 870646319, 883848, 580294641, 564592888, 320805474,
    559842024, 23254159, 693148688, 336860302, 465979221, 962761823,
    551332430, 340293277, 250703374, 326114528, 803872145, 269838835,
    761856556, 554981434, 892017713, 901576676, 783101535, 777408091,
    66295497, 489098449, 184901728, 873594669, 696027002, 346754778,
    915292806, 377743586, 987078581, 573593338, 6590464,
)

_SHORTCUTS = (
    "raw_only", "stationary", "counterclockwise", "absolute_time",
    "unbounded_terminal", "no_routing", "recolor_reference",
    "future_only", "noise_stamps",
)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    g = np.asarray(value)
    if g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(g.dtype, np.integer) or np.any((g < 0) | (g > 9)):
        raise ValueError("grid colors must be integers 0..9")
    return g


def _key(g):
    return g.shape, np.asarray(g, dtype=np.int8).tobytes()


def _puzzle_key(p):
    return tuple(tuple((_key(np.asarray(q['input'])), _key(np.asarray(q['output'])))
                       for q in p[s]) for s in ('train', 'test'))


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def _turn(r, c, n, k):
    for _ in range(k % 4):
        r, c = c, n - 1 - r
    return r, c


def _orbits(n):
    remaining = {(r, c) for r in range(n) for c in range(n)}
    result = []
    while remaining:
        r, c = min(remaining)
        orbit = list(dict.fromkeys(_turn(r, c, n, k) for k in range(4)))
        remaining.difference_update(orbit)
        result.append(orbit)
    return result


def _symmetry(mask):
    if np.array_equal(mask, np.rot90(mask)):
        return 'quarter'
    if np.array_equal(mask, np.rot90(mask, 2)):
        return 'half'
    return 'asymmetric'


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _outline(component):
    rr, cc = zip(*component)
    r, c, h, w = min(rr), min(cc), max(rr)-min(rr)+1, max(cc)-min(cc)+1
    if min(h, w) < 3:
        return None
    expected = {(y, x) for y in range(r, r+h) for x in range(c, c+w)
                if y in (r, r+h-1) or x in (c, c+w-1)}
    return (r, c, h, w) if expected == component else None


def _separate(first, second):
    r, c, h, w = first
    y, x, a, b = second
    return r+h < y or y+a < r or c+w < x or x+b < c


def _clock(values, bg, palette):
    positions = np.flatnonzero(values != bg)
    counts = {int(c): int(np.count_nonzero(values == c)) for c in np.unique(values[positions])}
    repeated = [c for c, count in counts.items() if count > 1]
    if len(repeated) != 1 or counts[repeated[0]] != 2:
        raise ValueError("a clock must have one unique equal-colored pair")
    if not set(counts) <= set(palette):
        raise ValueError("clock hues must match source motifs")
    reference = repeated[0]
    a, b = map(int, np.flatnonzero(values == reference))
    period = b-a
    aligned = [int(x) for x in positions if values[x] != reference and (x-a) % period == 0]
    if len(aligned) > 1 or (aligned and a < aligned[0] < b):
        raise ValueError("ambiguous terminal")
    terminal = aligned[0] if aligned else None
    noise = [(int(x), int(values[x])) for x in positions
             if int(x) not in (a, b, terminal)]
    return dict(a=a, b=b, period=period, reference=reference, terminal=terminal,
                destination=reference if terminal is None else int(values[terminal]), noise=noise)


def _orbit_separation(source, bg):
    used = set()
    n = len(source)
    for color in np.unique(source):
        if color == bg:
            continue
        territory = {_turn(int(r), int(c), n, k) for r, c in np.argwhere(source == color)
                     for k in range(4)}
        if used & territory:
            raise ValueError("different colors share a rotation orbit")
        used.update(territory)


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if not (np.all(g[0] == bg) and np.all(g[-1] == bg)
            and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
        raise ValueError("the input perimeter must identify the exterior background")
    candidates = []
    for marker in np.unique(g):
        marker = int(marker)
        if marker == bg:
            continue
        comps = _components(g == marker)
        if len(comps) != 2:
            continue
        boxes = [_outline(c) for c in comps]
        if any(b is None for b in boxes) or not _separate(*boxes):
            continue
        square = [b for b in boxes if b[2] == b[3]]
        table = [b for b in boxes if b[2] != b[3]]
        if len(square) != 1 or len(table) != 1:
            continue
        sr, sc, sh, sw = square[0]
        tr, tc, th, tw = table[0]
        source = g[sr+1:sr+sh-1, sc+1:sc+sw-1].copy()
        timetable = g[tr+1:tr+th-1, tc+1:tc+tw-1].copy()
        vertical = th > tw
        tracks = timetable.T if vertical else timetable
        m, t = tracks.shape
        n = len(source)
        if not (3 <= n <= 7 and 1 <= m <= 4 and 3 <= t <= 7 and m < t
                and n*t+t-1 <= 30):
            continue
        if np.any(source == marker) or np.any(tracks == marker):
            continue
        covered = np.zeros(g.shape, dtype=bool)
        for r, c, h, w in boxes:
            covered[r:r+h, c:c+w] = True
        if np.any((g != bg) & ~covered):
            continue
        palette = [int(c) for c in np.unique(source) if c != bg]
        if not 2 <= len(palette) <= 5:
            continue
        try:
            clocks = [_clock(row, bg, palette) for row in tracks]
            signatures = {(s['a'], s['b'], s['reference'], s['terminal'], s['destination']) for s in clocks}
            if len(signatures) != len(clocks):
                continue
            _orbit_separation(source, bg)
        except ValueError:
            continue
        candidates.append(dict(background=bg, marker=marker, source=source, clocks=clocks,
                               vertical=vertical, n=n, t=t, m=m, palette=palette,
                               source_box=square[0], table_box=table[0]))
    if len(candidates) != 1:
        raise ValueError("input must identify exactly one source square and clock bank")
    return candidates[0]


def _events(clock, t):
    a, b, d, terminal = clock['a'], clock['b'], clock['period'], clock['terminal']
    lo, hi = (0, t-1) if terminal is None else (min(a, terminal), max(b, terminal))
    return [j for j in range(lo, hi+1) if (j-a) % d == 0]


def _join(frames, marker, vertical):
    n = len(frames[0])
    result = np.full((n, n*len(frames)+len(frames)-1), marker, dtype=np.int8)
    for j, frame in enumerate(frames):
        result[:, j*(n+1):j*(n+1)+n] = frame
    if vertical:
        # Rearrange tiles, without transposing their contents.
        result = np.full((n*len(frames)+len(frames)-1, n), marker, dtype=np.int8)
        for j, frame in enumerate(frames):
            result[j*(n+1):j*(n+1)+n, :] = frame
    return result


def _coordinate_render(info, mode=None):
    """Input-only renderer, with explicit alternative rules for evidence checks."""
    n, t, bg, source = info['n'], info['t'], info['background'], info['source']
    frames = [np.full((n, n), bg, dtype=np.int8) for _ in range(t)]
    for clock in info['clocks']:
        events = _events(clock, t)
        if mode == 'raw_only':
            events = sorted({clock['a'], clock['b']} |
                            (set() if clock['terminal'] is None else {clock['terminal']}))
        elif mode == 'unbounded_terminal':
            events = [j for j in range(t) if (j-clock['a']) % clock['period'] == 0]
        elif mode == 'future_only' and clock['terminal'] is None:
            events = [j for j in events if j >= clock['a']]
        color = clock['reference'] if mode == 'no_routing' else clock['destination']
        shape_color = clock['reference'] if mode == 'recolor_reference' else color
        coords = np.argwhere(source == shape_color)
        for j in events:
            k = (j-clock['a']) // clock['period']
            if mode == 'stationary':
                k = 0
            elif mode == 'counterclockwise':
                k = -k
            elif mode == 'absolute_time':
                k = j
            for r, c in coords:
                y, x = _turn(int(r), int(c), n, k)
                if mode is None and frames[j][y, x] not in (bg, color):
                    raise ValueError("different-color collision")
                frames[j][y, x] = color
        if mode == 'noise_stamps':
            for j, noise_color in clock['noise']:
                frames[j][source == noise_color] = noise_color
    return _join(frames, info['marker'], info['vertical'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover clock pulses, signed ticks, routed motifs, and strip from input."""
    return _coordinate_render(_parse(input_grid))


def _make_source(rng, n, bg, colors, required_asymmetric):
    orbits = _orbits(n)
    noncentral = [i for i, orbit in enumerate(orbits) if len(orbit) == 4]
    central = next((i for i, orbit in enumerate(orbits) if len(orbit) == 1), None)
    eligible = [k for k in range(len(colors)) if k not in required_asymmetric]
    center_owner = None
    if central is not None and eligible and (len(colors) > len(noncentral) or rng.random() < .2):
        center_owner = int(_pick(rng, eligible))
    if rng.random() < .5:
        noncentral.sort(key=lambda i: min(min(r, c, n-1-r, n-1-c) for r, c in orbits[i]))
        # Radial allocation with a random color order, rather than a privileged border color.
    else:
        rng.shuffle(noncentral)
    owners = [[] for _ in colors]
    active = [k for k in range(len(colors)) if k != center_owner]
    order = [int(k) for k in rng.permutation(active)]
    for k, i in zip(order, noncentral):
        owners[k].append(i)
    weights = rng.dirichlet(np.ones(len(active)))
    for i in noncentral[len(active):]:
        owners[int(rng.choice(active, p=weights))].append(i)
    for i, orbit in enumerate(orbits):
        if len(orbit) == 1:
            owners[int(rng.integers(len(colors))) if center_owner is None else center_owner].append(i)
    source = np.full((n, n), bg, dtype=np.int8)
    for k, color in enumerate(colors):
        if k == center_owner:
            r, c = orbits[central][0]
            source[r, c] = color
            continue
        symmetry = 'asymmetric' if k in required_asymmetric else _pick(rng, ('asymmetric', 'half', 'quarter'))
        density = float(rng.uniform(.25, .9))
        phase = int(rng.integers(4))
        # A phase-biased motif produces connected hooks as well as fragments.
        chosen = set()
        for i in owners[k]:
            orbit = orbits[i]
            if len(orbit) == 1:
                if rng.random() < density:
                    chosen.update(orbit)
            elif symmetry == 'quarter':
                if rng.random() < density:
                    chosen.update(orbit)
            elif symmetry == 'half':
                if rng.random() < density:
                    offset = int(rng.integers(2))
                    chosen.update((orbit[offset], orbit[offset+2]))
            else:
                width = int(rng.integers(1, 4))
                offset = (phase + int(rng.integers(2))) % 4
                chosen.update(orbit[(offset+j) % 4] for j in range(width))
        # A private noncentral orbit certifies the requested symmetry.
        witness = next(i for i in owners[k] if len(orbits[i]) == 4)
        orbit = orbits[witness]
        chosen.difference_update(orbit)
        if symmetry == 'quarter':
            chosen.update(orbit)
        elif symmetry == 'half':
            chosen.update((orbit[phase], orbit[(phase+2) % 4]))
        else:
            width = int(rng.integers(1, 4))
            chosen.update(orbit[(phase+j) % 4] for j in range(width))
        for r, c in chosen:
            source[r, c] = color
    return source


def _spec(a, d, ref, terminal=None, destination=None, noise=()):
    return dict(a=a, b=a+d, period=d, reference=ref, terminal=terminal,
                destination=ref if terminal is None else destination, noise=list(noise))


def _random_clock(rng, t, colors, max_period=2, max_boundary_anchor=None):
    periods = [d for d in range(1, max_period+1) if (t-1)//d >= 2]
    d = int(_pick(rng, periods))
    terminal_mode = rng.random() < .45
    options = []
    for a in range(t-d):
        b = a+d
        if terminal_mode:
            for endpoint in range(t):
                if (endpoint < a or endpoint > b) and (endpoint-a) % d == 0:
                    options.append((a, endpoint))
        elif max_boundary_anchor is None or a <= max_boundary_anchor:
            if sum((j-a) % d == 0 for j in range(t)) >= 3:
                options.append((a, None))
    a, terminal = _pick(rng, options)
    ref = int(_pick(rng, colors))
    dest = ref if terminal is None else int(_pick(rng, [c for c in colors if c != ref]))
    noise = []
    positions = [j for j in range(t) if (j-a) % d]
    noise_colors = [c for c in colors if c not in (ref, dest)] if terminal is not None else [c for c in colors if c != ref]
    if positions and noise_colors and rng.random() < .45:
        count = int(rng.integers(1, min(2, len(positions), len(noise_colors))+1))
        for j, color in zip(rng.choice(positions, count, replace=False), rng.choice(noise_colors, count, replace=False)):
            noise.append((int(j), int(color)))
    return _spec(a, d, ref, terminal, dest, noise)


def _array_render(source, bg, clocks, t, marker, vertical):
    """Construction renderer: array rotations from sampled geometry, no parser."""
    n = len(source)
    frames = [np.full((n, n), bg, dtype=np.int8) for _ in range(t)]
    for clock in clocks:
        a, d, endpoint = clock['a'], clock['period'], clock['terminal']
        if endpoint is None:
            start, stop = a % d, t-1
        else:
            start, stop = min(a, endpoint), max(a+d, endpoint)
        color = clock['destination']
        for j in range(start, stop+1, d):
            mask = np.rot90(source == color, -((j-a)//d) % 4)
            assert np.all((frames[j][mask] == bg) | (frames[j][mask] == color))
            frames[j][mask] = color
    return _join(frames, marker, vertical)


def _framed(interior, marker):
    g = np.full((interior.shape[0]+2, interior.shape[1]+2), marker, dtype=np.int8)
    g[1:-1, 1:-1] = interior
    return g


def _layout(rng, first, second, bg):
    h1, w1 = first.shape
    h2, w2 = second.shape
    for attempt in range(40):
        h = 30 if attempt == 39 else int(rng.integers(max(h1, h2)+3, 31))
        w = 30 if attempt == 39 else int(rng.integers(max(w1, w2)+3, 31))
        r = int(rng.integers(1, h-h1))
        c = int(rng.integers(1, w-w1))
        positions = [(y, x) for y in range(1, h-h2) for x in range(1, w-w2)
                     if _separate((r, c, h1, w1), (y, x, h2, w2))]
        if positions:
            y, x = _pick(rng, positions)
            g = np.full((h, w), bg, dtype=np.int8)
            g[r:r+h1, c:c+w1] = first
            g[y:y+h2, x:x+w2] = second
            return g
    raise RuntimeError("bounded rectangle packing exhausted")


_GEOMETRIES = [(n, t) for n in range(3, 8) for t in range(3, 8) if n*t+t-1 <= 30]


def _build_pair(rng, settings, split, index, vertical):
    witness = index if split == 'train' and index < 2 else None
    mode = settings['curriculum']
    if witness == 0:
        n, t = _pick(rng, [(n, t) for n, t in _GEOMETRIES if t >= 5])
        m = 2
    elif witness == 1:
        n, t = _pick(rng, [(n, t) for n, t in _GEOMETRIES if t >= 6])
        m = 3
    elif split == 'test' and mode == 'rate':
        n, t, m = 3, 7, int(rng.integers(2, 5))
    elif split == 'test' and mode == 'concurrency':
        n, t = _pick(rng, [(n, t) for n, t in _GEOMETRIES if t >= 5])
        m = 4
    elif split == 'test':
        n, t = _pick(rng, [(n, t) for n, t in _GEOMETRIES if t >= 5])
        m = int(rng.integers(1, min(4, t-1)+1))
    else:
        n, t = _pick(rng, _GEOMETRIES)
        m = int(rng.integers(1, min(3, t-1)+1))
    orbit_count = len(_orbits(n))
    layer_count = int(rng.integers(2, min(5, orbit_count)+1))
    if settings['stable']:
        bg, marker = settings['background'], settings['marker']
        pool = settings['colors']
    else:
        palette = [int(c) for c in rng.permutation(10)]
        bg, marker, pool = palette[0], palette[1], palette[2:]
    colors = [int(c) for c in rng.choice(pool, layer_count, replace=False)]
    source = _make_source(rng, n, bg, colors, {0, 1} if witness is not None else set())
    if witness == 0:
        a = int(_pick(rng, [v for v in (0, 1) if sum((j-v) % 2 == 0 for j in range(t)) >= 3]))
        terminal = int(rng.integers(3, t-1))
        noise_options = [j for j in range(t) if (j-a) % 2
                         and (j < 1 or j > terminal or (j-1) % 4 != 0)]
        clocks = [_spec(a, 2, colors[0], noise=[(int(_pick(rng, noise_options)), colors[1])]),
                  _spec(1, 1, colors[0], terminal, colors[1])]
    elif witness == 1:
        clocks = [_spec(1, 1, colors[0]), _spec(2, 2, colors[0], 0, colors[1]),
                  _spec(2, 1, colors[1])]
    else:
        clocks = []
        if split == 'test' and mode == 'rate':
            a = int(_pick(rng, (0, 3)))
            clocks.append(_spec(a, 3, colors[0]))
        elif split == 'test' and mode == 'concurrency':
            clocks = [_spec(1, 1, colors[0]), _spec(2, 1, colors[1], 0, colors[0])]
        elif split == 'test':
            clocks.append(_spec(int(rng.integers(3, t-1)), 1, colors[0]))
        for _ in range(100):
            if len(clocks) == m:
                break
            s = _random_clock(rng, t, colors, max_period=2,
                              max_boundary_anchor=2 if split == 'train' else None)
            signature = (s['a'], s['b'], s['reference'], s['terminal'], s['destination'])
            if not any(signature == (q['a'], q['b'], q['reference'], q['terminal'], q['destination']) for q in clocks):
                clocks.append(s)
        if len(clocks) != m:
            raise RuntimeError("bounded clock construction exhausted")
    rng.shuffle(clocks)
    tracks = np.full((m, t), bg, dtype=np.int8)
    for row, s in zip(tracks, clocks):
        row[[s['a'], s['b']]] = s['reference']
        if s['terminal'] is not None:
            row[s['terminal']] = s['destination']
        for j, color in s['noise']:
            row[j] = color
    timetable = tracks.T if vertical else tracks
    inp = _layout(rng, _framed(source, marker), _framed(timetable, marker), bg)
    out = _array_render(source, bg, clocks, t, marker, vertical)
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    settings = dict(curriculum=_pick(rng, ('rate', 'concurrency', 'late_anchor')),
                    stable=bool(rng.random() < .25), background=palette[0],
                    marker=palette[1], colors=palette[2:7])
    initial_axis = bool(rng.integers(2))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            vertical = (initial_axis if index == 0 else not initial_axis) if split == 'train' and index < 2 else bool(rng.integers(2))
            for _ in range(20):
                pair = _build_pair(rng, settings, split, index, vertical)
                key = _key(pair['input'])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded input uniqueness construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _frame(output, info, j):
    n, start = info['n'], j*(info['n']+1)
    return output[start:start+n, :] if info['vertical'] else output[:, start:start+n]


def validate_generated_puzzle(puzzle):
    """Check the visible grammar, exact motion/union, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _parse(inp)
            expected = _coordinate_render(info)
            assert np.array_equal(out, expected), 'incorrect clocked motifs, union, or dividers'
            # Reconstruct individual supports and per-frame area from actual clocks.
            for j in range(info['t']):
                support = {}
                for s in info['clocks']:
                    if j not in _events(s, info['t']):
                        continue
                    color = s['destination']
                    k = (j-s['a'])//s['period']
                    stamp = {_turn(int(r), int(c), info['n'], k)
                             for r, c in np.argwhere(info['source'] == color)}
                    assert len(stamp) == int(np.count_nonzero(info['source'] == color))
                    support.setdefault(color, set()).update(stamp)
                frame = _frame(out, info, j)
                assert int(np.count_nonzero(frame != info['background'])) == sum(map(len, support.values()))
                assert set(int(c) for c in np.unique(frame) if c != info['background']) == set(support)
            if split == 'train':
                training.append((info, out))
    for shortcut in _SHORTCUTS:
        assert any(not np.array_equal(_coordinate_render(info, shortcut), out)
                   for info, out in training), 'training fails to distinguish '+shortcut
    assert {info['vertical'] for info, _ in training} == {False, True}, 'demonstrate both strip axes'
    assert any((j-s['a'])//s['period'] >= 4 and _symmetry(info['source'] == s['destination']) == 'asymmetric'
               for info, _ in training for s in info['clocks'] for j in _events(s, info['t'])), 'demonstrate a full cycle'
    assert any(sum(j in _events(s, info['t']) and s['destination'] == color for s in info['clocks']) >= 2
               for info, _ in training for color in info['palette'] for j in range(info['t'])), 'demonstrate concurrent same-color clocks'


def self_test():
    """Bounded tests on the 42 requested seeds and all legal explicit counts."""
    _semantic_fixtures()
    shapes, geometries, fingerprints, roles = set(), set(), set(), set()
    periods, terminal_sides, symmetries, track_counts, sizes, layouts = set(), set(), set(), set(), set(), set()
    blank_frames = same_color_union = asymmetric_clock = central_singletons = 0
    observed_shifts = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        fingerprints.add(_puzzle_key(puzzle))
        train, test = [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info = _parse(inp)
                (train if split == 'train' else test).append(info)
                copy = inp.copy()
                assert np.array_equal(solve(copy), out) and np.array_equal(copy, inp)
                shapes.add(inp.shape)
                geometries.add((info['n'], info['t'], info['m'], info['vertical']))
                roles.update((('background', info['background']), ('marker', info['marker'])))
                roles.update(('motif', c) for c in info['palette'])
                sizes.add(info['n'])
                track_counts.add(info['m'])
                sr, sc, sh, sw = info['source_box']
                tr, tc, th, tw = info['table_box']
                layouts.add(('above' if sr+sh < tr else 'below' if tr+th < sr else 'side',
                             'left' if sc+sw < tc else 'right' if tc+tw < sc else 'aligned'))
                symmetries.update(_symmetry(info['source'] == c) for c in info['palette'])
                central_singletons += sum(np.count_nonzero(info['source'] == c) == 1
                                          and info['n'] % 2 == 1
                                          and info['source'][info['n']//2, info['n']//2] == c
                                          for c in info['palette'])
                for s in info['clocks']:
                    periods.add(s['period'])
                    if s['terminal'] is not None:
                        terminal_sides.add('before' if s['terminal'] < s['a'] else 'after')
                    color = s['destination']
                    mask = info['source'] == color
                    for j in _events(s, info['t']):
                        k = (j-s['a'])//s['period']
                        assert {_turn(int(r), int(c), info['n'], k) for r,c in np.argwhere(mask)} == {
                            _turn(int(r), int(c), info['n'], k+4) for r,c in np.argwhere(mask)}
                    asymmetric_clock += _symmetry(mask) == 'asymmetric'
                for j in range(info['t']):
                    frame = _frame(out, info, j)
                    blank_frames += bool(np.all(frame == info['background']))
                    for color in info['palette']:
                        active = [s for s in info['clocks'] if s['destination'] == color and j in _events(s, info['t'])]
                        if len(active) >= 2:
                            area = int(np.count_nonzero(frame == color))
                            same_color_union += area > int(np.count_nonzero(info['source'] == color))
        if any(s['period'] == 3 for info in test for s in info['clocks']):
            assert all(s['period'] <= 2 for info in train for s in info['clocks'])
            observed_shifts.add('rate')
        if any(info['m'] == 4 for info in test):
            assert max(info['m'] for info in train) <= 3
            observed_shifts.add('concurrency')
        if any(s['terminal'] is None and s['a'] >= 3 for info in test for s in info['clocks']):
            observed_shifts.add('late_anchor')
    assert len(fingerprints) == 42
    assert len(shapes) >= 50 and len(geometries) >= 25
    assert sizes == set(range(3, 8)) and track_counts == {1, 2, 3, 4}
    assert periods == {1, 2, 3} and terminal_sides == {'before', 'after'}
    assert symmetries == {'asymmetric', 'half', 'quarter'}
    assert all({c for r, c in roles if r == role} == set(range(10)) for role in ('background', 'marker', 'motif'))
    assert observed_shifts == {'rate', 'concurrency', 'late_anchor'}
    assert blank_frames >= 5 and same_color_union >= 42 and asymmetric_clock >= 100
    assert central_singletons >= 5
    assert len(layouts) >= 5
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(917+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1),
                 (4,False), (2.0,1), (4,1.5), (None,1), (4,None), (-1,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(97, 2, 1)
    damaged = {s: [{k: v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    damaged['test'][0]['output'][0,0] = (int(damaged['test'][0]['output'][0,0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    malformed = p['train'][0]['input'].copy()
    info = _parse(malformed)
    r, c, _, _ = info['source_box']
    malformed[r, c] = info['background']
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError('broken outline accepted')
    return {'seeds_checked': 42, 'distinct_puzzles': len(fingerprints),
            'input_shapes': len(shapes), 'structural_geometries': len(geometries),
            'source_sizes': sorted(sizes), 'periods': sorted(periods),
            'same_color_union_frames': same_color_union, 'blank_frames': blank_frames,
            'central_singleton_layers': int(central_singletons),
            'train_test_shifts': sorted(observed_shifts)}


def _semantic_fixtures():
    """Hand-authored answers cover routing and union outside the curriculum."""
    def digits(rows):
        return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)

    source = digits(('100', '200', '000'))
    # All-terminal, single-track case. The terminal selects the edge motif 2;
    # ticks -1,0,1 put it bottom-center, left-center, then top-center.
    inp = np.zeros((11, 12), dtype=np.int8)
    inp[1:6, 1:6] = _framed(source, 9)
    inp[7:10, 1:10] = _framed(digits(('2010100',)), 9)
    expected = digits(tuple('9'.join(row) for row in (
        ('000', '000', '000', '000', '020', '000', '000'),
        ('000', '000', '200', '000', '000', '000', '000'),
        ('020', '000', '000', '000', '000', '000', '000'),
    )))
    assert np.array_equal(solve(inp), expected)
    # The vertical layout must preserve each motif's orientation.
    vertical = np.zeros((12, 13), dtype=np.int8)
    vertical[1:6, 1:6] = _framed(source, 9)
    vertical[1:10, 8:11] = _framed(digits(('2', '0', '1', '0', '1', '0', '0')), 9)
    vertical_answer = digits((
        '000', '000', '020', '999', '000', '000', '000', '999',
        '000', '200', '000', '999', '000', '000', '000', '999',
        '020', '000', '000', '999', '000', '000', '000', '999', '000', '000', '000',
    ))
    assert np.array_equal(solve(vertical), vertical_answer)
    # Two clocks select the same corner motif at adjacent angular phases.
    # Motif 2 is unclocked. At each slot both corner copies must survive.
    inp = np.zeros((8, 16), dtype=np.int8)
    inp[1:6, 1:6] = _framed(source, 9)
    inp[1:5, 8:15] = _framed(digits(('11000', '01100')), 9)
    union_answer = digits(tuple('9'.join(row) for row in (
        ('100', '101', '001', '000', '100'),
        ('000', '000', '000', '000', '000'),
        ('100', '000', '001', '101', '100'),
    )))
    assert np.array_equal(solve(inp), union_answer)
