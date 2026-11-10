"""Mirrored routers: rays selectively reveal missing, handed control pads.

Only Python and NumPy are used. All decoding conventions are visible or fixed:
0=empty, 5=unknown pad, 9=key frame; all other colors are interchangeable data.
"""
from itertools import product
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_DATA = (1, 2, 3, 4, 6, 7, 8)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _orbit(shape, r, c):
    h, w = shape
    return tuple(sorted({(r, c), (h - 1 - r, c),
                         (r, w - 1 - c), (h - 1 - r, w - 1 - c)}))


def _parity(shape, r, c, axes=(True, True)):
    h, w = shape
    # Pads never lie on an odd-sized field's central row or central column.
    return (bool(axes[0]) and 2 * r > h - 1) ^ (bool(axes[1]) and 2 * c > w - 1)


def _decode(input_grid):
    a = _grid(input_grid)
    _require(np.all(a[0] == 9), "missing top key frame")
    bottoms = np.flatnonzero(np.all(a[1:] == 9, axis=1)) + 1
    _require(len(bottoms) > 0, "missing bottom key frame")
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(a) - 1, "invalid key height or empty field")
    band = a[1:bottom]
    _require(np.all(band == band[0]), "key bars must fill their panel height")
    row = band[0]
    _require(row[0] == row[-1] == 9, "missing key side walls")
    keys, widths = {}, []
    c = 0
    while c < a.shape[1]:
        if row[c] == 9:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != 9:
            c += 1
        values = row[start:c]
        _require(len(values) >= 3 and np.all(values[1:-1] == 0),
                 "a key panel needs a clear interior")
        left, right = int(values[0]), int(values[-1])
        _require((left == 0) != (right == 0), "ambiguous side bar")
        color, side = (left, -1) if right == 0 else (right, 1)
        _require(color in _DATA and color not in keys, "invalid or duplicate key")
        keys[color] = side
        widths.append(len(values) + 2)
    _require(2 <= len(keys) <= 5, "need two to five keyed colors")
    field = a[bottom + 1:].copy()
    others = set(map(int, np.unique(field))) - {0, 5} - set(keys)
    _require(len(others) == 1, "unique unkeyed source color required")
    source = others.pop()
    _require(source in _DATA, "invalid source color")
    cells = np.argwhere(field == source)
    _require(len(cells) == 1, "source must be a single cell")
    position = tuple(map(int, cells[0]))
    h, w = field.shape
    pads = np.argwhere(np.isin(field, list(keys) + [5]))
    _require(len(pads) > 0, "no control pads")
    for r, c in pads:
        _require(2 * r != h - 1 and 2 * c != w - 1,
                 "control pads cannot be on a fold")
    return dict(field=field, keys=keys, source=source, position=position,
                crop=bottom + 1, widths=tuple(widths))


def _complete(info):
    """Complete the latent pad field; the nonsymmetric source is an overlay."""
    field, keys = info['field'], info['keys']
    clean = field.copy()
    clean[info['position']] = 0
    seen = set()
    for r in range(field.shape[0]):
        for c in range(field.shape[1]):
            if (r, c) in seen:
                continue
            orbit = _orbit(field.shape, r, c)
            seen.update(orbit)
            witnesses = {int(clean[p]) for p in orbit if clean[p] != 5}
            _require(len(witnesses) == 1, "missing or conflicting orbit witness")
            value = next(iter(witnesses))
            _require(value == 0 or value in keys, "unkeyed orbit value")
            _require(value != 0 or all(clean[p] != 5 for p in orbit),
                     "an unknown marker must denote a pad, never empty space")
            for p in orbit:
                clean[p] = value
    _require(clean[info['position']] == 0, "source overlay must cover empty space")
    return clean


def _walk(clean, position, keys, axes=(True, True), stop_cycles=False):
    """Cell-step tracer; no output paint is consulted while routing."""
    h, w = clean.shape
    visited = np.zeros(clean.shape, dtype=bool)
    events, chains = [], []
    for branch in range(4):
        r, c = position
        d = branch
        states, chain = set(), []
        while True:
            state = (r, c, d)
            if stop_cycles and state in states:
                # Counterfactuals may loop: finish their finite reachability
                # mask so ablations still compare actual empty-cell tracks.
                break
            _require(state not in states, "cyclic ray is outside this family")
            states.add(state)
            dr, dc = _DIRS[d]
            r, c = r + dr, c + dc
            if not (0 <= r < h and 0 <= c < w):
                break
            visited[r, c] = True
            color = int(clean[r, c])
            if color in keys:
                flip = _parity(clean.shape, r, c, axes)
                step = keys[color] * (-1 if flip else 1)
                events.append((branch, r, c, d, color, int(flip)))
                chain.append((r, c))
                d = (d + step) % 4
        chains.append(tuple(chain))
    return visited, tuple(events), tuple(chains)


def _jump(clean, position, keys):
    """Geometric check: jump between nearest pads instead of stepping cells."""
    h, w = clean.shape
    pads = [(int(r), int(c), int(clean[r, c])) for r, c in np.argwhere(clean != 0)]
    visited = np.zeros(clean.shape, dtype=bool)
    for branch in range(4):
        r, c = position
        d, seen = branch, set()
        while True:
            _require((r, c, d) not in seen, "cyclic geometric route")
            seen.add((r, c, d))
            if d == 0:
                hits = [(r - y, y, x, color) for y, x, color in pads if x == c and y < r]
                distance = r
            elif d == 1:
                hits = [(x - c, y, x, color) for y, x, color in pads if y == r and x > c]
                distance = w - c - 1
            elif d == 2:
                hits = [(y - r, y, x, color) for y, x, color in pads if x == c and y > r]
                distance = h - r - 1
            else:
                hits = [(c - x, y, x, color) for y, x, color in pads if y == r and x < c]
                distance = c
            nearest = min(hits) if hits else None
            if nearest:
                distance = nearest[0]
            dr, dc = _DIRS[d]
            for t in range(1, distance + 1):
                visited[r + t * dr, c + t * dc] = True
            if nearest is None:
                break
            _, r, c, color = nearest
            # Independently spell out the mirror-orientation calculation.
            north = r < h / 2
            west = c < w / 2
            turn = keys[color] if north == west else -keys[color]
            d = (d + turn) % 4
    return visited


def _render(info, clean, visited, reveal_all=False):
    out = info['field'].copy()
    out[(out == 0) & visited] = info['source']
    reveal = (out == 5) & (np.ones(out.shape, dtype=bool) if reveal_all else visited)
    out[reveal] = clean[reveal]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, hidden pad and trajectory entirely from the input."""
    info = _decode(input_grid)
    clean = _complete(info)
    visited, _, _ = _walk(clean, info['position'], info['keys'])
    return _render(info, clean, visited)


def _header(rng, width, keys):
    sizes = np.full(len(keys), 5, dtype=int)
    sizes += rng.multinomial(width - int(sizes.sum()), np.full(len(keys), 1 / len(keys)))
    header = np.zeros((int(rng.integers(3, 5)), width), dtype=np.int8)
    header[[0, -1]] = 9
    col = 0
    for color, size in zip(rng.permutation(list(keys)), sizes):
        header[:, col] = header[:, col + size - 1] = 9
        header[1:-1, col + 1 if keys[int(color)] == -1 else col + size - 2] = color
        col += int(size)
    return header


def _scene(rng, keys, test, profile):
    """Construct a new symmetric controller, not a parent puzzle or template."""
    h = int(rng.integers(20, 27) if test else rng.integers(10, 19))
    w = int(rng.integers(max(5 * len(keys), 16 if test else 12), 31))
    clean = np.zeros((h, w), dtype=np.int8)
    reps = [(r, c) for r in range(h // 2) for c in range(w // 2)]
    # A puzzle-level bias combines freely with pair-level scatter/lattice choices.
    geometry = int(rng.integers(3))
    if geometry == 0:
        rng.shuffle(reps)
    else:
        scores = rng.random(len(reps))
        for i, (r, c) in enumerate(reps):
            if geometry == 1:
                scores[i] += profile * min(r, c) / max(h, w)
            else:
                scores[i] += .7 * ((r % 3 == 0) + (c % 3 == 0))
        reps = [reps[i] for i in np.argsort(-scores)]
    target = int(rng.integers(6, min(22, max(7, h * w // 18)) + 1))
    orbits = []
    separate_diagonals = bool(rng.integers(2))
    for r, c in reps:
        orbit = _orbit(clean.shape, r, c)
        if any(any(abs(y - yy) + abs(x - xx) <= 1
                   for yy, xx in orbit if (y, x) != (yy, xx)) for y, x in orbit):
            continue
        blocked = False
        for y, x in orbit:
            if separate_diagonals:
                blocked |= bool(clean[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any())
            else:
                blocked |= any(0 <= y + dy < h and 0 <= x + dx < w and clean[y + dy, x + dx]
                               for dy, dx in _DIRS)
        if blocked:
            continue
        color = int(rng.choice(list(keys)))
        for p in orbit:
            clean[p] = color
        orbits.append(orbit)
        if len(orbits) == target:
            break
    if len(orbits) < 5:
        return None
    available = np.argwhere(clean == 0)
    position = tuple(map(int, available[int(rng.integers(len(available)))]))
    try:
        visited, events, chains = _walk(clean, position, keys)
    except AssertionError:
        return None
    if len({e[1:3] for e in events}) < 3:
        return None
    return clean, orbits, position, visited, events, chains


def _mask(rng, clean, orbits, visited, teacher, test):
    hidden = np.zeros(clean.shape, dtype=bool)
    used = [p for p in zip(*np.nonzero(visited & (clean != 0)))]
    if not used:
        return None
    # A diagonal witness demonstrates why both folds are necessary.
    focal = used[int(rng.integers(len(used)))]
    orb = _orbit(clean.shape, *focal)
    if teacher or test:
        diagonal = (clean.shape[0] - 1 - focal[0], clean.shape[1] - 1 - focal[1])
        for p in orb:
            hidden[p] = p != diagonal
    else:
        hidden[focal] = True
    unused = [p for orbit in orbits for p in orbit if not visited[p] and not hidden[p]]
    rng.shuffle(unused)
    for p in unused:
        orbit = _orbit(clean.shape, *p)
        if sum(hidden[q] for q in orbit) < 3:
            hidden[p] = True
            break
    else:
        return None
    fraction = float(rng.uniform(.52, .69) if test else rng.uniform(.22, .38))
    budget = int(round(fraction * 4 * len(orbits)))
    points = [p for orbit in orbits for p in orbit]
    rng.shuffle(points)
    for p in points:
        if int(hidden.sum()) >= budget:
            break
        if not hidden[p] and sum(hidden[q] for q in _orbit(clean.shape, *p)) < 3:
            hidden[p] = True
    return hidden


def _features(info, clean, visited, events, chains):
    hidden = info['field'] == 5
    headings = {side: {e[3] for e in events if info['keys'][e[4]] == side}
                for side in (-1, 1)}
    diagonal = 0
    for r, c in np.argwhere(hidden & visited):
        rr, cc = clean.shape[0] - 1 - r, clean.shape[1] - 1 - c
        diagonal += bool(hidden[rr, c] and hidden[r, cc] and not hidden[rr, cc])
    return dict(shape=clean.shape, source=info['source'], position=info['position'],
                keys=len(info['keys']), orbits=int(np.count_nonzero(clean) // 4),
                hidden=int(hidden.sum()), hidden_fraction=float(hidden.sum() / np.count_nonzero(clean)),
                revealed=int((hidden & visited).sum()), unrevealed=int((hidden & ~visited).sum()),
                diagonal=diagonal, max_turns=max(map(len, chains)),
                hits=len({e[1:3] for e in events}),
                parity={e[5] for e in events}, headings=headings,
                quadrants={(int(2 * e[1] > clean.shape[0] - 1),
                            int(2 * e[2] > clean.shape[1] - 1)) for e in events})


def _ablations(info, clean, visited):
    expected = _render(info, clean, visited)
    # Missing controls act as empty space, without symmetry decoding.
    absent = clean.copy()
    absent[info['field'] == 5] = 0
    without_a = _walk(absent, info['position'], info['keys'], stop_cycles=True)[0]
    a_changes_paths = np.any((visited ^ without_a) & (clean == 0))
    # No routing instructions: all four rays keep their launch heading.
    straight = _walk(np.zeros(clean.shape, dtype=np.int8), info['position'], {})[0]
    b_changes_paths = np.any((visited ^ straight) & (clean == 0))
    try:
        unflipped = _walk(clean, info['position'], info['keys'], (False, False))[0]
        chirality_changes = not np.array_equal(_render(info, clean, unflipped), expected)
    except AssertionError:
        chirality_changes = True
    return bool(a_changes_paths), bool(b_changes_paths), bool(chirality_changes)


def _make_pair(rng, keys, teacher, test, profile, teaching_context=None):
    for _ in range(240):
        scene = _scene(rng, keys, test, profile)
        if scene is None:
            continue
        clean, orbits, position, visited, events, chains = scene
        if teacher:
            headings = [{e[3] for e in events if keys[e[4]] == side} for side in (-1, 1)]
            if min(map(len, headings)) < 2 or {e[5] for e in events} != {0, 1}:
                continue
        source = int(rng.choice([c for c in _DATA if c not in keys]))
        hidden = _mask(rng, clean, orbits, visited, teacher, test)
        if hidden is None:
            continue
        field = clean.copy()
        field[hidden] = 5
        field[position] = source
        info = dict(field=field, keys=keys, source=source, position=position)
        if not all(_ablations(info, clean, visited)):
            continue
        header = _header(rng, field.shape[1], keys)
        inp = np.vstack((header, field))
        # Construction uses the geometric tracer; solve uses the cell-step tracer.
        geometric = _jump(clean, position, keys)
        _require(np.array_equal(geometric, visited), "construction tracers disagree")
        out = _render(info, clean, geometric)
        f = _features(info, clean, visited, events, chains)
        if f['revealed'] == 0 or f['unrevealed'] == 0:
            continue
        pair = dict(input=inp, output=out)
        if teaching_context is not None:
            lessons = teaching_context + [pair]
            quadrants, observed = set(), []
            for lesson in lessons:
                q = _decode(lesson['input'])
                a = _complete(q)
                v, e, ch = _walk(a, q['position'], q['keys'])
                quadrants.update(_features(q, a, v, e, ch)['quadrants'])
                observed.append({event[4]: q['keys'][event[4]] for event in e})
            color_flip = any(len({t[c] for t in observed if c in t}) == 2 for c in _DATA)
            if (len(quadrants) != 4 or not color_flip
                    or _hypotheses(lessons) != [(-1, 1, True, True, False)]):
                continue
        return pair
    raise RuntimeError("bounded scene construction exhausted")


def _hypotheses(train):
    """Finite teaching audit, not an independent ARC solver.

    Compete four glyph meanings, four reflection-parity conventions, and two
    reveal policies. Exactly one must fit the returned demonstrations.
    """
    decoded = []
    for pair in train:
        q = _decode(pair['input'])
        decoded.append((q, _complete(q), np.asarray(pair['output'])))
    fits = []
    for left, right, row_flip, col_flip, all_hidden in product((-1, 1), repeat=5):
        axes = row_flip == 1, col_flip == 1
        reveal_all = all_hidden == 1
        for q, clean, expected in decoded:
            keys = {c: left if side == -1 else right for c, side in q['keys'].items()}
            try:
                visited = _walk(clean, q['position'], keys, axes)[0]
                prediction = _render(q, clean, visited, reveal_all)
            except AssertionError:
                break
            if not np.array_equal(prediction, expected):
                break
        else:
            fits.append((left, right, *axes, reveal_all))
    return fits


def _check_count(name, value, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not lower <= int(value) <= upper:
        raise ValueError(name + " is outside its allowed range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _check_count('num_train', num_train, 2, 10)
    nq = _check_count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    profile = float(rng.uniform(-2, 2))
    anchor = list(map(int, rng.choice(_DATA, size=int(rng.integers(2, 5)), replace=False)))
    first = {c: int(rng.choice((-1, 1))) for c in anchor}
    first[anchor[0]], first[anchor[1]] = -1, 1
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            teacher = split == 'train' and i < 2
            if teacher:
                keys = first if i == 0 else {c: -side for c, side in first.items()}
            else:
                colors = list(map(int, rng.choice(_DATA, size=int(rng.integers(2, 6)), replace=False)))
                keys = {c: int(rng.choice((-1, 1))) for c in colors}
            for _ in range(16):
                pair = _make_pair(rng, keys, teacher, split == 'test', profile,
                                  puzzle['train'] if teacher and i == 1 else None)
                sig = _signature(pair['input'])
                if sig not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(sig)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, features, infos = set(), {'train': [], 'test': []}, {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_signature(inp) not in seen, "duplicate input")
            seen.add(_signature(inp))
            info = _decode(inp)
            clean = _complete(info)
            visited, events, chains = _walk(clean, info['position'], info['keys'])
            _require(np.array_equal(visited, _jump(clean, info['position'], info['keys'])),
                     "cell/geometric route disagreement")
            _require(out.shape == clean.shape, "output must omit the key")
            _require(np.array_equal(out, _render(info, clean, visited)), "incorrect selective reveal or ray paint")
            field = info['field']
            visible = (field != 0) & (field != 5)
            _require(np.array_equal(out[visible], field[visible]), "visible pads or source changed")
            _require(np.array_equal(out == 5, (field == 5) & ~visited), "incorrect remaining gaps")
            _require(all(_ablations(info, clean, visited)), "a parent mechanism or chirality has no effect")
            f = _features(info, clean, visited, events, chains)
            _require(f['revealed'] > 0 and f['unrevealed'] > 0, "selective reveal must be observable")
            features[split].append(f)
            infos[split].append(info)
    train = features['train']
    _require(any(f['diagonal'] for f in train), "both-fold reconstruction is undemonstrated")
    for side in (-1, 1):
        headings = set().union(*(f['headings'][side] for f in train))
        _require(len(headings) >= 2, "relative turns need multiple arrival headings")
    quadrants = set().union(*(f['quadrants'] for f in train))
    _require(len(quadrants) == 4, "training must expose both folds' handedness")
    _require(_hypotheses(puzzle['train']) == [(-1, 1, True, True, False)],
             "demonstrations do not uniquely expose the intended conventions")
    tables = []
    for q in infos['train']:
        events = _walk(_complete(q), q['position'], q['keys'])[1]
        tables.append({e[4]: q['keys'][e[4]] for e in events})
    _require(any(len({t[c] for t in tables if c in t}) == 2 for c in _DATA),
             "an encountered color must change its keyed meaning in training")
    _require(min(f['shape'][0] for f in features['test']) > max(f['shape'][0] for f in train),
             "test must extend the demonstrated field height")
    _require(min(f['hidden_fraction'] for f in features['test']) > max(f['hidden_fraction'] for f in train),
             "test must increase occlusion")
    return True


def self_test():
    """Bounded 42-seed audit plus count, determinism and corruption checks."""
    # Hand-derived small fixture outside sampling ranges. Adjacent pads straddle
    # the vertical fold; a triple-hidden orbit is decoded from its diagonal.
    field = np.zeros((6, 10), dtype=np.int8)
    field[1, 3:7] = field[4, 3:7] = (1, 2, 2, 1)
    field[1, 4] = field[1, 5] = field[4, 4] = 5
    field[2, 4] = 8
    header = np.array([[9] * 10, [9, 1, 0, 0, 9, 9, 0, 0, 2, 9], [9] * 10], dtype=np.int8)
    fixture = np.vstack((header, field))
    expected = field.copy()
    expected[expected == 5] = 2
    expected[2] = 8
    expected[0, 5] = expected[3, 4] = expected[5, 5] = 8
    _require(np.array_equal(solve(fixture), expected), "hand-derived adjacent-pad fixture failed")
    for contradictory in (False, True):
        bad = fixture.copy()
        bad[3 + 4, 5] = 1 if contradictory else 5
        if contradictory:
            bad[3 + 1, 4] = 2
        try:
            solve(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("invalid reflection orbit accepted")
    signatures, shapes, layouts, normalized = set(), set(), set(), set()
    sources, keys, quadrants, heights, orbit_counts, turns = set(), set(), set(), set(), set(), set()
    diagonal_train = diagonal_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                q = _decode(pair['input'])
                clean = _complete(q)
                v, e, chains = _walk(clean, q['position'], q['keys'])
                f = _features(q, clean, v, e, chains)
                shapes.add(clean.shape)
                layouts.add((clean.shape, q['position'], tuple(map(tuple, np.argwhere(clean != 0)))))
                roles = {}
                normalized.add((pair['input'].shape, tuple(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)))
                sources.add(q['source']); keys.add(f['keys']); orbit_counts.add(f['orbits'])
                quadrants.update(f['quadrants']); heights.add(q['crop']); turns.add(f['max_turns'])
                if split == 'train':
                    diagonal_train += f['diagonal'] > 0
                else:
                    diagonal_test += f['diagonal'] > 0
                _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), "input-only/list solve failed")
    _require(len(signatures) == 42 and len(layouts) >= 190 and len(normalized) >= 190,
             "structural diversity collapsed")
    _require(len(shapes) >= 60 and len(sources) == 7 and keys == {2, 3, 4, 5}, "geometry/palette diversity collapsed")
    _require(len(orbit_counts) >= 10 and len(turns) >= 8 and heights == {3, 4}, "count/path/header variation collapsed")
    _require(len(quadrants) == 4 and diagonal_train >= 84 and diagonal_test == 42,
             "two-fold teaching coverage collapsed")
    p = generate_puzzle(0)
    p2 = generate_puzzle(0)
    _require(all(np.array_equal(a[n], b[n]) for s in p for a, b in zip(p[s], p2[s]) for n in a),
             "nondeterministic generation")
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 3)):
        q = generate_puzzle(120 + nt + nq, nt, nq)
        _require(len(q['train']) == nt and len(q['test']) == nq, "explicit counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Negative checks distinguish a wrong repair, repainting a visible pad, and
    # prematurely filling an unvisited pad; no huge randomized test loop.
    for kind in ('visited', 'visible', 'unvisited'):
        bad = {s: [{n: a.copy() for n, a in pair.items()} for pair in p[s]] for s in p}
        pair = bad['train'][0]
        info = _decode(pair['input']); clean = _complete(info)
        v = _walk(clean, info['position'], info['keys'])[0]
        field = info['field']
        mask = ((field == 5) & v if kind == 'visited' else
                (np.isin(field, list(info['keys'])) if kind == 'visible' else (field == 5) & ~v))
        point = tuple(np.argwhere(mask)[0])
        pair['output'][point] = (int(pair['output'][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted output corruption")
    return dict(seeds=42, pairs=210, shapes=len(shapes), layouts=len(layouts),
                normalized_inputs=len(normalized), source_colors=len(sources),
                key_counts=sorted(keys), orbit_counts=sorted(orbit_counts),
                route_lengths=sorted(turns), diagonal_train=diagonal_train, diagonal_test=diagonal_test)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
