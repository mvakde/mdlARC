"""Mixed rigid/flexible exact-cover ARC family. Standard library + NumPy only."""
from math import isqrt
import numpy as np

EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(cells, neighbors=NEIGHBORS):
    left = set(cells)
    result = []
    while left:
        first = min(left)
        left.remove(first)
        reached, todo = {first}, [first]
        while todo:
            r, c = todo.pop()
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    reached.add(p)
                    todo.append(p)
        result.append(reached)
    return result


def _turn(cells, k):
    out = set(cells)
    for _ in range(k):
        out = {(c, -r) for r, c in out}
    return out


def _normal(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _orientations(cells):
    return sorted({_normal(_turn(cells, k)) for k in range(4)}, key=lambda s: tuple(sorted(s)))


def _route(core, length):
    """Taut contour in the fixed marked-core coordinate system.

    These three supporting lines enclose every core cell with one cell of
    right clearance. Their lower envelope bridges concave bays.
    """
    _require((0, 0) in core and all(0 <= c <= r for r, c in core), "core outside starting cone")
    peak = max(c for r, c in core)
    diagonal = max(r + c for r, c in core)
    return [(r, min(r + 1, peak + 1, diagonal + 1 - r)) for r in range(length)]


def _read_input(inp):
    inp = _grid(inp)
    colors, counts = np.unique(inp, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background must be unique mode")
    background = int(colors[counts.argmax()])
    singleton = [int(c) for c, n in zip(colors, counts) if n == 1 and c != background]
    _require(len(singleton) == 1, "one singleton marker required")
    marker = singleton[0]
    marker_at = tuple(map(int, np.argwhere(inp == marker)[0]))
    foreground = {tuple(map(int, p)) for p in np.argwhere(inp != background)}
    side = isqrt(len(foreground))
    _require(side * side == len(foreground), "all material must have square area")
    components = _components(foreground)
    compound = next(c for c in components if marker_at in c)
    role_colors = {int(inp[p]) for p in compound} - {marker}
    _require(len(role_colors) == 2, "marked component must contain core and thread")
    candidates = []
    for carrier in sorted(role_colors):
        thread_color = next(iter(role_colors - {carrier}))
        raw_core = {(r - marker_at[0], c - marker_at[1]) for r, c in compound
                    if int(inp[r, c]) in (carrier, marker)}
        raw_thread = {(r - marker_at[0], c - marker_at[1]) for r, c in compound
                      if int(inp[r, c]) == thread_color}
        for k in range(4):
            core, slack = _turn(raw_core, k), _turn(raw_thread, k)
            if not core or not slack or min(len(core), len(slack)) < 4:
                continue
            if not all(0 <= c <= r for r, c in core):
                continue
            if max(c for r, c in core) < 1 or {p for p in core if p[0] == 0} != {(0, 0)}:
                continue
            if not any(r == c and c > 0 for r, c in core):
                continue
            if not all((r, 0) in core for r in range(max(r for r, c in core) + 1)):
                continue
            if (0, 1) not in slack or not all((r, c) == (0, 1) or (r < 0 and c >= 2) for r, c in slack):
                continue
            if len(_components(core)) != 1 or len(_components(slack)) != 1:
                continue
            route = _route(core, len(slack))
            if not all(0 <= r < side and 0 <= c < side for r, c in core | set(route)):
                continue
            if core & set(route):
                continue
            candidates.append((carrier, thread_color, core, slack, route, k))
    _require(len(candidates) == 1, "ambiguous core/thread role or orientation")
    carrier, thread_color, core, slack, route, k = candidates[0]
    pieces = []
    used_colors = {background, marker, carrier, thread_color}
    for component in components:
        if component is compound:
            continue
        values = {int(inp[p]) for p in component}
        _require(len(values) == 1 and len(component) >= 3, "rigid fillers must be monochrome objects")
        color = next(iter(values))
        _require(color not in used_colors, "distinct material colors required")
        used_colors.add(color)
        pieces.append((color, _normal(component)))
    _require(2 <= len(pieces) <= 4, "three to five rigid pieces required, including core")
    return dict(input=inp, background=background, marker=marker, carrier=carrier,
                thread_color=thread_color, core=core, slack=slack, route=route,
                side=side, pieces=pieces, marker_at=marker_at, anchor_turn=k)


def _pack(facts, rotation=True, limit=2):
    """Enumerate all colored exact covers after the fixed thread is reserved."""
    n = facts['side']
    fixed = facts['core'] | set(facts['route'])
    blocked = sum(1 << (r * n + c) for r, c in fixed)
    full = (1 << (n * n)) - 1
    candidates = []
    for color, piece in facts['pieces']:
        options = {}
        orientations = _orientations(piece) if rotation else [piece]
        for shape in orientations:
            h, w = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
            base = sum(1 << (r * n + c) for r, c in shape)
            for r in range(n - h + 1):
                for c in range(n - w + 1):
                    mask = base << (r * n + c)
                    if not mask & blocked:
                        options.setdefault(mask, (mask, shape, r, c))
        if not options:
            return []
        candidates.append(list(options.values()))
    answers, seen, placed = [], set(), [None] * len(candidates)

    def search(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            out = np.full((n, n), facts['carrier'], dtype=np.int8)
            out[0, 0] = facts['marker']
            for p in facts['route']:
                out[p] = facts['thread_color']
            for i, (_, shape, r, c) in enumerate(placed):
                for dr, dc in shape:
                    out[r + dr, c + dc] = facts['pieces'][i][0]
            key = out.tobytes()
            if key not in seen:
                seen.add(key)
                answers.append(out)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        choices = []
        for i in remaining:
            fitting = [p for p in candidates[i] if p[0] & first and not p[0] & occupied]
            if fitting:
                choices.append((i, fitting))
        choices.sort(key=lambda q: len(q[1]))
        for i, fitting in choices:
            rest = tuple(j for j in remaining if j != i)
            for placement in fitting:
                # Every other piece needs at least one remaining placement.
                occupied2 = occupied | placement[0]
                if any(not any(not p[0] & occupied2 for p in candidates[j]) for j in rest):
                    continue
                placed[i] = placement
                search(occupied2, rest)
                if len(answers) >= limit:
                    return
            placed[i] = None

    search(blocked, tuple(range(len(candidates))))
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _read_input(input_grid)
    answers = _pack(facts)
    _require(len(answers) == 1, "input must determine one colored assembly")
    return answers[0]


def _grow_fillers(rng, free, count):
    components = _components(free)
    if len(components) > count or min(map(len, components)) < 3 or len(free) < 3 * count:
        return None
    seeds = [sorted(c)[int(rng.integers(len(c)))] for c in components]
    rest = sorted(free - set(seeds))
    seeds += [rest[int(i)] for i in rng.choice(len(rest), count - len(seeds), replace=False)]
    pieces = [{p} for p in seeds]
    unfilled = free - set(seeds)
    weights = np.exp(rng.uniform(-1, 1, count))
    frontiers = [set() for _ in seeds]
    for i, piece in enumerate(pieces):
        frontiers[i] = {(r + dr, c + dc) for r, c in piece for dr, dc in NEIGHBORS} & unfilled
    while unfilled:
        eligible = [i for i in range(count) if frontiers[i]]
        if not eligible:
            return None
        i = min(eligible, key=lambda j: len(pieces[j]) / weights[j] + float(rng.random()) * .5)
        choices = sorted(frontiers[i])
        edge_choices = [p for p in choices if any((p[0] + dr, p[1] + dc) in pieces[i] for dr, dc in EDGE)]
        if edge_choices and rng.random() < .9:
            choices = edge_choices
        p = choices[int(rng.integers(len(choices)))]
        pieces[i].add(p)
        unfilled.remove(p)
        for frontier in frontiers:
            frontier.discard(p)
        frontiers[i].update({(p[0] + dr, p[1] + dc) for dr, dc in NEIGHBORS} & unfilled)
    if min(map(len, pieces)) < 3:
        return None
    return pieces


def _core(rng, side, recess):
    p = int(rng.integers(3 if recess else 2, (side - 2) // 2 + 1))
    plateau = int(rng.integers(1, side - 2 * p))
    e = 2 * p + plateau - 1
    trim = int(rng.integers(p + 1))
    cells = {(r, c) for r in range(e - trim + 1) for c in range(min(r, p, e - r) + 1)}
    if recess:
        # A triangular bay is bridged by the taut thread and becomes a rigid insert.
        depth = int(rng.integers(2, p))
        cells -= {(r, c) for r in range(1, depth + 1) for c in range(1, r + 1)}
    elif p >= 4 and rng.random() < .35:
        # A small enclosed cavity adds another compatible packing topology.
        cells -= {(r, c) for r in (3, 4) for c in (1, 2)}
    _require(len(_components(cells)) == 1, "core must remain rigidly connected")
    return cells, e


def _slack(rng, length):
    cells = {(0, 1), (-1, 2)}
    style = int(rng.integers(3))
    width = int(rng.integers(3, 7))
    while len(cells) < length:
        options = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in NEIGHBORS
                          if -6 <= r + dr < 0 and 2 <= c + dc <= width} - cells)
        weights = []
        for r, c in options:
            neighbors = sum((r + dr, c + dc) in cells for dr, dc in NEIGHBORS)
            weights.append((neighbors + 1) ** 2 if style == 0 else 1 / (neighbors + 1) if style == 1 else 1)
        probabilities = np.array(weights, dtype=float)
        cells.add(options[int(rng.choice(len(options), p=probabilities / probabilities.sum()))])
    return cells


def _scatter(rng, core, slack, fillers, colors, marker, bg, anchor_rotation):
    objects = []
    compound = {p: colors[0] for p in core}
    compound[0, 0] = marker
    compound.update({p: colors[1] for p in slack})
    for index, colored in enumerate([compound] + [{p: colors[i + 2] for p in s} for i, s in enumerate(fillers)]):
        rotation = anchor_rotation if index == 0 else int(rng.integers(4))
        if index == 1:
            choices = [k for k in range(1, 4) if _normal(_turn(set(colored), k)) != _normal(set(colored))]
            if choices:
                rotation = int(rng.choice(choices))
        transformed = {}
        for p, color in colored.items():
            q = next(iter(_turn({p}, rotation)))
            transformed[q] = color
        r0, c0 = min(r for r, c in transformed), min(c for r, c in transformed)
        objects.append({(r - r0, c - c0): color for (r, c), color in transformed.items()})
    for attempt in range(16):
        canvas = np.full((30, 30), bg, dtype=np.int8)
        blocked = set()
        okay = True
        maxr = maxc = 0
        order = sorted(range(len(objects)), key=lambda i: -len(objects[i])) if attempt == 0 else list(map(int, rng.permutation(len(objects))))
        for i in order:
            obj = objects[i]
            h, w = max(r for r, c in obj) + 1, max(c for r, c in obj) + 1
            best = None
            for r in range(31 - h):
                for c in range(31 - w):
                    if any((r + dr, c + dc) in blocked for dr, dc in obj):
                        continue
                    score = max(maxr, r + h) * max(maxc, c + w) + float(rng.random()) * 70
                    if best is None or score < best[0]:
                        best = score, r, c
            if best is None:
                okay = False
                break
            score, r, c = best
            for (dr, dc), color in obj.items():
                q = r + dr, c + dc
                canvas[q] = color
                blocked.update((q[0] + a, q[1] + b) for a, b in NEIGHBORS + ((0, 0),))
            maxr, maxc = max(maxr, r + h), max(maxc, c + w)
        if okay:
            points = np.argwhere(canvas != bg)
            r0, c0 = points.min(axis=0)
            r1, c1 = points.max(axis=0) + 1
            cropped = canvas[r0:r1, c0:c1]
            margins = [int(rng.integers(0, min(4, (30 - cropped.shape[axis]) // 2) + 1)) for axis in (0, 0, 1, 1)]
            result = np.pad(cropped, ((margins[0], margins[1]), (margins[2], margins[3])), constant_values=bg)
            counts = np.bincount(result.ravel(), minlength=10)
            while counts[bg] <= max(counts[c] for c in range(10) if c != bg):
                axis = 0 if result.shape[0] < result.shape[1] else 1
                if result.shape[axis] == 30:
                    return None
                padding = [(0, 0), (0, 0)]
                padding[axis] = (0, 1)
                result = np.pad(result, padding, constant_values=bg)
                counts = np.bincount(result.ravel(), minlength=10)
            return result
    return None


def _make_pair(rng, side, rigid_count, marker, bg, rotation, recess, long_thread):
    for _ in range(180):
        core, e = _core(rng, side, recess)
        length = e + 2 if long_thread else int(rng.integers(4, e + 2))
        route = _route(core, length)
        free = {(r, c) for r in range(side) for c in range(side)} - core - set(route)
        fillers = _grow_fillers(rng, free, rigid_count - 1)
        if fillers is None or not any(len(_orientations(s)) == 4 for s in fillers):
            continue
        # Place an asymmetric filler first so the scatter teaches a real rotation.
        fillers.sort(key=lambda s: len(_orientations(s)) != 4)
        palette = [c for c in range(10) if c not in (marker, bg)]
        colors = list(map(int, rng.choice(palette, rigid_count + 1, replace=False)))
        facts = dict(side=side, core=core, route=route, marker=marker,
                     carrier=colors[0], thread_color=colors[1],
                     pieces=[(colors[i + 2], _normal(s)) for i, s in enumerate(fillers)])
        answers = _pack(facts)
        if len(answers) != 1:
            continue
        slack = _slack(rng, length)
        if _normal(route) in _orientations(slack):
            continue
        inp = _scatter(rng, core, slack, fillers, colors, marker, bg, rotation)
        if inp is None:
            continue
        parsed = _read_input(inp)
        # A translation-only interpretation cannot reproduce the colored answer.
        if any(np.array_equal(a, answers[0]) for a in _pack(parsed, rotation=False)):
            continue
        return {'input': inp, 'output': answers[0]}
    raise RuntimeError('bounded construction could not find a unique mixed assembly')


def _counts(seed, num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train is None:
        low = 2 if num_test is None else int(num_test) + 1
        num_train = low + seed % (11 - low)
    if num_test is None:
        num_test = 1 + seed % min(4, int(num_train) - 1)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    num_train, num_test = _counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    backgrounds = list(map(int, rng.permutation([c for c in range(10) if c != marker])))
    regime = int(rng.integers(3))
    held = int(rng.integers(4))
    turns = list(map(int, rng.permutation([k for k in range(4) if k != held])))
    first_counts = list(map(int, rng.permutation([3, 4])))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            train = split == 'train'
            side = int(rng.integers(8, 11) if train and i == 0 else rng.integers(6, 11) if train else rng.integers(11, 13) if regime == 0 else rng.integers(8, 13))
            recess = (train and i == 0) or (not train and rng.random() < .5)
            if recess:
                side = max(side, 8)
            rigid_count = first_counts[i] if train and i < 2 else int(rng.integers(3, 5)) if train or regime != 1 else 5
            rotation = turns[i % 3] if train else held if regime == 2 else int(rng.integers(4))
            long_thread = train and i == 0 or (not train and i % 2 == 0)
            bg = backgrounds[(i + (0 if train else num_train)) % len(backgrounds)]
            for _ in range(12):
                pair = _make_pair(rng, side, rigid_count, marker, bg, rotation, recess, long_thread)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal pair counts')
    seen, markers, training = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            f = _read_input(inp)
            answers = _pack(f)
            _require(len(answers) == 1 and np.array_equal(answers[0], out), 'not the unique mixed exact cover')
            _require(out.shape == (f['side'], f['side']) and out[0, 0] == f['marker'], 'wrong square or anchor')
            _require(not np.any(out == f['background']), 'output must be filled')
            for color in np.unique(inp):
                if color != f['background']:
                    _require(np.sum(inp == color) == np.sum(out == color), 'material count changed')
            _require(_normal(f['route']) not in _orientations(f['slack']), 'thread could be treated as rigid')
            _require(not any(np.array_equal(q, out) for q in _pack(f, rotation=False)), 'no rotation lesson')
            _require(all(abs(a[1] - b[1]) <= 1 and b[0] == a[0] + 1 for a, b in zip(f['route'], f['route'][1:])), 'broken thread')
            slopes = [b[1] - a[1] for a, b in zip(f['route'], f['route'][1:])]
            _require(slopes == sorted(slopes, reverse=True), 'thread bends into a recess')
            _require(all(f['route'][r][1] > c for r, c in f['core'] if r < len(f['route'])), 'thread must clear the core')
            for color, shape in f['pieces']:
                output_shape = _normal(set(map(tuple, np.argwhere(out == color))))
                _require(output_shape in _orientations(shape), 'rigid shape deformed')
            rigid_area = f['side'] ** 2 - len(f['slack'])
            _require(isqrt(rigid_area) ** 2 != rigid_area, 'omitting flexible material should change area inference')
            markers.add(f['marker'])
            if split == 'train':
                training.append(f)
    _require(len(markers) == 1, 'marker convention changes within puzzle')
    _require(len({len(f['pieces']) for f in training}) >= 2, 'missing rigid count variation lesson')
    _require(len({f['background'] for f in training}) >= 2, 'missing background variation lesson')
    _require(len({f['anchor_turn'] for f in training}) >= 2, 'missing compound rotation lesson')
    _require(any(len(f['slack']) == max(r + c for r, c in f['core']) + 2 for f in training), 'missing outgoing thread lesson')
    _require(any(len(f['slack']) <= max(r + c for r, c in f['core']) + 1 for f in training), 'missing partial thread lesson')
    _require(any(sum((r, c) not in f['core'] for r, col in f['route']
                     if r <= max(u for u, v in f['core']) for c in range(col)) >= 3
                 for f in training), 'missing bridged recess lesson')


def _measure(pair):
    f = _read_input(pair['input'])
    out = np.asarray(pair['output'])
    e = max(r + c for r, c in f['core'])
    bay = sum((r, c) not in f['core'] for r, col in f['route'] if r <= max(u for u, v in f['core']) for c in range(col))
    shapes = tuple(sorted(tuple(sorted(_normal(set(map(tuple, np.argwhere(out == color)))))) for color, _ in f['pieces']))
    return {'side': f['side'], 'rigid_count': len(f['pieces']) + 1, 'thread_length': len(f['slack']),
            'peak': max(c for r, c in f['core']), 'bay_area': bay, 'outgoing': len(f['slack']) == e + 2,
            'diagonal_rigid': any(len(_components(s, EDGE)) > 1 for color, s in f['pieces']),
            'background': f['background'], 'anchor_turn': f['anchor_turn'], 'input_shape': f['input'].shape,
            'structure': (tuple(sorted(f['core'])), tuple(f['route']), shapes)}


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k]) for p, q in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    rows = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            rows.extend(_measure(p) for p in puzzle[split])
    _require({r['rigid_count'] for r in rows} == {3, 4, 5}, 'piece count diversity missing')
    _require({r['side'] for r in rows} >= {6, 7, 8, 9, 10}, 'scale diversity missing')
    _require(len({r['input_shape'] for r in rows}) >= 20, 'scatter geometry diversity missing')
    _require(len({r['structure'] for r in rows}) >= 150, 'material structure diversity missing')
    _require(len({r['thread_length'] for r in rows}) >= 5, 'thread lengths do not vary')
    _require(any(r['diagonal_rigid'] for r in rows), 'diagonal rigid attachments missing')
    _require(any(r['bay_area'] >= 3 for r in rows), 'bridged bays missing')
    _require({r['anchor_turn'] for r in rows} == {0, 1, 2, 3}, 'anchor orientations missing')
    _require(len({r['background'] for r in rows}) == 10, 'palette diversity missing')
    _require(_equal(generate_puzzle(70), generate_puzzle(70)), 'seed determinism failed')
    _require(not _equal(generate_puzzle(70), generate_puzzle(71)), 'different seeds coincide')
    for a, b in ((2, 1), (10, 1), (10, 4), (None, None), (None, 4), (2, None)):
        validate_generated_puzzle(generate_puzzle(123, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (3, 3), (2, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(99, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds)')
