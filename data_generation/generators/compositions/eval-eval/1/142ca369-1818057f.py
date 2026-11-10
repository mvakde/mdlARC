"""Crossfire circuits: L-directed signals activate and split at local pluses.

Only the standard library and NumPy are used. No parent code or saved labels are
read. All recognition is from the original input, including in feedback loops.
"""

from collections import deque
import numpy as np


CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _plus(p):
    r, c = p
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _l_cells(p, direction):
    r, c = p
    dr, dc = direction
    return {(r, c), (r - dr, c), (r, c - dc)}


def _decode(inp):
    """Three visible roles: majority background, unique whole-color L, marks."""
    inp = _grid(inp)
    colors, counts = np.unique(inp, return_counts=True)
    if len(colors) != 3 or counts.max() <= inp.size / 2:
        raise ValueError("three input colors and a majority background required")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for color in colors:
        if color == bg:
            continue
        cells = set(map(tuple, np.argwhere(inp == color)))
        if len(cells) != 3:
            continue
        r0, c0 = np.min(list(cells), axis=0)
        r1, c1 = np.max(list(cells), axis=0)
        if r1 - r0 == c1 - c0 == 1:
            missing = ({(r, c) for r in (r0, r1) for c in (c0, c1)} - cells).pop()
            elbow = (int(r0 + r1 - missing[0]), int(c0 + c1 - missing[1]))
            direction = (elbow[0] - int(missing[0]), elbow[1] - int(missing[1]))
            candidates.append((int(color), elbow, direction))
    if len(candidates) != 1:
        raise ValueError("one unambiguous transmitter color required")
    signal, elbow, direction = candidates[0]
    mark = next(int(c) for c in colors if c not in (bg, signal))
    return bg, signal, mark, elbow, direction


def _centers(mask):
    result = np.zeros_like(mask, dtype=bool)
    if min(mask.shape) >= 3:
        result[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                                      & mask[2:, 1:-1] & mask[1:-1, :-2]
                                      & mask[1:-1, 2:])
    return result


def _trace(inp, *, mutable=False, ignore_junctions=False, keep_straight=False):
    """Cell-state propagation, also offering two deliberately wrong ablations."""
    bg, signal, mark, elbow, direction = _decode(inp)
    mask = inp == mark
    gates = _centers(mask)
    if ignore_junctions:
        gates[:] = False
    out = inp.copy()
    queue = deque([(elbow[0] + direction[0], elbow[1] + direction[1], *direction)])
    seen, active, traversed = set(), set(), set()
    while queue:
        r, c, dr, dc = queue.popleft()
        state = (r, c, dr, dc)
        if not _inside((r, c), inp.shape) or state in seen:
            continue
        seen.add(state)
        traversed.add((r, c))
        if inp[r, c] == bg:
            out[r, c] = signal
        hit = bool(gates[r, c])
        if mutable and hit:
            hit = all(mask[p] for p in _plus((r, c)))
        if hit:
            active.add((r, c))
            for p in _plus((r, c)):
                out[p] = signal
                if mutable:
                    mask[p] = False
            for a, b in ((-dr, dc), (dr, -dc)):
                queue.append((r + a, c + b, a, b))
            if keep_straight:
                queue.append((r + dr, c + dc, dr, dc))
        else:
            queue.append((r + dr, c + dc, dr, dc))
    return out, active, traversed


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any valid family input without generation metadata or examples."""
    return _trace(_grid(input_grid))[0]


def _audit(inp):
    """Event-graph check: jump between locally scanned gates and canvas exits.

    This uses a direct coordinate scanner and segment reachability rather than
    the solver's pixel queue. It is an author-written check, not an independent
    reference solver.
    """
    bg, signal, mark, elbow, direction = _decode(inp)
    h, w = inp.shape
    gates = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
             if all(inp[p] == mark for p in _plus((r, c)))}
    start = (*elbow, *direction)
    todo = deque([(start, 0)])
    graph, depths, active, traversed = {}, {}, set(), set()
    while todo:
        state, depth = todo.popleft()
        if state in graph:
            continue
        r, c, dr, dc = state
        edge = min(h - 1 - r if dr > 0 else r, w - 1 - c if dc > 0 else c)
        hits = [(abs(a - r), a, b) for a, b in gates
                if (a - r) * dr > 0 and (a - r) * dr == (b - c) * dc]
        nearest = min(hits) if hits else None
        distance = nearest[0] if nearest else edge
        traversed.update((r + k * dr, c + k * dc) for k in range(1, distance + 1))
        graph[state] = []
        if nearest:
            _, a, b = nearest
            active.add((a, b))
            depths[(a, b)] = min(depths.get((a, b), depth + 1), depth + 1)
            for v, z in ((-dr, dc), (dr, -dc)):
                child = (a, b, v, z)
                graph[state].append(child)
                todo.append((child, depth + 1))
    # Removing all acyclic nodes leaves exactly the cycle-dependent remainder.
    indegree = {s: 0 for s in graph}
    for children in graph.values():
        for child in children:
            indegree[child] += 1
    pending = deque(s for s, degree in indegree.items() if degree == 0)
    removed = 0
    while pending:
        s = pending.popleft()
        removed += 1
        for child in graph[s]:
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    union = set().union(*(_plus(p) for p in active)) if active else set()
    out = inp.copy()
    for p in traversed:
        if inp[p] == bg:
            out[p] = signal
    for p in union:
        out[p] = signal
    overlap = sum(len(_plus(p)) for p in active) > len(union)
    return out, dict(bg=bg, signal=signal, mark=mark, elbow=elbow,
                     direction=direction, gates=gates, active=active, union=union,
                     traversed=traversed, depth=max(depths.values(), default=0),
                     cycle=removed < len(graph), overlap=overlap,
                     intact_inactive=any(_plus(p).isdisjoint(union) for p in gates - active),
                     crossed_clutter=any(inp[p] == mark and p not in union for p in traversed),
                     attached=any(inp[q] == mark and q not in union
                                  for p in union for d in CARDINALS
                                  if _inside(q := (p[0] + d[0], p[1] + d[1]), inp.shape)))


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _draw(mask, points, reserved):
    """Never clip a planted motif or overwrite the transmitter's clear halo."""
    if not all(_inside(p, mask.shape) and p not in reserved for p in points):
        return False
    if int(mask.sum()) + sum(not mask[p] for p in points) > int(mask.size * .44):
        return False
    for p in points:
        mask[p] = True
    return True


def _render(mask, source, direction, colors):
    bg, signal, mark = colors
    inp = np.full(mask.shape, bg, dtype=np.int8)
    inp[mask] = mark
    for p in _l_cells(source, direction):
        inp[p] = signal
    return inp


def _reserved(source, direction, shape):
    cells = _l_cells(source, direction)
    return {q for p in cells for dr in range(-1, 2) for dc in range(-1, 2)
            if _inside(q := (p[0] + dr, p[1] + dc), shape)}


def _seed_scene(rng, colors, direction, overlap, loop):
    """Seed a reachable circuit, not an entire solved parent scene."""
    if loop:
        d, lead = int(rng.integers(4, 9)), int(rng.integers(3, 6))
        h0, w0 = lead + 2 * d + 5, 2 * d + 5
        h, w = int(rng.integers(h0, 31)), int(rng.integers(w0, 31))
        rr, cc = int(rng.integers(h - h0 + 1)), int(rng.integers(w - w0 + 1))
        top = (rr + lead + 2, cc + d + 2)
        source, direction = (top[0] - lead, top[1] - lead), (1, 1)
        mask = np.zeros((h, w), dtype=bool)
        reserved = _reserved(source, direction, mask.shape)
        for p in (top, (top[0] + d, top[1] + d), (top[0] + 2*d, top[1]),
                  (top[0] + d, top[1] - d)):
            assert _draw(mask, _plus(p), reserved)
        if overlap:
            assert _draw(mask, _plus((top[0] - 1, top[1] + 1)), reserved)
    else:
        h, w = int(rng.integers(10, 31)), int(rng.integers(10, 31))
        dr, dc = direction
        # Choose source position, inward or outward, from the full feasible set.
        options = []
        for r in range(h):
            for c in range(w):
                if not all(_inside(p, (h, w)) for p in _l_cells((r, c), direction)):
                    continue
                room = min(h - 2 - r if dr > 0 else r - 1,
                           w - 2 - c if dc > 0 else c - 1)
                if room >= 5:
                    options.append((r, c, room))
        r, c, room = options[int(rng.integers(len(options)))]
        distance = int(rng.integers(3, room))
        top = (r + distance*dr, c + distance*dc)
        source = (r, c)
        mask = np.zeros((h, w), dtype=bool)
        reserved = _reserved(source, direction, mask.shape)
        assert _draw(mask, _plus(top), reserved)
        if overlap:
            choices = [(top[0] - dr, top[1] + dc), (top[0] + dr, top[1] - dc)]
            rng.shuffle(choices)
            assert any(_draw(mask, _plus(p), reserved) for p in choices)
    return mask, source, direction, reserved


def _grow(rng, mask, source, direction, colors, reserved, steps):
    for _ in range(steps):
        inp = _render(mask, source, direction, colors)
        _, f = _audit(inp)
        candidates = [p for p in f['traversed']
                      if 1 <= p[0] < mask.shape[0]-1 and 1 <= p[1] < mask.shape[1]-1
                      and all(abs(p[0]-q[0]) + abs(p[1]-q[1]) >= 4 for q in f['gates'])
                      and all(q not in reserved for q in _plus(p))]
        # Sort before RNG selection: set iteration must never govern a seed.
        candidates.sort()
        rng.shuffle(candidates)
        for p in candidates:
            if _draw(mask, _plus(p), reserved):
                break


def _decorate(rng, mask, source, direction, colors, reserved, rich, clean=False):
    """Add unreachable junctions, crossed near misses, and varied surroundings."""
    _, f = _audit(_render(mask, source, direction, colors))
    candidates = [(r, c) for r in range(1, mask.shape[0]-1)
                  for c in range(1, mask.shape[1]-1) if (r, c) not in f['traversed']]
    rng.shuffle(candidates)
    remaining = int(rng.choice([0, 1, 2, 3, 4], p=[.2, .45, .2, .1, .05]))
    for p in candidates:
        if not remaining:
            break
        if all(q not in reserved and not mask[q] for q in _plus(p)):
            if _draw(mask, _plus(p), reserved):
                remaining -= 1
    if clean:
        return
    # Place an incomplete cross directly on an otherwise empty signal segment.
    candidates = sorted(f['traversed'] - f['union'])
    rng.shuffle(candidates)
    for p in candidates:
        points = _plus(p)
        if all(_inside(q, mask.shape) and q not in reserved and not mask[q] for q in points):
            missing = CARDINALS[int(rng.integers(4))]
            points.remove((p[0] + missing[0], p[1] + missing[1]))
            _draw(mask, points, reserved)
            break
    # Independent arm extensions and diagonals around arbitrary genuine centers.
    gates = sorted(f['gates'])
    for p in gates:
        if rng.random() < .7:
            for i in rng.permutation(4)[:int(rng.integers(1, 5))]:
                dr, dc = CARDINALS[int(i)]
                for k in range(2, int(rng.integers(3, 8))):
                    _draw(mask, {(p[0] + k*dr, p[1] + k*dc)}, reserved)
        for dr, dc in DIAGONALS:
            if rng.random() < .22:
                _draw(mask, {(p[0] + dr, p[1] + dc)}, reserved)
    target = int(mask.size * rng.uniform(.17, .41)) if rich else int(mask.sum()) + int(rng.integers(0, 9))
    target = min(target, int(mask.size * .44))
    for _ in range(mask.size):
        if mask.sum() >= target:
            break
        r, c = int(rng.integers(mask.shape[0])), int(rng.integers(mask.shape[1]))
        kind = int(rng.integers(5))
        if kind == 0:  # hollow loop
            a, b = int(rng.integers(2, 6)), int(rng.integers(2, 7))
            points = {(r+i, c+j) for i in range(a) for j in range(b)
                      if i in (0, a-1) or j in (0, b-1)}
        elif kind == 1:  # compact block
            points = {(r+i, c+j) for i in range(int(rng.integers(2, 5)))
                      for j in range(int(rng.integers(2, 5)))}
        elif kind == 2:  # bent stroke
            a, b = int(rng.integers(2, 7)), int(rng.integers(2, 7))
            points = {(r+i, c) for i in range(a)} | {(r+a-1, c+j) for j in range(b)}
        elif kind == 3:  # X decoy, distinct from a cardinal plus
            points = {(r, c)} | {(r+dr, c+dc) for dr, dc in DIAGONALS}
        else:
            points = {(r, c)}
        if mask.sum() + len(points) <= target:
            _draw(mask, points, reserved)


def _transform(rng, inp):
    inp = np.rot90(inp, int(rng.integers(4)))
    if rng.integers(2):
        inp = inp[:, ::-1]
    return inp.copy()


def _make_pair(rng, colors, direction, kind, seen):
    """Bounded evidence selection from the same scene vocabulary in both splits."""
    overlap = kind == 'cascade'
    for attempt in range(180):
        loop = kind != 'contrast' and (rng.random() < (.3 if kind == 'cascade' else .35))
        mask, source, d, reserved = _seed_scene(rng, colors, direction, overlap, loop)
        additions = int(rng.integers(0, 2) if kind == 'contrast' else
                        rng.integers(1, 5) if kind == 'cascade' else rng.integers(3, 9))
        _grow(rng, mask, source, d, colors, reserved, additions)
        clean = kind == 'transfer' and rng.random() < .22
        _decorate(rng, mask, source, d, colors, reserved,
                  rich=kind == 'transfer' and not clean and rng.random() < .5,
                  clean=clean)
        inp = _transform(rng, _render(mask, source, d, colors))
        if _key(inp) in seen:
            continue
        out, f = _audit(inp)
        if not f['active']:
            continue
        if kind == 'contrast':
            if (not f['intact_inactive'] or not f['crossed_clutter']
                    or len(f['active']) > 3 or f['depth'] > 2
                    or len(f['gates']) > 7
                    or np.array_equal(_trace(inp, keep_straight=True)[0], out)):
                continue
        if kind == 'cascade' and (f['depth'] < 2 or not f['overlap'] or not f['attached']
                                 or np.array_equal(_trace(inp, mutable=True)[0], out)):
            continue
        if kind == 'transfer' and f['depth'] < 3:
            continue
        return {'input': inp, 'output': out}
    return _fallback(rng, colors, kind, seen)


def _fallback(rng, colors, kind, seen):
    """Finite clear construction; >300 shapes for at most 14 example slots."""
    shapes = [(h, w) for h in range(12, 31) for w in range(14, 31)]
    rng.shuffle(shapes)
    for shape in shapes:
        mask = np.zeros(shape, dtype=bool)
        points = (_plus((5, 5)) | _plus((6, 4)) | _plus((9, 10))
                  | {(3, 3), (5, 7)})
        if kind == 'transfer':
            points |= _plus((8, 6))
        for p in points:
            mask[p] = True
        inp = _transform(rng, _render(mask, (1, 1), (1, 1), colors))
        if _key(inp) not in seen:
            return {'input': inp, 'output': _audit(inp)[0]}
    raise AssertionError("uniqueness bound exceeds maximum legal puzzle size")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, 2, 10, 'num_train')
    ntest = _count(num_test, 1, 4, 'num_test')
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .25)
    base = tuple(map(int, rng.choice(10, 3, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for i in range(count):
            colors = base if stable else tuple(map(int, rng.choice(10, 3, replace=False)))
            kind = ('contrast' if i == 0 else 'cascade' if i == 1 else
                    str(rng.choice(['contrast', 'cascade', 'transfer']))) if split == 'train' else 'transfer'
            for _ in range(24):
                pair = _make_pair(rng, colors, DIAGONALS[int(rng.integers(4))], kind, seen)
                if i != 1 or split != 'train':
                    break
                if _decode(pair['input'])[4] != _decode(puzzle['train'][0]['input'])[4]:
                    break
            else:
                # Rotation changes the visible launch direction and all circuit
                # geometry consistently; no different semantic convention.
                pair = {k: np.rot90(v).copy() for k, v in pair.items()}
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            expected, f = _audit(inp)
            assert np.array_equal(out, expected), "event graph disagrees with output"
            assert np.array_equal(solve(inp), out), "cell tracer disagrees with output"
            assert f['active']
            assert np.array_equal(out[inp == f['signal']], inp[inp == f['signal']])
            mark_changes = set(map(tuple, np.argwhere((inp == f['mark']) & (out != inp))))
            assert mark_changes == f['union'], "change exactly the activated five-cell union"
            assert set(np.unique(out)) <= set(np.unique(inp)), "no hidden output color"
            assert np.all(out[inp != out] == f['signal'])
            assert not np.array_equal(_trace(inp, ignore_junctions=True)[0], out), "B must affect route"
            all_plus = inp.copy()
            for p in f['gates']:
                for q in _plus(p):
                    all_plus[q] = f['signal']
            assert not np.array_equal(all_plus, out), "A must affect activation"
            if split == 'train':
                f['frozen_witness'] = not np.array_equal(_trace(inp, mutable=True)[0], out)
                f['no_straight_witness'] = not np.array_equal(_trace(inp, keep_straight=True)[0], out)
                training.append(f)
            else:
                assert f['depth'] >= 3, "test must compose at least three junction dependencies"
    assert len({f['direction'] for f in training}) >= 2
    assert any(f['crossed_clutter'] for f in training)
    assert any(f['intact_inactive'] for f in training)
    assert any(f['attached'] for f in training)
    assert any(f['frozen_witness'] for f in training)
    assert any(f['no_straight_witness'] and len(f['active']) <= 3 and f['depth'] <= 2
               for f in training), "include a readable split-versus-straight contrast"


def self_test():
    """Bounded deterministic diversity, semantic regressions, count validation."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features, shapes, structures, palettes, fingerprints = [], set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        fingerprint = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                _, f = _audit(inp)
                features.append(f)
                shapes.add(inp.shape)
                palettes.add((f['bg'], f['signal'], f['mark']))
                normalized = np.zeros_like(inp, dtype=np.uint8)
                normalized[inp == f['signal']] = 1
                normalized[inp == f['mark']] = 2
                structures.add(_key(normalized))
                fingerprint.append((_key(inp), _key(pair['output'])))
        fingerprints.add(tuple(fingerprint))
    assert len(fingerprints) == 42
    assert len(shapes) >= 65 and len(structures) >= 180 and len(palettes) >= 80
    assert {f['direction'] for f in features} == set(DIAGONALS)
    assert any(f['cycle'] for f in features) and any(not f['cycle'] for f in features)
    assert max(len(f['active']) for f in features) >= 7
    assert max(len(f['gates']) - len(f['active']) for f in features) >= 4
    assert len({len(f['active']) for f in features}) >= 8
    assert any(30 in shape for shape in shapes)
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            validate_generated_puzzle(generate_puzzle(81, ntrain, ntest))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _regressions()
    # Detect both omitted signal pixels and over-recoloring of disconnected gates.
    good = puzzles[0]
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in good[s]] for s in good}
    pair = bad['test'][0]
    p = tuple(np.argwhere(pair['input'] != pair['output'])[0])
    pair['output'][p] = pair['input'][p]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an erased required cell")


def _regressions():
    # Exact hand-enumerated two-splitter case, with overlapping original pluses,
    # transparent clutter on the launch path, and an unreachable plus.
    mask = np.zeros((12, 14), dtype=bool)
    for p in _plus((5, 5)) | _plus((6, 4)) | _plus((9, 10)) | {(3, 3)}:
        mask[p] = True
    inp = _render(mask, (1, 1), (1, 1), (0, 2, 7))
    expected = inp.copy()
    background_signal = {(2, 2), (4, 4), (4, 6), (3, 7), (2, 8), (1, 9), (0, 10),
                         (7, 5), (8, 6), (9, 7), (10, 8), (11, 9),
                         (5, 3), (4, 2), (3, 1), (2, 0)}
    for p in background_signal | _plus((5, 5)) | _plus((6, 4)):
        expected[p] = 2
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(_audit(inp)[0], expected)
    assert expected[3, 3] == 7 and expected[9, 10] == 7
    assert not np.array_equal(_trace(inp, mutable=True)[0], expected)
    # An unreached center can be lit as an active neighbor's arm. That does not
    # activate its other arms or emit additional branches from its center.
    shared = inp.copy()
    for p in _plus((5, 6)):
        shared[p] = 7
    shared_expected = shared.copy()
    for p in background_signal:
        if shared[p] == 0:
            shared_expected[p] = 2
    for p in _plus((5, 5)) | _plus((6, 4)):
        shared_expected[p] = 2
    assert np.array_equal(solve(shared), shared_expected)
    _, f = _audit(shared)
    assert (5, 6) not in f['active'] and shared_expected[5, 6] == 2
    assert shared_expected[4, 6] == 7
    # Exhaustive local B recognition, including diagonals and extensions.
    for bits in range(512):
        block = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        assert bool(_centers(block)[1, 1]) == all(block[p] for p in _plus((1, 1)))
    clipped = np.zeros((4, 5), dtype=bool)
    clipped[0, 1:4] = True
    clipped[1, 2] = clipped[3, 2] = True
    assert not _centers(clipped).any(), "outside cells never wrap"
    # D4 and arbitrary palette covariance are exact rule properties.
    for k in range(4):
        for flip in (False, True):
            a, b = np.rot90(inp, k), np.rot90(expected, k)
            if flip:
                a, b = a[:, ::-1], b[:, ::-1]
            assert np.array_equal(solve(a), b)
    palette = np.array([9, 1, 0, 3, 4, 5, 6, 8, 7, 2])
    assert np.array_equal(solve(palette[inp]), palette[expected])
    # The bounded sampler's recovery path must preserve the same evidence.
    rng, seen = np.random.default_rng(31), set()
    for kind in ('contrast', 'cascade', 'transfer'):
        pair = _fallback(rng, (0, 2, 7), kind, seen)
        seen.add(_key(pair['input']))
        predicted, f = _audit(pair['input'])
        assert np.array_equal(predicted, solve(pair['input']))
        assert f['crossed_clutter'] and f['attached'] and f['gates'] - f['active']
        assert not np.array_equal(_trace(pair['input'], mutable=True)[0], predicted)
        assert not np.array_equal(_trace(pair['input'], keep_straight=True)[0], predicted)
        assert f['depth'] >= (3 if kind == 'transfer' else 2)


if __name__ == '__main__':
    self_test()
    print('Crossfire circuits: self_test passed (42 diversity seeds).')
