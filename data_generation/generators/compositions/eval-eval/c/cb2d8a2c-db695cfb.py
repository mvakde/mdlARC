"""Signal escape: finite diagonal fences carry local unary detour commands.

Only Python's standard library and NumPy are used. All role decoding and solving
read the query grid alone. Sampling profiles never enter the solver.
"""
from __future__ import annotations

import itertools
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _whole_key(p):
    return tuple(tuple((_key(np.asarray(q['input'])), _key(np.asarray(q['output'])))
                       for q in p[s]) for s in ('train', 'test'))


def _physical(model, f, l):
    r, c = model['origin']
    slope = model['slope']
    return r + f + l, c - slope * f + slope * l


def _coordinates(model, point):
    dr, dc = point[0] - model['origin'][0], point[1] - model['origin'][1]
    f2, l2 = dr - model['slope'] * dc, dr + model['slope'] * dc
    _require(f2 % 2 == 0 and l2 % 2 == 0, "fences must share launch parity")
    return f2 // 2, l2 // 2


def _inside(shape, point, margin=0):
    return all(margin <= p < n - margin for p, n in zip(point, shape))


def _l_bounds(shape, origin, f):
    # Constructor uses slope -1. The input decoder supports both slopes.
    r, c = origin
    h, w = shape
    return max(-r-f, c+f-w+1), min(h-1-r-f, c+f)


def _span_cells(model, fence):
    return [_physical(model, fence['f'], l) for l in range(fence['lo'], fence['hi'] + 1)]


def _trace(model, shape):
    """Trace two arms; also reconstruct hit/miss decisions from finite spans."""
    path = np.zeros(shape, dtype=bool)
    decisions = []
    for sign in (-1, 1):
        current, start, previous_plane = 0, 0, 0
        fences = sorted((b for b in model['fences'] if sign * b['f'] > 0),
                        key=lambda b: abs(b['f']))
        for b in fences:
            f, d = b['f'], b['n'] + 1
            _require(sign * (f - previous_plane) >= d + 1, "fence planes too close")
            hit = b['lo'] <= current <= b['hi']
            decision = dict(b, hit=hit, incoming=current, arm=sign)
            if hit:
                turn = f - sign * d
                target = b['hi'] + d if b['side'] == 1 else b['lo'] - d
                _require(sign * (turn - start) >= 0, "turn goes backwards")
                _require(_inside(shape, _physical(model, turn, current), 1)
                         and _inside(shape, _physical(model, turn, target), 1),
                         "turn touches or exceeds boundary")
                _require(_inside(shape, _physical(model, f + sign, target)),
                         "signal exits before passing fence")
                for t in range(start, turn + sign, sign):
                    p = _physical(model, t, current)
                    _require(_inside(shape, p, 1), "approach crosses boundary")
                    path[p] = True
                step = 1 if target > current else -1
                for l in range(current, target + step, step):
                    p = _physical(model, turn, l)
                    _require(_inside(shape, p, 1), "lateral turn crosses boundary")
                    path[p] = True
                current, start = target, turn
                decision.update(turn=turn, target=target)
            previous_plane = f
            decisions.append(decision)
        f = start
        while _inside(shape, _physical(model, f, current)):
            path[_physical(model, f, current)] = True
            f += sign

    spans = set(_span_cells(model, model['launch']))
    for b in model['fences']:
        spans.update(_span_cells(model, b))
    path_cells = set(map(tuple, np.argwhere(path).tolist()))
    _require(path_cells & spans == {model['origin']}, "route collides with a fence")
    return path, decisions


def _finish(grid, model, path):
    bg, primary, instruction, signal = model['roles']
    out = grid.copy()
    for b in [model['launch']] + model['fences']:
        for p in _span_cells(model, b):
            out[p] = primary
    out[path] = signal
    return out


def _decode_roles(grid, background, primary, instruction, signal):
    signal_points = np.argwhere(grid == signal)
    _require(signal_points.shape == (1, 2), "signal role is not a singleton")
    origin = tuple(map(int, signal_points[0]))
    _require(_inside(grid.shape, origin, 2), "launch too near boundary")
    endpoints = [tuple(map(int, p)) for p in np.argwhere(grid == primary)]
    pairs, degree = [], {p: 0 for p in endpoints}
    for a, b in itertools.combinations(endpoints, 2):
        if abs(a[0]-b[0]) == abs(a[1]-b[1]):
            degree[a] += 1
            degree[b] += 1
            pairs.append((a, b))
    _require(len(pairs) >= 2 and max(degree.values(), default=0) <= 1,
             "endpoint pairs not uniquely recoverable")
    slopes = {1 if (b[0]-a[0]) * (b[1]-a[1]) > 0 else -1 for a, b in pairs}
    _require(len(slopes) == 1, "fences not parallel")
    model = {'origin': origin, 'slope': slopes.pop(),
             'roles': (background, primary, instruction, signal), 'fences': []}
    launch = None
    planes = set()
    embedded = set()
    for a, b in pairs:
        f, la = _coordinates(model, a)
        fb, lb = _coordinates(model, b)
        _require(f == fb and f not in planes, "duplicate fence plane")
        planes.add(f)
        lo, hi = sorted((la, lb))
        _require(hi - lo >= 3, "fence too short")
        fence = {'f': f, 'lo': lo, 'hi': hi}
        cells = _span_cells(model, fence)
        _require(all(int(grid[p]) in (background, primary, instruction, signal) for p in cells),
                 "unrecognized fence contents")
        markers = [l for l in range(lo, hi+1) if grid[_physical(model, f, l)] == instruction]
        embedded.update(_physical(model, f, l) for l in markers)
        if origin in cells:
            _require(launch is None and not markers and lo <= -2 and hi >= 2,
                     "invalid launch fence")
            launch = fence
        else:
            n = len(markers)
            _require(1 <= n <= 3 and hi - lo >= 2*n + 1, "invalid unary code")
            low = markers == [lo + 1 + 2*j for j in range(n)]
            high = markers == sorted(hi - 1 - 2*j for j in range(n))
            _require(low != high, "no unique marked endpoint")
            fence.update(n=n, side=1 if low else -1)
            model['fences'].append(fence)
    _require(launch is not None, "launch not on a finite fence")
    model['launch'] = launch
    _require(all(b['f'] != 0 for b in model['fences']), "coded fence on launch plane")
    path, decisions = _trace(model, grid.shape)
    _require(bool(np.all(grid[path & (grid != signal)] == background)),
             "route would overwrite an inert dot")
    model.update(path=path, decisions=decisions, embedded=embedded,
                 inert_instruction=set(map(tuple, np.argwhere(grid == instruction).tolist())) - embedded,
                 unpaired={p for p, n in degree.items() if n == 0})
    return _finish(grid, model, path), model


def _decode(value):
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 4 and counts.max() > grid.size // 2, "need four roles and majority canvas")
    background = int(colors[counts.argmax()])
    foreground = [int(c) for c in colors if c != background]
    candidates = []
    for primary, instruction, signal in itertools.permutations(foreground):
        try:
            candidates.append(_decode_roles(grid, background, primary, instruction, signal))
        except ValueError:
            pass
    _require(len(candidates) == 1, "foreground roles are not uniquely identified by geometry")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only original input geometry and visible color roles."""
    return _decode(input_grid)[0]


def _choose_span(rng, shape, model, f, n, current, used_l, spec):
    lo_min, hi_max = _l_bounds(shape, model['origin'], f)
    options = []
    for lo in range(lo_min, hi_max + 1):
        if lo in used_l:
            continue
        for hi in range(lo + 2*n + 1, hi_max + 1):
            if hi in used_l:
                continue
            hit = lo <= current <= hi
            if 'hit' in spec and hit != spec['hit']:
                continue
            if spec.get('new') and lo <= 0 <= hi:
                continue
            if spec.get('original') and not lo <= 0 <= hi:
                continue
            for side in (-1, 1):
                if 'side' in spec and side != spec['side']:
                    continue
                d = n + 1
                arm = 1 if f > 0 else -1
                target = hi + d if side == 1 else lo - d
                turn = f - arm*d
                if hit and not (_inside(shape, _physical(model, turn, current), 1)
                                and _inside(shape, _physical(model, turn, target), 1)
                                and _inside(shape, _physical(model, f+arm, target))):
                    continue
                # Teaching skips occur while the displaced signal is still visible.
                if spec.get('original') and not _inside(shape, _physical(model, f+arm, current)):
                    continue
                options.append({'f': f, 'lo': lo, 'hi': hi, 'n': n, 'side': side})
    if not options:
        return None
    return options[int(rng.integers(len(options)))]


def _make_plan(rng, branches, witness):
    count = sum(len(v) for v in branches.values())
    high_code = max(s['n'] for specs in branches.values() for s in specs)
    for _ in range(160):
        low = 27 if witness == 'repeat' else 24 if witness == 'miss' or count >= 3 or high_code == 3 else 12
        shape = tuple(map(int, rng.integers(low, 31, size=2)))
        if rng.random() < .15:
            shape = (30, 30)
        h, w = shape
        if set(branches) == {1}:
            origin = (int(rng.integers(3, min(12, h-3))), int(rng.integers(3, min(12, w-3))))
        elif set(branches) == {-1}:
            origin = (h-1-int(rng.integers(3, min(12, h-3))),
                      w-1-int(rng.integers(3, min(12, w-3))))
        else:
            origin = (h//2 + int(rng.integers(-2, 3)), w//2 + int(rng.integers(-2, 3)))
        model = {'origin': origin, 'slope': -1, 'fences': []}
        bounds = _l_bounds(shape, origin, 0)
        left = int(rng.integers(max(bounds[0], -5), -1))
        right = int(rng.integers(2, min(bounds[1], 5) + 1))
        model['launch'] = {'f': 0, 'lo': left, 'hi': right}
        used_l, failed = {left, right}, False
        for arm, specs in branches.items():
            previous, current = 0, 0
            for spec in specs:
                n = spec['n']
                f = previous + arm * (n + 2 + int(rng.integers(0, 3)))
                fence = _choose_span(rng, shape, model, f, n, current, used_l, spec)
                if fence is None:
                    failed = True
                    break
                model['fences'].append(fence)
                used_l.update((fence['lo'], fence['hi']))
                if fence['lo'] <= current <= fence['hi']:
                    current = fence['hi'] + n + 1 if fence['side'] == 1 else fence['lo'] - n - 1
                previous = f
            if failed:
                break
        if not failed:
            try:
                _trace(model, shape)
                return shape, model
            except ValueError:
                pass
    return None


def _fallback_plan(branches, witness):
    """Finite constructive backup, with the same structural teaching requests."""
    if witness == 'repeat':
        model = {'origin': (8, 8), 'slope': -1,
                 'launch': {'f': 0, 'lo': -4, 'hi': 4},
                 'fences': [dict(f=3, lo=-2, hi=1, n=1, side=1),
                            dict(f=7, lo=2, hi=7, n=2, side=1)]}
    elif witness == 'miss':
        n = branches[-1][0]['n']
        model = {'origin': (21, 21), 'slope': -1,
                 'launch': {'f': 0, 'lo': -4, 'hi': 4},
                 'fences': [dict(f=-n-2, lo=-2, hi=2*n-1, n=n, side=-1),
                            dict(f=-n-5, lo=-1, hi=2, n=1, side=1)]}
    elif len(branches) == 2:
        total = sum(map(len, branches.values()))
        if total == 2:
            # Different-length spans on opposite arms, both intercepted.
            model = {'origin': (14, 14), 'slope': -1,
                     'launch': {'f': 0, 'lo': -4, 'hi': 4},
                     'fences': [dict(f=3, lo=-2, hi=1, n=1, side=1),
                                dict(f=-4, lo=-1, hi=5, n=2, side=-1)]}
        else:
            positive = [dict(f=3, lo=-2, hi=1, n=1, side=1),
                        dict(f=6, lo=2, hi=5, n=1, side=-1),
                        dict(f=9, lo=-3, hi=0, n=1, side=1)]
            negative = [dict(f=-3, lo=-1, hi=3, n=1, side=-1),
                        dict(f=-6, lo=-5, hi=4, n=1, side=1)]
            model = {'origin': (14, 14), 'slope': -1,
                     'launch': {'f': 0, 'lo': -6, 'hi': 6},
                     'fences': positive[:2 if total < 5 else 3] + negative[:1 if total == 3 else 2]}
    elif sum(map(len, branches.values())) >= 3:
        model = {'origin': (7, 7), 'slope': -1,
                 'launch': {'f': 0, 'lo': -4, 'hi': 4},
                 'fences': [dict(f=3, lo=-2, hi=1, n=1, side=1),
                            dict(f=6, lo=2, hi=5, n=1, side=-1),
                            dict(f=9, lo=-3, hi=0, n=1, side=1)]}
    else:
        n = max(s['n'] for v in branches.values() for s in v)
        model = {'origin': (10, 10), 'slope': -1,
                 'launch': {'f': 0, 'lo': -4, 'hi': 4},
                 'fences': [dict(f=n+2, lo=-2, hi=2*n-1, n=n, side=1)]}
    _trace(model, (30, 30))
    return (30, 30), model


def _render(rng, shape, model, roles, need_inert):
    model['roles'] = roles
    background, primary, instruction, signal = roles
    grid = np.full(shape, background, dtype=np.int8)
    grid[model['origin']] = signal
    spans = set()
    for b in [model['launch']] + model['fences']:
        cells = _span_cells(model, b)
        spans.update(cells)
        grid[cells[0]] = grid[cells[-1]] = primary
        if b['f']:
            for j in range(b['n']):
                l = b['lo']+1+2*j if b['side'] == 1 else b['hi']-1-2*j
                grid[_physical(model, b['f'], l)] = instruction
    path, _ = _trace(model, shape)
    # Useful finite-versus-infinite witness: a dot on a coded fence continuation.
    if need_inert or rng.random() < .55:
        beyond = []
        for b in model['fences']:
            lo, hi = _l_bounds(shape, model['origin'], b['f'])
            for l in range(lo, hi+1):
                p = _physical(model, b['f'], l)
                if p not in spans and not path[p]:
                    beyond.append((min(abs(l-b['lo']), abs(l-b['hi'])), p))
        if not beyond:
            return None
        # Prefer immediately beyond endpoints, but allow broader continuation dots.
        beyond.sort(key=lambda item: item[0])
        pool = beyond[:min(len(beyond), 4)]
        grid[pool[int(rng.integers(len(pool)))][1]] = instruction
    primary_points = [tuple(map(int, p)) for p in np.argwhere(grid == primary)]
    distractors = int(rng.choice((0, 0, 0, 1, 1, 2)))
    for index in rng.permutation(grid.size):
        if not distractors:
            break
        p = (int(index)//shape[1], int(index)%shape[1])
        if grid[p] != background or p in spans or path[p]:
            continue
        if any(abs(p[0]-q[0]) == abs(p[1]-q[1]) for q in primary_points):
            continue
        grid[p] = primary
        primary_points.append(p)
        distractors -= 1
    for index in rng.permutation(grid.size)[:int(rng.integers(0, 5))]:
        p = (int(index)//shape[1], int(index)%shape[1])
        if grid[p] == background and p not in spans and not path[p]:
            grid[p] = instruction
    output = _finish(grid, model, path)
    return {'input': grid, 'output': output}


def _transform(array, rotation, transpose):
    a = np.rot90(array, rotation)
    return (a.T if transpose else a).copy()


def _counts(a, b):
    for name, value, low, high in (('num_train', a, 2, 10), ('num_test', b, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        _require(low <= int(value) <= high, f'invalid {name}')
    _require(a > b, 'num_train must exceed num_test')
    return int(a), int(b)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, 'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    maximum = int(rng.choice((2, 3)))
    shift = int(rng.integers(3))  # bilateral; longer sequence; held-out slope
    recolor = bool(rng.integers(2))
    train_slope_rotation = int(rng.integers(2))
    witnesses = ['repeat', 'miss'] + ['ordinary'] * (num_train-2)
    rng.shuffle(witnesses)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            witness = witnesses[i] if split == 'train' else 'ordinary'
            for attempt in range(40):
                if witness == 'repeat':
                    branches = {1: [dict(n=1, hit=True, side=1),
                                    dict(n=2, hit=True, side=1, new=True)]}
                elif witness == 'miss':
                    branches = {-1: [dict(n=maximum, hit=True, side=-1),
                                     dict(n=1, hit=False, original=True)]}
                else:
                    if split == 'test' and shift == 0:
                        # At least one intercepted command on each outgoing arm.
                        branches = {s: [dict(n=int(rng.integers(1, maximum+1)), hit=True)]
                                    for s in (-1, 1)}
                        if rng.random() < .3:
                            branches[int(rng.choice((-1, 1)))].append(dict(n=1))
                    elif shift == 2:
                        # A held-out slope can accompany simple or longer scenes
                        # in either split, rather than making count identify test.
                        number = int(rng.choice((1, 1, 2, 2, 3, 4, 5)))
                        arm = int(rng.choice((-1, 1)))
                        if number == 1 or (number == 2 and rng.random() < .6):
                            allocation = {arm: number}
                        else:
                            first = int(rng.integers(max(1, number-3), min(3, number-1)+1))
                            allocation = {arm: first, -arm: number-first}
                        branches = {}
                        for s, size in allocation.items():
                            ns = rng.choice((1, 1, 1, 2) if number >= 3 else tuple(range(1, maximum+1)),
                                            size=size)
                            branches[s] = [dict(n=int(n), **({'hit': True} if j == 0 else {}))
                                           for j, n in enumerate(ns)]
                    else:
                        arm = int(rng.choice((-1, 1)))
                        number = 3 if split == 'test' and shift == 1 else int(rng.integers(1, 3))
                        branches = {arm: [dict(n=int(rng.integers(1, maximum+1)), hit=True)]
                                    + [dict(n=int(rng.integers(1, maximum+1))) for _ in range(number-1)]}
                built = _make_plan(rng, branches, witness)
                if built is None:
                    built = _fallback_plan(branches, witness)
                shape, model = built
                roles = palette
                if recolor:
                    bg, signal = map(int, rng.choice([c for c in range(10) if c not in palette[1:3]],
                                                    2, replace=False))
                    roles = (bg, palette[1], palette[2], signal)
                pair = _render(rng, shape, model, roles, witness != 'ordinary')
                if pair is None:
                    continue
                if shift == 2:
                    parity = train_slope_rotation if split == 'train' else 1-train_slope_rotation
                    rotation = parity + 2*int(rng.integers(2))
                else:
                    rotation = int(rng.integers(4))
                transpose = bool(rng.integers(2))
                pair = {k: _transform(v, rotation, transpose) for k, v in pair.items()}
                if _key(pair['input']) in seen:
                    continue
                try:
                    predicted, decoded = _decode(pair['input'])
                except ValueError:
                    continue
                _require(np.array_equal(predicted, pair['output']), 'construction/decoding disagreement')
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('unique readable scene budget exhausted')
        if split == 'train':
            sides = {d['side'] for q in puzzle[split] for d in _decode(q['input'])[1]['decisions']
                     if d['hit']}
            if len(sides) == 1:
                # A half-turn keeps all structural witnesses and the fence slope,
                # while ensuring that both physical marked-end meanings are taught.
                q = puzzle[split][-1]
                for k in q:
                    q[k] = np.rot90(q[k], 2).copy()
                seen = {_key(q['input']) for q in puzzle['train']}
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _path_invariants(model):
    path = model['path']
    cells = set(map(tuple, np.argwhere(path).tolist()))
    degrees = []
    for r, c in cells:
        degrees.append(sum((r+dr, c+dc) in cells for dr in (-1, 0, 1)
                           for dc in (-1, 0, 1) if dr or dc))
    _require(degrees.count(1) == 2 and all(n in (1, 2) for n in degrees),
             'path is not one simple diagonal chain')
    visited, stack = set(), [model['origin']]
    while stack:
        p = stack.pop()
        if p in visited:
            continue
        visited.add(p)
        r, c = p
        stack.extend((r+dr, c+dc) for dr in (-1, 1) for dc in (-1, 1)
                     if (r+dr, c+dc) in cells and (r+dr, c+dc) not in visited)
    _require(visited == cells, 'disconnected escaping signal')
    h, w = path.shape
    boundary = {(r, c) for r, c in cells if r in (0, h-1) or c in (0, w-1)}
    _require(len(boundary) == 2, 'signal must have exactly two boundary contacts')


def _features(grid):
    _, m = _decode(grid)
    codes = m['fences']
    decisions = m['decisions']
    hit = [d for d in decisions if d['hit']]
    repeats = any(a['side'] == b['side'] for s in (-1, 1)
                  for a, b in zip([d for d in hit if d['arm'] == s],
                                  [d for d in hit if d['arm'] == s][1:]))
    return {'shape': list(np.asarray(grid).shape), 'roles': list(m['roles']),
            'slope': m['slope'], 'coded_fences': len(codes),
            'bilateral': any(b['f'] < 0 for b in codes) and any(b['f'] > 0 for b in codes),
            'counts': [b['n'] for b in codes], 'lengths': [b['hi']-b['lo']+1 for b in codes],
            'hits': len(hit), 'misses': len(decisions)-len(hit), 'repeat': repeats,
            'new_hit': any(d['hit'] and not d['lo'] <= 0 <= d['hi'] for d in decisions),
            'original_miss': any(not d['hit'] and d['lo'] <= 0 <= d['hi'] for d in decisions),
            'bypass_sides': [d['side'] for d in hit],
            'inert_instruction': len(m['inert_instruction']), 'unpaired': len(m['unpaired'])}


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records, conventions = set(), {'train': [], 'test': []}, set()
    ablate_a, ablate_b = False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            grid, out = _grid(pair['input']), _grid(pair['output'])
            _require(grid.shape == out.shape, 'shape changed')
            _require(_key(grid) not in seen, 'duplicate input')
            seen.add(_key(grid))
            expected, model = _decode(grid)
            _require(np.array_equal(expected, out), 'incorrect output')
            _path_invariants(model)
            bg, primary, instruction, signal = model['roles']
            conventions.add((primary, instruction))
            _require(bool(np.all(out[grid == primary] == primary)), 'endpoint was changed')
            _require(out[model['origin']] == signal, 'launch was changed')
            _require(all(out[p] == primary for p in model['embedded']), 'instruction not normalized')
            _require(all(out[p] == instruction for p in model['inert_instruction']), 'inert instruction changed')
            _require(all(out[p] == primary for p in model['unpaired']), 'unpaired dot changed')
            features = _features(grid)
            records[split].append(features)
            if split == 'train':
                straight = np.zeros(grid.shape, dtype=bool)
                for s in (-1, 1):
                    f = 0
                    while _inside(grid.shape, _physical(model, f, 0)):
                        straight[_physical(model, f, 0)] = True
                        f += s
                ablate_a |= not np.array_equal(_finish(grid, model, straight), out)
                for d in model['decisions']:
                    if not d['hit']:
                        continue
                    # Infinite-collinearity counting would move this visible turn.
                    extra = 0
                    for p in model['inert_instruction']:
                        try:
                            f, l = _coordinates(model, p)
                        except ValueError:
                            continue
                        extra += f == d['f'] and not d['lo'] <= l <= d['hi']
                    if extra:
                        wrong_corner = _physical(model, d['turn'] - d['arm']*extra,
                                                 d['target'] + d['side']*extra)
                        ablate_b |= (not _inside(grid.shape, wrong_corner)
                                     or out[wrong_corner] != signal)
    _require(len(conventions) == 1, 'endpoint/instruction convention changed')
    tr, te = records['train'], records['test']
    train_counts = {n for r in tr for n in r['counts']}
    _require({1, 2} <= train_counts, 'counting not demonstrated')
    _require({n for r in te for n in r['counts']} <= train_counts, 'undemonstrated test count')
    _require({s for r in tr for s in r['bypass_sides']} == {-1, 1}, 'both bypass ends not demonstrated')
    _require(any(r['repeat'] and r['new_hit'] for r in tr), 'history-dependent repeated detour missing')
    _require(any(r['original_miss'] for r in tr), 'history-dependent skipped fence missing')
    _require(ablate_a and ablate_b, 'parent ablation witness missing')
    bilateral_shift = all(not r['bilateral'] for r in tr) and all(r['bilateral'] for r in te)
    count_shift = max(r['coded_fences'] for r in tr) < min(r['coded_fences'] for r in te)
    slope_shift = {r['slope'] for r in tr}.isdisjoint({r['slope'] for r in te})
    _require(bilateral_shift or count_shift or slope_shift, 'meaningful structural shift missing')


def self_test() -> dict:
    """Bounded consistency, contrast, mutation and 42-seed variation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_whole_key(p) for p in puzzles}) == 42, 'cross-seed repetition')
    _require(_whole_key(puzzles[7]) == _whole_key(generate_puzzle(7)), 'nondeterministic generation')
    records, normalized = [], set()
    for p in puzzles:
        for s in ('train', 'test'):
            for q in p[s]:
                a = q['input']
                r = _features(a)
                records.append(dict(r, split=s))
                semantic = np.empty_like(a)
                for role, color in enumerate(r['roles']):
                    semantic[a == color] = role
                normalized.add(_key(semantic))
    shapes = {tuple(r['shape']) for r in records}
    _require(len(shapes) >= 45 and len(normalized) >= 180, 'insufficient geometric variation')
    _require({n for r in records for n in r['counts']} == {1, 2, 3}, 'unary variation missing')
    _require({r['slope'] for r in records} == {-1, 1}, 'slope variation missing')
    _require({1, 2, 3, 4, 5} <= {r['coded_fences'] for r in records}, 'fence count variation missing')
    _require(len({n for r in records for n in r['lengths']}) >= 8, 'length variation missing')
    _require({0, 1, 2} <= {r['unpaired'] for r in records}, 'optional distractor variation missing')
    _require(any(r['bilateral'] for r in records) and any(r['misses'] for r in records), 'interaction variation missing')
    _require(len({tuple(r['roles']) for r in records}) >= 30, 'palette variation missing')
    legal = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(103, a, b)
            _require(len(p['train']) == a and len(p['test']) == b, 'count mismatch')
            legal += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # A literal contrast fixture specifies physical cells without using the
    # constructor's plan, coordinate conversion, completion or path helpers.
    literal = np.zeros((30, 30), dtype=np.int8)
    for p in ((4, 12), (12, 4), (9, 13), (12, 10), (17, 13), (22, 8)):
        literal[p] = 1
    for p in ((10, 12), (18, 12), (20, 10), (8, 14)):
        literal[p] = 2
    literal[8, 8] = 3
    expected = np.zeros_like(literal)
    expected[8, 14] = 2  # collinear but beyond the first fence's endpoint
    for first, last, diagonal_sum in ((4, 12, 16), (9, 12, 22), (17, 22, 30)):
        for r in range(first, last+1):
            expected[r, diagonal_sum-r] = 1
    for first, last, sign, intercept in ((0, 9, 1, 0), (9, 12, -1, 18),
                                         (12, 15, 1, -6), (15, 22, -1, 24),
                                         (22, 29, 1, -20)):
        for r in range(first, last+1):
            expected[r, sign*r+intercept] = 3
    _require(np.array_equal(solve(literal), expected), 'literal detour fixture failed')
    removed_inert = literal.copy()
    removed_inert[8, 14] = 0
    _require(np.array_equal(solve(removed_inert) == 3, expected == 3),
             'off-span instruction changed the escaping path')
    reversed_code = literal.copy()
    reversed_code[18, 12] = reversed_code[20, 10] = 0
    reversed_code[19, 11] = reversed_code[21, 9] = 2
    reversed_output = solve(reversed_code)
    _require(reversed_output[11, 13] == 3 and reversed_output[22, 2] == 0,
             'changing the marked endpoint did not change the bypass')
    # Exercise emergency construction directly, including both-arm collections.
    backup_cases = [({1: [dict(n=1), dict(n=2)]}, 'repeat'),
                    ({-1: [dict(n=2), dict(n=1)]}, 'miss'),
                    ({-1: [dict(n=3), dict(n=1)]}, 'miss'),
                    ({1: [dict(n=3)]}, 'ordinary'),
                    ({1: [dict(n=1)]*3}, 'ordinary')]
    backup_cases += [({1: [dict(n=1)]*(total-2), -1: [dict(n=1)]*2}, 'ordinary')
                     for total in (3, 4, 5)]
    for branches, witness in backup_cases:
        shape, model = _fallback_plan(branches, witness)
        pair = _render(np.random.default_rng(218), shape, model, (0, 1, 2, 3), True)
        _require(pair is not None and np.array_equal(solve(pair['input']), pair['output']),
                 'emergency construction does not obey the rule')
    # Solving is covariant under geometry and arbitrary relabeling of the colors.
    sample = puzzles[0]['train'][0]
    recoloring = np.array([8, 5, 0, 7, 3, 1, 9, 2, 6, 4], dtype=np.int8)
    for k in range(4):
        for transpose in (False, True):
            a = recoloring[_transform(sample['input'], k, transpose)]
            b = recoloring[_transform(sample['output'], k, transpose)]
            _require(np.array_equal(solve(a), b), 'geometry/palette metamorphism failed')
    damaged = {s: [{k: v.copy() for k, v in q.items()} for q in puzzles[0][s]] for s in puzzles[0]}
    q = damaged['test'][0]
    _, m = _decode(q['input'])
    p = tuple(map(int, np.argwhere(m['path'] & (q['input'] == m['roles'][0]))[0]))
    q['output'][p] = m['roles'][0]
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('missing route cell accepted')
    # Moving the source off its finite launcher must not create a signal.
    bad_input = sample['input'].copy()
    _, m = _decode(bad_input)
    bad_input[m['origin']] = m['roles'][0]
    point = next(tuple(map(int, p)) for p in np.argwhere(bad_input == m['roles'][0])
                 if tuple(map(int, p)) not in set(_span_cells(m, m['launch'])))
    bad_input[point] = m['roles'][3]
    try:
        solve(bad_input)
    except ValueError:
        pass
    else:
        raise AssertionError('off-span launch accepted')
    return {'seeds': 42, 'pairs': len(records), 'legal_count_combinations': legal,
            'distinct_shapes': len(shapes), 'normalized_structures': len(normalized),
            'coded_fence_counts': sorted({r['coded_fences'] for r in records}),
            'span_lengths': sorted({n for r in records for n in r['lengths']}),
            'bilateral_pairs': sum(r['bilateral'] for r in records),
            'missed_fences': sum(r['misses'] for r in records),
            'semantic_checks': 'authored generator-owned invariants, not independent certification'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
