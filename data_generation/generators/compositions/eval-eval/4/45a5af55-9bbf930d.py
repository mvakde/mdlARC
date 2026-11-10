"""Courier loom: matching rails deliver colors to boundary-distance addresses.

No files, parent generators, seed tables, or generation state enter solve().
All construction retries and self tests are bounded and retain no grid archive.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0,353103945,507718913,365005106,372659159,410648054,319977874,
              787398502,748622808,922262783,554558034,439266348,535812737,
              950548332,823758271,286035455,205265761,476271664,6113455,
              343120911,273753673,693554931,958084184,322895304,723606417,
              303602016,279637879,743182982,403330881,717079362,548394502,
              320909551,280904445,577570138,999379659,949652063,703138281,
              264994703,111056981,920531416,166634322,154356086)
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= s <= 30 for s in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "grid colors")
    return g


def _plus(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _depth(shape, p):
    return min(p[0], p[1], shape[0] - 1 - p[0], shape[1] - 1 - p[1])


def _normalise(inp):
    """The multicolor edge ribbon and clear approach identify all roles."""
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        h, w = g.shape
        if w < 6 or h < 6 or h % 2:
            continue
        bg = int(g[0, 1])
        if not np.all(g[:, 1] == bg):
            continue
        payload = set(map(int, g[:, 0]))
        interior = set(map(int, np.unique(g[:, 1:])))
        if 2 <= len(payload) <= 4 and not payload.intersection(interior):
            candidates.append((g, k, bg, payload, interior - {bg}))
    _require(len(candidates) == 1, "one identifiable courier ribbon required")
    g, k, bg, payload, guides = candidates[0]
    # A partly packed scene can temporarily contain just one rail identity.
    _require(len(guides) >= 1, "rail identity required")
    return dict(grid=g, rotation=k, background=bg, payload=payload, guides=guides)


def _paired(g, p, direction, guides, ignore_equality=False):
    n = (-direction[1], direction[0])
    a, b = _plus(p, n), _plus(p, (-n[0], -n[1]))
    if not (_inside(g, a) and _inside(g, b)):
        return False
    x, y = int(g[a]), int(g[b])
    return x in guides and y in guides and (x == y or ignore_equality)


def _trace(g, row, bg, guides, ignore_equality=False):
    """Inertia in free space; only a blocked ray can take a paired-rail turn."""
    p, direction = (row, 1), (0, 1)
    path, directions, states, turns = [p], [], set(), 0
    while True:
        _require((p, direction) not in states, "cyclic route")
        states.add((p, direction))
        q = _plus(p, direction)
        if not _inside(g, q):
            break
        if int(g[q]) == bg:
            path.append(q)
            directions.append(direction)
            p = q
            continue
        choices = []
        for d in ((-direction[1], direction[0]), (direction[1], -direction[0])):
            q = _plus(p, d)
            if (_inside(g, q) and int(g[q]) == bg
                    and _paired(g, q, d, guides, ignore_equality)):
                choices.append(d)
        _require(len(choices) <= 1, "ambiguous turn")
        if not choices:
            break
        direction = choices[0]
        turns += 1
    _require(len(path) == len(set(path)), "route revisits a cell")
    support = [_paired(g, q, d, guides) for q, d in zip(path[1:], directions)]
    gaps = 0
    # Look within maximal straight runs, excluding the naturally bare corners.
    start = 0
    while start < len(directions):
        end = start + 1
        while end < len(directions) and directions[end] == directions[start]:
            end += 1
        yes = [i for i in range(start, end) if support[i]]
        if yes:
            gaps += sum(not support[i] for i in range(yes[0], yes[-1] + 1))
        start = end
    return dict(row=row, destination=p, depth=_depth(g.shape, p), path=path,
                directions=directions, turns=turns, gaps=gaps,
                open=p[0] in (0, g.shape[0] - 1) or p[1] == g.shape[1] - 1,
                color=int(g[row, 0]))


def _analyse(inp):
    a = _normalise(_grid(inp))
    g, bg, guides = a['grid'], a['background'], a['guides']
    routes, mismatches = [], []
    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1]):
            if int(g[r, c]) != bg:
                break
            x, y = int(g[r - 1, c]), int(g[r + 1, c])
            if x in guides and y in guides:
                if x == y:
                    route = _trace(g, r, bg, guides)
                    route['rail'] = x
                    routes.append(route)
                else:
                    mismatches.append((r, c))
                break
    _require(routes, "no active couriers")
    _require(len({r['depth'] for r in routes}) == len(routes), "duplicate frame address")
    cells = [p for r in routes for p in r['path']]
    _require(len(cells) == len(set(cells)), "courier paths must be disjoint")
    a.update(routes=routes, mismatches=mismatches)
    return a


def _frames(side, bg, deliveries):
    code = np.full(side // 2, bg, dtype=np.int8)
    for depth, color in deliveries:
        _require(0 <= depth < len(code), "frame address out of range")
        code[depth] = color
    edge = np.minimum(np.arange(side), np.arange(side)[::-1])
    return code[np.minimum(edge[:, None], edge[None, :])].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return a square loom oriented with the input (frames are D4 symmetric)."""
    a = _analyse(input_grid)
    return _frames(a['grid'].shape[0], a['background'],
                   [(r['depth'], r['color']) for r in a['routes']])


def _polyline(points):
    path = [points[0]]
    for end in points[1:]:
        p = path[-1]
        if (p[0] == end[0]) == (p[1] == end[1]):
            return None
        d = (int(np.sign(end[0] - p[0])), int(np.sign(end[1] - p[1])))
        while p != end:
            p = _plus(p, d)
            path.append(p)
    cells = set(path)
    if len(cells) != len(path):
        return None
    for i, p in enumerate(path):
        degree = sum(_plus(p, d) in cells for d in DIRS)
        if degree != (1 if i in (0, len(path) - 1) else 2):
            return None
    return path


def _random_path(rng, h, w, depth, bends):
    targets = [(r, c) for r in range(h) for c in range(4, w)
               if _depth((h, w), (r, c)) == depth]
    if not targets:
        return None
    er, ec = targets[int(rng.integers(len(targets)))]
    sr = int(rng.integers(1, h - 1)) if bends else er
    if not 1 <= sr < h - 1:
        return None
    c1, c2 = map(int, rng.integers(5, w - 1, size=2))
    r2 = int(rng.integers(1, h - 1))
    recipes = (
        [(sr, 1), (sr, ec)],
        [(sr, 1), (sr, ec), (er, ec)],
        [(sr, 1), (sr, c1), (er, c1), (er, ec)],
        [(sr, 1), (sr, c1), (r2, c1), (r2, ec), (er, ec)],
        [(sr, 1), (sr, c1), (r2, c1), (r2, c2), (er, c2), (er, ec)],
    )
    points = recipes[bends]
    if points[1][1] < 5:
        return None
    for i, (p, q) in enumerate(zip(points, points[1:])):
        length = abs(p[0] - q[0]) + abs(p[1] - q[1])
        if length < (1 if i == len(points) - 2 and depth == 0 else 2):
            return None
    # Only the final point may touch a non-entry edge.
    if any(r in (0, h - 1) or c in (0, w - 1) for r, c in points[1:-1]):
        return None
    return _polyline(points)


def _walls(rng, path, h, w, color, guides, inactive=False):
    cells, walls = set(path), {}
    margin = int(rng.integers(2, 4))
    for p in path:
        for d in DIRS:
            q = _plus(p, d)
            if 0 <= q[0] < h and margin <= q[1] < w and q not in cells:
                walls[q] = color
    directions = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(path, path[1:])]
    if inactive:
        other = int(rng.choice([c for c in guides if c != color]))
        for q in walls:
            if q[0] > path[0][0]:
                walls[q] = other
        return walls
    # Remove only flanks well inside a straight run. Turns retain evidence.
    eligible = []
    for i in range(3, len(path) - 3):
        if len(set(directions[i - 2:i + 2])) != 1 or path[i][1] < margin + 2:
            continue
        d = directions[i]
        n = (-d[1], d[0])
        pair = (_plus(path[i], n), _plus(path[i], (-n[0], -n[1])))
        if all(q in walls for q in pair):
            eligible.append((i, pair))
    if eligible:
        start = eligible[int(rng.integers(len(eligible)))][0]
        length = int(rng.integers(1, 4))
        for i, pair in eligible:
            if start <= i < start + length:
                for q in pair:
                    walls.pop(q, None)
    last = directions[-1]
    cap = _plus(path[-1], last)
    if cap in walls:
        walls[cap] = int(rng.choice(guides))
    # An unsupported tail is independent of the cap's identity.
    for i in range(len(path) - 1, max(2, len(path) - 1 - int(rng.integers(4))), -1):
        if len(set(directions[max(0, i - 2):i])) != 1:
            break
        n = (-last[1], last[0])
        for q in (_plus(path[i], n), _plus(path[i], (-n[0], -n[1]))):
            walls.pop(q, None)
    return walls


def _put(g, path, walls, reserved, intended, bg, active):
    if path is None or any(p in reserved or int(g[p]) != bg for p in path):
        return None
    if any(q in reserved or (int(g[q]) != bg and int(g[q]) != c)
           for q, c in walls.items()):
        return None
    candidate = g.copy()
    for q, c in walls.items():
        candidate[q] = c
    wanted = dict(intended)
    if active:
        wanted[path[0][0]] = path
    try:
        a = _analyse(candidate)
        found = {r['row']: r['path'] for r in a['routes']}
        if found != wanted:
            return None
    except ValueError:
        return None
    return candidate


def _mismatch_witness(a):
    addresses = {r['depth'] for r in a['routes']}
    for row, _ in a['mismatches']:
        try:
            r = _trace(a['grid'], row, a['background'], a['guides'], True)
        except ValueError:
            continue
        if r['depth'] not in addresses:
            return r
    return None


def _features(a):
    routes = a['routes']
    depths = [r['depth'] for r in routes]
    def straight_depth(route):
        p = (route['row'], 1)
        while (_inside(a['grid'], _plus(p, (0, 1)))
               and int(a['grid'][_plus(p, (0, 1))]) == a['background']):
            p = _plus(p, (0, 1))
        return _depth(a['grid'].shape, p)
    return dict(turn=any(r['turns'] for r in routes),
                turn_matters=any(straight_depth(r) != r['depth'] for r in routes),
                address_gap=max(depths) + 1 > len(depths),
                gap=any(r['gaps'] for r in routes),
                open=any(r['open'] for r in routes),
                closed=any(not r['open'] for r in routes),
                scrambled=depths != sorted(depths) and depths != sorted(depths, reverse=True))


def _fallback(rng, h, w, bg, payload, guides, count, deep):
    """Finite spare construction with an elbow, gaps and a genuine decoy."""
    if deep is None:
        rows = {2: [1, h // 2], 3: [1, 5, h - 5], 4: [1, 5, 8, h - 5]}[count]
        depths = {2: [0, 4], 3: [0, 4, 2], 4: [0, 4, 2, 3]}[count]
    else:
        rows = [1, h // 2, h - 6] if count == 3 else [1, 5, h // 2, h - 6]
        depths = [0, deep, 2] if count == 3 else [0, 3, deep, 2]
    g = np.full((h, w), bg, dtype=np.int8)
    g[:, 0] = rng.choice(payload, h)
    intended, reserved = {}, set()
    for i, (row, depth) in enumerate(zip(rows, depths)):
        if depth == 0:
            col = int(rng.integers(5, w - 3))
            path = _polyline([(row, 1), (row, col), (0, col)])
        else:
            path = _polyline([(row, 1), (row, w - 1 - depth)])
        walls = _walls(rng, path, h, w, guides[i % len(guides)], guides)
        candidate = _put(g, path, walls, reserved, intended, bg, True)
        _require(candidate is not None, "fallback active route did not fit")
        g, reserved = candidate, reserved | set(path)
        intended[row] = path
    row = h - 2
    path = _polyline([(row, 1), (row, w - 2)])
    walls = _walls(rng, path, h, w, guides[0], guides, True)
    candidate = _put(g, path, walls, reserved, intended, bg, False)
    _require(candidate is not None, "fallback inactive route did not fit")
    return candidate


def _scene(rng, h, w, bg, payload, guides, count, deep, scrambled):
    for attempt in range(7):
        small = list(map(int, rng.choice(np.arange(1, 5), count - 1 - int(deep is not None), replace=False)))
        depths = [0] + small + ([] if deep is None else [deep])
        rng.shuffle(depths)
        g = np.full((h, w), bg, dtype=np.int8)
        g[:, 0] = rng.choice(payload, h)
        intended, reserved = {}, set()
        for i, depth in enumerate(depths):
            for _ in range(65):
                bends = int(rng.choice(5, p=[.22, .30, .26, .15, .07]))
                path = _random_path(rng, h, w, depth, bends)
                if path is None:
                    continue
                walls = _walls(rng, path, h, w, guides[i % len(guides)], guides)
                candidate = _put(g, path, walls, reserved, intended, bg, True)
                if candidate is not None:
                    g, reserved = candidate, reserved | set(path)
                    intended[path[0][0]] = path
                    break
            else:
                break
        if len(intended) != count:
            continue
        # Add one or two geometrically plausible, mismatched straight routes.
        for _ in range(2):
            for row in map(int, rng.permutation(np.arange(1, h - 1))):
                end = int(rng.integers(6, w))
                path = _polyline([(row, 1), (row, end)])
                walls = _walls(rng, path, h, w, int(rng.choice(guides)), guides, True)
                candidate = _put(g, path, walls, reserved, intended, bg, False)
                if candidate is not None:
                    g, reserved = candidate, reserved | set(path)
                    break
        a = _analyse(g)
        f = _features(a)
        if (all(f[k] for k in ('turn', 'turn_matters', 'gap', 'open', 'closed'))
                and (not scrambled or (f['scrambled'] and f['address_gap']))
                and _mismatch_witness(a) is not None):
            return g, False
    return _fallback(rng, h, w, bg, payload, guides, count, deep), True


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    _require(low <= int(value) <= high, name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 'num_train', 2, 10)
    m = _count(num_test, 'num_test', 1, 4)
    _require(n > m, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    payload_count = int(rng.integers(2, 5))
    guide_count = int(rng.integers(2, min(5, 10 - payload_count)))
    mixed = bool(rng.integers(2))
    base_rotation = int(rng.integers(4))
    rotate_pairs = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    first_height = int(rng.choice([16, 18, 20, 22]))
    for split, size in (('train', n), ('test', m)):
        for index in range(size):
            for retry in range(20):
                colors = list(map(int, rng.permutation(palette))) if mixed else palette
                bg = colors[0]
                payload = colors[1:1 + payload_count]
                guides = colors[1 + payload_count:1 + payload_count + guide_count]
                h = int(rng.choice([24, 26, 28, 30])) if split == 'test' else int(rng.choice([16, 18, 20, 22]))
                if split == 'train' and index == 0:
                    h = first_height
                if split == 'train' and index == 1:
                    h = int(rng.choice([v for v in [16, 18, 20, 22] if v != first_height]))
                w = int(rng.integers(24, 31)) if split == 'test' else int(rng.integers(16, 27))
                count = (int(rng.integers(3, 5)) if split == 'test' else
                         3 if index == 0 else 2 if index == 1 else int(rng.integers(2, 5)))
                deep = int(rng.integers(5, min(h, w) // 2 - 1)) if split == 'test' else None
                g, _ = _scene(rng, h, w, bg, payload, guides, count, deep,
                              split == 'train' and index == 0)
                a = _analyse(g)
                # The moving payloads use at least two colors, with no color command.
                order = list(map(int, rng.permutation(payload)))
                for i, route in enumerate(a['routes']):
                    g[route['row'], 0] = order[i % len(order)]
                # Keep the whole payload alphabet visibly confined to the ribbon.
                free_rows = [r for r in range(h) if r not in {q['row'] for q in a['routes']}]
                for row, color in zip(free_rows, payload):
                    g[row, 0] = color
                k = (base_rotation + index) % 4 if rotate_pairs else base_rotation
                inp = np.rot90(g, k).copy()
                signature = (inp.shape, inp.tobytes())
                if signature in seen:
                    continue
                seen.add(signature)
                # The destinations were checked against constructed paths by _put.
                deliveries = [(r['depth'], int(g[r['row'], 0])) for r in a['routes']]
                puzzle[split].append(dict(input=inp, output=_frames(h, bg, deliveries)))
                break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _wrapped(side, bg, routes):
    """Alternate frame construction: grow from the 2x2 center outward."""
    code = {r['depth']: r['color'] for r in routes}
    out = np.full((2, 2), code.get(side // 2 - 1, bg), dtype=np.int8)
    for depth in range(side // 2 - 2, -1, -1):
        out = np.pad(out, 1, constant_values=code.get(depth, bg))
    return out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _count(n, 'num_train', 2, 10)
    _count(m, 'num_test', 1, 4)
    _require(n > m, "pair counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = inp.shape, inp.astype(np.int8).tobytes()
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyse(inp)
            g, bg, routes = a['grid'], a['background'], a['routes']
            h = g.shape[0]
            _require(out.shape == (h, h), "output size must use ribbon length")
            _require(np.array_equal(out, _wrapped(h, bg, routes)), "wrong addressed frames")
            _require(set(map(int, np.unique(out))) == {bg} | {r['color'] for r in routes}, "output palette")
            _require(np.array_equal(out, out.T) and np.array_equal(out, out[::-1]), "frame symmetry")
            _require(len({r['color'] for r in routes}) >= 2, "distinct moving payloads")
            _require(len({r['rail'] for r in routes}) >= 2, "distinct active rail identities")
            f = _features(a)
            _require(all(f[k] for k in ('turn', 'turn_matters', 'gap', 'open', 'closed')), "route evidence")
            witness = _mismatch_witness(a)
            _require(witness is not None, "equality must change an unaddressed frame")
            wrong_b = _frames(h, bg, [(r['depth'], r['color']) for r in routes] +
                              [(witness['depth'], witness['color'])])
            _require(not np.array_equal(out, wrong_b), "parent B ablation")
            dots = np.full_like(g, bg)
            for route in routes:
                dots[route['destination']] = route['color']
            _require(dots.shape != out.shape or not np.array_equal(dots, out), "parent A ablation")
            if split == 'train':
                training.append(a)
            else:
                _require(max(r['depth'] for r in routes) >
                         max(r['depth'] for t in training for r in t['routes']), "deeper test address required")
                _require(h > max(t['grid'].shape[0] for t in training), "larger test loom required")
    _require(any(_features(t)['scrambled'] for t in training), "scrambled delivery order required")
    _require(any(_features(t)['address_gap'] for t in training), "blank intermediate address required")
    _require(len({t['grid'].shape[0] for t in training}) >= 2, "training sizes must vary")
    _require(len({len(t['routes']) for t in training}) >= 2, "training counts must vary")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """42 specified seeds, all counts, metamorphic checks; no stored payloads."""
    signatures, shapes, structures, depths, bends, colors, rotations, geometry = (set() for _ in range(8))
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for pair in puzzle['train'] + puzzle['test']:
            inp, out = pair['input'], pair['output']
            a = _analyse(inp)
            shapes.add(inp.shape)
            rotations.add(a['rotation'])
            colors.update(map(int, np.unique(inp)))
            roles = {}
            normalized = bytes(roles.setdefault(int(c), len(roles)) for c in a['grid'].flat)
            structures.add(hashlib.sha256(normalized).digest())
            description = (a['grid'].shape, tuple(tuple(r['path']) for r in a['routes']),
                           tuple(a['mismatches']))
            geometry.add(hashlib.sha256(repr(description).encode()).digest())
            depths.update(r['depth'] for r in a['routes'])
            bends.update(r['turns'] for r in a['routes'])
            _require(np.array_equal(solve(inp), out), "input-only prediction")
        # Rotation and literal color changes preserve meaning without labels.
        pair = puzzle['test'][0]
        perm = np.random.default_rng(seed + 17).permutation(10)
        _require(np.array_equal(solve(np.rot90(pair['input'])), pair['output']), "rotation equivariance")
        _require(np.array_equal(solve(perm[pair['input']]), perm[pair['output']]), "palette equivariance")
    _require(len(signatures) == 42 and len(structures) >= 160, "structural diversity")
    _require(len(geometry) >= 160, "palette-independent route geometry")
    _require(len(shapes) >= 25 and len(depths) >= 8, "geometry/address diversity")
    _require({0, 1, 2} <= bends and len(rotations) == 4 and len(colors) == 10, "route/role diversity")
    _require(_digest(generate_puzzle(17, 7, 3)) == _digest(generate_puzzle(17, 7, 3)), "determinism")
    # A center-address fixture goes beyond the sampler's shallower-depth ceiling.
    g = np.zeros((16, 18), dtype=np.int8)
    g[:, 0] = np.resize(np.array([1, 2], dtype=np.int8), 16)
    fixture_rng = np.random.default_rng(47)
    for row, end, color in ((1, 17, 3), (7, 7, 4)):
        path = _polyline([(row, 1), (row, end)])
        for q, c in _walls(fixture_rng, path, 16, 18, color, [3, 4]).items():
            g[q] = c
    g[1, 0], g[7, 0] = 1, 2
    center = solve(g.tolist())
    _require(center.shape == (16, 16) and np.all(center[7:9, 7:9] == 2), "deepest center address")
    _require(np.all(center[0] == 1) and np.all(center[1:7, 1:15] == 0), "sparse center fixture")
    for h, w, row, end, depth in ((6, 10, 4, 7, 1), (10, 6, 4, 3, 2)):
        small = np.zeros((h, w), dtype=np.int8)
        small[:, 0] = np.resize(np.array([1, 2], dtype=np.int8), h)
        for y, x, color in ((1, w - 1, 3), (row, end, 4)):
            path = _polyline([(y, 1), (y, x)])
            for q, c in _walls(fixture_rng, path, h, w, color, [3, 4]).items():
                small[q] = c
        small[1, 0], small[row, 0] = 1, 2
        _require(np.array_equal(solve(small), _frames(h, 0, [(0, 1), (depth, 2)])),
                 "minimum ribbon/width fixture")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            puzzle = generate_puzzle(100 + 10 * n + m, n, m)
            _require(len(puzzle['train']) == n and len(puzzle['test']) == m, "explicit counts")
    for n, m in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(3.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0, 2, 1)
    puzzle['test'][0]['output'][0, 0] = (int(puzzle['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted frame accepted")
