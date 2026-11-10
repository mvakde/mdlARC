"""Shared-rail deliveries of reference-defined rigid stencils.

Only NumPy and the standard library are used. Nothing is loaded from parents.
The public solver reconstructs the dictionary and rail from the input alone.
"""
from collections import Counter
import hashlib
import json

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        todo, found = [p], {p}
        for r, c in todo:
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    todo.append(q)
        result.append(found)
    return result


def _footprint(shape, point):
    r, c = point
    return {(r + dr, c + dc) for dr, dc in shape}


def _box_shape(shape):
    rows, cols = zip(*shape)
    return tuple((r, c) for r in range(min(rows), max(rows) + 1)
                 for c in range(min(cols), max(cols) + 1))


def _motion(shape, path, allowed):
    """Local fit at every rail cell, and the prefix reachable before first hit."""
    h, w = allowed.shape
    fits = [all(0 <= r < h and 0 <= c < w and allowed[r, c]
                for r, c in _footprint(shape, p)) for p in path]
    end = next((i - 1 for i, fit in enumerate(fits) if not fit), len(path) - 1)
    return fits, end


def _turn_prefix(path):
    headings = [tuple(np.subtract(q, p)) for p, q in zip(path, path[1:])]
    result, count = [0], 0
    for i, direction in enumerate(headings):
        if i and direction != headings[i - 1]:
            count += 1
        result.append(count)
    return result


def _decode(input_grid):
    a = _grid(input_grid)
    _require(np.count_nonzero(a == 3) == np.count_nonzero(a == 4) == 1,
             "one start (3) and one end (4) are required")
    start = tuple(map(int, np.argwhere(a == 3)[0]))
    end = tuple(map(int, np.argwhere(a == 4)[0]))
    components = _components(a != 0)
    arenas = [v for v in components if start in v and end in v]
    _require(len(arenas) == 1, "start and end must belong to one floor area")
    arena = arenas[0]
    points = np.array(sorted(arena))
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    bounds = tuple(map(int, (*lo, *hi)))
    templates, repeats = {}, Counter()
    for comp in components:
        if comp is arena:
            continue
        _require(len(comp) >= 3 and all(a[p] >= 5 for p in comp),
                 "outside the floor, only complete connected specimens are allowed")
        counts = Counter(int(a[p]) for p in comp)
        singletons = [c for c, n in counts.items() if n == 1]
        _require(len(singletons) == 1, "each specimen needs one singleton key")
        key = singletons[0]
        ar, ac = next(p for p in comp if a[p] == key)
        sig = tuple(sorted((int(r - ar), int(c - ac), int(a[r, c])) for r, c in comp))
        _require(key not in templates or templates[key] == sig,
                 "inconsistent specimens for one key")
        templates[key] = sig
        repeats[key] += 1
    _require(1 <= len(templates) <= 3, "one to three stencil types are supported")
    keys = set(templates)
    _require(all(color not in keys for sig in templates.values()
                 for dr, dc, color in sig if (dr, dc) != (0, 0)),
             "decoration colors cannot also be keys")
    r0, c0, r1, c1 = bounds
    # Reference specimens may not hide inside holes in the floor area.
    _require(all(not (r0 <= r < r1 and c0 <= c < c1)
                 for comp in components if comp is not arena for r, c in comp),
             "specimens must be outside the floor rectangle")
    _require(all(a[p] in {1, 2, 3, 4} | keys for p in arena),
             "unexpected decoration or unknown key on the floor")
    rail = {p for p in arena if a[p] in {2, 3, 4} | keys}
    neighbors = {p: [q for dr, dc in DIRS if (q := (p[0] + dr, p[1] + dc)) in rail]
                 for p in rail}
    _require(all(len(ns) == (1 if p in (start, end) else 2)
                 for p, ns in neighbors.items()), "rail must be a simple unbranched path")
    path, previous, p = [], None, start
    while True:
        _require(p not in path, "rail contains a cycle")
        path.append(p)
        if p == end:
            break
        options = [q for q in neighbors[p] if q != previous]
        _require(len(options) == 1, "ambiguous rail direction")
        previous, p = p, options[0]
    _require(set(path) == rail, "extra disconnected rail cells")
    allowed = np.zeros(a.shape, dtype=bool)
    for p in arena:
        allowed[p] = True
    shapes = {k: tuple((r, c) for r, c, _ in sig) for k, sig in templates.items()}
    states = {k: _motion(shape, path, allowed) for k, shape in shapes.items()}
    boxes = {k: _motion(_box_shape(shape), path, allowed) for k, shape in shapes.items()}
    _require(all(fits[0] for fits, _ in states.values()),
             "every stencil must fit at the visible start")
    requests, used = [], set()
    turn_prefix = _turn_prefix(path)
    out = a.copy()
    for i, p in enumerate(path):
        key = int(a[p])
        if key not in keys:
            continue
        local, last = states[key]
        _require(local[i], "every request must fit locally, including unreachable ones")
        cells = _footprint(shapes[key], p)
        _require(all(q == p or a[q] in (1, 2) for q in cells),
                 "a request footprint would overwrite another key or an endpoint")
        completed = i <= last
        _require(not (cells & used), "request footprints may not overlap")
        used.update(cells)
        if completed:
            for dr, dc, color in templates[key]:
                out[p[0] + dr, p[1] + dc] = color
        requests.append({'key': key, 'index': i, 'completed': completed,
                         'bbox_reachable': i <= boxes[key][1],
                         'turns_before': turn_prefix[i]})
    _require(requests, "rail must contain requests")
    info = {'templates': templates, 'repeats': dict(repeats), 'shapes': shapes,
            'path': path, 'allowed': allowed, 'bounds': bounds, 'states': states,
            'requests': requests, 'turns': turn_prefix[-1],
            'wall_count': int(np.count_nonzero(a[r0:r1, c0:c1] == 0))}
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute deliveries entirely from visible specimens, rail, and walls."""
    return _decode(input_grid)[0]


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _random_shape(rng):
    h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    n = int(rng.integers(3, min(11, h * w) + 1))
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    style = int(rng.integers(3))
    while len(cells) < n:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in DIRS
                           if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
        degree = np.array([sum((r + dr, c + dc) in cells for dr, dc in DIRS)
                           for r, c in frontier], dtype=float)
        weight = np.ones(len(frontier)) if style == 0 else degree ** (2 if style == 1 else -2)
        cells.add(frontier[int(rng.choice(len(frontier), p=weight / weight.sum()))])
    ar, ac = _pick(rng, sorted(cells))
    return tuple(sorted((r - ar, c - ac) for r, c in cells))


def _make_templates(rng, n):
    palette = list(map(int, rng.permutation(np.arange(5, 10))))
    keys, decorations = palette[:n], palette[n:]
    templates = {}
    seen = set()
    for key in keys:
        for _ in range(30):
            shape = _random_shape(rng)
            if shape not in seen and len(_box_shape(shape)) > len(shape):
                break
        else:
            return None
        seen.add(shape)
        offsets = [p for p in shape if p != (0, 0)]
        ncolors = 2 if len(offsets) >= 4 and rng.random() < .45 else 1
        colors = list(map(int, rng.choice(decorations, ncolors, replace=False)))
        assignment = colors * 2 + [_pick(rng, colors) for _ in range(len(offsets) - 2 * ncolors)]
        if ncolors == 1:
            assignment = colors * len(offsets)
        rng.shuffle(assignment)
        templates[key] = tuple(sorted([(0, 0, key)] +
                                     [(r, c, color) for (r, c), color in zip(offsets, assignment)]))
    return templates


def _expand(nodes):
    path = [nodes[0]]
    for p, q in zip(nodes, nodes[1:]):
        dr, dc = int(np.sign(q[0] - p[0])), int(np.sign(q[1] - p[1]))
        while path[-1] != q:
            r, c = path[-1]
            path.append((r + dr, c + dc))
    return path


def _sample_path(rng, h, w, mode):
    if mode == 'straight':
        r = int(rng.integers(3, h - 3))
        return _expand([(r, 3), (r, w - 4)])
    if mode == 'bend':
        return _expand([(3, 3), (3, w - 4), (h - 4, w - 4)])
    rows = [3, (h - 1) // 2, h - 4] if h >= 17 else [3, h - 4]
    cols = [3, (w - 1) // 2, w - 4]
    target = int(rng.integers(6 if mode == 'test' else 3, len(rows) * len(cols) + 1))
    best = []
    for _ in range(20):
        p = int(rng.integers(len(rows))), int(rng.integers(len(cols)))
        nodes, used = [p], {p}
        while len(nodes) < target:
            r, c = nodes[-1]
            options = [(r + dr, c + dc) for dr, dc in DIRS
                       if 0 <= r + dr < len(rows) and 0 <= c + dc < len(cols)
                       and (r + dr, c + dc) not in used]
            if not options:
                break
            p = _pick(rng, options)
            nodes.append(p)
            used.add(p)
        if len(nodes) > len(best):
            best = nodes
    if len(best) < (6 if mode == 'test' else 3):
        return None
    return _expand([(rows[r], cols[c]) for r, c in best])


def _obstacles(rng, shapes, path, allowed):
    """Construct a discriminatory wall and a real-mask/bounding-box contrast."""
    rail = set(path)
    starts = set().union(*(_footprint(s, path[0]) for s in shapes.values()))
    footprints = {k: [_footprint(s, p) for p in path] for k, s in shapes.items()}
    sweeps = {}
    for k, footprints_k in footprints.items():
        sweep, records = set(), []
        for cells in footprints_k:
            sweep = sweep | cells
            records.append(sweep)
        sweeps[k] = records
    choices = []
    for blocked in shapes:
        for passing in shapes:
            if blocked == passing:
                continue
            for i in range(4, len(path) - 7):
                protected = sweeps[passing][i + 4] | starts | rail
                for q in sorted(footprints[blocked][i] - protected):
                    if 0 <= q[0] < allowed.shape[0] and 0 <= q[1] < allowed.shape[1]:
                        choices.append((q, passing, i + 4))
    if not choices:
        return None
    q, passing, protect_to = _pick(rng, choices)
    allowed[q] = False
    protected = sweeps[passing][protect_to] | rail | starts
    # A wall may be a peg or a longer side of a constriction.
    for dr, dc in [DIRS[int(rng.integers(4))]]:
        for distance in range(1, int(rng.integers(1, 4))):
            v = q[0] + distance * dr, q[1] + distance * dc
            if not (0 <= v[0] < allowed.shape[0] and 0 <= v[1] < allowed.shape[1]) or v in protected:
                break
            allowed[v] = False
    choices = []
    for k, shape in shapes.items():
        _, last = _motion(shape, path, allowed)
        for i in range(3, min(last, len(path) - 4) + 1):
            forbidden = sweeps[k][i] | protected
            empty = _footprint(_box_shape(shape), path[i]) - forbidden
            for q in sorted(empty):
                if 0 <= q[0] < allowed.shape[0] and 0 <= q[1] < allowed.shape[1]:
                    choices.append((q, k, i))
    if not choices:
        return None
    q, k, i = _pick(rng, choices)
    allowed[q] = False
    protected |= sweeps[k][i]
    for _ in range(int(rng.integers(0, 7))):
        v = int(rng.integers(allowed.shape[0])), int(rng.integers(allowed.shape[1]))
        if v not in protected:
            allowed[v] = False
    return allowed


def _choose_requests(rng, shapes, path, allowed, required_turns=0):
    states = {k: _motion(s, path, allowed) for k, s in shapes.items()}
    bbox = {k: _motion(_box_shape(s), path, allowed)[1] for k, s in shapes.items()}
    if any(not fits[0] for fits, _ in states.values()):
        return None
    first = min(last for _, last in states.values())
    turn_prefix = _turn_prefix(path)
    pool = []
    for key, shape in shapes.items():
        fits, last = states[key]
        for i in range(2, len(path) - 2):
            cells = _footprint(shape, path[i])
            if not fits[i] or cells & {path[0], path[-1]}:
                continue
            tags = {('type', key)}
            if i > last:
                tags.add('history')
            if first < i <= last:
                tags.add('contrast')
            if i <= last and i > bbox[key]:
                tags.add('bbox')
            if i <= last < len(path) - 1:
                tags.add('before')
            if i <= last and turn_prefix[i] >= required_turns:
                tags.add('bends')
            pool.append((key, i, cells, tags))
    needed = {'history', 'contrast', 'bbox', 'before'} | {('type', k) for k in shapes}
    if required_turns:
        needed.add('bends')
    for _ in range(40):
        uncovered, options, chosen = needed.copy(), pool.copy(), []
        while uncovered and options and len(chosen) < 7:
            scores = [len(v[3] & uncovered) for v in options]
            best = max(scores)
            if best == 0:
                break
            v = _pick(rng, [v for v, score in zip(options, scores) if score == best])
            chosen.append(v)
            uncovered -= v[3]
            # Even suppressed requests have separated masks. This avoids
            # learning an unstated conflict policy from sparse markers.
            options = [x for x in options if not (x[2] & v[2])
                       and path[x[1]] not in v[2] and path[v[1]] not in x[2]]
        if uncovered:
            continue
        extras = int(rng.integers(0, 3))
        while options and extras and len(chosen) < 7:
            v = _pick(rng, options)
            chosen.append(v)
            options = [x for x in options if not (x[2] & v[2])]
            extras -= 1
        # A completed request AFTER an incomplete request of another type
        # rules out a single shared cutoff independent of stencil identity.
        if not any(i <= states[k][1] and j > states[l][1] and i > j and k != l
                   for k, i, _, _ in chosen for l, j, _, _ in chosen):
            continue
        return sorted((k, i) for k, i, _, _ in chosen)
    return None


def _reference_layout(rng, templates, width, height_limit, repeats):
    ids = [k for k in templates for _ in range(repeats)]
    rng.shuffle(ids)
    row, col, row_height, placements = 1, 1, 0, []
    for key in ids:
        shape = [(r, c) for r, c, _ in templates[key]]
        rs, cs = zip(*shape)
        h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        if col + w >= width:
            row += row_height + 1
            col, row_height = 1, 0
        if row + h + 1 > height_limit:
            return None
        placements.append((key, row - min(rs), col - min(cs)))
        col += w + 1
        row_height = max(row_height, h)
    return row + row_height + 1, placements


def _make_pair(rng, mode, ntypes, used):
    # Each retry changes shapes and geometry; no stored vocabulary or parent
    # pair is used. Exhaustion is an explicit error, not a silent rule change.
    for _ in range(240):
        h = int(rng.integers(17, 22)) if mode == 'test' else int(rng.integers(12, 22))
        w = int(rng.integers(22, 29))
        templates = _make_templates(rng, ntypes)
        if templates is None:
            continue
        layout = _reference_layout(rng, templates, w, 29 - h, 2 if mode != 'test' else 1)
        if layout is None:
            continue
        path = _sample_path(rng, h, w, mode)
        if path is None:
            continue
        shapes = {k: tuple((r, c) for r, c, _ in sig) for k, sig in templates.items()}
        allowed = _obstacles(rng, shapes, path, np.ones((h, w), dtype=bool))
        if allowed is None:
            continue
        if len(_components(allowed)) != 1:
            continue
        required_turns = 2 if mode == 'test' else 1 if mode == 'bend' else 0
        requests = _choose_requests(rng, shapes, path, allowed, required_turns)
        if requests is None:
            continue
        ref_h, placements = layout
        top = ref_h + 1
        a = np.zeros((top + h, w), dtype=np.int8)
        for key, ar, ac in placements:
            for dr, dc, color in templates[key]:
                a[ar + dr, ac + dc] = color
        arena = a[top:]
        arena[allowed] = 1
        for r, c in path:
            arena[r, c] = 2
        arena[path[0]], arena[path[-1]] = 3, 4
        for key, i in requests:
            arena[path[i]] = key
        # Actual occupied-cell stamping computes the label at construction
        # time; the public parser independently recovers these items from cells.
        out = a.copy()
        for key, i in requests:
            _, last = _motion(shapes[key], path, allowed)
            if i <= last:
                r, c = path[i]
                for dr, dc, color in templates[key]:
                    out[top + r + dr, c + dc] = color
        # Whole-scene orientation changes reference offsets and rail together.
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        a, out = np.rot90(a, turns), np.rot90(out, turns)
        if flip:
            a, out = np.fliplr(a), np.fliplr(out)
        pad = [(0, 0), (0, 0)]
        for axis in (0, 1):
            extra = int(rng.integers(0, min(2, 30 - a.shape[axis]) + 1))
            before = int(rng.integers(extra + 1))
            pad[axis] = before, extra - before
        a, out = np.pad(a, pad), np.pad(out, pad)
        signature = (a.shape, a.tobytes())
        if signature in used:
            continue
        predicted, info = _decode(a)
        _require(np.array_equal(out, predicted), "construction disagrees with input reconstruction")
        if mode == 'test' and info['turns'] < 2:
            continue
        used.add(signature)
        return {'input': a.copy(), 'output': out.copy()}
    raise RuntimeError("bounded shape/rail construction exhausted")


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10),
                                    (num_test, 'num_test', 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    result, used = {'train': [], 'test': []}, set()
    modes = ['straight', 'bend'] + [_pick(rng, ['straight', 'bend', 'walk'])
                                   for _ in range(num_train - 2)]
    rng.shuffle(modes)
    for mode in modes:
        result['train'].append(_make_pair(rng, mode, 2, used))
    for _ in range(num_test):
        result['test'].append(_make_pair(rng, 'test', 3, used))
    validate_generated_puzzle(result)
    return result


def _evidence(info):
    requests = info['requests']
    return {
        'history': any(not r['completed'] for r in requests),
        'contrast': any(a['completed'] and not b['completed'] and a['index'] > b['index']
                        and a['key'] != b['key'] for a in requests for b in requests),
        'bbox': any(r['completed'] and not r['bbox_reachable'] for r in requests),
        'before': any(r['completed'] and info['states'][r['key']][1] < len(info['path']) - 1
                      for r in requests),
    }


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    try:
        _counts(len(puzzle['train']), len(puzzle['test']))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "scene dimensions must be preserved")
            signature = a.shape, a.tobytes()
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            expected, info = _decode(a)
            _require(np.array_equal(expected, b), "wrong deliveries or altered scenery")
            _require(all(_evidence(info).values()), "missing transport/template contrast")
            _require(set(r['key'] for r in info['requests']) == set(info['templates']),
                     "generated scenes request every visible type")
            changed = a != b
            _require(np.any(changed) and np.all(np.isin(a[changed], [1, 2])),
                     "only floor or ordinary rail cells may change")
            # Explicit preservation checks make errors in scenery or sparse
            # unreachable requests fail even if they look like a valid motif.
            _require(np.array_equal(a[np.isin(a, [0, 3, 4])], b[np.isin(a, [0, 3, 4])]),
                     "walls and endpoint markers changed")
            for r in info['requests']:
                _require(b[info['path'][r['index']]] == r['key'], "request key changed")
            infos[split].append(info)
    _require(all(len(v['templates']) == 2 and min(v['repeats'].values()) >= 2
                 for v in infos['train']), "training needs repeated two-key dictionaries")
    _require(all(len(v['templates']) == 3 and v['turns'] >= 2 for v in infos['test']),
             "test must combine three types and a winding rail")
    _require(all(any(r['completed'] and r['turns_before'] >= 2 for r in v['requests'])
                 for v in infos['test']), "tests must deliver a stencil after two bends")
    _require(any(v['turns'] == 0 for v in infos['train']), "training needs a straight rail")
    _require(any(v['turns'] >= 1 for v in infos['train']), "training must expose fixed orientation at bends")
    _require(any(any(r['completed'] and r['turns_before'] >= 1 for r in v['requests'])
                 for v in infos['train']), "training needs an actual delivery after a bend")


def _fingerprint(puzzle):
    return hashlib.sha256(b''.join(np.asarray(p[k], dtype=np.int8).tobytes()
                                  + bytes(np.asarray(p[k]).shape)
                                  for s in ('train', 'test') for p in puzzle[s]
                                  for k in ('input', 'output'))).hexdigest()


def self_test():
    """Bounded 42-seed coverage plus counterfactual and mutation regressions."""
    # A concrete scene with an explicitly written answer checks both parser
    # and semantics without obtaining its expected grid from generation.
    demo = np.zeros((18, 21), dtype=np.int8)
    sigs = {5: ((0, 0, 5), (-1, 0, 7), (0, 1, 7)),
            6: ((0, 0, 6), (-2, 0, 8), (-1, 0, 8))}
    for key, ar, ac in ((5, 3, 2), (6, 3, 8)):
        for dr, dc, color in sigs[key]:
            demo[ar + dr, ac + dc] = color
    demo[6:] = 1
    demo[11, 2:19] = 2
    demo[11, 2], demo[11, 18] = 3, 4
    demo[9, 7], demo[10, 13] = 0, 0
    for key, col in ((6, 4), (6, 9), (5, 12), (5, 16)):
        demo[11, col] = key
    gold = demo.copy()
    gold[9, 4] = gold[10, 4] = 8
    gold[10, 12] = gold[11, 13] = 7
    _require(np.array_equal(solve(demo), gold), "explicit two-stencil delivery regression failed")
    for turns in range(4):
        for flip in (False, True):
            x, y = np.rot90(demo, turns), np.rot90(gold, turns)
            if flip:
                x, y = np.fliplr(x), np.fliplr(y)
            _require(np.array_equal(solve(x), y), "whole-scene orientation regression failed")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fingerprints, geometries, profiles, shapes, headings, palettes = set(), set(), set(), set(), set(), set()
    counts, turns, sizes, decorations, anchors = set(), set(), set(), set(), set()
    for puzzle in puzzles:
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                original = a.copy()
                expected, info = _decode(a)
                _require(np.array_equal(expected, b) and np.array_equal(a, original), "solver mutation or mismatch")
                shapes.add(a.shape)
                counts.add(len(info['requests']))
                turns.add(info['turns'])
                headings.add(tuple(np.subtract(info['path'][1], info['path'][0])))
                palettes.add(tuple(sorted(info['templates'])))
                profiles.add(tuple((k, last, len(info['path'])) for k, (_, last) in info['states'].items()))
                for key, sig in info['templates'].items():
                    shape = info['shapes'][key]
                    geometries.add(shape)
                    sizes.add(len(shape))
                    decorations.add(len({color for r, c, color in sig if (r, c) != (0, 0)}))
                    rows, cols = zip(*shape)
                    anchors.add((min(rows), max(rows), min(cols), max(cols)))
                # Omitting A: blindly stamp every locally fitting request.
                blind = a.copy()
                for request in info['requests']:
                    p = info['path'][request['index']]
                    for dr, dc, color in info['templates'][request['key']]:
                        blind[p[0] + dr, p[1] + dc] = color
                _require(not np.array_equal(blind, b), "ignoring movement gave the same answer")
                # Omitting B's mask: use an anchor point as the moving object.
                # Every point can follow this rail, so this again delivers all
                # requests. A bounding rectangle also demonstrably fails.
                _require(any(r['completed'] and not r['bbox_reachable'] for r in info['requests']),
                         "bounding-box shortcut not distinguished")
                # A stencil permutation changes at least one delivery's
                # reachability or its colored occupied offsets.
                keys = sorted(info['templates'])
                other = keys[1:] + keys[:1]
                differs = False
                for request in info['requests']:
                    key = request['key']
                    replacement = other[keys.index(key)]
                    i = request['index']
                    _, last = info['states'][replacement]
                    if (i <= last) != request['completed'] or (
                        request['completed'] and info['shapes'][replacement] != info['shapes'][key]):
                        differs = True
                _require(differs, "dictionary selection did not affect the answer")
    _require(len(fingerprints) == 42, "whole-puzzle repetition")
    _require(len(geometries) >= 90 and len(profiles) >= 100, "structural variation collapsed")
    _require(len(shapes) >= 35 and len(anchors) >= 30, "geometry/anchor coverage collapsed")
    _require({3, 4, 5, 6, 7, 8} <= sizes, "small and larger masks missing")
    _require(decorations == {1, 2}, "multicolor offsets missing")
    _require(len(counts) >= 3 and max(turns) >= 4, "request-count/rail variation missing")
    _require(headings == set(DIRS) and len(palettes) >= 12, "orientation/palette variation missing")
    _require(_fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0)), "nondeterminism")
    # Invalid outputs and input definitions must be rejected.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
           for s in ('train', 'test')}
    wall = tuple(np.argwhere(bad['test'][0]['input'] == 0)[0])
    bad['test'][0]['output'][wall] = 1
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("altered wall label accepted")
    # Conflicting repeated definitions are an input error, not a license to
    # choose one specimen by spatial order.
    conflict = demo.copy()
    for dr, dc, color in sigs[5]:
        conflict[3 + dr, 13 + dc] = color
    conflict[2, 13] = 0
    conflict[4, 13] = 7
    try:
        solve(conflict)
    except AssertionError:
        pass
    else:
        raise AssertionError("conflicting specimen definitions accepted")
    # The hand-written L has an empty leading corner. A peg there does not
    # block its current docking placement, but blocks the next translation.
    allowed = np.ones((5, 10), dtype=bool)
    path = [(2, c) for c in range(1, 9)]
    allowed[1, 5] = False
    shape = ((0, 0), (-1, 0), (0, 1))
    fits, last = _motion(shape, path, allowed)
    _, boxed_last = _motion(_box_shape(shape), path, allowed)
    _require(last == 3 and boxed_last == 2 and fits[6],
             "explicit empty-corner/first-collision regression failed")
    for a, b in ((2, 1), (10, 4), (10, 1)):
        validate_generated_puzzle(generate_puzzle(73, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return {'seeds': 42, 'distinct_puzzles': len(fingerprints),
            'anchor_relative_masks': len(geometries), 'motion_profiles': len(profiles),
            'grid_shapes': len(shapes), 'request_counts': sorted(counts),
            'rail_turns': sorted(turns), 'motif_sizes': sorted(sizes)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
