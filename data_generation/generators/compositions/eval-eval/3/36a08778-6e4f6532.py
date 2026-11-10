"""Wake-on-contact baffles: compass docking coupled to causal ray assembly.

Only Python's standard library and NumPy are used. No parent files, stored
examples, generation state, or random choices are read by solve(). Diagnostics
are in memory; this module never writes files or saved execution traces.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ADJ8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "grid colors must be integers 0..9")
    return a.astype(np.int8, copy=True)


def _components(points, diagonal=False):
    remaining, result = set(points), []
    steps = ADJ8 if diagonal else DIRS
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _transform(cells, t):
    transformed = {}
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        transformed[r, c] = v
    r0 = min(r for r, _ in transformed)
    c0 = min(c for _, c in transformed)
    return {(r-r0, c-c0): v for (r, c), v in transformed.items()}


def _shift(cells, delta):
    return {(r+delta[0], c+delta[1]): v for (r, c), v in cells.items()}


def _shape_key(points):
    return min(tuple(sorted(_transform({p: 1 for p in points}, t))) for t in range(8))


def _halo(points):
    return {(r+dr, c+dc) for r, c in points for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _parse(input_grid):
    actual = _grid(input_grid)
    h, w = actual.shape
    _require(h >= 8 and w >= 8, "field too small for framed baffles")
    bg = int(actual[0, 0])
    _require(all(int(actual[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))), "corners must identify background")
    interpretations = []
    for rotation in range(4):
        a = np.rot90(actual, rotation).copy()
        h, w = a.shape
        edges = (a[0, 1:-1], a[1:-1, -1], a[-1, 1:-1], a[1:-1, 0])
        bands = []
        for edge in edges:
            values, counts = np.unique(edge, return_counts=True)
            if np.max(counts) <= len(edge)/2:
                break
            bands.append(int(values[np.argmax(counts)]))
        if len(bands) != 4 or len(set(bands) | {bg}) != 5:
            continue
        if any(np.any(edge != color) for edge, color in zip(edges[1:], bands[1:])):
            continue
        guide_values = set(map(int, edges[0])) - {bands[0]}
        if len(guide_values) != 1:
            continue
        guide = guide_values.pop()
        if guide in set(bands) | {bg}:
            continue
        entries = list(map(int, np.flatnonzero(a[0] == guide)))
        if not entries or any(b-a < 2 for a, b in zip(entries, entries[1:])):
            continue
        expected = set()
        for c in entries:
            r = 0
            while r < h and a[r, c] == guide:
                expected.add((r, c))
                r += 1
            if r < 2 or r >= h-1:
                break
        else:
            if expected == set(zip(*np.nonzero(a == guide))):
                interpretations.append((a, rotation, bg, guide, tuple(bands), entries))
    _require(len(interpretations) == 1, "entry direction must be uniquely visible")
    a, rotation, bg, guide, bands, entries = interpretations[0]
    interior = {(r, c) for r in range(1, a.shape[0]-1) for c in range(1, a.shape[1]-1)
                if int(a[r, c]) not in (bg, guide)}
    sources, targets, target_colors = [], [], set()
    for part in _components(interior, diagonal=True):
        colors = {int(a[p]) for p in part}
        _require(len(_components(part)) == 1, "each object must be four-connected")
        if len(colors) == 1:
            targets.append(part)
            target_colors |= colors
        else:
            sources.append({p: int(a[p]) for p in part})
    _require(sources and len(sources) == len(targets) and len(target_colors) == 1, "need equally many sources and sockets")
    reg = next(iter(target_colors))
    body_colors = set().union(*(set(s.values()) for s in sources)) - set(bands) - {reg}
    _require(len(body_colors) == 1, "one common body role required")
    body = next(iter(body_colors))
    _require(len({bg, guide, reg, body} | set(bands)) == 8, "eight distinct visible color roles required")
    for source in sources:
        _require({body, reg} <= set(source.values()), "source lacks body or registration")
        _require(len(set(source.values()) & set(bands)) >= 2, "source lacks compass evidence")
    return dict(a=a, rotation=rotation, bg=bg, guide=guide, reg=reg, body=body,
                bands=bands, entries=entries, sources=sources, targets=targets)


def _labels(cells, bands):
    core = [p for p, v in cells.items() if v not in bands]
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    labels = []
    for (r, c), v in cells.items():
        if v in bands:
            facing = [d for d, flag in enumerate((r < r0, c > c1, r > r1, c < c0)) if flag]
            if len(facing) != 1:
                return None
            labels.append((facing[0], v))
    return labels


def _candidates(source, info, use_labels=True):
    h, w = info['a'].shape
    reg, bands = info['reg'], info['bands']
    results, seen = [], set()
    for t in range(8):
        transformed = _transform(source, t)
        labels = _labels(transformed, bands)
        if not labels or (use_labels and any(bands[d] != v for d, v in labels)):
            continue
        marks = {p for p, v in transformed.items() if v == reg}
        for j, target in enumerate(info['targets']):
            if len(marks) != len(target):
                continue
            p, q = min(marks), min(target)
            delta = q[0]-p[0], q[1]-p[1]
            if {(r+delta[0], c+delta[1]) for r, c in marks} != target:
                continue
            placed = _shift(transformed, delta)
            if any(not (1 <= r < h-1 and 1 <= c < w-1) for r, c in placed):
                continue
            if any(info['a'][p] == info['guide'] for p in placed):
                continue
            # Forward-causal pieces must not appear behind their socket front.
            if min(r for r, _ in placed) != min(r for r, _ in target):
                continue
            signature = j, tuple(sorted(placed.items()))
            if signature not in seen:
                seen.add(signature)
                results.append((j, placed))
    return results


def _dockings(info, use_labels=True, limit=2):
    choices = [_candidates(s, info, use_labels) for s in info['sources']]
    order = sorted(range(len(choices)), key=lambda j: len(choices[j]))
    solutions, signatures = [], set()

    def visit(depth, used, occupied, maps):
        if len(solutions) >= limit:
            return
        if depth == len(order):
            # Conditional installation needs a unique footprint per socket,
            # not merely an identical union obtained by repartitioning pieces.
            signature = tuple((j, tuple(sorted(maps[j].items()))) for j in sorted(maps))
            if signature not in signatures:
                signatures.add(signature)
                solutions.append(dict(maps))
            return
        for j, placed in choices[order[depth]]:
            if j in used or occupied.keys() & placed.keys():
                continue
            visit(depth+1, used | {j}, {**occupied, **placed}, {**maps, j: placed})
    visit(0, set(), {}, {})
    return solutions


def _trace(info, maps, mutation=None):
    """Layered causal assembly. All returned diagnostics are transient in memory."""
    out = info['a'].copy()
    for source in info['sources']:
        for p in source:
            out[p] = info['bg']
    h, w = out.shape
    solid = np.zeros(out.shape, dtype=bool)
    owner = np.full(out.shape, -1, dtype=np.int16)
    installed, activation, collisions, incoming = set(), [], [], {}
    simultaneous = 0

    def install(j, depth):
        installed.add(j)
        activation.append((depth, j))
        cells = maps[j]
        for p, color in cells.items():
            out[p] = color
        if mutation == 'stencil':
            blocking = info['targets'][j]
        elif mutation == 'box':
            r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
            c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
            blocking = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)}
        else:
            blocking = cells
        for p in blocking:
            solid[p] = True
            owner[p] = j

    if mutation == 'eager':
        for j in range(len(info['targets'])):
            install(j, 0)
    active = set(info['entries'])
    for r in range(1, h-1):
        incoming[r] = frozenset(active)
        wake = [j for j, target in enumerate(info['targets']) if j not in installed
                and any((r, c) in target for c in active)]
        if len(wake) >= 2:
            simultaneous += 1
        for j in wake:
            if mutation is None:
                _require(r == min(y for y, _ in info['targets'][j]), "socket first contacted behind its leading depth")
            install(j, r)
        hits, following = {}, set()
        for c in sorted(active):
            if solid[r, c] and mutation != 'straight':
                left = right = c
                while left > 1 and solid[r, left-1]:
                    left -= 1
                while right < w-2 and solid[r, right+1]:
                    right += 1
                hits.setdefault((left, right), set()).add(c)
            else:
                following.add(c)
                if out[r, c] == info['bg']:
                    out[r, c] = info['guide']
        for (left, right), columns in sorted(hits.items()):
            ids = tuple(sorted(set(map(int, owner[r, left:right+1])) - {-1}))
            collisions.append((r, left, right, ids, tuple(sorted(columns))))
            for c in range(max(1, left-1), min(w-2, right+1)+1):
                if out[r-1, c] == info['bg']:
                    out[r-1, c] = info['guide']
            for c in (left-1, right+1):
                if 1 <= c < w-1:
                    following.add(c)
                    if out[r, c] == info['bg']:
                        out[r, c] = info['guide']
        active = following
    counts = {}
    for _, _, _, ids, _ in collisions:
        for j in ids:
            counts[j] = counts.get(j, 0)+1
    stats = dict(installed=frozenset(installed), activation=activation, collisions=collisions,
                 incoming=incoming, simultaneous=simultaneous, repeated_baffle=any(n > 1 for n in counts.values()),
                 clipped=any(a == 1 or b == w-2 for _, a, b, _, _ in collisions))
    return out, stats


def _analyze(input_grid):
    info = _parse(input_grid)
    options = _dockings(info)
    _require(len(options) == 1, "registration and compass must determine one docking map")
    out, stats = _trace(info, options[0])
    return info, options[0], out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the entire answer from one input; no seed or examples are used."""
    info, _, out, _ = _analyze(input_grid)
    return np.rot90(out, -info['rotation']).copy()


def _pattern(rng, used, simple=False, size=None):
    for attempt in range(100):
        if simple:
            points = {(0, 0)} if rng.random() < .5 else {(0, 0), (0, 1)}
        else:
            points = {(0, 0)}
            count = int(rng.integers(1, 7)) if size is None else size
            for _ in range(50):
                if len(points) == count:
                    break
                r, c = sorted(points)[int(rng.integers(len(points)))]
                dr, dc = DIRS[int(rng.integers(4))]
                trial = points | {(r+dr, c+dc)}
                if max(y for y, _ in trial)-min(y for y, _ in trial) <= 2 and max(x for _, x in trial)-min(x for _, x in trial) <= 3:
                    points = trial
            if size is not None and len(points) != size:
                continue
        points = set(_transform({p: 1 for p in points}, int(rng.integers(8))))
        key = _shape_key(points)
        if key not in used:
            used.add(key)
            return points
    # More than four deterministic distinct shapes are available for four pieces.
    for n in (range(1, 7) if size is None else (size,)):
        for points in ({(0, c) for c in range(n)}, {(0, c) for c in range(n-1)} | {(1, 0)}):
            key = _shape_key(points)
            if key not in used:
                used.add(key)
                return points
    raise RuntimeError('registration construction exhausted')


def _connect(core, endpoint, rng):
    distances = [(abs(r-endpoint[0])+abs(c-endpoint[1]), r, c) for r, c in core]
    shortest = min(d[0] for d in distances)
    anchors = [(r, c) for d, r, c in distances if d == shortest]
    r, c = anchors[int(rng.integers(len(anchors)))]
    while (r, c) != endpoint:
        if r != endpoint[0] and (c == endpoint[1] or rng.random() < .5):
            r += 1 if endpoint[0] > r else -1
        else:
            c += 1 if endpoint[1] > c else -1
        core.add((r, c))


def _piece(rng, palette, used, simple=False, registration_size=None):
    bg, guide, body, reg, north, east, south, west = palette
    pattern = _pattern(rng, used, simple, registration_size)
    ph, pw = 1+max(r for r, _ in pattern), 1+max(c for _, c in pattern)
    h = int(rng.integers(max(3, ph), max(3, ph)+2))
    w = int(rng.integers(max(3, pw), max(3, pw)+3))
    offset = int(rng.integers(w-pw+1))
    marks = {(r, c+offset) for r, c in pattern}
    core = set(marks)
    # Lower attachments can be wider than the first struck transverse run.
    endpoints = [(int(rng.integers(1, h)), 0), (int(rng.integers(1, h)), w-1),
                 (h-1, int(rng.integers(w)))]
    side_set = (3, 2) if rng.random() < .5 else (1, 2)
    if rng.random() < .25:
        side_set = (1, 2, 3)
    labels = {}
    for d in side_set:
        if d == 2:
            length = int(rng.integers(1, min(2, w)+1))
            start = int(rng.integers(w-length+1))
            for c in range(start, start+length):
                labels[h, c] = south
                endpoints.append((h-1, c))
        else:
            length = int(rng.integers(1, min(2, h-1)+1))
            start = int(rng.integers(1, h-length+1))
            for r in range(start, start+length):
                c = -1 if d == 3 else w
                labels[r, c] = west if d == 3 else east
                endpoints.append((r, 0 if d == 3 else w-1))
    for endpoint in endpoints:
        _connect(core, endpoint, rng)
    # Optional growth remains inside the actual core; gaps are never filled by
    # bounding-box transforms. Growth and stencil geometry are separate draws.
    for _ in range(int(rng.integers(0, h*w//2+1))):
        frontier = sorted({(r+dr, c+dc) for r, c in core for dr, dc in DIRS
                           if 0 <= r+dr < h and 0 <= c+dc < w} - core)
        if frontier:
            core.add(frontier[int(rng.integers(len(frontier)))])
    cells = {p: body for p in core}
    cells.update({p: reg for p in marks})
    cells.update(labels)
    return _transform(cells, 0)


def _base(h, w, palette, entries, lengths):
    bg, guide, _, _, north, east, south, west = palette
    a = np.full((h, w), bg, dtype=np.int8)
    a[0, 1:-1], a[-1, 1:-1] = north, south
    a[1:-1, 0], a[1:-1, -1] = west, east
    for c, length in zip(entries, lengths):
        a[:length, c] = guide
    return a


def _scene_info(base, palette, entries, finals):
    return dict(a=base, bg=palette[0], guide=palette[1], body=palette[2], reg=palette[3],
                bands=tuple(palette[4:]), entries=entries, sources=[],
                targets=[{p for p, v in cells.items() if v == palette[3]} for cells in finals])


def _place_next(rng, piece, row, active, h, w, blocked, reg):
    marks = [(r, c) for (r, c), v in piece.items() if v == reg and r == 0]
    proposals = [(c-p[1], c) for c in sorted(active) for p in marks]
    rng.shuffle(proposals)
    for col, _ in proposals:
        placed = _shift(piece, (row, col))
        if any(not (2 <= r < h-2 and 2 <= c < w-2) for r, c in placed):
            continue
        if not (placed.keys() & blocked):
            return placed
    return None


def _construct(rng, palette, count, witness=None):
    h = int(rng.integers(25 if count == 4 else 22, 31))
    w = int(rng.integers(25 if count >= 3 else 22, 31))
    used = set()
    tied_size = int(rng.integers(3, 7)) if witness == 'cascade' else None
    pieces = [_piece(rng, palette, used, simple=(j == 0),
                     registration_size=tied_size if j in (1, 2) else None) for j in range(count)]
    finals, blocked = [], set()
    row = int(rng.integers(3, 6))
    # Begin with two parallel activations for the cascade teaching example.
    parallel = witness == 'cascade' or (count >= 3 and rng.random() < .55)
    first_count = 2 if parallel else 1
    first_widths = [1+max(c for _, c in p) for p in pieces[:first_count]]
    starts = [int(rng.integers(3, max(4, w-first_widths[0]-3)))]
    if first_count == 2:
        starts = [2, w-first_widths[1]-3]
    entries = []
    for j in range(first_count):
        placed = _shift(pieces[j], (row, starts[j]))
        if placed.keys() & blocked or any(not (2 <= y < h-2 and 2 <= x < w-2) for y, x in placed):
            return None
        finals.append(placed)
        blocked |= _halo(placed)
        top = sorted(c for (r, c), v in placed.items() if r == row and v == palette[3])
        entries.append(top[int(rng.integers(len(top)))])
    entries.sort()
    if any(b-a < 2 for a, b in zip(entries, entries[1:])):
        return None
    lengths = [int(rng.integers(2, min(4, row)+1)) for _ in entries]
    base = _base(h, w, palette, entries, lengths)
    want_dormant = witness == 'selective' or (witness is None and rng.random() < .35)
    for j in range(first_count, count):
        scene = base.copy()
        for cells in finals:
            for p, v in cells.items():
                if v == palette[3]:
                    scene[p] = v
        info = _scene_info(scene, palette, entries, finals)
        try:
            _, stats = _trace(info, dict(enumerate(finals)))
        except ValueError:
            return None
        # A later wave may start before earlier shapes end, if they fit without
        # overlap. This permits more than a stack of isolated rectangular slots.
        if j == count-1 and want_dormant:
            rows = list(range(3, h-7))
            rng.shuffle(rows)
            placed = None
            for depth in rows:
                columns = list(range(2, w-7))
                rng.shuffle(columns)
                for col in columns:
                    trial = _shift(pieces[j], (depth, col))
                    if trial.keys() & blocked or any(not (2 <= y < h-2 and 2 <= x < w-2) for y, x in trial):
                        continue
                    # Dormancy is tested by simulation, not just by a column
                    # chosen to look distant from the initial entry.
                    trial_finals = finals+[trial]
                    test_scene = scene.copy()
                    for p, v in trial.items():
                        if v == palette[3]:
                            test_scene[p] = v
                    try:
                        _, trial_stats = _trace(_scene_info(test_scene, palette, entries, trial_finals), dict(enumerate(trial_finals)))
                    except ValueError:
                        continue
                    if j not in trial_stats['installed']:
                        placed = trial
                        break
                if placed is not None:
                    break
        else:
            first_depth = max(min(r for r, _ in f) for f in finals)+int(rng.integers(3, 6))
            rows = list(range(first_depth, h-5))
            rng.shuffle(rows)
            placed = None
            for depth in rows:
                placed = _place_next(rng, pieces[j], depth, stats['incoming'][depth], h, w, blocked, palette[3])
                if placed is not None:
                    break
        if placed is None:
            return None
        finals.append(placed)
        blocked |= _halo(placed)
    inp = base.copy()
    for cells in finals:
        for p, v in cells.items():
            if v == palette[3]:
                inp[p] = v
    # Supply can overlap future bodies, but never sockets or other supply.
    forbidden = _halo({p for cells in finals for p, v in cells.items() if v == palette[3]})
    forbidden |= _halo(set(zip(*np.nonzero(base == palette[1]))))
    for j, final in enumerate(finals):
        transforms = list(map(int, rng.permutation(8)))
        if j == 0 and witness:
            transforms = [t for t in transforms if (0 < t < 4 if witness == 'selective' else t >= 4)]
        placed_source = None
        for t in transforms:
            source = _transform(final, t)
            sh, sw = 1+max(r for r, _ in source), 1+max(c for _, c in source)
            origins = [(r, c) for r in range(2, h-sh-1) for c in range(2, w-sw-1)]
            rng.shuffle(origins)
            for origin in origins:
                trial = _shift(source, origin)
                if not (trial.keys() & forbidden):
                    placed_source = trial
                    break
            if placed_source is not None:
                break
        if placed_source is None:
            return None
        for p, v in placed_source.items():
            inp[p] = v
        forbidden |= _halo(placed_source)
    try:
        info, maps, output, stats = _analyze(inp)
    except ValueError:
        return None
    if witness == 'selective' and len(stats['installed']) != 1:
        return None
    if witness == 'cascade' and not (len(stats['installed']) == 3 and stats['simultaneous'] and len({r for r, _ in stats['activation']}) >= 2):
        return None
    if witness == 'cascade' and _trace(info, maps, 'straight')[1]['installed'] == stats['installed']:
        return None
    # Require real teaching counterexamples, rather than event-count proxies.
    for mutation in ('straight', 'stencil'):
        wrong = _trace(info, maps, mutation)[0] if witness else None
        if witness and np.array_equal(wrong == info['guide'], output == info['guide']):
            return None
    if witness == 'selective' and not _compass_changes_answer(info, output):
        return None
    return {'input': inp, 'output': output}


def _compass_changes_answer(info, expected):
    for maps in _dockings(info, use_labels=False, limit=16):
        try:
            other = _trace(info, maps)[0]
        except ValueError:
            continue
        if not np.array_equal(other == info['guide'], expected == info['guide']):
            return True
    return False


def _integer(value, name, lo, hi=None):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)), f'{name} must be an integer')
    value = int(value)
    _require(value >= lo and (hi is None or value <= hi), f'illegal {name}')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    _require(nt > nq, 'num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice(10, 8, replace=False)))
    mixed = bool(rng.random() < .6)
    base_rotation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', nt), ('test', nq)):
        for i in range(number):
            local_palette = list(map(int, rng.choice(10, 8, replace=False))) if mixed and i+int(split == 'test') else palette
            witness = ('selective' if i == 0 else 'cascade' if i == 1 else None) if split == 'train' else None
            count = (2 if witness == 'selective' else 3 if witness == 'cascade' else int(rng.integers(2, 4))) if split == 'train' else 4
            for attempt in range(180):
                pair = _construct(rng, local_palette, count, witness)
                if pair is None:
                    continue
                rotation = (base_rotation+i+int(split == 'test')) % 4
                reflected = bool(rng.integers(2))
                pair = {k: np.rot90(np.fliplr(v) if reflected else v, rotation).copy() for k, v in pair.items()}
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'invalid pair counts')
    seen, teachings, moves = set(), [], set()
    registration_witness = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            a, o = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == o.shape and not np.array_equal(a, o), 'invalid output shape or unchanged output')
            _require(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            info, maps, expected, stats = _analyze(a)
            canonical_out = np.rot90(o, info['rotation'])
            _require(np.array_equal(canonical_out, expected), 'output violates causal assembly')
            ca = info['a']
            border = np.zeros(ca.shape, dtype=bool)
            border[[0, -1], :] = True
            border[:, [0, -1]] = True
            _require(np.array_equal(ca[border], canonical_out[border]), 'frame changed')
            _require(np.all(canonical_out[ca == info['guide']] == info['guide']), 'stub changed')
            _require(len(_dockings(info)) == 1, 'ambiguous full docking')
            for j, target in enumerate(info['targets']):
                if j in stats['installed']:
                    _require(all(canonical_out[p] == v for p, v in maps[j].items()), 'installed piece was not rigidly preserved')
                else:
                    _require(all(canonical_out[p] == info['reg'] for p in target), 'dormant stencil changed')
            if split == 'train':
                teachings.append((info, maps, expected, stats))
                patterns = [_shape_key(t) for t in info['targets']]
                registration_witness |= any(len(x) == len(y) and x != y for j, x in enumerate(patterns) for y in patterns[j+1:])
                for source in info['sources']:
                    candidate = _candidates(source, info)[0]
                    final_signature = tuple(sorted(_transform(candidate[1], 0).items()))
                    poses = [t for t in range(8) if tuple(sorted(_transform(source, t).items())) == final_signature]
                    moves.add('translation' if 0 in poses else 'rotation' if any(t < 4 for t in poses) else 'reflection')
    _require(any(len(s['installed']) < len(i['targets']) for i, _, _, s in teachings), 'training lacks dormancy')
    _require(any(s['simultaneous'] for _, _, _, s in teachings), 'training lacks parallel activation')
    _require(any(len({r for r, _ in s['activation']}) >= 2 for _, _, _, s in teachings), 'training lacks cascade')
    _require(any(_trace(i, m, 'straight')[1]['installed'] != s['installed'] for i, m, _, s in teachings), 'training lacks a branch-dependent activation')
    for mutation in ('eager', 'straight', 'stencil'):
        _require(any(not np.array_equal(_trace(i, m, mutation)[0], out) for i, m, out, _ in teachings), f'training admits {mutation} shortcut')
    for mutation in ('straight', 'stencil'):
        _require(any(not np.array_equal(_trace(i, m, mutation)[0] == i['guide'], out == i['guide']) for i, m, out, _ in teachings), f'training lacks path evidence against {mutation}')
    _require(any(_compass_changes_answer(i, out) for i, _, out, _ in teachings), 'training does not require compass labels')
    _require({'rotation', 'reflection'} <= moves, 'training must show both rotation and reflection')
    _require(registration_witness, 'training must distinguish equal-count registration geometries')
    return True


def _analytic_regressions():
    """Hand-specified answers for simultaneous hits, dormancy, erasure and clipping."""
    palette = [0, 1, 2, 3, 4, 5, 6, 7]
    for clipped in (False, True):
        entries = [1] if clipped else [5, 7]
        base = _base(16, 19, palette, entries, [2]*len(entries))
        if clipped:
            first = {(4, 1): 3, (4, 2): 3, (5, 1): 2, (5, 2): 2, (5, 3): 5, (6, 2): 6}
        else:
            first = {(4, 5): 3, (4, 6): 3, (4, 7): 3, (5, 4): 7,
                     (5, 5): 2, (5, 6): 2, (5, 7): 2, (6, 6): 6}
        dormant = {(9, 13): 3, (10, 13): 2, (10, 14): 2, (10, 12): 7, (11, 14): 6}
        inp, expected = base.copy(), base.copy()
        for piece in (first, dormant):
            for p, v in piece.items():
                if v == 3:
                    inp[p] = expected[p] = 3
        for p, v in first.items():
            expected[p] = v
        for j, piece in enumerate((first, dormant)):
            source = _shift(_transform(piece, 1 if j == 0 else 4), (9, 2) if j == 0 else (11, 10))
            for p, v in source.items():
                assert inp[p] == 0
                inp[p] = v
        if clipped:
            guide_pixels = {(r, 1) for r in range(1, 3)}
            guide_pixels |= {(3, c) for c in range(1, 4)}
            guide_pixels |= {(4, 3), (4, 4)}
            guide_pixels |= {(r, 4) for r in range(5, 15)}
        else:
            guide_pixels = {(r, c) for r in (1, 2) for c in (5, 7)}
            guide_pixels |= {(3, c) for c in range(4, 9)}
            guide_pixels |= {(4, c) for c in (3, 4, 8)}
            guide_pixels |= {(r, c) for r in range(5, 15) for c in (3, 8)}
        for p in guide_pixels:
            assert expected[p] in (0, 1)
            expected[p] = 1
        assert np.array_equal(solve(inp), expected), 'hand-derived assembly regression'
        _, _, _, stats = _analyze(inp)
        assert len(stats['installed']) == 1 and len(stats['collisions']) == 2
        assert stats['clipped'] == clipped
        if not clipped:
            assert len(stats['collisions'][0][4]) == 2, 'same-run multi-hit must be processed once'


def self_test():
    """Stream 42 seeds; bound memory to one puzzle and small signature sets."""
    _analytic_regressions()
    shapes, structures, patterns, palettes, directions, counts = set(), set(), set(), set(), set(), set()
    topology = set()
    repeated = dormant = parallel = box_counterexamples = equal_count_distinct = 0
    puzzle_hashes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                info, maps, out, stats = _analyze(a)
                assert np.array_equal(solve(a.copy()), pair['output'])
                digest.update(bytes(a.shape)+a.tobytes()+pair['output'].tobytes())
                shapes.add(a.shape)
                norm = {}
                normalized = bytes(norm.setdefault(int(v), len(norm)) for v in a.flat)
                structures.add(hashlib.sha256(bytes(a.shape)+normalized).digest())
                directions.add(info['rotation'])
                counts.add(len(maps))
                palettes.add((info['bg'], info['guide'], info['body'], info['reg'], *info['bands']))
                local_patterns = [_shape_key(t) for t in info['targets']]
                patterns.update(local_patterns)
                equal_count_distinct += any(len(x) == len(y) and x != y for j, x in enumerate(local_patterns) for y in local_patterns[j+1:])
                for cells in maps.values():
                    normalized_cells = _transform(cells, 0)
                    topology.add(tuple(sorted(normalized_cells)))
                repeated += stats['repeated_baffle']
                dormant += len(stats['installed']) < len(maps)
                parallel += stats['simultaneous'] > 0
                box_counterexamples += not np.array_equal(_trace(info, maps, 'box')[0], out)
        puzzle_hashes.add(digest.digest())
    assert len(puzzle_hashes) == 42 and len(structures) >= 200
    assert len(shapes) >= 25 and len(patterns) >= 20 and len(topology) >= 100
    assert len(palettes) >= 35 and directions == {0, 1, 2, 3} and counts == {2, 3, 4}
    assert repeated and dormant >= 42 and parallel >= 42 and box_counterexamples and equal_count_distinct >= 42
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(8300+nt*7+nq, nt, nq)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    first, again = generate_puzzle(0), generate_puzzle(0)
    assert all(_key(x[k]) == _key(y[k]) for s in ('train', 'test') for x, y in zip(first[s], again[s]) for k in ('input', 'output'))
    # Full-piece transformations are lossless, including reflections and gaps.
    tile = {(0, 0): 2, (1, 0): 4, (1, 2): 7, (2, 2): 1}
    assert all(sorted(_transform(tile, t).values()) == sorted(tile.values()) for t in range(8))
    return dict(seeds=42, pairs=210, distinct_puzzles=len(puzzle_hashes), grid_shapes=len(shapes),
                normalized_structures=len(structures), registration_shapes=len(patterns),
                piece_footprints=len(topology), palettes=len(palettes), repeated_baffle_pairs=int(repeated),
                dormant_pairs=int(dormant), parallel_pairs=int(parallel),
                bounding_box_counterexamples=int(box_counterexamples), equal_count_distinct_pairs=int(equal_count_distinct))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
