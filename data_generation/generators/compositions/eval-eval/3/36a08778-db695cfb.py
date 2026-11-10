"""Gated diagonal baffles. Pure input-only geometry; no external data or I/O.

Working grids are <=900 int8 cells. Generation retains at most 14 pairs,
uses bounded retries, and writes no samples, logs, caches, or execution traces.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('a grid must be an integer matrix of colors 0..9, size 1..30')
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _point(u, v):
    return (u - v) // 2, (u + v) // 2


def _inside(shape, u, v):
    if (u - v) % 2:
        return False
    r, c = _point(u, v)
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _lanes(shape, u):
    h, w = shape
    lo, hi = max(-u, u - 2 * (h - 1)), min(u, 2 * (w - 1) - u)
    return list(range(lo, hi + 1, 2))


def _sources(a, guide):
    """Only boundary-attached runs of >=2 consecutive forward diagonal cells launch."""
    h, w = a.shape
    starts, cells = [], set()
    boundary = [(0, c) for c in range(w)] + [(r, 0) for r in range(1, h)]
    for r, c in boundary:
        if r + 1 >= h or c + 1 >= w or a[r, c] != guide or a[r + 1, c + 1] != guide:
            continue
        run = []
        rr, cc = r, c
        while rr < h and cc < w and a[rr, cc] == guide:
            run.append((rr, cc))
            rr += 1
            cc += 1
        if rr == h or cc == w:
            return [], set()  # A full boundary-to-boundary input line is ambiguous.
        starts.append((r + c, c - r))
        cells.update(run)
    return starts, cells


def _recognize(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    if len(values) != 3 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError('exactly three colors with a unique dominant background are required')
    background = int(values[counts.argmax()])
    candidates = []
    for rotation in range(4):
        g = np.rot90(a, rotation).copy()
        for guide in map(int, values):
            if guide == background:
                continue
            starts, source_cells = _sources(g, guide)
            if not starts:
                continue
            primary = next(int(x) for x in values if x not in (background, guide))
            groups = defaultdict(list)
            used_v = set()
            valid = True
            for r, c in np.argwhere(g == primary):
                u, v = int(r + c), int(c - r)
                if v in used_v:
                    valid = False  # An unintended partner on the other diagonal.
                used_v.add(v)
                groups[u].append(v)
            boards = []
            for u, lanes in sorted(groups.items()):
                if len(lanes) == 1:
                    continue  # Unpaired primary marker: no barrier.
                if len(lanes) != 2 or max(lanes) - min(lanes) < 4:
                    valid = False
                    break
                lo, hi = min(lanes), max(lanes)
                cells = {_point(u, v) for v in range(lo, hi + 1, 2)}
                if source_cells & cells:
                    valid = False
                    break
                gates = tuple(v for v in range(lo + 2, hi, 2)
                              if g[_point(u, v)] == guide)
                boards.append((u, lo, hi, gates))
            if valid and boards:
                candidates.append((rotation, g, background, guide, primary,
                                   starts, source_cells, boards))
    if len(candidates) != 1:
        raise ValueError('source direction or endpoint pairing is ambiguous')
    return candidates[0]


def _propagate(a, background, guide, primary, starts, boards, mode='normal'):
    """Sweep diagonal depth u=r+c; lane v=c-r. Immutable baffles determine events.

    Modes other than normal are deliberately wrong rules for teaching checks.
    No snapshots or execution traces are retained.
    """
    out = a.copy()
    board_at = {u: (lo, hi, gates) for u, lo, hi, gates in boards}
    wall = set()
    for u, lo, hi, gates in boards:
        for v in range(lo, hi + 1, 2):
            p = _point(u, v)
            wall.add(p)
            if a[p] == background:
                out[p] = primary

    def paint(u, v):
        if _inside(a.shape, u, v):
            p = _point(u, v)
            if a[p] == background and p not in wall:
                out[p] = guide

    sources_at = defaultdict(set)
    for u, v in starts:
        sources_at[u].add(v)
    if mode == 'all_dots':
        for r, c in np.argwhere(a == guide):
            sources_at[int(r + c)].add(int(c - r))
    if mode == 'all_gates':
        # Omit reachability only for gate emission. A dormant gate emits from
        # its next forward cell without falsely activating its own end detours.
        for u, lo, hi, gates in boards:
            sources_at[u + 2].update(gates)
    lattice_active = [set(), set()]
    incoming, reached, remote, clipped, multi, mergers, two_flank = {}, [], 0, 0, 0, 0, 0
    for u in range(sum(a.shape) - 1):
        active = lattice_active[u % 2]
        active |= sources_at.get(u, set())
        active = {v for v in active if _inside(a.shape, u, v)}
        lattice_active[u % 2] = active
        incoming[u] = set(active)
        current = board_at.get(u)
        if current is None:
            for v in active:
                paint(u, v)
            continue
        lo, hi, gates = current
        hits = active & set(range(lo, hi + 1, 2))
        if mode == 'endpoint_collisions':
            hits &= {lo, hi}
        triggered = bool(hits) or mode == 'all_boards'
        if triggered:
            reached.append(u)
            multi += len(hits) > 1
            remote += sum(v not in hits for v in gates)
            for v in range(lo - 2, hi + 3, 2):
                paint(u - 2, v)
            exits = {lo - 2, hi + 2}
            clipped += sum(not _inside(a.shape, u, v) for v in exits)
            two_flank += all(_inside(a.shape, u, v) for v in exits)
            if mode != 'no_gates':
                exits.update(set(gates) & hits if mode == 'direct_gates' else gates)
            active -= hits
            safe = {v for v in exits if _inside(a.shape, u, v)}
            mergers += len(active & safe)
            active |= safe
        for v in active:
            paint(u, v)
    stats = {'reached': tuple(reached), 'incoming': incoming,
             'remote_gates': remote, 'clipped': clipped, 'multi_hit': multi,
             'mergers': mergers, 'two_flank': two_flank, 'board_count': len(boards),
             'lengths': tuple((hi - lo) // 2 + 1 for _, lo, hi, _ in boards),
             'gate_counts': tuple(len(gs) for _, _, _, gs in boards)}
    return out, stats


def _evaluate(input_grid, mode='normal'):
    params = _recognize(input_grid)
    rotation, a, bg, guide, primary, starts, source_cells, boards = params
    out, stats = _propagate(a, bg, guide, primary, starts, boards, mode)
    stats.update(rotation=rotation, background=bg, guide=guide, primary=primary,
                 sources=len(starts), stub_lengths=tuple(
                     sum((r + c >= u and c - r == v) for r, c in source_cells)
                     for u, v in starts),
                 dormant=len(boards) - len(stats['reached']))
    return np.rot90(out, -rotation).copy(), stats, params


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, finite baffles, direction and the full beam network from input."""
    return _evaluate(input_grid)[0]


def _intervals(shape, u, used_v, minimum=3, maximum=12, hit=None, avoid=()):
    lanes = _lanes(shape, u)
    options = []
    for i, lo in enumerate(lanes):
        if lo in used_v:
            continue
        for j in range(i + minimum - 1, min(len(lanes), i + maximum)):
            hi = lanes[j]
            if hi in used_v:
                continue
            if hit is not None and not lo <= hit <= hi:
                continue
            if any(lo <= v <= hi for v in avoid):
                continue
            options.append((lo, hi))
    return options


def _sample_interval(rng, options):
    # Short, broad, and near-spanning baffles share the same candidate pool.
    roll = rng.random()
    if roll < .3:
        shortest = min(hi - lo for lo, hi in options)
        options = [(lo, hi) for lo, hi in options if hi - lo <= shortest + 2]
    elif roll < .5:
        longest = max(hi - lo for lo, hi in options)
        options = [(lo, hi) for lo, hi in options if hi - lo >= longest - 2]
    return options[int(rng.integers(len(options)))]


def _build_scene(rng, kind):
    """Compose baffles against actual incoming lanes; no layout lookup table."""
    if kind == 'plain':
        h, w = map(int, rng.integers(13, 27, size=2))
        count = int(rng.integers(1, 4))
    else:
        h, w = map(int, rng.integers(18, 31, size=2))
        count = int(rng.integers(2, 6))
    a = np.zeros((h, w), dtype=np.int8)
    parity = int(rng.integers(2))
    source_options = [v for v in range(-3, 4) if v % 2 == parity]
    source_count = 1 if kind == 'relay' else int(rng.integers(1, min(3, len(source_options)) + 1))
    sources = sorted(map(int, rng.choice(source_options, source_count, replace=False)))
    for v in sources:
        r, c = _point(abs(v), v)
        n = int(rng.integers(2, 4))
        for step in range(n):
            a[r + step, c + step] = 1
    starts = [(abs(v), v) for v in sources]
    source_cells = set(map(tuple, np.argwhere(a == 1)))
    first = int(rng.choice([u for u in range(10, 17) if u % 2 == parity]))
    # Event depths need not have regular spacing. They share a lattice parity.
    possible = [u for u in range(first + 2, h + w - 5, 2)]
    if len(possible) < count - 1:
        return None
    depths = [first] + sorted(map(int, rng.choice(possible, count - 1, replace=False)))
    if kind == 'relay':
        # Keep the relay early enough for an interior gate to reach another span.
        depths[1] = first + int(rng.choice([4, 6]))
        depths = sorted(set(depths))
    boards, used_v = [], set()
    selected_gate = None
    for index, u in enumerate(depths):
        _, before = _propagate(a, 0, 1, 2, starts, boards)
        active = before['incoming'][u]
        if not active:
            return None
        if kind == 'relay' and index == 1:
            _, no_gates = _propagate(a, 0, 1, 2, starts, boards, 'no_gates')
            options = _intervals(a.shape, u, used_v, 3, 6,
                                 hit=selected_gate, avoid=no_gates['incoming'][u])
        else:
            pivot = int(rng.choice(sorted(active)))
            minimum = 7 if kind == 'relay' and index == 0 else 3
            options = _intervals(a.shape, u, used_v, minimum, min(a.shape), hit=pivot)
        options = [(lo, hi) for lo, hi in options
                   if not any(_point(u, v) in source_cells for v in range(lo, hi + 1, 2))]
        if not options:
            return None
        lo, hi = _sample_interval(rng, options)
        interior = list(range(lo + 2, hi, 2))
        if kind == 'plain':
            gates = []
        elif kind == 'relay' and index == 0:
            eligible = [v for v in interior if v not in active and lo + 4 <= v <= hi - 4]
            if not eligible:
                return None
            selected_gate = int(rng.choice(eligible))
            gates = [selected_gate]
            extra = [v for v in interior if v != selected_gate]
            if extra and rng.random() < .7:
                gates.append(int(rng.choice(extra)))
        else:
            n = int(rng.integers(0, min(4, len(interior)) + 1))
            gates = sorted(map(int, rng.choice(interior, n, replace=False)))
        boards.append((u, lo, hi, tuple(gates)))
        used_v |= {lo, hi}
        a[_point(u, lo)] = a[_point(u, hi)] = 2
        for v in gates:
            a[_point(u, v)] = 1
    if kind == 'relay' or rng.random() < .55:
        # A gated but unreached baffle in the prefix is the causal contrast.
        _, stats = _propagate(a, 0, 1, 2, starts, boards)
        inert_options = []
        for u in range(6 + parity, first, 2):
            for lo, hi in _intervals(a.shape, u, used_v, 3, 7, avoid=stats['incoming'][u]):
                if not any(_point(u, v) in source_cells for v in range(lo, hi + 1, 2)):
                    inert_options.append((u, lo, hi))
        if not inert_options:
            if kind == 'relay':
                return None
        else:
            u, lo, hi = inert_options[int(rng.integers(len(inert_options)))]
            gs = () if kind == 'plain' else (int(rng.choice(list(range(lo + 2, hi, 2)))),)
            boards.append((u, lo, hi, gs))
            used_v |= {lo, hi}
            a[_point(u, lo)] = a[_point(u, hi)] = 2
            for v in gs:
                a[_point(u, v)] = 1
    # Off-span markers are chosen where their forward continuation would differ.
    out, _ = _propagate(a, 0, 1, 2, starts, boards)
    walls = {_point(u, v) for u, lo, hi, _ in boards for v in range(lo, hi + 1, 2)}
    free = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)
            if a[r, c] == 0 and (r, c) not in walls and out[r + 1, c + 1] == 0]
    rng.shuffle(free)
    n = int(rng.integers(0, 4)) if kind != 'relay' else int(rng.integers(1, 4))
    if len(free) < n:
        return None
    for p in free[:n]:
        a[p] = 1
    # Unpaired primary dots have neither diagonal coordinate in common with endpoints.
    endpoint_u = {u for u, _, _, _ in boards}
    desired = int(rng.integers(0, 3))
    for flat in rng.permutation(h * w):
        if desired == 0:
            break
        r, c = int(flat) // w, int(flat) % w
        if a[r, c] == 0 and r + c not in endpoint_u and c - r not in used_v:
            a[r, c] = 2
            endpoint_u.add(r + c)
            used_v.add(c - r)
            desired -= 1
    return a


def _fallback(kind, index):
    """Finite emergency construction; 169 shapes exceed the 14-pair request cap."""
    h, w = 18 + index // 13, 18 + index % 13
    a = np.zeros((h, w), dtype=np.int8)
    a[0, 0] = a[1, 1] = 1
    boards = [(12, -6, 6, ())] if kind == 'plain' else [
        (12, -6, 8, (-2, 2)), (20, -4, 0, ()), (6, 2, 6, (4,))]
    for u, lo, hi, gates in boards:
        a[_point(u, lo)] = a[_point(u, hi)] = 2
        for v in gates:
            a[_point(u, v)] = 1
    a[3, 13] = 1  # Passive marker with an otherwise empty forward continuation.
    return a


def _witness_ok(a, kind):
    try:
        out, stats, params = _evaluate(a)
    except ValueError:
        return False
    if kind == 'plain':
        return stats['two_flank'] > 0 and not any(stats['gate_counts']) and not np.array_equal(
            out, _evaluate(a, 'endpoint_collisions')[0])
    if kind != 'relay':
        return not np.array_equal(a, out)
    no_gate_out, no_gate_stats, _ = _evaluate(a, 'no_gates')
    _, canonical, _, guide, _, _, source_cells, boards = params
    dormant_gates = any(gs and u not in stats['reached'] for u, _, _, gs in boards)
    walls = {_point(u, v) for u, lo, hi, _ in boards for v in range(lo, hi + 1, 2)}
    off = set(map(tuple, np.argwhere(canonical == guide))) - source_cells - walls
    return (stats['remote_gates'] > 0 and dormant_gates and bool(off)
            and bool(set(stats['reached']) - set(no_gate_stats['reached']))
            and not np.array_equal(out, no_gate_out)
            and not np.array_equal(out, _evaluate(a, 'direct_gates')[0])
            and not np.array_equal(out, _evaluate(a, 'all_gates')[0])
            and not np.array_equal(out, _evaluate(a, 'all_boards')[0])
            and not np.array_equal(out, _evaluate(a, 'all_dots')[0]))


def _make_pair(rng, kind, palette, rotation, seen, allow_clipping=True):
    for _ in range(90):
        a = _build_scene(rng, kind)
        if a is None or not _witness_ok(a, kind):
            continue
        if not allow_clipping and _evaluate(a)[1]['clipped']:
            continue
        mapped = np.array(palette, dtype=np.int8)[a]
        actual = np.rot90(mapped, rotation).copy()
        if _key(actual) not in seen:
            return {'input': actual, 'output': solve(actual)}
    for i in range(169):
        a = _fallback(kind, i)
        if not _witness_ok(a, kind):
            continue
        if not allow_clipping and _evaluate(a)[1]['clipped']:
            continue
        actual = np.rot90(np.array(palette, dtype=np.int8)[a], rotation).copy()
        if _key(actual) not in seen:
            return {'input': actual, 'output': solve(actual)}
    raise AssertionError('finite fallback exhausted')


def _counts(num_train, num_test):
    for x in (num_train, num_test):
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
            raise ValueError('example counts must be integers')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and num_train > num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    base = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    kinds = ['plain', 'relay'] + [str(rng.choice(['plain', 'relay', 'free'])) for _ in range(nt - 2)]
    for i, kind in enumerate(kinds):
        rotation = (base + (i % 2)) % 4
        pair = _make_pair(rng, kind, palette, rotation, seen)
        puzzle['train'].append(pair)
        seen.add(_key(pair['input']))
    for _ in range(nq):
        kind = str(rng.choice(['relay', 'free'], p=[.6, .4]))
        rotation = (base + int(rng.choice([2, 3]))) % 4
        allow_clipping = any(_evaluate(p['input'])[1]['clipped'] for p in puzzle['train'])
        pair = _make_pair(rng, kind, palette, rotation, seen, allow_clipping)
        puzzle['test'].append(pair)
        seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, roles, training = set(), set(), []
    clipped = {'train': False, 'test': False}
    orientations = {'train': set(), 'test': set()}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, out = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == out.shape and not np.array_equal(a, out), 'shape or identity error')
            _require(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            expected, stats, params = _evaluate(a)
            _require(np.array_equal(out, expected), 'incorrect baffle network')
            k, canonical, bg, guide, primary, starts, source_cells, boards = params
            roles.add((bg, guide, primary))
            orientations[split].add(k)
            clipped[split] |= bool(stats['clipped'])
            _require(np.all(out[a != bg] == a[a != bg]), 'original marks must be preserved')
            _require(set(map(int, np.unique(out))) <= {bg, guide, primary}, 'new palette role')
            # Every inferred interior is completed, with visible gates retained.
            co = np.rot90(out, k)
            for u, lo, hi, gs in boards:
                for v in range(lo, hi + 1, 2):
                    p = _point(u, v)
                    _require(co[p] == (guide if canonical[p] == guide else primary), 'incomplete finite baffle')
            if split == 'train':
                training.append(a)
    _require(len(roles) == 1, 'palette conventions vary within puzzle')
    _require(not clipped['test'] or clipped['train'], 'clipping in test needs training evidence')
    _require(len(orientations['train']) == 2 and not orientations['train'] & orientations['test'],
             'expected two demonstrated and held-out travel directions')
    _require(any(_witness_ok(a, 'plain') for a in training), 'missing ungated latent-bar collision witness')
    _require(any(_witness_ok(a, 'relay') for a in training), 'missing remote gate / dormant contrast')


def self_test():
    """Bounded 42-seed coverage, mutation and hand-calculated geometry checks."""
    # Three baffles: two reachable only with a remote gate; one dormant gate.
    a = _fallback('relay', 0)
    out, stats, params = _evaluate(a)
    _require(set(stats['reached']) == {12, 20}, 'hand relay reachability')
    _require(6 not in stats['reached'] and stats['remote_gates'] == 2, 'hand dormant/remote gates')
    # First baffle bypass lanes -8,+10; gates -2,+2. Second catches -2 and
    # outputs -6,+2. At depth 24 the union is {-8,-6,2,10}.
    for v in _lanes(a.shape, 24):
        p = _point(24, v)
        _require((out[p] == 1) == (v in {-8, -6, 2, 10}), 'hand exit-lane union')
    _require(out[_point(12, 0)] == 2, 'latent baffle must be completed')
    for mode in ('no_gates', 'direct_gates', 'endpoint_collisions', 'all_boards', 'all_gates', 'all_dots'):
        _require(not np.array_equal(out, _evaluate(a, mode)[0]), 'shortcut was not distinguished')
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(a, k)), np.rot90(out, k)), 'rotation covariance')

    # Both checkerboard parities run independently. The odd-parity gate is a
    # direct hit; the even-parity baffle has remote gates. An unpaired primary
    # dot interrupts the color of a beam without redirecting its continuation.
    dual = np.zeros((20, 20), dtype=np.int8)
    dual[0, 0] = dual[1, 1] = dual[0, 1] = dual[1, 2] = 1
    for u, lo, hi, gs in ((12, -6, 6, (-2, 2)), (13, -5, 5, (1,))):
        dual[_point(u, lo)] = dual[_point(u, hi)] = 2
        for v in gs:
            dual[_point(u, v)] = 1
    dual[_point(24, 8)] = 2
    result = solve(dual)
    for u, expected in ((16, {-8, -2, 2, 8}), (17, {-7, 1, 7})):
        for v in _lanes(dual.shape, u):
            _require((result[_point(u, v)] == 1) == (v in expected), 'two-parity ray lanes')
    _require(result[_point(24, 8)] == 2 and result[_point(26, 8)] == 1,
             'unpaired primary must be transparent and preserved')

    # A long boundary-touching baffle loses both end outlets, yet its central
    # gate continues a five-cell source. These exceed the sampler's short stubs.
    long = np.zeros((30, 30), dtype=np.int8)
    for i in range(5):
        long[i, i] = 1
    long[28, 0] = long[0, 28] = 2
    long[14, 14] = 1
    result, s, _ = _evaluate(long)
    _require(s['clipped'] == 2 and s['lengths'] == (29,) and s['stub_lengths'] == (5,),
             'long clipped baffle or long source support')
    _require(result[15, 15] == 1 and result[13, 15] == 2,
             'a clipped baffle must retain its wall and gate outlet')

    # Crosspiece painting must respect a completed, rather than just visible,
    # upstream baffle when event depths differ by exactly one diagonal step.
    close = np.zeros((18, 18), dtype=np.int8)
    close[0, 0] = close[1, 1] = close[5, 5] = 1
    for u, lo, hi in ((10, -4, 4), (12, -2, 2)):
        close[_point(u, lo)] = close[_point(u, hi)] = 2
    result, s, _ = _evaluate(close)
    _require(set(s['reached']) == {10, 12} and result[_point(10, 2)] == 2,
             'upstream completed baffle must interrupt the crosspiece')

    signatures, structures, shapes, palettes = set(), set(), set(), set()
    lengths, counts, gates, source_counts, stub_lengths, reached_counts = set(), set(), set(), set(), set(), set()
    clipped = mergers = multi = unpaired = absent_gates = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, o = pair['input'], pair['output']
                digest.update(bytes(a.shape) + a.tobytes() + o.tobytes())
                _, s, params = _evaluate(a)
                k, ca, bg, guide, primary, starts, sc, boards = params
                normalized = np.zeros_like(ca)
                normalized[ca == guide] = 1
                normalized[ca == primary] = 2
                structures.add(_key(normalized))
                shapes.add(a.shape)
                palettes.add((bg, guide, primary))
                lengths.update(s['lengths'])
                counts.add(s['board_count'])
                gates.update(s['gate_counts'])
                source_counts.add(s['sources'])
                stub_lengths.update(s['stub_lengths'])
                reached_counts.add(len(s['reached']))
                clipped += s['clipped'] > 0
                mergers += s['mergers'] > 0
                multi += s['multi_hit'] > 0
                unpaired += np.count_nonzero(ca == primary) > 2 * len(boards)
                absent_gates += not any(s['gate_counts'])
        signatures.add(digest.hexdigest())
    _require(len(signatures) == 42 and len(structures) >= 170, '42-seed geometry diversity')
    _require(len(shapes) >= 65 and len(palettes) >= 35, 'shape/palette diversity')
    _require(len(lengths) >= 10 and {3, 4, 7} <= lengths, 'span length diversity')
    _require({1, 2, 3, 4, 5} <= counts and {0, 1, 2, 3} <= gates, 'board/gate diversity')
    _require({1, 2, 3} <= source_counts and {2, 3} <= stub_lengths, 'source diversity')
    _require(len(reached_counts) >= 4 and min(clipped, mergers, multi, unpaired, absent_gates) > 0,
             'missing causal, clipping, merge, multi-hit, or inert cases')
    # Determinism, every legal count, and illegal value/type handling.
    def signature(p):
        return tuple(_key(np.asarray(pair[x])) for s in ('train', 'test') for pair in p[s]
                     for x in ('input', 'output'))
    _require(signature(generate_puzzle(19)) == signature(generate_puzzle(19)), 'seed determinism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1234, nt, nq)
            _require((len(p['train']), len(p['test'])) == (nt, nq), 'explicit count mismatch')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(3, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupted output accepted')
