"""ARC family: turning relation gates route, compose, absorb and pack signals.

Only standard-library modules and NumPy are used. All solving conventions are
fixed in the family or recovered from the input's geometry.
"""
from __future__ import annotations

import json
import numpy as np

_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _trace(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    graph = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _D4
                 if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    paths = []
    while cells:
        first = min(cells)
        cells.remove(first)
        part, queue = {first}, [first]
        for p in queue:
            for q in graph[p]:
                if q in cells:
                    cells.remove(q)
                    part.add(q)
                    queue.append(q)
        ends = sorted(p for p in part if len(graph[p]) == 1)
        assert len(ends) == 2 and all(len(graph[p]) in (1, 2) for p in part)
        path, previous = [ends[0]], None
        while path[-1] != ends[1]:
            current = path[-1]
            choices = [q for q in graph[current] if q != previous]
            assert len(choices) == 1
            path.append(choices[0])
            previous = current
        assert len(path) == len(part)
        # Traversal from the high terminal never needs to go against gravity.
        assert all(b[0] >= a[0] for a, b in zip(path, path[1:]))
        paths.append(path)
    return sorted(paths)


def _gate_runs(core, background):
    gates = []
    for r in range(1, core.shape[0], 2):
        xs = np.flatnonzero(core[r] != background).tolist()
        i = 0
        while i < len(xs):
            start = xs[i]
            run = [start]
            i += 1
            while i < len(xs) and xs[i] == run[-1] + 1:
                run.append(xs[i])
                i += 1
            assert len(run) == 3 and start >= 1 and start + 3 < core.shape[1]
            x, y, z = map(int, core[r, start:start + 3])
            assert x == z
            gates.append((r, start + 1, x, y))
    assert gates
    return gates


def _events(path, gates):
    """A gate is read only on a full vertical crossing of one side column."""
    positions = {p: i for i, p in enumerate(path)}
    events = []
    for j, (r, c, _, _) in enumerate(gates):
        for side in (c - 1, c + 1):
            triplet = [(r - 1, side), (r, side), (r + 1, side)]
            if all(p in positions for p in triplet):
                indexes = [positions[p] for p in triplet]
                assert indexes == list(range(indexes[0], indexes[0] + 3))
                events.append((indexes[1], j))
    assert len({j for _, j in events}) == len(events)
    return [j for _, j in sorted(events)]


def _analyze_core(core, background, track, gates=None):
    h, w = core.shape
    assert 3 <= h <= 29 and h % 2 == 1 and 7 <= w <= 30
    gates = _gate_runs(core, background) if gates is None else gates
    identities = {v for _, _, a, b in gates for v in (a, b)}
    assert track != background and track not in identities and background not in identities
    assert np.all(core[1::2] != track)
    assert np.all(core[::2, [0, w - 1]] != background)
    beads = {tuple(p): int(core[tuple(p)]) for p in np.argwhere(
        (np.indices(core.shape)[0] % 2 == 0) & (core != background) & (core != track))}
    breaks = []
    for r in range(0, h, 2):
        xs = np.flatnonzero(core[r] == background).tolist()
        assert all(2 <= c <= w - 3 for c in xs)
        assert all(b - a >= 3 for a, b in zip(xs, xs[1:]))
        breaks.extend((r, c) for c in xs)
    result = core.copy()
    for p in beads:
        result[p] = track
    writes, overlaps = {}, 0
    for r, c, body, value in gates:
        assert core[r - 1, c] == track and core[r + 1, c] == track
        assert np.all(core[r - 1, c - 1:c + 2] != background)
        assert np.all(core[r + 1, c - 1:c + 2] != background)
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                color = (value if rr == r else body) if cc == c else track
                if (rr, cc) in writes:
                    assert writes[rr, cc] == color
                    overlaps += 1
                writes[rr, cc] = color
    for p, color in writes.items():
        result[p] = color
    paths = _trace(result == track)
    endpoints = set()
    for path in paths:
        for r, c in (path[0], path[-1]):
            assert r % 2 == 0
            assert c in (0, w - 1) or core[r, c - 1] == background or core[r, c + 1] == background
            endpoints.add((r, c))
    assert set(beads) <= endpoints
    events = [_events(path, gates) for path in paths]
    assert all(bool(event) == (path[0][0] != path[-1][0])
               for path, event in zip(paths, events))
    return dict(core=core, background=background, track=track, gates=gates,
                structural=result, writes=writes, overlaps=overlaps, breaks=breaks,
                paths=paths, events=events, beads=beads)


def _interpret(grid):
    """Try geometric roles; reject inputs with zero or multiple interpretations."""
    colors = set(map(int, np.unique(grid)))
    candidates = []
    for background in sorted(colors):
        loc = np.argwhere(grid != background)
        if not len(loc):
            continue
        r0, c0 = map(int, loc.min(axis=0))
        r1, c1 = map(int, loc.max(axis=0) + 1)
        cropped = grid[r0:r1, c0:c1]
        for axis in (0, 1):
            core = cropped if axis == 0 else cropped.T
            if core.shape[0] % 2 != 1 or core.shape[0] < 3 or core.shape[1] < 7:
                continue
            try:
                gates = _gate_runs(core, background)
            except AssertionError:
                continue
            identities = {v for _, _, a, b in gates for v in (a, b)}
            for track in sorted(colors - identities - {background}):
                try:
                    info = _analyze_core(core, background, track, gates)
                except AssertionError:
                    continue
                info.update(box=(r0, r1, c0, c1), axis=axis, shape=grid.shape)
                candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous lane/gate interpretation")
    return candidates[0]


def _execute(info, mode="ordered"):
    answer = info['structural'].copy()
    beads, gates = info['beads'], info['gates']
    records = []
    for path, event in zip(info['paths'], info['events']):
        high, low = path[0], path[-1]
        record = dict(high=high, low=low, events=event, initial=None, final=None,
                      lost=False, failed_at=None, target=None, collision=False,
                      level=high[0] == low[0], trace=[])
        if record['level']:
            for p in (high, low):
                if p in beads:
                    answer[p] = beads[p]
            records.append(record)
            continue
        if low in beads:
            answer[low] = beads[low]
        if high in beads:
            state = beads[high]
            record['initial'] = state
            sequence = event[::-1] if mode == 'reverse' else event
            if mode == 'first_only':
                sequence = sequence[:1]
            for j in sequence:
                _, _, body, value = gates[j]
                record['trace'].append((j, state, body, value))
                if mode != 'ignore_relations' and state != body:
                    record['lost'], record['failed_at'] = True, j
                    break
                if mode != 'ignore_relations':
                    state = value
            if not record['lost']:
                target = path[-2] if low in beads else low
                answer[target] = state
                record.update(final=state, target=target, collision=low in beads)
        records.append(record)
    return answer, records


def _restore(core, info):
    answer = np.full(info['shape'], info['background'], dtype=np.int8)
    r0, r1, c0, c1 = info['box']
    answer[r0:r1, c0:c1] = core if info['axis'] == 0 else core.T
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every role and route from the input, then apply the fixed rule."""
    grid = _grid(input_grid)
    info = _interpret(grid)
    answer, _ = _execute(info)
    return _restore(answer, info)


def _make_structure(rng, lanes, width, depth):
    """Random route backbone plus independently sampled other divider ports."""
    rows = [[] for _ in range(lanes - 1)]
    first = int(rng.integers(lanes - depth))
    # Zigzag centers promote a serial route, without fixing its translation,
    # distances, length or which terminal eventually belongs to it.
    middle = int(rng.integers(4, width - 4))
    left = list(range(2, middle - 1))
    right = list(range(middle + 1, width - 2))
    swap = bool(rng.integers(2))
    for k in range(depth):
        pool = right if (k % 2 == 0) == swap else left
        rows[first + k] = [int(rng.choice(pool))]
    richness = float(rng.random())
    for i in range(lanes - 1):
        wanted = int(rng.choice([0, 1, 1, 2, 3] if richness > .35 else [0, 0, 1, 1, 2]))
        before = rows[i - 1] if i else []
        after = rows[i + 1] if i + 1 < len(rows) else []
        for c in rng.permutation(np.arange(2, width - 2)):
            c = int(c)
            if len(rows[i]) >= wanted:
                break
            if (all(abs(c - p) >= 4 for p in rows[i])
                    and all(abs(c - p) >= 2 for p in before + after)):
                rows[i].append(c)
        rows[i].sort()
    centers = [(2 * i + 1, c) for i, row in enumerate(rows) for c in row]
    breaks = []
    for r in range(0, 2 * lanes - 1, 2):
        ports = sorted((c, dr) for dr, c in centers if abs(dr - r) == 1)
        for (a, sa), (b, sb) in zip(ports, ports[1:]):
            if sa == sb:
                breaks.append((r, int(rng.integers(a + 2, b - 1))))
    # Add optional exposed terminals, checking they do not touch gate tiles.
    for _ in range(int(rng.integers(4))):
        pool = [(r, c) for r in range(0, 2 * lanes - 1, 2) for c in range(2, width - 2)
                if all(abs(c - x) >= 2 for y, x in centers if abs(y - r) == 1)
                and all(abs(c - x) >= 3 for y, x in breaks if y == r)]
        if pool:
            breaks.append(pool[int(rng.integers(len(pool)))])
    # The temporary colors stand for geometry only and are replaced below.
    core = np.zeros((2 * lanes - 1, width), dtype=np.int8)
    core[::2] = 1
    for r, c in centers:
        core[r, c - 1:c + 2] = 2
    for p in breaks:
        core[p] = 0
    info = _analyze_core(core, 0, 1)
    return core, info


def _sample_pair(rng, profile, depth, roles, axis, dense=False):
    # Bounded geometry search. No fixed saved grids or layout tables.
    for _ in range(160):
        lanes = int(rng.integers(max(5, depth + 2), min(15, depth + 7)))
        width = int(rng.integers(11, 31))
        core, geometry = _make_structure(rng, lanes, width, depth)
        choices = [i for i, events in enumerate(geometry['events'])
                   if len(events) >= depth and len(events) < len(geometry['gates'])]
        levels = [i for i, p in enumerate(geometry['paths']) if p[0][0] == p[-1][0]]
        # Training teaches serial composition at depth two/three; tests carry
        # the same rule to depth four/five, rather than only getting recolored.
        depth_ok = depth != 2 or max(map(len, geometry['events'])) <= 3
        if choices and levels and depth_ok:
            break
    else:
        raise RuntimeError("bounded search failed to find a serial route and off-route gate")
    bg, track = roles
    identities = list(map(int, rng.permutation([c for c in range(10) if c not in roles])))
    gates = [(r, c, int(rng.choice(identities)), int(rng.choice(identities)))
             for r, c, _, _ in geometry['gates']]
    focus = int(rng.choice(choices))
    sequence = geometry['events'][focus]
    other_gates = [j for j in range(len(gates)) if j not in sequence]
    decoy = int(rng.choice(other_gates))
    x, y, z, u = identities[:4]
    if profile == 'late_fail':
        r, c, _, _ = gates[sequence[0]]
        gates[sequence[0]] = (r, c, x, y)
        r, c, _, _ = gates[sequence[1]]
        gates[sequence[1]] = (r, c, z, u)  # a later mismatch, after one good lookup
        r, c, _, _ = gates[decoy]
        gates[decoy] = (r, c, y, z)       # a matching off-route donor cannot rescue it
    else:
        state = x
        for k, j in enumerate(sequence):
            value = y if k == 0 else z if k == 1 else int(rng.choice(identities))
            if k == len(sequence) - 1:
                value = int(rng.choice([v for v in identities if v not in (x, y, u)]))
            r, c, _, _ = gates[j]
            gates[j] = (r, c, state, value)
            state = value
        r, c, _, _ = gates[decoy]
        gates[decoy] = (r, c, x, z)  # conflicting global donor, distinct visible route
    inp = np.full(core.shape, bg, dtype=np.int8)
    inp[::2] = track
    for p in geometry['breaks']:
        inp[p] = bg
    for r, c, body, value in gates:
        inp[r, c - 1:c + 2] = (body, value, body)
    beads = {}
    p = float(rng.uniform(.55, .95) if dense else rng.uniform(.12, .5))
    for path, events in zip(geometry['paths'], geometry['events']):
        high, low = path[0], path[-1]
        if rng.random() < p:
            beads[low] = int(rng.choice(identities))
        if rng.random() < p:
            # Both compatible arrivals and wrong-first-gate losses occur.
            beads[high] = gates[events[0]][2] if events and rng.random() < .65 else int(rng.choice(identities))
    path = geometry['paths'][focus]
    beads[path[0]] = x
    beads[path[-1]] = u  # visibly different stationary signal; fixed colored order
    level = geometry['paths'][int(rng.choice(levels))]
    beads[level[int(rng.choice([0, len(level) - 1]))]] = int(rng.choice(identities))
    if profile != 'late_fail':
        # A clearly mismatching high signal teaches that gates are filters,
        # rather than unconditional substitutions.
        others = [i for i, e in enumerate(geometry['events']) if e and i != focus]
        if others:
            k = int(rng.choice(others))
            first_body = gates[geometry['events'][k][0]][2]
            beads[geometry['paths'][k][0]] = int(rng.choice([v for v in identities if v != first_body]))
    for cell, color in beads.items():
        inp[cell] = color
    if rng.random() < .5:
        inp = np.fliplr(inp).copy()
    margins = []
    for length in inp.shape:
        total = int(rng.integers(min(3, 30 - length) + 1))
        before = int(rng.integers(total + 1))
        margins.append((before, total - before))
    inp = np.pad(inp, margins, constant_values=bg)
    if axis:
        inp = inp.T.copy()
    info = _interpret(inp)
    result, _ = _execute(info)
    return dict(input=inp.copy(), output=_restore(result, info))


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    fixed_roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    varied_roles = bool(rng.random() < .65)
    orientation = int(rng.integers(3))  # horizontal, vertical, or shared mixed orientation
    profiles = ['success', 'late_fail'] + ['random'] * (nt - 2)
    rng.shuffle(profiles)
    axes = [0] * nt if orientation == 0 else [1] * nt if orientation == 1 else [0, 1] + list(map(int, rng.integers(2, size=nt - 2)))
    rng.shuffle(axes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            profile = profiles[i] if split == 'train' else str(rng.choice(['success', 'late_fail']))
            depth = 2 if split == 'train' else int(rng.integers(4, 6))
            axis = axes[i] if split == 'train' else int(rng.integers(2)) if orientation == 2 else orientation
            for _ in range(24):
                roles = tuple(map(int, rng.choice(10, 2, replace=False))) if varied_roles else fixed_roles
                pair = _sample_pair(rng, profile, depth, roles, axis, dense=split == 'test')
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _features(info, records):
    rows = [sum(r == rr for r, _, _, _ in info['gates'])
            for rr in range(1, info['core'].shape[0], 2)]
    off_route_conflict = late_rescue = 0
    for rec in records:
        if rec['initial'] is None:
            continue
        route = set(rec['events'])
        if not rec['lost'] and len(route) >= 2:
            first = info['gates'][rec['events'][0]]
            off_route_conflict += any(j not in route and g[2] == first[2] and g[3] != first[3]
                                      for j, g in enumerate(info['gates']))
        if rec['lost'] and len(rec['trace']) >= 2:
            state = rec['trace'][-1][1]
            late_rescue += any(j not in route and g[2] == state
                               for j, g in enumerate(info['gates']))
    return dict(shape=info['shape'], axis=info['axis'], roles=(info['background'], info['track']),
                lanes=(info['core'].shape[0] + 1) // 2, width=info['core'].shape[1],
                gates=len(info['gates']), row_counts=rows, breaks=len(info['breaks']),
                overlaps=info['overlaps'], paths=len(info['paths']), beads=len(info['beads']),
                max_depth=max(map(len, info['events'])),
                success=sum(r['final'] is not None for r in records),
                serial_success=sum(r['final'] is not None and len(r['events']) >= 2 for r in records),
                changed=sum(r['final'] is not None and r['final'] != r['initial'] for r in records),
                loss=sum(r['lost'] for r in records),
                late_loss=sum(r['lost'] and len(r['trace']) >= 2 for r in records),
                collision=sum(r['collision'] for r in records),
                colored_collision=sum(r['collision'] and r['final'] != info['beads'][r['low']]
                                      for r in records),
                levels=sum(r['level'] and (r['high'] in info['beads'] or r['low'] in info['beads']) for r in records),
                off_route_conflict=off_route_conflict, late_rescue=late_rescue,
                monochrome=sum(a == b for _, _, a, b in info['gates']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, features, ablations = set(), {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            info = _interpret(inp)
            core_out, records = _execute(info)
            assert np.array_equal(out, _restore(core_out, info))
            assert not np.array_equal(inp, out)
            # Every local write, including gates themselves, remains exactly
            # the simultaneous quarter-turn construction.
            for cell, color in info['writes'].items():
                if color != info['track']:
                    assert core_out[cell] == color
            assert all(core_out[p] == info['background'] for p in info['breaks'])
            # Per-path accounting: one deletion per failed moving signal.
            for path, rec in zip(info['paths'], records):
                initial = sum(p in info['beads'] for p in (path[0], path[-1]))
                final = sum(core_out[p] != info['track'] for p in path)
                assert final == initial - int(rec['lost'])
                if rec['level']:
                    assert all(core_out[p] == info['core'][p] for p in path)
                elif path[-1] in info['beads']:
                    assert core_out[path[-1]] == info['beads'][path[-1]]
                if rec['target'] is not None:
                    assert core_out[rec['target']] == rec['final']
                    assert rec['target'] == (path[-2] if path[-1] in info['beads'] else path[-1])
            # Outside gate tiles and path cells nothing is changed.
            allowed = set(info['writes']) | {p for path in info['paths'] for p in path}
            assert all(core_out[p] == info['core'][p]
                       for p in np.ndindex(info['core'].shape) if p not in allowed)
            features[split].append(_features(info, records))
            if split == 'train':
                for mode in ('ignore_relations', 'reverse', 'first_only'):
                    alternative, _ = _execute(info, mode)
                    if not np.array_equal(alternative, core_out):
                        ablations.add(mode)
                # Ignore A's connectivity, but still show the turned scaffold:
                # leave the signal states at their original lane terminals.
                unchanged_signals = info['structural'].copy()
                for cell, state in info['beads'].items():
                    unchanged_signals[cell] = state
                if not np.array_equal(unchanged_signals, core_out):
                    ablations.add('ignore_routing')
    train = features['train']
    for name in ('serial_success', 'changed', 'collision', 'colored_collision', 'levels', 'loss',
                 'late_loss', 'off_route_conflict', 'late_rescue'):
        assert any(f[name] for f in train), f"missing training witness: {name}"
    assert ablations == {'ignore_relations', 'reverse', 'first_only', 'ignore_routing'}
    assert {f['axis'] for f in features['test']} <= {f['axis'] for f in train}
    assert max(f['max_depth'] for f in train) <= 3
    assert min(f['max_depth'] for f in features['test']) >= 4
    # All four counts, palettes and shifts are checked from visible inputs;
    # no generation metadata is read by the validator or solver.


def self_test():
    # Hand-derived literal regressions: colored collision, mismatch deletion,
    # two-gate composition and compatible tile overlap. Expected outputs are
    # written directly, not obtained by calling a construction/solve helper.
    def digits(rows):
        return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    regressions = [
        (['2555557', '0023200', '4555555'],
         ['5552555', '0053500', '4352555']),
        (['555555552', '023200000', '555555555', '000343000', '555555555'],
         ['552555555', '053500000', '552535555', '000545000', '455535555']),
    ]
    for raw, expected in regressions:
        inp, out = digits(raw), digits(expected)
        assert np.array_equal(solve(inp), out)
        assert np.array_equal(solve(inp.T), out.T)
        assert np.array_equal(solve(np.pad(inp, ((1, 2), (2, 1)))),
                              np.pad(out, ((1, 2), (2, 1))))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    records, layouts = [], set()
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input'].copy()
                predicted = solve(inp)
                assert np.array_equal(inp, pair['input']), "solver must not mutate its input"
                assert np.array_equal(predicted, pair['output'])
                info = _interpret(inp)
                _, rec = _execute(info)
                f = _features(info, rec)
                f['split'] = split
                records.append(f)
                # Separate geometry from the literal palette and relations.
                geometry = np.full(info['core'].shape, 0, dtype=np.int8)
                geometry[::2] = 1
                for p in info['breaks']:
                    geometry[p] = 0
                for r, c, _, _ in info['gates']:
                    geometry[r, c - 1:c + 2] = 2
                for p in info['beads']:
                    geometry[p] = 3
                layouts.add(_grid_key(geometry))
    assert len(layouts) >= 150
    assert len({f['shape'] for f in records}) >= 50
    assert len({f['roles'] for f in records}) >= 35
    assert {f['axis'] for f in records} == {0, 1}
    assert len({f['gates'] for f in records}) >= 10
    assert len({f['beads'] for f in records}) >= 12
    assert len({f['breaks'] for f in records}) >= 7
    assert {0, 1, 2, 3} <= {n for f in records for n in f['row_counts']}
    assert any(f['overlaps'] for f in records)
    assert any(f['monochrome'] for f in records)
    assert any(30 in f['shape'] for f in records)
    for split in ('train', 'test'):
        fs = [f for f in records if f['split'] == split]
        for name in ('collision', 'serial_success', 'late_loss', 'levels', 'off_route_conflict'):
            assert any(f[name] for f in fs), (split, name)
    assert max(f['max_depth'] for f in records if f['split'] == 'test') >= 5
    assert any(len({_interpret(p['input'])['axis'] for s in ('train', 'test') for p in q[s]}) == 2
               for q in puzzles)
    assert any(len({_interpret(p['input'])['background'] for s in ('train', 'test') for p in q[s]}) == 1
               for q in puzzles)
    # All legal explicit counts, including the smallest demonstration set.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1700 + nt * 10 + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Negative geometry and semantic-label checks.
    invalid = np.zeros((5, 11), dtype=np.int8)
    invalid[::2] = 1
    invalid[1, 3:6] = 2
    invalid[3, 3:6] = 2  # aligned gates branch after rotation
    try:
        solve(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("branching geometry accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, pairs=len(records), geometry_layouts=len(layouts),
                shapes=len({f['shape'] for f in records}),
                palettes=len({f['roles'] for f in records}),
                max_gate_depth=max(f['max_depth'] for f in records))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
