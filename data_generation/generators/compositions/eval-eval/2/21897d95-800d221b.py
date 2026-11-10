"""District referendums: original local signals vote through directed wall gates.

Only Python's standard library and NumPy are used. Construction keeps room
records; solve reconstructs a fresh color-union graph from the input alone.
"""
from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _add(p, d, n=1):
    return p[0] + n * d[0], p[1] + n * d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(mask):
    ids = np.full(mask.shape, -1, dtype=np.int16)
    groups = []
    for start in zip(*np.nonzero(mask)):
        if ids[start] >= 0:
            continue
        label = len(groups)
        ids[start] = label
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for d in DIRS:
                q = _add(p, d)
                if _inside(q, mask.shape) and mask[q] and ids[q] < 0:
                    ids[q] = label
                    stack.append(q)
        groups.append(cells)
    return groups, ids


def _connected(mask):
    return bool(np.any(mask)) and len(_components(mask)[0]) == 1


def _stencil(center, direction):
    perpendicular = (-direction[1], direction[0])
    return {center, _add(center, direction), _add(center, perpendicular),
            _add(center, perpendicular, -1)}


def _read_ts(grid, ink):
    groups, _ = _components(grid == ink)
    if not groups:
        return None
    gates = []
    for group in groups:
        if len(group) != 4:
            return None
        cells = set(group)
        interpretations = [(p, d) for p in group for d in DIRS
                           if _stencil(p, d) == cells]
        if len(interpretations) != 1:
            return None
        p, d = interpretations[0]
        back, forward = _add(p, d, -1), _add(p, d, 2)
        if not _inside(back, grid.shape) or not _inside(forward, grid.shape):
            return None
        gates.append(dict(center=p, direction=d, cells=cells,
                          back=back, forward=forward))
    return gates


def _interpret(input_grid):
    """Recover roles, original electorates, directions, and gate multiplicities."""
    grid = _grid(input_grid)
    values = list(map(int, np.unique(grid)))
    if len(values) != 4:
        raise ValueError("Family inputs require exactly four geometric color roles")
    valid = []
    for ink in values:
        ts = _read_ts(grid, ink)
        if ts is None:
            continue
        for wall in values:
            if wall == ink:
                continue
            signals = tuple(c for c in values if c not in (ink, wall))
            rooms, ids = _components(np.isin(grid, signals))
            if len(rooms) < 2:
                continue
            original, local_counts = [], []
            good = True
            for cells in rooms:
                n0 = sum(grid[p] == signals[0] for p in cells)
                n1 = len(cells) - n0
                if not (n0 and n1 and n0 != n1):
                    good = False
                    break
                local_counts.append((int(n0), int(n1)))
                original.append(signals[int(n1 > n0)])
            if not good:
                continue
            gates = []
            incoming = np.zeros((len(rooms), 2), dtype=int)
            for t in ts:
                source, target = int(ids[t['back']]), int(ids[t['forward']])
                if source < 0 or target < 0 or source == target:
                    good = False
                    break
                payload = original[source]
                incoming[target, signals.index(payload)] += 1
                gates.append(dict(t, source=source, target=target, payload=payload))
            if not good:
                continue
            final = [signals[int(b > a)] if a != b else None for a, b in incoming]
            valid.append(dict(grid=grid, wall=wall, ink=ink, signals=signals,
                              rooms=rooms, ids=ids, original=original,
                              local_counts=local_counts, gates=gates,
                              incoming=incoming, final=final))
    if len(valid) != 1:
        raise ValueError("Input must have one district/gate interpretation")
    return valid[0]


def _render_solution(info):
    out = info['grid'].copy()
    out[out == info['ink']] = info['wall']
    for label, winner in enumerate(info['final']):
        if winner is not None:
            out[info['ids'] == label] = winner
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, puzzle state, labels, or generation metadata are consulted."""
    return _render_solution(_interpret(input_grid))


def _gate_candidates(labels):
    candidates = []
    for center in zip(*np.nonzero(labels < 0)):
        for d in DIRS:
            cells = _stencil(center, d)
            back, forward = _add(center, d, -1), _add(center, d, 2)
            if not all(_inside(p, labels.shape) for p in cells | {back, forward}):
                continue
            source, target = int(labels[back]), int(labels[forward])
            if source < 0 or target < 0 or source == target:
                continue
            if not all(labels[p] < 0 for p in cells):
                continue
            candidates.append(dict(center=center, direction=d, cells=cells,
                                   back=back, forward=forward,
                                   source=source, target=target))
    return candidates


def _halo(cells):
    return cells | {_add(p, d) for p in cells for d in DIRS}


def _pattern(rng, cells, majority, minority, style=None, strength=None, forced=()):
    """Arrange an exact strict vote, independently of gate weights and room area."""
    n = len(cells)
    forced = set(forced)
    strength = int(rng.integers(3)) if strength is None else strength
    if strength == 0:
        k = (n - 1) // 2
    elif strength == 1:
        k = max(1, int(n * rng.uniform(.16, .36)))
    else:
        k = int(rng.integers(1, max(2, int(n * .10) + 1)))
    k = min((n - 1) // 2, max(k, len(forced), 1))
    _need(len(forced) <= k, "Too many forced minorities")
    style = int(rng.integers(4)) if style is None else style
    points = sorted(cells)
    if style == 0:
        order = [points[int(i)] for i in rng.permutation(n)]
    elif style == 1:
        origin = points[int(rng.integers(n))]
        noise = {p: float(rng.random()) for p in points}
        order = sorted(points, key=lambda p: abs(p[0] - origin[0])
                       + abs(p[1] - origin[1]) + noise[p] * .4)
    elif style == 2:
        parity = int(rng.integers(2))
        order = sorted(points, key=lambda p: ((sum(p) + parity) % 2, p))
    else:
        axis = int(rng.integers(2))
        reverse = bool(rng.integers(2))
        order = sorted(points, key=lambda p: (p[axis], p[1 - axis]), reverse=reverse)
    selected = set(forced)
    for p in order:
        if len(selected) == k:
            break
        selected.add(p)
    return {p: minority if p in selected else majority for p in points}


def _build_from_records(rng, labels, gates, roles, majority_bits=None, witness=False):
    """Write expectations from construction records, then check input recovery."""
    wall, ink, signal0, signal1 = roles
    count = int(labels.max()) + 1
    if majority_bits is None:
        majority_bits = list(map(int, rng.integers(2, size=count)))
        if len(set(majority_bits)) == 1:
            majority_bits[int(rng.integers(count))] ^= 1
    colors = [signal0, signal1]
    majorities = [colors[b] for b in majority_bits]
    inp = np.full(labels.shape, wall, dtype=np.int8)
    for i in range(count):
        cells = set(zip(*np.nonzero(labels == i)))
        forced = {g['back'] for g in gates if g['source'] == i} if witness else set()
        strength = (0 if i == 0 else 2) if witness else None
        values = _pattern(rng, cells, majorities[i], colors[1 - majority_bits[i]],
                          strength=strength, forced=forced)
        for p, value in values.items():
            inp[p] = value
    out = inp.copy()
    votes = np.zeros((count, 2), dtype=int)
    for g in gates:
        for p in g['cells']:
            inp[p] = ink
        votes[g['target'], majority_bits[g['source']]] += 1
    for i, (a, b) in enumerate(votes):
        if a != b:
            out[labels == i] = colors[int(b > a)]
    # Decoding never sees labels, sampled bits, or vote records.
    info = _interpret(inp)
    _need(np.array_equal(_render_solution(info), out), "Record/reconstruction disagreement")
    return dict(input=inp, output=out)


def _transform_pair(rng, pair):
    k, reflect = int(rng.integers(4)), bool(rng.integers(2))
    return {name: (np.fliplr(np.rot90(a, k)) if reflect else np.rot90(a, k)).copy()
            for name, a in pair.items()}


def _witness(rng, roles):
    """3:2 port winner, 1:2 source loser, minority backing pixels, tie, dependency."""
    heights = [(a, b, c) for a in range(9, 12) for b in range(6, 8)
               for c in range(6, 8) if a + b + c + 6 <= 30]
    a, b, c = heights[int(rng.integers(len(heights)))]
    left = int(rng.integers(5, 13))
    right = int(rng.integers(5, min(17, 27 - left)))
    h, w = a + b + c + 6, left + right + 4
    labels = np.full((h, w), -1, dtype=np.int16)
    starts = [1, a + 3, a + b + 5]
    for i, (start, height) in enumerate(zip(starts, (a, b, c))):
        labels[start:start + height, 1:1 + left] = i
    labels[1:-1, left + 3:-1] = 3
    candidates = _gate_candidates(labels)
    lookup = {(g['center'], g['direction']): g for g in candidates}
    gates = []
    # Three separated ports from the close-majority source.
    r0 = 1 + int(rng.integers(a - 8))
    for r in (r0, r0 + 4, r0 + 8):
        gates.append(lookup[((r, left + 1), (0, 1))])
    # One vote into the principal recipient and one return vote in each minor room.
    for start, height in zip(starts[1:], (b, c)):
        # Leave one row before the first center. Otherwise gates at the last
        # source row and the next room's first row can touch across a wall junction.
        first = start + int(rng.integers(1, height - 4))
        r_in, r_out = (first, first + 4) if rng.random() < .5 else (first + 4, first)
        gates.append(lookup[((r_in, left + 1), (0, 1))])
        gates.append(lookup[((r_out, left + 2), (0, -1))])
    # Middle room receives opposite original signals: its complete mosaic survives.
    col = int(rng.integers(2, left - 1))
    gates.append(lookup[((a + 1, col), (1, 0))])
    all_cells = set()
    for g in gates:
        _need(not g['cells'] & _halo(all_cells), "Witness gate collision")
        all_cells |= g['cells']
    pair = _build_from_records(rng, labels, gates, roles, [0, 1, 1, 1], witness=True)
    pair = _transform_pair(rng, pair)
    _need(_features(_interpret(pair['input']))['witness'], "Witness contrast lost")
    return pair


def _partition(rng, shape, count):
    """Pixel-coordinate straight/stepped cuts and occasional enclosing wall rings."""
    h, w = shape
    labels = np.full(shape, -1, dtype=np.int16)
    labels[1:h - 1, 1:w - 1] = 0
    rr, cc = np.indices(shape)
    for new_id in range(1, count):
        completed = False
        for _ in range(70):
            areas = np.array([np.sum(labels == i) for i in range(new_id)], dtype=float)
            owner = int(rng.choice(new_id, p=areas / areas.sum()))
            mask = labels == owner
            rows, cols = np.nonzero(mask)
            r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
            if min(r1 - r0, c1 - c0) >= 10 and rng.random() < .2:
                top = int(rng.integers(r0 + 3, r1 - 5))
                bottom = int(rng.integers(top + 2, r1 - 2))
                left = int(rng.integers(c0 + 3, c1 - 5))
                right = int(rng.integers(left + 2, c1 - 2))
                patch = (rr >= top - 2) & (rr < bottom + 2) & (cc >= left - 2) & (cc < right + 2)
                if not np.all(mask[patch]):
                    continue
                part = mask & (rr >= top) & (rr < bottom) & (cc >= left) & (cc < right)
                separator = patch & ~part
                other = mask & ~patch
            else:
                axis = int(rng.integers(2))
                lo, hi = (r0, r1) if axis == 0 else (c0, c1)
                if hi - lo < 6:
                    continue
                cut = int(rng.integers(lo + 2, hi - 3))
                path = np.full(w if axis == 0 else h, cut)
                if rng.random() < .52 and hi - lo >= 8:
                    step = int(rng.integers(3, 8))
                    value = cut
                    for start in range(0, len(path), step):
                        value = int(np.clip(value + rng.integers(-1, 2), lo + 2, hi - 4))
                        path[start:start + step] = value
                coord, along = (rr, cc) if axis == 0 else (cc, rr)
                part = mask & (coord < path[along])
                other = mask & (coord >= path[along] + 2)
                separator = mask & ~(part | other)
            if (int(part.sum()) < 6 or int(other.sum()) < 6
                    or not _connected(part) or not _connected(other)):
                continue
            labels[part] = new_id
            labels[separator] = -1
            completed = True
            break
        if not completed:
            return None
    return labels


def _random_gates(rng, candidates, desired):
    # Spatially separated annotations; contextual source/destination pixels may be shared.
    by_pair = {}
    for g in candidates:
        by_pair.setdefault((g['source'], g['target']), []).append(g)
    chosen, blocked = [], set()
    order = list(map(int, rng.permutation(len(candidates))))
    if by_pair and rng.random() < .6:
        keys = list(by_pair)
        preferred = keys[int(rng.integers(len(keys)))]
        preferred_cells = {g['center'] for g in by_pair[preferred]}
        order.sort(key=lambda i: candidates[i]['center'] not in preferred_cells)
    for i in order:
        g = candidates[i]
        if g['cells'] & blocked:
            continue
        chosen.append(g)
        blocked |= _halo(g['cells'])
        if len(chosen) == desired:
            break
    return chosen


def _competitive_gates(rng, candidates, desired, count):
    """Sample a port/source-count disagreement on this newly generated floorplan.

    The star is only a subset of the graph: other edges may form cycles, ties,
    and branches. Sources can share several boundary segments or collector sides.
    """
    by_target = {}
    for g in candidates:
        by_target.setdefault(g['target'], {}).setdefault(g['source'], []).append(g)
    targets = [t for t, sources in by_target.items() if len(sources) >= 3]
    if not targets:
        return None
    for _ in range(40):
        target = int(rng.choice(targets))
        available = list(by_target[target])
        nsenders = int(rng.integers(3, min(5, len(available)) + 1))
        senders = list(map(int, rng.choice(available, size=nsenders, replace=False)))
        nwinning = 2 if nsenders == 5 and rng.random() < .5 else 1
        if nwinning == 1:
            weights = [nsenders - 1 + int(rng.integers(1, 3))] + [1] * (nsenders - 1)
        else:
            weights = list(map(int, rng.integers(2, 4, size=2))) + [1] * 3
        # Occasionally strengthen one losing source while keeping a strict port win.
        if rng.random() < .35 and sum(weights[:nwinning]) > sum(weights[nwinning:]) + 1:
            weights[int(rng.integers(nwinning, nsenders))] += 1
        gates, blocked = [], set()
        good = True
        for index in sorted(range(nsenders), key=lambda i: -weights[i]):
            source = senders[index]
            found = 0
            pool = by_target[target][source]
            for j in rng.permutation(len(pool)):
                g = pool[int(j)]
                if g['cells'] & blocked:
                    continue
                gates.append(g)
                blocked |= _halo(g['cells'])
                found += 1
                if found == weights[index]:
                    break
            if found != weights[index]:
                good = False
                break
        if not good:
            continue
        # Extra graph edges do not secretly alter the sampled contest's weights.
        limit = max(desired, len(gates) + int(rng.integers(1, 4)))
        for j in rng.permutation(len(candidates)):
            g = candidates[int(j)]
            if g['target'] == target or g['cells'] & blocked:
                continue
            gates.append(g)
            blocked |= _halo(g['cells'])
            if len(gates) >= limit:
                break
        bits = list(map(int, rng.integers(2, size=count)))
        winning_bit = int(rng.integers(2))
        for i, source in enumerate(senders):
            bits[source] = winning_bit if i < nwinning else 1 - winning_bit
        bits[target] = 1 - winning_bit
        return gates, bits
    return None


def _random_pair(rng, roles, is_test, require_competition=False):
    for _ in range(160):
        bounds = (20, 31) if is_test else (13, 28)
        shape = tuple(map(int, rng.integers(*bounds, size=2)))
        if rng.random() < .18:
            shape = shape[0], shape[0]
        count = int(rng.integers(6, 10) if is_test else rng.integers(3, 6))
        labels = _partition(rng, shape, count)
        if labels is None:
            continue
        candidates = _gate_candidates(labels)
        desired = int(rng.integers(7, 19) if is_test else rng.integers(2, 12))
        competitive = require_competition or rng.random() < .25
        plan = _competitive_gates(rng, candidates, desired, count) if competitive else None
        if require_competition and plan is None:
            continue
        gates, bits = plan if plan is not None else (_random_gates(rng, candidates, desired), None)
        if len(gates) < (5 if is_test else 2):
            continue
        try:
            pair = _build_from_records(rng, labels, gates, roles, bits)
            if require_competition and not _features(_interpret(pair['input']))['weighted']:
                continue
            return _transform_pair(rng, pair)
        except (ValueError, AssertionError):
            continue
    raise RuntimeError("Bounded floorplan construction exhausted")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("Require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, size=4, replace=False)))
    varying = bool(rng.random() < .6)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(20):
                roles = tuple(map(int, rng.choice(10, size=4, replace=False))) if varying else palette
                pair = (_witness(rng, roles) if split == 'train' and index == 0 else
                        _random_pair(rng, roles, split == 'test', split == 'test' and index == 0))
                key = _key(pair['input'])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("Could not construct distinct inputs")
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    signals = info['signals']
    local = info['original']
    incoming = info['incoming']
    final = info['final']
    gates = info['gates']
    areas = list(map(len, info['rooms']))
    weighted_counterexample = raw_counterexample = area_counterexample = pixel_counterexample = False
    combined = False
    competitive_profiles = []
    for target, winner in enumerate(final):
        if winner is None:
            continue
        senders = {g['source'] for g in gates if g['target'] == target}
        n = [sum(local[s] == c for s in senders) for c in signals]
        raw = [sum(info['local_counts'][s][i] for s in senders) for i in range(2)]
        area = [sum(areas[s] for s in senders if local[s] == c) for c in signals]
        pixels = [sum(info['grid'][g['back']] == c for g in gates if g['target'] == target)
                  for c in signals]
        loser_index = 1 - signals.index(winner)
        contrasts = [counts[loser_index] > counts[1 - loser_index]
                     for counts in (n, raw, area, pixels)]
        weighted_counterexample |= contrasts[0]
        raw_counterexample |= contrasts[1]
        area_counterexample |= contrasts[2]
        pixel_counterexample |= contrasts[3]
        combined |= all(contrasts) and winner != local[target]
        if contrasts[0]:
            competitive_profiles.append(tuple(sorted(Counter(
                g['source'] for g in gates if g['target'] == target).values())))
    dependency = any(final[g['source']] is not None
                     and final[g['source']] != local[g['source']]
                     and incoming[g['target']].sum() == 1
                     and final[g['target']] == local[g['source']] for g in gates)
    ties = [i for i, (a, b) in enumerate(incoming) if a == b and a > 0]
    untouched = [i for i, v in enumerate(incoming) if v.sum() == 0]
    nonrectangular, possible_enclosers = 0, []
    h, w = info['grid'].shape
    for label, cells in enumerate(info['rooms']):
        rows, cols = zip(*cells)
        irregular = len(cells) != (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        nonrectangular += irregular
        if irregular:
            possible_enclosers.append(label)
    # Actual topological enclosure: treating one room as a barrier, another room
    # cannot reach the canvas boundary even when all walls are traversable.
    enclosed_ids = set()
    for barrier in possible_enclosers:
        groups, _ = _components(info['ids'] != barrier)
        for group in groups:
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in group):
                continue
            enclosed_ids.update(int(info['ids'][p]) for p in group if info['ids'][p] >= 0)
    pair_counts = Counter((g['source'], g['target']) for g in gates)
    room_out = Counter(g['source'] for g in gates)
    fanout = any(len({g['target'] for g in gates if g['source'] == s}) > 1 for s in room_out)
    return dict(rooms=len(local), gates=len(gates), shape=info['grid'].shape,
                roles=(info['wall'], info['ink'], *signals), areas=areas,
                margins=[abs(a - b) for a, b in info['local_counts']],
                minorities=[min(a, b) / (a + b) for a, b in info['local_counts']],
                directions={g['direction'] for g in gates}, ties=len(ties),
                untouched=len(untouched), dependency=dependency,
                weighted=weighted_counterexample, raw=raw_counterexample,
                area=area_counterexample, pixel=pixel_counterexample,
                competitive_profiles=competitive_profiles,
                fanout=fanout, parallel=max(pair_counts.values()),
                nonrectangular=nonrectangular, enclosed=len(enclosed_ids),
                witness=combined and dependency and bool(ties) and bool(untouched))


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Wrong puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal example counts")
    seen, measured = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Wrong pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, "Output geometry changed")
            key = _key(inp)
            _need(key not in seen, "Repeated input")
            seen.add(key)
            info = _interpret(inp)
            _need(np.array_equal(_render_solution(info), out), "Incorrect referendum output")
            _need(not np.array_equal(inp, out), "No transformation")
            _need(np.all(out[inp == info['wall']] == info['wall']), "Wall damage")
            _need(np.all(out[inp == info['ink']] == info['wall']), "Unremoved annotation")
            # These explicit checks concern the intended scope, including retention.
            for i, winner in enumerate(info['final']):
                mask = info['ids'] == i
                if winner is None:
                    _need(np.array_equal(out[mask], inp[mask]), "Tie/no-vote mosaic damaged")
                else:
                    _need(np.all(out[mask] == winner), "Only part of a district painted")
            _need(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "Invented color")
            measured[split].append(_features(info))
    _need(any(m['witness'] for m in measured['train']), "Training lacks joint parent counterexamples")
    _need(all(3 <= m['rooms'] <= 5 for m in measured['train']), "Training room regime")
    _need(all(6 <= m['rooms'] <= 9 for m in measured['test']), "Missing graph-size test shift")
    _need(any(m['weighted'] for m in measured['test']), "Test omits port/source-count contrast")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test() -> dict:
    """Bounded 42-seed exercise plus input-only metamorphic/mutation checks."""
    puzzles, structures, shapes, roles = set(), set(), set(), set()
    metrics, split_directions = [], {'train': set(), 'test': set()}
    test_profiles = set()
    stable, changing = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_puzzle_key(puzzle))
        if seed < 2:
            _need(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "Nondeterministic seed")
        palette_set = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info = _interpret(inp)
                m = _features(info)
                metrics.append(m)
                shapes.add(inp.shape)
                roles.add(m['roles'])
                palette_set.add(m['roles'])
                split_directions[split].update(m['directions'])
                if split == 'test':
                    test_profiles.update(m['competitive_profiles'])
                # Normalize all roles to measure geometry and signal arrangement.
                mapping = {info['wall']: 0, info['ink']: 1,
                           info['signals'][0]: 2, info['signals'][1]: 3}
                norm = np.array([[mapping[int(c)] for c in row] for row in inp], dtype=np.int8)
                structures.add(_key(norm))
                _need(np.array_equal(solve(inp), out), "Input-only solution failed")
        stable += len(palette_set) == 1
        changing += len(palette_set) > 1
    _need(len(puzzles) == 42 and len(structures) >= 200, "Collapsed structural variation")
    _need(len(shapes) >= 50 and len(roles) >= 50, "Collapsed geometry/palette variation")
    _need(stable >= 3 and changing >= 3, "Missing palette policies")
    _need({m['rooms'] for m in metrics} >= set(range(3, 10)), "Collapsed room counts")
    _need(len({m['gates'] for m in metrics}) >= 8, "Collapsed gate counts")
    _need(len(test_profiles) >= 3, "Test contests use one weight template")
    _need(all(ds == set(DIRS) for ds in split_directions.values()), "Missing cardinal gate directions")
    for feature in ('weighted', 'raw', 'area', 'pixel', 'dependency', 'ties',
                    'untouched', 'fanout', 'nonrectangular', 'enclosed'):
        _need(sum(bool(m[feature]) for m in metrics) >= 3, "Missing variation: " + feature)
    _need(max(max(m['areas']) for m in metrics) > 200, "No large districts")
    _need(min(f for m in metrics for f in m['minorities']) < .08
          and max(f for m in metrics for f in m['minorities']) > .45, "Collapsed local margins")
    example = generate_puzzle(71, 2, 1)
    inp, out = example['train'][0]['input'], example['train'][0]['output']
    # Both geometry and arbitrary palette permutations are equivariant.
    for k in range(4):
        _need(np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k)), "Rotation dependence")
    perm = np.random.default_rng(918).permutation(10)
    _need(np.array_equal(solve(perm[inp]), perm[out]), "Literal color dependence")
    _need(np.array_equal(solve(np.fliplr(inp)), np.fliplr(out)), "Reflection dependence")
    malformed = inp.copy()
    annotation = _interpret(malformed)
    first_gate = annotation['gates'][0]
    malformed[_add(first_gate['center'], first_gate['direction'])] = annotation['wall']
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed T accepted")
    # Incorrect expected labels must fail validation.
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in example[s]] for s in example}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong output accepted")
    # Create an actual local tie without moving any geometric pixel.
    tied = inp.copy()
    info = _interpret(tied)
    even_room = next(cells for cells in info['rooms'] if len(cells) % 2 == 0)
    for i, p in enumerate(even_room):
        tied[p] = info['signals'][int(i >= len(even_room) // 2)]
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("Illegal local tie accepted")
    for counts in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(98, *counts)
        _need((len(p['train']), len(p['test'])) == counts, "Explicit counts ignored")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    return dict(seeds=42, pairs=len(metrics), distinct_puzzles=len(puzzles),
                normalized_inputs=len(structures), shapes=len(shapes), role_palettes=len(roles),
                room_counts=sorted({m['rooms'] for m in metrics}),
                gate_counts=sorted({m['gates'] for m in metrics}),
                test_weight_profiles=sorted(test_profiles),
                nonrectangular_pairs=sum(bool(m['nonrectangular']) for m in metrics),
                enclosed_pairs=sum(bool(m['enclosed']) for m in metrics),
                stable_puzzles=stable, changing_palette_puzzles=changing)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
