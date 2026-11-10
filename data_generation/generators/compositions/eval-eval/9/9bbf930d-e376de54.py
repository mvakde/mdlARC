"""Synchronize admitted tokens by ordinal-reference distance along guided routes.

Only standard-library modules and NumPy are used. No parent code is imported.
The constructor draws its own path-based labels; solve reconstructs routes from
the cells, so it has no access to the constructor's path or sampling decisions.
"""
from __future__ import annotations

import json
import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 566019022, 30182378, 676117528, 54426177, 910587198,
              275585687, 301520496, 483897836, 767450700, 179956636,
              289770636, 432340425, 370486741, 744890453, 26898473,
              907237115, 714719559, 241889607, 7533496, 436120667,
              415171891, 818324283, 771119714, 896010443, 949963975,
              548267087, 532013365, 71202928, 523610434, 170643737,
              192569770, 24103081, 285181661, 784402060, 810035326,
              662709281, 182164352, 910825656, 262848639, 871297416,
              32765966)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((a >= 0) & (a <= 9)), "ARC color bounds")
    return a


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _roles(inp):
    # Background is the unique most frequent color, not a literal color.
    values, counts = np.unique(inp, return_counts=True)
    _require(np.sum(counts == counts.max()) == 1, "background must be unique modal color")
    bg = int(values[np.argmax(counts)])
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        if g.shape[1] < 3 or np.any(g[:, 0] == bg) or not np.all(g[:, 1] == bg):
            continue
        edge = list(map(int, np.unique(g[:, 0])))
        if len(edge) != 2:
            continue
        only_edge = [c for c in edge if not np.any(g[:, 1:] == c)]
        if len(only_edge) != 1:
            continue
        backing = only_edge[0]
        token = next(c for c in edge if c != backing)
        ports = list(map(int, np.flatnonzero(g[:, 0] == token)))
        if len(ports) >= 3 and all(0 < r < g.shape[0] - 1 for r in ports):
            candidates.append((g, k, bg, backing, token, ports))
    _require(len(candidates) == 1, "one visible backed port edge required")
    return candidates[0]


def _flanks(g, p, direction, bg, backing, token):
    normal = (-direction[1], direction[0])
    a, b = _add(p, normal), _add(p, (-normal[0], -normal[1]))
    if not (_inside(g, a) and _inside(g, b)):
        return None
    colors = int(g[a]), int(g[b])
    if any(c in (bg, backing, token) for c in colors):
        return None
    return colors


def _read(inp):
    g, rotation, bg, backing, token, ports = _roles(_grid(inp))
    routes = []
    for row in ports:
        admission = None
        for col in range(1, g.shape[1]):
            if int(g[row, col]) not in (bg, token):
                break
            sides = _flanks(g, (row, col), (0, 1), bg, backing, token)
            if sides is not None:
                admission = sides[0] == sides[1]
                break
        _require(admission is not None, "port needs visible entrance rails")
        path = [(row, 0), (row, 1)]
        p, direction, states = path[-1], (0, 1), set()
        while True:
            state = p, direction
            _require(state not in states, "cyclic route")
            states.add(state)
            nxt = _add(p, direction)
            if not _inside(g, nxt):
                break
            if int(g[nxt]) in (bg, token):
                path.append(nxt)
                p = nxt
                continue
            turns = []
            for d in ((-direction[1], direction[0]), (direction[1], -direction[0])):
                q = _add(p, d)
                if not _inside(g, q) or int(g[q]) not in (bg, token):
                    continue
                flanks = _flanks(g, q, d, bg, backing, token)
                if flanks is not None and flanks[0] == flanks[1]:
                    turns.append(d)
            _require(len(turns) <= 1, "ambiguous turn")
            if not turns:
                break
            direction = turns[0]
        _require(len(path) == len(set(path)), "self-intersecting route")
        beads = [i for i, q in enumerate(path[1:], 1) if int(g[q]) == token]
        _require(len(beads) == 1 and beads[0] >= 2, "one interior token per port")
        directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        bends = sum(a != b for a, b in zip(directions, directions[1:]))
        routes.append(dict(row=row, active=admission, path=path, progress=beads[0],
                           turns=bends, capacity=len(path) - 1,
                           open=path[-1][0] in (0, g.shape[0] - 1) or path[-1][1] == g.shape[1] - 1))
    all_cells = [q for r in routes for q in r['path']]
    _require(len(all_cells) == len(set(all_cells)), "routes must be disjoint")
    _require(np.count_nonzero(g[:, 1:] == token) == len(routes), "unassigned token")
    active = [r for r in routes if r['active']]
    _require(len(active) >= 3 and len(active) % 2 == 1, "odd active cohort of at least three")
    reference = active[len(active) // 2]
    target = reference['progress']
    _require(all(r['capacity'] >= target for r in active), "target must fit every active route")
    return dict(grid=g, rotation=rotation, bg=bg, backing=backing, token=token,
                routes=routes, active=active, reference=reference, target=target)


def _answer(a, routes=None, target=None):
    routes = a['active'] if routes is None else routes
    target = a['target'] if target is None else target
    out = a['grid'].copy()
    # Simultaneous erasure and insertion; neither routes nor destinations merge.
    for r in routes:
        out[r['path'][r['progress']]] = a['bg']
    for r in routes:
        _require(target <= r['capacity'], "alternate target outside route")
        out[r['path'][target]] = a['token']
    return np.rot90(out, -a['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer visible roles, admission, order and travelled distance from input."""
    return _answer(_read(input_grid))


def _positions(rng, count, height, forced_index):
    gaps = [3] * (count - 1)
    # Reserve two cells for a vertical excursion by the required route.
    direction = 1 if forced_index < count - 1 else -1
    if forced_index == 0:
        direction = 1
    gaps[forced_index if direction == 1 else forced_index - 1] += 2
    slack = height - 3 - sum(gaps)
    for _ in range(int(rng.integers(slack + 1))):
        gaps[int(rng.integers(count - 1))] += 1
    span = sum(gaps)
    top = int(rng.integers(1, height - span - 1))
    return [top] + [top + int(v) for v in np.cumsum(gaps)], direction


def _paths(rng, rows, width, height, mover, moves, direction, extra, exclude, move_limit):
    paths = [[(r, 0)] for r in rows]
    current = list(rows)
    used_moves = [0] * len(rows)
    schedule = {}
    first = int(rng.integers(4, 6))
    for j in range(moves):
        schedule[first + 3 * j] = (mover, 2 * direction if j % 2 == 0 else -2 * direction)
    if extra:
        for col in range(first + 3 * moves + 3, width - 3, 3):
            if rng.random() < .65:
                schedule[col] = (None, None)
    for col in range(1, width):
        for i, path in enumerate(paths):
            path.append((current[i], col))
        if col not in schedule:
            continue
        index, dy = schedule[col]
        if index is None:
            options = []
            for i, r in enumerate(current):
                if i in exclude or used_moves[i] >= move_limit:
                    continue
                low = current[i - 1] + 3 if i else 1
                high = current[i + 1] - 3 if i + 1 < len(rows) else height - 2
                for q in range(low, high + 1):
                    if 2 <= abs(q - r) <= 5:
                        options.append((i, q - r))
            if not options:
                continue
            index, dy = options[int(rng.integers(len(options)))]
        low = current[index - 1] + 3 if index else 1
        high = current[index + 1] - 3 if index + 1 < len(rows) else height - 2
        _require(low <= current[index] + dy <= high, "excursion reservation")
        used_moves[index] += 1
        for _ in range(abs(dy)):
            current[index] += 1 if dy > 0 else -1
            paths[index].append((current[index], col))
    required_distance = first + 3 * (moves - 1) + 2 * moves + 1
    return paths, required_distance


def _make_pair(rng, kind, roles, avoid_target=None):
    bg, backing, token, guides, rotation = roles
    if kind == 'witness':
        # Five admitted ports; two decoys lie in a large outer gap. The middle
        # of seven visible ports and the geometric midpoint are both wrong.
        gaps = [int(rng.integers(3, 5)) for _ in range(3)] + [int(rng.integers(9, 12))]
        active_rows = [0] + list(map(int, np.cumsum(gaps)))
        rows = active_rows[:-1] + [active_rows[-2] + 3, active_rows[-2] + 6, active_rows[-1]]
        height = int(rng.integers(max(rows) + 5, 31))
        top = int(rng.integers(1, height - max(rows) - 3))
        rows = [r + top for r in rows]
        flags = [True, True, True, True, False, False, True]
        mover, moves, direction = 6, 1, 1
        width = int(rng.integers(15, 24))
    else:
        active_count = 3 if kind == 'return' else int(rng.choice((3, 5))) if kind == 'train' else int(rng.choice((5, 7)))
        decoys = int(rng.integers(1, 3)) if active_count < 7 else 1
        count = active_count + decoys
        inactive_indices = set(map(int, rng.choice(count, decoys, replace=False)))
        flags = [i not in inactive_indices for i in range(count)]
        active_indices = [i for i, yes in enumerate(flags) if yes]
        mover = active_indices[len(active_indices) // 2] if kind == 'return' else int(rng.choice(active_indices))
        moves = 2 if kind == 'return' else 3 if kind == 'test' else int(rng.integers(1, 3))
        height = int(rng.integers(3 * count + 3, 31))
        rows, direction = _positions(rng, count, height, mover)
        width = int(rng.integers(21, 29)) if kind == 'return' else int(rng.integers(24, 31)) if kind == 'test' else int(rng.integers(20, 29))
    active_indices = [i for i, yes in enumerate(flags) if yes]
    middle = active_indices[len(active_indices) // 2]
    paths, minimum = _paths(rng, rows, width, height, mover, moves, direction,
                           extra=kind in ('train', 'test'), exclude={middle} if kind == 'witness' else set(),
                           move_limit=4 if kind == 'test' else 2)
    choices = [v for v in range(minimum, width - 3) if v != avoid_target]
    _require(choices, "reference range")
    target = int(rng.choice(choices))
    # Trim the unused far route, or retain its boundary exit. Every target has
    # two spare steps, ensuring synchronization cannot mean 'go to the end'.
    for i, path in enumerate(paths):
        if rng.random() < .48:
            capacity = int(rng.integers(target + 2, len(path)))
            paths[i] = path[:capacity + 1]
    progress = [target] * len(paths)
    others = [i for i in active_indices if i != middle]
    rng.shuffle(others)
    mode = 'mixed' if kind in ('witness', 'return') else str(rng.choice(('mixed', 'advance', 'retreat', 'one')))
    for i in others:
        if mode == 'one' or rng.random() < .2:
            continue
        short = mode == 'advance' or (mode == 'mixed' and rng.random() < .5)
        progress[i] = int(rng.integers(2, target)) if short else int(rng.integers(target + 1, len(paths[i])))
    if kind == 'witness':
        for i in others:
            progress[i] = int(rng.integers(2, target))
        progress[others[0]] = int(rng.integers(target + 1, len(paths[others[0]])))
    elif mode == 'mixed':
        progress[others[0]] = int(rng.integers(2, target))
        progress[others[1]] = int(rng.integers(target + 1, len(paths[others[1]])))
    elif all(progress[i] == target for i in others):
        i = others[0]
        progress[i] = target - 1 if mode in ('advance', 'one') else target + 1
    for i, yes in enumerate(flags):
        if not yes:
            progress[i] = target + int(rng.integers(1, 3))

    g = np.full((height, width), bg, dtype=np.int8)
    g[:, 0] = backing
    for row in rows:
        g[row, 0] = token
    cells = {q for p in paths for q in p}
    walls = {}
    # Repeated guide identities have no grouping meaning: equality is local.
    colors = list(map(int, rng.choice(guides, size=len(paths), replace=True)))
    colors[0], colors[-1] = guides[0], guides[1]
    for i, path in enumerate(paths):
        local = {}
        path_cells = set(path)
        for p in path[1:]:
            for d in DIRS:
                q = _add(p, d)
                if _inside(g, q) and q[1] >= 2 and q not in path_cells:
                    _require(q not in cells, "route-wall collision")
                    local[q] = colors[i]
        if not flags[i]:
            other = int(rng.choice([c for c in guides if c != colors[i]]))
            first_turn = next((a[1] for a, b in zip(path, path[1:]) if a[0] != b[0]), width)
            for col in range(2, first_turn):
                if (rows[i] + 1, col) in local:
                    local[(rows[i] + 1, col)] = other
        # Remove paired evidence only within a straight run, retaining entrance,
        # blockers and the outgoing matching flanks of every bend.
        directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        eligible = []
        for j in range(4, len(path) - 3):
            if len(set(directions[j - 2:j + 2])) != 1:
                continue
            d = directions[j]
            n = (-d[1], d[0])
            sides = (_add(path[j], n), _add(path[j], (-n[0], -n[1])))
            if all(q in local for q in sides):
                eligible.append((j, sides))
        if eligible and (kind == 'witness' and i == middle or rng.random() < .55):
            _, sides = eligible[int(rng.integers(len(eligible)))]
            for q in sides:
                del local[q]
        for q, c in local.items():
            _require(q not in walls or walls[q] == c, "conflicting rails")
            walls[q] = c
    for q, c in walls.items():
        g[q] = c
    for path, distance in zip(paths, progress):
        g[path[distance]] = token
    out = g.copy()
    for i in active_indices:
        out[paths[i][progress[i]]] = bg
        out[paths[i][target]] = token
    inp, output = np.rot90(g, rotation).copy(), np.rot90(out, rotation).copy()
    a = _read(inp)
    _require([r['path'] for r in a['routes']] == paths, "visible route differs from drawn route")
    _require([r['active'] for r in a['routes']] == flags, "visible admission differs")
    _require(a['target'] == target and np.array_equal(solve(inp), output), "construction/reader mismatch")
    return {'input': inp, 'output': output}


def _counts(num_train, num_test):
    for v, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not lo <= int(v) <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    fixed_roles = list(map(int, rng.permutation(10)))
    mixing = bool(rng.integers(2))
    base_rotation = int(rng.integers(4))
    rotate_pairs = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    first_target = None
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            kind = 'test' if split == 'test' else 'witness' if index == 0 else 'return' if index == 1 else 'train'
            for _ in range(160):
                palette = list(map(int, rng.permutation(10))) if mixing else fixed_roles
                size = int(rng.integers(2, 8))
                rotation = int(rng.integers(4)) if rotate_pairs else base_rotation
                roles = palette[0], palette[1], palette[2], palette[3:3 + size], rotation
                try:
                    pair = _make_pair(rng, kind, roles, first_target if split == 'train' and index == 1 else None)
                    a = _read(pair['input'])
                    if kind == 'witness' and not _witness(a):
                        continue
                    if kind == 'return' and not _return_witness(a):
                        continue
                except AssertionError:
                    continue
                signature = _key(pair['input'])
                if signature in seen:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                if split == 'train' and index == 0:
                    first_target = a['target']
                break
            else:
                raise RuntimeError('bounded scene sampling exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _witness(a):
    active = a['active']
    positions = np.array([r['row'] for r in active])
    distances = abs(2 * positions - positions[0] - positions[-1])
    nearest = np.flatnonzero(distances == distances.min())
    wrong_middle = a['routes'][len(a['routes']) // 2]
    return (len(nearest) == 1 and active[int(nearest[0])] is not a['reference']
            and active[int(nearest[0])]['progress'] != a['target']
            and sorted(r['progress'] for r in active)[len(active) // 2] != a['target']
            and wrong_middle['progress'] != a['target'])


def _return_witness(a):
    r = a['reference']
    q = r['path'][r['progress']]
    displacement = abs(q[0] - r['row']) + q[1]
    return r['progress'] > displacement and r['turns'] >= 4


def _gaps(a, r):
    """Actual one-cell interruptions bracketed by paired straight-run support."""
    g, path = a['grid'], r['path']
    directions = [(b[0] - p[0], b[1] - p[1]) for p, b in zip(path, path[1:])]
    found = []
    for j in range(3, len(path) - 2):
        if len(set(directions[j - 2:j + 2])) != 1:
            continue
        d = directions[j]
        n = (-d[1], d[0])
        sides = _add(path[j], n), _add(path[j], (-n[0], -n[1]))
        if not all(_inside(g, q) and g[q] == a['bg'] for q in sides):
            continue
        supports = [_flanks(g, path[k], d, a['bg'], a['backing'], a['token']) for k in (j - 1, j + 1)]
        if all(s is not None and s[0] == s[1] for s in supports):
            found.append((j, sides))
    return found


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[k], list) for k in puzzle), 'split lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'shape preservation')
            _require(_key(inp) not in seen, 'distinct inputs')
            seen.add(_key(inp))
            a = _read(inp)
            g, normalized_out = a['grid'], np.rot90(out, a['rotation'])
            _require(np.array_equal(_answer(a), out), 'incorrect synchronization')
            _require(np.any(inp != out), 'nontrivial example')
            _require(any(not r['active'] for r in a['routes']), 'mismatched decoy evidence')
            _require(all(r['capacity'] >= a['target'] + 2 for r in a['routes']), 'spare route beyond synchronization')
            _require(np.array_equal(g[:, 0], normalized_out[:, 0]), 'origin edge preservation')
            guide_mask = (g != a['bg']) & (g != a['token'])
            _require(np.array_equal(g[guide_mask], normalized_out[guide_mask]), 'backing/rail preservation')
            _require(np.count_nonzero(g == a['token']) == np.count_nonzero(normalized_out == a['token']), 'token conservation')
            changes = sum(r['progress'] != a['target'] for r in a['active'])
            _require(np.count_nonzero(g != normalized_out) == 2 * changes, 'only moving interior tokens change')
            for r in a['routes']:
                observed = [i for i, q in enumerate(r['path'][1:], 1) if normalized_out[q] == a['token']]
                _require(observed == [a['target'] if r['active'] else r['progress']], 'route progress/inactivity')
            _require(np.array_equal(g[a['reference']['path'][a['target']]], normalized_out[a['reference']['path'][a['target']]]), 'fixed reference')
            # Ablating A's equality gate moves at least one inactive bead.
            _require(not np.array_equal(_answer(a, a['routes']), out), 'matching must affect the answer')
            # Ablating B and sending each admitted bead to its terminal moves
            # even the reference. Its spare tail guarantees a different answer.
            terminal = g.copy()
            for r in a['active']:
                terminal[r['path'][r['progress']]] = a['bg']
                terminal[r['path'][-1]] = a['token']
            _require(not np.array_equal(terminal, normalized_out), 'reference normalization must affect answer')
            if split == 'train':
                _require(max(r['turns'] for r in a['routes']) <= 4, 'training bend limit')
                training.append(a)
            else:
                _require(max(r['turns'] for r in a['routes']) >= 6, 'test repeated-bend recombination')
    _require(any(_witness(a) for a in training), 'ordinal/nonmedian/filtered reference demonstration')
    _require(any(_witness(a) and _gaps(a, a['reference']) for a in training), 'straight-gap demonstration')
    _require(any(_return_witness(a) for a in training), 'intrinsic-distance demonstration')
    _require(len({a['target'] for a in training}) >= 2, 'variable reference distance')
    _require(len({len(a['active']) for a in training}) >= 2, 'variable admitted counts')
    _require(any(r['progress'] < a['target'] for a in training for r in a['active']), 'advance demonstration')
    _require(any(r['progress'] > a['target'] for a in training for r in a['active']), 'retreat demonstration')


def measurements(puzzle):
    """Cell-derived family features; no construction metadata is retained."""
    records = []
    for split in puzzle:
        for pair in puzzle[split]:
            a = _read(pair['input'])
            routes = a['routes']
            boxes = [(min(q[0] for q in r['path']), max(q[0] for q in r['path']),
                      min(q[1] for q in r['path']), max(q[1] for q in r['path'])) for r in routes]
            overlap = sum(max(b[0], c[0]) <= min(b[1], c[1]) and max(b[2], c[2]) <= min(b[3], c[3])
                          for i, b in enumerate(boxes) for c in boxes[i + 1:])
            entrance_colors = [int(a['grid'][r['row'] - 1, 2]) for r in a['active']]
            target = a['target']
            records.append(dict(split=split, shape=list(np.asarray(pair['input']).shape), normalized_shape=list(a['grid'].shape),
                                rotation=a['rotation'], background=a['bg'], token=a['token'], backing=a['backing'],
                                guides=len(set(map(int, np.unique(a['grid']))) - {a['bg'], a['token'], a['backing']}),
                                active=len(a['active']), inactive=len(routes) - len(a['active']),
                                ports=[r['row'] for r in routes], target=target,
                                progress=[r['progress'] for r in routes], turns=[r['turns'] for r in routes],
                                capacities=[r['capacity'] for r in routes],
                                advance=sum(r['progress'] < target for r in a['active']),
                                retreat=sum(r['progress'] > target for r in a['active']),
                                already_correct=sum(r['progress'] == target for r in a['active']),
                                open=sum(r['open'] for r in routes), closed=sum(not r['open'] for r in routes),
                                gaps=sum(len(_gaps(a, r)) for r in routes), row_range_overlaps=overlap,
                                repeated_active_guide=len(entrance_colors) > len(set(entrance_colors)),
                                ordinal_witness=bool(_witness(a)), return_witness=bool(_return_witness(a))))
    return records


def self_test():
    """Bounded semantic, mutation, equivariance and 42-seed diversity checks."""
    shapes, targets, counts, structural, colors, modes, turn_counts = set(), set(), set(), set(), set(), set(), set()
    rng = np.random.default_rng(72107)
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        _require(all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s]) for k in p), 'determinism')
        for split in puzzle:
            for pair in puzzle[split]:
                a = _read(pair['input'])
                shapes.add(pair['input'].shape)
                targets.add(a['target'])
                counts.add(len(a['active']))
                colors.add(a['bg'])
                turn_counts.update(r['turns'] for r in a['routes'])
                progresses = [r['progress'] for r in a['active']]
                modes.add((any(v < a['target'] for v in progresses), any(v > a['target'] for v in progresses)))
                role_map = {}
                signature = tuple(role_map.setdefault(int(c), len(role_map)) for c in a['grid'].flat)
                structural.add((a['grid'].shape, signature))
        pair = puzzle['test'][0]
        # Role inference and synchronization commute with arbitrary recoloring
        # and all eight rotations/reflections, without training or stored state.
        permutation = rng.permutation(10)
        _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]), 'palette equivariance')
        for k in range(4):
            for mirror in (False, True):
                inp, out = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                if mirror:
                    inp, out = np.fliplr(inp), np.fliplr(out)
                _require(np.array_equal(solve(inp), out), 'D4 equivariance')
        a = _read(puzzle['train'][1]['input'])
        altered = a['grid'].copy()
        ref = a['reference']
        altered[ref['path'][a['target']]] = a['bg']
        altered[ref['path'][a['target'] - 1]] = a['token']
        altered = np.rot90(altered, -a['rotation']).copy()
        _require(not np.array_equal(solve(altered), puzzle['train'][1]['output']), 'visible reference dependency')
        # Closing the straight reference rail interruption changes the guides
        # but must leave the synchronized token positions exactly unchanged.
        a = _read(puzzle['train'][0]['input'])
        filled = a['grid'].copy()
        j, sides = _gaps(a, a['reference'])[0]
        d = (0, 1)
        guide = _flanks(filled, a['reference']['path'][j - 1], d, a['bg'], a['backing'], a['token'])[0]
        for q in sides:
            filled[q] = guide
        filled = np.rot90(filled, -a['rotation']).copy()
        _require(np.array_equal(solve(filled) == a['token'], puzzle['train'][0]['output'] == a['token']), 'rail-gap inertia')
    _require(len(shapes) >= 40 and len(structural) >= 190, 'structural/shape diversity')
    _require(len(targets) >= 10 and counts == {3, 5, 7}, 'distance/count diversity')
    _require(colors == set(range(10)), 'background literal diversity')
    _require({0, 2, 4, 6} <= turn_counts, 'bend diversity')
    _require({(True, True), (True, False), (False, True)} <= modes, 'correction diversity')
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(173, n, m)
            _require(len(p['train']) == n and len(p['test']) == m, 'explicit counts')
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    broken = {s: [{k: g.copy() for k, g in p.items()} for p in pairs] for s, pairs in puzzle.items()}
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    return {'seeds': 42, 'pairs': 210, 'shapes': len(shapes), 'normalized_structures': len(structural),
            'targets': sorted(targets), 'active_counts': sorted(counts), 'turn_counts': sorted(turn_counts)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
