"""Rekeying courier: a relay's pigment is the key for its outgoing rails.

Only standard-library modules and NumPy are used. No files, traces, caches,
parent generators, seeds, or example labels are consulted by solve().
"""
from itertools import combinations
import hashlib
import json

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _add(p, direction):
    dr, dc = DIRECTIONS[direction]
    return p[0] + dr, p[1] + dc


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            p = stack.pop()
            for d in range(4):
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), 'grid dimensions')
    _require(np.issubdtype(g.dtype, np.integer), 'integer grid required')
    _require(np.all((0 <= g) & (g <= 9)), 'ARC colors required')
    return g


def _decode(value):
    g = _grid(value)
    h, w = g.shape
    bg = int(g[0, 0])
    _require(all(g[p] == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1))), 'clear corners')
    components = _components(map(tuple, np.argwhere(g != bg)))
    boundary = [c for c in components if any(r in (0, h - 1) or s in (0, w - 1) for r, s in c)]
    _require(len(boundary) == 1, 'unique boundary launcher')
    launcher = boundary[0]
    starts = [p for p in launcher if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    _require(len(starts) == 1, 'one launcher root')
    start = starts[0]
    direction = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    line, p = [], start
    for _ in range(len(launcher)):
        line.append(p)
        p = _add(p, direction)
    _require(set(line) == launcher and len(line) >= 2, 'straight tail and cap')
    signal, marker = int(g[line[0]]), int(g[line[-1]])
    _require(len({bg, signal, marker}) == 3 and all(g[p] == signal for p in line[:-1]), 'launcher colors')
    objects = []
    for cells in components:
        if cells == launcher or not any(g[p] == marker for p in cells):
            continue
        ports = [p for p in cells if g[p] == marker]
        _require(len(ports) == 1 and len(cells) >= 5, 'one port per relay')
        port = ports[0]
        body = cells - {port}
        keys = {int(g[p]) for p in body}
        _require(len(keys) == 1 and not keys & {bg, signal, marker}, 'uniform distinct relay key')
        _require(len(_components(body)) == 1, 'connected body')
        r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
        c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
        sides = [d for d, yes in enumerate((port[0] == r0, port[1] == c1, port[0] == r1, port[1] == c0)) if yes]
        _require(len(sides) == 1 and _add(port, (sides[0] + 2) % 4) in body, 'unambiguous extreme-side port')
        _require(0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, 'interior relay')
        objects.append(dict(cells=cells, port=port, direction=sides[0], key=next(iter(keys)),
                            shape=(r1 - r0 + 1, c1 - c0 + 1)))
    _require(objects, 'at least one relay')
    owner = {p: i for i, obj in enumerate(objects) for p in obj['cells']}
    _require(int(np.count_nonzero(g == marker)) == len(objects) + 1, 'all markers assigned')
    guides = {p for p in map(tuple, np.argwhere(g != bg)) if p not in owner and p not in launcher}
    _require(all(g[p] != marker for p in guides), 'marker cannot be a rail')
    return dict(grid=g, bg=bg, signal=signal, marker=marker, launcher=launcher,
                cap=line[-1], direction=direction, objects=objects, owner=owner, guides=guides)


def _flanks(scene, p, direction):
    a = _add(p, (direction + 1) % 4)
    b = _add(p, (direction - 1) % 4)
    if a in scene['guides'] and b in scene['guides']:
        return a, b
    return None


def _analyse(input_grid, ignore_rails=False):
    """Directional input reader; key is loaded from each original relay body.

    Every two-sided guide opening is a socket. Straight inertia crosses gaps
    between sockets. A blocked courier can turn only
    into a locally key-matching pair of rails. Ties/cycles are invalid inputs.
    """
    s = _decode(input_grid)
    g, objects = s['grid'], s['objects']
    p, direction, key, admitted = s['cap'], s['direction'], s['signal'], False
    reached, turns, sockets, key_history, port_turns = [], [], [], [], []
    visited, travel = set(), []
    rejected = None
    for _ in range(g.size * (len(objects) + 2)):
        state = p, direction, key, admitted
        _require(state not in visited, 'courier cycle')
        visited.add(state)
        q = _add(p, direction)
        if not _inside(g, q):
            break
        if q in s['owner']:
            index = s['owner'][q]
            _require(index not in reached, 'relay reactivation')
            obj = objects[index]
            reached.append(index)
            key_history.append(obj['key'])
            port_turns.append((obj['direction'] - direction) % 4)
            p, direction, key = obj['port'], obj['direction'], obj['key']
            admitted = False
            continue
        if int(g[q]) == s['bg']:
            flanks = _flanks(s, q, direction)
            if flanks:
                colors = tuple(int(g[a]) for a in flanks)
                if not ignore_rails and colors != (key, key):
                    rejected = (q, colors, key)
                    break
                if not admitted:
                    sockets.append((q, colors, key))
                admitted = True
            p = q
            travel.append(p)
            continue
        choices = []
        for d in ((direction + 1) % 4, (direction - 1) % 4):
            q = _add(p, d)
            if not _inside(g, q) or int(g[q]) != s['bg']:
                continue
            flanks = _flanks(s, q, d)
            if flanks and (ignore_rails or all(int(g[a]) == key for a in flanks)):
                choices.append(d)
        _require(len(choices) <= 1, 'ambiguous rail turn')
        if not choices:
            break
        turns.append((p, direction, choices[0], key))
        direction, admitted = choices[0], True
    else:
        raise AssertionError('route step bound exceeded')
    _require(int(g[p]) == s['bg'], 'terminal must be a free cell')
    out = g.copy()
    for cell in s['launcher']:
        out[cell] = s['bg']
    for i, obj in enumerate(objects):
        for cell in obj['cells']:
            out[cell] = s['signal'] if i in reached else s['bg']
    out[p] = s['signal']
    s.update(output=out, reached=reached, turns=turns, sockets=sockets, terminal=p,
             rejected=rejected, key_history=key_history, port_turns=port_turns, travel=travel)
    return s


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer solely from the supplied input."""
    return _analyse(input_grid)['output'].copy()


def _gap_count(scene):
    """Count missing straight support bracketed by visible guide pairs."""
    path, count = scene['travel'], 0
    for a, p, b in zip(path, path[1:], path[2:]):
        delta = (p[0] - a[0], p[1] - a[1])
        if delta not in DIRECTIONS or delta != (b[0] - p[0], b[1] - p[1]):
            continue
        d = DIRECTIONS.index(delta)
        if _flanks(scene, a, d) and not _flanks(scene, p, d) and _flanks(scene, b, d):
            count += 1
    return count


def _has_outgoing_socket(scene, obj):
    p = _add(obj['port'], obj['direction'])
    while _inside(scene['grid'], p) and int(scene['grid'][p]) == scene['bg']:
        pair = _flanks(scene, p, obj['direction'])
        if pair and all(int(scene['grid'][q]) == obj['key'] for q in pair):
            return True
        p = _add(p, obj['direction'])
    return False


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), 'orthogonal segment')
    return 0 if dr < 0 else 2 if dr > 0 else 1 if dc > 0 else 3


def _axis(rng, count):
    first = int(rng.integers(6, 9 if count == 2 else 7))
    coords = [first]
    for _ in range(count - 1):
        maximum = min(13, 26 - coords[-1] - 8 * (count - len(coords) - 1))
        coords.append(coords[-1] + int(rng.integers(8, maximum + 1)))
    size = int(rng.integers(coords[-1] + 4, 31))
    return coords, size


def _walk(rng, rows, cols, length, elbow_count):
    """Bounded randomized DFS; complexity is selected before any placement."""
    starts = [(r, c) for r in range(rows) for c in range(cols) if r == 0 or c == 0]
    rng.shuffle(starts)
    budget = [20000]

    def search(path):
        budget[0] -= 1
        if budget[0] <= 0:
            return None
        if len(path) == length:
            corners = [i for i in range(1, length - 1) if _direction(path[i - 1], path[i]) != _direction(path[i], path[i + 1])]
            options = list(combinations(corners, elbow_count))
            rng.shuffle(options)
            entries = ([2] if path[0][0] == 0 else []) + ([1] if path[0][1] == 0 else [])
            rng.shuffle(entries)
            for elbows in options:
                for entry in entries:
                    active = [i for i in range(length - 1) if i not in elbows]
                    if any(_direction(path[i], path[i + 1]) != (entry if i == 0 else _direction(path[i - 1], path[i])) for i in active):
                        return path, set(elbows), entry
            return None
        neighbors = [_add(path[-1], d) for d in range(4)]
        rng.shuffle(neighbors)
        for q in neighbors:
            if 0 <= q[0] < rows and 0 <= q[1] < cols and q not in path:
                found = search(path + [q])
                if found is not None:
                    return found
        return None

    for start in starts:
        result = search([start])
        if result is not None:
            return result
    raise AssertionError('no feasible coarse route')


def _relay(rng, center, direction, key):
    h, w = (int(x) for x in rng.choice([3, 5], size=2))
    rh, cw = h // 2, w // 2
    kind = int(rng.integers(4))
    depth = min(2 if kind == 2 else 1 if kind == 1 else 0, rh, cw)
    local = {(r, c) for r in range(-rh, rh + 1) for c in range(-cw, cw + 1)
             if min(r + rh, rh - r) + min(c + cw, cw - c) >= depth}
    if kind == 3:
        # One clipped corner, with a randomly chosen orientation.
        local.remove((rh if rng.integers(2) else -rh, cw if rng.integers(2) else -cw))
    candidates = []
    for p in sorted(local):
        extreme = (p[0] == -rh, p[1] == cw, p[0] == rh, p[1] == -cw)
        if extreme[direction] and sum(extreme) == 1 and _add(p, (direction + 2) % 4) in local:
            if len(_components(local - {p})) == 1:
                candidates.append(p)
    _require(candidates, 'relay port fit')
    # Off-center ports are allowed on five-cell sides.
    port = candidates[int(rng.integers(len(candidates)))]
    translate = lambda p: (p[0] + center[0], p[1] + center[1])
    return dict(cells={translate(p) for p in local}, port=translate(port),
                direction=direction, key=key)


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        if target == path[-1]:
            continue
        direction = _direction(path[-1], target)
        while path[-1] != target:
            path.append(_add(path[-1], direction))
    _require(len(path) == len(set(path)), 'self-overlapping path')
    return path


def _construct_pair(rng, active_count, elbow_count, gate_type, colors, force_gap=False):
    bg, signal, marker, *keys = colors
    length = active_count + elbow_count + 1
    sizes = [(r, c) for r in (2, 3) for c in (2, 3) if r * c >= length]
    rows, cols = sizes[int(rng.integers(len(sizes)))]
    coarse, elbows, entry = _walk(rng, rows, cols, length, elbow_count)
    ys, h = _axis(rng, rows)
    xs, w = _axis(rng, cols)
    centers = [(ys[r], xs[c]) for r, c in coarse]
    object_indices = [i for i in range(length) if i not in elbows]
    relays = []
    for i in object_indices:
        direction = _direction(coarse[i], coarse[i + 1]) if i < length - 1 else int(rng.integers(4))
        relays.append(_relay(rng, centers[i], direction, int(rng.choice(keys))))
    unused = [(ys[r], xs[c]) for r in range(rows) for c in range(cols) if (r, c) not in coarse]
    rng.shuffle(unused)
    extra = int(rng.integers(0, min(3, len(unused)) + 1))
    for center in unused[:extra]:
        relays.append(_relay(rng, center, int(rng.integers(4)), int(rng.choice(keys))))
    first = relays[0]
    root = (0, centers[0][1]) if entry == 2 else (centers[0][0], 0)
    near = min(p[0] if entry == 2 else p[1] for p in first['cells'])
    tail_length = int(rng.integers(1, min(2, near - 3) + 1))
    launcher, p = [], root
    for _ in range(tail_length + 1):
        launcher.append(p)
        p = _add(p, entry)
    cap = launcher[-1]
    all_objects = set().union(*(obj['cells'] for obj in relays))
    paths, departure_keys = [], []
    # The initial approach plus one leg for each actually reached relay.
    for leg in range(active_count + 1):
        if leg == 0:
            start, direction, key, intermediates, target = cap, entry, signal, [], first
        else:
            source = relays[leg - 1]
            start, direction, key = source['port'], source['direction'], source['key']
            lo, hi = object_indices[leg - 1], object_indices[leg]
            intermediates = centers[lo + 1:hi]
            target = relays[leg]
        points, p = [start], start
        for j, center in enumerate(intermediates + [centers[object_indices[leg]]]):
            q = (center[0], p[1]) if direction % 2 == 0 else (p[0], center[1])
            points.append(q)
            p = q
            if j < len(intermediates):
                next_center = (intermediates + [centers[object_indices[leg]]])[j + 1]
                direction = _direction(center, next_center)
        complete = _polyline(points)
        contact = next((i for i, q in enumerate(complete[1:], 1) if q in target['cells']), None)
        _require(contact is not None and contact >= 3, 'target contact')
        path = complete[:contact]
        _require(not set(path[1:]) & all_objects, 'intervening relay')
        paths.append(path)
        departure_keys.append(key)
    all_paths = set().union(*(set(path) for path in paths))
    # The blocked relay has a genuine keyed outgoing rail spur as well. Its
    # erasure cannot be predicted just by looking for an unwired last node.
    blocked = relays[active_count]
    spur, p = [blocked['port']], blocked['port']
    for _ in range(4):
        p = _add(p, blocked['direction'])
        if not (1 <= p[0] < h - 1 and 1 <= p[1] < w - 1):
            break
        _require(p not in all_objects and p not in all_paths and p not in launcher, 'spur obstruction')
        spur.append(p)
    _require(len(spur) >= 4, 'visible outgoing spur on blocked relay')
    all_paths.update(spur)
    protected = all_objects | set(launcher)
    forbidden_walls = protected | {_add(p, d) for p in protected for d in range(4)}
    walls, final_walls = {}, set()
    for leg_index, (path, key) in enumerate(zip(paths, departure_keys)):
        for p in path[1:]:
            for d in range(4):
                q = _add(p, d)
                if not (0 <= q[0] < h and 0 <= q[1] < w) or q in forbidden_walls or q in all_paths:
                    continue
                _require(q not in walls or walls[q] == key, 'rail color collision')
                walls[q] = key
                if leg_index == len(paths) - 1:
                    final_walls.add(q)
    for p in spur[1:]:
        for d in ((blocked['direction'] + 1) % 4, (blocked['direction'] - 1) % 4):
            q = _add(p, d)
            if q in forbidden_walls:
                continue
            _require(q not in all_paths and (q not in walls or walls[q] == blocked['key']), 'spur rail collision')
            walls[q] = blocked['key']
    # Remove only straight interior support; preserve admission and every turn.
    gaps = 0
    for path in paths:
        candidates = []
        directions = [_direction(a, b) for a, b in zip(path, path[1:])]
        for i in range(4, len(path) - 3):
            if len(set(directions[i - 2:i + 2])) == 1:
                d = directions[i]
                pair = (_add(path[i], (d + 1) % 4), _add(path[i], (d - 1) % 4))
                if all(p in walls for p in pair):
                    candidates.append(pair)
        if candidates and (rng.random() < .65 or force_gap):
            for q in candidates[int(rng.integers(len(candidates)))]:
                del walls[q]
            gaps += 1
    g = np.full((h, w), bg, dtype=np.int8)
    for p, color in walls.items():
        g[p] = color
    for obj in relays:
        for p in obj['cells']:
            g[p] = marker if p == obj['port'] else obj['key']
    for p in launcher[:-1]:
        g[p] = signal
    g[cap] = marker
    scene = _decode(g)
    final_path = paths[-1]
    final_socket = None
    uniform_foreign = gate_type == 'foreign' and active_count >= 2
    for i in range(1, len(final_path)):
        d = _direction(final_path[i - 1], final_path[i])
        pair = _flanks(scene, final_path[i], d)
        if pair:
            final_socket = i, pair
            if uniform_foreign:
                break
    _require(final_socket is not None, 'terminal admission socket')
    i, pair = final_socket
    key = departure_keys[-1]
    other = int(rng.choice([c for c in keys if c != key]))
    if gate_type == 'foreign':
        if uniform_foreign:
            for p in final_walls:
                if p in walls:
                    g[p] = other
        g[pair[0]] = g[pair[1]] = other
    else:
        g[pair[0]], g[pair[1]] = (key, other) if rng.integers(2) else (other, key)
    terminal = final_path[i - 1]
    _require(g[terminal] == bg, 'free parking cell before socket')
    # Ground truth comes from the construction plan, not solve().
    out = g.copy()
    for p in launcher:
        out[p] = bg
    for j, obj in enumerate(relays):
        for p in obj['cells']:
            out[p] = signal if j < active_count else bg
    out[terminal] = signal
    a = _analyse(g)
    _require(np.array_equal(a['output'], out), 'planned answer disagrees with input reader')
    _require(len(a['reached']) == active_count and len(a['turns']) == elbow_count, 'planned route differs')
    _require(any(t in (1, 3) for t in a['port_turns']), 'relay port must redirect')
    _require(active_count <= len(a['sockets']) <= active_count + 1, 'admitted sockets on relay legs')
    _require(not force_gap or _gap_count(a) >= 1, 'demonstrated interrupted straight rails')
    # Every scene defeats an ablation of rail identity using an actual extra hit.
    ablated = _analyse(g, ignore_rails=True)
    _require(len(ablated['reached']) > active_count, 'rail ablation must reach blocked relay')
    target_index = ablated['reached'][active_count]
    _require(_has_outgoing_socket(a, a['objects'][target_index]), 'blocked target retains its own usable output rails')
    rotation = int(rng.integers(4))
    if rng.integers(2):
        g, out = np.fliplr(g), np.fliplr(out)
    return dict(input=np.rot90(g, rotation).copy(), output=np.rot90(out, rotation).copy())


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _count(num_train, 'num_train', 2, 10)
    _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    maximum_train = int(rng.integers(2, 4))
    vary_palette = bool(rng.integers(2))
    palette = list(map(int, rng.permutation(10)))
    key_count = int(rng.integers(2, 6))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            active = (1 if index == 0 else maximum_train if index == 1 else int(rng.integers(1, maximum_train + 1))) if split == 'train' else int(rng.integers(maximum_train + 1, min(5, maximum_train + 2) + 1))
            elbows = 1 if split == 'train' else int(rng.integers(1, min(2, 8 - active) + 1))
            gate = ('mismatch' if index == 0 else 'foreign') if split == 'train' and index < 2 else str(rng.choice(['foreign', 'mismatch']))
            colors = (list(map(int, rng.permutation(10))) if vary_palette else palette)[:3 + key_count]
            last_error = None
            for _ in range(120):
                try:
                    pair = _construct_pair(rng, active, elbows, gate, colors,
                                           force_gap=split == 'train' and index == 0)
                    key = _signature(pair['input'])
                    _require(key not in seen, 'duplicate proposal')
                except AssertionError as error:
                    last_error = str(error)
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError(f'bounded layout search exhausted: {last_error}')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, 'pair counts')
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'same canvas')
            sig = _signature(inp)
            _require(sig not in seen, 'distinct inputs')
            seen.add(sig)
            a = _analyse(inp)
            _require(np.array_equal(a['output'], out), 'incorrect output')
            _require(1 <= len(a['reached']) <= 5 and a['turns'], 'relay and rail routing')
            _require(any(t in (1, 3) for t in a['port_turns']), 'port redirection')
            _require(a['rejected'] is not None, 'visible rejected socket')
            _require(len(a['reached']) < len(a['objects']), 'unreachable relay')
            _require(len(a['reached']) <= len(a['sockets']) <= len(a['reached']) + 1, 'admission on relay legs')
            for p in a['guides']:
                _require(out[p] == inp[p], 'preserve every rail cell')
            _require(all(out[p] == a['bg'] for p in a['launcher']), 'courier leaves launcher')
            expected_signal = {a['terminal']}
            for i in a['reached']:
                expected_signal.update(a['objects'][i]['cells'])
            actual_new = {p for p in map(tuple, np.argwhere(out == a['signal'])) if p not in a['guides']}
            _require(actual_new == expected_signal, 'footprints plus one parked courier; no trail')
            # Omitting A: preactivate every relay. The blocked target must differ.
            unreachable = [o for i, o in enumerate(a['objects']) if i not in a['reached']]
            _require(all(out[p] == a['bg'] for o in unreachable for p in o['cells']), 'A reachability matters')
            # Omitting B: accept all two-sided openings regardless of colors.
            wrong = _analyse(inp, ignore_rails=True)
            _require(len(wrong['reached']) > len(a['reached']) and not np.array_equal(wrong['output'], out), 'B rail identity matters')
            blocked_index = wrong['reached'][len(a['reached'])]
            _require(_has_outgoing_socket(a, a['objects'][blocked_index]), 'blocked target has keyed outgoing rails')
            records[split].append(a)
    train = records['train']
    _require(len({len(a['reached']) for a in train}) >= 2, 'demonstrated chain-count variation')
    kinds = {'foreign' if a['rejected'][1][0] == a['rejected'][1][1] else 'mismatch' for a in train}
    _require(kinds == {'foreign', 'mismatch'}, 'demonstrate equality and loaded-key requirement')
    _require(any(a['rejected'][1][0] == a['rejected'][1][1] and len(a['sockets']) == len(a['reached']) for a in train), 'wrong key at first outgoing socket')
    _require(any(_gap_count(a) for a in train), 'training demonstrates interrupted support')
    _require(all(len(a['reached']) > max(len(b['reached']) for b in train) for a in records['test']), 'longer test chains')
    return None


def self_test():
    """42 seeds, streamed summaries, mutations and count boundaries; no I/O."""
    signatures, structures, shapes, roles, hit_counts, footprints = set(), set(), set(), set(), set(), set()
    port_offsets, turn_senses, gate_kinds, terminal_cells = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                before = inp.copy()
                _require(np.array_equal(solve(inp), out) and np.array_equal(inp, before), 'pure input-only solver')
                a = _analyse(inp)
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                shapes.add(inp.shape)
                roles.add((a['bg'], a['signal'], a['marker']))
                hit_counts.add(len(a['reached']))
                footprints.update(len(o['cells']) for o in a['objects'])
                turn_senses.update((after - before) % 4 for p, before, after, key in a['turns'])
                gate_kinds.add('foreign' if a['rejected'][1][0] == a['rejected'][1][1] else 'mismatch')
                terminal_cells.add(a['terminal'])
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add((inp.shape, hashlib.sha256(normalized).digest()))
                for o in a['objects']:
                    rs, cs = zip(*o['cells'])
                    offset = 2 * o['port'][1] - min(cs) - max(cs) if o['direction'] % 2 == 0 else 2 * o['port'][0] - min(rs) - max(rs)
                    port_offsets.add(int(offset))
        signatures.add(digest.digest())
    _require(len(signatures) == 42 and len(structures) >= 180, 'structural variation across 42 seeds')
    _require(len(shapes) >= 30 and len(roles) >= 30, 'size and palette variation')
    _require({1, 2, 3, 4, 5} <= hit_counts and len(footprints) >= 6, 'count and footprint variation')
    _require({1, 3} <= turn_senses and len(port_offsets) >= 3, 'turn senses and port alignment variation')
    _require(gate_kinds == {'foreign', 'mismatch'} and len(terminal_cells) > 30, 'gate/terminal variation')
    # Every legal explicit combination, with only one puzzle retained at a time.
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            validate_generated_puzzle(generate_puzzle(817 + n * 7 + m, n, m))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    p, q = generate_puzzle(103, 3, 2), generate_puzzle(103, 3, 2)
    _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in ('input', 'output')), 'determinism')
    # General reader also supports free boundary termination and straight gaps.
    g = np.zeros((12, 20), dtype=np.int8)
    g[6, 0], g[6, 1] = 1, 9
    g[5:8, 7:10] = 2
    g[6, 9] = 9
    g[5, 3:6] = g[7, 3:6] = 1
    g[5, 12:16] = g[7, 12:16] = 2
    a = _analyse(g)
    _require(a['terminal'] == (6, 19) and len(a['reached']) == 1, 'open boundary stop')
    g[5, 13:15] = g[7, 13:15] = 0
    _require(np.array_equal(solve(g), a['output']) is False, 'rails themselves remain visible')
    _require(_analyse(g)['terminal'] == (6, 19), 'straight gap inertia')
    # Replacing the carried key with a fixed launch color must fail here.
    g[5, 12:16] = g[7, 12:16] = 1
    b = _analyse(g)
    _require(b['terminal'] == (6, 11) and b['rejected'][2] == 2, 'relay rekeys courier')
    bad = generate_puzzle(19, 2, 1)
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('mutated answer accepted')
    return {'seeds': 42, 'pairs': 210, 'distinct_puzzles': len(signatures),
            'color_normalized_structures': len(structures), 'shapes': len(shapes),
            'role_assignments': len(roles), 'active_counts': sorted(hit_counts),
            'footprint_sizes': sorted(footprints), 'port_offsets': sorted(port_offsets)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
