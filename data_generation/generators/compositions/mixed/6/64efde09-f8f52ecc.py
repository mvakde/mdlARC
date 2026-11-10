"""Registered emitter ports routed by shortest length, then fewest bends.

All inference uses only the input. Search is a finite checkpoint-progress graph;
the family admits only unique relaxed optima which are induced simple paths.
No search histories, puzzle collections, caches or execution traces are saved.
"""
from heapq import heappop, heappush
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 302571558, 971963134, 733770511, 287463382, 136240713,
              181960619, 503281118, 800266256, 433256317, 705204161,
              441401833, 754808772, 558746057, 581297494, 280278567,
              371854400, 430290809, 263089660, 340761312, 183895959,
              95415971, 141487120, 815615683, 966167036, 474394317,
              876154705, 92183105, 580975712, 856003420, 684200061,
              778793341, 391041877, 646792007, 923609762, 873268316,
              427084661, 99170953, 740417459, 667964473, 745489777,
              521696987)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _neighbors(p, shape):
    for dr, dc in DIRS:
        q = (p[0] + dr, p[1] + dc)
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        group, stack = {start}, [start]
        while stack:
            for q in _neighbors(stack.pop(), mask.shape):
                if q in unseen:
                    unseen.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _port(emitter, side, slot, length):
    labels = emitter['labels']
    a = tuple(map(int, np.argwhere(labels == side * length + slot)[0]))
    b = tuple(map(int, np.argwhere(labels == (1 - side) * length + slot)[0]))
    direction = (a[0] - b[0], a[1] - b[1])
    return (emitter['r'] + a[0] + direction[0],
            emitter['c'] + a[1] + direction[1]), direction


def _ray(port, direction, shape):
    result = []
    r, c = port
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        result.append((r, c))
        r, c = r + direction[0], c + direction[1]
    return tuple(result)


def _decode(a):
    """Parent A: segment, register exact D4 copies, pool boundary slot clues."""
    colors, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous background")
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > a.size // 2, "background must be a majority")
    emitters, clues = [], []
    for cells in _components(a != bg):
        if len(cells) == 1:
            p = next(iter(cells))
            clues.append((p, int(a[p])))
        else:
            rs, cs = zip(*cells)
            r, c = min(rs), min(cs)
            arr = a[r:max(rs) + 1, c:max(cs) + 1]
            _require(arr.size == len(cells) and min(arr.shape) == 2
                     and 4 <= max(arr.shape) <= 10, "invalid emitter")
            emitters.append({'r': r, 'c': c, 'arr': arr.copy(), 'cells': cells})
    _require(2 <= len(emitters) <= 5 and 1 <= len(clues) <= 3,
             "need 2..5 emitters and 1..3 clues")
    # Each clue supplies its own source edge and hence its own sweep axis.
    edges = []
    h, w = a.shape
    for p, _ in clues:
        matches = [i for i, yes in enumerate((p[0] == 0, p[1] == w - 1,
                                               p[0] == h - 1, p[1] == 0)) if yes]
        _require(len(matches) == 1, "clues must lie on an edge, away from corners")
        edges.append(matches[0])
    structural = set(map(int, emitters[0]['arr'].flat))
    _require(2 <= len(structural) <= 4 and not structural.intersection(c for _, c in clues),
             "structural and signal palettes must be disjoint")
    canonical = emitters[0]['arr']
    if canonical.shape[0] != 2:
        canonical = np.rot90(canonical)
    length = canonical.shape[1]
    forms = [_orient(canonical, t) for t in range(8)]
    _require(len({_key(form) for form in forms}) == 8, "symmetric template")
    labels = np.arange(2 * length).reshape(2, length)
    for emitter in emitters:
        matches = [t for t, form in enumerate(forms)
                   if np.array_equal(emitter['arr'], form)]
        _require(len(matches) == 1, "copies must have unique exact registration")
        emitter['t'] = matches[0]
        emitter['labels'] = _orient(labels, matches[0])
        # Doubled center avoids any floating point or even-length ambiguity.
        center = (2 * emitter['r'] + emitter['arr'].shape[0] - 1,
                  2 * emitter['c'] + emitter['arr'].shape[1] - 1)
        emitter['center'] = center
    occupied = set().union(*(e['cells'] for e in emitters))
    code, sides, allocations = {}, set(), [set() for _ in emitters]
    for (endpoint, color), edge in zip(clues, edges):
        interpretations = []
        for i, emitter in enumerate(emitters):
            for side in (0, 1):
                for slot in range(length):
                    port, direction = _port(emitter, side, slot, length)
                    ray = _ray(port, direction, a.shape)
                    if ray and ray[-1] == endpoint and all(int(a[p]) == bg for p in ray[:-1]):
                        interpretations.append((i, side, slot))
        _require(len(interpretations) == 1, "ambiguous boundary attribution")
        i, side, slot = interpretations[0]
        _require(slot not in code and color not in (entry[0] for entry in code.values()),
                 "exactly one clue and one distinct color per occupied slot")
        code[slot] = (color, endpoint, edge)
        sides.add(side)
        allocations[i].add(slot)
    _require(len(sides) == 1, "inconsistent template-local active side")
    side = sides.pop()
    routes = []
    for slot, (color, endpoint, edge) in sorted(code.items()):
        inward = DIRS[(edge + 2) % 4]
        depth = lambda e: e['center'][0] * inward[0] + e['center'][1] * inward[1]
        _require(len({depth(e) for e in emitters}) == len(emitters), "tied emitter depths")
        ordered = sorted(emitters, key=depth)
        ports, directions = zip(*(_port(e, side, slot, length) for e in ordered))
        _require(all(0 <= p[0] < h and 0 <= p[1] < w and int(a[p]) == bg for p in ports),
                 "ports must be free interior cells")
        _require(all(0 < p[0] < h - 1 and 0 < p[1] < w - 1 for p in ports),
                 "ports must be strictly interior")
        _require(directions[-1] == inward, "terminal emitter must face away from source edge")
        tail = _ray(ports[-1], directions[-1], a.shape)
        _require(len(tail) >= 2 and all(int(a[p]) == bg for p in tail), "blocked terminal ray")
        routes.append({'color': color, 'slot': slot, 'checkpoints': (endpoint,) + ports,
                       'tail': tail, 'directions': directions, 'edge': edge})
    all_ports = [p for route in routes for p in route['checkpoints']]
    _require(len(all_ports) == len(set(all_ports)), "overlapping checkpoints")
    tails = [set(route['tail']) for route in routes]
    _require(all(not tails[i].intersection(tails[j]) for i in range(len(tails))
                 for j in range(i)), "overlapping terminal rays")
    for i, route in enumerate(routes):
        foreign = set().union(*(set(r['checkpoints']) | set(r['tail'])
                                for j, r in enumerate(routes) if j != i))
        _require(not set(route['checkpoints']).intersection(foreign), "foreign checkpoint on ray")
        # Own tail is reserved, including its adjacency halo: it must not create
        # a same-color shortcut/branch when appended to the completed main path.
        tail_halo = set(route['tail'][1:])
        for p in route['tail'][1:]:
            tail_halo.update(_neighbors(p, a.shape))
        tail_halo.discard(route['checkpoints'][-1])
        blocked = occupied | foreign | tail_halo
        # Boundary cells other than the one source are used only by fixed tails.
        blocked.update((r, c) for r in range(h) for c in range(w)
                       if r in (0, h - 1) or c in (0, w - 1))
        blocked.discard(route['checkpoints'][0])
        _require(not blocked.intersection(route['checkpoints']), "blocked own checkpoint")
        route['blocked'] = blocked
    return {'background': bg, 'emitters': emitters, 'routes': routes,
            'length': length, 'allocations': tuple(map(len, allocations)),
            'structural_count': len(structural)}


def _route_search(shape, checkpoints, blocked, objective='length_bends'):
    """Exact lexicographic Dijkstra on (cell, checkpoint progress, heading).

    Returns one optimal walk, physical (edges, bends), and number of optimal
    walks capped at two. If a relaxed optimum is unique and induced/simple,
    it certifies the optimum among nonbranching paths as well. Ambiguous or
    self-touching relaxed cases are outside this family's input constraints.
    """
    h, w = shape
    points = tuple(r * w + c for r, c in checkpoints)
    owner = {p: i for i, p in enumerate(points)}
    walls = {r * w + c for r, c in blocked}
    start = (points[0], 0, 4)
    dist, ways, parent = {start: (0, 0)}, {start: 1}, {}
    queue = [(0, 0, start)]
    best, ends = None, []
    while queue:
        primary, secondary, state = heappop(queue)
        cost = (primary, secondary)
        if cost != dist[state]:
            continue
        if best is not None and cost > best:
            break
        pos, progress, direction = state
        if progress == len(points) - 1:
            best = cost
            ends.append(state)
            continue
        r, c = divmod(pos, w)
        for nd, (dr, dc) in enumerate(DIRS):
            rr, cc = r + dr, c + dc
            q = rr * w + cc
            if not (0 <= rr < h and 0 <= cc < w) or q in walls:
                continue
            index = owner.get(q)
            if index is not None and index != progress + 1:
                continue
            new_progress = progress + 1 if index is not None else progress
            bend = int(direction != 4 and direction != nd)
            if objective == 'length_bends':
                increment = (1, bend)
            elif objective == 'bends_length':
                increment = (bend, 1)
            else:
                increment = (1, 0)
            new_cost = (primary + increment[0], secondary + increment[1])
            new_state = (q, new_progress, nd)
            old = dist.get(new_state)
            if old is None or new_cost < old:
                dist[new_state] = new_cost
                ways[new_state] = ways[state]
                parent[new_state] = state
                heappush(queue, (*new_cost, new_state))
            elif new_cost == old:
                ways[new_state] = min(2, ways[new_state] + ways[state])
    if not ends:
        return None, None, 0
    count = min(2, sum(ways[state] for state in ends))
    state = ends[0]
    path = []
    while True:
        path.append(divmod(state[0], w))
        if state == start:
            break
        state = parent[state]
    path.reverse()
    return tuple(path), (len(path) - 1, _bends(path)), count


def _bends(path):
    ds = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _simple(path, shape):
    cells = set(path)
    if len(cells) != len(path):
        return False
    for i, p in enumerate(path):
        expected = set(path[max(0, i - 1):i] + path[i + 1:i + 2])
        if set(_neighbors(p, shape)).intersection(cells) != expected:
            return False
    return True


def _complete(a):
    info = _decode(a)
    output, painted = a.copy(), set()
    for route in info['routes']:
        path, cost, multiplicity = _route_search(a.shape, route['checkpoints'], route['blocked'])
        _require(multiplicity == 1, "route has no unique length-then-bend optimum")
        full = path + route['tail'][1:]
        _require(_simple(full, a.shape), "relaxed optimum is not an induced nonbranching path")
        _require(not painted.intersection(full), "independently optimal colors collide")
        for p in full:
            _require(int(a[p]) in (info['background'], route['color']), "route overwrites input")
            output[p] = route['color']
        painted.update(full)
        route['path'], route['cost'] = path, cost
        route['full'] = full
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve a family member from its pixels; invalid/tied inputs raise ValueError."""
    return _complete(_grid(input_grid))[0]


def _template(rng, length, colors):
    for _ in range(64):
        a = rng.choice(colors, (2, length)).astype(np.int8)
        if rng.random() < .5:
            a[0] = colors[0]
            a[1] = colors[1]
            a[1, :int(rng.integers(1, length))] = colors[-1]
        for i, color in enumerate(colors):
            a[1, i] = color
        if len({_key(_orient(a, t)) for t in range(8)}) == 8:
            return a
    # Asymmetric for every allowed length and palette.
    a = np.full((2, length), colors[0], dtype=np.int8)
    a[1, 1:] = colors[1]
    for i, color in enumerate(colors[2:], 2):
        a[1, i] = color
    return a


def _candidate(rng, count, slot_count, mixed, pooled, longer=False):
    """Sample geometry, then let the input-only optimization define the answer."""
    length = int(rng.integers(9, 11) if longer else rng.integers(4, 9))
    palette = list(map(int, rng.permutation(10)))
    bg = palette[0]
    structural_count = int(rng.integers(2, 5))
    template = _template(rng, length, palette[1:1 + structural_count])
    signal_colors = palette[1 + structural_count:1 + structural_count + slot_count]
    if slot_count == 2:
        if rng.random() < .5:
            slots = tuple(sorted(map(int, rng.choice(length, 2, replace=False))))
        else:
            start = int(rng.integers(length - 1))
            slots = (start, start + 1)
    else:
        slots = (int(rng.integers(length)),)
    transforms = [0] * (count - 1) + [6]  # row reflection at the terminal
    if slot_count == 1:
        for i in range(1, count - 1):
            if i == 1 and mixed and count <= 4 and rng.random() < .65:
                transforms[i] = int(rng.choice((1, 3, 5, 7)))
            else:
                transforms[i] = int(rng.choice((2, 6)))
        if rng.random() < .5:
            transforms[-1] = 2  # reverse longitudinal slot order too
    first_row = int(rng.integers(2, 5))
    rows, row = [], first_row
    for t in transforms:
        rows.append(row)
        row += _orient(template, t).shape[0] + int(rng.integers(2, 4 if count == 5 else 5))
    height = row + int(rng.integers(1, 4))
    if height > 30:
        return None
    width = int(rng.integers(max(12, length + 7), min(28, length + 16)))
    base_col = int(rng.integers(2, width - length - 2))
    cols = [base_col]
    for i in range(1, count):
        ew = _orient(template, transforms[i]).shape[1]
        col = (int(rng.integers(2, width - ew - 2)) if rng.random() < .5
               else base_col + int(rng.integers(-3, 4)))
        cols.append(max(2, min(width - ew - 3, col)))
    a = np.full((height, width), bg, dtype=np.int8)
    emitters = []
    for row, col, t in zip(rows, cols, transforms):
        arr = _orient(template, t)
        eh, ew = arr.shape
        if col + ew >= width - 1:
            return None
        a[row:row + eh, col:col + ew] = arr
        emitters.append({'r': row, 'c': col, 'labels': _orient(np.arange(2 * length).reshape(2, length), t)})
    for j, (slot, color) in enumerate(zip(slots, signal_colors)):
        owner = j if pooled else 0
        port, direction = _port(emitters[owner], 0, slot, length)
        if direction not in ((-1, 0), (1, 0)):
            return None
        endpoint = (0 if direction[0] < 0 else height - 1, port[1])
        if not all(int(a[p]) == bg for p in _ray(port, direction, a.shape)):
            return None
        a[endpoint] = color
    return _orient(a, int(rng.integers(8)))


def _foreign_sensitive(a, info):
    for route in info['routes']:
        foreign = set().union(*(set(r['checkpoints'][1:]) for r in info['routes'] if r is not route))
        path, cost, count = _route_search(a.shape, route['checkpoints'], route['blocked'] - foreign)
        if path is not None and set(path).intersection(foreign) and cost < route['cost']:
            return True
    return False


def _make_pair(rng, count, slot_count, mixed=False, pooled=False, require_axis_mix=False,
               require_foreign=False, longer=False):
    for _ in range(350):
        a = _candidate(rng, count, slot_count, mixed, pooled, longer)
        if a is None:
            continue
        try:
            output, info = _complete(a)
        except ValueError:
            continue
        if require_axis_mix and len({e['arr'].shape[0] == 2 for e in info['emitters']}) < 2:
            continue
        if require_foreign and not _foreign_sensitive(a, info):
            continue
        if pooled and not (sum(n > 0 for n in info['allocations']) >= 2
                           and max(info['allocations']) < slot_count):
            continue
        return {'input': a, 'output': output}
    raise RuntimeError("bounded sampling exhausted: no unambiguous compatible layout")


def _count(value, name, low, high):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
             and low <= int(value) <= high, f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', num_train), ('test', num_test)):
        for i in range(n):
            if split == 'train' and i == 0:
                count, slot_count, mixed, pooled = 2, 2, False, True
            elif split == 'train' and i == 1:
                count, slot_count, mixed, pooled = 3, 1, True, False
            else:
                slot_count = int(rng.integers(1, 3))
                count = (int(rng.integers(2, 4)) if split == 'train'
                         else int(rng.integers(4, 6))) if slot_count == 1 else 2
                mixed, pooled = slot_count == 1, slot_count == 2
            for _ in range(8):
                try:
                    pair = _make_pair(rng, count, slot_count, mixed, pooled,
                                      require_axis_mix=split == 'train' and i == 1,
                                      require_foreign=split == 'train' and i == 0,
                                      longer=split == 'test' and slot_count == 2)
                except RuntimeError as error:
                    raise RuntimeError(f"seed {seed}, {split} {i}, emitters {count}, slots {slot_count}: {error}") from error
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain distinct inputs")
    return puzzle


def _features(a, info, diagnostic=False):
    result = {'shape': tuple(a.shape), 'background': info['background'],
              'emitters': len(info['emitters']), 'slots': len(info['routes']),
              'length': info['length'], 'edges': tuple(r['edge'] for r in info['routes']),
              'adjacent_slots': len(info['routes']) > 1 and any(
                  b['slot'] - a['slot'] == 1 for a, b in zip(info['routes'], info['routes'][1:])),
              'structural_colors': info['structural_count'],
              'pooled': max(info['allocations']) < len(info['routes']),
              'mixed_axes': len({e['arr'].shape[0] == 2 for e in info['emitters']}) > 1,
              'bends': tuple(r['cost'][1] for r in info['routes']),
              'detour': False, 'bend_tie_broken': False, 'length_beats_bends': False}
    result['foreign_sensitive'] = diagnostic and _foreign_sensitive(a, info)
    for route in info['routes']:
        cps = route['checkpoints']
        manhattan = sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p, q in zip(cps, cps[1:]))
        result['detour'] |= bool(route['cost'][0] > manhattan)
        if diagnostic:
            _, _, count = _route_search(a.shape, cps, route['blocked'], 'length_only')
            result['bend_tie_broken'] |= count > 1
            path, cost, _ = _route_search(a.shape, cps, route['blocked'], 'bends_length')
            result['length_beats_bends'] |= (path is not None and _simple(path + route['tail'][1:], a.shape)
                                             and cost[0] > route['cost'][0]
                                             and cost[1] < route['cost'][1])
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _count(nt, 'num_train', 2, 10)
    _count(ne, 'num_test', 1, 4)
    _require(nt > ne, "num_train must exceed num_test")
    seen, train_features = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape and _key(a) not in seen, "duplicate input or changed shape")
            seen.add(_key(a))
            expected, info = _complete(a)
            _require(np.array_equal(expected, b), "incorrect completed output")
            _require(np.array_equal(a[a != info['background']], b[a != info['background']]),
                     "input marks were changed")
            for route in info['routes']:
                _require(set(map(tuple, np.argwhere(b == route['color']))) == set(route['full']),
                         "extra/missing colored path cells")
                positions = {p: i for i, p in enumerate(route['full'])}
                order = [positions[p] for p in route['checkpoints']]
                _require(order == sorted(order), "checkpoint order changed")
            if split == 'train':
                feature = _features(a, info)
                feature['foreign_sensitive'] = _foreign_sensitive(a, info)
                train_features.append(feature)
    _require(any(f['pooled'] and f['slots'] >= 2 for f in train_features),
             "training must demonstrate distributed pooling")
    _require(any(f['detour'] for f in train_features), "training must demonstrate obstacle detours")
    _require(any(f['mixed_axes'] for f in train_features), "training must demonstrate local quarter-turn transfer")
    _require(any(f['foreign_sensitive'] for f in train_features), "training must expose foreign virtual ports")
    return True


def self_test():
    """42 seeds, in-memory variation/ablation checks, and bounded edge cases."""
    features, signatures = [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for p in puzzle['train'] + puzzle['test']))
        for pair in puzzle['train'] + puzzle['test']:
            a = pair['input']
            expected, info = _complete(a.copy())
            _require(np.array_equal(expected, pair['output']), "fresh-input inference disagrees")
            features.append(_features(a, info, diagnostic=True))
            # Removing A gives B one visible checkpoint per color: no extension.
            _require(not np.array_equal(a, expected), "A omission has no effect")
            # Removing B gives A's broadcast rays; a completed routed cell must
            # exist outside that straight-ray set, proving a real interaction.
            a_only = a.copy()
            for route in info['routes']:
                for port, direction in zip(route['checkpoints'][1:], route['directions']):
                    for p in _ray(port, direction, a.shape):
                        if int(a_only[p]) in (info['background'], route['color']):
                            a_only[p] = route['color']
            _require(not np.array_equal(a_only, expected), "B omission has no effect")
    _require(len(signatures) == 42, "seeds lack puzzle diversity")
    _require(len({f['shape'] for f in features}) >= 30, "too few shapes")
    _require({f['background'] for f in features} == set(range(10)), "palette roles not varied")
    _require({f['length'] for f in features} >= {4, 5, 6, 7, 8}, "template lengths not varied")
    _require({f['emitters'] for f in features} >= {2, 3, 4}, "emitter counts not varied")
    _require({f['slots'] for f in features} == {1, 2}, "slot counts not varied")
    _require({e for f in features for e in f['edges']} == set(range(4)), "source edges not varied")
    _require(any(f['mixed_axes'] for f in features), "no local quarter-turn transfer")
    _require(sum(f['foreign_sensitive'] for f in features) >= 42, "foreign checkpoints unexercised")
    _require(sum(f['bend_tie_broken'] for f in features) >= 42, "bend criterion unexercised")
    # Recoloring and all eight whole-grid isometries preserve the rule.
    small = generate_puzzle(20261023, 2, 1)
    recolor = np.array([3, 6, 1, 8, 0, 9, 2, 7, 5, 4], dtype=np.int8)
    for t in range(8):
        for pair in small['train'] + small['test']:
            _require(np.array_equal(solve(_orient(recolor[pair['input']], t)),
                                    _orient(recolor[pair['output']], t)), "isometry/recoloring failed")
    again = generate_puzzle(20261023, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for split in small
                 for p, q in zip(small[split], again[split]) for k in p), "nondeterminism")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in small[s]] for s in small}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("output corruption accepted")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1234 + 10 * nt + ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, "wrong counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {'seeds': 42, 'pairs': len(features),
            'shapes': len({f['shape'] for f in features}),
            'template_lengths': sorted({f['length'] for f in features}),
            'emitter_counts': sorted({f['emitters'] for f in features}),
            'structural_color_counts': sorted({f['structural_colors'] for f in features}),
            'source_edges': sorted({e for f in features for e in f['edges']}),
            'background_colors': sorted({f['background'] for f in features}),
            'pooled_pairs': sum(f['pooled'] for f in features),
            'adjacent_slot_pairs': sum(f['adjacent_slots'] for f in features),
            'foreign_sensitive_pairs': sum(f['foreign_sensitive'] for f in features),
            'mixed_axis_pairs': sum(f['mixed_axes'] for f in features),
            'detour_pairs': sum(f['detour'] for f in features),
            'bend_tie_pairs': sum(f['bend_tie_broken'] for f in features),
            'length_priority_pairs': sum(f['length_beats_bends'] for f in features)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
