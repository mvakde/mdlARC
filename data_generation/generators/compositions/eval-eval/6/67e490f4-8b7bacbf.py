"""Shape-selective, conserved ballots competing for hollow collectors.

No file I/O, saved traces, network access, or parent imports. Arrays and bounded
searches are held in memory. See task_spec.md for the complete input grammar.
"""
from collections import Counter, deque
from functools import lru_cache
import hashlib
import json
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _norm(cells):
    cells = tuple(cells)
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


@lru_cache(maxsize=2048)
def _variants(shape, reflection=True):
    result = set()
    for mirror in range(2 if reflection else 1):
        s = _norm((r, -c if mirror else c) for r, c in shape)
        for _ in range(4):
            result.add(s)
            s = _norm((c, -r) for r, c in s)
    return tuple(sorted(result))


def _signature(cells, mode="shape"):
    s = _norm(cells)
    if mode == "area":
        return len(s)
    if mode == "bbox":
        return tuple(sorted((max(r for r, c in s)+1, max(c for r, c in s)+1)))
    if mode == "translation":
        return s
    return min(_variants(s, mode != "rotation"))


def _neighbors(cells, steps=D4):
    return {(r+dr, c+dc) for r, c in cells for dr, dc in steps} - set(cells)


def _components(cells):
    left = set(cells)
    result = []
    while left:
        first = min(left)
        left.remove(first)
        group = {first}
        queue = [first]
        for r, c in queue:
            for dr, dc in D4:
                q = (r+dr, c+dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    queue.append(q)
        result.append(group)
    return result


def _shell(cells):
    return _neighbors(cells, D8)


def _distance(wire, start):
    distances = {start: 0}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for dr, dc in D4:
            q = (p[0]+dr, p[1]+dc)
            if q in wire and q not in distances:
                distances[q] = distances[p]+1
                queue.append(q)
    return distances


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _analyze(input_grid):
    """Read colors, shapes, unique ports and wire geometry from input only."""
    a = _grid(input_grid)
    h, w = a.shape
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)
    blobs = _components(zip(*np.where(a == bg)))
    enclosed = [b for b in blobs if not any(r in (0, h-1) or c in (0, w-1) for r, c in b)]
    candidates = {}
    for cells in enclosed:
        colors = {int(a[p]) for p in _neighbors(cells)}
        if len(colors) == 1:
            color = colors.pop()
            candidates.setdefault(color, []).append(cells)
    matches = []
    for color, holes in candidates.items():
        expected = set().union(*(_shell(c) for c in holes))
        actual = set(zip(*np.where(a == color)))
        if expected == actual:
            matches.append((color, holes))
    assert len(matches) == 1, "Exactly one color forms the collectors' complete shells"
    shell_color, holes = matches[0]
    shells = [_shell(c) for c in holes]
    all_shells = set().union(*shells)
    adjacent = _neighbors(all_shells) - set().union(*holes)
    wire_colors = {int(a[p]) for p in adjacent if a[p] not in (bg, shell_color)}
    assert len(wire_colors) == 1, "All collector ports have one local wire color"
    wire_color = wire_colors.pop()
    wire = set(zip(*np.where(a == wire_color)))
    circuit_cells = _components(wire)
    membership = {p: i for i, cells in enumerate(circuit_cells) for p in cells}
    collectors = []
    for cells, shell in zip(holes, shells):
        ports = _neighbors(shell) & wire
        assert len(ports) == 1, "A collector has exactly one wire port"
        port = next(iter(ports))
        collectors.append(dict(cells=cells, port=port, circuit=membership[port],
                               shape=_signature(cells)))
    sources = []
    for color in range(10):
        if color in (bg, shell_color, wire_color):
            continue
        for cells in _components(zip(*np.where(a == color))):
            ports = _neighbors(cells) & wire
            assert len(ports) == 1, "A source has exactly one wire port"
            port = next(iter(ports))
            sources.append(dict(cells=cells, port=port, circuit=membership[port],
                                shape=_signature(cells), color=color))
    assert collectors and sources
    assert all(any(c['circuit'] == i for c in collectors) for i in range(len(circuit_cells)))
    return dict(grid=a, bg=bg, shell_color=shell_color, wire_color=wire_color,
                wire=wire, collectors=collectors, sources=sources,
                circuits=circuit_cells)


def _recipients(data, mode="shape", metric="wire", scope="circuit"):
    collectors, sources = data['collectors'], data['sources']
    distances = [_distance(data['wire'], c['port']) for c in collectors]
    result = []
    for source in sources:
        options = []
        for i, target in enumerate(collectors):
            if scope == "circuit" and source['circuit'] != target['circuit']:
                continue
            if mode != "any" and _signature(source['cells'], mode) != _signature(target['cells'], mode):
                continue
            if metric == "wire":
                d = distances[i].get(source['port'])
                if d is None:
                    continue
            elif metric == "euclidean":
                d = sum((x-y)**2 for x, y in zip(source['port'], target['port']))
            else:
                d = 0
            options.append((d, i))
        if not options:
            result.append(None)
        else:
            best = min(d for d, i in options)
            winners = [i for d, i in options if d == best]
            result.append(winners[0] if len(winners) == 1 else -1)
    return result


def _prediction(data, mode="shape", metric="wire", broadcast=False, global_scope=False):
    ballots = [Counter() for _ in data['collectors']]
    if broadcast:
        for source in data['sources']:
            for i, target in enumerate(data['collectors']):
                if (global_scope or source['circuit'] == target['circuit']) and source['shape'] == target['shape']:
                    ballots[i][source['color']] += 1
    else:
        recipients = _recipients(data, mode, metric)
        for source, target in zip(data['sources'], recipients):
            if target is not None and target >= 0:
                ballots[target][source['color']] += 1
    out = data['grid'].copy()
    for target, counts in zip(data['collectors'], ballots):
        if not counts:
            continue
        high = max(counts.values())
        winners = [c for c, n in counts.items() if n == high]
        if len(winners) == 1:
            for p in target['cells']:
                out[p] = winners[0]
    return out, ballots


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The input supplies every role, circuit, shape, distance and ballot."""
    return _prediction(_analyze(input_grid))[0]


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


@lru_cache(maxsize=1)
def _pool():
    """All small free polyominoes, filtered for clean thin hollow outlines."""
    current = {((0, 0),)}
    pool = set(current)
    for _ in range(5):
        following = set()
        for s in sorted(current):
            for p in sorted(_neighbors(s)):
                following.add(_signature(s+(p,)))
        current = following
        pool.update(current)
    good = []
    for s in sorted(pool):
        if max(max(p) for p in s) > 3:
            continue
        shell = _shell(s)
        occupied = set(s) | shell
        r0 = min(r for r, c in occupied)-1
        c0 = min(c for r, c in occupied)-1
        r1 = max(r for r, c in occupied)+1
        c1 = max(c for r, c in occupied)+1
        empty = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)}-shell
        components = _components(empty)
        if len(components) == 2 and set(s) in components:
            good.append(s)
    return tuple(good)


def _port_candidates(cells):
    r0 = min(r for r, c in cells)
    r1 = max(r for r, c in cells)
    c0 = min(c for r, c in cells)
    c1 = max(c for r, c in cells)
    result = set()
    for r, c in cells:
        for dr, dc in D4:
            q = (r+dr, c+dc)
            if q[0] < r0 or q[0] > r1 or q[1] < c0 or q[1] > c1:
                result.add(q)
    return sorted(result)


def _pack(rng, objects, region):
    """Free coordinates and independently chosen outward ports; finite search."""
    blocked = set()
    reserved = set()
    placed = [None]*len(objects)
    order = list(map(int, rng.permutation(len(objects))))
    order.sort(key=lambda i: (objects[i]['kind'] == 'target', len(objects[i]['shape'])), reverse=True)
    for index in order:
        item = objects[index]
        success = False
        variants = list(_variants(item['shape']))
        rng.shuffle(variants)
        for shape in variants:
            occupied = set(shape) | (_shell(shape) if item['kind'] == 'target' else set())
            mr = min(r for r, c in occupied)
            mc = min(c for r, c in occupied)
            occupied = {(r-mr, c-mc) for r, c in occupied}
            inner = {(r-mr, c-mc) for r, c in shape}
            hh = max(r for r, c in occupied)+1
            ww = max(c for r, c in occupied)+1
            rlo = min(r for r, c in region)
            rhi = max(r for r, c in region)-hh+1
            clo = min(c for r, c in region)
            chi = max(c for r, c in region)-ww+1
            if rlo > rhi or clo > chi:
                continue
            for _ in range(90):
                r = int(rng.integers(rlo, rhi+1))
                c = int(rng.integers(clo, chi+1))
                cells = {(r+y, c+x) for y, x in occupied}
                if not cells <= region or cells & blocked or (cells | _neighbors(cells)) & reserved:
                    continue
                free = region-cells-blocked
                ports = []
                exits = {}
                for p in _port_candidates(cells):
                    if p not in free:
                        continue
                    # Reserve the straight outward step as well as the port.
                    # Later objects cannot seal this terminal inside a halo.
                    outward = [(dr, dc) for dr, dc in D4 if (p[0]-dr, p[1]-dc) in cells]
                    choices = [(p[0]+dr, p[1]+dc) for dr, dc in outward
                               if (p[0]+dr, p[1]+dc) in free]
                    if choices:
                        ports.append(p)
                        exits[p] = choices
                if not ports:
                    continue
                port = _pick(rng, ports)
                # Reserve a little space for the eventual wire to exit the port.
                placed[index] = dict(item, cells=cells, inner={(r+y, c+x) for y, x in inner}, port=port)
                blocked |= cells | _neighbors(cells, D8) | {port}
                reserved |= {port, _pick(rng, exits[port])}
                success = True
                break
            if success:
                break
        if not success:
            return None
    return placed


def _route(rng, objects, region):
    occupied = set().union(*(o['cells'] for o in objects))
    ports = {o['port'] for o in objects}
    allowed = region - occupied - _neighbors(occupied)
    allowed |= ports
    # A port must be the only wire pixel adjacent to its object.
    if any((_neighbors(o['cells']) & ports) != {o['port']} for o in objects):
        return None
    todo = [o['port'] for o in objects]
    rng.shuffle(todo)
    wire = {todo.pop()}
    steps = [D4[int(i)] for i in rng.permutation(4)]
    for start in todo:
        if start in wire:
            continue
        queue = deque([start])
        previous = {start: None}
        end = None
        while queue:
            p = queue.popleft()
            if p in wire:
                end = p
                break
            for dr, dc in steps:
                q = (p[0]+dr, p[1]+dc)
                if q in allowed and q not in previous:
                    previous[q] = p
                    queue.append(q)
        if end is None:
            return None
        while end is not None:
            wire.add(end)
            end = previous[end]
    assert all((_neighbors(o['cells']) & wire) == {o['port']} for o in objects)
    return wire


def _ballot_colors(rng, n, winner, colors):
    losers = [c for c in colors if c != winner]
    rng.shuffle(losers)
    if n <= 2:
        return [winner]*n
    patterns = {3: [(2, 1)], 4: [(2, 1, 1), (3, 1)],
                5: [(3, 2), (3, 1, 1)], 6: [(3, 2, 1), (4, 2)],
                7: [(3, 2, 2), (4, 2, 1)], 8: [(4, 3, 1), (3, 2, 2, 1)]}
    counts = _pick(rng, patterns.get(n, [(n//2+1, n-n//2-1)]))
    result = [c for c, count in zip([winner]+losers, counts) for _ in range(count)]
    rng.shuffle(result)
    return result


def _make_pair(rng, anchor, companion, lesson, shift, novel_shapes=()):
    if lesson == 0:
        target_specs = [(anchor, 0), (anchor, 0), (companion, 0)]
        source_specs = [(anchor, 0)]*8 + [(companion, 0)]*4
    elif lesson == 1:
        target_specs = [(anchor, 0), (companion, 0), (anchor, 1), (companion, 1)]
        source_specs = [(anchor, 0)]*4 + [(companion, 0)]*4 + [(anchor, 1)]*4
    else:
        pool = _pool()
        k = int(rng.integers(2, 5)) if shift else int(rng.integers(1, 4))
        shapes = [_pick(rng, pool) for _ in range(k)]
        shapes = list(dict.fromkeys(shapes))
        ncircuit = 2 if rng.random() < .3 else 1
        n = int(rng.integers(3, 6 if shift else 5))
        target_specs = [(_pick(rng, shapes), i % ncircuit) for i in range(n)]
        if novel_shapes:
            novel = _pick(rng, novel_shapes)
            target_specs[0] = (novel, 0)
            target_specs[1] = (novel, 0)
            if ncircuit == 2:
                target_specs[2] = (target_specs[2][0], 1)
        source_specs = []
        for s, circuit in target_specs:
            source_specs.extend([(s, circuit)]*int(rng.integers(2, 5)))
        # Keep packing bounded: at most 17 sources and five collectors.
        source_specs = source_specs[:17]
    ncircuit = max(c for s, c in target_specs)+1
    h = int(rng.integers(27, 31)) if lesson is not None or len(source_specs) > 10 else int(rng.integers(22, 31))
    w = int(rng.integers(27, 31)) if lesson is not None or len(source_specs) > 10 else int(rng.integers(22, 31))
    if ncircuit == 2:
        w = max(28, w)
    colors = list(map(int, rng.permutation(10)))
    bg, shell_color, wire_color = colors[:3]
    paint_colors = colors[3:]
    regions = []
    cut = int(rng.integers(15, w-10)) if ncircuit == 2 else w
    for circuit in range(ncircuit):
        c0, c1 = (1, w-2) if ncircuit == 1 else ((1, cut-1) if circuit == 0 else (cut+1, w-2))
        regions.append({(r, c) for r in range(1, h-1) for c in range(c0, c1+1)})
    objects = [dict(kind='target', shape=s, circuit=c) for s, c in target_specs]
    objects += [dict(kind='source', shape=s, circuit=c) for s, c in source_specs]
    placed = []
    wire = set()
    for circuit, region in enumerate(regions):
        group = _pack(rng, [o for o in objects if o['circuit'] == circuit], region)
        if group is None:
            return None
        route = _route(rng, group, region)
        if route is None:
            return None
        placed.extend(group)
        wire |= route
    targets = [o for o in placed if o['kind'] == 'target']
    sources = [o for o in placed if o['kind'] == 'source']
    distances = [_distance(wire, o['port']) for o in targets]
    assigned = [[] for _ in targets]
    for source in sources:
        candidates = [(d[source['port']], i) for i, (o, d) in enumerate(zip(targets, distances))
                      if o['shape'] == source['shape'] and source['port'] in d]
        if not candidates:
            source['color'] = _pick(rng, paint_colors)
            continue
        best = min(d for d, i in candidates)
        nearest = [i for d, i in candidates if d == best]
        if len(nearest) != 1:
            return None  # No distance ties in the generated input grammar.
        assigned[nearest[0]].append(source)
    if lesson == 0 and (min(map(len, assigned)) < 3):
        return None
    if lesson == 1 and sorted(map(len, assigned)) != [0, 4, 4, 4]:
        return None
    if not any(assigned):
        return None
    winners = list(map(int, rng.choice(paint_colors, len(targets), replace=False)))
    # Non-teaching examples may share winners on independent collectors.
    if lesson is None and rng.random() < .35:
        winners[-1] = winners[0]
    for i, group in enumerate(assigned):
        for source, color in zip(group, _ballot_colors(rng, len(group), winners[i], paint_colors)):
            source['color'] = color
    a = np.full((h, w), bg, dtype=np.int8)
    for p in wire:
        a[p] = wire_color
    for target in targets:
        for p in target['cells']-target['inner']:
            a[p] = shell_color
    for source in sources:
        for p in source['cells']:
            a[p] = source['color']
    b = a.copy()
    for i, target in enumerate(targets):
        if assigned[i]:
            for p in target['inner']:
                b[p] = winners[i]
    rotation = int(rng.integers(4))
    a, b = np.rot90(a, rotation), np.rot90(b, rotation)
    if rng.random() < .5:
        a, b = np.fliplr(a), np.fliplr(b)
    pair = dict(input=a.copy(), output=b.copy())
    try:
        data = _analyze(a)
        if not np.array_equal(_prediction(data)[0], b):
            return None
        if lesson == 0:
            for kwargs in (dict(mode='any'), dict(mode='area'), dict(mode='bbox'), dict(metric='euclidean'),
                           dict(broadcast=True), dict(mode='rotation'), dict(mode='translation')):
                if np.array_equal(_prediction(data, **kwargs)[0], b):
                    return None
        if lesson == 1 and np.array_equal(_prediction(data, broadcast=True, global_scope=True)[0], b):
            return None
        if shift and (np.array_equal(_prediction(data, mode='any')[0], b)
                      or np.array_equal(_prediction(data, broadcast=True)[0], b)
                      or np.array_equal(_prediction(data, broadcast=True, global_scope=True)[0], b)):
            return None
    except AssertionError:
        return None
    return pair


def _counts(num_train, num_test):
    for n in (num_train, num_test):
        if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)):
            raise ValueError('Example counts must be integers')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('Require 2..10 train, 1..4 test, and train > test')


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    pool = _pool()
    anchors = [s for s in pool if 4 <= len(s) <= 5 and max(max(p) for p in s) <= 2
               and len(_variants(s)) > len(_variants(s, False))]
    anchor = _pick(rng, anchors)
    extent = lambda s: sorted((max(r for r, c in s), max(c for r, c in s)))
    companions = [s for s in pool if len(s) == len(anchor) and s != anchor and extent(s) == extent(anchor)]
    companion = _pick(rng, companions)
    puzzle = {'train': [], 'test': []}
    seen = set()
    taught_shapes = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            lesson = index if split == 'train' and index < 2 else None
            novel_shapes = tuple(s for s in pool if s not in taught_shapes) if split == 'test' else ()
            for _ in range(160):
                pair = _make_pair(rng, anchor, companion, lesson, split == 'test', novel_shapes)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('Bounded collector layout search exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                taught_shapes.update(c['shape'] for c in _analyze(pair['input'])['collectors'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    lessons = []
    taught_shapes = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            data = _analyze(a)
            recipients = _recipients(data)
            assert -1 not in recipients, 'Tied wire distance is outside the generated grammar'
            predicted, ballots = _prediction(data)
            assert np.array_equal(predicted, b), 'Wrong collector election or preservation'
            assert np.any(a != b) and np.all(a[a != b] == data['bg'])
            for counts in ballots:
                if counts:
                    high = max(counts.values())
                    assert sum(n == high for n in counts.values()) == 1
            target_cells = set().union(*(c['cells'] for c in data['collectors']))
            assert all(tuple(p) in target_cells for p in np.argwhere(a != b))
            all_ports = [o['port'] for o in data['sources']+data['collectors']]
            assert len(set(all_ports)) == len(all_ports), 'Object ports cannot be shared'
            observed_shapes = {c['shape'] for c in data['collectors']}
            if split == 'train':
                lessons.append((data, b))
                taught_shapes.update(observed_shapes)
            else:
                assert observed_shapes-taught_shapes, 'Test must transfer to novel geometry'
                assert not np.array_equal(_prediction(data, mode='any')[0], b)
                assert not np.array_equal(_prediction(data, broadcast=True)[0], b)
                assert not np.array_equal(_prediction(data, broadcast=True, global_scope=True)[0], b)
    # Necessary witnesses, including for a minimum 2/1 puzzle.
    for kwargs in (dict(mode='any'), dict(mode='area'), dict(mode='bbox'), dict(metric='euclidean'),
                   dict(broadcast=True), dict(mode='rotation'), dict(mode='translation')):
        assert any(not np.array_equal(_prediction(d, **kwargs)[0], b) for d, b in lessons)
    assert any(len(d['circuits']) >= 2 for d, b in lessons)
    assert any(not counts for d, b in lessons for counts in _prediction(d)[1])
    assert any(not np.array_equal(_prediction(d, broadcast=True, global_scope=True)[0], b) for d, b in lessons)


def self_test():
    """Bounded streaming audit over 42 seeds; retains hashes and sets only."""
    signatures, dimensions, shapes, roles, source_counts, target_counts, wires = set(), set(), set(), set(), set(), set(), set()
    nonmajority = empty = reflected = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                digest.update(str(a.shape).encode())
                digest.update(a.tobytes())
                digest.update(b.tobytes())
                d = _analyze(a)
                _, ballots = _prediction(d)
                dimensions.add(a.shape)
                shapes.update(c['shape'] for c in d['collectors'])
                roles.add((d['bg'], d['shell_color'], d['wire_color']))
                source_counts.add(len(d['sources']))
                target_counts.add(len(d['collectors']))
                wires.add(len(d['wire']))
                empty += sum(not c for c in ballots)
                nonmajority += sum(bool(c) and max(c.values())*2 <= sum(c.values()) for c in ballots)
                reflected += not np.array_equal(_prediction(d, mode='rotation')[0], b)
        signatures.add(digest.hexdigest())
        if seed == 0:
            first = puzzle
    assert len(signatures) == 42 and len(dimensions) >= 15
    assert len(shapes) >= 20 and len(roles) >= 70
    assert len(source_counts) >= 7 and len(target_counts) >= 3 and len(wires) >= 50
    assert empty > 0 and nonmajority >= 42 and reflected >= 42
    repeat = generate_puzzle(0)
    assert all(_key(p[k]) == _key(q[k]) for s in first for p, q in zip(first[s], repeat[s]) for k in ('input', 'output'))
    # The host also checks all 30 legal combinations. Keep the repeated
    # self-audit to the minimum and maximum boundaries.
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(101+nt*5+ne, nt, ne)
        assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    p = broken['test'][0]
    r, c = np.argwhere(p['input'] != p['output'])[0]
    p['output'][r, c] = p['input'][r, c]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('A partial collector fill was accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), dimensions=len(dimensions),
                shape_classes=len(shapes), role_triples=len(roles), source_counts=sorted(source_counts),
                collector_counts=sorted(target_counts), wire_lengths=len(wires),
                empty_collectors=empty, nonmajority_elections=nonmajority,
                reflection_sensitive_pairs=reflected)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
