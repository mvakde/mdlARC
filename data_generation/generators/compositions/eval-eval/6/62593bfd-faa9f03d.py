"""Shape-addressed rerouting. Only Python's standard library and NumPy.

Connectors=5, repair waypoints=8, commands=9. All semantic inference is
input-only; the constructor renders labels from its separate circuit records.
"""
from collections import Counter
import json

import numpy as np


ROOT, REPAIR, COMMAND = 5, 8, 9
DATA_COLORS = (0, 1, 2, 3, 4, 6, 7)
D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _neg(d):
    return -d[0], -d[1]


def _rot(p, k):
    for _ in range(k % 4):
        p = (-p[1], p[0])
    return p


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _boundary(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
    return [(a[0] + i*d[0], a[1] + i*d[1])
            for i in range(max(abs(a[0]-b[0]), abs(a[1]-b[1])) + 1)]


def _polyline(points):
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _ray(root, d, shape):
    cells, p = set(), _add(root, d)
    while _inside(p, shape):
        cells.add(p)
        p = _add(p, d)
    return cells


def _neighbors(p, cells):
    return [q for d in D4 if (q := _add(p, d)) in cells]


def _halo(cells):
    return {(r+dr, c+dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _components(cells):
    todo, result = set(cells), []
    while todo:
        start = min(todo)
        todo.remove(start)
        component, stack = {start}, [start]
        while stack:
            for q in _neighbors(stack.pop(), todo):
                todo.remove(q)
                component.add(q)
                stack.append(q)
        result.append(component)
    return result


def _key(core):
    """A core includes its root at (0,0); no translation of that root."""
    return min(tuple(sorted(_rot(p, k) for p in core)) for k in range(4))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _background(inp):
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > inp.size // 2 and bg in DATA_COLORS
    return bg


def _repair(inp):
    """Marked turns/endpoints plus one/two-cell straight omissions.

    Bridge from the restored visible samples simultaneously. Two turning
    endpoints are not a witness for a connection across blank space.
    """
    bg = _background(inp)
    work = inp.copy()
    payloads = set(map(int, np.unique(inp))) - {bg, ROOT, REPAIR, COMMAND}
    pending = set(map(tuple, np.argwhere(work == REPAIR)))
    for _ in range(len(pending) + 1):
        changes = {}
        for p in sorted(pending):
            colors = {int(work[q]) for d in D4
                      if _inside(q := _add(p, d), work.shape) and int(work[q]) in payloads}
            if len(colors) == 1:
                changes[p] = next(iter(colors))
        if not changes:
            break
        for p, c in changes.items():
            work[p] = c
            pending.remove(p)
    assert not pending, 'repair waypoint lacks a unique color witness'
    additions = {}
    for c in sorted(payloads):
        cells = set(map(tuple, np.argwhere(work == c)))
        for p in sorted(cells):
            for d in ((1, 0), (0, 1)):
                side = (-d[1], d[0])
                for distance in (2, 3):
                    q = p[0]+distance*d[0], p[1]+distance*d[1]
                    if q not in cells:
                        continue
                    between = _line(p, q)[1:-1]
                    if not all(work[x] == bg for x in between):
                        continue
                    if (any(_add(p, s) in cells for s in (side, _neg(side)))
                            and any(_add(q, s) in cells for s in (side, _neg(side)))):
                        continue
                    for x in between:
                        assert x not in additions or additions[x] == c, 'ambiguous gap'
                        additions[x] = c
    for p, c in additions.items():
        work[p] = c
    return bg, work


def _analyse(inp):
    bg, repaired = _repair(inp)
    foreground = set(map(tuple, np.argwhere((repaired != bg) & (repaired != COMMAND))))
    circuits = []
    used_commands = set()
    for component in _components(foreground):
        roots = [p for p in component if repaired[p] == ROOT]
        assert len(roots) == 1, 'each separated circuit needs one connector'
        root = roots[0]
        branches = _components(component - {root})
        assert len(branches) == 2 and len(_neighbors(root, component)) == 2
        terminal = [b for b in branches if any(_boundary(p, inp.shape) for p in b)]
        assert len(terminal) == 1, 'exactly one cable must reach the boundary'
        cable = terminal[0]
        body = next(b for b in branches if b is not cable)
        assert len(_neighbors(root, cable)) == len(_neighbors(root, body)) == 1
        assert all(len(_neighbors(p, cable | {root})) <= 2 for p in cable)
        assert sum(_boundary(p, inp.shape) for p in cable) == 1
        colors = {int(repaired[p]) for p in component - {root}}
        assert len(colors) == 1 and next(iter(colors)) in payloads_for(bg)
        color = next(iter(colors))
        core = {(p[0]-root[0], p[1]-root[1]) for p in body} | {(0, 0)}
        commands = [(q, d) for d in D4
                    if _inside(q := _add(root, d), inp.shape) and inp[q] == COMMAND]
        assert len(commands) <= 1
        command = commands[0][1] if commands else None
        if command is not None:
            assert commands[0][0] not in used_commands, 'command has two adjacent connectors'
            used_commands.add(commands[0][0])
            assert _add(root, _neg(command)) in cable
            assert _add(root, command) not in component
        circuits.append(dict(root=root, body=body, cable=cable, color=color,
                             core=core, key=_key(core), command=command))
    assert used_commands == set(map(tuple, np.argwhere(inp == COMMAND)))
    leaders = {}
    for c in circuits:
        if c['command'] is not None:
            assert c['key'] not in leaders, 'one command per shape address'
            leaders[c['key']] = c
    for c in circuits:
        c['new_direction'] = None
        if c['key'] in leaders:
            leader = leaders[c['key']]
            turns = [k for k in range(4) if {_rot(p, k) for p in leader['core']} == c['core']]
            assert len(turns) == 1, 'rooted match must have a unique rotation'
            c['new_direction'] = _rot(leader['command'], turns[0])
            c['new'] = _ray(c['root'], c['new_direction'], inp.shape)
            assert not c['new'] & c['body'], 'new ray may not cover its own address'
        else:
            c['new'] = set()
    _check_ray_conflicts(circuits)
    return bg, repaired, circuits


def payloads_for(bg):
    return set(DATA_COLORS) - {bg}


def _check_ray_conflicts(circuits):
    roots = {c['root'] for c in circuits}
    for i, c in enumerate(circuits):
        assert not c['new'] & roots, 'a ray may not cover a connector'
        for b in circuits[i+1:]:
            assert c['color'] == b['color'] or not c['new'] & b['new'], 'conflicting new rays'


def _render(shape, bg, circuits, output):
    grid = np.full(shape, bg, dtype=np.int8)
    for c in circuits:
        for p in c['body'] | (set() if output and c['new_direction'] is not None else c['cable']):
            grid[p] = c['color']
        grid[c['root']] = ROOT
    if output:
        # Simultaneous removal precedes drawing. Rays cover retained traces.
        for c in circuits:
            for p in c['new']:
                grid[p] = c['color']
    else:
        for c in circuits:
            if c['command'] is not None:
                grid[_add(c['root'], c['command'])] = COMMAND
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No training labels, seed, RNG, or constructor records are consulted."""
    inp = _grid(input_grid)
    bg, _, circuits = _analyse(inp)
    return _render(inp.shape, bg, circuits, True)


def _simple_path(path):
    cells = set(path)
    return len(cells) == len(path) and all(len(_neighbors(p, cells)) <= 2 for p in cells)


def _random_core(rng):
    """Finite rooted paths with optional side branches; variable run lengths."""
    for _ in range(100):
        points = [(0, 0)]
        direction = (1, 0)
        for _ in range(int(rng.integers(1, 5))):
            length = int(rng.integers(2, 5))
            end = points[-1][0]+length*direction[0], points[-1][1]+length*direction[1]
            candidate = points + [end]
            path = _polyline(candidate)
            rows, cols = zip(*path)
            if (max(rows)-min(rows) > 6 or max(cols)-min(cols) > 6
                    or not _simple_path(path) or len(_neighbors((0, 0), set(path))) != 1):
                break
            points = candidate
            direction = _rot(direction, int(rng.choice([1, 3])))
        core = set(_polyline(points))
        if len(core) < 4:
            continue
        if rng.random() < .28:
            candidates = list(sorted(core - {(0, 0)}))
            rng.shuffle(candidates)
            for p in candidates:
                for d in [D4[int(i)] for i in rng.permutation(4)]:
                    branch = set(_line(p, (p[0]+2*d[0], p[1]+2*d[1]))) - {p}
                    if branch & core or any(len(_neighbors(q, core)) != (1 if q == _add(p, d) else 0) for q in branch):
                        continue
                    trial = core | branch
                    rows, cols = zip(*trial)
                    if (max(rows)-min(rows) <= 6 and max(cols)-min(cols) <= 6
                            and len(_neighbors((0, 0), trial)) == 1):
                        core = trial
                        break
                else:
                    continue
                break
        return core
    # A small rooted chiral elbow, only if random growth is exhausted.
    return {(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)}


def _make_objects(rng, profile, bg):
    objects, forbidden = [], set()

    def add_class(count, commanded, mirror=False, turns=None):
        for _ in range(100):
            core = _random_core(rng)
            key = _key(core)
            reflected = {(r, -c) for r, c in core}
            if key not in forbidden and (not mirror or _key(reflected) != key):
                break
        else:
            raise AssertionError('core vocabulary exhausted')
        forbidden.add(key)
        if mirror:
            forbidden.add(_key(reflected))
        base_command = _rot(next(p for p in core if abs(p[0])+abs(p[1]) == 1), int(rng.choice([1, 3])))
        leader = int(rng.integers(count)) if commanded else -1
        turns = list(map(int, rng.integers(4, size=count))) if turns is None else turns
        for i, k in enumerate(turns):
            objects.append(dict(core={_rot(p, k) for p in core}, key=key,
                                command=_rot(base_command, k) if i == leader else None,
                                new_direction=_rot(base_command, k) if commanded else None,
                                nonopposite=commanded and i != leader and (profile == 'witness' or rng.random() < .5)))
        if mirror:
            k = int(rng.integers(4))
            objects.append(dict(core={_rot(p, k) for p in reflected}, key=_key(reflected),
                                command=None, new_direction=None, nonopposite=False))

    if profile == 'witness':
        start = int(rng.integers(4))
        add_class(2, True, mirror=True, turns=[start, (start+int(rng.choice([1, 3]))) % 4])
        add_class(1, False)
    elif profile == 'triplet':
        add_class(3, True)
        add_class(2, False)
    elif profile == 'query':
        add_class(2, True)
        add_class(int(rng.choice([1, 2, 3], p=[.15, .65, .20])), True)
        add_class(1, False)
    else:
        add_class(int(rng.choice([1, 2, 3], p=[.10, .60, .30])), True)
        if rng.random() < .40:
            add_class(int(rng.choice([1, 2])), True)
        add_class(int(rng.choice([1, 2], p=[.75, .25])), False, mirror=rng.random() < .25)
        if len(objects) > 6:
            return _make_objects(rng, 'triplet', bg)
    palette = [int(c) for c in rng.permutation(DATA_COLORS) if c != bg]
    size = int(rng.integers(1, min(5, len(objects)) + 1))
    for c in objects:
        c['color'] = int(rng.choice(palette[:size]))
    if profile == 'witness':
        objects[0]['color'], objects[1]['color'], objects[2]['color'] = palette[0], palette[1], palette[0]
    rng.shuffle(objects)
    return objects


def _cable_options(rng, root, d, shape):
    """Straight, dogleg, offset, U and zigzag routes to any boundary."""
    def boundary_end(p, direction):
        ray = _ray(p, direction, shape)
        return max(ray, key=lambda q: abs(q[0]-p[0])+abs(q[1]-p[1])) if ray else None
    options = []
    for _ in range(55):
        points, current_d = [root], d
        bends = int(rng.choice([0, 1, 2, 3], p=[.25, .30, .30, .15]))
        for _ in range(bends):
            length = int(rng.integers(2, 8))
            q = points[-1][0]+length*current_d[0], points[-1][1]+length*current_d[1]
            if not _inside(q, shape) or _boundary(q, shape):
                break
            points.append(q)
            current_d = _rot(current_d, int(rng.choice([1, 3])))
        else:
            end = boundary_end(points[-1], current_d)
            if end is not None:
                path = _polyline(points + [end])
                if _simple_path(path):
                    options.append(path)
    return options


def _place(rng, shape, objects):
    placed, occupied = [], set()
    # Larger address masks first, independently placed throughout the interior.
    order = sorted(range(len(objects)), key=lambda i: -len(objects[i]['core']))
    for i in order:
        obj = objects[i]
        core = obj['core']
        rmin, rmax = min(r for r, c in core), max(r for r, c in core)
        cmin, cmax = min(c for r, c in core), max(c for r, c in core)
        roots = [(r, c) for r in range(1-rmin, shape[0]-1-rmax)
                 for c in range(1-cmin, shape[1]-1-cmax)]
        rng.shuffle(roots)
        for root in roots[:200]:
            body = {_add(root, p) for p in core if p != (0, 0)}
            cells = body | {root}
            if cells & _halo(occupied):
                continue
            direction = obj['new_direction']
            new = set() if direction is None else _ray(root, direction, shape)
            if new & body or any(new & c['new'] and obj['color'] != c['color'] for c in placed):
                continue
            if any(c['root'] in new or root in c['new'] for c in placed):
                continue
            # Commands occupy background, rather than replacing trace pixels.
            if obj['command'] is not None:
                q = _add(root, obj['command'])
                if not _inside(q, shape) or q in cells or q in _halo(occupied):
                    continue
            placed.append(dict(obj, root=root, body=body, new=new))
            occupied.update(cells)
            break
        else:
            return None
    command_list = [_add(c['root'], c['command']) for c in placed if c['command'] is not None]
    commands = set(command_list)
    if len(commands) != len(command_list):
        return None
    if commands & occupied:
        return None
    # All finite cores are placed before old cables are routed around them.
    rng.shuffle(placed)
    for c in placed:
        root = c['root']
        neighbor = _neighbors(root, c['body'])[0]
        body_direction = (neighbor[0]-root[0], neighbor[1]-root[1])
        if c['command'] is not None:
            choices = [_neg(c['command'])]
        elif c['new_direction'] is not None:
            choices = [_neg(body_direction)] if c['nonopposite'] else [_neg(c['new_direction'])]
        else:
            choices = [d for d in D4 if d != body_direction]
            rng.shuffle(choices)
        other = occupied - c['body'] - {root}
        for d in choices:
            for path in _cable_options(rng, root, d, shape):
                cable = set(path[1:])
                if (cable & (_halo(other) | commands) or cable & c['body']
                        or any(len(_neighbors(p, c['body'] | {root})) != (1 if p == _add(root, d) else 0) for p in cable)
                        or sum(_boundary(p, shape) for p in cable) != 1):
                    continue
                c['cable'] = cable
                occupied.update(cable)
                break
            else:
                continue
            break
        else:
            return None
    return placed


def _damage(rng, complete, circuits, bg, must_core_gap=False):
    inp = complete.copy()
    candidates = [(p, c) for c in circuits for p in sorted(c['body'] | c['cable'])]
    rng.shuffle(candidates)
    if must_core_gap:
        def priority(pc):
            p, c = pc
            ns = _neighbors(p, c['body'] | c['cable'] | {c['root']})
            bend = len(ns) == 2 and (ns[0][0]+ns[1][0] != 2*p[0] or ns[0][1]+ns[1][1] != 2*p[1])
            return 0 if bend else 1 if len(ns) != 2 else 2
        candidates.sort(key=priority)
    # Restore each attempted corruption using input-only evidence before accepting.
    desired = int(rng.integers(1, min(9, len(candidates)//4) + 1))
    marked = 0
    for p, c in candidates:
        old = int(inp[p])
        inp[p] = REPAIR
        try:
            _, recovered = _repair(inp)
            assert np.array_equal(recovered, np.where(complete == COMMAND, COMMAND, complete))
        except AssertionError:
            inp[p] = old
        else:
            marked += 1
            if marked == desired:
                break
    gap_candidates = [(p, c) for p, c in candidates if inp[p] != REPAIR
                      and p not in _neighbors(c['root'], c['body'] | c['cable'])]
    if must_core_gap:
        gap_candidates.sort(key=lambda pc: not (pc[0] in pc[1]['body'] and pc[1]['command'] is None
                                                and pc[1]['new_direction'] is not None))
    desired_gaps = int(rng.integers(1 if must_core_gap else 0, 5))
    gaps, core_gap = 0, False
    for p, c in gap_candidates:
        if gaps >= desired_gaps and (not must_core_gap or core_gap):
            break
        if inp[p] == REPAIR:
            continue
        old = int(inp[p])
        inp[p] = bg
        try:
            _, recovered = _repair(inp)
            assert np.array_equal(recovered, complete)
        except AssertionError:
            inp[p] = old
        else:
            gaps += 1
            core_gap |= p in c['body'] and c['command'] is None and c['new_direction'] is not None
    if marked == 0 or (must_core_gap and not core_gap):
        return None
    return inp


def _make_pair(rng, profile, require_hit=False):
    for attempt in range(300):
        bg = int(rng.choice(DATA_COLORS))
        objects = _make_objects(rng, profile, bg)
        low = 12 if len(objects) <= 4 else 18 if len(objects) == 5 else 20
        shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        circuits = _place(rng, shape, objects)
        if circuits is None:
            continue
        complete = _render(shape, bg, circuits, False)
        try:
            if not np.array_equal(_repair(complete)[1], complete):
                continue
            out = _render(shape, bg, circuits, True)
            if not _has_ray_hit(circuits) and require_hit:
                continue
            inp = _damage(rng, complete, circuits, bg, must_core_gap=profile in ('witness', 'query'))
            if inp is None or not np.array_equal(solve(inp), out):
                continue
            if profile in ('witness', 'query'):
                if (not _features({'input': inp, 'output': out})['gap_changes_address']
                        or np.array_equal(_alternative_output(inp, 'raw_match'), out)):
                    continue
            required_probes = (('reflection', 'marked_only', 'reverse_tail', 'global_direction')
                               if profile == 'witness' else ('exactly_two',) if profile == 'triplet' else ())
            if any(np.array_equal(_alternative_output(inp, mode), out) for mode in required_probes):
                continue
            if profile == 'triplet' and not any(_features({'input': inp, 'output': out})['old_bends']):
                continue
        except AssertionError:
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError(f'bounded circuit construction exhausted: {profile}, hit={require_hit}')


def _has_ray_hit(circuits):
    return any(c['new'] & (b['body'] | (b['cable'] if b['new_direction'] is None else set()))
               and c['color'] != b['color'] for c in circuits for b in circuits if c is not b)


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _fingerprint(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, profiles in (('train', ['witness', 'triplet'] + ['random']*(num_train-2)),
                            ('test', ['query']*num_test)):
        for i, profile in enumerate(profiles):
            for _ in range(10):
                pair = _make_pair(rng, profile, require_hit=split == 'train' and i == 1)
                key = _fingerprint(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input budget exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    inp = _grid(pair['input'])
    bg, repaired, cs = _analyse(inp)
    groups = Counter(c['key'] for c in cs)
    active = [c for c in cs if c['new_direction'] is not None]
    raw_cores = [{(p[0]-c['root'][0], p[1]-c['root'][1]) for p in c['body'] if inp[p] != bg}
                 | {(0, 0)} for c in cs]
    raw_keys = {c['key']: _key(raw) for raw, c in zip(raw_cores, cs) if c['command'] is not None}
    marker_sites = Counter()
    for c in cs:
        cells = c['body'] | c['cable'] | {c['root']}
        for p in cells:
            if inp[p] != REPAIR:
                continue
            ns = _neighbors(p, cells)
            site = ('endpoint' if len(ns) == 1 else 'junction' if len(ns) > 2 else
                    'straight' if ns[0][0]+ns[1][0] == 2*p[0] and ns[0][1]+ns[1][1] == 2*p[1]
                    else 'bend')
            marker_sites[site] += 1
    def bends(c):
        cells = c['cable'] | {c['root']}
        return sum(len(ns := _neighbors(p, cells)) == 2
                   and (ns[0][0]+ns[1][0] != 2*p[0] or ns[0][1]+ns[1][1] != 2*p[1])
                   for p in c['cable'])
    return dict(shape=list(inp.shape), background=bg, circuits=len(cs),
                class_sizes=sorted(groups.values()), active=len(active),
                follower_count=sum(c['command'] is None for c in active),
                directions=[list(c['new_direction']) for c in active],
                mirror_negative=any(c['new_direction'] is None and _key({(r, -x) for r, x in c['core']})
                                    == a['key'] and c['key'] != a['key'] for c in cs for a in active),
                color_independence=any(a['key'] == b['key'] and a['color'] != b['color'] for a in active for b in active),
                color_negative=any(a['color'] == b['color'] and b['new_direction'] is None for a in active for b in cs),
                uncommanded_repeat=any(groups[c['key']] > 1 and c['new_direction'] is None for c in cs),
                command_count=sum(c['command'] is not None for c in cs),
                nonopposite=sum(_add(c['root'], _neg(c['new_direction'])) not in c['cable'] for c in active),
                gap_changes_address=any(_key(raw) != raw_keys.get(c['key']) and c['command'] is None and c['new_direction'] is not None
                                        for raw, c in zip(raw_cores, cs)),
                repair_count=int(np.count_nonzero(inp == REPAIR)),
                gap_count=int(np.count_nonzero((inp == bg) & (repaired != bg))),
                marker_sites=dict(marker_sites),
                deleted_markers=sum(inp[p] == REPAIR for c in active for p in c['cable']),
                deleted_gaps=sum(inp[p] == bg for c in active for p in c['cable']),
                core_areas=[len(c['body']) for c in cs],
                core_extents=[[max(r for r, x in c['core'])-min(r for r, x in c['core'])+1,
                               max(x for r, x in c['core'])-min(x for r, x in c['core'])+1] for c in cs],
                branched_cores=sum(any(len(_neighbors(p, c['core'])) > 2 for p in c['core']) for c in cs),
                old_bends=[bends(c) for c in active],
                ray_hit=_has_ray_hit(cs),
                old_lengths=[len(c['cable']) for c in active],
                ray_lengths=[len(c['new']) for c in active])


def _alternative_output(inp, mode):
    """Named shortcut probes for validation, not a second semantic solver.

    Geometry/color reconstruction remains correct; each probe drops precisely
    one essential operation or replaces the matching/transport convention.
    """
    bg, _, circuits = _analyse(inp)
    cs = [dict(c) for c in circuits]
    raw = [{(p[0]-c['root'][0], p[1]-c['root'][1]) for p in c['body'] if inp[p] != bg}
           | {(0, 0)} for c in cs]
    leaders = [i for i, c in enumerate(cs) if c['command'] is not None]
    for i, c in enumerate(cs):
        direction = c['new_direction']
        if mode == 'marked_only' and c['command'] is None:
            direction = None
        elif mode == 'raw_match' and c['command'] is None:
            candidates = [(j, k) for j in leaders for k in range(4)
                          if {_rot(p, k) for p in raw[j]} == raw[i]]
            direction = _rot(cs[candidates[0][0]]['command'], candidates[0][1]) if len(candidates) == 1 else None
        elif mode == 'reflection' and direction is None:
            candidates = [(j, k) for j in leaders for k in range(4)
                          if {_rot((p[0], -p[1]), k) for p in cs[j]['core']} == c['core']]
            if len(candidates) == 1:
                j, k = candidates[0]
                d = cs[j]['command']
                direction = _rot((d[0], -d[1]), k)
        elif mode == 'reverse_tail' and direction is not None:
            p = _neighbors(c['root'], c['cable'])[0]
            direction = (c['root'][0]-p[0], c['root'][1]-p[1])
        elif mode == 'global_direction' and direction is not None:
            direction = next(b['command'] for b in cs if b['key'] == c['key'] and b['command'] is not None)
        elif mode == 'exactly_two' and sum(b['key'] == c['key'] for b in cs) != 2:
            direction = c['command']
        elif mode == 'keep_old':
            c['new_direction'] = None
            continue
        c['new_direction'] = direction
        c['new'] = set() if direction is None else _ray(c['root'], direction, inp.shape)
    return _render(inp.shape, bg, cs, True)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _fingerprint(inp)
            assert key not in seen
            seen.add(key)
            bg, repaired, circuits = _analyse(inp)
            assert np.array_equal(_render(inp.shape, bg, circuits, True), out), 'wrong full output'
            assert not np.any((out == REPAIR) | (out == COMMAND))
            assert np.array_equal(inp == ROOT, out == ROOT), 'connectors changed'
            assert 1 <= len(circuits) <= 6
            ray_owner = {}
            for c in circuits:
                for p in c['new']:
                    assert p not in ray_owner or ray_owner[p] == c['color']
                    ray_owner[p] = c['color']
                    assert out[p] == c['color'], 'boundary ray missing or recolored'
            for c in circuits:
                for p in c['body'] | c['cable']:
                    expected = (ray_owner[p] if p in ray_owner else bg
                                if c['new_direction'] is not None and p in c['cable'] else c['color'])
                    assert out[p] == expected, 'retention, deletion, or crossing ownership wrong'
            # Input circuits are separated even diagonally, and are thin.
            occupied = set()
            for c in circuits:
                cells = c['body'] | c['cable'] | {c['root']}
                assert not cells & _halo(occupied)
                occupied.update(cells)
            for color in payloads_for(bg):
                m = repaired == color
                assert not np.any(m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:])
            features[split].append(_features(pair))
    train = features['train']
    for field in ('mirror_negative', 'color_independence', 'color_negative', 'gap_changes_address', 'ray_hit', 'uncommanded_repeat'):
        assert any(f[field] for f in train), f'missing demonstration: {field}'
    assert any(f['nonopposite'] for f in train), 'must teach transported direction, not cable reversal'
    assert any(3 in f['class_sizes'] and f['active'] >= 3 for f in train), 'triplet broadcast not taught'
    assert any(f['follower_count'] for f in train)
    assert any(f['marker_sites'].get('bend', 0) for f in train), 'bend repair not taught'
    assert any(any(f['old_bends']) for f in train), 'deletion through bends not taught'
    for mode in ('marked_only', 'raw_match', 'reflection', 'reverse_tail',
                 'global_direction', 'exactly_two', 'keep_old'):
        assert any(not np.array_equal(_alternative_output(_grid(p['input']), mode), p['output'])
                   for p in puzzle['train']), f'training admits shortcut: {mode}'
    for p, f in zip(puzzle['test'], features['test']):
        assert f['command_count'] == 2 and f['gap_changes_address'], 'missing compositional query shift'
        assert not np.array_equal(_alternative_output(_grid(p['input']), 'marked_only'), p['output'])
        assert not np.array_equal(_alternative_output(_grid(p['input']), 'raw_match'), p['output'])


def _puzzle_key(puzzle):
    return tuple((_fingerprint(np.asarray(p['input'])), _fingerprint(np.asarray(p['output'])))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """42 seeds, finite count/type checks, equivariance, and semantic negatives."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    fs = [_features(pair) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    assert len({tuple(f['shape']) for f in fs}) >= 30
    assert {f['background'] for f in fs} == set(DATA_COLORS)
    assert {tuple(d) for f in fs for d in f['directions']} == set(D4)
    assert len({f['circuits'] for f in fs}) >= 3
    assert {0, 1, 2} <= {b for f in fs for b in f['old_bends']}
    assert any(f['branched_cores'] for f in fs)
    assert min(n for f in fs for n in f['ray_lengths']) == 1
    assert max(n for f in fs for n in f['ray_lengths']) >= 20
    assert max(n for f in fs for n in f['core_areas']) >= 10
    for site in ('straight', 'bend', 'endpoint'):
        assert sum(f['marker_sites'].get(site, 0) for f in fs) > 0
    assert sum(f['deleted_markers'] for f in fs) and sum(f['deleted_gaps'] for f in fs)
    for seed, counts in ((701, (2, 1)), (702, (10, 1)), (703, (10, 4))):
        p = generate_puzzle(seed, *counts)
        assert (len(p['train']), len(p['test'])) == counts
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f'illegal counts accepted: {counts}')
    for p in puzzles[:2]:
        for pair in p['train'] + p['test']:
            for k in range(4):
                for reflect in (False, True):
                    inp = np.rot90(np.fliplr(pair['input']) if reflect else pair['input'], k).copy()
                    out = np.rot90(np.fliplr(pair['output']) if reflect else pair['output'], k).copy()
                    assert np.array_equal(solve(inp), out), 'rotation/reflection covariance failed'
            # Reassign all background/payload roles, preserving marker roles.
            mapping = np.arange(10, dtype=np.int8)
            for a, b in zip(DATA_COLORS, DATA_COLORS[1:] + DATA_COLORS[:1]):
                mapping[a] = b
            assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
    bad = {s: [dict(p) for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'] = bad['test'][0]['output'].copy()
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('output corruption accepted')
    # Target actual family-specific cells, including a command transmitted to
    # an unmarked recipient. This is stronger than a background-only negative.
    pair = puzzles[0]['test'][0]
    bg, _, circuits = _analyse(pair['input'])
    follower = next(c for c in circuits if c['command'] is None and c['new_direction'] is not None)
    ray_cells = set().union(*(c['new'] for c in circuits))
    sites = {'connector': follower['root'], 'transmitted ray': min(follower['new'])}
    deleted = follower['cable'] - ray_cells
    assert deleted
    sites['deleted cable'] = min(deleted)
    for name, point in sites.items():
        mutated = {s: [dict(p) for p in puzzles[0][s]] for s in puzzles[0]}
        mutated['test'][0]['output'] = pair['output'].copy()
        mutated['test'][0]['output'][point] = (int(pair['output'][point])+1) % 10
        try:
            validate_generated_puzzle(mutated)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'{name} mutation accepted')
    return {
        'valid': True, 'seeds': list(range(42)), 'puzzles': len(puzzles),
        'pairs': len(fs), 'distinct_grid_shapes': len({tuple(f['shape']) for f in fs}),
        'circuit_counts': sorted({f['circuits'] for f in fs}),
        'backgrounds': sorted({f['background'] for f in fs}),
        'directions': [list(d) for d in sorted({tuple(d) for f in fs for d in f['directions']})],
        'old_bend_counts': sorted({int(b) for f in fs for b in f['old_bends']}),
        'ray_length_range': [min(n for f in fs for n in f['ray_lengths']), max(n for f in fs for n in f['ray_lengths'])],
        'core_area_range': [min(n for f in fs for n in f['core_areas']), max(n for f in fs for n in f['core_areas'])],
        'branched_cores': int(sum(f['branched_cores'] for f in fs)),
        'marker_sites': {s: int(sum(f['marker_sites'].get(s, 0) for f in fs))
                         for s in ('straight', 'bend', 'endpoint', 'junction')},
        'deleted_markers': int(sum(f['deleted_markers'] for f in fs)),
        'deleted_gaps': int(sum(f['deleted_gaps'] for f in fs)),
        'ray_hit_pairs': sum(f['ray_hit'] for f in fs),
        'semantic_scope': 'author-owned solver, invariants and shortcut probes; no independent solver',
    }


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
