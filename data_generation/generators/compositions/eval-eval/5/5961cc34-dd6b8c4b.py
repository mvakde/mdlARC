"""Fuel-operated shutters: a coupled beam-routing and collection family.

Only standard library and NumPy are used. All decoding and simulation use the
visible input. No parent generators or data are imported.
"""
from collections import deque
import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 909515541, 959784830, 704214129, 141441485, 315055216,
              48327285, 51145152, 978998059, 421262184, 732694955, 180067225,
              80685085, 881820185, 755493232, 215346607, 89687435, 296577430,
              241559603, 755790156, 148153415, 420392077, 958184771, 69222739,
              60044906, 233382372, 82852702, 908287883, 997432530, 935340637,
              449479706, 296483106, 583831798, 320371245, 197160346, 469687528,
              464704900, 719217127, 943402193, 604102175, 545758554, 267947322)


def _need(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), 'Bad grid dimensions')
    _need(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
          'Grid must contain integer colors 0..9')
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _step(cell, direction):
    dr, dc = DIRS[direction]
    return cell[0] + dr, cell[1] + dc


def _inside(cell, shape):
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _relay(pos, side, start=0, length=3):
    r, c = pos
    core = [(r+i//3, c+i%3) for i in range(9)]
    if side == 0:
        ports = [(r-1, c+j) for j in range(start, start+length)]
        indices = list(range(start, start+length))
    elif side == 1:
        ports = [(r+j, c+3) for j in range(start, start+length)]
        indices = [3*j+2 for j in range(start, start+length)]
    elif side == 2:
        ports = [(r+3, c+j) for j in range(start, start+length)]
        indices = [6+j for j in range(start, start+length)]
    else:
        ports = [(r+j, c-1) for j in range(start, start+length)]
        indices = [3*j for j in range(start, start+length)]
    return dict(pos=pos, core=core, ports=ports, indices=indices, side=side,
                cells=set(core+ports))


def _decode(a):
    """Infer the six roles from corners, the launcher, and repeated motifs."""
    h, w = a.shape
    bg = int(a[0, 0])
    _need(all(int(a[p]) == bg for p in ((0, w-1), (h-1, 0), (h-1, w-1))),
          'Corners must identify the background')
    candidates = []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        cells = set(map(tuple, np.argwhere(a == color).tolist()))
        if not 2 <= len(cells) <= 4:
            continue
        boundary = [p for p in cells if p[0] in (0, h-1) or p[1] in (0, w-1)]
        if len(boundary) != 1:
            continue
        p = boundary[0]
        direction = 2 if p[0] == 0 else 0 if p[0] == h-1 else 1 if p[1] == 0 else 3
        line = [p]
        for _ in range(len(cells)-1):
            line.append(_step(line[-1], direction))
        cap = _step(line[-1], direction)
        if set(line) == cells and _inside(cap, a.shape) and int(a[cap]) not in (bg, color):
            candidates.append((color, int(a[cap]), line, cap, direction))
    _need(len(candidates) == 1, 'Unique boundary tail required')
    signal, fuel, tail, cap, initial = candidates[0]
    assignments = []
    remaining_colors = set(map(int, np.unique(a))) - {bg, signal, fuel}
    for slot in sorted(remaining_colors):
        for mark in sorted(remaining_colors - {slot}):
            relays = []
            for r in range(1, h-3):
                for c in range(1, w-3):
                    patch = a[r:r+3, c:c+3]
                    if int(patch[1, 1]) != mark:
                        continue
                    perimeter = patch.copy()
                    perimeter[1, 1] = slot
                    if not np.all(perimeter == slot):
                        continue
                    options = []
                    for side in range(4):
                        full = _relay((r, c), side)
                        occupied = [j for j, p in enumerate(full['ports']) if int(a[p]) == mark]
                        if occupied and occupied == list(range(min(occupied), max(occupied)+1)):
                            options.append(_relay((r, c), side, min(occupied), len(occupied)))
                    if len(options) != 1 or not all(0 < p[0] < h-1 and 0 < p[1] < w-1
                                                   for p in options[0]['cells']):
                        continue
                    relays.append(options[0])
            if not relays:
                continue
            slot_cells = {p for x in relays for j, p in enumerate(x['core']) if j != 4}
            mark_cells = {p for x in relays for p in x['ports']+[x['core'][4]]}
            if slot_cells != set(map(tuple, np.argwhere(a == slot).tolist())):
                continue
            if mark_cells != set(map(tuple, np.argwhere(a == mark).tolist())):
                continue
            all_cells = [p for x in relays for p in x['cells']]
            if len(all_cells) != len(set(all_cells)):
                continue
            wall_colors = remaining_colors - {slot, mark}
            if len(wall_colors) > 1:
                continue
            assignments.append((slot, mark, relays, next(iter(wall_colors), None)))
    _need(len(assignments) == 1, 'Unique counter/port/wall role assignment required')
    slot, mark, relays, wall_color = assignments[0]
    _need(not any(set(tail+[cap]) & x['cells'] for x in relays), 'Launcher overlaps relay')
    wall = a == wall_color if wall_color is not None else np.zeros(a.shape, bool)
    return dict(bg=bg, signal=signal, fuel=fuel, slot=slot, mark=mark,
                wall_color=wall_color, wall=wall, relays=relays,
                tail=tail, cap=cap, initial=initial)


def _distances(shape, wall, relays, index):
    blocked = wall.copy()
    for relay in relays:
        for p in relay['cells']:
            blocked[p] = True
    core = relays[index]['core']
    distance = np.full(shape, -1, dtype=np.int16)
    queue = deque(core)
    for p in core:
        blocked[p] = False
        distance[p] = 0
    while queue:
        p = queue.popleft()
        for d in range(4):
            q = _step(p, d)
            if _inside(q, shape) and not blocked[q] and distance[q] < 0:
                distance[q] = distance[p] + 1
                queue.append(q)
    return distance


def _select(remaining, distance):
    ordered = sorted((int(distance[p]), p) for p in remaining if distance[p] >= 0)
    _need(len(ordered) <= 9 or ordered[8][0] < ordered[9][0],
          'A strict ninth/tenth path-distance gap is required')
    return [p for _, p in ordered[:9]], len(ordered)


def _run(a, decoded=None):
    scene = _decode(a) if decoded is None else decoded
    out = a.copy()
    bg, signal, fuel = (scene[k] for k in ('bg', 'signal', 'fuel'))
    relays, wall = scene['relays'], scene['wall']
    owner = {p: i for i, x in enumerate(relays) for p in x['cells']}
    remaining = set(map(tuple, np.argwhere(a == fuel).tolist())) - {scene['cap']}
    original_fuel = remaining.copy()
    out[scene['cap']] = signal
    direction = scene['initial']
    front = [_step(scene['cap'], direction)]
    visits, visited, segments = [], set(), []
    current_segment = []
    reason = None
    # Each relay charges at most once. Between contacts a front advances
    # monotonically, so this bound covers cycles as well as crossings.
    for _ in range((len(relays)+1) * (max(a.shape)+1)):
        if not all(_inside(p, a.shape) for p in front):
            reason = 'edge'
            break
        if any(wall[p] for p in front):
            reason = 'wall'
            break
        hits = {owner[p] for p in front if p in owner}
        _need(len(hits) <= 1, 'Simultaneous contact with two relays is excluded')
        if hits:
            index = next(iter(hits))
            segments.append(current_segment)
            current_segment = []
            if index in visited:
                reason = 'revisit'
                break
            visited.add(index)
            relay = relays[index]
            distance = _distances(a.shape, wall, relays, index)
            selected, reachable = _select(remaining, distance)
            n = len(selected)
            for p in selected:
                out[p] = bg
            remaining.difference_update(selected)
            for p in relay['core'][:n]:
                out[p] = fuel
            ports = [p for p, j in zip(relay['ports'], relay['indices']) if j < n]
            for p in ports:
                out[p] = signal
            r, c = relay['pos']
            def direct(p):
                return max(r-p[0], 0, p[0]-r-2) + max(c-p[1], 0, p[1]-c-2)
            retained_reachable = [p for p in remaining if distance[p] >= 0]
            reversal = bool(selected and retained_reachable and
                            max(map(direct, selected)) > min(map(direct, retained_reachable)))
            visits.append(dict(index=index, count=n, reachable=reachable,
                               selected=selected, width=len(ports), port_size=len(relay['ports']),
                               incoming_width=len(front), side=relay['side'], reversal=reversal,
                               max_distance=max((int(distance[p]) for p in selected), default=0),
                               partial_contact=sum(p in relay['cells'] for p in front) < len(front)))
            if not ports:
                reason = 'sealed'
                break
            direction = relay['side']
            front = [_step(p, direction) for p in ports]
        else:
            current_segment.extend(front)
            for p in front:
                # Fuel is transparent to the beam but is only erased by collection.
                if int(out[p]) == bg:
                    out[p] = signal
            front = [_step(p, direction) for p in front]
    else:
        raise AssertionError('Signal failed to terminate')
    segments.append(current_segment)
    unvisited = [i for i in range(len(relays)) if i not in visited]
    decoy_supply = False
    for i in unvisited:
        d = _distances(a.shape, wall, relays, i)
        if any(d[p] >= 0 for p in remaining):
            decoy_supply = True
            break
    unreachable = 0
    if visits:
        first_distance = _distances(a.shape, wall, relays, visits[0]['index'])
        unreachable = sum(first_distance[p] < 0 for p in original_fuel)
    facts = dict(visits=visits, reason=reason, remaining=remaining,
                 original_fuel=original_fuel, unvisited=unvisited,
                 decoy_supply=decoy_supply, unreachable=unreachable,
                 segments=segments, shape=a.shape, initial=scene['initial'],
                 palette=tuple(scene[k] for k in ('bg', 'signal', 'fuel', 'slot', 'mark', 'wall_color')))
    return out, facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _run(_grid(input_grid))[0]


def _separated(cells, occupied, gap=1):
    return not any((r+dr, c+dc) in occupied for r, c in cells
                   for dr in range(-gap, gap+1) for dc in range(-gap, gap+1))


def _geometric_route(rng, n, shape, low, variable_ports=False):
    """Grow a route with locally sampled turns, spacings, and transverse offsets."""
    h, w = shape
    initial = int(rng.integers(4))
    axis = int(rng.integers(4, (w if initial % 2 == 0 else h)-4))
    length = int(rng.integers(2, 5))
    boundary = (0, axis) if initial == 2 else (h-1, axis) if initial == 0 else (axis, 0) if initial == 1 else (axis, w-1)
    tail = [boundary]
    for _ in range(length-1):
        tail.append(_step(tail[-1], initial))
    cap = _step(tail[-1], initial)
    occupied, reserved = set(tail+[cap]), set(tail+[cap])
    origin, incoming = [cap], initial
    relays = []
    for i in range(n):
        options = []
        # The low-load demonstration has two open shutters on its first relay.
        allowed_sides = [0, 3] if low and i == 0 else [0, 1, 2, 3]
        for side in allowed_sides:
            if side == (incoming+2) % 4:
                continue
            for gap in range(2, min(14, max(shape))):
                target = origin[0]
                for _ in range(gap):
                    target = _step(target, incoming)
                for offset in range(-2, 3):
                    if incoming == 0:
                        pos = (target[0]-2, target[1]+offset)
                    elif incoming == 2:
                        pos = (target[0], target[1]+offset)
                    elif incoming == 1:
                        pos = (target[0]+offset, target[1])
                    else:
                        pos = (target[0]+offset, target[1]-2)
                    vary_here = variable_ports == 'all' or (variable_ports == 'later' and i > 0)
                    port_length = int(rng.integers(1, 3 if variable_ports == 'later' else 4)) if vary_here else 3
                    port_start = int(rng.integers(4-port_length)) if vary_here else 0
                    relay = _relay(pos, side, port_start, port_length)
                    if not all(1 <= p[0] < h-1 and 1 <= p[1] < w-1 for p in relay['cells']):
                        continue
                    if not _separated(relay['cells'], occupied) or relay['cells'] & reserved:
                        continue
                    # Project the first contact of the current beam. Partial
                    # capture is allowed; earlier route segments remain clear.
                    approach, front, found = set(), [_step(p, incoming) for p in origin], False
                    for _ in range(max(shape)):
                        if not all(_inside(p, shape) for p in front) or set(front) & occupied:
                            break
                        if set(front) & relay['cells']:
                            found = True
                            break
                        approach.update(front)
                        front = [_step(p, incoming) for p in front]
                    if found:
                        options.append((relay, approach))
        if not options:
            return None
        relay, approach = options[int(rng.integers(len(options)))]
        relays.append(relay)
        occupied |= relay['cells']
        reserved |= approach
        count = (2 if relay['side'] == 0 else 4) if low and i == 0 else 9
        origin = [p for p, j in zip(relay['ports'], relay['indices']) if j < count]
        incoming = relay['side']
    return dict(relays=relays, tail=tail, cap=cap, initial=initial,
                reserved=reserved, occupied=occupied)


def _add_pocket(rng, shape, layout, wall):
    h, w = shape
    ph, pw = map(int, rng.integers(7, 10, size=2))
    options = []
    for r in range(1, h-ph):
        for c in range(1, w-pw):
            square = {(rr, cc) for rr in range(r, r+ph) for cc in range(c, c+pw)}
            if _separated(square, layout['occupied']) and not square & layout['reserved']:
                options.append((r, c))
    if not options:
        return None
    r, c = options[int(rng.integers(len(options)))]
    wall[r, c:c+pw] = True
    wall[r+ph-1, c:c+pw] = True
    wall[r:r+ph, c] = True
    wall[r:r+ph, c+pw-1] = True
    pos = (r+int(rng.integers(2, ph-4)), c+int(rng.integers(2, pw-4)))
    length = int(rng.integers(1, 4))
    relay = _relay(pos, int(rng.integers(4)), int(rng.integers(4-length)), length)
    layout['relays'].append(relay)
    layout['occupied'] |= relay['cells']
    return (r, c)


def _wall_scene(rng, shape, layout, detour):
    wall = np.zeros(shape, bool)
    # Independent baffles and bends; segment positions and lengths are not
    # tied to route indices. Preserve the constructed incoming approaches.
    forbidden = layout['occupied'] | layout['reserved']
    for _ in range(int(rng.integers(3, 8)) if detour else int(rng.integers(0, 5))):
        for _ in range(25):
            horizontal = bool(rng.integers(2))
            h, w = shape
            if horizontal:
                r = int(rng.integers(1, h-1))
                lo = int(rng.integers(0, w-6))
                hi = int(rng.integers(lo+5, w))
                cells = {(r, c) for c in range(lo, hi+1)}
            else:
                c = int(rng.integers(1, w-1))
                lo = int(rng.integers(0, h-6))
                hi = int(rng.integers(lo+5, h))
                cells = {(r, c) for r in range(lo, hi+1)}
            if not cells & forbidden:
                for p in cells:
                    wall[p] = True
                break
    wall[0, 0] = wall[0, -1] = wall[-1, 0] = wall[-1, -1] = False
    return wall


def _pick_supply(rng, available, count, shape, style, forced=()):
    chosen = list(dict.fromkeys(forced))
    available = set(available) - set(chosen)
    if len(chosen)+len(available) < count:
        return None
    if count == 0:
        return []
    anchor = chosen[0] if chosen else sorted(available)[int(rng.integers(len(available)))]
    while len(chosen) < count:
        choices = sorted(available)
        if style != 'scattered':
            if style == 'cluster':
                scores = [(p[0]-anchor[0])**2 + (p[1]-anchor[1])**2 for p in choices]
            else:
                scores = [abs(p[0]-anchor[0])*shape[1] + abs(p[1]-anchor[1]) for p in choices]
            best = min(scores)
            choices = [p for p, score in zip(choices, scores) if score == best]
        p = choices[int(rng.integers(len(choices)))]
        chosen.append(p)
        available.remove(p)
    return chosen


def _supply(rng, shape, layout, wall, amount, detour):
    relays = layout['relays']
    distance = _distances(shape, wall, relays, 0)
    excluded = layout['occupied']
    available = {tuple(p) for p in np.argwhere((distance > 0) & ~wall).tolist()} - excluded
    style = str(rng.choice(['scattered', 'cluster', 'line']))
    if amount <= 9:
        return _pick_supply(rng, available, amount, shape, style)
    r, c = relays[0]['pos']
    def direct(p):
        return max(r-p[0], 0, p[0]-r-2) + max(c-p[1], 0, p[1]-c-2)
    frontier = sorted(available)
    for j in rng.permutation(len(frontier)):
        p = frontier[int(j)]
        near = {q for q in available if distance[q] <= distance[p]}
        far = {q for q in available if distance[q] > distance[p]}
        if len(near) < 9 or len(far) < amount-9:
            continue
        reversed_cells = [q for q in sorted(far) if direct(q) < direct(p)]
        if detour and not reversed_cells:
            continue
        forced_far = [reversed_cells[int(rng.integers(len(reversed_cells)))]] if detour else []
        selected = _pick_supply(rng, near, 9, shape, style, [p])
        retained = _pick_supply(rng, far, amount-9, shape, style, forced_far)
        return selected+retained
    return None


def _make_pair(rng, colors, mode, n):
    low, detour = mode == 'low', mode == 'excess'
    want_pocket = low or detour or rng.random() < .65
    for attempt in range(220):
        shape = tuple(map(int, rng.integers(22 if n >= 3 or detour else 18, 31, size=2)))
        port_policy = 'all' if mode in ('free', 'long') else 'later' if detour else False
        layout = _geometric_route(rng, n, shape, low, variable_ports=port_policy)
        if layout is None:
            continue
        wall = _wall_scene(rng, shape, layout, detour)
        if want_pocket:
            pocket = _add_pocket(rng, shape, layout, wall)
            if pocket is None:
                continue
        if low:
            amount = 2 if layout['relays'][0]['side'] == 0 else 4
        elif detour:
            amount = int(rng.integers(13, 18))
        elif mode == 'long':
            amount = 9*(n-1) + int(rng.integers(1, 10))
        else:
            amount = int(rng.integers(0, 9*n+7))
        supply = _supply(rng, shape, layout, wall, amount, detour)
        if supply is None:
            continue
        # Sealed chamber fuel is a witness for selective activation, not noise.
        decoy_fuel = []
        if want_pocket:
            distance = _distances(shape, wall, layout['relays'], len(layout['relays'])-1)
            available = {tuple(p) for p in np.argwhere((distance > 0) & ~wall).tolist()} - layout['occupied']
            if len(available) < 4:
                continue
            decoy_fuel = _pick_supply(rng, available, int(rng.integers(4, min(10, len(available))+1)),
                                      shape, str(rng.choice(['cluster', 'scattered', 'line'])))
            if decoy_fuel is None:
                continue
        bg, signal, fuel, slot, mark, wc = colors
        a = np.full(shape, bg, np.int8)
        a[wall] = wc
        for relay in layout['relays']:
            for p in relay['core']:
                a[p] = slot
            a[relay['core'][4]] = mark
            for p in relay['ports']:
                a[p] = mark
        for p in layout['tail']:
            a[p] = signal
        a[layout['cap']] = fuel
        for p in supply+decoy_fuel:
            a[p] = fuel
        try:
            out, f = _run(a)
        except AssertionError:
            continue
        visits = f['visits']
        if low and not (len(visits) >= 2 and visits[0]['count'] in (2, 4)
                        and visits[0]['width'] == 2 and visits[1]['count'] == 0):
            continue
        if detour and not (len(visits) >= 2 and visits[0]['reachable'] > 9
                           and visits[0]['reversal'] and visits[1]['count'] > 0
                           and any(v['partial_contact'] for v in visits)):
            continue
        if mode == 'long' and len(visits) < n:
            continue
        if (low or detour) and (not f['decoy_supply'] or not f['unreachable']):
            continue
        return {'input': a, 'output': out}
    raise RuntimeError('Unable to place an informative fuel-shutter scene in bounded attempts')


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer from 2 through 10')
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError('num_test must be 1..4 and smaller than num_train')
    rng = np.random.default_rng(seed)
    base_colors = list(map(int, rng.choice(10, size=6, replace=False)))
    recolor = bool(rng.random() < .65)
    specs = [('low', 2), ('excess', int(rng.integers(2, 4)))]
    specs += [('free', int(rng.integers(1, 5))) for _ in range(num_train-2)]
    # Test shift: more potential relays and offset captures, rather than a
    # compulsory counter load or a different color convention.
    tests = [('long', int(rng.integers(3, 5))) if rng.random() < .5
             else ('free', int(rng.integers(3, 6))) for _ in range(num_test)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, requests in (('train', specs), ('test', tests)):
        for mode, n in requests:
            colors = list(map(int, rng.choice(10, size=6, replace=False))) if recolor else base_colors
            for _ in range(12):
                pair = _make_pair(rng, colors, mode, n)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Distinct input construction exhausted')
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Bad puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    _need(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
          and len(puzzle['train']) > len(puzzle['test']), 'Bad pair counts')
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Bad pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _need(a.shape == b.shape, 'Dimensions must be preserved')
            _need(_key(a) not in seen, 'Duplicate inputs')
            seen.add(_key(a))
            scene = _decode(a)
            expected, f = _run(a, scene)
            _need(np.array_equal(expected, b), 'Incorrect transformation')
            _need(not np.array_equal(a, b), 'Launcher cap must change')
            total = sum(v['count'] for v in f['visits'])
            _need(len(f['original_fuel'])-len(f['remaining']) == total,
                  'Fuel conservation failed')
            for i in f['unvisited']:
                for p in scene['relays'][i]['cells']:
                    _need(a[p] == b[p], 'Unvisited relay changed')
            for p in f['remaining']:
                _need(b[p] == scene['fuel'], 'Uncollected fuel changed')
            _need(np.array_equal(a[scene['wall']], b[scene['wall']]), 'Wall changed')
            for v in f['visits']:
                x = scene['relays'][v['index']]
                for j, p in enumerate(x['core']):
                    _need(b[p] == (scene['fuel'] if j < v['count'] else a[p]), 'Bad unary prefix')
                for p, j in zip(x['ports'], x['indices']):
                    _need(b[p] == (scene['signal'] if j < v['count'] else a[p]), 'Bad shutter opening')
            records[split].append(f)
    train = records['train']
    visits = [v for f in train for v in f['visits']]
    _need(any(v['count'] in (2, 4) and 0 < v['width'] < v['port_size'] for v in visits),
          'Training must show partial fill and reduced port width')
    _need(any(v['count'] == 9 and v['reachable'] > 9 and v['reversal'] for v in visits),
          'Training must distinguish nearest paths from direct distance and capacity')
    _need(any(v['width'] == 0 for v in visits), 'Training must show a sealed relay')
    _need(any(f['decoy_supply'] and f['unreachable'] for f in train),
          'Training must show unvisited eligible collectors and unreachable fuel')
    _need(any(len(f['visits']) >= 2 and f['visits'][1]['count'] > 0 for f in train),
          'Training must show ordered collection from a shared field')
    _need(any(v['partial_contact'] for v in visits), 'Training must show whole-beam offset capture')


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def _targeted_checks():
    """Small explicit counterexamples for the coupled rule, not parent copies."""
    def paint(relays, supply, walls=()):
        a = np.zeros((20, 20), dtype=np.int8)
        for p in walls:
            a[p] = 5
        for relay in relays:
            for p in relay['core']:
                a[p] = 1
            a[relay['core'][4]] = 3
            for p in relay['ports']:
                a[p] = 3
        a[10, 0:2] = 2
        a[10, 2] = 6
        for p in supply:
            a[p] = 6
        return a

    # Every gate threshold is explained by its adjacent row-major slot.
    supply = [(r, c) for r in range(13, 16) for c in range(8, 11)]
    for side in range(4):
        relay = _relay((9, 8), side)
        for n in range(10):
            a = paint([relay], supply[:n])
            b, f = _run(a)
            _need(f['visits'][0]['count'] == n, 'Charge regression')
            _need(f['visits'][0]['width'] == sum(j < n for j in relay['indices']),
                  'Adjacent-slot shutter threshold regression')
            for p, j in zip(relay['ports'], relay['indices']):
                _need(b[p] == (2 if j < n else 3), 'Gate color regression')

    # One extra unit of fuel changes the next target, not just drawn thickness.
    first = _relay((9, 8), 0)
    near = _relay((4, 10), 1, 0, 1)
    far = _relay((1, 6), 3, 1, 1)
    low = paint([first, near, far], [(12, 8), (12, 9)])
    high = paint([first, near, far], [(12, 8), (12, 9), (12, 10)])
    lf, hf = _run(low)[1], _run(high)[1]
    lp, hp = _decode(low), _decode(high)
    _need(lp['relays'][lf['visits'][1]['index']]['pos'] == far['pos'] and
          hp['relays'][hf['visits'][1]['index']]['pos'] == near['pos'],
          'Width-controlled target selection regression')
    _need(hf['visits'][1]['partial_contact'], 'Whole-beam capture regression')

    # Excess fuel on the beam remains fuel; rays do not scoop it up.
    close = [(r, c) for r in range(12, 15) for c in range(8, 11)]
    a = paint([_relay((9, 8), 1)], close+[(10, 3)])
    b, f = _run(a)
    _need(f['visits'][0]['count'] == 9 and b[10, 3] == 6 and b[10, 4] == 2,
          'Capacity or transparent-fuel regression')

    # A returning beam is absorbed without charging a relay twice.
    relays = [_relay((9, 8), 1, 1, 1), _relay((9, 15), 3, 1, 1)]
    loads = supply+[(r, c) for r in range(13, 16) for c in range(15, 18)]
    a = paint(relays, loads)
    _, f = _run(a)
    _need(f['reason'] == 'revisit' and [v['count'] for v in f['visits']] == [9, 9],
          'Cycle absorption or shared-fuel regression')
    walls = [(r, 13) for r in range(20)]
    _, f = _run(paint(relays, loads, walls))
    _need(f['reason'] == 'wall' and len(f['visits']) == 1,
          'Walls must stop the signal and separate fuel')

    # No coordinate or queue-order tie breaker is permitted at capacity.
    d = np.ones((4, 4), dtype=int)
    try:
        _select({(r, c) for r in range(3) for c in range(4)}, d)
    except AssertionError:
        pass
    else:
        raise AssertionError('Capacity tie silently accepted')


def self_test():
    _targeted_checks()
    puzzles, records = [], []
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input'].copy()
                out, f = _run(a)
                _need(np.array_equal(out, pair['output']) and np.array_equal(a, pair['input']),
                      'Solver mismatch or input mutation')
                records.append((split, _decode(a), f))
    _need(len({_puzzle_key(p) for p in puzzles}) == 42, 'Seed diversity collapsed')
    _need(len({f['shape'] for _, _, f in records}) >= 35, 'Canvas variation collapsed')
    _need({f['initial'] for _, _, f in records} == set(range(4)), 'Launcher directions missing')
    _need(len({f['palette'] for _, _, f in records}) >= 35, 'Palette variation collapsed')
    visits = [v for _, _, f in records for v in f['visits']]
    _need({0, 1, 2, 3} <= {v['side'] for v in visits}, 'Port sides missing')
    _need({0, 1, 2, 3} <= {v['width'] for v in visits}, 'Shutter widths missing')
    _need(any(v['partial_contact'] for v in visits), 'Offset capture absent')
    _need(len({len(f['visits']) for _, _, f in records}) >= 4, 'Actual route counts collapsed')
    _need(any(len(s['relays']) >= 5 for _, s, _ in records), 'Larger networks absent')
    _need(_puzzle_key(generate_puzzle(723, 2, 1)) == _puzzle_key(generate_puzzle(723, 2, 1)),
          'Determinism failed')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(913+a*5+b, a, b)
            _need(len(p['train']) == a and len(p['test']) == b, 'Count ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
              for s in ('train', 'test')}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted label accepted')


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': len(HOST_SEEDS)}, separators=(',', ':')))
