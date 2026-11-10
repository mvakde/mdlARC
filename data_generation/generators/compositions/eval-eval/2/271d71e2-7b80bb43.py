"""Counter-powered rail repair: an input-only, four-role ARC family.

The station stays put; its probe walks a simple orthogonal route. Moving costs
one counter cell. Rebuilding a supported three-cell break costs three additional
cells, atomically, and happens only when the probe reaches that break.
"""

from __future__ import annotations

import numbers
import numpy as np


STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _need(np.all((0 <= g) & (g <= 9)), "invalid colors")
    return g.astype(np.int8, copy=True)


def _add(p, d, k=1):
    return p[0] + k*d[0], p[1] + k*d[1]


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _at(mask, p):
    return _inside(mask.shape, p) and bool(mask[p])


def _neighbors(p):
    return [_add(p, d) for d in STEPS]


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while cells:
        first = min(cells)
        cells.remove(first)
        stack, component = [first], {first}
        while stack:
            for p in _neighbors(stack.pop()):
                if p in cells:
                    cells.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _counter_order(bounds, direction):
    """A's travel-up raster, rotated with the station's launch direction."""
    t, b, l, r = bounds
    if direction == (-1, 0):
        return [(y, x) for y in range(t+1, b) for x in range(l+1, r)]
    if direction == (1, 0):
        return [(y, x) for y in range(b-1, t, -1) for x in range(r-1, l, -1)]
    if direction == (0, -1):
        return [(y, x) for x in range(l+1, r) for y in range(b-1, t, -1)]
    if direction == (0, 1):
        return [(y, x) for x in range(r-1, l, -1) for y in range(t+1, b)]
    raise AssertionError("nonorthogonal launch")


def _strong(mask):
    strong = np.zeros_like(mask)
    for m, s in ((mask, strong), (mask.T, strong.T)):
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for k in range(3):
            s[:, k:k+triples.shape[1]] |= triples
    return strong


def _brackets(mask):
    """Visible straight three-cell breaks with continued support at one end."""
    result = []
    for d in ((0, 1), (1, 0)):
        for a in map(tuple, np.argwhere(mask).tolist()):
            z = _add(a, d, 4)
            if (_at(mask, z) and not any(_at(mask, _add(a, d, k)) for k in (1, 2, 3))
                    and (_at(mask, _add(a, d, -1)) or _at(mask, _add(a, d, 5)))):
                result.append((a, z, tuple(_add(a, d, k) for k in (1, 2, 3)), d))
    return result


def _line_evidence(mask):
    """B's two off-axis cells identify a straight three-cell repair order."""
    strong = _strong(mask)
    brackets = _brackets(mask)
    supported = {}
    anchors = set()
    for a, z, gap, d in brackets:
        anchors.update((a, z))
        matches = []
        perpendicular = (d[1], -d[0])
        for anchor, toward in ((a, d), (z, (-d[0], -d[1]))):
            for side in (-1, 1):
                diag = (toward[0]+side*perpendicular[0], toward[1]+side*perpendicular[1])
                spur = tuple(_add(anchor, diag, k) for k in (1, 2))
                if (all(_at(mask, p) and not strong[p] for p in spur)
                        and not _at(mask, _add(anchor, diag, 3))):
                    matches.append(spur)
        _need(len(matches) <= 1, "ambiguous repair spur")
        if matches:
            supported[frozenset(gap)] = matches[0]
    # Separate thin orthogonal rail from short diagonal fragments. Bracket
    # anchors survive even when one retained piece is only a single cell.
    weak = {p for p in map(tuple, np.argwhere(mask & ~strong).tolist()) if p not in anchors}
    diagonal = {p for p in weak if any(_add(p, d) in weak
                for d in ((1, 1), (1, -1), (-1, 1), (-1, -1)))}
    work_spurs = set()
    for spur in supported.values():
        _need(not (work_spurs & set(spur)), "shared spur")
        work_spurs.update(spur)
    _need(work_spurs <= diagonal, "repair evidence is not a short isolated deviation")
    rail = mask.copy()
    for p in diagonal:
        rail[p] = False
    return rail, supported, diagonal - work_spurs


def _ordered_route(virtual, start, end):
    cells = set(map(tuple, np.argwhere(virtual).tolist()))
    _need(start in cells and end in cells and start != end, "missing route endpoints")
    for p in cells:
        degree = sum(q in cells for q in _neighbors(p))
        _need(degree == (1 if p in (start, end) else 2), "route branches or self-touches")
    path, previous, current = [start], None, start
    while current != end:
        choices = [q for q in _neighbors(current) if q in cells and q != previous]
        _need(len(choices) == 1, "ambiguous route")
        previous, current = current, choices[0]
        _need(current not in path, "route cycle")
        path.append(current)
    _need(len(path) == len(cells), "disconnected orthogonal rail")
    return path


def _explain(g, bg, bounds, frame, probe, line):
    t, b, l, r = bounds
    panel = np.zeros(g.shape, bool)
    panel[t:b+1, l:r+1] = True
    outside = ~panel
    probe_points = list(map(tuple, np.argwhere((g == probe) & outside).tolist()))
    _need(len(probe_points) == 1, "one external probe is required")
    start = probe_points[0]
    y, x = start
    launches = []
    if x == r+1 and t < y < b:
        launches.append((0, 1))
    if x == l-1 and t < y < b:
        launches.append((0, -1))
    if y == t-1 and l < x < r:
        launches.append((-1, 0))
    if y == b+1 and l < x < r:
        launches.append((1, 0))
    _need(len(launches) == 1, "probe must launch just outside a frame side")
    direction = launches[0]
    order = _counter_order(bounds, direction)
    values = [int(g[p]) for p in order]
    advanced = values.count(line)
    _need(values == [line]*advanced + [probe]*(len(values)-advanced), "nonprefix counter")
    goal = set(map(tuple, np.argwhere((g == frame) & outside).tolist()))
    _need(len(goal) == 3, "destination must be one three-cell bar")
    ys, xs = {p[0] for p in goal}, {p[1] for p in goal}
    if len(ys) == 1 and len(xs) == 3 and max(xs)-min(xs) == 2:
        center = (next(iter(ys)), min(xs)+1)
        sides = ((-1, 0), (1, 0))
    elif len(xs) == 1 and len(ys) == 3 and max(ys)-min(ys) == 2:
        center = (min(ys)+1, next(iter(xs)))
        sides = ((0, -1), (0, 1))
    else:
        raise AssertionError("destination is not a straight bar")
    mask = ((g == line) | ((g == probe) & outside)) & outside
    rail, supported, clutter = _line_evidence(mask)
    ends = [_add(center, d) for d in sides if _at(rail, _add(center, d))]
    _need(len(ends) == 1, "goal lacks unique rail approach")
    end = ends[0]
    _need({q for p in goal for q in _neighbors(p) if _at(rail, q)} == {end},
          "goal touches the route in multiple places")
    gaps = _brackets(rail)
    virtual = rail.copy()
    gap_map = {}
    for a, z, gap, d in gaps:
        _need(all(int(g[p]) == bg for p in gap), "gap contains another role")
        key = frozenset(gap)
        _need(not any(key & k for k in gap_map), "overlapping breaks")
        gap_map[key] = supported.get(key)
        for p in gap:
            virtual[p] = True
    _need(set(supported) <= set(gap_map), "spur does not belong to the route")
    path = _ordered_route(virtual, start, end)
    _need(len(path) >= 8 and path[1] == _add(start, direction), "route launch disagrees with frame")
    # Every break is straight along this path, not an incidental alignment.
    positions = {p: i for i, p in enumerate(path)}
    for gap in gap_map:
        indices = sorted(positions[p] for p in gap)
        _need(indices == list(range(indices[0], indices[0]+3)), "break is not consecutive")
        i = indices[0]-1
        _need(i >= 0 and i+4 < len(path), "unbracketed route break")
        d = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        _need(all(path[i+k] == _add(path[i], d, k) for k in range(5)), "break bends")
    _need(set(map(int, np.unique(g))) <= {bg, frame, probe, line}, "unexplained color")
    return dict(bg=bg, frame=frame, probe=probe, line=line, bounds=bounds,
                order=order, advanced=advanced, fuel=len(order)-advanced,
                direction=direction, start=start, end=end, goal=goal, rail=rail,
                path=path, gaps=gap_map, clutter=clutter)


def _read_scene(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    _need(len(colors) == 4 and sum(counts == counts.max()) == 1, "four roles and unique dominant background required")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for frame in map(int, colors):
        if frame == bg:
            continue
        for component in _components(g == frame):
            ys, xs = zip(*component)
            t, b, l, r = min(ys), max(ys), min(xs), max(xs)
            if b-t < 2 or r-l < 2 or len(component) != 2*(b-t+r-l):
                continue
            if any(y not in (t, b) and x not in (l, r) for y, x in component):
                continue
            remaining_roles = [int(c) for c in colors if c not in (bg, frame)]
            for probe, line in (remaining_roles, remaining_roles[::-1]):
                try:
                    candidates.append(_explain(g, bg, (t, b, l, r), frame, probe, line))
                except (AssertionError, KeyError, IndexError):
                    pass
    _need(len(candidates) == 1, "input lacks a unique station/route explanation")
    return g, candidates[0]


def _execute(g, scene):
    """One-cell motion, with a separate atomic construction transaction."""
    out = g.copy()
    bg, line, probe = (scene[k] for k in ('bg', 'line', 'probe'))
    for p in scene['clutter']:
        out[p] = bg
    path, rail = scene['path'], scene['rail'].copy()
    fuel, i, repaired = scene['fuel'], 0, []
    stop = 'destination'
    while i < len(path)-1:
        if rail[path[i+1]]:
            if fuel == 0:
                stop = 'fuel'
                break
            fuel -= 1
            i += 1
        else:
            gap = frozenset(path[i+1:i+4])
            _need(gap in scene['gaps'] and len(gap) == 3, "bad route break")
            spur = scene['gaps'][gap]
            if spur is None:
                stop = 'unsupported'
                break
            if fuel < 3:
                stop = 'repair_budget'
                break
            fuel -= 3
            for p in gap:
                rail[p] = True
                out[p] = line
            for p in spur:
                out[p] = bg
            repaired.append(gap)
    consumed = scene['fuel'] - fuel
    for j, p in enumerate(scene['order']):
        out[p] = line if j < scene['advanced']+consumed else probe
    out[path[0]] = line
    out[path[i]] = probe
    return out, dict(moves=i, repaired=repaired, spent=consumed, remaining=fuel,
                     stop=stop, position=path[i])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all conventions, topology, work orders and fuel from this input."""
    g, scene = _read_scene(input_grid)
    return _execute(g, scene)[0]


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _polyline(rng, shape, bounds, minimum_legs, maximum_legs):
    t, b, l, r = bounds
    start = (int(rng.integers(t+1, b)), r+1)
    path, occupied = [start], {start}
    direction = (0, 1)
    wanted = int(rng.integers(minimum_legs, maximum_legs+1))
    lengths = []
    for leg in range(wanted):
        directions = [direction] if leg == 0 else [(-direction[1], direction[0]), (direction[1], -direction[0])]
        choices = []
        for d in directions:
            extension = []
            for length in range(1, 16):
                p = _add(path[-1], d, length)
                if not _inside(shape, p):
                    break
                if t-1 <= p[0] <= b+1 and l-1 <= p[1] <= r+1:
                    break
                previous = extension[-1] if extension else path[-1]
                if p in occupied or any(q in occupied and q != previous for q in _neighbors(p)):
                    break
                extension.append(p)
                if length >= (8 if leg == 0 else 6):
                    choices.append((d, extension[:]))
        if not choices:
            return None
        direction, extension = choices[int(rng.integers(len(choices)))]
        path.extend(extension)
        occupied.update(extension)
        lengths.append(len(extension))
    perpendicular = (direction[1], -direction[0])
    center = _add(path[-1], direction)
    goal = {_add(center, perpendicular, k) for k in (-1, 0, 1)}
    if any(not _inside(shape, p) or p in occupied or
           (t-1 <= p[0] <= b+1 and l-1 <= p[1] <= r+1) for p in goal):
        return None
    if {q for p in goal for q in _neighbors(p) if q in occupied} != {path[-1]}:
        return None
    return path, goal, lengths


def _damage_options(shape, bounds, path, goal):
    t, b, l, r = bounds
    base = set(path) | goal | {(y, x) for y in range(t, b+1) for x in range(l, r+1)}
    choices = []
    for i in range(2, len(path)-5):
        d = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        if not all(path[i+k] == _add(path[i], d, k) for k in range(5)):
            continue
        if path[i-1] != _add(path[i], d, -1) and path[i+5] != _add(path[i], d, 5):
            continue
        gap = tuple(path[i+1:i+4])
        retained = base-set(gap)
        for endpoint in (i, i+4):
            toward = d if endpoint == i else (-d[0], -d[1])
            for side in (-1, 1):
                perpendicular = (side*d[1], -side*d[0])
                diag = (toward[0]+perpendicular[0], toward[1]+perpendicular[1])
                spur = tuple(_add(path[endpoint], diag, k) for k in (1, 2))
                if all(_inside(shape, p) and p not in base and
                       not any(q in retained for q in _neighbors(p)) for p in spur):
                    choices.append(dict(index=i, gap=gap, spur=spur, direction=d,
                                        anchor=path[endpoint], toward=toward, side=perpendicular))
    return choices, base


def _choose_damage(rng, choices, count):
    selected = []
    for j in rng.permutation(len(choices)):
        option = choices[int(j)]
        if any(abs(option['index']-x['index']) < 7 for x in selected):
            continue
        footprint = set(option['gap']+option['spur'])
        if any(any(abs(p[0]-q[0]) <= 1 and abs(p[1]-q[1]) <= 1
                   for p in footprint for q in x['gap']+x['spur']) for x in selected):
            continue
        selected.append(option)
        if len(selected) == count:
            return sorted(selected, key=lambda x: x['index'])
    return None


def _make_pair(rng, palette, lesson, gap_count, degree):
    bg, frame, line, probe = palette
    for attempt in range(500):
        H, W = int(rng.integers(23, 31)), int(rng.integers(23, 31))
        if lesson in ('goal', 'later_budget'):
            ih, iw = int(rng.integers(5, 9)), int(rng.integers(6, 10))
        elif lesson == 'budget':
            ih, iw = int(rng.integers(4, 8)), int(rng.integers(5, 9))
        else:
            ih, iw = int(rng.integers(3, 7)), int(rng.integers(4, 9))
        if lesson in ('random', 'zero') and rng.random() < .25:
            ih = int(rng.integers(1, 4))
            iw = int(rng.integers(1, 6))
        t = int(rng.integers(1, H-ih-2))
        l = int(rng.integers(0, max(1, min(5, W-iw-14))))
        bounds = (t, t+ih+1, l, l+iw+1)
        construction = _polyline(rng, (H, W), bounds,
                                 max(3, gap_count), max(3, gap_count)+degree)
        if construction is None:
            continue
        path, goal, lengths = construction
        choices, base = _damage_options((H, W), bounds, path, goal)
        selected = _choose_damage(rng, choices, gap_count)
        if selected is None:
            continue
        capacity = ih*iw
        if lesson == 'budget':
            fuel = selected[2]['index']+6+int(rng.integers(1, 3))
            flags = [True]*gap_count
        elif lesson == 'break':
            fuel = selected[1]['index']+3+int(rng.integers(4, 9))
            flags = [True, False]+[True]*(gap_count-2)
        elif lesson == 'later_budget':
            fuel = selected[3]['index']+9+int(rng.integers(1, 3))
            flags = [True]*gap_count
        elif lesson == 'atomic':
            fuel = selected[0]['index']+3
            flags = [True]*gap_count
        elif lesson == 'goal':
            fuel = len(path)-1+3*gap_count+int(rng.integers(0, 6))
            flags = [True]*gap_count
        elif lesson == 'zero':
            fuel = 0
            flags = [bool(rng.integers(2)) for _ in selected]
        else:
            fuel = int(rng.integers(capacity+1))
            flags = [bool(rng.random() < .75) for _ in selected]
        if fuel > capacity or lesson in ('budget', 'break') and fuel == capacity:
            continue
        g = np.full((H, W), bg, dtype=np.int8)
        bt, bb, bl, br = bounds
        g[bt:bb+1, bl:br+1] = frame
        order = _counter_order(bounds, (0, 1))
        advanced = capacity-fuel
        for j, p in enumerate(order):
            g[p] = line if j < advanced else probe
        for p in path:
            g[p] = line
        for p in goal:
            g[p] = frame
        g[path[0]] = probe
        gaps, expected_spurs, clutter = {}, set(), set()
        footprint = set(base)
        for option, supported in zip(selected, flags):
            for p in option['gap']:
                g[p] = bg
            if supported:
                for p in option['spur']:
                    g[p] = line
                gaps[frozenset(option['gap'])] = option['spur']
                expected_spurs.update(option['spur'])
                footprint.update(option['spur'])
            else:
                gaps[frozenset(option['gap'])] = None
                # An offset diagonal resembles a work order but its extrapolated
                # anchor is beside the rail, not on its retained endpoint.
                offset_anchor = _add(option['anchor'], option['side'])
                diag = (option['toward'][0]+option['side'][0], option['toward'][1]+option['side'][1])
                near_miss = tuple(_add(offset_anchor, diag, k) for k in (1, 2))
                if all(_inside(g.shape, p) and p not in footprint and
                       not any(q in footprint for q in _neighbors(p)) for p in near_miss):
                    for p in near_miss:
                        g[p] = line
                    clutter.update(near_miss)
                    footprint.update(near_miss)
        # Detached diagonal clutter varies in location, length and count.
        for _ in range(int(rng.integers(1, degree+3))):
            for _ in range(40):
                a = (int(rng.integers(H)), int(rng.integers(W)))
                d = (int(rng.choice((-1, 1))), int(rng.choice((-1, 1))))
                cells = tuple(_add(a, d, k) for k in range(int(rng.integers(2, 4))))
                if all(_inside(g.shape, p) and int(g[p]) == bg and
                       all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) > 1 for q in footprint)
                       for p in cells):
                    for p in cells:
                        g[p] = line
                    clutter.update(cells)
                    footprint.update(cells)
                    break
        rail = np.zeros(g.shape, bool)
        for p in path:
            rail[p] = True
        for option in selected:
            for p in option['gap']:
                rail[p] = False
        blueprint = dict(bg=bg, frame=frame, probe=probe, line=line, bounds=bounds,
                         order=order, advanced=advanced, fuel=fuel, direction=(0, 1),
                         start=path[0], end=path[-1], goal=goal, rail=rail,
                         path=path, gaps=gaps, clutter=clutter)
        try:
            _, scene = _read_scene(g)
            _need(scene['path'] == path and scene['gaps'] == gaps and scene['clutter'] == clutter,
                  "accidental scene-level evidence")
            out, result = _execute(g, blueprint)
            if lesson == 'budget':
                _need(result['stop'] == 'repair_budget' and len(result['repaired']) == 2,
                      "missing budget lesson")
            if lesson == 'break':
                _need(result['stop'] == 'unsupported' and len(result['repaired']) == 1,
                      "missing break lesson")
            if lesson == 'later_budget':
                _need(result['stop'] == 'repair_budget' and len(result['repaired']) == 3,
                      "missing repeated-work shift")
            if lesson == 'goal':
                _need(result['stop'] == 'destination', "missing arrival lesson")
        except (AssertionError, KeyError):
            continue
        # Crop actual content, then add independently sampled asymmetric margins.
        ys, xs = np.nonzero(g != bg)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1
        g, out = g[y0:y1, x0:x1], out[y0:y1, x0:x1]
        margins = [int(rng.integers(0, min(3, 30-n)+1)) for n in g.shape]
        before = [int(rng.integers(m+1)) for m in margins]
        pad = tuple((a, m-a) for a, m in zip(before, margins))
        g, out = np.pad(g, pad, constant_values=bg), np.pad(out, pad, constant_values=bg)
        return dict(input=g.copy(), output=out.copy())
    raise RuntimeError(f"bounded construction failed for {lesson}/{gap_count}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 'num_train', 2, 10)
    t = _count(num_test, 'num_test', 1, 4)
    if n <= t:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    palette_style = str(rng.choice(('fixed', 'pair'), p=(.60, .40)))
    degree = int(rng.choice((0, 1, 2), p=(.25, .50, .25)))
    launch_turn = int(rng.integers(4))
    lessons = ['budget', 'break']
    rng.shuffle(lessons)
    for i in range(n-2):
        lessons.append(('goal', 'atomic')[i] if i < 2 else str(rng.choice(('goal', 'atomic', 'zero', 'random'))))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for i in range(count):
            lesson = lessons[i] if split == 'train' else ('later_budget' if i == 0 else str(rng.choice(('break', 'random', 'atomic', 'goal'))))
            gap_count = (3 if lesson in ('break', 'budget') else int(rng.integers(2, 4))) if split == 'train' else 4
            local_palette = palette if palette_style == 'fixed' else tuple(map(int, rng.choice(10, 4, replace=False)))
            for _ in range(24):
                pair = _make_pair(rng, local_palette, lesson, gap_count, degree)
                turns = (launch_turn+i) % 4 if i < 2 and split == 'train' else int(rng.integers(4))
                pair = {k: np.rot90(v, turns).copy() for k, v in pair.items()}
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not construct distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    g, s = _read_scene(pair['input'])
    out, result = _execute(g, s)
    return s, result, out


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "wrong puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, t = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= n <= 10 and 1 <= t <= 4 and n > t, "invalid counts")
    seen, lessons, training_directions = set(), set(), set()
    max_train_gaps = 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "wrong pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, "shape changed")
            key = (inp.shape, inp.tobytes())
            _need(key not in seen, "duplicate input")
            seen.add(key)
            s, result, expected = _features(pair)
            _need(np.array_equal(out, expected), "incorrect or incomplete output")
            _need(result['spent'] == result['moves']+3*len(result['repaired']), "resource conservation failed")
            _need(sum(out[p] == s['probe'] for p in s['order']) == result['remaining'], "counter disagrees with work")
            _need(all(out[p] == s['frame'] for p in s['goal']), "destination changed")
            t0, b0, l0, r0 = s['bounds']
            border = {(y, x) for y in range(t0, b0+1) for x in range(l0, r0+1)
                      if y in (t0, b0) or x in (l0, r0)}
            _need(all(out[p] == s['frame'] for p in border), "station frame changed")
            _need(out[result['position']] == s['probe'], "probe missing")
            for gap, spur in s['gaps'].items():
                if gap not in result['repaired']:
                    _need(all(out[p] == s['bg'] for p in gap), "unvisited/unaffordable gap repaired")
                    if spur:
                        _need(all(out[p] == s['line'] for p in spur), "unfulfilled work order erased")
                else:
                    _need(all(out[p] in (s['line'], s['probe']) for p in gap), "partial bridge")
                    _need(all(out[p] == s['bg'] for p in spur), "fulfilled work spur retained")
            _need(all(out[p] == s['bg'] for p in s['clutter']), "diagonal clutter retained")
            if split == 'train':
                training_directions.add(s['direction'])
                max_train_gaps = max(max_train_gaps, len(s['gaps']))
                if result['stop'] == 'repair_budget' and result['remaining'] in (1, 2) and len(result['repaired']) >= 2:
                    lessons.add('budget')
                if result['stop'] == 'unsupported' and result['remaining'] >= 4 and result['repaired']:
                    lessons.add('break')
            else:
                _need(len(s['gaps']) > max_train_gaps, "test lacks the longer work-sequence shift")
    _need({'budget', 'break'} <= lessons, "training lacks the two distinguishing lessons")
    _need(len(training_directions) >= 2, "training does not expose rotated counter order")


def self_test():
    """Bounded 42-seed sweep, semantic counterexamples and label mutations."""
    shapes, palettes, route_shapes, capacities, fuels, moves = set(), set(), set(), set(), set(), set()
    launches, repair_axes, spur_sides, stops, gap_counts, turns = set(), set(), set(), set(), set(), set()
    unique = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple((p['input'].shape, p['input'].tobytes(), p['output'].tobytes())
                          for split in puzzle.values() for p in split)
        unique.add(signature)
        for split in puzzle.values():
            for pair in split:
                inp = pair['input'].copy()
                s, result, out = _features(pair)
                _need(np.array_equal(solve(inp), pair['output']) and np.array_equal(inp, pair['input']), "solver mutation/agreement")
                shapes.add(inp.shape)
                palettes.add(tuple(s[k] for k in ('bg', 'frame', 'line', 'probe')))
                capacities.add(len(s['order']))
                fuels.add(s['fuel'])
                moves.add(result['moves'])
                launches.add(s['direction'])
                stops.add(result['stop'])
                gap_counts.add(len(s['gaps']))
                deltas = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(s['path'], s['path'][1:])]
                turns.add(sum(a != b for a, b in zip(deltas, deltas[1:])))
                origin = s['path'][0]
                route_shapes.add(tuple((p[0]-origin[0], p[1]-origin[1]) for p in s['path']))
                for gap, spur in s['gaps'].items():
                    repair_axes.add('h' if len({p[0] for p in gap}) == 1 else 'v')
                    if spur:
                        spur_sides.add((spur[1][0]-spur[0][0], spur[1][1]-spur[0][1]))
                # Rotations preserve both the counter raster and route semantics.
                for k in (1, 2, 3):
                    _need(np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k)), "rotation inconsistency")
        if seed == 0:
            repeat = generate_puzzle(seed)
            _need(all(np.array_equal(p[k], q[k]) for split in puzzle
                      for p, q in zip(puzzle[split], repeat[split]) for k in p), "nondeterminism")
            # Removing a whole completed repair must fail, even if every other
            # output edit remains correct (the parent's partial-label defect).
            bad = {split: [{k: v.copy() for k, v in p.items()} for p in ps]
                   for split, ps in puzzle.items()}
            pair = bad['test'][0]
            s, result, _ = _features(pair)
            gap = result['repaired'][0]
            for p in gap:
                pair['output'][p] = s['bg']
            for p in s['gaps'][gap]:
                pair['output'][p] = s['line']
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('validator accepted an omitted repair')
            # A visible counter intervention alters which work gets done.
            original = puzzle['train'][next(i for i, p in enumerate(puzzle['train'])
                                           if _features(p)[1]['stop'] == 'repair_budget')]['input']
            g, s = _read_scene(original)
            exhausted = g.copy()
            for p in s['order']:
                exhausted[p] = s['line']
            answer = solve(exhausted)
            _need(not np.array_equal(answer, solve(g)), "omitting A has no effect")
            # Remove the first supported spur without touching fuel or rails.
            first = min((gap for gap, spur in s['gaps'].items() if spur),
                        key=lambda gap: min(s['path'].index(p) for p in gap))
            no_evidence = g.copy()
            for p in s['gaps'][first]:
                no_evidence[p] = s['bg']
            _need(not np.array_equal(solve(no_evidence), solve(g)), "omitting B has no effect")
    _need(len(unique) == 42, "seed diversity collapsed")
    _need(len(shapes) >= 40 and len(route_shapes) >= 150, "geometry diversity collapsed")
    _need(len(palettes) >= 30 and len(capacities) >= 15 and len(fuels) >= 20 and len(moves) >= 20,
          "palette/counter variation collapsed")
    _need(launches == set(STEPS) and repair_axes == {'h', 'v'}, "orientation coverage")
    _need(spur_sides == {(1, 1), (1, -1), (-1, 1), (-1, -1)}, "spur orientation coverage")
    _need({'fuel', 'destination', 'repair_budget', 'unsupported'} <= stops, "stopping coverage")
    _need({2, 3, 4} <= gap_counts and len(turns) >= 3, "work/topology variation collapsed")
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            puzzle = generate_puzzle(123, n, t)
            _need(len(puzzle['train']) == n and len(puzzle['test']) == t, "counts ignored")
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, shapes=len(shapes), palettes=len(palettes), routes=len(route_shapes),
                capacities=len(capacities), fuel_values=len(fuels), movements=len(moves),
                stops=sorted(stops), turn_counts=sorted(turns), gap_counts=sorted(gap_counts))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
