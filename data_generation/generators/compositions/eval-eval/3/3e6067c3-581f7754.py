"""Selective bridge assembly: ordered routing through registered loose pieces.

Only chosen pieces move. Unchosen ORIGINAL footprints determine clearance.
No file access, parent imports, generation records, or training data enter solve.
"""
import numpy as np

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_SEEDS = (0, 793886299, 337333575, 985866833, 24541737, 83858546,
          120225869, 601322891, 767368667, 906863832, 787875062,
          460894076, 892298636, 600359695, 99827478, 74507780,
          784645077, 252273771, 82055591, 194451664, 421824498,
          578788704, 254580505, 48853183, 930771475, 63754448,
          609416712, 961821065, 936366201, 114211813, 33062803,
          915833743, 236971016, 701076324, 262932419, 463779747,
          620175979, 978571471, 974430588, 718867788, 290071847,
          151651728)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'bad dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'noninteger cells')
    _require(np.all((0 <= a) & (a <= 9)), 'bad colors')
    return a


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, todo = {p}, [p]
        while todo:
            r, c = todo.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(component)
    return result


def _decode(value):
    grid = _grid(value)
    h, w = grid.shape
    bg = int(grid[0, 0])
    _require(all(int(grid[p]) == bg for p in ((0, w-1), (h-1, 0), (h-1, w-1))),
             'corners must encode background')
    anchors, tokens, bodies = {}, [], []
    for component in _components(map(tuple, np.argwhere(grid != bg))):
        if len(component) != 1:
            bodies.append(component)
            continue
        p = next(iter(component))
        r, c = p
        if r in (0, h-1) or c in (0, w-1):
            anchors.setdefault(int(grid[p]), set()).add(p)
        else:
            tokens.append(p)
    _require(2 <= len(anchors) <= 7 and 3 <= len(bodies) <= 12, 'bad object counts')
    _require(3 <= len(tokens) <= 8, 'bad key length')
    tokens.sort(key=lambda p: p[1])
    ky = tokens[0][0]
    _require(all(r == ky for r, c in tokens), 'key must occupy one row')
    _require(all(b[1]-a[1] == 2 for a, b in zip(tokens, tokens[1:])), 'key spacing')
    _require(max(r for body in bodies for r, c in body) + 1 < ky, 'key not detached below bodies')
    key = [int(grid[p]) for p in tokens]
    _require(set(key) <= set(anchors) and len(set(key)) >= 2, 'unanchored key identity')
    systems, fixed = {}, {p: int(grid[p]) for p in tokens}
    for color, points in anchors.items():
        horizontal = len({r for r, c in points}) == 1 and all(c in (0, w-1) for r, c in points)
        vertical = len({c for r, c in points}) == 1 and all(r in (0, h-1) for r, c in points)
        _require(horizontal != vertical, 'ambiguous guide')
        axis = 0 if horizontal else 1
        guide = next(iter(points))[axis]
        _require(0 < guide < grid.shape[axis]-1 and len(points) <= 2, 'corner/inconsistent anchor')
        systems[color] = (axis, guide)
        fixed.update({p: color for p in points})
    objects = []
    for body in bodies:
        marks = [p for p in body if int(grid[p]) in systems]
        _require(len(marks) == 1, 'each body needs exactly one anchored port')
        port = marks[0]
        color = int(grid[port])
        axis, guide = systems[color]
        delta = [0, 0]
        delta[axis] = guide - port[axis]
        dest = port[0] + delta[0], port[1] + delta[1]
        target = {(r+delta[0], c+delta[1]): int(grid[r, c]) for r, c in body}
        objects.append(dict(points=body, port=port, marker=color, axis=axis,
                            delta=tuple(delta), dest=dest, target=target))
    _require(set(o['marker'] for o in objects) == set(systems), 'anchor without piece')
    info = dict(grid=grid, bg=bg, key=key, key_y=ky, anchors=anchors,
                fixed=fixed, systems=systems, objects=objects)
    links = {}
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i != j:
                gap = _gap(a, b)
                if gap:
                    links[i, j] = gap
    info['links'] = links
    return info


def _gap(a, b):
    """Actual facing surfaces on the common port line, in either direction."""
    u, v = a['dest'], b['dest']
    if u == v:
        return None
    if u[0] == v[0]:
        axis = 1
    elif u[1] == v[1]:
        axis = 0
    else:
        return None
    other = 1-axis
    left, right = (a, b) if u[axis] < v[axis] else (b, a)
    lo = max(p[axis] for p in left['target'] if p[other] == u[other])
    hi = min(p[axis] for p in right['target'] if p[other] == u[other])
    if hi-lo <= 1:
        return None
    return {(u[0], x) if axis == 1 else (x, u[1]) for x in range(lo+1, hi)}


def _assembled(info, path, stationary=True):
    """Return the assembly, or None for a collision/illegal link.

    stationary=False is a diagnostic geometric hypothesis: omit every unselected
    body. The real rule always preserves unselected original footprints.
    """
    grid, bg = info['grid'], info['bg']
    h, w = grid.shape
    result = np.full(grid.shape, bg, dtype=grid.dtype)
    for p, color in info['fixed'].items():
        result[p] = color
    chosen = set(path)
    if stationary:
        for i, o in enumerate(info['objects']):
            if i not in chosen:
                for p in o['points']:
                    result[p] = grid[p]
    for i in path:
        for p, color in info['objects'][i]['target'].items():
            r, c = p
            if not (0 <= r < h and 0 <= c < w) or result[p] != bg:
                return None
            result[p] = color
    for i, j in zip(path, path[1:]):
        gap = info['links'].get((i, j))
        if gap is None or any(result[p] != bg for p in gap):
            return None
        for p in gap:
            result[p] = info['objects'][i]['marker']
    return result


def _routes(info, stationary=True, limit=2):
    objects, key = info['objects'], info['key']
    candidates = [[i for i, o in enumerate(objects) if o['marker'] == color] for color in key]
    found = []

    def visit(path, used, footprints, painted):
        if len(found) >= limit:
            return
        if len(path) == len(key):
            if _assembled(info, path, stationary) is not None:
                found.append(tuple(path))
            return
        for i in candidates[len(path)]:
            if i in used:
                continue
            target = set(objects[i]['target'])
            if target & (footprints | painted | set(info['fixed'])):
                continue
            gap = set()
            if path:
                gap = info['links'].get((path[-1], i))
                if gap is None or gap & (footprints | painted | target | set(info['fixed'])):
                    continue
            visit(path + [i], used | {i}, footprints | target, painted | gap)
    visit([], set(), set(), set())
    return found


def _analysis(value):
    info = _decode(value)
    routes = _routes(info)
    _require(len(routes) == 1, 'itinerary must have exactly one feasible assembly')
    info['route'] = routes[0]
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode hypothetical registrations and select the one clear ordered chain."""
    info = _analysis(input_grid)
    return _assembled(info, info['route'])


def _shape(rng, small=False):
    kind = int(rng.integers(5))
    upper = 3 if small else 4
    h, w = int(rng.integers(2, upper+1)), int(rng.integers(2, upper+1))
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        h = w = 3 if small else int(rng.integers(3, 6))
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h-1) or c in (0, w-1)}
    elif kind == 2:
        cells = {(r, 0) for r in range(h)} | {(h-1, c) for c in range(w)}
    elif kind == 3:
        cells = {(0, c) for c in range(w)} | {(r, w//2) for r in range(h)}
    else:
        cells = {(0, 0)}
        for _ in range(int(rng.integers(2, 6 if small else 13))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _DIRS} - cells)
            frontier = [p for p in frontier if max(r for r, c in cells | {p})-min(r for r, c in cells | {p}) < upper
                        and max(c for r, c in cells | {p})-min(c for r, c in cells | {p}) < upper]
            if frontier:
                cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    points = sorted(cells)
    eccentric = [p for p in points if any(2*p[k] != min(q[k] for q in points)+max(q[k] for q in points) for k in (0, 1))]
    pool = eccentric if rng.random() < .8 else points
    mark = pool[int(rng.integers(len(pool)))]
    return [(r-mark[0], c-mark[1]) for r, c in points]


def _touches(points, occupied):
    return bool(points & occupied) or any((r+dr, c+dc) in occupied for r, c in points for dr, dc in _DIRS)


def _draw_scene(rng, ports, axes, length, palette, small=False, pinned=None):
    groups = list(dict.fromkeys((axis, p[axis]) for p, axis in zip(ports, axes)))
    if not 2 <= len(groups) <= 7:
        return None
    bg = int(palette[0])
    markers = {group: int(palette[i+1]) for i, group in enumerate(groups)}
    payload = list(map(int, palette[len(groups)+1:]))
    objects = []
    for p, axis in zip(ports, axes):
        offsets = _shape(rng, small)
        marker = markers[axis, p[axis]]
        body = int(rng.choice(payload))
        colors = {q: body for q in offsets}
        colors[0, 0] = marker
        if len(offsets) >= 6 and len(payload) > 1 and rng.random() < .35:
            q = [q for q in offsets if q != (0, 0)][int(rng.integers(len(offsets)-1))]
            colors[q] = int(rng.choice([c for c in payload if c != body]))
        target = {(p[0]+r, p[1]+c): color for (r, c), color in colors.items()}
        objects.append(dict(dest=p, marker=marker, axis=axis, offsets=colors, target=target))
    max_r = max(r for o in objects for r, c in o['target'])
    max_c = max(c for o in objects for r, c in o['target'])
    if min(r for o in objects for r, c in o['target']) < 1 or min(c for o in objects for r, c in o['target']) < 1:
        return None
    min_h, min_w = max_r+6, max(max_c+3, 2*length+3)
    if min_h > 30 or min_w > 30:
        return None
    h = int(rng.integers(min_h, min(30, min_h+5)+1))
    w = int(rng.integers(min_w, min(30, min_w+6)+1))
    if rng.random() < .12:
        h = w = 30
    grid = np.full((h, w), bg, dtype=np.int8)
    ky = int(rng.integers(max_r+3, h-2))
    kx = int(rng.integers(1, w-2*length+1))
    key = [o['marker'] for o in objects[:length]]
    if len(set(key)) < 2:
        return None
    fixed = {(ky, kx+2*i): color for i, color in enumerate(key)}
    for (axis, guide), marker in markers.items():
        ends = (0, 1) if rng.random() < .3 else (int(rng.integers(2)),)
        for end in ends:
            p = (guide, 0 if end == 0 else w-1) if axis == 0 else (0 if end == 0 else h-1, guide)
            if _touches({p}, set(fixed)):
                return None
            fixed[p] = marker
    for p, color in fixed.items():
        grid[p] = color
    # Known intended final footprints/links guide construction only.
    intended = set(fixed)
    for o in objects[:length]:
        if set(o['target']) & intended:
            return None
        intended.update(o['target'])
    gaps = []
    for a, b in zip(objects[:length], objects[1:length]):
        gap = _gap(a, b)
        if not gap or gap & intended or any(gap & g for g in gaps):
            return None
        gaps.append(gap)
    intended.update(set().union(*gaps))
    occupied = set(fixed)
    order = list(range(length, len(objects))) + list(map(int, rng.permutation(length)))
    if pinned:
        order = list(pinned) + [i for i in order if i not in pinned]
    for i in order:
        o = objects[i]
        axis = o['axis']
        choices = []
        for coordinate in range(1, (h, w)[axis]-1):
            source = list(o['dest'])
            source[axis] = coordinate
            source = tuple(source)
            if pinned and i in pinned and source != pinned[i]:
                continue
            points = {(source[0]+r, source[1]+c) for r, c in o['offsets']}
            if not all(1 <= r <= ky-2 and 1 <= c < w-1 for r, c in points):
                continue
            if _touches(points, occupied) or (i >= length and points & intended):
                continue
            choices.append((source, points))
        if not choices:
            return None
        moved = [v for v in choices if abs(v[0][axis]-o['dest'][axis]) >= 2]
        pool = moved if moved and rng.random() < .88 else choices
        source, points = pool[int(rng.integers(len(pool)))]
        o['points'], o['port'] = points, source
        occupied.update(points)
        for (r, c), color in o['offsets'].items():
            grid[source[0]+r, source[1]+c] = color
    # Build the output from construction records, then decode/validate afresh.
    out = grid.copy()
    for o in objects[:length]:
        for p in o['points']:
            out[p] = bg
    for o in objects[:length]:
        for p, color in o['target'].items():
            if out[p] != bg:
                return None
            out[p] = color
    for a, gap in zip(objects, gaps):
        for p in gap:
            if out[p] != bg:
                return None
            out[p] = a['marker']
    try:
        info = _analysis(grid)
    except ValueError:
        return None
    if not np.array_equal(out, _assembled(info, info['route'])):
        return None
    if not any(info['objects'][i]['delta'] != (0, 0) for i in info['route']):
        return None
    if not any(info['objects'][i]['delta'] != (0, 0)
               and any(2*info['objects'][i]['port'][k] !=
                       min(p[k] for p in info['objects'][i]['points'])+
                       max(p[k] for p in info['objects'][i]['points']) for k in (0, 1))
               for i in info['route']):
        return None
    if not any(o['delta'] != (0, 0) for i, o in enumerate(info['objects']) if i not in info['route']):
        return None
    if all(info['objects'][a]['port'][0] == info['objects'][b]['port'][0]
           or info['objects'][a]['port'][1] == info['objects'][b]['port'][1]
           for a, b in zip(info['route'], info['route'][1:])):
        return None
    return dict(input=grid, output=out)


def _clearance_scene(rng, palette):
    # Two geometric endings; a stationary non-key blocker removes the wrong one.
    # Its hypothetical destination also overlaps the correct ending's port.
    for _ in range(100):
        r0, c0 = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        r1 = r0+int(rng.integers(6, 9))
        r2 = r1+int(rng.integers(6, 9))
        c1 = c0+int(rng.integers(7, 14))
        rb = r0+3
        ports = [(r0, c0), (r1, c0), (r1, c1), (r2, c1), (r0, c1), (r2, c1)]
        axes = [1, 1, 0, 1, 1, 0]
        blocker = (rb, c1)
        if rng.random() < .5:
            ports = [(c, r) for r, c in ports]
            axes = [1-a for a in axes]
            blocker = blocker[::-1]
        # Reflections affect the board's geometry, not the key reading direction.
        for axis in (0, 1):
            if rng.random() < .5:
                extent = max(p[axis] for p in ports)+3
                ports = [tuple(extent-v if k == axis else v for k, v in enumerate(p)) for p in ports]
                blocker = tuple(extent-v if k == axis else v for k, v in enumerate(blocker))
        pair = _draw_scene(rng, ports, axes, 4, palette, small=True, pinned={5: blocker})
        if pair is not None and len(_routes(_decode(pair['input']), stationary=False)) >= 2:
            return pair
    return _clearance_fallback(palette)


def _walk(rng, rows, cols, length, both):
    universe = [(r, c) for r in range(rows) for c in range(cols)]
    budget = 1000
    def visit(path):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            axes = {0 if a[1] == b[1] else 1 for a, b in zip(path, path[1:])}
            return path if not both or len(axes) == 2 else None
        r, c = path[-1]
        # Visibility jumps over empty lattice positions add genuinely long links.
        # Full pixel/footprint checks later reject crossed or obstructed routes.
        choices = [(rr, cc) for rr, cc in universe if (rr, cc) not in path
                   and ((rr == r and 1 <= abs(cc-c) <= 2)
                        or (cc == c and 1 <= abs(rr-r) <= 2))
                   and not any((a == rr == r and min(c, cc) < b < max(c, cc))
                               or (b == cc == c and min(r, rr) < a < max(r, rr))
                               for a, b in path)]
        for j in rng.permutation(len(choices)):
            found = visit(path+[choices[int(j)]])
            if found is not None:
                return found
        return None
    return visit([universe[int(rng.integers(len(universe)))]])


def _random_scene(rng, palette, test=False):
    for _ in range(140):
        length = int(rng.integers(6, 9) if test else rng.integers(3, 6))
        unused = int(rng.integers(1, 4))
        options = [(r, c) for r in range(2, 5) for c in range(2, 5)
                   if length+unused <= r*c]
        rows, cols = options[int(rng.integers(len(options)))]
        path = _walk(rng, rows, cols, length, both=True)
        if path is None:
            continue
        xs, ys = [int(rng.integers(3, 6))], [int(rng.integers(3, 6))]
        for _ in range(cols-1):
            xs.append(xs[-1]+int(rng.integers(5, 8)))
        for _ in range(rows-1):
            ys.append(ys[-1]+int(rng.integers(5, 8)))
        remaining = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in path]
        chosen = [remaining[int(i)] for i in rng.choice(len(remaining), unused, replace=False)]
        cells = path+chosen
        ports = [(ys[r], xs[c]) for r, c in cells]
        axes = list(map(int, rng.integers(2, size=len(cells))))
        # Include a same-identity unused alternative, not only absent-key clutter.
        matches = [(j, axis) for j in range(length, len(ports)) for axis in (0, 1)
                   if any(axes[k] == axis and ports[k][axis] == ports[j][axis] for k in range(length))]
        if not matches:
            continue
        j, axis = matches[int(rng.integers(len(matches)))]
        axes[j] = axis
        pair = _draw_scene(rng, ports, axes, length, palette)
        if pair is not None:
            return pair
    return _ordinary_fallback(palette)


def _fixed_pair(palette, ports, axes, origins, path, offsets, key_y=27):
    """Finite-search fallback with proved spacious geometry; no new output rule."""
    grid = np.full((30, 30), int(palette[0]), dtype=np.int8)
    groups = list(dict.fromkeys((a, p[a]) for p, a in zip(ports, axes)))
    marks = {g: int(palette[i+1]) for i, g in enumerate(groups)}
    body = int(palette[len(groups)+1])
    objects = []
    for p, axis, origin in zip(ports, axes, origins):
        marker = marks[axis, p[axis]]
        target, points = {}, set()
        for r, c in offsets:
            color = marker if (r, c) == (0, 0) else body
            q = origin[0]+r, origin[1]+c
            grid[q] = color
            points.add(q)
            target[p[0]+r, p[1]+c] = color
        objects.append(dict(dest=p, target=target, points=points, marker=marker))
    for (axis, guide), marker in marks.items():
        grid[(guide, 0) if axis == 0 else (0, guide)] = marker
    for k, i in enumerate(path):
        grid[key_y, 3+2*k] = objects[i]['marker']
    out = grid.copy()
    for i in path:
        for p in objects[i]['points']:
            out[p] = palette[0]
    for i in path:
        for p, color in objects[i]['target'].items():
            assert out[p] == palette[0]
            out[p] = color
    for i, j in zip(path, path[1:]):
        for p in _gap(objects[i], objects[j]):
            assert out[p] == palette[0]
            out[p] = objects[i]['marker']
    assert np.array_equal(solve(grid), out)
    return dict(input=grid, output=out)


def _ordinary_fallback(palette):
    return _fixed_pair(palette,
                       [(4, 5), (4, 12), (4, 19), (12, 19), (12, 12), (12, 5), (20, 5)],
                       [1, 1, 1, 0, 0, 0, 1],
                       [(4, 2), (4, 9), (4, 16), (8, 19), (17, 12), (23, 5), (20, 23)],
                       list(range(6)), [(0, 0), (1, 0), (1, 1)])


def _clearance_fallback(palette):
    return _fixed_pair(palette,
                       [(4, 5), (12, 5), (12, 19), (20, 19), (4, 19), (20, 19)],
                       [1, 1, 0, 1, 1, 0],
                       [(4, 9), (12, 2), (16, 19), (20, 13), (4, 24), (8, 19)],
                       list(range(4)), [(0, 0), (1, 0), (1, 1)])


def _counts(rng, num_train, num_test):
    def check(value, lo, hi, name):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(name+' is outside its legal integer range')
        return int(value)
    a, b = check(num_train, 2, 10, 'num_train'), check(num_test, 1, 4, 'num_test')
    if a is None and b is None:
        b = int(rng.integers(1, 5))
        a = int(rng.integers(b+1, 11))
    elif a is None:
        a = int(rng.integers(b+1, 11))
    elif b is None:
        b = int(rng.integers(1, min(4, a-1)+1))
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    return a, b


def _signature(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    nt, nq = _counts(rng, num_train, num_test)
    base = rng.permutation(10)
    stable = bool(rng.random() < .3)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            palette = base if stable else rng.permutation(10)
            if split == 'train' and index == 1:
                pair = _clearance_scene(rng, palette)
            else:
                pair = _random_scene(rng, palette, test=split == 'test')
            if _signature(pair['input']) in seen:
                # 40 injective color permutations exceed the 13 possible earlier
                # inputs. This only guards a rare repeated bounded fallback.
                for multiplier in (1, 3, 7, 9):
                    for shift in range(10):
                        mapping = (multiplier*np.arange(10)+shift) % 10
                        alternative = {k: mapping[v].astype(np.int8) for k, v in pair.items()}
                        if _signature(alternative['input']) not in seen:
                            pair = alternative
                            break
                    else:
                        continue
                    break
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _analysis(pair['input'])
    route, objects = info['route'], info['objects']
    moving = [objects[i] for i in route if objects[i]['delta'] != (0, 0)]
    unused = [o for i, o in enumerate(objects) if i not in route]
    # Projecting everything is often not even a legal assembly.
    all_targets = [set(o['target']) for o in objects]
    collision = any(a & b for i, a in enumerate(all_targets) for b in all_targets[i+1:])
    source_broken = any(objects[a]['port'][0] != objects[b]['port'][0]
                        and objects[a]['port'][1] != objects[b]['port'][1]
                        for a, b in zip(route, route[1:]))
    eccentric = sum(any(2*o['port'][k] != min(p[k] for p in o['points'])+
                        max(p[k] for p in o['points']) for k in (0, 1)) for o in moving)
    return dict(info=info, length=len(route), unused=len(unused),
                guides=len(info['systems']), shape=tuple(pair['input'].shape),
                repeats=len(set(info['key'])) < len(info['key']),
                geometric_ambiguity=len(_routes(info, stationary=False)) > 1,
                all_projection_collision=collision, moving=len(moving),
                unused_would_move=any(o['delta'] != (0, 0) for o in unused),
                source_broken=source_broken, eccentric_moving=eccentric,
                axes=tuple(sorted({o['axis'] for o in objects})),
                gaps=tuple(len(info['links'][a, b]) for a, b in zip(route, route[1:])),
                sizes=tuple(sorted(len(o['points']) for o in objects)),
                bg=info['bg'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in seen, 'duplicate input'
            seen.add(signature)
            f = _features(dict(input=inp, output=out))
            info = f['info']
            assert np.array_equal(out, _assembled(info, info['route'])), 'incorrect assembly'
            assert f['moving'] and f['unused_would_move'] and f['source_broken'], 'missing parent counterexample'
            # Every port/payload offset, fixed cue, and stationary piece is checked
            # from grids, independently of construction-time object records.
            for p, color in info['fixed'].items():
                assert out[p] == color, 'fixed cue changed'
            for i, o in enumerate(info['objects']):
                if i in info['route']:
                    assert all(out[p] == color for p, color in o['target'].items())
                    assert sum(d != 0 for d in o['delta']) <= 1
                    assert o['delta'][1-o['axis']] == 0
                else:
                    assert all(out[p] == inp[p] for p in o['points']), 'unused piece changed'
            features[split].append(f)
    train = features['train']
    assert any(f['geometric_ambiguity'] and f['all_projection_collision'] and f['repeats'] for f in train), 'conditional-clearance witness missing'
    assert any(f['axes'] == (0, 1) for f in train), 'mixed guides not demonstrated'
    assert any(len(set(f['gaps'])) > 1 for f in train), 'unequal-gap evidence missing'
    assert any(f['eccentric_moving'] for f in train), 'registration handle evidence missing'


def _fingerprint(puzzle):
    return tuple(tuple((_signature(np.asarray(p['input'])), _signature(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed semantics/diversity sweep, metamorphisms and negative cases."""
    shapes, structures, palettes, sizes, lengths, guide_counts = set(), set(), set(), set(), set(), set()
    fingerprints = set()
    for seed in _SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(f['shape'])
                sizes.update(f['sizes'])
                lengths.add(f['length'])
                guide_counts.add(f['guides'])
                palettes.add(f['bg'])
                roles = {}
                structures.add((f['shape'], tuple(roles.setdefault(int(c), len(roles)) for c in pair['input'].ravel())))
                assert np.array_equal(solve(pair['input']), pair['output'])
    assert len(fingerprints) == 42 and len(shapes) >= 30 and len(structures) >= 150
    assert len(sizes) >= 8 and lengths >= {3, 4, 5, 6} and len(guide_counts) >= 4
    assert len(palettes) >= 8
    p = generate_puzzle(0, 2, 1)
    assert _fingerprint(p) == _fingerprint(generate_puzzle(0, 2, 1))
    # Arbitrary per-example color bijections preserve all visible roles.
    rng = np.random.default_rng(991)
    q = {'train': [], 'test': []}
    for s in q:
        for pair in p[s]:
            colors = rng.permutation(10)
            q[s].append({k: colors[v].astype(np.int8) for k, v in pair.items()})
    validate_generated_puzzle(q)
    # Horizontal reflection reverses the key too: restore its token order.
    for pair in p['train']:
        a, b = np.fliplr(pair['input']).copy(), np.fliplr(pair['output']).copy()
        info = _decode(a)
        xs = sorted(c for r, c in info['fixed'] if r == info['key_y'] and 0 < c < a.shape[1]-1)
        values = [a[info['key_y'], c] for c in xs][::-1]
        for c, color in zip(xs, values):
            a[info['key_y'], c] = b[info['key_y'], c] = color
        assert np.array_equal(solve(a), b)
    # Force both deterministic fallbacks through input-only decoding.
    for maker in (_ordinary_fallback, _clearance_fallback):
        pair = maker(np.arange(10))
        assert np.array_equal(solve(pair['input']), pair['output'])
    # A valid broader member: all pieces selected, two DIFFERENT identities on
    # the same vertical guide, and a whole source piece touching the left edge.
    a, b = np.zeros((30, 30), np.int8), np.zeros((30, 30), np.int8)
    anchors = {(0, 5): 1, (29, 5): 2, (12, 29): 3}
    for pos, color in anchors.items():
        a[pos] = b[pos] = color
    for x, color in ((3, 1), (5, 2), (7, 3)):
        a[27, x] = b[27, x] = color
    for marker, origin, target in ((1, (4, 11), (4, 5)),
                                    (2, (12, 0), (12, 5)),
                                    (3, (19, 19), (12, 19))):
        for r, c in ((0, 0), (1, 0), (1, 1)):
            color = marker if (r, c) == (0, 0) else 4
            a[origin[0]+r, origin[1]+c] = color
            b[target[0]+r, target[1]+c] = color
    b[6:12, 5], b[12, 6:19] = 1, 2
    assert np.array_equal(solve(a), b)
    # An unused piece's hypothetical projection can leave the canvas. It stays
    # fully visible in its ORIGINAL location and is neither clipped nor moved.
    for pos, color in {(1, 0): 5, (18, 25): 6, (19, 25): 6, (20, 25): 5}.items():
        a[pos] = b[pos] = color
    assert np.array_equal(solve(a), b)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            z = generate_puzzle(100+nt*10+nq, nt, nq)
            assert len(z['train']) == nt and len(z['test']) == nq
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for coordinate in ('anchor', 'payload', 'stationary', 'erasure'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad['train'][1]
        f = _features(pair)
        info, out = f['info'], pair['output']
        if coordinate == 'anchor':
            pos = next(iter(next(iter(info['anchors'].values()))))
        elif coordinate == 'payload':
            obj = info['objects'][info['route'][0]]
            pos = next(p for p in obj['target'] if p != obj['dest'])
        elif coordinate == 'stationary':
            pos = next(iter(next(o['points'] for i, o in enumerate(info['objects']) if i not in info['route'])))
        else:
            pos = next(p for i in info['route'] for p in info['objects'][i]['points'] if out[p] == info['bg'])
        out[pos] = (int(out[pos])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError('corruption accepted: '+coordinate)
    return dict(seeds=42, puzzles=len(fingerprints), shapes=len(shapes),
                structures=len(structures), body_sizes=sorted(sizes),
                route_lengths=sorted(lengths), guide_counts=sorted(guide_counts),
                background_colors=sorted(palettes))
