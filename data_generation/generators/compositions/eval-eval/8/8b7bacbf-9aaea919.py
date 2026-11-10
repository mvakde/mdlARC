"""Communicating reservoirs: routed token counts raise a common waterline.

Only Python's standard library and NumPy are used. Generation constructs labels
from explicit scene objects; solving reconstructs all roles from input pixels.
"""
from collections import deque
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
AUDIT_SEEDS = (0, 671743069, 928914058, 604143224, 558519887, 847018922,
               257702530, 123020054, 226316636, 88273372, 894648662,
               275822782, 868295391, 683951075, 46141514, 323195980,
               169541475, 565856296, 377705008, 140465324, 755380779,
               744319220, 880194218, 943505731, 936600796, 344650053,
               582555172, 876038723, 179103863, 974577247, 783118115,
               611458691, 319027194, 36114150, 248701515, 324858878,
               534111695, 571204122, 696766454, 280468856, 69497757,
               103302762)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape),
             'grid dimensions must be 1..30')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'colors must be 0..9')
    return a.astype(np.int64, copy=True)


def _components(cells, steps=D4):
    left = set(cells)
    groups = []
    while left:
        p = min(left)
        left.remove(p)
        group = {p}
        queue = [p]
        for r, c in queue:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    queue.append(q)
        groups.append(group)
    return groups


def _neighbors(cells, steps=D8):
    cells = set(cells)
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - cells


def _translate(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _key(a):
    return (a.shape, a.tobytes())


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _token_descriptor(cells, color):
    r0, r1, c0, c1 = _bounds(cells)
    mask = tuple(sorted((r - r0, c - c0) for r, c in cells))
    return dict(cells=cells, color=color, r0=r0, r1=r1, c0=c0, c1=c1,
                height=r1-r0+1, width=c1-c0+1, mask=mask)


def _parse(input_grid):
    """Discover tanks by enclosure, hoppers by bar geometry, and pipes by contact."""
    a = _grid(input_grid)
    h, w = a.shape
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    bg = int(border[0])
    _require(bool(np.all(border == bg)), 'the outer border must be background')
    canvas = {(r, c) for r in range(h) for c in range(w)}
    candidates = []
    # Token silhouettes and bottom water have no holes. Only the pipe/wall
    # color encloses regions, even when those regions contain old paint.
    for color in sorted(set(int(v) for v in np.unique(a)) - {bg}):
        regions = _components({p for p in canvas if a[p] != color})
        holes = [g for g in regions if not any(r in (0, h-1) or c in (0, w-1)
                                              for r, c in g)]
        if holes:
            candidates.append((color, holes))
    _require(len(candidates) == 1, 'a unique wall color must enclose the tanks')
    wall, holes = candidates[0]
    wall_groups = _components({p for p in canvas if a[p] == wall}, D8)
    owner = {p: i for i, group in enumerate(wall_groups) for p in group}
    networks = [dict(tanks=[], sources=[]) for _ in wall_groups]
    tank_cells = set()
    tanks = []
    for cells in holes:
        values = {int(a[p]) for p in cells} - {bg}
        _require(len(values) <= 1, 'a tank may contain at most one paint color')
        paint = next(iter(values), None)
        r0, r1, c0, c1 = _bounds(cells)
        for r in range(r0, r1+1):
            columns = {c for rr, c in cells if rr == r}
            _require(bool(columns) and len(columns) == max(columns)-min(columns)+1,
                     'tank interiors must be contiguous row intervals')
        shell = _neighbors(cells, D4)
        _require(all(p in canvas and a[p] == wall for p in shell),
                 'tanks must have complete closed walls')
        ids = {owner[p] for p in shell}
        _require(len(ids) == 1, 'each tank must have one pipe owner')
        i = ids.pop()
        level = min((r for r, c in cells if a[r, c] == paint), default=None)
        if paint is not None:
            _require(r0 < level <= r1, 'wet tanks need old water and empty headroom')
            _require(all(a[r, c] == (paint if r >= level else bg) for r, c in cells),
                     'old paint must be exactly the cells below one horizontal level')
        tank = dict(cells=cells, shell=shell, paint=paint, level=level,
                    top=r0, bottom=r1, owner=i, width=c1-c0+1, dry=paint is None)
        tanks.append(tank)
        networks[i]['tanks'].append(tank)
        tank_cells.update(cells)
    external = []
    for color in sorted(set(int(v) for v in np.unique(a)) - {bg, wall}):
        for cells in _components({p for p in canvas - tank_cells if a[p] == color}):
            d = _token_descriptor(cells, color)
            _require(1 <= d['height'] <= 3 and 1 <= d['width'] <= 3,
                     'external paint objects must be compact tokens')
            # A hole in a token would have competed with wall detection above.
            external.append(d)
    lookup = {(t['color'], t['r0'], t['c0'], t['height'], t['width'], t['mask']): j
              for j, t in enumerate(external)}
    used = set()
    sources = []
    for j, t in enumerate(external):
        y = t['r1'] + 2  # one blank row between lowest token and control bar
        bar = {(y, c) for c in range(t['c0']-1, t['c1']+2)}
        if not (bar <= canvas and all(a[p] == wall for p in bar)):
            continue
        gap = {(y-1, c) for c in range(t['c0']-1, t['c1']+2)}
        _require(all(a[p] == bg for p in gap), 'the hopper control gap must be blank')
        tokens, indices = [], []
        row = t['r0']
        while True:
            k = (t['color'], row, t['c0'], t['height'], t['width'], t['mask'])
            if k not in lookup:
                break
            index = lookup[k]
            _require(index not in used, 'a token cannot belong to two hoppers')
            tokens.append(external[index])
            indices.append(index)
            row -= t['height'] + 1
        used.update(indices)
        ids = {owner[p] for p in bar}
        _require(len(ids) == 1, 'a hopper must attach to exactly one pipe network')
        i = ids.pop()
        source = dict(tokens=tokens, count=len(tokens), paint=t['color'],
                      bar=bar, owner=i, mask=t['mask'], height=t['height'],
                      width=t['width'])
        sources.append(source)
        networks[i]['sources'].append(source)
    _require(bool(sources), 'at least one routed hopper is required')
    for n in networks:
        _require(bool(n['tanks']), 'all pipe networks must contain a tank')
        paints = {t['paint'] for t in n['tanks'] if not t['dry']}
        paints.update(s['paint'] for s in n['sources'])
        levels = {t['level'] for t in n['tanks'] if not t['dry']}
        _require(len(paints) <= 1 and len(levels) <= 1,
                 'a network must share one paint and initial waterline')
        n['paint'] = next(iter(paints), None)
        n['level'] = next(iter(levels), None)
        n['total'] = sum(s['count'] for s in n['sources'])
        _require(all(s['paint'] == n['paint'] for s in n['sources']),
                 'hoppers and their tanks must share the local paint')
        if n['sources']:
            _require(n['level'] is not None, 'a supplied network needs a visible initial waterline')
            for t in n['tanks']:
                _require(n['level']-n['total'] > t['top'],
                         'every tank needs at least one empty row after the rise')
                _require(n['level']-n['total'] <= t['bottom'],
                         'the new surface must intersect every supplied tank')
                if t['dry']:
                    _require(t['bottom'] < n['level'], 'dry tanks must lie above the old surface')
                    t['level'], t['paint'] = n['level'], n['paint']
    return dict(grid=a, bg=bg, wall=wall, networks=networks, tanks=tanks,
                sources=sources, unmarked=[t for j, t in enumerate(external) if j not in used],
                wall_groups=wall_groups)


def _render_solution(info):
    result = info['grid'].copy()
    for n in info['networks']:
        if not n['sources']:
            continue
        new_level = n['level'] - n['total']
        for t in n['tanks']:
            for r, c in t['cells']:
                if r >= new_level:
                    result[r, c] = n['paint']
        for s in n['sources']:
            for token in s['tokens']:
                for p in token['cells']:
                    result[p] = info['wall']
            for p in s['bar']:
                result[p] = info['bg']
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve only from the supplied grid; no training, seed, or cached state."""
    return _render_solution(_parse(input_grid))


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi+1))


def _pick(rng, values):
    return values[_ri(rng, 0, len(values)-1)]


def _mask(rng, count, informative=False):
    height = _ri(rng, 1, 2 if count >= 3 else 3)
    width = _ri(rng, 1, 3)
    if informative and height * width == 1:
        width = 2
    kind = _pick(rng, ('block', 'block', 'L', 'steps'))
    cells = {(r, 0) for r in range(height)} | {(height-1, c) for c in range(width)}
    if kind == 'block':
        cells = {(r, c) for r in range(height) for c in range(width)}
    elif kind == 'steps':
        for r in range(height):
            cells.update((r, c) for c in range(_ri(rng, 1, width)))
    if rng.random() < .5:
        cells = {(r, width-1-c) for r, c in cells}
    if rng.random() < .5:
        cells = {(height-1-r, c) for r, c in cells}
    return cells, height, width


def _tank_shape(rng, height, width):
    """Procedural overlapping row intervals with a minimal closed shell."""
    kind = _pick(rng, ('box', 'waist', 'steps', 'slant'))
    left, cells = 0, set()
    previous = None
    for r in range(height):
        row_width = width
        if kind == 'waist' and 0 < r < height-1:
            row_width = max(1, width-1)
        elif kind == 'steps':
            row_width = max(1, width + _ri(rng, -1, 1))
            if r:
                left += _ri(rng, -1, 1)
        elif kind == 'slant':
            left = r // 3
        if previous is not None:
            # Adjacent row intervals overlap, so water and cavities remain 4-connected.
            left = min(max(left, min(previous)-row_width+1), max(previous))
        current = set(range(left, left+row_width))
        cells.update((r, c) for c in current)
        previous = current
    shell = _neighbors(cells, D4)
    if kind == 'box' and rng.random() < .5:
        shell = _neighbors(cells, D8)
    r0, _, c0, _ = _bounds(cells | shell)
    return _translate(cells, -r0, -c0), _translate(shell, -r0, -c0)


def _path(rng, starts, goals, allowed):
    queue = deque(sorted(starts))
    parents = {p: None for p in starts}
    directions = [D8[i] for i in rng.permutation(8)]
    last = None
    while queue:
        r, c = queue.popleft()
        if (r, c) in goals:
            last = (r, c)
            break
        for dr, dc in directions:
            q = (r+dr, c+dc)
            if q in allowed and q not in parents:
                parents[q] = (r, c)
                queue.append(q)
    if last is None:
        return None
    path = set()
    while last is not None:
        path.add(last)
        last = parents[last]
    return path


def _scene(rng, convention, spec, spacious=False):
    """Free placement and bounded routing; returns None for infeasible geometry."""
    bg, wall = convention['bg'], convention['wall']
    if spacious:
        h = w = 30
    elif (len(spec['counts']) == 1 and spec['tank_counts'] == [1]
          and not spec['decoys'] and not spec['unmarked']):
        lower = max(12, sum(spec['counts'][0])+10, max(spec['counts'][0])*3+4)
        h, w = _ri(rng, lower, min(30, lower+7)), _ri(rng, 10, 22)
    else:
        lower = 23 if spec['lesson'] == 'level' else 21
        if spec['test']:
            lower = 25
        h, w = _ri(rng, lower, 30), _ri(rng, 20, 30)
    canvas = {(r, c) for r in range(1, h-1) for c in range(1, w-1)}
    a = np.full((h, w), bg, dtype=np.int64)
    objects, tanks, hoppers, unmarked = [], [], [], []
    occupied = set()
    network_count = len(spec['counts'])
    paints = [_pick(rng, convention['paints']) for _ in range(network_count)]
    if spec.get('paint') is not None:
        paints = [spec['paint']] * network_count
    if spec['lesson'] == 'ownership' or rng.random() < .25:
        paints = [paints[0]] * network_count
    levels = []
    for i, counts in enumerate(spec['counts']):
        total = sum(counts)
        level = _ri(rng, total+4, h-6)
        levels.append(level)
        depths = [_ri(rng, 1, 3) for _ in range(spec['tank_counts'][i])]
        dry_last = (len(depths) > 1 and total >= 2 and
                    (spec['lesson'] == 'level' or rng.random() < .45))
        for index, depth in enumerate(depths):
            extra = _ri(rng, 1, 3)
            dry = dry_last and index == len(depths)-1
            gap = 2 if spec['lesson'] == 'level' and dry else _ri(rng, 1, max(1, total-1))
            inner_height = total + extra - gap + 1 if dry else total + extra + depth
            inner, shell = _tank_shape(rng, inner_height, _ri(rng, 1, 4))
            local_top = min(r for r, c in inner)
            rr = level-total-extra-local_top
            allcells = inner | shell
            _, _, lc0, lc1 = _bounds(allcells)
            if lc1-lc0+1 > w-2:
                return None
            placed = False
            for _ in range(80):
                cc = _ri(rng, 1-lc0, w-2-lc1)
                cells = _translate(inner, rr, cc)
                edge = _translate(shell, rr, cc)
                footprint = cells | edge
                if not footprint <= canvas or (footprint | _neighbors(footprint)) & occupied:
                    continue
                t = dict(cells=cells, shell=edge, owner=i, paint=paints[i], level=level,
                         footprint=footprint, protected=cells)
                tanks.append(t)
                objects.append(t)
                occupied.update(footprint)
                placed = True
                break
            if not placed:
                return None
    # A detached partly filled tank uses the same paint as a productive network.
    for _ in range(spec['decoys']):
        placed = False
        inner, shell = _tank_shape(rng, _ri(rng, 4, 7), _ri(rng, 1, 3))
        _, r1, _, c1 = _bounds(inner | shell)
        for _ in range(100):
            rr, cc = _ri(rng, 1, h-2-r1), _ri(rng, 1, w-2-c1)
            cells, edge = _translate(inner, rr, cc), _translate(shell, rr, cc)
            footprint = cells | edge
            if (footprint | _neighbors(footprint)) & occupied:
                continue
            level = max(r for r, c in cells) - _ri(rng, 0, 1)
            t = dict(cells=cells, shell=edge, owner=None, paint=paints[0], level=level,
                     footprint=footprint, protected=cells)
            tanks.append(t)
            objects.append(t)
            occupied.update(footprint)
            placed = True
            break
        if not placed:
            return None
    for i, counts in enumerate(spec['counts']):
        for count in counts:
            mask, th, tw = _mask(rng, count, informative=spec['lesson'] == 'level')
            stack_height = th + (count-1)*(th+1)
            placed = False
            for _ in range(100):
                rr, cc = _ri(rng, 1, h-stack_height-4), _ri(rng, 2, w-tw-2)
                y = rr+stack_height+1
                bar = {(y, c) for c in range(cc-1, cc+tw+1)}
                tokens = [_translate(mask, rr+k*(th+1), cc) for k in range(count)]
                protected = {(r, c) for r in range(rr, y) for c in range(cc-1, cc+tw+1)}
                footprint = protected | bar
                if (footprint | _neighbors(footprint)) & occupied:
                    continue
                s = dict(tokens=tokens, bar=bar, shell=bar, protected=protected,
                         footprint=footprint, owner=i, paint=paints[i])
                hoppers.append(s)
                objects.append(s)
                occupied.update(footprint)
                placed = True
                break
            if not placed:
                return None
    for _ in range(spec['unmarked']):
        count = _ri(rng, 1, 3)
        mask, th, tw = _mask(rng, count)
        stack_height = th+(count-1)*(th+1)
        placed = False
        for _ in range(100):
            rr, cc = _ri(rng, 1, h-stack_height-4), _ri(rng, 2, w-tw-2)
            footprint = {(r, c) for r in range(rr, rr+stack_height+2)
                         for c in range(cc-1, cc+tw+1)}
            if (footprint | _neighbors(footprint)) & occupied:
                continue
            s = dict(tokens=[_translate(mask, rr+k*(th+1), cc) for k in range(count)],
                     protected=footprint, footprint=footprint, shell=set(), owner=None,
                     paint=paints[0])
            unmarked.append(s)
            objects.append(s)
            occupied.update(footprint)
            placed = True
            break
        if not placed:
            return None
    routes = {}
    for i in rng.permutation(network_count):
        i = int(i)
        own = [o for o in objects if o['owner'] == i]
        foreign = set().union(*(o['footprint'] for o in objects if o['owner'] != i))
        foreign.update(set().union(*routes.values()))
        # Previously routed networks also remain isolated.
        blocked = foreign | _neighbors(foreign)
        blocked.update(set().union(*(o['protected'] for o in own)))
        allowed = canvas - blocked
        terminals = [o['shell'] for o in own]
        order = rng.permutation(len(terminals)).tolist()
        tree = set(terminals[order[0]])
        wire = set()
        for k in order[1:]:
            target = terminals[k]
            path = _path(rng, tree, target, allowed | tree | target)
            if path is None:
                return None
            wire.update(path)
            tree.update(path)
            tree.update(target)
        routes[i] = wire
    for t in tanks:
        for p in t['shell']:
            a[p] = wall
        for r, c in t['cells']:
            if r >= t['level']:
                a[r, c] = t['paint']
    for s in hoppers:
        for p in s['bar']:
            a[p] = wall
    for path in routes.values():
        for p in path:
            a[p] = wall
    for s in hoppers + unmarked:
        for token in s['tokens']:
            for p in token:
                a[p] = s['paint']
    # Construct the answer from intended scene objects, before pixel parsing.
    b = a.copy()
    for t in tanks:
        if t['owner'] is not None:
            new_level = t['level'] - sum(spec['counts'][t['owner']])
            for r, c in t['cells']:
                if r >= new_level:
                    b[r, c] = t['paint']
    for s in hoppers:
        for token in s['tokens']:
            for p in token:
                b[p] = wall
        for p in s['bar']:
            b[p] = bg
    try:
        info = _parse(a)
    except ValueError:
        return None
    if not np.array_equal(_render_solution(info), b):
        return None
    # Parsing must recover intended interiors, all sources, and no fake holes.
    if {frozenset(t['cells']) for t in info['tanks']} != {frozenset(t['cells']) for t in tanks}:
        return None
    if len(info['sources']) != len(hoppers):
        return None
    if rng.random() < .5:
        a, b = np.fliplr(a), np.fliplr(b)
    return {'input': a.copy(), 'output': b.copy()}


def _fallback(convention, lesson, test, variant, forced_paint=None):
    """Guaranteed sparse valid geometry if free placement exhausts its budget.

    This is a fallback for the same rule and evidence, never a different rule.
    Its paint/reflection/depth variants also provide a finite uniqueness reserve.
    """
    bg, wall = convention['bg'], convention['wall']
    paint = (convention['paints'][variant % len(convention['paints'])]
             if forced_paint is None else forced_paint)
    a = np.full((30, 30), bg, dtype=np.int64)
    tanks, sources = [], []

    def tank(r0, r1, c0, c1, level, total):
        cells = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)}
        shell = _neighbors(cells, D8)
        for p in shell:
            a[p] = wall
        for r, c in cells:
            if r >= level:
                a[r, c] = paint
        tanks.append((cells, level, total))

    def stack(ys, x):
        tokens = [{(y, x), (y, x+1)} for y in ys]
        bar = {(ys[-1]+2, c) for c in range(x-1, x+3)}
        for t in tokens:
            for p in t:
                a[p] = paint
        for p in bar:
            a[p] = wall
        sources.append((tokens, bar))

    def wire(cells):
        for p in cells:
            a[p] = wall

    deeper = (variant // 16) % 2
    if lesson == 'ownership' and not test:
        tank(5, 10+deeper, 2, 4, 9, 1)
        tank(19, 26, 17, 19, 24, 2)
        stack([3], 10)
        stack([19, 21], 25)
        wire({(r, 13) for r in range(6, 15)} |
             {(14, c) for c in range(5, 14)} |
             {(r, 5) for r in range(11+deeper, 15)})
        wire({(r, 23) for r in range(24, 29)} |
             {(28, c) for c in range(20, 24)} | {(27, 20)})
    else:
        total = 5 + (variant // 32) % 2 if test else 3
        level = 8 + total
        tank(6, level+1+deeper, 2, 3, level, total)
        second_bottom = level-2 if not test else level+2+deeper
        tank(6, second_bottom, 10, 11, level, total)
        tank(23, 26, 2, 4, 25, 0)
        stack([3] if not test else ([2, 4] if total == 5 else [1, 3, 5]), 18)
        stack([9, 11] if not test else [9, 11, 13], 18)
        first_y = 5 if not test else (6 if total == 5 else 7)
        wire({(r, 21) for r in range(first_y+1, 21)} |
             {(20, c) for c in range(4, 22)} |
             {(r, 4) for r in range(level+2+deeper, 21)} |
             {(r, 12) for r in range(second_bottom+1, 21)})
        for y in (23, 25):
            a[y, 12:14] = paint  # an unmarked stack of the same paint
    b = a.copy()
    for cells, level, total in tanks:
        if total:
            for r, c in cells:
                if r >= level-total:
                    b[r, c] = paint
    for tokens, bar in sources:
        for t in tokens:
            for p in t:
                b[p] = wall
        for p in bar:
            b[p] = bg
    if (variant // 8) % 2:
        a, b = np.fliplr(a), np.fliplr(b)
    _require(np.array_equal(solve(a), b), 'sparse fallback construction mismatch')
    return {'input': a.copy(), 'output': b.copy()}


def _spec(rng, lesson=None, test=False):
    if lesson == 'level':
        counts = list(_pick(rng, ((1, 2), (1, 3), (1, 1, 2))))
        rng.shuffle(counts)
        return dict(lesson=lesson, test=False, counts=[counts], tank_counts=[2],
                    decoys=1, unmarked=1)
    if lesson == 'ownership':
        values = list(_pick(rng, ((1, 2), (1, 3), (2, 3))))
        rng.shuffle(values)
        return dict(lesson=lesson, test=False, counts=[[values[0]], [values[1]]],
                    tank_counts=[1, 1], decoys=0, unmarked=_ri(rng, 0, 1))
    n = _pick(rng, (1, 1, 2))
    counts, tanks = [], []
    for _ in range(n):
        total = _ri(rng, 5, 6) if test else _ri(rng, 1, 4)
        number = _ri(rng, 1, min(3, total))
        cuts = sorted(rng.choice(np.arange(1, total), number-1, replace=False).tolist())
        counts.append(np.diff([0]+cuts+[total]).astype(int).tolist())
        tanks.append(_ri(rng, 1, 2))
    return dict(lesson=None, test=test, counts=counts, tank_counts=tanks,
                decoys=_ri(rng, 0, 1), unmarked=_ri(rng, 0, 1))


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10),
                                ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f'{name} must be an integer from {lo} through {hi}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = rng.permutation(10).astype(int).tolist()
    convention = dict(bg=colors[0], wall=colors[1], paints=colors[2:])
    lessons = ['level', 'ownership']
    rng.shuffle(lessons)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            lesson = lessons[index] if split == 'train' and index < 2 else None
            pair = None
            for attempt in range(180):
                spec = _spec(rng, lesson, test=split == 'test')
                if lesson is not None:
                    spec['paint'] = convention['paints'][index]
                candidate = _scene(rng, convention, spec, spacious=attempt >= 25)
                if candidate is not None and _key(candidate['input']) not in seen:
                    pair = candidate
                    break
            if pair is None:
                for variant in range(64):
                    forced = convention['paints'][index] if lesson is not None else None
                    candidate = _fallback(convention, lesson, split == 'test', variant, forced)
                    if _key(candidate['input']) not in seen:
                        pair = candidate
                        break
                _require(pair is not None, 'finite uniqueness reserve exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    active = [n for n in info['networks'] if n['sources']]
    return dict(networks=len(active), tanks=len(info['tanks']),
                total=tuple(sorted(n['total'] for n in active)),
                sources=tuple(sorted(len(n['sources']) for n in active)),
                counts=tuple(sorted(s['count'] for s in info['sources'])),
                depths=tuple(sorted(max(0, t['bottom']-t['level']+1)
                                    if t['level'] is not None else 0 for t in info['tanks'])),
                masks=tuple(sorted((s['height'], s['width'], s['mask']) for s in info['sources'])),
                cavity_shapes=tuple(sorted(tuple(sorted(_translate(t['cells'], -_bounds(t['cells'])[0],
                                                               -_bounds(t['cells'])[2])))
                                          for t in info['tanks'])),
                decoys=sum(not n['sources'] for n in info['networks']),
                dry_tanks=sum(t['dry'] for t in info['tanks']),
                unmarked=len(info['unmarked']),
                same_paint_different_total=any(x['paint'] == y['paint'] and x['total'] != y['total']
                                               for i, x in enumerate(active) for y in active[:i]),
                unequal_depths=any(len({t['bottom'] for t in n['tanks']}) > 1 for n in active),
                unequal_sources=any(len({s['count'] for s in n['sources']}) > 1 for n in active))


def _alternative_output(info, mode):
    """Simple competing rules used only to audit demonstration evidence."""
    result = _render_solution(info)
    for n in info['networks']:
        for t in n['tanks']:
            for p in t['cells']:
                result[p] = info['grid'][p]
            if mode == 'global_color':
                amount = sum(s['count'] for s in info['sources'] if s['paint'] == t['paint'])
                if t['level'] is None or not amount:
                    continue
                threshold = t['level']-amount
            elif not n['sources']:
                continue
            elif mode == 'fill_all':
                threshold = t['top']
            elif mode == 'count_bars':
                threshold = n['level']-len(n['sources'])
            elif mode == 'tallest':
                threshold = n['level']-max(s['count'] for s in n['sources'])
            elif mode == 'pixel_volume':
                threshold = n['level']-sum(len(q['cells']) for s in n['sources'] for q in s['tokens'])
            elif mode == 'double_depth':
                threshold = t['bottom']-2*max(0, t['bottom']-t['level']+1)+1
            elif mode == 'ignore_existing':
                threshold = t['bottom']-n['total']+1
            elif mode == 'separate_bottoms':
                depth = max(0, t['bottom']-t['level']+1)
                threshold = t['bottom']-(depth+n['total'])+1
            else:
                raise ValueError('unknown evidence alternative')
            for r, c in t['cells']:
                if r >= threshold:
                    result[r, c] = t['paint']
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, conventions, training_infos = set(), set(), []
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'dimensions must be preserved')
            _require(_key(a) not in seen, 'inputs must be distinct')
            seen.add(_key(a))
            info = _parse(a)
            conventions.add((info['bg'], info['wall']))
            _require(np.array_equal(_render_solution(info), b), 'wrong routed waterline or source processing')
            _require(not np.array_equal(a, b), 'answers must change')
            # Wall preservation excludes only explicitly recognized control bars.
            bars = set().union(*(s['bar'] for s in info['sources']))
            _require(all(b[p] == info['wall'] for group in info['wall_groups'] for p in group - bars),
                     'pipes and tank walls must be preserved')
            for n in info['networks']:
                for t in n['tanks']:
                    if not n['sources']:
                        _require(all(a[p] == b[p] for p in t['cells']), 'unsupplied tank changed')
                    else:
                        added = [p for p in t['cells'] if a[p] != b[p]]
                        _require(bool(added), 'every supplied tank must gain paint')
                        _require({r for r, c in added} == set(range(n['level']-n['total'],
                                                                  min(n['level'], t['bottom']+1))),
                                 'rise must equal summed object counts, not area')
                for s in n['sources']:
                    _require(all(b[p] == info['bg'] for p in s['bar']), 'control bars must vanish')
                    _require(all(b[p] == info['wall'] for t in s['tokens'] for p in t['cells']),
                             'contributing tokens must be highlighted without changing shape')
            _require(all(a[p] == b[p] for t in info['unmarked'] for p in t['cells']),
                     'unmarked tokens changed')
            features[split].append(_features(info))
            if split == 'train':
                training_infos.append(info)
    _require(len(conventions) == 1, 'background and pipe roles must be consistent in a puzzle')
    _require(len({n['paint'] for info in training_infos for n in info['networks']
                  if n['sources']}) >= 2, 'training must demonstrate copying two local paint colors')
    train = features['train']
    _require(any(f['unequal_sources'] and f['unequal_depths'] for f in train),
             'training must show unequal aggregation and communicating depths together')
    _require(any(f['dry_tanks'] for f in train),
             'training must teach transferring the shared surface into a dry tank')
    _require(any(f['same_paint_different_total'] for f in train),
             'training must distinguish route ownership from global paint grouping')
    _require(any(f['decoys'] and f['unmarked'] for f in train),
             'training must teach ignoring disconnected tanks and unmarked tokens')
    training_max = max(v for f in train for v in f['total'])
    _require(all(max(f['total']) > training_max for f in features['test']),
             'tests must extrapolate the demonstrated level increments')
    for mode in ('global_color', 'fill_all', 'count_bars', 'tallest', 'pixel_volume',
                 'double_depth', 'ignore_existing', 'separate_bottoms'):
        _require(any(not np.array_equal(_alternative_output(info, mode), pair['output'])
                     for info, pair in zip(training_infos, puzzle['train'])),
                 f'training fails to distinguish the {mode} shortcut')
    return None


def self_test():
    """Bounded 42-seed semantic/diversity audit plus corruption and count checks."""
    shapes, masks, cavities, totals, counts, fingerprints, roles = set(), set(), set(), set(), set(), set(), set()
    for seed in AUDIT_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(tuple(_key(q['input']) for s in ('train', 'test') for q in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _parse(pair['input'])
                f = _features(info)
                shapes.add(pair['input'].shape)
                roles.add((info['bg'], info['wall']))
                masks.update(f['masks'])
                cavities.update(f['cavity_shapes'])
                totals.update(f['total'])
                counts.update(f['counts'])
                original = pair['input'].copy()
                _require(np.array_equal(solve(original), pair['output']), 'input-only solve mismatch')
                _require(np.array_equal(original, pair['input']), 'solve mutated its input')
    _require(len(fingerprints) == 42, 'seed diversity collapsed')
    _require(len(shapes) >= 20 and len(masks) >= 12 and len(cavities) >= 50,
             'insufficient structural variation across 42 seeds')
    _require({1, 2, 3, 4, 5, 6} <= totals and {1, 2, 3} <= counts and len(roles) >= 20,
             'insufficient arithmetic or role variation')
    p = generate_puzzle(0, 2, 1)
    again = generate_puzzle(0, 2, 1)
    _require(all(np.array_equal(x[r], y[r]) for s in p for x, y in zip(p[s], again[s])
                 for r in ('input', 'output')), 'seed is not deterministic')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            q = generate_puzzle(8100+10*nt+ne, nt, ne)
            _require(len(q['train']) == nt and len(q['test']) == ne, 'wrong explicit counts')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # These wrong answers exercise both parent contributions and the connector.
    for defect in ('water', 'source', 'bar', 'decoy'):
        bad = {s: [{r: a.copy() for r, a in q.items()} for q in p[s]] for s in p}
        changed = False
        for pair in bad['train']:
            info = _parse(pair['input'])
            if defect == 'water':
                t = next(t for n in info['networks'] if n['sources'] for t in n['tanks'])
                point = next(p for p in sorted(t['cells']) if pair['input'][p] != pair['output'][p])
                pair['output'][point] = info['bg']
            elif defect == 'source':
                s = info['sources'][0]
                point = min(s['tokens'][0]['cells'])
                pair['output'][point] = s['paint']
            elif defect == 'bar':
                point = min(info['sources'][0]['bar'])
                pair['output'][point] = info['wall']
            else:
                inactive = [n for n in info['networks'] if not n['sources']]
                if not inactive:
                    continue
                t = inactive[0]['tanks'][0]
                point = min(t['cells'])
                pair['output'][point] = t['paint']
            changed = True
            break
        _require(changed, 'mutation witness absent')
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'wrong {defect} output accepted')
    return None
