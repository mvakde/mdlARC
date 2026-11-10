"""Cooperative chambers: frozen, pairwise color queries contract a room graph.

Only standard-library modules and NumPy are used.  Generation constructs clean
walls; solve reconstructs walls from input pixels and counts graph components.
The validator additionally opens actual wall pixels and flood-fills the result.
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _view(a, corner):
    if corner & 2:
        a = a[::-1, :]
    if corner & 1:
        a = a[:, ::-1]
    return a


def _selector(a):
    found = []
    if min(a.shape) < 5:
        raise ValueError("No selector")
    for corner in range(4):
        v = _view(a, corner)
        scar = int(v[3, 3])
        if not (np.all(v[3, :4] == scar) and np.all(v[:4, 3] == scar)):
            continue
        colors, counts = np.unique(v[:3, :3], return_counts=True)
        bg = colors[counts == 6]
        if len(bg) != 1 or len(colors) != 4:
            continue
        bg = int(bg[0])
        q = frozenset(int(c) for c in colors if c != bg)
        if scar == bg or scar in q:
            continue
        # A one-cell clear separator makes the legend distinct from a building.
        if not (np.all(v[4, :5] == bg) and np.all(v[:5, 4] == bg)):
            continue
        found.append((corner, bg, scar, q))
    if len(found) != 1:
        raise ValueError("Selector must be unique")
    return found[0]


def _regions(wall):
    """Return bounded free components; the padded exterior is always excluded."""
    free = np.pad(~wall, 1, constant_values=True)
    seen = np.zeros_like(free)
    result = []
    h, w = free.shape
    for y, x in np.argwhere(free):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells, outside = [(y, x)], [], False
        while stack:
            r, c = stack.pop()
            cells.append((r - 1, c - 1))
            outside |= r == 0 or c == 0 or r == h - 1 or c == w - 1
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and free[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        if not outside:
            result.append(np.asarray(cells, dtype=int))
    result.sort(key=lambda cells: tuple(cells.min(axis=0)))
    return result


def _boundary(shape, boxes):
    wall = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        wall[t, l:r+1] = wall[b, l:r+1] = True
        wall[t:b+1, l] = wall[t:b+1, r] = True
    return wall


def _repair(a, color, scar):
    visible = a == color
    mask = visible.copy()
    # A scar is a missing wall pixel only with opposite same-color support.
    mask[1:-1, :] |= ((a[1:-1, :] == scar) & visible[:-2, :] & visible[2:, :])
    mask[:, 1:-1] |= ((a[:, 1:-1] == scar) & visible[:, :-2] & visible[:, 2:])
    return mask


def _network(a, color, bg, scar):
    points = np.argwhere(a == color)
    if len(points) < 8:
        return None
    t, l = map(int, points.min(axis=0))
    b, r = map(int, points.max(axis=0))
    if min(b-t, r-l) < 2:
        return None
    mask = _repair(a, color, scar)
    if not (mask[t, l:r+1].all() and mask[b, l:r+1].all()
            and mask[t:b+1, l].all() and mask[t:b+1, r].all()):
        return None
    local = mask[t:b+1, l:r+1]
    rooms = _regions(local)
    if not rooms:
        return None
    boxes = []
    for cells in rooms:
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        if len(cells) != int(np.prod(hi-lo+1)):
            return None
        boxes.append((int(lo[0])-1, int(lo[1])-1, int(hi[0])+1, int(hi[1])+1))
    if not np.array_equal(local, _boundary(local.shape, boxes)):
        return None
    return dict(color=int(color), box=(t, l, b, r), wall=local,
                rooms=rooms, grid=a[t:b+1, l:r+1])


def _room_data(network, bg, scar):
    wall, rooms, a = network['wall'], network['rooms'], network['grid']
    ids = np.full(wall.shape, -1, dtype=int)
    labels, counts = [], []
    for i, cells in enumerate(rooms):
        ids[cells[:, 0], cells[:, 1]] = i
        values, numbers = np.unique(a[cells[:, 0], cells[:, 1]], return_counts=True)
        counts.append({int(c): int(n) for c, n in zip(values, numbers) if c not in (bg, scar)})
        labels.append(frozenset(counts[-1]))
    gates = {}
    h, w = wall.shape
    for y, x in np.argwhere(wall):
        y, x = int(y), int(x)
        opposite = []
        if 0 < y < h-1:
            opposite.append((ids[y-1, x], ids[y+1, x]))
        if 0 < x < w-1:
            opposite.append((ids[y, x-1], ids[y, x+1]))
        for i, j in opposite:
            if i >= 0 and j >= 0 and i != j:
                gates.setdefault(tuple(sorted((int(i), int(j)))), []).append((y, x))
    network.update(ids=ids, labels=labels, counts=counts, gates=gates)


def _groups(n, edges):
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in edges:
        parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def _analyze(value):
    a = _grid(value)
    corner, bg, scar, query = _selector(a)
    work = a.copy()
    _view(work, corner)[:4, :4] = bg
    networks = []
    for color in np.unique(work):
        if color in query or color in (bg, scar):
            continue
        n = _network(work, int(color), bg, scar)
        if n is not None:
            _room_data(n, bg, scar)
            n['eligible'] = {e for e in n['gates'] if query <= n['labels'][e[0]] | n['labels'][e[1]]}
            n['count'] = _groups(len(n['rooms']), n['eligible'])
            networks.append(n)
    # Labels may themselves form geometric shapes. An interior colored loop is
    # still a label pattern: only maximal, mutually separated outer frames are
    # buildings. This avoids making mark arrangement a hidden restriction.
    networks = [n for n in networks if not any(
        other is not n and other['box'][0] <= n['box'][0] and other['box'][1] <= n['box'][1]
        and other['box'][2] >= n['box'][2] and other['box'][3] >= n['box'][3]
        for other in networks)]
    if not networks:
        raise ValueError("No closed building")
    # Every non-marker colored pixel must belong to a recovered building.
    covered = np.zeros(a.shape, dtype=bool)
    wall_colors = {n['color'] for n in networks}
    for i, n in enumerate(networks):
        t, l, b, r = n['box']
        for other in networks[:i]:
            tt, ll, bb, rr = other['box']
            if not (b+1 < tt or bb+1 < t or r+1 < ll or rr+1 < l):
                raise ValueError("Buildings need a background gap")
        covered[t:b+1, l:r+1] = True
        if any(s & wall_colors for s in n['labels']):
            raise ValueError("Wall colors cannot also be room labels")
    if np.any(~covered & ~np.isin(work, [bg, scar])):
        raise ValueError("Unexplained marks outside buildings")
    networks.sort(key=lambda n: (n['count'], n['box'][1], n['box'][0]))
    return dict(grid=a, corner=corner, bg=bg, scar=scar, query=query, networks=networks)


def _encode(networks, scar):
    ns = sorted(networks, key=lambda n: (n['count'], n['box'][1], n['box'][0]))
    out = np.full((len(ns), max(n['count'] for n in ns)), scar, dtype=np.int8)
    for i, n in enumerate(ns):
        out[i, :n['count']] = n['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    return _encode(info['networks'], info['scar'])


def _physical_count(n, eligible=None):
    wall = n['wall'].copy()
    for edge in n['eligible'] if eligible is None else eligible:
        for y, x in n['gates'][edge]:
            wall[y, x] = False
    return len(_regions(wall))


def _cuts(box):
    t, l, b, r = box
    answer = []
    for axis, start, end in ((0, t, b), (1, l, r)):
        for cut in range(start+2, end-1):
            children = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
            if all((bb-tt-1)*(rr-ll-1) >= 6 for tt, ll, bb, rr in children):
                answer.append((axis, children))
    return answer


def _partition(rng, h, w, count, partial=False):
    for _ in range(30):
        boxes, first_axis = [(0, 0, h-1, w-1)], None
        while len(boxes) < count:
            options = []
            for i, box in enumerate(boxes):
                for axis, children in _cuts(box):
                    if partial and len(boxes) == 2 and axis == first_axis:
                        continue
                    # Favor preserving room area when a large count is requested.
                    weight = min((b-t-1)*(r-l-1) for t, l, b, r in children)
                    options.append((i, axis, children, weight))
            if not options:
                break
            weights = np.asarray([o[3] for o in options], dtype=float)
            i, axis, children, _ = options[int(rng.choice(len(options), p=weights/weights.sum()))]
            if len(boxes) == 1:
                first_axis = axis
            boxes[i:i+1] = children
        if len(boxes) == count:
            wall = _boundary((h, w), boxes)
            if not partial or _t_junctions(wall) > 0:
                return wall
    return None


def _t_junctions(wall):
    degree = np.zeros(wall.shape, dtype=int)
    degree[1:] += wall[:-1]
    degree[:-1] += wall[1:]
    degree[:, 1:] += wall[:, :-1]
    degree[:, :-1] += wall[:, 1:]
    return int(np.count_nonzero(wall[1:-1, 1:-1] & (degree[1:-1, 1:-1] == 3)))


def _grid_wall(rng, h, w, rows, columns):
    if h-1 < 3*rows or w-1 < 4*columns:
        return None
    def positions(parts, span, minimum):
        gaps = np.full(parts, minimum)
        for _ in range(span - minimum*parts):
            gaps[int(rng.integers(parts))] += 1
        return [0] + list(map(int, np.cumsum(gaps)))
    ys, xs = positions(rows, h-1, 3), positions(columns, w-1, 4)
    return _boundary((h, w), [(ys[i], xs[j], ys[i+1], xs[j+1])
                             for i in range(rows) for j in range(columns)])


def _free_marks(rng, q, extras, area):
    mode = int(rng.integers(7))
    if mode == 0:
        return []
    labels = list(rng.permutation(q)[:int(rng.integers(4))])
    labels += list(rng.permutation(extras)[:int(rng.integers(3))])
    if mode == 1:
        labels = list(q)
    if not labels:
        return []
    target = int(rng.integers(len(labels), min(area-1, max(len(labels), area//2))+1))
    return [int(c) for c in labels] + [int(rng.choice(labels)) for _ in range(target-len(labels))]


def _local(rng, h, w, spec, color, bg, scar, q, extras):
    style, count = spec
    transpose = bool(rng.integers(2))
    hh, ww = (w, h) if transpose else (h, w)
    if style == 'cycle':
        columns = count//2
        wall = _grid_wall(rng, hh, ww, 2, columns)
    elif style == 'scope':
        wall = _grid_wall(rng, hh, ww, 2, 3)
    elif style == 'strip':
        wall = _grid_wall(rng, hh, ww, 1, count)
    else:
        wall = _partition(rng, hh, ww, count, count >= 3 and style in ('partial', 'sealed'))
    if wall is None:
        return None
    rooms = _regions(wall)
    if len(rooms) != count:
        return None
    a = np.full(wall.shape, bg, dtype=np.int8)
    a[wall] = color
    aa, bb, cc = q
    xx, yy = extras
    if style == 'cycle':
        columns = count//2
        marks = [[aa, bb, aa, xx], [cc]] + [[] for _ in range(columns-2)]
        marks += [[cc, yy, cc], [aa, bb, bb, yy]] + [[] for _ in range(columns-2)]
    elif style == 'scope':
        marks = [[aa], [bb, cc], [xx], [cc, xx, yy, cc, xx], [], [aa, bb, aa, bb]]
    elif style == 'sealed':
        # All rooms miss the same query identity, regardless of mark density.
        missing = int(rng.integers(3))
        labels = [q[i] for i in range(3) if i != missing] + [xx, yy]
        marks = [list(rng.permutation(labels)[:int(rng.integers(1, 5))]) for _ in rooms]
    else:
        marks = [_free_marks(rng, q, extras, len(cells)) for cells in rooms]
    for cells, labels in zip(rooms, marks):
        if len(labels) >= len(cells):
            return None
        sites = rng.choice(len(cells), len(labels), replace=False)
        for site, label in zip(sites, rng.permutation(labels)):
            y, x = cells[int(site)]
            a[y, x] = int(label)
    # Reflections preserve both witness graphs and the pairwise predicate.
    flip = int(rng.integers(4))
    a, wall = _view(a, flip).copy(), _view(wall, flip).copy()
    if transpose:
        a, wall = a.T.copy(), wall.T.copy()
    n = dict(color=color, wall=wall, rooms=_regions(wall), grid=a)
    _room_data(n, bg, scar)
    n['eligible'] = {e for e in n['gates'] if set(q) <= n['labels'][e[0]] | n['labels'][e[1]]}
    # Construction truth uses the physically opened mask, not the graph formula.
    n['count'] = _physical_count(n)
    return a, n


def _slots(rng, h, w, count):
    # The selector's clear strip can be horizontal or vertical. The solver does
    # not depend on this sampling convenience, only on separation of buildings.
    leaves = [(5, 0, h, w)] if rng.integers(2) else [(0, 5, h, w)]
    while len(leaves) < count:
        choices = []
        for i, (t, l, b, r) in enumerate(leaves):
            for cut in range(t+7, b-7):
                choices.append((i, [(t, l, cut, r), (cut+1, l, b, r)]))
            for cut in range(l+7, r-7):
                choices.append((i, [(t, l, b, cut), (t, cut+1, b, r)]))
        if not choices:
            return None
        i, children = choices[int(rng.integers(len(choices)))]
        leaves[i:i+1] = children
    return sorted(leaves, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)


def _make_pair(rng, package, palette, corner, test=False):
    bg, scar = map(int, palette[:2])
    q, extras = list(map(int, palette[2:5])), list(map(int, palette[5:7]))
    if package == 'cycle':
        count = int(rng.choice([4, 6]))
        result = count-3
        k = int(rng.choice([x for x in range(3, 7) if x != result]))
        specs = [('cycle', count), ('sealed', k), ('sealed', k)]
    elif package == 'scope':
        specs = [('scope', 6)]
        specs += [(str(rng.choice(['partial', 'free', 'strip'])), int(rng.integers(1, 5)))
                  for _ in range(int(rng.integers(3)))]
    else:
        number = int(rng.integers(1, 4))
        specs = [('partial' if test else str(rng.choice(['partial', 'free', 'strip'])),
                  int(rng.integers(7, 11) if test else rng.integers(1, 7)))]
        specs += [(str(rng.choice(['partial', 'free', 'strip'])), int(rng.integers(1, 5)))
                  for _ in range(number-1)]
    colors = list(map(int, rng.permutation(palette[7:])[:len(specs)]))
    for attempt in range(160):
        low = 24 if len(specs) == 3 else 20 if len(specs) == 2 else 15 if package == 'scope' or test else 10
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        slots = _slots(rng, h, w, len(specs))
        if slots is None:
            continue
        a = np.full((h, w), bg, dtype=np.int8)
        a[3, :4] = a[:4, 3] = scar
        positions = rng.choice(9, 3, replace=False)
        for site, c in zip(positions, rng.permutation(q)):
            a[int(site)//3, int(site)%3] = c
        networks, okay = [], True
        for slot, spec, color in zip(slots, specs, colors):
            t, l, b, r = slot
            built = None
            for _ in range(20):
                # Independent cropping produces varying dimensions and margins.
                sh = int(rng.integers(max(3, (b-t)*3//4), b-t+1))
                sw = int(rng.integers(max(3, (r-l)*3//4), r-l+1))
                built = _local(rng, sh, sw, spec, color, bg, scar, q, extras)
                if built is not None:
                    break
            if built is None:
                okay = False
                break
            patch, n = built
            top, left = int(rng.integers(t, b-sh+1)), int(rng.integers(l, r-sw+1))
            n['box'] = (top, left, top+sh-1, left+sw-1)
            a[top:top+sh, left:left+sw] = patch
            networks.append(n)
        if not okay:
            continue
        # Ties have visibly disjoint horizontal extents. Edge/center conventions
        # therefore agree, including after reflection into a different corner.
        if any(x['count'] == y['count'] and not (x['box'][3] < y['box'][1] or y['box'][3] < x['box'][1])
               for x, y in combinations(networks, 2)):
            continue
        if package == 'cycle':
            spatial = sorted(networks, key=lambda n: (w-1-n['box'][3] if corner & 1 else n['box'][1],
                                                      h-1-n['box'][2] if corner & 2 else n['box'][0]))
            if [n['count'] for n in spatial] == sorted(n['count'] for n in spatial):
                continue
        # Isolated scars never erase a junction or corner and cannot be confused
        # with intentionally absent wall continuations (which remain background).
        damage = 0
        for n in networks:
            wall = n['wall']
            safe = np.zeros_like(wall)
            safe[1:-1, :] |= wall[:-2, :] & wall[1:-1, :] & wall[2:, :]
            safe[:, 1:-1] |= wall[:, :-2] & wall[:, 1:-1] & wall[:, 2:]
            degree = np.zeros(wall.shape, dtype=int)
            degree[1:] += wall[:-1]; degree[:-1] += wall[1:]
            degree[:, 1:] += wall[:, :-1]; degree[:, :-1] += wall[:, 1:]
            safe &= degree == 2
            sites = list(map(tuple, np.argwhere(safe)))
            rng.shuffle(sites)
            wanted = int(rng.integers(0, max(2, len(sites)//12)+1))
            if package == 'cycle' and n is not networks[0]:
                # Corrupt an unopenable internal divider: raw flood-fill fails.
                internal = [p for e, ps in n['gates'].items() if e not in n['eligible']
                            for p in ps if safe[p]]
                if internal:
                    chosen = internal[int(rng.integers(len(internal)))]
                    sites = [chosen] + [p for p in sites if p != chosen]
                    wanted = max(1, wanted)
            chosen_sites = []
            t, l, _, _ = n['box']
            for y, x in sites:
                if len(chosen_sites) >= wanted:
                    break
                if any(max(abs(y-yy), abs(x-xx)) <= 1 for yy, xx in chosen_sites):
                    continue
                a[t+y, l+x] = scar
                chosen_sites.append((y, x))
            damage += len(chosen_sites)
        if package == 'cycle' and not damage:
            continue
        # Irrelevant scars only occupy exterior background, away from buildings.
        forbidden = np.zeros(a.shape, dtype=bool)
        forbidden[:5, :5] = True
        for n in networks:
            t, l, b, r = n['box']
            forbidden[max(0,t-1):min(h,b+2), max(0,l-1):min(w,r+2)] = True
        sites = np.argwhere((a == bg) & ~forbidden)
        if len(sites):
            for y, x in sites[rng.choice(len(sites), min(len(sites), int(rng.integers(0, 9))), replace=False)]:
                a[y, x] = scar
        truth = []
        for n in networks:
            t, l, b, r = n['box']
            if corner & 2:
                t, b = h-1-b, h-1-t
            if corner & 1:
                l, r = w-1-r, w-1-l
            truth.append(dict(color=n['color'], count=n['count'], box=(t,l,b,r)))
        pair = dict(input=_view(a, corner).copy(), output=_encode(truth, scar))
        try:
            parsed = _analyze(pair['input'])
        except ValueError:
            continue
        if np.array_equal(_encode(parsed['networks'], scar), pair['output']):
            # Check exact recovery, not merely agreement of counts.
            intended = {n['color']: n for n in networks}
            if all(np.array_equal(n['wall'], _view(intended[n['color']]['wall'], corner))
                   for n in parsed['networks']):
                return pair
    raise RuntimeError("Bounded construction failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in [('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    vary_roles = bool(rng.integers(2))
    move_corner = bool(rng.integers(2))
    first_corner = int(rng.integers(4))
    corners = [first_corner, (first_corner + int(rng.integers(1, 4))) % 4] if move_corner else [first_corner]
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_query = None
    for split, number in [('train', int(num_train)), ('test', int(num_test))]:
        for i in range(number):
            package = ('cycle' if i == 0 else 'scope' if i == 1 else 'free') if split == 'train' else 'free'
            local_palette = rng.permutation(10) if vary_roles else palette.copy()
            if not vary_roles:
                # Keep background/marker/wall roles stable, but vary which three
                # identities are selected from the five-label palette.
                local_palette[2:7] = rng.permutation(palette[2:7])
            if split == 'train' and i == 1 and frozenset(map(int, local_palette[2:5])) == previous_query:
                local_palette[4], local_palette[5] = local_palette[5], local_palette[4]
            if split == 'train' and i == 0:
                previous_query = frozenset(map(int, local_palette[2:5]))
            corner = corners[i % len(corners)] if split == 'train' and i < 2 else int(rng.choice(corners))
            for _ in range(8):
                pair = _make_pair(rng, package, local_palette, corner, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not avoid duplicate input')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _coverage(infos):
    checks = dict(cycle=False, frozen=False, missing=False, scalar=False,
                  extras=False, empty=False, partial=False, repair=False,
                  sorting=False, ties=False, padding=False)
    for info in infos:
        ns, query = info['networks'], info['query']
        spatial = sorted(ns, key=lambda n: (n['box'][1], n['box'][0]))
        checks['sorting'] |= [n['count'] for n in spatial] != sorted(n['count'] for n in spatial)
        counts = [n['count'] for n in ns]
        checks['ties'] |= len(set(counts)) < len(counts)
        checks['padding'] |= min(counts) < max(counts)
        for n in ns:
            labels, eligible = n['labels'], n['eligible']
            checks['cycle'] |= len(eligible) > len(labels)-n['count']
            checks['frozen'] |= bool(eligible) and query <= frozenset().union(*labels) and n['count'] > 1
            unions = [(labels[i] | labels[j], sum(n['counts'][i].values())+sum(n['counts'][j].values()),
                       (i,j) in eligible) for i,j in n['gates']]
            checks['missing'] |= all(any(not pos and s & query == query-{q} for s, _, pos in unions) for q in query)
            checks['scalar'] |= any(pos and any(not otherpos and len(s) == len(ss) and marks == mm
                                               for ss, mm, otherpos in unions)
                                    for s, marks, pos in unions)
            checks['extras'] |= any(pos and bool(s-query) for s, _, pos in unions)
            checks['empty'] |= any(not s for s in labels)
            checks['partial'] |= _t_junctions(n['wall']) > 0
            checks['repair'] |= any(any(n['grid'][p] == info['scar'] for p in ps)
                                    for e, ps in n['gates'].items() if e not in eligible)
    return checks


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            assert np.array_equal(_encode(info['networks'], info['scar']), out)
            for n in info['networks']:
                assert _physical_count(n) == n['count'], 'Graph and physical topology disagree'
                assert 1 <= n['count'] <= len(n['rooms']) <= 10
                assert len(n['rooms']) == _physical_count(n, set())
                assert all(len(cells) >= 6 for cells in n['rooms'])
                assert np.all(np.isin(n['grid'][n['wall']], [n['color'], info['scar']]))
            assert all(x['count'] != y['count'] or x['box'][3] < y['box'][1] or y['box'][3] < x['box'][1]
                       for x, y in combinations(info['networks'], 2))
            infos[split].append(info)
    checks = _coverage(infos['train'])
    assert all(checks.values()), f'Missing teaching evidence: {checks}'
    train_max = max(len(n['rooms']) for info in infos['train'] for n in info['networks'])
    assert train_max <= 6
    assert all(max(len(n['rooms']) for n in info['networks']) > train_max for info in infos['test'])
    corners = {i['corner'] for i in infos['train']}
    assert all(i['corner'] in corners for i in infos['test'])
    assert len({i['query'] for i in infos['train']}) >= 2, 'The selector must vary in demonstrations'


def _counterfactuals(puzzle):
    # B ablation: close every divider. A ablation: pool the whole building.
    info_list = [_analyze(p['input']) for p in puzzle['train']]
    assert any(n['count'] < len(n['rooms']) for info in info_list for n in info['networks'])
    assert any(info['query'] <= frozenset().union(*n['labels']) and n['count'] > 1
               for info in info_list for n in info['networks'])
    assert any(not info['query'] <= n['labels'][i] and not info['query'] <= n['labels'][j]
               for info in info_list for n in info['networks'] for i,j in n['eligible'])
    repair_matters = False
    for info in info_list:
        for n in info['networks']:
            raw = dict(n, wall=n['grid'] == n['color'])
            repair_matters |= _physical_count(raw) != n['count']
    assert repair_matters, 'Ignoring supported wall repair must change a training count'
    changed_selector = False
    for pair, info in zip(puzzle['train'], info_list):
        a = pair['input']
        # Repositioning query marks has no meaning; keep their identities.
        moved = a.copy()
        v = _view(moved, info['corner'])
        v[:3, :3] = np.roll(v[:3, :3].copy(), 1, axis=0)
        assert np.array_equal(solve(moved), pair['output'])
        all_labels = set().union(*(set().union(*n['labels']) for n in info['networks']))
        for query in combinations(sorted(all_labels), 3):
            if set(query) == info['query']:
                continue
            changed = a.copy()
            v = _view(changed, info['corner'])
            sites = np.argwhere(v[:3, :3] != info['bg'])
            for point, c in zip(sites, query):
                v[tuple(point)] = c
            try:
                result = solve(changed)
            except ValueError:
                continue
            if not np.array_equal(result, pair['output']):
                changed_selector = True
                break
    assert changed_selector, 'Selector-only edits must matter'


def self_test() -> dict:
    # Two direct family fixtures test arrangements outside the packing sampler.
    # An unqueried label drawn as a loop must not become a nested building.
    loop = np.zeros((14,20), dtype=np.int8)
    loop[3,:4] = loop[:4,3] = 9
    loop[0,0], loop[1,1], loop[2,2] = 1,2,3
    loop[5,:] = loop[13,:] = 5
    loop[5:14,0] = loop[5:14,19] = loop[5:14,9] = 5
    loop[7,2:7] = loop[11,2:7] = 4
    loop[7:12,2] = loop[7:12,6] = 4
    loop[7,13], loop[8,14] = 1,2
    assert np.array_equal(solve(loop), np.array([[5,5]]))
    loop[9,15] = 3
    assert np.array_equal(solve(loop), np.array([[5]]))
    # One-cell-high rooms, an edge-touching frame and an internal scar.
    thin = np.zeros((10,24), dtype=np.int8)
    thin[3,:4] = thin[:4,3] = 9
    thin[0,0], thin[1,1], thin[2,2] = 1,2,3
    thin[7,:22] = thin[9,:22] = 5
    thin[7:10, [0,7,14,21]] = 5
    thin[8,1], thin[8,2], thin[8,10], thin[8,14] = 1,2,3,9
    assert np.array_equal(solve(thin), np.array([[5,5]]))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, roles, topologies, original_counts, final_counts = set(), set(), set(), set(), set(), set()
    corners, dimensions, junctions, open_counts, palette_sizes = set(), set(), set(), set(), set()
    metrics = dict(puzzles=42, pairs=0, buildings=0, eligible_edges=0, cyclic_buildings=0,
                   partial_buildings=0, damaged_buildings=0, frozen_witnesses=0)
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        _counterfactuals(puzzle)
        signature = tuple((_key(p['input']), _key(p['output'])) for s in ('train','test') for p in puzzle[s])
        signatures.add(signature)
        if seed < 3:
            again = generate_puzzle(seed)
            assert signature == tuple((_key(p['input']), _key(p['output'])) for s in ('train','test') for p in again[s])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze(pair['input'])
                metrics['pairs'] += 1
                shapes.add(pair['input'].shape)
                dimensions.update(pair['input'].shape)
                palette_sizes.add(len(np.unique(pair['input'])))
                roles.add((info['bg'], info['scar'], tuple(sorted(info['query']))))
                corners.add(info['corner'])
                for n in info['networks']:
                    metrics['buildings'] += 1
                    metrics['eligible_edges'] += len(n['eligible'])
                    metrics['cyclic_buildings'] += len(n['eligible']) > len(n['rooms'])-n['count']
                    metrics['partial_buildings'] += _t_junctions(n['wall']) > 0
                    metrics['damaged_buildings'] += int(np.any(n['grid'][n['wall']] == info['scar']))
                    metrics['frozen_witnesses'] += bool(n['eligible']) and n['count'] > 1 and info['query'] <= frozenset().union(*n['labels'])
                    original_counts.add(len(n['rooms']))
                    final_counts.add(n['count'])
                    junctions.add(_t_junctions(n['wall']))
                    open_counts.add(len(n['eligible']))
                    # Normalize spacing and color while retaining room adjacency.
                    topologies.add((len(n['rooms']), tuple(sorted(n['gates']))))
                # Whole-scene reflections re-evaluate spatial tie order.
                if seed < 3:
                    for transformed in (pair['input'].T, np.fliplr(pair['input']), np.rot90(pair['input'])):
                        changed_info = _analyze(transformed)
                        assert {n['color']: n['count'] for n in changed_info['networks']} == {n['color']: n['count'] for n in info['networks']}
                    perm = np.array([9,7,5,3,1,8,6,4,2,0], dtype=np.int8)
                    assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(roles) >= 35 and corners == {0,1,2,3}
    assert len(topologies) >= 25 and original_counts == set(range(1,11))
    assert len(final_counts) >= 6 and len(junctions) >= 5 and len(open_counts) >= 6
    assert len(palette_sizes) >= 3 and min(dimensions) <= 15 and max(dimensions) == 30
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(827, a, b))
    for a, b in [(1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(2.5,1),(None,1)]:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Output mutations must not be accepted as an alternative topology/count.
    bad = {s: [{k: np.asarray(v).copy() for k,v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['train'][0]['output'][0,0] = (int(bad['train'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Changed output accepted')
    metrics.update(distinct_puzzles=len(signatures), input_shapes=len(shapes), local_roles=len(roles),
                   room_graphs=len(topologies), initial_counts=sorted(original_counts),
                   final_counts=sorted(final_counts), selector_corners=sorted(corners),
                   palette_sizes=sorted(palette_sizes))
    return metrics


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
