"""Backlit cable circuits: optical columns choose labels, cables route them.

No parent generator is imported. Construction records explicit orthogonal paths;
solve reconstructs those paths and the optical probe from the input alone.
"""
from itertools import permutations
import numpy as np

_D = ((-1, 0), (0, 1), (1, 0), (0, -1))
_DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
_CHECKER = ~_DIAMOND


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= k <= 30 for k in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grid must be an integer 1..30 by 1..30 array of colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(cells):
    unseen, result = set(cells), []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, group = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in _D:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _rectangles(g, color):
    rectangles = []
    for cells in _components(map(tuple, np.argwhere(g == color))):
        rr, cc = zip(*cells)
        a, b, x, y = min(rr), max(rr), min(cc), max(cc)
        assert b > a and y > x and len(cells) == (b-a+1)*(y-x+1)
        rectangles.append((a, b, x, y))
    return rectangles


def _mask_links(rect, visible):
    a, b, x, y = rect
    candidates = [((a-1, c), 2) for c in range(x, y+1)]
    candidates += [((b+1, c), 0) for c in range(x, y+1)]
    candidates += [((r, x-1), 1) for r in range(a, b+1)]
    candidates += [((r, y+1), 3) for r in range(a, b+1)]
    ports = []
    for p, direction in candidates:
        degree = sum((p[0]+dr, p[1]+dc) in visible for dr, dc in _D)
        if p in visible and degree in (1, 3):
            ports.append((p, direction))
    assert len(ports) >= 2 and len(ports) % 2 == 0
    if len(ports) == 2:
        return [(ports[0], ports[1])]
    remaining, links = set(ports), []
    while remaining:
        p, d = min(remaining)
        remaining.remove((p, d))
        axis = 1 if d % 2 == 0 else 0
        choices = [(q, e) for q, e in remaining
                   if e == (d+2) % 4 and q[axis] == p[axis]]
        assert len(choices) == 1, "a screen hides an ambiguous cable pairing"
        q, e = choices[0]
        remaining.remove((q, e))
        links.append(((p, d), (q, e)))
    return links


def _trace(field, start, end, centers, carrier, guard, screen):
    visible = set(map(tuple, np.argwhere(field == carrier)))
    if guard is not None:
        visible.update(map(tuple, np.argwhere(field == guard)))
    graph = {p: {} for p in visible}
    endpoints = {(start, c) for c in centers} | {(end, c) for c in centers}
    for p in visible:
        for d, (dr, dc) in enumerate(_D):
            q = p[0]+dr, p[1]+dc
            if q in visible:
                graph[p][d] = (q, (d+2) % 4)
    mask_info = []
    for rect in ([] if screen is None else _rectangles(field, screen)):
        assert start < rect[0] <= rect[1] < end
        links = _mask_links(rect, visible)
        for (p, d), (q, e) in links:
            assert d not in graph[p] and e not in graph[q]
            graph[p][d], graph[q][e] = (q, e), (p, d)
        mask_info.append((rect, links))
    crossings = []
    for p, edges in graph.items():
        if p in endpoints:
            assert len(edges) == 1
            assert (2 if p[0] == start else 0) in edges
        elif len(edges) == 4:
            r, c = p
            assert guard is not None
            assert any(all(field[r+_D[d][0], c+_D[d][1]] == guard for d in ds)
                       for ds in ((0, 2), (1, 3))), "unmarked intersection"
            crossings.append(p)
        else:
            assert len(edges) == 2, "broken, branching or touching cable"
    routes, pairing, covered = [], [], set()
    for c in centers:
        p, incoming, states, route = (start, c), 0, set(), []
        while True:
            assert (p, incoming) not in states
            states.add((p, incoming))
            route.append(p)
            covered.add(p)
            if p in endpoints and p[0] == end:
                pairing.append(centers.index(p[1]))
                break
            if p == (start, c):
                outgoing = 2
            else:
                assert incoming in graph[p]
                outgoing = ((incoming+2) % 4 if len(graph[p]) == 4
                            else next(d for d in graph[p] if d != incoming))
            p, incoming = graph[p][outgoing]
            assert p[0] != start
        routes.append(route)
    assert sorted(pairing) == list(range(len(centers)))
    assert covered == visible, "unattached cable material"
    return pairing, routes, mask_info, crossings


def _fit_flank(values):
    """Longest terminal periodic outward stair, with its least period."""
    v = np.asarray(values, dtype=int)
    t = np.arange(len(v)) - len(v) + 1
    fits = []
    for period in (1, 2, 3):
        for phase in range(period):
            predicted = v[-1] + (t+phase)//period
            length = 0
            for same in (predicted == v)[::-1]:
                if not same:
                    break
                length += 1
            if length >= 2*period+1 and v[-1]-v[-length] >= 2:
                fits.append((length, -period, phase))
    assert fits, "insufficient terminal flank evidence"
    length, neg, phase = max(fits)
    return -neg, phase, int(v[-1]), length


def _emitter_candidates(g, bg):
    h, w = g.shape
    for wall in set(map(int, np.unique(g))) - {bg}:
        cells = np.argwhere(g == wall)
        if len(cells) < 9:
            continue
        mouth, rear = int(cells[:, 0].min()), int(cells[:, 0].max())
        if rear-mouth < 3 or mouth < 9:
            continue
        cap = np.flatnonzero(g[rear] == wall)
        if len(cap) < 3 or np.any(np.diff(cap) != 1):
            continue
        rails = [np.flatnonzero(g[r] == wall) for r in range(rear-1, mouth-1, -1)]
        if any(len(v) != 2 or v[1]-v[0] < 2 for v in rails):
            continue
        left, right = np.array(rails).T
        if left[0] != cap[0] or right[0] != cap[-1]:
            continue
        try:
            fits = (_fit_flank(-left), _fit_flank(right))
        except AssertionError:
            continue
        footprint = np.zeros(g.shape, dtype=bool)
        for r in range(rear):
            if r >= mouth:
                lo, hi = int(left[rear-1-r]), int(right[rear-1-r])
            else:
                advance = mouth-r
                p, phase, value, _ = fits[0]
                lo = -value-(advance+phase)//p
                p, phase, value, _ = fits[1]
                hi = value+(advance+phase)//p
            footprint[r] = (np.arange(w) > lo) & (np.arange(w) < hi)
        chamber = footprint.copy()
        chamber[:mouth] = False
        markers = np.argwhere(chamber & (g != bg) & (g != wall))
        if not len(markers):
            continue
        colors = {int(g[tuple(p)]) for p in markers}
        if len(colors) != 1:
            continue
        payload = colors.pop()
        patch = set(map(tuple, np.argwhere(g == payload)))
        if any(not chamber[p] for p in patch) or len(_components(patch)) != 1:
            continue
        yield dict(wall=wall, payload=payload, mouth=mouth, rear=rear,
                   fits=fits, footprint=footprint, chamber=chamber)


def _parse(input_grid):
    original = _grid(input_grid)
    corners = [int(original[r, c]) for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    assert len(set(corners)) == 1
    bg, matches = corners[0], []
    for rotation in range(4):
        g = np.rot90(original, -rotation).copy()
        for optical in _emitter_candidates(g, bg):
            circuit_limit = optical['mouth']-4
            for carrier in set(map(int, np.unique(g[:circuit_limit]))) - {bg, optical['wall'], optical['payload']}:
                points = np.argwhere(g[:circuit_limit] == carrier)
                start, end = int(points[:, 0].min()), int(points[:, 0].max())
                if start < 3 or end > optical['mouth']-5 or end-start < 4:
                    continue
                centers = list(map(int, np.flatnonzero(g[start] == carrier)))
                if (not 2 <= len(centers) <= 4
                        or centers != list(np.flatnonzero(g[end] == carrier))
                        or len(set(np.diff(centers))) != 1
                        or centers[1]-centers[0] < 4
                        or centers[0] < 1 or centers[-1] >= g.shape[1]-1):
                    continue
                try:
                    labels, terminal_mask = [], np.zeros(g.shape, dtype=bool)
                    for c in centers:
                        source = g[start-3:start, c-1:c+2]
                        label = int(source[1, 1])
                        assert label != bg and np.all(source == label)
                        assert label not in {carrier, optical['wall'], optical['payload']}
                        labels.append(label)
                        assert np.all(g[end+1:end+4, c-1:c+2] == bg)
                        terminal_mask[start-3:start, c-1:c+2] = True
                        terminal_mask[end+1:end+4, c-1:c+2] = True
                    field = np.full(g.shape, bg, dtype=np.int8)
                    field[start:end+1] = g[start:end+1]
                    other = set(map(int, np.unique(field))) - {bg, carrier}
                    assert len(other) <= 2
                    assert not other & (set(labels) | {optical['wall'], optical['payload']})
                    guard, screen = None, None
                    for color in other:
                        try:
                            _rectangles(field, color)
                            assert screen is None
                            screen = color
                        except AssertionError:
                            assert guard is None
                            guard = color
                    pairing, routes, masks, crosses = _trace(field, start, end, centers, carrier, guard, screen)
                    permitted = terminal_mask | (field != bg) | (g == optical['wall']) | (g == optical['payload'])
                    assert np.all(g[~permitted] == bg), "extra material outside the family grammar"
                    opaque = np.zeros(g.shape, dtype=bool)
                    for role in (guard, screen):
                        if role is not None:
                            opaque |= g == role
                    # Scan upward: shadows stay in screen columns even as cables bend.
                    lit, shadow = np.zeros(g.shape, dtype=bool), np.zeros(g.shape, dtype=bool)
                    for c in range(g.shape[1]):
                        blocked = False
                        for r in range(optical['rear']-1, -1, -1):
                            if not optical['footprint'][r, c]:
                                continue
                            shadow[r, c] = blocked
                            if opaque[r, c]:
                                blocked = True
                            elif not blocked:
                                lit[r, c] = True
                    active = tuple(bool(lit[start-2, c]) for c in centers)
                    matches.append(dict(optical, grid=g, rotation=rotation, background=bg,
                                        carrier=carrier, guard=guard, screen=screen, start=start,
                                        end=end, centers=centers, labels=tuple(labels), pairing=tuple(pairing),
                                        routes=routes, masks=masks, crossings=crosses,
                                        terminal_mask=terminal_mask, opaque=opaque, lit=lit,
                                        shadow=shadow, active=active))
                except (AssertionError, StopIteration, IndexError):
                    continue
    assert len(matches) == 1, f"expected one input interpretation, found {len(matches)}"
    return matches[0]


def _render(info, active=None, pairing=None):
    g = info['grid'].copy()
    active = info['active'] if active is None else active
    pairing = info['pairing'] if pairing is None else pairing
    for i, j in enumerate(pairing):
        color = info['payload'] if active[i] else info['labels'][i]
        for row, c, pattern in ((info['start']-3, info['centers'][i], _DIAMOND),
                                (info['end']+1, info['centers'][j], _CHECKER)):
            g[row:row+3, c-1:c+2] = np.where(pattern, color, info['background'])
    return np.rot90(g, info['rotation']).copy()


def _activation_without(info, role):
    """Optical counterfactual; cable connectivity remains unchanged."""
    opaque = info['opaque'] & (info['grid'] != role)
    return tuple(bool(info['footprint'][info['start']-2, c]
                      and not np.any(opaque[info['start']-1:, c]
                                     & info['footprint'][info['start']-1:, c]))
                 for c in info['centers'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover optical activation and cable identity using only input cells."""
    return _render(_parse(input_grid))


def _moves(target, rng):
    """Route arbitrary cycles through a vacant exterior track."""
    n, state, result = len(target), list(range(len(target)))+[None], []
    for j in map(int, rng.permutation(n)):
        if state[j] == target[j]:
            continue
        saved, vacant = state[j], j
        result.append((j, n))
        state[n], state[j] = saved, None
        while target[vacant] != saved:
            old = state.index(target[vacant])
            result.append((old, vacant))
            state[vacant], state[old] = state[old], None
            vacant = old
        result.append((n, vacant))
        state[vacant], state[n] = saved, None
    assert state[:n] == target
    return result


def _make_pair(rng, palette, rotation, n, lesson, wanted_masks, longer=False, target_override=None):
    bg, carrier, guard, screen, wall, payload = map(int, palette[:6])
    target = list(range(n)) if lesson == 'clean' else list(map(int, rng.permutation(n)))
    if target_override is not None:
        target = list(target_override)
    if lesson == 'coupled' and target == list(range(n)):
        target[0], target[-1] = target[-1], target[0]
    moves = _moves(target, rng)
    # A reversible detour varies geometry even for an identity permutation.
    if lesson != 'coupled' and len(moves) <= 3 and rng.random() < .3:
        k = int(rng.integers(n))
        moves = [(k, n), (n, k)] + moves
    if lesson == 'clean':
        moves = []
    pitch_max = min(6, (30-7)//(n-1))
    pitch = int(rng.integers(4, pitch_max+1))
    spare_gap = int(rng.integers(3, 6))
    minimum_width = 4+(n-1)*pitch+spare_gap
    width = int(rng.integers(minimum_width, 31))
    left = int(rng.integers(1, width-(n-1)*pitch-spare_gap-2))
    centers = [left+1+i*pitch for i in range(n)]
    columns = centers+[centers[-1]+spare_gap]
    minimum_route_length = 4+2*max(0, len(moves)-1)
    max_run = min(3, (18-minimum_route_length)//2)
    run = int(rng.integers(1, max_run+1))
    flare = 2*run+1
    gaps = [2]*max(0, len(moves)-1)
    lead, tail, top, gap, bottom = 2, 2, 1, 1, 1
    visible = flare
    route_length = lead+tail+sum(gaps)
    minimum_height = top+3+route_length+5+visible+bottom+1
    budget = 30-minimum_height
    assert budget >= 0
    rear_length = int(rng.integers(min(3, budget)+1))
    visible += rear_length
    budget -= rear_length
    extras = int(rng.integers(budget+1))
    for _ in range(extras):
        options = ['top', 'lead', 'tail', 'gap', 'bottom']+['segment']*len(gaps)
        if longer:
            options += ['lead', 'tail']*3
        option = str(rng.choice(options))
        if option == 'top': top += 1
        elif option == 'lead': lead += 1
        elif option == 'tail': tail += 1
        elif option == 'gap': gap += 1
        elif option == 'bottom': bottom += 1
        else: gaps[int(rng.integers(len(gaps)))] += 1
    start, move_rows = top+3, []
    row = start+lead
    for k in range(len(moves)):
        move_rows.append(row)
        if k < len(gaps): row += gaps[k]
    end = move_rows[-1]+tail if moves else start+lead+tail
    mouth = end+4+gap
    rear = mouth+visible
    height = rear+bottom+1
    paths, state, previous = [[(start, c)] for c in centers], list(range(n))+[None], start
    for r, (old, new) in zip(move_rows, moves):
        for lane, owner in enumerate(state):
            if owner is not None:
                paths[owner].extend((rr, columns[lane]) for rr in range(previous+1, r+1))
        owner = state[old]
        step = 1 if columns[new] > columns[old] else -1
        paths[owner].extend((r, c) for c in range(columns[old]+step, columns[new]+step, step))
        state[new], state[old], previous = state[old], None, r
    for lane, owner in enumerate(state):
        if owner is not None:
            paths[owner].extend((r, columns[lane]) for r in range(previous+1, end+1))
    field, visits = np.full((height, width), bg, dtype=np.int8), {}
    for owner, path in enumerate(paths):
        for p in path:
            field[p] = carrier
            visits.setdefault(p, set()).add(owner)
    for (r, c), owners in visits.items():
        if len(owners) > 1:
            assert len(owners) == 2
            axes = (0, 2) if rng.random() < .5 else (1, 3)
            for d in axes:
                dr, dc = _D[d]
                field[r+dr, c+dc] = guard
    masks = []
    for _ in range(90):
        if len(masks) >= wanted_masks: break
        r, c = sorted(visits)[int(rng.integers(len(visits)))]
        hh, ww = int(rng.integers(2, 5)), int(rng.integers(2, min(9, width)))
        r0 = max(start+1, min(end-hh, r-int(rng.integers(hh))))
        c0 = max(1, min(width-ww-1, c-int(rng.integers(ww))))
        rect = (r0, r0+hh-1, c0, c0+ww-1)
        a, b, x, y = rect
        if b >= end or any(not (b+1 < aa or bb+1 < a or y+1 < xx or yy+1 < x)
                           for aa, bb, xx, yy in masks): continue
        inside = lambda p: a <= p[0] <= b and x <= p[1] <= y
        truth = set()
        for path in paths:
            entry = None
            for p, q in zip(path, path[1:]):
                if not inside(p) and inside(q): entry = p
                if inside(p) and not inside(q):
                    if entry is None: break
                    truth.add(frozenset((entry, q)))
                    entry = None
        if not truth: continue
        candidate = field.copy()
        candidate[a:b+1, x:y+1] = screen
        try:
            pairing, _, mask_info, crossings = _trace(candidate, start, end, centers, carrier,
                                                     guard if np.any(candidate == guard) else None, screen)
            links = next(links for rr, links in mask_info if rr == rect)
            assert {frozenset((p, q)) for (p, _), (q, _) in links} == truth
            assert pairing == [target.index(i) for i in range(n)]
            if lesson == 'coupled': assert crossings
        except (AssertionError, StopIteration, KeyError): continue
        field, masks = candidate, masks+[rect]
    if lesson == 'coupled' and not masks:
        return None
    # Separate rear chamber profile and visibly periodic flanks.
    phase_left, phase_right = int(rng.integers(run)), int(rng.integers(run))
    widths = []
    base_left, base_right = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    for t in range(1, visible+1):
        if t <= rear_length:
            u = max(1, base_left+(1 if t == 2 else 0))
            v = max(1, base_right+(1 if t == 1 and rear_length > 1 else 0))
        else:
            k = t-rear_length-1
            u = base_left+(k+phase_left)//run
            v = base_right+(k+phase_right)//run
        widths.append((u, v))
    span_left, span_right = max(u for u, v in widths), max(v for u, v in widths)
    available = list(range(span_left+1, width-span_right-1))
    if not available: return None
    center = int(rng.choice(available))
    if lesson == 'clean':
        # Center the probe; the clean lesson must illuminate every detector.
        center = min(available, key=lambda c: abs(c-(centers[0]+centers[-1])/2))
    first_u, first_v = widths[0]
    field[rear, center-first_u:center+first_v+1] = wall
    footprint = np.zeros(field.shape, dtype=bool)
    for t, (u, v) in enumerate(widths, 1):
        r = rear-t
        field[r, center-u] = field[r, center+v] = wall
        footprint[r, center-u+1:center+v] = True
    for r in range(mouth):
        k = rear-r-rear_length-1
        lo, hi = center-base_left-(k+phase_left)//run, center+base_right+(k+phase_right)//run
        footprint[r] = (np.arange(width) > lo) & (np.arange(width) < hi)
    patch_options = [(r, c) for r, c in map(tuple, np.argwhere(footprint)) if mouth < r < rear]
    p = patch_options[int(rng.integers(len(patch_options)))]
    patch = {p}
    for _ in range(int(rng.integers(4))):
        frontier = sorted({(r+dr, c+dc) for r, c in patch for dr, dc in _D
                           if (r+dr, c+dc) in patch_options and (r+dr, c+dc) not in patch})
        if frontier: patch.add(frontier[int(rng.integers(len(frontier)))])
    for p in patch: field[p] = payload
    label_pool = list(map(int, palette[6:]))
    labels = label_pool[:n] if lesson else list(map(int, rng.choice(label_pool, n)))
    rng.shuffle(labels)
    for i, c in enumerate(centers): field[start-3:start, c-1:c+2] = labels[i]
    # Construction uses a union of strict downstream shadows; solve uses a scan.
    shadow = np.zeros(field.shape, dtype=bool)
    opaque = (field == guard) | (field == screen)
    for r, c in np.argwhere(opaque & footprint): shadow[:r, c] = True
    active = tuple(bool(footprint[start-2, c] and not shadow[start-2, c]) for c in centers)
    pairing = tuple(centers.index(path[-1][1]) for path in paths)
    if lesson == 'clean' and not all(active): return None
    if lesson == 'coupled':
        if not any(active) or all(active): return None
        exposures = [int(np.sum(footprint[start-3:start, c-1:c+2]
                                & ~shadow[start-3:start, c-1:c+2])) for c in centers]
        if not any(not state and exposure > 0 for state, exposure in zip(active, exposures)):
            return None
        if not any(state and exposure < 9 for state, exposure in zip(active, exposures)):
            return None
        # Two separate dark contacts teach both kinds of opaque material.
        for role in (guard, screen):
            other_shadow = np.zeros(field.shape, dtype=bool)
            for r, c in np.argwhere(opaque & (field != role) & footprint):
                other_shadow[:r, c] = True
            without_role = tuple(bool(footprint[start-2, c] and not other_shadow[start-2, c])
                                 for c in centers)
            if without_role == active: return None
        colors = [payload if a else labels[i] for i, a in enumerate(active)]
        received = [colors[pairing.index(j)] for j in range(n)]
        if received == colors: return None
    out = field.copy()
    for i, j in enumerate(pairing):
        color = payload if active[i] else labels[i]
        out[start-3:start, centers[i]-1:centers[i]+2] = np.where(_DIAMOND, color, bg)
        out[end+1:end+4, centers[j]-1:centers[j]+2] = np.where(_CHECKER, color, bg)
    if rng.random() < .5: field, out = field[:, ::-1].copy(), out[:, ::-1].copy()
    return {'input': np.rot90(field, rotation).copy(), 'output': np.rot90(out, rotation).copy()}


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= nq: raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shared_palette, palette = bool(rng.random() < .2), rng.permutation(10)
    shift = str(rng.choice(['count', 'screens', 'orientation', 'span']))
    preferred = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            lesson = ('clean' if index == 0 else 'coupled' if index == 1 else None) if split == 'train' else None
            n = int(rng.integers(2, 4 if split == 'train' and shift == 'count' else 5))
            if split == 'test' and shift == 'count': n = 4
            if lesson == 'clean': n = int(rng.integers(2, 4))
            if lesson == 'coupled': n = 3
            wanted = 0 if lesson == 'clean' else 1 if lesson == 'coupled' else int(rng.integers(4))
            if shift == 'screens':
                if split == 'test':
                    wanted, n = int(rng.integers(2, 4)), int(rng.integers(3, 5))
                elif not lesson:
                    wanted = int(rng.integers(2))
            rotation = ((preferred+1) % 4 if split == 'test' else preferred) if shift == 'orientation' else int(rng.integers(4))
            colors = palette if shared_palette else rng.permutation(10)
            for attempt in range(240):
                pair = _make_pair(rng, colors, rotation, n, lesson, wanted, split == 'test' and shift == 'span')
                if pair is None: continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key in seen: continue
                try:
                    analysis = _parse(pair['input'])
                    assert np.array_equal(_render(analysis), pair['output'])
                    if shift == 'screens' and split == 'test':
                        assert len(analysis['masks']) >= 2
                    if shift == 'span':
                        distance = analysis['mouth']-analysis['start']+2
                        assert distance >= 19 if split == 'test' else distance <= 17
                except (AssertionError, ValueError, KeyError): continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded circuit construction exhausted")
    rng.shuffle(puzzle['train'])
    return puzzle


def measure_pair(input_grid):
    a = _parse(input_grid)
    lengths = [sum(abs(p[0]-q[0])+abs(p[1]-q[1]) for p, q in zip(path, path[1:])) for path in a['routes']]
    shadowed = [bool(a['footprint'][a['start']-2, c] and a['shadow'][a['start']-2, c]) for c in a['centers']]
    hidden_elbows = sum(d % 2 != e % 2 for _, links in a['masks'] for (_, d), (_, e) in links)
    return dict(shape=tuple(a['grid'].shape), direction=a['rotation'], routes=len(a['centers']),
                pitch=a['centers'][1]-a['centers'][0], pairing=a['pairing'], lengths=tuple(lengths),
                crossings=len(a['crossings']), screens=len(a['masks']), elbows=hidden_elbows,
                screen_shapes=tuple((b-r+1, y-x+1) for (r, b, x, y), _ in a['masks']),
                periods=tuple(f[0] for f in a['fits']), phases=tuple(f[1] for f in a['fits']),
                visible=a['rear']-a['mouth'], projection=a['mouth']-(a['start']-2),
                active=a['active'], shadowed=tuple(shadowed),
                outside=tuple(not a['footprint'][a['start']-2, c] for c in a['centers']),
                roles=(a['background'], a['carrier'], a['guard'], a['screen'], a['wall'], a['payload']),
                labels=a['labels'], payload_area=int(np.sum(a['grid'] == a['payload'])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, clean, coupled = set(), False, False
    regimes = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = inp.shape, inp.tobytes()
            assert key not in seen
            seen.add(key)
            a = _parse(inp)
            regimes[split].append((len(a['centers']), len(a['masks']), a['rotation'],
                                   a['mouth']-a['start']+2))
            assert np.array_equal(_render(a), out), 'input-derived answer mismatch'
            output = np.rot90(out, -a['rotation'])
            assert np.array_equal(a['grid'][~a['terminal_mask']], output[~a['terminal_mask']])
            assert not np.array_equal(inp, out)
            if split == 'train':
                clean |= all(a['active']) and not np.any(a['opaque'])
                ignored_a = _render(a, pairing=tuple(range(len(a['centers']))))
                ignored_b = _render(a, active=(False,)*len(a['centers']))
                exposures = [int(a['lit'][a['start']-3:a['start'], c-1:c+2].sum())
                             for c in a['centers']]
                coupled |= (bool(a['crossings']) and bool(a['masks'])
                            and any(a['active']) and not all(a['active'])
                            and not np.array_equal(ignored_a, out)
                            and not np.array_equal(ignored_b, out)
                            and _activation_without(a, a['guard']) != a['active']
                            and _activation_without(a, a['screen']) != a['active']
                            and any(not state and exposure > 0 for state, exposure in zip(a['active'], exposures))
                            and any(state and exposure < 9 for state, exposure in zip(a['active'], exposures)))
    assert clean, 'missing clean optical/stencil lesson'
    assert coupled, 'missing routed, screened, mixed-activation lesson'
    t, q = regimes['train'], regimes['test']
    count_shift = max(v[0] for v in t) < min(v[0] for v in q)
    screen_shift = max(v[1] for v in t) < min(v[1] for v in q)
    directions_t, directions_q = {v[2] for v in t}, {v[2] for v in q}
    orientation_shift = (len(directions_t) == len(directions_q) == 1
                         and next(iter(directions_q)) == (next(iter(directions_t))+1) % 4)
    span_shift = max(v[3] for v in t) <= 17 and min(v[3] for v in q) >= 19
    assert count_shift or screen_shift or orientation_shift or span_shift, 'missing structural test shift'


def self_test():
    """Bounded schema, semantic ablation and structural variation audit."""
    records, keys = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _parse(pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                records.append((split, measure_pair(pair['input'])))
                keys.add((pair['input'].shape, pair['input'].tobytes()))
                # Shadows follow optical columns, not the routes that carry labels.
                for i, j in enumerate(a['pairing']):
                    c = a['centers'][j]
                    color = a['payload'] if a['active'][i] else a['labels'][i]
                    assert a['grid'][a['end']+2, c] == a['background']
                    o = np.rot90(pair['output'], -a['rotation'])
                    assert o[a['end']+2, c] == color
        if seed in (0, 13, 41):
            q = generate_puzzle(seed)
            assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    fs = [f for _, f in records]
    assert len(keys) == len(records) == 210
    assert len({f['shape'] for f in fs}) >= 35
    assert {f['routes'] for f in fs} == {2, 3, 4}
    assert {f['pitch'] for f in fs} == {4, 5, 6}
    assert {p for f in fs for p in f['periods']} == {1, 2, 3}
    assert len({f['pairing'] for f in fs}) >= 12
    assert len({f['roles'] for f in fs}) >= 50
    assert len({f['screen_shapes'] for f in fs if f['screens']}) >= 15
    assert any(f['elbows'] for f in fs)
    assert any(any(f['outside']) for f in fs)
    assert any(f['screens'] >= 2 for f in fs)
    assert any(len(set(f['labels'])) < f['routes'] for f in fs)
    assert any(f['phases'][0] != f['phases'][1] for f in fs)
    for split in ('train', 'test'):
        part = [f for s, f in records if s == split]
        assert {f['direction'] for f in part} == {0, 1, 2, 3}
        assert any(any(f['active']) and not all(f['active']) for f in part)
    # Exhaust the supported four-cable permutation space with finished inputs.
    routing_rng = np.random.default_rng(72819)
    for target in permutations(range(4)):
        for _ in range(12):
            pair = _make_pair(routing_rng, np.arange(10), 0, 4, None, 0,
                              target_override=target)
            if pair is not None:
                a = _parse(pair['input'])
                expected = tuple(target.index(i) for i in range(4))
                # Undo a sampled lateral reflection so every literal pairing,
                # rather than just its reflection class, is tested in an input.
                if a['pairing'] != expected:
                    pair = {k: v[:, ::-1].copy() for k, v in pair.items()}
                    a = _parse(pair['input'])
                assert a['pairing'] == expected
                assert np.array_equal(_render(a), pair['output'])
                break
        else: raise AssertionError('permutation construction exhausted')
    # Hand-written thin-stair continuation checks, including all period-3 phases.
    for values, period, phase, next_value in (
        ((1, 2, 3), 1, 0, 4), ((1, 1, 2, 2, 3), 2, 0, 3),
        ((1, 2, 2, 3, 3), 2, 1, 4),
        ((1, 1, 1, 2, 2, 2, 3), 3, 0, 3),
        ((1, 1, 2, 2, 2, 3, 3), 3, 1, 3),
        ((1, 2, 2, 2, 3, 3, 3), 3, 2, 4)):
        p, ph, last, _ = _fit_flank(values)
        assert (p, ph) == (period, phase)
        assert last+(1+ph)//p == next_value
    # Whole-grid rotations and role permutations preserve the answer.
    base = generate_puzzle(91, 2, 1)['test'][0]
    lut = np.array([7, 3, 0, 8, 1, 9, 2, 5, 4, 6], dtype=np.int8)
    for rotation in range(4):
        inp = np.rot90(lut[base['input']], rotation).copy()
        expected = np.rot90(lut[base['output']], rotation).copy()
        assert np.array_equal(solve(inp), expected)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(10000+nt*10+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try: generate_puzzle(0, nt, nq)
        except ValueError: pass
        else: raise AssertionError('illegal counts accepted')
    # Reject a missing checker cell and a nonterminal mutation.
    p = generate_puzzle(83, 2, 1)
    for terminal in (True, False):
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = damaged['test'][0]
        if terminal:
            r, c = np.argwhere(pair['input'] != pair['output'])[0]
        else: r, c = 0, 0
        pair['output'][r, c] = (int(pair['output'][r, c])+1) % 10
        try: validate_generated_puzzle(damaged)
        except AssertionError: pass
        else: raise AssertionError('corrupted output accepted')
