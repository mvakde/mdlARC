"""Spectral exposure of homologous compartment maps."""
from __future__ import annotations

from collections import Counter
import numpy as np

ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG = ORTH + ((-1, -1), (-1, 1), (1, -1), (1, 1))
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in DIAG if diagonal else ORTH:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(np.asarray(sorted(cells), dtype=int))
    return result


def _bbox(cells):
    return (*map(int, cells.min(axis=0)), *map(int, cells.max(axis=0) + 1))


def _rooms(walls):
    empty = np.pad(~walls, 1, constant_values=True)
    return [part - 1 for part in _components(empty)
            if np.all(part > 0) and np.all(part < np.asarray(empty.shape) - 1)]


def _relations(rooms):
    boxes = [_bbox(room) for room in rooms]
    return [[tuple(-1 if a[k + 2] <= b[k] else
                   1 if b[k + 2] <= a[k] else 0 for k in (0, 1))
             for b in boxes] for a in boxes]


def _mappings(donor, target):
    """Match the complete oriented spatial graph, without areas or colors."""
    a, b = _relations(donor), _relations(target)
    n = len(a)
    if n != len(b):
        raise ValueError("unequal room counts")
    signatures_a = [Counter(row) for row in a]
    signatures_b = [Counter(row) for row in b]
    options = [[j for j in range(n) if signatures_a[i] == signatures_b[j]]
               for i in range(n)]
    answers, visits = [], 0

    def search(mapping, used):
        nonlocal visits
        visits += 1
        if visits > 20000:
            raise ValueError("insufficient spatial distinction")
        if len(mapping) == n:
            answers.append(dict(mapping))
            return
        available = {i: [j for j in options[i] if j not in used and all(
            a[i][k] == b[j][v] for k, v in mapping.items())]
                     for i in range(n) if i not in mapping}
        i = min(available, key=lambda k: (len(available[k]), k))
        for j in available[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({}, set())
    if not answers:
        raise ValueError("no oriented spatial correspondence")
    return answers


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or not all(1 <= s <= 30 for s in grid.shape)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("invalid ARC grid")
    return grid.astype(int, copy=False)


def _parse(input_grid):
    grid = _grid(input_grid)
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    counts = np.bincount(border, minlength=10)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("ambiguous exterior background")
    bg = int(counts.argmax())
    candidates = []
    for wall in sorted(set(map(int, grid.flat)) - {bg}):
        frameworks = []
        for part in _components(grid == wall, diagonal=True):
            if len(part) < 12:
                continue
            r, c, R, C = _bbox(part)
            mask = np.zeros((R - r, C - c), dtype=bool)
            mask[part[:, 0] - r, part[:, 1] - c] = True
            rooms = _rooms(mask)
            if 4 <= len(rooms) <= 8:
                frameworks.append((part, (r, c, R, C), mask, rooms))
        if len(frameworks) == 2:
            candidates.append((wall, frameworks))
    if len(candidates) != 1:
        raise ValueError("expected one shared two-framework wall role")
    wall, frames = candidates[0]
    objects = []
    figure_mask = np.zeros(grid.shape, dtype=bool)
    for part, box, mask, local_rooms in frames:
        r, c, R, C = box
        crop = np.full(mask.shape, bg, dtype=int)
        crop[mask] = wall
        rooms = [p + (r, c) for p in local_rooms]
        shades = []
        figure_mask[tuple(part.T)] = True
        for room, local in zip(rooms, local_rooms):
            vals = np.unique(grid[tuple(room.T)])
            if len(vals) != 1 or vals[0] == wall:
                raise ValueError("room must be uniformly colored or empty")
            shades.append(int(vals[0]))
            crop[tuple(local.T)] = int(vals[0])
            figure_mask[tuple(room.T)] = True
        objects.append(dict(box=box, crop=crop, rooms=rooms,
                            local_rooms=local_rooms, colors=shades))
    donors = [obj for obj in objects if all(c != bg for c in obj['colors'])]
    targets = [obj for obj in objects if all(c == bg for c in obj['colors'])]
    if len(donors) != 1 or len(targets) != 1:
        raise ValueError("expected a fully colored donor and empty target")
    donor, target = donors[0], targets[0]
    sources, mirrors = [], []
    outside = (grid != bg) & ~figure_mask
    for shade in sorted(set(map(int, grid[outside]))):
        if shade == wall:
            raise ValueError("optics cannot use the framework color")
        for cells in _components(outside & (grid == shade)):
            r, c, R, C = _bbox(cells)
            if len(cells) == 3 and (R - r, C - c) == (2, 2):
                missing = next(p for p in ((r, c), (r, c + 1),
                                          (r + 1, c), (r + 1, c + 1))
                               if p not in set(map(tuple, cells)))
                elbow = (r + R - 1 - missing[0], c + C - 1 - missing[1])
                direction = (elbow[0] - missing[0], elbow[1] - missing[1])
                sources.append(dict(cells=set(map(tuple, cells)), elbow=elbow,
                                    direction=direction, color=shade))
            elif len(cells) % 2 == 1 and (R - r == 1 or C - c == 1):
                if len(cells) != max(R - r, C - c):
                    raise ValueError("discontinuous bar")
                mirrors.append(dict(cells=set(map(tuple, cells)),
                                    midpoint=((r + R - 1) // 2, (c + C - 1) // 2),
                                    axis=None if len(cells) == 1 else (0 if R - r == 1 else 1),
                                    color=shade))
            else:
                raise ValueError("unrecognized exterior optic")
    if not sources:
        raise ValueError("missing L emitter")
    return dict(grid=grid, background=bg, wall=wall, donor=donor,
                target=target, sources=sources, mirrors=mirrors)


def _trace(shape, sources, mirrors, *, recolor=True, reflect=True):
    """Only optical objects interact with the ray; maps are transparent."""
    h, w = shape
    occupied = {p for obj in sources + mirrors for p in obj['cells']}
    illuminated, routes = {}, []
    for source in sources:
        dr, dc = source['direction']
        shade = source['color']
        r, c = source['elbow']
        r, c = r + dr, c + dc
        states, path, contacts = set(), [], []
        while 0 <= r < h and 0 <= c < w:
            state = (r, c, dr, dc)
            if state in states:
                raise ValueError("cyclic optical route")
            states.add(state)
            if (r, c) in occupied:
                raise ValueError("ray strikes an optic rather than its midpoint face")
            hits = []
            for i, mirror in enumerate(mirrors):
                mr, mc = mirror['midpoint']
                if mirror['axis'] in (None, 0) and mc == c and mr - r == dr:
                    hits.append((i, 0))
                if mirror['axis'] in (None, 1) and mr == r and mc - c == dc:
                    hits.append((i, 1))
            if len(hits) > 1:
                raise ValueError("ambiguous midpoint contact")
            if hits:
                i, axis = hits[0]
                contacts.append((i, axis, (r, c), shade, mirrors[i]['color']))
                if recolor:
                    shade = mirrors[i]['color']
                if reflect:
                    dr, dc = (-dr, dc) if axis == 0 else (dr, -dc)
            illuminated.setdefault((r, c), set()).add(shade)
            path.append(((r, c), shade))
            r, c = r + dr, c + dc
        routes.append(dict(path=path, contacts=contacts, direction=(dr, dc)))
    return illuminated, routes


def _room_index(rooms):
    """Index disjoint interiors once when trying many routes on one geometry."""
    return {tuple(p): i for i, room in enumerate(rooms) for p in room.tolist()}


def _room_light(rooms, illuminated, room_index=None):
    if room_index is None:
        room_index = _room_index(rooms)
    seen = [set() for _ in rooms]
    for p, colors in illuminated.items():
        i = room_index.get(p)
        if i is not None:
            seen[i].update(colors)
    return seen


def _analyze(input_grid):
    info = _parse(input_grid)
    light, routes = _trace(info['grid'].shape, info['sources'], info['mirrors'])
    seen = _room_light(info['donor']['rooms'], light)
    colors = info['donor']['colors']
    exposed = [color in seen[i] for i, color in enumerate(colors)]
    mappings = _mappings(info['donor']['rooms'], info['target']['rooms'])
    answer = None
    for mapping in mappings:
        out = info['target']['crop'].copy()
        for i, j in mapping.items():
            if exposed[i]:
                out[tuple(info['target']['local_rooms'][j].T)] = colors[i]
        if answer is not None and not np.array_equal(out, answer):
            raise ValueError("ambiguous transferred exposure")
        answer = out
    info.update(light=light, routes=routes, seen=seen, exposed=exposed,
                mappings=mappings)
    return answer, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _analyze(input_grid)[0]


def _plan(rng, count):
    options = [(2, 2), (2, 3), (3, 2), (3, 3)]
    options = [s for s in options if s[0] * s[1] >= count]
    rows, cols = options[int(rng.integers(len(options)))]
    plan = np.arange(rows * cols).reshape(rows, cols)
    if rows * cols > count and rng.random() < .35:
        corner = ((0, 0), (0, cols - 1), (rows - 1, 0),
                  (rows - 1, cols - 1))[int(rng.integers(4))]
        plan[corner] = -1
    while len(set(map(int, plan.flat)) - {-1}) > count:
        edges = [(int(plan[r, c]), int(plan[r + dr, c + dc]))
                 for r in range(rows) for c in range(cols) for dr, dc in ((1, 0), (0, 1))
                 if r + dr < rows and c + dc < cols and plan[r, c] >= 0
                 and plan[r + dr, c + dc] >= 0 and plan[r, c] != plan[r + dr, c + dc]]
        a, b = edges[int(rng.integers(len(edges)))]
        plan[plan == b] = a
    labels = sorted(set(map(int, plan.flat)) - {-1})
    result = np.full(plan.shape, -1, dtype=int)
    for i, old in enumerate(labels):
        result[plan == old] = i
    return result


def _geometry(rng, plan, lo, hi):
    heights = rng.integers(lo, hi + 1, size=plan.shape[0])
    widths = rng.integers(lo, hi + 1, size=plan.shape[1])
    ys = np.r_[0, np.cumsum(heights + 1)]
    xs = np.r_[0, np.cumsum(widths + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r, c in np.ndindex(plan.shape):
        label = int(plan[r, c])
        if label < 0:
            continue
        labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = label
        if r and plan[r - 1, c] == label:
            labels[ys[r], xs[c] + 1:xs[c + 1]] = label
        if c and plan[r, c - 1] == label:
            labels[ys[r] + 1:ys[r + 1], xs[c]] = label
        if r and c and np.all(plan[r - 1:r + 1, c - 1:c + 1] == label):
            labels[ys[r], xs[c]] = label
    for label in sorted(set(map(int, plan.flat)) - {-1}):
        pts = np.argwhere(labels == label)
        r, c, R, C = _bbox(pts)
        # Independent boundary displacement can reverse discovery order for
        # side-by-side rooms, without reversing their spatial relationships.
        if rng.random() < .6:
            edge = int(rng.integers(4))
            trial = labels == label
            if edge == 0 and R - r > 1:
                trial[r] = False
            elif edge == 1 and R - r > 1:
                trial[R - 1] = False
            elif edge == 2 and C - c > 1:
                trial[:, c] = False
            elif edge == 3 and C - c > 1:
                trial[:, C - 1] = False
            if np.any(trial) and len(_components(trial)) == 1:
                labels[(labels == label) & ~trial] = -1
            pts = np.argwhere(labels == label)
            r, c, R, C = _bbox(pts)
        # A local corner notch changes shape without changing spatial projections.
        if R - r > 1 and C - c > 1 and rng.random() < .35:
            p = ((r, c), (r, C - 1), (R - 1, c), (R - 1, C - 1))[int(rng.integers(4))]
            trial = labels == label
            trial[p] = False
            if len(_components(trial)) == 1 and _bbox(np.argwhere(trial)) == (r, c, R, C):
                labels[p] = -1
    inside = labels >= 0
    near = np.zeros(labels.shape, dtype=bool)
    padded = np.pad(inside, 1)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]]
    labels[near & ~inside] = -2
    r, c, R, C = _bbox(np.argwhere(near))
    return labels[r:R, c:C]


def _pack(rng, d, t, bg, wall):
    dh, dw = d.shape
    th, tw = t.shape
    layouts = []
    # Offset rows/columns preserve common local orientation while changing placement.
    if dw + tw + 3 <= 30:
        for offset in range(-4, 5):
            layouts.append(((max(0, -offset), 1), (max(0, offset), dw + 3),
                            max(dh + max(0, -offset), th + max(0, offset)), dw + tw + 3))
    if dh + th + 3 <= 30:
        for offset in range(-4, 5):
            layouts.append(((1, max(0, -offset)), (dh + 3, max(0, offset)),
                            dh + th + 3, max(dw + max(0, -offset), tw + max(0, offset))))
    layouts = [x for x in layouts if x[2] <= 28 and x[3] <= 28]
    if not layouts:
        raise ValueError("frameworks do not fit with optical space")
    dp, tp, used_h, used_w = layouts[int(rng.integers(len(layouts)))]
    h = int(rng.integers(max(used_h + 1, 22), 31))
    w = int(rng.integers(max(used_w + 1, 22), 31))
    top = int(rng.integers(h - used_h))
    left = int(rng.integers(w - used_w))
    dp, tp = (dp[0] + top, dp[1] + left), (tp[0] + top, tp[1] + left)
    inp = np.full((h, w), bg, dtype=int)
    figure = np.zeros(inp.shape, dtype=bool)
    all_rooms = []
    for geom, p in ((d, dp), (t, tp)):
        r, c = p
        patch = inp[r:r + geom.shape[0], c:c + geom.shape[1]]
        patch[geom == -2] = wall
        figure[r:r + geom.shape[0], c:c + geom.shape[1]] |= geom != -1
        all_rooms.append([np.argwhere(geom == i) + p for i in range(int(geom.max()) + 1)])
    return inp, figure, all_rooms[0], all_rooms[1], tp


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r - dr, c), (r, c - dc)}


def _bar_cells(midpoint, axis, length):
    r, c = midpoint
    return {(r, c + i) if axis == 0 else (r + i, c)
            for i in range(-(length // 2), length // 2 + 1)}


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _free(cells, forbidden, sources, mirrors, shape):
    if not _inside(cells, shape) or any(forbidden[p] for p in cells):
        return False
    occupied = {p for obj in sources + mirrors for p in obj['cells']}
    return not any((r + dr, c + dc) in occupied for r, c in cells
                   for dr, dc in ((0, 0),) + DIAG)


def _optics(rng, shape, figure, rooms, palette, axes, source_count):
    padded = np.pad(figure, 1)
    forbidden = np.zeros(shape, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            forbidden |= padded[dr:dr + shape[0], dc:dc + shape[1]]
    sources, mirrors = [], []
    room_index = _room_index(rooms)
    # Vectorize the empty-scene L placement check. argwhere retains the same
    # row/column/direction order as the original candidate enumeration.
    h, w = shape
    clear = np.pad(~forbidden, 1, constant_values=False)
    legal = np.zeros((h, w, len(DIRECTIONS)), dtype=bool)
    for i, (dr, dc) in enumerate(DIRECTIONS):
        legal[:, :, i] = (~forbidden & clear[1 - dr:h + 1 - dr, 1:w + 1]
                          & clear[1:h + 1, 1 - dc:w + 1 - dc])
    candidates = [((r, c), DIRECTIONS[i]) for r, c, i in np.argwhere(legal).tolist()]
    rng.shuffle(candidates)
    # Choose launches that actually interrogate the donor, not empty decorative rays.
    for elbow, direction in candidates[:800]:
        source = dict(cells=_source_cells(elbow, direction), elbow=elbow,
                      direction=direction, color=int(rng.choice(palette)))
        light, routes = _trace(shape, [source], [])
        if sum(bool(x) for x in _room_light(rooms, light, room_index)) >= 2:
            sources.append(source)
            break
    if not sources:
        raise ValueError("no donor-crossing launch")
    for wanted_axis in axes:
        light, routes = _trace(shape, sources, mirrors)
        route = routes[0]
        anchor = route['contacts'][-1][2] if route['contacts'] else sources[0]['elbow']
        dr, dc = route['direction']
        options = []
        for dist in range(2, 31):
            contact = anchor[0] + dist * dr, anchor[1] + dist * dc
            if not _inside({contact}, shape):
                break
            # Internal request 2 asks for a point, whose contacted face is
            # inferred from visible geometry rather than an invisible axis.
            axis_options = (wanted_axis,) if wanted_axis in (0, 1) else (0, 1)
            for axis in axis_options:
                mid = contact[0] + (dr if axis == 0 else 0), contact[1] + (dc if axis == 1 else 0)
                for length in ((1,) if wanted_axis == 2 else (1, 3, 5, 7)):
                    if wanted_axis in (0, 1) and length == 1:
                        continue
                    cells = _bar_cells(mid, axis, length)
                    if _free(cells, forbidden, sources, mirrors, shape):
                        options.append((mid, axis, length, cells))
        rng.shuffle(options)
        viable = []
        for mid, axis, length, cells in options[:180]:
            mirror = dict(cells=cells, midpoint=mid, axis=None if length == 1 else axis,
                          color=int(rng.choice(palette)))
            try:
                after, rs = _trace(shape, sources, mirrors + [mirror])
            except ValueError:
                continue
            contacts = rs[0]['contacts']
            if (len(contacts) != len(route['contacts']) + 1
                    or contacts[:-1] != route['contacts']
                    or contacts[-1][0] != len(mirrors)):
                continue
            seen = _room_light(rooms, after, room_index)
            score = sum(bool(x) for x in seen)
            if score < len(rooms):
                viable.append((score, mirror))
        if not viable:
            raise ValueError("no legal extension")
        best = max(score for score, _ in viable)
        choices = [mirror for score, mirror in viable if score == best]
        mirrors.append(choices[int(rng.integers(len(choices)))])
    for _ in range(source_count - 1):
        rng.shuffle(candidates)
        for elbow, direction in candidates[:800]:
            cells = _source_cells(elbow, direction)
            if not _free(cells, forbidden, sources, mirrors, shape):
                continue
            source = dict(cells=cells, elbow=elbow, direction=direction,
                          color=int(rng.choice(palette)))
            try:
                light, rs = _trace(shape, sources + [source], mirrors)
            except ValueError:
                continue
            seen = _room_light(rooms, light, room_index)
            if 3 <= sum(bool(x) for x in seen) < len(rooms):
                # The individual new source also has to reach some room.
                new_light = {p: {c} for p, c in rs[-1]['path']}
                if any(_room_light(rooms, new_light, room_index)):
                    sources.append(source)
                    break
        else:
            raise ValueError("no additional useful launch")
    light, routes = _trace(shape, sources, mirrors)
    return sources, mirrors, light, routes


def _render_optics(inp, sources, mirrors):
    for obj in sources + mirrors:
        for p in obj['cells']:
            inp[p] = obj['color']


def _pair(rng, count, axes, source_count, require_scan=False, excluded_roles=None):
    errors = Counter()
    for attempt in range(96):
        try:
            palette = list(map(int, rng.permutation(10)))
            if excluded_roles is not None:
                background = next(c for c in palette if c != excluded_roles[0])
                framework = next(c for c in palette if c not in (background, excluded_roles[1]))
                palette = [background, framework] + [c for c in palette if c not in (background, framework)]
            bg, wall = palette[:2]
            optical = palette[2:2 + int(rng.integers(2, 6))]
            plan = _plan(rng, count)
            d = _geometry(rng, plan, 2, 3)
            t = _geometry(rng, plan, 1, 3)
            inp, figure, dr, tr, tp = _pack(rng, d, t, bg, wall)
            if len(_mappings(dr, tr)) != 1:
                raise ValueError("ambiguous plan")
            sources, mirrors, light, routes = _optics(rng, inp.shape, figure, dr, optical, axes, source_count)
            seen = _room_light(dr, light)
            hit = [i for i, colors in enumerate(seen) if colors]
            unhit = [i for i, colors in enumerate(seen) if not colors]
            if len(hit) < 3 or not unhit:
                raise ValueError("insufficient exposure contrasts")
            beam_colors = set().union(*light.values())
            wrong = [i for i in hit if beam_colors - seen[i]]
            if not wrong:
                raise ValueError("no wrong-color crossing")
            witnesses = None
            if len(sources) == 2:
                individual = [_room_light(dr, _trace(inp.shape, [source], mirrors)[0])
                              for source in sources]
                first = [(i, c) for i in hit for c in sorted(individual[0][i] - individual[1][i])]
                second = [(i, c) for i in hit for c in sorted(individual[1][i] - individual[0][i])]
                options = [(i, a, j, b, w) for i, a in first for j, b in second
                           if i != j and a != b for w in wrong if w not in (i, j)]
                if not options:
                    raise ValueError("no distinct exposure witness for each emitter")
                i, a, j, b, wrong_room = options[int(rng.integers(len(options)))]
                witnesses = {i: a, j: b}
            else:
                wrong_room = int(rng.choice(wrong))
            activated = [i for i in hit if i != wrong_room]
            rng.shuffle(activated)
            activated = activated[:int(rng.integers(2, len(activated) + 1))]
            activated += [i for i in hit if i not in activated and not set(optical) - seen[i]]
            if witnesses:
                activated += [i for i in witnesses if i not in activated]
            colors = [int(rng.choice(optical)) for _ in dr]
            for i in hit:
                colors[i] = int(rng.choice(sorted(seen[i] if i in activated else set(optical) - seen[i])))
            if witnesses:
                for i, color in witnesses.items():
                    colors[i] = color
            colors[wrong_room] = int(rng.choice(sorted(beam_colors - seen[wrong_room])))
            colors[unhit[0]] = colors[activated[0]]
            if len({colors[i] for i in activated}) < 2:
                raise ValueError("need two transferred colors")
            expected = np.full(t.shape, bg, dtype=int)
            expected[t == -2] = wall
            for i, room in enumerate(dr):
                inp[tuple(room.T)] = colors[i]
                if i in activated:
                    expected[t == i] = colors[i]
            _render_optics(inp, sources, mirrors)
            predicted, info = _analyze(inp)
            if not np.array_equal(predicted, expected):
                raise ValueError("constructed correspondence mismatch")
            scan_answer = info['target']['crop'].copy()
            for i, room in enumerate(info['target']['local_rooms']):
                if info['exposed'][i]:
                    scan_answer[tuple(room.T)] = info['donor']['colors'][i]
            if require_scan and np.array_equal(scan_answer, predicted):
                raise ValueError("scan-order shortcut still works")
            # Actual answers must rely on routing and the adopted mirror colors.
            direct, _ = _trace(inp.shape, sources, [])
            decoded_rooms = info['donor']['rooms']
            decoded_colors = info['donor']['colors']
            direct_active = [c in shades for c, shades in zip(decoded_colors, _room_light(decoded_rooms, direct))]
            unchanged, _ = _trace(inp.shape, sources, mirrors, recolor=False)
            unchanged_active = [c in shades for c, shades in zip(decoded_colors, _room_light(decoded_rooms, unchanged))]
            if direct_active == info['exposed'] or unchanged_active == info['exposed']:
                raise ValueError("reflection or chroma ablation has no effect")
            return dict(input=inp, output=expected)
        except ValueError as error:
            errors[str(error)] += 1
    # A procedural six-room construction guarantees completion if an unusual
    # requested combination rejects every randomized candidate.
    return _fallback(rng, len(axes), source_count, axes[0], excluded_roles)


def _fallback(rng, mirror_count, source_count, first_axis=None, excluded_roles=None):
    """Analytic geometry, not a saved input/output pair or solver fixture."""
    inp = np.zeros((30, 30), dtype=int)
    d = np.full((9, 13), 4, dtype=int)
    t = np.full((7, 10), 4, dtype=int)
    dr, tr = [], []
    for row in range(2):
        for col in range(3):
            i = row * 3 + col
            local_d = np.array([(r, c) for r in range(1 + 4 * row, 4 + 4 * row)
                                for c in range(1 + 4 * col, 4 + 4 * col)])
            local_t = np.array([(r, c) for r in range(1 + 3 * row, 3 + 3 * row)
                                for c in range(1 + 3 * col, 3 + 3 * col)
                                if not (i == 0 and c == 1) and not (i == 1 and r == 1)])
            dr.append(local_d + (6, 8))
            tr.append(local_t)
            t[tuple(local_t.T)] = 0
    sources = [dict(cells=_source_cells((4, 4), (1, 1)), elbow=(4, 4),
                    direction=(1, 1), color=1)]
    if source_count == 2:
        sources.append(dict(cells=_source_cells((1, 6), (1, 1)), elbow=(1, 6),
                            direction=(1, 1), color=3))
    configs = [((17, 16), 0, 2), ((8, 25), 1, 3), ((2, 19), 0, 2)]
    mirrors = [dict(cells=_bar_cells(mid, axis, 1 if i == 1 else 3),
                    midpoint=mid, axis=None if i == 1 else axis, color=color)
               for i, (mid, axis, color) in enumerate(configs[:mirror_count])]
    colors = [1, (3 if source_count == 2 else 2), 1, 2,
              (2 if mirror_count == 3 else 1), 2]
    for i, room in enumerate(dr):
        d[tuple((room - (6, 8)).T)] = colors[i]
    inp[6:15, 8:21] = d
    inp[21:28, 1:11] = t
    _render_optics(inp, sources, mirrors)
    light, _ = _trace(inp.shape, sources, mirrors)
    exposed = [color in shades for color, shades in zip(colors, _room_light(dr, light))]
    expected = t.copy()
    for i, room in enumerate(tr):
        if exposed[i]:
            expected[tuple(room.T)] = colors[i]
    # The dual target boundary shifts defeat scan-order matching both before
    # and after transposition. The first mirror teaches the requested axis.
    axis = int(rng.integers(2)) if first_axis is None else first_axis
    if axis == 1:
        inp, expected = inp.T.copy(), expected.T.copy()
    palette = list(map(int, rng.permutation(10)))
    if excluded_roles is not None:
        bg = next(c for c in palette if c != excluded_roles[0])
        wall = next(c for c in palette if c not in (bg, excluded_roles[1]))
        palette = [bg, wall] + [c for c in palette if c not in (bg, wall)]
    original = [0, 4, 1, 2, 3, 5, 6, 7, 8, 9]
    recolor = np.empty(10, dtype=int)
    recolor[original] = palette
    return dict(input=recolor[inp], output=recolor[expected])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    first_axis = int(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    seen = set()
    first_roles = None
    for split, amount in (('train', num_train), ('test', num_test)):
        for index in range(amount):
            if split == 'train' and index == 0:
                count, axes, nsource, scan = int(rng.integers(5, 7)), [first_axis], 1, False
            elif split == 'train' and index == 1:
                count, axes, nsource, scan = int(rng.integers(6, 9)), [1 - first_axis, 2], 2, True
            elif split == 'test':
                count, axes, nsource, scan = int(rng.integers(5, 9)), [None, None, None], 2, False
            else:
                count, axes, nsource, scan = int(rng.integers(4, 9)), [None] * int(rng.integers(1, 3)), int(rng.integers(1, 3)), False
            for _ in range(4):
                pair = _pair(rng, count, axes, nsource, scan,
                             first_roles if split == 'train' and index == 1 else None)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                pair = _unique_recolor(pair, seen,
                                       first_roles if split == 'train' and index == 1 else None)
                key = (pair['input'].shape, pair['input'].tobytes())
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train' and index == 0:
                info = _parse(pair['input'])
                first_roles = info['background'], info['wall']
    return puzzle


def _unique_recolor(pair, seen, excluded_roles=None):
    info = _parse(pair['input'])
    old = [info['background'], info['wall']]
    old += [c for c in range(10) if c not in old]
    for bg in range(10):
        for wall in range(10):
            if bg == wall or (excluded_roles is not None and
                             (bg == excluded_roles[0] or wall == excluded_roles[1])):
                continue
            new = [bg, wall] + [c for c in range(10) if c not in (bg, wall)]
            permutation = np.empty(10, dtype=int)
            permutation[old] = new
            candidate = {k: permutation[v] for k, v in pair.items()}
            if (candidate['input'].shape, candidate['input'].tobytes()) not in seen:
                return candidate
    raise AssertionError('at most fourteen inputs cannot exhaust sixty-four role recolorings')


def _scan_answer(info):
    answer = info['target']['crop'].copy()
    for i, room in enumerate(info['target']['local_rooms']):
        if info['exposed'][i]:
            answer[tuple(room.T)] = info['donor']['colors'][i]
    return answer


def _measure_pair(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    predicted, info = _analyze(inp)
    assert np.array_equal(out, predicted), 'incorrect spectral transfer'
    target, donor = info['target'], info['donor']
    bg, wall = info['background'], info['wall']
    assert len(info['mappings']) == 1, 'generated correspondence must be unique'
    assert 2 <= sum(info['exposed']) <= len(donor['rooms']) - 2
    assert len({c for c, active in zip(donor['colors'], info['exposed']) if active}) >= 2
    assert any(shades and c not in shades for c, shades in zip(donor['colors'], info['seen'])), 'missing wrong-color crossing'
    assert any(not shades and c in {color for color, active in zip(donor['colors'], info['exposed']) if active}
               for c, shades in zip(donor['colors'], info['seen'])), 'missing unvisited same-color decoy'
    assert {i for route in info['routes'] for i, *_ in route['contacts']} == set(range(len(info['mirrors'])))
    interior = np.zeros(target['crop'].shape, dtype=bool)
    for room in target['local_rooms']:
        interior[tuple(room.T)] = True
    assert np.array_equal(out == wall, target['crop'] == wall), 'changed target walls'
    assert np.array_equal(out[~interior], target['crop'][~interior]), 'changed target exterior'
    assert set(map(int, out.flat)) <= set(map(int, inp.flat)), 'invented output color'
    mapping = info['mappings'][0]
    for i, j in mapping.items():
        values = np.unique(out[tuple(target['local_rooms'][j].T)])
        expected = donor['colors'][i] if info['exposed'][i] else bg
        assert len(values) == 1 and values[0] == expected, 'partial or wrong room fill'
    direct, _ = _trace(inp.shape, info['sources'], [])
    no_chroma, _ = _trace(inp.shape, info['sources'], info['mirrors'], recolor=False)
    direct_active = [c in colors for c, colors in zip(donor['colors'], _room_light(donor['rooms'], direct))]
    no_chroma_active = [c in colors for c, colors in zip(donor['colors'], _room_light(donor['rooms'], no_chroma))]
    assert direct_active != info['exposed'], 'mirror geometry does not affect answer'
    assert no_chroma_active != info['exposed'], 'mirror color does not affect answer'
    assert [bool(colors) for colors in info['seen']] != info['exposed'], 'color gate does not affect answer'
    if len(info['sources']) == 2:
        for omitted in range(2):
            light, _ = _trace(inp.shape, [source for i, source in enumerate(info['sources']) if i != omitted], info['mirrors'])
            active = [c in shades for c, shades in zip(donor['colors'], _room_light(donor['rooms'], light))]
            assert active != info['exposed'], 'an emitter has no independent effect on the answer'
    assert not np.array_equal(out, inp)
    area_changes = [int(np.sign(len(target['rooms'][j]) - len(donor['rooms'][i])))
                    for i, j in mapping.items()]
    return dict(background=bg, wall=wall, room_count=len(donor['rooms']),
                selected=sum(info['exposed']), source_count=len(info['sources']),
                mirror_count=len(info['mirrors']), input_shape=inp.shape,
                output_shape=out.shape, area_changes=tuple(area_changes),
                directions=tuple(source['direction'] for source in info['sources']),
                bar_axes=tuple(mirror['axis'] for mirror in info['mirrors'] if len(mirror['cells']) > 1),
                bar_lengths=tuple(len(mirror['cells']) for mirror in info['mirrors']),
                scan_failure=not np.array_equal(_scan_answer(info), out),
                nonrectangular=sum(len(room) != (_bbox(room)[2] - _bbox(room)[0]) *
                                   (_bbox(room)[3] - _bbox(room)[1])
                                   for room in donor['rooms'] + target['rooms']),
                frame_structure=(target['crop'].shape, (target['crop'] == wall).tobytes()))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = inp.shape, inp.tobytes()
            assert key not in seen, 'duplicate input'
            seen.add(key)
            records[split].append(_measure_pair(dict(input=inp, output=out)))
    train = records['train']
    assert {axis for record in train for axis in record['bar_axes']} == {0, 1}
    assert any(1 in record['bar_lengths'] for record in train), 'missing point-mirror demonstration'
    assert any(record['scan_failure'] for record in train), 'no correspondence counterexample'
    assert {record['source_count'] for record in train} >= {1, 2}
    assert {record['mirror_count'] for record in train} >= {1, 2}
    assert len({record['background'] for record in train}) >= 2
    assert len({record['wall'] for record in train}) >= 2
    assert all(record['mirror_count'] <= 2 for record in train)
    assert all(record['mirror_count'] == 3 for record in records['test'])
    return None


def _normalized_key(grid):
    roles, encoded = {}, []
    for value in grid.flat:
        value = int(value)
        encoded.append(roles.setdefault(value, len(roles)))
    return grid.shape, bytes(encoded)


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ('input', 'output'))
               for s in ('train', 'test'))


def _regressions():
    source = dict(cells=_source_cells((1, 1), (1, 1)), elbow=(1, 1), direction=(1, 1), color=2)
    mirror = dict(cells=_bar_cells((6, 5), 0, 3), midpoint=(6, 5), axis=0, color=3)
    light, _ = _trace((12, 12), [source], [mirror])
    expected = {(2, 2): {2}, (3, 3): {2}, (4, 4): {2}, (5, 5): {3},
                (4, 6): {3}, (3, 7): {3}, (2, 8): {3}, (1, 9): {3}, (0, 10): {3}}
    assert light == expected, 'contact color, reflection, or boundary regression'
    point = dict(cells={(6, 5)}, midpoint=(6, 5), axis=None, color=3)
    assert _trace((12, 12), [source], [point])[0] == expected
    transposed_source = dict(cells={(c, r) for r, c in source['cells']}, elbow=(1, 1), direction=(1, 1), color=2)
    transposed_mirror = dict(cells={(c, r) for r, c in mirror['cells']}, midpoint=(5, 6), axis=1, color=3)
    assert _trace((12, 12), [transposed_source], [transposed_mirror])[0] == {(c, r): colors for (r, c), colors in expected.items()}
    second = dict(cells=_source_cells((1, 9), (1, -1)), elbow=(1, 9), direction=(1, -1), color=4)
    light, _ = _trace((11, 11), [source, second], [])
    assert light[(5, 5)] == {2, 4}, 'multiple spectral exposures must be a union'
    assert light == _trace((11, 11), [second, source], [])[0], 'source order must not matter'
    for mirrors in (1, 2, 3):
        for sources in (1, 2):
            for axis in (0, 1):
                pair = _fallback(np.random.default_rng(100 + mirrors), mirrors, sources, axis)
                record = _measure_pair(pair)
                assert record['mirror_count'] == mirrors
                assert record['selected'] == (4 if sources == 2 or mirrors == 3 else 3)
                if sources == 2:
                    assert record['scan_failure']


def self_test():
    """Bounded 42-seed variation audit plus small explicit rule regressions."""
    _regressions()
    puzzles, metrics, structures, frame_structures = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                before = pair['input'].copy()
                prediction = solve(before)
                assert np.array_equal(before, pair['input']), 'solve mutated its input'
                assert np.array_equal(prediction, pair['output'])
                metric = _measure_pair(pair)
                metrics.append(metric)
                structures.add(_normalized_key(pair['input']))
                frame_structures.add(metric['frame_structure'])
    assert len({tuple(_normalized_key(pair['input']) for s in ('train', 'test')
                      for pair in puzzle[s]) for puzzle in puzzles}) == 42
    assert len(structures) >= 150
    assert len(frame_structures) >= 100
    assert len({m['input_shape'] for m in metrics}) >= 40
    assert len({m['output_shape'] for m in metrics}) >= 30
    assert len({m['room_count'] for m in metrics}) >= 4
    assert {m['background'] for m in metrics} == set(range(10))
    assert {m['wall'] for m in metrics} == set(range(10))
    assert {d for m in metrics for d in m['directions']} == set(DIRECTIONS)
    assert {axis for m in metrics for axis in m['bar_axes']} == {0, 1}
    assert {length for m in metrics for length in m['bar_lengths']} >= {1, 3, 5}
    assert {change for m in metrics for change in m['area_changes']} == {-1, 0, 1}
    assert sum(m['nonrectangular'] > 0 for m in metrics) >= 25
    assert _equal(puzzles[0], generate_puzzle(0))
    for counts in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(123, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, pairs=len(metrics), normalized_inputs=len(structures),
                target_frameworks=len(frame_structures),
                input_shapes=len({m['input_shape'] for m in metrics}),
                output_shapes=len({m['output_shape'] for m in metrics}),
                room_counts=sorted({m['room_count'] for m in metrics}),
                selected_counts=sorted({m['selected'] for m in metrics}),
                mirror_lengths=sorted({length for m in metrics for length in m['bar_lengths']}),
                scan_order_counterexamples=sum(m['scan_failure'] for m in metrics),
                nonrectangular_pairs=sum(m['nonrectangular'] > 0 for m in metrics),
                mixed_growth_and_shrinkage=sum(-1 in m['area_changes'] and 1 in m['area_changes'] for m in metrics))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
