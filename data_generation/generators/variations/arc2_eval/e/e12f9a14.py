"""Incomplete-ring launchers and spatially interacting colored strands.

Cardinal/diagonal collisions use a two-step staircase. Encounters are
spatial, not synchronized by distance from the launcher. Joined strands
subsequently turn as a bundle, preserving their offsets and colors.
"""

import hashlib
import numpy as np


_PORT_OFFSETS = {
    "NW": (-1, -1),
    "N0": (-1, 0),
    "N1": (-1, 1),
    "NE": (-1, 2),
    "W0": (0, -1),
    "E0": (0, 2),
    "W1": (1, -1),
    "E1": (1, 2),
    "SW": (2, -1),
    "S0": (2, 0),
    "S1": (2, 1),
    "SE": (2, 2),
}

_ALL_OFFSETS = tuple(_PORT_OFFSETS.values())

_CARDINAL_NAMES = (
    "N0", "N1", "S0", "S1",
    "W0", "W1", "E0", "E1",
)

_DIAGONAL_NAMES = ("NW", "NE", "SW", "SE")

_TRAIN_SIZES = (
    9, 10, 11, 13, 14, 15, 16,
    17, 18, 19, 21, 22, 23,
)

_SOURCE_TRAIN_SHAPES = {
    (24, 24),
    (8, 8),
    (20, 20),
    (12, 12),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _inside(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _direction_from_offset(
    offset: tuple[int, int],
) -> tuple[int, int]:
    dr, dc = offset

    if dr == -1:
        out_r = -1
    elif dr == 2:
        out_r = 1
    else:
        out_r = 0

    if dc == -1:
        out_c = -1
    elif dc == 2:
        out_c = 1
    else:
        out_c = 0

    return out_r, out_c


def _build_input(
    height: int,
    width: int,
    background: int,
    structure: int,
    specs: list[dict],
) -> np.ndarray:
    _require(1 <= height <= 30, "invalid height")
    _require(1 <= width <= 30, "invalid width")
    _require(background != structure, "background and structure must differ")

    occupied_footprints: set[tuple[int, int]] = set()
    normalized = []

    for spec in specs:
        core_r, core_c = spec["core"]
        payload = int(spec["color"])
        port_names = tuple(spec["ports"])

        _require(
            payload not in (background, structure),
            "payload collides with a semantic role",
        )
        _require(
            0 <= core_r < height - 1
            and 0 <= core_c < width - 1,
            "payload core is outside the grid",
        )
        _require(
            len(port_names) >= 1
            and len(port_names) == len(set(port_names)),
            "launcher has missing or duplicate ports",
        )
        _require(
            all(name in _PORT_OFFSETS for name in port_names),
            "unknown port name",
        )

        footprint = {
            (r, c)
            for r in range(core_r - 1, core_r + 3)
            for c in range(core_c - 1, core_c + 3)
            if _inside(r, c, height, width)
        }

        _require(
            not (occupied_footprints & footprint),
            "launcher footprints overlap",
        )
        occupied_footprints |= footprint

        ports = set()
        for name in port_names:
            dr, dc = _PORT_OFFSETS[name]
            position = (core_r + dr, core_c + dc)
            _require(
                _inside(position[0], position[1], height, width),
                "a selected port is clipped outside the grid",
            )
            ports.add(position)

        ring_in_grid = [
            (core_r + dr, core_c + dc)
            for dr, dc in _ALL_OFFSETS
            if _inside(core_r + dr, core_c + dc, height, width)
        ]

        _require(
            len(ring_in_grid) - len(ports) >= 3,
            "too little structural ring remains",
        )

        normalized.append((core_r, core_c, payload, ports))

    grid = np.full(
        (height, width),
        int(background),
        dtype=np.int8,
    )

    for core_r, core_c, payload, ports in normalized:
        grid[core_r:core_r + 2, core_c:core_c + 2] = payload

        for dr, dc in _ALL_OFFSETS:
            r = core_r + dr
            c = core_c + dc
            if (
                _inside(r, c, height, width)
                and (r, c) not in ports
            ):
                grid[r, c] = structure

    return grid


def _analyze_input(grid: np.ndarray) -> dict:
    _require(
        isinstance(grid, np.ndarray) and grid.ndim == 2,
        "input must be a two-dimensional NumPy array",
    )

    height, width = grid.shape
    counts = np.bincount(
        grid.astype(np.int64, copy=False).ravel(),
        minlength=10,
    )
    background = int(np.argmax(counts))

    by_structure: dict[int, list[dict]] = {}

    for core_r in range(height - 1):
        for core_c in range(width - 1):
            payload = int(grid[core_r, core_c])

            if payload == background:
                continue

            block = grid[
                core_r:core_r + 2,
                core_c:core_c + 2,
            ]
            if not np.all(block == payload):
                continue

            ring = []
            for dr, dc in _ALL_OFFSETS:
                r = core_r + dr
                c = core_c + dc
                if _inside(r, c, height, width):
                    ring.append(
                        (r, c, int(grid[r, c]), (dr, dc))
                    )

            nonbackground = [
                value
                for _, _, value, _ in ring
                if value != background
            ]
            background_count = len(ring) - len(nonbackground)
            distinct = set(nonbackground)

            if (
                len(distinct) != 1
                or background_count < 1
                or len(nonbackground) < 3
            ):
                continue

            structure = next(iter(distinct))
            if structure == payload:
                continue

            candidate = {
                "core": (core_r, core_c),
                "color": payload,
                "ring": ring,
                "structure_count": len(nonbackground),
            }
            by_structure.setdefault(structure, []).append(candidate)

    _require(bool(by_structure), "no launcher vocabulary found")

    structure = max(
        by_structure,
        key=lambda color: (
            len(by_structure[color]),
            sum(
                candidate["structure_count"]
                for candidate in by_structure[color]
            ),
            -color,
        ),
    )

    launchers = sorted(
        by_structure[structure],
        key=lambda launcher: launcher["core"],
    )
    _require(len(launchers) >= 1, "no launchers found")

    occupied_footprints: set[tuple[int, int]] = set()
    covered_nonbackground: set[tuple[int, int]] = set()
    port_directions = []
    clipped_count = 0

    for launcher in launchers:
        core_r, core_c = launcher["core"]

        footprint = {
            (r, c)
            for r in range(core_r - 1, core_r + 3)
            for c in range(core_c - 1, core_c + 3)
            if _inside(r, c, height, width)
        }

        _require(
            not (occupied_footprints & footprint),
            "inferred launcher footprints overlap",
        )
        occupied_footprints |= footprint

        if len(footprint) < 16:
            clipped_count += 1

        ports = []
        for r, c, value, offset in launcher["ring"]:
            if value == background:
                direction = _direction_from_offset(offset)
                ports.append(((r, c), direction, offset))
                port_directions.append(direction)
            else:
                _require(
                    value == structure,
                    "launcher has a mixed structural ring",
                )
                covered_nonbackground.add((r, c))

        _require(len(ports) >= 1, "launcher has no port")
        launcher["ports"] = ports

        for r in range(core_r, core_r + 2):
            for c in range(core_c, core_c + 2):
                covered_nonbackground.add((r, c))

    actual_nonbackground = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if int(grid[r, c]) != background
    }

    _require(
        actual_nonbackground == covered_nonbackground,
        "input contains noise or unaccounted objects",
    )

    return {
        "height": height,
        "width": width,
        "background": background,
        "structure": int(structure),
        "launchers": launchers,
        "port_directions": port_directions,
        "clipped_count": clipped_count,
    }



def _validate_count_value(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None

    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
    ):
        raise ValueError(
            f"{name} must be an integer or None"
        )

    result = int(value)

    if not lower <= result <= upper:
        raise ValueError(
            f"{name} is outside its legal range"
        )

    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_count_value(
        "num_train",
        num_train,
        2,
        10,
    )
    num_test = _validate_count_value(
        "num_test",
        num_test,
        1,
        4,
    )

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))

    elif num_train is None:
        lower = max(2, num_test + 1)
        num_train = int(rng.integers(lower, 11))

    elif num_test is None:
        upper = min(4, num_train - 1)

        if upper < 1:
            raise ValueError(
                "no legal test count exists"
            )

        num_test = int(rng.integers(1, upper + 1))

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test



_SOURCE_INPUTS = (((30, 30), 'c2639e56671b9083b57b9a4a21fb98b85b973ac4b40fb88fa2c429bf28e291e1'), ((30, 30), '0ef3140a9ac46006893dc741689ecdeb5da84cd362bcaf2af0de8747d4ac9aa1'), ((24, 24), '5cfa8d979374067c1729b98c3d311ba31946a6e82d5a1ef7a641cd239740b8b6'), ((8, 8), 'f7b0e8591cba57bf629dc8a2a370bfba9664fd6ea3ac05c082f32cf73c08aafa'), ((20, 20), '3001cdb38d71f9292d019bfd84525b7118dec87fb4d092d0c0d59c1b91abcd40'), ((12, 12), '33bbccfc774aabf8c10740643630857c28ed4f6d68228e7a52e196f3e2743b7b'))


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _heading(a, b):
    """A bundle has one heading, irrespective of its strand count."""
    v = _add(a, b)
    _require(v != (0, 0), "head-on cancellation")
    divisor = int(np.gcd(abs(v[0]), abs(v[1])))
    return v[0] // divisor, v[1] // divisor


def _walk(start, velocity, shape):
    """Rasterize rational headings; the first step favors the major axis."""
    row, col = start
    vr, vc = velocity
    scale = max(abs(vr), abs(vc))
    points = []
    for tick in range(3 * max(shape) + 1):
        point = (row + (1 if vr >= 0 else -1) * (tick * abs(vr) // scale),
                 col + (1 if vc >= 0 else -1) * (tick * abs(vc) // scale))
        if not _inside(*point, *shape):
            break
        points.append(point)
    return points


def _initial_groups(analysis):
    groups = []
    owners = []
    for launcher_id, launcher in enumerate(analysis['launchers']):
        for point, direction, _ in launcher['ports']:
            strand_id = len(owners)
            owners.append((launcher_id, launcher['color']))
            groups.append({'v': direction, 'members': [(strand_id, point)], 'depth': 0})
    return groups, owners


def _pixel_encounter(paths_a, paths_b):
    """Find intersections of complete future paths, without a global clock."""
    hits = []
    for ia, a in enumerate(paths_a):
        lookup = {point: k for k, point in enumerate(a)}
        midpoints = {_add(p, q): k for k, (p, q) in enumerate(zip(a, a[1:]))
                     if p[0] != q[0] and p[1] != q[1]}
        for ib, b in enumerate(paths_b):
            for j, point in enumerate(b):
                if point in lookup:
                    i = lookup[point]
                    hits.append((i - 1, j - 1, ia, ib, 'cell', float(i), float(j)))
            for j, (p, q) in enumerate(zip(b, b[1:])):
                if p[0] == q[0] or p[1] == q[1]:
                    continue
                midpoint = _add(p, q)
                if midpoint in midpoints:
                    i = midpoints[midpoint]
                    # A shared segment is overlap, not an X crossing.
                    if {a[i], a[i+1]} == {p, q}:
                        continue
                    hits.append((i, j, ia, ib, 'between', i + .5, j + .5))
    if not hits:
        return None
    return min(hits, key=lambda h: (h[0] + h[1], max(h[0], h[1]), h[2], h[3]))


def _collision_kind(a, b):
    _require(max(map(abs, a)) == max(map(abs, b)) == 1,
             'an encounter with an established staircase has no source-derived law')
    na, nb = sum(map(abs, a)), sum(map(abs, b))
    if na == nb == 1:
        return 'perpendicular'
    if na == nb == 2:
        return 'opposing_diagonals'
    _require(a[0] * b[0] + a[1] * b[1] > 0,
             'obtuse cardinal/diagonal encounter is not demonstrated')
    return 'staircase'


def _simulate(grid, analysis):
    groups, owners = _initial_groups(analysis)
    output = grid.copy()
    occupied = {}
    last_painted = {}
    segments = set()
    events = []
    travelled = [0] * len(owners)
    background = analysis['background']

    def paint(strand, points):
        for index, point in enumerate(points):
            _require(int(grid[point]) == background, 'a strand overwrites a launcher')
            _require(point not in occupied or (index == 0 and occupied[point] == strand
                     and last_painted.get(strand) == point), 'strand overlap or self-intersection')
            occupied[point] = strand
            output[point] = owners[strand][1]
            last_painted[strand] = point
        for p, q in zip(points, points[1:]):
            if p[0] != q[0] and p[1] != q[1]:
                midpoint = _add(p, q)
                key = (midpoint, 1 if (q[0]-p[0])*(q[1]-p[1]) > 0 else -1)
                _require((midpoint, -key[1]) not in segments, 'unresolved crossing')
                segments.add(key)

    for _ in range(len(owners) + 1):
        forecasts = [[_walk(point, group['v'], grid.shape)
                      for _, point in group['members']] for group in groups]
        candidates = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                hit = _pixel_encounter(forecasts[i], forecasts[j])
                if hit is not None:
                    _require(groups[i]['v'] != groups[j]['v'], 'parallel overlap')
                    candidates.append((i, j, hit))
        if not candidates:
            for group, paths in zip(groups, forecasts):
                for (strand, _), path in zip(group['members'], paths):
                    paint(strand, path)
            break
        # Only mutual first encounters may be processed. Disjoint encounters commute.
        first = {}
        for i, j, hit in candidates:
            first[i] = min(first.get(i, 1000), hit[0])
            first[j] = min(first.get(j, 1000), hit[1])
        ready = [c for c in candidates if c[2][0] == first[c[0]] and c[2][1] == first[c[1]]]
        _require(bool(ready), 'cyclic encounter ordering')
        i, j, hit = min(ready, key=lambda c: (max(c[2][:2]), sum(c[2][:2]), c[0], c[1]))
        for k, l, other in candidates:
            if (k, l) == (i, j):
                continue
            _require(not ((k in (i, j) and other[0] == first[k]) or
                          (l in (i, j) and other[1] == first[l])), 'triple encounter')
        a, b = groups[i], groups[j]
        kind = _collision_kind(a['v'], b['v'])
        velocity = _heading(a['v'], b['v'])
        _require(min(hit[:2]) >= 0, 'collision at an existing strand origin')
        members = []
        distances = []
        for index, cut in ((i, hit[0]), (j, hit[1])):
            group = groups[index]
            for (strand, _), path in zip(group['members'], forecasts[index]):
                paint(strand, path[:cut + 1])
                if cut < len(path) - 1:
                    members.append((strand, path[cut]))
                travelled[strand] += min(cut, len(path) - 1)
            representative = hit[2] if index == i else hit[3]
            strand = group['members'][representative][0]
            distances.append(travelled[strand] + (1 if hit[4] == 'cell' else .5))
        _require(len(members) == len(a['members']) + len(b['members']),
                 'a bundle is partly clipped before its next encounter')
        depth = 1 + max(a['depth'], b['depth'])
        events.append({'kind': kind, 'intersection': hit[4],
                       'unequal': distances[0] != distances[1],
                       'distances': tuple(distances), 'depth': depth,
                       'incoming': (a['v'], b['v']), 'outgoing': velocity,
                       'strand_count': len(members),
                       'launchers': tuple(sorted({owners[s][0] for s, _ in members})),
                       'colors': tuple(owners[s][1] for s, _ in members),
                       'position': forecasts[i][hit[2]][hit[0]]})
        groups = [g for k, g in enumerate(groups) if k not in (i, j)]
        groups.append({'v': velocity, 'members': members, 'depth': depth})
    else:
        raise AssertionError('too many merges')
    info = dict(analysis)
    info.update(events=events, changed_cells=int(np.count_nonzero(output != grid)),
                collision_events=len(events),
                payload_colors=tuple(l['color'] for l in analysis['launchers']))
    return output, info


def _solve_grid(grid):
    return _simulate(grid, _analyze_input(grid))


def _analytic_predict(grid, weighted=False, raster='floor'):
    """Second predictor: continuous line intersections, not pixel/midpoint scans.

    It shares the launcher parser, but neither collision detection, scheduling,
    path rasterization nor output drawing with _simulate.
    """
    info = _analyze_input(grid)
    bundles = []
    for launcher in info['launchers']:
        for position, direction, _ in launcher['ports']:
            bundles.append((direction, [(position, launcher['color'])]))
    answer = grid.copy()

    def draw(point, direction, color, stop=None):
        r, c = point
        dr, dc = direction
        major = max(abs(dr), abs(dc))
        er = major-1 if raster == 'ceil' and dr else 0
        ec = major-1 if raster == 'ceil' and dc else 0
        last = point
        for k in range(100):
            if not _inside(r, c, *grid.shape) or (stop is not None and k > stop):
                break
            answer[r, c] = color
            last = (r, c)
            er += abs(dr)
            ec += abs(dc)
            if er >= major:
                r += 1 if dr > 0 else -1
                er -= major
            if ec >= major:
                c += 1 if dc > 0 else -1
                ec -= major
        return last

    for _ in range(100):
        possible = []
        nearest = [1000] * len(bundles)
        for i, (u, aa) in enumerate(bundles):
            for j in range(i + 1, len(bundles)):
                v, bb = bundles[j]
                determinant = u[0]*v[1] - u[1]*v[0]
                if not determinant:
                    continue
                # Staircases are terminal in the supported construction domain.
                if max(map(abs,u)) > 1 or max(map(abs,v)) > 1:
                    continue
                for p, _ in aa:
                    for q, _ in bb:
                        delta = (q[0]-p[0], q[1]-p[1])
                        t = (delta[0]*v[1] - delta[1]*v[0]) / determinant
                        s = (delta[0]*u[1] - delta[1]*u[0]) / determinant
                        cross = (p[0]+t*u[0], p[1]+t*u[1])
                        if t <= 0 or s <= 0 or not (0 <= cross[0] <= grid.shape[0]-1 and
                                                                   0 <= cross[1] <= grid.shape[1]-1):
                            continue
                        x, y = int(np.ceil(t))-1, int(np.ceil(s))-1
                        possible.append((i, j, x, y))
                        nearest[i] = min(nearest[i], x)
                        nearest[j] = min(nearest[j], y)
        if not possible:
            for direction, members in bundles:
                for point, color in members:
                    draw(point, direction, color)
            return answer
        ready = [e for e in possible if e[2] == nearest[e[0]] and e[3] == nearest[e[1]]]
        _require(bool(ready), 'independent predictor found cyclic intersections')
        i,j,x,y = min(ready, key=lambda e: (max(e[2:]),sum(e[2:]),e[0],e[1]))
        u, aa = bundles[i]
        v, bb = bundles[j]
        members = [(draw(p,u,c,x),c) for p,c in aa] + [(draw(p,v,c,y),c) for p,c in bb]
        wa,wb = (len(aa),len(bb)) if weighted else (1,1)
        vr, vc = wa*u[0]+wb*v[0], wa*u[1]+wb*v[1]
        divisor = int(np.gcd(abs(vr),abs(vc)))
        _require(divisor > 0, 'independent predictor found head-on cancellation')
        bundles = [b for k,b in enumerate(bundles) if k not in (i,j)]
        direction = (vr//divisor, vc//divisor)
        if raster == 'sign':
            direction = tuple(int(np.sign(x)) for x in direction)
        bundles.append((direction,members))
    raise AssertionError('independent predictor did not terminate')


def _spec_at_port(point, name, color):
    offset = _PORT_OFFSETS[name]
    return {'core': (point[0]-offset[0], point[1]-offset[1]),
            'color': color, 'ports': (name,)}


def _trial(shape, specs, allowed=None):
    grid = _build_input(*shape, 0, 8, specs)
    out, info = _solve_grid(grid)
    if allowed is not None:
        _require({e['kind'] for e in info['events']} <= set(allowed), 'unplanned collision mechanism')
    _require(np.array_equal(out, _analytic_predict(grid)), 'predictors disagree')
    return grid, out, info


def _primary(rng, shape, kind, unequal=False):
    h, w = shape
    last_error = None
    for attempt in range(160):
        row = int(rng.integers(2, max(3, h-5)))
        col = int(rng.integers(4, max(5, w-4)))
        a, b = (int(x) for x in rng.integers(2, 10, 2))
        if unequal and a == b:
            continue
        if kind == 'opposing_diagonals':
            # Half-cell intersections demonstrate two adjacent outgoing strands.
            points = ((row+a+1, col-a), (row+b+1, col+b+1))
            names = ('NE', 'NW')
        elif kind == 'perpendicular':
            points = ((row,col-a),(row+b,col))
            names = (str(rng.choice(('E0','E1'))), str(rng.choice(('N0','N1'))))
        else:
            points = ((row+a,col-a),(row+b,col))
            names = ('NE',str(rng.choice(('N0','N1'))))
        specs = [_spec_at_port(p,n,c) for p,n,c in zip(points,names,(1,2))]
        try:
            _,_,info = _trial(shape,specs,{kind})
            _require(len(info['events']) == 1,'primary encounter missing')
            event = info['events'][0]
            _require(event['position'][0] >= 3, 'too little visible outgoing path')
            if unequal:
                _require(event['unequal'],'unequal distances not visible')
            return specs
        except AssertionError as error:
            last_error = error
    raise AssertionError(f'primary construction exhausted: {last_error}')


def _transform_specs(specs, shape, rotation, reflection):
    grid = _build_input(*shape,0,8,specs)
    grid = np.rot90(grid,rotation)
    if reflection:
        grid = np.fliplr(grid)
    analysis = _analyze_input(grid)
    reverse = {v:k for k,v in _PORT_OFFSETS.items()}
    return [{'core': l['core'], 'color': l['color'],
             'ports': tuple(reverse[o] for _,_,o in l['ports'])}
            for l in analysis['launchers']], grid.shape


def _add_launcher(rng, shape, specs, allowed, interacting=False, later=False):
    """Attach to a sampled ray segment, or add a spatially independent emitter."""
    base, out, before = _trial(shape, specs, allowed)
    used_colors = {s['color'] for s in specs}
    color = next(c for c in (1,2,3,4,5,6,7,9) if c not in used_colors)
    target_cells = list(zip(*np.where(out != base)))
    last_depth = max((e['depth'] for e in before['events']), default=0)
    for _ in range(110):
        if interacting:
            r,c = target_cells[int(rng.integers(len(target_cells)))]
            name = str(rng.choice(tuple(_PORT_OFFSETS)))
            direction = _direction_from_offset(_PORT_OFFSETS[name])
            distance = int(rng.integers(2,11))
            # Also target segment midpoints for unequal-distance diagonal crossings.
            half = bool(rng.integers(2)) and direction[0] != 0 and direction[1] != 0
            if half:
                r, c = r + .5, c + .5
                distance += .5
            point = (r-distance*direction[0], c-distance*direction[1])
            if any(x != int(x) for x in point):
                continue
            new = _spec_at_port(tuple(int(x) for x in point),name,color)
        else:
            h,w = shape
            clipping = int(rng.choice(3,p=(.48,.34,.18)))
            r,c = int(rng.integers(h-1)),int(rng.integers(w-1))
            if clipping:
                if bool(rng.integers(2)):
                    r = int(rng.choice((0,h-2)))
                    if clipping == 2:
                        c = int(rng.choice((0,w-2)))
                else:
                    c = int(rng.choice((0,w-2)))
                    if clipping == 2:
                        r = int(rng.choice((0,h-2)))
            available = [n for n,(dr,dc) in _PORT_OFFSETS.items()
                         if _inside(r+dr,c+dc,h,w)]
            nports = min(len(available)-3,int(rng.integers(1,5)))
            ports = tuple(str(n) for n in rng.choice(available,nports,replace=False))
            new = {'core':(r,c),'color':color,'ports':ports}
        candidate = specs + [new]
        try:
            _,_,info = _trial(shape,candidate,allowed)
            if interacting:
                _require(len(info['events']) > len(before['events']),'new launcher does not interact')
                _require(any(color in e['colors'] for e in info['events']), 'attachment not involved')
                if later:
                    _require(max(e['depth'] for e in info['events']) > last_depth, 'not a later encounter')
            else:
                _require(len(info['events']) == len(before['events']),'isolated addition interacted')
            return candidate
        except AssertionError:
            continue
    return None


def _extra_ports(rng, shape, specs, allowed):
    result = [dict(s) for s in specs]
    for _ in range(int(rng.integers(2,10))):
        index = int(rng.integers(len(result)))
        spec = result[index]
        options = [p for p in _PORT_OFFSETS if p not in spec['ports']]
        if not options or len(spec['ports']) >= 7:
            continue
        candidate = [dict(s) for s in result]
        candidate[index]['ports'] = spec['ports'] + (str(rng.choice(options)),)
        try:
            _trial(shape,candidate,allowed)
            result = candidate
        except AssertionError:
            pass
    return result


def _independent(rng, shape, kinds):
    h,w = shape
    top = h//2
    first = _primary(rng,(top,w),str(rng.choice(kinds)),True)
    second = _primary(rng,(h-top,w),str(rng.choice(kinds)),True)
    # Face the second encounter toward the opposite edge. The gap between
    # modules is empty background, not a new kind of visible compartment.
    second,_ = _transform_specs(second,(h-top,w),2,bool(rng.integers(2)))
    for spec in second:
        spec['core'] = (spec['core'][0]+top,spec['core'][1])
        spec['color'] += 2
    combined = first+second
    _,_,info = _trial(shape,combined,kinds)
    _require(len(info['events']) == 2 and max(e['depth'] for e in info['events']) == 1,
             'independent encounters interfered')
    return combined


def _make_scene(rng, mode, kinds, shape_range, unequal=False):
    """Bounded randomized construction, followed by deterministic safe geometry."""
    low,high = shape_range
    for attempt in range(18):
        if mode == 'isolated':
            h,w = (int(x) for x in rng.integers(low,high+1,2))
            specs = []
            # Seed with one arbitrary emitter, then add only readable emissions.
            r,c = int(rng.integers(h-1)),int(rng.integers(w-1))
            if rng.random() < .55:
                r = int(rng.choice((0,h-2)))
            if rng.random() < .3:
                c = int(rng.choice((0,w-2)))
            available = [n for n,(dr,dc) in _PORT_OFFSETS.items() if _inside(r+dr,c+dc,h,w)]
            count = min(len(available)-3,int(rng.integers(2,7)))
            chosen = [str(rng.choice([n for n in available if n in _CARDINAL_NAMES])),
                      str(rng.choice([n for n in available if n in _DIAGONAL_NAMES]))]
            rest = [n for n in available if n not in chosen]
            chosen += [str(x) for x in rng.choice(rest,max(0,count-2),replace=False)]
            specs = [{'core':(r,c),'color':1,'ports':tuple(chosen)}]
            allowed = set()
        else:
            minsize = 24 if mode == 'independent' else (21 if mode == 'chain' else 13)
            cap = max(high,minsize)
            h = int(rng.integers(max(low,minsize),cap+1))
            w = int(rng.integers(max(14,low),cap+1))
            specs = None
            allowed = set(kinds)
        shape = (h,w)
        try:
            if mode == 'independent':
                specs = _independent(rng,shape,kinds)
            elif mode == 'chain':
                specs = _primary(rng,shape,'opposing_diagonals',True)
                attached = _add_launcher(rng,shape,specs,allowed,True,True)
                _require(attached is not None,'no readable chain attachment')
                specs = attached
            elif mode != 'isolated':
                specs = _primary(rng,shape,mode,unequal)
            _trial(shape,specs,allowed)
            target = int(rng.choice((len(specs), min(6,len(specs)+1), min(6,len(specs)+2)),
                                    p=(.46,.36,.18)))
            while len(specs) < target:
                attached = None
                if mode != 'isolated' and rng.random() < .7:
                    attached = _add_launcher(rng,shape,specs,allowed,True,mode=='chain')
                if attached is None:
                    attached = _add_launcher(rng,shape,specs,allowed,False)
                if attached is None:
                    break
                specs = attached
            specs = _extra_ports(rng,shape,specs,allowed)
            return _trial(shape,specs,allowed)
        except AssertionError:
            continue
    # Safe formulae are a bounded-construction fallback, not the sampling bank.
    # Every coordinate and port remains visible; no fallback changes the rule.
    shape = (24,24)
    if mode == 'isolated':
        specs = [{'core':(0,0),'color':1,'ports':('E0','SE')}]
    elif mode == 'perpendicular':
        specs = [{'core':(6,3),'color':1,'ports':('E0',)},
                 {'core':(13,12),'color':2,'ports':('N0',)}]
    elif mode == 'staircase':
        specs = [{'core':(12,3),'color':1,'ports':('NE',)},
                 {'core':(10,12),'color':2,'ports':('N0',)}]
    else:
        specs = [{'core':(15,5),'color':1,'ports':('NE',)},
                 {'core':(14,14),'color':2,'ports':('NW',)}]
        if mode == 'chain':
            specs.append({'core':(6,2),'color':3,'ports':('E0',)})
        elif mode == 'independent':
            shape = (30,30)
            kind = kinds[0]
            if kind == 'perpendicular':
                first = [{'core':(5,2),'color':1,'ports':('E0',)},
                         {'core':(10,11),'color':2,'ports':('N0',)}]
            elif kind == 'staircase':
                first = [{'core':(11,2),'color':1,'ports':('NE',)},
                         {'core':(9,11),'color':2,'ports':('N0',)}]
            else:
                first = [{'core':(9,3),'color':1,'ports':('NE',)},
                         {'core':(8,12),'color':2,'ports':('NW',)}]
            second,_ = _transform_specs(first,(15,30),2,False)
            for spec in second:
                spec['core'] = (spec['core'][0]+15,spec['core'][1])
                spec['color'] += 2
            specs = first+second
    return _trial(shape,specs,kinds)


def _recolor(rng, grid, output, roles, repeats):
    background, structure = roles
    available = [c for c in range(10) if c not in roles]
    rng.shuffle(available)
    info = _analyze_input(grid)
    payloads = sorted({l['color'] for l in info['launchers']})
    palette = {0:background,8:structure}
    for i,color in enumerate(payloads):
        palette[color] = available[i]
    # Repetition is confined to noninteracting launchers; collision examples
    # retain separate colors so the strand-preservation convention is visible.
    if repeats and len(payloads) > 1:
        _, metrics = _solve_grid(grid)
        involved = {c for event in metrics['events'] for c in event['colors']}
        free = [c for c in payloads if c not in involved]
        if len(free) >= 2:
            for color in free[1:]:
                palette[color] = palette[free[0]]
    lookup = np.arange(10,dtype=np.int8)
    for old,new in palette.items():
        lookup[old] = new
    return lookup[grid],lookup[output]


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _source_copy(grid):
    return (grid.shape,hashlib.sha256(grid.astype(np.int8).tobytes()).hexdigest()) in _SOURCE_INPUTS


def _coverage(info):
    tags = set()
    for event in info['events']:
        tags.add(event['kind'])
        tags.add(event['intersection'])
        if event['unequal']:
            tags.add('unequal')
            tags.add('unequal:' + event['kind'] + ':' + event['intersection'])
        if event['depth'] > 1:
            tags.add('chain')
            tags.add('bundle:' + event['kind'])
    return tags


def _fitted_variants(train):
    """Fit plausible alternatives using training labels only.

    Simple puzzles need not identify unused collision conventions. All rules
    still consistent with training must, however, agree on each test answer.
    """
    fitted = []
    for weighted in (False,True):
        for raster in ('floor','ceil','sign'):
            try:
                if all(np.array_equal(_analytic_predict(p['input'],weighted,raster),p['output'])
                       for p in train):
                    fitted.append((weighted,raster))
            except AssertionError:
                continue
    _require((False,'floor') in fitted,'training contradicts the source-derived convention')
    return fitted


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt,ne = _resolve_counts(seed,rng,num_train,num_test)
    vocabulary = ['opposing_diagonals','perpendicular','staircase']
    rng.shuffle(vocabulary)
    simple = rng.random() < .12
    kinds = [] if simple else vocabulary[:int(rng.integers(1,min(3,nt-1)+1))]
    # Curriculum and sampling style are puzzle-level. They are not hidden laws:
    # all emission and collision conventions are the same for every puzzle.
    larger_tests = rng.random() < .4
    shared_orientation = rng.random() < .25
    orientation = (int(rng.integers(4)),bool(rng.integers(2)))
    base_roles = tuple(int(x) for x in rng.choice(10,2,replace=False))
    train_modes = ['isolated'] + kinds
    while len(train_modes) < nt:
        options = ['isolated'] + kinds
        if {'opposing_diagonals','perpendicular'} <= set(kinds):
            options += ['chain']
        if kinds:
            options += ['independent']
        train_modes.append(str(rng.choice(options)))
    # Sometimes the final demonstration explicitly teaches repeated bending.
    if nt > len(kinds)+1 and {'opposing_diagonals','perpendicular'} <= set(kinds) and rng.random() < .7:
        train_modes[-1] = 'chain'
    rng.shuffle(train_modes)
    puzzle = {'train':[],'test':[]}
    seen = set()
    taught = set()
    for split,count in (('train',nt),('test',ne)):
        variants = _fitted_variants(puzzle['train']) if split == 'test' else []
        for index in range(count):
            if split == 'train':
                mode = train_modes[index]
            else:
                options = ['isolated'] + kinds*2
                if kinds:
                    options += ['independent']
                if 'chain' in taught and {'opposing_diagonals','perpendicular'} <= set(kinds):
                    options += ['chain']*2
                mode = str(rng.choice(options))
            for attempt in range(30):
                sizes = (8,24) if larger_tests and split == 'train' else (
                        (14,30) if larger_tests else (8,30))
                grid,out,_ = _make_scene(rng,mode,kinds,sizes,unequal=True)
                if split == 'train' and index == 0:
                    roles = base_roles
                elif split == 'train' and index == 1:
                    roles = tuple(int(x) for x in rng.choice(10,2,replace=False))
                    if roles == base_roles:
                        roles = roles[::-1]
                else:
                    roles = base_roles if rng.random() < .35 else tuple(int(x) for x in rng.choice(10,2,replace=False))
                grid,out = _recolor(rng,grid,out,roles,rng.random()<.65)
                rotation,reflection = orientation if shared_orientation else (
                    int(rng.integers(4)),bool(rng.integers(2)))
                grid,out = np.rot90(grid,rotation),np.rot90(out,rotation)
                if reflection:
                    grid,out = np.fliplr(grid),np.fliplr(out)
                grid,out = np.ascontiguousarray(grid),np.ascontiguousarray(out)
                if _grid_key(grid) in seen or _source_copy(grid):
                    continue
                # Certify the actual returned grids. Rotation/reflection changes
                # parser ordering and can change the representative contact of
                # a bundle encounter, even when its painted output is identical.
                # Pre-transform event metadata is therefore not training evidence.
                try:
                    expected,info = _solve_grid(grid)
                    _require(np.array_equal(expected,out),'transformed output changed the rule')
                    _require(np.array_equal(_analytic_predict(grid),out),
                             'transformed analytic prediction disagrees')
                    if split == 'test':
                        _require(_coverage(info) <= taught,'untaught final-grid encounter')
                        _require(all(np.array_equal(_analytic_predict(grid,*variant),out)
                                     for variant in variants),
                                 'unresolved final-grid test interpretation')
                except AssertionError:
                    if split == 'test' and attempt > 8:
                        mode = str(rng.choice(['isolated']+kinds))
                    continue
                pair = {'input':grid,'output':out}
                puzzle[split].append(pair)
                seen.add(_grid_key(grid))
                if split == 'train':
                    taught |= _coverage(info)
                break
            else:
                # Exhaustion of coverage rejection has a legal isolated fallback.
                # Index-dependent geometry guarantees uniqueness across 14 pairs.
                serial = len(puzzle['train'])+len(puzzle['test'])
                h,w = 16+serial,30
                grid = _build_input(h,w,0,8,[{'core':(2,serial+2),'color':1,
                                            'ports':('NW','N0','SE')}])
                out,_ = _solve_grid(grid)
                roles = base_roles[::-1] if index == 1 else base_roles
                grid,out = _recolor(rng,grid,out,roles,False)
                _require(_grid_key(grid) not in seen,'fallback duplicate')
                puzzle[split].append({'input':grid,'output':out})
                seen.add(_grid_key(grid))
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    _,info = _solve_grid(pair['input'])
    h,w = pair['input'].shape
    clipping = []
    margins = []
    for launcher in info['launchers']:
        r,c = launcher['core']
        edges = int(r==0 or r==h-2)+int(c==0 or c==w-2)
        clipping.append(('interior','edge','corner')[edges])
        margins.append(min(r,c,h-2-r,w-2-c))
    events = info['events']
    graph = [set(e['launchers']) for e in events]
    components = []
    for vertices in graph:
        merged = set(vertices)
        rest = []
        for component in components:
            if merged & component:
                merged |= component
            else:
                rest.append(component)
        components = rest+[merged]
    # Repeatedly merge components bridged by a later event.
    for _ in range(len(components)):
        for i in range(len(components)):
            for j in range(i):
                if components[i] & components[j]:
                    components[j] |= components[i]
                    components[i] = set()
    components = [c for c in components if c]
    payloads = info['payload_colors']
    return {'shape':(h,w),'launcher_count':len(info['launchers']),
            'roles':(info['background'],info['structure']),
            'clipping':tuple(clipping),'margins':tuple(margins),
            'port_counts':tuple(len(l['ports']) for l in info['launchers']),
            'port_sets':tuple(tuple(o for _,_,o in l['ports']) for l in info['launchers']),
            'directions':tuple(sorted(set(info['port_directions']))),
            'repeated_payloads':len(set(payloads)) < len(payloads),
            'events':len(events),'kinds':tuple(sorted({e['kind'] for e in events})),
            'intersections':tuple(e['intersection'] for e in events),
            'unequal':sum(e['unequal'] for e in events),
            'max_depth':max((e['depth'] for e in events),default=0),
            'max_bundle':max((e['strand_count'] for e in events),default=1),
            'components':tuple(sorted(len(c) for c in components)),
            'participating_launchers':len(set().union(*graph)) if graph else 0,
            'orientations':tuple(sorted({e['outgoing'] for e in events})),
            'collision_positions':tuple(e['position'] for e in events),
            'collision_distances':tuple(e['distances'] for e in events),
            'coverage':tuple(sorted(_coverage(info)))}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle,dict) and set(puzzle)=={'train','test'},'invalid puzzle keys')
    _require(all(isinstance(puzzle[s],list) for s in puzzle),'splits must be lists')
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    _require(2<=nt<=10 and 1<=ne<=4 and nt>ne,'illegal counts')
    inputs,pairs = set(),set()
    taught,needed = set(),set()
    train_roles = set()
    emission = set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair)=={'input','output'},'invalid pair keys')
            for grid in pair.values():
                _require(isinstance(grid,np.ndarray) and grid.ndim==2,'invalid grid type')
                _require(np.issubdtype(grid.dtype,np.integer),'noninteger grid')
                _require(all(1<=v<=30 for v in grid.shape),'invalid dimensions')
                _require(np.all((0<=grid)&(grid<=9)),'invalid colors')
            a,b = pair['input'],pair['output']
            _require(a.shape==b.shape,'dimensions changed')
            ik,pk = _grid_key(a),(_grid_key(a),_grid_key(b))
            _require(ik not in inputs and pk not in pairs,'duplicate pair')
            inputs.add(ik)
            pairs.add(pk)
            _require(not _source_copy(a),'source contents copied')
            out,info = _solve_grid(a)
            _require(np.array_equal(out,b),'incorrect output')
            _require(np.array_equal(_analytic_predict(a),b),'analytic predictor disagrees')
            _require(info['changed_cells']>0,'unchanged pair')
            _require(1<=len(info['launchers'])<=6,'launcher count out of range')
            _require(np.array_equal(a[a!=info['background']],b[a!=info['background']]),
                     'core or structural cells changed')
            for event in info['events']:
                _require(len(set(event['colors'])) >= 2,
                         'collision needs visibly distinct payload colors')
            if split=='train':
                taught |= _coverage(info)
                train_roles.add((info['background'],info['structure']))
                emission |= {sum(map(abs,d)) for d in info['port_directions']}
            else:
                needed |= _coverage(info)
    _require(needed<=taught,'test collision mechanism is not demonstrated in training')
    _require(emission=={1,2},'training must demonstrate side and corner emission')
    _require(len(train_roles)>=2,'visible color roles must vary within training')
    for variant in _fitted_variants(puzzle['train']):
        for pair in puzzle['test']:
            _require(np.array_equal(_analytic_predict(pair['input'],*variant),pair['output']),
                     'training leaves competing test predictions unresolved')


def _puzzles_equal(a,b):
    return all(len(a[s])==len(b[s]) and all(np.array_equal(x[k],y[k])
               for x,y in zip(a[s],b[s]) for k in ('input','output')) for s in ('train','test'))


def self_test() -> None:
    _source_regressions()
    # Host counterexample: pre-transform training metadata advertised a
    # shared-cell encounter absent from the actual transformed demonstrations.
    coverage_regression = generate_puzzle(96002200507320448)
    _require((len(coverage_regression['train']),len(coverage_regression['test'])) == (7,4),
             'host regression counts changed')
    returned_coverage = {
        split: set().union(*(_coverage(_solve_grid(pair['input'])[1])
                            for pair in coverage_regression[split]))
        for split in ('train','test')
    }
    _require(returned_coverage['test'] <= returned_coverage['train'],
             'host regression has an untaught transformed encounter')
    _require(_puzzles_equal(coverage_regression,generate_puzzle(96002200507320448)),
             'host regression is not deterministic')
    features = []
    train_counts,test_counts = set(),set()
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        local = [_features(p) for split in puzzle.values() for p in split]
        _require(len({f['roles'] for f in local})>=2,'within-puzzle role diversity collapsed')
        features += local
    _require(len(train_counts)>=4 and len(test_counts)>=3,'automatic counts collapsed')
    _require(_puzzles_equal(generate_puzzle(17),puzzles[17]),'automatic nondeterminism')
    _require(not _puzzles_equal(puzzles[0],puzzles[1]),'cross-seed identity')
    explicit = generate_puzzle(45,10,4)
    _require(_puzzles_equal(explicit,generate_puzzle(45,10,4)),'explicit nondeterminism')
    for train,test in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        puzzle = generate_puzzle(77,train,test)
        _require(train is None or len(puzzle['train'])==train,'train count ignored')
        _require(test is None or len(puzzle['test'])==test,'test count ignored')
        validate_generated_puzzle(puzzle)
    for train,test in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(True,1),(2.0,1)):
        try:
            generate_puzzle(90,train,test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    _require(len({f['shape'] for f in features})>=40,'size diversity collapsed')
    _require({1,2,3,4,5,6}<={f['launcher_count'] for f in features},'launcher count diversity')
    _require({'interior','edge','corner'}<=set().union(*(set(f['clipping']) for f in features)),
             'clipping modes missing')
    _require(any(f['repeated_payloads'] for f in features),'repeated payloads missing')
    _require(set().union(*(set(f['kinds']) for f in features))==
             {'opposing_diagonals','perpendicular','staircase'},'collision mechanism missing')
    _require(any(f['max_depth']>=2 for f in features),'no chains')
    _require(any(len(f['components'])>=2 for f in features),'no independent encounters')
    _require(any(f['participating_launchers']>=4 for f in features),'extra launchers do not participate')
    _require(any(f['unequal'] for f in features),'unequal travel distances missing')
    _require(len(set().union(*(set(f['orientations']) for f in features)))>=12,'orientation collapse')
    _require({8,12,20,24,30}<=set(x for f in features for x in f['shape']), 'legal source dimensions excluded')
    _require(max(max(f['port_counts']) for f in features)>=6,'port count range collapsed')
    _require(len({f['components'] for f in features})>=6,'network topology collapsed')
    # Verify rejection of a valid-looking but semantically corrupted label.
    corrupt = {s:[{k:v.copy() for k,v in pair.items()} for pair in explicit[s]] for s in explicit}
    corrupt['test'][0]['output'][0,0] = (int(corrupt['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a corrupted label')


# Exact source geometries and output digests are regression evidence only.
_SOURCE_FIXTURES = (('test', 0, (30, 30), 0, 8, [{'core': (7, 23), 'color': 2, 'ports': ('NW', 'NE', 'SE')}, {'core': (11, 5), 'color': 3, 'ports': ('NW', 'NE', 'S0', 'SE')}, {'core': (11, 11), 'color': 4, 'ports': ('NW', 'NE', 'SW', 'S1')}, {'core': (25, 24), 'color': 7, 'ports': ('NE', 'E1', 'S0')}], 'd683d94a903bc5ed0aff2564708a09f1fe8fff9d0eb2d572f1868dae16ec130f'), ('test', 1, (30, 30), 8, 3, [{'core': (5, 4), 'color': 2, 'ports': ('N0', 'W0', 'SE')}, {'core': (5, 15), 'color': 4, 'ports': ('NW', 'E0', 'S0')}, {'core': (14, 23), 'color': 6, 'ports': ('N0', 'NE', 'SE')}, {'core': (20, 22), 'color': 7, 'ports': ('E0', 'SE')}, {'core': (25, 1), 'color': 9, 'ports': ('N1', 'SE')}], 'db15a11e15653b363903146a51eabf186db21b347ad07fe287812d2f19fc2a0a'), ('train', 0, (24, 24), 8, 3, [{'core': (10, 4), 'color': 9, 'ports': ('NW', 'NE', 'SW', 'SE')}, {'core': (10, 12), 'color': 1, 'ports': ('NW', 'NE', 'SW', 'SE')}, {'core': (10, 18), 'color': 4, 'ports': ('N0', 'S0')}, {'core': (20, 2), 'color': 6, 'ports': ('E0',)}], 'b310f18b59082f91488067c80a98ea6a50b3dcde563f7f83ee86151bdeaf2f71'), ('train', 1, (8, 8), 8, 3, [{'core': (3, 2), 'color': 4, 'ports': ('N1', 'E1', 'SW')}], 'cd76ba88cbe9f42800809e9067b1cf27437604d9509c6a036e1af61fedd5c9d2'), ('train', 2, (20, 20), 8, 3, [{'core': (6, 4), 'color': 2, 'ports': ('E0', 'W1')}, {'core': (11, 12), 'color': 4, 'ports': ('N1', 'W0', 'SE')}, {'core': (18, 6), 'color': 7, 'ports': ('W0',)}], 'c62ceb3656f81157d22d114b6e6b7d7add05043a0c4938fe544f2b589ad16d3c'), ('train', 3, (12, 12), 8, 3, [{'core': (3, 2), 'color': 6, 'ports': ('N0', 'NE', 'SW')}], '967772e5d915b387f0675338e5c6a446f748ff22b5c49a08440132925ae6c5e1'))


def _source_regressions():
    for split,index,shape,bg,ring,specs,digest in _SOURCE_FIXTURES:
        grid = _build_input(*shape,bg,ring,specs)
        result,_ = _solve_grid(grid)
        _require(np.array_equal(result,_analytic_predict(grid)), 'source predictors disagree')
        if split == 'train' and index == 2:
            # The source alone suppresses this unobstructed southeast port.
            # Record the exception; it is never used as a generation rule.
            recorded = result.copy()
            for r,c in ((13,14),(14,15),(15,16),(16,17),(17,18),(18,19)):
                _require(int(result[r,c]) == 4, 'unexpected source anomaly')
                recorded[r,c] = bg
        else:
            recorded = result
        _require(hashlib.sha256(recorded.astype(np.int8).tobytes()).hexdigest()==digest,
                 'source-derived collision or staircase-phase regression')
