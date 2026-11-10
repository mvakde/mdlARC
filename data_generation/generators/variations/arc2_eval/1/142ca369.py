import hashlib

import numpy as np


_DIRECTIONS = (
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

# Exact source contents, including dimensions; all source shapes remain legal.
_SOURCE_DIGESTS = ('d483ad6ee06376daac08e376b14a4fd21b1172282ccff8a60e78dd25eb2117ef', '4329b3ff74b72655ad964961de36a00dbcd6a9fc90a8d0afdc1f984d9c31cfbc', '1e56663b047d665488f05875ce18370f5216b3b5d5dc93a866704cd31b167eaf', '319ecadf0e42c2dc912a7ecaf448250f661b18296f750cbef97ae391549f984b', '043c8a79ff65e2246b48a809d5363d33063cc8ae9ee50de23697a22082d4c669')


def _as_optional_count(value, name):
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    num_train = _as_optional_count(num_train, "num_train")
    num_test = _as_optional_count(num_test, "num_test")

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        upper = min(4, num_train - 1)
        num_test = int(rng.integers(1, upper + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _infer_background(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    maximum = int(counts.max())
    winners = np.flatnonzero(counts == maximum)

    assert len(winners) == 1, "background must be uniquely identifiable"
    background = int(winners[0])
    assert maximum > grid.size // 2, "background must occupy a majority"

    return background


def _nonbackground_components(grid, background):
    remaining = set(map(tuple, np.argwhere(grid != background)))
    components = []

    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        cells = []

        while stack:
            row, col = stack.pop()
            cells.append((row, col))

            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbor in remaining and grid[neighbor] == grid[start]:
                    remaining.remove(neighbor)
                    stack.append(neighbor)

        components.append(tuple(sorted(cells)))

    return sorted(components, key=lambda component: component[0])


def _parse_scene(grid):
    background = _infer_background(grid)
    sources = []
    reflectors = []

    for cells in _nonbackground_components(grid, background):
        colors = {int(grid[cell]) for cell in cells}
        assert len(colors) == 1, "objects must be monochromatic"
        color = next(iter(colors))

        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]
        row_set = set(rows)
        col_set = set(cols)

        if len(cells) == 3 and len(row_set) == 2 and len(col_set) == 2:
            bounding_box = {
                (row, col)
                for row in row_set
                for col in col_set
            }
            missing = next(iter(bounding_box - set(cells)))

            degrees = {
                cell: sum(
                    (cell[0] + dr, cell[1] + dc) in cells
                    for dr, dc in (
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                    )
                )
                for cell in cells
            }
            elbows = [
                cell
                for cell, degree in degrees.items()
                if degree == 2
            ]
            assert len(elbows) == 1, "invalid L source"

            elbow = elbows[0]
            direction = (
                elbow[0] - missing[0],
                elbow[1] - missing[1],
            )
            assert direction in _DIRECTIONS

            sources.append(
                {
                    "cells": cells,
                    "color": color,
                    "elbow": elbow,
                    "direction": direction,
                }
            )

        elif len(cells) == 1:
            reflectors.append(
                {
                    "cells": cells,
                    "color": color,
                    "midpoint": cells[0],
                    "orientation": "point",
                }
            )

        elif len(cells) >= 3 and len(cells) % 2 == 1 and len(row_set) == 1:
            assert max(cols) - min(cols) + 1 == len(cells)
            reflectors.append(
                {
                    "cells": cells,
                    "color": color,
                    "midpoint": (
                        rows[0],
                        (min(cols) + max(cols)) // 2,
                    ),
                    "orientation": "horizontal",
                }
            )

        elif len(cells) >= 3 and len(cells) % 2 == 1 and len(col_set) == 1:
            assert max(rows) - min(rows) + 1 == len(cells)
            reflectors.append(
                {
                    "cells": cells,
                    "color": color,
                    "midpoint": (
                        (min(rows) + max(rows)) // 2,
                        cols[0],
                    ),
                    "orientation": "vertical",
                }
            )

        else:
            raise AssertionError("invalid source or reflector object")

    assert sources, "a scene must contain at least one source"
    return background, sources, reflectors


def _contact_axis(reflector, position, direction):
    midpoint_row, midpoint_col = reflector["midpoint"]
    row, col = position
    dr, dc = direction
    orientation = reflector["orientation"]

    if (
        orientation in ("horizontal", "point")
        and col == midpoint_col
        and abs(row - midpoint_row) == 1
        and midpoint_row - row == dr
    ):
        return "row"

    if (
        orientation in ("vertical", "point")
        and row == midpoint_row
        and abs(col - midpoint_col) == 1
        and midpoint_col - col == dc
    ):
        return "col"

    return None


def _solve_scene(input_grid):
    background, sources, reflectors = _parse_scene(input_grid)
    height, width = input_grid.shape

    occupied = {
        cell
        for obj in sources + reflectors
        for cell in obj["cells"]
    }

    output = input_grid.copy()
    painted = {}
    routes = []
    used_reflectors = set()

    for source_index, source in enumerate(sources):
        direction = source["direction"]
        current_color = source["color"]
        position = (
            source["elbow"][0] + direction[0],
            source["elbow"][1] + direction[1],
        )

        visited_states = set()
        path = []
        contacts = []

        while (
            0 <= position[0] < height
            and 0 <= position[1] < width
        ):
            state = (position, direction)
            assert state not in visited_states, "ray entered a loop"
            visited_states.add(state)
            assert len(visited_states) <= height * width * 8

            assert position not in occupied, (
                "ray struck an object cell instead of a midpoint face"
            )

            hits = []
            for reflector_index, reflector in enumerate(reflectors):
                axis = _contact_axis(
                    reflector,
                    position,
                    direction,
                )
                if axis is not None:
                    hits.append((reflector_index, axis))

            assert len(hits) <= 1, "ambiguous reflector contact"

            if hits:
                reflector_index, axis = hits[0]
                reflector = reflectors[reflector_index]
                used_reflectors.add(reflector_index)
                current_color = reflector["color"]

                if axis == "row":
                    direction = (-direction[0], direction[1])
                else:
                    direction = (direction[0], -direction[1])

                contacts.append(
                    (
                        reflector_index,
                        axis,
                        position,
                        current_color,
                    )
                )

            existing_color = painted.get(position)
            assert (
                existing_color is None
                or existing_color == current_color
            ), "incompatible ray overlap"

            painted[position] = current_color
            output[position] = current_color
            path.append((position, current_color))

            position = (
                position[0] + direction[0],
                position[1] + direction[1],
            )

        routes.append(
            {
                "source": source_index,
                "path": path,
                "contacts": contacts,
            }
        )

    assert used_reflectors == set(range(len(reflectors))), (
        "every reflector must participate in a route"
    )

    information = {
        "background": background,
        "sources": sources,
        "reflectors": reflectors,
        "routes": routes,
    }
    return output, information


def _inside(cells, height, width):
    return all(
        0 <= row < height and 0 <= col < width
        for row, col in cells
    )


def _source_cells(elbow, direction):
    missing = (
        elbow[0] - direction[0],
        elbow[1] - direction[1],
    )
    return {
        elbow,
        (elbow[0], missing[1]),
        (missing[0], elbow[1]),
    }


def _reflector_cells(midpoint, axis, length):
    if length == 1:
        return {midpoint}

    radius = length // 2

    if axis == "row":
        return {
            (midpoint[0], midpoint[1] + offset)
            for offset in range(-radius, radius + 1)
        }

    return {
        (midpoint[0] + offset, midpoint[1])
        for offset in range(-radius, radius + 1)
    }


def _grid_key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _source_digest(pair):
    inp, out = pair['input'], pair['output']
    return hashlib.sha256(bytes(inp.shape) + inp.astype(np.uint8).tobytes()
                          + out.astype(np.uint8).tobytes()).hexdigest()


def _place(grid, cells, color, background):
    """Different-colored objects may touch; same-color components stay unique."""
    if not _inside(cells, *grid.shape):
        return None
    for r, c in cells:
        if grid[r, c] != background:
            return None
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (rr, cc) not in cells and 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                if grid[rr, cc] == color:
                    return None
    result = grid.copy()
    for cell in cells:
        result[cell] = color
    return result


def _source_candidates(rng, height, width):
    corners, edges, interior = [], [], []
    for direction in _DIRECTIONS:
        for r in range(height):
            for c in range(width):
                cells = _source_cells((r, c), direction)
                if not _inside(cells, height, width):
                    continue
                # At least two visible steps after launch.
                if not _inside({(r+2*direction[0], c+2*direction[1])}, height, width):
                    continue
                rows = {x[0] for x in cells}
                cols = {x[1] for x in cells}
                sides = int(0 in rows or height-1 in rows) + int(0 in cols or width-1 in cols)
                (corners if sides == 2 else edges if sides == 1 else interior).append(((r, c), direction))
    for group in (corners, edges, interior):
        rng.shuffle(group)
    # Choose a placement class independently for every attempt.
    candidates = []
    for _ in range(180):
        group = (corners, edges, interior)[int(rng.choice(3, p=(.18, .40, .42)))]
        if group:
            candidates.append(group.pop())
    return candidates


def _add_source(grid, rng, palette, candidates, color=None):
    background = _infer_background(grid)
    for elbow, direction in candidates:
        shade = int(rng.choice(palette)) if color is None else color
        trial = _place(grid, _source_cells(elbow, direction), shade, background)
        if trial is None:
            continue
        try:
            _, info = _solve_scene(trial)
        except AssertionError:
            continue
        route = next(r for s, r in zip(info['sources'], info['routes']) if s['elbow'] == elbow)
        if len(route['path']) >= 2:
            return trial, elbow
    return None


def _extend(grid, elbow, rng, palette, axis=None, length=None, color=None):
    """Append one mirror to a visible final flight, then retrace every source."""
    _, before = _solve_scene(grid)
    source, route = next((s, r) for s, r in zip(before['sources'], before['routes'])
                         if s['elbow'] == elbow)
    direction = source['direction']
    for _, component, _, _ in route['contacts']:
        direction = (-direction[0], direction[1]) if component == 'row' else (direction[0], -direction[1])
    anchor = route['contacts'][-1][2] if route['contacts'] else elbow
    incoming = route['contacts'][-1][3] if route['contacts'] else source['color']
    tail = []
    for step in range(1, 31):
        cell = (anchor[0]+step*direction[0], anchor[1]+step*direction[1])
        if not _inside({cell}, *grid.shape):
            break
        tail.append(cell)
    options = [(cell, a, n) for cell in tail
               for a in ([axis] if axis else ['row', 'col'])
               for n in ([length] if length else [1, 3, 5, 7, 9])]
    rng.shuffle(options)
    for contact, a, n in options[:100]:
        # Midpoint is one step farther along the incoming normal.
        midpoint = (contact[0]+(direction[0] if a == 'row' else 0),
                    contact[1]+(direction[1] if a == 'col' else 0))
        reflected = (-direction[0], direction[1]) if a == 'row' else (direction[0], -direction[1])
        if not _inside({(contact[0]+2*reflected[0], contact[1]+2*reflected[1])}, *grid.shape):
            continue
        shade = color
        if shade is None:
            shade = incoming if rng.random() < .28 else int(rng.choice(palette))
        cells = _reflector_cells(midpoint, a, n)
        trial = _place(grid, cells, shade, before['background'])
        if trial is None:
            continue
        try:
            _, after = _solve_scene(trial)
        except AssertionError:
            continue
        new_route = next(r for s, r in zip(after['sources'], after['routes']) if s['elbow'] == elbow)
        old_contacts = [(before['reflectors'][i]['midpoint'], a0, p) for i, a0, p, _ in route['contacts']]
        new_contacts = [(after['reflectors'][i]['midpoint'], a0, p) for i, a0, p, _ in new_route['contacts']]
        if new_contacts == old_contacts + [(midpoint, a, contact)]:
            return trial
    return None


def _palette(rng, profile, background=None, anchor=None):
    background = int(rng.integers(10)) if background is None else background
    colors = [c for c in range(10) if c != background]
    rng.shuffle(colors)
    if anchor is not None:
        colors.remove(anchor)
        colors.insert(0, anchor)
    if profile == 'simple':
        count = int(rng.choice([1, 2, 3, 4], p=[.35, .40, .20, .05]))
    else:
        count = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], p=[.08, .15, .15, .13, .12, .11, .10, .08, .08]))
    if anchor is not None:
        count = max(2, count)
    return background, colors[:count]


def _shape(rng, dense=False):
    # Sampling dimensions independently includes compact and elongated scenes.
    if rng.random() < .12:
        return ((18, 18), (20, 19), (20, 20))[int(rng.integers(3))]
    low = 18 if dense else 8
    return int(rng.integers(low, 31)), int(rng.integers(low, 31))


def _compose(rng, profile, regime, teaching=None):
    dense_probability = (.15 if regime == 'target_test' else .08) if profile == 'simple' else (.28 if regime == 'target_test' else .20)
    dense = teaching is None and rng.random() < dense_probability
    height, width = _shape(rng, dense)
    if teaching:
        # Teaching is selected by evidence, without a fixed route template.
        height, width = int(rng.integers(16, 31)), int(rng.integers(16, 31))
        background, palette = _palette(rng, profile, teaching['background'], teaching['color'])
    else:
        background, palette = _palette(rng, profile)
    grid = np.full((height, width), background, dtype=np.int8)
    candidates = _source_candidates(rng, height, width)
    first = _add_source(grid, rng, palette, candidates, teaching['color'] if teaching else None)
    assert first is not None
    grid, elbow = first
    if dense:
        desired_sources = int(rng.integers(10, 21))
        first_bounces = int(rng.integers(1, 4)) if rng.random() < .28 else 0
    else:
        source_weights = [.45, .30, .15, .05, .03, .012, .004, .003, .001] if profile == 'simple' else [.23, .22, .20, .14, .10, .07, .02, .015, .005]
        bounce_weights = [.25, .30, .22, .13, .07, .02, .01] if profile == 'simple' else [.12, .16, .19, .18, .17, .11, .07]
        desired_sources = int(rng.choice(np.arange(1, 10), p=source_weights))
        first_bounces = int(rng.choice([0, 1, 2, 3, 4, 5, 6], p=bounce_weights))
    if teaching:
        first_bounces = int(rng.choice([2, 3, 4], p=[.70, .25, .05])) if profile == 'simple' else int(rng.integers(2, 7))
    for bounce in range(first_bounces):
        axis = teaching['axis'] if teaching and bounce == 0 else None
        length = int(rng.choice([3, 5, 7])) if axis else None
        color = None
        if teaching and bounce == 0:
            color = teaching['color'] if teaching['unchanged'] else int(rng.choice(palette[1:]))
        trial = _extend(grid, elbow, rng, palette, axis, length, color)
        if trial is None:
            break
        grid = trial
    if teaching:
        f = _scene_features(grid)
        transitions = f['transitions']
        expected = any(a == b for a, b in transitions) if teaching['unchanged'] else any(a != b for a, b in transitions)
        if teaching['axis'] not in f['bar_axes'] or not expected:
            return None
    # Each additional source has its own route length; no split-specific cap.
    for _ in range(desired_sources - 1):
        new_source = _add_source(grid, rng, palette, _source_candidates(rng, height, width))
        if new_source is None:
            break
        grid, new_elbow = new_source
        if dense:
            bounces = int(rng.integers(1, 3)) if rng.random() < .12 else 0
        else:
            bounces = int(rng.choice([0, 1, 2, 3, 4], p=[.38, .27, .20, .10, .05]))
        for _ in range(bounces):
            trial = _extend(grid, new_elbow, rng, palette)
            if trial is None:
                break
            grid = trial
    return {'input': grid, 'output': _solve_scene(grid)[0]}


def _teaching_fallback(rng, spec):
    """Guaranteed two-contact construction, with random scale and rotation."""
    # Two reversals on one axis use a monotone coordinate, so cannot loop.
    span = int(rng.integers(4, 7))
    grid = np.full((24, 26), spec['background'], dtype=np.int8)
    color = spec['color']
    other = next(c for c in range(10) if c not in (color, spec['background']))
    objects = [(_source_cells((2, 3), (1, 1)), color),
               (_reflector_cells((2+span, 4+span), 'col', 3), color if spec['unchanged'] else other),
               ({(2+2*span, 2)}, other)]
    if not spec['unchanged']:
        objects.append((_source_cells((21, 23), (1, 1)), color))
    for cells, shade in objects:
        for p in cells:
            grid[p] = shade
    if spec['axis'] == 'row':
        grid = grid.T.copy()
    if rng.integers(2):
        grid = grid[::-1].copy()
    if rng.integers(2):
        grid = grid[:, ::-1].copy()
    return {'input': grid, 'output': _solve_scene(grid)[0]}


def _scene_features(grid):
    output, info = _solve_scene(grid)
    sources, reflectors = info['sources'], info['reflectors']
    axes, transitions, flights, bounce_counts = [], [], [], []
    repeated = returning = overlap = 0
    visited = set()
    for source, route in zip(sources, info['routes']):
        color, anchor = source['color'], source['elbow']
        sequence = [color]
        local_axes = []
        for i, axis, contact, shade in route['contacts']:
            local_axes.append(axis)
            transitions.append((color, shade))
            flights.append(abs(contact[0]-anchor[0]))
            sequence.append(shade)
            color, anchor = shade, contact
        axes.extend(local_axes)
        repeated += sum(a == b for a, b in zip(local_axes, local_axes[1:]))
        returning += sum(shade in sequence[:i-1] and shade != sequence[i-1]
                         for i, shade in enumerate(sequence) if i >= 2)
        bounce_counts.append(len(route['contacts']))
        for p, _ in route['path']:
            overlap += p in visited
            visited.add(p)
    placements = []
    for source in sources:
        rows = {p[0] for p in source['cells']}
        cols = {p[1] for p in source['cells']}
        sides = int(0 in rows or grid.shape[0]-1 in rows) + int(0 in cols or grid.shape[1]-1 in cols)
        placements.append(('interior', 'edge', 'corner')[sides])
    touch = 0
    for i, source in enumerate(sources):
        for other in sources[i+1:]:
            if any(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a in source['cells'] for b in other['cells']):
                touch += 1
    foreground = np.argwhere(grid != info['background'])
    return {'shape': grid.shape, 'background': info['background'],
            'palette': tuple(int(c) for c in np.unique(grid) if c != info['background']),
            'source_count': len(sources), 'reflector_count': len(reflectors),
            'directions': tuple(sorted({s['direction'] for s in sources})),
            'placements': tuple(placements), 'bounce_counts': tuple(bounce_counts),
            'axes': tuple(axes), 'bar_axes': tuple(sorted({axis for route in info['routes']
                for i, axis, _, _ in route['contacts'] if len(reflectors[i]['cells']) > 1})),
            'lengths': tuple(len(r['cells']) for r in reflectors),
            'orientations': tuple(r['orientation'] for r in reflectors),
            'transitions': tuple(transitions), 'flights': tuple(flights),
            'repeated_axes': repeated, 'returning_colors': returning,
            'touching_sources': touch, 'same_color_overlap': overlap,
            'density': float(np.count_nonzero(output != grid)/grid.size),
            'margins': (int(foreground[:, 0].min()), int(foreground[:, 1].min()),
                        int(grid.shape[0]-1-foreground[:, 0].max()), int(grid.shape[1]-1-foreground[:, 1].max()))}


def _training_coverage(pairs):
    features = [_scene_features(p['input']) for p in pairs]
    axes = {a for f in features for a in f['bar_axes']}
    directions = {d for f in features for d in f['directions']}
    transitions = [t for f in features for t in f['transitions']]
    successors = {}
    for a, b in transitions:
        successors.setdefault(a, set()).add(b)
    return (axes == {'row', 'col'} and len(directions) >= 2
            and any(1 in f['lengths'] for f in features)
            and any(a == b for a, b in transitions)
            and any(a != b for a, b in transitions)
            and any(len(v) > 1 for v in successors.values()))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    ntrain, ntest = _resolve_counts(rng, num_train, num_test)
    # Only sampling preferences are puzzle-level. Geometry and visible colors
    # have the same meaning for every seed; there is no hidden color mapping.
    profile = 'simple' if rng.random() < .22 else 'varied'
    anchor_color = int(rng.integers(10))
    backgrounds = [c for c in range(10) if c != anchor_color]
    rng.shuffle(backgrounds)
    first_axis = 'row' if rng.integers(2) else 'col'
    specs = [{'color': anchor_color, 'background': backgrounds[i],
              'axis': first_axis if i == 0 else ('col' if first_axis == 'row' else 'row'),
              'unchanged': i == 0} for i in range(2)]
    puzzle = {'train': [], 'test': []}
    # Select demonstrations by what their completed visible trajectories teach.
    for attempt in range(24):
        demos = [_compose(rng, profile, 'source_train', spec) for spec in specs]
        if (all(p is not None for p in demos) and _training_coverage(demos)
                and demos[0]['input'].shape != demos[1]['input'].shape
                and _scene_features(demos[0]['input'])['source_count'] != _scene_features(demos[1]['input'])['source_count']
                and all(_source_digest(p) not in _SOURCE_DIGESTS for p in demos)):
            puzzle['train'] = demos
            break
    else:
        puzzle['train'] = [_teaching_fallback(rng, spec) for spec in specs]
        if not _training_coverage(puzzle['train']):
            pair = puzzle['train'][1]
            puzzle['train'][1] = {k: v[::-1, ::-1].copy() for k, v in pair.items()}
    seen = {_grid_key(p['input']) for p in puzzle['train']}
    for split, regime, count in [('train', 'source_train', ntrain), ('test', 'target_test', ntest)]:
        while len(puzzle[split]) < count:
            for attempt in range(20):
                pair = _compose(rng, profile, regime)
                if _grid_key(pair['input']) not in seen and _source_digest(pair) not in _SOURCE_DIGESTS:
                    break
            else:
                # Deterministic enumeration of distinct source locations on a
                # clear canvas guarantees an available pair without retries.
                pair = _clear_fallback(seen, anchor_color, backgrounds[0])
            seen.add(_grid_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _clear_fallback(seen, color, background):
    for h in range(8, 31):
        for w in range(8, 31):
            for c in range(1, w-3):
                grid = np.full((h, w), background, dtype=np.int8)
                for p in _source_cells((1, c), (1, 1)):
                    grid[p] = color
                if _grid_key(grid) not in seen:
                    return {'input': grid, 'output': _solve_scene(grid)[0]}
    raise AssertionError('more scenes than the legal pair-count maximum')
def _event_prediction(grid):
    """A separate, event-based checker; no builder, parser, or painted labels."""
    h, w = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    assert counts.max() > grid.size / 2
    checked = np.zeros((h, w), dtype=bool)
    launches, mirrors, events = [], [], []
    for row in range(h):
        for col in range(w):
            if checked[row, col] or grid[row, col] == bg:
                continue
            shade = int(grid[row, col])
            todo, component = [(row, col)], []
            checked[row, col] = True
            while todo:
                r, c = todo.pop()
                component.append((r, c))
                for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if 0 <= a < h and 0 <= b < w and not checked[a, b] and grid[a, b] == shade:
                        checked[a, b] = True
                        todo.append((a, b))
            r0, c0 = np.min(component, axis=0)
            r1, c1 = np.max(component, axis=0)
            if len(component) == 3 and r1-r0 == c1-c0 == 1:
                absent = [(r, c) for r in (int(r0), int(r1)) for c in (int(c0), int(c1))
                          if (r, c) not in component][0]
                elbow = (int(r0+r1)-absent[0], int(c0+c1)-absent[1])
                launches.append((elbow, (elbow[0]-absent[0], elbow[1]-absent[1]), shade))
            else:
                assert len(component) % 2 == 1 and (r0 == r1 or c0 == c1), 'ambiguous object decomposition'
                assert len(component) == max(r1-r0, c1-c0)+1
                mid = ((int(r0+r1)//2), (int(c0+c1)//2))
                index = len(mirrors)
                mirrors.append(component)
                for axis in (0, 1):
                    if len(component) > 1 and ((axis == 0 and r0 != r1) or (axis == 1 and c0 != c1)):
                        continue
                    for sign in (-1, 1):
                        point = list(mid)
                        point[axis] -= sign
                        if 0 <= point[0] < h and 0 <= point[1] < w:
                            events.append((tuple(point), axis, sign, shade, index))
    assert launches
    answer = grid.copy()
    painted, used = {}, set()
    for position, direction, shade in launches:
        visited = set()
        while True:
            state = position, direction
            assert state not in visited, 'loop'
            visited.add(state)
            dr, dc = direction
            edge_distance = min(h-1-position[0] if dr > 0 else position[0],
                                w-1-position[1] if dc > 0 else position[1])
            reachable = []
            for point, axis, sign, new_shade, index in events:
                distance = (point[0]-position[0])*dr
                if 0 < distance <= edge_distance and (point[1]-position[1])*dc == distance and direction[axis] == sign:
                    reachable.append((distance, point, axis, new_shade, index))
            reachable.sort()
            if reachable:
                event = reachable[0]
                assert len(reachable) == 1 or event[0] != reachable[1][0], 'ambiguous contact'
                distance = event[0]
            else:
                event, distance = None, edge_distance
            for step in range(1, distance+1):
                point = (position[0]+step*dr, position[1]+step*dc)
                assert grid[point] == bg, 'path hits an object'
                cell_color = event[3] if event and step == distance else shade
                assert point not in painted or painted[point] == cell_color, 'conflicting shared cell'
                painted[point] = cell_color
                answer[point] = cell_color
            if event is None:
                break
            _, position, axis, shade, index = event
            direction = (-dr, dc) if axis == 0 else (dr, -dc)
            used.add(index)
    assert used == set(range(len(mirrors))), 'unused reflector'
    return answer


def _predict_from_training(train_pairs, test_input):
    # Only training labels and the queried input enter this prediction API.
    assert _training_coverage(train_pairs), 'insufficient demonstration evidence'
    for pair in train_pairs:
        assert np.array_equal(_event_prediction(pair['input']), pair['output'])
    return _event_prediction(test_input)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    inputs, pairs = set(), set()
    features = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            ikey = _grid_key(inp)
            pkey = ikey, _grid_key(out)
            assert ikey not in inputs and pkey not in pairs
            assert _source_digest(pair) not in _SOURCE_DIGESTS, 'exact source copy'
            inputs.add(ikey)
            pairs.add(pkey)
            assert np.array_equal(out, _event_prediction(inp)), 'event prediction mismatch'
            f = _scene_features(inp)
            features.append(f)
            assert np.array_equal(out[inp != f['background']], inp[inp != f['background']])
            assert set(np.unique(out)) <= set(np.unique(inp))
    assert _training_coverage(puzzle['train'])
    assert len({f['background'] for f in features}) >= 2
    assert len({f['shape'] for f in features}) >= 2
    assert len({f['source_count'] for f in features}) >= 2
    # Both regimes use all visible object types. Test novelty is checked by
    # input uniqueness, not by an artificial source-count or palette rule.


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
                   for k in ('input', 'output')) for s in ('train', 'test'))


def _structural_key(grid):
    background = _infer_background(grid)
    r, c = np.nonzero(grid != background)
    crop = grid[r.min():r.max()+1, c.min():c.max()+1]
    mapping, encoded = {background: 0}, []
    for value in crop.flat:
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        encoded.append(mapping[value])
    return crop.shape, bytes(encoded)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    records = []
    structures = set()
    new_color_puzzles = 0
    converging_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [{k: v.copy() for k, v in pair.items()} for pair in puzzle['train']]
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _scene_features(pair['input'])
                records.append((split, f))
                structures.add(_structural_key(pair['input']))
                # Test outputs are accessed only after prediction is complete.
                prediction = _predict_from_training(train, pair['input'].copy()) if split == 'test' else _event_prediction(pair['input'])
                assert np.array_equal(prediction, pair['output'])
        fs = [_scene_features(p['input']) for s in ('train', 'test') for p in puzzle[s]]
        assert len({f['palette'] for f in fs}) >= 2
        training_features = [_scene_features(p['input']) for p in puzzle['train']]
        test_features = [_scene_features(p['input']) for p in puzzle['test']]
        train_colors = {c for f in training_features for c in f['palette']}
        new_color_puzzles += bool({c for f in test_features for c in f['palette']} - train_colors)
        transitions = [t for f in training_features for t in f['transitions']]
        converging_puzzles += any(len({a for a, b in transitions if b == shade}) > 1 for shade in range(10))
    assert len(structures) >= 250
    assert len({f['shape'] for _, f in records}) >= 100
    assert {f['background'] for _, f in records} == set(range(10))
    assert new_color_puzzles >= 5 and converging_puzzles >= 30
    for split in ('train', 'test'):
        fs = [f for s, f in records if s == split]
        assert {n for f in fs for n in f['lengths']} >= {1, 3, 5, 7, 9}
        assert {p for f in fs for p in f['placements']} == {'interior', 'edge', 'corner'}
        assert {d for f in fs for d in f['directions']} == set(_DIRECTIONS)
        assert {n for f in fs for n in f['bounce_counts']} >= {0, 1, 2, 3, 4}
        assert max(max(f['bounce_counts']) for f in fs) >= 5
        assert any(f['repeated_axes'] for f in fs)
        assert any(f['returning_colors'] for f in fs)
        assert any(f['touching_sources'] for f in fs)
        assert any(f['same_color_overlap'] for f in fs)
        assert any(f['source_count'] >= 10 and f['reflector_count'] > 0 for f in fs)
        assert any(f['touching_sources'] and f['reflector_count'] > 0 for f in fs)
        assert any(len(f['palette']) >= 7 for f in fs)
        assert any(len(f['palette']) == 1 for f in fs)
        assert any(len(f['directions']) >= 3 for f in fs)
        assert any(len(set(f['bounce_counts'])) >= 3 for f in fs)
        assert any(f['repeated_axes'] and len(f['directions']) > 1 and
                   len(f['palette']) > 2 and len(set(f['orientations'])) > 1 for f in fs)
        assert min(n for f in fs for n in f['shape']) <= 10
        assert max(n for f in fs for n in f['shape']) == 30
        assert {18, 19, 20, 29, 30} <= {n for f in fs for n in f['shape']}
        assert min(d for f in fs for d in f['flights']) <= 2
        assert max(d for f in fs for d in f['flights']) >= 10
        assert max(f['source_count'] for f in fs) >= 14
        assert len({f['margins'] for f in fs}) >= 20
        assert max(f['density'] for f in fs) - min(f['density'] for f in fs) > .20
    assert _puzzles_equal(puzzles[0], generate_puzzle(0))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    for i, counts in enumerate([(2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)]):
        p = generate_puzzle(200+i, *counts)
        validate_generated_puzzle(p)
        for split, count in zip(('train', 'test'), counts):
            if count is not None:
                assert len(p[split]) == count
        assert _puzzles_equal(p, generate_puzzle(200+i, *counts))
    for counts in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 4), (4, 4), (True, 1), (3.5, 1)]:
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('invalid count accepted')
    # The second checker must reject a missing tail pixel and modified objects.
    sample = puzzles[0]['test'][0]
    changed = np.argwhere(sample['input'] != sample['output'])[0]
    bad = {'train': puzzles[0]['train'], 'test': [dict(sample)]}
    bad['test'][0]['output'] = sample['output'].copy()
    bad['test'][0]['output'][tuple(changed)] = sample['input'][tuple(changed)]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    _rule_regressions()


def _rule_regressions():
    # Directly specified expectations for the review's parser and contact rules.
    adjacent = np.zeros((12, 12), dtype=np.int8)
    adjacent[0, 1] = adjacent[1, 0] = adjacent[1, 1] = 1
    adjacent[0, 3] = adjacent[1, 2] = adjacent[1, 3] = 2
    expected = adjacent.copy()
    for r in range(2, 12):
        expected[r, r] = 1
        if r+2 < 12:
            expected[r, r+2] = 2
    assert np.array_equal(_solve_scene(adjacent)[0], expected)
    assert np.array_equal(_event_prediction(adjacent), expected)

    seven = np.zeros((12, 13), dtype=np.int8)
    seven[0, 2] = seven[1, 1] = seven[1, 2] = 1
    seven[7, 4:11] = 2
    expected = seven.copy()
    for r, c in ((2, 3), (3, 4), (4, 5), (5, 6)):
        expected[r, c] = 1
    for r, c in ((6, 7), (5, 8), (4, 9), (3, 10), (2, 11), (1, 12)):
        expected[r, c] = 2
    assert np.array_equal(_solve_scene(seven)[0], expected)
    assert np.array_equal(_event_prediction(seven), expected)

    for size, second_color in ((12, 2), (11, 1), (11, 2)):
        crossing = np.zeros((size, size), dtype=np.int8)
        crossing[0, 1] = crossing[1, 0] = crossing[1, 1] = 1
        crossing[0, size-2] = crossing[1, size-2] = crossing[1, size-1] = second_color
        if size == 11 and second_color == 2:
            conflicting = crossing
        else:
            # Opposite parity crosses between cells; equal colors can share.
            assert np.array_equal(_event_prediction(crossing), _solve_scene(crossing)[0])
    monochrome_touch = adjacent.copy()
    monochrome_touch[monochrome_touch == 2] = 1
    ambiguous = np.zeros((8, 8), dtype=np.int8)
    ambiguous[0, 1] = ambiguous[1, 0] = ambiguous[1, 1] = 1
    ambiguous[4, 3], ambiguous[3, 4] = 2, 3
    for bad in (conflicting, monochrome_touch, ambiguous):
        for solver in (_event_prediction, _solve_scene):
            try:
                solver(bad)
            except AssertionError:
                continue
            raise AssertionError('ambiguous or conflicting scene accepted')
    # Exercise the guaranteed fallback independently of random rejection rates.
    for seed in range(12):
        rng = np.random.default_rng(seed)
        first = {'color': 1, 'background': 0, 'axis': 'row', 'unchanged': True}
        second = {'color': 1, 'background': 3, 'axis': 'col', 'unchanged': False}
        demos = [_teaching_fallback(rng, s) for s in (first, second)]
        if not _training_coverage(demos):
            demos[1] = {k: v[::-1, ::-1].copy() for k, v in demos[1].items()}
        assert _training_coverage(demos)
        for pair in demos:
            assert np.array_equal(_event_prediction(pair['input']), pair['output'])
