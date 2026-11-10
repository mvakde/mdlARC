"""Connected payloads, completed marker rays, and collective corner translation."""

import numpy as np

# Source fixtures support exact-content exclusion and semantic regressions only.
_SOURCE_PAIRS = (
    ('111111112111111111/111111111111111111/111111111111111111/111111111111111111/111177111111111111/111177111111111111/111117111711111111/111117177777111111/111117111771111112/111177777771111111/111177111777771111/111111111117711111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111',
     '771111112111111711/771111112111111777/171111112111111111/171711112111111111/111111112111111111/111111112111111111/111111112111111111/111111117111111111/111117222772222222/111111117111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111177111/111111111111177777/111111111111111771'),
    ('111111111211111111/111111111111111111/111111111111111111/111111111111111111/111111117711111111/111111177777111111/111111171117711111/111117777117711111/111111171177111111/111111177771111111/111117771777711111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111',
     '111711111211111771/117711111211111177/117111111211111177/777711111211111771/117111111711111711/117711111711111777/777111111211111111/111111111211111111/111111111211111111/111111111711111111/111111111711111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111'),
    ('111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111177111711111/111111777111711111/111117711171711111/111117777777711111/111111771771711111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111211111111',
     '111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111111111111/111111111711111111/111111111711111111/111111111211111111/117711111211111117/177711111211111117/771111111211111717/777711111211111777/177111111211111717'),
    ('11111111111111111111/11111111111111111111/11111111111111111111/11111111111111111111/11111111111111111111/11111111111111111111/11111111777777111111/11117711171111111111/11111771171111111111/11111177771177111111/11111177177771111111/11111171111711111111/11117771111777111111/11111111111111111111/11111111111111111111/11111111111111111111/11111111111111111111/11111111111111111111/11111111111111111111/11111111211111111111',
     '11111111111111111111/11111111111111111111/11111111111111111111/11111111111111111111/11111111111111111111/11111111111111111111/11111111711111111111/11111111211111111111/11111111211111111111/11111111711111111111/11111111211111111111/11111111211111111111/11111111211111111111/11111111211111177777/77111111211111171111/17711111211111171111/11771111211111171177/11771111211111177771/11711111211111111711/77711111211111111777'),
)

def _count_value(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)

def _resolve_counts(seed, num_train, num_test):
    n_train = _count_value(num_train, "num_train")
    n_test = _count_value(num_test, "num_test")

    if n_train is not None and not 2 <= n_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if n_test is not None and not 1 <= n_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if n_train is not None and n_test is not None:
        if n_train <= n_test:
            raise ValueError("num_train must be greater than num_test")
        return n_train, n_test

    if n_train is None and n_test is None:
        n_test = 1 + seed % 4
        low = n_test + 1
        n_train = low + (seed // 4) % (11 - low)
    elif n_train is None:
        low = n_test + 1
        n_train = low + seed % (11 - low)
    else:
        high = min(4, n_train - 1)
        if high < 1:
            raise ValueError("no legal num_test exists")
        n_test = 1 + seed % high

    if not (
        2 <= n_train <= 10
        and 1 <= n_test <= 4
        and n_train > n_test
    ):
        raise ValueError("the requested pair counts are incompatible")

    return n_train, n_test

def _guide_path(side, marker_coord, payload_coords):
    row, col = marker_coord

    if side == "top":
        aligned = [
            payload_row
            for payload_row, payload_col in payload_coords
            if payload_col == col
        ]
        assert aligned
        return [
            (current_row, col)
            for current_row in range(0, max(aligned) + 1)
        ]

    if side == "bottom":
        aligned = [
            payload_row
            for payload_row, payload_col in payload_coords
            if payload_col == col
        ]
        assert aligned
        return [
            (current_row, col)
            for current_row in range(min(aligned), row + 1)
        ]

    if side == "left":
        aligned = [
            payload_col
            for payload_row, payload_col in payload_coords
            if payload_row == row
        ]
        assert aligned
        return [
            (row, current_col)
            for current_col in range(0, max(aligned) + 1)
        ]

    assert side == "right"
    aligned = [
        payload_col
        for payload_row, payload_col in payload_coords
        if payload_row == row
    ]
    assert aligned
    return [
        (row, current_col)
        for current_col in range(min(aligned), col + 1)
    ]

def _analyze_input(grid):
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert grid.size
    assert int(grid.min()) >= 0
    assert int(grid.max()) <= 9

    height, width = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3

    # Border occupancy identifies the canvas even for unusually dense shapes.
    rim = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    rim_colors, rim_counts = np.unique(rim, return_counts=True)
    background = int(rim_colors[np.argmax(rim_counts)])

    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    nonbackground_border = border & (grid != background)
    marker_coords_array = np.argwhere(nonbackground_border)
    assert 1 <= len(marker_coords_array) <= 2

    marker_values = np.unique(grid[nonbackground_border])
    assert len(marker_values) == 1
    marker = int(marker_values[0])

    assert int(np.count_nonzero(grid == marker)) == len(
        marker_coords_array
    )

    payload_candidates = (
        set(int(value) for value in colors)
        - {background, marker}
    )
    assert len(payload_candidates) == 1
    payload = payload_candidates.pop()

    payload_array = np.argwhere(grid == payload)
    assert len(payload_array) >= 3

    assert np.all(
        (payload_array[:, 0] > 0)
        & (payload_array[:, 0] < height - 1)
    )
    assert np.all(
        (payload_array[:, 1] > 0)
        & (payload_array[:, 1] < width - 1)
    )

    payload_coords = [
        tuple(map(int, point))
        for point in payload_array
    ]
    payload_set = set(payload_coords)

    seen = {payload_coords[0]}
    stack = [payload_coords[0]]

    while stack:
        row, col = stack.pop()

        for neighbor in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if neighbor in payload_set and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)

    assert seen == payload_set

    min_row, min_col = payload_array.min(axis=0)
    max_row, max_col = payload_array.max(axis=0)
    bounding_area = int(
        (max_row - min_row + 1)
        * (max_col - min_col + 1)
    )
    assert len(payload_coords) < bounding_area

    markers = {}

    for raw_row, raw_col in marker_coords_array:
        row = int(raw_row)
        col = int(raw_col)
        side_flags = []

        if row == 0:
            side_flags.append("top")
        if row == height - 1:
            side_flags.append("bottom")
        if col == 0:
            side_flags.append("left")
        if col == width - 1:
            side_flags.append("right")

        assert len(side_flags) == 1
        side = side_flags[0]
        assert side not in markers
        markers[side] = (row, col)

    assert len(markers) in (1, 2)

    vertical_sides = set(markers) & {"top", "bottom"}
    horizontal_sides = set(markers) & {"left", "right"}

    if len(markers) == 2:
        assert len(vertical_sides) == 1
        assert len(horizontal_sides) == 1

    vertical_col = None
    horizontal_row = None

    if vertical_sides:
        side = next(iter(vertical_sides))
        vertical_col = markers[side][1]

    if horizontal_sides:
        side = next(iter(horizontal_sides))
        horizontal_row = markers[side][0]

    guide_spans = {}

    for side, coord in markers.items():
        path = _guide_path(side, coord, payload_coords)
        guide_spans[side] = len(path)

    if len(markers) == 1:
        side = next(iter(markers))

        if side in ("top", "bottom"):
            assert any(
                col < vertical_col
                for row, col in payload_coords
            )
            assert any(
                col > vertical_col
                for row, col in payload_coords
            )
        else:
            assert any(
                row < horizontal_row
                for row, col in payload_coords
            )
            assert any(
                row > horizontal_row
                for row, col in payload_coords
            )
    else:
        occupied_quadrants = {
            (
                "top" if row < horizontal_row else "bottom",
                "left" if col < vertical_col else "right",
            )
            for row, col in payload_coords
            if row != horizontal_row and col != vertical_col
        }

        assert occupied_quadrants == {
            ("top", "left"),
            ("top", "right"),
            ("bottom", "left"),
            ("bottom", "right"),
        }

    return {
        "background": background,
        "payload": payload,
        "marker": marker,
        "markers": markers,
        "payload_coords": payload_coords,
        "vertical_col": vertical_col,
        "horizontal_row": horizontal_row,
        "guide_spans": guide_spans,
    }

def _place_partition(
    output,
    cells,
    corner,
    color,
    background,
):
    if not cells:
        return

    height, width = output.shape
    rows = [row for row, col in cells]
    cols = [col for row, col in cells]
    vertical, horizontal = corner

    if vertical == "top":
        row_shift = -min(rows)
    else:
        row_shift = height - 1 - max(rows)

    if horizontal == "left":
        col_shift = -min(cols)
    else:
        col_shift = width - 1 - max(cols)

    targets = [
        (row + row_shift, col + col_shift)
        for row, col in cells
    ]

    assert all(
        0 <= row < height and 0 <= col < width
        for row, col in targets
    )
    assert all(
        int(output[row, col]) == background
        for row, col in targets
    )

    for row, col in targets:
        output[row, col] = color

def _solve_grid(grid):
    info = _analyze_input(grid)

    background = info["background"]
    payload = info["payload"]
    marker = info["marker"]
    markers = info["markers"]
    vertical_col = info["vertical_col"]
    horizontal_row = info["horizontal_row"]
    payload_coords = info["payload_coords"]

    output = np.full(
        grid.shape,
        background,
        dtype=grid.dtype,
    )

    fixed = []
    moving = []

    for row, col in payload_coords:
        on_axis = (
            vertical_col is not None
            and col == vertical_col
        ) or (
            horizontal_row is not None
            and row == horizontal_row
        )

        if on_axis:
            fixed.append((row, col))
        else:
            moving.append((row, col))

    for row, col in fixed:
        output[row, col] = payload

    for side, coord in markers.items():
        path = _guide_path(side, coord, payload_coords)

        for row, col in path:
            if int(output[row, col]) == background:
                output[row, col] = marker

    if len(markers) == 1:
        side = next(iter(markers))

        if side in ("top", "bottom"):
            left = [
                (row, col)
                for row, col in moving
                if col < vertical_col
            ]
            right = [
                (row, col)
                for row, col in moving
                if col > vertical_col
            ]

            _place_partition(
                output,
                left,
                (side, "left"),
                payload,
                background,
            )
            _place_partition(
                output,
                right,
                (side, "right"),
                payload,
                background,
            )
        else:
            above = [
                (row, col)
                for row, col in moving
                if row < horizontal_row
            ]
            below = [
                (row, col)
                for row, col in moving
                if row > horizontal_row
            ]

            _place_partition(
                output,
                above,
                ("top", side),
                payload,
                background,
            )
            _place_partition(
                output,
                below,
                ("bottom", side),
                payload,
                background,
            )
    else:
        groups = {}

        for row, col in moving:
            corner = (
                "top" if row < horizontal_row else "bottom",
                "left" if col < vertical_col else "right",
            )
            groups.setdefault(corner, []).append((row, col))

        marked_sides = set(markers)

        for corner, cells in groups.items():
            if (
                corner[0] in marked_sides
                or corner[1] in marked_sides
            ):
                _place_partition(
                    output,
                    cells,
                    corner,
                    payload,
                    background,
                )

    return output

def _puzzles_equal(first, second):
    if set(first) != set(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
        ):
            if set(pair_a) != set(pair_b):
                return False
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False
            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True

def _grid_key(grid):
    # Contents, including dimensions, are independent of the integer dtype.
    return (grid.shape, grid.astype(np.int8).tobytes())


def _decode(rows):
    return np.array([[int(c) for c in row] for row in rows.split('/')], dtype=np.int8)


_SOURCE_KEYS = {(_grid_key(_decode(a)), _grid_key(_decode(b))) for a, b in _SOURCE_PAIRS}
_SIDES = ('top', 'right', 'bottom', 'left')
_ORIENTATIONS = tuple(frozenset((_SIDES[k],)) for k in range(4)) + tuple(
    frozenset((_SIDES[k], _SIDES[(k + 1) % 4])) for k in range(4))


def _neighbors(point):
    r, c = point
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        found, stack = {start}, [start]
        while stack:
            for neighbor in _neighbors(stack.pop()):
                if neighbor in todo:
                    todo.remove(neighbor)
                    found.add(neighbor)
                    stack.append(neighbor)
        result.append(found)
    return result


def _groups(cells, ar, ac, sides):
    groups = {}
    for r, c in cells:
        if r == ar or c == ac:
            continue
        vertical = ('top' if r < ar else 'bottom') if ar is not None else next(iter(sides))
        horizontal = ('left' if c < ac else 'right') if ac is not None else next(iter(sides))
        groups.setdefault((vertical, horizontal), []).append((r, c))
    return groups


def _gap_lengths(values):
    values = sorted(set(values))
    return [b - a - 1 for a, b in zip(values, values[1:]) if b > a + 1]


def _run_lengths(values):
    result, previous = [], None
    for value in sorted(set(values)):
        if previous is None or value != previous + 1:
            result.append(1)
        else:
            result[-1] += 1
        previous = value
    return result


def _short_endpoint(cells, ar, ac):
    # Canonical construction uses top and (for two axes) right markers.
    vertical = [r for r, c in cells if c == ac]
    return (max(vertical) < max(r for r, c in cells)
            or (ar is not None and min(c for r, c in cells if r == ar)
                > min(c for r, c in cells)))


def _cut_features(cells, ar, ac, sides):
    groups = _groups(cells, ar, ac, sides)
    if len(groups) != (4 if len(sides) == 2 else 2):
        return None
    gaps, runs = [], []
    for axis in (0, 1):
        coordinate = ac if axis == 0 else ar
        if coordinate is not None:
            values = [p[axis] for p in cells if p[1 - axis] == coordinate]
            if not values:
                return None
            axis_gaps = _gap_lengths(values)
            gaps.extend(axis_gaps)
            runs.append(len(axis_gaps) + 1)
    retained = [v for corner, v in groups.items() if set(corner) & sides]
    counts = [len(_components(v)) for v in retained]
    return groups, gaps, runs, counts


def pair_features(grid):
    """Review measurements reconstructed from the returned input, without metadata."""
    info = _analyze_input(grid)
    cells = info['payload_coords']
    ar, ac, sides = info['horizontal_row'], info['vertical_col'], set(info['markers'])
    groups, gaps, runs, components = _cut_features(cells, ar, ac, sides)
    run_lengths = []
    noncentral_gaps = 0
    if ac is not None:
        aligned = [r for r, c in cells if c == ac]
        run_lengths.extend(_run_lengths(aligned))
        noncentral_gaps += sum(r not in aligned and r != ar
                               for r in range(min(aligned), max(aligned)+1))
    if ar is not None:
        aligned = [c for r, c in cells if r == ar]
        run_lengths.extend(_run_lengths(aligned))
        noncentral_gaps += sum(c not in aligned and c != ac
                               for c in range(min(aligned), max(aligned)+1))
    sizes, skinny, singleton, unoccupied_corner = [], 0, 0, 0
    for corner, group in groups.items():
        if not set(corner) & sides:
            continue
        rs, cs = zip(*group)
        sizes.append(len(group))
        skinny += int(min(max(rs) - min(rs), max(cs) - min(cs)) == 0)
        singleton += int(len(group) == 1)
        corner_cell = (min(rs) if corner[0] == 'top' else max(rs),
                       min(cs) if corner[1] == 'left' else max(cs))
        unoccupied_corner += int(corner_cell not in group)
    distances, endpoint_short = [], False
    for side, p in info['markers'].items():
        axis = 0 if side in ('top', 'bottom') else 1
        aligned = [v[axis] for v in cells if v[1-axis] == p[1-axis]]
        distances.append(min(abs(v - p[axis]) for v in aligned) - 1)
        inward_far = max if side in ('top', 'left') else min
        endpoint_short |= inward_far(aligned) != inward_far([v[axis] for v in cells])
    rs, cs = zip(*cells)
    return {
        'orientation': '+'.join(sorted(sides)), 'marker_count': len(sides),
        'roles': [info[k] for k in ('background', 'payload', 'marker')],
        'shape': list(grid.shape), 'payload_size': len(cells),
        'density': len(cells) / ((max(rs)-min(rs)+1)*(max(cs)-min(cs)+1)),
        'payload_box': [min(rs), max(rs), min(cs), max(cs)],
        'axes': [ar, ac], 'gap_lengths': gaps, 'axis_runs': runs,
        'axial_run_lengths': run_lengths,
        'noncentral_gap_cells': noncentral_gaps,
        'guide_gaps': distances, 'guide_spans': list(info['guide_spans'].values()),
        'endpoint_short_of_box': bool(endpoint_short),
        'partition_sizes': sizes, 'partition_components': components,
        'disconnected': max(components) > 1, 'singleton_partitions': singleton,
        'line_partitions': skinny, 'unoccupied_corners': unoccupied_corner,
        'size_ratio': max(sizes) / min(sizes),
        'crossing': ('payload' if (ar, ac) in cells else 'background') if len(sides) == 2 else 'none',
        'deleted': sum(len(v) for corner, v in groups.items() if not set(corner) & sides),
    }


def _grow(rng, cells, bounds, count, thin=False):
    """Attach cells without imposing a two-dimensional seed on a partition."""
    cells = set(cells)
    r0, r1, c0, c1 = bounds
    for _ in range(max(0, count - len(cells))):
        frontier = sorted({q for p in cells for q in _neighbors(p)
                           if r0 <= q[0] <= r1 and c0 <= q[1] <= c1 and q not in cells})
        if not frontier:
            break
        if thin:
            degree = [sum(q in cells for q in _neighbors(p)) for p in frontier]
            best = min(degree)
            frontier = [p for p, degree in zip(frontier, degree) if degree == best]
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _lobe_shape(rng, h, w, double):
    ar = int(rng.integers(1, h - 1)) if double else None
    ac = int(rng.integers(1, w - 1))
    cells = {(r, ac) for r in range(h)}
    if double:
        cells |= {(ar, c) for c in range(w)}
    row_bands = [(0, ar - 1), (ar + 1, h - 1)] if double else [(0, h - 1)]
    for r0, r1 in row_bands:
        for c0, c1 in [(0, ac - 1), (ac + 1, w - 1)]:
            r = (r1 if r1 < ar else r0) if double else int(rng.integers(h))
            c = c1 if c1 < ac else c0
            area = (r1-r0+1)*(c1-c0+1)
            # Log-uniform size spans singleton branches through substantial lobes.
            size = int(np.exp(rng.uniform(0, np.log(min(45, area) + 1))))
            cells |= _grow(rng, {(r, c)}, (r0, r1, c0, c1), size, rng.random() < .5)
    return cells, ar, ac


def _winding_shape(rng, h, w):
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    if rng.random() < .65:
        # Independently placed waypoints form detours, loops, and unequal branches.
        for _ in range(int(rng.integers(3, 12))):
            r, c = sorted(cells)[int(rng.integers(len(cells)))]
            tr, tc = int(rng.integers(h)), int(rng.integers(w))
            while (r, c) != (tr, tc):
                if r != tr and (c == tc or rng.random() < .5):
                    r += 1 if tr > r else -1
                else:
                    c += 1 if tc > c else -1
                cells.add((r, c))
        count = len(cells) + int(rng.integers(0, max(2, h*w//5)))
    else:
        count = int(rng.integers(6, max(7, min(110, h*w*3//4))))
    return _grow(rng, cells, (0, h-1, 0, w-1), count, rng.random() < .75)


def _fallback_shape(double, simple, variant=0):
    # A connected five-cell-wide witness with two separated upper-right fragments.
    if simple:
        cells = {(r, 2) for r in range(5)} | {(1, 1), (1, 3), (3, 1), (3, 3)}
        if double:
            cells |= {(2, c) for c in range(5)}
        else:
            cells |= {(2, 0), (2, 1), (2, 3), (2, 4)}
        optional = [(0, 1), (0, 3), (4, 1), (4, 3)]
    else:
        cells = {(0, 1), (1, 1), (0, 2), (0, 3), (1, 4),
                 (3, 1), (3, 2), (3, 3), (4, 1)} | {(2, c) for c in range(5)}
        optional = [(0, 0), (1, 0), (4, 0), (4, 3)]
    cells |= {point for k, point in enumerate(optional) if variant & (1 << k)}
    return cells, 2 if double else None, 2


def _sample_shape(rng, height, width, double, simple, witness):
    for _ in range(32):
        h = int(rng.integers(3, min(17, height - 1)))
        w = int(rng.integers(3, min(17, width - 1)))
        if simple:
            cells, ar, ac = _lobe_shape(rng, h, w, double)
            candidates = [(ar, ac)]
        else:
            cells = _winding_shape(rng, h, w)
            rs, cs = zip(*cells)
            candidates = [(ar, ac) for ac in range(min(cs)+1, max(cs))
                          for ar in (range(min(rs)+1, max(rs)) if double else [None])]
            rng.shuffle(candidates)
        rs, cs = zip(*cells)
        if len(cells) * 2 + (2 if double else 1) >= height * width:
            continue
        if len(cells) == (max(rs)-min(rs)+1)*(max(cs)-min(cs)+1):
            continue
        for ar, ac in candidates:
            f = _cut_features(cells, ar, ac, {'top', 'right'} if double else {'top'})
            if f is None or (witness and (not f[1] or max(f[3]) < 2)):
                continue
            if witness and not _short_endpoint(cells, ar, ac):
                continue
            # Translate local coordinates to a tight crop before choosing placement.
            r0, c0 = min(rs), min(cs)
            return ({(r-r0, c-c0) for r,c in cells},
                    ar-r0 if ar is not None else None, ac-c0)
    return _fallback_shape(double, simple)


def _draw_pair(rng, dimensions, orientation, roles, simple, witness, fallback=None):
    turn, double = orientation % 4, orientation >= 4
    height, width = dimensions[::-1] if turn % 2 else dimensions
    cells, ar, ac = (_sample_shape(rng, height, width, double, simple, witness)
                     if fallback is None else _fallback_shape(double, simple, fallback))
    ph, pw = max(r for r,c in cells)+1, max(c for r,c in cells)+1
    r0 = int(rng.integers(1, height-ph))
    c0 = int(rng.integers(1, width-pw))
    bg, payload, marker = roles
    grid = np.full((height, width), bg, dtype=np.int8)
    for r,c in cells:
        grid[r+r0, c+c0] = payload
    grid[0, ac+c0] = marker
    if double:
        grid[ar+r0, -1] = marker
    grid = np.rot90(grid, -turn).copy()
    return {'input': grid, 'output': _solve_grid(grid)}


def _schedules(rng, ntrain, ntest):
    mode = int(rng.integers(4))
    if mode == 0:
        orientation = int(rng.integers(8))
        return [orientation]*ntrain, [orientation]*ntest
    if mode == 1:
        pool = [int(v) for v in rng.choice(8, size=int(rng.integers(2, min(ntrain, 5)+1)), replace=False)]
        train = pool + [int(rng.choice(pool)) for _ in range(ntrain-len(pool))]
        rng.shuffle(train)
        return train, [int(rng.choice(pool)) for _ in range(ntest)]
    pool = [int(rng.integers(4)), int(rng.integers(4, 8))]
    train = pool + [int(rng.choice(pool if mode == 3 else list(range(8)))) for _ in range(ntrain-2)]
    rng.shuffle(train)
    choices = [v for v in range(8) if v not in train] if mode == 3 else list(range(8))
    return train, [int(rng.choice(choices)) for _ in range(ntest)]


def _dimension(rng, low, high):
    h, w = (int(v) for v in rng.integers(low, high+1, size=2))
    if rng.random() < .23:
        w = h
    return h, w


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(seed, num_train, num_test)
    train_schedule, test_schedule = _schedules(rng, ntrain, ntest)
    # These are sampling regimes, not hidden transformation parameters.
    size_mode = int(rng.integers(5))  # overlap, equal, grow, shrink, source-style
    palette_varies = rng.random() < .72
    roles = tuple(int(v) for v in rng.choice(10, 3, replace=False))
    complexity = int(rng.choice(3, p=[.18, .42, .40]))
    equal_dimensions = _dimension(rng, 7, 30)
    small, large = _dimension(rng, 7, 17), _dimension(rng, 20, 30)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, regime, schedule in [('train', 'source_train', train_schedule),
                                    ('test', 'target_test', test_schedule)]:
        for index, orientation in enumerate(schedule):
            train = regime == 'source_train'
            if size_mode == 0:
                dimensions = _dimension(rng, 7, 30)
            elif size_mode == 1:
                dimensions = equal_dimensions
            elif size_mode == 2:
                dimensions = _dimension(rng, 7, 17) if train else _dimension(rng, 20, 30)
            elif size_mode == 3:
                dimensions = _dimension(rng, 20, 30) if train else _dimension(rng, 7, 17)
            else:
                dimensions = small if train else large
            if palette_varies:
                available = [c for c in range(10) if c != roles[0]]
                bg = int(rng.choice(available))
                others = [c for c in range(10) if c != bg]
                roles = (bg, *(int(v) for v in rng.choice(others, 2, replace=False)))
            witness = complexity > 0 and train and index == 0
            simple = complexity == 0 or (not witness and rng.random() < (.5 if complexity == 1 else .15))
            for attempt in range(48):
                pair = _draw_pair(rng, dimensions, orientation, roles, simple, witness,
                                  None if attempt < 32 else attempt-32)
                key = _grid_key(pair['input'])
                if key not in seen and (key, _grid_key(pair['output'])) not in _SOURCE_KEYS:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('exhausted distinct deterministic fallback geometries')
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict_grid(grid, nearest_only=False, separate_fragments=False):
    """An alternate raster/crop implementation; receives no generation metadata."""
    h, w = grid.shape
    rim = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    colors, counts = np.unique(rim, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    mark = int(next(c for c in colors if c != bg))
    obj = int(next(c for c in np.unique(grid) if c not in (bg, mark)))
    mask = grid == obj
    out = np.full_like(grid, bg)
    row_axis = col_axis = None
    selected = set()
    for r,c in np.argwhere(grid == mark):
        dr,dc,side = ((1,0,'top') if r == 0 else (-1,0,'bottom') if r == h-1
                      else (0,1,'left') if c == 0 else (0,-1,'right'))
        selected.add(side)
        if dr:
            col_axis = int(c)
        else:
            row_axis = int(r)
        ray = []
        while 0 <= r < h and 0 <= c < w:
            ray.append((int(r),int(c)))
            r,c = r+dr,c+dc
        hits = [i for i,p in enumerate(ray) if mask[p]]
        stop = min(hits) if nearest_only else max(hits)
        for p in ray[:stop+1]:
            out[p] = obj if mask[p] else mark
    fixed = np.zeros_like(mask)
    if row_axis is not None:
        fixed[row_axis, :] = mask[row_axis, :]
    if col_axis is not None:
        fixed[:, col_axis] |= mask[:, col_axis]
    out[fixed] = obj
    # The known defective nearest-hit rule separately paints the empty cross center.
    if nearest_only and row_axis is not None and col_axis is not None and not mask[row_axis, col_axis]:
        out[row_axis, col_axis] = mark
    rr, cc = np.indices(grid.shape)
    for v in ('top','bottom'):
        for s in ('left','right'):
            if v not in selected and s not in selected:
                continue
            region = mask & ~fixed
            if row_axis is None:
                if v not in selected:
                    continue
            else:
                region &= rr < row_axis if v == 'top' else rr > row_axis
            if col_axis is None:
                if s not in selected:
                    continue
            else:
                region &= cc < col_axis if s == 'left' else cc > col_axis
            sets = (_components(zip(*np.nonzero(region))) if separate_fragments
                    else [set(zip(*np.nonzero(region)))])
            for cells in sets:
                if not cells:
                    continue
                rs,cs = zip(*cells)
                cut = np.zeros((max(rs)-min(rs)+1,max(cs)-min(cs)+1), dtype=bool)
                for r,c in cells:
                    cut[r-min(rs),c-min(cs)] = True
                r0 = 0 if v == 'top' else h-cut.shape[0]
                c0 = 0 if s == 'left' else w-cut.shape[1]
                target = out[r0:r0+cut.shape[0], c0:c0+cut.shape[1]]
                if not separate_fragments:
                    assert np.all(target[cut] == bg)
                target[cut] = obj
    return out


def predict(train_pairs, test_input):
    """Infer with visible roles; check that demonstrations support the same rule."""
    for pair in train_pairs:
        assert np.array_equal(_predict_grid(pair['input']), pair['output'])
    return _predict_grid(test_input)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train','test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, pairs, features = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {'input', 'output'}
            for g in pair.values():
                assert isinstance(g, np.ndarray) and g.ndim == 2
                assert np.issubdtype(g.dtype, np.integer)
                assert all(1 <= n <= 30 for n in g.shape)
                assert np.all((0 <= g) & (g <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key, pair_key = _grid_key(inp), (_grid_key(inp), _grid_key(out))
            assert key not in seen and pair_key not in pairs and pair_key not in _SOURCE_KEYS
            seen.add(key)
            pairs.add(pair_key)
            f = pair_features(inp)
            assert np.array_equal(_solve_grid(inp), out)
            assert set(np.unique(out)) == set(f['roles'])
            assert np.count_nonzero(out == f['roles'][1]) == f['payload_size'] - f['deleted']
            features[split].append(f)
    train, test = features['train'], features['test']
    # Visible palette roles and rotational geometry are invariant; no palette/size bans.
    assert {f['marker_count'] for f in test} <= {f['marker_count'] for f in train}
    if any(f['marker_count'] == 2 for f in test):
        assert any(f['marker_count'] == 2 and f['deleted'] > 0 for f in train)
    if any(f['gap_lengths'] for f in test):
        assert any(f['gap_lengths'] for f in train), 'missing farthest-hit training witness'
    if any(f['disconnected'] for f in test):
        assert any(f['disconnected'] for f in train), 'missing collective-translation witness'
    if any(f['endpoint_short_of_box'] for f in test):
        assert any(f['endpoint_short_of_box'] for f in train), 'missing axial-endpoint witness'


def self_test() -> None:
    sources = [{'input': _decode(a), 'output': _decode(b)} for a,b in _SOURCE_PAIRS]
    for p in sources:
        assert np.array_equal(_solve_grid(p['input']), p['output'])
        assert np.array_equal(_predict_grid(p['input']), p['output'])
    assert sum(not np.array_equal(_predict_grid(p['input'], nearest_only=True), p['output'])
               for p in sources) == 3
    assert not np.array_equal(_predict_grid(sources[0]['input'], separate_fragments=True), sources[0]['output'])
    assert np.array_equal(predict(sources[:3], sources[3]['input'].copy()), sources[3]['output'])
    # Boundary construction is deterministic even when random proposals are exhausted.
    for orientation in range(8):
        for simple in (False, True):
            for variant in range(16):
                pair = _draw_pair(np.random.default_rng(variant), (7, 7), orientation,
                                  (1, 7, 2), simple, not simple, variant)
                f = pair_features(pair['input'])
                assert np.array_equal(_predict_grid(pair['input']), pair['output'])
                if not simple:
                    assert f['gap_lengths'] and f['disconnected'] and f['endpoint_short_of_box']
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert _puzzles_equal(generate_puzzle(271828, 6, 2), generate_puzzle(271828, 6, 2))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    for train, test in [(2,1),(10,1),(10,4),(7,3),(None,1),(None,4),(2,None),(10,None)]:
        p = generate_puzzle(314159, train, test)
        assert train is None or len(p['train']) == train
        assert test is None or len(p['test']) == test
        validate_generated_puzzle(p)
    for train, test in [(1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(True,1),(3.5,1)]:
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    all_features, per_puzzle, structures = [], [], set()
    orientations = {'train':set(), 'test':set()}
    mutants = [0,0]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        fs = []
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp,out = pair['input'],pair['output']
                f = pair_features(inp)
                fs.append(f)
                orientations[split].add(f['orientation'])
                assert np.array_equal(_predict_grid(inp),out)
                if split == 'test':
                    # Prediction gets only copied demonstrations and this input.
                    training = [{k:v.copy() for k,v in p.items()} for p in puzzle['train']]
                    predicted = predict(training, inp.copy())
                    assert np.array_equal(predicted, out)
                mutants[0] += not np.array_equal(_predict_grid(inp, nearest_only=True),out)
                mutants[1] += not np.array_equal(_predict_grid(inp, separate_fragments=True),out)
                coords = np.argwhere(inp == f['roles'][1])
                coords -= coords.min(axis=0)
                structures.add(tuple(map(tuple,coords)))
        all_features.extend(fs)
        per_puzzle.append(fs)
    for split in orientations:
        assert len(orientations[split]) == 8
    assert len(structures) >= len(all_features)*.8
    assert len({tuple(f['shape']) for f in all_features}) >= 60
    assert min(min(f['shape']) for f in all_features) == 7
    assert max(max(f['shape']) for f in all_features) == 30
    assert {f['roles'][0] for f in all_features} == set(range(10))
    assert sum(bool(f['gap_lengths']) for f in all_features) > 50
    assert sum(f['marker_count'] == 2 and f['noncentral_gap_cells'] > 0
               for f in all_features) > 20
    assert sum(f['disconnected'] for f in all_features) > 50
    assert sum(f['singleton_partitions'] for f in all_features) > 30
    assert sum(f['line_partitions'] for f in all_features) > 50
    assert max(f['size_ratio'] for f in all_features) >= 10
    assert max(g for f in all_features for g in f['gap_lengths']) >= 5
    assert {1, 2, 3, 4}.issubset({n for f in all_features for n in f['axial_run_lengths']})
    assert max(c for f in all_features for c in f['partition_components']) >= 3
    assert {f['crossing'] for f in all_features} == {'none','payload','background'}
    assert sum(f['endpoint_short_of_box'] for f in all_features) > 20
    assert min(g for f in all_features for g in f['guide_gaps']) == 0
    assert max(g for f in all_features for g in f['guide_gaps']) >= 15
    assert sum(f['unoccupied_corners'] for f in all_features) > 50
    varied = [fs for fs in per_puzzle if len({tuple(f['roles']) for f in fs}) > 1]
    assert 10 <= len(varied) <= 40
    assert sum(len({f['orientation'] for f in fs}) > 1 for fs in per_puzzle) > 15
    assert sum(len({tuple(f['shape']) for f in fs}) > 1 for fs in per_puzzle) > 15
    assert sum(any(f['gap_lengths'] and f['disconnected'] and f['shape'][0] != f['shape'][1]
                   for f in fs) and len({f['marker_count'] for f in fs}) == 2 for fs in varied) >= 5
    growth = shrinkage = equal_size = held_out = 0
    for puzzle, fs in zip(puzzles, per_puzzle):
        ntrain = len(puzzle['train'])
        tr, te = fs[:ntrain], fs[ntrain:]
        growth += min(min(f['shape']) for f in te) > max(max(f['shape']) for f in tr)
        shrinkage += max(max(f['shape']) for f in te) < min(min(f['shape']) for f in tr)
        equal_size += len({tuple(f['shape']) for f in fs}) == 1
        held_out += bool({f['orientation'] for f in te} - {f['orientation'] for f in tr})
    assert min(growth, shrinkage, equal_size, held_out) >= 3
    assert sum(len({f['orientation'] for f in fs}) == 1 for fs in per_puzzle) >= 3
    assert sum(all(not f['gap_lengths'] and not f['disconnected'] for f in fs)
               for fs in per_puzzle) >= 3
    relative_axes = [coordinate / (f['shape'][axis]-1)
                     for f in all_features for axis, coordinate in enumerate(f['axes'])
                     if coordinate is not None]
    assert min(relative_axes) < .2 and max(relative_axes) > .8
    assert min(mutants) > 50
