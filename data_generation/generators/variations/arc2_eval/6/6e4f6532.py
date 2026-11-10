"""Registration docking with pair-local color compasses and rigid D4 motion.

Construction samples solved pieces and places their stencils separately from
scrambled sources. An input-only exhaustive solver certifies the resulting
pixels, including assignments between repeated stencils. No generation state
is consulted by the validator or predictor.
"""
import hashlib
import numpy as np


_DIRECTIONS = {"top": (-1, 0), "right": (0, 1),
               "bottom": (1, 0), "left": (0, -1)}
_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                   if r or c)
_SOURCE_DIGESTS = {
    "5088c42ba9cee39f27971e24f270ee1ffaedf8d402f0569a29170b4fc7aef17f",
    "20c6051a8b5b8b0519f4c52bd7e30fb09d306085d6d0cd4fad0b1a1cb0d1c7ad",
    "c03bfe2e1be5dd6a1a69e3c181a9ea162bfdddd0c679f9df43dacfde4f71604f",
    "187ca79e69178898473abb31095fa507980017ecc561ca81e92151fc268fc7a5",
}


def _validate_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _validate_count(num_train, "num_train", 2, 10)
    num_test = _validate_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        span = 10 - num_test
        num_train = num_test + 1 + (seed // 4) % span
    elif num_train is None:
        span = 10 - num_test
        num_train = num_test + 1 + seed % span
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + seed % maximum_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _transform_colored(cells, transform):
    """Apply one of the eight dihedral transformations and normalize."""
    transformed = []

    for (row, col), value in cells.items():
        r, c = int(row), int(col)

        if transform >= 4:
            c = -c

        for _ in range(transform % 4):
            r, c = c, -r

        transformed.append((r, c, int(value)))

    min_row = min(item[0] for item in transformed)
    min_col = min(item[1] for item in transformed)

    return {
        (row - min_row, col - min_col): value
        for row, col, value in transformed
    }


def _transform_points(points, transform):
    colored = {tuple(point): 1 for point in points}
    return set(_transform_colored(colored, transform))


def _normalized_points(points):
    points = set(points)
    min_row = min(row for row, _ in points)
    min_col = min(col for _, col in points)
    return tuple(
        sorted((row - min_row, col - min_col) for row, col in points)
    )


def _signature(cells):
    normalized = _transform_colored(cells, 0)
    return tuple(
        sorted((row, col, value) for (row, col), value in normalized.items())
    )


def _canonical_signature(cells):
    return min(
        tuple(
            sorted(
                (row, col, value)
                for (row, col), value
                in _transform_colored(cells, transform).items()
            )
        )
        for transform in range(8)
    )


def _translated(cells, origin):
    return {(r + origin[0], c + origin[1]): v for (r, c), v in cells.items()}


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _components(mask, diagonal=False):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    neighbors = _NEIGHBORS if diagonal else tuple(_DIRECTIONS.values())
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    stack.append(point)
        result.append(component)
    return result


def _groups(indices):
    groups = []
    for i in sorted(indices):
        if not groups or i != groups[-1][-1] + 1:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def _detect_structure(grid, background):
    """Find full bands, including bands interrupted by a crossing band."""
    h, w = grid.shape
    rows, cols = set(), set()
    for _ in range(3):
        active_c = [c for c in range(w) if c not in cols]
        for r in range(h):
            values = grid[r, active_c]
            if len(values) and values[0] != background and np.all(values == values[0]):
                rows.add(r)
        active_r = [r for r in range(h) if r not in rows]
        for c in range(w):
            values = grid[active_r, c]
            if len(values) and values[0] != background and np.all(values == values[0]):
                cols.add(c)
    assert rows or cols, "no structural bands"
    assert len(rows) < h and len(cols) < w
    assert all(len(g) == 2 for g in _groups(rows) + _groups(cols))
    structural = np.zeros(grid.shape, dtype=bool)
    structural[list(rows), :] = True
    structural[:, list(cols)] = True
    internal_r = [g for g in _groups(rows) if g[0] > 0 and g[-1] < h - 1]
    internal_c = [g for g in _groups(cols) if g[0] > 0 and g[-1] < w - 1]
    assert not (internal_r and internal_c), "crossed internal dividers"
    if internal_r:
        layout = "horizontal"
    elif internal_c:
        layout = "vertical"
    else:
        layout = "frame" if len(rows) == len(cols) == 4 else "partial"
    return structural, layout


def _region_information(grid, structural):
    regions = _components(~structural)
    region_index = np.full(grid.shape, -1, dtype=np.int16)
    side_colors = []
    for index, region in enumerate(regions):
        for r, c in region:
            region_index[r, c] = index
        colors = {direction: set() for direction in _DIRECTIONS}
        for r, c in region:
            for direction, (dr, dc) in _DIRECTIONS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and structural[nr, nc]:
                    colors[direction].add(int(grid[nr, nc]))
        side_colors.append(colors)
    return regions, region_index, side_colors


def _label_directions(cells, band_colors):
    """Labels project beyond the body/registration envelope on their side.

    Several patches may name a side; their positions along that side do not
    encode a second rule. Corners of the core are never used as compass labels.
    """
    core = [p for p, v in cells.items() if v not in band_colors]
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    result = {}
    for (r, c), value in cells.items():
        if value in band_colors:
            directions = []
            if r < r0:
                directions.append("top")
            if r > r1:
                directions.append("bottom")
            if c < c0:
                directions.append("left")
            if c > c1:
                directions.append("right")
            if len(directions) != 1:
                return None
            result[(r, c)] = directions[0]
    return result


def _parse_input(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert sum(counts == counts[background]) == 1, "background is ambiguous"
    structural, layout = _detect_structure(grid, background)
    band_colors = set(int(v) for v in grid[structural])
    objects = _components((grid != background) & ~structural, diagonal=True)
    sources, targets, target_colors = [], [], set()
    for component in objects:
        cells = {p: int(grid[p]) for p in component}
        values = set(cells.values())
        if len(values) == 1:
            targets.append(component)
            target_colors.update(values)
        else:
            sources.append(cells)
    assert sources and len(sources) == len(targets)
    assert len(target_colors) == 1
    registration = next(iter(target_colors))
    body_colors = set()
    for cells in sources:
        values = set(cells.values())
        assert registration in values and values & band_colors
        assert _label_directions(cells, band_colors)
        body_colors.update(values - band_colors - {registration})
    assert len(body_colors) == 1
    body = next(iter(body_colors))
    assert len({background, registration, body} | band_colors) == len(band_colors) + 3
    regions, region_index, sides = _region_information(grid, structural)
    return {"background": background, "body": body, "registration": registration,
            "structural": structural, "band_colors": band_colors,
            "sources": sources, "targets": targets, "regions": regions,
            "region_index": region_index, "sides": sides, "layout": layout}


def _candidate_placements(source, info, use_labels=True):
    candidates, seen = [], set()
    reg = info["registration"]
    h, w = info["structural"].shape
    for transform in range(8):
        cells = _transform_colored(source, transform)
        marks = {p for p, v in cells.items() if v == reg}
        directions = _label_directions(cells, info["band_colors"])
        if not directions:
            continue
        # Conservatively allow either outward direction at a bounding corner.
        # A candidate must be unique even under this looser source-compatible
        # reading; generation never needs a hidden reflection tie-break.
        rmax = max(r for r, _ in cells)
        cmax = max(c for _, c in cells)
        facing = {p: [d for d, ok in (("top", p[0] == 0),
                                      ("bottom", p[0] == rmax),
                                      ("left", p[1] == 0),
                                      ("right", p[1] == cmax)) if ok]
                  for p in directions}
        first = min(marks)
        for index, target in enumerate(info["targets"]):
            if len(marks) != len(target):
                continue
            start = min(target)
            delta = (start[0] - first[0], start[1] - first[1])
            shifted = {(r + delta[0], c + delta[1]) for r, c in marks}
            if shifted != target:
                continue
            placed = _translated(cells, delta)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in placed):
                continue
            region_ids = {int(info["region_index"][p]) for p in placed}
            if len(region_ids) != 1 or -1 in region_ids:
                continue
            sides = info["sides"][next(iter(region_ids))]
            if use_labels and any(not any(cells[p] in sides[d] for d in ds)
                                  for p, ds in facing.items()):
                continue
            signature = tuple(sorted(placed.items()))
            if signature not in seen:
                seen.add(signature)
                candidates.append((index, placed))
    return candidates


def _solve_maps(info, use_labels=True, limit=2):
    """Enumerate distinct resulting pixel maps, not transformation IDs."""
    choices = [_candidate_placements(source, info, use_labels) for source in info["sources"]]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    solutions = {}

    def search(depth, used, occupied):
        if len(solutions) >= limit:
            return
        if depth == len(order):
            signature = tuple(sorted(occupied.items()))
            solutions[signature] = dict(occupied)
            return
        for target, cells in choices[order[depth]]:
            if target in used or occupied.keys() & cells.keys():
                continue
            search(depth + 1, used | {target}, {**occupied, **cells})
    search(0, set(), {})
    return list(solutions.values())


def _predict(input_grid):
    info = _parse_input(input_grid)
    solutions = _solve_maps(info)
    assert len(solutions) == 1, "input does not determine a unique pixel output"
    predicted = np.full_like(input_grid, info["background"])
    predicted[info["structural"]] = input_grid[info["structural"]]
    for p, v in solutions[0].items():
        predicted[p] = v
    return predicted, info


def predict_from_training(train_pairs, test_input):
    """Prediction API: only demonstrations and a test INPUT are accepted."""
    assert len(train_pairs) >= 2
    # Verify the demonstrated convention before applying it to the unseen grid.
    for pair in train_pairs:
        predicted, _ = _predict(pair["input"])
        assert np.array_equal(predicted, pair["output"])
    return _predict(test_input)[0]


def _pattern(rng, max_h, max_w, simple=False):
    common = [({(0, 0)}), {(0, 0), (0, 1)}, {(0, 0), (1, 1)},
              {(0, 0), (0, 1), (1, 0), (1, 1)}]
    if simple or rng.random() < 0.42:
        points = common[int(rng.integers(len(common)))]
    else:
        # A random polyomino / diagonally connected cluster, not a motif bank.
        points = {(0, 0)}
        size = int(rng.integers(2, 7))
        steps = _NEIGHBORS if rng.random() < 0.35 else tuple(_DIRECTIONS.values())
        for _ in range(40):
            if len(points) >= size:
                break
            r, c = sorted(points)[int(rng.integers(len(points)))]
            dr, dc = steps[int(rng.integers(len(steps)))]
            trial = points | {(r + dr, c + dc)}
            if (max(r for r, _ in trial) - min(r for r, _ in trial) < max_h
                    and max(c for _, c in trial) - min(c for _, c in trial) < max_w):
                points = trial
    poses = [_transform_points(points, t) for t in range(8)]
    poses = [p for p in poses if max(r for r, _ in p) < max_h and max(c for _, c in p) < max_w]
    return poses[int(rng.integers(len(poses)))]


def _path(rng, start, end):
    r, c = start
    points = {(r, c)}
    while (r, c) != end:
        if r != end[0] and (c == end[1] or rng.random() < 0.5):
            r += 1 if end[0] > r else -1
        else:
            c += 1 if end[1] > c else -1
        points.add((r, c))
    return points


def _make_piece(rng, sides, palette, max_h, max_w, size_limit,
                simple=False, repeated=None):
    bg, body, reg = palette[:3]
    available = list(sides)
    number = int(rng.integers(1, len(available) + 1))
    if simple:
        number = max(2, number)
    selected = list(rng.choice(available, size=min(number, len(available)), replace=False))
    extra_h = int("top" in selected) + int("bottom" in selected)
    extra_w = int("left" in selected) + int("right" in selected)
    cap_h, cap_w = min(size_limit, max_h - extra_h), min(size_limit, max_w - extra_w)
    if min(cap_h, cap_w) < 2:
        return None
    h, w = int(rng.integers(2, cap_h + 1)), int(rng.integers(2, cap_w + 1))
    if h * w < 6:
        return None
    if repeated is None:
        pattern = _pattern(rng, h, w, simple)
    else:
        fits = [_transform_points(repeated, t) for t in range(8)]
        fits = [p for p in fits if max(r for r, _ in p) < h and max(c for _, c in p) < w]
        if not fits:
            return None
        pattern = fits[int(rng.integers(len(fits)))]
    ph, pw = 1 + max(r for r, _ in pattern), 1 + max(c for _, c in pattern)
    offset = (int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1)))
    marks = {(r + offset[0], c + offset[1]) for r, c in pattern}
    labels = {}
    supports = []
    for d in selected:
        length = w if d in ("top", "bottom") else h
        patch = int(rng.integers(1, min(3, length) + 1))
        start = int(rng.integers(length - patch + 1))
        positions = set(range(start, start + patch))
        if length >= 3 and rng.random() < 0.3:
            positions = set(int(x) for x in rng.choice(length, size=min(3, length - 1), replace=False))
        for i in positions:
            p, q = {"top": ((-1, i), (0, i)), "bottom": ((h, i), (h - 1, i)),
                    "left": ((i, -1), (i, 0)), "right": ((i, w), (i, w - 1))}[d]
            labels[p] = sides[d]
            supports.append(q)
    # Independent endpoints establish unequal extents and thin branches.
    endpoints = supports + sorted(marks) + [
        (0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
        (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
    # Connect from one seed, including diagonal registration cells, so the
    # complete composite is four-connected. Short attachments retain thin
    # branches and open indentations; occasional longer routes add loops.
    core = {min(marks)}
    sparse = rng.random() < .45
    rng.shuffle(endpoints)
    for endpoint in endpoints:
        if sparse or rng.random() < .55:
            distance = min(abs(p[0] - endpoint[0]) + abs(p[1] - endpoint[1]) for p in core)
            anchors = sorted(p for p in core if abs(p[0] - endpoint[0]) + abs(p[1] - endpoint[1]) == distance)
        else:
            anchors = sorted(core)
        anchor = anchors[int(rng.integers(len(anchors)))]
        core.update(_path(rng, anchor, endpoint))
    density = float(rng.uniform(.28, .48) if sparse else rng.uniform(.55, .93))
    for _ in range(h * w):
        if len(core) >= h * w * density:
            break
        frontier = sorted({(r + dr, c + dc) for r, c in core for dr, dc in _DIRECTIONS.values()
                           if 0 <= r + dr < h and 0 <= c + dc < w} - core)
        if not frontier:
            break
        core.add(frontier[int(rng.integers(len(frontier)))])
    if len(core - marks) < 3:
        return None
    return _transform_colored({**{p: body for p in core},
                               **{p: reg for p in marks}, **labels}, 0)


def _sample_scene(rng, layout, count, palette, level):
    # Size depends on packing load, while orientation and aspect remain free.
    low = 12 if count == 1 else 16 if count == 2 else 21 if count == 3 else 25
    h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    if layout in ("vertical", "horizontal"):
        lane_count = 3 if level == 2 and count >= 3 and rng.random() < 0.25 else 2
        long_dim = int(rng.integers(max(20, low), 31))
        across = int(rng.integers(max(18, low, lane_count * 5 + 2 * (lane_count + 1)), 31))
        if layout == "vertical":
            h, w = long_dim, across
        else:
            h, w = across, long_dim
        # Narrow/tall and short/wide cases still have adequate packing area.
        if count <= 2 and rng.random() < 0.35:
            if layout == "horizontal":
                w = int(rng.integers(12, 20))
            else:
                h = int(rng.integers(12, 20))
    else:
        lane_count = 1
    base = np.full((h, w), palette[0], dtype=np.int8)
    colors = iter(palette[3:])
    vertical, horizontal = [], []
    if layout in ("frame", "partial"):
        selected = list(_DIRECTIONS) if layout == "frame" else list(rng.choice(list(_DIRECTIONS), int(rng.integers(2, 4)), replace=False))
        for d in selected:
            color = next(colors)
            if d in ("left", "right"):
                vertical.append((0 if d == "left" else w - 2, color))
            else:
                horizontal.append((0 if d == "top" else h - 2, color))
    else:
        extent = w if layout == "vertical" else h
        free = extent - 2 * (lane_count + 1)
        lengths = [5] * lane_count
        for _ in range(free - sum(lengths)):
            lengths[int(rng.integers(lane_count))] += 1
        # Redistribute to make off-center dividers common, including empty lanes.
        for _ in range(4):
            a, b = int(rng.integers(lane_count)), int(rng.integers(lane_count))
            if lengths[a] > 5:
                lengths[a] -= 1
                lengths[b] += 1
        positions = [0]
        for length in lengths:
            positions.append(positions[-1] + 2 + length)
        bands = vertical if layout == "vertical" else horizontal
        bands.extend((position, next(colors)) for position in positions)
        caps = int(rng.choice(3, p=[0.35, 0.5, 0.15]))
        cap_positions = [0, h - 2] if layout == "vertical" else [0, w - 2]
        rng.shuffle(cap_positions)
        other = horizontal if layout == "vertical" else vertical
        other.extend((position, next(colors)) for position in cap_positions[:caps])
    for c, color in vertical:
        base[:, c:c + 2] = color
    for r, color in horizontal:
        base[r:r + 2, :] = color
    structural, _ = _detect_structure(base, palette[0])
    regions, region_index, sides = _region_information(base, structural)
    rectangles = []
    for region, side in zip(regions, sides):
        rectangles.append((min(r for r, _ in region), max(r for r, _ in region) + 1,
                           min(c for _, c in region), max(c for _, c in region) + 1,
                           {d: next(iter(cs)) for d, cs in side.items() if cs}))
    return base, rectangles, structural, region_index


def _halo(points):
    return {(r + dr, c + dc) for r, c in points for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _origins(cells, rectangles):
    h = 1 + max(r for r, _ in cells)
    w = 1 + max(c for _, c in cells)
    return [(r, c) for r0, r1, c0, c1, _ in rectangles
            for r in range(r0, r1 - h + 1) for c in range(c0, c1 - w + 1)]


def _construct_pair(rng, palette, layout, level, count, demonstration=None):
    base, rectangles, structural, region_index = _sample_scene(rng, layout, count, palette, level)
    finals, final_blocked = [], set()
    repeated = None
    repeat_mode = rng.random() < 0.3 and count > 1
    for i in range(count):
        found = False
        for _ in range(35):
            region = rectangles[int(rng.integers(len(rectangles)))]
            r0, r1, c0, c1, sides = region
            size_limit = int(rng.choice([3, 4, 5, 6, 7, 8], p=[.10, .16, .25, .21, .16, .12]))
            if level == 0:
                size_limit = int(rng.integers(3, 6))
            piece = _make_piece(rng, sides, palette, r1 - r0, c1 - c0, size_limit,
                                simple=(demonstration == "rotate" and i == 0),
                                repeated=repeated if repeat_mode and i else None)
            if piece is None:
                continue
            origins = _origins(piece, [region])
            rng.shuffle(origins)
            for origin in origins:
                placed = _translated(piece, origin)
                if placed.keys() & final_blocked:
                    continue
                finals.append(placed)
                final_blocked.update(_halo(placed))
                if i == 0:
                    repeated = set(_normalized_points(p for p, v in piece.items() if v == palette[2]))
                found = True
                break
            if found:
                break
        if not found:
            return None
    targets = {p: palette[2] for piece in finals for p, v in piece.items() if v == palette[2]}
    blocked = _halo(targets)
    final_cells = set().union(*(set(piece) for piece in finals))
    sources = []
    want_overlap = rng.random() < 0.45
    for i, final in enumerate(finals):
        transform_order = list(int(t) for t in rng.permutation(8))
        if i == 0 and demonstration:
            transform_order = [t for t in transform_order if (0 < t < 4 if demonstration == "rotate" else t >= 4)]
        elif rng.random() < .22:
            transform_order = [0] + [t for t in transform_order if t]
        found = False
        for transform in transform_order:
            piece = _transform_colored(final, transform)
            if i == 0 and demonstration == "reflect" and any(
                _signature(_transform_colored(piece, t)) == _signature(final)
                for t in range(4)
            ):
                continue
            origins = _origins(piece, rectangles)
            rng.shuffle(origins)
            # Evaluate a bounded random pool first. All positions remain eligible.
            candidates = []
            for origin in origins:
                placed = _translated(piece, origin)
                if placed.keys() & blocked:
                    continue
                if demonstration and i == 0 and _signature(piece) == _signature(final):
                    continue
                overlap = bool(placed.keys() & final_cells)
                candidates.append((overlap, placed))
                if overlap == want_overlap:
                    break
            if candidates:
                preferred = [p for overlap, p in candidates if overlap == want_overlap]
                placed = preferred[0] if preferred else candidates[0][1]
                sources.append(placed)
                blocked.update(_halo(placed))
                found = True
                break
        if not found:
            return None
    inp, out = base.copy(), base.copy()
    for p, value in targets.items():
        inp[p] = value
    for piece in sources:
        for p, value in piece.items():
            inp[p] = value
    # Building on the clean structural canvas implements erase-all, then paste-all.
    # Old and final footprints may intersect; only simultaneous finals conflict.
    for piece in finals:
        for p, value in piece.items():
            out[p] = value
    try:
        predicted, info = _predict(inp)
        if not np.array_equal(predicted, out):
            return None
        if demonstration == "rotate" and len(_solve_maps(info, use_labels=False)) < 2:
            return None
    except AssertionError:
        return None
    return {"input": inp, "output": out}


def _fallback(rng, palette, demonstration):
    """Guaranteed small, asymmetric piece with two adjacent compass labels."""
    base = np.full((22, 24), palette[0], dtype=np.int8)
    base[:2, :] = palette[3]
    base[-2:, :] = palette[4]
    base[2:-2, :2] = palette[5]
    base[2:-2, -2:] = palette[6]
    body, reg = palette[1:3]
    cells = {(1, 1): body, (1, 2): body, (1, 3): body,
             (2, 1): body, (2, 2): reg, (3, 1): body,
             (0, 2): palette[3], (2, 0): palette[5]}
    final = _translated(cells, (int(rng.integers(3, 7)), int(rng.integers(4, 9))))
    transform = int(rng.integers(1, 4) if demonstration == "rotate" else rng.integers(4, 8))
    source = _translated(_transform_colored(cells, transform), (14, 15))
    inp, out = base.copy(), base.copy()
    for p, v in final.items():
        out[p] = v
        if v == reg:
            inp[p] = v
    for p, v in source.items():
        inp[p] = v
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Puzzle-level choices govern curriculum and palette continuity, not hidden
    # color-to-direction maps. Equality to the current visible band is universal.
    level = int(rng.choice(3, p=[.18, .57, .25]))
    regime = str(rng.choice(["shared", "mixed", "orientation", "count"]))
    family = str(rng.choice(["vertical", "horizontal", "frame", "partial"]))
    held_out = str(rng.choice(["vertical", "horizontal"]))
    shared_roles = bool(rng.random() < .60)
    role_colors = [int(v) for v in rng.choice(10, 2, replace=False)]
    puzzle, used = {"train": [], "test": []}, set()
    previous_background = None
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            colors = [int(v) for v in rng.permutation(10)]
            if shared_roles:
                colors = [c for c in colors if c not in role_colors]
                background = next(c for c in colors if c != previous_background)
                palette = [background] + role_colors + [c for c in colors if c != background]
            else:
                if colors[0] == previous_background:
                    colors[0], colors[3] = colors[3], colors[0]
                palette = colors
            previous_background = palette[0]
            if regime == "shared" or level == 0:
                layout = family
            elif regime == "orientation":
                layout = held_out if split == "test" else str(rng.choice([l for l in ("vertical", "horizontal", "frame", "partial") if l != held_out]))
            else:
                layout = str(rng.choice(["vertical", "horizontal", "frame", "partial"]))
            maximum = 2 if level == 0 else 4 if level == 1 else 5
            if regime == "count":
                count = int(rng.integers(1, 3) if split == "train" else rng.integers(3, 6))
            else:
                count = int(rng.integers(1, maximum + 1))
            demonstration = ("rotate" if index == 0 else "reflect" if index == 1 else None) if split == "train" else None
            for attempt in range(100):
                pair = _construct_pair(rng, palette, layout, level, count, demonstration)
                if pair is not None and _grid_key(pair["input"]) not in used:
                    break
            else:
                pair = _fallback(rng, palette, demonstration)
                assert _grid_key(pair["input"]) not in used
            used.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle
def _inspect_pair(inp, out):
    predicted, info = _predict(inp)
    assert np.array_equal(predicted, out), "output violates registration/compass docking"
    assert not np.array_equal(inp, out)
    structural_out, _ = _detect_structure(out, info["background"])
    assert np.array_equal(structural_out, info["structural"])
    assert np.array_equal(inp[info["structural"]], out[info["structural"]])
    digest = hashlib.sha256(bytes(inp.shape) + inp.astype(np.uint8).tobytes()
                            + out.astype(np.uint8).tobytes()).hexdigest()
    assert digest not in _SOURCE_DIGESTS
    output_objects = _components((out != info["background"]) & ~info["structural"], diagonal=True)
    assert len(output_objects) == len(info["sources"])
    outputs = [{p: int(out[p]) for p in obj} for obj in output_objects]
    remaining = list(outputs)
    old = set().union(*(set(p) for p in info["sources"]))
    new = set().union(*output_objects)
    piece_shapes, densities, patterns, pattern_poses, holes = [], [], [], [], []
    sides_counts, patch_sizes, moves, distances, cross_region, offsets = [], [], [], [], [], []
    separated_patches = 0
    for source in info["sources"]:
        matches = [j for j, final in enumerate(remaining)
                   if _canonical_signature(source) == _canonical_signature(final)]
        assert matches
        final = remaining.pop(matches[0])
        transforms = [t for t in range(8) if _signature(_transform_colored(source, t)) == _signature(final)]
        moves.append("translation" if 0 in transforms else "rotation" if any(t < 4 for t in transforms) else "reflection")
        sr, sc = min(source)
        fr, fc = min(final)
        distances.append(abs(sr - fr) + abs(sc - fc))
        cross_region.append(info["region_index"][min(source)] != info["region_index"][min(final)])
        normalized = _transform_colored(final, 0)
        h, w = 1 + max(r for r, _ in normalized), 1 + max(c for _, c in normalized)
        mask = np.zeros((h, w), dtype=bool)
        for p in normalized:
            mask[p] = True
        assert len(_components(mask)) == 1, "a composite piece must be connected"
        holes.append(sum(all(0 < r < h - 1 and 0 < c < w - 1 for r, c in component)
                         for component in _components(~mask)))
        piece_shapes.append((h, w))
        densities.append(len(final) / (h * w))
        marks = {p for p, v in normalized.items() if v == info["registration"]}
        offsets.append(min(marks))
        pattern_poses.append(_normalized_points(marks))
        patterns.append(min(_normalized_points(_transform_points(marks, t)) for t in range(8)))
        directions = _label_directions(final, info["band_colors"])
        sides_counts.append(len(set(directions.values())))
        for d in set(directions.values()):
            label_points = sorted(p for p, direction in directions.items() if direction == d)
            patch_sizes.append(len(label_points))
            axis = sorted(p[1] if d in ("top", "bottom") else p[0] for p in label_points)
            separated_patches += int(any(b > a + 1 for a, b in zip(axis, axis[1:])))
    assert not remaining
    target_counts = [len(p) for p in patterns]
    same_count_distinct = any(len(a) == len(b) and a != b for i, a in enumerate(patterns) for b in patterns[i + 1:])
    region_loads = [sum(int(info["region_index"][min(piece)]) == i for piece in outputs)
                    for i in range(len(info["regions"]))]
    extents = [(max(r for r, _ in reg) - min(r for r, _ in reg) + 1,
                max(c for _, c in reg) - min(c for _, c in reg) + 1) for reg in info["regions"]]
    return {"layout": info["layout"], "shape": tuple(inp.shape),
            "piece_count": len(outputs), "piece_shapes": piece_shapes,
            "densities": densities, "holes": holes,
            "patterns": patterns, "pattern_poses": pattern_poses,
            "target_counts": target_counts, "offsets": offsets,
            "same_count_distinct": same_count_distinct,
            "repeated_congruent": len(set(patterns)) < len(patterns),
            "label_side_counts": sides_counts, "patch_sizes": patch_sizes,
            "separated_patches": separated_patches, "moves": moves,
            "distances": distances, "cross_region": cross_region,
            "overlap_cells": len(old & new),
            "overlap_recolored": sum(int(inp[p]) != int(out[p]) for p in old & new),
            "background": info["background"], "body": info["body"],
            "registration": info["registration"], "bands": tuple(sorted(info["band_colors"])),
            "region_loads": region_loads, "region_extents": extents,
            "region_side_counts": [sum(bool(v) for v in sides.values()) for sides in info["sides"]],
            "compass_needed": len(_solve_maps(info, use_labels=False)) > 1}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, training_info = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            key = _grid_key(inp)
            pair_key = key, _grid_key(out)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            info = _inspect_pair(inp, out)
            assert 1 <= info["piece_count"] <= 5
            if split == "train":
                training_info.append(info)
    # These are observed demonstrations of the universal rule, not a fixed
    # assignment of colors, target sizes, layouts, poses or counts to a split.
    assert len({i["background"] for i in training_info}) >= 2
    assert any(i["compass_needed"] for i in training_info)
    assert any("rotation" in i["moves"] for i in training_info)
    assert any("reflection" in i["moves"] for i in training_info)


def self_test() -> None:
    def key(puzzle):
        return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[s])
                     for s in ("train", "test"))
    # A mirror-symmetric complete piece has two equivalent D4 identifiers.
    # Its one resulting pixel map is a valid unique answer.
    base = np.zeros((14, 15), dtype=np.int8)
    base[:2, :] = 3
    base[-2:, :] = 4
    base[2:-2, :2] = 5
    base[2:-2, -2:] = 6
    symmetric = {(0, 1): 3, (1, 0): 1, (1, 1): 1, (1, 2): 1, (2, 1): 2}
    assert len({_signature(_transform_colored(symmetric, t)) for t in range(8)}) == 4
    inp, expected = base.copy(), base.copy()
    for p, v in _translated(symmetric, (2, 5)).items():
        expected[p] = v
        if v == 2:
            inp[p] = v
    for p, v in _translated(_transform_colored(symmetric, 1), (8, 9)).items():
        inp[p] = v
    assert np.array_equal(_predict(inp)[0], expected)
    assert key(generate_puzzle(777, 6, 3)) == key(generate_puzzle(777, 6, 3))
    assert key(generate_puzzle(777, 6, 3)) != key(generate_puzzle(778, 6, 3))
    for j, (a, b) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(9001 + j, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
        validate_generated_puzzle(puzzle)
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(9200, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    records, train_counts, test_counts, input_keys, within = [], set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair["input"], pair["output"])
                info["split"] = split
                records.append(info)
                local.append(info)
                input_keys.add(_grid_key(pair["input"]))
        within.append(local)
        demonstrations = [{k: v.copy() for k, v in p.items()} for p in puzzle["train"]]
        # The prediction function has no test output, seed or hidden metadata.
        for pair in puzzle["test"]:
            prediction = predict_from_training(demonstrations, pair["input"].copy())
            assert np.array_equal(prediction, pair["output"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(input_keys) == len(records)
    assert len({r["shape"] for r in records}) >= 50
    assert {r["piece_count"] for r in records} >= {1, 2, 3, 4, 5}
    assert {r["background"] for r in records} == set(range(10))
    assert all(len({r["background"] for r in group}) >= 2 for group in within)
    assert sum(len({r["bands"] for r in group}) > 1 for group in within) >= 35
    assert sum(len({(r["body"], r["registration"]) for r in group}) > 1 for group in within) >= 8
    assert sum(len({r["piece_count"] for r in group}) > 1 for group in within) >= 25
    assert sum(len({r["layout"] for r in group}) > 1 for group in within) >= 12
    assert all(len({p for r in group for p in r["pattern_poses"]}) > 1 for group in within)
    assert sum(any(len({pose for r in group for p, pose in zip(r["patterns"], r["pattern_poses"])
                        if p == pattern}) > 1
                   for pattern in {p for r in group for p in r["patterns"]})
               for group in within) >= 20
    for split in ("train", "test"):
        assert {r["layout"] for r in records if r["split"] == split} == {"vertical", "horizontal", "frame", "partial"}
    shapes = {s for r in records for s in r["piece_shapes"]}
    patterns = {s for r in records for s in r["patterns"]}
    assert len(shapes) >= 35 and len(patterns) >= 25
    assert max(max(s) for s in shapes) >= 9
    assert sum(min(r["densities"]) < .5 for r in records) >= 12
    assert sum(any(r["holes"]) for r in records) >= 10
    assert len({s for r in records for p, s in zip(r["patterns"], r["piece_shapes"])
                if len(p) == 1}) >= 10
    assert any(max(s) >= 2 * min(s) for s in shapes)
    simple_patterns = [{(0, 0)}, {(0, 0), (0, 1)}, {(0, 0), (1, 1)}, {(0, 0), (1, 0), (0, 1), (1, 1)}]
    assert all(min(_normalized_points(_transform_points(p, t)) for t in range(8)) in patterns for p in simple_patterns)
    assert sum(r["same_count_distinct"] for r in records) >= 15
    assert sum(r["repeated_congruent"] for r in records) >= 15
    assert {n for r in records for n in r["label_side_counts"]} == {1, 2, 3, 4}
    assert max(n for r in records for n in r["patch_sizes"]) >= 3
    assert sum(r["separated_patches"] for r in records) >= 20
    assert sum(r["overlap_cells"] > 0 for r in records) >= 30
    assert sum(r["overlap_cells"] == 0 for r in records) >= 30
    assert sum(r["overlap_recolored"] > 0 for r in records) >= 15
    assert {m for r in records for m in r["moves"]} == {"translation", "rotation", "reflection"}
    assert {bool(v) for r in records for v in r["cross_region"]} == {True, False}
    assert any(0 in r["region_loads"] for r in records)
    assert any(max(r["region_loads"]) >= 3 for r in records)
    assert any(len(set(r["region_extents"])) > 1 for r in records)
    assert any(3 in r["region_side_counts"] and r["layout"] == "vertical" for r in records)
    assert any(3 in r["region_side_counts"] and r["layout"] == "horizontal" for r in records)
    assert min(min(r["shape"]) for r in records) <= 16
    assert any(r["shape"] in {(13, 26), (26, 26), (30, 28), (30, 16)} for r in records)
    assert sum(len(set(r["piece_shapes"])) > 1 for r in records) >= 80
    assert sum(r["compass_needed"] for r in records) >= 60
    assert sum(r["overlap_cells"] > 0 and r["same_count_distinct"]
               and len(set(r["piece_shapes"])) > 1 for r in records) >= 5
