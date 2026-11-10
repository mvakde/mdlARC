import numpy as np


# Each template is an embedded rectilinear graph. Coordinates identify
# structural vertices; edges are rendered as horizontal or vertical segments.
_TEMPLATE_EDGES = {
    "h": (
        ((0, 0), (1, 0)),
        ((1, 0), (2, 0)),
        ((0, 1), (1, 1)),
        ((1, 1), (2, 1)),
        ((1, 0), (1, 1)),
    ),
    "c": (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
    ),
    "loop": (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
    ),
    "double_loop": (
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((0, 2), (1, 2)),
    ),
    "line": (
        ((0, 0), (0, 1)),
    ),
    "elbow": (
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
    ),
    "zigzag": (
        ((0, 0), (0, 1)),
        ((0, 1), (1, 1)),
        ((1, 1), (1, 2)),
    ),
    "t": (
        ((1, 1), (0, 1)),
        ((1, 1), (1, 0)),
        ((1, 1), (1, 2)),
    ),
    "plus": (
        ((1, 1), (0, 1)),
        ((1, 1), (2, 1)),
        ((1, 1), (1, 0)),
        ((1, 1), (1, 2)),
    ),
    "loop_tail": (
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
        ((1, 0), (2, 0)),
        ((1, 1), (2, 1)),
        ((2, 0), (2, 1)),
    ),
    "branch_c": (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((1, 0), (2, 0)),
        ((2, 0), (2, 1)),
        ((1, 0), (1, -1)),
    ),
    "triple_loop": (
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (0, 3)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 2), (1, 3)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((0, 2), (1, 2)),
        ((0, 3), (1, 3)),
    ),
    "bowtie": (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 1), (2, 1)),
        ((1, 2), (2, 2)),
        ((2, 1), (2, 2)),
    ),
    "barbell": (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((0, 2), (0, 3)),
        ((0, 2), (1, 2)),
        ((0, 3), (1, 3)),
        ((1, 2), (1, 3)),
        ((0, 1), (0, 2)),
    ),
    "chiral_path": (
        ((0, 0), (0, 1)),
        ((0, 1), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 2), (0, 2)),
        ((0, 2), (0, 3)),
        ((0, 3), (1, 3)),
    ),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _neighbors(p):
    r, c = p
    return ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))


def _components(mask):
    seen = set()
    result = []
    for r, c in zip(*np.nonzero(mask)):
        start = (int(r), int(c))
        if start in seen:
            continue
        component = {start}
        seen.add(start)
        stack = [start]
        while stack:
            for q in _neighbors(stack.pop()):
                if (0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1]
                        and mask[q] and q not in seen):
                    seen.add(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _bbox(pixels):
    rr = [p[0] for p in pixels]
    cc = [p[1] for p in pixels]
    return max(rr) - min(rr) + 1, max(cc) - min(cc) + 1


def _normalize(pixels):
    mr = min(r for r, _ in pixels)
    mc = min(c for _, c in pixels)
    return {(r - mr, c - mc) for r, c in pixels}


def _transform_coord(p, code):
    r, c = p
    if code >= 4:
        c = -c
    for _ in range(code % 4):
        r, c = c, -r
    return r, c


def _transform_pixels(pixels, code):
    return _normalize({_transform_coord(p, code) for p in pixels})


def _is_one_cell_wide(pixels):
    return not any({(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= pixels
                   for r, c in pixels)


def _graph(pixels):
    """Structural vertices and directed straight-run edges."""
    _require(len(pixels) > 1 and _is_one_cell_wide(pixels), "invalid thin object")
    adj = {p: [q for q in _neighbors(p) if q in pixels] for p in pixels}
    vertices = set()
    for p, ns in adj.items():
        if len(ns) != 2 or (ns[0][0] != ns[1][0] and ns[0][1] != ns[1][1]):
            vertices.add(p)
    _require(vertices, "object has no structural vertices")
    edges = set()
    for start in vertices:
        for q in adj[start]:
            previous, current = start, q
            while current not in vertices:
                next_cells = [v for v in adj[current] if v != previous]
                _require(len(next_cells) == 1, "nonstraight edge interior")
                previous, current = current, next_cells[0]
            _require(current != start, "unexpected self-loop")
            edges.add(tuple(sorted((start, current))))
    return vertices, edges


def _object_signature(pixels, reflect=True):
    """Canonical embedded graph, independent of each straight-run length.

    Vertex incidence and compass directions survive; coordinates and the
    relative lengths or rank ordering of unrelated arms do not.
    """
    vertices, edges = _graph(pixels)
    adjacency = {v: [] for v in vertices}
    for a, b in edges:
        dr, dc = b[0] - a[0], b[1] - a[1]
        _require((dr == 0) != (dc == 0), "edge is not straight")
        direction = (0 if dr < 0 else 2) if dc == 0 else (1 if dc > 0 else 3)
        adjacency[a].append((b, direction))
        adjacency[b].append((a, (direction + 2) % 4))
    representations = []
    codes = range(8) if reflect else range(4)
    for code in codes:
        mapped = {}
        for v in vertices:
            entries = []
            for q, direction in adjacency[v]:
                unit = ((-1, 0), (0, 1), (1, 0), (0, -1))[direction]
                tr, tc = _transform_coord(unit, code)
                new_direction = {(-1, 0): 0, (0, 1): 1,
                                 (1, 0): 2, (0, -1): 3}[(tr, tc)]
                entries.append((new_direction, q))
            mapped[v] = sorted(entries)
        for root in vertices:
            ids = {root: 0}
            queue = [root]
            for v in queue:
                for _, q in mapped[v]:
                    if q not in ids:
                        ids[q] = len(ids)
                        queue.append(q)
            _require(len(ids) == len(vertices), "disconnected object")
            representations.append(tuple(
                tuple((d, ids[q]) for d, q in mapped[v]) for v in queue
            ))
    return min(representations)


def _render_named_object(name, code, row_gaps, col_gaps):
    raw = [(_transform_coord(a, code), _transform_coord(b, code))
           for a, b in _TEMPLATE_EDGES[name]]
    rows = sorted({p[0] for edge in raw for p in edge})
    cols = sorted({p[1] for edge in raw for p in edge})
    _require(len(row_gaps) == len(rows) - 1 and len(col_gaps) == len(cols) - 1,
             "bad gap list")
    rpos = {rows[0]: 0}
    cpos = {cols[0]: 0}
    for i in range(1, len(rows)):
        rpos[rows[i]] = rpos[rows[i - 1]] + int(row_gaps[i - 1])
    for i in range(1, len(cols)):
        cpos[cols[i]] = cpos[cols[i - 1]] + int(col_gaps[i - 1])
    pixels = set()
    for a, b in raw:
        ar, ac = rpos[a[0]], cpos[a[1]]
        br, bc = rpos[b[0]], cpos[b[1]]
        _require(ar == br or ac == bc, "diagonal template edge")
        if ar == br:
            pixels.update((ar, c) for c in range(min(ac, bc), max(ac, bc) + 1))
        else:
            pixels.update((r, ac) for r in range(min(ar, br), max(ar, br) + 1))
    return _normalize(pixels)


def _make_object(name, rng, max_gap=4, code=None, gap_mode="random"):
    if code is None:
        code = int(rng.integers(0, 8))
    if name == "c" and gap_mode == "stagger":
        top = int(rng.integers(3, max_gap + 3))
        bottom = int(rng.integers(3, max_gap + 3))
        if top == bottom:
            bottom += 1
        height = int(rng.integers(3, max_gap + 3))
        pixels = {(0, c) for c in range(top)} | {(height - 1, c) for c in range(bottom)}
        pixels |= {(r, 0) for r in range(height)}
        return _transform_pixels(pixels, code)
    raw = [(_transform_coord(a, code), _transform_coord(b, code))
           for a, b in _TEMPLATE_EDGES[name]]
    nr = len({p[0] for edge in raw for p in edge}) - 1
    nc = len({p[1] for edge in raw for p in edge}) - 1
    if gap_mode == "compact":
        rg, cg = [2] * nr, [2] * nc
    else:
        rg = rng.integers(2, max_gap + 1, size=nr).tolist()
        cg = rng.integers(2, max_gap + 1, size=nc).tolist()
    return _render_named_object(name, code, rg, cg)


def _catalog_signature(name):
    return _object_signature(_render_named_object(
        name, 0,
        [2] * (len({p[0] for edge in _TEMPLATE_EDGES[name] for p in edge}) - 1),
        [2] * (len({p[1] for edge in _TEMPLATE_EDGES[name] for p in edge}) - 1),
    ))


_CATALOG_SIGNATURES = {name: _catalog_signature(name) for name in _TEMPLATE_EDGES}
_require(len(set(_CATALOG_SIGNATURES.values())) == len(_CATALOG_SIGNATURES),
         "catalog contains equivalent structures")


def _near_miss(pixels, rng):
    """Delete one meaningful bar or add a short bar at a free junction."""
    base_signature = _object_signature(pixels)
    vertices, edges = _graph(pixels)
    choices = list(edges)
    for index in rng.permutation(len(choices)):
        a, b = choices[int(index)]
        if a[0] == b[0]:
            interior = {(a[0], c) for c in range(min(a[1], b[1]) + 1,
                                                  max(a[1], b[1]))}
        else:
            interior = {(r, a[1]) for r in range(min(a[0], b[0]) + 1,
                                                  max(a[0], b[0]))}
        trial = pixels - interior
        if (interior and len(_components_from_pixels(trial)) == 1
                and _is_one_cell_wide(trial)):
            try:
                if _object_signature(trial) != base_signature:
                    return _normalize(trial)
            except AssertionError:
                pass
    for index in rng.permutation(len(vertices)):
        v = list(sorted(vertices))[int(index)]
        for direction in rng.permutation(4):
            dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[int(direction)]
            trial = set(pixels)
            trial.add((v[0] + dr, v[1] + dc))
            trial.add((v[0] + 2 * dr, v[1] + 2 * dc))
            if len(trial) == len(pixels) + 2 and _is_one_cell_wide(trial):
                try:
                    if _object_signature(trial) != base_signature:
                        return _normalize(trial)
                except AssertionError:
                    pass
    return None


def _components_from_pixels(pixels):
    remaining = set(pixels)
    result = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            for q in _neighbors(stack.pop()):
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _pack_objects(objects, height, width, rng):
    placements = [None] * len(objects)
    blocked = np.zeros((height, width), dtype=bool)
    areas = [(_bbox(p)[0] * _bbox(p)[1], float(rng.random()), i)
             for i, p in enumerate(objects)]
    for _, _, index in sorted(areas, reverse=True):
        pixels = objects[index]
        bh, bw = _bbox(pixels)
        if bh > height or bw > width:
            return None
        positions = [(r, c) for r in range(height - bh + 1)
                     for c in range(width - bw + 1)]
        chosen = None
        for k in rng.permutation(len(positions)):
            top, left = positions[int(k)]
            if all(not blocked[top + r, left + c] for r, c in pixels):
                chosen = {(top + r, left + c) for r, c in pixels}
                break
        if chosen is None:
            return None
        placements[index] = chosen
        for r, c in chosen:
            for nr, nc in ((r, c), *_neighbors((r, c))):
                if 0 <= nr < height and 0 <= nc < width:
                    blocked[nr, nc] = True
    return placements


def _solve_grid(input_grid, background, candidate, selector):
    output = input_grid.copy()
    references = _components(input_grid == selector)
    _require(len(references) == 1, "one selector object required")
    signature = _object_signature(references[0])
    for component in _components(input_grid == candidate):
        if _object_signature(component) == signature:
            for r, c in component:
                output[r, c] = selector
    return output
_REFERENCE_POOL = ("h", "c", "loop", "double_loop", "chiral_path",
                   "loop_tail", "branch_c", "bowtie")
_DECOY_POOL = tuple(_TEMPLATE_EDGES)
_SOURCE_INPUT_HASHES = {
    ((18, 18), "ba20a9c43c1c7a414865cbedc69fe38b03a68006bfaac1bbdf72532b153a0b42"),
    ((15, 15), "4c5b9d90104aa48c6b26da1c1e5995195de947773baeea2a2bf5f358aefa105d"),
    ((18, 18), "39ebc14f646caba0755e32170cf351c450234f243ed086815b8f33775ead5367"),
    ((19, 19), "1ef73584545e0353a7c011af9e365de2225316b2e61fe063ee8b3561f15f1dee"),
}


def _stagger_c(top, bottom, height, code):
    pixels = {(0, c) for c in range(top)}
    pixels |= {(height - 1, c) for c in range(bottom)}
    pixels |= {(r, 0) for r in range(height)}
    return _transform_pixels(pixels, code)


def sample_puzzle_parameters(rng, train_count):
    colors = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    palette_mode = "constant" if rng.random() < 0.38 else "varying"
    permutations = [(colors[a], colors[b], colors[c])
                    for a, b, c in ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                                    (1, 2, 0), (2, 0, 1), (2, 1, 0))]
    if palette_mode == "constant":
        train_palettes = [permutations[0]] * train_count
    else:
        train_palettes = [permutations[0], permutations[int(rng.integers(1, 6))]]
        train_palettes += [permutations[int(rng.integers(0, 6))]
                           for _ in range(train_count - 2)]
    mode = ("unseen_double", "familiar", "unseen_other", "chiral")[
        int(rng.integers(0, 4))]
    pool = [name for name in _REFERENCE_POOL if
            name != "double_loop" or mode != "unseen_double"]
    if mode == "chiral":
        pool.remove("chiral_path")
        train_names = ["chiral_path"]
        train_names += [pool[int(i)] for i in rng.permutation(len(pool))[:min(2, train_count - 1)]]
        test_name = "chiral_path"
    else:
        train_names = [pool[int(i)] for i in rng.permutation(len(pool))[:min(3, train_count)]]
        if mode == "unseen_double":
            test_name = "double_loop"
        elif mode == "familiar":
            test_name = train_names[int(rng.integers(0, len(train_names)))]
        else:
            unseen = [name for name in ("barbell", "triple_loop", "branch_c",
                                             "bowtie", "loop_tail", "double_loop")
                      if name not in train_names]
            test_name = unseen[int(rng.integers(0, len(unseen)))]
    train_names = [train_names[int(i)] for i in rng.permutation(len(train_names))]
    return {"colors": colors, "palette_mode": palette_mode,
            "train_palettes": train_palettes, "permutations": permutations,
            "train_names": train_names, "test_name": test_name,
            "mode": mode}


def _generate_pair(rng, regime, palette, reference_name, force_mirror=False):
    background, candidate, selector = palette
    for attempt in range(220):
        small = rng.random() < (0.36 if regime == "source_train" else 0.23)
        if small:
            height = int(rng.integers(15, 22))
            width = int(rng.integers(16, 23))
        else:
            height = int(rng.integers(22, 29 if regime == "source_train" else 31))
            width = int(rng.integers(21, 29 if regime == "source_train" else 31))
        if rng.random() < 0.5:
            height, width = width, height
        max_gap = 3 if small else (4 if rng.random() < 0.4 else 3)
        match_count = int(rng.integers(2, 5))
        decoy_count = int(rng.integers(2, 6))
        if attempt >= 180:
            height = width = 30
            small = False
            max_gap = 2
            match_count = decoy_count = 2
        if small and match_count + decoy_count > 5:
            match_count, decoy_count = 2, 2 + int(rng.integers(0, 2))
        ref_code = int(rng.integers(0, 8))
        stagger = reference_name == "c" and rng.random() < 0.55
        if stagger:
            top = int(rng.integers(3, 6))
            bottom = int(rng.integers(3, 6))
            if top == bottom:
                bottom = 3 if top == 5 else top + 1
            spine = int(rng.integers(3, 6))
            reference = _stagger_c(top, bottom, spine, ref_code)
        else:
            reference = _make_object(reference_name, rng, max_gap, ref_code,
                                     "random" if rng.random() < 0.56 else "compact")
        reference_signature = _object_signature(reference)
        objects = [reference]
        exact_match = rng.random() < 0.50
        if force_mirror:
            objects.append(_transform_pixels(reference, 4))
            exact_match = True
        elif exact_match:
            objects.append(_transform_pixels(reference, int(rng.integers(0, 8))))
        else:
            objects.append(_make_object(reference_name, rng, max_gap,
                                         (ref_code + 1) % 4,
                                         "stagger" if stagger else "random"))
        if stagger:
            # A shorter top arm and longer bottom arm (or vice versa) are
            # compared by graph structure, not by row/column ranks.
            mixed = _stagger_c(min(top, bottom) - 1,
                               max(top, bottom) + 1,
                               max(3, spine + (1 if spine < 5 else -1)),
                               int(rng.integers(0, 8)))
            objects.append(mixed)
        else:
            objects.append(_make_object(reference_name, rng, max_gap,
                                         int(rng.integers(0, 8)), "random"))
        for _ in range(match_count - 2):
            objects.append(_make_object(reference_name, rng, max_gap,
                                         int(rng.integers(0, 8)),
                                         "stagger" if reference_name == "c" and rng.random() < 0.4
                                         else "random"))
        if any(_object_signature(obj) != reference_signature for obj in objects[1:]):
            continue
        if not any(_bbox(obj) != _bbox(reference) or len(obj) != len(reference)
                   for obj in objects[1:]):
            continue
        near = _near_miss(reference, rng)
        if near is None:
            continue
        objects.append(near)
        other_names = [name for name in _DECOY_POOL
                       if _CATALOG_SIGNATURES[name] != reference_signature]
        for index in rng.permutation(len(other_names)):
            name = other_names[int(index)]
            obj = _make_object(name, rng, max_gap,
                               int(rng.integers(0, 8)),
                               "compact" if small else "random")
            if _object_signature(obj) != reference_signature:
                objects.append(obj)
            if len(objects) == 1 + match_count + decoy_count:
                break
        if len(objects) != 1 + match_count + decoy_count:
            continue
        placements = _pack_objects(objects, height, width, rng)
        if placements is None:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for r, c in placements[0]:
            grid[r, c] = selector
        for component in placements[1:]:
            for r, c in component:
                grid[r, c] = candidate
        output = _solve_grid(grid, background, candidate, selector)
        return grid, output
    raise RuntimeError(f"could not pack {regime} {reference_name}")


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        minimum = max(2, num_test + 1)
        num_train = minimum + seed % (11 - minimum)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid_key(grid):
    return tuple(grid.shape), grid.astype(np.uint8, copy=False).tobytes()


def _has_bar_edit_decoy(reference, nonmatches):
    reference = _normalize(reference)
    for component in nonmatches:
        decoy = _normalize(component)
        if abs(len(reference) - len(decoy)) > 4:
            continue
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                shifted = {(r + dr, c + dc) for r, c in reference}
                difference = shifted ^ decoy
                if 1 <= len(difference) <= 4 and (shifted <= decoy or decoy <= shifted):
                    return True
    return False


def _analyze_pair(input_grid, output_grid):
    colors, counts = np.unique(input_grid, return_counts=True)
    _require(len(colors) == 3, "pair needs exactly three colors")
    background = int(colors[int(np.argmax(counts))])
    changed = input_grid != output_grid
    _require(bool(np.any(changed)), "pair must recolor at least one match")
    from_colors = set(int(x) for x in input_grid[changed])
    to_colors = set(int(x) for x in output_grid[changed])
    _require(len(from_colors) == len(to_colors) == 1, "inconsistent recoloring")
    candidate = next(iter(from_colors))
    selector = next(iter(to_colors))
    _require(len({background, candidate, selector}) == 3, "ambiguous roles")
    _require(set(int(x) for x in output_grid.flat) == set(int(x) for x in colors),
             "output changes palette")
    refs = _components(input_grid == selector)
    candidates = _components(input_grid == candidate)
    _require(len(refs) == 1 and len(candidates) >= 4,
             "pair needs one reference and several candidates")
    _require(len(_components(input_grid != background)) == 1 + len(candidates),
             "objects of different colors touch")
    reference = refs[0]
    _require(_is_one_cell_wide(reference), "reference is too thick")
    signature = _object_signature(reference)
    matches = []
    nonmatches = []
    for component in candidates:
        _require(_is_one_cell_wide(component), "candidate is too thick")
        (matches if _object_signature(component) == signature else nonmatches).append(component)
    _require(len(matches) >= 2 and len(nonmatches) >= 2,
             "pair needs multiple matches and near alternatives")
    _require(_has_bar_edit_decoy(reference, nonmatches),
             "pair lacks a close one-bar structural decoy")
    expected = _solve_grid(input_grid, background, candidate, selector)
    _require(np.array_equal(expected, output_grid), "incorrect structural recoloring")
    _require(any(_bbox(m) != _bbox(reference) or len(m) != len(reference)
                 for m in matches), "pair lacks an elastic comparison")
    return {"roles": (background, candidate, selector), "colors": frozenset(colors),
            "reference": reference, "signature": signature,
            "matches": matches, "nonmatches": nonmatches}
def validate_generated_puzzle(puzzle: dict) -> None:
    import hashlib
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair has wrong schema")
            for key in ("input", "output"):
                grid = pair[key]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2D NumPy array")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integer")
                h, w = grid.shape
                _require(1 <= h <= 30 and 1 <= w <= 30, "grid outside ARC dimensions")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid ARC color")
            a, b = pair["input"], pair["output"]
            _require(a.shape == b.shape, "output changes dimensions")
            input_key = _grid_key(a)
            pair_key = (input_key, _grid_key(b))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate generated pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            digest = hashlib.sha256(a.astype(np.uint8, copy=False).tobytes()).hexdigest()
            _require((tuple(a.shape), digest) not in _SOURCE_INPUT_HASHES,
                     "source task input copied")
            analyses[split].append(_analyze_pair(a, b))
    all_analyses = analyses["train"] + analyses["test"]
    color_sets = {a["colors"] for a in all_analyses}
    _require(len(color_sets) == 1, "puzzle palette convention changed")
    train_roles = {a["roles"] for a in analyses["train"]}
    test_roles = {a["roles"] for a in analyses["test"]}
    _require(len(train_roles) > 1 or len(train_roles | test_roles) == 1,
             "test role reassignment absent from training")
    training_signatures = [a["signature"] for a in analyses["train"]]
    _require(len(set(training_signatures)) >= min(2, nt),
             "training reference topology does not vary")
    _require(any(a["signature"] in training_signatures for a in analyses["test"])
             or len(set(training_signatures)) >= 2,
             "unseen test structure needs two training classes")
    chiral = _CATALOG_SIGNATURES["chiral_path"]
    if any(a["signature"] == chiral for a in analyses["test"]):
        _require(chiral in training_signatures,
                 "chiral test class needs a training reflection demonstration")
        _require(any(
            a["signature"] == chiral and
            any(_object_signature(m, reflect=False) !=
                _object_signature(a["reference"], reflect=False)
                for m in a["matches"])
            for a in analyses["train"]),
            "training lacks a genuine mirror comparison")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    parameters = sample_puzzle_parameters(rng, train_count)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train":
                name = parameters["train_names"][index % len(parameters["train_names"])]
                palette = parameters["train_palettes"][index]
                mirror = name == "chiral_path"
            else:
                name = parameters["test_name"]
                palette = (parameters["permutations"][0] if parameters["palette_mode"] == "constant"
                           else parameters["permutations"][int(rng.integers(0, 6))])
                mirror = False
            for _ in range(8):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(child_seed)
                a, b = _generate_pair(pair_rng,
                                      "source_train" if split == "train" else "target_test",
                                      palette, name, mirror)
                key = _grid_key(a)
                if key not in seen_inputs:
                    seen_inputs.add(key)
                    puzzle[split].append({"input": a, "output": b})
                    break
            else:
                raise RuntimeError("could not produce unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_fingerprint(puzzle):
    return tuple((split, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def _c_arm_lengths(pixels):
    vertices, edges = _graph(pixels)
    degree = {v: 0 for v in vertices}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    arms = [abs(a[0] - b[0]) + abs(a[1] - b[1])
            for a, b in edges if degree[a] == 1 or degree[b] == 1]
    _require(len(arms) == 2, "C needs two arms")
    return tuple(sorted(arms))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic counts lack diversity")
    _require(len({_puzzle_fingerprint(p) for p in puzzles}) >= 40,
             "whole puzzles repeat")
    layouts = set()
    dimensions = set()
    color_sets = set()
    reference_signatures = set()
    count_profiles = set()
    role_changed = 0
    small = 0
    large = 0
    exact = 0
    stretched_references = 0
    chiral_mirrors = 0
    mixed_c_arms = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        analyses = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                analysis = _analyze_pair(grid, pair["output"])
                analyses.append(analysis)
                dimensions.add(tuple(grid.shape))
                color_sets.add(analysis["colors"])
                reference_signatures.add(analysis["signature"])
                count_profiles.add((len(analysis["matches"]), len(analysis["nonmatches"])))
                background = analysis["roles"][0]
                layouts.add((grid.shape, (grid != background).tobytes()))
                small += min(grid.shape) <= 19
                large += max(grid.shape) >= 27
                exact += any(_bbox(m) == _bbox(analysis["reference"])
                             and len(m) == len(analysis["reference"])
                             for m in analysis["matches"])
                stretched_references += max(_bbox(analysis["reference"])) >= 5
                if analysis["signature"] == _CATALOG_SIGNATURES["chiral_path"]:
                    chiral_mirrors += any(
                        _object_signature(m, reflect=False) !=
                        _object_signature(analysis["reference"], reflect=False)
                        for m in analysis["matches"])
                if analysis["signature"] == _CATALOG_SIGNATURES["c"]:
                    short, long = _c_arm_lengths(analysis["reference"])
                    mixed_c_arms += any(
                        _c_arm_lengths(m)[0] < short and
                        _c_arm_lengths(m)[1] > long
                        for m in analysis["matches"])
        role_changed += len({a["roles"] for a in analyses}) > 1
    _require(len(layouts) >= 150 and len(dimensions) >= 35,
             "layouts or dimensions lack diversity")
    _require(len(color_sets) >= 8 and len(reference_signatures) >= 6,
             "palettes or reference classes lack diversity")
    _require(len(count_profiles) >= 5 and role_changed >= 8,
             "object counts or pair-level roles lack diversity")
    _require(small >= 8 and large >= 20 and exact >= 15 and
             stretched_references >= 25 and chiral_mirrors >= 2 and
             mixed_c_arms >= 2,
             "size and reflection cases lack coverage")
    first = generate_puzzle(10001, 6, 3)
    second = generate_puzzle(10001, 6, 3)
    _require(_puzzle_fingerprint(first) == _puzzle_fingerprint(second),
             "generator is not deterministic")
    _require(_puzzle_fingerprint(first) != _puzzle_fingerprint(generate_puzzle(10002, 6, 3)),
             "neighboring seeds are identical")
    for seed, nt, ns in ((20001, 2, 1), (20002, 10, 1), (20003, 10, 4),
                         (30001, None, 1), (30002, None, 4),
                         (30003, 2, None), (30004, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        _require(nt is None or len(puzzle["train"]) == nt, "train count ignored")
        _require(ns is None or len(puzzle["test"]) == ns, "test count ignored")
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(40000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
