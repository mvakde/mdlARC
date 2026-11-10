"""Registered perforations: rigidly placed holes toggle a binary field by parity.

Only NumPy and the Python standard library are used.  Generation paints from
sampled placements; solve recovers an exact cover of marker triples from input.
"""

from itertools import combinations

import numpy as np


_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((a >= 0) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=False)


def _pose(point, kind):
    r, c = point
    return ((r, c), (c, -r), (-r, -c), (-c, r),
            (-r, c), (r, -c), (c, r), (-c, -r))[kind]


def _noncollinear(points):
    a, b, c = points
    return ((b[0] - a[0]) * (c[1] - a[1])
            != (c[0] - a[0]) * (b[1] - a[1]))


def _bbox(points):
    points = list(points)
    return (min(r for r, _ in points), min(c for _, c in points),
            max(r for r, _ in points) + 1, max(c for _, c in points) + 1)


def _rect(box):
    r0, c0, r1, c1 = box
    return (slice(r0, r1), slice(c0, c1))


def _boundary(height, width):
    return {(r, c) for r in range(height) for c in range(width)
            if r in (0, height - 1) or c in (0, width - 1)}


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        stack, part = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in _STEPS:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _parse_candidate(a, bg, result, board):
    """Read roles for one candidate monochrome field perimeter."""
    outside = a != bg
    outside[_rect(board)] = False
    points = [tuple(map(int, p)) for p in np.argwhere(outside)]
    _require(points, "missing stencil")
    source = _bbox(points)
    sr, sc, er, ec = source
    br, bc, be, ce = board
    _require(er < br or be < sr or ec < bc or ce < sc, "separate rectangles")
    stencil = a[_rect(source)]
    height, width = stencil.shape
    _require(height >= 3 and width >= 3, "stencil size")
    _require(len(points) == np.count_nonzero(stencil != bg), "outside clutter")
    groups = {int(color): [tuple(map(int, p)) for p in np.argwhere(stencil == color)]
              for color in np.unique(stencil) if int(color) != bg}
    repeated = [color for color, cells in groups.items() if len(cells) > 1]
    _require(len(groups) == 4 and len(repeated) == 1, "stencil roles")
    material = repeated[0]
    identities = tuple(sorted(color for color in groups if color != material))
    landmarks = {color: groups[color][0] for color in identities}
    edge = _boundary(height, width)
    _require(all(stencil[p] != bg for p in edge), "closed stencil perimeter")
    _require(set(landmarks.values()) <= edge, "perimeter landmarks")
    _require(_noncollinear(list(landmarks.values())), "collinear landmarks")
    _require(all(abs(p[0] - q[0]) + abs(p[1] - q[1]) >= 2
                 for p, q in combinations(landmarks.values(), 2)), "adjacent landmarks")
    solid = {tuple(map(int, p)) for p in np.argwhere(stencil != bg)}
    _require(len(_components(solid)) == 1, "disconnected stencil material")
    holes = tuple(tuple(map(int, p)) for p in np.argwhere(stencil == bg))
    _require(holes, "no holes")
    apertures = _components(holes)
    for part in apertures:
        r0, c0, r1, c1 = _bbox(part)
        _require(len(part) == (r1 - r0) * (c1 - c0), "nonrectangular aperture")
    interior = a[br + 1:be - 1, bc + 1:ce - 1]
    ink_choices = set(map(int, np.unique(interior))) - {bg, *identities}
    _require(len(ink_choices) == 1, "binary field ink")
    ink = ink_choices.pop()
    _require(len({bg, material, result, ink, *identities}) == 7, "distinct roles")
    _require(set(map(int, np.unique(a))) == {bg, material, result, ink, *identities}, "clutter colors")
    by_color = {color: [tuple(map(int, p)) for p in np.argwhere(a == color)
                        if br < p[0] < be - 1 and bc < p[1] < ce - 1]
                for color in identities}
    counts = {len(cells) for cells in by_color.values()}
    _require(len(counts) == 1 and next(iter(counts)) >= 1, "target landmark counts")
    count = next(iter(counts))
    for color in identities:
        _require(np.count_nonzero(a == color) == count + 1, "landmark outside roles")
    _require(np.count_nonzero(a == material) == len(groups[material]), "material outside stencil")
    return dict(bg=bg, material=material, result=result, ink=ink,
                source=source, board=board, shape=stencil.shape, holes=holes,
                landmarks=landmarks, identities=identities, by_color=by_color,
                markers=frozenset(p for cells in by_color.values() for p in cells),
                count=count, apertures=tuple(sorted(len(p) for p in apertures)))


def _parse(a):
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg)
             and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "background canvas edge")
    candidates = []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        coords = [tuple(map(int, p)) for p in np.argwhere(a == color)]
        box = _bbox(coords)
        r, c, e, f = box
        h, w = e - r, f - c
        if h < 5 or w < 5:
            continue
        boundary = {(r + i, c + j) for i, j in _boundary(h, w)}
        if set(coords) != boundary:
            continue
        try:
            candidates.append(_parse_candidate(a, bg, color, box))
        except AssertionError:
            continue
    _require(len(candidates) == 1, "field/stencil decomposition must be unique")
    return candidates[0]


def _register(model):
    """Find an unordered, unique exact cover of all bare landmark cells."""
    colors = model["identities"]
    first = colors[0]
    source = model["landmarks"]
    h, w = model["shape"]
    corners = ((0, 0), (h - 1, 0), (0, w - 1), (h - 1, w - 1))
    br, bc, be, ce = model["board"]
    targets = {color: set(cells) for color, cells in model["by_color"].items()}
    options = []
    for anchor in sorted(targets[first]):
        for kind in range(8):
            origin = _pose(source[first], kind)
            delta = (anchor[0] - origin[0], anchor[1] - origin[1])
            matched = []
            for color in colors:
                q = _pose(source[color], kind)
                p = (q[0] + delta[0], q[1] + delta[1])
                if p not in targets[color]:
                    break
                matched.append(p)
            else:
                moved = [(_pose(p, kind)[0] + delta[0],
                          _pose(p, kind)[1] + delta[1]) for p in corners]
                if all(br < r < be - 1 and bc < c < ce - 1 for r, c in moved):
                    options.append((kind, delta, frozenset(matched)))
    index = {p: [] for p in model["markers"]}
    for i, option in enumerate(options):
        for p in option[2]:
            index[p].append(i)
    solutions = []

    def visit(unused, selected):
        if len(solutions) > 1:
            return
        if not unused:
            solutions.append(tuple(sorted(selected)))
            return
        cell = min(unused, key=lambda p: (sum(options[i][2] <= unused for i in index[p]), p))
        for i in index[cell]:
            if options[i][2] <= unused:
                visit(unused - options[i][2], selected + [i])

    visit(model["markers"], [])
    _require(len(solutions) == 1, "landmarks must have exactly one registration cover")
    return tuple(options[i] for i in solutions[0])


def _coverage(a, model, placements, unregistered=False):
    exposure = np.zeros(a.shape, dtype=np.int16)
    footprints = np.zeros(a.shape, dtype=np.int16)
    sh, sw = model["shape"]
    br, bc, be, ce = model["board"]
    anchor = (model["landmarks"][model["identities"][0]]
              if unregistered else None)
    for kind, delta, _ in placements:
        if unregistered:
            moved = _pose(anchor, kind)
            delta = (moved[0] + delta[0] - anchor[0], moved[1] + delta[1] - anchor[1])
            kind = 0
        for r, c in model["holes"]:
            p = _pose((r, c), kind)
            q = (p[0] + delta[0], p[1] + delta[1])
            if br < q[0] < be - 1 and bc < q[1] < ce - 1:
                exposure[q] += 1
        if not unregistered:
            for r in range(sh):
                for c in range(sw):
                    p = _pose((r, c), kind)
                    footprints[p[0] + delta[0], p[1] + delta[1]] += 1
    return exposure, footprints


def _paint(a, model, mask):
    mask = mask.copy()
    for p in model["markers"]:
        mask[p] = False
    out = a.copy()
    out[_rect(model["source"])] = model["bg"]
    out[mask & (a == model["bg"])] = model["result"]
    out[mask & (a == model["ink"])] = model["bg"]
    return out


def _analyze(a):
    model = _parse(a)
    placements = _register(model)
    exposure, footprints = _coverage(a, model, placements)
    model.update(placements=placements, exposure=exposure, footprints=footprints)
    return model


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from visible roles, geometry, and parity alone."""
    a = _grid(input_grid)
    model = _analyze(a)
    return _paint(a, model, model["exposure"] % 2 == 1)


def _field_mask(a, model):
    mask = np.zeros(a.shape, dtype=bool)
    r, c, e, f = model["board"]
    mask[r + 1:e - 1, c + 1:f - 1] = True
    for p in model["markers"]:
        mask[p] = False
    return mask


def _pair_info(inp, out):
    a, b = _grid(inp), _grid(out)
    _require(a.shape == b.shape, "same canvas required")
    m = _analyze(a)
    exposure = m["exposure"]
    field = _field_mask(a, m)
    odd = field & (exposure % 2 == 1)
    even_shared = field & (exposure > 0) & (exposure % 2 == 0)
    _require(np.array_equal(_paint(a, m, odd), b), "incorrect registered complement")
    _require(np.all(b[_rect(m["source"])] == m["bg"]), "source not erased")
    for p in m["markers"]:
        _require(b[p] == a[p], "target landmark changed")
    r, c, e, f = m["board"]
    for i, j in _boundary(e - r, f - c):
        _require(b[r + i, c + j] == m["result"], "frame changed")
    for old, new in ((m["bg"], m["result"]), (m["ink"], m["bg"])):
        _require(np.any(odd & (a == old)), "missing odd binary-state witness")
        _require(np.all(b[odd & (a == old)] == new), "odd-state truth table")
        _require(np.all(b[field & ~odd & (a == old)] == old), "even/nonaction state changed")
    if m["count"] > 1:
        for state in (m["bg"], m["ink"]):
            _require(np.any(even_shared & (a == state)), "missing cancellation state")
        _require(not np.array_equal(_paint(a, m, exposure > 0), b), "union ablation did not fail")
    material_only = field & (m["footprints"] > 0) & (exposure == 0)
    _require(np.any(material_only & (a == m["bg"])), "missing protected material witness")
    _require(np.any(field & (exposure == 0) & (a == m["ink"])), "missing nonaction ink")
    wrong, _ = _coverage(a, m, m["placements"], unregistered=True)
    m["a_essential"] = not np.array_equal(_paint(a, m, wrong % 2 == 1), b)
    # A fill-only ablation preserves the ink that complement must erase.
    fill_only = _paint(a, m, odd)
    fill_only[odd & (a == m["ink"])] = m["ink"]
    _require(not np.array_equal(fill_only, b), "B fill-only ablation did not fail")
    m["wall_hole_crossing"] = bool(np.any(field & (exposure == 1) & (m["footprints"] > 1)))
    m["triple_exposure"] = bool(np.any(field & (exposure == 3)))
    _require(len(m["holes"]) >= 6, "sample has too little aperture area")
    return m


def _positive_parts(rng, total, count):
    if count == 1:
        return [total]
    cuts = sorted(map(int, rng.choice(np.arange(1, total), size=count - 1, replace=False)))
    ends = [0] + cuts + [total]
    return [b - a for a, b in zip(ends, ends[1:])]


def _slots(rng, length, thickness):
    maximum = min(3, (length + thickness) // (thickness + 1))
    count = int(rng.integers(1, maximum + 1))
    sizes = _positive_parts(rng, length - (count - 1) * thickness, count)
    intervals, start = [], 1
    for size in sizes:
        intervals.append((start, start + size))
        start += size + thickness
    return intervals


def _stencil(rng):
    for _ in range(100):
        h, w = map(int, rng.integers(5, 11, size=2))
        thick = int(rng.choice((1, 2), p=(.75, .25)))
        rows, cols = _slots(rng, h - 2, thick), _slots(rng, w - 2, thick)
        holes = []
        for r0, r1 in rows:
            for c0, c1 in cols:
                if rng.random() < .82:
                    holes.extend((r, c) for r in range(r0, r1) for c in range(c0, c1))
        if len(holes) < 6:
            continue
        edge = sorted(_boundary(h, w))
        for _ in range(100):
            marks = [edge[int(i)] for i in rng.choice(len(edge), size=3, replace=False)]
            if (_noncollinear(marks)
                    and all(abs(p[0] - q[0]) + abs(p[1] - q[1]) >= 2
                            for p, q in combinations(marks, 2))):
                return (h, w), tuple(holes), tuple(marks)
    raise RuntimeError("could not sample a stencil")


def _palette(rng):
    bg = 0 if rng.random() < .5 else int(rng.integers(1, 10))
    colors = list(map(int, rng.choice([c for c in range(10) if c != bg], size=6, replace=False)))
    return (bg, *colors)  # background, material, ink, result, three identities


def _scene(rng, shape):
    """Pack two separated rectangles, with all four relative directions possible."""
    sh, sw = shape
    vertical = bool(rng.integers(2))
    source_axis = sh if vertical else sw
    lower = max(shape) + 2
    board_axis = int(rng.integers(lower, min(21, 27 - source_axis) + 1))
    board_cross = int(rng.integers(lower, 24))
    source_cross = sw if vertical else sh
    gap = int(rng.integers(1, min(3, 28 - source_axis - board_axis) + 1))
    required = source_axis + board_axis + gap + 2
    axis_size = required + int(rng.integers(0, min(2, 30 - required) + 1))
    cross_size = max(source_cross, board_cross) + 2 + int(rng.integers(4))
    axis_start = 1 + int(rng.integers(axis_size - required + 1))
    if rng.integers(2):
        source_start, board_start = axis_start, axis_start + source_axis + gap
    else:
        board_start, source_start = axis_start, axis_start + board_axis + gap
    source_side = int(rng.integers(1, cross_size - source_cross))
    board_side = int(rng.integers(1, cross_size - board_cross))
    if vertical:
        canvas = (axis_size, cross_size)
        sb = (source_start, source_side, source_start + sh, source_side + sw)
        bb = (board_start, board_side, board_start + board_axis, board_side + board_cross)
    else:
        canvas = (cross_size, axis_size)
        sb = (source_side, source_start, source_side + sh, source_start + sw)
        bb = (board_side, board_start, board_side + board_cross, board_start + board_axis)
    return canvas, sb, bb


def _texture(rng, height, width):
    mode = int(rng.integers(6))
    r, c = np.indices((height, width))
    if mode <= 2:
        p = (.16, .5, .82)[mode]
        return rng.random((height, width)) < p
    if mode == 3:
        step, offset = int(rng.integers(2, 5)), int(rng.integers(4))
        return ((r if rng.integers(2) else c) + offset) % step == 0
    if mode == 4:
        rh, cw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        return (r // rh + c // cw) % 2 == int(rng.integers(2))
    mask = np.zeros((height, width), dtype=bool)
    for _ in range(int(rng.integers(3, 10))):
        r0, r1 = sorted(map(int, rng.choice(height + 1, size=2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(width + 1, size=2, replace=False)))
        mask[r0:r1, c0:c1] ^= True
    return mask


def _sample_placements(rng, shape, landmarks, kinds, board):
    br, bc, be, ce = board
    height, width = be - br - 2, ce - bc - 2
    h, w = shape
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    placements, used, origins = [], set(), []
    for kind in kinds:
        moved_corners = [_pose(p, kind) for p in corners]
        r0, c0, r1, c1 = _bbox(moved_corners)
        ph, pw = r1 - r0, c1 - c0
        if ph > height or pw > width:
            return None
        for _ in range(150):
            if origins and rng.random() < .85:
                origin = origins[int(rng.integers(len(origins)))]
                top = origin[0] + int(rng.integers(-4, 5))
                left = origin[1] + int(rng.integers(-4, 5))
                if not (0 <= top <= height - ph and 0 <= left <= width - pw):
                    continue
            else:
                top = int(rng.integers(height - ph + 1))
                left = int(rng.integers(width - pw + 1))
            delta = (br + 1 + top - r0, bc + 1 + left - c0)
            marks = frozenset((_pose(p, kind)[0] + delta[0],
                               _pose(p, kind)[1] + delta[1]) for p in landmarks)
            if marks & used:
                continue
            placements.append((kind, delta, marks))
            used.update(marks)
            origins.append((top, left))
            break
        else:
            return None
    return tuple(placements)


def _force_states(rng, ink_mask, candidates):
    cells = np.argwhere(candidates)
    if len(cells) < 2:
        return False
    chosen = rng.choice(len(cells), size=2, replace=False)
    ink_mask[tuple(cells[int(chosen[0])])] = False
    ink_mask[tuple(cells[int(chosen[1])])] = True
    return True


def _make_pair(rng, colors, kinds, require_a):
    bg, material, ink, result, *identities = colors
    for _ in range(350):
        shape, holes, landmarks = _stencil(rng)
        canvas, source, board = _scene(rng, shape)
        placements = _sample_placements(rng, shape, landmarks, kinds, board)
        if placements is None:
            continue
        a = np.full(canvas, bg, dtype=np.int8)
        sr, sc, se, sf = source
        a[_rect(source)] = material
        for r, c in holes:
            a[sr + r, sc + c] = bg
        for color, (r, c) in zip(identities, landmarks):
            a[sr + r, sc + c] = color
        br, bc, be, ce = board
        for r, c in _boundary(be - br, ce - bc):
            a[br + r, bc + c] = result
        m = dict(shape=shape, holes=holes, board=board,
                 markers=frozenset(p for option in placements for p in option[2]))
        exposure, footprints = _coverage(a, m, placements)
        field = _field_mask(a, m)
        odd = field & (exposure % 2 == 1)
        even_shared = field & (exposure > 0) & (exposure % 2 == 0)
        material_only = field & (footprints > 0) & (exposure == 0)
        if not material_only.any():
            continue
        ink_mask = np.zeros(canvas, dtype=bool)
        ink_mask[br + 1:be - 1, bc + 1:ce - 1] = _texture(rng, be - br - 2, ce - bc - 2)
        if not _force_states(rng, ink_mask, odd):
            continue
        if len(kinds) > 1 and not _force_states(rng, ink_mask, even_shared):
            continue
        # A blank below solid material distinguishes holes from whole rectangles.
        wall_cell = np.argwhere(material_only)[int(rng.integers(np.count_nonzero(material_only)))]
        ink_mask[tuple(wall_cell)] = False
        outside = field & (exposure == 0)
        outside[tuple(wall_cell)] = False
        remaining = np.argwhere(outside)
        if not len(remaining):
            continue
        ink_mask[tuple(remaining[int(rng.integers(len(remaining)))])] = True
        a[br + 1:be - 1, bc + 1:ce - 1] = np.where(
            ink_mask[br + 1:be - 1, bc + 1:ce - 1], ink, bg)
        for kind, delta, _ in placements:
            for color, point in zip(identities, landmarks):
                q = _pose(point, kind)
                a[q[0] + delta[0], q[1] + delta[1]] = color
        # Ground truth is painted directly from the sampled coverage, not solve.
        b = a.copy()
        b[_rect(source)] = bg
        for r in range(br + 1, be - 1):
            for c in range(bc + 1, ce - 1):
                if (r, c) not in m["markers"] and int(exposure[r, c]) % 2:
                    b[r, c] = result if a[r, c] == bg else bg
        try:
            info = _pair_info(a, b)
            _require(sorted((k, d) for k, d, _ in info["placements"])
                     == sorted((k, d) for k, d, _ in placements), "sample/registration discrepancy")
            if require_a and not info["a_essential"]:
                continue
        except AssertionError:
            continue
        return {"input": a, "output": b}
    raise RuntimeError("could not sample diagnostic registered perforations")


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    fixed = bool(rng.random() < .4)
    palette = _palette(rng)
    withheld = int(rng.choice((2, 6, 7)))
    allowed = [k for k in range(8) if k != withheld]
    rotation = int(rng.choice([k for k in (1, 2, 3) if k != withheld]))
    reflection = int(rng.choice([k for k in (4, 5, 6, 7) if k != withheld]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                if index == 0:
                    kinds = [rotation]
                elif index == 1:
                    kinds = [reflection, int(rng.choice(allowed))]
                else:
                    kinds = list(map(int, rng.choice(allowed, size=int(rng.integers(1, 3)))))
            else:
                k = int(rng.choice((1, 2, 3), p=(.2, .3, .5)))
                kinds = [withheld] + list(map(int, rng.integers(8, size=k - 1)))
            for _ in range(20):
                colors = palette if fixed else _palette(rng)
                pair = _make_pair(rng, colors, kinds, split == "test" or index == 0)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not create distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            key = (a.shape, a.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _pair_info(a, b)
            _require(1 <= info["count"] <= 3, "generated placement count")
            _require(all(5 <= d <= 10 for d in info["shape"]), "generated stencil size")
            infos[split].append(info)
    first, second = infos["train"][:2]
    _require(first["count"] == 1 and first["placements"][0][0] in (1, 2, 3), "rotation demonstration")
    _require(first["a_essential"], "missing registration ablation witness")
    _require(second["count"] == 2 and any(k >= 4 for k, _, _ in second["placements"]), "overlap/reflection demonstration")
    train_poses = {k for m in infos["train"] for k, _, _ in m["placements"]}
    _require(all(m["count"] <= 2 for m in infos["train"]), "training placement limit")
    for info in infos["test"]:
        _require(info["a_essential"], "test registration can be omitted")
        _require(any(k not in train_poses for k, _, _ in info["placements"]), "missing novel-pose shift")


def _signature(puzzle):
    return tuple((split, pair["input"].shape, pair["input"].tobytes(), pair["output"].tobytes())
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded, generator-owned semantic and diversity checks over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, stencil_shapes, counts, apertures, palettes, poses, directions = set(), set(), set(), set(), set(), set(), set()
    crosses = triples = 0
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                a, b = pair["input"], pair["output"]
                before = a.copy()
                _require(np.array_equal(solve(a), b), "solver mismatch")
                _require(np.array_equal(a, before), "solver mutated input")
                m = _pair_info(a, b)
                shapes.add(a.shape)
                stencil_shapes.add(m["shape"])
                counts.add(m["count"])
                apertures.add(m["apertures"])
                palettes.add((m["bg"], m["material"], m["ink"], m["result"], m["identities"]))
                poses.update(k for k, _, _ in m["placements"])
                sr, sc, se, sf = m["source"]
                br, bc, be, ce = m["board"]
                directions.add("up" if se < br else "down" if be < sr else "left" if sf < bc else "right")
                crosses += m["wall_hole_crossing"]
                triples += m["triple_exposure"]
    _require(len({_signature(p) for p in puzzles}) == 42, "distinct seed puzzles")
    _require(len(shapes) >= 30 and len(stencil_shapes) >= 20, "geometry diversity")
    _require(len(apertures) >= 40 and len(palettes) >= 30, "aperture/palette diversity")
    _require(counts == {1, 2, 3} and poses == set(range(8)), "placement/pose diversity")
    _require(len(directions) == 4 and crosses > 20 and triples > 0, "interaction diversity")
    # Whole-scene symmetries and arbitrary color renaming must commute with
    # solving.  These test conventions independently of a generated pose label.
    rename = np.array([7, 0, 9, 2, 5, 1, 8, 4, 6, 3], dtype=np.int8)
    for p in puzzles[:3]:
        for pair in (p["train"][0], p["test"][0]):
            a, b = pair["input"], pair["output"]
            _require(np.array_equal(solve(np.rot90(a)), np.rot90(b)), "scene rotation equivariance")
            _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(b)), "scene reflection equivariance")
            _require(np.array_equal(solve(rename[a]), rename[b]), "color renaming equivariance")
    _require(_signature(generate_puzzle(19)) == _signature(generate_puzzle(19)), "determinism")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ns, nt, ns)
            _require(len(p["train"]) == nt and len(p["test"]) == ns, "explicit counts")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = puzzles[0]
    damaged = {s: [dict(pair) for pair in p[s]] for s in ("train", "test")}
    damaged["test"][0]["output"] = damaged["test"][0]["output"].copy()
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, semantic invariants)")
