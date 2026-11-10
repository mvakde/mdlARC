"""Registered color mosaic revealed through occluded toggle fences.

Only NumPy and the standard library are used.  `solve` reads pixels; the scene
descriptors used by the sampler never enter it.  All color roles are local.
"""

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any(g < 0) or np.any(g > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return g.astype(np.int8, copy=False)


def _key(g):
    return tuple(g.shape), g.astype(np.int8, copy=False).tobytes()


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _halo(points):
    return {q for y, x in points for q in
            ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))}


def _components(points):
    remaining, groups = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            y, x = stack.pop()
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    group.add(p)
        groups.append(group)
    return groups


def _window(g, bg):
    # The frame color is exclusive to this complete perimeter. Locate that
    # perimeter directly instead of flood-filling the large background/fill.
    candidates = []
    for color in np.unique(g):
        if color == bg:
            continue
        points = np.argwhere(g == color)
        t, l = map(int, points.min(axis=0))
        b, r = map(int, points.max(axis=0))
        if min(b - t - 1, r - l - 1) < 6:
            continue
        box = t, b, l, r
        ring = _perimeter(box)
        if len(points) != len(ring) or not all(int(g[p]) == int(color) for p in ring):
            continue
        fill = int(g[t + 1, l + 1])
        if fill != int(color) and np.all(g[t + 1:b, l + 1:r] == fill):
            candidates.append((box, fill, int(color)))
    assert len(candidates) == 1, "expected one uniform opaque window with a complete frame"
    return candidates[0]


def _fences(parts, mask):
    """Unique exact cover of exposed fragments by non-touching rectangular loops."""
    candidates = []
    for i, first in enumerate(parts):
        for j in range(i, len(parts)):
            visible = first if i == j else first | parts[j]
            box = _bbox(visible)
            t, b, l, r = box
            if min(b - t, r - l) < 3:
                continue
            full = _perimeter(box)
            if full - mask == visible and full & mask:
                candidates.append((frozenset((i, j)), box, full))
    answers = []

    def visit(remaining, chosen, occupied):
        if len(answers) > 1:
            return
        if not remaining:
            answers.append(tuple(chosen))
            return
        first = min(remaining)
        for ids, box, full in candidates:
            if first in ids and ids <= remaining and not full & _halo(occupied):
                visit(remaining - ids, chosen + [box], occupied | full)

    visit(set(range(len(parts))), [], set())
    assert len(answers) == 1, "ambiguous or incomplete fence geometry"
    assert 1 <= len(answers[0]) <= 3
    return tuple(sorted(answers[0]))


def _read(g):
    bg = int(g[0, 0])
    assert all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    window, fill, frame = _window(g, bg)
    mask = _area(window)
    exterior = set(map(tuple, np.argwhere(g != bg).tolist())) - mask
    swatches, squares, fragments = [], [], []
    for group in _components(exterior):
        t, b, l, r = _bbox(group)
        colors = {int(g[p]) for p in group}
        if len(group) == 1:
            swatches.append(next(iter(group)))
        elif (b - t == r - l and 2 <= b - t + 1 <= 4
              and len(group) == (b - t + 1) ** 2 and len(colors) == 1):
            squares.append((t, l, b - t + 1, next(iter(colors))))
        else:
            fragments.append(group)
    assert len(swatches) >= 3 and len(squares) >= 2 and fragments
    ink_colors = {int(g[p]) for group in fragments for p in group}
    assert len(ink_colors) == 1, "all toggle fences share one ink color"
    ink = next(iter(ink_colors))
    assert len({bg, frame, ink}) == 3 and fill not in (frame, ink)
    fences = _fences(fragments, mask)
    t, b, l, r = window
    inside = _area((t + 1, b - 1, l + 1, r - 1))
    assert all(_perimeter(box) & inside for box in fences)

    mt, mb, ml, mr = _bbox(swatches)
    assert (mb - mt) % 2 == 0 and (mr - ml) % 2 == 0
    assert all((y - mt) % 2 == 0 and (x - ml) % 2 == 0 for y, x in swatches)
    assert any(np.diff(sorted({y for y, x in swatches})) == 2)
    assert any(np.diff(sorted({x for y, x in swatches})) == 2)
    plan = g[mt:mb + 1:2, ml:mr + 1:2].copy()
    assert min(plan.shape) >= 2 and max(plan.shape) <= 5
    mini_cells = {(mt + 2 * y, ml + 2 * x) for y, x in np.ndindex(plan.shape)}
    assert all(int(g[p]) == bg for p in _area((mt, mb, ml, mr)) - mini_cells)
    assert np.count_nonzero(plan != bg) == len(swatches)
    assert not (set(map(int, plan.flat)) - {bg}) & {fill, frame, ink}
    assert not _area((mt, mb, ml, mr)) & mask

    if len({s[0] for s in squares}) == 1:
        axis = 1
    else:
        assert len({s[1] for s in squares}) == 1, "calibration squares must form one straight run"
        axis = 0
    squares.sort(key=lambda s: s[axis])
    side = squares[0][2]
    assert all(s[2] == side for s in squares)
    diffs = np.diff([s[axis] for s in squares])
    stride = int(diffs[0])
    assert np.all(diffs == stride) and stride > side
    length = len(squares)
    seq = tuple(s[3] for s in squares)
    matches = []
    for y in range(plan.shape[0] - (length - 1 if axis == 0 else 0)):
        for x in range(plan.shape[1] - (length - 1 if axis == 1 else 0)):
            if tuple(int(plan[y + (axis == 0) * k, x + (axis == 1) * k])
                     for k in range(length)) == seq:
                matches.append((y, x))
    assert len(matches) == 1, "calibration colors must select exactly one plan segment"
    match = matches[0]
    origin = squares[0][0] - match[0] * stride, squares[0][1] - match[1] * stride
    oy, ox = origin
    eh = (plan.shape[0] - 1) * stride + side
    ew = (plan.shape[1] - 1) * stride + side
    assert 0 <= oy and 0 <= ox and oy + eh <= g.shape[0] and ox + ew <= g.shape[1]
    assert not _area((oy, oy + eh - 1, ox, ox + ew - 1)) & _area((mt, mb, ml, mr))
    assert all(c != bg and c not in (fill, frame, ink) for c in seq)
    return dict(bg=bg, fill=fill, frame=frame, ink=ink, window=window,
                fences=fences, plan=plan, mini=(mt, ml), origin=origin,
                side=side, stride=stride, axis=axis, match=match, length=length)


def _tile_pixels(scene, yy, xx, origin=None):
    """Gather registered tile pixels on a crop, including blank/gutter masks."""
    oy, ox = scene["origin"] if origin is None else origin
    side, stride, plan = scene["side"], scene["stride"], scene["plan"]
    py, dy = np.divmod(yy - oy, stride)
    px, dx = np.divmod(xx - ox, stride)
    valid = ((0 <= py) & (py < plan.shape[0]) & (0 <= px) & (px < plan.shape[1])
             & (dy < side) & (dx < side))
    values = plan[np.clip(py, 0, plan.shape[0] - 1), np.clip(px, 0, plan.shape[1] - 1)]
    tiled = np.where(valid, values, scene["bg"])
    return tiled, valid, py, px


def _pixel_layers(scene, origin=None):
    """Small NumPy masks implement the same strict pixelwise inequalities."""
    t, b, l, r = scene["window"]
    yy = np.arange(t + 1, b)[:, None]
    xx = np.arange(l + 1, r)[None, :]
    shape = b - t - 1, r - l - 1
    depth = np.zeros(shape, dtype=np.int8)
    edges = np.zeros(shape, dtype=bool)
    for a, z, c, d in scene["fences"]:
        depth += (a < yy) & (yy < z) & (c < xx) & (xx < d)
        edges |= (((yy == a) | (yy == z)) & (c <= xx) & (xx <= d)
                  | ((xx == c) | (xx == d)) & (a <= yy) & (yy <= z))
    tiled, valid, py, px = _tile_pixels(scene, yy, xx, origin)
    return yy, xx, depth, edges, tiled, valid, py, px


def _center_gate(scene, py, px, origin=None):
    oy, ox = scene["origin"] if origin is None else origin
    cy = 2 * (oy + py * scene["stride"]) + scene["side"] - 1
    cx = 2 * (ox + px * scene["stride"]) + scene["side"] - 1
    depth = np.zeros(np.broadcast_shapes(cy.shape, cx.shape), dtype=np.int8)
    for a, z, c, d in scene["fences"]:
        depth += (2 * a < cy) & (cy < 2 * z) & (2 * c < cx) & (cx < 2 * d)
    return depth % 2 == 1


def _paint(scene, edges, tiled, allowed, use_tiles=True):
    output = np.where(use_tiles & allowed & (tiled != scene["bg"]), tiled, scene["fill"])
    output[edges] = scene["ink"]
    return output.astype(np.int8, copy=False)


def _answer(scene, gate="parity", origin=None, use_tiles=True):
    """Pixelwise interpretation used by the input-only solver and ablations."""
    _, _, depth, edges, tiled, _, py, px = _pixel_layers(scene, origin)
    allowed = depth % 2 == 1 if gate == "parity" else depth > 0 if gate == "union" else True
    if gate == "center":
        allowed = _center_gate(scene, py, px, origin)
    return _paint(scene, edges, tiled, allowed, use_tiles)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete fences, register the motif, intersect it with odd enclosure depth."""
    return _answer(_read(_grid(input_grid)))


def _randint(rng, low, high):
    return int(rng.integers(low, high + 1))


def _geometry(rng, h, w, count, style):
    # These local boxes are later freely reflected/rotated with the whole input.
    if count == 1:
        oh, ow = _randint(rng, 7, min(14, h - 8)), _randint(rng, 7, min(12, w - 7))
        t, l = _randint(rng, 3, h - oh - 4), _randint(rng, 3, w - ow - 4)
        main = t, t + oh + 1, l, l + ow + 1
        mask = _area(main)
        inside = _area((main[0] + 1, main[1] - 1, main[2] + 1, main[3] - 1))
        for _ in range(100):
            a, z = sorted(map(int, rng.choice(np.arange(1, h - 1), 2, replace=False)))
            c, d = sorted(map(int, rng.choice(np.arange(1, w - 1), 2, replace=False)))
            box = a, z, c, d
            full, visible = _perimeter(box), _perimeter(box) - mask
            if min(z - a, d - c) >= 4 and len(full & inside) >= 4 and visible and _bbox(visible) == box:
                return main, [box]
        return None
    depth = count if style == "nested" else count - 1 if count == 3 else 1
    need = 2 * depth + 1
    t, l = _randint(rng, need, min(need + 3, h - 12)), _randint(rng, need, min(need + 2, w - 10))
    b, r = _randint(rng, t + 8, h - 3), _randint(rng, l + 8, w - 3)
    main = t, b, l, r
    a, c = _randint(rng, 1, t - 2 * depth), _randint(rng, 1, l - 2 * depth)
    z = _randint(rng, t + 2 * depth, b - (4 if style == "opposed" else 1))
    d = _randint(rng, l + 2 * depth, r - (4 if style == "opposed" else 1))
    boxes = []
    for k in range(depth):
        boxes.append((a, z, c, d))
        if k + 1 < depth:
            a = _randint(rng, a + 2, t - 2 * (depth - k - 1))
            c = _randint(rng, c + 2, l - 2 * (depth - k - 1))
            z = _randint(rng, t + 2 * (depth - k - 1), z - 2)
            d = _randint(rng, l + 2 * (depth - k - 1), d - 2)
    if style == "opposed":
        z0, d0 = boxes[0][1], boxes[0][3]
        boxes.append((_randint(rng, z0 + 2, b - 1), _randint(rng, b + 1, h - 2),
                      _randint(rng, d0 + 2, r - 1), _randint(rng, r + 1, w - 2)))
    return main, boxes


def _asymmetric(plan):
    variants = [np.flipud(plan), np.fliplr(plan), np.flip(plan, (0, 1))]
    if plan.shape[0] == plan.shape[1]:
        variants += [plan.T, np.rot90(plan), np.rot90(plan, 3), np.flip(plan.T, (0, 1))]
    return all(not np.array_equal(plan, p) for p in variants)


def _sample_plan(rng, ph, pw, bg, palette, match, length, holes, demo):
    # Repeated colors/rows are legal: only the actual calibration sequence must
    # identify its position. Blanks do not shrink the inferred miniature extent.
    for _ in range(60):
        plan = rng.choice(palette, size=(ph, pw)).astype(np.int8)
        blank = rng.random((ph, pw)) < holes
        blank[match[0], match[1]:match[1] + length] = False
        plan[blank] = bg
        if demo and not _asymmetric(plan):
            continue
        if any(np.all(edge == bg) for edge in (plan[0], plan[-1], plan[:, 0], plan[:, -1])):
            continue
        if holes >= .1 and not np.any(plan == bg):
            continue
        if any(not np.any(np.diff(np.unique(np.argwhere(plan != bg)[:, d])) == 1) for d in (0, 1)):
            continue
        sequence = tuple(map(int, plan[match[0], match[1]:match[1] + length]))
        matches = [(y, x) for y in range(ph) for x in range(pw - length + 1)
                   if tuple(map(int, plan[y, x:x + length])) == sequence]
        if matches == [match]:
            return plan
    return None


def _render(shape, window, boxes, plan, mini, origin, side, stride, match, length, colors):
    """Truth from complete layer masks; then draw only clues and apply occlusion."""
    bg, fill, frame, ink = colors
    h, w = shape
    grid = np.full(shape, bg, dtype=np.int8)
    edge_mask = np.zeros(shape, dtype=bool)
    depth = np.zeros(shape, dtype=np.int8)
    for a, z, c, d in boxes:
        depth[a + 1:z, c + 1:d] += 1
        for p in _perimeter((a, z, c, d)):
            edge_mask[p] = True
            grid[p] = ink
    tiled = np.full(shape, bg, dtype=np.int8)
    for (y, x), value in np.ndenumerate(plan):
        yy, xx = origin[0] + y * stride, origin[1] + x * stride
        tiled[yy:yy + side, xx:xx + side] = value
    revealed = np.full(shape, fill, dtype=np.int8)
    paint = (depth % 2 == 1) & (tiled != bg)
    revealed[paint] = tiled[paint]
    revealed[edge_mask] = ink
    t, b, l, r = window
    output = revealed[t + 1:b, l + 1:r].copy()
    for k in range(length):
        yy = origin[0] + match[0] * stride
        xx = origin[1] + (match[1] + k) * stride
        grid[yy:yy + side, xx:xx + side] = plan[match[0], match[1] + k]
    mt, ml = mini
    grid[mt:mt + 2 * plan.shape[0]:2, ml:ml + 2 * plan.shape[1]:2] = plan
    grid[t + 1:b, l + 1:r] = fill
    for p in _perimeter(window):
        grid[p] = frame
    return {"input": grid, "output": output}


def _metrics(scene):
    # Reuse crop masks for every witness; do not rerender each ablation pixel by
    # pixel or recompute the same enclosure depth for every tile.
    yy, xx, depth, edges, tiled, valid, py, px = _pixel_layers(scene)
    odd = depth % 2 == 1
    answer = _paint(scene, edges, tiled, odd)
    oy, ox = scene["origin"]
    occupied = (tiled != scene["bg"]) & ~edges
    sites = py * scene["plan"].shape[1] + px
    even_sites = np.unique(sites[occupied & ~odd])
    odd_sites = np.unique(sites[occupied & odd])
    clip = bool(np.intersect1d(even_sites, odd_sites).size)
    payload = set(map(int, answer.flat)) - {scene["fill"], scene["ink"]}
    painted = (answer != scene["fill"]) & (answer != scene["ink"])
    depths, counts = np.unique(depth[painted], return_counts=True)
    painted_depths = {int(d): int(n) for d, n in zip(depths, counts)}
    blank_active = bool(np.any(valid & (tiled == scene["bg"]) & ~edges & odd))
    # Assign the observed first square to plan (0,0), an ordinal/layout shortcut.
    wrong_origin = (oy + scene["match"][0] * scene["stride"],
                    ox + scene["match"][1] * scene["stride"])
    wrong_tiles = _tile_pixels(scene, yy, xx, wrong_origin)[0]
    return dict(count=len(scene["fences"]), depth=int(depth.max()), holes=int(np.count_nonzero(scene["plan"] == scene["bg"])),
                clip=bool(clip), payload=len(payload),
                painted_depths=dict(painted_depths), blank_active=blank_active,
                no_a_diff=not np.array_equal(answer, _paint(scene, edges, tiled, True)),
                no_b_diff=not np.array_equal(answer, _paint(scene, edges, tiled, odd, False)),
                union_diff=not np.array_equal(answer, _paint(scene, edges, tiled, depth > 0)),
                center_diff=not np.array_equal(answer, _paint(scene, edges, tiled, _center_gate(scene, py, px))),
                wrong_origin_diff=not np.array_equal(answer, _paint(scene, edges, wrong_tiles, odd)))


def _make_pair(rng, bg, count, style, demo, test, axis, require_clip=False, avoid=None):
    # Bounded scene/registration sampling. No generation fallback changes a rule.
    for _ in range(500):
        ph = _randint(rng, 3 if test else 2, 5)
        pw = _randint(rng, 3 if test else 2, 4)
        if avoid is not None and (ph, pw) == (avoid["plan"].shape if axis == 1 else avoid["plan"].T.shape):
            continue
        sw = _randint(rng, 20, 30 - (2 * pw - 1) - 3)
        sh = _randint(rng, 23, 29)
        try:
            geometry = _geometry(rng, sh, sw, count, style)
        except ValueError:
            continue
        if geometry is None:
            continue
        window, boxes = geometry
        sides = [s for s in (2, 3, 4) if (avoid is None or s != avoid["side"]) and (s >= 3 or not require_clip)]
        side = int(rng.choice(sides))
        stride = side + _randint(rng, 1, 3)
        eh, ew = (ph - 1) * stride + side, (pw - 1) * stride + side
        if eh > sh - 2 or ew > sw - 2:
            continue
        blocked = _halo(set().union(*(_perimeter(box) - _area(window) for box in boxes))) | _area(window)
        poses = []
        for _ in range(70):
            origin = _randint(rng, 1, sh - eh - 1), _randint(rng, 1, sw - ew - 1)
            length = _randint(rng, 2, min(pw, 3))
            rr = _randint(rng, 1, ph - 2) if test else _randint(rng, 0, ph - 1)
            cc = _randint(rng, 0, pw - length)
            if demo and (rr, cc) == (0, 0):
                continue
            cells = {(origin[0] + rr * stride + dy, origin[1] + (cc + k) * stride + dx)
                     for k in range(length) for dy in range(side) for dx in range(side)}
            if not cells & blocked:
                poses.append((origin, (rr, cc), length))
        if not poses:
            continue
        rng.shuffle(poses)
        colors = [int(v) for v in rng.permutation([c for c in range(10) if c != bg])]
        fill, frame, ink = colors[:3]
        if demo == 1 or (demo != 2 and rng.random() < .3):
            fill = bg
        palette = colors[3:3 + _randint(rng, 2, 6)]
        holes = .18 if demo >= 2 or test else float(rng.choice([0, .12, .23]))
        mini = _randint(rng, 1, sh - (2 * ph - 1) - 1), sw + 1
        shape = sh, sw + (2 * pw - 1) + 3
        for origin, match, length in poses[:12]:
            plan = _sample_plan(rng, ph, pw, bg, palette, match, length, holes, bool(demo))
            if plan is None:
                continue
            pair = _render(shape, window, boxes, plan, mini, origin, side, stride, match, length,
                           (bg, fill, frame, ink))
            try:
                scene = _read(pair["input"])
                assert np.array_equal(_answer(scene), pair["output"])
                m = _metrics(scene)
                assert m["payload"] >= 2 and m["no_a_diff"] and m["no_b_diff"]
                assert scene["fill"] in pair["output"]
                assert not require_clip or m["clip"]
                assert demo < 2 or style != "nested" or m["union_diff"]
                assert demo != 2 or m["blank_active"]
                assert demo != 3 or m["painted_depths"].get(3, 0) >= 2
                assert not demo or m["wrong_origin_diff"]
                assert m["count"] == count
            except AssertionError:
                continue
            turns = int(rng.choice([0, 2] if axis == 1 else [1, 3]))
            flip = bool(rng.integers(2))
            pair = {k: np.rot90(v, turns).copy() for k, v in pair.items()}
            if flip:
                pair = {k: np.fliplr(v).copy() for k, v in pair.items()}
            oriented = _read(pair["input"])
            if demo and not _metrics(oriented)["wrong_origin_diff"]:
                continue
            if avoid is not None and (oriented["match"] == avoid["match"] or oriented["plan"].shape == avoid["plan"].shape):
                continue
            return pair
    raise RuntimeError("bounded sampling could not find an informative scene")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """One fixed pixel rule, with varied local role palettes and registrations."""
    seed = _integer(seed, "seed")
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(seed % (1 << 64))
    axis_mode = int(rng.integers(3))
    axes = [1] if axis_mode == 0 else [0] if axis_mode == 1 else [0, 1]
    backgrounds = list(map(int, rng.permutation(10)))
    result, seen = {"train": [], "test": []}, set()
    first_scene = None
    for split, n in (("train", nt), ("test", ns)):
        for i in range(n):
            test = split == "test"
            demo = 0 if test else i + 1 if i < 3 else 0
            count = 3 if test or demo == 3 else 1 if demo == 1 else 2 if demo == 2 else _randint(rng, 1, 2)
            style = "nested" if demo in (2, 3) else "opposed" if test and nt == 2 else str(rng.choice(["nested", "opposed"]))
            axis = axes[i % len(axes)] if not test else int(rng.choice(axes))
            for _ in range(8):
                pair = _make_pair(rng, backgrounds[len(seen) % 10], count, style, demo, test, axis,
                                  require_clip=demo == 1, avoid=first_scene if demo == 2 else None)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("duplicate input after bounded retries")
            seen.add(key)
            result[split].append(pair)
            if demo == 1:
                first_scene = _read(pair["input"])
        rng.shuffle(result[split])
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, scenes, metrics = set(), {"train": [], "test": []}, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            scene = _read(inp)
            assert np.array_equal(out, _answer(scene)), "incorrect toggle-stencil crop"
            m = _metrics(scene)
            assert m["payload"] >= 2 and m["no_a_diff"] and m["no_b_diff"]
            assert scene["fill"] in out and scene["ink"] in out
            # A source input has no miniature-plus-calibration-plus-window scene.
            # Reconstruct all foreground pixels to reject noise or clue damage.
            bg_canvas = np.full(inp.shape, scene["bg"], dtype=np.int8)
            for box in scene["fences"]:
                for p in _perimeter(box) - _area(scene["window"]):
                    bg_canvas[p] = scene["ink"]
            mt, ml = scene["mini"]
            ph, pw = scene["plan"].shape
            bg_canvas[mt:mt + 2 * ph:2, ml:ml + 2 * pw:2] = scene["plan"]
            for k in range(scene["length"]):
                py = scene["match"][0] + (scene["axis"] == 0) * k
                px = scene["match"][1] + (scene["axis"] == 1) * k
                yy, xx = scene["origin"][0] + py * scene["stride"], scene["origin"][1] + px * scene["stride"]
                bg_canvas[yy:yy + scene["side"], xx:xx + scene["side"]] = scene["plan"][py, px]
            t, b, l, r = scene["window"]
            bg_canvas[t + 1:b, l + 1:r] = scene["fill"]
            for p in _perimeter(scene["window"]):
                bg_canvas[p] = scene["frame"]
            assert np.array_equal(bg_canvas, inp), "unexpected or overwritten input material"
            scenes[split].append(scene)
            metrics[split].append(m)
    train = scenes["train"]
    assert len({s["bg"] for s in train}) >= 2
    assert len({s["side"] for s in train}) >= 2
    assert len({s["match"] for s in train}) >= 2 and len({s["plan"].shape for s in train}) >= 2
    assert any(m["count"] == 1 and m["clip"] and m["center_diff"] for m in metrics["train"])
    assert any(m["depth"] == 2 and m["union_diff"] for m in metrics["train"])
    assert any(m["blank_active"] for m in metrics["train"])
    assert {s["fill"] == s["bg"] for s in train} == {False, True}
    assert sum(m["wrong_origin_diff"] for m in metrics["train"]) >= 2
    taught_axes = {s["axis"] for s in train}
    assert {s["axis"] for s in scenes["test"]} <= taught_axes
    assert max(m["depth"] for m in metrics["test"]) <= max(m["depth"] for m in metrics["train"])
    if any(m["depth"] == 3 for m in metrics["test"]):
        assert any(m["painted_depths"].get(3, 0) >= 2 for m in metrics["train"])
    for s, m in zip(scenes["test"], metrics["test"]):
        assert m["count"] == 3
        assert 0 < s["match"][1 - s["axis"]] < s["plan"].shape[1 - s["axis"]] - 1


def self_test():
    """Bounded checks of 42 seeds, teaching witnesses, diversity and failures."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, layouts = set(), set(), set()
    data = []
    for puzzle in puzzles:
        signatures.add(tuple(_key(p[k]) for split in ("train", "test") for p in puzzle[split] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _read(pair["input"])
                m = _metrics(scene)
                data.append((scene, m, split))
                shapes.add(pair["input"].shape)
                layouts.add((scene["window"], scene["fences"], scene["plan"].shape,
                             scene["origin"], scene["side"], scene["stride"], scene["match"]))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
    assert len(signatures) == 42 and len(shapes) >= 20 and len(layouts) >= 180
    assert {s["bg"] for s, m, split in data} == set(range(10))
    assert {s["axis"] for s, m, split in data} == {0, 1}
    assert {m["depth"] for s, m, split in data} >= {1, 2, 3}
    assert {s["side"] for s, m, split in data} == {2, 3, 4}
    assert len({s["plan"].shape for s, m, split in data}) >= 8
    assert len({s["match"] for s, m, split in data if split == "test"}) >= 8
    assert len({s["stride"] - s["side"] for s, m, split in data}) >= 2
    assert any(s["length"] >= 3 for s, m, split in data)
    assert sum(m["clip"] for s, m, split in data) >= 42
    assert any(m["union_diff"] for s, m, split in data if split == "test")
    for a, b in ((2, 1), (3, 2), (5, 4), (10, 4)):
        p, q = generate_puzzle(1800 + a, a, b), generate_puzzle(1800 + a, a, b)
        assert all(np.array_equal(x[k], y[k]) for split in ("train", "test")
                   for x, y in zip(p[split], q[split]) for k in ("input", "output"))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, pixel solver, ablations and diversity")
