"""Directional hollow frames export their payload cell count past the tip.

There is no color dictionary: background and both object roles are local.
Construction uses variable boundary walks; recognition uses monochromatic
8-connected components and cross-sections, independently of those walks.
"""
import numpy as np


_ORIENTATIONS = ("U", "R", "D", "L")
_DIRECTION = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}


def _components(cells, diagonal=True):
    remaining = set(cells)
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    stack.append(point)
        result.append(component)
    return result


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _canonical(point, orientation):
    r, c = point
    return {"U": (r, c), "R": (-c, r), "D": (-r, -c), "L": (c, -r)}[orientation]


def _world(point, orientation):
    y, x = point
    return {"U": (y, x), "R": (x, -y), "D": (-y, -x), "L": (-x, y)}[orientation]


def _ray_cells(height, width, tip, orientation, quantity=None):
    dr, dc = _DIRECTION[orientation]
    r, c = tip
    cells = []
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < height and 0 <= c < width):
            break
        cells.append((r, c))
        if quantity is not None and len(cells) >= quantity:
            break
    return tuple(cells)


def _read_component(frame, orientation):
    """Read a pointed envelope, allowing thick caps and a shorter side.

    Once the two sides separate, missing terminal side sections continue
    at their last inner boundary, only to the other side's endpoint. This
    includes the payload near the original green frame's unequal opening.
    """
    local = {_canonical(p, orientation) for p in frame}
    ys = [p[0] for p in local]
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 2:
        return None
    rows = [sorted(x for y, x in local if y == level) for level in range(y0, y1 + 1)]
    if len(rows[0]) != 1 or any(not row for row in rows):
        return None
    tip_x = rows[0][0]
    cavity = set()
    split = False
    missing_side = None
    left_outer = right_outer = left_inner = right_inner = tip_x
    for offset, row in enumerate(rows[1:], 1):
        runs = []
        for x in row:
            if not runs or x != runs[-1][-1] + 1:
                runs.append([x])
            else:
                runs[-1].append(x)
        if len(runs) > 2:
            return None
        lo, hi = row[0], row[-1]
        if len(runs) == 1 and not split:
            if lo > left_outer or hi < right_outer:
                return None
            left_outer, right_outer = lo, hi
            continue
        if len(runs) == 2:
            if missing_side is not None or lo > left_outer or hi < right_outer:
                return None
            left_inner, right_inner = runs[0][-1], runs[1][0]
            left_outer, right_outer = lo, hi
            split = True
        else:
            # An unequal arm can stop, but cannot resume or switch sides.
            if hi <= left_inner and missing_side in (None, "right"):
                if lo > left_outer:
                    return None
                left_outer, left_inner, missing_side = lo, hi, "right"
            elif lo >= right_inner and missing_side in (None, "left"):
                if hi < right_outer:
                    return None
                right_inner, right_outer, missing_side = lo, hi, "left"
            else:
                return None
        if right_inner - left_inner < 2:
            return None
        cavity.update(_world((y0 + offset, x), orientation)
                      for x in range(left_inner + 1, right_inner))
    if not cavity or not left_outer < tip_x < right_outer:
        return None
    return _world((y0, tip_x), orientation), cavity


def _find_objects(grid):
    background = _background(grid)
    found = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != background):
        for frame in _components(map(tuple, np.argwhere(grid == color))):
            if len(frame) < 5:
                continue
            candidates = []
            for orientation in _ORIENTATIONS:
                geometry = _read_component(frame, orientation)
                if geometry is None:
                    continue
                tip, cavity = geometry
                if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) for r, c in cavity):
                    continue
                values = {int(grid[p]) for p in cavity} - {background}
                if len(values) != 1 or color in values:
                    continue
                payload_color = next(iter(values))
                payload = {p for p in cavity if grid[p] == payload_color}
                ray = _ray_cells(*grid.shape, tip, orientation, len(payload))
                if any(grid[p] != background for p in ray):
                    continue
                rs, cs = zip(*frame)
                bbox = {(r, c) for r in range(min(rs), max(rs) + 1)
                        for c in range(min(cs), max(cs) + 1)}
                candidates.append(dict(frame=frame, cavity=cavity, tip=tip,
                    orientation=orientation, frame_color=color, payload_color=payload_color,
                    payload_cells=payload, quantity=len(payload), ray=ray,
                    available=len(_ray_cells(*grid.shape, tip, orientation)), bbox=bbox))
            found.extend(candidates)
    # A payload cluster can coincidentally resemble a tiny arrow. Its cells
    # still belong to the larger enclosing hollow, not to a nested object.
    outer = [obj for obj in found if not any(
        obj is not other and obj["frame"] <= other["cavity"] for other in found)]
    assert len({frozenset(o["frame"]) for o in outer}) == len(outer), "frame has ambiguous pointing direction"
    return sorted(outer, key=lambda o: min(o["frame"]))


def _checked_objects(grid):
    objects = _find_objects(grid)
    assert objects, "no directional frame found"
    occupied, rays = set(), set()
    for obj in objects:
        assert occupied.isdisjoint(obj["bbox"]), "overlapping object regions"
        assert rays.isdisjoint(obj["ray"]), "overlapping rays"
        occupied.update(obj["bbox"])
        rays.update(obj["ray"])
    assert occupied.isdisjoint(rays), "ray enters another object"
    return objects


def _solve_grid(grid):
    objects = _checked_objects(grid)
    output = grid.copy()
    for obj in objects:
        for cell in obj["payload_cells"]:
            output[cell] = _background(grid)
        for cell in obj["ray"]:
            output[cell] = obj["payload_color"]
    return output


def _profile(rng, anchor=False, small=False):
    """Sample monotone, possibly asymmetric boundary walks, not a stencil."""
    if anchor:
        a, b = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        tail, stem = int(rng.integers(3, 6)), int(rng.integers(0, 2))
    elif small:
        a, b, tail, stem = 1, int(rng.integers(1, 3)), int(rng.integers(1, 4)), 0
    else:
        a, b = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        tail, stem = int(rng.integers(1, 7)), int(rng.integers(0, 3))
    symmetric = bool(rng.random() < .32)
    if symmetric:
        b = a
    rows = [(0, 0)] * (stem + 1)
    l = r = 0
    while l < a or r < b:
        dl = l < a and (symmetric or r == b or rng.random() < .8)
        dr = r < b and (symmetric or l == a or rng.random() < .8)
        if not dl and not dr:
            dl = l < a
            dr = not dl
        l, r = l + int(dl), r + int(dr)
        rows.append((-l, r))
    rows.extend([(-a, b)] * tail)
    thick = bool(rng.random() < .32)
    short_side = int(rng.integers(0, 3)) if tail >= 3 and rng.random() < .5 else 0
    trim = int(rng.integers(1, min(2, tail - 2) + 1)) if short_side else 0
    frame, cavity = set(), set()
    for y, (lo, hi) in enumerate(rows):
        thickness = 2 if thick and y <= len(rows) - tail else 1
        li, ri = lo + thickness - 1, hi - thickness + 1
        if li >= ri:
            frame.update((y, x) for x in range(lo, hi + 1))
            continue
        if not (short_side == 1 and y >= len(rows) - trim):
            frame.update((y, x) for x in range(lo, li + 1))
        if not (short_side == 2 and y >= len(rows) - trim):
            frame.update((y, x) for x in range(ri, hi + 1))
        cavity.update((y, x) for x in range(li + 1, ri))
    # Translate only; the reader has no knowledge of these sampled walks.
    frame = {(y, x + a) for y, x in frame}
    cavity = {(y, x + a) for y, x in cavity}
    return dict(frame=frame, cavity=cavity, tip=(0, a))


def _orient(shape, orientation):
    points = {p: _world(p, orientation) for p in shape["frame"] | shape["cavity"] | {shape["tip"]}}
    min_r = min(p[0] for p in points.values())
    min_c = min(p[1] for p in points.values())
    def transform(p):
        r, c = points[p]
        return r - min_r, c - min_c
    frame = {transform(p) for p in shape["frame"]}
    return dict(frame=frame, cavity={transform(p) for p in shape["cavity"]},
                tip=transform(shape["tip"]), orientation=orientation,
                height=max(p[0] for p in frame) + 1, width=max(p[1] for p in frame) + 1)


def _payload(rng, cavity, quantity, arrangement):
    cells = sorted(cavity)
    start = cells[int(rng.integers(len(cells)))]
    if arrangement == "compact":
        # Grow a connected cluster. Canonical cavities are connected.
        selected = {start}
        while len(selected) < quantity:
            frontier = sorted(p for p in cavity - selected
                              if any(abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1 for q in selected))
            if not frontier:
                frontier = sorted(cavity - selected)
            selected.add(frontier[int(rng.integers(len(frontier)))])
        return selected
    if arrangement in ("sparse", "extended"):
        selected = {start}
        while len(selected) < quantity:
            candidates = sorted(cavity - selected)
            distances = [min(max(abs(p[0]-q[0]), abs(p[1]-q[1])) for q in selected) for p in candidates]
            best = max(distances)
            choices = [p for p, distance in zip(candidates, distances) if distance == best]
            selected.add(choices[int(rng.integers(len(choices)))])
        return selected
    return {cells[int(i)] for i in rng.choice(len(cells), quantity, replace=False)}


def _make_spec(rng, directions, palette, shape=None, quantity=None, colors=None,
               arrangement=None, clearance=None):
    shape = shape or _profile(rng, small=rng.random() < .3)
    oriented = _orient(shape, str(rng.choice(directions)))
    capacity = len(oriented["cavity"])
    if capacity < 2:
        return None
    clearance = clearance or str(rng.choice(["free", "free", "exact", "clip", "any"]))
    arrangement = arrangement or str(rng.choice(["compact", "sparse", "extended", "random", "dense"]))
    if quantity is None:
        axial_size = oriented["height"] if oriented["orientation"] in ("U", "D") else oriented["width"]
        limit = 29 - axial_size if clearance == "free" else 30 - axial_size if clearance == "exact" else 18
        high = min(18, capacity - 1, limit)
        quantity = max(1, int(.8 * high)) if arrangement == "dense" else int(rng.integers(1, high + 1))
    if quantity >= capacity:
        return None
    if quantity == 1 and clearance == "clip":
        clearance = "zero"
    colors = colors or tuple(int(v) for v in rng.choice(palette, 2, replace=False))
    oriented.update(payload=_payload(rng, oriented["cavity"], quantity, arrangement),
                    frame_color=colors[0], payload_color=colors[1], quantity=quantity,
                    clearance=clearance)
    return oriented


def _placements(spec, height, width, rng):
    h, w = spec["height"], spec["width"]
    positions = []
    for r in range(height - h + 1):
        for c in range(width - w + 1):
            tr, tc = spec["tip"][0] + r, spec["tip"][1] + c
            room = {"U": tr, "D": height - 1 - tr, "L": tc, "R": width - 1 - tc}[spec["orientation"]]
            q, mode = spec["quantity"], spec["clearance"]
            if (mode == "free" and room <= q or mode == "exact" and room != q
                    or mode == "clip" and not 0 < room < q or mode == "zero" and room != 0):
                continue
            positions.append((r, c))
    return [positions[int(i)] for i in rng.permutation(len(positions))]


def _pack(rng, height, width, specs, required):
    placed, boxes, rays, guards = [], set(), set(), set()
    for index, spec in enumerate(specs):
        accepted = None
        for r, c in _placements(spec, height, width, rng):
            box = {(r+y, c+x) for y in range(spec["height"]) for x in range(spec["width"])}
            if not box.isdisjoint(guards | rays):
                continue
            tip = (r + spec["tip"][0], c + spec["tip"][1])
            # Reserve one empty cell after complete rays, so stopping is visible.
            ray_guard = set(_ray_cells(height, width, tip, spec["orientation"], spec["quantity"] + 1))
            if not ray_guard.isdisjoint(boxes | rays):
                continue
            accepted = (r, c, tip, box, ray_guard)
            break
        if accepted is None:
            if index < required:
                return None
            continue
        r, c, tip, box, ray_guard = accepted
        placed.append((spec, r, c, tip))
        boxes.update(box)
        rays.update(ray_guard)
        guards.update((y+dy, x+dx) for y, x in box for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return placed


def _scene(rng, directions, count, anchors=(), background=None, larger=False, require_free=False):
    background = int(background if background is not None else (0 if rng.random() < .5 else rng.integers(10)))
    mandatory_colors = {c for a in anchors for c in a["colors"]}
    available = [c for c in range(10) if c != background and c not in mandatory_colors]
    palette_size = int(rng.integers(max(2, len(mandatory_colors)), 10))
    palette = sorted(mandatory_colors | {int(c) for c in rng.choice(available, palette_size-len(mandatory_colors), replace=False)})
    specs = []
    for anchor in anchors:
        specs.append(_make_spec(rng, [anchor["direction"]], palette, shape=anchor["shape"],
            quantity=anchor["q"], colors=anchor["colors"], arrangement=anchor["pattern"], clearance=anchor["clearance"]))
    required = max(1, len(specs))
    # A fresh local palette still allows intentional role sharing between objects.
    color_mode = str(rng.choice(["independent", "frame", "payload", "roles"]))
    for index in range(len(specs), count):
        colors = None
        if specs:
            f, p = specs[0]["frame_color"], specs[0]["payload_color"]
            if color_mode == "frame":
                colors = (f, int(rng.choice([c for c in palette if c != f])))
            elif color_mode == "payload":
                colors = (int(rng.choice([c for c in palette if c != p])), p)
            elif color_mode == "roles":
                colors = (p, f) if index % 2 else (f, p)
        spec = _make_spec(rng, directions, palette, colors=colors,
                          clearance="free" if not specs and require_free else ("zero" if rng.random() < .035 else None))
        if spec is not None:
            specs.append(spec)
    if not specs:
        shape = dict(frame={(0,1),(1,0),(1,2),(2,0),(2,2)}, cavity={(1,1),(2,1)}, tip=(0,1))
        specs.append(_make_spec(rng, directions, palette, shape=shape, quantity=1, clearance="free"))
    low = 8 if count == 1 and not anchors else 13
    if larger:
        low = 22
    height, width = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    if not anchors and count == 1 and not larger and rng.random() < .4:
        primary = specs[0]
        vertical = primary["orientation"] in ("U", "D")
        room = primary["quantity"] + 1 if primary["clearance"] == "free" else primary["quantity"] if primary["clearance"] == "exact" else 1
        min_h = primary["height"] + (room if vertical else 1)
        min_w = primary["width"] + (1 if vertical else room)
        height = int(rng.integers(min_h, min(30, min_h + 6) + 1))
        width = int(rng.integers(min_w, min(30, min_w + 6) + 1))
    placed = None
    # Enumerated coordinates, a bounded number of layouts, then a roomy fallback.
    for attempt in range(9):
        if attempt >= 5:
            height = width = 30
        placed = _pack(rng, height, width, specs, required)
        if placed is not None:
            break
    # On the final 30x30 attempt the required frames fit: there is only one
    # free-ray object, or two <=7-cell-wide anchors sharing the same axis.
    assert placed is not None, "required demonstrations do not fit"
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    protected = set()
    for spec, r, c, tip in placed:
        for y, x in spec["frame"]:
            inp[r+y, c+x] = out[r+y, c+x] = spec["frame_color"]
        for y, x in spec["payload"]:
            inp[r+y, c+x] = spec["payload_color"]
        for cell in _ray_cells(height, width, tip, spec["orientation"], spec["quantity"]):
            out[cell] = spec["payload_color"]
        protected.update((r+y, c+x) for y in range(-2, spec["height"]+2) for x in range(-2, spec["width"]+2))
        protected.update(_ray_cells(height, width, tip, spec["orientation"]))
    choices = [(r,c) for r in range(height) for c in range(width) if (r,c) not in protected]
    for _ in range(int(rng.choice([0,0,0,1,1,2]))):
        if not choices:
            break
        cell = choices[int(rng.integers(len(choices)))]
        color = int(rng.choice(palette))
        inp[cell] = out[cell] = color
        choices = [p for p in choices if max(abs(p[0]-cell[0]), abs(p[1]-cell[1])) > 1]
    observed = _checked_objects(inp)
    expected = {(tip, spec["orientation"], spec["quantity"], spec["frame_color"], spec["payload_color"])
                for spec, _, _, tip in placed}
    assert expected == {(o["tip"], o["orientation"], o["quantity"], o["frame_color"], o["payload_color"]) for o in observed}, "construction/reader disagreement"
    assert np.array_equal(out, _solve_grid(inp)), "independent construction output disagrees with input reader"
    return {"input": inp, "output": out}


_SOURCE_RAW = (
    (
        (
            "000000000000",
            "004400000000",
            "600644000000",
            "006044400000",
            "004444000000",
            "000000000000",
            "000000000000",
            "000000333300",
            "000033700000",
            "000033070000",
            "000333000000",
            "000000333000",
        ),
        (
            "000000000000",
            "004400000000",
            "600044000000",
            "000044466000",
            "004444000000",
            "000000000000",
            "000000000000",
            "000000333300",
            "000033000000",
            "000033000000",
            "077333000000",
            "000000333000",
        ),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0030000600",
            "0030006060",
            "0393006960",
            "3900300000",
            "0090300000",
            "0000300000",
            "0000000000",
        ),
        (
            "0090000000",
            "0090000000",
            "0090000900",
            "0030000600",
            "0030006060",
            "0303006060",
            "3000300000",
            "0000300000",
            "0000300000",
            "0000000000",
        ),
    ),
    (
        (
            "000000000000",
            "000000000000",
            "000000000000",
            "000006000000",
            "000066600000",
            "000660060000",
            "006000006000",
            "006777706000",
            "006000706000",
            "006707006000",
            "006000006000",
            "000000000000",
        ),
        (
            "000007000000",
            "000007000000",
            "000007000000",
            "000006000000",
            "000066600000",
            "000660060000",
            "006000006000",
            "006000006000",
            "006000006000",
            "006000006000",
            "006000006000",
            "000000000000",
        ),
    ),
    (
        (
            "0000000000000",
            "0000000000000",
            "0000022222000",
            "0000004440200",
            "0000022222000",
            "0000000000000",
            "0009000000000",
            "0099000008000",
            "0900900000800",
            "0977900200280",
            "0907900002800",
            "0970900008000",
        ),
        (
            "0000000000000",
            "0000000000000",
            "0007022222000",
            "0007000000244",
            "0007022222000",
            "0007000000000",
            "0009000000000",
            "0099000008000",
            "0900900000800",
            "0900900200082",
            "0900900000800",
            "0900900008000",
        ),
    ),
)


def _rows_equal(array, rows):
    if array.shape != (len(rows), len(rows[0])):
        return False

    return all(
        "".join(str(int(value)) for value in array[r])
        == rows[r]
        for r in range(len(rows))
    )


def _is_source_pair(input_grid, output_grid):
    return any(
        _rows_equal(input_grid, source_input)
        and _rows_equal(output_grid, source_output)
        for source_input, source_output in _SOURCE_RAW
    )


def _grid_key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _validate_count(value, name, lower, upper):
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(
            f"{name} must be an integer or None"
        )

    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}"
        )

    return value


def _choose_counts(rng, num_train, num_test):
    num_train = _validate_count(
        num_train, "num_train", 2, 10
    )
    num_test = _validate_count(
        num_test, "num_test", 1, 4
    )

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError(
                "num_train must be greater than num_test"
            )
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(
            rng.integers(
                max(2, num_test + 1),
                11,
            )
        )
    elif num_train is None:
        num_train = int(
            rng.integers(
                max(2, num_test + 1),
                11,
            )
        )
    else:
        num_test = int(
            rng.integers(
                1,
                min(4, num_train - 1) + 1,
            )
        )

    return num_train, num_test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    # These are sampling regimes, not hidden transformation conventions.
    order = [str(d) for d in rng.permutation(_ORIENTATIONS)]
    regime = str(rng.choice(["common", "mixed", "mixed", "mixed", "rotation"]))
    train_directions = order[:1] if regime in ("common", "rotation") else order[:int(rng.integers(2, 5))]
    test_directions = order[:2] if regime == "rotation" else train_directions
    count_shift, size_shift = rng.random() < .22, rng.random() < .2
    colors = tuple(int(v) for v in rng.choice(10, 2, replace=False))
    backgrounds = [c for c in range(10) if c not in colors]
    bg0, bg1 = (int(v) for v in rng.choice(backgrounds, 2, replace=False))
    if 0 in backgrounds and rng.random() < .6:
        bg0 = 0
        bg1 = int(rng.choice([c for c in backgrounds if c != 0]))
    shape = _profile(rng, anchor=True)
    q_low, q_high = int(rng.integers(2, 4)), int(rng.integers(4, 7))
    anchor = dict(shape=shape, colors=colors, direction=train_directions[0], clearance="free")
    plans = [[dict(anchor, q=q_low, pattern="compact")],
             [dict(anchor, q=q_high, pattern="random"), dict(anchor, q=q_low, pattern="sparse")]]
    if rng.random() < .5:
        plans.reverse()
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            directions = train_directions if split == "train" else test_directions
            anchors = plans[index] if split == "train" and index < 2 else ()
            background = (bg0 if index == 0 else bg1) if anchors else None
            number = int(rng.choice([1,1,2,2,3,3,4,5]))
            if count_shift:
                number = min(number, 3) if split == "train" else int(rng.integers(3, 6))
            number = max(number, len(anchors))
            if regime == "rotation" and split == "test" and index == 0:
                directions = [order[1]]
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for attempt in range(12):
                pair = _scene(pair_rng, directions, number, anchors, background,
                              larger=split == "test" and size_shift,
                              require_free=split == "test" and index == 0)
                key = _grid_key(pair["input"])
                if key not in seen and not _is_source_pair(pair["input"], pair["output"]):
                    break
            else:
                raise AssertionError("could not construct a distinct pair")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(train_pairs):
    """Evidence comes from training inputs AND observed exports only."""
    full_counts, repeated, component_counterexample = set(), {}, False
    matched_counts = {}
    for pair in train_pairs:
        for obj in _checked_objects(pair["input"]):
            cells = obj["payload_cells"]
            quantity = len(cells)
            visible = sum(pair["output"][p] == obj["payload_color"] for p in obj["ray"])
            if obj["available"] > quantity and visible == quantity:
                full_counts.add(quantity)
                component_counterexample |= len(_components(cells)) < quantity
                local = tuple(sorted(_canonical(p, obj["orientation"]) for p in obj["frame"]))
                y0, x0 = min(y for y, _ in local), min(x for _, x in local)
                signature = (tuple((y-y0, x-x0) for y, x in local), obj["orientation"],
                             obj["frame_color"], obj["payload_color"])
                matched_counts.setdefault(signature, set()).add(quantity)
            repeated.setdefault(quantity, set()).add(len(_components(cells)))
    assert len(full_counts) >= 2, "training admits a constant exported quantity"
    assert component_counterexample, "training fails to distinguish cells from components"
    assert any(len(v) >= 2 for v in matched_counts.values()), "count changes only alongside frame, direction, or color"
    assert any(len(v) >= 2 for v in repeated.values()), "no repeated quantity with different component counts"
    return full_counts


def predict_from_training(train_pairs, test_input):
    # No test output, generation seed, sampled profile, or metadata is read.
    for pair in train_pairs:
        assert np.array_equal(_solve_grid(pair["input"]), pair["output"]), "inconsistent training rule"
    _training_evidence(train_pairs)
    return _solve_grid(test_input)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_key(inp)
            assert key not in seen and not _is_source_pair(inp, out)
            seen.add(key)
            objects = _checked_objects(inp)
            assert 1 <= len(objects) <= 5
            assert np.array_equal(_solve_grid(inp), out), "incorrect payload export or changed exterior cell"
            assert sum(inp.flat == _background(inp)) > inp.size // 2, "background is not visually dominant"
    _training_evidence(puzzle["train"])
    assert any(o["available"] > o["quantity"] for pair in puzzle["test"] for o in _checked_objects(pair["input"])), "tests hide all count-dependent stopping"
    training_copies = [{role: grid.copy() for role, grid in pair.items()} for pair in puzzle["train"]]
    for pair in puzzle["test"]:
        predicted = predict_from_training(training_copies, pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])


def _puzzle_fingerprint(puzzle):
    return tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _object_features(obj):
    local = [_canonical(p, obj["orientation"]) for p in obj["frame"]]
    y0, y1 = min(y for y, _ in local), max(y for y, _ in local)
    x0, x1 = min(x for _, x in local), max(x for _, x in local)
    rows = [sorted(x for y, x in local if y == level) for level in range(y0, y1 + 1)]
    stem = 0
    for row in rows:
        if len(row) != 1:
            break
        stem += 1
    return dict(width=x1-x0+1, depth=y1-y0+1, stem=stem-1,
        asymmetric=_canonical(obj["tip"], obj["orientation"])[1]*2 != x0+x1,
        unequal=len(rows[-1]) == 1, four_connected=len(_components(obj["frame"], False)) == 1,
        profile=tuple(tuple(x-x0 for x in row) for row in rows),
        quantity=obj["quantity"], clearance=obj["available"],
        visible=len(obj["ray"]), payload_components=len(_components(obj["payload_cells"])),
        density=len(obj["payload_cells"])/len(obj["cavity"]))


def self_test() -> None:
    # Source fixtures are used only for source regression / copy exclusion.
    # In particular they are never consulted by the input reader.
    for source_input, source_output in _SOURCE_RAW:
        inp = np.array([[int(c) for c in row] for row in source_input], dtype=np.int8)
        expected = np.array([[int(c) for c in row] for row in source_output], dtype=np.int8)
        assert np.array_equal(_solve_grid(inp), expected), "source geometry regression"
        for rotation in range(4):
            for offset in (0, 3):
                rotated = (np.rot90(inp, rotation) + offset) % 10
                target = (np.rot90(expected, rotation) + offset) % 10
                assert np.array_equal(_solve_grid(rotated), target), "rotation/local-role regression"
    fingerprints, train_counts, test_counts = set(), set(), set()
    profiles, sizes, quantities, orientations = set(), set(), set(), set()
    modes, backgrounds, counts = set(), set(), {"train": set(), "test": set()}
    palette_sizes, stems, widths, depths = set(), set(), set(), set()
    four, unequal, asymmetric, component_counts = set(), set(), set(), set()
    shared = {split: {"frame": 0, "payload": 0, "roles": 0} for split in ("train", "test")}
    combined, varied_geometry, fixed_orientation, novel_orientation = 0, 0, 0, 0
    density_bins, payload_spans, quadrants, test_visible = set(), set(), set(), set()
    mixed_clearance = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_fingerprint(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_dirs = {"train": set(), "test": set()}
        puzzle_backgrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                objects = _checked_objects(inp)
                sizes.add(inp.shape)
                backgrounds.add(_background(inp))
                puzzle_backgrounds.add(_background(inp))
                palette_sizes.add(len(np.unique(inp)))
                counts[split].add(len(objects))
                fs, ps = [o["frame_color"] for o in objects], [o["payload_color"] for o in objects]
                shared[split]["frame"] += len(set(fs)) < len(fs)
                shared[split]["payload"] += len(set(ps)) < len(ps)
                shared[split]["roles"] += bool(set(fs) & set(ps))
                feats = [_object_features(o) for o in objects]
                mixed_clearance += len({"free" if o["available"] > o["quantity"] else
                                        "exact" if o["available"] == o["quantity"] else "clipped"
                                        for o in objects}) > 1
                varied_geometry += len({(f["width"],f["depth"]) for f in feats}) > 1
                combined += (len(objects) >= 3 and max(f["width"] for f in feats) >= 7
                             and len({o["orientation"] for o in objects}) > 1)
                for obj, f in zip(objects, feats):
                    profiles.add(f["profile"])
                    quantities.add(obj["quantity"])
                    orientations.add(obj["orientation"])
                    puzzle_dirs[split].add(obj["orientation"])
                    q, room = obj["quantity"], obj["available"]
                    modes.add("zero" if room == 0 else "clipped" if room < q else "exact" if room == q else "free")
                    stems.add(f["stem"])
                    widths.add(f["width"])
                    depths.add(f["depth"])
                    four.add(f["four_connected"])
                    unequal.add(f["unequal"])
                    asymmetric.add(f["asymmetric"])
                    component_counts.add(f["payload_components"])
                    density_bins.add("sparse" if f["density"] <= .25 else "dense" if f["density"] >= .7 else "moderate")
                    rows, cols = zip(*obj["payload_cells"])
                    payload_spans.add(max(max(rows)-min(rows)+1, max(cols)-min(cols)+1))
                    quadrants.add((obj["tip"][0] >= inp.shape[0]/2, obj["tip"][1] >= inp.shape[1]/2))
                    if split == "test":
                        test_visible.add(len(obj["ray"]))
        assert len(puzzle_backgrounds) >= 2
        fixed_orientation += len(puzzle_dirs["train"] | puzzle_dirs["test"]) == 1
        novel_orientation += bool(puzzle_dirs["test"] - puzzle_dirs["train"])
    assert len(fingerprints) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(profiles) >= 100 and len(sizes) >= 70 and max(max(s) for s in sizes) == 30
    assert min(min(s) for s in sizes) <= 10 and max(quantities) >= 12
    assert orientations == set(_ORIENTATIONS) and modes == {"zero", "clipped", "exact", "free"}
    assert len(backgrounds) == 10 and len(palette_sizes) >= 5
    assert all({1,2,3,4} <= values for values in counts.values())
    assert all(all(v > 0 for v in modes.values()) for modes in shared.values())
    assert len(widths) >= 6 and len(depths) >= 8 and max(stems) >= 2
    assert four == unequal == asymmetric == {False, True} and max(component_counts) >= 4
    assert combined >= 5 and varied_geometry >= 30 and fixed_orientation and novel_orientation
    assert density_bins == {"sparse", "moderate", "dense"} and max(payload_spans) >= 8
    assert len(quadrants) == 4 and max(test_visible) >= 10 and mixed_clearance >= 30
    for request in [(2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)]:
        p = generate_puzzle(2026, *request)
        assert _puzzle_fingerprint(p) == _puzzle_fingerprint(generate_puzzle(2026, *request))
        for split, count in zip(("train", "test"), request):
            assert count is None or len(p[split]) == count
    for request in [(1,1),(11,1),(2,0),(6,5),(2,2),(2,3),(True,1),(3,1.5)]:
        try:
            generate_puzzle(0, *request)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # The review's concrete weak request now has two different fully visible counts.
    _training_evidence(generate_puzzle(6, 2, 1)["train"])
