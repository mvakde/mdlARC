"""Generate framed, one-sided mirror-completion ARC puzzles."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    "a57639f670c45e201914c28635bbf5b1dd17fc061a86d59cc90ae1fe31ec1aad",
    "792ec221f0556e267882a6141b85c6475010c49341db878e01399e1f866ac418",
    "e6bb0267110901b090e80b50a884120a6bcee67105300f4fbf4a10caea8d216a",
    "555757d9280631e8dbcc44bdb8f7de9b84f63c449eadaf70288d73b6765417b9",
}


def _check_count(value, name, low, high):
    if value is not None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer or None")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(rng, num_train, num_test):
    _check_count(num_train, "num_train", 2, 10)
    _check_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _make_motif(rng, orientation, side, cross_size, axis_size, strong=False):
    """The unfinished motif is already symmetric perpendicular to its mirror axis."""
    strict = axis_size // 2
    representatives = (cross_size + 1) // 2
    half = np.zeros((cross_size, axis_size) if orientation == "lr" else (axis_size, cross_size), dtype=bool)

    def put(primary, rep):
        if orientation == "lr":
            half[rep, primary] = True
            half[cross_size - 1 - rep, primary] = True
        else:
            half[primary, rep] = True
            half[primary, cross_size - 1 - rep] = True

    if strong:
        # Two different cross-sections make reflection distinguishable from
        # translating an unchanged motif into the empty half.
        first = int(rng.integers(0, max(1, strict - 1)))
        first = min(first, strict - 2)
        inner_rep = representatives - 1
        outer_rep = inner_rep - 1
        put(first, inner_rep)
        put(first + 1, outer_rep)
        if rng.random() < 0.45:
            put(first, outer_rep)
    else:
        mode = int(rng.choice(5, p=[0.21, 0.20, 0.19, 0.18, 0.22]))
        length = int(rng.integers(1, min(strict, 5) + 1))
        starts = np.arange(strict - length + 1)
        weights = np.array([2.4 if start > 0 else 1.0 for start in starts], dtype=float)
        weights /= weights.sum()
        start = int(rng.choice(starts, p=weights))
        padding_options = np.arange(min(3, representatives))
        padding_weights = np.array([0.24, 0.60, 0.16])[:len(padding_options)]
        padding = int(rng.choice(padding_options, p=padding_weights / padding_weights.sum()))
        reps = list(range(padding, representatives))
        if mode == 0:  # thin line
            rep = int(rng.choice(reps))
            for primary in range(start, start + length):
                put(primary, rep)
        else:
            density = (0.44, 0.78, 0.34, 0.55)[mode - 1]
            for primary in range(start, start + length):
                for rep in reps:
                    if rng.random() < density:
                        put(primary, rep)
            if mode == 2 and length > 1:
                put(start, int(rng.choice(reps)))
                put(start + length - 1, int(rng.choice(reps)))
        strict_cells = half[:, :strict] if orientation == "lr" else half[:strict, :]
        while int(strict_cells.sum()) < 2:
            put(int(rng.integers(start, start + length)), int(rng.choice(reps)))
            if int(strict_cells.sum()) < 2 and length == 1 and len(reps) == 1:
                if start + 1 < strict:
                    put(start + 1, reps[0])
                else:
                    put(start - 1, reps[0])

    if axis_size % 2 and rng.random() < 0.26:
        rep = int(rng.integers(0, representatives))
        put(strict, rep)
    if side in ("right", "lower"):
        half = np.flip(half, axis=1 if orientation == "lr" else 0).copy()
    return half


def _marker_widths(cross_size, style):
    if style == "prong":
        return [1 if cross_size % 2 else 2]
    return [width for width in (3, 4, 5, 6) if width <= cross_size and width % 2 == cross_size % 2]


def _make_spec(rng, orientation, side, object_count, strong=False, style=None):
    cross_high = 10 if object_count == 1 else 9 if object_count == 2 else 8 if object_count == 3 else 7
    axis_high = 12 if object_count == 1 else 11 if object_count == 2 else 9 if object_count == 3 else 8
    cross_size = int(rng.integers(3, cross_high + 1))
    axis_size = int(rng.integers(6 if strong else 4, axis_high + 1))
    style = style or ("hollow" if rng.random() < 0.55 else "prong")
    marker_width = int(rng.choice(_marker_widths(cross_size, style)))
    depth = int(rng.integers(1, 4))
    core_h, core_w = ((cross_size + 2, axis_size + 2) if orientation == "lr" else (axis_size + 2, cross_size + 2))
    footprint_h = core_h + (2 * depth if orientation == "tb" else 0)
    footprint_w = core_w + (2 * depth if orientation == "lr" else 0)
    return {
        "orientation": orientation, "side": side, "style": style,
        "marker_width": marker_width, "depth": depth,
        "core_h": core_h, "core_w": core_w,
        "footprint_h": footprint_h, "footprint_w": footprint_w,
        "motif": _make_motif(rng, orientation, side, cross_size, axis_size, strong),
    }


def _perimeter(r0, r1, c0, c1):
    return ({(r0, c) for c in range(c0, c1 + 1)} |
            {(r1, c) for c in range(c0, c1 + 1)} |
            {(r, c0) for r in range(r0, r1 + 1)} |
            {(r, c1) for r in range(r0, r1 + 1)})


def _frame_cells(orientation, style, marker_width, depth, r0, r1, c0, c1):
    cells = _perimeter(r0, r1, c0, c1)
    if orientation == "lr":
        start = (r0 + r1 - marker_width + 1) // 2
        lo, hi = start, start + marker_width - 1
        for wall, direction in ((c0, -1), (c1, 1)):
            if style == "hollow":
                for row in range(lo + 1, hi):
                    cells.remove((row, wall))
                for col in range(wall + direction, wall + direction * (depth + 1), direction):
                    cells.add((lo, col))
                    cells.add((hi, col))
                for row in range(lo, hi + 1):
                    cells.add((row, wall + direction * depth))
            else:
                for row in range(lo, hi + 1):
                    for col in range(wall + direction, wall + direction * (depth + 1), direction):
                        cells.add((row, col))
    else:
        start = (c0 + c1 - marker_width + 1) // 2
        lo, hi = start, start + marker_width - 1
        for wall, direction in ((r0, -1), (r1, 1)):
            if style == "hollow":
                for col in range(lo + 1, hi):
                    cells.remove((wall, col))
                for row in range(wall + direction, wall + direction * (depth + 1), direction):
                    cells.add((row, lo))
                    cells.add((row, hi))
                for col in range(lo, hi + 1):
                    cells.add((wall + direction * depth, col))
            else:
                for col in range(lo, hi + 1):
                    for row in range(wall + direction, wall + direction * (depth + 1), direction):
                        cells.add((row, col))
    return cells


def _boxes_separated(a, b):
    return a[1] + 2 <= b[0] or b[1] + 2 <= a[0] or a[3] + 2 <= b[2] or b[3] + 2 <= a[2]


def _canvas_size(rng, specs, object_count):
    if object_count == 1:
        spec = specs[0]
        return (min(30, spec["footprint_h"] + int(rng.integers(0, 13))),
                min(30, spec["footprint_w"] + int(rng.integers(0, 13))))
    low = (13, 18, 22)[min(object_count - 2, 2)]
    if rng.random() < 0.32 and object_count <= 3:
        short = int(rng.integers(low, min(24, low + 8)))
        long = int(rng.integers(max(short + 4, 23), 31))
        return (short, long) if rng.random() < 0.5 else (long, short)
    return int(rng.integers(low, 31)), int(rng.integers(low, 31))


def _place_specs(rng, specs, height, width):
    if any(s["footprint_h"] > height or s["footprint_w"] > width for s in specs):
        return None
    for _ in range(12):
        placements = [None] * len(specs)
        boxes = []
        order = list(rng.permutation(len(specs)))
        order.sort(key=lambda i: specs[i]["footprint_h"] * specs[i]["footprint_w"], reverse=True)
        for index in order:
            spec = specs[index]
            candidates = [(r, c) for r in range(height - spec["footprint_h"] + 1)
                          for c in range(width - spec["footprint_w"] + 1)]
            rng.shuffle(candidates)
            for top, left in candidates:
                box = (top, top + spec["footprint_h"] - 1,
                       left, left + spec["footprint_w"] - 1)
                if all(_boxes_separated(box, previous) for previous in boxes):
                    placements[index] = (top, left)
                    boxes.append(box)
                    break
            if placements[index] is None:
                break
        if all(p is not None for p in placements):
            return placements
    return None


def _paint_pair(height, width, specs, placements, background):
    inp = np.full((height, width), background, dtype=np.int8)
    objects = []
    for spec, (top, left) in zip(specs, placements):
        depth = spec["depth"]
        r0 = top + (depth if spec["orientation"] == "tb" else 0)
        c0 = left + (depth if spec["orientation"] == "lr" else 0)
        r1 = r0 + spec["core_h"] - 1
        c1 = c0 + spec["core_w"] - 1
        for row, col in _frame_cells(spec["orientation"], spec["style"], spec["marker_width"], depth, r0, r1, c0, c1):
            inp[row, col] = spec["frame_color"]
        rows, cols = np.nonzero(spec["motif"])
        inp[r0 + 1 + rows, c0 + 1 + cols] = spec["payload_color"]
        objects.append((spec, r0, r1, c0, c1))
    out = inp.copy()
    for spec, r0, r1, c0, c1 in objects:
        rows, cols = np.nonzero(spec["motif"])
        for local_row, local_col in zip(rows, cols):
            row, col = r0 + 1 + int(local_row), c0 + 1 + int(local_col)
            mirrored = (row, c0 + c1 - col) if spec["orientation"] == "lr" else (r0 + r1 - row, col)
            out[mirrored] = spec["payload_color"]
    return inp, out


def _pair_colors(rng, specs, palette, demonstrate=False):
    colors = list(rng.permutation(palette))
    background = int(colors[0])
    n = len(specs)
    if demonstrate:
        frame_pool = [int(v) for v in colors[1:3]]
        payload_pool = [int(v) for v in colors[3:5]]
    else:
        frame_count = int(rng.integers(1, min(n, 3) + 1))
        payload_count = int(rng.integers(1, min(n, 3) + 1))
        frame_pool = [int(v) for v in colors[1:1 + frame_count]]
        payload_pool = [int(v) for v in colors[1 + frame_count:1 + frame_count + payload_count]]
    for index, spec in enumerate(specs):
        spec["frame_color"] = frame_pool[index % len(frame_pool)] if demonstrate else int(rng.choice(frame_pool))
        spec["payload_color"] = payload_pool[index % len(payload_pool)] if demonstrate else int(rng.choice(payload_pool))
    return background


def _generate_pair(rng, plan, palette, strong=False, demonstrate=False):
    count = len(plan)
    for attempt in range(65):
        specs = []
        for index, (orientation, side) in enumerate(plan):
            style = ("hollow" if index == 0 else "prong") if demonstrate and index < 2 else None
            specs.append(_make_spec(rng, orientation, side, count, strong and index == 0, style))
        try:
            for spec in specs:
                r0 = spec["depth"] if spec["orientation"] == "tb" else 0
                c0 = spec["depth"] if spec["orientation"] == "lr" else 0
                contour = _frame_cells(spec["orientation"], spec["style"],
                                       spec["marker_width"], spec["depth"],
                                       r0, r0 + spec["core_h"] - 1,
                                       c0, c0 + spec["core_w"] - 1)
                _parse_frame(contour, (30, 30))
        except AssertionError:
            continue
        height, width = _canvas_size(rng, specs, count)
        if attempt >= 45:
            height = max(height, 28)
            width = max(width, 28)
        placements = _place_specs(rng, specs, height, width)
        if placements is not None:
            background = _pair_colors(rng, specs, palette, demonstrate)
            return _paint_pair(height, width, specs, placements, background)
    raise RuntimeError("could not pack the requested framed objects")


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for near in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if near in remaining:
                    remaining.remove(near)
                    component.add(near)
                    stack.append(near)
        components.append(component)
    return components


def _parse_frame(component, shape):
    min_row = min(r for r, c in component)
    max_row = max(r for r, c in component)
    min_col = min(c for r, c in component)
    max_col = max(c for r, c in component)
    candidates = []
    for orientation in ("lr", "tb"):
        for depth in (1, 2, 3):
            r0 = min_row + (depth if orientation == "tb" else 0)
            r1 = max_row - (depth if orientation == "tb" else 0)
            c0 = min_col + (depth if orientation == "lr" else 0)
            c1 = max_col - (depth if orientation == "lr" else 0)
            cross = (r1 - r0 - 1) if orientation == "lr" else (c1 - c0 - 1)
            axis = (c1 - c0 - 1) if orientation == "lr" else (r1 - r0 - 1)
            if not (3 <= cross <= 10 and 4 <= axis <= 12):
                continue
            for style in ("hollow", "prong"):
                for width in _marker_widths(cross, style):
                    if _frame_cells(orientation, style, width, depth, r0, r1, c0, c1) == component:
                        candidates.append((orientation, style, width, depth, r0, r1, c0, c1))
    assert len(candidates) == 1, "frame contour must identify one axis and marker"
    orientation, style, marker_width, depth, r0, r1, c0, c1 = candidates[0]
    assert 0 <= min_row <= max_row < shape[0] and 0 <= min_col <= max_col < shape[1]
    return {
        "orientation": orientation, "style": style, "marker_width": marker_width,
        "depth": depth, "r0": r0, "r1": r1, "c0": c0, "c1": c1,
        "cross": (r1 - r0 - 1) if orientation == "lr" else (c1 - c0 - 1),
        "axis": (c1 - c0 - 1) if orientation == "lr" else (r1 - r0 - 1),
        "box": (min_row, max_row, min_col, max_col),
    }


def _copy_distinguishing(frame, cells):
    r0, r1, c0, c1 = (frame[k] for k in ("r0", "r1", "c0", "c1"))
    if frame["orientation"] == "lr":
        profiles = {col: frozenset(row for row, cc in cells if cc == col)
                    for col in range(c0 + 1, c1)}
        occupied = [col for col, profile in profiles.items() if profile and (col < (c0 + c1) / 2 if frame["side"] == "left" else col > (c0 + c1) / 2)]
    else:
        profiles = {row: frozenset(col for rr, col in cells if rr == row)
                    for row in range(r0 + 1, r1)}
        occupied = [row for row, profile in profiles.items() if profile and (row < (r0 + r1) / 2 if frame["side"] == "upper" else row > (r0 + r1) / 2)]
    return len(occupied) >= 2 and tuple(profiles[p] for p in occupied) != tuple(profiles[p] for p in reversed(occupied))


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    changed = inp != out
    assert bool(changed.any()), "a pair must require completion"
    backgrounds = {int(x) for x in inp[changed]}
    assert len(backgrounds) == 1
    background = backgrounds.pop()
    payload_colors = {int(x) for x in out[changed]}
    assert background not in payload_colors
    frame_colors = {int(x) for x in np.unique(inp)} - payload_colors - {background}
    assert frame_colors and not (frame_colors & payload_colors)
    assert set(int(x) for x in np.unique(out)) == set(int(x) for x in np.unique(inp))
    frames = []
    for color in frame_colors:
        for component in _components(inp == color):
            frame = _parse_frame(component, inp.shape)
            frame["frame_color"] = color
            frames.append(frame)
    assert frames
    expected = inp.copy()
    owned = set()
    for frame in frames:
        assert all(_boxes_separated(frame["box"], other["box"]) for other in frames if other is not frame)
        r0, r1, c0, c1 = (frame[k] for k in ("r0", "r1", "c0", "c1"))
        boundary = _frame_cells(frame["orientation"], frame["style"], frame["marker_width"], frame["depth"], r0, r1, c0, c1)
        owned.update(boundary)
        cells = set()
        colors = set()
        for row in range(r0 + 1, r1):
            for col in range(c0 + 1, c1):
                assert (row, col) not in owned
                owned.add((row, col))
                value = int(inp[row, col])
                assert value == background or value in payload_colors
                if value != background:
                    cells.add((row, col))
                    colors.add(value)
        assert len(colors) == 1, "each object has one local payload color"
        payload = colors.pop()
        frame["payload_color"] = payload
        if frame["orientation"] == "lr":
            left = {(r, c) for r, c in cells if 2 * c < c0 + c1}
            right = {(r, c) for r, c in cells if 2 * c > c0 + c1}
            assert bool(left) != bool(right)
            frame["side"] = "left" if left else "right"
            active = left or right
            for row, col in cells:
                assert (r0 + r1 - row, col) in cells
                reflected = (row, c0 + c1 - col)
                assert r0 < reflected[0] < r1 and c0 < reflected[1] < c1
                expected[reflected] = payload
        else:
            upper = {(r, c) for r, c in cells if 2 * r < r0 + r1}
            lower = {(r, c) for r, c in cells if 2 * r > r0 + r1}
            assert bool(upper) != bool(lower)
            frame["side"] = "upper" if upper else "lower"
            active = upper or lower
            for row, col in cells:
                assert (row, c0 + c1 - col) in cells
                reflected = (r0 + r1 - row, col)
                assert r0 < reflected[0] < r1 and c0 < reflected[1] < c1
                expected[reflected] = payload
        assert len(active) >= 2
        frame["copy_distinguishing"] = _copy_distinguishing(frame, cells)
        frame["wall_clear"] = all((col > c0 + 1 if frame["side"] == "left" else col < c1 - 1) for _, col in active) if frame["orientation"] == "lr" else all((row > r0 + 1 if frame["side"] == "upper" else row < r1 - 1) for row, _ in active)
        frame["cross_clear"] = all(r0 + 1 < row < r1 - 1 for row, _ in cells) if frame["orientation"] == "lr" else all(c0 + 1 < col < c1 - 1 for _, col in cells)
        frame["motif_signature"] = (frame["cross"], frame["axis"], tuple(sorted((r - r0, c - c0) for r, c in cells)))
    assert all(int(inp[r, c]) == background for r in range(inp.shape[0]) for c in range(inp.shape[1]) if (r, c) not in owned)
    assert np.array_equal(out, expected), "output must add exactly the local frame reflections"
    return {"background": background, "frames": frames}


def _grid_key(grid):
    return (tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes())


def _source_digest(inp, out):
    return hashlib.sha256(bytes(inp.shape) + inp.astype(np.int8).tobytes() + out.astype(np.int8).tobytes()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    keys = set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            inp, out = pair["input"], pair["output"]
            key = _grid_key(inp)
            assert key not in keys, "duplicate input"
            keys.add(key)
            assert _source_digest(inp, out) not in _SOURCE_DIGESTS
            analyses[section].append(_analyze_pair(inp, out))
    train_frames = [f for a in analyses["train"] for f in a["frames"]]
    test_frames = [f for a in analyses["test"] for f in a["frames"]]
    assert any(f["copy_distinguishing"] for f in train_frames), "training must establish reflection instead of motif translation"
    assert {f["orientation"] for f in test_frames} <= {f["orientation"] for f in train_frames}
    assert {f["style"] for f in test_frames} <= {f["style"] for f in train_frames}
    assert any(len({f["frame_color"] for f in a["frames"]}) >= 2 and
               len({f["payload_color"] for f in a["frames"]}) >= 2
               for a in analyses["train"]), "training must show local color roles"


def _sample_plans(rng, n_train, n_test):
    mode = str(rng.choice(["mixed", "lr_only", "tb_only"], p=[0.70, 0.15, 0.15]))
    source_like = bool(rng.random() < 0.48)
    plans = {"train": [], "test": []}
    for section, count in (("train", n_train), ("test", n_test)):
        for _ in range(count):
            object_count = int(rng.choice([1, 2, 3, 4], p=([0.33, 0.35, 0.24, 0.08] if section == "train" else [0.20, 0.30, 0.29, 0.21])))
            plan = []
            for _ in range(object_count):
                orientation = mode[0:2] if mode != "mixed" else ("lr" if rng.random() < 0.5 else "tb")
                if orientation == "lr":
                    side = "left" if section == "train" and source_like else ("right" if rng.random() < (0.46 if section == "train" else 0.55) else "left")
                else:
                    side = "upper" if section == "train" and source_like else ("lower" if rng.random() < (0.46 if section == "train" else 0.55) else "upper")
                plan.append((orientation, side))
            plans[section].append(plan)
    # A multicolor, mixed-marker demonstration teaches that color is local.
    demo_index = int(rng.integers(n_train))
    while len(plans["train"][demo_index]) < 2:
        orientation = mode[0:2] if mode != "mixed" else ("lr" if rng.random() < 0.5 else "tb")
        side = ("left" if orientation == "lr" else "upper") if source_like else (("left" if rng.random() < 0.5 else "right") if orientation == "lr" else ("upper" if rng.random() < 0.5 else "lower"))
        plans["train"][demo_index].append((orientation, side))
    if mode == "mixed":
        present = {o for plan in plans["train"] for o, _ in plan}
        if len(present) == 1:
            orientation = "tb" if "lr" in present else "lr"
            side = ("upper" if orientation == "tb" else "left") if source_like else (("upper" if rng.random() < 0.5 else "lower") if orientation == "tb" else ("left" if rng.random() < 0.5 else "right"))
            plans["train"][demo_index][1] = (orientation, side)
    return plans, demo_index


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    n_train, n_test = _resolve_counts(rng, num_train, num_test)
    palette = tuple(int(x) for x in rng.permutation(10))
    plans, demo_index = _sample_plans(rng, n_train, n_test)
    strong_index = int(rng.integers(n_train))
    puzzle = {"train": [], "test": []}
    used = set()
    for section in ("train", "test"):
        for index, plan in enumerate(plans[section]):
            for _ in range(8):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                inp, out = _generate_pair(pair_rng, plan, palette,
                                          strong=(section == "train" and index == strong_index),
                                          demonstrate=(section == "train" and index == demo_index))
                if _grid_key(inp) not in used and _source_digest(inp, out) not in _SOURCE_DIGESTS:
                    used.add(_grid_key(inp))
                    puzzle[section].append({"input": inp, "output": out})
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((section, tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[section])) for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_signature(p) for p in puzzles}) == 42
    shapes, geometries, motifs, layouts = set(), set(), set(), set()
    counts, directions, styles, colors = set(), set(), set(), set()
    edge_frames = compact_motifs = diverse_background_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        analyses = [_analyze_pair(pair["input"], pair["output"]) for section in ("train", "test") for pair in puzzle[section]]
        diverse_background_puzzles += len({a["background"] for a in analyses}) > 1
        layout = []
        for pair, analysis in zip([p for section in ("train", "test") for p in puzzle[section]], analyses):
            shape = tuple(pair["input"].shape)
            shapes.add(shape)
            counts.add(len(analysis["frames"]))
            local = []
            for frame in analysis["frames"]:
                geometries.add((frame["orientation"], frame["style"], frame["marker_width"], frame["depth"], frame["cross"], frame["axis"]))
                motifs.add(frame["motif_signature"])
                directions.add(frame["side"])
                styles.add(frame["style"])
                colors.add((analysis["background"], frame["frame_color"], frame["payload_color"]))
                edge_frames += (frame["box"][0] == 0 or frame["box"][1] == shape[0] - 1 or frame["box"][2] == 0 or frame["box"][3] == shape[1] - 1)
                compact_motifs += frame["wall_clear"] and frame["cross_clear"]
                local.append((frame["box"], frame["orientation"], frame["side"], frame["style"]))
            layout.append((shape, tuple(sorted(local))))
        layouts.add(tuple(layout))
    assert counts == {1, 2, 3, 4} and directions == {"left", "right", "upper", "lower"}
    assert styles == {"hollow", "prong"}
    assert len(shapes) >= 50 and len(geometries) >= 45 and len(motifs) >= 100
    assert len(colors) >= 40 and len(layouts) == 42
    assert edge_frames > 0 and compact_motifs > 0 and diverse_background_puzzles >= 30
    for seed, train, test in ((6001, 2, 1), (6002, 10, 1), (6003, 10, 4)):
        puzzle = generate_puzzle(seed, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
        assert _signature(puzzle) == _signature(generate_puzzle(seed, train, test))
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(7001, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(8001, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair-count request was accepted")
    assert _signature(generate_puzzle(77)) == _signature(generate_puzzle(77))
    assert _signature(generate_puzzle(77)) != _signature(generate_puzzle(78))
