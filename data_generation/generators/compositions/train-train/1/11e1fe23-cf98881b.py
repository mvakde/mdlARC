"""Nested incomplete frames composited into one fixed 3x3 centre miniature.

Parent A (11e1fe23): three corners of a large odd square are reproduced as a
fixed-size 3x3 miniature at the square's centre (missing corner left empty,
gray centre marker), regardless of the square's size.

Parent B (cf98881b): aligned transparent layers are composited cell by cell;
at every coordinate the first non-background value in priority order wins and
a coordinate that is background in every layer stays background.

Composition: the input holds one stack of concentric, strictly nested,
incomplete rectangular frames (odd height and width, shared centre).  Each
frame is drawn only by sparse markers at some of its eight landmarks (four
corners and, in some puzzles, four edge midpoints) and always lacks at least
one corner.  Every frame is projected at fixed 3x3 scale onto the shared
centre (corner -> miniature corner, edge midpoint -> miniature edge cell), so
all frames become aligned layers of one 3x3 canvas.  The innermost frame is
the top layer: each miniature cell shows the landmark colour of the innermost
frame that has that landmark; a frame's missing landmark is a transparent
hole; a landmark missing from every frame leaves the cell background.  The
centre cell is always gray (5).  Everything else is unchanged.
"""
from __future__ import annotations

import numpy as np

CENTER_COLOR = 5
CORNERS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
AXES = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRECTIONS = CORNERS + AXES
MAX_SIDE = 30
MAX_HALF = 14
SHIFTS = ("deeper", "larger", "smaller")
ALTERNATIVES = ("outer_on_top", "outer_only", "inner_only")
# Reading-order painter shortcuts: project every marker in raster order and
# let the first (or last) write win.  Each agrees with the true rule in half
# of the directions, so training must refute both.
TRAIN_ALTERNATIVES = ALTERNATIVES + ("raster_first", "raster_last")

# Parent A's original inputs.  With bg 0 and a gray centre they are legal
# one-frame members of this family, so exact reuse is rejected explicitly.
_A_SOURCE_INPUTS = (
    ("00000000000", "00000000000", "00000000000", "00000000200",
     "00000000000", "00000000000", "00000000000", "00000000000",
     "00000000000", "00800000600", "00000000000", "00000000000"),
    ("000000000000", "000200000100", "000000000000", "000000000000",
     "000000000000", "000000000000", "000000000000", "000300000000",
     "000000000000", "000000000000"),
    ("00000000000000", "00000000000000", "00030000000800", "00000000000000",
     "00000000000000", "00000000000000", "00000000000000", "00000000000000",
     "00000000000000", "00000000000000", "00090000000000", "00000000000000"),
)


def _key(grid) -> tuple:
    array = np.asarray(grid, dtype=np.int64)
    return (array.shape, array.tobytes())


_A_SOURCE_KEYS = frozenset(
    _key(np.array([[int(ch) for ch in row] for row in rows], dtype=np.int64))
    for rows in _A_SOURCE_INPUTS
)


def _check(condition, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


# --------------------------------------------------------------------------
# Solver (input only)
# --------------------------------------------------------------------------

def solve(input_grid: np.ndarray) -> np.ndarray:
    """Composite every frame's landmarks onto the centre 3x3, innermost on top.

    Frames are strictly nested, so along each of the eight landmark directions
    the innermost frame owning that landmark is the marker nearest the centre
    in that direction (corners in a quadrant, midpoints on an axis).
    """
    grid = np.array(input_grid, dtype=np.int64)
    if grid.ndim != 2:
        raise ValueError("input must be a 2D grid")
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    cells = np.argwhere(grid != background)
    output = grid.copy()
    if len(cells) == 0:
        return output
    top, left = (int(v) for v in cells.min(axis=0))
    bottom, right = (int(v) for v in cells.max(axis=0))
    center_row, center_col = (top + bottom) // 2, (left + right) // 2
    for direction in DIRECTIONS:
        best = None
        for row, col in cells:
            offset_row, offset_col = int(row) - center_row, int(col) - center_col
            if (_sign(offset_row), _sign(offset_col)) != direction:
                continue
            distance = abs(offset_row) + abs(offset_col)
            if best is None or distance < best[0]:
                best = (distance, int(grid[row, col]))
        if best is not None:
            output[center_row + direction[0], center_col + direction[1]] = best[1]
    output[center_row, center_col] = CENTER_COLOR
    return output


# --------------------------------------------------------------------------
# Input analysis shared by generation checks and validation
# --------------------------------------------------------------------------

def _analyze_input(grid) -> dict:
    """Reconstruct background, centre and the nested frames from an input."""
    grid = np.asarray(grid)
    values, counts = np.unique(grid, return_counts=True)
    _check(len(values) >= 2, "input needs at least one marker")
    order = np.argsort(-counts, kind="stable")
    _check(counts[order[0]] > counts[order[1]], "background must be the unique majority colour")
    background = int(values[order[0]])
    _check(CENTER_COLOR not in {int(v) for v in values}, "gray is reserved for the output centre")
    cells = np.argwhere(grid != background)
    top, left = (int(v) for v in cells.min(axis=0))
    bottom, right = (int(v) for v in cells.max(axis=0))
    _check((bottom - top) % 2 == 0 and (right - left) % 2 == 0,
           "marker bounding box must have odd height and width")
    center = ((top + bottom) // 2, (left + right) // 2)
    corner_frames, vertical, horizontal = {}, [], []
    for row, col in cells:
        dr, dc = int(row) - center[0], int(col) - center[1]
        color = int(grid[row, col])
        _check(max(abs(dr), abs(dc)) >= 2, "marker inside the miniature footprint")
        if dr and dc:
            _check(abs(dr) >= 2 and abs(dc) >= 2, "frame half-sizes must be at least 2")
            corner_frames.setdefault((abs(dr), abs(dc)), {})[(_sign(dr), _sign(dc))] = color
        elif dc == 0:
            vertical.append((abs(dr), (_sign(dr), 0), color))
        else:
            horizontal.append((abs(dc), (0, _sign(dc)), color))
    sizes = sorted(corner_frames)
    _check(sizes, "every frame needs at least one corner")
    for (a0, b0), (a1, b1) in zip(sizes, sizes[1:]):
        _check(a0 < a1 and b0 < b1, "frames must be strictly nested")
    frames = [(a, b, dict(corner_frames[(a, b)])) for a, b in sizes]
    by_height = {frame[0]: frame for frame in frames}
    by_width = {frame[1]: frame for frame in frames}
    for half, direction, color in vertical:
        _check(half in by_height, "vertical midpoint does not belong to a frame")
        by_height[half][2][direction] = color
    for half, direction, color in horizontal:
        _check(half in by_width, "horizontal midpoint does not belong to a frame")
        by_width[half][2][direction] = color
    for _, _, points in frames:
        _check(sum(d in points for d in CORNERS) <= 3, "each frame must lack a corner")
    outer = frames[-1]
    _check({d[0] for d in outer[2]} >= {-1, 1} and {d[1] for d in outer[2]} >= {-1, 1},
           "outer frame must reach all four sides")
    _check((outer[0], outer[1]) == ((bottom - top) // 2, (right - left) // 2),
           "outer frame must define the bounding box")
    return {"background": background, "center": center, "frames": frames,
            "shape": tuple(int(v) for v in grid.shape)}


def _composite(frames, order) -> dict:
    """order: list of frame indices, or a callable direction -> list."""
    result = {}
    for direction in DIRECTIONS:
        for index in (order(direction) if callable(order) else order):
            if direction in frames[index][2]:
                result[direction] = (frames[index][2][direction], index)
                break
    return result


def _paint(grid, center, composite) -> np.ndarray:
    output = np.array(grid, dtype=np.int64, copy=True)
    for (dr, dc), (color, _) in composite.items():
        output[center[0] + dr, center[1] + dc] = color
    output[center] = CENTER_COLOR
    return output


def _predictions(info) -> dict:
    frames = info["frames"]
    count = len(frames)
    inward, outward = list(range(count)), list(range(count - 1, -1, -1))

    def raster_first(direction):
        # Earlier in reading order = outer frame for N, NW, NE and W.
        early_is_outer = direction[0] == -1 or direction == (0, -1)
        return outward if early_is_outer else inward

    def raster_last(direction):
        return inward if raster_first(direction) is outward else outward

    orders = {"truth": inward, "outer_on_top": outward,
              "outer_only": [count - 1], "inner_only": [0],
              "raster_first": raster_first, "raster_last": raster_last}
    return {name: _composite(frames, order) for name, order in orders.items()}


def _colors(composite) -> dict:
    return {d: c for d, (c, _) in composite.items()}


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _resolve_counts(num_train, num_test) -> tuple[int, int]:
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _pick(rng, options):
    options = list(options)
    if not options:
        return None
    return options[int(rng.integers(len(options)))]


def _pick_background(rng) -> int:
    if rng.random() < 0.35:
        return 0
    return int(_pick(rng, [c for c in range(10) if c not in (0, CENTER_COLOR)]))


def _extra(rng, maximum: int) -> int:
    """Margin per axis: tight, medium or anything that fits."""
    if maximum <= 0:
        return 0
    band = int(rng.integers(0, 3))
    upper = (min(1, maximum), min(7, maximum), maximum)[band]
    return int(rng.integers(0, upper + 1))


def _half_sizes(rng, count, low, high, rect):
    """Strictly nested (half_height, half_width) pairs, innermost first."""
    low, high = max(low, count + 1, 2), min(high, MAX_HALF)
    if rect:
        low = max(low, count + 2)
    if low > high:
        return None
    scale = int(rng.integers(low, high + 1))
    if rect:
        other = int(rng.integers(count + 1, scale))
        heights_outer, widths_outer = (scale, other) if rng.random() < 0.5 else (other, scale)
    else:
        heights_outer = widths_outer = scale

    def nested(outer):
        inner = rng.choice(np.arange(2, outer), size=count - 1, replace=False)
        return sorted(int(v) for v in inner) + [outer]

    heights = nested(heights_outer)
    widths = nested(widths_outer) if rect else list(heights)
    return list(zip(heights, widths))


def _point_sets(rng, count, midpoints, need):
    """Choose each frame's landmark subset; innermost frame is index 0."""
    allowed = DIRECTIONS if midpoints else CORNERS
    outer = count - 1
    present = [set() for _ in range(count)]
    banned = [set() for _ in range(count)]
    required = [set() for _ in range(count)]
    # The outer frame is a parent-A incomplete square: three corners, or
    # occasionally two opposite corners; either reaches all four sides.
    if rng.random() < 0.8:
        missing = {CORNERS[int(rng.integers(4))]}
    else:
        missing = set(_pick(rng, (((-1, -1), (1, 1)), ((-1, 1), (1, -1)))))
    present[outer] |= set(CORNERS) - missing
    required[outer] |= set(CORNERS) - missing
    banned[outer] |= missing
    if need["empty"]:
        empty = _pick(rng, sorted(missing))
        for index in range(count):
            banned[index].add(empty)

    def place(direction, index):
        present[index].add(direction)
        required[index].add(direction)

    if need["deep"]:
        options = [(d, k) for d in allowed for k in range(2, count)
                   if d not in banned[k] and d not in present[0] and d not in present[1]]
        choice = _pick(rng, options)
        if choice is None:
            return None
        place(choice[0], choice[1])
        banned[0].add(choice[0])
        banned[1].add(choice[0])
    if need["show"]:
        options = [(d, k) for d in allowed for k in range(1, count)
                   if d not in banned[k] and d not in present[0]]
        choice = _pick(rng, options)
        if choice is None:
            return None
        place(choice[0], choice[1])
        banned[0].add(choice[0])
    conflicts = []
    if need["conflict"]:
        options = [(d, k) for d in allowed for k in range(1, count)
                   if d not in banned[k] and d not in banned[0]]
        choice = _pick(rng, options)
        if choice is None:
            return None
        place(choice[0], 0)
        place(choice[0], choice[1])
        conflicts.append(choice)
    union = set().union(*present)
    for direction in need["fill"]:
        if direction in union:
            continue
        options = [k for k in range(count) if direction not in banned[k]]
        if not options:
            return None
        place(direction, _pick(rng, options))
        union.add(direction)
    for index in range(count):
        probability = float(rng.uniform(0.15, 0.55))
        for direction in allowed:
            if index == outer and direction in CORNERS:
                continue
            if direction not in banned[index] and rng.random() < probability:
                present[index].add(direction)
    for index in range(count - 1):
        corners = [d for d in CORNERS if d in present[index]]
        if not corners:
            option = _pick(rng, [d for d in CORNERS if d not in banned[index]])
            if option is None:
                return None
            present[index].add(option)
        elif len(corners) == 4:
            removable = [d for d in corners if d not in required[index]]
            if not removable:
                return None
            present[index].discard(_pick(rng, removable))
    if not set(need["fill"]).issubset(set().union(*present)):
        return None
    return present, conflicts


def _colorize(rng, present, scheme, palette, conflicts):
    count = len(present)
    if scheme == "layered":
        layer_colors = [int(c) for c in rng.choice(palette, size=count, replace=False)]
        return [{d: layer_colors[k] for d in DIRECTIONS if d in present[k]}
                for k in range(count)]
    size = int(rng.integers(2, min(5, len(palette)) + 1))
    subset = [int(c) for c in rng.choice(palette, size=size, replace=False)]
    frames = [{d: int(_pick(rng, subset)) for d in DIRECTIONS if d in present[k]}
              for k in range(count)]
    for direction, deeper in conflicts:
        if frames[deeper][direction] == frames[0][direction]:
            frames[deeper][direction] = int(_pick(
                rng, [c for c in subset if c != frames[0][direction]]))
    return frames


def _pair_features(info) -> dict:
    preds = _predictions(info)
    truth = preds["truth"]
    frames = info["frames"]
    filled = set(truth)
    return {
        "frames": len(frames),
        "filled": filled,
        "empty_corner": any(d not in filled for d in CORNERS),
        "depth": max(index for _, index in truth.values()) + 1,
        "refuted": {name for name in TRAIN_ALTERNATIVES
                    if _colors(preds[name]) != _colors(truth)},
        "scale": max(frames[-1][0], frames[-1][1]),
        "outer": (frames[-1][0], frames[-1][1]),
        "rect": any(a != b for a, b, _ in frames),
        "midpoints": any(d in AXES for _, _, points in frames for d in points),
        "layered": len(frames) > 1 and all(len(set(p.values())) == 1 for _, _, p in frames)
        and len({next(iter(p.values())) for _, _, p in frames}) == len(frames),
    }


def _make_pair(rng, spec, used):
    palette = [c for c in range(10) if c not in (spec["bg"], CENTER_COLOR)]
    for _ in range(120):
        sizes = _half_sizes(rng, spec["count"], spec["low"], spec["high"], spec["rect"])
        if sizes is None:
            return None
        chosen = _point_sets(rng, spec["count"], spec["midpoints"], spec["need"])
        if chosen is None:
            continue
        present, conflicts = chosen
        colored = _colorize(rng, present, spec["scheme"], palette, conflicts)
        half_height, half_width = sizes[-1]
        extra_h = _extra(rng, MAX_SIDE - (2 * half_height + 1))
        extra_w = _extra(rng, MAX_SIDE - (2 * half_width + 1))
        height, width = 2 * half_height + 1 + extra_h, 2 * half_width + 1 + extra_w
        center = (int(rng.integers(0, extra_h + 1)) + half_height,
                  int(rng.integers(0, extra_w + 1)) + half_width)
        grid = np.full((height, width), spec["bg"], dtype=np.int64)
        for (a, b), points in zip(sizes, colored):
            for (dr, dc), color in points.items():
                grid[center[0] + dr * a, center[1] + dc * b] = color
        key = _key(grid)
        if key in used or key in _A_SOURCE_KEYS:
            continue
        try:
            info = _analyze_input(grid)
        except AssertionError:
            continue
        expected = [(a, b, points) for (a, b), points in zip(sizes, colored)]
        if info["frames"] != expected or info["center"] != center:
            continue
        features = _pair_features(info)
        need = spec["need"]
        if need["show"] and "inner_only" not in features["refuted"]:
            continue
        if need["conflict"] and "outer_on_top" not in features["refuted"]:
            continue
        if need["deep"] and features["depth"] < 3:
            continue
        if need["empty"] and not features["empty_corner"]:
            continue
        if spec["strict"] and not set(ALTERNATIVES) <= features["refuted"]:
            continue
        if spec.get("max_depth") and features["depth"] > spec["max_depth"]:
            continue
        output = _paint(grid, center, _predictions(info)["truth"])
        used.add(key)
        return {"input": grid, "output": output}, features
    return None


def _attempt(rng, num_train, num_test):
    shift = SHIFTS[int(rng.choice(3, p=(0.42, 0.34, 0.24)))]
    midpoints = bool(rng.random() < 0.5)
    shape_mode = ("square", "rect", "mixed")[int(rng.choice(3, p=(0.4, 0.25, 0.35)))]
    color_mode = ("layered", "mixed", "both")[int(rng.choice(3, p=(0.3, 0.3, 0.4)))]
    fixed_bg = _pick_background(rng) if rng.random() < 0.4 else None
    allowed = set(DIRECTIONS if midpoints else CORNERS)

    def background():
        return fixed_bg if fixed_bg is not None else _pick_background(rng)

    def scheme():
        if color_mode != "both":
            return color_mode
        return ("layered", "mixed")[int(rng.integers(2))]

    def rect_flag():
        if shape_mode == "mixed":
            return bool(rng.integers(2))
        return shape_mode == "rect"

    if shift == "deeper":
        train_counts = [2] * num_train
    else:
        train_counts = [int(rng.choice((2, 3))) for _ in range(num_train)]
    if num_train >= 3 and rng.random() < 0.5:
        train_counts[int(rng.integers(num_train))] = 1
    rich = [i for i, c in enumerate(train_counts) if c >= 2]
    train_rect = [rect_flag() for _ in range(num_train)]
    if shape_mode == "mixed":
        first, second = (int(v) for v in rng.choice(num_train, size=2, replace=False))
        train_rect[first], train_rect[second] = True, False
    cover = _pick(rng, rich)
    others = [i for i in range(num_train) if i != cover]
    guaranteed_empty = _pick(rng, others)
    guaranteed_show = _pick(rng, rich)
    guaranteed_conflict = _pick(rng, rich)
    ranges = {"deeper": (3, 10), "larger": (2, 7), "smaller": (7, 14)}[shift]

    used, filled = set(), set()
    train, train_features = [None] * num_train, [None] * num_train
    for index in [others[i] for i in rng.permutation(len(others))] + [cover]:
        count = train_counts[index]
        need = {
            "show": count >= 2 and (index == guaranteed_show or rng.random() < 0.5),
            "conflict": count >= 2 and (index == guaranteed_conflict or rng.random() < 0.6),
            "deep": False,
            "empty": index != cover and (index == guaranteed_empty or rng.random() < 0.3),
            "fill": sorted(allowed - filled) if index == cover else [],
        }
        spec = {"count": count, "low": ranges[0], "high": ranges[1],
                "rect": train_rect[index], "midpoints": midpoints, "scheme": scheme(),
                "bg": background(), "need": need, "strict": False,
                "max_depth": 2 if shift == "deeper" else None}
        made = _make_pair(rng, spec, used)
        if made is None:
            return None
        train[index], train_features[index] = made
        filled |= made[1]["filled"]
    if filled != allowed:
        return None

    scales = [f["scale"] for f in train_features]
    test = []
    for _ in range(num_test):
        if shift == "deeper":
            count = int(rng.choice((3, 4)))
            low, high = count + 1, MAX_HALF
        else:
            count = int(_pick(rng, [c for c in (2, 3) if c <= max(train_counts)]))
            if shift == "larger":
                low, high = max(scales) + 1, MAX_HALF
            else:
                low, high = count + 1, min(scales) - 1
        need = {"show": True, "conflict": True, "deep": shift == "deeper",
                "empty": bool(rng.random() < 0.3), "fill": []}
        rect = rect_flag() if any(train_rect) else False
        spec = {"count": count, "low": low, "high": high, "rect": rect,
                "midpoints": midpoints, "scheme": scheme(), "bg": background(),
                "need": need, "strict": True, "max_depth": None}
        made = _make_pair(rng, spec, used)
        if made is None:
            return None
        test.append(made[0])
    order = [int(i) for i in rng.permutation(num_train)]
    return {"train": [train[i] for i in order], "test": test}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _resolve_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    for _ in range(80):
        puzzle = _attempt(rng, num_train, num_test)
        if puzzle is None:
            continue
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError("could not construct a valid puzzle")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "keys must be train/test")
    train, test = puzzle["train"], puzzle["test"]
    _check(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _check(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
           "illegal pair counts")
    seen = set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair must have input/output")
            grids = []
            for role in ("input", "output"):
                array = np.asarray(pair[role])
                _check(array.ndim == 2 and all(1 <= n <= MAX_SIDE for n in array.shape),
                       "grid dimensions must be 1..30")
                _check(np.issubdtype(array.dtype, np.integer), "grid cells must be integers")
                _check(np.all((array >= 0) & (array <= 9)), "colours must be 0..9")
                grids.append(array.astype(np.int64))
            before, after = grids
            _check(before.shape == after.shape, "output keeps the input shape")
            key = _key(before)
            _check(key not in seen, "duplicate input")
            _check(key not in _A_SOURCE_KEYS, "parent A source input reused")
            seen.add(key)
            info = _analyze_input(before)
            expected = _paint(before, info["center"], _predictions(info)["truth"])
            _check(np.array_equal(after, expected), "output is not the innermost-first composite")
            _check(np.array_equal(solve(before), after), "solve disagrees with output")
            changed = np.argwhere(before != after)
            center = info["center"]
            _check(all(abs(int(r) - center[0]) <= 1 and abs(int(c) - center[1]) <= 1
                       for r, c in changed), "changes must stay in the centre 3x3")
            _check(int(after[center]) == CENTER_COLOR, "centre must be gray")
            features[split].append(_pair_features(info))
    train_f, test_f = features["train"], features["test"]
    midpoints = any(f["midpoints"] for f in train_f + test_f)
    allowed = set(DIRECTIONS if midpoints else CORNERS)
    train_filled = set().union(*(f["filled"] for f in train_f))
    _check(train_filled == allowed, "training must fill every miniature cell of the mode")
    _check(set().union(*(f["filled"] for f in test_f)) <= train_filled,
           "test fills a cell never demonstrated")
    if midpoints:
        _check(any(f["midpoints"] for f in train_f), "midpoints must be demonstrated")
    for name in TRAIN_ALTERNATIVES:
        _check(any(name in f["refuted"] for f in train_f), f"training never refutes {name}")
    _check(any(f["empty_corner"] for f in train_f), "training never leaves a corner empty")
    _check(len({f["outer"] for f in train_f}) >= 2, "training outer frames must vary in size")
    for f in test_f:
        _check(f["frames"] >= 2 and set(ALTERNATIVES) <= f["refuted"],
               "every test must defeat each single-parent shortcut")
    if any(f["rect"] for f in test_f):
        _check(any(f["rect"] for f in train_f), "rectangular tests need rectangular training")
    deeper = max(f["depth"] for f in test_f) > max(f["depth"] for f in train_f)
    larger = min(f["scale"] for f in test_f) > max(f["scale"] for f in train_f)
    smaller = max(f["scale"] for f in test_f) < min(f["scale"] for f in train_f)
    _check(deeper or larger or smaller, "no train/test shift")


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def _shift_kind(features) -> str:
    train_f, test_f = features["train"], features["test"]
    if max(f["depth"] for f in test_f) > max(f["depth"] for f in train_f):
        return "deeper"
    if min(f["scale"] for f in test_f) > max(f["scale"] for f in train_f):
        return "larger"
    return "smaller"


def self_test() -> None:
    for counts in ((2, 1), (10, 1), (10, 4), (5, 4), (6, 3)):
        puzzle = generate_puzzle(11 + counts[0] * 10 + counts[1], *counts)
        _check((len(puzzle["train"]), len(puzzle["test"])) == counts, "explicit counts ignored")
    for bad in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(3, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {bad}")
    signatures = set()
    shifts, frame_counts, backgrounds, shapes = set(), set(), set(), set()
    modes = {"midpoints": 0, "corners": 0, "rect": 0, "square": 0,
             "layered": 0, "mixed": 0, "single_frame_train": 0, "varied_bg": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        signature = tuple(_key(p[r]) for s in ("train", "test") for p in puzzle[s]
                          for r in ("input", "output"))
        _check(signature == tuple(_key(p[r]) for s in ("train", "test") for p in again[s]
                                  for r in ("input", "output")), "nondeterministic seed")
        signatures.add(signature)
        features = {s: [_pair_features(_analyze_input(p["input"])) for p in puzzle[s]]
                    for s in ("train", "test")}
        shifts.add(_shift_kind(features))
        all_f = features["train"] + features["test"]
        modes["midpoints" if any(f["midpoints"] for f in all_f) else "corners"] += 1
        modes["single_frame_train"] += any(f["frames"] == 1 for f in features["train"])
        bgs = set()
        for split in ("train", "test"):
            for pair, f in zip(puzzle[split], features[split]):
                frame_counts.add(f["frames"])
                modes["rect" if f["rect"] else "square"] += 1
                if f["frames"] > 1:
                    modes["layered" if f["layered"] else "mixed"] += 1
                bg = _analyze_input(pair["input"])["background"]
                bgs.add(bg)
                backgrounds.add(bg)
                shapes.add(tuple(pair["input"].shape))
        modes["varied_bg"] += len(bgs) > 1
    _check(len(signatures) == 42, "seeds collide")
    _check(shifts == set(SHIFTS), f"missing shift kinds: {shifts}")
    _check(frame_counts == {1, 2, 3, 4}, f"frame counts {frame_counts}")
    _check(len(backgrounds) >= 7 and len(shapes) >= 60, "too little canvas/palette variety")
    _check(all(v >= 3 for v in modes.values()), f"mode coverage too thin: {modes}")
