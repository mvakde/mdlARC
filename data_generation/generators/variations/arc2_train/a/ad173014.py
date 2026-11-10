from __future__ import annotations

import numpy as np


_SIDES = ("top", "right", "bottom", "left")

_SOURCE_INPUTS = (
    (
        "0000000000000000000000",
        "0000002222200000000000",
        "0011112060211111110000",
        "0010002666200000010000",
        "0010002060200000010000",
        "0010002222200000010000",
        "0010000000000002222200",
        "0010000000000002030200",
        "0010000000000002303200",
        "0010000000000002030200",
        "0010000000000002222200",
        "0010000022222000010000",
        "0010000020802000010000",
        "0010000020882000010000",
        "0010000028002000010000",
        "0011111122222111110000",
        "0000000000000000000000",
        "0000000000000000000000",
        "0000000000000000000000",
        "0000000000000000000000",
    ),
    (
        "000000000000000000000",
        "000000022222000000000",
        "000000020302000000000",
        "001111123332111111000",
        "001000020302000001000",
        "001000022222000001000",
        "022222000000000001000",
        "020772000000000222220",
        "027702000000000288020",
        "027002000000000288020",
        "022222000000000200820",
        "001000000000000222220",
        "001000000000000001000",
        "001000000000000001000",
        "001000222220000001000",
        "001000206020000001000",
        "001111206621111111000",
        "000000200020000000000",
        "000000222220000000000",
        "000000000000000000000",
        "000000000000000000000",
        "000000000000000000000",
    ),
    (
        "00000000000000000000000000",
        "00000000222222200000000000",
        "00000000200800200000000000",
        "00000000208880200000000000",
        "00011111208080211111100000",
        "00010000208080200000100000",
        "00010000200000200000100000",
        "00010000222222200000100000",
        "00010000000000000222222200",
        "00010000000000000200000200",
        "02222222000000000200440200",
        "02000002000000000244404200",
        "02303002000000000200440200",
        "02333302000000000200000200",
        "02303002000000000222222200",
        "02000002000000000000100000",
        "02222222000000000000100000",
        "00010000000000000000100000",
        "00010000000000000000100000",
        "00011111111111111111100000",
        "00000000000000000000000000",
        "00000000000000000000000000",
        "00000000000000000000000000",
        "00000000000000000000000000",
        "00000000000000000000000000",
        "00000000000000000000000000",
    ),
    (
        "000000222222200222222200000",
        "000000200000200200000200000",
        "000111204040211203330211000",
        "000100204440200233033201000",
        "000100204440200203330201000",
        "000100200400200200000201000",
        "000100222222200222222201000",
        "000100000000000000000001000",
        "000100000000000000000001000",
        "222222200000000000000001000",
        "208000200000000000022222220",
        "208800200000000000020070020",
        "200880200000000000027777020",
        "200088200000000000020070020",
        "200080200000000000020077720",
        "222222200000000000020000020",
        "000100000000000000022222220",
        "000100000000000000000001000",
        "000100000000000000000001000",
        "000100000000000000000001000",
        "000100000022222220000001000",
        "000100000020060020000001000",
        "000100000026666620000001000",
        "000111111120060021111111000",
        "000000000020060020000000000",
        "000000000020660020000000000",
        "000000000022222220000000000",
    ),
)


def _require(condition: bool, message: str = "validation failed") -> None:
    if not condition:
        raise AssertionError(message)


def _validated_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed_key: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _validated_count(num_train, "num_train", 2, 10)
    test = _validated_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed_key % 4
        low = max(2, test + 1)
        train = low + (seed_key // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed_key % (11 - low)
    elif test is None:
        test = 1 + seed_key % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return int(train), int(test)


def _side_counts(rng: np.random.Generator, count: int, clustered: bool) -> dict:
    for _ in range(80):
        counts = [0, 0, 0, 0]
        for side in rng.choice(4, size=count, replace=True):
            counts[int(side)] += 1
        if max(counts) > 3:
            continue
        if clustered != (max(counts) >= 2):
            continue
        if count == 3 and max(counts) == 3:
            continue
        return dict(zip(_SIDES, counts))
    # A constructive fallback also covers the low-probability three-panel case.
    counts = [0, 0, 0, 0]
    order = [int(x) for x in rng.permutation(4)]
    if clustered:
        counts[order[0]] = 2
        for index in range(count - 2):
            counts[order[1 + index % 3]] += 1
    else:
        for index in range(count):
            counts[order[index]] = 1
    return dict(zip(_SIDES, counts))


def _clockwise_key(panel: dict) -> tuple[int, int]:
    r0, _, c0, _ = panel["rect"]
    side = panel["side"]
    return (_SIDES.index(side), c0 if side == "top" else r0 if side == "right" else -c0 if side == "bottom" else -r0)


def _sample_motif(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    kind = int(rng.integers(0, 7))
    if kind == 0:
        mask = rng.random((height, width)) < float(rng.uniform(.2, .7))
    elif kind == 1:
        r = int(rng.integers(0, height))
        mask[r, :] = True
        mask[:, int(rng.integers(0, width))] = True
    elif kind == 2:
        for r in range(height):
            mask[r, min(width - 1, int((r + .5) * width / height))] = True
        for _ in range(max(1, height // 2)):
            mask[int(rng.integers(0, height)), int(rng.integers(0, width))] = True
    elif kind == 3:
        r0 = int(rng.integers(0, height))
        r1 = int(rng.integers(r0, height))
        c0 = int(rng.integers(0, width))
        c1 = int(rng.integers(c0, width))
        mask[r0:r1 + 1, c0:c1 + 1] = True
    elif kind == 4:
        mask[::2, ::2] = True
        if int(rng.integers(0, 2)):
            mask = np.fliplr(mask)
    elif kind == 5:
        r, c = int(rng.integers(0, height)), int(rng.integers(0, width))
        mask[r, c] = True
        for _ in range(2 * (height + width)):
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(0, 4))]
            r, c = min(height - 1, max(0, r + dr)), min(width - 1, max(0, c + dc))
            mask[r, c] = True
    else:
        center = int(rng.integers(0, width))
        for r in range(height):
            left = max(0, center - r % 3)
            right = min(width, center + r % 3 + 1)
            mask[r, left:right] = True
    if not mask.any():
        mask[int(rng.integers(0, height)), int(rng.integers(0, width))] = True
    if mask.all():
        mask[int(rng.integers(0, height)), int(rng.integers(0, width))] = False
    return mask


def _place_panels(rng: np.random.Generator, counts: dict, box: tuple, shape: tuple, square_size: int | None) -> list[dict] | None:
    top, bottom, left, right = box
    height, width = shape
    panels = []
    for side in _SIDES:
        count = counts[side]
        if not count:
            continue
        axis_start, axis_end = (left, right) if side in ("top", "bottom") else (top, bottom)
        capacity = axis_end - axis_start - 3
        min_need = 4 * count + 2 * (count - 1)
        if min_need > capacity:
            return None
        lengths = [square_size] * count if square_size is not None else [4] * count
        if square_size is None:
            extra = int(rng.integers(0, min(capacity - min_need, 4 * count) + 1))
            for _ in range(extra):
                choices = [i for i, size in enumerate(lengths) if size < 8]
                lengths[int(rng.choice(choices))] += 1
        if sum(lengths) + 2 * (count - 1) > capacity:
            return None
        slack = capacity - sum(lengths) - 2 * (count - 1)
        gaps = rng.multinomial(slack, np.ones(count + 1) / (count + 1))
        start = axis_start + 2 + int(gaps[0])
        for index, long_size in enumerate(lengths):
            short_size = square_size if square_size is not None else int(rng.integers(4, 9))
            line = top if side == "top" else bottom if side == "bottom" else left if side == "left" else right
            possible_offsets = [o for o in range(short_size) if 0 <= line - o and line - o + short_size <= (height if side in ("top", "bottom") else width)]
            if not possible_offsets:
                return None
            # Interior crossings are common; border-flush mounts remain possible.
            interior_offsets = [o for o in possible_offsets if 0 < o < short_size - 1]
            offsets = interior_offsets if interior_offsets and int(rng.integers(0, 5)) else possible_offsets
            offset = int(rng.choice(offsets))
            if side in ("top", "bottom"):
                rect = (line - offset, line - offset + short_size - 1, start, start + long_size - 1)
            else:
                rect = (start, start + long_size - 1, line - offset, line - offset + short_size - 1)
            for old in panels:
                a, b, c, d = old["rect"]
                e, f, g, h = rect
                if not (b + 1 < e or f + 1 < a or d + 1 < g or h + 1 < c):
                    return None
            panels.append({"side": side, "rect": rect})
            start += long_size + 2 + int(gaps[index + 1])
    return sorted(panels, key=_clockwise_key)


def _sample_geometry(rng: np.random.Generator, counts: dict, regime: str) -> tuple | None:
    vertical_cluster = max(counts["left"], counts["right"])
    horizontal_cluster = max(counts["top"], counts["bottom"])
    min_span_h = max(12, 4 * vertical_cluster + 2 * (vertical_cluster - 1) + 3)
    min_span_w = max(12, 4 * horizontal_cluster + 2 * (horizontal_cluster - 1) + 3)
    min_h = max(19, min_span_h + 3)
    min_w = max(19, min_span_w + 3)
    if min_h > 30 or min_w > 30:
        return None
    h = int(rng.integers(min_h, 31 if regime == "target_test" else 30))
    w = int(rng.integers(min_w, 31 if regime == "target_test" else 30))
    if regime == "target_test" and int(rng.integers(0, 5)) == 0:
        h, w = max(27, h), max(27, w)
    span_h = int(rng.integers(min_span_h, min(h - 3, 24) + 1))
    span_w = int(rng.integers(min_span_w, min(w - 3, 24) + 1))
    top = int(rng.integers(1, h - span_h - 1))
    left = int(rng.integers(1, w - span_w - 1))
    box = (top, top + span_h, left, left + span_w)
    return (h, w), box


def _pair_palette(rng: np.random.Generator, parameters: dict, split: str, count: int) -> tuple:
    is_test = split == "target_test"
    if parameters["fixed_roles"]:
        background, track, frame = parameters["roles"]
        choices = [color for color in parameters["accent_pool"] if color != parameters["reserved"] or is_test]
    else:
        role_choices = [i for i in range(10) if i != parameters["reserved"]]
        roles = [int(x) for x in rng.choice(role_choices, size=3, replace=False)]
        background, track, frame = roles
        choices = [i for i in range(10) if i not in roles and (i != parameters["reserved"] or is_test)]
    if is_test and parameters["reserved"] is not None and parameters["reserved"] in choices:
        other = [i for i in choices if i != parameters["reserved"]]
        accents = [parameters["reserved"]] + [int(x) for x in rng.choice(other, size=count - 1, replace=False)]
        accents = [accents[int(i)] for i in rng.permutation(count)]
    else:
        accents = [int(x) for x in rng.choice(choices, size=count, replace=False)]
    return int(background), int(track), int(frame), accents


def generate_pair(rng: np.random.Generator, regime: str, parameters: dict, plan: dict) -> tuple[np.ndarray, np.ndarray]:
    counts = plan["side_counts"]
    uniform_square = int(rng.integers(0, 4)) == 0
    for _ in range(400):
        geometry = _sample_geometry(rng, counts, regime)
        if geometry is None:
            continue
        shape, box = geometry
        square_size = None
        if uniform_square:
            top, bottom, left, right = box
            limits = []
            for side in _SIDES:
                n = counts[side]
                if n:
                    span = right - left if side in ("top", "bottom") else bottom - top
                    limits.append((span - 3 - 2 * (n - 1)) // n)
            square_size = int(rng.integers(4, min(7, min(limits)) + 1))
        panels = _place_panels(rng, counts, box, shape, square_size)
        if panels is not None:
            break
    else:
        raise ValueError("could not place nonoverlapping panels")
    background, track, frame, accents = _pair_palette(rng, parameters, regime, len(panels))
    grid = np.full(shape, background, dtype=np.int8)
    top, bottom, left, right = box
    grid[top, left:right + 1] = track
    grid[bottom, left:right + 1] = track
    grid[top:bottom + 1, left] = track
    grid[top:bottom + 1, right] = track
    for panel, accent in zip(panels, accents):
        r0, r1, c0, c1 = panel["rect"]
        grid[r0:r1 + 1, c0:c1 + 1] = background
        grid[r0, c0:c1 + 1] = frame
        grid[r1, c0:c1 + 1] = frame
        grid[r0:r1 + 1, c0] = frame
        grid[r0:r1 + 1, c1] = frame
        mask = _sample_motif(rng, r1 - r0 - 1, c1 - c0 - 1)
        grid[r0 + 1:r1, c0 + 1:c1][mask] = accent
        panel["accent"] = accent
        panel["mask"] = mask
    output = grid.copy()
    for i, panel in enumerate(panels):
        r0, r1, c0, c1 = panel["rect"]
        output[r0 + 1:r1, c0 + 1:c1][panel["mask"]] = panels[(i + 1) % len(panels)]["accent"]
    return grid, output


def _rectangle_components(grid: np.ndarray, color: int) -> list[tuple] | None:
    mask = grid == color
    visited = np.zeros(grid.shape, dtype=bool)
    rectangles = []
    for raw_r, raw_c in np.argwhere(mask):
        r, c = int(raw_r), int(raw_c)
        if visited[r, c]:
            continue
        stack = [(r, c)]
        visited[r, c] = True
        cells = set()
        while stack:
            x, y = stack.pop()
            cells.add((x, y))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                xx, yy = x + dr, y + dc
                if 0 <= xx < grid.shape[0] and 0 <= yy < grid.shape[1] and mask[xx, yy] and not visited[xx, yy]:
                    visited[xx, yy] = True
                    stack.append((xx, yy))
        r0, r1 = min(x for x, _ in cells), max(x for x, _ in cells)
        c0, c1 = min(y for _, y in cells), max(y for _, y in cells)
        if r1 - r0 < 3 or c1 - c0 < 3:
            return None
        border = {(x, y) for x in range(r0, r1 + 1) for y in range(c0, c1 + 1) if x in (r0, r1) or y in (c0, c1)}
        if cells != border:
            return None
        rectangles.append((r0, r1, c0, c1))
    return sorted(rectangles)


def _try_analyze(grid: np.ndarray, background: int, frame: int, rectangles: list[tuple]) -> dict:
    _require(3 <= len(rectangles) <= 6)
    panels = []
    all_rect_cells = set()
    accents = []
    for rect in rectangles:
        r0, r1, c0, c1 = rect
        values = {int(v) for v in np.unique(grid[r0 + 1:r1, c0 + 1:c1])}
        _require(background in values and len(values) == 2, "one nonempty accent and background per panel")
        accent = next(v for v in values if v != background)
        _require(accent != frame)
        cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        _require(not all_rect_cells.intersection(cells), "overlapping panels")
        all_rect_cells.update(cells)
        panels.append({"rect": rect, "accent": accent})
        accents.append(accent)
    _require(len(set(accents)) == len(accents), "accent roles must be distinct")
    other = {int(v) for v in np.unique(grid)} - {background, frame} - set(accents)
    _require(len(other) == 1, "one distinct track role")
    track = next(iter(other))
    visible_track = {(int(r), int(c)) for r, c in np.argwhere(grid == track)}
    _require(visible_track)
    top = min(r for r, _ in visible_track)
    bottom = max(r for r, _ in visible_track)
    left = min(c for _, c in visible_track)
    right = max(c for _, c in visible_track)
    _require(0 < top < bottom < grid.shape[0] - 1 and 0 < left < right < grid.shape[1] - 1)
    perimeter = ({(top, c) for c in range(left, right + 1)} | {(bottom, c) for c in range(left, right + 1)} | {(r, left) for r in range(top, bottom + 1)} | {(r, right) for r in range(top, bottom + 1)})
    _require(visible_track == perimeter - all_rect_cells, "track must be one rectangle hidden only under panels")
    for panel in panels:
        r0, r1, c0, c1 = panel["rect"]
        sides = []
        if r0 <= top <= r1 and left + 1 < c0 <= c1 < right - 1 and r1 < bottom:
            sides.append("top")
        if c0 <= right <= c1 and top + 1 < r0 <= r1 < bottom - 1 and c0 > left:
            sides.append("right")
        if r0 <= bottom <= r1 and left + 1 < c0 <= c1 < right - 1 and r0 > top:
            sides.append("bottom")
        if c0 <= left <= c1 and top + 1 < r0 <= r1 < bottom - 1 and c1 < right:
            sides.append("left")
        _require(len(sides) == 1, "panel must cross one track side away from corners")
        panel["side"] = sides[0]
        _require(any((r, c) in perimeter for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)))
    allowed = np.zeros(grid.shape, dtype=bool)
    for r, c in perimeter:
        allowed[r, c] = True
    for r0, r1, c0, c1 in rectangles:
        allowed[r0:r1 + 1, c0:c1 + 1] = True
    _require(np.all(grid[~allowed] == background), "unrelated cells")
    _require(int(grid[0, 0]) == background)
    return {"background": background, "track": track, "frame": frame, "track_box": (top, bottom, left, right), "panels": sorted(panels, key=_clockwise_key)}


def _analyze_input(grid: np.ndarray) -> dict:
    background = int(grid[0, 0])
    parses = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == background:
            continue
        rectangles = _rectangle_components(grid, color)
        if rectangles is None or not 3 <= len(rectangles) <= 6:
            continue
        try:
            parses.append(_try_analyze(grid, background, color, rectangles))
        except AssertionError:
            pass
    _require(len(parses) == 1, "input must have one structural parse")
    return parses[0]


def _expected_output(grid: np.ndarray, analysis: dict) -> np.ndarray:
    expected = grid.copy()
    panels = analysis["panels"]
    for i, panel in enumerate(panels):
        r0, r1, c0, c1 = panel["rect"]
        source = grid[r0 + 1:r1, c0 + 1:c1]
        target = expected[r0 + 1:r1, c0 + 1:c1]
        target[source == panel["accent"]] = panels[(i + 1) % len(panels)]["accent"]
    return expected


def _grid_checks(grid: np.ndarray) -> None:
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2)
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30)
    _require(np.issubdtype(grid.dtype, np.integer))
    _require(bool(np.all((0 <= grid) & (grid <= 9))))


def _matches_source_input(grid: np.ndarray) -> bool:
    return tuple("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUTS


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"})
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list))
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test)
    analyses = {"train": [], "test": []}
    signatures = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"})
            grid, output = pair["input"], pair["output"]
            _grid_checks(grid)
            _grid_checks(output)
            _require(grid.shape == output.shape and not _matches_source_input(grid))
            signature = (grid.shape, grid.tobytes())
            _require(signature not in signatures, "duplicate input")
            signatures.add(signature)
            analysis = _analyze_input(grid)
            _require(np.array_equal(output, _expected_output(grid, analysis)), "incorrect successor recoloring")
            _require(not np.array_equal(grid, output))
            _require(3 <= len(analysis["panels"]) <= 6)
            analyses[split].append(analysis)
    train_cluster = any(max(sum(p["side"] == side for p in a["panels"]) for side in _SIDES) >= 2 for a in analyses["train"])
    test_cluster = any(max(sum(p["side"] == side for p in a["panels"]) for side in _SIDES) >= 2 for a in analyses["test"])
    _require(not test_cluster or train_cluster, "training must demonstrate same-side ordering")
    # Direction and wraparound are shown by every three-or-more-panel training cycle.
    _require(all(len(a["panels"]) >= 3 for a in analyses["train"]))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_key = int(seed) % (1 << 64)
    train_count, test_count = _resolve_counts(seed_key, num_train, num_test)
    rng = np.random.default_rng(np.uint64(seed_key))
    fixed_roles = bool(rng.integers(0, 2))
    palette = [int(v) for v in rng.permutation(10)]
    reserved = palette[-1] if int(rng.integers(0, 2)) else None
    parameters = {"fixed_roles": fixed_roles, "roles": tuple(palette[:3]), "accent_pool": tuple(palette[3:]), "reserved": reserved}
    test_cluster_flags = [bool(rng.integers(0, 4)) for _ in range(test_count)]
    # True is the common shifted case; a simple test occurs in some puzzles.
    test_cluster_flags = [flag or bool(rng.integers(0, 2)) for flag in test_cluster_flags]
    train_cluster_flags = [any(test_cluster_flags)] + [False]
    train_cluster_flags += [bool(rng.integers(0, 2)) for _ in range(train_count - 2)]
    plans = []
    for i in range(train_count):
        clustered = train_cluster_flags[i]
        count = int(rng.integers(3, 7 if clustered else 5))
        if clustered:
            count = max(4, count)
        plans.append(("train", {"side_counts": _side_counts(rng, count, clustered)}))
    for clustered in test_cluster_flags:
        count = int(rng.integers(4, 7)) if clustered else int(rng.integers(3, 5))
        plans.append(("test", {"side_counts": _side_counts(rng, count, clustered)}))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, plan in plans:
        pair_rng = np.random.default_rng(int(rng.integers(0, (1 << 63) - 1)))
        for _ in range(20):
            grid, output = generate_pair(pair_rng, "source_train" if split == "train" else "target_test", parameters, plan)
            sig = (grid.shape, grid.tobytes())
            if sig not in seen and not _matches_source_input(grid):
                break
        else:
            raise ValueError("could not generate unique pair")
        seen.add(sig)
        puzzle[split].append({"input": grid, "output": output})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, role_patterns = set(), set(), set(), set(), set()
    clustered_train = clustered_test = simple_test = edge = straddled = rectangular = mixed_size = novel_test = 0
    multiple_clusters = fixed_role_puzzles = varying_role_puzzles = 0
    side_clusters = set()
    track_sizes = set()
    panel_counts = {"train": set(), "test": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = {int(v) for p in puzzle["train"] for v in np.unique(p["input"])}
        test_accents = set()
        pair_roles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                analysis = _analyze_input(grid)
                _require(np.array_equal(pair["output"], _expected_output(grid, analysis)))
                shapes.add(grid.shape)
                pair_roles.add((analysis["background"], analysis["track"], analysis["frame"]))
                top, bottom, left, right = analysis["track_box"]
                track_sizes.add((bottom - top, right - left))
                panels = analysis["panels"]
                panel_counts[split].add(len(panels))
                sizes = {(p["rect"][1] - p["rect"][0] + 1, p["rect"][3] - p["rect"][2] + 1) for p in panels}
                mixed_size += len(sizes) > 1
                rectangular += any(h != w for h, w in sizes)
                edge += any(0 in (p["rect"][0], p["rect"][2]) or p["rect"][1] == grid.shape[0] - 1 or p["rect"][3] == grid.shape[1] - 1 for p in panels)
                straddled += any(p["rect"][0] < top < p["rect"][1] if p["side"] == "top" else p["rect"][2] < right < p["rect"][3] if p["side"] == "right" else p["rect"][0] < bottom < p["rect"][1] if p["side"] == "bottom" else p["rect"][2] < left < p["rect"][3] for p in panels)
                counts = tuple(sum(p["side"] == side for p in panels) for side in _SIDES)
                multiple_clusters += sum(n >= 2 for n in counts) >= 2
                cluster = max(counts) >= 2
                if split == "train":
                    clustered_train += cluster
                else:
                    clustered_test += cluster
                    simple_test += not cluster
                    test_accents.update(p["accent"] for p in panels)
                for i, n in enumerate(counts):
                    if n >= 2:
                        side_clusters.add(_SIDES[i])
                layouts.add((grid.shape, analysis["track_box"], tuple((p["side"], p["rect"]) for p in panels)))
        role_patterns.add(tuple(sorted(pair_roles)))
        fixed_role_puzzles += len(pair_roles) == 1
        varying_role_puzzles += len(pair_roles) > 1
        novel_test += bool(test_accents - train_colors)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3)
    _require(len(shapes) >= 35 and len(track_sizes) >= 35 and len(layouts) >= 150)
    _require(len(side_clusters) == 4 and clustered_train >= 15 and clustered_test >= 15 and simple_test >= 2)
    _require(panel_counts["train"] == {3, 4, 5, 6} and panel_counts["test"] == {3, 4, 5, 6})
    _require(multiple_clusters >= 10 and (27, 27) in shapes)
    _require(min(h for h, _ in track_sizes) <= 13 and max(h for h, _ in track_sizes) >= 22)
    _require(min(w for _, w in track_sizes) <= 13 and max(w for _, w in track_sizes) >= 22)
    _require(edge >= 5 and straddled >= 50 and rectangular >= 50 and mixed_size >= 50)
    _require(novel_test >= 3 and len(role_patterns) >= 15)
    _require(fixed_role_puzzles >= 5 and varying_role_puzzles >= 5)
    first = generate_puzzle(314159, 6, 3)
    _require(_puzzles_equal(first, generate_puzzle(314159, 6, 3)))
    _require(not _puzzles_equal(first, generate_puzzle(314160, 6, 3)))
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(5000 + i, train, test)
        _require(len(puzzle["train"]) == train and len(puzzle["test"]) == test)
    for i, arguments in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        generate_puzzle(6000 + i, *arguments)
    for arguments in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(7000, *arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
