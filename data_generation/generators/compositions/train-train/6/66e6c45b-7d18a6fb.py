"""Diagonal docking of silhouettes retrieved by a central 2x2 color key.

Only the standard library and NumPy are used. The solver reads the framed
arena, keys, reference silhouettes and walls entirely from the supplied input.
"""

from collections import Counter

import numpy as np


_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_LOCATIONS = ("top_left", "top_right", "bottom_left", "bottom_right", "inset")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside the permitted range")
    return value


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "invalid grid shape")
    _require(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid color")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _centers(shape):
    r, c = shape[0] // 2, shape[1] // 2
    return ((r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c))


def _start(shape, mask, quadrant):
    r, c = shape[0] // 2, shape[1] // 2
    h, w = mask.shape
    return (r - h if quadrant < 2 else r,
            c - w if quadrant % 2 == 0 else c)


def _origin(shape, mask, quadrant, step):
    r, c = _start(shape, mask, quadrant)
    dy, dx = _DIRECTIONS[quadrant]
    return r + step * dy, c + step * dx


def _cells(shape, mask, quadrant, step):
    r, c = _origin(shape, mask, quadrant, step)
    return {(r + int(y), c + int(x)) for y, x in np.argwhere(mask)}


def _limit(shape, mask, quadrant):
    r, c = _start(shape, mask, quadrant)
    h, w = mask.shape
    dy, dx = _DIRECTIONS[quadrant]
    return min(r - 1 if dy < 0 else shape[0] - 1 - r - h,
               c - 1 if dx < 0 else shape[1] - 1 - c - w)


def _dock(arena, wall, mask, quadrant, *, ignore_walls=False):
    """First blocked next step; never jump over a collision to a later fit."""
    limit = _limit(arena.shape, mask, quadrant)
    _require(limit >= 0, "reference does not fit its initial quadrant")
    _require(all(arena[p] != wall for p in _cells(arena.shape, mask, quadrant, 0)),
             "wall intersects an initial silhouette")
    for step in range(1, limit + 1):
        if not ignore_walls and any(arena[p] == wall
                                    for p in _cells(arena.shape, mask, quadrant, step)):
            return step - 1
    return limit


def _separated(a, b):
    r, c, h, w = a
    y, x, bh, bw = b
    return r + h < y or y + bh < r or c + w < x or x + bw < c


def _inspect_input(value):
    grid = _grid(value)
    values, counts = np.unique(grid, return_counts=True)
    _require(int(np.sum(counts == counts.max())) == 1, "ambiguous background")
    background = int(values[np.argmax(counts)])
    candidates = []
    for color in values:
        color = int(color)
        if color == background:
            continue
        coords = np.argwhere(grid == color)
        r, c = coords.min(axis=0).tolist()
        bottom, right = coords.max(axis=0).tolist()
        h, w = bottom - r + 1, right - c + 1
        if min(h, w) < 12 or h % 2 or w % 2:
            continue
        arena = grid[r:r + h, c:c + w]
        if not (np.all(arena[0] == color) and np.all(arena[-1] == color)
                and np.all(arena[:, 0] == color) and np.all(arena[:, -1] == color)):
            continue
        centers = _centers(arena.shape)
        markers = tuple(int(arena[p]) for p in centers)
        if len(set(markers)) < 2 or any(v in (background, color) for v in markers):
            continue
        cleared = arena.copy()
        for p in centers:
            cleared[p] = background
        if not set(np.unique(cleared)) <= {background, color}:
            continue
        candidates.append((r, c, h, w, color, markers))
    _require(len(candidates) == 1, "expected one complete rectangular arena frame")
    r, c, h, w, wall, markers = candidates[0]
    arena = grid[r:r + h, c:c + w].copy()
    outside = grid.copy()
    outside[r:r + h, c:c + w] = background
    colors = set(map(int, np.unique(outside))) - {background}
    _require(wall not in colors and set(markers) <= colors, "missing reference identity")
    _require(0 <= len(colors - set(markers)) <= 2, "too many distractors")
    masks, boxes = {}, {}
    for color in sorted(colors):
        coords = np.argwhere(outside == color)
        y, x = coords.min(axis=0).tolist()
        by, bx = coords.max(axis=0).tolist()
        mh, mw = by - y + 1, bx - x + 1
        _require(2 <= mh <= 4 and 2 <= mw <= 4, "reference must have tight 2..4 bounds")
        patch = outside[y:by + 1, x:bx + 1]
        _require(set(np.unique(patch)) <= {background, color}, "mixed reference box")
        mask = patch == color
        _require(3 <= int(mask.sum()) < mask.size, "reference must contain cells and holes")
        boxes[color] = (y, x, mh, mw)
        masks[color] = mask
    panel_box = (r, c, h, w)
    _require(all(_separated(box, panel_box) for box in boxes.values()), "reference touches frame")
    box_list = list(boxes.values())
    _require(all(_separated(a, b) for i, a in enumerate(box_list)
                 for b in box_list[i + 1:]), "references touch")
    for q, color in enumerate(markers):
        y, x = _start(arena.shape, masks[color], q)
        mh, mw = masks[color].shape
        _require(not np.any(arena[y:y + mh, x:x + mw] == wall),
                 "wall intersects an initial reference box")
        _dock(arena, wall, masks[color], q)
    return {"grid": grid, "background": background, "wall": wall,
            "box": panel_box, "arena": arena, "markers": markers,
            "masks": masks, "boxes": boxes, "distractors": colors - set(markers)}


def _render(info, steps, *, point_payload=False):
    output = info["arena"].copy()
    for p in _centers(output.shape):
        output[p] = info["background"]
    for q, color in enumerate(info["markers"]):
        mask = np.ones((1, 1), dtype=bool) if point_payload else info["masks"][color]
        for p in _cells(output.shape, mask, q, steps[q]):
            _require(output[p] != info["wall"], "rendered material overlaps wall")
            output[p] = color
    return output


def _steps(info, mode="normal"):
    result = []
    for q, color in enumerate(info["markers"]):
        mask = info["masks"][color]
        if mode == "box":
            mask = np.ones(mask.shape, dtype=bool)
        elif mode == "point":
            mask = np.ones((1, 1), dtype=bool)
        result.append(_dock(info["arena"], info["wall"], mask, q,
                            ignore_walls=mode == "rim"))
    return tuple(result)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Retrieve each key's silhouette, dock it diagonally, and crop the arena."""
    info = _inspect_input(input_grid)
    return _render(info, _steps(info))


def _mask(rng, shape, absent=None, present=None):
    for _ in range(100):
        flat = np.zeros(shape[0] * shape[1], dtype=bool)
        count = int(rng.integers(3, flat.size))
        flat[rng.choice(flat.size, count, replace=False)] = True
        mask = flat.reshape(shape)
        if absent is not None:
            mask[absent] = False
        if present is not None:
            mask[present] = True
        if (3 <= int(mask.sum()) < mask.size and mask[0].any() and mask[-1].any()
                and mask[:, 0].any() and mask[:, -1].any()):
            return mask
    # Tight, deterministic fallback, still with the requested transparent corner.
    mask = np.ones(shape, dtype=bool)
    mask[absent if absent is not None else (0, 0)] = False
    return mask


def _corner(mask, q):
    return (0 if q < 2 else mask.shape[0] - 1,
            0 if q % 2 == 0 else mask.shape[1] - 1)


def _frame_origin(rng, height, width, fh, fw, location):
    if location == "top_left":
        return 0, 0
    if location == "top_right":
        return 0, width - fw
    if location == "bottom_left":
        return height - fh, 0
    if location == "bottom_right":
        return height - fh, width - fw
    return int(rng.integers(1, height - fh)), int(rng.integers(1, width - fw))


def _pack(rng, frame_shape, masks, location):
    fh, fw = frame_shape
    order = sorted(masks, key=lambda c: -masks[c].size)
    for trial in range(24):
        height = 30 if trial >= 20 else int(rng.integers(max(20, fh + 4), 31))
        width = 30 if trial >= 20 else int(rng.integers(max(20, fw + 4), 31))
        pr, pc = _frame_origin(rng, height, width, fh, fw, location)
        panel = (pr, pc, fh, fw)
        boxes = {}
        for color in order:
            mh, mw = masks[color].shape
            candidates = [(r, c, mh, mw)
                          for r in range(height - mh + 1)
                          for c in range(width - mw + 1)
                          if _separated((r, c, mh, mw), panel)]
            for i in rng.permutation(len(candidates)):
                box = candidates[int(i)]
                if all(_separated(box, other) for other in boxes.values()):
                    boxes[color] = box
                    break
            else:
                break
        if len(boxes) == len(masks):
            return (height, width), panel, boxes
    raise RuntimeError("bounded reference packing failed")


def _make_pair(rng, interior_shape, background, wall, markers, distractors,
               location, obstacle_count, *, marker_hole=False, clone=False,
               force_gate=False):
    shape = (interior_shape[0] + 2, interior_shape[1] + 2)
    max_side = min(4, min(interior_shape) // 2 - 2)
    selected = list(dict.fromkeys(markers))
    hole_q = int(rng.integers(4)) if obstacle_count else None
    active = []
    if obstacle_count:
        active = [hole_q]
        active += list(map(int, rng.choice([q for q in range(4) if q != hole_q],
                                           obstacle_count - 1, replace=False)))
    gate_q = active[1] if force_gate else None
    masks = {}
    for color in selected + list(distractors):
        mask_shape = tuple(map(int, rng.integers(2, max_side + 1, size=2)))
        if gate_q is not None and color == markers[gate_q]:
            mask_shape = (2, 2)
        absent = present = None
        if hole_q is not None and color == markers[hole_q]:
            dummy = np.zeros(mask_shape, dtype=bool)
            absent = _corner(dummy, hole_q)
            dy, dx = _DIRECTIONS[hole_q]
            present = absent[0] - dy, absent[1] - dx
        elif marker_hole and color == markers[0]:
            absent = (mask_shape[0] - 1, mask_shape[1] - 1)
        masks[color] = _mask(rng, mask_shape, absent, present)
    if clone and distractors:
        masks[distractors[0]] = masks[selected[0]].copy()
    arena = np.full(shape, background, dtype=np.int8)
    arena[0] = arena[-1] = wall
    arena[:, 0] = arena[:, -1] = wall
    for p, color in zip(_centers(shape), markers):
        arena[p] = color
    steps = [_limit(shape, masks[color], q) for q, color in enumerate(markers)]
    for q in active:
        mask = masks[markers[q]]
        step = 1 if q == gate_q else int(rng.integers(1, steps[q]))
        steps[q] = step
        if q == hole_q:
            r, c = _origin(shape, mask, q, step)
            cy, cx = _corner(mask, q)
            blocker = r + cy, c + cx
        else:
            previous_box = _cells(shape, np.ones(mask.shape, dtype=bool), q, step)
            frontier = sorted(_cells(shape, mask, q, step + 1) - previous_box)
            blocker = frontier[int(rng.integers(len(frontier)))]
        arena[blocker] = wall
    if obstacle_count:
        # These decoys cannot intersect any actual silhouette at any stage.
        swept = set(_centers(shape))
        for q, color in enumerate(markers):
            swept.update(_cells(shape, np.ones(masks[color].shape, dtype=bool), q, 0))
            swept.update(set().union(*(_cells(shape, masks[color], q, s)
                                       for s in range(_limit(shape, masks[color], q) + 1))))
        quiet = [(r, c) for r in range(1, shape[0] - 1)
                 for c in range(1, shape[1] - 1)
                 if (r, c) not in swept and arena[r, c] == background]
        number = min(len(quiet), int(rng.integers(5)))
        for i in rng.choice(len(quiet), number, replace=False):
            arena[quiet[int(i)]] = wall
    info = {"arena": arena, "markers": markers, "masks": masks,
            "background": background, "wall": wall}
    # Construct the output from the explicit sampled stop plan. The public
    # solver independently rediscovers first collisions from the visible grid.
    output = _render(info, steps)
    canvas_shape, panel, boxes = _pack(rng, shape, masks, location)
    grid = np.full(canvas_shape, background, dtype=np.int8)
    r, c, fh, fw = panel
    grid[r:r + fh, c:c + fw] = arena
    for color, (y, x, mh, mw) in boxes.items():
        grid[y:y + mh, x:x + mw][masks[color]] = color
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    initial_roles = list(map(int, rng.permutation(10)))
    first_markers = tuple(initial_roles[2:6])
    repeat_mode = int(rng.choice((0, 2, 3)))
    locations = list(map(str, rng.permutation(_LOCATIONS)))
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(num_train + num_test):
        is_test = index >= num_train
        if index == 0:
            background, wall = initial_roles[:2]
            markers = first_markers
            distractors = (initial_roles[6],)
            interior = (int(rng.choice((10, 12))), int(rng.choice((10, 12))))
            obstacles, location = 0, locations[0]
        elif index == 1:
            spare = [v for v in range(10) if v not in first_markers]
            background = int(rng.choice([v for v in spare if v != initial_roles[0]]))
            wall = int(rng.choice([v for v in spare
                                  if v not in (background, initial_roles[1])]))
            markers = first_markers[1:] + first_markers[:1]
            if repeat_mode:
                markers = (markers[0], markers[1], markers[0],
                           markers[1] if repeat_mode == 2 else markers[2])
            available = [v for v in range(10) if v not in (background, wall, *markers)]
            distractors = tuple(map(int, rng.choice(available, 2, replace=False)))
            interior = (12, 14) if rng.integers(2) else (14, 12)
            obstacles, location = 2, locations[1]
        else:
            roles = list(map(int, rng.permutation(10)))
            background, wall = roles[:2]
            markers = tuple(roles[2:6])
            if repeat_mode and rng.random() < .65:
                markers = (roles[2], roles[3], roles[2],
                           roles[3] if repeat_mode == 2 else roles[4])
                markers = tuple(np.asarray(markers)[rng.permutation(4)].tolist())
            available = [v for v in roles if v not in (background, wall, *markers)]
            distractors = tuple(available[:int(rng.integers(3))])
            if is_test:
                options = ((14, 16), (16, 14), (16, 18), (18, 16),
                           (18, 18), (12, 18), (18, 12))
                interior = options[int(rng.integers(len(options)))]
            else:
                interior = tuple(map(int, rng.choice((10, 12, 14), size=2)))
            obstacles = int(rng.integers(1, 4)) if is_test or rng.random() < .8 else 0
            location = str(rng.choice(_LOCATIONS))
        for _ in range(30):
            pair = _make_pair(rng, interior, background, wall, markers, distractors,
                              location, obstacles, marker_hole=index == 0,
                              clone=index == 1, force_gate=index == 1)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["test" if is_test else "train"].append(pair)
                break
        else:
            raise RuntimeError("could not generate a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _inspect_input(inp)
            steps = _steps(info)
            expected = _render(info, steps)
            _require(np.array_equal(out, expected), "incorrect docking output")
            _require(all(s >= 1 for s in steps), "generated motifs must actually move")
            _require(np.array_equal(out == info["wall"], info["arena"] == info["wall"]),
                     "frame or obstacle altered")
            _require(all(out[p] == info["background"] for p in _centers(out.shape)),
                     "central keys must be erased")
            for q, color in enumerate(info["markers"]):
                mask = info["masks"][color]
                positions = _cells(out.shape, mask, q, steps[q])
                rmid, cmid = out.shape[0] // 2, out.shape[1] // 2
                _require(all((r < rmid) == (q < 2) and (c < cmid) == (q % 2 == 0)
                             for r, c in positions), "motif left its quadrant")
                if steps[q] < _limit(out.shape, mask, q):
                    _require(any(info["arena"][p] == info["wall"]
                                 for p in _cells(out.shape, mask, q, steps[q] + 1)),
                             "early stop lacks a visible collision")
            # Ablate each inherited mechanism on the actual example.
            _require(not np.array_equal(out, _render(info, (0, 0, 0, 0))),
                     "outward motion is irrelevant")
            point = _render(info, _steps(info, "point"), point_payload=True)
            _require(not np.array_equal(out, point), "reference lookup is irrelevant")
            info["steps"] = steps
            analyses[split].append(info)
    first, second = analyses["train"][:2]
    _require(len(set(first["markers"])) == 4, "first demo must expose all four identities")
    _require(not first["masks"][first["markers"][0]][-1, -1],
             "first demo must show an empty inward anchor")
    _require(first["background"] != second["background"] and first["wall"] != second["wall"],
             "first two demos must vary background and wall roles")
    shared = set(first["markers"]) & set(second["markers"])
    _require(sum(first["markers"].index(c) != second["markers"].index(c)
                 for c in shared) >= 2, "color-to-direction shortcut not distinguished")
    _require(not np.any(first["arena"][1:-1, 1:-1] == first["wall"]), "first demo must show rim docking")
    _require(any(a > b for a, b in zip(second["steps"], _steps(second, "box"))),
             "second demo must show a wall passing through a silhouette hole")
    _require(any(s < _limit(second["arena"].shape, second["masks"][color], q)
                 for q, (color, s) in enumerate(zip(second["markers"], second["steps"]))),
             "second demo must show wall-controlled stopping")
    _require(any(np.array_equal(second["masks"][d], second["masks"][m])
                 for d in second["distractors"] for m in second["markers"]),
             "second demo must distinguish color lookup from shape matching")
    _require(any(s < _limit(second["arena"].shape, second["masks"][color], q)
                 and all(second["arena"][p] != second["wall"]
                         for p in _cells(second["arena"].shape, second["masks"][color], q,
                                         _limit(second["arena"].shape, second["masks"][color], q)))
                 for q, (color, s) in enumerate(zip(second["markers"], second["steps"]))),
             "second demo must distinguish first collision from furthest free placement")
    train_max = max(max(i["arena"].shape) for i in analyses["train"])
    _require(all(max(i["arena"].shape) > train_max for i in analyses["test"]),
             "test must extrapolate arena scale")
    return None


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _check_hand_cases():
    """Two hand-derived placements, independent of the sampled stop planner."""
    other_masks = (np.array([[1, 1], [0, 1], [1, 0]], dtype=bool),
                   np.array([[1, 0, 1], [1, 1, 0]], dtype=bool),
                   np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool))
    other_cells = (((1, 10), (1, 11), (2, 11), (3, 10)),
                   ((10, 1), (10, 3), (11, 1), (11, 2)),
                   ((10, 10), (10, 11), (11, 11), (11, 12), (12, 10), (12, 12)))
    cases = ((np.array([[1, 1], [1, 0]], dtype=bool), (3, 3),
              ((4, 4), (4, 5), (5, 4))),
             (np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool), (2, 2),
              ((2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4))))
    for nw, blocker, nw_cells in cases:
        arena = np.zeros((14, 14), dtype=np.int8)
        arena[0] = arena[-1] = 1
        arena[:, 0] = arena[:, -1] = 1
        arena[blocker] = 1
        expected = arena.copy()
        for color, cells in zip((2, 3, 4, 5), (nw_cells,) + other_cells):
            for p in cells:
                expected[p] = color
        arena[6:8, 6:8] = [[2, 3], [4, 5]]
        grid = np.zeros((24, 25), dtype=np.int8)
        grid[1:15, 1:15] = arena
        for color, mask, row in zip((2, 3, 4, 5), (nw,) + other_masks, (0, 6, 12, 18)):
            mh, mw = mask.shape
            grid[row:row + mh, 18:18 + mw][mask] = color
        original = grid.copy()
        _require(np.array_equal(solve(grid), expected), "hand-derived collision case failed")
        _require(np.array_equal(grid, original), "solver modified its input")


def self_test():
    """Bounded, visible-grid invariants and structural coverage over 42 seeds."""
    signatures, shapes, masks, palettes, structures = set(), set(), set(), set(), set()
    counts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_input(pair["input"])
                output = solve(pair["input"])
                _require(np.array_equal(output, pair["output"]), "solver disagrees")
                _require(np.array_equal(solve(pair["input"].tolist()), output), "list input disagrees")
                _require(not np.shares_memory(output, pair["input"]), "solver aliases input")
                shapes.add(info["arena"].shape)
                palettes.add((info["background"], info["wall"]))
                masks.update(_key(mask) for mask in info["masks"].values())
                structures.add((info["arena"].shape,
                                (info["arena"] == info["wall"]).tobytes(),
                                tuple(_key(info["masks"][c]) for c in info["markers"])))
                counts[("distractors", len(info["distractors"]))] += 1
                counts[("identities", len(set(info["markers"])))] += 1
                for q, color in enumerate(info["markers"]):
                    mask = info["masks"][color]
                    counts[("motif_shape", mask.shape)] += 1
                    counts[("stop", _steps(info)[q])] += 1
                counts[("zero_background", info["background"] == 0)] += 1
                counts[("zero_payload", 0 in info["markers"])] += 1
    _require(len(signatures) == 42, "42 seeds did not yield 42 puzzles")
    _require(len(shapes) >= 12 and len(masks) >= 100 and len(structures) >= 180,
             "insufficient structural variation")
    _require(len(palettes) >= 40, "palette roles did not vary")
    _require(all(counts[("distractors", n)] for n in range(3)), "distractor counts did not vary")
    _require(all(counts[("identities", n)] for n in (2, 3, 4)), "repeated selectors absent")
    _require(all(counts[("motif_shape", (h, w))] for h in range(2, 5) for w in range(2, 5)),
             "rectangular reference dimensions did not vary")
    _require(counts[("zero_background", True)] and counts[("zero_payload", True)],
             "zero did not change semantic roles")
    _require(_signature(generate_puzzle(17)) == _signature(generate_puzzle(17)), "nondeterministic seed")
    _check_hand_cases()
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(100 + 10 * a + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "arena_shapes": len(shapes),
            "reference_masks": len(masks), "structural_signatures": len(structures),
            "background_wall_roles": len(palettes)}
