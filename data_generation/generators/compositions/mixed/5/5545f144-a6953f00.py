"""A tip-led journey, followed by a parity-selected crop of its terminal board.

No parent module, saved example, generation metadata, or RNG is used by solve.
All sampling is bounded; self_test streams 42 puzzles without writing artifacts.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _rotate(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _arrow(span, shaft, solid, rear):
    """Right-pointing mask, expressed relative to its projecting tip."""
    cells = {(0, -i) for i in range(shaft)}
    for width in range(1, span + 1):
        rows = range(-width, width + 1) if solid else (-width, width)
        cells.update((r, 1 - shaft - width) for r in rows)
    for k in range(1, rear + 1):
        cells.update((r, 1 - shaft - span - k) for r in (-span, span))
    return tuple(sorted(cells))


MASKS = tuple(_arrow(span, shaft, solid, rear)
              for span in (1, 2) for shaft in ((1, 2) if span == 1 else (1,))
              for solid in (False, True) for rear in range(3))


def _footprint(mask, tip, facing):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rotate(p, facing - 1) for p in mask)}


def _fits(cells, n):
    return all(0 <= r < n and 0 <= c < n for r, c in cells)


def _separated(cells, points):
    return all(max(abs(r - q[0]), abs(c - q[1])) > 1
               for r, c in cells for q in points)


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), "references must be distinct and aligned")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _order(tip, waypoints):
    points, directions, remaining = [tip], [], set(waypoints)
    while remaining:
        aligned = [p for p in remaining if p[0] == tip[0] or p[1] == tip[1]]
        rays = {_direction(tip, p) for p in aligned}
        _require(len(rays) == 1, "route needs exactly one occupied orthogonal ray")
        chosen = min(aligned, key=lambda p: abs(p[0] - tip[0]) + abs(p[1] - tip[1]))
        directions.append(next(iter(rays)))
        points.append(chosen)
        remaining.remove(chosen)
        tip = chosen
    return points, directions


def _components(cells):
    unseen, result = set(cells), []
    while unseen:
        component = {min(unseen)}
        unseen -= component
        stack = list(component)
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in unseen:
                        unseen.remove(p)
                        component.add(p)
                        stack.append(p)
        result.append(component)
    return result


def _geometry(cells):
    """Infer the input mask's reflection axis and unique extreme tip."""
    matches = []
    for tip in sorted(cells):
        for facing in range(4):
            offsets = {_rotate((r - tip[0], c - tip[1]), 1 - facing)
                       for r, c in cells}
            if (any(c > 0 for r, c in offsets)
                    or {p for p in offsets if p[1] == 0} != {(0, 0)}
                    or max(abs(r) for r, c in offsets) == 0
                    or {(-r, c) for r, c in offsets} != offsets):
                continue
            matches.append((tuple(sorted(offsets)), tip, facing))
    _require(len(matches) == 1, "mover must have one geometrically identifiable tip")
    return matches[0]


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= v <= 30 for v in grid.shape), "invalid grid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), "integer grids required")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid colors")
    return grid.astype(np.int8, copy=False)


def _split(grid):
    colors = np.unique(grid)
    _require(len(colors) in (2, 3), "expected background, ink, and optional divider")
    if len(colors) == 2:
        boards, axis, divider = [grid], "single", None
    else:
        possibilities = []
        for divider_color in colors:
            for axis_index in (0, 1):
                lines = np.flatnonzero(np.all(grid == divider_color, axis=1 - axis_index)).tolist()
                if not lines:
                    continue
                ends = [-1] + lines + [grid.shape[axis_index]]
                sizes = [b - a - 1 for a, b in zip(ends, ends[1:])]
                if min(sizes) <= 0 or len(set(sizes)) != 1:
                    continue
                parts = [grid[a + 1:b, :] if axis_index == 0 else grid[:, a + 1:b]
                         for a, b in zip(ends, ends[1:])]
                if all(p.shape[0] == p.shape[1] and divider_color not in p for p in parts):
                    possibilities.append((parts, "vertical" if axis_index == 0 else "horizontal",
                                          int(divider_color)))
        _require(len(possibilities) == 1, "divider segmentation must be unique")
        boards, axis, divider = possibilities[0]
    n = boards[0].shape[0]
    _require(n >= 2 and all(b.shape == (n, n) for b in boards), "boards must be equal squares")
    colors, counts = np.unique(boards[0], return_counts=True)
    _require(len(colors) == 2 and max(counts) > n * n // 2, "background must be the strict majority")
    bg, ink = int(colors[np.argmax(counts)]), int(colors[np.argmin(counts)])
    _require(all(set(np.unique(b)) == {bg, ink} for b in boards), "panel palette changed")
    return boards, bg, ink, divider, axis


def _render(n, bg, ink, mask, points, directions, index):
    facing = directions[0] if index == 0 else directions[index - 1]
    cells = _footprint(mask, points[index], facing)
    _require(_fits(cells, n), "mover extends outside board")
    grid = np.full((n, n), bg, dtype=np.int8)
    for r, c in cells | set(points[index + 1:]):
        grid[r, c] = ink
    return grid


def _crop(grid, opposite=False):
    left = bool(grid.shape[0] % 2) != opposite
    return (grid[:2, :2] if left else grid[:2, -2:]).copy()


def _analyse(input_grid):
    boards, bg, ink, divider, axis = _split(_grid(input_grid))
    n = boards[0].shape[0]
    components = _components({(int(r), int(c)) for r, c in zip(*np.where(boards[0] == ink))})
    movers = [c for c in components if len(c) > 1]
    _require(len(movers) == 1, "first panel must separate mover from singleton waypoints")
    mask, tip, initial = _geometry(movers[0])
    waypoints = {next(iter(c)) for c in components if len(c) == 1}
    _require(len(waypoints) >= 2, "at least two remaining waypoints required")
    points, directions = _order(tip, waypoints)
    _require(directions[0] == initial, "first leg must follow the visible tip")
    states = []
    for i, p in enumerate(points):
        cells = _footprint(mask, p, initial if i == 0 else directions[i - 1])
        _require(_fits(cells, n), "an arrival footprint does not fit")
        _require(_separated(cells, points[i + 1:]), "future waypoint touches an arrival footprint")
        states.append(_render(n, bg, ink, mask, points, directions, i))
    indices = []
    for board in boards:
        matches = [i for i, state in enumerate(states[:-1]) if np.array_equal(board, state)]
        _require(len(matches) == 1, "panel is not an exact nonterminal journey snapshot")
        _require(not indices or matches[0] > indices[-1], "snapshots must advance")
        indices.append(matches[0])
    info = dict(n=n, bg=bg, ink=ink, divider=divider, axis=axis, mask=mask,
                points=points, directions=directions, snapshots=indices)
    return states[-1], info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Finish the visible journey, then crop using the square BOARD's parity."""
    terminal, _ = _analyse(input_grid)
    return _crop(terminal)


def _predecessors(points, mask, n, terminal_facing):
    """Grow a route backwards; every old reference keeps the same future suffix."""
    q = points[0]
    options = []
    for facing, (dr, dc) in enumerate(DIRS):
        if len(points) == 1 and facing != terminal_facing:
            continue
        cells = _footprint(mask, q, facing)
        if not _fits(cells, n) or not _separated(cells, points[1:]):
            continue
        for distance in range(2, n):
            p = (q[0] - distance * dr, q[1] - distance * dc)
            if not _fits({p}, n) or not _separated({p}, points):
                continue
            aligned = [x for x in points if x[0] == p[0] or x[1] == p[1]]
            if any(_direction(p, x) != facing or abs(x[0] - p[0]) + abs(x[1] - p[1]) < distance
                   for x in aligned):
                continue
            options.append(p)
    return options


def _backward_route(rng, mask, n, tip, final, target, minimum):
    best = None
    for _ in range(7):
        points = [tip]
        for _ in range(target + 2):
            options = _predecessors(points, mask, n, final)
            if not options:
                break
            points.insert(0, options[int(rng.integers(len(options)))])
            facing = _direction(points[0], points[1])
            cells = _footprint(mask, points[0], facing)
            if len(points) - 1 >= minimum and _fits(cells, n) and _separated(cells, points[1:]):
                if best is None or len(points) > len(best):
                    best = points.copy()
                if len(points) - 1 >= target:
                    return points
    return best


def _poses(mask, n, corner, min_ink=1):
    extent = -min(c for r, c in mask) + max(abs(r) for r, c in mask) + 2
    columns = range(extent + 1) if corner == "left" else range(n - extent - 1, n)
    crop_cells = {(r, c) for r in (0, 1)
                  for c in ((0, 1) if corner == "left" else (n - 2, n - 1))}
    result = []
    for r in range(extent + 1):
        for c in columns:
            for facing in range(4):
                tip = (r, c)
                cells = _footprint(mask, tip, facing)
                if _fits(cells, n) and len(cells & crop_cells) >= min_ink and (cells - {tip}) & crop_cells:
                    result.append((tip, facing))
    return result


def _fallback(mask, n, corner, target):
    """Finite geometric fallback: a bend, then variable-length collinear arrivals."""
    span = max(abs(r) for r, c in mask)
    depth = -min(c for r, c in mask)
    top, left = (3, 3) if corner == "interior" else (0, 0)
    usable = n - 6 if corner == "interior" else n
    start = min(2 * target + 2, usable - depth - 1, usable - span - 1)
    start -= start % 2
    _require(start >= 4, "fallback requires room for a bend")
    points = [(start, span + 2), (start, span)]
    points += [(r, span) for r in range(start - 4, -1, -2)]
    points = [(r + top, c + left) for r, c in points]
    if corner == "right":
        points = [(r, n - 1 - c) for r, c in points]
    return points


def _combine(boards, axis, divider):
    if axis == "single":
        return boards[0].copy()
    dim = 0 if axis == "vertical" else 1
    bar = np.full((1, boards[0].shape[1]) if dim == 0 else (boards[0].shape[0], 1),
                  divider, dtype=np.int8)
    return np.concatenate([boards[0], bar, boards[1]], axis=dim)


def _no_rotation_crop(info):
    n, bg, ink = info["n"], info["bg"], info["ink"]
    grid = np.full((n, n), bg, dtype=np.int8)
    # A deliberately wrong shortcut, clipped only to make its comparison defined.
    for r, c in _footprint(info["mask"], info["points"][-1], info["directions"][0]):
        if 0 <= r < n and 0 <= c < n:
            grid[r, c] = ink
    return _crop(grid)


def _pair_from_route(rng, n, mask, points, axis, palette):
    directions = [_direction(a, b) for a, b in zip(points, points[1:])]
    indices = [0] if axis == "single" else [0, int(rng.integers(1, len(points) - 1))]
    bg, ink, divider = palette
    boards = [_render(n, bg, ink, mask, points, directions, i) for i in indices]
    inp = _combine(boards, axis, divider)
    # Ground truth is constructed from the sampled rigid mask, not by solve.
    terminal = _render(n, bg, ink, mask, points, directions, len(points) - 1)
    return {"input": inp, "output": _crop(terminal)}


def _contrast_route(rng, n, mask):
    """Visit the selected corner, then leave it for the discarded corner."""
    depth = -min(c for r, c in mask)
    row = int(rng.integers(4, n - depth))
    points = [(row, 1), (1, 1), (1, n - 1)]
    if rng.random() < .5 and n >= 12:
        points.insert(2, (1, int(rng.integers(4, n - 3))))
    if n % 2 == 0:
        points = [(r, n - 1 - c) for r, c in points]
    return points


def _make_pair(rng, n, mask, axis, palette, mode, target, teach_rotation=False):
    if mode == "contrast":
        return _pair_from_route(rng, n, mask, _contrast_route(rng, n, mask), axis, palette)
    selected = "left" if n % 2 else "right"
    corner = selected if mode == "selected" else ("right" if selected == "left" else "left")
    minimum = 5 if n >= 21 else 2
    if mode == "interior":
        poses = [((int(rng.integers(5, n - 5)), int(rng.integers(5, n - 5))),
                  int(rng.integers(4))) for _ in range(24)]
        corner = "interior"
    else:
        poses = _poses(mask, n, corner, min_ink=2 if teach_rotation else 1)
        rng.shuffle(poses)
    for tip, final in poses[:24]:
        if not _fits(_footprint(mask, tip, final), n):
            continue
        points = _backward_route(rng, mask, n, tip, final, target, minimum)
        if points is None:
            continue
        pair = _pair_from_route(rng, n, mask, points, axis, palette)
        terminal, info = _analyse(pair["input"])
        if mode == "selected" and np.array_equal(pair["output"], _crop(_split(pair["input"])[0][0])):
            continue
        if teach_rotation and np.array_equal(pair["output"], _no_rotation_crop(info)):
            continue
        if mode == "interior" and np.any(_crop(terminal) != palette[0]):
            continue
        return pair
    points = _fallback(mask, n, corner, target)
    return _pair_from_route(rng, n, mask, points, axis, palette)


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    axis = str(rng.choice(["horizontal", "vertical"]))
    shared_palette = tuple(int(c) for c in rng.choice(10, 3, replace=False)) if rng.random() < .2 else None
    first_mask = MASKS[int(rng.choice([0, 1, 2]))]  # hollow, short head; expose preservation/rotation
    puzzle, seen, train_sizes = {"train": [], "test": []}, set(), set()
    first_test_parity = int(rng.integers(2))
    for split, count in (("train", a), ("test", b)):
        for i in range(count):
            if split == "train":
                parity = 1 - i % 2
                pool = [n for n in range(10, 15 if i < 2 else 21) if n % 2 == parity and n not in train_sizes]
                if not pool:
                    pool = [n for n in range(10, 21) if n % 2 == parity]
                n = int(rng.choice(pool))
                train_sizes.add(n)
                layout = axis if i < 2 else "single"
                mode = "contrast" if i == 2 else "selected"
                pool_masks = MASKS[:12] if mode == "contrast" else MASKS
                mask = first_mask if i == 0 else pool_masks[int(rng.integers(len(pool_masks)))]
                if i == 1 and mask == first_mask:
                    mask = MASKS[3]
                target = int(rng.integers(3, 6))
            else:
                parity = (first_test_parity + i) % 2
                n = int(rng.choice([n for n in range(21, 31) if n % 2 == parity]))
                layout = "single"
                mode = "selected" if a == 2 or (b > 1 and i == 0) else str(rng.choice(
                    ["selected", "opposite", "interior"], p=[.7, .2, .1]))
                if b > 1 and i == 1 and a >= 3:
                    mode = "opposite"
                mask = MASKS[int(rng.integers(len(MASKS)))]
                target = int(rng.integers(6, 10))
            palette = shared_palette or tuple(int(c) for c in rng.choice(10, 3, replace=False))
            pair = _make_pair(rng, n, mask, layout, palette, mode, target,
                              teach_rotation=(split == "train" and i == 0))
            if _key(pair["input"]) in seen:
                # A finite role-permutation fallback; no unbounded retries.
                old_bg, old_ink, old_divider = palette
                for bg in range(10):
                    for ink in range(10):
                        if bg == ink:
                            continue
                        divider = next(c for c in range(10) if c not in (bg, ink))
                        mapping = np.arange(10, dtype=np.int8)
                        mapping[[old_bg, old_ink, old_divider]] = [bg, ink, divider]
                        recolored = {k: mapping[v] for k, v in pair.items()}
                        if _key(recolored["input"]) not in seen:
                            pair = recolored
                            break
                    else:
                        continue
                    break
                else:
                    raise RuntimeError("exhausted distinct role assignments")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_info, test_info = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(out.shape == (2, 2), "output must be 2x2")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            terminal, info = _analyse(inp)
            _require(np.array_equal(out, _crop(terminal)), "incorrect terminal parity crop")
            _require(np.count_nonzero(terminal == info["ink"]) == len(info["mask"]), "mask changed")
            (train_info if split == "train" else test_info).append((info, terminal, out, inp))
    for parity in (0, 1):
        witnesses = [(info, terminal, out, inp) for info, terminal, out, inp in train_info
                     if info["n"] % 2 == parity and np.any(out != info["bg"])]
        _require(bool(witnesses), "training must show mover content in each parity's selected corner")
        _require(any(not np.array_equal(out, _crop(terminal, opposite=True))
                     and not np.array_equal(out, _crop(_split(inp)[0][0]))
                     for info, terminal, out, inp in witnesses), "missing joint A/B witness")
    _require(len({v[0]["mask"] for v in train_info}) >= 2, "training must vary the visible mask")
    _require(any(not np.array_equal(out, _no_rotation_crop(info))
                 for info, terminal, out, inp in train_info), "rotation is not observable")
    if len(train_info) >= 3:
        _require(any(np.all(out == info["bg"]) and np.any(_crop(terminal, opposite=True) != info["bg"])
                     and not np.array_equal(out, _crop(_split(inp)[0][0]))
                     for info, terminal, out, inp in train_info), "missing discarded-corner contrast")
    if len(train_info) >= 4:
        for parity in (0, 1):
            _require(len({v[0]["n"] for v in train_info if v[0]["n"] % 2 == parity}) >= 2,
                     "need multiple training sizes for each parity")
    _require(all(info["n"] > max(v[0]["n"] for v in train_info) for info, _, _, _ in test_info),
             "missing larger-board test shift")


def _puzzle_digest(puzzle):
    value = {s: [{k: np.asarray(v).tolist() for k, v in pair.items()} for pair in puzzle[s]]
             for s in ("train", "test")}
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Bounded grid-derived semantic, counterfactual, diversity, and API checks."""
    digests, masks, routes, sizes, palettes, output_patterns, directions, axes = (set() for _ in range(8))
    mixed_turns = straight = discarded = 0
    parity_counterfactuals = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                terminal, info = _analyse(pair["input"])
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list input solve")
                masks.add(info["mask"])
                points, ds = info["points"], info["directions"]
                relative = [(r - points[0][0], c - points[0][1]) for r, c in points]
                routes.add(min(tuple(_rotate(p, k) for p in relative) for k in range(4)))
                sizes.add(info["n"])
                palettes.add((info["bg"], info["ink"]))
                output_patterns.add(tuple(int(v == info["ink"]) for v in pair["output"].flat))
                directions.add(ds[-1])
                axes.add(info["axis"])
                turns = {(b - a) % 4 for a, b in zip(ds, ds[1:])}
                mixed_turns += {1, 3} <= turns
                straight += any(a == b for a, b in zip(ds, ds[1:]))
                discarded += np.all(pair["output"] == info["bg"])
        # Hold the entire route and mask fixed. Adding a background row and
        # column flips board parity, so the left-corner terminal is discarded.
        # Pad each panel separately: the outer snapshot-strip parity is irrelevant.
        witness = puzzle["train"][0]
        boards, bg, _, divider, axis = _split(witness["input"])
        padded = [np.pad(board, ((0, 1), (0, 1)), constant_values=bg) for board in boards]
        changed = _combine(padded, axis, divider)
        _, before = _analyse(witness["input"])
        _, after = _analyse(changed)
        _require(before["points"] == after["points"] and before["mask"] == after["mask"],
                 "parity counterfactual changed the journey")
        _require(np.all(solve(changed) == bg) and np.any(witness["output"] != bg),
                 "parity must change the answer while the journey is fixed")
        parity_counterfactuals += 1
    _require(len(digests) == 42, "puzzles repeat across seeds")
    _require(len(masks) >= 12 and len(routes) >= 100, "structural diversity collapsed")
    _require(len(sizes) >= 17 and len(palettes) >= 40, "size/palette diversity collapsed")
    _require(len(output_patterns) >= 6 and directions == set(range(4)), "terminal variety collapsed")
    _require(axes == {"single", "horizontal", "vertical"}, "snapshot format variety collapsed")
    _require(mixed_turns >= 10 and straight >= 10 and discarded >= 42, "route/contrast coverage collapsed")
    # Hold the selected input corner, board, colors, and initial mover fixed.
    # One extra future waypoint sends the terminal shape out of the aperture.
    journey_counterfactuals = 0
    for n in (13, 14):
        points = [(5, 7), (5, 1), (1, 1)]
        extended = points + [(1, n - 1)]
        if n % 2 == 0:
            points = [(r, n - 1 - c) for r, c in points]
            extended = [(r, n - 1 - c) for r, c in extended]
        rng = np.random.default_rng(104)
        a = _pair_from_route(rng, n, MASKS[0], points, "single", (0, 8, 4))
        b = _pair_from_route(rng, n, MASKS[0], extended, "single", (0, 8, 4))
        _require(np.array_equal(_crop(a["input"]), _crop(b["input"])),
                 "journey counterfactual changed the raw selected crop")
        _require(np.array_equal(solve(a["input"]), a["output"])
                 and np.array_equal(solve(b["input"]), b["output"])
                 and not np.array_equal(a["output"], b["output"]),
                 "a remote route change must alter the terminal crop")
        journey_counterfactuals += 1
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(9000 + 10 * a + b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "explicit counts ignored")
    _require(_puzzle_digest(generate_puzzle(730, 7, 3)) == _puzzle_digest(generate_puzzle(730, 7, 3)),
             "nondeterministic generation")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4, False), (4., 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for bad in ({(3, 5), (5, 3)}, {(3, 5), (6, 7)}):
        try:
            _order((3, 3), bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("junction or disconnected route accepted")
    p = generate_puzzle(123)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    p = generate_puzzle(123)
    boards, _, _, divider, axis = _split(p["train"][0]["input"])
    try:
        solve(_combine([boards[0], boards[0]], axis, divider))
    except AssertionError:
        pass
    else:
        raise AssertionError("repeated snapshots accepted")
    return {"seeds": 42, "distinct_puzzles": len(digests), "masks": len(masks),
            "routes_up_to_rotation": len(routes), "board_sizes": sorted(sizes),
            "palettes": len(palettes), "binary_output_patterns": len(output_patterns),
            "mixed_turn_routes": mixed_turns, "straight_continuations": straight,
            "blank_crops": int(discarded), "parity_counterfactuals": parity_counterfactuals,
            "journey_counterfactuals": journey_counterfactuals, "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
