"""Compass docking followed by a parity-controlled crop of the inner canvas.

Only standard-library modules and NumPy are used. Generation and bounded tests
are in memory; this module never writes payloads, logs, or execution traces.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W
_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _components(points, diagonal=True):
    remaining = set(points)
    result = []
    steps = _NEIGHBORS if diagonal else _DIRECTIONS
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _transform(cells, pose):
    transformed = []
    for (r, c), color in cells.items():
        if pose >= 4:
            c = -c
        for _ in range(pose % 4):
            r, c = c, -r
        transformed.append((r, c, color))
    r0 = min(r for r, c, color in transformed)
    c0 = min(c for r, c, color in transformed)
    return {(r - r0, c - c0): color for r, c, color in transformed}


def _signature(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _canonical(points):
    points = tuple(points)
    return min(_signature(_transform({p: 1 for p in points}, t)) for t in range(8))


def _shift(cells, origin):
    return {(r + origin[0], c + origin[1]): color for (r, c), color in cells.items()}


def _extent(cells):
    return 1 + max(r for r, c in cells), 1 + max(c for r, c in cells)


def _halo(points):
    return {(r + dr, c + dc) for r, c in points for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _label_directions(cells, bands):
    core = [p for p, color in cells.items() if color not in bands]
    if not core:
        return None
    r0, r1 = min(r for r, c in core), max(r for r, c in core)
    c0, c1 = min(c for r, c in core), max(c for r, c in core)
    labels = {}
    for (r, c), color in cells.items():
        if color not in bands:
            continue
        options = [i for i, ok in enumerate((r == r0 - 1 and c0 <= c <= c1,
                                             c == c1 + 1 and r0 <= r <= r1,
                                             r == r1 + 1 and c0 <= c <= c1,
                                             c == c0 - 1 and r0 <= r <= r1)) if ok]
        if len(options) != 1:
            return None
        labels[(r, c)] = options[0]
    return labels or None


def _parse(input_grid):
    grid = _grid(input_grid)
    side = grid.shape[0]
    _require(grid.shape == (side, side) and side >= 6, "square framed canvas required")
    sides = (int(grid[0, 1]), int(grid[1, -1]), int(grid[-1, 1]), int(grid[1, 0]))
    _require(len(set(sides)) == 4, "four distinct compass bands required")
    _require(bool(np.all(grid[0] == sides[0]) and np.all(grid[-1] == sides[2])
                  and np.all(grid[1:-1, 0] == sides[3])
                  and np.all(grid[1:-1, -1] == sides[1])), "invalid one-cell frame")
    board = grid[1:-1, 1:-1].copy()
    counts = np.bincount(board.ravel(), minlength=10)
    background = int(np.argmax(counts))
    _require(counts[background] > board.size // 2, "background must occupy over half the canvas")
    _require(background not in sides, "background cannot be a band identity")
    objects = _components(zip(*np.nonzero(board != background)))
    sources, targets, target_colors = [], [], set()
    for obj in objects:
        cells = {p: int(board[p]) for p in obj}
        values = set(cells.values())
        if len(values) == 1:
            targets.append(frozenset(obj))
            target_colors.update(values)
        else:
            sources.append(cells)
    _require(bool(sources) and len(sources) == len(targets), "one detached stencil per piece required")
    _require(len(target_colors) == 1, "detached stencils must share one registration color")
    registration = next(iter(target_colors))
    body_colors = set()
    for cells in sources:
        values = set(cells.values())
        _require(registration in values and bool(values & set(sides)), "piece needs marks and labels")
        _require(_label_directions(cells, set(sides)) is not None, "unclear label facing")
        _require(len(_components(cells, diagonal=False)) == 1, "pieces must be four-connected")
        body_colors.update(values - set(sides) - {registration})
    _require(len(body_colors) == 1, "one visible body color required")
    body = next(iter(body_colors))
    _require(len({body, registration, background} | set(sides)) == 7, "color roles must be distinct")
    return {"board": board, "n": len(board), "background": background,
            "body": body, "registration": registration, "sides": sides,
            "sources": sources, "targets": targets}


def _candidates(source, info, use_labels=True):
    choices, seen = [], set()
    n, reg, sides = info["n"], info["registration"], info["sides"]
    for pose in range(8):
        piece = _transform(source, pose)
        directions = _label_directions(piece, set(sides))
        if directions is None:
            continue
        if use_labels and any(piece[p] != sides[d] for p, d in directions.items()):
            continue
        marks = {p for p, color in piece.items() if color == reg}
        mr, mc = min(marks)
        for i, target in enumerate(info["targets"]):
            if len(target) != len(marks):
                continue
            tr, tc = min(target)
            delta = tr - mr, tc - mc
            if {(r + delta[0], c + delta[1]) for r, c in marks} != target:
                continue
            placed = _shift(piece, delta)
            if any(not (0 <= r < n and 0 <= c < n) for r, c in placed):
                continue
            sig = (i, tuple(sorted(placed.items())))
            if sig not in seen:
                seen.add(sig)
                choices.append((i, placed))
    return choices


def _crop(canvas):
    return canvas[:2, :2].copy() if len(canvas) % 2 else canvas[:2, -2:].copy()


def _canvases(info, use_labels=True, crop_only=False, limit=2):
    """Search complete assignments, deduplicating pixels rather than pose IDs.

    crop_only is used solely for an ablation diagnostic. solve always requires
    a unique COMPLETE docked canvas, even if ambiguity would be cropped away.
    """
    choices = [_candidates(source, info, use_labels) for source in info["sources"]]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    results = {}

    def search(depth, used, occupied):
        if len(results) >= limit:
            return
        if depth == len(order):
            canvas = np.full_like(info["board"], info["background"])
            for p, color in occupied.items():
                canvas[p] = color
            sig = _crop(canvas).tobytes() if crop_only else canvas.tobytes()
            results[sig] = canvas
            return
        for target, placed in choices[order[depth]]:
            if target in used or occupied.keys() & placed.keys():
                continue
            search(depth + 1, used | {target}, {**occupied, **placed})

    search(0, set(), {})
    return list(results.values())


def _reconstruct(input_grid):
    info = _parse(input_grid)
    canvases = _canvases(info)
    _require(len(canvases) == 1, "docking must determine one complete canvas")
    return canvases[0], info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the uniquely docked canvas from this input, then crop by parity."""
    canvas, _ = _reconstruct(input_grid)
    return _crop(canvas)


def _path(rng, start, end):
    r, c = start
    result = {(r, c)}
    while (r, c) != end:
        if r != end[0] and (c == end[1] or rng.random() < 0.5):
            r += 1 if r < end[0] else -1
        else:
            c += 1 if c < end[1] else -1
        result.add((r, c))
    return result


def _pattern(rng, h, w, symmetric):
    simple = [{(0, 0)}, {(0, 0), (0, 1)}, {(0, 0), (1, 1)},
              {(0, 0), (0, 1), (0, 2)}, {(0, 0), (1, 0), (1, 1)},
              {(0, 0), (0, 1), (1, 0), (1, 1)},
              {(0, 1), (1, 0), (1, 1), (1, 2)},
              {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}]
    if symmetric or rng.random() < 0.5:
        marks = simple[int(rng.integers(len(simple)))]
    else:
        marks = {(0, 0)}
        goal = int(rng.integers(2, 7))
        for _ in range(40):
            if len(marks) == goal:
                break
            r, c = sorted(marks)[int(rng.integers(len(marks)))]
            dr, dc = _NEIGHBORS[int(rng.integers(len(_NEIGHBORS)))]
            trial = marks | {(r + dr, c + dc)}
            if max(r for r, c in trial) - min(r for r, c in trial) < h and max(c for r, c in trial) - min(c for r, c in trial) < w:
                marks = trial
    variants = [_transform({p: 1 for p in marks}, t) for t in range(8)]
    variants = [v for v in variants if _extent(v)[0] <= h and _extent(v)[1] <= w]
    pattern = variants[int(rng.integers(len(variants)))]
    ph, pw = _extent(pattern)
    offset = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
    return set(_shift(pattern, offset))


def _make_piece(rng, palette, max_core, symmetric=False):
    bg, body, reg, *sides = palette
    h, w = (int(rng.integers(3, max_core + 1)) for _ in range(2))
    marks = _pattern(rng, h, w, symmetric)
    # Two adjacent identities are sufficient to identify a D4 pose; other
    # subsets are admitted when registration and fit eliminate all ambiguity.
    if rng.random() < 0.65:
        d = int(rng.integers(4))
        selected = [d, (d + 1) % 4]
    else:
        selected = [int(d) for d in rng.choice(4, int(rng.integers(1, 5)), replace=False)]
    labels, supports = {}, []
    for d in selected:
        length = w if d % 2 == 0 else h
        patch = int(rng.integers(1, min(3, length) + 1))
        positions = sorted(int(v) for v in rng.choice(length, patch, replace=False))
        for v in positions:
            p, q = (((-1, v), (0, v)), ((v, w), (v, w - 1)),
                    ((h, v), (h - 1, v)), ((v, -1), (v, 0)))[d]
            labels[p] = sides[d]
            supports.append(q)
    endpoints = sorted(marks) + supports + [(0, int(rng.integers(w))),
                 (h - 1, int(rng.integers(w))), (int(rng.integers(h)), 0),
                 (int(rng.integers(h)), w - 1)]
    rng.shuffle(endpoints)
    core = {endpoints[0]}
    for endpoint in endpoints[1:]:
        distance = min(abs(r - endpoint[0]) + abs(c - endpoint[1]) for r, c in core)
        anchors = sorted(p for p in core if abs(p[0] - endpoint[0]) + abs(p[1] - endpoint[1]) == distance)
        core.update(_path(rng, anchors[int(rng.integers(len(anchors)))], endpoint))
    density = float(rng.uniform(0.45, 0.98))
    for _ in range(h * w):
        if len(core) >= h * w * density:
            break
        frontier = sorted({(r + dr, c + dc) for r, c in core for dr, dc in _DIRECTIONS
                           if 0 <= r + dr < h and 0 <= c + dc < w} - core)
        if not frontier:
            break
        core.add(frontier[int(rng.integers(len(frontier)))])
    if len(core - marks) < 2:
        return None
    return _transform({**{p: body for p in core}, **{p: reg for p in marks}, **labels}, 0)


def _construct(rng, n, count, palette, demonstration=None, require_compass=False):
    bg, body, reg, *sides = palette
    finals, final_blocked, patterns = [], set(), set()
    for i in range(count):
        found = False
        for _ in range(35):
            selected_corner = (i == 0 if n % 2 else i == 1)
            piece = _make_piece(rng, palette, 6 if n < 22 else 7,
                                symmetric=require_compass and selected_corner)
            if piece is None:
                continue
            pattern = _canonical(p for p, v in piece.items() if v == reg)
            if pattern in patterns:
                continue
            h, w = _extent(piece)
            if i < 2:
                row = int(rng.integers(2))
                col = int(rng.integers(2)) if i == 0 else n - w - int(rng.integers(2))
                origins = [(row, col)]
            else:
                origins = [(r, c) for r in range(3, n - h + 1) for c in range(n - w + 1)]
                rng.shuffle(origins)
            for origin in origins:
                placed = _shift(piece, origin)
                if placed.keys() & final_blocked:
                    continue
                finals.append(placed)
                final_blocked.update(_halo(placed))
                patterns.add(pattern)
                found = True
                break
            if found:
                break
        if not found:
            return None
    target_cells = {p: reg for piece in finals for p, v in piece.items() if v == reg}
    # Both corner patches are reserved from loose sources, but their stencils
    # may enter those patches. Everywhere else old and final footprints may overlap.
    corner_cells = {(r, c) for r in range(2) for c in (0, 1, n - 2, n - 1)}
    blocked, sources = _halo(target_cells) | corner_cells, []
    selected_index = 0 if n % 2 else 1
    for i, final in enumerate(finals):
        if i == selected_index and demonstration == "rotate":
            poses = [int(t) for t in rng.permutation([1, 2, 3])]
        elif i == selected_index and demonstration == "reflect":
            poses = [int(t) for t in rng.permutation([4, 5, 6, 7])]
        else:
            poses = [int(t) for t in rng.permutation(8)]
        found = False
        for pose in poses:
            piece = _transform(final, pose)
            if demonstration == "reflect" and i == selected_index and any(
                _signature(_transform(piece, t)) == _signature(final) for t in range(4)):
                continue
            h, w = _extent(piece)
            origins = [(r, c) for r in range(n - h + 1) for c in range(n - w + 1)]
            rng.shuffle(origins)
            for origin in origins:
                placed = _shift(piece, origin)
                if placed.keys() & blocked:
                    continue
                sources.append(placed)
                blocked.update(_halo(placed))
                found = True
                break
            if found:
                break
        if not found:
            return None
    inp = np.full((n + 2, n + 2), bg, dtype=np.int8)
    inp[0], inp[-1] = sides[0], sides[2]
    inp[1:-1, 0], inp[1:-1, -1] = sides[3], sides[1]
    for p, v in target_cells.items():
        inp[p[0] + 1, p[1] + 1] = v
    for piece in sources:
        for (r, c), v in piece.items():
            inp[r + 1, c + 1] = v
    expected_canvas = np.full((n, n), bg, dtype=np.int8)
    for piece in finals:
        for p, v in piece.items():
            expected_canvas[p] = v
    try:
        canvas, info = _reconstruct(inp)
        if not np.array_equal(canvas, expected_canvas):
            return None
        if any(np.array_equal(canvas[:2, sl], info["board"][:2, sl]) for sl in (slice(0, 2), slice(-2, None))):
            return None
        if np.array_equal(canvas[:2, :2], canvas[:2, -2:]):
            return None
        if require_compass and len(_canvases(info, use_labels=False, crop_only=True)) < 2:
            return None
        if demonstration:
            patch = _crop(canvas)
            if (len(np.unique(patch)) < 2 or not np.any(patch == body)
                    or int(np.sum(patch != _crop(info["board"]))) < 2):
                return None
    except ValueError:
        return None
    return {"input": inp, "output": _crop(canvas)}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Interleaved sizes disprove a size-threshold proxy for parity when >=4
    # demonstrations are requested. The first two always cover both branches.
    odd = int(rng.choice([17, 19, 21, 23]))
    even = int(rng.choice([16, 18, 20, 22, 24]))
    sides = [odd, even]
    if rng.random() < 0.5:
        sides.reverse()
    if num_train >= 4:
        sides += [int(rng.choice([v for v in (15, 17, 19, 21, 23, 25) if v != odd and v > even])),
                  int(rng.choice([v for v in (16, 18, 20, 22, 24, 26) if v != even and v > odd]))]
    while len(sides) < num_train:
        sides.append(int(rng.integers(16, 27)))
    # Tests increase assignment load, or use a new canvas scale. Neither shift
    # changes the parity mapping or introduces unseen label conventions.
    load_shift = bool(rng.random() < 0.5)
    test_parity = int(rng.integers(2))
    test_sides = [int(rng.choice([v for v in range(20 if load_shift else max(sides) + 1, 29)
                                  if v % 2 == (test_parity + j) % 2])) for j in range(num_test)]
    puzzle, used = {"train": [], "test": []}, set()
    for split, sizes in (("train", sides), ("test", test_sides)):
        for index, n in enumerate(sizes):
            count = (int(rng.integers(3, 5)) if split == "test" else 2) if load_shift else int(rng.integers(2, 4))
            demonstration = ("rotate" if index == 0 else "reflect" if index == 1 else None) if split == "train" else None
            for _ in range(160):
                palette = [int(v) for v in rng.permutation(10)[:7]]
                pair = _construct(rng, n, count, palette, demonstration,
                                  require_compass=demonstration is not None)
                if pair is not None and _key(pair["input"]) not in used:
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(input_grid, output_grid):
    canvas, info = _reconstruct(input_grid)
    out = _grid(output_grid)
    _require(out.shape == (2, 2) and np.array_equal(out, _crop(canvas)), "wrong parity crop")
    _require(not np.array_equal(canvas[:2, :2], canvas[:2, -2:]), "top corners must distinguish parity")
    for sl in (slice(0, 2), slice(-2, None)):
        _require(not np.array_equal(canvas[:2, sl], info["board"][:2, sl]), "both corner crops must change through docking")
    old = set().union(*(set(s) for s in info["sources"]))
    _require(not any((r, c) in old for r in range(2) for c in (0, 1, info["n"] - 2, info["n"] - 1)), "loose sources cannot occupy crop patches")
    _require(int(np.sum(canvas == info["body"])) == sum(list(s.values()).count(info["body"]) for s in info["sources"]), "complete body preservation")
    _require(int(np.sum(canvas == info["registration"])) == sum(len(t) for t in info["targets"]), "registration preservation")
    for color in info["sides"]:
        _require(int(np.sum(canvas == color)) == sum(list(s.values()).count(color) for s in info["sources"]), "label preservation")
    marks = [{p for p, v in s.items() if v == info["registration"]} for s in info["sources"]]
    moves = []
    for s in info["sources"]:
        cs = _candidates(s, info)
        _require(bool(cs), "source has no dock")
        final = cs[0][1]
        poses = [t for t in range(8) if _signature(_transform(s, t)) == _signature(final)]
        moves.append("translation" if 0 in poses else "rotation" if any(t < 4 for t in poses) else "reflection")
    return {"n": info["n"], "count": len(info["sources"]),
            "background": info["background"], "roles": (info["body"], info["registration"], *info["sides"]),
            "patterns": [_canonical(m) for m in marks],
            "piece_shapes": [_extent(_transform(s, 0)) for s in info["sources"]],
            "moves": moves,
            "compass_changes_crop": len(_canvases(info, use_labels=False, crop_only=True)) > 1,
            "new_cells": int(np.sum(_crop(canvas) != _crop(info["board"]))),
            "sampled_body": int(np.sum(out == info["body"])),
            "output_colors": len(np.unique(out)),
            "normalized": _normalized(input_grid),
            "crop_pattern": _normalized(out)}


def _normalized(grid):
    mapping = {}
    return tuple(mapping.setdefault(int(v), len(mapping)) for v in np.asarray(grid).flat)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid example counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            record = _inspect(inp, out)
            _require(2 <= record["count"] <= 4, "family uses two to four pieces")
            if split == "train":
                training.append(record)
    _require({r["n"] % 2 for r in training} == {0, 1}, "both parity demonstrations required")
    _require({r["n"] % 2 for r in training if r["compass_changes_crop"]} == {0, 1}, "both branches need a compass-sensitive answer witness")
    _require({r["n"] % 2 for r in training if r["compass_changes_crop"] and r["sampled_body"]
              and r["new_cells"] >= 2 and r["output_colors"] >= 2} == {0, 1},
             "both parity witnesses must reveal moved body cells in a nonuniform patch")
    _require(any("rotation" in r["moves"] for r in training), "rotation demonstration missing")
    _require(any("reflection" in r["moves"] for r in training), "reflection demonstration missing")
    if a >= 4:
        _require(all(len({r["n"] for r in training if r["n"] % 2 == p}) >= 2 for p in (0, 1)), "two sizes per parity required")
        odd = [r["n"] for r in training if r["n"] % 2]
        even = [r["n"] for r in training if not r["n"] % 2]
        _require(max(odd) > min(even) and max(even) > min(odd), "size-threshold shortcut not distinguished")
    train_sizes = [r["n"] for r in training]
    train_load = max(r["count"] for r in training)
    for pair in puzzle["test"]:
        info = _parse(pair["input"])
        _require(info["n"] > max(train_sizes) or len(info["sources"]) > train_load, "test needs a scale or assignment-load shift")


def _puzzle_digest(puzzle):
    serial = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]] for s in ("train", "test")}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """42 seeds, streaming metrics, no saved grids or traces; bounded retries."""
    # Independently hand-assembled geometry: congruent singleton registrations
    # must be assigned by compass pose AND boundary fit. This also exercises a
    # smaller board and target positions outside the constructor's usual marks.
    n = 14
    a = {(0, 0): 2, (0, 1): 1, (0, 2): 1,
         (1, 0): 1, (1, 1): 1, (1, 2): 1, (2, 1): 5, (0, 3): 4}
    b = {(0, 0): 1, (0, 1): 1, (0, 2): 2,
         (1, 0): 1, (1, 1): 1, (1, 2): 1, (2, 1): 5, (0, -1): 6}
    b = _transform(b, 0)
    finals = [a, _shift(b, (0, n - 4))]
    manual = np.zeros((n + 2, n + 2), dtype=np.int8)
    manual[0], manual[-1] = 3, 5
    manual[1:-1, 0], manual[1:-1, -1] = 6, 4
    expected = np.zeros((n, n), dtype=np.int8)
    for piece in finals:
        for (r, c), color in piece.items():
            expected[r, c] = color
            if color == 2:
                manual[r + 1, c + 1] = color
    for piece, pose, origin in ((a, 1, (8, 1)), (b, 5, (8, 8))):
        for (r, c), color in _shift(_transform(piece, pose), origin).items():
            manual[r + 1, c + 1] = color
    rebuilt, info = _reconstruct(manual)
    _require(np.array_equal(rebuilt, expected), "repeated-stencil fit test failed")
    _require(np.array_equal(solve(manual), np.array([[1, 2], [1, 1]])), "hand-built crop test failed")
    _require(len({_canonical(t) for t in info["targets"]}) == 1, "manual test must repeat a stencil")
    # A symmetric complete piece can have multiple equivalent D4 identifiers;
    # pixel-map deduplication must still accept its unique reconstruction.
    symmetric = {(0, 0): 1, (0, 1): 2, (0, 2): 1, (-1, 1): 3}
    single = np.zeros((12, 12), dtype=np.int8)
    single[0], single[-1] = 3, 5
    single[1:-1, 0], single[1:-1, -1] = 6, 4
    single[4, 5] = 2
    for (r, c), color in _shift(_transform(symmetric, 1), (7, 6)).items():
        single[r + 1, c + 1] = color
    symmetric_canvas, _ = _reconstruct(single)
    _require(int(symmetric_canvas[2, 4]) == 3 and int(symmetric_canvas[3, 4]) == 2,
             "equivalent-pose deduplication failed")
    digests, shapes, structures, patterns, piece_shapes, roles, backgrounds = (set() for _ in range(7))
    crop_patterns, moves, counts, test_parities = set(), set(), set(), set()
    pairs = compass = changed = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                r = _inspect(pair["input"], pair["output"])
                shapes.add(r["n"])
                structures.add((r["n"], r["normalized"]))
                patterns.update(r["patterns"])
                piece_shapes.update(r["piece_shapes"])
                backgrounds.add(r["background"])
                roles.add(r["roles"])
                crop_patterns.add(r["crop_pattern"])
                moves.update(r["moves"])
                counts.add(r["count"])
                compass += r["compass_changes_crop"]
                changed += r["new_cells"] > 0
                pairs += 1
                if split == "test":
                    test_parities.add(r["n"] % 2)
                original = pair["input"].copy()
                _require(np.array_equal(solve(original), pair["output"]), "input-only solve failed")
                _require(np.array_equal(original, pair["input"]), "solve mutated input")
    _require(len(digests) == 42 and len(structures) >= 190, "insufficient structural variation")
    _require(len(shapes) >= 10 and len(patterns) >= 15 and len(piece_shapes) >= 20, "geometry variation collapsed")
    _require(backgrounds == set(range(10)) and len(roles) >= 150, "palette role variation collapsed")
    _require(counts == {2, 3, 4} and moves == {"translation", "rotation", "reflection"}, "piece or motion variation collapsed")
    _require(test_parities == {0, 1} and compass >= 84 and changed == pairs, "causal coverage failed")
    _require(len(crop_patterns) >= 8, "crop geometry variation collapsed")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(13000 + 10 * a + b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_puzzle_digest(generate_puzzle(913, 6, 3)) == _puzzle_digest(generate_puzzle(913, 6, 3)), "seed reproducibility failed")
    return {"seeds": 42, "pairs": pairs, "canvas_sizes": sorted(shapes),
            "normalized_inputs": len(structures), "registration_shapes": len(patterns),
            "piece_extents": len(piece_shapes), "role_assignments": len(roles),
            "crop_patterns": len(crop_patterns), "compass_sensitive_pairs": compass,
            "docking_changes_answer": changed}
