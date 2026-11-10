"""Compass-sealed reservoirs: orient, register, union, then flood enclosed space.

Only Python/NumPy are required. No files, seeds, examples, or generation metadata
are read by solve. All trials are bounded and no diagnostics are stored here.
"""
from __future__ import annotations

import json
import numpy as np

_D4 = tuple(range(8))
_N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0,16203515,129456500,367825991,402145502,275822072,453690175,511316176,205480214,373148399,681241802,718863001,654122918,805807930,570169205,578573671,338878134,808544948,580518889,564391239,718626975,145712919,373775805,215998953,241918113,756427383,53357857,723102491,603707339,227698267,906447780,182688491,944577543,540364225,709006430,433467612,252210589,546064879,722604706,520872418,558138866,616293837)


def _components(points, diagonal=False):
    todo, parts = set(points), []
    directions = _N8 if diagonal else _N4
    while todo:
        start = min(todo)
        todo.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _bbox(points):
    return min(r for r, c in points), max(r for r, c in points), min(c for r, c in points), max(c for r, c in points)


def _shift(points, dr, dc):
    return {(r + dr, c + dc) for r, c in points}


def _transform(cells, t):
    transformed = {}
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        transformed[r, c] = v
    top, _, left, _ = _bbox(transformed)
    return {(r - top, c - left): v for (r, c), v in transformed.items()}


def _signature(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _checked_grid(grid):
    a = np.asarray(grid)
    if a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape):
        raise ValueError("grid must have dimensions 1..30")
    if not np.issubdtype(a.dtype, np.integer) or not np.all((0 <= a) & (a <= 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _label_directions(cells, band_colors):
    core = {p for p, v in cells.items() if v not in band_colors}
    top, bottom, left, right = _bbox(core)
    labels = []
    for (r, c), v in cells.items():
        if v not in band_colors:
            continue
        directions = [d for d, yes in enumerate((r < top, c > right, r > bottom, c < left)) if yes]
        if len(directions) != 1:
            return None
        labels.append((directions[0], v))
    return labels


def _orientations(cells, bands, use_compass=True):
    options, seen = [], set()
    for t in _D4:
        moved = _transform(cells, t)
        directions = _label_directions(moved, set(bands))
        if not directions:
            continue
        if use_compass and any(bands[d] != v for d, v in directions):
            continue
        signature = tuple(sorted(moved.items()))
        if signature not in seen:
            seen.add(signature)
            options.append(moved)
    return options


def _parse(input_grid):
    a = _checked_grid(input_grid)
    if min(a.shape) < 8:
        raise ValueError("missing compass frame")
    bg = int(a[0, 0])
    if any(int(a[p]) != bg for p in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("the four corners must be background")
    bands = tuple(int(a[p]) for p in ((0, 1), (1, -1), (-1, 1), (1, 0)))
    if len(set(bands) | {bg}) != 5:
        raise ValueError("compass identities must be distinct")
    if not (np.all(a[0, 1:-1] == bands[0]) and np.all(a[1:-1, -1] == bands[1])
            and np.all(a[-1, 1:-1] == bands[2]) and np.all(a[1:-1, 0] == bands[3])):
        raise ValueError("incomplete compass frame")
    objects = _components({(r, c) for r, c in np.argwhere(a != bg)
                           if 0 < r < a.shape[0] - 1 and 0 < c < a.shape[1] - 1}, True)
    if len(objects) != 2:
        raise ValueError("expected two separate fragments")
    raw = [{p: int(a[p]) for p in obj} for obj in objects]
    role_colors = set(int(v) for v in np.unique(a)) - set(bands) - {bg}
    if len(role_colors) != 2:
        raise ValueError("expected one wall and one guide color")
    oriented = []
    for obj in raw:
        labels = _label_directions(obj, set(bands))
        if labels is None:
            raise ValueError("labels must project beyond the core")
        ds = {d for d, v in labels}
        if not any((d + 1) % 4 in ds for d in ds):
            raise ValueError("two adjacent compass directions must be labeled")
        choices = _orientations(obj, bands)
        if len(choices) != 1:
            raise ValueError("compass does not uniquely orient the fragment")
        oriented.append(choices[0])
    assignments = []
    for guide in sorted(role_colors):
        wall = next(c for c in role_colors if c != guide)
        for axis in (0, 1):
            records = [_registration(obj, wall, guide, axis) for obj in oriented]
            if any(r is None for r in records):
                continue
            if records[0]["length"] != records[1]["length"] or records[0]["negative"] == records[1]["negative"]:
                continue
            assignments.append((wall, guide, axis, records))
    if len(assignments) != 1:
        raise ValueError("guide role, axis, or backing is ambiguous")
    wall, guide, axis, records = assignments[0]
    records.sort(key=lambda rec: not rec["negative"])
    return {"background": bg, "bands": bands, "wall": wall, "guide": guide,
            "axis": axis, "records": records, "raw": raw, "oriented": oriented}


def _registration(cells, wall, guide, axis):
    marks = {(p[axis], p[1 - axis]) for p, v in cells.items() if v == guide}
    body = {(p[axis], p[1 - axis]) for p, v in cells.items() if v == wall}
    if not marks or not body:
        return None
    top, bottom, left, right = _bbox(marks)
    if top != bottom or right - left + 1 != len(marks) or not 2 <= len(marks) <= 5:
        return None
    body = _shift(body, -top, -left)
    length = len(marks)
    neg = any((-1, c) in body for c in range(length))
    pos = any((1, c) in body for c in range(length))
    if neg == pos:
        return None
    return {"body": body, "length": length, "negative": neg}


def _assembled(info, seam=-1, blend="union"):
    a = info["records"][0]["body"]
    b = _shift(info["records"][1]["body"], seam, 0)
    body = a | b if blend == "union" else a ^ b
    if info["axis"]:
        body = {(c, r) for r, c in body}
        a, b = ({(c, r) for r, c in part} for part in (a, b))
    return body, a, b


def _interior(body):
    """Four-connected empty cells that cannot reach outside the wall bbox."""
    top, bottom, left, right = _bbox(body)
    free = {(r, c) for r in range(top - 1, bottom + 2)
            for c in range(left - 1, right + 2)} - body
    outside, stack = {(top - 1, left - 1)}, [(top - 1, left - 1)]
    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            q = r + dr, c + dc
            if q in free and q not in outside:
                outside.add(q)
                stack.append(q)
    return free - outside


def _render(body, bg, wall, guide, fill=True):
    top, bottom, left, right = _bbox(body)
    out = np.full((bottom - top + 1, right - left + 1), bg, dtype=np.int8)
    for r, c in body:
        out[r - top, c - left] = wall
    if fill:
        for r, c in _interior(body):
            out[r - top, c - left] = guide
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the fixed family rule using only the current input pixels."""
    info = _parse(input_grid)
    body, _, _ = _assembled(info)
    return _render(body, info["background"], info["wall"], info["guide"])


def _compass_free_outputs(info, limit=2):
    """Ablation: retain guide geometry but allow every compass orientation."""
    possibilities = [_orientations(obj, info["bands"], False) for obj in info["raw"]]
    answers = set()
    for x in possibilities[0]:
        for y in possibilities[1]:
            for axis in (0, 1):
                records = [_registration(obj, info["wall"], info["guide"], axis) for obj in (x, y)]
                if any(r is None for r in records) or records[0]["negative"] == records[1]["negative"]:
                    continue
                records.sort(key=lambda rec: not rec["negative"])
                candidate = {**info, "axis": axis, "records": records}
                body, _, _ = _assembled(candidate)
                out = _render(body, info["background"], info["wall"], info["guide"])
                answers.add(_grid_key(out))
                if len(answers) >= limit:
                    return answers
    return answers


def _grow_overlap(own, other, cut, start, end, depth):
    """Copy a connected continuation of the other half outside the guide span."""
    result = set(own)
    allowed = {p for p in other if abs(p[0] - cut) <= depth and (p[1] < start or p[1] > end)}
    for _ in range(depth + 2):
        extra = {p for p in allowed - result if any((p[0] + dr, p[1] + dc) in result for dr, dc in _N8)}
        if not extra:
            break
        result |= extra
    return result


def _geometry(rng, height, width, length, open_bay, overlap, extra_walls, guide_divider):
    """A stepped perimeter with independent partitions, gates, and seam tails."""
    cut = int(rng.integers(3, height - 2))
    middle = int(rng.integers(3, width - 3))
    lefts, rights = {}, {}
    l, r = 1, width - 2
    for y in range(1, height - 1):
        l = int(np.clip(l + int(rng.integers(-1, 2)), 1, 2))
        r = int(np.clip(r + int(rng.integers(-1, 2)), width - 3, width - 2))
        lefts[y], rights[y] = min(l, middle - 2), max(r, middle + 2)
    interior = {(y, x) for y in range(1, height - 1) for x in range(lefts[y], rights[y] + 1)}
    body = {(y + dr, x + dc) for y, x in interior for dr, dc in _N4} - interior
    # A full internal divider makes an open bay and a sealed room coexist.
    body |= {(y, middle) for y in range(height)}
    if extra_walls:
        options = [y for y in range(2, height - 2) if abs(y - cut) >= 2]
        rng.shuffle(options)
        for y in options[:extra_walls]:
            if rng.integers(2):
                body |= {(y, x) for x in range(lefts[y], middle)}
            else:
                body |= {(y, x) for x in range(middle + 1, rights[y] + 1)}
    # Short bent baffles grow from existing walls without spanning a chamber.
    # Their lengths and locations are independent of guide length and palette.
    for _ in range(int(rng.integers(4))):
        y = int(rng.integers(1, height - 1))
        if rng.integers(2):
            stop = int(rng.integers(lefts[y], middle - 1))
            body |= {(y, x) for x in range(lefts[y], stop + 1)}
        else:
            stop = int(rng.integers(middle + 2, rights[y] + 1))
            body |= {(y, x) for x in range(stop, rights[y] + 1)}
        tip = y + int(rng.choice((-1, 1))), stop
        if tip in interior and tip[1] != middle:
            body.add(tip)
    if open_bay:
        gate_row = cut - 1 if rng.integers(2) else cut
        body -= {(gate_row, x) for x in range(rights[gate_row] + 1, width + 1)}
    # Either a flank or the internal divider backs the guide. The open teaching
    # example uses the divider so uncovered guide cells visibly become exterior.
    start = middle if guide_divider else min(x for y, x in body if y == cut - 1)
    if not any((cut, x) in body for x in range(start, start + length)):
        return None
    a, b = {p for p in body if p[0] < cut}, {p for p in body if p[0] >= cut}
    if overlap:
        depth = int(rng.integers(1, 4))
        a, b = (_grow_overlap(a, b, cut, start, start + length - 1, depth),
                _grow_overlap(b, a, cut, start, start + length - 1, depth))
    # Canonical registration coordinates: A guide row 0, B guide row 0;
    # the solved B guide will be row -1.
    a = _shift(a, -cut, -start)
    b = _shift(b, -cut + 1, -start)
    marks = {(0, c) for c in range(length)}
    if a & marks or b & marks:
        return None
    if len(_components(a | marks, True)) != 1 or len(_components(b | marks, True)) != 1:
        return None
    return a, b


def _add_labels(rng, body, length, wall, guide, bands):
    marks = {(0, c) for c in range(length)}
    cells = {**{p: wall for p in body}, **{p: guide for p in marks}}
    core_points = set(cells)
    top, bottom, left, right = _bbox(cells)
    first = int(rng.integers(4))
    sides = {first, (first + 1) % 4}
    for d in range(4):
        if rng.random() < .28:
            sides.add(d)
    for d in sorted(sides):
        supports = sorted(p for p in core_points if
                          (p[0] == top if d == 0 else p[1] == right if d == 1
                           else p[0] == bottom if d == 2 else p[1] == left))
        count = int(rng.integers(1, min(2, len(supports)) + 1))
        for j in rng.choice(len(supports), count, replace=False):
            r, c = supports[int(j)]
            dr, dc = _N4[d]
            cells[r + dr, c + dc] = bands[d]
    return cells


def _stage(rng, pieces, bg, bands):
    normalized = [_transform(p, 0) for p in pieces]
    dims = [(max(r for r, c in p) + 1, max(c for r, c in p) + 1) for p in normalized]
    # Try free placement first, then a guaranteed fitting shelf layout.
    for attempt in range(24):
        if attempt < 18:
            h, w = int(rng.integers(20, 31)), int(rng.integers(20, 31))
            origins = [(int(rng.integers(2, h - ph - 1)), int(rng.integers(2, w - pw - 1))) for ph, pw in dims]
        else:
            horizontal = attempt % 2 == 0
            gap = 1 + int(rng.integers(2))
            ph, pw = dims[0]
            qh, qw = dims[1]
            if horizontal:
                h, w = max(ph, qh) + 4, pw + qw + gap + 4
                origins = [(2, 2), (2, 2 + pw + gap)]
            else:
                h, w = ph + qh + gap + 4, max(pw, qw) + 4
                origins = [(2, 2), (2 + ph + gap, 2)]
            if h > 30 or w > 30:
                continue
        moved = [{(r + dr, c + dc): v for (r, c), v in p.items()} for p, (dr, dc) in zip(normalized, origins)]
        halo = {(r + dy, c + dx) for r, c in moved[0] for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        if halo & moved[1].keys():
            continue
        grid = np.full((h, w), bg, dtype=np.int8)
        grid[0, 1:-1], grid[1:-1, -1], grid[-1, 1:-1], grid[1:-1, 0] = bands
        for p in moved:
            for pos, color in p.items():
                grid[pos] = color
        return grid
    return None


def _make_pair(rng, palette, height_range, lengths, open_bay, overlap, extra_walls, pose_kind, seen):
    bg, wall, guide, *bands = palette
    for attempt in range(120):
        height = int(rng.integers(*height_range))
        width = int(rng.integers(8, 12))
        length = int(rng.choice(lengths))
        guide_divider = pose_kind == "reflection" or bool(rng.integers(2))
        geometry = _geometry(rng, height, width, length, open_bay, overlap, extra_walls, guide_divider)
        if geometry is None:
            continue
        a, b = geometry
        constructed_body = a | _shift(b, -1, 0)
        pieces = [_add_labels(rng, part, length, wall, guide, bands) for part in (a, b)]
        # Whole assembly may use either seam axis and either compass polarity.
        scene_pose = int(rng.integers(8))
        pieces = [_transform(p, scene_pose) for p in pieces]
        # Rotate compass identities together with the canonical final bodies.
        # We instead relabel to the actual outward direction after this motion.
        for piece in pieces:
            core = {p for p, v in piece.items() if v not in bands}
            top, bottom, left, right = _bbox(core)
            for (r, c), v in list(piece.items()):
                if v in bands:
                    d = 0 if r < top else 1 if c > right else 2 if r > bottom else 3
                    piece[r, c] = bands[d]
        if pose_kind == "rotation":
            source_poses = [int(rng.integers(1, 4)), int(rng.integers(8))]
        elif pose_kind == "reflection":
            source_poses = [int(rng.integers(4, 8)), int(rng.integers(8))]
        else:
            source_poses = [int(rng.integers(8)), int(rng.integers(8))]
        staged = _stage(rng, [_transform(p, t) for p, t in zip(pieces, source_poses)], bg, bands)
        if staged is None or _grid_key(staged) in seen:
            continue
        try:
            info = _parse(staged)
        except ValueError:
            continue
        body, upper, lower = _assembled(info)
        filled = _interior(body)
        joint = filled - _interior(upper) - _interior(lower)
        if not joint or not filled:
            continue
        if pose_kind == "rotation" and not (_interior(upper) | _interior(lower)):
            continue
        if pose_kind == "reflection":
            marks = {(r, c) for r in (-1, 0) for c in range(length)}
            if info["axis"]:
                marks = {(c, r) for r, c in marks}
            top, bottom, left, right = _bbox(body)
            if not any(top <= r <= bottom and left <= c <= right
                       for r, c in marks - body - filled):
                continue
        if open_bay:
            top, bottom, left, right = _bbox(body)
            if not any((r, c) not in body | filled for r in range(top + 1, bottom) for c in range(left + 1, right)):
                continue
        if bool(upper & lower) != overlap:
            continue
        out = _render(body, bg, wall, guide)
        intended = set(_transform({p: wall for p in constructed_body}, scene_pose))
        if not np.array_equal(out, _render(intended, bg, wall, guide)):
            raise AssertionError("pixel solver disagrees with constructed wall assembly")
        wrong = _render(_assembled(info, seam=0)[0], bg, wall, guide)
        if np.array_equal(out, wrong):
            continue
        if len(_compass_free_outputs(info)) < 2:
            continue
        return {"input": staged, "output": out}
    raise RuntimeError("bounded reservoir construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(("guide_interpolation", "height", "partitions")))
    changing_palette = bool(rng.integers(2))
    palette = tuple(map(int, rng.choice(10, 7, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for i in range(count):
            if changing_palette:
                palette = tuple(map(int, rng.choice(10, 7, replace=False)))
            training = split == "train"
            height_range = (7, 10) if training and regime == "height" else (10, 13) if regime == "height" else (9, 12) if regime == "partitions" else (7, 12)
            lengths = (2,) if training and i % 2 == 0 else (4,)
            if regime != "guide_interpolation":
                lengths = (2, 3, 4, 5)
            elif not training:
                lengths = (3,)
            open_bay = bool(i % 2) if training and i < 2 else bool(rng.integers(2))
            overlap = bool(i % 2) if training and i < 2 else bool(rng.integers(2))
            extra = (1 - i % 2 if training else 2) if regime == "partitions" else int(rng.integers(3))
            if training and i == 0:
                extra = max(1, extra)
                height_range = (max(8, height_range[0]), height_range[1])
            pose = "rotation" if training and i == 0 else "reflection" if training and i == 1 else "any"
            pair = _make_pair(rng, palette, height_range, lengths, open_bay, overlap, extra, pose, seen)
            seen.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair, check_ablation=False):
    inp, out = (_checked_grid(pair[k]) for k in ("input", "output"))
    info = _parse(inp)
    body, a, b = _assembled(info)
    assert np.array_equal(out, _render(body, info["background"], info["wall"], info["guide"])), "wrong reservoir output"
    assert max(out.shape) <= 30
    assert set(map(int, np.unique(out))) <= {info["background"], info["wall"], info["guide"]}
    filled = _interior(body)
    joint = filled - _interior(a) - _interior(b)
    marks = {(r, c) for r in (-1, 0) for c in range(info["records"][0]["length"])}
    if info["axis"]:
        marks = {(c, r) for r, c in marks}
    top, bottom, left, right = _bbox(body)
    visible_marks = {p for p in marks if top <= p[0] <= bottom and left <= p[1] <= right}
    assert joint, "assembly must create newly enclosed space"
    assert int(np.sum(out == info["wall"])) == len(a) + len(b) - len(a & b)
    assert int(np.sum(out == info["guide"])) == len(filled)
    assert np.any(out[0] == info["wall"]) and np.any(out[-1] == info["wall"])
    assert np.any(out[:, 0] == info["wall"]) and np.any(out[:, -1] == info["wall"])
    assert not np.array_equal(out, _render(_assembled(info, seam=0)[0], info["background"], info["wall"], info["guide"]))
    if check_ablation:
        assert len(_compass_free_outputs(info)) >= 2, "compass should affect the answer"
    directions = [len({d for d, v in _label_directions(obj, set(info["bands"]))}) for obj in info["oriented"]]
    poses = []
    for raw, orient in zip(info["raw"], info["oriented"]):
        candidates = [t for t in _D4 if _signature(_transform(raw, t)) == _signature(orient)]
        poses.append("translation" if 0 in candidates else "rotation" if any(t < 4 for t in candidates) else "reflection")
    return {"input_shape": list(inp.shape), "output_shape": list(out.shape), "axis": info["axis"],
            "length": info["records"][0]["length"], "overlap": len(a & b), "filled": len(filled),
            "joint_fill": len(joint), "preexisting_fill": len(_interior(a) | _interior(b)),
            "erased_guides": len(visible_marks - body - filled),
            "open_bbox_cells": int(np.sum(out[1:-1, 1:-1] == info["background"])),
            "cavities": len(_components(filled)), "label_sides": directions,
            "moves": poses, "palette": [info["background"], info["wall"], info["guide"], *info["bands"]]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for i, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _grid_key(_checked_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _pair_info(pair, check_ablation=(split == "train" and i < 2))
            if split == "train":
                training.append(info)
    assert any(x["overlap"] == 0 for x in training)
    assert any(x["overlap"] > 0 for x in training)
    assert training[0]["preexisting_fill"] > 0
    assert training[1]["open_bbox_cells"] > 0
    assert training[1]["erased_guides"] > 0
    assert any("rotation" in x["moves"] for x in training)
    assert any("reflection" in x["moves"] for x in training)


def self_test():
    """42 seeds, streamed metrics and small mutation checks; writes no files."""
    shapes, structures, palettes, lengths, axes, moves, fills, signatures = (set() for _ in range(8))
    pairs = overlaps = joint = 0
    mutations = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_grid_key(p["input"]) for s in puzzle.values() for p in s))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _parse(inp)
                features = _pair_info(pair)
                body, a, b = _assembled(info)
                assert np.array_equal(solve(inp.copy()), out)
                shapes.add((inp.shape, out.shape))
                roles = {}
                structures.add(tuple(roles.setdefault(int(v), len(roles)) for v in inp.ravel()))
                palettes.add(tuple(features["palette"]))
                lengths.add(features["length"])
                axes.add(features["axis"])
                moves.update(features["moves"])
                fills.add(features["cavities"])
                pairs += 1
                overlaps += bool(a & b)
                joint += features["joint_fill"]
                for name, candidate in (("no_fill", _render(body, info["background"], info["wall"], info["guide"], False)),
                                        ("coincident_guides", _render(_assembled(info, 0)[0], info["background"], info["wall"], info["guide"])),
                                        ("reversed_seam", _render(_assembled(info, 1)[0], info["background"], info["wall"], info["guide"])),
                                        ("xor", _render(_assembled(info, blend="xor")[0], info["background"], info["wall"], info["guide"]))):
                    if not np.array_equal(candidate, out):
                        mutations.add(name)
    assert len(signatures) == 42 and pairs == 210
    assert len(shapes) >= 50 and len(structures) >= 180 and len(palettes) >= 35
    assert lengths == {2, 3, 4, 5} and axes == {0, 1}
    assert moves == {"translation", "rotation", "reflection"}
    assert 42 <= overlaps < pairs - 30 and joint > 500
    assert len(fills) >= 3
    assert mutations == {"no_fill", "coincident_guides", "reversed_seam", "xor"}
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            p = generate_puzzle(91 + a * 7 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (2, 2), (4, 0), (6, 5), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    return {"seeds": 42, "pairs": pairs, "input_output_shape_pairs": len(shapes),
            "normalized_input_structures": len(structures), "palettes": len(palettes),
            "guide_lengths": sorted(lengths), "axes": sorted(axes), "moves": sorted(moves),
            "overlapping_pairs": overlaps, "joint_fill_cells": joint, "cavity_counts": sorted(fills),
            "rejected_alternatives": sorted(mutations)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
