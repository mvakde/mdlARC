"""Phase-casting: fit macro-pieces, then grow interacting pixel-width contours.

Only Python's standard library and NumPy are used. No files are read or written
by this module. All solving conventions are visible in the queried input.
"""

from collections import Counter, deque
import hashlib
from math import isqrt

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in D4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _normal(cells):
    r0, _, c0, _ = _bounds(cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _orientations(cells, reflect=True):
    values = set()
    for flip in range(2 if reflect else 1):
        shape = [(r, -c if flip else c) for r, c in cells]
        for _ in range(4):
            values.add(_normal(shape))
            shape = [(c, -r) for r, c in shape]
    return tuple(sorted(values))


def _canonical(cells, reflect=True):
    return _orientations(cells, reflect)[0]


def _mask(cells):
    cells = _normal(cells)
    a = np.zeros((max(r for r, c in cells)+1,
                  max(c for r, c in cells)+1), dtype=bool)
    for r, c in cells:
        a[r, c] = True
    return a


def _expand(a, scale):
    return np.repeat(np.repeat(a, scale, axis=0), scale, axis=1)


def _distance(mask, metric="chebyshev"):
    """Exact distance to a silhouette, with no centroid/bounding-box shortcut."""
    rows, cols = np.indices(mask.shape)
    points = np.argwhere(mask)
    dr = np.abs(rows[:, :, None] - points[:, 0])
    dc = np.abs(cols[:, :, None] - points[:, 1])
    values = dr + dc if metric == "manhattan" else np.maximum(dr, dc)
    return values.min(axis=2).astype(np.int16)


def _cast(board_shape, scale, holes, colors, period, frame,
          metric="chebyshev", boxes=False, ignore_phase=False,
          all_ties=False):
    shape = tuple(n*scale for n in board_shape)
    masks = []
    for hole in holes:
        mask = np.zeros(board_shape, dtype=bool)
        for r, c in hole:
            mask[r, c] = True
        if boxes:
            r0, r1, c0, c1 = _bounds(hole)
            mask[r0:r1, c0:c1] = True
        masks.append(_expand(mask, scale))
    distances = np.stack([_distance(m, metric) for m in masks])
    nearest = distances.min(axis=0)
    close = distances == nearest
    phases = [0 if ignore_phase else period.index(int(c)) for c in colors]
    palette = np.asarray(period, dtype=np.int8)
    proposals = palette[(nearest[None, :, :] + np.asarray(phases)[:, None, None]) % len(period)]
    low = np.where(close, proposals, 10).min(axis=0)
    high = np.where(close, proposals, -1).max(axis=0)
    conflict = low != high
    if all_ties:
        conflict = close.sum(axis=0) > 1
    output = np.where(conflict, frame, low).astype(np.int8)
    interior = np.zeros(shape, dtype=bool)
    interior[scale:-scale, scale:-scale] = True
    output[~interior] = frame
    return output, {"masks": masks, "distances": distances, "nearest": nearest,
                    "close": close, "conflict": conflict, "interior": interior,
                    "same_phase_ties": (close.sum(axis=0) > 1) & ~conflict & interior}


def _decode(input_grid):
    a = _grid(input_grid)
    background = int(a[0, 0])
    if not (np.all(a[0] == background) and np.all(a[-1] == background)
            and np.all(a[:, 0] == background) and np.all(a[:, -1] == background)):
        raise ValueError("the canvas must have a uniform empty margin")
    solutions = []
    for component in _components(a != background):
        r0, r1, c0, c1 = bounds = _bounds(component)
        board = a[r0:r1, c0:c1]
        if min(board.shape) < 6 or len(np.unique(board)) < 5:
            continue
        frame = int(board[0, 0])
        if not (np.all(board[0] == frame) and np.all(board[-1] == frame)
                and np.all(board[:, 0] == frame) and np.all(board[:, -1] == frame)):
            continue
        interior = board[1:-1, 1:-1]
        material_counts = Counter(int(v) for v in interior.ravel()
                                  if v not in (background, frame))
        if not material_counts:
            continue
        material, count = material_counts.most_common(1)[0]
        if count <= 3 or sum(n == count for n in material_counts.values()) != 1:
            continue
        if np.any(interior == frame):
            continue
        cues = set(map(tuple, np.argwhere((board != background) & (board != frame)
                                         & (board != material)).tolist()))
        if len(cues) not in (2, 3):
            continue
        periods = []
        h, w = board.shape
        for r, c, dr, dc in ((1, 1, 1, 1), (1, w-2, 1, -1),
                             (h-2, 1, -1, 1), (h-2, w-2, -1, -1)):
            path = [(r+d*dr, c+d*dc) for d in range(len(cues))]
            if set(path) == cues:
                p = tuple(int(board[x, y]) for x, y in path)
                if len(set(p)) == len(p):
                    periods.append(p)
        if len(periods) != 1:
            continue
        period = periods[0]
        holes = _components(board == background)
        if not 2 <= len(holes) <= 5:
            continue
        outside = a != background
        outside[r0:r1, c0:c1] = False
        external = _components(outside)
        if len(external) != len(holes):
            continue
        total_area = sum(map(len, external))
        logical_area = sum(map(len, holes))
        if total_area % logical_area:
            continue
        scale = isqrt(total_area // logical_area)
        if (scale < 2 or total_area != logical_area*scale*scale
                or max(board.shape)*scale > 30):
            continue
        pieces, good = [], True
        for cells in external:
            piece_colors = {int(a[r, c]) for r, c in cells}
            pr0, pr1, pc0, pc1 = _bounds(cells)
            if (len(piece_colors) != 1 or not piece_colors <= set(period)
                    or (pr1-pr0) % scale or (pc1-pc0) % scale):
                good = False
                break
            blocks = Counter(((r-pr0)//scale, (c-pc0)//scale) for r, c in cells)
            if any(n != scale*scale for n in blocks.values()):
                good = False
                break
            pieces.append({"shape": tuple(sorted(blocks)), "color": next(iter(piece_colors)),
                           "cells": cells})
        if not good:
            continue
        hg, pg = {}, {}
        for cells in holes:
            hg.setdefault(_canonical(cells), []).append(cells)
        for piece in pieces:
            pg.setdefault(_canonical(piece["shape"]), []).append(piece)
        if set(hg) != set(pg) or any(len(hg[k]) != len(pg[k]) for k in hg):
            continue
        if any(len({p["color"] for p in group}) != 1 for group in pg.values()):
            continue
        colors = [pg[_canonical(hole)][0]["color"] for hole in holes]
        output, field = _cast(board.shape, scale, holes, colors, period, frame)
        destination_rot = Counter((_canonical(hole, False), color)
                                  for hole, color in zip(holes, colors))
        piece_rot = Counter((_canonical(p["shape"], False), p["color"]) for p in pieces)
        identity = Counter((_normal(hole), color) for hole, color in zip(holes, colors))
        loose_identity = Counter((p["shape"], p["color"]) for p in pieces)
        info = {"background": background, "frame": frame, "material": material,
                "period": period, "scale": scale, "board_shape": board.shape,
                "bounds": bounds, "holes": holes, "colors": colors, "pieces": pieces,
                "field": field, "classes": tuple(sorted(hg)),
                "reflection_required": destination_rot != piece_rot,
                "orientation_required": identity != loose_identity}
        solutions.append((output, info))
    if len(solutions) != 1:
        raise ValueError("input must determine one blueprint, scale and colored assembly")
    return solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every parameter from this input; training and seed are unnecessary."""
    return _decode(input_grid)[0]


def _grow(rng, area, span=4):
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in D4} - cells)
        feasible = []
        for p in frontier:
            r0, r1, c0, c1 = _bounds(cells | {p})
            if r1-r0 <= span and c1-c0 <= span:
                feasible.append(p)
        cells.add(feasible[int(rng.integers(len(feasible)))])
    return _normal(cells)


def _shapes(rng, count, witness, scale):
    span = 3 if scale == 3 else 4
    if witness:
        # Equal area with different geometry prevents area-only matching.
        for _ in range(160):
            area = int(rng.integers(4 if witness == "reflection" else 3, 7))
            first, second = _grow(rng, area, span), _grow(rng, area, span)
            if _canonical(first) == _canonical(second):
                continue
            if len(_orientations(first, False)) < 2:
                continue
            if witness == "reflection" and len(_orientations(first)) < 8:
                continue
            return [first, second] + ([first] if count == 3 else [])
        # Feasible finite fallback, still with the required geometric witnesses.
        if witness == "reflection":
            first = ((0, 0), (1, 0), (2, 0), (2, 1))
            second = ((0, 0), (0, 1), (0, 2), (1, 1))
            return [first, second, first]
        return [((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (1, 1))]
    result = []
    for _ in range(count):
        if result and rng.random() < .22:
            result.append(result[int(rng.integers(len(result)))])
        else:
            result.append(_grow(rng, int(rng.integers(1, 8)), span))
    return result


def _place(rng, shapes, shape, key_positions):
    h, w = shape
    for _ in range(12):
        occupied = set()
        holes = []
        for base in shapes:
            options = _orientations(base, False)
            base = options[int(rng.integers(len(options)))]
            sh, sw = _mask(base).shape
            candidates = [(r, c) for r in range(1, h-sh) for c in range(1, w-sw)]
            placed = None
            for j in rng.permutation(len(candidates)):
                r0, c0 = candidates[int(j)]
                cells = {(r+r0, c+c0) for r, c in base}
                if any(p in occupied or p in key_positions
                       or any((p[0]+dr, p[1]+dc) in occupied for dr, dc in D4)
                       for p in cells):
                    continue
                placed = tuple(sorted(cells))
                occupied.update(cells)
                break
            if placed is None:
                break
            holes.append(placed)
        else:
            return holes
    return None


def _origins(blocked, h, w):
    if h > blocked.shape[0]-2 or w > blocked.shape[1]-2:
        return []
    integral = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    valid = totals == 0
    valid[0, :] = valid[-1, :] = False
    valid[:, 0] = valid[:, -1] = False
    return np.argwhere(valid)


def _pack(rng, board, pieces, colors, scale, background):
    objects = [board] + [np.where(_expand(_mask(p), scale), color, background).astype(np.int8)
                         for p, color in zip(pieces, colors)]
    for attempt in range(36):
        h, w = (map(int, rng.integers(23, 31, size=2)) if attempt < 18 else (30, 30))
        canvas = np.full((h, w), background, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, float(rng.random())))
        for i in order:
            obj = objects[i]
            oh, ow = obj.shape
            positions = _origins(blocked, oh, ow)
            if not len(positions):
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            canvas[r:r+oh, c:c+ow] = obj
            blocked[r-1:r+oh+1, c-1:c+ow+1] = True
        else:
            return canvas
    return None


def _witnesses(output, info):
    h = info["holes"]
    c = info["colors"]
    b, k, p, f = info["board_shape"], info["scale"], info["period"], info["frame"]
    field = info["field"]
    variants = {
        "manhattan": _cast(b, k, h, c, p, f, metric="manhattan")[0],
        "bounding_boxes": _cast(b, k, h, c, p, f, boxes=True)[0],
        "ignore_phase": _cast(b, k, h, c, p, f, ignore_phase=True)[0],
        "coarse_then_expand": _expand(_cast(b, 1, h, c, p, f)[0], k),
    }
    result = {name: bool(np.any(value != output)) for name, value in variants.items()}
    result["conflicting_tie"] = bool(np.any(field["conflict"] & field["interior"]))
    result["agreeing_tie"] = bool(np.any(field["same_phase_ties"]))
    result["repeat_cycle"] = bool(np.any(field["nearest"][field["interior"]] >= len(p)))
    result["equal_area_different_color"] = any(
        len(x) == len(y) and _canonical(x) != _canonical(y) and cx != cy
        for i, (x, cx) in enumerate(zip(h, c)) for y, cy in zip(h[i+1:], c[i+1:]))
    return result


def _make_pair(rng, index, test, forbidden_classes=()):
    witness = None if test or index >= 2 else ("rotation" if index == 0 else "reflection")
    for _attempt in range(120):
        count = (2 if index == 0 else 3) if witness else int(rng.integers(2, 6 if test else 5))
        if test and index == 0:
            count = int(rng.integers(4, 6))
        scale = (2 if index == 0 else 3) if witness else (2 if count == 5 else int(rng.choice([2, 3])))
        length = (2 if index == 0 else 3) if witness else int(rng.choice([2, 3]))
        palette = list(map(int, rng.permutation(10)))
        background, frame, material = palette[:3]
        period = tuple(palette[3:3+length])
        shapes = _shapes(rng, count, witness, scale)
        if test and all(_canonical(s) in forbidden_classes for s in shapes):
            continue
        limit = 30//scale
        low = 9 if witness == "reflection" else (9 if count >= 4 else 7)
        bh, bw = map(int, rng.integers(low, limit+1, size=2))
        corner = int(rng.integers(4))
        r, c, dr, dc = ((1, 1, 1, 1), (1, bw-2, 1, -1),
                         (bh-2, 1, -1, 1), (bh-2, bw-2, -1, -1))[corner]
        key_positions = [(r+d*dr, c+d*dc) for d in range(length)]
        holes = _place(rng, shapes, (bh, bw), set(key_positions))
        if holes is None:
            continue
        classes = list(dict.fromkeys(_canonical(s) for s in shapes))
        if len(classes) < 2:
            continue
        assignment = {key: period[j % length] if j < 2 else int(rng.choice(period))
                      for j, key in enumerate(classes)}
        colors = [assignment[_canonical(s)] for s in shapes]
        board = np.full((bh, bw), material, dtype=np.int8)
        board[0] = board[-1] = frame
        board[:, 0] = board[:, -1] = frame
        for hole in holes:
            for x, y in hole:
                board[x, y] = background
        for (x, y), color in zip(key_positions, period):
            board[x, y] = color
        pieces = []
        for j, hole in enumerate(holes):
            options = _orientations(hole)
            if witness == "rotation":
                options = _orientations(hole, False)
                if j == 0:
                    options = tuple(s for s in options if s != _normal(hole))
            elif witness == "reflection" and _canonical(hole) == _canonical(holes[0]):
                rotations = _orientations(hole, False)
                options = tuple(s for s in options if s not in rotations)
            pieces.append(options[int(rng.integers(len(options)))])
        output, field = _cast(board.shape, scale, holes, colors, period, frame)
        if witness:
            info = {"holes": holes, "colors": colors, "board_shape": board.shape,
                    "scale": scale, "period": period, "frame": frame, "field": field}
            evidence = _witnesses(output, info)
            required = ("manhattan", "bounding_boxes", "ignore_phase", "coarse_then_expand",
                        "conflicting_tie", "repeat_cycle", "equal_area_different_color")
            if not all(evidence[name] for name in required):
                continue
            if witness == "reflection" and not evidence["agreeing_tie"]:
                continue
        canvas = _pack(rng, board, pieces, colors, scale, background)
        if canvas is None:
            continue
        return {"input": canvas, "output": output}
    raise RuntimeError("bounded construction exhausted")


def _valid_count(value, lo, hi):
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
            and lo <= value <= hi)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _valid_count(num_train, 2, 10) or not _valid_count(num_test, 1, 4) or num_train <= num_test:
        raise ValueError("require 2..10 training, 1..4 test, and num_train > num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen, train_classes = {"train": [], "test": []}, set(), set()
    previous_roles = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(24):
                pair = _make_pair(rng, index, split == "test", train_classes)
                _, info = _decode(pair["input"])
                roles = info["background"], info["frame"], info["material"]
                if _key(pair["input"]) in seen:
                    continue
                # Local role changes are demonstrated even with just two trains.
                if previous_roles is not None and any(x == y for x, y in zip(roles, previous_roles)):
                    continue
                break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            previous_roles = roles
            if split == "train":
                train_classes.update(info["classes"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert _valid_count(len(puzzle["train"]), 2, 10)
    assert _valid_count(len(puzzle["test"]), 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            predicted, info = _decode(inp)
            assert np.array_equal(predicted, out), "incorrect phase-cast output"
            k, field = info["scale"], info["field"]
            assert out.shape == tuple(k*n for n in info["board_shape"])
            assert set(map(int, np.unique(out))) <= set(info["period"]) | {info["frame"]}
            assert np.all(out[~field["interior"]] == info["frame"])
            assert np.all((out == info["frame"])[field["interior"]]
                          == field["conflict"][field["interior"]])
            for mask, color in zip(field["masks"], info["colors"]):
                assert np.all(out[mask] == color), "fitted sources must keep their colors"
            infos[split].append(info)
    # Compulsory demonstrations cover both metrics, both tie outcomes, cycles,
    # physical-pixel band width, and shape/phase dependence.
    evidence = [_witnesses(np.asarray(p["output"]), i)
                for p, i in zip(puzzle["train"][:2], infos["train"][:2])]
    required = ("manhattan", "bounding_boxes", "ignore_phase", "coarse_then_expand",
                "conflicting_tie", "repeat_cycle", "equal_area_different_color")
    assert all(all(e[name] for name in required) for e in evidence)
    assert evidence[1]["agreeing_tie"]
    assert infos["train"][0]["orientation_required"]
    assert not infos["train"][0]["reflection_required"]
    assert infos["train"][1]["reflection_required"]
    assert {len(i["period"]) for i in infos["train"]} == {2, 3}
    assert {i["scale"] for i in infos["train"]} == {2, 3}
    training_classes = set(c for i in infos["train"] for c in i["classes"])
    assert all(set(i["classes"]) - training_classes for i in infos["test"])
    assert len(infos["test"][0]["holes"]) >= 4


def _bfs_distance(mask):
    """Different distance implementation for bounded semantic consistency checks."""
    distance = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque(map(tuple, np.argwhere(mask).tolist()))
    distance[mask] = 0
    while queue:
        r, c = queue.popleft()
        for dr, dc in D8:
            rr, cc = r+dr, c+dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and distance[rr, cc] < 0:
                distance[rr, cc] = distance[r, c]+1
                queue.append((rr, cc))
    return distance


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, streamed diagnostics, no filesystem or retained grid archive."""
    signatures, classes, inputs, outputs, palettes, geometry = (set() for _ in range(6))
    counts, scales, periods, corners, areas = (set() for _ in range(5))
    matching_changes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                out, info = _decode(pair["input"])
                assert np.array_equal(out, pair["output"])
                classes.update(info["classes"])
                inputs.add(pair["input"].shape)
                outputs.add(out.shape)
                palettes.add((info["background"], info["frame"], info["material"], info["period"]))
                counts.add(len(info["holes"]))
                scales.add(info["scale"])
                periods.add(len(info["period"]))
                areas.update(map(len, info["holes"]))
                geometry.add(hashlib.sha256(repr((info["board_shape"], info["holes"])).encode()).hexdigest())
                r0, r1, c0, c1 = info["bounds"]
                board = pair["input"][r0:r1, c0:c1]
                corners.add(next(j for j, (r, c) in enumerate(((1, 1), (1, board.shape[1]-2),
                    (board.shape[0]-2, 1), (board.shape[0]-2, board.shape[1]-2)))
                                 if int(board[r, c]) == info["period"][0]))
                for mask, distance in zip(info["field"]["masks"], info["field"]["distances"]):
                    assert np.array_equal(distance, _bfs_distance(mask))
        # Explicitly omitting matching changes the two equal-area source colors.
        out, info = _decode(puzzle["train"][0]["input"])
        colors = info["colors"][::-1]
        wrong = _cast(info["board_shape"], info["scale"], info["holes"], colors,
                      info["period"], info["frame"])[0]
        assert np.any(out != wrong)
        matching_changes += 1
    assert len(signatures) == 42 and matching_changes == 42
    assert len(classes) >= 40 and len(inputs) >= 20 and len(outputs) >= 25
    assert len(palettes) >= 150 and len(geometry) >= 180
    assert counts == {2, 3, 4, 5} and scales == {2, 3} and periods == {2, 3}
    assert corners == {0, 1, 2, 3} and areas >= set(range(1, 8))
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(391, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(13, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted frame accepted")
    # Differently colored congruent pieces must not be given an arbitrary order.
    p = generate_puzzle(17, 2, 1)
    inp = p["train"][1]["input"].copy()
    _, info = _decode(inp)
    groups = {}
    for piece in info["pieces"]:
        groups.setdefault(_canonical(piece["shape"]), []).append(piece)
    duplicate = next(group for group in groups.values() if len(group) > 1)[0]
    new_color = next(c for c in info["period"] if c != duplicate["color"])
    for r, c in duplicate["cells"]:
        inp[r, c] = new_color
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous colored assembly accepted")
    return {"seeds": 42, "pairs": 210, "shape_classes": len(classes),
            "input_shapes": len(inputs), "output_shapes": len(outputs),
            "geometries": len(geometry), "palettes": len(palettes),
            "source_counts": sorted(counts), "source_areas": sorted(areas),
            "scales": sorted(scales), "periods": sorted(periods),
            "key_corners": len(corners), "matching_omission_changes": matching_changes}
