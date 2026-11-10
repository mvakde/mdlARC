"""Docked repair pieces: transport pixel defects through scaled D4 assembly.

No parent code, saved puzzles, seed metadata, or training labels are used by
solve. Five local color roles are recoverable from each individual input.
"""

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _transform(array, reflection, turns):
    return np.rot90(np.fliplr(array) if reflection else array, turns).copy()


def _transforms(array):
    return [_transform(array, f, q) for f in (False, True) for q in range(4)]


def _canonical(array):
    return min(_key(a) for a in _transforms(array))


def _expand(array, scale):
    return np.repeat(np.repeat(array, scale, 0), scale, 1)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _decode_piece(grid, cells, background, dock, fills):
    r0, r1, c0, c1 = _bbox(cells)
    raster = grid[r0:r1, c0:c1].copy()
    # Objects have separated bounding boxes; nothing else is inside this crop.
    roots = np.argwhere(raster == dock).tolist()
    assert roots, "large loose object lacks a dock"
    a, b, c, d = _bbox(roots)
    scale = b - a
    assert scale >= 2 and d - c == scale
    assert len(roots) == scale * scale, "dock must be a full square"
    assert a % scale == c % scale == 0
    assert raster.shape[0] % scale == raster.shape[1] % scale == 0
    observed = raster[(raster != background) & (raster != dock)]
    assert observed.size
    counts = [int(np.count_nonzero(observed == f)) for f in fills]
    assert counts[0] != counts[1], "piece fill is tied"
    fill = fills[int(counts[1] > counts[0])]
    logical = np.zeros((raster.shape[0] // scale, raster.shape[1] // scale), dtype=np.int8)
    for r in range(logical.shape[0]):
        for col in range(logical.shape[1]):
            block = raster[r*scale:(r+1)*scale, col*scale:(col+1)*scale]
            if np.all(block == background):
                continue
            if np.all(block == dock):
                logical[r, col] = 2
            else:
                assert np.count_nonzero(block == fill) > block.size / 2, "damaged macrocell lacks majority"
                assert not np.any(block == dock)
                assert np.all(np.isin(block, (background, *fills)))
                logical[r, col] = 1
    assert np.count_nonzero(logical == 2) == 1
    assert len(_components(logical != 0)) == 1
    assert len({_key(a) for a in _transforms(logical)}) == 8, "dock does not resolve orientation"
    assert all(np.any(edge) for edge in (logical[0], logical[-1], logical[:, 0], logical[:, -1]))
    expected = np.where(_expand(logical != 0, scale), fill, background).astype(np.int8)
    expected[_expand(logical == 2, scale)] = dock
    erasures = (expected == fill) & (raster == background)
    wrong = (expected == fill) & (raster != background) & (raster != fill)
    assert np.all(raster[expected == background] == background)
    return {"logical": logical, "raster": raster, "scale": scale, "fill": fill,
            "erasures": erasures, "wrong": wrong, "bounds": (r0, r1, c0, c1)}


def _render(clean, centers, fills, structural):
    """All rings are simultaneous constraints; all centers have precedence."""
    center_set = set(centers)
    demands = {}
    for r, c in centers:
        assert clean[r, c] in fills
        opposite = fills[1] if clean[r, c] == fills[0] else fills[0]
        for rr in range(max(0, r-1), min(clean.shape[0], r+2)):
            for cc in range(max(0, c-1), min(clean.shape[1], c+2)):
                if (rr, cc) in center_set:
                    continue
                assert (rr, cc) not in demands or demands[rr, cc] == opposite, "incompatible rings"
                demands[rr, cc] = opposite
    out = clean.copy()
    for p, color in demands.items():
        out[p] = color
    for p in centers:
        out[p] = structural
    return out


def _recover(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > grid.size / 2, "exterior background lacks majority"
    colors = set(map(int, np.unique(grid)))
    assert len(colors) == 5, "five visible semantic roles are required"
    solutions = []
    for structural in sorted(colors - {background}):
        r0, r1, c0, c1 = _bbox(np.argwhere(grid == structural).tolist())
        board = grid[r0:r1, c0:c1]
        if min(board.shape) < 4:
            continue
        if not all(np.all(edge == structural) for edge in (board[0], board[-1], board[:, 0], board[:, -1])):
            continue
        extras = set(map(int, np.unique(board))) - {background, structural}
        if len(extras) != 1:
            continue
        dock = next(iter(extras))
        fills = tuple(sorted(colors - {background, structural, dock}))
        try:
            holes = []
            for cells in _components(board != structural):
                a, b, c, d = _bbox(cells)
                pattern = np.zeros((b-a, d-c), dtype=np.int8)
                for r, col in cells:
                    pattern[r-a, col-c] = 2 if board[r, col] == dock else 1
                assert np.count_nonzero(pattern == 2) == 1
                assert len({_key(x) for x in _transforms(pattern)}) == 8
                holes.append({"logical": pattern, "origin": (a, c), "cells": cells})
            assert len(holes) >= 2
            assert len({_canonical(h["logical"]) for h in holes}) == len(holes), "interchangeable docks"
            outside = grid.copy()
            outside[r0:r1, c0:c1] = background
            moat = outside[max(0, r0-1):min(grid.shape[0], r1+1), max(0, c0-1):min(grid.shape[1], c1+1)]
            assert np.all(moat == background)
            components = _components(outside != background)
            pieces = [_decode_piece(outside, cells, background, dock, fills)
                      for cells in components if len(cells) >= 4]
            noise = [cells for cells in components if len(cells) < 4]
            assert len(pieces) == len(holes)
            assert {p["fill"] for p in pieces} == set(fills)
            scales = {p["scale"] for p in pieces}
            assert len(scales) == 1
            scale = next(iter(scales))
            assert max(board.shape) * scale <= 30
            clean = np.full((board.shape[0]*scale, board.shape[1]*scale), structural, dtype=np.int8)
            centers, wrong_points, assignments = [], [], []
            used = set()
            for piece in pieces:
                matches = [(i, f, q) for i, hole in enumerate(holes)
                           for f in (False, True) for q in range(4)
                           if np.array_equal(_transform(piece["logical"], f, q), hole["logical"])]
                assert len(matches) == 1, "dock matching or orientation is ambiguous"
                i, f, q = matches[0]
                assert i not in used
                used.add(i)
                hole = holes[i]
                a, c = hole["origin"]
                mask = _expand(hole["logical"] != 0, scale)
                view = clean[a*scale:a*scale+mask.shape[0], c*scale:c*scale+mask.shape[1]]
                view[mask] = piece["fill"]  # Dock is registration ink, then removed.
                erasures = _transform(piece["erasures"], f, q)
                wrong = _transform(piece["wrong"], f, q)
                centers.extend((a*scale+r, c*scale+col) for r, col in np.argwhere(erasures).tolist())
                wrong_points.extend((a*scale+r, c*scale+col) for r, col in np.argwhere(wrong).tolist())
                assignments.append({"hole": i, "reflection": f, "turns": q, "piece": piece})
            assert len(used) == len(holes) and centers
            out = _render(clean, centers, fills, structural)
            info = {"background": background, "structural": structural, "dock": dock,
                    "fills": fills, "scale": scale, "board": board.copy(),
                    "board_bounds": (r0, r1, c0, c1), "holes": holes, "pieces": pieces,
                    "assignments": assignments, "clean": clean, "centers": centers,
                    "wrong_points": wrong_points, "noise": noise}
            solutions.append((out, info))
        except AssertionError:
            continue
    assert len(solutions) == 1, "expected one complete input interpretation"
    return solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the answer from just this input; no puzzle-wide latent choices."""
    return _recover(input_grid)[0]


def _shape(rng, max_area, limit=4):
    for _ in range(160):
        cells = {(0, 0)}
        area = int(rng.integers(3, max_area+1))
        style = int(rng.integers(3))
        while len(cells) < area:
            options = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in N4} - cells)
            options = [p for p in options if max(r for r, c in cells | {p}) - min(r for r, c in cells | {p}) < limit
                       and max(c for r, c in cells | {p}) - min(c for r, c in cells | {p}) < limit]
            if not options:
                break
            weights = []
            for r, c in options:
                neighbors = sum((r+dr, c+dc) in cells for dr, dc in N4)
                weights.append(3**neighbors if style == 0 else (1/neighbors if style == 1 else 1))
            cells.add(options[int(rng.choice(len(options), p=np.array(weights)/sum(weights)))])
        a, b, c, d = _bbox(list(cells))
        base = np.zeros((b-a, d-c), dtype=np.int8)
        for r, col in cells:
            base[r-a, col-c] = 1
        rooted = []
        for r, col in np.argwhere(base).tolist():
            candidate = base.copy()
            candidate[r, col] = 2
            if len({_key(x) for x in _transforms(candidate)}) == 8:
                rooted.append(candidate)
        if rooted:
            return rooted[int(rng.integers(len(rooted)))]
    raise RuntimeError("could not grow a dock-resolved shape")


def _place(rng, shapes, height, width, contact):
    occupied = set()
    placements = []
    for i, shape in enumerate(shapes):
        cells = np.argwhere(shape != 0).tolist()
        candidates = [(r, c) for r in range(1, height-shape.shape[0])
                      for c in range(1, width-shape.shape[1])]
        rng.shuffle(candidates)
        for a, c in candidates:
            translated = [(a+r, c+col) for r, col in cells]
            if any(p in occupied or any((p[0]+dr, p[1]+dc) in occupied for dr, dc in N4) for p in translated):
                continue
            if contact and i == 1:
                first_shape, (aa, cc) = placements[0]
                first_cells = [(aa+r, cc+col) for r, col in np.argwhere(first_shape == 1).tolist()]
                second_cells = [(a+r, c+col) for r, col in np.argwhere(shape == 1).tolist()]
                if not any(abs(p[0]-q[0]) == abs(p[1]-q[1]) == 1 for p in first_cells for q in second_cells):
                    continue
            occupied.update(translated)
            placements.append((shape, (a, c)))
            break
        else:
            return None
    return placements


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, board, donors, background, noise_colors):
    objects = [board] + donors
    for attempt in range(55):
        shape = (int(rng.integers(20, 31)), int(rng.integers(20, 31))) if attempt < 35 else (30, 30)
        canvas = np.full(shape, background, dtype=np.int8)
        blocked = np.zeros(shape, dtype=bool)
        # Random two-dimensional packing. The board has no privileged shelf.
        order = sorted(range(len(objects)), key=lambda i: -objects[i].size + float(rng.random())*20)
        for i in order:
            obj = objects[i]
            h, w = obj.shape
            origins = _free_origins(blocked, h, w)
            if not len(origins):
                break
            r, c = origins[int(rng.integers(len(origins)))].tolist()
            canvas[r:r+h, c:c+w] = obj
            blocked[max(0, r-1):min(shape[0], r+h+1), max(0, c-1):min(shape[1], c+w+1)] = True
        else:
            for _ in range(int(rng.integers(0, 9))):
                options = _free_origins(blocked, 2, 2)
                if not len(options):
                    break
                r, c = options[int(rng.integers(len(options)))].tolist()
                size = int(rng.integers(1, 4))
                for dr, dc in ((0, 0), (0, 1), (1, 0))[:size]:
                    canvas[r+dr, c+dc] = int(rng.choice(noise_colors))
                blocked[max(0, r-1):min(shape[0], r+3), max(0, c-1):min(shape[1], c+3)] = True
            if np.count_nonzero(canvas == background) <= canvas.size / 2:
                continue
            if rng.random() < .6:
                a, b, c, d = _bbox(np.argwhere(canvas != background).tolist())
                a, c = max(0, a-int(rng.integers(4))), max(0, c-int(rng.integers(4)))
                b, d = min(shape[0], b+int(rng.integers(4))), min(shape[1], d+int(rng.integers(4)))
                crop = canvas[a:b, c:d]
                if np.count_nonzero(crop == background) > crop.size/2:
                    canvas = crop.copy()
            return canvas
    return None


def _make_pair(rng, scale, kind, palette, excluded_shapes, reserved_shape=None):
    background, structural, dock, f0, f1 = palette
    fills = (f0, f1)
    contact = kind == "contact"
    for attempt in range(240):
        limit = 30 // scale
        low = 7 if scale < 4 else 6
        bh, bw = int(rng.integers(low, limit+1)), int(rng.integers(low, limit+1))
        max_area = {2: 10, 3: 7, 4: 5}[scale]
        count = 3 if contact else (2 if kind == "docks" or scale == 4 else int(rng.integers(2, 6)))
        shapes = [_shape(rng, max_area, min(4, bh-2, bw-2)) for _ in range(count)]
        if reserved_shape is not None:
            shapes[0] = reserved_shape.copy()
        if kind == "docks":
            # Equal silhouettes and equal areas, but different dock positions.
            base = shapes[0] != 0
            alternatives = []
            for r, c in np.argwhere(base).tolist():
                s = base.astype(np.int8)
                s[r, c] = 2
                if len({_key(a) for a in _transforms(s)}) == 8 and _canonical(s) != _canonical(shapes[0]):
                    alternatives.append(s)
            if not alternatives:
                continue
            shapes[1] = alternatives[int(rng.integers(len(alternatives)))]
        if len({_canonical(s) for s in shapes}) != count:
            continue
        if kind != "test" and any(_canonical(s != 0) in excluded_shapes for s in shapes):
            continue
        if kind == "test" and all(_canonical(s != 0) in excluded_shapes for s in shapes):
            continue
        # Logical cavity orientations, independently sampled from loose ones.
        shapes = [_transform(s, bool(rng.integers(2)), int(rng.integers(4))) for s in shapes]
        placements = _place(rng, shapes, bh, bw, contact)
        if placements is None:
            continue
        colors = [f0, f0, f1] if contact else [f0, f1] + [int(rng.choice(fills)) for _ in range(count-2)]
        board = np.full((bh, bw), structural, dtype=np.int8)
        clean = np.full((bh*scale, bw*scale), structural, dtype=np.int8)
        rasters, seeds_of_holes, wrongs = [], [[] for _ in shapes], [[] for _ in shapes]
        for i, (s, (a, c)) in enumerate(placements):
            view = board[a:a+s.shape[0], c:c+s.shape[1]]
            view[s == 1] = background
            view[s == 2] = dock
            mask = _expand(s != 0, scale)
            clean[a*scale:a*scale+mask.shape[0], c*scale:c*scale+mask.shape[1]][mask] = colors[i]
            raster = np.where(mask, colors[i], background).astype(np.int8)
            raster[_expand(s == 2, scale)] = dock
            rasters.append(raster)
        def add(i, p, erasure=True):
            r, c = p
            if rasters[i][r, c] != colors[i]:
                return False
            sr, sc = r // scale, c // scale
            block = rasters[i][sr*scale:(sr+1)*scale, sc*scale:(sc+1)*scale]
            if np.count_nonzero(block == colors[i]) - 1 <= block.size / 2:
                return False
            a, b = placements[i][1]
            dest = (a*scale+r, b*scale+c)
            all_centers = [(placements[j][1][0]*scale+rr, placements[j][1][1]*scale+cc, colors[j])
                           for j in range(count) for rr, cc in seeds_of_holes[j]]
            if erasure and any(color != colors[i] and max(abs(dest[0]-rr), abs(dest[1]-cc)) <= 2
                               for rr, cc, color in all_centers):
                return False
            if not erasure and any(max(abs(dest[0]-rr), abs(dest[1]-cc)) <= 1 for rr, cc, color in all_centers):
                return False
            rasters[i][p] = background if erasure else fills[1] if colors[i] == fills[0] else fills[0]
            (seeds_of_holes if erasure else wrongs)[i].append(p)
            return True
        if contact:
            s0, (a0, c0) = placements[0]
            s1, (a1, c1) = placements[1]
            contacts = [(p, q) for p in np.argwhere(s0 == 1).tolist() for q in np.argwhere(s1 == 1).tolist()
                        if abs(a0+p[0]-a1-q[0]) == abs(c0+p[1]-c1-q[1]) == 1]
            p, q = contacts[int(rng.integers(len(contacts)))]
            dr, dc = a1+q[0]-a0-p[0], c1+q[1]-c0-p[1]
            pp = (p[0]*scale + (scale-1 if dr > 0 else 0), p[1]*scale + (scale-1 if dc > 0 else 0))
            qq = (q[0]*scale + (0 if dr > 0 else scale-1), q[1]*scale + (0 if dc > 0 else scale-1))
            assert add(0, pp) and add(1, qq)
        for i, raster in enumerate(rasters):
            candidates = [tuple(p) for p in np.argwhere(raster == colors[i]).tolist()]
            rng.shuffle(candidates)
            target = int(rng.integers(1, 4 if kind != "test" else 7))
            if kind == "random" and i > 1 and rng.random() < .35:
                target = 0
            for p in candidates:
                if len(seeds_of_holes[i]) >= target:
                    break
                add(i, p)
        if not all(seeds_of_holes[i] for i in range(count) if kind in ("contact", "docks")):
            continue
        # Ordinary wrong-fill damage is repaired without creating a symbol.
        desired_wrong = 1 if kind in ("contact", "docks") else int(rng.integers(0, 5))
        candidates = [(i, tuple(p)) for i, raster in enumerate(rasters) for p in np.argwhere(raster == colors[i]).tolist()]
        rng.shuffle(candidates)
        for i, p in candidates:
            if sum(map(len, wrongs)) >= desired_wrong:
                break
            add(i, p, False)
        if kind in ("contact", "docks") and not any(wrongs):
            continue
        if any(len(_components(raster != background)) != 1 for raster in rasters):
            continue
        donors = []
        for i, raster in enumerate(rasters):
            reflection = True if contact and i == 0 else bool(rng.integers(2))
            turns = int(rng.integers(1, 4)) if kind == "docks" and i == 0 else int(rng.integers(4))
            donors.append(_transform(raster, reflection, turns))
        canvas = _pack(rng, board, donors, background, (dock, *fills))
        if canvas is None:
            continue
        centers = [(placements[i][1][0]*scale+r, placements[i][1][1]*scale+c)
                   for i in range(count) for r, c in seeds_of_holes[i]]
        try:
            output = _render(clean, centers, fills, structural)
            predicted, info = _recover(canvas)
            assert np.array_equal(output, predicted), "construction and recovery disagree"
            if contact and not _features(info)["transport_moved"]:
                continue
        except AssertionError:
            continue
        return {"input": canvas, "output": output}, info
    raise RuntimeError("bounded docked-piece construction exhausted")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(4))
    train_scales = ([2]*num_train if mode == 0 else [3]*num_train if mode == 1
                    else [2, 3] + [int(rng.choice([2, 3])) for _ in range(num_train-2)])
    test_scales = ([3]*num_test if mode == 0 else [2]*num_test if mode == 1
                   else [4]*num_test if mode == 2 else [int(rng.choice([2, 3])) for _ in range(num_test)])
    # Scale four has a small feasible vocabulary. Reserve query silhouettes
    # before training so even ten demonstrations cannot consume that vocabulary.
    reserved = [_shape(rng, 5, 3) for _ in range(num_test)] if mode == 2 else [None]*num_test
    reserved_keys = {_canonical(s != 0) for s in reserved if s is not None}
    base_palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    changing_palette = bool(rng.integers(2))
    puzzle, seen, training_shapes = {"train": [], "test": []}, set(), set()
    for split, scales in (("train", train_scales), ("test", test_scales)):
        for i, scale in enumerate(scales):
            palette = tuple(map(int, rng.choice(10, 5, replace=False))) if changing_palette else base_palette
            kind = "contact" if split == "train" and i == 0 else "docks" if split == "train" and i == 1 else "test" if split == "test" else "random"
            for _ in range(12):
                exclusions = training_shapes if split == "test" else reserved_keys
                pair, info = _make_pair(rng, scale, kind, palette, exclusions,
                                        reserved[i] if split == "test" else None)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                training_shapes.update(_canonical(h["logical"] != 0) for h in info["holes"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    centers, clean = info["centers"], info["clean"]
    owner = np.full(clean.shape, -1, dtype=int)
    scale = info["scale"]
    for i, hole in enumerate(info["holes"]):
        for r, c in hole["cells"]:
            owner[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = i
    pairs = [(p, q) for i, p in enumerate(centers) for q in centers[:i]]
    untransported = set()
    for assignment in info["assignments"]:
        piece = assignment["piece"]
        hole = info["holes"][assignment["hole"]]
        src = np.argwhere(piece["logical"] == 2)[0] * scale
        dst = np.argwhere(hole["logical"] == 2)[0] * scale
        a, c = hole["origin"]
        for r, col in np.argwhere(piece["erasures"]).tolist():
            untransported.add((int(a*scale + dst[0] + r-src[0]),
                              int(c*scale + dst[1] + col-src[1])))
    return {"scale": scale, "count": len(info["holes"]), "board_shape": info["board"].shape,
            "board_pattern": _key(info["board"] == info["structural"]),
            "shapes": {_canonical(h["logical"] != 0) for h in info["holes"]},
            "rooted": {_canonical(h["logical"]) for h in info["holes"]},
            "erasures": len(centers), "wrong": len(info["wrong_points"]),
            "noise": sum(map(len, info["noise"])),
            "both_fills": {int(clean[p]) for p in centers} == set(info["fills"]),
            "adjacent": any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1 for p, q in pairs),
            "interpiece_contact": any(owner[p] != owner[q] and max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1 for p, q in pairs),
            "reflection": any(a["reflection"] for a in info["assignments"]),
            "rotation": any(a["turns"] != 0 for a in info["assignments"]),
            "transport_moved": bool(set(centers) - untransported),
            "repeated_silhouette": len({_canonical(h["logical"] != 0) for h in info["holes"]}) < len(info["holes"]),
            "into_structure": any(clean[rr, cc] == info["structural"] for r, c in centers
                                  for rr in range(max(0, r-1), min(clean.shape[0], r+2))
                                  for cc in range(max(0, c-1), min(clean.shape[1], c+2)))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    training_shapes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(expected, out), "input-derived answer differs"
            scale = info["scale"]
            # Enlarged structural cells survive except precisely where a ring
            # crosses them. Dock ink and exterior background never survive.
            assert not np.any((out == info["dock"]) | (out == info["background"]))
            assert out.shape == tuple(n*scale for n in info["board"].shape)
            assert all(out[p] == info["structural"] for p in info["centers"])
            assert np.array_equal(_render(info["clean"], list(reversed(info["centers"])), info["fills"], info["structural"]), out)
            assert not np.array_equal(out, info["clean"]), "omitting B must change the answer"
            f = _features(info)
            features[split].append(f)
            if split == "train":
                training_shapes.update(f["shapes"])
            else:
                assert f["shapes"] - training_shapes, "test must include an unseen silhouette"
    train = features["train"]
    assert any(f["both_fills"] and f["interpiece_contact"] and f["into_structure"] for f in train)
    assert any(f["reflection"] for f in train) and any(f["rotation"] for f in train)
    assert any(f["transport_moved"] for f in train), "omitting A's coordinate transport must lose a center"
    assert any(f["repeated_silhouette"] for f in train), "dock-based assignment needs evidence"
    assert any(f["wrong"] > 0 for f in train), "wrong-fill repair needs evidence"
    assert len({f["board_pattern"] for f in train}) >= 2
    # A's dock correspondence is answer-relevant even for identical silhouettes.
    assert any(len(f["shapes"]) < f["count"] and f["both_fills"] for f in train)


def self_test():
    """Bounded structural and semantic diagnostics over 42 reproducible seeds."""
    signatures, shapes, rooted, board_patterns, scales, counts, palettes = set(), set(), set(), set(), set(), set(), set()
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                predicted, info = _recover(pair["input"])
                assert np.array_equal(predicted, pair["output"])
                f = _features(info)
                pair_count += 1
                shapes.update(f["shapes"])
                rooted.update(f["rooted"])
                board_patterns.add(f["board_pattern"])
                scales.add(f["scale"])
                counts.add(f["count"])
                palettes.add((info["background"], info["structural"], info["dock"], info["fills"]))
                # Second ring implementation: dilation by each intended fill.
                expected = info["clean"].copy()
                for color in info["fills"]:
                    paint = np.zeros(expected.shape, dtype=bool)
                    for r, c in info["centers"]:
                        if info["clean"][r, c] == color:
                            paint[max(0, r-1):r+2, max(0, c-1):c+2] = True
                    expected[paint] = next(f for f in info["fills"] if f != color)
                for p in info["centers"]:
                    expected[p] = info["structural"]
                assert np.array_equal(expected, predicted)
    assert len(signatures) == 42 and pair_count == 210
    assert scales == {2, 3, 4} and counts >= {2, 3, 4, 5}
    assert len(shapes) >= 40 and len(rooted) >= 80 and len(board_patterns) >= 120 and len(palettes) >= 30
    a, b = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    # Concrete omission probes: identical silhouettes cannot be colored without
    # docks, and wrong-fill defects cannot be marked like erasures.
    _, same_shapes = _recover(a["train"][1]["input"])
    swapped = same_shapes["clean"].copy()
    x, y = same_shapes["fills"]
    swapped[same_shapes["clean"] == x] = y
    swapped[same_shapes["clean"] == y] = x
    assert not np.array_equal(_render(swapped, same_shapes["centers"], (x, y), same_shapes["structural"]), a["train"][1]["output"])
    _, repaired = _recover(a["train"][0]["input"])
    assert repaired["wrong_points"]
    marked_noise = _render(repaired["clean"], repaired["centers"] + repaired["wrong_points"], repaired["fills"], repaired["structural"])
    assert not np.array_equal(marked_noise, a["train"][0]["output"])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(700+nt*10+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output was accepted")
    clean = np.ones((7, 7), dtype=np.int8)
    clean[:, 4:] = 2
    try:
        _render(clean, [(3, 3), (3, 4)], (1, 2), 8)
    except AssertionError:
        pass
    else:
        raise AssertionError("incompatible rings were accepted")
    clipped = _render(np.ones((4, 4), dtype=np.int8), [(0, 0)], (1, 2), 8)
    assert np.array_equal(clipped[:2, :2], [[8, 2], [2, 2]]) and clipped[-1, -1] == 1
    # Change the second demonstration's dock into an interchangeable rooted
    # class. It must be rejected rather than resolved by order or proximity.
    inp = a["train"][1]["input"].copy()
    _, info = _recover(inp)
    h0, h1 = info["holes"]
    equivalent = next(s for s in _transforms(h0["logical"])
                      if np.array_equal(s != 0, h1["logical"] != 0))
    r0, r1, c0, c1 = info["board_bounds"]
    row, col = h1["origin"]
    view = inp[r0+row:r0+row+h1["logical"].shape[0], c0+col:c0+col+h1["logical"].shape[1]]
    view[view == info["dock"]] = info["background"]
    view[equivalent == 2] = info["dock"]
    try:
        solve(inp)
    except AssertionError:
        pass
    else:
        raise AssertionError("interchangeable docks were accepted")
