"""Repair a damaged contour scaffold, then label its signed bends.

Only NumPy and the standard library are used. There is no import-time sampling,
file access, stored puzzle, learned palette, or state shared between calls.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


CONVEX = 4
CONCAVE = 2
INPUT_COLORS = (0, 1, 3, 5, 6, 7, 8, 9)
STEPS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if (len(colors) != 2 or max(counts) <= grid.size // 2
            or any(int(c) not in INPUT_COLORS for c in colors)):
        raise ValueError("Input needs two non-label colors and a majority background")
    return int(colors[np.argmax(counts)]), int(colors[np.argmin(counts)])


def _at(mask, cell):
    r, c = cell
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = ([(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c]
             if diagonal else STEPS)
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                nxt = r + dr, c + dc
                if nxt in remaining:
                    remaining.remove(nxt)
                    cells.add(nxt)
                    stack.append(nxt)
        components.append(cells)
    return components


def _degree(mask):
    padded = np.pad(mask.astype(np.int8), 1)
    return (padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:])


def _halo(mask):
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[r:r+mask.shape[0], c:c+mask.shape[1]]
                                 for r in range(3) for c in range(3)])


def _is_diagonal(cells):
    ordered = sorted(cells)
    return (len(ordered) in (2, 3)
            and all(ordered[i+1][0] - ordered[i][0] == 1
                    for i in range(len(ordered)-1))
            and len({ordered[i+1][1] - ordered[i][1]
                     for i in range(len(ordered)-1)}) == 1
            and abs(ordered[1][1] - ordered[0][1]) == 1)


def _diagonals(mask):
    """Diagonal-only strokes cannot contain an orthogonally supported cell."""
    erased = np.zeros_like(mask)
    chains = []
    for cells in _components(mask & (_degree(mask) == 0), diagonal=True):
        if _is_diagonal(cells):
            chains.append(cells)
            for p in cells:
                erased[p] = True
    return erased, chains


def _gaps(mask):
    """Five-cell windows with two retained brackets and exactly three blanks."""
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        view = mask if dc else mask.T
        if view.shape[1] < 5:
            continue
        bracket = view[:, :-4] & view[:, 4:]
        bracket &= ~(view[:, 1:-3] | view[:, 2:-2] | view[:, 3:-1])
        for y, x in np.argwhere(bracket):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _repair_evidence(mask, erased):
    repairs = []
    for r, c, dr, dc in _gaps(mask):
        if erased[r, c] or erased[r+4*dr, c+4*dc]:
            continue
        if not (_at(mask, (r-dr, c-dc)) or _at(mask, (r+5*dr, c+5*dc))):
            continue
        matches = []
        for end in (0, 4):
            ar, ac = r + end*dr, c + end*dc
            toward = 1 if end == 0 else -1
            for side in (-1, 1):
                vr, vc = toward*dr + side*dc, toward*dc + side*dr
                spur = tuple((ar+k*vr, ac+k*vc) for k in (1, 2))
                if (all(_at(erased, p) for p in spur)
                        and not _at(mask, (ar+3*vr, ac+3*vc))):
                    matches.append((end, side, spur))
        if len(matches) > 1:
            raise ValueError("Ambiguous spur attachment")
        if matches:
            repairs.append(((r, c, dr, dc), matches[0]))
    return repairs


def _reconstruct(mask):
    erased, chains = _diagonals(mask)
    repairs = _repair_evidence(mask, erased)
    clean = mask & ~erased
    added = np.zeros_like(mask)
    used_spurs = set()
    for (r, c, dr, dc), (_, _, spur) in repairs:
        if used_spurs.intersection(spur):
            raise ValueError("A diagonal cannot support two gaps")
        used_spurs.update(spur)
        for k in (1, 2, 3):
            p = r+k*dr, c+k*dc
            if added[p]:
                raise ValueError("Overlapping repairs")
            added[p] = True
    clean |= added
    return clean, added, erased, repairs, chains


def _neighbors(p, cells):
    r, c = p
    return sorted((r+dr, c+dc) for dr, dc in STEPS if (r+dr, c+dc) in cells)


def _trace(cells):
    if len(cells) < 8 or any(len(_neighbors(p, cells)) != 2 for p in cells):
        raise ValueError("Not a simple orthogonal cycle")
    start = min(cells)
    path, seen = [start], {start}
    previous, current = start, _neighbors(start, cells)[0]
    while current != start:
        if current in seen:
            raise ValueError("Self-intersecting contour")
        path.append(current)
        seen.add(current)
        following = [p for p in _neighbors(current, cells) if p != previous]
        previous, current = current, following[0]
    if seen != cells:
        raise ValueError("Disconnected contour")
    return path


def _classify(cells):
    path = _trace(cells)
    area = sum(c*path[(i+1) % len(path)][0] - r*path[(i+1) % len(path)][1]
               for i, (r, c) in enumerate(path))
    if not area:
        raise ValueError("Contour has no area")
    kinds = {}
    for i, (r, c) in enumerate(path):
        pr, pc = path[i-1]
        nr, nc = path[(i+1) % len(path)]
        cross = (c-pc)*(nr-r) - (r-pr)*(nc-c)
        kinds[(r, c)] = 0 if cross == 0 else (1 if cross*area > 0 else 2)
    return {"cells": cells, "path": path, "kinds": kinds,
            "convex": sum(k == 1 for k in kinds.values()),
            "concave": sum(k == 2 for k in kinds.values())}


def _analyze_mask(mask):
    loops, opened = [], []
    for cells in _components(mask):
        degrees = [len(_neighbors(p, cells)) for p in cells]
        if all(d == 2 for d in degrees):
            loops.append(_classify(cells))
        else:
            opened.append(cells)
    return loops, opened


def _paint(clean, bg, line):
    result = np.where(clean, line, bg).astype(np.int8)
    loops, _ = _analyze_mask(clean)
    for info in loops:
        for p, kind in info["kinds"].items():
            if kind:
                result[p] = CONVEX if kind == 1 else CONCAVE
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only visible geometry and the fixed 4/2 corner convention."""
    grid = _grid(input_grid)
    bg, line = _roles(grid)
    clean, _, _, _, _ = _reconstruct(grid == line)
    return _paint(clean, bg, line)


def _polyomino_contour(rng, limit=23, spacing_low=3, spacing_high=7):
    """A random rectangle union supplies arms, shoulders, and several-side notches.

    Perimeter vertices live on a coarse lattice; uneven coordinate expansion
    changes edge lengths independently. Point contacts and holes are rejected.
    This constructs geometry, rather than composing either parent's puzzles.
    """
    for _ in range(90):
        mh, mw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        tiles = {(int(rng.integers(mh)), int(rng.integers(mw)))}
        target = int(rng.integers(3, min(10, mh*mw)+1))
        while len(tiles) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in tiles for dr, dc in STEPS
                               if 0 <= r+dr < mh and 0 <= c+dc < mw} - tiles)
            tiles.add(frontier[int(rng.integers(len(frontier)))])
        edges = set()
        for r, c in sorted(tiles):
            square = ((r, c), (r, c+1), (r+1, c+1), (r+1, c))
            for p, q in zip(square, square[1:]+square[:1]):
                edge = tuple(sorted((p, q)))
                if edge in edges:
                    edges.remove(edge)
                else:
                    edges.add(edge)
        adj = {}
        for p, q in sorted(edges):
            adj.setdefault(p, []).append(q)
            adj.setdefault(q, []).append(p)
        if any(len(v) != 2 for v in adj.values()):
            continue
        start = min(adj)
        path, previous, current = [start], start, sorted(adj[start])[0]
        while current != start and current not in path:
            path.append(current)
            nxt = next(p for p in adj[current] if p != previous)
            previous, current = current, nxt
        if current != start or len(path) != len(adj):
            continue
        min_r, min_c = min(r for r, _ in path), min(c for _, c in path)
        max_r, max_c = max(r for r, _ in path), max(c for _, c in path)
        rs = np.concatenate(([0], np.cumsum(rng.integers(spacing_low, spacing_high, max_r-min_r))))
        cs = np.concatenate(([0], np.cumsum(rng.integers(spacing_low, spacing_high, max_c-min_c))))
        if max(rs[-1], cs[-1]) >= limit:
            continue
        vertices = [(int(rs[r-min_r]), int(cs[c-min_c])) for r, c in path]
        mask = np.zeros((int(rs[-1])+1, int(cs[-1])+1), dtype=bool)
        for p, q in zip(vertices, vertices[1:]+vertices[:1]):
            dr, dc = int(np.sign(q[0]-p[0])), int(np.sign(q[1]-p[1]))
            for k in range(abs(q[0]-p[0])+abs(q[1]-p[1])):
                mask[p[0]+k*dr, p[1]+k*dc] = True
        try:
            info = _classify(set(map(tuple, np.argwhere(mask).tolist())))
        except ValueError:
            continue
        if info["concave"]:
            return _orient(mask, int(rng.integers(8)))
    raise RuntimeError("Could not sample an irregular contour")


def _orient(mask, code):
    result = np.rot90(mask, code % 4)
    return np.ascontiguousarray(result[:, ::-1] if code >= 4 else result)


def _rectangle(rng, compact=False):
    h = int(rng.integers(3 if compact else 6, 7 if compact else 10))
    w = int(rng.integers(3 if compact else 7, 9 if compact else 13))
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    return _orient(mask, int(rng.integers(8)))


def _pack(rng, shapes, height, width):
    """Occupied-cell halos allow interlocking and nested bounding boxes."""
    masks, occupied = [], np.zeros((height, width), dtype=bool)
    for shape in shapes:
        h, w = shape.shape
        if h > height or w > width:
            return None
        blocked = _halo(occupied)
        choices = [(r, c) for r in range(height-h+1) for c in range(width-w+1)
                   if not (blocked[r:r+h, c:c+w] & shape).any()]
        if not choices:
            return None
        if rng.random() < .25:
            edge = [(r, c) for r, c in choices
                    if r == 0 or c == 0 or r+h == height or c+w == width]
            choices = edge or choices
        r, c = choices[int(rng.integers(len(choices)))]
        placed = np.zeros_like(occupied)
        placed[r:r+h, c:c+w] = shape
        masks.append(placed)
        occupied |= placed
    return masks


def _damage_candidates(clean, allowed):
    candidates = []
    for dr, dc in ((0, 1), (1, 0)):
        view = allowed if dc else allowed.T
        if view.shape[1] < 5:
            continue
        windows = np.logical_and.reduce([view[:, k:view.shape[1]-4+k] for k in range(5)])
        for y, x in np.argwhere(windows):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            if not (_at(clean, (r-dr, c-dc)) or _at(clean, (r+5*dr, c+5*dc))):
                continue
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            # A gap never hides a bend or junction.
            if any(_at(clean, (rr+dc, cc+dr)) or _at(clean, (rr-dc, cc-dr))
                   for rr, cc in gap):
                continue
            broken = clean.copy()
            for p in gap:
                broken[p] = False
            degree = _degree(broken)
            if (broken & (degree == 0)).any():
                continue
            for end in (0, 4):
                ar, ac = r+end*dr, c+end*dc
                toward = 1 if end == 0 else -1
                for side in (-1, 1):
                    vr, vc = toward*dr + side*dc, toward*dc + side*dr
                    spur = tuple((ar+k*vr, ac+k*vc) for k in (1, 2))
                    if all(0 <= rr < clean.shape[0] and 0 <= cc < clean.shape[1]
                           and not clean[rr, cc] and degree[rr, cc] == 0 for rr, cc in spur):
                        candidates.append((gap, spur, (r, c, dr, dc), end, side))
    return candidates


def _corrupt(rng, clean, primary, target, required_axes):
    candidates = _damage_candidates(clean, primary)
    inp, touched, count = clean.copy(), np.zeros_like(clean), 0
    for index in rng.permutation(len(candidates)):
        gap, spur, _, _, _ = candidates[int(index)]
        footprint = np.zeros_like(clean)
        for p in gap+spur:
            footprint[p] = True
        if (_halo(touched) & footprint).any():
            continue
        trial = inp.copy()
        for p in gap:
            trial[p] = False
        for p in spur:
            trial[p] = True
        try:
            rebuilt, _, erased, repairs, _ = _reconstruct(trial)
        except ValueError:
            continue
        if (np.array_equal(rebuilt, clean) and len(repairs) == count+1
                and np.array_equal(trial & (_degree(trial) == 0), erased)):
            inp, count = trial, count+1
            touched |= footprint
            if count == target:
                axes = {"h" if dr == 0 else "v" for (r, c, dr, dc), _ in repairs}
                return inp if required_axes <= axes else None
    return None


def _make_open(rng, skeleton, object_mask):
    """A bracketed but unsupported break, with an offset diagonal near miss."""
    candidates = _damage_candidates(skeleton, object_mask)
    for index in rng.permutation(len(candidates)):
        gap, _, (r, c, dr, dc), end, side = candidates[int(index)]
        clean = skeleton.copy()
        for p in gap:
            clean[p] = False
        ar, ac = r+end*dr, c+end*dc
        toward = 1 if end == 0 else -1
        # Shift the entire genuine deviation one cell sideways: its inferred
        # attachment is now background, not the retained endpoint.
        near = tuple((ar+k*toward*dr+(k+1)*side*dc,
                      ac+k*toward*dc+(k+1)*side*dr) for k in (1, 2))
        degree = _degree(clean)
        if not all(0 <= rr < clean.shape[0] and 0 <= cc < clean.shape[1]
                   and not clean[rr, cc] and degree[rr, cc] == 0 for rr, cc in near):
            continue
        trial = clean.copy()
        for p in near:
            trial[p] = True
        try:
            rebuilt, _, _, repairs, _ = _reconstruct(trial)
        except ValueError:
            continue
        if np.array_equal(rebuilt, clean) and not repairs:
            return clean, near
    return None


def _add_clutter(rng, inp, clean):
    result = inp.copy()
    for _ in range(int(rng.integers(4))):
        for _ in range(35):
            r, c = int(rng.integers(inp.shape[0])), int(rng.integers(inp.shape[1]))
            dr, dc = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
            length = int(rng.integers(2, 4))
            cells = tuple((r+k*dr, c+k*dc) for k in range(length))
            blocked = _halo(result | clean)
            if not all(0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]
                       and not blocked[rr, cc] for rr, cc in cells):
                continue
            trial = result.copy()
            for p in cells:
                trial[p] = True
            try:
                rebuilt = _reconstruct(trial)[0]
            except ValueError:
                continue
            if np.array_equal(rebuilt, clean):
                result = trial
                break
    return result


def _make_pair(rng, target, contrast, palette, required_axes=frozenset()):
    for _ in range(160):
        primary = _polyomino_contour(rng)
        shapes = [primary]
        extra = int(rng.integers(3))
        shapes.extend(_polyomino_contour(rng, 13, 1, 4) if rng.random() < .4
                      else _rectangle(rng, compact=True) for _ in range(extra))
        if contrast:
            shapes.append(_polyomino_contour(rng, 15, 2, 5) if rng.random() < .35 else _rectangle(rng))
        low_h = max(primary.shape[0], 21 if len(shapes) >= 3 else 16)
        low_w = max(primary.shape[1], 21 if len(shapes) >= 3 else 16)
        h, w = int(rng.integers(low_h, 31)), int(rng.integers(low_w, 31))
        placed = _pack(rng, shapes, h, w)
        if placed is None:
            continue
        clean = np.logical_or.reduce(placed)
        near = ()
        if contrast:
            opened = _make_open(rng, clean, placed[-1])
            if opened is None:
                continue
            clean, near = opened
        inp = _corrupt(rng, clean, placed[0], target, required_axes)
        if inp is None:
            continue
        for p in near:
            inp[p] = True
        try:
            if not np.array_equal(_reconstruct(inp)[0], clean):
                continue
        except ValueError:
            continue
        inp = _add_clutter(rng, inp, clean)
        if int(inp.sum()) >= inp.size/2:
            continue
        bg, line = palette
        return {"input": np.where(inp, line, bg).astype(np.int8),
                "output": _paint(clean, bg, line)}
    raise RuntimeError("Bounded contour construction exhausted")


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    n, t = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base = tuple(map(int, rng.choice(INPUT_COLORS, 2, replace=False)))
    palette_mode = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", n), ("test", t)):
        for index in range(count):
            # The first two demonstrations expose increasing simultaneous damage;
            # tests require three gaps on the same irregular contour.
            target = (1 if index == 0 else 2 if index == 1 else int(rng.integers(1, 3))) if split == "train" else 3
            contrast = (split == "train" and index == 0) or bool(rng.integers(2))
            palette = (tuple(map(int, rng.choice(INPUT_COLORS, 2, replace=False)))
                       if palette_mode == 2 else base[::-1] if palette_mode == 1 and rng.integers(2) else base)
            required_axes = set()
            if split == "train" and index == 1:
                first = puzzle["train"][0]["input"]
                _, first_line = _roles(first)
                first_axis = _reconstruct(first == first_line)[3][0][0][2]
                required_axes = {"h" if first_axis else "v"}
            for _ in range(8):
                pair = _make_pair(rng, target, contrast, palette, required_axes)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct distinct inputs")
    return puzzle


def _inside(path, r, c):
    # Samples are quarter-grid offsets, so no sample lies on an edge.
    return sum(c1 == c2 and min(r1, r2) < r < max(r1, r2) and c1 > c
               for (r1, c1), (r2, c2) in zip(path, path[1:]+path[:1])) % 2 == 1


def _independent_output(clean, bg, line):
    """Validate corner labels by interior quadrants, without the signed-turn test."""
    result = np.where(clean, line, bg).astype(np.int8)
    loops, _ = _analyze_mask(clean)
    for info in loops:
        for r, c in info["path"]:
            neighbors = _neighbors((r, c), info["cells"])
            a, b = neighbors
            if a[0]+b[0] == 2*r and a[1]+b[1] == 2*c:
                continue
            quadrants = sum(_inside(info["path"], r+dr, c+dc)
                            for dr in (-.25, .25) for dc in (-.25, .25))
            assert quadrants in (1, 3), "Undefined contour interior"
            result[r, c] = CONVEX if quadrants == 1 else CONCAVE
    return result


def _unsupported_closures(clean, opened):
    """Actual broken near-loops, excluding incidental inter-object alignments."""
    closures = []
    for r, c, dr, dc in _gaps(clean):
        for cells in opened:
            if (r, c) in cells and (r+4*dr, c+4*dc) in cells:
                added = {(r+k*dr, c+k*dc) for k in (1, 2, 3)}
                try:
                    _trace(cells | added)
                except ValueError:
                    continue
                closures.append((r, c, dr, dc))
    return closures


def _pair_info(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    bg, line = _roles(inp)
    mask = inp == line
    clean, added, erased, repairs, chains = _reconstruct(mask)
    assert repairs and added.any() and erased.any()
    assert np.array_equal(mask & (_degree(mask) == 0), erased)
    assert not (clean & (_degree(clean) == 0)).any()
    assert np.array_equal(clean & ~mask, added)
    assert np.array_equal(mask & ~clean, erased)
    loops, opened = _analyze_mask(clean)
    assert loops and any(info["concave"] for info in loops)
    loop_cells = set().union(*(x["cells"] for x in loops))
    assert all(tuple(p) in loop_cells for p in np.argwhere(added))
    assert all(1 <= len(_neighbors(p, cells)) <= 2 for cells in opened for p in cells)
    for info in loops:
        assert info["convex"] - info["concave"] == 4
        assert any(_inside(info["path"], r+.25, c+.25) for r, c in info["path"])
    for cells in _components(added):
        assert len(cells) == 3
        assert len({p[0] for p in cells}) == 1 or len({p[1] for p in cells}) == 1
    assert len(_components(added)) == len(repairs)
    assert all(set(spur) in chains for _, (_, _, spur) in repairs)
    assert np.array_equal(out, _independent_output(clean, bg, line)), "Wrong complete transformation"
    assert np.array_equal(solve(inp), out)
    # Explicit counterfactuals: neither parent's mechanism alone suffices.
    assert not np.array_equal(out, np.where(clean, line, bg))
    assert not np.array_equal(out, _paint(mask & ~erased, bg, line))
    assert any(out[p] == CONCAVE for info in loops for p in info["cells"])
    unsupported = _unsupported_closures(clean, opened)
    return {"shape": inp.shape, "palette": (bg, line), "repairs": len(repairs),
            "loops": len(loops), "open_paths": len(opened), "decoys": len(chains)-len(repairs),
            "unsupported": len(unsupported), "concaves": tuple(info["concave"] for info in loops),
            "axes": tuple(sorted({"h" if dr == 0 else "v" for (r, c, dr, dc), _ in repairs})),
            "attachments": tuple((end, side) for _, (end, side, _) in repairs),
            "added": added, "clean": clean, "loop_infos": loops}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, t = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(_grid(pair["input"]))
            assert key not in seen
            seen.add(key)
            infos[split].append(_pair_info(pair))
    assert any(x["open_paths"] and x["unsupported"] and x["decoys"] for x in infos["train"])
    assert set().union(*(set(x["axes"]) for x in infos["train"])) == {"h", "v"}
    assert max(x["repairs"] for x in infos["train"]) == 2
    assert all(x["repairs"] == 3 for x in infos["test"])


def self_test():
    """42 seeds, streamed in memory; no files, unbounded logs, or retained runs."""
    shapes, palettes, structures, complexities, counts, axes, attachments = (set() for _ in range(7))
    patterns, digests = set(), set()
    open_pairs = decoy_pairs = edge_loops = adjacent_bends = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _pair_info(pair)
                inp = pair["input"]
                bg, line = info["palette"]
                digest.update(bytes(inp.shape)+inp.tobytes()+pair["output"].tobytes())
                shapes.add(inp.shape)
                palettes.add(info["palette"])
                structures.add((inp.shape, (inp == line).tobytes()))
                complexities.update(info["concaves"])
                counts.add(info["loops"])
                axes.add(info["axes"])
                attachments.update(info["attachments"])
                open_pairs += bool(info["open_paths"])
                decoy_pairs += info["decoys"] > 0
                for loop in info["loop_infos"]:
                    turns = [k for k in loop["kinds"].values() if k]
                    reverse = turns[::-1]
                    patterns.add(min(tuple(seq[i:]+seq[:i]) for seq in (turns, reverse) for i in range(len(seq))))
                    edge_loops += any(r in (0, inp.shape[0]-1) or c in (0, inp.shape[1]-1) for r, c in loop["cells"])
                    adjacent_bends += any(loop["kinds"][p] and loop["kinds"][q]
                                          for p, q in zip(loop["path"], loop["path"][1:]+loop["path"][:1]))
        digests.add(digest.hexdigest())
        # One fresh input checks all eight rigid symmetries at each seed.
        pair = puzzle["test"][0]
        for code in range(8):
            assert np.array_equal(solve(_orient(pair["input"], code)), _orient(pair["output"], code))
        # A wrong sign and an omitted repair must both be rejected in full.
        for mutation in ("sign", "repair", "unsupported"):
            bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in ("train", "test")}
            changed = bad["train"][0]
            info = _pair_info(changed)
            bg, line = info["palette"]
            if mutation == "sign":
                changed["output"][changed["output"] == CONCAVE] = CONVEX
            elif mutation == "repair":
                mask = changed["input"] == line
                (r, c, dr, dc), (_, _, spur) = _reconstruct(mask)[3][0]
                for k in (1, 2, 3):
                    changed["output"][r+k*dr, c+k*dc] = bg
                for p in spur:
                    changed["output"][p] = line
            else:
                r, c, dr, dc = _unsupported_closures(info["clean"], _analyze_mask(info["clean"])[1])[0]
                changed["output"][r+dr, c+dc] = line
            try:
                validate_generated_puzzle(bad)
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError(f"Accepted {mutation} mutation")
    assert len(digests) == 42 and len(structures) >= 200
    assert len(shapes) >= 60 and len(palettes) >= 25
    assert {1, 2, 3}.issubset(counts) and len(complexities) >= 5 and len(patterns) >= 12
    assert {("h",), ("v",), ("h", "v")}.issubset(axes)
    assert attachments == {(0, -1), (0, 1), (4, -1), (4, 1)}
    assert open_pairs >= 42 and decoy_pairs >= 42 and edge_loops and adjacent_bends
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in ("train", "test")
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(digests),
            "grid_shapes": len(shapes), "ordered_palettes": len(palettes),
            "input_structures": len(structures), "loop_counts": sorted(counts),
            "concave_counts": sorted(complexities), "turn_patterns": len(patterns),
            "open_pairs": open_pairs, "decoy_pairs": decoy_pairs,
            "edge_touching_loops": edge_loops, "adjacent_bend_loops": adjacent_bends,
            "puzzle_hashes": sorted(digests)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
