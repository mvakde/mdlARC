"""Assemble a torn, occluded rectangular blueprint and reveal keyed traces.

Only standard-library modules and NumPy are used.  The solver enumerates the
visible tile arrangements; it does not consult the sampler or any saved scene.
"""
from __future__ import annotations

import itertools
import math
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _N4:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    group.append(p)
        groups.append(np.array(sorted(group), dtype=int))
    return groups


def _box(points):
    a = np.asarray(points, dtype=int)
    return (int(a[:, 0].min()), int(a[:, 0].max()),
            int(a[:, 1].min()), int(a[:, 1].max()))


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)} |
            {(b, x) for x in range(l, r + 1)} |
            {(y, l) for y in range(t + 1, b)} |
            {(y, r) for y in range(t + 1, b)})


def _rectangle(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _has_block(a, color):
    m = a == color
    return bool(np.any(m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]))


def _signature(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "a blank outer border is required")
    groups = _components(grid != bg)
    _require(len(groups) == 4, "exactly four separated tiles are required")
    for a, b in itertools.combinations(groups, 2):
        gap = np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2).min()
        _require(gap >= 2, "tiles require a background-cell gap")
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    singletons = colors[counts == 1]
    _require(len(singletons) == 4, "the four key entries must be singleton colors")
    key_points = np.argwhere(np.isin(grid, singletons))
    t, b, l, r = _box(key_points)
    _require(b == t + 1 and r == l + 1, "key must be a 2 by 2 block")
    key = grid[t:b + 1, l:r + 1].copy()
    anchor = [i for i, g in enumerate(groups) if (t, l) in set(map(tuple, g.tolist()))]
    _require(len(anchor) == 1, "missing anchor")
    groups = [groups[anchor[0]]] + [g for i, g in enumerate(groups) if i != anchor[0]]
    repeated = [(int(c), int(n)) for c, n in zip(colors, counts) if n > 1]
    largest = max(n for _, n in repeated)
    carriers = [c for c, n in repeated if n == largest]
    _require(len(carriers) == 1, "carrier must be the unique most frequent foreground color")
    carrier = carriers[0]
    masks = [c for c, _ in repeated if c != carrier and _has_block(grid, c)]
    _require(len(masks) == 1, "one noncarrier color must contain a solid 2 by 2 patch")
    mask_color = masks[0]
    inks = tuple(c for c, _ in repeated if c not in (carrier, mask_color))
    _require(1 <= len(inks) <= 3, "one to three ink colors are required")
    _require(all(n >= 3 for c, n in repeated if c in inks), "insufficient loop witnesses")
    tiles = []
    for g in groups:
        y0, y1, x0, x1 = _box(g)
        tile = np.full((y1 - y0 + 1, x1 - x0 + 1), -1, dtype=np.int8)
        tile[g[:, 0] - y0, g[:, 1] - x0] = grid[g[:, 0], g[:, 1]]
        _require(np.any(tile == carrier), "each tile must contain carrier cells")
        tiles.append(tile)
    total = sum(len(g) for g in groups)
    side = math.isqrt(total)
    _require(side * side == total and 8 <= side <= 14, "tile area must form a supported square")
    return {"grid": grid, "background": bg, "carrier": carrier,
            "mask_color": mask_color, "inks": inks, "key": key,
            "tiles": tiles, "side": side}


def _placements(tile, side, corner, anchor=False):
    result, seen = [], set()
    for k in ([0] if anchor else range(4)):
        a = np.rot90(tile, k)
        h, w = a.shape
        if h > side or w > side:
            continue
        y = 0 if corner < 2 else side - h
        x = 0 if corner % 2 == 0 else side - w
        coords = np.argwhere(a >= 0)
        rr, cc = coords[:, 0] + y, coords[:, 1] + x
        corners = {(0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)}
        target = ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1))[corner]
        hits = set(zip(rr.tolist(), cc.tolist())) & corners
        if hits != {target}:
            continue
        values = a[coords[:, 0], coords[:, 1]]
        mask = sum(1 << (int(r) * side + int(c)) for r, c in zip(rr, cc))
        sig = (mask, values.tobytes())
        if sig in seen:
            continue
        seen.add(sig)
        result.append((mask, rr, cc, values, k))
    return result


def _scene(board, owner, data, check_loops=True):
    """Read a candidate assembled scene and restore its concealed trace union."""
    mask_color, side = data["mask_color"], data["side"]
    masked = set(map(tuple, np.argwhere(board == mask_color).tolist()))
    if not masked:
        return None
    box = _box(list(masked))
    t, b, l, r = box
    if b - t < 2 or r - l < 2 or not (0 < t < b < side - 1 and 0 < l < r < side - 1):
        return None
    if masked != _rectangle(box):
        return None
    if not check_loops:
        return {"mask_box": box, "board": board.copy(), "owner": owner.copy()}
    hidden = set()
    loop_boxes = []
    perimeters = []
    for ink in data["inks"]:
        visible = set(map(tuple, np.argwhere(board == ink).tolist()))
        if not visible:
            return None
        bounds = _box(list(visible))
        y0, y1, x0, x1 = bounds
        if y1 - y0 < 2 or x1 - x0 < 2:
            return None
        perimeter = _perimeter(bounds)
        if perimeter - masked != visible or len(perimeter & masked) < 2:
            return None
        perimeters.append(perimeter)
        hidden.update(perimeter & masked)
        loop_boxes.append(bounds)
    # Crossings beneath the mask are allowed: every incident loop is painted
    # by the same tile-owner key, so intersections have no color conflict.
    for a, bnd in itertools.combinations(perimeters, 2):
        if (a & bnd) - masked:
            return None
    output = np.full((b - t + 1, r - l + 1), mask_color, dtype=np.int8)
    key = data["key"].ravel()
    for y, x in hidden:
        output[y - t, x - l] = key[owner[y, x]]
    return {"output": output, "mask_box": box, "hidden": hidden,
            "loop_boxes": tuple(loop_boxes), "owner": owner.copy(), "board": board.copy(),
            "hidden_crossings": sum(len(a & bnd & masked) for a, bnd in itertools.combinations(perimeters, 2))}


def _arrangements(data, check_loops=True, limit=None):
    side, tiles = data["side"], data["tiles"]
    table = {(i, c): _placements(tiles[i], side, c, anchor=(i == 0))
             for i in range(4) for c in ([0] if i == 0 else range(1, 4))}
    full = (1 << (side * side)) - 1
    board = np.full((side, side), -1, dtype=np.int8)
    owner = np.full((side, side), -1, dtype=np.int8)
    results, seen, chosen = [], set(), []

    def visit(corner, available, occupied):
        if limit is not None and len(results) >= limit:
            return
        if corner == 4:
            if occupied != full:
                return
            scene = _scene(board, owner, data, check_loops)
            if scene is None:
                return
            if check_loops:
                # Uniqueness is of the reconstructed answer, not tile identity.
                signature = _signature(scene["output"])
            else:
                cleaned = board.copy()
                cleaned[np.isin(cleaned, data["key"])] = data["carrier"]
                signature = _signature(cleaned)
            if signature not in seen:
                seen.add(signature)
                if check_loops:
                    scene["motion"] = tuple(chosen)
                results.append(scene)
            return
        for i in available:
            for placement in table.get((i, corner), []):
                mask, rr, cc, values, k = placement
                if mask & occupied:
                    continue
                board[rr, cc], owner[rr, cc] = values, corner
                chosen.append((i, k))
                visit(corner + 1, tuple(j for j in available if j != i), occupied | mask)
                chosen.pop()
                board[rr, cc], owner[rr, cc] = -1, -1
                if limit is not None and len(results) >= limit:
                    return

    # Anchor is handled separately so the subsequent available list is correct.
    for mask, rr, cc, values, k in table[(0, 0)]:
        board[rr, cc], owner[rr, cc] = values, 0
        chosen.append((0, 0))
        visit(1, (1, 2, 3), mask)
        chosen.pop()
        board[rr, cc], owner[rr, cc] = -1, -1
    return results


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the uniquely determined hidden, corner-keyed blueprint crop."""
    data = _parse(input_grid)
    answers = _arrangements(data, limit=2)
    _require(len(answers) == 1, "the input must determine one reconstructed answer")
    return answers[0]["output"].copy()


def _unchecked_completion(scene, data):
    """A deliberately weaker alternative, used only as an ablation witness.

    It completes ink bounding boxes even when their visible pixels fail the
    rectangular-perimeter condition.  It is never used by solve().
    """
    t, b, l, r = scene["mask_box"]
    output = np.full((b - t + 1, r - l + 1), data["mask_color"], dtype=np.int8)
    masked = _rectangle(scene["mask_box"])
    for color in data["inks"]:
        coords = np.argwhere(scene["board"] == color)
        for y, x in _perimeter(_box(coords)) & masked:
            output[y - t, x - l] = data["key"].ravel()[scene["owner"][y, x]]
    return output


def _partition(rng, side, style):
    """Connected corner cuts, wandering seams, or a large wrapping region."""
    if style == 0:
        cuts = (side // 2, side // 2, side // 2)
    else:
        cuts = (int(rng.integers(3, side - 2)),
                int(rng.integers(3, side - 2)), int(rng.integers(2, side - 2)))
    y, xt, xb = cuts
    owner = np.empty((side, side), dtype=np.int8)
    owner[:y, :xt], owner[:y, xt:] = 0, 1
    owner[y:, :xb], owner[y:, xb:] = 2, 3
    if style == 0:
        return owner
    if style == 2:
        # Three variable-width corner spines carved from a wrapping fourth.
        large = int(rng.integers(4))
        owner[:] = large
        for corner in rng.permutation([i for i in range(4) if i != large]):
            h = int(rng.integers(3, side - 2))
            w = int(rng.integers(2, side - 2))
            profile = int(rng.integers(3))
            for d in range(h):
                width = (w if profile == 0 or d in (0, h - 1) else
                         max(1, w - d) if profile == 1 else int(rng.integers(1, w + 1)))
                if corner == 0 and d < 2:
                    width = max(2, width)
                for x in range(width):
                    rr = d if corner < 2 else side - 1 - d
                    cc = x if corner % 2 == 0 else side - 1 - x
                    if owner[rr, cc] != large:
                        return None
                    owner[rr, cc] = corner
    else:
        # Local seam trades retain connectivity rather than imposing quadrants.
        for _ in range(side * side):
            r, c = int(rng.integers(1, side - 1)), int(rng.integers(1, side - 1))
            old = int(owner[r, c])
            neighbor = _N4[int(rng.integers(4))]
            new = int(owner[r + neighbor[0], c + neighbor[1]])
            if old == new or (r < 2 and c < 2):
                continue
            owner[r, c] = new
            if any(len(_components(owner == k)) != 1 for k in (old, new)):
                owner[r, c] = old
    corners = tuple(int(owner[r, c]) for r, c in
                    ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)))
    if corners != (0, 1, 2, 3):
        return None
    if any(np.count_nonzero(owner == k) < 6 or len(_components(owner == k)) != 1 for k in range(4)):
        return None
    return owner


def _loops(rng, side, masked, count):
    chosen, visible_used = [], set()
    for _ in range(count):
        for attempt in range(350):
            y0, y1 = sorted(map(int, rng.choice(side, 2, replace=False)))
            x0, x1 = sorted(map(int, rng.choice(side, 2, replace=False)))
            box = y0, y1, x0, x1
            if y1 - y0 < 2 or x1 - x0 < 2:
                continue
            full = _perimeter(box)
            visible, hidden = full - masked, full & masked
            if len(hidden) < 2 or len(visible) < 5 or visible & visible_used:
                continue
            if _box(list(visible)) != box:
                continue
            chosen.append(box)
            visible_used.update(visible)
            break
        else:
            return None
    return chosen


def _scatter(rng, tiles, bg):
    """Random shelf groupings plus free placements, independent of corner role."""
    dims = [t.shape for t in tiles]
    layouts = []
    for transposed in (False, True):
        dd = [d[::-1] if transposed else d for d in dims]
        for _ in range(32):
            order = list(map(int, rng.permutation(4)))
            # Includes rows, columns, unequal shelves, and a guaranteed 2+2.
            breaks = int(rng.integers(8))
            groups, group = [], []
            for j, i in enumerate(order):
                group.append(i)
                if j == 3 or (breaks >> j) & 1:
                    groups.append(group)
                    group = []
            gap = int(rng.integers(1, 4))
            heights = [max(dd[i][0] for i in g) for g in groups]
            widths = [sum(dd[i][1] for i in g) + gap * (len(g) - 1) for g in groups]
            h, w = sum(heights) + gap * (len(groups) - 1), max(widths)
            if max(h, w) > 28:
                continue
            offsets, row = [None] * 4, 0
            for g, gh, gw in zip(groups, heights, widths):
                col = int(rng.integers(w - gw + 1))
                for i in g:
                    rr = row + int(rng.integers(gh - dd[i][0] + 1))
                    offsets[i] = (col, rr) if transposed else (rr, col)
                    col += dd[i][1] + gap
                row += gh + gap
            layouts.append(offsets)
    if not layouts:
        order = list(map(int, rng.permutation(4)))
        offsets = [None] * 4
        second_row = max(dims[i][0] for i in order[:2]) + 1
        for group, row in ((order[:2], 0), (order[2:], second_row)):
            col = 0
            for i in group:
                offsets[i] = (row, col)
                col += dims[i][1] + 1
    else:
        offsets = layouts[int(rng.integers(len(layouts)))]
    if rng.random() < 0.4:
        # A free scatter can overlap bounding boxes but never foreground halos.
        forbidden = np.zeros((28, 28), dtype=bool)
        free = [None] * 4
        for i in map(int, rng.permutation(4)):
            coords = np.argwhere(tiles[i] >= 0)
            h, w = dims[i]
            for _ in range(100):
                rr, cc = int(rng.integers(29 - h)), int(rng.integers(29 - w))
                yy, xx = coords[:, 0] + rr, coords[:, 1] + cc
                if forbidden[yy, xx].any():
                    continue
                free[i] = rr, cc
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        forbidden[np.clip(yy + dy, 0, 27), np.clip(xx + dx, 0, 27)] = True
                break
            else:
                break
        if all(p is not None for p in free):
            offsets = free
    mins = np.min(offsets, axis=0)
    offsets = [(r - int(mins[0]), c - int(mins[1])) for r, c in offsets]
    extent = [max(offsets[i][a] + dims[i][a] for i in range(4)) for a in (0, 1)]
    padding = []
    for size in extent:
        room = 30 - size
        _require(room >= 2, "scatter does not fit")
        before = int(rng.integers(1, min(3, room - 1) + 1))
        after = int(rng.integers(1, min(3, room - before) + 1))
        padding.append((before, after))
    shape = tuple(extent[a] + sum(padding[a]) for a in (0, 1))
    grid = np.full(shape, bg, dtype=np.int8)
    for i, tile in enumerate(tiles):
        coords = np.argwhere(tile >= 0)
        y, x = offsets[i][0] + padding[0][0], offsets[i][1] + padding[1][0]
        grid[coords[:, 0] + y, coords[:, 1] + x] = tile[coords[:, 0], coords[:, 1]]
    return grid


def _make_pair(rng, bg, loop_count, first=False, second=False, test=False):
    for attempt in range(240):
        side = int(rng.choice([10, 12])) if first else int(rng.integers(10 if not test else 12, 14))
        style = 0 if first else (1 if second else int(rng.choice([0, 1, 2], p=[0.15, 0.50, 0.35])))
        owner = _partition(rng, side, style)
        if owner is None:
            continue
        mh, mw = int(rng.choice([4, 5, 6])), int(rng.choice([4, 5, 6]))
        if first:
            mh = mw = 4 if side == 10 else int(rng.choice([4, 6]))
        top = (side - mh) // 2 if first else int(rng.integers(2, side - mh - 1))
        left = (side - mw) // 2 if first else int(rng.integers(2, side - mw - 1))
        box = top, top + mh - 1, left, left + mw - 1
        masked = _rectangle(box)
        loops = _loops(rng, side, masked, loop_count)
        if loops is None:
            continue
        colors = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
        carrier, mask_color = colors[:2]
        key = np.array(colors[2:6], dtype=np.int8).reshape(2, 2)
        inks = colors[6:6 + loop_count]
        board = np.full((side, side), carrier, dtype=np.int8)
        hidden = set()
        for ink, bounds in zip(inks, loops):
            for p in _perimeter(bounds):
                if p not in masked:
                    board[p] = ink
                else:
                    hidden.add(p)
        board[top:top + mh, left:left + mw] = mask_color
        if hidden == masked:
            continue
        if second and not any(_perimeter(a) & _perimeter(b) & masked
                              for a, b in itertools.combinations(loops, 2)):
            continue
        if len(set(int(owner[p]) for p in hidden)) < (4 if first or second else 2):
            continue
        key_locations = [(r, c) for r in range(side - 1) for c in range(side - 1)
                         if np.all(owner[r:r + 2, c:c + 2] == 0)
                         and np.all(board[r:r + 2, c:c + 2] == carrier)]
        if not key_locations:
            continue
        r, c = key_locations[int(rng.integers(len(key_locations)))]
        board[r:r + 2, c:c + 2] = key
        tiles = []
        turns = [0] + ([1, 2, 3] if first else list(map(int, rng.integers(0, 4, 3))))
        for i in range(4):
            coords = np.argwhere(owner == i)
            t, b, l, r = _box(coords)
            tile = np.full((b - t + 1, r - l + 1), -1, dtype=np.int8)
            coords = coords - [t, l]
            tile[coords[:, 0], coords[:, 1]] = board[owner == i]
            tiles.append(np.rot90(tile, turns[i]).copy())
        grid = _scatter(rng, tiles, bg)
        try:
            data = _parse(grid)
        except AssertionError:
            continue
        if data["carrier"] != carrier or data["mask_color"] != mask_color:
            continue
        solutions = _arrangements(data, limit=2)
        if len(solutions) != 1:
            continue
        if first and {k for _, k in solutions[0]["motion"][1:]} != {1, 2, 3}:
            continue
        output = np.full((mh, mw), mask_color, dtype=np.int8)
        for y, x in hidden:
            output[y - top, x - left] = key.ravel()[owner[y, x]]
        if not np.array_equal(solutions[0]["output"], output):
            continue
        if first:
            alternatives = _arrangements(data, check_loops=False)
            if not any(not np.array_equal(_unchecked_completion(s, data), output) for s in alternatives):
                continue
        return {"input": grid, "output": output}
    raise RuntimeError("bounded search could not construct a readable blueprint")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    num_train, num_test = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    backgrounds = list(map(int, rng.permutation(10)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            first, second = split == "train" and index == 0, split == "train" and index == 1
            loops = 3 if split == "test" else 2 if first or second else int(rng.integers(1, 3))
            bg = backgrounds[len(seen) % 10]
            for _ in range(8):
                pair = _make_pair(rng, bg, loops, first, second, split == "test")
                sig = _signature(pair["input"])
                if second:
                    old_key = _parse(puzzle["train"][0]["input"])["key"]
                    if np.array_equal(_parse(pair["input"])["key"], old_key):
                        continue
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("failed to obtain a distinct input")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            data = _parse(inp)
            answers = _arrangements(data, limit=2)
            _require(len(answers) == 1, "ambiguous reconstruction")
            scene = answers[0]
            _require(np.array_equal(out, scene["output"]), "incorrect restored crop")
            _require(np.any(out == data["mask_color"]), "mask must leave some uninked cells")
            _require(np.count_nonzero(out != data["mask_color"]) >= 2, "occlusion must affect the answer")
            _require(len(set(int(v) for v in out.flat) - {data["mask_color"]}) >= 2,
                     "keyed ownership must affect the answer")
            _require(set(map(int, np.unique(out))) <= set(map(int, data["key"].flat)) | {data["mask_color"]},
                     "output contains carrier or source ink")
            records[split].append((data, scene))
    for data, scene in records["train"][:2]:
        _require(len(data["inks"]) == 2, "first demonstrations require two loops")
        _require(set(map(int, data["key"].flat)) <= set(map(int, scene["output"].flat)),
                 "first demonstrations must expose all four key positions")
    first_data, first_scene = records["train"][0]
    alternatives = _arrangements(first_data, check_loops=False)
    _require(any(not np.array_equal(_unchecked_completion(s, first_data), first_scene["output"])
                 for s in alternatives), "first example must disprove silhouette-plus-mask assembly")
    _require({k for _, k in first_scene["motion"][1:]} == {1, 2, 3},
             "first example must demonstrate all three nonzero quarter turns")
    _require(records["train"][1][1]["hidden_crossings"] > 0,
             "second example must demonstrate crossings beneath the mask")
    _require(records["train"][0][0]["background"] != records["train"][1][0]["background"],
             "background roles must vary")
    _require(not np.array_equal(first_data["key"], records["train"][1][0]["key"]),
             "key colors must change between the first demonstrations")
    _require(all(len(data["inks"]) == 3 for data, _ in records["test"]), "test must combine three loops")


def self_test():
    """Bounded 42-seed invariant and measured-variation audit."""
    shapes, masks, cuts, loop_layouts, backgrounds, ink_counts = set(), set(), set(), set(), set(), set()
    motions, crossings, irregular, puzzles = set(), 0, 0, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _parse(pair["input"])
                scene = _arrangements(data, limit=2)[0]
                shapes.add(pair["input"].shape)
                masks.add(pair["output"].shape)
                backgrounds.add(data["background"])
                ink_counts.add(len(data["inks"]))
                cuts.add(_signature(scene["owner"]))
                loop_layouts.add((data["side"], scene["mask_box"], scene["loop_boxes"]))
                motions.update(k for _, k in scene["motion"])
                crossings += scene["hidden_crossings"] > 0
                irregular += any(np.count_nonzero(tile >= 0) < tile.size for tile in data["tiles"])
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list-input solve failed")
    _require(len(puzzles) == 42 and len(shapes) >= 40 and len(cuts) >= 80, "insufficient structural variation")
    _require(len(masks) >= 6 and len(loop_layouts) >= 180, "mask or loop layouts collapsed")
    _require(backgrounds == set(range(10)) and ink_counts == {1, 2, 3}, "role/count variation missing")
    _require(motions == {0, 1, 2, 3} and crossings >= 10 and irregular >= 60, "joint mechanisms not exercised")
    one, two = generate_puzzle(566797796, 2, 1), generate_puzzle(566797796, 2, 1)
    _require(all(np.array_equal(a[k], b[k]) for s in one for a, b in zip(one[s], two[s])
                 for k in ("input", "output")), "nondeterministic generation")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts were accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in one[s]] for s in one}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output was accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds")
