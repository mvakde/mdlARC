"""Broken lantern windows: jointly restore a jigsaw and its optical contour.

Only standard-library modules and NumPy are used. No parent code, generation
state, source answers, or demonstration lookup is needed by solve().
"""
from __future__ import annotations

import itertools
import math
import numpy as np

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("a grid must be an integer rectangle of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    groups = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _NEIGHBORS:
                point = r + dr, c + dc
                if point in todo:
                    todo.remove(point)
                    group.append(point)
                    stack.append(point)
        groups.append(np.array(group, dtype=int))
    return groups


def _fit_flank(rows, sign):
    """Longest terminal thin stair, with at least two complete outward steps.

    This fits the observed pixels; it does not use the sampler's profiles.
    The longest suffix wins, then the least period. Phase is measured at the
    last visible column, so the first virtual boundary has the same cadence.
    """
    xs, candidates = np.arange(len(rows)), []
    for run in (1, 2, 3):
        for phase in range(run):
            predicted = rows[-1] + sign * ((xs - xs[-1] + phase) // run)
            length = 0
            for same in (predicted == rows)[::-1]:
                if not same:
                    break
                length += 1
            if length >= 2 * run + 1 and len(set(rows[-length:])) >= 3:
                candidates.append((length, -run, phase))
    if not candidates:
        return None
    length, neg_run, phase = max(candidates)
    return -neg_run, phase, length


def _optical_geometry(board, wall):
    """Recognize exactly one restored cap and pair of outward terminal rails."""
    matches = []
    for rotation in range(4):
        a = np.rot90(board, -rotation)  # canonical emission is rightward
        positions = np.argwhere(a == wall)
        if len(positions) < 13:
            continue
        rear, mouth = map(int, (positions[:, 1].min(), positions[:, 1].max()))
        cap = np.flatnonzero(a[:, rear] == wall)
        if len(cap) < 3 or np.any(np.diff(cap) != 1):
            continue
        if mouth - rear < 5 or mouth >= a.shape[1] - 1:
            continue
        bounds = []
        for x in range(rear + 1, mouth + 1):
            rows = np.flatnonzero(a[:, x] == wall)
            if len(rows) != 2 or rows[1] - rows[0] < 2:
                break
            bounds.append(tuple(map(int, rows)))
        if len(bounds) != mouth - rear:
            continue
        if cap[0] != bounds[0][0] or cap[-1] != bounds[0][1]:
            continue
        # Thin visible rails remain connected even at diagonal contacts.
        if any(abs(bounds[j][i] - bounds[j - 1][i]) > 1
               for j in range(1, len(bounds)) for i in (0, 1)):
            continue
        fits = [_fit_flank(np.array([b[i] for b in bounds]), sign)
                for i, sign in ((0, -1), (1, 1))]
        if any(f is None for f in fits):
            continue
        footprint = np.zeros(a.shape, dtype=bool)
        for x in range(rear + 1, a.shape[1]):
            if x <= mouth:
                lo, hi = bounds[x - rear - 1]
            else:
                lo = bounds[-1][0] - (x - mouth + fits[0][1]) // fits[0][0]
                hi = bounds[-1][1] + (x - mouth + fits[1][1]) // fits[1][0]
            footprint[max(0, lo + 1):min(a.shape[0], hi), x] = True
        if np.any(footprint & (a == wall)):
            continue
        matches.append({"rotation": rotation, "rear": rear, "mouth": mouth,
                        "fits": tuple(fits), "bounds": tuple(bounds),
                        "footprint": np.rot90(footprint, rotation).copy()})
    return matches[0] if len(matches) == 1 else None


def _parse_input(grid):
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    bg = int(np.bincount(border, minlength=10).argmax())
    _require(np.count_nonzero(border == bg) > len(border) // 2,
             "the background must dominate the border")
    groups = _components(grid != bg)
    _require(len(groups) == 4, "exactly four separated panes are required")
    for a, b in itertools.combinations(groups, 2):
        _require(np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2).min() >= 2,
                 "panes must have a background-cell gap, including diagonally")
    counts = np.bincount(grid[grid != bg], minlength=10)
    carrier = int(counts.argmax())
    keys = []
    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1] - 1):
            block = grid[r:r + 2, c:c + 2]
            values = list(map(int, block.flat))
            if (len(set(values)) == 4 and bg not in values and carrier not in values
                    and all(counts[v] == 1 for v in values)):
                keys.append((r, c, block.copy()))
    _require(len(keys) == 1, "exactly one four-singleton 2x2 tint legend required")
    kr, kc, key = keys[0]
    anchor = [i for i, g in enumerate(groups) if (kr, kc) in set(map(tuple, g))]
    _require(len(anchor) == 1, "the legend must be embedded in one pane")
    ordered = [groups[anchor[0]]] + [g for i, g in enumerate(groups) if i != anchor[0]]
    candidates = [v for v in range(10) if v not in (bg, carrier) and v not in key]
    wall = max(candidates, key=lambda v: counts[v])
    _require(counts[wall] >= 13 and counts[carrier] > counts[wall], "unreadable material roles")
    extras = [v for v in candidates if counts[v] and v != wall]
    _require(len(extras) <= 2 and sum(int(counts[v]) for v in extras) <= 8,
             "there may be up to eight opaque cells in one or two colors")
    tiles = []
    for coords in ordered:
        lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
        t = np.full(tuple(hi - lo), bg, dtype=np.int8)
        q = coords - lo
        t[q[:, 0], q[:, 1]] = grid[coords[:, 0], coords[:, 1]]
        tiles.append(t)
    area = sum(len(g) for g in ordered)
    side = math.isqrt(area)
    _require(side * side == area, "the combined pane area must be square")
    return {"background": bg, "carrier": carrier, "wall": wall, "key": key,
            "blocker_colors": tuple(extras), "tiles": tiles, "side": side,
            "components": ordered}


def _restore(tiles, bg, wall):
    """Exhaustive corner fitting, filtered by restored contour continuity.

    A corner contact fixes a rotated bounding box's translation. Symmetries
    are collapsed by the complete marked tile, not by its silhouette alone.
    Equivalent motions may exist, but the marked board AND pane ownership
    must be unique. Early exit occurs only after a second distinct restoration.
    """
    area = sum(np.count_nonzero(t != bg) for t in tiles)
    side = math.isqrt(area)
    if side * side != area:
        return [], 0
    corner_indices = (0, side - 1, side * (side - 1), side * side - 1)
    tables = {}
    for i, tile in enumerate(tiles):
        for corner in range(4):
            pool, seen = [], set()
            for k in ((0,) if i == 0 else range(4)):
                a = np.rot90(tile, k)
                h, w = a.shape
                if h > side or w > side:
                    continue
                r = 0 if corner < 2 else side - h
                c = 0 if corner % 2 == 0 else side - w
                cells = np.argwhere(a != bg) + [r, c]
                mask = sum(1 << (int(y) * side + int(x)) for y, x in cells)
                hits = [bool(mask & (1 << v)) for v in corner_indices]
                if sum(hits) != 1 or not hits[corner]:
                    continue
                signature = mask, a.tobytes()
                if signature not in seen:
                    seen.add(signature)
                    pool.append((mask, k, a, r, c))
            tables[i, corner] = pool
    if not tables[0, 0]:
        return [], 0
    solutions, shape_fits = {}, 0

    def search(corner, remaining, occupied, placed):
        nonlocal shape_fits
        if len(solutions) >= 2:
            return
        if corner == 4:
            if occupied != (1 << (side * side)) - 1:
                return
            shape_fits += 1
            board = np.full((side, side), bg, dtype=np.int8)
            owner = np.full((side, side), -1, dtype=np.int8)
            for co, (piece, item) in enumerate(placed):
                mask, k, a, r, c = item
                pts = np.argwhere(a != bg)
                board[pts[:, 0] + r, pts[:, 1] + c] = a[pts[:, 0], pts[:, 1]]
                owner[pts[:, 0] + r, pts[:, 1] + c] = co
            optic = _optical_geometry(board, wall)
            if optic is not None:
                signature = board.tobytes(), owner.tobytes()
                solutions[signature] = {
                    "board": board, "owner": owner, "optics": optic,
                    "motions": tuple((p, item[1]) for p, item in placed)}
            return
        for piece in remaining:
            for item in tables[piece, corner]:
                if item[0] & occupied:
                    continue
                search(corner + 1, tuple(v for v in remaining if v != piece),
                       occupied | item[0], placed + [(piece, item)])

    for item in tables[0, 0]:
        search(1, (1, 2, 3), item[0], [(0, item)])
    return list(solutions.values()), shape_fits


def _illuminate(board, owner, a):
    """Input-only lane scan; pane seams change tint but never stop a ray."""
    rotation = a["optics"]["rotation"]
    material = np.rot90(board, -rotation)
    ownership = np.rot90(owner, -rotation)
    footprint = np.rot90(a["optics"]["footprint"], -rotation)
    out = np.full(material.shape, a["background"], dtype=np.int8)
    wall = material == a["wall"]
    opaque = np.isin(material, a["blocker_colors"])
    out[wall | opaque] = material[wall | opaque]
    lit, shadow = np.zeros(material.shape, bool), np.zeros(material.shape, bool)
    for y in range(material.shape[0]):
        blocked = False
        for x in range(a["optics"]["rear"] + 1, material.shape[1]):
            if not footprint[y, x]:
                continue
            if opaque[y, x]:
                blocked = True
            elif blocked:
                shadow[y, x] = True
            else:
                lit[y, x] = True
                out[y, x] = a["key"].flat[int(ownership[y, x])]
    return tuple(np.rot90(v, rotation).copy() for v in (out, lit, shadow))


def _analyze_input(grid):
    a = _parse_input(grid)
    solutions, fits = _restore(a["tiles"], a["background"], a["wall"])
    _require(len(solutions) == 1, "the assembled marked window must be unique")
    a.update(solutions[0])
    a["shape_fits"] = fits
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the answer entirely from the pixels of one input."""
    a = _analyze_input(_grid(input_grid))
    return _illuminate(a["board"], a["owner"], a)[0]


def _make_wall(rng, n, rotation):
    # Construction uses forward profiles; recognition fits terminal pixels.
    runs = [int(rng.integers(1, 4 if n >= 12 else 3))] * 2
    if rng.random() < 0.35:
        runs[1] = int(rng.integers(1, 4 if n >= 12 else 3))
    rear = int(rng.integers(2))
    before = int(rng.integers(2, 4))
    flare = 2 * max(runs) + 1
    if rear + before + flare > n - 2:
        before, rear = 2, 0
    if before + flare > n - 2:
        return None
    center = n // 2 + int(rng.integers(-1, 2))
    bases = [int(rng.integers(1, 3)) for _ in range(2)]
    profiles = []
    for base, run in zip(bases, runs):
        phase = int(rng.integers(run))
        start = base
        if before == 3 and rng.random() < 0.5:
            start = base + 1
        chamber = [start] + [base] * (before - 1)
        profiles.append(chamber + [base + (j + phase) // run for j in range(flare)])
    if max(profiles[0]) > center or max(profiles[1]) >= n - center:
        return None
    wall = np.zeros((n, n), bool)
    wall[center - profiles[0][0]:center + profiles[1][0] + 1, rear] = True
    for j, (up, down) in enumerate(zip(*profiles), rear + 1):
        wall[center - up, j] = wall[center + down, j] = True
    return np.rot90(wall, rotation).copy()


def _partition(rng, n, mode):
    # Rectangles expose optical ambiguity; walks and exchanges create jigsaws.
    if mode == 0:
        h, cuts = n // 2, [n // 2] * n
    else:
        h, cuts = int(rng.integers(3, n - 2)), []
        cut = int(rng.integers(3, n - 2))
        for y in range(n):
            if y == h:
                cut = int(rng.integers(2, n - 1))
            elif rng.random() < 0.7:
                cut = int(np.clip(cut + rng.integers(-1, 2), 2, n - 2))
            cuts.append(cut)
    owner = np.empty((n, n), dtype=np.int8)
    for y, cut in enumerate(cuts):
        owner[y, :cut] = 0 if y < h else 2
        owner[y, cut:] = 1 if y < h else 3
    if mode == 2:
        for _ in range(n * 2):
            y, x = map(int, rng.integers(1, n - 1, 2))
            old = int(owner[y, x])
            new = int(rng.choice([owner[y + dr, x + dc] for dr, dc in _NEIGHBORS]))
            owner[y, x] = new
            if any(len(_components(owner == v)) != 1 for v in (old, new)):
                owner[y, x] = old
    return owner


def _tile_arrays(board, owner, bg):
    tiles = []
    for i in range(4):
        pts = np.argwhere(owner == i)
        lo, hi = pts.min(axis=0), pts.max(axis=0) + 1
        t = np.full(tuple(hi - lo), bg, dtype=np.int8)
        q = pts - lo
        t[q[:, 0], q[:, 1]] = board[pts[:, 0], pts[:, 1]]
        tiles.append(t)
    return tiles


def _scatter(rng, tiles, bg):
    """Free halo placement or randomly grouped horizontal/vertical shelves."""
    if rng.random() < 0.4:
        shape = tuple(map(int, rng.integers(22, 31, 2)))
        grid = np.full(shape, bg, dtype=np.int8)
        forbidden = np.zeros(shape, bool)
        success = True
        for i in rng.permutation(4):
            tile = tiles[i]
            h, w = tile.shape
            for _ in range(60):
                r = int(rng.integers(1, shape[0] - h))
                c = int(rng.integers(1, shape[1] - w))
                pts = np.argwhere(tile != bg)
                yy, xx = pts[:, 0] + r, pts[:, 1] + c
                if forbidden[yy, xx].any():
                    continue
                grid[yy, xx] = tile[pts[:, 0], pts[:, 1]]
                for dr, dc in itertools.product((-1, 0, 1), repeat=2):
                    forbidden[yy + dr, xx + dc] = True
                break
            else:
                success = False
                break
        if success:
            return grid
    gap = int(rng.integers(1, 4))
    layouts = []
    for spacing in (gap, 1):
        for transpose in (False, True):
            dims = [t.shape[::-1] if transpose else t.shape for t in tiles]
            for order in itertools.permutations(range(4)):
                for cuts in range(8):
                    groups, group = [], []
                    for j, i in enumerate(order):
                        group.append(i)
                        if j == 3 or (cuts >> j) & 1:
                            groups.append(group)
                            group = []
                    heights = [max(dims[i][0] for i in g) for g in groups]
                    widths = [sum(dims[i][1] for i in g) + spacing * (len(g) - 1)
                              for g in groups]
                    height = sum(heights) + spacing * (len(groups) - 1)
                    width = max(widths)
                    if max(height, width) <= 28:
                        layouts.append((transpose, groups, dims, heights, widths, height, width, spacing))
        if layouts:
            break
    _require(bool(layouts), "no bounded scattering layout")
    linear = [x for x in layouts if len(x[1]) == 1]
    pool = linear if linear and rng.random() < 0.35 else layouts
    tr, groups, dims, heights, widths, height, width, gap = pool[int(rng.integers(len(pool)))]
    pads = []
    for extent in (height, width):
        before = int(rng.integers(1, min(4, 30 - extent)))
        after = int(rng.integers(1, min(4, 31 - extent - before)))
        pads.append((before, after))
    work = np.full((height + sum(pads[0]), width + sum(pads[1])), bg, dtype=np.int8)
    row = pads[0][0]
    for g, gh, gw in zip(groups, heights, widths):
        col = pads[1][0] + int(rng.integers(width - gw + 1))
        for i in g:
            rr = row + int(rng.integers(gh - dims[i][0] + 1))
            tile = tiles[i].T if tr else tiles[i]
            pts = np.argwhere(tile != bg)
            work[pts[:, 0] + rr, pts[:, 1] + col] = tile[pts[:, 0], pts[:, 1]]
            col += dims[i][1] + gap
        row += gh + gap
    return work.T.copy() if tr else work


def _cross_shadow(board, owner, a, shadow):
    k = a["optics"]["rotation"]
    b, own, sh = (np.rot90(v, -k) for v in (board, owner, shadow))
    footprint = np.rot90(a["optics"]["footprint"], -k)
    opaque = np.isin(b, a["blocker_colors"])
    for y, x in np.argwhere(opaque & footprint):
        if np.any(sh[y, x + 1:] & (own[y, x + 1:] != own[y, x])):
            return True
    return False


def _quadrant_disagreement(owner, lit):
    """Count counterexamples away from ambiguous odd-size middle lines."""
    n = len(owner)
    yy, xx = np.indices(owner.shape)
    quadrants = 2 * (yy >= n / 2) + (xx >= n / 2)
    eligible = lit.copy()
    if n % 2:
        eligible &= (yy != n // 2) & (xx != n // 2)
    return int(np.count_nonzero(eligible & (owner != quadrants)))


def _visible_expansion(optics, lit):
    canonical = np.rot90(lit, -optics["rotation"])
    lo, hi = optics["bounds"][-1]
    rows = np.arange(len(canonical))[:, None]
    return int(np.count_nonzero(canonical[:, optics["mouth"] + 1:]
                                & ((rows <= lo) | (rows >= hi))))


def _put_blockers(rng, board, owner, a, style):
    if style == 0:
        return True
    k = a["optics"]["rotation"]
    b = np.rot90(board, -k).copy()
    own = np.rot90(owner, -k)
    footprint = np.rot90(a["optics"]["footprint"], -k)
    bar = style == 2 or (style == 3 and rng.random() < 0.5)
    options = []
    for y, x in np.argwhere(footprint & (b == a["carrier"])):
        y, x = int(y), int(x)
        future = [xx for xx in range(x + 1, len(b))
                  if footprint[y, xx] and b[y, xx] == a["carrier"]]
        if not any(own[y, xx] != own[y, x] for xx in future):
            continue
        if style == 3 and len(future) < 3:
            continue
        if bar and (y + 1 >= len(b) or not footprint[y + 1, x]
                    or b[y + 1, x] != a["carrier"]):
            continue
        options.append((y, x, future))
    if not options:
        return False
    y, x, future = options[int(rng.integers(len(options)))]
    colors = a["blocker_colors"]
    b[y, x] = colors[0]
    if bar:
        b[y + 1, x] = colors[-1]
    if style == 3:
        # A later opaque cell in the same lane adds no shadow width.
        xx = int(rng.choice(future[:-1]))
        b[y, xx] = colors[-1]
    if style >= 2:
        inert = np.argwhere(~footprint & (b == a["carrier"]))
        if len(inert):
            r, c = inert[int(rng.integers(len(inert)))]
            b[r, c] = colors[0]
    if style == 3 and rng.random() < 0.5:
        choices = np.argwhere(footprint & (b == a["carrier"]))
        if len(choices):
            r, c = choices[int(rng.integers(len(choices)))]
            b[r, c] = colors[-1]
    board[:] = np.rot90(b, k)
    return True


def _construction_output(board, owner, a):
    # Independent arithmetic shadow union, distinct from solve's lane scan.
    k = a["optics"]["rotation"]
    b, own = (np.rot90(v, -k) for v in (board, owner))
    footprint = np.rot90(a["optics"]["footprint"], -k)
    opaque = np.isin(b, a["blocker_colors"])
    shadow = np.zeros(b.shape, bool)
    for y, x in np.argwhere(opaque & footprint):
        shadow[y, x + 1:] = True
    out = np.full(b.shape, a["background"], dtype=np.int8)
    illuminated = footprint & ~shadow & ~opaque
    out[illuminated] = a["key"].reshape(-1)[own[illuminated]]
    preserve = opaque | (b == a["wall"])
    out[preserve] = b[preserve]
    return np.rot90(out, k).copy()


def _make_pair(rng, n, mode, style, rotation, colors, motions):
    bg, carrier, wall, blocker1, blocker2 = map(int, colors[:5])
    key = np.array(colors[5:9], dtype=np.int8).reshape(2, 2)
    walls = _make_wall(rng, n, rotation)
    if walls is None:
        return None
    owner = _partition(rng, n, mode)
    if len(set(owner[walls])) < (4 if mode == 0 else 3):
        return None
    board = np.full((n, n), carrier, dtype=np.int8)
    board[walls] = wall
    optics = _optical_geometry(board, wall)
    if optics is None:
        return None
    places = [(r, c) for r in range(n - 1) for c in range(n - 1)
              if np.all(owner[r:r + 2, c:c + 2] == 0)
              and not walls[r:r + 2, c:c + 2].any()]
    if not places:
        return None
    r, c = places[int(rng.integers(len(places)))]
    board[r:r + 2, c:c + 2] = key
    a = {"background": bg, "carrier": carrier, "wall": wall, "key": key,
         "blocker_colors": (blocker1, blocker2), "optics": optics}
    if not _put_blockers(rng, board, owner, a, style):
        return None
    output, lit, shadow = _illuminate(board, owner, a)
    if len(set(owner[lit])) < 3 or not np.any(~optics["footprint"] & ~walls):
        return None
    if style and not _cross_shadow(board, owner, a, shadow):
        return None
    if style == 2 and not _quadrant_disagreement(owner, lit):
        return None
    if mode == 0 and not _visible_expansion(optics, lit):
        return None
    tiles = [np.rot90(t, k).copy() for t, k in zip(_tile_arrays(board, owner, bg), motions)]
    solutions, fits = _restore(tiles, bg, wall)
    if len(solutions) != 1:
        return None
    s = solutions[0]
    if not np.array_equal(s["board"], board) or not np.array_equal(s["owner"], owner):
        return None
    if mode == 0 and fits <= 1:
        return None
    grid = _scatter(rng, tiles, bg)
    # Also verify legend detection after scattering, rather than assuming it.
    try:
        parsed = _parse_input(grid)
    except AssertionError:
        return None
    if not np.array_equal(parsed["key"], key):
        return None
    expected = _construction_output(board, owner, a)
    _require(np.array_equal(expected, output), "construction and lane scan disagree")
    return {"input": grid, "output": expected}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    shared_palette = bool(rng.random() < 0.25)
    preferred = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    first_key = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index == 0:
                n, mode, style = int(rng.choice([10, 12])), 0, 0
                direction = preferred
                motions = [0, 0, 1, 2]
            elif split == "train" and index == 1:
                n, mode, style = int(rng.choice([10, 11, 12])), 2, 2
                direction = (preferred + 1) % 4
                motions = [0, 1, 2, 3]
            elif split == "test":
                n, mode, style = 13, 2, 3
                direction = (preferred + int(rng.integers(4))) % 4
                motions = [0] + list(map(int, rng.integers(4, size=3)))
            else:
                n, mode, style = int(rng.integers(10, 13)), int(rng.integers(1, 3)), int(rng.integers(4))
                direction = int(rng.integers(4))
                motions = [0] + list(map(int, rng.integers(4, size=3)))
            colors = palette.copy() if shared_palette else rng.permutation(10)
            if shared_palette and (split == "test" or index > 0):
                colors[5:9] = rng.permutation(colors[5:9])
            if split == "train" and index == 0:
                first_key = tuple(map(int, colors[5:9]))
            elif split == "train" and index == 1 and tuple(map(int, colors[5:9])) == first_key:
                colors[5:9] = np.roll(colors[5:9], 1)
            for _ in range(160):
                pair = _make_pair(rng, n, mode, style, direction, colors, motions)
                if pair is None or _signature(pair["input"]) in seen:
                    continue
                seen.add(_signature(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError("bounded window construction exhausted")
    rng.shuffle(puzzle["train"])
    return puzzle


def _measure(a, lit, shadow):
    rotation = a["optics"]["rotation"]
    board = np.rot90(a["board"], -rotation)
    footprint = np.rot90(a["optics"]["footprint"], -rotation)
    opaque = np.isin(board, a["blocker_colors"])
    return {
        "side": len(board), "direction": rotation,
        "periods": tuple(f[0] for f in a["optics"]["fits"]),
        "projection": len(board) - a["optics"]["mouth"] - 1,
        "shape_fits": a["shape_fits"],
        "owner": _signature(a["owner"]),
        "pane_areas": tuple(int(np.count_nonzero(a["owner"] == i)) for i in range(4)),
        "lit_panes": len(set(a["owner"][lit])),
        "quadrant_disagreements": _quadrant_disagreement(a["owner"], lit),
        "visible_expansion": _visible_expansion(a["optics"], lit),
        "shadow_cells": int(shadow.sum()),
        "cross_shadow": _cross_shadow(a["board"], a["owner"], a, shadow),
        "repeated_lane": bool(np.any((opaque & footprint).sum(axis=1) >= 2)),
        "inert": int(np.count_nonzero(opaque & ~footprint)),
        "wall_panes": len(set(a["owner"][a["board"] == a["wall"]])),
        "key": tuple(map(int, a["key"].flat)),
        "palette": (a["background"], a["carrier"], a["wall"]),
        "rotations": tuple(k for _, k in a["motions"]),
    }


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "illegal counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            a = _analyze_input(inp)
            expected, lit, shadow = _illuminate(a["board"], a["owner"], a)
            _require(np.array_equal(out, expected), "incorrect optical assembly")
            _require(out.shape == (a["side"], a["side"]), "incorrect output area")
            preserve = (a["board"] == a["wall"]) | np.isin(a["board"], a["blocker_colors"])
            _require(np.array_equal(out[preserve], a["board"][preserve]), "lost optical marks")
            _require(np.all(out[lit] == a["key"].reshape(-1)[a["owner"][lit]]), "wrong pane tint")
            _require(np.all(out[shadow] == a["background"]), "a shadow was illuminated")
            _require(len(set(a["owner"][lit])) >= 3, "insufficient corner-key evidence")
            _require(np.any((out == a["background"]) & ~preserve), "selective erasure missing")
            _require(len(set(a["owner"][a["board"] == a["wall"]])) >= 3, "funnel not fragmented")
            records[split].append(_measure(a, lit, shadow))
    training = records["train"]
    _require(any(r["shadow_cells"] == 0 and r["shape_fits"] > 1 and r["wall_panes"] == 4
                 and r["visible_expansion"] > 0
                 for r in training), "missing clean contour-disambiguation demonstration")
    _require(any(r["cross_shadow"] and r["inert"] > 0 for r in training), "missing cross-pane opacity teaching")
    _require(any(r["quadrant_disagreements"] > 0 for r in training), "missing pane-versus-quadrant contrast")
    _require(len({r["owner"] for r in training}) >= 2, "cuts must change within training")
    _require(len({r["key"] for r in training}) >= 2, "the visible tint legend must change")
    for r in records["test"]:
        _require(r["side"] > max(v["side"] for v in training), "missing size shift")
        _require(r["cross_shadow"] and r["repeated_lane"], "missing shadow recombination shift")


def self_test():
    """Bounded 42-seed structural, causal, reproducibility and corruption checks."""
    # Literal rails and literal next-boundary expectations, not sampler masks.
    # Each fixture also checks a shadow crossing a pane seam, an unaffected
    # neighboring lane, and an outside-footprint opaque cell.
    fixtures = (
        ((5, 5, 4, 3, 2), 1, (1, 0)),
        ((6, 6, 5, 5, 4, 4, 3, 3), 2, (2, 2)),
        ((6, 5, 5, 4, 4, 3, 3, 2), 2, (2, 1)),
        ((6, 6, 5, 5, 5, 4, 4, 4, 3), 3, (3, 3)),
        ((6, 5, 5, 5, 4, 4, 4, 3, 3), 3, (3, 2)),
        ((6, 5, 5, 5, 4, 4, 4, 3, 3, 3), 3, (2, 2)),
    )
    for upper, period, future_boundaries in fixtures:
        b = np.ones((15, 15), dtype=np.int8)
        b[upper[0]:15 - upper[0], 0] = 2
        for x, y in enumerate(upper, 1):
            b[y, x] = b[14 - y, x] = 2
        optic = _optical_geometry(b, 2)
        _require(optic is not None and optic["fits"][0][0] == period, "literal cadence regression")
        for offset, boundary in enumerate(future_boundaries, 1):
            x = len(upper) + offset
            _require(not optic["footprint"][boundary, x]
                     and optic["footprint"][boundary + 1, x], "thin virtual boundary regression")
        yy, xx = np.indices(b.shape)
        owner = (2 * (yy >= 8) + (xx >= 8)).astype(np.int8)
        b[7, 6], b[0, 0] = 7, 8
        a = {"optics": optic, "background": 0, "wall": 2,
             "key": np.array([[3, 4], [5, 6]]), "blocker_colors": (7, 8)}
        out, _, _ = _illuminate(b, owner, a)
        _require(out[7, 9] == 0 and out[8, 9] == 6 and out[0, 0] == 8,
                 "literal cross-pane shadow regression")
        _require(out[8, 5] == 5 and out[8, 9] == 6, "seams must transmit and change tint")
    records, inputs, shapes, cuts, keys, palettes = [], set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed in (0, 13, 41):
            again = generate_puzzle(seed)
            _require(all(np.array_equal(p[k], q[k])
                         for s in puzzle for p, q in zip(puzzle[s], again[s])
                         for k in ("input", "output")), "determinism failed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                a = _analyze_input(inp)
                out, lit, shadow = _illuminate(a["board"], a["owner"], a)
                m = _measure(a, lit, shadow)
                records.append((split, m))
                sig = _signature(inp)
                _require(sig not in inputs, "cross-seed duplicate")
                inputs.add(sig); shapes.add(inp.shape); cuts.add(m["owner"])
                keys.add(m["key"]); palettes.add(m["palette"])
                # Causal ablations: no beam would tint dark material; no
                # legend would give one tint; no occlusion would light shadow.
                material = ~((a["board"] == a["wall"]) | np.isin(a["board"], a["blocker_colors"]))
                all_tinted = out.copy()
                all_tinted[material] = a["key"].reshape(-1)[a["owner"][material]]
                _require(not np.array_equal(all_tinted, out), "beam cutoff has no effect")
                one_tint = out.copy(); one_tint[lit] = a["key"][0, 0]
                _require(not np.array_equal(one_tint, out), "corner key has no effect")
                if m["quadrant_disagreements"]:
                    yy, xx = np.indices(out.shape)
                    quadrant_tint = out.copy()
                    quadrants = 2 * (yy >= len(out) / 2) + (xx >= len(out) / 2)
                    quadrant_tint[lit] = a["key"].reshape(-1)[quadrants[lit]]
                    _require(not np.array_equal(quadrant_tint, out), "pane shape has no effect on tint")
                if m["cross_shadow"]:
                    clear = a["board"].copy()
                    clear[np.isin(clear, a["blocker_colors"])] = a["carrier"]
                    no_occlusion = _illuminate(clear, a["owner"], a)[0]
                    _require(np.any(no_occlusion[shadow] != out[shadow]), "opacity has no effect")
    _require(len(inputs) == 210 and len(shapes) >= 30 and len(cuts) >= 120, "structural diversity collapsed")
    _require(len(keys) >= 90 and len(palettes) >= 80, "visible palettes collapsed")
    _require({r["side"] for _, r in records} == {10, 11, 12, 13}, "scale coverage")
    _require({p for _, r in records for p in r["periods"]} == {1, 2, 3}, "stair coverage")
    _require(any(r["periods"][0] != r["periods"][1] for _, r in records), "asymmetric cadence missing")
    _require(any(max(r["pane_areas"]) / min(r["pane_areas"]) >= 3 for _, r in records), "unequal panes missing")
    for split in ("train", "test"):
        rs = [r for s, r in records if s == split]
        _require({r["direction"] for r in rs} == {0, 1, 2, 3}, "direction coverage")
        _require({k for r in rs for k in r["rotations"]} == {0, 1, 2, 3}, "quarter-turn coverage")
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(42000 + 10 * tr + te, tr, te)
            _require(len(p["train"]) == tr and len(p["test"]) == te, "explicit count failure")
    for tr, te in ((1, 1), (11, 1), (4, 0), (5, 5), (6, 5), (2, 2),
                   (True, 1), (2.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal count combinations)")
