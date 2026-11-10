"""Queried glyphs become a rigid formation exactly when they complete a line.

The controller, token lattice and enclosure are reconstructed from pixels.
There is no learned palette, stored puzzle, parent import or generation state
in solve(). All movement tests the whole occupied footprint, including corners.
"""

from collections import deque
import itertools

import numpy as np


CARDINAL = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
DIRECTIONS = CARDINAL + DIAGONAL


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside its permitted range")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "color outside 0..9")
    return a


def _fingerprint(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _trim(a, occupied):
    rows, cols = np.where(occupied)
    _require(len(rows), "empty token")
    return a[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _token_key(a, bg):
    return _fingerprint(_trim(a, a != bg))


def _center(mask, g):
    mask = _trim(mask, mask)
    result = np.zeros((g, g), bool)
    r, c = (g - mask.shape[0]) // 2, (g - mask.shape[1]) // 2
    result[r:r + mask.shape[0], c:c + mask.shape[1]] = mask
    return result


def _is_centered(mask):
    rows, cols = np.where(mask)
    if not len(rows):
        return False
    return (abs(int(rows.min()) - (mask.shape[0] - 1 - int(rows.max()))) <= 1 and
            abs(int(cols.min()) - (mask.shape[1] - 1 - int(cols.max()))) <= 1)


def _shape_bank(g):
    bank = {}
    for bits in range(1, 1 << (g * g)):
        mask = np.array([(bits >> i) & 1 for i in range(g * g)], bool).reshape(g, g)
        mask = _center(mask, g)
        bank[_fingerprint(mask)] = mask
    return list(bank.values())


def _orientation(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _line(selected, n):
    if len(selected) != n:
        return None
    if len({p // n for p in selected}) == 1:
        return "row"
    if len({p % n for p in selected}) == 1:
        return "col"
    return None


def _decode_arrow(header, wall):
    colors = set(map(int, np.unique(header))) - {wall}
    _require(len(colors) == 2, "arrow must have two non-carrier roles")
    interpretations = []
    for tip in sorted(colors):
        trail = next(iter(colors - {tip}))
        tips = set(map(tuple, np.argwhere(header == tip)))
        trails = set(map(tuple, np.argwhere(header == trail)))
        if len(tips) != 2 or len(trails) != 4:
            continue
        for dr, dc in DIRECTIONS:
            expected = {(r - k * dr, c - k * dc) for r, c in tips for k in (1, 2)}
            if len(expected) == 4 and expected == trails:
                interpretations.append((tip, trail, (dr, dc)))
    _require(len(interpretations) == 1, "arrow polarity or direction ambiguous")
    return interpretations[0]


def _closed_regions(walls):
    """Label background components; enclosure membership does not use boxes."""
    h, w = walls.shape
    labels = np.full((h, w), -1, int)
    closed = set()
    label = 0
    for rr, cc in np.argwhere(~walls):
        r, c = int(rr), int(cc)
        if labels[r, c] >= 0:
            continue
        todo = deque([(r, c)])
        labels[r, c] = label
        edge = False
        while todo:
            r, c = todo.popleft()
            edge |= r in (0, h - 1) or c in (0, w - 1)
            for dr, dc in CARDINAL:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] and labels[nr, nc] < 0:
                    labels[nr, nc] = label
                    todo.append((nr, nc))
        if not edge:
            closed.add(label)
        label += 1
    return labels, closed


def _parse_canonical(a):
    """Canonical controller is above the arena; reflections need no undoing."""
    h, w = a.shape
    _require(h >= 22 and w >= 12, "scene too small")
    wall = int(a[0, 0])
    _require(np.all(a[0] == wall) and np.all(a[5] == wall), "missing controller borders")
    tip, trail, direction = _decode_arrow(a[:5], wall)
    _require(np.all(a[:5, [0, -1]] == wall), "open arrow compartment")
    choices = []
    for g in (2, 3):
        end = 6 + g
        if not np.all(a[end] == wall):
            continue
        wells = np.any(a[6:end] != wall, axis=0)
        runs = []
        for c in range(w):
            if wells[c] and (c == 0 or not wells[c - 1]):
                stop = c + 1
                while stop < w and wells[stop]:
                    stop += 1
                runs.append((c, stop))
        if len(runs) not in (3, 4) or any(c1 - c0 != g for c0, c1 in runs):
            continue
        if any(runs[i + 1][0] - runs[i][0] != g + 1 for i in range(len(runs) - 1)):
            continue
        query = [a[6:end, c0:c1].copy() for c0, c1 in runs]
        expected = np.full((g, w), wall, a.dtype)
        for (c0, c1), block in zip(runs, query):
            expected[:, c0:c1] = block
        if np.array_equal(expected, a[6:end]):
            choices.append((g, end, runs, query))
    _require(len(choices) == 1, "query well geometry ambiguous")
    g, end, runs, query = choices[0]
    n = len(query)
    bg = int(a[end + 1, 0])
    _require(bg not in (wall, tip, trail) and np.all(a[end + 1] == bg), "missing controller gap")
    arena0 = end + 2
    while arena0 < h and np.all(a[arena0] == bg):
        arena0 += 1
    _require(arena0 - end - 1 in (1, 2), "controller gap must be one or two cells")
    arena = a[arena0:]
    _require(not np.any((arena == tip) | (arena == trail)), "arrow roles used as arena ink")
    occupied = (arena != bg) & (arena != wall)
    rr, cc = np.where(occupied)
    _require(len(rr), "no board glyphs")
    r0, c0, r1, c1 = map(int, (rr.min(), cc.min(), rr.max(), cc.max()))
    _require((r1 - r0 + 1 - g) % (n - 1) == 0 and (c1 - c0 + 1 - g) % (n - 1) == 0,
             "glyphs do not form the indicated square lattice")
    pr, pc = (r1 - r0 + 1 - g) // (n - 1), (c1 - c0 + 1 - g) // (n - 1)
    _require(pr in (g + 2, g + 3) and pc in (g + 2, g + 3), "invalid lattice pitch")
    expected_ink = np.zeros(arena.shape, bool)
    tokens, cells, lookup, shape_keys, inks = [], [], {}, [], []
    for r, c in itertools.product(range(n), repeat=2):
        top, left = r0 + r * pr, c0 + c * pc
        block = arena[top:top + g, left:left + g].copy()
        _require(block.shape == (g, g) and not np.any(block == wall), "token canvas intersects wall")
        mask = block != bg
        colors = np.unique(block[mask])
        _require(len(colors) == 1 and int(colors[0]) not in (tip, trail), "glyph must use one ink")
        _require(_is_centered(mask), "glyph is not consistently centered")
        key = _token_key(block, bg)
        _require(key not in lookup, "duplicate trimmed colored token")
        lookup[key] = len(tokens)
        tokens.append(block)
        shape_keys.append(_fingerprint(_trim(mask, mask)))
        inks.append(int(colors[0]))
        points = {(int(y) + top + arena0, int(x) + left) for y, x in np.argwhere(mask)}
        cells.append(points)
        expected_ink[top:top + g, left:left + g] = mask
    _require(np.array_equal(occupied, expected_ink), "ink outside the board lattice")
    for r in range(n):
        _require(any(_trim(tokens[r * n + c], tokens[r * n + c] != bg).shape[0] == g
                     for c in range(n)), "each lattice row needs a full-height glyph")
    for c in range(n):
        _require(any(_trim(tokens[r * n + c], tokens[r * n + c] != bg).shape[1] == g
                     for r in range(n)), "each lattice column needs a full-width glyph")
    order = []
    for block in query:
        _require(not np.any(np.isin(block, [wall, trail, tip])), "unexplained query color")
        if np.all(block == bg):
            continue
        _require(_is_centered(block != bg), "query centering changed")
        key = _token_key(block, bg)
        _require(key in lookup, "query lacks an exact shape-and-color board copy")
        p = lookup[key]
        _require(np.array_equal(block, tokens[p]), "query is not a faithful canvas copy")
        order.append(p)
    _require(1 <= len(order) <= n and len(set(order)) == len(order), "invalid query multiplicity")
    wall_mask = np.zeros(a.shape, bool)
    wall_mask[arena0:] = arena == wall
    labels, closed = _closed_regions(arena == wall)
    regions = {int(labels[r - arena0, c]) for item in cells for r, c in item}
    _require(len(regions) == 1 and regions <= closed, "glyphs must share one closed arena region")
    return dict(grid=a, wall=wall, bg=bg, tip=tip, trail=trail, direction=direction,
                g=g, n=n, panel_end=end, arena0=arena0, pitch=(pr, pc), origin=(r0 + arena0, c0),
                tokens=tokens, cells=cells, order=order, selected=set(order),
                line=_line(set(order), n), wall_mask=wall_mask, shape_keys=shape_keys, inks=inks)


def _read(input_grid):
    a = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        try:
            info = _parse_canonical(np.rot90(a, rotation))
        except (AssertionError, IndexError):
            continue
        info["rotation"] = rotation
        candidates.append(info)
    _require(len(candidates) == 1, "controller/arena orientation ambiguous")
    return candidates[0]


def _trace(cells, obstacles, direction, shape, corner=True):
    """Translate an entire footprint one unit at a time on a frozen scene."""
    dr, dc = direction
    footprint = set(cells)
    distance = 0
    sweep = set(cells)
    h, w = shape
    def hits(points):
        return any(not (0 <= r < h and 0 <= c < w) or (r, c) in obstacles for r, c in points)
    while distance <= max(shape):
        proposed = {(r + dr, c + dc) for r, c in footprint}
        direct = hits(proposed)
        side = bool(dr and dc and (hits({(r + dr, c) for r, c in footprint}) or
                                   hits({(r, c + dc) for r, c in footprint})))
        if direct or (corner and side):
            return distance, sweep, bool(side and not direct)
        footprint = proposed
        distance += 1
        sweep.update(footprint)
    raise AssertionError("unbounded translation")


def _motion(info, force=None, corner=True, direction=None):
    a, selected = info["grid"], info["selected"]
    static = set(map(tuple, np.argwhere(info["wall_mask"])))
    static.update(p for i, cells in enumerate(info["cells"]) if i not in selected for p in cells)
    direction = info["direction"] if direction is None else direction
    rigid = bool(info["line"]) if force is None else force
    independent = {}
    sweeps, witnesses = {}, {}
    for i in sorted(selected):
        independent[i], sweeps[i], witnesses[i] = _trace(info["cells"][i], static, direction, a.shape, corner)
    if rigid:
        group = set().union(*(info["cells"][i] for i in selected))
        distance, _, witness = _trace(group, static, direction, a.shape, corner)
        distances = {i: distance for i in selected}
        _require(distance == min(independent.values()), "rigid first-contact invariant failed")
        corner_witness = witness
    else:
        distances = independent.copy()
        used = set()
        for i in sorted(selected):
            _require(used.isdisjoint(sweeps[i]), "loose glyphs have interacting swept ink")
            used.update(sweeps[i])
        corner_witness = any(witnesses.values())
    out = a.copy()
    for i in selected:
        for p in info["cells"][i]:
            out[p] = info["bg"]
    destinations = set()
    dr, dc = direction
    for i in sorted(selected):
        distance = distances[i]
        for r, c in info["cells"][i]:
            p = r + distance * dr, c + distance * dc
            _require(p not in static and p not in destinations, "destination collision")
            _require(p[0] > info["panel_end"], "glyph left its arena")
            destinations.add(p)
            out[p] = info["inks"][i]
    return out, dict(rigid=rigid, distances=distances, independent=independent,
                    corner_witness=corner_witness, sweeps=sweeps, obstacles=static)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone; preserve the controller and all fixed ink."""
    info = _read(input_grid)
    out, _ = _motion(info)
    return np.rot90(out, -info["rotation"]).copy()


def _arrow(rng, direction, width, wall, trail, tip):
    points = []
    dr, dc = direction
    for j in range(2):
        base = (2 * j, 0) if dr == 0 else (0, 2 * j)
        for k in range(3):
            points.append((base[0] + k * dr, base[1] + k * dc, tip if k == 2 else trail))
    r0, c0 = min(p[0] for p in points), min(p[1] for p in points)
    glyph_width = max(p[1] for p in points) - c0 + 1
    left = int(rng.integers(1, width - glyph_width))
    header = np.full((5, width), wall, np.int8)
    for r, c, color in points:
        header[r - r0 + 1, c - c0 + left] = color
    return header


def _board(rng, n, g, inks, selected, bank):
    full = [i for i, m in enumerate(bank) if _trim(m, m).shape == (g, g)]
    # Several vocabulary densities are supported, without a fixed mask bank per seed.
    keep = len(bank) if rng.random() < .5 else max(n, (n * n + len(inks) - 1) // len(inks) + 2)
    subset = list(map(int, rng.choice(len(bank), min(keep, len(bank)), replace=False)))
    # A selected/unselected same-shape pair and a same-ink different-shape pair
    # directly expose both parts of exact token identity in every example.
    p = int(rng.choice(sorted(selected)))
    outside = [i for i in range(n * n) if i not in selected]
    u, v = map(int, rng.choice(outside, 2, replace=False))
    s, z = map(int, rng.choice(full, 2, replace=False))
    x, y = map(int, rng.choice(inks, 2, replace=False))
    assignment = [None] * (n * n)
    assignment[p], assignment[u] = (s, x), (s, y)
    assignment[v] = (z, x)
    used = {assignment[p], assignment[u], assignment[v]}
    for i in range(n):
        index = i * n + i
        if assignment[index] is None:
            candidates = [(s, ink) for s in full for ink in inks if (s, ink) not in used]
            assignment[index] = candidates[int(rng.integers(len(candidates)))]
            used.add(assignment[index])
    options = [(s, ink) for s in subset for ink in inks if (s, ink) not in used]
    rng.shuffle(options)
    for i in range(n * n):
        if assignment[i] is None:
            assignment[i] = options.pop()
    return [(bank[s].copy(), int(ink)) for s, ink in assignment]


def _selection(rng, n, kind, direction, lesson=False):
    if kind in ("row", "col"):
        dr, dc = direction
        if lesson:
            index = (n - 1 if dr > 0 else 0) if kind == "row" else (n - 1 if dc > 0 else 0)
        else:
            index = int(rng.integers(n))
        return {index * n + j if kind == "row" else j * n + index for j in range(n)}
    if kind == "partial":
        return set(map(int, rng.choice(n * n, int(rng.integers(1, n)), replace=False)))
    dr, dc = direction
    if dr and dc:
        # A full diagonal is explicitly a loose selection, never a winning line.
        if lesson or rng.random() < .3:
            return {i * n + (i if dr * dc < 0 else n - 1 - i) for i in range(n)}
        choices = set(map(int, rng.choice(n * n, n, replace=False)))
    elif dr == 0:
        choices = {r * n + int(rng.integers(n)) for r in range(n)}
    else:
        choices = {int(rng.integers(n)) * n + c for c in range(n)}
    if _line(choices, n):
        # Preserve distinct transverse lanes while making one near-miss.
        old = min(choices)
        choices.remove(old)
        if dr and dc:
            candidates = [i for i in range(n * n) if i not in choices and i != old
                          and _line(choices | {i}, n) is None]
            choices.add(int(rng.choice(candidates)))
        elif dr == 0:
            choices.add(old + 1 if old % n < n - 1 else old - 1)
        else:
            choices.add((old + n) % (n * n))
    return choices


def _arena(rng, h, w, r0, c0, ext_r, ext_c, richer):
    # Four independent shores; a full central rectangle protects the visible board.
    max_top, max_bottom = r0 - 2, h - r0 - ext_r - 2
    max_left, max_right = c0 - 2, w - c0 - ext_c - 2
    def shore(length, maximum):
        if maximum <= 0 or (not richer and rng.random() < .2):
            return np.zeros(length, int)
        values = rng.integers(0, maximum + 1, length)
        if not richer:
            # Piecewise flats retain some simple contours in training.
            for j in range(0, length, 3):
                values[j:j + 3] = values[j]
        return values
    top, bottom = shore(w, max_top), shore(w, max_bottom)
    left, right = shore(h, max_left), shore(h, max_right)
    yy, xx = np.indices((h, w))
    footprint = ((yy >= top[None, :]) & (yy < h - bottom[None, :]) &
                 (xx >= left[:, None]) & (xx < w - right[:, None]))
    padded = np.pad(footprint, 1)
    free = footprint.copy()
    for dr, dc in CARDINAL:
        free &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return footprint & ~free


def _make_pair(rng, n, g, palette, bank, plan, test=False, allow_diagonal_queries=True):
    kind, direction, lesson, need_corner = plan
    bg, wall, trail, tip, *inks = palette
    # Shape/layout rejection is bounded. Dense glyphs and small margins ensure
    # ordinary members are easy to construct; only deliberate lessons need filters.
    for attempt in range(320):
        panel_end = 6 + g
        gap_choices = [gap for gap in (1, 2)
                       if (n - 1) * (g + 2) + g + 6 <= 30 - (panel_end + 1) - gap]
        gap = int(rng.choice(gap_choices))
        max_h = 30 - (panel_end + 1) - gap
        possible_pr = [p for p in (g + 2, g + 3) if (n - 1) * p + g + 6 <= max_h]
        pr = int(rng.choice(possible_pr))
        pc = int(rng.choice([p for p in (g + 2, g + 3) if (n - 1) * p + g + 6 <= 30]))
        er, ec = (n - 1) * pr + g, (n - 1) * pc + g
        h = int(rng.integers(er + 6, max_h + 1))
        w = int(rng.integers(max(ec + 6, n * (g + 1) + 3), 31))
        r0 = int(rng.integers(1, h - er))
        c0 = int(rng.integers(1, w - ec))
        selected = _selection(rng, n, kind, direction, lesson)
        if not allow_diagonal_queries and selected in (
                {i * n + i for i in range(n)}, {i * n + n - 1 - i for i in range(n)}):
            continue
        board = _board(rng, n, g, inks, selected, bank)
        walls = _arena(rng, h, w, r0, c0, er, ec, test or need_corner)
        arena0 = panel_end + 1 + gap
        a = np.full((arena0 + h, w), bg, np.int8)
        a[:panel_end + 1] = wall
        a[:5] = _arrow(rng, direction, w, wall, trail, tip)
        start = int(rng.integers(1, w - (n * (g + 1) - 1)))
        query = sorted(selected)
        if rng.random() < .75:
            rng.shuffle(query)
        slots = sorted(map(int, rng.choice(n, len(query), replace=False)))
        by_slot = dict(zip(slots, query))
        for i in range(n):
            block = a[6:6 + g, start + i * (g + 1):start + i * (g + 1) + g]
            block[:] = bg
            if i in by_slot:
                mask, ink = board[by_slot[i]]
                block[mask] = ink
        a[arena0:][walls] = wall
        for i, (mask, ink) in enumerate(board):
            r, c = divmod(i, n)
            block = a[arena0 + r0 + r * pr:arena0 + r0 + r * pr + g,
                      c0 + c * pc:c0 + c * pc + g]
            if np.any(block == wall):
                break
            block[mask] = ink
        else:
            try:
                info = _parse_canonical(a)
                out, move = _motion(info)
            except AssertionError:
                continue
            if max(move["distances"].values()) < 1:
                continue
            if info["line"] and min(move["distances"].values()) < 1:
                continue
            # Both modes have mandatory training witnesses against treating
            # every query as independent or every query as a rigid group.
            if lesson and len(set(move["independent"].values())) < 2:
                continue
            if need_corner and not move["corner_witness"]:
                continue
            # Every example has a boundary-limited selected glyph, so walls
            # are part of the transformation rather than incidental decoration.
            wall_points = set(map(tuple, np.argwhere(info["wall_mask"])))
            wall_limited = False
            dr, dc = direction
            for i, distance in move["independent"].items():
                final = {(r + distance * dr, c + distance * dc) for r, c in info["cells"][i]}
                front = {(r + dr, c + dc) for r, c in final}
                if dr and dc:
                    front |= {(r + dr, c) for r, c in final} | {(r, c + dc) for r, c in final}
                wall_limited |= bool(front & wall_points)
            if not wall_limited:
                continue
            orientation = int(rng.integers(8))
            return {"input": _orientation(a, orientation), "output": _orientation(out, orientation)}
    raise AssertionError("bounded construction exhausted for " + str(plan))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    n, g = ((3, 2), (4, 2), (3, 3))[int(rng.integers(3))]
    bank = _shape_bank(g)
    palette = list(map(int, rng.choice(10, int(rng.integers(7 if n == 4 else 6, 9)), replace=False)))
    varying_palette = bool(rng.integers(2))
    row_dir = CARDINAL[int(rng.choice([0, 2]))]
    col_dir = CARDINAL[int(rng.choice([1, 3]))]
    first_axis = "row" if rng.integers(2) else "col"
    first_dir = row_dir if first_axis == "row" else col_dir
    plans = [(first_axis, first_dir, True, False),
             ("scatter", tuple(-x for x in first_dir), True, False)]
    if nt >= 3:
        plans.append(("col" if first_axis == "row" else "row",
                      col_dir if first_axis == "row" else row_dir, True, False))
    if nt >= 4:
        plans.append(("scatter", DIAGONAL[int(rng.integers(4))], True, True))
    if nt >= 5:
        plans.append(("partial", CARDINAL[int(rng.integers(4))], False, False))
    while len(plans) < nt:
        direction = DIRECTIONS[int(rng.integers(8))]
        kind = str(rng.choice(["row", "col", "scatter", "partial"]))
        plans.append((kind, direction, False, False))
    train_axes = {p[0] for p in plans if p[0] in ("row", "col")}
    for _ in range(nq):
        direction = DIRECTIONS[int(rng.integers(8 if nt >= 4 else 4))]
        if rng.random() < .5:
            kind = str(rng.choice(sorted(train_axes)))
        else:
            kind = "partial" if nt >= 5 and rng.random() < .3 else "scatter"
        plans.append((kind, direction, False, False))
    pairs, seen = [], set()
    for index, plan in enumerate(plans):
        for _ in range(8):
            local_palette = (list(map(int, rng.choice(10, len(palette), replace=False)))
                             if varying_palette else palette)
            pair = _make_pair(rng, n, g, local_palette, bank, plan, index >= nt, nt >= 4)
            key = _fingerprint(pair["input"])
            if key not in seen:
                seen.add(key)
                pairs.append(pair)
                break
        else:
            raise AssertionError("duplicate input construction")
    puzzle = {"train": [pairs[int(i)] for i in rng.permutation(nt)],
              "test": [pairs[nt + int(i)] for i in rng.permutation(nq)]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure_pair(pair):
    info = _read(pair["input"])
    out, move = _motion(info)
    selected = info["selected"]
    mask = info["wall_mask"][info["arena0"]:]
    ys, xs = np.where(mask)
    box = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    irregular = not (box[0].all() and box[-1].all() and box[:, 0].all() and box[:, -1].all())
    color_sensitive = any(info["shape_keys"][p] == info["shape_keys"][q]
                          for p in selected for q in range(info["n"] ** 2) if q not in selected)
    shape_sensitive = any(info["inks"][p] == info["inks"][q] and info["shape_keys"][p] != info["shape_keys"][q]
                          for p in selected for q in range(info["n"] ** 2) if q not in selected)
    # Footprint-limited contact with the wall versus merely the anchor point.
    return info, out, move, dict(shape=tuple(np.asarray(pair["input"]).shape), n=info["n"], g=info["g"],
        pitch=info["pitch"], direction=info["direction"], rotation=info["rotation"],
        outcome=info["line"] or "loose", occupancy=len(selected), irregular=irregular,
        distances=tuple(move["distances"].values()), independent=tuple(move["independent"].values()),
        corner=move["corner_witness"], color_sensitive=color_sensitive, shape_sensitive=shape_sensitive,
        ordered=info["order"] == sorted(selected), bg=info["bg"],
        palette=tuple(map(int, np.unique(pair["input"]))))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid example counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "dimensions changed")
            key = _fingerprint(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info, canonical_out, move, record = _measure_pair(pair)
            expected = np.rot90(canonical_out, -info["rotation"])
            _require(np.array_equal(expected, b), "output violates the motion rule")
            _require(not np.array_equal(a, b), "example demonstrates no motion")
            canonical_b = np.rot90(b, info["rotation"])
            fixed = info["wall_mask"].copy()
            fixed[:info["arena0"]] = True
            for i, cells in enumerate(info["cells"]):
                if i not in info["selected"]:
                    for cell in cells:
                        fixed[cell] = True
            _require(np.array_equal(info["grid"][fixed], canonical_b[fixed]), "fixed structure or ink changed")
            _require(np.array_equal(np.bincount(a.ravel(), minlength=10), np.bincount(b.ravel(), minlength=10)),
                     "ink/background counts not conserved")
            _require(record["color_sensitive"] and record["shape_sensitive"], "missing exact-identity contrast")
            wall_cells = set(map(tuple, np.argwhere(info["wall_mask"])))
            dr, dc = info["direction"]
            wall_limited = False
            for i, distance in move["independent"].items():
                final = {(r + distance * dr, c + distance * dc) for r, c in info["cells"][i]}
                probes = {(r + dr, c + dc) for r, c in final}
                if dr and dc:
                    probes |= {(r + dr, c) for r, c in final} | {(r, c + dc) for r, c in final}
                wall_limited |= bool(probes & wall_cells)
            _require(wall_limited, "arena wall has no causal role")
            analyses[split].append(record)
    train, test = analyses["train"], analyses["test"]
    _require(len({r["direction"] for r in train}) >= 2, "key control must be demonstrated")
    _require(any(r["outcome"] != "loose" and len(set(r["independent"])) >= 2 for r in train),
             "missing rigid first-contact witness")
    _require(any(r["outcome"] == "loose" and len(set(r["independent"])) >= 2 for r in train),
             "missing loose independent-motion witness")
    taught_axes = {r["outcome"] for r in train if r["outcome"] != "loose"}
    _require(all(r["outcome"] == "loose" or r["outcome"] in taught_axes for r in test), "untaught winning axis")
    _require(not any(r["corner"] for r in test) or any(r["corner"] for r in train), "untaught corner blocking")
    _require(not any(0 not in r["direction"] for r in test) or any(0 not in r["direction"] for r in train),
             "untaught diagonal motion")
    _require(not any(r["occupancy"] < r["n"] for r in test) or
             any(r["occupancy"] < r["n"] for r in train), "untaught empty wells")
    _require(len({(r["n"], r["g"]) for r in train + test}) == 1, "generated puzzle changed board/glyph scale")


def self_test():
    """Bounded checks of 42 seeds, metamorphic rules and deliberate corruptions."""
    puzzles, records, digests = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        repeated = generate_puzzle(seed)
        _require(all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], repeated[s])
                     for k in ("input", "output")), "nondeterminism")
        digests.add(tuple(_fingerprint(p["input"]) for s in puzzle for p in puzzle[s]))
        puzzles.append(puzzle)
        records.extend((s, _measure_pair(p)[3]) for s in ("train", "test") for p in puzzle[s])
    rows = [r for _, r in records]
    tests = [r for s, r in records if s == "test"]
    _require(len(digests) == 42, "seed diversity")
    _require({(r["n"], r["g"]) for r in rows} == {(3, 2), (4, 2), (3, 3)}, "scale variation")
    _require(len({r["shape"] for r in rows}) >= 20, "geometry variation")
    _require(len({r["pitch"] for r in rows}) >= 5, "lattice spacing variation")
    _require({r["direction"] for r in rows} == set(DIRECTIONS), "direction coverage")
    _require({r["outcome"] for r in tests} == {"row", "col", "loose"}, "test outcome variety")
    _require(sum(r["irregular"] for r in rows) >= len(rows) // 2, "irregular shore coverage")
    _require(len({r["palette"] for r in rows}) >= 15, "palette variety")
    _require(any(0 in r["distances"] for r in rows), "stationary members excluded")
    _require(max(max(r["distances"]) for r in rows) >= 4, "longer movement excluded")
    _require({r["ordered"] for r in rows} == {True, False}, "query ordering variation")
    # Controller order carries no semantic information.
    baseline = puzzles[0]
    for code in range(8):
        transformed = {s: [{k: _orientation(v, code) for k, v in p.items()} for p in baseline[s]]
                       for s in baseline}
        validate_generated_puzzle(transformed)
    for pair in baseline["train"]:
        info = _read(pair["input"])
        a = info["grid"].copy()
        end, g, wall = info["panel_end"], info["g"], info["wall"]
        starts = [c for c in range(a.shape[1]) if a[6, c] != wall and (c == 0 or a[6, c - 1] == wall)]
        blocks = [a[6:end, c:c + g].copy() for c in starts]
        for c, block in zip(starts, blocks[1:] + blocks[:1]):
            a[6:end, c:c + g] = block
        expected = np.rot90(pair["output"], info["rotation"]).copy()
        expected[:info["arena0"]] = a[:info["arena0"]]
        _require(np.array_equal(solve(a), expected), "query order affected result")
    # The two inherited ideas affect answers, not just input decoration.
    for puzzle in puzzles:
        rigid_seen = loose_seen = direction_seen = False
        for pair in puzzle["train"]:
            info = _read(pair["input"])
            actual, move = _motion(info)
            if info["line"]:
                if len(set(move["independent"].values())) >= 2:
                    alternate, _ = _motion(info, force=False)
                    rigid_seen |= not np.array_equal(actual, alternate)
                reverse = tuple(-x for x in info["direction"])
                alternate, _ = _motion(info, direction=reverse)
                direction_seen |= not np.array_equal(actual, alternate)
            elif len(set(move["independent"].values())) >= 2:
                alternate, _ = _motion(info, force=True)
                loose_seen |= not np.array_equal(actual, alternate)
        _require(rigid_seen and loose_seen and direction_seen, "parent ablation did not change answers")
    recolor = np.array([9, 2, 7, 0, 5, 8, 1, 6, 4, 3], np.int8)
    for pair in baseline["train"] + baseline["test"]:
        _require(np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]]),
                 "literal palette affected the rule")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * nt + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    partial = generate_puzzle(300, 6, 4)
    _require(any(len(_read(p["input"])["order"]) < _read(p["input"])["n"] for p in partial["train"]),
             "partial-query lesson absent")
    for defect in ("identity", "wall", "duplicate"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in baseline[s]] for s in baseline}
        if defect == "identity":
            bad["test"][0]["output"] = bad["test"][0]["input"].copy()
        elif defect == "wall":
            info = _read(bad["test"][0]["input"])
            out = np.rot90(bad["test"][0]["output"], info["rotation"])
            out[0, 0] = info["bg"]
        else:
            bad["train"][1] = {k: v.copy() for k, v in bad["train"][0].items()}
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("validator accepted " + defect)
    return {"seeds": 42, "pairs": len(rows), "distinct_puzzles": len(digests),
            "grid_shapes": len({r["shape"] for r in rows}),
            "irregular_arenas": sum(r["irregular"] for r in rows),
            "corner_witnesses": sum(r["corner"] for r in rows),
            "test_outcomes": {kind: sum(r["outcome"] == kind for r in tests) for kind in ("row", "col", "loose")}}


if __name__ == "__main__":
    print(self_test())
