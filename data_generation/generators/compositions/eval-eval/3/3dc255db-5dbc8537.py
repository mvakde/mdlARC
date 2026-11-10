"""Counted arrow projections disambiguate a finite rectangular mosaic.

Only NumPy and the standard library are used. Neither parsing nor solving reads
generation metadata. Generation paints its own intended output, then checks it
against an input-only joint packing/count solver.
"""
from __future__ import annotations

import json
from collections import Counter
import numpy as np


DIRS = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(int(v) for v in a.flat)


def _components(cells, diagonal=False):
    remaining = set(cells)
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, comp = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    comp.add(p)
                    stack.append(p)
        result.append(comp)
    return result


def _box(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr) + 1, max(cc) + 1


def _box_cells(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _local(p, direction):
    r, c = p
    return {"U": (r, c), "R": (-c, r), "D": (-r, -c), "L": (c, -r)}[direction]


def _world(p, direction):
    y, x = p
    return {"U": (y, x), "R": (x, -y), "D": (-y, -x), "L": (-x, y)}[direction]


def _read_arrow(frame, direction):
    points = {_local(p, direction) for p in frame}
    y0, x0, y1, x1 = _box(points)
    rows = [sorted(x for y, x in points if y == level) for level in range(y0, y1)]
    if len(rows) < 3 or len(rows[0]) != 1:
        return None
    tip_x = rows[0][0]
    left = right = tip_x
    split, cavity = False, set()
    for i, row in enumerate(rows[1:], 1):
        if not split and row == [tip_x]:
            continue
        if len(row) != 2 or not row[0] < tip_x < row[1]:
            return None
        lo, hi = row
        if not (0 <= left - lo <= 1 and 0 <= hi - right <= 1):
            return None
        split = True
        cavity.update(_world((y0 + i, x), direction) for x in range(lo + 1, hi))
        left, right = lo, hi
    if not split or not cavity:
        return None
    return _world((y0, tip_x), direction), cavity


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _parse(value):
    g = _grid(value)
    bg = int(g[0, 0])
    if not all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])):
        raise ValueError("the scene must have an empty outer margin")
    candidates = []
    for color in sorted(set(int(v) for v in g.flat) - {bg}):
        for comp in _components(map(tuple, np.argwhere(g == color)), True):
            box = _box(comp)
            r0, c0, r1, c1 = box
            patch = g[r0:r1, c0:c1]
            if min(patch.shape) < 4:
                continue
            if not all(np.all(edge == color) for edge in
                       (patch[0], patch[-1], patch[:, 0], patch[:, -1])):
                continue
            interior = patch[1:-1, 1:-1]
            if np.any(interior == bg) and set(int(v) for v in interior.flat) <= {bg, color}:
                candidates.append((color, box))
    if len(candidates) != 1:
        raise ValueError("expected one closed work frame")
    structure, box = candidates[0]
    r0, c0, r1, c1 = box
    mask = g[r0+1:r1-1, c0+1:c1-1] == bg
    if max(mask.shape) > 12:
        raise ValueError("work interior is too large")
    board_cells = _box_cells(box)
    outside_structure = set(map(tuple, np.argwhere(g == structure))) - board_cells
    arrows, excluded = [], set(board_cells)
    for frame in _components(outside_structure, True):
        matches = []
        for direction in DIRS:
            geometry = _read_arrow(frame, direction)
            if geometry is None:
                continue
            tip, cavity = geometry
            if not all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] for r, c in cavity):
                continue
            colors = {int(g[p]) for p in cavity} - {bg}
            if len(colors) != 1 or structure in colors:
                continue
            color = next(iter(colors))
            payload = {p for p in cavity if int(g[p]) == color}
            bbox = _box_cells(_box(frame))
            if any(g[p] != bg for p in bbox - frame - cavity):
                continue
            dr, dc = DIRS[direction]
            tr, tc = tip
            if dr == 0:
                if not r0 < tr < r1 - 1:
                    continue
                if not ((direction == "R" and tc < c0) or (direction == "L" and tc >= c1)):
                    continue
                axis, line = 0, tr - r0 - 1
                end = c0 if dc == 1 else c1 - 1
                corridor = [(tr, c) for c in range(min(tc, end)+1, max(tc, end))]
            else:
                if not c0 < tc < c1 - 1:
                    continue
                if not ((direction == "D" and tr < r0) or (direction == "U" and tr >= r1)):
                    continue
                axis, line = 1, tc - c0 - 1
                end = r0 if dr == 1 else r1 - 1
                corridor = [(r, tc) for r in range(min(tr, end)+1, max(tr, end))]
            if any(g[p] != bg for p in corridor):
                continue
            matches.append(dict(frame=frame, cavity=cavity, payload=payload,
                                color=color, q=len(payload), direction=direction,
                                tip=tip, axis=axis, line=line, bbox=bbox))
        if len(matches) != 1:
            raise ValueError("unreadable or ambiguous hollow arrow")
        arrow = matches[0]
        if not excluded.isdisjoint(arrow["bbox"]):
            raise ValueError("overlapping objects")
        excluded.update(arrow["bbox"])
        arrows.append(arrow)
    if not 1 <= len(arrows) <= 6:
        raise ValueError("expected 1..6 counted arrows")
    foreground = {(int(r), int(c)) for r, c in np.argwhere((g != bg) & (g != structure))}
    foreground -= excluded
    tiles, positions = [], []
    for comp in _components(foreground):
        b = _box(comp)
        cells = _box_cells(b)
        if cells != comp:
            raise ValueError("a source component is not a filled rectangle")
        y0, x0, y1, x1 = b
        tiles.append(g[y0:y1, x0:x1].copy())
        positions.append(b)
    if not 2 <= len(tiles) <= 8:
        raise ValueError("expected 2..8 rectangular source copies")
    return dict(grid=g, bg=bg, structure=structure, box=box, mask=mask,
                arrows=arrows, tiles=tiles, positions=positions)


def _line_mask(mask, clue):
    result = np.zeros_like(mask)
    if clue["axis"] == 0:
        result[clue["line"], :] = mask[clue["line"], :]
    else:
        result[:, clue["line"]] = mask[:, clue["line"]]
    return result


def _count(image, mask, clue):
    return int(np.sum(image[_line_mask(mask, clue)] == clue["color"]))


def _covers(mask, tiles, clues=(), limit=2, node_limit=None):
    """Enumerate rendered covers, grouping interchangeable physical copies.

    Generation may bound enumeration; solve uses no search budget. The returned
    boolean is false if a generation budget or outcome limit stopped the search.
    """
    coords = [tuple(int(v) for v in p) for p in np.argwhere(mask)]
    index = {p: i for i, p in enumerate(coords)}
    groups, lookup = [], {}
    for tile in tiles:
        key = _key(tile)
        if key in lookup:
            groups[lookup[key]][1] += 1
        else:
            lookup[key] = len(groups)
            groups.append([tile, 1])
    linebits = []
    for clue in clues:
        linebits.append(sum(1 << i for i, p in enumerate(coords)
                            if p[clue["axis"]] == clue["line"]))
    by_cell = [[] for _ in coords]
    placements = []
    for gi, (tile, _) in enumerate(groups):
        h, w = tile.shape
        for r in range(mask.shape[0] - h + 1):
            for c in range(mask.shape[1] - w + 1):
                if not np.all(mask[r:r+h, c:c+w]):
                    continue
                assignments = [(index[r+y, c+x], int(tile[y, x]))
                               for y in range(h) for x in range(w)]
                bits = sum(1 << i for i, _ in assignments)
                amounts = tuple(sum(value == clue["color"] for i, value in assignments
                                    if linebits[k] & (1 << i)) for k, clue in enumerate(clues))
                pi = len(placements)
                placements.append((gi, bits, assignments, amounts, r, c))
                for i, _ in assignments:
                    by_cell[i].append(pi)
    full = (1 << len(coords)) - 1
    remaining = [count for _, count in groups]
    totals = [0] * len(clues)
    rendering = [-1] * len(coords)
    outcomes, path, dead = {}, [], set()
    nodes, stopped = 0, False

    def search(covered):
        nonlocal nodes, stopped
        if stopped:
            return False
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            stopped = True
            return False
        for k, clue in enumerate(clues):
            if not totals[k] <= clue["q"] <= totals[k] + (linebits[k] & ~covered).bit_count():
                return False
        if covered == full:
            key = tuple(rendering)
            if key not in outcomes:
                image = np.full(mask.shape, -1, dtype=np.int8)
                image[mask] = key
                outcomes[key] = (image, tuple(path))
                if len(outcomes) >= limit:
                    stopped = True
            return True
        state = covered, tuple(remaining), tuple(totals)
        if state in dead:
            return False
        options = None
        for i in range(len(coords)):
            if covered & (1 << i):
                continue
            possible = [pi for pi in by_cell[i] if remaining[placements[pi][0]]
                        and not covered & placements[pi][1]]
            if not possible:
                dead.add(state)
                return False
            if options is None or len(possible) < len(options):
                options = possible
                if len(options) == 1:
                    break
        success = False
        for pi in options:
            gi, bits, assignments, amounts, r, c = placements[pi]
            remaining[gi] -= 1
            for i, value in assignments:
                rendering[i] = value
            for k, amount in enumerate(amounts):
                totals[k] += amount
            path.append((gi, r, c))
            success = search(covered | bits) or success
            path.pop()
            for k, amount in enumerate(amounts):
                totals[k] -= amount
            for i, _ in assignments:
                rendering[i] = -1
            remaining[gi] += 1
            if stopped:
                break
        if not success and not stopped:
            dead.add(state)
        return success

    search(0)
    return list(outcomes.values()), not stopped, nodes, groups


def _render(parsed, mosaic):
    out = parsed["grid"].copy()
    r0, c0, r1, c1 = parsed["box"]
    interior = out[r0+1:r1-1, c0+1:c1-1]
    interior[parsed["mask"]] = mosaic[parsed["mask"]]
    for arrow in parsed["arrows"]:
        for p in arrow["payload"]:
            out[p] = parsed["bg"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    parsed = _parse(input_grid)
    answers, _, _, _ = _covers(parsed["mask"], parsed["tiles"], parsed["arrows"])
    if len(answers) != 1:
        raise ValueError("the visible inventory and counted projections must give one mosaic")
    return _render(parsed, answers[0][0])


def _pattern(rng, shape, palette):
    a, b = (int(v) for v in rng.choice(palette, 2, replace=False))
    yy, xx = np.indices(shape)
    mode = int(rng.integers(6))
    tile = np.full(shape, a, dtype=np.int8)
    if mode == 1:
        tile[yy % 2 == 1] = b
    elif mode == 2:
        tile[xx % 2 == 1] = b
    elif mode == 3:
        tile[(yy + xx) % 2 == 1] = b
    elif mode == 4:
        tile[0, 0] = b
    elif mode == 5:
        tile = rng.choice(palette, size=shape).astype(np.int8)
    return tile


def _grow(rng, count):
    """Edge-joined rectangles, independently offset along their shared edge."""
    rects = []
    occupied = set()
    repeated = (int(rng.integers(1, 4)), int(rng.integers(2, 5)))
    for i in range(count):
        h, w = repeated if i < 2 else (int(rng.integers(1, 4)), int(rng.integers(1, 5)))
        options = []
        for _ in range(80):
            if not rects:
                r, c = 0, 0
            else:
                y, x, th, tw = rects[int(rng.integers(len(rects)))]
                side = int(rng.integers(4))
                if side < 2:
                    r = y - h if side == 0 else y + th
                    c = x + int(rng.integers(1-w, tw))
                else:
                    c = x - w if side == 2 else x + tw
                    r = y + int(rng.integers(1-h, th))
            cells = {(r+dy, c+dx) for dy in range(h) for dx in range(w)}
            if not occupied.isdisjoint(cells):
                continue
            union = occupied | cells
            y0, x0, y1, x1 = _box(union)
            if y1-y0 <= 9 and x1-x0 <= 11:
                options.append((r, c, cells))
        if not options:
            return None
        r, c, cells = options[int(rng.integers(len(options)))]
        rects.append((r, c, h, w))
        occupied.update(cells)
    r0, c0, r1, c1 = _box(occupied)
    if min(r1-r0, c1-c0) < 3:
        return None
    # Independent margins inside the work frame prevent bbox-only inference.
    pads = [int(rng.integers(0, 2)) for _ in range(4)]
    shape = r1-r0+pads[0]+pads[1], c1-c0+pads[2]+pads[3]
    mask = np.zeros(shape, dtype=bool)
    normalized = []
    for r, c, h, w in rects:
        r, c = r-r0+pads[0], c-c0+pads[2]
        mask[r:r+h, c:c+w] = True
        normalized.append((r, c, h, w))
    return mask, normalized


def _choose_clues(rng, mask, target, base):
    candidates = []
    for axis in (0, 1):
        for line in range(mask.shape[axis]):
            for color in sorted(set(int(v) for v in target[mask])):
                clue = dict(axis=axis, line=line, color=color)
                q = _count(target, mask, clue)
                if not 1 <= q <= 6 or q == int(_line_mask(mask, clue).sum()):
                    continue
                clue["q"] = q
                rejects = {i for i, image in enumerate(base) if _count(image, mask, clue) != q}
                if rejects:
                    candidates.append((clue, rejects))
    if not candidates:
        return None
    target_key = tuple(int(v) for v in target[mask])
    remaining = {i for i, im in enumerate(base) if tuple(int(v) for v in im[mask]) != target_key}
    chosen = []
    # Begin with an informative non-singleton count; arrangement cannot stand in
    # for cell count. Further clues greedily separate the remaining renderings.
    while remaining and len(chosen) < 4:
        pool = [(c, rej) for c, rej in candidates
                if (c["axis"], c["line"]) not in {(x["axis"], x["line"]) for x in chosen}
                and rej & remaining and (chosen or c["q"] >= 2)]
        if not pool:
            return None
        scores = [len(rej & remaining) for _, rej in pool]
        high = max(scores)
        good = [item for item, score in zip(pool, scores) if score >= max(1, high * .8)]
        clue, rejects = good[int(rng.integers(len(good)))]
        chosen.append(clue)
        remaining -= rejects
    if remaining:
        return None
    if len(chosen) < 2 or len({c["q"] for c in chosen}) < 2:
        pool = [c for c, _ in candidates if c["q"] not in {x["q"] for x in chosen}
                and (c["axis"], c["line"]) not in {(x["axis"], x["line"]) for x in chosen}]
        if not pool or len(chosen) == 4:
            return None
        chosen.append(pool[int(rng.integers(len(pool)))])
    queried = np.zeros_like(mask)
    for clue in chosen:
        queried |= _line_mask(mask, clue)
    if not np.any(mask & ~queried):
        return None
    rng.shuffle(chosen)
    return chosen


def _arrow_shape(rng, q, direction, profile=None):
    radius = int(rng.integers(1, 4)) if profile is None else profile[0]
    stem = int(rng.integers(0, 3)) if profile is None else profile[1]
    # Vary reach and depth independently, subject only to payload capacity.
    tail = int(rng.integers(1, 4)) if profile is None else profile[2]
    while sum(2*d-1 for d in range(1, radius+1)) + tail*(2*radius-1) < q+1:
        tail += 1
    frame = {(y, 0) for y in range(stem+1)}
    cavity = set()
    for y in range(stem+1, stem+radius+tail+1):
        reach = min(y-stem, radius)
        frame.update(((y, -reach), (y, reach)))
        cavity.update((y, x) for x in range(-reach+1, reach))
    return ({_world(p, direction) for p in frame},
            {_world(p, direction) for p in cavity})


def _arrange_arrows(rng, mask, clues, profile=None):
    h, w = mask.shape
    board = _box_cells((0, 0, h+2, w+2))
    for _ in range(45):
        occupied = set(board)
        arrows = []
        for clue in clues:
            if clue["axis"] == 0:
                direction = str(rng.choice(["L", "R"]))
                gap = int(rng.integers(2, 5))
                tip = (1+clue["line"], -gap if direction == "R" else w+1+gap)
            else:
                direction = str(rng.choice(["U", "D"]))
                gap = int(rng.integers(2, 5))
                tip = (-gap if direction == "D" else h+1+gap, 1+clue["line"])
            frame, cavity = _arrow_shape(rng, clue["q"], direction, profile)
            frame = {(r+tip[0], c+tip[1]) for r, c in frame}
            cavity = {(r+tip[0], c+tip[1]) for r, c in cavity}
            bbox = _box_cells(_box(frame))
            guard = {(r+dr, c+dc) for r, c in bbox for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if not occupied.isdisjoint(guard):
                break
            occupied.update(bbox)
            cells = sorted(cavity)
            if rng.random() < .5:
                # Compact count witness; many touching payload cells still count separately.
                start = cells[int(rng.integers(len(cells)))]
                cells.sort(key=lambda p: (abs(p[0]-start[0])+abs(p[1]-start[1]), p))
                payload = set(cells[:clue["q"]])
            else:
                payload = {cells[int(i)] for i in rng.choice(len(cells), clue["q"], replace=False)}
            arrows.append(dict(clue, frame=frame, cavity=cavity, payload=payload,
                               direction=direction, tip=tip))
        if len(arrows) == len(clues):
            box = _box(occupied)
            if box[2]-box[0] <= 26 and box[3]-box[1] <= 26:
                return arrows, occupied
    return None


def _scene(rng, mask, target, tiles, clues, bg, structure, profile=None):
    arranged = _arrange_arrows(rng, mask, clues, profile)
    if arranged is None:
        return None
    arrows, occupied = arranged
    y0, x0, y1, x1 = _box(occupied)
    bh, bw = mask.shape
    for attempt in range(12):
        min_h, min_w = y1-y0+2, x1-x0+2
        height = int(rng.integers(max(18, min_h), 31))
        width = int(rng.integers(max(18, min_w), 31))
        dy = int(rng.integers(1, height-(y1-y0))) - y0
        dx = int(rng.integers(1, width-(x1-x0))) - x0
        used = {(r+dy, c+dx) for r, c in occupied}
        # Keep the visible corridors empty as well as object bounding boxes.
        for arrow in arrows:
            tr, tc = arrow["tip"]
            dr, dc = DIRS[arrow["direction"]]
            r, c = tr+dr, tc+dc
            while not (0 <= r < bh+2 and 0 <= c < bw+2):
                used.add((r+dy, c+dx))
                r, c = r+dr, c+dc
        placements = {}
        for i in sorted(range(len(tiles)), key=lambda i: -tiles[i].size):
            h, w = tiles[i].shape
            legal = []
            for r in range(1, height-h):
                for c in range(1, width-w):
                    guard = {(y, x) for y in range(r-1, r+h+1) for x in range(c-1, c+w+1)}
                    if used.isdisjoint(guard):
                        legal.append((r, c))
            if not legal:
                break
            r, c = legal[int(rng.integers(len(legal)))]
            placements[i] = r, c
            used.update((r+y, c+x) for y in range(h) for x in range(w))
        if len(placements) != len(tiles):
            continue
        inp = np.full((height, width), bg, dtype=np.int8)
        inp[dy:dy+bh+2, dx:dx+bw+2] = structure
        view = inp[dy+1:dy+bh+1, dx+1:dx+bw+1]
        view[mask] = bg
        for arrow in arrows:
            for r, c in arrow["frame"]:
                inp[r+dy, c+dx] = structure
            for r, c in arrow["payload"]:
                inp[r+dy, c+dx] = arrow["color"]
        for i, (r, c) in placements.items():
            h, w = tiles[i].shape
            inp[r:r+h, c:c+w] = tiles[i]
        out = inp.copy()
        view = out[dy+1:dy+bh+1, dx+1:dx+bw+1]
        view[mask] = target[mask]
        for arrow in arrows:
            for r, c in arrow["payload"]:
                out[r+dy, c+dx] = bg
        # Reflect/rotate the whole relation, including every source matrix.
        k = int(rng.integers(4))
        inp, out = np.rot90(inp, k), np.rot90(out, k)
        if rng.random() < .5:
            inp, out = inp[:, ::-1], out[:, ::-1]
        return {"input": inp.copy(), "output": out.copy()}
    return None


def _make_pair(rng, count, surplus, profile=None, force_asym=False):
    for _ in range(250):
        grown = _grow(rng, count)
        if grown is None:
            continue
        mask, rects = grown
        roles = [int(v) for v in rng.choice(10, size=int(rng.integers(4, 8)), replace=False)]
        bg, structure, *palette = roles
        tiles = [_pattern(rng, (h, w), palette) for r, c, h, w in rects]
        if force_asym:
            tiles[0].fill(palette[0])
            tiles[0][0, 0] = palette[1]
        if np.array_equal(tiles[0], tiles[1]):
            tiles[1] = np.full(tiles[1].shape, next(c for c in palette if c != int(tiles[0][0, 0])), dtype=np.int8)
        target = np.full(mask.shape, -1, dtype=np.int8)
        for (r, c, h, w), tile in zip(rects, tiles):
            target[r:r+h, c:c+w] = tile
        if surplus:
            h, w = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            tiles.append(_pattern(rng, (h, w), palette))
        base, done, _, _ = _covers(mask, tiles, limit=130, node_limit=6500)
        if not done or not 2 <= len(base) <= 120:
            continue
        clues = _choose_clues(rng, mask, target, [im for im, _ in base])
        if clues is None:
            continue
        pair = _scene(rng, mask, target, tiles, clues, bg, structure, profile)
        if pair is None:
            continue
        try:
            parsed = _parse(pair["input"])
            answers, _, _, _ = _covers(parsed["mask"], parsed["tiles"], parsed["arrows"])
            if (len(answers) != 1 or len(answers[0][1]) != count
                    or not np.array_equal(_render(parsed, answers[0][0]), pair["output"])):
                continue
            if not any(len(_components(a["payload"], True)) < a["q"] for a in parsed["arrows"]):
                continue
        except ValueError:
            continue
        return pair
    raise RuntimeError("bounded mosaic construction exhausted")


def _counts(num_train, num_test):
    for value, lower, upper in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lower <= value <= upper:
            raise ValueError("illegal train/test count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    allow_surplus = bool(rng.integers(2))
    # A matched visible geometry witnesses different quantities even at (2,1).
    anchor_profile = (2, int(rng.integers(0, 3)), int(rng.integers(2, 4)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, n in (("train", num_train), ("test", num_test)):
        for i in range(n):
            count = int(rng.integers(3, 5) if split == "train" else rng.integers(5, 7))
            surplus = allow_surplus and (split == "train" and i == 0 or rng.random() < .4)
            for _ in range(10):
                pair = _make_pair(rng, count, surplus,
                                  anchor_profile if split == "train" and i < 2 else None,
                                  force_asym=split == "train" and i == 0)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _features(pair):
    parsed = _parse(pair["input"])
    answers, _, _, groups = _covers(parsed["mask"], parsed["tiles"], parsed["arrows"])
    assert len(answers) == 1
    mosaic, path = answers[0]
    assert np.array_equal(_render(parsed, mosaic), pair["output"])
    mask = parsed["mask"]
    by_shape = {}
    for tile in parsed["tiles"]:
        by_shape.setdefault(tile.shape, set()).add(_key(tile))
    assert any(len(patterns) > 1 for patterns in by_shape.values()), "missing same-footprint pattern alternatives"
    base, done, _, _ = _covers(mask, parsed["tiles"], limit=130, node_limit=20000)
    assert done and len(base) >= 2, "arrows do not affect packing"
    assert any(any(_count(im, mask, a) != a["q"] for a in parsed["arrows"]) for im, _ in base)
    queried = np.zeros_like(mask)
    for arrow in parsed["arrows"]:
        queried |= _line_mask(mask, arrow)
        assert _count(mosaic, mask, arrow) == arrow["q"]
        assert 0 < arrow["q"] < int(_line_mask(mask, arrow).sum())
    free = list(map(tuple, np.argwhere(mask & ~queried)))
    assert free, "line clues alone already specify every cell"
    # Concrete B-ablation: a changed unqueried cell still satisfies every arrow,
    # but cannot be made from the available physical rectangles.
    base_keys = {tuple(int(v) for v in im[mask]) for im, _ in base}
    found = False
    colors = sorted({int(c) for tile in parsed["tiles"] for c in tile.flat})
    for p in free:
        for color in colors:
            if mosaic[p] == color:
                continue
            changed = mosaic.copy()
            changed[p] = color
            if tuple(int(v) for v in changed[mask]) not in base_keys:
                assert all(_count(changed, mask, a) == a["q"] for a in parsed["arrows"])
                found = True
                break
        if found:
            break
    assert found, "missing inventory counterexample"
    return dict(shape=parsed["grid"].shape, board=mask.shape,
                mask=tuple(tuple(int(v) for v in row) for row in mask),
                quantities={a["q"] for a in parsed["arrows"]},
                directions={a["direction"] for a in parsed["arrows"]},
                arrows=len(parsed["arrows"]), sources=len(parsed["tiles"]),
                used=len(path), surplus=len(parsed["tiles"])-len(path),
                patterns={_key(t) for t in parsed["tiles"]},
                dimensions={t.shape for t in parsed["tiles"]},
                background=parsed["bg"], structure=parsed["structure"],
                alternatives=len(base),
                asymmetric=any(not np.array_equal(groups[gi][0], groups[gi][0][::-1, ::-1]) for gi, _, _ in path),
                profiles={tuple(sorted((y-min(p[0] for p in local), x-min(p[1] for p in local)) for y, x in local))
                          for a in parsed["arrows"]
                          for local in [[_local(p, a["direction"]) for p in a["frame"]]]},
                component_witness=any(len(_components(a["payload"], True)) < a["q"] for a in parsed["arrows"]))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, feats = set(), {}
    for split in ("train", "test"):
        feats[split] = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            f = _features({"input": inp, "output": out})
            assert len(f["quantities"]) >= 2 and f["component_witness"]
            feats[split].append(f)
    assert max(f["used"] for f in feats["train"]) < min(f["used"] for f in feats["test"])
    assert any(f["asymmetric"] for f in feats["train"])
    assert feats["train"][0]["profiles"] & feats["train"][1]["profiles"], "no matched training arrow geometry"
    if any(f["surplus"] for f in feats["test"]):
        assert any(f["surplus"] for f in feats["train"])


def self_test() -> dict:
    signatures, shapes, masks, dims, dirs, quantities = set(), set(), set(), set(), set(), set()
    backgrounds, structures, alternatives, source_counts, clue_counts = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(f["shape"]); masks.add(f["mask"]); dims.update(f["dimensions"])
                dirs.update(f["directions"]); quantities.update(f["quantities"])
                backgrounds.add(f["background"]); structures.add(f["structure"])
                alternatives.add(f["alternatives"]); source_counts.add(f["sources"]); clue_counts.add(f["arrows"])
    assert len(signatures) == 42 and len(shapes) >= 30 and len(masks) >= 100
    assert dirs == set(DIRS) and len(quantities) >= 4 and len(dims) >= 10
    assert len(backgrounds) == len(structures) == 10
    assert len(alternatives) >= 5 and len(source_counts) >= 4 and len(clue_counts) >= 2
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            _counts(a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(71, 2, 1), generate_puzzle(71, 2, 1)
    assert all(np.array_equal(p[role], q[role]) for s in a for p, q in zip(a[s], b[s]) for role in p)
    # Finite inventory, differing same-size patterns, and actual count constraints.
    mask = np.ones((2, 4), dtype=bool)
    tile1, tile2 = np.array([[2, 2], [2, 3]]), np.array([[3, 3], [2, 3]])
    assert not _covers(mask, [tile1])[0]
    base = _covers(mask, [tile1, tile2], limit=10)[0]
    assert len(base) == 2
    clue = dict(axis=1, line=0, color=2, q=2)
    assert len(_covers(mask, [tile1, tile2], [clue])[0]) == 1
    return dict(seeds=42, distinct_puzzles=len(signatures), scene_shapes=len(shapes),
                vacancy_masks=len(masks), tile_dimensions=sorted(dims),
                directions=sorted(dirs), quantities=sorted(quantities),
                alternative_counts=sorted(alternatives), source_counts=sorted(source_counts),
                clue_counts=sorted(clue_counts))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
