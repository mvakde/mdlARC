"""Color-selective stencil printing along terrain-guided cyclic staircases.

No parent modules, files, hidden generation state, or external dependencies are
used. Generation and diagnostics are bounded and do not write files.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and _integer(num_test), "counts must be integers")
    _require(2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test,
             "require 2..10 train, 1..4 test, and more train than test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _point(point, shape, code):
    r, c = point
    h, w = shape
    if code >= 4:
        c = w - 1 - c
    return ((r, c), (w - 1 - c, r), (h - 1 - r, w - 1 - c), (c, h - 1 - r))[code % 4]


def _transformed_box(box, shape, code):
    r0, r1, c0, c1 = box
    corners = [_point(p, shape, code) for p in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))]
    return min(p[0] for p in corners), max(p[0] for p in corners), min(p[1] for p in corners), max(p[1] for p in corners)


def _components(binary):
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    components = []
    for rr, cc in np.argwhere(binary):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(cells)
    return components


def _bbox(cells):
    return min(p[0] for p in cells), max(p[0] for p in cells), min(p[1] for p in cells), max(p[1] for p in cells)


def _perimeter(box):
    r0, r1, c0, c1 = box
    return ({(r0, c) for c in range(c0, c1 + 1)} |
            {(r1, c) for c in range(c0, c1 + 1)} |
            {(r, c0) for r in range(r0, r1 + 1)} |
            {(r, c1) for r in range(r0, r1 + 1)})


def _separated(a, b):
    return a[1] + 1 < b[0] or b[1] + 1 < a[0] or a[3] + 1 < b[2] or b[3] + 1 < a[2]


def _find_frames(g, bg):
    candidates = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        found = {}
        for cells in _components(g == color):
            box = _bbox(cells)
            r0, r1, c0, c1 = box
            if r1 - r0 < 3 or c1 - c0 < 3 or box in found:
                continue
            if any(g[p] != color for p in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
                continue
            if np.any(g[r0 + 1:r1, c0 + 1:c1] == color):
                continue
            perimeter = _perimeter(box)
            gates = sorted(p for p in perimeter if g[p] != color)
            if gates:
                if len({int(g[p]) for p in gates}) != 1 or g[gates[0]] == bg:
                    continue
                if not (all(p[0] == r0 for p in gates) or all(p[0] == r1 for p in gates) or
                        all(p[1] == c0 for p in gates) or all(p[1] == c1 for p in gates)):
                    continue
                if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) < 2 for a, b in zip(gates, gates[1:])):
                    continue
            found[box] = dict(box=box, perimeter=perimeter, gates=gates)
        closed = [v for v in found.values() if not v["gates"]]
        gated = [v for v in found.values() if v["gates"]]
        all_color = set(map(tuple, np.argwhere(g == color)))
        for m in gated:
            for s in closed:
                if _separated(m["box"], s["box"]) and all_color <= m["perimeter"] | s["perimeter"]:
                    candidates.append((color, m, s))
    _require(len(candidates) == 1, "need one closed stencil frame and one uniquely gated map frame")
    return candidates[0]


def _decode(input_grid):
    """Recover every semantic role from the input alone, in canonical E/N axes."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    rim = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    _require(bool(np.all(rim == bg)), "uniform background rim required")
    frame, board, stencil = _find_frames(g, bg)
    outside = g != bg
    for box in (board["box"], stencil["box"]):
        r0, r1, c0, c1 = box
        outside[r0:r1 + 1, c0:c1 + 1] = False
    swatches = []
    for cells in _components(outside):
        box = _bbox(cells)
        _require(len(cells) == 4 and box[1] - box[0] == 1 and box[3] - box[2] == 1,
                 "only separated 2x2 program swatches may lie outside the frames")
        colors = {int(g[p]) for p in cells}
        _require(len(colors) == 1, "program swatches must be solid")
        swatches.append((cells, colors.pop()))
    _require(2 <= len(swatches) <= 6, "program must have 2..6 entries")
    choices = []
    for code in range(8):
        mb = _transformed_box(board["box"], g.shape, code)
        gates = sorted(_point(p, g.shape, code) for p in board["gates"])
        if not all(c == mb[2] for r, c in gates):
            continue
        entries = []
        for cells, color in swatches:
            box = _bbox([_point(p, g.shape, code) for p in cells])
            entries.append((box[0], box[2], color))
        entries.sort(key=lambda x: x[1])
        if len({p[0] for p in entries}) != 1 or entries[0][0] + 1 >= mb[0]:
            continue
        if any(b[1] - a[1] < 3 for a, b in zip(entries, entries[1:])):
            continue
        sequence = tuple(p[2] for p in entries)
        if any(g[p] != sequence[0] for p in board["gates"]):
            continue
        choices.append((code, mb, gates, entries, sequence))
    _require(len(choices) == 1, "gate and legend sides must determine unique axes and reading order")
    code, mb, gates, entries, sequence = choices[0]
    canonical = _orient(g, code)
    sb = _transformed_box(stencil["box"], g.shape, code)
    r0, r1, c0, c1 = mb
    board_grid = canonical[r0 + 1:r1, c0 + 1:c1].copy()
    starts = tuple((r - r0 - 1, 0) for r, c in gates)
    first = sequence[0]
    _require(set(map(tuple, np.argwhere(board_grid == first))) == set(starts), "gates must align with exactly the marked starts")
    terrain_colors = set(map(int, np.unique(board_grid))) - {bg, first}
    _require(len(terrain_colors) == 1, "map must have one terrain color")
    terrain = terrain_colors.pop()
    _require(terrain != frame, "terrain and frame roles must differ")
    mask = (board_grid == terrain) | (board_grid == first)
    _require(mask.shape[1] >= 2 and all(mask[r, 1] for r, c in starts), "each gate needs an inward terrain neighbor")
    sr0, sr1, sc0, sc1 = sb
    stamp = canonical[sr0 + 1:sr1, sc0 + 1:sc1].copy()
    pigments = set(map(int, np.unique(stamp))) - {bg}
    program_colors = set(sequence)
    _require(len(program_colors) >= 2 and not ({bg, frame, terrain} & program_colors), "program roles must be distinct from terrain/frame/background")
    _require(not ({frame, terrain} & pigments) and program_colors <= pigments, "stencil must contain every program pigment")
    missing = pigments - program_colors
    _require(len(missing) == 1, "one stencil pigment omitted from program identifies collisions")
    _require(mask.shape[0] * stamp.shape[0] <= 30 and mask.shape[1] * stamp.shape[1] <= 30,
             "dimension products must not exceed 30")
    return dict(mask=mask, starts=starts, stamp=stamp, sequence=sequence,
                marker=missing.pop(), background=bg, terrain=terrain, frame=frame,
                orientation=code, map_box=mb, stencil_box=sb, entries=entries)


_DIRECTIONS = ((0, 1), (-1, 0))


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _trace(mask, start, edge_stop=True):
    """Maximal runs; edge_stop=False is used only for a negative ablation."""
    current, direction = start, 0
    path, collisions, legs = [], [], []
    for _ in range(sum(mask.shape) + 2):
        dr, dc = _DIRECTIONS[direction]
        r, c = current
        leg = []
        while _inside(mask.shape, (r, c)) and mask[r, c]:
            leg.append((r, c))
            r, c = r + dr, c + dc
        _require(bool(leg), "empty route leg")
        path.extend(leg)
        legs.append(len(leg))
        edge = not _inside(mask.shape, (r, c))
        if edge and edge_stop:
            end = "edge"
            break
        if not edge:
            collisions.append((r, c))
        direction = 1 - direction
        dr, dc = _DIRECTIONS[direction]
        current = (leg[-1][0] + dr, leg[-1][1] + dc)
        if not _inside(mask.shape, current) or not mask[current]:
            end = "edge" if edge else "blocked"
            break
    else:
        raise ValueError("route failed to terminate")
    _require(len(set(path)) == len(path), "route revisits a cell")
    return dict(path=tuple(path), collisions=tuple(collisions), legs=tuple(legs), end=end)


def _records(info):
    occupied, records = set(), []
    for start in info["starts"]:
        rec = _trace(info["mask"], start)
        footprint = set(rec["path"]) | set(rec["collisions"])
        _require(not footprint & occupied, "routes or collision footprints intersect")
        _require(all(not info["mask"][p] for p in rec["collisions"]), "collision must hit a wall")
        occupied |= footprint
        records.append(rec)
    return records


def _field(info, mode="correct"):
    word, bg = info["sequence"], info["background"]
    field = np.full(info["mask"].shape, bg, dtype=np.int8)
    if mode in ("all_terrain", "row_major"):
        for i, p in enumerate(map(tuple, np.argwhere(info["mask"]))):
            field[p] = word[i % len(word)] if mode == "row_major" else word[0]
        return field
    if mode == "deduplicate":
        word = tuple(dict.fromkeys(word))
    offset = 0
    for start, rec in zip(info["starts"], _records(info)):
        if mode == "turn_after_edge":
            rec = _trace(info["mask"], start, edge_stop=False)
        cursor = 0
        for li, length in enumerate(rec["legs"]):
            for j in range(length):
                index = j if mode == "reset_bend" else cursor + j
                if mode == "marker_tick":
                    index += li
                elif mode == "global_clock":
                    index += offset
                field[rec["path"][cursor + j]] = word[index % len(word)]
            cursor += length
        if mode != "omit_collisions":
            for p in rec["collisions"]:
                field[p] = info["marker"]
        offset += len(rec["path"])
    return field


def _render(info, field, style="layers", stamp=None):
    stamp = info["stamp"] if stamp is None else stamp
    sh, sw = stamp.shape
    bg = info["background"]
    out = np.full((field.shape[0] * sh, field.shape[1] * sw), bg, dtype=np.int8)
    for rr, cc in np.argwhere(field != bg):
        r, c = int(rr), int(cc)
        color = int(field[r, c])
        if style == "whole":
            block = stamp
        elif style == "solid":
            block = np.full(stamp.shape, color, dtype=np.int8)
        else:
            block = np.where(stamp == color, color, bg)
        out[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = block
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution, with no generator state or example lookup."""
    info = _decode(input_grid)
    out = _render(info, _field(info))
    # The decoder transformed the scene to canonical axes; invert that transform.
    code = info["orientation"]
    out = np.rot90(out, -(code % 4))
    return (np.fliplr(out) if code >= 4 else out).copy()


_ALTERNATIVES = ("deduplicate", "reset_bend", "marker_tick", "global_clock",
                 "turn_after_edge", "omit_collisions", "all_terrain", "row_major")
_REQUIRED = set(_ALTERNATIVES) | {"cycling", "unused_terrain", "whole_stamp", "solid_blocks",
                                 "holes", "stencil_orientation", "map_orientation"}


def _evidence(info):
    correct = _field(info)
    records = _records(info)
    evidence = {mode for mode in _ALTERNATIVES if not np.array_equal(correct, _field(info, mode))}
    if any(len(r["path"]) >= 2 * len(info["sequence"]) for r in records):
        evidence.add("cycling")
    if np.any(info["mask"] & (correct == info["background"])):
        evidence.add("unused_terrain")
    if np.any(info["stamp"] == info["background"]):
        evidence.add("holes")
    out = _render(info, correct)
    if not np.array_equal(out, _render(info, correct, "whole")):
        evidence.add("whole_stamp")
    if not np.array_equal(out, _render(info, correct, "solid")):
        evidence.add("solid_blocks")
    if all(not np.array_equal(out, _render(info, correct, stamp=_orient(info["stamp"], k))) for k in range(1, 8)):
        evidence.add("stencil_orientation")
    if all(not np.array_equal(out, _render(info, _orient(correct, k))) for k in range(1, 8)):
        evidence.add("map_orientation")
    return evidence


def _sample_word(rng, low=2, high=6, witness=False):
    for _ in range(24):
        n = int(rng.integers(3, 5)) if witness else int(rng.integers(low, high + 1))
        distinct = int(rng.integers(2, min(4, n - int(witness)) + 1))
        word = list(range(distinct)) + list(map(int, rng.integers(distinct, size=n - distinct)))
        rng.shuffle(word)
        unique = tuple(dict.fromkeys(word))
        if not witness or any(word[i % n] != unique[i % len(unique)] for i in range(2 * n)):
            return tuple(word)
    return (0, 0, 1)


def _asymmetric(a):
    return all(not np.array_equal(a, _orient(a, k)) for k in range(1, 8))


def _make_stamp(rng, shape, pigments, bg, witness):
    h, w = shape
    q = len(pigments)
    rr, cc = np.indices(shape)
    holes = witness or rng.random() < .75
    _require(h * w >= q + int(holes), "not enough stencil cells")
    for _ in range(32):
        family = int(rng.integers(6))
        if family == 0:
            labels = rr + int(rng.integers(q))
        elif family == 1:
            labels = cc + int(rng.integers(q))
        elif family == 2:
            labels = rr * int(rng.integers(1, q + 1)) + cc
        elif family == 3:
            labels = np.minimum.reduce((rr, cc, h - 1 - rr, w - 1 - cc))
        elif family == 4:
            anchors = [(int(rng.integers(h)), int(rng.integers(w))) for _ in range(q)]
            labels = np.argmin(np.stack([abs(rr - r) + abs(cc - c) for r, c in anchors]), axis=0)
        else:
            labels = rng.integers(q, size=shape)
        labels = np.broadcast_to(labels % q, shape).copy()
        stamp = np.asarray(pigments, dtype=np.int8)[labels]
        if holes:
            stamp[rng.random(shape) < float(rng.uniform(.08, .30))] = bg
        # Place one of each pigment at arbitrary cells, never a fixed palette row.
        positions = rng.permutation(h * w)[:q + int(holes)]
        for index, color in zip(positions, list(pigments) + ([bg] if holes else [])):
            stamp.flat[int(index)] = color
        if not witness or _asymmetric(stamp):
            return stamp
    raise RuntimeError("bounded asymmetric stencil construction exhausted")


def _planned_route(shape, startrow, columns, rows):
    """Alternating cumulative cut positions, not a call to either parent."""
    r, c = startrow, 0
    path, collisions, legs, false = [(r, c)], [], [], set()
    destinations = []
    for j in range(len(columns) + len(rows)):
        destinations.append((r, columns[j // 2]) if j % 2 == 0 else (rows[j // 2], c))
        r, c = destinations[-1]
    r, c = startrow, 0
    for j, (er, ec) in enumerate(destinations):
        old_count = len(path)
        dr, dc = _DIRECTIONS[j % 2]
        while (r, c) != (er, ec):
            r, c = r + dr, c + dc
            path.append((r, c))
        legs.append(len(path) - old_count + int(j == 0))
        ahead = r + dr, c + dc
        if _inside(shape, ahead):
            collisions.append(ahead)
            false.add(ahead)
        else:
            _require(j == len(destinations) - 1, "intermediate leg exits map")
    end = "edge" if not _inside(shape, ahead) else "blocked"
    if end == "blocked":
        dr, dc = _DIRECTIONS[len(destinations) % 2]
        turn = r + dr, c + dc
        if _inside(shape, turn):
            false.add(turn)
    return dict(path=tuple(path), collisions=tuple(collisions), legs=tuple(legs), end=end), false


def _random_plan(rng, shape, startrow, top):
    h, w = shape
    options = []
    for k in range(1, 7):
        ne, nn = (k + 1) // 2, k // 2
        vchoices = list(range(max(1, top) if k % 2 else top, startrow))
        if ne <= w - 2 and nn <= len(vchoices):
            options.append((k, vchoices))
    k, vchoices = options[int(rng.integers(len(options)))]
    ne, nn = (k + 1) // 2, k // 2
    edge = k % 2 == 1 and rng.random() < .45
    columns = sorted(map(int, rng.choice(np.arange(1, w - 1), size=ne - int(edge), replace=False)))
    if edge:
        columns.append(w - 1)
    rows = sorted(map(int, rng.choice(vchoices, size=nn, replace=False)), reverse=True)
    return _planned_route(shape, startrow, columns, rows)


def _make_map(rng, shape, count, witness):
    h, w = shape
    if witness:
        startrow = int(rng.integers(h - 4, h - 2))
        first_col = int(rng.integers(1, w - 2))
        upper_row = int(rng.integers(1, startrow))
        main, f1 = _planned_route(shape, startrow, [first_col, w - 1], [upper_row])
        secondary, f2 = _planned_route(shape, h - 1, [int(rng.integers(1, w - 1))], [])
        starts, plans = ((startrow, 0), (h - 1, 0)), [main, secondary]
        forced_false = f1 | f2
        extras = {(upper_row - 1, w - 1)}  # open corner past the real edge exit
    else:
        choices = [x for x in itertools.combinations(range(h), count)
                   if all(b - a >= 2 for a, b in zip(x, x[1:]))]
        _require(bool(choices), "not enough separated entry rows")
        selected = choices[int(rng.integers(len(choices)))]
        starts = tuple((r, 0) for r in selected)
        plans, forced_false, extras = [], set(), set()
        top = 0
        for r in selected:
            plan, reserved = _random_plan(rng, shape, r, top)
            plans.append(plan)
            forced_false |= reserved
            top = r + 2
    paths = {p for rec in plans for p in rec["path"]}
    _require(not forced_false & (paths | extras), "incompatible path reservations")
    occupied = set()
    for plan in plans:
        footprint = set(plan["path"]) | set(plan["collisions"])
        _require(not footprint & occupied, "planned footprint overlap")
        occupied |= footprint
    # Rectangular additions/carvings vary both used and unused portions of terrain.
    mask = np.full(shape, bool(rng.integers(2)), dtype=bool)
    for _ in range(int(rng.integers(2, 8))):
        r0, r1 = sorted(map(int, rng.choice(h + 1, size=2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(w + 1, size=2, replace=False)))
        mask[r0:r1, c0:c1] = bool(rng.integers(2))
    for p in forced_false:
        mask[p] = False
    for p in paths | extras:
        mask[p] = True
    if not any(mask[p] and p not in paths for p in np.ndindex(shape)):
        available = [p for p in np.ndindex(shape) if p not in forced_false | paths]
        _require(bool(available), "need at least one unused terrain cell")
        mask[available[int(rng.integers(len(available)))]] = True
    for start, plan in zip(starts, plans):
        _require(_trace(mask, start) == plan, "maximal trace differs from reserved guide")
    return mask, starts


def _shape_choices(regime, test, witness, pigment_count):
    ml, mu, sl, su = 4, 10, 2, 5
    if regime == "map_grow":
        ml, mu = (9, 10) if test else (4, 8)
    elif regime == "stamp_grow":
        sl, su = (4, 5) if test else (2, 3)
    elif regime == "starts" and test:
        ml = 6
    if witness:
        ml, mu, sl, su = 7, min(mu, 9), 3, 3
    choices = {}
    for mh in range(ml, mu + 1):
        for mw in range(ml, mu + 1):
            if regime == "aspect" and not ((mw > mh) if test else (mh > mw)):
                continue
            stamps = [(sh, sw) for sh in range(sl, su + 1) for sw in range(sl, su + 1)
                      if mh * sh <= 30 and mw * sw <= 30 and sh * sw >= pigment_count + 1]
            if stamps:
                choices[mh, mw] = stamps
    _require(bool(choices), "no compatible shapes")
    return choices


def _framed(interior, frame):
    h, w = interior.shape
    result = np.full((h + 2, w + 2), frame, dtype=np.int8)
    result[1:-1, 1:-1] = interior
    return result


def _layout(rng, info):
    bg, frame, word = info["background"], info["frame"], info["sequence"]
    board = _framed(np.where(info["mask"], info["terrain"], bg).astype(np.int8), frame)
    for r, c in info["starts"]:
        board[r + 1, :2] = word[0]
    stencil = _framed(info["stamp"], frame)
    bh, bw = board.shape
    sh, sw = stencil.shape
    gap = int(rng.integers(1, 3))
    reverse = bool(rng.integers(2))
    if rng.random() < .55:
        br, sr = map(int, rng.integers(0, 3, size=2))
        bc, sc = (sw + gap, 0) if reverse else (0, bw + gap)
    else:
        bc, sc = map(int, rng.integers(0, 3, size=2))
        br, sr = (sh + gap, 0) if reverse else (0, bh + gap)
    ph, pw = max(br + bh, sr + sh), max(bc + bw, sc + sw)
    lr, lc = map(int, rng.integers(1, 3, size=2))
    columns = [lc]
    for _ in word[1:]:
        columns.append(columns[-1] + 2 + int(rng.integers(1, 3)))
    origin_r = lr + 2 + int(rng.integers(1, 3))
    origin_c = int(rng.integers(1, 3))
    h = origin_r + ph + 1
    w = max(origin_c + pw, columns[-1] + 2) + 1
    _require(h <= 30 and w <= 30, "packed scene exceeds canvas")
    h += int(rng.integers(min(2, 30 - h) + 1))
    w += int(rng.integers(min(2, 30 - w) + 1))
    g = np.full((h, w), bg, dtype=np.int8)
    g[origin_r + br:origin_r + br + bh, origin_c + bc:origin_c + bc + bw] = board
    g[origin_r + sr:origin_r + sr + sh, origin_c + sc:origin_c + sc + sw] = stencil
    for c, color in zip(columns, word):
        g[lr:lr + 2, c:c + 2] = color
    return g


def _generate_pair(rng, regime, test, code, witness=False, previous=None):
    for _ in range(160):
        low, high = ((5, 6) if test else (2, 4)) if regime == "word" else (2, 6)
        labels = _sample_word(rng, low, high, witness)
        palette = list(map(int, rng.permutation(10)))
        bg, frame, terrain = palette[:3]
        word = tuple(palette[3 + x] for x in labels)
        cycle_colors = tuple(dict.fromkeys(word))
        marker = palette[3 + len(set(labels))]
        choices = _shape_choices(regime, test, witness, len(cycle_colors) + 1)
        if previous is not None:
            filtered = {m: [s for s in ss if s != previous[1]] for m, ss in choices.items() if m != previous[0]}
            choices = {m: ss for m, ss in filtered.items() if ss}
        maps = list(choices)
        mshape = maps[int(rng.integers(len(maps)))]
        stamps = choices[mshape]
        sshape = stamps[int(rng.integers(len(stamps)))]
        count = 2 if witness else (3 if test and regime == "starts" else int(rng.integers(1, 3)))
        try:
            mask, starts = _make_map(rng, mshape, count, witness)
            stamp = _make_stamp(rng, sshape, cycle_colors + (marker,), bg, witness)
            info = dict(mask=mask, starts=starts, stamp=stamp, sequence=word, marker=marker,
                        background=bg, frame=frame, terrain=terrain)
            if witness and not _REQUIRED <= _evidence(info):
                continue
            inp = _orient(_layout(rng, info), code)
            # Rendering here uses construction roles. solve/validation re-decode
            # the actual input and recover all roles independently of those roles.
            out = _orient(_render(info, _field(info)), code)
            decoded = _decode(inp)
            _require(np.array_equal(decoded["mask"], mask) and decoded["sequence"] == word,
                     "rendered input changed the meaning")
            return {"input": inp, "output": out}, (mshape, sshape)
        except ValueError:
            continue
    raise RuntimeError("bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one reproducible puzzle with an explicit structural test shift."""
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(("map_grow", "stamp_grow", "starts", "word", "aspect")))
    policy, base = int(rng.integers(3)), int(rng.integers(8))
    puzzle, seen, first_shape = {"train": [], "test": []}, set(), None
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            test = split == "test"
            code = int(rng.integers(8)) if policy == 1 else base
            if policy == 2 and test:
                code = (base // 4) * 4 + (base % 4 + 1) % 4
            for _ in range(12):
                pair, shapes = _generate_pair(rng, regime, test, code,
                                              witness=not test and i == 0,
                                              previous=first_shape if not test and i == 1 else None)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
            if not test and i == 0:
                first_shape = shapes
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _indexed_prediction(info, field):
    """Cell-index validation expression, separate from production block writes."""
    stamp, bg = info["stamp"], info["background"]
    sh, sw = stamp.shape
    rr, cc = np.indices((field.shape[0] * sh, field.shape[1] * sw))
    color = field[rr // sh, cc // sw]
    pigment = stamp[rr % sh, cc % sw]
    return np.where((color != bg) & (color == pigment), pigment, bg).astype(np.int8)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "exact train/test keys required")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence, train_shapes = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "exact input/output keys required")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _decode(inp)
            field = _field(info)
            canonical_out = _orient(out, info["orientation"])
            expected = _indexed_prediction(info, field)
            _require(np.array_equal(canonical_out, expected), "incorrect color-layer substitution")
            _require(not np.array_equal(inp, out), "nontrivial printing required")
            # Each active coarse label must print exactly its stencil channel.
            for color in set(info["sequence"]) | {info["marker"]}:
                count = int(np.count_nonzero(field == color)) * int(np.count_nonzero(info["stamp"] == color))
                _require(np.count_nonzero(out == color) == count, "channel population mismatch")
            _require(not np.any(np.isin(out, [info["terrain"], info["frame"]])), "terrain/frame leaked into output")
            if split == "train":
                evidence |= _evidence(info)
                train_shapes.append((info["mask"].shape, info["stamp"].shape))
    _require(_REQUIRED <= evidence, "missing training evidence: " + ",".join(sorted(_REQUIRED - evidence)))
    _require(len({m for m, s in train_shapes}) >= 2 and len({s for m, s in train_shapes}) >= 2,
             "training must vary both map and stencil shapes")


def _regressions():
    """Hand-authored color labels, including a bend, collision and real exit."""
    mask = np.array([[int(c) for c in row] for row in
                     ("0000100", "0000001", "0011111", "0010000", "1110000", "0000000", "1111000")], dtype=bool)
    stamp = np.array([[2, 2, 0], [3, 4, 4], [4, 3, 2]], dtype=np.int8)
    labels = np.array([[int(c) for c in row] for row in
                       ("0000000", "0040000", "0023222", "0020000", "2324000", "0000000", "2322400")], dtype=np.int8)
    board = _framed(mask.astype(np.int8), 9)
    board[5, :2] = board[7, :2] = 2
    inp = np.zeros((16, 18), dtype=np.int8)
    inp[5:14, 1:10] = board
    inp[6:11, 12:17] = _framed(stamp, 9)
    for c, color in zip((1, 4, 7, 10), (2, 3, 2, 2)):
        inp[1:3, c:c + 2] = color
    rr, cc = np.indices((21, 21))
    expected = np.where(labels[rr // 3, cc // 3] == stamp[rr % 3, cc % 3], stamp[rr % 3, cc % 3], 0)
    info = _decode(inp)
    _require(np.array_equal(_field(info), labels), "hand-authored trace regression")
    _require(info["mask"][1, 6] and _field(info)[1, 6] == 0, "must stop at canvas exit")
    for code in range(8):
        _require(np.array_equal(solve(_orient(inp, code)), _orient(expected, code)), "scene symmetry regression")
    changed = inp.copy()
    changed[7, 13], changed[7, 15] = changed[7, 15], changed[7, 13]
    _require(not np.array_equal(solve(inp), solve(changed)), "stencil geometry must affect answer")


def _pattern(a):
    colors = {}
    b = bytes(colors.setdefault(int(x), len(colors)) for x in a.ravel())
    return a.shape, hashlib.sha256(b).hexdigest()


def self_test():
    """Bounded 42-seed semantic, ablation, variation and API checks; no writes."""
    _regressions()
    structures, stencil_patterns, map_patterns, words, legs = set(), set(), set(), set(), set()
    input_shapes, output_shapes, map_shapes, stamp_shapes, orientations, starts, palettes = (set() for _ in range(7))
    puzzle_hashes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzle_hashes.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                _require(np.array_equal(solve(inp), out), "input-only solve mismatch")
                info = _decode(inp)
                structures.add(_pattern(inp))
                stencil_patterns.add(_pattern(info["stamp"]))
                map_patterns.add(_pattern(info["mask"]))
                input_shapes.add(inp.shape)
                output_shapes.add(out.shape)
                map_shapes.add(info["mask"].shape)
                stamp_shapes.add(info["stamp"].shape)
                orientations.add(info["orientation"])
                starts.add(len(info["starts"]))
                palettes.add((info["background"], info["frame"], info["terrain"]))
                words.add(_pattern(np.array([info["sequence"]], dtype=np.int8)))
                legs.update(len(r["legs"]) for r in _records(info))
    _require(len(puzzle_hashes) == 42 and len(structures) >= 180, "42-seed structural diversity")
    _require(len(map_patterns) >= 100 and len(stencil_patterns) >= 120, "map/stencil diversity")
    _require(len(input_shapes) >= 40 and len(output_shapes) >= 25, "shape diversity")
    _require(len(map_shapes) >= 15 and len(stamp_shapes) >= 8 and len(words) >= 15, "factor diversity")
    _require(len(orientations) == 8 and starts >= {1, 2, 3} and legs >= {1, 2, 3, 4}, "route diversity")
    _require(len(palettes) >= 80, "semantic palette diversity")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(8675309, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit count mismatch")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            continue
        raise AssertionError("illegal seed accepted")
    p = generate_puzzle(0)
    _require(_digest(p) == _digest(generate_puzzle(0)), "whole-puzzle determinism")
    # Both parent omissions are explicitly rejected by actual output validation.
    for style in ("whole", "solid"):
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        inp = damaged["train"][0]["input"]
        info = _decode(inp)
        out = _render(info, _field(info), style)
        code = info["orientation"]
        out = np.rot90(out, -(code % 4))
        damaged["train"][0]["output"] = np.fliplr(out).copy() if code >= 4 else out.copy()
        try:
            validate_generated_puzzle(damaged)
        except ValueError:
            continue
        raise AssertionError("incorrect stencil ablation accepted")
    return {"seeds": 42, "pairs": 210, "puzzles": len(puzzle_hashes),
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes),
            "map_shapes": len(map_shapes), "stencil_shapes": len(stamp_shapes),
            "structures": len(structures), "map_patterns": len(map_patterns),
            "stencil_patterns": len(stencil_patterns), "word_patterns": len(words),
            "orientations": sorted(orientations), "start_counts": sorted(starts),
            "leg_counts": sorted(legs), "palette_roles": len(palettes),
            "independent_solver": False, "blind_prediction": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
