"""A route-controlled stencil printer. Standard library and NumPy only.

Generation writes blocks from sampled geometry; solve reads only the supplied
input and renders a decoded direction/ink field. No parent generator is used.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (
    0, 395672552, 706246995, 691815602, 442316020, 12400065, 498794453,
    793955488, 618009258, 310201840, 345269321, 180620597, 48985254,
    223702810, 626278736, 535422622, 990295624, 547322451, 262700630,
    139440375, 109887604, 356605418, 455866442, 147663854, 264835842,
    188680320, 107383491, 658680570, 880923791, 149184775, 387497027,
    161111372, 899901057, 598774284, 210689460, 208847839, 921418516,
    5394171, 369105861, 763221021, 456830163, 721472061,
)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
               if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in offsets:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    ys, xs = zip(*cells)
    return min(ys), min(xs), max(ys) + 1, max(xs) + 1


def _touches_sides(mask):
    return all(np.any(x) for x in (mask[0], mask[-1], mask[:, 0], mask[:, -1]))


def _parse_candidate(a, bg, carrier, box):
    y, x, yy, xx = box
    board = a[y + 1:yy - 1, x + 1:xx - 1].copy()
    assert 3 <= board.shape[0] <= 10 and 3 <= board.shape[1] <= 10
    assert not np.any(board == carrier)
    stations = [(int(r), int(c), int(board[r, c]))
                for r, c in zip(*np.nonzero(board != bg))]
    assert 3 <= len(stations) <= 12
    labels = {p[2] for p in stations}
    assert len(labels) >= 3 and not labels & {bg, carrier}
    assert all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 2
               for i, p in enumerate(stations) for q in stations[i + 1:])
    remaining = a.copy()
    remaining[y:yy, x:xx] = bg
    components = _components(remaining != bg, diagonal=True)
    singles = [p[0] for p in components if len(p) == 1]
    motifs = [p for p in components if len(p) > 1]
    assert len(motifs) == 1 and 3 <= len(singles) <= 10
    sy, sx, syy, sxx = _box(motifs[0])
    s = syy - sy
    assert 2 <= s <= 5 and sxx - sx == s
    stamp = a[sy:syy, sx:sxx].copy()
    assert np.any(stamp == carrier)
    payload = set(map(int, np.unique(stamp))) - {bg, carrier}
    assert len(payload) >= 2 and not payload & labels
    assert _touches_sides(stamp != bg)
    assert board.shape[0] * s <= 30 and board.shape[1] * s <= 30
    # Full blank separation between the two rectangular encodings.
    assert yy < sy or syy < y or xx < sx or sxx < x
    singles.sort(key=lambda p: p[1])
    ky, kx = singles[0]
    assert all(r == ky for r, c in singles)
    assert all(b[1] - p[1] == 2 for p, b in zip(singles, singles[1:]))
    assert ky >= max(yy, syy) + 1
    key = [int(a[r, c]) for r, c in singles]
    assert set(key) <= labels
    rebuilt = np.full(a.shape, bg, dtype=np.int8)
    rebuilt[y:yy, x:xx] = carrier
    rebuilt[y + 1:yy - 1, x + 1:xx - 1] = board
    rebuilt[sy:syy, sx:sxx] = stamp
    rebuilt[ky, kx:kx + 2 * len(key):2] = key
    assert np.array_equal(a, rebuilt), "unaccounted source pixels"
    return dict(background=bg, carrier=carrier, board=board,
                stations=stations, key=key, stamp=stamp, payload=payload,
                map_box=box, stamp_box=(sy, sx, syy, sxx), key_y=ky, key_x=kx)


def _read_input(value):
    a = _grid(value)
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in
               (a[0], a[-1], a[:, 0], a[:, -1])), "background rim required"
    candidates = []
    for carrier in sorted(set(map(int, np.unique(a))) - {bg}):
        for cells in _components(a == carrier):
            y, x, yy, xx = _box(cells)
            h, w = yy - y, xx - x
            if h < 5 or w < 5 or len(cells) != 2 * h + 2 * w - 4:
                continue
            b = a[y:yy, x:xx]
            if not all(np.all(side == carrier) for side in
                       (b[0], b[-1], b[:, 0], b[:, -1])):
                continue
            try:
                candidates.append(_parse_candidate(a, bg, carrier, (y, x, yy, xx)))
            except AssertionError:
                continue
    assert len(candidates) == 1, "map, stencil and key must have one reading"
    return candidates[0]


def _visible_graph(stations):
    graph = [[] for _ in stations]
    for i, a in enumerate(stations):
        for j, b in enumerate(stations):
            if i == j:
                continue
            if a[0] == b[0]:
                lo, hi = sorted((a[1], b[1]))
                clear = not any(p[0] == a[0] and lo < p[1] < hi for p in stations)
            elif a[1] == b[1]:
                lo, hi = sorted((a[0], b[0]))
                clear = not any(p[1] == a[1] and lo < p[0] < hi for p in stations)
            else:
                continue
            if clear:
                graph[i].append(j)
    return graph


def _route_solutions(graph, labels, key, limit=2):
    candidates = [[i for i, label in enumerate(labels) if label == k] for k in key]
    answers = []
    def visit(path, used):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for i in candidates[len(path)]:
            if i not in used and (not path or i in graph[path[-1]]):
                visit(path + [i], used | {i})
    visit([], set())
    return answers


def _movement(a, b):
    if a[0] == b[0] and a[1] != b[1]:
        dc = 1 if b[1] > a[1] else -1
        return 0, dc, 0 if dc == 1 else 2
    assert a[1] == b[1] and a[0] != b[0]
    dr = 1 if b[0] > a[0] else -1
    return dr, 0, 1 if dr == 1 else 3


def _segment_cells(a, b):
    dr, dc, direction = _movement(a, b)
    r, c = a[:2]
    cells = []
    while (r, c) != tuple(b[:2]):
        cells.append((r, c))
        r, c = r + dr, c + dc
    return cells, direction


def _field(info, target_ink=False, gap_only=False):
    board = info["board"]
    directions = np.full(board.shape, -1, dtype=np.int8)
    inks = np.full(board.shape, info["background"], dtype=np.int8)
    stations, route = info["stations"], info["route"]
    for u, v in zip(route, route[1:]):
        a, b = stations[u], stations[v]
        cells, direction = _segment_cells(a, b)
        for r, c in cells:
            assert 0 <= r < board.shape[0] and 0 <= c < board.shape[1]
            assert directions[r, c] == -1, "route segments intersect or overlap"
            if gap_only and (r, c) == a[:2]:
                continue
            directions[r, c] = direction
            inks[r, c] = b[2] if target_ink else a[2]
    end = stations[route[-1]]
    assert directions[end[0], end[1]] == -1
    return directions, inks


def _decode(value):
    info = _read_input(value)
    graph = _visible_graph(info["stations"])
    answers = _route_solutions(graph, [p[2] for p in info["stations"]], info["key"])
    assert len(answers) == 1, "the key must determine one distinct-checkpoint route"
    info["route"], info["graph"] = answers[0], graph
    info["directions"], info["inks"] = _field(info)
    return info


def _render_field(info, directions=None, inks=None, binding=True, fill_holes=False):
    directions = info["directions"] if directions is None else directions
    inks = info["inks"] if inks is None else inks
    s = info["stamp"].shape[0]
    rr, cc = np.indices((directions.shape[0] * s, directions.shape[1] * s))
    motion = directions[rr // s, cc // s]
    ink = inks[rr // s, cc // s]
    out = np.full(rr.shape, info["background"], dtype=np.int8)
    for direction in range(4):
        stamp = np.rot90(info["stamp"], -direction)
        pixels = stamp[rr % s, cc % s]
        if fill_holes:
            pixels = np.where(pixels == info["background"], info["carrier"], pixels)
        if binding:
            pixels = np.where(pixels == info["carrier"], ink, pixels)
        out[motion == direction] = pixels[motion == direction]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible input, with no seed or generation state."""
    return _render_field(_decode(input_grid))


def _write_output(shape, stations, path, stamp, bg, carrier):
    """Generation's direct walk/block writes, independent of pixel indexing."""
    s = stamp.shape[0]
    out = np.full((shape[0] * s, shape[1] * s), bg, dtype=np.int8)
    visited = set()
    for u, v in zip(path, path[1:]):
        a, b = stations[u], stations[v]
        dr, dc, direction = _movement(a, b)
        tile = np.rot90(stamp, -direction).copy()
        tile[tile == carrier] = a[2]
        r, c = a[:2]
        while (r, c) != b[:2]:
            assert (r, c) not in visited
            visited.add((r, c))
            out[r * s:(r + 1) * s, c * s:(c + 1) * s] = tile
            r, c = r + dr, c + dc
    return out


def _asymmetric(stamp):
    return all(not np.array_equal(stamp, np.rot90(stamp, k)) for k in (1, 2, 3)) and all(
        not np.array_equal(stamp, np.fliplr(np.rot90(stamp, k))) for k in range(4))


def _make_stamp(rng, s, bg, carrier, payload, witness=False):
    colors = np.array([carrier] + list(payload), dtype=np.int8)
    r, c = np.indices((s, s))
    families = ("bands", "rings", "diagonal", "random", "random", "symmetric")
    family = "random" if witness else str(rng.choice(families))
    if family == "symmetric" and s >= 3:
        # Deliberate dihedral symmetry: carrier edges, payload corners/interior.
        # Two payloads suffice even if a larger pool was available for this pair.
        stamp = np.full((s, s), carrier, dtype=np.int8)
        stamp[1:-1, 1:-1] = payload[1]
        stamp[0, 0] = stamp[0, -1] = stamp[-1, 0] = stamp[-1, -1] = payload[0]
        if s >= 5 and rng.random() < .5:
            stamp[s // 2, s // 2] = bg
        return stamp
    for _ in range(80):
        if family == "bands":
            idx = r if rng.random() < .5 else c
        elif family == "rings":
            idx = np.minimum.reduce([r, c, s - 1 - r, s - 1 - c])
        elif family == "diagonal":
            idx = r + c * int(rng.integers(1, len(colors)))
        elif family == "symmetric" and s >= 3:
            a, b = np.minimum(r, s - 1 - r), np.minimum(c, s - 1 - c)
            idx = np.minimum(a, b) + 2 * np.maximum(a, b)
        else:
            idx = rng.integers(len(colors), size=(s, s))
        stamp = colors[(idx + int(rng.integers(len(colors)))) % len(colors)].copy()
        # Missing colors are inserted at random positions, never in a palette row.
        for color, pos in zip(colors, rng.permutation(s * s)):
            stamp.flat[int(pos)] = color
        holes = int(rng.integers(1 if witness else 0, max(2, s * s // 4 + 1)))
        for pos in rng.choice(s * s, holes, replace=False):
            stamp.flat[int(pos)] = bg
        if not set(map(int, colors)) <= set(map(int, np.unique(stamp))):
            continue
        if not _touches_sides(stamp != bg) or len(_components(stamp != bg, True)) != 1:
            continue
        if witness and not _asymmetric(stamp):
            continue
        return stamp
    # Guaranteed visible asymmetric pattern, including one edge hole.
    stamp = np.full((s, s), carrier, dtype=np.int8)
    stamp[0, 0], stamp[-1, -1] = payload[:2]
    stamp[0, -1] = bg
    if len(payload) > 2:
        stamp[-1, 0] = payload[2]
    if s >= 3:
        stamp[1, 0] = payload[1]
    assert not witness or _asymmetric(stamp)
    return stamp


def _axis(rng, n, limit, minimum_extent=0):
    gaps = np.full(n - 1, 2, dtype=int)
    spare = limit - (1 + int(gaps.sum()))
    for i in rng.permutation(n - 1):
        extra = int(rng.integers(0, min(2, spare) + 1))
        gaps[i] += extra
        spare -= extra
    while 1 + int(gaps.sum()) < minimum_extent:
        choices = np.flatnonzero(gaps < 4)
        i = int(rng.choice(choices))
        gaps[i] += 1
    return [0] + list(map(int, np.cumsum(gaps)))


def _find_path(rng, cells, graph, length, all_directions=False):
    budget = 5000
    def visit(path, printed, directions):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            return path if not all_directions or len(directions) == 4 else None
        for nxt in rng.permutation(graph[path[-1]]):
            nxt = int(nxt)
            if nxt in path:
                continue
            segment, direction = _segment_cells(cells[path[-1]], cells[nxt])
            if printed.intersection(segment):
                continue
            found = visit(path + [nxt], printed | set(segment), directions | {direction})
            if found is not None:
                return found
        return None
    for start in rng.permutation(len(cells)):
        found = visit([int(start)], set(), set())
        if found is not None:
            return found
        if budget < 0:
            break
    return None


def _assign_labels(rng, cells, graph, path, pool, repeat=False):
    length = len(path)
    cap = min(len(pool), length - int(repeat))
    k = int(rng.integers(3, cap + 1))
    palette = list(map(int, rng.choice(pool, k, replace=False)))
    unused = [i for i in range(len(cells)) if i not in path]
    for _ in range(100):
        key = palette + list(map(int, rng.choice(palette, length - k)))
        key = list(map(int, rng.permutation(key)))
        if key[0] == key[-1]:
            continue
        labels = [0] * len(cells)
        for i, color in zip(path, key):
            labels[i] = color
        absent = [int(c) for c in pool if c not in key]
        for i in unused:
            labels[i] = int(rng.choice(absent if absent and rng.random() < .4 else key))
        solutions = _route_solutions(graph, labels, key)
        if len(solutions) == 1 and solutions[0] == tuple(path):
            return labels, key
    return None


def _compose_input(rng, board, stamp, key, bg, carrier):
    framed = np.pad(board, 1, constant_values=carrier)
    mh, mw = framed.shape
    sh, sw = stamp.shape
    horizontal, reverse = bool(rng.integers(2)), bool(rng.integers(2))
    gap = int(rng.integers(1, 5))
    bh = max(mh, sh) if horizontal else mh + sh + gap
    bw = mw + sw + gap if horizontal else max(mw, sw)
    min_h, min_w = bh + 4, max(bw + 2, 2 * len(key) + 1)
    roomy = rng.random() < .3
    h = int(rng.integers(min_h, (30 if roomy else min(30, min_h + 5)) + 1))
    w = int(rng.integers(min_w, (30 if roomy else min(30, min_w + 5)) + 1))
    top = int(rng.integers(1, h - bh - 2))
    left = int(rng.integers(1, w - bw))
    if horizontal:
        my = top + int(rng.integers(bh - mh + 1))
        sy = top + int(rng.integers(bh - sh + 1))
        mx, sx = (left + sw + gap, left) if reverse else (left, left + mw + gap)
    else:
        my, sy = (top + sh + gap, top) if reverse else (top, top + mh + gap)
        mx = left + int(rng.integers(bw - mw + 1))
        sx = left + int(rng.integers(bw - sw + 1))
    ky = int(rng.integers(top + bh + 1, h - 1))
    kx = int(rng.integers(1, w - (2 * len(key) - 1)))
    a = np.full((h, w), bg, dtype=np.int8)
    a[my:my + mh, mx:mx + mw] = framed
    a[sy:sy + sh, sx:sx + sw] = stamp
    a[ky, kx:kx + 2 * len(key):2] = key
    return a


def _sample_pair(rng, params, split, index):
    train = split == "train"
    anchor = train and index < 2
    palette = params["palette"] if params["stable"] else list(map(int, rng.permutation(10)))
    bg, carrier = palette[:2]
    # A 2x2 motif needs room for ink, two payloads, and optional background.
    s = (3 if index == 0 else 2) if anchor else int(rng.choice(
        (2, 3) if train or params["shift"] == "map" else (4, 5)))
    payload_count = 2 if s == 2 else int(rng.integers(2, 4))
    payload, pool = palette[2:2 + payload_count], palette[2 + payload_count:]
    stamp = _make_stamp(rng, s, bg, carrier, payload, witness=anchor and index == 0)
    limit = 7 if train else (10 if params["shift"] == "map" else 30 // s)
    for _ in range(64):
        if anchor and index == 0:
            nr = nc = 3
        elif train or params["shift"] == "stamp":
            nr, nc = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        elif rng.random() < .5:
            nr, nc = 4, int(rng.integers(2, 5))
        else:
            nr, nc = int(rng.integers(2, 5)), 4
        map_test = not train and params["shift"] == "map"
        ys = _axis(rng, nr, limit, 8 if map_test and nr == 4 else 0)
        xs = _axis(rng, nc, limit, 8 if map_test and nc == 4 else 0)
        universe = [(r, c) for r in ys for c in xs]
        unused = int(rng.integers(1, 3)) if anchor and index == 1 else int(rng.integers(0, 4))
        low = 5 if anchor and index == 0 else 4 if anchor or map_test else 3
        high = min(7 if anchor else 9 if train else 10, len(universe) - unused, 12 - unused)
        if high < low:
            continue
        length = int(rng.integers(low, high + 1))
        cells = sorted(universe[int(i)] for i in rng.choice(len(universe), length + unused, replace=False))
        if map_test and not (
            len({r for r, c in cells}) == 4 and max(r for r, c in cells) >= 7
            or len({c for r, c in cells}) == 4 and max(c for r, c in cells) >= 7
        ):
            continue
        graph = _visible_graph(cells)
        path = _find_path(rng, cells, graph, length, anchor and index == 0)
        if path is None:
            continue
        assignment = _assign_labels(rng, cells, graph, path, pool, anchor and index == 1)
        if assignment is None:
            continue
        labels, key = assignment
        h = max(3, max(r for r, c in cells) + 1)
        w = max(3, max(c for r, c in cells) + 1)
        if map_test and max(h, w) < 8:
            continue
        h += int(rng.integers(0, min(2, limit - h) + 1))
        w += int(rng.integers(0, min(2, limit - w) + 1))
        dy = int(rng.integers(0, h - max(r for r, c in cells)))
        dx = int(rng.integers(0, w - max(c for r, c in cells)))
        stations = [(r + dy, c + dx, color) for (r, c), color in zip(cells, labels)]
        break
    else:
        # An open five-waypoint spiral with a globally resolvable repeated label.
        q = 4 if not train and params["shift"] == "map" else 2
        cells = [(0, 0), (0, 2 * q), (2 * q, 2 * q), (2 * q, q), (q, q)]
        labels = [pool[0], pool[1], pool[2], pool[0], pool[3]]
        if not train and params["shift"] == "map":
            cells.append((2, q))
            labels.append(pool[4])
        path, key = list(range(len(cells))), labels.copy()
        if anchor and index == 1:
            cells.append((q, 0))
            labels.append(pool[1])
        stations = [(r, c, int(color)) for (r, c), color in zip(cells, labels)]
        h = w = 2 * q + 1
    board = np.full((h, w), bg, dtype=np.int8)
    for r, c, label in stations:
        board[r, c] = label
    a = _compose_input(rng, board, stamp, key, bg, carrier)
    out = _write_output(board.shape, stations, path, stamp, bg, carrier)
    return {"input": a, "output": out}


def _check_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _check_count(num_train, "num_train", 2, 10)
    nq = _check_count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    params = dict(palette=list(map(int, rng.permutation(10))), stable=bool(rng.integers(2)),
                  shift=str(rng.choice(("map", "stamp"))))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for _ in range(32):
                pair = _sample_pair(rng, params, split, index)
                signature = _signature(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(info, kind):
    directions, inks = info["directions"].copy(), info["inks"].copy()
    if kind == "no_a":
        directions[:] = -1
        for r, c, label in info["stations"]:
            directions[r, c] = 0
        return _render_field(info, directions, inks, binding=False)
    if kind == "no_b":
        s = info["stamp"].shape[0]
        coarse = np.where(directions >= 0, inks, info["background"])
        return np.repeat(np.repeat(coarse, s, axis=0), s, axis=1)
    if kind == "direction_blind":
        directions[directions >= 0] = 0
    elif kind == "binding_blind":
        return _render_field(info, binding=False)
    elif kind == "target_ink":
        directions, inks = _field(info, target_ink=True)
    elif kind == "gap_only":
        directions, inks = _field(info, gap_only=True)
    elif kind == "terminal_print":
        end = info["stations"][info["route"][-1]]
        previous = info["stations"][info["route"][-2]]
        directions[end[0], end[1]] = _movement(previous, end)[2]
        inks[end[0], end[1]] = end[2]
    elif kind == "holes_filled":
        return _render_field(info, fill_holes=True)
    else:
        raise ValueError(kind)
    return _render_field(info, directions, inks)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert _signature(a) not in seen
            seen.add(_signature(a))
            info = _decode(a)
            predicted = _render_field(info)
            assert np.array_equal(predicted, out), "incorrect stencil impressions"
            assert not np.array_equal(a, out)
            stamp, bg, carrier = info["stamp"], info["background"], info["carrier"]
            active = info["directions"] >= 0
            s = stamp.shape[0]
            assert out.shape == tuple(n * s for n in info["board"].shape)
            # Whole-pattern conservation, not just solver equality.
            assert not np.any(out == carrier)
            for color in info["payload"]:
                assert np.count_nonzero(out == color) == int(active.sum()) * np.count_nonzero(stamp == color)
            for color in {p[2] for p in info["stations"]}:
                assert np.count_nonzero(out == color) == np.count_nonzero(active & (info["inks"] == color)) * np.count_nonzero(stamp == carrier)
            expected_background = int((~active).sum()) * s * s + int(active.sum()) * np.count_nonzero(stamp == bg)
            assert np.count_nonzero(out == bg) == expected_background
            unused = set(range(len(info["stations"]))) - set(info["route"])
            for i in unused | {info["route"][-1]}:
                r, c, label = info["stations"][i]
                assert np.all(out[r * s:(r + 1) * s, c * s:(c + 1) * s] == bg)
            info["output"] = out
            infos[split].append(info)
    train, test = infos["train"], infos["test"]
    assert {2, 3} <= {i["stamp"].shape[0] for i in train}
    assert any(i["stamp"].shape == (3, 3) and _asymmetric(i["stamp"])
               and np.any(i["stamp"] == i["background"])
               and set(map(int, np.unique(i["directions"]))) >= {0, 1, 2, 3} for i in train)
    assert any(len(set(i["key"])) < len(i["key"])
               and len(i["stations"]) > len(i["route"]) for i in train)
    assert all(max(i["board"].shape) <= 7 and i["stamp"].shape[0] <= 3 for i in train)
    map_shift = all(max(i["board"].shape) >= 8 and i["stamp"].shape[0] <= 3 for i in test)
    stamp_shift = all(i["stamp"].shape[0] >= 4 for i in test)
    assert map_shift or stamp_shift, "missing structural train/test shift"
    if map_shift:
        assert all(any(len({p[axis] for p in i["stations"]}) >= 4
                           and max(p[axis] for p in i["stations"]) - min(p[axis] for p in i["stations"]) >= 7
                           for axis in (0, 1)) for i in test)
    for kind in ("no_a", "no_b", "direction_blind", "binding_blind", "target_ink",
                 "gap_only", "terminal_print", "holes_filled"):
        assert any(not np.array_equal(_alternative(i, kind), i["output"]) for i in train), kind
    for i in train[:2]:
        for kind in ("no_a", "no_b"):
            assert not np.array_equal(_alternative(i, kind), i["output"])


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _canonical(a):
    roles, values = {}, []
    for x in np.asarray(a).ravel():
        x = int(x)
        values.append(roles.setdefault(x, len(roles)))
    return a.shape, tuple(values)


def self_test():
    """Bounded 42-seed semantic/diversity audit plus interface/metamorphic checks."""
    puzzles, records = [], []
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                info = _decode(pair["input"])
                info["split"] = split
                info["input_shape"] = pair["input"].shape
                records.append(info)
        # Order counterfactual: exactly the same map/stencil and opposite key.
        pair = p["train"][0]
        info = _decode(pair["input"])
        a = pair["input"].copy()
        a[info["key_y"], info["key_x"]:info["key_x"] + 2 * len(info["key"]):2] = info["key"][::-1]
        backwards = _decode(a)
        assert backwards["route"] == info["route"][::-1]
        assert not np.array_equal(solve(a), pair["output"])
    assert _equal(puzzles[0], generate_puzzle(HOST_SEEDS[0]))
    assert len({tuple(_signature(pair["input"]) for s in ("train", "test") for pair in p[s]) for p in puzzles}) == 42
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(81731 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal example counts accepted")
    # Literal color identities have no semantic effect.
    permutation = np.array([4, 7, 0, 9, 1, 6, 2, 8, 5, 3], dtype=np.int8)
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    metrics = {
        "seeds": len(HOST_SEEDS), "pairs": len(records),
        "input_shapes": len({i["input_shape"] for i in records}),
        "map_shapes": len({i["board"].shape for i in records}),
        "layouts": len({(i["board"].shape, tuple(p[:2] for p in i["stations"])) for i in records}),
        "direction_fields": len({_signature(i["directions"]) for i in records}),
        "stencil_patterns": len({_canonical(i["stamp"]) for i in records}),
        "role_pairs": len({(i["background"], i["carrier"]) for i in records}),
        "stencil_sides": sorted({i["stamp"].shape[0] for i in records}),
        "route_lengths": sorted({len(i["route"]) for i in records}),
        "unused_counts": sorted({len(i["stations"]) - len(i["route"]) for i in records}),
        "map_growth_tests": sum(i["split"] == "test" and max(i["board"].shape) >= 8 for i in records),
        "stencil_growth_tests": sum(i["split"] == "test" and i["stamp"].shape[0] >= 4 for i in records),
    }
    for name, minimum in (("input_shapes", 35), ("map_shapes", 15), ("layouts", 100),
                          ("direction_fields", 100), ("stencil_patterns", 75), ("role_pairs", 25)):
        assert metrics[name] >= minimum, (name, metrics[name])
    assert metrics["stencil_sides"] == [2, 3, 4, 5]
    assert metrics["unused_counts"] == [0, 1, 2, 3]
    assert min(metrics["route_lengths"]) == 3 and max(metrics["route_lengths"]) >= 8
    assert metrics["map_growth_tests"] and metrics["stencil_growth_tests"]
    return metrics


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
