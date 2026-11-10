"""Coordinate-constrained packing: candidate centers, not copy instructions.

Only NumPy and the standard library are used. Generation paints a construction;
the solver independently extracts the visible inventory and enumerates covers.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, todo = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.append(p)
                    todo.append(p)
        result.append(cells)
    return result


def _canonical_model(a, d):
    """Parse a left tray and right framed board; no output information is used."""
    H, W = a.shape
    if d < 2 or W - d - 1 < 5 or H < 5:
        raise ValueError("Panels too small")
    bg, divider = int(a[0, 0]), int(a[0, d])
    if not np.all(a[:, d] == divider):
        raise ValueError("No divider")
    tray, board = a[:, :d], a[:, d + 1:]
    frame = int(board[0, 0])
    if not (board[0, -1] == board[-1, 0] == board[-1, -1] == frame):
        raise ValueError("No frame corners")
    if not np.all(tray[[0, -1], :] == bg):
        raise ValueError("Tray end margins must show its background")
    interior = board[1:-1, 1:-1]
    colors = set(map(int, np.unique(interior)))
    if bg not in colors or len(colors) != 2:
        raise ValueError("Board must show background and vacancy")
    wall = next(iter(colors - {bg}))
    structural = {bg, divider, frame, wall}
    if len(structural) != 4:
        raise ValueError("Structural roles must differ")
    if any(np.any(tray == color) for color in structural - {bg}):
        raise ValueError("Structural colors cannot be tile payload")
    if not (np.array_equal(board[0, 1:-1], board[-1, 1:-1])
            and np.array_equal(board[1:-1, 0], board[1:-1, -1])):
        raise ValueError("Coordinate marks need aligned opposite partners")
    row_marks, col_marks = {}, {}
    for axis, values, dest in ((0, board[1:-1, 0], row_marks),
                               (1, board[0, 1:-1], col_marks)):
        for pos, color in enumerate(values, 1):
            color = int(color)
            if color == frame:
                continue
            if color in structural:
                raise ValueError("Invalid label color")
            dest.setdefault(color, []).append(pos)
    tiles = []
    for cells in _components(tray != bg):
        ys, xs = zip(*cells)
        y, x = min(ys), min(xs)
        h, w = max(ys) - y + 1, max(xs) - x + 1
        pattern = tray[y:y + h, x:x + w].copy()
        if h % 2 == 0 or w % 2 == 0 or len(cells) != h * w:
            raise ValueError("Tiles must be filled odd rectangles")
        if np.any(pattern == bg):
            raise ValueError("Tile has holes")
        tiles.append(dict(pattern=pattern, color=int(pattern[h // 2, w // 2]),
                          source=(y, x)))
    if not 1 <= len(tiles) <= 10:
        raise ValueError("Expected 1..10 physical tiles")
    groups, lookup = [], {}
    for tile in tiles:
        k = _key(tile["pattern"])
        if k in lookup:
            groups[lookup[k]]["count"] += 1
        else:
            lookup[k] = len(groups)
            groups.append(dict(pattern=tile["pattern"], color=tile["color"], count=1))
    return dict(board=board.copy(), tray=tray.copy(), bg=bg, wall=wall,
                frame=frame, divider=divider, rows=row_marks, cols=col_marks,
                mask=board == bg, tiles=tiles, groups=groups)


def _parse(input_grid):
    a = _grid(input_grid)
    models = []
    for turn in range(4):
        canonical = np.rot90(a, turn)
        for d in range(2, canonical.shape[1] - 5):
            if not np.all(canonical[:, d] == canonical[0, d]):
                continue
            try:
                m = _canonical_model(canonical, d)
            except ValueError:
                continue
            m["turn"] = turn
            models.append(m)
    if len(models) != 1:
        raise ValueError("Input must identify one framed board and inventory")
    return models[0]


def _placements(m, use_coordinates=True):
    coords = list(map(tuple, np.argwhere(m["mask"])))
    indices = {p: i for i, p in enumerate(coords)}
    candidates = []
    by_cell = [[] for _ in coords]
    H, W = m["board"].shape
    for gi, group in enumerate(m["groups"]):
        pattern, color = group["pattern"], group["color"]
        h, w = pattern.shape
        rs = m["rows"].get(color, []) if use_coordinates else range(1, H - 1)
        cs = m["cols"].get(color, []) if use_coordinates else range(1, W - 1)
        for r in rs:
            for c in cs:
                y, x = r - h // 2, c - w // 2
                if y < 1 or x < 1 or y + h > H - 1 or x + w > W - 1:
                    continue
                if not np.all(m["mask"][y:y + h, x:x + w]):
                    continue
                ids = tuple(indices[(y + dy, x + dx)]
                            for dy in range(h) for dx in range(w))
                bits = sum(1 << i for i in ids)
                index = len(candidates)
                candidates.append((gi, bits, ids, tuple(map(int, pattern.flat)), y, x))
                for i in ids:
                    by_cell[i].append(index)
    return coords, candidates, by_cell


def _pack(m, use_coordinates=True, finite=True, node_limit=50000):
    """Exhaustive rendered exact cover; exchanges of identical copies are grouped."""
    coords, candidates, by_cell = _placements(m, use_coordinates)
    N = len(coords)
    full = (1 << N) - 1
    counts = [g["count"] if finite else N for g in m["groups"]]
    initial = counts.copy()
    rendered = [-1] * N
    outcomes, dead = {}, set()
    path = []
    stats = dict(nodes=0, dead_ends=0, candidates=len(candidates), exhaustive=True)

    def search(covered):
        stats["nodes"] += 1
        if stats["nodes"] > node_limit:
            stats["exhaustive"] = False
            return False
        if covered == full:
            signature = tuple(rendered)
            used = tuple(initial[i] - counts[i] for i in range(len(counts)))
            if signature not in outcomes:
                outcomes[signature] = dict(path=tuple(path), used=used,
                                           min_used=sum(used), max_used=sum(used))
            else:
                outcomes[signature]["min_used"] = min(outcomes[signature]["min_used"], sum(used))
                outcomes[signature]["max_used"] = max(outcomes[signature]["max_used"], sum(used))
            return True
        state = covered, tuple(counts)
        if state in dead:
            return False
        best = None
        for cell in range(N):
            if covered & (1 << cell):
                continue
            options = [j for j in by_cell[cell]
                       if counts[candidates[j][0]] and not covered & candidates[j][1]]
            if not options:
                dead.add(state)
                stats["dead_ends"] += 1
                return False
            if best is None or len(options) < len(best):
                best = options
        success = False
        for j in best:
            gi, bits, ids, pixels, y, x = candidates[j]
            counts[gi] -= 1
            for i, color in zip(ids, pixels):
                rendered[i] = color
            path.append(j)
            success = search(covered | bits) or success
            path.pop()
            for i in ids:
                rendered[i] = -1
            counts[gi] += 1
            if len(outcomes) >= 2 or not stats["exhaustive"]:
                return success
        if not success:
            dead.add(state)
            stats["dead_ends"] += 1
        return success

    search(0)
    return outcomes, stats, (coords, candidates)


def _completion(m):
    outcomes, stats, placements = _pack(m)
    if not stats["exhaustive"]:
        raise ValueError("Exact-cover search exceeded its bounded family budget")
    if len(outcomes) != 1:
        raise ValueError("Expected exactly one rendered completion")
    rendering, record = next(iter(outcomes.items()))
    out = m["board"].copy()
    out[m["mask"]] = rendering
    return out, record, stats, placements


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, candidate center, inventory count, and pixel from input."""
    m = _parse(input_grid)
    out, _, _, _ = _completion(m)
    return np.rot90(out, -m["turn"]).copy()


def _stamp_legal(m):
    """A deliberately wrong shortcut: paint every individually fitting candidate."""
    out = m["board"].copy()
    occupied = np.zeros(out.shape, dtype=bool)
    _, candidates, _ = _placements(m)
    for gi, _, _, _, y, x in candidates:
        p = m["groups"][gi]["pattern"]
        h, w = p.shape
        region = out[y:y + h, x:x + w]
        free = ~occupied[y:y + h, x:x + w]
        region[free] = p[free]
        occupied[y:y + h, x:x + w] = True
    return out


def _features(m):
    out, rec, stats, data = _completion(m)
    coords, candidates = data
    counts = [g["count"] for g in m["groups"]]
    components = _components(m["mask"])
    owner = np.full(m["mask"].shape, -1, dtype=int)
    shapes = []
    for pi, j in enumerate(rec["path"]):
        gi, _, ids, _, y, x = candidates[j]
        shapes.append(m["groups"][gi]["pattern"].shape)
        for index in ids:
            owner[coords[index]] = pi
    max_component = max(len({int(owner[p]) for p in comp}) for comp in components)
    local_surplus = sum(counts[i] - rec["used"][i] for i in range(len(counts))
                        if any(p[0] == i for p in candidates))
    keys = {g["color"] for g in m["groups"]}
    return dict(roles=(m["bg"], m["wall"], m["frame"], m["divider"]),
                side=m["turn"], shape=out.shape, tile_count=len(m["tiles"]),
                used=rec["min_used"], surplus=len(m["tiles"]) - rec["min_used"],
                local_surplus=local_surplus, keys=len(keys), shapes=tuple(shapes),
                repeats=any(v >= 2 for v in rec["used"]),
                products=tuple((len(m["rows"].get(c, [])), len(m["cols"].get(c, [])))
                               for c in sorted(keys)),
                components=len(components), max_component_tiles=max_component,
                nonrectangular=any((max(r for r, c in comp) - min(r for r, c in comp) + 1)
                                   * (max(c for r, c in comp) - min(c for r, c in comp) + 1)
                                   != len(comp) for comp in components),
                asymmetric=any(not np.array_equal(t["pattern"], t["pattern"][::-1, ::-1])
                               for t in m["tiles"]),
                legal_stamp_difference=int(np.count_nonzero(out != _stamp_legal(m))),
                candidates=stats["candidates"], dead_ends=stats["dead_ends"],
                coverage=float(m["mask"][1:-1, 1:-1].mean()))


def _pattern(rng, h, w, key, payload, asymmetric=False):
    yy, xx = np.indices((h, w))
    a, b = map(int, rng.choice(payload, 2, replace=False))
    mode = int(rng.integers(6))
    if mode == 0:
        p = np.full((h, w), key, dtype=np.int8)
    elif mode == 1:
        p = np.where(yy % 2, a, b).astype(np.int8)
    elif mode == 2:
        p = np.where(xx % 2, a, b).astype(np.int8)
    elif mode == 3:
        p = np.where((yy + xx) % 2, a, b).astype(np.int8)
    elif mode == 4:
        p = np.full((h, w), a, dtype=np.int8)
    else:
        p = rng.choice(payload, size=(h, w)).astype(np.int8)
    if asymmetric and h * w > 1:
        a, b = map(int, rng.choice([c for c in payload if c != key], 2, replace=False))
        p.fill(a)
        p[0, 0] = b
    p[h // 2, w // 2] = key
    return p


def _grow(rng, H, W, count):
    mask = np.zeros((H, W), dtype=bool)
    rects = []
    dims = [(1, 1), (1, 3), (3, 1), (3, 3), (1, 5), (5, 1), (3, 5), (5, 3), (5, 5)]
    weights = np.array([1, 3, 3, 4, 2, 2, 1, 1, .4], dtype=float)
    connected = rng.random() < .8
    for i in range(count):
        if i and rng.random() < .35:
            h, w = rects[int(rng.integers(i))][2:]
        else:
            h, w = dims[int(rng.choice(len(dims), p=weights / weights.sum()))]
        choices = set()
        for _ in range(90):
            if not rects or (not connected and rng.random() < .35):
                y, x = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
            else:
                ry, rx, rh, rw = rects[int(rng.integers(len(rects)))]
                direction = int(rng.integers(4))
                if direction < 2:
                    y = ry - h if direction == 0 else ry + rh
                    x = rx + int(rng.integers(1 - w, rw))
                else:
                    x = rx - w if direction == 2 else rx + rw
                    y = ry + int(rng.integers(1 - h, rh))
            if y >= 0 and x >= 0 and y + h <= H and x + w <= W and not mask[y:y + h, x:x + w].any():
                choices.add((y, x))
        if not choices:
            return None
        choices = sorted(choices)
        y, x = choices[int(rng.integers(len(choices)))]
        mask[y:y + h, x:x + w] = True
        rects.append((y, x, h, w))
    return rects


def _assign_keys(rng, rects, key_colors):
    """Centers sharing either projection must share a key: one border cell, one label."""
    centers = [(y + h // 2, x + w // 2) for y, x, h, w in rects]
    parent = list(range(len(rects)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, (r, c) in enumerate(centers):
        for j in range(i):
            if r == centers[j][0] or c == centers[j][1]:
                parent[root(i)] = root(j)
    components = sorted({root(i) for i in range(len(rects))})
    if len(components) < 2:
        return None
    order = list(map(int, rng.permutation(key_colors)))
    mapping = {component: order[i % len(order)] for i, component in enumerate(components)}
    return [mapping[root(i)] for i in range(len(rects))]


def _scatter(rng, tiles, H, width, bg):
    for _ in range(18):
        tray = np.full((H, width), bg, dtype=np.int8)
        occupied = np.zeros((H, width), dtype=bool)
        order = sorted(range(len(tiles)), key=lambda i: (-tiles[i].size, float(rng.random())))
        success = True
        for i in order:
            p = tiles[i]
            h, w = p.shape
            choices = [(y, x) for y in range(1, H - h)
                       for x in range(width - w + 1)
                       if not occupied[max(0, y - 1):y + h + 1,
                                       max(0, x - 1):min(width, x + w + 1)].any()]
            if not choices:
                success = False
                break
            y, x = choices[int(rng.integers(len(choices)))]
            tray[y:y + h, x:x + w] = p
            occupied[y:y + h, x:x + w] = True
        if success:
            return tray
    return None


def _assemble(rng, rects, colors, H, W, tray_width, roles, payload,
              extra_marks, surplus=False, asymmetric=False, dense_marks=False,
              cross_keys=False):
    bg, wall, frame, divider = roles
    board = np.full((H, W), wall, dtype=np.int8)
    board[[0, -1], :] = frame
    board[:, [0, -1]] = frame
    out = board.copy()
    patterns, tiles = {}, []
    rows, cols = {}, {}
    for (y, x, h, w), color in zip(rects, colors):
        y, x = y + 1, x + 1  # construction rectangles use interior coordinates
        if board[y:y + h, x:x + w].shape != (h, w) or np.any(board[y:y + h, x:x + w] != wall):
            return None
        pk = color, h, w
        if pk not in patterns:
            patterns[pk] = _pattern(rng, h, w, color, payload, asymmetric)
            if cross_keys:
                other = next(c for c in colors if c != color)
                patterns[pk].fill(other)
                patterns[pk][0, 0] = int(rng.choice([c for c in payload if c not in (color, other)]))
                patterns[pk][h // 2, w // 2] = color
        p = patterns[pk]
        tiles.append(p.copy())
        board[y:y + h, x:x + w] = bg
        out[y:y + h, x:x + w] = p
        rows.setdefault(color, set()).add(y + h // 2)
        cols.setdefault(color, set()).add(x + w // 2)
    if dense_marks:
        # A one-key witness exposes finite inventory: the singleton fits at
        # every vacancy, but one supplied singleton cannot replace a whole tile.
        # Unlimited singleton copies would instead give a different full cover.
        if len(set(colors)) != 1:
            raise ValueError("Dense witness marks require one key")
        color = colors[0]
        ys, xs = np.nonzero(board == bg)
        rows[color] = set(map(int, ys))
        cols[color] = set(map(int, xs))
    for values, length in ((rows, H), (cols, W)):
        used = set().union(*values.values())
        # Projections of different keys cannot occupy the same border position.
        if len(used) != sum(map(len, values.values())):
            return None
        for color in rng.permutation(list(values)):
            free = [v for v in range(1, length - 1) if v not in used]
            rng.shuffle(free)
            n = int(rng.integers(extra_marks + 1))
            values[color].update(free[:n])
            used.update(free[:n])
    for color, values in rows.items():
        for r in values:
            board[r, 0] = board[r, -1] = out[r, 0] = out[r, -1] = color
    for color, values in cols.items():
        for c in values:
            board[0, c] = board[-1, c] = out[0, c] = out[-1, c] = color
    if surplus:
        # One singleton cannot replace an area>=3 tile. With mixed inventories,
        # the exact-cover acceptance check, rather than this heuristic, decides.
        color = int(rng.choice(colors))
        tiles.append(np.array([[color]], dtype=np.int8))
    tray = _scatter(rng, tiles, H, tray_width, bg)
    if tray is None:
        return None
    inp = np.concatenate((tray, np.full((H, 1), divider, dtype=np.int8), board), axis=1)
    try:
        m = _canonical_model(inp, tray_width)
        m["turn"] = 0
        predicted, rec, _, _ = _completion(m)
    except ValueError:
        return None
    if not np.array_equal(predicted, out) or (surplus and rec["max_used"] >= len(tiles)):
        return None
    return {"input": inp, "output": out}


def _witness(rng, kind, surplus):
    for _ in range(100):
        palette = list(map(int, rng.permutation(10)))
        roles, payload = tuple(palette[:4]), palette[4:]
        h, w = [(1, 3), (3, 1), (3, 3), (3, 5), (5, 3)][int(rng.integers(5))]
        if kind == "packing" and w == 1:
            h, w = w, h
        if kind == "coordinates":
            if w > 1:
                offset = int(rng.integers(1, w))
                rects = [(0, 0, h, w), (h, offset, h, w)]
            else:
                offset = int(rng.integers(1, h))
                rects = [(0, 0, h, w), (offset, w, h, w)]
            colors = payload[:2]
        else:
            offset = int(rng.integers(1, w))
            rects = [(0, 0, h, w), (0, w, h, w), (h, offset, h, w)]
            colors = [payload[0]] * 3
        hh = max(y + bh for y, x, bh, bw in rects)
        ww = max(x + bw for y, x, bh, bw in rects)
        H = int(rng.integers(max(12, hh + 4, 3 * h + 4), 29))
        W = int(rng.integers(ww + 4, min(20, ww + 9)))
        tray_width = int(rng.integers(max(w, 5), min(10, 29 - W) + 1))
        dy, dx = int(rng.integers(H - hh - 1)), int(rng.integers(W - ww - 1))
        rects = [(y + dy, x + dx, bh, bw) for y, x, bh, bw in rects]
        pair = _assemble(rng, rects, colors, H, W, tray_width, roles, payload,
                         extra_marks=1, surplus=surplus and kind == "packing", asymmetric=True,
                         dense_marks=surplus and kind == "packing",
                         cross_keys=kind == "coordinates")
        if pair is None:
            continue
        m = _parse(pair["input"])
        if kind == "packing" and not _features(m)["legal_stamp_difference"]:
            continue
        if kind == "coordinates" and len(_pack(m, use_coordinates=False)[0]) < 2:
            continue
        return pair
    raise RuntimeError("Bounded witness construction exhausted")


def _random_pair(rng, test, surplus):
    for _ in range(100):
        H = int(rng.integers(16 if test else 12, 31))
        W = int(rng.integers(11, 20))
        tray_width = int(rng.integers(5, min(10, 29 - W) + 1))
        count = int(rng.integers(4 if test else 3, 8 if test else 6))
        rects = _grow(rng, H - 2, W - 2, count)
        if rects is None:
            continue
        palette = list(map(int, rng.permutation(10)))
        roles, payload = tuple(palette[:4]), palette[4:]
        key_colors = payload[:int(rng.integers(2, 5))]
        colors = _assign_keys(rng, rects, key_colors)
        if colors is None:
            continue
        pair = _assemble(rng, rects, colors, H, W, tray_width, roles, payload,
                         extra_marks=2 if test else 1, surplus=surplus)
        if pair is not None:
            return pair
    # Same rule, still a variable connected silhouette; never change an answer
    # or accept an unresolved search to meet the attempt budget.
    return _witness(rng, "packing", surplus)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    allow_surplus = bool(rng.random() < .65)
    orientation_regime = int(rng.integers(3))
    familiar = list(map(int, rng.permutation(4)))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(int(num_train + num_test)):
        test = i >= num_train
        if i == 0:
            pair = _witness(rng, "coordinates", False)
        elif i == 1:
            pair = _witness(rng, "packing", allow_surplus)
        else:
            pair = _random_pair(rng, test, allow_surplus and rng.random() < .5)
        if orientation_regime == 0:
            turn = familiar[0]
        elif orientation_regime == 1:
            turn = familiar[i % 2]
        else:
            turn = familiar[2 + (i % 2)] if test else familiar[i % 2]
        # Whole-scene isometries rotate the source matrices and their destinations
        # together. At solve time each tile is translated in its supplied orientation.
        if rng.random() < .5:
            pair = {k: a[::-1].copy() for k, a in pair.items()}
        pair = {k: np.rot90(a, turn).copy() for k, a in pair.items()}
        # Up to 14 inputs; fourteen distinct permutations of structural roles
        # suffice even in the unlikely event of an identical sampled geometry.
        original = pair
        for attempt in range(24):
            mapping = np.arange(10, dtype=np.int8)
            if attempt:
                m = _parse(original["input"])
                roles = (m["bg"], m["wall"], m["frame"], m["divider"])
                import itertools
                perm = list(itertools.permutations(roles))[attempt]
                for a, b in zip(roles, perm):
                    mapping[a] = b
            pair = {k: mapping[a] for k, a in original.items()}
            if _key(pair["input"]) not in seen:
                break
        else:
            raise AssertionError("Distinct-input recoloring exhausted")
        seen.add(_key(pair["input"]))
        puzzle["test" if test else "train"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    features, models, seen = [], [], set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = _grid(pair["input"]), _grid(pair["output"])
        assert _key(inp) not in seen, "Duplicate input"
        seen.add(_key(inp))
        m = _parse(inp)
        canonical, rec, stats, _ = _completion(m)
        assert stats["exhaustive"] and np.array_equal(np.rot90(canonical, -m["turn"]), out)
        assert np.array_equal(canonical[~m["mask"]], m["board"][~m["mask"]]), "Changed frame or walls"
        assert not np.any(canonical[m["mask"]] == m["bg"]), "Uncovered vacancy"
        assert all(n <= g["count"] for n, g in zip(rec["used"], m["groups"])), "Reused a physical tile"
        features.append(_features(m))
        models.append(m)
    training, testing = features[:nt], features[nt:]
    assert any(f["repeats"] and f["nonrectangular"] for f in training), "No merged repeated-tile evidence"
    assert any(f["legal_stamp_difference"] for f in training), "Packing is underdemonstrated"
    assert any(f["asymmetric"] for f in training), "No pattern-preserving translation evidence"
    assert len(_pack(models[0], use_coordinates=False)[0]) >= 2, "Coordinate ablation should be ambiguous"
    assert not any(f["surplus"] for f in testing) or any(f["surplus"] for f in training)
    if any(f["local_surplus"] for f in testing):
        assert any(f["local_surplus"] for f in training)
    if training[1]["local_surplus"]:
        assert len(_pack(models[1], finite=False)[0]) >= 2, "Finite inventory is underdemonstrated"


def self_test():
    """Bounded checks over 42 seeds, with actual-grid diversity and ablations."""
    signatures, shapes, patterns, roles, sides, products = set(), set(), set(), set(), set(), set()
    tile_counts, coverages, component_counts = set(), [], set()
    local_surplus = repeated = rejected = inventory_witnesses = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for pair in puzzle["train"] + puzzle["test"]:
            m = _parse(pair["input"])
            f = _features(m)
            shapes.add(f["shape"])
            roles.add(f["roles"])
            sides.add(f["side"])
            products.update(f["products"])
            tile_counts.add(f["used"])
            coverages.append(f["coverage"])
            component_counts.add(f["components"])
            local_surplus += bool(f["local_surplus"])
            repeated += bool(f["repeats"])
            patterns.update(_key(t["pattern"]) for t in m["tiles"])
        finite_model = _parse(puzzle["train"][1]["input"])
        inventory_witnesses += len(_pack(finite_model, finite=False)[0]) >= 2
        # An output from unconditional candidate stamping must be rejected.
        wrong = {s: [{k: _grid(a) for k, a in p.items()} for p in puzzle[s]]
                 for s in ("train", "test")}
        m = _parse(wrong["train"][1]["input"])
        wrong["train"][1]["output"] = np.rot90(_stamp_legal(m), -m["turn"]).copy()
        try:
            validate_generated_puzzle(wrong)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError("Accepted the all-legal-candidates shortcut")
    assert len(signatures) == 42 and len(shapes) >= 40 and len(patterns) >= 100
    assert len(roles) >= 60 and sides == {0, 1, 2, 3}
    assert {2, 3, 4, 5, 6, 7} <= tile_counts
    assert any(r > 1 and c > 1 for r, c in products)
    assert len(products) >= 8 and max(component_counts) > 1
    assert local_surplus >= 10 and repeated >= 42 and rejected == 42
    assert inventory_witnesses >= 10
    assert max(coverages) - min(coverages) > .2
    first = generate_puzzle(25170998, 2, 1)
    again = generate_puzzle(25170998, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in ("train", "test")
               for a, b in zip(first[s], again[s]) for k in ("input", "output"))
    # A single physical copy cannot fill the three-copy witness.
    m = _parse(first["train"][1]["input"])
    for g in m["groups"]:
        g["count"] = 1
    assert not _pack(m)[0]
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Accepted invalid counts")
    return dict(seeds=42, distinct_puzzles=len(signatures), output_shapes=len(shapes),
                tile_patterns=len(patterns), roles=len(roles), used_counts=sorted(tile_counts),
                coordinate_products=sorted(products), locally_fitting_surplus_pairs=local_surplus,
                finite_inventory_witnesses=inventory_witnesses,
                shortcut_mutations_rejected=rejected)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
