"""Addressed compound eyes: nearest-sector thumbnails and outward half-turn echoes.

All semantics are visible in the input. Generation uses sparse node layouts for
packing only; solve also accepts off-lattice coordinates satisfying the spec.
No files, network, parent generators, or stored puzzle examples are needed.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CHECK_SEEDS = (0, 295979516, 30291001, 897764124, 736840150, 551846276,
               536240777, 550294115, 449100474, 391099089, 110481376,
               925974637, 313579011, 647206681, 983070158, 87489554,
               671000716, 616491726, 187208542, 325647832, 632778824,
               27815894, 152965680, 736368216, 806958216, 489377582,
               417191700, 975948010, 788676059, 766629108, 986776882,
               913688026, 132568386, 871378265, 482656296, 664214861,
               671366534, 278719979, 647793408, 512369026, 674288126,
               451425423)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _sign(s, h):
    return tuple(int(np.sign(x - y)) for x, y in zip(s, h))


def _box(center):
    r, c = center
    return {(r + y, c + x) for y in (-1, 0, 1) for x in (-1, 0, 1)}


def _slice(a, center):
    r, c = center
    return a[r - 1:r + 2, c - 1:c + 2]


def _d4(tile):
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _plan(centers, hubs, shape, all_sources=None):
    """Geometry only. Each record is (hub, source_index, sector, echo_center)."""
    all_sources = centers if all_sources is None else all_sources
    h, w = shape
    for p in list(all_sources) + list(hubs):
        _require(2 <= p[0] <= h - 3 and 2 <= p[1] <= w - 3, "patch crosses frame")
    _require(len(set(all_sources)) == len(all_sources), "duplicate source")
    for i, a in enumerate(all_sources):
        for b in all_sources[i + 1:]:
            _require(max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 4,
                     "source squares need a background gap")
    source_cells = set().union(*(_box(s) for s in all_sources))
    hub_cells = set()
    for hub in hubs:
        cells = _box(hub)
        _require(not cells & (source_cells | hub_cells), "viewpoint patch is not vacant")
        hub_cells.update(cells)
    echoes, shadowed = [], 0
    for hub in hubs:
        sectors = {}
        for i, s in enumerate(centers):
            d = _sign(s, hub)
            _require(d != (0, 0), "source on viewpoint")
            distance = (s[0] - hub[0]) ** 2 + (s[1] - hub[1]) ** 2
            sectors.setdefault(d, []).append((distance, i))
        for d in sorted(sectors):
            ranked = sorted(sectors[d])
            _require(len(ranked) == 1 or ranked[0][0] != ranked[1][0],
                     "nearest distance tie within a sector")
            i = ranked[0][1]
            s = centers[i]
            e = (2 * s[0] - hub[0], 2 * s[1] - hub[1])
            _require(2 <= e[0] <= h - 3 and 2 <= e[1] <= w - 3, "echo crosses frame")
            echoes.append((hub, i, d, e))
            shadowed += len(ranked) - 1
    used = source_cells | hub_cells
    for _, _, _, e in echoes:
        cells = _box(e)
        _require(not cells & used, "echo patches must be disjoint and vacant")
        used.update(cells)
    return dict(echoes=echoes, hubs=list(hubs), shadowed=shadowed)


def _parse(input_grid):
    a = _grid(input_grid)
    _require(min(a.shape) >= 7, "framed field too small")
    frame = int(a[0, 0])
    _require(all(int(a[r, c]) == frame for r, c in
                 ((0, -1), (-1, 0), (-1, -1))), "frame corners disagree")
    interior = a[1:-1, 1:-1]
    counts = np.bincount(interior.ravel().astype(np.int8), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > interior.size / 2 and bg != frame, "background must be dominant")
    _require(not np.any(interior == frame), "frame color inside field")
    _require(np.array_equal(a[0, 1:-1], a[-1, 1:-1]) and
             np.array_equal(a[1:-1, 0], a[1:-1, -1]), "unpaired border marks")
    _require(not np.any(a[0, :] == bg) and not np.any(a[-1, :] == bg) and
             not np.any(a[:, 0] == bg) and not np.any(a[:, -1] == bg), "background in frame")
    rows, cols = {}, {}
    for r in range(1, a.shape[0] - 1):
        color = int(a[r, 0])
        if color != frame:
            rows.setdefault(color, []).append(r)
    for c in range(1, a.shape[1] - 1):
        color = int(a[0, c])
        if color != frame:
            cols.setdefault(color, []).append(c)
    unseen = set(map(tuple, np.argwhere(a != bg)))
    unseen = {(r, c) for r, c in unseen if 0 < r < a.shape[0] - 1 and 0 < c < a.shape[1] - 1}
    tiles = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    todo.append(q)
        r0, c0 = min(r for r, _ in component), min(c for _, c in component)
        center = (r0 + 1, c0 + 1)
        _require(component == _box(center), "source must be a full isolated 3x3 square")
        tile = _slice(a, center).copy()
        cc = np.bincount(tile.ravel().astype(np.int8), minlength=10)
        carrier = int(np.argmax(cc))
        _require(cc[carrier] >= 5, "tile lacks majority carrier")
        tiles.append(dict(center=center, color=carrier, tile=tile))
    tiles.sort(key=lambda t: t["center"])
    _require(1 <= len(tiles) <= 16, "source count")
    carriers = {t["color"] for t in tiles}
    active = carriers & set(rows) & set(cols)
    _require(len(active) == 1, "exactly one carrier must have both coordinate axes")
    color = next(iter(active))
    group = [t for t in tiles if t["color"] == color]
    hubs = list(itertools.product(rows[color], cols[color]))
    _require(1 <= len(hubs) <= 4, "one to four viewpoints")
    plan = _plan([t["center"] for t in group], hubs, a.shape,
                 [t["center"] for t in tiles])
    return dict(a=a, bg=bg, frame=frame, color=color, tiles=tiles, group=group,
                rows=rows, cols=cols, plan=plan)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve only the visible input. No RNG or puzzle-wide hidden parameters."""
    p = _parse(input_grid)
    out = p["a"].copy()
    for hub in p["plan"]["hubs"]:
        out[hub] = p["color"]
    for hub, i, d, e in p["plan"]["echoes"]:
        tile = p["group"][i]["tile"]
        out[hub[0] + d[0], hub[1] + d[1]] = tile[1 - d[0], 1 - d[1]]
        _slice(out, e)[:] = tile[::-1, ::-1]
    return out


def _node_plan(nr, nc, rows, cols, sources):
    centers = [(2 + 4 * r, 2 + 4 * c) for r, c in sources]
    hubs = [(2 + 4 * r, 2 + 4 * c) for r, c in itertools.product(rows, cols)]
    return _plan(centers, hubs, (4 * (nr - 1) + 5, 4 * (nc - 1) + 5))


def _orient(rng, geometry):
    nr, nc, rows, cols, ss = geometry
    if rng.random() < .5:
        rows = [nr - 1 - r for r in rows]
        ss = [(nr - 1 - r, c) for r, c in ss]
    if rng.random() < .5:
        cols = [nc - 1 - c for c in cols]
        ss = [(r, nc - 1 - c) for r, c in ss]
    if rng.random() < .5:
        nr, nc, rows, cols = nc, nr, cols, rows
        ss = [(c, r) for r, c in ss]
    return nr, nc, sorted(rows), sorted(cols), ss


def _geometry(rng, kind):
    if kind == "occlusion":
        nr, nc = int(rng.integers(5, 8)), 7
        hub = (nr // 2, 2)
        ss = [(hub[0] + r, hub[1] + c) for r, c in DIRECTIONS]
        ss.append((hub[0] + int(rng.choice([-1, 1])), hub[1] + 2))
        return _orient(rng, (nr, nc, [hub[0]], [hub[1]], ss))
    if kind == "cartesian":
        nr, nc = 7, int(rng.integers(5, 8))
        mid = nc // 2
        return _orient(rng, (nr, nc, [0, 1], [mid - 1, mid + 1], [(3, mid)]))
    if kind == "single_view":
        nr, nc = map(int, rng.integers(5, 8, size=2))
        hub = (nr // 2, nc // 2)
        ds = rng.permutation(len(DIRECTIONS))[:int(rng.integers(2, 9))]
        ss = [(hub[0] + DIRECTIONS[i][0], hub[1] + DIRECTIONS[i][1]) for i in ds]
        return _orient(rng, (nr, nc, [hub[0]], [hub[1]], ss))
    # A bounded greedy search on at most 49 source positions per attempt.
    for _ in range(80):
        nr, nc = map(int, rng.integers(4, 8, size=2))
        ar, ac = ((1, 2), (2, 1), (2, 2))[int(rng.integers(3))]
        rows = sorted(map(int, rng.choice(nr, ar, replace=False)))
        cols = sorted(map(int, rng.choice(nc, ac, replace=False)))
        candidates = list(itertools.product(range(nr), range(nc)))
        rng.shuffle(candidates)
        ss = []
        desired = int(rng.integers(2, 5))
        for s in candidates:
            try:
                _node_plan(nr, nc, rows, cols, ss + [s])
            except AssertionError:
                continue
            ss.append(s)
            if len(ss) == desired:
                break
        if len(ss) >= 2:
            p = _node_plan(nr, nc, rows, cols, ss)
            if len({i for _, i, _, _ in p["echoes"]}) >= 2:
                return _orient(rng, (nr, nc, rows, cols, ss))
    # This two-source/four-view construction is always valid; no rule changes.
    return _orient(rng, (7, 7, [0, 1], [2, 4], [(2, 2), (3, 4)]))


def _random_tile(rng, carrier):
    tile = np.full((3, 3), carrier, dtype=np.int8)
    count = int(rng.integers(1, 5))
    positions = list(map(int, rng.permutation(9)))
    mode = int(rng.integers(3))
    if mode == 1:
        row = int(rng.integers(3))
        positions.sort(key=lambda j: j // 3 != row)
    elif mode == 2:
        positions.sort(key=lambda j: abs(j // 3 - j % 3))
    repeated = int(rng.integers(4, 8))
    for j in positions[:count]:
        tile.flat[j] = repeated if mode else int(rng.integers(4, 8))
    return tile


def _witness_tile(rng, d):
    inward = (1 - d[0], 1 - d[1])
    sample, auxiliary = map(int, rng.choice(np.arange(4, 8), 2, replace=False))
    for j in rng.permutation(9):
        if divmod(int(j), 3) == inward:
            continue
        tile = np.full((3, 3), 1, dtype=np.int8)
        tile[inward] = sample
        tile.flat[int(j)] = auxiliary
        if len({v.tobytes() for v in _d4(tile)}) == 8:
            return tile
    raise AssertionError("no D4 witness")


def _add_marks(rng, a, color, state):
    for axis, enabled in ((0, state in ("R", "A")), (1, state in ("C", "A"))):
        if not enabled:
            continue
        n = a.shape[axis]
        free = [i for i in range(2, n - 2)
                if (a[i, 0] == 9 if axis == 0 else a[0, i] == 9)]
        _require(bool(free), "border capacity")
        pos = int(rng.choice(free))
        if axis == 0:
            a[pos, 0] = a[pos, -1] = color
        else:
            a[0, pos] = a[-1, pos] = color


def _make_pair(rng, kind):
    nr, nc, rows, cols, source_nodes = _geometry(rng, kind)
    max_step = min(5, 25 // (max(nr, nc) - 1))
    step = int(rng.integers(4, max_step + 1))
    base_h, base_w = step * (nr - 1) + 5, step * (nc - 1) + 5
    eh, ew = int(rng.integers(31 - base_h)), int(rng.integers(31 - base_w))
    top, left = 2 + int(rng.integers(eh + 1)), 2 + int(rng.integers(ew + 1))
    shape = (base_h + eh, base_w + ew)
    at = lambda s: (top + step * s[0], left + step * s[1])
    centers = [at(s) for s in source_nodes]
    hubs = [at(s) for s in itertools.product(rows, cols)]
    plan = _plan(centers, hubs, shape)
    active_tiles = [_random_tile(rng, 1) for _ in centers]
    if kind == "occlusion":
        for _, i, d, _ in plan["echoes"]:
            active_tiles[i] = _witness_tile(rng, d)
        # The farther tile has a different inward color, so the miniature also
        # distinguishes nearest selection, independently of echo presence.
        far = len(centers) - 1
        d = _sign(centers[far], hubs[0])
        near = next(i for _, i, dd, _ in plan["echoes"] if dd == d)
        old = int(active_tiles[near][1 - d[0], 1 - d[1]])
        point = (1 - d[0], 1 - d[1])
        if active_tiles[far][point] == 1 and np.count_nonzero(active_tiles[far] == 1) == 5:
            erase = tuple(np.argwhere(active_tiles[far] != 1)[0])
            active_tiles[far][erase] = 1
        active_tiles[far][1 - d[0], 1 - d[1]] = next(c for c in range(4, 8) if c != old)
    elif kind == "cartesian":
        ds = sorted({d for _, _, d, _ in plan["echoes"]})
        _require(len(ds) == 2, "two perspective sample positions expected")
        tile = np.full((3, 3), 1, dtype=np.int8)
        details = list(map(int, rng.permutation(np.arange(4, 8))))
        for d, color in zip(ds, details):
            tile[1 - d[0], 1 - d[1]] = color
        free = [j for j in range(9) if tile.flat[j] == 1]
        tile.flat[int(rng.choice(free))] = details[2]
        active_tiles[0] = tile
    occupied_nodes = set(source_nodes) | set(itertools.product(rows, cols))
    echo_nodes = {((e[0] - top) // step, (e[1] - left) // step)
                  for _, _, _, e in plan["echoes"]}
    free = [s for s in itertools.product(range(nr), range(nc))
            if s not in occupied_nodes | echo_nodes]
    rng.shuffle(free)
    states = ["N", "R"] if kind == "occlusion" else ["C"] if kind == "cartesian" else (
        list(map(str, rng.choice(["N", "R", "C"], size=int(rng.integers(3))))))
    states = states[:len(free)]
    inactive = [(at(s), j + 2, _random_tile(rng, j + 2), state)
                for j, (s, state) in enumerate(zip(free, states))]
    # Ensure modifying the hidden tile above did not remove its strict carrier.
    _require(all(np.count_nonzero(t == 1) >= 5 for t in active_tiles), "carrier budget")
    a = np.zeros(shape, dtype=np.int8)
    a[[0, -1], :] = 9
    a[:, [0, -1]] = 9
    for center, tile in zip(centers, active_tiles):
        _slice(a, center)[:] = tile
    for center, _, tile, _ in inactive:
        _slice(a, center)[:] = tile
    for r in rows:
        y = top + step * r
        a[y, 0] = a[y, -1] = 1
    for c in cols:
        x = left + step * c
        a[0, x] = a[-1, x] = 1
    for _, color, _, state in inactive:
        _add_marks(rng, a, color, state)
    if kind == "cartesian" or rng.random() < .4:
        _add_marks(rng, a, 8, "A")  # Full coordinates with no corresponding carrier.
    # Ground-truth rendering uses construction records, never solve or _parse.
    out = a.copy()
    for hub in hubs:
        out[hub] = 1
    for hub, i, d, e in plan["echoes"]:
        out[hub[0] + d[0], hub[1] + d[1]] = active_tiles[i][1 - d[0], 1 - d[1]]
        _slice(out, e)[:] = np.rot90(active_tiles[i], 2)
    palette = rng.permutation(10).astype(np.int8)
    return {"input": palette[a], "output": palette[out]}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train = [_make_pair(rng, "occlusion"), _make_pair(rng, "cartesian")]
    for _ in range(int(num_train) - 2):
        train.append(_make_pair(rng, str(rng.choice(["single_view", "cartesian"]))))
    tests = [_make_pair(rng, "multi") for _ in range(int(num_test))]
    rng.shuffle(train)
    seen = set()
    for pair in train + tests:
        original = pair
        # Ninety bijections with distinct background/frame assignments exceed
        # the thirteen preceding inputs. This fallback never changes the rule.
        if _key(pair["input"]) in seen:
            p = _parse(pair["input"])
            order = [p["bg"], p["frame"]] + [c for c in range(10) if c not in (p["bg"], p["frame"])]
            for bg, frame in itertools.permutations(range(10), 2):
                target = [bg, frame] + [c for c in range(10) if c not in (bg, frame)]
                mapping = np.empty(10, dtype=np.int8)
                mapping[order] = target
                candidate = mapping[original["input"]]
                if _key(candidate) not in seen:
                    pair["input"] = candidate
                    pair["output"] = mapping[original["output"]]
                    break
            else:
                raise AssertionError("distinct-palette fallback exhausted")
        seen.add(_key(pair["input"]))
    puzzle = {"train": train, "test": tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    """Scalar provenance checks, including every unchanged cell and exact area."""
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    a, out = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == out.shape, "output shape")
    p = _parse(a)
    assigned = set()
    for hub in p["plan"]["hubs"]:
        _require(out[hub] == p["color"], "thumbnail carrier")
        assigned.add(hub)
    for hub, i, d, e in p["plan"]["echoes"]:
        s = p["group"][i]["center"]
        target = (hub[0] + d[0], hub[1] + d[1])
        _require(out[target] == a[s[0] - d[0], s[1] - d[1]], "inward sample")
        assigned.add(target)
        for dy, dx in itertools.product((-1, 0, 1), repeat=2):
            target = (e[0] + dy, e[1] + dx)
            _require(out[target] == a[s[0] - dy, s[1] - dx], "half-turn echo")
            assigned.add(target)
    mask = np.zeros(a.shape, dtype=bool)
    for pos in assigned:
        mask[pos] = True
    _require(np.array_equal(a[~mask], out[~mask]), "source, frame, or unused background altered")
    area = 10 * len(p["plan"]["echoes"]) + len(p["plan"]["hubs"])
    _require(len(assigned) == area == np.count_nonzero(a != out), "changed-cell accounting")
    _require(np.array_equal(np.unique(a), np.unique(out)), "palette changed")
    return p


def _training_candidates(train, parsed):
    positions = {d: set(range(9)) for d in DIRECTIONS}
    operations = {d: set(range(8)) for d in DIRECTIONS}
    for pair, p in zip(train, parsed):
        out = np.asarray(pair["output"])
        for hub, i, d, e in p["plan"]["echoes"]:
            tile = p["group"][i]["tile"]
            sample = out[hub[0] + d[0], hub[1] + d[1]]
            positions[d].intersection_update(j for j in range(9) if tile.flat[j] == sample)
            operations[d].intersection_update(j for j, v in enumerate(_d4(tile))
                                              if np.array_equal(v, _slice(out, e)))
    return positions, operations


def _nearest_witness(p):
    """A farther same-sector tile proposes a different miniature color."""
    group = p["group"]
    for hub, i, d, _ in p["plan"]["echoes"]:
        near = int(group[i]["tile"][1 - d[0], 1 - d[1]])
        for j, t in enumerate(group):
            if j != i and _sign(t["center"], hub) == d:
                if int(t["tile"][1 - d[0], 1 - d[1]]) != near:
                    return True
    return False


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, parsed = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            p = _check_pair(pair)
            key = _key(p["a"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            parsed[split].append(p)
    # Two complementary witnesses fit even the minimum two-training budget.
    _require(any(len(p["plan"]["hubs"]) == 1 and
                 {d for _, _, d, _ in p["plan"]["echoes"]} == set(DIRECTIONS) and
                 p["plan"]["shadowed"] > 0 for p in parsed["train"]), "occlusion witness missing")
    _require(any(_nearest_witness(p) for p in parsed["train"]), "nearest color witness missing")
    _require(any(len(p["group"]) == 1 and len(p["rows"][p["color"]]) == 2 and
                 len(p["cols"][p["color"]]) == 2 and len({int(p["group"][i]["tile"][1-d[0], 1-d[1]])
                 for _, i, d, _ in p["plan"]["echoes"]}) >= 2
                 for p in parsed["train"]), "perspective/Cartesian witness missing")
    states = set()
    for p in parsed["train"]:
        for t in p["tiles"]:
            if t["color"] != p["color"]:
                states.add((bool(p["rows"].get(t["color"])), bool(p["cols"].get(t["color"]))))
    _require({(False, False), (True, False), (False, True)} <= states, "inactivity witnesses missing")
    _require(any((set(p["rows"]) & set(p["cols"])) - {t["color"] for t in p["tiles"]}
                 for p in parsed["train"]), "unused coordinate-color witness missing")
    positions, operations = _training_candidates(puzzle["train"], parsed["train"])
    _require(all(positions[d] and operations[d] for d in DIRECTIONS), "inconsistent training")
    for p in parsed["test"]:
        _require(len(p["plan"]["hubs"]) >= 2 and len(p["group"]) >= 2, "recombination shift missing")
        _require(len({i for _, i, _, _ in p["plan"]["echoes"]}) >= 2, "only one source contributes")
        for _, i, d, _ in p["plan"]["echoes"]:
            tile = p["group"][i]["tile"]
            _require(len({int(tile.flat[j]) for j in positions[d]}) == 1, "sample position underdemonstrated")
            views = _d4(tile)
            _require(len({views[j].tobytes() for j in operations[d]}) == 1, "rotation underdemonstrated")


def _normalized_key(a):
    names, values = {}, []
    for value in a.flat:
        value = int(value)
        names.setdefault(value, len(names))
        values.append(names[value])
    return a.shape, hashlib.sha256(bytes(values)).hexdigest()


def _expect_rejected(p):
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        return
    raise AssertionError("validator accepted corruption")


def self_test():
    """42 seeds, streaming statistics, bounded mutation checks, no disk output."""
    signatures, structures, shapes, layouts, patterns, backgrounds, carriers = (set() for _ in range(7))
    sectors, products, source_counts = set(), set(), set()
    shadowed_pairs = changed = pair_count = nearest_witnesses = 0
    mutation_checks = set()
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                a, out = pair["input"], pair["output"]
                before = a.copy()
                _require(np.array_equal(solve(a), out), "solve mismatch")
                _require(np.array_equal(a, before), "solve mutated input")
                p = _parse(a)
                digest.update(repr(a.shape).encode())
                digest.update(a.tobytes()); digest.update(out.tobytes())
                structures.add(_normalized_key(a)); shapes.add(a.shape)
                layouts.add((a.shape, tuple(t["center"] for t in p["tiles"]), tuple(p["plan"]["hubs"])))
                backgrounds.add(p["bg"]); carriers.add(p["color"])
                source_counts.add(len(p["group"]))
                products.add((len(p["rows"][p["color"]]), len(p["cols"][p["color"]])))
                sectors.update(d for _, _, d, _ in p["plan"]["echoes"])
                patterns.update(t["tile"].tobytes() for t in p["tiles"])
                shadowed_pairs += p["plan"]["shadowed"] > 0
                nearest_witnesses += _nearest_witness(p)
                changed += int(np.count_nonzero(a != out)); pair_count += 1
                # At most five small copies over the entire cohort, not per seed.
                if len(mutation_checks) < 5:
                    for kind in ("echo", "thumbnail", "source", "frame", "cartesian"):
                        if kind in mutation_checks:
                            continue
                        hub, i, d, e = p["plan"]["echoes"][0]
                        s = p["group"][i]["center"]
                        bad = {k: [dict(x) for x in puzzle[k]] for k in ("train", "test")}
                        wrong = out.copy()
                        if kind == "cartesian":
                            if len(p["plan"]["hubs"]) != 4:
                                continue
                            h2 = p["plan"]["hubs"][-1]
                            wrong[h2] = p["bg"]
                        else:
                            pos = {"echo": e, "thumbnail": hub, "source": s, "frame": (0, 0)}[kind]
                            wrong[pos] = (int(wrong[pos]) + 1) % 10
                        bad[split][index] = {"input": a, "output": wrong}
                        _expect_rejected(bad)
                        mutation_checks.add(kind)
        signatures.add(digest.hexdigest())
    _require(len(signatures) == 42, "seed diversity")
    _require(len(structures) >= 180 and len(layouts) >= 180 and len(shapes) >= 30, "structural diversity")
    _require(len(patterns) >= 200, "tile vocabulary")
    _require(backgrounds == carriers == set(range(10)), "palette role diversity")
    _require(sectors == set(DIRECTIONS), "direction coverage")
    _require({(1, 1), (1, 2), (2, 1), (2, 2)} <= products, "coordinate products")
    _require({1, 2, 9} <= source_counts and shadowed_pairs >= 42, "selection/count variation")
    _require(nearest_witnesses >= 42, "nearest selection is not visibly distinguished")
    _require(len(mutation_checks) == 5, "mutation coverage")
    p = generate_puzzle(123, 2, 1)
    q = generate_puzzle(123, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for split in p
                 for x, y in zip(p[split], q[split]) for k in x), "determinism")
    # Counterfactual A: move only one paired coordinate by one pixel. This also
    # supplies an off-lattice regression; parsing never assumes the packing grid.
    addressed = next(x for x in p["train"] if len(_parse(x["input"])["plan"]["hubs"]) == 4)
    info = _parse(addressed["input"])
    moved = False
    for axis in (0, 1):
        positions = info["rows"] if axis == 0 else info["cols"]
        for old in positions[info["color"]]:
            for delta in (-1, 1):
                new = old + delta
                if not 2 <= new <= addressed["input"].shape[axis] - 3:
                    continue
                a = addressed["input"].copy()
                if axis == 0:
                    if a[new, 0] != info["frame"]:
                        continue
                    a[old, 0] = a[old, -1] = info["frame"]
                    a[new, 0] = a[new, -1] = info["color"]
                else:
                    if a[0, new] != info["frame"]:
                        continue
                    a[0, old] = a[-1, old] = info["frame"]
                    a[0, new] = a[-1, new] = info["color"]
                try:
                    answer = solve(a)
                except AssertionError:
                    continue
                _check_pair({"input": a, "output": answer})
                _require(not np.array_equal(answer, addressed["output"]), "coordinates have no effect")
                moved = True
                break
            if moved:
                break
        if moved:
            break
    _require(moved, "off-lattice coordinate regression unavailable")
    for transform in range(8):
        _require(np.array_equal(solve(_d4(addressed["input"])[transform]),
                                _d4(addressed["output"])[transform]), "D4 equivariance")
    recolored = (addressed["input"].astype(int) + 1) % 10
    _require(np.array_equal(solve(recolored), (addressed["output"].astype(int) + 1) % 10),
             "palette equivariance")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            test = generate_puzzle(4000 + 10 * n + m, n, m)
            _require(len(test["train"]) == n and len(test["test"]) == m, "explicit counts")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=pair_count, distinct_puzzles=len(signatures),
                input_shapes=len(shapes), normalized_inputs=len(structures),
                layouts=len(layouts), tile_patterns=len(patterns),
                source_counts=sorted(source_counts), coordinate_products=sorted(products),
                shadowed_pairs=shadowed_pairs, changed_cells=changed,
                nearest_color_witnesses=nearest_witnesses, off_lattice_coordinate_check=moved,
                mutation_checks=sorted(mutation_checks), independent_semantic_check=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
