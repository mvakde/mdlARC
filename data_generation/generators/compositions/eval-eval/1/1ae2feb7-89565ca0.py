"""Room clock: periodic instructions advance on repaired enclosed regions.

All roles and the entrance direction are recovered from input pixels. The
generator's lattice is only a construction aid, never used by solve().
"""
from collections import deque
from math import lcm

import numpy as np


def _neighbors(point, shape):
    y, x = point
    for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
        if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
            yield yy, xx


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a


def _regions(walls):
    """Label bounded four-connected free regions; -1 means exterior/wall."""
    free = np.pad(~walls, 1, constant_values=True)
    visited = np.zeros(free.shape, dtype=bool)
    labels = np.full(walls.shape, -1, dtype=np.int16)
    areas = []
    for yy, xx in np.argwhere(free):
        start = int(yy), int(xx)
        if visited[start]:
            continue
        todo = deque([start])
        visited[start] = True
        cells, outside = [], False
        while todo:
            p = todo.popleft()
            cells.append(p)
            outside |= (p[0] in (0, free.shape[0] - 1)
                        or p[1] in (0, free.shape[1] - 1))
            for q in _neighbors(p, free.shape):
                if free[q] and not visited[q]:
                    visited[q] = True
                    todo.append(q)
        if not outside:
            label = len(areas)
            for y, x in cells:
                labels[y - 1, x - 1] = label
            areas.append(len(cells))
    return labels, areas


def _repair(a, background, wall):
    """Simultaneous one-pixel completion; never bridge background."""
    visible = a == wall
    support = np.zeros(a.shape, dtype=bool)
    support[:, 1:-1] |= visible[:, :-2] & visible[:, 2:]
    support[1:-1, :] |= visible[:-2, :] & visible[2:, :]
    return visible | ((a != background) & support)


def _in_box(p, box):
    t, l, b, r = box
    return t <= p[0] <= b and l <= p[1] <= r


def _trace(a, color, box):
    points = {tuple(map(int, p)) for p in np.argwhere(a == color)}
    if len(points) < 4:
        return None
    links = {p: [q for q in _neighbors(p, a.shape) if q in points] for p in points}
    endpoints = [p for p in points if len(links[p]) == 1]
    if len(endpoints) != 2 or any(len(links[p]) != 2 for p in points - set(endpoints)):
        return None
    roots = [p for p in endpoints if not _in_box(p, box)]
    if len(roots) != 1:
        return None
    path, previous, current = [], None, roots[0]
    while True:
        path.append(current)
        following = [q for q in links[current] if q != previous]
        if not following:
            break
        previous, current = current, following[0]
        if len(path) > len(points):
            return None
    if len(path) != len(points):
        return None
    # Exactly two exterior guide pixels lead straight into the outer wall.
    if (any(_in_box(p, box) for p in path[:2])
            or any(not _in_box(p, box) for p in path[2:])):
        return None
    direction = tuple(path[1][i] - path[0][i] for i in (0, 1))
    if tuple(path[2][i] - path[1][i] for i in (0, 1)) != direction:
        return None
    return path, direction


def _analyze(input_grid):
    a = _grid(input_grid)
    corners = [int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1])]
    background = max(set(corners), key=corners.count)
    assert corners.count(background) >= 3, "background needs three canvas corners"
    candidates = []
    for c in np.unique(a):
        color = int(c)
        if color == background:
            continue
        positions = np.argwhere(a == color)
        t, l = map(int, positions.min(axis=0))
        b, r = map(int, positions.max(axis=0))
        if b - t < 4 or r - l < 4:
            continue
        if not all(a[p] == color for p in ((t, l), (t, r), (b, l), (b, r))):
            continue
        if not (np.all(a[t, l:r + 1] != background)
                and np.all(a[b, l:r + 1] != background)
                and np.all(a[t:b + 1, l] != background)
                and np.all(a[t:b + 1, r] != background)):
            continue
        candidates.append((color, (t, l, b, r)))
    assert len(candidates) == 1, "one corner-supported outer frame required"
    wall, box = candidates[0]
    t, l, b, r = box
    walls = _repair(a, background, wall)
    assert (walls[t, l:r + 1].all() and walls[b, l:r + 1].all()
            and walls[t:b + 1, l].all() and walls[t:b + 1, r].all())
    labels, areas = _regions(walls)
    assert areas, "no enclosed rooms"
    routes = []
    for c in np.unique(a):
        color = int(c)
        if color not in (background, wall):
            traced = _trace(a, color, box)
            if traced is not None:
                routes.append((color, *traced))
    assert len(routes) == 1, "route must be a unique unbranched chain"
    guide, path, direction = routes[0]
    assert walls[path[2]], "route must cross the outer wall"
    order, entries = [], []
    for step, p in enumerate(path):
        label = int(labels[p])
        if label >= 0 and (not order or label != order[-1]):
            assert label not in order, "route re-enters a room"
            order.append(label)
            entries.append(step)
    assert set(order) == set(range(len(areas))), "route must visit every room"
    root = path[0]
    dy, dx = direction
    cursor = root[0] - dy, root[1] - dx
    assert 0 <= cursor[0] < a.shape[0] and 0 <= cursor[1] < a.shape[1]
    assert a[cursor] == background, "source and guide need a background gap"
    runs, source_cells = [], set()
    while 0 <= cursor[0] < a.shape[0] and 0 <= cursor[1] < a.shape[1]:
        color = int(a[cursor])
        if color == background:
            cursor = cursor[0] - dy, cursor[1] - dx
            continue
        assert color not in (wall, guide)
        length, near = 0, cursor
        while (0 <= cursor[0] < a.shape[0] and 0 <= cursor[1] < a.shape[1]
               and a[cursor] == color):
            source_cells.add(cursor)
            length += 1
            cursor = cursor[0] - dy, cursor[1] - dx
        runs.append({"color": color, "length": length, "near": near})
    assert runs, "empty program"
    outer = np.ones(a.shape, dtype=bool)
    outer[t:b + 1, l:r + 1] = False
    allowed = source_cells | set(path[:2])
    assert all(tuple(map(int, p)) in allowed for p in np.argwhere(outer & (a != background)))
    interior_colors = set(map(int, np.unique(a[t:b + 1, l:r + 1]))) - {background, wall, guide}
    assert len(interior_colors) <= 1, "at most one wall-corruption color"
    noise = next(iter(interior_colors), None)
    if noise is not None:
        assert noise not in {run["color"] for run in runs}
        assert np.all(walls[a == noise]), "corruption must be supported wall pixels"
    # No payload is hidden in rooms; after recovery every free pixel is room interior.
    assert np.all(labels[t + 1:b, l + 1:r][~walls[t + 1:b, l + 1:r]] >= 0)
    return {"input": a, "background": background, "wall": wall, "guide": guide,
            "noise": noise, "box": box, "walls": walls, "labels": labels,
            "areas": areas, "order": order, "entries": entries,
            "path": path, "direction": direction, "runs": runs}


def _render(info, variant="room"):
    out = info["input"].copy()
    t, l, b, r = info["box"]
    out[t:b + 1, l:r + 1] = info["background"]
    out[info["walls"]] = info["wall"]
    for tick, label in enumerate(info["order"]):
        clock = tick
        if variant == "pixels":
            clock = info["entries"][tick] - info["entries"][0]
        elif variant == "area":
            clock = sum(info["areas"][q] for q in info["order"][:tick])
        firing = [run for run in info["runs"] if clock % run["length"] == 0]
        if variant == "constant":
            firing = info["runs"][:1]
        if not firing:
            continue
        if variant == "short":
            winner = min(firing, key=lambda run: run["length"])
        elif variant == "long":
            winner = max(firing, key=lambda run: run["length"])
        elif variant == "far":
            winner = firing[-1]
        elif variant == "numeric":
            winner = min(firing, key=lambda run: run["color"])
        else:
            winner = firing[0]
        out[info["labels"] == label] = winner["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone; does not modify the supplied array."""
    return _render(_analyze(input_grid))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _extras(rng, base, caps, budget):
    values = list(base)
    for _ in range(budget):
        eligible = [i for i, value in enumerate(values) if value < caps[i]]
        if not eligible:
            break
        values[int(rng.choice(eligible))] += 1
    return values


def _snake(nr, nc, columns_first):
    if columns_first:
        return [(y, x) for x in range(nc)
                for y in (range(nr) if x % 2 == 0 else range(nr - 1, -1, -1))]
    return [(y, x) for y in range(nr)
            for x in (range(nc) if y % 2 == 0 else range(nc - 1, -1, -1))]


def _build(rng, lengths, colors, roles, count, witness, columns_first):
    """Paint a clean network from cells/groups, then add observable occlusion."""
    bg, wall, guide, noise = roles
    minimum_gaps = [int(colors[i] == colors[i - 1]) for i in range(1, len(colors))]
    source_min = sum(lengths) + sum(minimum_gaps) + 3  # gap + exterior guide
    options = [(nr, nc) for nr in range(1, 8) for nc in range(1, 6)
               if source_min + 4 * nc + 1 <= 30 and 4 * nr + 2 <= 30
               and count + int(witness) <= nr * nc <= min(28, count + max(3, count // 2))
               and (not witness or (nr >= 2 and nc >= 2))]
    if not options:
        return None
    nr, nc = options[int(rng.integers(len(options)))]
    # Width budget varies source gaps, margins and actual room widths together.
    base = [0, 0, 1] + minimum_gaps + [4] * nc
    caps = [4, 3, 3] + [2] * len(minimum_gaps) + [10] * nc
    spare = 30 - (sum(base) + sum(lengths) + 3)  # guide 2 + outer-frame endpoint 1
    values = _extras(rng, base, caps, int(rng.integers(spare + 1)))
    margin, right_margin, near_gap = values[:3]
    gaps = values[3:3 + len(minimum_gaps)]
    widths = values[3 + len(minimum_gaps):]
    source_width = sum(lengths) + sum(gaps)
    left = margin + source_width + near_gap + 2
    hbase = [0, 1] + [4] * nr
    hc = [3, 4] + [10] * nr
    height_values = _extras(rng, hbase, hc, int(rng.integers(30 - sum(hbase))))
    top, bottom_margin = height_values[:2]
    heights = height_values[2:]
    ys = [top] + [top + int(v) for v in np.cumsum(heights)]
    xs = [left] + [left + int(v) for v in np.cumsum(widths)]
    shape = (ys[-1] + 1 + bottom_margin, xs[-1] + 1 + right_margin)
    clean = np.zeros(shape, dtype=bool)
    for y in ys:
        clean[y, left:xs[-1] + 1] = True
    for x in xs:
        clean[top:ys[-1] + 1, x] = True
    cells = _snake(nr, nc, columns_first)
    sizes = [1] * count
    for _ in range(len(cells) - count):
        # A witness has a genuine merge before the last two rooms.
        limit = max(1, count - 2) if witness else count
        sizes[int(rng.integers(limit))] += 1
    groups, cursor = [], 0
    for size in sizes:
        group = cells[cursor:cursor + size]
        groups.append(group)
        cursor += size
        for (y1, x1), (y2, x2) in zip(group, group[1:]):
            if y1 == y2:
                x = xs[max(x1, x2)]
                clean[ys[y1] + 1:ys[y1 + 1], x] = False
            else:
                y = ys[max(y1, y2)]
                clean[y, xs[x1] + 1:xs[x1 + 1]] = False
    centers = {(y, x): ((ys[y] + ys[y + 1]) // 2, (xs[x] + xs[x + 1]) // 2)
               for y, x in cells}
    first = centers[cells[0]]
    path = [(first[0], left - 2)]
    targets = [centers[cell] for cell in cells]
    for target in targets:
        y, x = path[-1]
        while (y, x) != target:
            if y != target[0]:
                y += 1 if y < target[0] else -1
            else:
                x += 1 if x < target[1] else -1
            path.append((y, x))
    inp = np.full(shape, bg, dtype=np.int8)
    inp[clean] = wall
    source_cursor = left - 2 - near_gap
    for i, (length, color) in enumerate(zip(lengths, colors)):
        inp[first[0], source_cursor - length:source_cursor] = color
        source_cursor -= length
        if i < len(gaps):
            source_cursor -= gaps[i]
    for p in path:
        inp[p] = guide
    # Additional corruption is one cell per retained segment, never a corner,
    # junction, or a neighbor of another missing wall cell.
    visible = inp == wall
    supported = np.zeros(shape, dtype=bool)
    supported[:, 1:-1] |= visible[:, :-2] & visible[:, 2:]
    supported[1:-1, :] |= visible[:-2, :] & visible[2:, :]
    degree = np.zeros(shape, dtype=int)
    degree[:, 1:] += clean[:, :-1]
    degree[:, :-1] += clean[:, 1:]
    degree[1:, :] += clean[:-1, :]
    degree[:-1, :] += clean[1:, :]
    eligible = np.argwhere(visible & supported & (degree == 2))
    requested = int(rng.integers(1 if witness else 0, 5))
    for index in rng.permutation(len(eligible)):
        if requested == 0:
            break
        p = tuple(map(int, eligible[int(index)]))
        if all(inp[q] == wall for q in _neighbors(p, shape) if clean[q]):
            inp[p] = noise
            requested -= 1
    # Construction uses intended group order and overwrite superposition.
    labels, areas = _regions(clean)
    if len(areas) != count:
        return None
    group_labels = [int(labels[centers[group[0]]]) for group in groups]
    if len(set(group_labels)) != count:
        return None
    out = inp.copy()
    out[top:ys[-1] + 1, left:xs[-1] + 1] = bg
    out[clean] = wall
    for length, color in reversed(list(zip(lengths, colors))):
        for tick in range(0, count, length):
            out[labels == group_labels[tick]] = color
    try:
        info = _analyze(inp)
        if not np.array_equal(info["walls"], clean):
            return None
        if info["order"] != group_labels:
            return None
        if witness:
            if info["noise"] is None or len(set(info["areas"])) < 2:
                return None
            if any(np.array_equal(out, _render(info, rival)) for rival in ("pixels", "area", "constant")):
                return None
    except AssertionError:
        return None
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    period_options = [(2, 3), (2, 5), (3, 4), (3, 5), (4, 6)]
    a, b = period_options[int(rng.integers(len(period_options)))]
    palette = list(map(int, rng.permutation(10)))
    base_roles = palette[:4]
    witness_colors = palette[4:6]
    local_roles = bool(rng.integers(2))
    axis = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", ne)):
        for index in range(number):
            witness = split == "train" and index < 2
            if witness:
                lengths = [a, b] if index == 0 else [b, a]
                colors = witness_colors if index == 0 else witness_colors[::-1]
                count = lcm(a, b) + 1 + int(rng.integers(3))
            else:
                run_count = int(rng.choice([1, 2, 3], p=[.42, .40, .18]))
                lengths = list(map(int, rng.integers(1, 7, size=run_count)))
                for _ in range(32):
                    if sum(lengths) <= 13:
                        break
                    lengths[int(rng.integers(run_count))] = int(rng.integers(1, 7))
                # Finite safeguard: at most five decrements are needed.
                while sum(lengths) > 13:
                    lengths[int(np.argmax(lengths))] -= 1
                colors = None
                if split == "train":
                    count = int(rng.integers(max(lengths) + 1, 19))
                else:
                    count = int(rng.choice(list(range(1, 7)) + list(range(10, 23))))
            rotation = (axis + (int(rng.integers(2)) * 2) + int(split == "test")) % 4
            for _ in range(120):
                roles = (list(map(int, rng.choice([c for c in range(10) if c not in witness_colors],
                                                   4, replace=False))) if local_roles else base_roles)
                available = [c for c in range(10) if c not in roles]
                chosen = colors if colors is not None else list(map(int, rng.choice(available, len(lengths), replace=False)))
                # Very wide source words may require a lower room count; retries
                # preserve the requested witness and adjust ordinary sampling.
                pair = _build(rng, lengths, chosen, roles, count, witness, bool(rng.integers(2)))
                if pair is None:
                    if not witness:
                        count = max(max(lengths) + 1 if split == "train" else 1, count - 1)
                    continue
                pair = {k: np.rot90(v, rotation).copy() for k, v in pair.items()}
                if _key(pair["input"]) in seen:
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded room-clock construction failed")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _sequence(info, out):
    result = []
    for label in info["order"]:
        colors = np.unique(out[info["labels"] == label])
        assert len(colors) == 1, "each room must be painted uniformly"
        result.append(int(colors[0]))
    return result


def _wall_cycle_count(walls):
    """Euler's bounded-face count for the connected wall-pixel graph."""
    points = {tuple(map(int, p)) for p in np.argwhere(walls)}
    todo, visited = [next(iter(points))], set()
    while todo:
        p = todo.pop()
        if p in visited:
            continue
        visited.add(p)
        todo.extend(q for q in _neighbors(p, walls.shape) if q in points and q not in visited)
    assert visited == points, "wall network must be connected"
    edges = int(np.count_nonzero(walls[:, :-1] & walls[:, 1:]))
    edges += int(np.count_nonzero(walls[:-1, :] & walls[1:, :]))
    solid_faces = int(np.count_nonzero(walls[:-1, :-1] & walls[:-1, 1:]
                                     & walls[1:, :-1] & walls[1:, 1:]))
    return edges - len(points) + 1 - solid_faces


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_infos, test_infos = set(), [], []
    edges, short_near, long_near, later_collision = set(), False, False, False
    partial, damage, unequal = False, False, False
    rival_rejected = {r: False for r in ("constant", "short", "long", "far", "numeric", "pixels", "area")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyze(inp)
            assert _wall_cycle_count(info["walls"]) == len(info["areas"]), "Euler/flood count disagreement"
            raw_labels, _ = _regions(inp == info["wall"])
            assert raw_labels[info["path"][info["entries"][0]]] < 0, "repair ablation must open the entrance"
            assert np.array_equal(out, _render(info)), "wrong pulses, repair, paint, or preservation"
            assert not np.array_equal(inp, out)
            seq = _sequence(info, out)
            # Check every painted room against direct modular selection.
            for tick, color in enumerate(seq):
                candidates = [run for run in info["runs"] if tick % run["length"] == 0]
                assert color == (candidates[0]["color"] if candidates else info["background"])
            assert np.all(out[info["walls"]] == info["wall"])
            if split == "train":
                train_infos.append(info)
                t, l, b, r = info["box"]
                mask = info["walls"][t:b + 1, l:r + 1]
                hs = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
                vs = np.flatnonzero(np.any(mask[:-1, :] & mask[1:, :], axis=0))
                partial |= any(not mask[y].all() for y in hs) or any(not mask[:, x].all() for x in vs)
                damage |= info["noise"] is not None
                unequal |= len(set(info["areas"])) > 1
                for rival in rival_rejected:
                    rival_rejected[rival] |= not np.array_equal(out, _render(info, rival))
                for near_index, near in enumerate(info["runs"]):
                    for far in info["runs"][near_index + 1:]:
                        if near["color"] == far["color"]:
                            continue
                        wins, survives, later = False, False, False
                        for tick, actual in enumerate(seq):
                            firing = [run for run in info["runs"] if tick % run["length"] == 0]
                            if firing and firing[0] is near and far in firing and actual == near["color"]:
                                wins, later = True, later or tick > 0
                            if firing and firing[0] is far and actual == far["color"]:
                                survives = True
                        if wins and survives:
                            edges.add((near["color"], far["color"]))
                            short_near |= near["length"] < far["length"]
                            long_near |= near["length"] > far["length"]
                            later_collision |= later
            else:
                test_infos.append(info)
    assert short_near and long_near and later_collision, "missing collision evidence"
    assert any((b, a) in edges for a, b in edges), "fixed color priority not refuted"
    assert partial and damage and unequal, "topology/reconstruction evidence missing"
    assert all(rival_rejected.values()), "simpler alternative fits all demonstrations"
    train_axes = {int(i["direction"][0] != 0) for i in train_infos}
    test_axes = {int(i["direction"][0] != 0) for i in test_infos}
    assert len(train_axes) == len(test_axes) == 1 and train_axes != test_axes, "missing axis shift"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    """Bounded semantic diagnostics, metamorphisms, counts, and 42-seed diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    infos = [_analyze(p["input"]) for puzzle in puzzles
             for s in ("train", "test") for p in puzzle[s]]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({i["input"].shape for i in infos}) >= 35
    assert {i["direction"] for i in infos} == {(0, 1), (0, -1), (1, 0), (-1, 0)}
    assert len({len(i["order"]) for i in infos}) >= 12
    assert {len(i["runs"]) for i in infos} == {1, 2, 3}
    assert {r["length"] for i in infos for r in i["runs"]} == set(range(1, 7))
    assert len({i["background"] for i in infos}) >= 8
    assert len({i["wall"] for i in infos}) >= 8
    assert len({tuple(i["areas"][q] for q in i["order"]) for i in infos}) >= 100
    # Determinism and every legal explicit count combination, including 2/1.
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    for nt in range(2, 11):
        for ne in range(1, min(nt - 1, 4) + 1):
            p = generate_puzzle(902 + nt * 5 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Hand-computed room-color sequences: arithmetic does not call the solver.
    rng = np.random.default_rng(551)
    diagnostics = (([2, 3], [1, 2], [1, 0, 1, 2, 1, 0, 1, 0]),
                   ([3, 2], [2, 1], [2, 0, 1, 2, 1, 0, 2, 0]),
                   ([2, 2], [1, 1], [1, 0, 1, 0, 1, 0, 1, 0]),
                   ([3], [1], [1, 0, 0, 1, 0, 0, 1, 0]),
                   ([1, 4], [2, 1], [2, 2, 2, 2, 2, 2, 2, 2]),
                   ([2, 3], [1, 2], [1]))
    for d, (lengths, colors, expected) in enumerate(diagnostics):
        for _ in range(120):
            pair = _build(rng, lengths, colors, [0, 5, 8, 9], len(expected), d < 2, False)
            if pair is not None:
                break
        else:
            raise AssertionError("diagnostic construction failed")
        info = _analyze(pair["input"])
        assert _sequence(info, solve(pair["input"])) == expected
        assert len(info["order"]) == len(expected)
        if d < 2:
            assert len(set(info["areas"])) > 1
    # A hand-built inward spiral is neither a row nor a column snake. Its
    # nine room decisions are fixed before the decoder inspects the input.
    fixture = np.zeros((14, 19), dtype=np.int8)
    for y in (0, 4, 8, 12):
        fixture[y, 5:18] = 5
    for x in (5, 9, 13, 17):
        fixture[0:13, x] = 5
    expected = fixture.copy()
    fixture[2, 0:2] = 1
    expected[2, 0:2] = 1
    expected[2, 3:5] = 8
    cell_order = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2),
                  (2, 1), (2, 0), (1, 0), (1, 1)]
    route = [(2, 3)]
    for row, col in cell_order:
        target = 2 + 4 * row, 7 + 4 * col
        y, x = route[-1]
        while (y, x) != target:
            if y != target[0]:
                y += 1 if y < target[0] else -1
            else:
                x += 1 if x < target[1] else -1
            route.append((y, x))
    for p in route:
        fixture[p] = 8
    fixture[0, 7] = 9
    for tick, (row, col) in enumerate(cell_order):
        expected[1 + 4 * row:4 + 4 * row, 6 + 4 * col:9 + 4 * col] = 1 if tick % 2 == 0 else 0
    assert np.array_equal(solve(fixture), expected)
    # Deleting a true divider merges the first two rooms and changes later
    # addresses. Keep the guide intact and expose background at the open gap.
    merged = fixture.copy()
    merged[1, 9] = merged[3, 9] = 0
    info = _analyze(merged)
    assert len(info["order"]) == 8
    result = solve(merged)
    assert result[2, 11] == 1 and result[2, 15] == 0
    assert not np.array_equal(result, expected)
    for p in puzzles[0]["train"] + puzzles[0]["test"]:
        for k in range(4):
            assert np.array_equal(solve(np.rot90(p["input"], k)), np.rot90(p["output"], k))
        assert np.array_equal(solve(np.fliplr(p["input"])), np.fliplr(p["output"]))
        mapping = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
        assert np.array_equal(solve(mapping[p["input"]]), mapping[p["output"]])
        untouched = p["input"].copy()
        solve(p["input"])
        assert np.array_equal(p["input"], untouched)
    changed = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
    changed["test"][0]["output"][0, 0] = (int(changed["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(changed)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed output accepted")


if __name__ == "__main__":
    self_test()
    print("room-clock self_test passed (42 seeds and all legal counts)")
