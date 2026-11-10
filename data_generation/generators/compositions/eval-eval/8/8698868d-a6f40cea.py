"""Hidden chamber maps: complete walls, then decode chamber populations.

Only Python and NumPy are used. Labels are rendered from complete geometry;
solve independently reads the visible input. No examples or scene state are cached.
"""
from collections import Counter
import hashlib
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(points, diagonal=False):
    remaining = set(points)
    groups = []
    steps = _N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _N4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    group.add(q)
        groups.append(group)
    return groups


def _points(mask):
    return set(map(tuple, np.argwhere(mask).tolist()))


def _box(points):
    rr, cc = zip(*points)
    return min(rr), max(rr), min(cc), max(cc)


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _edge(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)} | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)} | {(y, r) for y in range(t + 1, b)})


def _halo(points):
    return set(points) | {(r + dr, c + dc) for r, c in points for dr, dc in _N4}


def _inside(box):
    t, b, l, r = box
    return _area((t + 1, b - 1, l + 1, r - 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds"
    assert np.issubdtype(a.dtype, np.integer), "integer grids required"
    assert np.all((0 <= a) & (a <= 9)), "color bounds"
    return a


def _read(input_grid):
    """Recover every semantic role, key, wall and chamber from pixels alone."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    assert all(int(grid[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    frames = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for group in _components(_points(grid == color)):
            box = _box(group)
            t, b, l, r = box
            if b - t >= 8 and r - l >= 8 and group == _edge(box):
                frames.append((box, color))
    assert len(frames) == 1, "unique large rectangular frame required"
    main, frame = frames[0]
    masked, interior = _area(main), _inside(main)
    fill = Counter(int(grid[p]) for p in sorted(interior)).most_common(1)[0][0]
    dots = {p for p in interior if int(grid[p]) != fill}
    assert not dots or len({int(grid[p]) for p in dots}) == 1, "one visible dot role"
    dot_color = int(grid[min(dots)]) if dots else None
    assert all(not any((p[0] + dr, p[1] + dc) in dots for dr, dc in _N4) for p in dots)
    outside = _points(grid != bg) - masked
    keys, outlines = {}, []
    for group in _components(outside):
        box = _box(group)
        t, b, l, r = box
        colors = {int(grid[p]) for p in group}
        assert len(colors) == 1, "exterior objects must be separated"
        color = next(iter(colors))
        block = grid[t:b + 1, l:r + 1]
        if (5 <= b - t + 1 <= 6 and 5 <= r - l + 1 <= 6
                and not (_area(box) & masked) and _edge(box) <= group):
            assert set(map(int, np.unique(block))) <= {color, bg}
            holes = _points(block == bg)
            cavities = _components(holes)
            k = len(cavities)
            assert k not in keys, "duplicate key population"
            keys[k] = {"color": color, "box": box, "mask": block == bg,
                       "area": len(holes), "eight_count": len(_components(holes, True)),
                       "sizes": tuple(sorted(map(len, cavities)))}
        else:
            assert b - t >= 3 and r - l >= 3
            assert group == _edge(box) - masked, "visible fragment is not a complete rectangle minus the mask"
            assert _edge(box) & interior, "unoccluded outline is not a wall fragment"
            outlines.append({"box": box, "color": color, "visible": group})
    assert set(keys) == {0, 1, 2, 3}, "four keys with populations 0..3 required"
    assert 1 <= len(outlines) <= 3
    wall_colors = {o["color"] for o in outlines}
    assert len(wall_colors) == 1
    wall_color = next(iter(wall_colors))
    roles = [bg, frame, wall_color] + ([dot_color] if dots else []) + [keys[k]["color"] for k in range(4)]
    assert len(set(roles)) == len(roles), "distinct role colors required"
    assert fill == bg or fill not in roles
    assert len(set(int(grid[p]) for p in interior)) == 1 + bool(dots)
    walls = set().union(*(_edge(o["box"]) & interior for o in outlines))
    assert not walls & dots, "a clue cannot lie on a hidden wall"
    chambers = _components(interior - walls)
    assert all(len(c) >= 6 for c in chambers)
    counts = [len(c & dots) for c in chambers]
    assert all(k in keys for k in counts)
    t, b, l, r = main
    output = np.full((b - t - 1, r - l - 1), wall_color, dtype=np.int8)
    for cells, k in zip(chambers, counts):
        for y, x in cells:
            output[y - t - 1, x - l - 1] = keys[k]["color"]
    hidden_sets = [_edge(o["box"]) & interior for o in outlines]
    crossings = sum(len(a & b) for i, a in enumerate(hidden_sets) for b in hidden_sets[i + 1:])
    per_loop_split = any(len({k for c, k in zip(chambers, counts) if c <= _inside(o["box"])}) > 1
                         for o in outlines)
    population_at = {p: k for chamber, k in zip(chambers, counts) for p in chamber}
    wall_separates_counts = any(len({population_at[q] for dr, dc in _N4
                                    if (q := (y + dr, x + dc)) in population_at}) > 1
                               for y, x in walls)
    info = {"main": main, "background": bg, "fill": fill, "frame": frame,
            "dot_color": dot_color, "dots": dots, "wall_color": wall_color,
            "walls": walls, "outlines": outlines, "keys": keys,
            "chambers": chambers, "counts": counts, "crossings": crossings,
            "per_loop_split": per_loop_split, "wall_separates_counts": wall_separates_counts}
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the restored, populated strict interior; input is never modified."""
    return _read(input_grid)[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _geometry_ok(main, boxes, mode):
    masked, interior = _area(main), _inside(main)
    occupied, hidden = set(), []
    for box in boxes:
        full = _edge(box)
        visible = full - masked
        if (not visible or _box(visible) != box or len(_components(visible)) != 1
                or visible & _halo(occupied) or len(full & interior) < 3):
            return None
        occupied.update(visible)
        hidden.append(full & interior)
    walls = set().union(*hidden)
    chambers = _components(interior - walls)
    if not chambers or min(map(len, chambers)) < 6:
        return None
    crossings = sum(len(a & b) for i, a in enumerate(hidden) for b in hidden[i + 1:])
    if mode == "separate" and (crossings or len(chambers) < 3):
        return None
    if mode == "cross" and (crossings < 2 or len(chambers) < 5):
        return None
    if mode == "test" and (crossings < 2 or len(chambers) < 5):
        return None
    return walls, chambers, occupied


def _geometry(rng, n, mode):
    # General rectangles include corner cuts, spanning sides, nesting and hidden
    # intersections. Their exposed fragments must retain all four extrema.
    for _ in range(12):
        h, w = int(rng.integers(11, 16)), int(rng.integers(11, 18))
        t = int(rng.integers(4, 21 - h))
        l = int(rng.integers(4, 28 - w))
        main = (t, t + h - 1, l, l + w - 1)
        boxes, occupied = [], set()
        for _ in range(n):
            for _ in range(90):
                y0, y1 = sorted(map(int, rng.choice(np.arange(1, 22), 2, replace=False)))
                x0, x1 = sorted(map(int, rng.choice(np.arange(1, 29), 2, replace=False)))
                box = y0, y1, x0, x1
                if y1 - y0 < 3 or x1 - x0 < 3:
                    continue
                visible = _edge(box) - _area(main)
                if (not visible or _box(visible) != box or len(_components(visible)) != 1
                        or visible & _halo(occupied) or len(_edge(box) & _inside(main)) < 3):
                    continue
                boxes.append(box)
                occupied.update(visible)
                break
            else:
                break
        if len(boxes) == n:
            result = _geometry_ok(main, boxes, mode)
            if result is not None:
                return main, boxes, result
    return _constructive_geometry(rng, n, mode)


def _constructive_geometry(rng, n, mode):
    # Constructive continuous-coordinate sampler. Opposite-corner loops cross
    # only behind the panel; a third, nonconcentric nested loop divides a chamber
    # again. The inequalities preserve six-cell rooms and visible extrema.
    t, l = int(rng.integers(5, 7)), int(rng.integers(5, 9))
    b, r = int(rng.integers(t + 13, 20)), int(rng.integers(l + 14, 27))
    main = t, b, l, r
    y0, x0 = int(rng.integers(1, t - 2)), int(rng.integers(1, l - 2))
    y1, x1 = int(rng.integers(b + 2, 22)), int(rng.integers(r + 2, 29))
    if mode == "separate":
        a, d = int(rng.integers(t + 3, t + 6)), int(rng.integers(l + 3, l + 6))
        if (a - t - 1) * (d - l - 1) < 6:
            d += 1
        c, e = int(rng.integers(a + 3, b - 2)), int(rng.integers(d + 3, r - 2))
        if (b - c - 1) * (r - e - 1) < 6:
            e -= 1
        boxes = [(y0, a, x0, d), (c, y1, e, x1)]
    else:
        f, g = int(rng.integers(t + 3, t + 5)), int(rng.integers(l + 3, l + 5))
        if (f - t - 1) * (g - l - 1) < 6:
            g += 1
        c, e = int(rng.integers(f + 2, b - 6)), int(rng.integers(g + 2, r - 6))
        a, d = int(rng.integers(c + 4, b - 2)), int(rng.integers(e + 4, r - 2))
        yy, xx = int(rng.integers(y0 + 2, t)), int(rng.integers(x0 + 2, l))
        boxes = [(y0, a, x0, d), (c, y1, e, x1), (yy, f, xx, g)][:n]
    result = _geometry_ok(main, boxes, mode)
    assert result is not None, ("constructive geometry failed", main, boxes, mode)
    return main, boxes, result


def _cavity_mask(rng, h, w, count, witness):
    mask = np.zeros((h, w), dtype=bool)
    if count == 0:
        return mask
    if count == 1 and (witness or rng.random() < 0.3):
        mask[1:-1, 1:-1] = True
    elif count == 2 and witness:
        mask[1, 1] = mask[2, 2] = mask[2, 3] = True
    else:
        candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        for _ in range(24):
            mask[:] = False
            chosen = rng.choice(len(candidates), count, replace=False)
            for index in chosen:
                mask[candidates[int(index)]] = True
            if len(_components(_points(mask))) == count and len(_components(_points(~mask))) == 1:
                break
        else:
            mask[:] = False
            for p in [(1, 1), (1, 3), (3, 1)][:count]:
                mask[p] = True
        target = int(rng.integers(count, (h - 2) * (w - 2) + 1))
        for _ in range((h - 2) * (w - 2)):
            if int(mask.sum()) >= target:
                break
            available = []
            for p in candidates:
                if mask[p] or not any(mask[p[0] + dr, p[1] + dc] for dr, dc in _N4):
                    continue
                mask[p] = True
                if len(_components(_points(mask))) == count and len(_components(_points(~mask))) == 1:
                    available.append(p)
                mask[p] = False
            if not available:
                break
            mask[_pick(rng, available)] = True
    if bool(rng.integers(2)):
        mask = np.flipud(mask)
    if bool(rng.integers(2)):
        mask = np.fliplr(mask)
    return mask.copy()


def _pack_keys(rng, masked, visible, shapes):
    blocked = _halo(masked | visible)
    boxes = []
    for h, w in shapes:
        choices = [(r, c) for r in range(1, 30 - h) for c in range(1, 30 - w)
                   if not (_area((r, r + h - 1, c, c + w - 1)) & blocked)]
        if not choices:
            # Four <=6x6 keys fit in 27x6 cells, below every possible scene.
            return [
                (23, 23 + hh - 1, 1 + i * 7, i * 7 + ww)
                for i, (hh, ww) in enumerate(shapes)]
        r, c = _pick(rng, choices)
        box = r, r + h - 1, c, c + w - 1
        boxes.append(box)
        blocked.update(_halo(_area(box)))
    return boxes


def _dot_positions(rng, chamber, count):
    if count == 0:
        return set()
    chosen = set()
    positions = sorted(chamber)
    for i in rng.permutation(len(positions)):
        p = positions[int(i)]
        if p not in _halo(chosen):
            chosen.add(p)
            if len(chosen) == count:
                return chosen
    # Every grid subgraph is bipartite, so a six-cell chamber has a parity
    # class of at least three cells. This proves the fallback can place 0..3.
    parity = max(([p for p in positions if sum(p) % 2 == q] for q in (0, 1)), key=len)
    return {parity[int(i)] for i in rng.permutation(len(parity))[:count]}


def _make_pair(rng, n, mode, witness=False):
    main, boxes, (walls, chambers, visible) = _geometry(rng, n, mode)
    palette = list(map(int, rng.permutation(10)))
    bg, frame, fill, wall, dot = palette[:5]
    if rng.random() < 0.5:
        fill = bg
    colors = dict(zip(range(4), palette[5:9]))
    key_order = list(map(int, rng.permutation(4)))
    shapes = [(int(rng.integers(5, 7)), int(rng.integers(5, 7))) for _ in range(4)]
    key_boxes = _pack_keys(rng, _area(main), visible, shapes)
    grid = np.full((30, 30), bg, dtype=np.int8)
    for box in boxes:
        for p in _edge(box):
            grid[p] = wall
    for k, box, (h, w) in zip(key_order, key_boxes, shapes):
        mask = _cavity_mask(rng, h, w, k, witness)
        t, b, l, r = box
        grid[t:b + 1, l:r + 1] = np.where(mask, bg, colors[k])
    for p in _area(main):
        grid[p] = fill
    for p in _edge(main):
        grid[p] = frame
    required = [0, 1, 2, 3] if mode in ("cross", "test") else [1, 2] if witness else []
    counts = required + list(map(int, rng.integers(0, 4, len(chambers) - len(required))))
    rng.shuffle(counts)
    t, b, l, r = main
    output = np.full((b - t - 1, r - l - 1), wall, dtype=np.int8)
    for chamber, count in zip(chambers, counts):
        if count:
            for p in _dot_positions(rng, chamber, count):
                grid[p] = dot
        for y, x in chamber:
            output[y - t - 1, x - l - 1] = colors[count]
    # Crop surplus blank margins, retaining at least one blank cell on all sides.
    foreground = _points(grid != bg)
    a, z, c, d = _box(foreground)
    lo_r, lo_c = int(rng.integers(a)), int(rng.integers(c))
    hi_r, hi_c = int(rng.integers(z + 2, 31)), int(rng.integers(d + 2, 31))
    grid = grid[lo_r:hi_r, lo_c:hi_c].copy()
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    grid, output = np.rot90(grid, turns), np.rot90(output, turns)
    if reflect:
        grid, output = np.fliplr(grid), np.fliplr(output)
    return {"input": grid.copy(), "output": output.copy()}


def _count_argument(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(name + " is outside its legal integer range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count_argument(num_train, "num_train", 2, 10)
    ne = _count_argument(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            mode = "test" if split == "test" else "separate" if i == 0 else "cross" if i == 1 else "free"
            n = 3 if split == "test" else 2 if i < 2 else int(rng.integers(1, 3))
            for _ in range(8):
                pair = _make_pair(rng, n, mode, witness=split == "train" and i == 0)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input retries exhausted")
    # Training order conveys no convention.
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _area_rank_output(info):
    """Plausible competitor: rank key hole areas instead of counting cavities."""
    table = {k: info["keys"][q]["color"] for k, q in enumerate(
        sorted(info["keys"], key=lambda q: (info["keys"][q]["area"], q)))}
    t, b, l, r = info["main"]
    out = np.full((b - t - 1, r - l - 1), info["wall_color"], dtype=np.int8)
    for cells, k in zip(info["chambers"], info["counts"]):
        for y, x in cells:
            out[y - t - 1, x - l - 1] = table[k]
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {"train": [], "test": []}
    area_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            signature = grid.shape, np.asarray(grid, dtype=np.int8).tobytes()
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            expected, a = _read(grid)
            assert np.array_equal(out, expected), "incorrect chamber fill or wall completion"
            assert out.shape != grid.shape
            assert set(map(int, np.unique(out))) <= {a["wall_color"]} | {k["color"] for k in a["keys"].values()}
            assert len(set(a["counts"])) >= 2 or split == "train"
            analyses[split].append(a)
            if split == "train" and not np.array_equal(out, _area_rank_output(a)):
                area_witness = True
    train = analyses["train"]
    assert area_witness, "training must refute hole-area ranking"
    assert any(a["keys"][2]["eight_count"] < 2 and 2 in a["counts"] for a in train)
    assert any(a["keys"][1]["area"] > a["keys"][2]["area"] and {1, 2} <= set(a["counts"]) for a in train)
    assert any(a["crossings"] >= 2 and a["per_loop_split"] and len(a["chambers"]) >= 5 for a in train)
    assert any(a["wall_separates_counts"] for a in train), "omitting hidden walls merges unequal populations"
    assert any(a["crossings"] == 0 and len(a["outlines"]) == 2 for a in train)
    assert set().union(*(set(a["counts"]) for a in train)) == {0, 1, 2, 3}
    assert all(len(a["outlines"]) <= 2 for a in train)
    assert all(len(a["outlines"]) == 3 and a["crossings"] >= 2 for a in analyses["test"])
    return None


def self_test():
    """Stream 42 seeds; keep only small sets of digests and measurements."""
    seen, shapes, outputs, geometries, masks, populations, backgrounds = (set() for _ in range(7))
    test_geometries = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                before = grid.copy()
                predicted, a = _read(grid)
                assert np.array_equal(predicted, pair["output"])
                assert np.array_equal(grid, before), "solver modified its input"
                shapes.add(grid.shape)
                outputs.add(predicted.shape)
                backgrounds.add(a["background"])
                geometry = (a["main"], tuple(sorted(o["box"] for o in a["outlines"])))
                geometries.add(geometry)
                if split == "test":
                    t, b, l, r = a["main"]
                    test_geometries.add((b - t, r - l, tuple(sorted(
                        (o["box"][0] - t, o["box"][1] - t, o["box"][2] - l, o["box"][3] - l)
                        for o in a["outlines"]))))
                populations.add((len(a["chambers"]), tuple(sorted(a["counts"]))))
                for key in a["keys"].values():
                    masks.add((key["mask"].shape, key["mask"].tobytes()))
                digest.update(grid.tobytes())
                digest.update(predicted.tobytes())
        seen.add(digest.digest())
    assert len(seen) == 42 and len(shapes) >= 15 and len(outputs) >= 15
    assert len(geometries) >= 100 and len(masks) >= 60 and len(populations) >= 30
    assert len(test_geometries) >= 30, "test geometry collapsed to a fallback template"
    assert backgrounds == set(range(10))
    # Directly exercise the rare fallback rather than hoping random scenes
    # exhaust the ordinary sampler, including both smallest corner rooms.
    for seed in range(42):
        rng = np.random.default_rng(seed)
        for n, mode in ((1, "free"), (2, "free"), (2, "separate"), (2, "cross"), (3, "test")):
            main, boxes, (_, chambers, _) = _constructive_geometry(rng, n, mode)
            assert len(boxes) == n and min(map(len, chambers)) >= 6
    a, b = generate_puzzle(519, 2, 1), generate_puzzle(519, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    # All count combinations, not merely boundary requests.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(991 + nt * 4 + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    broken = generate_puzzle(9, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
