"""Square-powered exploration of the common free space of two maps.

The only special literal colors are 0 (empty) and 9 (start marker).
The cell opposite 9 selects the battery and output color for that input.
"""

from collections import deque

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        cells = []
        todo = [(r, c)]
        seen[r, c] = True
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for dy, dx in _DIRS:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        result.append(cells)
    return result


def _square(cells):
    return (len(cells) == 4
            and max(r for r, _ in cells) - min(r for r, _ in cells) == 1
            and max(c for _, c in cells) - min(c for _, c in cells) == 1)


def _decode(input_grid):
    a = _grid(input_grid)
    if a.shape[0] % 2:
        raise ValueError("two equal vertically stacked maps are required")
    h = a.shape[0] // 2
    maps = (a[:h], a[h:])
    markers = np.argwhere(a == 9)
    if len(markers) != 1:
        raise ValueError("exactly one start marker is required")
    row, col = (int(v) for v in markers[0])
    marker_panel = row // h
    root = (row % h, col)
    target = int(maps[1 - marker_panel][root])
    if not 1 <= target <= 8:
        raise ValueError("the cell opposite the marker must select a color 1..8")
    components = [_components(m == target) for m in maps]
    counts = [sum(_square(c) for c in cs) for cs in components]
    free = (maps[0] == 0) & (maps[1] == 0)
    free[root] = True
    return {"maps": maps, "root": root, "target": target,
            "marker_panel": marker_panel, "components": components,
            "counts": counts, "budget": sum(counts), "free": free}


def _distances(free, root):
    """Four-neighbor shortest paths; -1 denotes blocked or unreachable."""
    h, w = free.shape
    distance = np.full((h, w), -1, dtype=np.int16)
    distance[root] = 0
    todo = deque([root])
    while todo:
        r, c = todo.popleft()
        for dr, dc in _DIRS:
            rr, cc = r + dr, c + dc
            if (0 <= rr < h and 0 <= cc < w and free[rr, cc]
                    and distance[rr, cc] < 0):
                distance[rr, cc] = distance[r, c] + 1
                todo.append((rr, cc))
    return distance


def _paint(distance, budget, color):
    return np.where((distance >= 0) & (distance <= budget), color, 0).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read every convention from this input, then count and explore."""
    info = _decode(input_grid)
    return _paint(_distances(info["free"], info["root"]),
                  info["budget"], info["target"])


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _walls(rng, h, w):
    """One map's terrain, sampled without reference to the other map."""
    mask = np.zeros((h, w), dtype=bool)
    style = int(rng.integers(5))
    if style == 0:
        mask = rng.random((h, w)) < rng.uniform(0.08, 0.22)
    elif style == 1:
        # Baffles with doors, in independently chosen directions.
        for _ in range(int(rng.integers(2, 5))):
            if rng.random() < 0.5:
                r = int(rng.integers(h))
                mask[r, :] = True
                c = int(rng.integers(w - 2))
                mask[r, c:c + int(rng.integers(2, 5))] = False
            else:
                c = int(rng.integers(w))
                mask[:, c] = True
                r = int(rng.integers(h - 2))
                mask[r:r + int(rng.integers(2, 5)), c] = False
    elif style == 2:
        for _ in range(int(rng.integers(2, 6))):
            r0, c0 = int(rng.integers(h - 1)), int(rng.integers(w - 1))
            r1 = min(h, r0 + int(rng.integers(2, 5)))
            c1 = min(w, c0 + int(rng.integers(2, 6)))
            if rng.random() < 0.5:
                mask[r0:r1, c0:c1] = True
            else:
                mask[r0:r1, c0] = mask[r0:r1, c1 - 1] = True
                mask[r0, c0:c1] = mask[r1 - 1, c0:c1] = True
                mask[int(rng.integers(r0, r1)), c0] = False
    elif style == 3:
        for _ in range(int(rng.integers(2, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(int(rng.integers(5, h + w))):
                mask[r, c] = True
                dr, dc = _DIRS[int(rng.integers(4))]
                r = min(h - 1, max(0, r + dr))
                c = min(w - 1, max(0, c + dc))
    else:
        # Staggered incomplete walls produce rooms, loops and narrow doors.
        for r in range(int(rng.integers(1, 4)), h, int(rng.integers(3, 5))):
            mask[r, :] = True
            door = int(rng.integers(1, w - 2))
            mask[r, door:door + int(rng.integers(2, 5))] = False
        mask |= rng.random((h, w)) < 0.025
    return mask


def _square_cells(r, c):
    return ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1))


def _ring(cells, h, w):
    local = set(cells)
    return {(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS
            if 0 <= r + dr < h and 0 <= c + dc < w} - local


def _target_ok(m, protected, cells, target):
    local = set(cells)
    if any(protected[cell] or m[cell] == target for cell in cells):
        return False
    return all(m[cell] != target for cell in _ring(local, *m.shape))


def _find_square(rng, maps, protected, target, panels):
    h, w = maps[0].shape
    for index in rng.permutation((h - 1) * (w - 1)):
        r, c = divmod(int(index), w - 1)
        cells = _square_cells(r, c)
        if all(_target_ok(maps[p], protected[p], cells, target) for p in panels):
            return cells
    raise RuntimeError("no battery placement")


def _set_cells(m, cells, color):
    for cell in cells:
        m[cell] = color


def _ablations(info, expected):
    """Concrete competing rules, reconstructed from the returned input."""
    maps, root, target = info["maps"], info["root"], info["target"]
    n, free = info["budget"], info["free"]
    d = _distances(free, root)
    variants = {"unlimited": np.where(d >= 0, target, 0),
                "pixel_area": _paint(d, sum(int(np.sum(m == target)) for m in maps) // 4, target)}
    all_squares = sum(_square(c) for m in maps for color in np.unique(m)
                      if color not in (0, 9) for c in _components(m == color))
    variants["all_colors"] = _paint(d, all_squares, target)
    merged_count = sum(_square(c) for c in _components((maps[0] == target) | (maps[1] == target)))
    variants["merge_before_count"] = _paint(d, merged_count, target)
    for p in range(2):
        one_map_free = maps[p] == 0
        one_map_free[root] = True
        variants[f"only_map_{p}"] = _paint(_distances(one_map_free, root), n, target)
        variants[f"only_count_{p}"] = _paint(d, info["counts"][p], target)
    rr, cc = np.indices(free.shape)
    variants["manhattan"] = np.where(free & (abs(rr - root[0]) + abs(cc - root[1]) <= n), target, 0)
    variants["plain_nor"] = np.where((maps[0] == 0) & (maps[1] == 0), target, 0)
    return {name: not np.array_equal(grid, expected) for name, grid in variants.items()}


def _make_pair(rng, split, index, target):
    anchor = split == "train" and index < 2
    budget = (2 + index if anchor else
              int(rng.integers(1, 4)) if split == "train" else int(rng.integers(4, 7)))
    for _ in range(100):
        h = int(rng.integers(9, 14 if split == "train" else 16))
        w = int(rng.integers(10, 18)) if split == "train" else int(rng.integers(19, 31))
        wall_colors = [int(x) for x in rng.choice([c for c in range(1, 9) if c != target], 2, replace=False)]
        maps = [np.where(_walls(rng, h, w), wall_colors[p], 0).astype(np.int8) for p in range(2)]
        protected = [np.zeros((h, w), dtype=bool) for _ in range(2)]
        if anchor or rng.random() < 0.62:
            root = (int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))
        else:
            # Boundaries are genuine geometry, not padded-away cases.
            root = (int(rng.integers(h)), int(rng.integers(w)))
            side = int(rng.integers(4))
            root = ((0, root[1]) if side == 0 else (h - 1, root[1]) if side == 1
                    else (root[0], 0) if side == 2 else (root[0], w - 1))
        r, c = root
        for p in range(2):
            protected[p][root] = True
            maps[p][root] = 0
        if anchor:
            # One-sided blockers and empty cells behind them expose geodesic
            # distance and the necessity of BOTH maps at budgets two and three.
            turn = int(rng.integers(4))
            vectors = ((-1, 0), (0, 1), (1, 0), (0, -1))
            for p in range(2):
                dr, dc = vectors[(turn + p) % 4]
                for k in (1, 2):
                    cell = (r + k * dr, c + k * dc)
                    for q in range(2):
                        protected[q][cell] = True
                        maps[q][cell] = wall_colors[q] if q == p and k == 1 else 0
                # The two opposite immediate neighbors remain empty.
                cell = (r - dr, c - dc)
                for q in range(2):
                    maps[q][cell] = 0
                    protected[q][cell] = True
        marker_panel = int(rng.integers(2))
        maps[marker_panel][root] = 9
        maps[1 - marker_panel][root] = target
        try:
            # An isolated non-target square is an explicit counting decoy.
            p = int(rng.integers(2))
            cells = _find_square(rng, maps, protected, target, (p,))
            ring = _ring(cells, h, w)
            if any(protected[p][cell] for cell in ring):
                continue
            _set_cells(maps[p], ring, 0)
            _set_cells(maps[p], cells, wall_colors[p])
            for cell in set(cells) | ring:
                protected[p][cell] = True

            remaining = budget
            if anchor and index == 1 or (not anchor and budget >= 2 and rng.random() < 0.30):
                # Coincident tokens still belong to two separate inventories.
                cells = _find_square(rng, maps, protected, target, (0, 1))
                for p in range(2):
                    _set_cells(maps[p], cells, target)
                remaining -= 2
            for j in range(remaining):
                p = j % 2 if anchor else int(rng.integers(2))
                cells = _find_square(rng, maps, protected, target, (p,))
                _set_cells(maps[p], cells, target)

            if anchor and index == 0 or (not anchor and rng.random() < 0.24):
                # Four isolated specks form a square only after overlay.
                cells = _find_square(rng, maps, protected, target, (0, 1))
                y, x = cells[0]
                _set_cells(maps[0], ((y, x), (y + 1, x + 1)), target)
                _set_cells(maps[1], ((y + 1, x), (y, x + 1)), target)
            for p in range(2):
                extra = int(rng.integers(0, 6))
                for _ in range(extra):
                    for flat in rng.permutation(h * w):
                        cell = divmod(int(flat), w)
                        if _target_ok(maps[p], protected[p], (cell,), target):
                            maps[p][cell] = target
                            break
            inp = np.vstack(maps)
            info = _decode(inp)
            if info["budget"] != budget:
                continue
            d = _distances(info["free"], root)
            if not np.any(d == budget + 1):
                continue
            out = _paint(d, budget, target)
            if anchor:
                evidence = _ablations(info, out)
                needed = ("unlimited", "pixel_area", "all_colors", "only_map_0", "only_map_1",
                          "only_count_0", "only_count_1", "manhattan", "plain_nor")
                if not all(evidence[k] for k in needed):
                    continue
                if index == 1 and not evidence["merge_before_count"]:
                    continue
            return {"input": inp, "output": out}
        except RuntimeError:
            continue
    raise RuntimeError("could not construct an informative map pair in 100 attempts")


def _signature(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    used = set()
    previous_target = None
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            target = int(rng.choice([c for c in range(1, 9) if c != previous_target]))
            previous_target = target
            for _ in range(12):
                pair = _make_pair(rng, split, index, target)
                key = _signature(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check semantics, geometry, inference evidence, and the intended shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nx = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nx <= 4 and nt > nx
    seen, train_info, test_info = set(), [], []
    ruled_out = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, tuple(int(x) for x in inp.flat))
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _decode(inp)
            assert out.shape == info["free"].shape
            assert all(len(c) == 1 or _square(c) for cs in info["components"] for c in cs), "merged battery/noise"
            assert len(next(c for c in info["components"][1 - info["marker_panel"]]
                            if info["root"] in c)) == 1, "start key must be a singleton"
            assert 0 <= info["budget"] <= 6
            d = _distances(info["free"], info["root"])
            assert np.array_equal(out, _paint(d, info["budget"], info["target"])), "incorrect flood"
            assert np.any(d == info["budget"] + 1), "the budget must visibly limit exploration"
            assert out[info["root"]] == info["target"]
            assert np.all((out == 0) | (out == info["target"]))
            occupied = ~((info["maps"][0] == 0) & (info["maps"][1] == 0))
            occupied[info["root"]] = False
            assert not np.any(out[occupied]), "signal traverses a wall"
            info["shape"] = out.shape
            if split == "train":
                train_info.append(info)
                ruled_out.update(k for k, changed in _ablations(info, out).items() if changed)
            else:
                test_info.append(info)
    required = {"unlimited", "pixel_area", "all_colors", "merge_before_count", "only_map_0", "only_map_1",
                "only_count_0", "only_count_1", "manhattan", "plain_nor"}
    assert required <= ruled_out, "demonstrations leave a named shortcut unchallenged"
    assert {2, 3} <= {i["budget"] for i in train_info}
    assert len({i["target"] for i in train_info}) >= 2, "relational color selection is not demonstrated"
    assert max(i["budget"] for i in train_info) < min(i["budget"] for i in test_info)
    assert max(i["shape"][1] for i in train_info) < min(i["shape"][1] for i in test_info)


def _same(a, b):
    return all(np.array_equal(x[k], y[k]) for s in ("train", "test")
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))


def _semantic_fixtures():
    """Small, hand-specified witnesses for double counting and blocked detours."""
    top, bottom = (np.zeros((9, 9), dtype=np.int8) for _ in range(2))
    top[0, 0], bottom[0, 0] = 9, 2
    top[6:8, 6:8] = bottom[6:8, 6:8] = 2
    expected = np.zeros((9, 9), dtype=np.int8)
    expected[0, :3] = 2
    expected[1, :2] = 2
    expected[2, 0] = 2
    inp = np.vstack((top, bottom))
    assert np.array_equal(solve(inp), expected)
    # A wall in only the lower map blocks the shortcut to (0, 2).
    bottom[0, 1] = 5
    expected[0, 1:3] = 0
    inp = np.vstack((top, bottom))
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(solve(np.vstack((bottom, top))), expected)
    # Recoloring the lower battery preserves occupancy, but removes one charge.
    bottom[6:8, 6:8] = 5
    shorter = np.zeros((9, 9), dtype=np.int8)
    shorter[0, 0] = shorter[1, 0] = 2
    assert np.array_equal(solve(np.vstack((top, bottom))), shorter)
    top[6:8, 6:8] = 5
    shorter[1, 0] = 0
    assert np.array_equal(solve(np.vstack((top, bottom))), shorter)


def self_test():
    """Bounded 42-seed diversity, count API, metamorphic and semantic checks."""
    _semantic_fixtures()
    assert _same(generate_puzzle(314159, 5, 2), generate_puzzle(314159, 5, 2))
    for nt in range(2, 11):
        for nx in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + nx, nt, nx)
            assert len(p["train"]) == nt and len(p["test"]) == nx
    for nt, nx in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nx)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {nt}/{nx}")
    shapes, palettes, budgets, roots, structures, reached, topologies = (set() for _ in range(7))
    marker_panels = set()
    boundary = overlap = detour = total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total += 1
                info = _decode(pair["input"])
                a, b = info["maps"]
                h, w = a.shape
                r, c = info["root"]
                assert np.array_equal(solve(pair["input"]), pair["output"])
                assert np.array_equal(solve(np.vstack((b, a))), pair["output"])
                shapes.add((h, w))
                palettes.add(tuple(int(x) for x in np.unique(pair["input"])))
                budgets.add(info["budget"])
                roots.add((r, c))
                marker_panels.add(info["marker_panel"])
                boundary += r in (0, h - 1) or c in (0, w - 1)
                squares = [{frozenset(cells) for cells in cs if _square(cells)} for cs in info["components"]]
                overlap += bool(squares[0] & squares[1])
                detour += _ablations(info, pair["output"])["manhattan"]
                reached.add(int(np.count_nonzero(pair["output"])))
                topologies.add((len(_components(a != 0)), len(_components(b != 0)),
                                len(_components(info["free"]))))
                # Normalize semantic roles, including per-panel wall colors.
                norm = np.where(pair["input"] == 0, 0,
                                np.where(pair["input"] == 9, 3,
                                         np.where(pair["input"] == info["target"], 2, 1)))
                structures.add((norm.shape, norm.tobytes()))
    assert len(shapes) >= 60 and len(palettes) >= 25
    assert budgets == set(range(1, 7)) and marker_panels == {0, 1}
    assert len(structures) >= 200 and len(roots) >= 50 and len(reached) >= 20
    assert len(topologies) >= 100 and boundary >= 10 and overlap >= 42 and detour >= 84
    return {"seeds": 42, "pairs": total, "panel_shapes": len(shapes),
            "palettes": len(palettes), "budgets": sorted(budgets), "start_positions": len(roots),
            "normalized_structures": len(structures), "reached_cell_counts": len(reached),
            "topology_profiles": len(topologies), "boundary_starts": int(boundary),
            "overlapping_batteries": int(overlap), "detour_sensitive_pairs": int(detour)}
