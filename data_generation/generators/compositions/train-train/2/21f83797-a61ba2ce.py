"""ARC family: competing colors from orientation-addressed rectangle corners.

Only the two public grids of a pair are needed by solve.  No parent module,
file, random state, or training example is consulted by the solver.
"""
from collections import deque
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer 1..30 by 1..30 grid of colors 0..9")
    return a.astype(np.int8)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _parse(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[int(np.argmax(counts))])
    if int(counts.max()) <= a.size // 2:
        raise ValueError("background must occupy more than half the input")
    todo = set(map(tuple, np.argwhere(a != bg).tolist()))
    components = []
    while todo:
        point = min(todo)
        todo.remove(point)
        stack, cells = [point], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        if len({int(a[p]) for p in cells}) != 1:
            raise ValueError("foreground components must be monochrome and separated")
        components.append(cells)
    dots = [cells[0] for cells in components if len(cells) == 1]
    pieces = [cells for cells in components if len(cells) != 1]
    if len(dots) != 2 or not 2 <= len(pieces) <= 4:
        raise ValueError("need two dots and two to four Ls")
    wall = int(a[dots[0]])
    if int(a[dots[1]]) != wall:
        raise ValueError("the two dots must have the same color")
    top, bottom = sorted(p[0] for p in dots)
    left, right = sorted(p[1] for p in dots)
    if bottom - top < 3 or right - left < 3:
        raise ValueError("strict interior must be at least 2 by 2")
    sources, positions = {}, {}
    for cells in pieces:
        r, c = min(p[0] for p in cells), min(p[1] for p in cells)
        if len(cells) != 3 or max(p[0] for p in cells) != r + 1 or max(p[1] for p in cells) != c + 1:
            raise ValueError("each piece must be a 2 by 2 L-tromino")
        missing = {(0, 0), (0, 1), (1, 0), (1, 1)} - {(rr - r, cc - c) for rr, cc in cells}
        if len(missing) != 1:
            raise ValueError("invalid L")
        mr, mc = missing.pop()
        # The missing corner faces inward: missing BR -> destination TL, etc.
        corner = 3 - (2 * mr + mc)
        color = int(a[cells[0]])
        if corner in sources or color == wall or color in sources.values():
            raise ValueError("use distinct orientations and distinct source colors")
        sources[corner], positions[corner] = color, (r, c)
    return {"grid": a, "bg": bg, "wall": wall,
            "bounds": (top, bottom, left, right), "sources": sources,
            "positions": positions,
            "geometry": (bottom - top - 1, right - left - 1),
            "diagonal": int(a[top, left]) == wall}


def _front_output(shape, bg, wall, bounds, sources):
    top, bottom, left, right = bounds
    h, w = bottom - top - 1, right - left - 1
    rr, cc = np.indices((h, w))
    corners = sorted(sources)
    distances = np.stack([
        np.abs(rr - (h - 1 if k // 2 else 0))
        + np.abs(cc - (w - 1 if k % 2 else 0)) for k in corners
    ])
    minimum = distances.min(axis=0)
    tied = (distances == minimum).sum(axis=0) > 1
    palette = np.array([sources[k] for k in corners], dtype=np.int8)
    interior = palette[distances.argmin(axis=0)]
    interior[tied] = wall
    out = np.full(shape, bg, dtype=np.int8)
    out[top + 1:bottom, left + 1:right] = interior
    out[[top, bottom], :] = wall
    out[:, [left, right]] = wall
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase loose pieces, extend the anchor lines, and resolve corner fronts."""
    p = _parse(input_grid)
    return _front_output(p["grid"].shape, p["bg"], p["wall"], p["bounds"], p["sources"])


def _axis(rng, interior, ordinary, common):
    minimum = max(9, interior + (6 if ordinary else 2))
    if common is not None:
        size = common
    elif rng.random() < 0.16:
        size = 30
    else:
        size = int(rng.integers(minimum, min(30, minimum + 9) + 1))
    slack = size - interior - 2
    if ordinary:
        before = int(rng.integers(2, slack - 1))
    else:
        choice = int(rng.integers(4))
        before = 0 if choice == 0 else slack if choice == 1 else int(rng.integers(slack + 1))
    return size, before


def _scatter(rng, shape, bg, wall, bounds, sources, reverse=None, ordinary=False):
    top, bottom, left, right = bounds
    # Anchor diagonal is independent of the active source corners.
    dots = [(top, left), (bottom, right)] if rng.integers(2) else [(top, right), (bottom, left)]
    candidates = [(r, c) for r in range(shape[0] - 1) for c in range(shape[1] - 1)]
    corners = sorted(sources)
    for _ in range(60):
        a = np.full(shape, bg, dtype=np.int8)
        for p in dots:
            a[p] = wall
        positions, successful = {}, True
        for k in corners:
            missing = 3 - k
            found = False
            for raw in rng.permutation(len(candidates)):
                r, c = candidates[int(raw)]
                if any(abs(r - rr) < 3 and abs(c - cc) < 3 for rr, cc in positions.values()):
                    continue
                cells = [(r + j // 2, c + j % 2) for j in range(4) if j != missing]
                if any(abs(rr - dr) + abs(cc - dc) <= 1 for rr, cc in cells for dr, dc in dots):
                    continue
                for p in cells:
                    a[p] = sources[k]
                positions[k] = (r, c)
                found = True
                break
            if not found:
                successful = False
                break
        if not successful:
            continue
        order = sorted(corners, key=lambda k: positions[k])
        if reverse is not None and order != (list(reversed(corners)) if reverse else corners):
            continue
        if ordinary:
            # At least one loose colored cell visibly disappears into exterior background.
            exterior = np.ones(shape, dtype=bool)
            exterior[top:bottom + 1, left:right + 1] = False
            exterior[[top, bottom], :] = False
            exterior[:, [left, right]] = False
            if not np.any(exterior & (a != bg) & (a != wall)):
                continue
        return a
    raise RuntimeError("could not scatter separated Ls")


def _geometry(rng, index, training):
    if training and index == 0:
        h = int(rng.choice([3, 5, 7, 9]))
        w = int(rng.choice([v for v in (3, 5, 7, 9, 11) if v != h]))
        return h, w
    if training and index == 1:
        h, w = int(rng.choice([3, 5, 7, 9])), int(rng.choice([4, 6, 8, 10]))
        return (h, w) if rng.integers(2) else (w, h)
    if not training and index == 0:
        long = int(rng.integers(13, 25))
        return (2, long) if rng.integers(2) else (long, 2)
    cap = 12 if training else 23
    kind = int(rng.integers(4))
    if kind == 0:
        side = int(rng.integers(3, cap + 1))
        return side, side
    if kind == 1:
        return int(rng.integers(3, 6)), int(rng.integers(8, cap + 1))
    if kind == 2:
        return int(rng.integers(8, cap + 1)), int(rng.integers(3, 6))
    return int(rng.integers(3, cap + 1)), int(rng.integers(3, cap + 1))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    digits = [int(x) for x in rng.permutation(10)]
    bg0, wall0 = digits[:2]
    core = digits[2:6]
    stable_roles = bool(rng.integers(2))
    common = (int(rng.integers(26, 31)), int(rng.integers(26, 31))) if rng.random() < 0.24 else None
    diag = [0, 3] if rng.integers(2) else [1, 2]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        training = split == "train"
        for index in range(count):
            if training and index < 2:
                corners = diag if index == 0 else [k for k in range(4) if k not in diag]
                colors = core[:2]
                bg, wall = bg0, wall0
            else:
                n = (3 if training and index == 2 else 4 if training and index == 3
                     else 3 if not training and index == 0 and num_train == 2
                     else int(rng.choice([2, 3, 4], p=[0.4, 0.4, 0.2])))
                corners = sorted(int(x) for x in rng.choice(4, size=n, replace=False))
                if stable_roles:
                    bg, wall = bg0, wall0
                    pool = [c for c in range(10) if c not in (bg, wall)]
                else:
                    perm = [int(x) for x in rng.permutation(10)]
                    bg, wall, pool = perm[0], perm[1], perm[2:]
                colors = [int(x) for x in rng.choice(pool, size=n, replace=False)]
            sources = dict(zip(corners, colors))
            h, w = _geometry(rng, index, training)
            ordinary = training and index == 0
            for _ in range(40):
                height, top = _axis(rng, h, ordinary, None if common is None else common[0])
                width, left = _axis(rng, w, ordinary, None if common is None else common[1])
                bounds = (top, top + h + 1, left, left + w + 1)
                inp = _scatter(rng, (height, width), bg, wall, bounds, sources,
                               reverse=(index == 0) if training and index < 2 else None,
                               ordinary=ordinary)
                if _key(inp) not in seen:
                    break
            else:
                raise RuntimeError("could not construct distinct input")
            seen.add(_key(inp))
            puzzle[split].append({"input": inp, "output": _front_output(inp.shape, bg, wall, bounds, sources)})
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    p = _parse(pair["input"])
    out = _grid(pair["output"])
    assert out.shape == p["grid"].shape
    assert not np.array_equal(out, p["grid"])
    assert set(np.unique(out)) <= set(np.unique(p["grid"]))
    top, bottom, left, right = p["bounds"]
    h, w = p["geometry"]
    tie_count, wins, square_difference = 0, set(), False
    # Cellwise conditions separately check walls, exterior, winners, and ties.
    for r, c in np.ndindex(out.shape):
        if r in (top, bottom) or c in (left, right):
            expected = p["wall"]
        elif top < r < bottom and left < c < right:
            distances = {k: abs(r - (bottom if k // 2 else top))
                         + abs(c - (right if k % 2 else left)) for k in p["sources"]}
            nearest = [k for k, d in distances.items() if d == min(distances.values())]
            expected = p["sources"][nearest[0]] if len(nearest) == 1 else p["wall"]
            tie_count += len(nearest) > 1
            if len(nearest) == 1:
                wins.add(nearest[0])
            # Counterfactual: normalize both axes to a square before competing.
            normalized = {k: abs((r - top - 1) * (w - 1) - ((h - 1) * (w - 1) if k // 2 else 0))
                          + abs((c - left - 1) * (h - 1) - ((h - 1) * (w - 1) if k % 2 else 0))
                          for k in p["sources"]}
            nn = [k for k, d in normalized.items() if d == min(normalized.values())]
            square_expected = p["sources"][nn[0]] if len(nn) == 1 else p["wall"]
            square_difference |= square_expected != expected
        else:
            expected = p["bg"]
        assert int(out[r, c]) == expected, f"incorrect cell {r},{c}"
    assert wins == set(p["sources"]), "every source must own part of the chamber"
    p["ties"] = tie_count
    p["square_difference"] = square_difference
    return p


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            p = _check_pair(pair)
            key = _key(p["grid"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            analyses[split].append(p)
    train = analyses["train"]
    assert set().union(*(set(p["sources"]) for p in train)) == set(range(4))
    assert any(p["ties"] for p in train) and any(p["ties"] == 0 for p in train)
    assert any(p["square_difference"] for p in train), "show the effect of actual rectangle distances"
    # Ablation of B: give the earliest loose L the earliest active corner,
    # without reading its missing cell. Distinct hues make a moved owner visible.
    scan_order_fails = False
    for p in train:
        order = sorted(p["positions"], key=p["positions"].get)
        wrong = dict(zip(sorted(p["sources"]), [p["sources"][k] for k in order]))
        scan_order_fails |= not np.array_equal(
            _front_output(p["grid"].shape, p["bg"], p["wall"], p["bounds"], wrong),
            _front_output(p["grid"].shape, p["bg"], p["wall"], p["bounds"], p["sources"]))
    assert scan_order_fails, "show that source scan order is not the corner address"
    history = {}
    for p in train:
        for corner, color in p["sources"].items():
            history.setdefault(color, set()).add(corner)
    assert any(len(v) > 1 for v in history.values()), "show that colors do not fix corners"
    assert any(t > 0 and b < p["grid"].shape[0] - 1 and l > 0 and r < p["grid"].shape[1] - 1
               for p in train for t, b, l, r in [p["bounds"]]), "show full-line continuation"


def _wave_reference(p):
    """Self-authored check by unit-speed four-neighbor growth, not a protected solver."""
    h, w = p["geometry"]
    maps, keys = [], sorted(p["sources"])
    for k in keys:
        distance = np.full((h, w), -1, dtype=int)
        start = (h - 1 if k // 2 else 0, w - 1 if k % 2 else 0)
        distance[start] = 0
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and distance[rr, cc] < 0:
                    distance[rr, cc] = distance[r, c] + 1
                    queue.append((rr, cc))
        maps.append(distance)
    answer = np.full((h, w), p["wall"], dtype=np.int8)
    for r, c in np.ndindex(answer.shape):
        times = [int(d[r, c]) for d in maps]
        first = min(times)
        if times.count(first) == 1:
            answer[r, c] = p["sources"][keys[times.index(first)]]
    return answer


def self_test():
    signatures, shapes, geometries, palettes, subsets = (set() for _ in range(5))
    backgrounds, walls, diagonals, counts = (set() for _ in range(4))
    metrics = {"pairs": 0, "edge_anchors": 0, "tie_pairs": 0, "no_tie_pairs": 0,
               "wide_tests": 0, "tall_tests": 0, "common_canvas_puzzles": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        sig = tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))
        assert sig == tuple(_key(p[k]) for s in ("train", "test") for p in again[s] for k in ("input", "output"))
        signatures.add(sig)
        all_pairs = puzzle["train"] + puzzle["test"]
        metrics["common_canvas_puzzles"] += len({p["input"].shape for p in all_pairs}) == 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _parse(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(p["grid"].shape)
                geometries.add(p["geometry"])
                palettes.add((p["bg"], p["wall"], tuple(sorted(p["sources"].items()))))
                subsets.add(tuple(sorted(p["sources"])))
                backgrounds.add(p["bg"])
                walls.add(p["wall"])
                diagonals.add(p["diagonal"])
                counts.add(len(p["sources"]))
                top, bottom, left, right = p["bounds"]
                inner = pair["output"][top + 1:bottom, left + 1:right]
                ties = int(np.count_nonzero(inner == p["wall"]))
                metrics["pairs"] += 1
                metrics["edge_anchors"] += top == 0 or left == 0 or bottom == p["grid"].shape[0] - 1 or right == p["grid"].shape[1] - 1
                metrics["tie_pairs"] += ties > 0
                metrics["no_tie_pairs"] += ties == 0
                if split == "test":
                    assert np.array_equal(inner, _wave_reference(p))
                    h, w = p["geometry"]
                    metrics["wide_tests"] += w >= 6 * h
                    metrics["tall_tests"] += h >= 6 * w
                    for transform in (lambda a: a.T, lambda a: np.rot90(a), lambda a: np.fliplr(a)):
                        assert np.array_equal(solve(transform(pair["input"])), transform(pair["output"]))
                    permutation = np.array([(c + 3) % 10 for c in range(10)], dtype=np.int8)
                    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
        infos = [_parse(p["input"]) for p in puzzle["train"]]
        test_h, test_w = _parse(puzzle["test"][0]["input"])["geometry"]
        assert max(test_h, test_w) / min(test_h, test_w) > max(max(p["geometry"]) / min(p["geometry"]) for p in infos)
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(geometries) >= 60 and len(palettes) >= 80
    assert len(subsets) >= 9 and counts == {2, 3, 4}
    assert len(backgrounds) >= 8 and len(walls) >= 8 and len(diagonals) == 2
    assert all(metrics[k] > 0 for k in metrics)
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(42000 + 10 * n + m, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {n}/{m}")
    # Tight/maximal interiors, multiway ties, and all-edge chambers are not
    # left to random coverage. A second scattering must preserve the answer.
    rng = np.random.default_rng(773)
    for shape, bounds, src in (
        ((9, 12), (0, 3, 0, 3), {0: 2, 1: 3, 2: 4, 3: 5}),
        ((9, 9), (0, 8, 0, 8), {0: 2, 1: 3, 2: 4, 3: 5}),
        ((9, 9), (0, 8, 0, 8), {0: 2, 1: 3, 2: 4}),
        ((9, 30), (0, 3, 0, 29), {0: 2, 3: 5}),
        ((30, 30), (0, 29, 0, 29), {0: 2, 1: 3, 2: 4, 3: 5}),
    ):
        inp = _scatter(rng, shape, 0, 1, bounds, src)
        out = _front_output(shape, 0, 1, bounds, src)
        p = _check_pair({"input": inp, "output": out})
        assert np.array_equal(solve(inp), out)
        top, bottom, left, right = bounds
        assert np.array_equal(_wave_reference(p), out[top + 1:bottom, left + 1:right])
        relocated = _scatter(rng, shape, 0, 1, bounds, src)
        assert not np.array_equal(inp, relocated)
        assert np.array_equal(solve(relocated), out)
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "interior_shapes": len(geometries), "palettes": len(palettes),
            "corner_subsets": len(subsets), "backgrounds": len(backgrounds),
            "wall_colors": len(walls), **metrics}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
