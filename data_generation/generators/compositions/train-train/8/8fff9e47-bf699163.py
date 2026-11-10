"""Coupled containment and corner-coded shell growth. Standard library + NumPy."""

from collections import deque
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(int(x) for x in a.flat)


def _components(mask):
    """Eight-connected components, independent of literal accent colors."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component = {first}
        queue = deque([first])
        while queue:
            r, c = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in remaining:
                        remaining.remove(p)
                        component.add(p)
                        queue.append(p)
        result.append(component)
    return result


def _parse(grid):
    """Read only visible geometry; no generator parameters are consulted."""
    grid = _grid(grid)
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    if int(counts[index]) * 2 <= grid.size:
        raise ValueError("background must occupy a strict majority")
    bg = int(colors[index])
    tiles, residual = [], set()
    for comp in _components(grid != bg):
        r0 = min(r for r, c in comp)
        c0 = min(c for r, c in comp)
        expected = {(r0 + dr, c0 + dc) for dr in (0, 1) for dc in (0, 1)}
        if comp == expected:
            tiles.append((r0, c0, grid[r0:r0 + 2, c0:c0 + 2].copy()))
        else:
            residual.update(comp)
    if not tiles or len(residual) < 4:
        raise ValueError("missing tiles or bracket")
    marker_colors = {int(grid[r, c]) for r, c in residual}
    if len(marker_colors) != 1:
        raise ValueError("bracket must have one separate color")
    marker = marker_colors.pop()
    if any(np.any(tile == marker) for r, c, tile in tiles):
        raise ValueError("bracket color cannot be payload")
    box = (min(r for r, c in residual), min(c for r, c in residual),
           max(r for r, c in residual), max(c for r, c in residual))
    r0, c0, r1, c1 = box
    if r1 - r0 < 4 or c1 - c0 < 4:
        raise ValueError("initial bracket spans must be at least five")
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in residual):
        raise ValueError("marker cells must lie on the initial perimeter")
    tiles.sort(key=lambda t: t[:2])
    return {"background": bg, "marker": marker, "box": box,
            "marker_cells": residual, "tiles": tiles}


def _entry(pos, box):
    """Expansions required before an entire 2x2 footprint fits."""
    r, c = pos
    r0, c0, r1, c1 = box
    return max(0, r0 - r, c0 - c, r + 1 - r1, c + 1 - c1)


def _harvest(tiles, box, *, grow=True, overlap=False, batch=False, axis=None):
    remaining = list(range(len(tiles)))
    order = []
    while remaining:
        t = len(order) if grow else 0
        r0, c0, r1, c1 = box
        r0, c0, r1, c1 = r0 - t, c0 - t, r1 + t, c1 + t

        def eligible(i):
            r, c = tiles[i][:2]
            if overlap:
                return r <= r1 and r + 1 >= r0 and c <= c1 and c + 1 >= c0
            rows = r0 <= r and r + 1 <= r1
            cols = c0 <= c and c + 1 <= c1
            return rows if axis == "row" else cols if axis == "col" else rows and cols

        available = [i for i in remaining if eligible(i)]
        if not available:
            break
        # tiles is in global top-left order; revisit after each single harvest.
        take = available if batch else available[:1]
        order.extend(take)
        taken = set(take)
        remaining = [i for i in remaining if i not in taken]
    return order


def _shells(codes):
    if not 1 <= len(codes) <= 15:
        raise ValueError("harvest must contain 1..15 tiles")
    out = np.asarray(codes[0], dtype=np.int8).copy()
    for radius, tile in enumerate(codes[1:], start=2):
        nw, ne, sw, se = map(int, np.asarray(tile).flat)
        next_grid = np.empty((2 * radius, 2 * radius), dtype=np.int8)
        next_grid[0, :radius] = next_grid[:radius, 0] = nw
        next_grid[0, radius:] = next_grid[:radius, -1] = ne
        next_grid[-1, :radius] = next_grid[radius:, 0] = sw
        next_grid[-1, radius:] = next_grid[radius:, -1] = se
        next_grid[1:-1, 1:-1] = out
        out = next_grid
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    order = _harvest(info["tiles"], info["box"])
    return _shells([info["tiles"][i][2] for i in order])


def _separated(a, b):
    return abs(a[0] - b[0]) >= 3 or abs(a[1] - b[1]) >= 3


def _marker_cells(rng, box, positions):
    r0, c0, r1, c1 = box
    # The blank halo prevents payloads and bracket fragments from becoming one
    # component, including where a tile straddles the invisible rectangle.
    halo = {(r + dr, c + dc) for r, c in positions
            for dr in range(-1, 3) for dc in range(-1, 3)}
    sides = [[(r0, c) for c in range(c0, c1 + 1)],
             [(r, c1) for r in range(r0, r1 + 1)],
             [(r1, c) for c in range(c0, c1 + 1)],
             [(r, c0) for r in range(r0, r1 + 1)]]
    sides = [[p for p in side if p not in halo] for side in sides]
    if any(not side for side in sides):
        return None
    chosen = {side[int(rng.integers(len(side)))] for side in sides}
    possible = sorted(set().union(*map(set, sides)))
    if len(possible) < 5:
        return None
    target = int(rng.integers(5, min(24, len(possible)) + 1))
    run_bias = float(rng.choice([0., 2., 8.]))
    while len(chosen) < target:
        options = [p for p in possible if p not in chosen]
        scores = np.array([1 + run_bias * sum(
            abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in chosen)
            for p in options], dtype=float)
        chosen.add(options[int(rng.choice(len(options), p=scores / scores.sum()))])
    return chosen


_PATTERNS = ((0, 1, 2, 3), (0, 0, 1, 1), (0, 1, 0, 1),
             (0, 1, 1, 0), (0, 0, 0, 1), (0, 1, 2, 0), (0, 0, 0, 0))


def _make_pair(rng, count, teaching, seen):
    for attempt in range(250):
        max_span = min(9, 28 - 2 * count)
        bh = int(rng.integers(5, max_span + 1))
        bw = int(rng.integers(5, max_span + 1))
        height = int(rng.integers(max(18, bh + 2 * count + 2), 31))
        width = int(rng.integers(max(18, bw + 2 * count + 2), 31))
        r0 = int(rng.integers(1 if teaching == "anchor" else 0, height - bh + 1))
        c0 = int(rng.integers(width - bw + 1))
        box = r0, c0, r0 + bh - 1, c0 + bw - 1
        groups = {}
        for r in range(height - 1):
            for c in range(width - 1):
                d = _entry((r, c), box)
                groups.setdefault(d, []).append((r, c))
        n_initial = 2 if teaching else int(rng.integers(1, min(3, count - 1) + 1))
        delays = [0] * n_initial
        for i in range(n_initial, count):
            delays.append(int(rng.integers(1, i + 1)))
        if teaching:
            delays[2] = 1
        if count > n_initial + 1:
            delays[-1] = count - 1
        positions = []
        failed = False
        for i, d in enumerate(delays):
            options = [p for p in groups.get(d, [])
                       if all(_separated(p, q) for q in positions)]
            if teaching == "anchor" and d == 0:
                options = [p for p in options if p[0] >= r0 + 2]
            if teaching == "anchor" and i == 2:
                options = [p for p in options if p[0] == r0 - 1]
            if not options:
                failed = True
                break
            positions.append(options[int(rng.integers(len(options)))])
        if failed:
            continue
        # A final straddler is a decisive witness against intersection/center
        # selection. It never becomes fully contained, so cannot power growth.
        fr0, fc0, fr1, fc1 = r0 - count, c0 - count, box[2] + count, box[3] + count
        options = [p for p in groups.get(count + 1, [])
                   if p[0] <= fr1 and p[0] + 1 >= fr0
                   and p[1] <= fc1 and p[1] + 1 >= fc0
                   and all(_separated(p, q) for q in positions)]
        if not options:
            continue
        positions.append(options[int(rng.integers(len(options)))])
        extra_count = int(rng.integers(0, 4))
        for _ in range(extra_count):
            options = [p for d, ps in groups.items() if d > count for p in ps
                       if all(_separated(p, q) for q in positions)]
            if options:
                positions.append(options[int(rng.integers(len(options)))])
        marker_cells = _marker_cells(rng, box, positions)
        if marker_cells is None:
            continue
        colors = [int(x) for x in rng.permutation(10)]
        bg, marker = colors[:2]
        palette_size = int(rng.integers(4 if teaching else 2, 9))
        palette = colors[2:2 + palette_size]
        codes, signatures = [], set()
        for i in range(len(positions)):
            if teaching == "anchor":
                pattern = _PATTERNS[0]
            elif teaching == "repeat" and i == 0:
                pattern = _PATTERNS[-1]
            elif teaching == "repeat" and i == 1:
                pattern = _PATTERNS[0]
            else:
                allowed = [p for p in _PATTERNS if len(set(p)) <= palette_size]
                pattern = allowed[int(rng.integers(len(allowed)))]
            for _ in range(100):
                labels = [int(x) for x in rng.choice(palette, size=len(set(pattern)), replace=False)]
                signature = tuple(labels[j] for j in pattern)
                if teaching != "anchor" or signature not in signatures:
                    break
            signatures.add(signature)
            codes.append(np.array(signature, dtype=np.int8).reshape(2, 2))
        if not teaching and len(codes) >= 4 and rng.random() < .35:
            a, b = [int(x) for x in rng.choice(count, size=2, replace=False)]
            codes[b] = codes[a].copy()
        inp = np.full((height, width), bg, dtype=np.int8)
        for r, c in marker_cells:
            inp[r, c] = marker
        for (r, c), code in zip(positions, codes):
            inp[r:r + 2, c:c + 2] = code
        if _key(inp) in seen:
            continue
        info = _parse(inp)
        order = _harvest(info["tiles"], box)
        if len(order) != count:
            raise AssertionError("planned activation schedule did not harvest exactly its tiles")
        if teaching == "anchor":
            if (order == sorted(order)
                    or order == _harvest(info["tiles"], box, batch=True)
                    or order == _harvest(info["tiles"], box, axis="row")
                    or order == _harvest(info["tiles"], box, axis="col")):
                continue
        seen.add(_key(inp))
        return {"input": inp, "output": _shells([info["tiles"][i][2] for i in order])}
    raise RuntimeError("could not place a separated, informative harvest")


def _count(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt = _count(num_train, 2, 10, "num_train")
    ns = _count(num_test, 1, 4, "num_test")
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # A per-puzzle withheld harvest length, not a fixed split/size template.
    heldout = int(rng.integers(3, 11))
    anchors = [k for k in range(4, 8) if k != heldout]
    first_two = [int(x) for x in rng.choice(anchors, size=2, replace=False)]
    training_counts = first_two + [int(rng.choice([k for k in range(3, 11) if k != heldout]))
                                   for _ in range(nt - 2)]
    test_counts = [heldout] + [int(rng.integers(3, 11)) for _ in range(ns - 1)]
    result, seen = {"train": [], "test": []}, set()
    for split, counts in (("train", training_counts), ("test", test_counts)):
        for i, k in enumerate(counts):
            teaching = ("anchor" if i == 0 else "repeat" if i == 1 else None) if split == "train" else None
            result[split].append(_make_pair(rng, k, teaching, seen))
    validate_generated_puzzle(result)
    return result


def _coordinate_answer(codes):
    """Check shell pixels by depth/quadrant, not the border-growing routine."""
    k = len(codes)
    out = np.empty((2 * k, 2 * k), dtype=np.int8)
    for r in range(2 * k):
        for c in range(2 * k):
            radius = max(k - r, r - k + 1, k - c, c - k + 1)
            out[r, c] = codes[radius - 1][int(r >= k), int(c >= k)]
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, training_counts, train_infos = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _parse(inp)
            tiles, box = info["tiles"], info["box"]
            order = _harvest(tiles, box)
            k = len(order)
            assert 3 <= k <= 10 and k + 1 <= len(tiles) <= k + 4
            assert all(_separated(a[:2], b[:2]) for j, a in enumerate(tiles) for b in tiles[j + 1:])
            # An entry-threshold audit checks the feedback trajectory directly.
            left = set(range(len(tiles)))
            for t, i in enumerate(order):
                possible = [j for j in left if _entry(tiles[j][:2], box) <= t]
                assert possible and i == min(possible)
                left.remove(i)
            assert left and all(_entry(tiles[i][:2], box) > k for i in left)
            assert any(_entry(tiles[i][:2], box) > 0 for i in order)
            r0, c0, r1, c1 = box
            assert any(_entry(tiles[i][:2], box) == k + 1
                       and tiles[i][0] <= r1 + k and tiles[i][0] + 1 >= r0 - k
                       and tiles[i][1] <= c1 + k and tiles[i][1] + 1 >= c0 - k for i in left)
            codes = [tiles[i][2] for i in order]
            assert out.shape == (2 * k, 2 * k)
            assert np.array_equal(out, _coordinate_answer(codes))
            assert info["background"] not in out and info["marker"] not in out
            for radius, code in enumerate(codes, start=1):
                a, b = k - radius, k + radius - 1
                assert np.array_equal(code, out[np.ix_([a, b], [a, b])])
            if split == "train":
                training_counts.append(k)
                train_infos.append((info, order))
    first, order = train_infos[0]
    tiles, box = first["tiles"], first["box"]
    assert all(len(set(map(int, t[2].flat))) == 4 for t in tiles)
    assert len({_key(t[2]) for t in tiles}) == len(tiles)
    assert order != sorted(order)
    assert order != _harvest(tiles, box, batch=True)
    assert order != _harvest(tiles, box, axis="row")
    assert order != _harvest(tiles, box, axis="col")
    assert len(_harvest(tiles, box, grow=False)) < len(order)
    assert len(_harvest(tiles, box, overlap=True)) > len(order)
    assert len(tiles) > len(order)  # ignoring the bracket adds shells
    second, second_order = train_infos[1]
    assert any(len(np.unique(second["tiles"][i][2])) == 1 for i in second_order)
    assert len(set(training_counts[:2])) == 2
    test_k = np.asarray(puzzle["test"][0]["output"]).shape[0] // 2
    assert test_k not in training_counts


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test():
    shapes, lengths, windows, palettes, initial_counts, patterns = set(), set(), set(), set(), set(), set()
    roles, marker_counts, normalized, signatures = set(), set(), set(), set()
    repeated = uniform = clipped = edge = gaps = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp), out)
                info = _parse(inp)
                tiles, box = info["tiles"], info["box"]
                order = _harvest(tiles, box)
                k = len(order)
                shapes.add(inp.shape)
                lengths.add(k)
                windows.add((box[2] - box[0] + 1, box[3] - box[1] + 1))
                palettes.add(len(np.unique(inp)))
                roles.add((info["background"], info["marker"]))
                marker_counts.add(len(info["marker_cells"]))
                initial_counts.add(sum(_entry(t[:2], box) == 0 for t in tiles))
                codes = [tuple(map(int, tiles[i][2].flat)) for i in order]
                repeated += len(set(codes)) < len(codes)
                uniform += any(len(set(code)) == 1 for code in codes)
                for code in codes:
                    labels = {}
                    patterns.add(tuple(labels.setdefault(x, len(labels)) for x in code))
                clipped += box[0] - k < 0 or box[1] - k < 0 or box[2] + k >= inp.shape[0] or box[3] + k >= inp.shape[1]
                edge += box[0] == 0 or box[1] == 0 or box[2] == inp.shape[0] - 1 or box[3] == inp.shape[1] - 1
                gaps += any(_entry(tiles[i][:2], box) >= 3 for i in order)
                labels = {}
                normalized.add((inp.shape, tuple(labels.setdefault(int(x), len(labels)) for x in inp.flat)))
                signatures.add(_key(inp))
        # Color relabeling and translation preserve geometry-driven behavior.
        pair = puzzle["train"][0]
        perm = np.random.default_rng(seed + 10000).permutation(10)
        assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
        info = _parse(pair["input"])
        # Removing a non-extremal marker cell changes its density/connectivity,
        # but cannot change the geometric window or its harvest.
        for p in sorted(info["marker_cells"]):
            rest = info["marker_cells"] - {p}
            rest_box = (min(r for r, c in rest), min(c for r, c in rest),
                        max(r for r, c in rest), max(c for r, c in rest))
            if rest_box == info["box"] and len(rest) >= 4:
                changed = pair["input"].copy()
                changed[p] = info["background"]
                assert np.array_equal(solve(changed), pair["output"])
                break
        if max(pair["input"].shape) <= 28:
            info = _parse(pair["input"])
            padded = np.pad(pair["input"], 1, constant_values=info["background"])
            assert np.array_equal(solve(padded), pair["output"])
    assert len(shapes) >= 40 and lengths == set(range(3, 11))
    assert len(windows) >= 15 and len(roles) >= 30 and len(marker_counts) >= 10
    assert len(palettes) >= 5 and {1, 2, 3} <= initial_counts
    assert len(patterns) >= 7 and len(normalized) >= 200 and len(signatures) == 210
    assert min(repeated, uniform, clipped, edge, gaps) > 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(12000 + 10 * nt + ns, nt, ns)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ns
    assert _equal(generate_puzzle(7123, 5, 3), generate_puzzle(7123, 5, 3))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, semantic invariants, all legal counts")
