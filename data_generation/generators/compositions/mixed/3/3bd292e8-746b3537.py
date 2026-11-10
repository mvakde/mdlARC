"""Divider-crossing parity, followed by adjacent equal-vector compression.

Only standard library and NumPy. No imports from parents, files, or latent state.
Generation and bounded self-tests are in-memory; this module writes no artifacts.
"""
from __future__ import annotations

from collections import deque
import hashlib
import numpy as np


_N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((a >= 0) & (a <= 9)), "invalid grid color")
    return a.astype(np.int8, copy=False)


def _components(mask):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        label = len(groups)
        stack, cells = [(r, c)], []
        labels[r, c] = label
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _N4:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] < 0:
                    labels[ny, nx] = label
                    stack.append((ny, nx))
        groups.append(cells)
    return labels, groups


def _boundary(p, h, w):
    return p[0] in (0, h - 1) or p[1] in (0, w - 1)


def _paint(input_grid):
    """Read the local key, recover the region tree, then paint at full resolution."""
    a = _grid(input_grid)
    h, w = a.shape
    _require(h >= 3 and w >= 4, "no room for field and ordered paint key")
    even, odd, field = (int(v) for v in a[-1, :3])
    _require(len({even, odd, field}) == 3, "paint key and field must be distinct")
    remaining = set(int(v) for v in np.unique(a)) - {even, odd, field}
    _require(len(remaining) == 1, "expected exactly four color roles")
    divider = remaining.pop()
    _require(np.count_nonzero(a == even) == np.count_nonzero(a == odd) == 1,
             "paint colors appear only in the two key cells")
    walls = a == divider
    _require(not np.any(walls[-1, :3]), "lower-left key must be inside the field")
    wl, paths = _components(walls)
    rl, regions = _components(~walls)
    _require(len(paths) >= 1 and len(regions) == len(paths) + 1,
             "each divider must add one field region")
    graph = [set() for _ in regions]
    turns, boundaries, endpoints_all = [], [], []
    for pid, cells in enumerate(paths):
        _require(len(cells) >= 2, "divider too short")
        endpoints, neighbors = [], set()
        cell_set = set(cells)
        for r, c in cells:
            degree = 0
            for dr, dc in _N4:
                y, x = r + dr, c + dc
                if 0 <= y < h and 0 <= x < w:
                    degree += int(wl[y, x] == pid)
                    if rl[y, x] >= 0:
                        neighbors.add(int(rl[y, x]))
            _require(degree in (1, 2), "divider branches, loops, or self-touches")
            if degree == 1:
                endpoints.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    y, x = r + dr, c + dc
                    if 0 <= y < h and 0 <= x < w:
                        _require(wl[y, x] in (-1, pid), "different dividers touch")
        _require(len(endpoints) == 2 and all(_boundary(p, h, w) for p in endpoints),
                 "dividers must be simple boundary-to-boundary paths")
        _require(len(neighbors) == 2, "divider must separate exactly two nonempty regions")
        u, v = sorted(neighbors)
        _require(v not in graph[u], "duplicate region adjacency")
        graph[u].add(v)
        graph[v].add(u)
        previous, current, directions = None, endpoints[0], []
        while current != endpoints[1]:
            choices = [(current[0] + dr, current[1] + dc) for dr, dc in _N4
                       if (current[0] + dr, current[1] + dc) in cell_set
                       and (current[0] + dr, current[1] + dc) != previous]
            _require(len(choices) == 1, "ambiguous divider traversal")
            nxt = choices[0]
            directions.append((nxt[0] - current[0], nxt[1] - current[1]))
            previous, current = current, nxt
        turns.append(sum(u != v for u, v in zip(directions, directions[1:])))
        boundaries.append(sum(_boundary(p, h, w) for p in cells))
        endpoints_all.append(tuple(endpoints))
    root = int(rl[-1, 0])
    depths = [-1] * len(regions)
    depths[root] = 0
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in sorted(graph[u]):
            if depths[v] < 0:
                depths[v] = depths[u] + 1
                queue.append(v)
            else:
                _require((depths[u] - depths[v]) % 2 == 1, "ambiguous crossing parity")
    _require(all(d >= 0 for d in depths), "disconnected region graph")
    _require(sum(map(len, graph)) == 2 * (len(regions) - 1), "region graph must be a tree")
    painted = np.full(a.shape, divider, dtype=np.int8)
    for rid, depth in enumerate(depths):
        painted[rl == rid] = odd if depth % 2 else even
    return painted, {"roles": (field, divider, even, odd), "labels": rl,
                     "depths": tuple(depths), "turns": tuple(turns),
                     "boundary_cells": tuple(boundaries), "endpoints": tuple(endpoints_all),
                     "divider_count": len(paths), "max_degree": max(map(len, graph)),
                     "max_depth": max(depths)}


def _run_starts(a, axis):
    changes = np.any(np.diff(a, axis=axis) != 0, axis=1 - axis)
    return np.r_[0, np.flatnonzero(changes) + 1]


def _collapse(a):
    """Each axis is tested on complete vectors of the same painted grid."""
    return a[np.ix_(_run_starts(a, 0), _run_starts(a, 1))].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    painted, _ = _paint(input_grid)
    return _collapse(painted)


def _line(points):
    cells = [points[0]]
    for y, x in points[1:]:
        r, c = cells[-1]
        _require(r == y or c == x, "polyline segment is not orthogonal")
        dr, dc = int(np.sign(y - r)), int(np.sign(x - c))
        while (r, c) != (y, x):
            r, c = r + dr, c + dc
            cells.append((r, c))
    return cells


def _parts(rng, total, count):
    if count == 1:
        return [total]
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, total), count - 1, replace=False))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def _spaced(rng, low, high, count, gap):
    end = high - (count - 1) * (gap - 1)
    _require(end - low + 1 >= count, "not enough path spacing")
    return [int(v) + i * (gap - 1) for i, v in enumerate(
        sorted(rng.choice(np.arange(low, end + 1), count, replace=False)))]


def _reflect(path, h, w, corner):
    return [(h - 1 - r if corner in (2, 3) else r,
             w - 1 - c if corner in (1, 2) else c) for r, c in path]


def _corner(rng, h, w, corner, a, b, complexity=3, outside_col=0, terminal=False):
    """A monotone staircase, optionally constrained outside a nested inner box."""
    count = int(rng.integers(1, min(complexity, a, b - outside_col) + 1))
    horizontal, vertical = _parts(rng, b - outside_col, count), _parts(rng, a, count)
    vertices, r, c = [(a, 0)], a, outside_col
    if c:
        vertices.append((r, c))
    for dx, dy in zip(horizontal, vertical):
        c += dx
        vertices.append((r, c))
        r -= dy
        vertices.append((r, c))
    path = _line(vertices)
    if terminal and corner != 3:
        extension = int(rng.integers(1, a + 1))
        path = [(r, 0) for r in range(a - extension, a)] + path
    return _reflect(path, h, w, corner)


def _meander(rng, h, w, center, vertical=False, max_changes=3, straight=False):
    span = h if vertical else w
    count = 0 if straight else int(rng.integers(1, min(max_changes, (span - 3) // 3) + 1))
    sites = _spaced(rng, 2, span - 3, count, 3) if count else []
    level, vertices = center, [(center, 0)]
    for site in sites:
        vertices.append((level, site))
        level = int(rng.choice([v for v in range(center - 1, center + 2) if v != level]))
        vertices.append((level, site))
    vertices.append((level, span - 1))
    path = _line(vertices)
    return [(c, r) for r, c in path] if vertical else path


def _paths(rng, h, w, style, rich=False):
    if style in ("anchor", "star"):
        chosen = [0, 1, 2] if style == "anchor" else [int(v) for v in
                 rng.choice(4, int(rng.integers(1, 5)), replace=False)]
        result = []
        for corner in chosen:
            a = int(rng.integers(2, (h - 4) // 2 + 1))
            low_b = 3 if corner == 3 else 2
            b = int(rng.integers(low_b, (w - 4) // 2 + 1))
            result.append(_corner(rng, h, w, corner, a, b,
                                  4 if rich else 2, terminal=style != "anchor" and rng.random() < .3))
        return result
    if style in ("bands", "crossings"):
        vertical = bool(rng.integers(2))
        thickness = w if vertical else h
        low, high = (4 if vertical else 2), thickness - 3
        gap = 4
        maximum = min(5, 1 + (high - low) // gap)
        count = 2 if style == "bands" else int(rng.integers(2 if rich else 1, maximum + 1))
        centers = _spaced(rng, low, high, count, gap)
        return [_meander(rng, h, w, c, vertical, 4 if rich else 2, style == "bands")
                for c in centers]
    if style == "nested":
        corner = int(rng.integers(4))
        maximum = min(5, 1 + (min(h, w) - 5) // 3)
        count = int(rng.integers(3 if rich else 2, maximum + 1))
        aa = _spaced(rng, 3, h - 2, count, 3)
        bb = _spaced(rng, 3, w - 2, count, 3)
        result, previous_b = [], -2
        for a, b in zip(aa, bb):
            result.append(_corner(rng, h, w, corner, a, b, 4 if rich else 2,
                                  outside_col=previous_b + 2))
            previous_b = b
        return result
    if style == "mixed":
        middle = int(rng.integers(5, h - 5))
        result = [_meander(rng, h, w, middle, max_changes=3 if rich else 2)]
        corners = [int(v) for v in rng.choice(4, int(rng.integers(1, 5)), replace=False)]
        for corner in corners:
            upper_a = middle - 3 if corner in (0, 1) else h - middle - 4
            a = int(rng.integers(2, upper_a + 1))
            b = int(rng.integers(3 if corner == 3 else 2, (w - 4) // 2 + 1))
            result.append(_corner(rng, h, w, corner, a, b, 4 if rich else 2))
        return result
    if style == "same_side":
        # Rotate a simple U-shaped cut; the lower-left three-cell condition is
        # checked on the finished input rather than rotating an existing coloring.
        first, second = sorted(int(v) for v in rng.choice(np.arange(2, h - 2), 2, replace=False))
        _require(second - first >= 3, "U is too tight")
        depth = int(rng.integers(2, w - 2))
        path = _line([(first, 0), (first, depth), (second, depth), (second, 0)])
        rotation = int(rng.integers(4))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in path:
            mask[r, c] = True
        mask = np.rot90(mask, rotation)
        return mask
    raise AssertionError("unknown layout")


def _key(a):
    return a.shape, a.tobytes()


def _pair(rng, style, fills, seen, rich=False):
    # At most 96 candidates; only the current candidate and <=14 accepted pairs
    # are retained. No logs, journals, backups, retries, or traces are written.
    for _ in range(96):
        try:
            h = int(rng.integers(14 if rich else 12, 31 if rich else 27))
            w = int(rng.integers(14 if rich else 12, 31 if rich else 27))
            paths = _paths(rng, h, w, style, rich)
            if isinstance(paths, np.ndarray):
                mask = paths
            else:
                mask = np.zeros((h, w), dtype=bool)
                for path in paths:
                    for r, c in path:
                        mask[r, c] = True
            _require(not np.any(mask[-1, :3]), "divider intersects paint key")
            pool = [v for v in range(10) if v not in fills]
            field, divider = (int(v) for v in rng.choice(pool, 2, replace=False))
            a = np.where(mask, divider, field).astype(np.int8)
            a[-1, :2] = fills
            painted, info = _paint(a)
            out = _collapse(painted)
            _require(out.shape[0] < a.shape[0] and out.shape[1] < a.shape[1],
                     "both axes must contain a repeated painted vector")
            if style == "anchor":
                _require(all(painted[r, c] == fills[1] for r, c in ((0, 0), (0, -1), (-1, -1))),
                         "other-corner anchor shortcut is not exposed")
            if rich:
                _require(max(info["turns"]) >= 2, "test must exercise bending paths")
            signature = _key(a)
            if signature in seen:
                continue
            seen.add(signature)
            return {"input": a, "output": out}
        except AssertionError:
            continue
    raise RuntimeError(f"could not construct {style} within bounded attempts")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    n = _count(num_train, 2, 10, "num_train")
    m = _count(num_test, 1, 4, "num_test")
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    fills = tuple(int(v) for v in rng.choice(10, 2, replace=False))
    seen = set()
    styles = ["anchor", "bands"]
    styles += [str(rng.choice(["star", "crossings", "nested", "mixed", "same_side"]))
               for _ in range(n - 2)]
    rng.shuffle(styles)
    train = [_pair(rng, style, fills, seen) for style in styles]
    test = [_pair(rng, str(rng.choice(["nested", "mixed", "crossings"])), fills, seen, rich=True)
            for _ in range(m)]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_separated_repeat(a, axis):
    vectors = np.moveaxis(a, axis, 0)
    keys = [v.tobytes() for v in vectors]
    return len(set(keys)) < len(keys)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid pair counts")
    seen, infos, keys = set(), {"train": [], "test": []}, set()
    anchor_evidence = recurrence_evidence = order_evidence = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _key(a)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            painted, info = _paint(a)
            rows, cols = _run_starts(painted, 0), _run_starts(painted, 1)
            expected = painted[np.ix_(rows, cols)]
            _require(np.array_equal(out, expected), "incorrect paint-then-collapse output")
            _require(out.shape[0] < a.shape[0] and out.shape[1] < a.shape[1], "both axes must shrink")
            _require(np.array_equal(_collapse(out), out), "output still has adjacent repeated vectors")
            # Every original painted cell, including walls, is recoverable by
            # expanding the exact runs. No crossing line is silently discarded.
            row_lengths = np.diff(np.r_[rows, a.shape[0]])
            col_lengths = np.diff(np.r_[cols, a.shape[1]])
            reconstructed = np.repeat(np.repeat(out, row_lengths, axis=0), col_lengths, axis=1)
            _require(np.array_equal(reconstructed, painted), "compression lost nonredundant cells")
            _require(np.array_equal(_collapse(painted[rows]), out), "axis order disagreement")
            _require(np.array_equal(_collapse(painted[:, cols]), out), "axis order disagreement")
            field, divider, even, odd = info["roles"]
            keys.add((even, odd))
            _require(set(np.unique(out)) == {divider, even, odd}, "a color role disappeared")
            _require(not np.array_equal(_collapse(a), out), "omitting parity has no effect")
            _require(painted.shape != out.shape, "omitting compression has no effect")
            infos[split].append(info)
            if split == "train":
                anchor_evidence |= all(painted[r, c] == odd for r, c in ((0, 0), (0, -1), (-1, -1)))
                recurrence_evidence |= any(_has_separated_repeat(out, axis) for axis in (0, 1))
                order_evidence |= _collapse(a).shape != out.shape
            else:
                _require(max(info["turns"]) >= 2, "test lacks bending-path shift")
    _require(len(keys) == 1, "generated puzzle must share its two key colors")
    _require(anchor_evidence, "training does not distinguish other-corner anchoring")
    _require(any(info["max_depth"] >= 2 for info in infos["train"]), "training lacks even return")
    _require(recurrence_evidence, "training does not distinguish global vector deduplication")
    _require(order_evidence, "training does not expose painting before compression")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded regressions and 42 seeds; retains hashes/metrics, no puzzle corpus."""
    # Two cuts: the even color returns, and separated equal wall/fill vectors stay.
    for vertical in (False, True):
        a = np.zeros((9, 11), dtype=np.int8)
        if vertical:
            a[:, [3, 7]] = 8
        else:
            a[[2, 6], :] = 8
        a[-1, :2] = (5, 3)
        expected = np.array([[5, 8, 3, 8, 5]], dtype=np.int8)
        if not vertical:
            expected = expected.T
        _require(np.array_equal(solve(a), expected), "two-cut parity/run regression")
    # Terminal boundary stretch and three bends; this is not a parent pair.
    a = np.zeros((10, 12), dtype=np.int8)
    for r, c in _line([(0, 0), (3, 0), (3, 4), (6, 4), (6, 11)]):
        a[r, c] = 8
    a[-1, :2] = (5, 3)
    painted, info = _paint(a)
    _require(info["turns"] == (3,) and info["boundary_cells"][0] > 2, "boundary regression")
    _require(painted[-1, 1] == 5 and painted[0, -1] == 3, "paint key/anchor regression")
    # All legal explicit counts; schema, counts and demonstrations are checked.
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(1700 + 10 * n + m, n, m)
            _require(len(p["train"]) == n and len(p["test"]) == m, "count ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4.0, 1), (4, None), (None, 1), (4, np.bool_(True))):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    shapes, structures, fills, dividers, turns, depths, degrees, signatures = (set() for _ in range(8))
    test_axes = set()
    boundary_cases = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _digest(puzzle)
        _require(signature == _digest(generate_puzzle(seed)), "seed not reproducible")
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                painted, info = _paint(a)
                _require(np.array_equal(solve(a), out), "input-only solver disagrees")
                shapes.add(a.shape)
                mask = a == info["roles"][1]
                structures.add((a.shape, hashlib.sha256(mask.tobytes()).digest()))
                fills.add(info["roles"][2:])
                dividers.add(info["divider_count"])
                turns.update(info["turns"])
                depths.add(info["max_depth"])
                degrees.add(info["max_degree"])
                boundary_cases += sum(v > 2 for v in info["boundary_cells"])
                if split == "test":
                    h, w = a.shape
                    for p, q in info["endpoints"]:
                        if {p[0], q[0]} == {0, h - 1}:
                            test_axes.add("vertical")
                        if {p[1], q[1]} == {0, w - 1}:
                            test_axes.add("horizontal")
        # Color permutation covariance checks roles rather than literal colors.
        if seed < 4:
            permutation = np.random.default_rng(seed + 987).permutation(10).astype(np.int8)
            a = puzzle["test"][0]["input"]
            _require(np.array_equal(solve(permutation[a]), permutation[solve(a)]), "palette dependence")
    _require(len(signatures) == 42 and len(shapes) >= 60 and len(structures) >= 170, "weak structure variation")
    _require(len(fills) >= 20 and min(dividers) == 1 and max(dividers) >= 4, "weak palette/count variation")
    _require(max(turns) >= 5 and max(depths) >= 3 and max(degrees) >= 3, "weak geometry/topology variation")
    _require(boundary_cases > 0 and test_axes == {"horizontal", "vertical"}, "missing boundary/orientation variation")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds; all legal counts; bounded in-memory checks)")
