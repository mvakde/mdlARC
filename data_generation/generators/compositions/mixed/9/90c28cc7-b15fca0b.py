"""Collapse the pixel bands before drawing a unique shortest logical route.

Standard library and NumPy only. Generation and checks retain no seed state,
examples, logs, or execution traces. See task_spec.md for the family contract.
"""
from collections import deque
import hashlib
import json

import numpy as np


BACKGROUND = 0
TERMINAL = 2
ROUTE = 4
WALL_COLORS = (1, 3, 5, 6, 7, 8, 9)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_CANDIDATES = 768
MAX_PROFILES = 24


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _collapse(a):
    """Compare full original vectors, independently on the two axes; no crop."""
    rows = np.r_[0, np.flatnonzero(np.any(a[1:] != a[:-1], axis=1)) + 1]
    cols = np.r_[0, np.flatnonzero(np.any(a[:, 1:] != a[:, :-1], axis=0)) + 1]
    logical = a[np.ix_(rows, cols)].copy()
    heights = np.diff(np.r_[rows, a.shape[0]])
    widths = np.diff(np.r_[cols, a.shape[1]])
    return logical, heights, widths


def _neighbors(cell, shape):
    r, c = cell
    for dr, dc in DIRECTIONS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            yield rr, cc


def _shortest(mask, start, goal):
    """BFS with shortest-path multiplicity saturated at two."""
    h, w = mask.shape
    distance = np.full((h, w), -1, dtype=np.int16)
    ways = np.zeros((h, w), dtype=np.int8)
    previous = {}
    distance[start], ways[start] = 0, 1
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, mask.shape):
            if mask[q]:
                continue
            if distance[q] < 0:
                distance[q] = distance[p] + 1
                ways[q] = ways[p]
                previous[q] = p
                queue.append(q)
            elif distance[q] == distance[p] + 1:
                ways[q] = min(2, int(ways[q]) + int(ways[p]))
    if distance[goal] < 0:
        return None, 0
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return path[::-1], int(ways[goal])


def _roles_and_path(logical):
    palette = set(map(int, np.unique(logical)))
    walls = palette - {BACKGROUND, TERMINAL}
    if len(walls) != 1 or next(iter(walls)) not in WALL_COLORS:
        raise ValueError("Input must have background 0, terminals 2, and one wall color")
    if BACKGROUND not in palette:
        raise ValueError("Background is missing")
    endpoints = [tuple(map(int, p)) for p in np.argwhere(logical == TERMINAL)]
    if len(endpoints) != 2:
        raise ValueError("Collapsed input must have exactly two terminal cells")
    wall = next(iter(walls))
    path, count = _shortest(logical == wall, endpoints[0], endpoints[1])
    if count != 1:
        raise ValueError("Collapsed input must have one unique shortest route")
    return wall, path


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover logical cells and connect terminals; uses this input only."""
    logical, _, _ = _collapse(_grid(input_grid))
    _, path = _roles_and_path(logical)
    output = logical.copy()
    for cell in path[1:-1]:
        output[cell] = ROUTE
    return output


def _turns(path):
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _seam(logical):
    """A band boundary locally hidden in background but exposed elsewhere."""
    for a in (logical, logical.T):
        for i in range(1, len(a)):
            if (np.any((a[i] == BACKGROUND) & (a[i - 1] == BACKGROUND))
                    and np.any(a[i] != a[i - 1])):
                return True
    return False


def _single_witness(logical):
    return any(np.count_nonzero(a[i] != a[i - 1]) == 1
               for a in (logical, logical.T) for i in range(1, len(a)))


def _recurrence(logical):
    """Equal full vectors separated by a different vector must both survive."""
    return any(np.array_equal(a[i], a[j])
               for a in (logical, logical.T) for i in range(len(a))
               for j in range(i + 2, len(a)))


def _offroute_dead_end(logical, path, wall):
    used = set(path)
    return any(tuple(map(int, p)) not in used
               and sum(logical[q] != wall for q in _neighbors(tuple(map(int, p)), logical.shape)) == 1
               for p in np.argwhere(logical == BACKGROUND))


def _pixel_distance(blocked, source, target):
    """Distance between two whole terminal blocks, using free endpoint pixels.

    A flat list queue avoids retaining pixel paths. This is a counterfactual
    metric check, never part of solve. At most 900 cells are visited.
    """
    h, w = blocked.shape
    forbidden = blocked.ravel().tolist()
    targets = target.ravel().tolist()
    distance = [-1] * (h * w)
    queue = deque(map(int, np.flatnonzero(source)))
    for p in queue:
        distance[p] = 0
    while queue:
        p = queue.popleft()
        if targets[p]:
            return distance[p]
        r, c = divmod(p, w)
        neighbors = []
        if r:
            neighbors.append(p - w)
        if r + 1 < h:
            neighbors.append(p + w)
        if c:
            neighbors.append(p - 1)
        if c + 1 < w:
            neighbors.append(p + 1)
        for q in neighbors:
            if not forbidden[q] and distance[q] < 0:
                distance[q] = distance[p] + 1
                queue.append(q)
    return -1


def _metric_distances(logical, path, heights, widths, wall):
    """Compare pixel optimum to the best pixel realization of logical path."""
    row_owner = np.repeat(np.arange(logical.shape[0]), heights)
    col_owner = np.repeat(np.arange(logical.shape[1]), widths)
    source = ((row_owner[:, None] == path[0][0])
              & (col_owner[None, :] == path[0][1]))
    target = ((row_owner[:, None] == path[-1][0])
              & (col_owner[None, :] == path[-1][1]))
    expanded_walls = np.repeat(np.repeat(logical == wall, heights, axis=0), widths, axis=1)
    allowed = np.zeros(logical.shape, dtype=bool)
    for cell in path:
        allowed[cell] = True
    expanded_allowed = np.repeat(np.repeat(allowed, heights, axis=0), widths, axis=1)
    unrestricted = _pixel_distance(expanded_walls, source, target)
    restricted = _pixel_distance(~expanded_allowed, source, target)
    return unrestricted, restricted


def _maze_mask(rng):
    """A random spanning-tree corridor map with some walls opened into loops."""
    nr, nc = map(int, rng.integers(3, 6, size=2))
    mask = np.ones((2 * nr - 1, 2 * nc - 1), dtype=bool)
    mask[::2, ::2] = False
    root = (int(rng.integers(nr)), int(rng.integers(nc)))
    seen, stack = {root}, [root]
    while stack:
        p = stack[-1]
        choices = [q for q in _neighbors(p, (nr, nc)) if q not in seen]
        if not choices:
            stack.pop()
            continue
        q = choices[int(rng.integers(len(choices)))]
        mask[p[0] + q[0], p[1] + q[1]] = False
        seen.add(q)
        stack.append(q)
    closed = [(r, c) for r in range(mask.shape[0]) for c in range(mask.shape[1])
              if (r + c) % 2 and mask[r, c]]
    count = int(rng.integers(1, min(6, len(closed)) + 1))
    for i in rng.choice(len(closed), size=count, replace=False):
        mask[closed[int(i)]] = False
    return mask


def _barrier_mask(rng):
    """Different geometry: staggered barriers, including gaps and branches."""
    h = int(rng.integers(6, 12))
    n = int(rng.integers(1, 5))
    w = 2 * n + 1
    mask = np.zeros((h, w), dtype=bool)
    top_first = bool(rng.integers(2))
    for i in range(n):
        col = 2 * i + 1
        gap = int(rng.integers(1, max(2, h // 2)))
        if bool(i % 2) == top_first:
            mask[:h - gap, col] = True
        else:
            mask[gap:, col] = True
        if rng.random() < .35:
            occupied = np.flatnonzero(mask[:, col])
            if len(occupied) > 3:
                mask[int(rng.choice(occupied[1:-1])), col] = False
    if rng.random() < .45:
        options = [(r, c) for r in range(1, h - 1) for c in range(w)
                   if not mask[r, c]
                   and any(mask[q] for q in _neighbors((r, c), mask.shape))]
        if options:
            mask[options[int(rng.integers(len(options)))]] = True
    return mask


def _candidate(rng, kind):
    mask = _barrier_mask(rng) if kind == "barrier" else _maze_mask(rng)
    free = np.argwhere(~mask)
    # Boundary and interior endpoints are both eligible; no corner/parity code.
    if rng.random() < .45:
        boundary = np.array([p for p in free if p[0] in (0, mask.shape[0] - 1)
                             or p[1] in (0, mask.shape[1] - 1)])
        if len(boundary) >= 2:
            free = boundary
    if len(free) < 2:
        return None
    endpoints = free[rng.choice(len(free), size=2, replace=False)]
    a = mask.astype(np.int8)
    for p in endpoints:
        a[tuple(p)] = TERMINAL
    # Remove invisible latent boundaries, rather than pretending they exist.
    a, _, _ = _collapse(a)
    points = [tuple(map(int, p)) for p in np.argwhere(a == TERMINAL)]
    if len(points) != 2 or min(a.shape) < 3:
        return None
    path, count = _shortest(a == 1, points[0], points[1])
    if count != 1 or len(path) < 7 or _turns(path) < 2:
        return None
    manhattan = sum(abs(x - y) for x, y in zip(points[0], points[1]))
    if len(path) - 1 <= manhattan or np.count_nonzero(a == 0) <= len(path):
        return None
    if not _seam(a):
        return None
    # Rotation/reflection are part of geometry variation, not solving steps.
    a = np.rot90(a, int(rng.integers(4))).copy()
    if rng.integers(2):
        a = a[:, ::-1].copy()
    points = [tuple(map(int, p)) for p in np.argwhere(a == TERMINAL)]
    path, count = _shortest(a == 1, points[0], points[1])
    assert count == 1
    return a, path


def _axis_bands(rng, length, style):
    if style == "uniform":
        return np.full(length, int(rng.integers(2, min(4, 30 // length) + 1)), dtype=int)
    if style == "mild":
        bands = rng.integers(1, 4, size=length)
    else:
        bands = rng.integers(1, 3, size=length)
        peak = int(rng.integers(length))
        maximum = min(12, 30 - int(bands.sum()) + int(bands[peak]))
        bands[peak] = int(rng.integers(4, maximum + 1))
    if np.max(bands) == 1:
        bands[int(rng.integers(length))] = 2
    for _ in range(max(0, int(bands.sum()) - 30)):
        bands[int(rng.choice(np.flatnonzero(bands > 1)))] -= 1
    return bands


def _bands(rng, shape, must_reverse):
    if must_reverse:
        # Both one-axis and two-axis distortion can change route preference.
        styles = (("strong", "mild"), ("mild", "strong"), ("strong", "strong"))
    else:
        styles = (("uniform", "uniform"), ("uniform", "strong"),
                  ("strong", "uniform"), ("mild", "mild"), ("strong", "strong"))
    rstyle, cstyle = styles[int(rng.integers(len(styles)))]
    return _axis_bands(rng, shape[0], rstyle), _axis_bands(rng, shape[1], cstyle)


def _make_pair(rng, requirement, wall):
    """Rejection is bounded; candidates/profiles are discarded immediately."""
    must_reverse = requirement in ("metric", "combined")
    for _ in range(MAX_CANDIDATES):
        kind = "maze" if must_reverse else str(rng.choice(("maze", "barrier")))
        built = _candidate(rng, kind)
        if built is None:
            continue
        a, path = built
        if requirement == "recurrence" and not (_recurrence(a) and _single_witness(a)
                                                and _turns(path) >= 4
                                                and _offroute_dead_end(a, path, 1)):
            continue
        if requirement == "combined" and not (_turns(path) >= 4
                                               and _offroute_dead_end(a, path, 1)):
            continue
        for _ in range(MAX_PROFILES if must_reverse else 1):
            rows, cols = _bands(rng, a.shape, must_reverse)
            if must_reverse:
                short, restricted = _metric_distances(a, path, rows, cols, 1)
                if not 0 <= short < restricted:
                    continue
            logical = a.copy()
            logical[a == 1] = wall
            inp = np.repeat(np.repeat(logical, rows, axis=0), cols, axis=1)
            out = logical.copy()
            for cell in path[1:-1]:
                out[cell] = ROUTE
            return {"input": inp.astype(np.int8), "output": out.astype(np.int8)}
    raise RuntimeError("Bounded construction could not satisfy " + requirement)


def _signature(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count, test_count = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shared_walls = bool(rng.integers(2))
    puzzle_wall = int(rng.choice(WALL_COLORS))
    requirements = ["metric", "recurrence"]
    requirements += ["free"] * (train_count - 2)
    rng.shuffle(requirements)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, plan in (("train", requirements), ("test", ["combined"] * test_count)):
        for requirement in plan:
            for _ in range(16):
                wall = puzzle_wall if shared_walls else int(rng.choice(WALL_COLORS))
                pair = _make_pair(rng, requirement, wall)
                signature = _signature(pair["input"])
                if signature in used:
                    continue
                puzzle[split].append(pair)
                used.add(signature)
                break
            else:
                raise RuntimeError("Could not construct distinct input grids")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    logical, rows, cols = _collapse(inp)
    wall, path = _roles_and_path(logical)
    expected = logical.copy()
    for p in path[1:-1]:
        assert logical[p] == BACKGROUND
        expected[p] = ROUTE
    assert np.array_equal(out, expected), "Output must draw exactly the logical route"
    assert np.array_equal(inp, np.repeat(np.repeat(logical, rows, axis=0), cols, axis=1))
    assert max(rows) > 1 and max(cols) > 1, "Both axes must visibly compress"
    assert _seam(logical)
    assert len(path) >= 7 and _turns(path) >= 2
    assert len(path) - 1 > sum(abs(x - y) for x, y in zip(path[0], path[-1]))
    assert np.count_nonzero(logical == BACKGROUND) > len(path)
    assert np.all(out[logical != BACKGROUND] == logical[logical != BACKGROUND])
    assert np.count_nonzero(out == ROUTE) == len(path) - 2
    assert np.any((logical == BACKGROUND) & (out == BACKGROUND)), "Flood fill must differ"
    assert not np.array_equal(logical, out), "Omitting B must change the answer"
    assert inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1]
    short, restricted = _metric_distances(logical, path, rows, cols, wall)
    return {"metric": short < restricted, "pixel_shortest": short,
            "pixel_along_logical_route": restricted, "shape": logical.shape,
            "input_shape": inp.shape, "rows": tuple(map(int, rows)),
            "cols": tuple(map(int, cols)), "wall": wall, "turns": _turns(path),
            "path_cells": len(path), "recurrence": _recurrence(logical),
            "single_witness": _single_witness(logical),
            "dead_end": _offroute_dead_end(logical, path, wall)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            key = _signature(inp)
            assert key not in seen, "Inputs must be distinct within a puzzle"
            seen.add(key)
            analysis = _analyze(pair)
            analyses[split].append(analysis)
            assert np.array_equal(solve(inp), pair["output"])
    assert any(x["metric"] for x in analyses["train"]), "Need a causal metric witness"
    assert any(x["recurrence"] and x["single_witness"] and x["turns"] >= 4 and x["dead_end"]
               for x in analyses["train"])
    assert all(x["metric"] and x["turns"] >= 4 and x["dead_end"]
               for x in analyses["test"]), "Tests combine metric choice, bends, and a blind branch"


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded checks on 42 seeds; returns small metrics and writes no files."""
    # Small hand-computed checks of the rule outside the sampled maze builders.
    hand_input = np.array([[int(c) for c in row] for row in
                           ("20000", "11110", "00000", "01111", "00002")], dtype=np.int8)
    hand_output = np.array([[int(c) for c in row] for row in
                            ("244", "114", "444", "411", "442")], dtype=np.int8)
    stretched = np.repeat(np.repeat(hand_input, [2, 1, 4, 2, 3], axis=0),
                          [1, 3, 2, 4, 1], axis=1)
    before = stretched.copy()
    assert np.array_equal(solve(stretched), hand_output)
    assert np.array_equal(stretched, before), "solve must not mutate its input"
    repeated = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=np.int8)
    stretched = np.repeat(np.repeat(repeated, [2, 3, 1], axis=0), [3, 1, 2], axis=1)
    assert np.array_equal(_collapse(stretched)[0], repeated)
    row_only = stretched[np.r_[True, np.any(stretched[1:] != stretched[:-1], axis=1)]]
    col_only = stretched[:, np.r_[True, np.any(stretched[:, 1:] != stretched[:, :-1], axis=0)]]
    assert np.array_equal(_collapse(row_only)[0], repeated)
    assert np.array_equal(_collapse(col_only)[0], repeated)
    tied = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.int8)
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("Tied shortest routes must not be arbitrarily resolved")
    signatures, logical_shapes, input_shapes, structures, scales, walls = (set() for _ in range(6))
    totals = {"pairs": 0, "metric_pairs": 0, "recurrence_pairs": 0,
              "single_witness_pairs": 0, "dead_end_pairs": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = _analyze(pair)
                logical, _, _ = _collapse(pair["input"])
                # Normalize only the variable wall color; geometry remains.
                logical[logical == x["wall"]] = 1
                structures.add(_signature(logical))
                logical_shapes.add(x["shape"])
                input_shapes.add(x["input_shape"])
                scales.add((x["rows"], x["cols"]))
                walls.add(x["wall"])
                totals["pairs"] += 1
                for name in ("metric", "recurrence", "single_witness", "dead_end"):
                    totals[name + "_pairs"] += int(x[name])
    assert len(signatures) == 42
    assert len(logical_shapes) >= 12 and len(input_shapes) >= 40
    assert len(structures) >= 150 and len(scales) >= 150
    assert len(walls) == len(WALL_COLORS)
    assert totals["metric_pairs"] >= 84 and totals["recurrence_pairs"] >= 42
    assert totals["single_witness_pairs"] >= 42 and totals["dead_end_pairs"] >= 42
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(42000 + 10 * ntrain + ntest, ntrain, ntest)
            assert (len(p["train"]), len(p["test"])) == (ntrain, ntest)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    first = generate_puzzle(314159, 2, 1)
    assert _puzzle_digest(first) == _puzzle_digest(generate_puzzle(314159, 2, 1))
    assert _puzzle_digest(first) != _puzzle_digest(generate_puzzle(314160, 2, 1))
    return {"seeds": 42, **totals, "logical_shapes": len(logical_shapes),
            "input_shapes": len(input_shapes), "wall_normalized_structures": len(structures),
            "scale_profiles": len(scales), "wall_colors": len(walls),
            "legal_count_combinations": 30}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
