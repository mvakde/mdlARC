"""Paired-stencil watersheds: shared voids, surviving column ranks, geodesic fill.

Only the input is needed by solve.  The header visibly supplies the obstacle
swatch and ordered output palette; it does not encode any answer geometry.
"""
from __future__ import annotations

from collections import deque
import heapq

import numpy as np


_NEIGHBORS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "bad grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _parse(input_grid):
    a = _grid(input_grid)
    _require(a.shape[0] >= 5 and a.shape[1] >= 8, "missing header or maps")
    wall = int(a[0, 0])
    _require(wall != 0 and a[0, 1] == 0, "bad obstacle swatch")
    header = a[0, 2:]
    end = next((i for i, x in enumerate(header) if x == 0), len(header))
    labels = tuple(int(x) for x in header[:end])
    _require(3 <= len(labels) <= 5 and len(set(labels)) == len(labels), "bad rank key")
    _require(wall not in labels and np.all(header[end:] == 0), "bad key padding")
    separator = int(a[1, 0])
    _require(separator not in {0, wall, *labels} and np.all(a[1] == separator),
             "bad header separator")
    body = a[2:]
    candidates = []
    h, w = body.shape
    if w % 2 and np.all(body[:, w // 2] == separator):
        first, second = body[:, :w // 2], body[:, w // 2 + 1:]
        if not np.any(first == separator) and not np.any(second == separator):
            candidates.append(("side", first, second))
    if h % 2 and np.all(body[h // 2] == separator):
        first, second = body[:h // 2], body[h // 2 + 1:]
        if not np.any(first == separator) and not np.any(second == separator):
            candidates.append(("stack", first, second))
    _require(len(candidates) == 1, "maps need one unambiguous divider")
    orientation, first, second = candidates[0]
    values = set(int(x) for x in np.unique(np.concatenate((first.ravel(), second.ravel()))))
    markers = values - {0, wall}
    _require(len(markers) <= 1 and not markers.intersection({separator, *labels}),
             "maps need one distinct well color")
    marker = next(iter(markers), None)
    opened = (first != wall) & (second != wall)
    wells = np.zeros(first.shape, dtype=bool) if marker is None else (
        ((first == marker) | (second == marker)) & opened)
    columns = tuple(int(c) for c in np.flatnonzero(wells.any(axis=0)))
    _require(len(columns) <= len(labels), "more surviving well columns than key colors")
    return {"wall": wall, "marker": marker, "separator": separator,
            "labels": labels, "orientation": orientation, "first": first,
            "second": second, "open": opened, "wells": wells, "columns": columns}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill the common void from column-ranked surviving wells, input only."""
    p = _parse(input_grid)
    opened, wells = p["open"], p["wells"]
    h, w = opened.shape
    best_distance = np.full((h, w), h * w + 1, dtype=int)
    best_rank = np.full((h, w), len(p["labels"]), dtype=int)
    pending = []
    for rank, column in enumerate(p["columns"]):
        for row in np.flatnonzero(wells[:, column]):
            row = int(row)
            best_distance[row, column] = 0
            best_rank[row, column] = rank
            heapq.heappush(pending, (0, rank, row, column))
    while pending:
        distance, rank, row, column = heapq.heappop(pending)
        if (distance, rank) != (best_distance[row, column], best_rank[row, column]):
            continue
        for dr, dc in _NEIGHBORS:
            nr, nc = row + dr, column + dc
            if 0 <= nr < h and 0 <= nc < w and opened[nr, nc]:
                proposed = (distance + 1, rank)
                if proposed < (best_distance[nr, nc], best_rank[nr, nc]):
                    best_distance[nr, nc], best_rank[nr, nc] = proposed
                    heapq.heappush(pending, (*proposed, nr, nc))
    out = np.where(opened, 0, p["wall"]).astype(np.int8)
    for rank, color in enumerate(p["labels"]):
        out[best_rank == rank] = color
    return out


def _distances(opened, wells, columns):
    """Separate unweighted searches, used for validation and measured features."""
    h, w = opened.shape
    infinity = h * w + 1
    distance = np.full((len(columns), h, w), infinity, dtype=int)
    for rank, column in enumerate(columns):
        queue = deque()
        for row in np.flatnonzero(wells[:, column]):
            row = int(row)
            distance[rank, row, column] = 0
            queue.append((row, column))
        while queue:
            row, col = queue.popleft()
            for dr, dc in _NEIGHBORS:
                nr, nc = row + dr, col + dc
                if (0 <= nr < h and 0 <= nc < w and opened[nr, nc]
                        and distance[rank, nr, nc] == infinity):
                    distance[rank, nr, nc] = distance[rank, row, col] + 1
                    queue.append((nr, nc))
    return distance


def _features(p):
    opened, wells, columns = p["open"], p["wells"], p["columns"]
    h, w = opened.shape
    infinity = h * w + 1
    distances = _distances(opened, wells, columns)
    counts = tuple(int(wells[:, c].sum()) for c in columns)
    dead = np.zeros(opened.shape, dtype=bool) if p["marker"] is None else (
        ((p["first"] == p["marker"]) | (p["second"] == p["marker"])) & ~opened)
    dead_columns = set(int(c) for c in np.flatnonzero(dead.any(axis=0))) - set(columns)
    codes = (p["first"] == p["wall"]).astype(int) * 2 + (p["second"] == p["wall"])
    result = {"counts": counts, "cases": set(int(x) for x in np.unique(codes)),
              "dead_shift": bool(columns and any(c < columns[-1] for c in dead_columns)),
              "repeat": bool(counts and max(counts) >= 2), "tie": False,
              "detour": False, "competition": False, "unreachable": bool(opened.any()),
              "growth": False, "max_distance": 0, "distances": distances}
    if not columns:
        return result
    nearest = distances.min(axis=0)
    reachable = opened & (nearest < infinity)
    rank = distances.argmin(axis=0)
    result["unreachable"] = bool(np.any(opened & ~reachable))
    result["growth"] = bool(np.any(reachable & ~wells))
    result["max_distance"] = int(nearest[reachable].max())
    result["tie"] = bool(np.any(reachable & ((distances == nearest).sum(axis=0) >= 2)))
    result["competition"] = bool(np.any((distances < infinity).sum(axis=0) >= 2))
    rr, cc = np.indices(opened.shape)
    manhattan = np.full(distances.shape, infinity, dtype=int)
    for i, col in enumerate(columns):
        for row in np.flatnonzero(wells[:, col]):
            manhattan[i] = np.minimum(manhattan[i], abs(rr - int(row)) + abs(cc - col))
    # Only a different winning COLOR counts, not merely a longer path.
    result["detour"] = bool(np.any(reachable & (rank != manhattan.argmin(axis=0))))
    return result


def _mask(rng, h, w, style):
    if style == "maze":
        opened = np.zeros((h, w), dtype=bool)
        start = (int(rng.integers((h + 1) // 2)) * 2,
                 int(rng.integers((w + 1) // 2)) * 2)
        opened[start] = True
        stack = [start]
        while stack:
            r, c = stack[-1]
            choices = [(r + 2 * dr, c + 2 * dc, dr, dc) for dr, dc in _NEIGHBORS
                       if 0 <= r + 2 * dr < h and 0 <= c + 2 * dc < w
                       and not opened[r + 2 * dr, c + 2 * dc]]
            if not choices:
                stack.pop()
                continue
            nr, nc, dr, dc = choices[int(rng.integers(len(choices)))]
            opened[r + dr, c + dc] = opened[nr, nc] = True
            stack.append((nr, nc))
        # Break the regular lattice with rooms, loops, and occasional closures.
        for _ in range(int(rng.integers(1, 4))):
            r, c = int(rng.integers(h - 1)), int(rng.integers(w - 1))
            opened[r:r + 2, c:c + 2] = True
        opened |= rng.random((h, w)) < rng.uniform(.015, .10)
    elif style == "rooms":
        opened = np.ones((h, w), dtype=bool)
        for c in range(int(rng.integers(2, 4)), w - 1, int(rng.integers(3, 6))):
            opened[:, c] = False
            opened[int(rng.integers(h)), c] = True
        for r in range(int(rng.integers(2, 4)), h - 1, int(rng.integers(3, 6))):
            opened[r, :] = False
            for _ in range(int(rng.integers(1, 3))):
                opened[r, int(rng.integers(w))] = True
        opened &= rng.random((h, w)) > .06
    elif style == "walk":
        opened = np.zeros((h, w), dtype=bool)
        for _ in range(int(rng.integers(2, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(h * w):
                opened[r, c] = True
                dr, dc = _NEIGHBORS[int(rng.integers(4))]
                r, c = min(h - 1, max(0, r + dr)), min(w - 1, max(0, c + dc))
    else:
        opened = rng.random((h, w)) < rng.uniform(.53, .81)
        if rng.random() < .5:
            r = int(rng.integers(h))
            opened[r] = True
    # Some enclosed, unseeded negative space is intentional, but not universal.
    if rng.random() < .70:
        r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        opened[r - 1:r + 2, c - 1:c + 2] = False
        opened[r, c] = True
    if rng.random() < .5:
        opened = opened[::-1].copy()
    if rng.random() < .5:
        opened = opened[:, ::-1].copy()
    return opened


def _assemble(codes, wells, dead, orientation, convention, rng):
    wall, marker, separator, labels = convention
    first = np.where((codes & 2) != 0, wall, 0).astype(np.int8)
    second = np.where((codes & 1) != 0, wall, 0).astype(np.int8)
    for r, c in zip(*np.nonzero(wells)):
        view = int(rng.integers(3))
        if view != 1:
            first[r, c] = marker
        if view != 0:
            second[r, c] = marker
    for r, c in dead:
        if rng.random() < .5:
            first[r, c], second[r, c] = marker, wall
        else:
            first[r, c], second[r, c] = wall, marker
    h, w = codes.shape
    if orientation == "side":
        body = np.concatenate((first, np.full((h, 1), separator), second), axis=1)
    else:
        body = np.concatenate((first, np.full((1, w), separator), second), axis=0)
    inp = np.zeros((body.shape[0] + 2, body.shape[1]), dtype=np.int8)
    inp[0, 0] = wall
    inp[0, 2:2 + len(labels)] = labels
    inp[1, :] = separator
    inp[2:] = body
    return inp


def _sample_input(rng, shape, orientation, convention, active_count, profile, style):
    h, w = shape
    opened = _mask(rng, h, w, style)
    available = np.flatnonzero(opened.any(axis=0))
    if len(available) < active_count:
        return None
    if active_count and rng.random() < .3:
        possible = [c for c in range(w - active_count + 1)
                    if np.all(opened[:, c:c + active_count].any(axis=0))]
        if possible:
            start = possible[int(rng.integers(len(possible)))]
            columns = tuple(range(start, start + active_count))
        else:
            columns = tuple(sorted(int(c) for c in rng.choice(available, active_count, replace=False)))
    else:
        columns = tuple(sorted(int(c) for c in rng.choice(available, active_count, replace=False)))
    wells = np.zeros(shape, dtype=bool)
    equal = int(rng.integers(1, 3))
    for rank, c in enumerate(columns):
        rows = np.flatnonzero(opened[:, c])
        if profile == "equal":
            count = min(equal, len(rows))
        elif profile == "single":
            count = 1
        else:
            count = int(rng.integers(1, min(4, len(rows)) + 1))
        if profile == "mixed" and rank == 0:
            count = min(len(rows), 2 + int(rng.integers(2)))
        chosen = rng.choice(rows, count, replace=False)
        wells[chosen, c] = True
    # Code 0 is the shared void; codes 1/2/3 split the walls between views.
    weights = rng.dirichlet([1.4, 1.4, 1.4])
    codes = rng.choice(np.array([1, 2, 3], dtype=np.int8), size=shape, p=weights)
    codes[opened] = 0
    dead_candidates = [(r, c) for r in range(h) for c in range(w)
                       if not opened[r, c] and c not in columns
                       and (not columns or c < columns[-1])]
    if not dead_candidates:
        return None
    dead = [dead_candidates[int(rng.integers(len(dead_candidates)))]]
    extra = [(r, c) for r in range(h) for c in range(w) if not opened[r, c] and (r, c) != dead[0]]
    for i in np.atleast_1d(rng.choice(len(extra), size=min(len(extra), int(rng.integers(0, 5))), replace=False)):
        dead.append(extra[int(i)])
    return _assemble(codes, wells, dead, orientation, convention, rng)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nv = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    k = int(rng.integers(3, 6))
    colors = [int(x) for x in rng.permutation(np.arange(1, 10))]
    convention = (colors[0], colors[1], colors[2], tuple(colors[3:3 + k]))
    layout = str(rng.choice(["side", "stack", "mixed"]))
    orientations = (["side", "stack"] if layout == "mixed" else [layout, layout])
    if layout == "mixed" and rng.random() < .5:
        orientations.reverse()
    orientations += [str(rng.choice(["side", "stack"])) if layout == "mixed" else layout
                     for _ in range(nt + nv - 2)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(nt + nv):
        training = index < nt
        if layout == "mixed":
            shape = (int(rng.integers(7, 11)), int(rng.integers(9, 13))) if training else (
                int(rng.integers(11, 14)), int(rng.integers(13, 15)))
        elif layout == "side":
            shape = (int(rng.integers(7, 12)), int(rng.integers(8, 13))) if training else (
                int(rng.integers(20, 29) if rng.random() < .25 else rng.integers(12, 19)),
                int(rng.integers(10, 15)))
        else:
            shape = (int(rng.integers(7, 11)), int(rng.integers(9, 17))) if training else (
                int(rng.integers(11, 14)),
                int(rng.integers(24, 31) if rng.random() < .25 else rng.integers(14, 25)))
        anchor = index if index < 2 else -1
        active_count = k if anchor == 0 else k - 1 if anchor == 1 else int(rng.choice(
            np.arange(k + 1), p=np.array([.03, .06] + [1.] * (k - 1)) / (k - 1 + .09)))
        profile = "mixed" if anchor == 0 else "single" if anchor == 1 else str(rng.choice(["mixed", "equal", "single"]))
        for _ in range(600):
            style = (str(rng.choice(["maze", "rooms"])) if anchor == 0 else
                     str(rng.choice(["maze", "rooms", "random", "walk"])) if training else
                     str(rng.choice(["maze", "walk", "rooms"], p=[.6, .2, .2])))
            inp = _sample_input(rng, shape, orientations[index], convention, active_count, profile, style)
            if inp is None or _key(inp) in seen:
                continue
            p = _parse(inp)
            f = _features(p)
            if anchor == 0 and not (f["repeat"] and f["tie"] and f["competition"]
                                    and len(set(f["counts"])) > 1 and f["growth"]
                                    and f["cases"] == {0, 1, 2, 3}):
                continue
            if anchor == 1 and not (f["detour"] and f["unreachable"] and f["growth"]):
                continue
            pair = {"input": inp, "output": solve(inp)}
            puzzle["train" if training else "test"].append(pair)
            seen.add(_key(inp))
            break
        else:
            raise RuntimeError("could not construct an informative distinct example")
    # Evidence examples do not have a fixed presentation position.
    permutation = rng.permutation(nt)
    puzzle["train"] = [puzzle["train"][int(i)] for i in permutation]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate the rule, visible conventions, evidence, and size shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "bad splits")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, convention, features = set(), None, []
    training_heights = []
    training_orientations = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            p = _parse(inp)
            _require(p["marker"] is not None, "generated input must show wells, possibly occluded")
            roles = (p["wall"], p["marker"], p["separator"], p["labels"])
            if convention is None:
                convention = roles
            _require(roles == convention, "puzzle palette changed")
            opened, wells, columns = p["open"], p["wells"], p["columns"]
            _require(out.shape == opened.shape, "output must have one map's shape")
            f = _features(p)
            # Unlike solve's priority flood, reconstruct with one BFS per rank.
            expected = np.where(opened, 0, p["wall"]).astype(np.int8)
            if columns:
                distance = f["distances"]
                reachable = opened & (distance.min(axis=0) <= opened.size)
                rank = distance.argmin(axis=0)
                for i, color in enumerate(p["labels"][:len(columns)]):
                    expected[reachable & (rank == i)] = color
                for i, col in enumerate(columns):
                    _require(np.all(out[wells[:, col], col] == p["labels"][i]), "bad well-column rank")
            _require(np.array_equal(out, expected), "wrong geodesic watershed")
            _require(np.array_equal(out, solve(inp)), "solve disagrees")
            if split == "train":
                features.append((p, f))
                training_heights.append(out.shape[0])
                training_orientations.add(p["orientation"])
            else:
                _require(out.shape[0] > max(training_heights), "test must shift to a taller map")
                _require(p["orientation"] in training_orientations, "test layout not demonstrated")
    k = len(convention[3])
    _require(any(len(p["columns"]) == k for p, f in features), "training hides a key rank")
    _require(any(len(p["columns"]) == k - 1 for p, f in features), "training must show dynamic column count")
    _require(set.union(*(f["cases"] for p, f in features)) == {0, 1, 2, 3}, "incomplete NOR evidence")
    for name in ("repeat", "tie", "competition", "growth", "detour", "unreachable", "dead_shift"):
        _require(any(f[name] for p, f in features), f"training lacks {name} evidence")
    _require(any(len(set(f["counts"])) == 1 and len(f["counts"]) >= 2 for p, f in features),
             "training must distinguish column order from frequency")
    _require(any(len(set(f["counts"])) > 1 for p, f in features), "training must include mixed well counts")


def self_test():
    """Bounded 42-seed semantic, count, determinism, and diversity audit."""
    palettes, shapes, geometries, profiles, orientations, rank_counts = set(), set(), set(), set(), set(), set()
    tall = wide = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _parse(pair["input"])
                f = _features(p)
                palettes.add((p["wall"], p["marker"], p["separator"], p["labels"]))
                shapes.add(p["open"].shape)
                geometries.add((p["open"].shape, p["open"].tobytes(), p["wells"].tobytes()))
                profiles.add(f["counts"])
                orientations.add(p["orientation"])
                rank_counts.add(len(p["columns"]))
                tall += p["open"].shape[0] >= 20
                wide += p["open"].shape[1] >= 20
        if seed in (0, 17, 41):
            again = generate_puzzle(seed)
            _require(all(np.array_equal(x[key], y[key]) for split in puzzle
                         for x, y in zip(puzzle[split], again[split]) for key in ("input", "output")),
                     "nondeterministic puzzle")
    _require(len(palettes) >= 35 and len(shapes) >= 35 and len(geometries) >= 200,
             "insufficient palette, size, or topology variation")
    _require(len(profiles) >= 30 and orientations == {"side", "stack"}, "missing structural variety")
    _require({2, 3, 4, 5} <= rank_counts and tall > 0 and wide > 0, "missing count or size regimes")
    for nt in range(2, 11):
        for nv in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(10000 + nt * 10 + nv, nt, nv)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == nv, "counts ignored")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Deleting occluded wells and changing which view carries a live well must
    # preserve the answer; the blocker masks and surviving sources are unchanged.
    probe = generate_puzzle(9, 2, 1)["train"][0]["input"]
    p = _parse(probe)
    _require(np.array_equal(solve(probe), solve(_rebuild_without_dead(p))),
             "occluded wells or source view affected the result")
    # A hand-constructed map checks actual expected colors, including the
    # left-rank tie at (5,5), a Manhattan counterexample, and an unseeded pocket.
    codes = np.full((7, 9), 3, dtype=np.int8)
    codes[1:6, 1:8] = 0
    codes[1:5, 4] = 3
    codes[0, 0] = 0
    wells = np.zeros((7, 9), dtype=bool)
    wells[3, 1] = wells[5, 1] = wells[1, 5] = True
    expected_rows = ["044444444"] + ["499942224"] * 4 + ["499999994", "444444444"]
    expected = np.array([[int(c) for c in row] for row in expected_rows], dtype=np.int8)
    for orientation in ("side", "stack"):
        inp = _assemble(codes, wells, [(2, 0)], orientation, (4, 7, 8, (9, 2, 6)), np.random.default_rng(5))
        _require(np.array_equal(solve(inp), expected), "hand-checked watershed failed")
        # Moving the wall gate moves the watershed while all sources stay put.
        altered = codes.copy()
        altered[1, 4], altered[5, 4] = 0, 3
        changed = _assemble(altered, wells, [(2, 0)], orientation, (4, 7, 8, (9, 2, 6)), np.random.default_rng(5))
        _require(solve(changed)[1, 3] == 2, "solver ignored corridor geometry")
        # Accepting the covered well creates an earlier column and relabels the
        # old right-hand source. This is a direct A-to-B interaction witness.
        uncovered, new_wells = codes.copy(), wells.copy()
        uncovered[2, 0], new_wells[2, 0] = 0, True
        changed = _assemble(uncovered, new_wells, [], orientation,
                            (4, 7, 8, (9, 2, 6)), np.random.default_rng(5))
        _require(solve(changed)[1, 5] == 6 and solve(inp)[1, 5] == 2,
                 "well survival did not change later column ranks")


def _rebuild_without_dead(p):
    codes = (p["first"] == p["wall"]).astype(np.int8) * 2 + (p["second"] == p["wall"])
    return _assemble(codes, p["wells"], [], p["orientation"],
                     (p["wall"], p["marker"], p["separator"], p["labels"]), np.random.default_rng(81))
