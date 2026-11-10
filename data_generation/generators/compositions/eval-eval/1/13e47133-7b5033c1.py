"""Peel the thread: stable path order inside eight-neighbor contour shelves.

No parent code is imported. Construction uses a distance flood; solve uses
literal erosion. Generation and self_test do not write files or retain runs.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib

import numpy as np


DIR4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _connected(mask):
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if not cells:
        return False
    visited = {next(iter(cells))}
    stack = list(visited)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            q = (r + dr, c + dc)
            if q in cells and q not in visited:
                visited.add(q)
                stack.append(q)
    return visited == cells


def _erode_depth(mask, diagonal=True):
    """Layer zero touches the complement, including diagonal/off-grid cells."""
    h, w = mask.shape
    current = mask.copy()
    depth = np.full(mask.shape, -1, dtype=np.int16)
    level = 0
    directions = DIR8 if diagonal else DIR4
    while current.any():
        padded = np.pad(current, 1)
        interior = current.copy()
        for dr, dc in directions:
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~interior] = level
        current = interior
        level += 1
    return depth


def _flood_depth(mask):
    """Separate construction implementation of the same contour metric."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w
                        and mask[r + dr, c + dc]) for dr, dc in DIR8):
            depth[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and depth[rr, cc] < 0:
                depth[rr, cc] = depth[r, c] + 1
                queue.append((rr, cc))
    return depth


def _trace(cells):
    """Recover the entire induced chain, starting at its higher endpoint."""
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in DIR4
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    ends = sorted(p for p in cells if len(neighbors[p]) == 1)
    if (not 2 <= len(cells) <= 30 or len(ends) != 2
            or ends[0][0] == ends[1][0]
            or any(len(n) not in (1, 2) for n in neighbors.values())):
        raise ValueError("thread must be a simple chain with unequal endpoint rows")
    path, previous, current = [], None, ends[0]
    while current is not None:
        if current in path:
            raise ValueError("thread cycle")
        path.append(current)
        onward = [p for p in neighbors[current] if p != previous]
        previous, current = current, onward[0] if onward else None
    if len(path) != len(cells):
        raise ValueError("disconnected thread")
    return path


def _decode(input_grid):
    a = _grid(input_grid)
    outside = int(a[0, 0])
    perimeter = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    if np.any(perimeter != outside):
        raise ValueError("perimeter must be uniformly exterior")
    board = a != outside
    if not _connected(board):
        raise ValueError("expected one connected board")
    counts = np.bincount(a[board], minlength=10)
    material = int(counts.argmax())
    if int(counts[material]) * 2 <= int(board.sum()):
        raise ValueError("board material must have strict interior majority")
    cells = {tuple(map(int, p)) for p in np.argwhere(board & (a != material))}
    path = _trace(cells)
    return a, outside, material, board, path


def _pack(path, colors, depth, outside):
    levels = [int(depth[p]) for p in path]
    buckets = [[] for _ in range(max(levels) + 1)]
    for d, color in zip(levels, colors):
        buckets[d].append(int(color))
    out = np.full((len(buckets), max(map(len, buckets))), outside, dtype=np.int8)
    for d, row in enumerate(buckets):
        out[d, :len(row)] = row
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, outside, _, board, path = _decode(input_grid)
    depth = _erode_depth(board)
    return _pack(path, [int(a[p]) for p in path], depth, outside)


def _box_depth(board):
    rr, cc = np.nonzero(board)
    r, c = np.indices(board.shape)
    return np.minimum.reduce((r - rr.min(), rr.max() - r, c - cc.min(), cc.max() - c))


def _raster_orders(path):
    # Four common alternatives, all subsequently grouped by the correct depth.
    rows = sorted(path)
    cols = sorted(path, key=lambda p: (p[1], p[0]))
    return (rows, rows[::-1], cols, cols[::-1])


def _facts(a, outside, board, path, depth, out):
    color = {p: int(a[p]) for p in path}
    levels = [int(depth[p]) for p in path]
    groups = [[i for i, d in enumerate(levels) if d == level]
              for level in range(max(levels) + 1)]
    raster_failure = all(not np.array_equal(
        out, _pack(order, [color[p] for p in order], depth, outside))
        for order in _raster_orders(path))
    reverse_failure = not np.array_equal(
        out, _pack(path[::-1], [color[p] for p in path[::-1]], depth, outside))
    box_failure = not np.array_equal(
        out, _pack(path, [color[p] for p in path], _box_depth(board), outside))
    l1_failure = not np.array_equal(
        out, _pack(path, [color[p] for p in path], _erode_depth(board, False), outside))
    revisit = any(any(b > a + 1 for a, b in zip(group, group[1:])) for group in groups)
    recurrence = any(any(row[i] == row[k] != row[j]
                         for i in range(len(row)) for j in range(i + 1, len(row))
                         for k in range(j + 1, len(row)))
                     for row in ([color[path[i]] for i in group] for group in groups))
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return {"levels": levels, "raster_failure": raster_failure,
            "reverse_failure": reverse_failure, "box_failure": box_failure,
            "l1_failure": l1_failure, "revisit": revisit, "recurrence": recurrence,
            "above": min(p[0] for p in path) < path[0][0],
            "turns": sum(a != b for a, b in zip(steps, steps[1:])),
            "reversal": (0, 1) in steps and (0, -1) in steps,
            "empty_row": any(d not in levels for d in range(max(levels) + 1)),
            "depth_count": len(set(levels)), "length": len(path)}


def _hole_count(mask):
    # Count bounded four-connected components of the complement.
    seen, holes = set(), 0
    h, w = mask.shape
    for point in map(tuple, np.argwhere(~mask)):
        if point in seen:
            continue
        stack, touches = [point], False
        seen.add(point)
        while stack:
            r, c = stack.pop()
            touches |= r in (0, h - 1) or c in (0, w - 1)
            for dr, dc in DIR4:
                q = (r + dr, c + dc)
                if 0 <= q[0] < h and 0 <= q[1] < w and not mask[q] and q not in seen:
                    seen.add(q)
                    stack.append(q)
        holes += not touches
    return int(holes)


def _board(rng, shape, holes, notches):
    """Carve independently placed edge recesses and bounded rectangular holes."""
    h, w = shape
    mask = np.ones(shape, dtype=bool)
    for _ in range(notches):
        axis = int(rng.integers(2))
        work = mask if axis == 0 else mask.T
        hh, ww = work.shape
        width = int(rng.integers(2, max(3, ww // 2)))
        start = int(rng.integers(1, ww - width))
        reach = int(rng.integers(2, max(3, hh // 2)))
        trial = work.copy()
        if rng.integers(2):
            trial[:reach, start:start + width] = False
        else:
            trial[-reach:, start:start + width] = False
        candidate = trial if axis == 0 else trial.T
        if _connected(candidate) and int(candidate.sum()) >= h * w // 2:
            mask = candidate.copy()
    for _ in range(holes):
        placed = False
        for _attempt in range(60):
            hh, ww = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            r, c = int(rng.integers(2, h - hh - 1)), int(rng.integers(2, w - ww - 1))
            if not np.all(mask[r - 1:r + hh + 1, c - 1:c + ww + 1]):
                continue
            trial = mask.copy()
            trial[r:r + hh, c:c + ww] = False
            if _connected(trial):
                mask, placed = trial, True
                break
        if not placed:
            return None
    return mask if _hole_count(np.pad(mask, 1)) == holes else None


def _grow(rng, mask, depth, length, witness, min_depth=0):
    allowed = mask & (depth >= min_depth)
    starts = np.argwhere(allowed)
    if witness:
        # At least one visited pixel must feel a diagonal recess/hole boundary.
        special = np.argwhere(allowed & (depth != _erode_depth(mask, False)))
        if not len(special):
            return None
        starts = special
    start = tuple(map(int, starts[int(rng.integers(len(starts)))]))
    path, occupied, previous = [start], {start}, None
    h, w = mask.shape
    persistence = float(rng.choice((0.5, 1.0, 2.5, 7.0)))
    for _ in range(length - 1):
        candidates, weights = [], []
        for dr, dc in DIR4:
            q = (path[-1][0] + dr, path[-1][1] + dc)
            if not (0 <= q[0] < h and 0 <= q[1] < w and allowed[q]) or q in occupied:
                continue
            if sum((q[0] + rr, q[1] + cc) in occupied for rr, cc in DIR4) != 1:
                continue
            weight = persistence if previous == (dr, dc) else 1.0
            if depth[q] == 0 or int(depth[q]) not in {int(depth[p]) for p in path}:
                weight *= 1.8
            candidates.append((q, (dr, dc)))
            weights.append(weight)
        if not candidates:
            return None
        weights = np.asarray(weights, dtype=float)
        q, previous = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
        path.append(q)
        occupied.add(q)
    if path[0][0] == path[-1][0]:
        return None
    return path if path[0][0] < path[-1][0] else path[::-1]


def _geometry_ok(path, depth, witness, above):
    if above and min(p[0] for p in path) >= path[0][0]:
        return False
    if not witness:
        return True
    levels = [int(depth[p]) for p in path]
    if len(set(levels)) < 3 or max(Counter(levels).values()) < 4:
        return False
    # A color assignment cannot distinguish two identical geometric orders.
    groups = lambda order: [[p for p in order if int(depth[p]) == d]
                            for d in range(max(levels) + 1)]
    correct = groups(path)
    if any(groups(order) == correct for order in _raster_orders(path)):
        return False
    return any(any(b > a + 1 for a, b in zip(indices, indices[1:]))
               for indices in ([i for i, v in enumerate(levels) if v == d] for d in set(levels)))


def _color(rng, path, depth, palette, witness):
    size = int(rng.integers(3 if witness else 1, len(palette) + 1))
    pool = palette[:size]
    colors = []
    stay = float(rng.choice((0.05, 0.3, 0.65)))
    for _ in path:
        colors.append(colors[-1] if colors and rng.random() < stay else int(rng.choice(pool)))
    if witness:
        levels = [int(depth[p]) for p in path]
        group = max(([i for i, d in enumerate(levels) if d == level]
                     for level in sorted(set(levels))), key=len)
        for i, color in zip(group[:4], (pool[0], pool[1], pool[0], pool[2])):
            colors[i] = int(color)
    return colors


def _fallback(shape, holes, length, witness, empty=False):
    """Constructive, family-valid safety net after bounded random exploration."""
    h, w = shape
    mask = np.ones(shape, dtype=bool)
    if witness:
        # The diagonal concavity affects the horizontal bridge at row seven.
        mask[:7, :7] = False
        if empty:
            path = [(10, 5), (9, 5)] + [(8, c) for c in range(5, 12)]
            path += [(r, 11) for r in range(9, 15)]
            path += [(14, c) for c in range(10, 7, -1)] + [(15, 8), (16, 8)]
            length = max(20, min(length, 30))
            path += [(16, c) for c in range(9, 9 + length - len(path))]
        else:
            path = [(9, 5), (8, 5)] + [(7, c) for c in range(5, 11)]
            path += [(r, 10) for r in range(8, 14)]
            path += [(13, c) for c in range(9, 6, -1)] + [(14, 7), (15, 7)]
            # Base length 19; extend an induced tail with no new side contacts.
            length = max(19, min(length, 30))
            path += [(15, c) for c in range(8, 8 + length - len(path))]
        sites = [(3, w - 5), (h - 4, 2)]
    else:
        length = min(length, h)
        path = [(r, w // 2) for r in range(length)]
        sites = [(2, 2), (h - 4, w - 4)]
    for r, c in sites[:holes]:
        mask[r:r + 2, c:c + 2] = False
    assert all(0 <= r < h and 0 <= c < w and mask[r, c] for r, c in path)
    assert _connected(mask) and _hole_count(np.pad(mask, 1)) == holes
    return mask, path


def _make_pair(rng, full_shape, outside, material, holes, length, witness, above):
    palette = [int(x) for x in rng.permutation([c for c in range(10) if c not in (outside, material)])]
    for board_attempt in range(13):
        fallback = board_attempt == 12
        if fallback:
            top = left = 1
            shape = (full_shape[0] - 2, full_shape[1] - 2)
            mask, fixed_path = _fallback(shape, holes, length, witness, above)
        else:
            top, left = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            bottom, right = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            shape = (full_shape[0] - top - bottom, full_shape[1] - left - right)
            notches = int(rng.integers(1 if witness else 0, 4))
            mask = _board(rng, shape, holes, notches)
            if mask is None:
                continue
            fixed_path = None
        depth = _flood_depth(mask)
        for _walk_attempt in range(1 if fallback else 64):
            path = fixed_path if fallback else _grow(rng, mask, depth, length, witness, int(above))
            if path is None or not _geometry_ok(path, depth, witness, above):
                if fallback:
                    raise AssertionError("fallback geometry lost a required witness")
                continue
            if above and min(int(depth[p]) for p in path) < 1:
                continue
            if int(mask.sum()) <= 2 * len(path):
                continue
            for _color_attempt in range(24):
                colors = _color(rng, path, depth, palette, witness)
                local = np.full(shape, outside, dtype=np.int8)
                local[mask] = material
                for p, color in zip(path, colors):
                    local[p] = color
                out = _pack(path, colors, depth, outside)
                facts = _facts(local, outside, mask, path, depth, out)
                if witness and not all(facts[k] for k in (
                        "raster_failure", "reverse_failure", "box_failure", "l1_failure", "revisit", "recurrence")):
                    continue
                inp = np.full(full_shape, outside, dtype=np.int8)
                inp[top:top + shape[0], left:left + shape[1]] = local
                # Label construction did not call solve, nor recover its path.
                assert np.array_equal(out, solve(inp))
                return {"input": inp, "output": out}
    raise AssertionError("bounded construction exhausted")


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # scale, topology, or thread-length extrapolation
    roles = [int(c) for c in rng.permutation(10)]
    offset = int(rng.integers(1, 10))
    puzzle, used_shapes = {"train": [], "test": []}, set()
    for index in range(num_train + num_test):
        target = index >= num_train
        local_index = index - num_train if target else index
        witness = local_index == 0 or (not target and local_index == 1)
        above = not target and local_index == 1
        if mode == 0:
            lo, hi = (28, 30) if target else ((22, 24) if witness else (16, 24))
        elif mode == 1:
            lo, hi = (22, 30) if witness or target else (14, 30)
        else:
            lo, hi = (22, 30) if witness or target else (14, 28)
        candidates = [(h, w) for h in range(lo, hi + 1) for w in range(lo, hi + 1)
                      if (h, w) not in used_shapes]
        shape = candidates[int(rng.integers(len(candidates)))]
        used_shapes.add(shape)  # Unique dimensions guarantee distinct inputs.
        holes = (2 if mode == 1 and target else
                 0 if not target and local_index == 0 else
                 1 if not target and local_index == 1 else int(rng.integers(2)))
        if mode == 2:
            length = int(rng.integers(24, 31)) if target else int(rng.integers(19 if witness else 5, 23))
        else:
            length = int(rng.integers(19, 31)) if witness else int(rng.integers(4, 29))
        outside, material = roles[index % 10], roles[(index + offset) % 10]
        pair = _make_pair(rng, shape, outside, material, holes, length, witness, above)
        puzzle["test" if target else "train"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def _shape_signature(mask):
    """Binary geometry, normalized for translation and all eight symmetries."""
    rr, cc = np.nonzero(mask)
    cropped = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    return min(bytes(a.shape) + a.tobytes() for k in range(4)
               for a in (np.rot90(cropped, k), np.fliplr(np.rot90(cropped, k))))


def _pair_info(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, out = _grid(pair["input"]), _grid(pair["output"])
    a, outside, material, board, path = _decode(a)
    depth = _erode_depth(board)
    colors = [int(a[p]) for p in path]
    assert np.array_equal(_pack(path, colors, depth, outside), out), "wrong contour shelves"
    assert int(np.count_nonzero(out != outside)) == len(path), "thread cell lost or duplicated"
    assert Counter(map(int, out[out != outside])) == Counter(colors), "thread colors changed"
    for row in out:
        occupied = np.flatnonzero(row != outside)
        assert np.array_equal(occupied, np.arange(len(occupied))), "shelf is not left aligned"
    facts = _facts(a, outside, board, path, depth, out)
    facts.update({"outside": outside, "material": material, "board": board,
                  "path": path, "depth": depth, "holes": _hole_count(board),
                  "shape": a.shape, "palette_size": len(set(colors))})
    return facts


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _pair_info(pair)
            key = _key(np.asarray(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(info)
    # Teaching checks are separate from pair semantics; solve also supports
    # plain boards, single-color threads and paths confined to one contour.
    for split in ("train", "test"):
        assert any(all(i[k] for k in ("box_failure", "l1_failure", "raster_failure",
                                     "reverse_failure", "revisit", "recurrence"))
                   and i["depth_count"] >= 3 for i in infos[split]), "missing joint witness"
    assert any(i["above"] and i["reverse_failure"] and i["empty_row"]
               for i in infos["train"]), "endpoint/empty-shelf convention untaught"
    assert len({i["shape"] for s in infos.values() for i in s}) == len(seen)
    return None


def self_test():
    """42 seeds, semantic mutations, metric cross-checks, and legal boundaries.

    At most two puzzles are retained at once. No payloads, logs or traces are
    stored. Aggregate geometry signatures are hashes, not duplicate grids.
    """
    # Small hand-computed fixture, independent of randomized construction.
    fixture = np.zeros((9, 9), dtype=np.int8)
    fixture[1:8, 1:8] = 1
    fixture_path = [(2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (4, 5), (4, 6),
                    (5, 6), (6, 6), (6, 5), (6, 4), (6, 3), (6, 2), (5, 2), (4, 2)]
    for p, c in zip(fixture_path, "234567823456789"):
        fixture[p] = int(c)
    expected = np.array([[int(c) for c in row] for row in
                         ("000000000000", "234823456789", "570000000000", "600000000000")])
    assert np.array_equal(solve(fixture), expected)
    recolor = np.asarray((9, 0, 8, 7, 6, 5, 4, 3, 2, 1), dtype=np.int8)
    assert np.array_equal(solve(recolor[fixture]), recolor[expected])
    perforated = fixture.copy()
    perforated[3, 3] = 0
    assert not np.array_equal(solve(perforated), expected)
    assert _erode_depth(perforated != 0)[4, 4] == 0
    assert _erode_depth(perforated != 0, False)[4, 4] == 1
    swapped = fixture.copy()
    swapped[2, 2], swapped[6, 2] = fixture[6, 2], fixture[2, 2]
    changed = expected.copy()
    changed[1, 0], changed[1, 9] = expected[1, 9], expected[1, 0]
    assert np.array_equal(solve(swapped), changed)
    for shape in ((20, 20), (20, 28), (28, 20), (28, 28)):
        for count in range(3):
            for length in (19, 22, 24, 30):
                for empty_case in (False, True):
                    mask, path = _fallback(shape, count, length, True, empty_case)
                    depth = _flood_depth(mask)
                    assert _geometry_ok(path, depth, True, True)
                    assert not empty_case or min(int(depth[p]) for p in path) >= 1
    shapes, boards, paths, lengths, turns, roles, holes, depths, palettes = (set() for _ in range(9))
    canonical_boards, canonical_paths = set(), set()
    complete, above, empty, reversals, recurrences = set(), 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _pair_info(pair)
                for grid in pair.values():
                    digest.update(bytes(grid.shape) + grid.tobytes())
                shapes.add(info["shape"])
                boards.add(hashlib.sha256(info["board"].tobytes() + bytes(info["shape"])).digest())
                canonical_boards.add(hashlib.sha256(_shape_signature(info["board"])).digest())
                path_mask = np.zeros(info["shape"], dtype=bool)
                for p in info["path"]:
                    path_mask[p] = True
                canonical_paths.add(hashlib.sha256(_shape_signature(path_mask)).digest())
                origin = info["path"][0]
                paths.add(tuple((r - origin[0], c - origin[1]) for r, c in info["path"]))
                lengths.add(info["length"])
                turns.add(info["turns"])
                roles.add((info["outside"], info["material"]))
                holes.add(info["holes"])
                depths.update(info["levels"])
                palettes.add(info["palette_size"])
                above += info["above"]
                empty += info["empty_row"]
                reversals += info["reversal"]
                recurrences += info["recurrence"]
                assert np.array_equal(info["depth"], _flood_depth(info["board"]))
                assert np.array_equal(pair["output"], solve(pair["input"].tolist()))
        complete.add(digest.digest())
    assert len(complete) == 42
    assert len(shapes) >= 60 and len(boards) >= 180 and len(paths) >= 150
    assert len(canonical_boards) >= 140 and len(canonical_paths) >= 140
    assert len(lengths) >= 20 and len(turns) >= 12 and len(roles) >= 65
    assert {0, 1, 2} <= holes and len(depths) >= 6 and len(palettes) >= 7
    assert above >= 42 and empty >= 5 and reversals >= 30 and recurrences >= 100
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(901 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    first, second = generate_puzzle(9173, 2, 1), generate_puzzle(9173, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first
               for a, b in zip(first[s], second[s]) for k in a)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    original = generate_puzzle(0, 2, 1)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted shelf accepted")
    # Input graph and role failures, independent of construction metadata.
    for cells in ({(2, 2), (2, 3), (2, 4)},
                  {(2, 2), (1, 2), (2, 1), (2, 3)},
                  {(2, 2), (2, 3), (3, 2), (3, 3)},
                  {(1, 1), (2, 1), (4, 4)}):
        try:
            _trace(cells)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed thread accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "boards": len(boards),
            "boards_up_to_symmetry": len(canonical_boards), "paths_up_to_symmetry": len(canonical_paths),
            "paths": len(paths), "lengths": sorted(lengths), "turns": sorted(turns),
            "role_pairs": len(roles), "hole_counts": sorted(holes), "depths": sorted(depths),
            "palette_sizes": sorted(palettes), "above_endpoint_pairs": above,
            "empty_shelf_pairs": empty, "horizontal_reversal_pairs": reversals,
            "recurrence_pairs": recurrences}
