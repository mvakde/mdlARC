"""Seed every blank 3x3, exhaust full-side expansions, union terminal squares.

Only NumPy and the standard library are used. All geometry is read from the
original input. No palette or rule is learned from another example by solve().
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


BLANK = 0
HIGHLIGHT = 1


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _integral(mask):
    return np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _area_sum(prefix, top, left, bottom, right):
    # Rectangles use half-open coordinates throughout this module.
    return int(prefix[bottom, right] - prefix[top, right]
               - prefix[bottom, left] + prefix[top, left])


def _seed_starts(mask):
    h, w = mask.shape
    if min(h, w) < 3:
        return []
    p = _integral(mask)
    sums = p[3:, 3:] - p[:-3, 3:] - p[3:, :-3] + p[:-3, :-3]
    return [tuple(map(int, point)) for point in np.argwhere(sums == 9)]


def _terminals_from_seeds(mask, starts):
    """Explore every expansion order, sharing states between all seed paths.

    Each move adds one complete blank row or column. This graph is acyclic
    because area strictly increases. Rectangles reached from different seeds
    are deduplicated; no execution history or grid copies are retained.
    """
    h, w = mask.shape
    p = _integral(mask)
    pending = [(r, c, r + 3, c + 3) for r, c in starts]
    seen = set(pending)
    terminal = []
    while pending:
        t, l, b, r = pending.pop()
        extensions = []
        if t > 0 and _area_sum(p, t - 1, l, t, r) == r - l:
            extensions.append((t - 1, l, b, r))
        if b < h and _area_sum(p, b, l, b + 1, r) == r - l:
            extensions.append((t, l, b + 1, r))
        if l > 0 and _area_sum(p, t, l - 1, b, l) == b - t:
            extensions.append((t, l - 1, b, r))
        if r < w and _area_sum(p, t, r, b, r + 1) == b - t:
            extensions.append((t, l, b, r + 1))
        if not extensions:
            terminal.append((t, l, b, r))
        for rect in extensions:
            if rect not in seen:
                seen.add(rect)
                pending.append(rect)
    return sorted(terminal)


def _accepted(terminals):
    return [rect for rect in terminals
            if rect[2] - rect[0] == rect[3] - rect[1] > 2]


def _footprint(shape, rectangles):
    result = np.zeros(shape, dtype=bool)
    for t, l, b, r in rectangles:
        result[t:b, l:r] = True
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    mask = a == BLANK
    starts = _seed_starts(mask)
    terminal = _terminals_from_seeds(mask, starts)
    out = a.copy()
    out[_footprint(a.shape, _accepted(terminal))] = HIGHLIGHT
    return out


def _maximal_rectangles(mask, minimum=1):
    """Row-span checker used by construction and validation, not solve.

    Horizontal maximality follows from taking whole common-column runs;
    vertical maximality is checked against the two adjacent complete rows.
    """
    h, w = mask.shape
    rectangles = []
    for t in range(h):
        common = np.ones(w, dtype=bool)
        for b in range(t + 1, h + 1):
            common &= mask[b - 1]
            if b - t < minimum:
                continue
            changes = np.flatnonzero(np.diff(np.r_[False, common, False]))
            for l, r in zip(changes[::2], changes[1::2]):
                l, r = int(l), int(r)
                if r - l < minimum:
                    continue
                if t and mask[t - 1, l:r].all():
                    continue
                if b < h and mask[b, l:r].all():
                    continue
                rectangles.append((t, l, b, r))
    return sorted(rectangles)


def _squares(mask, minimum=3):
    return [q for q in _maximal_rectangles(mask, minimum)
            if q[2] - q[0] == q[3] - q[1]]


def _overlap(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _count(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def _motif(kind, side, rng):
    """Return a local blank mask and the rectangles whose geometry to preserve."""
    if kind == "overlap":
        smaller = int(rng.integers(3, side))
        dr = side - int(rng.integers(1, min(3, smaller)))
        dc = side - int(rng.integers(1, min(3, smaller)))
        # Each square extends beyond the other in both axes. Neither can
        # absorb a complete side, even when their blank components merge.
        patch = np.zeros((dr + smaller, dc + smaller), dtype=bool)
        patch[:side, :side] = True
        patch[dr:dr + smaller, dc:dc + smaller] = True
        return patch, [(0, 0, side, side),
                       (dr, dc, dr + smaller, dc + smaller)]
    if kind == "run":
        short = int(rng.integers(3, 5))
        long = short + int(rng.integers(1, 6))
        patch = np.ones((short, long), dtype=bool)
        if rng.integers(2):
            patch = patch.T.copy()
    elif kind == "small":
        patch = np.ones((2, 2), dtype=bool)
    elif kind == "near":
        patch = np.ones((3, 3), dtype=bool)
        patch[int(rng.integers(3)), int(rng.integers(3))] = False
    else:
        patch = np.ones((side, side), dtype=bool)
    h, w = patch.shape
    rectangles = [(0, 0, h, w)] if kind != "near" else []
    return patch, rectangles


def _place(mask, occupied, fixed, patch, rng, edge=False):
    h, w = mask.shape
    ph, pw = patch.shape
    options = [(t, l) for t in range(h - ph + 1)
               for l in range(w - pw + 1)]
    rng.shuffle(options)
    if edge:
        options.sort(key=lambda p: p[0] not in (0, h - ph)
                     and p[1] not in (0, w - pw))
    for t, l in options:
        b, r = t + ph, l + pw
        t0, l0, b0, r0 = max(0, t - 1), max(0, l - 1), min(h, b + 1), min(w, r + 1)
        if occupied[t0:b0, l0:r0].any():
            continue
        mask[t0:b0, l0:r0] = False
        mask[t:b, l:r] = patch
        occupied[t0:b0, l0:r0] = True
        fixed[t0:b0, l0:r0] = True
        return t, l
    return None


def _pair(rng, blockers, large_side, teaching, test):
    for attempt in range(48):
        if teaching == 0:
            plans = [("square", 3), ("square", large_side), ("run", 0), ("small", 2)]
        elif teaching == 1:
            plans = [("overlap", large_side), ("run", 0), ("small", 2)]
        else:
            primary = large_side + int(test)
            plans = [("overlap" if test or rng.random() < .5 else "square", primary)]
            for _ in range(int(rng.integers(0, 4))):
                kind = str(rng.choice(["square", "run", "small", "near"],
                                      p=[.45, .28, .13, .14]))
                plans.append((kind, int(rng.integers(3, primary + 1))))
        # Larger teaching canvases fit all contrast cases. Other pairs can
        # be elongated or compact; dimensions are independent of color.
        low = 18 if teaching in (0, 1) else max(11, large_side + 5)
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        density = float(rng.uniform(.20, .66 if not test else .74))
        mask = rng.random((h, w)) < density
        occupied = np.zeros((h, w), dtype=bool)
        fixed = np.zeros((h, w), dtype=bool)
        intended_squares, intended_runs, smalls = [], [], []
        # Largest motifs first improves packing without a visible tiling.
        patches = [(kind, *_motif(kind, side, rng)) for kind, side in plans]
        patches.sort(key=lambda x: -x[1].size)
        failed = False
        for kind, patch, rectangles in patches:
            position = _place(mask, occupied, fixed, patch, rng, edge=rng.random() < .32)
            if position is None:
                failed = True
                break
            t, l = position
            translated = [(t + a, l + b, t + c, l + d) for a, b, c, d in rectangles]
            if kind in ("square", "overlap"):
                intended_squares.extend(translated)
            elif kind == "run":
                intended_runs.extend(translated)
            elif kind == "small":
                smalls.extend(translated)
        if failed:
            continue
        # Narrow, turning attachments can connect a square to the noisy
        # exterior. A final full-side check protects its maximality.
        for rect in intended_squares:
            if rng.random() < (.35 if teaching == 0 else .8):
                t, l, b, r = rect
                direction = int(rng.integers(4))
                if direction < 2:
                    row, col = (t - 1 if direction == 0 else b), int(rng.integers(l, r))
                    dr, dc = (-1 if direction == 0 else 1), 0
                else:
                    row, col = int(rng.integers(t, b)), (l - 1 if direction == 2 else r)
                    dr, dc = 0, (-1 if direction == 2 else 1)
                for step in range(int(rng.integers(1, 6))):
                    if not (0 <= row < h and 0 <= col < w):
                        break
                    if fixed[row, col] and not (t - 1 <= row <= b and l - 1 <= col <= r):
                        break
                    mask[row, col] = True
                    if step and rng.random() < .3:
                        dr, dc = dc, -dr
                    row, col = row + dr, col + dc
        all_rectangles = _maximal_rectangles(mask)
        squares = [q for q in all_rectangles if q[2] - q[0] == q[3] - q[1] >= 3]
        if not set(intended_squares) <= set(squares):
            continue
        if not set(intended_runs + smalls) <= set(all_rectangles):
            continue
        target = _footprint(mask.shape, squares)
        if any(target[t:b, l:r].any() for t, l, b, r in intended_runs + smalls):
            continue
        if not squares or len(squares) > 24:
            continue
        if max(q[2] - q[0] for q in squares) > large_side + int(test):
            continue
        # Palette colors are semantic blockers, with fixed 0/1 roles.
        grid = rng.choice(blockers, size=mask.shape).astype(np.int8)
        grid[mask] = BLANK
        # In particular, the first demonstration exposes the full puzzle
        # blocker palette rather than relying on a probabilistic occurrence.
        for color, point in zip(blockers, np.argwhere(~mask)):
            grid[tuple(point)] = color
        out = grid.copy()
        out[target] = HIGHLIGHT
        rotations = int(rng.integers(4))
        if rng.integers(2):
            grid, out = np.fliplr(grid), np.fliplr(out)
        grid, out = np.rot90(grid, rotations).copy(), np.rot90(out, rotations).copy()
        return {"input": grid, "output": out}
    raise RuntimeError("bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, "num_train", 2, 10)
    m = _count(num_test, "num_test", 1, 4)
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    blockers = list(map(int, rng.choice(np.arange(2, 10),
                                       size=int(rng.integers(1, 5)), replace=False)))
    large_side = int(rng.integers(4, 7))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", n), ("test", m)):
        for i in range(count):
            for _ in range(12):
                if split == "train" and i == 0:
                    palette = blockers
                else:
                    size = int(rng.integers(1, len(blockers) + 1))
                    palette = rng.choice(blockers, size=size, replace=False)
                pair = _pair(rng, palette, large_side,
                             i if split == "train" and i < 2 else None, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    return puzzle


def _features(before, after):
    mask = before == BLANK
    maximal = _maximal_rectangles(mask)
    chosen = [q for q in maximal if q[2] - q[0] == q[3] - q[1] >= 3]
    target = _footprint(before.shape, chosen)
    starts = _seed_starts(mask)
    raw = _footprint(before.shape, [(t, l, t + 3, l + 3) for t, l in starts])
    small = [q for q in maximal if q[2] - q[0] == q[3] - q[1] == 2]
    # A's final generator would prioritize larger squares. Here every
    # qualifying square survives, even if a smaller one overlaps it.
    greedy = []
    for q in sorted(chosen, key=lambda q: (-(q[2] - q[0]), q)):
        if not any(_overlap(q, p) for p in greedy):
            greedy.append(q)
    return {"chosen": chosen, "target": target, "seeds": starts,
            "raw_difference": bool(np.any(raw != target)),
            "small_unmarked": any(not target[t:b, l:r].any() for t, l, b, r in small),
            "overlap_difference": bool(np.any(_footprint(before.shape, greedy) != target)),
            "has_run": any(b - t != r - l and min(b - t, r - l) >= 3
                           and not target[t:b, l:r].any() for t, l, b, r in maximal),
            "sizes": {q[2] - q[0] for q in chosen}}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, train_features = set(), []
    training_colors = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            before, after = _grid(pair["input"]), _grid(pair["output"])
            assert before.shape == after.shape
            key = _key(before)
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert HIGHLIGHT not in before and BLANK in before
            assert np.any(before > 1), "missing blocker"
            f = _features(before, after)
            target = f["target"]
            assert target.any(), "every generated example shows recoloring"
            assert np.array_equal(before != after, target), "incorrect changed footprint"
            assert np.all(after[target] == HIGHLIGHT)
            assert np.array_equal(after[~target], before[~target])
            # This compares the two distinct internal formulations. It is
            # generator-owned consistency checking, not external approval.
            assert np.array_equal(solve(before), after)
            for t, l, b, r in f["chosen"]:
                assert np.all(before[t:b, l:r] == BLANK)
                assert any(t <= sr and sr + 3 <= b and l <= sc and sc + 3 <= r
                           for sr, sc in f["seeds"])
            if split == "train":
                train_features.append(f)
                training_colors.update(map(int, np.unique(before)))
            else:
                assert set(map(int, np.unique(before))) <= training_colors
                assert max(f["sizes"]) > max(set().union(*(x["sizes"] for x in train_features))), \
                    "test must extend demonstrated size range"
    assert any(f["small_unmarked"] for f in train_features), "missing 3x3 threshold evidence"
    assert any(f["has_run"] and f["raw_difference"] for f in train_features), "missing nonsquare rejection"
    assert any(f["overlap_difference"] for f in train_features), "missing union evidence"
    assert any(3 in f["sizes"] and max(f["sizes"]) > 3 for f in train_features)
    assert any(len(f["chosen"]) >= 2 for f in train_features), "missing exhaustive detection evidence"


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test():
    """Bounded sweep; returns a compact descriptive report, writes nothing."""
    # Hand-constructed semantic probes, including order-dependent branching.
    rect = np.full((6, 9), 5, dtype=np.int8)
    rect[1:4, 1:7] = 0
    assert np.array_equal(solve(rect), rect), "3x6 must not be tiled into squares"
    small = np.zeros((2, 2), dtype=np.int8)
    assert np.array_equal(solve(small), small)
    square = np.zeros((5, 5), dtype=np.int8)
    assert np.all(solve(square) == 1), "boundaries stop expansion"
    branch = np.full((9, 10), 6, dtype=np.int8)
    branch[1:5, 1:5] = 0
    branch[1:4, 1:8] = 0
    terminal = _terminals_from_seeds(branch == 0, _seed_starts(branch == 0))
    assert terminal == _maximal_rectangles(branch == 0, 3), "explore every growth order"
    from_one_seed = _terminals_from_seeds(branch == 0, [(1, 1)])
    assert (1, 1, 5, 5) in from_one_seed and (1, 1, 4, 8) in from_one_seed
    expected = branch.copy()
    expected[1:5, 1:5] = 1
    assert np.array_equal(solve(branch), expected), "discard nonsquare branches independently"
    # Equal-size and three-way overlap have union semantics too, even
    # though the sampler usually plants a different-size overlap pair.
    overlap = np.full((10, 10), 7, dtype=np.int8)
    rects = [(0, 0, 4, 4), (2, 2, 6, 6), (4, 4, 8, 8)]
    union = _footprint(overlap.shape, rects)
    overlap[union] = 0
    expected = overlap.copy()
    expected[union] = 1
    assert np.array_equal(solve(overlap), expected), "retain equal-size multiway overlaps"
    # Every small binary field is not needed: 32 controlled random cases
    # exercise seeds that reach several square and nonsquare terminal states.
    rng = np.random.default_rng(973)
    for _ in range(32):
        mask = rng.random((int(rng.integers(3, 10)), int(rng.integers(3, 10)))) < .78
        assert _terminals_from_seeds(mask, _seed_starts(mask)) == _maximal_rectangles(mask, 3)
    shapes, palettes, sizes, counts, signatures = set(), set(), set(), set(), set()
    overlap_pairs = edge_pairs = run_pairs = multicolor_pairs = 0
    density_min, density_max = 1., 0.
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        # Hash signatures avoid retaining reconstructible puzzle payloads.
        digest = hashlib.sha256()
        for pair in puzzle["train"] + puzzle["test"]:
            before, after = pair["input"], pair["output"]
            digest.update(str(before.shape).encode())
            digest.update(before.tobytes())
            f = _features(before, after)
            shapes.add(before.shape)
            palettes.add(tuple(map(int, np.unique(before))))
            sizes |= f["sizes"]
            counts.add(len(f["chosen"]))
            overlap_pairs += f["overlap_difference"]
            run_pairs += f["has_run"]
            multicolor_pairs += len(np.unique(before)) > 2
            edge_pairs += any(t == 0 or l == 0 or b == before.shape[0] or r == before.shape[1]
                              for t, l, b, r in f["chosen"])
            density = float(np.mean(before == 0))
            density_min, density_max = min(density_min, density), max(density_max, density)
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 45 and len(palettes) >= 12
    assert {3, 4, 5, 6, 7} <= sizes and len(counts) >= 3
    assert overlap_pairs >= 42 and run_pairs >= 84 and edge_pairs > 10 and multicolor_pairs > 10
    assert density_max - density_min > .25
    assert _same_puzzle(generate_puzzle(839, 7, 3), generate_puzzle(839, 7, 3))
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(8821 + 10 * n + m, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
            validate_generated_puzzle(p)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4., 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_shapes": len(shapes),
            "input_palettes": len(palettes), "target_sizes": sorted(sizes),
            "target_counts": sorted(counts), "overlap_union_pairs": overlap_pairs,
            "preserved_run_pairs": run_pairs, "edge_pairs": edge_pairs,
            "multiple_blocker_pairs": multicolor_pairs,
            "blank_fraction_range": [round(density_min, 4), round(density_max, 4)],
            "independent_semantic_approval": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
