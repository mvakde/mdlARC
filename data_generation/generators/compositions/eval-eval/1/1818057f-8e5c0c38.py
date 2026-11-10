"""Plus-landmark consensus elects the frame for per-color mirror cleanup.

Only NumPy and the standard library are used. solve() is input-only; neither
parent modules, sampled parameters, nor demonstrations enter its code path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grid must be an integer 1..30 by 1..30 array of colors 0..9")
    return a


def _background(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    color = int(counts.argmax())
    if counts[color] <= grid.size // 2:
        raise ValueError("a strict-majority background is required")
    if counts[color] == grid.size:
        raise ValueError("at least one foreground color is required")
    return color


def _centers(mask):
    centers = np.zeros_like(mask, dtype=bool)
    if min(mask.shape) >= 3:
        centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                               & mask[2:, 1:-1] & mask[1:-1, :-2]
                               & mask[1:-1, 2:])
    return centers


def _cross_union(centers):
    union = centers.copy()
    union[:-1] |= centers[1:]
    union[1:] |= centers[:-1]
    union[:, :-1] |= centers[:, 1:]
    union[:, 1:] |= centers[:, :-1]
    return union


def _scores(mask):
    """Scores count cells, not pairs: ordered convolution includes self-support."""
    x = mask.astype(np.int16)
    vertical = sum((np.convolve(row, row) for row in x),
                   np.zeros(2 * x.shape[1] - 1, dtype=int))
    horizontal = sum((np.convolve(col, col) for col in x.T),
                     np.zeros(2 * x.shape[0] - 1, dtype=int))
    return vertical, horizontal


def _elect(centers):
    scores = _scores(centers)
    best = max(int(s.max()) for s in scores)
    winners = [(direction, int(a2)) for direction, s in enumerate(scores)
               for a2 in np.flatnonzero(s == best)]
    if best < 3 or len(winners) != 1:
        raise ValueError("each color needs a unique landmark axis with score at least three")
    return winners[0]


def _support(mask, election):
    direction, a2 = election
    n = mask.shape[1 - direction]
    reflected = a2 - np.arange(n)
    valid = (reflected >= 0) & (reflected < n)
    keep = np.zeros_like(mask, dtype=bool)
    if direction == 0:
        keep[:, valid] = mask[:, valid] & mask[:, reflected[valid]]
    else:
        keep[valid, :] = mask[valid, :] & mask[reflected[valid], :]
    return keep


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Elect axes using input pluses, then retain original reflected pixels."""
    grid = _grid(input_grid)
    bg = _background(grid)
    output = np.full(grid.shape, bg, dtype=grid.dtype)
    for color in np.unique(grid):
        if int(color) == bg:
            continue
        mask = grid == color
        axis = _elect(_centers(mask))
        output[_support(mask, axis)] = color
    return output


def _paint_cross(mask, r, c):
    for dr, dc in _OFFSETS:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]):
            raise ValueError("construction attempted to crop a cross")
        mask[rr, cc] = True


def _int(rng, low, high):
    return int(rng.integers(low, high + 1))


def _sequential_centers(mask):
    work = mask.copy()
    found = np.zeros_like(mask)
    for r in range(1, mask.shape[0] - 1):
        for c in range(1, mask.shape[1] - 1):
            points = [(r + dr, c + dc) for dr, dc in _OFFSETS]
            if all(work[p] for p in points):
                found[r, c] = True
                for p in points:
                    work[p] = False
    return found


def _features(mask, axis=None):
    centers = _centers(mask)
    axis = _elect(centers) if axis is None else axis
    keep = _support(mask, axis)
    supported_centers = _support(centers, axis)
    paired_union = _cross_union(supported_centers)
    raw_scores = _scores(mask)
    raw_at_axis = int(raw_scores[axis[0]][axis[1]])
    raw_best = max(int(s.max()) for s in raw_scores)
    partial = False
    for r, c in np.argwhere(centers & ~supported_centers):
        points = [(int(r + dr), int(c + dc)) for dr, dc in _OFFSETS]
        partial |= any(keep[p] for p in points) and any(not keep[p] for p in points)
    dimension = 1 - axis[0]
    occupied = np.nonzero(mask)[dimension]
    bbox_axis = int(occupied.min() + occupied.max())
    bbox_wrong = not np.array_equal(keep, _support(mask, (axis[0], bbox_axis)))
    return {
        "direction": axis[0], "axis2": axis[1], "parity": axis[1] % 2,
        "centers": int(centers.sum()), "votes": int(supported_centers.sum()),
        "cells": int(mask.sum()), "kept": int(keep.sum()),
        "erased": int((mask & ~keep).sum()),
        "raw_wrong": raw_best > raw_at_axis,
        "ordinary": bool((keep & ~paired_union).any()),
        "partial": bool(partial), "bbox_wrong": bool(bbox_wrong),
        "overlap": 5 * int(centers.sum()) > int(_cross_union(centers).sum()),
        "destructive_wrong": not np.array_equal(centers, _sequential_centers(mask)),
        "edge": bool(keep[0].any() or keep[-1].any()
                     or keep[:, 0].any() or keep[:, -1].any()),
    }


def _add_noise(rng, mask, budget, rogue_crosses=0):
    """Accept only changes preserving the observable intended landmark election."""
    axis = _elect(_centers(mask))
    assert axis[0] == 0
    a2 = axis[1]
    h, w = mask.shape
    for _ in range(rogue_crosses):
        candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        for index in rng.permutation(len(candidates))[:90]:
            r, c = candidates[int(index)]
            trial = mask.copy()
            _paint_cross(trial, r, c)
            # A new center must be unpaired, not merely an extension of a
            # supported motif. Its cells may nevertheless partly survive.
            new_centers = _centers(trial)
            reflected = a2 - c
            if (not new_centers[r, c] or _centers(mask)[r, c]
                    or (0 <= reflected < w and new_centers[r, reflected])):
                continue
            try:
                if _elect(new_centers) != axis:
                    continue
            except ValueError:
                continue
            mask = trial
            break
    candidates = list(map(tuple, np.argwhere(~mask).tolist()))
    accepted = 0
    for index in rng.permutation(len(candidates)):
        if accepted >= budget:
            break
        r, c = candidates[int(index)]
        reflected = a2 - c
        if 2 * c == a2 or (0 <= reflected < w and mask[r, reflected]):
            continue
        trial = mask.copy()
        trial[r, c] = True
        try:
            if _elect(_centers(trial)) != axis:
                continue
        except ValueError:
            continue
        mask = trial
        accepted += 1
    return mask


def _strip_witness(rng, compact=False):
    """Area decoy plus a one-sided completion of a mirrored pair of T shapes."""
    a = 4 if compact else _int(rng, 4, 6)
    w = 12 if compact else _int(rng, a + 8, 17)
    lower = 6 if compact else _int(rng, 6, 7)
    bands = 2 if compact else _int(rng, 2, 3)
    start = lower + 3
    h = 15 if compact else _int(rng, start + 3 * (bands - 1) + 3, 20)
    mask = np.zeros((h, w), dtype=bool)
    top = 2 if compact else _int(rng, 1, 2)
    for r, c in ((top, 1), (top, 2 * a - 1), (lower, a)):
        _paint_cross(mask, r, c)
    # Two mirrored T shapes; adding only one south arm creates an unpaired
    # input plus whose other four cells are legitimately supported.
    for c in (1, 2 * a - 1):
        for dr, dc in ((0, 0), (-1, 0), (0, -1), (0, 1)):
            mask[lower - 1 + dr, c + dc] = True
    mask[lower, 1] = True
    mask[0, 3] = mask[0, 2 * a - 3] = True
    for band in range(bands):
        r = start + 3 * band
        mask[r:r + 2, a:] = True
    assert _elect(_centers(mask)) == (0, 2 * a)
    if not compact:
        mask = _add_noise(rng, mask, _int(rng, 0, 3))
    f = _features(mask)
    assert f["raw_wrong"] and f["ordinary"] and f["partial"] and f["bbox_wrong"]
    return mask


def _compact_core(parity):
    if parity == 1:
        mask = np.zeros((9, 10), dtype=bool)
        mask[1:4, 3:7] = True
        mask[2, 1:9] = True
        for c in (1, 8):
            _paint_cross(mask, 6, c)
        mask[0, 1] = mask[0, 8] = True
    else:
        mask = np.zeros((9, 9), dtype=bool)
        for r, c in ((2, 1), (2, 7), (6, 4)):
            _paint_cross(mask, r, c)
        mask[0, 2] = mask[0, 6] = True
    assert _elect(_centers(mask)) == (0, mask.shape[1] - 1)
    return mask


def _ordinary_core(rng, parity, richness):
    for _ in range(25):
        h = _int(rng, 7, 12)
        widths = [w for w in range(8, 14) if (w - 1) % 2 == parity]
        w = int(rng.choice(widths))
        a2 = w - 1
        mask = np.zeros((h, w), dtype=bool)
        top = _int(rng, 1, 2)
        bottom = _int(rng, h - 4, h - 2)
        for r, c in ((top, 1), (bottom, 2)):
            _paint_cross(mask, r, c)
            _paint_cross(mask, r, a2 - c)
        if parity == 0 and rng.random() < .75:
            _paint_cross(mask, _int(rng, 1, h - 2), a2 // 2)
        for _ in range(_int(rng, 0, 1 + richness)):
            r, c = _int(rng, 1, h - 2), _int(rng, 1, w // 2 - 1)
            _paint_cross(mask, r, c)
            _paint_cross(mask, r, a2 - c)
        try:
            if _elect(_centers(mask)) != (0, a2):
                continue
        except ValueError:
            continue
        # Independent mirrored clutter vocabulary. Accidental plus centers
        # count exactly like deliberately planted centers.
        for _ in range(_int(rng, 1, 3 + richness)):
            trial = mask.copy()
            kind = _int(rng, 0, 3)
            r, c = _int(rng, 0, h - 1), _int(rng, 0, w - 1)
            points = [(r, c)]
            if kind == 1:  # straight strokes
                dr, dc = ((1, 0), (0, 1))[_int(rng, 0, 1)]
                points += [(r + i * dr, c + i * dc) for i in range(_int(rng, 2, 5))]
            elif kind == 2:  # thin contours
                hh, ww = _int(rng, 2, 4), _int(rng, 2, 5)
                points += [(r + i, c + j) for i in range(hh) for j in range(ww)
                           if i in (0, hh - 1) or j in (0, ww - 1)]
            elif kind == 3:  # bends and branches
                for _ in range(_int(rng, 2, 7)):
                    dr, dc = _OFFSETS[_int(rng, 1, 4)]
                    r, c = r + dr, c + dc
                    points.append((r, c))
            for rr, cc in points:
                if 0 <= rr < h and 0 <= cc < w:
                    trial[rr, cc] = trial[rr, a2 - cc] = True
            try:
                if _elect(_centers(trial)) == (0, a2):
                    mask = trial
            except ValueError:
                pass
        return mask
    return _compact_core(parity)


def _object(rng, kind, parity, direction, richness):
    if kind == "strip":
        mask = _strip_witness(rng)
    elif kind == "overlap":
        mask = _compact_core(1)
        # Stretch the gaps between bands and their symmetric surroundings.
        if rng.random() < .6:
            gap = _int(rng, 1, 3)
            mask = np.concatenate((mask[:4], np.zeros((gap, 10), dtype=bool), mask[4:]))
        mask = _add_noise(rng, mask, _int(rng, 1, 6), _int(rng, 0, 1))
    else:
        mask = _ordinary_core(rng, parity, richness)
        mask = _add_noise(rng, mask, _int(rng, 0, 3 + 3 * richness), _int(rng, 0, 2))
    if rng.random() < .5:
        mask = mask[::-1].copy()
    if rng.random() < .5:
        mask = mask[:, ::-1].copy()
    return mask.T.copy() if direction else mask


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _pack(rng, objects, colors, background, seen):
    count = len(objects)
    for _ in range(55):
        low = (12, 19, 25, 26)[count - 1]
        min_h = max(low, max(m.shape[0] for m in objects))
        min_w = max(low, max(m.shape[1] for m in objects))
        h, w = _int(rng, min_h, 30), _int(rng, min_w, 30)
        if rng.random() < .20:
            h = w = max(h, w)
        if sum(int(m.sum()) for m in objects) >= h * w / 2:
            continue
        grid = np.full((h, w), background, dtype=np.int8)
        order = sorted(range(count), key=lambda i: -int(objects[i].sum()))
        success = True
        for i in order:
            mask = objects[i]
            mh, mw = mask.shape
            positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
            placed = False
            for index in rng.permutation(len(positions))[:100]:
                r, c = positions[int(index)]
                patch = grid[r:r + mh, c:c + mw]
                if np.any(patch[mask] != background):
                    continue
                patch[mask] = colors[i]
                placed = True
                break
            if not placed:
                success = False
                break
        if success and _key(grid) not in seen:
            return grid
    return None


def _fallback_grid(rng, count, colors, bg, seen, main_direction, first=False):
    # The first witness is single-color; the other witness and all test cases
    # can use the small, semantically identical half/integral constellations.
    if first:
        masks = [_strip_witness(rng)]
        if main_direction:
            masks[0] = masks[0].T.copy()
    else:
        masks = [_compact_core(1 if i == 0 else i % 2) for i in range(count)]
        if count >= 3:
            masks[0] = _strip_witness(rng, compact=True)
        masks = [m.T.copy() if ((main_direction + i + 1) % 2) else m
                 for i, m in enumerate(masks)]
    # At least 25 canvas shapes, each feasible. A puzzle has at most fourteen inputs,
    # so an unused shape/content is guaranteed even with fixed fallback cells.
    low = 26 if count >= 3 else 23
    gap = 15 if count >= 3 else 12
    for h in range(low, 31):
        for w in range(low, 31):
            grid = np.full((h, w), bg, dtype=np.int8)
            for i, (mask, color) in enumerate(zip(masks, colors)):
                r, c = (0, 0) if first else ((i // 2) * gap, (i % 2) * gap)
                patch = grid[r:r + mask.shape[0], c:c + mask.shape[1]]
                patch[mask] = color
            if _key(grid) not in seen:
                return grid
    raise AssertionError("finite fallback uniqueness bound exceeded")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    main = _int(rng, 0, 1)
    richness = _int(rng, 0, 2)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, number in (("train", ntrain), ("test", ntest)):
        for index in range(number):
            first = split == "train" and index == 0
            second = split == "train" and index == 1
            count = 1 if first else 2 if second else _int(rng, 1, 2) if split == "train" else _int(rng, 3, 4)
            palette = rng.choice(10, count + 1, replace=False)
            bg, colors = int(palette[0]), list(map(int, palette[1:]))
            objects = []
            for i in range(count):
                if first:
                    kind, parity, direction = "strip", 0, main
                elif second:
                    kind = "overlap" if i == 0 else "ordinary"
                    parity = 1 if i == 0 else _int(rng, 0, 1)
                    direction = 1 - main if i == 0 else main
                elif split == "test" and i == 0:
                    kind, parity, direction = "strip", 0, _int(rng, 0, 1)
                else:
                    # Extra demonstrations and remaining test figures sample
                    # a broad mixture of smaller constellations.
                    kind = "strip" if count == 1 and rng.random() < .28 else str(rng.choice(("ordinary", "overlap"), p=(.75, .25)))
                    parity, direction = _int(rng, 0, 1), _int(rng, 0, 1)
                objects.append(_object(rng, kind, parity, direction, richness))
            grid = _pack(rng, objects, colors, bg, seen)
            if grid is None:
                grid = _fallback_grid(rng, count, colors, bg, seen, main, first)
            pair = {"input": grid, "output": solve(grid)}
            puzzle[split].append(pair)
            seen.add(_key(grid))
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(ntrain)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    training = []
    mixed = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "inputs must be distinct within the puzzle"
            seen.add(_key(inp))
            bg = _background(inp)
            assert np.all((out == inp) | (out == bg)), "only erase to the background"
            assert np.array_equal(out[inp == bg], inp[inp == bg])
            colors = [int(c) for c in np.unique(inp) if int(c) != bg]
            assert (1 <= len(colors) <= 2) if split == "train" else (3 <= len(colors) <= 4)
            directions = set()
            pair_features = []
            for color in colors:
                mask = inp == color
                axis = _elect(_centers(mask))
                expected = _support(mask, axis)
                assert np.array_equal(out == color, expected), "label violates elected reflected support"
                assert np.array_equal(expected, _support(expected, axis)), "retained set must be symmetric"
                assert np.all(expected <= mask)
                directions.add(axis[0])
                features = _features(mask, axis)
                pair_features.append(features)
                if split == "train":
                    training.append(features)
            if split == "train":
                mixed |= len(directions) == 2
            else:
                assert any(f["raw_wrong"] for f in pair_features), "each query must require A's landmark election"
                assert any(f["erased"] > 0 for f in pair_features), "each query must require B's deletion"
    for feature in ("raw_wrong", "ordinary", "partial", "bbox_wrong", "destructive_wrong", "overlap"):
        assert any(f[feature] for f in training), f"training needs a {feature} counterexample"
    assert any(f["erased"] > 0 for f in training), "B omission must change a demonstration"
    assert {f["direction"] for f in training} == {0, 1}
    assert {f["parity"] for f in training} == {0, 1}
    assert mixed, "training must show independently oriented colors in one input"


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded semantic regressions and measurements of 42 actual puzzles."""
    shapes, palettes, elections, center_counts, fingerprints = set(), set(), set(), set(), set()
    metrics = {name: 0 for name in ("raw_wrong", "ordinary", "partial", "bbox_wrong", "destructive_wrong", "overlap", "edge")}
    erased_counts, colors_by_split, backgrounds = set(), {"train": set(), "test": set()}, set()
    rows = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        local_shapes, local_centers = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                assert np.array_equal(solve(inp.copy()), pair["output"])
                bg = _background(inp)
                foreground = [int(c) for c in np.unique(inp) if int(c) != bg]
                shapes.add(inp.shape)
                local_shapes.add(inp.shape)
                palettes.add(tuple(map(int, np.unique(inp))))
                backgrounds.add(bg)
                colors_by_split[split].add(len(foreground))
                for color in foreground:
                    f = _features(inp == color)
                    elections.add((f["direction"], f["parity"]))
                    center_counts.add(f["centers"])
                    local_centers.add(f["centers"])
                    erased_counts.add(f["erased"])
                    for name in metrics:
                        metrics[name] += int(f[name])
        rows.append({"seed": seed, "shapes": len(local_shapes), "landmark_counts": len(local_centers)})
        assert len(local_shapes) >= 2
        assert len(local_centers) >= 2
        if seed in (0, 17, 41):
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
    assert len(fingerprints) == 42
    assert len(shapes) >= 40 and any(30 in s for s in shapes)
    assert any(h == w for h, w in shapes) and any(h != w for h, w in shapes)
    assert len(palettes) >= 40 and backgrounds == set(range(10))
    assert elections == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(center_counts) >= 8 and len(erased_counts) >= 10
    assert colors_by_split == {"train": {1, 2}, "test": {3, 4}}
    assert all(n >= 10 for n in metrics.values())
    # Exhaust all binary 3x3 neighborhoods: only the five cardinal cells matter.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        expected = all(mask[p] for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)))
        assert bool(_centers(mask)[1, 1]) == expected
    # A reflected constellation: two off-axis centers plus one self-supporting
    # center. Padding and transposition change its coordinates, not its rule.
    landmark = np.zeros((9, 9), dtype=bool)
    landmark[2, 1] = landmark[2, 7] = landmark[6, 4] = True
    assert _elect(landmark) == (0, 8)
    assert _elect(landmark.T) == (1, 8)
    padded = np.pad(landmark, ((2, 0), (3, 0)))
    assert _elect(padded) == (0, 14)
    assert int(_scores(landmark)[0][8]) == 3
    assert np.array_equal(_support(landmark, (0, 8)), landmark)
    boundary = _compact_core(1)
    assert _features(boundary)["edge"]
    clipped = np.zeros((5, 6), dtype=bool)
    clipped[0, 1:4] = True
    clipped[1, 2] = clipped[-1, 2] = True
    assert not _centers(clipped).any(), "no wrapping or boundary-centered incomplete crosses"
    for invalid in (np.zeros((6, 6), dtype=int), np.array([[0, 1], [1, 0]])):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted landmark-free or tied-background input")
    symmetric_cross = np.zeros((5, 5), dtype=np.int8)
    _paint_cross(symmetric_cross, 2, 2)
    try:
        solve(symmetric_cross)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted tied/insufficient landmark evidence")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(42 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    # Wrong label: deleting a supported cell must fail exact-support validation.
    bad = generate_puzzle(99, 2, 1)
    out = bad["test"][0]["output"]
    bg = _background(out)
    r, c = np.argwhere(out != bg)[0]
    out[r, c] = bg
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted deletion of a supported cell")
    # All fallback counts/orientations must remain in the same solvable family.
    for direction in (0, 1):
        for count in range(1, 5):
            inp = _fallback_grid(np.random.default_rng(0), count, list(range(1, count + 1)),
                                 0, set(), direction, first=(count == 1))
            out = solve(inp)
            assert np.all((out == inp) | (out == 0))
    return {"seeds": 42, "puzzles": len(fingerprints), "grid_shapes": len(shapes),
            "palettes": len(palettes), "landmark_counts": sorted(center_counts),
            "erased_counts": sorted(erased_counts), "features": metrics, "within_puzzle": rows}


if __name__ == "__main__":
    # Optional local inspection; JSON stays compact and digit grids are readable.
    destination = Path("scratch")
    destination.mkdir(exist_ok=True)
    puzzle = generate_puzzle(0)
    encoded = {s: [{k: v.tolist() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    (destination / "seed-0.json").write_text(json.dumps(encoded, separators=(",", ":")) + "\n")
    lines = []
    for split in ("train", "test"):
        for i, pair in enumerate(puzzle[split], 1):
            lines.append(f"{split} pair {i}")
            for role in ("input", "output"):
                lines.append(role + ":")
                lines.extend("".join(map(str, row)) for row in pair[role].tolist())
            lines.append("")
    (destination / "seed-0.txt").write_text("\n".join(lines))
    report = self_test()
    (destination / "self-test.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report, separators=(",", ":")))
