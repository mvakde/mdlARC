"""Competing defect ripples: two colored islands share one square-wave chamber.

Only the standard library and NumPy are used. No parent code or data is imported.
The sampler builds geometry; solve reconstructs it from the input alone.
"""

from collections import Counter
import json

import numpy as np


SIDES = ("top", "right", "bottom", "left")


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _check_grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("Expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return g


def _perimeter(shape, bounds):
    t, b, l, r = bounds
    mask = np.zeros(shape, dtype=bool)
    mask[t, l:r + 1] = mask[b, l:r + 1] = True
    mask[t:b + 1, l] = mask[t:b + 1, r] = True
    return mask


def _rectangle_mask(shape, rect):
    t, b, l, r = rect
    mask = np.zeros(shape, dtype=bool)
    mask[t:b + 1, l:r + 1] = True
    return mask


def _edge_options(rect):
    t, b, l, r = rect
    return {
        "top": [((t, x), (t - 1, x)) for x in range(l + 1, r)],
        "bottom": [((b, x), (b + 1, x)) for x in range(l + 1, r)],
        "left": [((y, l), (y, l - 1)) for y in range(t + 1, b)],
        "right": [((y, r), (y, r + 1)) for y in range(t + 1, b)],
    }


def _parse(input_grid):
    """Recover the frame, ideal islands, and every tab/notch without outputs."""
    g = _check_grid(input_grid)
    colors = [int(c) for c in np.unique(g)]
    if len(colors) != 4:
        raise ValueError("The four visible roles must have distinct colors")
    frames = []
    for c in colors:
        yy, xx = np.where(g == c)
        bounds = (int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max()))
        t, b, l, r = bounds
        if b - t >= 8 and r - l >= 8:
            if np.array_equal(g == c, _perimeter(g.shape, bounds)):
                frames.append((c, bounds))
    if len(frames) != 1:
        raise ValueError("Expected one unambiguous complete rectangular frame")
    frame, bounds = frames[0]
    t, b, l, r = bounds
    bg = int(g[t + 1, l + 1])
    carriers = sorted(set(colors) - {frame, bg})
    if bg == frame or len(carriers) != 2:
        raise ValueError("The chamber corner must expose the background")
    expected = np.full(g.shape, bg, dtype=g.dtype)
    expected[_perimeter(g.shape, bounds)] = frame
    rects, defects = [], []
    for color in carriers:
        mask = g == color
        # Extra tab rows/columns have at most two cells. Every true rectangle
        # row/column has at least three, including boundary rows with notches.
        yy = np.flatnonzero(mask.sum(axis=1) >= 3)
        xx = np.flatnonzero(mask.sum(axis=0) >= 3)
        if len(yy) < 5 or len(xx) < 5:
            raise ValueError("Insufficient intact rectangle evidence")
        rect = (int(yy[0]), int(yy[-1]), int(xx[0]), int(xx[-1]))
        a, z, c, d = rect
        if (not np.array_equal(yy, np.arange(a, z + 1))
                or not np.array_equal(xx, np.arange(c, d + 1))
                or not (t + 2 <= a < z <= b - 2 and l + 2 <= c < d <= r - 2)):
            raise ValueError("Islands need intact cores and a halo inside the frame")
        ideal = _rectangle_mask(g.shape, rect)
        local = []
        options = _edge_options(rect)
        by_boundary = {p: (side, q) for side, entries in options.items() for p, q in entries}
        by_outward = {q: (side, p) for side, entries in options.items() for p, q in entries}
        for y, x in np.argwhere(ideal & ~mask):
            p = (int(y), int(x))
            if p not in by_boundary or int(g[p]) != bg:
                raise ValueError("Only noncorner one-cell boundary notches are allowed")
            side, q = by_boundary[p]
            local.append((side, "notch", p, q))
        for y, x in np.argwhere(mask & ~ideal):
            q = (int(y), int(x))
            if q not in by_outward:
                raise ValueError("Only noncorner one-cell outward tabs are allowed")
            side, p = by_outward[q]
            local.append((side, "tab", p, q))
        counts = Counter(v[0] for v in local)
        if not 1 <= len(local) <= 5 or max(counts.values()) > 2:
            raise ValueError("Each island has 1..5 sparse defects, at most two per side")
        for i, (_, _, p, _) in enumerate(local):
            if any(abs(p[0] - v[2][0]) + abs(p[1] - v[2][1]) < 2 for v in local[:i]):
                raise ValueError("Defects cannot occupy neighboring boundary cells")
        expected[ideal] = color
        for _, kind, p, q in local:
            expected[p if kind == "notch" else q] = bg if kind == "notch" else color
        rects.append(rect)
        defects.append(local)
    a, brect = rects
    if not (a[1] + 3 < brect[0] or brect[1] + 3 < a[0]
            or a[3] + 3 < brect[2] or brect[3] + 3 < a[2]):
        raise ValueError("The ideal islands need at least three separating background cells")
    if not np.array_equal(expected, g):
        raise ValueError("Unexpected noise, overlapping roles, or a malformed defect")
    return {"frame": frame, "bounds": bounds, "background": bg,
            "colors": tuple(carriers), "rects": rects, "defects": defects}


def _distances(shape, defects, metric="chebyshev", actual_only=False, outward_only=False):
    yy, xx = np.indices(shape)
    distances = []
    for local in defects:
        d = np.full(shape, 100, dtype=int)
        for _, kind, p, q in local:
            points = (q,) if outward_only else ((q if kind == "tab" else p,) if actual_only else (p, q))
            for y, x in points:
                dy, dx = np.abs(yy - y), np.abs(xx - x)
                candidate = dy + dx if metric == "manhattan" else np.maximum(dy, dx)
                d = np.minimum(d, candidate)
        distances.append(d)
    return distances


def _masks(shape, info):
    t, b, l, r = info["bounds"]
    chamber = np.zeros(shape, dtype=bool)
    chamber[t + 1:b, l + 1:r] = True
    protected = np.zeros(shape, dtype=bool)
    for rect in info["rects"]:
        protected |= _rectangle_mask(shape, rect)
    return chamber, protected


def _render(g, info, metric="chebyshev", actual_only=False, exchange=True,
            outward_only=False, markers_only=False, keep_ties=False):
    """Render from explicit geometry; optional hypotheses are for validation."""
    out = g.copy()
    chamber, protected = _masks(g.shape, info)
    free = chamber & ~protected
    # Restore the ideal shape first; every missing cell will then be part of its bar.
    for color, rect in zip(info["colors"], info["rects"]):
        out[_rectangle_mask(g.shape, rect)] = color
    if not markers_only:
        d0, d1 = _distances(g.shape, info["defects"], metric, actual_only, outward_only)
        out[free] = info["background"]
        for owner, (d, other) in enumerate(((d0, d1), (d1, d0))):
            won = free & (d < other)
            if keep_ties and owner == 0:
                won |= free & (d == other)
            even, odd = info["colors"][1 - owner], info["colors"][owner]
            if not exchange:
                even, odd = odd, even
            out[won] = np.where(d[won] % 2 == 0, even, odd)
    for owner, local in enumerate(info["defects"]):
        color = info["colors"][1 - owner if exchange else owner]
        for _, _, p, q in local:
            out[p] = out[q] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from visible geometry and local color relationships."""
    g = _check_grid(input_grid)
    return _render(g, _parse(g))


def _sample_rectangles(rng, room_h, room_w):
    # All disjoint axis-aligned rectangle pairs have a separating row or column.
    # Sample that axis, arbitrary independent cross-axis spans, order, gap and slack.
    axes = [axis for axis, length in enumerate((room_h, room_w)) if length >= 15]
    axis = int(rng.choice(axes))
    length, cross = (room_h, room_w) if axis == 0 else (room_w, room_h)
    n0 = _ri(rng, 5, min(12, length - 10))
    n1 = _ri(rng, 5, min(12, length - 5 - n0))
    gap = _ri(rng, 3, length - 2 - n0 - n1)
    slack = length - 2 - n0 - n1 - gap
    start = 1 + _ri(rng, 0, slack)
    rects = []
    for offset, size in ((start, n0), (start + n0 + gap, n1)):
        extent = _ri(rng, 5, min(12, cross - 2))
        cross_start = _ri(rng, 1, cross - 1 - extent)
        rect = (offset, offset + size - 1, cross_start, cross_start + extent - 1)
        rects.append(rect if axis == 0 else (rect[2], rect[3], rect[0], rect[1]))
    if rng.integers(2):
        rects.reverse()
    return rects


def _sample_defects(rng, rect, count, forced_sides=(), mixed=False, forced_kind=None):
    options = _edge_options(rect)
    for _ in range(40):
        selected = []
        for index in range(count):
            sides = (forced_sides[index],) if index < len(forced_sides) else SIDES
            candidates = [(side, p, q) for side in sides for p, q in options[side]
                          if sum(v[0] == side for v in selected) < 2
                          and all(abs(p[0] - v[2][0]) + abs(p[1] - v[2][1]) >= 2
                                  for v in selected)]
            if not candidates:
                break
            side, p, q = candidates[_ri(rng, 0, len(candidates) - 1)]
            kind = str(rng.choice(("tab", "notch")))
            if forced_kind is not None and index == 0:
                kind = forced_kind
            if mixed and index < 2:
                kind = ("tab", "notch")[index]
            selected.append((side, kind, p, q))
        if len(selected) == count:
            return selected
    raise RuntimeError("Could not select sparse defects")


def _diagnostics(g, info):
    chamber, protected = _masks(g.shape, info)
    free = chamber & ~protected
    d0, d1 = _distances(g.shape, info["defects"])
    nearest = np.minimum(d0, d1)
    ties = free & (d0 == d1)
    same_owner_tie = False
    for i, local in enumerate(info["defects"]):
        if len(local) < 2:
            continue
        individual = [_distances(g.shape, [[defect]])[0] for defect in local]
        stack = np.stack(individual)
        own_min = stack.min(axis=0)
        same_owner_tie |= bool(np.any(free & (own_min < (d1 if i == 0 else d0))
                                      & ((stack == own_min).sum(axis=0) >= 2)))
    return {"tie_cells": int(ties.sum()), "same_owner_tie": same_owner_tie,
            "levels": set(int(x) for x in nearest[free & ~ties]),
            "source_count": sum(map(len, info["defects"])),
            "sides": {v[0] for local in info["defects"] for v in local},
            "kinds": {v[1] for local in info["defects"] for v in local},
            "repeated_side": any(len({v[0] for v in local}) < len(local)
                                 for local in info["defects"])}


def _sample_pair(rng, shift, split, style, previous_carriers=None):
    for _ in range(240):
        if split == "test" and shift == "aspect":
            h, w = _ri(rng, 9, 14), _ri(rng, 22, 26)
            if rng.integers(2):
                h, w = w, h
        elif split == "test" and shift == "compression":
            h, w = _ri(rng, 15, 18), _ri(rng, 15, 18)
        elif shift == "compression":
            h, w = _ri(rng, 22, 26), _ri(rng, 22, 26)
        elif shift == "aspect":
            h, w = _ri(rng, 17, 24), _ri(rng, 17, 24)
        else:
            h, w = _ri(rng, 15, 26), _ri(rng, 15, 26)
        # Chamber is h by w; the complete frame adds two cells per axis.
        mt = _ri(rng, 0, min(2, 28 - h))
        mb = _ri(rng, 0, min(2, 28 - h - mt))
        ml = _ri(rng, 0, min(2, 28 - w))
        mr = _ri(rng, 0, min(2, 28 - w - ml))
        shape = (h + 2 + mt + mb, w + 2 + ml + mr)
        bounds = (mt, mt + h + 1, ml, ml + w + 1)
        bg, frame, c0, c1 = (int(v) for v in rng.choice(10, size=4, replace=False))
        if previous_carriers is not None and {c0, c1} == set(previous_carriers):
            continue
        rects = [(a + mt + 1, b + mt + 1, c + ml + 1, d + ml + 1)
                 for a, b, c, d in _sample_rectangles(rng, h, w)]
        sides = list(rng.permutation(SIDES))
        if style == "simple":
            counts, forced = (1, 1), ((sides[0],), (sides[1],))
        elif style == "witness":
            counts = (3, _ri(rng, 2, 3))
            forced = ((sides[0], sides[0], sides[1]), (sides[2], sides[3]))
        elif split == "test" and shift == "multiplicity":
            total = _ri(rng, 8, 10)
            counts = (4, 4) if total == 8 else ((5, 5) if total == 10 else (4, 5))
            if rng.integers(2):
                counts = counts[::-1]
            forced = ((), ())
        else:
            maximum = 5 if split == "test" else 3
            counts, forced = (_ri(rng, 1, maximum), _ri(rng, 1, maximum)), ((), ())
        try:
            defects = [_sample_defects(rng, rect, count, forced[i], mixed=style == "witness",
                                       forced_kind=("tab", "notch")[i] if style == "simple" else None)
                       for i, (rect, count) in enumerate(zip(rects, counts))]
        except RuntimeError:
            continue
        info = {"frame": frame, "background": bg, "bounds": bounds,
                "colors": (c0, c1), "rects": rects, "defects": defects}
        g = np.full(shape, bg, dtype=np.int8)
        g[_perimeter(shape, bounds)] = frame
        for color, rect, local in zip(info["colors"], rects, defects):
            g[_rectangle_mask(shape, rect)] = color
            for _, kind, p, q in local:
                g[p if kind == "notch" else q] = bg if kind == "notch" else color
        out = _render(g, info)
        if style in ("simple", "witness"):
            diag = _diagnostics(g, info)
            if not diag["tie_cells"] or not {1, 2, 3, 4} <= diag["levels"]:
                continue
            if style == "witness" and not diag["same_owner_tie"]:
                continue
            if any(np.array_equal(out, alternative) for alternative in (
                    _render(g, info, metric="manhattan"),
                    _render(g, info, actual_only=True),
                    _render(g, info, outward_only=True))):
                continue
        return {"input": g, "output": out}
    raise RuntimeError("Bounded sampling could not produce the requested evidence")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, count, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or not lo <= count <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("aspect", "multiplicity", "compression")))
    train_styles = ["simple", "witness"] + ["random"] * (num_train - 2)
    rng.shuffle(train_styles)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, styles in (("train", train_styles), ("test", ["random"] * num_test)):
        previous = None
        for style in styles:
            for _ in range(30):
                pair = _sample_pair(rng, shift, split, style, previous)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    previous = _parse(pair["input"])["colors"]
                    break
            else:
                raise RuntimeError("Could not generate a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, geometry, exact wave semantics, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_infos, train_diags = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert g.shape == out.shape
            key = (g.shape, g.astype(np.int8).tobytes())
            assert key not in seen
            seen.add(key)
            info = _parse(g)
            chamber, protected = _masks(g.shape, info)
            assert np.array_equal(out[~chamber], g[~chamber])
            # Scalar checks explicitly verify distance parity and conflicting ties.
            endpoints = [[point for v in local for point in v[2:]] for local in info["defects"]]
            for y, x in np.argwhere(chamber & ~protected):
                dd = [min(max(abs(int(y) - p[0]), abs(int(x) - p[1])) for p in group)
                      for group in endpoints]
                if dd[0] == dd[1]:
                    color = info["background"]
                else:
                    owner = 0 if dd[0] < dd[1] else 1
                    color = info["colors"][owner if dd[owner] % 2 else 1 - owner]
                assert int(out[y, x]) == color, "Incorrect shell, competition, or tie"
            for owner, (rect, local) in enumerate(zip(info["rects"], info["defects"])):
                intact = _rectangle_mask(g.shape, rect)
                for _, _, p, q in local:
                    assert out[p] == out[q] == info["colors"][1 - owner], "Incorrect exchanged domino"
                    intact[p] = False
                assert np.all(out[intact] == info["colors"][owner]), "Intact island changed"
            assert np.array_equal(solve(g), out)
            if split == "train":
                train_infos.append(info)
                train_diags.append(_diagnostics(g, info))
    assert set().union(*(d["sides"] for d in train_diags)) == set(SIDES)
    assert set().union(*(d["kinds"] for d in train_diags)) == {"tab", "notch"}
    assert any(d["source_count"] == 2 and d["tie_cells"] > 0
               and {1, 2, 3, 4} <= d["levels"] for d in train_diags)
    assert any(d["same_owner_tie"] and d["repeated_side"] for d in train_diags)
    assert len({info["colors"] for info in train_infos}) >= 2
    for kwargs in ({"markers_only": True}, {"exchange": False}, {"metric": "manhattan"},
                   {"actual_only": True}, {"outward_only": True}, {"keep_ties": True}):
        assert any(not np.array_equal(_render(np.asarray(pair["input"]), info, **kwargs), pair["output"])
                   for pair, info in zip(puzzle["train"], train_infos)), f"Missing evidence against {kwargs}"


def _puzzle_key(puzzle):
    return tuple((split, tuple((np.asarray(p["input"]).shape,
                               np.asarray(p["input"]).tobytes(), np.asarray(p["output"]).tobytes())
                              for p in puzzle[split])) for split in ("train", "test"))


def self_test():
    """A bounded 42-seed semantic/diversity suite, including all legal counts."""
    shapes, rooms, structures, palettes, source_counts, defect_types = set(), set(), set(), set(), set(), set()
    island_shapes, puzzle_keys, shifts = set(), set(), set()
    ties = repeated = near_thirty = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        puzzle_keys.add(_puzzle_key(puzzle))
        train_dims, test_dims, test_sources = [], [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, out = pair["input"], pair["output"]
                info, diag = _parse(g), _diagnostics(g, _parse(g))
                shapes.add(g.shape)
                t, b, l, r = info["bounds"]
                room = (b - t - 1, r - l - 1)
                rooms.add(room)
                (train_dims if split == "train" else test_dims).append(room)
                if split == "test":
                    test_sources.append(diag["source_count"])
                island_shapes.update((z - a + 1, d - c + 1) for a, z, c, d in info["rects"])
                source_counts.add(diag["source_count"])
                palettes.add(tuple(sorted(int(v) for v in np.unique(g))))
                role_map = {info["background"]: 0, info["frame"]: 1,
                            info["colors"][0]: 2, info["colors"][1]: 3}
                normalized = np.array([role_map[int(c)] for c in g.flat], dtype=np.int8)
                structures.add((g.shape, normalized.tobytes()))
                defect_types.update((v[0], v[1]) for local in info["defects"] for v in local)
                ties += diag["tie_cells"] > 0
                repeated += diag["repeated_side"]
                near_thirty += max(g.shape) == 30
                # Both representations of any defect must be exactly equivalent.
                flipped = g.copy()
                for owner, local in enumerate(info["defects"]):
                    for _, kind, p, q in local:
                        replacement = info["colors"][owner] if kind == "notch" else info["background"]
                        flipped[p] = flipped[q] = replacement
                assert np.array_equal(solve(flipped), out)
                if seed < 6:
                    permutation = np.random.default_rng(seed + 100).permutation(10)
                    assert np.array_equal(solve(permutation[g]), permutation[out])
                    assert np.array_equal(solve(g.T), out.T)
                    assert np.array_equal(solve(g[::-1, :]), out[::-1, :])
        if all(min(v) >= 22 for v in train_dims) and all(max(v) <= 18 for v in test_dims):
            shifts.add("compression")
        if all(max(v) / min(v) > 1.5 for v in test_dims):
            shifts.add("aspect")
        if min(test_sources) >= 8:
            shifts.add("multiplicity")
    assert len(puzzle_keys) == 42
    assert len(shapes) >= 50 and len(rooms) >= 50 and len(structures) >= 180
    assert len(palettes) >= 35 and len(island_shapes) >= 30
    assert source_counts == set(range(2, 11))
    assert len(defect_types) == 8 and shifts == {"compression", "aspect", "multiplicity"}
    assert ties >= 150 and repeated >= 50 and near_thirty >= 3
    legal_calls = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(42 + nt * 10 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
            legal_calls += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (None, 1), (4, None), (True, 1), (4, False), (4.0, 1), (4, 1.0)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzle_keys), "input_shapes": len(shapes),
            "room_shapes": len(rooms), "island_shapes": len(island_shapes),
            "color_normalized_inputs": len(structures), "palettes": len(palettes),
            "source_counts": sorted(source_counts), "side_polarity_combinations": len(defect_types),
            "pairs_with_conflicting_ties": ties, "pairs_with_repeated_sides": repeated,
            "pairs_reaching_30": near_thirty, "shift_modes": sorted(shifts),
            "legal_count_calls": legal_calls}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
