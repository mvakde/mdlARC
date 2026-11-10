"""Camouflaged mirror stamps: recover, complete by reflection, move once.

Only NumPy and the standard library are used.  Neither solve nor its decoder
reads examples, files, random state, or construction metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _wallpaper_info(a):
    """Check a cyclic successor down rows; return its cycle and least period."""
    colors = tuple(int(x) for x in np.unique(a))
    successor = {}
    for x, y in np.unique(np.stack((a[:-1].ravel(), a[1:].ravel()), axis=1), axis=0):
        x, y = int(x), int(y)
        if x in successor and successor[x] != y:
            return None
        successor[x] = y
    if set(successor) != set(colors) or set(successor.values()) != set(colors):
        return None
    cycle = [colors[0]]
    for _ in range(len(colors) - 1):
        cycle.append(successor[cycle[-1]])
    if len(set(cycle)) != len(colors) or successor[cycle[-1]] != cycle[0]:
        return None
    period = next(p for p in range(1, a.shape[1] + 1)
                  if p == a.shape[1] or np.array_equal(a[:, :-p], a[:, p:]))
    return tuple(cycle), period


def _reflect(mask, axis):
    rows, cols = np.nonzero(mask)
    mirrors = 2 * axis - cols
    if np.any(mirrors < 0) or np.any(mirrors >= mask.shape[1]):
        return None
    result = mask.copy()
    result[rows, mirrors] = True
    return result


def _decode(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    colors, counts = np.unique(a, return_counts=True)
    singletons = colors[counts == 1]
    if len(singletons) != 1:
        raise ValueError("one unique top-edge axis marker is required")
    marker = int(singletons[0])
    mr, axis = (int(x) for x in np.argwhere(a == marker)[0])
    if mr != 0:
        raise ValueError("axis marker must be on the top edge")
    palette = tuple(int(x) for x in colors if x != marker)
    k = len(palette)
    if not 3 <= k <= 6 or h < 2 * k:
        raise ValueError("wallpaper needs 3..6 colors and at least two vertical cycles")
    rr, cc = np.indices(a.shape)
    solutions = {}
    # Mask identity and horizontal period are inferred, not supplied by a seed.
    # Each non-mask lattice phase has an intact witness. A phase whose only
    # observations are ink is legitimately the ink color in the wallpaper.
    for ink in palette:
        observed = (a != marker) & (a != ink)
        values = a[observed]
        for period in range(1, w // 2 + 1):
            labels = ((rr % k) * period + cc % period)
            groups = labels[observed]
            low = np.full(k * period, 10, dtype=np.int8)
            high = np.full(k * period, -1, dtype=np.int8)
            np.minimum.at(low, groups, values)
            np.maximum.at(high, groups, values)
            present = high >= 0
            if np.any(low[present] != high[present]):
                continue
            tile = np.where(present, high, ink)
            field = tile[labels].astype(np.int8)
            if tuple(int(x) for x in np.unique(field)) != palette:
                continue
            info = _wallpaper_info(field)
            if info is None or info[1] > w // 2:
                continue
            defects = (a != field) & (a != marker)
            if not np.any(defects) or np.any(a[defects] != ink):
                continue
            support = _reflect(defects, axis)
            if (support is None or np.any(support[0]) or np.any(support[-1])
                    or np.any(a[support] != ink) or np.mean(support) > .35):
                continue
            # The sole extra-colored cell is a stationary registration pin.
            rebuilt = field.copy()
            rebuilt[support] = ink
            rebuilt[0, axis] = marker
            if not np.array_equal(rebuilt, a):
                continue
            identity = (_key(field), ink, _key(support))
            solutions[identity] = {"field": field, "ink": ink, "marker": marker,
                                   "axis": axis, "support": support,
                                   "defects": defects, "cycle": info[0],
                                   "period": info[1]}
    if len(solutions) != 1:
        raise ValueError(f"expected one supported mirror-stamp interpretation, got {len(solutions)}")
    return next(iter(solutions.values()))


def _render(model, support=None, distance=1):
    output = model["field"].copy()
    rr, cc = np.nonzero(model["support"] if support is None else support)
    rr = rr + distance
    if np.any(rr >= output.shape[0]):
        raise ValueError("stamp would leave the canvas")
    output[rr, cc] = model["ink"]
    output[0, model["axis"]] = model["marker"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the wallpaper and symmetric ink footprint, then move ink down once."""
    return _render(_decode(input_grid))


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    return int(value)


def _connected(mask):
    cells = set(zip(*np.nonzero(mask)))
    if not cells:
        return False
    seen = {min(cells)}
    frontier = list(seen)
    while frontier:
        r, c = frontier.pop()
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if q in cells and q not in seen:
                seen.add(q)
                frontier.append(q)
    return seen == cells


def _crop(mask):
    rr, cc = np.nonzero(mask)
    return mask[rr.min():rr.max()+1, cc.min():cc.max()+1]


def _has_hole(mask):
    """Flood outside the cropped footprint; test for an enclosed empty cell."""
    a = np.pad(_crop(mask), 1)
    outside = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (0 <= nr < a.shape[0] and 0 <= nc < a.shape[1]
                    and not a[nr, nc] and (nr, nc) not in outside):
                outside.add((nr, nc))
                stack.append((nr, nc))
    return len(outside) < int(np.sum(~a))


def _sample_word(rng, k, period):
    for _ in range(80):
        word = rng.integers(0, k, size=period)
        if len(set(int(x) for x in word)) < 2:
            continue
        if all(not np.array_equal(np.tile(word, 2)[:-p], np.tile(word, 2)[p:])
               for p in range(1, period)):
            return word
    raise RuntimeError("could not sample a fundamental word")


def _shape_mask(rng, kind, bh, bw, allowed):
    center = bw // 2
    if kind == "grow":
        # Grow reflection orbits, so the result stays connected and symmetric.
        starts = [r for r in range(bh) if allowed[r, center]]
        if not starts:
            return None
        shape = np.zeros((bh, bw), dtype=bool)
        shape[int(rng.choice(starts)), center] = True
        target = int(rng.integers(max(8, bh + bw), min(bh*bw, 3*(bh+bw)) + 1))
        while int(np.sum(shape)) < target:
            choices = []
            for r in range(bh):
                for c in range(center + 1):
                    mate = bw - 1 - c
                    if shape[r, c] or not allowed[r, c] or not allowed[r, mate]:
                        continue
                    if any(0 <= nr < bh and 0 <= nc < bw and shape[nr, nc]
                           for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))):
                        choices.append((r, c))
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            shape[r, c] = shape[r, bw-1-c] = True
        return shape
    shape = np.zeros((bh, bw), dtype=bool)
    if kind in ("bar", "block"):
        shape[:] = True
    elif kind == "ring":
        shape[0] = shape[-1] = True
        shape[:, 0] = shape[:, -1] = True
    elif kind == "cup":
        shape[-1] = True
        shape[:, 0] = shape[:, -1] = True
    elif kind == "comb":
        shape[-1] = True
        shape[:, 0] = shape[:, center] = shape[:, -1] = True
    else:
        raise ValueError("unknown footprint kind")
    return shape if np.all(allowed[shape]) else None


def _make_pair(rng, cycle, marker, regime, split, index):
    k = len(cycle)
    ink = int(cycle[0])
    anchor = split == "train" and index == 0
    hole_demo = split == "train" and index == 1
    for _ in range(500):
        if anchor:
            kind, bh, bw = "bar", 1, int(rng.choice([5, 7, 9]))
        elif hole_demo:
            kind, bh, bw = "ring", int(rng.integers(3, 6)), int(rng.choice([5, 7, 9]))
        elif split == "train":
            kind = str(rng.choice(["ring", "cup", "comb", "block", "grow"]))
            if kind == "comb" and k == 3:
                kind = "cup"
            bh = int(rng.integers(2 if kind == "block" else 3, min(5, k-1) + 1)) if kind in ("block", "comb") else int(rng.integers(3, 6))
            bw = int(rng.choice([5, 7, 9]))
        else:
            kind = str(rng.choice(["ring", "cup", "grow"]))
            if regime == "tall":
                bh, bw = int(rng.integers(7, 12)), int(rng.choice([5, 7, 9]))
            elif regime == "wide":
                bh, bw = int(rng.integers(3, 6)), int(rng.choice([13, 15, 17]))
            else:
                bh, bw = int(rng.integers(7, 10)), int(rng.choice([11, 13]))
        period = int(rng.integers(3, 10 if split == "test" else 8))
        h = int(rng.integers(max(12, 2*k+2, bh+4), 31 if split == "test" else 29))
        w = int(rng.integers(max(12, 2*period, bw+2), 31 if split == "test" else 29))
        axis = int(rng.integers(bw//2, w-bw//2))
        max_top = h-bh-1-int(anchor)
        top = max_top if not anchor and rng.random() < .25 else int(rng.integers(1, max_top+1))
        left = axis-bw//2
        word = _sample_word(rng, k, period)
        phase = int(rng.integers(k))
        indices = (phase + np.arange(h)[:, None] + word[np.arange(w) % period]) % k
        field = np.asarray(cycle, dtype=np.int8)[indices]
        local = field[top:top+bh, left:left+bw]
        # No reflection orbit may be completely camouflaged. This is the
        # information needed for minimum symmetric completion to be unique.
        allowed = (local != ink) | (local[:, ::-1] != ink)
        shape = _shape_mask(rng, kind, bh, bw, allowed)
        if shape is None or not _connected(shape):
            continue
        actual_h, actual_w = _crop(shape).shape
        if split == "test" and not (actual_h > 5 or actual_w > 9):
            continue
        support = np.zeros((h, w), dtype=bool)
        support[top:top+bh, left:left+bw] = shape
        if np.mean(support) > .35:
            continue
        defects = support & (field != ink)
        hidden = support & (field == ink)
        if not np.any(hidden) or np.array_equal(support, defects):
            continue
        # Repetition witnesses are checked independently of the solver.
        witness_ok = True
        for r in range(k):
            for c in range(period):
                if field[r, c] != ink:
                    witness = (~support)[r::k, c::period].copy()
                    if r == 0 and axis % period == c:
                        witness[0, axis//period] = False
                    if not np.any(witness):
                        witness_ok = False
        if not witness_ok:
            continue
        a = field.copy()
        a[support] = ink
        a[0, axis] = marker
        model = {"field": field, "support": support, "defects": defects,
                 "ink": ink, "marker": marker, "axis": axis}
        b = _render(model)
        if hole_demo:
            rr, cc = np.nonzero(support)
            box = np.zeros_like(support)
            box[rr.min():rr.max()+1, cc.min():cc.max()+1] = True
            if np.array_equal(b, _render(model, box)):
                continue
        try:
            decoded = _decode(a)
        except ValueError:
            continue
        if (np.array_equal(decoded["field"], field)
                and np.array_equal(decoded["support"], support)):
            return {"input": a, "output": b}
    raise RuntimeError("bounded sampling exhausted for a mirror stamp")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    k = int(rng.integers(3, 7))
    palette = rng.choice(10, size=k+1, replace=False)
    cycle = tuple(int(x) for x in palette[:k])
    marker = int(palette[-1])
    regime = str(rng.choice(["tall", "wide", "both"]))
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            for _ in range(20):
                pair = _make_pair(rng, cycle, marker, regime, split, index)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, latent-free inference and family-specific evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, conventions = set(), set()
    training_models = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            model = _decode(a)
            f, s, d = model["field"], model["support"], model["defects"]
            m, axis = model["ink"], model["axis"]
            cycle = model["cycle"]
            conventions.add((model["marker"], m, cycle))
            assert np.array_equal(b, _render(model))
            assert not np.array_equal(a, b)
            assert np.array_equal(_reflect(s, axis), s)
            assert np.array_equal(_reflect(d, axis), s)
            assert _connected(s)
            assert np.any(s & ~d)  # Some source ink is genuinely camouflaged.
            assert np.any((f == m) & ~s)  # Natural same-color wallpaper outside it.
            assert not np.any(s[0]) and not np.any(s[-1])
            assert b[0, axis] == model["marker"]
            assert np.array_equal(a[0], b[0])
            # Each non-ink phase must have independent intact evidence.
            k, p = len(cycle), model["period"]
            for r in range(k):
                for c in range(p):
                    if f[r, c] != m:
                        assert np.any((a[r::k, c::p] == f[r, c]))
            clean = f.copy()
            clean[0, axis] = model["marker"]
            assert not np.array_equal(b, clean)  # Omitting A fails.
            assert not np.array_equal(b, _render(model, d))  # No reflection fails.
            # Omitting B: treat every ink-colored cell as foreground (oracle
            # wallpaper underneath, clipping just this deliberately wrong rule).
            all_ink = a == m
            all_ink[-1] = False
            assert not np.array_equal(b, _render(model, all_ink))
            if split == "train":
                assert _crop(s).shape[0] <= 5 and _crop(s).shape[1] <= 9
                training_models.append(model)
            else:
                sh, sw = _crop(s).shape
                assert sh > 5 or sw > 9
    assert len(conventions) == 1
    first = training_models[0]
    assert _crop(first["support"]).shape[0] == 1
    assert not np.any(_render(first)[-1] != first["field"][-1])
    distance = first["field"].shape[0]-1-int(np.nonzero(first["support"])[0].max())
    assert distance > 1 and not np.array_equal(_render(first), _render(first, distance=distance))
    second = training_models[1]
    assert _has_hole(second["support"])
    rr, cc = np.nonzero(second["support"])
    box = np.zeros_like(second["support"])
    box[rr.min():rr.max()+1, cc.min():cc.max()+1] = True
    assert not np.array_equal(_render(second), _render(second, box))


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded coverage over 42 seeds, mutation checks, all count combinations."""
    signatures, shapes, periods, inks, pins, sizes, footprints, axes = (set() for _ in range(8))
    measures = {name: 0 for name in ("hole", "edge", "bottom", "hidden",
                                    "visible_asymmetric", "partial_h", "partial_w")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                model = _decode(a)
                s, d = model["support"], model["defects"]
                shapes.add(a.shape)
                periods.add(model["period"])
                inks.add(model["ink"])
                pins.add(model["marker"])
                sizes.add(len(model["cycle"]))
                crop = _crop(s)
                footprints.add(_key(crop))
                axes.add(model["axis"])
                measures["hole"] += _has_hole(s)
                measures["edge"] += bool(np.any(s[:, 0]) or np.any(s[:, -1]))
                measures["bottom"] += bool(np.any(s[-2]))
                measures["hidden"] += int(np.sum(s & ~d))
                measures["visible_asymmetric"] += not np.array_equal(_reflect(d, model["axis"]), d)
                measures["partial_h"] += a.shape[0] % len(model["cycle"]) != 0
                measures["partial_w"] += a.shape[1] % model["period"] != 0
                assert np.array_equal(solve(a.copy()), b)
                assert np.array_equal(solve(a.tolist()), b)
        if seed == 0:
            assert _signature(puzzle) == _signature(generate_puzzle(0))
            damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]]
                       for s in ("train", "test")}
            damaged["train"][0]["output"][0, 0] = (int(damaged["train"][0]["output"][0, 0])+1) % 10
            try:
                validate_generated_puzzle(damaged)
            except AssertionError:
                pass
            else:
                raise AssertionError("validator accepted a wrong output")
    assert len(signatures) == 42 and len(shapes) >= 50
    assert len(footprints) >= 50 and len(periods) >= 6 and sizes == {3, 4, 5, 6}
    assert len(inks) >= 8 and len(pins) >= 8 and len(axes) >= 8
    assert all(value > 0 for value in measures.values())
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(700 + 10*nt + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(shapes),
            "cropped_footprints": len(footprints), "periods": sorted(periods),
            "palette_sizes": sorted(sizes), "ink_colors": len(inks),
            "marker_colors": len(pins), "measures": measures}


if __name__ == "__main__":
    report = self_test()
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self_test.json").write_text(json.dumps(report, separators=(",", ":"))+"\n")
    print(json.dumps(report, separators=(",", ":")))
