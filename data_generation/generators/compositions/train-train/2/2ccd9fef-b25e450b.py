"""Growing sweepers: extrapolate moving brushes and their causal, cumulative wakes.

Only NumPy and the standard library are used. No parent code or saved examples
are loaded. Generation renders complete histories; solve updates the last visible
state using parameters recovered exclusively from the input.
"""
from __future__ import annotations

import json
import numpy as np


HOST_SEEDS = (0,115530366,402065511,597916983,299921024,996108822,322545333,
              895925846,865834815,620342439,497015128,385656957,890906654,
              520347652,491829613,438374906,829185868,617275174,883929731,
              328336939,623293332,2171448,24300405,673934279,262876673,
              295077866,229239266,857378961,676697957,562583896,59117630,
              759013969,997798106,209294491,686410504,436716396,658679393,
              166879346,496774461,659092795,540909088,14295815)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=False)


def _mask(box, shape):
    z = np.zeros(shape, dtype=bool)
    r0, r1, c0, c1 = map(int, box)
    z[r0:r1, c0:c1] = True
    return z


def _box_at(brush, t):
    return tuple(int(a + t * b) for a, b in zip(brush["start"], brush["delta"]))


def _stroke_box(old, new, axis):
    # Reshape at the OLD position, then move: new teeth do not clean the
    # earlier part of the journey. All intervals are half open.
    box = list(new)
    box[2 * axis] = min(old[2 * axis], new[2 * axis])
    box[2 * axis + 1] = max(old[2 * axis + 1], new[2 * axis + 1])
    return tuple(box)


def _history(brush, t, shape):
    cleaned = _mask(_box_at(brush, 0), shape)
    for j in range(1, t + 1):
        cleaned |= _mask(_stroke_box(_box_at(brush, j - 1),
                                     _box_at(brush, j), brush["axis"]), shape)
    return cleaned


def _masks(brushes, t, shape):
    histories = [_history(b, t, shape) for b in brushes]
    wake = np.logical_or.reduce(histories)
    active = np.logical_or.reduce([_mask(_box_at(b, t), shape) for b in brushes])
    return wake, active, histories


def _render(base, carrier, brushes, t):
    """Direct history rendering, used to construct labels and validate history."""
    out = base.copy()
    wake, _, _ = _masks(brushes, t, base.shape)
    out[wake] = carrier
    for brush in brushes:
        out[_mask(_box_at(brush, t), base.shape)] = brush["color"]
    return out


def _split_input(grid):
    candidates = []
    for color in np.unique(grid):
        for axis in (0, 1):
            lines = np.flatnonzero(np.all(grid == color, axis=1 - axis))
            if len(lines) not in (2, 3):
                continue
            # The delimiter color occurs nowhere inside any panel.
            if np.count_nonzero(grid == color) != len(lines) * grid.shape[1 - axis]:
                continue
            starts = [0] + [int(k) + 1 for k in lines]
            ends = [int(k) for k in lines] + [grid.shape[axis]]
            lengths = [b - a for a, b in zip(starts, ends)]
            if len(set(lengths)) != 1 or min(lengths) < 5:
                continue
            panels = [grid[a:b, :] if axis == 0 else grid[:, a:b]
                      for a, b in zip(starts, ends)]
            if min(panels[0].shape) < 5 or len(np.unique(panels[-1])) != 2:
                continue
            corners = [int(p[r, c]) for p in panels
                       for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
            if len(set(corners)) != 1:
                continue
            candidates.append((panels, axis, int(color), corners[0]))
    if len(candidates) != 1:
        raise ValueError("the film strip must have one unambiguous delimiter")
    return candidates[0]


def _rectangle(grid, color):
    cells = np.argwhere(grid == color)
    if not len(cells):
        raise ValueError("a brush must appear in every observed frame")
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0) + 1
    if len(cells) != (r1 - r0) * (c1 - c0):
        raise ValueError("each brush color must form one solid rectangle")
    return tuple(map(int, (r0, r1, c0, c1)))


def _decode(input_grid):
    grid = _grid(input_grid)
    panels, layout, separator, carrier = _split_input(grid)
    base = panels[-1]
    field_colors = set(map(int, np.unique(base)))
    texture = next(c for c in field_colors if c != carrier)
    colors = sorted(set(map(int, np.unique(grid))) - field_colors - {separator})
    if not 1 <= len(colors) <= 4:
        raise ValueError("there must be one to four identifiable brushes")
    h, w = base.shape
    horizon = len(panels) - 1
    brushes = []
    for color in colors:
        boxes = [_rectangle(p, color) for p in panels[:-1]]
        first = boxes[0]
        edges = (first[0] == 0, first[1] == h,
                 first[2] == 0, first[3] == w)
        if sum(edges) != 1:
            raise ValueError("the launch rectangle must touch exactly one edge")
        side = edges.index(True)
        axis = side // 2
        delta = tuple(b - a for a, b in zip(boxes[0], boxes[1]))
        speed = delta[2 * axis]
        if delta[2 * axis + 1] != speed or speed * (1 if side % 2 == 0 else -1) <= 0:
            raise ValueError("brush depth is fixed and motion goes inward")
        brush = {"start": first, "delta": delta, "axis": axis, "color": color}
        if any(_box_at(brush, t) != box for t, box in enumerate(boxes)):
            raise ValueError("rectangle endpoints must change at constant rates")
        da, db = delta[2 * (1 - axis):2 * (1 - axis) + 2]
        if not (da == db or da <= 0 <= db):
            raise ValueError("a brush either drifts rigidly or grows from its ends")
        for t in range(horizon + 1):
            r0, r1, c0, c1 = _box_at(brush, t)
            if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
                raise ValueError("the predicted brush must fit; there is no clipping")
            transverse = (c0, c1, w) if axis == 0 else (r0, r1, h)
            if not (1 <= transverse[0] < transverse[1] <= transverse[2] - 1):
                raise ValueError("brushes keep a margin from their transverse edges")
        brushes.append(brush)
    for t in range(horizon + 1):
        occupancy = sum((_mask(_box_at(b, t), base.shape).astype(int)
                         for b in brushes), np.zeros(base.shape, dtype=int))
        if np.any(occupancy > 1):
            raise ValueError("simultaneous brush footprints must be disjoint")
    return {"panels": panels, "base": base, "layout": layout,
            "separator": separator, "carrier": carrier, "texture": texture,
            "brushes": brushes, "horizon": horizon}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer velocity and changing endpoints, then extend the visible clean wake."""
    d = _decode(input_grid)
    out = d["panels"][-2].copy()
    brushes, t = d["brushes"], d["horizon"]
    # Old footprints have already been swept; removing them reveals carrier.
    for brush in brushes:
        out[out == brush["color"]] = d["carrier"]
    for brush in brushes:
        old, new = _box_at(brush, t - 1), _box_at(brush, t)
        out[_mask(_stroke_box(old, new, brush["axis"]), out.shape)] = d["carrier"]
    # Simultaneous destination priority, including crossings with other wakes.
    for brush in brushes:
        out[_mask(_box_at(brush, t), out.shape)] = brush["color"]
    return out


def _sample_brush(rng, shape, horizon, kind):
    h, w = shape
    side = int(rng.integers(4))
    axis = side // 2
    length, transverse = (h, w) if axis == 0 else (w, h)
    depth = int(rng.integers(1, min(3, length - horizon) + 1))
    speeds = [v for v in (1, 2, 3) if depth + v * horizon <= length]
    if not speeds:
        return None
    speed = int(rng.choice(speeds))
    if kind == "growth":
        rate = int(rng.choice((1, 1, 1, 2)))
        da, db = ((0, rate), (-rate, 0), (-1, 1))[int(rng.integers(3))]
    elif kind == "drift":
        da = db = int(rng.choice((-2, -1, -1, 1, 1, 2)))
    else:
        da = db = 0
    intervals = [(a, b) for a in range(1, transverse - 1)
                 for b in range(a + 1, transverse)
                 if all(1 <= a + da*t < b + db*t <= transverse - 1
                        for t in range(horizon + 1))]
    if not intervals:
        return None
    a, b = intervals[int(rng.integers(len(intervals)))]
    p = 0 if side % 2 == 0 else length - depth
    v = speed if side % 2 == 0 else -speed
    first, delta = (p, p + depth, a, b), (v, v, da, db)
    if axis == 1:
        first, delta = first[2:] + first[:2], delta[2:] + delta[:2]
    return {"start": first, "delta": delta, "axis": axis}


def _kind(brush):
    a = 2 * (1 - brush["axis"])
    da, db = brush["delta"][a:a + 2]
    return "growth" if db > da else "drift" if da else "static"


def _features(brushes, t, shape):
    wake, active, histories = _masks(brushes, t, shape)
    previous, _, _ = _masks(brushes, t - 1, shape)
    retroactive = np.zeros(shape, dtype=bool)
    priority = False
    for i, brush in enumerate(brushes):
        retroactive |= _mask(_stroke_box(_box_at(brush, 0),
                                         _box_at(brush, t), brush["axis"]), shape)
        for j, other in enumerate(brushes):
            if i != j and brush["axis"] != other["axis"]:
                priority |= bool(np.any(_mask(_box_at(brush, t), shape) & histories[j]))
    corners = np.zeros(shape, dtype=bool)
    corners[0, 0] = corners[0, -1] = corners[-1, 0] = corners[-1, -1] = True
    return {"wake": wake, "active": active, "previous": previous,
            "new_clean": wake & ~previous & ~active & ~corners,
            "lag": retroactive & ~wake & ~active & ~corners,
            "priority": priority}


def _texture_mask(rng, shape):
    h, w = shape
    mode = int(rng.integers(5))
    rr, cc = np.indices(shape)
    if mode == 0:
        z = rng.random(shape) < float(rng.uniform(0.04, 0.16))
    elif mode == 1:
        z = rng.random(shape) < float(rng.uniform(0.45, 0.78))
    elif mode == 2:
        period = int(rng.integers(2, 5))
        z = ((rr + int(rng.integers(1, 4)) * cc) % period == 0)
    elif mode == 3:
        z = rng.random(shape) < 0.08
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            z[max(0, r-1):min(h, r+3), max(0, c-1):min(w, c+3)] = True
    else:
        z = rng.random(shape) < float(rng.uniform(0.18, 0.40))
    z[0, 0] = z[0, -1] = z[-1, 0] = z[-1, -1] = False
    return z


def _make_pair(rng, kinds, layout, horizon, require_priority=False):
    # Rejection concerns only visible geometric compatibility and explicit
    # contrast witnesses, never a template chosen from the answer dimensions.
    for _ in range(400):
        short = int(rng.integers(5, (30 - horizon) // (horizon + 1) + 1))
        long = int(rng.integers(8, 26))
        shape = (short, long) if layout == 0 else (long, short)
        brushes = []
        occupied = [np.zeros(shape, dtype=bool) for _ in range(horizon + 1)]
        for kind in kinds:
            for _ in range(60):
                brush = _sample_brush(rng, shape, horizon, kind)
                if brush is not None and all(not np.any(
                    _mask(_box_at(brush, t), shape) & occupied[t])
                    for t in range(horizon + 1)):
                    brushes.append(brush)
                    for t in range(horizon + 1):
                        occupied[t] |= _mask(_box_at(brush, t), shape)
                    break
            else:
                break
        if len(brushes) != len(kinds):
            continue
        f = _features(brushes, horizon, shape)
        if not np.any(f["new_clean"]) or not np.any(f["lag"]):
            continue
        if require_priority and not f["priority"]:
            continue
        palette = list(map(int, rng.permutation(10)))
        carrier, texture, separator = palette[:3]
        for brush, color in zip(brushes, palette[3:]):
            brush["color"] = color
        textured = _texture_mask(rng, shape)
        # Two witnesses: a new erasure, and old texture a retroactive full-width
        # sweep would wrongly erase. These witnesses work even at counts 2/1.
        for eligible in (f["new_clean"], f["lag"]):
            choices = np.argwhere(eligible)
            r, c = choices[int(rng.integers(len(choices)))]
            textured[r, c] = True
        base = np.full(shape, carrier, dtype=np.int8)
        base[textured] = texture
        frames = [_render(base, carrier, brushes, t) for t in range(horizon)] + [base]
        delimiter_shape = (1, shape[1]) if layout == 0 else (shape[0], 1)
        delimiter = np.full(delimiter_shape, separator, dtype=np.int8)
        pieces = []
        for frame in frames:
            if pieces:
                pieces.append(delimiter)
            pieces.append(frame)
        pair = {"input": np.concatenate(pieces, axis=layout),
                "output": _render(base, carrier, brushes, horizon)}
        # Reject accidental delimiter ambiguity from unusual carrier/texture fields.
        try:
            _decode(pair["input"])
        except ValueError:
            continue
        return pair
    raise RuntimeError("bounded sweep-history construction exhausted")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for x, lo, hi, name in ((num_train, 2, 10, "num_train"),
                            (num_test, 1, 4, "num_test")):
        if (isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                or not lo <= int(x) <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    preferred = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train" and i == 0:
                kinds = ["growth"] + ["static"] * int(rng.integers(2))
                layout, horizon, priority = preferred, 2, False
            elif split == "train" and i == 1:
                kinds = ["drift"] + ["static"] * int(rng.integers(1, 3))
                layout, horizon, priority = 1 - preferred, int(rng.integers(2, 4)), True
            elif split == "train":
                kinds = [str(rng.choice(("growth", "drift")))]
                kinds += [str(rng.choice(("growth", "drift", "static")))
                          for _ in range(int(rng.integers(3)))]
                layout, horizon, priority = int(rng.integers(2)), int(rng.integers(2, 4)), False
            else:
                kinds = ["growth", "drift"]
                kinds += [str(rng.choice(("growth", "drift", "static")))
                          for _ in range(int(rng.integers(3)))]
                layout, horizon, priority = int(rng.integers(2)), int(rng.integers(2, 4)), False
            for _ in range(20):
                pair = _make_pair(rng, kinds, layout, horizon, priority)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _frozen_profile_prediction(d):
    """Ablation: use the visible velocity but keep every last brush width/location."""
    out = d["panels"][-2].copy()
    t, shape = d["horizon"], d["base"].shape
    boxes = []
    for b in d["brushes"]:
        out[out == b["color"]] = d["carrier"]
        old = _box_at(b, t - 1)
        new = list(old)
        for k in (2 * b["axis"], 2 * b["axis"] + 1):
            new[k] += b["delta"][k]
        boxes.append(new)
        out[_mask(_stroke_box(old, new, b["axis"]), shape)] = d["carrier"]
    for b, box in zip(d["brushes"], boxes):
        out[_mask(box, shape)] = b["color"]
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_kinds, train_layouts = set(), set(), set()
    priority_training = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _decode(inp)
            brushes, t, base = d["brushes"], d["horizon"], d["base"]
            assert out.shape == base.shape
            for j, frame in enumerate(d["panels"][:-1]):
                assert np.array_equal(frame, _render(base, d["carrier"], brushes, j))
            assert np.array_equal(out, _render(base, d["carrier"], brushes, t))
            assert np.array_equal(out, solve(inp))
            f = _features(brushes, t, base.shape)
            assert np.array_equal(out[~f["wake"]], base[~f["wake"]])
            assert np.all(out[f["wake"] & ~f["active"]] == d["carrier"])
            for b in brushes:
                assert np.all(out[_mask(_box_at(b, t), base.shape)] == b["color"])
            assert np.any((base == d["texture"]) & f["new_clean"])
            assert np.any((out == d["texture"]) & f["lag"])
            assert not np.array_equal(out, _frozen_profile_prediction(d))
            # Omitting cleaning, even while predicting all brush endpoints, fails.
            stamp_only = d["panels"][-2].copy()
            for b in brushes:
                stamp_only[stamp_only == b["color"]] = d["carrier"]
            for b in brushes:
                stamp_only[_mask(_box_at(b, t), base.shape)] = b["color"]
            assert not np.array_equal(out, stamp_only)
            kinds = {_kind(b) for b in brushes}
            if split == "train":
                train_kinds |= kinds
                train_layouts.add(d["layout"])
                priority_training |= f["priority"]
            else:
                assert {"growth", "drift"} <= kinds
    assert {"growth", "drift"} <= train_kinds
    assert train_layouts == {0, 1}
    assert priority_training


def self_test():
    """Bounded 42-seed semantic/variation audit; returns descriptive measurements."""
    # Hand-authored causal-wake regressions. Expected grids do not use any
    # generation, box, stroke, history, or rendering helper.
    def digits(rows):
        return np.array([[int(x) for x in row] for row in rows.split()], dtype=np.int8)
    raw = digits("0555550 5555555 5555555 5555555 5555555 5555555 0555550")
    delimiter = np.full((1, 7), 9, dtype=np.int8)
    growth0 = digits("0555550 2555555 5555555 5555555 5555555 5555555 0555550")
    growth1 = digits("0555550 0255555 0255555 5555555 5555555 5555555 0555550")
    growth2 = digits("0555550 0025555 0025555 5025555 5555555 5555555 0555550")
    film = np.concatenate((growth0, delimiter, growth1, delimiter, raw))
    assert np.array_equal(solve(film), growth2)
    cross0 = digits("0553550 5555555 2555555 5555555 5555555 5555555 0555550")
    cross1 = digits("0550550 5553555 0025555 5555555 5555555 5555555 0555550")
    cross2 = digits("0550550 5550555 0003255 5555555 5555555 5555555 0555550")
    film = np.concatenate((cross0, delimiter, cross1, delimiter, raw))
    assert np.array_equal(solve(film), cross2)
    measurements = {name: set() for name in ("shapes", "palettes", "counts", "speeds",
                    "depths", "endpoint_rates", "sides", "layouts", "horizons")}
    signatures, structures = set(), set()
    priority = {"train": 0, "test": 0}
    pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(x["input"]) for s in ("train", "test") for x in p[s]))
        for split in ("train", "test"):
            for pair in p[split]:
                pairs += 1
                d = _decode(pair["input"])
                shape, t, brushes = d["base"].shape, d["horizon"], d["brushes"]
                measurements["shapes"].add(shape)
                measurements["palettes"].add((d["carrier"], d["texture"], d["separator"]))
                measurements["counts"].add((split, len(brushes)))
                measurements["layouts"].add((split, d["layout"]))
                measurements["horizons"].add((split, t))
                priority[split] += int(_features(brushes, t, shape)["priority"])
                for b in brushes:
                    a, v = b["axis"], b["delta"][2 * b["axis"]]
                    measurements["speeds"].add(abs(v))
                    measurements["depths"].add(b["start"][2*a+1] - b["start"][2*a])
                    measurements["endpoint_rates"].add(b["delta"][2*(1-a):2*(1-a)+2])
                    measurements["sides"].add((a, 1 if v > 0 else -1))
                roles = {}
                normalized = bytes(roles.setdefault(int(x), len(roles))
                                   for x in np.asarray(pair["input"]).ravel())
                structures.add((pair["input"].shape, normalized))
    assert len(signatures) == 42
    assert len(structures) >= 190
    assert len(measurements["shapes"]) >= 35
    assert len(measurements["palettes"]) >= 80
    assert measurements["speeds"] == {1, 2, 3}
    assert measurements["depths"] == {1, 2, 3}
    assert len(measurements["sides"]) == 4
    assert len(measurements["layouts"]) == 4
    assert len(measurements["horizons"]) == 4
    assert {("test", 2), ("test", 3), ("test", 4)} <= measurements["counts"]
    assert priority["train"] >= 42 and priority["test"] >= 4
    assert len(measurements["endpoint_rates"]) >= 8
    # Exercise every explicit legal count, deterministic regeneration, and types.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(777 + 11*a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(0, 10, 4), generate_puzzle(0, 10, 4)
    assert all(np.array_equal(x[k], y[k]) for s in ("train", "test")
               for x, y in zip(first[s], second[s]) for k in ("input", "output"))
    # Input-only inference commutes with transposition and arbitrary palette swaps.
    perm = np.array([9,7,5,3,1,8,6,4,2,0], dtype=np.int8)
    for pair in first["train"][:2] + first["test"][:2]:
        assert np.array_equal(solve(pair["input"].T), pair["output"].T)
        assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "normalized_structures": len(structures), "priority_pairs": priority,
            **{k: sorted(v) for k, v in measurements.items()}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
