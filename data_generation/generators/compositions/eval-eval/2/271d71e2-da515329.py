"""A shared unary clock drives four dock-seeking rectangular pinwheel fronts.

No parent module, stored puzzle, filesystem, or RNG is used by solve().
Generation and checking retain no execution traces or sample bank.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # U, R, D, L


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid size")
    _need(np.issubdtype(g.dtype, np.integer), "grid must have integer cells")
    _need(bool(np.all((0 <= g) & (g <= 9))), "invalid color")
    return g


def _key(g):
    g = np.asarray(g, dtype=np.uint8)
    return g.shape, g.tobytes()


def _counts(num_train, num_test):
    for x in (num_train, num_test):
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return int(num_train), int(num_test)


def _order(cy, cx, a, b, side):
    """Interior raster after rotating the notched side to face upward."""
    ys, xs = range(cy-a+1, cy+a), range(cx-b+1, cx+b)
    if side == 0:
        return [(y, x) for y in ys for x in xs]
    if side == 1:
        return [(y, x) for x in reversed(xs) for y in ys]
    if side == 2:
        return [(y, x) for y in reversed(ys) for x in reversed(xs)]
    return [(y, x) for x in xs for y in reversed(ys)]


def _notch(cy, cx, a, b, side):
    dy, dx = _DIRS[side]
    return cy + a*dy, cx + b*dx


def _segment_lengths(a, b, arm, count=4):
    dy, dx = _DIRS[arm]
    dy, dx = dx, -dy
    lengths = []
    for k in range(count):
        radius = (a if dy else b) + 2
        lengths.append(radius+2 if k == 0 else 2*radius+4*k)
        dy, dx = dx, -dy
    return lengths


def _paths(cy, cx, a, b, shape):
    """Infinite-plane paths, bounded only after no in-grid re-entry is possible.

    The first tangent reaches two cells beyond the neighboring seed tip.
    Every later side reaches two cells beyond the corresponding preceding
    winding. Four stages expand the enclosing rectangle by eight per side.
    The extra three cycles below cover the first asymmetric stage as well.
    Index 0 is the old tip; index t is its position after t clock ticks.
    """
    distance = max(cy, cx, shape[0]-1-cy, shape[1]-1-cx)
    stages = 4*(distance//8+3)
    paths = []
    for arm, (dy, dx) in enumerate(_DIRS):
        y, x = cy+(a+2)*dy, cx+(b+2)*dx
        path = [(y, x)]
        dy, dx = dx, -dy
        for length in _segment_lengths(a, b, arm, stages):
            for _ in range(length):
                y, x = y+dy, x+dx
                path.append((y, x))
            dy, dx = dx, -dy
        paths.append(path)
    return paths


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _base(shape, center, a, b, side, fuel, palette):
    bg, frame, live, marker = palette
    cy, cx = center
    g = np.full(shape, bg, dtype=np.int8)
    g[cy-a:cy+a+1, cx-b:cx+b+1] = frame
    order = _order(cy, cx, a, b, side)
    for p in order[len(order)-fuel:]:
        g[p] = live
    for dy, dx in _DIRS:
        for extra in (1, 2):
            g[cy+(a+extra)*dy, cx+(b+extra)*dx] = frame
    g[_notch(cy, cx, a, b, side)] = marker
    return g


def _candidate(g, marker, notch, side, a, b):
    """Explain every input cell using one housing, its spokes and four docks."""
    ny, nx = notch
    dy, dx = _DIRS[side]
    cy, cx = ny-a*dy, nx-b*dx
    h, w = g.shape
    if not (a+2 <= cy < h-a-2 and b+2 <= cx < w-b-2):
        return None
    t, z, l, r = cy-a, cy+a, cx-b, cx+b
    frame = int(g[t, l])
    if frame == marker:
        return None
    border = [(y, x) for y in range(t, z+1) for x in range(l, r+1)
              if y in (t, z) or x in (l, r)]
    if any(int(g[p]) != (marker if p == notch else frame) for p in border):
        return None
    order = _order(cy, cx, a, b, side)
    interior = [int(g[p]) for p in order]
    other = set(interior)-{frame}
    if len(other) > 1 or marker in other:
        return None
    live = next(iter(other)) if other else None
    fuel = sum(v != frame for v in interior)
    if interior != [frame]*(len(order)-fuel)+[live]*fuel:
        return None
    remaining_colors = set(map(int, np.unique(g)))-{frame, marker, live}
    if len(remaining_colors) != 1:
        return None
    bg = remaining_colors.pop()
    # For an exhausted counter its absent live color has no effect on solving.
    palette = (bg, frame, live if live is not None else bg, marker)
    expected = _base(g.shape, (cy, cx), a, b, side, fuel, palette)
    docks = set(map(tuple, np.argwhere(g == marker)))-{notch}
    if len(docks) != 4:
        return None
    for p in docks:
        if expected[p] != bg:
            return None
        expected[p] = marker
    if not np.array_equal(g, expected):
        return None
    paths = _paths(cy, cx, a, b, g.shape)
    distances = []
    for path in paths:
        found = [i for i, p in enumerate(path[1:], 1) if p in docks]
        if len(found) != 1:
            return None
        distances.append(found[0])
    if {paths[i][d] for i, d in enumerate(distances)} != docks:
        return None
    return dict(center=(cy, cx), a=a, b=b, side=side, fuel=fuel,
                palette=palette, order=order, paths=paths, distances=distances)


def _analyze(input_grid):
    g = _grid(input_grid)
    values, counts = np.unique(g, return_counts=True)
    answers = []
    # The four docks and one border notch share a color appearing five times.
    for color, count in zip(values, counts):
        if count != 5:
            continue
        marker = int(color)
        for raw in np.argwhere(g == marker):
            notch = tuple(map(int, raw))
            for side in range(4):
                for a in range(2, 6):
                    for b in range(2, 6):
                        result = _candidate(g, marker, notch, side, a, b)
                        if result is not None:
                            answers.append(result)
    _need(len(answers) == 1, "input needs one complete geometric explanation")
    return answers[0]


def _render_formula(g, info, ticks=None):
    out = g.copy()
    bg, frame, live, marker = info["palette"]
    if ticks is None:
        ticks = min(info["fuel"], max(info["distances"]))
    for arm, d in enumerate(info["distances"]):
        path = info["paths"][arm]
        for p in path[1:min(ticks, d)+1]:
            if _inside(p, out.shape) and out[p] == bg:
                out[p] = frame
    advanced = len(info["order"])-info["fuel"]
    for p in info["order"][advanced:advanced+ticks]:
        out[p] = frame
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    return _render_formula(g, _analyze(g))


def _simulate(g, info):
    """Unit-tick check of simultaneous motion and reservoir consumption.

    This shares the geometric path definition, so it is not an independent
    semantic solver. It does not use the min/max prediction formula.
    """
    out = g.copy()
    bg, frame, live, marker = info["palette"]
    indices = [0]*4
    pending = [p for p in info["order"] if int(g[p]) != frame]
    used = 0
    while pending and any(indices[i] < info["distances"][i] for i in range(4)):
        p = pending.pop(0)
        out[p] = frame
        used += 1
        for i, d in enumerate(info["distances"]):
            if indices[i] == d:
                continue
            indices[i] += 1
            p = info["paths"][i][indices[i]]
            if _inside(p, out.shape) and out[p] == bg:
                out[p] = frame
    return out, used, indices


def _make_pair(rng, kind, side, palette):
    # All demonstrations stop before the third corner; tests pass that corner.
    if kind in ("fuel", "dock", "test_fuel", "test_dock"):
        a, b = map(int, rng.integers(4, 6, size=2))
    else:
        a, b = map(int, rng.integers(2, 6, size=2))
    capacity = (2*a-1)*(2*b-1)
    cumulative = np.cumsum([_segment_lengths(a, b, arm) for arm in range(4)], axis=1)
    second = int(cumulative[:, 1].max())
    third_low, third_high = int(cumulative[:, 2].min()), int(cumulative[:, 2].max())
    fourth_low = int(cumulative[:, 3].min())
    is_test = kind.startswith("test_")
    margin = 8 if is_test else 6
    h = int(rng.integers(2*(a+margin)+1, 31))
    w = int(rng.integers(2*(b+margin)+1, 31))
    cy = int(rng.integers(a+margin, h-a-margin))
    cx = int(rng.integers(b+margin, w-b-margin))
    if is_test:
        lo = third_high+1
        hi = min(capacity-2, fourth_low-5)
    else:
        lo, hi = second+2, min(capacity-2, third_low-5)
    if kind in ("fuel", "test_fuel"):
        fuel = int(rng.integers(lo, hi+1))
        ceiling = fourth_low-3 if is_test else third_low-3
        long = int(rng.integers(fuel+2, min(ceiling, fuel+6)+1))
        distances = [long, int(rng.integers(fuel+1, min(ceiling, fuel+6)+1)),
                     int(rng.integers(1, max(2, fuel//2))),
                     int(rng.integers(max(1, fuel//2), fuel))]
    elif kind in ("dock", "test_dock"):
        last = int(rng.integers(lo, hi+1))
        scan_width = 2*b-1 if side in (0, 2) else 2*a-1
        gap = int(rng.integers(1, min(scan_width-1, capacity-1-last)+1))
        fuel = last+gap
        distances = [last]+[int(rng.integers((2*last+2)//3, last)) for _ in range(3)]
    else:
        fuel = int(rng.integers(0, capacity+1))
        ceiling = third_low-3
        distances = list(map(int, rng.integers(1, ceiling+1, size=4)))
    rng.shuffle(distances)
    g = _base((h, w), (cy, cx), a, b, side, fuel, palette)
    paths = _paths(cy, cx, a, b, g.shape)
    for i, d in enumerate(distances):
        p = paths[i][d]
        _need(_inside(p, g.shape), "construction put a dock outside the canvas")
        _need(int(g[p]) == palette[0], "dock overlaps the housing or a seed")
        g[p] = palette[3]
    info = dict(center=(cy, cx), a=a, b=b, side=side, fuel=fuel,
                palette=palette, order=_order(cy, cx, a, b, side),
                paths=paths, distances=distances)
    out, _, _ = _simulate(g, info)
    return {"input": g, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    common_palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    shared = bool(rng.integers(0, 3) == 0)
    sides = list(map(int, rng.permutation(4)))
    lessons = ["fuel", "dock"]+["extra"]*(ntrain-2)
    rng.shuffle(lessons)
    used = set()
    puzzle = {"train": [], "test": []}
    for split, count in (("train", ntrain), ("test", ntest)):
        for i in range(count):
            kind = lessons[i] if split == "train" else str(rng.choice(("test_fuel", "test_dock")))
            side = sides[i % 4] if split == "train" else int(rng.integers(4))
            for _ in range(32):
                palette = common_palette if shared else tuple(map(int, rng.choice(10, 4, replace=False)))
                pair = _make_pair(rng, kind, side, palette)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded distinct-input construction exhausted")
    return puzzle


def _features(g, info):
    ticks = min(info["fuel"], max(info["distances"]))
    ends = [min(ticks, d) for d in info["distances"]]
    turns = [sum(t >= n for n in np.cumsum(_segment_lengths(info["a"], info["b"], arm)))
             for arm, t in enumerate(ends)]
    return dict(shape=tuple(g.shape), housing=(2*info["a"]+1, 2*info["b"]+1),
                side=info["side"], fuel=info["fuel"], ticks=ticks,
                distances=tuple(info["distances"]), turns=int(max(turns)),
                docked=sum(t == d for t, d in zip(ends, info["distances"])),
                stop="fuel" if info["fuel"] < max(info["distances"]) else "dock",
                initial_spent=len(info["order"])-info["fuel"],
                palette=info["palette"])


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    shared_witness = fuel_witness = winding_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _need(g.shape == out.shape, "output size changed")
            key = _key(g)
            _need(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(g)
            predicted = _render_formula(g, info)
            simulated, ticks, ends = _simulate(g, info)
            _need(np.array_equal(out, predicted) and np.array_equal(out, simulated), "wrong output")
            _need(set(map(int, np.unique(out))) <= set(map(int, np.unique(g))), "new output color")
            frame, bg = info["palette"][1], info["palette"][0]
            converted = sum(int(g[p]) != frame and int(out[p]) == frame for p in info["order"])
            _need(converted == ticks, "counter does not charge one cell per shared tick")
            _need(all(out[p] == g[p] for p in map(tuple, np.argwhere(g == info["palette"][3]))),
                  "notch or dock changed")
            # Every painted path cell has exactly one owner, including off-grid paths.
            ownership = set()
            for path in info["paths"]:
                for p in path:
                    if _inside(p, g.shape):
                        _need(p not in ownership, "winding arms intersect")
                        ownership.add(p)
            feat = _features(g, info)
            features[split].append(feat)
            if split == "train":
                # Even granting the true travel distances, straight radial
                # motion cannot paint these off-axis first tangent cells.
                winding_witness |= any(int(g[path[1]]) == bg and int(out[path[1]]) == frame
                                       for path in info["paths"])
            if split == "train" and feat["stop"] == "dock":
                # All-docked examples retain a suffix, with a partial raster row.
                width = 2*info["b"]-1 if info["side"] in (0, 2) else 2*info["a"]-1
                shared_witness |= (sum(info["distances"]) > info["fuel"] > ticks and
                                   (len(info["order"])-info["fuel"]+ticks) % width != 0)
            if split == "train" and feat["stop"] == "fuel" and feat["turns"] >= 2:
                # Omitting A's resource limit must change at least one visible cell.
                unlimited = _render_formula(g, info, ticks=max(info["distances"]))
                fuel_witness |= not np.array_equal(predicted, unlimited)
    train, test = features["train"], features["test"]
    _need({f["stop"] for f in train} == {"fuel", "dock"}, "training must expose both stopping causes")
    _need(shared_witness, "training does not distinguish shared time from summed travel")
    _need(fuel_witness, "fuel contribution invisible in training")
    _need(winding_witness, "winding contribution invisible in training")
    _need(max(f["turns"] for f in train) == 2, "training must expose two corners")
    _need(all(f["turns"] >= 3 for f in test), "test must extrapolate through the third corner")
    _need(len({f["side"] for f in train}) >= 2, "training lacks counter rotations")


def self_test():
    """Bounded, sequential 42-seed audit; returns compact descriptive metrics."""
    # Small hand-calculated witnesses guard geometry and shared-clock semantics.
    palette = (0, 1, 2, 3)
    fixture = _base((27, 27), (13, 13), 4, 4, 0, 32, palette)
    paths = _paths(13, 13, 4, 4, fixture.shape)
    for i, d in enumerate((7, 12, 35, 37)):
        fixture[paths[i][d]] = 3
    answer = solve(fixture)
    _need(answer[7, 14] == 1 and answer[7, 21] == 0, "north dock witness")
    _need(answer[21, 15] == 3 and answer[3, 13] == 1 and answer[13, 23] == 1,
          "hand-calculated winding witness")
    _need(bool(np.all(answer[10:17, 10:17] == 1)), "32 ticks must exhaust 32 cells")
    fixture = _base((27, 27), (13, 13), 4, 4, 0, 40, palette)
    for i, d in enumerate((7, 12, 25, 31)):
        fixture[paths[i][d]] = 3
    answer = solve(fixture)
    expected = np.array([1]*40+[2]*9, dtype=np.int8).reshape(7, 7)
    _need(np.array_equal(answer[10:17, 10:17], expected),
          "last arrival consumes 31, not summed travel 75")
    hist = {k: Counter() for k in ("housing", "side", "fuel", "ticks", "docked", "stop", "turns")}
    shapes, palettes, signatures = set(), set(), set()
    rectangular = fresh = exhausted = changed_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        local_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g = pair["input"]
                before = g.copy()
                info = _analyze(g)
                feat = _features(g, info)
                _need(np.array_equal(solve(g), pair["output"]), "input-only solve mismatch")
                _need(np.array_equal(before, g), "solve modified its input")
                for name in hist:
                    hist[name][feat[name]] += 1
                shapes.add(g.shape)
                palettes.add(info["palette"])
                local_palettes.add(info["palette"])
                rectangular += info["a"] != info["b"]
                fresh += feat["initial_spent"] == 0
                exhausted += feat["fuel"] == 0
                digest.update(repr(g.shape).encode()+g.tobytes()+pair["output"].tobytes())
                # Quarter turns also rotate the scan notch, docks and rectangular core.
                _need(np.array_equal(solve(np.rot90(g)), np.rot90(pair["output"])), "rotation failure")
        changed_palettes += len(local_palettes) > 1
        signatures.add(digest.digest())
    _need(len(signatures) == 42, "seeds did not yield distinct puzzles")
    _need(len(shapes) >= 25 and len(hist["housing"]) >= 10, "insufficient geometric variation")
    _need(set(hist["side"]) == set(range(4)), "missing scan orientation")
    _need(len(hist["ticks"]) >= 25 and len(hist["fuel"]) >= 30, "counter variation collapsed")
    _need({0, 2, 4} <= set(hist["docked"]), "missing distinct docking outcomes")
    _need(rectangular > 30 and fresh > 0 and exhausted > 0, "missing counter geometries or endpoint states")
    _need(len(palettes) >= 40 and 0 < changed_palettes < 42, "palette scopes collapsed")
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(1987, tr, te)
            validate_generated_puzzle(p)
            q = generate_puzzle(1987, tr, te)
            _need(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                      for k in ("input", "output")), "nondeterministic explicit counts")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Check the full geometric family, not just the sampling regime.
    for a in range(2, 6):
        for b in range(2, 6):
            paths = _paths(15, 15, a, b, (30, 30))
            points = [p for path in paths for p in path if _inside(p, (30, 30))]
            _need(len(points) == len(set(points)), "intersecting geometric phases")
            # Cropping an infinite-plane construction keeps later re-entry cells.
            long = _paths(15, 15, a, b, (80, 80))
            for p, q in zip(paths, long):
                _need({v for v in p if _inside(v, (30, 30))} ==
                      {v for v in q if _inside(v, (30, 30))}, "path bound truncates re-entry")
    # Compact, edge-touching, fresh and exhausted inputs are valid family members,
    # even though the lesson sampler uses roomier views for corner extrapolation.
    for side in range(4):
        for fuel in (0, 9, 25):
            g = _base((13, 17), (5, 7), 3, 3, side, fuel, palette)
            paths = _paths(5, 7, 3, 3, g.shape)
            distances = []
            for path in paths:
                d = max(i for i, p in enumerate(path[1:], 1) if _inside(p, g.shape))
                distances.append(d)
                g[path[d]] = 3
            info = _analyze(g)
            out, _, _ = _simulate(g, info)
            _need(np.array_equal(solve(g), out), "compact clipping regression")
    bad = generate_puzzle(0)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a changed output pixel")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), shapes=len(shapes),
                housings=len(hist["housing"]), palettes=len(palettes),
                tick_values=len(hist["ticks"]), fuel_values=len(hist["fuel"]),
                rectangular_pairs=int(rectangular), fresh_pairs=int(fresh),
                exhausted_pairs=int(exhausted), varied_palette_puzzles=int(changed_palettes),
                docking_counts=sorted(map(int, hist["docked"])),
                turns=sorted(map(int, hist["turns"])), independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
