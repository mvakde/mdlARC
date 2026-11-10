"""Census capsules: component counts become the lengths of stable packed loads.

No filesystem access or retained generation state. Only NumPy and the standard
library are used. Diagnostics are small in-memory aggregates; no traces are saved.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 990707650, 350203408, 679370627, 288741105, 837994128,
              872434678, 114440977, 156435372, 717565556, 956224694,
              974104640, 694943141, 612680486, 591949013, 668105504,
              114148555, 295462884, 131918886, 858316130, 163188659,
              790924750, 459425340, 85476265, 199339050, 255087722,
              685120481, 265758288, 81476706, 428771318, 422867447,
              741231177, 248860288, 148652428, 765676827, 374233891,
              317143007, 647819995, 903020989, 39468061, 463335238, 878681729)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(a, b):
    a, b = _integer(a, "num_train"), _integer(b, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 training, 1..4 test, and num_train > num_test")
    return a, b


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=False)


def _components(grid, bg):
    visited = grid == bg
    parts = []
    for r, c in np.ndindex(grid.shape):
        if visited[r, c]:
            continue
        color = int(grid[r, c])
        visited[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                        and not visited[nr, nc] and grid[nr, nc] == color):
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        r0, c0 = min(p[0] for p in cells), min(p[1] for p in cells)
        h, w = max(p[0] for p in cells)-r0+1, max(p[1] for p in cells)-c0+1
        parts.append(dict(r=r0, c=c0, h=h, w=w, color=color, cells=cells))
    return parts


def _capsule(h, w, color, bg, direction, count=None):
    """Canonical horizontal patch; 0 means double wall at the left."""
    patch = np.full((h, w), bg, dtype=np.int8)
    patch[0, :] = patch[-1, :] = color
    patch[:, 0] = patch[:, -1] = color
    patch[:, 1 if direction == 0 else -2] = color
    if count is not None:
        for k in range(count):
            c = 3+2*k if direction == 0 else w-4-2*k
            patch[h//2, c] = color
    return patch


def _band(a, b):
    return max(a["r"], b["r"]) < min(a["r"]+a["h"], b["r"]+b["h"])


def _overlap(a, b, padding=0):
    return (max(a["r"], b["r"])-padding < min(a["r"]+a["h"], b["r"]+b["h"])
            and max(a["c"], b["c"])-padding < min(a["c"]+a["w"], b["c"]+b["w"]))


def _decode(value):
    grid = _grid(value)
    freq = np.bincount(grid.ravel(), minlength=10)
    bg = int(freq.argmax())
    _require(freq[bg]*2 > grid.size, "input background must occupy more than half")
    parts = _components(grid, bg)
    frames, objects, axes = [], [], set()
    for p in parts:
        if len(p["cells"]) <= 8:
            objects.append(p)
            continue
        patch = grid[p["r"]:p["r"]+p["h"], p["c"]:p["c"]+p["w"]]
        options = []
        for axis in (0, 1):
            view = patch if axis == 1 else patch.T
            h, w = view.shape
            if h < 5 or h % 2 == 0 or w < 6:
                continue
            for d in (0, 1):
                if np.array_equal(view, _capsule(h, w, p["color"], bg, d)):
                    options.append((axis, d))
        _require(len(options) == 1, "component is not one unambiguous empty capsule")
        axis, direction = options[0]
        q = dict(p, d=direction)
        if axis == 0:
            q.update(r=p["c"], c=p["r"], h=p["w"], w=p["h"])
        axes.add(axis)
        frames.append(q)
    _require(1 <= len(frames) <= 6 and len(axes) == 1, "capsules need one common encoded axis")
    _require(len({o["color"] for o in frames}) == len(frames), "capsule categories must be unique")
    axis = axes.pop()
    # Every component has a moat, and loose objects cannot lie inside a capsule.
    labels = np.full(grid.shape, -1, dtype=np.int16)
    for i, p in enumerate(parts):
        for r, c in p["cells"]:
            labels[r, c] = i
    for r, c in np.ndindex(grid.shape):
        if labels[r, c] < 0:
            continue
        near = labels[max(0, r-1):r+2, max(0, c-1):c+2]
        _require(not np.any((near >= 0) & (near != labels[r, c])), "components touch")
    for f in frames:
        f["n"] = sum(o["color"] == f["color"] for o in objects)
        f["area"] = sum(len(o["cells"]) for o in objects if o["color"] == f["color"])
        _require(2*f["n"]+4 <= (grid.shape[1] if axis else grid.shape[0]), "load exceeds canvas")
    return dict(grid=grid, canonical=grid if axis else grid.T, bg=bg,
                axis=axis, frames=frames, objects=objects)


def _pack(frames, width, unit_lengths=False, edge_first=False):
    """Pack new loads, using original leading-edge order and original row spans."""
    result = [dict(o, w=6 if unit_lengths else 2*o["n"]+4) for o in frames]
    for direction in (0, 1):
        order = [i for i, o in enumerate(frames) if o["d"] == direction]
        order.sort(key=lambda i: (frames[i]["c"] if direction == 0 else
                                 -frames[i]["c"]-frames[i]["w"], frames[i]["r"]))
        settled = []
        for i in order:
            o = result[i]
            if edge_first:
                positions = range(width-o["w"]+1)
                if direction:
                    positions = reversed(positions)
                for c in positions:
                    o["c"] = c
                    if not any(_overlap(o, p) for p in settled):
                        break
                else:
                    raise ValueError("edge-first packing cannot fit")
            else:
                blockers = [p for p in settled if _band(o, p)]
                o["c"] = (max((p["c"]+p["w"] for p in blockers), default=0)
                          if direction == 0 else
                          min((p["c"] for p in blockers), default=width)-o["w"])
            if not 0 <= o["c"] <= width-o["w"]:
                raise ValueError("stable load chain exceeds canvas")
            settled.append(o)
    if any(_overlap(a, b) for i, a in enumerate(result) for b in result[i+1:]):
        raise ValueError("opposite final fleets collide")
    return result


def _render(shape, bg, frames, loaded):
    grid = np.full(shape, bg, dtype=np.int8)
    for o in frames:
        r, c, h, w = (o[k] for k in ("r", "c", "h", "w"))
        grid[r:r+h, c:c+w] = _capsule(h, w, o["color"], bg, o["d"], o["n"] if loaded else None)
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode counts, axis, and destination exclusively from this input."""
    d = _decode(input_grid)
    packed = _pack(d["frames"], d["canonical"].shape[1])
    out = _render(d["canonical"].shape, d["bg"], packed, True)
    return out if d["axis"] else out.T.copy()


def _shape(rng, area=None):
    """Geometry is sampled without category count; bounded connected masks."""
    area = int(rng.integers(1, 9)) if area is None else area
    cells = {(0, 0)}
    if rng.random() < .25:
        cells = {(i//4, i % 4) for i in range(area)}
    else:
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))}-cells)
            cells.add(_pick(rng, frontier))
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _chain_layout(rng, counts, colors, direction):
    w = int(rng.integers(27, 31))
    short = int(_pick(rng, (5, 7)))
    step = int(rng.integers((short+1)//2, short))
    lengths = [int(rng.integers(6, 9)) for _ in range(3)]
    gaps = [int(rng.integers(1, 3)) for _ in range(2)]
    extent = short+2*step
    extra = len(counts) == 4
    if extra:
        extent += 6
    h = int(rng.integers(max(23, extent), 31))
    top = int(rng.integers(h-extent+1))
    total = sum(lengths)+sum(gaps)
    col = int(rng.integers(w-total+1))
    frames = []
    reverse = bool(rng.integers(2))
    for i in range(3):
        row = top+(2-i if reverse else i)*step
        frames.append(dict(r=row, c=col, h=short, w=lengths[i], d=direction,
                           color=colors[i], n=counts[i]))
        col += lengths[i]+(gaps[i] if i < 2 else 0)
    if extra:
        length = int(rng.integers(6, 12))
        frames.append(dict(r=top+short+2*step+1, c=int(rng.integers(w-length+1)),
                           h=5, w=length, d=direction, color=colors[3], n=counts[3]))
    if direction:
        for f in frames:
            f["c"] = w-f["c"]-f["w"]
    return (h, w), frames


def _cross_layout(rng, counts, colors, first_direction):
    h, w = int(rng.integers(21, 31)), int(rng.integers(22, 31))
    heights = [int(_pick(rng, (5, 7))), int(_pick(rng, (5, 7))), int(_pick(rng, (5, 7, 9)))]
    lengths = [int(rng.integers(6, 11)) for _ in range(3)]
    row = int(rng.integers(0, 3))
    left = int(rng.integers(w-lengths[0]-lengths[1]))
    right = int(rng.integers(left+lengths[0]+1, w-lengths[1]+1))
    lower = int(rng.integers(row+max(heights[:2])+1, h-heights[2]+1))
    frames = [dict(r=row, c=left, h=heights[0], w=lengths[0], d=1,
                   color=colors[0], n=counts[0]),
              dict(r=row, c=right, h=heights[1], w=lengths[1], d=0,
                   color=colors[1], n=counts[1]),
              dict(r=lower, c=int(rng.integers(w-lengths[2]+1)), h=heights[2],
                   w=lengths[2], d=0, color=colors[2], n=counts[2])]
    if first_direction == 0:
        for f in frames:
            f["c"], f["d"] = w-f["c"]-f["w"], 1-f["d"]
    return (h, w), frames


def _free_layout(rng, counts, colors):
    h, w = int(rng.integers(18, 31)), int(rng.integers(20, 31))
    frames = []
    for n, color in zip(counts, colors):
        for _ in range(100):
            short, length = int(_pick(rng, (5, 5, 7, 9))), int(rng.integers(6, 13))
            r, c = int(rng.integers(h-short+1)), int(rng.integers(w-length+1))
            if frames and rng.random() < .65:
                other = _pick(rng, frames)
                r = int(_pick(rng, (other["r"], other["r"]+other["h"]-1,
                                    other["r"]-short+1, other["r"]+other["h"]+1)))
            if not 0 <= r <= h-short:
                continue
            f = dict(r=r, c=c, h=short, w=length, d=int(rng.integers(2)), color=color, n=n)
            if any(_overlap(f, p, 1) for p in frames):
                continue
            frames.append(f)
            break
        else:
            return None
    return (h, w), frames


def _scatter(rng, shape, frames, bg, specs):
    grid = _render(shape, bg, frames, False)
    forbidden = np.zeros(shape, dtype=bool)
    for f in frames:
        r, c, h, w = (f[k] for k in ("r", "c", "h", "w"))
        forbidden[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = True
    ordered = sorted(specs, key=lambda s: -len(s[1]))
    for color, cells in ordered:
        sh, sw = 1+max(r for r, c in cells), 1+max(c for r, c in cells)
        if sh > shape[0] or sw > shape[1]:
            return None
        possible = np.ones((shape[0]-sh+1, shape[1]-sw+1), dtype=bool)
        for dr, dc in cells:
            possible &= ~forbidden[dr:dr+possible.shape[0], dc:dc+possible.shape[1]]
        rr, cc = np.nonzero(possible)
        if not len(rr):
            return None
        k = int(rng.integers(len(rr)))
        for dr, dc in cells:
            r, c = int(rr[k])+dr, int(cc[k])+dc
            grid[r, c] = color
            forbidden[max(0, r-1):r+2, max(0, c-1):c+2] = True
    if np.count_nonzero(grid == bg)*2 <= grid.size:
        return None
    return grid


def _make_pair(rng, purpose, counts, colors, bg, axis, direction, forced, distractors):
    specs = [(color, _shape(rng, forced.get((i, j))))
             for i, (color, n) in enumerate(zip(colors, counts)) for j in range(n)]
    specs += [(color, _shape(rng)) for color in distractors]
    for _ in range(160):
        layout = (_chain_layout(rng, counts, colors, direction) if purpose == "chain" else
                  _cross_layout(rng, counts, colors, direction) if purpose == "cross" else
                  _free_layout(rng, counts, colors))
        if layout is None:
            continue
        shape, frames = layout
        try:
            packed = _pack(frames, shape[1])
        except ValueError:
            continue
        if purpose == "target" and not any(a["d"] == b["d"] and _band(a, b)
                                           for i, a in enumerate(frames) for b in frames[i+1:]):
            continue
        inp = _scatter(rng, shape, frames, bg, specs)
        if inp is None:
            continue
        out = _render(shape, bg, packed, True)
        if axis == 0:
            inp, out = inp.T.copy(), out.T.copy()
        return {"input": inp, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    anchor, reserved = [int(x) for x in rng.choice(10, 2, replace=False)]
    first_axis, first_direction = int(rng.integers(2)), int(rng.integers(2))
    teaching_area = int(rng.integers(4, 9))
    partition = int(rng.integers(1, teaching_area))
    puzzle, seen, first_colors = {"train": [], "test": []}, set(), None
    for index in range(nt+nq):
        train, forced = index < nt, {}
        if index == 0:
            number = int(_pick(rng, (3, 3, 4)))
            counts = [1, 2, 0]+([int(rng.integers(3))] if number == 4 else [])
            purpose, axis = "chain", first_axis
            forced = {(0, 0): teaching_area, (1, 0): 2, (1, 1): teaching_area-1}
        elif index == 1:
            number, counts, purpose, axis = 3, [2, 1, int(rng.integers(3))], "cross", 1-first_axis
            forced = {(0, 0): partition, (0, 1): teaching_area-partition,
                      (1, 0): 1 if teaching_area != 1 else 2}
        else:
            number = int(rng.integers(1 if train else 2, 5))
            counts = [int(rng.integers(3 if train else 4)) for _ in range(number)]
            if not train:
                counts[0] = int(rng.integers(3, 5))
                while sum(counts) > 9:
                    j = max(range(1, number), key=lambda k: counts[k])
                    counts[j] -= 1
            purpose, axis = ("free" if train else "target"), int(rng.integers(2))
        bg_pool = [c for c in range(10) if not (index < 2 and c == anchor)
                   and not (index == 1 and c == first_colors[2])
                   and not (not train and c == reserved)]
        bg = int(_pick(rng, bg_pool))
        pool = [c for c in range(10) if c != bg and (not train or c != reserved)
                and not (index == 1 and c == first_colors[2])]
        if index < 2:
            pool.remove(anchor)
            colors = [anchor]+[int(x) for x in rng.choice(pool, number-1, replace=False)]
        elif not train:
            pool.remove(reserved)
            colors = [reserved]+[int(x) for x in rng.choice(pool, number-1, replace=False)]
        else:
            colors = [int(x) for x in rng.choice(pool, number, replace=False)]
        if index == 0:
            first_colors = colors[:]
        dpool = [c for c in range(10) if c != bg and c not in colors]
        distractors = [int(_pick(rng, dpool)) for _ in range(int(rng.integers(3)))]
        if index == 1:
            distractors.append(first_colors[2])
        for _ in range(12):
            pair = _make_pair(rng, purpose, counts, colors, bg, axis,
                              first_direction, forced, distractors)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                break
        else:
            raise RuntimeError("distinct scene construction exhausted")
        seen.add(key)
        puzzle["train" if train else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _constraint_positions(frames, width):
    """Validator: recursive predecessor constraints rather than a settling scan."""
    cache, active = {}, set()
    def position(i):
        if i in cache:
            return cache[i]
        _require(i not in active, "cyclic predecessor graph")
        active.add(i)
        a = frames[i]
        predecessors = [j for j, b in enumerate(frames) if i != j and a["d"] == b["d"]
                        and _band(a, b) and (b["c"]+b["w"] <= a["c"] if a["d"] == 0
                                               else b["c"] >= a["c"]+a["w"])]
        length = 2*a["n"]+4
        cache[i] = (max((position(j)+2*frames[j]["n"]+4 for j in predecessors), default=0)
                    if a["d"] == 0 else min((position(j) for j in predecessors), default=width)-length)
        active.remove(i)
        return cache[i]
    return [position(i) for i in range(len(frames))]


def _different_pack(frames, width):
    correct = _pack(frames, width)
    try:
        alternative = _pack(frames, width, edge_first=True)
    except ValueError:
        return True
    return [o["c"] for o in correct] != [o["c"] for o in alternative]


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas changed")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _decode(inp)
            shape, frames = d["canonical"].shape, d["frames"]
            columns = _constraint_positions(frames, shape[1])
            settled = [dict(f, c=c, w=2*f["n"]+4) for f, c in zip(frames, columns)]
            _require(all(0 <= o["c"] <= shape[1]-o["w"] for o in settled), "load bounds")
            _require(not any(_overlap(a, b) for i, a in enumerate(settled) for b in settled[i+1:]), "final overlap")
            expected = _render(shape, d["bg"], settled, True)
            actual = out if d["axis"] else out.T
            _require(np.array_equal(actual, expected), "wrong count, capsule geometry, stable packing, or erasure")
            _require(not np.array_equal(inp, out), "unmodified pair")
            d["settled"] = settled
            details[split].append(d)
    train, test = details["train"], details["test"]
    records = [(f["color"], f["n"], f["area"]) for d in train for f in d["frames"]]
    _require(any(c == cc and n != nn and a == aa for c, n, a in records for cc, nn, aa in records), "area/count collision witness")
    _require(any(n == nn and a != aa for c, n, a in records for cc, nn, aa in records), "same count at different areas")
    _require({0, 1, 2} <= {n for c, n, a in records}, "unary length progression")
    _require({d["axis"] for d in train} == {0, 1}, "visibly encoded axes need teaching examples")
    _require({f["d"] for d in train for f in d["frames"]} == {0, 1}, "both destination signs")
    _require(any(_different_pack(d["frames"], d["canonical"].shape[1]) for d in train), "stable-order training witness")
    categories = {f["color"] for d in train for f in d["frames"]}
    discards = {o["color"] for d in train for o in d["objects"]
               if o["color"] not in {f["color"] for f in d["frames"]}}
    _require(bool(categories & discards), "local category/distractor role witness")
    _require(any(f["color"] not in categories for d in test for f in d["frames"]), "new test category")
    _require(max(f["n"] for d in test for f in d["frames"]) > max(n for c, n, a in records), "count extrapolation shift")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = _grid(pair[role])
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def _metamorphic_checks(pair):
    """Concrete input interventions: area invariance and count-to-contact effect."""
    inp, out = pair["input"], pair["output"]
    d = _decode(inp)
    color_map = np.asarray([3, 6, 9, 2, 5, 8, 1, 4, 7, 0], dtype=np.int8)
    _require(np.array_equal(solve(color_map[inp]), color_map[out]), "color equivariance")
    _require(np.array_equal(solve(inp.T), out.T), "axis equivariance")
    # Grow one counted blob by one cell, using visible separation constraints.
    grown = None
    categories = {f["color"] for f in d["frames"]}
    for o in d["objects"]:
        if o["color"] not in categories or len(o["cells"]) == 8:
            continue
        cells = set(o["cells"])
        for r, c in sorted({(r+dr, c+dc) for r, c in cells
                            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))}-cells):
            if not (0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]) or inp[r, c] != d["bg"]:
                continue
            if any(inp[rr, cc] != d["bg"] and (rr, cc) not in cells
                   for rr in range(max(0, r-1), min(inp.shape[0], r+2))
                   for cc in range(max(0, c-1), min(inp.shape[1], c+2))):
                continue
            grown = inp.copy()
            grown[r, c] = o["color"]
            break
        if grown is not None:
            break
    _require(grown is not None and np.array_equal(solve(grown), out), "area must not change census")
    # Repartition one interacting one-object category into two objects with the
    # SAME total area. This changes both its length and a different capsule's
    # position. Object locations are resampled, which the rule ignores.
    frames, shape = d["frames"], d["canonical"].shape
    target = next(f for f in frames if f["n"] == 1 and f["area"] >= 4
                  and any(f is not b and f["d"] == b["d"] and _band(f, b) for b in frames))
    specs = []
    for o in d["objects"]:
        if o["color"] != target["color"]:
            cells = [(r-o["r"], c-o["c"]) for r, c in o["cells"]]
            if d["axis"] == 0:
                cells = [(c, r) for r, c in cells]
            specs.append((o["color"], tuple(cells)))
    rng = np.random.default_rng(20261023)
    specs += [(target["color"], _shape(rng, a)) for a in (2, target["area"]-2)]
    changed = None
    for _ in range(12):
        changed = _scatter(rng, shape, frames, d["bg"], specs)
        if changed is not None:
            break
    _require(changed is not None, "bounded intervention placement")
    changed = changed if d["axis"] else changed.T.copy()
    dd = _decode(changed)
    before = {f["color"]: f for f in _pack(frames, shape[1])}
    after = {f["color"]: f for f in _pack(dd["frames"], shape[1])}
    other = next(f for f in dd["frames"] if f["color"] == target["color"])
    _require(other["area"] == target["area"] and other["n"] == 2, "area-preserving count intervention")
    _require(after[target["color"]]["w"] == before[target["color"]]["w"]+2, "count-to-length link")
    _require(any(before[c]["c"] != after[c]["c"] for c in before if c != target["color"]),
             "changing A must change a different capsule's B placement")
    _require(not np.array_equal(solve(changed), out), "omitting component counting changes answer")
    return True


def self_test():
    """42 puzzles, streamed metrics, concrete mutations; no sample corpus saved."""
    signatures, shapes, masks, palettes, heights, numbers, lengths, backgrounds = (set() for _ in range(8))
    expansions = contractions = crossings = independent = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        _metamorphic_checks(puzzle["train"][0])
        signatures.add(_fingerprint(puzzle))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "determinism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                d = _decode(pair["input"])
                frames, width = d["frames"], d["canonical"].shape[1]
                settled = _pack(frames, width)
                shapes.add(pair["input"].shape)
                palettes.add(tuple(sorted(f["color"] for f in frames)))
                backgrounds.add(d["bg"])
                heights.update(f["h"] for f in frames)
                numbers.add(len(frames))
                lengths.update(f["w"] for f in settled)
                expansions += sum(b["w"] > a["w"] for a, b in zip(frames, settled))
                contractions += sum(b["w"] < a["w"] for a, b in zip(frames, settled))
                crossings += any(a["d"] == 0 and b["d"] == 1 and _band(a, b)
                                 and a["c"] > b["c"] for a in frames for b in frames)
                independent += any(a["d"] == b["d"] and not _band(a, b) and a["c"] == b["c"]
                                   for i, a in enumerate(settled) for b in settled[i+1:])
                for o in d["objects"]:
                    masks.add(tuple(sorted((r-o["r"], c-o["c"]) for r, c in o["cells"])))
    _require(len(signatures) == 42 and len(shapes) >= 30 and len(palettes) >= 30, "structural variation")
    _require(backgrounds == set(range(10)) and heights == {5, 7, 9}, "geometry/palette roles")
    _require(numbers == {1, 2, 3, 4} and lengths == {4, 6, 8, 10, 12}, "load/count variation")
    _require(len(masks) >= 50 and min(expansions, contractions, crossings, independent) >= 20, "mechanism variation")
    # Every legal explicit count combination, and strict rejection of bad types.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(137, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    _metamorphic_checks(p["train"][0])
    d = _decode(p["train"][0]["input"])
    _require(_different_pack(d["frames"], d["canonical"].shape[1]), "overtaking regression")
    correct = _pack(d["frames"], d["canonical"].shape[1])
    uncounted = _pack(d["frames"], d["canonical"].shape[1], unit_lengths=True)
    _require(any(a["c"] != b["c"] for a, b in zip(correct, uncounted)), "count-independent packing shortcut")
    _require(any(a["c"] != b["c"] for a, b in zip(d["frames"], correct)), "omitting edge packing changes answer")
    for mutation in ("dot", "position", "unerased", "overtaking"):
        broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        if mutation == "overtaking":
            old = _pack(d["frames"], d["canonical"].shape[1], edge_first=True)
            wrong = _render(d["canonical"].shape, d["bg"], old, True)
            broken["train"][0]["output"] = wrong if d["axis"] else wrong.T.copy()
        elif mutation == "unerased":
            broken["test"][0]["output"] = broken["test"][0]["input"].copy()
        else:
            out = broken["test"][0]["output"]
            if mutation == "position":
                broken["test"][0]["output"] = np.roll(out, 1, axis=0)
            else:
                dd = _decode(broken["test"][0]["input"])
                loaded = _pack(dd["frames"], dd["canonical"].shape[1])
                f = next(o for o in loaded if o["n"])
                canonical = out if dd["axis"] else out.T
                canonical[f["r"]+f["h"]//2, f["c"]+(3 if f["d"] == 0 else f["w"]-4)] = dd["bg"]
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted semantic corruption: "+mutation)
    return dict(seeds=42, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                category_palettes=len(palettes), object_masks=len(masks),
                expansions=expansions, contractions=contractions,
                crossing_pairs=crossings, edge_sharing_pairs=independent)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
