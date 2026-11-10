"""Welded symmetry contours: reconstruct a solid, then band its joint interior.

Background is 0; the embedded symmetry glyph is 5. The most frequent other
input color is the temporary solid material. Each solid has its own diagonal
palette key. Nothing is inferred from a seed or from other examples by solve.
"""

from collections import deque
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_ROTATION = {"N": 0, "W": 1, "S": 2, "E": 3}
_OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    answer = []
    steps = _D8 if diagonal else _D4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        answer.append(sorted(cells))
    return answer


def _depth(mask, diagonal=True):
    """Repeated erosion; marker pixels count as material, as do palette pixels."""
    current = mask.copy()
    result = np.full(mask.shape, -1, dtype=np.int16)
    h, w = mask.shape
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        interior = current.copy()
        for dr, dc in (_D8 if diagonal else _D4):
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[current & ~interior] = level
        current = interior
        level += 1
    return result


def _depth_bfs(mask):
    """A separate boundary-distance implementation, used by validation."""
    h, w = mask.shape
    result = np.full(mask.shape, -1, dtype=np.int16)
    todo = deque()
    for r, c in np.argwhere(mask).tolist():
        if any(not (0 <= r + dr < h and 0 <= c + dc < w)
               or not mask[r + dr, c + dc] for dr, dc in _D8):
            result[r, c] = 0
            todo.append((r, c))
    while todo:
        r, c = todo.popleft()
        for dr, dc in _D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and result[rr, cc] < 0:
                result[rr, cc] = result[r, c] + 1
                todo.append((rr, cc))
    return result


def _mask(shape, cells):
    m = np.zeros(shape, dtype=bool)
    if cells:
        a = np.asarray(cells, dtype=int)
        m[a[:, 0], a[:, 1]] = True
    return m


def _classify(marker):
    if len(marker) == 1:
        return "P"
    rows = {r for r, _ in marker}
    cols = {c for _, c in marker}
    if len(rows) == 1 and len(marker) == max(cols) - min(cols) + 1:
        return "H"
    if len(cols) == 1 and len(marker) == max(rows) - min(rows) + 1:
        return "V"
    raise ValueError("the marker must be one straight segment or one point")


def _transform(cells, kind, pivot):
    ar, ac = pivot
    if kind == "H":
        return [(2 * ar - r, c) for r, c in cells]
    if kind == "V":
        return [(r, 2 * ac - c) for r, c in cells]
    return [(2 * ar - r, 2 * ac - c) for r, c in cells]


def _source_side(cells, marker, kind):
    ar, ac = marker[0]
    rr, cc = np.asarray(cells).T
    possible = []
    if kind in ("H", "P"):
        if rr.max() == ar and rr.min() < ar:
            possible.append("N")
        if rr.min() == ar and rr.max() > ar:
            possible.append("S")
    if kind in ("V", "P"):
        if cc.max() == ac and cc.min() < ac:
            possible.append("W")
        if cc.min() == ac and cc.max() > ac:
            possible.append("E")
    _need(len(possible) == 1, "source must occupy one unambiguous closed half-plane")
    return possible[0]


def _corner_paths(source, union, depth, length, marker_mask):
    h, w = union.shape

    def inside(r, c):
        return 0 <= r < h and 0 <= c < w and union[r, c]

    paths = []
    for r, c in np.argwhere(source & (depth == 0)).tolist():
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            if inside(r - dr, c) or inside(r, c - dc):
                continue
            if not inside(r + dr, c) or not inside(r, c + dc):
                continue
            cells = [(r + d * dr, c + d * dc) for d in range(length)]
            if all(inside(a, b) and source[a, b] and not marker_mask[a, b]
                   and depth[a, b] == d for d, (a, b) in enumerate(cells)):
                paths.append(cells)
    return paths


def _read_input(inp):
    counts = np.bincount(inp.ravel(), minlength=10)
    candidates = [c for c in range(1, 10) if c != 5 and counts[c]]
    _need(bool(candidates), "no solid material")
    carrier = max(candidates, key=lambda c: int(counts[c]))
    _need(sum(counts[c] == counts[carrier] for c in candidates) == 1,
          "temporary material must be the unique most frequent non-role color")
    groups = []
    owners = np.full(inp.shape, -1, dtype=np.int16)
    h, w = inp.shape
    for cells in _components(inp != 0):
        marker = [(r, c) for r, c in cells if inp[r, c] == 5]
        _need(bool(marker), "every solid needs a marker")
        kind = _classify(marker)
        side = _source_side(cells, marker, kind)
        reflected = _transform(cells, kind, marker[0])
        _need(all(0 <= r < h and 0 <= c < w for r, c in reflected), "copy leaves grid")
        source = _mask(inp.shape, cells)
        union = source | _mask(inp.shape, reflected)
        _need(len(_components(source, False)) == 1, "source must be four-connected")
        _need(len(_components(union, False)) == 1, "halves must weld into one solid")
        _need(not np.any((inp != 0) & ~source & union), "copy hits another source")
        _need(not np.any((owners >= 0) & union), "completed solids overlap")
        owners[union] = len(groups)
        depth = _depth(union)
        cues = [(r, c) for r, c in cells if inp[r, c] not in (carrier, 5)]
        _need(1 <= len(cues) <= 4, "a solid needs a key of length 1..4")
        cues.sort(key=lambda q: int(depth[q]))
        _need([int(depth[q]) for q in cues] == list(range(len(cues))),
              "key must give exactly one cue per consecutive depth")
        if len(cues) > 1:
            dr, dc = cues[1][0] - cues[0][0], cues[1][1] - cues[0][1]
            _need(abs(dr) == abs(dc) == 1, "key must be diagonal")
            _need(all(q == (cues[0][0] + d * dr, cues[0][1] + d * dc)
                      for d, q in enumerate(cues)), "key must be straight")
        marker_mask = inp == 5
        _need(cues in _corner_paths(source, union, depth, len(cues), marker_mask),
              "key must enter a convex corner of the completed solid")
        period = tuple(int(inp[q]) for q in cues)
        _need(len(set(period)) == len(period), "explicit key entries must be distinct")
        _need(int(depth.max()) >= len(period), "solid must expose a repeated cycle")
        groups.append({"kind": kind, "side": side, "marker": marker,
                       "source": source, "union": union, "cells": cells,
                       "depth": depth, "period": period, "cues": cues})
    _need(bool(groups), "empty input")
    # Separation is tested after all unions exist, not just on the source halves.
    for dr, dc in _D8:
        a = owners[max(0, dr):h + min(0, dr), max(0, dc):w + min(0, dc)]
        b = owners[max(0, -dr):h + min(0, -dr), max(0, -dc):w + min(0, -dc)]
        _need(not np.any((a >= 0) & (b >= 0) & (a != b)), "completed solids touch")
    return carrier, groups


def _render(inp, groups, mode="true"):
    result = np.zeros_like(inp)
    for g in groups:
        source, union, period = g["source"], g["union"], np.array(g["period"])
        if mode == "source":
            result[source] = period[_depth(source)[source] % len(period)]
        elif mode == "precolor":
            d = _depth(source)
            target = _transform(g["cells"], g["kind"], g["marker"][0])
            for q, dest in zip(g["cells"], target):
                result[dest] = period[d[q] % len(period)]
            result[source] = period[d[source] % len(period)]
        elif mode == "flat":
            result[union] = period[0]
        else:
            d = g["depth"]
            if mode == "manhattan":
                d = _depth(union, False)
            elif mode == "box":
                rr, cc = np.nonzero(union)
                d = np.full(inp.shape, -1, dtype=np.int16)
                d[rr, cc] = np.minimum.reduce((rr - rr.min(), rr.max() - rr,
                                               cc - cc.min(), cc.max() - cc))
            result[union] = period[d[union] % len(period)]
    result[inp == 5] = 5
    return result.astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive solid union, palette order, and all final depths from this grid."""
    inp = _grid(input_grid)
    _, groups = _read_input(inp)
    return _render(inp, groups)


def _witnesses(inp, g):
    true = _render(inp, [g])
    a = np.asarray(g["cells"])
    dest = np.asarray(_transform(g["cells"], g["kind"], g["marker"][0]))
    translation = a + dest.min(axis=0) - a.min(axis=0)
    translated_union = set(g["cells"]) | set(map(tuple, translation.tolist()))
    actual_union = set(map(tuple, np.argwhere(g["union"]).tolist()))
    return {
        "weld": not np.array_equal(true, _render(inp, [g], "precolor")),
        "contour": not np.array_equal(true, _render(inp, [g], "box")),
        "eight": not np.array_equal(true, _render(inp, [g], "manhattan")),
        "turn": translated_union != actual_union,
        "A": not np.array_equal(true, _render(inp, [g], "flat")),
        "B": not np.array_equal(true, _render(inp, [g], "source")),
    }


def _prototype(rng, kind, side, length, carrier, count, core):
    """Procedural thick half-silhouettes, with open notches and optional holes."""
    for attempt in range(180):
        if count == 1:
            h, w = int(rng.integers(5, 11)), int(rng.integers(9, 22))
        elif count == 2:
            h, w = int(rng.integers(5, 8)), int(rng.integers(9, 13))
        else:
            h, w = int(rng.integers(5, 7)), int(rng.integers(9, 12))
        s = np.ones((h, w), dtype=bool)
        if core or rng.random() < .85:
            # A notch open to a side, strictly above the join, introduces a
            # true concavity. Its shape and position are sampled independently.
            for _ in range(int(rng.integers(1, 4))):
                if rng.random() < .7:
                    a = int(rng.integers(1, h - 2))
                    b = int(rng.integers(a + 1, h))
                    cut = int(rng.integers(1, min(4, w // 3)))
                    if rng.integers(2):
                        s[a:b, :cut] = False
                    else:
                        s[a:b, -cut:] = False
                else:
                    a = int(rng.integers(1, w - 2))
                    b = int(rng.integers(a + 1, min(w - 1, a + 4)))
                    cut = int(rng.integers(1, min(3, h - 2)))
                    s[:cut, a:b] = False
        if rng.random() < .22 and h >= 6:
            r, c = int(rng.integers(2, h - 2)), int(rng.integers(3, w - 3))
            s[r, c:c + int(rng.integers(1, 3))] = False
        cells = list(map(tuple, np.argwhere(s).tolist()))
        center = w // 2 + int(rng.integers(-1, 2))
        if kind == "P":
            marker = [(h - 1, center)]
            canonical_kind = "P"
        else:
            span = int(rng.integers(2, min(7, w)))
            start = int(rng.integers(0, w - span + 1))
            marker = [(h - 1, c) for c in range(start, start + span)]
            canonical_kind = "H"
        reflected = _transform(cells, canonical_kind, marker[0])
        all_cells = np.asarray(cells + reflected)
        offset = all_cells.min(axis=0)
        shape = tuple((all_cells.max(axis=0) - offset + 1).tolist())
        moved = lambda qs: [(r - int(offset[0]), c - int(offset[1])) for r, c in qs]
        source = _mask(shape, moved(cells))
        union = source | _mask(shape, moved(reflected))
        marker_mask = _mask(shape, moved(marker))
        d = _depth(union)
        if d.max() < length or len(_components(source, False)) != 1:
            continue
        paths = _corner_paths(source, union, d, length, marker_mask)
        if not paths:
            continue
        path = paths[int(rng.integers(len(paths)))]
        pool = [c for c in range(1, 10) if c not in (5, carrier)]
        period = tuple(int(c) for c in rng.choice(pool, length, replace=False))
        # Core examples always disprove numerical sorting of a palette key.
        if core and length > 1 and list(period) == sorted(period):
            period = period[::-1]
        inp = np.zeros(shape, dtype=np.int8)
        inp[source] = carrier
        inp[marker_mask] = 5
        for q, color in zip(path, period):
            inp[q] = color
        turns = _ROTATION[side]
        inp = np.rot90(inp, turns).copy()
        union = np.rot90(union, turns).copy()
        d = np.rot90(d, turns).copy()
        out = np.zeros_like(inp)
        out[union] = np.asarray(period)[d[union] % length]
        out[inp == 5] = 5
        try:
            _, groups = _read_input(inp)
        except ValueError:
            continue
        g = groups[0]
        if g["kind"] != kind or g["side"] != side:
            continue
        if core and not all(_witnesses(inp, g).values()):
            continue
        return {"input": inp, "output": out, "union": union}
    raise RuntimeError("could not construct a discriminating thick half-shape")


def _pack(rng, prototypes):
    count = len(prototypes)
    order = sorted(range(count), key=lambda i: -int(prototypes[i]["union"].sum()))
    min_h = max(p["input"].shape[0] for p in prototypes)
    min_w = max(p["input"].shape[1] for p in prototypes)
    for attempt in range(20):
        if attempt == 19:
            h = w = 30
        else:
            h = int(rng.integers(max(12, min_h), 31))
            w = int(rng.integers(max(12, min_w), 31))
        blocked = np.zeros((h, w), dtype=bool)
        inp, out = np.zeros((h, w), np.int8), np.zeros((h, w), np.int8)
        success = True
        for i in order:
            p = prototypes[i]
            ph, pw = p["input"].shape
            rr, cc = np.nonzero(p["union"])
            origins = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            if attempt < 19:
                rng.shuffle(origins)
            accepted = None
            for r, c in origins:
                if not blocked[rr + r, cc + c].any():
                    accepted = (r, c)
                    break
            if accepted is None:
                success = False
                break
            r, c = accepted
            local = inp[r:r + ph, c:c + pw]
            local[p["input"] != 0] = p["input"][p["input"] != 0]
            local_out = out[r:r + ph, c:c + pw]
            local_out[p["union"]] = p["output"][p["union"]]
            for a, b in zip(rr + r, cc + c):
                blocked[max(0, a - 1):min(h, a + 2), max(0, b - 1):min(w, b + 2)] = True
        if success:
            return {"input": inp, "output": out}
    raise RuntimeError("bounded complete-footprint packing failed")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    sides = {"H": str(rng.choice(["N", "S"])), "V": str(rng.choice(["W", "E"])),
             "P": str(rng.choice(["N", "S", "W", "E"]))}
    plans = [[] for _ in range(nt)]
    kind_order = [str(k) for k in rng.permutation(["H", "V", "P"])]
    lengths = [2, 3, int(rng.integers(2, 4))]
    slots = list(map(int, rng.permutation(nt)))
    for j, kind in enumerate(kind_order):
        plans[slots[j % nt]].append((kind, lengths[j], True))
    carriers = [int(c) for c in rng.permutation([1, 2, 3, 4, 6, 7, 8, 9])]
    puzzle, seen, training_counts = {"train": [], "test": []}, set(), set()
    for i in range(nt + ne):
        test = i >= nt
        specs = [] if test else list(plans[i])
        desired = max(len(specs), int(rng.choice([1, 2, 3], p=[.45, .40, .15])))
        if not test and i == nt - 1 and len(training_counts) == 1 and desired in training_counts:
            desired = 1 if desired != 1 and len(specs) <= 1 else max(2, len(specs))
            if desired in training_counts:
                desired = 3 if desired == 2 else 2
        while len(specs) < desired:
            specs.append((str(rng.choice(["H", "V", "P"])),
                          int(rng.choice([1, 2, 3], p=[.12, .48, .40])), False))
        rng.shuffle(specs)
        carrier = carriers[i % len(carriers)]
        for attempt in range(24):
            prototypes = [_prototype(rng, k, _OPPOSITE[sides[k]] if test else sides[k],
                                     length, carrier, desired, core)
                          for k, length, core in specs]
            pair = _pack(rng, prototypes)
            key = _signature(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("could not make a distinct input")
        seen.add(key)
        puzzle["test" if test else "train"].append(pair)
        if not test:
            training_counts.add(desired)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, shapes, counts, carriers = set(), set(), set(), set()
    witnessed, training_lengths = set(), set()
    train_sides = {k: set() for k in ("H", "V", "P")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            sig = _signature(inp)
            assert sig not in seen
            seen.add(sig)
            carrier, groups = _read_input(inp)
            assert 1 <= len(groups) <= 3
            assert np.array_equal(_render(inp, groups), out)
            assert not np.array_equal(inp, out)
            union = np.logical_or.reduce([g["union"] for g in groups])
            assert np.array_equal(out != 0, union)
            assert np.array_equal(inp == 5, out == 5)
            assert not np.any(out == carrier)
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert np.array_equal(inp[(inp != 0) & (inp != carrier)],
                                  out[(inp != 0) & (inp != carrier)])
            shapes.add(inp.shape)
            counts.add(len(groups))
            carriers.add(carrier)
            for g in groups:
                assert np.array_equal(g["depth"], _depth_bfs(g["union"]))
                assert (g["union"] & ~g["source"]).any()
                if split == "train":
                    train_sides[g["kind"]].add(g["side"])
                    training_lengths.add(len(g["period"]))
                    if all(_witnesses(inp, g).values()):
                        witnessed.add(g["kind"])
                else:
                    assert train_sides[g["kind"]] == {_OPPOSITE[g["side"]]}
    assert witnessed == {"H", "V", "P"}, "training must visibly distinguish every mechanism"
    assert all(len(s) == 1 for s in train_sides.values())
    assert {2, 3} <= training_lengths
    assert len(counts) >= 2 and len(carriers) >= 3
    assert len(shapes) >= 2


def self_test():
    """Bounded 42-seed structural/semantic diversity and adversarial checks."""
    shapes, carriers, silhouettes, marker_lengths, depths, periods, kinds = (set() for _ in range(7))
    signatures, group_counts, sides = set(), set(), set()
    holes = edges = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_signature(x["input"]) for s in ("train", "test") for x in p[s]))
        for split in ("train", "test"):
            for pair in p[split]:
                inp = pair["input"]
                before = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, before), "solve mutated its input"
                carrier, groups = _read_input(inp)
                shapes.add(inp.shape)
                carriers.add(carrier)
                group_counts.add(len(groups))
                for g in groups:
                    rr, cc = np.nonzero(g["union"])
                    crop = g["union"][rr.min():rr.max() + 1, cc.min():cc.max() + 1]
                    silhouettes.add(_signature(crop))
                    marker_lengths.add(len(g["marker"]))
                    depths.add(int(g["depth"].max()))
                    periods.add(len(g["period"]))
                    kinds.add(g["kind"])
                    sides.add((g["kind"], g["side"]))
                    holes += len(_components(~np.pad(crop, 1), False)) > 1
                    edges += bool(np.any(g["union"][0]) or np.any(g["union"][-1])
                                  or np.any(g["union"][:, 0]) or np.any(g["union"][:, -1]))
    assert len(signatures) == 42
    assert len(shapes) >= 60 and len(silhouettes) >= 120
    assert carriers == {1, 2, 3, 4, 6, 7, 8, 9}
    assert group_counts == {1, 2, 3} and kinds == {"H", "V", "P"}
    assert periods >= {1, 2, 3} and len(depths) >= 5 and max(depths) >= 6
    assert len(marker_lengths) >= 5 and len(sides) == 8
    assert holes >= 5 and edges >= 10
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(41003 + 31 * nt + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(8831, 2, 1), generate_puzzle(8831, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in ("train", "test")
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
    # Wrong parent ablations, wrong order, and marker corruption must all fail
    # validation. These mutations are derived exclusively from returned arrays.
    inp = a["train"][0]["input"]
    _, groups = _read_input(inp)
    for mode in ("source", "flat", "precolor", "box", "manhattan", "marker"):
        broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in a[s]] for s in a}
        if mode == "marker":
            out = broken["train"][0]["output"]
            out[tuple(np.argwhere(inp == 5)[0])] = 0
        else:
            broken["train"][0]["output"] = _render(inp, groups, mode)
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted {mode} corruption")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, parent ablations)")
