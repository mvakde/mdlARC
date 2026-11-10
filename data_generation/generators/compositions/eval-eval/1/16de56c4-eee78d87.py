"""Motif lattice with terminal-bounded recoloring.

Three congruent 3x3 masks form an L. Its arms encode independent translation
periods. Zero, one, or two differently colored congruent masks extend the arms;
their inclusive spans select a stripe or rectangle in the completed lattice.
Only occupied motif cells are recolored. Everything needed is in the input.

No file I/O, saved traces, source fixtures, or puzzle-dependent solver state.
Generation is bounded to 64 proposals per pair (at most 14 pairs). Self-test
streams 42 puzzles plus count/boundary regressions without retaining grids.
"""

from itertools import permutations
import numpy as np


def _full_box(mask):
    return (mask.shape == (3, 3) and 2 <= int(mask.sum()) <= 8
            and all((mask[0].any(), mask[2].any(),
                     mask[:, 0].any(), mask[:, 2].any())))


_MASKS = tuple(m for code in range(1, 511)
               if _full_box(m := np.array([(code >> i) & 1 for i in range(9)],
                                          dtype=bool).reshape(3, 3)))
_ASYMMETRIC = tuple(m for m in _MASKS
                    if len({np.rot90(f, k).tobytes()
                            for f in (m, m[:, ::-1]) for k in range(4)}) == 8)
_ORDERS = tuple((t, r, c) for t in (False, True)
                for r in permutations(range(3)) for c in permutations(range(3)))


def _ordered(mask, order):
    t, rows, cols = order
    return (mask.T if t else mask)[np.ix_(rows, cols)]


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or any(not 1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _reference_geometry(g, color):
    """Recognize three equal full-box masks at three corners of a rectangle.

    Bounding extrema locate the four possible boxes even for disconnected
    masks, including masks with an empty middle row or column.
    """
    yy, xx = np.nonzero(g == color)
    if not len(yy):
        return None
    rows = (int(yy.min()), int(yy.max()) - 2)
    cols = (int(xx.min()), int(xx.max()) - 2)
    py, px = rows[1] - rows[0], cols[1] - cols[0]
    if min(py, px) < 4:
        return None
    present, missing, masks = [], [], []
    covered = np.zeros(g.shape, dtype=bool)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            m = g[r:r + 3, c:c + 3] == color
            if m.any():
                if not _full_box(m):
                    return None
                present.append((i, j))
                masks.append(m)
                covered[r:r + 3, c:c + 3] |= m
            else:
                missing.append((i, j))
    if (len(present) != 3 or len(missing) != 1
            or not all(np.array_equal(masks[0], m) for m in masks[1:])
            or not np.array_equal(covered, g == color)):
        return None
    # The elbow is diagonally opposite the absent corner.
    i, j = (1 - v for v in missing[0])
    return {"mask": masks[0].copy(), "periods": (py, px),
            "elbow": (rows[i], cols[j]),
            "neighbors": (rows[1 - i], cols[1 - j]),
            "reference": color}


def _parse(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    if len(colors) not in (2, 3) or int(counts.max()) * 2 <= g.size:
        raise ValueError("input must have a strict-majority carrier and 1..2 inks")
    carrier = int(colors[np.argmax(counts)])
    candidates = []
    for color in (int(x) for x in colors if x != carrier):
        info = _reference_geometry(g, color)
        if info is None:
            continue
        others = [int(x) for x in colors if x not in (carrier, color)]
        highlight = others[0] if others else None
        terminals = [None, None]  # row-axis terminal, column-axis terminal
        terminal_cells = np.zeros(g.shape, dtype=bool)
        valid = True
        if highlight is not None:
            for axis in range(2):
                a = info["elbow"][axis]
                b = info["neighbors"][axis]
                period = info["periods"][axis]
                matches = []
                for x in range(a % period, g.shape[axis] - 2, period):
                    if min(a, b) <= x <= max(a, b):
                        continue
                    origin = list(info["elbow"])
                    origin[axis] = x
                    r, c = origin
                    patch = g[r:r + 3, c:c + 3] == highlight
                    if np.array_equal(patch, info["mask"]):
                        matches.append(x)
                        terminal_cells[r:r + 3, c:c + 3] |= patch
                if len(matches) > 1:
                    valid = False
                elif matches:
                    terminals[axis] = matches[0]
            valid &= np.array_equal(terminal_cells, g == highlight)
            valid &= any(t is not None for t in terminals)
        if not valid:
            continue
        info.update({"carrier": carrier, "highlight": highlight,
                     "terminals": tuple(terminals), "shape": g.shape})
        candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous L and at most one terminal per arm")
    return candidates[0]


def _bounds(info):
    result = []
    for axis, t in enumerate(info["terminals"]):
        result.append(None if t is None else
                      (min(info["elbow"][axis], info["neighbors"][axis], t),
                       max(info["elbow"][axis], info["neighbors"][axis], t)))
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only reconstruction by modular coordinates; does not call generation."""
    info = _parse(input_grid)
    h, w = info["shape"]
    py, px = info["periods"]
    er, ec = info["elbow"]
    rows, cols = np.indices((h, w))
    dr, dc = (rows - er) % py, (cols - ec) % px
    occupied = ((dr < 3) & (dc < 3)
                & info["mask"][np.minimum(dr, 2), np.minimum(dc, 2)])
    result = np.full((h, w), info["carrier"], dtype=np.int8)
    result[occupied] = info["reference"]
    if info["highlight"] is not None:
        selected = np.ones((h, w), dtype=bool)
        for origin, bound in zip((rows - dr, cols - dc), _bounds(info)):
            if bound is not None:
                selected &= (origin >= bound[0]) & (origin <= bound[1])
        result[occupied & selected] = info["highlight"]
    return result


def _stamps(info, *, ignore_terminals=False, solid=False, union=False):
    """Loop-based geometry renderer, independent of solve's modular indexing.

    Also constructs explicit simpler-rule ablations for validation.
    """
    h, w = info["shape"]
    py, px = info["periods"]
    er, ec = info["elbow"]
    bounds = _bounds(info)
    result = np.full((h, w), info["carrier"], dtype=np.int8)
    for r in range(er % py - py, h, py):
        for c in range(ec % px - px, w, px):
            inside = [b is None or b[0] <= x <= b[1]
                      for x, b in zip((r, c), bounds)]
            selected = any(inside) if union else all(inside)
            ink = (info["highlight"] if info["highlight"] is not None
                   and selected and not ignore_terminals else info["reference"])
            for dr in range(3):
                for dc in range(3):
                    y, x = r + dr, c + dc
                    if (0 <= y < h and 0 <= x < w
                            and (solid or info["mask"][dr, dc])):
                        result[y, x] = ink
    return result


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _sample_axis(rng, has_terminal, long_span, bilateral=False):
    # Reserve a full repeated motif beyond a bounded span. One open-axis
    # training witness reserves full copies on BOTH sides of its references.
    minimum = 5 if long_span else 4 if has_terminal or bilateral else 3
    periods = [p for p in range(4, 10) if (minimum - 1) * p + 3 <= 30]
    p = int(_pick(rng, periods))
    n = int(rng.integers(minimum, (30 - 3) // p + 2))
    base = (n - 1) * p + 3
    margin = 30 - base
    before = int(rng.integers(min(p - 1, margin) + 1))
    after = int(rng.integers(min(p - 1, margin - before) + 1))
    length = before + base + after
    options = []
    for low in range(n - 1):
        for direction in (-1, 1):
            a, b = (low, low + 1) if direction > 0 else (low + 1, low)
            if not has_terminal:
                if not bilateral or (low > 0 and low + 1 < n - 1):
                    options.append((a, b, None))
                continue
            steps = range(2, n - 2) if long_span else (1,)
            for step in steps:
                for t in (low - step, low + 1 + step):
                    if (0 <= t < n and
                            max(a, b, t) - min(a, b, t) + 1 < n):
                        options.append((a, b, t))
    a, b, t = _pick(rng, options)
    return {"length": length, "period": p, "a": before + a * p,
            "b": before + b * p,
            "t": None if t is None else before + t * p}


def _make_pair(rng, mask, carrier, terminal_count, test):
    axes = [False, False]
    for axis in rng.choice(2, size=terminal_count, replace=False):
        axes[int(axis)] = True
    long_axis = int(_pick(rng, [i for i in range(2) if axes[i]])) if test else -1
    axis_info = [_sample_axis(rng, flag, test and i == long_axis,
                              bilateral=not flag and terminal_count == 1)
                 for i, flag in enumerate(axes)]
    reference, highlight = [int(c) for c in rng.choice(
        [c for c in range(10) if c != carrier], 2, replace=False)]
    info = {"shape": tuple(a["length"] for a in axis_info),
            "periods": tuple(a["period"] for a in axis_info),
            "elbow": tuple(a["a"] for a in axis_info),
            "neighbors": tuple(a["b"] for a in axis_info),
            "terminals": tuple(a["t"] for a in axis_info),
            "carrier": carrier, "reference": reference,
            "highlight": highlight if terminal_count else None, "mask": mask}
    g = np.full(info["shape"], carrier, dtype=np.int8)
    er, ec = info["elbow"]
    nr, nc = info["neighbors"]
    placements = [(er, ec, reference), (nr, ec, reference), (er, nc, reference)]
    for axis, t in enumerate(info["terminals"]):
        if t is not None:
            pos = [er, ec]
            pos[axis] = t
            placements.append((*pos, highlight))
    for r, c, ink in placements:
        g[r:r + 3, c:c + 3][mask] = ink
    return {"input": g, "output": _stamps(info)}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    carrier = int(rng.integers(10))
    first = _pick(rng, _ASYMMETRIC)
    # Two masks jointly distinguish even independent row/column permutations
    # and transpose, a stricter check than merely having an asymmetric mask.
    alternatives = [order for order in _ORDERS
                    if np.array_equal(_ordered(first, order), first)]
    second_choices = [m for m in _MASKS
                      if m[1, 1] != first[1, 1] and m.sum() != first.sum()
                      and sum(np.array_equal(_ordered(m, o), m)
                              for o in alternatives) == 1]
    second = _pick(rng, second_choices)
    masks = [first, second]
    for _ in range(2, num_train):
        masks.append(_pick(rng, _MASKS))
    used_masks = {m.tobytes() for m in masks}
    for _ in range(num_test):
        m = _pick(rng, [m for m in _MASKS if m.tobytes() not in used_masks])
        masks.append(m)
        used_masks.add(m.tobytes())
    puzzle, seen = {"train": [], "test": []}, set()
    for index, mask in enumerate(masks):
        test = index >= num_train
        terminal_count = (int(rng.integers(1, 3)) if test else
                          1 if index == 0 else 2 if index == 1 else
                          int(rng.integers(3)))
        for _ in range(64):
            pair = _make_pair(rng, mask, carrier, terminal_count, test)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["test" if test else "train"].append(pair)
                break
        else:
            raise RuntimeError("bounded distinct-input construction exhausted")
    return puzzle


def _features(info):
    axes = []
    for axis in range(2):
        a, b = info["elbow"][axis], info["neighbors"][axis]
        p, length = info["periods"][axis], info["shape"][axis]
        t = info["terminals"][axis]
        full = list(range(a % p, length - 2, p))
        bound = _bounds(info)[axis]
        axes.append({"bilateral": full[0] < min(a, b) and full[-1] > max(a, b),
                     "outside": bound is not None and any(
                         x < bound[0] or x > bound[1] for x in full),
                     "steps": 0 if t is None else min(abs(t-a), abs(t-b)) // p,
                     "side": 0 if t is None else -1 if t < min(a, b) else 1,
                     "clipped": a % p >= p - 2 or (length - 3 - a) % p >= p - 2})
    return axes


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _parse(inp)
            assert inp.shape == out.shape
            assert np.array_equal(out, solve(inp)), "modular solver disagreement"
            assert np.array_equal(out, _stamps(info)), "stamp geometry disagreement"
            assert not np.array_equal(inp, out)
            nterm = sum(t is not None for t in info["terminals"])
            assert np.count_nonzero(inp != info["carrier"]) == (3 + nterm) * info["mask"].sum()
            assert set(np.unique(out)) <= {c for c in
                (info["carrier"], info["reference"], info["highlight"]) if c is not None}
            # Foreground support depends only on the periodic stencil, never
            # on highlight bounds. Thus no gap can be destroyed by recoloring.
            plain = _stamps(info, ignore_terminals=True)
            assert np.array_equal(out != info["carrier"], plain != info["carrier"])
            for f, t in zip(_features(info), info["terminals"]):
                if t is not None:
                    assert f["outside"], "generated bounded axis needs visible stopping evidence"
            infos[split].append(info)
    all_infos = infos["train"] + infos["test"]
    assert len({i["carrier"] for i in all_infos}) == 1
    train = infos["train"]
    assert {1, 2} <= {sum(t is not None for t in i["terminals"]) for i in train}
    assert len({int(i["mask"].sum()) for i in train}) >= 2
    assert {bool(i["mask"][1, 1]) for i in train} == {False, True}
    assert any(t is None and f["bilateral"] for i in train
               for t, f in zip(i["terminals"], _features(i)))
    orders = [o for o in _ORDERS if all(
        np.array_equal(_ordered(i["mask"], o), i["mask"]) for i in train)]
    assert orders == [(False, (0, 1, 2), (0, 1, 2))]
    # Concrete ablations: ignoring A's terminals, ignoring B's gaps, and
    # combining the two bounded axes by union rather than intersection.
    for option in ({"ignore_terminals": True}, {"solid": True}, {"union": True}):
        assert any(not np.array_equal(_stamps(i), _stamps(i, **option)) for i in train)
    assert all(max(f["steps"] for f in _features(i)) <= 1 for i in train)
    assert all(max(f["steps"] for f in _features(i)) >= 2 for i in infos["test"])
    train_masks = {i["mask"].tobytes() for i in train}
    assert all(i["mask"].tobytes() not in train_masks for i in infos["test"])


def self_test():
    """Bounded, streaming family, ablation, API, equivariance and diversity checks."""
    shapes, masks, periods, backgrounds, layouts, signatures = (set() for _ in range(6))
    occupancies, centers, terminal_counts, sides, clips = (set() for _ in range(5))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _parse(pair["input"])
                shapes.add(i["shape"])
                masks.add(i["mask"].tobytes())
                periods.update(i["periods"])
                backgrounds.add(i["carrier"])
                occupancies.add(int(i["mask"].sum()))
                centers.add(bool(i["mask"][1, 1]))
                terminal_counts.add(sum(t is not None for t in i["terminals"]))
                layouts.add((i["periods"], i["elbow"], i["neighbors"], i["terminals"]))
                for axis, f in enumerate(_features(i)):
                    sides.add(f["side"])
                    if f["clipped"]:
                        clips.add(axis)
        # Geometry has no privileged absolute orientation or literal color.
        p = puzzle["train"][0]
        for k in range(4):
            assert np.array_equal(solve(np.rot90(p["input"], k)), np.rot90(p["output"], k))
        palette = np.array([3, 8, 1, 6, 9, 4, 0, 2, 7, 5], dtype=np.int8)
        assert np.array_equal(solve(palette[p["input"]]), palette[p["output"]])
    assert len(signatures) == 42 and len(shapes) >= 55 and len(masks) >= 90
    assert len(layouts) >= 180 and periods == set(range(4, 10))
    assert backgrounds == set(range(10)) and occupancies == set(range(2, 9))
    assert centers == {False, True} and terminal_counts == {0, 1, 2}
    assert sides == {-1, 0, 1} and clips == {0, 1}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            validate_generated_puzzle(p)
            q = generate_puzzle(1000 + 10 * a + b, a, b)
            assert all(np.array_equal(x[k], y[k]) for s in ("train", "test")
                       for x, y in zip(p[s], q[s]) for k in ("input", "output"))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Negative validation tests address both inherited mechanisms explicitly.
    for option in ({"ignore_terminals": True}, {"solid": True}, {"union": True}):
        bad = generate_puzzle(2026)
        bad["train"][1]["output"] = _stamps(_parse(bad["train"][1]["input"]), **option)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("a concept-ablated answer was accepted")

