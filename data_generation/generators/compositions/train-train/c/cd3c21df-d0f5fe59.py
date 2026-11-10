"""Partner lookup: exact object pairs form a symmetric permutation matrix.

Only NumPy and the Python standard library are used. Generation labels are local
construction data; solve recovers everything afresh from its input pixels.
"""
from __future__ import annotations

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not bool(np.all((0 <= a) & (a <= 9)))):
        raise ValueError("expected an integer 1..30 by 1..30 grid of colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return tuple(a.shape), tuple(int(x) for x in a.flat)


def _gap(a, b):
    # Half-open bounding boxes, with at least one whole blank row or column.
    r, c, rr, cc = a
    s, d, ss, dd = b
    return rr + 1 <= s or ss + 1 <= r or cc + 1 <= d or dd + 1 <= c


def _parse(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if int(counts[bg]) * 2 <= grid.size:
        raise ValueError("background must be a strict majority")
    remaining = set(map(tuple, np.argwhere(grid != bg).tolist()))
    records = []
    while remaining:
        anchor = min(remaining)
        remaining.remove(anchor)
        cells, stack = {anchor}, [anchor]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        rows, cols = zip(*cells)
        box = min(rows), min(cols), max(rows) + 1, max(cols) + 1
        r, c, rr, cc = box
        crop = grid[r:rr, c:cc].copy()
        if int(np.count_nonzero(crop != bg)) != len(cells):
            raise ValueError("another object lies inside this object's crop")
        records.append({"anchor": anchor, "box": box, "crop": crop,
                        "key": _key(crop), "cells": cells})
    records.sort(key=lambda x: x["anchor"])
    if not 3 <= len(records) <= 29:
        raise ValueError("family requires 3..29 objects")
    ink = int(grid[records[0]["anchor"]])
    if any(int(grid[x["anchor"]]) != ink for x in records):
        raise ValueError("all objects must have the same first occupied color")
    for i, rec in enumerate(records):
        if any(not _gap(rec["box"], other["box"]) for other in records[i + 1:]):
            raise ValueError("object boxes need a blank separating row or column")
    groups = {}
    for i, rec in enumerate(records):
        groups.setdefault(rec["key"], []).append(i)
    sizes = sorted(map(len, groups.values()))
    if sizes[0] != 1 or sizes.count(1) != 1 or any(s != 2 for s in sizes[1:]):
        raise ValueError("need exactly one singleton and otherwise exact pairs")
    return bg, ink, records, groups


def _table(records, bg, ink):
    groups = {}
    for i, rec in enumerate(records):
        groups.setdefault(rec["key"], []).append(i)
    n = len(records)
    out = np.full((n, n), bg, dtype=np.int8)
    for indices in groups.values():
        if len(indices) == 1:
            out[indices[0], indices[0]] = ink
        elif len(indices) == 2:
            i, j = indices
            out[i, j] = out[j, i] = ink
        else:
            raise ValueError("ambiguous partner")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover scan indices and exact partners from this input alone."""
    bg, ink, records, _ = _parse(input_grid)
    return _table(records, bg, ink)


def _crop_mask(mask):
    r, c = np.where(mask)
    return mask[r.min():r.max() + 1, c.min():c.max() + 1].copy()


def _mask(rng, kind=None):
    kinds = ("pixel", "bar", "rect", "bend", "ring", "stair", "grow")
    if kind is None:
        kind = kinds[int(rng.integers(len(kinds)))]
    if kind == "pixel":
        a = np.ones((1, 1), dtype=bool)
    elif kind == "bar":
        a = np.ones((1, int(rng.integers(2, 7))), dtype=bool)
    elif kind == "rect":
        a = np.ones((int(rng.integers(2, 5)), int(rng.integers(2, 5))), dtype=bool)
    elif kind == "bend":
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        a = np.zeros((h, w), dtype=bool)
        a[:, 0] = a[-1, :] = True
    elif kind == "ring":
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        a = np.ones((h, w), dtype=bool)
        a[1:-1, 1:-1] = False
    elif kind == "stair":
        h = int(rng.integers(2, 5))
        a = np.zeros((h, h), dtype=bool)
        for i in range(h):
            a[i, i] = True
            if i:
                a[i - 1, i] = True
    elif kind == "grow":
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        wanted = int(rng.integers(4, min(h * w, 15) + 1))
        while len(cells) < wanted:
            frontier = sorted({(rr, cc) for r, c in cells
                               for rr, cc in ((r - 1, c), (r + 1, c),
                                              (r, c - 1), (r, c + 1))
                               if 0 <= rr < h and 0 <= cc < w} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
        a = np.zeros((h, w), dtype=bool)
        for p in cells:
            a[p] = True
        a = _crop_mask(a)
    else:
        raise ValueError("unknown shape")
    a = np.rot90(a, int(rng.integers(4)))
    if rng.random() < .5:
        a = np.fliplr(a)
    return a.copy()


def _color(rng, mask, palette):
    bg, ink, p, q = palette
    a = np.full(mask.shape, bg, dtype=np.int8)
    a[mask] = ink
    indices = np.flatnonzero(mask)[1:]
    if len(indices) and rng.random() < .72:
        num = int(rng.integers(1, len(indices) + 1))
        locations = rng.choice(indices, size=num, replace=False)
        colors = (p,) if rng.random() < .6 else (p, q)
        a.flat[locations] = rng.choice(colors, size=num)
    return a


def _transform(a, op):
    if op < 4:
        return np.rot90(a, op).copy()
    return np.rot90(np.fliplr(a), op - 4).copy()


def _colored_challenge(rng, palette):
    bg, ink, payload, _ = palette
    if rng.random() < .35:
        mask = np.ones((1, int(rng.integers(5, 7))), dtype=bool)
        mask = np.rot90(mask, int(rng.integers(2)))
    else:
        mask = np.ones((int(rng.integers(2, 4)), int(rng.integers(3, 5))), dtype=bool)
    a = np.full(mask.shape, payload, dtype=np.int8)
    a.flat[0] = ink
    second = int(rng.integers(1, a.size))
    a.flat[second] = ink
    other = int(rng.choice([i for i in range(1, a.size) if i != second]))
    b = a.copy()
    b.flat[second], b.flat[other] = b.flat[other], b.flat[second]
    # Equal dimensions, silhouette, palette, and histogram; arrangement differs.
    return b, a


def _orientation_challenge(rng, palette):
    bg, ink, p, q = palette
    for _ in range(60):
        mask = _mask(rng, "bend")
        op = int(rng.integers(1, 8))
        transformed = _transform(mask, op)
        if _key(mask) == _key(transformed):
            continue
        labels = np.full(mask.shape, -1, dtype=int)
        labels[mask] = np.arange(int(mask.sum()))
        moved_labels = _transform(labels, op)
        other_first = int(moved_labels[moved_labels >= 0][0])
        values = np.full(int(mask.sum()), ink, dtype=np.int8)
        free = [i for i in range(len(values)) if i not in (0, other_first)]
        if not free:
            continue
        values[free] = rng.choice((ink, p, q), size=len(free))
        values[free[0]] = p
        a = np.full(mask.shape, bg, dtype=np.int8)
        a[mask] = values
        b = _transform(a, op)
        assert int(b[b != bg][0]) == ink
        return b, a
    raise RuntimeError("failed to make an oriented nonmatch")


def _inventory(rng, families, mode, palette):
    bg = palette[0]
    if mode == "colors":
        target, first = _colored_challenge(rng, palette)
        decoys = [first]
    elif mode == "orientation":
        target, first = _orientation_challenge(rng, palette)
        decoys = [first]
    else:
        target, decoys = _color(rng, _mask(rng), palette), []
    forbidden = {_key(target)} | {_key(x) for x in decoys}
    while len(decoys) < families:
        for _ in range(100):
            kind = "ring" if mode == "orientation" and len(decoys) == 1 else None
            candidate = _color(rng, _mask(rng, kind), palette)
            if kind == "ring" and np.count_nonzero(candidate != bg) == np.count_nonzero(target != bg):
                continue
            if _key(candidate) not in forbidden:
                forbidden.add(_key(candidate))
                decoys.append(candidate)
                break
        else:
            raise RuntimeError("failed to make distinct patterns")
    entries = [(0, target)]
    for family, a in enumerate(decoys, start=1):
        entries.extend(((family, a), (family, a.copy())))
    return entries


def _try_place(rng, entries, bg, h, w, edge):
    grid = np.full((h, w), bg, dtype=np.int8)
    boxes, placed = [], []
    # Area order aids packing; random tie breaks avoid paired layout templates.
    order = list(rng.permutation(len(entries)))
    order.sort(key=lambda i: -entries[i][1].size)
    for position, index in enumerate(order):
        family, a = entries[index]
        ph, pw = a.shape
        if ph > h or pw > w:
            return None
        chosen = None
        for location in rng.permutation((h - ph + 1) * (w - pw + 1)):
            r, c = divmod(int(location), w - pw + 1)
            if edge and position == 0 and not (r == 0 or c == 0 or r + ph == h or c + pw == w):
                continue
            box = r, c, r + ph, c + pw
            if all(_gap(box, other) for other in boxes):
                chosen = box
                break
        if chosen is None:
            return None
        r, c, rr, cc = chosen
        grid[r:rr, c:cc] = a
        first = np.argwhere(a != bg)[0]
        placed.append((family, (r + int(first[0]), c + int(first[1]))))
        boxes.append(chosen)
    if int(np.count_nonzero(grid != bg)) * 2 >= grid.size:
        return None
    return grid, placed


def _place(rng, entries, bg):
    area = sum(int(np.count_nonzero(a != bg)) for _, a in entries)
    regime = int(rng.integers(4))
    if regime == 0:
        side = max(8, int(np.ceil(np.sqrt(area * 3))))
        h, w = side + int(rng.integers(4)), side + int(rng.integers(4))
    elif regime == 1:
        h, w = int(rng.integers(6, 13)), int(rng.integers(21, 31))
    elif regime == 2:
        h, w = int(rng.integers(21, 31)), int(rng.integers(6, 13))
    else:
        h, w = int(rng.integers(16, 31)), int(rng.integers(16, 31))
    edge = bool(rng.integers(2))
    for growth in range(0, 19, 2):
        hh, ww = min(30, h + growth), min(30, w + growth)
        for _ in range(3):
            result = _try_place(rng, entries, bg, hh, ww, edge)
            if result is not None:
                return result
    # All masks fit 6x6 and at most eleven objects occupy sixteen 7x7 slots.
    # This bounded fallback never changes an object's identity or count.
    grid = np.full((30, 30), bg, dtype=np.int8)
    placed = []
    row_offset, col_offset = int(rng.integers(4)), int(rng.integers(4))
    for slot, (family, a) in zip(rng.permutation(16), entries):
        r, c = divmod(int(slot), 4)
        r, c = 7 * r + row_offset, 7 * c + col_offset
        ph, pw = a.shape
        grid[r:r + ph, c:c + pw] = a
        first = np.argwhere(a != bg)[0]
        placed.append((family, (r + int(first[0]), c + int(first[1]))))
    return grid, placed


def _constructed_output(placed, bg, ink):
    """Use construction labels, not the pixel solver, for generated answers."""
    ordered = sorted(placed, key=lambda x: x[1])
    groups = {}
    for index, (family, _) in enumerate(ordered):
        groups.setdefault(family, []).append(index)
    out = np.full((len(ordered), len(ordered)), bg, dtype=np.int8)
    for family, indices in groups.items():
        if family == 0:
            i, = indices
            out[i, i] = ink
        else:
            i, j = indices
            out[i, j] = out[j, i] = ink
    return out


def _features(input_grid):
    bg, ink, records, groups = _parse(input_grid)
    n = len(records)
    singleton = next(v[0] for v in groups.values() if len(v) == 1)
    out = _table(records, bg, ink)
    by_column = sorted(records, key=lambda x: min((c, r) for r, c in x["cells"]))
    unique = [records[v[0]]["crop"] for v in groups.values()]
    masks, histograms = {}, {}
    for a in unique:
        masks.setdefault(_key(a != bg), []).append(_key(a))
        hist = tuple(int(x) for x in np.bincount(a[a != bg], minlength=10))
        histograms.setdefault((_key(a != bg), hist), []).append(_key(a))
    dihedral = any(_key(a) != _key(b) and any(
        _key(_transform(a, op)) == _key(b) for op in range(1, 8))
        for i, a in enumerate(unique) for b in unique[i + 1:])
    fg_counts = np.bincount(np.asarray(input_grid).ravel(), minlength=10)
    fg_counts[bg] = 0
    return {"bg": bg, "ink": ink, "n": n, "families": (n - 1) // 2,
            "singleton": singleton, "interior": 0 < singleton < n - 1,
            "area": sum(len(x["cells"]) for x in records),
            "column_diff": not np.array_equal(out, _table(by_column, bg, ink)),
            "nonlocal": any(len(v) == 2 and abs(v[0] - v[1]) > 1 for v in groups.values()),
            "silhouette_collision": any(len(v) > 1 for v in masks.values()),
            "histogram_collision": any(len(v) > 1 for v in histograms.values()),
            "dihedral_collision": dihedral,
            "ink_not_modal": int(fg_counts[ink]) < int(fg_counts.max()),
            "hole": any(np.any(a == bg) for a in unique),
            "records": records, "groups": groups}


def _pair(rng, families, mode, palette, *, interior=None,
          column_witness=False, nonlocal_witness=False, different_area=None,
          different_average=None):
    for _ in range(100):
        entries = _inventory(rng, families, mode, palette)
        inp, placed = _place(rng, entries, palette[0])
        out = _constructed_output(placed, palette[0], palette[1])
        features = _features(inp)
        if interior is not None and features["interior"] != interior:
            continue
        if column_witness and not features["column_diff"]:
            continue
        if nonlocal_witness and not features["nonlocal"]:
            continue
        if different_area is not None and abs(features["area"] - different_area) < 6:
            continue
        if different_average is not None:
            other_n, other_area = different_average
            if features["area"] * other_n == other_area * features["n"]:
                continue
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct the requested demonstration witnesses")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    fixed_palette = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    variable_palette = bool(rng.integers(2))
    palettes = []
    for i in range(a + b):
        pal = (tuple(int(x) for x in rng.choice(10, size=4, replace=False))
               if variable_palette else fixed_palette)
        if variable_palette and i == 1 and pal[:2] == palettes[0][:2]:
            pal = (pal[1], pal[0], pal[2], pal[3])
        palettes.append(pal)
    first_interior = bool(rng.integers(2))
    plans = [(1, "colors", first_interior), (2, "orientation", not first_interior)]
    rng.shuffle(plans)
    puzzle = {"train": [], "test": []}
    seen = set()

    def add(split, families, mode, palette, **kwargs):
        for _ in range(12):
            pair = _pair(rng, families, mode, palette, **kwargs)
            signature = _key(pair["input"])
            if signature not in seen:
                seen.add(signature)
                puzzle[split].append(pair)
                return pair
        raise RuntimeError("duplicate inputs")

    for i, (families, mode, interior) in enumerate(plans):
        previous = (_features(puzzle["train"][0]["input"]) if i else None)
        add("train", families, mode, palettes[i], interior=interior,
            column_witness=(mode == "orientation"),
            nonlocal_witness=(mode == "orientation"),
            different_average=(previous["n"], previous["area"]) if previous else None)
    if a >= 3:
        reference = puzzle["train"][int(rng.integers(2))]
        f = _features(reference["input"])
        add("train", f["families"], "generic", palettes[2], different_area=f["area"])
    for i in range(3, a):
        add("train", int(rng.integers(1, 5)), "generic", palettes[i])
    maximum = max(_features(pair["input"])["families"] for pair in puzzle["train"])
    test_families = [int(rng.integers(maximum + 1, 6))]
    for _ in range(b - 1):
        available = [k for k in range(1, 6) if k not in test_families]
        test_families.append(int(rng.choice(available)))
    for i, families in enumerate(test_families):
        add("test", families, "generic", palettes[a + i])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate returned pixels, including diagnostic demonstration evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, all_features = set(), {}
    for split in ("train", "test"):
        features = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            f = _features(inp)
            bg, ink, n = f["bg"], f["ink"], f["n"]
            assert 1 <= f["families"] <= 5
            assert out.shape == (n, n)
            assert set(np.unique(out)) == {bg, ink}
            marked = out == ink
            assert np.all(marked.sum(axis=0) == 1)
            assert np.all(marked.sum(axis=1) == 1)
            assert np.array_equal(out, out.T)
            assert int(np.trace(marked)) == 1
            assert marked[f["singleton"], f["singleton"]]
            # Each row must point to its actual matching crop, not any involution.
            for indices in f["groups"].values():
                if len(indices) == 2:
                    i, j = indices
                    assert marked[i, j] and marked[j, i]
                else:
                    i, = indices
                    assert marked[i, i]
            assert np.array_equal(solve(inp), out)
            count_only = np.full((n, n), bg, dtype=np.int8)
            np.fill_diagonal(count_only, ink)
            assert not np.array_equal(out, count_only), "A ablation did not change answer"
            singleton_crop = f["records"][f["singleton"]]["crop"]
            assert not np.array_equal(out, singleton_crop), "B ablation did not change answer"
            features.append(f)
        all_features[split] = features
    train, test = all_features["train"], all_features["test"]
    assert {1, 2} <= {f["families"] for f in train}
    assert {False, True} <= {f["interior"] for f in train}
    assert any(f["histogram_collision"] and f["ink_not_modal"] for f in train)
    assert any(f["dihedral_collision"] and f["hole"] for f in train)
    assert any(f["column_diff"] for f in train)
    assert any(f["nonlocal"] for f in train)
    assert len({(f["n"], f["area"]) for f in train}) >= 2
    assert train[0]["area"] * train[1]["n"] != train[1]["area"] * train[0]["n"]
    if len(train) >= 3:
        assert any(x["n"] == y["n"] and abs(x["area"] - y["area"]) >= 6
                   for i, x in enumerate(train) for y in train[i + 1:])
    assert test[0]["families"] > max(f["families"] for f in train)
    assert len({f["families"] for f in test}) == len(test)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _analytic_checks():
    # Hand-specified tie, color-arrangement, relocation, and palette fixtures.
    p = np.array([[1, 2], [1, 1]], dtype=np.int8)
    q = np.array([[1, 1], [2, 1]], dtype=np.int8)
    original = np.zeros((6, 7), dtype=np.int8)
    original[0:2, 0:2] = p
    original[0:2, 4:6] = q
    original[3:5, 0:2] = p
    expected = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int8)
    assert np.array_equal(solve(original), expected)
    # Swap two objects without rotating either crop: same count, new relation.
    relocated = original.copy()
    relocated[0:2, 0:2] = q
    relocated[0:2, 4:6] = p
    moved_answer = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int8)
    assert np.array_equal(solve(relocated), moved_answer)
    assert not np.array_equal(expected, moved_answer)
    mapping = np.array([8, 0, 7, 1, 2, 3, 4, 5, 6, 9], dtype=np.int8)
    assert np.array_equal(solve(mapping[original]), mapping[expected])
    assert np.array_equal(solve(np.pad(original, ((2, 1), (1, 3)))), expected)
    # Arbitrary shapes beyond the generator's mask library remain supported.
    tall = np.ones((9, 1), dtype=np.int8)
    other = np.array([[1, 2, 1, 2, 1, 2, 1]], dtype=np.int8)
    arbitrary = np.zeros((15, 20), dtype=np.int8)
    arbitrary[0:9, 0:1] = tall
    arbitrary[0:9, 5:6] = tall
    arbitrary[12:13, 10:17] = other
    assert np.array_equal(solve(arbitrary), np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))


def self_test():
    """Bounded 42-seed survey, analytic fixtures, counts, and determinism."""
    _analytic_checks()
    shapes, backgrounds, inks, matrices, profiles = set(), set(), set(), set(), set()
    train_k, test_k, singleton_indices, sizes = set(), set(), set(), set()
    puzzles, fixed, variable, edge, hollow, monochrome, multicolor = set(), 0, 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        roles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                f = _features(inp)
                roles.add((f["bg"], f["ink"]))
                (train_k if split == "train" else test_k).add(f["families"])
                shapes.add(inp.shape)
                backgrounds.add(f["bg"])
                inks.add(f["ink"])
                matrices.add((f["n"], tuple(int(x) for x in np.argmax(out == f["ink"], axis=1))))
                singleton_indices.add((f["n"], f["singleton"]))
                profiles.add((f["n"], tuple(sorted(len(r["cells"]) for r in f["records"]))))
                for r in f["records"]:
                    sizes.add(len(r["cells"]))
                    colors = len(np.unique(r["crop"][r["crop"] != f["bg"]]))
                    monochrome += colors == 1
                    multicolor += colors > 1
                edge += bool(np.any(inp[0] != f["bg"]) or np.any(inp[-1] != f["bg"])
                             or np.any(inp[:, 0] != f["bg"]) or np.any(inp[:, -1] != f["bg"]))
                hollow += f["hole"]
        fixed += len(roles) == 1
        variable += len(roles) > 1
        if seed in (0, 13, 41):
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
    assert len(puzzles) == 42
    assert train_k == {1, 2, 3, 4} and {3, 4, 5} <= test_k
    assert len(shapes) >= 35 and len(profiles) >= 70 and len(matrices) >= 55
    assert len(backgrounds) >= 8 and len(inks) >= 8
    assert len(singleton_indices) >= 20 and {1, 2} <= sizes and max(sizes) >= 16
    assert fixed > 0 and variable > 0 and edge > 0 and hollow >= 42
    assert monochrome > 0 and multicolor > 0
    # Every legal explicit count combination, including all upper boundaries.
    count_cases = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(70000 + 10 * a + b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
            count_cases += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {a!r}/{b!r}")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "input_shapes": len(shapes),
            "partner_matrices": len(matrices), "object_size_profiles": len(profiles),
            "train_paired_family_counts": sorted(train_k), "test_paired_family_counts": sorted(test_k),
            "background_colors": sorted(backgrounds), "writing_colors": sorted(inks),
            "singleton_index_cases": len(singleton_indices), "object_sizes": sorted(sizes),
            "fixed_palette_puzzles": fixed, "variable_palette_puzzles": variable,
            "edge_inputs": edge, "inputs_with_crop_background": hollow,
            "legal_count_cases": count_cases}
