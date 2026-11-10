"""Falling core samples: area-controlled transport of the hidden substrate.

Only standard-library Python and NumPy are used. Neither solve nor the parser
consults the generator, a seed, a stored puzzle, or an external file.
"""

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a


def _components(mask):
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    answer = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _DIRS:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        answer.append(tuple(sorted(cells)))
    return answer


def _normal(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _read(grid):
    a = _grid(grid)
    if a.shape[1] < 3 or not np.array_equal(a[:, 0], a[:, -1]):
        raise ValueError("Two intact edge columns must specify the terrain")
    terrain = np.repeat(a[:, :1], a.shape[1], axis=1)
    cells = _components(a != terrain)
    for component in cells:
        if len({int(a[p]) for p in component}) != 1:
            raise ValueError("Each marked component must be monochromatic")
    return terrain, cells


def _render(terrain, components, ink_grid=None, displacement="area"):
    """Also supports explicit counterfactuals for training-witness checks."""
    result = terrain.copy()
    occupied = set()
    for cells in components:
        distance = len(cells) if displacement == "area" else int(displacement)
        for r, c in cells:
            q = (r + distance, c)
            if q[0] >= terrain.shape[0]:
                continue
            if q in occupied:
                raise ValueError("Deposits overlap: outside this family's domain")
            occupied.add(q)
            result[q] = terrain[r, c] if ink_grid is None else ink_grid[r, c]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore the row terrain; drop each mask with its sampled source colors."""
    terrain, components = _read(input_grid)
    return _render(terrain, components)


def _facts(grid):
    terrain, components = _read(grid)
    facts = []
    for cells in components:
        area = len(cells)
        survivors = [(r, c) for r, c in cells if r + area < grid.shape[0]]
        visible = [(r, c) for r, c in survivors
                   if terrain[r, c] != terrain[r + area, c]]
        facts.append({
            "cells": cells, "area": area, "shape": _normal(cells),
            "ink": int(grid[cells[0]]), "survivors": len(survivors),
            "visible": len(visible),
            "source_colors": len({int(terrain[p]) for p in cells}),
            "visible_colors": len({int(terrain[p]) for p in visible}),
            "box": (max(r for r, c in cells)-min(r for r, c in cells)+1,
                    max(c for r, c in cells)-min(c for r, c in cells)+1),
        })
    return terrain, facts


def _shape(rng, area, kind=None):
    if kind == "L":
        cells = {(0, 0), (1, 0), (1, 1)}
    elif kind == "square":
        cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    elif kind == "tall":
        # A long backbone with an optional off-axis chip; exact target area.
        cells = {(r, 0) for r in range(area - int(area >= 6))}
        if area >= 6:
            cells.add((int(rng.integers(1, area-2)), 1))
    elif kind == "flat" or rng.random() < .15:
        cells = {(0, c) for c in range(area)}
    elif area == 8 and rng.random() < .12:
        cells = {(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _DIRS}
                              - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if kind not in ("tall", "flat"):
        for _ in range(int(rng.integers(4))):
            cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    return _normal(cells)


def _terrain(rng, height, width, test, bottom_cap=None):
    count = int(rng.integers(5, 8) if test else rng.integers(3, 6))
    # Random composition of the height with at least two rows in every band.
    lengths = np.full(count, 2, dtype=int)
    if bottom_cap is not None:
        lengths[-1] = int(rng.integers(2, min(5, bottom_cap)+1))
    for _ in range(height - int(lengths.sum())):
        lengths[int(rng.integers(count - int(bottom_cap is not None)))] += 1
    colors = [int(x) for x in rng.choice(10, size=int(rng.integers(3, 5)),
                                       replace=False)]
    sequence = []
    for i in range(count):
        # The first three strata have distinct colors; later ones can recur.
        options = (colors if i == 0 else [c for c in colors if c != sequence[-1]])
        if i in (1, 2):
            options = [c for c in options if c not in sequence]
        sequence.append(int(rng.choice(options)))
    rows = np.concatenate([np.full(n, color, dtype=np.int8)
                           for n, color in zip(lengths, sequence)])
    return np.repeat(rows[:, None], width, axis=1)


def _eligible(terrain, shape, mode, test):
    height, width = terrain.shape
    sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
    if sw > width-2 or sh > height:
        return []
    area = len(shape)
    placements = []
    for top in range(height-sh+1):
        rows = [(top+r, c) for r, c in shape]
        source_colors = {int(terrain[r, 0]) for r, c in rows}
        if not test and len(source_colors) > 2:
            continue
        surviving = [(r, c) for r, c in rows if r+area < height]
        visible = [(r, c) for r, c in surviving
                   if terrain[r, 0] != terrain[r+area, 0]]
        visible_colors = {int(terrain[r, 0]) for r, c in visible}
        if mode == "visible" and not (len(surviving) == area and len(visible) == area):
            continue
        if mode == "hidden" and not (len(surviving) == area and not visible):
            continue
        if mode == "mixed" and not (len(surviving) == area and 0 < len(visible) < area):
            continue
        if mode == "texture" and not (len(source_colors) == 2 and
                                      len(surviving) == area and len(visible_colors) == 2):
            continue
        if mode == "triple" and not (len(source_colors) >= 3 and
                                     len(surviving) == area and len(visible_colors) >= 2):
            continue
        if mode == "partial" and not (0 < len(surviving) < area
                                      and len(visible) == len(surviving)):
            continue
        if mode == "gone" and surviving:
            continue
        for left in range(1, width-sw):
            cells = tuple((r, c+left) for r, c in rows)
            dest = tuple((r+area, c) for r, c in cells if r+area < height)
            placements.append((cells, dest))
    return placements


def _make_pair(rng, specs, test=False):
    for _ in range(180):
        height = int(rng.integers(23, 31))
        width = int(rng.integers(max(13, 3*len(specs)+1), 29))
        partial_areas = [area for area, mode, kind in specs if mode == "partial"]
        terrain = _terrain(rng, height, width, test,
                           min(partial_areas) if partial_areas else None)
        entries = []
        for index, (area, mode, kind) in enumerate(specs):
            choices = []
            for _ in range(16):
                shape = _shape(rng, area, kind)
                choices = _eligible(terrain, shape, mode, test)
                if choices:
                    break
            if not choices:
                break
            entries.append((index, choices))
        if len(entries) != len(specs):
            continue
        entries.sort(key=lambda item: len(item[1]))
        used, targets, placed = set(), set(), [None]*len(specs)
        failed = False
        for index, choices in entries:
            accepted = False
            for j in rng.permutation(len(choices)):
                cells, dest = choices[int(j)]
                if (any(p in used or any((p[0]+dr, p[1]+dc) in used
                                        for dr, dc in _DIRS) for p in cells)
                        or any(p in targets for p in dest)):
                    continue
                used.update(cells)
                targets.update(dest)
                placed[index] = cells
                accepted = True
                break
            if not accepted:
                failed = True
                break
        if failed:
            continue
        a = terrain.copy()
        # Marker colors are local roles. Reuse is deliberate, not a color code.
        unused = [c for c in range(10) if c not in set(terrain[:, 0].tolist())]
        shared = int(rng.choice(unused))
        for cells in placed:
            touched = {int(terrain[p]) for p in cells}
            options = [c for c in range(10) if c not in touched]
            ink = shared if rng.random() < .75 else int(rng.choice(options))
            for p in cells:
                a[p] = ink
        out = _render(terrain, placed)
        if not np.any(out != terrain):
            continue
        return {"input": a, "output": out}
    raise RuntimeError(f"Failed to place core samples: test={test}, specs={specs}")


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    max_area = int(rng.integers(6, 10))
    # Two compact sets of witnesses remain available even at the 2/1 minimum.
    anchors = [
        [(max_area, "texture", None), (1, "visible", None),
         (2, "hidden", "flat"), (3, "mixed", "tall")],
        [(3, "visible", "L"), (4, "visible", "square"),
         (int(rng.integers(2, 6)), "partial", "tall")],
    ]
    anchor_slots = [int(x) for x in rng.choice(num_train, size=2, replace=False)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(70):
                if split == "train" and index in anchor_slots:
                    specs = list(anchors[anchor_slots.index(index)])
                else:
                    n = int(rng.integers(2, 7))
                    specs = []
                    if split == "test":
                        specs.append((max_area, "triple", "tall"))
                    else:
                        specs.append((int(rng.integers(1, max_area+1)), "visible", None))
                    for _ in range(n-1):
                        area = int(rng.integers(1, max_area+1))
                        mode = str(rng.choice(["any", "any", "visible", "hidden",
                                               "partial", "gone", "mixed"]))
                        if mode in ("partial", "mixed") and area == 1:
                            mode = "visible"
                        specs.append((area, mode, None))
                rng.shuffle(specs)
                pair = _make_pair(rng, specs, split == "test")
                terrain, facts = _facts(pair["input"])
                # Ensure the same-ink/different-area witness in the mechanics demo.
                if split == "train" and index == anchor_slots[1]:
                    ink = next(c for c in range(10) if c not in set(terrain[:, 0].tolist()))
                    for f in facts:
                        for p in f["cells"]:
                            pair["input"][p] = ink
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Failed to produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "schema")
    _check(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pair lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _check(a.shape == out.shape, "shape changed")
            k = _key(a)
            _check(k not in seen, "duplicate input")
            seen.add(k)
            terrain, facts = _facts(a)
            rows = terrain[:, 0]
            cuts = [0] + [r for r in range(1, len(rows)) if rows[r] != rows[r-1]] + [len(rows)]
            _check(all(b-a >= 2 for a, b in zip(cuts, cuts[1:])), "thin terrain stratum")
            _check((3 <= len(cuts)-1 <= 5) if split == "train" else
                   (5 <= len(cuts)-1 <= 7), "stratum count")
            _check(len({int(rows[r]) for r in cuts[:3]}) == 3, "first three source strata")
            _check(3 <= len(set(rows.tolist())) <= 4, "terrain palette size")
            _check(2 <= len(facts) <= 6, "object count")
            _check(all(1 <= f["area"] <= 9 for f in facts), "area bound")
            _check(np.array_equal(out, solve(a)), "incorrect substrate transport")
            _check(np.any(out != terrain), "no visible deposit")
            _check(not np.array_equal(a, out), "identity pair")
            _check(np.array_equal(a[:, [0, -1]], out[:, [0, -1]]), "edge references changed")
            _check(set(np.unique(out)) <= set(terrain[:, 0].tolist()), "output retained marker ink")
            if split == "train":
                _check(all(f["source_colors"] <= 2 for f in facts), "train texture range")
            else:
                _check(any(f["source_colors"] >= 3 and f["visible_colors"] >= 2
                           and f["survivors"] == f["area"] for f in facts), "missing test shift")
            records[split].append(facts)
    train = [f for fs in records["train"] for f in fs]
    test = [f for fs in records["test"] for f in fs]
    _check({1, 2, 3, 4} <= {f["area"] for f in train}, "insufficient area evidence")
    _check(max(f["area"] for f in test) <= max(f["area"] for f in train), "unshown distance")
    _check(any(f["visible_colors"] == 2 for f in train), "missing source-color sampling")
    _check(any(f["visible"] == 0 and f["survivors"] == f["area"] for f in train), "missing camouflage")
    _check(any(0 < f["visible"] < f["survivors"] == f["area"] for f in train), "missing partial camouflage")
    _check(any(0 < f["survivors"] < f["area"] and f["visible"] == f["survivors"]
               for f in train), "missing visible bottom clipping")
    _check(any(f["box"] == g["box"] == (2, 2) and {f["area"], g["area"]} == {3, 4}
               and f["visible"] == f["area"] and g["visible"] == g["area"]
               for fs in records["train"] for f in fs for g in fs), "missing box/area contrast")
    _check(any(f["ink"] == g["ink"] and f["area"] != g["area"]
               for fs in records["train"] for f in fs for g in fs), "missing ink contrast")
    # Actual output contrasts, not just the existence of named features.
    for alternative in ("ink", "stationary", "one_step", "single_base"):
        witnessed = False
        for pair in puzzle["train"]:
            a = np.asarray(pair["input"])
            terrain, components = _read(a)
            if alternative == "ink":
                predicted = _render(terrain, components, ink_grid=a)
            elif alternative == "single_base":
                predicted = terrain.copy()
                for cells in components:
                    for r, c in cells:
                        if r+len(cells) < a.shape[0]:
                            predicted[r+len(cells), c] = terrain[cells[0]]
            else:
                try:
                    predicted = _render(terrain, components,
                                        displacement=0 if alternative == "stationary" else 1)
                except ValueError:
                    witnessed = True
                    continue
            witnessed |= not np.array_equal(predicted, pair["output"])
        _check(witnessed, f"training does not reject {alternative}")


def self_test():
    """Bounded diversity, invariants, determinism, counterfactuals and API checks."""
    signatures, dimensions, strata, shapes, layouts, palettes = (set() for _ in range(6))
    areas, counts, behaviors, inks = set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = tuple(_key(pair[k]) for s in ("train", "test")
                          for pair in p[s] for k in ("input", "output"))
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                terrain, facts = _facts(a)
                dimensions.add(a.shape)
                rows = terrain[:, 0].tolist()
                edges = tuple(i for i in range(1, len(rows)) if rows[i] != rows[i-1])
                strata.add(edges)
                palettes.add(tuple(sorted(set(rows))))
                layouts.add((a.shape, edges, tuple(f["cells"] for f in facts)))
                counts.add(len(facts))
                for f in facts:
                    shapes.add(f["shape"])
                    areas.add(f["area"])
                    inks.add(f["ink"])
                    behaviors.add((f["source_colors"], f["area"], f["survivors"], f["visible"]))
    _check(len(signatures) == 42, "seed diversity")
    _check(len(dimensions) >= 25 and len(strata) >= 80, "geometry diversity")
    _check(len(shapes) >= 50 and len(layouts) >= 180, "shape/placement diversity")
    _check(areas == set(range(1, 10)) and counts == set(range(2, 7)), "area/count diversity")
    _check(len(behaviors) >= 50 and len(palettes) >= 25 and len(inks) == 10,
           "visibility or palette diversity")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(42000+nt*10+ns, nt, ns))
    p, q = generate_puzzle(123456, 6, 3), generate_puzzle(123456, 6, 3)
    _check(all(np.array_equal(a[k], b[k]) for s in ("train", "test")
               for a, b in zip(p[s], q[s]) for k in ("input", "output")), "determinism")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Small literal expectations exercise solve outside sampled canvas sizes.
    terrain = np.repeat(np.array([2]*4 + [3]*3 + [2]*2 + [4]*3)[:, None], 8, axis=1)
    src, expected = terrain.copy(), terrain.copy()
    for p in ((3, 2), (4, 2), (4, 3), (1, 5), (10, 5), (11, 5)):
        src[p] = 8
    expected[6, 2] = 2
    expected[7, 2] = expected[7, 3] = 3
    _check(np.array_equal(solve(src.tolist()), expected), "texture/camouflage/full-clip fixture")
    src, expected = terrain.copy(), terrain.copy()
    src[2, 1:5] = 8
    src[6:8, 2] = 8
    expected[6, 1:5] = 2
    expected[8, 2], expected[9, 2] = 3, 2
    _check(np.array_equal(solve(src), expected), "simultaneous vacated-cell fixture")
    return {"seeds": 42, "puzzles": len(signatures), "dimensions": len(dimensions),
            "stratum_layouts": len(strata), "shapes": len(shapes),
            "placements": len(layouts), "visibility_signatures": len(behaviors)}
