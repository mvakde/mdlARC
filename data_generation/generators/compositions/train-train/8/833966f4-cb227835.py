"""Radial exchanges: paired color transport with extreme king-path footprints.

Only the standard library and NumPy are used. Run this file for bounded checks.
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np


HOST_SEEDS = (
    0, 42372004, 601963214, 452454904, 504258933, 201491271, 755116294,
    971775459, 155674859, 696580957, 92793488, 991642832, 769673276,
    659508762, 493029975, 648240882, 618434480, 422585777, 514195607,
    165251141, 778214150, 392058779, 191994508, 736455609, 969928060,
    196866284, 35304617, 752218859, 889372748, 970242765, 573904776,
    173341246, 502014182, 789808745, 561424906, 86571772, 783100342,
    436835945, 5755846, 347016573, 5294543, 621637774,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("a grid must be a 1..30 by 1..30 integer ARC grid")
    return a.astype(int, copy=False)


def _king(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _cross(hub):
    r, c = hub
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _grouping(points):
    return frozenset((frozenset(points[:2]), frozenset(points[2:])))


def _minimum_pairing_cost(points):
    costs = []
    for j in (1, 2, 3):
        k, m = [i for i in (1, 2, 3) if i != j]
        costs.append(_king(points[0], points[j]) + _king(points[k], points[m]))
    return min(costs)


def _alternative_groups(points, hub, metric):
    if metric == "manhattan":
        key = lambda p: abs(p[0] - hub[0]) + abs(p[1] - hub[1])
    elif metric == "euclidean":
        key = lambda p: (p[0] - hub[0]) ** 2 + (p[1] - hub[1]) ** 2
    elif metric == "reading":
        key = lambda p: p
    else:
        raise ValueError("unknown metric")
    # A tied alternative cannot serve as our decisive counterexample.
    if len({key(p) for p in points}) < 4:
        return None
    return _grouping(sorted(points, key=key))


def _profile(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return min(dr, dc), abs(dr - dc)


def _routes(a, b):
    """The two extreme shortest eight-neighbor routes, excluding endpoints."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dr == 0 or dc == 0 or abs(dr) == abs(dc):
        raise ValueError("exchange endpoints need nonzero unequal offsets")
    sr, sc = (1 if dr > 0 else -1), (1 if dc > 0 else -1)
    diagonal, straight = _profile(a, b)
    axial = (sr, 0) if abs(dr) > abs(dc) else (0, sc)
    diagonal_move = (sr, sc)
    paths = []
    for segments in (((axial, straight), (diagonal_move, diagonal)),
                     ((diagonal_move, diagonal), (axial, straight))):
        p = a
        cells = set()
        for move, count in segments:
            for _ in range(count):
                p = (p[0] + move[0], p[1] + move[1])
                if p != b:
                    cells.add(p)
        assert p == b
        paths.append(cells)
    return paths[0], paths[1]


def _outline_formula(a, b):
    """Check the footprint as four lattice segments, without walking routes."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    sr, sc = (1 if dr > 0 else -1), (1 if dc > 0 else -1)
    d, s = _profile(a, b)
    axial = (sr, 0) if abs(dr) > abs(dc) else (0, sc)
    cells = set()
    for k in range(s + 1):
        cells.add((a[0] + k * axial[0], a[1] + k * axial[1]))
        cells.add((a[0] + d * sr + k * axial[0],
                   a[1] + d * sc + k * axial[1]))
    for k in range(d + 1):
        cells.add((a[0] + k * sr, a[1] + k * sc))
        cells.add((a[0] + s * axial[0] + k * sr,
                   a[1] + s * axial[1] + k * sc))
    return cells - {a, b}


def _interior(a, b):
    """Strict lattice interior of the axial/diagonal parallelogram."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    sr, sc = (1 if dr > 0 else -1), (1 if dc > 0 else -1)
    d, s = _profile(a, b)
    ar, ac = (sr, 0) if abs(dr) > abs(dc) else (0, sc)
    return {(a[0] + i * ar + j * sr, a[1] + i * ac + j * sc)
            for i in range(1, s) for j in range(1, d)}


def _analyze(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 6 or sorted(map(int, counts)) != [1, 1, 1, 1, 5, a.size - 9]:
        raise ValueError("input must have background, a five-cell hub and four colors")
    bg = int(colors[np.argmax(counts)])
    if int(np.max(counts)) <= a.size // 2:
        raise ValueError("background must be a strict majority")
    hub_color = int(colors[np.flatnonzero(counts == 5)[0]])
    hub_cells = {tuple(map(int, p)) for p in np.argwhere(a == hub_color)}
    # A plus has a unique center; no orientation or absolute location is assumed.
    centers = [p for p in hub_cells if _cross(p) == hub_cells]
    if len(centers) != 1:
        raise ValueError("the hub must be a complete unit plus")
    hub = centers[0]
    points = [tuple(map(int, np.argwhere(a == color)[0]))
              for color, count in zip(colors, counts) if count == 1]
    if any(_king(p, hub) < 3 for p in points):
        raise ValueError("stations must be separated from the hub")
    if any(_king(p, q) < 2 for p, q in combinations(points, 2)):
        raise ValueError("stations must be mutually eight-neighbor isolated")
    if len({_king(p, hub) for p in points}) != 4:
        raise ValueError("king-distance ranks must be distinct")
    points.sort(key=lambda p: _king(p, hub))
    outlines = []
    for i in (0, 2):
        paths = _routes(points[i], points[i + 1])
        outline = paths[0] | paths[1]
        if outline & (set(points) - {points[i], points[i + 1]}):
            raise ValueError("an exchange footprint cannot run through a foreign station")
        outlines.append(outline)
    return {"grid": a, "background": bg, "ink": hub_color, "hub": hub,
            "hub_cells": hub_cells, "points": points, "outlines": outlines,
            "overlap": len(outlines[0] & outlines[1]),
            "visible_overlap": len((outlines[0] & outlines[1]) - hub_cells)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all colors, radial partners and footprints solely from the input."""
    scene = _analyze(input_grid)
    a, points = scene["grid"], scene["points"]
    out = a.copy()
    for outline in scene["outlines"]:
        for p in outline:
            out[p] = scene["ink"]
    # Read every payload from the original, so each exchange is simultaneous.
    for i, j in ((0, 1), (2, 3)):
        out[points[i]], out[points[j]] = int(a[points[j]]), int(a[points[i]])
    return out


def _legal_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _sample_scene(rng, kind, avoid_roles=None):
    """Fresh scattered geometry; rejection criteria select visible evidence."""
    for _ in range(50000):
        h, w = map(int, rng.integers(14, 31, size=2))
        hub = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
        points = [tuple(map(int, p)) for p in
                  rng.integers([0, 0], [h, w], size=(4, 2))]
        radii = [_king(p, hub) for p in points]
        if min(radii) < 3 or len(set(radii)) < 4:
            continue
        if any(_king(p, q) < 2 for p, q in combinations(points, 2)):
            continue
        points.sort(key=lambda p: _king(p, hub))
        profiles = [_profile(points[i], points[i + 1]) for i in (0, 2)]
        minimum = 2 if kind in ("clear", "counterexample") else 1
        if any(min(profile) < minimum for profile in profiles):
            continue
        outlines = [set.union(*_routes(points[i], points[i + 1])) for i in (0, 2)]
        if outlines[0] & set(points[2:]) or outlines[1] & set(points[:2]):
            continue
        overlap = len(outlines[0] & outlines[1])
        visible_overlap = len((outlines[0] & outlines[1]) - _cross(hub))
        if kind == "clear" and overlap != 0:
            continue
        if kind == "counterexample" and (not 2 <= overlap <= 3 or visible_overlap < 2):
            continue
        if kind == "extra" and overlap > 3:
            continue
        if kind == "test" and visible_overlap < 4:
            continue
        if kind == "clear":
            occupied = set(points) | _cross(hub) | outlines[0] | outlines[1]
            if any(not (_interior(points[i], points[i + 1]) - occupied) for i in (0, 2)):
                continue
            if any(not (path - _cross(hub)) for i in (0, 2)
                   for path in _routes(points[i], points[i + 1])):
                continue
        grouping = _grouping(points)
        metrics = ("reading",) if kind == "clear" else (
            ("manhattan", "euclidean", "reading") if kind == "counterexample" else ())
        if any(_alternative_groups(points, hub, m) in (None, grouping) for m in metrics):
            continue
        if kind == "counterexample" and (
                _king(points[0], points[1]) + _king(points[2], points[3])
                <= _minimum_pairing_cost(points)):
            continue
        break
    else:
        raise RuntimeError(f"could not sample {kind} evidence within the bounded budget")

    # Colors have no numeric meaning. Both input roles change in the first two
    # demonstrations; all six role colors are otherwise fresh per example.
    for _ in range(1000):
        palette = list(map(int, rng.choice(10, size=6, replace=False)))
        if avoid_roles is not None and (palette[0] == avoid_roles[0]
                                        or palette[1] == avoid_roles[1]):
            continue
        if kind == "clear":
            numeric_order = sorted(range(4), key=lambda i: palette[i + 2])
            if {frozenset(numeric_order[:2]), frozenset(numeric_order[2:])} == {
                    frozenset((0, 1)), frozenset((2, 3))}:
                continue
        break
    else:
        raise RuntimeError("could not choose a palette")
    a = np.full((h, w), palette[0], dtype=int)
    for p in _cross(hub):
        a[p] = palette[1]
    for p, color in zip(points, palette[2:]):
        a[p] = color
    return a


def _key(a):
    return tuple(a.shape), tuple(map(int, a.flat))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _legal_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    used = set()
    first_roles = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            kind = "test" if split == "test" else (
                "clear" if index == 0 else "counterexample" if index == 1 else "extra")
            for _ in range(100):
                a = _sample_scene(rng, kind, first_roles if kind == "counterexample" else None)
                if _key(a) not in used:
                    break
            else:
                raise RuntimeError("could not create distinct inputs")
            used.add(_key(a))
            if kind == "clear":
                scene = _analyze(a)
                first_roles = (scene["background"], scene["ink"])
            puzzle[split].append({"input": a, "output": solve(a)})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    _legal_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "each pair needs exactly input/output")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            scene = _analyze(a)
            records[split].append(scene)
            _require(out.shape == a.shape, "canvas changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            points, ink = scene["points"], scene["ink"]
            formula_outlines = []
            for i in (0, 2):
                p, q = points[i:i + 2]
                cells = _outline_formula(p, q)
                _require(cells == scene["outlines"][i // 2], "route/segment disagreement")
                _require(len(cells) == 2 * (_king(p, q) - 1), "bad perimeter size")
                _require(all(0 <= r < a.shape[0] and 0 <= c < a.shape[1]
                             for r, c in cells), "footprint left the canvas")
                formula_outlines.append(cells)
            footprint = formula_outlines[0] | formula_outlines[1]
            expected = a.copy()
            for p in footprint:
                expected[p] = ink
            for i, j in ((0, 1), (2, 3)):
                expected[points[i]] = a[points[j]]
                expected[points[j]] = a[points[i]]
            _require(np.array_equal(out, expected), "radial exchange rule violated")
            _require(np.array_equal(solve(a), out), "input-only solve disagrees")
            _require(all(out[p] == ink for p in scene["hub_cells"]), "hub moved")
            _require(all(np.count_nonzero(out == a[p]) == 1 for p in points),
                     "a payload was lost or copied into a footprint")
            _require(np.count_nonzero(out != a) ==
                     4 + len(footprint - scene["hub_cells"]), "unexpected changed cells")
            # Concrete concept ablations on EVERY pair, not just a family average.
            no_a = a.copy()
            for p in footprint:
                no_a[p] = ink
            no_b = a.copy()
            for i, j in ((0, 1), (2, 3)):
                no_b[points[i]], no_b[points[j]] = int(a[points[j]]), int(a[points[i]])
            _require(not np.array_equal(out, no_a), "A makes no observable difference")
            _require(not np.array_equal(out, no_b), "B makes no observable difference")

    first, second = records["train"][:2]
    _require(first["overlap"] == 0 and 2 <= second["overlap"] <= 3,
             "missing clear and interacting demonstrations")
    _require(second["visible_overlap"] >= 2,
             "shared ink is hidden inside the precolored hub")
    occupied = (set(first["points"]) | first["hub_cells"]
                | first["outlines"][0] | first["outlines"][1])
    for i in (0, 2):
        a, b = first["points"][i:i + 2]
        _require(bool(_interior(a, b) - occupied), "no visible hollow interior")
        _require(all(path - first["hub_cells"] for path in _routes(a, b)),
                 "an extreme route is hidden by the hub")
    for scene in (first, second):
        _require(all(min(_profile(scene["points"][i], scene["points"][i + 1])) >= 2
                     for i in (0, 2)), "first demonstrations need visible segments")
    _require(first["background"] != second["background"] and first["ink"] != second["ink"],
             "first demonstrations must change background and hub colors")
    for metric in ("manhattan", "euclidean", "reading"):
        alt = _alternative_groups(second["points"], second["hub"], metric)
        _require(alt is not None and alt != _grouping(second["points"]),
                 f"second demonstration fails to distinguish {metric} pairing")
    p = second["points"]
    _require(_king(p[0], p[1]) + _king(p[2], p[3]) > _minimum_pairing_cost(p),
             "minimum-total-distance pairing explains the second demonstration")
    numeric = sorted(first["points"], key=lambda p: int(first["grid"][p]))
    _require(_grouping(numeric) != _grouping(first["points"]),
             "numeric color order accidentally explains first pairing")
    _require(all(s["overlap"] <= 3 for s in records["train"])
             and all(s["visible_overlap"] >= 4 for s in records["test"]),
             "missing overlap-complexity train/test shift")


def _fingerprint(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """42 supplied seeds, all legal counts, ablations and symmetry checks."""
    fingerprints, shapes, layouts, palettes, inks = set(), set(), set(), set(), set()
    profiles, regimes, overlaps, visible_overlaps, edge_sides = set(), set(), set(), set(), set()
    pair_count = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)),
                 "seed does not reproduce the whole puzzle")
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                scene = _analyze(a)
                points, hub = scene["points"], scene["hub"]
                shapes.add(a.shape)
                layouts.add((a.shape, hub, tuple(points)))
                palettes.add(tuple(sorted(map(int, np.unique(a)))))
                inks.add(scene["ink"])
                overlaps.add((split, scene["overlap"]))
                visible_overlaps.add((split, scene["visible_overlap"]))
                for i in (0, 2):
                    p, q = sorted(points[i:i + 2])
                    profiles.add(_profile(p, q))
                    regimes.add((abs(q[0] - p[0]) > abs(q[1] - p[1]), q[1] > p[1]))
                for r, c in points:
                    if r == 0:
                        edge_sides.add("top")
                    if r == a.shape[0] - 1:
                        edge_sides.add("bottom")
                    if c == 0:
                        edge_sides.add("left")
                    if c == a.shape[1] - 1:
                        edge_sides.add("right")
                pair_count += 1
        # Rotations/reflections test unseen geometries beyond the sampler's filters.
        pair = puzzle["test"][0]
        for k in range(4):
            for reflect in (False, True):
                a, expected = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
                if reflect:
                    a, expected = np.fliplr(a), np.fliplr(expected)
                _require(np.array_equal(solve(a), expected), "D4 symmetry failed")
        permutation = np.random.default_rng(seed + 17).permutation(10)
        _require(np.array_equal(solve(permutation[pair["input"]]),
                                permutation[pair["output"]]), "color equivariance failed")
        occupied = np.argwhere(pair["input"] != _analyze(pair["input"])["background"])
        lo, hi = occupied.min(axis=0), occupied.max(axis=0) + 1
        crop = (slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])))
        _require(np.array_equal(solve(pair["input"][crop]), pair["output"][crop]),
                 "solving depends on blank margins")
        before = pair["input"].copy()
        solve(pair["input"])
        _require(np.array_equal(before, pair["input"]), "solve mutated its input")

    _require(len(fingerprints) == 42, "puzzle diversity collapsed")
    _require(len(shapes) >= 70 and len(layouts) == pair_count,
             "insufficient geometric variation")
    _require(len(palettes) >= 60 and len(inks) == 10, "palette-role variation collapsed")
    _require(len(profiles) >= 60 and len(regimes) == 4, "slope variation collapsed")
    _require(edge_sides == {"top", "bottom", "left", "right"}, "edge cases missing")
    _require({("train", 0), ("train", 2), ("train", 3)} <= overlaps
             and len({n for split, n in overlaps if split == "test"}) >= 5,
             "overlap geometry is too narrow")
    legal = 0
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            puzzle = generate_puzzle(8100 + 10 * a + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b),
                     "explicit counts were not honored")
            legal += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (4.0, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(9, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(fingerprints),
            "grid_shapes": len(shapes), "spatial_layouts": len(layouts),
            "palettes": len(palettes), "hub_colors": len(inks),
            "diagonal_straight_profiles": len(profiles), "directions": len(regimes),
            "overlaps": {s: sorted(n for t, n in overlaps if t == s)
                         for s in ("train", "test")},
            "visible_overlaps": {s: sorted(n for t, n in visible_overlaps if t == s)
                                 for s in ("train", "test")},
            "legal_count_combinations": legal, "symmetry_checks": 42 * 8,
            "crop_checks": 42,
            "concept_ablations": pair_count * 2}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
