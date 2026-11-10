"""Cylindrical stencil completion: duplicated seed layouts define a seam."""

from collections import Counter
from itertools import combinations
import numpy as np


ORTHOGONAL = ((-1, 0), (0, -1), (0, 1), (1, 0))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r, c) != (0, 0))
SPANNING_SUBSETS = {
    n: tuple(offsets for offsets in combinations(NEIGHBORS, n)
             if all({p[a] for p in offsets} >= {-1, 1} for a in (0, 1)))
    for n in range(3, 8)
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(np.all((a >= 0) & (a <= 9))), "integer ARC colors required")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _render(macro, tile_h, tile_w, divider):
    h, w = macro.shape
    grid = np.full((h * (tile_h + 1) - 1, w * (tile_w + 1) - 1),
                   divider, dtype=np.int8)
    for r in range(h):
        for c in range(w):
            grid[r * (tile_h + 1):r * (tile_h + 1) + tile_h,
                 c * (tile_w + 1):c * (tile_w + 1) + tile_w] = macro[r, c]
    return grid


def _lattice(grid):
    """Full-length divider lines supply both tile dimensions and their color."""
    grid = _grid(grid)
    rr = np.flatnonzero(np.all(grid == grid[:, :1], axis=1))
    cc = np.flatnonzero(np.all(grid == grid[:1, :], axis=0))
    _require(len(rr) > 0 and len(cc) > 0, "missing lattice dividers")
    th, tw = int(rr[0]), int(cc[0])
    _require(th >= 1 and tw >= 1, "invalid tile size")
    _require((grid.shape[0] + 1) % (th + 1) == 0 and
             (grid.shape[1] + 1) % (tw + 1) == 0, "incomplete lattice")
    _require(np.array_equal(rr, np.arange(th, grid.shape[0], th + 1)) and
             np.array_equal(cc, np.arange(tw, grid.shape[1], tw + 1)),
             "irregular lattice")
    divider = int(grid[rr[0], cc[0]])
    _require(bool(np.all(grid[rr, :] == divider)) and
             bool(np.all(grid[:, cc] == divider)), "inconsistent divider")
    macro = grid[::th + 1, ::tw + 1].copy()
    _require(bool(np.all(macro != divider)) and
             np.array_equal(grid, _render(macro, th, tw, divider)),
             "tiles must be uniform and distinct from dividers")
    return macro, th, tw, divider


def _matching_axes(mask):
    return [axis for axis in (0, 1) if mask.shape[axis] % 2 == 0 and
            np.array_equal(*np.split(mask, 2, axis=axis))]


def _stamp(shape, centers, stencil, background, seed, wrap_axes):
    """Simultaneous colored offset constraints; return None for a conflict."""
    out = np.full(shape, background, dtype=np.int8)
    centers = set(centers)
    for center in centers:
        out[center] = seed
    paint = {}
    hits = Counter()
    wrapped, clipped = set(), 0
    for r, c in sorted(centers):
        for (dr, dc), color in stencil.items():
            raw = (r + dr, c + dc)
            if any(not 0 <= raw[a] < shape[a] for a in (0, 1)
                   if a not in wrap_axes):
                clipped += 1
                continue
            dest = tuple(raw[a] % shape[a] if a in wrap_axes else raw[a]
                         for a in (0, 1))
            if dest in centers or (dest in paint and paint[dest] != color):
                return None
            paint[dest] = color
            hits[dest] += 1
            if raw != dest:
                wrapped.add(dest)
    for dest, color in paint.items():
        out[dest] = color
    return out, {"wrapped": wrapped, "clipped": clipped,
                 "shared": {p for p, count in hits.items() if count > 1}}


def _analyze(input_grid):
    """Recover every role, the exact stencil, and the doubled axis from input."""
    macro, th, tw, divider = _lattice(input_grid)
    colors, counts = np.unique(macro, return_counts=True)
    _require(int(np.sum(counts == counts.max())) == 1, "ambiguous background")
    background = int(colors[np.argmax(counts)])
    occupied = set(map(tuple, np.argwhere(macro != background).tolist()))
    candidates = []
    for seed in map(int, colors):
        if seed == background:
            continue
        mask = macro == seed
        centers = set(map(tuple, np.argwhere(mask).tolist()))
        axes = _matching_axes(mask)
        if len(centers) < 4 or len(axes) != 1:
            continue
        supports = occupied - centers
        if not 3 <= len(supports) <= 8:
            continue
        axis = axes[0]
        shape = list(macro.shape)
        shape[axis] //= 2
        shape = tuple(shape)
        base_centers = {tuple(p[a] % shape[a] if a == axis else p[a]
                              for a in (0, 1)) for p in centers}
        for prototype in sorted(centers):
            if not all(0 < prototype[a] < macro.shape[a] - 1 for a in (0, 1)):
                continue
            if any(max(abs(p[a] - prototype[a]) for a in (0, 1)) != 1
                   for p in supports):
                continue
            stencil = {(r - prototype[0], c - prototype[1]): int(macro[r, c])
                       for r, c in sorted(supports)}
            result = _stamp(shape, base_centers, stencil, background, seed, {axis})
            if result is None:
                continue
            output, events = result
            # All the visible teaching cells survive their identification.
            if any(output[tuple(p[a] % shape[a] if a == axis else p[a]
                                for a in (0, 1))] != macro[p] for p in supports):
                continue
            candidates.append(dict(macro=macro, tile_h=th, tile_w=tw,
                                   divider=divider, background=background,
                                   seed=seed, centers=centers,
                                   base_centers=base_centers, axis=axis,
                                   prototype=prototype, stencil=stencil,
                                   output=output, **events))
    _require(len(candidates) == 1, "input must determine one prototype and seed role")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _analyze(input_grid)
    return _render(a["output"], a["tile_h"], a["tile_w"], a["divider"])


def _counterfactuals(a):
    shape = a["output"].shape
    args = (a["base_centers"], a["stencil"], a["background"], a["seed"])
    flat = _stamp(a["macro"].shape, a["centers"], a["stencil"],
                  a["background"], a["seed"], set())
    # Flat completion is conflict-free whenever the cylindrical version is.
    _require(flat is not None, "inconsistent flat stencil")
    crops = np.split(flat[0], 2, axis=a["axis"])
    unwrapped = _stamp(shape, *args, set())
    torus = _stamp(shape, *args, {0, 1})
    skeleton = np.full(shape, a["background"], dtype=np.int8)
    for p in a["base_centers"]:
        skeleton[p] = a["seed"]
    # Fold the visible supports without copying them: strongest B-only baseline.
    projected = skeleton.copy()
    for p in np.argwhere((a["macro"] != a["background"]) &
                         (a["macro"] != a["seed"])):
        q = tuple(int(p[d]) % shape[d] if d == a["axis"] else int(p[d])
                  for d in (0, 1))
        projected[q] = a["macro"][tuple(p)]
    return {
        "both_flat_crops_wrong": all(not np.array_equal(crop, a["output"])
                                      for crop in crops),
        "no_wrap_wrong": unwrapped is None or not np.array_equal(unwrapped[0], a["output"]),
        "both_axes_wrap_wrong": torus is None or not np.array_equal(torus[0], a["output"]),
        "no_stencil_wrong": not np.array_equal(projected, a["output"]),
    }


def _motif(rng, kind):
    if kind == "simple":
        offsets = list(ORTHOGONAL if rng.random() < .5 else NEIGHBORS)
        ncolors = 1
    else:
        if kind == "asymmetric" or rng.random() < .65:
            options = SPANNING_SUBSETS[int(rng.integers(3, 8))]
            offsets = list(options[int(rng.integers(len(options)))])
        else:
            offsets = list(ORTHOGONAL if rng.random() < .5 else NEIGHBORS)
        ncolors = int(rng.integers(2, 4)) if kind == "asymmetric" else int(rng.integers(1, 4))
    palette = [int(x) for x in rng.permutation(10)[:3 + ncolors]]
    background, divider, seed = palette[:3]
    assignments = rng.integers(0, ncolors, len(offsets))
    for i, j in enumerate(rng.permutation(len(offsets))[:ncolors]):
        assignments[int(j)] = i
    stencil = {p: palette[3 + int(i)] for p, i in zip(offsets, assignments)}
    return background, divider, seed, stencil


def _shape(rng, axis, dense, nested):
    # Coordinates here are (transverse, periodic), then oriented on the canvas.
    if nested:
        period, along_tile = 6, 1
    elif dense:
        along_tile = int(rng.choice([1, 1, 1, 2]))
        period = int(rng.integers(4, 31 // (along_tile + 1) // 2 + 1))
    else:
        along_tile = int(rng.integers(1, 4))
        period = int(rng.integers(3, 31 // (along_tile + 1) // 2 + 1))
    across_tile = int(rng.integers(1, 4))
    maximum = 31 // (across_tile + 1)
    minimum = 7 if dense else 6
    transverse = int(rng.integers(minimum, maximum + 1))
    if axis == 1:
        return (transverse, period), (across_tile, along_tile)
    return (period, transverse), (along_tile, across_tile)


def _layout(rng, shape, axis, number, stencil, background, seed, nested):
    perpendicular = 1 - axis
    length, period = shape[perpendicular], shape[axis]
    edge = 0 if rng.random() < .5 else length - 1
    interior = int(rng.integers(3, length - 1)) if edge == 0 else int(rng.integers(1, length - 3))

    def orient(across, along):
        return (across, along) if axis == 1 else (along, across)

    if nested:
        centers = {orient(edge, 0), orient(edge, 3),
                   orient(interior, 2), orient(interior, 5)}
    else:
        flip = bool(rng.integers(2))
        centers = {orient(edge, period - 1 if flip else 0),
                   orient(interior, 0 if flip else period - 1)}
    if _stamp(shape, centers, stencil, background, seed, {axis}) is None:
        return None
    candidates = [tuple(map(int, p)) for p in np.ndindex(shape) if p not in centers]
    rng.shuffle(candidates)
    for p in candidates:
        if len(centers) >= number:
            break
        if _stamp(shape, centers | {p}, stencil, background, seed, {axis}) is not None:
            centers.add(p)
    return centers if len(centers) == number else None


def _make_pair(rng, axis, number, kind, nested=False):
    for _ in range(500):
        background, divider, seed, stencil = _motif(rng, kind)
        shape, tiles = _shape(rng, axis, number >= 5, nested)
        centers = _layout(rng, shape, axis, number, stencil, background, seed, nested)
        if centers is None:
            continue
        full_shape = list(shape)
        full_shape[axis] *= 2
        macro = np.full(full_shape, background, dtype=np.int8)
        full_centers = set()
        for p in centers:
            for copy in (0, 1):
                q = list(p)
                q[axis] += copy * shape[axis]
                full_centers.add(tuple(q))
                macro[tuple(q)] = seed
        prototypes = [p for p in sorted(full_centers) if
                      all(0 < p[d] < full_shape[d] - 1 for d in (0, 1))]
        rng.shuffle(prototypes)
        for prototype in prototypes:
            first = macro.copy()
            legal = True
            for (dr, dc), color in stencil.items():
                p = (prototype[0] + dr, prototype[1] + dc)
                if first[p] != background:
                    legal = False
                    break
                first[p] = color
            if not legal:
                continue
            grid = _render(first, *tiles, divider)
            try:
                a = _analyze(grid)
            except ValueError:
                continue
            if not all(_counterfactuals(a).values()):
                continue
            expected = _stamp(shape, centers, stencil, background, seed, {axis})[0]
            _require(np.array_equal(expected, a["output"]), "construction/inference mismatch")
            return {"input": grid,
                    "output": _render(expected, *tiles, divider)}
    raise RuntimeError("bounded search could not construct an unambiguous cylinder")


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and
             isinstance(value, (int, np.integer)) and low <= int(value) <= high,
             f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    _require(nt > ns, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and
             isinstance(seed, (int, np.integer)), "seed must be an integer")
    rng = np.random.default_rng(seed)
    first_axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for section, count in (("train", nt), ("test", ns)):
        for i in range(count):
            if section == "train" and i == 0:
                axis, number, kind, nested = first_axis, 2, "simple", False
            elif section == "train" and i == 1:
                axis, number, kind, nested = 1 - first_axis, 4, "asymmetric", True
            else:
                axis = int(rng.integers(2))
                number = int(rng.integers(5, 7) if section == "test" else rng.integers(2, 5))
                kind, nested = "mixed", False
            for _ in range(20):
                pair = _make_pair(rng, axis, number, kind, nested)
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[section].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("unable to obtain distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    nt = _count(len(puzzle["train"]), "num_train", 2, 10)
    ns = _count(len(puzzle["test"]), "num_test", 1, 4)
    _require(nt > ns, "train/test counts")
    seen, analyses = set(), {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            first, second = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(first) not in seen, "duplicate input")
            seen.add(_key(first))
            a = _analyze(first)
            actual, th, tw, divider = _lattice(second)
            _require((th, tw, divider) == (a["tile_h"], a["tile_w"], a["divider"]),
                     "lattice conventions changed")
            _require(np.array_equal(actual, a["output"]), "incorrect cylinder completion")
            _require(actual.shape[a["axis"]] * 2 == a["macro"].shape[a["axis"]] and
                     actual.shape[1 - a["axis"]] == a["macro"].shape[1 - a["axis"]],
                     "must remove exactly one seed-layout copy")
            _require(np.array_equal(actual == a["seed"],
                                   np.split(a["macro"] == a["seed"], 2, a["axis"])[0]),
                     "seed layout changed")
            _require(a["wrapped"] and a["clipped"] > 0 and
                     all(_counterfactuals(a).values()), "missing conceptual witnesses")
            _require(not _matching_axes(a["macro"]), "raw input accidentally duplicated")
            analyses[section].append(a)
    train, test = analyses["train"], analyses["test"]
    _require({a["axis"] for a in train} == {0, 1}, "training must expose both axes")
    _require(any(len(set(a["stencil"].values())) > 1 and
                 set(a["stencil"]) not in (set(ORTHOGONAL), set(NEIGHBORS))
                 for a in train), "missing colored asymmetric demonstration")
    _require(any(a["axis"] in _matching_axes(a["output"] == a["seed"]) and
                 a["axis"] in _matching_axes(a["output"]) for a in train),
             "missing one-step versus smallest-period demonstration")
    _require(min(len(a["base_centers"]) for a in test) >
             max(len(a["base_centers"]) for a in train), "missing seed-count shift")


def _same(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(p[name], q[name])
                   for p, q in zip(first[s], second[s]) for name in ("input", "output"))
               for s in ("train", "test"))


def self_test():
    # Hand-written asymmetric-color seam witness, checked in both orientations.
    macro = np.zeros((7, 10), dtype=np.int8)
    for p in ((0, 0), (0, 5), (4, 4), (4, 9)):
        macro[p] = 1
    for p, color in (((3, 4), 2), ((4, 3), 3), ((4, 5), 4), ((5, 4), 5)):
        macro[p] = color
    expected = np.array([[int(c) for c in row] for row in
                         ("14003", "50000", "00000", "00002", "40031", "00005", "00000")],
                        dtype=np.int8)
    for x, y in ((macro, expected), (macro.T, expected.T)):
        assert np.array_equal(solve(_render(x, 2, 1, 9)), _render(y, 2, 1, 9))
    shapes, stencils, roles, normalized, signatures = set(), set(), set(), set(), set()
    axes, counts, tile_shapes, overlap = set(), set(), set(), set()
    prototype_copies, clipped, support_counts = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                a = _analyze(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(a["macro"].shape)
                stencils.add(tuple(sorted(a["stencil"])))
                roles.add((a["background"], a["divider"], a["seed"]))
                axes.add(a["axis"])
                counts.add(len(a["base_centers"]))
                tile_shapes.add((a["tile_h"], a["tile_w"]))
                overlap.add(bool(a["shared"]))
                clipped.add(a["clipped"])
                support_counts.add(len(set(a["stencil"].values())))
                prototype_copies.add(a["prototype"][a["axis"]] // a["output"].shape[a["axis"]])
                mapping = {}
                normalized.add((pair["input"].shape, tuple(
                    mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)))
    assert len(signatures) == 42 and len(normalized) >= 180
    assert len(shapes) >= 35 and len(stencils) >= 30 and len(roles) >= 100
    assert axes == {0, 1} and counts == {2, 3, 4, 5, 6}
    assert len(tile_shapes) >= 7 and overlap == {False, True}
    assert prototype_copies == {0, 1} and support_counts == {1, 2, 3} and len(clipped) >= 4
    # All legal explicit count combinations, and type/range boundaries.
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(10000 + train * 10 + test, train, test)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _same(generate_puzzle(314159, 10, 4), generate_puzzle(314159, 10, 4))
    assert _same(generate_puzzle(0), generate_puzzle(0))
    # Solving and validation must not depend on ndarray storage or random state.
    p = generate_puzzle(2718, 2, 1)
    as_lists = {s: [{k: v.tolist() for k, v in q.items()} for q in p[s]] for s in p}
    validate_generated_puzzle(as_lists)
    # Input-only geometry and literal color invariance, with no generator state.
    permutation = np.array([7, 2, 9, 0, 5, 1, 8, 4, 6, 3], dtype=np.int8)
    for section in ("train", "test"):
        for pair in p[section]:
            x, y = pair["input"], pair["output"]
            for turns in (1, 2, 3):
                assert np.array_equal(solve(np.rot90(x, turns)), np.rot90(y, turns))
            assert np.array_equal(solve(x[:, ::-1]), y[:, ::-1])
            assert np.array_equal(solve(permutation[x]), permutation[y])


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal explicit counts)")
