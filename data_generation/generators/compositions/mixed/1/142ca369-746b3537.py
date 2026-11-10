"""Stretched ray mosaics: L-directed tile rays, then adjacent-vector collapse.

All semantics are input-visible. No parent module, file, seed, or example is
consulted by solve(). See task_spec.md for the coordinate convention.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
HOST_SEEDS = (0, 360957456, 883630886, 614304156, 330610019, 95815558,
              75683040, 230502641, 66579452, 472383377, 471661971, 963824568,
              275451067, 654903197, 67886545, 292132011, 613209034, 951059949,
              28385012, 48858197, 721949952, 979376988, 547641703, 482614531,
              722494073, 897787432, 851262155, 557416923, 422610220, 480318480,
              207437285, 667551896, 493214911, 547053603, 906622405, 939936485,
              407223275, 206475238, 388710401, 957020667, 913058669, 47420703)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _starts(a, axis):
    """Maximal adjacent runs of WHOLE vectors, never per-line color runs."""
    if axis == 0:
        changed = np.any(a[1:, :] != a[:-1, :], axis=1)
    else:
        changed = np.any(a[:, 1:] != a[:, :-1], axis=0)
    return np.r_[0, np.flatnonzero(changed) + 1].astype(int)


def _collapse(a):
    # Both index sets refer to the same uncompressed array. Removing duplicate
    # vectors cannot remove the only witness that two other vectors differ.
    return a[np.ix_(_starts(a, 0), _starts(a, 1))].copy()


def _expand(a, heights, widths):
    return np.repeat(np.repeat(a, heights, axis=0), widths, axis=1)


def _cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return ((r, c), (r - dr, c), (r, c - dc))


def _parse(a):
    """Decode band coordinates, visible field colors, and monochrome Ls."""
    rows, cols = _starts(a, 0), _starts(a, 1)
    tiles = a[np.ix_(rows, cols)].copy()
    h, w = tiles.shape
    if min(h, w) < 5:
        raise ValueError("the band field must have at least five cells per axis")
    field = (int(tiles[0, 0]), int(tiles[0, 1]))
    if field[0] == field[1]:
        raise ValueError("top edge must expose both field colors")
    base = np.array(field, dtype=np.int8)[np.indices((h, w)).sum(axis=0) % 2]
    if not (np.array_equal(tiles[0], base[0])
            and np.array_equal(tiles[-1], base[-1])
            and np.array_equal(tiles[:, 0], base[:, 0])
            and np.array_equal(tiles[:, -1], base[:, -1])):
        raise ValueError("the input perimeter must be an intact checkerboard field")
    marker = ~np.isin(tiles, field)
    if np.any(tiles[~marker] != base[~marker]):
        raise ValueError("field colors must alternate in both band coordinates")
    remaining = {(int(r), int(c)) for r, c in np.argwhere(marker)}
    sources = []
    while remaining:
        root = min(remaining)
        shade = int(tiles[root])
        todo, component = [root], set()
        remaining.remove(root)
        while todo:
            r, c = todo.pop()
            component.add((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining and tiles[q] == shade:
                    remaining.remove(q)
                    todo.append(q)
        rr = sorted({r for r, _ in component})
        cc = sorted({c for _, c in component})
        if (len(component) != 3 or len(rr) != 2 or len(cc) != 2
                or rr[1] - rr[0] != 1 or cc[1] - cc[0] != 1):
            raise ValueError("each monochrome component must be a three-cell L")
        absent = next(iter({(r, c) for r in rr for c in cc} - component))
        elbow = (rr[0] + rr[1] - absent[0], cc[0] + cc[1] - absent[1])
        direction = (elbow[0] - absent[0], elbow[1] - absent[1])
        sources.append((elbow, direction, shade))
    if not sources:
        raise ValueError("missing L source")
    heights = np.diff(np.r_[rows, a.shape[0]])
    widths = np.diff(np.r_[cols, a.shape[1]])
    return tiles, heights, widths, field, sources


def _paint(tiles, sources):
    """Superimpose straight diagonal rays; collisions have no priority order."""
    result = tiles.copy()
    occupied = {cell for elbow, direction, _ in sources for cell in _cells(elbow, direction)}
    painted, lengths, overlaps = {}, [], 0
    h, w = tiles.shape
    for (r, c), (dr, dc), shade in sources:
        r, c = r + dr, c + dc
        length = 0
        while 0 <= r < h and 0 <= c < w:
            cell = (r, c)
            if cell in occupied:
                raise ValueError("a ray may not strike an original source")
            if cell in painted:
                if painted[cell] != shade:
                    raise ValueError("different-color rays may not share a cell")
                overlaps += 1
            painted[cell] = shade
            result[cell] = shade
            length += 1
            r, c = r + dr, c + dc
        if length < 2:
            raise ValueError("every ray needs at least two visible added cells")
        lengths.append(length)
    return result, {"lengths": tuple(lengths), "overlaps": overlaps,
                    "added_tiles": len(painted)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    tiles, _, _, _, sources = _parse(a)
    painted, _ = _paint(tiles, sources)
    # Painting constant rectangles then collapsing is exactly this computation;
    # the final collapse also handles any new equal neighboring tile vectors.
    return _collapse(painted)


def _thickness(rng, count, mode, irregular=False):
    if mode == "unit":
        values = np.ones(count, dtype=int)
    elif mode == "uniform":
        values = np.full(count, int(rng.integers(2, min(4, 30 // count) + 1)))
    elif mode == "exception":
        values = np.ones(count, dtype=int)
        values[int(rng.integers(count))] = int(rng.integers(2, min(7, 31 - count)))
    else:
        values = rng.integers(1, 5, size=count)
        # At most 3*count decrements, so this is bounded by <=39 iterations.
        while int(values.sum()) > 30:
            choices = np.flatnonzero(values > 1)
            values[int(rng.choice(choices))] -= 1
    if irregular:
        if np.all(values == values[0]):
            values[int(rng.integers(count))] = 1 if values[0] > 1 else 2
    return values


def _repeat_ray_band(rng, sizes, indices):
    """Ensure a ray-bearing band repeats, preserving the 30-pixel budget."""
    active = np.unique(indices)
    if np.any(sizes[active] > 1):
        return
    index = int(rng.choice(active))
    if int(sizes.sum()) == 30:
        # All active bands are unit-sized, so a donor is necessarily outside
        # the ray. Transfer one pixel rather than exceeding the canvas limit.
        donors = np.flatnonzero(sizes > 1)
        sizes[int(rng.choice(donors))] -= 1
    sizes[index] = 2


def _field(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _place(base, sources):
    result = base.copy()
    for elbow, direction, shade in sources:
        for q in _cells(elbow, direction):
            if result[q] not in (base[0, 0], base[0, 1]):
                raise ValueError("sources overlap")
            result[q] = shade
    # Reject touching same-color components, but allow differently colored Ls
    # to share an edge (as in parent A's original many-source example).
    owner = {q: i for i, (e, d, _) in enumerate(sources) for q in _cells(e, d)}
    for q, i in owner.items():
        r, c = q
        for other in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            j = owner.get(other, i)
            if j != i and sources[i][2] == sources[j][2]:
                raise ValueError("same-color sources may not touch")
    return result


def _scene(rng, kind, direction):
    permutation = rng.permutation(10)
    field = tuple(int(x) for x in permutation[:2])
    if kind == "recurrence":
        distance = int(rng.integers(2, 4))
        h, w = (int(rng.integers(distance + 5, distance + 8)) for _ in range(2))
        # Ray toward a nearby corner leaves >=3 untouched row/column bands.
        dr, dc = direction
        elbow = (distance if dr < 0 else h - 1 - distance,
                 distance if dc < 0 else w - 1 - distance)
        sources = [(elbow, direction, int(permutation[2]))]
        base = _field((h, w), field)
        tiles = _place(base, sources)
    else:
        if kind == "target":
            shape = [int(rng.integers(8, 14)), int(rng.integers(8, 14))]
            shape[int(rng.integers(2))] = int(rng.integers(12, 14))
            desired = int(rng.integers(1, 7))
        elif kind == "routes":
            shape = [int(rng.integers(8, 12)), int(rng.integers(8, 12))]
            desired = int(rng.integers(2, 4))
        else:
            shape = [int(rng.integers(6, 12)), int(rng.integers(6, 12))]
            desired = int(rng.integers(1, 6))
        h, w = shape
        base = _field((h, w), field)
        sources, tiles = [], base.copy()
        palette = [int(x) for x in permutation[2:2 + int(rng.integers(2, 7))]]
        candidates = [(r, c, dr, dc) for r in range(1, h - 1)
                      for c in range(1, w - 1) for dr, dc in DIRECTIONS
                      if all(1 <= a < h - 1 and 1 <= b < w - 1
                             for a, b in _cells((r, c), (dr, dc)))]
        rng.shuffle(candidates)
        for index in range(desired):
            chosen = None
            shade = palette[index] if index < 2 else int(rng.choice(palette))
            for r, c, dr, dc in candidates:
                if index == 0 and (dr, dc) != direction:
                    continue
                if kind == "routes" and index == 1 and (dr, dc) == direction:
                    continue
                candidate = ((r, c), (dr, dc), shade)
                trial_sources = sources + [candidate]
                try:
                    trial = _place(base, trial_sources)
                    _, info = _paint(trial, trial_sources)
                    if index == 0 and kind in ("routes", "target") and info["lengths"][0] < 4:
                        continue
                except ValueError:
                    continue
                chosen = trial, trial_sources
                break
            if chosen is None:
                if index < (2 if kind == "routes" else 1):
                    return None
                break
            tiles, sources = chosen
    painted, _ = _paint(tiles, sources)
    modes = ("unit", "uniform", "exception", "mixed")
    if kind in ("recurrence", "routes"):
        heights = _thickness(rng, h, "mixed", irregular=True)
        widths = _thickness(rng, w, "mixed", irregular=True)
    else:
        heights = _thickness(rng, h, modes[int(rng.integers(4))])
        widths = _thickness(rng, w, modes[int(rng.integers(4))])
    added_rows, added_columns = np.nonzero(painted != tiles)
    if kind in ("recurrence", "routes"):
        _repeat_ray_band(rng, heights, added_rows)
        _repeat_ray_band(rng, widths, added_columns)
    elif not (np.any(heights[added_rows] > 1) or np.any(widths[added_columns] > 1)):
        # B must collapse vectors containing A's newly painted values, rather
        # than getting all its evidence from repetitions in untouched field.
        if rng.integers(2):
            _repeat_ray_band(rng, heights, added_rows)
        else:
            _repeat_ray_band(rng, widths, added_columns)
    inp = _expand(tiles, heights, widths)
    # Labels are constructed from the planned sources, independently of the
    # input parser. This is literally paint first, collapse physical vectors.
    output = _collapse(_expand(painted, heights, widths))
    return {"input": inp, "output": output}


def _key(a):
    return tuple(a.shape), np.asarray(a, dtype=np.int8).tobytes()


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    directions = [DIRECTIONS[int(i)] for i in rng.permutation(4)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            kind = ("target" if split == "test" else
                    "recurrence" if index == 0 else "routes" if index == 1 else "free")
            direction = directions[index] if split == "train" and index < 2 else DIRECTIONS[int(rng.integers(4))]
            for _ in range(64):
                pair = _scene(rng, kind, direction)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction could not find a distinct valid scene")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _separated_repeats(a, axis):
    vectors = a if axis == 0 else a.T
    first = {}
    for i, row in enumerate(vectors):
        key = row.tobytes()
        if key in first and i - first[key] > 1:
            return True
        first.setdefault(key, i)
    return False


def _features(inp):
    tiles, heights, widths, field, sources = _parse(inp)
    painted, info = _paint(tiles, sources)
    answer = _collapse(painted)
    added_rows, added_columns = np.nonzero(painted != tiles)
    return {"shape": tuple(inp.shape), "tile_shape": tuple(tiles.shape),
            "field": field, "sources": len(sources),
            "directions": tuple(sorted({d for _, d, _ in sources})),
            "palette": tuple(sorted({c for _, _, c in sources})),
            "heights": tuple(int(x) for x in heights),
            "widths": tuple(int(x) for x in widths),
            "ray_row_repeats": bool(np.any(heights[added_rows] > 1)),
            "ray_column_repeats": bool(np.any(widths[added_columns] > 1)),
            "recurring_rows": _separated_repeats(answer, 0),
            "recurring_columns": _separated_repeats(answer, 1), **info}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    row_only_fails = column_only_fails = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            tiles, heights, widths, field, sources = _parse(inp)
            painted, info = _paint(tiles, sources)
            expanded = _expand(painted, heights, widths)
            assert np.array_equal(out, _collapse(expanded)), "wrong paint-then-collapse answer"
            assert np.array_equal(out, solve(inp)), "input-only solver disagreement"
            assert np.array_equal(_expand(tiles, heights, widths), inp)
            assert not np.array_equal(_collapse(inp), out), "parent A omission is invisible"
            assert not np.array_equal(expanded, out), "parent B omission is invisible"
            added_rows, added_columns = np.nonzero(painted != tiles)
            assert (np.any(heights[added_rows] > 1)
                    or np.any(widths[added_columns] > 1)), "no repeated ray-bearing vectors"
            if split == "train":
                row_only_fails |= not np.array_equal(expanded[_starts(expanded, 0), :], out)
                column_only_fails |= not np.array_equal(expanded[:, _starts(expanded, 1)], out)
            assert len(_starts(out, 0)) == out.shape[0]
            assert len(_starts(out, 1)) == out.shape[1]
            assert set(np.unique(out)) <= set(np.unique(inp))
            # No preserved object or same-color ray crossing is overwritten.
            for e, d, color in sources:
                assert all(painted[q] == color for q in _cells(e, d))
            assert min(info["lengths"]) >= 2
            if split == "train":
                training.append(_features(inp))
    assert len({d for f in training for d in f["directions"]}) >= 2
    assert any(f["sources"] == 1 for f in training)
    assert any(f["sources"] >= 2 and len(f["directions"]) >= 2 for f in training)
    assert any(len(f["palette"]) >= 2 for f in training)
    assert any(f["recurring_rows"] and f["recurring_columns"] for f in training)
    assert any(len(set(f["heights"])) > 1 and len(set(f["widths"])) > 1
               and max(f["heights"]) > 1 and max(f["widths"]) > 1 for f in training)
    assert any(f["ray_row_repeats"] and f["ray_column_repeats"] for f in training)
    assert max(n for f in training for n in f["lengths"]) >= 4
    assert row_only_fails and column_only_fails, "both compression axes need witnesses"


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _regressions():
    # Whole-vector compression: repeated separated rows AND columns survive,
    # including every value at their intersections. Local color RLE is wrong.
    coarse = np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]], dtype=np.int8)
    expanded = _expand(coarse, [2, 1, 3], [1, 4, 2])
    assert np.array_equal(_collapse(expanded), coarse)
    assert np.array_equal(_collapse(expanded.T), coarse.T)
    assert np.array_equal(_collapse(expanded[::-1, ::-1]), coarse[::-1, ::-1])
    # Explicit four-orientation expectations, not generated labels.
    base = _field((7, 8), (0, 5))
    scene = base.copy()
    scene[2, 2] = scene[2, 3] = scene[3, 2] = 8
    expected = scene.copy()
    expected[1, 1] = expected[0, 0] = 8
    for k in range(4):
        a, b = np.rot90(scene, k), np.rot90(expected, k)
        inp = _expand(a, np.arange(a.shape[0]) % 3 + 1, np.arange(a.shape[1]) % 2 + 1)
        assert np.array_equal(solve(inp), _collapse(b))
        if inp.shape[0] <= 15:
            assert np.array_equal(solve(inp), solve(np.repeat(inp, 2, axis=0)))
    # Same-color crossing keeps its value; different colors at the very same
    # tile are invalid. Crossing between tiles remains unproblematic.
    base = _field((9, 9), (0, 5))
    sources = [((2, 2), (1, 1), 8), ((2, 6), (1, -1), 8)]
    tiles = _place(base, sources)
    painted, info = _paint(tiles, sources)
    assert painted[4, 4] == 8 and info["overlaps"] >= 1
    sources[1] = ((2, 6), (1, -1), 7)
    try:
        _paint(_place(base, sources), sources)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting crossing accepted")
    return True


def self_test():
    """42 streamed puzzles; no files, retained runs, trace logs or subprocesses."""
    signatures, structures, shapes, tile_shapes = set(), set(), set(), set()
    counts, directions, fields, thicknesses = set(), set(), set(), set()
    ray_repeats = {"rows": 0, "columns": 0, "both": 0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                f = _features(inp)
                shapes.add(f["shape"])
                tile_shapes.add(f["tile_shape"])
                counts.add(f["sources"])
                directions.update(f["directions"])
                fields.add(f["field"])
                thicknesses.add((f["heights"], f["widths"]))
                ray_repeats["rows"] += f["ray_row_repeats"]
                ray_repeats["columns"] += f["ray_column_repeats"]
                ray_repeats["both"] += f["ray_row_repeats"] and f["ray_column_repeats"]
                mapping, normalized = {}, []
                for c in inp.flat:
                    normalized.append(mapping.setdefault(int(c), len(mapping)))
                structures.add((inp.shape, bytes(normalized)))
                # Retessellating the same scene cannot alter the compact answer.
                tiles, _, _, _, _ = _parse(inp)
                assert np.array_equal(solve(tiles), pair["output"])
                assert np.array_equal(solve(inp.T), pair["output"].T)
        train_size = max(max(_features(p["input"])["tile_shape"]) for p in puzzle["train"])
        assert all(max(_features(p["input"])["tile_shape"]) > train_size
                   for p in puzzle["test"]), "missing logical-dimension shift"
    assert len(signatures) == 42
    assert len(structures) >= 150 and len(shapes) >= 60
    assert len(tile_shapes) >= 30 and len(thicknesses) >= 150
    assert {1, 2, 3, 4} <= counts and directions == set(DIRECTIONS)
    assert len(fields) >= 40
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _regressions()
    # The validator must actually reject an altered, still in-range answer.
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("altered answer accepted")
    return {"valid": True, "seeds": 42, "pairs": 210,
            "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "tile_shapes": len(tile_shapes), "color_normalized_structures": len(structures),
            "source_counts": sorted(counts), "field_color_pairs": len(fields),
            "thickness_layouts": len(thicknesses), "ray_bearing_repetitions": ray_repeats}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
