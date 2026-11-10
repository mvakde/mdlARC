"""Restore a scene by stable column deduplication, then exchange arrangements.

Only NumPy and the standard library are used. Geometry is read after compaction;
no generation metadata, parent module, stored grid, or seed is used by solve.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape):
        raise ValueError("a grid must be two-dimensional with dimensions 1..30")
    if not np.issubdtype(a.dtype, np.integer) or not np.all((0 <= a) & (a <= 9)):
        raise ValueError("grid cells must be integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _column_keys(a):
    return [tuple(map(int, a[:, j])) for j in range(a.shape[1])]


def _compact(a):
    seen, indices = set(), []
    for j, key in enumerate(_column_keys(a)):
        if key not in seen:
            seen.add(key)
            indices.append(j)
    return a[:, indices].copy()


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        stack.append(q)
        result.append(frozenset(component))
    return tuple(sorted(result, key=min))


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        raise ValueError("only horizontal, vertical and 45-degree edges are allowed")
    n = max(abs(dr), abs(dc))
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return frozenset((a[0] + i * sr, a[1] + i * sc) for i in range(n + 1))


def _figure(centers):
    return frozenset().union(*(_line(a, b) for a, b in combinations(centers, 2)))


def _stamps(centers, motif):
    return frozenset((r + dr, c + dc) for r, c in centers for dr, dc in motif)


def _separated(first, second):
    expanded = {(r + dr, c + dc) for r, c in first
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    return expanded.isdisjoint(second)


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _direction(endpoints):
    a, b = endpoints
    return "H" if a[0] == b[0] else ("V" if a[1] == b[1] else "D")


def _motif_geometry(cells):
    parts = _components(cells)
    if len(parts) not in (2, 3):
        raise ValueError("need two or three disconnected glyph copies")
    centers, motifs = [], []
    for part in parts:
        lo = tuple(min(p[k] for p in part) for k in (0, 1))
        hi = tuple(max(p[k] for p in part) for k in (0, 1))
        if any((lo[k] + hi[k]) % 2 for k in (0, 1)):
            raise ValueError("glyph bounding boxes must have odd dimensions")
        center = tuple((lo[k] + hi[k]) // 2 for k in (0, 1))
        motif = frozenset((r - center[0], c - center[1]) for r, c in part)
        if (0, 0) not in motif:
            raise ValueError("the bounding-box center must belong to the glyph")
        centers.append(center)
        motifs.append(motif)
    if any(m != motifs[0] for m in motifs):
        raise ValueError("glyphs must be unchanged translations")
    centers = tuple(sorted(centers))
    _figure(centers)
    if len(centers) == 3:
        a, b, c = centers
        if (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0]):
            raise ValueError("three centers must form a nondegenerate triangle")
    return centers, motifs[0]


def _analyze(panel):
    """Recognize roles by geometry on a compacted, uniquely columned panel."""
    values, counts = np.unique(panel, return_counts=True)
    if len(values) != 3 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("need three colors and a uniquely most frequent background")
    background = int(values[int(np.argmax(counts))])
    foreground = [int(v) for v in values if v != background]
    candidates = []
    for edge, glyph in (foreground, foreground[::-1]):
        edge_cells = frozenset(map(tuple, np.argwhere(panel == edge).tolist()))
        glyph_cells = frozenset(map(tuple, np.argwhere(panel == glyph).tolist()))
        if len(edge_cells) < 4:
            continue
        # Extremal cells of any permitted straight segment are lexicographic
        # extrema, including a descending diagonal.
        endpoints = (min(edge_cells), max(edge_cells))
        try:
            if _line(*endpoints) != edge_cells:
                continue
            centers, motif = _motif_geometry(glyph_cells)
            if not _separated(edge_cells, glyph_cells):
                continue
            moved = _stamps(endpoints, motif)
            figure = _figure(centers)
            if not _inside(moved, panel.shape) or len(_components(moved)) != 2:
                continue
            if not _separated(figure, moved):
                continue
        except ValueError:
            continue
        candidates.append({"background": background, "edge": edge, "glyph": glyph,
                           "edge_cells": edge_cells, "glyph_cells": glyph_cells,
                           "endpoints": endpoints, "centers": centers, "motif": motif})
    if len(candidates) != 1:
        raise ValueError("the compacted scene must have exactly one role assignment")
    return candidates[0]


def _exchange(panel, info):
    out = np.full(panel.shape, info["background"], dtype=np.int8)
    for r, c in _figure(info["centers"]):
        out[r, c] = info["edge"]
    for r, c in _stamps(info["endpoints"], info["motif"]):
        out[r, c] = info["glyph"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The sole transformation: first-occurrence columns, then role exchange."""
    panel = _compact(_grid(input_grid))
    return _exchange(panel, _analyze(panel))


# Catalog entries are construction conveniences, never solver lookup keys.
_MOTIFS = {
    "dot": frozenset({(0, 0)}),
    "plus": frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)}),
    "x": frozenset({(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)}),
    "arrow": frozenset({(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1)}),
    "hook": frozenset({(-1, -1), (0, -1), (0, 0), (0, 1), (1, 1)}),
    "flag": frozenset({(-1, -1), (-1, 0), (-1, 1), (0, 0), (0, 1), (1, 0)}),
    "tall_arrow": frozenset({(-2, 0), (-1, 0), (0, -1), (0, 0), (0, 1),
                              (1, 0), (1, 1), (2, 1)}),
    "wide_skew": frozenset({(-1, -2), (-1, -1), (-1, 1), (0, -1), (0, 0),
                             (0, 1), (1, 1), (1, 2)}),
}
_ASYMMETRIC = tuple(name for name, m in _MOTIFS.items()
                    if frozenset((-r, -c) for r, c in m) != m)


def _radii(motif):
    return max(abs(r) for r, c in motif), max(abs(c) for r, c in motif)


def _translate(points, motif, shape, rng, edge_bias=False):
    rh, rw = _radii(motif)
    low_r, high_r = rh - min(r for r, c in points), shape[0] - 1 - rh - max(r for r, c in points)
    low_c, high_c = rw - min(c for r, c in points), shape[1] - 1 - rw - max(c for r, c in points)
    if high_r < low_r or high_c < low_c:
        return None
    dr = int(rng.integers(low_r, high_r + 1))
    if edge_bias and rng.random() < 0.8:
        dc = low_c if rng.random() < 0.5 else high_c
    else:
        dc = int(rng.integers(low_c, high_c + 1))
    return tuple((r + dr, c + dc) for r, c in points)


def _center_layout(rng, count, motif, shape, orientation):
    rh, rw = _radii(motif)
    w = shape[1]
    if orientation in ("H", "V"):
        # Both horizontal extents are occupied by glyphs. A source line can
        # then distinguish the intervening column without adding a barcode.
        distance = w - 2 * rw - 1
        if count == 2:
            relative = ((0, 0), (distance, distance))
        else:
            relative = ((0, 0), (0, distance), (distance, 0))
    else:
        if rw == 2:
            distance = w - 5
        else:
            distance = int(rng.integers(4, min(10, shape[0] - 2 * rh)))
        if count == 2:
            direction = ((0, 1), (1, 0), (1, 1), (1, -1))[int(rng.integers(4))]
            if rw == 2:
                direction = (1, 1)
            relative = ((0, 0), (distance * direction[0], distance * direction[1]))
        elif rw < 2 and rng.random() < 0.3:
            relative = ((0, 0), (distance, distance), (distance, -distance))
        else:
            relative = ((0, 0), (0, distance), (distance, 0))
    # Reflect arrangements independently of motif appearance.
    sr, sc = (1 if rng.random() < 0.5 else -1), (1 if rng.random() < 0.5 else -1)
    relative = tuple((sr * r, sc * c) for r, c in relative)
    return _translate(relative, motif, shape, rng, edge_bias=True)


def _canonical(rng, count, name, orientation):
    motif = _MOTIFS[name]
    rh, rw = _radii(motif)
    for _ in range(900):
        if orientation in ("H", "V"):
            width = 3 if rw == 0 else (7 if rw == 1 else int(rng.integers(9, 12)))
        else:
            width = int(rng.integers(5 if rw == 0 else (7 if rw == 1 else 9), 14))
        height = int(rng.integers(16 + rh, 31))
        shape = (height, width)
        centers = _center_layout(rng, count, motif, shape, orientation)
        if centers is None:
            continue
        if orientation in ("D", "H"):
            distance = width - 2 * rw - 1
            if distance < 3:
                continue
            r0 = int(rng.integers(rh, height - rh - (distance if orientation == "D" else 0)))
            if orientation == "D":
                endpoints = ((r0, rw), (r0 + distance, width - rw - 1))
                if rng.random() < 0.5:
                    endpoints = tuple((r, width - 1 - c) for r, c in endpoints)
            else:
                endpoints = ((r0, rw), (r0, width - rw - 1))
        else:
            col = width // 2
            distance = int(rng.integers(max(4, 2 * rh + 2), min(13, height - 2 * rh)))
            r0 = int(rng.integers(rh, height - rh - distance))
            endpoints = ((r0, col), (r0 + distance, col))
        original_edge = _line(*endpoints)
        original_glyphs = _stamps(centers, motif)
        if len(_components(original_glyphs)) != count or not _separated(original_edge, original_glyphs):
            continue
        moved = _stamps(endpoints, motif)
        figure = _figure(centers)
        if len(_components(moved)) != 2 or not _separated(figure, moved):
            continue
        panel = np.zeros(shape, dtype=np.int8)
        for r, c in original_edge:
            panel[r, c] = 1
        for r, c in original_glyphs:
            panel[r, c] = 2
        if len(set(_column_keys(panel))) != width:
            continue
        info = _analyze(panel)
        if (info["motif"] != motif or set(info["centers"]) != set(centers)
                or set(info["endpoints"]) != set(endpoints)):
            continue
        # Labels come from the sampled geometry, rather than calling solve.
        target = np.zeros(shape, dtype=np.int8)
        for r, c in figure:
            target[r, c] = 1
        for r, c in moved:
            target[r, c] = 2
        _require(np.array_equal(target, _exchange(panel, info)), "scene recognition changed geometry")
        return panel, target
    raise RuntimeError(f"bounded placement exhausted: {count}/{name}/{orientation}")


def _compatible(name, count, orientation):
    if orientation == "D":
        return True
    motif = _MOTIFS[name]
    rh, rw = _radii(motif)
    if rw == 0 or (count == 3 and rh >= 2):
        return False
    if orientation == "V":
        # On either side of the single source column there must be enough
        # distinct glyph profiles to avoid erasing a legitimate motif column.
        columns = [tuple(sorted(r for r, c in motif if c == j)) for j in range(-rw, rw + 1)]
        return len(set(columns)) == len(columns)
    return True


def _repeat_columns(panel, rng, style, harder):
    width = panel.shape[1]
    counts = [2] * width
    if harder:
        index = int(rng.integers(width))
        counts[index] = int(rng.integers(4, min(6, 32 - 2 * width) + 1))
    else:
        for index in rng.permutation(width)[:int(rng.integers(0, min(width, 30 - 2 * width) + 1))]:
            counts[int(index)] += 1
    if style == "adjacent":
        order = [j for j, n in enumerate(counts) for _ in range(n)]
    else:
        # First appearances encode spatial order, while late appearances do
        # not. Force an early duplicate, then append distant copies in reverse.
        selected = 0 if style == "mixed" else int(rng.integers(width - 1))
        insertion = 1 if style == "mixed" else int(rng.integers(selected + 1, width))
        order = list(range(width))
        order.insert(insertion, selected)
        remaining = [j for j in range(width) for _ in range(counts[j] - order.count(j))]
        if style == "mixed":
            rng.shuffle(remaining)
            order.extend(remaining)
        else:
            order.extend(sorted(remaining, reverse=True))
    return panel[:, order].copy()


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        h.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be an integer in 1..4 below num_train")
    rng = np.random.default_rng(seed)
    counts = [2, 3]
    rng.shuffle(counts)
    counts.extend(int(rng.integers(2, 4)) for _ in range(num_train - 2))
    directions = [str(v) for v in rng.choice(("H", "V", "D"), 2, replace=False)]
    directions.extend(str(rng.choice(("H", "V", "D"))) for _ in range(num_train - 2))
    train_specs = []
    for i, (count, direction) in enumerate(zip(counts, directions)):
        names = [n for n in _MOTIFS if _compatible(n, count, direction)]
        if i == 0:
            names = [n for n in names if n in _ASYMMETRIC]
        train_specs.append((count, str(rng.choice(names)), direction))
    test_specs = []
    for _ in range(num_test):
        _, name, direction = train_specs[int(rng.integers(num_train))]
        options = [n for n in (2, 3) if _compatible(name, n, direction)]
        test_specs.append((int(rng.choice(options)), name, direction))
    puzzle, seen = {"train": [], "test": []}, set()
    first_palette = None
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for i, (count, name, direction) in enumerate(specs):
            for _ in range(12):
                panel, target = _canonical(rng, count, name, direction)
                style = "adjacent" if split == "train" and i % 2 == 0 else ("mixed" if split == "test" else "distant")
                source = _repeat_columns(panel, rng, style, split == "test")
                palette = rng.choice(10, 3, replace=False).astype(np.int8)
                if split == "train" and i == 0:
                    first_palette = palette.copy()
                elif split == "train" and i == 1 and np.array_equal(palette, first_palette):
                    palette = palette[[0, 2, 1]]
                source, target = palette[source], palette[target]
                key = (source.shape, source.tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append({"input": source, "output": target})
                    break
            else:
                raise RuntimeError("could not construct distinct input grids")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, infos = set(), {"train": [], "test": []}
    distant_witness, last_witness = False, False
    max_train_multiplicity, min_test_multiplicity = 0, 30
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            source, target = _grid(pair["input"]), _grid(pair["output"])
            key = (source.shape, source.tobytes())
            _require(key not in seen, "inputs must be distinct")
            seen.add(key)
            panel = _compact(source)
            info = _analyze(panel)
            infos[split].append(info)
            _require(panel.shape == target.shape and panel.shape[1] < source.shape[1], "compacted dimensions")
            keys = _column_keys(source)
            multiplicities = [keys.count(k) for k in set(keys)]
            _require(min(multiplicities) >= 2, "each canonical vector must recur")
            _require(np.array_equal(_exchange(panel, info), target), "arrangement exchange mismatch")
            _require(not np.array_equal(panel, target), "exchange must change scene")
            _require(set(np.unique(source)) == set(np.unique(target)), "palette preservation")
            if split == "train":
                max_train_multiplicity = max(max_train_multiplicity, max(multiplicities))
                _require(not np.array_equal(source[:, :panel.shape[1]], panel), "training must defeat a first-tile crop")
                consecutive = [k for j, k in enumerate(keys) if j == 0 or keys[j - 1] != k]
                distant_witness |= len(consecutive) > panel.shape[1]
                last_order = [k for j, k in enumerate(keys) if k not in keys[j + 1:]]
                last_witness |= last_order != list(dict.fromkeys(keys))
            else:
                min_test_multiplicity = min(min_test_multiplicity, max(multiplicities))
    _require(distant_witness and last_witness, "need nonadjacent and first-occurrence witnesses")
    _require({len(i["centers"]) for i in infos["train"]} == {2, 3}, "train both figures")
    _require(len({_direction(i["endpoints"]) for i in infos["train"]}) >= 2, "train multiple source directions")
    _require(len({(i["background"], i["edge"], i["glyph"]) for i in infos["train"]}) >= 2,
             "train pair-local color roles")
    _require(any(frozenset((-r, -c) for r, c in i["motif"]) != i["motif"] for i in infos["train"]), "train exact asymmetric copying")
    motifs = {i["motif"] for i in infos["train"]}
    directions = {_direction(i["endpoints"]) for i in infos["train"]}
    _require(max_train_multiplicity <= 3 and min_test_multiplicity >= 4,
             "generated tests must shift to higher redundancy")
    for i in infos["test"]:
        _require(i["motif"] in motifs and _direction(i["endpoints"]) in directions, "test geometry must be demonstrated")


def self_test():
    """Bounded 42-seed checks; keeps only hashes and aggregate diagnostics."""
    fingerprints, shapes, motifs, palettes, structures = set(), set(), set(), set(), set()
    orientations = {"train": set(), "test": set()}
    duplicate_output_columns, duplicate_panel_rows, no_a_failures = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                panel = _compact(source)
                info = _analyze(panel)
                _require(np.array_equal(solve(source), target), "input-only solve mismatch")
                # Rebuild stable column selection with a separate expression,
                # and check the two output color supports directly.
                vectors = [tuple(map(int, col)) for col in source.T]
                retained = [v for j, v in enumerate(vectors) if v not in vectors[:j]]
                _require(np.array_equal(np.array(retained, dtype=np.int8).T, panel),
                         "first-occurrence order mismatch")
                edge_support = set(map(tuple, np.argwhere(target == info["edge"]).tolist()))
                glyph_support = set(map(tuple, np.argwhere(target == info["glyph"]).tolist()))
                _require(edge_support == set(_figure(info["centers"])), "center connections")
                _require(glyph_support == set(_stamps(info["endpoints"], info["motif"])), "endpoint copies")
                _require(not np.array_equal(panel, target), "omitting B must change the answer")
                try:
                    raw_info = _analyze(source)
                except ValueError:
                    no_a_failures += 1
                else:
                    _require(not np.array_equal(_exchange(source, raw_info), target),
                             "omitting A must change the answer")
                duplicate_output_columns += len(set(_column_keys(target))) < target.shape[1]
                duplicate_panel_rows += len({tuple(map(int, row)) for row in panel}) < panel.shape[0]
                shapes.add((source.shape, target.shape))
                motifs.add(info["motif"])
                palettes.add((info["background"], info["edge"], info["glyph"]))
                orientations[split].add(_direction(info["endpoints"]))
                normalized = np.full(panel.shape, 0, dtype=np.int8)
                normalized[panel == info["edge"]] = 1
                normalized[panel == info["glyph"]] = 2
                structures.add((panel.shape, normalized.tobytes()))
    _require(len(fingerprints) == 42, "seed diversity")
    _require(len(shapes) >= 50 and len(structures) >= 150, "structural diversity")
    _require(len(motifs) >= 6 and len(palettes) >= 50, "motif and role diversity")
    _require(all(v == {"H", "V", "D"} for v in orientations.values()), "orientation coverage")
    _require(duplicate_output_columns >= 15 and duplicate_panel_rows >= 100,
             "need witnesses against recompressing outputs or both axes")
    _require(no_a_failures >= 100, "raw duplicated geometry should require canonicalization")
    first = generate_puzzle(17)
    _require(_fingerprint(first) == _fingerprint(generate_puzzle(17)), "determinism")
    before = _fingerprint(first)
    for pair in first["train"] + first["test"]:
        solve(pair["input"])
    _require(_fingerprint(first) == before, "solve mutated an input")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * train + test, train, test)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "explicit counts")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # A hand-constructed member beyond the sampler's catalog and width range.
    # It also proves that solve does not consult generation's motif names.
    motif = frozenset((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                      if (r, c) != (-1, -1))
    _require(motif not in _MOTIFS.values(), "custom motif accidentally became catalogued")
    canonical = np.zeros((30, 15), dtype=np.int8)
    expected = np.zeros((30, 15), dtype=np.int8)
    for i in range(13):
        canonical[2 + i, 1 + i] = 1
    for r, c in ((22, 1), (27, 1)):
        for dr, dc in motif:
            canonical[r + dr, c + dc] = 2
    expected[22:28, 1] = 1
    for r, c in ((2, 1), (14, 13)):
        for dr, dc in motif:
            expected[r + dr, c + dc] = 2
    source = np.repeat(canonical, 2, axis=1)
    _require(np.array_equal(solve(source), expected), "noncatalog motif or maximum-width member")
    _require(np.array_equal(solve(9 - source), 9 - expected), "palette equivariance")
    _require(np.array_equal(solve(source[:, ::-1]), expected[:, ::-1]), "reflection equivariance")
    return {"seeds": 42, "puzzles": len(fingerprints), "shape_pairs": len(shapes),
            "canonical_structures": len(structures), "motifs": len(motifs),
            "role_palettes": len(palettes), "duplicate_output_columns": duplicate_output_columns,
            "duplicate_panel_rows": duplicate_panel_rows, "exchange_before_dedup_invalid": no_a_failures}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
