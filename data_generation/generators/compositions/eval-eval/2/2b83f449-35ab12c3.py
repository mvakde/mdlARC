"""Sparse ribbon switches: the copy vector also defines terminal attraction.

Only standard library and NumPy are used. No parent generator is imported.
All diagnostics are bounded and computed in memory; self_test writes no files.
"""

from __future__ import annotations

import hashlib
import itertools

import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=False)


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), "orthogonal nonzero link required")
    n = max(abs(dr), abs(dc))
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return [(a[0] + sr * k, a[1] + sc * k) for k in range(n + 1)]


def _source(points):
    """Reconstruct an open column-monotone rectilinear chain from its corners.

    Every interior landmark column contains the two ends of a vertical turn;
    the extreme columns contain one endpoint each. This makes ordering visible
    even when other nonconsecutive landmarks happen to share a row.
    """
    columns = {}
    for r, c in points:
        columns.setdefault(c, []).append(r)
    xs = sorted(columns)
    _require(3 <= len(xs) <= 6, "expected two to five horizontal runs")
    _require(len(columns[xs[0]]) == len(columns[xs[-1]]) == 1, "invalid endpoints")
    _require(all(len(columns[x]) == 2 for x in xs[1:-1]), "missing turn corners")
    _require(all(b - a >= 2 for a, b in zip(xs, xs[1:])), "runs too short")
    ordered = [(columns[xs[0]][0], xs[0])]
    for x in xs[1:]:
        rows = columns[x]
        _require(ordered[-1][0] in rows, "corners do not form a monotone chain")
        ordered.append((ordered[-1][0], x))
        if len(rows) == 2:
            other = next(r for r in rows if r != ordered[-1][0])
            ordered.append((other, x))
    _require(len(ordered) == len(points), "unused corner")
    pixels = set()
    for a, b in zip(ordered, ordered[1:]):
        pixels.update(_segment(a, b))
    return ordered, pixels


def _paths(cells):
    """Return complete ordered 4-neighbor paths; reject branches/loops/stubs."""
    adjacent = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _NEIGHBORS
                    if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    _require(all(1 <= len(v) <= 2 for v in adjacent.values()), "branched or isolated track")
    unseen, result = set(cells), []
    while unseen:
        root = min(unseen)
        component, stack = set(), [root]
        while stack:
            p = stack.pop()
            if p not in component:
                component.add(p)
                stack.extend(adjacent[p])
        ends = sorted(p for p in component if len(adjacent[p]) == 1)
        _require(len(ends) == 2, "closed track loop")
        path, previous = [ends[0]], None
        while path[-1] != ends[1]:
            choices = [p for p in adjacent[path[-1]] if p != previous]
            _require(len(choices) == 1, "ambiguous track")
            previous = path[-1]
            path.append(choices[0])
        _require(len(path) == len(component), "incomplete trace")
        unseen.difference_update(component)
        result.append(path)
    return sorted(result, key=lambda p: (p[0], p[-1]))


def _structure(shape, bg, source_color, copy_color, wall, source, delta, gates, plugs):
    companion = {(r + delta, c) for r, c in source}
    _require(not source & companion, "overlapping source and companion")
    out = np.full(shape, bg, dtype=np.int8)
    for p in source:
        out[p] = source_color
    for p in companion:
        out[p] = copy_color
    cells = (source | companion) - set(plugs)
    for p in plugs:
        out[p] = wall
    bridge_cells = set()
    for r, c in gates:
        lo, hi = sorted((r, r + delta))
        for rr in range(lo, hi + 1):
            cells.discard((rr, c))
            out[rr, c] = wall
            for cc in (c - 1, c + 1):
                cells.add((rr, cc))
                if (rr, cc) not in source:
                    out[rr, cc] = copy_color
                bridge_cells.add((rr, cc))
    return out, cells, companion, bridge_cells


def _settle(paths, beads, delta):
    """Global terminal attraction, not a simulation with intermediate traps."""
    targets = set()
    sign = int(np.sign(delta))
    for path in paths:
        here = set(path) & beads
        if path[0][0] == path[-1][0]:
            targets.update(here)
        else:
            inward = path if sign * path[0][0] > sign * path[-1][0] else path[::-1]
            targets.update(inward[:len(here)])
    _require(len(targets) == len(beads), "bead conservation failed")
    return targets


def _interpret(a, bg, s, c, g, p):
    """Interpret one horizontal-source role assignment using this input alone."""
    landmarks = set(map(tuple, np.argwhere(a == s).tolist()))
    ordered, source = _source(landmarks)
    markers = np.argwhere(a == c)
    _require(len(markers) == 1, "companion needs one endpoint seed")
    marker = tuple(map(int, markers[0]))
    anchors = [v for v in (ordered[0], ordered[-1]) if v[1] == marker[1]]
    _require(len(anchors) == 1, "marker does not align with a source endpoint")
    delta = marker[0] - anchors[0][0]
    span = max(r for r, _ in source) - min(r for r, _ in source)
    _require(3 <= abs(delta) <= 7 and abs(delta) >= span + 2, "invalid copy displacement")
    companion = {(r + delta, col) for r, col in source}
    _require(all(0 <= r < a.shape[0] and 0 <= col < a.shape[1] for r, col in companion), "clipped copy")
    walls = set(map(tuple, np.argwhere(a == g).tolist()))
    outside = walls - source - companion
    by_row = {}
    for r, col in outside:
        by_row.setdefault(r, []).append(col)
    gates = []
    for row, columns in sorted(by_row.items()):
        columns.sort()
        runs = []
        for _, group in itertools.groupby(enumerate(columns), lambda z: z[1] - z[0]):
            runs.append([v for _, v in group])
        for run in runs:
            _require(len(run) == 3, "a switch must be a three-cell horizontal bar")
            center = run[1]
            source_rows = [r for r, col in source if col == center]
            _require(len(source_rows) == 1, "switch at a bend")
            r = source_rows[0]
            _require(row == r + int(np.sign(delta)) * (abs(delta) // 2), "off-center switch")
            _require(all((r, center + k) in source for k in range(-2, 3)), "switch needs a flat neighborhood")
            _require(not any((r + dr, center + k) in source for dr in (-1, 1) for k in range(-2, 3)), "bend beside switch")
            _require(not any((r, center + k) in landmarks for k in (-1, 0, 1)), "switch covers a landmark")
            gates.append((r, center))
    gates.sort(key=lambda v: v[1])
    _require(1 <= len(gates) <= 4, "one to four switches required")
    _require(all(b[1] - aa[1] >= 6 for aa, b in zip(gates, gates[1:])), "switches too close")
    plugs = walls & (source | companion)
    _require(not plugs & (landmarks | {marker}), "plug covers an input landmark")
    _require(all(abs(col - gc) >= 2 for _, col in plugs for _, gc in gates), "plug conflicts with switch")
    structural, cells, companion, bridges = _structure(a.shape, bg, s, c, g, source, delta, gates, plugs)
    paths = _paths(cells)
    beads = set(map(tuple, np.argwhere(a == p).tolist()))
    _require(bool(beads) and beads <= cells & (source | companion), "beads must lie on the original rails")
    targets = _settle(paths, beads, delta)
    out = structural.copy()
    for pos in targets:
        out[pos] = p
    return out, {"palette": (bg, s, c, g, p), "ordered": ordered, "source": source,
                 "companion": companion, "marker": marker, "delta": delta,
                 "gates": gates, "plugs": plugs, "paths": paths, "cells": cells,
                 "bridges": bridges, "beads": beads, "targets": targets,
                 "structural": structural}


def _analyze(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    _require(len(colors) == 5 and int(max(counts)) > a.size // 2, "expected five visible roles and a majority background")
    bg = int(colors[int(np.argmax(counts))])
    foreground = [int(v) for v in colors if v != bg]
    candidates = []
    for axis in (0, 1):
        core = a if axis == 0 else a.T
        for s in foreground:
            if not 4 <= int(np.count_nonzero(core == s)) <= 10:
                continue
            for c in foreground:
                if c == s or np.count_nonzero(core == c) != 1:
                    continue
                for g, p in itertools.permutations(v for v in foreground if v not in (s, c)):
                    try:
                        out, info = _interpret(core, bg, s, c, g, p)
                    except ValueError:
                        continue
                    info["axis"] = axis
                    info["input"] = core
                    candidates.append((out if axis == 0 else out.T, info))
    _require(len(candidates) == 1, "input roles/rail geometry must have exactly one interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible geometry, endpoint seed, switches, plugs and beads."""
    return _analyze(input_grid)[0].copy()


def _without_switch_targets(info):
    """Ablation: complete both rails, but keep their original connectivity."""
    cells = (info["source"] | info["companion"]) - info["plugs"]
    return _settle(_paths(cells), info["beads"], info["delta"])


def _features(info):
    a = info["input"]
    bg, s, c, g, p = info["palette"]
    collisions = singles = crossings = empty = levels = interior_levels = occupied = transfer_gaps = 0
    for path in info["paths"]:
        here = set(path) & info["beads"]
        occupied += bool(here)
        empty += not here
        levels += bool(here) and path[0][0] == path[-1][0]
        interior_levels += path[0][0] == path[-1][0] and bool(here - {path[0], path[-1]})
        if path[0][0] != path[-1][0]:
            inward = path if np.sign(info["delta"]) * path[0][0] > np.sign(info["delta"]) * path[-1][0] else path[::-1]
            target = set(inward[:len(here)])
            collisions += len(here) >= 2 and target != here
            singles += len(here) == 1 and target != here
            transfer = bool(here & info["source"]) and bool(target & info["companion"])
            crossings += transfer
            if transfer:
                transfer_gaps += sum(a[pos] == bg for pos in set(path) & (info["source"] | info["companion"]))
    landmarks = set(info["ordered"])
    all_alignments = set()
    for aa, b in itertools.combinations(info["ordered"], 2):
        if aa[0] == b[0] or aa[1] == b[1]:
            line = _segment(aa, b)
            if not set(line[1:-1]) & landmarks:
                all_alignments.update(line)
    return {"corners": len(info["ordered"]), "runs": len(info["ordered"]) // 2,
            "gates": len(info["gates"]), "plugs": len(info["plugs"]),
            "paths": len(info["paths"]), "beads": len(info["beads"]),
            "displacement": abs(info["delta"]), "axis": info["axis"],
            "direction": (int(np.sign(info["delta"])), 0) if info["axis"] == 0 else (0, int(np.sign(info["delta"]))),
            "packing": int(collisions), "single_motion": int(singles), "cross_rail": int(crossings),
            "empty": int(empty), "level": int(levels), "interior_level": int(interior_levels),
            "occupied_paths": int(occupied), "rejected_chord_cells": len(all_alignments - info["source"]),
            "missing_on_transfer": int(transfer_gaps),
            "missing_source": int(sum(a[pos] == bg for pos in info["source"])),
            "missing_copy": int(sum(a[pos] == bg for pos in info["companion"])),
            "gate_sensitive": info["targets"] != _without_switch_targets(info)}


def _construct(rng, palette, runs, packing=False, single=False, multi=False):
    bg, s, c, g, p = palette
    # At most 160 proposals; only one small grid exists at a time.
    for _ in range(160):
        lengths = [int(x) for x in rng.integers(4, 10 if runs <= 3 else 7, size=runs)]
        if sum(lengths) > 27:
            continue
        levels = [int(rng.integers(4))]
        for _ in range(runs - 1):
            levels.append(int(rng.choice([v for v in range(4) if v != levels[-1]])))
        if single and runs == 3:
            # A returning row offers an incidental aligned chord. Its missing
            # cells are visibly omitted in this required demonstration.
            levels[-1] = levels[0]
        shift = min(levels)
        levels = [v - shift for v in levels]
        vertices, x = [(levels[0], 0)], 0
        for i, length in enumerate(lengths):
            x += length
            vertices.append((levels[i], x))
            if i + 1 < runs:
                vertices.append((levels[i + 1], x))
        _, source = _source(set(vertices))
        delta = int(rng.integers(max(3, max(levels) + 2), 8))
        shape = (max(levels) + delta + 1, x + 1)
        anchor = vertices[0] if rng.random() < .5 else vertices[-1]
        marker = (anchor[0] + delta, anchor[1])
        centers = []
        start = 0
        for level, length in zip(levels, lengths):
            centers.extend((level, col) for col in range(start + 3, start + length - 2))
            start += length
        rng.shuffle(centers)
        wanted = int(rng.integers(2 if multi else 1, 5))
        gates = []
        for candidate in centers:
            if len(gates) < wanted and all(abs(candidate[1] - q[1]) >= 6 for q in gates):
                gates.append(candidate)
        gates.sort(key=lambda q: q[1])
        if not gates or (multi and len(gates) < 2):
            continue
        companion = {(r + delta, col) for r, col in source}
        plugs = set()
        for aa, b in zip(gates, gates[1:]):
            choices = sorted(pos for pos in source | companion
                             if aa[1] + 2 <= pos[1] <= b[1] - 2 and pos not in set(vertices) | {marker})
            if not choices:
                break
            plugs.add(choices[int(rng.integers(len(choices)))])
        if len(plugs) != len(gates) - 1:
            continue
        try:
            structural, cells, companion, bridges = _structure(shape, bg, s, c, g, source, delta, gates, plugs)
            paths = _paths(cells)
        except ValueError:
            continue
        # Optional additional stops split paths, never create hidden breaks.
        for _ in range(int(rng.integers(3))):
            choices = sorted(pos for pos in (source | companion) & cells
                             if pos not in set(vertices) | {marker}
                             and all(abs(pos[1] - gc) >= 3 for _, gc in gates)
                             and all(max(abs(pos[0] - r), abs(pos[1] - col)) >= 3 for r, col in plugs))
            if not choices:
                break
            stop = choices[int(rng.integers(len(choices)))]
            try:
                trial = _structure(shape, bg, s, c, g, source, delta, gates, plugs | {stop})
                new_paths = _paths(trial[1])
            except ValueError:
                continue
            plugs.add(stop)
            structural, cells, companion, bridges = trial
            paths = new_paths
        protected = set(vertices) | {marker}
        allowed = ((source | companion) & cells) - protected
        level_examples = []
        if single:
            for path in paths:
                pool = sorted((set(path) & allowed) - {path[0], path[-1]})
                if path[0][0] == path[-1][0] and pool:
                    level_examples.append((path, pool))
            if not level_examples:
                continue
        eligible = []
        for path in paths:
            sink = max((path[0], path[-1]))
            choices = sorted(set(path) & source & allowed)
            if sink in companion and len(choices) >= (2 if packing else 1):
                eligible.append((path, choices))
        if not eligible:
            continue
        active, choices = eligible[int(rng.integers(len(eligible)))]
        mandatory = 2 if packing else 1
        beads = {choices[int(i)] for i in rng.choice(len(choices), size=mandatory, replace=False)}
        level_path = None
        if single:
            level_path, pool = level_examples[int(rng.integers(len(level_examples)))]
            beads.add(pool[int(rng.integers(len(pool)))])
        # Keep the designated explanatory path's occupancy exact.
        for path in paths:
            if path is active or path is level_path:
                continue
            pool = sorted(set(path) & allowed)
            if pool and rng.random() < .58:
                count = int(rng.integers(1, min(5, len(pool)) + 1))
                beads.update(pool[int(i)] for i in rng.choice(len(pool), size=count, replace=False))
        inp = np.full(shape, bg, dtype=np.int8)
        for pos in vertices:
            inp[pos] = s
        inp[marker] = c
        for r, col in gates:
            middle = r + delta // 2
            inp[middle, col - 1:col + 2] = g
        for pos in plugs:
            inp[pos] = g
        for pos in beads:
            inp[pos] = p
        out = structural.copy()
        for pos in _settle(paths, beads, delta):
            out[pos] = p
        padding = []
        for n in shape:
            total = int(rng.integers(min(3, 30 - n) + 1))
            before = int(rng.integers(total + 1))
            padding.append((before, total - before))
        inp = np.pad(inp, padding, constant_values=bg)
        out = np.pad(out, padding, constant_values=bg)
        if rng.random() < .5:
            inp, out = np.fliplr(inp), np.fliplr(out)
        turns = int(rng.integers(4))
        inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
        try:
            predicted, info = _analyze(inp)
            feature = _features(info)
        except ValueError:
            continue
        if not np.array_equal(predicted, out):
            raise AssertionError("construction and visible reconstruction disagree")
        if not feature["gate_sensitive"] or not feature["cross_rail"] or not feature["missing_on_transfer"]:
            continue
        if packing and not feature["packing"]:
            continue
        if single and not feature["single_motion"]:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded ribbon construction exhausted")


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    seed = int(seed)
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    fixed_palette = tuple(int(v) for v in rng.choice(10, 5, replace=False))
    resample_palette = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            for _ in range(12):
                palette = tuple(int(v) for v in rng.choice(10, 5, replace=False)) if resample_palette else fixed_palette
                runs = (2 + (i % 2)) if split == "train" else int(rng.integers(4, 6))
                pair = _construct(rng, palette, runs, packing=(split == "train" and i == 0),
                                  single=(split == "train" and i == 1), multi=(split == "train" and i == 1))
                if split == "train" and i == 1:
                    first_direction = _features(_analyze(puzzle["train"][0]["input"])[1])["direction"]
                    if _features(_analyze(pair["input"])[1])["direction"] == first_direction:
                        pair = {k: np.rot90(v).copy() for k, v in pair.items()}
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _count("num_train", nt, 2, 10)
    _count("num_test", nq, 1, 4)
    _require(nt > nq, "invalid split counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output dimensions changed")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate inputs")
            seen.add(key)
            predicted, info = _analyze(inp)
            _require(np.array_equal(predicted, out), "incorrect output")
            bg, s, c, g, p = info["palette"]
            canonical_out = out if info["axis"] == 0 else out.T
            _require(np.count_nonzero(inp == p) == np.count_nonzero(out == p), "beads not conserved")
            for path in info["paths"]:
                _require(len(set(path) & info["beads"]) == sum(canonical_out[pos] == p for pos in path),
                         "per-path bead count changed")
            _require(set(map(tuple, np.argwhere(canonical_out == p).tolist())) == info["targets"], "wrong bead positions")
            # Every non-bead rail cell must have the exact reconstructed role.
            restored = canonical_out.copy()
            for pos in info["targets"]:
                restored[pos] = info["structural"][pos]
            _require(np.array_equal(restored, info["structural"]), "changed an unrelated cell")
            feature = _features(info)
            _require(feature["missing_source"] > 0 and feature["missing_copy"] > 0, "completion is vacuous")
            _require(feature["gate_sensitive"] and feature["cross_rail"], "parent A does not affect bead allocation")
            _require(feature["missing_on_transfer"] > 0, "parent B completion is unnecessary on the transfer path")
            features[split].append(feature)
    _require(any(f["packing"] for f in features["train"]), "missing packing demonstration")
    _require(any(f["single_motion"] for f in features["train"]), "missing single-bead demonstration")
    _require(any(f["gates"] >= 2 and f["plugs"] for f in features["train"]), "missing multiple-switch/stop demonstration")
    _require(any(f["interior_level"] for f in features["train"]), "missing tied-terminal interior bead demonstration")
    _require(any(f["rejected_chord_cells"] for f in features["train"]), "missing rejected-chord demonstration")
    _require(len({f["direction"] for f in features["train"]}) >= 2, "compass direction is not distinguished from fixed gravity")
    _require(max(f["corners"] for f in features["train"]) < min(f["corners"] for f in features["test"]), "missing geometric train/test shift")


def self_test():
    """42 seeds, all count combinations, corruption checks, and equivariance.

    Streams puzzles, retaining hashes and scalar measurements only. A maximum
    14-pair int8 puzzle uses <=25,200 array bytes; no samples/traces are saved.
    Returns descriptive measurements, not independent semantic certification.
    """
    # Hand-specified four-switch case: five beads, four pack around the
    # left bridge and one reaches the right companion endpoint. The expected
    # bead coordinates below are literal, not supplied by _settle or solve.
    vertices = [(0, 0), (0, 6), (1, 6), (1, 12), (0, 12), (0, 18), (2, 18), (2, 24)]
    four_switch = np.zeros((7, 25), dtype=np.int8)
    for pos in vertices:
        four_switch[pos] = 8
    four_switch[4, 0] = 1
    for row, col in ((2, 3), (3, 9), (2, 15), (4, 21)):
        four_switch[row, col - 1:col + 2] = 7
    for pos in ((4, 6), (5, 12), (4, 18)):
        four_switch[pos] = 7
    for pos in ((0, 1), (0, 2), (4, 1), (4, 2), (2, 23)):
        four_switch[pos] = 3
    solved, four_info = _analyze(four_switch)
    assert len(four_info["gates"]) == 4
    assert set(map(tuple, np.argwhere(solved == 3).tolist())) == {(4, 0), (4, 1), (4, 2), (3, 2), (6, 24)}
    signatures, structures, shapes, palettes = set(), set(), set(), set()
    rows = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                _, info = _analyze(inp)
                f = _features(info)
                rows.append(f)
                shapes.add(inp.shape)
                palettes.add(info["palette"])
                normalized = np.zeros(inp.shape, dtype=np.int8)
                for code, color in enumerate(info["palette"]):
                    normalized[inp == color] = code
                structures.add(hashlib.sha256(bytes(inp.shape) + normalized.tobytes()).hexdigest())
                # The displacement is a visible compass: rotate/reflection
                # must transform the answer with no hidden gravity parameter.
                for turns in range(4):
                    a, b = np.rot90(inp, turns), np.rot90(out, turns)
                    assert np.array_equal(solve(a), b)
                    assert np.array_equal(solve(np.fliplr(a)), np.fliplr(b))
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 40 and len(palettes) >= 30
    assert {f["direction"] for f in rows} == set(_NEIGHBORS)
    assert {f["runs"] for f in rows} == {2, 3, 4, 5}
    assert {1, 2, 3} <= {f["gates"] for f in rows}
    assert len({f["beads"] for f in rows}) >= 8
    assert len({f["displacement"] for f in rows}) >= 4
    assert all(f["gate_sensitive"] and f["cross_rail"] for f in rows)
    assert any(f["empty"] for f in rows) and any(f["level"] for f in rows)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(10000 + 10 * nt + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(123, 2, 1)
    repeated = generate_puzzle(123, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for split in original
               for a, b in zip(original[split], repeated[split]) for k in ("input", "output"))
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]] for s in original}
    pair = corrupt["test"][0]
    changed = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][changed] = pair["input"][changed]
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    # Explicit rejection of cycles and branches, independent of labels.
    for cells in ({(0, 0), (0, 1), (1, 0), (1, 1)},
                  {(0, 1), (1, 0), (1, 1), (1, 2)}):
        try:
            _paths(cells)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid topology accepted")
    return {"seeds": 42, "pairs": len(rows), "distinct_puzzles": len(signatures),
            "normalized_inputs": len(structures), "shapes": len(shapes),
            "palettes": len(palettes), "gate_counts": sorted({f["gates"] for f in rows}),
            "bead_counts": sorted({f["beads"] for f in rows}),
            "displacements": sorted({f["displacement"] for f in rows})}
