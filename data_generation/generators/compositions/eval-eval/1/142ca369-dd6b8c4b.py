"""Spectral harvester: reflected color filters feed an ordered nine-slot store.

Only standard-library modules and NumPy are used. No files are read or written.
All route records are transient and bounded by the 30 by 30 canvas.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ORTHOGONALS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Invalid grid colors")
    return a


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _source(elbow, direction, color):
    r, c = elbow
    dr, dc = direction
    return {"elbow": elbow, "direction": direction, "color": color,
            "cells": {(r, c), (r-dr, c), (r, c-dc)}}


def _mirror(midpoint, axis, length, color):
    r, c = midpoint
    radius = length//2
    cells = {(r, c+d) if axis == 0 else (r+d, c)
             for d in range(-radius, radius+1)}
    return {"midpoint": midpoint, "axis": axis, "color": color, "cells": cells}


def _components(a, background, excluded):
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != background)} - excluded
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], {start}
        for r, c in todo:
            for dr, dc in ORTHOGONALS:
                q = r+dr, c+dc
                if q in remaining and a[q] == a[start]:
                    remaining.remove(q)
                    cells.add(q)
                    todo.append(q)
        result.append((int(a[start]), cells))
    return result


def _parse(input_grid):
    a = _array(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    background = int(np.argmax(counts))
    _require(counts[background] > a.size//2, "Background must be a strict majority")
    counters = []
    for r in range(a.shape[0]-2):
        for c in range(a.shape[1]-2):
            slot, marker = int(a[r, c]), int(a[r+1, c+1])
            if slot == background or marker in (slot, background):
                continue
            patch = a[r:r+3, c:c+3].copy()
            patch[1, 1] = slot
            if np.all(patch == slot) and counts[slot] == 8 and counts[marker] == 1:
                counters.append((r, c, slot, marker))
    _require(len(counters) == 1, "Exactly one marked 3x3 collector is required")
    r, c, slot, marker = counters[0]
    counter_cells = {(r+i, c+j) for i in range(3) for j in range(3)}
    sources, mirrors, dots = [], [], {}
    for shade, cells in _components(a, background, counter_cells):
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if len(cells) == 1:
            dots[next(iter(cells))] = shade
        elif len(cells) == 3 and r1-r0 == c1-c0 == 1:
            missing = next(iter({(rr, cc) for rr in (r0, r1) for cc in (c0, c1)} - cells))
            elbow = r0+r1-missing[0], c0+c1-missing[1]
            direction = elbow[0]-missing[0], elbow[1]-missing[1]
            sources.append(_source(elbow, direction, shade))
        elif len(cells) >= 3 and len(cells) % 2 == 1 and (r0 == r1 or c0 == c1):
            _require(len(cells) == max(r1-r0, c1-c0)+1, "Mirror must be a solid odd bar")
            mirrors.append(_mirror(((r0+r1)//2, (c0+c1)//2),
                                   0 if r0 == r1 else 1, len(cells), shade))
        else:
            raise AssertionError("Unrecognized optical object or non-isolated dot")
    _require(len(sources) == 1 and dots, "Need one L source and at least one dot")
    return a, background, counters[0], sources[0], mirrors, dots, counter_cells


def _trace(shape, source, mirrors, extra_blocked=()):
    """Return ordered (position, active color) visits; contacts recolor immediately."""
    blocked = set(extra_blocked) | source["cells"]
    for mirror in mirrors:
        blocked.update(mirror["cells"])
    r, c = source["elbow"]
    dr, dc = source["direction"]
    shade = source["color"]
    path, events, visited, used = [], [], set(), set()
    while True:
        r, c = r+dr, c+dc
        p = r, c
        if not _inside(p, shape):
            break
        _require(p not in blocked, "Ray hits a solid object")
        _require(p not in visited, "Self-crossing or looping routes are outside the family")
        visited.add(p)
        hits = []
        for i, mirror in enumerate(mirrors):
            mr, mc = mirror["midpoint"]
            if ((mirror["axis"] == 0 and c == mc and mr-r == dr)
                    or (mirror["axis"] == 1 and r == mr and mc-c == dc)):
                hits.append(i)
        _require(len(hits) <= 1, "Ambiguous simultaneous mirror contact")
        if hits:
            i = hits[0]
            _require(i not in used, "A mirror is visited more than once")
            used.add(i)
            mirror = mirrors[i]
            previous, shade = shade, mirror["color"]
            events.append((len(path), i, mirror["axis"], previous, shade))
            if mirror["axis"] == 0:
                dr = -dr
            else:
                dc = -dc
        path.append((p, shade))
    _require(used == set(range(len(mirrors))), "Every mirror must be used")
    return path, events


def _analyze(input_grid):
    a, bg, counter, source, mirrors, dots, counter_cells = _parse(input_grid)
    path, events = _trace(a.shape, source, mirrors, counter_cells)
    matches = [(p, shade) for p, shade in path if dots.get(p) == shade]
    selected = matches[:9]
    return {"grid": a, "background": bg, "counter": counter, "source": source,
            "mirrors": mirrors, "dots": dots, "counter_cells": counter_cells,
            "path": path, "events": events, "matches": matches, "selected": selected}


def _write_selection(a, background, counter, selected):
    out = a.copy()
    r, c = counter[:2]
    for i, (p, shade) in enumerate(selected):
        out[p] = background
        out[r+i//3, c+i%3] = shade
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible geometry and colors only; never changes its argument."""
    f = _analyze(input_grid)
    return _write_selection(f["grid"], f["background"], f["counter"], f["selected"])


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _near(cells):
    return cells | {(r+dr, c+dc) for r, c in cells for dr, dc in ORTHOGONALS}


def _route_geometry(rng, profile, spec):
    """Append mirrors to the last flight, accepting only the intended new contact."""
    low = 22 if spec == "capacity" else 16 if spec == "partial" else 13
    shape = tuple(int(x) for x in rng.integers(low, 31, size=2))
    direction = profile.get("direction", DIAGONALS[int(rng.integers(4))])
    elbow = int(rng.integers(1, shape[0]-1)), int(rng.integers(1, shape[1]-1))
    source = _source(elbow, direction, int(rng.choice(profile["colors"])))
    if not all(_inside(p, shape) for p in source["cells"]):
        return None
    path, events = _trace(shape, source, [])
    if len(path) < 5:
        return None
    mirrors = []
    if spec == "capacity":
        wanted = int(rng.integers(2, 5))
    elif spec == "partial":
        wanted = int(rng.integers(1, 3))
    else:
        # The same route builder serves both splits. Test weighting changes only
        # the mix, not the optics or filtering conventions.
        wanted = int(rng.choice([0, 1, 2, 3, 4], p=profile["bounce_weights"]))
    for bounce in range(wanted):
        start = events[-1][0]+2 if events else 1
        options = [(i, axis, length) for i in range(start, len(path)-2)
                   for axis in (0, 1) for length in profile["lengths"]]
        rng.shuffle(options)
        accepted = False
        for index, axis, length in options[:160]:
            if bounce == 0 and "axis" in profile and axis != profile["axis"]:
                continue
            if bounce == 0 and "first_length" in profile and length != profile["first_length"]:
                continue
            # Direction at this contact is recovered from its previous visit.
            p = path[index][0]
            previous = source["elbow"] if index == 0 else path[index-1][0]
            incoming = p[0]-previous[0], p[1]-previous[1]
            midpoint = p[0]+(incoming[0] if axis == 0 else 0), p[1]+(incoming[1] if axis == 1 else 0)
            shades = profile["colors"]
            if bounce == 0 and spec in ("partial", "capacity"):
                shades = [x for x in shades if x != source["color"]]
            shade = int(rng.choice(shades))
            mirror = _mirror(midpoint, axis, length, shade)
            occupied = source["cells"] | set().union(*(m["cells"] for m in mirrors))
            if not all(_inside(q, shape) for q in mirror["cells"]) or _near(occupied) & mirror["cells"]:
                continue
            try:
                trial_path, trial_events = _trace(shape, source, mirrors+[mirror])
            except AssertionError:
                continue
            if [e[1] for e in trial_events] != list(range(bounce+1)):
                continue
            if [e[:3] for e in trial_events[:-1]] != [e[:3] for e in events]:
                continue
            if trial_events[-1][0] != index or len(trial_path)-index < 4:
                continue
            mirrors.append(mirror)
            path, events = trial_path, trial_events
            accepted = True
            break
        if not accepted:
            return None
    return shape, source, mirrors, path, events


def _ablation_flags(f):
    selected = f["selected"]
    chosen = {p for p, _ in selected}
    path_positions = {p for p, _ in f["path"]}
    unfiltered = [(p, f["dots"][p]) for p, _ in f["path"] if p in f["dots"]][:9]
    constant = [(p, f["dots"][p]) for p, _ in f["path"]
                if f["dots"].get(p) == f["source"]["color"]][:9]
    er, ec = f["source"]["elbow"]
    dr, dc = f["source"]["direction"]
    straight = []
    for step in range(1, 31):
        p = er+step*dr, ec+step*dc
        if not _inside(p, f["grid"].shape):
            break
        if f["dots"].get(p) == f["source"]["color"]:
            straight.append((p, f["dots"][p]))
    sequence = [shade for _, shade in selected]
    spatial_sequence = [shade for _, shade in sorted(selected)]
    return {"filter_matters": unfiltered != selected,
            "recolor_matters": constant != selected,
            "reflection_matters": straight[:9] != selected,
            "capacity_matters": len(f["matches"]) > 9,
            "spatial_order_matters": sequence != spatial_sequence,
            "color_sort_matters": sequence != sorted(sequence),
            "off_path": sum(p not in path_positions for p in f["dots"]),
            "wrong_color": sum(p in f["dots"] and f["dots"][p] != shade for p, shade in f["path"]),
            "selected_colors": len(set(sequence)),
            "removed": len(chosen)}


def _make_pair(rng, palette, spec, forced_axis=None, forced_direction=None, shifted=False,
               holdout_length=9):
    bg, slot, marker, colors = palette
    profile = {"colors": colors, "bounce_weights": (0, .24, .30, .27, .19) if shifted
               else (.18, .30, .28, .17, .07),
               "lengths": tuple(range(3, holdout_length+(1 if shifted else -1), 2))}
    if shifted:
        profile["first_length"] = holdout_length
    elif spec in ("partial", "capacity"):
        profile["first_length"] = 3 if spec == "partial" else 5
    if forced_axis is not None:
        profile["axis"] = forced_axis
    if forced_direction is not None:
        profile["direction"] = forced_direction
    for _ in range(240):
        geometry = _route_geometry(rng, profile, spec)
        if geometry is None:
            continue
        shape, source, mirrors, path, events = geometry
        occupied = source["cells"] | set().union(*(m["cells"] for m in mirrors))
        route_cells = {p for p, _ in path}
        patches = [(r, c) for r in range(shape[0]-2) for c in range(shape[1]-2)
                   if not ({(r+i, c+j) for i in range(3) for j in range(3)} & (occupied | route_cells))]
        if not patches:
            continue
        r, c = patches[int(rng.integers(len(patches)))]
        counter = r, c, slot, marker
        counter_cells = {(r+i, c+j) for i in range(3) for j in range(3)}
        forbidden = _near(occupied | counter_cells)
        eligible = [i for i, (p, _) in enumerate(path) if p not in forbidden]
        load = int(rng.choice([2, 4])) if spec == "partial" else int(rng.integers(10, 16)) if spec == "capacity" else int(rng.integers(0, 15))
        if len(eligible) < load+2:
            continue
        base = np.full(shape, bg, dtype=np.int8)
        for obj in [source]+mirrors:
            for p in obj["cells"]:
                base[p] = obj["color"]
        base[r:r+3, c:c+3] = slot
        base[r+1, c+1] = marker
        for _ in range(45):
            indices = sorted(int(i) for i in rng.choice(eligible, size=load, replace=False))
            matches = {path[i][0] for i in indices}
            remaining = [i for i in eligible if i not in indices]
            if spec in ("partial", "capacity"):
                # A wrong-color dot after a color-changing mirror survives even
                # though a later matching dot is collected. In capacity scenes
                # it occurs before the ninth successful capture.
                limit = indices[min(8, len(indices)-1)]
                wrong_options = [i for i in remaining if path[i][1] != source["color"] and i < limit]
                if not wrong_options:
                    continue
                first_wrong = int(rng.choice(wrong_options))
            else:
                first_wrong = int(rng.choice(remaining))
            wrong_indices = [first_wrong]
            spare = [i for i in remaining if i != first_wrong]
            nwrong = min(len(spare), int(rng.integers(0, 4)))
            wrong_indices.extend(int(i) for i in rng.choice(spare, size=nwrong, replace=False))
            a = base.copy()
            for i in indices:
                p, shade = path[i]
                a[p] = shade
            for i in wrong_indices:
                p, active = path[i]
                alternatives = [x for x in colors if x != active]
                a[p] = source["color"] if i == first_wrong and active != source["color"] else int(rng.choice(alternatives))
            tokens = matches | {path[i][0] for i in wrong_indices}
            banned = forbidden | route_cells | _near(tokens)
            off_options = [(rr, cc) for rr in range(shape[0]) for cc in range(shape[1])
                           if (rr, cc) not in banned]
            rng.shuffle(off_options)
            wanted_off = int(rng.integers(1, 9))
            placed = 0
            for p in off_options:
                if p in _near(tokens):
                    continue
                a[p] = int(rng.choice(colors))
                tokens.add(p)
                placed += 1
                if placed == wanted_off:
                    break
            if placed < 1:
                continue
            try:
                f = _analyze(a)
            except AssertionError:
                continue
            flags = _ablation_flags(f)
            if spec in ("partial", "capacity") and not (
                    flags["filter_matters"] and flags["recolor_matters"]
                    and flags["reflection_matters"] and flags["selected_colors"] >= 2):
                continue
            if spec == "partial" and not (flags["spatial_order_matters"] and flags["color_sort_matters"]):
                continue
            if spec == "capacity" and not flags["capacity_matters"]:
                continue
            out = _write_selection(a, bg, counter, f["selected"])
            return {"input": a, "output": out}
    raise RuntimeError("Bounded construction could not satisfy the requested evidence")


def _valid_int(x):
    return isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _valid_int(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not (_valid_int(num_train) and _valid_int(num_test) and 2 <= num_train <= 10
            and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require 2..10 training, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.permutation(10)]
    bg, slot, marker = palette[:3]
    colors = palette[3:3+int(rng.integers(2, 6))]
    fixed_palette = bg, slot, marker, colors
    holdout_length = int(rng.choice([7, 9]))
    directions = [DIAGONALS[int(i)] for i in rng.permutation(4)]
    puzzle = {"train": [], "test": []}
    seen = set()
    specs = [("partial", 0, directions[0]), ("capacity", 1, directions[1])]
    specs += [("general", None, None)] * (num_train-2)
    rng.shuffle(specs)
    for split, requests in (("train", specs), ("test", [("general", None, None)]*num_test)):
        for spec, axis, direction in requests:
            for _ in range(12):
                pair = _make_pair(rng, fixed_palette, spec, axis, direction, split == "test",
                                  holdout_length)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct input construction exhausted bounded retries")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Pair counts")
    seen, roles, training = set(), set(), []
    train_lengths, test_maxima = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Pair keys")
            a, b = _array(pair["input"]), _array(pair["output"])
            _require(a.shape == b.shape, "Shape must be preserved")
            key = _key(a)
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            f = _analyze(a)
            roles.add((f["background"], *f["counter"][2:]))
            # Validate every changed cell and conservation, rather than only an
            # output hash. The optical tracing routine is still author-owned.
            selected = f["selected"]
            chosen = {p for p, _ in selected}
            r, c = f["counter"][:2]
            selected_colors = [shade for _, shade in selected]
            for rr in range(a.shape[0]):
                for cc in range(a.shape[1]):
                    p = rr, cc
                    expected = f["background"] if p in chosen else int(a[p])
                    if p in f["counter_cells"]:
                        i = (rr-r)*3+cc-c
                        if i < len(selected):
                            expected = selected_colors[i]
                    _require(int(b[p]) == expected, "Incorrect collection, order, or preservation")
            removed = sum(int(a[p]) != int(b[p]) for p in f["dots"])
            _require(removed == min(9, len(f["matches"])), "Conservation/capacity failure")
            structural = f["source"]["cells"] | set().union(*(m["cells"] for m in f["mirrors"]))
            _require(all(a[p] == b[p] for p in structural), "Optical objects changed")
            _require(all(p not in _near(structural | f["counter_cells"]) for p in f["dots"]), "Dots touch structural objects")
            _require(all(not any((p[0]+dr, p[1]+dc) in f["dots"] for dr, dc in ORTHOGONALS)
                         for p in f["dots"]), "Dots must be orthogonally isolated")
            if split == "train":
                training.append((f, _ablation_flags(f)))
                train_lengths.update(len(m["cells"]) for m in f["mirrors"])
            else:
                test_maxima.append(max((len(m["cells"]) for m in f["mirrors"]), default=0))
    _require(len(roles) == 1, "Palette roles must stay fixed within a puzzle")
    _require(any(len(f["selected"]) in (2, 4) and x["spatial_order_matters"] and x["color_sort_matters"]
                 for f, x in training), "Need an ordered partial-fill demonstration")
    _require(any(x["capacity_matters"] for _, x in training), "Need retained excess in training")
    _require({e[2] for f, _ in training for e in f["events"]} == {0, 1}, "Teach both reflection axes")
    _require(len({f["source"]["direction"] for f, _ in training}) >= 2, "Teach different launch directions")
    _require(sum(x["filter_matters"] and x["recolor_matters"] and x["reflection_matters"]
                 and x["off_path"] > 0 and x["selected_colors"] >= 2 for _, x in training) >= 2,
             "Two demonstrations must expose the coupled optical and collection concepts")
    _require({3, 5} <= train_lengths and all(n > max(train_lengths) for n in test_maxima),
             "Training must teach different odd lengths before the longer-mirror test shift")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, all legal counts, targeted semantic mutations; no disk writes."""
    _rule_regressions()
    structures, shapes, signatures = set(), set(), set()
    axes, lengths, directions, backgrounds, loads, bounces = set(), set(), set(), set(), set(), set()
    test_loads, returning = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = np.asarray(pair["input"])
                original = a.copy()
                _require(np.array_equal(solve(a), pair["output"]), "solve disagrees")
                _require(np.array_equal(a, original), "solve mutated input")
                f = _analyze(a)
                mapping = {f["background"]: 0}
                normalized = bytes(mapping.setdefault(int(x), len(mapping)) for x in a.flat)
                structures.add((a.shape, normalized))
                shapes.add(a.shape)
                axes.update(e[2] for e in f["events"])
                lengths.update(len(m["cells"]) for m in f["mirrors"])
                directions.add(f["source"]["direction"])
                backgrounds.add(f["background"])
                loads.add(len(f["selected"]))
                bounces.add(len(f["events"]))
                if split == "test":
                    test_loads.add(len(f["selected"]))
                shades = [f["source"]["color"]] + [e[4] for e in f["events"]]
                returning += any(shades[i] in shades[:i-1] and shades[i] != shades[i-1]
                                 for i in range(2, len(shades)))
        if seed == 0:
            _require(_digest(puzzle) == _digest(generate_puzzle(0)), "Nondeterminism")
            broken = {"train": puzzle["train"], "test": [dict(puzzle["test"][0])]}
            broken["test"][0]["output"] = np.asarray(broken["test"][0]["output"]).copy()
            # A fixed structural source cell must be preserved even for zero load.
            p = next(iter(_analyze(broken["test"][0]["input"])["source"]["cells"]))
            broken["test"][0]["output"][p] = (int(broken["test"][0]["output"][p])+1) % 10
            try:
                validate_generated_puzzle(broken)
            except AssertionError:
                pass
            else:
                raise AssertionError("Corrupted output accepted")
    _require(len(signatures) == 42 and len(structures) >= 180, "Structural diversity collapsed")
    _require(len(shapes) >= 70 and len(backgrounds) >= 8, "Canvas or palette diversity collapsed")
    _require(axes == {0, 1} and lengths >= {3, 5, 7} and directions == set(DIAGONALS), "Optics diversity collapsed")
    _require(loads == set(range(10)) and len(test_loads) >= 6, "Load variation collapsed")
    _require(bounces >= {0, 1, 2, 3, 4} and returning >= 5, "Route/color history variation collapsed")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42000+nt*10+ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "Explicit counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")


def _rule_regressions():
    # Numeric expectations are specified directly, without the route builder.
    a = np.zeros((14, 16), dtype=np.int8)
    a[2, 2] = a[1, 2] = a[2, 1] = 1
    a[8, 6:9] = 2
    a[10:13, 1:4] = 3
    a[11, 2] = 4
    for p, shade in [((3, 3), 1), ((4, 4), 2), ((6, 8), 1),
                     ((5, 9), 2), ((2, 12), 2), ((10, 10), 1)]:
        a[p] = shade
    erased = np.zeros_like(a, dtype=bool)
    for p in ((3, 3), (5, 9), (2, 12)):
        erased[p] = True
    for k in range(4):
        for flip in (False, True):
            inp, mask = np.rot90(a, k).copy(), np.rot90(erased, k).copy()
            if flip:
                inp, mask = inp[:, ::-1].copy(), mask[:, ::-1].copy()
            expected = inp.copy()
            expected[mask] = 0
            rr, cc = np.where(inp == 3)
            r, c = int(rr.min()), int(cc.min())
            for i, shade in enumerate((1, 2, 2)):
                expected[r+i//3, c+i%3] = shade
            _require(np.array_equal(solve(inp), expected), "Rotation/filter/order regression")
    a = np.zeros((20, 30), dtype=np.int8)
    a[1, 1] = a[0, 1] = a[1, 0] = 1
    a[12:17, 15] = 2
    a[4:7, 21:24] = 3
    a[5, 22] = 4
    matches = [(r, r) for r in (2, 3, 4, 5, 7, 8, 9, 10, 11, 12)]
    for p in matches:
        a[p] = 1
    for p in ((6, 6), (17, 11), (18, 10), (19, 9), (0, 20)):
        a[p] = 2
    expected = a.copy()
    for p in matches[:9]:
        expected[p] = 0
    expected[4:7, 21:24] = 1
    _require(np.array_equal(solve(a), expected), "Capacity/excess/marker regression")
    # The same fixture checks that literal color numbers have no hidden meaning.
    recolor = np.array([9, 0, 8, 2, 1, 3, 4, 5, 6, 7], dtype=np.int8)
    _require(np.array_equal(solve(recolor[a]), recolor[expected]), "Palette-role regression")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds and all legal explicit counts")
