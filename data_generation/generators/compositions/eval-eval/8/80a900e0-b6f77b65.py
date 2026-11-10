"""Optical demolition: permanent endpoint beams interact with falling rods.

Standard library + NumPy only. No imports, examples, or state from the parents.
All diagnostics are in memory; this module writes no files or execution traces.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _outline(center, radius):
    r, c = center
    return {(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius}


def _cap(center, radius, direction):
    r, c = center
    dr, dc = direction
    return [(r + dr * (k + 1), c + dc * (radius - k + 1))
            for k in range(radius + 1)]


def _base(shape, background):
    return np.asarray(background, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _cells(piece):
    orientation, r, c, length, _ = piece
    return {(r + (k if orientation == "V" else 0),
             c + (k if orientation == "H" else 0)) for k in range(length)}


def _shift(piece):
    return piece[0], piece[1] + 1, piece[2], piece[3], piece[4]


def _array(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid.astype(np.int8, copy=False)


def _parse(value):
    """Recover checkerboard, complete diamonds/caps, and unambiguous straight rods."""
    grid = _array(value)
    shape = grid.shape
    parity = np.indices(shape).sum(axis=0) % 2
    background = []
    for p in (0, 1):
        counts = np.bincount(grid[parity == p], minlength=10)
        color = int(counts.argmax())
        assert counts[color] * 2 > int(np.sum(parity == p)), "background is not identifiable"
        background.append(color)
    assert background[0] != background[1]
    base = _base(shape, background)
    foreground = set(map(tuple, np.argwhere(grid != base)))
    assert all(int(grid[p]) not in background for p in foreground)
    emitters = []
    h, w = shape
    for radius in (2, 3):
        if min(shape) < 2 * radius + 1:
            continue
        sh = (h - 2 * radius, w - 2 * radius)
        offsets = sorted(_outline((0, 0), radius))
        dr, dc = offsets[0]
        colors = grid[radius + dr:radius + dr + sh[0], radius + dc:radius + dc + sh[1]]
        mask = (colors != background[0]) & (colors != background[1])
        for dr, dc in offsets[1:]:
            mask &= grid[radius + dr:radius + dr + sh[0], radius + dc:radius + dc + sh[1]] == colors
        for r, c in np.argwhere(mask):
            center = (int(r) + radius, int(c) + radius)
            # An emitter has an empty checkerboard interior, including its center.
            interior = [(center[0] + dr, center[1] + dc)
                        for dr in range(-radius + 1, radius)
                        for dc in range(-radius + 1, radius) if abs(dr) + abs(dc) < radius]
            if all(grid[p] == base[p] for p in interior):
                emitters.append({"center": center, "radius": radius,
                                 "color": int(colors[r, c]), "caps": {}})
    assert emitters, "no emitter"
    fixed = {}
    for e in emitters:
        for p in _outline(e["center"], e["radius"]):
            assert p not in fixed
            fixed[p] = e["color"]
    for e in emitters:
        for direction in DIRECTIONS:
            cap = _cap(e["center"], e["radius"], direction)
            if not all(_inside(p, shape) for p in cap):
                continue
            values = {int(grid[p]) for p in cap}
            if len(values) == 1 and not values.intersection(background + [e["color"]]):
                color = values.pop()
                assert not set(cap).intersection(fixed)
                e["caps"][direction] = color
                fixed.update({p: color for p in cap})
        assert e["caps"], "diamond lacks a complete cap"
    remaining = foreground - fixed.keys()
    pieces = []
    while remaining:
        start = min(remaining)
        color = int(grid[start])
        component, pending = {start}, [start]
        remaining.remove(start)
        while pending:
            r, c = pending.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining and int(grid[p]) == color:
                    remaining.remove(p)
                    component.add(p)
                    pending.append(p)
        rows, cols = {r for r, _ in component}, {c for _, c in component}
        assert len(component) >= 2 and (len(rows) == 1 or len(cols) == 1), "ambiguous rod"
        orientation = "H" if len(rows) == 1 else "V"
        piece = (orientation, min(rows), min(cols), len(component), color)
        assert _cells(piece) == component
        pieces.append(piece)
    assert pieces
    return {"shape": shape, "background": tuple(background), "emitters": emitters,
            "fixed": fixed, "pieces": sorted(pieces)}


def _beams(analysis, filled=False):
    """Cells strictly beyond cap endpoints. Sets also support optical ablations."""
    beams = {}
    for e in analysis["emitters"]:
        for direction, color in e["caps"].items():
            cap = _cap(e["center"], e["radius"], direction)
            starts = cap if filled else (cap[0], cap[-1])
            dr, dc = direction
            for r, c in starts:
                p = (r + dr, c + dc)
                while _inside(p, analysis["shape"]):
                    beams.setdefault(p, set()).add(color)
                    p = (p[0] + dr, p[1] + dc)
    return beams


def _evolve(pieces, shape, fixed, beams, *, gravity=True, once=False, global_colors=False):
    """Delete, then greatest jointly movable set, one row per optical tick.

    IDs preserve original rod identity even if same-colored rods later touch.
    Only bounded event summaries are retained, never grid histories.
    """
    live = dict(enumerate(pieces))
    cells = {i: _cells(piece) for i, piece in live.items()}
    distances = [0] * len(pieces)
    deleted, late, initial = set(), set(), set()
    beam_colors = set().union(*beams.values()) if beams else set()
    fields = {color: {p for p, colors in beams.items() if color in colors}
              for color in beam_colors}
    for tick in range(shape[0] * len(pieces) + len(pieces) + 2):
        hit = set()
        if not once or tick == 0:
            hit = {i for i, piece in live.items() if
                   (global_colors and piece[4] in beam_colors) or
                   cells[i].intersection(fields.get(piece[4], ()))}
        for i in hit:
            if tick == 0:
                initial.add(i)
            if distances[i]:
                late.add(i)
            del live[i]
            del cells[i]
        deleted.update(hit)
        targets = {i: {(r + 1, c) for r, c in s} for i, s in cells.items()}
        moving = {i for i, s in targets.items() if gravity and
                  all(r < shape[0] for r, _ in s) and not s.intersection(fixed)}
        while True:
            stationary = set().union(*(s for i, s in cells.items() if i not in moving))
            blocked = {i for i in moving if targets[i].intersection(stationary)}
            if not blocked:
                break
            moving.difference_update(blocked)
        if not hit and not moving:
            return live, {"deleted": deleted, "late": late, "initial": initial,
                          "distances": distances, "ticks": tick}
        for i in moving:
            live[i] = _shift(live[i])
            cells[i] = targets[i]
            distances[i] += 1
    raise AssertionError("monotone optical gravity failed to terminate")


def _render(analysis, live, beams):
    grid = _base(analysis["shape"], analysis["background"])
    for p, colors in beams.items():
        assert len(colors) == 1, "unequal beam colors cross"
        grid[p] = next(iter(colors))
    for p, color in analysis["fixed"].items():
        grid[p] = color
    occupied = set(analysis["fixed"])
    for piece in live.values():
        cells = _cells(piece)
        assert not cells.intersection(occupied)
        occupied.update(cells)
        for p in cells:
            grid[p] = piece[4]
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pure input-only reconstruction and deterministic optical gravity."""
    analysis = _parse(input_grid)
    beams = _beams(analysis)
    assert all(len(colors) == 1 for colors in beams.values())
    assert not analysis["fixed"].keys() & beams.keys(), "beam crosses a fixed emitter"
    live, _ = _evolve(analysis["pieces"], analysis["shape"], analysis["fixed"], beams)
    return _render(analysis, live, beams)


def _put(pieces, fixed, piece, shape):
    """Maintain input-only component segmentation, including repeated colors."""
    cells = _cells(piece)
    occupied = set(fixed).union(*(_cells(p) for p in pieces))
    if any(not _inside(p, shape) for p in cells) or cells.intersection(occupied):
        return False
    halo = cells | {(r + dr, c + dc) for r, c in cells
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
    if any(p[4] == piece[4] and halo.intersection(_cells(p)) for p in pieces):
        return False
    pieces.append(piece)
    return True


def _drop(pieces, fixed, orientation, col, length, color, shape):
    piece = (orientation, 0, col, length, color)
    occupied = set(fixed).union(*(_cells(p) for p in pieces))
    if _cells(piece).intersection(occupied):
        return False
    while all(_inside(p, shape) for p in _cells(_shift(piece))) and not _cells(_shift(piece)).intersection(occupied):
        piece = _shift(piece)
    return _put(pieces, fixed, piece, shape)


def _physical_signature(live):
    return tuple(sorted(live.items()))


def _settle_then_fire(pieces, shape, fixed, beams):
    """A tempting alternative: inspect optics only after each complete fall."""
    live = dict(enumerate(pieces))
    for _ in range(len(pieces) + 1):
        hit = {i for i, piece in live.items()
               if any(piece[4] in beams.get(p, ()) for p in _cells(piece))}
        if not hit:
            return live
        live = {i: p for i, p in live.items() if i not in hit}
        ids = list(live)
        settled, _ = _evolve(list(live.values()), shape, fixed, {})
        live = {ids[i]: p for i, p in settled.items()}
    raise AssertionError("alternative failed to terminate")


def _features(analysis):
    beams = _beams(analysis)
    pieces, shape, fixed = analysis["pieces"], analysis["shape"], analysis["fixed"]
    live, events = _evolve(pieces, shape, fixed, beams)
    signature = _physical_signature(live)
    output = _render(analysis, live, beams)
    contrasts = {}
    for name, alternative, kwargs in (
        ("filled", _beams(analysis, filled=True), {}),
        ("global", beams, {"global_colors": True}),
        ("once", beams, {"once": True}),
        ("no_gravity", beams, {"gravity": False}),
        ("ignore_color", {p: set(range(10)) for p in beams}, {}),
    ):
        other, _ = _evolve(pieces, shape, fixed, alternative, **kwargs)
        contrasts[name] = (signature != _physical_signature(other) and
                           not np.array_equal(output, _render(analysis, other, beams)))
    other = _settle_then_fire(pieces, shape, fixed, beams)
    contrasts["settle_then_fire"] = not np.array_equal(output, _render(analysis, other, beams))
    cap_colors = {c for e in analysis["emitters"] for c in e["caps"].values()}
    matching_safe = [i for i, p in live.items() if p[4] in cap_colors]
    nonmatching_hit = [i for i, p in live.items() if p[4] not in cap_colors and _cells(p).intersection(beams)]
    moved_survivors = [i for i in live if events["distances"][i] > 0]
    # At least one directly hit rod is only partially illuminated: deleting cells
    # alone would retain part of the rigid piece instead of deleting its identity.
    partial_hits = [i for i in events["initial"] if not _cells(pieces[i]) <= beams.keys()]
    return {"live": live, "events": events, "contrasts": contrasts,
            "safe": matching_safe, "nonmatching_hit": nonmatching_hit,
            "moved_survivors": moved_survivors, "partial_hits": partial_hits}


def _make_pair(rng, radius, *, reflected, second, extras, witness):
    """A parameterized interacting bridge plus freely dropped additional rods.

    The bridge guarantees an optical release and a later crossing. Additional
    rods can make shared supports, shelves, intermediate landings and cascades.
    Rejection retains only visible, informative members of this same rule.
    """
    for attempt in range(160):
        h = int(rng.integers(23, 31))
        w = int(rng.integers(25 if radius == 2 else 28, 31))
        shape = (h, w)
        palette = list(map(int, rng.permutation(10)))
        background = tuple(palette[:2])
        structural, upper, lower, cargo = palette[2:6]
        ec = radius + 1
        distance = int(rng.integers(radius + 2, radius + 5))
        shelf = int(rng.integers(3, 7))
        er = shelf + distance + radius + 1
        if er + radius + 1 >= h:
            continue
        emitters = [{"center": (er, ec), "radius": radius, "color": structural,
                     "caps": {(-1, 1): upper, (1, 1): lower}}]
        if second:
            rr = int(rng.integers(3, min(11, h - 4)))
            # Opposite checkerboard parity permits different-colored optical
            # paths to cross between cells, without a pixel-color conflict.
            if (rr + w - 4 + 2) % 2 == (er + ec + radius) % 2:
                rr += 1
            emitters.append({"center": (rr, w - 4), "radius": 2, "color": structural,
                             "caps": {(1, -1): cargo}})
        fixed = {}
        for e in emitters:
            for p in _outline(e["center"], e["radius"]):
                fixed[p] = e["color"]
            for direction, color in e["caps"].items():
                fixed.update({p: color for p in _cap(e["center"], e["radius"], direction)})
        analysis = {"shape": shape, "background": background, "emitters": emitters,
                    "fixed": fixed, "pieces": []}
        beams = _beams(analysis)
        if any(len(colors) != 1 for colors in beams.values()) or fixed.keys() & beams.keys():
            continue
        left, right = ec + radius + 2, w - 8 if second else w - 1
        x = ec + distance
        if x > right - 1:
            continue
        a = int(rng.integers(left, x + 1))
        b = int(rng.integers(max(x + 1, a + 3), min(right, x + 6) + 1))
        pieces = []
        assert _put(pieces, fixed, ("V", shelf + 1, x, h - shelf - 1, upper), shape)
        # Optional redundant leg: both legs are separately optically removed.
        candidates = [xx for xx in range(a, b + 1) if abs(xx - x) >= 2
                      and any(upper in beams.get((r, xx), ()) for r in range(shelf + 1, h))]
        if candidates and rng.random() < .55:
            xx = int(rng.choice(candidates))
            _put(pieces, fixed, ("V", shelf + 1, xx, h - shelf - 1, upper), shape)
        assert _put(pieces, fixed, ("H", shelf, a, b - a + 1, lower), shape)
        if rng.random() < .5:
            length = int(rng.integers(2, shelf if second else shelf + 1))
            cargo_piece = ("V", shelf - length, int(rng.integers(a, b + 1)), length, cargo)
        else:
            length = int(rng.integers(2, b - a + 2))
            cargo_piece = ("H", shelf - 1, int(rng.integers(a, b - length + 2)), length, cargo)
        assert _put(pieces, fixed, cargo_piece, shape)
        if second or rng.random() < .35:
            # Another independent piece rides on the cargo. In a two-source
            # scene it survives the new source's destruction of its carrier.
            top, col = cargo_piece[1], cargo_piece[2]
            if top >= 1:
                assert _put(pieces, fixed, ("H", top - 1, col, 2, palette[7]), shape)
        # A floor rod INSIDE the hollow lower beam is a controlled counterexample
        # to full-cap propagation and global color deletion, not random noise.
        filled = _beams(analysis, filled=True)
        safe = [cc for cc in range(left, right) if
                all(lower not in beams.get((h - 1, k), ()) for k in (cc, cc + 1)) and
                any(lower in filled.get((h - 1, k), ()) for k in (cc, cc + 1))]
        rng.shuffle(safe)
        for cc in safe:
            if _put(pieces, fixed, ("H", h - 1, cc, 2, lower), shape):
                break
        else:
            continue
        # A nonmatching floor stop visibly occludes one beam but is preserved.
        floor_hits = [cc for cc in range(left, right) if (h - 1, cc) in beams]
        rng.shuffle(floor_hits)
        for cc in floor_hits:
            if _put(pieces, fixed, ("H", h - 1, cc, 2, palette[6]), shape):
                break
        target = len(pieces) + extras
        for _ in range(extras * 10):
            if len(pieces) >= target:
                break
            orientation = "H" if rng.random() < .58 else "V"
            length = int(rng.integers(2, min(10, right - left + 1) + 1)) if orientation == "H" else int(rng.integers(2, 8))
            col = int(rng.integers(left, right - length + 2)) if orientation == "H" else int(rng.integers(left, right + 1))
            color = int(rng.choice(palette[3:]))
            _drop(pieces, fixed, orientation, col, length, color, shape)
        analysis["pieces"] = sorted(pieces)
        inp = _render(analysis, dict(enumerate(analysis["pieces"])), {})
        try:
            decoded = _parse(inp)
            assert decoded["pieces"] == analysis["pieces"]
            assert decoded["fixed"] == fixed
            # Reject non-interacting sources before running all six ablations.
            live, events = _evolve(decoded["pieces"], shape, fixed, beams)
            assert events["initial"] and events["late"]
            assert any(events["distances"][i] for i in live)
            if second:
                assert decoded["pieces"].index(cargo_piece) in events["late"]
                only_first = dict(decoded, emitters=[e for e in decoded["emitters"]
                                                    if e["center"] == emitters[0]["center"]])
                alternative, _ = _evolve(decoded["pieces"], shape, fixed, _beams(only_first))
                assert _physical_signature(alternative) != _physical_signature(live)
                assert not np.array_equal(_render(decoded, alternative, beams),
                                          _render(decoded, live, beams))
            feature = _features(decoded)
            assert feature["events"]["initial"] and feature["events"]["late"]
            assert feature["moved_survivors"]
            assert all(feature["contrasts"].values())
            if witness:
                assert feature["nonmatching_hit"] and feature["safe"] and feature["partial_hits"]
        except AssertionError:
            continue
        out = _render(decoded, feature["live"], _beams(decoded))
        if reflected:
            inp, out = np.fliplr(inp).copy(), np.fliplr(out).copy()
        return {"input": inp, "output": out}
    raise AssertionError("bounded scene construction exhausted")


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    maximum_radius = int(rng.integers(2, 4))
    first_reflected = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = split == "train" and index < 2
            radius = maximum_radius if witness else int(rng.integers(2, maximum_radius + 1))
            reflected = (first_reflected ^ bool(index)) if witness else bool(rng.integers(2))
            second = not witness and (split == "test" or rng.random() < .4)
            extras = int(rng.integers(0, 5)) if witness else int(rng.integers(3, 8) if split == "train" else rng.integers(6, 12))
            for _ in range(12):
                pair = _make_pair(rng, radius, reflected=reflected, second=second,
                                  extras=extras, witness=witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, directions, trained_radii = set(), set(), set()
    direction_radii, tested_radii = {}, set()
    contrast_evidence = Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            analysis = _parse(inp)
            beams = _beams(analysis)
            assert all(len(colors) == 1 for colors in beams.values())
            assert not beams.keys() & analysis["fixed"].keys()
            settled, stable_events = _evolve(analysis["pieces"], inp.shape, analysis["fixed"], {})
            assert all(stable_events["distances"][i] == 0 for i in settled), "input is not gravity stable"
            feature = _features(analysis)
            live, events = feature["live"], feature["events"]
            assert np.array_equal(out, _render(analysis, live, beams))
            if len(analysis["emitters"]) > 1:
                for disabled in analysis["emitters"]:
                    fewer = dict(analysis, emitters=[e for e in analysis["emitters"] if e is not disabled])
                    alternative, _ = _evolve(analysis["pieces"], inp.shape, analysis["fixed"], _beams(fewer))
                    assert not np.array_equal(out, _render(analysis, alternative, beams)), "decorative source"
            assert events["initial"] and events["late"] and feature["moved_survivors"]
            assert all(feature["contrasts"].values())
            assert all(out[p] == color for p, color in analysis["fixed"].items())
            assert all(not any(piece[4] in beams.get(p, ()) for p in _cells(piece)) for piece in live.values())
            final, final_events = _evolve(list(live.values()), inp.shape, analysis["fixed"], beams)
            assert not final_events["deleted"] and not any(final_events["distances"])
            assert all(piece[1] >= analysis["pieces"][i][1] and piece[2:] == analysis["pieces"][i][2:]
                       and piece[0] == analysis["pieces"][i][0] for i, piece in live.items())
            if split == "train":
                for e in analysis["emitters"]:
                    directions.update(e["caps"])
                    trained_radii.add(e["radius"])
                    for direction in e["caps"]:
                        direction_radii[direction] = max(e["radius"], direction_radii.get(direction, 0))
                contrast_evidence.update(k for k, v in feature["contrasts"].items() if v)
                contrast_evidence["safe"] += bool(feature["safe"])
                contrast_evidence["nonmatching_hit"] += bool(feature["nonmatching_hit"])
                contrast_evidence["whole_rod"] += bool(feature["partial_hits"])
            else:
                assert len(analysis["emitters"]) >= 2
                assert all(e["radius"] <= max(trained_radii) for e in analysis["emitters"])
                tested_radii.update(e["radius"] for e in analysis["emitters"])
    assert directions == set(DIRECTIONS)
    assert min(direction_radii.values()) >= max(tested_radii)
    assert all(contrast_evidence[k] for k in ("filled", "global", "once", "no_gravity", "ignore_color",
                                            "settle_then_fire", "safe", "nonmatching_hit", "whole_rod"))


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """42 streaming seeds; structural/mechanical coverage, counts and mutations.

    At most one puzzle plus one determinism replay is retained. No files, no
    per-tick snapshots, and no random unbounded retries.
    """
    # Exact optical geometry, including the interior lane that must stay empty.
    optical = {"shape": (12, 13), "emitters": [
        {"center": (4, 4), "radius": 2, "caps": {(1, 1): 7}}]}
    assert _cap((4, 4), 2, (1, 1)) == [(5, 7), (6, 6), (7, 5)]
    ray_map = _beams(optical)
    assert ray_map[(6, 8)] == ray_map[(8, 6)] == {7}
    assert (7, 7) not in ray_map and (7, 7) in _beams(optical, filled=True)
    # Hand-specified transient interception: the shelf crosses row 4, is erased,
    # and its two-cell cargo reaches rows 5..6. A full fall before firing misses
    # that interception and leaves the shelf at row 6.
    pieces = [("V", 3, 2, 4, 1), ("H", 2, 1, 3, 2),
              ("V", 0, 2, 2, 3), ("H", 6, 5, 2, 2)]
    beams = {(3, 2): {1}, (4, 2): {2}}
    live, events = _evolve(pieces, (7, 8), {}, beams)
    assert live == {2: ("V", 5, 2, 2, 3), 3: pieces[3]}
    assert events["initial"] == {0} and events["late"] == {1}
    assert events["distances"] == [0, 2, 5, 0]
    assert 1 in _settle_then_fire(pieces, (7, 8), {}, beams)
    fingerprints, shapes, layouts, backgrounds = set(), set(), set(), set()
    counts, radii, lengths, distances, directions = set(), set(), set(), set(), set()
    repeated = redundant = nonmatching = late = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            a = _parse(pair["input"])
            f = _features(a)
            shapes.add(a["shape"])
            backgrounds.add(a["background"])
            counts.add(len(a["pieces"]))
            lengths.update(p[3] for p in a["pieces"])
            distances.update(f["events"]["distances"])
            radii.update(e["radius"] for e in a["emitters"])
            directions.update(d for e in a["emitters"] for d in e["caps"])
            layouts.add((a["shape"], tuple(p[:4] for p in a["pieces"]),
                         tuple((e["center"], e["radius"]) for e in a["emitters"])))
            repeated += len({p[4] for p in a["pieces"]}) < len(a["pieces"])
            redundant += sum(p[0] == "V" and p[4] in {c for e in a["emitters"] for c in e["caps"].values()}
                             for p in a["pieces"]) >= 2
            nonmatching += bool(f["nonmatching_hit"])
            late += bool(f["events"]["late"])
            assert np.array_equal(solve(pair["input"].copy()), pair["output"])
        if seed in (0, 17, 41):
            assert _fingerprint(generate_puzzle(seed)) == _fingerprint(puzzle)
    assert len(fingerprints) == 42 and len(layouts) >= 200
    assert len(shapes) >= 25 and len(backgrounds) >= 60 and len(counts) >= 10
    assert radii == {2, 3} and directions == set(DIRECTIONS)
    assert max(lengths) >= 20 and max(distances) >= 12
    assert repeated >= 100 and redundant >= 20 and nonmatching >= 80 and late == 210
    # The host verifier exhausts all 30 combinations. Keep this repeated test
    # bounded while covering both extrema and every legal test count.
    for a, b in ((2, 1), (3, 2), (4, 3), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(2900 + a * 10 + b, a, b)
        assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A damaged answer must be rejected, even when all schema checks still pass.
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"] = bad["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an unchanged answer")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
