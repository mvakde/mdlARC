"""Clock-tailed arrows: periodic reception with whole-object cancellation.

Only NumPy and the standard library are used. No parent code, data, or files are
needed at runtime. The pixel solver has no puzzle/seed/global-state dependency.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 369215417, 58336707, 593557793, 718753619, 885280888,
              603101951, 942650277, 871508705, 218711223, 448627755,
              172060002, 953546286, 207623048, 315641460, 83280371,
              151019273, 176198456, 639146439, 268802391, 236005013,
              474504382, 37093969, 162173482, 704469380, 827214679,
              359279974, 48416638, 244458239, 387244682, 833409105,
              801616426, 587742629, 535751228, 826418254, 268757269,
              8339051, 487833286, 297505148, 515773844, 756126584, 341763404)


def _head(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return tuple((r + a, c + b) for a, b in offsets)


def _object(tip, direction, period, head_color, tail_color):
    tip, direction = tuple(map(int, tip)), tuple(map(int, direction))
    head = _head(tip, direction)
    tail = tuple((tip[0] - d * direction[0], tip[1] - d * direction[1])
                 for d in range(1, int(period) + 1))
    return {"tip": tip, "direction": direction, "period": int(period),
            "head_color": int(head_color), "tail_color": int(tail_color),
            "head": head, "tail": tail, "cells": head + tail}


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with sides 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _parse(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= a.size / 2:
        raise ValueError("Background must occupy more than half the input")
    remaining = set(map(tuple, np.argwhere(a != bg).tolist()))
    objects = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, frontier = {start}, [start]
        while frontier:
            r, c = frontier.pop()
            neighbors = {(r + dr, c + dc) for dr in (-1, 0, 1)
                         for dc in (-1, 0, 1)} & remaining
            component.update(neighbors)
            remaining.difference_update(neighbors)
            frontier.extend(neighbors)
        colors = {int(a[p]) for p in component}
        matches = []
        if len(colors) == 2:
            for head_color in sorted(colors):
                head = {p for p in component if a[p] == head_color}
                if len(head) != 3:
                    continue
                tail_color = next(c for c in colors if c != head_color)
                period = len(component) - 3
                if not 1 <= period <= 6:
                    continue
                for tip in sorted(head):
                    for direction in DIRECTIONS:
                        if set(_head(tip, direction)) != head:
                            continue
                        candidate = _object(tip, direction, period, head_color, tail_color)
                        if set(candidate["cells"]) == component:
                            matches.append(candidate)
        if len(matches) != 1:
            raise ValueError("Each separated component must be one uniquely clock-tailed arrow")
        objects.append(matches[0])
    if not objects:
        raise ValueError("At least one arrow is required")
    return a, bg, objects


def _collect(objects, shape, *, emission="periodic", pixels=False, active=None):
    """Pixel solver's explicit receiver map, always based on original objects."""
    owner = {p: j for j, obj in enumerate(objects) for p in obj["cells"]}
    receivers, events, coordinates = defaultdict(set), defaultdict(list), defaultdict(set)
    for i, obj in enumerate(objects):
        if active is not None and i not in active:
            continue
        start = obj["period"] if emission == "late" else 1
        step = 1 if emission == "continuous" else obj["period"]
        distances = (5,) if emission == "five" else range(start, 30, step)
        r, c = obj["tip"]
        dr, dc = obj["direction"]
        stopped = False
        for d in distances:
            p = (r + d * dr, c + d * dc)
            if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]):
                break
            if stopped:
                continue
            key = ("object", owner[p]) if p in owner and not pixels else ("cell", *p)
            receivers[key].add(i)
            events[key].append((i, p))
            coordinates[p].add(i)
            if emission == "stop" and p in owner:
                stopped = True
    return dict(receivers), dict(events), dict(coordinates)


def _predict(a, bg, objects, *, alternative=None):
    emission = alternative if alternative in ("continuous", "five", "late", "stop") else "periodic"
    pixels = alternative == "pixels"
    receivers, events, coords = _collect(objects, a.shape, emission=emission, pixels=pixels)
    if alternative == "suppress":
        erased = {key[1] for key, sources in receivers.items()
                  if key[0] == "object" and len(sources) >= 2}
        receivers, events, coords = _collect(objects, a.shape, active=set(range(len(objects))) - erased)
    out = a.copy()
    for key, sources in receivers.items():
        cells = objects[key[1]]["cells"] if key[0] == "object" else (key[1:],)
        cancel = len(sources) >= 2
        if alternative == "events":
            cancel = len(events[key]) >= 2
        elif alternative == "coordinate" and key[0] == "object":
            cancel = any(len(coords.get(p, ())) >= 2 for p in cells)
        color = bg if cancel else objects[min(sources)]["tail_color"]
        for p in cells:
            out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the output solely from pixels; accept every specified family member."""
    a, bg, objects = _parse(input_grid)
    return _predict(a, bg, objects)


def _construct(objects, shape, bg):
    """Label construction uses per-source NumPy masks, not the pixel solver."""
    inp = np.full(shape, bg, dtype=np.int8)
    masks = np.zeros((len(objects), *shape), dtype=bool)
    for i, obj in enumerate(objects):
        for p in obj["head"]:
            inp[p] = obj["head_color"]
        for p in obj["tail"]:
            inp[p] = obj["tail_color"]
        distances = np.arange(1, 30, obj["period"], dtype=int)
        points = np.array(obj["tip"]) + distances[:, None] * np.array(obj["direction"])
        points = points[np.all((points >= 0) & (points < np.array(shape)), axis=1)]
        masks[i, points[:, 0], points[:, 1]] = True
    out = inp.copy()
    counts = masks.sum(axis=0)
    for i, obj in enumerate(objects):
        out[(inp == bg) & (counts == 1) & masks[i]] = obj["tail_color"]
    # Background collisions already have the cancelling background value.
    for obj in objects:
        cells = np.array(obj["cells"])
        sources = np.flatnonzero(masks[:, cells[:, 0], cells[:, 1]].any(axis=1))
        if len(sources):
            color = bg if len(sources) >= 2 else objects[int(sources[0])]["tail_color"]
            out[cells[:, 0], cells[:, 1]] = color
    return {"input": inp, "output": out}


def _compatible(objects, candidate, shape):
    if any(not (0 <= r < shape[0] and 0 <= c < shape[1]) for r, c in candidate["cells"]):
        return False
    occupied = {p for obj in objects for p in obj["cells"]}
    return not any((r + dr, c + dc) in occupied for r, c in candidate["cells"]
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _move(obj, shift):
    return _object(tuple(a + b for a, b in zip(obj["tip"], shift)), obj["direction"],
                   obj["period"], obj["head_color"], obj["tail_color"])


def _choose(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _core(rng, kind, bg, periods=tuple(range(1, 7))):
    """Solve small contact equations; no saved grids or parent layouts."""
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != bg])]
    gap = 2 * int(rng.integers(3, 6))
    if kind == "single":
        period = int(_choose(rng, [p for p in periods if p >= 2]))
        # Two odd-distance pulses strike opposite wings. Off-ray tip/tail paint.
        objects = [_object((0, 0), (-1, 0), period, palette[0], palette[1]),
                   _object((1, -gap), (0, 1), 2, palette[2], palette[3])]
    else:
        eligible = [h for h in (4, 6) if h - 1 in periods and any(p >= h for p in periods)]
        height = int(_choose(rng, eligible))
        period = int(_choose(rng, [p for p in periods if p >= height]))
        diagonal_period = height - 1
        distance = 1 + diagonal_period * int(rng.integers(1, 3))
        # A cardinal and a diagonal source hit different parts of one target.
        # Their rays also meet on background beyond it at resonant distances.
        same_color = bool(rng.integers(2))
        objects = [_object((0, 0), (-1, 0), period, palette[0], palette[1]),
                   _object((1, -gap), (0, 1), 2, palette[2], palette[3]),
                   _object((height + distance, -distance), (-1, 1), diagonal_period,
                           palette[4], palette[3] if same_color else palette[5])]
    return objects


def _fit_core(rng, objects, *, spacious):
    points = [p for obj in objects for p in obj["cells"]]
    target = objects[0]
    # Room for at least two visible pulses from a struck/erased N transmitter.
    lo = [min(min(p[0] for p in points), -1 - target["period"]), min(p[1] for p in points)]
    hi = [max(p[0] for p in points), max(6, max(p[1] for p in points))]
    minimum = [hi[i] - lo[i] + 1 for i in (0, 1)]
    assert max(minimum) <= 30
    shape = []
    shift = []
    for axis in (0, 1):
        extra = int(rng.integers(0, min(30 - minimum[axis], 9 if spacious else 5) + 1))
        if spacious:
            extra = max(extra, min(30 - minimum[axis], 4))
        before = int(rng.integers(extra + 1))
        shape.append(minimum[axis] + extra)
        shift.append(before - lo[axis])
    return [_move(obj, shift) for obj in objects], tuple(shape)


def _add_random(rng, objects, shape, bg, directions, periods, desired, protect=False):
    initial_count = len(objects)
    _, _, core_coordinates = _collect(objects, shape)
    reserved = {p for p, sources in core_coordinates.items() if len(sources) >= 2} if protect else set()
    for _ in range(140):
        if len(objects) >= desired:
            break
        colors = rng.choice([c for c in range(10) if c != bg], 2, replace=False)
        candidate = _object((int(rng.integers(shape[0])), int(rng.integers(shape[1]))),
                            _choose(rng, directions), _choose(rng, periods), *colors)
        if not _compatible(objects, candidate, shape):
            continue
        if reserved.intersection(candidate["cells"]):
            continue
        trial = objects + [candidate]
        if sum(len(o["cells"]) for o in trial) >= shape[0] * shape[1] / 2:
            continue
        if protect:
            incoming, _, _ = _collect(trial, shape)
            # Preserve the teaching contacts and all original source emissions.
            if any(len(trial) - 1 in incoming.get(("object", i), ()) for i in range(initial_count)):
                continue
        objects.append(candidate)
    return objects


def _pair(rng, bg, kind, periods, rotation, reflect, test=False):
    objects = _core(rng, kind, bg, periods)
    objects, shape = _fit_core(rng, objects, spacious=test)
    directions = ((-1, 0), (0, 1), (-1, 1))
    desired = len(objects) + int(rng.integers(2, 8) if test else rng.integers(0, 4))
    objects = _add_random(rng, objects, shape, bg, directions, periods, desired, protect=True)
    pair = _construct(objects, shape, bg)
    if reflect:
        pair = {key: np.fliplr(value).copy() for key, value in pair.items()}
    return {key: np.rot90(value, rotation).copy() for key, value in pair.items()}


def _random_pair(rng, bg, periods, rotation, reflect):
    shape = (int(rng.integers(12, 31)), int(rng.integers(12, 31)))
    directions = ((-1, 0), (0, 1), (-1, 1))
    objects = _add_random(rng, [], shape, bg, directions, periods, int(rng.integers(2, 11)))
    assert objects
    pair = _construct(objects, shape, bg)
    if reflect:
        pair = {key: np.fliplr(value).copy() for key, value in pair.items()}
    return {key: np.rot90(value, rotation).copy() for key, value in pair.items()}


def _counts(num_train, num_test):
    for name, count, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or not low <= count <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
    palette = [int(c) for c in rng.permutation(10)]
    puzzle, seen = {"train": [], "test": []}, set()
    # These lengths are all visibly present in training before queries are made.
    for i, kind in enumerate(("single", "cancel")):
        pair = _pair(rng, palette[i], kind, (1, 2, 3, 4, 5, 6), rotation, reflect)
        seen.add(_key(pair["input"]))
        puzzle["train"].append(pair)
    periods = sorted({obj["period"] for p in puzzle["train"] for obj in _parse(p["input"])[2]})
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for i in range(count):
            for attempt in range(20):
                bg = palette[(i + (2 if split == "train" else num_train)) % 10]
                if split == "train":
                    pair = _random_pair(rng, bg, periods, rotation, reflect)
                else:
                    pair = _pair(rng, bg, _choose(rng, ("single", "cancel")), periods,
                                 (rotation + 2) % 4, reflect, test=True)
                key = _key(pair["input"])
                a, b, obs = _parse(pair["input"])
                # Each query visibly depends on both inherited mechanisms.
                good = all(not np.array_equal(pair["output"], _predict(a, b, obs, alternative=alt))
                           for alt in ("continuous", "pixels")) if split == "test" else True
                if key not in seen and good and not np.array_equal(pair["input"], pair["output"]):
                    puzzle[split].append(pair)
                    seen.add(key)
                    if split == "train":
                        periods = sorted(set(periods) | {o["period"] for o in obs})
                    break
            else:
                raise RuntimeError("Bounded unique-scene construction exhausted")
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(pairs):
    flags = {"repeated_single": False, "distinct_cell_cancel": False,
             "same_color_cancel": False, "background_cancel": False,
             "tail_hit": False, "continuation": False}
    periods, directions = set(), set()
    for pair in pairs:
        a, bg, objects = _parse(pair["input"])
        receivers, events, coords = _collect(objects, a.shape)
        periods.update(o["period"] for o in objects)
        directions.update(o["direction"] for o in objects)
        for key, sources in receivers.items():
            if key[0] == "cell":
                flags["background_cancel"] |= len(sources) >= 2
                continue
            target = objects[key[1]]
            flags["tail_hit"] |= any(p in target["tail"] for _, p in events[key])
            flags["repeated_single"] |= len(sources) == 1 and len(events[key]) >= 2
            flags["distinct_cell_cancel"] |= (len(sources) >= 2
                                               and all(len(coords.get(p, ())) <= 1 for p in target["cells"]))
            flags["same_color_cancel"] |= (len(sources) >= 2
                                            and len({objects[s]["tail_color"] for s in sources}) == 1)
            # A source that hits an object has a later visible singleton pulse.
            for source, point in events[key]:
                obj = objects[source]
                hit_d = max(abs(point[k] - obj["tip"][k]) for k in (0, 1))
                for other, contributors in receivers.items():
                    if other[0] == "cell" and contributors == {source}:
                        d = max(abs(other[k + 1] - obj["tip"][k]) for k in (0, 1))
                        flags["continuation"] |= d > hit_d
    return flags, periods, directions


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, parsed = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, bg, objects = _parse(pair["input"])
            out = _grid(pair["output"])
            assert a.shape == out.shape
            assert _key(a) not in seen, "Repeated input"
            seen.add(_key(a))
            assert np.array_equal(out, _predict(a, bg, objects)), "Wrong receiver effect or changed preserved cell"
            assert not np.array_equal(a, out), "Generated examples must reveal an effect"
            receivers, _, _ = _collect(objects, a.shape)
            for i, obj in enumerate(objects):
                sources = receivers.get(("object", i), set())
                values = [int(out[p]) for p in obj["cells"]]
                if sources:
                    expected = bg if len(sources) >= 2 else objects[next(iter(sources))]["tail_color"]
                    assert set(values) == {expected}, "Partial object update"
                else:
                    assert all(out[p] == a[p] for p in obj["cells"]), "Unstruck object changed"
            parsed[split].append((a, bg, objects))
    flags, periods, directions = _evidence(puzzle["train"])
    assert flags["repeated_single"] and flags["distinct_cell_cancel"] and flags["tail_hit"]
    assert flags["background_cancel"] and flags["continuation"]
    assert len(periods) >= 2
    for alt in ("continuous", "five", "late", "pixels", "coordinate", "events", "suppress", "stop"):
        assert any(not np.array_equal(p["output"], _predict(a, bg, objects, alternative=alt))
                   for p, (a, bg, objects) in zip(puzzle["train"], parsed["train"])), f"Teaching gap: {alt}"
    test_dirs, test_periods = set(), set()
    for pair, (a, bg, objects) in zip(puzzle["test"], parsed["test"]):
        test_dirs.update(o["direction"] for o in objects)
        test_periods.update(o["period"] for o in objects)
        for alt in ("continuous", "pixels"):
            assert not np.array_equal(pair["output"], _predict(a, bg, objects, alternative=alt))
    assert directions.isdisjoint(test_dirs), "Queries must use held-out directions"
    assert test_periods <= periods, "Query introduced an untaught period"


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _regressions():
    # Fixed expected receivers, independent of generated labels.
    objects = [_object((8, 10), (-1, 0), 4, 3, 4),
               _object((9, 2), (0, 1), 2, 5, 6)]
    pair = _construct(objects, (20, 23), 0)
    out = solve(pair["input"])
    assert all(out[p] == 6 for p in objects[0]["cells"])
    assert out[9, 3] == 6 and out[9, 4] == 0 and out[9, 5] == 6
    assert out[7, 10] == 4 and out[3, 10] == 4 and out[6, 10] == 0
    # Third arrow hits a different tail cell; all target cells must disappear.
    objects.append(_object((16, 6), (-1, 1), 3, 7, 6))
    pair = _construct(objects, (23, 25), 0)
    out = solve(pair["input"])
    assert all(out[p] == 0 for p in objects[0]["cells"])
    assert out[7, 10] == 4, "An erased source still fires"
    assert out[9, 13] == 0 and out[9, 15] == 6, "Cancellation and continuation"
    # A third transmitter, of the same color, must still cancel, not restore.
    objects.append(_object((19, 10), (-1, 0), 2, 8, 6))
    pair = _construct(objects, (25, 25), 0)
    out = solve(pair["input"])
    assert all(out[p] == 0 for p in objects[0]["cells"])
    # Whole-family symmetries and color permutations, including changed bg.
    mapping = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
    for reflected in (False, True):
        for turn in range(4):
            inp = np.fliplr(pair["input"]) if reflected else pair["input"]
            expected = np.fliplr(out) if reflected else out
            assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(expected, turn))
            assert np.array_equal(solve(mapping[np.rot90(inp, turn)]), mapping[np.rot90(expected, turn)])


def self_test():
    """Bounded, disk-free streaming coverage over the 42 supplied seeds."""
    _regressions()
    fingerprints, shapes, directions, periods, counts, backgrounds, structures = (set() for _ in range(7))
    flags_seen = defaultdict(int)
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        flags, _, _ = _evidence(puzzle["train"])
        for flag, present in flags.items():
            flags_seen[flag] += bool(present)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, bg, objects = _parse(pair["input"])
                before = a.copy()
                assert np.array_equal(solve(a), pair["output"])
                assert np.array_equal(a, before), "Solver mutated its input"
                shapes.add(a.shape)
                backgrounds.add(bg)
                directions.update(o["direction"] for o in objects)
                periods.update(o["period"] for o in objects)
                counts.add(len(objects))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, hashlib.sha256(normalized).digest()))
    assert len(fingerprints) == 42
    assert len(shapes) >= 45 and len(structures) >= 150
    assert directions == set(DIRECTIONS) and periods == set(range(1, 7))
    assert len(counts) >= 7 and len(backgrounds) == 10
    assert flags_seen["same_color_cancel"] >= 8
    # Determinism, nondefault counts, and strict count types/ranges.
    for train, test in ((2, 1), (10, 4)):
        a = generate_puzzle(991, train, test)
        assert _fingerprint(a) == _fingerprint(generate_puzzle(991, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    puzzle = generate_puzzle(991, 2, 1)
    puzzle["test"][0]["output"] = puzzle["test"][0]["output"].copy()
    puzzle["test"][0]["output"][0, 0] = (int(puzzle["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return {"seeds": len(HOST_SEEDS), "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "normalized_inputs": len(structures),
            "directions": len(directions), "periods": sorted(periods),
            "object_counts": sorted(counts), "backgrounds": len(backgrounds),
            "teaching_flags": dict(flags_seen)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
