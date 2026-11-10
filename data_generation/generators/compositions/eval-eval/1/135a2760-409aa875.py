"""Periodic redundancy repairs arrowheads; only repaired arrows fire one period.

No files, training labels, seed, or generation metadata enter solve().
All diagnostics are bounded and in memory; this module writes no artifacts.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


VECTORS = ((-1, 0), (1, 0), (0, -1), (0, 1),
           (-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _arrow(r, c, dr, dc):
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return (r, c, dr, dc, tuple(sorted((r + a, c + b) for a, b in offsets)))


def _arrows(motif, background):
    remaining = set(map(tuple, np.argwhere(motif != background).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            neighbors = {(r + a, c + b) for a in (-1, 0, 1)
                         for b in (-1, 0, 1)} & remaining
            remaining -= neighbors
            component.update(neighbors)
            todo.extend(sorted(neighbors))
        matches = [_arrow(r, c, dr, dc) for r, c in sorted(component)
                   for dr, dc in VECTORS
                   if set(_arrow(r, c, dr, dc)[4]) == component]
        if len(component) != 3 or len(matches) != 1:
            raise ValueError("Recovered foreground must be separated three-cell arrows")
        result.append(matches[0])
    if not result:
        raise ValueError("Each lane must contain arrowheads")
    return sorted(result)


def _recover(motif):
    """Compare every supported strict-majority periodic erasure explanation."""
    colors, counts = np.unique(motif, return_counts=True)
    if len(colors) != 2 or counts.max() * 2 <= motif.size:
        raise ValueError("Lane needs a majority background and one arrow color")
    background = int(colors[counts.argmax()])
    accent = int(colors[counts.argmin()])
    bits = motif == accent
    cross, length = bits.shape
    candidates = []
    for period in range(1, length // 3 + 1):
        unit = np.zeros((cross, period), dtype=bool)
        valid = True
        for phase in range(period):
            samples = bits[:, phase::period]
            votes = samples.sum(axis=1)
            if np.any(2 * votes == samples.shape[1]):
                valid = False
                break
            unit[:, phase] = 2 * votes > samples.shape[1]
        if not valid:
            continue
        repaired = unit[:, np.arange(length) % period]
        # Foreground observations are trusted; only absent pixels are repaired.
        if np.any(bits & ~repaired):
            continue
        errors = int(np.count_nonzero(repaired != bits))
        if errors > max(1, motif.size // 10):
            continue
        candidates.append((period, repaired))
    if not candidates:
        raise ValueError("No supported periodic repair")
    period, repaired = candidates[0]
    if any(not np.array_equal(repaired, other) for _, other in candidates):
        raise ValueError("Competing periods imply different repairs")
    clean = np.where(repaired, accent, background).astype(np.int8)
    _arrows(clean, background)
    return clean, period, background, accent


def _scene(value):
    original = _grid(value)
    frame = int(original[0, 0])
    if not (np.all(original[0] == frame) and np.all(original[-1] == frame)
            and np.all(original[:, 0] == frame) and np.all(original[:, -1] == frame)):
        raise ValueError("Missing outer frame")
    rows = np.flatnonzero(np.all(original == frame, axis=1))
    cols = np.flatnonzero(np.all(original == frame, axis=0))
    if len(rows) > 2 and len(cols) > 2:
        raise ValueError("Lanes must be parallel")
    transpose = (len(cols) > 2 or
                 (len(rows) == len(cols) == 2 and original.shape[0] > original.shape[1]))
    a = original.T.copy() if transpose else original.copy()
    h, w = a.shape
    dividers = np.flatnonzero(np.all(a == frame, axis=1)).tolist()
    if (np.flatnonzero(np.all(a == frame, axis=0)).tolist() != [0, w - 1]
            or len(dividers) < 2 or min(np.diff(dividers)) < 6):
        raise ValueError("Invalid frame geometry")
    padding = int(a[1, 1])
    if padding == frame:
        raise ValueError("Padding must differ from frame")
    clean = a.copy()
    mask = np.zeros(a.shape, dtype=bool)
    fixed = np.zeros(a.shape, dtype=bool)
    fixed[:, (0, w - 1)] = True
    fixed[dividers, :] = True
    objects, lanes, backgrounds, accents = [], [], [], []
    for lane_id, (top, bottom) in enumerate(zip(dividers[:-1], dividers[1:])):
        rect = (slice(top + 2, bottom - 1), slice(2, w - 2))
        motif = a[rect]
        if not 3 <= motif.shape[0] < motif.shape[1]:
            raise ValueError("Lane motif must have a distinct long axis")
        repaired, period, background, accent = _recover(motif)
        if len({frame, padding, background, accent}) != 4:
            raise ValueError("Structural and motif roles must be distinct")
        clean[rect] = repaired
        mask[rect] = True
        backgrounds.append(background)
        accents.append(accent)
        lanes.append({"rect": rect, "period": period, "cross": motif.shape[0],
                      "length": motif.shape[1], "accent": accent})
        for obj in _arrows(repaired, background):
            r, c, dr, dc, _ = obj
            moved = _arrow(r + top + 2, c + 2, dr, dc)
            objects.append({"arrow": moved, "lane": lane_id, "period": period})
    if len(set(backgrounds)) != 1 or len(set(accents)) != len(accents):
        raise ValueError("Lanes share one void color and have distinct accents")
    if not np.all(a[~(fixed | mask)] == padding):
        raise ValueError("Padding must be a uniform one-cell margin")
    owner = {cell: i for i, obj in enumerate(objects) for cell in obj["arrow"][4]}
    missing = set(map(tuple, np.argwhere(a != clean).tolist()))
    if not missing <= set(owner):
        raise ValueError("Every defect must be an erased arrow pixel")
    active = sorted({owner[p] for p in missing})
    return {"input": a, "clean": clean, "transpose": transpose,
            "frame": frame, "padding": padding, "background": backgrounds[0],
            "lanes": lanes, "objects": objects, "owner": owner,
            "mask": mask, "missing": missing, "active": active}


def _events(scene, active=None, distance=None, strict=True):
    """Strict production events; permissive mode is only a shortcut diagnostic."""
    if active is None:
        active = scene["active"]
    objects, owner, mask = scene["objects"], scene["owner"], scene["mask"]
    counts = Counter()
    shooters = {}
    for i in active:
        obj = objects[i]
        r, c, dr, dc, _ = obj["arrow"]
        d = obj["period"] if distance is None else distance
        point = (r + d * dr, c + d * dc)
        legal = (0 <= point[0] < mask.shape[0] and 0 <= point[1] < mask.shape[1]
                 and mask[point])
        if not legal:
            if strict:
                raise ValueError("An activated arrow must end inside a motif")
            continue
        counts[point] += 1
        shooters.setdefault(point, []).append(i)
    paint, hits = {}, {}
    for point, count in sorted(counts.items()):
        color = scene["frame"] if count == 1 else scene["padding"]
        if point in owner:
            target = owner[point]
            hits.setdefault(target, []).append((point, count))
            cells = objects[target]["arrow"][4]
        else:
            cells = (point,)
        for cell in cells:
            if cell in paint and paint[cell] != color:
                if strict:
                    raise ValueError("One target cannot receive conflicting impact roles")
                color = scene["padding"]  # Defined only for the ablation.
            paint[cell] = color
    return {"counts": counts, "shooters": shooters, "paint": paint, "hits": hits}


def _render(scene, active=None, distance=None, strict=True):
    events = _events(scene, active, distance, strict)
    result = scene["clean"].copy()
    for point, color in events["paint"].items():
        result[point] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _scene(input_grid)
    result = _render(scene)
    return result.T.copy() if scene["transpose"] else result


def _phases(period, direction, length):
    dr, dc = direction
    result = []
    tail = length % period
    for c in range(period):
        cells = _arrow(1, c, dr, dc)[4]
        columns = [x[1] for x in cells]
        if min(columns) < 0 or max(columns) >= period:
            continue
        if tail and not (max(columns) < tail or min(columns) >= tail):
            continue
        result.append(c)
    return result


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _lane(rng, kind, length, max_cross=13):
    periods = [p for p in range(4, min(8, length // 3) + 1)
               if kind not in ("open", "collision") or p + 3 <= max_cross]
    if not periods:
        kind = "chain"
        periods = list(range(4, min(8, length // 3) + 1))
    period = int(_choice(rng, periods))
    if kind == "collision":
        cross = int(rng.integers(period + 3, min(max_cross, period + 5, length - 1) + 1))
        offset = int(rng.integers(cross - period - 2))
        vertical, horizontal = _choice(rng, (-1, 1)), _choice(rng, (-1, 1))
        target_row = offset + (1 if vertical == 1 else period + 1)
        phase = int(_choice(rng, _phases(period, (0, horizontal), length)))
        strands = [(target_row, phase, 0, horizontal),
                   (target_row + vertical * period, phase, -vertical, horizontal)]
    elif kind == "open":
        cross = int(rng.integers(period + 3, min(max_cross, period + 5, length - 1) + 1))
        dr = _choice(rng, (-1, 1))
        directions = [(dr, dc) for dc in (-1, 0, 1) if _phases(period, (dr, dc), length)]
        dr, dc = _choice(rng, directions)
        offset = int(rng.integers(cross - period - 2))
        row = offset + (period + 1 if dr == -1 else 1)
        phase = int(_choice(rng, _phases(period, (dr, dc), length)))
        strands = [(row, phase, dr, dc)]
    else:
        cross = int(rng.integers(3, min(max_cross, 12, length - 1) + 1))
        strands = []
        desired = 1 if kind == "chain" else int(rng.integers(1, min(3, (cross + 1) // 4) + 1))
        for _ in range(45):
            if len(strands) >= desired:
                break
            direction = (0, _choice(rng, (-1, 1))) if kind == "chain" else _choice(rng, VECTORS)
            phases = _phases(period, direction, length)
            if not phases:
                continue
            row = int(rng.integers(1, cross - 1))
            candidate = (row, int(_choice(rng, phases)), *direction)
            trial = strands + [candidate]
            field = np.zeros((cross, 3 * period), dtype=np.int8)
            overlap = False
            for r, c, dr, dc in trial:
                for k in range(3):
                    for point in _arrow(r, c + k * period, dr, dc)[4]:
                        if field[point]:
                            overlap = True
                        field[point] = 1
            try:
                parsed = _arrows(field, 0)
            except ValueError:
                continue
            if not overlap and len(parsed) == 3 * len(trial):
                strands = trial
        if not strands:
            # Geometry-derived finite fallback, still using the sampled period.
            direction = (0, _choice(rng, (-1, 1)))
            strands = [(1, int(_choice(rng, _phases(period, direction, length))), *direction)]
    return {"period": period, "cross": cross, "strands": strands}


def _clean_scene(rng, length, kind, colors):
    if kind in ("bridge", "bridge_open"):
        distances = list(range(6, min(8, length // 3) + 1))
        if not distances:
            raise ValueError("A cross-lane shot requires a longer repetition")
        source_period = int(_choice(rng, distances))
        target_period = int(_choice(rng, [p for p in range(4, min(8, length // 3) + 1)
                                        if p != source_period]))
        horizontal = _choice(rng, (-1, 1))
        source_phases = _phases(source_period, (-1, 0), length)
        target_phases = _phases(target_period, (0, horizontal), length)
        combinations = [(a, b) for a in source_phases for b in target_phases
                        if (a == b if kind == "bridge" else a != b)]
        if not combinations:
            raise ValueError("No complete boundary-safe cross-lane arrow arrangement")
        source_phase, target_phase = _choice(rng, combinations)
        specs = [{"cross": source_period - 3, "period": target_period,
                  "strands": [(1, target_phase, 0, horizontal)]},
                 {"cross": int(rng.integers(3, 7)), "period": source_period,
                  "strands": [(1, source_phase, -1, 0)]}]
        primary_index = 1
    else:
        specs = [_lane(rng, kind, length)]
        primary_index = 0
    initial_count = len(specs)
    budget = 29 - sum(spec["cross"] + 3 for spec in specs)
    max_count = min(4, initial_count + budget // 6)
    count = int(rng.integers(initial_count, max_count + 1))
    for i in range(initial_count, count):
        cap = budget - 6 * (count - i - 1) - 3
        other = _lane(rng, _choice(rng, ("chain", "free")), length, max_cross=cap)
        specs.append(other)
        budget -= other["cross"] + 3
    order = rng.permutation(count).tolist()
    # Keep the two bridge lanes adjacent: their separation encodes the shot.
    if kind in ("bridge", "bridge_open"):
        block_at = int(rng.integers(count - 1))
        extras = [i for i in order if i not in (0, 1)]
        order = extras[:block_at] + [0, 1] + extras[block_at:]
    engine = order.index(primary_index)
    specs = [specs[i] for i in order]
    frame, padding, background = map(int, colors[:3])
    height = 1 + sum(spec["cross"] + 3 for spec in specs)
    a = np.full((height, length + 4), padding, dtype=np.int8)
    a[:, (0, -1)] = frame
    top = 0
    for i, spec in enumerate(specs):
        cross, period = spec["cross"], spec["period"]
        bottom = top + cross + 3
        a[top, :] = a[bottom, :] = frame
        unit = np.full((cross, period), background, dtype=np.int8)
        for r, c, dr, dc in spec["strands"]:
            for point in _arrow(r, c, dr, dc)[4]:
                unit[point] = int(colors[3 + i])
        a[top + 2:bottom - 1, 2:-2] = unit[:, np.arange(length) % period]
        top = bottom
    return a, engine


def _select_active(rng, scene, engine, kind):
    objects = scene["objects"]
    valid = []
    for i in range(len(objects)):
        try:
            event = _events(scene, [i])
        except ValueError:
            continue
        valid.append((i, next(iter(event["counts"]))))
    engine_valid = [(i, p) for i, p in valid if objects[i]["lane"] == engine]
    if kind in ("open", "bridge_open"):
        choices = [i for i, p in engine_valid if p not in scene["owner"]
                   and (kind == "open" or not (
                       scene["lanes"][engine]["rect"][0].start <= p[0]
                       < scene["lanes"][engine]["rect"][0].stop))]
        if not choices:
            raise ValueError("Need an empty endpoint demonstration")
        return [int(_choice(rng, choices))]
    if kind == "bridge":
        valid_ids = {i for i, _ in valid}
        choices = []
        for i, p in engine_valid:
            target = scene["owner"].get(p)
            if (target in valid_ids and objects[i]["lane"] != objects[target]["lane"]
                    and objects[i]["period"] != objects[target]["period"]):
                active = sorted([i, target])
                _events(scene, active)
                choices.append(active)
        if not choices:
            raise ValueError("Need a cross-lane target that also fires")
        return _choice(rng, choices)
    if kind == "collision":
        grouped = {}
        for i, p in engine_valid:
            grouped.setdefault(p, []).append(i)
        choices = []
        valid_ids = {i for i, _ in valid}
        for p, ids in sorted(grouped.items()):
            target = scene["owner"].get(p)
            if len(ids) >= 2 and target in valid_ids and target not in ids:
                active = sorted(ids[:2] + [target])
                event = _events(scene, active)
                if (any(n == 1 and q in scene["owner"] for q, n in event["counts"].items())
                        and any(n >= 2 for n in event["counts"].values())):
                    choices.append(active)
        if not choices:
            raise ValueError("Need a convergence whose repaired target also fires")
        return _choice(rng, choices)
    ids = [i for i, _ in valid]
    if not ids:
        raise ValueError("No legal activated arrows")
    count = int(rng.integers(1, min(7, len(ids), len(objects) - 1) + 1))
    active = sorted(int(i) for i in rng.choice(ids, size=count, replace=False))
    _events(scene, active)
    return active


def _damage(rng, scene, active, force_tip=False):
    a = scene["clean"].copy()
    used = set()
    budget = max(1, int(np.count_nonzero(scene["mask"])) // 10)
    removed = 0
    ordered = list(active)
    rng.shuffle(ordered)
    for serial, i in enumerate(ordered):
        obj = scene["objects"][i]
        lane = scene["lanes"][obj["lane"]]
        top, left = lane["rect"][0].start, lane["rect"][1].start
        cells = list(obj["arrow"][4])
        rng.shuffle(cells)
        tip = obj["arrow"][:2]
        if force_tip and serial == 0:
            cells.remove(tip)
            cells.insert(0, tip)
        available = [(p, (obj["lane"], p[0] - top, (p[1] - left) % obj["period"]))
                     for p in cells]
        available = [(p, key) for p, key in available if key not in used]
        if not available:
            raise ValueError("Too many defects at one repeated phase")
        cap = min(len(available), budget - removed - (len(ordered) - serial - 1))
        if cap < 1:
            raise ValueError("Erasure budget exhausted")
        count = int(rng.integers(1, cap + 1))
        for point, key in available[:count]:
            a[point] = scene["background"]
            used.add(key)
            removed += 1
    return a


def _pair(rng, kind, length, colors):
    clean, engine = _clean_scene(rng, length, kind, colors)
    original_scene = _scene(clean)
    active = _select_active(rng, original_scene, engine, kind)
    a = _damage(rng, original_scene, active, force_tip=(kind in ("open", "bridge_open")))
    scene = _scene(a)
    if not np.array_equal(scene["clean"], clean) or scene["active"] != sorted(active):
        raise ValueError("Corruption did not preserve the intended unique reconstruction")
    b = _render(scene)
    if np.array_equal(b, clean):
        raise ValueError("Need a visible projection effect")
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    def transform(grid):
        grid = np.rot90(grid, turns)
        return (np.fliplr(grid) if reflect else grid).copy()
    return {"input": transform(a), "output": transform(b)}


def _integer(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _integer("num_train", num_train, 2, 10)
    num_test = _integer("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.permutation(10)]
    puzzle, seen = {"train": [], "test": []}, set()
    first_period = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            essential = split == "train" and index < 2
            kind = ("bridge_open", "collision")[index] if essential else (
                "bridge" if split == "train" and index == 2 else _choice(
                    rng, ("chain", "open", "collision", "bridge", "free", "free")))
            serial = len(seen)
            colors = palette[serial % 10:] + palette[:serial % 10]
            for attempt in range(80):
                length = int(rng.integers(12, 22)) if split == "train" else int(rng.integers(22, 27))
                # A second distinct visible scale rules out a single fixed distance.
                if essential and index == 1 and length // 3 == 4 and first_period == 4:
                    continue
                try:
                    pair = _pair(rng, kind, length, colors)
                    scene = _scene(pair["input"])
                    active_periods = {scene["objects"][i]["period"] for i in scene["active"]}
                    if essential and index == 1 and active_periods == {first_period}:
                        continue
                    key = (pair["input"].shape, pair["input"].tobytes())
                    if key in seen:
                        continue
                except ValueError:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if essential and index == 0:
                    first_period = next(iter(active_periods))
                break
            else:
                raise RuntimeError("Bounded geometric construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    scene = _scene(pair["input"])
    result = _render(scene)
    expected = _grid(pair["output"])
    expected = expected.T if scene["transpose"] else expected
    assert np.array_equal(result, expected), "Incorrect output"
    events = _events(scene)
    owner, objects = scene["owner"], scene["objects"]
    assert scene["missing"] and 0 < len(scene["active"]) < len(objects)
    assert not np.array_equal(result, scene["clean"]), "Projection must affect the answer"
    assert np.array_equal(scene["input"][~scene["mask"]], result[~scene["mask"]])
    groups = Counter()
    for r, c in scene["missing"]:
        i = owner[r, c]
        obj = objects[i]
        lane = scene["lanes"][obj["lane"]]
        groups[obj["lane"], r - lane["rect"][0].start,
               (c - lane["rect"][1].start) % obj["period"]] += 1
    assert max(groups.values()) == 1, "Generated errors must preserve majority evidence"
    counts = events["counts"]
    return {"scene": scene, "events": events, "shape": tuple(pair["input"].shape),
            "lanes": len(scene["lanes"]), "length": scene["lanes"][0]["length"],
            "crosses": tuple(lane["cross"] for lane in scene["lanes"]),
            "periods": tuple(lane["period"] for lane in scene["lanes"]),
            "directions": {obj["arrow"][2:4] for obj in objects},
            "arrows": len(objects), "active": len(scene["active"]),
            "erasures": len(scene["missing"]),
            "erased_arrows": sum(all(p in scene["missing"] for p in obj["arrow"][4])
                                  for obj in objects),
            "single_empty": any(n == 1 and p not in owner for p, n in counts.items()),
            "single_arrow": any(n == 1 and p in owner for p, n in counts.items()),
            "collision": any(n >= 2 for n in counts.values()),
            "active_hit": bool(set(events["hits"]) & set(scene["active"])),
            "visible_heal": any(result[p] == scene["clean"][p] for p in scene["missing"]),
            "cross_lane": any(objects[i]["lane"] != objects[owner[p]]["lane"]
                              for p, ids in events["shooters"].items() if p in owner for i in ids),
            "different_period_strike": any(objects[i]["period"] != objects[owner[p]]["period"]
                                            for p, ids in events["shooters"].items() if p in owner
                                            for i in ids),
            "cross_lane_empty": any(
                any(lane["rect"][0].start <= p[0] < lane["rect"][0].stop
                    and j != objects[i]["lane"] for j, lane in enumerate(scene["lanes"]))
                for p, ids in events["shooters"].items() if p not in owner for i in ids),
            "erased_tip": any(obj["arrow"][:2] in scene["missing"] for obj in objects),
            "whole_arrow": any(sum(cell in events["paint"] for cell in objects[i]["arrow"][4]) == 3
                               for i in events["hits"])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            record = _features({"input": a, "output": b})
            assert all(4 <= p <= 8 for p in record["periods"])
            assert 1 <= record["lanes"] <= 4
            assert (12 <= record["length"] <= 21 if split == "train"
                    else 22 <= record["length"] <= 26)
            records[split].append(record)
    train = records["train"]
    for feature in ("single_empty", "cross_lane_empty", "single_arrow", "collision", "active_hit",
                    "visible_heal", "erased_tip", "whole_arrow"):
        assert any(r[feature] for r in train), f"Missing teaching evidence: {feature}"
    if len(train) >= 3:
        assert any(r["cross_lane"] and r["different_period_strike"] for r in train)
    assert len({r["scene"]["objects"][i]["period"] for r in train
                for i in r["scene"]["active"]}) >= 2
    # These ablations use only input-derived information, never seed parameters.
    assert any(not np.array_equal(_render(r["scene"], distance=5, strict=False),
                                  _render(r["scene"])) for r in train)
    assert any(not np.array_equal(_render(r["scene"], active=range(r["arrows"]), strict=False),
                                  _render(r["scene"])) for r in train)


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _regression_tests():
    """Small decoded-snapshot fixtures with explicit geometric expectations."""
    def snapshot(specs):
        clean = np.zeros((20, 20), dtype=np.int8)
        objects = []
        for i, (tip, direction, period) in enumerate(specs):
            arrow = _arrow(*tip, *direction)
            objects.append({"arrow": arrow, "period": period, "lane": i})
            for p in arrow[4]:
                clean[p] = 4 + i
        return {"clean": clean, "mask": np.ones((20, 20), dtype=bool),
                "objects": objects, "owner": {p: i for i, obj in enumerate(objects)
                                               for p in obj["arrow"][4]},
                "frame": 1, "padding": 2, "background": 0,
                "active": list(range(len(objects)))}

    # Two separate single endpoints on one arrow do not make a collision.
    specs = [((8, 9), (0, 1), 4), ((14, 9), (-1, 0), 6), ((13, 2), (-1, 1), 6)]
    scene = snapshot(specs)
    events = _events(scene)
    assert len(events["hits"][0]) == 2 and all(n == 1 for n in events["counts"].values())
    out = _render(scene)
    assert all(out[p] == 1 for p in scene["objects"][0]["arrow"][4])
    assert out[8, 13] == 1  # Target fires at its own period four, not six.
    assert np.array_equal(out, _render(scene, active=[2, 1, 0]))
    incompatible = snapshot(specs + [((2, 9), (1, 0), 6)])
    try:
        _events(incompatible)
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting whole-object roles accepted")

    # Triple convergence on an active arrow; being struck does not stop it.
    scene = snapshot([((8, 8), (0, 1), 4), ((8, 2), (0, 1), 6),
                      ((14, 8), (-1, 0), 6), ((2, 2), (1, 1), 6)])
    out = _render(scene)
    assert _events(scene)["counts"][8, 8] == 3
    assert all(out[p] == 2 for p in scene["objects"][0]["arrow"][4])
    assert out[8, 12] == 1
    dormant = _render(scene, active=[1])
    assert all(dormant[p] == 1 for p in scene["objects"][0]["arrow"][4])
    assert dormant[8, 12] == 0  # An intact struck target acquires no shot.

    # Paths are not painted, and intermediate objects do not stop a shot.
    scene = snapshot([((8, 8), (0, 1), 4), ((8, 2), (0, 1), 8)])
    out = _render(scene, active=[1])
    assert out[8, 10] == 1 and out[8, 5] == 0
    assert all(out[p] == scene["clean"][p] for p in scene["objects"][0]["arrow"][4])
    scene = snapshot([((8, 2), (0, 1), 6), ((4, 6), (1, 0), 6)])
    out = _render(scene)
    assert out[8, 8] == out[10, 6] == 1 and out[8, 6] == 0


def self_test():
    """Stream 42 seeds, keep only small feature sets and digests, write nothing."""
    signatures, shapes, structures = set(), set(), set()
    periods, directions, lane_counts, crosses, lengths = set(), set(), set(), set(), set()
    active_counts, erasure_counts, arrow_counts, palettes = set(), set(), set(), set()
    interactions = Counter()
    _regression_tests()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                r = _features(pair)
                shapes.add(r["shape"])
                periods.update(r["periods"])
                directions.update(r["directions"])
                lane_counts.add(r["lanes"])
                crosses.update(r["crosses"])
                lengths.add(r["length"])
                active_counts.add(r["active"])
                erasure_counts.add(r["erasures"])
                arrow_counts.add(r["arrows"])
                scene = r["scene"]
                palettes.add((scene["frame"], scene["padding"], scene["background"]))
                roles = {}
                normalized = bytes(roles.setdefault(int(x), len(roles)) for x in pair["input"].flat)
                structures.add(hashlib.sha256(bytes(r["shape"]) + normalized).hexdigest())
                for name in ("single_empty", "single_arrow", "collision", "active_hit", "cross_lane"):
                    interactions[name] += int(r[name])
                interactions["partial"] += int(any(r["length"] % p for p in r["periods"]))
                interactions["erased_arrows"] += r["erased_arrows"]
                before = pair["input"].copy()
                assert np.array_equal(solve(pair["input"]), pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutated its input"
        if seed in (0, 13):
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
            mapping = np.array([7, 2, 9, 0, 5, 8, 1, 4, 6, 3], dtype=np.int8)
            transformed = {s: [{k: mapping[np.rot90(p[k])].copy() for k in p}
                               for p in puzzle[s]] for s in puzzle}
            validate_generated_puzzle(transformed)
    assert len(signatures) == 42 and len(structures) >= 150
    assert len(shapes) >= 40 and len(palettes) >= 80
    assert periods == set(range(4, 9)) and directions == set(VECTORS)
    assert lane_counts == {1, 2, 3, 4} and len(crosses) >= 8 and len(lengths) >= 12
    assert len(active_counts) >= 4 and len(erasure_counts) >= 6 and len(arrow_counts) >= 15
    assert interactions["partial"] >= 30 and interactions["cross_lane"] >= 1
    assert interactions["erased_arrows"] >= 1
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # A wrong impact or a changed border cannot pass validation.
    broken = generate_puzzle(91, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupt output accepted")
