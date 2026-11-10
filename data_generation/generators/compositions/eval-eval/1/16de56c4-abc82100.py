"""Interference ribbons: registered stencil translations combined modulo two.

Only NumPy and the standard library are used. No generated labels or solving
conventions depend on seeds, previous examples, files, or module-level state.
"""

import hashlib
import numpy as np


BACKGROUND = 0
NEUTRAL = 8
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
AUDIT_SEEDS = (0,3363173,730162086,416067481,407464807,993041597,740365923,
               210679262,945437639,416321920,795081722,819115676,194075295,
               38915211,988681292,83919456,443940613,391047344,243029397,
               149485009,513618925,471075052,505887505,759093801,304577070,
               896634769,361507897,169548554,198197098,72582835,205980227,
               502035112,846515901,812833491,325265304,73317846,837095294,
               267716473,71135619,593140476,112052391,342768367)


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _components(grid):
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid == NEUTRAL)}
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        stack.append(q)
        result.append(component)
    return result


def _instructions(grid):
    """Read unique label rays around exact neutral components, input only."""
    components = _components(grid)
    if not components:
        raise ValueError("no stencil instructions")
    near = {}
    for i, component in enumerate(components):
        for r, c in component:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    near.setdefault((r + dr, c + dc), set()).add(i)
    candidates = [[] for _ in components]
    h, w = grid.shape
    for sr, sc in np.argwhere((grid != BACKGROUND) & (grid != NEUTRAL)):
        sr, sc = int(sr), int(sc)
        source = int(grid[sr, sc])
        for dr, dc in DIRECTIONS:
            tr, tc = sr + dr, sc + dc
            qr, qc = sr + 2 * dr, sc + 2 * dc
            if not (0 <= qr < h and 0 <= qc < w):
                continue
            target = int(grid[tr, tc])
            if target in (BACKGROUND, NEUTRAL, source):
                continue
            if int(grid[qr, qc]) not in (BACKGROUND, NEUTRAL):
                continue
            for i in sorted(near.get((qr, qc), ())):
                mask = components[i]
                candidates[i].append({
                    "mask": set(mask), "q": (qr, qc), "source": source,
                    "target": target, "direction": (dr, dc),
                    "labels": {(sr, sc), (tr, tc)},
                    "offsets": tuple(sorted((r - qr, c - qc) for r, c in mask))})
    if any(len(options) != 1 for options in candidates):
        raise ValueError("a stencil has missing or ambiguous labels")
    records = [options[0] for options in candidates]
    labels = [p for rec in records for p in rec["labels"]]
    if len(labels) != len(set(labels)):
        raise ValueError("shared instruction labels")
    if len({r["source"] for r in records}) != len(records):
        raise ValueError("a source selects multiple stencils")
    return records


def _decode(grid):
    records = _instructions(grid)
    controls = set().union(*(r["mask"] | r["labels"] for r in records))
    axes = set()
    for rec in records:
        pins = [tuple(map(int, p)) for p in np.argwhere(grid == rec["source"])
                if tuple(p) not in controls]
        if len(pins) != 2:
            raise ValueError("each source needs exactly two external pins")
        if pins[0][0] == pins[1][0]:
            axis, line = 1, pins[0][0]
        elif pins[0][1] == pins[1][1]:
            axis, line = 0, pins[0][1]
        else:
            raise ValueError("pins must be horizontally or vertically aligned")
        a, b = sorted(p[axis] for p in pins)
        period, length = b - a, grid.shape[axis]
        def position(x):
            return (line, x) if axis == 1 else (x, line)
        extras = [x for x in range(length)
                  if position(x) not in controls and position(x) not in pins
                  and int(grid[position(x)]) != BACKGROUND]
        aligned = [x for x in extras if (x - a) % period == 0]
        if len(aligned) > 1 or (aligned and a < aligned[0] < b):
            raise ValueError("ambiguous or interior rhythm terminal")
        terminal = aligned[0] if aligned else None
        lo, hi = (0, length - 1) if terminal is None else (min(a, terminal), max(b, terminal))
        sites = [position(x) for x in range(lo, hi + 1) if (x - a) % period == 0]
        ink = rec["target"] if terminal is None else int(grid[position(terminal)])
        if ink in (BACKGROUND, NEUTRAL, rec["source"]):
            raise ValueError("invalid ink role")
        rec.update({"pins": pins, "axis": axis, "line": line, "a": a, "b": b,
                    "period": period, "terminal": terminal, "ink": ink, "sites": sites,
                    "extras": [position(x) for x in extras if x != terminal]})
        axes.add(axis)
        controls.update(pins)
        # Do not add terminals until all records have found their pins. Source
        # hues are permitted as target labels, but not as unrelated content.
    if len(axes) != 1:
        raise ValueError("all ribbons in one input must share an active axis")
    for rec in records:
        if rec["terminal"] is not None:
            t = rec["terminal"]
            controls.add((rec["line"], t) if rec["axis"] == 1 else (t, rec["line"]))
    return records, controls


def _counts(shape, records, variant="parity"):
    counts = np.zeros((10,) + tuple(shape), dtype=np.int16)
    h, w = shape
    for rec in records:
        sites = rec["pins"] if variant == "pins_only" else rec["sites"]
        offsets = ((0, 0),) if variant == "points_only" else rec["offsets"]
        for r, c in sites:
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    counts[rec["ink"], rr, cc] += 1
    return counts


def _render(grid, records, controls, variant="parity"):
    counts = _counts(grid.shape, records, variant)
    painted = counts > 0 if variant == "opaque" else counts % 2 == 1
    if np.any(np.count_nonzero(painted, axis=0) > 1):
        raise ValueError("different surviving inks conflict")
    output = grid.copy()
    for p in controls:
        output[p] = BACKGROUND
    for color in range(1, 10):
        output[painted[color]] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode everything from this input and combine all strikes simultaneously."""
    grid = _array(input_grid)
    records, controls = _decode(grid)
    return _render(grid, records, controls)


def _turn(offsets, turns):
    result = tuple(offsets)
    for _ in range(turns % 4):
        result = tuple((-c, r) for r, c in result)
    return tuple(sorted(result))


def _geometry_key(offsets):
    variants = []
    for sign in (-1, 1):
        for turns in range(4):
            points = _turn(tuple((r * sign, c) for r, c in offsets), turns)
            r0, c0 = min(r for r, c in points), min(c for r, c in points)
            variants.append(tuple(sorted((r - r0, c - c0) for r, c in points)))
    return min(variants)


def _mask(rng, kind, witness=None, small=False):
    if witness == "overlap":
        length = int(rng.integers(5, 8))
        cells = {(0, c) for c in range(length)}
        cells.add((int(_pick(rng, (-1, 1))), int(rng.integers(0, length))))
        return tuple(sorted(cells)), (0, 1)
    if kind == "occupied":
        cells = {(0, 0), (0, 1), (1, 0)}
    elif kind == "edge_hole":
        cells = {(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0)}
    else:
        cells = {(r, c) for r in (-1, 1) for c in (-1, 0, 1)} | {(0, 1)}
    # Grow the actual shape, rather than selecting a fixed stencil catalog.
    # The empty label approach stays clear, making the ray uniquely readable.
    desired = int(rng.integers(len(cells), (10 if small else 21) + 1))
    max_c = 1 if witness == "hole" else 3 if small else 5
    for _ in range(180):
        if len(cells) >= desired:
            break
        r, c = _pick(rng, sorted(cells))
        dr, dc = _pick(rng, DIRECTIONS + ((1, 1), (-1, 1), (1, -1), (-1, -1)))
        p = (r + dr, c + dc)
        if not (-3 <= p[0] <= 3 and 0 <= p[1] <= max_c):
            continue
        if kind != "occupied" and p == (0, 0):
            continue
        cells.add(p)
    turns = 0 if witness == "hole" else int(rng.integers(4))
    offsets = _turn(tuple(cells), turns)
    direction = _turn(((0, 1),), turns)[0]
    return offsets, direction


def _schedule(rng, length, width, witness=None, terminal=None):
    if witness == "overlap":
        period = int(_pick(rng, (2, 3)))
        terminal = False
    elif witness == "hole":
        period = int(rng.integers(3, 6))
        terminal = True
    else:
        period = int(_pick(rng, tuple(range(1, min(8, (length - 1) // 2) + 1))))
        terminal = bool(rng.integers(2)) if terminal is None else terminal
    phase = int(rng.integers(period))
    full = list(range(phase, length, period))
    if len(full) < 3:
        return None
    options = []
    for i in range(len(full) - 1):
        if not terminal:
            if witness == "overlap" and not (i >= 1 and i + 2 < len(full)):
                continue
            options.append((full[i], full[i + 1], None))
        else:
            for j in range(len(full)):
                if i <= j <= i + 1:
                    continue
                a, b, t = full[i], full[i + 1], full[j]
                if witness == "hole":
                    # A complete, isolated hole stamp and a substantial unused
                    # end teach registration and stopping independently of XOR.
                    if period < width:
                        continue
                    lo, hi = min(a, t), max(b, t)
                    if max(lo, length - 1 - hi) < width + 3:
                        continue
                options.append((a, b, t))
    if not options:
        return None
    a, b, t = _pick(rng, options)
    lo, hi = (0, length - 1) if t is None else (min(a, t), max(b, t))
    return a, b, t, period, [x for x in full if lo <= x <= hi]


def _construction_output(grid, records, controls):
    """Sampled geometry ground truth, using set toggles instead of count arrays."""
    h, w = grid.shape
    strikes = set()
    for rec in records:
        for r, c in rec["sites"]:
            for dr, dc in rec["offsets"]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    key = (rec["ink"], rr, cc)
                    if key in strikes:
                        strikes.remove(key)
                    else:
                        strikes.add(key)
    output = grid.copy()
    for p in controls:
        output[p] = BACKGROUND
    ink_at = {}
    for color, r, c in strikes:
        if (r, c) in ink_at and ink_at[(r, c)] != color:
            raise ValueError("construction ink conflict")
        ink_at[(r, c)] = color
        output[r, c] = color
    return output


def _make_pair(rng, source_pool, ink_pool, witness, novel, extra=False):
    n = int(_pick(rng, (1, 1, 2, 2, 3)))
    if witness == "overlap":
        n = int(_pick(rng, (1, 1, 2)))
    if n == 3:
        small = True
    else:
        small = bool(rng.random() < .35)
    length = int(rng.integers(26 if witness == "hole" else 15, 31))
    records = []
    for i in range(n):
        teaching = witness if i == 0 else None
        kind = "occupied" if teaching == "overlap" else "interior_hole" if teaching == "hole" else _pick(rng, ("occupied", "occupied", "edge_hole", "interior_hole"))
        offsets, direction = _mask(rng, kind, teaching, small)
        if i == 0 and novel and _geometry_key(offsets) in novel:
            return None
        width = max(c for r, c in offsets) - min(c for r, c in offsets) + 1
        schedule = _schedule(rng, length, width, teaching)
        if schedule is None:
            return None
        a, b, t, period, sites = schedule
        source = source_pool[i]
        target = ink_pool[0 if witness == "overlap" and i == 0 else 1] if teaching else int(_pick(rng, ink_pool + [c for c in source_pool[:n] if c != source]))
        ink = target if t is None else int(_pick(rng, [c for c in ink_pool if c != target]))
        records.append({"offsets": offsets, "direction": direction, "source": source,
                        "target": target, "ink": ink, "a": a, "b": b, "terminal": t,
                        "period": period, "xs": sites})
    heights = [max(r for r, c in rec["offsets"]) - min(r for r, c in rec["offsets"]) + 1 for rec in records]
    minimum = max(11, sum(heights) + 6)
    if minimum > 30:
        return None
    height = int(rng.integers(minimum, 31))
    free = height - sum(heights)
    gaps = rng.multinomial(free, np.full(n + 1, 1.0 / (n + 1)))
    order = [int(i) for i in rng.permutation(n)]
    start = int(gaps[0])
    for j, i in enumerate(order):
        rec = records[i]
        row = start - min(r for r, c in rec["offsets"])
        rec.update({"line": row, "axis": 1, "sites": [(row, x) for x in rec.pop("xs")],
                    "pins": [(row, rec["a"]), (row, rec["b"])]})
        start += heights[i] + int(gaps[j + 1])
    grid = np.zeros((height, length), dtype=np.int8)
    controls = set()
    for rec in records:
        for p in rec["pins"]:
            grid[p] = rec["source"]
            controls.add(p)
        if rec["terminal"] is not None:
            p = (rec["line"], rec["terminal"])
            grid[p] = rec["ink"]
            controls.add(p)
    coverage = _counts(grid.shape, records) > 0
    footprint = {tuple(map(int, p)) for p in np.argwhere(np.any(coverage, axis=0))}
    # Off-phase content uses local ink hues too, not a permanently inert palette.
    for i, rec in enumerate(records):
        options = [(rec["line"], x) for x in range(length)
                   if (x - rec["a"]) % rec["period"] and (rec["line"], x) not in footprint
                   and grid[rec["line"], x] == BACKGROUND]
        required = witness == "hole" and i == 0
        if required and not options:
            return None
        desired = 1 if required else int(rng.integers(3)) if rng.random() < .6 else 0
        if options:
            for k in rng.permutation(len(options))[:desired]:
                grid[options[int(k)]] = rec["ink"] if required else int(_pick(rng, ink_pool))
    # Optional unrelated pixels away from ribbons; preservation is observable.
    if extra:
        options = [(r, c) for r in range(height) for c in range(length)
                   if grid[r, c] == BACKGROUND and (r, c) not in footprint]
        for k in rng.permutation(len(options))[:int(rng.integers(4))]:
            grid[options[int(k)]] = int(_pick(rng, ink_pool))
    occupied = {tuple(map(int, p)) for p in np.argwhere(grid != BACKGROUND)}
    neutral_halo = set()
    for i, rec in enumerate(records):
        dr, dc = rec["direction"]
        relative = set(rec["offsets"]) | {(0, 0), (-dr, -dc), (-2 * dr, -2 * dc)}
        r0, r1 = min(r for r, c in relative), max(r for r, c in relative)
        c0, c1 = min(c for r, c in relative), max(c for r, c in relative)
        choices = [(r, c) for r in range(-r0, height - r1) for c in range(-c0, length - c1)]
        accepted = False
        for k in rng.permutation(len(choices)):
            qr, qc = choices[int(k)]
            mask = {(qr + r, qc + c) for r, c in rec["offsets"]}
            sl, tl = (qr - 2 * dr, qc - 2 * dc), (qr - dr, qc - dc)
            cells = mask | {sl, tl, (qr, qc)}
            if cells & (occupied | footprint) or mask & neutral_halo:
                continue
            candidate = grid.copy()
            for p in mask:
                candidate[p] = NEUTRAL
            candidate[sl], candidate[tl] = rec["source"], rec["target"]
            try:
                parsed = _instructions(candidate)
            except ValueError:
                continue
            if len(parsed) != i + 1:
                continue
            rec.update({"mask": mask, "q": (qr, qc), "labels": {sl, tl}})
            grid = candidate
            occupied |= cells
            controls |= mask | {sl, tl}
            neutral_halo |= {(r + a, c + b) for r, c in mask for a in (-1, 0, 1) for b in (-1, 0, 1)}
            accepted = True
            break
        if not accepted:
            return None
    try:
        parsed, decoded_controls = _decode(grid)
        output = _construction_output(grid, records, controls)
        if decoded_controls != controls or not np.array_equal(_render(grid, parsed, decoded_controls), output):
            return None
        if witness:
            f = _features(grid, output)
            if witness == "overlap" and not (f["even_cells"] and f["both_extensions"] and f["clipped"]):
                return None
            if witness == "hole" and not (f["hole"] and f["stopped"] and f["offphase"] and f["override"]):
                return None
        # Every example, not just the demonstrations, needs both mechanisms.
        if any(np.array_equal(_render(grid, parsed, decoded_controls, alt), output)
               for alt in ("pins_only", "points_only")):
            return None
    except ValueError:
        return None
    turns = int(rng.integers(4))
    grid, output = np.rot90(grid, turns).copy(), np.rot90(output, turns).copy()
    if rng.random() < .5:
        grid, output = grid[:, ::-1].copy(), output[:, ::-1].copy()
    return {"input": grid, "output": output}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(name + " is an illegal example count")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation([1, 2, 3, 4, 5, 6, 7, 9])]
    source_pool, ink_pool = palette[:3], palette[3:]
    puzzle, seen, familiar = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = ("overlap" if i == 0 else "hole" if i == 1 else None) if split == "train" else None
            for _ in range(160):
                pair = _make_pair(rng, source_pool, ink_pool, witness,
                                  familiar if split == "test" else set(), extra=i > 1)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if split == "train":
                    familiar.update(_geometry_key(r["offsets"]) for r in _instructions(pair["input"]))
                break
            else:
                raise RuntimeError("bounded placement exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid, output):
    records, controls = _decode(grid)
    counts = _counts(grid.shape, records)
    features = {"ribbons": len(records), "axis": records[0]["axis"],
                "periods": [r["period"] for r in records],
                "sizes": [len(r["offsets"]) for r in records],
                "geometries": [_geometry_key(r["offsets"]) for r in records],
                "even_cells": int(np.count_nonzero((counts > 0) & (counts % 2 == 0))),
                "hole": False, "occupied": False, "offphase": False,
                "clipped": False, "both_extensions": False, "stopped": False,
                "override": False, "terminal_directions": set(),
                "pin_ablation": not np.array_equal(_render(grid, records, controls, "pins_only"), output),
                "point_ablation": not np.array_equal(_render(grid, records, controls, "points_only"), output),
                "opaque_ablation": not np.array_equal(_render(grid, records, controls, "opaque"), output)}
    for rec in records:
        hole = (0, 0) not in rec["offsets"]
        features["hole"] |= hole
        features["occupied"] |= not hole
        features["offphase"] |= any(int(output[p]) == int(grid[p]) for p in rec["extras"])
        for r, c in rec["sites"]:
            for dr, dc in rec["offsets"]:
                features["clipped"] |= not (0 <= r + dr < grid.shape[0] and 0 <= c + dc < grid.shape[1])
        if rec["terminal"] is None:
            coords = [p[rec["axis"]] for p in rec["sites"]]
            features["both_extensions"] |= min(coords) < rec["a"] and max(coords) > rec["b"]
        else:
            features["override"] |= rec["ink"] != rec["target"]
            features["terminal_directions"].add(-1 if rec["terminal"] < rec["a"] else 1)
            # Compare actual outputs, so a nominal unused lattice site hidden
            # by parity or clipping cannot count as teaching terminal stopping.
            expanded = [dict(r) for r in records]
            j = records.index(rec)
            xs = range(rec["a"] % rec["period"], grid.shape[rec["axis"]], rec["period"])
            expanded[j]["sites"] = [(rec["line"], x) if rec["axis"] == 1 else (x, rec["line"]) for x in xs]
            features["stopped"] |= not np.array_equal(_render(grid, expanded, controls), output)
    return features


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, training_geometry = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _array(pair["input"]), _array(pair["output"])
            assert grid.shape == output.shape and not np.array_equal(grid, output)
            assert _key(grid) not in seen, "duplicate input"
            seen.add(_key(grid))
            records, controls = _decode(grid)
            assert np.array_equal(_render(grid, records, controls), output), "incorrect toggle ribbon"
            features = _features(grid, output)
            assert features["pin_ablation"] and features["point_ablation"], "one parent is dispensable"
            if split == "train":
                training.append(features)
                training_geometry.update(features["geometries"])
            else:
                assert set(features["geometries"]) - training_geometry, "test replays training stencils"
    for name in ("hole", "occupied", "offphase", "clipped", "both_extensions", "stopped", "override", "opaque_ablation"):
        assert any(f[name] for f in training), "training lacks evidence: " + name
    assert any(f["even_cells"] > 0 for f in training)
    targets = {}
    for pair in puzzle["train"]:
        for rec in _instructions(_array(pair["input"])):
            targets.setdefault(rec["source"], set()).add(rec["target"])
    assert any(len(t) > 1 for t in targets.values()), "local labels not demonstrated"


def _manual_regressions():
    grid = np.zeros((9, 16), dtype=np.int8)
    grid[1, 0:2] = (1, 2)
    grid[1, 2:5], grid[2, 2] = NEUTRAL, NEUTRAL
    grid[6, [4, 6]] = 1
    expected = np.zeros_like(grid)
    expected[6, 0] = 2
    expected[6, 1::2] = 2
    expected[7, ::2] = 2
    yield grid, expected
    grid = np.zeros((9, 18), dtype=np.int8)
    grid[0, 2:5], grid[2, 2:5], grid[1, 4] = NEUTRAL, NEUTRAL, NEUTRAL
    grid[1, 1:3] = (1, 2)
    grid[6, [5, 7]], grid[6, 11], grid[6, 14] = 1, 3, 3
    expected = np.zeros_like(grid)
    expected[5, [4, 5, 7, 9, 11, 12]] = 3
    expected[7, [4, 5, 7, 9, 11, 12]] = 3
    expected[6, [6, 8, 10, 12, 14]] = 3
    yield grid, expected
    # Two different stencils share ink. Their footprints cross an original
    # anchor row, and cancellation is global, not separately composited.
    grid = np.zeros((9, 16), dtype=np.int8)
    for column, source in ((3, 1), (10, 3)):
        grid[1, column - 2:column] = (source, 2)
        grid[1, column:column + 2], grid[2, column] = NEUTRAL, NEUTRAL
    grid[5, [2, 4]], grid[6, [2, 4]] = 1, 3
    expected = np.zeros_like(grid)
    expected[5, :], expected[6, 1::2], expected[7, ::2] = 2, 2, 2
    yield grid, expected
    # An exterior registration hole follows exactly the same label-ray rule.
    grid = np.zeros((9, 18), dtype=np.int8)
    grid[0:3, 4] = NEUTRAL
    grid[1, 1:3] = (1, 2)
    grid[6, [4, 7]] = 1
    expected = np.zeros_like(grid)
    expected[5:8, 2::3] = 2
    yield grid, expected


def self_test():
    """Bounded 42-seed structural audit plus hand-calculated semantic cases."""
    for grid, expected in _manual_regressions():
        for turns in range(4):
            for flip in (False, True):
                x, y = np.rot90(grid, turns), np.rot90(expected, turns)
                if flip:
                    x, y = x[:, ::-1], y[:, ::-1]
                assert np.array_equal(solve(x), y), "hand-calculated regression failed"
    conflicting = list(_manual_regressions())[2][0].copy()
    conflicting[1, 9] = 4
    try:
        solve(conflicting)
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible surviving inks accepted")
    shapes, geometries, periods, counts, axes, palettes = set(), set(), set(), set(), set(), set()
    signatures, terminals, even, pairs = set(), set(), 0, 0
    for seed in AUDIT_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                f = _features(grid, output)
                shapes.add(grid.shape)
                geometries.update(f["geometries"])
                periods.update(f["periods"])
                counts.add(f["ribbons"])
                axes.add(f["axis"])
                palettes.add(tuple(np.unique(grid)))
                terminals.update(f["terminal_directions"])
                even += f["even_cells"] > 0
                pairs += 1
                digest.update(bytes(grid.shape) + grid.tobytes() + output.tobytes())
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 45 and len(geometries) >= 70
    assert {1, 2, 3, 4, 5, 6} <= periods and counts == {1, 2, 3}
    assert axes == {0, 1} and terminals == {-1, 1} and even >= 42
    assert len(palettes) >= 20
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(9182 + a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    first, second = generate_puzzle(1771, 2, 1), generate_puzzle(1771, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for split in first
               for a, b in zip(first[split], second[split]) for k in ("input", "output"))
    broken = generate_puzzle(775, 2, 1)
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "stencil_geometries": len(geometries),
            "periods": sorted(periods), "ribbon_counts": sorted(counts),
            "palettes": len(palettes), "pairs_with_even_overlap": even,
            "semantic_verification": "author checks and manual cases; no independent solver"}
